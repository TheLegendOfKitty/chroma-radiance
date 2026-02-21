#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <string>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// GPU Memory Pool: eliminates cudaMalloc/cudaFree overhead by reusing buffers.
// Uses a size-bucketed free list. Sizes are rounded up to 256-byte alignment.
// ============================================================================
struct GPUMemPool {
    std::unordered_map<size_t, std::vector<void*>> free_lists;

    void* alloc(size_t bytes) {
        bytes = (bytes + 255) & ~(size_t)255;
        auto it = free_lists.find(bytes);
        if (it != free_lists.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
        void* ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    void free(void* ptr, size_t bytes) {
        if (!ptr) return;
        bytes = (bytes + 255) & ~(size_t)255;
        free_lists[bytes].push_back(ptr);
    }

    void release_all() {
        for (auto& [sz, ptrs] : free_lists) {
            for (void* p : ptrs) cudaFree(p);
        }
        free_lists.clear();
    }

    ~GPUMemPool() { release_all(); }
};

static GPUMemPool& gpu_pool() {
    static GPUMemPool pool;
    return pool;
}

enum class DType { F32, FP16, BF16 };

static inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
    }
    return 0;
}

static inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32:  return "F32";
        case DType::FP16: return "FP16";
        case DType::BF16: return "BF16";
    }
    return "?";
}

struct Tensor {
    void* data = nullptr;
    int64_t shape[4] = {0, 0, 0, 0};
    int64_t stride[4] = {0, 0, 0, 0};  // in elements
    int ndim = 0;
    DType dtype = DType::F32;
    bool on_gpu = false;
    bool owns_data = false;

    Tensor() = default;

    // Move
    Tensor(Tensor&& o) noexcept {
        memcpy(this, &o, sizeof(Tensor));
        o.data = nullptr;
        o.owns_data = false;
    }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            free_data();
            memcpy(this, &o, sizeof(Tensor));
            o.data = nullptr;
            o.owns_data = false;
        }
        return *this;
    }

    // No copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    ~Tensor() { free_data(); }

    void free_data() {
        if (owns_data && data) {
            if (on_gpu) {
                gpu_pool().free(data, nbytes());
            } else {
                free(data);
            }
        }
        data = nullptr;
        owns_data = false;
    }

    int64_t numel() const {
        if (ndim == 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < ndim; i++) n *= shape[i];
        return n;
    }

    size_t nbytes() const {
        return numel() * dtype_size(dtype);
    }

    bool is_contiguous() const {
        if (ndim == 0) return true;
        int64_t expected = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            if (stride[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }

    void compute_strides() {
        if (ndim == 0) return;
        stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    }

    static Tensor alloc(std::initializer_list<int64_t> dims, DType dt, bool gpu) {
        Tensor t;
        t.ndim = (int)dims.size();
        assert(t.ndim <= 4);
        int i = 0;
        for (auto d : dims) t.shape[i++] = d;
        t.dtype = dt;
        t.on_gpu = gpu;
        t.compute_strides();
        size_t bytes = t.nbytes();
        if (bytes == 0) return t;
        if (gpu) {
            t.data = gpu_pool().alloc(bytes);
        } else {
            t.data = malloc(bytes);
            assert(t.data);
        }
        t.owns_data = true;
        return t;
    }

    static Tensor alloc_like(const Tensor& ref) {
        return alloc_shape(ref.ndim, ref.shape, ref.dtype, ref.on_gpu);
    }

    static Tensor alloc_shape(int ndim, const int64_t* dims, DType dt, bool gpu) {
        Tensor t;
        t.ndim = ndim;
        for (int i = 0; i < ndim; i++) t.shape[i] = dims[i];
        t.dtype = dt;
        t.on_gpu = gpu;
        t.compute_strides();
        size_t bytes = t.nbytes();
        if (bytes == 0) return t;
        if (gpu) {
            t.data = gpu_pool().alloc(bytes);
        } else {
            t.data = malloc(bytes);
            assert(t.data);
        }
        t.owns_data = true;
        return t;
    }

    static Tensor wrap_gpu(void* ptr, std::initializer_list<int64_t> dims, DType dt) {
        Tensor t;
        t.ndim = (int)dims.size();
        int i = 0;
        for (auto d : dims) t.shape[i++] = d;
        t.dtype = dt;
        t.on_gpu = true;
        t.owns_data = false;
        t.data = ptr;
        t.compute_strides();
        return t;
    }

    static Tensor wrap_cpu(void* ptr, std::initializer_list<int64_t> dims, DType dt) {
        Tensor t;
        t.ndim = (int)dims.size();
        int i = 0;
        for (auto d : dims) t.shape[i++] = d;
        t.dtype = dt;
        t.on_gpu = false;
        t.owns_data = false;
        t.data = ptr;
        t.compute_strides();
        return t;
    }

    // Copy to GPU
    Tensor to_gpu() const {
        assert(is_contiguous());
        if (on_gpu) {
            Tensor out = Tensor::alloc_shape(ndim, shape, dtype, true);
            CHECK_CUDA(cudaMemcpy(out.data, data, nbytes(), cudaMemcpyDeviceToDevice));
            return out;
        }
        Tensor out = Tensor::alloc_shape(ndim, shape, dtype, true);
        CHECK_CUDA(cudaMemcpy(out.data, data, nbytes(), cudaMemcpyHostToDevice));
        return out;
    }

    // Copy to CPU
    Tensor to_cpu() const {
        assert(is_contiguous());
        Tensor out = Tensor::alloc_shape(ndim, shape, dtype, false);
        if (on_gpu) {
            CHECK_CUDA(cudaMemcpy(out.data, data, nbytes(), cudaMemcpyDeviceToHost));
        } else {
            memcpy(out.data, data, nbytes());
        }
        return out;
    }

    // Zero fill
    void zero() {
        if (on_gpu) {
            CHECK_CUDA(cudaMemset(data, 0, nbytes()));
        } else {
            memset(data, 0, nbytes());
        }
    }

    // Reshape (must be contiguous, same number of elements)
    Tensor reshape(std::initializer_list<int64_t> new_shape) const {
        assert(is_contiguous());
        Tensor t;
        t.data = const_cast<void*>(data);
        t.ndim = (int)new_shape.size();
        t.dtype = dtype;
        t.on_gpu = on_gpu;
        t.owns_data = false;
        int i = 0;
        for (auto d : new_shape) t.shape[i++] = d;
        t.compute_strides();
        assert(t.numel() == numel());
        return t;
    }

    // View a slice along dim 0: [start:start+count, ...]
    Tensor slice(int64_t start, int64_t count) const {
        assert(ndim >= 1);
        assert(start >= 0 && start + count <= shape[0]);
        Tensor t;
        t.data = (char*)data + start * stride[0] * dtype_size(dtype);
        t.ndim = ndim;
        for (int i = 0; i < ndim; i++) {
            t.shape[i] = (i == 0) ? count : shape[i];
            t.stride[i] = stride[i];
        }
        t.dtype = dtype;
        t.on_gpu = on_gpu;
        t.owns_data = false;
        return t;
    }

    // Flat pointer offset (in bytes) for element access
    float* f32() const { assert(dtype == DType::F32); return (float*)data; }
    __half* fp16() const { assert(dtype == DType::FP16); return (__half*)data; }
    __nv_bfloat16* bf16() const { assert(dtype == DType::BF16); return (__nv_bfloat16*)data; }

    void print_info(const char* name = "") const {
        printf("Tensor %s: [", name);
        for (int i = 0; i < ndim; i++) {
            if (i > 0) printf(", ");
            printf("%ld", shape[i]);
        }
        printf("] %s %s\n", dtype_name(dtype), on_gpu ? "GPU" : "CPU");
    }
};

// Conversion kernels declared here, defined in kernels.cu
void bf16_to_f32_cuda(const __nv_bfloat16* src, float* dst, int64_t n);
void fp16_to_f32_cuda(const __half* src, float* dst, int64_t n);
void f32_to_bf16_cuda(const float* src, __nv_bfloat16* dst, int64_t n);
void f32_to_fp16_cuda(const float* src, __half* dst, int64_t n);

static inline Tensor to_f32_gpu(const Tensor& t) {
    if (t.dtype == DType::F32) return t.to_gpu();
    assert(t.on_gpu && t.is_contiguous());
    Tensor out = Tensor::alloc_shape(t.ndim, t.shape, DType::F32, true);
    int64_t n = t.numel();
    if (t.dtype == DType::BF16) {
        bf16_to_f32_cuda(t.bf16(), out.f32(), n);
    } else if (t.dtype == DType::FP16) {
        fp16_to_f32_cuda(t.fp16(), out.f32(), n);
    }
    return out;
}
