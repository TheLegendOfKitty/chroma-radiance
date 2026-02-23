#include "tensor.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cfloat>

// ============================================================================
// BF16/FP16 conversion
// ============================================================================

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* src, float* dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void fp16_to_f32_kernel(const __half* src, float* dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void fp16_to_bf16_kernel(const __half* src, __nv_bfloat16* dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx = i * 4;
    if (idx + 3 < n) {
        __half2 h0 = *reinterpret_cast<const __half2*>(src + idx);
        __half2 h1 = *reinterpret_cast<const __half2*>(src + idx + 2);
        float2 f0 = __half22float2(h0);
        float2 f1 = __half22float2(h1);
        *reinterpret_cast<__nv_bfloat162*>(dst + idx) = __floats2bfloat162_rn(f0.x, f0.y);
        *reinterpret_cast<__nv_bfloat162*>(dst + idx + 2) = __floats2bfloat162_rn(f1.x, f1.y);
    } else {
        for (int64_t j = idx; j < n && j < idx + 4; j++) {
            dst[j] = __float2bfloat16(__half2float(src[j]));
        }
    }
}

__global__ void f32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx = i * 4;
    if (idx + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(src + idx);
        __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
        __nv_bfloat162 b = __floats2bfloat162_rn(v.z, v.w);
        *reinterpret_cast<__nv_bfloat162*>(dst + idx) = a;
        *reinterpret_cast<__nv_bfloat162*>(dst + idx + 2) = b;
    } else {
        // Tail handling
        for (int64_t j = idx; j < n && j < idx + 4; j++) {
            dst[j] = __float2bfloat16(src[j]);
        }
    }
}

__global__ void f32_to_fp16_kernel(const float* src, __half* dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void bf16_to_f32_cuda(const __nv_bfloat16* src, float* dst, int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bf16_to_f32_kernel<<<blocks, threads>>>(src, dst, n);
}

void fp16_to_f32_cuda(const __half* src, float* dst, int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp16_to_f32_kernel<<<blocks, threads>>>(src, dst, n);
}

void fp16_to_bf16_cuda(const __half* src, __nv_bfloat16* dst, int64_t n) {
    int threads = 256;
    int64_t work_items = (n + 3) / 4;  // each thread handles 4 elements
    int blocks = (work_items + threads - 1) / threads;
    fp16_to_bf16_kernel<<<blocks, threads>>>(src, dst, n);
}

void f32_to_bf16_cuda(const float* src, __nv_bfloat16* dst, int64_t n) {
    int threads = 256;
    int64_t work_items = (n + 3) / 4;  // each thread handles 4 elements
    int blocks = (work_items + threads - 1) / threads;
    f32_to_bf16_kernel<<<blocks, threads>>>(src, dst, n);
}

void f32_to_fp16_cuda(const float* src, __half* dst, int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    f32_to_fp16_kernel<<<blocks, threads>>>(src, dst, n);
}

// ============================================================================
// Broadcast bias: output[i,j] = bias[j] for i in [0, M)
// ============================================================================

__global__ void broadcast_bias_kernel(const float* bias, float* output, int64_t M, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int64_t j = idx % N;
        output[idx] = bias[j];
    }
}

void broadcast_bias_cuda(const float* bias, float* output, int64_t M, int64_t N) {
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;
    broadcast_bias_kernel<<<blocks, threads>>>(bias, output, M, N);
}

// ============================================================================
// LayerNorm (no affine): output = (x - mean) / sqrt(var + eps)
// x: [rows, C], output: [rows, C]
// ============================================================================

__global__ void layer_norm_kernel(const float* x, float* out, int64_t rows, int64_t C, float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * C;
    float* or_ = out + row * C;

    extern __shared__ float sdata[];

    // Compute mean
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        sum += xr[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / C;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float d = xr[i] - mean;
        var_sum += d * d;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float var = sdata[0] / C;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        or_[i] = (xr[i] - mean) * inv_std;
    }
}

void layer_norm_cuda(const float* x, float* out, int64_t rows, int64_t C, float eps) {
    int threads = min((int64_t)256, C);
    // Round up to power of 2
    int t = 1;
    while (t < threads) t *= 2;
    threads = t;
    layer_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(x, out, rows, C, eps);
}

// ============================================================================
// RMSNorm: output = x / sqrt(mean(x^2) + eps) * scale
// x: [rows, C], scale: [C], output: [rows, C]
// ============================================================================

__global__ void rms_norm_kernel(const float* x, const float* scale, float* out, int64_t rows, int64_t C, float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * C;
    float* or_ = out + row * C;

    extern __shared__ float sdata[];

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float v = xr[i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / C + eps);
    __syncthreads();

    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        or_[i] = xr[i] * rms * scale[i];
    }
}

void rms_norm_cuda(const float* x, const float* scale, float* out, int64_t rows, int64_t C, float eps) {
    int threads = min((int64_t)256, C);
    int t = 1;
    while (t < threads) t *= 2;
    threads = t;
    rms_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(x, scale, out, rows, C, eps);
}

// ============================================================================
// GELU (tanh approximation): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// Used by Chroma transformer
// ============================================================================

__device__ __forceinline__ float gelu_val(float v) {
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(inner));
}

__global__ void gelu_kernel(const float* x, float* out, int64_t n) {
    int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(x + i);
        float4 r;
        r.x = gelu_val(v.x);
        r.y = gelu_val(v.y);
        r.z = gelu_val(v.z);
        r.w = gelu_val(v.w);
        *reinterpret_cast<float4*>(out + i) = r;
    } else {
        for (int64_t j = i; j < n && j < i + 4; j++) {
            out[j] = gelu_val(x[j]);
        }
    }
}

void gelu_cuda(const float* x, float* out, int64_t n) {
    int threads = 256;
    int64_t work = (n + 3) / 4;
    int blocks = (work + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(x, out, n);
}

// Fused GELU → BF16: applies GELU on F32 input, writes BF16 output.
// Eliminates the separate f32_to_bf16 conversion before the next linear layer.
__global__ void gelu_to_bf16_kernel(const float* x, __nv_bfloat16* out, int64_t n) {
    int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(x + i);
        __nv_bfloat162 a = __floats2bfloat162_rn(gelu_val(v.x), gelu_val(v.y));
        __nv_bfloat162 b = __floats2bfloat162_rn(gelu_val(v.z), gelu_val(v.w));
        *reinterpret_cast<__nv_bfloat162*>(out + i) = a;
        *reinterpret_cast<__nv_bfloat162*>(out + i + 2) = b;
    } else {
        for (int64_t j = i; j < n && j < i + 4; j++) {
            out[j] = __float2bfloat16(gelu_val(x[j]));
        }
    }
}

void gelu_to_bf16_cuda(const float* x, __nv_bfloat16* out, int64_t n) {
    int threads = 256;
    int64_t work = (n + 3) / 4;
    int blocks = (work + threads - 1) / threads;
    gelu_to_bf16_kernel<<<blocks, threads>>>(x, out, n);
}

// ============================================================================
// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
// Used by T5-XXL encoder (matches ggml_gelu)
// ============================================================================

__global__ void gelu_exact_kernel(const float* x, float* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
}

void gelu_exact_cuda(const float* x, float* out, int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_exact_kernel<<<blocks, threads>>>(x, out, n);
}

// ============================================================================
// SiLU: x * sigmoid(x)
// ============================================================================

__device__ __forceinline__ float silu_val(float v) {
    return v / (1.0f + expf(-v));
}

__global__ void silu_kernel(const float* x, float* out, int64_t n) {
    int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(x + i);
        float4 r = {silu_val(v.x), silu_val(v.y), silu_val(v.z), silu_val(v.w)};
        *reinterpret_cast<float4*>(out + i) = r;
    } else {
        for (int64_t j = i; j < n && j < i + 4; j++) out[j] = silu_val(x[j]);
    }
}

void silu_cuda(const float* x, float* out, int64_t n) {
    int threads = 256;
    int64_t work = (n + 3) / 4;
    int blocks = (work + threads - 1) / threads;
    silu_kernel<<<blocks, threads>>>(x, out, n);
}

// ============================================================================
// Element-wise operations
// ============================================================================

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 va = *reinterpret_cast<const float4*>(a + i);
        float4 vb = *reinterpret_cast<const float4*>(b + i);
        float4 r = {va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};
        *reinterpret_cast<float4*>(out + i) = r;
    } else {
        for (int64_t j = i; j < n && j < i + 4; j++) out[j] = a[j] + b[j];
    }
}

void add_cuda(const float* a, const float* b, float* out, int64_t n) {
    int threads = 256;
    int64_t work = (n + 3) / 4;
    int blocks = (work + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, out, n);
}

__global__ void mul_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 va = *reinterpret_cast<const float4*>(a + i);
        float4 vb = *reinterpret_cast<const float4*>(b + i);
        float4 r = {va.x * vb.x, va.y * vb.y, va.z * vb.z, va.w * vb.w};
        *reinterpret_cast<float4*>(out + i) = r;
    } else {
        for (int64_t j = i; j < n && j < i + 4; j++) out[j] = a[j] * b[j];
    }
}

void mul_cuda(const float* a, const float* b, float* out, int64_t n) {
    int threads = 256;
    int64_t work = (n + 3) / 4;
    int blocks = (work + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(a, b, out, n);
}

__global__ void add_scaled_kernel(const float* x, const float* delta, float* out, float scale, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + delta[i] * scale;
}

void add_scaled_cuda(const float* x, const float* delta, float* out, float scale, int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_scaled_kernel<<<blocks, threads>>>(x, delta, out, scale, n);
}

__global__ void scale_kernel(float* x, float scale, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

void scale_cuda(float* x, float scale, int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(x, scale, n);
}

__global__ void add_scalar_kernel(float* x, float val, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += val;
}

void add_scalar_cuda(float* x, float val, int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads>>>(x, val, n);
}

// ============================================================================
// Modulate: out = LayerNorm(x) * (1 + scale) + shift
// x: [M, C], scale: [C] (broadcast), shift: [C] (broadcast), out: [M, C]
// scale and shift are the same for all rows in M
// ============================================================================

__global__ void modulate_kernel(const float* x, const float* shift, const float* scale,
                                float* out, int64_t M, int64_t C, float eps) {
    int64_t row = blockIdx.x;
    if (row >= M) return;
    const float* xr = x + row * C;
    float* or_ = out + row * C;

    extern __shared__ float sdata[];

    // LayerNorm: compute mean
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) sum += xr[i];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / C;
    __syncthreads();

    // Variance
    float var_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float d = xr[i] - mean;
        var_sum += d * d;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / C + eps);
    __syncthreads();

    // Apply modulation
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float normed = (xr[i] - mean) * inv_std;
        or_[i] = normed * (1.0f + scale[i]) + shift[i];
    }
}

void modulate_cuda(const float* x, const float* shift, const float* scale,
                   float* out, int64_t M, int64_t C, float eps) {
    int threads = min((int64_t)256, C);
    int t = 1;
    while (t < threads) t *= 2;
    threads = t;
    modulate_kernel<<<M, threads, threads * sizeof(float)>>>(x, shift, scale, out, M, C, eps);
}

// Fused modulate → BF16: same computation but writes BF16 output.
// Eliminates separate f32_to_bf16 conversion pass for per-group INT8 linear path.
__global__ void modulate_to_bf16_kernel(const float* x, const float* shift, const float* scale,
                                         __nv_bfloat16* out, int64_t M, int64_t C, float eps) {
    int64_t row = blockIdx.x;
    if (row >= M) return;
    const float* xr = x + row * C;
    __nv_bfloat16* or_ = out + row * C;

    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) sum += xr[i];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / C;
    __syncthreads();

    float var_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float d = xr[i] - mean;
        var_sum += d * d;
    }
    sdata[threadIdx.x] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / C + eps);
    __syncthreads();

    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float normed = (xr[i] - mean) * inv_std;
        or_[i] = __float2bfloat16(normed * (1.0f + scale[i]) + shift[i]);
    }
}

void modulate_to_bf16_cuda(const float* x, const float* shift, const float* scale,
                            __nv_bfloat16* out, int64_t M, int64_t C, float eps) {
    int threads = min((int64_t)256, C);
    int t = 1;
    while (t < threads) t *= 2;
    threads = t;
    modulate_to_bf16_kernel<<<M, threads, threads * sizeof(float)>>>(x, shift, scale, out, M, C, eps);
}

// ============================================================================
// Timestep embedding: sinusoidal embedding
// input: [N] float timestamps, output: [N, dim] where dim = 2*half_dim
// freq[i] = exp(-ln(10000) * i / half_dim), val = input * time_factor * freq
// output = [cos(val_0), ..., cos(val_{half-1}), sin(val_0), ..., sin(val_{half-1})]
// ============================================================================

__global__ void timestep_embedding_kernel(const float* input, float* output,
                                          int N, int dim, float time_factor, float max_period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    int total = N * dim;
    if (idx >= total) return;

    int n = idx / dim;
    int d = idx % dim;
    float t = input[n] * time_factor;

    if (d < half_dim) {
        float freq = expf(-logf(max_period) * (float)d / (float)half_dim);
        output[idx] = cosf(t * freq);
    } else {
        int d2 = d - half_dim;
        float freq = expf(-logf(max_period) * (float)d2 / (float)half_dim);
        output[idx] = sinf(t * freq);
    }
}

void timestep_embedding_cuda(const float* input, float* output,
                             int N, int dim, float time_factor, float max_period) {
    int threads = 256;
    int total = N * dim;
    int blocks = (total + threads - 1) / threads;
    timestep_embedding_kernel<<<blocks, threads>>>(input, output, N, dim, time_factor, max_period);
}

// ============================================================================
// L2 Normalize along last axis
// x: [rows, C], out: [rows, C], out[i] = x[i] / max(||x[i]||, eps)
// ============================================================================

__global__ void l2_norm_kernel(const float* x, float* out, int64_t rows, int64_t C, float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * C;
    float* or_ = out + row * C;

    extern __shared__ float sdata[];

    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float v = xr[i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float norm = sqrtf(sdata[0]);
    if (norm < eps) norm = eps;
    float inv_norm = 1.0f / norm;
    __syncthreads();

    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        or_[i] = xr[i] * inv_norm;
    }
}

void l2_norm_cuda(const float* x, float* out, int64_t rows, int64_t C, float eps) {
    int threads = min((int64_t)256, C);
    int t = 1;
    while (t < threads) t *= 2;
    threads = t;
    l2_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(x, out, rows, C, eps);
}

// ============================================================================
// Conv2d: for img_in_patch (kernel=16x16, stride=16, no padding)
// Input: [N, C_in, H, W], Weight: [C_out, C_in, kH, kW], Bias: [C_out]
// Output: [N, C_out, H/kH, W/kW]
// Implemented via im2col + GEMM
// ============================================================================

// im2col for stride=kernel (non-overlapping patches)
__global__ void im2col_patch_kernel(const float* input, float* col,
                                     int N, int C, int H, int W,
                                     int kH, int kW, int out_h, int out_w) {
    // col: [N * out_h * out_w, C * kH * kW]
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * out_h * out_w * C * kH * kW;
    if (idx >= total) return;

    int64_t col_w = C * kH * kW;
    int64_t patch_idx = idx / col_w;
    int64_t feat_idx = idx % col_w;

    int n = patch_idx / (out_h * out_w);
    int rem = patch_idx % (out_h * out_w);
    int oh = rem / out_w;
    int ow = rem % out_w;

    int c = feat_idx / (kH * kW);
    int rem2 = feat_idx % (kH * kW);
    int kh = rem2 / kW;
    int kw_idx = rem2 % kW;

    int ih = oh * kH + kh;
    int iw = ow * kW + kw_idx;

    col[idx] = input[((n * C + c) * H + ih) * W + iw];
}

// Conv2d for nerf_final_layer_conv (kernel=3x3, stride=1, padding=1)
__global__ void conv2d_3x3_kernel(const float* input, const float* weight, const float* bias,
                                   float* output, int N, int C_in, int C_out, int H, int W) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C_out * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int co = (idx / (W * H)) % C_out;
    int n = idx / (W * H * C_out);

    float sum = bias ? bias[co] : 0.0f;

    for (int ci = 0; ci < C_in; ci++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int ih = h + kh - 1;
                int iw = w + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float inp = input[((n * C_in + ci) * H + ih) * W + iw];
                    float wt = weight[((co * C_in + ci) * 3 + kh) * 3 + kw];
                    sum += inp * wt;
                }
            }
        }
    }
    output[idx] = sum;
}

void conv2d_3x3_cuda(const float* input, const float* weight, const float* bias,
                     float* output, int N, int C_in, int C_out, int H, int W) {
    int64_t total = (int64_t)N * C_out * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_3x3_kernel<<<blocks, threads>>>(input, weight, bias, output, N, C_in, C_out, H, W);
}

// ============================================================================
// Transpose: [rows, cols] -> [cols, rows]
// ============================================================================

// Forward declaration
void batched_transpose_2d_cuda(const float* input, float* output,
                                int64_t batch, int64_t rows, int64_t cols);

void transpose_2d_cuda(const float* input, float* output, int64_t rows, int64_t cols) {
    batched_transpose_2d_cuda(input, output, 1, rows, cols);
}

// Batched transpose: batch matrices of [rows, cols] -> [cols, rows]
// Uses shared memory tiles for coalesced reads AND writes.
#define TILE_DIM 32
#define BLOCK_ROWS 8
__global__ void batched_transpose_2d_kernel(const float* input, float* output,
                                             int64_t batch, int64_t rows, int64_t cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 avoids bank conflicts

    int64_t b = blockIdx.z;
    if (b >= batch) return;
    const float* in = input + b * rows * cols;
    float* out = output + b * rows * cols;

    // Block position in the matrix
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    // Load tile with coalesced reads: reading rows [by..by+TILE_DIM) cols [bx..bx+TILE_DIM)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int r = by + threadIdx.y + j;
        int c = bx + threadIdx.x;
        if (r < rows && c < cols)
            tile[threadIdx.y + j][threadIdx.x] = in[r * cols + c];
    }
    __syncthreads();

    // Write transposed tile with coalesced writes
    // Output position: rows [bx..bx+TILE_DIM) cols [by..by+TILE_DIM)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int r = bx + threadIdx.y + j;
        int c = by + threadIdx.x;
        if (r < cols && c < rows)
            out[r * rows + c] = tile[threadIdx.x][threadIdx.y + j];
    }
}

void batched_transpose_2d_cuda(const float* input, float* output,
                                int64_t batch, int64_t rows, int64_t cols) {
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks((cols + TILE_DIM - 1) / TILE_DIM,
                (rows + TILE_DIM - 1) / TILE_DIM,
                batch);
    batched_transpose_2d_kernel<<<blocks, threads>>>(input, output, batch, rows, cols);
}
#undef TILE_DIM
#undef BLOCK_ROWS

// ============================================================================
// Concatenate two tensors along a specified dimension
// For dim=0 (last dim in our row-major convention, first in shape):
//   a: [M, Da], b: [M, Db] -> out: [M, Da+Db]
// ============================================================================

__global__ void concat_last_dim_kernel(const float* a, const float* b, float* out,
                                        int64_t M, int64_t Da, int64_t Db) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t D_out = Da + Db;
    if (idx >= M * D_out) return;
    int64_t row = idx / D_out;
    int64_t col = idx % D_out;
    if (col < Da) {
        out[idx] = a[row * Da + col];
    } else {
        out[idx] = b[row * Db + (col - Da)];
    }
}

void concat_last_dim_cuda(const float* a, const float* b, float* out,
                          int64_t M, int64_t Da, int64_t Db) {
    int64_t total = M * (Da + Db);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat_last_dim_kernel<<<blocks, threads>>>(a, b, out, M, Da, Db);
}

// Fused concat → BF16: concatenates two F32 tensors and converts to BF16 in one pass.
// Eliminates the separate concat + f32_to_bf16 for the single-block linear2 path.
__global__ void concat_last_dim_to_bf16_kernel(const float* a, const float* b,
                                                __nv_bfloat16* out,
                                                int64_t M, int64_t Da, int64_t Db) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t D_out = Da + Db;
    if (idx >= M * D_out) return;
    int64_t row = idx / D_out;
    int64_t col = idx % D_out;
    if (col < Da) {
        out[idx] = __float2bfloat16(a[row * Da + col]);
    } else {
        out[idx] = __float2bfloat16(b[row * Db + (col - Da)]);
    }
}

void concat_last_dim_to_bf16_cuda(const float* a, const float* b, __nv_bfloat16* out,
                                   int64_t M, int64_t Da, int64_t Db) {
    int64_t total = M * (Da + Db);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat_last_dim_to_bf16_kernel<<<blocks, threads>>>(a, b, out, M, Da, Db);
}

// Concat along first dim (seq dim): a: [Sa, D], b: [Sb, D] -> [Sa+Sb, D]
__global__ void concat_first_dim_kernel(const float* a, const float* b, float* out,
                                         int64_t Sa, int64_t Sb, int64_t D) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (Sa + Sb) * D;
    if (idx >= total) return;
    int64_t row = idx / D;
    int64_t col = idx % D;
    if (row < Sa) {
        out[idx] = a[row * D + col];
    } else {
        out[idx] = b[(row - Sa) * D + col];
    }
}

void concat_first_dim_cuda(const float* a, const float* b, float* out,
                           int64_t Sa, int64_t Sb, int64_t D) {
    int64_t total = (Sa + Sb) * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    concat_first_dim_kernel<<<blocks, threads>>>(a, b, out, Sa, Sb, D);
}

// ============================================================================
// Gated residual: out = x + gate * y
// x, y, gate: all [M, C], but gate is broadcast: [C] applied to all M rows
// ============================================================================

__global__ void gated_residual_kernel(const float* x, const float* y, const float* gate,
                                       float* out, int64_t M, int64_t C) {
    int64_t idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int64_t total = M * C;
    if (idx + 3 < total) {
        float4 vx = *reinterpret_cast<const float4*>(x + idx);
        float4 vy = *reinterpret_cast<const float4*>(y + idx);
        int64_t c0 = idx % C, c1 = (idx+1) % C, c2 = (idx+2) % C, c3 = (idx+3) % C;
        float4 r = {vx.x + gate[c0] * vy.x, vx.y + gate[c1] * vy.y,
                    vx.z + gate[c2] * vy.z, vx.w + gate[c3] * vy.w};
        *reinterpret_cast<float4*>(out + idx) = r;
    } else {
        for (int64_t j = idx; j < total && j < idx + 4; j++) {
            int64_t c = j % C;
            out[j] = x[j] + gate[c] * y[j];
        }
    }
}

void gated_residual_cuda(const float* x, const float* y, const float* gate,
                         float* out, int64_t M, int64_t C) {
    int64_t total = M * C;
    int threads = 256;
    int64_t work = (total + 3) / 4;
    int blocks = (work + threads - 1) / threads;
    gated_residual_kernel<<<blocks, threads>>>(x, y, gate, out, M, C);
}

// ============================================================================
// Patchify: [N, C, H, W] -> [N, (H/p)*(W/p), C*p*p]
// For pixel-space Chroma: C=3, p=16
// ============================================================================

__global__ void patchify_kernel(const float* input, float* output,
                                 int N, int C, int H, int W, int p) {
    int h_patches = H / p;
    int w_patches = W / p;
    int num_patches = h_patches * w_patches;
    int patch_feat = C * p * p;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * num_patches * patch_feat;
    if (idx >= total) return;

    int n = idx / (num_patches * patch_feat);
    int rem = idx % (num_patches * patch_feat);
    int patch = rem / patch_feat;
    int feat = rem % patch_feat;

    int ph = patch / w_patches;
    int pw = patch % w_patches;

    // feat = c * p * p + py * p + px
    int c = feat / (p * p);
    int rem2 = feat % (p * p);
    int py = rem2 / p;
    int px = rem2 % p;

    int ih = ph * p + py;
    int iw = pw * p + px;

    output[idx] = input[((n * C + c) * H + ih) * W + iw];
}

void patchify_cuda(const float* input, float* output, int N, int C, int H, int W, int p) {
    int h_patches = H / p;
    int w_patches = W / p;
    int num_patches = h_patches * w_patches;
    int patch_feat = C * p * p;
    int64_t total = (int64_t)N * num_patches * patch_feat;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    patchify_kernel<<<blocks, threads>>>(input, output, N, C, H, W, p);
}

// ============================================================================
// Unpatchify: [N, (H/p)*(W/p), C*p*p] -> [N, C, H, W]
// ============================================================================

__global__ void unpatchify_kernel(const float* input, float* output,
                                   int N, int C, int H, int W, int p,
                                   int h_patches, int w_patches) {
    int num_patches = h_patches * w_patches;
    int patch_feat = C * p * p;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H * W;
    if (idx >= total) return;

    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    int rem2 = rem % (H * W);
    int ih = rem2 / W;
    int iw = rem2 % W;

    int ph = ih / p;
    int pw = iw / p;
    int py = ih % p;
    int px = iw % p;

    int patch = ph * w_patches + pw;
    int feat = c * p * p + py * p + px;

    output[idx] = input[(n * num_patches + patch) * patch_feat + feat];
}

void unpatchify_cuda(const float* input, float* output,
                     int N, int C, int H, int W, int p,
                     int h_patches, int w_patches) {
    int64_t total = (int64_t)N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    unpatchify_kernel<<<blocks, threads>>>(input, output, N, C, H, W, p, h_patches, w_patches);
}

// ============================================================================
// Embedding lookup: output[i] = table[ids[i]]
// table: [vocab, dim], ids: [N], output: [N, dim]
// ============================================================================

__global__ void embedding_lookup_f16_kernel(const __half* table, const int* ids, float* output,
                                             int N, int dim) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)N * dim) return;
    int n = idx / dim;
    int d = idx % dim;
    int id = ids[n];
    output[idx] = __half2float(table[(int64_t)id * dim + d]);
}

void embedding_lookup_f16_cuda(const __half* table, const int* ids, float* output,
                               int N, int dim) {
    int64_t total = (int64_t)N * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    embedding_lookup_f16_kernel<<<blocks, threads>>>(table, ids, output, N, dim);
}

// ============================================================================
// Softmax along last dimension
// x: [rows, C], out: [rows, C]
// ============================================================================

__global__ void softmax_kernel(const float* x, float* out, int64_t rows, int64_t C) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * C;
    float* or_ = out + row * C;

    extern __shared__ float sdata[];
    float* smax = sdata;
    float* ssum = sdata + blockDim.x;

    // Online softmax: find max AND compute exp sum in a single pass
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float v = xr[i];
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }
    }
    smax[threadIdx.x] = local_max;
    ssum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float m1 = smax[threadIdx.x];
            float m2 = smax[threadIdx.x + s];
            float new_max = fmaxf(m1, m2);
            ssum[threadIdx.x] = ssum[threadIdx.x] * expf(m1 - new_max)
                              + ssum[threadIdx.x + s] * expf(m2 - new_max);
            smax[threadIdx.x] = new_max;
        }
        __syncthreads();
    }
    float max_val = smax[0];
    float inv_sum = 1.0f / ssum[0];
    __syncthreads();

    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        or_[i] = expf(xr[i] - max_val) * inv_sum;
    }
}

void softmax_cuda(const float* x, float* out, int64_t rows, int64_t C) {
    int threads = min((int64_t)256, C);
    int t = 1;
    while (t < threads) t *= 2;
    threads = t;
    softmax_kernel<<<rows, threads, 2 * threads * sizeof(float)>>>(x, out, rows, C);
}

// ============================================================================
// Fused softmax + attention mask: scores[row, col] += mask[col], then softmax
// mask can be nullptr (no mask applied)
// ============================================================================

__global__ void softmax_with_mask_kernel(const float* x, const float* mask, float* out,
                                          int64_t rows, int64_t C) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * C;
    float* or_ = out + row * C;

    extern __shared__ float sdata[];
    float* smax = sdata;
    float* ssum = sdata + blockDim.x;

    // Online softmax: find max AND compute exp sum in a single pass
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float v = mask ? xr[i] + mask[i] : xr[i];
        if (v > local_max) {
            local_sum = local_sum * expf(local_max - v) + 1.0f;
            local_max = v;
        } else {
            local_sum += expf(v - local_max);
        }
    }
    smax[threadIdx.x] = local_max;
    ssum[threadIdx.x] = local_sum;
    __syncthreads();

    // Parallel reduction combining max and sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float m1 = smax[threadIdx.x];
            float m2 = smax[threadIdx.x + s];
            float new_max = fmaxf(m1, m2);
            ssum[threadIdx.x] = ssum[threadIdx.x] * expf(m1 - new_max)
                              + ssum[threadIdx.x + s] * expf(m2 - new_max);
            smax[threadIdx.x] = new_max;
        }
        __syncthreads();
    }
    float max_val = smax[0];
    float inv_sum = 1.0f / ssum[0];
    __syncthreads();

    // Single normalize pass (writes output)
    for (int64_t i = threadIdx.x; i < C; i += blockDim.x) {
        float v = mask ? xr[i] + mask[i] : xr[i];
        or_[i] = expf(v - max_val) * inv_sum;
    }
}

void softmax_with_mask_cuda(const float* x, const float* mask, float* out,
                             int64_t rows, int64_t C) {
    int threads = min((int64_t)256, C);
    int t = 1;
    while (t < threads) t *= 2;
    threads = t;
    softmax_with_mask_kernel<<<rows, threads, 2 * threads * sizeof(float)>>>(x, mask, out, rows, C);
}

// ============================================================================
// Copy kernel (for non-contiguous copies, permutations, etc.)
// ============================================================================

__global__ void copy_kernel(const float* src, float* dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

void copy_cuda(const float* src, float* dst, int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    copy_kernel<<<blocks, threads>>>(src, dst, n);
}

// ============================================================================
// Permute 3D: [A, B, C] -> specified permutation
// perm specifies which source axis maps to each output axis
// ============================================================================

// Permute [A, B, C] with perm[3]
__global__ void permute_3d_kernel(const float* input, float* output,
                                   int64_t d0, int64_t d1, int64_t d2,
                                   int64_t out_d0, int64_t out_d1, int64_t out_d2,
                                   int p0, int p1, int p2) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = d0 * d1 * d2;
    if (idx >= total) return;

    int64_t i0 = idx / (d1 * d2);
    int64_t rem = idx % (d1 * d2);
    int64_t i1 = rem / d2;
    int64_t i2 = rem % d2;

    int64_t src_idx[3] = {i0, i1, i2};
    int64_t out_idx[3];
    out_idx[0] = src_idx[p0];
    out_idx[1] = src_idx[p1];
    out_idx[2] = src_idx[p2];

    // Actually, perm means: out[i][j][k] = input[...] where output axis 0 comes from input axis p0
    // So: output[src_idx[p0], src_idx[p1], src_idx[p2]] = input[i0, i1, i2]
    int64_t out_flat = out_idx[0] * (out_d1 * out_d2) + out_idx[1] * out_d2 + out_idx[2];
    output[out_flat] = input[idx];
}

void permute_3d_cuda(const float* input, float* output,
                     int64_t d0, int64_t d1, int64_t d2,
                     int p0, int p1, int p2) {
    int64_t dims[3] = {d0, d1, d2};
    int64_t out_d0 = dims[p0], out_d1 = dims[p1], out_d2 = dims[p2];
    int64_t total = d0 * d1 * d2;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    permute_3d_kernel<<<blocks, threads>>>(input, output, d0, d1, d2, out_d0, out_d1, out_d2, p0, p1, p2);
}

// ============================================================================
// Permute 4D: [A, B, C, D] -> specified permutation
// ============================================================================

__global__ void permute_4d_kernel(const float* input, float* output,
                                   int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                                   int64_t od0, int64_t od1, int64_t od2, int64_t od3,
                                   int p0, int p1, int p2, int p3) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = d0 * d1 * d2 * d3;
    if (idx >= total) return;

    int64_t i0 = idx / (d1 * d2 * d3);
    int64_t rem = idx % (d1 * d2 * d3);
    int64_t i1 = rem / (d2 * d3);
    rem = rem % (d2 * d3);
    int64_t i2 = rem / d3;
    int64_t i3 = rem % d3;

    int64_t src[4] = {i0, i1, i2, i3};
    int64_t out_flat = src[p0] * (od1 * od2 * od3) + src[p1] * (od2 * od3) + src[p2] * od3 + src[p3];
    output[out_flat] = input[idx];
}

void permute_4d_cuda(const float* input, float* output,
                     int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                     int p0, int p1, int p2, int p3) {
    int64_t dims[4] = {d0, d1, d2, d3};
    int64_t od0 = dims[p0], od1 = dims[p1], od2 = dims[p2], od3 = dims[p3];
    int64_t total = d0 * d1 * d2 * d3;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    permute_4d_kernel<<<blocks, threads>>>(input, output, d0, d1, d2, d3, od0, od1, od2, od3, p0, p1, p2, p3);
}

// ============================================================================
// Mul-add with broadcast: out[i*C + j] = a[i*C + j] * b[j]  (b broadcast over rows)
// ============================================================================
__global__ void mul_broadcast_kernel(const float* a, const float* b, float* out, int64_t M, int64_t C) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * C) return;
    int64_t j = idx % C;
    out[idx] = a[idx] * b[j];
}

void mul_broadcast_cuda(const float* a, const float* b, float* out, int64_t M, int64_t C) {
    int64_t total = M * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mul_broadcast_kernel<<<blocks, threads>>>(a, b, out, M, C);
}

// ============================================================================
// Random normal (Philox 4x32-10 + Box-Muller, matching PyTorch/cuRAND exactly)
// ============================================================================

// Philox 4x32-10 RNG matching cuRAND's curandStatePhilox4_32_10_t
// Verified: produces identical raw uint32 values as cuRAND
struct Philox4x32 {
    unsigned int counter[4];
    unsigned int key[2];

    __device__ Philox4x32(unsigned long long seed, unsigned long long subsequence, unsigned long long offset) {
        key[0] = (unsigned int)seed;
        key[1] = (unsigned int)(seed >> 32);
        counter[0] = (unsigned int)offset;
        counter[1] = (unsigned int)(offset >> 32);
        counter[2] = (unsigned int)subsequence;
        counter[3] = (unsigned int)(subsequence >> 32);
    }

    __device__ static void mulhilo32(unsigned int a, unsigned int b, unsigned int* hi, unsigned int* lo) {
        unsigned long long product = (unsigned long long)a * (unsigned long long)b;
        *lo = (unsigned int)product;
        *hi = (unsigned int)(product >> 32);
    }

    __device__ void single_round(unsigned int* ctr, const unsigned int* k) {
        unsigned int hi0, lo0, hi1, lo1;
        mulhilo32(0xD2511F53u, ctr[0], &hi0, &lo0);
        mulhilo32(0xCD9E8D57u, ctr[2], &hi1, &lo1);
        ctr[0] = hi1 ^ ctr[1] ^ k[0];
        ctr[1] = lo1;
        ctr[2] = hi0 ^ ctr[3] ^ k[1];
        ctr[3] = lo0;
    }

    __device__ void operator()() {
        unsigned int k0 = key[0], k1 = key[1];
        for (int i = 0; i < 10; i++) {
            single_round(counter, key);
            key[0] = k0 + (i + 1) * 0x9E3779B9u;
            key[1] = k1 + (i + 1) * 0xBB67AE85u;
        }
    }
};

// cuRAND-matching Box-Muller transform
// Takes 2 raw uint32 from Philox, returns 2 normal floats
// Matches cuRAND's _curand_box_muller: sin first, cos second
__device__ inline void curand_box_muller(unsigned int x, unsigned int y, float* n0, float* n1) {
    // cuRAND's uniform conversion: x * 2^-32 + 2^-33
    const float CURAND_2POW32_INV = 2.3283064365386963e-10f;
    const float CURAND_2POW32_INV_2PI = 1.4629180792671596e-09f; // 2^-32 * 2*PI

    float u = (float)x * CURAND_2POW32_INV + (CURAND_2POW32_INV * 0.5f);
    float v = (float)y * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI * 0.5f);

    float s = sqrtf(-2.0f * logf(u));
    __sincosf(v, n0, n1);
    *n0 *= s;
    *n1 *= s;
}

// PyTorch-matching randn kernel
// Uses Philox 4x32-10 with cuRAND-matching Box-Muller transform
// Interleaved storage: output[tid], output[tid+stride], output[tid+2*stride], output[tid+3*stride]
// PyTorch uses occupancy-based grid sizing: grid = SM_count * max_blocks_per_SM
__global__ void rand_normal_philox_kernel(float* output, int64_t n, unsigned long long seed) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    // Match PyTorch/cuRAND: subsequence=tid, offset=0
    Philox4x32 rng(seed, (unsigned long long)tid, 0ULL);
    rng();

    // Box-Muller matching cuRAND's curand_normal4
    float n0, n1, n2, n3;
    curand_box_muller(rng.counter[0], rng.counter[1], &n0, &n1);
    curand_box_muller(rng.counter[2], rng.counter[3], &n2, &n3);

    // Interleaved storage matching PyTorch's distribution_nullary_kernel
    if (tid < n) output[tid] = n0;
    if (tid + stride < n) output[tid + stride] = n1;
    if (tid + 2 * stride < n) output[tid + 2 * stride] = n2;
    if (tid + 3 * stride < n) output[tid + 3 * stride] = n3;
}

void rand_normal_cuda(float* output, int64_t n, unsigned long long seed) {
    // Match PyTorch's occupancy-based grid sizing:
    // grid = SM_count * max_blocks_per_SM (not simple ceil(n/bs))
    // This ensures the interleaved storage pattern matches PyTorch exactly
    const int block_size = 128;  // PyTorch uses 128 for distribution kernels

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;
    int grid = prop.multiProcessorCount * max_blocks_per_sm;

    // Don't launch more blocks than needed (1 value per thread)
    int min_blocks = (int)((n + block_size - 1) / block_size);
    if (grid > min_blocks) grid = min_blocks;

    // Ensure full coverage: each thread writes 4 values, so we need
    // grid * block_size * 4 >= n. Without this, large outputs (e.g. 512x512x3)
    // have uninitialized regions.
    int min_coverage = (int)((n + (int64_t)block_size * 4 - 1) / ((int64_t)block_size * 4));
    if (grid < min_coverage) grid = min_coverage;

    rand_normal_philox_kernel<<<grid, block_size>>>(output, n, seed);
}

// sd.cpp-matching randn kernel
// Uses same Philox 4x32-10 but with sequential storage:
// For each element i: counter=(0, 0, i, 0), take only sin from Box-Muller(g[0], g[1])
// This matches sd.cpp's PhiloxRNG::randn() / rng_philox.hpp
__global__ void rand_normal_sdcpp_kernel(float* output, int64_t n, unsigned long long seed) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // counter = (0, 0, idx, 0), key from seed — same as subsequence=idx, offset=0
    Philox4x32 rng(seed, (unsigned long long)idx, 0ULL);
    rng();

    // Box-Muller: only take sin result (first output), matching sd.cpp
    const float CURAND_2POW32_INV = 2.3283064365386963e-10f;
    const float CURAND_2POW32_INV_2PI = 1.4629180792671596e-09f;

    float u = (float)rng.counter[0] * CURAND_2POW32_INV + (CURAND_2POW32_INV * 0.5f);
    float v = (float)rng.counter[1] * CURAND_2POW32_INV_2PI + (CURAND_2POW32_INV_2PI * 0.5f);

    float s = sqrtf(-2.0f * logf(u));
    output[idx] = s * sinf(v);
}

void rand_normal_sdcpp_cuda(float* output, int64_t n, unsigned long long seed) {
    int block_size = 256;
    int grid = (int)((n + block_size - 1) / block_size);
    rand_normal_sdcpp_kernel<<<grid, block_size>>>(output, n, seed);
}

// ============================================================================
// Slice columns: extract [L, slice_cols] from [L, total_cols] starting at col_offset
// src: [L, total_cols] row-major, dst: [L, slice_cols] contiguous row-major
// ============================================================================

__global__ void slice_columns_kernel(const float* src, float* dst,
                                      int64_t L, int64_t total_cols,
                                      int64_t col_offset, int64_t slice_cols) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = L * slice_cols;
    if (idx >= total) return;
    int64_t row = idx / slice_cols;
    int64_t col = idx % slice_cols;
    dst[idx] = src[row * total_cols + col_offset + col];
}

void slice_columns_cuda(const float* src, float* dst,
                         int64_t L, int64_t total_cols,
                         int64_t col_offset, int64_t slice_cols) {
    int64_t total = L * slice_cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    slice_columns_kernel<<<blocks, threads>>>(src, dst, L, total_cols, col_offset, slice_cols);
}

// ============================================================================
// Apply attention mask: scores[h, q, k] += mask[k]
// scores: [n_head * L_q, L_k] flat, mask: [L_k]
// ============================================================================

__global__ void apply_attention_mask_kernel(float* scores, const float* mask,
                                             int64_t L_k, int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int64_t k = idx % L_k;
    scores[idx] += mask[k];
}

void apply_attention_mask_cuda(float* scores, const float* mask,
                                int64_t n_head, int64_t L_q, int64_t L_k) {
    int64_t total = n_head * L_q * L_k;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    apply_attention_mask_kernel<<<blocks, threads>>>(scores, mask, L_k, total);
}

// ============================================================================
// Post-GEMM bias add: data[i*N + j] += bias[j]
// data: [M, N], bias: [N]
// ============================================================================

__global__ void add_bias_kernel(float* data, const float* bias, int64_t M, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int64_t j = idx % N;
        data[idx] += bias[j];
    }
}

void add_bias_cuda(float* data, const float* bias, int64_t M, int64_t N) {
    int threads = 256;
    int blocks = (int)((M * N + threads - 1) / threads);
    add_bias_kernel<<<blocks, threads>>>(data, bias, M, N);
}

// ============================================================================
// Broadcast rows: copy src[seq, feat] into each of batch slices of dst[batch, seq, feat]
// ============================================================================

__global__ void broadcast_rows_kernel(const float* src, float* dst,
                                       int64_t batch, int64_t seq_feat) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * seq_feat;
    if (idx >= total) return;
    int64_t sf = idx % seq_feat;
    dst[idx] = src[sf];
}

void broadcast_rows_cuda(const float* src, float* dst,
                          int64_t batch, int64_t seq, int64_t feat) {
    int64_t seq_feat = seq * feat;
    int64_t total = batch * seq_feat;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    broadcast_rows_kernel<<<blocks, threads>>>(src, dst, batch, seq_feat);
}

// ============================================================================
// INT8 Dequantization to BF16 with per-group scales
// w: [N, K] INT8, scales: [N, num_groups] F32, out: [N, K] BF16
// Each group of `group_size` elements along K shares one scale.
// ============================================================================

__device__ __forceinline__ float load_quant_scale_device(const void* scales, int scales_dtype, int64_t idx) {
    if (scales_dtype == (int)DType::F32) {
        return ((const float*)scales)[idx];
    }
    if (scales_dtype == (int)DType::FP16) {
        return __half2float(((const __half*)scales)[idx]);
    }
    // Default/fallback to BF16 for compact scale storage.
    return __bfloat162float(((const __nv_bfloat16*)scales)[idx]);
}

__global__ void dequant_int8_to_bf16_kernel(const int8_t* __restrict__ w,
                                             const void* __restrict__ scales,
                                             int scales_dtype,
                                             __nv_bfloat16* __restrict__ out,
                                             int64_t N, int64_t K,
                                             int num_groups, int group_size) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int64_t n = idx / K;
    int64_t k = idx % K;
    int g = (int)(k / group_size);
    float s = load_quant_scale_device(scales, scales_dtype, n * num_groups + g);
    out[idx] = __float2bfloat16((float)w[idx] * s);
}

void dequant_int8_to_bf16_cuda(const int8_t* src, const void* scales, DType scales_dtype,
                                __nv_bfloat16* dst, int64_t N, int64_t K,
                                int group_size) {
    int num_groups = (int)((K + group_size - 1) / group_size);
    int64_t total = N * K;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    dequant_int8_to_bf16_kernel<<<blocks, threads>>>(src, scales, (int)scales_dtype,
                                                     dst, N, K, num_groups, group_size);
}

// ============================================================================
// Vectorized INT8→BF16 dequant for pre-dequant + cuBLAS path
// Each thread processes 16 INT8 elements via uint4 load, writes 16 BF16.
// Handles per-group scales and optional zero_points.
// ============================================================================

__global__ void dequant_int8_to_bf16_vec_kernel(
    const int8_t* __restrict__ w,
    const void* __restrict__ scales,
    int scales_dtype,
    const int8_t* __restrict__ zero_points,
    __nv_bfloat16* __restrict__ out,
    int64_t N, int64_t K,
    int num_groups, int group_size)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t elem_start = tid * 16;
    int64_t total = N * K;

    if (elem_start >= total) return;

    // Scalar tail: last partial chunk OR chunk that would cross a row boundary
    if (elem_start + 16 > total || (elem_start % K) + 16 > K) {
        for (int i = 0; i < 16 && elem_start + i < total; i++) {
            int64_t idx = elem_start + i;
            int64_t n = idx / K;
            int64_t k = idx % K;
            int g = (int)(k / group_size);
            int64_t gi = n * num_groups + g;
            float s = load_quant_scale_device(scales, scales_dtype, gi);
            float z = zero_points ? (float)zero_points[gi] : 0.0f;
            out[idx] = __float2bfloat16(((float)w[idx] - z) * s);
        }
        return;
    }

    int64_t n = elem_start / K;
    int64_t k = elem_start % K;

    // Vectorized read: 16 INT8 values (uint4 = 16 bytes, coalesced)
    uint4 raw = *reinterpret_cast<const uint4*>(w + elem_start);
    const int8_t* q = reinterpret_cast<const int8_t*>(&raw);

    // Load scale and zero_point for the starting group
    int g_start = (int)(k / group_size);
    int g_end = (int)((k + 15) / group_size);
    int64_t gi = n * num_groups + g_start;
    float s = load_quant_scale_device(scales, scales_dtype, gi);
    float z = zero_points ? (float)zero_points[gi] : 0.0f;

    if (g_start == g_end) {
        // Fast path: all 16 elements in the same group (common for group_size >= 16)
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            *reinterpret_cast<__nv_bfloat162*>(out + elem_start + i) =
                __floats2bfloat162_rn(((float)q[i] - z) * s, ((float)q[i+1] - z) * s);
        }
    } else {
        // Slow path: group boundary within this 16-element chunk
        int64_t scale_base = n * num_groups;
        #pragma unroll
        for (int i = 0; i < 16; i += 2) {
            int gi0 = (int)((k + i) / group_size);
            int gi1 = (int)((k + i + 1) / group_size);
            float s0 = load_quant_scale_device(scales, scales_dtype, scale_base + gi0);
            float s1 = load_quant_scale_device(scales, scales_dtype, scale_base + gi1);
            float z0 = zero_points ? (float)zero_points[scale_base + gi0] : 0.0f;
            float z1 = zero_points ? (float)zero_points[scale_base + gi1] : 0.0f;
            *reinterpret_cast<__nv_bfloat162*>(out + elem_start + i) =
                __floats2bfloat162_rn(((float)q[i] - z0) * s0, ((float)q[i+1] - z1) * s1);
        }
    }
}

void dequant_int8_to_bf16_vec_cuda(const int8_t* src, const void* scales, DType scales_dtype,
                                    const int8_t* zero_points,
                                    __nv_bfloat16* dst, int64_t N, int64_t K,
                                    int group_size) {
    int num_groups = (int)((K + group_size - 1) / group_size);
    int64_t total_elems = N * K;
    int64_t work_items = (total_elems + 15) / 16;
    int threads = 256;
    int blocks = (int)((work_items + threads - 1) / threads);
    dequant_int8_to_bf16_vec_kernel<<<blocks, threads>>>(
        src, scales, (int)scales_dtype, zero_points, dst, N, K, num_groups, group_size);
}

// ============================================================================
// Dynamic INT8 activation quantization for INT8 GEMM path
// Per-row symmetric: x_int8 = round(x * 127/absmax), x_scale = absmax/127
// Also computes row sums of x_int8 for asymmetric weight dequant correction:
//   Y[m,n] = x_scale[m] * w_scale[n] * (GEMM[m,n] - zp[n] * x_rowsum[m])
// ============================================================================

// Template on NEED_ROWSUM and InputT (float or __nv_bfloat16):
// - NEED_ROWSUM=false skips integer rowsum accumulation + reduction (symmetric quantization)
// - InputT=float avoids the f32→bf16 conversion kernel when activations are already F32
template<bool NEED_ROWSUM, typename InputT>
__global__ void quantize_activations_kernel(
    const InputT* __restrict__ X,          // [M, K]
    int8_t* __restrict__ X_int8,           // [M, K]
    float* __restrict__ x_scale,           // [M]
    float* __restrict__ x_rowsum,          // [M] or nullptr (only written when NEED_ROWSUM)
    const float* __restrict__ smooth,      // [K] or nullptr
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const InputT* x_row = X + (int64_t)row * K;
    int8_t* out_row = X_int8 + (int64_t)row * K;

    extern __shared__ char qsmem[];
    float* fdata = reinterpret_cast<float*>(qsmem);

    // Pass 1: find per-row absmax with vectorized loads (4 elements per iteration)
    // K%4==0 is guaranteed by the per-channel INT8 entry point.
    float local_max = 0.0f;
    const int k4_limit = K & ~3;

    if constexpr (__is_same(InputT, float)) {
        for (int k = threadIdx.x * 4; k < k4_limit; k += blockDim.x * 4) {
            float4 v = *reinterpret_cast<const float4*>(x_row + k);
            if (smooth) {
                v.x /= smooth[k]; v.y /= smooth[k+1]; v.z /= smooth[k+2]; v.w /= smooth[k+3];
            }
            local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                                                fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    } else {
        for (int k = threadIdx.x * 4; k < k4_limit; k += blockDim.x * 4) {
            __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k);
            __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k + 2);
            float f0 = __bfloat162float(v01.x), f1 = __bfloat162float(v01.y);
            float f2 = __bfloat162float(v23.x), f3 = __bfloat162float(v23.y);
            if (smooth) {
                f0 /= smooth[k]; f1 /= smooth[k+1]; f2 /= smooth[k+2]; f3 /= smooth[k+3];
            }
            local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(f0), fabsf(f1)),
                                                fmaxf(fabsf(f2), fabsf(f3))));
        }
    }
    for (int k = k4_limit + threadIdx.x; k < K; k += blockDim.x) {
        float v;
        if constexpr (__is_same(InputT, __nv_bfloat16))
            v = __bfloat162float(x_row[k]);
        else
            v = x_row[k];
        if (smooth) v /= smooth[k];
        local_max = fmaxf(local_max, fabsf(v));
    }

    fdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s)
            fdata[threadIdx.x] = fmaxf(fdata[threadIdx.x], fdata[threadIdx.x + s]);
        __syncthreads();
    }
    float absmax = fdata[0];
    float inv_scale = (absmax > 1e-10f) ? 127.0f / absmax : 0.0f;
    if (threadIdx.x == 0) x_scale[row] = absmax / 127.0f;
    __syncthreads();

    // Pass 2: quantize with vectorized loads and packed INT8 stores
    int local_sum = 0;

    if constexpr (__is_same(InputT, float)) {
        for (int k = threadIdx.x * 4; k < k4_limit; k += blockDim.x * 4) {
            float4 v = *reinterpret_cast<const float4*>(x_row + k);
            if (smooth) {
                v.x /= smooth[k]; v.y /= smooth[k+1]; v.z /= smooth[k+2]; v.w /= smooth[k+3];
            }
            int q0 = max(-128, min(127, __float2int_rn(v.x * inv_scale)));
            int q1 = max(-128, min(127, __float2int_rn(v.y * inv_scale)));
            int q2 = max(-128, min(127, __float2int_rn(v.z * inv_scale)));
            int q3 = max(-128, min(127, __float2int_rn(v.w * inv_scale)));
            *reinterpret_cast<int32_t*>(out_row + k) =
                (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
            if constexpr (NEED_ROWSUM) local_sum += q0 + q1 + q2 + q3;
        }
    } else {
        for (int k = threadIdx.x * 4; k < k4_limit; k += blockDim.x * 4) {
            __nv_bfloat162 v01 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k);
            __nv_bfloat162 v23 = *reinterpret_cast<const __nv_bfloat162*>(x_row + k + 2);
            float f0 = __bfloat162float(v01.x), f1 = __bfloat162float(v01.y);
            float f2 = __bfloat162float(v23.x), f3 = __bfloat162float(v23.y);
            if (smooth) {
                f0 /= smooth[k]; f1 /= smooth[k+1]; f2 /= smooth[k+2]; f3 /= smooth[k+3];
            }
            int q0 = max(-128, min(127, __float2int_rn(f0 * inv_scale)));
            int q1 = max(-128, min(127, __float2int_rn(f1 * inv_scale)));
            int q2 = max(-128, min(127, __float2int_rn(f2 * inv_scale)));
            int q3 = max(-128, min(127, __float2int_rn(f3 * inv_scale)));
            *reinterpret_cast<int32_t*>(out_row + k) =
                (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
            if constexpr (NEED_ROWSUM) local_sum += q0 + q1 + q2 + q3;
        }
    }
    for (int k = k4_limit + threadIdx.x; k < K; k += blockDim.x) {
        float v;
        if constexpr (__is_same(InputT, __nv_bfloat16))
            v = __bfloat162float(x_row[k]);
        else
            v = x_row[k];
        if (smooth) v /= smooth[k];
        int q = __float2int_rn(v * inv_scale);
        q = max(-128, min(127, q));
        out_row[k] = (int8_t)q;
        if constexpr (NEED_ROWSUM) local_sum += q;
    }

    if constexpr (NEED_ROWSUM) {
        int* idata = reinterpret_cast<int*>(qsmem);
        idata[threadIdx.x] = local_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < (unsigned)s)
                idata[threadIdx.x] += idata[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) x_rowsum[row] = (float)idata[0];
    }
}

// Launcher helper (dispatches NEED_ROWSUM × InputT template)
template<typename InputT>
static void quantize_activations_dispatch(const InputT* X, int8_t* X_int8,
                                           float* x_scale, float* x_rowsum,
                                           int M, int K, const float* smooth,
                                           bool need_rowsum) {
    int threads = 256;
    if (need_rowsum) {
        quantize_activations_kernel<true, InputT><<<M, threads, threads * sizeof(float)>>>(
            X, X_int8, x_scale, x_rowsum, smooth, M, K);
    } else {
        quantize_activations_kernel<false, InputT><<<M, threads, threads * sizeof(float)>>>(
            X, X_int8, x_scale, x_rowsum, smooth, M, K);
    }
}

void quantize_activations_int8_cuda(const __nv_bfloat16* X, int8_t* X_int8,
                                     float* x_scale, float* x_rowsum,
                                     int M, int K, const float* smooth,
                                     bool need_rowsum) {
    quantize_activations_dispatch(X, X_int8, x_scale, x_rowsum, M, K, smooth, need_rowsum);
}

void quantize_activations_int8_cuda(const float* X, int8_t* X_int8,
                                     float* x_scale, float* x_rowsum,
                                     int M, int K, const float* smooth,
                                     bool need_rowsum) {
    quantize_activations_dispatch(X, X_int8, x_scale, x_rowsum, M, K, smooth, need_rowsum);
}

// ============================================================================
// Post-process INT8 GEMM: apply dequant correction + bias
// Y[m,n] = x_scale[m] * w_scale[n] * (Y[m,n] - zp[n] * x_rowsum[m]) + bias[n]
// ============================================================================

// Vectorized: processes 4 elements per thread via int4 load / float4 store (16B coalesced)
// Template on HAS_ZP to avoid branching on zp/x_rowsum in the hot loop
template<bool HAS_ZP>
__global__ void int8_gemm_dequant_kernel(
    const int32_t* __restrict__ Y_i32,        // [M, N] INT32 from GEMM
    float* __restrict__ Y_out,                // [M, N] F32 output
    const float* __restrict__ x_scale,        // [M]
    const void* __restrict__ w_scale,         // [N] F32/FP16/BF16
    int w_scale_dtype,
    const int8_t* __restrict__ zp,            // [N] or nullptr (only read when HAS_ZP)
    const float* __restrict__ x_rowsum,       // [M] or nullptr (only read when HAS_ZP)
    const float* __restrict__ bias,           // [N] or nullptr
    int M, int N)
{
    int64_t total = (int64_t)M * N;
    int64_t vid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t base = vid * 4;
    if (base >= total) return;

    int m = (int)(base / N);
    int n = (int)(base % N);

    // Vectorized path: all 4 elements in the same row and within bounds
    if (n + 3 < N && base + 3 < total) {
        // Coalesced int4 load (4 x INT32 = 16 bytes)
        int4 raw4 = *reinterpret_cast<const int4*>(Y_i32 + base);
        float xs = x_scale[m];

        // Load 4 consecutive w_scale values
        float ws0 = load_quant_scale_device(w_scale, w_scale_dtype, (int64_t)n);
        float ws1 = load_quant_scale_device(w_scale, w_scale_dtype, (int64_t)(n + 1));
        float ws2 = load_quant_scale_device(w_scale, w_scale_dtype, (int64_t)(n + 2));
        float ws3 = load_quant_scale_device(w_scale, w_scale_dtype, (int64_t)(n + 3));

        float4 out;
        if constexpr (HAS_ZP) {
            float rowsum = x_rowsum[m];
            out.x = xs * ws0 * ((float)raw4.x - (float)zp[n]     * rowsum);
            out.y = xs * ws1 * ((float)raw4.y - (float)zp[n + 1] * rowsum);
            out.z = xs * ws2 * ((float)raw4.z - (float)zp[n + 2] * rowsum);
            out.w = xs * ws3 * ((float)raw4.w - (float)zp[n + 3] * rowsum);
        } else {
            out.x = xs * ws0 * (float)raw4.x;
            out.y = xs * ws1 * (float)raw4.y;
            out.z = xs * ws2 * (float)raw4.z;
            out.w = xs * ws3 * (float)raw4.w;
        }

        if (bias) {
            out.x += bias[n];
            out.y += bias[n + 1];
            out.z += bias[n + 2];
            out.w += bias[n + 3];
        }

        *reinterpret_cast<float4*>(Y_out + base) = out;
    } else {
        // Scalar fallback: row boundary or tail elements
        for (int i = 0; i < 4; i++) {
            int64_t idx = base + i;
            if (idx >= total) break;
            int mi = (int)(idx / N);
            int ni = (int)(idx % N);
            float raw = (float)Y_i32[idx];
            float xs = x_scale[mi];
            float ws = load_quant_scale_device(w_scale, w_scale_dtype, (int64_t)ni);
            float correction = HAS_ZP ? (float)zp[ni] * x_rowsum[mi] : 0.0f;
            float val = xs * ws * (raw - correction);
            if (bias) val += bias[ni];
            Y_out[idx] = val;
        }
    }
}

void int8_gemm_dequant_cuda(const int32_t* Y_i32, float* Y_out,
                             const float* x_scale,
                             const void* w_scale, DType w_scale_dtype,
                             const int8_t* zp, const float* x_rowsum,
                             const float* bias, int M, int N) {
    int threads = 256;
    int64_t total = (int64_t)M * N;
    // Each thread handles 4 elements (vectorized or scalar fallback)
    int64_t n_threads = (total + 3) / 4;
    int blocks = (int)((n_threads + threads - 1) / threads);
    if (zp) {
        int8_gemm_dequant_kernel<true><<<blocks, threads>>>(
            Y_i32, Y_out, x_scale, w_scale, (int)w_scale_dtype, zp, x_rowsum, bias, M, N);
    } else {
        int8_gemm_dequant_kernel<false><<<blocks, threads>>>(
            Y_i32, Y_out, x_scale, w_scale, (int)w_scale_dtype, zp, x_rowsum, bias, M, N);
    }
}

// ============================================================================
// Fused INT8 Dequant + BF16 WMMA GEMM
// Y[M,N] = X[M,K] @ W[N,K]^T + bias[N]
//
// Optimizations:
// - specialized paths for common group_size=128 and per-channel scale rows
// - BK=32 / BK=64 variants (heuristic selection)
// - full-tile kernel (no bounds checks) + edge fallback
// - vectorized X (bf162/cp.async 16B) and W (uint4/16-wide) loads
// - vectorized epilogue stores (float4) on interior tiles
// - cp.async for X activations (Ampere+), overlapped with synchronous W loads
// - scalar byte loads in edge tiles to handle K not divisible by 4
// ============================================================================

#define FUSED_BM 64
#define FUSED_BN 64
#define FUSED_PAD 8

using namespace nvcuda;

__device__ __forceinline__ void cp_async_copy16(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_src));
#else
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

template<int BK, int GROUP_SIZE_CONST, bool FULL_TILE, bool USE_CP_ASYNC_X>
__launch_bounds__(128, 5)
__global__ void fused_dequant_gemm_kernel_t(
    const __nv_bfloat16* __restrict__ X,     // [M, K] row-major BF16
    const int8_t* __restrict__ W,            // [N, K] row-major INT8
    const void* __restrict__ scales,         // [N, num_groups] F32/FP16/BF16
    int scales_dtype,
    const int8_t* __restrict__ zero_points,  // [N, num_groups] INT8 or nullptr (symmetric)
    const float* __restrict__ bias,          // [N] F32 or nullptr
    float* __restrict__ Y,                   // [M, N] row-major F32
    int M, int N, int K, int group_size,
    int skip_full_n, int skip_full_m)
{
    static_assert(BK == 32 || BK == 64, "Unsupported BK");
    static_assert((BK % 16) == 0, "BK must be multiple of 16");
    static_assert((BK % 4) == 0, "BK must be multiple of 4");

    constexpr int STRIDE = BK + FUSED_PAD;
    constexpr int X_PAIR_COUNT = FUSED_BM * (BK / 2);
    constexpr int W_VEC4_COUNT = FUSED_BN * (BK / 4);

    if constexpr (!FULL_TILE) {
        if (blockIdx.x < skip_full_n && blockIdx.y < skip_full_m) return;
    }

    const int block_n = blockIdx.x * FUSED_BN;
    const int block_m = blockIdx.y * FUSED_BM;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id >> 1;   // 0 or 1
    const int warp_n = warp_id & 1;    // 0 or 1
    const int num_groups = (K + group_size - 1) / group_size;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_base = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_x = smem_base;
    __nv_bfloat16* smem_w = smem_x + FUSED_BM * STRIDE;
    float* smem_scales = reinterpret_cast<float*>(smem_w + FUSED_BN * STRIDE);
    float* smem_zp = smem_scales + FUSED_BN;

    auto load_x_tile = [&](int k_start) {
        __nv_bfloat16* sx = smem_x;
        if constexpr (FULL_TILE && USE_CP_ASYNC_X) {
            constexpr int X_CHUNK16_COUNT = FUSED_BM * (BK / 8);  // 16B per chunk = 8 bf16
            #pragma unroll
            for (int idx16 = tid; idx16 < X_CHUNK16_COUNT; idx16 += 128) {
                int local_m = idx16 / (BK / 8);
                int local_k = (idx16 % (BK / 8)) * 8;
                __nv_bfloat16* dst = sx + local_m * STRIDE + local_k;
                const int gm = block_m + local_m;
                const int gk = k_start + local_k;
                cp_async_copy16(dst, X + (int64_t)gm * K + gk);
            }
            // Note: commit moved to caller so X and W cp.async share one group
        } else {
            #pragma unroll
            for (int idx2 = tid; idx2 < X_PAIR_COUNT; idx2 += 128) {
                int local_m = idx2 / (BK / 2);
                int local_k = (idx2 % (BK / 2)) * 2;
                __nv_bfloat16* dst = sx + local_m * STRIDE + local_k;
                const int gm = block_m + local_m;
                const int gk = k_start + local_k;
                if constexpr (FULL_TILE) {
                    const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(
                        X + (int64_t)gm * K + gk);
                    *reinterpret_cast<__nv_bfloat162*>(dst) = *src;
                } else {
                    __nv_bfloat16 v0 = __float2bfloat16(0.0f);
                    __nv_bfloat16 v1 = __float2bfloat16(0.0f);
                    if (gm < M && gk < K) v0 = X[(int64_t)gm * K + gk];
                    if (gm < M && gk + 1 < K) v1 = X[(int64_t)gm * K + gk + 1];
                    dst[0] = v0;
                    dst[1] = v1;
                }
            }
        }
    };

    auto store_dequant4 = [](__nv_bfloat16* dst, float f0, float f1, float f2, float f3) {
        *reinterpret_cast<__nv_bfloat162*>(dst + 0) = __floats2bfloat162_rn(f0, f1);
        *reinterpret_cast<__nv_bfloat162*>(dst + 2) = __floats2bfloat162_rn(f2, f3);
    };

    auto load_w_tile = [&](int k_start) {
        __nv_bfloat16* sw = smem_w;
        if constexpr (FULL_TILE) {
            // 16-wide vectorized: uint4 load (16 INT8 values), 4x fewer iterations
            constexpr int W_VEC16_COUNT = FUSED_BN * (BK / 16);
            #pragma unroll
            for (int idx16 = tid; idx16 < W_VEC16_COUNT; idx16 += 128) {
                int local_n = idx16 / (BK / 16);
                int local_k = (idx16 % (BK / 16)) * 16;
                __nv_bfloat16* dst = sw + local_n * STRIDE + local_k;
                const int gn = block_n + local_n;
                const int gk = k_start + local_k;

                uint4 raw = *reinterpret_cast<const uint4*>(W + (int64_t)gn * K + gk);
                const int8_t* q = reinterpret_cast<const int8_t*>(&raw);

                if constexpr (GROUP_SIZE_CONST == 128 || GROUP_SIZE_CONST == -1) {
                    float s = smem_scales[local_n];
                    float z = smem_zp[local_n];
                    store_dequant4(dst + 0,  ((float)q[0]-z)*s,  ((float)q[1]-z)*s,  ((float)q[2]-z)*s,  ((float)q[3]-z)*s);
                    store_dequant4(dst + 4,  ((float)q[4]-z)*s,  ((float)q[5]-z)*s,  ((float)q[6]-z)*s,  ((float)q[7]-z)*s);
                    store_dequant4(dst + 8,  ((float)q[8]-z)*s,  ((float)q[9]-z)*s,  ((float)q[10]-z)*s, ((float)q[11]-z)*s);
                    store_dequant4(dst + 12, ((float)q[12]-z)*s, ((float)q[13]-z)*s, ((float)q[14]-z)*s, ((float)q[15]-z)*s);
                } else {
                    int64_t scale_base = (int64_t)gn * num_groups;
                    #pragma unroll
                    for (int i = 0; i < 16; i += 4) {
                        int gki = gk + i;
                        int gi0 = gki / group_size, gi1 = (gki+1) / group_size;
                        int gi2 = (gki+2) / group_size, gi3 = (gki+3) / group_size;
                        float s0 = load_quant_scale_device(scales, scales_dtype, scale_base + gi0);
                        float s1 = load_quant_scale_device(scales, scales_dtype, scale_base + gi1);
                        float s2 = load_quant_scale_device(scales, scales_dtype, scale_base + gi2);
                        float s3 = load_quant_scale_device(scales, scales_dtype, scale_base + gi3);
                        float z0 = zero_points ? (float)zero_points[scale_base + gi0] : 0.0f;
                        float z1 = zero_points ? (float)zero_points[scale_base + gi1] : 0.0f;
                        float z2 = zero_points ? (float)zero_points[scale_base + gi2] : 0.0f;
                        float z3 = zero_points ? (float)zero_points[scale_base + gi3] : 0.0f;
                        store_dequant4(dst + i,
                                       ((float)q[i]-z0)*s0, ((float)q[i+1]-z1)*s1,
                                       ((float)q[i+2]-z2)*s2, ((float)q[i+3]-z3)*s3);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int idx4 = tid; idx4 < W_VEC4_COUNT; idx4 += 128) {
                int local_n = idx4 / (BK / 4);
                int local_k = (idx4 % (BK / 4)) * 4;
                __nv_bfloat16* dst = sw + local_n * STRIDE + local_k;
                const int gn = block_n + local_n;
                const int gk = k_start + local_k;
                if (gn < N && gk + 3 < K) {
                    // Scalar loads to avoid misaligned char4 when K % 4 != 0
                    const int8_t* src = W + (int64_t)gn * K + gk;
                    int8_t q0 = src[0], q1 = src[1], q2 = src[2], q3 = src[3];
                    if constexpr (GROUP_SIZE_CONST == 128 || GROUP_SIZE_CONST == -1) {
                        float s = smem_scales[local_n];
                        float z = smem_zp[local_n];
                        store_dequant4(dst,
                                       ((float)q0-z) * s, ((float)q1-z) * s,
                                       ((float)q2-z) * s, ((float)q3-z) * s);
                    } else {
                        int64_t scale_base = (int64_t)gn * num_groups;
                        int gi0 = (gk + 0) / group_size, gi1 = (gk + 1) / group_size;
                        int gi2 = (gk + 2) / group_size, gi3 = (gk + 3) / group_size;
                        float s0 = load_quant_scale_device(scales, scales_dtype, scale_base + gi0);
                        float s1 = load_quant_scale_device(scales, scales_dtype, scale_base + gi1);
                        float s2 = load_quant_scale_device(scales, scales_dtype, scale_base + gi2);
                        float s3 = load_quant_scale_device(scales, scales_dtype, scale_base + gi3);
                        float z0 = zero_points ? (float)zero_points[scale_base + gi0] : 0.0f;
                        float z1 = zero_points ? (float)zero_points[scale_base + gi1] : 0.0f;
                        float z2 = zero_points ? (float)zero_points[scale_base + gi2] : 0.0f;
                        float z3 = zero_points ? (float)zero_points[scale_base + gi3] : 0.0f;
                        store_dequant4(dst,
                                       ((float)q0-z0) * s0, ((float)q1-z1) * s1,
                                       ((float)q2-z2) * s2, ((float)q3-z3) * s3);
                    }
                } else {
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        int kk = gk + t;
                        __nv_bfloat16 v = __float2bfloat16(0.0f);
                        if (gn < N && kk < K) {
                            int8_t q = W[(int64_t)gn * K + kk];
                            float s, z;
                            if constexpr (GROUP_SIZE_CONST == 128 || GROUP_SIZE_CONST == -1) {
                                s = smem_scales[local_n];
                                z = smem_zp[local_n];
                            } else {
                                int64_t gi = (int64_t)gn * num_groups + kk / group_size;
                                s = load_quant_scale_device(scales, scales_dtype, gi);
                                z = zero_points ? (float)zero_points[gi] : 0.0f;
                            }
                            v = __float2bfloat16(((float)q - z) * s);
                        }
                        dst[t] = v;
                    }
                }
            }
        }
    };

    // WMMA accumulators: each warp has 2x2 tiles of 16x16
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    auto mma_stage = [&]() {
        __nv_bfloat16* sx = smem_x;
        __nv_bfloat16* sw = smem_w;
        #pragma unroll
        for (int ki = 0; ki < BK; ki += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_b[2];

            #pragma unroll
            for (int sm = 0; sm < 2; sm++) {
                wmma::load_matrix_sync(
                    frag_a[sm],
                    &sx[(warp_m * 32 + sm * 16) * STRIDE + ki],
                    STRIDE);
            }
            #pragma unroll
            for (int sn = 0; sn < 2; sn++) {
                wmma::load_matrix_sync(
                    frag_b[sn],
                    &sw[(warp_n * 32 + sn * 16) * STRIDE + ki],
                    STRIDE);
            }
            #pragma unroll
            for (int sm = 0; sm < 2; sm++)
                #pragma unroll
                for (int sn = 0; sn < 2; sn++)
                    wmma::mma_sync(acc[sm][sn], frag_a[sm], frag_b[sn], acc[sm][sn]);
        }
    };

    // Pre-load per-channel scales once (constant across all k iterations)
    if constexpr (GROUP_SIZE_CONST == -1) {
        if (tid < FUSED_BN) {
            int gn = block_n + tid;
            if (FULL_TILE || gn < N) {
                smem_scales[tid] = load_quant_scale_device(scales, scales_dtype, (int64_t)gn);
                smem_zp[tid] = zero_points ? (float)zero_points[gn] : 0.0f;
            }
        }
        __syncthreads();
    }

    if constexpr (FULL_TILE && USE_CP_ASYNC_X) {
        // Single-buffer: cp.async X overlaps with synchronous W load
        int cached_group = -1;
        for (int k_start = 0; k_start < K; k_start += BK) {
            // Cooperatively cache per-group scales in shared memory
            if constexpr (GROUP_SIZE_CONST == 128) {
                int group = k_start >> 7;
                if (group != cached_group) {
                    if (tid < FUSED_BN) {
                        int gn = block_n + tid;
                        int64_t gi = (int64_t)gn * num_groups + group;
                        smem_scales[tid] = load_quant_scale_device(scales, scales_dtype, gi);
                        smem_zp[tid] = zero_points ? (float)zero_points[gi] : 0.0f;
                    }
                    cached_group = group;
                    __syncthreads();
                }
            }
            load_x_tile(k_start);
            cp_async_commit_group();
            load_w_tile(k_start);
            cp_async_wait_all();
            __syncthreads();
            mma_stage();
            __syncthreads();
        }
    } else {
        int cached_group = -1;
        for (int k_start = 0; k_start < K; k_start += BK) {
            // Cooperatively cache per-group scales in shared memory
            if constexpr (GROUP_SIZE_CONST == 128) {
                int group = k_start >> 7;
                if (group != cached_group) {
                    if (tid < FUSED_BN) {
                        int gn = block_n + tid;
                        if (FULL_TILE || gn < N) {
                            int64_t gi = (int64_t)gn * num_groups + group;
                            smem_scales[tid] = load_quant_scale_device(scales, scales_dtype, gi);
                            smem_zp[tid] = zero_points ? (float)zero_points[gi] : 0.0f;
                        }
                    }
                    cached_group = group;
                    __syncthreads();
                }
            }
            load_x_tile(k_start);
            load_w_tile(k_start);
            __syncthreads();
            mma_stage();
            __syncthreads();
        }
    }

    // Output phase: store accumulators to shared memory, then write to global.
    // Pad output stride by 2 floats to eliminate shared memory bank conflicts
    // on wmma::store_matrix_sync (stride 66 → row j hits bank (j*66)%32 = j*2, all unique).
    constexpr int OUT_STRIDE = FUSED_BN + 2;
    float* smem_out = reinterpret_cast<float*>(smem_raw);

    #pragma unroll
    for (int sm = 0; sm < 2; sm++)
        #pragma unroll
        for (int sn = 0; sn < 2; sn++)
            wmma::store_matrix_sync(
                &smem_out[(warp_m * 32 + sm * 16) * OUT_STRIDE + (warp_n * 32 + sn * 16)],
                acc[sm][sn], OUT_STRIDE, wmma::mem_row_major);

    __syncthreads();

    if constexpr (FULL_TILE) {
        constexpr int OUT_VEC4 = FUSED_BM * (FUSED_BN / 4);  // 1024
        const int col_vec = tid & (FUSED_BN / 4 - 1);        // 0..15
        float4 b4 = make_float4(0.f, 0.f, 0.f, 0.f);
        if (bias) {
            b4 = *reinterpret_cast<const float4*>(bias + block_n + col_vec * 4);
        }

        for (int i4 = tid; i4 < OUT_VEC4; i4 += 128) {
            int local_m = i4 / (FUSED_BN / 4);
            int local_n4 = (i4 % (FUSED_BN / 4)) * 4;
            // Scalar smem reads avoid float4 misalignment from padded stride
            int si = local_m * OUT_STRIDE + local_n4;
            float4 v = make_float4(smem_out[si], smem_out[si + 1],
                                   smem_out[si + 2], smem_out[si + 3]);
            if (bias) {
                v.x += b4.x; v.y += b4.y; v.z += b4.z; v.w += b4.w;
            }
            *reinterpret_cast<float4*>(Y + (int64_t)(block_m + local_m) * N + block_n + local_n4) = v;
        }
    } else {
        const int lane_col = tid & (FUSED_BN - 1);  // constant across loop because stride=128
        float bias_val = 0.0f;
        bool bias_valid = false;
        if (bias) {
            int gn0 = block_n + lane_col;
            if (gn0 < N) {
                bias_val = bias[gn0];
                bias_valid = true;
            }
        }

        for (int i = tid; i < FUSED_BM * FUSED_BN; i += 128) {
            int local_m = i / FUSED_BN;
            int local_n = i % FUSED_BN;
            int gm = block_m + local_m;
            int gn = block_n + local_n;
            if (gm < M && gn < N) {
                float val = smem_out[local_m * OUT_STRIDE + local_n];
                if (bias_valid) val += bias_val;
                Y[(int64_t)gm * N + gn] = val;
            }
        }
    }
}

template<int BK, int GROUP_SIZE_CONST>
static void launch_fused_dequant_gemm_variant(const __nv_bfloat16* X, const int8_t* W,
                                              const void* scales, DType scales_dtype,
                                              const int8_t* zero_points,
                                              const float* bias, float* Y,
                                              int M, int N, int K, int group_size) {
    dim3 grid_all((N + FUSED_BN - 1) / FUSED_BN, (M + FUSED_BM - 1) / FUSED_BM);
    dim3 block(128);

    const size_t out_smem = (size_t)FUSED_BM * (FUSED_BN + 2) * sizeof(float);
    const size_t tile_smem_single = (size_t)(FUSED_BM + FUSED_BN) * (BK + FUSED_PAD) * sizeof(__nv_bfloat16)
                                  + (size_t)FUSED_BN * sizeof(float)    // scales
                                  + (size_t)FUSED_BN * sizeof(float);   // zero_points
    const size_t smem_edge = tile_smem_single > out_smem ? tile_smem_single : out_smem;
    const size_t smem_full = tile_smem_single > out_smem ? tile_smem_single : out_smem;

    const bool full_k = (K % BK) == 0;
    const int full_n = full_k ? (N / FUSED_BN) : 0;
    const int full_m = full_k ? (M / FUSED_BM) : 0;

    if (full_n > 0 && full_m > 0) {
        dim3 grid_full(full_n, full_m);
        fused_dequant_gemm_kernel_t<BK, GROUP_SIZE_CONST, true, true>
            <<<grid_full, block, smem_full>>>(
                X, W, scales, (int)scales_dtype, zero_points, bias, Y, M, N, K, group_size, 0, 0);
    }

    if (full_n != (int)grid_all.x || full_m != (int)grid_all.y) {
        fused_dequant_gemm_kernel_t<BK, GROUP_SIZE_CONST, false, false>
            <<<grid_all, block, smem_edge>>>(
                X, W, scales, (int)scales_dtype, zero_points, bias, Y, M, N, K, group_size, full_n, full_m);
    } else if (full_n == 0 || full_m == 0) {
        fused_dequant_gemm_kernel_t<BK, GROUP_SIZE_CONST, false, false>
            <<<grid_all, block, smem_edge>>>(
                X, W, scales, (int)scales_dtype, zero_points, bias, Y, M, N, K, group_size, 0, 0);
    }
}

void fused_dequant_gemm_cuda(const __nv_bfloat16* X, const int8_t* W,
                              const void* scales, DType scales_dtype,
                              const int8_t* zero_points,
                              const float* bias,
                              float* Y, int M, int N, int K, int group_size) {
    // Lightweight heuristic "autotune": prefer BK=64 on larger, aligned problems.
    bool use_bk64 = (K % 64 == 0) && (K >= 256) && (M >= 32) && (N >= 32);

    if (group_size == 128) {
        if (use_bk64) {
            launch_fused_dequant_gemm_variant<64, 128>(X, W, scales, scales_dtype, zero_points, bias, Y, M, N, K, group_size);
        } else {
            launch_fused_dequant_gemm_variant<32, 128>(X, W, scales, scales_dtype, zero_points, bias, Y, M, N, K, group_size);
        }
        return;
    }

    // Legacy per-channel scales (single scale per row) benefit from a dedicated specialization.
    if (group_size >= K) {
        if (use_bk64) {
            launch_fused_dequant_gemm_variant<64, -1>(X, W, scales, scales_dtype, zero_points, bias, Y, M, N, K, group_size);
        } else {
            launch_fused_dequant_gemm_variant<32, -1>(X, W, scales, scales_dtype, zero_points, bias, Y, M, N, K, group_size);
        }
        return;
    }

    if (use_bk64) {
        launch_fused_dequant_gemm_variant<64, 0>(X, W, scales, scales_dtype, zero_points, bias, Y, M, N, K, group_size);
    } else {
        launch_fused_dequant_gemm_variant<32, 0>(X, W, scales, scales_dtype, zero_points, bias, Y, M, N, K, group_size);
    }
}

#undef FUSED_BM
#undef FUSED_BN
#undef FUSED_PAD

// ============================================================================
// SmoothQuant: divide BF16 [M, K] activations by float [K] smooth factors
// out[m,k] = X[m,k] / smooth[k]
// ============================================================================

__global__ void smooth_div_bf16_kernel(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ smooth,
    __nv_bfloat16* __restrict__ out,
    int M, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * K;
    if (idx >= total) return;
    int k = idx % K;
    float v = __bfloat162float(X[idx]) / smooth[k];
    out[idx] = __float2bfloat16(v);
}

void smooth_div_bf16_cuda(const __nv_bfloat16* X, const float* smooth,
                           __nv_bfloat16* out, int M, int K) {
    int threads = 256;
    int total = M * K;
    int blocks = (total + threads - 1) / threads;
    smooth_div_bf16_kernel<<<blocks, threads>>>(X, smooth, out, M, K);
}

// ============================================================================
// Fused F32→BF16 + smooth division: out[i] = bf16(src[i] / smooth[i % K])
// Replaces separate f32_to_bf16 + smooth_div_bf16 (2 passes → 1 pass).
// Each element computes its own column index to handle row boundaries safely.
// ============================================================================

__global__ void f32_to_bf16_smooth_kernel(
    const float* __restrict__ src,
    const float* __restrict__ smooth,
    __nv_bfloat16* __restrict__ dst,
    int64_t total, int64_t K)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t k = idx % K;
    dst[idx] = __float2bfloat16(src[idx] / smooth[k]);
}

void f32_to_bf16_smooth_cuda(const float* src, const float* smooth,
                              __nv_bfloat16* dst, int64_t M, int64_t K) {
    int threads = 256;
    int64_t total = M * K;
    int blocks = (int)((total + threads - 1) / threads);
    f32_to_bf16_smooth_kernel<<<blocks, threads>>>(src, smooth, dst, total, K);
}

// ============================================================================
// Calibration: per-column absmax of BF16 [M, K] matrix
// Accumulates (element-wise max) into existing [K] float buffer.
// ============================================================================

__global__ void calibrate_act_absmax_kernel(
    const __nv_bfloat16* __restrict__ X,  // [M, K]
    float* __restrict__ accum,            // [K] running max (read-modify-write)
    int M, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float mx = 0.0f;
    for (int m = 0; m < M; m++) {
        mx = fmaxf(mx, fabsf(__bfloat162float(X[(int64_t)m * K + k])));
    }
    accum[k] = fmaxf(accum[k], mx);
}

void calibrate_act_absmax_cuda(const __nv_bfloat16* X, float* accum, int M, int K) {
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    calibrate_act_absmax_kernel<<<blocks, threads>>>(X, accum, M, K);
}

// ============================================================================
// Permute + F32→BF16: [L, n_head, d_head] F32 → [n_head, L, d_head] BF16
// Simple element-wise kernel for V preparation before flash attention.
// ============================================================================

__global__ void permute_and_convert_to_bf16_kernel(
    const float* __restrict__ src,       // [L, n_head, d_head] F32
    __nv_bfloat16* __restrict__ dst,     // [n_head, L, d_head] BF16
    int n_head, int L, int d_head)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n_head * L * d_head;
    if (idx >= total) return;

    // Decode output index: dst[h, l, d]
    int d = (int)(idx % d_head);
    int64_t rem = idx / d_head;
    int l = (int)(rem % L);
    int h = (int)(rem / L);

    // Read from src[l, h, d]
    int64_t src_idx = (int64_t)l * n_head * d_head + h * d_head + d;
    dst[idx] = __float2bfloat16(src[src_idx]);
}

void permute_and_convert_to_bf16_cuda(const float* src, __nv_bfloat16* dst,
                                       int n_head, int L, int d_head) {
    int64_t total = (int64_t)n_head * L * d_head;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    permute_and_convert_to_bf16_kernel<<<blocks, threads>>>(src, dst, n_head, L, d_head);
}
