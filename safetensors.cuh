#pragma once
#include "tensor.cuh"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

struct TensorInfo {
    DType dtype;
    std::vector<int64_t> shape;
    int64_t data_offset;  // offset from start of data section
    int64_t nbytes;
};

struct SafetensorsFile {
    int fd = -1;
    void* mapped = nullptr;
    size_t file_size = 0;
    size_t header_size = 0;
    const char* data_start = nullptr;
    std::unordered_map<std::string, TensorInfo> tensors;

    ~SafetensorsFile() {
        if (mapped) munmap(mapped, file_size);
        if (fd >= 0) close(fd);
    }

    bool open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) { perror("open"); return false; }

        struct stat st;
        fstat(fd, &st);
        file_size = st.st_size;

        mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) { perror("mmap"); return false; }

        // Read 8-byte header size (little-endian u64)
        uint64_t hdr_size;
        memcpy(&hdr_size, mapped, 8);
        header_size = hdr_size;
        data_start = (const char*)mapped + 8 + header_size;

        // Parse JSON header
        std::string json((const char*)mapped + 8, header_size);
        parse_header(json);
        return true;
    }

    // Load a tensor to GPU, converting to target dtype if needed
    Tensor load_tensor(const std::string& name, DType target_dt = DType::F32) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            fprintf(stderr, "Tensor not found: %s\n", name.c_str());
            exit(1);
        }
        const TensorInfo& info = it->second;
        int ndim = (int)info.shape.size();
        int64_t numel = 1;
        for (auto d : info.shape) numel *= d;

        // First load raw bytes to CPU
        const void* src = data_start + info.data_offset;

        // Upload to GPU in original dtype
        Tensor gpu_orig = Tensor::alloc_shape(ndim, info.shape.data(), info.dtype, true);
        CHECK_CUDA(cudaMemcpy(gpu_orig.data, src, info.nbytes, cudaMemcpyHostToDevice));

        if (info.dtype == target_dt) return gpu_orig;

        // Convert on GPU
        if (target_dt == DType::F32) {
            Tensor out = to_f32_gpu(gpu_orig);
            return out;
        }

        // For other conversions, go through F32
        if (info.dtype != DType::F32) {
            Tensor f32 = to_f32_gpu(gpu_orig);
            if (target_dt == DType::BF16) {
                Tensor out = Tensor::alloc_shape(ndim, info.shape.data(), DType::BF16, true);
                f32_to_bf16_cuda(f32.f32(), out.bf16(), numel);
                return out;
            } else if (target_dt == DType::FP16) {
                Tensor out = Tensor::alloc_shape(ndim, info.shape.data(), DType::FP16, true);
                f32_to_fp16_cuda(f32.f32(), out.fp16(), numel);
                return out;
            }
        }
        return gpu_orig;
    }

    // Load tensor keeping original dtype
    Tensor load_tensor_native(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            fprintf(stderr, "Tensor not found: %s\n", name.c_str());
            exit(1);
        }
        return load_tensor(name, it->second.dtype);
    }

    bool has_tensor(const std::string& name) const {
        return tensors.count(name) > 0;
    }

private:
    // Simple JSON parser for safetensors header
    // Format: {"tensor_name": {"dtype": "F16", "shape": [a,b], "data_offsets": [start, end]}, ...}
    void parse_header(const std::string& json) {
        size_t pos = 0;
        auto skip_ws = [&]() { while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) pos++; };
        auto expect = [&](char c) { skip_ws(); assert(pos < json.size() && json[pos] == c); pos++; };
        auto parse_string = [&]() -> std::string {
            skip_ws();
            assert(json[pos] == '"'); pos++;
            std::string s;
            while (pos < json.size() && json[pos] != '"') {
                if (json[pos] == '\\') { pos++; s += json[pos++]; }
                else s += json[pos++];
            }
            pos++; // closing "
            return s;
        };
        auto parse_int = [&]() -> int64_t {
            skip_ws();
            int64_t val = 0;
            bool neg = false;
            if (json[pos] == '-') { neg = true; pos++; }
            while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
                val = val * 10 + (json[pos] - '0');
                pos++;
            }
            return neg ? -val : val;
        };

        // Skip any JSON value (string, number, object, array, bool, null)
        std::function<void()> skip_value = [&]() {
            skip_ws();
            if (pos >= json.size()) return;
            char c = json[pos];
            if (c == '"') {
                parse_string();
            } else if (c == '{') {
                pos++;
                skip_ws();
                while (pos < json.size() && json[pos] != '}') {
                    parse_string(); // key
                    expect(':');
                    skip_value();
                    skip_ws();
                    if (pos < json.size() && json[pos] == ',') pos++;
                    skip_ws();
                }
                if (pos < json.size()) pos++; // }
            } else if (c == '[') {
                pos++;
                skip_ws();
                while (pos < json.size() && json[pos] != ']') {
                    skip_value();
                    skip_ws();
                    if (pos < json.size() && json[pos] == ',') pos++;
                    skip_ws();
                }
                if (pos < json.size()) pos++; // ]
            } else {
                // number, bool, null
                while (pos < json.size() && json[pos] != ',' && json[pos] != '}' && json[pos] != ']') pos++;
            }
        };

        expect('{');
        skip_ws();
        while (pos < json.size() && json[pos] != '}') {
            std::string key = parse_string();
            expect(':');
            skip_ws();

            if (key == "__metadata__") {
                skip_value();
            } else if (json[pos] == '{') {
                // Tensor entry
                pos++; // {
                TensorInfo info;
                std::string dt_str;
                std::vector<int64_t> offsets;

                while (pos < json.size() && json[pos] != '}') {
                    skip_ws();
                    std::string field = parse_string();
                    expect(':');
                    skip_ws();
                    if (field == "dtype") {
                        dt_str = parse_string();
                    } else if (field == "shape") {
                        expect('[');
                        skip_ws();
                        while (pos < json.size() && json[pos] != ']') {
                            info.shape.push_back(parse_int());
                            skip_ws();
                            if (json[pos] == ',') pos++;
                        }
                        pos++; // ]
                    } else if (field == "data_offsets") {
                        expect('[');
                        offsets.push_back(parse_int());
                        skip_ws();
                        if (json[pos] == ',') pos++;
                        offsets.push_back(parse_int());
                        skip_ws();
                        pos++; // ]
                    } else {
                        skip_value();
                    }
                    skip_ws();
                    if (json[pos] == ',') pos++;
                }
                pos++; // }

                if (dt_str == "F32") info.dtype = DType::F32;
                else if (dt_str == "F16") info.dtype = DType::FP16;
                else if (dt_str == "BF16") info.dtype = DType::BF16;
                else if (dt_str == "I8") info.dtype = DType::INT8;
                else {
                    skip_ws();
                    if (json[pos] == ',') pos++;
                    continue;
                }

                if (offsets.size() == 2) {
                    info.data_offset = offsets[0];
                    info.nbytes = offsets[1] - offsets[0];
                }

                tensors[key] = info;
            } else {
                skip_value();
            }

            skip_ws();
            if (json[pos] == ',') pos++;
            skip_ws();
        }
    }
};
