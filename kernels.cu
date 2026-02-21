#include "tensor.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
