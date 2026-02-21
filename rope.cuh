#pragma once
#include "tensor.cuh"
#include <vector>
#include <cmath>
#include <cassert>

// Forward declarations of CUDA kernels
void apply_rope_cuda(const float* x, const float* pe, float* out,
                     int N_heads, int L, int d_head);

// ============================================================================
// RoPE position embedding generation (CPU side)
// ============================================================================

namespace RoPE {

// linspace(start, end, n) -> vector of n evenly spaced values
static std::vector<float> linspace(float start, float end, int n) {
    std::vector<float> out(n);
    if (n == 1) { out[0] = start; return out; }
    for (int i = 0; i < n; i++) {
        out[i] = start + (end - start) * i / (n - 1);
    }
    return out;
}

// Compute rope rotation matrices for a set of positions and frequency dimension
// Returns flat array: [pos_count, half_dim * 4] where each group of 4 is [cos, -sin, sin, cos]
static std::vector<float> rope(const std::vector<float>& pos, int dim, int theta) {
    int half_dim = dim / 2;
    std::vector<float> scale = linspace(0.f, (dim * 1.f - 2) / dim, half_dim);
    std::vector<float> omega(half_dim);
    for (int i = 0; i < half_dim; i++) {
        omega[i] = 1.0f / powf((float)theta, scale[i]);
    }

    size_t pos_count = pos.size();
    std::vector<float> result(pos_count * half_dim * 4);

    for (size_t i = 0; i < pos_count; i++) {
        for (int j = 0; j < half_dim; j++) {
            float angle = pos[i] * omega[j];
            float c = cosf(angle);
            float s = sinf(angle);
            size_t base = (i * half_dim + j) * 4;
            result[base + 0] = c;     // cos
            result[base + 1] = -s;    // -sin
            result[base + 2] = s;     // sin
            result[base + 3] = c;     // cos
        }
    }
    return result;
}

// Generate text position IDs for Chroma Radiance path
// Match stable-diffusion.cpp (FluxRunner with empty txt_arange_dims for this version):
// all text axes are 0.
static std::vector<std::vector<float>> gen_txt_ids(int context_len) {
    std::vector<std::vector<float>> ids(context_len, std::vector<float>(3, 0.0f));
    return ids;
}

// Generate image position IDs for Flux/Chroma
// Returns [h_patches * w_patches, 3] = [index, row, col]
static std::vector<std::vector<float>> gen_img_ids(int H, int W, int patch_size) {
    int h_len = (H + patch_size / 2) / patch_size;
    int w_len = (W + patch_size / 2) / patch_size;

    std::vector<std::vector<float>> ids(h_len * w_len, std::vector<float>(3, 0.0f));
    for (int i = 0; i < h_len; i++) {
        for (int j = 0; j < w_len; j++) {
            ids[i * w_len + j][0] = 0.0f;    // index
            ids[i * w_len + j][1] = (float)i; // row
            ids[i * w_len + j][2] = (float)j; // col
        }
    }
    return ids;
}

// Compute positional embeddings for all tokens (text + image)
// axes_dim = [16, 56, 56], theta = 10000
// Returns flat array: [total_tokens, d_head/2 * 4] = [total_tokens, 256]
// where d_head/2 = sum(axes_dim)/2 = (16+56+56)/2 = 64
static std::vector<float> gen_pe(int H, int W, int patch_size, int context_len,
                                  int theta, const std::vector<int>& axes_dim) {
    auto txt_ids = gen_txt_ids(context_len);
    auto img_ids = gen_img_ids(H, W, patch_size);

    int total_tokens = context_len + (int)img_ids.size();

    // Concatenate: [txt_ids; img_ids], each [N, 3]
    std::vector<std::vector<float>> all_ids(total_tokens, std::vector<float>(3, 0.0f));
    for (int i = 0; i < context_len; i++) all_ids[i] = txt_ids[i];
    for (int i = 0; i < (int)img_ids.size(); i++) all_ids[context_len + i] = img_ids[i];

    // Transpose to [3, total_tokens]
    std::vector<std::vector<float>> trans(3, std::vector<float>(total_tokens));
    for (int i = 0; i < total_tokens; i++) {
        for (int j = 0; j < 3; j++) trans[j][i] = all_ids[i][j];
    }

    // Compute rope embeddings for each axis and concatenate
    int emb_half = 0;
    for (int d : axes_dim) emb_half += d / 2;
    // emb_half = 64, total PE per token = 64 * 4 = 256

    std::vector<float> pe(total_tokens * emb_half * 4, 0.0f);
    int offset = 0;

    for (int axis = 0; axis < (int)axes_dim.size(); axis++) {
        auto emb = rope(trans[axis], axes_dim[axis], theta);
        // emb: [total_tokens, axes_dim[axis]/2 * 4]
        int feat_size = axes_dim[axis] / 2 * 4;
        for (int t = 0; t < total_tokens; t++) {
            for (int f = 0; f < feat_size; f++) {
                pe[t * emb_half * 4 + offset + f] = emb[t * feat_size + f];
            }
        }
        offset += feat_size;
    }

    return pe;
}

} // namespace RoPE

// ============================================================================
// CUDA kernel for applying RoPE (interleaved mode)
// x: [N_heads, L, d_head] (pre-permuted from [N, L, n_head, d_head])
// pe: [L, d_head/2, 4] = [L, d_head/2, 2, 2] flattened
// out: [N_heads, L, d_head]
// For each head h, position p, pair j:
//   x_even = x[h, p, 2*j], x_odd = x[h, p, 2*j+1]
//   out[h, p, 2*j]   = x_even * pe[p, j, 0] + x_odd * pe[p, j, 2]  (= x_e*cos + x_o*sin)
//   out[h, p, 2*j+1] = x_even * pe[p, j, 1] + x_odd * pe[p, j, 3]  (= -x_e*sin + x_o*cos)
// ============================================================================

__global__ void apply_rope_kernel(const float* x, const float* pe, float* out,
                                   int N_heads, int L, int d_head) {
    int half_d = d_head / 2;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N_heads * L * half_d;
    if (idx >= total) return;

    int h = idx / (L * half_d);
    int rem = idx % (L * half_d);
    int p = rem / half_d;
    int j = rem % half_d;

    float x_even = x[(int64_t)h * L * d_head + p * d_head + 2 * j];
    float x_odd  = x[(int64_t)h * L * d_head + p * d_head + 2 * j + 1];

    int pe_base = p * half_d * 4 + j * 4;
    float pe_cos     = pe[pe_base + 0]; // cos
    float pe_neg_sin = pe[pe_base + 1]; // -sin
    float pe_sin     = pe[pe_base + 2]; // sin
    float pe_cos2    = pe[pe_base + 3]; // cos

    // PE is a row-major 2x2 rotation matrix: [[cos, -sin], [sin, cos]]
    // Apply: out_even = cos*x_even + (-sin)*x_odd, out_odd = sin*x_even + cos*x_odd
    out[(int64_t)h * L * d_head + p * d_head + 2 * j]     = x_even * pe_cos     + x_odd * pe_neg_sin;
    out[(int64_t)h * L * d_head + p * d_head + 2 * j + 1] = x_even * pe_sin     + x_odd * pe_cos2;
}

void apply_rope_cuda(const float* x, const float* pe, float* out,
                     int N_heads, int L, int d_head) {
    int half_d = d_head / 2;
    int64_t total = (int64_t)N_heads * L * half_d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    apply_rope_kernel<<<blocks, threads>>>(x, pe, out, N_heads, L, d_head);
}

// ============================================================================
// Fused permute + RoPE + F32→BF16 kernel
// Reads from [L, N_heads, d_head] layout, applies RoPE, writes BF16 to [N_heads, L, d_head]
// Fuses 3 operations: permute_4d + apply_rope + f32_to_bf16
// ============================================================================

__global__ void apply_rope_fused_bf16_kernel(const float* x, const float* pe,
                                              __nv_bfloat16* out,
                                              int N_heads, int L, int d_head) {
    int half_d = d_head / 2;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N_heads * L * half_d;
    if (idx >= total) return;

    int h = idx / (L * half_d);
    int rem = idx % (L * half_d);
    int l = rem / half_d;
    int j = rem % half_d;

    // Read from [L, N_heads, d_head] layout
    int64_t in_base = (int64_t)l * N_heads * d_head + h * d_head;
    float x_even = x[in_base + 2 * j];
    float x_odd  = x[in_base + 2 * j + 1];

    // RoPE rotation
    int pe_base = l * half_d * 4 + j * 4;
    float pe_cos     = pe[pe_base + 0];
    float pe_neg_sin = pe[pe_base + 1];
    float pe_sin     = pe[pe_base + 2];
    float pe_cos2    = pe[pe_base + 3];

    float out_even = x_even * pe_cos     + x_odd * pe_neg_sin;
    float out_odd  = x_even * pe_sin     + x_odd * pe_cos2;

    // Write to [N_heads, L, d_head] layout in BF16
    int64_t out_base = (int64_t)h * L * d_head + l * d_head;
    out[out_base + 2 * j]     = __float2bfloat16(out_even);
    out[out_base + 2 * j + 1] = __float2bfloat16(out_odd);
}

void apply_rope_fused_bf16_cuda(const float* x, const float* pe, __nv_bfloat16* out,
                                 int N_heads, int L, int d_head) {
    int half_d = d_head / 2;
    int64_t total = (int64_t)N_heads * L * half_d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    apply_rope_fused_bf16_kernel<<<blocks, threads>>>(x, pe, out, N_heads, L, d_head);
}

// F32 variant: reads [L, N_heads, d_head], applies RoPE, writes F32 to [N_heads, L, d_head]
// Used when BF16 attention is not desired (e.g., T5)
__global__ void apply_rope_fused_f32_kernel(const float* x, const float* pe,
                                             float* out,
                                             int N_heads, int L, int d_head) {
    int half_d = d_head / 2;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N_heads * L * half_d;
    if (idx >= total) return;

    int h = idx / (L * half_d);
    int rem = idx % (L * half_d);
    int l = rem / half_d;
    int j = rem % half_d;

    // Read from [L, N_heads, d_head] layout
    int64_t in_base = (int64_t)l * N_heads * d_head + h * d_head;
    float x_even = x[in_base + 2 * j];
    float x_odd  = x[in_base + 2 * j + 1];

    // RoPE rotation
    int pe_base = l * half_d * 4 + j * 4;
    float pe_cos     = pe[pe_base + 0];
    float pe_neg_sin = pe[pe_base + 1];
    float pe_sin     = pe[pe_base + 2];
    float pe_cos2    = pe[pe_base + 3];

    float out_even = x_even * pe_cos     + x_odd * pe_neg_sin;
    float out_odd  = x_even * pe_sin     + x_odd * pe_cos2;

    // Write to [N_heads, L, d_head] layout in F32
    int64_t out_base = (int64_t)h * L * d_head + l * d_head;
    out[out_base + 2 * j]     = out_even;
    out[out_base + 2 * j + 1] = out_odd;
}

void apply_rope_fused_f32_cuda(const float* x, const float* pe, float* out,
                                int N_heads, int L, int d_head) {
    int half_d = d_head / 2;
    int64_t total = (int64_t)N_heads * L * half_d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    apply_rope_fused_f32_kernel<<<blocks, threads>>>(x, pe, out, N_heads, L, d_head);
}
