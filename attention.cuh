#pragma once
#include "tensor.cuh"
#include "cublas_ops.cuh"
#include "rope.cuh"

// Forward declarations
void softmax_cuda(const float* x, float* out, int64_t rows, int64_t C);
void softmax_with_mask_cuda(const float* x, const float* mask, float* out,
                             int64_t rows, int64_t C);
void permute_4d_cuda(const float* input, float* output,
                     int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                     int p0, int p1, int p2, int p3);
void copy_cuda(const float* src, float* dst, int64_t n);
void apply_attention_mask_cuda(float* scores, const float* mask,
                                int64_t n_head, int64_t L_q, int64_t L_k);

// Scaled dot-product attention with RoPE — fully optimized
// Q, K, V: [L, n_head, d_head] (batch=1, already squeezed)
// pe: [L, d_head/2 * 4] on GPU
// output: [L, n_head * d_head]
//
// Optimizations vs original:
// 1. Fused permute+RoPE+BF16 for Q,K (eliminates 2 permute_4d + 2 apply_rope + 2 f32_to_bf16)
// 2. BF16 tensor core Q@K^T GEMM (replaces F32 sgemm)
// 3. Fused softmax+mask (eliminates apply_attention_mask kernel)
// 4. Strided V GEMM — no V permute needed
// 5. Strided output write — no output permute needed

static Tensor attention_with_rope(const Tensor& Q, const Tensor& K, const Tensor& V,
                                   const Tensor& pe, int n_head, int d_head,
                                   const float* attn_mask = nullptr) {
    assert(Q.on_gpu && K.on_gpu && V.on_gpu && pe.on_gpu);
    int64_t L = Q.shape[0];  // total sequence length

    // 1. Fused permute+RoPE+BF16: [L, n_head, d_head] → [n_head, L, d_head] BF16
    Tensor Q_rope_bf16 = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::BF16, true);
    Tensor K_rope_bf16 = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::BF16, true);

    apply_rope_fused_bf16_cuda(Q.f32(), pe.f32(), Q_rope_bf16.bf16(), n_head, L, d_head);
    apply_rope_fused_bf16_cuda(K.f32(), pe.f32(), K_rope_bf16.bf16(), n_head, L, d_head);

    // 2. Q@K^T: BF16 tensor core GEMM with F32 accumulation
    // scores[h] = Q_rope[h] @ K_rope[h]^T, shape [n_head, L, L]
    float scale = 1.0f / sqrtf((float)d_head);
    Tensor scores = Tensor::alloc({(int64_t)n_head, L, L}, DType::F32, true);
    {
        float alpha = scale, beta = 0.0f;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            L, L, d_head,
            &alpha,
            K_rope_bf16.data, CUDA_R_16BF, d_head, L * d_head,
            Q_rope_bf16.data, CUDA_R_16BF, d_head, L * d_head,
            &beta,
            scores.data, CUDA_R_32F, L, L * L,
            n_head,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // 3. Fused softmax + mask (eliminates separate apply_attention_mask kernel)
    softmax_with_mask_cuda(scores.f32(), attn_mask, scores.f32(), (int64_t)n_head * L, L);

    // 4. scores @ V using strided GEMM — V stays in [L, n_head, d_head] layout
    // Output written directly in [L, n_head, d_head] layout (strided)
    Tensor output = Tensor::alloc({L, (int64_t)(n_head * d_head)}, DType::F32, true);
    {
        float alpha = 1.0f, beta = 0.0f;
        // V is [L, n_head, d_head] row-major:
        //   V[l][h][d] = V_ptr + l * n_head * d_head + h * d_head + d
        // For batch h: base = V_ptr + h * d_head, ld = n_head * d_head, stride = d_head
        // Output is [L, n_head, d_head] row-major (same layout as [L, n_head*d_head]):
        //   out[l][h][d] = out_ptr + l * n_head * d_head + h * d_head + d
        // For batch h: base = out_ptr + h * d_head, ld = n_head * d_head, stride = d_head
        int64_t V_ld = (int64_t)n_head * d_head;
        int64_t V_stride = d_head;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_head, L, L,
            &alpha,
            V.data, CUDA_R_32F, V_ld, V_stride,         // V in [L, n_head, d_head]
            scores.data, CUDA_R_32F, L, L * L,           // scores in [n_head, L, L]
            &beta,
            output.data, CUDA_R_32F, V_ld, V_stride,     // output in [L, n_head, d_head]
            n_head,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // Output is [L, n_head * d_head] — no permute needed!
    return output;
}

// T5 attention (no RoPE, uses relative position bias)
// Q, K, V: [L, n_head, d_head]
// rel_bias: [n_head, L, L] or nullptr
// attn_mask: [L] where 0.0 = attend, -inf = block (for key dim), or nullptr
// output: [L, n_head * d_head]
static Tensor t5_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                            const float* rel_bias, int n_head, int d_head, float scale,
                            const float* attn_mask = nullptr) {
    int64_t L = Q.shape[0];

    // Permute to [n_head, L, d_head] — T5 doesn't use RoPE so we use F32 fused permute
    // (just the permute part without RoPE, writing F32)
    Tensor Q_p = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::F32, true);
    Tensor K_p = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::F32, true);

    permute_4d_cuda(Q.f32(), Q_p.f32(), L, n_head, d_head, 1, 1, 0, 2, 3);
    permute_4d_cuda(K.f32(), K_p.f32(), L, n_head, d_head, 1, 1, 0, 2, 3);

    // Scores = Q @ K^T * scale
    Tensor scores = Tensor::alloc({(int64_t)n_head, L, L}, DType::F32, true);
    {
        float alpha = scale, beta = 0.0f;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            L, L, d_head,
            &alpha,
            K_p.data, CUDA_R_32F, d_head, L * d_head,
            Q_p.data, CUDA_R_32F, d_head, L * d_head,
            &beta,
            scores.data, CUDA_R_32F, L, L * L,
            n_head,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // Add relative position bias
    if (rel_bias) {
        extern void add_cuda(const float* a, const float* b, float* out, int64_t n);
        add_cuda(scores.f32(), rel_bias, scores.f32(), (int64_t)n_head * L * L);
    }

    // Fused softmax + mask
    softmax_with_mask_cuda(scores.f32(), attn_mask, scores.f32(), (int64_t)n_head * L, L);

    // Output = scores @ V using strided V layout
    Tensor output = Tensor::alloc({L, (int64_t)(n_head * d_head)}, DType::F32, true);
    {
        float alpha = 1.0f, beta = 0.0f;
        int64_t V_ld = (int64_t)n_head * d_head;
        int64_t V_stride = d_head;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_head, L, L,
            &alpha,
            V.data, CUDA_R_32F, V_ld, V_stride,
            scores.data, CUDA_R_32F, L, L * L,
            &beta,
            output.data, CUDA_R_32F, V_ld, V_stride,
            n_head,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    return output;
}
