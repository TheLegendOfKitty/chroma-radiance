#pragma once
#include "tensor.cuh"
#include "cublas_ops.cuh"
#include "rope.cuh"
#include "cudnn_attention.cuh"

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

// Scaled dot-product attention with RoPE — cuDNN flash attention
// Q, K, V: [L, n_head, d_head] (batch=1, already squeezed)
// pe: [L, d_head/2 * 4] on GPU
// output: [L, n_head * d_head]

static Tensor attention_with_rope(const Tensor& Q, const Tensor& K, const Tensor& V,
                                   const Tensor& pe, int n_head, int d_head,
                                   const float* attn_mask = nullptr) {
    return cudnn_attention_with_rope(Q, K, V, pe, n_head, d_head, attn_mask);
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
