#pragma once
#include "tensor.cuh"
#include "cublas_ops.cuh"
#include "rope.cuh"

// Forward declarations
void softmax_cuda(const float* x, float* out, int64_t rows, int64_t C);
void permute_4d_cuda(const float* input, float* output,
                     int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                     int p0, int p1, int p2, int p3);
void copy_cuda(const float* src, float* dst, int64_t n);
void apply_attention_mask_cuda(float* scores, const float* mask,
                                int64_t n_head, int64_t L_q, int64_t L_k);

// Scaled dot-product attention with RoPE
// Q, K, V: [L, n_head, d_head] (batch=1, already squeezed)
// pe: [L, d_head/2 * 4] on GPU
// output: [L, n_head * d_head]
//
// Steps:
// 1. Permute Q,K to [n_head, L, d_head]
// 2. Apply RoPE to Q and K
// 3. scores = Q @ K^T / sqrt(d_head) → [n_head, L, L]
// 4. softmax(scores)
// 5. output = scores @ V → [n_head, L, d_head]
// 6. Permute to [L, n_head, d_head] and reshape to [L, n_head*d_head]

static Tensor attention_with_rope(const Tensor& Q, const Tensor& K, const Tensor& V,
                                   const Tensor& pe, int n_head, int d_head,
                                   const float* attn_mask = nullptr) {
    assert(Q.on_gpu && K.on_gpu && V.on_gpu && pe.on_gpu);
    int64_t L = Q.shape[0];  // total sequence length

    // 1. Permute Q,K,V from [L, n_head, d_head] to [n_head, L, d_head]
    Tensor Q_p = Tensor::alloc({n_head, L, d_head}, DType::F32, true);
    Tensor K_p = Tensor::alloc({n_head, L, d_head}, DType::F32, true);
    Tensor V_p = Tensor::alloc({n_head, L, d_head}, DType::F32, true);

    permute_4d_cuda(Q.f32(), Q_p.f32(), L, n_head, d_head, 1, 1, 0, 2, 3);
    permute_4d_cuda(K.f32(), K_p.f32(), L, n_head, d_head, 1, 1, 0, 2, 3);
    permute_4d_cuda(V.f32(), V_p.f32(), L, n_head, d_head, 1, 1, 0, 2, 3);

    // 2. Apply RoPE to Q and K
    Tensor Q_rope = Tensor::alloc({n_head, L, d_head}, DType::F32, true);
    Tensor K_rope = Tensor::alloc({n_head, L, d_head}, DType::F32, true);
    apply_rope_cuda(Q_p.f32(), pe.f32(), Q_rope.f32(), n_head, L, d_head);
    apply_rope_cuda(K_p.f32(), pe.f32(), K_rope.f32(), n_head, L, d_head);

    // 3. Attention scores: [n_head, L, L] = Q_rope @ K_rope^T / sqrt(d_head)
    // For each head h: scores[h] = Q_rope[h] @ K_rope[h]^T
    float scale = 1.0f / sqrtf((float)d_head);
    Tensor scores = Tensor::alloc({(int64_t)n_head, L, L}, DType::F32, true);

    // K_rope^T: we need [n_head, d_head, L] for matmul
    // scores[h] = Q[h] @ K[h]^T where Q[h] is [L, d_head] and K[h]^T is [d_head, L]
    // Using batched: A = Q_rope [n_head, L, d_head], B_T = K_rope [n_head, L, d_head]
    // We want C[h] = A[h] @ B[h]^T = [L, L]
    // cuBLAS: for each batch, C = A @ B^T
    {
        float alpha = scale, beta = 0.0f;
        // For C = A @ B^T in row-major:
        // cuBLAS: C^T = B @ A^T
        // A is [L, d_head] row-major → cuBLAS sees [d_head, L]
        // B is [L, d_head] row-major → we want B^T = [d_head, L]
        // C is [L, L]
        // C^T = B @ A^T: cuBLAS(N, T, L, L, d_head, B_data, d_head, A_data, d_head, C_data, L)
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            L, L, d_head,
            &alpha,
            K_rope.data, CUDA_R_32F, d_head, L * d_head,
            Q_rope.data, CUDA_R_32F, d_head, L * d_head,
            &beta,
            scores.data, CUDA_R_32F, L, L * L,
            n_head,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // 3.5. Apply attention mask (if provided): scores[h, q, k] += mask[k]
    if (attn_mask) {
        apply_attention_mask_cuda(scores.f32(), attn_mask, n_head, L, L);
    }

    // 4. Softmax along last dim
    softmax_cuda(scores.f32(), scores.f32(), (int64_t)n_head * L, L);

    // 5. output = scores @ V → [n_head, L, d_head]
    Tensor attn_out = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::F32, true);
    {
        float alpha = 1.0f, beta = 0.0f;
        // scores[h]: [L, L], V_p[h]: [L, d_head] -> out[h]: [L, d_head]
        // C = A @ B in row-major:
        // cuBLAS: C^T = B^T @ A^T → cublasGemm(N, N, d_head, L, L, B, d_head, A, L, C, d_head)
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_head, L, L,
            &alpha,
            V_p.data, CUDA_R_32F, d_head, L * d_head,
            scores.data, CUDA_R_32F, L, L * L,
            &beta,
            attn_out.data, CUDA_R_32F, d_head, L * d_head,
            n_head,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // 6. Permute from [n_head, L, d_head] to [L, n_head, d_head] then reshape to [L, n_head*d_head]
    Tensor output = Tensor::alloc({L, (int64_t)(n_head * d_head)}, DType::F32, true);
    // Permute [n_head, L, d_head] → [L, n_head, d_head]
    permute_4d_cuda(attn_out.f32(), output.f32(),
                    n_head, L, d_head, 1,  // treat as 4D with last dim=1
                    1, 0, 2, 3);

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

    // Permute to [n_head, L, d_head]
    Tensor Q_p = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::F32, true);
    Tensor K_p = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::F32, true);
    Tensor V_p = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::F32, true);

    permute_4d_cuda(Q.f32(), Q_p.f32(), L, n_head, d_head, 1, 1, 0, 2, 3);
    permute_4d_cuda(K.f32(), K_p.f32(), L, n_head, d_head, 1, 1, 0, 2, 3);
    permute_4d_cuda(V.f32(), V_p.f32(), L, n_head, d_head, 1, 1, 0, 2, 3);

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
        add_cuda(scores.f32(), rel_bias, scores.f32(), (int64_t)n_head * L * L);
    }

    // Apply attention mask (block padding positions in key dimension)
    if (attn_mask) {
        apply_attention_mask_cuda(scores.f32(), attn_mask, n_head, L, L);
    }

    // Softmax
    softmax_cuda(scores.f32(), scores.f32(), (int64_t)n_head * L, L);

    // Output = scores @ V
    Tensor attn_out = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::F32, true);
    {
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_head, L, L,
            &alpha,
            V_p.data, CUDA_R_32F, d_head, L * d_head,
            scores.data, CUDA_R_32F, L, L * L,
            &beta,
            attn_out.data, CUDA_R_32F, d_head, L * d_head,
            n_head,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // Permute back to [L, n_head*d_head]
    Tensor output = Tensor::alloc({L, (int64_t)(n_head * d_head)}, DType::F32, true);
    permute_4d_cuda(attn_out.f32(), output.f32(),
                    n_head, L, d_head, 1,
                    1, 0, 2, 3);
    return output;
}
