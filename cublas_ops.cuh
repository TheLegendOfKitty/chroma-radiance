#pragma once
#include "tensor.cuh"
#include <cublas_v2.h>

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        exit(1); \
    } \
} while(0)

static cublasHandle_t g_cublas = nullptr;

static void init_cublas() {
    if (!g_cublas) CHECK_CUBLAS(cublasCreate(&g_cublas));
    CHECK_CUBLAS(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));
}

// Linear: output = x @ W^T + bias
// x: [M, K], W: [N, K], bias: [N] or null, output: [M, N]
// All stored row-major. Uses F32 accumulation.
static void linear(const Tensor& x, const Tensor& W, const Tensor* bias, Tensor& output) {
    assert(x.on_gpu && W.on_gpu && output.on_gpu);
    assert(x.ndim >= 1 && W.ndim == 2);

    int64_t M = x.numel() / x.shape[x.ndim - 1]; // batch*seq
    int64_t K = x.shape[x.ndim - 1];
    int64_t N = W.shape[0];
    assert(W.shape[1] == K);
    assert(output.numel() == M * N);

    float alpha = 1.0f, beta = 0.0f;

    // If bias, copy it to output first then use beta=1
    if (bias) {
        assert(bias->on_gpu);
        // Broadcast bias [N] to [M, N]
        // Use a kernel for this
        extern void broadcast_bias_cuda(const float* bias, float* output, int64_t M, int64_t N);
        if (bias->dtype == DType::F32 && output.dtype == DType::F32) {
            broadcast_bias_cuda(bias->f32(), output.f32(), M, N);
        }
        beta = 1.0f;
    }

    if (x.dtype == DType::F32 && W.dtype == DType::F32) {
        // F32 GEMM: output = x @ W^T
        // cuBLAS column-major: C^T = W @ x^T
        // op(A) = W [N,K], op(B) = x^T [K,M], C = output^T [N,M]
        CHECK_CUBLAS(cublasGemmEx(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            W.data, CUDA_R_32F, K,    // W row-major [N,K], transposed
            x.data, CUDA_R_32F, K,    // x row-major [M,K]
            &beta,
            output.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else if (x.dtype == DType::BF16 && W.dtype == DType::BF16) {
        // BF16 GEMM with F32 accumulation
        CHECK_CUBLAS(cublasGemmEx(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            W.data, CUDA_R_16BF, K,
            x.data, CUDA_R_16BF, K,
            &beta,
            output.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else if (x.dtype == DType::FP16 && W.dtype == DType::FP16) {
        // FP16 GEMM with F32 accumulation
        CHECK_CUBLAS(cublasGemmEx(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            W.data, CUDA_R_16F, K,
            x.data, CUDA_R_16F, K,
            &beta,
            output.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else if (x.dtype == DType::F32 && W.dtype == DType::BF16) {
        // Mixed: F32 input, BF16 weight — convert W to F32 for full precision
        // (BF16×BF16 GEMM loses accuracy for large K even with CUBLAS_COMPUTE_32F)
        Tensor W_f32 = Tensor::alloc_shape(W.ndim, W.shape, DType::F32, true);
        bf16_to_f32_cuda(W.bf16(), W_f32.f32(), W.numel());
        CHECK_CUBLAS(cublasGemmEx(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            W_f32.data, CUDA_R_32F, K,
            x.data, CUDA_R_32F, K,
            &beta,
            output.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else if (x.dtype == DType::F32 && W.dtype == DType::FP16) {
        // Mixed: F32 input, FP16 weight — convert W to F32 for full precision
        Tensor W_f32 = Tensor::alloc_shape(W.ndim, W.shape, DType::F32, true);
        fp16_to_f32_cuda(W.fp16(), W_f32.f32(), W.numel());
        CHECK_CUBLAS(cublasGemmEx(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            W_f32.data, CUDA_R_32F, K,
            x.data, CUDA_R_32F, K,
            &beta,
            output.data, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else {
        fprintf(stderr, "Unsupported dtype combo in linear: x=%s W=%s\n", dtype_name(x.dtype), dtype_name(W.dtype));
        exit(1);
    }
}

// Matmul: C = A @ B
// A: [M, K], B: [K, N], C: [M, N], all F32
static void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    assert(A.on_gpu && B.on_gpu && C.on_gpu);
    assert(A.dtype == DType::F32 && B.dtype == DType::F32 && C.dtype == DType::F32);

    int64_t M = A.numel() / A.shape[A.ndim - 1];
    int64_t K = A.shape[A.ndim - 1];
    int64_t N = B.shape[B.ndim - 1];

    float alpha = 1.0f, beta = 0.0f;

    // cuBLAS column-major: C^T = B^T @ A^T
    // B row-major [K,N] → cuBLAS sees [N,K] = B^T already
    // A row-major [M,K] → cuBLAS sees [K,M] = A^T already
    CHECK_CUBLAS(cublasGemmEx(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.data, CUDA_R_32F, N,
        A.data, CUDA_R_32F, K,
        &beta,
        C.data, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Batched matmul: C[b] = A[b] @ B[b]
// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N], all F32
static void batched_matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    assert(A.on_gpu && B.on_gpu && C.on_gpu);
    assert(A.dtype == DType::F32 && B.dtype == DType::F32 && C.dtype == DType::F32);
    assert(A.ndim == 3 && B.ndim == 3 && C.ndim == 3);

    int64_t batch = A.shape[0];
    int64_t M = A.shape[1];
    int64_t K = A.shape[2];
    int64_t N = B.shape[2];
    assert(B.shape[0] == batch && B.shape[1] == K);
    assert(C.shape[0] == batch && C.shape[1] == M && C.shape[2] == N);

    float alpha = 1.0f, beta = 0.0f;
    int64_t strideA = M * K;
    int64_t strideB = K * N;
    int64_t strideC = M * N;

    CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.data, CUDA_R_32F, N, strideB,
        A.data, CUDA_R_32F, K, strideA,
        &beta,
        C.data, CUDA_R_32F, N, strideC,
        batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
