#pragma once
#include "tensor.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)err); \
        exit(1); \
    } \
} while(0)

static cublasHandle_t g_cublas = nullptr;
static cublasLtHandle_t g_cublaslt = nullptr;

static void init_cublas() {
    if (!g_cublas) CHECK_CUBLAS(cublasCreate(&g_cublas));
    CHECK_CUBLAS(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));
    if (!g_cublaslt) CHECK_CUBLAS(cublasLtCreate(&g_cublaslt));
}

// cublasLt linear with fused bias epilogue
// Row-major: output = x @ W^T + bias
// x: [M, K], W: [N, K], bias: [N] or null, output: [M, N]
// All with F32 accumulation. Bias is fused into the GEMM epilogue (zero extra cost).
static void linear_lt(int64_t M, int64_t K, int64_t N,
                       const void* x_data, cudaDataType_t x_type,
                       const void* w_data, cudaDataType_t w_type,
                       const void* bias_data,
                       void* out_data) {
    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmulDesc_t matmul_desc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Set transpose ops for row-major: C = x @ W^T
    // cuBLAS col-major: C'^[N,M] = W[N,K] @ x'^[K,M]
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set bias epilogue if bias is provided
    if (bias_data) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_data, sizeof(bias_data)));
        cudaDataType_t bias_type = CUDA_R_32F;
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
    }

    // Matrix layouts (column-major perspective)
    // A = W row-major [N,K] → col-major [K,N], ld=K (transposed to [N,K])
    // B = x row-major [M,K] → col-major [K,M], ld=K
    // C = output row-major [M,N] → col-major [N,M], ld=N
    cublasLtMatrixLayout_t layout_a, layout_b, layout_c;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_a, w_type, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_b, x_type, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_32F, N, M, N));

    CHECK_CUBLAS(cublasLtMatmul(g_cublaslt,
        matmul_desc,
        &alpha,
        w_data, layout_a,     // A = W
        x_data, layout_b,     // B = x
        &beta,
        out_data, layout_c,   // C = output
        out_data, layout_c,   // D = output (in-place)
        nullptr,              // algo (nullptr = heuristic)
        nullptr, 0,           // workspace
        0));                  // stream

    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// INT8 x INT8 → INT32 GEMM via cublasGemmEx with INT8 tensor cores
// Y_int32[M,N] = X_int8[M,K] @ W_int8[N,K]^T
static void linear_int8_gemm(int64_t M, int64_t K, int64_t N,
                               const int8_t* x_int8, const int8_t* w_int8,
                               int32_t* Y_int32) {
    int32_t alpha = 1, beta = 0;
    CHECK_CUBLAS(cublasGemmEx(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        w_int8, CUDA_R_8I, (int)K,
        x_int8, CUDA_R_8I, (int)K,
        &beta,
        Y_int32, CUDA_R_32I, (int)N,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
}

// Linear: output = x @ W^T + bias
// x: [M, K], W: [N, K], bias: [N] or null, output: [M, N]
// All stored row-major. Uses F32 accumulation.
// Bias is fused into the GEMM epilogue via cublasLt (zero extra kernel launches).
// INT8 weights: per-channel asymmetric uses INT8 GEMM, otherwise fused dequant + BF16 WMMA.
static void linear(const Tensor& x, const Tensor& W, const Tensor* bias, Tensor& output) {
    assert(x.on_gpu && W.on_gpu && output.on_gpu);
    assert(x.ndim >= 1 && W.ndim == 2);

    int64_t M = x.numel() / x.shape[x.ndim - 1]; // batch*seq
    int64_t K = x.shape[x.ndim - 1];
    int64_t N = W.shape[0];
    assert(W.shape[1] == K);
    assert(output.numel() == M * N);

    const void* bias_data = (bias && bias->on_gpu && bias->dtype == DType::F32) ? bias->data : nullptr;

    // INT8 weight path
    if (W.dtype == DType::INT8) {
        assert(W.quant_scales && W.quant_group_size > 0 && "INT8 weight missing quant_scales or group_size");

        // Convert activation to BF16
        const __nv_bfloat16* x_bf16;
        Tensor x_conv;
        if (x.dtype == DType::BF16) {
            x_bf16 = x.bf16();
        } else if (x.dtype == DType::F32) {
            x_conv = Tensor::alloc_shape(x.ndim, x.shape, DType::BF16, true);
            f32_to_bf16_cuda(x.f32(), x_conv.bf16(), x.numel());
            x_bf16 = x_conv.bf16();
        } else if (x.dtype == DType::FP16) {
            x_conv = Tensor::alloc_shape(x.ndim, x.shape, DType::BF16, true);
            fp16_to_bf16_cuda(x.fp16(), x_conv.bf16(), x.numel());
            x_bf16 = x_conv.bf16();
        } else {
            fprintf(stderr, "Unsupported activation dtype for INT8 linear: %s\n", dtype_name(x.dtype));
            exit(1);
        }

        // Per-channel asymmetric: INT8 GEMM path (uses INT8 tensor cores, 2x BF16 throughput)
        // Decomposition: Y = x_scale * w_scale * (X_i8 @ W_i8^T - zp * x_rowsum) + bias
        // Requires K%4==0 for cuBLAS INT8 tensor cores; falls through to WMMA otherwise
        if (W.quant_zero_points && W.quant_group_size >= (int)K && K % 4 == 0) {
            Tensor x_int8 = Tensor::alloc({M, K}, DType::INT8, true);
            Tensor x_scale = Tensor::alloc({M}, DType::F32, true);
            Tensor x_rowsum = Tensor::alloc({M}, DType::F32, true);
            quantize_activations_int8_cuda(x_bf16, x_int8.i8(), x_scale.f32(), x_rowsum.f32(),
                                            (int)M, (int)K);

            // INT8 GEMM → INT32 accumulation buffer
            Tensor y_i32 = Tensor::alloc({M, N}, DType::F32, true);  // 4 bytes/elem, same as INT32
            linear_int8_gemm(M, K, N, x_int8.i8(), W.i8(), (int32_t*)y_i32.data);

            // Dequant: Y_f32 = x_scale * w_scale * (Y_i32 - zp * x_rowsum) + bias
            int8_gemm_dequant_cuda((const int32_t*)y_i32.data, output.f32(),
                                    x_scale.f32(),
                                    W.quant_scales, W.quant_scales_dtype,
                                    (const int8_t*)W.quant_zero_points, x_rowsum.f32(),
                                    (const float*)bias_data, (int)M, (int)N);
            return;
        }

        // Fused INT8 dequant + BF16 WMMA GEMM (per-group or symmetric per-channel)
        fused_dequant_gemm_cuda(x_bf16, W.i8(), W.quant_scales, W.quant_scales_dtype,
                                (const int8_t*)W.quant_zero_points,
                                (const float*)bias_data, output.f32(),
                                (int)M, (int)N, (int)K, W.quant_group_size);
        return;
    }

    if (x.dtype == DType::F32 && W.dtype == DType::F32) {
        linear_lt(M, K, N, x.data, CUDA_R_32F, W.data, CUDA_R_32F, bias_data, output.data);
    } else if (x.dtype == DType::BF16 && W.dtype == DType::BF16) {
        linear_lt(M, K, N, x.data, CUDA_R_16BF, W.data, CUDA_R_16BF, bias_data, output.data);
    } else if (x.dtype == DType::FP16 && W.dtype == DType::FP16) {
        linear_lt(M, K, N, x.data, CUDA_R_16F, W.data, CUDA_R_16F, bias_data, output.data);
    } else if (x.dtype == DType::F32 && W.dtype == DType::BF16) {
        // Mixed: F32 input, BF16 weight — convert x to BF16 and use BF16 GEMM
        Tensor x_bf16 = Tensor::alloc_shape(x.ndim, x.shape, DType::BF16, true);
        f32_to_bf16_cuda(x.f32(), x_bf16.bf16(), x.numel());
        linear_lt(M, K, N, x_bf16.data, CUDA_R_16BF, W.data, CUDA_R_16BF, bias_data, output.data);
    } else if (x.dtype == DType::F32 && W.dtype == DType::FP16) {
        // Mixed: F32 input, FP16 weight — convert x to FP16 and use FP16 GEMM
        Tensor x_fp16 = Tensor::alloc_shape(x.ndim, x.shape, DType::FP16, true);
        f32_to_fp16_cuda(x.f32(), x_fp16.fp16(), x.numel());
        linear_lt(M, K, N, x_fp16.data, CUDA_R_16F, W.data, CUDA_R_16F, bias_data, output.data);
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

// Batched matmul with B transposed: C[b] = A[b] @ B[b]^T
// A: [batch, M, K], B: [batch, N, K], C: [batch, M, N], all F32
static void batched_matmul_Bt(const Tensor& A, const Tensor& B, Tensor& C) {
    assert(A.on_gpu && B.on_gpu && C.on_gpu);
    assert(A.dtype == DType::F32 && B.dtype == DType::F32 && C.dtype == DType::F32);
    assert(A.ndim == 3 && B.ndim == 3 && C.ndim == 3);

    int64_t batch = A.shape[0];
    int64_t M = A.shape[1];
    int64_t K = A.shape[2];
    int64_t N = B.shape[1];
    assert(B.shape[0] == batch && B.shape[2] == K);
    assert(C.shape[0] == batch && C.shape[1] == M && C.shape[2] == N);

    float alpha = 1.0f, beta = 0.0f;
    int64_t strideA = M * K;
    int64_t strideB = N * K;
    int64_t strideC = M * N;

    CHECK_CUBLAS(cublasGemmStridedBatchedEx(g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.data, CUDA_R_32F, K, strideB,
        A.data, CUDA_R_32F, K, strideA,
        &beta,
        C.data, CUDA_R_32F, N, strideC,
        batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
