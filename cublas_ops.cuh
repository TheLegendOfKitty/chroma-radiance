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

// cublasLt workspace: enables better algorithm selection (tiled algorithms)
static void* g_cublas_workspace = nullptr;
static size_t g_cublas_workspace_size = 32 * 1024 * 1024;  // 32MB

// Scratch buffer for INT8→BF16 weight dequantization (reused across linear calls)
static __nv_bfloat16* g_dequant_scratch = nullptr;
static size_t g_dequant_scratch_elems = 0;

// Persistent scratch buffers for INT8 activation quantization (reused across linear calls)
static int8_t* g_act_int8_scratch = nullptr;
static size_t g_act_int8_scratch_bytes = 0;
static float* g_act_scale_scratch = nullptr;
static size_t g_act_scale_scratch_elems = 0;
static float* g_act_rowsum_scratch = nullptr;
static size_t g_act_rowsum_scratch_elems = 0;

// Algorithm cache: avoid repeated heuristic lookups for the same (M,N,K,types) combo
struct LtAlgoKey {
    int64_t M, K, N;
    cudaDataType_t x_type, w_type;
    bool has_bias;
    bool operator==(const LtAlgoKey& o) const {
        return M == o.M && K == o.K && N == o.N &&
               x_type == o.x_type && w_type == o.w_type && has_bias == o.has_bias;
    }
};
struct LtAlgoKeyHash {
    size_t operator()(const LtAlgoKey& k) const {
        size_t h = std::hash<int64_t>{}(k.M);
        h ^= std::hash<int64_t>{}(k.K) * 2654435761ULL;
        h ^= std::hash<int64_t>{}(k.N) * 40503ULL;
        h ^= (size_t)k.x_type * 97;
        h ^= (size_t)k.w_type * 31;
        h ^= (size_t)k.has_bias;
        return h;
    }
};
struct LtAlgoCached {
    cublasLtMatmulAlgo_t algo;
    bool valid;
};
static std::unordered_map<LtAlgoKey, LtAlgoCached, LtAlgoKeyHash> g_algo_cache;

struct CalibrationState {
    bool active = false;
    std::unordered_map<std::string, float*> act_absmax;  // weight_name → [K] GPU float accumulator
    std::unordered_map<std::string, int> k_dims;
};
static CalibrationState g_calib;

static void init_cublas() {
    if (!g_cublas) CHECK_CUBLAS(cublasCreate(&g_cublas));
    CHECK_CUBLAS(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));
    if (!g_cublaslt) CHECK_CUBLAS(cublasLtCreate(&g_cublaslt));
    if (!g_cublas_workspace) {
        CHECK_CUDA(cudaMalloc(&g_cublas_workspace, g_cublas_workspace_size));
    }
}

// Ensure scratch buffer can hold at least `elems` BF16 elements
static __nv_bfloat16* ensure_dequant_scratch(int64_t elems) {
    if (elems <= (int64_t)g_dequant_scratch_elems) return g_dequant_scratch;
    if (g_dequant_scratch) CHECK_CUDA(cudaFree(g_dequant_scratch));
    CHECK_CUDA(cudaMalloc(&g_dequant_scratch, elems * sizeof(__nv_bfloat16)));
    g_dequant_scratch_elems = (size_t)elems;
    return g_dequant_scratch;
}

static int8_t* ensure_act_int8_scratch(int64_t bytes) {
    if (bytes <= (int64_t)g_act_int8_scratch_bytes) return g_act_int8_scratch;
    if (g_act_int8_scratch) CHECK_CUDA(cudaFree(g_act_int8_scratch));
    CHECK_CUDA(cudaMalloc(&g_act_int8_scratch, bytes));
    g_act_int8_scratch_bytes = (size_t)bytes;
    return g_act_int8_scratch;
}

static float* ensure_act_scale_scratch(int64_t elems) {
    if (elems <= (int64_t)g_act_scale_scratch_elems) return g_act_scale_scratch;
    if (g_act_scale_scratch) CHECK_CUDA(cudaFree(g_act_scale_scratch));
    CHECK_CUDA(cudaMalloc(&g_act_scale_scratch, elems * sizeof(float)));
    g_act_scale_scratch_elems = (size_t)elems;
    return g_act_scale_scratch;
}

static float* ensure_act_rowsum_scratch(int64_t elems) {
    if (elems <= (int64_t)g_act_rowsum_scratch_elems) return g_act_rowsum_scratch;
    if (g_act_rowsum_scratch) CHECK_CUDA(cudaFree(g_act_rowsum_scratch));
    CHECK_CUDA(cudaMalloc(&g_act_rowsum_scratch, elems * sizeof(float)));
    g_act_rowsum_scratch_elems = (size_t)elems;
    return g_act_rowsum_scratch;
}

// Pre-quantized activation state for INT8 TC GEMM path.
struct QuantizedAct {
    int8_t* x_int8 = nullptr;
    float* x_scale = nullptr;
    float* x_rowsum = nullptr;
    int64_t M = 0, K = 0;
};

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

    // Cached heuristic algorithm selection: search once per unique (M,N,K,types) combo
    LtAlgoKey algo_key = {M, K, N, x_type, w_type, bias_data != nullptr};
    auto cache_it = g_algo_cache.find(algo_key);
    if (cache_it == g_algo_cache.end()) {
        cublasLtMatmulPreference_t pref;
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &g_cublas_workspace_size, sizeof(g_cublas_workspace_size)));

        cublasLtMatmulHeuristicResult_t heuristic;
        int n_results = 0;
        cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(g_cublaslt,
            matmul_desc, layout_a, layout_b, layout_c, layout_c,
            pref, 1, &heuristic, &n_results);

        LtAlgoCached cached;
        cached.valid = (heur_status == CUBLAS_STATUS_SUCCESS && n_results > 0);
        if (cached.valid) cached.algo = heuristic.algo;
        cache_it = g_algo_cache.emplace(algo_key, cached).first;

        cublasLtMatmulPreferenceDestroy(pref);
    }

    CHECK_CUBLAS(cublasLtMatmul(g_cublaslt,
        matmul_desc,
        &alpha,
        w_data, layout_a,     // A = W
        x_data, layout_b,     // B = x
        &beta,
        out_data, layout_c,   // C = output
        out_data, layout_c,   // D = output (in-place)
        cache_it->second.valid ? &cache_it->second.algo : nullptr,
        g_cublas_workspace, g_cublas_workspace_size,
        0));                  // stream
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// INT8 x INT8 → INT32 GEMM via cublasLt with INT8 tensor cores
// Y_int32[M,N] = X_int8[M,K] @ W_int8[N,K]^T
// Uses 32MB workspace + heuristic algorithm selection + caching (same as BF16 path)
static void linear_int8_gemm(int64_t M, int64_t K, int64_t N,
                               const int8_t* x_int8, const int8_t* w_int8,
                               int32_t* Y_int32) {
    int32_t alpha = 1, beta = 0;

    cublasLtMatmulDesc_t matmul_desc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    // Row-major: C = X @ W^T
    // Col-major perspective: C'[N,M] = W[N,K] @ X'[K,M]
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Matrix layouts (column-major perspective):
    // A = W row-major [N,K] → col-major [K,N], ld=K (transposed to [N,K])
    // B = X row-major [M,K] → col-major [K,M], ld=K
    // C = Y row-major [M,N] → col-major [N,M], ld=N
    cublasLtMatrixLayout_t layout_a, layout_b, layout_c;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_8I, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_8I, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_32I, N, M, N));

    // Cached heuristic algorithm selection
    LtAlgoKey algo_key = {M, K, N, CUDA_R_8I, CUDA_R_8I, false};
    auto cache_it = g_algo_cache.find(algo_key);
    if (cache_it == g_algo_cache.end()) {
        cublasLtMatmulPreference_t pref;
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &g_cublas_workspace_size, sizeof(g_cublas_workspace_size)));

        cublasLtMatmulHeuristicResult_t heuristic;
        int n_results = 0;
        cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(g_cublaslt,
            matmul_desc, layout_a, layout_b, layout_c, layout_c,
            pref, 1, &heuristic, &n_results);

        LtAlgoCached cached;
        cached.valid = (heur_status == CUBLAS_STATUS_SUCCESS && n_results > 0);
        if (cached.valid) cached.algo = heuristic.algo;
        cache_it = g_algo_cache.emplace(algo_key, cached).first;

        cublasLtMatmulPreferenceDestroy(pref);
    }

    CHECK_CUBLAS(cublasLtMatmul(g_cublaslt,
        matmul_desc,
        &alpha,
        w_int8, layout_a,     // A = W
        x_int8, layout_b,     // B = X
        &beta,
        Y_int32, layout_c,    // C = output
        Y_int32, layout_c,    // D = output (in-place)
        cache_it->second.valid ? &cache_it->second.algo : nullptr,
        g_cublas_workspace, g_cublas_workspace_size,
        0));                   // stream

    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulDescDestroy(matmul_desc);
}

// Linear: output = x @ W^T + bias
// x: [M, K], W: [N, K], bias: [N] or null, output: [M, N]
// All stored row-major. Uses F32 accumulation.
// Bias is fused into the GEMM epilogue via cublasLt (zero extra kernel launches).
// INT8 weights: per-channel uses INT8 GEMM (2x throughput), per-group uses dequant + cuBLAS BF16.
static void linear(const Tensor& x, const Tensor& W, const Tensor* bias, Tensor& output) {
    assert(x.on_gpu && W.on_gpu && output.on_gpu);
    assert(x.ndim >= 1 && W.ndim == 2);

    int64_t M = x.numel() / x.shape[x.ndim - 1];
    int64_t K = x.shape[x.ndim - 1];
    int64_t N = W.shape[0];
    assert(W.shape[1] == K);
    assert(output.numel() == M * N);

    const void* bias_data = (bias && bias->on_gpu && bias->dtype == DType::F32) ? bias->data : nullptr;

    // Calibration: record per-K-channel activation absmax
    if (g_calib.active && W.calib_name) {
        std::string name(W.calib_name);
        if (g_calib.act_absmax.find(name) == g_calib.act_absmax.end()) {
            float* buf;
            CHECK_CUDA(cudaMalloc(&buf, K * sizeof(float)));
            CHECK_CUDA(cudaMemset(buf, 0, K * sizeof(float)));
            g_calib.act_absmax[name] = buf;
            g_calib.k_dims[name] = (int)K;
        }
        const __nv_bfloat16* x_bf16_cal = nullptr;
        Tensor x_conv_cal;
        if (x.dtype == DType::BF16) {
            x_bf16_cal = x.bf16();
        } else if (x.dtype == DType::F32) {
            x_conv_cal = Tensor::alloc_shape(x.ndim, x.shape, DType::BF16, true);
            f32_to_bf16_cuda(x.f32(), x_conv_cal.bf16(), x.numel());
            x_bf16_cal = x_conv_cal.bf16();
        }
        if (x_bf16_cal) {
            calibrate_act_absmax_cuda(x_bf16_cal, g_calib.act_absmax[name], (int)M, (int)K);
        }
    }

    // INT8 weight path
    if (W.dtype == DType::INT8) {
        assert(W.quant_scales && W.quant_group_size > 0 && "INT8 weight missing quant_scales or group_size");

        // Per-channel INT8 GEMM path (uses INT8 tensor cores, 2x BF16 throughput)
        // Quantizes activations directly from native dtype (F32/BF16) — no intermediate conversion.
        // Decomposition: Y = x_scale * w_scale * (X_i8 @ W_i8^T - zp * x_rowsum) + bias
        // In-place: GEMM writes INT32 to output buffer, dequant overwrites with F32.
        // Uses persistent scratch buffers to avoid per-call pool alloc/free overhead.
        if (W.quant_group_size >= (int)K && K % 4 == 0) {
            bool need_rowsum = (W.quant_zero_points != nullptr);
            int8_t* x_i8 = ensure_act_int8_scratch(M * K);
            float* x_sc = ensure_act_scale_scratch(M);
            float* x_rs = need_rowsum ? ensure_act_rowsum_scratch(M) : nullptr;

            if (x.dtype == DType::F32) {
                quantize_activations_int8_cuda(x.f32(), x_i8, x_sc, x_rs,
                                                (int)M, (int)K, W.quant_smooth, need_rowsum);
            } else if (x.dtype == DType::BF16) {
                quantize_activations_int8_cuda(x.bf16(), x_i8, x_sc, x_rs,
                                                (int)M, (int)K, W.quant_smooth, need_rowsum);
            } else {
                Tensor x_conv = Tensor::alloc_shape(x.ndim, x.shape, DType::BF16, true);
                fp16_to_bf16_cuda(x.fp16(), x_conv.bf16(), x.numel());
                quantize_activations_int8_cuda(x_conv.bf16(), x_i8, x_sc, x_rs,
                                                (int)M, (int)K, W.quant_smooth, need_rowsum);
            }

            linear_int8_gemm(M, K, N, x_i8, W.i8(), (int32_t*)output.data);

            int8_gemm_dequant_cuda((const int32_t*)output.data, output.f32(),
                                    x_sc, W.quant_scales, W.quant_scales_dtype,
                                    (const int8_t*)W.quant_zero_points, x_rs,
                                    (const float*)bias_data, (int)M, (int)N);
            return;
        }

        // Convert activation to BF16 (needed for dequant+cuBLAS BF16 fallback path)
        // When input is F32 and smooth is present, fuse F32→BF16 + smooth division
        // into a single pass (avoids intermediate BF16 buffer + second kernel launch).
        const __nv_bfloat16* x_bf16;
        Tensor x_conv;
        bool smooth_applied = false;
        if (x.dtype == DType::F32 && W.quant_smooth) {
            x_conv = Tensor::alloc_shape(x.ndim, x.shape, DType::BF16, true);
            f32_to_bf16_smooth_cuda(x.f32(), W.quant_smooth, x_conv.bf16(), M, K);
            x_bf16 = x_conv.bf16();
            smooth_applied = true;
        } else if (x.dtype == DType::BF16) {
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

        // SmoothQuant: divide activations by smooth factors before BF16 GEMM
        // Identity: X @ W^T = (X/s) @ (s*W)^T — s*W is baked into quantized weights
        // Skip if already fused into the F32→BF16 conversion above.
        Tensor x_smooth;
        if (W.quant_smooth && !smooth_applied) {
            x_smooth = Tensor::alloc({M, K}, DType::BF16, true);
            smooth_div_bf16_cuda(x_bf16, W.quant_smooth, x_smooth.bf16(), (int)M, (int)K);
            x_bf16 = x_smooth.bf16();
        }

        // Pre-dequant INT8→BF16 into scratch buffer, then cuBLAS BF16 GEMM.
        // The dequant kernel is bandwidth-bound (coalesced reads), and cuBLAS BF16
        // uses cp.async for both operands — beating the fused WMMA kernel which is
        // memory-latency bound on strided W loads (long_scoreboard stalls).
        __nv_bfloat16* w_bf16 = ensure_dequant_scratch(N * K);
        dequant_int8_to_bf16_vec_cuda(W.i8(), W.quant_scales, W.quant_scales_dtype,
                                       (const int8_t*)W.quant_zero_points,
                                       w_bf16, N, K, W.quant_group_size);
        linear_lt(M, K, N, x_bf16, CUDA_R_16BF, w_bf16, CUDA_R_16BF, bias_data, output.data);
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

// Quantize activations into persistent scratch buffers for INT8 TC GEMM path.
// Returns QuantizedAct with pointers valid until next quantize_act call.
static QuantizedAct quantize_act(const Tensor& x, const float* smooth, bool need_rowsum) {
    int64_t M = x.numel() / x.shape[x.ndim - 1];
    int64_t K = x.shape[x.ndim - 1];

    QuantizedAct qa;
    qa.M = M;
    qa.K = K;
    qa.x_int8 = ensure_act_int8_scratch(M * K);
    qa.x_scale = ensure_act_scale_scratch(M);
    qa.x_rowsum = need_rowsum ? ensure_act_rowsum_scratch(M) : nullptr;

    if (x.dtype == DType::F32) {
        quantize_activations_int8_cuda(x.f32(), qa.x_int8, qa.x_scale, qa.x_rowsum,
                                        (int)M, (int)K, smooth, need_rowsum);
    } else if (x.dtype == DType::BF16) {
        quantize_activations_int8_cuda(x.bf16(), qa.x_int8, qa.x_scale, qa.x_rowsum,
                                        (int)M, (int)K, smooth, need_rowsum);
    } else {
        Tensor x_conv = Tensor::alloc_shape(x.ndim, x.shape, DType::BF16, true);
        fp16_to_bf16_cuda(x.fp16(), x_conv.bf16(), x.numel());
        quantize_activations_int8_cuda(x_conv.bf16(), qa.x_int8, qa.x_scale, qa.x_rowsum,
                                        (int)M, (int)K, smooth, need_rowsum);
    }
    return qa;
}

// INT8 TC GEMM + dequant with pre-quantized activations.
static void linear_prequant(const QuantizedAct& qa, const Tensor& W,
                             const Tensor* bias, Tensor& output) {
    int64_t M = qa.M, K = qa.K;
    int64_t N = W.shape[0];
    assert(W.dtype == DType::INT8);
    assert(W.shape[1] == K);

    const void* bias_data = (bias && bias->on_gpu && bias->dtype == DType::F32) ? bias->data : nullptr;

    linear_int8_gemm(M, K, N, qa.x_int8, W.i8(), (int32_t*)output.data);

    bool need_rowsum = (W.quant_zero_points != nullptr);
    int8_gemm_dequant_cuda((const int32_t*)output.data, output.f32(),
                            qa.x_scale, W.quant_scales, W.quant_scales_dtype,
                            (const int8_t*)W.quant_zero_points,
                            need_rowsum ? qa.x_rowsum : nullptr,
                            (const float*)bias_data, (int)M, (int)N);
}
