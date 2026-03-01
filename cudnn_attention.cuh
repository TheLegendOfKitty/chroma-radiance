#pragma once
#include <cudnn_frontend.h>
#include "tensor.cuh"

namespace fe = cudnn_frontend;

#define CHECK_CUDNN(call) do { \
    cudnnStatus_t _s = (call); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudnnGetErrorString(_s)); \
        exit(1); \
    } \
} while(0)

// UIDs for cuDNN tensor mapping
enum CudnnUID : int64_t {
    UID_Q = 1, UID_K = 2, UID_V = 3, UID_O = 4,
    UID_STATS = 5, UID_BIAS = 6
};

// Global cuDNN state
static cudnnHandle_t g_cudnn = nullptr;

// Cached SDPA graph (built once per shape configuration)
struct CudnnSdpaCache {
    std::shared_ptr<fe::graph::Graph> graph;
    void* workspace = nullptr;
    int64_t workspace_size = 0;
    float* stats = nullptr;
    bool has_stats = false;
    int n_head = 0;
    int seq_len = 0;
    int d_head = 0;
    bool has_bias = false;

    void release() {
        graph.reset();
        if (workspace) { cudaFree(workspace); workspace = nullptr; }
        if (stats) { cudaFree(stats); stats = nullptr; }
        n_head = seq_len = d_head = 0;
        has_stats = false;
    }
};

static CudnnSdpaCache g_sdpa_cache;

// Un-permute BF16: [n_head, L, d_head] → [L, n_head, d_head] (= [L, n_head*d_head])
// Then convert BF16 → F32
__global__ void unpermute_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ src,  // [n_head, L, d_head]
    float* __restrict__ dst,                // [L, n_head*d_head]
    int n_head, int L, int d_head) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n_head * L * d_head;
    if (idx >= total) return;

    // src layout: [n_head, L, d_head] contiguous
    int d = (int)(idx % d_head);
    int64_t rem = idx / d_head;
    int l = (int)(rem % L);
    int h = (int)(rem / L);

    // dst layout: [L, n_head, d_head]
    int64_t dst_idx = (int64_t)l * n_head * d_head + h * d_head + d;
    dst[dst_idx] = __bfloat162float(src[idx]);
}

static void unpermute_bf16_to_f32_cuda(const __nv_bfloat16* src, float* dst,
                                        int n_head, int L, int d_head) {
    int64_t total = (int64_t)n_head * L * d_head;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    unpermute_bf16_to_f32_kernel<<<blocks, threads>>>(src, dst, n_head, L, d_head);
}

// ============================================================================
// Fused unpermute BF16 [n_head, L, d_head] → INT8 quantized [L, n_head*d_head]
// Combines unpermute + optional smooth + Hadamard + absmax + quantize in one kernel.
// Each block handles one row (token) of the output [L, hidden_size].
// ============================================================================
extern bool g_hadamard_enabled;

// Device helper: block-diagonal Walsh-Hadamard Transform in smem
__device__ void wht_inplace_smem_attn(float* smem, int dim, int block_size, int num_blocks) {
    float norm = rsqrtf((float)block_size);
    int n_stages = __ffs(block_size) - 1;
    int pairs_per_block = block_size >> 1;
    int total_pairs = num_blocks * pairs_per_block;

    for (int stage = 0; stage < n_stages; stage++) {
        int half = 1 << stage;
        int span = half << 1;
        for (int idx = threadIdx.x; idx < total_pairs; idx += blockDim.x) {
            int block = idx / pairs_per_block;
            int pair = idx % pairs_per_block;
            int base = block * block_size;
            int i = base + (pair / half) * span + (pair % half);
            int j = i + half;
            float a = smem[i], b = smem[j];
            smem[i] = a + b;
            smem[j] = a - b;
        }
        __syncthreads();
    }
    // Normalize
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        smem[i] *= norm;
    __syncthreads();
}

template<bool NEED_ROWSUM, bool HADAMARD>
__global__ void unpermute_bf16_quantize_int8_kernel(
    const __nv_bfloat16* __restrict__ src,  // [n_head, L_stride, d_head]
    int8_t* __restrict__ X_int8,            // [num_tokens, hidden_size]
    float* __restrict__ x_scale,            // [num_tokens]
    float* __restrict__ x_rowsum,           // [num_tokens] or nullptr
    const float* __restrict__ smooth,       // [hidden_size] or nullptr
    int n_head, int L_stride, int d_head,
    int num_tokens, int token_offset)
{
    int token = blockIdx.x;
    if (token >= num_tokens) return;

    int hidden_size = n_head * d_head;
    int8_t* out_row = X_int8 + (int64_t)token * hidden_size;
    int src_token = token + token_offset;  // offset into BF16 [n_head, L_stride, d_head]

    extern __shared__ char uq_smem[];

    if constexpr (HADAMARD) {
        // Hadamard path: gather → smooth → cache in smem → WHT → absmax → quantize
        float* cached = reinterpret_cast<float*>(uq_smem);
        float* sdata = cached + hidden_size;

        // Gather from BF16 [n_head, L_stride, d_head] → F32 [hidden_size] with unpermute
        for (int k = threadIdx.x; k < hidden_size; k += blockDim.x) {
            int h = k / d_head;
            int d = k % d_head;
            int64_t src_idx = (int64_t)h * L_stride * d_head + (int64_t)src_token * d_head + d;
            float val = __bfloat162float(src[src_idx]);
            if (smooth) val /= smooth[k];
            cached[k] = val;
        }
        __syncthreads();

        // Apply WHT in-place in smem
        int block_size = hidden_size & (-hidden_size);
        int num_blocks_wht = hidden_size / block_size;
        wht_inplace_smem_attn(cached, hidden_size, block_size, num_blocks_wht);

        // Find absmax
        float local_max = 0.0f;
        for (int k = threadIdx.x; k < hidden_size; k += blockDim.x)
            local_max = fmaxf(local_max, fabsf(cached[k]));
        sdata[threadIdx.x] = local_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < (unsigned)s)
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
            __syncthreads();
        }
        float absmax = sdata[0];
        float inv_scale = (absmax > 1e-10f) ? 127.0f / absmax : 0.0f;
        if (threadIdx.x == 0) x_scale[token] = absmax / 127.0f;
        __syncthreads();

        // Quantize from smem cache (packed INT8 writes)
        int local_sum = 0;
        int k4_limit = hidden_size & ~3;
        for (int k = threadIdx.x * 4; k < k4_limit; k += blockDim.x * 4) {
            int q0 = max(-128, min(127, __float2int_rn(cached[k]     * inv_scale)));
            int q1 = max(-128, min(127, __float2int_rn(cached[k + 1] * inv_scale)));
            int q2 = max(-128, min(127, __float2int_rn(cached[k + 2] * inv_scale)));
            int q3 = max(-128, min(127, __float2int_rn(cached[k + 3] * inv_scale)));
            *reinterpret_cast<int32_t*>(out_row + k) =
                (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
            if constexpr (NEED_ROWSUM) local_sum += q0 + q1 + q2 + q3;
        }

        if constexpr (NEED_ROWSUM) {
            int* idata = reinterpret_cast<int*>(sdata);
            idata[threadIdx.x] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < (unsigned)s)
                    idata[threadIdx.x] += idata[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) x_rowsum[token] = (float)idata[0];
        }
    } else {
        // Non-Hadamard: 2-pass (absmax, quantize) with gather from BF16
        float* sdata = reinterpret_cast<float*>(uq_smem);

        // Pass 1: gather from BF16 + find absmax
        float local_max = 0.0f;
        int k4_limit = hidden_size & ~3;
        for (int k = threadIdx.x; k < hidden_size; k += blockDim.x) {
            int h = k / d_head;
            int d = k % d_head;
            int64_t src_idx = (int64_t)h * L_stride * d_head + (int64_t)src_token * d_head + d;
            float val = __bfloat162float(src[src_idx]);
            if (smooth) val /= smooth[k];
            local_max = fmaxf(local_max, fabsf(val));
        }
        sdata[threadIdx.x] = local_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < (unsigned)s)
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
            __syncthreads();
        }
        float absmax = sdata[0];
        float inv_scale = (absmax > 1e-10f) ? 127.0f / absmax : 0.0f;
        if (threadIdx.x == 0) x_scale[token] = absmax / 127.0f;
        __syncthreads();

        // Pass 2: gather from BF16 + quantize
        int local_sum = 0;
        for (int k = threadIdx.x * 4; k < k4_limit; k += blockDim.x * 4) {
            int q[4];
            for (int j = 0; j < 4; j++) {
                int kj = k + j;
                int h = kj / d_head;
                int d = kj % d_head;
                int64_t src_idx = (int64_t)h * L_stride * d_head + (int64_t)src_token * d_head + d;
                float val = __bfloat162float(src[src_idx]);
                if (smooth) val /= smooth[kj];
                q[j] = max(-128, min(127, __float2int_rn(val * inv_scale)));
                if constexpr (NEED_ROWSUM) local_sum += q[j];
            }
            *reinterpret_cast<int32_t*>(out_row + k) =
                (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | ((q[3] & 0xFF) << 24);
        }

        if constexpr (NEED_ROWSUM) {
            int* idata = reinterpret_cast<int*>(sdata);
            idata[threadIdx.x] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < (unsigned)s)
                    idata[threadIdx.x] += idata[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) x_rowsum[token] = (float)idata[0];
        }
    }
}

static void unpermute_bf16_quantize_int8_cuda(
    const __nv_bfloat16* src,  // [n_head, L_stride, d_head]
    int8_t* X_int8,            // [num_tokens, hidden_size]
    float* x_scale,            // [num_tokens]
    float* x_rowsum,           // [num_tokens] or nullptr
    const float* smooth,       // [hidden_size] or nullptr
    int n_head, int L_stride, int d_head,
    int num_tokens, int token_offset,
    bool need_rowsum, bool hadamard)
{
    int hidden_size = n_head * d_head;
    int threads = 256;
    if (hadamard) {
        size_t smem_bytes = (hidden_size + threads) * sizeof(float);
        if (need_rowsum)
            unpermute_bf16_quantize_int8_kernel<true, true><<<num_tokens, threads, smem_bytes>>>(
                src, X_int8, x_scale, x_rowsum, smooth, n_head, L_stride, d_head, num_tokens, token_offset);
        else
            unpermute_bf16_quantize_int8_kernel<false, true><<<num_tokens, threads, smem_bytes>>>(
                src, X_int8, x_scale, x_rowsum, smooth, n_head, L_stride, d_head, num_tokens, token_offset);
    } else {
        size_t smem_bytes = threads * sizeof(float);
        if (need_rowsum)
            unpermute_bf16_quantize_int8_kernel<true, false><<<num_tokens, threads, smem_bytes>>>(
                src, X_int8, x_scale, x_rowsum, smooth, n_head, L_stride, d_head, num_tokens, token_offset);
        else
            unpermute_bf16_quantize_int8_kernel<false, false><<<num_tokens, threads, smem_bytes>>>(
                src, X_int8, x_scale, x_rowsum, smooth, n_head, L_stride, d_head, num_tokens, token_offset);
    }
}

static void init_cudnn() {
    if (!g_cudnn) CHECK_CUDNN(cudnnCreate(&g_cudnn));
}

static void build_sdpa_graph(int n_head, int L, int d_head, bool has_bias) {
    g_sdpa_cache.release();

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t b = 1, h = n_head, s = L, d = d_head;
    float attn_scale = 1.0f / sqrtf((float)d_head);

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q").set_uid(UID_Q)
        .set_dim({b, h, s, d})
        .set_stride({h*s*d, s*d, d, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K").set_uid(UID_K)
        .set_dim({b, h, s, d})
        .set_stride({h*s*d, s*d, d, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V").set_uid(UID_V)
        .set_dim({b, h, s, d})
        .set_stride({h*s*d, s*d, d, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes()
        .set_name("flash_attention")
        .set_attn_scale(attn_scale);

    // Attention bias = our mask [L] broadcast as [1, 1, 1, L]
    if (has_bias) {
        auto bias = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("bias").set_uid(UID_BIAS)
            .set_dim({b, 1, 1, s})
            .set_stride({s, s, s, 1})
            .set_data_type(fe::DataType_t::FLOAT));
        sdpa_options.set_bias(bias);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Standard contiguous output: [b, h, s, d] BF16
    O->set_output(true)
        .set_dim({b, h, s, d})
        .set_stride({h*s*d, s*d, d, 1})
        .set_uid(UID_O);

    bool stats_generated = (Stats != nullptr);
    if (stats_generated) {
        Stats->set_output(true)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_uid(UID_STATS);
    }

    // Build the graph
    auto status = graph->validate();
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA validate failed: %s\n", status.get_message().c_str());
        exit(1);
    }

    status = graph->build_operation_graph(g_cudnn);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA build_operation_graph failed: %s\n", status.get_message().c_str());
        exit(1);
    }

    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA create_execution_plans failed: %s\n", status.get_message().c_str());
        exit(1);
    }

    status = graph->check_support(g_cudnn);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA check_support failed: %s\n", status.get_message().c_str());
        fprintf(stderr, "This GPU/config may not support cuDNN flash attention.\n");
        exit(1);
    }

    status = graph->build_plans(g_cudnn, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA build_plans failed: %s\n", status.get_message().c_str());
        exit(1);
    }

    // Allocate workspace
    int64_t ws_size = 0;
    graph->get_workspace_size(ws_size);
    void* ws = nullptr;
    if (ws_size > 0) {
        CHECK_CUDA(cudaMalloc(&ws, ws_size));
    }

    // Allocate stats buffer if needed
    float* stats_buf = nullptr;
    if (stats_generated) {
        CHECK_CUDA(cudaMalloc(&stats_buf, b * h * s * sizeof(float)));
    }

    g_sdpa_cache.graph = graph;
    g_sdpa_cache.workspace = ws;
    g_sdpa_cache.workspace_size = ws_size;
    g_sdpa_cache.stats = stats_buf;
    g_sdpa_cache.has_stats = stats_generated;
    g_sdpa_cache.n_head = n_head;
    g_sdpa_cache.seq_len = L;
    g_sdpa_cache.d_head = d_head;
    g_sdpa_cache.has_bias = has_bias;

    printf("cuDNN SDPA graph built: n_head=%d, L=%d, d_head=%d, bias=%d, stats=%d, workspace=%.1f KB\n",
           n_head, L, d_head, has_bias, stats_generated, ws_size / 1024.0);
}

// Execute cuDNN SDPA
// Q, K, V: [n_head, L, d_head] BF16 (contiguous, batch=1 implicit)
// bias: [L] F32 (attention mask, broadcast as [1,1,1,L]) or nullptr
// O: [n_head, L, d_head] BF16 (contiguous, standard layout)
static void cudnn_sdpa_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const float* bias, __nv_bfloat16* O,
    int n_head, int L, int d_head) {

    bool need_bias = (bias != nullptr);

    // Rebuild graph if shape or bias presence changed
    if (!g_sdpa_cache.graph ||
        g_sdpa_cache.n_head != n_head ||
        g_sdpa_cache.seq_len != L ||
        g_sdpa_cache.d_head != d_head ||
        g_sdpa_cache.has_bias != need_bias) {
        build_sdpa_graph(n_head, L, d_head, need_bias);
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {UID_Q, (void*)Q},
        {UID_K, (void*)K},
        {UID_V, (void*)V},
        {UID_O, (void*)O}
    };
    if (g_sdpa_cache.has_stats) {
        variant_pack[UID_STATS] = (void*)g_sdpa_cache.stats;
    }
    if (need_bias) {
        variant_pack[UID_BIAS] = (void*)bias;
    }

    auto status = g_sdpa_cache.graph->execute(g_cudnn, variant_pack, g_sdpa_cache.workspace);
    if (!status.is_good()) {
        fprintf(stderr, "cuDNN SDPA execute failed: %s\n", status.get_message().c_str());
        exit(1);
    }
}

// Attention with RoPE using cuDNN SDPA
// Q, K, V: [L, n_head, d_head] F32
// pe: [L, d_head/2 * 4] F32
// output: [L, n_head * d_head] F32
static Tensor cudnn_attention_with_rope(const Tensor& Q, const Tensor& K, const Tensor& V,
                                         const Tensor& pe, int n_head, int d_head,
                                         const float* attn_mask = nullptr) {
    assert(Q.on_gpu && K.on_gpu && V.on_gpu && pe.on_gpu);
    int64_t L = Q.shape[0];

    // 1. Fused permute+RoPE+BF16: [L, n_head, d_head] → [n_head, L, d_head] BF16
    Tensor Q_bf16 = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::BF16, true);
    Tensor K_bf16 = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::BF16, true);
    apply_rope_fused_bf16_cuda(Q.f32(), pe.f32(), Q_bf16.bf16(), n_head, L, d_head);
    apply_rope_fused_bf16_cuda(K.f32(), pe.f32(), K_bf16.bf16(), n_head, L, d_head);
    // No Q pre-scaling — cuDNN SDPA handles attn_scale internally

    // 2. Permute + convert V: [L, n_head, d_head] F32 → [n_head, L, d_head] BF16
    Tensor V_bf16 = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::BF16, true);
    permute_and_convert_to_bf16_cuda(V.f32(), V_bf16.bf16(), n_head, (int)L, d_head);

    // 3. cuDNN SDPA → O_bf16 in [n_head, L, d_head] contiguous layout
    Tensor O_bf16 = Tensor::alloc({(int64_t)n_head, L, (int64_t)d_head}, DType::BF16, true);
    cudnn_sdpa_forward(Q_bf16.bf16(), K_bf16.bf16(), V_bf16.bf16(),
                       attn_mask, O_bf16.bf16(), n_head, (int)L, d_head);

    // 4. Un-permute [n_head, L, d_head] BF16 → [L, n_head*d_head] F32
    Tensor output = Tensor::alloc({L, (int64_t)(n_head * d_head)}, DType::F32, true);
    unpermute_bf16_to_f32_cuda(O_bf16.bf16(), output.f32(), n_head, (int)L, d_head);

    return output;
}
