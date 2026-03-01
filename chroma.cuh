#pragma once
#include "tensor.cuh"
#include "cublas_ops.cuh"
#include "attention.cuh"
#include "rope.cuh"
#include "safetensors.cuh"
#include <string>
#include <vector>
#include <cmath>

// Forward declarations
void layer_norm_cuda(const float* x, float* out, int64_t rows, int64_t C, float eps);
void rms_norm_cuda(const float* x, const float* scale, float* out, int64_t rows, int64_t C, float eps);
void gelu_cuda(const float* x, float* out, int64_t n);
void gelu_to_bf16_cuda(const float* x, __nv_bfloat16* out, int64_t n);
void silu_cuda(const float* x, float* out, int64_t n);
void silu_mul_cuda(const float* a, const float* b, float* out, int64_t n);
void add_cuda(const float* a, const float* b, float* out, int64_t n);
void mul_cuda(const float* a, const float* b, float* out, int64_t n);
void modulate_cuda(const float* x, const float* shift, const float* scale,
                   float* out, int64_t M, int64_t C, float eps);
void modulate_to_bf16_cuda(const float* x, const float* shift, const float* scale,
                            __nv_bfloat16* out, int64_t M, int64_t C, float eps);
void timestep_embedding_cuda(const float* input, float* output,
                             int N, int dim, float time_factor, float max_period);
void l2_norm_cuda(const float* x, float* out, int64_t rows, int64_t C, float eps);
void gated_residual_cuda(const float* x, const float* y, const float* gate,
                         float* out, int64_t M, int64_t C);
void concat_last_dim_cuda(const float* a, const float* b, float* out,
                          int64_t M, int64_t Da, int64_t Db);
void concat_last_dim_to_bf16_cuda(const float* a, const float* b, __nv_bfloat16* out,
                                   int64_t M, int64_t Da, int64_t Db);
void concat_first_dim_cuda(const float* a, const float* b, float* out,
                           int64_t Sa, int64_t Sb, int64_t D);
void copy_cuda(const float* src, float* dst, int64_t n);
void patchify_cuda(const float* input, float* output, int N, int C, int H, int W, int p);
void unpatchify_cuda(const float* input, float* output, int N, int C, int H, int W, int p,
                     int h_patches, int w_patches);
void permute_3d_cuda(const float* input, float* output,
                     int64_t d0, int64_t d1, int64_t d2, int p0, int p1, int p2);
void permute_4d_cuda(const float* input, float* output,
                     int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                     int p0, int p1, int p2, int p3);
void conv2d_3x3_cuda(const float* input, const float* weight, const float* bias,
                     float* output, int N, int C_in, int C_out, int H, int W);
void mul_broadcast_cuda(const float* a, const float* b, float* out, int64_t M, int64_t C);
void broadcast_bias_cuda(const float* bias, float* output, int64_t M, int64_t N);
void softmax_cuda(const float* x, float* out, int64_t rows, int64_t C);
void batched_transpose_2d_cuda(const float* input, float* output,
                                int64_t batch, int64_t rows, int64_t cols);
void scale_cuda(float* x, float scale, int64_t n);
void cfg_blend_cuda(const float* cond, const float* uncond, float* out,
                     float cfg_scale, int64_t n);
void rms_norm_nchw_cuda(const float* input, const float* scale, float* output,
                         int C, int H, int W, float eps);
void slice_columns_cuda(const float* src, float* dst,
                         int64_t L, int64_t total_cols,
                         int64_t col_offset, int64_t slice_cols);
void slice_transpose_cuda(const float* src, float* output,
                            int64_t batch, int64_t total_cols,
                            int64_t col_offset, int64_t rows, int64_t cols);
void modulate_quantize_int8_cuda(const float* x, const float* shift, const float* scale,
                                  const float* smooth,
                                  int8_t* X_int8, float* x_scale, float* x_rowsum,
                                  int M, int C, float eps, bool need_rowsum,
                                  bool hadamard);
void int8_dequant_gelu_quantize_cuda(
    const int32_t* Y_i32, int8_t* X_int8_out,
    float* x_scale_out, float* x_rowsum_out,
    const float* x_scale_prev, const void* w_scale_prev, int w_scale_dtype,
    const int8_t* zp_prev, const float* x_rowsum_prev,
    const float* bias, const float* smooth_next,
    int M, int N, bool need_rowsum);
void int8_gemm_dequant_gelu_cuda(const int32_t* Y_i32, float* Y_out,
                                  const float* x_scale,
                                  const void* w_scale, DType w_scale_dtype,
                                  const int8_t* zp, const float* x_rowsum,
                                  const float* bias, int M, int N);

// ============================================================================
// Helper: Linear layer (x @ W^T + bias)
// ============================================================================
static Tensor linear_forward(const Tensor& x, const Tensor& W, const Tensor* bias) {
    int64_t M = x.numel() / x.shape[x.ndim - 1];
    int64_t N = W.shape[0];
    Tensor out = Tensor::alloc({M, N}, DType::F32, true);
    linear(x, W, bias, out);
    return out;
}

// ============================================================================
// Helper: Load weight with optional INT8 companion scales
// If the tensor is INT8, loads the companion .scale tensor and attaches it.
// Weights stay in original [N,K] row-major layout; the GEMM handles transpose.
// Works transparently for BF16 files (no scales loaded).
// ============================================================================
static Tensor load_weight(const SafetensorsFile& sf, const std::string& name) {
    Tensor w = sf.load_tensor_native(name);
    w.calib_name = strdup(name.c_str());
    if (w.dtype == DType::INT8 && w.ndim == 2) {
        // Load companion scale tensor
        std::string scale_name = name + ".scale";
        Tensor scale_t = sf.load_tensor_native(scale_name);
        assert(scale_t.dtype == DType::F32 || scale_t.dtype == DType::FP16 || scale_t.dtype == DType::BF16);
        w.quant_scales = scale_t.data;
        w.quant_scales_dtype = scale_t.dtype;
        scale_t.owns_data = false;  // arena owns memory
        if (scale_t.ndim == 2) {
            // Per-group: scale shape [N, num_groups]
            int64_t num_groups = scale_t.shape[1];
            w.quant_group_size = (int)((w.shape[1] + num_groups - 1) / num_groups);
        } else {
            // Per-channel (legacy): group_size = K (entire row is one group)
            w.quant_group_size = (int)w.shape[1];
        }
        // Load companion zero_point tensor (asymmetric quantization)
        std::string zp_name = name + ".zp";
        if (sf.has_tensor(zp_name)) {
            Tensor zp_t = sf.load_tensor_native(zp_name);
            assert(zp_t.dtype == DType::INT8);
            w.quant_zero_points = zp_t.data;
            zp_t.owns_data = false;  // arena owns memory
        }
        // Load companion smooth tensor (SmoothQuant channel factors)
        std::string smooth_name = name + ".smooth";
        if (sf.has_tensor(smooth_name)) {
            Tensor smooth_t = sf.load_tensor_native(smooth_name);
            assert(smooth_t.dtype == DType::F32 && smooth_t.ndim == 1);
            w.quant_smooth = (float*)smooth_t.data;
            smooth_t.owns_data = false;  // arena owns memory
        }
    }
    return w;
}

// ============================================================================
// ChromaApproximator
// ============================================================================
struct ChromaApproximator {
    Tensor in_proj_w, in_proj_b;        // [5120, 64], [5120]
    Tensor out_proj_w, out_proj_b;      // [3072, 5120], [3072]
    struct ResLayer {
        Tensor norm_scale;              // [5120]
        Tensor in_w, in_b;             // [5120, 5120], [5120]
        Tensor out_w, out_b;           // [5120, 5120], [5120]
    };
    ResLayer layers[5];

    void load(const SafetensorsFile& sf) {
        std::string p = "distilled_guidance_layer.";
        in_proj_w = load_weight(sf, p + "in_proj.weight");
        in_proj_b = sf.load_tensor(p + "in_proj.bias", DType::F32);
        out_proj_w = load_weight(sf, p + "out_proj.weight");
        out_proj_b = sf.load_tensor(p + "out_proj.bias", DType::F32);

        for (int i = 0; i < 5; i++) {
            std::string lp = p + "layers." + std::to_string(i) + ".";
            std::string np = p + "norms." + std::to_string(i) + ".";
            layers[i].norm_scale = sf.load_tensor(np + "scale", DType::F32);
            layers[i].in_w = load_weight(sf, lp + "in_layer.weight");
            layers[i].in_b = sf.load_tensor(lp + "in_layer.bias", DType::F32);
            layers[i].out_w = load_weight(sf, lp + "out_layer.weight");
            layers[i].out_b = sf.load_tensor(lp + "out_layer.bias", DType::F32);
        }
    }

    void free_weights() {
        in_proj_w.free_data(); in_proj_b.free_data();
        out_proj_w.free_data(); out_proj_b.free_data();
        for (int i = 0; i < 5; i++) {
            layers[i].norm_scale.free_data();
            layers[i].in_w.free_data(); layers[i].in_b.free_data();
            layers[i].out_w.free_data(); layers[i].out_b.free_data();
        }
    }

    void disown() {
        in_proj_w.owns_data = false; in_proj_b.owns_data = false;
        out_proj_w.owns_data = false; out_proj_b.owns_data = false;
        for (int i = 0; i < 5; i++) {
            layers[i].norm_scale.owns_data = false;
            layers[i].in_w.owns_data = false; layers[i].in_b.owns_data = false;
            layers[i].out_w.owns_data = false; layers[i].out_b.owns_data = false;
        }
    }

    // input: [344, 64] → output: [344, 3072]
    Tensor forward(const Tensor& x_in) {
        int64_t N = x_in.shape[0]; // 344
        int64_t inner = 5120;

        // in_proj
        Tensor x = linear_forward(x_in, in_proj_w, &in_proj_b);

        // 5 residual layers: RMSNorm → MLPEmbedder (Linear+SiLU+Linear) + residual
        for (int i = 0; i < 5; i++) {
            Tensor normed = Tensor::alloc({N, inner}, DType::F32, true);
            rms_norm_cuda(x.f32(), layers[i].norm_scale.f32(), normed.f32(), N, inner, 1e-6f);

            // MLPEmbedder: in_layer → SiLU → out_layer
            Tensor h = linear_forward(normed, layers[i].in_w, &layers[i].in_b);
            silu_cuda(h.f32(), h.f32(), N * inner);
            Tensor h2 = linear_forward(h, layers[i].out_w, &layers[i].out_b);

            // Residual
            add_cuda(x.f32(), h2.f32(), x.f32(), N * inner);
        }

        // out_proj
        Tensor out = linear_forward(x, out_proj_w, &out_proj_b);
        return out;
    }
};

// ============================================================================
// Double-stream block weights
// ============================================================================
struct DoubleStreamBlockWeights {
    // Image stream
    Tensor img_qkv_w, img_qkv_b;   // [9216, 3072], [9216]
    Tensor img_q_norm, img_k_norm;  // [128]
    Tensor img_proj_w, img_proj_b;  // [3072, 3072], [3072]
    Tensor img_mlp_0_w, img_mlp_0_b;  // [12288, 3072], [12288]
    Tensor img_mlp_2_w, img_mlp_2_b;  // [3072, 12288], [3072]

    // Text stream
    Tensor txt_qkv_w, txt_qkv_b;   // [9216, 3072], [9216]
    Tensor txt_q_norm, txt_k_norm;  // [128]
    Tensor txt_proj_w, txt_proj_b;  // [3072, 3072], [3072]
    Tensor txt_mlp_0_w, txt_mlp_0_b;  // [12288, 3072], [12288]
    Tensor txt_mlp_2_w, txt_mlp_2_b;  // [3072, 12288], [3072]

    void disown() {
        img_qkv_w.owns_data = false; img_qkv_b.owns_data = false;
        img_q_norm.owns_data = false; img_k_norm.owns_data = false;
        img_proj_w.owns_data = false; img_proj_b.owns_data = false;
        img_mlp_0_w.owns_data = false; img_mlp_0_b.owns_data = false;
        img_mlp_2_w.owns_data = false; img_mlp_2_b.owns_data = false;
        txt_qkv_w.owns_data = false; txt_qkv_b.owns_data = false;
        txt_q_norm.owns_data = false; txt_k_norm.owns_data = false;
        txt_proj_w.owns_data = false; txt_proj_b.owns_data = false;
        txt_mlp_0_w.owns_data = false; txt_mlp_0_b.owns_data = false;
        txt_mlp_2_w.owns_data = false; txt_mlp_2_b.owns_data = false;
    }

    static DoubleStreamBlockWeights load(const SafetensorsFile& sf, int idx) {
        DoubleStreamBlockWeights w;
        std::string p = "double_blocks." + std::to_string(idx) + ".";

        w.img_qkv_w = load_weight(sf, p + "img_attn.qkv.weight");
        w.img_qkv_b = sf.load_tensor(p + "img_attn.qkv.bias", DType::F32);
        w.img_q_norm = sf.load_tensor(p + "img_attn.norm.query_norm.scale", DType::F32);
        w.img_k_norm = sf.load_tensor(p + "img_attn.norm.key_norm.scale", DType::F32);
        w.img_proj_w = load_weight(sf, p + "img_attn.proj.weight");
        w.img_proj_b = sf.load_tensor(p + "img_attn.proj.bias", DType::F32);
        w.img_mlp_0_w = load_weight(sf, p + "img_mlp.0.weight");
        w.img_mlp_0_b = sf.load_tensor(p + "img_mlp.0.bias", DType::F32);
        w.img_mlp_2_w = load_weight(sf, p + "img_mlp.2.weight");
        w.img_mlp_2_b = sf.load_tensor(p + "img_mlp.2.bias", DType::F32);

        w.txt_qkv_w = load_weight(sf, p + "txt_attn.qkv.weight");
        w.txt_qkv_b = sf.load_tensor(p + "txt_attn.qkv.bias", DType::F32);
        w.txt_q_norm = sf.load_tensor(p + "txt_attn.norm.query_norm.scale", DType::F32);
        w.txt_k_norm = sf.load_tensor(p + "txt_attn.norm.key_norm.scale", DType::F32);
        w.txt_proj_w = load_weight(sf, p + "txt_attn.proj.weight");
        w.txt_proj_b = sf.load_tensor(p + "txt_attn.proj.bias", DType::F32);
        w.txt_mlp_0_w = load_weight(sf, p + "txt_mlp.0.weight");
        w.txt_mlp_0_b = sf.load_tensor(p + "txt_mlp.0.bias", DType::F32);
        w.txt_mlp_2_w = load_weight(sf, p + "txt_mlp.2.weight");
        w.txt_mlp_2_b = sf.load_tensor(p + "txt_mlp.2.bias", DType::F32);

        return w;
    }
};

// ============================================================================
// Single-stream block weights
// ============================================================================
struct SingleStreamBlockWeights {
    Tensor linear1_w, linear1_b;   // [21504, 3072], [21504]
    Tensor linear2_w, linear2_b;   // [3072, 15360], [3072]
    Tensor q_norm, k_norm;         // [128]

    void disown() {
        linear1_w.owns_data = false; linear1_b.owns_data = false;
        linear2_w.owns_data = false; linear2_b.owns_data = false;
        q_norm.owns_data = false; k_norm.owns_data = false;
    }

    static SingleStreamBlockWeights load(const SafetensorsFile& sf, int idx) {
        SingleStreamBlockWeights w;
        std::string p = "single_blocks." + std::to_string(idx) + ".";

        w.linear1_w = load_weight(sf, p + "linear1.weight");
        w.linear1_b = sf.load_tensor(p + "linear1.bias", DType::F32);
        w.linear2_w = load_weight(sf, p + "linear2.weight");
        w.linear2_b = sf.load_tensor(p + "linear2.bias", DType::F32);
        w.q_norm = sf.load_tensor(p + "norm.query_norm.scale", DType::F32);
        w.k_norm = sf.load_tensor(p + "norm.key_norm.scale", DType::F32);

        return w;
    }
};

// ============================================================================
// NeRF GLU block weights
// ============================================================================
struct NerfGLUBlockWeights {
    Tensor param_gen_w, param_gen_b;  // [49152, 3072], [49152]
    Tensor norm_scale;                // [64]

    void disown() {
        param_gen_w.owns_data = false; param_gen_b.owns_data = false;
        norm_scale.owns_data = false;
    }

    static NerfGLUBlockWeights load(const SafetensorsFile& sf, int idx) {
        NerfGLUBlockWeights w;
        std::string p = "nerf_blocks." + std::to_string(idx) + ".";
        w.param_gen_w = load_weight(sf, p + "param_generator.weight");
        w.param_gen_b = sf.load_tensor(p + "param_generator.bias", DType::F32);
        w.norm_scale = sf.load_tensor(p + "norm.scale", DType::F32);
        return w;
    }
};

// ============================================================================
// NeRF small weights (always resident)
// ============================================================================
struct NerfSmallWeights {
    Tensor embedder_w, embedder_b;     // [64, 67], [64]
    Tensor final_norm_scale;           // [64]
    Tensor final_conv_w, final_conv_b; // [3, 64, 3, 3], [3]
};

// ============================================================================
// Chroma Radiance Model (all weights pre-loaded on GPU)
// ============================================================================
struct ChromaRadiance {
    static constexpr int hidden_size = 3072;
    static constexpr int num_heads = 24;
    static constexpr int d_head = 128;
    static constexpr int mlp_hidden = 12288;
    static constexpr int depth = 19;
    static constexpr int depth_single = 38;
    static constexpr int patch_size = 16;
    static constexpr int nerf_hidden = 64;
    static constexpr int nerf_depth = 4;
    static constexpr int nerf_mlp_ratio = 4;
    static constexpr int nerf_max_freqs = 8;
    static constexpr int mod_index_length = 344;

    // Cached constant: mod_index_emb [344, 32] F32 (computed once, reused across steps)
    Tensor cached_mod_index_emb;

    // GPU arena: single allocation for all weights (declared first → destroyed last)
    GPUArena weight_arena;

    // Auto-detected from file: true if weights are INT8 quantized
    bool is_int8 = false;

    // Small always-resident weights
    Tensor img_in_patch_w, img_in_patch_b;  // Conv2d(3→3072, 16×16)
    Tensor txt_in_w, txt_in_b;              // Linear(4096→3072)
    NerfSmallWeights nerf_small;

    // ChromaApproximator
    ChromaApproximator approx;

    // Pre-loaded block weights (all resident on GPU via arena)
    std::vector<DoubleStreamBlockWeights> double_block_weights;  // [19]
    std::vector<SingleStreamBlockWeights> single_block_weights;  // [38]
    std::vector<NerfGLUBlockWeights> nerf_block_weights;         // [4]

    void load(const SafetensorsFile& sf) {
        printf("Loading Chroma Radiance model (pre-loading all weights to GPU)...\n");

        // Calculate arena size: sum of all tensor file sizes + alignment padding
        // Weights are BF16 (loaded native), biases/norms are F32 (loaded as-is)
        size_t arena_size = 0;
        for (auto& [name, info] : sf.tensors)
            arena_size += (info.nbytes + 255) & ~(size_t)255;
        arena_size += 16 * 1024 * 1024;  // 16 MB safety margin

        printf("  Allocating %.2f GB GPU arena...\n", arena_size / 1e9);
        if (!weight_arena.init(arena_size)) {
            fprintf(stderr, "Failed to allocate GPU weight arena (%.2f GB)\n", arena_size / 1e9);
            exit(1);
        }
        gpu_pool().arena = &weight_arena;

        // Auto-detect INT8: check dtype of a representative weight tensor
        {
            auto it = sf.tensors.find("double_blocks.0.img_attn.qkv.weight");
            if (it != sf.tensors.end() && it->second.dtype == DType::INT8) {
                is_int8 = true;
                printf("  Detected INT8 quantized weights\n");
            }
        }

        // Detect Hadamard rotation from safetensors metadata
        if (sf.metadata_get("quantization_hadamard") == "true") {
            g_hadamard_enabled = true;
            printf("  Hadamard rotation enabled\n");
        }

        // Small resident weights
        img_in_patch_w = load_weight(sf, "img_in_patch.weight");
        img_in_patch_b = sf.load_tensor("img_in_patch.bias", DType::F32);
        txt_in_w = load_weight(sf, "txt_in.weight");
        txt_in_b = sf.load_tensor("txt_in.bias", DType::F32);

        // NeRF small weights
        nerf_small.embedder_w = load_weight(sf, "nerf_image_embedder.embedder.0.weight");
        nerf_small.embedder_b = sf.load_tensor("nerf_image_embedder.embedder.0.bias", DType::F32);
        nerf_small.final_norm_scale = sf.load_tensor("nerf_final_layer_conv.norm.scale", DType::F32);
        nerf_small.final_conv_w = sf.load_tensor("nerf_final_layer_conv.conv.weight", DType::F32);
        nerf_small.final_conv_b = sf.load_tensor("nerf_final_layer_conv.conv.bias", DType::F32);

        // ChromaApproximator
        approx.load(sf);

        // Pre-load all block weights
        double_block_weights.reserve(depth);
        for (int i = 0; i < depth; i++) {
            double_block_weights.push_back(DoubleStreamBlockWeights::load(sf, i));
            printf("  Loaded double_block %d/%d\r", i + 1, depth);
            fflush(stdout);
        }
        printf("  Loaded %d double-stream blocks                \n", depth);

        single_block_weights.reserve(depth_single);
        for (int i = 0; i < depth_single; i++) {
            single_block_weights.push_back(SingleStreamBlockWeights::load(sf, i));
            printf("  Loaded single_block %d/%d\r", i + 1, depth_single);
            fflush(stdout);
        }
        printf("  Loaded %d single-stream blocks                \n", depth_single);

        nerf_block_weights.reserve(nerf_depth);
        for (int i = 0; i < nerf_depth; i++)
            nerf_block_weights.push_back(NerfGLUBlockWeights::load(sf, i));
        printf("  Loaded %d NeRF GLU blocks\n", nerf_depth);

        // Deactivate arena — all further allocations go through normal pool/cudaMalloc
        gpu_pool().arena = nullptr;

        // Disown all weight tensors (arena owns the memory, freed all at once)
        img_in_patch_w.owns_data = false; img_in_patch_b.owns_data = false;
        txt_in_w.owns_data = false; txt_in_b.owns_data = false;
        nerf_small.embedder_w.owns_data = false; nerf_small.embedder_b.owns_data = false;
        nerf_small.final_norm_scale.owns_data = false;
        nerf_small.final_conv_w.owns_data = false; nerf_small.final_conv_b.owns_data = false;
        approx.disown();
        for (auto& w : double_block_weights) w.disown();
        for (auto& w : single_block_weights) w.disown();
        for (auto& w : nerf_block_weights) w.disown();

        printf("Chroma Radiance model loaded: %.2f GB arena (%.2f GB used)%s.\n",
               weight_arena.capacity / 1e9, weight_arena.used / 1e9,
               is_int8 ? " [INT8 quantized]" : "");
    }

    // ======================================================================
    // ChromaApproximator input construction + forward
    // ======================================================================
    Tensor compute_modulation(float timestep, float guidance) {
        // timestep_embedding(t*1000, 16) → [1, 16]
        float ts_val = timestep;
        Tensor ts_gpu = Tensor::alloc({1}, DType::F32, true);
        CHECK_CUDA(cudaMemcpy(ts_gpu.data, &ts_val, sizeof(float), cudaMemcpyHostToDevice));
        Tensor ts_emb = Tensor::alloc({1, 16}, DType::F32, true);
        timestep_embedding_cuda(ts_gpu.f32(), ts_emb.f32(), 1, 16, 1000.0f, 10000.0f);

        // timestep_embedding(guidance, 16) → [1, 16]
        float g_val = guidance;
        Tensor g_gpu = Tensor::alloc({1}, DType::F32, true);
        CHECK_CUDA(cudaMemcpy(g_gpu.data, &g_val, sizeof(float), cudaMemcpyHostToDevice));
        Tensor g_emb = Tensor::alloc({1, 16}, DType::F32, true);
        timestep_embedding_cuda(g_gpu.f32(), g_emb.f32(), 1, 16, 1000.0f, 10000.0f);

        // concat → [1, 32]
        Tensor ts_guidance = Tensor::alloc({1, 32}, DType::F32, true);
        concat_last_dim_cuda(ts_emb.f32(), g_emb.f32(), ts_guidance.f32(), 1, 16, 16);

        // Cache mod_index_emb (constant across all steps)
        if (!cached_mod_index_emb.data) {
            std::vector<float> arange(mod_index_length);
            for (int i = 0; i < mod_index_length; i++) arange[i] = (float)i;
            Tensor ar_gpu = Tensor::alloc({(int64_t)mod_index_length}, DType::F32, true);
            CHECK_CUDA(cudaMemcpy(ar_gpu.data, arange.data(), mod_index_length * sizeof(float), cudaMemcpyHostToDevice));
            cached_mod_index_emb = Tensor::alloc({(int64_t)mod_index_length, 32}, DType::F32, true);
            timestep_embedding_cuda(ar_gpu.f32(), cached_mod_index_emb.f32(), mod_index_length, 32, 1000.0f, 10000.0f);
        }

        // Broadcast ts_guidance [1, 32] → [344, 32]
        Tensor ts_guidance_broad = Tensor::alloc({(int64_t)mod_index_length, 32}, DType::F32, true);
        broadcast_bias_cuda(ts_guidance.f32(), ts_guidance_broad.f32(), mod_index_length, 32);

        // concat → [344, 64]
        Tensor vec = Tensor::alloc({(int64_t)mod_index_length, 64}, DType::F32, true);
        concat_last_dim_cuda(ts_guidance_broad.f32(), cached_mod_index_emb.f32(), vec.f32(),
                             mod_index_length, 32, 32);

        // Forward through ChromaApproximator → [344, 3072]
        Tensor mod = approx.forward(vec);
        return mod;
    }

    // ======================================================================
    // RMSNorm for Q/K (per-head, applied to [L, n_head, d_head])
    // ======================================================================
    void qk_norm(float* data, const float* scale, int64_t L, int n_head, int d_head) {
        rms_norm_cuda(data, scale, data, L * n_head, d_head, 1e-6f);
    }

    // ======================================================================
    // Double-stream block forward
    // ======================================================================
    void double_block_forward(int block_idx,
                               Tensor& img, Tensor& txt,
                               const Tensor& mod, const Tensor& pe,
                               const float* attn_mask = nullptr) {
        const auto& w = double_block_weights[block_idx];

        int64_t img_tokens = img.shape[0];
        int64_t txt_tokens = txt.shape[0];
        int64_t total_tokens = img_tokens + txt_tokens;

        // Get modulation vectors
        int img_off = 6 * block_idx + 3 * depth_single;
        int txt_off = 6 * block_idx + 6 * depth + 3 * depth_single;

        float* mod_data = mod.f32();
        auto get_mod = [&](int off) -> float* { return mod_data + off * hidden_size; };

        float* img_shift1 = get_mod(img_off);
        float* img_scale1 = get_mod(img_off + 1);
        float* img_gate1  = get_mod(img_off + 2);
        float* img_shift2 = get_mod(img_off + 3);
        float* img_scale2 = get_mod(img_off + 4);
        float* img_gate2  = get_mod(img_off + 5);

        float* txt_shift1 = get_mod(txt_off);
        float* txt_scale1 = get_mod(txt_off + 1);
        float* txt_gate1  = get_mod(txt_off + 2);
        float* txt_shift2 = get_mod(txt_off + 3);
        float* txt_scale2 = get_mod(txt_off + 4);
        float* txt_gate2  = get_mod(txt_off + 5);

        // Helper: create a weight view for a row slice of a concatenated weight matrix
        // Handles both INT8 (offsets data + per-group quant_scales) and BF16/FP16 (offsets data only)
        auto wrap_weight_view = [](const Tensor& parent, int64_t row_off, int64_t nrows, int64_t ncols) {
            size_t row_bytes = ncols * dtype_size(parent.dtype);
            if (parent.dtype == DType::INT8) {
                int num_groups = (int)((ncols + parent.quant_group_size - 1) / parent.quant_group_size);
                auto t = Tensor::wrap_gpu_int8(
                    (char*)parent.data + row_bytes * row_off, {nrows, ncols},
                    parent.quant_scales
                        ? (char*)parent.quant_scales + (int64_t)row_off * num_groups * dtype_size(parent.quant_scales_dtype)
                        : nullptr,
                    parent.quant_scales_dtype,
                    parent.quant_group_size);
                if (parent.quant_zero_points) {
                    t.quant_zero_points = (char*)parent.quant_zero_points + row_off * num_groups;
                }
                t.quant_smooth = parent.quant_smooth;
                t.calib_name = parent.calib_name;
                return t;
            }
            auto t = Tensor::wrap_gpu(
                (char*)parent.data + row_bytes * row_off, {nrows, ncols}, parent.dtype);
            t.calib_name = parent.calib_name;
            return t;
        };

        // === Image attention ===
        // Split QKV weight [9216, 3072] into 3 views of [3072, 3072]
        Tensor iq_w = wrap_weight_view(w.img_qkv_w, 0, hidden_size, hidden_size);
        Tensor ik_w = wrap_weight_view(w.img_qkv_w, hidden_size, hidden_size, hidden_size);
        Tensor iv_w = wrap_weight_view(w.img_qkv_w, 2 * hidden_size, hidden_size, hidden_size);
        Tensor iq_b = Tensor::wrap_gpu(w.img_qkv_b.f32(), {(int64_t)hidden_size}, DType::F32);
        Tensor ik_b = Tensor::wrap_gpu(w.img_qkv_b.f32() + hidden_size, {(int64_t)hidden_size}, DType::F32);
        Tensor iv_b = Tensor::wrap_gpu(w.img_qkv_b.f32() + 2 * hidden_size, {(int64_t)hidden_size}, DType::F32);

        // === Text attention ===
        // Split txt QKV weight views
        Tensor tq_w = wrap_weight_view(w.txt_qkv_w, 0, hidden_size, hidden_size);
        Tensor tk_w = wrap_weight_view(w.txt_qkv_w, hidden_size, hidden_size, hidden_size);
        Tensor tv_w = wrap_weight_view(w.txt_qkv_w, 2 * hidden_size, hidden_size, hidden_size);
        Tensor tq_b = Tensor::wrap_gpu(w.txt_qkv_b.f32(), {(int64_t)hidden_size}, DType::F32);
        Tensor tk_b = Tensor::wrap_gpu(w.txt_qkv_b.f32() + hidden_size, {(int64_t)hidden_size}, DType::F32);
        Tensor tv_b = Tensor::wrap_gpu(w.txt_qkv_b.f32() + 2 * hidden_size, {(int64_t)hidden_size}, DType::F32);

        // Attention result: [total_tokens, hidden_size] F32
        Tensor attn;
        bool fused_attn_proj = false;  // set true if unpermute+quantize+proj done inside the block

        bool use_fused_attn = (iq_w.dtype == DType::INT8
                               && iq_w.quant_group_size >= (int)iq_w.shape[1]
                               && iq_w.shape[1] % 4 == 0);

        if (use_fused_attn) {
            // Per-channel INT8: fused GEMM → dequant+qknorm+rope+permute → BF16
            // Bypasses F32 intermediates, qk_norm, and attention_with_rope entirely
            Tensor Q_bf16 = Tensor::alloc({(int64_t)num_heads, total_tokens, (int64_t)d_head}, DType::BF16, true);
            Tensor K_bf16 = Tensor::alloc({(int64_t)num_heads, total_tokens, (int64_t)d_head}, DType::BF16, true);
            Tensor V_bf16 = Tensor::alloc({(int64_t)num_heads, total_tokens, (int64_t)d_head}, DType::BF16, true);

            int i32_elems = std::max(img_tokens, txt_tokens) * hidden_size;
            int32_t* i32_scratch = ensure_gemm_i32_scratch(i32_elems);

            // --- Image Q/K/V ---
            QuantizedAct qa_img = quantize_act_modulate(img.f32(), img_shift1, img_scale1,
                1e-6f, img_tokens, hidden_size, iq_w.quant_smooth,
                iq_w.quant_zero_points != nullptr);
            bool need_rowsum_img = (iq_w.quant_zero_points != nullptr);

            // Image Q: GEMM → fused dequant+qknorm+rope → BF16
            linear_int8_gemm(img_tokens, hidden_size, hidden_size, qa_img.x_int8, iq_w.i8(), i32_scratch);
            int8_dequant_qknorm_rope_bf16_cuda(i32_scratch, qa_img.x_scale,
                iq_w.quant_scales, (int)iq_w.quant_scales_dtype,
                need_rowsum_img ? (const int8_t*)iq_w.quant_zero_points : nullptr,
                need_rowsum_img ? qa_img.x_rowsum : nullptr,
                iq_b.f32(), w.img_q_norm.f32(), pe.f32(), Q_bf16.bf16(),
                (int)img_tokens, hidden_size, num_heads, d_head,
                (int)total_tokens, (int)txt_tokens, (int)txt_tokens);

            // Image K: GEMM → fused dequant+qknorm+rope → BF16
            linear_int8_gemm(img_tokens, hidden_size, hidden_size, qa_img.x_int8, ik_w.i8(), i32_scratch);
            int8_dequant_qknorm_rope_bf16_cuda(i32_scratch, qa_img.x_scale,
                ik_w.quant_scales, (int)ik_w.quant_scales_dtype,
                need_rowsum_img ? (const int8_t*)ik_w.quant_zero_points : nullptr,
                need_rowsum_img ? qa_img.x_rowsum : nullptr,
                ik_b.f32(), w.img_k_norm.f32(), pe.f32(), K_bf16.bf16(),
                (int)img_tokens, hidden_size, num_heads, d_head,
                (int)total_tokens, (int)txt_tokens, (int)txt_tokens);

            // Image V: GEMM → fused dequant+permute → BF16 (no norm/RoPE)
            linear_int8_gemm(img_tokens, hidden_size, hidden_size, qa_img.x_int8, iv_w.i8(), i32_scratch);
            int8_dequant_permute_bf16_cuda(i32_scratch, qa_img.x_scale,
                iv_w.quant_scales, (int)iv_w.quant_scales_dtype,
                need_rowsum_img ? (const int8_t*)iv_w.quant_zero_points : nullptr,
                need_rowsum_img ? qa_img.x_rowsum : nullptr,
                iv_b.f32(), V_bf16.bf16(),
                (int)img_tokens, hidden_size, num_heads, d_head,
                (int)total_tokens, (int)txt_tokens);

            // --- Text Q/K/V ---
            QuantizedAct qa_txt = quantize_act_modulate(txt.f32(), txt_shift1, txt_scale1,
                1e-6f, txt_tokens, hidden_size, tq_w.quant_smooth,
                tq_w.quant_zero_points != nullptr);
            bool need_rowsum_txt = (tq_w.quant_zero_points != nullptr);

            // Text Q: GEMM → fused dequant+qknorm+rope → BF16
            linear_int8_gemm(txt_tokens, hidden_size, hidden_size, qa_txt.x_int8, tq_w.i8(), i32_scratch);
            int8_dequant_qknorm_rope_bf16_cuda(i32_scratch, qa_txt.x_scale,
                tq_w.quant_scales, (int)tq_w.quant_scales_dtype,
                need_rowsum_txt ? (const int8_t*)tq_w.quant_zero_points : nullptr,
                need_rowsum_txt ? qa_txt.x_rowsum : nullptr,
                tq_b.f32(), w.txt_q_norm.f32(), pe.f32(), Q_bf16.bf16(),
                (int)txt_tokens, hidden_size, num_heads, d_head,
                (int)total_tokens, 0, 0);

            // Text K
            linear_int8_gemm(txt_tokens, hidden_size, hidden_size, qa_txt.x_int8, tk_w.i8(), i32_scratch);
            int8_dequant_qknorm_rope_bf16_cuda(i32_scratch, qa_txt.x_scale,
                tk_w.quant_scales, (int)tk_w.quant_scales_dtype,
                need_rowsum_txt ? (const int8_t*)tk_w.quant_zero_points : nullptr,
                need_rowsum_txt ? qa_txt.x_rowsum : nullptr,
                tk_b.f32(), w.txt_k_norm.f32(), pe.f32(), K_bf16.bf16(),
                (int)txt_tokens, hidden_size, num_heads, d_head,
                (int)total_tokens, 0, 0);

            // Text V
            linear_int8_gemm(txt_tokens, hidden_size, hidden_size, qa_txt.x_int8, tv_w.i8(), i32_scratch);
            int8_dequant_permute_bf16_cuda(i32_scratch, qa_txt.x_scale,
                tv_w.quant_scales, (int)tv_w.quant_scales_dtype,
                need_rowsum_txt ? (const int8_t*)tv_w.quant_zero_points : nullptr,
                need_rowsum_txt ? qa_txt.x_rowsum : nullptr,
                tv_b.f32(), V_bf16.bf16(),
                (int)txt_tokens, hidden_size, num_heads, d_head,
                (int)total_tokens, 0);

            // cuDNN SDPA directly on fused BF16 buffers
            Tensor O_bf16 = Tensor::alloc({(int64_t)num_heads, total_tokens, (int64_t)d_head}, DType::BF16, true);
            cudnn_sdpa_forward(Q_bf16.bf16(), K_bf16.bf16(), V_bf16.bf16(),
                               attn_mask, O_bf16.bf16(), num_heads, (int)total_tokens, d_head);

            // Check if projection weights support fused unpermute+quantize path
            bool can_fuse_img_proj = (w.img_proj_w.dtype == DType::INT8
                && w.img_proj_w.quant_group_size >= (int)w.img_proj_w.shape[1]
                && w.img_proj_w.shape[1] % 4 == 0);
            bool can_fuse_txt_proj = (w.txt_proj_w.dtype == DType::INT8
                && w.txt_proj_w.quant_group_size >= (int)w.txt_proj_w.shape[1]
                && w.txt_proj_w.shape[1] % 4 == 0);

            if (can_fuse_img_proj && can_fuse_txt_proj) {
                // Fused unpermute BF16 → INT8 quantize, called separately for txt and img
                // Token layout in SDPA output [n_head, total_tokens, d_head]: [txt, img]
                int8_t* attn_i8 = ensure_act_int8_scratch(total_tokens * hidden_size);
                float* attn_sc = ensure_act_scale_scratch(total_tokens);
                bool need_rowsum = (w.img_proj_w.quant_zero_points != nullptr);
                float* attn_rs = need_rowsum ? ensure_act_rowsum_scratch(total_tokens) : nullptr;

                // Text tokens: offset 0 in SDPA output, use txt_proj smooth
                unpermute_bf16_quantize_int8_cuda(
                    O_bf16.bf16(),
                    attn_i8, attn_sc, need_rowsum ? attn_rs : nullptr,
                    w.txt_proj_w.quant_smooth,
                    num_heads, (int)total_tokens, d_head,
                    (int)txt_tokens, 0,
                    need_rowsum, g_hadamard_enabled);

                // Image tokens: offset txt_tokens in SDPA output, use img_proj smooth
                unpermute_bf16_quantize_int8_cuda(
                    O_bf16.bf16(),
                    attn_i8 + txt_tokens * hidden_size,
                    attn_sc + txt_tokens,
                    need_rowsum ? attn_rs + txt_tokens : nullptr,
                    w.img_proj_w.quant_smooth,
                    num_heads, (int)total_tokens, d_head,
                    (int)img_tokens, (int)txt_tokens,
                    need_rowsum, g_hadamard_enabled);

                // Image projection: use pre-quantized INT8 slice
                QuantizedAct qa_img_attn;
                qa_img_attn.x_int8 = attn_i8 + txt_tokens * hidden_size;
                qa_img_attn.x_scale = attn_sc + txt_tokens;
                qa_img_attn.x_rowsum = need_rowsum ? attn_rs + txt_tokens : nullptr;
                qa_img_attn.M = img_tokens;
                qa_img_attn.K = hidden_size;
                linear_prequant_gated_residual(qa_img_attn, w.img_proj_w, &w.img_proj_b,
                                                img.f32(), img_gate1, img_tokens, hidden_size);

                // Text projection: use pre-quantized INT8 slice
                QuantizedAct qa_txt_attn;
                qa_txt_attn.x_int8 = attn_i8;
                qa_txt_attn.x_scale = attn_sc;
                qa_txt_attn.x_rowsum = need_rowsum ? attn_rs : nullptr;
                qa_txt_attn.M = txt_tokens;
                qa_txt_attn.K = hidden_size;
                linear_prequant_gated_residual(qa_txt_attn, w.txt_proj_w, &w.txt_proj_b,
                                                txt.f32(), txt_gate1, txt_tokens, hidden_size);
                fused_attn_proj = true;
            } else {
                // Fallback: unpermute to F32
                attn = Tensor::alloc({total_tokens, (int64_t)(num_heads * d_head)}, DType::F32, true);
                unpermute_bf16_to_f32_cuda(O_bf16.bf16(), attn.f32(), num_heads, (int)total_tokens, d_head);
            }
        } else {
            // Per-group INT8 or BF16: use F32 Q/K/V + separate qk_norm + attention_with_rope
            Tensor all_q = Tensor::alloc({total_tokens, (int64_t)hidden_size}, DType::F32, true);
            Tensor all_k = Tensor::alloc({total_tokens, (int64_t)hidden_size}, DType::F32, true);
            Tensor all_v = Tensor::alloc({total_tokens, (int64_t)hidden_size}, DType::F32, true);

            Tensor txt_q_view = Tensor::wrap_gpu(all_q.f32(), {txt_tokens, (int64_t)hidden_size}, DType::F32);
            Tensor txt_k_view = Tensor::wrap_gpu(all_k.f32(), {txt_tokens, (int64_t)hidden_size}, DType::F32);
            Tensor txt_v_view = Tensor::wrap_gpu(all_v.f32(), {txt_tokens, (int64_t)hidden_size}, DType::F32);
            Tensor img_q_view = Tensor::wrap_gpu(all_q.f32() + txt_tokens * hidden_size, {img_tokens, (int64_t)hidden_size}, DType::F32);
            Tensor img_k_view = Tensor::wrap_gpu(all_k.f32() + txt_tokens * hidden_size, {img_tokens, (int64_t)hidden_size}, DType::F32);
            Tensor img_v_view = Tensor::wrap_gpu(all_v.f32() + txt_tokens * hidden_size, {img_tokens, (int64_t)hidden_size}, DType::F32);

            // Image QKV
            Tensor img_mod_bf16 = Tensor::alloc({img_tokens, (int64_t)hidden_size}, DType::BF16, true);
            modulate_to_bf16_cuda(img.f32(), img_shift1, img_scale1, img_mod_bf16.bf16(),
                                  img_tokens, hidden_size, 1e-6f);
            linear(img_mod_bf16, iq_w, &iq_b, img_q_view);
            linear(img_mod_bf16, ik_w, &ik_b, img_k_view);
            linear(img_mod_bf16, iv_w, &iv_b, img_v_view);

            qk_norm(img_q_view.f32(), w.img_q_norm.f32(), img_tokens, num_heads, d_head);
            qk_norm(img_k_view.f32(), w.img_k_norm.f32(), img_tokens, num_heads, d_head);

            // Text QKV
            Tensor txt_mod_bf16 = Tensor::alloc({txt_tokens, (int64_t)hidden_size}, DType::BF16, true);
            modulate_to_bf16_cuda(txt.f32(), txt_shift1, txt_scale1, txt_mod_bf16.bf16(),
                                  txt_tokens, hidden_size, 1e-6f);
            linear(txt_mod_bf16, tq_w, &tq_b, txt_q_view);
            linear(txt_mod_bf16, tk_w, &tk_b, txt_k_view);
            linear(txt_mod_bf16, tv_w, &tv_b, txt_v_view);

            qk_norm(txt_q_view.f32(), w.txt_q_norm.f32(), txt_tokens, num_heads, d_head);
            qk_norm(txt_k_view.f32(), w.txt_k_norm.f32(), txt_tokens, num_heads, d_head);

            Tensor q_r = all_q.reshape({total_tokens, (int64_t)num_heads, (int64_t)d_head});
            Tensor k_r = all_k.reshape({total_tokens, (int64_t)num_heads, (int64_t)d_head});
            Tensor v_r = all_v.reshape({total_tokens, (int64_t)num_heads, (int64_t)d_head});

            attn = attention_with_rope(q_r, k_r, v_r, pe, num_heads, d_head, attn_mask);
        }

        if (!fused_attn_proj) {
            // Fallback path: F32 attention output → quantize → proj → gated_residual
            Tensor txt_attn = Tensor::wrap_gpu(attn.f32(),
                                                {txt_tokens, (int64_t)hidden_size}, DType::F32);
            Tensor img_attn = Tensor::wrap_gpu(attn.f32() + txt_tokens * hidden_size,
                                                {img_tokens, (int64_t)hidden_size}, DType::F32);

            // === Image: proj + gated residual ===
            linear_gated_residual(img_attn, w.img_proj_w, &w.img_proj_b,
                                   img.f32(), img_gate1, img_tokens, hidden_size);

            // === Text: proj + gated residual ===
            linear_gated_residual(txt_attn, w.txt_proj_w, &w.txt_proj_b,
                                   txt.f32(), txt_gate1, txt_tokens, hidden_size);
        }

        // Image MLP sublayer
        int64_t K_mlp = w.img_mlp_0_w.shape[1];
        if (w.img_mlp_0_w.dtype == DType::INT8
            && w.img_mlp_0_w.quant_group_size >= (int)K_mlp
            && K_mlp % 4 == 0) {
            // Per-channel INT8: fused modulate→quantize, 3-way dequant+GELU+quantize
            QuantizedAct qa_mlp = quantize_act_modulate(img.f32(), img_shift2, img_scale2,
                1e-6f, img_tokens, hidden_size, w.img_mlp_0_w.quant_smooth,
                w.img_mlp_0_w.quant_zero_points != nullptr);

            Tensor img_mlp_h = Tensor::alloc({img_tokens, (int64_t)mlp_hidden}, DType::F32, true);
            linear_int8_gemm(img_tokens, hidden_size, mlp_hidden,
                             qa_mlp.x_int8, w.img_mlp_0_w.i8(), (int32_t*)img_mlp_h.data);

            QuantizedAct qa_mlp2 = dequant_gelu_quantize(
                (const int32_t*)img_mlp_h.data, qa_mlp, w.img_mlp_0_w, &w.img_mlp_0_b,
                w.img_mlp_2_w.quant_smooth, w.img_mlp_2_w.quant_zero_points != nullptr,
                img_tokens, mlp_hidden);

            linear_prequant_gated_residual(qa_mlp2, w.img_mlp_2_w, &w.img_mlp_2_b,
                                            img.f32(), img_gate2, img_tokens, hidden_size);
        } else {
            // Per-group INT8 or BF16: fused modulate→BF16 + fused GELU→BF16
            Tensor img_mod2 = Tensor::alloc({img_tokens, (int64_t)hidden_size}, DType::BF16, true);
            modulate_to_bf16_cuda(img.f32(), img_shift2, img_scale2, img_mod2.bf16(),
                                  img_tokens, hidden_size, 1e-6f);

            Tensor img_mlp_h = linear_forward(img_mod2, w.img_mlp_0_w, &w.img_mlp_0_b);
            Tensor img_mlp_h_bf16 = Tensor::alloc({img_tokens, (int64_t)mlp_hidden}, DType::BF16, true);
            gelu_to_bf16_cuda(img_mlp_h.f32(), img_mlp_h_bf16.bf16(), img_tokens * mlp_hidden);
            Tensor img_mlp_out = linear_forward(img_mlp_h_bf16, w.img_mlp_2_w, &w.img_mlp_2_b);
            gated_residual_cuda(img.f32(), img_mlp_out.f32(), img_gate2,
                                img.f32(), img_tokens, hidden_size);
        }

        // Text MLP sublayer
        int64_t K_mlp_txt = w.txt_mlp_0_w.shape[1];
        if (w.txt_mlp_0_w.dtype == DType::INT8
            && w.txt_mlp_0_w.quant_group_size >= (int)K_mlp_txt
            && K_mlp_txt % 4 == 0) {
            // Per-channel INT8: fused modulate→quantize, 3-way dequant+GELU+quantize
            QuantizedAct qa_mlp = quantize_act_modulate(txt.f32(), txt_shift2, txt_scale2,
                1e-6f, txt_tokens, hidden_size, w.txt_mlp_0_w.quant_smooth,
                w.txt_mlp_0_w.quant_zero_points != nullptr);

            Tensor txt_mlp_h = Tensor::alloc({txt_tokens, (int64_t)mlp_hidden}, DType::F32, true);
            linear_int8_gemm(txt_tokens, hidden_size, mlp_hidden,
                             qa_mlp.x_int8, w.txt_mlp_0_w.i8(), (int32_t*)txt_mlp_h.data);

            QuantizedAct qa_mlp2 = dequant_gelu_quantize(
                (const int32_t*)txt_mlp_h.data, qa_mlp, w.txt_mlp_0_w, &w.txt_mlp_0_b,
                w.txt_mlp_2_w.quant_smooth, w.txt_mlp_2_w.quant_zero_points != nullptr,
                txt_tokens, mlp_hidden);

            linear_prequant_gated_residual(qa_mlp2, w.txt_mlp_2_w, &w.txt_mlp_2_b,
                                            txt.f32(), txt_gate2, txt_tokens, hidden_size);
        } else {
            // Per-group INT8 or BF16: fused modulate→BF16 + fused GELU→BF16
            Tensor txt_mod2 = Tensor::alloc({txt_tokens, (int64_t)hidden_size}, DType::BF16, true);
            modulate_to_bf16_cuda(txt.f32(), txt_shift2, txt_scale2, txt_mod2.bf16(),
                                  txt_tokens, hidden_size, 1e-6f);

            Tensor txt_mlp_h = linear_forward(txt_mod2, w.txt_mlp_0_w, &w.txt_mlp_0_b);
            Tensor txt_mlp_h_bf16 = Tensor::alloc({txt_tokens, (int64_t)mlp_hidden}, DType::BF16, true);
            gelu_to_bf16_cuda(txt_mlp_h.f32(), txt_mlp_h_bf16.bf16(), txt_tokens * mlp_hidden);
            Tensor txt_mlp_out = linear_forward(txt_mlp_h_bf16, w.txt_mlp_2_w, &w.txt_mlp_2_b);
            gated_residual_cuda(txt.f32(), txt_mlp_out.f32(), txt_gate2,
                                txt.f32(), txt_tokens, hidden_size);
        }
    }

    // ======================================================================
    // Single-stream block forward
    // ======================================================================
    void single_block_forward(int block_idx, Tensor& x,
                               const Tensor& mod, const Tensor& pe,
                               const float* attn_mask = nullptr) {
        const auto& w = single_block_weights[block_idx];

        int64_t L = x.shape[0];

        // Modulation: offset = 3*idx
        int off = 3 * block_idx;
        float* mod_data = mod.f32();
        float* shift = mod_data + off * hidden_size;
        float* scale = mod_data + (off + 1) * hidden_size;
        float* gate  = mod_data + (off + 2) * hidden_size;

        // Separate Q/K/V/MLP projections using weight views (eliminates slice_columns)
        // linear1_w is [21504, 3072]: rows [0:3072]=Q, [3072:6144]=K, [6144:9216]=V, [9216:21504]=MLP
        auto wrap_wv = [](const Tensor& parent, int64_t row_off, int64_t nrows, int64_t ncols) {
            size_t row_bytes = ncols * dtype_size(parent.dtype);
            if (parent.dtype == DType::INT8) {
                int num_groups = (int)((ncols + parent.quant_group_size - 1) / parent.quant_group_size);
                auto t = Tensor::wrap_gpu_int8(
                    (char*)parent.data + row_bytes * row_off, {nrows, ncols},
                    parent.quant_scales
                        ? (char*)parent.quant_scales + (int64_t)row_off * num_groups * dtype_size(parent.quant_scales_dtype)
                        : nullptr,
                    parent.quant_scales_dtype,
                    parent.quant_group_size);
                if (parent.quant_zero_points) {
                    t.quant_zero_points = (char*)parent.quant_zero_points + row_off * num_groups;
                }
                t.quant_smooth = parent.quant_smooth;
                t.calib_name = parent.calib_name;
                return t;
            }
            auto t = Tensor::wrap_gpu(
                (char*)parent.data + row_bytes * row_off, {nrows, ncols}, parent.dtype);
            t.calib_name = parent.calib_name;
            return t;
        };

        Tensor sq_w = wrap_wv(w.linear1_w, 0, hidden_size, hidden_size);
        Tensor sk_w = wrap_wv(w.linear1_w, hidden_size, hidden_size, hidden_size);
        Tensor sv_w = wrap_wv(w.linear1_w, 2 * hidden_size, hidden_size, hidden_size);
        Tensor sm_w = wrap_wv(w.linear1_w, 3 * hidden_size, mlp_hidden, hidden_size);
        // Split bias [21504] into Q[0:3072], K[3072:6144], V[6144:9216], MLP[9216:21504]
        Tensor sq_b = Tensor::wrap_gpu(w.linear1_b.f32(), {(int64_t)hidden_size}, DType::F32);
        Tensor sk_b = Tensor::wrap_gpu(w.linear1_b.f32() + hidden_size, {(int64_t)hidden_size}, DType::F32);
        Tensor sv_b = Tensor::wrap_gpu(w.linear1_b.f32() + 2 * hidden_size, {(int64_t)hidden_size}, DType::F32);
        Tensor sm_b = Tensor::wrap_gpu(w.linear1_b.f32() + 3 * hidden_size, {(int64_t)mlp_hidden}, DType::F32);

        Tensor mlp_t = Tensor::alloc({L, (int64_t)mlp_hidden}, DType::F32, true);
        Tensor attn_out;

        bool use_fused_attn = (sq_w.dtype == DType::INT8
                               && sq_w.quant_group_size >= (int)sq_w.shape[1]
                               && sq_w.shape[1] % 4 == 0);

        if (use_fused_attn) {
            // Per-channel INT8: fused GEMM → dequant+qknorm+rope+permute → BF16
            QuantizedAct qa = quantize_act_modulate(x.f32(), shift, scale,
                1e-6f, L, hidden_size, sq_w.quant_smooth,
                sq_w.quant_zero_points != nullptr);
            bool need_rowsum = (sq_w.quant_zero_points != nullptr);
            int32_t* i32_scratch = ensure_gemm_i32_scratch(L * hidden_size);

            // BF16 Q/K/V in [n_head, L, d_head] layout
            Tensor Q_bf16 = Tensor::alloc({(int64_t)num_heads, L, (int64_t)d_head}, DType::BF16, true);
            Tensor K_bf16 = Tensor::alloc({(int64_t)num_heads, L, (int64_t)d_head}, DType::BF16, true);
            Tensor V_bf16 = Tensor::alloc({(int64_t)num_heads, L, (int64_t)d_head}, DType::BF16, true);

            // Q: GEMM → fused dequant+qknorm+rope → BF16
            linear_int8_gemm(L, hidden_size, hidden_size, qa.x_int8, sq_w.i8(), i32_scratch);
            int8_dequant_qknorm_rope_bf16_cuda(i32_scratch, qa.x_scale,
                sq_w.quant_scales, (int)sq_w.quant_scales_dtype,
                need_rowsum ? (const int8_t*)sq_w.quant_zero_points : nullptr,
                need_rowsum ? qa.x_rowsum : nullptr,
                sq_b.f32(), w.q_norm.f32(), pe.f32(), Q_bf16.bf16(),
                (int)L, hidden_size, num_heads, d_head,
                (int)L, 0, 0);

            // K: GEMM → fused dequant+qknorm+rope → BF16
            linear_int8_gemm(L, hidden_size, hidden_size, qa.x_int8, sk_w.i8(), i32_scratch);
            int8_dequant_qknorm_rope_bf16_cuda(i32_scratch, qa.x_scale,
                sk_w.quant_scales, (int)sk_w.quant_scales_dtype,
                need_rowsum ? (const int8_t*)sk_w.quant_zero_points : nullptr,
                need_rowsum ? qa.x_rowsum : nullptr,
                sk_b.f32(), w.k_norm.f32(), pe.f32(), K_bf16.bf16(),
                (int)L, hidden_size, num_heads, d_head,
                (int)L, 0, 0);

            // V: GEMM → fused dequant+permute → BF16 (no norm/RoPE)
            linear_int8_gemm(L, hidden_size, hidden_size, qa.x_int8, sv_w.i8(), i32_scratch);
            int8_dequant_permute_bf16_cuda(i32_scratch, qa.x_scale,
                sv_w.quant_scales, (int)sv_w.quant_scales_dtype,
                need_rowsum ? (const int8_t*)sv_w.quant_zero_points : nullptr,
                need_rowsum ? qa.x_rowsum : nullptr,
                sv_b.f32(), V_bf16.bf16(),
                (int)L, hidden_size, num_heads, d_head,
                (int)L, 0);

            // MLP: unchanged — uses linear_prequant_gelu writing to mlp_t F32
            linear_prequant_gelu(qa, sm_w, &sm_b, mlp_t);

            // cuDNN SDPA
            Tensor O_bf16 = Tensor::alloc({(int64_t)num_heads, L, (int64_t)d_head}, DType::BF16, true);
            cudnn_sdpa_forward(Q_bf16.bf16(), K_bf16.bf16(), V_bf16.bf16(),
                               attn_mask, O_bf16.bf16(), num_heads, (int)L, d_head);
            attn_out = Tensor::alloc({L, (int64_t)(num_heads * d_head)}, DType::F32, true);
            unpermute_bf16_to_f32_cuda(O_bf16.bf16(), attn_out.f32(), num_heads, (int)L, d_head);
        } else {
            // Per-group INT8 or BF16: use F32 Q/K/V + separate qk_norm + attention_with_rope
            Tensor q_t = Tensor::alloc({L, (int64_t)hidden_size}, DType::F32, true);
            Tensor k_t = Tensor::alloc({L, (int64_t)hidden_size}, DType::F32, true);
            Tensor v_t = Tensor::alloc({L, (int64_t)hidden_size}, DType::F32, true);

            Tensor x_mod_bf16 = Tensor::alloc({L, (int64_t)hidden_size}, DType::BF16, true);
            modulate_to_bf16_cuda(x.f32(), shift, scale, x_mod_bf16.bf16(), L, hidden_size, 1e-6f);
            linear(x_mod_bf16, sq_w, &sq_b, q_t);
            linear(x_mod_bf16, sk_w, &sk_b, k_t);
            linear(x_mod_bf16, sv_w, &sv_b, v_t);
            linear(x_mod_bf16, sm_w, &sm_b, mlp_t);
            gelu_cuda(mlp_t.f32(), mlp_t.f32(), L * mlp_hidden);

            qk_norm(q_t.f32(), w.q_norm.f32(), L, num_heads, d_head);
            qk_norm(k_t.f32(), w.k_norm.f32(), L, num_heads, d_head);

            Tensor q_r = q_t.reshape({L, (int64_t)num_heads, (int64_t)d_head});
            Tensor k_r = k_t.reshape({L, (int64_t)num_heads, (int64_t)d_head});
            Tensor v_r = v_t.reshape({L, (int64_t)num_heads, (int64_t)d_head});

            attn_out = attention_with_rope(q_r, k_r, v_r, pe, num_heads, d_head, attn_mask);
        }

        // Concat attn + MLP, then linear2 + gated residual
        int64_t concat_dim = hidden_size + mlp_hidden;
        if (w.linear2_w.dtype == DType::INT8
            && w.linear2_w.quant_group_size >= (int)w.linear2_w.shape[1]
            && w.linear2_w.shape[1] % 4 == 0) {
            // Per-channel INT8: fused concat+quantize → GEMM → fused dequant+gated_residual
            QuantizedAct qa_concat = concat_quantize_act(
                attn_out.f32(), mlp_t.f32(), L, hidden_size, mlp_hidden,
                w.linear2_w.quant_smooth, w.linear2_w.quant_zero_points != nullptr);
            linear_prequant_gated_residual(qa_concat, w.linear2_w, &w.linear2_b,
                                            x.f32(), gate, L, hidden_size);
        } else {
            // Per-group INT8 or BF16: concat→BF16 + linear + gated_residual
            Tensor attn_mlp = Tensor::alloc({L, concat_dim}, DType::BF16, true);
            concat_last_dim_to_bf16_cuda(attn_out.f32(), mlp_t.f32(), attn_mlp.bf16(),
                                          L, hidden_size, mlp_hidden);
            Tensor output = linear_forward(attn_mlp, w.linear2_w, &w.linear2_b);
            gated_residual_cuda(x.f32(), output.f32(), gate, x.f32(), L, hidden_size);
        }
    }

    // ======================================================================
    // NeRF decoder
    // ======================================================================
    Tensor nerf_decode(const Tensor& transformer_out, const Tensor& orig_img,
                       const Tensor& dct_features,
                       int H, int W) {
        int h_patches = H / patch_size;
        int w_patches = W / patch_size;
        int num_patches = h_patches * w_patches;
        int pixels_per_patch = patch_size * patch_size; // 256

        // 1. Patchify original image: [1, 3, H, W] → [num_patches, 256, 3]
        Tensor patches = Tensor::alloc({1, (int64_t)num_patches, (int64_t)(3 * pixels_per_patch)},
                                        DType::F32, true);
        patchify_cuda(orig_img.f32(), patches.f32(), 1, 3, H, W, patch_size);

        Tensor patches_perm = Tensor::alloc({(int64_t)num_patches, (int64_t)pixels_per_patch, 3},
                                             DType::F32, true);
        permute_3d_cuda(patches.f32(), patches_perm.f32(),
                        num_patches, 3, pixels_per_patch, 0, 2, 1);

        // 2. Broadcast pre-uploaded DCT features [256, 64] to [num_patches, 256, 64]
        Tensor dct_broad = Tensor::alloc({(int64_t)num_patches, (int64_t)pixels_per_patch,
                                           (int64_t)(nerf_max_freqs * nerf_max_freqs)}, DType::F32, true);
        broadcast_rows_cuda(dct_features.f32(), dct_broad.f32(), num_patches,
                            pixels_per_patch, nerf_max_freqs * nerf_max_freqs);

        // Concat: [num_patches, 256, 3] + [num_patches, 256, 64] → [num_patches, 256, 67]
        int64_t feat_dim = 3 + nerf_max_freqs * nerf_max_freqs; // 67
        Tensor nerf_input = Tensor::alloc({(int64_t)num_patches, (int64_t)pixels_per_patch, feat_dim},
                                           DType::F32, true);
        concat_last_dim_cuda(patches_perm.f32(), dct_broad.f32(), nerf_input.f32(),
                             (int64_t)num_patches * pixels_per_patch, 3,
                             nerf_max_freqs * nerf_max_freqs);

        // 3. NerfEmbedder: Linear(67→64)
        Tensor nerf_flat = nerf_input.reshape({(int64_t)num_patches * pixels_per_patch, feat_dim});
        Tensor img_dct = Tensor::alloc({(int64_t)num_patches * pixels_per_patch, (int64_t)nerf_hidden},
                                        DType::F32, true);
        linear(nerf_flat, nerf_small.embedder_w, &nerf_small.embedder_b, img_dct);
        Tensor img_dct_3d = img_dct.reshape({(int64_t)num_patches, (int64_t)pixels_per_patch,
                                              (int64_t)nerf_hidden});

        // 4. Get conditioning
        Tensor nerf_hidden_t = transformer_out.reshape({(int64_t)num_patches, (int64_t)hidden_size});

        // 5. NerfGLUBlocks
        if (fwd_call_count <= 1) {
            print_tensor_stats("nerf_embedder_out", img_dct.f32(), img_dct.numel());
            print_tensor_stats("nerf_conditioning", nerf_hidden_t.f32(), nerf_hidden_t.numel());
        }
        for (int i = 0; i < nerf_depth; i++) {
            const auto& bw = nerf_block_weights[i];
            nerf_glu_block(img_dct_3d, nerf_hidden_t, bw);
            if (fwd_call_count <= 1) {
                char buf[64];
                snprintf(buf, sizeof(buf), "after_nerf_glu_%d", i);
                print_tensor_stats(buf, img_dct_3d.f32(), img_dct_3d.numel());
            }
        }

        // 6. Permute [num_patches, 256, 64] → [num_patches, 64, 256]
        Tensor img_perm = Tensor::alloc({(int64_t)num_patches, (int64_t)nerf_hidden,
                                          (int64_t)pixels_per_patch}, DType::F32, true);
        permute_3d_cuda(img_dct_3d.f32(), img_perm.f32(),
                        num_patches, pixels_per_patch, nerf_hidden, 0, 2, 1);

        // Reshape to [1, num_patches, 64*256] for unpatchify
        Tensor for_unpatch = img_perm.reshape({1, (int64_t)num_patches,
                                                (int64_t)(nerf_hidden * pixels_per_patch)});

        // 7. Unpatchify → [1, 64, H, W]
        Tensor img_out = Tensor::alloc({1, (int64_t)nerf_hidden, (int64_t)H, (int64_t)W},
                                        DType::F32, true);
        unpatchify_cuda(for_unpatch.f32(), img_out.f32(), 1, nerf_hidden, H, W, patch_size,
                        h_patches, w_patches);

        // 8. NerfFinalLayerConv: fused NCHW RMSNorm + Conv2d(64→3)
        // Fused kernel normalizes across C at each (h,w) directly in NCHW layout
        // Replaces: NCHW→NHWC permute + RMSNorm + NHWC→NCHW permute (3 kernels → 1)
        Tensor nchw = Tensor::alloc({1, (int64_t)nerf_hidden, (int64_t)H, (int64_t)W},
                                     DType::F32, true);
        rms_norm_nchw_cuda(img_out.f32(), nerf_small.final_norm_scale.f32(), nchw.f32(),
                            nerf_hidden, H, W, 1e-6f);

        Tensor final_out = Tensor::alloc({1, 3, (int64_t)H, (int64_t)W}, DType::F32, true);
        conv2d_3x3_cuda(nchw.f32(), nerf_small.final_conv_w.f32(), nerf_small.final_conv_b.f32(),
                        final_out.f32(), 1, nerf_hidden, 3, H, W);

        return final_out;
    }

    // NerfGLUBlock forward
    void nerf_glu_block(Tensor& x, const Tensor& s, const NerfGLUBlockWeights& w) {
        // x: [num_patches, 256, 64]
        // s: [num_patches, 3072]
        int64_t batch = x.shape[0];
        int64_t seq = x.shape[1];       // 256
        int64_t hid = x.shape[2];       // 64
        int64_t mlp_dim = hid * nerf_mlp_ratio; // 256

        // param_generator: [batch, 3 * 64 * 256] from s [batch, 3072]
        Tensor params = linear_forward(s, w.param_gen_w, &w.param_gen_b);

        // Fused slice+transpose: read column slice from params, write transposed directly
        // fc1_gate, fc1_value: [hid, mlp_dim] per batch → transpose to [mlp_dim, hid]
        // fc2: [mlp_dim, hid] per batch → transpose to [hid, mlp_dim]
        int64_t chunk = hid * mlp_dim;
        int64_t total_param_cols = 3 * chunk;
        Tensor w_gate = Tensor::alloc({batch, mlp_dim, hid}, DType::F32, true);
        Tensor w_value = Tensor::alloc({batch, mlp_dim, hid}, DType::F32, true);
        Tensor w_fc2 = Tensor::alloc({batch, hid, mlp_dim}, DType::F32, true);

        slice_transpose_cuda(params.f32(), w_gate.f32(), batch, total_param_cols, 0, hid, mlp_dim);
        slice_transpose_cuda(params.f32(), w_value.f32(), batch, total_param_cols, chunk, hid, mlp_dim);
        slice_transpose_cuda(params.f32(), w_fc2.f32(), batch, total_param_cols, 2 * chunk, mlp_dim, hid);

        // L2 normalize along last dim
        l2_norm_cuda(w_gate.f32(), w_gate.f32(), batch * mlp_dim, hid, 1e-12f);
        l2_norm_cuda(w_value.f32(), w_value.f32(), batch * mlp_dim, hid, 1e-12f);
        l2_norm_cuda(w_fc2.f32(), w_fc2.f32(), batch * hid, mlp_dim, 1e-12f);

        // Save residual (async D2D — no CPU sync needed)
        Tensor res = Tensor::alloc({batch, seq, hid}, DType::F32, true);
        CHECK_CUDA(cudaMemcpyAsync(res.data, x.data, batch * seq * hid * sizeof(float), cudaMemcpyDeviceToDevice));

        // RMSNorm on x
        Tensor x_normed = Tensor::alloc({batch, seq, hid}, DType::F32, true);
        rms_norm_cuda(x.f32(), w.norm_scale.f32(), x_normed.f32(), batch * seq, hid, 1e-6f);

        // Batched matmul with B transposed
        Tensor x1 = Tensor::alloc({batch, seq, mlp_dim}, DType::F32, true);
        Tensor x2 = Tensor::alloc({batch, seq, mlp_dim}, DType::F32, true);

        batched_matmul_Bt(x_normed, w_gate, x1);
        batched_matmul_Bt(x_normed, w_value, x2);

        // Fused SiLU + mul: x1 = silu(x1) * x2
        silu_mul_cuda(x1.f32(), x2.f32(), x1.f32(), batch * seq * mlp_dim);

        // x = x1 @ w_fc2^T
        Tensor x_out = Tensor::alloc({batch, seq, hid}, DType::F32, true);
        batched_matmul_Bt(x1, w_fc2, x_out);

        // Residual
        add_cuda(x_out.f32(), res.f32(), x.f32(), batch * seq * hid);
    }

    // ======================================================================
    // Conv2d for img_in_patch (16×16 stride 16, implemented as im2col + GEMM)
    // ======================================================================
    Tensor conv2d_img_in(const Tensor& input, int H, int W) {
        int out_h = H / patch_size;
        int out_w = W / patch_size;
        int C_in = 3;
        int C_out = hidden_size;
        int kHW = patch_size * patch_size;
        int64_t num_patches = out_h * out_w;

        Tensor col = Tensor::alloc({num_patches, (int64_t)(C_in * kHW)}, DType::F32, true);
        patchify_cuda(input.f32(), col.f32(), 1, C_in, H, W, patch_size);

        Tensor W_flat = img_in_patch_w.reshape({(int64_t)C_out, (int64_t)(C_in * kHW)});
        Tensor output = Tensor::alloc({num_patches, (int64_t)C_out}, DType::F32, true);
        linear(col, W_flat, &img_in_patch_b, output);

        return output;
    }

    // ======================================================================
    // Full forward pass
    // ======================================================================
    static int fwd_call_count;
    static bool debug_diag;

    void print_tensor_stats(const char* name, const float* gpu_data, int64_t n) {
        if (!debug_diag) return;
        std::vector<float> v(n);
        CHECK_CUDA(cudaMemcpy(v.data(), gpu_data, n * sizeof(float), cudaMemcpyDeviceToHost));
        double sum = 0, abs_sum = 0;
        for (int64_t i = 0; i < n; i++) { sum += v[i]; abs_sum += fabs(v[i]); }
        printf("  [diag] %s: sum=%.6f, mean_abs=%.12f, first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
               name, sum, abs_sum / n, v[0], v[1], v[2], v[3], v[4]);
    }

    Tensor forward(const Tensor& x, const Tensor& context, float timestep,
                    const Tensor& pe, const Tensor& dct_features,
                    const float* attn_mask = nullptr,
                    const Tensor* shared_img = nullptr,
                    const Tensor* shared_mod = nullptr) {
        // GPU pool reuses activation buffers by size across steps — no release needed

        int H = x.shape[2], W = x.shape[3];
        int64_t txt_tokens = context.shape[0];
        bool diag = debug_diag && (fwd_call_count == 0);  // diagnostics on first call only

        if (diag) {
            printf("[DIAG] Forward call #0, timestep=%.6f, H=%d, W=%d\n", timestep, H, W);
            print_tensor_stats("input_x", x.f32(), std::min(x.numel(), (int64_t)196608));
            // Print weight stats
            if (img_in_patch_w.dtype == DType::BF16) {
                printf("  [diag] img_in_patch_w: BF16, shape=[%lld,%lld,%lld,%lld], nbytes=%zu\n",
                    (long long)img_in_patch_w.shape[0], (long long)(img_in_patch_w.ndim > 1 ? img_in_patch_w.shape[1] : 0),
                    (long long)(img_in_patch_w.ndim > 2 ? img_in_patch_w.shape[2] : 0),
                    (long long)(img_in_patch_w.ndim > 3 ? img_in_patch_w.shape[3] : 0),
                    img_in_patch_w.nbytes());
                // Convert first 10 BF16 weights to F32 to print
                uint16_t wbuf[10];
                CHECK_CUDA(cudaMemcpy(wbuf, img_in_patch_w.data, 10 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                printf("  [diag] img_in_patch_w first10 BF16 raw: ");
                for (int i = 0; i < 10; i++) {
                    uint32_t f32bits = (uint32_t)wbuf[i] << 16;
                    float fval;
                    memcpy(&fval, &f32bits, 4);
                    printf("%.6f ", fval);
                }
                printf("\n");
            }
        }

        // 1. Image tokenization: Conv2d (or clone precomputed for CFG sharing)
        Tensor img;
        if (shared_img) {
            // Clone shared img (blocks modify it in-place via gated_residual)
            img = Tensor::alloc({shared_img->shape[0], shared_img->shape[1]}, shared_img->dtype, true);
            CHECK_CUDA(cudaMemcpyAsync(img.data, shared_img->data, shared_img->nbytes(),
                                        cudaMemcpyDeviceToDevice));
        } else {
            img = conv2d_img_in(x, H, W);
        }
        int64_t img_tokens = img.shape[0];

        if (diag) {
            print_tensor_stats("img_after_conv", img.f32(), img.numel());
        }

        // 2. Text projection: Linear(4096→3072)
        Tensor txt = Tensor::alloc({txt_tokens, (int64_t)hidden_size}, DType::F32, true);

        if (diag) {
            // Verify txt_in weight values
            printf("  [diag] txt_in_w: dtype=%d, shape=[%lld,%lld], nbytes=%zu\n",
                (int)txt_in_w.dtype, (long long)txt_in_w.shape[0],
                (long long)(txt_in_w.ndim > 1 ? txt_in_w.shape[1] : 0), txt_in_w.nbytes());

            // Convert weight to F32 and print first values
            Tensor w_f32 = to_f32_gpu(txt_in_w);
            std::vector<float> wv(10);
            CHECK_CUDA(cudaMemcpy(wv.data(), w_f32.f32(), 10 * sizeof(float), cudaMemcpyDeviceToHost));
            printf("  [diag] txt_in_w first10 (as f32): ");
            for (int i = 0; i < 10; i++) printf("%.8f ", wv[i]);
            printf("\n");
            // Expected from PyTorch: [-0.0811, -0.0156, -0.0503, 0.0188, -0.0027, ...]

            // Print context first values
            std::vector<float> cv(10);
            CHECK_CUDA(cudaMemcpy(cv.data(), context.f32(), 10 * sizeof(float), cudaMemcpyDeviceToHost));
            printf("  [diag] context first10: ");
            for (int i = 0; i < 10; i++) printf("%.8f ", cv[i]);
            printf("\n");

            // Print bias first values
            std::vector<float> bv(5);
            CHECK_CUDA(cudaMemcpy(bv.data(), txt_in_b.f32(), 5 * sizeof(float), cudaMemcpyDeviceToHost));
            printf("  [diag] txt_in_b first5: ");
            for (int i = 0; i < 5; i++) printf("%.8f ", bv[i]);
            printf("\n");

            printf("  [diag] context dtype=%d, txt_in_w dtype=%d\n", (int)context.dtype, (int)txt_in_w.dtype);
        }

        linear(context, txt_in_w, &txt_in_b, txt);

        if (diag) {
            print_tensor_stats("txt_after_proj", txt.f32(), txt.numel());
        }

        // 3. Compute or reuse modulation vectors (shared_mod avoids recomputing for CFG)
        Tensor mod_view;
        if (shared_mod) {
            mod_view = Tensor::wrap_gpu(shared_mod->data,
                                         {(int64_t)mod_index_length, (int64_t)hidden_size}, DType::F32);
        }
        Tensor mod_owned;
        if (!shared_mod) {
            mod_owned = compute_modulation(timestep, 0.0f);
        }
        const Tensor& mod = shared_mod ? mod_view : mod_owned;

        if (diag) {
            print_tensor_stats("modulation", mod.f32(), mod.numel());
        }

        fwd_call_count++;

        // 4. Double-stream blocks
        for (int i = 0; i < depth; i++) {
            double_block_forward(i, img, txt, mod, pe, attn_mask);
            if (diag && i == 0) {
                print_tensor_stats("img_after_dblk0", img.f32(), img.numel());
                print_tensor_stats("txt_after_dblk0", txt.f32(), txt.numel());
            }
        }

        // 5. Concat txt + img for single-stream blocks
        int64_t total = txt_tokens + img_tokens;
        Tensor combined = Tensor::alloc({total, (int64_t)hidden_size}, DType::F32, true);
        concat_first_dim_cuda(txt.f32(), img.f32(), combined.f32(),
                              txt_tokens, img_tokens, hidden_size);

        // 6. Single-stream blocks
        for (int i = 0; i < depth_single; i++) {
            single_block_forward(i, combined, mod, pe, attn_mask);
        }

        // 7. Extract img tokens — zero-copy view into combined (no D2D copy needed)
        Tensor img_out = Tensor::wrap_gpu(combined.f32() + txt_tokens * hidden_size,
                                           {img_tokens, (int64_t)hidden_size}, DType::F32);

        if (diag) {
            print_tensor_stats("transformer_out (img_out)", img_out.f32(), img_out.numel());
        }

        // Pool keeps single-block buffers for reuse (NeRF allocates different sizes from pool)

        // 8. NeRF decode
        Tensor output = nerf_decode(img_out, x, dct_features, H, W);

        if (diag) {
            print_tensor_stats("nerf_output (velocity)", output.f32(), output.numel());
        }

        return output;
    }
};

int ChromaRadiance::fwd_call_count = 0;
bool ChromaRadiance::debug_diag = false;
