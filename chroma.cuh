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
void silu_cuda(const float* x, float* out, int64_t n);
void add_cuda(const float* a, const float* b, float* out, int64_t n);
void mul_cuda(const float* a, const float* b, float* out, int64_t n);
void modulate_cuda(const float* x, const float* shift, const float* scale,
                   float* out, int64_t M, int64_t C, float eps);
void timestep_embedding_cuda(const float* input, float* output,
                             int N, int dim, float time_factor, float max_period);
void l2_norm_cuda(const float* x, float* out, int64_t rows, int64_t C, float eps);
void gated_residual_cuda(const float* x, const float* y, const float* gate,
                         float* out, int64_t M, int64_t C);
void concat_last_dim_cuda(const float* a, const float* b, float* out,
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
void slice_columns_cuda(const float* src, float* dst,
                         int64_t L, int64_t total_cols,
                         int64_t col_offset, int64_t slice_cols);

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
        in_proj_w = sf.load_tensor_native(p + "in_proj.weight");
        in_proj_b = sf.load_tensor(p + "in_proj.bias", DType::F32);
        out_proj_w = sf.load_tensor_native(p + "out_proj.weight");
        out_proj_b = sf.load_tensor(p + "out_proj.bias", DType::F32);

        for (int i = 0; i < 5; i++) {
            std::string lp = p + "layers." + std::to_string(i) + ".";
            std::string np = p + "norms." + std::to_string(i) + ".";
            layers[i].norm_scale = sf.load_tensor(np + "scale", DType::F32);
            layers[i].in_w = sf.load_tensor_native(lp + "in_layer.weight");
            layers[i].in_b = sf.load_tensor(lp + "in_layer.bias", DType::F32);
            layers[i].out_w = sf.load_tensor_native(lp + "out_layer.weight");
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
// Double-stream block weights (loaded on-demand)
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

    static DoubleStreamBlockWeights load(const SafetensorsFile& sf, int idx) {
        DoubleStreamBlockWeights w;
        std::string p = "double_blocks." + std::to_string(idx) + ".";

        w.img_qkv_w = sf.load_tensor_native(p + "img_attn.qkv.weight");
        w.img_qkv_b = sf.load_tensor(p + "img_attn.qkv.bias", DType::F32);
        w.img_q_norm = sf.load_tensor(p + "img_attn.norm.query_norm.scale", DType::F32);
        w.img_k_norm = sf.load_tensor(p + "img_attn.norm.key_norm.scale", DType::F32);
        w.img_proj_w = sf.load_tensor_native(p + "img_attn.proj.weight");
        w.img_proj_b = sf.load_tensor(p + "img_attn.proj.bias", DType::F32);
        w.img_mlp_0_w = sf.load_tensor_native(p + "img_mlp.0.weight");
        w.img_mlp_0_b = sf.load_tensor(p + "img_mlp.0.bias", DType::F32);
        w.img_mlp_2_w = sf.load_tensor_native(p + "img_mlp.2.weight");
        w.img_mlp_2_b = sf.load_tensor(p + "img_mlp.2.bias", DType::F32);

        w.txt_qkv_w = sf.load_tensor_native(p + "txt_attn.qkv.weight");
        w.txt_qkv_b = sf.load_tensor(p + "txt_attn.qkv.bias", DType::F32);
        w.txt_q_norm = sf.load_tensor(p + "txt_attn.norm.query_norm.scale", DType::F32);
        w.txt_k_norm = sf.load_tensor(p + "txt_attn.norm.key_norm.scale", DType::F32);
        w.txt_proj_w = sf.load_tensor_native(p + "txt_attn.proj.weight");
        w.txt_proj_b = sf.load_tensor(p + "txt_attn.proj.bias", DType::F32);
        w.txt_mlp_0_w = sf.load_tensor_native(p + "txt_mlp.0.weight");
        w.txt_mlp_0_b = sf.load_tensor(p + "txt_mlp.0.bias", DType::F32);
        w.txt_mlp_2_w = sf.load_tensor_native(p + "txt_mlp.2.weight");
        w.txt_mlp_2_b = sf.load_tensor(p + "txt_mlp.2.bias", DType::F32);

        return w;
    }
};

// ============================================================================
// Single-stream block weights (loaded on-demand)
// ============================================================================
struct SingleStreamBlockWeights {
    Tensor linear1_w, linear1_b;   // [21504, 3072], [21504]
    Tensor linear2_w, linear2_b;   // [3072, 15360], [3072]
    Tensor q_norm, k_norm;         // [128]

    static SingleStreamBlockWeights load(const SafetensorsFile& sf, int idx) {
        SingleStreamBlockWeights w;
        std::string p = "single_blocks." + std::to_string(idx) + ".";

        w.linear1_w = sf.load_tensor_native(p + "linear1.weight");
        w.linear1_b = sf.load_tensor(p + "linear1.bias", DType::F32);
        w.linear2_w = sf.load_tensor_native(p + "linear2.weight");
        w.linear2_b = sf.load_tensor(p + "linear2.bias", DType::F32);
        w.q_norm = sf.load_tensor(p + "norm.query_norm.scale", DType::F32);
        w.k_norm = sf.load_tensor(p + "norm.key_norm.scale", DType::F32);

        return w;
    }
};

// ============================================================================
// NeRF GLU block weights (loaded on-demand)
// ============================================================================
struct NerfGLUBlockWeights {
    Tensor param_gen_w, param_gen_b;  // [49152, 3072], [49152]
    Tensor norm_scale;                // [64]

    static NerfGLUBlockWeights load(const SafetensorsFile& sf, int idx) {
        NerfGLUBlockWeights w;
        std::string p = "nerf_blocks." + std::to_string(idx) + ".";
        w.param_gen_w = sf.load_tensor_native(p + "param_generator.weight");
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
// Chroma Radiance Model (weight-streaming)
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

    // Small always-resident weights
    Tensor img_in_patch_w, img_in_patch_b;  // Conv2d(3→3072, 16×16)
    Tensor txt_in_w, txt_in_b;              // Linear(4096→3072)
    NerfSmallWeights nerf_small;

    // ChromaApproximator (loaded once per step, freed after modulation)
    ChromaApproximator approx;

    // SafetensorsFile reference for on-demand weight loading
    const SafetensorsFile* sf_ptr = nullptr;

    void load(const SafetensorsFile& sf) {
        printf("Loading Chroma Radiance model (weight-streaming mode)...\n");
        sf_ptr = &sf;

        // Small resident weights
        img_in_patch_w = sf.load_tensor_native("img_in_patch.weight");
        img_in_patch_b = sf.load_tensor("img_in_patch.bias", DType::F32);
        txt_in_w = sf.load_tensor_native("txt_in.weight");
        txt_in_b = sf.load_tensor("txt_in.bias", DType::F32);

        // NeRF small weights (tiny, always resident)
        nerf_small.embedder_w = sf.load_tensor_native("nerf_image_embedder.embedder.0.weight");
        nerf_small.embedder_b = sf.load_tensor("nerf_image_embedder.embedder.0.bias", DType::F32);
        nerf_small.final_norm_scale = sf.load_tensor("nerf_final_layer_conv.norm.scale", DType::F32);
        nerf_small.final_conv_w = sf.load_tensor("nerf_final_layer_conv.conv.weight", DType::F32);
        nerf_small.final_conv_b = sf.load_tensor("nerf_final_layer_conv.conv.bias", DType::F32);

        // ChromaApproximator (loaded once, used every step for modulation)
        approx.load(sf);

        printf("Chroma Radiance model loaded (resident: ~%.0f MB, blocks streamed on-demand).\n",
               (img_in_patch_w.nbytes() + txt_in_w.nbytes() +
                nerf_small.embedder_w.nbytes() + nerf_small.final_conv_w.nbytes()) / 1e6);
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

        // timestep_embedding(arange(0..344), 32) → [344, 32]
        std::vector<float> arange(mod_index_length);
        for (int i = 0; i < mod_index_length; i++) arange[i] = (float)i;
        Tensor ar_gpu = Tensor::alloc({(int64_t)mod_index_length}, DType::F32, true);
        CHECK_CUDA(cudaMemcpy(ar_gpu.data, arange.data(), mod_index_length * sizeof(float), cudaMemcpyHostToDevice));
        Tensor mod_index_emb = Tensor::alloc({(int64_t)mod_index_length, 32}, DType::F32, true);
        timestep_embedding_cuda(ar_gpu.f32(), mod_index_emb.f32(), mod_index_length, 32, 1000.0f, 10000.0f);

        // Broadcast ts_guidance [1, 32] → [344, 32]
        Tensor ts_guidance_broad = Tensor::alloc({(int64_t)mod_index_length, 32}, DType::F32, true);
        broadcast_bias_cuda(ts_guidance.f32(), ts_guidance_broad.f32(), mod_index_length, 32);

        // concat → [344, 64]
        Tensor vec = Tensor::alloc({(int64_t)mod_index_length, 64}, DType::F32, true);
        concat_last_dim_cuda(ts_guidance_broad.f32(), mod_index_emb.f32(), vec.f32(),
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
    // Double-stream block forward (weights loaded on-demand)
    // ======================================================================
    void double_block_forward(int block_idx,
                               Tensor& img, Tensor& txt,
                               const Tensor& mod, const Tensor& pe,
                               const float* attn_mask = nullptr) {
        // Load weights for this block
        DoubleStreamBlockWeights w = DoubleStreamBlockWeights::load(*sf_ptr, block_idx);

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

        // === Image attention ===
        Tensor img_mod = Tensor::alloc({img_tokens, (int64_t)hidden_size}, DType::F32, true);
        modulate_cuda(img.f32(), img_shift1, img_scale1, img_mod.f32(),
                      img_tokens, hidden_size, 1e-6f);

        Tensor img_qkv = linear_forward(img_mod, w.img_qkv_w, &w.img_qkv_b);
        // QKV is [img_tokens, 9216] row-major; Q=cols[0:3072], K=cols[3072:6144], V=cols[6144:9216]
        Tensor img_q = Tensor::alloc({img_tokens, (int64_t)hidden_size}, DType::F32, true);
        Tensor img_k = Tensor::alloc({img_tokens, (int64_t)hidden_size}, DType::F32, true);
        Tensor img_v = Tensor::alloc({img_tokens, (int64_t)hidden_size}, DType::F32, true);
        slice_columns_cuda(img_qkv.f32(), img_q.f32(), img_tokens, 3 * hidden_size, 0, hidden_size);
        slice_columns_cuda(img_qkv.f32(), img_k.f32(), img_tokens, 3 * hidden_size, hidden_size, hidden_size);
        slice_columns_cuda(img_qkv.f32(), img_v.f32(), img_tokens, 3 * hidden_size, 2 * hidden_size, hidden_size);

        qk_norm(img_q.f32(), w.img_q_norm.f32(), img_tokens, num_heads, d_head);
        qk_norm(img_k.f32(), w.img_k_norm.f32(), img_tokens, num_heads, d_head);

        // === Text attention ===
        Tensor txt_mod_t = Tensor::alloc({txt_tokens, (int64_t)hidden_size}, DType::F32, true);
        modulate_cuda(txt.f32(), txt_shift1, txt_scale1, txt_mod_t.f32(),
                      txt_tokens, hidden_size, 1e-6f);

        Tensor txt_qkv = linear_forward(txt_mod_t, w.txt_qkv_w, &w.txt_qkv_b);
        // QKV is [txt_tokens, 9216] row-major; Q=cols[0:3072], K=cols[3072:6144], V=cols[6144:9216]
        Tensor txt_q = Tensor::alloc({txt_tokens, (int64_t)hidden_size}, DType::F32, true);
        Tensor txt_k = Tensor::alloc({txt_tokens, (int64_t)hidden_size}, DType::F32, true);
        Tensor txt_v = Tensor::alloc({txt_tokens, (int64_t)hidden_size}, DType::F32, true);
        slice_columns_cuda(txt_qkv.f32(), txt_q.f32(), txt_tokens, 3 * hidden_size, 0, hidden_size);
        slice_columns_cuda(txt_qkv.f32(), txt_k.f32(), txt_tokens, 3 * hidden_size, hidden_size, hidden_size);
        slice_columns_cuda(txt_qkv.f32(), txt_v.f32(), txt_tokens, 3 * hidden_size, 2 * hidden_size, hidden_size);

        qk_norm(txt_q.f32(), w.txt_q_norm.f32(), txt_tokens, num_heads, d_head);
        qk_norm(txt_k.f32(), w.txt_k_norm.f32(), txt_tokens, num_heads, d_head);

        // === Joint attention ===
        Tensor all_q = Tensor::alloc({total_tokens, (int64_t)hidden_size}, DType::F32, true);
        Tensor all_k = Tensor::alloc({total_tokens, (int64_t)hidden_size}, DType::F32, true);
        Tensor all_v = Tensor::alloc({total_tokens, (int64_t)hidden_size}, DType::F32, true);
        concat_first_dim_cuda(txt_q.f32(), img_q.f32(), all_q.f32(),
                              txt_tokens, img_tokens, hidden_size);
        concat_first_dim_cuda(txt_k.f32(), img_k.f32(), all_k.f32(),
                              txt_tokens, img_tokens, hidden_size);
        concat_first_dim_cuda(txt_v.f32(), img_v.f32(), all_v.f32(),
                              txt_tokens, img_tokens, hidden_size);

        Tensor q_r = all_q.reshape({total_tokens, (int64_t)num_heads, (int64_t)d_head});
        Tensor k_r = all_k.reshape({total_tokens, (int64_t)num_heads, (int64_t)d_head});
        Tensor v_r = all_v.reshape({total_tokens, (int64_t)num_heads, (int64_t)d_head});

        Tensor attn = attention_with_rope(q_r, k_r, v_r, pe, num_heads, d_head, attn_mask);

        Tensor txt_attn = Tensor::wrap_gpu(attn.f32(),
                                            {txt_tokens, (int64_t)hidden_size}, DType::F32);
        Tensor img_attn = Tensor::wrap_gpu(attn.f32() + txt_tokens * hidden_size,
                                            {img_tokens, (int64_t)hidden_size}, DType::F32);

        // === Image: proj + gated residual ===
        Tensor img_attn_proj = linear_forward(img_attn, w.img_proj_w, &w.img_proj_b);
        gated_residual_cuda(img.f32(), img_attn_proj.f32(), img_gate1,
                            img.f32(), img_tokens, hidden_size);

        // Image MLP sublayer
        Tensor img_mod2 = Tensor::alloc({img_tokens, (int64_t)hidden_size}, DType::F32, true);
        modulate_cuda(img.f32(), img_shift2, img_scale2, img_mod2.f32(),
                      img_tokens, hidden_size, 1e-6f);

        Tensor img_mlp_h = linear_forward(img_mod2, w.img_mlp_0_w, &w.img_mlp_0_b);
        gelu_cuda(img_mlp_h.f32(), img_mlp_h.f32(), img_tokens * mlp_hidden);
        Tensor img_mlp_out = linear_forward(img_mlp_h, w.img_mlp_2_w, &w.img_mlp_2_b);
        gated_residual_cuda(img.f32(), img_mlp_out.f32(), img_gate2,
                            img.f32(), img_tokens, hidden_size);

        // === Text: proj + gated residual ===
        Tensor txt_attn_proj = linear_forward(txt_attn, w.txt_proj_w, &w.txt_proj_b);
        gated_residual_cuda(txt.f32(), txt_attn_proj.f32(), txt_gate1,
                            txt.f32(), txt_tokens, hidden_size);

        // Text MLP sublayer
        Tensor txt_mod2 = Tensor::alloc({txt_tokens, (int64_t)hidden_size}, DType::F32, true);
        modulate_cuda(txt.f32(), txt_shift2, txt_scale2, txt_mod2.f32(),
                      txt_tokens, hidden_size, 1e-6f);

        Tensor txt_mlp_h = linear_forward(txt_mod2, w.txt_mlp_0_w, &w.txt_mlp_0_b);
        gelu_cuda(txt_mlp_h.f32(), txt_mlp_h.f32(), txt_tokens * mlp_hidden);
        Tensor txt_mlp_out = linear_forward(txt_mlp_h, w.txt_mlp_2_w, &w.txt_mlp_2_b);
        gated_residual_cuda(txt.f32(), txt_mlp_out.f32(), txt_gate2,
                            txt.f32(), txt_tokens, hidden_size);

        // w goes out of scope here → all block weights freed by destructors
    }

    // ======================================================================
    // Single-stream block forward (weights loaded on-demand)
    // ======================================================================
    void single_block_forward(int block_idx, Tensor& x,
                               const Tensor& mod, const Tensor& pe,
                               const float* attn_mask = nullptr) {
        // Load weights for this block
        SingleStreamBlockWeights w = SingleStreamBlockWeights::load(*sf_ptr, block_idx);

        int64_t L = x.shape[0];

        // Modulation: offset = 3*idx
        int off = 3 * block_idx;
        float* mod_data = mod.f32();
        float* shift = mod_data + off * hidden_size;
        float* scale = mod_data + (off + 1) * hidden_size;
        float* gate  = mod_data + (off + 2) * hidden_size;

        // Modulate
        Tensor x_mod = Tensor::alloc({L, (int64_t)hidden_size}, DType::F32, true);
        modulate_cuda(x.f32(), shift, scale, x_mod.f32(), L, hidden_size, 1e-6f);

        // linear1: [L, 21504] = [L, 3*3072 + 12288]
        Tensor qkv_mlp = linear_forward(x_mod, w.linear1_w, &w.linear1_b);
        int64_t total_cols = 3 * hidden_size + mlp_hidden; // 21504

        // Split: Q(3072), K(3072), V(3072), MLP(12288) - column slices from row-major [L, 21504]
        Tensor q_t = Tensor::alloc({L, (int64_t)hidden_size}, DType::F32, true);
        Tensor k_t = Tensor::alloc({L, (int64_t)hidden_size}, DType::F32, true);
        Tensor v_t = Tensor::alloc({L, (int64_t)hidden_size}, DType::F32, true);
        Tensor mlp_t = Tensor::alloc({L, (int64_t)mlp_hidden}, DType::F32, true);
        slice_columns_cuda(qkv_mlp.f32(), q_t.f32(), L, total_cols, 0, hidden_size);
        slice_columns_cuda(qkv_mlp.f32(), k_t.f32(), L, total_cols, hidden_size, hidden_size);
        slice_columns_cuda(qkv_mlp.f32(), v_t.f32(), L, total_cols, 2 * hidden_size, hidden_size);
        slice_columns_cuda(qkv_mlp.f32(), mlp_t.f32(), L, total_cols, 3 * hidden_size, mlp_hidden);

        // QKNorm
        qk_norm(q_t.f32(), w.q_norm.f32(), L, num_heads, d_head);
        qk_norm(k_t.f32(), w.k_norm.f32(), L, num_heads, d_head);

        // Attention with RoPE
        Tensor q_r = q_t.reshape({L, (int64_t)num_heads, (int64_t)d_head});
        Tensor k_r = k_t.reshape({L, (int64_t)num_heads, (int64_t)d_head});
        Tensor v_r = v_t.reshape({L, (int64_t)num_heads, (int64_t)d_head});

        Tensor attn_out = attention_with_rope(q_r, k_r, v_r, pe, num_heads, d_head, attn_mask);

        // GELU on MLP portion
        gelu_cuda(mlp_t.f32(), mlp_t.f32(), L * mlp_hidden);

        // Concat attn(3072) + mlp(12288) → [L, 15360]
        int64_t concat_dim = hidden_size + mlp_hidden;
        Tensor attn_mlp = Tensor::alloc({L, concat_dim}, DType::F32, true);
        concat_last_dim_cuda(attn_out.f32(), mlp_t.f32(), attn_mlp.f32(),
                             L, hidden_size, mlp_hidden);

        // linear2: [L, 3072]
        Tensor output = linear_forward(attn_mlp, w.linear2_w, &w.linear2_b);

        // Gated residual
        gated_residual_cuda(x.f32(), output.f32(), gate, x.f32(), L, hidden_size);

        // w goes out of scope → block weights freed
    }

    // ======================================================================
    // NeRF decoder
    // ======================================================================
    Tensor nerf_decode(const Tensor& transformer_out, const Tensor& orig_img,
                       const std::vector<float>& dct_features,
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

        // 2. Upload DCT features [256, 64] and concat with patches
        Tensor dct_gpu = Tensor::alloc({(int64_t)pixels_per_patch, (int64_t)(nerf_max_freqs * nerf_max_freqs)},
                                        DType::F32, true);
        CHECK_CUDA(cudaMemcpy(dct_gpu.data, dct_features.data(),
                              dct_features.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Broadcast DCT to [num_patches, 256, 64]
        Tensor dct_broad = Tensor::alloc({(int64_t)num_patches, (int64_t)pixels_per_patch,
                                           (int64_t)(nerf_max_freqs * nerf_max_freqs)}, DType::F32, true);
        for (int i = 0; i < num_patches; i++) {
            CHECK_CUDA(cudaMemcpy(
                (float*)dct_broad.data + i * pixels_per_patch * nerf_max_freqs * nerf_max_freqs,
                dct_gpu.data,
                pixels_per_patch * nerf_max_freqs * nerf_max_freqs * sizeof(float),
                cudaMemcpyDeviceToDevice));
        }

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

        // 5. NerfGLUBlocks (streamed on-demand)
        if (fwd_call_count <= 1) {
            print_tensor_stats("nerf_embedder_out", img_dct.f32(), img_dct.numel());
            print_tensor_stats("nerf_conditioning", nerf_hidden_t.f32(), nerf_hidden_t.numel());
        }
        for (int i = 0; i < nerf_depth; i++) {
            NerfGLUBlockWeights bw = NerfGLUBlockWeights::load(*sf_ptr, i);
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

        // 8. NerfFinalLayerConv: permute to [1, H, W, 64], RMSNorm, permute back, Conv2d(64→3)
        Tensor nhwc = Tensor::alloc({1, (int64_t)H, (int64_t)W, (int64_t)nerf_hidden},
                                     DType::F32, true);
        permute_4d_cuda(img_out.f32(), nhwc.f32(), 1, nerf_hidden, H, W, 0, 2, 3, 1);

        rms_norm_cuda(nhwc.f32(), nerf_small.final_norm_scale.f32(), nhwc.f32(),
                      (int64_t)H * W, nerf_hidden, 1e-6f);

        Tensor nchw = Tensor::alloc({1, (int64_t)nerf_hidden, (int64_t)H, (int64_t)W},
                                     DType::F32, true);
        permute_4d_cuda(nhwc.f32(), nchw.f32(), 1, H, W, nerf_hidden, 0, 3, 1, 2);

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

        // Split into 3 matrices, each [batch, hid*mlp_dim = 64*256 = 16384]
        // params is [batch, 3*chunk] row-major; split along columns
        int64_t chunk = hid * mlp_dim;
        int64_t total_param_cols = 3 * chunk;
        Tensor fc1_gate_t = Tensor::alloc({batch, chunk}, DType::F32, true);
        Tensor fc1_value_t = Tensor::alloc({batch, chunk}, DType::F32, true);
        Tensor fc2_t = Tensor::alloc({batch, chunk}, DType::F32, true);
        slice_columns_cuda(params.f32(), fc1_gate_t.f32(), batch, total_param_cols, 0, chunk);
        slice_columns_cuda(params.f32(), fc1_value_t.f32(), batch, total_param_cols, chunk, chunk);
        slice_columns_cuda(params.f32(), fc2_t.f32(), batch, total_param_cols, 2 * chunk, chunk);

        // fc1_gate, fc1_value: flat data in C order is [hid, mlp_dim] per batch
        //   → transpose to [mlp_dim, hid] per batch (ggml permute)
        //   → L2 norm on last dim (hid)
        // fc2: flat data in C order is [mlp_dim, hid] per batch
        //   → transpose to [hid, mlp_dim] per batch (ggml permute)
        //   → L2 norm on last dim (mlp_dim)
        Tensor w_gate = Tensor::alloc({batch, mlp_dim, hid}, DType::F32, true);
        Tensor w_value = Tensor::alloc({batch, mlp_dim, hid}, DType::F32, true);
        Tensor w_fc2 = Tensor::alloc({batch, hid, mlp_dim}, DType::F32, true);

        batched_transpose_2d_cuda(fc1_gate_t.f32(), w_gate.f32(), batch, hid, mlp_dim);
        batched_transpose_2d_cuda(fc1_value_t.f32(), w_value.f32(), batch, hid, mlp_dim);
        batched_transpose_2d_cuda(fc2_t.f32(), w_fc2.f32(), batch, mlp_dim, hid);

        // L2 normalize along last dim
        l2_norm_cuda(w_gate.f32(), w_gate.f32(), batch * mlp_dim, hid, 1e-12f);
        l2_norm_cuda(w_value.f32(), w_value.f32(), batch * mlp_dim, hid, 1e-12f);
        l2_norm_cuda(w_fc2.f32(), w_fc2.f32(), batch * hid, mlp_dim, 1e-12f);

        // Save residual
        Tensor res = Tensor::alloc({batch, seq, hid}, DType::F32, true);
        CHECK_CUDA(cudaMemcpy(res.data, x.data, batch * seq * hid * sizeof(float), cudaMemcpyDeviceToDevice));

        // RMSNorm on x
        Tensor x_normed = Tensor::alloc({batch, seq, hid}, DType::F32, true);
        rms_norm_cuda(x.f32(), w.norm_scale.f32(), x_normed.f32(), batch * seq, hid, 1e-6f);

        // Transpose weight matrices for batched matmul: C = A @ B
        Tensor w_gate_t = Tensor::alloc({batch, hid, mlp_dim}, DType::F32, true);
        Tensor w_value_t = Tensor::alloc({batch, hid, mlp_dim}, DType::F32, true);
        Tensor w_fc2_t = Tensor::alloc({batch, mlp_dim, hid}, DType::F32, true);
        batched_transpose_2d_cuda(w_gate.f32(), w_gate_t.f32(), batch, mlp_dim, hid);
        batched_transpose_2d_cuda(w_value.f32(), w_value_t.f32(), batch, mlp_dim, hid);
        batched_transpose_2d_cuda(w_fc2.f32(), w_fc2_t.f32(), batch, hid, mlp_dim);

        Tensor x1 = Tensor::alloc({batch, seq, mlp_dim}, DType::F32, true);
        Tensor x2 = Tensor::alloc({batch, seq, mlp_dim}, DType::F32, true);

        batched_matmul(x_normed, w_gate_t, x1);
        batched_matmul(x_normed, w_value_t, x2);

        // SiLU on x1
        silu_cuda(x1.f32(), x1.f32(), batch * seq * mlp_dim);

        // x = x1 * x2
        mul_cuda(x1.f32(), x2.f32(), x1.f32(), batch * seq * mlp_dim);

        // x = x1 @ w_fc2^T = [batch, seq, mlp_dim] @ [batch, mlp_dim, hid]
        Tensor x_out = Tensor::alloc({batch, seq, hid}, DType::F32, true);
        batched_matmul(x1, w_fc2_t, x_out);

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
                    const Tensor& pe, const std::vector<float>& dct_features,
                    const float* attn_mask = nullptr) {
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

        // 1. Image tokenization: Conv2d
        Tensor img = conv2d_img_in(x, H, W);
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

        // 3. Compute modulation vectors (uses ChromaApproximator)
        Tensor mod = compute_modulation(timestep, 0.0f);

        if (diag) {
            print_tensor_stats("modulation", mod.f32(), mod.numel());
        }

        fwd_call_count++;

        // 4. Double-stream blocks (weights streamed)
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

        // 6. Single-stream blocks (weights streamed)
        for (int i = 0; i < depth_single; i++) {
            single_block_forward(i, combined, mod, pe, attn_mask);
        }

        // 7. Extract img tokens (last img_tokens entries)
        Tensor img_out = Tensor::alloc({img_tokens, (int64_t)hidden_size}, DType::F32, true);
        CHECK_CUDA(cudaMemcpy(img_out.data,
                              combined.f32() + txt_tokens * hidden_size,
                              img_tokens * hidden_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        if (diag) {
            print_tensor_stats("transformer_out (img_out)", img_out.f32(), img_out.numel());
        }

        // 8. NeRF decode (NeRF block weights streamed)
        Tensor output = nerf_decode(img_out, x, dct_features, H, W);

        if (diag) {
            print_tensor_stats("nerf_output (velocity)", output.f32(), output.numel());
        }

        return output;
    }
};

int ChromaRadiance::fwd_call_count = 0;
bool ChromaRadiance::debug_diag = false;
