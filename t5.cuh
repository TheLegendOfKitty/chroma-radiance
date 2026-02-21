#pragma once
#include "tensor.cuh"
#include "cublas_ops.cuh"
#include "attention.cuh"
#include "safetensors.cuh"
#include <string>
#include <vector>
#include <cmath>
#include <unordered_map>

// Forward declarations
void rms_norm_cuda(const float* x, const float* scale, float* out, int64_t rows, int64_t C, float eps);
void gelu_cuda(const float* x, float* out, int64_t n);
void gelu_exact_cuda(const float* x, float* out, int64_t n);
void add_cuda(const float* a, const float* b, float* out, int64_t n);
void mul_cuda(const float* a, const float* b, float* out, int64_t n);
void scale_cuda(float* x, float scale, int64_t n);
void embedding_lookup_f16_cuda(const __half* table, const int* ids, float* output, int N, int dim);
void softmax_cuda(const float* x, float* out, int64_t rows, int64_t C);
void permute_4d_cuda(const float* input, float* output,
                     int64_t d0, int64_t d1, int64_t d2, int64_t d3,
                     int p0, int p1, int p2, int p3);

// T5 relative position bucket computation
static std::vector<int> compute_relative_position_bucket(int query_len, int key_len,
                                                          bool bidirectional = true,
                                                          int num_buckets = 32,
                                                          int max_distance = 128) {
    std::vector<int> result(query_len * key_len);

    int half_buckets = bidirectional ? num_buckets / 2 : num_buckets;
    int max_exact = half_buckets / 2;

    for (int q = 0; q < query_len; q++) {
        for (int k = 0; k < key_len; k++) {
            int rel_pos = k - q;
            int bucket = 0;

            if (bidirectional) {
                if (rel_pos > 0) bucket += half_buckets;
                int abs_pos = std::abs(rel_pos);
                if (abs_pos < max_exact) {
                    bucket += abs_pos;
                } else {
                    float log_pos = logf((float)abs_pos / max_exact);
                    float log_base = logf((float)max_distance / max_exact);
                    int b = max_exact + (int)(log_pos / log_base * (half_buckets - max_exact));
                    b = std::min(b, half_buckets - 1);
                    bucket += b;
                }
            } else {
                int abs_pos = std::max(-rel_pos, 0);
                if (abs_pos < max_exact) {
                    bucket += abs_pos;
                } else {
                    float log_pos = logf((float)abs_pos / max_exact);
                    float log_base = logf((float)max_distance / max_exact);
                    int b = max_exact + (int)(log_pos / log_base * (num_buckets - max_exact));
                    b = std::min(b, num_buckets - 1);
                    bucket += b;
                }
            }

            result[q * key_len + k] = bucket;
        }
    }
    return result;
}

struct T5Encoder {
    // Model parameters (all on GPU)
    Tensor shared_weight;       // [32128, 4096] FP16 (kept native)
    Tensor final_norm_weight;   // [4096] F32

    struct Block {
        // Self-attention
        Tensor attn_norm_weight;   // [4096]
        Tensor q_weight, k_weight, v_weight, o_weight;  // [4096, 4096] FP16
        Tensor rel_attn_bias;      // [32, 64] FP16 — only block 0

        // FFN
        Tensor ff_norm_weight;     // [4096]
        Tensor wi_0_weight, wi_1_weight, wo_weight;  // FP16
    };
    std::vector<Block> blocks;

    int num_layers = 24;
    int model_dim = 4096;
    int ff_dim = 10240;
    int num_heads = 64;
    int d_head = 64;
    int vocab_size = 32128;

    void load(const SafetensorsFile& sf) {
        printf("Loading T5-XXL encoder...\n");

        // Load shared embedding (keep as FP16 for lookup)
        shared_weight = sf.load_tensor_native("shared.weight");
        final_norm_weight = sf.load_tensor("encoder.final_layer_norm.weight", DType::F32);

        blocks.resize(num_layers);
        for (int i = 0; i < num_layers; i++) {
            std::string prefix = "encoder.block." + std::to_string(i) + ".";
            Block& b = blocks[i];

            // Attention norm
            b.attn_norm_weight = sf.load_tensor(prefix + "layer.0.layer_norm.weight", DType::F32);

            // Q, K, V, O weights (keep as FP16)
            b.q_weight = sf.load_tensor_native(prefix + "layer.0.SelfAttention.q.weight");
            b.k_weight = sf.load_tensor_native(prefix + "layer.0.SelfAttention.k.weight");
            b.v_weight = sf.load_tensor_native(prefix + "layer.0.SelfAttention.v.weight");
            b.o_weight = sf.load_tensor_native(prefix + "layer.0.SelfAttention.o.weight");

            // Relative attention bias (only block 0)
            if (i == 0) {
                b.rel_attn_bias = sf.load_tensor(prefix + "layer.0.SelfAttention.relative_attention_bias.weight", DType::F32);
            }

            // FFN norm
            b.ff_norm_weight = sf.load_tensor(prefix + "layer.1.layer_norm.weight", DType::F32);

            // FFN weights (keep as FP16)
            b.wi_0_weight = sf.load_tensor_native(prefix + "layer.1.DenseReluDense.wi_0.weight");
            b.wi_1_weight = sf.load_tensor_native(prefix + "layer.1.DenseReluDense.wi_1.weight");
            b.wo_weight = sf.load_tensor_native(prefix + "layer.1.DenseReluDense.wo.weight");
        }
        printf("T5-XXL encoder loaded.\n");
    }

    // Compute relative position bias from block 0's embedding
    // rel_attn_bias: [32, 64] (num_buckets, num_heads)
    // buckets: [L, L] int array
    // output: [num_heads, L, L]
    Tensor compute_position_bias(const std::vector<int>& buckets, int L) {
        // Upload buckets to GPU
        Tensor buckets_gpu = Tensor::alloc({(int64_t)L * L}, DType::F32, true);
        {
            // Convert int buckets to float indices for embedding lookup
            // Actually we need int indices on GPU
            Tensor bucket_ids_cpu = Tensor::alloc({(int64_t)L * L}, DType::F32, false);
            for (int i = 0; i < L * L; i++) {
                ((float*)bucket_ids_cpu.data)[i] = 0; // placeholder
            }

            // We'll do the lookup manually: for each position pair (q,k), look up bias[bucket, head]
            // bias: [32, 64], bucket_ids: [L*L]
            // output: [L*L, 64] then permute to [64, L, L]
            Tensor bias_cpu = blocks[0].rel_attn_bias.to_cpu();
            float* bias_data = bias_cpu.f32();

            std::vector<float> result(L * L * num_heads);
            for (int i = 0; i < L * L; i++) {
                int bucket = buckets[i];
                for (int h = 0; h < num_heads; h++) {
                    result[i * num_heads + h] = bias_data[bucket * num_heads + h];
                }
            }

            // Shape: [L*L, num_heads] → [L, L, num_heads]
            // Need to permute to [num_heads, L, L]
            Tensor bias_flat = Tensor::alloc({(int64_t)L * L, (int64_t)num_heads}, DType::F32, false);
            memcpy(bias_flat.data, result.data(), result.size() * sizeof(float));
            Tensor bias_gpu = bias_flat.to_gpu();

            // Permute [L*L, num_heads] → reshape to [L, L, num_heads] → permute to [num_heads, L, L]
            Tensor bias_out = Tensor::alloc({(int64_t)num_heads, (int64_t)L, (int64_t)L}, DType::F32, true);
            permute_4d_cuda(bias_gpu.f32(), bias_out.f32(),
                            L, L, num_heads, 1,
                            2, 0, 1, 3);
            return bias_out;
        }
    }

    // Forward pass
    // input_ids: vector of token IDs (includes padding)
    // n_real_tokens: number of real tokens before padding (including EOS)
    // Returns: [seq_len, model_dim] tensor on GPU
    Tensor forward(const std::vector<int>& input_ids, int n_real_tokens = -1) {
        int L = (int)input_ids.size();

        // 1. Embedding lookup
        Tensor ids_gpu = Tensor::alloc({(int64_t)L}, DType::F32, true);
        {
            std::vector<int> ids_int(input_ids.begin(), input_ids.end());
            Tensor ids_int_gpu = Tensor::alloc({(int64_t)L}, DType::F32, true);
            CHECK_CUDA(cudaMemcpy(ids_int_gpu.data, ids_int.data(), L * sizeof(int), cudaMemcpyHostToDevice));
            ids_gpu = std::move(ids_int_gpu);
        }

        Tensor hidden = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
        embedding_lookup_f16_cuda(shared_weight.fp16(), (const int*)ids_gpu.data,
                                   hidden.f32(), L, model_dim);

        // 2. Create attention mask for padding tokens
        // mask[j] = 0.0 for real tokens, -inf for padding
        // This blocks attention to padding positions in the key dimension
        Tensor t5_attn_mask;
        float* t5_mask_ptr = nullptr;
        if (n_real_tokens >= 0 && n_real_tokens < L) {
            std::vector<float> mask_vec(L, 0.0f);
            for (int i = n_real_tokens; i < L; i++) {
                mask_vec[i] = -HUGE_VALF;
            }
            t5_attn_mask = Tensor::alloc({(int64_t)L}, DType::F32, true);
            CHECK_CUDA(cudaMemcpy(t5_attn_mask.data, mask_vec.data(), L * sizeof(float), cudaMemcpyHostToDevice));
            t5_mask_ptr = t5_attn_mask.f32();
            printf("T5 attention mask: %d real tokens, %d padding tokens masked\n", n_real_tokens, L - n_real_tokens);
        }

        // 3. Compute relative position buckets
        auto buckets = compute_relative_position_bucket(L, L);
        Tensor pos_bias = compute_position_bias(buckets, L);

        // 4. Process through blocks
        // T5 attention: reference scales K by sqrt(d_head) then divides by sqrt(d_head)
        // in attention, so net effect is Q @ K^T with no scaling. Use scale=1.0.
        float attn_scale = 1.0f;

        for (int i = 0; i < num_layers; i++) {
            Block& b = blocks[i];

            // Self-attention sublayer
            {
                // RMSNorm
                Tensor normed = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
                rms_norm_cuda(hidden.f32(), b.attn_norm_weight.f32(), normed.f32(), L, model_dim, 1e-6f);

                // Q, K, V projections
                Tensor q = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
                Tensor k = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
                Tensor v = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
                linear(normed, b.q_weight, nullptr, q);
                linear(normed, b.k_weight, nullptr, k);
                linear(normed, b.v_weight, nullptr, v);

                // Attention (T5-style: scale applied in attention function)
                // Reshape to [L, num_heads, d_head]
                Tensor q_r = q.reshape({(int64_t)L, (int64_t)num_heads, (int64_t)d_head});
                Tensor k_r = k.reshape({(int64_t)L, (int64_t)num_heads, (int64_t)d_head});
                Tensor v_r = v.reshape({(int64_t)L, (int64_t)num_heads, (int64_t)d_head});

                Tensor attn_out = t5_attention(q_r, k_r, v_r, pos_bias.f32(),
                                                num_heads, d_head, attn_scale, t5_mask_ptr);

                // Output projection
                Tensor proj = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
                linear(attn_out, b.o_weight, nullptr, proj);

                // Residual
                add_cuda(proj.f32(), hidden.f32(), hidden.f32(), (int64_t)L * model_dim);
            }

            // FFN sublayer
            {
                // RMSNorm
                Tensor normed = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
                rms_norm_cuda(hidden.f32(), b.ff_norm_weight.f32(), normed.f32(), L, model_dim, 1e-6f);

                // Gated FFN: GELU(wi_0(x)) * wi_1(x) -> wo
                Tensor gate = Tensor::alloc({(int64_t)L, (int64_t)ff_dim}, DType::F32, true);
                Tensor value = Tensor::alloc({(int64_t)L, (int64_t)ff_dim}, DType::F32, true);
                linear(normed, b.wi_0_weight, nullptr, gate);
                linear(normed, b.wi_1_weight, nullptr, value);

                // Tanh-approximate GELU on gate (matches ComfyUI's gelu_pytorch_tanh)
                gelu_cuda(gate.f32(), gate.f32(), (int64_t)L * ff_dim);

                // gate * value
                mul_cuda(gate.f32(), value.f32(), gate.f32(), (int64_t)L * ff_dim);

                // wo projection (reference pre-scales by 1/32 then post-scales by 32 for
                // numerical stability, net effect is no scaling)
                Tensor ff_out = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
                linear(gate, b.wo_weight, nullptr, ff_out);

                // Residual
                add_cuda(ff_out.f32(), hidden.f32(), hidden.f32(), (int64_t)L * model_dim);
            }
        }

        // 4. Final layer norm
        Tensor output = Tensor::alloc({(int64_t)L, (int64_t)model_dim}, DType::F32, true);
        rms_norm_cuda(hidden.f32(), final_norm_weight.f32(), output.f32(), L, model_dim, 1e-6f);

        return output;
    }

    void free_weights() {
        shared_weight.free_data();
        final_norm_weight.free_data();
        for (auto& b : blocks) {
            b.attn_norm_weight.free_data();
            b.q_weight.free_data();
            b.k_weight.free_data();
            b.v_weight.free_data();
            b.o_weight.free_data();
            b.rel_attn_bias.free_data();
            b.ff_norm_weight.free_data();
            b.wi_0_weight.free_data();
            b.wi_1_weight.free_data();
            b.wo_weight.free_data();
        }
        blocks.clear();
    }
};
