#include "tensor.cuh"
#include "safetensors.cuh"
#include "cublas_ops.cuh"
#include "kernels.cu"
#include "rope.cuh"
#include "attention.cuh"
#include "t5_tokenizer.cuh"
#include "t5.cuh"
#include "chroma.cuh"
#include "sampler.cuh"
#include "image_io.cuh"
#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>

// Precompute DCT position features for NeRF decoder
// patch_size=16, max_freqs=8
// Returns [256, 64] flat array
static std::vector<float> compute_dct_features(int patch_size, int max_freqs) {
    const float PI = 3.14159265358979323846f;

    std::vector<float> pos(patch_size);
    for (int i = 0; i < patch_size; i++) {
        pos[i] = (float)i / (float)(patch_size - 1);
    }

    // 2D grid positions
    std::vector<float> pos_x(patch_size * patch_size);
    std::vector<float> pos_y(patch_size * patch_size);
    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < patch_size; j++) {
            pos_x[i * patch_size + j] = pos[j];
            pos_y[i * patch_size + j] = pos[i];
        }
    }

    // Frequency coefficients
    std::vector<float> coeffs(max_freqs * max_freqs);
    for (int fx = 0; fx < max_freqs; fx++) {
        for (int fy = 0; fy < max_freqs; fy++) {
            coeffs[fx * max_freqs + fy] = 1.0f / (1.0f + (float)fx * (float)fy);
        }
    }

    int num_positions = patch_size * patch_size;
    int num_features = max_freqs * max_freqs;
    std::vector<float> dct(num_positions * num_features);

    for (int p = 0; p < num_positions; p++) {
        float px = pos_x[p];
        float py = pos_y[p];
        for (int fx = 0; fx < max_freqs; fx++) {
            float cx = cosf(px * (float)fx * PI);
            for (int fy = 0; fy < max_freqs; fy++) {
                float cy = cosf(py * (float)fy * PI);
                dct[p * num_features + fx * max_freqs + fy] = cx * cy * coeffs[fx * max_freqs + fy];
            }
        }
    }

    return dct;
}

struct Args {
    std::string prompt = "a photo of a cat";
    std::string negative_prompt = "";
    float cfg_scale = 1.0f;
    int width = 512;
    int height = 512;
    int steps = 20;
    unsigned long long seed = 42;
    std::string output = "output.png";
    std::string model_path = "";
    std::string t5_path = "";
    std::string tokenizer_path = "";
    int rng_mode = 1;  // 0 = pytorch, 1 = sd.cpp (default to sd.cpp for comparison)
    bool no_mask = false;
    bool debug = false;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--negative-prompt") == 0) {
            args.negative_prompt = argv[++i];
        } else if (strcmp(argv[i], "--cfg-scale") == 0) {
            args.cfg_scale = atof(argv[++i]);
        } else if (strcmp(argv[i], "-W") == 0 || strcmp(argv[i], "--width") == 0) {
            args.width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--height") == 0) {
            args.height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0) {
            args.steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0) {
            args.seed = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            args.output = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            args.model_path = argv[++i];
        } else if (strcmp(argv[i], "--t5") == 0) {
            args.t5_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            args.tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--rng") == 0) {
            std::string rng_str = argv[++i];
            if (rng_str == "pytorch") args.rng_mode = 0;
            else if (rng_str == "sdcpp") args.rng_mode = 1;
            else { fprintf(stderr, "Unknown --rng: %s (use pytorch or sdcpp)\n", rng_str.c_str()); exit(1); }
        } else if (strcmp(argv[i], "--no-mask") == 0) {
            args.no_mask = true;
        } else if (strcmp(argv[i], "--debug") == 0) {
            args.debug = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: chroma-radiance [options]\n");
            printf("  -p, --prompt TEXT     Prompt text (default: 'a photo of a cat')\n");
            printf("  -n, --negative-prompt TEXT  Negative prompt for CFG uncond pass (default: '')\n");
            printf("  --cfg-scale F         Classifier-free guidance scale (default: 1.0)\n");
            printf("  -W, --width N         Image width (default: 512)\n");
            printf("  -H, --height N        Image height (default: 512)\n");
            printf("  --steps N             Number of sampling steps (default: 20)\n");
            printf("  --seed N              Random seed (default: 42)\n");
            printf("  -o, --output PATH     Output image path (default: output.png)\n");
            printf("  -m, --model PATH      Chroma model path (default: auto-detect)\n");
            printf("  --t5 PATH             T5 model path (default: auto-detect)\n");
            printf("  --tokenizer PATH      T5 tokenizer JSON path (default: auto-detect)\n");
            printf("  --rng MODE            RNG mode: pytorch or sdcpp (default: sdcpp)\n");
            exit(0);
        }
    }

    // Auto-detect paths
    if (args.model_path.empty()) {
        args.model_path = "Chroma1-Radiance-v0.4.safetensors";
    }
    if (args.t5_path.empty()) {
        args.t5_path = "t5xxl_fp16.safetensors";
    }
    if (args.tokenizer_path.empty()) {
        args.tokenizer_path = "t5_tokenizer.json";
    }

    return args;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    printf("=== Chroma Radiance Standalone (cuBLAS) ===\n");
    printf("Prompt: %s\n", args.prompt.c_str());
    printf("Negative prompt: %s\n", args.negative_prompt.c_str());
    printf("CFG scale: %.3f\n", args.cfg_scale);
    printf("Size: %dx%d, Steps: %d, Seed: %llu, RNG: %s\n", args.width, args.height, args.steps, args.seed,
           args.rng_mode == 0 ? "pytorch" : "sdcpp");

    bool use_cfg = std::fabs(args.cfg_scale - 1.0f) > 1e-6f;
    if (args.cfg_scale <= 0.0f) {
        fprintf(stderr, "Error: --cfg-scale must be > 0\n");
        return 1;
    }
    if (!use_cfg && !args.negative_prompt.empty()) {
        printf("Note: --negative-prompt is ignored when --cfg-scale is 1.0\n");
    }

    // Validate dimensions
    if (args.width % 16 != 0 || args.height % 16 != 0) {
        fprintf(stderr, "Error: width and height must be multiples of 16\n");
        return 1;
    }

    // Initialize CUDA
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (%.1f GB)\n", prop.name, prop.totalGlobalMem / 1e9);

    init_cublas();

    auto t0 = std::chrono::high_resolution_clock::now();

    // ========================================
    // Phase 1: T5 text encoding
    // ========================================
    printf("\n--- Phase 1: T5 Text Encoding ---\n");

    // Load tokenizer
    T5Tokenizer tokenizer;
    tokenizer.load(args.tokenizer_path);

    int context_len = 512;

    // Positive prompt tokenize
    std::vector<int> tokens = tokenizer.encode(args.prompt);
    int n_real_tokens = std::min((int)tokens.size(), context_len);  // before padding/truncation
    if ((int)tokens.size() > context_len) {
        tokens.resize(context_len);
    }
    printf("Tokenized (+): %d tokens:", n_real_tokens);
    for (int i = 0; i < std::min<int>((int)tokens.size(), 16); i++) printf(" %d", tokens[i]);
    if ((int)tokens.size() > 16) printf(" ...");
    printf("\n");

    // Pad to context length
    tokenizer.pad(tokens, context_len);

    // Negative/unconditioned tokenize (used only when CFG is enabled)
    std::vector<int> tokens_uncond;
    int n_real_tokens_uncond = 0;
    if (use_cfg) {
        tokens_uncond = tokenizer.encode(args.negative_prompt);
        n_real_tokens_uncond = std::min((int)tokens_uncond.size(), context_len);
        if ((int)tokens_uncond.size() > context_len) {
            tokens_uncond.resize(context_len);
        }
        printf("Tokenized (-): %d tokens:", n_real_tokens_uncond);
        for (int i = 0; i < std::min<int>((int)tokens_uncond.size(), 16); i++) printf(" %d", tokens_uncond[i]);
        if ((int)tokens_uncond.size() > 16) printf(" ...");
        printf("\n");
        tokenizer.pad(tokens_uncond, context_len);
    }

    // Load T5 model
    SafetensorsFile t5_file;
    if (!t5_file.open(args.t5_path.c_str())) {
        fprintf(stderr, "Failed to open T5 model: %s\n", args.t5_path.c_str());
        return 1;
    }

    T5Encoder t5;
    t5.load(t5_file);

    // Run T5 encoder
    auto t1 = std::chrono::high_resolution_clock::now();
    Tensor context = t5.forward(tokens);  // No T5 attention mask (matches ComfyUI)
    Tensor context_uncond;
    if (use_cfg) {
        context_uncond = t5.forward(tokens_uncond);  // unconditioned/negative context
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("T5 encoding done in %.1f ms\n",
           std::chrono::duration<float, std::milli>(t2 - t1).count());
    context.print_info("T5 output");
    if (use_cfg) {
        context_uncond.print_info("T5 output (uncond)");
    }

    // Print T5 output checksum and dump to file
    if (args.debug) {
        int64_t n = context.numel();
        std::vector<float> v(n);
        CHECK_CUDA(cudaMemcpy(v.data(), context.f32(), n * sizeof(float), cudaMemcpyDeviceToHost));
        double sum = 0, abs_sum = 0;
        for (int64_t i = 0; i < n; i++) {
            sum += v[i];
            abs_sum += fabsf(v[i]);
        }
        printf("T5 output checksum: %.6f, mean_abs: %.12f\n", sum, abs_sum / n);

        FILE* f = fopen("/tmp/t5_context_f32.bin", "wb");
        if (f) { fwrite(v.data(), sizeof(float), n, f); fclose(f); printf("Dumped T5 context to /tmp/t5_context_f32.bin\n"); }

        if (use_cfg) {
            int64_t n2 = context_uncond.numel();
            std::vector<float> v2(n2);
            CHECK_CUDA(cudaMemcpy(v2.data(), context_uncond.f32(), n2 * sizeof(float), cudaMemcpyDeviceToHost));
            FILE* f2 = fopen("/tmp/t5_context_uncond_f32.bin", "wb");
            if (f2) { fwrite(v2.data(), sizeof(float), n2, f2); fclose(f2); printf("Dumped uncond T5 context to /tmp/t5_context_uncond_f32.bin\n"); }
        }
    }

    // Free T5 weights, release GPU pool, and close T5 file
    t5.free_weights();
    gpu_pool().release_all();  // Return pooled T5 buffers to CUDA before loading Chroma
    t5_file.~SafetensorsFile();  // Release mmap
    new (&t5_file) SafetensorsFile();  // Reset to empty state

    {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("T5 weights freed. GPU memory: %.1f GB free / %.1f GB total\n",
               free_mem / 1e9, total_mem / 1e9);
    }

    // ========================================
    // Phase 2: Load Chroma model
    // ========================================
    printf("\n--- Phase 2: Loading Chroma Radiance ---\n");

    SafetensorsFile chroma_file;
    if (!chroma_file.open(args.model_path.c_str())) {
        fprintf(stderr, "Failed to open Chroma model: %s\n", args.model_path.c_str());
        return 1;
    }

    ChromaRadiance chroma;
    chroma.debug_diag = args.debug;
    chroma.load(chroma_file);

    if (chroma.is_int8) {
        printf("Weight precision: INT8 (quantized)\n");
    }

    // Release chroma mmap — all weights are now on GPU
    chroma_file.~SafetensorsFile();
    new (&chroma_file) SafetensorsFile();

    // ========================================
    // Phase 3: Precompute
    // ========================================
    printf("\n--- Phase 3: Precompute ---\n");

    // RoPE positional embeddings
    std::vector<int> axes_dim = {16, 56, 56};
    auto pe_vec = RoPE::gen_pe(args.height, args.width, ChromaRadiance::patch_size,
                                context_len, 10000, axes_dim);

    int h_patches = args.height / ChromaRadiance::patch_size;
    int w_patches = args.width / ChromaRadiance::patch_size;
    int img_tokens = h_patches * w_patches;
    int total_tokens = context_len + img_tokens;
    int pe_feat = 64 * 4; // d_head/2 * 4

    // Upload PE to GPU
    Tensor pe = Tensor::alloc({(int64_t)total_tokens, (int64_t)pe_feat}, DType::F32, true);
    CHECK_CUDA(cudaMemcpy(pe.data, pe_vec.data(), pe_vec.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // DCT features for NeRF decoder
    auto dct_features = compute_dct_features(ChromaRadiance::patch_size,
                                              ChromaRadiance::nerf_max_freqs);
    printf("DCT features: %zu values\n", dct_features.size());

    // Attention mask: block attending to padding tokens in T5 context
    // Matches sd.cpp's chroma_use_dit_mask=true (default)
    // mask[i] = 0.0 for attend, -inf for block
    // Layout: [context_len + img_tokens] = [txt_real: 0, txt_pad+1: 0, txt_pad_rest: -inf, img: 0]
    // mask_pad=1: allow 1 extra padding token after real text (matches sd.cpp chroma_t5_mask_pad=1)
    int mask_pad = 1;
    auto make_attn_mask = [&](int real_tokens, const char* label) -> Tensor {
        std::vector<float> attn_mask_vec(total_tokens, 0.0f);
        if (!args.no_mask) {
            int start_block = std::min(real_tokens + mask_pad, context_len);
            for (int i = start_block; i < context_len; i++) {
                attn_mask_vec[i] = -HUGE_VALF;
            }
            printf("Attention mask (%s): %d real tokens + %d mask_pad, %d blocked, %d image tokens\n",
                   label, real_tokens, mask_pad, context_len - start_block, img_tokens);
        } else {
            printf("Attention mask (%s): DISABLED (--no-mask)\n", label);
        }
        Tensor attn_mask = Tensor::alloc({(int64_t)total_tokens}, DType::F32, true);
        CHECK_CUDA(cudaMemcpy(attn_mask.data, attn_mask_vec.data(),
                              total_tokens * sizeof(float), cudaMemcpyHostToDevice));
        return attn_mask;
    };

    Tensor attn_mask = make_attn_mask(n_real_tokens, "+");
    Tensor attn_mask_uncond;
    if (use_cfg) {
        attn_mask_uncond = make_attn_mask(n_real_tokens_uncond, "-");
    }

    // ========================================
    // Phase 4: Sampling
    // ========================================
    printf("\n--- Phase 4: Sampling (%d steps) ---\n", args.steps);

    EulerSampler sampler(args.steps);
    sampler.print_schedule();

    // Initialize noise
    Tensor x = sampler.init_noise(3, args.height, args.width, args.seed, args.rng_mode);
    printf("Initialized noise: [1, 3, %d, %d]\n", args.height, args.width);

    // Dump noise to file for comparison
    if (args.debug) {
        int64_t n = x.numel();
        std::vector<float> v(n);
        CHECK_CUDA(cudaMemcpy(v.data(), x.f32(), n * sizeof(float), cudaMemcpyDeviceToHost));
        FILE* f = fopen("/tmp/noise_f32.bin", "wb");
        if (f) { fwrite(v.data(), sizeof(float), n, f); fclose(f); printf("Dumped noise to /tmp/noise_f32.bin\n"); }
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < args.steps; step++) {
        float sigma = sampler.get_sigma(step);
        auto step_start = std::chrono::high_resolution_clock::now();

        if (use_cfg) {
            // Two-pass CFG: guided = uncond + cfg * (cond - uncond)
            Tensor velocity_cond = chroma.forward(x, context, sigma, pe, dct_features, attn_mask.f32());
            Tensor velocity_uncond = chroma.forward(x, context_uncond, sigma, pe, dct_features, attn_mask_uncond.f32());
            int64_t n = velocity_cond.numel();
            add_scaled_cuda(velocity_cond.f32(), velocity_uncond.f32(), velocity_cond.f32(), -1.0f, n);
            scale_cuda(velocity_cond.f32(), args.cfg_scale, n);
            add_cuda(velocity_cond.f32(), velocity_uncond.f32(), velocity_cond.f32(), n);

            // Diagnostic: print guided velocity stats for step 0
            if (args.debug && step == 0) {
                std::vector<float> v(n);
                CHECK_CUDA(cudaMemcpy(v.data(), velocity_cond.f32(), n * sizeof(float), cudaMemcpyDeviceToHost));
                double sum = 0, abs_sum = 0;
                for (int64_t i = 0; i < n; i++) { sum += v[i]; abs_sum += fabs(v[i]); }
                printf("[DIAG] velocity (CFG): sum=%.6f, mean=%.6f, mean_abs=%.6f, first10=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                       sum, sum/n, abs_sum/n, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);

                FILE* f = fopen("/tmp/velocity_f32.bin", "wb");
                if (f) { fwrite(v.data(), sizeof(float), n, f); fclose(f); printf("Dumped velocity to /tmp/velocity_f32.bin\n"); }
            }

            sampler.step(x, velocity_cond, step);
        } else {
            // Single conditioned pass
            Tensor velocity = chroma.forward(x, context, sigma, pe, dct_features, attn_mask.f32());

            if (args.debug && step == 0) {
                int64_t n = velocity.numel();
                std::vector<float> v(n);
                CHECK_CUDA(cudaMemcpy(v.data(), velocity.f32(), n * sizeof(float), cudaMemcpyDeviceToHost));
                double sum = 0, abs_sum = 0;
                for (int64_t i = 0; i < n; i++) { sum += v[i]; abs_sum += fabs(v[i]); }
                printf("[DIAG] velocity: sum=%.6f, mean=%.6f, mean_abs=%.6f, first10=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                       sum, sum/n, abs_sum/n, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);

                FILE* f = fopen("/tmp/velocity_f32.bin", "wb");
                if (f) { fwrite(v.data(), sizeof(float), n, f); fclose(f); printf("Dumped velocity to /tmp/velocity_f32.bin\n"); }
            }

            sampler.step(x, velocity, step);
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        auto step_end = std::chrono::high_resolution_clock::now();
        float step_ms = std::chrono::duration<float, std::milli>(step_end - step_start).count();
        printf("Step %d/%d (sigma=%.4f): %.1f ms\n", step + 1, args.steps, sigma, step_ms);
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    printf("Sampling done in %.1f ms (%.1f ms/step)\n",
           std::chrono::duration<float, std::milli>(t4 - t3).count(),
           std::chrono::duration<float, std::milli>(t4 - t3).count() / args.steps);

    // ========================================
    // Phase 5: Save output
    // ========================================
    printf("\n--- Phase 5: Output ---\n");
    print_image_stats(x);

    // Denormalize: model outputs [-1, 1], convert to [0, 1] via (x + 1) / 2
    // This matches sd.cpp's process_vae_output_tensor
    {
        int64_t n = x.numel();
        // x = (x + 1) * 0.5
        add_scalar_cuda(x.f32(), 1.0f, n);
        scale_cuda(x.f32(), 0.5f, n);
        printf("After denormalization:\n");
        print_image_stats(x);
    }

    // save_image dispatches to PNG or PPM based on file extension
    save_image(x, args.output.c_str());

    auto t5_end = std::chrono::high_resolution_clock::now();
    printf("\nTotal time: %.1f s\n",
           std::chrono::duration<float>(t5_end - t0).count());

    return 0;
}
