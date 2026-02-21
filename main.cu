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
    int width = 512;
    int height = 512;
    int steps = 20;
    unsigned long long seed = 42;
    std::string output = "output.ppm";
    std::string model_path = "";
    std::string t5_path = "";
    std::string tokenizer_path = "";
    int rng_mode = 1;  // 0 = pytorch, 1 = sd.cpp (default to sd.cpp for comparison)
    bool no_mask = false;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            args.prompt = argv[++i];
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
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: chroma-radiance [options]\n");
            printf("  -p, --prompt TEXT     Prompt text (default: 'a photo of a cat')\n");
            printf("  -W, --width N         Image width (default: 512)\n");
            printf("  -H, --height N        Image height (default: 512)\n");
            printf("  --steps N             Number of sampling steps (default: 20)\n");
            printf("  --seed N              Random seed (default: 42)\n");
            printf("  -o, --output PATH     Output image path (default: output.ppm)\n");
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
    printf("Size: %dx%d, Steps: %d, Seed: %llu, RNG: %s\n", args.width, args.height, args.steps, args.seed,
           args.rng_mode == 0 ? "pytorch" : "sdcpp");

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

    // Tokenize
    std::vector<int> tokens = tokenizer.encode(args.prompt);
    int n_real_tokens = (int)tokens.size();  // before padding
    printf("Tokenized: %d tokens:", n_real_tokens);
    for (size_t i = 0; i < tokens.size(); i++) printf(" %d", tokens[i]);
    printf("\n");

    // Pad to 512
    int context_len = 512;
    tokenizer.pad(tokens, context_len);

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
    CHECK_CUDA(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("T5 encoding done in %.1f ms\n",
           std::chrono::duration<float, std::milli>(t2 - t1).count());
    context.print_info("T5 output");

    // Print T5 output checksum and dump to file
    {
        int64_t n = context.numel();
        std::vector<float> v(n);
        CHECK_CUDA(cudaMemcpy(v.data(), context.f32(), n * sizeof(float), cudaMemcpyDeviceToHost));
        double sum = 0, abs_sum = 0;
        for (int64_t i = 0; i < n; i++) {
            sum += v[i];
            abs_sum += fabsf(v[i]);
        }
        printf("T5 output checksum: %.6f, mean_abs: %.12f\n", sum, abs_sum / n);

        // Dump T5 context to binary file for comparison
        FILE* f = fopen("/tmp/t5_context_f32.bin", "wb");
        if (f) { fwrite(v.data(), sizeof(float), n, f); fclose(f); printf("Dumped T5 context to /tmp/t5_context_f32.bin\n"); }
    }

    // Free T5 weights and close T5 file
    t5.free_weights();
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
    chroma.load(chroma_file);

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

    printf("Image tokens: %d, Total tokens: %d\n", img_tokens, total_tokens);

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
    std::vector<float> attn_mask_vec(total_tokens, 0.0f);
    if (!args.no_mask) {
        for (int i = n_real_tokens + mask_pad; i < context_len; i++) {
            attn_mask_vec[i] = -HUGE_VALF;
        }
        printf("Attention mask: %d real tokens + %d mask_pad, %d blocked, %d image tokens\n",
               n_real_tokens, mask_pad, context_len - n_real_tokens - mask_pad, img_tokens);
    } else {
        printf("Attention mask: DISABLED (--no-mask)\n");
    }

    Tensor attn_mask = Tensor::alloc({(int64_t)total_tokens}, DType::F32, true);
    CHECK_CUDA(cudaMemcpy(attn_mask.data, attn_mask_vec.data(),
                          total_tokens * sizeof(float), cudaMemcpyHostToDevice));

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
    {
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

        // Forward pass
        Tensor velocity = chroma.forward(x, context, sigma, pe, dct_features, attn_mask.f32());

        // Diagnostic: print velocity stats for step 0
        if (step == 0) {
            int64_t n = velocity.numel();
            std::vector<float> v(n);
            CHECK_CUDA(cudaMemcpy(v.data(), velocity.f32(), n * sizeof(float), cudaMemcpyDeviceToHost));
            double sum = 0, abs_sum = 0;
            for (int64_t i = 0; i < n; i++) { sum += v[i]; abs_sum += fabs(v[i]); }
            printf("[DIAG] velocity: sum=%.6f, mean=%.6f, mean_abs=%.6f, first10=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                   sum, sum/n, abs_sum/n, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);

            // Dump velocity to file
            FILE* f = fopen("/tmp/velocity_f32.bin", "wb");
            if (f) { fwrite(v.data(), sizeof(float), n, f); fclose(f); printf("Dumped velocity to /tmp/velocity_f32.bin\n"); }
        }

        // Euler step
        sampler.step(x, velocity, step);

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

    // save_ppm clamps to [0, 1] internally
    save_ppm(x, args.output.c_str());

    auto t5_end = std::chrono::high_resolution_clock::now();
    printf("\nTotal time: %.1f s\n",
           std::chrono::duration<float>(t5_end - t0).count());

    return 0;
}
