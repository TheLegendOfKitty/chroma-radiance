#pragma once
#include "tensor.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>

// Save [1, 3, H, W] float tensor (values in [0, 1]) as PPM image
static bool save_ppm(const Tensor& img, const char* path) {
    assert(img.ndim == 4 && img.shape[0] == 1 && img.shape[1] == 3);
    int H = img.shape[2], W = img.shape[3];

    Tensor cpu = img.to_cpu();
    float* data = cpu.f32();

    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen"); return false; }

    fprintf(f, "P6\n%d %d\n255\n", W, H);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < 3; c++) {
                float v = data[c * H * W + y * W + x];
                v = std::max(0.0f, std::min(1.0f, v));
                unsigned char byte = (unsigned char)(v * 255.0f + 0.5f);
                fwrite(&byte, 1, 1, f);
            }
        }
    }

    fclose(f);
    printf("Saved image to %s (%dx%d)\n", path, W, H);
    return true;
}

// Load PPM image as [1, 3, H, W] float tensor on CPU (values in [0, 1])
static Tensor load_ppm(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); exit(1); }

    char magic[3];
    int W, H, maxval;
    fscanf(f, "%2s %d %d %d", magic, &W, &H, &maxval);
    fgetc(f); // skip newline

    Tensor img = Tensor::alloc({1, 3, (int64_t)H, (int64_t)W}, DType::F32, false);
    float* data = img.f32();

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < 3; c++) {
                unsigned char byte;
                fread(&byte, 1, 1, f);
                data[c * H * W + y * W + x] = (float)byte / 255.0f;
            }
        }
    }

    fclose(f);
    return img;
}

// Print image statistics
static void print_image_stats(const Tensor& img) {
    Tensor cpu = img.to_cpu();
    float* data = cpu.f32();
    int64_t n = cpu.numel();

    float min_val = data[0], max_val = data[0], sum = 0;
    for (int64_t i = 0; i < n; i++) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
        sum += data[i];
    }
    printf("Image stats: min=%.4f, max=%.4f, mean=%.4f\n", min_val, max_val, sum / n);
}
