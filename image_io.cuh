#pragma once
#include "tensor.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstring>
#include <png.h>

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

// Save [1, 3, H, W] float tensor (values in [0, 1]) as PNG image
static bool save_png(const Tensor& img, const char* path) {
    assert(img.ndim == 4 && img.shape[0] == 1 && img.shape[1] == 3);
    int H = img.shape[2], W = img.shape[3];

    Tensor cpu = img.to_cpu();
    float* data = cpu.f32();

    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen"); return false; }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(f); fprintf(stderr, "png_create_write_struct failed\n"); return false; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); fclose(f); fprintf(stderr, "png_create_info_struct failed\n"); return false; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(f);
        fprintf(stderr, "PNG write error\n");
        return false;
    }

    png_init_io(png, f);

    png_set_IHDR(png, info, W, H, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // Convert CHW planar float to HWC interleaved uint8 row by row
    std::vector<unsigned char> row(W * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < 3; c++) {
                float v = data[c * H * W + y * W + x];
                v = std::max(0.0f, std::min(1.0f, v));
                row[x * 3 + c] = (unsigned char)(v * 255.0f + 0.5f);
            }
        }
        png_write_row(png, row.data());
    }

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(f);
    printf("Saved image to %s (%dx%d)\n", path, W, H);
    return true;
}

// Dispatch to save_png or save_ppm based on file extension
static bool save_image(const Tensor& img, const char* path) {
    std::string p(path);
    size_t dot = p.rfind('.');
    if (dot != std::string::npos) {
        std::string ext = p.substr(dot);
        if (ext == ".ppm") return save_ppm(img, path);
    }
    // Default to PNG for .png or any unrecognized extension
    return save_png(img, path);
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
