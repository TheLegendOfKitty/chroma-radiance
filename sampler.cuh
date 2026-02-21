#pragma once
#include "tensor.cuh"
#include <vector>
#include <cmath>

// Forward declarations
void add_scaled_cuda(const float* x, const float* delta, float* out, float scale, int64_t n);
void rand_normal_cuda(float* output, int64_t n, unsigned long long seed);
void rand_normal_sdcpp_cuda(float* output, int64_t n, unsigned long long seed);

// CPU implementation of sd.cpp's PhiloxRNG::randn (double-precision Box-Muller)
// This produces bit-identical results to sd.cpp's rng_philox.hpp
static std::vector<float> philox_randn_cpu(uint64_t seed, uint32_t n) {
    const uint32_t philox_m0 = 0xD2511F53u, philox_m1 = 0xCD9E8D57u;
    const uint32_t philox_w0 = 0x9E3779B9u, philox_w1 = 0xBB67AE85u;
    const float two_pow32_inv = 2.3283064e-10f;
    const float two_pow32_inv_2pi = 2.3283064e-10f * 6.2831855f;

    // Key from seed
    uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFF);
    uint32_t k1 = (uint32_t)(seed >> 32);

    std::vector<float> result(n);
    uint32_t offset = 0;

    // For each element: counter = (offset, 0, i, 0)
    for (uint32_t i = 0; i < n; i++) {
        uint32_t ctr[4] = {offset, 0, i, 0};
        uint32_t key[2] = {k0, k1};

        // 10 rounds of Philox
        for (int r = 0; r < 10; r++) {
            uint64_t p0 = (uint64_t)philox_m0 * (uint64_t)ctr[0];
            uint64_t p1 = (uint64_t)philox_m1 * (uint64_t)ctr[2];
            uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
            uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
            ctr[0] = hi1 ^ ctr[1] ^ key[0];
            ctr[1] = lo1;
            ctr[2] = hi0 ^ ctr[3] ^ key[1];
            ctr[3] = lo0;
            key[0] += philox_w0;
            key[1] += philox_w1;
        }

        // Box-Muller (matching sd.cpp: uses implicit double promotion via log/sqrt/sin)
        float u = (float)ctr[0] * two_pow32_inv + two_pow32_inv / 2;
        float v = (float)ctr[1] * two_pow32_inv_2pi + two_pow32_inv_2pi / 2;
        float s = sqrt(-2.0f * log(u));
        result[i] = s * sin(v);
    }
    return result;
}

// flux_time_shift: sigma(t) = exp(mu) / (exp(mu) + (1/t - 1)^sigma_param)
// For Chroma: mu=1.0, sigma_param=1.0
// sigma(t) = e / (e + (1/t - 1))
static float flux_time_shift(float t, float mu = 1.0f, float sigma_param = 1.0f) {
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    float e_mu = expf(mu);
    float base = 1.0f / t - 1.0f;
    float shifted = powf(base, sigma_param);
    return e_mu / (e_mu + shifted);
}

// Generate sigma schedule matching sd.cpp's DiscreteScheduler + FluxFlowDenoiser
// Integer timesteps spaced uniformly in [999, 0] with steps points (n-1 intervals),
// mapped via (t+1)/1000, then flux_time_shift. Terminal 0.0 appended.
static std::vector<float> generate_schedule(int steps) {
    const int TIMESTEPS = 1000;
    const float shift = 1.0f;
    int t_max = TIMESTEPS - 1;  // 999

    std::vector<float> sigmas;
    if (steps <= 1) {
        float t_cont = (float)(t_max + 1) / (float)TIMESTEPS;
        sigmas.push_back(flux_time_shift(t_cont, shift));
        sigmas.push_back(0.0f);
        return sigmas;
    }

    float step_size = (float)t_max / (float)(steps - 1);
    for (int i = 0; i < steps; i++) {
        float t_disc = (float)t_max - step_size * (float)i;
        float t_cont = (t_disc + 1.0f) / (float)TIMESTEPS;
        sigmas.push_back(flux_time_shift(t_cont, shift));
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

// CONST denoiser scalings
// c_skip = 1, c_out = -sigma, c_in = 1
// denoised = c_skip * x + c_out * model_output = x - sigma * model_output
// For velocity prediction in Euler: x_next = x + velocity * (sigma_next - sigma)

struct EulerSampler {
    int steps;
    std::vector<float> sigmas;

    EulerSampler(int steps) : steps(steps) {
        sigmas = generate_schedule(steps);
    }

    // Initialize with random noise scaled by initial sigma
    // rng_mode: 0 = PyTorch-matching (interleaved), 1 = sd.cpp-matching (CPU Philox, exact match)
    Tensor init_noise(int C, int H, int W, unsigned long long seed, int rng_mode = 0) {
        Tensor x = Tensor::alloc({1, (int64_t)C, (int64_t)H, (int64_t)W}, DType::F32, true);
        if (rng_mode == 1) {
            // CPU-based Philox RNG matching sd.cpp's rng_philox.hpp exactly
            // (GPU sinf/logf/sqrtf differ from CPU sin/log/sqrt in last ULPs)
            int64_t n = x.numel();
            std::vector<float> noise = philox_randn_cpu(seed, (uint32_t)n);
            CHECK_CUDA(cudaMemcpy(x.f32(), noise.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            rand_normal_cuda(x.f32(), x.numel(), seed);
        }
        // Scale by initial sigma
        scale_cuda(x.f32(), sigmas[0], x.numel());
        return x;
    }

    // Get timestep for step i (the sigma value)
    float get_sigma(int step) const {
        return sigmas[step];
    }

    // Euler step: x = x + model_output * (sigma_next - sigma_current)
    // model_output is the velocity (for v-prediction)
    // For CONST denoiser: the model IS called with x directly (c_in=1)
    // and outputs velocity
    void step(Tensor& x, const Tensor& model_output, int step_idx) {
        float sigma = sigmas[step_idx];
        float sigma_next = sigmas[step_idx + 1];
        float dt = sigma_next - sigma;

        // x = x + model_output * dt
        add_scaled_cuda(x.f32(), model_output.f32(), x.f32(), dt, x.numel());
    }

    // Print schedule
    void print_schedule() const {
        printf("Sigma schedule (%d steps):\n", steps);
        for (int i = 0; i <= steps; i++) {
            printf("  step %d: sigma = %.6f\n", i, sigmas[i]);
        }
    }
};
