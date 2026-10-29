#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <immintrin.h>

extern "C" {
    void print_heap_audit(void);
    size_t get_dynamic_allocations_count(void);
    size_t get_dynamic_bytes_requested(void);
    void reset_heap_audit(void);
}

// 120 KB Compile-time static execution arena (30,720 floats = exactly 120 KB)
alignas(64) static float g_smr_scratchpad[30720];
alignas(64) static float g_weights[16384];
alignas(64) static float g_folded_weights[16384];
alignas(64) static float g_gamma[512];
alignas(64) static float g_running_mean[512];
alignas(64) static float g_running_var[512];

// AVX-512 SIMD forward moment recalibration kernel
void avx512_forward_recalibrate(const float* activations, int num_samples, int channels,
                                float* out_mean, float* out_var) {
    for (int c = 0; c < channels; c += 16) {
        __m512 sum = _mm512_setzero_ps();
        __m512 sum_sq = _mm512_setzero_ps();

        for (int i = 0; i < num_samples; ++i) {
            __m512 val = _mm512_loadu_ps(&activations[i * channels + c]);
            sum = _mm512_add_ps(sum, val);
            sum_sq = _mm512_fmadd_ps(val, val, sum_sq);
        }

        __m512 n_inv = _mm512_set1_ps(1.0f / (float)num_samples);
        __m512 mean = _mm512_mul_ps(sum, n_inv);
        __m512 mean_sq = _mm512_mul_ps(mean, mean);
        __m512 var = _mm512_sub_ps(_mm512_mul_ps(sum_sq, n_inv), mean_sq);

        _mm512_storeu_ps(&out_mean[c], mean);
        _mm512_storeu_ps(&out_var[c], var);
    }
}

// AVX-512 Conv2D-BatchNorm static weight folding: \tilde{W} = (\gamma / \sqrt{\sigma^2 + \epsilon}) * W
void avx512_fold_weights(const float* weights, const float* gamma, const float* var,
                         float* out_folded, int total_weights, int channels) {
    const float eps = 1e-5f;
    __m512 eps_vec = _mm512_set1_ps(eps);

    for (int c = 0; c < channels; c += 16) {
        __m512 g = _mm512_loadu_ps(&gamma[c]);
        __m512 v = _mm512_loadu_ps(&var[c]);
        __m512 std_dev = _mm512_sqrt_ps(_mm512_add_ps(v, eps_vec));
        __m512 scale = _mm512_div_ps(g, std_dev);

        // Fold into weights for this channel block
        for (int w = 0; w < total_weights / channels; ++w) {
            int idx = w * channels + c;
            __m512 wt = _mm512_loadu_ps(&weights[idx]);
            __m512 folded = _mm512_mul_ps(wt, scale);
            _mm512_storeu_ps(&out_folded[idx], folded);
        }
    }
}

int main(int argc, char** argv) {
    std::string mode = "recal";
    int K = 20;
    int batch_size = 128;
    int buffer_size = 2000;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--buffer") == 0 && i + 1 < argc) buffer_size = std::stoi(argv[++i]);
    }

    std::cout << "[SMR Zen 5 Runner] AMD Ryzen 9 9900X Execution Engine" << std::endl;
    std::cout << "[CONFIG] Mode: " << mode << " | K: " << K << " | Batch: " << batch_size << std::endl;

    // Initialize statically allocated data
    for (int i = 0; i < 30720; ++i) g_smr_scratchpad[i] = (float)(i % 256) * 0.01f;
    for (int i = 0; i < 16384; ++i) g_weights[i] = 0.05f * ((i % 17) - 8);
    for (int i = 0; i < 512; ++i) {
        g_gamma[i] = 1.0f;
        g_running_mean[i] = 0.0f;
        g_running_var[i] = 1.0f;
    }

    reset_heap_audit();

    auto start = std::chrono::high_resolution_clock::now();

    if (mode == "recal") {
        // Forward moment recalibration across K microbatches without any dynamic allocation
        for (int step = 0; step < K; ++step) {
            avx512_forward_recalibrate(g_smr_scratchpad, batch_size, 128, g_running_mean, g_running_var);
            avx512_fold_weights(g_weights, g_gamma, g_running_var, g_folded_weights, 16384, 128);
        }
    } else {
        // Simulated rehearsal replay step over exemplar buffer
        float replay_acc = 0.0f;
        for (int i = 0; i < buffer_size; ++i) {
            int idx = (i * 13) % 30720;
            replay_acc += g_smr_scratchpad[idx];
        }
        g_smr_scratchpad[0] = replay_acc;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "[RESULT] Elapsed time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "[RESULT] Dynamic heap allocations during loop: " << get_dynamic_allocations_count() << std::endl;
    std::cout << "[RESULT] Dynamic bytes requested: " << get_dynamic_bytes_requested() << std::endl;
    print_heap_audit();

    return 0;
}
