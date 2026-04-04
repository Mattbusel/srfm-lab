#include "matrix.hpp"
#include "simd_extra.hpp"
#include "simd_ops.cpp"
#include "decomposition.cpp"
#include "portfolio_math.cpp"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>
#include <string>

using namespace linalg;

// Timing helper
struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;
    Timer() : start(Clock::now()) {}
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    }
};

static MatrixF random_matrix_f(size_t n, size_t m, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    MatrixF A(n, m);
    for (size_t i = 0; i < n*m; ++i) A.data()[i] = dist(rng);
    return A;
}

static MatrixD random_matrix_d(size_t n, size_t m, uint32_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    MatrixD A(n, m);
    for (size_t i = 0; i < n*m; ++i) A.data()[i] = dist(rng);
    return A;
}

// Make symmetric positive-definite matrix
static MatrixD spd_matrix(size_t n) {
    MatrixD A = random_matrix_d(n, n);
    MatrixD S(n, n, 0.0);
    // S = A^T A + n*I
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n; ++k)
            for (size_t j = 0; j < n; ++j)
                S(i,j) += A(k,i) * A(k,j);
    for (size_t i = 0; i < n; ++i) S(i,i) += n;
    return S;
}

void bench_matmul(size_t M, size_t K, size_t N, int repeats = 5) {
    MatrixF A = random_matrix_f(M, K);
    MatrixF B = random_matrix_f(K, N);

    // Warmup
    auto C = simd::matmul(A, B);

    // SIMD
    Timer t;
    for (int r = 0; r < repeats; ++r) C = simd::matmul(A, B);
    double simd_ms = t.elapsed_ms() / repeats;

    // Naive
    Timer t2;
    for (int r = 0; r < repeats; ++r) C = A.matmul_naive(B);
    double naive_ms = t2.elapsed_ms() / repeats;

    double gflops_simd  = 2.0 * M * K * N / (simd_ms * 1e-3) / 1e9;
    double gflops_naive = 2.0 * M * K * N / (naive_ms * 1e-3) / 1e9;

    std::cout << std::setw(5) << M << "x" << std::setw(4) << K
              << "x" << std::setw(4) << N
              << "  SIMD: " << std::setw(8) << std::fixed << std::setprecision(2)
              << simd_ms << " ms  " << std::setw(6) << std::setprecision(2)
              << gflops_simd << " GFLOPS"
              << "  Naive: " << std::setw(8) << naive_ms << " ms  "
              << std::setw(6) << gflops_naive << " GFLOPS"
              << "  Speedup: " << std::setw(5) << std::setprecision(1)
              << naive_ms / simd_ms << "x"
              << std::endl;
}

void bench_dot(size_t n, int repeats = 100) {
    MatrixF a = random_matrix_f(1, n);
    MatrixF b = random_matrix_f(1, n);

    // Warmup
    volatile float r = simd::dot(a, b);
    (void)r;

    Timer t;
    for (int i = 0; i < repeats; ++i) r = simd::dot(a, b);
    double us = t.elapsed_us() / repeats;

    double gbps = (2.0 * n * 4) / (us * 1e-6) / 1e9;
    std::cout << "  dot(" << n << "): " << us << " us  "
              << gbps << " GB/s" << std::endl;
}

void bench_transpose(size_t n, int repeats = 20) {
    MatrixF A = random_matrix_f(n, n);
    MatrixF B(n, n);

    Timer t;
    for (int r = 0; r < repeats; ++r) B = simd::transpose(A);
    double ms = t.elapsed_ms() / repeats;

    double gbps = (2.0 * n * n * 4) / (ms * 1e-3) / 1e9;
    std::cout << "  transpose(" << n << "x" << n << "): "
              << ms << " ms  " << gbps << " GB/s" << std::endl;
}

void bench_cholesky(size_t n, int repeats = 10) {
    MatrixD S = spd_matrix(n);
    Timer t;
    MatrixD L;
    for (int r = 0; r < repeats; ++r) L = decomp::cholesky(S);
    double ms = t.elapsed_ms() / repeats;
    std::cout << "  Cholesky(" << n << "x" << n << "): " << ms << " ms" << std::endl;
}

void bench_lu(size_t n, int repeats = 10) {
    MatrixD A = spd_matrix(n);
    Timer t;
    for (int r = 0; r < repeats; ++r) {
        MatrixD Acopy = A;
        decomp::lu_inplace(Acopy);
    }
    double ms = t.elapsed_ms() / repeats;
    std::cout << "  LU(" << n << "x" << n << "): " << ms << " ms" << std::endl;
}

void bench_portfolio(size_t n, int repeats = 5) {
    size_t T = n * 5;
    MatrixD returns = random_matrix_d(T, n);
    MatrixD S = portfolio::sample_covariance(returns);
    auto [S_shrunk, alpha, mu_shrink] = portfolio::ledoit_wolf_analytic(S, T);

    std::vector<double> mu_vec(n);
    for (size_t i = 0; i < n; ++i) mu_vec[i] = 0.001 + i * 0.0001;

    Timer t;
    for (int r = 0; r < repeats; ++r) {
        auto frontier = portfolio::efficient_frontier(mu_vec, S_shrunk, 20);
        (void)frontier;
    }
    double ms = t.elapsed_ms() / repeats;
    std::cout << "  EfficientFrontier(n=" << n << ", 20 pts): " << ms << " ms"
              << "  Shrinkage alpha=" << std::setprecision(4) << alpha << std::endl;
}

void verify_correctness() {
    std::cout << "\n=== Correctness Checks ===" << std::endl;

    // 1. Matmul: 4x4 known result
    {
        MatrixF A(2, 3, {1,2,3, 4,5,6});
        MatrixF B(3, 2, {7,8, 9,10, 11,12});
        auto C_simd  = simd::matmul(A, B);
        auto C_naive = A.matmul_naive(B);
        float err = 0.0f;
        for (size_t i=0; i<4; ++i) err += std::fabs(C_simd.data()[i] - C_naive.data()[i]);
        std::cout << "  matmul 2x3 * 3x2 error: " << err << (err < 1e-4f ? " [OK]" : " [FAIL]") << std::endl;
    }

    // 2. Dot product
    {
        MatrixD a(1, 5, {1,2,3,4,5});
        MatrixD b(1, 5, {5,4,3,2,1});
        double d = simd::dot(a, b);
        std::cout << "  dot([1..5],[5..1]) = " << d << (std::fabs(d - 35.0) < 1e-10 ? " [OK]" : " [FAIL]") << std::endl;
    }

    // 3. Cholesky of identity → identity
    {
        MatrixD I = MatrixD::identity(4);
        MatrixD L = decomp::cholesky(I);
        double err = 0.0;
        for (size_t i=0; i<4; ++i) for (size_t j=0; j<4; ++j)
            err += std::fabs(L(i,j) - (i==j ? 1.0 : 0.0));
        std::cout << "  Cholesky(I4) error: " << err << (err < 1e-10 ? " [OK]" : " [FAIL]") << std::endl;
    }

    // 4. LU solve
    {
        MatrixD A(3,3, {2,1,-1, -3,-1,2, -2,1,2});
        std::vector<double> b = {8, -11, -3};
        MatrixD Acopy = A;
        auto P = decomp::lu_inplace(Acopy);
        auto x = decomp::lu_solve(Acopy, P, b);
        // Expected: [2, 3, -1]
        double err = std::fabs(x[0]-2)+std::fabs(x[1]-3)+std::fabs(x[2]+1);
        std::cout << "  LU solve x=[" << x[0] << "," << x[1] << "," << x[2]
                  << "] err=" << err << (err < 1e-8 ? " [OK]" : " [FAIL]") << std::endl;
    }

    // 5. QR decomposition: Q^T Q = I
    {
        MatrixD A = random_matrix_d(5, 5, 99);
        auto [Q, R] = decomp::qr_householder(A);
        // Q^T Q should be ~I
        double err = 0.0;
        for (size_t i=0; i<5; ++i)
            for (size_t j=0; j<5; ++j) {
                double dot = 0.0;
                for (size_t k=0; k<5; ++k) dot += Q(k,i)*Q(k,j);
                err += std::fabs(dot - (i==j ? 1.0 : 0.0));
            }
        std::cout << "  QR Q^TQ error: " << err << (err < 1e-8 ? " [OK]" : " [FAIL]") << std::endl;
    }

    // 6. Risk parity: equal risk contributions
    {
        size_t n = 4;
        MatrixD Sigma = spd_matrix(n);
        // Normalize to correlation-like scale
        for (size_t i=0; i<n; ++i) for (size_t j=0; j<n; ++j)
            Sigma(i,j) /= (n * 10.0);
        auto w = portfolio::risk_parity_weights(Sigma, 2000);
        auto rc = portfolio::risk_contributions(Sigma, w);
        double mean_rc = std::accumulate(rc.begin(), rc.end(), 0.0) / n;
        double err = 0.0;
        for (auto r : rc) err += std::fabs(r - mean_rc);
        std::cout << "  Risk parity RC error: " << err << (err < 0.01 ? " [OK]" : " [FAIL]") << std::endl;
    }
}

int main() {
    std::cout << "=== SIMD Linear Algebra Benchmark ===" << std::endl;
    std::cout << "Platform: AVX2, float32 (8-wide) / float64 (4-wide)\n" << std::endl;

    // Matrix multiply benchmarks
    std::cout << "--- Matrix Multiply (float32) ---" << std::endl;
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};
    for (size_t s : sizes) {
        int reps = (s >= 1024) ? 3 : 10;
        bench_matmul(s, s, s, reps);
    }

    // Rectangular
    std::cout << "\n--- Rectangular Matmul ---" << std::endl;
    bench_matmul(512, 64,  512, 10);
    bench_matmul(1024, 32, 1024, 10);
    bench_matmul(2048, 16, 2048, 5);

    // Dot product
    std::cout << "\n--- Dot Product (float32) ---" << std::endl;
    for (size_t n : {64, 256, 1024, 4096, 65536}) bench_dot(n, 500);

    // Transpose
    std::cout << "\n--- Transpose ---" << std::endl;
    for (size_t n : {64, 256, 512, 1024, 2048}) bench_transpose(n, 20);

    // Decompositions
    std::cout << "\n--- Cholesky ---" << std::endl;
    for (size_t n : {32, 64, 128, 256, 512}) bench_cholesky(n, 10);

    std::cout << "\n--- LU Decomposition ---" << std::endl;
    for (size_t n : {32, 64, 128, 256, 512}) bench_lu(n, 10);

    // Portfolio math
    std::cout << "\n--- Portfolio Math ---" << std::endl;
    for (size_t n : {10, 30, 50, 100}) bench_portfolio(n, 5);

    verify_correctness();

    // ── SIMD Extra benchmarks ─────────────────────────────────────────────────
    std::cout << "\n--- SIMD Extra Ops ---\n";
    {
        const size_t N = 1 << 22; // 4M floats
        std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N);
        std::mt19937 rng(77);
        std::uniform_real_distribution<float> uf(-1, 1);
        for (auto& v : a) v = uf(rng);
        for (auto& v : b) v = uf(rng);

        // dot product
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 10; ++rep) {
            volatile float d = simd::dot_avx512(a.data(), b.data(), N);
            (void)d;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
        double gbps = (double)N * 4 * 2 / ms / 1e6;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  dot_avx512        " << std::setw(8) << ms << " ms  "
                  << std::setw(6) << gbps << " GB/s\n";

        // fma_scale
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 10; ++rep)
            simd::fma_scale(c.data(), a.data(), b.data(), 2.0f, N);
        t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
        gbps = (double)N * 4 * 3 / ms / 1e6;
        std::cout << "  fma_scale         " << std::setw(8) << ms << " ms  "
                  << std::setw(6) << gbps << " GB/s\n";

        // norm2
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 10; ++rep) {
            volatile float n2 = simd::norm2(a.data(), N);
            (void)n2;
        }
        t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
        std::cout << "  norm2             " << std::setw(8) << ms << " ms\n";

        // mean + variance
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 10; ++rep) {
            volatile float m = simd::mean_f32(a.data(), N);
            volatile float v = simd::variance_f32(a.data(), N);
            (void)m; (void)v;
        }
        t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
        std::cout << "  mean+variance     " << std::setw(8) << ms << " ms\n";

        // minmax
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 10; ++rep) {
            auto [mn, mx] = simd::minmax_f32(a.data(), N);
            (void)mn; (void)mx;
        }
        t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
        std::cout << "  minmax            " << std::setw(8) << ms << " ms\n";

        // softmax (N=512)
        std::vector<float> sm_in(512), sm_out(512);
        for (auto& v : sm_in) v = uf(rng);
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 100000; ++rep)
            simd::softmax(sm_out.data(), sm_in.data(), 512, 1.0f);
        t1 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 100000;
        std::cout << "  softmax(512)      " << std::setw(8) << std::setprecision(4) << ms << " ms\n";
    }

    // ── Memory bandwidth ──────────────────────────────────────────────────────
    std::cout << "\n--- Memory Bandwidth ---\n";
    {
        auto bw = simd::measure_bandwidth(1 << 23);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Read:  " << bw.read_gbps  << " GB/s\n";
        std::cout << "  Write: " << bw.write_gbps << " GB/s\n";
        std::cout << "  Copy:  " << bw.copy_gbps  << " GB/s\n";
    }

    // ── Welford parallel merge ────────────────────────────────────────────────
    std::cout << "\n--- Welford Stats ---\n";
    {
        std::mt19937 rng2(88);
        std::normal_distribution<double> nd(5.0, 2.0);
        const int N = 1000000;
        std::vector<float> arr(N);
        for (auto& v : arr) v = static_cast<float>(nd(rng2));

        simd::WelfordState ws;
        auto t0 = std::chrono::high_resolution_clock::now();
        ws.update_batch(arr.data(), arr.size());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  mean=" << ws.mean << "  std=" << ws.std_dev()
                  << "  n=" << ws.count
                  << "  (" << std::setprecision(2) << ms << " ms)\n";
    }

    // ── Rolling correlation ───────────────────────────────────────────────────
    std::cout << "\n--- Rolling Correlation (5 assets, 60-period window) ---\n";
    {
        simd::RollingCorr rc(60, 5);
        std::mt19937 rng3(33);
        std::normal_distribution<double> nd2(0, 0.01);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < 500; ++t) {
            std::vector<double> row(5);
            for (auto& v : row) v = nd2(rng3);
            rc.update(row);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const auto& corr = rc.get_corr();
        std::cout << "  500 updates in " << std::setprecision(2) << ms << " ms\n";
        std::cout << "  corr[0,1]=" << std::setprecision(4) << corr(0,1) << "\n";
    }

    std::cout << "\nBenchmark complete." << std::endl;
    return 0;
}
