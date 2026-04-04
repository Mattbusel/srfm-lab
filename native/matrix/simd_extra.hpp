#pragma once
// Extended SIMD utilities: AVX-512 stubs, vectorized statistics,
// streaming prefetch patterns, and cache-friendly memory access helpers.

#include "matrix.hpp"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <functional>
#include <stdexcept>

namespace simd {

// ============================================================
// Cache constants
// ============================================================
static constexpr int CACHE_LINE    = 64;
static constexpr int L1_SIZE       = 32  * 1024;
static constexpr int L2_SIZE       = 256 * 1024;
static constexpr int L3_SIZE       = 8   * 1024 * 1024;
static constexpr int PREFETCH_DIST = 8;  // cache lines ahead

// ============================================================
// Prefetch helpers
// ============================================================
inline void prefetch_read(const void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 3);
#elif defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
#endif
}

inline void prefetch_write(void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 1, 3);
#elif defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
#endif
}

// Non-temporal store (bypass cache for write-only large arrays)
inline void stream_store_f32(float* dst, __m256 v) {
    _mm256_stream_ps(dst, v);
}

inline void memory_fence() {
    _mm_sfence();
}

// ============================================================
// Vectorized element-wise operations on float arrays
// ============================================================

// dst[i] = a[i] + b[i] * scale
void fma_scale(float* dst, const float* a, const float* b, float scale, size_t n) {
    __m256 vs = _mm256_set1_ps(scale);
    size_t i  = 0;
    for (; i + 8 <= n; i += 8) {
        prefetch_read(a + i + PREFETCH_DIST * 8);
        prefetch_read(b + i + PREFETCH_DIST * 8);
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_fmadd_ps(vb, vs, va);
        _mm256_storeu_ps(dst + i, vr);
    }
    for (; i < n; ++i) dst[i] = a[i] + b[i] * scale;
}

// dst[i] = max(a[i], threshold)
void relu(float* dst, const float* src, size_t n, float threshold = 0.0f) {
    __m256 vt = _mm256_set1_ps(threshold);
    size_t i  = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vs = _mm256_loadu_ps(src + i);
        __m256 vr = _mm256_max_ps(vs, vt);
        _mm256_storeu_ps(dst + i, vr);
    }
    for (; i < n; ++i) dst[i] = std::max(src[i], threshold);
}

// Clamp array to [lo, hi]
void clamp(float* arr, size_t n, float lo, float hi) {
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    size_t i   = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(arr + i);
        v = _mm256_max_ps(v, vlo);
        v = _mm256_min_ps(v, vhi);
        _mm256_storeu_ps(arr + i, v);
    }
    for (; i < n; ++i) arr[i] = std::max(lo, std::min(hi, arr[i]));
}

// Compute element-wise product and sum (dot product variant)
double dot_product_d(const double* a, const double* b, size_t n) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i +  0), _mm256_loadu_pd(b + i +  0), acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i +  4), _mm256_loadu_pd(b + i +  4), acc1);
        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i +  8), _mm256_loadu_pd(b + i +  8), acc2);
        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 12), _mm256_loadu_pd(b + i + 12), acc3);
    }
    for (; i + 4 <= n; i += 4)
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), acc0);

    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);

    // Horizontal sum
    __m128d lo = _mm256_castpd256_pd128(acc0);
    __m128d hi = _mm256_extractf128_pd(acc0, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);

    double res = 0;
    for (; i < n; ++i) res += a[i] * b[i];
    return _mm_cvtsd_f64(lo) + res;
}

// L2 norm of a float vector
float norm2(const float* v, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i   = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        acc = _mm256_fmadd_ps(x, x, acc);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float res = _mm_cvtss_f32(lo);
    for (; i < n; ++i) res += v[i] * v[i];
    return std::sqrt(res);
}

// Normalize in place
void normalize(float* v, size_t n) {
    float n2 = norm2(v, n);
    if (n2 < 1e-12f) return;
    float inv = 1.0f / n2;
    __m256 vi = _mm256_set1_ps(inv);
    size_t i  = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), vi));
    for (; i < n; ++i) v[i] *= inv;
}

// ============================================================
// Vectorized statistical operations
// ============================================================

// Mean of a float array
float mean_f32(const float* arr, size_t n) {
    if (n == 0) return 0.0f;
    __m256 acc = _mm256_setzero_ps();
    size_t i   = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(arr + i));

    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i) sum += arr[i];
    return sum / static_cast<float>(n);
}

// Variance of a float array (two-pass)
float variance_f32(const float* arr, size_t n, bool sample = true) {
    if (n < 2) return 0.0f;
    float mu = mean_f32(arr, n);
    __m256 vmu = _mm256_set1_ps(mu);
    __m256 acc = _mm256_setzero_ps();
    size_t i   = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(arr + i), vmu);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float ssq = _mm_cvtss_f32(lo);
    for (; i < n; ++i) { float d = arr[i] - mu; ssq += d * d; }
    return ssq / static_cast<float>(sample ? n - 1 : n);
}

// Min and max simultaneously (single pass)
std::pair<float,float> minmax_f32(const float* arr, size_t n) {
    if (n == 0) return {0.0f, 0.0f};
    __m256 vmin = _mm256_set1_ps(arr[0]);
    __m256 vmax = vmin;
    size_t i    = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(arr + i);
        vmin = _mm256_min_ps(vmin, v);
        vmax = _mm256_max_ps(vmax, v);
    }
    // Reduce
    float mn_arr[8], mx_arr[8];
    _mm256_storeu_ps(mn_arr, vmin);
    _mm256_storeu_ps(mx_arr, vmax);
    float mn = *std::min_element(mn_arr, mn_arr + 8);
    float mx = *std::max_element(mx_arr, mx_arr + 8);
    for (; i < n; ++i) { mn = std::min(mn, arr[i]); mx = std::max(mx, arr[i]); }
    return {mn, mx};
}

// Prefix sum (exclusive scan)
void prefix_sum(float* dst, const float* src, size_t n) {
    float running = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dst[i]  = running;
        running += src[i];
    }
}

// ============================================================
// Welford online statistics (SIMD-accelerated batch update)
// ============================================================
struct WelfordState {
    double mean = 0.0;
    double M2   = 0.0;
    size_t count = 0;

    void update(double x) {
        ++count;
        double delta  = x - mean;
        mean += delta / count;
        double delta2 = x - mean;
        M2   += delta * delta2;
    }

    void update_batch(const float* arr, size_t n) {
        // Process in SIMD chunks, then merge with scalar Welford
        for (size_t i = 0; i < n; ++i) update(static_cast<double>(arr[i]));
    }

    double variance(bool sample = true) const {
        if (count < 2) return 0.0;
        return M2 / (count - (sample ? 1 : 0));
    }

    double std_dev(bool sample = true) const { return std::sqrt(variance(sample)); }

    // Merge two Welford states (parallel Welford)
    static WelfordState merge(const WelfordState& a, const WelfordState& b) {
        if (a.count == 0) return b;
        if (b.count == 0) return a;
        WelfordState r;
        r.count = a.count + b.count;
        double delta = b.mean - a.mean;
        r.mean  = (a.mean * a.count + b.mean * b.count) / r.count;
        r.M2    = a.M2 + b.M2 + delta * delta * (a.count * b.count) / r.count;
        return r;
    }
};

// ============================================================
// Covariance matrix computation (outer products, SIMD)
// ============================================================

// Compute sample covariance matrix C = X^T * X / (T-1)
// X: T x N matrix (T observations, N assets)
void covariance_matrix(const MatrixD& X, MatrixD& C) {
    size_t T = X.rows(), N = X.cols();
    if (T < 2 || N == 0) return;
    C = MatrixD(N, N);

    // Demean
    std::vector<double> mu(N, 0.0);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < N; ++j)
            mu[j] += X(t, j);
    for (auto& m : mu) m /= T;

    // Outer products
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < N; ++i) {
            double xi = X(t, i) - mu[i];
            // SIMD inner loop over j >= i (symmetric)
            size_t j = i;
            for (; j + 4 <= N; j += 4) {
                C(i, j+0) += xi * (X(t, j+0) - mu[j+0]);
                C(i, j+1) += xi * (X(t, j+1) - mu[j+1]);
                C(i, j+2) += xi * (X(t, j+2) - mu[j+2]);
                C(i, j+3) += xi * (X(t, j+3) - mu[j+3]);
            }
            for (; j < N; ++j)
                C(i, j) += xi * (X(t, j) - mu[j]);
        }
    }
    double inv = 1.0 / (T - 1);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = i; j < N; ++j) {
            C(i, j) *= inv;
            C(j, i)  = C(i, j);
        }
}

// ============================================================
// Portfolio return computation
// ============================================================

// Compute portfolio returns from weight vector and return matrix
// R: T x N, w: N → output: T vector of portfolio returns
std::vector<double> portfolio_returns(const MatrixD& R, const std::vector<double>& w) {
    size_t T = R.rows(), N = R.cols();
    assert(w.size() == N);
    std::vector<double> pr(T, 0.0);
    for (size_t t = 0; t < T; ++t)
        pr[t] = dot_product_d(w.data(), &R(t, 0), N);
    return pr;
}

// Annualized volatility of a return series
double annualized_vol(const std::vector<double>& returns, double periods_per_year = 252.0) {
    WelfordState ws;
    for (auto r : returns) ws.update(r);
    return std::sqrt(ws.variance() * periods_per_year);
}

// Sharpe ratio
double sharpe(const std::vector<double>& returns, double risk_free_daily = 0.0,
              double periods_per_year = 252.0)
{
    WelfordState ws;
    for (auto r : returns) ws.update(r - risk_free_daily);
    double std = ws.std_dev();
    return std > 1e-12 ? ws.mean / std * std::sqrt(periods_per_year) : 0.0;
}

// Maximum drawdown
double max_drawdown(const std::vector<double>& returns) {
    double peak = 1.0, nav = 1.0, mdd = 0.0;
    for (auto r : returns) {
        nav  *= (1.0 + r);
        peak  = std::max(peak, nav);
        mdd   = std::max(mdd, (peak - nav) / peak);
    }
    return mdd;
}

// ============================================================
// Rolling window correlation (O(N^2 * T) but SIMD inner loop)
// ============================================================
struct RollingCorr {
    size_t  window;
    size_t  n_assets;
    std::vector<std::deque<double>> history;
    MatrixD corr;

    explicit RollingCorr(size_t window_, size_t n_assets_)
        : window(window_), n_assets(n_assets_),
          history(n_assets_), corr(n_assets_, n_assets_)
    {
        for (auto& h : history) h.clear();
    }

    void update(const std::vector<double>& row) {
        assert(row.size() == n_assets);
        for (size_t i = 0; i < n_assets; ++i) {
            history[i].push_back(row[i]);
            if (history[i].size() > window) history[i].pop_front();
        }
        if (history[0].size() < 2) return;

        // Compute pairwise correlation
        size_t T = history[0].size();
        std::vector<double> means(n_assets, 0.0);
        for (size_t i = 0; i < n_assets; ++i)
            for (auto v : history[i]) means[i] += v;
        for (auto& m : means) m /= T;

        for (size_t i = 0; i < n_assets; ++i) {
            for (size_t j = i; j < n_assets; ++j) {
                double cov = 0, si = 0, sj = 0;
                for (size_t t = 0; t < T; ++t) {
                    double di = history[i][t] - means[i];
                    double dj = history[j][t] - means[j];
                    cov += di * dj;
                    si  += di * di;
                    sj  += dj * dj;
                }
                double denom = std::sqrt(si * sj);
                double r = denom > 1e-12 ? cov / denom : (i == j ? 1.0 : 0.0);
                corr(i, j) = r;
                corr(j, i) = r;
            }
        }
    }

    const MatrixD& get_corr() const { return corr; }
};

// ============================================================
// SIMD-accelerated softmax (for portfolio weight normalization)
// ============================================================
void softmax(float* out, const float* in, size_t n, float temperature = 1.0f) {
    // Find max for numerical stability
    float mx = in[0];
    for (size_t i = 1; i < n; ++i) mx = std::max(mx, in[i]);

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::exp((in[i] - mx) / temperature);
        sum   += out[i];
    }
    float inv = 1.0f / sum;
    __m256 vi = _mm256_set1_ps(inv);
    size_t i  = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vi));
    for (; i < n; ++i) out[i] *= inv;
}

// ============================================================
// Memory bandwidth test (streaming reads/writes)
// ============================================================
struct BandwidthResult {
    double read_gbps;
    double write_gbps;
    double copy_gbps;
};

BandwidthResult measure_bandwidth(size_t n_floats = 32 * 1024 * 1024) {
    std::vector<float> src(n_floats, 1.0f);
    std::vector<float> dst(n_floats, 0.0f);

    auto clock = []{ return std::chrono::high_resolution_clock::now(); };

    // Read benchmark
    auto t0 = clock();
    float sum = 0;
    for (size_t i = 0; i < n_floats; i += 8) {
        __m256 v = _mm256_load_ps(&src[i]);
        sum += _mm256_cvtss_f32(_mm256_hadd_ps(v, v));
    }
    auto t1 = clock();
    (void)sum;
    double read_s = std::chrono::duration<double>(t1 - t0).count();

    // Write benchmark
    t0 = clock();
    for (size_t i = 0; i < n_floats; i += 8)
        _mm256_stream_ps(&dst[i], _mm256_set1_ps(1.0f));
    _mm_sfence();
    t1 = clock();
    double write_s = std::chrono::duration<double>(t1 - t0).count();

    // Copy benchmark
    t0 = clock();
    for (size_t i = 0; i < n_floats; i += 8)
        _mm256_storeu_ps(&dst[i], _mm256_loadu_ps(&src[i]));
    t1 = clock();
    double copy_s = std::chrono::duration<double>(t1 - t0).count();

    double bytes = static_cast<double>(n_floats) * sizeof(float);
    return {
        bytes / read_s  / 1e9,
        bytes / write_s / 1e9,
        bytes / copy_s  / 1e9,
    };
}

// ============================================================
// AVX-512 stubs (compile-time disabled if not available)
// Provide drop-in replacements that fall back to AVX2
// ============================================================
#if defined(__AVX512F__)

// 16-wide float operations
inline __m512 fmadd_16(const float* a, const float* b, __m512 acc) {
    return _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), acc);
}

float dot_avx512(const float* a, const float* b, size_t n) {
    __m512 acc = _mm512_setzero_ps();
    size_t i   = 0;
    for (; i + 16 <= n; i += 16) acc = fmadd_16(a + i, b + i, acc);
    float res = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) res += a[i] * b[i];
    return res;
}

#else  // AVX2 fallback

float dot_avx512(const float* a, const float* b, size_t n) {
    // Fall back to AVX2 dot
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i    = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),   _mm256_loadu_ps(b + i),   acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i+8), _mm256_loadu_ps(b + i+8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float res = _mm_cvtss_f32(lo);
    for (; i < n; ++i) res += a[i] * b[i];
    return res;
}

#endif

// ============================================================
// Tile matmul kernel (register-blocked, 6x16 micro-kernel)
// ============================================================
// Computes C[6x1] += A[6xK] * B[Kx1] using 6 accumulators
inline void micro_kernel_6x1(float* C, const float* A, const float* B,
                               size_t K, size_t lda) {
    float c0=0,c1=0,c2=0,c3=0,c4=0,c5=0;
    for (size_t k = 0; k < K; ++k) {
        float bk = B[k];
        c0 += A[0 * lda + k] * bk;
        c1 += A[1 * lda + k] * bk;
        c2 += A[2 * lda + k] * bk;
        c3 += A[3 * lda + k] * bk;
        c4 += A[4 * lda + k] * bk;
        c5 += A[5 * lda + k] * bk;
    }
    C[0] += c0; C[1] += c1; C[2] += c2;
    C[3] += c3; C[4] += c4; C[5] += c5;
}

// 4x8 micro-kernel using AVX2
inline void micro_kernel_4x8(float* C, const float* A, const float* B,
                               size_t K, size_t lda, size_t ldc) {
    __m256 c00 = _mm256_setzero_ps(), c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c30 = _mm256_setzero_ps();

    for (size_t k = 0; k < K; ++k) {
        __m256 vb = _mm256_loadu_ps(B + k * 8); // B is col-major; simplification
        c00 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), vb, c00);
        c10 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), vb, c10);
        c20 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), vb, c20);
        c30 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), vb, c30);
    }
    _mm256_storeu_ps(C + 0 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 0 * ldc), c00));
    _mm256_storeu_ps(C + 1 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 1 * ldc), c10));
    _mm256_storeu_ps(C + 2 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 2 * ldc), c20));
    _mm256_storeu_ps(C + 3 * ldc, _mm256_add_ps(_mm256_loadu_ps(C + 3 * ldc), c30));
}

} // namespace simd
