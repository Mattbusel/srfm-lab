#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

// AVX2 support detection
#if defined(__AVX2__)
#  include <immintrin.h>
#  define SRFM_AVX2 1
#endif

#if defined(__SSE2__)
#  include <emmintrin.h>
#  define SRFM_SSE2 1
#endif

namespace srfm {
namespace simd {

// ============================================================
// Scalar fallbacks (always available)
// ============================================================

/// Scalar EMA update: new_ema = alpha * price + (1-alpha) * prev_ema
inline double ema_update_scalar(double price, double prev_ema, double alpha) noexcept {
    return alpha * price + (1.0 - alpha) * prev_ema;
}

/// Fast exp approximation using double arithmetic (Schraudolph 1999 variant).
/// Max relative error ~3.5%. Suitable for non-critical paths.
inline double fast_exp(double x) noexcept {
    // Clamp to avoid overflow
    if (x > 700.0)  return 1e300;
    if (x < -700.0) return 0.0;
    // Use union trick to set IEEE exponent bits
    union { double d; int64_t i; } u;
    u.i = static_cast<int64_t>(6497320848556798LL * x + 4607182418800017408LL);
    return u.d;
}

/// Horizontal sum of an array of doubles (scalar)
inline double hsum_scalar(const double* arr, std::size_t n) noexcept {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += arr[i];
    return sum;
}

/// Dot product (scalar)
inline double dot_scalar(const double* a, const double* b, std::size_t n) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// ============================================================
// AVX2 implementations (8 doubles at a time)
// ============================================================

#if defined(SRFM_AVX2)

/// Vectorized EMA update for 4 instruments simultaneously using AVX2.
/// prices[4], emas[4] are updated in place.
/// alpha is broadcast as a scalar.
inline void ema_update_avx2_4(const double* prices, double* emas,
                               double alpha) noexcept {
    __m256d vp    = _mm256_loadu_pd(prices);
    __m256d ve    = _mm256_loadu_pd(emas);
    __m256d va    = _mm256_set1_pd(alpha);
    __m256d v1ma  = _mm256_set1_pd(1.0 - alpha);
    __m256d res   = _mm256_fmadd_pd(va, vp, _mm256_mul_pd(v1ma, ve));
    _mm256_storeu_pd(emas, res);
}

/// Vectorized EMA update for N instruments (AVX2, 4-wide, with scalar tail).
inline void ema_update_batch_avx2(const double* __restrict__ prices,
                                   double* __restrict__ emas,
                                   std::size_t n,
                                   double alpha) noexcept {
    const __m256d va   = _mm256_set1_pd(alpha);
    const __m256d v1ma = _mm256_set1_pd(1.0 - alpha);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vp  = _mm256_loadu_pd(prices + i);
        __m256d ve  = _mm256_loadu_pd(emas   + i);
        __m256d res = _mm256_fmadd_pd(va, vp, _mm256_mul_pd(v1ma, ve));
        _mm256_storeu_pd(emas + i, res);
    }
    // Scalar tail
    for (; i < n; ++i)
        emas[i] = alpha * prices[i] + (1.0 - alpha) * emas[i];
}

/// Horizontal sum of AVX2 register (4 doubles → 1 double)
inline double hsum_avx2(__m256d v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    __m128d sh = _mm_unpackhi_pd(s, s);
    return _mm_cvtsd_f64(_mm_add_sd(s, sh));
}

/// Vectorized dot product (AVX2)
inline double dot_avx2(const double* __restrict__ a,
                        const double* __restrict__ b,
                        std::size_t n) noexcept {
    __m256d acc = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        acc = _mm256_fmadd_pd(va, vb, acc);
    }
    double result = hsum_avx2(acc);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

/// Vectorized multiply-add: out[i] = a[i] * b + c[i]
inline void fmadd_batch_avx2(const double* __restrict__ a,
                               double b,
                               const double* __restrict__ c,
                               double* __restrict__ out,
                               std::size_t n) noexcept {
    __m256d vb = _mm256_set1_pd(b);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va  = _mm256_loadu_pd(a + i);
        __m256d vc  = _mm256_loadu_pd(c + i);
        __m256d res = _mm256_fmadd_pd(va, vb, vc);
        _mm256_storeu_pd(out + i, res);
    }
    for (; i < n; ++i) out[i] = a[i] * b + c[i];
}

/// Batch EMA update for multiple alphas (different instruments may have different alphas)
inline void ema_update_multi_alpha_avx2(const double* __restrict__ prices,
                                         double* __restrict__ emas,
                                         const double* __restrict__ alphas,
                                         std::size_t n) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vp   = _mm256_loadu_pd(prices + i);
        __m256d ve   = _mm256_loadu_pd(emas   + i);
        __m256d va   = _mm256_loadu_pd(alphas + i);
        __m256d v1   = _mm256_set1_pd(1.0);
        __m256d v1ma = _mm256_sub_pd(v1, va);
        __m256d res  = _mm256_fmadd_pd(va, vp, _mm256_mul_pd(v1ma, ve));
        _mm256_storeu_pd(emas + i, res);
    }
    for (; i < n; ++i)
        emas[i] = alphas[i] * prices[i] + (1.0 - alphas[i]) * emas[i];
}

#else // No AVX2 — use scalar implementations

inline void ema_update_batch_avx2(const double* prices, double* emas,
                                   std::size_t n, double alpha) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        emas[i] = ema_update_scalar(prices[i], emas[i], alpha);
}

inline void ema_update_multi_alpha_avx2(const double* prices, double* emas,
                                         const double* alphas, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        emas[i] = alphas[i] * prices[i] + (1.0 - alphas[i]) * emas[i];
}

inline double dot_avx2(const double* a, const double* b, std::size_t n) noexcept {
    return dot_scalar(a, b, n);
}

inline void fmadd_batch_avx2(const double* a, double b, const double* c,
                               double* out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b + c[i];
}

#endif // SRFM_AVX2

// ============================================================
// Runtime dispatch wrappers (use best available ISA)
// ============================================================

/// EMA update for a batch of N instruments.
inline void ema_update_batch(const double* prices, double* emas,
                              std::size_t n, double alpha) noexcept {
#if defined(SRFM_AVX2)
    ema_update_batch_avx2(prices, emas, n, alpha);
#else
    for (std::size_t i = 0; i < n; ++i)
        emas[i] = ema_update_scalar(prices[i], emas[i], alpha);
#endif
}

/// Dot product dispatcher
inline double dot(const double* a, const double* b, std::size_t n) noexcept {
#if defined(SRFM_AVX2)
    return dot_avx2(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
}

/// Horizontal sum dispatcher
inline double hsum(const double* arr, std::size_t n) noexcept {
#if defined(SRFM_AVX2)
    __m256d acc = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4)
        acc = _mm256_add_pd(acc, _mm256_loadu_pd(arr + i));
    double r = hsum_avx2(acc);
    for (; i < n; ++i) r += arr[i];
    return r;
#else
    return hsum_scalar(arr, n);
#endif
}

} // namespace simd
} // namespace srfm
