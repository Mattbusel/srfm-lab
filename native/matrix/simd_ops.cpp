#include "matrix.hpp"
#include <immintrin.h>
#include <cstring>
#include <cstddef>
#include <stdexcept>
#include <cmath>

namespace linalg {
namespace simd {

// ============================================================
// AVX2 single-precision matrix multiply
// Computes C = A * B  (all row-major, float32)
// Uses 8-wide AVX2 float lanes
// ============================================================
void matmul_avx2(const float* __restrict__ A,
                 const float* __restrict__ B,
                 float*       __restrict__ C,
                 size_t M, size_t K, size_t N)
{
    // Zero output
    std::memset(C, 0, sizeof(float) * M * N);

    // Tiled implementation with 8-wide SIMD inner loop
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 32; // 4 AVX2 vectors of 8
    constexpr size_t TILE_K = 64;

    for (size_t i = 0; i < M; i += TILE_M) {
        size_t imax = std::min(i + TILE_M, M);
        for (size_t j = 0; j < N; j += TILE_N) {
            size_t jmax = std::min(j + TILE_N, N);
            // Accumulator registers for up to TILE_M rows
            // We'll handle up to 4 output rows at a time
            for (size_t ii = i; ii < imax; ++ii) {
                for (size_t jj = j; jj < jmax; jj += 8) {
                    size_t rem = std::min(jmax - jj, (size_t)8);
                    if (rem == 8) {
                        __m256 acc = _mm256_setzero_ps();
                        for (size_t k = 0; k < K; ++k) {
                            __m256 a_val = _mm256_set1_ps(A[ii * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + jj]);
                            acc = _mm256_fmadd_ps(a_val, b_vec, acc);
                        }
                        __m256 c_vec = _mm256_loadu_ps(&C[ii * N + jj]);
                        c_vec = _mm256_add_ps(c_vec, acc);
                        _mm256_storeu_ps(&C[ii * N + jj], c_vec);
                    } else {
                        // Scalar fallback for remainder
                        for (size_t jjj = jj; jjj < jj + rem; ++jjj) {
                            float sum = 0.0f;
                            for (size_t k = 0; k < K; ++k)
                                sum += A[ii * K + k] * B[k * N + jjj];
                            C[ii * N + jjj] += sum;
                        }
                    }
                }
            }
        }
    }
}

// AVX2 matrix multiply (double-precision, 4-wide)
void matmul_avx2_d(const double* __restrict__ A,
                   const double* __restrict__ B,
                   double*       __restrict__ C,
                   size_t M, size_t K, size_t N)
{
    std::memset(C, 0, sizeof(double) * M * N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; j += 4) {
            size_t rem = std::min(N - j, (size_t)4);
            if (rem == 4) {
                __m256d acc = _mm256_setzero_pd();
                for (size_t k = 0; k < K; ++k) {
                    __m256d a_val = _mm256_set1_pd(A[i * K + k]);
                    __m256d b_vec = _mm256_loadu_pd(&B[k * N + j]);
                    acc = _mm256_fmadd_pd(a_val, b_vec, acc);
                }
                __m256d c_vec = _mm256_loadu_pd(&C[i * N + j]);
                c_vec = _mm256_add_pd(c_vec, acc);
                _mm256_storeu_pd(&C[i * N + j], c_vec);
            } else {
                for (size_t jj = j; jj < j + rem; ++jj) {
                    double sum = 0.0;
                    for (size_t k = 0; k < K; ++k)
                        sum += A[i * K + k] * B[k * N + jj];
                    C[i * N + jj] += sum;
                }
            }
        }
    }
}

// Vectorized dot product (float, AVX2)
float dot_avx2(const float* __restrict__ a,
               const float* __restrict__ b,
               size_t n)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);

    // Horizontal sum of acc0
    __m128 lo  = _mm256_castps256_ps128(acc0);
    __m128 hi  = _mm256_extractf128_ps(acc0, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);

    // Scalar tail
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

// Vectorized dot product (double, AVX2)
double dot_avx2_d(const double* __restrict__ a,
                  const double* __restrict__ b,
                  size_t n)
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i),   _mm256_loadu_pd(b+i),   acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+4), _mm256_loadu_pd(b+i+4), acc1);
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    for (; i + 4 <= n; i += 4)
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i), _mm256_loadu_pd(b+i), acc0);

    __m128d lo = _mm256_castpd256_pd128(acc0);
    __m128d hi = _mm256_extractf128_pd(acc0, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    double result = _mm_cvtsd_f64(s);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

// Element-wise add (float, AVX2)
void add_avx2(const float* __restrict__ a,
              const float* __restrict__ b,
              float* __restrict__ c,
              size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(c+i,
            _mm256_add_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i)));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

// Element-wise sub (float, AVX2)
void sub_avx2(const float* __restrict__ a,
              const float* __restrict__ b,
              float* __restrict__ c,
              size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(c+i,
            _mm256_add_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i)));
    }
    for (; i < n; ++i) c[i] = a[i] - b[i];
}

// Element-wise mul (float, AVX2)
void mul_avx2(const float* __restrict__ a,
              const float* __restrict__ b,
              float* __restrict__ c,
              size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(c+i,
            _mm256_mul_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i)));
    }
    for (; i < n; ++i) c[i] = a[i] * b[i];
}

// Scalar multiply (float, AVX2)
void scale_avx2(const float* __restrict__ a,
                float s,
                float* __restrict__ c,
                size_t n)
{
    __m256 sv = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(c+i, _mm256_mul_ps(_mm256_loadu_ps(a+i), sv));
    for (; i < n; ++i) c[i] = a[i] * s;
}

// Cache-oblivious transpose for square power-of-2 (float)
static void transpose_recursive(float* A, float* B,
                                 size_t r0, size_t c0,
                                 size_t r1, size_t c1,
                                 size_t rows, size_t cols,
                                 size_t N)
{
    size_t dr = r1 - r0, dc = c1 - c0;
    constexpr size_t BLOCK = 32;
    if (dr <= BLOCK && dc <= BLOCK) {
        for (size_t i = r0; i < r1; ++i)
            for (size_t j = c0; j < c1; ++j)
                B[j * rows + i] = A[i * cols + j];
        return;
    }
    if (dr >= dc) {
        size_t rm = r0 + dr/2;
        transpose_recursive(A, B, r0, c0, rm, c1, rows, cols, N);
        transpose_recursive(A, B, rm, c0, r1, c1, rows, cols, N);
    } else {
        size_t cm = c0 + dc/2;
        transpose_recursive(A, B, r0, c0, r1, cm, rows, cols, N);
        transpose_recursive(A, B, r0, cm, r1, c1, rows, cols, N);
    }
}

void transpose_avx2(const float* __restrict__ A,
                    float* __restrict__ B,
                    size_t rows, size_t cols)
{
    transpose_recursive(const_cast<float*>(A), B,
                        0, 0, rows, cols, rows, cols, rows);
}

// Double-precision element-wise add/sub/mul
void add_avx2_d(const double* __restrict__ a,
                const double* __restrict__ b,
                double* __restrict__ c, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(c+i, _mm256_add_pd(_mm256_loadu_pd(a+i), _mm256_loadu_pd(b+i)));
    for (; i < n; ++i) c[i] = a[i] + b[i];
}

void sub_avx2_d(const double* __restrict__ a,
                const double* __restrict__ b,
                double* __restrict__ c, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(c+i, _mm256_sub_pd(_mm256_loadu_pd(a+i), _mm256_loadu_pd(b+i)));
    for (; i < n; ++i) c[i] = a[i] - b[i];
}

void mul_avx2_d(const double* __restrict__ a,
                const double* __restrict__ b,
                double* __restrict__ c, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm256_storeu_pd(c+i, _mm256_mul_pd(_mm256_loadu_pd(a+i), _mm256_loadu_pd(b+i)));
    for (; i < n; ++i) c[i] = a[i] * b[i];
}

// ---- Public API wrappers ----

MatrixF matmul(const MatrixF& A, const MatrixF& B) {
    if (A.cols() != B.rows())
        throw std::invalid_argument("matmul: shape mismatch");
    MatrixF C(A.rows(), B.cols());
    matmul_avx2(A.data(), B.data(), C.data(), A.rows(), A.cols(), B.cols());
    return C;
}

MatrixD matmul(const MatrixD& A, const MatrixD& B) {
    if (A.cols() != B.rows())
        throw std::invalid_argument("matmul: shape mismatch");
    MatrixD C(A.rows(), B.cols());
    matmul_avx2_d(A.data(), B.data(), C.data(), A.rows(), A.cols(), B.cols());
    return C;
}

float dot(const MatrixF& a, const MatrixF& b) {
    if (a.size() != b.size()) throw std::invalid_argument("dot: size mismatch");
    return dot_avx2(a.data(), b.data(), a.size());
}
double dot(const MatrixD& a, const MatrixD& b) {
    if (a.size() != b.size()) throw std::invalid_argument("dot: size mismatch");
    return dot_avx2_d(a.data(), b.data(), a.size());
}

MatrixF add(const MatrixF& a, const MatrixF& b) {
    if (!a.is_compatible(b)) throw std::invalid_argument("add: shape mismatch");
    MatrixF c(a.rows(), a.cols());
    add_avx2(a.data(), b.data(), c.data(), a.size());
    return c;
}
MatrixF sub(const MatrixF& a, const MatrixF& b) {
    if (!a.is_compatible(b)) throw std::invalid_argument("sub: shape mismatch");
    MatrixF c(a.rows(), a.cols());
    sub_avx2(a.data(), b.data(), c.data(), a.size());
    return c;
}
MatrixF mul_ewise(const MatrixF& a, const MatrixF& b) {
    if (!a.is_compatible(b)) throw std::invalid_argument("mul: shape mismatch");
    MatrixF c(a.rows(), a.cols());
    mul_avx2(a.data(), b.data(), c.data(), a.size());
    return c;
}
MatrixF scale(const MatrixF& a, float s) {
    MatrixF c(a.rows(), a.cols());
    scale_avx2(a.data(), s, c.data(), a.size());
    return c;
}
MatrixF transpose(const MatrixF& A) {
    MatrixF B(A.cols(), A.rows());
    transpose_avx2(A.data(), B.data(), A.rows(), A.cols());
    return B;
}

MatrixD add(const MatrixD& a, const MatrixD& b) {
    MatrixD c(a.rows(), a.cols());
    add_avx2_d(a.data(), b.data(), c.data(), a.size());
    return c;
}
MatrixD sub(const MatrixD& a, const MatrixD& b) {
    MatrixD c(a.rows(), a.cols());
    sub_avx2_d(a.data(), b.data(), c.data(), a.size());
    return c;
}
MatrixD mul_ewise(const MatrixD& a, const MatrixD& b) {
    MatrixD c(a.rows(), a.cols());
    mul_avx2_d(a.data(), b.data(), c.data(), a.size());
    return c;
}

} // namespace simd
} // namespace linalg
