#include "matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

namespace linalg {
namespace decomp {

// ============================================================
// LU Decomposition with partial pivoting
// PA = LU  (in-place, returns permutation)
// Returns: permutation vector P such that A[P[i],:] is the i-th row
// ============================================================
std::vector<int> lu_inplace(MatrixD& A) {
    if (!A.is_square())
        throw std::invalid_argument("LU: non-square matrix");
    const size_t n = A.rows();
    std::vector<int> P(n);
    std::iota(P.begin(), P.end(), 0);

    for (size_t k = 0; k < n; ++k) {
        // Find pivot
        size_t pivot = k;
        double max_val = std::fabs(A(k,k));
        for (size_t i = k+1; i < n; ++i) {
            if (std::fabs(A(i,k)) > max_val) {
                max_val = std::fabs(A(i,k));
                pivot = i;
            }
        }
        if (max_val < 1e-14) continue; // singular or near-singular

        // Swap rows k and pivot
        if (pivot != k) {
            std::swap(P[k], P[pivot]);
            for (size_t j = 0; j < n; ++j)
                std::swap(A(k,j), A(pivot,j));
        }

        // Eliminate below
        for (size_t i = k+1; i < n; ++i) {
            A(i,k) /= A(k,k);
            for (size_t j = k+1; j < n; ++j)
                A(i,j) -= A(i,k) * A(k,j);
        }
    }
    return P;
}

// Solve Ax = b given LU decomposition and permutation P
std::vector<double> lu_solve(const MatrixD& LU,
                              const std::vector<int>& P,
                              const std::vector<double>& b)
{
    const size_t n = LU.rows();
    std::vector<double> x(n), y(n);

    // Apply permutation
    for (size_t i = 0; i < n; ++i) y[i] = b[P[i]];

    // Forward substitution: Ly = Pb
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < i; ++j)
            y[i] -= LU(i,j) * y[j];
    }

    // Back substitution: Ux = y
    for (int i = (int)n-1; i >= 0; --i) {
        x[i] = y[i];
        for (size_t j = i+1; j < n; ++j)
            x[i] -= LU(i,j) * x[j];
        x[i] /= LU(i,i);
    }
    return x;
}

// Matrix inverse via LU
MatrixD inverse(MatrixD A) {
    if (!A.is_square()) throw std::invalid_argument("inverse: non-square");
    const size_t n = A.rows();
    auto P = lu_inplace(A);
    MatrixD inv(n, n);

    for (size_t col = 0; col < n; ++col) {
        std::vector<double> e(n, 0.0);
        e[col] = 1.0;
        auto x = lu_solve(A, P, e);
        for (size_t row = 0; row < n; ++row)
            inv(row, col) = x[row];
    }
    return inv;
}

double determinant(MatrixD A) {
    if (!A.is_square()) throw std::invalid_argument("det: non-square");
    const size_t n = A.rows();
    auto P = lu_inplace(A);

    double det = 1.0;
    for (size_t i = 0; i < n; ++i) det *= A(i,i);

    // Sign from permutation
    int swaps = 0;
    std::vector<int> p2(P);
    for (size_t i = 0; i < n; ++i) {
        while ((size_t)p2[i] != i) {
            std::swap(p2[i], p2[p2[i]]);
            ++swaps;
        }
    }
    return (swaps % 2 == 0) ? det : -det;
}

// ============================================================
// Cholesky Decomposition (for symmetric positive-definite matrices)
// A = L * L^T  (lower triangular L)
// ============================================================
MatrixD cholesky(const MatrixD& A) {
    if (!A.is_square()) throw std::invalid_argument("Cholesky: non-square");
    const size_t n = A.rows();
    MatrixD L(n, n, 0.0);

    for (size_t j = 0; j < n; ++j) {
        double sum = A(j,j);
        for (size_t k = 0; k < j; ++k)
            sum -= L(j,k) * L(j,k);
        if (sum < 0) sum = 0.0; // clamp numerical noise
        L(j,j) = std::sqrt(sum);

        if (L(j,j) < 1e-15) {
            // Not positive definite; regularize
            L(j,j) = 1e-8;
        }

        for (size_t i = j+1; i < n; ++i) {
            double s = A(i,j);
            for (size_t k = 0; k < j; ++k)
                s -= L(i,k) * L(j,k);
            L(i,j) = s / L(j,j);
        }
    }
    return L;
}

// Solve Ax = b via Cholesky L L^T x = b
std::vector<double> cholesky_solve(const MatrixD& L,
                                    const std::vector<double>& b)
{
    const size_t n = L.rows();
    std::vector<double> y(n), x(n);

    // Forward: Ly = b
    for (size_t i = 0; i < n; ++i) {
        y[i] = b[i];
        for (size_t j = 0; j < i; ++j)
            y[i] -= L(i,j) * y[j];
        y[i] /= L(i,i);
    }

    // Backward: L^T x = y
    for (int i = (int)n-1; i >= 0; --i) {
        x[i] = y[i];
        for (size_t j = i+1; j < n; ++j)
            x[i] -= L(j,i) * x[j];
        x[i] /= L(i,i);
    }
    return x;
}

// ============================================================
// QR Decomposition via Householder reflections
// Returns Q and R such that A = Q * R
// ============================================================
struct QRResult {
    MatrixD Q;
    MatrixD R;
};

QRResult qr_householder(MatrixD A) {
    const size_t m = A.rows(), n = A.cols();
    MatrixD Q = MatrixD::identity(m);

    for (size_t k = 0; k < std::min(m-1, n); ++k) {
        // Extract column k below diagonal
        size_t len = m - k;
        std::vector<double> x(len);
        for (size_t i = 0; i < len; ++i) x[i] = A(k+i, k);

        // Compute Householder vector v
        double norm_x = 0.0;
        for (auto v : x) norm_x += v * v;
        norm_x = std::sqrt(norm_x);

        std::vector<double> u = x;
        u[0] += (x[0] >= 0 ? 1.0 : -1.0) * norm_x;

        double norm_u = 0.0;
        for (auto v : u) norm_u += v * v;
        if (norm_u < 1e-14) continue;

        // Scale: v = u / ||u||
        for (auto& v : u) v /= std::sqrt(norm_u);

        // Apply H = I - 2*v*v^T to A (rows k..m, cols k..n)
        // A[k:, k:] = A[k:, k:] - 2*v*(v^T * A[k:, k:])
        for (size_t j = k; j < n; ++j) {
            double dot = 0.0;
            for (size_t i = 0; i < len; ++i) dot += u[i] * A(k+i, j);
            for (size_t i = 0; i < len; ++i) A(k+i, j) -= 2.0 * u[i] * dot;
        }

        // Apply H to Q (accumulate)
        for (size_t j = 0; j < m; ++j) {
            double dot = 0.0;
            for (size_t i = 0; i < len; ++i) dot += u[i] * Q(k+i, j);
            for (size_t i = 0; i < len; ++i) Q(k+i, j) -= 2.0 * u[i] * dot;
        }
    }

    // Q is accumulated transposed; transpose to get actual Q
    MatrixD Qt = Q.transpose();
    return {std::move(Qt), std::move(A)};
}

// Solve linear system Ax = b using QR
std::vector<double> qr_solve(const MatrixD& A_orig,
                               const std::vector<double>& b)
{
    MatrixD A = A_orig;
    auto [Q, R] = qr_householder(A);

    // x = R^{-1} Q^T b
    // Compute c = Q^T b
    size_t m = Q.rows(), n = R.cols();
    std::vector<double> c(m);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j)
            c[i] += Q(j,i) * b[j]; // Q^T

    // Back-substitute R x = c (n x n system)
    std::vector<double> x(n);
    for (int i = (int)n-1; i >= 0; --i) {
        x[i] = c[i];
        for (size_t j = i+1; j < n; ++j)
            x[i] -= R(i,j) * x[j];
        if (std::fabs(R(i,i)) > 1e-14)
            x[i] /= R(i,i);
        else
            x[i] = 0.0;
    }
    return x;
}

// ============================================================
// Eigendecomposition: Power Iteration + Deflation
// Finds largest eigenvalues/vectors iteratively
// Returns eigenvalues in descending order and corresponding eigenvectors
// ============================================================
struct EigenResult {
    std::vector<double>  eigenvalues;
    MatrixD              eigenvectors; // columns are eigenvectors
};

static std::pair<double, std::vector<double>>
power_iteration(const MatrixD& A, const std::vector<double>& init,
                int max_iter = 1000, double tol = 1e-10)
{
    size_t n = A.rows();
    std::vector<double> v = init;

    // Normalize
    double norm = 0.0;
    for (auto x : v) norm += x * x;
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;

    double lambda = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        // w = A * v
        std::vector<double> w(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                w[i] += A(i,j) * v[j];

        // Rayleigh quotient
        double new_lambda = 0.0;
        double ww = 0.0;
        for (size_t i = 0; i < n; ++i) { new_lambda += w[i] * v[i]; ww += w[i]*w[i]; }

        // Normalize w to get new v
        double wn = std::sqrt(ww);
        if (wn < 1e-14) break;
        for (auto& x : w) x /= wn;

        if (std::fabs(new_lambda - lambda) < tol) {
            lambda = new_lambda;
            v = w;
            break;
        }
        lambda = new_lambda;
        v = w;
    }
    return {lambda, v};
}

EigenResult eigendecompose(MatrixD A, int num_eigenvalues = -1) {
    if (!A.is_square()) throw std::invalid_argument("eigen: non-square");
    const size_t n = A.rows();
    int k = (num_eigenvalues <= 0) ? static_cast<int>(n) : std::min((size_t)num_eigenvalues, n);

    EigenResult result;
    result.eigenvalues.resize(k);
    result.eigenvectors = MatrixD(n, k);

    // Seed RNG for initial vectors
    uint64_t rng = 0xdeadbeef12345678ULL;
    auto randf = [&]() -> double {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return static_cast<double>(rng & 0xFFFFFFFFFF) / static_cast<double>(0xFFFFFFFFFFULL) - 0.5;
    };

    for (int ev = 0; ev < k; ++ev) {
        // Random initial vector
        std::vector<double> v0(n);
        for (auto& x : v0) x = randf();

        auto [lambda, vec] = power_iteration(A, v0);
        result.eigenvalues[ev] = lambda;
        for (size_t i = 0; i < n; ++i)
            result.eigenvectors(i, ev) = vec[i];

        // Deflation: A = A - lambda * v * v^T
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                A(i,j) -= lambda * vec[i] * vec[j];
    }
    return result;
}

// Symmetric matrix eigenvalues via QR iteration (more accurate for symmetric)
EigenResult eigen_symmetric(MatrixD A, int max_iter = 200, double tol = 1e-10) {
    if (!A.is_square()) throw std::invalid_argument("eigen_sym: non-square");
    const size_t n = A.rows();

    MatrixD V = MatrixD::identity(n);

    for (int iter = 0; iter < max_iter; ++iter) {
        auto [Q, R] = qr_householder(A);
        // A = R * Q (QR iteration step)
        MatrixD newA(n, n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < n; ++k)
                for (size_t j = 0; j < n; ++j)
                    newA(i,j) += R(i,k) * Q(k,j);
        A = std::move(newA);

        // Accumulate Q
        MatrixD newV(n, n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < n; ++k)
                for (size_t j = 0; j < n; ++j)
                    newV(i,j) += V(i,k) * Q(k,j);
        V = std::move(newV);

        // Check off-diagonal norm
        double off = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (i != j) off += A(i,j) * A(i,j);
        if (std::sqrt(off) < tol) break;
    }

    EigenResult r;
    r.eigenvalues.resize(n);
    r.eigenvectors = V;
    for (size_t i = 0; i < n; ++i) r.eigenvalues[i] = A(i,i);

    // Sort by descending eigenvalue
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
        return r.eigenvalues[a] > r.eigenvalues[b];
    });
    std::vector<double> ev_sorted(n);
    MatrixD evec_sorted(n, n);
    for (size_t i = 0; i < n; ++i) {
        ev_sorted[i] = r.eigenvalues[idx[i]];
        for (size_t j = 0; j < n; ++j)
            evec_sorted(j, i) = V(j, idx[i]);
    }
    r.eigenvalues  = ev_sorted;
    r.eigenvectors = evec_sorted;
    return r;
}

} // namespace decomp
} // namespace linalg
