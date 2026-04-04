#include "matrix.hpp"
#include "decomposition.cpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace linalg {
namespace portfolio {

// ============================================================
// Ledoit-Wolf Analytical Shrinkage (fast, O(n^2))
// Shrinks sample covariance toward scaled identity:
//   Sigma* = (1 - alpha) * S + alpha * mu * I
// Oracle approximating shrinkage (OAS) formula
// ============================================================
struct ShrinkageResult {
    MatrixD sigma;          // shrunk covariance matrix
    double  shrinkage;      // optimal alpha in [0,1]
    double  mu;             // target scalar (mean eigenvalue)
};

ShrinkageResult ledoit_wolf_analytic(const MatrixD& S, size_t T) {
    // S = sample covariance (n x n), T = number of observations
    if (!S.is_square()) throw std::invalid_argument("ledoit_wolf: non-square");
    const size_t n = S.rows();

    // Trace of S
    double trace_S = 0.0;
    for (size_t i = 0; i < n; ++i) trace_S += S(i,i);

    // Trace of S^2
    double trace_S2 = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            trace_S2 += S(i,j) * S(j,i);

    double mu = trace_S / n;

    // Ledoit-Wolf analytical formula
    // rho* = min(1, ((n+2)/T) * trace(S^2) + trace(S)^2) / ((T+n+2)/T * (trace(S^2) - trace(S)^2/n))
    double numer = ((double)(n + 2) / T) * trace_S2 + trace_S * trace_S;
    double denom = ((double)(T + n + 2) / T) * (trace_S2 - trace_S * trace_S / n);

    double alpha = 0.0;
    if (std::fabs(denom) > 1e-14) {
        alpha = std::min(1.0, numer / denom);
        alpha = std::max(0.0, alpha);
    }

    // Shrunk estimate
    MatrixD sigma_star(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            sigma_star(i,j) = (1.0 - alpha) * S(i,j);
        }
        sigma_star(i,i) += alpha * mu;
    }

    return {std::move(sigma_star), alpha, mu};
}

// Oracle Approximating Shrinkage (OAS) estimator
// More aggressive shrinkage, better for small T/n
ShrinkageResult oas_shrinkage(const MatrixD& S, size_t T) {
    if (!S.is_square()) throw std::invalid_argument("oas: non-square");
    const size_t n = S.rows();
    const double p = static_cast<double>(n);
    const double t = static_cast<double>(T);

    double trace_S  = 0.0;
    double trace_S2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        trace_S += S(i,i);
        for (size_t j = 0; j < n; ++j)
            trace_S2 += S(i,j) * S(j,i);
    }

    double mu = trace_S / p;

    // OAS shrinkage intensity
    double rho_numer = (1.0 - 2.0/p) * trace_S2 + trace_S * trace_S;
    double rho_denom = (t + 1.0 - 2.0/p) * (trace_S2 - trace_S * trace_S / p);
    double alpha = 0.0;
    if (std::fabs(rho_denom) > 1e-14)
        alpha = std::min(1.0, std::max(0.0, rho_numer / rho_denom));

    MatrixD sigma(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            sigma(i,j) = (1.0 - alpha) * S(i,j) + (i==j ? alpha * mu : 0.0);

    return {std::move(sigma), alpha, mu};
}

// ============================================================
// Compute sample covariance matrix from returns matrix
// returns: T x n matrix of asset returns
// ============================================================
MatrixD sample_covariance(const MatrixD& returns) {
    const size_t T = returns.rows();
    const size_t n = returns.cols();
    if (T < 2) throw std::invalid_argument("covariance: need at least 2 observations");

    // Demean each column
    std::vector<double> means(n, 0.0);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < n; ++j)
            means[j] += returns(t, j);
    for (auto& m : means) m /= T;

    MatrixD X(T, n);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < n; ++j)
            X(t,j) = returns(t,j) - means[j];

    // S = X^T X / (T-1)
    MatrixD S(n, n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i; j < n; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < T; ++t)
                s += X(t,i) * X(t,j);
            S(i,j) = S(j,i) = s / (T - 1);
        }
    return S;
}

// ============================================================
// Risk Parity Weights
// Equalize risk contribution of each asset
// w_i * (Sigma w)_i = constant for all i
// Solved via iterative scaling (CCD)
// ============================================================
std::vector<double> risk_parity_weights(const MatrixD& Sigma,
                                         int max_iter = 1000,
                                         double tol   = 1e-8) {
    if (!Sigma.is_square()) throw std::invalid_argument("risk_parity: non-square");
    const size_t n = Sigma.rows();

    std::vector<double> w(n, 1.0 / n);

    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<double> w_old = w;

        for (size_t i = 0; i < n; ++i) {
            // Portfolio variance with current weights
            double sigma_w_i = 0.0;
            for (size_t j = 0; j < n; ++j)
                sigma_w_i += Sigma(i,j) * w[j];

            // Update weight to equalize risk contribution
            // w_i = sqrt(w_i / sigma_w_i) (approximation)
            if (sigma_w_i > 1e-14 && w[i] > 0)
                w[i] = std::sqrt(w[i] / sigma_w_i);
        }

        // Normalize
        double sum = 0.0;
        for (auto wi : w) sum += wi;
        if (sum < 1e-14) { std::fill(w.begin(), w.end(), 1.0/n); break; }
        for (auto& wi : w) wi /= sum;

        // Check convergence
        double diff = 0.0;
        for (size_t i = 0; i < n; ++i) diff += (w[i] - w_old[i]) * (w[i] - w_old[i]);
        if (std::sqrt(diff) < tol) break;
    }
    return w;
}

// Verify risk parity: returns risk contributions (should all be equal)
std::vector<double> risk_contributions(const MatrixD& Sigma,
                                        const std::vector<double>& w) {
    const size_t n = Sigma.rows();
    std::vector<double> rc(n);

    // Portfolio variance
    double port_var = 0.0;
    std::vector<double> sigma_w(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            sigma_w[i] += Sigma(i,j) * w[j];
    for (size_t i = 0; i < n; ++i) port_var += w[i] * sigma_w[i];

    double port_vol = std::sqrt(port_var > 0 ? port_var : 1e-14);
    for (size_t i = 0; i < n; ++i)
        rc[i] = w[i] * sigma_w[i] / port_vol;
    return rc;
}

// ============================================================
// Efficient Frontier: Mean-Variance Optimization
// Sweep over target returns to trace frontier
// min w^T Sigma w   s.t.  w^T mu = target_ret, sum(w) = 1
// Solved analytically via Lagrangian
// ============================================================
struct EfficientPoint {
    double target_return;
    double portfolio_return;
    double portfolio_vol;
    double sharpe_ratio;
    std::vector<double> weights;
};

std::vector<EfficientPoint> efficient_frontier(
    const std::vector<double>& mu,         // expected returns (n)
    const MatrixD& Sigma,                   // covariance matrix (n x n)
    int num_points = 50,
    double rf_rate = 0.0)
{
    const size_t n = mu.size();
    if (Sigma.rows() != n) throw std::invalid_argument("efficient_frontier: shape mismatch");

    // Invert Sigma (regularized)
    MatrixD S_reg = Sigma;
    double eps = 1e-8;
    for (size_t i = 0; i < n; ++i) S_reg(i,i) += eps;
    MatrixD Sinv = decomp::inverse(S_reg);

    // Compute A, B, C from Lagrangian
    // A = 1^T Sigma^{-1} mu
    // B = mu^T Sigma^{-1} mu
    // C = 1^T Sigma^{-1} 1
    // D = B*C - A^2
    std::vector<double> ones(n, 1.0);
    std::vector<double> Sinv_mu(n, 0.0), Sinv_1(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            Sinv_mu[i] += Sinv(i,j) * mu[j];
            Sinv_1[i]  += Sinv(i,j) * ones[j];
        }

    double A = 0.0, B = 0.0, C = 0.0;
    for (size_t i = 0; i < n; ++i) {
        A += Sinv_1[i] * mu[i];
        B += Sinv_mu[i] * mu[i];
        C += Sinv_1[i];
    }
    double D = B*C - A*A;
    if (std::fabs(D) < 1e-14) D = 1e-14;

    // Min-variance portfolio return
    double mu_min = A / C;
    double mu_max = *std::max_element(mu.begin(), mu.end());

    std::vector<EfficientPoint> frontier;
    frontier.reserve(num_points);

    for (int p = 0; p < num_points; ++p) {
        double target = mu_min + (mu_max - mu_min) * p / (num_points - 1.0);

        // Weights: w* = (1/D) * [ (C*target - A) * Sinv_mu + (B - A*target) * Sinv_1 ]
        std::vector<double> w(n);
        double coeff_mu = (C * target - A) / D;
        double coeff_1  = (B - A * target) / D;
        for (size_t i = 0; i < n; ++i)
            w[i] = coeff_mu * Sinv_mu[i] + coeff_1 * Sinv_1[i];

        // Compute portfolio stats
        double port_ret = 0.0;
        for (size_t i = 0; i < n; ++i) port_ret += w[i] * mu[i];

        double port_var = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                port_var += w[i] * Sigma(i,j) * w[j];

        double port_vol = std::sqrt(std::max(port_var, 0.0));
        double sharpe   = (port_vol > 1e-14) ? (port_ret - rf_rate) / port_vol : 0.0;

        frontier.push_back({target, port_ret, port_vol, sharpe, w});
    }
    return frontier;
}

// ============================================================
// Maximum Sharpe Ratio Portfolio (Tangency portfolio)
// ============================================================
std::vector<double> max_sharpe_weights(
    const std::vector<double>& mu,
    const MatrixD& Sigma,
    double rf_rate = 0.0)
{
    const size_t n = mu.size();
    MatrixD S_reg = Sigma;
    for (size_t i = 0; i < n; ++i) S_reg(i,i) += 1e-8;
    MatrixD Sinv = decomp::inverse(S_reg);

    std::vector<double> excess(n);
    for (size_t i = 0; i < n; ++i) excess[i] = mu[i] - rf_rate;

    // w = Sinv * (mu - rf) / (1^T Sinv (mu - rf))
    std::vector<double> w(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            w[i] += Sinv(i,j) * excess[j];

    double sum = 0.0;
    for (auto wi : w) sum += wi;
    if (std::fabs(sum) > 1e-14)
        for (auto& wi : w) wi /= sum;
    else
        std::fill(w.begin(), w.end(), 1.0/n);

    return w;
}

// ============================================================
// Global Minimum Variance Portfolio
// ============================================================
std::vector<double> gmv_weights(const MatrixD& Sigma) {
    const size_t n = Sigma.rows();
    MatrixD S_reg = Sigma;
    for (size_t i = 0; i < n; ++i) S_reg(i,i) += 1e-8;
    MatrixD Sinv = decomp::inverse(S_reg);

    std::vector<double> ones(n, 1.0);
    std::vector<double> Sinv_1(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            Sinv_1[i] += Sinv(i,j) * ones[j];

    double sum = std::accumulate(Sinv_1.begin(), Sinv_1.end(), 0.0);
    if (std::fabs(sum) < 1e-14) return std::vector<double>(n, 1.0/n);
    for (auto& w : Sinv_1) w /= sum;
    return Sinv_1;
}

} // namespace portfolio
} // namespace linalg
