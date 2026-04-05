#include "risk_parity.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace srfm {

// ============================================================
// RiskParity
// ============================================================

RiskParity::RiskParity(int n_instruments, double per_inst_risk,
                        double max_position, int corr_window) noexcept
    : n_instruments_(std::min(n_instruments, MAX_INSTRUMENTS))
    , per_inst_risk_(per_inst_risk)
    , max_position_(max_position)
    , corr_window_(corr_window)
    , update_count_(0)
{
    vols_.fill(0.15);        // default 15% vol
    returns_.fill(0.0);
    return_ema_.fill(0.0);
    return_var_.fill(0.0001);
    cov_matrix_.fill(0.0);
    corr_matrix_.fill(0.0);
    cov_sum_.fill(0.0);
    ret_counts_.fill(0);
    corr_factors_.fill(1.0);

    // Initialize diagonal of correlation matrix
    for (int i = 0; i < n_instruments_; ++i) {
        corr_matrix_[i * MAX_INSTRUMENTS + i] = 1.0;
    }
}

void RiskParity::update_vol(int symbol_id, double vol) noexcept {
    if (symbol_id < 0 || symbol_id >= n_instruments_) return;
    vols_[symbol_id] = std::max(vol, constants::EPSILON);
}

void RiskParity::update_return(int symbol_id, double log_return) noexcept {
    if (symbol_id < 0 || symbol_id >= n_instruments_) return;

    returns_[symbol_id] = log_return;

    // Welford online update for variance
    int&    cnt = ret_counts_[symbol_id];
    double& ema = return_ema_[symbol_id];
    double& var = return_var_[symbol_id];

    ++cnt;
    double alpha = 2.0 / (std::min(cnt, corr_window_) + 1.0);
    double old_ema = ema;
    ema = alpha * log_return + (1.0 - alpha) * ema;
    // EMA variance
    var = (1.0 - alpha) * (var + alpha * (log_return - old_ema) * (log_return - old_ema));
    if (var < constants::EPSILON) var = constants::EPSILON;

    // Update covariance sums with all other instruments
    for (int j = 0; j < n_instruments_; ++j) {
        if (j == symbol_id) continue;
        double rj    = returns_[j] - return_ema_[j];
        double ri    = log_return   - return_ema_[symbol_id];
        int    flat  = symbol_id * MAX_INSTRUMENTS + j;
        int    flatT = j * MAX_INSTRUMENTS + symbol_id;
        double decay = 1.0 - 2.0 / (corr_window_ + 1.0);
        cov_sum_[flat]  = decay * cov_sum_[flat]  + (1.0 - decay) * ri * rj;
        cov_sum_[flatT] = cov_sum_[flat];
    }

    ++update_count_;

    // Recompute correlation matrix periodically
    if (update_count_ % 100 == 0) {
        update_corr_matrix();
    }
}

void RiskParity::update_corr_matrix() noexcept {
    // Build covariance matrix from cov_sum_
    for (int i = 0; i < n_instruments_; ++i) {
        cov_matrix_[i * MAX_INSTRUMENTS + i] = return_var_[i];
        for (int j = i + 1; j < n_instruments_; ++j) {
            double cv = cov_sum_[i * MAX_INSTRUMENTS + j];
            cov_matrix_[i * MAX_INSTRUMENTS + j] = cv;
            cov_matrix_[j * MAX_INSTRUMENTS + i] = cv;
        }
    }

    // Convert to correlation
    cov_to_corr(cov_matrix_.data(), corr_matrix_.data(), n_instruments_);

    // Compute per-instrument average correlation with portfolio
    for (int i = 0; i < n_instruments_; ++i) {
        double sum_corr = 0.0;
        int    cnt      = 0;
        for (int j = 0; j < n_instruments_; ++j) {
            if (j == i) continue;
            double c = corr_matrix_[i * MAX_INSTRUMENTS + j];
            sum_corr += std::abs(c);
            ++cnt;
        }
        double avg = (cnt > 0) ? sum_corr / cnt : 0.0;
        // Higher average correlation → reduce position (less diversification benefit)
        corr_factors_[i] = 1.0 / (1.0 + avg);
    }
}

void RiskParity::compute_positions(PositionSize* out, int n_out) const noexcept {
    int n = std::min(n_out, n_instruments_);
    for (int i = 0; i < n; ++i) {
        double vol     = std::max(vols_[i], constants::EPSILON);
        double budget  = per_inst_risk_ / vol;
        double cf      = corr_factors_[i];
        double raw     = budget * cf;
        double final_  = std::min(raw, max_position_);
        out[i] = { i, raw, final_, budget, cf };
    }
}

double RiskParity::correlation(int i, int j) const noexcept {
    if (i < 0 || j < 0 || i >= n_instruments_ || j >= n_instruments_) return 0.0;
    return corr_matrix_[i * MAX_INSTRUMENTS + j];
}

double RiskParity::avg_pairwise_correlation() const noexcept {
    double sum = 0.0;
    int    cnt = 0;
    for (int i = 0; i < n_instruments_; ++i)
        for (int j = i + 1; j < n_instruments_; ++j) {
            sum += std::abs(corr_matrix_[i * MAX_INSTRUMENTS + j]);
            ++cnt;
        }
    return (cnt > 0) ? sum / cnt : 0.0;
}

double RiskParity::corr_factor(int i) const noexcept {
    if (i < 0 || i >= n_instruments_) return 1.0;
    return corr_factors_[i];
}

void RiskParity::reset() noexcept {
    vols_.fill(0.15);
    returns_.fill(0.0);
    return_ema_.fill(0.0);
    return_var_.fill(0.0001);
    cov_sum_.fill(0.0);
    corr_matrix_.fill(0.0);
    cov_matrix_.fill(0.0);
    ret_counts_.fill(0);
    corr_factors_.fill(1.0);
    update_count_ = 0;
    for (int i = 0; i < n_instruments_; ++i)
        corr_matrix_[i * MAX_INSTRUMENTS + i] = 1.0;
}

// ============================================================
// Matrix utilities
// ============================================================

void mat_mul(const double* A, const double* B, double* C, int n) noexcept {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

void cov_to_corr(const double* cov, double* corr, int n) noexcept {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double vi = cov[i * n + i];
            double vj = cov[j * n + j];
            double denom = std::sqrt(std::max(vi, 1e-15) * std::max(vj, 1e-15));
            corr[i * n + j] = cov[i * n + j] / denom;
        }
    // Enforce diagonal = 1
    for (int i = 0; i < n; ++i) corr[i * n + i] = 1.0;
}

int mat_eigenvalues_sym(const double* A_in, double* out_eigenvalues,
                         int n, int max_iter, double tol) noexcept {
    // Jacobi iteration for symmetric matrix
    // Copy A to working buffer (stack allocation for n <= 25)
    double a[MAX_INSTRUMENTS * MAX_INSTRUMENTS];
    std::memcpy(a, A_in, sizeof(double) * n * n);

    for (int i = 0; i < n; ++i) out_eigenvalues[i] = a[i * n + i];

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        // Find largest off-diagonal element
        double max_val = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double v = std::abs(a[i * n + j]);
                if (v > max_val) { max_val = v; p = i; q = j; }
            }
        if (max_val < tol) break;

        // Compute Givens rotation angle
        double app = a[p * n + p], aqq = a[q * n + q], apq = a[p * n + q];
        double tau = (aqq - app) / (2.0 * apq);
        double t   = (tau >= 0.0) ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                  : 1.0 / (tau - std::sqrt(1.0 + tau * tau));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;

        // Apply rotation
        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = a[q * n + p] = 0.0;
        for (int r = 0; r < n; ++r) {
            if (r == p || r == q) continue;
            double arp = a[r * n + p], arq = a[r * n + q];
            a[r * n + p] = a[p * n + r] = c * arp - s * arq;
            a[r * n + q] = a[q * n + r] = s * arp + c * arq;
        }
    }

    for (int i = 0; i < n; ++i) out_eigenvalues[i] = a[i * n + i];
    return iter;
}

double mat_condition_number(const double* A, int n) noexcept {
    double eigenvalues[MAX_INSTRUMENTS];
    mat_eigenvalues_sym(A, eigenvalues, n);
    double max_ev = constants::EPSILON, min_ev = 1e300;
    for (int i = 0; i < n; ++i) {
        double ev = std::abs(eigenvalues[i]);
        if (ev > max_ev) max_ev = ev;
        if (ev < min_ev) min_ev = ev;
    }
    return (min_ev > constants::EPSILON) ? max_ev / min_ev : 1e15;
}

} // namespace srfm
