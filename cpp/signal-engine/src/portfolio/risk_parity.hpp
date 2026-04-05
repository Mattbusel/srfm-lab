#pragma once
#include "srfm/types.hpp"
#include <array>

namespace srfm {

struct PositionSize {
    int    symbol_id;
    double raw_size;       // before cap
    double final_size;     // after cap and correlation adjustment
    double vol_budget;     // PER_INST_RISK / (vol + epsilon)
    double corr_factor;    // dynamic correlation adjustment
};

/// Per-instrument risk budget: PER_INST_RISK / (vol + epsilon).
/// Dynamic CORR factor from rolling realized correlation matrix.
/// Position size cap enforcement.
class RiskParity {
public:
    RiskParity(int n_instruments,
               double per_inst_risk = constants::PER_INST_RISK,
               double max_position  = 1.0,
               int    corr_window   = 1440 * 30) noexcept;  // 30 days of 1m bars

    /// Update volatility for one instrument.
    void update_vol(int symbol_id, double vol) noexcept;

    /// Update return for one instrument (used for correlation matrix).
    void update_return(int symbol_id, double log_return) noexcept;

    /// Compute position sizes for all instruments given current vols.
    void compute_positions(PositionSize* out, int n_out) const noexcept;

    /// Get the correlation matrix element [i][j].
    double correlation(int i, int j) const noexcept;

    /// Average pairwise correlation (proxy for portfolio diversification).
    double avg_pairwise_correlation() const noexcept;

    /// Returns the dynamic correlation factor for instrument i.
    double corr_factor(int i) const noexcept;

    int n_instruments() const noexcept { return n_instruments_; }

    void reset() noexcept;

private:
    void   update_corr_matrix() noexcept;

    int    n_instruments_;
    double per_inst_risk_;
    double max_position_;
    int    corr_window_;

    // Per-instrument state
    std::array<double, MAX_INSTRUMENTS> vols_{};
    std::array<double, MAX_INSTRUMENTS> returns_{};  // latest return
    std::array<double, MAX_INSTRUMENTS> return_ema_{}; // EMA for mean
    std::array<double, MAX_INSTRUMENTS> return_var_{}; // running variance

    // Rolling covariance / correlation matrix (n x n, stored flat)
    // Max 25x25 = 625 elements
    std::array<double, MAX_INSTRUMENTS * MAX_INSTRUMENTS> cov_matrix_{};
    std::array<double, MAX_INSTRUMENTS * MAX_INSTRUMENTS> corr_matrix_{};

    // Online covariance update: Welford-style cross products
    std::array<double, MAX_INSTRUMENTS * MAX_INSTRUMENTS> cov_sum_{};
    std::array<int,    MAX_INSTRUMENTS> ret_counts_{};

    // Correlation factor per instrument (avg correlation with portfolio)
    std::array<double, MAX_INSTRUMENTS> corr_factors_{};

    int    update_count_;
};

// ============================================================
// Matrix operations (no Eigen — hand-rolled for n <= 25)
// ============================================================

/// Multiply two n x n matrices: C = A * B (stored row-major)
void mat_mul(const double* A, const double* B, double* C, int n) noexcept;

/// Compute eigenvalues of symmetric n x n matrix using Jacobi iteration.
/// out_eigenvalues: n values. Returns iteration count.
int mat_eigenvalues_sym(const double* A, double* out_eigenvalues, int n,
                         int max_iter = 100, double tol = 1e-9) noexcept;

/// Correlation from covariance matrix in-place.
void cov_to_corr(const double* cov, double* corr, int n) noexcept;

/// Condition number of symmetric positive definite matrix.
double mat_condition_number(const double* A, int n) noexcept;

} // namespace srfm
