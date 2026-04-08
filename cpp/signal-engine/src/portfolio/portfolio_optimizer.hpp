///////////////////////////////////////////////////////////////////////////////
// portfolio_optimizer.hpp
// Portfolio Optimization Engine in C++
// Mean-Variance, Risk Parity, Black-Litterman, Robust Methods
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <limits>
#include <cassert>
#include <memory>
#include <unordered_map>

namespace signal_engine {
namespace portfolio {

///////////////////////////////////////////////////////////////////////////////
// Matrix / Vector Utilities
///////////////////////////////////////////////////////////////////////////////

using Vec = std::vector<double>;
using Mat = std::vector<std::vector<double>>;

/// Create zero matrix
inline Mat zeros(size_t rows, size_t cols) {
    return Mat(rows, Vec(cols, 0.0));
}

/// Create identity matrix
inline Mat eye(size_t n) {
    auto I = zeros(n, n);
    for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

/// Matrix-vector multiply
inline Vec mat_vec(const Mat& A, const Vec& x) {
    size_t m = A.size();
    size_t n = x.size();
    Vec result(m, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

/// Matrix-matrix multiply
inline Mat mat_mul(const Mat& A, const Mat& B) {
    size_t m = A.size();
    size_t p = B.size();
    size_t n = B[0].size();
    auto C = zeros(m, n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < p; ++k) {
            double a_ik = A[i][k];
            for (size_t j = 0; j < n; ++j) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }
    return C;
}

/// Transpose
inline Mat transpose(const Mat& A) {
    if (A.empty()) return {};
    size_t m = A.size(), n = A[0].size();
    auto T = zeros(n, m);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

/// Dot product
inline double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

/// Vector addition
inline Vec vec_add(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

/// Vector subtraction
inline Vec vec_sub(const Vec& a, const Vec& b) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

/// Scalar multiply
inline Vec vec_scale(const Vec& a, double s) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * s;
    return r;
}

/// L2 norm
inline double norm2(const Vec& a) {
    return std::sqrt(dot(a, a));
}

/// Sum
inline double vec_sum(const Vec& a) {
    return std::accumulate(a.begin(), a.end(), 0.0);
}

/// Cholesky decomposition: A = L * L^T (lower triangular)
inline Mat cholesky(const Mat& A) {
    size_t n = A.size();
    auto L = zeros(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double diag = A[i][i] - sum;
                if (diag <= 0.0) diag = 1e-10;
                L[i][j] = std::sqrt(diag);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return L;
}

/// Solve L * x = b (forward substitution)
inline Vec forward_solve(const Mat& L, const Vec& b) {
    size_t n = b.size();
    Vec x(n);
    for (size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (size_t j = 0; j < i; ++j) s -= L[i][j] * x[j];
        x[i] = s / L[i][i];
    }
    return x;
}

/// Solve L^T * x = b (backward substitution)
inline Vec backward_solve(const Mat& L, const Vec& b) {
    size_t n = b.size();
    Vec x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double s = b[i];
        for (size_t j = i + 1; j < n; ++j) s -= L[j][i] * x[j];
        x[i] = s / L[i][i];
    }
    return x;
}

/// Solve A * x = b via Cholesky
inline Vec cholesky_solve(const Mat& A, const Vec& b) {
    auto L = cholesky(A);
    auto y = forward_solve(L, b);
    return backward_solve(L, y);
}

/// Invert symmetric positive-definite matrix via Cholesky
inline Mat cholesky_inverse(const Mat& A) {
    size_t n = A.size();
    auto L = cholesky(A);
    auto inv = zeros(n, n);
    for (size_t i = 0; i < n; ++i) {
        Vec e(n, 0.0);
        e[i] = 1.0;
        auto y = forward_solve(L, e);
        auto col = backward_solve(L, y);
        for (size_t j = 0; j < n; ++j) {
            inv[j][i] = col[j];
        }
    }
    return inv;
}

///////////////////////////////////////////////////////////////////////////////
// Constraint Structures
///////////////////////////////////////////////////////////////////////////////

/// Box constraint: lower[i] <= w[i] <= upper[i]
struct BoxConstraint {
    Vec lower;
    Vec upper;
};

/// Group constraint: group_lower <= sum(w[group_indices]) <= group_upper
struct GroupConstraint {
    std::vector<size_t> indices;
    double lower;
    double upper;
    std::string name;
};

/// Turnover constraint: sum(|w[i] - w_prev[i]|) <= max_turnover
struct TurnoverConstraint {
    Vec prev_weights;
    double max_turnover;
};

/// Factor exposure constraint: lower <= B^T w <= upper
struct FactorExposureConstraint {
    Mat factor_loadings;  // n_assets x n_factors
    Vec lower;
    Vec upper;
};

/// Cardinality constraint: at most max_assets with non-zero weight
struct CardinalityConstraint {
    size_t max_assets;
    double min_weight;  // minimum non-zero weight
};

///////////////////////////////////////////////////////////////////////////////
// Covariance Estimation
///////////////////////////////////////////////////////////////////////////////

/// Sample covariance matrix
Mat sample_covariance(const Mat& returns);

/// Exponentially weighted covariance
Mat ewma_covariance(const Mat& returns, double lambda = 0.94);

/// Ledoit-Wolf shrinkage covariance
struct LedoitWolfResult {
    Mat shrunk_cov;
    double shrinkage_intensity;
    Mat sample_cov;
    Mat target;
};
LedoitWolfResult ledoit_wolf_shrinkage(const Mat& returns);

/// Factor model covariance
Mat factor_model_covariance(const Mat& returns, size_t n_factors = 3);

///////////////////////////////////////////////////////////////////////////////
// Portfolio Optimization Results
///////////////////////////////////////////////////////////////////////////////

struct OptimizationResult {
    Vec weights;
    double expected_return;
    double volatility;
    double sharpe_ratio;
    Vec risk_contribution;
    Vec marginal_risk;
    int iterations;
    bool converged;
    std::string method;
    double objective_value;
};

///////////////////////////////////////////////////////////////////////////////
// Transaction Cost Model
///////////////////////////////////////////////////////////////////////////////

struct TransactionCostModel {
    double proportional_cost;     // e.g., 0.001 (10 bps)
    double fixed_cost;            // per-trade fixed cost
    Vec impact_coefficients;      // market impact per asset
    double tax_rate;              // short-term cap gains rate
    Vec cost_basis;               // for tax-lot awareness

    double compute_cost(const Vec& old_weights, const Vec& new_weights,
                        double portfolio_value) const;
};

///////////////////////////////////////////////////////////////////////////////
// PortfolioOptimizer Class
///////////////////////////////////////////////////////////////////////////////

class PortfolioOptimizer {
public:
    PortfolioOptimizer() = default;
    ~PortfolioOptimizer() = default;

    // -----------------------------------------------------------------------
    // Core Optimization Methods
    // -----------------------------------------------------------------------

    /// Mean-Variance Optimization via Active Set QP
    OptimizationResult mean_variance(
        const Vec& mu,
        const Mat& Sigma,
        double target_return = std::numeric_limits<double>::quiet_NaN(),
        double risk_aversion = std::numeric_limits<double>::quiet_NaN(),
        const BoxConstraint* box = nullptr
    );

    /// Minimum Variance Portfolio
    OptimizationResult minimum_variance(
        const Mat& Sigma,
        const BoxConstraint* box = nullptr
    );

    /// Maximum Sharpe Ratio Portfolio
    OptimizationResult max_sharpe(
        const Vec& mu,
        const Mat& Sigma,
        double risk_free_rate = 0.0,
        const BoxConstraint* box = nullptr
    );

    /// Risk Parity (Equal Risk Contribution)
    OptimizationResult risk_parity(
        const Mat& Sigma,
        const Vec& risk_budget = {},
        double tol = 1e-10,
        int max_iter = 5000
    );

    /// Black-Litterman
    OptimizationResult black_litterman(
        const Vec& market_weights,
        const Mat& Sigma,
        const Mat& P,             // view pick matrix
        const Vec& Q,             // view returns
        const Mat& Omega,         // view uncertainty
        double tau = 0.05,
        double risk_aversion = 2.5,
        const BoxConstraint* box = nullptr
    );

    /// Maximum Diversification
    OptimizationResult max_diversification(
        const Vec& sigma,
        const Mat& Sigma,
        const BoxConstraint* box = nullptr
    );

    /// Robust Mean-Variance (with shrinkage + resampling)
    OptimizationResult robust_markowitz(
        const Mat& returns,
        int n_resample = 500,
        double risk_aversion = 2.5,
        const BoxConstraint* box = nullptr
    );

    // -----------------------------------------------------------------------
    // Constraint Handling
    // -----------------------------------------------------------------------

    void set_box_constraints(const Vec& lower, const Vec& upper);
    void add_group_constraint(const GroupConstraint& gc);
    void set_turnover_constraint(const TurnoverConstraint& tc);
    void set_factor_exposure_constraint(const FactorExposureConstraint& fec);
    void set_cardinality_constraint(const CardinalityConstraint& cc);
    void clear_constraints();

    // -----------------------------------------------------------------------
    // Rebalancing
    // -----------------------------------------------------------------------

    struct RebalanceResult {
        Vec new_weights;
        Vec trades;           // new - old
        double turnover;
        double total_cost;
        bool rebalance_triggered;
    };

    RebalanceResult compute_rebalance(
        const Vec& current_weights,
        const Vec& target_weights,
        double portfolio_value,
        const TransactionCostModel& cost_model,
        double threshold = 0.0
    );

    Vec threshold_rebalance(
        const Vec& current_weights,
        const Vec& target_weights,
        double threshold
    );

    // -----------------------------------------------------------------------
    // Risk Decomposition
    // -----------------------------------------------------------------------

    struct RiskDecomposition {
        double portfolio_vol;
        Vec marginal_risk;
        Vec component_risk;
        Vec pct_contribution;
        Vec component_var;
        double diversification_ratio;
    };

    RiskDecomposition decompose_risk(
        const Vec& weights,
        const Mat& Sigma,
        double alpha = 0.95
    );

    // -----------------------------------------------------------------------
    // Efficient Frontier
    // -----------------------------------------------------------------------

    struct FrontierPoint {
        double expected_return;
        double volatility;
        double sharpe_ratio;
        Vec weights;
    };

    std::vector<FrontierPoint> efficient_frontier(
        const Vec& mu,
        const Mat& Sigma,
        size_t n_points = 50,
        const BoxConstraint* box = nullptr
    );

private:
    // Active constraints
    std::unique_ptr<BoxConstraint> box_constraint_;
    std::vector<GroupConstraint> group_constraints_;
    std::unique_ptr<TurnoverConstraint> turnover_constraint_;
    std::unique_ptr<FactorExposureConstraint> factor_constraint_;
    std::unique_ptr<CardinalityConstraint> cardinality_constraint_;

    // Internal solvers
    Vec solve_qp_active_set(const Mat& Q, const Vec& c,
                            const Mat& A_eq, const Vec& b_eq,
                            const Mat& A_ineq, const Vec& b_ineq,
                            int max_iter = 1000);

    Vec project_onto_simplex(const Vec& v, double z = 1.0);
    Vec project_onto_box(const Vec& v, const BoxConstraint& box);

    /// Greedy cardinality constraint enforcement
    Vec enforce_cardinality(const Vec& w, size_t max_assets, double min_weight);
};

} // namespace portfolio
} // namespace signal_engine
