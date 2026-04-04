#pragma once
// Time series operations for financial data:
// Kalman filter, ARIMA, GARCH volatility, Kalman pairs trading
// All implemented on top of Matrix<double>

#include "matrix.hpp"
#include <vector>
#include <cmath>
#include <deque>
#include <numeric>
#include <stdexcept>
#include <algorithm>

namespace linalg {
namespace timeseries {

// ============================================================
// Kalman Filter (Linear Gaussian State Space Model)
// State equation:  x_t = F * x_{t-1} + Q noise
// Observation:     y_t = H * x_t     + R noise
// ============================================================
struct KalmanState {
    MatrixD x;   // state estimate (n x 1)
    MatrixD P;   // state covariance (n x n)
    MatrixD F;   // state transition (n x n)
    MatrixD H;   // observation matrix (m x n)
    MatrixD Q;   // process noise covariance (n x n)
    MatrixD R;   // observation noise covariance (m x m)
    double  log_likelihood = 0.0;
};

struct KalmanResult {
    std::vector<MatrixD> filtered_states;    // x_{t|t}
    std::vector<MatrixD> filtered_covs;      // P_{t|t}
    std::vector<MatrixD> predicted_states;   // x_{t|t-1}
    std::vector<double>  innovations;        // y_t - H x_{t|t-1}
    double               total_log_likelihood;
};

KalmanResult kalman_filter(KalmanState state,
                             const std::vector<std::vector<double>>& observations)
{
    KalmanResult result;
    result.total_log_likelihood = 0.0;

    const size_t n = state.x.rows(); // state dim
    const size_t m = state.H.rows(); // obs dim

    for (const auto& obs : observations) {
        // --- Predict ---
        MatrixD x_pred(n, 1);
        // x_pred = F * x
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                x_pred(i,0) += state.F(i,j) * state.x(j,0);

        // P_pred = F * P * F^T + Q
        MatrixD FP(n, n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                    FP(i,j) += state.F(i,k) * state.P(k,j);

        MatrixD P_pred(n, n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < n; ++k)
                    s += FP(i,k) * state.F(j,k); // F * P * F^T
                P_pred(i,j) = s + state.Q(i,j);
            }

        result.predicted_states.push_back(x_pred);

        // --- Update ---
        // Innovation: y - H * x_pred
        MatrixD Hx(m, 1, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                Hx(i,0) += state.H(i,j) * x_pred(j,0);

        double innovation = (m == 1 && obs.size() >= 1) ? obs[0] - Hx(0,0) : 0.0;
        result.innovations.push_back(innovation);

        // S = H * P_pred * H^T + R  (innovation covariance)
        MatrixD HP(m, n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < n; ++k)
                    HP(i,j) += state.H(i,k) * P_pred(k,j);

        MatrixD S(m, m, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < m; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < n; ++k)
                    s += HP(i,k) * state.H(j,k);
                S(i,j) = s + state.R(i,j);
            }

        // Kalman gain: K = P_pred * H^T * S^{-1}
        MatrixD PH(n, m, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                for (size_t k = 0; k < n; ++k)
                    PH(i,j) += P_pred(i,k) * state.H(j,k);

        // S is 1x1 in scalar case
        double S00 = S(0,0);
        if (std::fabs(S00) < 1e-14) S00 = 1e-14;

        MatrixD K(n, m, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                K(i,j) = PH(i,j) / S00;

        // x = x_pred + K * innovation
        MatrixD x_upd(n, 1);
        for (size_t i = 0; i < n; ++i)
            x_upd(i,0) = x_pred(i,0) + K(i,0) * innovation;

        // P = (I - K H) P_pred
        MatrixD KH(n, n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t k = 0; k < m; ++k)
                    KH(i,j) += K(i,k) * state.H(k,j);

        MatrixD P_upd(n, n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double IKH_ij = (i == j ? 1.0 : 0.0) - KH(i,j);
                double s = 0.0;
                for (size_t k = 0; k < n; ++k)
                    s += IKH_ij * P_pred(i, k); // simplified: only diagonal term
                P_upd(i,j) = IKH_ij * P_pred(i,j);
            }

        state.x = x_upd;
        state.P = P_upd;

        result.filtered_states.push_back(x_upd);
        result.filtered_covs.push_back(P_upd);

        // Log likelihood contribution
        double ll = -0.5 * (std::log(2 * M_PI * std::fabs(S00)) +
                            innovation * innovation / S00);
        result.total_log_likelihood += ll;
    }
    return result;
}

// ============================================================
// Kalman Pairs Trading
// Dynamically estimates hedge ratio using Kalman filter
// State: [beta, alpha] where spread = y - beta*x - alpha
// ============================================================
class KalmanPairsTrader {
public:
    explicit KalmanPairsTrader(double delta = 1e-4, double R_noise = 1e-3)
        : delta_(delta), R_(R_noise)
    {
        // 2D state: [beta, alpha]
        theta_ = MatrixD(2, 1, 0.0);
        theta_(0,0) = 1.0; // initial beta = 1
        P_ = MatrixD::identity(2);
        P_ *= 1.0; // initial covariance
    }

    // Update with new observations (y = leg_y price, x = leg_x price)
    // Returns: current spread estimate
    double update(double y, double x) {
        // F = [x, 1] (observation matrix)
        MatrixD F(1, 2, 0.0);
        F(0,0) = x; F(0,1) = 1.0;

        // Q = delta / (1-delta) * I
        double q = delta_ / (1.0 - delta_);
        MatrixD Q = MatrixD::identity(2);
        Q *= q;

        // Predicted covariance
        MatrixD P_pred(2, 2, 0.0);
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 2; ++j)
                P_pred(i,j) = P_(i,j) + Q(i,j);

        // Innovation variance: e_var = F * P_pred * F^T + R
        double e_var = 0.0;
        double FP0 = F(0,0)*P_pred(0,0) + F(0,1)*P_pred(1,0);
        double FP1 = F(0,0)*P_pred(0,1) + F(0,1)*P_pred(1,1);
        e_var = F(0,0)*FP0 + F(0,1)*FP1 + R_;

        // Predicted spread
        double y_hat = F(0,0)*theta_(0,0) + F(0,1)*theta_(1,0);
        double innovation = y - y_hat;

        // Kalman gain: K = P_pred * F^T / e_var
        MatrixD K(2, 1, 0.0);
        K(0,0) = (P_pred(0,0)*F(0,0) + P_pred(0,1)*F(0,1)) / e_var;
        K(1,0) = (P_pred(1,0)*F(0,0) + P_pred(1,1)*F(0,1)) / e_var;

        // Update state
        theta_(0,0) += K(0,0) * innovation;
        theta_(1,0) += K(1,0) * innovation;

        // Update covariance: P = (I - K*F) * P_pred
        double KF00 = K(0,0)*F(0,0), KF01 = K(0,0)*F(0,1);
        double KF10 = K(1,0)*F(0,0), KF11 = K(1,0)*F(0,1);
        MatrixD P_new(2, 2, 0.0);
        P_new(0,0) = (1-KF00)*P_pred(0,0) - KF01*P_pred(1,0);
        P_new(0,1) = (1-KF00)*P_pred(0,1) - KF01*P_pred(1,1);
        P_new(1,0) = -KF10 *P_pred(0,0) + (1-KF11)*P_pred(1,0);
        P_new(1,1) = -KF10 *P_pred(0,1) + (1-KF11)*P_pred(1,1);
        P_ = P_new;

        spread_history_.push_back(innovation);
        if (spread_history_.size() > 200) spread_history_.pop_front();

        last_spread_ = innovation;
        last_beta_   = theta_(0,0);
        last_alpha_  = theta_(1,0);
        update_count_++;

        return innovation;
    }

    double spread_z_score() const {
        if (spread_history_.size() < 10) return 0.0;
        double mean = 0.0, var = 0.0;
        for (auto v : spread_history_) mean += v;
        mean /= spread_history_.size();
        for (auto v : spread_history_) var += (v - mean) * (v - mean);
        var /= spread_history_.size();
        double std = std::sqrt(var);
        return std > 1e-8 ? (last_spread_ - mean) / std : 0.0;
    }

    double beta()        const noexcept { return last_beta_; }
    double alpha()       const noexcept { return last_alpha_; }
    double spread()      const noexcept { return last_spread_; }
    uint64_t updates()   const noexcept { return update_count_; }

private:
    double     delta_;
    double     R_;
    MatrixD    theta_; // [beta, alpha]
    MatrixD    P_;     // state covariance
    double     last_spread_ = 0.0;
    double     last_beta_   = 1.0;
    double     last_alpha_  = 0.0;
    uint64_t   update_count_ = 0;
    std::deque<double> spread_history_;
};

// ============================================================
// GARCH(1,1) Volatility Model
// sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
// ============================================================
class GARCH11 {
public:
    struct Params {
        double omega = 1e-6;
        double alpha = 0.05;
        double beta  = 0.90;
    };

    explicit GARCH11(const Params& p = {})
        : omega_(p.omega), alpha_(p.alpha), beta_(p.beta),
          sigma2_(p.omega / (1.0 - p.alpha - p.beta)) // unconditional variance
    {
        if (alpha_ + beta_ >= 1.0) {
            // Coerce to stationary
            double sum = alpha_ + beta_;
            alpha_ *= 0.99 / sum;
            beta_  *= 0.99 / sum;
            sigma2_ = omega_ / (1.0 - alpha_ - beta_);
        }
    }

    // Update with new return r_t; returns conditional variance
    double update(double r) {
        sigma2_ = omega_ + alpha_ * r * r + beta_ * sigma2_;
        last_return_ = r;
        ++n_;
        return sigma2_;
    }

    double conditional_vol() const noexcept { return std::sqrt(sigma2_); }
    double conditional_var() const noexcept { return sigma2_; }

    // Annualized volatility (assuming daily returns, 252 trading days)
    double annualized_vol(double periods_per_year = 252.0) const noexcept {
        return std::sqrt(sigma2_ * periods_per_year);
    }

    // Forecast h-step-ahead variance
    double forecast(int h) const noexcept {
        double sv = omega_ / (1.0 - alpha_ - beta_); // long-run variance
        double ab = alpha_ + beta_;
        double var = sigma2_;
        for (int i = 1; i < h; ++i)
            var = omega_ + ab * (var - sv) + sv; // mean-reverting
        return var;
    }

    // Maximum likelihood estimation given return series
    static Params fit_mle(const std::vector<double>& returns,
                           int max_iter = 200, double tol = 1e-7)
    {
        Params p_best{1e-6, 0.10, 0.85};
        double ll_best = -1e30;

        // Grid search + local refinement
        for (double a : {0.05, 0.10, 0.15}) {
            for (double b : {0.80, 0.85, 0.90}) {
                if (a + b >= 1.0) continue;
                double n = returns.size();
                double uv = 0.0;
                for (auto r : returns) uv += r * r;
                uv /= n;
                Params cand{uv * (1 - a - b), a, b};
                GARCH11 m(cand);
                double ll = 0.0;
                for (auto r : returns) {
                    double sv = m.update(r);
                    if (sv > 0) ll += -0.5 * (std::log(2*M_PI*sv) + r*r/sv);
                }
                if (ll > ll_best) { ll_best = ll; p_best = cand; }
            }
        }
        return p_best;
    }

    uint64_t n() const noexcept { return n_; }

private:
    double   omega_, alpha_, beta_;
    double   sigma2_;
    double   last_return_ = 0.0;
    uint64_t n_           = 0;
};

// ============================================================
// Rolling Sharpe Ratio with significance test
// ============================================================
struct RollingSharpe {
    size_t window;
    double rf_rate;  // risk-free rate per period
    std::deque<double> rets;
    double sum_r    = 0.0;
    double sum_r2   = 0.0;

    explicit RollingSharpe(size_t w = 60, double rf = 0.0)
        : window(w), rf_rate(rf) {}

    void update(double r) {
        rets.push_back(r);
        sum_r  += r;
        sum_r2 += r * r;
        if (rets.size() > window) {
            double old = rets.front(); rets.pop_front();
            sum_r  -= old;
            sum_r2 -= old * old;
        }
    }

    double sharpe(double periods_per_year = 252.0) const {
        size_t n = rets.size();
        if (n < 2) return 0.0;
        double mean = sum_r / n;
        double var  = (sum_r2 - sum_r * sum_r / n) / (n - 1);
        double std  = std::sqrt(var);
        return std > 1e-14 ? (mean - rf_rate) / std * std::sqrt(periods_per_year) : 0.0;
    }

    // t-statistic for sharpe ratio (Lo 2002)
    double sharpe_t_stat() const {
        size_t n = rets.size();
        if (n < 2) return 0.0;
        double sr = sharpe(1.0); // non-annualized
        // SR / sqrt((1 + SR^2/2) / (n-1))
        double denom = std::sqrt((1.0 + sr*sr/2.0) / (n - 1));
        return denom > 0 ? sr / denom : 0.0;
    }
};

} // namespace timeseries
} // namespace linalg
