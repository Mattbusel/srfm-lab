/// heston_engine.cpp — Heston SV model in C++.
///
/// Implements:
/// - Euler-Maruyama discretization for variance process (CIR, full truncation)
/// - Correlated Brownian motions via Cholesky decomposition
/// - Parameter validation and Feller condition enforcement
/// - Path generation that drives the LOB mid-price
/// - Integrated variance (realized variance) computation
/// - Method of moments calibration from return statistics

#include "../include/lob_types.hpp"

#include <cmath>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <random>
#include <numeric>
#include <algorithm>

namespace chronos {
namespace heston {

// ── Parameters ────────────────────────────────────────────────────────────────

struct HestonParams {
    double mu;       ///< Log-price drift.
    double kappa;    ///< Mean-reversion speed of variance.
    double theta;    ///< Long-run variance.
    double sigma;    ///< Vol-of-vol.
    double rho;      ///< Correlation of dW_S, dW_V.
    double v0;       ///< Initial variance.

    bool feller_satisfied() const noexcept {
        return 2.0 * kappa * theta > sigma * sigma;
    }

    double unconditional_vol() const noexcept {
        return std::sqrt(theta);
    }

    double variance_half_life() const noexcept {
        return std::log(2.0) / kappa;
    }

    void enforce_constraints() noexcept {
        kappa = std::max(kappa, 0.01);
        theta = std::max(theta, 1e-6);
        sigma = std::max(sigma, 1e-6);
        v0    = std::max(v0, 1e-6);
        rho   = std::max(-0.999, std::min(0.999, rho));

        // Enforce Feller: 2κθ > σ².
        if (!feller_satisfied()) {
            sigma = std::sqrt(2.0 * kappa * theta * 0.99);
        }
    }

    static HestonParams default_params() noexcept {
        return HestonParams{
            .mu    = 0.0,
            .kappa = 2.0,
            .theta = 0.04,
            .sigma = 0.3,
            .rho   = -0.7,
            .v0    = 0.04,
        };
    }
};

// ── Path Step ────────────────────────────────────────────────────────────────

struct HestonStep {
    Nanos   timestamp_ns;
    double  price;
    double  variance;
    double  log_return;
    double  inst_vol;   ///< sqrt(variance).
};

// ── Pseudo-random Normal using Box-Muller ─────────────────────────────────────

struct NormalPair {
    double z1, z2;
};

static NormalPair box_muller(double u1, double u2) noexcept {
    constexpr double TWO_PI = 2.0 * M_PI;
    u1 = std::max(u1, 1e-20);
    double mag = std::sqrt(-2.0 * std::log(u1));
    return { mag * std::cos(TWO_PI * u2), mag * std::sin(TWO_PI * u2) };
}

// ── Heston Simulator ──────────────────────────────────────────────────────────

class HestonSimulator {
public:
    HestonSimulator(HestonParams params, uint64_t seed = 42)
        : params_(params)
        , rng_(seed)
        , uniform_(0.0, 1.0)
    {
        params_.enforce_constraints();
        rho_bar_ = std::sqrt(1.0 - params_.rho * params_.rho);
    }

    /// Simulate a complete path.
    /// Returns vector of HestonStep.
    std::vector<HestonStep> simulate(
        double s0,
        size_t n_steps,
        double horizon,
        Nanos start_ts_ns = 0
    ) {
        std::vector<HestonStep> path;
        path.reserve(n_steps + 1);

        double dt = horizon / static_cast<double>(n_steps);
        double sqrt_dt = std::sqrt(dt);
        Nanos step_ns = static_cast<Nanos>(dt * 1e9);

        double s = s0;
        double v = params_.v0;
        Nanos ts = start_ts_ns;

        path.push_back({ ts, s, v, 0.0, std::sqrt(v) });

        for (size_t i = 0; i < n_steps; ++i) {
            // Sample correlated Brownian increments.
            auto [n1, n1b] = box_muller(uniform_(rng_), uniform_(rng_));
            auto [n2, n2b] = box_muller(uniform_(rng_), uniform_(rng_));
            (void)n1b; (void)n2b;

            double dw_s = n1 * sqrt_dt;
            double dw_v = (params_.rho * n1 + rho_bar_ * n2) * sqrt_dt;

            // Full-truncation Euler.
            double v_plus = std::max(v, 0.0);
            double sqrt_v = std::sqrt(v_plus);

            // Log-price step.
            double d_log_s = (params_.mu - 0.5 * v_plus) * dt + sqrt_v * dw_s;
            double s_new = s * std::exp(d_log_s);

            // Variance step.
            double dv = params_.kappa * (params_.theta - v_plus) * dt
                      + params_.sigma * sqrt_v * dw_v;
            double v_new = v + dv;  // Allow negative; clamp on next step.

            ts += step_ns;
            path.push_back({
                ts,
                s_new,
                std::max(v_new, 0.0),
                d_log_s,
                std::sqrt(std::max(v_new, 0.0))
            });

            s = s_new;
            v = v_new;
        }

        return path;
    }

    /// Run one step (for streaming simulation).
    HestonStep step(double s, double v, double dt, Nanos ts) {
        double sqrt_dt = std::sqrt(dt);
        auto [n1, n1b] = box_muller(uniform_(rng_), uniform_(rng_));
        auto [n2, n2b] = box_muller(uniform_(rng_), uniform_(rng_));
        (void)n1b; (void)n2b;

        double dw_s = n1 * sqrt_dt;
        double dw_v = (params_.rho * n1 + rho_bar_ * n2) * sqrt_dt;

        double v_plus = std::max(v, 0.0);
        double sqrt_v = std::sqrt(v_plus);

        double d_log_s = (params_.mu - 0.5 * v_plus) * dt + sqrt_v * dw_s;
        double s_new = s * std::exp(d_log_s);

        double v_new = std::max(v + params_.kappa * (params_.theta - v_plus) * dt
                               + params_.sigma * sqrt_v * dw_v, 0.0);

        return HestonStep{
            ts,
            s_new,
            v_new,
            d_log_s,
            std::sqrt(v_new)
        };
    }

    const HestonParams& params() const { return params_; }

private:
    HestonParams params_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_;
    double rho_bar_;
};

// ── Method of Moments Calibration ────────────────────────────────────────────

struct ReturnStats {
    double mean;
    double variance;
    double skewness;
    double excess_kurtosis;
    double acf_sq_lag1;   ///< ACF of squared returns at lag 1.
    double dt;            ///< Observation interval.
    size_t n;
};

static ReturnStats compute_stats(const std::vector<double>& returns, double dt) {
    size_t n = returns.size();
    assert(n >= 4);

    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / n;
    double var = 0.0;
    double skew_sum = 0.0;
    double kurt_sum = 0.0;
    for (double r : returns) {
        double d = r - mean;
        var += d * d;
        skew_sum += d * d * d;
        kurt_sum += d * d * d * d;
    }
    var /= (n - 1);
    double std_dev = std::sqrt(var);
    double skew = (std_dev > 1e-10) ? (skew_sum / n) / (std_dev * std_dev * std_dev) : 0.0;
    double kurt = (std_dev > 1e-10) ? (kurt_sum / n) / (var * var) - 3.0 : 0.0;

    // ACF of squared returns at lag 1.
    std::vector<double> sq(n);
    for (size_t i = 0; i < n; ++i) sq[i] = returns[i] * returns[i];
    double sq_mean = std::accumulate(sq.begin(), sq.end(), 0.0) / n;
    double cov = 0.0, sq_var = 0.0;
    for (size_t i = 0; i + 1 < n; ++i) {
        cov += (sq[i] - sq_mean) * (sq[i + 1] - sq_mean);
        sq_var += (sq[i] - sq_mean) * (sq[i] - sq_mean);
    }
    cov /= (n - 1);
    sq_var /= (n - 1);
    double acf = (sq_var > 1e-20) ? cov / sq_var : 0.0;

    return ReturnStats{ mean, var, skew, kurt, acf, dt, n };
}

static HestonParams calibrate_mom(const ReturnStats& stats) {
    double dt = stats.dt;

    double mu = stats.mean / dt;
    double theta = std::max(stats.variance / dt, 1e-6);

    double acf_clamped = std::max(0.001, std::min(0.999, stats.acf_sq_lag1));
    double kappa = std::max(0.1, std::min(50.0, -std::log(acf_clamped) / dt));

    double kurt = std::max(0.0, stats.excess_kurtosis);
    double sigma_sq = std::max(1e-6, kurt * 2.0 * kappa * theta * theta / 3.0);
    double sigma = std::sqrt(sigma_sq);
    sigma = std::max(0.01, std::min(3.0, sigma));

    // Enforce Feller.
    if (2.0 * kappa * theta < sigma * sigma) {
        sigma = std::sqrt(2.0 * kappa * theta * 0.99);
    }

    // Correlation from leverage effect (skewness).
    double rho = std::max(-0.95, std::min(0.95, -std::copysign(0.5, stats.skewness)));

    HestonParams p{mu, kappa, theta, sigma, rho, theta};
    p.enforce_constraints();
    return p;
}

// ── LOB Driver ────────────────────────────────────────────────────────────────

/// Heston model that drives the LOB mid-price.
class HestonLobDriver {
public:
    HestonLobDriver(double initial_price, HestonParams params, uint64_t seed = 42)
        : sim_(params, seed)
        , price_(initial_price)
        , variance_(params.v0)
        , t_(0.0)
    {}

    /// Advance by dt seconds. Returns (new_price, new_variance, log_return).
    std::tuple<double, double, double> step(double dt) {
        Nanos ts = static_cast<Nanos>(t_ * 1e9);
        HestonStep s = sim_.step(price_, variance_, dt, ts);
        price_ = s.price;
        variance_ = s.variance;
        t_ += dt;
        return { s.price, s.variance, s.log_return };
    }

    double current_price() const { return price_; }
    double current_variance() const { return variance_; }
    double current_vol() const { return std::sqrt(std::max(variance_, 0.0)); }
    double current_time() const { return t_; }

    /// Set a new mid-price (external override, e.g., after a fill moves the price).
    void set_price(double p) { price_ = p; }

    /// Recalibrate model parameters from recent returns.
    void recalibrate(const std::vector<double>& returns, double dt_per_obs) {
        if (returns.size() < 10) return;
        auto stats = compute_stats(returns, dt_per_obs);
        HestonParams new_params = calibrate_mom(stats);
        new_params.v0 = variance_;  // Use current variance as initial.
        sim_ = HestonSimulator(new_params, static_cast<uint64_t>(t_ * 1e9));
    }

    const HestonParams& params() const { return sim_.params(); }

private:
    HestonSimulator sim_;
    double price_;
    double variance_;
    double t_;
};

// ── Realized Variance ─────────────────────────────────────────────────────────

/// Compute integrated realized variance from a path.
double realized_variance(const std::vector<HestonStep>& path) {
    if (path.size() < 2) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        double r = path[i].log_return;
        sum_sq += r * r;
    }
    return sum_sq;
}

/// Annualised realized volatility.
double realized_vol(const std::vector<HestonStep>& path, double ann_factor = 252.0) {
    return std::sqrt(realized_variance(path) * ann_factor / (path.size() - 1));
}

/// Bipower variation (robust to jumps).
double bipower_variation(const std::vector<double>& returns) {
    size_t n = returns.size();
    if (n < 2) return 0.0;
    double sum = 0.0;
    const double factor = M_PI / 2.0;
    for (size_t i = 1; i < n; ++i) {
        sum += std::abs(returns[i]) * std::abs(returns[i - 1]);
    }
    return factor * sum;
}

} // namespace heston
} // namespace chronos
