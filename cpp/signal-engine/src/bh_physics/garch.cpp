#include "garch.hpp"
#include <cmath>
#include <algorithm>

namespace srfm {

// ============================================================
// GARCHTracker
// ============================================================

GARCHTracker::GARCHTracker(double omega, double alpha, double beta,
                             double target_vol, double ann_factor) noexcept
    : omega_(omega)
    , alpha_(alpha)
    , beta_(beta)
    , target_vol_(target_vol)
    , ann_factor_(ann_factor)
    , variance_(omega / std::max(1.0 - alpha - beta, 1e-6))  // init to unconditional
    , prev_return_(0.0)
    , vol_scale_(1.0)
    , prev_close_(0.0)
    , count_(0)
    , has_prev_(false)
{}

double GARCHTracker::unconditional_variance() const noexcept {
    double denom = 1.0 - alpha_ - beta_;
    if (denom < constants::EPSILON) return omega_ / constants::EPSILON;
    return omega_ / denom;
}

double GARCHTracker::shock_half_life() const noexcept {
    double persist = alpha_ + beta_;
    if (persist >= 1.0 || persist <= 0.0) return 1e9;
    return std::log(2.0) / std::log(1.0 / persist);
}

GARCHOutput GARCHTracker::update_with_return(double log_ret) noexcept {
    if (count_ == 0) {
        // Initialize variance with squared return
        variance_    = log_ret * log_ret;
        if (variance_ < omega_) variance_ = omega_;
        prev_return_ = log_ret;
        ++count_;
        recompute_vol_scale();
        return { variance_, std::sqrt(variance_), vol_scale_, log_ret, false };
    }

    // GARCH(1,1) update
    double new_var = omega_
                   + alpha_ * (prev_return_ * prev_return_)
                   + beta_  * variance_;

    // Numerical guard
    if (new_var < omega_) new_var = omega_;
    if (new_var > 1.0)    new_var = 1.0;   // cap at 100% daily vol

    variance_    = new_var;
    prev_return_ = log_ret;
    ++count_;

    recompute_vol_scale();
    return { variance_, std::sqrt(variance_), vol_scale_, log_ret, is_warm() };
}

GARCHOutput GARCHTracker::update(double close) noexcept {
    double log_ret = 0.0;
    if (has_prev_ && prev_close_ > constants::EPSILON) {
        log_ret = std::log(close / prev_close_);
    }
    prev_close_ = close;
    has_prev_   = true;
    return update_with_return(log_ret);
}

void GARCHTracker::recompute_vol_scale() noexcept {
    double ann_var = variance_ * ann_factor_;
    double ann_vol = std::sqrt(std::max(ann_var, constants::EPSILON));
    vol_scale_     = target_vol_ / ann_vol;
    // Clamp to reasonable range
    vol_scale_     = std::clamp(vol_scale_, 0.01, 10.0);
}

void GARCHTracker::reset() noexcept {
    variance_    = unconditional_variance();
    prev_return_ = 0.0;
    vol_scale_   = 1.0;
    prev_close_  = 0.0;
    count_       = 0;
    has_prev_    = false;
}

// ============================================================
// EGARCHTracker
// ============================================================

EGARCHTracker::EGARCHTracker(double omega, double alpha,
                               double gamma, double beta) noexcept
    : omega_(omega), alpha_(alpha), gamma_(gamma), beta_(beta)
    , log_var_(std::log(0.0001))  // start at ~1% daily vol
    , prev_std_resid_(0.0)
    , count_(0)
{}

double EGARCHTracker::update(double log_return) noexcept {
    double cur_var = std::exp(log_var_);
    double cur_std = std::sqrt(std::max(cur_var, 1e-12));
    double z       = log_return / cur_std;  // standardized residual

    // EGARCH update: log(sigma^2_t) = omega + alpha*(|z| - E[|z|]) + gamma*z + beta*log(sigma^2_{t-1})
    // E[|z|] for N(0,1) = sqrt(2/pi) ≈ 0.7979
    constexpr double E_ABS_Z = 0.7978845608;
    double new_log_var = omega_
                       + alpha_ * (std::abs(prev_std_resid_) - E_ABS_Z)
                       + gamma_ * prev_std_resid_
                       + beta_  * log_var_;

    // Cap log_var
    new_log_var = std::clamp(new_log_var, -20.0, 2.0);

    log_var_        = new_log_var;
    prev_std_resid_ = z;
    ++count_;
    return variance();
}

double EGARCHTracker::variance() const noexcept {
    return std::exp(log_var_);
}

void EGARCHTracker::reset() noexcept {
    log_var_        = std::log(0.0001);
    prev_std_resid_ = 0.0;
    count_          = 0;
}

// ============================================================
// Batch GARCH computation
// ============================================================

void garch_batch(const double* closes, std::size_t n,
                 double omega, double alpha, double beta,
                 double* out_variance, double* out_vol_scale,
                 double target_vol, double ann_factor) noexcept {
    GARCHTracker garch(omega, alpha, beta, target_vol, ann_factor);
    for (std::size_t i = 0; i < n; ++i) {
        auto res         = garch.update(closes[i]);
        out_variance[i]  = res.variance;
        out_vol_scale[i] = res.vol_scale;
    }
}

// ============================================================
// GARCH model diagnostics
// ============================================================

/// Ljung-Box test statistic for GARCH residuals (up to max_lag lags).
/// Returns the Q statistic. Compare against chi-squared(max_lag) critical value.
double ljung_box_stat(const double* std_residuals, std::size_t n,
                       int max_lag) noexcept {
    if (n < 2 || max_lag < 1) return 0.0;

    // Compute autocorrelations of squared residuals
    double q = 0.0;
    for (int lag = 1; lag <= max_lag && lag < static_cast<int>(n); ++lag) {
        double num = 0.0, den1 = 0.0;
        for (std::size_t t = lag; t < n; ++t) {
            num  += std_residuals[t] * std_residuals[t - lag];
            den1 += std_residuals[t] * std_residuals[t];
        }
        double rho = (den1 > constants::EPSILON) ? num / den1 : 0.0;
        q += rho * rho / (n - lag);
    }
    return q * n * (n + 2.0);
}

} // namespace srfm
