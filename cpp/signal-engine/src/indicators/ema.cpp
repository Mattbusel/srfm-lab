#include "ema.hpp"
#include <cmath>
#include <stdexcept>

namespace srfm {

// ============================================================
// EMA implementation
// ============================================================

EMA::EMA(int period) noexcept
    : alpha_(2.0 / (period + 1.0))
    , value_(0.0)
    , period_(period)
    , count_(0)
    , warm_(false)
{}

EMA::EMA(double alpha) noexcept
    : alpha_(alpha)
    , value_(0.0)
    , period_(static_cast<int>(2.0 / alpha - 1.0))
    , count_(0)
    , warm_(false)
{}

double EMA::update(double price) noexcept {
    if (count_ == 0) {
        // Seed with first price (simple initialization)
        value_ = price;
    } else {
        value_ = alpha_ * price + (1.0 - alpha_) * value_;
    }
    ++count_;
    if (!warm_ && count_ >= period_) {
        warm_ = true;
    }
    return value_;
}

void EMA::reset() noexcept {
    value_ = 0.0;
    count_ = 0;
    warm_  = false;
}

// ============================================================
// EMA Cross implementation
// ============================================================

EMACross::EMACross(int fast_period, int slow_period) noexcept
    : fast_(fast_period)
    , slow_(slow_period)
    , prev_sign_(0)
{}

EMACrossSignal EMACross::update(double price) noexcept {
    double fv = fast_.update(price);
    double sv = slow_.update(price);
    double diff = fv - sv;

    int cur_sign = (diff > 0.0) ? 1 : (diff < 0.0 ? -1 : 0);
    int cross = 0;

    if (is_warm() && prev_sign_ != 0 && cur_sign != prev_sign_) {
        cross = cur_sign; // direction of the new signal
    }
    if (cur_sign != 0) prev_sign_ = cur_sign;

    return { fv, sv, diff, cross };
}

void EMACross::reset() noexcept {
    fast_.reset();
    slow_.reset();
    prev_sign_ = 0;
}

// ============================================================
// Utility: compute alpha from period and period from alpha
// ============================================================

double alpha_from_period(int period) noexcept {
    return 2.0 / (period + 1.0);
}

int period_from_alpha(double alpha) noexcept {
    return static_cast<int>(std::round(2.0 / alpha - 1.0));
}

// ============================================================
// Wilder's smoothing alpha = 1/period (used by ATR, RSI)
// ============================================================

double wilder_alpha(int period) noexcept {
    return 1.0 / period;
}

// ============================================================
// Batch EMA update standalone function
// ============================================================

void batch_ema_update(double* emas, const double* prices,
                       std::size_t n, double alpha) noexcept {
    simd::ema_update_batch(prices, emas, n, alpha);
}

// ============================================================
// EMA convergence test helper
// ============================================================

/// Returns the number of periods needed for EMA to converge to within
/// 'tolerance' fraction of the true value (assuming stable input).
int ema_convergence_periods(double alpha, double tolerance) noexcept {
    // After n periods: error ~ (1-alpha)^n
    // Solve (1-alpha)^n < tolerance  =>  n > ln(tolerance)/ln(1-alpha)
    if (alpha <= 0.0 || alpha >= 1.0 || tolerance <= 0.0 || tolerance >= 1.0)
        return -1;
    return static_cast<int>(std::ceil(std::log(tolerance) / std::log(1.0 - alpha)));
}

} // namespace srfm
