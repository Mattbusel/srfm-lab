#include "rsi.hpp"
#include <cmath>
#include <algorithm>

namespace srfm {

RSI::RSI(int period) noexcept
    : period_(period)
    , alpha_(1.0 / period)
    , avg_gain_(0.0)
    , avg_loss_(0.0)
    , prev_close_(0.0)
    , rsi_value_(50.0)  // neutral until warm
    , count_(0)
    , sum_gain_(0.0)
    , sum_loss_(0.0)
    , has_prev_(false)
{}

double RSI::update(double close) noexcept {
    if (!has_prev_) {
        prev_close_ = close;
        has_prev_   = true;
        ++count_;
        return rsi_value_;
    }

    double delta = close - prev_close_;
    double gain  = (delta > 0.0) ? delta : 0.0;
    double loss  = (delta < 0.0) ? -delta : 0.0;

    if (count_ <= period_) {
        // Accumulate for initial simple average
        sum_gain_ += gain;
        sum_loss_ += loss;

        if (count_ == period_) {
            avg_gain_ = sum_gain_ / period_;
            avg_loss_ = sum_loss_ / period_;
        }
    } else {
        // Wilder's smoothing
        avg_gain_ = avg_gain_ * (1.0 - alpha_) + gain * alpha_;
        avg_loss_ = avg_loss_ * (1.0 - alpha_) + loss * alpha_;
    }

    if (count_ >= period_) {
        if (avg_loss_ < constants::EPSILON) {
            rsi_value_ = 100.0;
        } else {
            double rs  = avg_gain_ / avg_loss_;
            rsi_value_ = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    prev_close_ = close;
    ++count_;
    return rsi_value_;
}

void RSI::reset() noexcept {
    avg_gain_   = 0.0;
    avg_loss_   = 0.0;
    prev_close_ = 0.0;
    rsi_value_  = 50.0;
    count_      = 0;
    sum_gain_   = 0.0;
    sum_loss_   = 0.0;
    has_prev_   = false;
}

// ============================================================
// Batch RSI computation
// ============================================================

void rsi_batch(const double* closes, std::size_t n, int period,
               double* out_rsi) noexcept {
    RSI rsi(period);
    for (std::size_t i = 0; i < n; ++i) {
        out_rsi[i] = rsi.update(closes[i]);
    }
}

// ============================================================
// RSI divergence helper: check for bullish/bearish divergence
// over a lookback window.
// Returns +1 for bullish, -1 for bearish, 0 for none.
// ============================================================

int rsi_divergence(const double* closes, const double* rsis,
                   std::size_t n, std::size_t lookback) noexcept {
    if (n < lookback + 1) return 0;

    // Find local minimum in price and RSI over the lookback
    double min_price_prev = closes[n - lookback - 1];
    double min_rsi_prev   = rsis[n - lookback - 1];
    double min_price_curr = closes[n - 1];
    double min_rsi_curr   = rsis[n - 1];

    for (std::size_t i = n - lookback; i < n - 1; ++i) {
        if (closes[i] < min_price_prev) min_price_prev = closes[i];
        if (rsis[i]   < min_rsi_prev)   min_rsi_prev   = rsis[i];
    }

    // Bullish divergence: price makes lower low but RSI makes higher low
    if (min_price_curr < min_price_prev && min_rsi_curr > min_rsi_prev)
        return +1;

    // Bearish divergence: price makes higher high but RSI makes lower high
    double max_price_prev = closes[n - lookback - 1];
    double max_rsi_prev   = rsis[n - lookback - 1];
    double max_price_curr = closes[n - 1];
    double max_rsi_curr   = rsis[n - 1];

    for (std::size_t i = n - lookback; i < n - 1; ++i) {
        if (closes[i] > max_price_prev) max_price_prev = closes[i];
        if (rsis[i]   > max_rsi_prev)   max_rsi_prev   = rsis[i];
    }

    if (max_price_curr > max_price_prev && max_rsi_curr < max_rsi_prev)
        return -1;

    return 0;
}

} // namespace srfm
