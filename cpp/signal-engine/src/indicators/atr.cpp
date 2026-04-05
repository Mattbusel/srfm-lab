#include "atr.hpp"
#include <cmath>
#include <algorithm>

namespace srfm {

ATR::ATR(int period) noexcept
    : period_(period)
    , alpha_(1.0 / period)
    , atr_value_(0.0)
    , prev_close_(0.0)
    , count_(0)
    , has_prev_(false)
{}

double ATR::true_range(double high, double low, double close) const noexcept {
    if (!has_prev_) {
        return high - low;
    }
    double hl  = high - low;
    double hpc = std::abs(high - prev_close_);
    double lpc = std::abs(low  - prev_close_);
    return std::max({hl, hpc, lpc});
}

double ATR::update(double high, double low, double close) noexcept {
    double tr = true_range(high, low, close);

    if (count_ == 0) {
        // First bar: seed the ATR with the first TR
        atr_value_ = tr;
    } else if (count_ < period_) {
        // During warmup: use simple average
        // Running mean: ((count_) * atr_value_ + tr) / (count_ + 1)
        atr_value_ = (atr_value_ * count_ + tr) / (count_ + 1);
    } else {
        // Wilder's smoothing: ATR = prev_ATR * (1 - 1/n) + TR * (1/n)
        atr_value_ = atr_value_ * (1.0 - alpha_) + tr * alpha_;
    }

    prev_close_ = close;
    has_prev_   = true;
    ++count_;
    return atr_value_;
}

double ATR::update(const OHLCVBar& bar) noexcept {
    return update(bar.high, bar.low, bar.close);
}

void ATR::reset() noexcept {
    atr_value_ = 0.0;
    prev_close_ = 0.0;
    count_     = 0;
    has_prev_  = false;
}

// ============================================================
// Standalone helpers
// ============================================================

/// Compute True Range for a single bar given previous close.
double true_range(double high, double low, double prev_close) noexcept {
    double hl  = high - low;
    double hpc = std::abs(high - prev_close);
    double lpc = std::abs(low  - prev_close);
    return std::max({hl, hpc, lpc});
}

/// Compute a batch of ATR values from an array of bars.
/// out_atr must be pre-allocated with n elements.
void atr_batch(const OHLCVBar* bars, std::size_t n, int period,
               double* out_atr) noexcept {
    ATR atr(period);
    for (std::size_t i = 0; i < n; ++i) {
        out_atr[i] = atr.update(bars[i]);
    }
}

} // namespace srfm
