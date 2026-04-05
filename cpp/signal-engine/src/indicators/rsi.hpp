#pragma once
#include "srfm/types.hpp"

namespace srfm {

/// RSI using Wilder's smoothing method.
/// Default 14-period. State: avg_gain, avg_loss.
class RSI {
public:
    explicit RSI(int period = 14) noexcept;

    /// Update with a new close price. Returns RSI in [0, 100].
    double update(double close) noexcept;

    double value()   const noexcept { return rsi_value_; }
    bool   is_warm() const noexcept { return count_ > period_; }
    int    count()   const noexcept { return count_; }

    /// Convenience: returns true if oversold (RSI < threshold, default 30).
    bool is_oversold(double threshold = 30.0)  const noexcept { return rsi_value_ < threshold; }
    /// Returns true if overbought (RSI > threshold, default 70).
    bool is_overbought(double threshold = 70.0) const noexcept { return rsi_value_ > threshold; }

    void reset() noexcept;

private:
    int    period_;
    double alpha_;       // Wilder = 1/period
    double avg_gain_;
    double avg_loss_;
    double prev_close_;
    double rsi_value_;
    int    count_;
    // Accumulator for initial simple average
    double sum_gain_;
    double sum_loss_;
    bool   has_prev_;
};

} // namespace srfm
