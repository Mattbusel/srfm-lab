#pragma once
#include "srfm/types.hpp"

namespace srfm {

/// Average True Range using Wilder's smoothing (alpha = 1/period).
class ATR {
public:
    explicit ATR(int period = 14) noexcept;

    /// Update with a full OHLCV bar. Returns current ATR.
    double update(const OHLCVBar& bar) noexcept;

    /// Update with explicit high, low, close values.
    double update(double high, double low, double close) noexcept;

    double value()    const noexcept { return atr_value_; }
    bool   is_warm()  const noexcept { return count_ >= period_; }
    int    count()    const noexcept { return count_; }

    void reset() noexcept;

private:
    double true_range(double high, double low, double close) const noexcept;

    int    period_;
    double alpha_;       // Wilder's alpha = 1/period
    double atr_value_;
    double prev_close_;
    int    count_;
    bool   has_prev_;
};

// Standalone helpers
double true_range(double high, double low, double prev_close) noexcept;
void   atr_batch(const OHLCVBar* bars, std::size_t n, int period,
                 double* out_atr) noexcept;

} // namespace srfm
