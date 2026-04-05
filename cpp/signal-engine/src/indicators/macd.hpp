#pragma once
#include "srfm/types.hpp"
#include "ema.hpp"

namespace srfm {

struct MACDOutput {
    double macd_line;    // fast_ema - slow_ema
    double signal_line;  // EMA of macd_line
    double histogram;    // macd_line - signal_line
    bool   is_warm;
};

/// MACD: Moving Average Convergence/Divergence.
/// Default periods: fast=12, slow=26, signal=9.
class MACD {
public:
    MACD(int fast_period  = 12,
         int slow_period  = 26,
         int signal_period = 9) noexcept;

    MACDOutput update(double close) noexcept;

    double macd_line()   const noexcept { return macd_line_; }
    double signal_line() const noexcept { return signal_line_; }
    double histogram()   const noexcept { return histogram_; }
    bool   is_warm()     const noexcept;

    void reset() noexcept;

private:
    EMA    fast_;
    EMA    slow_;
    EMA    signal_;
    double macd_line_;
    double signal_line_;
    double histogram_;
};

} // namespace srfm
