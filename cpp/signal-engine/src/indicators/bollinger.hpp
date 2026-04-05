#pragma once
#include "srfm/types.hpp"

namespace srfm {

struct BollingerOutput {
    double upper;
    double mid;
    double lower;
    double pct_b;       // (close - lower) / (upper - lower)
    double bandwidth;   // (upper - lower) / mid
    bool   is_warm;
};

/// Bollinger Bands using Welford's online algorithm for rolling mean and variance.
class BollingerBands {
public:
    explicit BollingerBands(int period = 20, double num_std = 2.0) noexcept;

    /// Update with new close price. Returns band values.
    BollingerOutput update(double close) noexcept;

    double upper()     const noexcept { return upper_; }
    double mid()       const noexcept { return mean_; }
    double lower()     const noexcept { return lower_; }
    double pct_b()     const noexcept { return pct_b_; }
    double bandwidth() const noexcept { return bandwidth_; }
    bool   is_warm()   const noexcept { return count_ >= period_; }

    void reset() noexcept;

private:
    void recompute_bands() noexcept;

    int     period_;
    double  num_std_;

    // Welford's online algorithm state
    double  mean_;
    double  m2_;       // sum of squared deviations from mean

    // Ring buffer for rolling window
    static constexpr int MAX_PERIOD = 500;
    double  window_[MAX_PERIOD];
    int     win_idx_;    // write position in ring buffer
    int     count_;      // total bars seen

    // Cached output
    double  upper_;
    double  lower_;
    double  std_dev_;
    double  pct_b_;
    double  bandwidth_;
};

} // namespace srfm
