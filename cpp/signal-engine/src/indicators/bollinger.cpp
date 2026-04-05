#include "bollinger.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace srfm {

BollingerBands::BollingerBands(int period, double num_std) noexcept
    : period_(std::min(period, MAX_PERIOD))
    , num_std_(num_std)
    , mean_(0.0)
    , m2_(0.0)
    , win_idx_(0)
    , count_(0)
    , upper_(0.0)
    , lower_(0.0)
    , std_dev_(0.0)
    , pct_b_(0.5)
    , bandwidth_(0.0)
{
    std::memset(window_, 0, sizeof(window_));
}

BollingerOutput BollingerBands::update(double close) noexcept {
    // Welford's online algorithm with sliding window removal
    int n = count_ + 1;

    if (n <= period_) {
        // Growing phase: add without removal
        double delta  = close - mean_;
        mean_        += delta / n;
        double delta2 = close - mean_;
        m2_          += delta * delta2;
        window_[win_idx_] = close;
        win_idx_           = (win_idx_ + 1) % period_;
        count_             = n;
    } else {
        // Sliding window: remove oldest, add new
        double oldest   = window_[win_idx_];
        double old_mean = mean_;

        // Update mean
        mean_    += (close - oldest) / period_;

        // Update M2 using the difference-of-squares update
        m2_      += (close - oldest) * (close - mean_ + oldest - old_mean);

        // Clamp M2 to avoid floating-point drift below zero
        if (m2_ < 0.0) m2_ = 0.0;

        window_[win_idx_] = close;
        win_idx_           = (win_idx_ + 1) % period_;
        count_             = n;
    }

    recompute_bands();

    return { upper_, mean_, lower_, pct_b_, bandwidth_, is_warm() };
}

void BollingerBands::recompute_bands() noexcept {
    int effective = std::min(count_, period_);
    if (effective < 2) {
        std_dev_ = 0.0;
        upper_   = mean_;
        lower_   = mean_;
        pct_b_   = 0.5;
        bandwidth_ = 0.0;
        return;
    }

    std_dev_ = std::sqrt(m2_ / effective);
    upper_   = mean_ + num_std_ * std_dev_;
    lower_   = mean_ - num_std_ * std_dev_;

    double band_width = upper_ - lower_;
    if (band_width > constants::EPSILON) {
        // Note: pct_b_ is computed against latest close, but we don't store it
        // here directly; we compute it in update() by passing close.
        // Since recompute_bands() is called after updating window_, we use
        // the current write position to find the most recent value.
        int latest_idx = (win_idx_ == 0) ? period_ - 1 : win_idx_ - 1;
        double last_close = window_[latest_idx];
        pct_b_     = (last_close - lower_) / band_width;
        bandwidth_ = band_width / (std::abs(mean_) > constants::EPSILON ? mean_ : 1.0);
    } else {
        pct_b_     = 0.5;
        bandwidth_ = 0.0;
    }
}

void BollingerBands::reset() noexcept {
    mean_    = 0.0;
    m2_      = 0.0;
    win_idx_ = 0;
    count_   = 0;
    upper_   = 0.0;
    lower_   = 0.0;
    std_dev_ = 0.0;
    pct_b_   = 0.5;
    bandwidth_ = 0.0;
    std::memset(window_, 0, sizeof(window_));
}

// ============================================================
// Standalone Bollinger band computation over historical array
// ============================================================

struct BollingerResult {
    double* upper;
    double* mid;
    double* lower;
    double* pct_b;
    double* bandwidth;
};

void bollinger_batch(const double* closes, std::size_t n,
                     int period, double num_std,
                     BollingerResult& out) noexcept {
    BollingerBands bb(period, num_std);
    for (std::size_t i = 0; i < n; ++i) {
        auto res = bb.update(closes[i]);
        if (out.upper)     out.upper[i]     = res.upper;
        if (out.mid)       out.mid[i]       = res.mid;
        if (out.lower)     out.lower[i]     = res.lower;
        if (out.pct_b)     out.pct_b[i]     = res.pct_b;
        if (out.bandwidth) out.bandwidth[i] = res.bandwidth;
    }
}

// ============================================================
// Squeeze detection: Bollinger inside Keltner Channel
// Returns true when bands are contracting (low volatility regime)
// ============================================================

bool bollinger_squeeze(double bb_bandwidth, double bb_bandwidth_ma,
                        double threshold_pct = 0.1) noexcept {
    // Squeeze: current bandwidth is below (1 + threshold) * moving average
    return bb_bandwidth < bb_bandwidth_ma * (1.0 + threshold_pct);
}

} // namespace srfm
