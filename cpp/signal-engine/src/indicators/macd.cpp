#include "macd.hpp"

namespace srfm {

MACD::MACD(int fast_period, int slow_period, int signal_period) noexcept
    : fast_(fast_period)
    , slow_(slow_period)
    , signal_(signal_period)
    , macd_line_(0.0)
    , signal_line_(0.0)
    , histogram_(0.0)
{}

MACDOutput MACD::update(double close) noexcept {
    double fv = fast_.update(close);
    double sv = slow_.update(close);

    // MACD line is only meaningful once slow EMA is warm
    if (slow_.is_warm()) {
        macd_line_ = fv - sv;
        signal_line_ = signal_.update(macd_line_);
        histogram_   = macd_line_ - signal_line_;
    }

    return { macd_line_, signal_line_, histogram_, is_warm() };
}

bool MACD::is_warm() const noexcept {
    return slow_.is_warm() && signal_.is_warm();
}

void MACD::reset() noexcept {
    fast_.reset();
    slow_.reset();
    signal_.reset();
    macd_line_   = 0.0;
    signal_line_ = 0.0;
    histogram_   = 0.0;
}

// ============================================================
// Standalone batch MACD computation
// ============================================================

struct MACDResult {
    double* macd_line;
    double* signal_line;
    double* histogram;
};

void macd_batch(const double* closes, std::size_t n,
                int fast_period, int slow_period, int signal_period,
                MACDResult& out) noexcept {
    MACD macd(fast_period, slow_period, signal_period);
    for (std::size_t i = 0; i < n; ++i) {
        auto res = macd.update(closes[i]);
        if (out.macd_line)   out.macd_line[i]   = res.macd_line;
        if (out.signal_line) out.signal_line[i] = res.signal_line;
        if (out.histogram)   out.histogram[i]   = res.histogram;
    }
}

// ============================================================
// MACD signal helpers
// ============================================================

/// Returns +1 for bullish MACD crossover, -1 for bearish, 0 for none.
int macd_crossover(double prev_hist, double curr_hist) noexcept {
    if (prev_hist < 0.0 && curr_hist >= 0.0) return  1;
    if (prev_hist > 0.0 && curr_hist <= 0.0) return -1;
    return 0;
}

/// MACD zero-line cross: MACD line crossing zero.
int macd_zero_cross(double prev_macd, double curr_macd) noexcept {
    if (prev_macd < 0.0 && curr_macd >= 0.0) return  1;
    if (prev_macd > 0.0 && curr_macd <= 0.0) return -1;
    return 0;
}

/// Check for MACD divergence with price.
/// Returns +1 bullish, -1 bearish, 0 none.
int macd_price_divergence(double price_change, double macd_change) noexcept {
    // Bearish: price rising but MACD falling
    if (price_change > 0.0 && macd_change < 0.0) return -1;
    // Bullish: price falling but MACD rising
    if (price_change < 0.0 && macd_change > 0.0) return  1;
    return 0;
}

} // namespace srfm
