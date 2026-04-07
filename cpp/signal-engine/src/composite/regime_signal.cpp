// regime_signal.cpp -- Out-of-line implementations for the regime signal engine.
//
// All hot computation paths are in the header (inlined).  This TU provides:
//   - batch_compute(): process an array of bars for one symbol in a tight loop
//   - Regime transition analytics
//   - Serialisation helpers for RegimeSignal to a JSON-like string
//
// Compile with: -std=c++17 -O3 -mavx2 -mfma

#include "composite/regime_signal.hpp"
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace srfm {
namespace composite {

// ------------------------------------------------------------------
// Batch processing
// ------------------------------------------------------------------

/// Process an array of bars through the engine, writing RegimeSignal into
/// `out[]` (must have capacity n).  Returns number of signals written (== n).
std::size_t batch_compute(RegimeSignalEngine&    engine,
                            const OHLCVBar* __restrict__ bars,
                            RegimeSignal*   __restrict__ out,
                            std::size_t                  n) noexcept
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = engine.compute(bars[i]);
    return n;
}

// ------------------------------------------------------------------
// Regime transition detection
// ------------------------------------------------------------------

/// A regime transition event.
struct RegimeTransition {
    int64_t timestamp_ns;
    int     from_regime;
    int     to_regime;
    int     bar_index;
    double  composite_at_transition;
};

/// Scan an array of RegimeSignal for regime changes.
/// Appends transitions to out_transitions.
void detect_regime_transitions(const RegimeSignal*             signals,
                                 std::size_t                      n,
                                 std::vector<RegimeTransition>&   out_transitions)
{
    if (n == 0) return;
    int current = signals[0].regime_idx;
    for (std::size_t i = 1; i < n; ++i) {
        int next = signals[i].regime_idx;
        if (next != current) {
            out_transitions.push_back({
                signals[i].timestamp_ns,
                current,
                next,
                static_cast<int>(i),
                signals[i].composite
            });
            current = next;
        }
    }
}

/// Compute the fraction of bars spent in each of the 4 regimes.
/// out[4] receives fractions (sum = 1.0 if n > 0).
void regime_distribution(const RegimeSignal* signals,
                           std::size_t         n,
                           double              out[4]) noexcept
{
    out[0] = out[1] = out[2] = out[3] = 0.0;
    if (n == 0) return;
    int counts[4] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < n; ++i) {
        int r = signals[i].regime_idx;
        if (r >= 0 && r < 4) ++counts[r];
    }
    for (int r = 0; r < 4; ++r)
        out[r] = static_cast<double>(counts[r]) / n;
}

// ------------------------------------------------------------------
// Composite score statistics
// ------------------------------------------------------------------

/// Mean of composite scores over n bars.
double composite_mean(const RegimeSignal* signals, std::size_t n) noexcept {
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += signals[i].composite;
    return sum / n;
}

/// Standard deviation of composite scores.
double composite_stddev(const RegimeSignal* signals, std::size_t n) noexcept {
    if (n < 2) return 0.0;
    double mean = composite_mean(signals, n);
    double var  = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = signals[i].composite - mean;
        var += d * d;
    }
    return std::sqrt(var / (n - 1));
}

/// Z-score of the latest composite relative to a rolling window.
double composite_zscore(const RegimeSignal* signals,
                          std::size_t         n,
                          int                 window) noexcept
{
    if (n == 0) return 0.0;
    int len = std::min(static_cast<int>(n), window);
    const RegimeSignal* start = signals + (n - static_cast<std::size_t>(len));
    double mean = composite_mean(start, static_cast<std::size_t>(len));
    double sd   = composite_stddev(start, static_cast<std::size_t>(len));
    if (sd < 1e-12) return 0.0;
    return (signals[n - 1].composite - mean) / sd;
}

// ------------------------------------------------------------------
// JSON serialisation (for debugging / IPC)
// ------------------------------------------------------------------

/// Serialise a single RegimeSignal to a compact JSON object string.
std::string regime_signal_to_json(const RegimeSignal& s) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{"
        << "\"ts\":"      << s.timestamp_ns    << ","
        << "\"sym\":"     << s.symbol_id        << ","
        << "\"regime\":\"" << regime_label(s.regime_idx) << "\","
        << "\"composite\":" << s.composite      << ","
        << "\"trend\":"    << s.trend_score      << ","
        << "\"momentum\":" << s.momentum_score   << ","
        << "\"vol\":"      << s.vol_score        << ","
        << "\"breadth\":"  << s.breadth_score    << ","
        << "\"flow\":"     << s.flow_score        << ","
        << "\"rsi\":"      << s.rsi               << ","
        << "\"macd_hist\":" << s.macd_hist        << ","
        << "\"adx\":"      << s.adx               << ","
        << "\"atr\":"      << s.atr               << ","
        << "\"ema200\":"   << s.ema200             << ","
        << "\"poc\":"      << s.poc               << ","
        << "\"of_imb\":"   << s.of_imbalance       << ","
        << "\"diverge\":"  << (s.of_divergence ? 1 : 0) << ","
        << "\"absorb\":"   << (s.of_absorption  ? 1 : 0)
        << "}";
    return oss.str();
}

/// Serialise an array of n signals to a JSON array string.
std::string regime_signals_to_json_array(const RegimeSignal* signals,
                                           std::size_t         n)
{
    std::string out;
    out.reserve(n * 300);
    out += "[";
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) out += ",";
        out += regime_signal_to_json(signals[i]);
    }
    out += "]";
    return out;
}

// ------------------------------------------------------------------
// RegimeSignalSummary -- aggregate stats over a lookback window
// ------------------------------------------------------------------

struct RegimeSignalSummary {
    double mean_composite    = 0.0;
    double std_composite     = 0.0;
    double z_composite       = 0.0;
    int    dominant_regime   = 0;
    double dominant_fraction = 0.0;
    int    transition_count  = 0;
    int    bars_in_regime    = 0;
};

/// Summarise the last `window` signals in the history.
RegimeSignalSummary summarise(const RegimeSignal* signals,
                               std::size_t         n,
                               int                 window = 50) noexcept
{
    RegimeSignalSummary out;
    if (n == 0) return out;
    int len = std::min(static_cast<int>(n), window);
    const RegimeSignal* start = signals + (n - static_cast<std::size_t>(len));

    out.mean_composite = composite_mean (start, static_cast<std::size_t>(len));
    out.std_composite  = composite_stddev(start, static_cast<std::size_t>(len));
    out.z_composite    = (out.std_composite > 1e-12)
                         ? (signals[n-1].composite - out.mean_composite) / out.std_composite
                         : 0.0;

    // Dominant regime
    int counts[4] = {0, 0, 0, 0};
    int current = start[0].regime_idx;
    int trans = 0;
    for (int i = 0; i < len; ++i) {
        int r = start[i].regime_idx;
        if (r >= 0 && r < 4) ++counts[r];
        if (i > 0 && r != start[i-1].regime_idx) ++trans;
    }
    int dom = static_cast<int>(std::max_element(counts, counts + 4) - counts);
    out.dominant_regime   = dom;
    out.dominant_fraction = static_cast<double>(counts[dom]) / len;
    out.transition_count  = trans;

    // Current regime run length
    current = signals[n-1].regime_idx;
    out.bars_in_regime = 0;
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        if (signals[i].regime_idx == current) ++out.bars_in_regime;
        else break;
    }

    return out;
}

} // namespace composite
} // namespace srfm
