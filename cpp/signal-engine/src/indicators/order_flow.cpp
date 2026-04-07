// order_flow.cpp -- Out-of-line implementations for order flow indicators.
//
// Compilation: -std=c++17 -O3 -mavx2 -mfma
// All hot paths use AVX2 when __AVX2__ is defined.

#include "indicators/order_flow.hpp"
#include <cassert>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace srfm {
namespace indicators {

// ------------------------------------------------------------------
// Batch delta computation from arrays of OHLCVBars
// ------------------------------------------------------------------

/// Compute DeltaBar for each bar in [bars, bars+n) into out[].
/// Uses the blended tick rule for all bars.
void compute_delta_bars(const OHLCVBar* __restrict__ bars,
                         DeltaBar*       __restrict__ out,
                         std::size_t                  n) noexcept
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = DeltaBar::from_bar_blended(bars[i]);
}

/// Compute per-bar deltas into a raw double array (delta only).
/// AVX2 accelerated: processes price and volume in 4-wide vectors.
/// Note: OHLCVBar is 64 bytes; we stride manually.
void compute_delta_array(const OHLCVBar* __restrict__ bars,
                          double*         __restrict__ deltas,
                          std::size_t                  n) noexcept
{
#if defined(SRFM_AVX2)
    // Process 4 bars at a time where possible.
    // Each bar: delta = vol * (2 * frac - 1), frac = (close-low)/(high-low)
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // Gather fields (bars are 64-byte structs, not contiguous doubles)
        alignas(32) double closes[4], lows[4], highs[4], vols[4];
        for (int k = 0; k < 4; ++k) {
            closes[k] = bars[i + k].close;
            lows[k]   = bars[i + k].low;
            highs[k]  = bars[i + k].high;
            vols[k]   = bars[i + k].volume;
        }
        __m256d vc   = _mm256_loadu_pd(closes);
        __m256d vl   = _mm256_loadu_pd(lows);
        __m256d vh   = _mm256_loadu_pd(highs);
        __m256d vv   = _mm256_loadu_pd(vols);
        __m256d vone = _mm256_set1_pd(1.0);
        __m256d vtwo = _mm256_set1_pd(2.0);
        __m256d veps = _mm256_set1_pd(1e-12);

        __m256d range = _mm256_add_pd(_mm256_sub_pd(vh, vl), veps);
        __m256d diff  = _mm256_sub_pd(vc, vl);
        // frac = clamp(diff/range, 0, 1)
        __m256d frac  = _mm256_div_pd(diff, range);
        frac          = _mm256_min_pd(_mm256_max_pd(frac, _mm256_setzero_pd()), vone);
        // delta = vol * (2*frac - 1)
        __m256d d2f   = _mm256_fmadd_pd(vtwo, frac, _mm256_set1_pd(-1.0));
        __m256d delta = _mm256_mul_pd(vv, d2f);
        _mm256_storeu_pd(deltas + i, delta);
    }
    for (; i < n; ++i) {
        DeltaBar db = DeltaBar::from_bar_blended(bars[i]);
        deltas[i] = db.delta;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        DeltaBar db = DeltaBar::from_bar_blended(bars[i]);
        deltas[i] = db.delta;
    }
#endif
}

/// Compute the running cumulative sum of delta array.
/// out[i] = sum(deltas[0..i]).
void cumulate_deltas(const double* __restrict__ deltas,
                      double*        __restrict__ out,
                      std::size_t                  n) noexcept
{
    double acc = 0.0;
#if defined(SRFM_AVX2)
    // Prefix sum in blocks of 4 -- each block adds to running scalar acc.
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vd = _mm256_loadu_pd(deltas + i);
        // Horizontal prefix within block (serial by dependency)
        alignas(32) double tmp[4];
        _mm256_storeu_pd(tmp, vd);
        tmp[0] += acc;
        tmp[1] += tmp[0];
        tmp[2] += tmp[1];
        tmp[3] += tmp[2];
        _mm256_storeu_pd(out + i, _mm256_loadu_pd(tmp));
        acc = tmp[3];
    }
    for (; i < n; ++i) {
        acc    += deltas[i];
        out[i]  = acc;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        acc    += deltas[i];
        out[i]  = acc;
    }
#endif
}

// ------------------------------------------------------------------
// OrderFlowImbalance -- batch helpers
// ------------------------------------------------------------------

/// Detect absorption across N bars simultaneously.
/// Returns bitmask (1 per bar) indicating absorption at that bar.
/// ATR array must be pre-computed.
void detect_absorption_batch(const OHLCVBar* __restrict__ bars,
                               const double*   __restrict__ atrs,
                               const double*   __restrict__ avg_volumes,
                               uint8_t*        __restrict__ flags,
                               std::size_t                   n,
                               double price_move_threshold,
                               double vol_mult) noexcept
{
    for (std::size_t i = 0; i < n; ++i) {
        double pc = bars[i].close - bars[i].open;
        flags[i]  = static_cast<uint8_t>(
            OrderFlowImbalance::absorption_detection(
                pc, atrs[i], bars[i].volume, avg_volumes[i],
                price_move_threshold, vol_mult) ? 1 : 0);
    }
}

/// Returns a Z-score of delta relative to its rolling mean and std-dev.
/// Useful for scaling imbalance signals.
double delta_zscore(const DeltaBar* history, int n, double current_delta) noexcept {
    if (n < 2) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += history[i].delta;
    double mean = sum / n;
    double var  = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = history[i].delta - mean;
        var += d * d;
    }
    double sd = std::sqrt(var / (n - 1));
    if (sd < 1e-12) return 0.0;
    return (current_delta - mean) / sd;
}

// ------------------------------------------------------------------
// FootprintBar -- aggregation helpers
// ------------------------------------------------------------------

/// Aggregate multiple FootprintBars into a single composite footprint.
/// All bars must share the same tick_size.
FootprintBar aggregate_footprints(const std::vector<FootprintBar>& bars) {
    if (bars.empty()) return FootprintBar{};
    FootprintBar result(bars[0].tick_size());

    for (auto& fb : bars) {
        for (auto& level : fb.levels()) {
            result.add_tick(level.price, level.buy_vol,  +1);
            result.add_tick(level.price, level.sell_vol, -1);
        }
    }
    return result;
}

/// Compute the delta profile (price -> delta map) as parallel arrays.
/// out_prices and out_deltas must have capacity >= footprint.level_count().
void footprint_delta_profile(const FootprintBar& fp,
                              double* out_prices,
                              double* out_deltas,
                              std::size_t& out_n) noexcept
{
    out_n = fp.level_count();
    for (std::size_t i = 0; i < out_n; ++i) {
        out_prices[i] = fp.levels()[i].price;
        out_deltas[i] = fp.levels()[i].delta();
    }
}

// ------------------------------------------------------------------
// Session analytics
// ------------------------------------------------------------------

/// Compute buy/sell ratio across a session (array of DeltaBars).
/// Returns buy_vol / sell_vol.  Returns 1.0 on empty or zero sell.
double session_buy_sell_ratio(const DeltaBar* bars, std::size_t n) noexcept {
    double total_buy  = 0.0;
    double total_sell = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        total_buy  += bars[i].buy_volume;
        total_sell += bars[i].sell_volume;
    }
    if (total_sell < 1e-12) return 1.0;
    return total_buy / total_sell;
}

/// Compute VWAP delta -- volume-weighted average delta per unit volume.
double session_vwap_delta(const DeltaBar* bars, std::size_t n) noexcept {
    double total_vol = 0.0;
    double total_dv  = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        total_vol += bars[i].volume;
        total_dv  += bars[i].delta * bars[i].volume;
    }
    if (total_vol < 1e-12) return 0.0;
    return total_dv / total_vol;
}

/// Identify the bar index with maximum absolute delta (conviction bar).
int peak_delta_bar(const DeltaBar* bars, std::size_t n) noexcept {
    if (n == 0) return -1;
    int best = 0;
    double best_abs = std::abs(bars[0].delta);
    for (std::size_t i = 1; i < n; ++i) {
        double a = std::abs(bars[i].delta);
        if (a > best_abs) {
            best_abs = a;
            best     = static_cast<int>(i);
        }
    }
    return best;
}

} // namespace indicators
} // namespace srfm
