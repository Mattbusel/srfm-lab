// market_breadth.cpp -- Out-of-line implementations for market breadth.
//
// Header-only core lives in market_breadth.hpp.  This TU provides:
//   - Batch SIMD helpers for advance/decline counting across large universes.
//   - NH/NL computation with AVX2-accelerated min/max reduction.
//   - Sector-level breadth aggregation.
//
// Compile with: -std=c++17 -O3 -mavx2 -mfma

#include "indicators/market_breadth.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace srfm {
namespace indicators {

// ------------------------------------------------------------------
// Batch return sign classification (AVX2)
// ------------------------------------------------------------------

/// Classify n returns into advance/decline/unchanged counts.
/// AVX2 processes 4 doubles per cycle, comparing to zero.
void classify_returns_avx2(const double* __restrict__ returns,
                             std::size_t                n,
                             int&                       out_adv,
                             int&                       out_dec,
                             int&                       out_unch) noexcept
{
    out_adv = out_dec = out_unch = 0;
#if defined(SRFM_AVX2)
    const __m256d vzero = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vr = _mm256_loadu_pd(returns + i);
        int gt_mask = _mm256_movemask_pd(_mm256_cmp_pd(vr, vzero, _CMP_GT_OQ));
        int lt_mask = _mm256_movemask_pd(_mm256_cmp_pd(vr, vzero, _CMP_LT_OQ));
        out_adv  += __builtin_popcount(static_cast<unsigned>(gt_mask));
        out_dec  += __builtin_popcount(static_cast<unsigned>(lt_mask));
    }
    // tail
    for (; i < n; ++i) {
        if      (returns[i] > 0.0) ++out_adv;
        else if (returns[i] < 0.0) ++out_dec;
        else                       ++out_unch;
    }
    out_unch = static_cast<int>(n) - out_adv - out_dec;
#else
    for (std::size_t i = 0; i < n; ++i) {
        if      (returns[i] > 0.0) ++out_adv;
        else if (returns[i] < 0.0) ++out_dec;
        else                       ++out_unch;
    }
#endif
}

// ------------------------------------------------------------------
// NH/NL with AVX2 min/max reduction
// ------------------------------------------------------------------

/// For each instrument, find max and min price over lookback bars,
/// compare to current price, and count new highs/lows.
/// history layout: history[i * lookback + j] = price j bars ago for inst i.
void compute_nh_nl_avx2(const double* __restrict__ current_prices,
                          const double* __restrict__ history,
                          int                        n_instruments,
                          int                        lookback,
                          int&                       out_highs,
                          int&                       out_lows) noexcept
{
    out_highs = out_lows = 0;
    for (int i = 0; i < n_instruments; ++i) {
        double cp = current_prices[i];
        const double* h = history + i * lookback;
        double max_h = h[0], min_l = h[0];

#if defined(SRFM_AVX2)
        __m256d vmax = _mm256_set1_pd(h[0]);
        __m256d vmin = _mm256_set1_pd(h[0]);
        int j = 0;
        for (; j + 4 <= lookback; j += 4) {
            __m256d vd = _mm256_loadu_pd(h + j);
            vmax = _mm256_max_pd(vmax, vd);
            vmin = _mm256_min_pd(vmin, vd);
        }
        // Horizontal max/min
        alignas(32) double tmax[4], tmin[4];
        _mm256_storeu_pd(tmax, vmax);
        _mm256_storeu_pd(tmin, vmin);
        for (int k = 0; k < 4; ++k) {
            if (tmax[k] > max_h) max_h = tmax[k];
            if (tmin[k] < min_l) min_l = tmin[k];
        }
        // scalar tail
        for (; j < lookback; ++j) {
            if (h[j] > max_h) max_h = h[j];
            if (h[j] < min_l) min_l = h[j];
        }
#else
        for (int j = 1; j < lookback; ++j) {
            if (h[j] > max_h) max_h = h[j];
            if (h[j] < min_l) min_l = h[j];
        }
#endif
        if (cp >= max_h) ++out_highs;
        if (cp <= min_l) ++out_lows;
    }
}

// ------------------------------------------------------------------
// Percent above MA with AVX2
// ------------------------------------------------------------------

/// Compute percent of n_instruments whose current price exceeds their
/// period-bar SMA.  Returns value in [0, 1].
double compute_pct_above_ma_avx2(const double* __restrict__ current_prices,
                                   const double* __restrict__ history,
                                   int                        n_instruments,
                                   int                        ma_period) noexcept
{
    if (n_instruments == 0 || ma_period <= 0) return 0.0;
    int above = 0;
    double inv_period = 1.0 / ma_period;

    for (int i = 0; i < n_instruments; ++i) {
        const double* h = history + i * ma_period;
        double sum = 0.0;
#if defined(SRFM_AVX2)
        __m256d vacc = _mm256_setzero_pd();
        int j = 0;
        for (; j + 4 <= ma_period; j += 4)
            vacc = _mm256_add_pd(vacc, _mm256_loadu_pd(h + j));
        sum = simd::hsum_avx2(vacc);
        for (; j < ma_period; ++j) sum += h[j];
#else
        for (int j = 0; j < ma_period; ++j) sum += h[j];
#endif
        double ma = sum * inv_period;
        if (current_prices[i] > ma) ++above;
    }
    return static_cast<double>(above) / n_instruments;
}

// ------------------------------------------------------------------
// Sector breadth breakdown
// ------------------------------------------------------------------

/// Per-sector advance/decline summary.
struct SectorBreadth {
    int    sector_id    = 0;
    int    advances     = 0;
    int    declines     = 0;
    int    unchanged    = 0;
    double ad_ratio     = 0.0;
    double adv_volume   = 0.0;
    double dec_volume   = 0.0;
    double arms_index   = 0.0;
};

/// Compute per-sector breadth from instrument returns/volumes and a
/// sector_ids array mapping instrument i to its sector.
std::vector<SectorBreadth> compute_sector_breadth(
    const double* returns,
    const double* volumes,
    const int*    sector_ids,
    int           n_instruments,
    int           n_sectors)
{
    std::vector<SectorBreadth> result(static_cast<std::size_t>(n_sectors));
    for (int s = 0; s < n_sectors; ++s) result[static_cast<std::size_t>(s)].sector_id = s;

    for (int i = 0; i < n_instruments; ++i) {
        int s = sector_ids[i];
        if (s < 0 || s >= n_sectors) continue;
        auto& sb = result[static_cast<std::size_t>(s)];
        if      (returns[i] > 0.0) { ++sb.advances; sb.adv_volume += volumes[i]; }
        else if (returns[i] < 0.0) { ++sb.declines; sb.dec_volume += volumes[i]; }
        else                        { ++sb.unchanged; }
    }

    for (auto& sb : result) {
        sb.ad_ratio   = (sb.declines > 0)
                        ? static_cast<double>(sb.advances) / sb.declines : 1.0;
        sb.arms_index = MarketBreadth::arms_index(
            sb.advances, sb.declines, sb.adv_volume, sb.dec_volume);
    }
    return result;
}

// ------------------------------------------------------------------
// McClellan Summation momentum (rate of change of summation index)
// ------------------------------------------------------------------

/// Returns the n-bar rate of change of the McClellan Summation Index.
double mcclellan_momentum(const std::deque<BreadthSnapshot>& history, int n) noexcept {
    int sz = static_cast<int>(history.size());
    if (sz < n + 1) return 0.0;
    double current = history[static_cast<std::size_t>(sz - 1)].mcclellan_summation;
    double prior   = history[static_cast<std::size_t>(sz - 1 - n)].mcclellan_summation;
    return current - prior;
}

} // namespace indicators
} // namespace srfm
