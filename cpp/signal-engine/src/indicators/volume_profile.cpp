// volume_profile.cpp -- Implementation of VolumeProfile / RollingVolumeProfile.
//
// The header-only portion lives in volume_profile.hpp (all methods are either
// inline or templated).  This translation unit houses:
//   - Out-of-line definitions kept here for SIMD specialisations that benefit
//     from a dedicated compilation unit compiled with -mavx2 -mfma.
//   - A thin factory API for constructing profiles from raw bar arrays.
//
// Build with: -std=c++17 -O3 -mavx2 -mfma

#include "indicators/volume_profile.hpp"
#include <cassert>
#include <numeric>
#include <cmath>

namespace srfm {
namespace indicators {

// ------------------------------------------------------------------
// Factory helpers
// ------------------------------------------------------------------

/// Build a VolumeProfile from parallel price / volume arrays of length n.
/// price_lo / price_hi define the initial bin range; pass (0,0) for auto.
VolumeProfile make_volume_profile(const double* prices,
                                   const double* volumes,
                                   std::size_t   n,
                                   double        price_lo,
                                   double        price_hi)
{
    VolumeProfile vp;
    if (n == 0) return vp;

    // Auto-range if not specified
    if (price_lo >= price_hi) {
        price_lo = prices[0];
        price_hi = prices[0];
        for (std::size_t i = 1; i < n; ++i) {
            if (prices[i] < price_lo) price_lo = prices[i];
            if (prices[i] > price_hi) price_hi = prices[i];
        }
        double margin = (price_hi - price_lo) * 0.05 + 1.0;
        price_lo -= margin;
        price_hi += margin;
    }
    vp.init_range(price_lo, price_hi);

    // SIMD-friendly loop -- update() is a single bin write, loop is memory
    // bandwidth bound; unroll 4x for prefetch friendliness.
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vp.update(prices[i + 0], volumes[i + 0]);
        vp.update(prices[i + 1], volumes[i + 1]);
        vp.update(prices[i + 2], volumes[i + 2]);
        vp.update(prices[i + 3], volumes[i + 3]);
    }
    for (; i < n; ++i)
        vp.update(prices[i], volumes[i]);

    return vp;
}

/// Build a VolumeProfile directly from OHLCVBar array.
/// Uses the (high+low)/2 typical price and the bar's volume field.
VolumeProfile make_volume_profile_from_bars(const OHLCVBar* bars,
                                             std::size_t     n)
{
    if (n == 0) return {};

    double lo = (bars[0].high + bars[0].low) * 0.5;
    double hi = lo;
    for (std::size_t i = 1; i < n; ++i) {
        double tp = (bars[i].high + bars[i].low) * 0.5;
        if (tp < lo) lo = tp;
        if (tp > hi) hi = tp;
    }
    double margin = (hi - lo) * 0.05 + 1.0;
    lo -= margin;
    hi += margin;

    VolumeProfile vp;
    vp.init_range(lo, hi);

    for (std::size_t i = 0; i < n; ++i) {
        double tp = (bars[i].high + bars[i].low) * 0.5;
        vp.update(tp, bars[i].volume);
    }
    return vp;
}

// ------------------------------------------------------------------
// Profile statistics
// ------------------------------------------------------------------

/// Compute the skewness of the volume distribution.
/// Positive skew = volume concentrated below POC (bearish lean).
double profile_skewness(const VolumeProfile& vp) noexcept {
    if (vp.total_volume() == 0.0) return 0.0;

    double mean  = 0.0;
    double var   = 0.0;
    double skew  = 0.0;
    double total = vp.total_volume();

    for (int i = 0; i < VP_BINS; ++i) {
        double w = vp.bin_volume(i) / total;
        double x = vp.bin_centre(i);
        mean += w * x;
    }
    for (int i = 0; i < VP_BINS; ++i) {
        double w  = vp.bin_volume(i) / total;
        double dx = vp.bin_centre(i) - mean;
        var  += w * dx * dx;
        skew += w * dx * dx * dx;
    }
    double sd = std::sqrt(var);
    if (sd < 1e-12) return 0.0;
    return skew / (sd * sd * sd);
}

/// Compute the kurtosis (excess) of the volume distribution.
/// High kurtosis = very peaked profile (thin market, single price dominates).
double profile_kurtosis(const VolumeProfile& vp) noexcept {
    if (vp.total_volume() == 0.0) return 0.0;

    double mean  = 0.0;
    double var   = 0.0;
    double kurt  = 0.0;
    double total = vp.total_volume();

    for (int i = 0; i < VP_BINS; ++i) {
        double w = vp.bin_volume(i) / total;
        double x = vp.bin_centre(i);
        mean += w * x;
    }
    for (int i = 0; i < VP_BINS; ++i) {
        double w  = vp.bin_volume(i) / total;
        double dx = vp.bin_centre(i) - mean;
        double d2 = dx * dx;
        var  += w * d2;
        kurt += w * d2 * d2;
    }
    double var2 = var * var;
    if (var2 < 1e-24) return 0.0;
    return (kurt / var2) - 3.0; // excess kurtosis
}

/// Compute VWAP from a VolumeProfile (volume-weighted average price of bins).
double profile_vwap(const VolumeProfile& vp) noexcept {
    if (vp.total_volume() == 0.0) return 0.0;

    double num = 0.0;
#if defined(SRFM_AVX2)
    __m256d vacc = _mm256_setzero_pd();
    for (int i = 0; i + 4 <= VP_BINS; i += 4) {
        // bin centres: lo + (i+0.5)*w, ..., lo + (i+3.5)*w
        __m256d vc = _mm256_set_pd(
            vp.bin_centre(i + 3),
            vp.bin_centre(i + 2),
            vp.bin_centre(i + 1),
            vp.bin_centre(i + 0));
        __m256d vv = _mm256_loadu_pd(vp.bins().data() + i);
        vacc = _mm256_fmadd_pd(vc, vv, vacc);
    }
    num = simd::hsum_avx2(vacc);
    for (int i = (VP_BINS / 4) * 4; i < VP_BINS; ++i)
        num += vp.bin_centre(i) * vp.bin_volume(i);
#else
    for (int i = 0; i < VP_BINS; ++i)
        num += vp.bin_centre(i) * vp.bin_volume(i);
#endif
    return num / vp.total_volume();
}

/// Returns the fraction of total volume traded at or below price.
/// Useful for percentile rank of current price in the profile.
double profile_volume_below(const VolumeProfile& vp, double price) noexcept {
    if (vp.total_volume() == 0.0) return 0.0;
    double acc = 0.0;
    for (int i = 0; i < VP_BINS; ++i) {
        if (vp.bin_centre(i) <= price)
            acc += vp.bin_volume(i);
        else
            break;
    }
    return acc / vp.total_volume();
}

} // namespace indicators
} // namespace srfm
