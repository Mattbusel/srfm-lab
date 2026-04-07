#pragma once
// market_breadth.hpp -- Market breadth indicators across an instrument universe.
//
// Provides:
//   MarketBreadth  -- stateless batch computations (A/D ratio, NH/NL,
//                     %above MA, McClellan oscillator, TRIN/Arms Index)
//   BreadthHistory -- rolling history of all breadth metrics

#include "srfm/types.hpp"
#include "srfm/simd_math.hpp"
#include <array>
#include <vector>
#include <deque>
#include <span>
#include <utility>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>

namespace srfm {
namespace indicators {

// ------------------------------------------------------------------
// BreadthSnapshot -- one bar's worth of breadth readings
// ------------------------------------------------------------------

struct BreadthSnapshot {
    int64_t timestamp_ns       = 0;
    double  ad_ratio           = 0.0; // advances / declines
    double  ad_net             = 0.0; // advances - declines
    int     advances           = 0;
    int     declines           = 0;
    int     unchanged          = 0;
    int     new_highs          = 0;
    int     new_lows           = 0;
    double  pct_above_ma       = 0.0; // 0..1
    double  mcclellan_osc      = 0.0;
    double  mcclellan_summation= 0.0;
    double  arms_index         = 0.0; // TRIN
};

// ------------------------------------------------------------------
// MarketBreadth -- stateless and stateful breadth computations
// ------------------------------------------------------------------

class MarketBreadth {
public:
    // ------------------------------------------------------------------
    // Stateless batch methods (operate on std::span)
    // ------------------------------------------------------------------

    /// Advance / Decline ratio from a span of per-instrument returns.
    /// Returns advances / declines.  Returns 1.0 if no declines.
    static double advance_decline_ratio(std::span<const double> returns) noexcept {
        int adv = 0, dec = 0;
        for (double r : returns) {
            if      (r > 0.0) ++adv;
            else if (r < 0.0) ++dec;
        }
        if (dec == 0) return static_cast<double>(adv > 0 ? adv : 1);
        return static_cast<double>(adv) / static_cast<double>(dec);
    }

    /// Count instruments making new highs or lows vs their `lookback`-bar
    /// price history.
    /// prices[i] = current price of instrument i (n instruments).
    /// history[i * lookback + j] = price j bars ago for instrument i.
    static std::pair<int,int> new_highs_new_lows(
        std::span<const double> current_prices,
        const double*           history,       // [n * lookback]
        int                     lookback) noexcept
    {
        int n = static_cast<int>(current_prices.size());
        int highs = 0, lows = 0;
        for (int i = 0; i < n; ++i) {
            double cp  = current_prices[i];
            double max_h = cp, min_l = cp;
            const double* h = history + i * lookback;
            for (int j = 0; j < lookback; ++j) {
                if (h[j] > max_h) max_h = h[j];
                if (h[j] < min_l) min_l = h[j];
            }
            if (cp >= max_h) ++highs;
            if (cp <= min_l) ++lows;
        }
        return {highs, lows};
    }

    /// Percent of instruments whose current price is above their `period`-bar
    /// simple moving average.  Returns value in [0.0, 1.0].
    static double percent_above_ma(std::span<const double> current_prices,
                                    const double*           history,
                                    int                     lookback) noexcept
    {
        int n = static_cast<int>(current_prices.size());
        if (n == 0 || lookback <= 0) return 0.0;
        int above = 0;
        for (int i = 0; i < n; ++i) {
            // Compute SMA from history row
            const double* h = history + i * lookback;
            double sum = 0.0;
#if defined(SRFM_AVX2)
            __m256d acc = _mm256_setzero_pd();
            int j = 0;
            for (; j + 4 <= lookback; j += 4)
                acc = _mm256_add_pd(acc, _mm256_loadu_pd(h + j));
            sum = simd::hsum_avx2(acc);
            for (; j < lookback; ++j) sum += h[j];
#else
            for (int j = 0; j < lookback; ++j) sum += h[j];
#endif
            double ma = sum / lookback;
            if (current_prices[i] > ma) ++above;
        }
        return static_cast<double>(above) / n;
    }

    /// McClellan Oscillator: 19-day EMA of (A-D net) minus 39-day EMA of (A-D net).
    /// Uses exponential smoothing: this overload computes one step given
    /// prior EMA values.
    static double mcclellan_oscillator_step(double ad_net,
                                             double prior_ema19,
                                             double prior_ema39,
                                             double& out_ema19,
                                             double& out_ema39) noexcept
    {
        constexpr double alpha19 = 2.0 / 20.0;
        constexpr double alpha39 = 2.0 / 40.0;
        out_ema19 = alpha19 * ad_net + (1.0 - alpha19) * prior_ema19;
        out_ema39 = alpha39 * ad_net + (1.0 - alpha39) * prior_ema39;
        return out_ema19 - out_ema39;
    }

    /// Arms Index (TRIN):
    ///   TRIN = (advances / declines) / (adv_volume / dec_volume)
    ///   < 1.0 = bullish money flow; > 1.0 = bearish.
    static double arms_index(double advances, double declines,
                               double adv_volume, double dec_volume) noexcept
    {
        if (declines < 1e-12 || dec_volume < 1e-12) return 1.0;
        double ad_ratio  = advances / declines;
        double vol_ratio = adv_volume / dec_volume;
        if (vol_ratio < 1e-12) return 1.0;
        return ad_ratio / vol_ratio;
    }

    /// Batch advance/decline computation from an array of returns.
    /// out_adv, out_dec, out_unch must point to int arrays of length 1.
    static void count_advances_declines(std::span<const double> returns,
                                         int& out_adv,
                                         int& out_dec,
                                         int& out_unch) noexcept
    {
        out_adv = out_dec = out_unch = 0;
        for (double r : returns) {
            if      (r > 0.0) ++out_adv;
            else if (r < 0.0) ++out_dec;
            else              ++out_unch;
        }
    }

    // ------------------------------------------------------------------
    // Stateful update -- maintains rolling BreadthSnapshot history
    // ------------------------------------------------------------------

    explicit MarketBreadth(int history_len = 500) noexcept
        : history_len_(history_len) {}

    /// Update breadth state from per-instrument returns and volumes.
    /// returns[i]  = (close[i] / prev_close[i]) - 1.0
    /// volumes[i]  = volume for instrument i this bar
    /// timestamp   = bar timestamp in ns
    BreadthSnapshot update(std::span<const double> returns,
                            std::span<const double> volumes,
                            int64_t                 timestamp_ns) noexcept
    {
        BreadthSnapshot snap;
        snap.timestamp_ns = timestamp_ns;

        count_advances_declines(returns, snap.advances, snap.declines, snap.unchanged);
        snap.ad_net   = snap.advances - snap.declines;
        snap.ad_ratio = (snap.declines > 0)
                        ? static_cast<double>(snap.advances) / snap.declines
                        : static_cast<double>(snap.advances > 0 ? snap.advances : 1);

        // Volume sums for advances / declines
        double adv_vol = 0.0, dec_vol = 0.0;
        int n = static_cast<int>(std::min(returns.size(), volumes.size()));
        for (int i = 0; i < n; ++i) {
            if      (returns[i] > 0.0) adv_vol += volumes[i];
            else if (returns[i] < 0.0) dec_vol += volumes[i];
        }
        snap.arms_index = arms_index(snap.advances, snap.declines, adv_vol, dec_vol);

        // McClellan oscillator step
        snap.mcclellan_osc = mcclellan_oscillator_step(
            snap.ad_net, ema19_, ema39_, ema19_, ema39_);
        mcclellan_sum_ += snap.mcclellan_osc;
        snap.mcclellan_summation = mcclellan_sum_;

        // Percent above MA -- requires price history; only computed when
        // price history is provided via update_with_prices().
        snap.pct_above_ma = last_pct_above_ma_;

        history_.push_back(snap);
        if (static_cast<int>(history_.size()) > history_len_)
            history_.pop_front();

        ++bar_count_;
        return snap;
    }

    /// Extended update that also computes %above MA.
    /// current_prices[n_instr], price_history[n_instr * ma_period]
    BreadthSnapshot update_with_prices(std::span<const double> returns,
                                        std::span<const double> volumes,
                                        std::span<const double> current_prices,
                                        const double*           price_history,
                                        int                     ma_period,
                                        int64_t                 timestamp_ns) noexcept
    {
        last_pct_above_ma_ = percent_above_ma(current_prices, price_history, ma_period);
        return update(returns, volumes, timestamp_ns);
    }

    // -- Accessors --
    double ema19()              const noexcept { return ema19_; }
    double ema39()              const noexcept { return ema39_; }
    double mcclellan_sum()      const noexcept { return mcclellan_sum_; }
    double last_pct_above_ma()  const noexcept { return last_pct_above_ma_; }
    int    bar_count()          const noexcept { return bar_count_; }

    const std::deque<BreadthSnapshot>& history() const noexcept { return history_; }

    const BreadthSnapshot& latest() const noexcept {
        static const BreadthSnapshot empty{};
        return history_.empty() ? empty : history_.back();
    }

    // Rolling AD line: cumulative sum of (advances - declines)
    double ad_line() const noexcept {
        double sum = 0.0;
        for (auto& s : history_) sum += s.ad_net;
        return sum;
    }

    void reset() noexcept {
        history_.clear();
        ema19_ = ema39_ = mcclellan_sum_ = last_pct_above_ma_ = 0.0;
        bar_count_ = 0;
    }

private:
    int    history_len_      = 500;
    double ema19_            = 0.0;
    double ema39_            = 0.0;
    double mcclellan_sum_    = 0.0;
    double last_pct_above_ma_= 0.0;
    int    bar_count_        = 0;
    std::deque<BreadthSnapshot> history_;
};

// ------------------------------------------------------------------
// BreadthOscillator -- derived signals from BreadthHistory
// ------------------------------------------------------------------

/// Computes second-order signals from a populated MarketBreadth object.
class BreadthOscillator {
public:
    explicit BreadthOscillator(const MarketBreadth& mb) : mb_(mb) {}

    /// McClellan Summation Index trend: slope of summation over last n bars.
    double summation_slope(int n = 20) const noexcept {
        auto& hist = mb_.history();
        int sz  = static_cast<int>(hist.size());
        int len = std::min(n, sz);
        if (len < 2) return 0.0;

        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        int start = sz - len;
        for (int i = 0; i < len; ++i) {
            double x = static_cast<double>(i);
            double y = hist[static_cast<std::size_t>(start + i)].mcclellan_summation;
            sx  += x; sy  += y;
            sxx += x * x; sxy += x * y;
        }
        double d = len * sxx - sx * sx;
        if (std::abs(d) < 1e-12) return 0.0;
        return (len * sxy - sx * sy) / d;
    }

    /// Arms Index moving average (smoothed TRIN).
    double trin_ma(int n = 10) const noexcept {
        auto& hist = mb_.history();
        int sz  = static_cast<int>(hist.size());
        int len = std::min(n, sz);
        if (len == 0) return 1.0;
        double sum = 0.0;
        int start = sz - len;
        for (int i = start; i < sz; ++i)
            sum += hist[static_cast<std::size_t>(i)].arms_index;
        return sum / len;
    }

    /// Percent above MA trend: has it been rising or falling over last n bars?
    double pct_above_ma_slope(int n = 10) const noexcept {
        auto& hist = mb_.history();
        int sz  = static_cast<int>(hist.size());
        int len = std::min(n, sz);
        if (len < 2) return 0.0;

        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        int start = sz - len;
        for (int i = 0; i < len; ++i) {
            double x = static_cast<double>(i);
            double y = hist[static_cast<std::size_t>(start + i)].pct_above_ma;
            sx += x; sy += y; sxx += x*x; sxy += x*y;
        }
        double d = len * sxx - sx * sx;
        if (std::abs(d) < 1e-12) return 0.0;
        return (len * sxy - sx * sy) / d;
    }

private:
    const MarketBreadth& mb_;
};

} // namespace indicators
} // namespace srfm
