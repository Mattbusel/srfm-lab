#pragma once
// order_flow.hpp -- Order flow imbalance indicators.
//
// Provides:
//   DeltaBar          -- per-bar buy/sell delta via tick rule on OHLCV
//   CumulativeDelta   -- running session delta with session-open reset
//   OrderFlowImbalance -- bid/ask imbalance, delta divergence, absorption
//   FootprintBar      -- per-price-level delta breakdown
//
// AVX2 accelerated where batch operations are beneficial.

#include "srfm/types.hpp"
#include "srfm/simd_math.hpp"
#include <array>
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <span>

namespace srfm {
namespace indicators {

// ------------------------------------------------------------------
// DeltaBar
// ------------------------------------------------------------------

/// Estimates buy_volume and sell_volume from an OHLCV bar using the
/// simplified tick rule:
///   - If close > open   => all volume is buy.
///   - If close < open   => all volume is sell.
///   - If close == open  => split 50/50.
/// A more accurate blended model is available via compute_blended().
struct DeltaBar {
    double buy_volume  = 0.0;
    double sell_volume = 0.0;
    double delta       = 0.0;   // buy_volume - sell_volume
    double open        = 0.0;
    double close_price = 0.0;
    double volume      = 0.0;
    int64_t timestamp_ns = 0;

    /// Compute delta from an OHLCVBar using the simple tick rule.
    static DeltaBar from_bar(const OHLCVBar& bar) noexcept {
        DeltaBar d;
        d.open         = bar.open;
        d.close_price  = bar.close;
        d.volume       = bar.volume;
        d.timestamp_ns = bar.timestamp_ns;

        if (bar.close > bar.open) {
            d.buy_volume  = bar.volume;
            d.sell_volume = 0.0;
        } else if (bar.close < bar.open) {
            d.buy_volume  = 0.0;
            d.sell_volume = bar.volume;
        } else {
            d.buy_volume  = bar.volume * 0.5;
            d.sell_volume = bar.volume * 0.5;
        }
        d.delta = d.buy_volume - d.sell_volume;
        return d;
    }

    /// Blended model: distribute volume by close position within H-L range.
    /// buy_frac = (close - low) / (high - low)  clamped [0,1].
    static DeltaBar from_bar_blended(const OHLCVBar& bar) noexcept {
        DeltaBar d;
        d.open         = bar.open;
        d.close_price  = bar.close;
        d.volume       = bar.volume;
        d.timestamp_ns = bar.timestamp_ns;

        double range = bar.high - bar.low;
        double frac  = (range > 1e-12)
                       ? std::clamp((bar.close - bar.low) / range, 0.0, 1.0)
                       : 0.5;
        d.buy_volume  = bar.volume * frac;
        d.sell_volume = bar.volume * (1.0 - frac);
        d.delta       = d.buy_volume - d.sell_volume;
        return d;
    }
};

// ------------------------------------------------------------------
// CumulativeDelta
// ------------------------------------------------------------------

/// Running sum of DeltaBars.  Resets automatically at each session open
/// (i.e., when the incoming bar's timestamp crosses a UTC day boundary).
class CumulativeDelta {
public:
    CumulativeDelta() = default;

    /// Feed the next bar.  Returns the current cumulative delta.
    double update(const OHLCVBar& bar) noexcept {
        DeltaBar db = DeltaBar::from_bar_blended(bar);

        // Session reset: new UTC day
        int64_t day = bar.timestamp_ns / constants::NS_PER_DAY;
        if (day != current_day_ && current_day_ >= 0) {
            cumulative_delta_     = 0.0;
            cumulative_buy_vol_   = 0.0;
            cumulative_sell_vol_  = 0.0;
            bars_in_session_      = 0;
        }
        current_day_ = day;

        cumulative_delta_    += db.delta;
        cumulative_buy_vol_  += db.buy_volume;
        cumulative_sell_vol_ += db.sell_volume;
        ++bars_in_session_;

        last_delta_ = db.delta;
        history_.push_back(cumulative_delta_);
        if (static_cast<int>(history_.size()) > max_history_)
            history_.pop_front();

        return cumulative_delta_;
    }

    // -- Accessors --
    double cumulative()         const noexcept { return cumulative_delta_; }
    double last_bar_delta()     const noexcept { return last_delta_; }
    double cum_buy_volume()     const noexcept { return cumulative_buy_vol_; }
    double cum_sell_volume()    const noexcept { return cumulative_sell_vol_; }
    int    bars_in_session()    const noexcept { return bars_in_session_; }

    /// Simple linear slope of cumulative delta over the last `n` readings.
    double delta_slope(int n = 10) const noexcept {
        if (static_cast<int>(history_.size()) < 2) return 0.0;
        int len = std::min(n, static_cast<int>(history_.size()));
        // Least-squares slope: sum((xi - xbar)(yi - ybar)) / sum((xi-xbar)^2)
        double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
        int start = static_cast<int>(history_.size()) - len;
        for (int i = 0; i < len; ++i) {
            double xi = static_cast<double>(i);
            double yi = history_[static_cast<std::size_t>(start + i)];
            sum_x  += xi;
            sum_y  += yi;
            sum_xx += xi * xi;
            sum_xy += xi * yi;
        }
        double denom = len * sum_xx - sum_x * sum_x;
        if (std::abs(denom) < 1e-12) return 0.0;
        return (len * sum_xy - sum_x * sum_y) / denom;
    }

    void reset() noexcept {
        cumulative_delta_    = 0.0;
        cumulative_buy_vol_  = 0.0;
        cumulative_sell_vol_ = 0.0;
        bars_in_session_     = 0;
        last_delta_          = 0.0;
        current_day_         = -1;
        history_.clear();
    }

private:
    double cumulative_delta_    = 0.0;
    double cumulative_buy_vol_  = 0.0;
    double cumulative_sell_vol_ = 0.0;
    double last_delta_          = 0.0;
    int    bars_in_session_     = 0;
    int64_t current_day_        = -1;
    int     max_history_        = 500;
    std::deque<double> history_;
};

// ------------------------------------------------------------------
// OrderFlowImbalance
// ------------------------------------------------------------------

/// Stateless order flow metrics derived from bid/ask volume streams
/// and bar-level delta information.
class OrderFlowImbalance {
public:
    /// Bid-ask imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
    /// Result in [-1, +1].  Positive = more bids = buying pressure.
    static double bid_ask_imbalance(double bid_vol, double ask_vol) noexcept {
        double denom = bid_vol + ask_vol;
        if (denom < 1e-12) return 0.0;
        return (bid_vol - ask_vol) / denom;
    }

    /// Batch bid-ask imbalance for N instruments simultaneously.
    /// out[i] = (bid[i] - ask[i]) / (bid[i] + ask[i]).
    /// AVX2 accelerated.
    static void bid_ask_imbalance_batch(const double* __restrict__ bid,
                                         const double* __restrict__ ask,
                                         double*       __restrict__ out,
                                         std::size_t n) noexcept
    {
#if defined(SRFM_AVX2)
        const __m256d eps = _mm256_set1_pd(1e-12);
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d vb  = _mm256_loadu_pd(bid + i);
            __m256d va  = _mm256_loadu_pd(ask + i);
            __m256d sum = _mm256_add_pd(vb, va);
            __m256d diff= _mm256_sub_pd(vb, va);
            // Guard against zero denominator
            __m256d safe= _mm256_max_pd(sum, eps);
            _mm256_storeu_pd(out + i, _mm256_div_pd(diff, safe));
        }
        for (; i < n; ++i) out[i] = bid_ask_imbalance(bid[i], ask[i]);
#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] = bid_ask_imbalance(bid[i], ask[i]);
#endif
    }

    /// Delta divergence: returns true when price and delta are moving in
    /// opposite directions (potential reversal signal).
    /// price_trend  > 0 => price rising; < 0 => price falling.
    /// delta_trend  > 0 => delta rising; < 0 => delta falling.
    static bool delta_divergence(double price_trend,
                                  double delta_trend) noexcept
    {
        // Meaningful divergence requires both trends to be non-trivial.
        constexpr double threshold = 0.0;
        if (price_trend > threshold && delta_trend < -threshold) return true;
        if (price_trend < -threshold && delta_trend > threshold) return true;
        return false;
    }

    /// Absorption detection: large volume with small price move indicates
    /// a large participant absorbing order flow at that level.
    /// Returns true when:
    ///   abs(price_change) / ATR < price_move_threshold   (small move)
    ///   AND volume > vol_mult * avg_volume                (big volume)
    static bool absorption_detection(double price_change,
                                      double atr,
                                      double volume,
                                      double avg_volume,
                                      double price_move_threshold = 0.20,
                                      double vol_mult = 2.0) noexcept
    {
        if (atr < 1e-12 || avg_volume < 1e-12) return false;
        double relative_move = std::abs(price_change) / atr;
        double relative_vol  = volume / avg_volume;
        return (relative_move < price_move_threshold) &&
               (relative_vol  > vol_mult);
    }

    // -- Stateful version: maintains rolling avg_volume internally --
    void update(const OHLCVBar& bar, double atr) noexcept {
        double price_change = bar.close - bar.open;

        // Update rolling volume average (EMA, alpha = 2/(window+1), window=20)
        constexpr double alpha = 2.0 / 21.0;
        if (avg_volume_ < 1e-12) {
            avg_volume_ = bar.volume;
        } else {
            avg_volume_ = alpha * bar.volume + (1.0 - alpha) * avg_volume_;
        }

        // Compute current bar metrics
        double total = bar.volume;
        double frac = (atr > 1e-12)
            ? std::clamp((bar.close - bar.low) / (bar.high - bar.low + 1e-12), 0.0, 1.0)
            : 0.5;
        double buy_v  = total * frac;
        double sell_v = total * (1.0 - frac);

        last_imbalance_   = bid_ask_imbalance(buy_v, sell_v);
        last_delta_       = buy_v - sell_v;
        last_absorption_  = absorption_detection(price_change, atr, total, avg_volume_);

        // Track price / delta trends (EMA of signed values)
        constexpr double trend_alpha = 2.0 / 11.0;
        price_trend_ema_ = trend_alpha * price_change + (1.0 - trend_alpha) * price_trend_ema_;
        delta_trend_ema_ = trend_alpha * last_delta_  + (1.0 - trend_alpha) * delta_trend_ema_;
        last_divergence_ = delta_divergence(price_trend_ema_, delta_trend_ema_);
        ++bars_seen_;
    }

    double last_imbalance()   const noexcept { return last_imbalance_; }
    double last_delta()       const noexcept { return last_delta_; }
    bool   last_absorption()  const noexcept { return last_absorption_; }
    bool   last_divergence()  const noexcept { return last_divergence_; }
    double price_trend_ema()  const noexcept { return price_trend_ema_; }
    double delta_trend_ema()  const noexcept { return delta_trend_ema_; }
    int    bars_seen()        const noexcept { return bars_seen_; }

    void reset() noexcept {
        avg_volume_       = 0.0;
        last_imbalance_   = 0.0;
        last_delta_       = 0.0;
        price_trend_ema_  = 0.0;
        delta_trend_ema_  = 0.0;
        last_absorption_  = false;
        last_divergence_  = false;
        bars_seen_        = 0;
    }

private:
    double avg_volume_       = 0.0;
    double last_imbalance_   = 0.0;
    double last_delta_       = 0.0;
    double price_trend_ema_  = 0.0;
    double delta_trend_ema_  = 0.0;
    bool   last_absorption_  = false;
    bool   last_divergence_  = false;
    int    bars_seen_        = 0;
};

// ------------------------------------------------------------------
// FootprintBar
// ------------------------------------------------------------------

/// Footprint chart bar: tracks buy/sell volume at each price level
/// within a single bar's H-L range.
/// Price levels are quantised to `tick_size` increments.
class FootprintBar {
public:
    struct Level {
        double price      = 0.0;
        double buy_vol    = 0.0;
        double sell_vol   = 0.0;
        double delta()     const noexcept { return buy_vol - sell_vol; }
        double total_vol() const noexcept { return buy_vol + sell_vol; }
        double imbalance() const noexcept {
            double d = buy_vol + sell_vol;
            return (d > 1e-12) ? (buy_vol - sell_vol) / d : 0.0;
        }
    };

    explicit FootprintBar(double tick_size = 0.25) noexcept
        : tick_size_(tick_size > 0 ? tick_size : 0.25) {}

    /// Process a single tick (price, qty, side: +1=buy, -1=sell).
    void add_tick(double price, double qty, int side) noexcept {
        double qp = std::round(price / tick_size_) * tick_size_;
        auto it = std::find_if(levels_.begin(), levels_.end(),
            [&](const Level& l) { return std::abs(l.price - qp) < 1e-9; });
        if (it == levels_.end()) {
            levels_.push_back({qp, 0.0, 0.0});
            it = levels_.end() - 1;
        }
        if (side > 0) it->buy_vol  += qty;
        else          it->sell_vol += qty;
    }

    /// Infer footprint from a single OHLCVBar using blended model.
    /// Distributes volume across N_LEVELS price levels in [low, high].
    void from_bar(const OHLCVBar& bar, int n_levels = 10) noexcept {
        levels_.clear();
        if (n_levels < 1) n_levels = 1;
        double range = bar.high - bar.low;
        if (range < 1e-12) {
            // No range -- put everything at close
            double vol_frac = (bar.close > bar.open) ? 1.0 : -1.0;
            Level l;
            l.price    = bar.close;
            l.buy_vol  = (vol_frac > 0) ? bar.volume : 0.0;
            l.sell_vol = (vol_frac < 0) ? bar.volume : 0.0;
            levels_.push_back(l);
            return;
        }

        double step = range / n_levels;
        double buy_frac = std::clamp((bar.close - bar.low) / range, 0.0, 1.0);

        for (int i = 0; i < n_levels; ++i) {
            Level l;
            l.price = bar.low + (i + 0.5) * step;
            // Simple triangular distribution centred on close
            double centre_frac = (l.price - bar.low) / range;
            double weight = 1.0 - std::abs(centre_frac - buy_frac);
            if (weight < 0.0) weight = 0.0;
            double vol_share = bar.volume * weight / n_levels;
            l.buy_vol  = vol_share * buy_frac;
            l.sell_vol = vol_share * (1.0 - buy_frac);
            levels_.push_back(l);
        }

        // Sort ascending by price
        std::sort(levels_.begin(), levels_.end(),
                  [](const Level& a, const Level& b){ return a.price < b.price; });
    }

    // -- Queries --
    double total_delta() const noexcept {
        double d = 0.0;
        for (auto& l : levels_) d += l.delta();
        return d;
    }

    /// Returns the price level with the greatest absolute delta.
    double max_delta_price() const noexcept {
        if (levels_.empty()) return 0.0;
        return std::max_element(levels_.begin(), levels_.end(),
            [](const Level& a, const Level& b){
                return std::abs(a.delta()) < std::abs(b.delta());
            })->price;
    }

    /// Stacked imbalance: count consecutive levels where buy >> sell.
    int stacked_imbalance_count(double threshold = 3.0) const noexcept {
        int cnt = 0, max_run = 0;
        for (auto& l : levels_) {
            if (l.sell_vol > 1e-9 && l.buy_vol / l.sell_vol >= threshold) {
                ++cnt;
                if (cnt > max_run) max_run = cnt;
            } else {
                cnt = 0;
            }
        }
        return max_run;
    }

    const std::vector<Level>& levels() const noexcept { return levels_; }
    std::size_t level_count()           const noexcept { return levels_.size(); }
    double tick_size()                  const noexcept { return tick_size_; }

    void clear() noexcept { levels_.clear(); }

private:
    double tick_size_;
    std::vector<Level> levels_;
};

// ------------------------------------------------------------------
// DeltaHistory -- ring-buffer of recent bar deltas for slope calc
// ------------------------------------------------------------------

/// Fixed-size ring buffer of recent DeltaBar values.
template <int N = 100>
class DeltaHistory {
public:
    DeltaHistory() noexcept : head_(0), size_(0) {}

    void push(const DeltaBar& db) noexcept {
        buf_[head_] = db;
        head_ = (head_ + 1) % N;
        if (size_ < N) ++size_;
    }

    int size() const noexcept { return size_; }

    /// Linear regression slope of cumulative delta over last `n` bars.
    double cumulative_delta_slope(int n) const noexcept {
        n = std::min(n, size_);
        if (n < 2) return 0.0;

        // Collect last n deltas in chronological order
        std::array<double, N> vals;
        double cum = 0.0;
        for (int i = 0; i < n; ++i) {
            int idx = (head_ - n + i + N) % N;
            cum += buf_[idx].delta;
            vals[static_cast<std::size_t>(i)] = cum;
        }

        // Least-squares slope
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n; ++i) {
            double x = static_cast<double>(i);
            sx  += x;
            sy  += vals[static_cast<std::size_t>(i)];
            sxx += x * x;
            sxy += x * vals[static_cast<std::size_t>(i)];
        }
        double denom = n * sxx - sx * sx;
        if (std::abs(denom) < 1e-12) return 0.0;
        return (n * sxy - sx * sy) / denom;
    }

    const DeltaBar& operator[](int i) const noexcept {
        int idx = (head_ - size_ + i + N * 2) % N;
        return buf_[static_cast<std::size_t>(idx)];
    }

private:
    std::array<DeltaBar, N> buf_;
    int head_;
    int size_;
};

} // namespace indicators
} // namespace srfm
