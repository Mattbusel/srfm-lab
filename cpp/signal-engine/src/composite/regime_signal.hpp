#pragma once
// regime_signal.hpp -- Composite market regime signal engine.
//
// RegimeSignalEngine synthesises RSI, MACD, ADX, ATR, VolumeProfile,
// OrderFlowImbalance, and MarketBreadth into a single regime score with
// labelled regime state.

#include "srfm/types.hpp"
#include "srfm/simd_math.hpp"
#include "indicators/volume_profile.hpp"
#include "indicators/order_flow.hpp"
#include "indicators/market_breadth.hpp"

// Indicators live in src/indicators/ with local includes
#include "../indicators/rsi.hpp"
#include "../indicators/macd.hpp"
#include "../indicators/atr.hpp"
#include "../indicators/ema.hpp"

#include <array>
#include <deque>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace srfm {
namespace composite {

// ------------------------------------------------------------------
// RegimeSignal -- output struct
// ------------------------------------------------------------------

struct RegimeSignal {
    int64_t timestamp_ns    = 0;
    int32_t symbol_id       = 0;

    // Component scores in [-1, +1] (or [0,1] where noted)
    double trend_score      = 0.0;  // ADX + EMA200 filter
    double momentum_score   = 0.0;  // RSI + MACD + ROC
    double vol_score        = 0.0;  // ATR percentile (0=calm, 1=chaotic)
    double breadth_score    = 0.0;  // A/D ratio, %above MA
    double flow_score       = 0.0;  // order flow imbalance, delta divergence

    double composite        = 0.0;  // weighted average of components

    // Regime label index: 0=RANGING, 1=TRENDING_BULL, 2=TRENDING_BEAR, 3=VOLATILE
    int    regime_idx       = 0;

    // Raw indicator snapshots (for downstream logging / feature extraction)
    double rsi              = 50.0;
    double macd_hist        = 0.0;
    double adx              = 0.0;
    double atr              = 0.0;
    double ema200           = 0.0;
    double poc              = 0.0;  // Volume Profile POC
    double of_imbalance     = 0.0;  // order flow imbalance
    double mcclellan_osc    = 0.0;
    double pct_above_ma     = 0.5;

    bool   trend_up         = false;
    bool   of_divergence    = false;
    bool   of_absorption    = false;
};

// ------------------------------------------------------------------
// RegimeLabel helpers
// ------------------------------------------------------------------

inline const char* regime_label(int idx) noexcept {
    static const char* labels[] = {
        "RANGING", "TRENDING_BULL", "TRENDING_BEAR", "VOLATILE"
    };
    if (idx < 0 || idx > 3) return "UNKNOWN";
    return labels[static_cast<std::size_t>(idx)];
}

inline int classify_regime(double trend_score,
                             double vol_score) noexcept
{
    // VOLATILE: ATR in top quartile
    if (vol_score > 0.75) return 3; // VOLATILE

    // TRENDING: strong trend score
    if (trend_score > 0.4)  return 1; // TRENDING_BULL
    if (trend_score < -0.4) return 2; // TRENDING_BEAR

    return 0; // RANGING
}

// ------------------------------------------------------------------
// ADX (Average Directional Index) -- self-contained implementation
// ------------------------------------------------------------------

class ADX {
public:
    explicit ADX(int period = 14) noexcept
        : period_(period)
        , alpha_(1.0 / period)
        , smoothed_tr_(0.0), smoothed_dm_pos_(0.0), smoothed_dm_neg_(0.0)
        , adx_(0.0), plus_di_(0.0), minus_di_(0.0)
        , prev_high_(0.0), prev_low_(0.0), prev_close_(0.0)
        , dx_ema_(0.0), count_(0)
    {}

    double update(const OHLCVBar& bar) noexcept {
        return update(bar.high, bar.low, bar.close);
    }

    double update(double high, double low, double close) noexcept {
        if (count_ == 0) {
            prev_high_  = high;
            prev_low_   = low;
            prev_close_ = close;
            ++count_;
            return 0.0;
        }

        // True range
        double hl    = high - low;
        double hpc   = std::abs(high - prev_close_);
        double lpc   = std::abs(low  - prev_close_);
        double tr    = std::max({hl, hpc, lpc});

        // Directional movement
        double dm_pos = 0.0, dm_neg = 0.0;
        double up_move   = high    - prev_high_;
        double down_move = prev_low_ - low;

        if (up_move > down_move && up_move > 0.0) dm_pos = up_move;
        if (down_move > up_move && down_move > 0.0) dm_neg = down_move;

        // Wilder's smoothing
        smoothed_tr_     = smoothed_tr_     * (1.0 - alpha_) + tr;
        smoothed_dm_pos_ = smoothed_dm_pos_ * (1.0 - alpha_) + dm_pos;
        smoothed_dm_neg_ = smoothed_dm_neg_ * (1.0 - alpha_) + dm_neg;

        if (smoothed_tr_ > 1e-12) {
            plus_di_  = 100.0 * smoothed_dm_pos_ / smoothed_tr_;
            minus_di_ = 100.0 * smoothed_dm_neg_ / smoothed_tr_;
        }

        double di_sum = plus_di_ + minus_di_;
        double dx = (di_sum > 1e-12)
                    ? 100.0 * std::abs(plus_di_ - minus_di_) / di_sum
                    : 0.0;

        dx_ema_ = dx_ema_ * (1.0 - alpha_) + dx * alpha_;
        adx_    = dx_ema_;

        prev_high_  = high;
        prev_low_   = low;
        prev_close_ = close;
        ++count_;
        return adx_;
    }

    double adx()       const noexcept { return adx_; }
    double plus_di()   const noexcept { return plus_di_; }
    double minus_di()  const noexcept { return minus_di_; }
    bool   is_warm()   const noexcept { return count_ > period_ * 2; }

    void reset() noexcept {
        smoothed_tr_ = smoothed_dm_pos_ = smoothed_dm_neg_ = 0.0;
        adx_ = plus_di_ = minus_di_ = 0.0;
        prev_high_ = prev_low_ = prev_close_ = 0.0;
        dx_ema_ = 0.0;
        count_ = 0;
    }

private:
    int    period_;
    double alpha_;
    double smoothed_tr_, smoothed_dm_pos_, smoothed_dm_neg_;
    double adx_, plus_di_, minus_di_;
    double prev_high_, prev_low_, prev_close_;
    double dx_ema_;
    int    count_;
};

// ------------------------------------------------------------------
// ATR percentile tracker -- rolling window ATR readings
// ------------------------------------------------------------------

class ATRPercentile {
public:
    explicit ATRPercentile(int window = 100) noexcept
        : window_(window) {}

    void push(double atr_val) noexcept {
        history_.push_back(atr_val);
        if (static_cast<int>(history_.size()) > window_)
            history_.pop_front();
    }

    /// Returns percentile rank of current ATR in [0, 1].
    double percentile(double current_atr) const noexcept {
        if (history_.empty()) return 0.5;
        int below = 0;
        for (double v : history_) if (v < current_atr) ++below;
        return static_cast<double>(below) / history_.size();
    }

    int size() const noexcept { return static_cast<int>(history_.size()); }

private:
    int window_;
    std::deque<double> history_;
};

// ------------------------------------------------------------------
// RateOfChange indicator
// ------------------------------------------------------------------

class RateOfChange {
public:
    explicit RateOfChange(int period = 10) noexcept
        : period_(period), count_(0), head_(0) {
        buf_.fill(0.0);
    }

    double update(double price) noexcept {
        buf_[static_cast<std::size_t>(head_)] = price;
        head_ = (head_ + 1) % period_;
        ++count_;
        if (count_ < period_) return 0.0;
        double old = buf_[static_cast<std::size_t>(head_)];
        if (std::abs(old) < 1e-12) return 0.0;
        return (price - old) / old;
    }

    bool is_warm() const noexcept { return count_ >= period_; }

private:
    int                period_;
    int                count_;
    int                head_;
    std::array<double, 200> buf_;
};

// ------------------------------------------------------------------
// RegimeSignalEngine
// ------------------------------------------------------------------

/// Configuration for weight blending and threshold tuning.
struct RegimeEngineConfig {
    // Component weights (must sum to 1 for a normalised composite, but
    // the engine normalises internally so any positive set is fine).
    double w_trend    = 0.35;
    double w_momentum = 0.25;
    double w_vol      = 0.15;
    double w_breadth  = 0.15;
    double w_flow     = 0.10;

    // Trend thresholds
    double adx_trend_threshold  = 25.0;  // ADX > this => trending
    double adx_strong_threshold = 40.0;  // ADX > this => strong trend

    // RSI bounds for momentum scoring
    double rsi_overbought = 70.0;
    double rsi_oversold   = 30.0;

    // ATR percentile for "volatile regime"
    double vol_high_pct   = 0.75;
    double vol_low_pct    = 0.25;

    // EMA period for long-term trend filter
    int ema200_period     = 200;

    // ADX period
    int adx_period        = 14;

    // ROC period
    int roc_period        = 10;

    // ATR history for percentile
    int atr_history       = 100;
};

class RegimeSignalEngine {
public:
    explicit RegimeSignalEngine(const RegimeEngineConfig& cfg = {}) noexcept
        : cfg_(cfg)
        , rsi_(14)
        , macd_(12, 26, 9)
        , atr_(14)
        , adx_(cfg.adx_period)
        , ema200_(cfg.ema200_period)
        , roc_(cfg.roc_period)
        , atr_pct_(cfg.atr_history)
        , bar_count_(0)
    {}

    // ------------------------------------------------------------------
    // External indicator feeds (optional -- if not wired, breadth/flow
    // scores default to neutral).
    // ------------------------------------------------------------------

    void set_breadth(indicators::MarketBreadth* mb) noexcept { mb_ = mb; }
    void set_order_flow(indicators::OrderFlowImbalance* ofi) noexcept { ofi_ = ofi; }
    void set_volume_profile(indicators::RollingVolumeProfile* rvp) noexcept { rvp_ = rvp; }

    // ------------------------------------------------------------------
    // Primary compute interface
    // ------------------------------------------------------------------

    RegimeSignal compute(const OHLCVBar& bar) noexcept {
        RegimeSignal sig;
        sig.timestamp_ns = bar.timestamp_ns;
        sig.symbol_id    = bar.symbol_id;

        // -- Update base indicators --
        double rsi_val  = rsi_.update(bar.close);
        auto   macd_out = macd_.update(bar.close);
        double atr_val  = atr_.update(bar);
        double adx_val  = adx_.update(bar);
        double ema200   = ema200_.update(bar.close);
        double roc_val  = roc_.update(bar.close);

        atr_pct_.push(atr_val);
        ++bar_count_;

        // -- Store raw snapshots --
        sig.rsi       = rsi_val;
        sig.macd_hist = macd_out.histogram;
        sig.adx       = adx_val;
        sig.atr       = atr_val;
        sig.ema200    = ema200;

        // -- Volume profile --
        if (rvp_) {
            double typical = (bar.high + bar.low + bar.close) / 3.0;
            rvp_->update(typical, bar.volume);
            sig.poc = rvp_->get_poc();
        }

        // -- Order flow --
        if (ofi_) {
            ofi_->update(bar, atr_val);
            sig.of_imbalance  = ofi_->last_imbalance();
            sig.of_divergence = ofi_->last_divergence();
            sig.of_absorption = ofi_->last_absorption();
        }

        // -- Breadth --
        if (mb_) {
            auto& snap = mb_->latest();
            sig.mcclellan_osc = snap.mcclellan_osc;
            sig.pct_above_ma  = snap.pct_above_ma;
        }

        // -- Compute component scores --
        sig.trend_score    = compute_trend_score(bar.close, adx_val, ema200);
        sig.momentum_score = compute_momentum_score(rsi_val, macd_out.histogram,
                                                      macd_out.macd_line,
                                                      macd_out.signal_line,
                                                      roc_val);
        sig.vol_score      = compute_vol_score(atr_val);
        sig.breadth_score  = compute_breadth_score(sig.mcclellan_osc,
                                                    sig.pct_above_ma);
        sig.flow_score     = compute_flow_score(sig.of_imbalance,
                                                 sig.of_divergence,
                                                 sig.of_absorption);

        sig.trend_up = (bar.close > ema200);

        // -- Weighted composite --
        sig.composite = weighted_composite(sig);

        // -- Regime label --
        sig.regime_idx = classify_regime(sig.trend_score, sig.vol_score);

        return sig;
    }

    std::string get_regime_label() const noexcept {
        if (bar_count_ == 0) return "UNKNOWN";
        return regime_label(last_regime_idx_);
    }

    int    bar_count()    const noexcept { return bar_count_; }
    double last_rsi()     const noexcept { return rsi_.value(); }
    double last_atr()     const noexcept { return atr_.value(); }
    double last_adx()     const noexcept { return adx_.adx(); }

    void reset() noexcept {
        rsi_.reset();
        macd_.reset();
        atr_.reset();
        adx_.reset();
        ema200_.reset();
        atr_pct_ = ATRPercentile(cfg_.atr_history);
        bar_count_ = 0;
        last_regime_idx_ = 0;
    }

private:
    // ------------------------------------------------------------------
    // Score computation helpers
    // ------------------------------------------------------------------

    double compute_trend_score(double close,
                                 double adx,
                                 double ema200) const noexcept
    {
        // Returns value in [-1, +1].
        // Pure directional component: +1 = strong uptrend, -1 = strong downtrend.
        bool trending = adx > cfg_.adx_trend_threshold;
        bool strong   = adx > cfg_.adx_strong_threshold;
        bool price_above = close > ema200;

        if (!trending) return 0.0;

        double base = strong ? 1.0 : 0.6;
        return price_above ? base : -base;
    }

    double compute_momentum_score(double rsi,
                                    double macd_hist,
                                    double macd_line,
                                    double signal_line,
                                    double roc) const noexcept
    {
        // RSI component: [-1, +1] normalised from [0, 100]
        double rsi_score = (rsi - 50.0) / 50.0;  // -1..+1

        // MACD histogram normalised by magnitude (tanh squashing)
        double macd_score = std::tanh(macd_hist * 10.0);

        // Signal line cross: +1 if macd > signal, -1 if below
        double cross_score = (macd_line > signal_line) ? 1.0 : -1.0;
        if (std::abs(macd_line - signal_line) < 1e-6 * std::abs(macd_line) + 1e-12)
            cross_score = 0.0;

        // Rate of Change squashed to [-1, +1]
        double roc_score = std::tanh(roc * 20.0);

        return 0.35 * rsi_score + 0.35 * macd_score +
               0.15 * cross_score + 0.15 * roc_score;
    }

    double compute_vol_score(double atr) const noexcept {
        // Returns percentile rank in [0, 1] -- high = uncertain/volatile.
        return atr_pct_.percentile(atr);
    }

    double compute_breadth_score(double mcclellan_osc,
                                   double pct_above_ma) const noexcept
    {
        // Both components normalised to [-1, +1].
        // McClellan: divide by a typical range (~100 for large universes).
        double m_score = std::tanh(mcclellan_osc / 50.0);
        // pct_above_ma: 0..1 rescaled to -1..+1
        double p_score = (pct_above_ma - 0.5) * 2.0;
        return 0.5 * m_score + 0.5 * p_score;
    }

    double compute_flow_score(double imbalance,
                                bool   divergence,
                                bool   absorption) const noexcept
    {
        // imbalance in [-1, +1]
        double score = imbalance;
        // Divergence weakens the signal (price and flow disagree)
        if (divergence) score *= 0.5;
        // Absorption often precedes reversal -- flip sign slightly
        if (absorption) score *= -0.7;
        return std::clamp(score, -1.0, 1.0);
    }

    double weighted_composite(const RegimeSignal& sig) noexcept {
        double w_sum = cfg_.w_trend + cfg_.w_momentum +
                       cfg_.w_vol   + cfg_.w_breadth  + cfg_.w_flow;
        if (w_sum < 1e-12) return 0.0;

        // Vol score is inverted: high vol => uncertain => pull composite toward 0
        double vol_adj = (1.0 - sig.vol_score * 2.0);  // 1.0 at calm, -1.0 at chaotic

        double c = cfg_.w_trend    * sig.trend_score    +
                   cfg_.w_momentum * sig.momentum_score +
                   cfg_.w_vol      * vol_adj             +
                   cfg_.w_breadth  * sig.breadth_score   +
                   cfg_.w_flow     * sig.flow_score;

        c /= w_sum;
        last_regime_idx_ = classify_regime(sig.trend_score, sig.vol_score);
        return std::clamp(c, -1.0, 1.0);
    }

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    RegimeEngineConfig          cfg_;
    RSI                         rsi_;
    MACD                        macd_;
    ATR                         atr_;
    ADX                         adx_;
    EMA                         ema200_;
    RateOfChange                roc_;
    ATRPercentile               atr_pct_;
    int                         bar_count_       = 0;
    mutable int                 last_regime_idx_ = 0;

    // Optional external indicator feeds
    indicators::MarketBreadth*          mb_  = nullptr;
    indicators::OrderFlowImbalance*     ofi_ = nullptr;
    indicators::RollingVolumeProfile*   rvp_ = nullptr;
};

// ------------------------------------------------------------------
// RegimeHistory -- ring buffer of recent regime signals
// ------------------------------------------------------------------

/// Keeps the last N RegimeSignal structs.  Useful for multi-bar lookback
/// in downstream strategies.
template <int N = 200>
class RegimeHistory {
public:
    RegimeHistory() noexcept : head_(0), size_(0) {}

    void push(const RegimeSignal& sig) noexcept {
        buf_[static_cast<std::size_t>(head_)] = sig;
        head_ = (head_ + 1) % N;
        if (size_ < N) ++size_;
    }

    int size() const noexcept { return size_; }

    const RegimeSignal& latest() const noexcept {
        static const RegimeSignal empty{};
        if (size_ == 0) return empty;
        int idx = (head_ - 1 + N) % N;
        return buf_[static_cast<std::size_t>(idx)];
    }

    const RegimeSignal& operator[](int i) const noexcept {
        // i=0 is oldest of last `size_` elements
        int idx = (head_ - size_ + i + N * 2) % N;
        return buf_[static_cast<std::size_t>(idx)];
    }

    /// Count bars in current regime without interruption.
    int bars_in_current_regime() const noexcept {
        if (size_ == 0) return 0;
        int target = latest().regime_idx;
        int count  = 0;
        for (int i = size_ - 1; i >= 0; --i) {
            if ((*this)[i].regime_idx == target) ++count;
            else break;
        }
        return count;
    }

    /// Slope of composite score over last n bars.
    double composite_slope(int n) const noexcept {
        n = std::min(n, size_);
        if (n < 2) return 0.0;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n; ++i) {
            double x = static_cast<double>(i);
            double y = (*this)[size_ - n + i].composite;
            sx += x; sy += y; sxx += x*x; sxy += x*y;
        }
        double d = n * sxx - sx * sx;
        if (std::abs(d) < 1e-12) return 0.0;
        return (n * sxy - sx * sy) / d;
    }

private:
    std::array<RegimeSignal, N> buf_;
    int head_;
    int size_;
};

} // namespace composite
} // namespace srfm
