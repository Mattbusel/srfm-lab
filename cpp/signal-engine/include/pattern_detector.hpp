#pragma once
// pattern_detector.hpp -- Candlestick pattern detection with confidence scores.
// All pattern methods return float in [0.0, 1.0].
// PatternSignalEngine maintains a 3-bar rolling buffer and emits a composite
// tanh-normalized signal in [-1, 1].

#include <cmath>
#include <cstdint>
#include <array>
#include <algorithm>

namespace srfm {
namespace signals {

// ============================================================
// Bar -- lightweight float OHLCV (pattern engine uses float precision)
// ============================================================

struct Bar {
    float open;
    float high;
    float low;
    float close;
    float volume;

    Bar() noexcept : open(0), high(0), low(0), close(0), volume(0) {}
    Bar(float o, float h, float l, float c, float v) noexcept
        : open(o), high(h), low(l), close(c), volume(v) {}

    // Convenience helpers
    float body()        const noexcept { return std::abs(close - open); }
    float range()       const noexcept { return high - low; }
    float upper_shadow() const noexcept { return high - std::max(open, close); }
    float lower_shadow() const noexcept { return std::min(open, close) - low; }
    bool  is_bullish()   const noexcept { return close >= open; }
    bool  is_bearish()   const noexcept { return close <  open; }
    float mid_body()     const noexcept { return (open + close) * 0.5f; }
};

// ============================================================
// CandlePattern -- stateless pattern detector
// ============================================================

class CandlePattern {
public:
    // Tunable thresholds (public for easy adjustment in tests)
    float doji_ratio        = 0.10f;  // body/range <= this => doji candidate
    float hammer_body_ratio = 0.35f;  // body/range <= 35%
    float hammer_shadow_min = 2.0f;   // lower shadow >= 2x body
    float star_shadow_min   = 2.0f;   // upper shadow >= 2x body (shooting star)
    float engulf_min_ratio  = 0.50f;  // curr body must cover >= 50% of prev range

    CandlePattern() noexcept = default;

    // ----------------------------------------------------------
    // Single-bar patterns
    // ----------------------------------------------------------

    // doji -- open ~= close, small body relative to range.
    // Returns 0 if range is zero. Confidence grows as body/range shrinks.
    float doji(const Bar& bar) const noexcept {
        const float rng = bar.range();
        if (rng < 1e-8f) return 0.0f;
        const float ratio = bar.body() / rng;
        if (ratio > doji_ratio * 2.0f) return 0.0f;
        // Linear falloff: perfect doji (ratio=0) => 1.0, threshold*2 => 0.0
        const float conf = 1.0f - ratio / (doji_ratio * 2.0f);
        return std::clamp(conf, 0.0f, 1.0f);
    }

    // hammer -- small body near top of range, long lower shadow.
    // Typically bullish when appearing after a downtrend.
    float hammer(const Bar& bar) const noexcept {
        const float rng = bar.range();
        if (rng < 1e-8f) return 0.0f;
        const float body    = bar.body();
        const float lo_shad = bar.lower_shadow();
        const float hi_shad = bar.upper_shadow();

        // Body must be small relative to range
        if (body / rng > hammer_body_ratio) return 0.0f;
        // Lower shadow must be at least 2x body
        if (body < 1e-8f) {
            // Treat as near-doji -- partial credit
            if (lo_shad / rng < 0.50f) return 0.0f;
            return 0.5f;
        }
        if (lo_shad / body < hammer_shadow_min) return 0.0f;
        // Upper shadow must be small (< body)
        if (hi_shad > body) return 0.0f;

        // Confidence: driven by how long the lower shadow is relative to range
        const float shad_ratio = lo_shad / rng;
        const float conf = std::clamp(shad_ratio * 1.5f, 0.0f, 1.0f);
        return conf;
    }

    // shooting_star -- small body near bottom of range, long upper shadow.
    // Typically bearish when appearing after an uptrend.
    float shooting_star(const Bar& bar) const noexcept {
        const float rng = bar.range();
        if (rng < 1e-8f) return 0.0f;
        const float body    = bar.body();
        const float hi_shad = bar.upper_shadow();
        const float lo_shad = bar.lower_shadow();

        if (body / rng > hammer_body_ratio) return 0.0f;
        if (body < 1e-8f) {
            if (hi_shad / rng < 0.50f) return 0.0f;
            return 0.5f;
        }
        if (hi_shad / body < star_shadow_min) return 0.0f;
        if (lo_shad > body) return 0.0f;

        const float shad_ratio = hi_shad / rng;
        return std::clamp(shad_ratio * 1.5f, 0.0f, 1.0f);
    }

    // ----------------------------------------------------------
    // Two-bar patterns
    // ----------------------------------------------------------

    // engulfing_bullish -- bearish prev bar fully engulfed by bullish curr bar.
    float engulfing_bullish(const Bar& prev, const Bar& curr) const noexcept {
        // Prev must be bearish, curr must be bullish
        if (!prev.is_bearish() || !curr.is_bullish()) return 0.0f;

        const float prev_body = prev.body();
        const float curr_body = curr.body();
        if (prev_body < 1e-8f || curr_body < 1e-8f) return 0.0f;

        // Curr open <= prev close AND curr close >= prev open (full engulf)
        const bool engulfs = (curr.open <= prev.close) && (curr.close >= prev.open);
        if (!engulfs) return 0.0f;

        // Confidence: ratio of curr body to prev body (larger => stronger)
        const float ratio = curr_body / (prev_body + 1e-8f);
        // Also scale by how much volume confirms (if available)
        float vol_factor = 1.0f;
        if (prev.volume > 1e-8f && curr.volume > 1e-8f) {
            vol_factor = std::min(1.5f, curr.volume / (prev.volume + 1e-8f));
            vol_factor = (vol_factor - 1.0f) * 0.5f + 1.0f; // soften
        }
        const float conf = std::clamp((ratio - 1.0f) * 0.5f + 0.6f, 0.0f, 1.0f) * vol_factor;
        return std::clamp(conf, 0.0f, 1.0f);
    }

    // engulfing_bearish -- bullish prev bar fully engulfed by bearish curr bar.
    float engulfing_bearish(const Bar& prev, const Bar& curr) const noexcept {
        if (!prev.is_bullish() || !curr.is_bearish()) return 0.0f;

        const float prev_body = prev.body();
        const float curr_body = curr.body();
        if (prev_body < 1e-8f || curr_body < 1e-8f) return 0.0f;

        const bool engulfs = (curr.open >= prev.close) && (curr.close <= prev.open);
        if (!engulfs) return 0.0f;

        float vol_factor = 1.0f;
        if (prev.volume > 1e-8f && curr.volume > 1e-8f) {
            vol_factor = std::min(1.5f, curr.volume / (prev.volume + 1e-8f));
            vol_factor = (vol_factor - 1.0f) * 0.5f + 1.0f;
        }
        const float ratio = curr_body / (prev_body + 1e-8f);
        const float conf = std::clamp((ratio - 1.0f) * 0.5f + 0.6f, 0.0f, 1.0f) * vol_factor;
        return std::clamp(conf, 0.0f, 1.0f);
    }

    // ----------------------------------------------------------
    // Three-bar patterns
    // ----------------------------------------------------------

    // morning_star -- [0]=bearish, [1]=small body (star), [2]=bullish closing into [0].
    // Bullish reversal signal.
    float morning_star(const Bar bars[3]) const noexcept {
        const Bar& first  = bars[0];
        const Bar& star   = bars[1];
        const Bar& third  = bars[2];

        // First must be bearish, third must be bullish
        if (!first.is_bearish() || !third.is_bullish()) return 0.0f;

        // Star bar must have a small body relative to its range
        const float star_rng = star.range();
        if (star_rng < 1e-8f) return 0.0f;
        const float star_body_ratio = star.body() / star_rng;
        if (star_body_ratio > 0.40f) return 0.0f;

        // Gap-down from bar[0] to star: star high < first close (ideal)
        // Relax: star mid_body < first close
        if (star.mid_body() >= first.close) return 0.0f;

        // Third bar must close above midpoint of first bar's body
        const float first_midpoint = first.mid_body();
        if (third.close <= first_midpoint) return 0.0f;

        // Confidence: based on how deep into [0] bar [2] recovers
        const float recovery = (third.close - first_midpoint) / (first.body() + 1e-8f);
        const float star_quality = 1.0f - star_body_ratio / 0.40f;
        const float conf = std::clamp(recovery * 0.5f + star_quality * 0.5f, 0.0f, 1.0f);
        return conf;
    }

    // evening_star -- [0]=bullish, [1]=small body (star), [2]=bearish closing into [0].
    // Bearish reversal signal.
    float evening_star(const Bar bars[3]) const noexcept {
        const Bar& first  = bars[0];
        const Bar& star   = bars[1];
        const Bar& third  = bars[2];

        if (!first.is_bullish() || !third.is_bearish()) return 0.0f;

        const float star_rng = star.range();
        if (star_rng < 1e-8f) return 0.0f;
        const float star_body_ratio = star.body() / star_rng;
        if (star_body_ratio > 0.40f) return 0.0f;

        // Star mid body above first close (gap up into star)
        if (star.mid_body() <= first.close) return 0.0f;

        // Third bar closes below midpoint of first bar
        const float first_midpoint = first.mid_body();
        if (third.close >= first_midpoint) return 0.0f;

        const float penetration = (first_midpoint - third.close) / (first.body() + 1e-8f);
        const float star_quality = 1.0f - star_body_ratio / 0.40f;
        const float conf = std::clamp(penetration * 0.5f + star_quality * 0.5f, 0.0f, 1.0f);
        return conf;
    }

    // three_white_soldiers -- three consecutive bullish bars, each closing near high.
    // Strong bullish continuation signal.
    float three_white_soldiers(const Bar bars[3]) const noexcept {
        for (int i = 0; i < 3; ++i) {
            if (!bars[i].is_bullish()) return 0.0f;
        }

        // Each bar must open within the prior bar's body
        for (int i = 1; i < 3; ++i) {
            if (bars[i].open < bars[i-1].open)  return 0.0f;
            if (bars[i].open > bars[i-1].close) return 0.0f;
        }

        // Each close must be higher than the previous close
        for (int i = 1; i < 3; ++i) {
            if (bars[i].close <= bars[i-1].close) return 0.0f;
        }

        // Upper shadows should be small (close near high) -- each < 20% of range
        float shadow_penalty = 0.0f;
        for (int i = 0; i < 3; ++i) {
            const float rng = bars[i].range();
            if (rng < 1e-8f) continue;
            const float us_ratio = bars[i].upper_shadow() / rng;
            shadow_penalty += us_ratio;
        }
        shadow_penalty /= 3.0f;

        // Confidence: penalize large upper shadows
        const float conf = std::clamp(1.0f - shadow_penalty * 2.0f, 0.0f, 1.0f);
        return conf;
    }

    // three_black_crows -- three consecutive bearish bars, each closing near low.
    // Strong bearish continuation signal.
    float three_black_crows(const Bar bars[3]) const noexcept {
        for (int i = 0; i < 3; ++i) {
            if (!bars[i].is_bearish()) return 0.0f;
        }

        // Each bar must open within the prior bar's body (i.e., inside prior decline)
        for (int i = 1; i < 3; ++i) {
            if (bars[i].open > bars[i-1].open)  return 0.0f;
            if (bars[i].open < bars[i-1].close) return 0.0f;
        }

        // Each close must be lower than the previous close
        for (int i = 1; i < 3; ++i) {
            if (bars[i].close >= bars[i-1].close) return 0.0f;
        }

        // Lower shadows should be small (close near low)
        float shadow_penalty = 0.0f;
        for (int i = 0; i < 3; ++i) {
            const float rng = bars[i].range();
            if (rng < 1e-8f) continue;
            const float ls_ratio = bars[i].lower_shadow() / rng;
            shadow_penalty += ls_ratio;
        }
        shadow_penalty /= 3.0f;

        const float conf = std::clamp(1.0f - shadow_penalty * 2.0f, 0.0f, 1.0f);
        return conf;
    }
};

// ============================================================
// PatternSignalEngine -- rolling 3-bar engine with composite signal
// ============================================================

class PatternSignalEngine {
public:
    // Pattern weights -- positive = bullish contribution, negative = bearish.
    // Applied to [-1,1]-mapped pattern scores before tanh normalization.
    struct Weights {
        float doji             = 0.10f;  // neutral -- reduce overall magnitude
        float hammer           = 0.80f;
        float shooting_star    = 0.80f;  // will be negated in composite
        float engulf_bull      = 1.20f;
        float engulf_bear      = 1.20f;  // negated
        float morning_star     = 1.50f;
        float evening_star     = 1.50f;  // negated
        float three_soldiers   = 1.80f;
        float three_crows      = 1.80f;  // negated
    };

    Weights weights;

    PatternSignalEngine() noexcept : detector_(), count_(0), last_signal_(0.0f) {
        bars_.fill(Bar{});
    }

    // update -- push a new bar into the rolling 3-bar buffer.
    void update(const Bar& bar) noexcept {
        bars_[0] = bars_[1];
        bars_[1] = bars_[2];
        bars_[2] = bar;
        ++count_;
    }

    // composite_pattern_signal -- weighted sum of all active patterns,
    // normalized through tanh to yield a value in (-1, 1).
    // Positive => net bullish, negative => net bearish.
    float composite_pattern_signal() noexcept {
        if (count_ < 1) { last_signal_ = 0.0f; return 0.0f; }

        float score = 0.0f;

        // Single-bar patterns (use bar[2] = most recent)
        const Bar& curr = bars_[2];
        score += weights.doji   * (detector_.doji(curr) * -0.5f);  // doji reduces confidence
        score += weights.hammer * detector_.hammer(curr);
        score -= weights.shooting_star * detector_.shooting_star(curr);

        // Two-bar patterns
        if (count_ >= 2) {
            const Bar& prev = bars_[1];
            score += weights.engulf_bull * detector_.engulfing_bullish(prev, curr);
            score -= weights.engulf_bear * detector_.engulfing_bearish(prev, curr);
        }

        // Three-bar patterns
        if (count_ >= 3) {
            score += weights.morning_star   * detector_.morning_star(bars_.data());
            score -= weights.evening_star   * detector_.evening_star(bars_.data());
            score += weights.three_soldiers * detector_.three_white_soldiers(bars_.data());
            score -= weights.three_crows    * detector_.three_black_crows(bars_.data());
        }

        // tanh normalization -- maps unbounded score to (-1, 1)
        last_signal_ = std::tanh(score);
        return last_signal_;
    }

    // Accessors
    float last_signal()         const noexcept { return last_signal_; }
    int   bars_available()      const noexcept { return static_cast<int>(count_ > 3 ? 3 : count_); }
    const Bar& bar(int i)       const noexcept { return bars_[i]; }  // 0=oldest, 2=newest
    const CandlePattern& detector() const noexcept { return detector_; }
    CandlePattern& detector()       noexcept { return detector_; }

    // reset -- clear buffer and counter
    void reset() noexcept {
        bars_.fill(Bar{});
        count_       = 0;
        last_signal_ = 0.0f;
    }

private:
    CandlePattern          detector_;
    std::array<Bar, 3>     bars_;
    uint64_t               count_;
    float                  last_signal_;
};

} // namespace signals
} // namespace srfm
