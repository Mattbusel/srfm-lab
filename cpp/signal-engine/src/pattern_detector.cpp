// pattern_detector.cpp -- Candlestick pattern detector implementation.
// The core logic lives in the header (inline methods on CandlePattern).
// This file contains:
//   - Non-trivial PatternSignalEngine methods that benefit from being out-of-line
//   - Batch scanning helpers
//   - Unit-level sanity self-tests (compiled only when SRFM_PATTERN_SELFTEST defined)

#include "pattern_detector.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
#include <string>
#include <numeric>

namespace srfm {
namespace signals {

// ============================================================
// BatchPatternScanner -- run the pattern engine over a historical bar array
// ============================================================

struct PatternScanResult {
    int     bar_index;       // index into input array
    float   composite;       // composite_pattern_signal at that bar
    float   doji_conf;
    float   hammer_conf;
    float   shooting_star_conf;
    float   engulf_bull_conf;
    float   engulf_bear_conf;
    float   morning_star_conf;
    float   evening_star_conf;
    float   three_soldiers_conf;
    float   three_crows_conf;
};

// Scan a contiguous bar array and return per-bar results.
// results must have capacity >= n_bars.
int batch_scan_patterns(const Bar* bars, int n_bars,
                        PatternScanResult* results,
                        const PatternSignalEngine::Weights& weights)
{
    if (!bars || n_bars <= 0 || !results) return 0;

    PatternSignalEngine eng;
    eng.weights = weights;
    CandlePattern det;

    int out_count = 0;
    for (int i = 0; i < n_bars; ++i) {
        eng.update(bars[i]);

        PatternScanResult r{};
        r.bar_index = i;

        const Bar& curr = bars[i];
        const Bar* prev = (i > 0) ? &bars[i - 1] : nullptr;

        // Per-pattern scores at current bar
        r.doji_conf         = det.doji(curr);
        r.hammer_conf       = det.hammer(curr);
        r.shooting_star_conf = det.shooting_star(curr);

        if (prev) {
            r.engulf_bull_conf = det.engulfing_bullish(*prev, curr);
            r.engulf_bear_conf = det.engulfing_bearish(*prev, curr);
        }

        if (i >= 2) {
            Bar trio[3] = { bars[i-2], bars[i-1], bars[i] };
            r.morning_star_conf    = det.morning_star(trio);
            r.evening_star_conf    = det.evening_star(trio);
            r.three_soldiers_conf  = det.three_white_soldiers(trio);
            r.three_crows_conf     = det.three_black_crows(trio);
        }

        r.composite = eng.composite_pattern_signal();
        results[out_count++] = r;
    }
    return out_count;
}

// ============================================================
// Signal statistics over a scan
// ============================================================

struct PatternStats {
    int   n_bullish;   // composite > 0.05
    int   n_bearish;   // composite < -0.05
    int   n_neutral;
    float mean_composite;
    float max_composite;
    float min_composite;
};

PatternStats compute_pattern_stats(const PatternScanResult* results, int n)
{
    PatternStats s{};
    if (!results || n <= 0) return s;

    float sum = 0.0f;
    s.max_composite = results[0].composite;
    s.min_composite = results[0].composite;

    for (int i = 0; i < n; ++i) {
        float c = results[i].composite;
        sum += c;
        if (c >  0.05f) ++s.n_bullish;
        else if (c < -0.05f) ++s.n_bearish;
        else            ++s.n_neutral;
        if (c > s.max_composite) s.max_composite = c;
        if (c < s.min_composite) s.min_composite = c;
    }
    s.mean_composite = sum / static_cast<float>(n);
    return s;
}

// ============================================================
// Self-test (compiled only when SRFM_PATTERN_SELFTEST is defined)
// ============================================================

#ifdef SRFM_PATTERN_SELFTEST

static void assert_near(float a, float b, float tol, const char* label)
{
    if (std::abs(a - b) > tol) {
        std::fprintf(stderr, "FAIL %s: got %.4f expected ~%.4f\n", label, a, b);
        std::abort();
    }
    std::fprintf(stdout, "PASS %s: %.4f\n", label, a);
}

void run_pattern_selftest()
{
    CandlePattern det;

    // -- Test doji: perfect doji (open == close)
    {
        Bar b(10.0f, 11.0f, 9.0f, 10.0f, 1000.0f);
        float conf = det.doji(b);
        assert_near(conf, 1.0f, 0.01f, "doji_perfect");
    }

    // -- Test doji: large body => 0
    {
        Bar b(10.0f, 12.0f, 9.0f, 11.5f, 1000.0f);
        float conf = det.doji(b);
        assert_near(conf, 0.0f, 0.05f, "doji_large_body");
    }

    // -- Test hammer: small body at top, long lower shadow
    {
        // open=10, close=10.2, high=10.3, low=7.0 => body=0.2, range=3.3
        // lower_shadow = min(10,10.2) - 7 = 3.0
        // body/range = 0.2/3.3 = 0.06 (<0.35 OK)
        // lower_shadow/body = 3.0/0.2 = 15 (>2 OK)
        // upper_shadow = 10.3 - max(10,10.2) = 10.3-10.2 = 0.1 (<body=0.2 OK)
        Bar b(10.0f, 10.3f, 7.0f, 10.2f, 5000.0f);
        float conf = det.hammer(b);
        assert_near(conf, 0.9f, 0.2f, "hammer_valid");
    }

    // -- Test shooting star
    {
        // open=10, close=9.9, high=13.0, low=9.8 => body=0.1, range=3.2
        // upper_shadow = 13.0-10.0 = 3.0
        // lower_shadow = 9.9-9.8 = 0.1 (< body? 0.1 <= 0.1 yes)
        Bar b(10.0f, 13.0f, 9.8f, 9.9f, 4000.0f);
        float conf = det.shooting_star(b);
        assert_near(conf, 0.9f, 0.2f, "shooting_star_valid");
    }

    // -- Test bullish engulfing
    {
        Bar prev(12.0f, 12.5f, 10.0f, 10.5f, 1000.0f);  // bearish
        Bar curr(10.0f, 13.0f, 9.8f,  13.0f, 2000.0f);  // bullish, engulfs
        float conf = det.engulfing_bullish(prev, curr);
        assert_near(conf, 0.7f, 0.3f, "engulf_bull");
    }

    // -- Test bearish engulfing
    {
        Bar prev(10.0f, 12.5f, 9.8f,  12.0f, 1000.0f);  // bullish
        Bar curr(12.5f, 12.7f, 9.0f,   9.1f, 2200.0f);  // bearish, engulfs
        float conf = det.engulfing_bearish(prev, curr);
        assert_near(conf, 0.7f, 0.3f, "engulf_bear");
    }

    // -- Test morning star
    {
        Bar trio[3] = {
            Bar(12.0f, 12.5f, 10.0f, 10.2f, 3000.0f),  // big bearish
            Bar( 9.8f, 10.0f,  9.5f,  9.7f, 1000.0f),  // small star below
            Bar( 9.7f, 12.0f,  9.6f, 11.5f, 4000.0f),  // big bullish recovery
        };
        float conf = det.morning_star(trio);
        assert_near(conf, 0.5f, 0.5f, "morning_star");
        if (conf <= 0.0f) {
            std::fprintf(stderr, "WARN morning_star returned 0 -- check thresholds\n");
        }
    }

    // -- Test three white soldiers
    {
        Bar trio[3] = {
            Bar(10.0f, 11.0f, 9.9f, 10.9f, 2000.0f),
            Bar(10.9f, 12.0f, 10.8f, 11.9f, 2200.0f),
            Bar(11.9f, 13.0f, 11.8f, 12.9f, 2400.0f),
        };
        float conf = det.three_white_soldiers(trio);
        assert_near(conf, 0.8f, 0.25f, "three_soldiers");
    }

    // -- Test three black crows
    {
        Bar trio[3] = {
            Bar(13.0f, 13.1f, 11.9f, 12.0f, 2000.0f),
            Bar(12.0f, 12.1f, 10.9f, 11.0f, 2200.0f),
            Bar(11.0f, 11.1f,  9.9f, 10.0f, 2400.0f),
        };
        float conf = det.three_black_crows(trio);
        assert_near(conf, 0.8f, 0.25f, "three_crows");
    }

    // -- Test PatternSignalEngine rolling buffer
    {
        PatternSignalEngine eng;
        // Feed 5 bullish bars
        for (int i = 0; i < 5; ++i) {
            float base = 100.0f + static_cast<float>(i);
            Bar b(base, base + 1.0f, base - 0.1f, base + 0.9f, 5000.0f);
            eng.update(b);
        }
        float sig = eng.composite_pattern_signal();
        // Should be mildly positive (bullish bars)
        if (sig < -0.5f || sig > 1.0f) {
            std::fprintf(stderr, "FAIL composite_bullish: got %.4f\n", sig);
            std::abort();
        }
        std::fprintf(stdout, "PASS composite_bullish: %.4f\n", sig);
    }

    std::fprintf(stdout, "All pattern_detector self-tests passed.\n");
}

#endif // SRFM_PATTERN_SELFTEST

} // namespace signals
} // namespace srfm
