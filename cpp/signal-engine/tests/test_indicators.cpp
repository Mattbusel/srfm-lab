/// test_indicators.cpp
/// Tests for EMA, ATR, RSI, Bollinger, MACD, VWAP.

#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "srfm/types.hpp"
#include "../src/indicators/ema.hpp"
#include "../src/indicators/atr.hpp"
#include "../src/indicators/rsi.hpp"
#include "../src/indicators/bollinger.hpp"
#include "../src/indicators/macd.hpp"
#include "../src/indicators/vwap.hpp"
#include "../src/indicators/realized_vol.hpp"

using namespace srfm;

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (!(expr)) { std::fprintf(stderr,"FAIL  %s:%d  %s\n",__FILE__,__LINE__,#expr); ++g_fail; } \
    else { ++g_pass; } \
} while(0)

#define CHECK_CLOSE(a, b, tol) do { \
    double _a=(a),_b=(b),_t=(tol); \
    if (std::abs(_a-_b)>_t) { \
        std::fprintf(stderr,"FAIL  %s:%d  %s  got=%.8g  exp=%.8g  diff=%.3e\n", \
                     __FILE__,__LINE__,#a,_a,_b,std::abs(_a-_b)); ++g_fail; \
    } else { ++g_pass; } \
} while(0)

static void section(const char* n) { std::printf("--- %s ---\n", n); }

static OHLCVBar make_bar(double o, double h, double l, double c, double v,
                          int idx) {
    return OHLCVBar(o, h, l, c, v,
                    1700000000LL * constants::NS_PER_SEC +
                    static_cast<int64_t>(idx) * constants::NS_PER_MIN);
}

// ============================================================
// EMA tests
// ============================================================

static void test_ema_basics() {
    section("EMA basics");

    EMA ema(3);
    CHECK(ema.alpha() == 0.5);  // 2/(3+1) = 0.5
    CHECK(!ema.is_warm());
    CHECK(ema.bar_count() == 0);

    // First bar seeds value
    double v1 = ema.update(100.0);
    CHECK_CLOSE(v1, 100.0, 1e-9);

    // Second bar: 0.5 * 200 + 0.5 * 100 = 150
    double v2 = ema.update(200.0);
    CHECK_CLOSE(v2, 150.0, 1e-9);

    // Third bar: 0.5 * 100 + 0.5 * 150 = 125
    double v3 = ema.update(100.0);
    CHECK_CLOSE(v3, 125.0, 1e-9);

    CHECK(ema.is_warm());  // period=3, 3 bars seen
}

static void test_ema_convergence() {
    section("EMA convergence");

    // EMA should converge to constant input
    EMA ema(10);
    for (int i = 0; i < 200; ++i) ema.update(100.0);
    CHECK_CLOSE(ema.value(), 100.0, 0.01);  // within 1% of true value

    // EMA of linearly growing series should lag
    EMA ema2(5);
    for (int i = 0; i < 100; ++i) ema2.update(static_cast<double>(i));
    // EMA lags by ~period/2 bars; last value=99, EMA ~ 99 - period/2
    CHECK(ema2.value() < 99.0);
    std::printf("  EMA(5) of [0..99] last value: %.4f (expected < 99)\n",
                ema2.value());
}

static void test_ema_reset() {
    section("EMA reset");

    EMA ema(5);
    for (int i = 0; i < 20; ++i) ema.update(100.0 + i);
    ema.reset();
    CHECK(ema.value()     == 0.0);
    CHECK(!ema.is_warm());
    CHECK(ema.bar_count() == 0);
}

static void test_ema_cross() {
    section("EMA cross signal");

    EMACross cross(3, 7);
    // No cross initially
    for (int i = 0; i < 20; ++i) {
        auto sig = cross.update(100.0);
        // No cross in flat series
        CHECK(sig.cross == 0 || !cross.is_warm());
    }

    // Now create a bullish cross: go from 90 to 110
    for (int i = 0; i < 10; ++i) {
        auto sig = cross.update(110.0);
        if (sig.cross != 0)
            std::printf("  Cross detected: %+d at iteration %d\n", sig.cross, i);
    }
}

static void test_batch_ema() {
    section("Batch EMA (4 instruments)");

    double prices[4] = { 100.0, 200.0, 300.0, 400.0 };
    double emas[4]   = { 100.0, 200.0, 300.0, 400.0 };
    double alpha = 0.2;

    simd::ema_update_batch(prices, emas, 4, alpha);

    // EMA = alpha * price + (1-alpha) * ema
    // Since price == ema initially, result should be unchanged
    CHECK_CLOSE(emas[0], 100.0, 1e-9);
    CHECK_CLOSE(emas[1], 200.0, 1e-9);
    CHECK_CLOSE(emas[2], 300.0, 1e-9);
    CHECK_CLOSE(emas[3], 400.0, 1e-9);

    // Now update with different prices
    double new_prices[4] = { 110.0, 190.0, 310.0, 390.0 };
    simd::ema_update_batch(new_prices, emas, 4, alpha);

    // Expected: alpha * new_price + (1-alpha) * old_ema
    CHECK_CLOSE(emas[0], 0.2 * 110.0 + 0.8 * 100.0, 1e-9);
    CHECK_CLOSE(emas[1], 0.2 * 190.0 + 0.8 * 200.0, 1e-9);
    CHECK_CLOSE(emas[2], 0.2 * 310.0 + 0.8 * 300.0, 1e-9);
    CHECK_CLOSE(emas[3], 0.2 * 390.0 + 0.8 * 400.0, 1e-9);

    std::printf("  emas after batch update: %.2f %.2f %.2f %.2f\n",
                emas[0], emas[1], emas[2], emas[3]);
}

// ============================================================
// ATR tests
// ============================================================

static void test_atr_manual() {
    section("ATR manual calculation");

    // Known sequence from Investopedia example:
    // High, Low, Close, True Range
    // Day 1: H=48.70, L=47.79, C=48.16, TR=0.91 (no prev close → HL only)
    // Day 2: H=48.72, L=48.14, C=48.61, TR=max(0.58, 0.56, 0.03) = 0.58
    // Day 3: H=48.90, L=48.39, C=48.75, TR=max(0.51, 0.29, 0.22) = 0.51

    ATR atr(3);
    double v1 = atr.update(48.70, 47.79, 48.16);
    CHECK_CLOSE(v1, 0.91, 0.001);

    double v2 = atr.update(48.72, 48.14, 48.61);
    // TR = max(0.58, |48.72-48.16|, |48.14-48.16|) = max(0.58, 0.56, 0.02) = 0.58
    // After 2 bars: simple avg = (0.91 + 0.58) / 2 = 0.745
    CHECK_CLOSE(v2, 0.745, 0.001);

    std::printf("  ATR values: v1=%.4f  v2=%.4f\n", v1, v2);
}

static void test_atr_wilder_smoothing() {
    section("ATR Wilder smoothing");

    ATR atr(14);
    // Feed 14 identical bars first to warm up
    for (int i = 0; i < 14; ++i) {
        atr.update(100.0, 99.0, 99.5);  // TR = 1.0
    }
    CHECK(atr.is_warm());
    CHECK_CLOSE(atr.value(), 1.0, 0.01);

    // Now a bar with TR = 2.0
    // Wilder: new_atr = prev_atr * (1 - 1/14) + TR * (1/14)
    double expected = 1.0 * (1.0 - 1.0/14.0) + 2.0 * (1.0/14.0);
    atr.update(101.0, 99.0, 100.0);
    CHECK_CLOSE(atr.value(), expected, 0.001);
}

// ============================================================
// RSI tests
// ============================================================

static void test_rsi_known_values() {
    section("RSI known values");

    // Sequence that gives RSI=70: 14 bars of +1 followed by observation
    // Known: if avg_gain=0.7, avg_loss=0.3, RS=7/3, RSI=87.5
    RSI rsi(14);

    // Feed 14 bars: alternating +1 and +0.5 gains (no losses)
    for (int i = 0; i < 14; ++i) {
        rsi.update(100.0 + i * 1.0);
    }
    CHECK(rsi.is_warm());
    // All gains → RSI should be high
    CHECK(rsi.value() > 70.0);
    std::printf("  RSI after 14 up bars: %.2f\n", rsi.value());

    // All-down sequence
    RSI rsi2(14);
    for (int i = 0; i < 20; ++i) {
        rsi2.update(100.0 - i * 1.0);
    }
    CHECK(rsi2.value() < 30.0);
    std::printf("  RSI after 14 down bars: %.2f\n", rsi2.value());
}

static void test_rsi_neutral() {
    section("RSI neutral (alternating)");

    RSI rsi(14);
    for (int i = 0; i < 50; ++i) {
        rsi.update(i % 2 == 0 ? 101.0 : 99.0);
    }
    // Equal ups and downs → RSI near 50
    std::printf("  RSI alternating up/down: %.2f (expect ~50)\n", rsi.value());
    CHECK(rsi.value() > 40.0 && rsi.value() < 60.0);
}

static void test_rsi_overbought_oversold() {
    section("RSI overbought/oversold detection");

    RSI rsi(14);
    for (int i = 0; i < 20; ++i) rsi.update(100.0 + i * 2.0);
    CHECK(rsi.is_overbought(70.0));
    CHECK(!rsi.is_oversold(30.0));

    RSI rsi2(14);
    for (int i = 0; i < 20; ++i) rsi2.update(100.0 - i * 2.0);
    CHECK(rsi2.is_oversold(30.0));
    CHECK(!rsi2.is_overbought(70.0));
}

// ============================================================
// Bollinger Band tests
// ============================================================

static void test_bollinger_basic() {
    section("Bollinger Bands basic");

    BollingerBands bb(20, 2.0);
    CHECK(!bb.is_warm());

    // Feed 20 bars of constant price
    for (int i = 0; i < 20; ++i) bb.update(100.0);
    CHECK(bb.is_warm());
    // With constant price: mean=100, std=0, upper=lower=mid=100
    CHECK_CLOSE(bb.mid(),   100.0, 0.001);
    CHECK_CLOSE(bb.upper(), 100.0, 0.001);
    CHECK_CLOSE(bb.lower(), 100.0, 0.001);
}

static void test_bollinger_bandwidth() {
    section("Bollinger bandwidth");

    BollingerBands bb(10, 2.0);
    // Higher variance → wider bands
    for (int i = 0; i < 20; ++i) {
        bb.update(100.0 + (i % 2 == 0 ? 5.0 : -5.0));
    }
    CHECK(bb.is_warm());
    CHECK(bb.bandwidth() > 0.0);
    double bw1 = bb.bandwidth();

    // Reset and feed lower variance series
    bb.reset();
    for (int i = 0; i < 20; ++i) {
        bb.update(100.0 + (i % 2 == 0 ? 1.0 : -1.0));
    }
    double bw2 = bb.bandwidth();
    std::printf("  High var BW=%.4f  Low var BW=%.4f\n", bw1, bw2);
    CHECK(bw1 > bw2);
}

static void test_bollinger_pct_b() {
    section("Bollinger %B");

    BollingerBands bb(20, 2.0);
    for (int i = 0; i < 20; ++i) bb.update(100.0 + i * 1.0);

    // Close at upper band: %B ≈ 1.0
    // Close at lower band: %B ≈ 0.0
    // Close at mid: %B ≈ 0.5
    double pct_b = bb.pct_b();
    std::printf("  %B = %.4f\n", pct_b);
    CHECK(pct_b >= 0.0);  // above lower band for uptrend
}

// ============================================================
// MACD tests
// ============================================================

static void test_macd_basics() {
    section("MACD basics");

    MACD macd(12, 26, 9);
    CHECK(!macd.is_warm());

    // After 26 bars, slow EMA should be warm
    for (int i = 0; i < 40; ++i) macd.update(100.0 + i * 0.5);
    CHECK(macd.is_warm());

    double line = macd.macd_line();
    // Uptrend: fast EMA > slow EMA → positive MACD
    std::printf("  MACD line=%.4f  signal=%.4f  hist=%.4f\n",
                line, macd.signal_line(), macd.histogram());
    CHECK(line > 0.0);
}

static void test_macd_crossover() {
    section("MACD crossover detection");

    // Test crossover helper function
    double prev_hist = -0.5, curr_hist = 0.5;
    CHECK(macd_crossover(prev_hist, curr_hist) == 1);   // bullish

    prev_hist = 0.5; curr_hist = -0.5;
    CHECK(macd_crossover(prev_hist, curr_hist) == -1);  // bearish

    prev_hist = curr_hist = 0.3;
    CHECK(macd_crossover(prev_hist, curr_hist) == 0);   // no cross
}

// ============================================================
// VWAP tests
// ============================================================

static void test_vwap_basic() {
    section("VWAP basic");

    VWAP vwap(constants::NS_PER_DAY);

    // Single bar: VWAP = typical price
    OHLCVBar bar(100.0, 102.0, 98.0, 101.0, 1000.0,
                 1700000000LL * constants::NS_PER_SEC);
    double tp1 = (102.0 + 98.0 + 101.0) / 3.0;  // 100.333...
    double v1  = vwap.update(bar);
    CHECK_CLOSE(v1, tp1, 0.001);

    // Second bar in same session
    OHLCVBar bar2(101.0, 103.0, 100.0, 102.0, 2000.0,
                  1700000000LL * constants::NS_PER_SEC + constants::NS_PER_MIN);
    double tp2 = (103.0 + 100.0 + 102.0) / 3.0;
    double v2  = vwap.update(bar2);

    // VWAP = (tp1*vol1 + tp2*vol2) / (vol1+vol2)
    double expected = (tp1 * 1000.0 + tp2 * 2000.0) / 3000.0;
    CHECK_CLOSE(v2, expected, 0.001);
    std::printf("  VWAP after 2 bars: %.4f  expected: %.4f\n", v2, expected);
}

static void test_vwap_session_reset() {
    section("VWAP session reset");

    VWAP vwap(constants::NS_PER_DAY);

    // Day 1 bar
    OHLCVBar bar1(100.0, 102.0, 98.0, 101.0, 1000.0,
                  1700000000LL * constants::NS_PER_SEC);
    vwap.update(bar1);
    double day1_vwap = vwap.value();

    // Day 2 bar (timestamp > day1 + 1 day)
    OHLCVBar bar2(200.0, 210.0, 190.0, 205.0, 500.0,
                  1700000000LL * constants::NS_PER_SEC + constants::NS_PER_DAY);
    double day2_vwap = vwap.update(bar2);

    // Day 2 VWAP should be based only on day 2 bar (session reset)
    double tp2 = (210.0 + 190.0 + 205.0) / 3.0;
    CHECK_CLOSE(day2_vwap, tp2, 0.001);
    std::printf("  Day1 VWAP=%.2f  Day2 VWAP=%.2f (should reset)\n",
                day1_vwap, day2_vwap);
}

// ============================================================
// Realized Vol tests
// ============================================================

static void test_parkinson_vol() {
    section("Parkinson volatility estimator");

    ParkinsonVol pv(20);

    // Feed bars with known H-L ratio
    // Parkinson = sqrt( (1/(4n*ln2)) * sum(ln(H/L)^2) * annualization )
    for (int i = 0; i < 30; ++i) {
        // H/L = 101/99, ln(H/L) ≈ 0.0202
        pv.update(101.0, 99.0);
    }
    CHECK(pv.is_warm());
    std::printf("  Parkinson vol (H=101,L=99): %.4f\n", pv.value());
    CHECK(pv.value() > 0.0);
    CHECK(pv.value() < 1.0);  // sanity: < 100% annual vol
}

static void test_garman_klass_vol() {
    section("Garman-Klass volatility estimator");

    GarmanKlassVol gkv(20);
    for (int i = 0; i < 30; ++i) {
        gkv.update(100.0, 101.0, 99.0, 100.5);
    }
    CHECK(gkv.is_warm());
    std::printf("  Garman-Klass vol: %.4f\n", gkv.value());
    CHECK(gkv.value() > 0.0);
}

static void test_yang_zhang_vol() {
    section("Yang-Zhang volatility estimator");

    YangZhangVol yzv(20);
    for (int i = 0; i < 40; ++i) {
        // Simulate overnight gap: open != prev_close
        yzv.update(100.0 + (i % 3) * 0.5, 102.0, 98.0, 101.0);
    }
    CHECK(yzv.is_warm());
    std::printf("  Yang-Zhang vol: %.4f\n", yzv.value());
    CHECK(yzv.value() > 0.0);
}

// ============================================================
// Main
// ============================================================

int main() {
    std::printf("=== Indicator Tests ===\n\n");

    test_ema_basics();
    test_ema_convergence();
    test_ema_reset();
    test_ema_cross();
    test_batch_ema();

    test_atr_manual();
    test_atr_wilder_smoothing();

    test_rsi_known_values();
    test_rsi_neutral();
    test_rsi_overbought_oversold();

    test_bollinger_basic();
    test_bollinger_bandwidth();
    test_bollinger_pct_b();

    test_macd_basics();
    test_macd_crossover();

    test_vwap_basic();
    test_vwap_session_reset();

    test_parkinson_vol();
    test_garman_klass_vol();
    test_yang_zhang_vol();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
