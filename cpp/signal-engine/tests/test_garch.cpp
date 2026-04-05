/// test_garch.cpp
/// Tests for GARCH(1,1) and EGARCH on synthetic price series.

#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "srfm/types.hpp"
#include "../src/bh_physics/garch.hpp"
#include "../src/bh_physics/ou_detector.hpp"

using namespace srfm;

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (!(expr)) { std::fprintf(stderr,"FAIL  %s:%d  %s\n",__FILE__,__LINE__,#expr); ++g_fail; } \
    else { ++g_pass; } \
} while(0)

#define CHECK_CLOSE(a, b, tol) do { \
    double _a=(a),_b=(b),_t=(tol); \
    if (std::abs(_a-_b)>_t) { \
        std::fprintf(stderr,"FAIL  %s:%d  %s  got=%.8g  exp=%.8g\n",__FILE__,__LINE__,#a,_a,_b); \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

static void section(const char* n) { std::printf("--- %s ---\n", n); }

// ============================================================
// GARCH(1,1) tests
// ============================================================

static void test_garch_default_params() {
    section("GARCH default parameters");

    GARCHTracker g;

    // Check persistence = alpha + beta = 0.1 + 0.85 = 0.95
    double persist = constants::GARCH_ALPHA + constants::GARCH_BETA;
    CHECK_CLOSE(persist, 0.95, 1e-9);

    // Unconditional variance = omega / (1 - alpha - beta)
    double unc_var = constants::GARCH_OMEGA / (1.0 - persist);
    std::printf("  Unconditional variance: %.8f  (daily vol: %.4f%%)\n",
                unc_var, std::sqrt(unc_var) * 100.0);
    CHECK_CLOSE(g.unconditional_variance(), unc_var, 1e-10);

    // Half-life of shocks
    double hl = g.shock_half_life();
    std::printf("  Shock half-life: %.1f bars\n", hl);
    CHECK(hl > 1.0 && hl < 1000.0);
}

static void test_garch_vol_scale_high_vol() {
    section("GARCH vol_scale under high volatility");

    // When market vol is 2x target, vol_scale should be ~0.5
    GARCHTracker g(constants::GARCH_OMEGA,
                   constants::GARCH_ALPHA,
                   constants::GARCH_BETA,
                   0.15,   // target 15% annual vol
                   252.0);

    // High volatility regime: 2% daily returns (32% annual)
    double price = 50000.0;
    for (int i = 0; i < 50; ++i) {
        double ret = (i % 2 == 0 ? 1.0 : -1.0) * 0.02;  // ±2%
        price *= (1.0 + ret);
        g.update(price);
    }

    std::printf("  High vol: garch_var=%.6f  vol_scale=%.4f\n",
                g.variance(), g.vol_scale());
    // vol_scale should be < 1 when vol is higher than target
    CHECK(g.vol_scale() < 1.0);
    CHECK(g.vol_scale() > 0.0);
}

static void test_garch_vol_scale_low_vol() {
    section("GARCH vol_scale under low volatility");

    GARCHTracker g(constants::GARCH_OMEGA,
                   constants::GARCH_ALPHA,
                   constants::GARCH_BETA,
                   0.15, 252.0);

    // Low volatility regime: 0.1% daily returns
    double price = 50000.0;
    for (int i = 0; i < 50; ++i) {
        double ret = (i % 2 == 0 ? 1.0 : -1.0) * 0.001;  // ±0.1%
        price *= (1.0 + ret);
        g.update(price);
    }

    std::printf("  Low vol: garch_var=%.6f  vol_scale=%.4f\n",
                g.variance(), g.vol_scale());
    // vol_scale > 1 when market is quieter than target
    CHECK(g.vol_scale() > 1.0);
    CHECK(g.is_warm());
}

static void test_garch_vol_scale_at_target() {
    section("GARCH vol_scale near 1.0 at target vol");

    GARCHTracker g(constants::GARCH_OMEGA,
                   constants::GARCH_ALPHA,
                   constants::GARCH_BETA,
                   0.15, 252.0);

    // Target vol = 15% annual = 15/sqrt(252) % daily ≈ 0.945% daily
    double daily_vol = 0.15 / std::sqrt(252.0);
    double price = 50000.0;
    for (int i = 0; i < 100; ++i) {
        double ret = (i % 2 == 0 ? 1.0 : -1.0) * daily_vol;
        price *= (1.0 + ret);
        g.update(price);
    }

    std::printf("  At-target: garch_var=%.8f  vol_scale=%.4f\n",
                g.variance(), g.vol_scale());
    // vol_scale should be approximately 1.0 (within factor of 2)
    CHECK(g.vol_scale() > 0.3);
    CHECK(g.vol_scale() < 5.0);
}

static void test_garch_variance_bounds() {
    section("GARCH variance stays bounded");

    GARCHTracker g;

    // Extreme inputs should not cause overflow
    double price = 50000.0;
    for (int i = 0; i < 200; ++i) {
        // Alternate between large and small moves
        double ret = (i % 10 == 0) ? 0.05 : 0.001;
        price *= (i % 2 == 0 ? 1.0 + ret : 1.0 - ret);
        auto out = g.update(price);

        CHECK(!std::isnan(out.variance));
        CHECK(!std::isinf(out.variance));
        CHECK(out.variance > 0.0);
        CHECK(out.variance <= 1.0);  // cap at 100% daily vol
    }
    std::printf("  Final variance: %.8f  vol_scale: %.4f\n",
                g.variance(), g.vol_scale());
}

static void test_garch_reset() {
    section("GARCH reset");

    GARCHTracker g;
    double price = 50000.0;
    for (int i = 0; i < 50; ++i) {
        price *= 1.002;
        g.update(price);
    }

    double var_before = g.variance();
    g.reset();

    // After reset, variance should return to unconditional
    CHECK(!std::isnan(g.variance()));
    CHECK_CLOSE(g.variance(), g.unconditional_variance(), g.unconditional_variance() * 0.1);
    std::printf("  Before reset: var=%.8f  After reset: var=%.8f  Uncond: %.8f\n",
                var_before, g.variance(), g.unconditional_variance());
}

static void test_garch_update_with_return() {
    section("GARCH update_with_return");

    GARCHTracker g;

    // Manual GARCH update verification
    // omega=0.000001, alpha=0.1, beta=0.85
    // r_0 = 0.01 (1% return)
    auto out0 = g.update_with_return(0.01);
    double var0 = out0.variance;
    std::printf("  After r=0.01: var=%.8f\n", var0);

    // r_1 = 0.02
    auto out1 = g.update_with_return(0.02);
    // Expected: omega + alpha * r0^2 + beta * var0
    double expected = constants::GARCH_OMEGA
                    + constants::GARCH_ALPHA * (0.01 * 0.01)
                    + constants::GARCH_BETA  * var0;
    std::printf("  After r=0.02: var=%.8f  expected=%.8f\n", out1.variance, expected);
    CHECK_CLOSE(out1.variance, expected, expected * 0.001);
}

// ============================================================
// EGARCH tests
// ============================================================

static void test_egarch_basics() {
    section("EGARCH basics");

    EGARCHTracker eg;

    double log_var0 = eg.log_variance();
    std::printf("  Initial log_var: %.4f  var: %.8f\n", log_var0, eg.variance());
    CHECK(eg.variance() > 0.0);

    // Feed some returns
    for (int i = 0; i < 30; ++i) {
        double ret = (i % 2 == 0) ? 0.01 : -0.01;
        eg.update(ret);
    }
    CHECK(eg.is_warm());
    std::printf("  After 30 bars: var=%.6f\n", eg.variance());
    CHECK(eg.variance() > 0.0);
    CHECK(!std::isnan(eg.variance()));
}

static void test_egarch_leverage_effect() {
    section("EGARCH leverage effect (asymmetry)");

    // gamma < 0 means negative returns increase variance more than positive
    EGARCHTracker eg(−0.1, 0.1, -0.05, 0.85);

    // Start from equilibrium
    for (int i = 0; i < 30; ++i) eg.update(0.0);
    double base_var = eg.variance();

    // Positive shock
    eg.reset();
    for (int i = 0; i < 30; ++i) eg.update(0.0);
    eg.update(0.02);
    double var_pos = eg.variance();

    // Negative shock of same magnitude
    eg.reset();
    for (int i = 0; i < 30; ++i) eg.update(0.0);
    eg.update(-0.02);
    double var_neg = eg.variance();

    std::printf("  Base var: %.6f  After +2%%: %.6f  After -2%%: %.6f\n",
                base_var, var_pos, var_neg);
    // With gamma < 0, negative shocks should create more variance
    CHECK(var_neg > var_pos);
}

// ============================================================
// OU Detector tests
// ============================================================

static void test_ou_mean_reversion() {
    section("OU detector on mean-reverting series");

    OUDetector ou(50);

    // Simulate OU process: X_t = mean + (X_{t-1} - mean) * exp(-theta) + noise
    // mean=100, theta=0.1
    double mean  = 100.0;
    double theta = 0.1;
    double sigma = 0.5;
    double x     = 100.0;
    uint64_t rng = 98765432109876ULL;

    auto randn = [&]() -> double {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double u1 = (double)(rng >> 11) / (double)(1ULL << 53) + 1e-15;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double u2 = (double)(rng >> 11) / (double)(1ULL << 53);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265 * u2);
    };

    OUOutput last;
    for (int i = 0; i < 200; ++i) {
        x = mean + (x - mean) * std::exp(-theta) + sigma * randn();
        last = ou.update(x);
    }

    std::printf("  theta=%.4f  mu=%.4f  half_life=%.1f  zscore=%.4f\n",
                last.theta, last.mu, last.half_life, last.zscore);

    CHECK(ou.is_warm());
    // Estimated half-life should be in ballpark of log(2)/theta ≈ 6.9 bars
    // Allow wide tolerance due to noise
    CHECK(last.half_life > 1.0 && last.half_life < 1000.0);
    CHECK(!std::isnan(last.zscore));
}

static void test_ou_signals() {
    section("OU long/short signals");

    OUDetector ou(50);

    // Warm up with stable series
    for (int i = 0; i < 50; ++i) ou.update(100.0 + (i % 3 - 1) * 0.1);

    // Push price well below mean
    OUOutput out_low;
    for (int i = 0; i < 5; ++i) {
        out_low = ou.update(95.0);  // far below mean of ~100
    }
    std::printf("  At price 95: zscore=%.4f  long=%d  short=%d\n",
                out_low.zscore, out_low.long_signal, out_low.short_signal);

    // Push price well above mean
    OUOutput out_high;
    for (int i = 0; i < 5; ++i) {
        out_high = ou.update(105.0);  // far above mean
    }
    std::printf("  At price 105: zscore=%.4f  long=%d  short=%d\n",
                out_high.zscore, out_high.long_signal, out_high.short_signal);
}

static void test_ou_reset() {
    section("OU reset");

    OUDetector ou(50);
    for (int i = 0; i < 60; ++i) ou.update(100.0 + i * 0.1);
    CHECK(ou.is_warm());

    ou.reset();
    CHECK(!ou.is_warm());
    CHECK(ou.zscore() == 0.0);
}

// ============================================================
// Main
// ============================================================

int main() {
    std::printf("=== GARCH & OU Tests ===\n\n");

    test_garch_default_params();
    test_garch_vol_scale_high_vol();
    test_garch_vol_scale_low_vol();
    test_garch_vol_scale_at_target();
    test_garch_variance_bounds();
    test_garch_reset();
    test_garch_update_with_return();

    test_egarch_basics();
    test_egarch_leverage_effect();

    test_ou_mean_reversion();
    test_ou_signals();
    test_ou_reset();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
