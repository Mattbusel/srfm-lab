//! test_stats.zig -- Unit tests for stats.zig and hurst.zig.
//!
//! Covers:
//!   - RunningMean accuracy
//!   - RunningVariance Welford correctness
//!   - EWMA convergence
//!   - EWMVariance lambda sensitivity
//!   - RunningMedian correctness
//!   - RollingQuantile P-squared accuracy vs sorted array
//!   - RunningCorrelation vs closed-form Pearson
//!   - LinearRegression slope recovery
//!   - HurstEstimator: GBM series H ~ 0.5
//!   - HurstEstimator: trending series H > 0.5

const std = @import("std");
const math = std.math;
const testing = std.testing;
const stats = @import("stats");
const hurst = @import("hurst");

const eps32: f32 = 1e-4;
const eps64: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mean_slice(xs: []const f64) f64 {
    var s: f64 = 0;
    for (xs) |x| s += x;
    return s / @as(f64, @floatFromInt(xs.len));
}

fn variance_slice(xs: []const f64) f64 {
    const m = mean_slice(xs);
    var s: f64 = 0;
    for (xs) |x| {
        const d = x - m;
        s += d * d;
    }
    return s / @as(f64, @floatFromInt(xs.len - 1));
}

fn pearson_slice(xs: []const f64, ys: []const f64) f64 {
    const n = xs.len;
    var mx: f64 = 0;
    var my: f64 = 0;
    for (xs) |x| mx += x;
    for (ys) |y| my += y;
    mx /= @floatFromInt(n);
    my /= @floatFromInt(n);
    var cov: f64 = 0;
    var vx: f64 = 0;
    var vy: f64 = 0;
    for (0..n) |i| {
        cov += (xs[i] - mx) * (ys[i] - my);
        vx  += (xs[i] - mx) * (xs[i] - mx);
        vy  += (ys[i] - my) * (ys[i] - my);
    }
    return cov / @sqrt(vx * vy);
}

// ---------------------------------------------------------------------------
// RunningMean tests
// ---------------------------------------------------------------------------

test "running_mean_accuracy_integers" {
    var rm = stats.RunningMean(f64).init();
    const vals = [_]f64{ 1, 2, 3, 4, 5 };
    for (vals) |v| rm.update(v);
    try testing.expectApproxEqAbs(@as(f64, 3.0), rm.get(), 1e-10);
    try testing.expectEqual(@as(u64, 5), rm.count());
}

test "running_mean_accuracy_floats" {
    var rm = stats.RunningMean(f64).init();
    const vals = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    for (vals) |v| rm.update(v);
    try testing.expectApproxEqAbs(@as(f64, 0.55), rm.get(), 1e-10);
}

test "running_mean_single_value" {
    var rm = stats.RunningMean(f32).init();
    rm.update(42.0);
    try testing.expectApproxEqAbs(@as(f32, 42.0), rm.get(), 1e-5);
    try testing.expectEqual(@as(u64, 1), rm.count());
}

test "running_mean_reset" {
    var rm = stats.RunningMean(f64).init();
    rm.update(100);
    rm.update(200);
    rm.reset();
    try testing.expectEqual(@as(u64, 0), rm.count());
    try testing.expectApproxEqAbs(@as(f64, 0.0), rm.get(), 1e-10);
}

test "running_mean_large_n" {
    var rm = stats.RunningMean(f64).init();
    for (0..10_000) |i| {
        rm.update(@floatFromInt(i));
    }
    // sum(0..9999) / 10000 = 4999.5
    try testing.expectApproxEqAbs(@as(f64, 4999.5), rm.get(), 1e-6);
}

// ---------------------------------------------------------------------------
// RunningVariance tests
// ---------------------------------------------------------------------------

test "running_variance_correctness" {
    const data = [_]f64{ 2, 4, 4, 4, 5, 5, 7, 9 };
    var rv = stats.RunningVariance(f64).init();
    for (data) |d| rv.update(d);

    // True sample variance = 4.571... (std dev ~2.138)
    const expected_var = variance_slice(&data);
    try testing.expectApproxEqAbs(expected_var, rv.varianceSample(), 1e-8);
}

test "running_variance_stddev" {
    var rv = stats.RunningVariance(f64).init();
    const data = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    for (data) |d| rv.update(d);
    const expected_std = @sqrt(variance_slice(&data));
    try testing.expectApproxEqAbs(expected_std, rv.stddev(), 1e-8);
}

test "running_variance_skewness_normal" {
    // For symmetric distribution, skewness should be near 0
    var rv = stats.RunningVariance(f64).init();
    const data = [_]f64{ -3, -2, -1, 0, 1, 2, 3 };
    for (data) |d| rv.update(d);
    try testing.expectApproxEqAbs(@as(f64, 0.0), rv.skewness(), 1e-8);
}

test "running_variance_kurtosis_insufficient" {
    var rv = stats.RunningVariance(f64).init();
    rv.update(1.0);
    rv.update(2.0);
    try testing.expectApproxEqAbs(@as(f64, 0.0), rv.kurtosis(), 1e-10);
}

test "running_variance_f32" {
    var rv = stats.RunningVariance(f32).init();
    rv.update(1.0);
    rv.update(3.0);
    // sample variance of [1, 3] = 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), rv.varianceSample(), 1e-5);
}

// ---------------------------------------------------------------------------
// EWMA tests
// ---------------------------------------------------------------------------

test "ewma_convergence_constant_series" {
    // Feeding a constant value, EWMA should converge to that value
    var e = stats.EWMA(f64).init(0.1);
    for (0..200) |_| e.update(5.0);
    try testing.expectApproxEqAbs(@as(f64, 5.0), e.get(), 1e-3);
}

test "ewma_init_from_span" {
    var e = stats.EWMA(f64).initFromSpan(9);
    // alpha = 2/(9+1) = 0.2
    e.update(10.0);
    e.update(10.0);
    e.update(10.0);
    // After 3 updates starting from 10, should still be 10
    try testing.expectApproxEqAbs(@as(f64, 10.0), e.get(), 1e-10);
}

test "ewma_alpha_sensitivity" {
    // Fast alpha converges faster than slow alpha
    var fast = stats.EWMA(f64).init(0.5);
    var slow = stats.EWMA(f64).init(0.05);
    for (0..10) |_| {
        fast.update(100.0);
        slow.update(100.0);
    }
    // After 10 updates toward 100, fast should be closer to 100 than slow
    try testing.expect(fast.get() > slow.get());
}

test "ewma_f32" {
    var e = stats.EWMA(f32).init(0.3);
    for (0..100) |_| e.update(7.0);
    try testing.expectApproxEqAbs(@as(f32, 7.0), e.get(), 1e-4);
}

test "ewma_reset" {
    var e = stats.EWMA(f64).init(0.1);
    e.update(999.0);
    e.reset();
    e.update(1.0);
    try testing.expectApproxEqAbs(@as(f64, 1.0), e.get(), 1e-10);
}

// ---------------------------------------------------------------------------
// EWMVariance tests
// ---------------------------------------------------------------------------

test "ewmvar_lambda_sensitivity" {
    // lambda=0.94 (standard RiskMetrics) -- variance should be bounded
    var ev = stats.EWMVariance(f64).init(0.94);
    const rets = [_]f64{ 0.01, -0.02, 0.015, -0.005, 0.03, -0.025, 0.01, -0.01 };
    for (rets) |r| ev.update(r);
    try testing.expect(ev.get() > 0);
    try testing.expect(ev.volatility() > 0);
    try testing.expect(ev.volatility() < 1.0); // sanity: vol should be < 100%
}

test "ewmvar_higher_lambda_smoother" {
    var high = stats.EWMVariance(f64).init(0.99);
    var low  = stats.EWMVariance(f64).init(0.90);

    // Spike in returns
    const normal_ret = 0.001;
    for (0..50) |_| {
        high.update(normal_ret);
        low.update(normal_ret);
    }
    high.update(0.10); // spike
    low.update(0.10);

    // After one spike, low lambda reacts more (higher variance)
    try testing.expect(low.get() > high.get());
}

test "ewmvar_constant_zero_returns" {
    var ev = stats.EWMVariance(f64).init(0.94);
    for (0..100) |_| ev.update(0.0);
    // Variance of zero returns decays to 0
    try testing.expectApproxEqAbs(@as(f64, 0.0), ev.get(), 1e-6);
}

// ---------------------------------------------------------------------------
// RunningMedian tests
// ---------------------------------------------------------------------------

test "running_median_odd_count" {
    const alloc = testing.allocator;
    var med = stats.RunningMedian(f64).init(alloc);
    defer med.deinit();

    try med.update(3.0);
    try med.update(1.0);
    try med.update(2.0);
    // Sorted: [1, 2, 3] -> median = 2
    try testing.expectApproxEqAbs(@as(f64, 2.0), med.get(), 1e-10);
}

test "running_median_even_count" {
    const alloc = testing.allocator;
    var med = stats.RunningMedian(f64).init(alloc);
    defer med.deinit();

    try med.update(1.0);
    try med.update(2.0);
    try med.update(3.0);
    try med.update(4.0);
    // Sorted: [1, 2, 3, 4] -> median = (2+3)/2 = 2.5
    try testing.expectApproxEqAbs(@as(f64, 2.5), med.get(), 1e-10);
}

test "running_median_single" {
    const alloc = testing.allocator;
    var med = stats.RunningMedian(f64).init(alloc);
    defer med.deinit();

    try med.update(7.0);
    try testing.expectApproxEqAbs(@as(f64, 7.0), med.get(), 1e-10);
}

test "running_median_duplicates" {
    const alloc = testing.allocator;
    var med = stats.RunningMedian(f64).init(alloc);
    defer med.deinit();

    try med.update(5.0);
    try med.update(5.0);
    try med.update(5.0);
    try testing.expectApproxEqAbs(@as(f64, 5.0), med.get(), 1e-10);
}

// ---------------------------------------------------------------------------
// RollingQuantile tests (P-squared vs sorted array)
// ---------------------------------------------------------------------------

test "rolling_quantile_p50_accuracy" {
    var q = stats.RollingQuantile(f64).init(0.5);
    // Feed 1000 uniform [0,1] values (deterministic sequence)
    var sum: f64 = 0;
    for (0..1000) |i| {
        const x = @as(f64, @floatFromInt(i)) / 999.0;
        q.update(x);
        sum += x;
    }
    // True median of [0, 1/999, 2/999, ..., 1] is ~0.5
    const est = q.get();
    try testing.expect(est > 0.3);
    try testing.expect(est < 0.7);
}

test "rolling_quantile_p90_accuracy" {
    var q = stats.RollingQuantile(f64).init(0.9);
    for (0..2000) |i| {
        const x = @as(f64, @floatFromInt(i % 100)) / 99.0;
        q.update(x);
    }
    const est = q.get();
    // True 90th percentile of uniform [0,1] is 0.9
    try testing.expect(est > 0.7);
    try testing.expect(est < 1.0);
}

test "rolling_quantile_requires_5_samples" {
    var q = stats.RollingQuantile(f64).init(0.5);
    q.update(1.0);
    q.update(2.0);
    q.update(3.0);
    // Fewer than 5 samples returns 0
    try testing.expectApproxEqAbs(@as(f64, 0.0), q.get(), 1e-10);
}

// ---------------------------------------------------------------------------
// RunningCorrelation tests
// ---------------------------------------------------------------------------

test "online_correlation_perfect_positive" {
    var rc = stats.RunningCorrelation(f64).init();
    for (0..100) |i| {
        const x: f64 = @floatFromInt(i);
        rc.update(x, x);
    }
    try testing.expectApproxEqAbs(@as(f64, 1.0), rc.get(), 1e-10);
}

test "online_correlation_perfect_negative" {
    var rc = stats.RunningCorrelation(f64).init();
    for (0..100) |i| {
        const x: f64 = @floatFromInt(i);
        rc.update(x, -x);
    }
    try testing.expectApproxEqAbs(@as(f64, -1.0), rc.get(), 1e-10);
}

test "online_correlation_near_zero" {
    // x = 1..100, y = 100..1 (perfect negative correlation)
    // Use orthogonal-ish sequences for near-zero
    var rc = stats.RunningCorrelation(f64).init();
    const n = 50;
    for (0..n) |i| {
        const x: f64 = @floatFromInt(i);
        // y cycles: 0,1,0,1,... -- uncorrelated with linear x
        const y: f64 = if (i % 2 == 0) 0.0 else 1.0;
        rc.update(x, y);
    }
    // Correlation with alternating series and linear should be small
    try testing.expect(@abs(rc.get()) < 0.5);
}

test "online_correlation_vs_closed_form" {
    const xs = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const ys = [_]f64{ 2, 4, 5, 4, 5, 7, 8, 9, 10, 12 };

    var rc = stats.RunningCorrelation(f64).init();
    for (0..xs.len) |i| rc.update(xs[i], ys[i]);

    const expected = pearson_slice(&xs, &ys);
    try testing.expectApproxEqAbs(expected, rc.get(), 1e-8);
}

test "online_correlation_reset" {
    var rc = stats.RunningCorrelation(f64).init();
    for (0..50) |i| {
        const x: f64 = @floatFromInt(i);
        rc.update(x, x);
    }
    rc.reset();
    try testing.expectEqual(@as(u64, 0), rc.count());
    try testing.expectApproxEqAbs(@as(f64, 0.0), rc.get(), 1e-10);
}

// ---------------------------------------------------------------------------
// LinearRegression tests
// ---------------------------------------------------------------------------

test "linear_regression_slope_recovery" {
    // Feed y = 2*t + 1, expect slope near 2, intercept near 1
    var lr = stats.LinearRegression(f64).initDefault();
    for (0..200) |i| {
        const t: f64 = @floatFromInt(i);
        lr.update(2.0 * t + 1.0);
    }
    const slope = lr.slope();
    try testing.expectApproxEqAbs(@as(f64, 2.0), slope, 0.05);
}

test "linear_regression_flat_series" {
    var lr = stats.LinearRegression(f64).initDefault();
    for (0..100) |_| lr.update(5.0);
    // Flat series: slope should be near 0
    try testing.expect(@abs(lr.slope()) < 0.1);
}

test "linear_regression_predict" {
    var lr = stats.LinearRegression(f64).initDefault();
    for (0..100) |i| {
        const t: f64 = @floatFromInt(i);
        lr.update(3.0 * t);
    }
    // After 100 steps, predict step 101 should be near 3*100=300
    const pred = lr.predict();
    try testing.expect(pred > 200.0);
    try testing.expect(pred < 400.0);
}

// ---------------------------------------------------------------------------
// Hurst exponent tests
// ---------------------------------------------------------------------------

/// Generate a simple Gaussian random walk (GBM proxy).
/// Uses a LCG for deterministic reproducibility.
fn generateGBM(allocator: std.mem.Allocator, n: usize, seed: u64) ![]f64 {
    var prices = try allocator.alloc(f64, n);
    var state: u64 = seed;
    prices[0] = 100.0;
    for (1..n) |i| {
        // LCG: Knuth's constants
        state = state *% 6364136223846793005 +% 1442695040888963407;
        // Map to [-1, 1] range, scale to small return
        const u: f64 = @as(f64, @floatFromInt(state >> 11)) / @as(f64, @floatFromInt(1 << 53));
        const noise = (u - 0.5) * 0.02;
        prices[i] = prices[i - 1] * @exp(noise);
    }
    return prices;
}

/// Generate a trending series (H > 0.5) by cumulative sums of positive drift.
fn generateTrending(allocator: std.mem.Allocator, n: usize) ![]f64 {
    var prices = try allocator.alloc(f64, n);
    prices[0] = 100.0;
    for (1..n) |i| {
        const trend = 0.003; // positive drift each step
        prices[i] = prices[i - 1] * (1.0 + trend);
    }
    return prices;
}

test "hurst_brownian_motion_near_half" {
    const alloc = testing.allocator;
    const n = 800;
    const prices = try generateGBM(alloc, n, 42);
    defer alloc.free(prices);

    const H = hurst.HurstEstimator(512);
    var h = H.init();
    for (prices) |p| h.update(@log(p));

    const hval = h.get();
    // GBM H should be in [0.3, 0.7] range
    try testing.expect(hval > 0.25);
    try testing.expect(hval < 0.75);
}

test "hurst_trending_above_half" {
    const alloc = testing.allocator;
    const n = 600;
    const prices = try generateTrending(alloc, n);
    defer alloc.free(prices);

    const H = hurst.HurstEstimator(512);
    var h = H.init();
    for (prices) |p| h.update(@log(p));

    const hval = h.get();
    // Strongly trending series should have H > 0.5
    try testing.expect(hval > 0.5);
}

test "hurst_sample_count_tracking" {
    const H = hurst.HurstEstimator(256);
    var h = H.init();
    try testing.expectEqual(@as(usize, 0), h.sampleCount());
    h.update(4.6);
    h.update(4.61);
    try testing.expectEqual(@as(usize, 2), h.sampleCount());
}

test "hurst_insufficient_data_default" {
    const H = hurst.HurstEstimator(256);
    var h = H.init();
    for (0..10) |i| {
        h.update(@floatFromInt(i));
    }
    // With < 16 samples, stays at default 0.5
    try testing.expectApproxEqAbs(@as(f64, 0.5), h.get(), 1e-10);
}

test "hurst_circular_buffer_full" {
    const H = hurst.HurstEstimator(64);
    var h = H.init();
    // Feed 100 values to overflow the 64-element buffer
    for (0..100) |i| {
        const p: f64 = 100.0 + @as(f64, @floatFromInt(i)) * 0.01;
        h.update(@log(p));
    }
    // Should not crash and H should be a valid number
    const hval = h.get();
    try testing.expect(!math.isNan(hval));
    try testing.expect(!math.isInf(hval));
    try testing.expect(hval >= 0.01);
    try testing.expect(hval <= 0.99);
}
