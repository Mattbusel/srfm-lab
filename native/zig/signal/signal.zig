// signal.zig -- SRFM signal computation library in pure Zig.
// Exports C ABI functions for use from Python ctypes or C FFI.
//
// Includes:
//   - BH mass sigmoid activation from Minkowski metric ds2
//   - Quaternion navigation curvature from angular velocity
//   - SIMD Hurst exponent batch computation via R/S analysis
//
// Zig 0.12+ syntax. Build with ReleaseFast for production use.

const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const builtin = @import("builtin");

// Detect AVX2 capability at comptime
const has_avx2 = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

// ============================================================
// BH Mass -- sigmoid activation of Minkowski metric ds2
// ============================================================
//
// ds2 = -(dT)^2 + (dX)^2 + (dY)^2 + (dZ)^2
// T = close, X = open, Y = high, Z = low (each normalized by ATR)
//
// Returns: 1.0 / (1.0 + exp(-ds2 / bh_mass_thresh))
// This is the logistic sigmoid. When ds2 > 0 (spacelike separation) =>
// mass > 0.5 (BH forming). When ds2 < 0 (timelike) => mass < 0.5.

pub fn computeBHMass(ds2: f64, bh_mass_thresh: f64) f64 {
    if (bh_mass_thresh == 0.0) return 0.5;
    const x = -ds2 / bh_mass_thresh;
    // Numerically stable sigmoid: avoid overflow for large |x|
    if (x >= 0.0) {
        const ex = @exp(-x);
        return 1.0 / (1.0 + ex);
    } else {
        const ex = @exp(x);
        return ex / (1.0 + ex);
    }
}

// computeBHMassFromBar -- convenience wrapper that builds ds2 from OHLCV bar.
// atr: average true range used for normalization. Must be > 0.
// Returns sigmoid activation in (0, 1).
pub fn computeBHMassFromBar(
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    prev_close: f64,
    atr: f64,
    bh_mass_thresh: f64,
) f64 {
    if (atr <= 0.0) return 0.5;
    const inv_atr = 1.0 / atr;
    // Coordinate displacements normalized by ATR
    const dT = (close - prev_close) * inv_atr;
    const dX = (open  - prev_close) * inv_atr;
    const dY = (high  - prev_close) * inv_atr;
    const dZ = (low   - prev_close) * inv_atr;
    // Minkowski metric: -(dT)^2 + (dX)^2 + (dY)^2 + (dZ)^2
    const ds2 = -(dT * dT) + (dX * dX) + (dY * dY) + (dZ * dZ);
    return computeBHMass(ds2, bh_mass_thresh);
}

// ============================================================
// Quaternion navigation curvature
// ============================================================
//
// q, prev_q: unit quaternions [w, x, y, z] representing orientation in SRFM 4-space.
// Bar quaternion is built from (close, open, high, low) normalized.
//
// Angular velocity = magnitude of the quaternion product q * conj(prev_q).
// This gives the rotation needed to go from prev_q to q.
// The rotation magnitude in radians is 2 * acos(|w_component|).
//
// Returns geodesic curvature estimate = angular_velocity.

pub fn computeNavCurvature(q: [4]f64, prev_q: [4]f64) f64 {
    // Conjugate of prev_q: [w, -x, -y, -z]
    const cw = prev_q[0];
    const cx = -prev_q[1];
    const cy = -prev_q[2];
    const cz = -prev_q[3];

    // Hamilton product: (q) * (conj(prev_q))
    // (w1+x1i+y1j+z1k) * (w2+x2i+y2j+z2k)
    const rw = q[0]*cw - q[1]*cx - q[2]*cy - q[3]*cz;
    const rx = q[0]*cx + q[1]*cw + q[2]*cz - q[3]*cy;
    const ry = q[0]*cy - q[1]*cz + q[2]*cw + q[3]*cx;
    const rz = q[0]*cz + q[1]*cy - q[2]*cx + q[3]*cw;

    // Magnitude of result quaternion (should be ~1 for unit quaternions)
    const mag = @sqrt(rw*rw + rx*rx + ry*ry + rz*rz);

    // Angular velocity in radians: 2 * acos(|w / mag|)
    // acos domain: [-1, 1], clamp for numerical safety
    if (mag < 1e-12) return 0.0;
    const w_norm = rw / mag;
    const w_clamped = @min(1.0, @max(-1.0, w_norm));
    const half_angle = math.acos(f64, @abs(w_clamped));
    return 2.0 * half_angle;
}

// buildBarQuaternion -- construct a unit quaternion from bar OHLC.
// Maps (close, open, high, low) to quaternion components and normalizes.
pub fn buildBarQuaternion(open: f64, high: f64, low: f64, close: f64) [4]f64 {
    const w = close;
    const x = open;
    const y = high;
    const z = low;
    const mag = @sqrt(w*w + x*x + y*y + z*z);
    if (mag < 1e-12) return .{ 1.0, 0.0, 0.0, 0.0 };
    const inv = 1.0 / mag;
    return .{ w * inv, x * inv, y * inv, z * inv };
}

// ============================================================
// R/S analysis helpers (scalar)
// ============================================================

// computeRS -- R/S statistic for a slice of log-prices.
// scratch must be at least len(prices) elements.
fn computeRS(prices: []const f64, scratch: []f64) f64 {
    const n = prices.len;
    if (n < 2) return 0.0;
    const m = n - 1;

    // Compute log-returns and their mean
    var mean: f64 = 0.0;
    for (0..m) |i| {
        scratch[i] = prices[i + 1] - prices[i];
        mean += scratch[i];
    }
    mean /= @as(f64, @floatFromInt(m));

    // Subtract mean and compute std dev
    var var_acc: f64 = 0.0;
    for (0..m) |i| {
        const d = scratch[i] - mean;
        scratch[i] = d;
        var_acc += d * d;
    }
    const std_dev = @sqrt(var_acc / @as(f64, @floatFromInt(m)));
    if (std_dev < 1e-15) return 1.0;

    // Cumulative sum => range
    var cum: f64 = 0.0;
    var max_cum: f64 = -math.floatMax(f64);
    var min_cum: f64 =  math.floatMax(f64);
    for (0..m) |i| {
        cum += scratch[i];
        if (cum > max_cum) max_cum = cum;
        if (cum < min_cum) min_cum = cum;
    }
    return (max_cum - min_cum) / std_dev;
}

// computeRSSimd -- SIMD-accelerated R/S using @Vector(4, f64).
fn computeRSSimd(prices: []const f64, scratch: []f64) f64 {
    const n = prices.len;
    if (n < 2) return 0.0;
    const m = n - 1;

    // Compute log-returns with SIMD if available
    var mean: f64 = 0.0;
    if (has_avx2 and m >= 4) {
        const Vec4 = @Vector(4, f64);
        var sum_vec: Vec4 = @splat(0.0);
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            const a: Vec4 = .{ prices[i],   prices[i+1], prices[i+2], prices[i+3] };
            const b: Vec4 = .{ prices[i+1], prices[i+2], prices[i+3], prices[i+4] };
            const r = b - a;
            scratch[i]   = r[0];
            scratch[i+1] = r[1];
            scratch[i+2] = r[2];
            scratch[i+3] = r[3];
            sum_vec += r;
        }
        mean = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
        while (i < m) : (i += 1) {
            scratch[i] = prices[i+1] - prices[i];
            mean += scratch[i];
        }
    } else {
        for (0..m) |i| {
            scratch[i] = prices[i+1] - prices[i];
            mean += scratch[i];
        }
    }
    mean /= @as(f64, @floatFromInt(m));

    // Subtract mean, accumulate variance
    var var_acc: f64 = 0.0;
    if (has_avx2 and m >= 4) {
        const Vec4 = @Vector(4, f64);
        const mean_v: Vec4 = @splat(mean);
        var var_v: Vec4 = @splat(0.0);
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            const r: Vec4 = .{ scratch[i], scratch[i+1], scratch[i+2], scratch[i+3] };
            const d = r - mean_v;
            scratch[i]   = d[0];
            scratch[i+1] = d[1];
            scratch[i+2] = d[2];
            scratch[i+3] = d[3];
            var_v += d * d;
        }
        var_acc = var_v[0] + var_v[1] + var_v[2] + var_v[3];
        while (i < m) : (i += 1) {
            const d = scratch[i] - mean;
            scratch[i] = d;
            var_acc += d * d;
        }
    } else {
        for (0..m) |i| {
            const d = scratch[i] - mean;
            scratch[i] = d;
            var_acc += d * d;
        }
    }

    const std_dev = @sqrt(var_acc / @as(f64, @floatFromInt(m)));
    if (std_dev < 1e-15) return 1.0;

    var cum: f64 = 0.0;
    var max_cum: f64 = -math.floatMax(f64);
    var min_cum: f64 =  math.floatMax(f64);
    for (0..m) |i| {
        cum += scratch[i];
        if (cum > max_cum) max_cum = cum;
        if (cum < min_cum) min_cum = cum;
    }
    return (max_cum - min_cum) / std_dev;
}

// OLS slope of log(y) vs log(x) for up to 4 points
fn fitOLS(log_x: []const f64, log_y: []const f64) f64 {
    const np = log_x.len;
    if (np < 2) return 0.5;
    assert(np == log_y.len);

    var sx: f64 = 0; var sy: f64 = 0;
    var sxx: f64 = 0; var sxy: f64 = 0;
    for (0..np) |i| {
        sx  += log_x[i];
        sy  += log_y[i];
        sxx += log_x[i] * log_x[i];
        sxy += log_x[i] * log_y[i];
    }
    const nf: f64 = @floatFromInt(np);
    const denom = nf * sxx - sx * sx;
    if (@abs(denom) < 1e-14) return 0.5;
    const h = (nf * sxy - sx * sy) / denom;
    return @max(0.01, @min(0.99, h));
}

// ============================================================
// computeHurstSingle -- Hurst exponent for a single window of closes
// ============================================================

fn computeHurstSingle(closes: []const f64, scratch: []f64) f64 {
    const n = closes.len;
    if (n < 8) return 0.5;

    // Sub-window sizes: n/8, n/4, n/2, n
    const subs = [4]usize{
        @max(8, n / 8),
        @max(8, n / 4),
        @max(8, n / 2),
        n,
    };

    var log_ns: [4]f64 = undefined;
    var log_rs: [4]f64 = undefined;
    var valid: usize = 0;

    for (subs) |sub_n| {
        if (sub_n < 8 or sub_n > n) continue;
        const start = n - sub_n;
        const rs = computeRSSimd(closes[start..n], scratch);
        if (rs > 0.0) {
            log_ns[valid] = @log(@as(f64, @floatFromInt(sub_n)));
            log_rs[valid] = @log(rs);
            valid += 1;
        }
    }

    if (valid < 2) return 0.5;
    return fitOLS(log_ns[0..valid], log_rs[0..valid]);
}

// ============================================================
// computeHurstBatch -- SIMD rolling Hurst for each position in closes
// ============================================================
//
// For each index i in [window-1, len(closes)), computes the Hurst exponent
// for the window closes[i-window+1 .. i+1].
// results must have len(closes) elements; positions < window-1 get 0.5.

pub fn computeHurstBatch(closes: []const f64, window: usize, results: []f64) void {
    assert(results.len >= closes.len);

    if (closes.len == 0 or window < 8) {
        for (results[0..closes.len]) |*r| r.* = 0.5;
        return;
    }

    // Allocate scratch on the stack for small windows, heap for large.
    // Stack: up to 512 doubles (4KB) -- safe for window <= 512.
    const STACK_MAX: usize = 512;
    if (window <= STACK_MAX) {
        var scratch_buf: [STACK_MAX]f64 = undefined;
        for (0..closes.len) |i| {
            if (i + 1 < window) {
                results[i] = 0.5;
                continue;
            }
            const start = i + 1 - window;
            results[i] = computeHurstSingle(
                closes[start .. i + 1],
                scratch_buf[0..window],
            );
        }
    } else {
        // Heap allocation for large windows
        const allocator = std.heap.c_allocator;
        const scratch = allocator.alloc(f64, window) catch {
            for (results[0..closes.len]) |*r| r.* = 0.5;
            return;
        };
        defer allocator.free(scratch);

        for (0..closes.len) |i| {
            if (i + 1 < window) {
                results[i] = 0.5;
                continue;
            }
            const start = i + 1 - window;
            results[i] = computeHurstSingle(closes[start .. i + 1], scratch);
        }
    }
}

// ============================================================
// C ABI exports
// ============================================================

// bh_mass_compute -- compute BH mass activation.
// ds2: Minkowski metric value. thresh: BH mass threshold (e.g., 1.92).
export fn bh_mass_compute(ds2: f64, thresh: f64) f64 {
    return computeBHMass(ds2, thresh);
}

// bh_mass_from_bar -- compute BH mass from bar OHLCV.
// prev_close: closing price of prior bar. atr: average true range.
export fn bh_mass_from_bar(
    open: f64, high: f64, low: f64, close: f64,
    prev_close: f64, atr: f64, thresh: f64,
) f64 {
    return computeBHMassFromBar(open, high, low, close, prev_close, atr, thresh);
}

// nav_curvature_compute -- geodesic curvature from quaternion pair.
// q and prev_q are pointers to 4-element f64 arrays [w, x, y, z].
export fn nav_curvature_compute(q: [*]const f64, prev_q: [*]const f64) f64 {
    const qa: [4]f64 = .{ q[0], q[1], q[2], q[3] };
    const pqa: [4]f64 = .{ prev_q[0], prev_q[1], prev_q[2], prev_q[3] };
    return computeNavCurvature(qa, pqa);
}

// nav_build_quaternion -- build unit quaternion from bar OHLC and write to out[4].
export fn nav_build_quaternion(
    open: f64, high: f64, low: f64, close: f64,
    out: [*]f64,
) void {
    const q = buildBarQuaternion(open, high, low, close);
    out[0] = q[0];
    out[1] = q[1];
    out[2] = q[2];
    out[3] = q[3];
}

// hurst_batch -- rolling Hurst batch computation.
// closes: pointer to n close prices (log prices recommended).
// window: rolling window size.
// out: output array of n Hurst estimates.
export fn hurst_batch(closes: [*]const f64, n: usize, window: usize, out: [*]f64) void {
    if (n == 0) return;
    const closes_slice = closes[0..n];
    const out_slice    = out[0..n];
    computeHurstBatch(closes_slice, window, out_slice);
}

// hurst_single -- compute Hurst for a single window.
// Returns Hurst exponent in [0.01, 0.99].
export fn hurst_single(closes: [*]const f64, n: usize) f64 {
    if (n < 8) return 0.5;
    const STACK_MAX: usize = 1024;
    if (n <= STACK_MAX) {
        var scratch: [STACK_MAX]f64 = undefined;
        return computeHurstSingle(closes[0..n], scratch[0..n]);
    }
    const allocator = std.heap.c_allocator;
    const scratch = allocator.alloc(f64, n) catch return 0.5;
    defer allocator.free(scratch);
    return computeHurstSingle(closes[0..n], scratch);
}

// ============================================================
// Tests (run with `zig test signal.zig`)
// ============================================================

test "bh_mass basic" {
    const testing = std.testing;
    // ds2 = 0 => sigmoid(0) = 0.5
    try testing.expectApproxEqAbs(computeBHMass(0.0, 1.92), 0.5, 1e-6);
    // ds2 large positive => mass > 0.5
    try testing.expect(computeBHMass(10.0, 1.92) > 0.5);
    // ds2 large negative => mass < 0.5
    try testing.expect(computeBHMass(-10.0, 1.92) < 0.5);
    // thresh = 0 => returns 0.5
    try testing.expectApproxEqAbs(computeBHMass(5.0, 0.0), 0.5, 1e-6);
}

test "nav curvature identity" {
    const testing = std.testing;
    // Same quaternion => angle = 0
    const q: [4]f64 = .{ 1.0, 0.0, 0.0, 0.0 };
    const curv = computeNavCurvature(q, q);
    try testing.expectApproxEqAbs(curv, 0.0, 1e-6);
}

test "nav curvature 180 rotation" {
    const testing = std.testing;
    // Rotation of pi around x-axis: q = [0, 1, 0, 0]
    const q1: [4]f64 = .{ 1.0, 0.0, 0.0, 0.0 };
    const q2: [4]f64 = .{ 0.0, 1.0, 0.0, 0.0 };
    const curv = computeNavCurvature(q2, q1);
    // Should be approximately pi
    try testing.expect(curv > 3.1 and curv < 3.2);
}

test "hurst trending series" {
    const testing = std.testing;
    // Monotonically increasing series => H should be > 0.5
    var closes: [64]f64 = undefined;
    for (0..64) |i| {
        closes[i] = @as(f64, @floatFromInt(i)) * 0.01 + 100.0;
    }
    var results: [64]f64 = undefined;
    computeHurstBatch(&closes, 32, &results);
    // Position 63 should have H > 0.5 (trending)
    try testing.expect(results[63] > 0.5);
}

test "hurst mean reverting series" {
    const testing = std.testing;
    // Alternating series => H should be < 0.5
    var closes: [64]f64 = undefined;
    for (0..64) |i| {
        closes[i] = 100.0 + (if (i % 2 == 0) @as(f64, 0.01) else @as(f64, -0.01));
    }
    var results: [64]f64 = undefined;
    computeHurstBatch(&closes, 32, &results);
    try testing.expect(results[63] < 0.55);
}
