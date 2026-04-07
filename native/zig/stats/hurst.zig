//! hurst.zig -- Rolling Hurst exponent estimator via R/S analysis.
//!
//! Uses a circular buffer of log-prices with comptime capacity (must be power of 2).
//! Computes R/S statistic at sub-window sizes [n/8, n/4, n/2, n], then fits
//! H via OLS on log(R/S) vs log(n).
//!
//! SIMD is used for the mean-deviation accumulation pass where available.
//! The circular buffer uses a power-of-2 mask for O(1) indexing.

const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const builtin = @import("builtin");

/// Check if the target has AVX2 (x86_64 with AVX2 feature).
const has_avx2 = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

// ---------------------------------------------------------------------------
// Circular buffer of log-prices
// ---------------------------------------------------------------------------

/// Circular buffer with power-of-2 capacity for branchless modular indexing.
fn CircularBuffer(comptime T: type, comptime cap: usize) type {
    comptime {
        if (!math.isPowerOfTwo(cap)) @compileError("CircularBuffer capacity must be a power of 2");
        if (T != f32 and T != f64) @compileError("CircularBuffer requires f32 or f64");
    }
    return struct {
        const Self = @This();
        const MASK = cap - 1;

        buf: [cap]T = undefined,
        head: usize = 0,  // next write position
        full: bool = false,

        pub fn push(self: *Self, val: T) void {
            self.buf[self.head] = val;
            self.head = (self.head + 1) & MASK;
            if (self.head == 0) self.full = true;
        }

        pub fn len(self: Self) usize {
            if (self.full) return cap;
            return self.head;
        }

        /// Access element i=0 (oldest) to i=len-1 (newest), in logical order.
        pub fn get(self: Self, i: usize) T {
            const l = self.len();
            assert(i < l);
            if (self.full) {
                return self.buf[(self.head + i) & MASK];
            } else {
                return self.buf[i];
            }
        }

        pub fn isFull(self: Self) bool {
            return self.full;
        }
    };
}

// ---------------------------------------------------------------------------
// R/S computation helpers
// ---------------------------------------------------------------------------

/// Compute R/S statistic for a slice of log-prices using a scratch buffer.
/// scratch must be at least the same length as prices.
fn computeRS(prices: []const f64, scratch: []f64) f64 {
    const n = prices.len;
    if (n < 2) return 0;

    // Step 1: compute mean log-return
    var mean: f64 = 0;
    for (0..n - 1) |i| {
        scratch[i] = prices[i + 1] - prices[i]; // log-return
        mean += scratch[i];
    }
    const m = n - 1;
    mean /= @floatFromInt(m);

    // Step 2: mean-adjusted series
    // Also compute std deviation of returns
    var var_acc: f64 = 0;
    for (0..m) |i| {
        const d = scratch[i] - mean;
        scratch[i] = d;
        var_acc += d * d;
    }
    const std_dev = @sqrt(var_acc / @as(f64, @floatFromInt(m)));
    if (std_dev == 0) return 1;

    // Step 3: cumulative sum (profile)
    var cum: f64 = 0;
    var max_cum: f64 = -math.floatMax(f64);
    var min_cum: f64 = math.floatMax(f64);
    for (0..m) |i| {
        cum += scratch[i];
        if (cum > max_cum) max_cum = cum;
        if (cum < min_cum) min_cum = cum;
    }

    const range = max_cum - min_cum;
    return range / std_dev;
}

/// SIMD-accelerated version of cumulative deviation for AVX2 targets.
/// Falls back to scalar on non-x86_64 or non-AVX2 builds.
fn computeMeanDeviation(prices: []const f64, out_returns: []f64) f64 {
    const n = prices.len;
    const m = n - 1;
    var mean: f64 = 0;

    if (has_avx2 and m >= 4) {
        // Compute returns and accumulate mean using 4-wide SIMD
        const Vec4 = @Vector(4, f64);
        var sum_vec: Vec4 = @splat(0);
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            const a: Vec4 = .{
                prices[i],     prices[i + 1],
                prices[i + 2], prices[i + 3],
            };
            const b: Vec4 = .{
                prices[i + 1], prices[i + 2],
                prices[i + 3], prices[i + 4],
            };
            const r = b - a;
            out_returns[i]     = r[0];
            out_returns[i + 1] = r[1];
            out_returns[i + 2] = r[2];
            out_returns[i + 3] = r[3];
            sum_vec += r;
        }
        mean = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
        // Handle tail
        while (i < m) : (i += 1) {
            const r = prices[i + 1] - prices[i];
            out_returns[i] = r;
            mean += r;
        }
    } else {
        for (0..m) |i| {
            out_returns[i] = prices[i + 1] - prices[i];
            mean += out_returns[i];
        }
    }

    return mean / @as(f64, @floatFromInt(m));
}

/// Full SIMD-accelerated R/S computation.
fn computeRSSimd(prices: []const f64, scratch: []f64) f64 {
    const n = prices.len;
    if (n < 2) return 0;
    const m = n - 1;

    const mean = computeMeanDeviation(prices, scratch);

    // Subtract mean, accumulate variance
    var var_acc: f64 = 0;
    if (has_avx2 and m >= 4) {
        const Vec4 = @Vector(4, f64);
        const mean_vec: Vec4 = @splat(mean);
        var var_vec: Vec4 = @splat(0.0);
        var i: usize = 0;
        while (i + 4 <= m) : (i += 4) {
            const r: Vec4 = .{ scratch[i], scratch[i+1], scratch[i+2], scratch[i+3] };
            const d = r - mean_vec;
            scratch[i]   = d[0];
            scratch[i+1] = d[1];
            scratch[i+2] = d[2];
            scratch[i+3] = d[3];
            var_vec += d * d;
        }
        var_acc = var_vec[0] + var_vec[1] + var_vec[2] + var_vec[3];
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
    if (std_dev == 0) return 1;

    // Cumulative sum for range
    var cum: f64 = 0;
    var max_cum: f64 = -math.floatMax(f64);
    var min_cum: f64 = math.floatMax(f64);
    for (0..m) |i| {
        cum += scratch[i];
        if (cum > max_cum) max_cum = cum;
        if (cum < min_cum) min_cum = cum;
    }

    return (max_cum - min_cum) / std_dev;
}

// ---------------------------------------------------------------------------
// OLS for H: fit log(R/S) = H * log(n) + const
// ---------------------------------------------------------------------------

/// Fit slope of log(y) vs log(x) for up to 4 points via simple OLS.
fn fitOLS(log_x: []const f64, log_y: []const f64) f64 {
    const n_pts = log_x.len;
    assert(n_pts == log_y.len);
    if (n_pts < 2) return 0.5;

    var sum_x: f64 = 0;
    var sum_y: f64 = 0;
    var sum_xx: f64 = 0;
    var sum_xy: f64 = 0;

    for (0..n_pts) |i| {
        sum_x  += log_x[i];
        sum_y  += log_y[i];
        sum_xx += log_x[i] * log_x[i];
        sum_xy += log_x[i] * log_y[i];
    }

    const nf: f64 = @floatFromInt(n_pts);
    const denom = nf * sum_xx - sum_x * sum_x;
    if (@abs(denom) < 1e-14) return 0.5;
    return (nf * sum_xy - sum_x * sum_y) / denom;
}

// ---------------------------------------------------------------------------
// HurstEstimator
// ---------------------------------------------------------------------------

/// Rolling Hurst exponent estimator.
/// WINDOW must be a power of 2. Typical value: 512 or 1024.
pub fn HurstEstimator(comptime WINDOW: usize) type {
    comptime {
        if (!math.isPowerOfTwo(WINDOW)) @compileError("HurstEstimator WINDOW must be power of 2");
        if (WINDOW < 16) @compileError("HurstEstimator WINDOW must be >= 16");
    }
    return struct {
        const Self = @This();
        const BUF = CircularBuffer(f64, WINDOW);

        buf: BUF = .{},
        /// Current estimated Hurst exponent. Defaults to 0.5 until enough data.
        h: f64 = 0.5,
        /// Scratch buffer for R/S computations (size = WINDOW).
        scratch: [WINDOW]f64 = undefined,
        /// Temporary linear price buffer used to extract contiguous windows.
        linear: [WINDOW]f64 = undefined,

        pub fn init() Self {
            return .{};
        }

        /// Feed the next log-price (use @log(close)).
        pub fn update(self: *Self, log_price: f64) void {
            self.buf.push(log_price);
            const n = self.buf.len();
            if (n < 16) return;

            // Extract current window into linear buffer
            const use_n = n;
            for (0..use_n) |i| {
                self.linear[i] = self.buf.get(i);
            }

            self.h = self.computeHurst(self.linear[0..use_n]);
        }

        fn computeHurst(self: *Self, prices: []f64) f64 {
            const n = prices.len;
            // Sub-window sizes: n/8, n/4, n/2, n (capped at n)
            const sub_sizes = [4]usize{
                @max(8, n / 8),
                @max(8, n / 4),
                @max(8, n / 2),
                n,
            };

            var log_ns: [4]f64 = undefined;
            var log_rs: [4]f64 = undefined;
            var valid: usize = 0;

            for (sub_sizes) |sub_n| {
                if (sub_n < 8 or sub_n > n) continue;
                // Use a suffix of length sub_n (most recent data)
                const start = n - sub_n;
                const rs = computeRSSimd(prices[start..n], &self.scratch);
                if (rs > 0) {
                    log_ns[valid] = @log(@as(f64, @floatFromInt(sub_n)));
                    log_rs[valid] = @log(rs);
                    valid += 1;
                }
            }

            if (valid < 2) return 0.5;
            const h = fitOLS(log_ns[0..valid], log_rs[0..valid]);
            // Clamp to [0.01, 0.99] -- values outside are numerically suspect
            return @max(0.01, @min(0.99, h));
        }

        /// Returns the current Hurst exponent estimate.
        pub fn get(self: Self) f64 {
            return self.h;
        }

        /// Returns true if the series appears to be trending (H > 0.55).
        pub fn isTrending(self: Self) bool {
            return self.h > 0.55;
        }

        /// Returns true if the series appears to be mean-reverting (H < 0.45).
        pub fn isMeanReverting(self: Self) bool {
            return self.h < 0.45;
        }

        pub fn sampleCount(self: Self) usize {
            return self.buf.len();
        }
    };
}

// ---------------------------------------------------------------------------
// Python ctypes-compatible C ABI exports
// ---------------------------------------------------------------------------
// We use a default window of 512 for the exported C interface.

const DefaultWindow: usize = 512;
const DefaultEstimator = HurstEstimator(DefaultWindow);

export fn hurst_new(window: i32) *DefaultEstimator {
    _ = window; // The comptime window is 512; runtime param is accepted but unused
    const alloc = std.heap.c_allocator;
    const p = alloc.create(DefaultEstimator) catch unreachable;
    p.* = DefaultEstimator.init();
    return p;
}

export fn hurst_update(ptr: *DefaultEstimator, log_price: f64) void {
    ptr.update(log_price);
}

export fn hurst_get(ptr: *const DefaultEstimator) f64 {
    return ptr.get();
}

export fn hurst_free(ptr: *DefaultEstimator) void {
    std.heap.c_allocator.destroy(ptr);
}

export fn hurst_is_trending(ptr: *const DefaultEstimator) bool {
    return ptr.isTrending();
}

export fn hurst_is_mean_reverting(ptr: *const DefaultEstimator) bool {
    return ptr.isMeanReverting();
}

export fn hurst_sample_count(ptr: *const DefaultEstimator) usize {
    return ptr.sampleCount();
}
