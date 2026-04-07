//! stats.zig -- Online statistical estimators for hot-path quantitative analytics.
//!
//! All structs are comptime-generic over f32 or f64. Algorithms:
//!   - Welford one-pass mean/variance (numerically stable)
//!   - 4-moment Welford extension for skewness + kurtosis
//!   - EWMA / EWMAVariance (RiskMetrics lambda formulation)
//!   - Running median via dual-heap (max-heap lower half, min-heap upper half)
//!   - P-squared algorithm for online quantile estimation
//!   - Online Pearson correlation via running co-moment
//!   - Recursive least squares (online OLS) with forgetting factor

const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

// ---------------------------------------------------------------------------
// RunningMean
// ---------------------------------------------------------------------------

/// Online mean via Welford's algorithm. O(1) per update, numerically stable.
pub fn RunningMean(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) @compileError("RunningMean requires f32 or f64");
    }
    return struct {
        const Self = @This();

        n: u64 = 0,
        mean: T = 0,

        pub fn init() Self {
            return .{};
        }

        /// Add a new observation.
        pub fn update(self: *Self, x: T) void {
            self.n += 1;
            const delta = x - self.mean;
            self.mean += delta / @as(T, @floatFromInt(self.n));
        }

        /// Return current mean. Returns 0 if no observations.
        pub fn get(self: Self) T {
            return self.mean;
        }

        pub fn count(self: Self) u64 {
            return self.n;
        }

        pub fn reset(self: *Self) void {
            self.n = 0;
            self.mean = 0;
        }
    };
}

// ---------------------------------------------------------------------------
// RunningVariance  (Welford, 4-moment extension)
// ---------------------------------------------------------------------------

/// Online variance, std, skewness, excess kurtosis via Welford.
/// Uses the Terriberry/Chan extension for M3 and M4 accumulators.
pub fn RunningVariance(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) @compileError("RunningVariance requires f32 or f64");
    }
    return struct {
        const Self = @This();

        n: u64 = 0,
        mean: T = 0,
        m2: T = 0,  // sum of squared deviations
        m3: T = 0,  // sum of cubed deviations (for skewness)
        m4: T = 0,  // sum of 4th-power deviations (for kurtosis)

        pub fn init() Self {
            return .{};
        }

        pub fn update(self: *Self, x: T) void {
            const n1 = self.n;
            self.n += 1;
            const n: T = @floatFromInt(self.n);
            const delta = x - self.mean;
            const delta_n = delta / n;
            const delta_n2 = delta_n * delta_n;
            const term1 = delta * delta_n * @as(T, @floatFromInt(n1));

            self.mean += delta_n;
            // M4 must be updated before M3, M3 before M2 (use old values)
            self.m4 += term1 * delta_n2 * (n * n - @as(T, 3) * n + @as(T, 3)) +
                       @as(T, 6) * delta_n2 * self.m2 -
                       @as(T, 4) * delta_n * self.m3;
            self.m3 += term1 * delta_n * (n - @as(T, 2)) -
                       @as(T, 3) * delta_n * self.m2;
            self.m2 += term1;
        }

        pub fn mean(self: Self) T {
            return self.mean;
        }

        /// Population variance (divide by n).
        pub fn variancePop(self: Self) T {
            if (self.n < 2) return 0;
            return self.m2 / @as(T, @floatFromInt(self.n));
        }

        /// Sample variance (divide by n-1, Bessel's correction).
        pub fn varianceSample(self: Self) T {
            if (self.n < 2) return 0;
            return self.m2 / @as(T, @floatFromInt(self.n - 1));
        }

        /// Sample standard deviation.
        pub fn stddev(self: Self) T {
            return @sqrt(self.varianceSample());
        }

        /// Population standard deviation.
        pub fn stddevPop(self: Self) T {
            return @sqrt(self.variancePop());
        }

        /// Skewness (Fisher's moment coefficient, unbiased G1 estimator).
        pub fn skewness(self: Self) T {
            if (self.n < 3) return 0;
            const n: T = @floatFromInt(self.n);
            const m2 = self.m2 / n;
            const m3 = self.m3 / n;
            if (m2 == 0) return 0;
            return m3 / @sqrt(m2 * m2 * m2);
        }

        /// Excess kurtosis (Kurt - 3, so normal distribution returns 0).
        pub fn kurtosis(self: Self) T {
            if (self.n < 4) return 0;
            const n: T = @floatFromInt(self.n);
            const m2 = self.m2 / n;
            const m4 = self.m4 / n;
            if (m2 == 0) return 0;
            return (m4 / (m2 * m2)) - @as(T, 3);
        }

        pub fn count(self: Self) u64 {
            return self.n;
        }

        pub fn reset(self: *Self) void {
            self.n = 0;
            self.mean = 0;
            self.m2 = 0;
            self.m3 = 0;
            self.m4 = 0;
        }
    };
}

// ---------------------------------------------------------------------------
// EWMA
// ---------------------------------------------------------------------------

/// Exponential weighted moving average. Alpha is per-instance (stored as T).
/// Uses the standard recursive: ewma_t = alpha * x_t + (1-alpha) * ewma_{t-1}
pub fn EWMA(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) @compileError("EWMA requires f32 or f64");
    }
    return struct {
        const Self = @This();

        alpha: T,
        value: T = 0,
        initialized: bool = false,

        pub fn init(alpha: T) Self {
            assert(alpha > 0 and alpha <= 1);
            return .{ .alpha = alpha };
        }

        /// Convenience: init from span (alpha = 2 / (span + 1)).
        pub fn initFromSpan(span: u32) Self {
            const s: T = @floatFromInt(span);
            return Self.init(@as(T, 2) / (s + @as(T, 1)));
        }

        pub fn update(self: *Self, x: T) void {
            if (!self.initialized) {
                self.value = x;
                self.initialized = true;
            } else {
                self.value = self.alpha * x + (@as(T, 1) - self.alpha) * self.value;
            }
        }

        pub fn get(self: Self) T {
            return self.value;
        }

        pub fn reset(self: *Self) void {
            self.value = 0;
            self.initialized = false;
        }
    };
}

// ---------------------------------------------------------------------------
// EWMVariance (RiskMetrics-style)
// ---------------------------------------------------------------------------

/// EWMA variance estimator using RiskMetrics lambda formulation.
/// sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
/// Lambda controls the decay; typical value 0.94 (daily) or 0.97 (monthly).
pub fn EWMVariance(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) @compileError("EWMVariance requires f32 or f64");
    }
    return struct {
        const Self = @This();

        lambda: T,
        variance: T = 0,
        prev_return: T = 0,
        initialized: bool = false,

        pub fn init(lambda: T) Self {
            assert(lambda > 0 and lambda < 1);
            return .{ .lambda = lambda };
        }

        /// Feed the current log-return (or simple return). The variance is
        /// the variance of the NEXT period, estimated from current data.
        pub fn update(self: *Self, ret: T) void {
            if (!self.initialized) {
                self.variance = ret * ret;
                self.prev_return = ret;
                self.initialized = true;
            } else {
                self.variance = self.lambda * self.variance +
                    (@as(T, 1) - self.lambda) * self.prev_return * self.prev_return;
                self.prev_return = ret;
            }
        }

        pub fn get(self: Self) T {
            return self.variance;
        }

        pub fn volatility(self: Self) T {
            return @sqrt(self.variance);
        }

        pub fn reset(self: *Self) void {
            self.variance = 0;
            self.prev_return = 0;
            self.initialized = false;
        }
    };
}

// ---------------------------------------------------------------------------
// RunningMedian  (dual-heap: max-heap for lower half, min-heap for upper half)
// ---------------------------------------------------------------------------

/// Incremental median via two-heap structure.
/// Allocates heap memory; caller must call deinit().
pub fn RunningMedian(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) @compileError("RunningMedian requires f32 or f64");
    }
    return struct {
        const Self = @This();

        // lower half: max-heap (negate values for std min-heap)
        lower: std.ArrayList(T),
        // upper half: min-heap
        upper: std.ArrayList(T),
        alloc: Allocator,

        pub fn init(alloc: Allocator) Self {
            return .{
                .lower = std.ArrayList(T).init(alloc),
                .upper = std.ArrayList(T).init(alloc),
                .alloc = alloc,
            };
        }

        pub fn deinit(self: *Self) void {
            self.lower.deinit();
            self.upper.deinit();
        }

        fn siftUp(heap: []T, idx: usize, comptime less: fn (T, T) bool) void {
            var i = idx;
            while (i > 0) {
                const parent = (i - 1) / 2;
                if (less(heap[i], heap[parent])) {
                    const tmp = heap[i];
                    heap[i] = heap[parent];
                    heap[parent] = tmp;
                    i = parent;
                } else break;
            }
        }

        fn siftDown(heap: []T, idx: usize, comptime less: fn (T, T) bool) void {
            const n = heap.len;
            var i = idx;
            while (true) {
                var smallest = i;
                const l = 2 * i + 1;
                const r = 2 * i + 2;
                if (l < n and less(heap[l], heap[smallest])) smallest = l;
                if (r < n and less(heap[r], heap[smallest])) smallest = r;
                if (smallest == i) break;
                const tmp = heap[i];
                heap[i] = heap[smallest];
                heap[smallest] = tmp;
                i = smallest;
            }
        }

        fn ltF(a: T, b: T) bool { return a < b; }
        fn gtF(a: T, b: T) bool { return a > b; }

        fn pushLower(self: *Self, x: T) !void {
            // lower is a max-heap; store as positive, compare as gt
            try self.lower.append(x);
            siftUp(self.lower.items, self.lower.items.len - 1, gtF);
        }

        fn pushUpper(self: *Self, x: T) !void {
            try self.upper.append(x);
            siftUp(self.upper.items, self.upper.items.len - 1, ltF);
        }

        fn popLower(self: *Self) T {
            const items = self.lower.items;
            const top = items[0];
            const last = self.lower.pop();
            if (self.lower.items.len > 0) {
                self.lower.items[0] = last;
                siftDown(self.lower.items, 0, gtF);
            }
            return top;
        }

        fn popUpper(self: *Self) T {
            const items = self.upper.items;
            const top = items[0];
            const last = self.upper.pop();
            if (self.upper.items.len > 0) {
                self.upper.items[0] = last;
                siftDown(self.upper.items, 0, ltF);
            }
            return top;
        }

        pub fn update(self: *Self, x: T) !void {
            // Route into correct heap
            if (self.lower.items.len == 0 or x <= self.lower.items[0]) {
                try self.pushLower(x);
            } else {
                try self.pushUpper(x);
            }
            // Rebalance: lower may have at most 1 more element than upper
            const ll = self.lower.items.len;
            const ul = self.upper.items.len;
            if (ll > ul + 1) {
                try self.pushUpper(self.popLower());
            } else if (ul > ll) {
                try self.pushLower(self.popUpper());
            }
        }

        pub fn get(self: Self) T {
            const ll = self.lower.items.len;
            const ul = self.upper.items.len;
            if (ll == 0) return 0;
            if (ll == ul) {
                return (self.lower.items[0] + self.upper.items[0]) / @as(T, 2);
            } else {
                return self.lower.items[0];
            }
        }

        pub fn count(self: Self) usize {
            return self.lower.items.len + self.upper.items.len;
        }

        pub fn reset(self: *Self) void {
            self.lower.clearRetainingCapacity();
            self.upper.clearRetainingCapacity();
        }
    };
}

// ---------------------------------------------------------------------------
// RollingQuantile  (P-squared algorithm, Jain & Chlamtac 1985)
// ---------------------------------------------------------------------------

/// Online quantile estimator using the P-squared algorithm.
/// Tracks a single quantile p in [0,1] without storing all data points.
pub fn RollingQuantile(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) @compileError("RollingQuantile requires f32 or f64");
    }
    return struct {
        const Self = @This();
        const N_MARKERS: usize = 5;

        p: T,                         // desired quantile
        q: [N_MARKERS]T = undefined,  // marker heights
        n: [N_MARKERS]i64 = undefined, // marker positions
        dn: [N_MARKERS]T = undefined, // desired marker positions
        count: usize = 0,

        pub fn init(p: T) Self {
            assert(p > 0 and p < 1);
            var s = Self{ .p = p };
            s.dn = .{ 0, p / 2, p, (1 + p) / 2, 1 };
            return s;
        }

        fn parabolic(self: Self, i: usize, d: i64) T {
            const qi = self.q[i];
            const qi1 = self.q[i + 1];
            const qi_1 = self.q[i - 1];
            const ni: T = @floatFromInt(self.n[i]);
            const ni1: T = @floatFromInt(self.n[i + 1]);
            const ni_1: T = @floatFromInt(self.n[i - 1]);
            const di: T = @floatFromInt(d);
            return qi + di / (ni1 - ni_1) * (
                (ni - ni_1 + di) * (qi1 - qi) / (ni1 - ni) +
                (ni1 - ni - di) * (qi - qi_1) / (ni - ni_1)
            );
        }

        pub fn update(self: *Self, x: T) void {
            const cnt = self.count;
            self.count += 1;

            if (cnt < N_MARKERS) {
                self.q[cnt] = x;
                if (cnt == N_MARKERS - 1) {
                    // Sort the first 5 observations and set initial positions
                    sortFirst5(&self.q);
                    self.n = .{ 1, 2, 3, 4, 5 };
                }
                return;
            }

            // Find cell k where x falls
            var k: usize = 0;
            if (x < self.q[0]) {
                self.q[0] = x;
                k = 0;
            } else if (x < self.q[1]) {
                k = 0;
            } else if (x < self.q[2]) {
                k = 1;
            } else if (x < self.q[3]) {
                k = 2;
            } else if (x <= self.q[4]) {
                k = 3;
            } else {
                self.q[4] = x;
                k = 3;
            }

            // Increment positions of markers k+1 through 4
            for (k + 1..N_MARKERS) |i| self.n[i] += 1;

            // Update desired positions
            const m: T = @floatFromInt(self.count);
            _ = m;
            const nf: T = @floatFromInt(self.count);
            self.dn = .{
                0,
                (nf - 1) * self.p / 2,
                (nf - 1) * self.p,
                (nf - 1) * (1 + self.p) / 2,
                nf - 1,
            };

            // Adjust marker heights
            for (1..4) |i| {
                const d = self.dn[i] - @as(T, @floatFromInt(self.n[i]));
                if ((d >= 1 and self.n[i + 1] - self.n[i] > 1) or
                    (d <= -1 and self.n[i - 1] - self.n[i] < -1))
                {
                    const sign: i64 = if (d > 0) 1 else -1;
                    const qp = self.parabolic(i, sign);
                    if (qp > self.q[i - 1] and qp < self.q[i + 1]) {
                        self.q[i] = qp;
                    } else {
                        // Linear interpolation fallback
                        const fs: T = @floatFromInt(sign);
                        const idx: usize = if (sign > 0) i + 1 else i - 1;
                        const ni: T = @floatFromInt(self.n[i]);
                        const nidx: T = @floatFromInt(self.n[idx]);
                        self.q[i] += fs * (self.q[idx] - self.q[i]) / (nidx - ni);
                    }
                    self.n[i] += sign;
                }
            }
        }

        fn sortFirst5(arr: *[5]T) void {
            // Simple insertion sort for 5 elements
            var i: usize = 1;
            while (i < 5) : (i += 1) {
                const key = arr[i];
                var j: usize = i;
                while (j > 0 and arr[j - 1] > key) : (j -= 1) {
                    arr[j] = arr[j - 1];
                }
                arr[j] = key;
            }
        }

        /// Returns the estimated quantile. Valid only after >= 5 observations.
        pub fn get(self: Self) T {
            if (self.count < N_MARKERS) return 0;
            return self.q[2];
        }

        pub fn reset(self: *Self) void {
            self.count = 0;
        }
    };
}

// ---------------------------------------------------------------------------
// RunningCorrelation  (Pearson, online co-moment method)
// ---------------------------------------------------------------------------

/// Online Pearson correlation coefficient via Welford's running covariance.
pub fn RunningCorrelation(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) @compileError("RunningCorrelation requires f32 or f64");
    }
    return struct {
        const Self = @This();

        n: u64 = 0,
        mean_x: T = 0,
        mean_y: T = 0,
        var_x: T = 0,   // M2 for x
        var_y: T = 0,   // M2 for y
        cov_xy: T = 0,  // running co-moment (sum of cross-deviations)

        pub fn init() Self {
            return .{};
        }

        pub fn update(self: *Self, x: T, y: T) void {
            self.n += 1;
            const n: T = @floatFromInt(self.n);
            const dx = x - self.mean_x;
            const dy = y - self.mean_y;
            self.mean_x += dx / n;
            self.mean_y += dy / n;
            self.var_x += dx * (x - self.mean_x);
            self.var_y += dy * (y - self.mean_y);
            self.cov_xy += dx * (y - self.mean_y);
        }

        /// Returns Pearson r in [-1, 1]. Returns 0 if insufficient data.
        pub fn get(self: Self) T {
            if (self.n < 2) return 0;
            const denom = @sqrt(self.var_x * self.var_y);
            if (denom == 0) return 0;
            return self.cov_xy / denom;
        }

        /// Sample covariance.
        pub fn covariance(self: Self) T {
            if (self.n < 2) return 0;
            return self.cov_xy / @as(T, @floatFromInt(self.n - 1));
        }

        pub fn count(self: Self) u64 {
            return self.n;
        }

        pub fn reset(self: *Self) void {
            self.n = 0;
            self.mean_x = 0;
            self.mean_y = 0;
            self.var_x = 0;
            self.var_y = 0;
            self.cov_xy = 0;
        }
    };
}

// ---------------------------------------------------------------------------
// LinearRegression  (recursive least squares, forgetting factor lambda)
// ---------------------------------------------------------------------------

/// Online OLS via recursive least squares with forgetting factor.
/// Fits y = beta0 + beta1 * x where x is a simple counter (1, 2, 3, ...).
/// Forgetting factor lambda < 1 discounts older observations exponentially.
pub fn LinearRegression(comptime T: type) type {
    comptime {
        if (T != f32 and T != f64) @compileError("LinearRegression requires f32 or f64");
    }
    return struct {
        const Self = @This();

        lambda: T,
        // theta = [beta0, beta1]
        theta: [2]T = .{ 0, 0 },
        // P = gain matrix (2x2 stored as [P00, P01, P10, P11])
        p: [4]T = .{ 1e6, 0, 0, 1e6 },
        n: u64 = 0,

        pub fn init(lambda: T) Self {
            assert(lambda > 0 and lambda <= 1);
            return .{ .lambda = lambda };
        }

        pub fn initDefault() Self {
            return Self.init(0.99);
        }

        pub fn update(self: *Self, y: T) void {
            self.n += 1;
            const x1: T = @floatFromInt(self.n);
            const phi: [2]T = .{ 1, x1 };

            // Error = y - phi' * theta
            const y_hat = phi[0] * self.theta[0] + phi[1] * self.theta[1];
            const e = y - y_hat;

            // P * phi
            const pp0 = self.p[0] * phi[0] + self.p[1] * phi[1];
            const pp1 = self.p[2] * phi[0] + self.p[3] * phi[1];

            // phi' * P * phi + lambda
            const denom = self.lambda + phi[0] * pp0 + phi[1] * pp1;

            // Kalman gain K = P * phi / denom
            const k0 = pp0 / denom;
            const k1 = pp1 / denom;

            // Update theta
            self.theta[0] += k0 * e;
            self.theta[1] += k1 * e;

            // Update P: P = (P - K * phi' * P) / lambda
            const kp00 = k0 * pp0;
            const kp01 = k0 * pp1;
            const kp10 = k1 * pp0;
            const kp11 = k1 * pp1;
            self.p[0] = (self.p[0] - kp00) / self.lambda;
            self.p[1] = (self.p[1] - kp01) / self.lambda;
            self.p[2] = (self.p[2] - kp10) / self.lambda;
            self.p[3] = (self.p[3] - kp11) / self.lambda;
        }

        /// Returns [intercept, slope].
        pub fn coefficients(self: Self) [2]T {
            return self.theta;
        }

        /// Predict at time step n+1 (next step forecast).
        pub fn predict(self: Self) T {
            const x_next: T = @floatFromInt(self.n + 1);
            return self.theta[0] + self.theta[1] * x_next;
        }

        /// Slope (trend direction and magnitude).
        pub fn slope(self: Self) T {
            return self.theta[1];
        }

        pub fn count(self: Self) u64 {
            return self.n;
        }

        pub fn reset(self: *Self) void {
            self.theta = .{ 0, 0 };
            self.p = .{ 1e6, 0, 0, 1e6 };
            self.n = 0;
        }
    };
}

// ---------------------------------------------------------------------------
// Python ctypes-compatible C ABI exports
// ---------------------------------------------------------------------------
// These are bare C functions that Python can call via ctypes after loading
// the shared library produced by the stats build target.

// ---- EWMA exports (f64) ----

const EWMAf64 = EWMA(f64);

export fn stats_ewma_new(alpha: f64) *EWMAf64 {
    const alloc = std.heap.c_allocator;
    const p = alloc.create(EWMAf64) catch unreachable;
    p.* = EWMAf64.init(alpha);
    return p;
}

export fn stats_ewma_update(ptr: *EWMAf64, x: f64) void {
    ptr.update(x);
}

export fn stats_ewma_get(ptr: *const EWMAf64) f64 {
    return ptr.get();
}

export fn stats_ewma_free(ptr: *EWMAf64) void {
    std.heap.c_allocator.destroy(ptr);
}

// ---- EWMVariance exports (f64) ----

const EWMVarf64 = EWMVariance(f64);

export fn stats_ewmvar_new(lambda: f64) *EWMVarf64 {
    const alloc = std.heap.c_allocator;
    const p = alloc.create(EWMVarf64) catch unreachable;
    p.* = EWMVarf64.init(lambda);
    return p;
}

export fn stats_ewmvar_update(ptr: *EWMVarf64, ret: f64) void {
    ptr.update(ret);
}

export fn stats_variance_get(ptr: *const EWMVarf64) f64 {
    return ptr.get();
}

export fn stats_ewmvar_volatility(ptr: *const EWMVarf64) f64 {
    return ptr.volatility();
}

export fn stats_ewmvar_free(ptr: *EWMVarf64) void {
    std.heap.c_allocator.destroy(ptr);
}

// ---- RunningVariance exports (f64) ----

const RunVarf64 = RunningVariance(f64);

export fn stats_runvar_new() *RunVarf64 {
    const alloc = std.heap.c_allocator;
    const p = alloc.create(RunVarf64) catch unreachable;
    p.* = RunVarf64.init();
    return p;
}

export fn stats_runvar_update(ptr: *RunVarf64, x: f64) void {
    ptr.update(x);
}

export fn stats_runvar_mean(ptr: *const RunVarf64) f64 {
    return ptr.mean;
}

export fn stats_runvar_variance(ptr: *const RunVarf64) f64 {
    return ptr.varianceSample();
}

export fn stats_runvar_stddev(ptr: *const RunVarf64) f64 {
    return ptr.stddev();
}

export fn stats_runvar_skewness(ptr: *const RunVarf64) f64 {
    return ptr.skewness();
}

export fn stats_runvar_kurtosis(ptr: *const RunVarf64) f64 {
    return ptr.kurtosis();
}

export fn stats_runvar_free(ptr: *RunVarf64) void {
    std.heap.c_allocator.destroy(ptr);
}

// ---- RunningCorrelation exports (f64) ----

const RunCorrf64 = RunningCorrelation(f64);

export fn stats_corr_new() *RunCorrf64 {
    const alloc = std.heap.c_allocator;
    const p = alloc.create(RunCorrf64) catch unreachable;
    p.* = RunCorrf64.init();
    return p;
}

export fn stats_corr_update(ptr: *RunCorrf64, x: f64, y: f64) void {
    ptr.update(x, y);
}

export fn stats_correlation_get(ptr: *const RunCorrf64) f64 {
    return ptr.get();
}

export fn stats_corr_free(ptr: *RunCorrf64) void {
    std.heap.c_allocator.destroy(ptr);
}

// ---- LinearRegression exports (f64) ----

const LinRegf64 = LinearRegression(f64);

export fn stats_linreg_new(lambda: f64) *LinRegf64 {
    const alloc = std.heap.c_allocator;
    const p = alloc.create(LinRegf64) catch unreachable;
    p.* = LinRegf64.init(lambda);
    return p;
}

export fn stats_linreg_update(ptr: *LinRegf64, y: f64) void {
    ptr.update(y);
}

export fn stats_linreg_slope(ptr: *const LinRegf64) f64 {
    return ptr.slope();
}

export fn stats_linreg_predict(ptr: *const LinRegf64) f64 {
    return ptr.predict();
}

export fn stats_linreg_free(ptr: *LinRegf64) void {
    std.heap.c_allocator.destroy(ptr);
}
