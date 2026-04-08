const std = @import("std");
const math = std.math;

// ============================================================================
// Constants
// ============================================================================

const SPEED_OF_LIGHT: f64 = 1.0; // normalized c=1 for Minkowski metric
const SQRT_2PI: f64 = 2.5066282746310002;
const INV_SQRT_2PI: f64 = 0.3989422804014327;

// ============================================================================
// MinkowskiClassifier: ds^2 = c^2*dt^2 - dx^2
// ============================================================================

pub const SpacetimeClass = enum(u8) {
    timelike, // ds^2 > 0: causal, signal propagates
    spacelike, // ds^2 < 0: acausal, decorrelation
    lightlike, // ds^2 = 0: boundary
};

pub const MinkowskiClassifier = struct {
    c: f64, // effective speed of light (price sensitivity)
    threshold: f64, // lightlike tolerance band
    prev_price: f64,
    prev_time: f64,
    ds2: f64,
    classification: SpacetimeClass,
    count: u64,

    pub fn init(c: f64, threshold: f64) MinkowskiClassifier {
        return .{
            .c = c,
            .threshold = threshold,
            .prev_price = 0,
            .prev_time = 0,
            .ds2 = 0,
            .classification = .lightlike,
            .count = 0,
        };
    }

    pub fn update(self: *MinkowskiClassifier, price: f64, time: f64) SpacetimeClass {
        if (self.count == 0) {
            self.prev_price = price;
            self.prev_time = time;
            self.count = 1;
            return .lightlike;
        }

        const dt = time - self.prev_time;
        const dx = price - self.prev_price;
        self.ds2 = self.c * self.c * dt * dt - dx * dx;

        self.classification = if (self.ds2 > self.threshold)
            .timelike
        else if (self.ds2 < -self.threshold)
            .spacelike
        else
            .lightlike;

        self.prev_price = price;
        self.prev_time = time;
        self.count += 1;
        return self.classification;
    }

    pub fn metricInterval(self: *const MinkowskiClassifier) f64 {
        return self.ds2;
    }

    pub fn properTime(self: *const MinkowskiClassifier) f64 {
        if (self.ds2 > 0) return @sqrt(self.ds2);
        return 0;
    }
};

// ============================================================================
// BHState: Black hole mass accumulation model
// ============================================================================

pub const BHState = struct {
    mass: f64,
    accrual_rate: f64,
    decay_rate: f64,
    spacelike_decay: f64,
    hawking_rate: f64, // slow radiation
    event_horizon: f64,
    mass_history: [256]f64,
    hist_idx: u8,
    count: u64,

    pub fn init() BHState {
        return .{
            .mass = 1.0,
            .accrual_rate = 0.03,
            .decay_rate = 0.97,
            .spacelike_decay = 0.924,
            .hawking_rate = 0.001,
            .event_horizon = 0,
            .mass_history = [_]f64{0} ** 256,
            .hist_idx = 0,
            .count = 0,
        };
    }

    pub fn initCustom(accrual: f64, decay: f64, spacelike: f64, hawking: f64) BHState {
        var s = BHState.init();
        s.accrual_rate = accrual;
        s.decay_rate = decay;
        s.spacelike_decay = spacelike;
        s.hawking_rate = hawking;
        return s;
    }

    pub fn update(self: *BHState, classification: SpacetimeClass, accrual_input: f64) f64 {
        switch (classification) {
            .timelike => {
                // Mass accumulation: momentum following regime
                self.mass = self.decay_rate * self.mass + self.accrual_rate * @abs(accrual_input);
            },
            .spacelike => {
                // Mass decay: mean-reverting regime, decorrelation
                self.mass *= self.spacelike_decay;
            },
            .lightlike => {
                // Boundary: slow Hawking radiation
                self.mass *= (1.0 - self.hawking_rate);
            },
        }

        // Schwarzschild radius proportional to mass
        self.event_horizon = 2.0 * self.mass; // r_s = 2GM/c^2, G=c=1

        self.mass_history[self.hist_idx] = self.mass;
        self.hist_idx +%= 1;
        self.count += 1;

        return self.mass;
    }

    pub fn signal(self: *const BHState) f64 {
        // Signal strength based on mass rate of change
        if (self.count < 2) return 0;
        const prev_idx = self.hist_idx -% 2;
        const prev_mass = self.mass_history[prev_idx];
        if (prev_mass == 0) return 0;
        return (self.mass - prev_mass) / prev_mass;
    }

    pub fn isAccreting(self: *const BHState) bool {
        return self.signal() > 0;
    }

    pub fn massPercentile(self: *const BHState) f64 {
        // Rank current mass vs history
        var below: u32 = 0;
        var total: u32 = 0;
        const n: u32 = @intCast(@min(self.count, 256));
        for (0..n) |i| {
            if (self.mass_history[i] > 0) {
                total += 1;
                if (self.mass_history[i] < self.mass) below += 1;
            }
        }
        if (total == 0) return 0.5;
        return @as(f64, @floatFromInt(below)) / @as(f64, @floatFromInt(total));
    }
};

// ============================================================================
// GARCHTracker: GARCH(1,1) conditional variance
// ============================================================================

pub const GARCHTracker = struct {
    omega: f64, // long-run variance weight
    alpha: f64, // ARCH coefficient (innovation)
    beta: f64, // GARCH coefficient (persistence)
    sigma2: f64, // conditional variance
    long_run_var: f64,
    prev_return: f64,
    count: u64,
    vol_forecast: [10]f64, // multi-step forecast

    pub fn init(omega: f64, alpha: f64, beta: f64) GARCHTracker {
        const lr_var = if (1.0 - alpha - beta > 0.001) omega / (1.0 - alpha - beta) else omega * 100.0;
        return .{
            .omega = omega,
            .alpha = alpha,
            .beta = beta,
            .sigma2 = lr_var,
            .long_run_var = lr_var,
            .prev_return = 0,
            .count = 0,
            .vol_forecast = [_]f64{0} ** 10,
        };
    }

    pub fn initDefault() GARCHTracker {
        return GARCHTracker.init(0.000002, 0.09, 0.90);
    }

    pub fn update(self: *GARCHTracker, log_return: f64) f64 {
        if (self.count == 0) {
            self.sigma2 = log_return * log_return;
            if (self.sigma2 < 1e-12) self.sigma2 = self.long_run_var;
        } else {
            self.sigma2 = self.omega + self.alpha * self.prev_return * self.prev_return + self.beta * self.sigma2;
        }
        self.prev_return = log_return;
        self.count += 1;

        // Multi-step forecast: h(t+k) = omega/(1-alpha-beta) + (alpha+beta)^k * (sigma2 - omega/(1-alpha-beta))
        const persistence = self.alpha + self.beta;
        var pk: f64 = 1.0;
        for (0..10) |k| {
            pk *= persistence;
            self.vol_forecast[k] = self.long_run_var + pk * (self.sigma2 - self.long_run_var);
        }

        return self.sigma2;
    }

    pub fn volatility(self: *const GARCHTracker) f64 {
        return @sqrt(self.sigma2);
    }

    pub fn annualizedVol(self: *const GARCHTracker) f64 {
        return @sqrt(self.sigma2 * 252.0);
    }

    pub fn forecastVol(self: *const GARCHTracker, steps: usize) f64 {
        if (steps == 0) return self.volatility();
        const idx = @min(steps - 1, 9);
        return @sqrt(self.vol_forecast[idx]);
    }

    pub fn volOfVol(self: *const GARCHTracker) f64 {
        // Kurtosis-based vol-of-vol estimate
        return self.alpha * @sqrt(2.0);
    }
};

// ============================================================================
// OUDetector: Ornstein-Uhlenbeck mean reversion
// ============================================================================

pub const OUDetector = struct {
    theta: f64, // mean reversion speed
    mu: f64, // long-run mean
    sigma: f64, // diffusion coefficient
    x: f64, // current value
    dt: f64, // time step
    window: [512]f64,
    win_idx: u16,
    count: u64,

    // Online parameter estimation
    sum_x: f64,
    sum_x2: f64,
    sum_xy: f64, // x(t) vs x(t-1) regression
    sum_y: f64,
    sum_y2: f64,
    n_pairs: u64,

    pub fn init(dt: f64) OUDetector {
        return .{
            .theta = 0,
            .mu = 0,
            .sigma = 0,
            .x = 0,
            .dt = dt,
            .window = [_]f64{0} ** 512,
            .win_idx = 0,
            .count = 0,
            .sum_x = 0,
            .sum_x2 = 0,
            .sum_xy = 0,
            .sum_y = 0,
            .sum_y2 = 0,
            .n_pairs = 0,
        };
    }

    pub fn update(self: *OUDetector, value: f64) void {
        const prev = self.x;
        self.x = value;

        self.window[self.win_idx] = value;
        self.win_idx +%= 1;
        self.count += 1;

        if (self.count < 2) return;

        // Online regression: x(t) = a + b * x(t-1) + noise
        // y = x(t), x = x(t-1)
        self.sum_x += prev;
        self.sum_x2 += prev * prev;
        self.sum_y += value;
        self.sum_y2 += value * value;
        self.sum_xy += prev * value;
        self.n_pairs += 1;

        if (self.n_pairs < 10) return;

        const n = @as(f64, @floatFromInt(self.n_pairs));
        const mean_x = self.sum_x / n;
        const mean_y = self.sum_y / n;
        const var_x = self.sum_x2 / n - mean_x * mean_x;

        if (var_x < 1e-15) return;

        const cov_xy = self.sum_xy / n - mean_x * mean_y;
        const b = cov_xy / var_x;
        const a = mean_y - b * mean_x;

        // OU parameters from AR(1): x(t) = (1-theta*dt)*x(t-1) + theta*mu*dt + sigma*sqrt(dt)*noise
        // b = exp(-theta*dt) ≈ 1 - theta*dt for small dt
        // a = mu*(1 - b)
        if (b > 0 and b < 1) {
            self.theta = -@log(b) / self.dt;
            self.mu = a / (1.0 - b);

            // Residual variance for sigma estimation
            const var_y = self.sum_y2 / n - mean_y * mean_y;
            const resid_var = var_y * (1.0 - b * b);
            if (resid_var > 0) {
                self.sigma = @sqrt(resid_var * 2.0 * self.theta / (1.0 - b * b));
            }
        }
    }

    pub fn halfLife(self: *const OUDetector) f64 {
        if (self.theta <= 0) return math.inf(f64);
        return @log(2.0) / self.theta;
    }

    pub fn zScore(self: *const OUDetector) f64 {
        if (self.sigma <= 0 or self.theta <= 0) return 0;
        const eq_std = self.sigma / @sqrt(2.0 * self.theta);
        if (eq_std < 1e-15) return 0;
        return (self.x - self.mu) / eq_std;
    }

    pub fn isMeanReverting(self: *const OUDetector) bool {
        return self.theta > 0 and self.halfLife() < 60.0; // reverts within 60 periods
    }

    pub fn expectedValue(self: *const OUDetector, horizon: f64) f64 {
        if (self.theta <= 0) return self.x;
        return self.mu + (self.x - self.mu) * @exp(-self.theta * horizon);
    }
};

// ============================================================================
// ATRTracker: Average True Range with EMA smoothing
// ============================================================================

pub const ATRTracker = struct {
    period: u32,
    atr: f64,
    prev_close: f64,
    count: u64,
    ema_alpha: f64,
    tr_sum: f64,

    pub fn init(period: u32) ATRTracker {
        return .{
            .period = period,
            .atr = 0,
            .prev_close = 0,
            .count = 0,
            .ema_alpha = 2.0 / (@as(f64, @floatFromInt(period)) + 1.0),
            .tr_sum = 0,
        };
    }

    pub fn update(self: *ATRTracker, high: f64, low: f64, close: f64) f64 {
        if (self.count == 0) {
            self.prev_close = close;
            self.atr = high - low;
            self.count = 1;
            return self.atr;
        }

        const tr1 = high - low;
        const tr2 = @abs(high - self.prev_close);
        const tr3 = @abs(low - self.prev_close);
        const tr = @max(tr1, @max(tr2, tr3));

        self.count += 1;
        if (self.count <= self.period) {
            self.tr_sum += tr;
            self.atr = self.tr_sum / @as(f64, @floatFromInt(self.count));
        } else {
            // EMA smoothing
            self.atr = self.ema_alpha * tr + (1.0 - self.ema_alpha) * self.atr;
        }

        self.prev_close = close;
        return self.atr;
    }

    pub fn normalizedATR(self: *const ATRTracker) f64 {
        if (self.prev_close <= 0) return 0;
        return self.atr / self.prev_close;
    }

    pub fn volatilityBand(self: *const ATRTracker, multiplier: f64) struct { upper: f64, lower: f64 } {
        return .{
            .upper = self.prev_close + multiplier * self.atr,
            .lower = self.prev_close - multiplier * self.atr,
        };
    }
};

// ============================================================================
// HurstExponent: R/S analysis
// ============================================================================

pub const HurstExponent = struct {
    window: [1024]f64,
    win_idx: u16,
    count: u64,
    hurst: f64,
    min_partition: u32,
    max_partition: u32,

    pub fn init() HurstExponent {
        return .{
            .window = [_]f64{0} ** 1024,
            .win_idx = 0,
            .count = 0,
            .hurst = 0.5,
            .min_partition = 8,
            .max_partition = 256,
        };
    }

    pub fn update(self: *HurstExponent, value: f64) f64 {
        self.window[self.win_idx] = value;
        self.win_idx +%= 1;
        self.count += 1;

        if (self.count < 64) return 0.5; // not enough data

        self.computeHurst();
        return self.hurst;
    }

    fn computeHurst(self: *HurstExponent) void {
        const n: usize = @intCast(@min(self.count, 1024));
        if (n < 64) return;

        // R/S analysis at multiple scales
        var log_n_sum: f64 = 0;
        var log_rs_sum: f64 = 0;
        var log_n_sq_sum: f64 = 0;
        var log_n_rs_sum: f64 = 0;
        var num_scales: f64 = 0;

        var part_size: u32 = self.min_partition;
        while (part_size <= @min(self.max_partition, @as(u32, @intCast(n / 2)))) : (part_size *= 2) {
            const num_parts = n / part_size;
            if (num_parts == 0) continue;

            var rs_sum: f64 = 0;
            var rs_count: u32 = 0;

            for (0..num_parts) |p| {
                const start = p * part_size;
                const end = start + part_size;

                // Mean of partition
                var mean: f64 = 0;
                for (start..end) |i| {
                    const idx = (self.win_idx -% @as(u16, @intCast(n)) +% @as(u16, @intCast(i)));
                    mean += self.window[idx];
                }
                mean /= @as(f64, @floatFromInt(part_size));

                // Cumulative deviation, range, and std dev
                var cum_dev: f64 = 0;
                var max_dev: f64 = -math.inf(f64);
                var min_dev: f64 = math.inf(f64);
                var sum_sq: f64 = 0;

                for (start..end) |i| {
                    const idx = (self.win_idx -% @as(u16, @intCast(n)) +% @as(u16, @intCast(i)));
                    const val = self.window[idx];
                    cum_dev += val - mean;
                    if (cum_dev > max_dev) max_dev = cum_dev;
                    if (cum_dev < min_dev) min_dev = cum_dev;
                    sum_sq += (val - mean) * (val - mean);
                }

                const range = max_dev - min_dev;
                const std_dev = @sqrt(sum_sq / @as(f64, @floatFromInt(part_size)));
                if (std_dev > 1e-15) {
                    rs_sum += range / std_dev;
                    rs_count += 1;
                }
            }

            if (rs_count > 0) {
                const avg_rs = rs_sum / @as(f64, @floatFromInt(rs_count));
                const ln_n = @log(@as(f64, @floatFromInt(part_size)));
                const ln_rs = @log(avg_rs);
                log_n_sum += ln_n;
                log_rs_sum += ln_rs;
                log_n_sq_sum += ln_n * ln_n;
                log_n_rs_sum += ln_n * ln_rs;
                num_scales += 1;
            }
        }

        // Linear regression: log(R/S) = H * log(n) + c
        if (num_scales >= 2) {
            const denom = num_scales * log_n_sq_sum - log_n_sum * log_n_sum;
            if (@abs(denom) > 1e-15) {
                self.hurst = (num_scales * log_n_rs_sum - log_n_sum * log_rs_sum) / denom;
                self.hurst = @min(@max(self.hurst, 0.0), 1.0);
            }
        }
    }

    pub fn isTrending(self: *const HurstExponent) bool {
        return self.hurst > 0.55;
    }

    pub fn isMeanReverting(self: *const HurstExponent) bool {
        return self.hurst < 0.45;
    }

    pub fn isRandom(self: *const HurstExponent) bool {
        return self.hurst >= 0.45 and self.hurst <= 0.55;
    }
};

// ============================================================================
// SignalCombiner: IC-weighted signal combination
// ============================================================================

pub const MAX_SIGNALS = 16;

pub const SignalCombiner = struct {
    weights: [MAX_SIGNALS]f64,
    signals: [MAX_SIGNALS]f64,
    ics: [MAX_SIGNALS]f64, // information coefficients
    n_signals: u32,
    combined: f64,
    conflict_score: f64,
    use_ic_weighting: bool,

    // Rolling IC computation
    signal_history: [MAX_SIGNALS][256]f64,
    outcome_history: [256]f64,
    hist_idx: u8,
    hist_count: u64,

    pub fn init(n_signals: u32) SignalCombiner {
        var s = SignalCombiner{
            .weights = [_]f64{0} ** MAX_SIGNALS,
            .signals = [_]f64{0} ** MAX_SIGNALS,
            .ics = [_]f64{0} ** MAX_SIGNALS,
            .n_signals = @min(n_signals, MAX_SIGNALS),
            .combined = 0,
            .conflict_score = 0,
            .use_ic_weighting = true,
            .signal_history = undefined,
            .outcome_history = [_]f64{0} ** 256,
            .hist_idx = 0,
            .hist_count = 0,
        };
        const eq_weight = 1.0 / @as(f64, @floatFromInt(s.n_signals));
        for (0..s.n_signals) |i| {
            s.weights[i] = eq_weight;
            s.signal_history[i] = [_]f64{0} ** 256;
        }
        return s;
    }

    pub fn setWeight(self: *SignalCombiner, idx: u32, weight: f64) void {
        if (idx < self.n_signals) self.weights[idx] = weight;
    }

    pub fn update(self: *SignalCombiner, new_signals: []const f64, outcome: f64) f64 {
        const n = @min(@as(u32, @intCast(new_signals.len)), self.n_signals);

        // Store history
        for (0..n) |i| {
            self.signals[i] = new_signals[i];
            self.signal_history[i][self.hist_idx] = new_signals[i];
        }
        self.outcome_history[self.hist_idx] = outcome;
        self.hist_idx +%= 1;
        self.hist_count += 1;

        // Update ICs periodically
        if (self.hist_count > 30 and self.hist_count % 10 == 0) {
            self.computeICs();
        }

        // Combine signals
        self.combined = 0;
        var total_weight: f64 = 0;
        for (0..n) |i| {
            const w = if (self.use_ic_weighting and self.hist_count > 30)
                @max(self.ics[i], 0) // only use positive IC signals
            else
                self.weights[i];
            self.combined += w * self.signals[i];
            total_weight += @abs(w);
        }
        if (total_weight > 0) self.combined /= total_weight;

        // Conflict detection: how many signals disagree with consensus
        self.computeConflict(n);

        return self.combined;
    }

    fn computeICs(self: *SignalCombiner) void {
        const n_obs: usize = @intCast(@min(self.hist_count, 256));
        if (n_obs < 20) return;

        for (0..self.n_signals) |sig| {
            // Pearson correlation between signal and outcome
            var sx: f64 = 0;
            var sy: f64 = 0;
            var sxx: f64 = 0;
            var syy: f64 = 0;
            var sxy: f64 = 0;
            const fn_obs = @as(f64, @floatFromInt(n_obs));

            for (0..n_obs) |i| {
                const idx = self.hist_idx -% @as(u8, @intCast(n_obs)) +% @as(u8, @intCast(i));
                const x = self.signal_history[sig][idx];
                const y = self.outcome_history[idx];
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }

            const denom = @sqrt((fn_obs * sxx - sx * sx) * (fn_obs * syy - sy * sy));
            if (denom > 1e-15) {
                self.ics[sig] = (fn_obs * sxy - sx * sy) / denom;
            } else {
                self.ics[sig] = 0;
            }
        }
    }

    fn computeConflict(self: *SignalCombiner, n: u32) void {
        var agree: u32 = 0;
        var disagree: u32 = 0;
        const sign = if (self.combined >= 0) @as(f64, 1.0) else @as(f64, -1.0);

        for (0..n) |i| {
            if (self.signals[i] * sign > 0) {
                agree += 1;
            } else if (@abs(self.signals[i]) > 0.01) {
                disagree += 1;
            }
        }
        const total = agree + disagree;
        self.conflict_score = if (total > 0)
            @as(f64, @floatFromInt(disagree)) / @as(f64, @floatFromInt(total))
        else
            0;
    }

    pub fn hasConflict(self: *const SignalCombiner) bool {
        return self.conflict_score > 0.4;
    }
};

// ============================================================================
// RegimeFilter: apply regime-conditional weights
// ============================================================================

pub const RegimeType = enum(u8) {
    trending,
    mean_reverting,
    volatile,
    calm,
    unknown,
};

pub const RegimeFilter = struct {
    current_regime: RegimeType,
    regime_weights: [5][MAX_SIGNALS]f64, // weights per regime per signal
    transition_smoothing: f64,
    prev_weights: [MAX_SIGNALS]f64,
    n_signals: u32,

    pub fn init(n_signals: u32) RegimeFilter {
        var rf = RegimeFilter{
            .current_regime = .unknown,
            .regime_weights = undefined,
            .transition_smoothing = 0.8,
            .prev_weights = [_]f64{0} ** MAX_SIGNALS,
            .n_signals = @min(n_signals, MAX_SIGNALS),
        };
        // Default: equal weights for all regimes
        const eq = 1.0 / @as(f64, @floatFromInt(rf.n_signals));
        for (0..5) |r| {
            for (0..rf.n_signals) |s| {
                rf.regime_weights[r][s] = eq;
            }
        }
        return rf;
    }

    pub fn setRegimeWeight(self: *RegimeFilter, regime: RegimeType, signal_idx: u32, weight: f64) void {
        if (signal_idx < self.n_signals) {
            self.regime_weights[@intFromEnum(regime)][signal_idx] = weight;
        }
    }

    pub fn updateRegime(self: *RegimeFilter, regime: RegimeType) void {
        self.current_regime = regime;
    }

    pub fn getWeights(self: *RegimeFilter) [MAX_SIGNALS]f64 {
        const ri = @intFromEnum(self.current_regime);
        var result: [MAX_SIGNALS]f64 = undefined;
        const alpha = self.transition_smoothing;

        for (0..self.n_signals) |i| {
            const target = self.regime_weights[ri][i];
            result[i] = alpha * self.prev_weights[i] + (1.0 - alpha) * target;
            self.prev_weights[i] = result[i];
        }
        return result;
    }

    pub fn applyToSignals(self: *RegimeFilter, signals: []const f64) f64 {
        const weights = self.getWeights();
        var combined: f64 = 0;
        var total_w: f64 = 0;
        const n = @min(@as(u32, @intCast(signals.len)), self.n_signals);
        for (0..n) |i| {
            combined += weights[i] * signals[i];
            total_w += @abs(weights[i]);
        }
        if (total_w > 0) combined /= total_w;
        return combined;
    }
};

// ============================================================================
// Full Signal Pipeline
// ============================================================================

pub const BarData = struct {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    timestamp: f64, // fractional seconds or bar index
};

pub const SignalOutput = struct {
    composite: f64,
    minkowski_class: SpacetimeClass,
    bh_mass: f64,
    bh_signal: f64,
    garch_vol: f64,
    ou_zscore: f64,
    ou_half_life: f64,
    atr: f64,
    hurst: f64,
    conflict: f64,
    regime: RegimeType,
};

pub const SignalPipeline = struct {
    mink: MinkowskiClassifier,
    bh: BHState,
    garch: GARCHTracker,
    ou: OUDetector,
    atr_tracker: ATRTracker,
    hurst_exp: HurstExponent,
    combiner: SignalCombiner,
    regime_filter: RegimeFilter,
    prev_close: f64,
    count: u64,

    pub fn init() SignalPipeline {
        return .{
            .mink = MinkowskiClassifier.init(1.0, 0.001),
            .bh = BHState.init(),
            .garch = GARCHTracker.initDefault(),
            .ou = OUDetector.init(1.0),
            .atr_tracker = ATRTracker.init(14),
            .hurst_exp = HurstExponent.init(),
            .combiner = SignalCombiner.init(5),
            .regime_filter = RegimeFilter.init(5),
            .prev_close = 0,
            .count = 0,
        };
    }

    pub fn processBar(self: *SignalPipeline, bar: BarData) SignalOutput {
        // 1. Minkowski classification
        const mink_class = self.mink.update(bar.close, bar.timestamp);

        // 2. BH mass update
        const log_ret = if (self.prev_close > 0) @log(bar.close / self.prev_close) else 0;
        _ = self.bh.update(mink_class, log_ret * bar.volume);

        // 3. GARCH vol
        _ = self.garch.update(log_ret);

        // 4. OU mean reversion
        self.ou.update(bar.close);

        // 5. ATR
        _ = self.atr_tracker.update(bar.high, bar.low, bar.close);

        // 6. Hurst exponent
        _ = self.hurst_exp.update(log_ret);

        // 7. Determine regime
        const regime: RegimeType = if (self.hurst_exp.isTrending())
            .trending
        else if (self.hurst_exp.isMeanReverting())
            .mean_reverting
        else if (self.garch.annualizedVol() > 0.3)
            .volatile
        else if (self.garch.annualizedVol() < 0.1)
            .calm
        else
            .unknown;

        self.regime_filter.updateRegime(regime);

        // 8. Assemble individual signals
        const signals = [5]f64{
            self.bh.signal(),
            -self.ou.zScore() * 0.1, // mean reversion signal
            if (mink_class == .timelike) log_ret * 10.0 else 0, // momentum
            self.atr_tracker.normalizedATR() * (if (log_ret > 0) @as(f64, -1.0) else @as(f64, 1.0)), // vol contrarian
            (self.hurst_exp.hurst - 0.5) * 2.0, // trend persistence
        };

        // 9. Combine with regime filter
        const outcome = log_ret; // use next return as outcome (lagged)
        _ = self.combiner.update(&signals, outcome);
        const composite = self.regime_filter.applyToSignals(&signals);

        self.prev_close = bar.close;
        self.count += 1;

        return .{
            .composite = composite,
            .minkowski_class = mink_class,
            .bh_mass = self.bh.mass,
            .bh_signal = self.bh.signal(),
            .garch_vol = self.garch.annualizedVol(),
            .ou_zscore = self.ou.zScore(),
            .ou_half_life = self.ou.halfLife(),
            .atr = self.atr_tracker.atr,
            .hurst = self.hurst_exp.hurst,
            .conflict = self.combiner.conflict_score,
            .regime = regime,
        };
    }
};

// ============================================================================
// SIMD-accelerated helpers
// ============================================================================

pub fn vecEMA(data: []const f64, alpha: f64, out: []f64) void {
    if (data.len == 0) return;
    out[0] = data[0];
    const one_minus_alpha = 1.0 - alpha;
    for (1..data.len) |i| {
        out[i] = alpha * data[i] + one_minus_alpha * out[i - 1];
    }
}

pub fn vecLogReturns(prices: []const f64, out: []f64) void {
    if (prices.len < 2) return;
    for (1..prices.len) |i| {
        out[i - 1] = if (prices[i - 1] > 0) @log(prices[i] / prices[i - 1]) else 0;
    }
}

pub fn vecRollingMean(data: []const f64, window: usize, out: []f64) void {
    if (data.len == 0 or window == 0) return;
    var sum: f64 = 0;
    for (0..data.len) |i| {
        sum += data[i];
        if (i >= window) sum -= data[i - window];
        const n = @min(i + 1, window);
        out[i] = sum / @as(f64, @floatFromInt(n));
    }
}

pub fn vecRollingStd(data: []const f64, window: usize, out: []f64) void {
    if (data.len == 0 or window < 2) return;
    for (0..data.len) |i| {
        if (i + 1 < window) {
            out[i] = 0;
            continue;
        }
        const start = i + 1 - window;
        var mean: f64 = 0;
        for (start..i + 1) |j| mean += data[j];
        mean /= @as(f64, @floatFromInt(window));
        var variance: f64 = 0;
        for (start..i + 1) |j| {
            const d = data[j] - mean;
            variance += d * d;
        }
        out[i] = @sqrt(variance / @as(f64, @floatFromInt(window - 1)));
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// KalmanFilter: 1D Kalman for signal smoothing
// ============================================================================

pub const KalmanFilter = struct {
    x: f64, // state estimate
    p: f64, // estimate covariance
    q: f64, // process noise
    r_noise: f64, // measurement noise
    k: f64, // Kalman gain
    count: u64,

    pub fn init(process_noise: f64, measurement_noise: f64) KalmanFilter {
        return .{
            .x = 0,
            .p = 1.0,
            .q = process_noise,
            .r_noise = measurement_noise,
            .k = 0,
            .count = 0,
        };
    }

    pub fn update(self: *KalmanFilter, measurement: f64) f64 {
        if (self.count == 0) {
            self.x = measurement;
            self.count = 1;
            return self.x;
        }

        // Predict
        // x_pred = x (random walk model)
        // p_pred = p + q
        const p_pred = self.p + self.q;

        // Update
        self.k = p_pred / (p_pred + self.r_noise);
        self.x = self.x + self.k * (measurement - self.x);
        self.p = (1.0 - self.k) * p_pred;
        self.count += 1;

        return self.x;
    }

    pub fn innovationVariance(self: *const KalmanFilter) f64 {
        return self.p + self.r_noise;
    }

    pub fn normalizedInnovation(self: *const KalmanFilter, measurement: f64) f64 {
        const iv = self.innovationVariance();
        if (iv <= 0) return 0;
        return (measurement - self.x) / @sqrt(iv);
    }
};

// ============================================================================
// ExponentialMovingAverage with multiple spans
// ============================================================================

pub const MultiEMA = struct {
    values: [8]f64,
    alphas: [8]f64,
    n_spans: u32,
    count: u64,

    pub fn init(spans: []const u32) MultiEMA {
        var ema = MultiEMA{
            .values = [_]f64{0} ** 8,
            .alphas = [_]f64{0} ** 8,
            .n_spans = @intCast(@min(spans.len, 8)),
            .count = 0,
        };
        for (0..ema.n_spans) |i| {
            ema.alphas[i] = 2.0 / (@as(f64, @floatFromInt(spans[i])) + 1.0);
        }
        return ema;
    }

    pub fn update(self: *MultiEMA, value: f64) void {
        if (self.count == 0) {
            for (0..self.n_spans) |i| self.values[i] = value;
        } else {
            for (0..self.n_spans) |i| {
                self.values[i] = self.alphas[i] * value + (1.0 - self.alphas[i]) * self.values[i];
            }
        }
        self.count += 1;
    }

    pub fn get(self: *const MultiEMA, idx: u32) f64 {
        if (idx >= self.n_spans) return 0;
        return self.values[idx];
    }

    pub fn crossover(self: *const MultiEMA, fast_idx: u32, slow_idx: u32) f64 {
        return self.get(fast_idx) - self.get(slow_idx);
    }
};

// ============================================================================
// ZScore: rolling z-score computation
// ============================================================================

pub const RollingZScore = struct {
    window: [512]f64,
    win_idx: u16,
    count: u64,
    sum: f64,
    sum_sq: f64,
    period: u32,

    pub fn init(period: u32) RollingZScore {
        return .{
            .window = [_]f64{0} ** 512,
            .win_idx = 0,
            .count = 0,
            .sum = 0,
            .sum_sq = 0,
            .period = @min(period, 512),
        };
    }

    pub fn update(self: *RollingZScore, value: f64) f64 {
        if (self.count >= self.period) {
            const old_idx = self.win_idx -% @as(u16, @intCast(self.period));
            const old_val = self.window[old_idx];
            self.sum -= old_val;
            self.sum_sq -= old_val * old_val;
        }

        self.window[self.win_idx] = value;
        self.win_idx +%= 1;
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;

        const n = @min(self.count, @as(u64, self.period));
        if (n < 2) return 0;

        const fn_val = @as(f64, @floatFromInt(n));
        const mean = self.sum / fn_val;
        const variance = self.sum_sq / fn_val - mean * mean;
        if (variance <= 0) return 0;
        return (value - mean) / @sqrt(variance);
    }
};

// ============================================================================
// MomentumTracker: multi-timeframe momentum
// ============================================================================

pub const MomentumTracker = struct {
    prices: [512]f64,
    idx: u16,
    count: u64,
    lookbacks: [6]u32,

    pub fn init() MomentumTracker {
        return .{
            .prices = [_]f64{0} ** 512,
            .idx = 0,
            .count = 0,
            .lookbacks = .{ 5, 10, 21, 63, 126, 252 },
        };
    }

    pub fn update(self: *MomentumTracker, price: f64) void {
        self.prices[self.idx] = price;
        self.idx +%= 1;
        self.count += 1;
    }

    pub fn momentum(self: *const MomentumTracker, lookback_idx: u32) f64 {
        if (lookback_idx >= 6) return 0;
        const lb = self.lookbacks[lookback_idx];
        if (self.count <= lb) return 0;
        const past_idx = self.idx -% @as(u16, @intCast(lb)) -% 1;
        const past_price = self.prices[past_idx];
        const current = self.prices[self.idx -% 1];
        if (past_price <= 0) return 0;
        return @log(current / past_price);
    }

    pub fn compositeMomentum(self: *const MomentumTracker) f64 {
        // Weighted average of all timeframes
        const weights = [6]f64{ 0.05, 0.10, 0.20, 0.25, 0.25, 0.15 };
        var sum: f64 = 0;
        var total_w: f64 = 0;
        for (0..6) |i| {
            const m = self.momentum(@intCast(i));
            if (m != 0) {
                sum += weights[i] * m;
                total_w += weights[i];
            }
        }
        if (total_w == 0) return 0;
        return sum / total_w;
    }
};

// ============================================================================
// VolatilityCone: realized vol percentiles by horizon
// ============================================================================

pub const VolatilityCone = struct {
    returns: [1024]f64,
    ret_idx: u16,
    count: u64,

    pub fn init() VolatilityCone {
        return .{
            .returns = [_]f64{0} ** 1024,
            .ret_idx = 0,
            .count = 0,
        };
    }

    pub fn update(self: *VolatilityCone, log_return: f64) void {
        self.returns[self.ret_idx] = log_return;
        self.ret_idx +%= 1;
        self.count += 1;
    }

    pub fn realizedVol(self: *const VolatilityCone, window: u32) f64 {
        if (self.count < window or window < 2) return 0;
        var sum: f64 = 0;
        var sum_sq: f64 = 0;
        for (0..window) |i| {
            const idx = self.ret_idx -% @as(u16, @intCast(window)) +% @as(u16, @intCast(i));
            const r = self.returns[idx];
            sum += r;
            sum_sq += r * r;
        }
        const n = @as(f64, @floatFromInt(window));
        const mean = sum / n;
        const variance = sum_sq / n - mean * mean;
        if (variance <= 0) return 0;
        return @sqrt(variance * 252.0);
    }

    pub fn volPercentile(self: *const VolatilityCone, window: u32, current_vol: f64) f64 {
        if (self.count < window * 2) return 0.5;
        // Compute historical vols and rank current
        var below: u32 = 0;
        var total: u32 = 0;
        const max_samples = @min(self.count / window, 50);
        for (0..max_samples) |s| {
            const offset: u16 = @intCast(s * window);
            var sum_sq: f64 = 0;
            for (0..window) |i| {
                const idx = self.ret_idx -% offset -% @as(u16, @intCast(window)) +% @as(u16, @intCast(i));
                sum_sq += self.returns[idx] * self.returns[idx];
            }
            const hist_vol = @sqrt(sum_sq / @as(f64, @floatFromInt(window)) * 252.0);
            if (hist_vol < current_vol) below += 1;
            total += 1;
        }
        if (total == 0) return 0.5;
        return @as(f64, @floatFromInt(below)) / @as(f64, @floatFromInt(total));
    }
};

// ============================================================================
// DrawdownTracker
// ============================================================================

pub const DrawdownTracker = struct {
    peak: f64,
    current: f64,
    max_drawdown: f64,
    current_drawdown: f64,
    drawdown_duration: u32,
    max_duration: u32,
    recovery_count: u32,

    pub fn init() DrawdownTracker {
        return .{
            .peak = 0,
            .current = 0,
            .max_drawdown = 0,
            .current_drawdown = 0,
            .drawdown_duration = 0,
            .max_duration = 0,
            .recovery_count = 0,
        };
    }

    pub fn update(self: *DrawdownTracker, value: f64) void {
        self.current = value;
        if (value > self.peak) {
            if (self.drawdown_duration > 0) self.recovery_count += 1;
            self.peak = value;
            self.drawdown_duration = 0;
        }
        self.current_drawdown = if (self.peak > 0) (self.peak - value) / self.peak else 0;
        if (self.current_drawdown > self.max_drawdown) self.max_drawdown = self.current_drawdown;
        if (self.current_drawdown > 0) {
            self.drawdown_duration += 1;
            if (self.drawdown_duration > self.max_duration) self.max_duration = self.drawdown_duration;
        }
    }

    pub fn isInDrawdown(self: *const DrawdownTracker) bool {
        return self.current_drawdown > 0.001;
    }

    pub fn drawdownSeverity(self: *const DrawdownTracker) f64 {
        // Combined duration and depth
        if (self.max_drawdown == 0) return 0;
        return self.current_drawdown / self.max_drawdown *
            @as(f64, @floatFromInt(self.drawdown_duration)) / @as(f64, @floatFromInt(@max(self.max_duration, 1)));
    }
};

// ============================================================================
// RiskParity: equal risk contribution weights
// ============================================================================

pub const RiskParity = struct {
    n_assets: u32,
    vols: [16]f64,
    weights: [16]f64,
    target_vol: f64,

    pub fn init(n_assets: u32, target_vol: f64) RiskParity {
        var rp = RiskParity{
            .n_assets = @min(n_assets, 16),
            .vols = [_]f64{0.2} ** 16,
            .weights = [_]f64{0} ** 16,
            .target_vol = target_vol,
        };
        const eq = 1.0 / @as(f64, @floatFromInt(rp.n_assets));
        for (0..rp.n_assets) |i| rp.weights[i] = eq;
        return rp;
    }

    pub fn updateVol(self: *RiskParity, idx: u32, vol: f64) void {
        if (idx < self.n_assets) self.vols[idx] = @max(vol, 0.01);
    }

    pub fn computeWeights(self: *RiskParity) void {
        // Inverse-vol weighting (simplified risk parity)
        var inv_vol_sum: f64 = 0;
        for (0..self.n_assets) |i| {
            inv_vol_sum += 1.0 / self.vols[i];
        }
        if (inv_vol_sum == 0) return;

        var portfolio_vol: f64 = 0;
        for (0..self.n_assets) |i| {
            self.weights[i] = (1.0 / self.vols[i]) / inv_vol_sum;
            portfolio_vol += self.weights[i] * self.weights[i] * self.vols[i] * self.vols[i];
        }
        portfolio_vol = @sqrt(portfolio_vol);

        // Scale to target vol
        if (portfolio_vol > 0) {
            const scale = self.target_vol / portfolio_vol;
            for (0..self.n_assets) |i| {
                self.weights[i] *= scale;
            }
        }
    }

    pub fn getWeight(self: *const RiskParity, idx: u32) f64 {
        if (idx >= self.n_assets) return 0;
        return self.weights[idx];
    }

    pub fn riskContribution(self: *const RiskParity, idx: u32) f64 {
        if (idx >= self.n_assets) return 0;
        return self.weights[idx] * self.vols[idx];
    }
};

// ============================================================================
// PositionSizer: Kelly criterion and vol targeting
// ============================================================================

pub const PositionSizer = struct {
    target_vol: f64,
    max_position: f64,
    kelly_fraction: f64, // fraction of full Kelly to use
    wins: u64,
    losses: u64,
    avg_win: f64,
    avg_loss: f64,

    pub fn init(target_vol: f64, max_position: f64, kelly_frac: f64) PositionSizer {
        return .{
            .target_vol = target_vol,
            .max_position = max_position,
            .kelly_fraction = kelly_frac,
            .wins = 0,
            .losses = 0,
            .avg_win = 0,
            .avg_loss = 0,
        };
    }

    pub fn recordTrade(self: *PositionSizer, pnl: f64) void {
        if (pnl > 0) {
            self.avg_win = (self.avg_win * @as(f64, @floatFromInt(self.wins)) + pnl) / @as(f64, @floatFromInt(self.wins + 1));
            self.wins += 1;
        } else if (pnl < 0) {
            self.avg_loss = (self.avg_loss * @as(f64, @floatFromInt(self.losses)) + @abs(pnl)) / @as(f64, @floatFromInt(self.losses + 1));
            self.losses += 1;
        }
    }

    pub fn kellySize(self: *const PositionSizer) f64 {
        if (self.wins + self.losses < 20) return 0.01; // insufficient data
        const total = @as(f64, @floatFromInt(self.wins + self.losses));
        const win_rate = @as(f64, @floatFromInt(self.wins)) / total;
        if (self.avg_loss == 0) return self.max_position;
        const win_loss_ratio = self.avg_win / self.avg_loss;
        const kelly = win_rate - (1.0 - win_rate) / win_loss_ratio;
        const sized = kelly * self.kelly_fraction;
        return @min(@max(sized, 0), self.max_position);
    }

    pub fn volTargetSize(self: *const PositionSizer, current_vol: f64, signal_strength: f64) f64 {
        if (current_vol <= 0) return 0;
        const raw = self.target_vol / current_vol * @abs(signal_strength);
        return @min(raw, self.max_position);
    }

    pub fn optimalSize(self: *const PositionSizer, current_vol: f64, signal_strength: f64) f64 {
        const kelly = self.kellySize();
        const vol_target = self.volTargetSize(current_vol, signal_strength);
        return @min(kelly, vol_target); // conservative: take minimum
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "MinkowskiClassifier basic" {
    var mc = MinkowskiClassifier.init(1.0, 0.001);
    _ = mc.update(100.0, 0.0);
    const cls = mc.update(100.001, 1.0);
    try testing.expect(cls == .timelike); // dt=1 >> dx=0.001
}

test "BHState accumulation" {
    var bh = BHState.init();
    _ = bh.update(.timelike, 0.05);
    _ = bh.update(.timelike, 0.05);
    try testing.expect(bh.mass > 1.0);
    _ = bh.update(.spacelike, 0);
    const after_decay = bh.mass;
    try testing.expect(after_decay < bh.mass_history[bh.hist_idx -% 2]);
}

test "GARCHTracker volatility" {
    var g = GARCHTracker.initDefault();
    _ = g.update(0.01);
    _ = g.update(-0.02);
    _ = g.update(0.015);
    try testing.expect(g.volatility() > 0);
    try testing.expect(g.annualizedVol() > g.volatility());
}

test "OUDetector mean reversion" {
    var ou = OUDetector.init(1.0);
    // Feed mean-reverting series
    var x: f64 = 0;
    for (0..200) |i| {
        x = 0.9 * x + 0.1 * @sin(@as(f64, @floatFromInt(i)) * 0.1);
        ou.update(x);
    }
    try testing.expect(ou.theta > 0);
}

test "SignalPipeline smoke" {
    var pipeline = SignalPipeline.init();
    const bar = BarData{ .open = 100, .high = 101, .low = 99, .close = 100.5, .volume = 1000, .timestamp = 1.0 };
    const out = pipeline.processBar(bar);
    _ = out;
    const bar2 = BarData{ .open = 100.5, .high = 102, .low = 100, .close = 101, .volume = 1200, .timestamp = 2.0 };
    const out2 = pipeline.processBar(bar2);
    try testing.expect(out2.garch_vol >= 0);
}
