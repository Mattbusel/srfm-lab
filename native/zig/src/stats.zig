//! Real-time market statistics:
//! VWAP, spread, order imbalance, trade intensity, tick-by-tick volatility.

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// ============================================================
// Rolling window accumulator
// Maintains a fixed-length circular buffer of samples.
// ============================================================
pub fn RollingWindow(comptime T: type, comptime N: usize) type {
    return struct {
        buf:    [N]T = undefined,
        head:   usize = 0,
        count:  usize = 0,
        sum:    T = 0,

        const Self = @This();

        pub fn push(self: *Self, v: T) void {
            if (self.count == N) {
                // Remove oldest
                self.sum -= self.buf[self.head];
                self.buf[self.head] = v;
                self.head = (self.head + 1) % N;
            } else {
                self.buf[(self.head + self.count) % N] = v;
                self.count += 1;
            }
            self.sum += v;
        }

        pub fn mean(self: *const Self) T {
            if (self.count == 0) return 0;
            return self.sum / @as(T, @floatFromInt(self.count));
        }

        pub fn variance(self: *const Self) T {
            if (self.count < 2) return 0;
            const m = self.mean();
            var v: T = 0;
            for (0..self.count) |i| {
                const x = self.buf[(self.head + i) % N] - m;
                v += x * x;
            }
            return v / @as(T, @floatFromInt(self.count - 1));
        }

        pub fn std_dev(self: *const Self) T {
            return @sqrt(self.variance());
        }

        pub fn min(self: *const Self) T {
            if (self.count == 0) return 0;
            var m = self.buf[self.head];
            for (1..self.count) |i| {
                const v = self.buf[(self.head + i) % N];
                if (v < m) m = v;
            }
            return m;
        }

        pub fn max(self: *const Self) T {
            if (self.count == 0) return 0;
            var m = self.buf[self.head];
            for (1..self.count) |i| {
                const v = self.buf[(self.head + i) % N];
                if (v > m) m = v;
            }
            return m;
        }

        pub fn full(self: *const Self) bool { return self.count == N; }
        pub fn size(self: *const Self) usize { return self.count; }
    };
}

// ============================================================
// VWAP Calculator
// Volume-Weighted Average Price over a rolling window.
// Supports session VWAP and rolling VWAP.
// ============================================================
pub const VWAPCalc = struct {
    // Session VWAP (from session open)
    session_num: f64 = 0.0,  // sum(price * qty)
    session_den: u64 = 0,    // sum(qty)

    // Rolling VWAP (circular buffer)
    const WINDOW = 1000;
    prices: [WINDOW]f64 = undefined,
    qtys:   [WINDOW]u64 = undefined,
    head:   usize = 0,
    count:  usize = 0,
    roll_num: f64 = 0.0,
    roll_den: u64 = 0,

    const Self = @This();

    pub fn update(self: *Self, price: f64, qty: u64) void {
        // Session
        self.session_num += price * @as(f64, @floatFromInt(qty));
        self.session_den += qty;

        // Rolling
        if (self.count == WINDOW) {
            // Remove oldest
            self.roll_num -= self.prices[self.head] * @as(f64, @floatFromInt(self.qtys[self.head]));
            self.roll_den -= self.qtys[self.head];
            self.prices[self.head] = price;
            self.qtys[self.head]   = qty;
            self.head = (self.head + 1) % WINDOW;
        } else {
            self.prices[(self.head + self.count) % WINDOW] = price;
            self.qtys[(self.head + self.count) % WINDOW]   = qty;
            self.count += 1;
        }
        self.roll_num += price * @as(f64, @floatFromInt(qty));
        self.roll_den += qty;
    }

    pub fn session_vwap(self: *const Self) f64 {
        if (self.session_den == 0) return 0.0;
        return self.session_num / @as(f64, @floatFromInt(self.session_den));
    }

    pub fn rolling_vwap(self: *const Self) f64 {
        if (self.roll_den == 0) return 0.0;
        return self.roll_num / @as(f64, @floatFromInt(self.roll_den));
    }

    pub fn reset_session(self: *Self) void {
        self.session_num = 0.0;
        self.session_den = 0;
    }
};

// ============================================================
// Spread Statistics
// Tracks bid-ask spread distribution over time.
// ============================================================
pub const SpreadStats = struct {
    window: RollingWindow(f64, 500) = .{},
    tick_size: f64,
    total_updates: u64 = 0,
    zero_spread_count: u64 = 0,

    const Self = @This();

    pub fn init(tick_size: f64) Self {
        return .{ .tick_size = tick_size };
    }

    pub fn update(self: *Self, bid: f64, ask: f64) void {
        const sp = ask - bid;
        if (sp < 0) return; // inverted book, skip
        if (sp < self.tick_size * 0.1) self.zero_spread_count += 1;
        self.window.push(sp);
        self.total_updates += 1;
    }

    pub fn mean_spread(self: *const Self) f64 { return self.window.mean(); }
    pub fn min_spread(self: *const Self) f64  { return self.window.min(); }
    pub fn max_spread(self: *const Self) f64  { return self.window.max(); }

    pub fn spread_in_ticks(self: *const Self) f64 {
        if (self.tick_size == 0) return 0.0;
        return self.mean_spread() / self.tick_size;
    }
};

// ============================================================
// Order Imbalance
// (bid_qty - ask_qty) / (bid_qty + ask_qty)
// Positive → more buying pressure, Negative → more selling pressure
// ============================================================
pub const ImbalanceTracker = struct {
    window: RollingWindow(f64, 200) = .{},
    signal_threshold: f64 = 0.3, // |imbalance| > threshold → signal

    const Self = @This();

    pub fn update(self: *Self, bid_qty: u64, ask_qty: u64) void {
        const total = bid_qty + ask_qty;
        if (total == 0) return;
        const imb = (@as(f64, @floatFromInt(bid_qty)) - @as(f64, @floatFromInt(ask_qty))) /
                    @as(f64, @floatFromInt(total));
        self.window.push(imb);
    }

    pub fn current(self: *const Self) f64 {
        if (self.window.count == 0) return 0.0;
        return self.window.buf[(self.window.head + self.window.count - 1) % 200];
    }

    pub fn avg(self: *const Self) f64 { return self.window.mean(); }

    pub fn signal(self: *const Self) i8 {
        const imb = self.current();
        if (imb > self.signal_threshold) return 1;   // buy signal
        if (imb < -self.signal_threshold) return -1; // sell signal
        return 0;
    }
};

// ============================================================
// Trade Intensity (Kyle's Lambda proxy)
// Tracks trade arrival rate and signed order flow
// ============================================================
pub const TradeIntensity = struct {
    const BUCKET_SIZE_NS: i64 = 1_000_000_000; // 1 second buckets
    const NUM_BUCKETS: usize  = 60;             // 60-second window

    trade_counts: [NUM_BUCKETS]u32 = [_]u32{0} ** NUM_BUCKETS,
    buy_volume:   [NUM_BUCKETS]u64 = [_]u64{0} ** NUM_BUCKETS,
    sell_volume:  [NUM_BUCKETS]u64 = [_]u64{0} ** NUM_BUCKETS,
    current_bucket: usize = 0,
    bucket_start_ns: i64 = 0,
    total_trades: u64 = 0,

    const Self = @This();

    pub fn update(self: *Self, side: u8, qty: u64, timestamp_ns: i64) void {
        // Advance bucket if time has passed
        if (self.bucket_start_ns == 0) self.bucket_start_ns = timestamp_ns;

        const elapsed = timestamp_ns - self.bucket_start_ns;
        const buckets_to_advance: usize = @intCast(@divFloor(elapsed, BUCKET_SIZE_NS));

        if (buckets_to_advance > 0) {
            const advance = @min(buckets_to_advance, NUM_BUCKETS);
            for (0..advance) |_| {
                self.current_bucket = (self.current_bucket + 1) % NUM_BUCKETS;
                self.trade_counts[self.current_bucket] = 0;
                self.buy_volume[self.current_bucket]   = 0;
                self.sell_volume[self.current_bucket]  = 0;
            }
            self.bucket_start_ns += @as(i64, @intCast(buckets_to_advance)) * BUCKET_SIZE_NS;
        }

        self.trade_counts[self.current_bucket] += 1;
        if (side == 0) self.buy_volume[self.current_bucket]  += qty
        else           self.sell_volume[self.current_bucket] += qty;
        self.total_trades += 1;
    }

    // Total trades per second (over window)
    pub fn trades_per_second(self: *const Self) f64 {
        var total: u64 = 0;
        for (self.trade_counts) |c| total += c;
        return @as(f64, @floatFromInt(total)) / NUM_BUCKETS;
    }

    // Signed order flow (buy_vol - sell_vol) over window
    pub fn signed_order_flow(self: *const Self) i64 {
        var buy: u64 = 0;
        var sell: u64 = 0;
        for (self.buy_volume) |v| buy += v;
        for (self.sell_volume) |v| sell += v;
        return @as(i64, @intCast(buy)) - @as(i64, @intCast(sell));
    }

    pub fn buy_sell_ratio(self: *const Self) f64 {
        var buy: u64 = 0;
        var sell: u64 = 0;
        for (self.buy_volume) |v| buy += v;
        for (self.sell_volume) |v| sell += v;
        const total = buy + sell;
        if (total == 0) return 0.5;
        return @as(f64, @floatFromInt(buy)) / @as(f64, @floatFromInt(total));
    }
};

// ============================================================
// Tick-by-Tick Volatility (Realized Variance)
// Computes rolling realized variance from log-returns
// ============================================================
pub const VolatilityCalc = struct {
    const WINDOW: usize = 300; // 300 ticks
    returns: RollingWindow(f64, WINDOW) = .{},
    last_price: f64 = 0.0,
    tick_count: u64 = 0,

    const Self = @This();

    pub fn update(self: *Self, price: f64) void {
        if (self.last_price > 0 and price > 0) {
            const ret = @log(price / self.last_price);
            self.returns.push(ret);
        }
        self.last_price = price;
        self.tick_count += 1;
    }

    // Realized variance (annualized, assumes 252 trading days, 6.5h/day)
    pub fn realized_variance(self: *const Self) f64 {
        const n = self.returns.size();
        if (n < 2) return 0.0;
        const var_per_tick = self.returns.variance();
        // Scale to annual: assume ~1M ticks/day * 252 days
        const ticks_per_year: f64 = 1_000_000.0 * 252.0;
        return var_per_tick * ticks_per_year;
    }

    // Annualized volatility
    pub fn annualized_vol(self: *const Self) f64 {
        return @sqrt(self.realized_variance());
    }

    // Parkinson volatility (high-low range based)
    // Requires high/low prices; uses returns window as proxy
    pub fn parkinson_vol_approx(self: *const Self) f64 {
        if (self.returns.size() < 10) return 0.0;
        const hi = self.returns.max();
        const lo = self.returns.min();
        const hl_sq = (hi - lo) * (hi - lo) / (4.0 * @log(2.0));
        const n = self.returns.size();
        const ticks_per_year: f64 = 1_000_000.0 * 252.0;
        return @sqrt(hl_sq * ticks_per_year / @as(f64, @floatFromInt(n)));
    }
};

// ============================================================
// Composite Market Stats (aggregates all metrics)
// ============================================================
pub const MarketStats = struct {
    symbol: [16]u8,
    vwap:      VWAPCalc      = .{},
    spread:    SpreadStats,
    imbalance: ImbalanceTracker = .{},
    intensity: TradeIntensity   = .{},
    vol:       VolatilityCalc   = .{},

    last_price: f64 = 0.0,
    last_bid:   f64 = 0.0,
    last_ask:   f64 = 0.0,
    tick_count: u64 = 0,

    const Self = @This();

    pub fn init(symbol: []const u8, tick_size: f64) Self {
        var sym: [16]u8 = [_]u8{0} ** 16;
        const len = @min(symbol.len, 15);
        @memcpy(sym[0..len], symbol[0..len]);
        return .{
            .symbol = sym,
            .spread = SpreadStats.init(tick_size),
        };
    }

    pub fn on_trade(self: *Self, price: f64, qty: u64, side: u8, ts_ns: i64) void {
        self.last_price = price;
        self.vwap.update(price, qty);
        self.vol.update(price);
        self.intensity.update(side, qty, ts_ns);
        self.tick_count += 1;
    }

    pub fn on_quote(self: *Self, bid: f64, bid_qty: u64, ask: f64, ask_qty: u64) void {
        self.last_bid = bid;
        self.last_ask = ask;
        self.spread.update(bid, ask);
        self.imbalance.update(bid_qty, ask_qty);
    }

    pub fn summary(self: *const Self) StatsSummary {
        return .{
            .session_vwap     = self.vwap.session_vwap(),
            .rolling_vwap     = self.vwap.rolling_vwap(),
            .mean_spread      = self.spread.mean_spread(),
            .spread_bps       = if (self.last_price > 0) self.spread.mean_spread() / self.last_price * 10000.0 else 0.0,
            .order_imbalance  = self.imbalance.current(),
            .imbalance_signal = self.imbalance.signal(),
            .trades_per_sec   = self.intensity.trades_per_second(),
            .signed_flow      = self.intensity.signed_order_flow(),
            .annualized_vol   = self.vol.annualized_vol(),
            .last_price       = self.last_price,
            .last_bid         = self.last_bid,
            .last_ask         = self.last_ask,
            .tick_count       = self.tick_count,
        };
    }
};

pub const StatsSummary = struct {
    session_vwap:     f64,
    rolling_vwap:     f64,
    mean_spread:      f64,
    spread_bps:       f64,
    order_imbalance:  f64,
    imbalance_signal: i8,
    trades_per_sec:   f64,
    signed_flow:      i64,
    annualized_vol:   f64,
    last_price:       f64,
    last_bid:         f64,
    last_ask:         f64,
    tick_count:       u64,
};

// ============================================================
// Tests
// ============================================================
test "rolling window" {
    var w: RollingWindow(f64, 5) = .{};
    w.push(1.0); w.push(2.0); w.push(3.0); w.push(4.0); w.push(5.0);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), w.mean(), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), w.min(), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), w.max(), 1e-10);

    // Overflow: oldest removed
    w.push(6.0); // removes 1.0
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), w.mean(), 1e-10);
}

test "vwap" {
    var v: VWAPCalc = .{};
    v.update(100.0, 1000);
    v.update(101.0, 500);
    v.update(99.0,  500);
    // VWAP = (100*1000 + 101*500 + 99*500) / 2000 = (100000+50500+49500)/2000 = 200000/2000 = 100.0
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), v.session_vwap(), 1e-6);
}

test "volatility calculation" {
    var vc: VolatilityCalc = .{};
    // Feed 100 prices with 1% moves
    var p: f64 = 100.0;
    for (0..100) |_| {
        p *= 1.001;
        vc.update(p);
    }
    const vol = vc.annualized_vol();
    try std.testing.expect(vol > 0.0);
    try std.testing.expect(vol < 100.0); // sanity: < 10000% annualized
}

test "trade intensity" {
    var ti: TradeIntensity = .{};
    const base_ts: i64 = 1_000_000_000;
    for (0..100) |i| {
        ti.update(0, 100, base_ts + @as(i64, @intCast(i)) * 10_000_000);
    }
    for (0..80) |i| {
        ti.update(1, 100, base_ts + @as(i64, @intCast(i)) * 12_000_000);
    }
    try std.testing.expect(ti.total_trades == 180);
    const ratio = ti.buy_sell_ratio();
    try std.testing.expect(ratio > 0.5); // more buys
}

test "imbalance" {
    var im: ImbalanceTracker = .{};
    im.update(1000, 100); // strong bid imbalance
    const imb = im.current();
    try std.testing.expect(imb > 0.5);
    try std.testing.expectEqual(@as(i8, 1), im.signal());
}
