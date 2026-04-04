//! Risk management for Zig market-making engine.
//! Pre-trade checks, position limits, drawdown monitoring, and kill switch.

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

const orderbook = @import("orderbook.zig");
const mm_mod    = @import("market_maker.zig");
const stats_mod = @import("stats.zig");

pub const Price    = orderbook.Price;
pub const Quantity = orderbook.Quantity;
pub const OrderId  = orderbook.OrderId;
pub const Side     = orderbook.Side;

// ============================================================
// Risk limits configuration
// ============================================================
pub const RiskLimits = struct {
    // Position limits
    max_long_position:   i64   = 50_000,
    max_short_position:  i64   = -50_000,
    max_gross_exposure:  f64   = 10_000_000.0,  // USD

    // Loss limits
    max_daily_loss:      f64   = 100_000.0,     // USD
    max_drawdown_pct:    f64   = 0.05,          // 5%
    max_session_loss:    f64   = 50_000.0,      // USD

    // Order rate limits
    max_orders_per_sec:  u64   = 500,
    max_orders_per_min:  u64   = 10_000,
    max_cancel_ratio:    f64   = 0.95,          // cancels/orders

    // Size limits
    max_order_qty:       u64   = 5_000,
    min_order_qty:       u64   = 1,
    max_order_notional:  f64   = 500_000.0,     // USD per order

    // Market sanity checks
    max_price_deviation_bps: f64 = 50.0,        // vs reference price
    min_spread_bps:          f64 = 0.5,
    max_spread_bps:          f64 = 200.0,

    // Fat finger checks
    fat_finger_qty_mult:     f64 = 10.0,        // vs avg order size
    fat_finger_px_mult:      f64 = 0.05,        // 5% from mid
};

// ============================================================
// Position state per symbol
// ============================================================
pub const PositionState = struct {
    symbol:          [16]u8,
    net_position:    i64 = 0,
    avg_cost:        f64 = 0.0,
    realized_pnl:    f64 = 0.0,
    unrealized_pnl:  f64 = 0.0,
    total_buys:      u64 = 0,
    total_sells:     u64 = 0,
    gross_exposure:  f64 = 0.0,
    max_long:        i64 = 0,
    max_short:       i64 = 0,

    const Self = @This();

    pub fn on_fill(self: *Self, side: Side, qty: u64, px: f64) void {
        const q = @as(f64, @floatFromInt(qty));
        if (side == .Buy) {
            const total_cost = self.avg_cost * @as(f64, @floatFromInt(@max(self.net_position, 0))) + px * q;
            self.net_position += @as(i64, @intCast(qty));
            if (self.net_position > 0)
                self.avg_cost = total_cost / @as(f64, @floatFromInt(self.net_position));
            self.total_buys += qty;
            if (self.net_position > self.max_long) self.max_long = self.net_position;
        } else {
            const pos = @as(f64, @floatFromInt(@max(self.net_position, 0)));
            const close = @min(pos, q);
            self.realized_pnl += (px - self.avg_cost) * close;
            self.net_position -= @as(i64, @intCast(qty));
            self.total_sells += qty;
            if (self.net_position < self.max_short) self.max_short = self.net_position;
        }
    }

    pub fn mark_to_market(self: *Self, mid: f64) void {
        if (self.net_position != 0 and self.avg_cost > 0) {
            self.unrealized_pnl = (mid - self.avg_cost) * @as(f64, @floatFromInt(self.net_position));
            self.gross_exposure = @fabs(@as(f64, @floatFromInt(self.net_position)) * mid);
        }
    }

    pub fn net_pnl(self: *const Self) f64 {
        return self.realized_pnl + self.unrealized_pnl;
    }

    pub fn is_long(self: *const Self) bool  { return self.net_position > 0; }
    pub fn is_short(self: *const Self) bool { return self.net_position < 0; }
    pub fn is_flat(self: *const Self) bool  { return self.net_position == 0; }
};

// ============================================================
// Token bucket rate limiter
// ============================================================
pub const TokenBucket = struct {
    capacity:    f64,
    tokens:      f64,
    refill_rate: f64,  // tokens/ns
    last_refill: i64,

    pub fn init(capacity: f64, refill_rate_per_sec: f64) TokenBucket {
        return .{
            .capacity    = capacity,
            .tokens      = capacity,
            .refill_rate = refill_rate_per_sec / 1e9,
            .last_refill = 0,
        };
    }

    pub fn try_consume(self: *TokenBucket, now_ns: i64, tokens: f64) bool {
        if (self.last_refill > 0) {
            const elapsed = @as(f64, @floatFromInt(now_ns - self.last_refill));
            self.tokens = @min(self.capacity, self.tokens + elapsed * self.refill_rate);
        }
        self.last_refill = now_ns;
        if (self.tokens >= tokens) {
            self.tokens -= tokens;
            return true;
        }
        return false;
    }

    pub fn available(self: *const TokenBucket) f64 { return self.tokens; }
};

// ============================================================
// Pre-trade risk check result
// ============================================================
pub const CheckResult = struct {
    passed: bool,
    reason: []const u8,

    pub const ok = CheckResult{ .passed = true, .reason = "" };

    pub fn fail(reason: []const u8) CheckResult {
        return .{ .passed = false, .reason = reason };
    }
};

// ============================================================
// Risk Manager
// ============================================================
pub const RiskManager = struct {
    limits:          RiskLimits,
    positions:       AutoHashMap([16]u8, PositionState),
    session_pnl:     f64,
    peak_pnl:        f64,
    daily_loss:      f64,
    orders_this_sec: u64,
    orders_this_min: u64,
    cancels_total:   u64,
    orders_total:    u64,
    sec_bucket:      TokenBucket,
    min_bucket:      TokenBucket,
    kill_switch:     bool,
    avg_order_size:  f64,
    ref_prices:      AutoHashMap([16]u8, f64),
    allocator:       Allocator,
    violations:      u64,
    last_reset_ns:   i64,

    const Self = @This();

    pub fn init(allocator: Allocator, limits: RiskLimits) !Self {
        return Self{
            .limits          = limits,
            .positions       = AutoHashMap([16]u8, PositionState).init(allocator),
            .session_pnl     = 0.0,
            .peak_pnl        = 0.0,
            .daily_loss      = 0.0,
            .orders_this_sec = 0,
            .orders_this_min = 0,
            .cancels_total   = 0,
            .orders_total    = 0,
            .sec_bucket      = TokenBucket.init(@as(f64, @floatFromInt(limits.max_orders_per_sec)),
                                                @as(f64, @floatFromInt(limits.max_orders_per_sec))),
            .min_bucket      = TokenBucket.init(@as(f64, @floatFromInt(limits.max_orders_per_min)),
                                                @as(f64, @floatFromInt(limits.max_orders_per_min)) / 60.0),
            .kill_switch     = false,
            .avg_order_size  = 100.0,
            .ref_prices      = AutoHashMap([16]u8, f64).init(allocator),
            .allocator       = allocator,
            .violations      = 0,
            .last_reset_ns   = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.positions.deinit();
        self.ref_prices.deinit();
    }

    // Update reference (mid) price
    pub fn set_ref_price(self: *Self, sym: *const [16]u8, mid: f64) !void {
        try self.ref_prices.put(sym.*, mid);
    }

    // Pre-trade check: returns ok or failure reason
    pub fn check_order(self: *Self, sym: *const [16]u8, side: Side,
                       qty: u64, px: f64, now_ns: i64) CheckResult
    {
        if (self.kill_switch)
            return CheckResult.fail("kill_switch_active");

        // Quantity checks
        if (qty < self.limits.min_order_qty)
            return self.violation("qty_below_min");
        if (qty > self.limits.max_order_qty)
            return self.violation("qty_above_max");

        // Fat finger: qty vs avg
        if (@as(f64, @floatFromInt(qty)) > self.avg_order_size * self.limits.fat_finger_qty_mult)
            return self.violation("fat_finger_qty");

        // Price sanity
        if (px > 0) {
            const ref = self.ref_prices.get(sym.*) orelse 0.0;
            if (ref > 0) {
                const dev_bps = @fabs(px - ref) / ref * 10000.0;
                if (dev_bps > self.limits.fat_finger_px_mult * 10000.0)
                    return self.violation("fat_finger_price");
                if (dev_bps > self.limits.max_price_deviation_bps)
                    return self.violation("price_deviation");
            }
            const notional = @as(f64, @floatFromInt(qty)) * px;
            if (notional > self.limits.max_order_notional)
                return self.violation("notional_too_large");
        }

        // Position check
        if (self.positions.get(sym.*)) |pos| {
            const new_pos: i64 = pos.net_position +
                (if (side == .Buy) @as(i64, @intCast(qty)) else -@as(i64, @intCast(qty)));
            if (new_pos > self.limits.max_long_position)
                return self.violation("long_position_limit");
            if (new_pos < self.limits.max_short_position)
                return self.violation("short_position_limit");
        }

        // Loss limits
        if (self.session_pnl < -self.limits.max_session_loss)
            return self.violation("session_loss_limit");
        if (self.daily_loss > self.limits.max_daily_loss)
            return self.violation("daily_loss_limit");

        // Drawdown
        if (self.peak_pnl > 0) {
            const dd = (self.peak_pnl - self.session_pnl) / self.peak_pnl;
            if (dd > self.limits.max_drawdown_pct)
                return self.violation("drawdown_limit");
        }

        // Cancel ratio
        if (self.orders_total > 100) {
            const cancel_ratio = @as(f64, @floatFromInt(self.cancels_total)) /
                                  @as(f64, @floatFromInt(self.orders_total));
            if (cancel_ratio > self.limits.max_cancel_ratio)
                return self.violation("cancel_ratio_limit");
        }

        // Rate limiting
        if (!self.sec_bucket.try_consume(now_ns, 1.0))
            return self.violation("rate_limit_per_sec");
        if (!self.min_bucket.try_consume(now_ns, 1.0))
            return self.violation("rate_limit_per_min");

        // Update order size EMA
        const alpha: f64 = 0.01;
        self.avg_order_size = (1.0 - alpha) * self.avg_order_size +
                               alpha * @as(f64, @floatFromInt(qty));
        self.orders_total += 1;
        return CheckResult.ok;
    }

    fn violation(self: *Self, reason: []const u8) CheckResult {
        self.violations += 1;
        return CheckResult.fail(reason);
    }

    // Record a fill
    pub fn on_fill(self: *Self, sym: *const [16]u8, side: Side,
                   qty: u64, px: f64) !void
    {
        var entry = try self.positions.getOrPut(sym.*);
        if (!entry.found_existing) {
            entry.value_ptr.* = std.mem.zeroes(PositionState);
            entry.value_ptr.symbol = sym.*;
        }
        entry.value_ptr.on_fill(side, qty, px);

        // Update PnL
        self.session_pnl = blk: {
            var total: f64 = 0;
            var it = self.positions.valueIterator();
            while (it.next()) |pos| total += pos.net_pnl();
            break :blk total;
        };
        if (self.session_pnl > self.peak_pnl) self.peak_pnl = self.session_pnl;
        if (self.session_pnl < 0) self.daily_loss = -self.session_pnl;
    }

    pub fn on_cancel(self: *Self) void {
        self.cancels_total += 1;
    }

    pub fn mark_to_market(self: *Self, sym: *const [16]u8, mid: f64) void {
        if (self.positions.getPtr(sym.*)) |pos| {
            pos.mark_to_market(mid);
        }
    }

    pub fn engage_kill_switch(self: *Self) void {
        self.kill_switch = true;
    }

    pub fn reset_kill_switch(self: *Self) void {
        self.kill_switch = false;
    }

    pub fn is_active(self: *const Self) bool { return !self.kill_switch; }

    pub fn get_position(self: *const Self, sym: *const [16]u8) ?PositionState {
        return self.positions.get(sym.*);
    }

    pub fn total_exposure(self: *Self) f64 {
        var total: f64 = 0;
        var it = self.positions.valueIterator();
        while (it.next()) |pos| total += pos.gross_exposure;
        return total;
    }

    pub fn print_summary(self: *Self) void {
        std.debug.print(
            "=== Risk Manager Summary ===\n" ++
            "  Kill switch:  {}\n" ++
            "  Session PnL:  ${d:.2}\n" ++
            "  Peak PnL:     ${d:.2}\n" ++
            "  Daily loss:   ${d:.2}\n" ++
            "  Orders:       {d}\n" ++
            "  Cancels:      {d}\n" ++
            "  Violations:   {d}\n" ++
            "  Exposure:     ${d:.0}\n",
            .{
                self.kill_switch,
                self.session_pnl,
                self.peak_pnl,
                self.daily_loss,
                self.orders_total,
                self.cancels_total,
                self.violations,
                self.total_exposure(),
            });
    }
};

// ============================================================
// Spread monitor: tracks bid-ask spread health
// ============================================================
pub const SpreadMonitor = struct {
    symbol:     [16]u8,
    ema_spread: f64 = 0.0,
    min_spread: f64 = math.floatMax(f64),
    max_spread: f64 = 0.0,
    n_updates:  u64 = 0,
    alpha:      f64 = 0.05,

    const Self = @This();

    pub fn update(self: *Self, bid: f64, ask: f64) void {
        if (ask <= bid) return;
        const spread = ask - bid;
        if (self.n_updates == 0) {
            self.ema_spread = spread;
        } else {
            self.ema_spread = (1.0 - self.alpha) * self.ema_spread + self.alpha * spread;
        }
        self.min_spread = @min(self.min_spread, spread);
        self.max_spread = @max(self.max_spread, spread);
        self.n_updates += 1;
    }

    pub fn spread_bps(self: *const Self, mid: f64) f64 {
        if (mid <= 0) return 0;
        return self.ema_spread / mid * 10000.0;
    }

    pub fn is_wide(self: *const Self, mid: f64, threshold_bps: f64) bool {
        return self.spread_bps(mid) > threshold_bps;
    }
};

// ============================================================
// Portfolio risk aggregator
// ============================================================
pub const PortfolioRisk = struct {
    positions:       ArrayList(PositionState),
    total_long:      f64 = 0.0,
    total_short:     f64 = 0.0,
    net_exposure:    f64 = 0.0,
    gross_exposure:  f64 = 0.0,
    total_pnl:       f64 = 0.0,
    peak_pnl:        f64 = 0.0,
    max_drawdown:    f64 = 0.0,
    allocator:       Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .positions  = ArrayList(PositionState).init(allocator),
            .allocator  = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.positions.deinit();
    }

    pub fn update(self: *Self) void {
        self.total_long  = 0;
        self.total_short = 0;
        self.total_pnl   = 0;

        for (self.positions.items) |*pos| {
            const exposure = pos.gross_exposure;
            if (pos.net_position > 0)  self.total_long  += exposure;
            if (pos.net_position < 0)  self.total_short += exposure;
            self.total_pnl += pos.net_pnl();
        }
        self.net_exposure   = self.total_long - self.total_short;
        self.gross_exposure = self.total_long + self.total_short;

        if (self.total_pnl > self.peak_pnl) self.peak_pnl = self.total_pnl;
        if (self.peak_pnl > 0) {
            const dd = (self.peak_pnl - self.total_pnl) / self.peak_pnl;
            if (dd > self.max_drawdown) self.max_drawdown = dd;
        }
    }

    pub fn beta_adjusted_exposure(self: *const Self, market_beta: f64) f64 {
        return self.net_exposure * market_beta;
    }

    pub fn print(self: *const Self) void {
        std.debug.print(
            "Portfolio Risk:\n" ++
            "  Long:         ${d:.0}\n" ++
            "  Short:        ${d:.0}\n" ++
            "  Net:          ${d:.0}\n" ++
            "  Gross:        ${d:.0}\n" ++
            "  Total PnL:    ${d:.2}\n" ++
            "  Max Drawdown: {d:.2}%\n",
            .{
                self.total_long,
                self.total_short,
                self.net_exposure,
                self.gross_exposure,
                self.total_pnl,
                self.max_drawdown * 100.0,
            });
    }
};

// ============================================================
// Tests
// ============================================================
test "token bucket rate limiting" {
    var tb = TokenBucket.init(10.0, 10.0); // 10 per sec
    const t0: i64 = 1_000_000_000;
    // Consume full burst
    var i: usize = 0;
    while (i < 10) : (i += 1) try std.testing.expect(tb.try_consume(t0, 1.0));
    // Exhausted
    try std.testing.expect(!tb.try_consume(t0, 1.0));
    // Wait 1 second
    try std.testing.expect(tb.try_consume(t0 + 1_000_000_000, 5.0));
}

test "position state buy/sell" {
    var sym = [_]u8{'T','E','S','T',0}++[_]u8{0}**11;
    var pos = PositionState{ .symbol = sym };
    pos.on_fill(.Buy, 100, 100.0);
    try std.testing.expectEqual(@as(i64, 100), pos.net_position);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), pos.avg_cost, 1e-6);

    pos.on_fill(.Sell, 100, 101.0);
    try std.testing.expectEqual(@as(i64, 0), pos.net_position);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), pos.realized_pnl, 1e-4);
}

test "risk manager basic check" {
    var rm = try RiskManager.init(std.testing.allocator, RiskLimits{});
    defer rm.deinit();

    var sym = [_]u8{'T','E','S','T',0}++[_]u8{0}**11;
    const res = rm.check_order(&sym, .Buy, 100, 100.0, 1_000_000_000);
    try std.testing.expect(res.passed);
}

test "risk manager qty limit" {
    var rm = try RiskManager.init(std.testing.allocator, RiskLimits{ .max_order_qty = 500 });
    defer rm.deinit();

    var sym = [_]u8{'T','E','S','T',0}++[_]u8{0}**11;
    const res = rm.check_order(&sym, .Buy, 1000, 100.0, 1_000_000_000);
    try std.testing.expect(!res.passed);
}

test "risk manager kill switch" {
    var rm = try RiskManager.init(std.testing.allocator, RiskLimits{});
    defer rm.deinit();

    rm.engage_kill_switch();
    var sym = [_]u8{'T','E','S','T',0}++[_]u8{0}**11;
    const res = rm.check_order(&sym, .Buy, 100, 100.0, 1_000_000_000);
    try std.testing.expect(!res.passed);

    rm.reset_kill_switch();
    const res2 = rm.check_order(&sym, .Buy, 100, 100.0, 1_000_000_000);
    try std.testing.expect(res2.passed);
}

test "spread monitor" {
    var sym = [_]u8{'A','A','P','L',0}++[_]u8{0}**11;
    var mon = SpreadMonitor{ .symbol = sym };
    mon.update(182.48, 182.52);
    mon.update(182.47, 182.53);
    try std.testing.expect(mon.ema_spread > 0);
    try std.testing.expect(mon.n_updates == 2);
}

test "portfolio risk aggregation" {
    var port = PortfolioRisk.init(std.testing.allocator);
    defer port.deinit();

    var sym1 = [_]u8{'A','A','P','L',0}++[_]u8{0}**11;
    var pos1 = PositionState{ .symbol = sym1 };
    pos1.on_fill(.Buy, 100, 182.50);
    pos1.mark_to_market(183.0);
    try port.positions.append(pos1);

    port.update();
    try std.testing.expect(port.total_long > 0);
    try std.testing.expect(port.gross_exposure > 0);
}
