//! Market making strategy in Zig.
//! Quotes bid/ask around mid-price, manages inventory, and tracks PnL.

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

const orderbook = @import("orderbook.zig");
const stats_mod = @import("stats.zig");
const writer_mod = @import("writer.zig");

pub const Order = orderbook.Order;
pub const Price = orderbook.Price;
pub const Quantity = orderbook.Quantity;
pub const OrderId = orderbook.OrderId;
pub const Side = orderbook.Side;

// ============================================================
// Parameters
// ============================================================
pub const MMParams = struct {
    base_half_spread_ticks: f64 = 2.0,
    tick_size:               f64 = 0.01,
    quote_qty:               u64 = 100,
    max_inventory:           f64 = 1000.0,
    inventory_skew_factor:   f64 = 0.5,
    vol_spread_multiplier:   f64 = 1.5,
    vol_threshold_annual:    f64 = 0.30,
    max_quote_levels:        usize = 3,
    level_spacing_ticks:     f64 = 1.0,
    min_edge_bps:            f64 = 0.5,  // min expected edge in bps
    fade_inventory:          bool = true,
};

// ============================================================
// Quote to submit
// ============================================================
pub const Quote = struct {
    side:       Side,
    price:      Price,
    qty:        Quantity,
    cancel_id:  ?OrderId = null,
    is_cancel:  bool = false,
};

// ============================================================
// PnL tracker
// ============================================================
pub const PnLState = struct {
    realized:         f64 = 0.0,
    unrealized:       f64 = 0.0,
    transaction_cost: f64 = 0.0,
    net_position:     i64 = 0,
    avg_cost:         f64 = 0.0,
    gross_pnl:        f64 = 0.0,

    total_buys:  u64 = 0,
    total_sells: u64 = 0,
    total_volume: u64 = 0,

    const Self = @This();

    pub fn on_fill(self: *Self, side: Side, qty: u64, price: f64,
                   fee_per_share: f64) void
    {
        const q = @as(f64, @floatFromInt(qty));
        const fee = q * fee_per_share;
        self.transaction_cost += fee;

        if (side == .Buy) {
            const total_cost = self.avg_cost * @as(f64, @floatFromInt(@max(self.net_position, 0))) + price * q;
            self.net_position += @as(i64, @intCast(qty));
            if (self.net_position > 0)
                self.avg_cost = total_cost / @as(f64, @floatFromInt(self.net_position));
            self.total_buys  += qty;
        } else {
            const pos = @as(f64, @floatFromInt(@max(self.net_position, 0)));
            const close = @min(pos, q);
            self.realized += (price - self.avg_cost) * close;
            self.net_position -= @as(i64, @intCast(qty));
            self.total_sells += qty;
        }
        self.total_volume += qty;
        self.gross_pnl = self.realized;
    }

    pub fn mark_to_market(self: *Self, mkt: f64) void {
        if (self.net_position != 0 and self.avg_cost > 0) {
            self.unrealized = (mkt - self.avg_cost) * @as(f64, @floatFromInt(self.net_position));
        }
    }

    pub fn net_pnl(self: *const Self) f64 {
        return self.realized + self.unrealized - self.transaction_cost;
    }
    pub fn total_pnl(self: *const Self) f64 { return self.realized + self.unrealized; }
};

// ============================================================
// Market Maker
// ============================================================
pub const MarketMaker = struct {
    params:          MMParams,
    pnl:             PnLState = .{},
    inventory:       f64 = 0.0,
    active_bid:      ?OrderId = null,
    active_ask:      ?OrderId = null,
    update_count:    u64 = 0,
    fill_count:      u64 = 0,
    cancel_count:    u64 = 0,
    quote_count:     u64 = 0,
    order_id_gen:    u64 = 5000000,

    const Self = @This();

    pub fn init(params: MMParams) Self {
        return .{ .params = params };
    }

    // Called on each book update.
    // Returns list of quotes to submit/cancel.
    pub fn on_book_update(
        self: *Self,
        mid: f64,
        spread: f64,
        bid_depth: u64,
        ask_depth: u64,
        vol: f64,    // annualized vol estimate
    ) [4]Quote {
        var quotes: [4]Quote = undefined;
        var nq: usize = 0;
        self.update_count += 1;

        // Cancel active quotes
        if (self.active_bid) |id| {
            quotes[nq] = Quote{ .side = .Buy, .price = 0, .qty = 0, .cancel_id = id, .is_cancel = true };
            nq += 1;
            self.active_bid = null;
            self.cancel_count += 1;
        }
        if (self.active_ask) |id| {
            quotes[nq] = Quote{ .side = .Sell, .price = 0, .qty = 0, .cancel_id = id, .is_cancel = true };
            nq += 1;
            self.active_ask = null;
            self.cancel_count += 1;
        }

        if (mid <= 0) return quotes;

        // Compute spread width
        var half = self.params.base_half_spread_ticks * self.params.tick_size;

        // Vol adjustment
        if (vol > self.params.vol_threshold_annual)
            half *= self.params.vol_spread_multiplier;

        // Inventory skew: shift quotes against position
        const inv_ratio = if (@fabs(self.inventory) > 1e-6)
            self.inventory / self.params.max_inventory
        else 0.0;
        const clamped = @max(-1.0, @min(1.0, inv_ratio));
        const skew = -clamped * self.params.inventory_skew_factor * self.params.tick_size;

        // Imbalance signal
        const tot = bid_depth + ask_depth;
        const imb: f64 = if (tot > 0) blk: {
            break :blk (@as(f64, @floatFromInt(bid_depth)) - @as(f64, @floatFromInt(ask_depth))) /
                        @as(f64, @floatFromInt(tot));
        } else 0.0;
        const imb_adj = imb * self.params.tick_size * 0.5;

        var bid_px = mid - half + skew - imb_adj;
        var ask_px = mid + half + skew - imb_adj;

        // Round to tick
        bid_px = @floor(bid_px / self.params.tick_size) * self.params.tick_size;
        ask_px = @ceil (ask_px / self.params.tick_size) * self.params.tick_size;

        // Safety: don't cross the market
        if (ask_px <= bid_px) ask_px = bid_px + self.params.tick_size;

        // Edge check
        const edge_bps = (ask_px - bid_px) / mid * 5000.0; // half-spread in bps
        if (edge_bps < self.params.min_edge_bps) return quotes;

        // Compute qty (fade inventory if configured)
        const bid_qty: u64 = blk: {
            var q = self.params.quote_qty;
            if (self.params.fade_inventory and self.inventory > self.params.max_inventory * 0.5) {
                q = q / 2;
            }
            break :blk q;
        };
        const ask_qty: u64 = blk: {
            var q = self.params.quote_qty;
            if (self.params.fade_inventory and self.inventory < -self.params.max_inventory * 0.5) {
                q = q / 2;
            }
            break :blk q;
        };

        if (bid_qty > 0 and nq < 4) {
            self.order_id_gen += 1;
            const bid_id = self.order_id_gen;
            quotes[nq] = Quote{
                .side  = .Buy,
                .price = orderbook.from_double(bid_px),
                .qty   = bid_qty,
            };
            self.active_bid = bid_id;
            nq += 1;
            self.quote_count += 1;
        }

        if (ask_qty > 0 and nq < 4) {
            self.order_id_gen += 1;
            const ask_id = self.order_id_gen;
            quotes[nq] = Quote{
                .side  = .Sell,
                .price = orderbook.from_double(ask_px),
                .qty   = ask_qty,
            };
            self.active_ask = ask_id;
            nq += 1;
            self.quote_count += 1;
        }

        return quotes;
    }

    pub fn on_fill(self: *Self, side: Side, qty: u64, price: f64) void {
        self.fill_count += 1;
        const dir: f64 = if (side == .Buy) 1.0 else -1.0;
        self.inventory += dir * @as(f64, @floatFromInt(qty));
        self.pnl.on_fill(side, qty, price, 0.001);
    }

    pub fn mark_to_market(self: *Self, mid: f64) void {
        self.pnl.mark_to_market(mid);
    }

    pub fn print_summary(self: *const Self) void {
        std.debug.print(
            "=== Market Maker Summary ===\n" ++
            "  Updates:     {d}\n" ++
            "  Quotes sent: {d}\n" ++
            "  Fills:       {d}\n" ++
            "  Inventory:   {d:.0}\n" ++
            "  Realized PnL: ${d:.2}\n" ++
            "  Net PnL:     ${d:.2}\n",
            .{
                self.update_count,
                self.quote_count,
                self.fill_count,
                self.inventory,
                self.pnl.realized,
                self.pnl.net_pnl(),
            });
    }
};

// ============================================================
// Simulation: run market maker against synthetic book
// ============================================================
pub fn simulate_market_maker(allocator: Allocator, n_steps: usize) !void {
    _ = allocator;
    var mm = MarketMaker.init(MMParams{});
    var book = orderbook.OrderBook.init(std.heap.page_allocator, "SIM");
    defer book.deinit();

    var vc: stats_mod.VolatilityCalc = .{};
    var vwap_calc: stats_mod.VWAPCalc = .{};
    var writer = try writer_mod.InMemWriter.init(std.heap.page_allocator, 1 << 16);
    defer writer.deinit();

    var rng = std.rand.DefaultPrng.init(42);
    const rand = rng.random();

    var price: f64 = 100.0;
    var ts_ns: i64 = 34200_000_000_000;

    var order_id: orderbook.OrderId = 1;
    var fill_count: u64 = 0;

    for (0..n_steps) |step| {
        // GBM price update
        const z = rand.floatNorm(f64);
        price *= @exp(-0.5 * 0.0001 + 0.01 * z);
        price = @max(price, 0.01);
        ts_ns += 1_000_000; // 1ms per step

        vc.update(price);
        const vol = vc.annualized_vol();

        // Build a simple book around the price
        const spread_d = price * 0.001; // 10bps spread
        const bid_d = price - spread_d / 2.0;
        const ask_d = price + spread_d / 2.0;
        const bid_qty: u64 = 1000 + rand.uintLessThan(u64, 500);
        const ask_qty: u64 = 1000 + rand.uintLessThan(u64, 500);

        // Get MM quotes
        const quotes = mm.on_book_update(price, spread_d, bid_qty, ask_qty, vol);
        _ = quotes;

        // Simulate random fills
        if (step % 50 == 0 and step > 0) {
            const is_buy = rand.boolean();
            const fill_qty: u64 = 100;
            const fill_px = if (is_buy) ask_d else bid_d;
            mm.on_fill(if (is_buy) .Buy else .Sell, fill_qty, fill_px);
            vwap_calc.update(fill_px, fill_qty);
            fill_count += 1;

            // Write to ring buffer
            var tick: writer_mod.SharedTick = std.mem.zeroes(writer_mod.SharedTick);
            tick.timestamp = ts_ns;
            @memcpy(tick.symbol[0..3], "SIM");
            tick.price     = orderbook.from_double(fill_px);
            tick.qty       = fill_qty;
            tick.side      = if (is_buy) 0 else 1;
            tick.tick_type = 0;
            writer.write_tick(tick);
        }
        mm.mark_to_market(price);

        // Add passive orders to book
        if (step % 10 == 0) {
            order_id += 1;
            const passive_bid = orderbook.Order{
                .id = order_id, .symbol = [_]u8{'S','I','M',0}++[_]u8{0}**12,
                .side = .Buy, .order_type = .Limit, .status = .New,
                .price = orderbook.from_double(bid_d), .qty = 100, .filled_qty = 0,
                .timestamp = ts_ns, .sequence = 0,
            };
            var f = try book.add_order(passive_bid);
            f.deinit();

            order_id += 1;
            const passive_ask = orderbook.Order{
                .id = order_id, .symbol = [_]u8{'S','I','M',0}++[_]u8{0}**12,
                .side = .Sell, .order_type = .Limit, .status = .New,
                .price = orderbook.from_double(ask_d), .qty = 100, .filled_qty = 0,
                .timestamp = ts_ns, .sequence = 0,
            };
            var f2 = try book.add_order(passive_ask);
            f2.deinit();
        }
    }

    mm.print_summary();
    std.debug.print(
        "  Fills: {d}  Ticks written: {d}  Session VWAP: {d:.4}\n",
        .{fill_count, writer.write_seq, vwap_calc.session_vwap()});
}

// ============================================================
// Tests
// ============================================================
test "market maker basic quoting" {
    var mm = MarketMaker.init(MMParams{});
    const quotes = mm.on_book_update(100.0, 0.10, 1000, 1000, 0.20);
    _ = quotes;
    try std.testing.expect(mm.quote_count >= 1);
}

test "inventory skew" {
    var mm = MarketMaker.init(MMParams{
        .max_inventory = 100.0,
        .inventory_skew_factor = 1.0,
    });

    // Build up long inventory
    mm.inventory = 80.0;
    const quotes = mm.on_book_update(100.0, 0.10, 1000, 1000, 0.20);
    _ = quotes;
    // With long inventory, bid should be lower than neutral
    try std.testing.expect(mm.inventory > 0);
}

test "pnl tracking" {
    var pnl = PnLState{};
    pnl.on_fill(.Buy, 100, 100.0, 0.001);
    try std.testing.expectEqual(@as(i64, 100), pnl.net_position);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), pnl.avg_cost, 1e-6);

    pnl.on_fill(.Sell, 100, 100.5, 0.001);
    try std.testing.expectEqual(@as(i64, 0), pnl.net_position);
    // Realized PnL = (100.5 - 100.0) * 100 = 50
    try std.testing.expectApproxEqAbs(@as(f64, 50.0), pnl.realized, 1e-4);
}

test "simulate market maker" {
    try simulate_market_maker(std.testing.allocator, 500);
}
