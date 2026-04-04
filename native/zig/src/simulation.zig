//! Full end-to-end simulation: synthetic ITCH feed → decoder → order book → stats → market maker.
//! Wires all Zig modules together in a realistic market session replay.

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;

const decoder_mod = @import("decoder.zig");
const orderbook   = @import("orderbook.zig");
const feed_mod    = @import("feed.zig");
const stats_mod   = @import("stats.zig");
const writer_mod  = @import("writer.zig");
const mm_mod      = @import("market_maker.zig");
const alloc_mod   = @import("allocator.zig");

pub const Price    = orderbook.Price;
pub const Quantity = orderbook.Quantity;
pub const OrderId  = orderbook.OrderId;
pub const Side     = orderbook.Side;

// ============================================================
// Simulation Configuration
// ============================================================
pub const SimConfig = struct {
    n_steps:             usize = 100_000,
    n_symbols:           usize = 5,
    initial_price:       f64 = 100.0,
    daily_vol:           f64 = 0.20,
    tick_size:           f64 = 0.01,
    seed:                u64 = 42,
    fee_per_share:       f64 = 0.001,
    mm_half_spread_ticks: f64 = 2.0,
    mm_quote_qty:        u64 = 100,
    ring_capacity:       usize = 1 << 16,
    print_interval:      usize = 10_000,
    enable_mm:           bool = true,
    enable_vwap:         bool = true,
    enable_vol_calc:     bool = true,
};

// ============================================================
// Per-symbol simulation state
// ============================================================
const SymbolState = struct {
    symbol:     [16]u8,
    book:       orderbook.OrderBook,
    stats:      stats_mod.MarketStats,
    mm:         mm_mod.MarketMaker,
    price:      f64,
    order_id:   OrderId,
    fill_count: u64,
    vol_calc:   stats_mod.VolatilityCalc,

    const Self = @This();

    pub fn init(allocator: Allocator, sym: []const u8, price: f64,
                mm_params: mm_mod.MMParams) !Self
    {
        var sym_buf = [_]u8{0} ** 16;
        const copy_len = @min(sym.len, 15);
        @memcpy(sym_buf[0..copy_len], sym[0..copy_len]);

        return Self{
            .symbol   = sym_buf,
            .book     = orderbook.OrderBook.init(allocator, sym),
            .stats    = stats_mod.MarketStats.init(),
            .mm       = mm_mod.MarketMaker.init(mm_params),
            .price    = price,
            .order_id = 1,
            .fill_count = 0,
            .vol_calc = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        self.book.deinit();
    }

    pub fn symbol_name(self: *const Self) []const u8 {
        var len: usize = 0;
        while (len < 16 and self.symbol[len] != 0) len += 1;
        return self.symbol[0..len];
    }
};

// ============================================================
// Session metrics aggregated across all symbols
// ============================================================
pub const SessionMetrics = struct {
    total_messages:    u64 = 0,
    total_orders:      u64 = 0,
    total_fills:       u64 = 0,
    total_cancels:     u64 = 0,
    total_quotes:      u64 = 0,
    total_volume_usd:  f64 = 0.0,
    total_pnl:         f64 = 0.0,
    total_fees:        f64 = 0.0,
    wall_time_us:      u64 = 0,
    ticks_written:     u64 = 0,

    pub fn print(self: *const SessionMetrics) void {
        std.debug.print(
            "\n╔══════════════════════════════════════════╗\n" ++
            "║          Session Summary                 ║\n" ++
            "╠══════════════════════════════════════════╣\n" ++
            "  Messages processed: {d}\n" ++
            "  Orders placed:      {d}\n" ++
            "  Fills:              {d}\n" ++
            "  Cancels:            {d}\n" ++
            "  MM Quotes:          {d}\n" ++
            "  Volume (USD):       ${d:.2}\n" ++
            "  Realized PnL:       ${d:.2}\n" ++
            "  Fees paid:          ${d:.2}\n" ++
            "  Net PnL:            ${d:.2}\n" ++
            "  Ticks written:      {d}\n" ++
            "  Wall time:          {d} µs\n" ++
            "╚══════════════════════════════════════════╝\n",
            .{
                self.total_messages,
                self.total_orders,
                self.total_fills,
                self.total_cancels,
                self.total_quotes,
                self.total_volume_usd,
                self.total_pnl,
                self.total_fees,
                self.total_pnl - self.total_fees,
                self.ticks_written,
                self.wall_time_us,
            });
    }
};

// ============================================================
// Pseudo-random number generator (xorshift64)
// ============================================================
const RNG = struct {
    state: u64,

    pub fn init(seed: u64) RNG { return .{ .state = seed | 1 }; }

    pub fn next(self: *RNG) u64 {
        var x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        return x;
    }

    // Returns f64 in [0, 1)
    pub fn f64_(self: *RNG) f64 {
        return @as(f64, @floatFromInt(self.next() >> 11)) / @as(f64, 1 << 53);
    }

    // Box-Muller normal sample
    pub fn normal(self: *RNG) f64 {
        const u1 = self.f64_() + 1e-300;
        const u2 = self.f64_();
        return @sqrt(-2.0 * @log(u1)) * @cos(2.0 * math.pi * u2);
    }

    pub fn range(self: *RNG, lo: u64, hi: u64) u64 {
        return lo + self.next() % (hi - lo);
    }
};

// ============================================================
// Synthetic ITCH message generator
// ============================================================
fn build_add_order(
    buf: []u8,
    order_id: u64,
    side: u8,  // 'B' or 'S'
    shares: u32,
    symbol: *const [8]u8,
    price_fixed: u32,  // 4 decimal places (ITCH convention)
    ts_ns: u64,
) usize
{
    // Message type 'A', total payload 36 bytes
    if (buf.len < 36) return 0;
    buf[0]  = 'A';
    buf[1]  = 0; buf[2] = 0;                   // stock_locate
    buf[3]  = 0; buf[4] = 0;                   // tracking_number
    // 6-byte timestamp
    buf[5]  = @truncate((ts_ns >> 40) & 0xFF);
    buf[6]  = @truncate((ts_ns >> 32) & 0xFF);
    buf[7]  = @truncate((ts_ns >> 24) & 0xFF);
    buf[8]  = @truncate((ts_ns >> 16) & 0xFF);
    buf[9]  = @truncate((ts_ns >>  8) & 0xFF);
    buf[10] = @truncate( ts_ns        & 0xFF);
    // 8-byte order reference
    buf[11] = @truncate((order_id >> 56) & 0xFF);
    buf[12] = @truncate((order_id >> 48) & 0xFF);
    buf[13] = @truncate((order_id >> 40) & 0xFF);
    buf[14] = @truncate((order_id >> 32) & 0xFF);
    buf[15] = @truncate((order_id >> 24) & 0xFF);
    buf[16] = @truncate((order_id >> 16) & 0xFF);
    buf[17] = @truncate((order_id >>  8) & 0xFF);
    buf[18] = @truncate( order_id        & 0xFF);
    buf[19] = side;
    // 4-byte shares
    buf[20] = @truncate((shares >> 24) & 0xFF);
    buf[21] = @truncate((shares >> 16) & 0xFF);
    buf[22] = @truncate((shares >>  8) & 0xFF);
    buf[23] = @truncate( shares        & 0xFF);
    // 8-byte stock
    @memcpy(buf[24..32], symbol);
    // 4-byte price (ITCH 4 decimal places)
    buf[32] = @truncate((price_fixed >> 24) & 0xFF);
    buf[33] = @truncate((price_fixed >> 16) & 0xFF);
    buf[34] = @truncate((price_fixed >>  8) & 0xFF);
    buf[35] = @truncate( price_fixed        & 0xFF);
    return 36;
}

fn build_execute(buf: []u8, order_id: u64, exec_shares: u32, match_num: u64, ts_ns: u64) usize {
    if (buf.len < 31) return 0;
    buf[0]  = 'E';
    buf[1]  = 0; buf[2] = 0;
    buf[3]  = 0; buf[4] = 0;
    buf[5]  = @truncate((ts_ns >> 40) & 0xFF);
    buf[6]  = @truncate((ts_ns >> 32) & 0xFF);
    buf[7]  = @truncate((ts_ns >> 24) & 0xFF);
    buf[8]  = @truncate((ts_ns >> 16) & 0xFF);
    buf[9]  = @truncate((ts_ns >>  8) & 0xFF);
    buf[10] = @truncate( ts_ns        & 0xFF);
    buf[11] = @truncate((order_id >> 56) & 0xFF);
    buf[12] = @truncate((order_id >> 48) & 0xFF);
    buf[13] = @truncate((order_id >> 40) & 0xFF);
    buf[14] = @truncate((order_id >> 32) & 0xFF);
    buf[15] = @truncate((order_id >> 24) & 0xFF);
    buf[16] = @truncate((order_id >> 16) & 0xFF);
    buf[17] = @truncate((order_id >>  8) & 0xFF);
    buf[18] = @truncate( order_id        & 0xFF);
    buf[19] = @truncate((exec_shares >> 24) & 0xFF);
    buf[20] = @truncate((exec_shares >> 16) & 0xFF);
    buf[21] = @truncate((exec_shares >>  8) & 0xFF);
    buf[22] = @truncate( exec_shares        & 0xFF);
    buf[23] = @truncate((match_num >> 56) & 0xFF);
    buf[24] = @truncate((match_num >> 48) & 0xFF);
    buf[25] = @truncate((match_num >> 40) & 0xFF);
    buf[26] = @truncate((match_num >> 32) & 0xFF);
    buf[27] = @truncate((match_num >> 24) & 0xFF);
    buf[28] = @truncate((match_num >> 16) & 0xFF);
    buf[29] = @truncate((match_num >>  8) & 0xFF);
    buf[30] = @truncate( match_num        & 0xFF);
    return 31;
}

fn build_cancel(buf: []u8, order_id: u64, cancel_shares: u32, ts_ns: u64) usize {
    if (buf.len < 23) return 0;
    buf[0]  = 'X';
    buf[1]  = 0; buf[2] = 0;
    buf[3]  = 0; buf[4] = 0;
    buf[5]  = @truncate((ts_ns >> 40) & 0xFF);
    buf[6]  = @truncate((ts_ns >> 32) & 0xFF);
    buf[7]  = @truncate((ts_ns >> 24) & 0xFF);
    buf[8]  = @truncate((ts_ns >> 16) & 0xFF);
    buf[9]  = @truncate((ts_ns >>  8) & 0xFF);
    buf[10] = @truncate( ts_ns        & 0xFF);
    buf[11] = @truncate((order_id >> 56) & 0xFF);
    buf[12] = @truncate((order_id >> 48) & 0xFF);
    buf[13] = @truncate((order_id >> 40) & 0xFF);
    buf[14] = @truncate((order_id >> 32) & 0xFF);
    buf[15] = @truncate((order_id >> 24) & 0xFF);
    buf[16] = @truncate((order_id >> 16) & 0xFF);
    buf[17] = @truncate((order_id >>  8) & 0xFF);
    buf[18] = @truncate( order_id        & 0xFF);
    buf[19] = @truncate((cancel_shares >> 24) & 0xFF);
    buf[20] = @truncate((cancel_shares >> 16) & 0xFF);
    buf[21] = @truncate((cancel_shares >>  8) & 0xFF);
    buf[22] = @truncate( cancel_shares        & 0xFF);
    return 23;
}

// ============================================================
// Engine: orchestrates all symbols
// ============================================================
pub const Engine = struct {
    cfg:        SimConfig,
    allocator:  Allocator,
    decoder:    decoder_mod.Decoder,
    writer:     writer_mod.InMemWriter,
    symbols:    ArrayList(SymbolState),
    metrics:    SessionMetrics,
    rng:        RNG,

    const Self = @This();

    pub fn init(allocator: Allocator, cfg: SimConfig) !Self {
        const writer = try writer_mod.InMemWriter.init(allocator, cfg.ring_capacity);
        var symbols  = ArrayList(SymbolState).init(allocator);

        const sym_names = [_][]const u8{ "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
                                          "META", "TSLA", "NFLX", "AMD",  "INTC" };

        const mm_params = mm_mod.MMParams{
            .base_half_spread_ticks = cfg.mm_half_spread_ticks,
            .tick_size              = cfg.tick_size,
            .quote_qty              = cfg.mm_quote_qty,
            .max_inventory          = 5000.0,
            .vol_spread_multiplier  = 1.5,
        };

        var price = cfg.initial_price;
        for (0..@min(cfg.n_symbols, sym_names.len)) |i| {
            const state = try SymbolState.init(allocator, sym_names[i], price, mm_params);
            try symbols.append(state);
            price *= 1.15 + @as(f64, @floatFromInt(i)) * 0.1; // different prices per symbol
        }

        return Self{
            .cfg       = cfg,
            .allocator = allocator,
            .decoder   = decoder_mod.Decoder.init(),
            .writer    = writer,
            .symbols   = symbols,
            .metrics   = .{},
            .rng       = RNG.init(cfg.seed),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.symbols.items) |*s| s.deinit();
        self.symbols.deinit();
        self.writer.deinit();
    }

    // Process one AddOrder ITCH message for a symbol
    fn process_add_order(self: *Self, sym_idx: usize, msg: decoder_mod.AddOrder, ts_ns: i64) !void {
        var s = &self.symbols.items[sym_idx];

        const side: Side = if (msg.side == 'B') .Buy else .Sell;
        const price_f = @as(f64, @floatFromInt(msg.price)) / 10000.0;
        const price_p = orderbook.from_double(price_f);

        const order = orderbook.Order{
            .id         = msg.order_reference,
            .symbol     = s.symbol,
            .side       = side,
            .order_type = .Limit,
            .status     = .New,
            .price      = price_p,
            .qty        = msg.shares,
            .filled_qty = 0,
            .timestamp  = ts_ns,
            .sequence   = 0,
        };

        var fills = try s.book.add_order(order);
        defer fills.deinit();

        self.metrics.total_orders += 1;

        // Record fills
        for (fills.items) |fill| {
            const fill_px = orderbook.to_double(fill.price);
            s.stats.vwap.update(fill_px, fill.qty);
            s.mm.on_fill(fill.side, fill.qty, fill_px);
            s.fill_count += 1;
            self.metrics.total_fills += 1;
            self.metrics.total_volume_usd += fill_px * @as(f64, @floatFromInt(fill.qty));
            self.metrics.total_pnl = blk: {
                var total: f64 = 0;
                for (self.symbols.items) |*sym| total += sym.mm.pnl.realized;
                break :blk total;
            };
            self.metrics.total_fees += @as(f64, @floatFromInt(fill.qty)) * self.cfg.fee_per_share;

            // Write to ring buffer
            var tick: writer_mod.SharedTick = std.mem.zeroes(writer_mod.SharedTick);
            tick.timestamp = ts_ns;
            @memcpy(tick.symbol[0..4], s.symbol[0..4]);
            tick.price    = fill.price;
            tick.qty      = fill.qty;
            tick.side     = if (fill.side == .Buy) 0 else 1;
            tick.tick_type = 0;
            self.writer.write_tick(tick);
            self.metrics.ticks_written += 1;
        }

        // Update stats
        const mid_price = orderbook.to_double(s.book.mid_price() orelse price_p);
        s.vol_calc.update(mid_price);
        s.stats.update_mid(mid_price, ts_ns);
        self.metrics.total_messages += 1;
    }

    // Run market maker on current book state
    fn run_mm(self: *Self, sym_idx: usize, ts_ns: i64) !void {
        _ = ts_ns;
        var s = &self.symbols.items[sym_idx];

        const mid_opt = s.book.mid_price();
        if (mid_opt == null) return;

        const mid_f     = orderbook.to_double(mid_opt.?);
        const spread_f  = orderbook.to_double(s.book.spread() orelse @as(Price, 0));
        const bid_depth = s.book.bid_total_qty();
        const ask_depth = s.book.ask_total_qty();
        const vol       = s.vol_calc.annualized_vol();

        const quotes = s.mm.on_book_update(mid_f, spread_f, bid_depth, ask_depth, vol);
        _ = quotes;
        self.metrics.total_quotes += s.mm.quote_count;
    }

    // Main simulation loop
    pub fn run(self: *Self) !SessionMetrics {
        const t_start = std.time.nanoTimestamp();

        const dt = 1.0 / (252.0 * 6.5 * 3600.0); // 1 second in trading years
        var ts_ns: i64 = 34_200_000_000_000; // 9:30 AM

        // Pre-allocate message buffer
        var msg_buf: [64]u8 = undefined;

        var match_num: u64 = 1;
        var sym_order_ids = [_]OrderId{1} ** 10;

        for (0..self.cfg.n_steps) |step| {
            ts_ns += 1_000_000; // 1ms per step

            // Update price for each symbol
            for (0..self.symbols.items.len) |si| {
                var s = &self.symbols.items[si];

                // GBM step
                const z     = self.rng.normal();
                const sigma = self.cfg.daily_vol / @sqrt(252.0 * 6.5 * 3600.0);
                const ret   = -0.5 * sigma * sigma * dt + sigma * z;
                s.price = @max(s.price * @exp(ret), 0.01);

                // Build ITCH add-order message (bid side)
                const half_sp = s.price * 0.001;
                const is_buy  = self.rng.f64_() < 0.5;
                const px_d    = if (is_buy)
                    s.price - half_sp - self.rng.f64_() * half_sp
                else
                    s.price + half_sp + self.rng.f64_() * half_sp;

                const px_itch = @as(u32, @intFromFloat(@max(px_d * 10000.0, 0.0)));
                const shares  = @as(u32, @intCast(self.rng.range(50, 500)));
                const side_ch: u8 = if (is_buy) 'B' else 'S';

                sym_order_ids[si] += 1;
                var sym8 = [_]u8{' '} ** 8;
                const sn = s.symbol_name();
                @memcpy(sym8[0..@min(sn.len, 8)], sn[0..@min(sn.len, 8)]);

                const msg_len = build_add_order(
                    &msg_buf, sym_order_ids[si], side_ch, shares, &sym8, px_itch,
                    @as(u64, @intCast(ts_ns)));

                if (msg_len > 0) {
                    const decoded = self.decoder.decode_one(msg_buf[0..msg_len]);
                    if (decoded) |msg| {
                        if (msg == .add_order) {
                            try self.process_add_order(si, msg.add_order, ts_ns);
                        }
                    }
                }

                // Occasional execution (simulated trade)
                if (step % 30 == 0) {
                    const exec_shares: u32 = 100;
                    const exec_len = build_execute(
                        &msg_buf, sym_order_ids[si] - 1, exec_shares, match_num, @as(u64, @intCast(ts_ns)));
                    match_num += 1;
                    if (exec_len > 0) {
                        const decoded = self.decoder.decode_one(msg_buf[0..exec_len]);
                        if (decoded) |msg| {
                            _ = msg; // execution handling
                        }
                    }
                }

                // Occasional cancel
                if (step % 50 == 0 and sym_order_ids[si] > 5) {
                    const cancel_len = build_cancel(
                        &msg_buf, sym_order_ids[si] - 2, 50, @as(u64, @intCast(ts_ns)));
                    if (cancel_len > 0) {
                        const decoded = self.decoder.decode_one(msg_buf[0..cancel_len]);
                        if (decoded) |msg| {
                            _ = msg; // cancel handling
                            self.metrics.total_cancels += 1;
                        }
                    }
                }

                // Run market maker
                if (self.cfg.enable_mm) {
                    try self.run_mm(si, ts_ns);
                }

                // Mark to market
                s.mm.mark_to_market(s.price);
            }

            // Periodic progress report
            if (step > 0 and step % self.cfg.print_interval == 0) {
                const pct = step * 100 / self.cfg.n_steps;
                std.debug.print("  Step {d}/{d} ({d}%)  fills={d}  ticks={d}\n",
                    .{ step, self.cfg.n_steps, pct,
                       self.metrics.total_fills, self.metrics.ticks_written });
            }
        }

        const t_end   = std.time.nanoTimestamp();
        self.metrics.wall_time_us = @as(u64, @intCast(@divTrunc(t_end - t_start, 1000)));

        return self.metrics;
    }

    // Print per-symbol summary
    pub fn print_symbol_summary(self: *const Self) void {
        std.debug.print("\n--- Per-Symbol Summary ---\n", .{});
        std.debug.print("{s:<6}  {s:>10}  {s:>8}  {s:>8}  {s:>8}  {s:>8}\n",
            .{ "Symbol", "Price", "Fills", "InvPos", "RealPnL", "NetPnL" });
        std.debug.print("{s}\n", .{"-" ** 60});
        for (self.symbols.items) |*s| {
            const sym = s.symbol_name();
            std.debug.print("{s:<6}  {d:>10.2}  {d:>8}  {d:>8.0}  {d:>8.2}  {d:>8.2}\n",
                .{
                    sym,
                    s.price,
                    s.fill_count,
                    s.mm.inventory,
                    s.mm.pnl.realized,
                    s.mm.pnl.net_pnl(),
                });
        }
    }
};

// ============================================================
// Convenience: run a full simulation and return metrics
// ============================================================
pub fn run_simulation(allocator: Allocator, cfg: SimConfig) !SessionMetrics {
    var engine = try Engine.init(allocator, cfg);
    defer engine.deinit();

    std.debug.print("Starting simulation: {d} steps, {d} symbols\n",
        .{ cfg.n_steps, cfg.n_symbols });

    const metrics = try engine.run();
    engine.print_symbol_summary();
    metrics.print();
    return metrics;
}

// ============================================================
// Stress test: many small allocations, arena vs pool
// ============================================================
pub fn stress_test_allocators(allocator: Allocator) !void {
    std.debug.print("\n--- Allocator Stress Test ---\n", .{});

    // Arena allocator test
    {
        var arena = alloc_mod.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const alloc = arena.allocator();

        var total: usize = 0;
        for (0..10_000) |i| {
            const sz = (i % 128) + 1;
            const buf = try alloc.alloc(u8, sz);
            @memset(buf, @truncate(i));
            total += sz;
        }
        std.debug.print("  Arena: 10K allocs, {d} bytes total\n", .{total});
    }

    // Pool allocator test
    {
        var pool = alloc_mod.PoolAllocator(orderbook.Order, 4096).init();
        var ptrs: [64]*orderbook.Order = undefined;
        for (0..64) |i| {
            ptrs[i] = try pool.alloc();
            ptrs[i].id = @as(OrderId, @intCast(i + 1));
        }
        for (ptrs) |ptr| pool.free(ptr);
        std.debug.print("  Pool: 64 Order alloc/free cycles\n", .{});
    }

    // Slab allocator test
    {
        var slab = alloc_mod.SlabAllocator.init(allocator);
        defer slab.deinit();
        const alloc = slab.allocator();

        var bufs: [256][]u8 = undefined;
        for (0..256) |i| {
            const sz = [_]usize{ 32, 64, 128, 256 }[i % 4];
            bufs[i] = try alloc.alloc(u8, sz);
            @memset(bufs[i], 0xAB);
        }
        for (bufs) |buf| alloc.free(buf);
        std.debug.print("  Slab: 256 alloc/free cycles across 4 size classes\n", .{});
    }
}

// ============================================================
// Feed handler integration test
// ============================================================
pub fn test_feed_handler(allocator: Allocator) !void {
    std.debug.print("\n--- Feed Handler Test ---\n", .{});

    var fh = feed_mod.FeedHandler.init();
    var dec = decoder_mod.Decoder.init();
    var book = orderbook.OrderBook.init(allocator, "FEED");
    defer book.deinit();

    var msg_buf: [64]u8 = undefined;
    var seq: u32 = 1;
    var ts_ns: u64 = 34_200_000_000_000;
    var sym8 = [_]u8{' '} ** 8;
    @memcpy(sym8[0..4], "FEED");

    var orders_added: usize = 0;
    var gaps_detected: usize = 0;

    for (0..1000) |i| {
        ts_ns += 1_000_000;

        // Simulate occasional gap (skip every 100th sequence number)
        if (i > 0 and i % 100 == 0) {
            seq += 1;
            gaps_detected += 1;
        }

        const px_itch = @as(u32, @intFromFloat(100.0 * 10000.0 + @as(f64, @floatFromInt(i % 100)) * 100.0));
        const len = build_add_order(&msg_buf, i + 1, 'B', 100, &sym8, px_itch, ts_ns);
        if (len == 0) continue;

        const decoded = dec.decode_one(msg_buf[0..len]);
        if (decoded == null) continue;
        const msg = decoded.?;

        const status = fh.process_message(seq, ts_ns, msg);
        seq += 1;

        switch (status) {
            .ok => {
                if (msg == .add_order) {
                    const ao = msg.add_order;
                    const order = orderbook.Order{
                        .id         = ao.order_reference,
                        .symbol     = [_]u8{'F','E','E','D',0}++[_]u8{0}**11,
                        .side       = if (ao.side == 'B') .Buy else .Sell,
                        .order_type = .Limit,
                        .status     = .New,
                        .price      = orderbook.from_double(@as(f64, @floatFromInt(ao.price)) / 10000.0),
                        .qty        = ao.shares,
                        .filled_qty = 0,
                        .timestamp  = @as(i64, @intCast(ts_ns)),
                        .sequence   = 0,
                    };
                    var fills = try book.add_order(order);
                    fills.deinit();
                    orders_added += 1;
                }
            },
            .gap => {},
            else => {},
        }
    }

    std.debug.print("  Orders added: {d}\n", .{orders_added});
    std.debug.print("  Gaps detected: {d}\n", .{gaps_detected});
    std.debug.print("  Bid levels: {d}  Ask levels: {d}\n",
        .{ book.bid_levels(), book.ask_levels() });
}

// ============================================================
// Tests
// ============================================================
test "simulation engine init and small run" {
    const cfg = SimConfig{
        .n_steps   = 100,
        .n_symbols = 2,
        .print_interval = 1000,
    };
    var engine = try Engine.init(std.testing.allocator, cfg);
    defer engine.deinit();
    const metrics = try engine.run();
    try std.testing.expect(metrics.total_messages >= 0);
}

test "feed handler integration" {
    try test_feed_handler(std.testing.allocator);
}

test "allocator stress test" {
    try stress_test_allocators(std.testing.allocator);
}

test "rng normal distribution" {
    var rng = RNG.init(123);
    var sum: f64 = 0;
    const N = 10000;
    for (0..N) |_| sum += rng.normal();
    const mean = sum / N;
    try std.testing.expect(@fabs(mean) < 0.1);
}
