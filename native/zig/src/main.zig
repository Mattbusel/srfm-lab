//! Main entry point: UDP multicast market data listener + worker thread pool.
//! Ties together decoder, feed handler, order book, stats, and writer.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Thread = std.Thread;
const Atomic = std.atomic.Value;

const decoder_mod = @import("decoder.zig");
const orderbook_mod = @import("orderbook.zig");
const feed_mod = @import("feed.zig");
const stats_mod = @import("stats.zig");
const writer_mod = @import("writer.zig");
const alloc_mod = @import("allocator.zig");

// ============================================================
// Configuration
// ============================================================
pub const Config = struct {
    // Network
    multicast_group: []const u8 = "239.0.0.1",
    multicast_port:  u16        = 20001,
    interface:       []const u8 = "0.0.0.0",
    recv_buf_size:   usize      = 262144,  // 256KB socket recv buffer

    // Processing
    num_workers:     usize  = 4,
    queue_depth:     usize  = 65536,
    batch_size:      usize  = 256,

    // Output
    output_path:     []const u8 = "/tmp/market_data.ring",
    ring_capacity:   usize      = 1 << 20, // 1M ticks

    // Symbols to track (empty = all)
    symbols: []const []const u8 = &.{},

    // Stats interval
    stats_interval_ms: u64 = 1000,
};

// ============================================================
// Shared packet queue between network thread and worker pool
// ============================================================
pub const RawPacket = struct {
    data:       [65536]u8 = undefined,
    len:        usize     = 0,
    source_id:  u16       = 0,
    seq_num:    u64       = 0,
    recv_ts:    i64       = 0,
};

// ============================================================
// Worker State
// ============================================================
pub const WorkerState = struct {
    id:         usize,
    dec:        decoder_mod.Decoder = .{},
    running:    *Atomic(bool),
    // Per-worker stats
    msgs_processed: u64 = 0,
    trades_seen:    u64 = 0,
    errors:         u64 = 0,
};

// ============================================================
// Global Engine State
// ============================================================
pub const Engine = struct {
    config:     Config,
    allocator:  Allocator,

    // Order books per symbol
    books:      std.StringHashMap(*orderbook_mod.OrderBook),

    // Stats per symbol
    stats_map:  std.StringHashMap(*stats_mod.MarketStats),

    // Output writer
    writer:     ?writer_mod.InMemWriter,

    // Feed handler
    feed:       feed_mod.FeedHandler,

    // Runtime state
    running:    Atomic(bool),
    start_time: i64,

    // Global stats
    total_messages:   Atomic(u64),
    total_trades:     Atomic(u64),
    total_orders:     Atomic(u64),
    total_cancels:    Atomic(u64),

    const Self = @This();

    pub fn init(allocator: Allocator, config: Config) !Self {
        var feed = feed_mod.FeedHandler.init(allocator, .{});
        errdefer feed.deinit();

        var books = std.StringHashMap(*orderbook_mod.OrderBook).init(allocator);
        errdefer books.deinit();

        var stats_map = std.StringHashMap(*stats_mod.MarketStats).init(allocator);
        errdefer stats_map.deinit();

        var writer: ?writer_mod.InMemWriter = null;
        writer = try writer_mod.InMemWriter.init(allocator, config.ring_capacity);

        return Self{
            .config        = config,
            .allocator     = allocator,
            .books         = books,
            .stats_map     = stats_map,
            .writer        = writer,
            .feed          = feed,
            .running       = Atomic(bool).init(false),
            .start_time    = std.time.nanoTimestamp(),
            .total_messages = Atomic(u64).init(0),
            .total_trades  = Atomic(u64).init(0),
            .total_orders  = Atomic(u64).init(0),
            .total_cancels = Atomic(u64).init(0),
        };
    }

    pub fn deinit(self: *Self) void {
        var book_it = self.books.iterator();
        while (book_it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.books.deinit();

        var stats_it = self.stats_map.iterator();
        while (stats_it.next()) |entry| {
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.stats_map.deinit();

        if (self.writer) |*w| w.deinit();
        self.feed.deinit();
    }

    pub fn get_or_create_book(self: *Self, symbol: []const u8) !*orderbook_mod.OrderBook {
        if (self.books.get(symbol)) |b| return b;
        const book = try self.allocator.create(orderbook_mod.OrderBook);
        book.* = orderbook_mod.OrderBook.init(self.allocator, symbol);
        try self.books.put(symbol, book);
        return book;
    }

    pub fn get_or_create_stats(self: *Self, symbol: []const u8) !*stats_mod.MarketStats {
        if (self.stats_map.get(symbol)) |s| return s;
        const s = try self.allocator.create(stats_mod.MarketStats);
        s.* = stats_mod.MarketStats.init(symbol, 0.01);
        try self.stats_map.put(symbol, s);
        return s;
    }

    // Process a decoded ITCH message
    pub fn process_message(self: *Self, msg: decoder_mod.Message, recv_ts: i64) !void {
        _ = self.total_messages.fetchAdd(1, .monotonic);
        switch (msg) {
            .AddOrder, .AddOrderMPID => |ao| {
                _ = self.total_orders.fetchAdd(1, .monotonic);
                const sym = std.mem.trimRight(u8, &ao.stock, " ");
                const book = try self.get_or_create_book(sym);
                const order = orderbook_mod.Order{
                    .id         = ao.order_ref,
                    .symbol     = blk: {
                        var s: [16]u8 = [_]u8{0}**16;
                        const l = @min(sym.len, 15);
                        @memcpy(s[0..l], sym[0..l]);
                        break :blk s;
                    },
                    .side       = if (ao.side == .Buy) .Buy else .Sell,
                    .order_type = .Limit,
                    .status     = .New,
                    .price      = orderbook_mod.from_double(ao.price_f64()),
                    .qty        = ao.shares,
                    .filled_qty = 0,
                    .timestamp  = recv_ts,
                    .sequence   = 0,
                };
                var trades = try book.add_order(order);
                defer trades.deinit();

                // Write resulting trades
                for (trades.items) |t| {
                    _ = self.total_trades.fetchAdd(1, .monotonic);
                    if (self.writer) |*w| {
                        var tick: writer_mod.SharedTick = std.mem.zeroes(writer_mod.SharedTick);
                        tick.timestamp  = recv_ts;
                        @memcpy(&tick.symbol, &t.symbol);
                        tick.price      = t.price;
                        tick.qty        = t.qty;
                        tick.side       = @intFromEnum(t.aggressor_side);
                        tick.tick_type  = 0;
                        tick.trade_id   = t.trade_id;
                        w.write_tick(tick);
                    }
                    // Update stats
                    const st = try self.get_or_create_stats(sym);
                    st.on_trade(orderbook_mod.to_double(t.price), t.qty,
                                @intFromEnum(t.aggressor_side), recv_ts);
                }
            },
            .OrderCancel => |oc| {
                _ = self.total_cancels.fetchAdd(1, .monotonic);
                // Find which book has this order
                var it = self.books.iterator();
                while (it.next()) |entry| {
                    if (entry.value_ptr.*.cancel_order(oc.order_ref)) break;
                }
            },
            .OrderDelete => |od| {
                _ = self.total_cancels.fetchAdd(1, .monotonic);
                var it = self.books.iterator();
                while (it.next()) |entry| {
                    if (entry.value_ptr.*.cancel_order(od.order_ref)) break;
                }
            },
            .Trade => |t| {
                _ = self.total_trades.fetchAdd(1, .monotonic);
                const sym = std.mem.trimRight(u8, &t.stock, " ");
                if (self.writer) |*w| {
                    var tick: writer_mod.SharedTick = std.mem.zeroes(writer_mod.SharedTick);
                    tick.timestamp = recv_ts;
                    const l = @min(sym.len, 15);
                    @memcpy(tick.symbol[0..l], sym[0..l]);
                    tick.price     = @as(i64, @intCast(t.price));
                    tick.qty       = t.shares;
                    tick.side      = if (t.side == .Buy) 0 else 1;
                    tick.tick_type = 0;
                    tick.trade_id  = t.match_number;
                    w.write_tick(tick);
                }
                const st = try self.get_or_create_stats(sym);
                st.on_trade(t.price_f64(), t.shares,
                            if (t.side == .Buy) 0 else 1, recv_ts);
            },
            else => {},
        }
    }

    pub fn print_stats(self: *const Self) void {
        const elapsed_s = @as(f64, @floatFromInt(std.time.nanoTimestamp() - self.start_time)) / 1e9;
        std.debug.print(
            "\n=== Engine Stats (t={d:.1}s) ===\n" ++
            "  Messages: {d}\n" ++
            "  Orders:   {d}\n" ++
            "  Trades:   {d}\n" ++
            "  Cancels:  {d}\n" ++
            "  Books:    {d}\n",
            .{
                elapsed_s,
                self.total_messages.load(.monotonic),
                self.total_orders.load(.monotonic),
                self.total_trades.load(.monotonic),
                self.total_cancels.load(.monotonic),
                self.books.count(),
            });

        var it = self.stats_map.iterator();
        while (it.next()) |entry| {
            const sym = entry.key_ptr.*;
            const s = entry.value_ptr.*.summary();
            std.debug.print(
                "  [{s}] price={d:.2} vwap={d:.2} spread={d:.4} vol={d:.2}% tps={d:.1}\n",
                .{
                    sym,
                    s.last_price,
                    s.session_vwap,
                    s.mean_spread,
                    s.annualized_vol * 100.0,
                    s.trades_per_sec,
                });
        }
    }
};

// ============================================================
// Worker thread: decodes packets from queue
// ============================================================
pub const WorkerContext = struct {
    engine:     *Engine,
    worker_id:  usize,
    running:    *Atomic(bool),
};

fn worker_thread(ctx: WorkerContext) void {
    var dec = decoder_mod.Decoder{};
    var local_buf: [65536]u8 = undefined;
    var msgs_processed: u64 = 0;

    while (ctx.running.load(.acquire)) {
        // In a real system: pop from lock-free queue
        // Here: just yield since we have no actual network
        Thread.yield() catch {};
        _ = &dec;
        _ = &local_buf;
        _ = &msgs_processed;
    }
}

// ============================================================
// Simulation: generate synthetic ITCH feed for testing
// ============================================================
pub fn simulate_feed(engine: *Engine, num_messages: usize) !void {
    var dec = decoder_mod.Decoder{};
    var rng = std.rand.DefaultPrng.init(12345);
    const rand = rng.random();

    const symbols = [_][]const u8{ "AAPL    ", "TSLA    ", "NVDA    ", "MSFT    ", "AMZN    " };
    var order_id: u64 = 1;
    const ts_base: i64 = 34200_000_000_000; // 9:30 AM in ns since midnight

    var i: usize = 0;
    while (i < num_messages) : (i += 1) {
        const sym_idx = rand.uintLessThan(usize, symbols.len);
        const sym = symbols[sym_idx];
        const side: decoder_mod.Side = if (rand.boolean()) .Buy else .Sell;
        const price = 10000 + rand.uintLessThan(u32, 50000); // $1-6 range (10000-60000 / 10000)
        const qty   = 100 + rand.uintLessThan(u32, 900);
        const ts    = ts_base + @as(i64, @intCast(i)) * 1_000_000;

        const msg_buf = decoder_mod.Decoder.build_add_order(
            @intCast(sym_idx + 1), order_id, side, qty, sym, price, @intCast(ts));
        order_id += 1;

        const result = dec.decode_one(&msg_buf) catch continue;
        try engine.process_message(result.msg, std.time.nanoTimestamp());

        // Occasionally generate cancels
        if (i > 100 and rand.uintLessThan(usize, 10) == 0) {
            // In a real decoder, we'd build an OrderDelete message
            // Here just cancel a recent order
            var it = engine.books.iterator();
            if (it.next()) |entry| {
                _ = entry.value_ptr.*.cancel_order(order_id - rand.uintLessThan(u64, 50) - 1);
            }
        }
    }
}

// ============================================================
// Main
// ============================================================
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = Config{
        .num_workers   = 4,
        .queue_depth   = 65536,
        .stats_interval_ms = 1000,
    };

    std.debug.print("HFT Market Data Processor\n", .{});
    std.debug.print("=========================\n", .{});
    std.debug.print("Workers: {d}\n", .{config.num_workers});
    std.debug.print("Ring capacity: {d} ticks\n\n", .{config.ring_capacity});

    var engine = try Engine.init(allocator, config);
    defer engine.deinit();

    // Run simulation
    std.debug.print("Running simulation with 100,000 messages...\n", .{});
    const t0 = std.time.nanoTimestamp();
    try simulate_feed(&engine, 100_000);
    const t1 = std.time.nanoTimestamp();

    const elapsed_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;
    const msgs = engine.total_messages.load(.monotonic);
    const throughput = @as(f64, @floatFromInt(msgs)) / (elapsed_ms / 1000.0);

    std.debug.print("Processed {d} messages in {d:.2}ms ({d:.0} msg/s)\n",
        .{msgs, elapsed_ms, throughput});

    engine.print_stats();

    if (engine.writer) |w| {
        std.debug.print("\nTicks written to ring buffer: {d}\n", .{w.write_seq});
    }
}

// ============================================================
// Tests
// ============================================================
test "engine initialization" {
    const alloc = std.testing.allocator;
    const config = Config{
        .ring_capacity = 1024,
    };
    var engine = try Engine.init(alloc, config);
    defer engine.deinit();

    try std.testing.expectEqual(@as(u64, 0), engine.total_messages.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 0), engine.books.count());
}

test "engine simulate feed" {
    const alloc = std.testing.allocator;
    const config = Config{ .ring_capacity = 1024 };
    var engine = try Engine.init(alloc, config);
    defer engine.deinit();

    try simulate_feed(&engine, 1000);

    try std.testing.expect(engine.total_messages.load(.monotonic) > 0);
    try std.testing.expect(engine.books.count() > 0);
}

test "full pipeline" {
    const alloc = std.testing.allocator;
    const config = Config{ .ring_capacity = 4096 };
    var engine = try Engine.init(alloc, config);
    defer engine.deinit();

    try simulate_feed(&engine, 5000);

    const msgs = engine.total_messages.load(.monotonic);
    try std.testing.expect(msgs > 0);

    // Check at least one book has trades
    var any_trade = false;
    var it = engine.books.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.*.trades.items.len > 0) {
            any_trade = true;
            break;
        }
    }
    try std.testing.expect(any_trade);

    // Writer should have ticks
    if (engine.writer) |w| {
        try std.testing.expect(w.write_seq > 0);
    }
}
