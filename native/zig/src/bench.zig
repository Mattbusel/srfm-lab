//! Comprehensive benchmarks for all Zig market data components.
//! Measures throughput, latency, and memory usage.

const std = @import("std");
const math = std.math;
const time_mod = std.time;
const Allocator = std.mem.Allocator;

const decoder_mod  = @import("decoder.zig");
const orderbook    = @import("orderbook.zig");
const stats_mod    = @import("stats.zig");
const writer_mod   = @import("writer.zig");
const alloc_mod    = @import("allocator.zig");
const mm_mod       = @import("market_maker.zig");
const proto_mod    = @import("protocol.zig");
const net_mod      = @import("network.zig");

const Timer = std.time.Timer;

// ============================================================
// Benchmark result
// ============================================================
const BenchResult = struct {
    name:        []const u8,
    n_ops:       u64,
    elapsed_ns:  u64,
    ns_per_op:   f64,
    ops_per_sec: f64,

    pub fn print(self: *const BenchResult) void {
        std.debug.print("  {s:<36}  {d:>10} ops  {d:>8.1} ns/op  {d:>12.0} ops/s\n",
            .{ self.name, self.n_ops, self.ns_per_op, self.ops_per_sec });
    }
};

fn make_result(name: []const u8, n: u64, elapsed_ns: u64) BenchResult {
    const ns_per_op = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(@max(n, 1)));
    return BenchResult{
        .name        = name,
        .n_ops       = n,
        .elapsed_ns  = elapsed_ns,
        .ns_per_op   = ns_per_op,
        .ops_per_sec = if (ns_per_op > 1e-9) 1e9 / ns_per_op else 0,
    };
}

// ============================================================
// Decoder benchmarks
// ============================================================
pub fn bench_decode() !void {
    std.debug.print("\n[Decoder]\n", .{});
    var dec = decoder_mod.Decoder.init();
    const N: u64 = 1_000_000;

    // AddOrder
    {
        const msg = decoder_mod.Decoder.build_add_order(
            1, 'B', 100, "AAPL    ", 1825000, 34_200_000_000_000);
        const t = try Timer.start();
        var i: u64 = 0;
        var count: u64 = 0;
        while (i < N) : (i += 1) {
            if (dec.decode_one(&msg)) |_| count += 1;
        }
        _ = count;
        make_result("decode_add_order", N, t.read()).print();
    }

    // Mixed message types
    {
        const add_msg = decoder_mod.Decoder.build_add_order(
            1, 'S', 200, "MSFT    ", 3750000, 34_200_000_000_000);
        const t = try Timer.start();
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            _ = dec.decode_one(&add_msg);
        }
        make_result("decode_mixed_batch", N, t.read()).print();
    }

    // Decoder stats
    std.debug.print("  Decoder stats: total={d} ok={d} unknown={d}\n",
        .{ dec.stats.total, dec.stats.add_orders, dec.stats.unknown });
}

// ============================================================
// Order book benchmarks
// ============================================================
pub fn bench_orderbook(allocator: Allocator) !void {
    std.debug.print("\n[Order Book]\n", .{});

    // Add-order throughput
    {
        var book = orderbook.OrderBook.init(allocator, "BENCH");
        defer book.deinit();
        const N: u64 = 100_000;
        const t = try Timer.start();
        for (0..N) |i| {
            const order = orderbook.Order{
                .id         = @as(orderbook.OrderId, @intCast(i + 1)),
                .symbol     = [_]u8{'B','E','N','C','H',0}++[_]u8{0}**10,
                .side       = if (i % 2 == 0) .Buy else .Sell,
                .order_type = .Limit,
                .status     = .New,
                .price      = orderbook.from_double(100.0 + @as(f64, @floatFromInt(i % 50)) * 0.01 - 0.25),
                .qty        = 100,
                .filled_qty = 0,
                .timestamp  = @as(i64, @intCast(i)) * 1_000_000,
                .sequence   = 0,
            };
            var fills = try book.add_order(order);
            fills.deinit();
        }
        make_result("add_order_limit", N, t.read()).print();
    }

    // Cancel-order throughput
    {
        var book = orderbook.OrderBook.init(allocator, "BENCH2");
        defer book.deinit();
        const N: u64 = 10_000;
        for (0..N) |i| {
            const order = orderbook.Order{
                .id         = @as(orderbook.OrderId, @intCast(i + 1)),
                .symbol     = [_]u8{'B','E','N','C','H',0}++[_]u8{0}**10,
                .side       = if (i % 2 == 0) .Buy else .Sell,
                .order_type = .Limit,
                .status     = .New,
                .price      = orderbook.from_double(100.0 + @as(f64, @floatFromInt(i % 20)) * 0.01),
                .qty        = 100,
                .filled_qty = 0,
                .timestamp  = 0,
                .sequence   = 0,
            };
            var fills = try book.add_order(order);
            fills.deinit();
        }
        const t = try Timer.start();
        for (0..N) |i| _ = book.cancel_order(@as(orderbook.OrderId, @intCast(i + 1)));
        make_result("cancel_order", N, t.read()).print();
    }

    // Mid-price lookup
    {
        var book = orderbook.OrderBook.init(allocator, "BENCH3");
        defer book.deinit();
        for (0..100) |i| {
            const order = orderbook.Order{
                .id         = @as(orderbook.OrderId, @intCast(i + 1)),
                .symbol     = [_]u8{'B','E','N','C','H',0}++[_]u8{0}**10,
                .side       = if (i % 2 == 0) .Buy else .Sell,
                .order_type = .Limit,
                .status     = .New,
                .price      = orderbook.from_double(100.0 + @as(f64, @floatFromInt(i % 10)) * 0.01),
                .qty        = 100,
                .filled_qty = 0,
                .timestamp  = 0,
                .sequence   = 0,
            };
            var fills = try book.add_order(order);
            fills.deinit();
        }
        const N: u64 = 1_000_000;
        const t = try Timer.start();
        var sum: i64 = 0;
        for (0..N) |_| if (book.mid_price()) |mp| { sum += mp; };
        _ = sum;
        make_result("mid_price_lookup", N, t.read()).print();
    }

    // Market order matching (with active levels)
    {
        const N: u64 = 1_000;
        const t = try Timer.start();
        for (0..N) |trial| {
            var book = orderbook.OrderBook.init(allocator, "BENCH4");
            defer book.deinit();
            _ = trial;
            // Add 10 passive asks
            for (0..10) |j| {
                const order = orderbook.Order{
                    .id         = @as(orderbook.OrderId, @intCast(j + 1)),
                    .symbol     = [_]u8{'B','E','N','C','H',0}++[_]u8{0}**10,
                    .side       = .Sell,
                    .order_type = .Limit,
                    .status     = .New,
                    .price      = orderbook.from_double(100.0 + @as(f64, @floatFromInt(j)) * 0.01),
                    .qty        = 100,
                    .filled_qty = 0,
                    .timestamp  = 0,
                    .sequence   = 0,
                };
                var fills = try book.add_order(order);
                fills.deinit();
            }
            // Market buy
            const mkt = orderbook.Order{
                .id = 9999, .symbol = [_]u8{'B','E','N','C','H',0}++[_]u8{0}**10,
                .side = .Buy, .order_type = .Market, .status = .New,
                .price = 0, .qty = 500, .filled_qty = 0, .timestamp = 0, .sequence = 0,
            };
            var fills = try book.add_order(mkt);
            fills.deinit();
        }
        make_result("market_order_match_10levels", N, t.read()).print();
    }
}

// ============================================================
// Statistics benchmarks
// ============================================================
pub fn bench_stats() void {
    std.debug.print("\n[Statistics]\n", .{});

    {
        var rw = stats_mod.RollingWindow(f64, 100){};
        const N: u64 = 5_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            rw.push(@as(f64, @floatFromInt(i % 1000)) * 0.001);
            _ = rw.mean();
        }
        make_result("rolling_window_push+mean", N, t.read()).print();
    }

    {
        var rw = stats_mod.RollingWindow(f64, 100){};
        const N: u64 = 2_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            rw.push(@as(f64, @floatFromInt(i % 1000)) * 0.01);
            _ = rw.variance();
        }
        make_result("rolling_window_variance", N, t.read()).print();
    }

    {
        var vwap = stats_mod.VWAPCalc{};
        const N: u64 = 2_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1)
            vwap.update(100.0 + @as(f64, @floatFromInt(i % 100)) * 0.01, 100);
        make_result("vwap_update", N, t.read()).print();
    }

    {
        var vc = stats_mod.VolatilityCalc{};
        const N: u64 = 2_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            vc.update(100.0 * (1.0 + @as(f64, @floatFromInt(i % 500)) * 0.0001));
            _ = vc.annualized_vol();
        }
        make_result("volatility_calc", N, t.read()).print();
    }

    {
        var imb = stats_mod.ImbalanceTracker{};
        const N: u64 = 5_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1)
            imb.update(1000 + i % 500, 1000 + (i + 250) % 500);
        make_result("imbalance_tracker", N, t.read()).print();
    }

    {
        var spread = stats_mod.SpreadStats{};
        const N: u64 = 3_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1)
            spread.update(100.0 - @as(f64, @floatFromInt(i % 5)) * 0.01,
                          100.0 + @as(f64, @floatFromInt(i % 5)) * 0.01);
        make_result("spread_stats", N, t.read()).print();
    }
}

// ============================================================
// Writer benchmarks
// ============================================================
pub fn bench_writer(allocator: Allocator) !void {
    std.debug.print("\n[Ring Buffer Writer]\n", .{});

    {
        var w = try writer_mod.InMemWriter.init(allocator, 1 << 20);
        defer w.deinit();
        const N: u64 = 1_000_000;
        const t = try Timer.start();
        for (0..N) |i| {
            var tick: writer_mod.SharedTick = std.mem.zeroes(writer_mod.SharedTick);
            tick.timestamp = @as(i64, @intCast(i)) * 1_000_000;
            tick.price     = orderbook.from_double(100.0 + @as(f64, @floatFromInt(i % 100)) * 0.01);
            tick.qty       = 100;
            tick.side      = @as(u8, @intCast(i % 2));
            w.write_tick(tick);
        }
        make_result("write_tick_inmem", N, t.read()).print();
        std.debug.print("  Ticks written (seq): {d}\n", .{w.write_seq});
    }

    // Write with overflow (ring buffer wraps)
    {
        const SMALL_CAP: usize = 256;
        var w = try writer_mod.InMemWriter.init(allocator, SMALL_CAP);
        defer w.deinit();
        const N: u64 = 100_000;
        const t = try Timer.start();
        for (0..N) |i| {
            var tick: writer_mod.SharedTick = std.mem.zeroes(writer_mod.SharedTick);
            tick.price = @as(i64, @intCast(i));
            w.write_tick(tick);
        }
        make_result("write_tick_overflow", N, t.read()).print();
    }
}

// ============================================================
// Allocator benchmarks
// ============================================================
pub fn bench_pool_alloc() !void {
    std.debug.print("\n[Allocators]\n", .{});

    {
        var pool = alloc_mod.PoolAllocator(orderbook.Order, 4096).init();
        const N: u64 = 1_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            const p = try pool.alloc();
            p.id = i + 1;
            pool.free(p);
        }
        make_result("pool_alloc_free_Order", N, t.read()).print();
    }

    {
        var pool = alloc_mod.PoolAllocator(writer_mod.SharedTick, 8192).init();
        const N: u64 = 2_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            const p = try pool.alloc();
            _ = p;
            pool.free(p);
        }
        make_result("pool_alloc_free_SharedTick", N, t.read()).print();
    }
}

// ============================================================
// Market Maker benchmarks
// ============================================================
pub fn bench_market_maker() void {
    std.debug.print("\n[Market Maker]\n", .{});

    var mm = mm_mod.MarketMaker.init(mm_mod.MMParams{});

    {
        const N: u64 = 2_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            const mid = 100.0 + @as(f64, @floatFromInt(i % 1000)) * 0.001;
            _ = mm.on_book_update(mid, 0.10, 1000, 1000, 0.20);
        }
        make_result("mm_on_book_update", N, t.read()).print();
    }

    {
        const N: u64 = 5_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1)
            mm.on_fill(if (i % 2 == 0) .Buy else .Sell, 100, 100.0);
        make_result("mm_on_fill", N, t.read()).print();
    }

    {
        const N: u64 = 5_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1)
            mm.mark_to_market(100.0 + @as(f64, @floatFromInt(i % 100)) * 0.001);
        make_result("mm_mark_to_market", N, t.read()).print();
    }

    {
        const N: u64 = 10_000_000;
        const t = Timer.start() catch return;
        var i: u64 = 0;
        while (i < N) : (i += 1)
            _ = mm.pnl.net_pnl();
        make_result("pnl_net_pnl_call", N, t.read()).print();
    }
}

// ============================================================
// Protocol benchmarks
// ============================================================
pub fn bench_protocol(allocator: Allocator) !void {
    std.debug.print("\n[Protocol Codecs]\n", .{});

    // SBE decode
    {
        var buf: [64]u8 = std.mem.zeroes([64]u8);
        buf[0] = 0;
        const mantissa: i64 = 1825000;
        std.mem.writeInt(i64, buf[2..10][0..8], mantissa, .little);
        buf[10] = @bitCast(@as(i8, -4));
        const qty: u64 = 100;
        std.mem.writeInt(u64, buf[18..26][0..8], qty, .little);

        const N: u64 = 5_000_000;
        const t = try Timer.start();
        var count: u64 = 0;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            var dec = proto_mod.SbeDecoder.init(buf[0..34]);
            if (dec.decode_md_entry()) |_| count += 1;
        }
        _ = count;
        make_result("sbe_decode_md_entry", N, t.read()).print();
    }

    // SBE header encode/decode
    {
        var buf: [8]u8 = undefined;
        const N: u64 = 10_000_000;
        const t = try Timer.start();
        var i: u64 = 0;
        var sum: u64 = 0;
        while (i < N) : (i += 1) {
            var enc = proto_mod.SbeEncoder.init(&buf);
            const hdr = proto_mod.SbeHeader{ .block_length = 64, .template_id = @as(u16, @intCast(i % 100)),
                                              .schema_id = 1, .version = 1 };
            _ = enc.write_header(hdr);
            var dec = proto_mod.SbeDecoder.init(enc.written());
            if (dec.decode_header()) |h| sum += h.template_id;
        }
        _ = sum;
        make_result("sbe_header_roundtrip", N, t.read()).print();
    }

    // FIX parse
    {
        const fix_msg = "8=FIX.4.4|35=D|49=S|56=T|34=1|55=AAPL|54=1|44=182.50|38=100|40=2|";
        const N: u64 = 100_000;
        const t = try Timer.start();
        var count: u64 = 0;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            var dec = proto_mod.FixDecoder.init(allocator, fix_msg);
            defer dec.deinit();
            count += dec.parse() catch 0;
        }
        _ = count;
        make_result("fix_parse_new_order", N, t.read()).print();
    }

    // ITCH normalize
    {
        const norm = proto_mod.Normalizer.init(.ITCH50, 1);
        const msg = decoder_mod.Message{
            .add_order = decoder_mod.AddOrder{
                .stock_locate = 1, .tracking_number = 0,
                .timestamp = 34_200_000_000_000,
                .order_reference = 42, .side = 'B', .shares = 100,
                .stock = [_]u8{'A','A','P','L',' ',' ',' ',' '},
                .price = 1825000,
            },
        };
        const N: u64 = 10_000_000;
        const t = try Timer.start();
        var count: u64 = 0;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            const tick = norm.normalize_itch(msg, 34_200_000_000_000);
            if (tick.price > 0) count += 1;
        }
        _ = count;
        make_result("itch_normalize", N, t.read()).print();
    }
}

// ============================================================
// Network benchmarks
// ============================================================
pub fn bench_network(allocator: Allocator) !void {
    std.debug.print("\n[Network Layer]\n", .{});

    {
        var pool = net_mod.PacketPool(256).init();
        const N: u64 = 5_000_000;
        const t = try Timer.start();
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            if (pool.alloc()) |p| { p.len = 64; pool.free(p); }
        }
        make_result("packet_pool_alloc_free", N, t.read()).print();
    }

    {
        var rl = net_mod.RateLimiter.init(100_000.0, 1000.0);
        const N: u64 = 5_000_000;
        const t = try Timer.start();
        var count: u64 = 0;
        var i: u64 = 0;
        while (i < N) : (i += 1) {
            const now: i64 = @intCast(i * 10_000);
            if (rl.try_consume(now, 1.0)) count += 1;
        }
        _ = count;
        make_result("rate_limiter_try_consume", N, t.read()).print();
    }

    {
        var tracker = net_mod.SequenceTracker.init(allocator, 1);
        defer tracker.deinit();
        const N: u64 = 1_000_000;
        const t = try Timer.start();
        var i: u64 = 0;
        while (i < N) : (i += 1)
            _ = tracker.process(@as(u32, @intCast(i + 1))) catch {};
        make_result("seq_tracker_in_order", N, t.read()).print();
    }

    {
        var recv = try net_mod.UdpReceiver.init(allocator, .{});
        defer recv.deinit();
        try recv.open();
        const N: u64 = 1_000_000;
        const t = try Timer.start();
        for (0..N) |i| {
            var pkt: [128]u8 = undefined;
            const seq = @as(u32, @intCast(i + 1));
            pkt[0] = @truncate(seq >> 24);
            pkt[1] = @truncate(seq >> 16);
            pkt[2] = @truncate(seq >>  8);
            pkt[3] = @truncate(seq);
            @memset(pkt[4..], 0xAB);
            try recv.recv_simulated(&pkt, @as(i64, @intCast(i)) * 1_000_000);
        }
        make_result("udp_recv_simulated", N, t.read()).print();
        std.debug.print("  Packets ok: {d}  dropped: {d}  gaps: {d}\n",
            .{ recv.stats.packets_received, recv.stats.packets_dropped, recv.stats.gaps_detected });
    }
}

// ============================================================
// Full pipeline benchmark
// ============================================================
pub fn bench_full_pipeline(allocator: Allocator) !void {
    std.debug.print("\n[Full Pipeline: Decoder→Book→Stats→Writer→MM]\n", .{});

    var dec    = decoder_mod.Decoder.init();
    var book   = orderbook.OrderBook.init(allocator, "PIPE");
    defer book.deinit();
    var vc     = stats_mod.VolatilityCalc{};
    var vwap   = stats_mod.VWAPCalc{};
    var mm     = mm_mod.MarketMaker.init(mm_mod.MMParams{});
    var writer = try writer_mod.InMemWriter.init(allocator, 1 << 18);
    defer writer.deinit();

    const N: u64 = 200_000;
    const t = try Timer.start();
    var fills_total: u64 = 0;

    for (0..N) |i| {
        const is_buy = i % 2 == 0;
        const px_d   = 100.0 + @as(f64, @floatFromInt(i % 200)) * 0.01 - 1.0;
        const px_itch = @as(u32, @intFromFloat(@max(px_d * 10000.0, 0.0)));
        const msg = decoder_mod.Decoder.build_add_order(
            @as(u64, @intCast(i + 1)), if (is_buy) 'B' else 'S',
            100, "PIPE    ", px_itch, @as(u64, @intCast(i)) * 1_000_000);

        const decoded = dec.decode_one(&msg) orelse continue;
        if (decoded != .add_order) continue;
        const ao = decoded.add_order;

        const order = orderbook.Order{
            .id         = ao.order_reference,
            .symbol     = [_]u8{'P','I','P','E',0}++[_]u8{0}**11,
            .side       = if (ao.side == 'B') .Buy else .Sell,
            .order_type = .Limit,
            .status     = .New,
            .price      = orderbook.from_double(@as(f64, @floatFromInt(ao.price)) / 10000.0),
            .qty        = ao.shares,
            .filled_qty = 0,
            .timestamp  = @as(i64, @intCast(i)) * 1_000_000,
            .sequence   = 0,
        };
        var book_fills = try book.add_order(order);
        defer book_fills.deinit();

        for (book_fills.items) |fill| {
            const fp = orderbook.to_double(fill.price);
            vwap.update(fp, fill.qty);
            mm.on_fill(fill.side, fill.qty, fp);
            fills_total += 1;
            var tick: writer_mod.SharedTick = std.mem.zeroes(writer_mod.SharedTick);
            tick.price = fill.price;
            tick.qty   = fill.qty;
            writer.write_tick(tick);
        }

        if (book.mid_price()) |mp| {
            const mid_d = orderbook.to_double(mp);
            vc.update(mid_d);
            const vol = vc.annualized_vol();
            _ = mm.on_book_update(mid_d, 0.10, book.bid_total_qty(), book.ask_total_qty(), vol);
            mm.mark_to_market(mid_d);
        }
    }
    make_result("full_pipeline_msg", N, t.read()).print();
    std.debug.print("  Fills: {d}  Ticks: {d}  VWAP: {d:.4}  MM fills: {d}\n",
        .{ fills_total, writer.write_seq, vwap.session_vwap(), mm.fill_count });
}

// ============================================================
// Memory footprint
// ============================================================
pub fn report_memory_footprint() void {
    std.debug.print("\n[Struct Sizes (bytes)]\n", .{});
    const entries = .{
        .{ "Order",            @sizeOf(orderbook.Order) },
        .{ "PriceLevel",       @sizeOf(orderbook.PriceLevel) },
        .{ "SharedTick",       @sizeOf(writer_mod.SharedTick) },
        .{ "NormalizedTick",   @sizeOf(proto_mod.NormalizedTick) },
        .{ "PacketBuffer",     @sizeOf(net_mod.PacketBuffer) },
        .{ "MMParams",         @sizeOf(mm_mod.MMParams) },
        .{ "PnLState",         @sizeOf(mm_mod.PnLState) },
        .{ "MarketMaker",      @sizeOf(mm_mod.MarketMaker) },
        .{ "VWAPCalc",         @sizeOf(stats_mod.VWAPCalc) },
        .{ "VolatilityCalc",   @sizeOf(stats_mod.VolatilityCalc) },
        .{ "SbeHeader",        @sizeOf(proto_mod.SbeHeader) },
    };
    inline for (entries) |e| {
        std.debug.print("  {s:<24}  {d:>6}\n", .{ e[0], e[1] });
    }
}

// ============================================================
// Main
// ============================================================
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("╔══════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║         Zig Market Data System — Benchmarks              ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════════════════╝\n", .{});

    try bench_decode();
    try bench_orderbook(allocator);
    bench_stats();
    try bench_writer(allocator);
    try bench_pool_alloc();
    bench_market_maker();
    try bench_protocol(allocator);
    try bench_network(allocator);
    try bench_full_pipeline(allocator);
    report_memory_footprint();

    std.debug.print("\nAll benchmarks complete.\n", .{});
}
