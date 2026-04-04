//! Additional market data protocol support: SBE (Simple Binary Encoding),
//! FIX-over-UDP, and a generic normalized tick format.
//! Provides encoding/decoding for multiple exchange protocols.

const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const orderbook = @import("orderbook.zig");
const decoder   = @import("decoder.zig");

pub const Price    = orderbook.Price;
pub const Quantity = orderbook.Quantity;
pub const OrderId  = orderbook.OrderId;
pub const Side     = orderbook.Side;

// ============================================================
// Normalized tick event (exchange-agnostic)
// ============================================================
pub const TickType = enum(u8) {
    Trade     = 0,
    BidUpdate = 1,
    AskUpdate = 2,
    OrderNew  = 3,
    OrderCancel = 4,
    OrderExec = 5,
    Heartbeat = 0xFF,
};

pub const NormalizedTick = extern struct {
    timestamp_ns: i64,
    symbol:       [16]u8,
    price:        i64,
    qty:          u64,
    bid_price:    i64,
    ask_price:    i64,
    bid_qty:      u64,
    ask_qty:      u64,
    order_id:     u64,
    tick_type:    TickType,
    side:         u8,  // 0=Buy, 1=Sell, 2=Unknown
    flags:        u8,
    _pad:         [5]u8 = [_]u8{0} ** 5,
    seq_num:      u32,
    venue_id:     u16,
    _pad2:        [2]u8 = [_]u8{0} ** 2,

    comptime {
        std.debug.assert(@sizeOf(NormalizedTick) == 96);
    }
};

// ============================================================
// SBE (Simple Binary Encoding) — minimal implementation
// Used by CME, CBOE, and other exchanges
// ============================================================
pub const SbeHeader = packed struct {
    block_length:  u16,
    template_id:   u16,
    schema_id:     u16,
    version:       u16,
};

pub const SbeMessageType = enum(u16) {
    NewOrderSingle      = 1,
    CancelOrder         = 2,
    OrderCancelReplace  = 3,
    ExecutionReport     = 8,
    MarketDataSnapshot  = 55,
    MarketDataIncrement = 56,
    Heartbeat           = 0,
};

// SBE Market Data Entry (price level)
pub const SbeMdEntry = packed struct {
    entry_type:  u8,   // 0=Bid, 1=Ask, 2=Trade
    _pad:        u8,
    price_mantissa: i64,
    price_exponent: i8,
    _pad2:       [7]u8,
    qty:         u64,
    level:       u8,
    action:      u8,   // 0=New, 1=Change, 2=Delete
    _pad3:       [6]u8,
};

pub const SbeDecoder = struct {
    buf:    []const u8,
    pos:    usize,

    const Self = @This();

    pub fn init(buf: []const u8) Self {
        return .{ .buf = buf, .pos = 0 };
    }

    pub fn read_u8(self: *Self) ?u8 {
        if (self.pos >= self.buf.len) return null;
        const v = self.buf[self.pos];
        self.pos += 1;
        return v;
    }

    pub fn read_u16_le(self: *Self) ?u16 {
        if (self.pos + 2 > self.buf.len) return null;
        const v = mem.readInt(u16, self.buf[self.pos..self.pos+2][0..2], .little);
        self.pos += 2;
        return v;
    }

    pub fn read_u32_le(self: *Self) ?u32 {
        if (self.pos + 4 > self.buf.len) return null;
        const v = mem.readInt(u32, self.buf[self.pos..self.pos+4][0..4], .little);
        self.pos += 4;
        return v;
    }

    pub fn read_i64_le(self: *Self) ?i64 {
        if (self.pos + 8 > self.buf.len) return null;
        const v = mem.readInt(i64, self.buf[self.pos..self.pos+8][0..8], .little);
        self.pos += 8;
        return v;
    }

    pub fn read_u64_le(self: *Self) ?u64 {
        if (self.pos + 8 > self.buf.len) return null;
        const v = mem.readInt(u64, self.buf[self.pos..self.pos+8][0..8], .little);
        self.pos += 8;
        return v;
    }

    pub fn skip(self: *Self, n: usize) void {
        self.pos = @min(self.pos + n, self.buf.len);
    }

    pub fn remaining(self: *const Self) usize {
        return if (self.pos < self.buf.len) self.buf.len - self.pos else 0;
    }

    pub fn decode_header(self: *Self) ?SbeHeader {
        const bl = self.read_u16_le() orelse return null;
        const ti = self.read_u16_le() orelse return null;
        const si = self.read_u16_le() orelse return null;
        const vr = self.read_u16_le() orelse return null;
        return SbeHeader{ .block_length = bl, .template_id = ti, .schema_id = si, .version = vr };
    }

    // Decode a market data snapshot entry
    pub fn decode_md_entry(self: *Self) ?NormalizedTick {
        if (self.remaining() < 32) return null;

        const entry_type = self.read_u8() orelse return null;
        self.skip(1); // pad
        const price_mantissa = self.read_i64_le() orelse return null;
        const price_exp_raw  = self.read_u8() orelse return null;
        const price_exp      = @as(i8, @bitCast(price_exp_raw));
        self.skip(7);
        const qty = self.read_u64_le() orelse return null;
        const level = self.read_u8() orelse return null;
        _ = level;
        const action = self.read_u8() orelse return null;
        self.skip(6);

        const price_d = @as(f64, @floatFromInt(price_mantissa)) *
                        math.pow(f64, 10.0, @as(f64, @floatFromInt(price_exp)));

        var tick = std.mem.zeroes(NormalizedTick);
        tick.price     = orderbook.from_double(price_d);
        tick.qty       = qty;
        tick.tick_type = switch (entry_type) {
            0 => TickType.BidUpdate,
            1 => TickType.AskUpdate,
            2 => TickType.Trade,
            else => TickType.Trade,
        };
        tick.side   = if (entry_type == 0) 0 else if (entry_type == 1) 1 else 2;
        tick.flags  = action;
        return tick;
    }
};

pub const SbeEncoder = struct {
    buf: []u8,
    pos: usize,

    const Self = @This();

    pub fn init(buf: []u8) Self { return .{ .buf = buf, .pos = 0 }; }

    pub fn write_u8(self: *Self, v: u8) bool {
        if (self.pos >= self.buf.len) return false;
        self.buf[self.pos] = v;
        self.pos += 1;
        return true;
    }

    pub fn write_u16_le(self: *Self, v: u16) bool {
        if (self.pos + 2 > self.buf.len) return false;
        mem.writeInt(u16, self.buf[self.pos..self.pos+2][0..2], v, .little);
        self.pos += 2;
        return true;
    }

    pub fn write_i64_le(self: *Self, v: i64) bool {
        if (self.pos + 8 > self.buf.len) return false;
        mem.writeInt(i64, self.buf[self.pos..self.pos+8][0..8], v, .little);
        self.pos += 8;
        return true;
    }

    pub fn write_u64_le(self: *Self, v: u64) bool {
        if (self.pos + 8 > self.buf.len) return false;
        mem.writeInt(u64, self.buf[self.pos..self.pos+8][0..8], v, .little);
        self.pos += 8;
        return true;
    }

    pub fn write_header(self: *Self, hdr: SbeHeader) bool {
        return self.write_u16_le(hdr.block_length) and
               self.write_u16_le(hdr.template_id) and
               self.write_u16_le(hdr.schema_id) and
               self.write_u16_le(hdr.version);
    }

    pub fn written(self: *const Self) []const u8 {
        return self.buf[0..self.pos];
    }
};

// ============================================================
// FIX-over-UDP simplified decoder
// FIX 4.4 fields: tag=value|tag=value|...
// ============================================================
pub const FixTag = enum(u32) {
    BeginString    = 8,
    BodyLength     = 9,
    MsgType        = 35,
    SenderCompID   = 49,
    TargetCompID   = 56,
    MsgSeqNum      = 34,
    SendingTime    = 52,
    Symbol         = 55,
    Side           = 54,
    OrderQty       = 38,
    Price          = 44,
    OrdType        = 40,
    TimeInForce    = 59,
    ExecType       = 150,
    OrderID        = 37,
    ClOrdID        = 11,
    LastPx         = 31,
    LastQty        = 32,
    CheckSum       = 10,
    _,
};

pub const FixField = struct {
    tag:   u32,
    value: []const u8,
};

pub const FixDecoder = struct {
    buf: []const u8,
    pos: usize,
    fields: ArrayList(FixField),

    const Self = @This();
    const DELIM = '|'; // use | as SOH substitute

    pub fn init(allocator: Allocator, buf: []const u8) Self {
        return .{
            .buf    = buf,
            .pos    = 0,
            .fields = ArrayList(FixField).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.fields.deinit();
    }

    pub fn parse(self: *Self) !usize {
        self.fields.clearRetainingCapacity();
        self.pos = 0;

        while (self.pos < self.buf.len) {
            // Read tag
            const tag_start = self.pos;
            while (self.pos < self.buf.len and self.buf[self.pos] != '=') : (self.pos += 1) {}
            if (self.pos >= self.buf.len) break;
            const tag_str = self.buf[tag_start..self.pos];
            self.pos += 1; // skip '='

            // Read value
            const val_start = self.pos;
            while (self.pos < self.buf.len and self.buf[self.pos] != DELIM) : (self.pos += 1) {}
            const val_str = self.buf[val_start..self.pos];
            if (self.pos < self.buf.len) self.pos += 1; // skip delim

            const tag = std.fmt.parseInt(u32, tag_str, 10) catch continue;
            try self.fields.append(.{ .tag = tag, .value = val_str });
        }
        return self.fields.items.len;
    }

    pub fn get_field(self: *const Self, tag: u32) ?[]const u8 {
        for (self.fields.items) |f| {
            if (f.tag == tag) return f.value;
        }
        return null;
    }

    pub fn to_normalized(self: *const Self) ?NormalizedTick {
        var tick = std.mem.zeroes(NormalizedTick);

        const msg_type = self.get_field(@intFromEnum(FixTag.MsgType)) orelse return null;

        if (mem.eql(u8, msg_type, "D")) {
            // New Order Single
            const sym  = self.get_field(@intFromEnum(FixTag.Symbol)) orelse return null;
            const side = self.get_field(@intFromEnum(FixTag.Side));
            const px   = self.get_field(@intFromEnum(FixTag.Price));
            const qty  = self.get_field(@intFromEnum(FixTag.OrderQty));

            const sym_len = @min(sym.len, 15);
            @memcpy(tick.symbol[0..sym_len], sym[0..sym_len]);
            tick.tick_type = TickType.OrderNew;

            if (side) |s| tick.side = if (mem.eql(u8, s, "1")) 0 else 1;
            if (px) |p| tick.price = orderbook.from_double(
                std.fmt.parseFloat(f64, p) catch return null);
            if (qty) |q| tick.qty = std.fmt.parseInt(u64, q, 10) catch return null;
            return tick;
        } else if (mem.eql(u8, msg_type, "8")) {
            // Execution Report
            const sym    = self.get_field(@intFromEnum(FixTag.Symbol)) orelse return null;
            const last_px = self.get_field(@intFromEnum(FixTag.LastPx));
            const last_qty = self.get_field(@intFromEnum(FixTag.LastQty));
            const exec_type = self.get_field(@intFromEnum(FixTag.ExecType));

            const sym_len = @min(sym.len, 15);
            @memcpy(tick.symbol[0..sym_len], sym[0..sym_len]);
            tick.tick_type = TickType.OrderExec;

            if (last_px)  |p| tick.price = orderbook.from_double(
                std.fmt.parseFloat(f64, p) catch return null);
            if (last_qty) |q| tick.qty = std.fmt.parseInt(u64, q, 10) catch return null;
            _ = exec_type;
            return tick;
        }
        return null;
    }
};

pub const FixEncoder = struct {
    buf: []u8,
    pos: usize,
    seq_num: u32,

    const Self = @This();
    const DELIM: u8 = '|';

    pub fn init(buf: []u8) Self { return .{ .buf = buf, .pos = 0, .seq_num = 1 }; }

    pub fn write_field(self: *Self, tag: u32, value: []const u8) bool {
        // tag=value|
        const tag_str_buf = std.fmt.bufPrint(self.buf[self.pos..], "{d}={s}", .{tag, value})
            catch return false;
        self.pos += tag_str_buf.len;
        if (self.pos >= self.buf.len) return false;
        self.buf[self.pos] = DELIM;
        self.pos += 1;
        return true;
    }

    pub fn encode_new_order(self: *Self, sym: []const u8, side: Side, px: f64, qty: u64) bool {
        var px_buf: [32]u8 = undefined;
        var qty_buf: [32]u8 = undefined;
        var seq_buf: [16]u8 = undefined;
        const px_s   = std.fmt.bufPrint(&px_buf, "{d:.4}", .{px}) catch return false;
        const qty_s  = std.fmt.bufPrint(&qty_buf, "{d}", .{qty}) catch return false;
        const seq_s  = std.fmt.bufPrint(&seq_buf, "{d}", .{self.seq_num}) catch return false;
        self.seq_num += 1;

        return self.write_field(8, "FIX.4.4") and
               self.write_field(35, "D") and
               self.write_field(34, seq_s) and
               self.write_field(55, sym) and
               self.write_field(54, if (side == .Buy) "1" else "2") and
               self.write_field(44, px_s) and
               self.write_field(38, qty_s) and
               self.write_field(40, "2"); // Limit
    }

    pub fn written(self: *const Self) []const u8 {
        return self.buf[0..self.pos];
    }
};

// ============================================================
// Normalizer: converts any protocol to NormalizedTick
// ============================================================
pub const ProtocolType = enum {
    ITCH50,
    SBE,
    FIX44,
    Direct,
};

pub const Normalizer = struct {
    protocol: ProtocolType,
    venue_id: u16,

    const Self = @This();

    pub fn init(protocol: ProtocolType, venue_id: u16) Self {
        return .{ .protocol = protocol, .venue_id = venue_id };
    }

    pub fn normalize_itch(self: *const Self, msg: decoder.Message, ts_ns: i64) NormalizedTick {
        var tick = std.mem.zeroes(NormalizedTick);
        tick.timestamp_ns = ts_ns;
        tick.venue_id     = self.venue_id;

        switch (msg) {
            .add_order => |ao| {
                tick.tick_type = TickType.OrderNew;
                tick.side      = if (ao.side == 'B') 0 else 1;
                tick.price     = orderbook.from_double(
                    @as(f64, @floatFromInt(ao.price)) / 10000.0);
                tick.qty       = ao.shares;
                tick.order_id  = ao.order_reference;
                @memcpy(tick.symbol[0..8], ao.stock[0..8]);
            },
            .order_executed => |oe| {
                tick.tick_type = TickType.OrderExec;
                tick.qty       = oe.executed_shares;
                tick.order_id  = oe.order_reference;
            },
            .order_cancel => |oc| {
                tick.tick_type = TickType.OrderCancel;
                tick.qty       = oc.cancelled_shares;
                tick.order_id  = oc.order_reference;
            },
            .trade => |tr| {
                tick.tick_type = TickType.Trade;
                tick.side      = if (tr.side == 'B') 0 else 1;
                tick.price     = orderbook.from_double(
                    @as(f64, @floatFromInt(tr.price)) / 10000.0);
                tick.qty       = tr.shares;
                @memcpy(tick.symbol[0..8], tr.stock[0..8]);
            },
            else => {},
        }
        return tick;
    }
};

// ============================================================
// Multi-venue feed aggregator
// Merges ticks from multiple venues, deduplicates, resolves NBBO
// ============================================================
pub const VenueConfig = struct {
    venue_id:   u16,
    protocol:   ProtocolType,
    name:       [16]u8,
    priority:   u8,  // lower = higher priority
};

pub const NBBOState = struct {
    symbol:     [16]u8,
    best_bid:   i64 = 0,
    best_ask:   i64 = math.maxInt(i64),
    bid_qty:    u64 = 0,
    ask_qty:    u64 = 0,
    bid_venue:  u16 = 0,
    ask_venue:  u16 = 0,
    ts_ns:      i64 = 0,

    pub fn spread(self: *const NBBOState) f64 {
        if (self.best_ask == math.maxInt(i64) or self.best_bid == 0) return 0;
        return orderbook.to_double(self.best_ask) - orderbook.to_double(self.best_bid);
    }

    pub fn mid(self: *const NBBOState) f64 {
        if (self.best_ask == math.maxInt(i64) or self.best_bid == 0) return 0;
        return (orderbook.to_double(self.best_bid) + orderbook.to_double(self.best_ask)) / 2.0;
    }
};

pub const FeedAggregator = struct {
    nbbo_map: std.StringHashMap(NBBOState),
    tick_count: u64,
    dedup_count: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return .{
            .nbbo_map    = std.StringHashMap(NBBOState).init(allocator),
            .tick_count  = 0,
            .dedup_count = 0,
            .allocator   = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.nbbo_map.deinit();
    }

    // Process incoming normalized tick, update NBBO
    pub fn process(self: *Self, tick: *const NormalizedTick) !?NBBOState {
        self.tick_count += 1;
        const sym_len = blk: {
            var l: usize = 0;
            while (l < 16 and tick.symbol[l] != 0) l += 1;
            break :blk l;
        };
        const sym_slice = tick.symbol[0..sym_len];

        var entry = try self.nbbo_map.getOrPut(sym_slice);
        if (!entry.found_existing) {
            entry.value_ptr.* = std.mem.zeroes(NBBOState);
            @memcpy(entry.value_ptr.symbol[0..sym_len], sym_slice);
            entry.value_ptr.best_ask = math.maxInt(i64);
        }
        var nbbo = entry.value_ptr;

        const changed = switch (tick.tick_type) {
            .BidUpdate => blk: {
                if (tick.price > nbbo.best_bid) {
                    nbbo.best_bid  = tick.price;
                    nbbo.bid_qty   = tick.qty;
                    nbbo.bid_venue = tick.venue_id;
                    nbbo.ts_ns     = tick.timestamp_ns;
                    break :blk true;
                }
                break :blk false;
            },
            .AskUpdate => blk: {
                if (tick.price < nbbo.best_ask) {
                    nbbo.best_ask  = tick.price;
                    nbbo.ask_qty   = tick.qty;
                    nbbo.ask_venue = tick.venue_id;
                    nbbo.ts_ns     = tick.timestamp_ns;
                    break :blk true;
                }
                break :blk false;
            },
            else => false,
        };

        return if (changed) nbbo.* else null;
    }

    pub fn get_nbbo(self: *const Self, symbol: []const u8) ?NBBOState {
        return self.nbbo_map.get(symbol);
    }
};

// ============================================================
// Protocol benchmarks
// ============================================================
pub fn bench_sbe_decode(n: usize) void {
    // Build a fake SBE market data entry
    var buf: [64]u8 = undefined;
    buf[0] = 0; // Bid entry
    buf[1] = 0;
    // price mantissa = 1500000 (150.0000)
    const mantissa: i64 = 1500000;
    mem.writeInt(i64, buf[2..10][0..8], mantissa, .little);
    buf[10] = @bitCast(@as(i8, -4)); // exponent -4 → 150.0000
    @memset(buf[11..18], 0);
    // qty = 100
    const qty: u64 = 100;
    mem.writeInt(u64, buf[18..26][0..8], qty, .little);
    buf[26] = 1; // level 1
    buf[27] = 0; // action: New
    @memset(buf[28..34], 0);

    var count: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var dec = SbeDecoder.init(buf[0..34]);
        if (dec.decode_md_entry()) |_| count += 1;
    }
    _ = count;
}

pub fn bench_fix_parse(allocator: Allocator, n: usize) !void {
    const fix_msg = "8=FIX.4.4|35=D|49=SENDER|56=TARGET|34=1|52=20240101-09:30:00|" ++
                    "55=AAPL|54=1|44=182.50|38=100|40=2|";

    var dec = FixDecoder.init(allocator, fix_msg);
    defer dec.deinit();

    var count: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        _ = try dec.parse();
        count += 1;
    }
    _ = count;
}

pub fn bench_normalize_itch(n: usize) void {
    const norm = Normalizer.init(.ITCH50, 1);
    const ts: i64 = 34_200_000_000_000;

    // Build a synthetic AddOrder
    const msg = decoder.Message{
        .add_order = decoder.AddOrder{
            .stock_locate    = 1,
            .tracking_number = 0,
            .timestamp       = @as(u64, @intCast(ts)),
            .order_reference = 12345,
            .side            = 'B',
            .shares          = 100,
            .stock           = [_]u8{'A','A','P','L',' ',' ',' ',' '},
            .price           = 1825000, // 182.5000 in 4 decimal places
        },
    };

    var count: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const tick = norm.normalize_itch(msg, ts);
        if (tick.price > 0) count += 1;
    }
    _ = count;
}

// ============================================================
// Tests
// ============================================================
test "SBE header encode/decode roundtrip" {
    var buf: [8]u8 = undefined;
    var enc = SbeEncoder.init(&buf);
    const hdr_in = SbeHeader{ .block_length = 64, .template_id = 55, .schema_id = 1, .version = 1 };
    _ = enc.write_header(hdr_in);

    var dec = SbeDecoder.init(enc.written());
    const hdr_out = dec.decode_header();
    try std.testing.expect(hdr_out != null);
    try std.testing.expectEqual(hdr_in.template_id, hdr_out.?.template_id);
    try std.testing.expectEqual(hdr_in.block_length, hdr_out.?.block_length);
}

test "FIX new order encode and parse" {
    var buf: [512]u8 = undefined;
    var enc = FixEncoder.init(&buf);
    _ = enc.encode_new_order("AAPL", .Buy, 182.50, 100);

    var dec = FixDecoder.init(std.testing.allocator, enc.written());
    defer dec.deinit();
    const n = try dec.parse();
    try std.testing.expect(n >= 6);

    const msg_type = dec.get_field(@intFromEnum(FixTag.MsgType));
    try std.testing.expect(msg_type != null);
    try std.testing.expectEqualSlices(u8, "D", msg_type.?);
}

test "NBBO aggregator bid/ask update" {
    var agg = FeedAggregator.init(std.testing.allocator);
    defer agg.deinit();

    var bid_tick = std.mem.zeroes(NormalizedTick);
    bid_tick.tick_type = .BidUpdate;
    bid_tick.price     = orderbook.from_double(182.50);
    bid_tick.qty       = 100;
    bid_tick.venue_id  = 1;
    @memcpy(bid_tick.symbol[0..4], "AAPL");

    const nbbo_updated = try agg.process(&bid_tick);
    try std.testing.expect(nbbo_updated != null);
    try std.testing.expectEqual(bid_tick.price, nbbo_updated.?.best_bid);
}

test "ITCH normalize" {
    const norm = Normalizer.init(.ITCH50, 2);
    const msg = decoder.Message{
        .add_order = decoder.AddOrder{
            .stock_locate    = 1,
            .tracking_number = 0,
            .timestamp       = 1000000,
            .order_reference = 42,
            .side            = 'S',
            .shares          = 200,
            .stock           = [_]u8{'M','S','F','T',' ',' ',' ',' '},
            .price           = 3750000,
        },
    };
    const tick = norm.normalize_itch(msg, 1000000);
    try std.testing.expect(tick.price > 0);
    try std.testing.expectEqual(@as(u8, 1), tick.side);
    try std.testing.expectEqual(@as(u64, 200), tick.qty);
}

test "SBE md entry decode" {
    var buf: [64]u8 = std.mem.zeroes([64]u8);
    buf[0] = 0; // Bid
    const mantissa: i64 = 1500000;
    mem.writeInt(i64, buf[2..10][0..8], mantissa, .little);
    buf[10] = @bitCast(@as(i8, -4));
    const qty: u64 = 500;
    mem.writeInt(u64, buf[18..26][0..8], qty, .little);

    var dec = SbeDecoder.init(buf[0..34]);
    const entry = dec.decode_md_entry();
    try std.testing.expect(entry != null);
    try std.testing.expectEqual(TickType.BidUpdate, entry.?.tick_type);
    try std.testing.expectEqual(@as(u64, 500), entry.?.qty);
}
