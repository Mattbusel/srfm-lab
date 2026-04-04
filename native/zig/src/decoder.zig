//! ITCH 5.0 Protocol Decoder (Nasdaq)
//! Decodes binary ITCH messages from raw UDP/TCP bytes.
//! Reference: https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf

const std = @import("std");
const mem = std.mem;

// ============================================================
// ITCH 5.0 Message Types
// ============================================================
pub const MessageType = enum(u8) {
    SystemEvent          = 'S',
    StockDirectory       = 'R',
    StockTradingAction   = 'H',
    RegSHORestriction    = 'Y',
    MarketParticipant    = 'L',
    MWCBDeclineLevel     = 'V',
    MWCBStatus           = 'W',
    IPOQuotingPeriod     = 'K',
    LULDAuctionCollar    = 'J',
    OperationalHalt      = 'h',
    AddOrder             = 'A',
    AddOrderMPID         = 'F',
    OrderExecuted        = 'E',
    OrderExecutedPrice   = 'C',
    OrderCancel          = 'X',
    OrderDelete          = 'D',
    OrderReplace         = 'U',
    Trade                = 'P',
    CrossTrade           = 'Q',
    BrokenTrade          = 'B',
    NOII                 = 'I',
    RPII                 = 'N',
    Unknown              = 0,
};

pub const Side = enum(u8) {
    Buy  = 'B',
    Sell = 'S',
};

// ============================================================
// Parsed message structs
// ============================================================
pub const SystemEvent = struct {
    msg_type:    MessageType,
    stock_locate: u16,
    tracking_num: u16,
    timestamp:   u64,  // nanoseconds since midnight
    event_code:  u8,   // 'O'=open, 'S'=start, 'Q'=mkt-open, etc.
};

pub const StockDirectory = struct {
    stock:                [8]u8,
    market_category:      u8,
    financial_status:     u8,
    round_lot_size:       u32,
    round_lots_only:      bool,
    issue_classification: u8,
    issue_subtype:        [2]u8,
    authenticity:         u8,
    short_sale_threshold: bool,
    ipo_flag:             bool,
    luld_ref_price_tier:  u8,
    etp_flag:             bool,
    etp_leverage:         u32,
    inverse_indicator:    bool,
};

pub const AddOrder = struct {
    msg_type:     MessageType,
    stock_locate: u16,
    tracking_num: u16,
    timestamp:    u64,
    order_ref:    u64,
    side:         Side,
    shares:       u32,
    stock:        [8]u8,
    price:        u32,  // fixed-point: actual * 10000
    mpid:         ?[4]u8, // non-null for AddOrderMPID ('F')

    pub fn price_f64(self: *const AddOrder) f64 {
        return @as(f64, @floatFromInt(self.price)) / 10000.0;
    }
};

pub const OrderExecuted = struct {
    msg_type:      MessageType,
    stock_locate:  u16,
    tracking_num:  u16,
    timestamp:     u64,
    order_ref:     u64,
    executed_shares: u32,
    match_number:  u64,
    execution_price: ?u32, // non-null for OrderExecutedPrice ('C')
    printable:       ?bool,
};

pub const OrderCancel = struct {
    msg_type:         MessageType,
    stock_locate:     u16,
    tracking_num:     u16,
    timestamp:        u64,
    order_ref:        u64,
    cancelled_shares: u32,
};

pub const OrderDelete = struct {
    msg_type:     MessageType,
    stock_locate: u16,
    tracking_num: u16,
    timestamp:    u64,
    order_ref:    u64,
};

pub const OrderReplace = struct {
    msg_type:          MessageType,
    stock_locate:      u16,
    tracking_num:      u16,
    timestamp:         u64,
    orig_order_ref:    u64,
    new_order_ref:     u64,
    shares:            u32,
    price:             u32,
};

pub const Trade = struct {
    msg_type:     MessageType,
    stock_locate: u16,
    tracking_num: u16,
    timestamp:    u64,
    order_ref:    u64,
    side:         Side,
    shares:       u32,
    stock:        [8]u8,
    price:        u32,
    match_number: u64,

    pub fn price_f64(self: *const Trade) f64 {
        return @as(f64, @floatFromInt(self.price)) / 10000.0;
    }
};

// Unified message union
pub const Message = union(MessageType) {
    SystemEvent:        SystemEvent,
    StockDirectory:     StockDirectory,
    StockTradingAction: void,
    RegSHORestriction:  void,
    MarketParticipant:  void,
    MWCBDeclineLevel:   void,
    MWCBStatus:         void,
    IPOQuotingPeriod:   void,
    LULDAuctionCollar:  void,
    OperationalHalt:    void,
    AddOrder:           AddOrder,
    AddOrderMPID:       AddOrder,
    OrderExecuted:      OrderExecuted,
    OrderExecutedPrice: OrderExecuted,
    OrderCancel:        OrderCancel,
    OrderDelete:        OrderDelete,
    OrderReplace:       OrderReplace,
    Trade:              Trade,
    CrossTrade:         void,
    BrokenTrade:        void,
    NOII:               void,
    RPII:               void,
    Unknown:            void,
};

// ============================================================
// Decoder
// ============================================================
pub const DecodeError = error{
    BufferTooSmall,
    InvalidMessageType,
    InvalidLength,
};

// Read big-endian integers (ITCH is big-endian)
fn read_u16_be(buf: []const u8, offset: usize) u16 {
    return @as(u16, buf[offset]) << 8 | @as(u16, buf[offset+1]);
}
fn read_u32_be(buf: []const u8, offset: usize) u32 {
    return @as(u32, buf[offset])   << 24 |
           @as(u32, buf[offset+1]) << 16 |
           @as(u32, buf[offset+2]) <<  8 |
           @as(u32, buf[offset+3]);
}
fn read_u48_be(buf: []const u8, offset: usize) u64 {
    var v: u64 = 0;
    for (0..6) |i| v = (v << 8) | @as(u64, buf[offset+i]);
    return v;
}
fn read_u64_be(buf: []const u8, offset: usize) u64 {
    var v: u64 = 0;
    for (0..8) |i| v = (v << 8) | @as(u64, buf[offset+i]);
    return v;
}

fn copy_stock(buf: []const u8, offset: usize) [8]u8 {
    var s: [8]u8 = undefined;
    @memcpy(&s, buf[offset..offset+8]);
    return s;
}

pub const Decoder = struct {
    stats: DecoderStats = .{},

    pub const DecoderStats = struct {
        messages_decoded: u64 = 0,
        bytes_consumed:   u64 = 0,
        decode_errors:    u64 = 0,
        add_orders:       u64 = 0,
        executions:       u64 = 0,
        cancels:          u64 = 0,
        deletes:          u64 = 0,
        replaces:         u64 = 0,
        trades:           u64 = 0,
        system_events:    u64 = 0,
    };

    // Decode a single ITCH message from `buf`.
    // ITCH messages are length-prefixed (2-byte big-endian length).
    // Returns: (message, bytes_consumed) or error
    pub fn decode_one(self: *Decoder, buf: []const u8) DecodeError!struct { msg: Message, len: usize } {
        if (buf.len < 3) return error.BufferTooSmall;

        const msg_len = read_u16_be(buf, 0);
        if (buf.len < @as(usize, msg_len) + 2) return error.BufferTooSmall;
        if (msg_len == 0) return error.InvalidLength;

        const data = buf[2 .. 2 + msg_len];
        const msg_type_byte = data[0];
        const msg_type: MessageType = @enumFromInt(msg_type_byte);

        const result = switch (msg_type) {
            .SystemEvent => blk: {
                if (data.len < 12) return error.InvalidLength;
                self.stats.system_events += 1;
                break :blk Message{ .SystemEvent = .{
                    .msg_type     = .SystemEvent,
                    .stock_locate = read_u16_be(data, 1),
                    .tracking_num = read_u16_be(data, 3),
                    .timestamp    = read_u48_be(data, 5),
                    .event_code   = data[11],
                }};
            },
            .AddOrder => blk: {
                if (data.len < 36) return error.InvalidLength;
                self.stats.add_orders += 1;
                break :blk Message{ .AddOrder = .{
                    .msg_type     = .AddOrder,
                    .stock_locate = read_u16_be(data, 1),
                    .tracking_num = read_u16_be(data, 3),
                    .timestamp    = read_u48_be(data, 5),
                    .order_ref    = read_u64_be(data, 11),
                    .side         = if (data[19] == 'B') .Buy else .Sell,
                    .shares       = read_u32_be(data, 20),
                    .stock        = copy_stock(data, 24),
                    .price        = read_u32_be(data, 32),
                    .mpid         = null,
                }};
            },
            .AddOrderMPID => blk: {
                if (data.len < 40) return error.InvalidLength;
                self.stats.add_orders += 1;
                var mpid: [4]u8 = undefined;
                @memcpy(&mpid, data[36..40]);
                break :blk Message{ .AddOrderMPID = .{
                    .msg_type     = .AddOrderMPID,
                    .stock_locate = read_u16_be(data, 1),
                    .tracking_num = read_u16_be(data, 3),
                    .timestamp    = read_u48_be(data, 5),
                    .order_ref    = read_u64_be(data, 11),
                    .side         = if (data[19] == 'B') .Buy else .Sell,
                    .shares       = read_u32_be(data, 20),
                    .stock        = copy_stock(data, 24),
                    .price        = read_u32_be(data, 32),
                    .mpid         = mpid,
                }};
            },
            .OrderExecuted => blk: {
                if (data.len < 31) return error.InvalidLength;
                self.stats.executions += 1;
                break :blk Message{ .OrderExecuted = .{
                    .msg_type         = .OrderExecuted,
                    .stock_locate     = read_u16_be(data, 1),
                    .tracking_num     = read_u16_be(data, 3),
                    .timestamp        = read_u48_be(data, 5),
                    .order_ref        = read_u64_be(data, 11),
                    .executed_shares  = read_u32_be(data, 19),
                    .match_number     = read_u64_be(data, 23),
                    .execution_price  = null,
                    .printable        = null,
                }};
            },
            .OrderExecutedPrice => blk: {
                if (data.len < 36) return error.InvalidLength;
                self.stats.executions += 1;
                break :blk Message{ .OrderExecutedPrice = .{
                    .msg_type         = .OrderExecutedPrice,
                    .stock_locate     = read_u16_be(data, 1),
                    .tracking_num     = read_u16_be(data, 3),
                    .timestamp        = read_u48_be(data, 5),
                    .order_ref        = read_u64_be(data, 11),
                    .executed_shares  = read_u32_be(data, 19),
                    .match_number     = read_u64_be(data, 23),
                    .printable        = data[31] == 'Y',
                    .execution_price  = read_u32_be(data, 32),
                }};
            },
            .OrderCancel => blk: {
                if (data.len < 23) return error.InvalidLength;
                self.stats.cancels += 1;
                break :blk Message{ .OrderCancel = .{
                    .msg_type         = .OrderCancel,
                    .stock_locate     = read_u16_be(data, 1),
                    .tracking_num     = read_u16_be(data, 3),
                    .timestamp        = read_u48_be(data, 5),
                    .order_ref        = read_u64_be(data, 11),
                    .cancelled_shares = read_u32_be(data, 19),
                }};
            },
            .OrderDelete => blk: {
                if (data.len < 19) return error.InvalidLength;
                self.stats.deletes += 1;
                break :blk Message{ .OrderDelete = .{
                    .msg_type     = .OrderDelete,
                    .stock_locate = read_u16_be(data, 1),
                    .tracking_num = read_u16_be(data, 3),
                    .timestamp    = read_u48_be(data, 5),
                    .order_ref    = read_u64_be(data, 11),
                }};
            },
            .OrderReplace => blk: {
                if (data.len < 35) return error.InvalidLength;
                self.stats.replaces += 1;
                break :blk Message{ .OrderReplace = .{
                    .msg_type         = .OrderReplace,
                    .stock_locate     = read_u16_be(data, 1),
                    .tracking_num     = read_u16_be(data, 3),
                    .timestamp        = read_u48_be(data, 5),
                    .orig_order_ref   = read_u64_be(data, 11),
                    .new_order_ref    = read_u64_be(data, 19),
                    .shares           = read_u32_be(data, 27),
                    .price            = read_u32_be(data, 31),
                }};
            },
            .Trade => blk: {
                if (data.len < 44) return error.InvalidLength;
                self.stats.trades += 1;
                break :blk Message{ .Trade = .{
                    .msg_type     = .Trade,
                    .stock_locate = read_u16_be(data, 1),
                    .tracking_num = read_u16_be(data, 3),
                    .timestamp    = read_u48_be(data, 5),
                    .order_ref    = read_u64_be(data, 11),
                    .side         = if (data[19] == 'B') .Buy else .Sell,
                    .shares       = read_u32_be(data, 20),
                    .stock        = copy_stock(data, 24),
                    .price        = read_u32_be(data, 32),
                    .match_number = read_u64_be(data, 36),
                }};
            },
            else => Message{ .Unknown = {} },
        };

        const total = 2 + msg_len;
        self.stats.messages_decoded += 1;
        self.stats.bytes_consumed += total;

        return .{ .msg = result, .len = total };
    }

    // Decode a buffer containing multiple length-prefixed messages
    // Calls `callback` for each decoded message.
    // Returns number of messages decoded.
    pub fn decode_buffer(
        self: *Decoder,
        buf: []const u8,
        callback: anytype,
    ) usize {
        var pos: usize = 0;
        var count: usize = 0;
        while (pos < buf.len) {
            const result = self.decode_one(buf[pos..]) catch |err| {
                self.stats.decode_errors += 1;
                _ = err;
                break;
            };
            callback(result.msg);
            pos += result.len;
            count += 1;
        }
        return count;
    }

    // Build a synthetic ITCH AddOrder message for testing
    pub fn build_add_order(
        stock_locate: u16,
        order_ref: u64,
        side: Side,
        shares: u32,
        stock: []const u8,
        price: u32,
        timestamp: u64,
    ) [38]u8 {
        var buf: [38]u8 = undefined;
        // 2-byte length (big-endian) = 36
        buf[0] = 0;
        buf[1] = 36;
        buf[2] = 'A';
        buf[3] = @intCast(stock_locate >> 8);
        buf[4] = @intCast(stock_locate & 0xFF);
        buf[5] = 0; buf[6] = 0; // tracking num
        // 6-byte timestamp
        const ts = timestamp;
        buf[7]  = @intCast((ts >> 40) & 0xFF);
        buf[8]  = @intCast((ts >> 32) & 0xFF);
        buf[9]  = @intCast((ts >> 24) & 0xFF);
        buf[10] = @intCast((ts >> 16) & 0xFF);
        buf[11] = @intCast((ts >>  8) & 0xFF);
        buf[12] = @intCast(ts & 0xFF);
        // 8-byte order ref
        var or2 = order_ref;
        for (0..8) |i| { buf[20-i] = @intCast(or2 & 0xFF); or2 >>= 8; }
        // side
        buf[21] = if (side == .Buy) 'B' else 'S';
        // shares (4 bytes BE)
        buf[22] = @intCast((shares >> 24) & 0xFF);
        buf[23] = @intCast((shares >> 16) & 0xFF);
        buf[24] = @intCast((shares >>  8) & 0xFF);
        buf[25] = @intCast(shares & 0xFF);
        // stock (8 bytes, space-padded)
        var stock_arr: [8]u8 = [_]u8{' '} ** 8;
        const copy_len = @min(stock.len, 8);
        @memcpy(stock_arr[0..copy_len], stock[0..copy_len]);
        @memcpy(buf[26..34], &stock_arr);
        // price (4 bytes BE)
        buf[34] = @intCast((price >> 24) & 0xFF);
        buf[35] = @intCast((price >> 16) & 0xFF);
        buf[36] = @intCast((price >>  8) & 0xFF);
        buf[37] = @intCast(price & 0xFF);
        return buf;
    }
};

// ============================================================
// Tests
// ============================================================
test "decode AddOrder" {
    var decoder = Decoder{};
    const msg_bytes = Decoder.build_add_order(1, 12345678, .Buy, 100, "AAPL    ", 1500000, 1000000000);
    const result = try decoder.decode_one(&msg_bytes);
    try std.testing.expectEqual(MessageType.AddOrder, result.msg);

    const ao = result.msg.AddOrder;
    try std.testing.expectEqual(@as(u64, 12345678), ao.order_ref);
    try std.testing.expectEqual(Side.Buy, ao.side);
    try std.testing.expectEqual(@as(u32, 100), ao.shares);
    try std.testing.expectEqual(@as(u32, 1500000), ao.price);
    try std.testing.expectApproxEqAbs(@as(f64, 150.0), ao.price_f64(), 1e-4);
}

test "decoder stats" {
    var decoder = Decoder{};
    for (0..5) |_| {
        const msg_bytes = Decoder.build_add_order(1, 1, .Sell, 50, "TSLA    ", 2000000, 0);
        _ = try decoder.decode_one(&msg_bytes);
    }
    try std.testing.expectEqual(@as(u64, 5), decoder.stats.messages_decoded);
    try std.testing.expectEqual(@as(u64, 5), decoder.stats.add_orders);
}

test "buffer too small" {
    var decoder = Decoder{};
    var tiny = [_]u8{0, 35}; // says 35 bytes but buffer is only 2
    const result = decoder.decode_one(&tiny);
    try std.testing.expectError(error.BufferTooSmall, result);
}
