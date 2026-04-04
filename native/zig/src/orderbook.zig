//! Zig Order Book: comptime-parameterized price levels,
//! sorted array for O(log n) price lookup, ArrayList for per-level orders.

const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;

pub const Side = enum(u8) { Buy = 0, Sell = 1 };
pub const OrderType = enum(u8) { Limit, Market, Stop, Iceberg };
pub const OrderStatus = enum(u8) { New, PartialFill, Filled, Cancelled };

pub const Price    = i64;  // fixed-point * 100000
pub const Quantity = u64;
pub const OrderId  = u64;
pub const Timestamp = i64;

pub const PRICE_SCALE: f64 = 100000.0;

pub inline fn to_double(p: Price) f64 { return @as(f64, @floatFromInt(p)) / PRICE_SCALE; }
pub inline fn from_double(d: f64) Price { return @intFromFloat(d * PRICE_SCALE + 0.5); }

pub const Order = struct {
    id:          OrderId,
    symbol:      [16]u8,
    side:        Side,
    order_type:  OrderType,
    status:      OrderStatus,
    price:       Price,
    qty:         Quantity,
    filled_qty:  Quantity,
    timestamp:   Timestamp,
    sequence:    u64,

    pub fn leaves_qty(self: *const Order) Quantity {
        return if (self.qty > self.filled_qty) self.qty - self.filled_qty else 0;
    }
    pub fn is_active(self: *const Order) bool {
        return self.status == .New or self.status == .PartialFill;
    }
    pub fn symbol_str(self: *const Order) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.symbol, 0) orelse 16;
        return self.symbol[0..len];
    }
};

pub const PriceLevel = struct {
    price:       Price,
    total_qty:   Quantity,
    orders:      ArrayList(OrderId),

    pub fn init(allocator: Allocator, price: Price) PriceLevel {
        return .{
            .price     = price,
            .total_qty = 0,
            .orders    = ArrayList(OrderId).init(allocator),
        };
    }

    pub fn deinit(self: *PriceLevel) void {
        self.orders.deinit();
    }
};

pub const Trade = struct {
    trade_id:       u64,
    aggressor_id:   OrderId,
    passive_id:     OrderId,
    symbol:         [16]u8,
    aggressor_side: Side,
    price:          Price,
    qty:            Quantity,
    timestamp:      Timestamp,
};

pub const BookSide = enum { Bid, Ask };

// Sorted price-level array with O(log n) lookup
// Bids: descending order; Asks: ascending order
pub const PriceLevelArray = struct {
    levels:    ArrayList(PriceLevel),
    side:      BookSide,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, side: BookSide) Self {
        return .{
            .levels    = ArrayList(PriceLevel).init(allocator),
            .side      = side,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.levels.items) |*lvl| lvl.deinit();
        self.levels.deinit();
    }

    // Binary search for price; returns index or insertion point
    pub fn find_index(self: *const Self, price: Price) ?usize {
        var lo: usize = 0;
        var hi: usize = self.levels.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const p = self.levels.items[mid].price;
            if (p == price) return mid;
            if (self.side == .Bid) {
                // Descending: higher prices first
                if (p > price) lo = mid + 1 else hi = mid;
            } else {
                // Ascending: lower prices first
                if (p < price) lo = mid + 1 else hi = mid;
            }
        }
        return null; // not found
    }

    // Get or create price level; returns pointer
    pub fn get_or_create(self: *Self, price: Price) !*PriceLevel {
        var lo: usize = 0;
        var hi: usize = self.levels.items.len;
        var ins: usize = 0;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const p = self.levels.items[mid].price;
            if (p == price) return &self.levels.items[mid];
            if (self.side == .Bid) {
                if (p > price) { lo = mid + 1; ins = lo; }
                else           { hi = mid; ins = mid; }
            } else {
                if (p < price) { lo = mid + 1; ins = lo; }
                else           { hi = mid; ins = mid; }
            }
        }
        if (lo == hi) ins = lo;

        // Insert new level at ins
        const new_level = PriceLevel.init(self.allocator, price);
        try self.levels.insert(ins, new_level);
        return &self.levels.items[ins];
    }

    pub fn remove_level(self: *Self, price: Price) void {
        if (self.find_index(price)) |idx| {
            self.levels.items[idx].deinit();
            _ = self.levels.orderedRemove(idx);
        }
    }

    pub fn best_price(self: *const Self) ?Price {
        if (self.levels.items.len == 0) return null;
        return self.levels.items[0].price;
    }

    pub fn total_depth(self: *const Self) Quantity {
        var d: Quantity = 0;
        for (self.levels.items) |lvl| d += lvl.total_qty;
        return d;
    }

    pub fn level_count(self: *const Self) usize {
        return self.levels.items.len;
    }
};

// ============================================================
// Main Order Book
// ============================================================
pub const OrderBookStats = struct {
    best_bid:    ?Price,
    best_ask:    ?Price,
    spread:      ?Price,
    mid_price:   ?f64,
    bid_depth:   Quantity,
    ask_depth:   Quantity,
    imbalance:   f64,
    bid_levels:  usize,
    ask_levels:  usize,
    total_trades: u64,
    total_volume: Quantity,
    vwap:        f64,
};

pub const OrderBook = struct {
    symbol:       [16]u8,
    bids:         PriceLevelArray,
    asks:         PriceLevelArray,
    orders:       AutoHashMap(OrderId, Order),
    trades:       ArrayList(Trade),
    allocator:    Allocator,
    trade_id_gen: u64 = 1,
    seq_gen:      u64 = 1,
    vwap_num:     f64 = 0.0,
    vwap_den:     u64 = 0,

    const Self = @This();

    pub fn init(allocator: Allocator, symbol: []const u8) Self {
        var sym: [16]u8 = [_]u8{0} ** 16;
        const len = @min(symbol.len, 15);
        @memcpy(sym[0..len], symbol[0..len]);

        return .{
            .symbol    = sym,
            .bids      = PriceLevelArray.init(allocator, .Bid),
            .asks      = PriceLevelArray.init(allocator, .Ask),
            .orders    = AutoHashMap(OrderId, Order).init(allocator),
            .trades    = ArrayList(Trade).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.bids.deinit();
        self.asks.deinit();
        self.orders.deinit();
        self.trades.deinit();
    }

    fn next_trade_id(self: *Self) u64 {
        const id = self.trade_id_gen;
        self.trade_id_gen += 1;
        return id;
    }
    fn next_seq(self: *Self) u64 {
        const s = self.seq_gen;
        self.seq_gen += 1;
        return s;
    }

    pub fn add_order(self: *Self, order: Order) !std.ArrayList(Trade) {
        var result = std.ArrayList(Trade).init(self.allocator);
        var o = order;
        o.sequence = self.next_seq();

        if (o.order_type == .Market) {
            try self.match_order(&o, &result);
            return result;
        }

        // Try matching first
        try self.match_limit_order(&o, &result);

        // Rest on book if still active
        if (o.is_active() and o.order_type == .Limit) {
            try self.orders.put(o.id, o);
            const book_side: *PriceLevelArray = if (o.side == .Buy) &self.bids else &self.asks;
            const level = try book_side.get_or_create(o.price);
            level.total_qty += o.leaves_qty();
            try level.orders.append(o.id);
        }
        return result;
    }

    pub fn cancel_order(self: *Self, id: OrderId) bool {
        const entry = self.orders.fetchRemove(id) orelse return false;
        var o = entry.value;
        const remaining = o.leaves_qty();
        o.status = .Cancelled;

        const book_side: *PriceLevelArray = if (o.side == .Buy) &self.bids else &self.asks;
        if (book_side.find_index(o.price)) |idx| {
            const level = &book_side.levels.items[idx];
            // Remove order id from level list
            for (level.orders.items, 0..) |oid, i| {
                if (oid == id) {
                    _ = level.orders.orderedRemove(i);
                    break;
                }
            }
            if (remaining <= level.total_qty)
                level.total_qty -= remaining
            else
                level.total_qty = 0;
            if (level.orders.items.len == 0)
                book_side.remove_level(o.price);
        }
        return true;
    }

    fn match_order(self: *Self, aggressor: *Order, trades: *ArrayList(Trade)) !void {
        if (aggressor.side == .Buy) {
            // Buy market: match against asks (ascending price)
            while (aggressor.is_active() and self.asks.level_count() > 0) {
                const level = &self.asks.levels.items[0];
                try self.match_level(aggressor, level, &self.asks, trades);
                if (level.orders.items.len == 0)
                    self.asks.remove_level(level.price);
            }
        } else {
            while (aggressor.is_active() and self.bids.level_count() > 0) {
                const level = &self.bids.levels.items[0];
                try self.match_level(aggressor, level, &self.bids, trades);
                if (level.orders.items.len == 0)
                    self.bids.remove_level(level.price);
            }
        }
    }

    fn match_limit_order(self: *Self, aggressor: *Order, trades: *ArrayList(Trade)) !void {
        if (aggressor.side == .Buy) {
            while (aggressor.is_active() and self.asks.level_count() > 0) {
                const level = &self.asks.levels.items[0];
                if (level.price > aggressor.price) break;
                try self.match_level(aggressor, level, &self.asks, trades);
                if (level.orders.items.len == 0)
                    self.asks.remove_level(level.price);
            }
        } else {
            while (aggressor.is_active() and self.bids.level_count() > 0) {
                const level = &self.bids.levels.items[0];
                if (level.price < aggressor.price) break;
                try self.match_level(aggressor, level, &self.bids, trades);
                if (level.orders.items.len == 0)
                    self.bids.remove_level(level.price);
            }
        }
    }

    fn match_level(self: *Self, aggressor: *Order, level: *PriceLevel,
                   book_side: *PriceLevelArray, trades: *ArrayList(Trade)) !void
    {
        _ = book_side;
        var i: usize = 0;
        while (i < level.orders.items.len and aggressor.is_active()) {
            const passive_id = level.orders.items[i];
            const passive_ptr = self.orders.getPtr(passive_id) orelse { i += 1; continue; };
            const fill = @min(aggressor.leaves_qty(), passive_ptr.leaves_qty());
            if (fill == 0) { i += 1; continue; }

            // Execute fill
            passive_ptr.filled_qty += fill;
            aggressor.filled_qty   += fill;

            if (passive_ptr.filled_qty >= passive_ptr.qty)
                passive_ptr.status = .Filled
            else
                passive_ptr.status = .PartialFill;

            if (aggressor.filled_qty >= aggressor.qty)
                aggressor.status = .Filled
            else
                aggressor.status = .PartialFill;

            if (level.total_qty >= fill) level.total_qty -= fill
            else level.total_qty = 0;

            // Record trade
            const t = Trade{
                .trade_id       = self.next_trade_id(),
                .aggressor_id   = aggressor.id,
                .passive_id     = passive_id,
                .symbol         = self.symbol,
                .aggressor_side = aggressor.side,
                .price          = level.price,
                .qty            = fill,
                .timestamp      = std.time.nanoTimestamp(),
            };
            try trades.append(t);
            try self.trades.append(t);
            self.vwap_num += to_double(level.price) * @as(f64, @floatFromInt(fill));
            self.vwap_den += fill;

            if (passive_ptr.status == .Filled) {
                _ = self.orders.remove(passive_id);
                _ = level.orders.orderedRemove(i);
                // Don't increment i (element shifted)
            } else {
                i += 1;
            }
        }
    }

    pub fn best_bid(self: *const Self) ?Price { return self.bids.best_price(); }
    pub fn best_ask(self: *const Self) ?Price { return self.asks.best_price(); }

    pub fn mid_price(self: *const Self) ?f64 {
        const bb = self.best_bid() orelse return null;
        const ba = self.best_ask() orelse return null;
        return (to_double(bb) + to_double(ba)) / 2.0;
    }

    pub fn spread(self: *const Self) ?f64 {
        const bb = self.best_bid() orelse return null;
        const ba = self.best_ask() orelse return null;
        return to_double(ba) - to_double(bb);
    }

    pub fn stats(self: *const Self) OrderBookStats {
        const bb = self.best_bid();
        const ba = self.best_ask();
        var sp: ?Price = null;
        var mid: ?f64 = null;
        if (bb != null and ba != null) {
            sp  = ba.? - bb.?;
            mid = (to_double(bb.?) + to_double(ba.?)) / 2.0;
        }
        const bd = self.bids.total_depth();
        const ad = self.asks.total_depth();
        const tot = bd + ad;
        const imb: f64 = if (tot > 0)
            (@as(f64, @floatFromInt(bd)) - @as(f64, @floatFromInt(ad))) /
            @as(f64, @floatFromInt(tot))
        else 0.0;

        return .{
            .best_bid     = bb,
            .best_ask     = ba,
            .spread       = sp,
            .mid_price    = mid,
            .bid_depth    = bd,
            .ask_depth    = ad,
            .imbalance    = imb,
            .bid_levels   = self.bids.level_count(),
            .ask_levels   = self.asks.level_count(),
            .total_trades = @intCast(self.trades.items.len),
            .total_volume = self.vwap_den,
            .vwap         = if (self.vwap_den > 0) self.vwap_num / @as(f64, @floatFromInt(self.vwap_den)) else 0.0,
        };
    }

    // L2 snapshot: top `depth` levels
    pub fn bid_snapshot(self: *const Self, depth: usize) []const PriceLevel {
        const n = @min(depth, self.bids.levels.items.len);
        return self.bids.levels.items[0..n];
    }
    pub fn ask_snapshot(self: *const Self, depth: usize) []const PriceLevel {
        const n = @min(depth, self.asks.levels.items.len);
        return self.asks.levels.items[0..n];
    }
};

// ============================================================
// Tests
// ============================================================
test "basic orderbook operations" {
    const alloc = std.testing.allocator;
    var book = OrderBook.init(alloc, "AAPL");
    defer book.deinit();

    // Add bids
    for (0..5) |i| {
        const bid = Order{
            .id         = @intCast(i + 1),
            .symbol     = [_]u8{'A','A','P','L',0,0,0,0,0,0,0,0,0,0,0,0},
            .side       = .Buy,
            .order_type = .Limit,
            .status     = .New,
            .price      = from_double(150.0 - @as(f64, @floatFromInt(i)) * 0.1),
            .qty        = 100,
            .filled_qty = 0,
            .timestamp  = 0,
            .sequence   = 0,
        };
        var fills = try book.add_order(bid);
        fills.deinit();
    }

    // Add asks
    for (0..5) |i| {
        const ask = Order{
            .id         = @intCast(100 + i),
            .symbol     = [_]u8{'A','A','P','L',0,0,0,0,0,0,0,0,0,0,0,0},
            .side       = .Sell,
            .order_type = .Limit,
            .status     = .New,
            .price      = from_double(150.1 + @as(f64, @floatFromInt(i)) * 0.1),
            .qty        = 100,
            .filled_qty = 0,
            .timestamp  = 0,
            .sequence   = 0,
        };
        var fills = try book.add_order(ask);
        fills.deinit();
    }

    const bb = book.best_bid() orelse return error.NoBid;
    const ba = book.best_ask() orelse return error.NoAsk;
    try std.testing.expect(to_double(bb) > 149.0);
    try std.testing.expect(to_double(ba) > 150.0);
    try std.testing.expect(book.spread().? > 0.0);

    const st = book.stats();
    try std.testing.expectEqual(@as(usize, 5), st.bid_levels);
    try std.testing.expectEqual(@as(usize, 5), st.ask_levels);
}

test "matching" {
    const alloc = std.testing.allocator;
    var book = OrderBook.init(alloc, "TSLA");
    defer book.deinit();

    // Passive bid
    const bid = Order{
        .id = 1, .symbol = [_]u8{'T','S','L','A',0}++[_]u8{0}**11,
        .side = .Buy, .order_type = .Limit, .status = .New,
        .price = from_double(200.0), .qty = 100, .filled_qty = 0,
        .timestamp = 0, .sequence = 0,
    };
    var f1 = try book.add_order(bid);
    defer f1.deinit();
    try std.testing.expectEqual(@as(usize, 0), f1.items.len);

    // Aggressive sell that crosses
    const ask = Order{
        .id = 2, .symbol = [_]u8{'T','S','L','A',0}++[_]u8{0}**11,
        .side = .Sell, .order_type = .Limit, .status = .New,
        .price = from_double(199.0), .qty = 60, .filled_qty = 0,
        .timestamp = 0, .sequence = 0,
    };
    var f2 = try book.add_order(ask);
    defer f2.deinit();
    try std.testing.expectEqual(@as(usize, 1), f2.items.len);
    try std.testing.expectEqual(@as(u64, 60), f2.items[0].qty);
}

test "cancel order" {
    const alloc = std.testing.allocator;
    var book = OrderBook.init(alloc, "SPY");
    defer book.deinit();

    const bid = Order{
        .id = 99, .symbol = [_]u8{'S','P','Y',0}++[_]u8{0}**12,
        .side = .Buy, .order_type = .Limit, .status = .New,
        .price = from_double(400.0), .qty = 500, .filled_qty = 0,
        .timestamp = 0, .sequence = 0,
    };
    var f = try book.add_order(bid);
    defer f.deinit();
    try std.testing.expectEqual(@as(usize, 1), book.bids.level_count());

    const ok = book.cancel_order(99);
    try std.testing.expect(ok);
    try std.testing.expectEqual(@as(usize, 0), book.bids.level_count());
}
