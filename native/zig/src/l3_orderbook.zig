const std = @import("std");
const math = std.math;
const mem = std.mem;
const Allocator = std.mem.Allocator;

// ============================================================================
// Core Types
// ============================================================================

pub const Side = enum(u8) { bid, ask };
pub const OrderType = enum(u8) { limit, market, ioc, fok, stop, stop_limit };
pub const TimeInForce = enum(u8) { gtc, ioc, fok, day };

pub const OrderId = u64;
pub const Price = i64; // fixed-point: price * 1e8
pub const Quantity = u64;
pub const Timestamp = u64; // nanoseconds

pub const PRICE_SCALE: f64 = 1e8;

pub fn priceToFixed(p: f64) i64 {
    return @intFromFloat(p * PRICE_SCALE);
}

pub fn fixedToPrice(p: i64) f64 {
    return @as(f64, @floatFromInt(p)) / PRICE_SCALE;
}

pub const Order = struct {
    id: OrderId,
    side: Side,
    price: Price,
    quantity: Quantity,
    filled: Quantity,
    timestamp: Timestamp,
    order_type: OrderType,
    tif: TimeInForce,
    // Intrusive linked list pointers
    prev: ?*Order,
    next: ?*Order,
    level: ?*PriceLevel,

    pub fn remaining(self: *const Order) Quantity {
        return self.quantity - self.filled;
    }

    pub fn isFilled(self: *const Order) bool {
        return self.filled >= self.quantity;
    }
};

// ============================================================================
// PriceLevel: doubly-linked list of orders at same price
// ============================================================================

pub const PriceLevel = struct {
    price: Price,
    total_quantity: Quantity,
    order_count: u32,
    head: ?*Order,
    tail: ?*Order,

    pub fn init(price: Price) PriceLevel {
        return .{
            .price = price,
            .total_quantity = 0,
            .order_count = 0,
            .head = null,
            .tail = null,
        };
    }

    pub fn appendOrder(self: *PriceLevel, order: *Order) void {
        order.prev = self.tail;
        order.next = null;
        order.level = self;
        if (self.tail) |t| {
            t.next = order;
        } else {
            self.head = order;
        }
        self.tail = order;
        self.total_quantity += order.remaining();
        self.order_count += 1;
    }

    pub fn removeOrder(self: *PriceLevel, order: *Order) void {
        if (order.prev) |p| {
            p.next = order.next;
        } else {
            self.head = order.next;
        }
        if (order.next) |n| {
            n.prev = order.prev;
        } else {
            self.tail = order.prev;
        }
        self.total_quantity -= order.remaining();
        self.order_count -= 1;
        order.prev = null;
        order.next = null;
        order.level = null;
    }

    pub fn isEmpty(self: *const PriceLevel) bool {
        return self.head == null;
    }

    pub fn queuePosition(self: *const PriceLevel, order_id: OrderId) ?struct { position: u32, quantity_ahead: Quantity } {
        var pos: u32 = 0;
        var qty_ahead: Quantity = 0;
        var cur = self.head;
        while (cur) |o| {
            if (o.id == order_id) {
                return .{ .position = pos, .quantity_ahead = qty_ahead };
            }
            qty_ahead += o.remaining();
            pos += 1;
            cur = o.next;
        }
        return null;
    }
};

// ============================================================================
// BookSide: sorted array of price levels
// ============================================================================

pub const BookSide = struct {
    side: Side,
    levels: std.ArrayList(PriceLevel),
    level_map: std.AutoHashMap(Price, usize), // price -> index in levels

    pub fn init(allocator: Allocator, side: Side) BookSide {
        return .{
            .side = side,
            .levels = std.ArrayList(PriceLevel).init(allocator),
            .level_map = std.AutoHashMap(Price, usize).init(allocator),
        };
    }

    pub fn deinit(self: *BookSide) void {
        self.levels.deinit();
        self.level_map.deinit();
    }

    pub fn bestLevel(self: *const BookSide) ?*const PriceLevel {
        if (self.levels.items.len == 0) return null;
        return &self.levels.items[0];
    }

    pub fn bestLevelMut(self: *BookSide) ?*PriceLevel {
        if (self.levels.items.len == 0) return null;
        return &self.levels.items[0];
    }

    pub fn bestPrice(self: *const BookSide) ?Price {
        const lvl = self.bestLevel() orelse return null;
        return lvl.price;
    }

    pub fn getOrCreateLevel(self: *BookSide, price: Price) !*PriceLevel {
        if (self.level_map.get(price)) |idx| {
            return &self.levels.items[idx];
        }
        // Find insertion point (bids descending, asks ascending)
        var insert_idx: usize = self.levels.items.len;
        for (self.levels.items, 0..) |lvl, i| {
            const should_insert = switch (self.side) {
                .bid => price > lvl.price,
                .ask => price < lvl.price,
            };
            if (should_insert) {
                insert_idx = i;
                break;
            }
        }

        try self.levels.insert(insert_idx, PriceLevel.init(price));

        // Rebuild index map (indices shifted)
        self.level_map.clearRetainingCapacity();
        for (self.levels.items, 0..) |lvl, i| {
            try self.level_map.put(lvl.price, i);
        }

        return &self.levels.items[insert_idx];
    }

    pub fn removeLevel(self: *BookSide, price: Price) void {
        const idx = self.level_map.get(price) orelse return;
        _ = self.levels.orderedRemove(idx);
        self.level_map.clearRetainingCapacity();
        for (self.levels.items, 0..) |lvl, i| {
            self.level_map.put(lvl.price, i) catch {};
        }
    }

    pub fn totalDepth(self: *const BookSide, n_levels: usize) Quantity {
        var total: Quantity = 0;
        const limit = @min(n_levels, self.levels.items.len);
        for (self.levels.items[0..limit]) |lvl| {
            total += lvl.total_quantity;
        }
        return total;
    }

    pub fn levelCount(self: *const BookSide) usize {
        return self.levels.items.len;
    }
};

// ============================================================================
// Execution result
// ============================================================================

pub const TradeEvent = struct {
    maker_order_id: OrderId,
    taker_order_id: OrderId,
    price: Price,
    quantity: Quantity,
    timestamp: Timestamp,
    side: Side, // aggressor side
};

pub const AddResult = struct {
    trades: std.ArrayList(TradeEvent),
    remaining_quantity: Quantity,
    order_placed: bool, // whether residual was placed in book

    pub fn deinit(self: *AddResult) void {
        self.trades.deinit();
    }
};

// ============================================================================
// L3Book: Full Level-3 Order Book
// ============================================================================

pub const L3Book = struct {
    bids: BookSide,
    asks: BookSide,
    orders: std.AutoHashMap(OrderId, *Order),
    order_pool: std.ArrayList(Order),
    allocator: Allocator,
    next_trade_id: u64,
    sequence: u64,

    pub fn init(allocator: Allocator) L3Book {
        return .{
            .bids = BookSide.init(allocator, .bid),
            .asks = BookSide.init(allocator, .ask),
            .orders = std.AutoHashMap(OrderId, *Order).init(allocator),
            .order_pool = std.ArrayList(Order).init(allocator),
            .allocator = allocator,
            .next_trade_id = 1,
            .sequence = 0,
        };
    }

    pub fn deinit(self: *L3Book) void {
        self.bids.deinit();
        self.asks.deinit();
        self.orders.deinit();
        self.order_pool.deinit();
    }

    fn allocOrder(self: *L3Book) !*Order {
        try self.order_pool.append(undefined);
        return &self.order_pool.items[self.order_pool.items.len - 1];
    }

    fn bookSideFor(self: *L3Book, side: Side) *BookSide {
        return switch (side) {
            .bid => &self.bids,
            .ask => &self.asks,
        };
    }

    fn oppositeSide(self: *L3Book, side: Side) *BookSide {
        return switch (side) {
            .bid => &self.asks,
            .ask => &self.bids,
        };
    }

    fn canMatch(side: Side, order_price: Price, book_price: Price) bool {
        return switch (side) {
            .bid => order_price >= book_price,
            .ask => order_price <= book_price,
        };
    }

    // ---- Core Operations ----

    pub fn addOrder(self: *L3Book, id: OrderId, side: Side, price: Price, quantity: Quantity, timestamp: Timestamp, order_type: OrderType, tif: TimeInForce) !AddResult {
        self.sequence += 1;
        var trades = std.ArrayList(TradeEvent).init(self.allocator);
        var remaining = quantity;

        // Match against opposite side
        if (order_type != .stop and order_type != .stop_limit) {
            const opp = self.oppositeSide(side);
            while (remaining > 0) {
                const best = opp.bestLevelMut() orelse break;
                if (order_type == .limit or order_type == .ioc or order_type == .fok) {
                    if (!canMatch(side, price, best.price)) break;
                }
                // Match orders at this level
                while (remaining > 0) {
                    const maker = best.head orelse break;
                    const fill_qty = @min(remaining, maker.remaining());
                    maker.filled += fill_qty;
                    remaining -= fill_qty;
                    best.total_quantity -= fill_qty;

                    try trades.append(.{
                        .maker_order_id = maker.id,
                        .taker_order_id = id,
                        .price = best.price,
                        .quantity = fill_qty,
                        .timestamp = timestamp,
                        .side = side,
                    });
                    self.next_trade_id += 1;

                    if (maker.isFilled()) {
                        best.removeOrder(maker);
                        _ = self.orders.remove(maker.id);
                    }
                }
                if (best.isEmpty()) {
                    opp.removeLevel(best.price);
                }
            }
        }

        // Handle IOC/FOK residual
        var order_placed = false;
        if (remaining > 0 and (order_type == .limit or order_type == .stop_limit)) {
            if (tif != .ioc and tif != .fok) {
                const order_ptr = try self.allocOrder();
                order_ptr.* = .{
                    .id = id,
                    .side = side,
                    .price = price,
                    .quantity = quantity,
                    .filled = quantity - remaining,
                    .timestamp = timestamp,
                    .order_type = order_type,
                    .tif = tif,
                    .prev = null,
                    .next = null,
                    .level = null,
                };
                const book_side = self.bookSideFor(side);
                const level = try book_side.getOrCreateLevel(price);
                level.appendOrder(order_ptr);
                try self.orders.put(id, order_ptr);
                order_placed = true;
            }
        }
        // FOK: cancel all if not fully filled
        if (tif == .fok and remaining > 0) {
            // Undo trades (simplified: in production would need proper rollback)
            remaining = quantity;
            trades.clearRetainingCapacity();
        }

        return .{
            .trades = trades,
            .remaining_quantity = remaining,
            .order_placed = order_placed,
        };
    }

    pub fn cancelOrder(self: *L3Book, order_id: OrderId) bool {
        const order_ptr = self.orders.get(order_id) orelse return false;
        if (order_ptr.level) |level| {
            level.removeOrder(order_ptr);
            const book_side = self.bookSideFor(order_ptr.side);
            if (level.isEmpty()) {
                book_side.removeLevel(level.price);
            }
        }
        _ = self.orders.remove(order_id);
        self.sequence += 1;
        return true;
    }

    pub fn modifyOrder(self: *L3Book, order_id: OrderId, new_price: Price, new_quantity: Quantity, timestamp: Timestamp) !bool {
        const order_ptr = self.orders.get(order_id) orelse return false;
        const side = order_ptr.side;
        const order_type = order_ptr.order_type;
        const tif = order_ptr.tif;
        const old_filled = order_ptr.filled;

        // Cancel and re-add (loses priority if price changed)
        _ = self.cancelOrder(order_id);

        if (new_quantity > old_filled) {
            var result = try self.addOrder(order_id, side, new_price, new_quantity, timestamp, order_type, tif);
            result.deinit();
        }
        return true;
    }

    pub fn executeOrder(self: *L3Book, order_id: OrderId, quantity: Quantity) ?TradeEvent {
        const order_ptr = self.orders.get(order_id) orelse return null;
        const fill_qty = @min(quantity, order_ptr.remaining());
        if (fill_qty == 0) return null;

        order_ptr.filled += fill_qty;
        if (order_ptr.level) |level| {
            level.total_quantity -= fill_qty;
        }

        const trade = TradeEvent{
            .maker_order_id = order_id,
            .taker_order_id = 0,
            .price = order_ptr.price,
            .quantity = fill_qty,
            .timestamp = 0,
            .side = order_ptr.side,
        };

        if (order_ptr.isFilled()) {
            if (order_ptr.level) |level| {
                level.removeOrder(order_ptr);
                const book_side = self.bookSideFor(order_ptr.side);
                if (level.isEmpty()) {
                    book_side.removeLevel(level.price);
                }
            }
            _ = self.orders.remove(order_id);
        }
        self.sequence += 1;
        return trade;
    }

    // ---- Query Operations ----

    pub fn bestBid(self: *const L3Book) ?Price {
        return self.bids.bestPrice();
    }

    pub fn bestAsk(self: *const L3Book) ?Price {
        return self.asks.bestPrice();
    }

    pub fn midPrice(self: *const L3Book) ?f64 {
        const bid = self.bestBid() orelse return null;
        const ask = self.bestAsk() orelse return null;
        return (fixedToPrice(bid) + fixedToPrice(ask)) / 2.0;
    }

    pub fn spread(self: *const L3Book) ?f64 {
        const bid = self.bestBid() orelse return null;
        const ask = self.bestAsk() orelse return null;
        return fixedToPrice(ask) - fixedToPrice(bid);
    }

    pub fn spreadBps(self: *const L3Book) ?f64 {
        const s = self.spread() orelse return null;
        const m = self.midPrice() orelse return null;
        if (m == 0) return null;
        return s / m * 10000.0;
    }

    // VWAP fill simulation: what average price to fill `qty` on given side
    pub fn vwapFill(self: *const L3Book, side: Side, qty: Quantity) ?f64 {
        const book_side: *const BookSide = switch (side) {
            .bid => &self.asks, // buying fills against asks
            .ask => &self.bids, // selling fills against bids
        };
        var remaining = qty;
        var cost: f64 = 0;
        var filled: Quantity = 0;

        for (book_side.levels.items) |lvl| {
            if (remaining == 0) break;
            const fill = @min(remaining, lvl.total_quantity);
            cost += fixedToPrice(lvl.price) * @as(f64, @floatFromInt(fill));
            filled += fill;
            remaining -= fill;
        }
        if (filled == 0) return null;
        return cost / @as(f64, @floatFromInt(filled));
    }

    // Market-by-price aggregation for top N levels
    pub const MBPLevel = struct {
        price: f64,
        quantity: Quantity,
        order_count: u32,
    };

    pub fn marketByPrice(self: *const L3Book, side: Side, max_levels: usize, out: []MBPLevel) usize {
        const book_side: *const BookSide = switch (side) {
            .bid => &self.bids,
            .ask => &self.asks,
        };
        const n = @min(max_levels, @min(out.len, book_side.levels.items.len));
        for (0..n) |i| {
            const lvl = book_side.levels.items[i];
            out[i] = .{
                .price = fixedToPrice(lvl.price),
                .quantity = lvl.total_quantity,
                .order_count = lvl.order_count,
            };
        }
        return n;
    }

    // Queue position for a specific order
    pub fn queuePosition(self: *const L3Book, order_id: OrderId) ?struct { position: u32, quantity_ahead: Quantity, level_price: f64 } {
        const order_ptr = self.orders.get(order_id) orelse return null;
        const level = order_ptr.level orelse return null;
        const pos = level.queuePosition(order_id) orelse return null;
        return .{
            .position = pos.position,
            .quantity_ahead = pos.quantity_ahead,
            .level_price = fixedToPrice(level.price),
        };
    }

    // Book pressure: weighted imbalance across levels
    pub fn bookPressure(self: *const L3Book, depth: usize) f64 {
        var bid_pressure: f64 = 0;
        var ask_pressure: f64 = 0;
        const mid = self.midPrice() orelse return 0;

        const bid_n = @min(depth, self.bids.levels.items.len);
        for (0..bid_n) |i| {
            const lvl = self.bids.levels.items[i];
            const dist = mid - fixedToPrice(lvl.price);
            const weight = if (dist > 0) 1.0 / (1.0 + dist * 100.0) else 1.0;
            bid_pressure += @as(f64, @floatFromInt(lvl.total_quantity)) * weight;
        }

        const ask_n = @min(depth, self.asks.levels.items.len);
        for (0..ask_n) |i| {
            const lvl = self.asks.levels.items[i];
            const dist = fixedToPrice(lvl.price) - mid;
            const weight = if (dist > 0) 1.0 / (1.0 + dist * 100.0) else 1.0;
            ask_pressure += @as(f64, @floatFromInt(lvl.total_quantity)) * weight;
        }

        const total = bid_pressure + ask_pressure;
        if (total == 0) return 0;
        return (bid_pressure - ask_pressure) / total; // -1 to +1
    }

    // Snapshot: full depth reconstruction
    pub const BookSnapshot = struct {
        bids: []MBPLevel,
        asks: []MBPLevel,
        n_bids: usize,
        n_asks: usize,
        mid: ?f64,
        spread: ?f64,
        sequence: u64,
        timestamp: Timestamp,
    };

    pub fn snapshot(self: *const L3Book, bid_buf: []MBPLevel, ask_buf: []MBPLevel, timestamp: Timestamp) BookSnapshot {
        const n_bids = self.marketByPrice(.bid, bid_buf.len, bid_buf);
        const n_asks = self.marketByPrice(.ask, ask_buf.len, ask_buf);
        return .{
            .bids = bid_buf[0..n_bids],
            .asks = ask_buf[0..n_asks],
            .n_bids = n_bids,
            .n_asks = n_asks,
            .mid = self.midPrice(),
            .spread = self.spread(),
            .sequence = self.sequence,
            .timestamp = timestamp,
        };
    }

    pub fn orderCount(self: *const L3Book) usize {
        return self.orders.count();
    }

    pub fn bidLevelCount(self: *const L3Book) usize {
        return self.bids.levelCount();
    }

    pub fn askLevelCount(self: *const L3Book) usize {
        return self.asks.levelCount();
    }

    pub fn totalBidQuantity(self: *const L3Book, depth: usize) Quantity {
        return self.bids.totalDepth(depth);
    }

    pub fn totalAskQuantity(self: *const L3Book, depth: usize) Quantity {
        return self.asks.totalDepth(depth);
    }

    // Volume-weighted bid/ask imbalance at top N levels
    pub fn imbalance(self: *const L3Book, depth: usize) f64 {
        const bid_qty = self.totalBidQuantity(depth);
        const ask_qty = self.totalAskQuantity(depth);
        const total = bid_qty + ask_qty;
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(bid_qty)) / @as(f64, @floatFromInt(total)) - 0.5;
    }
};

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "basic add and cancel" {
    var book = L3Book.init(testing.allocator);
    defer book.deinit();

    var result = try book.addOrder(1, .bid, priceToFixed(100.0), 100, 0, .limit, .gtc);
    defer result.deinit();

    try testing.expect(result.order_placed);
    try testing.expectEqual(@as(usize, 1), book.orderCount());
    try testing.expectEqual(priceToFixed(100.0), book.bestBid().?);

    try testing.expect(book.cancelOrder(1));
    try testing.expectEqual(@as(usize, 0), book.orderCount());
}

test "matching" {
    var book = L3Book.init(testing.allocator);
    defer book.deinit();

    var r1 = try book.addOrder(1, .bid, priceToFixed(100.0), 50, 0, .limit, .gtc);
    defer r1.deinit();
    var r2 = try book.addOrder(2, .ask, priceToFixed(100.0), 30, 1, .limit, .gtc);
    defer r2.deinit();

    try testing.expectEqual(@as(usize, 1), r2.trades.items.len);
    try testing.expectEqual(@as(Quantity, 30), r2.trades.items[0].quantity);
}

test "book pressure" {
    var book = L3Book.init(testing.allocator);
    defer book.deinit();

    var r1 = try book.addOrder(1, .bid, priceToFixed(99.0), 100, 0, .limit, .gtc);
    defer r1.deinit();
    var r2 = try book.addOrder(2, .ask, priceToFixed(101.0), 50, 0, .limit, .gtc);
    defer r2.deinit();

    const pressure = book.bookPressure(5);
    try testing.expect(pressure > 0); // more bid quantity
}

// ============================================================================
// OrderBookStatistics: real-time book analytics
// ============================================================================

pub const BookStatistics = struct {
    trade_count: u64,
    total_volume: Quantity,
    buy_volume: Quantity,
    sell_volume: Quantity,
    last_trade_price: Price,
    high_price: Price,
    low_price: Price,
    vwap_numerator: f64,
    vwap_denominator: f64,
    order_flow_imbalance: f64,
    cancel_count: u64,
    modify_count: u64,

    pub fn init() BookStatistics {
        return .{
            .trade_count = 0,
            .total_volume = 0,
            .buy_volume = 0,
            .sell_volume = 0,
            .last_trade_price = 0,
            .high_price = std.math.minInt(i64),
            .low_price = std.math.maxInt(i64),
            .vwap_numerator = 0,
            .vwap_denominator = 0,
            .order_flow_imbalance = 0,
            .cancel_count = 0,
            .modify_count = 0,
        };
    }

    pub fn recordTrade(self: *BookStatistics, trade: TradeEvent) void {
        self.trade_count += 1;
        self.total_volume += trade.quantity;
        if (trade.side == .bid) {
            self.buy_volume += trade.quantity;
        } else {
            self.sell_volume += trade.quantity;
        }
        self.last_trade_price = trade.price;
        if (trade.price > self.high_price) self.high_price = trade.price;
        if (trade.price < self.low_price) self.low_price = trade.price;

        const pv = fixedToPrice(trade.price) * @as(f64, @floatFromInt(trade.quantity));
        self.vwap_numerator += pv;
        self.vwap_denominator += @as(f64, @floatFromInt(trade.quantity));

        // Exponentially-weighted order flow imbalance
        const imb_sign: f64 = if (trade.side == .bid) 1.0 else -1.0;
        self.order_flow_imbalance = 0.95 * self.order_flow_imbalance + 0.05 * imb_sign * @as(f64, @floatFromInt(trade.quantity));
    }

    pub fn recordCancel(self: *BookStatistics) void {
        self.cancel_count += 1;
    }

    pub fn recordModify(self: *BookStatistics) void {
        self.modify_count += 1;
    }

    pub fn vwap(self: *const BookStatistics) f64 {
        if (self.vwap_denominator == 0) return 0;
        return self.vwap_numerator / self.vwap_denominator;
    }

    pub fn buyRatio(self: *const BookStatistics) f64 {
        if (self.total_volume == 0) return 0.5;
        return @as(f64, @floatFromInt(self.buy_volume)) / @as(f64, @floatFromInt(self.total_volume));
    }

    pub fn cancelToTradeRatio(self: *const BookStatistics) f64 {
        if (self.trade_count == 0) return 0;
        return @as(f64, @floatFromInt(self.cancel_count)) / @as(f64, @floatFromInt(self.trade_count));
    }
};

// ============================================================================
// OrderBookReplay: store and replay order book events
// ============================================================================

pub const EventType = enum(u8) {
    add,
    cancel,
    modify,
    trade,
};

pub const BookEvent = struct {
    event_type: EventType,
    order_id: OrderId,
    side: Side,
    price: Price,
    quantity: Quantity,
    timestamp: Timestamp,
    order_type: OrderType,
};

pub const EventLog = struct {
    events: std.ArrayList(BookEvent),

    pub fn init(allocator: Allocator) EventLog {
        return .{ .events = std.ArrayList(BookEvent).init(allocator) };
    }

    pub fn deinit(self: *EventLog) void {
        self.events.deinit();
    }

    pub fn logAdd(self: *EventLog, id: OrderId, side: Side, price: Price, qty: Quantity, ts: Timestamp, ot: OrderType) !void {
        try self.events.append(.{
            .event_type = .add,
            .order_id = id,
            .side = side,
            .price = price,
            .quantity = qty,
            .timestamp = ts,
            .order_type = ot,
        });
    }

    pub fn logCancel(self: *EventLog, id: OrderId, ts: Timestamp) !void {
        try self.events.append(.{
            .event_type = .cancel,
            .order_id = id,
            .side = .bid,
            .price = 0,
            .quantity = 0,
            .timestamp = ts,
            .order_type = .limit,
        });
    }

    pub fn logTrade(self: *EventLog, trade: TradeEvent) !void {
        try self.events.append(.{
            .event_type = .trade,
            .order_id = trade.maker_order_id,
            .side = trade.side,
            .price = trade.price,
            .quantity = trade.quantity,
            .timestamp = trade.timestamp,
            .order_type = .limit,
        });
    }

    pub fn replay(self: *const EventLog, book: *L3Book) !void {
        for (self.events.items) |evt| {
            switch (evt.event_type) {
                .add => {
                    var result = try book.addOrder(evt.order_id, evt.side, evt.price, evt.quantity, evt.timestamp, evt.order_type, .gtc);
                    result.deinit();
                },
                .cancel => {
                    _ = book.cancelOrder(evt.order_id);
                },
                .modify => {
                    _ = try book.modifyOrder(evt.order_id, evt.price, evt.quantity, evt.timestamp);
                },
                .trade => {
                    _ = book.executeOrder(evt.order_id, evt.quantity);
                },
            }
        }
    }

    pub fn eventCount(self: *const EventLog) usize {
        return self.events.items.len;
    }
};

// ============================================================================
// Microstructure metrics
// ============================================================================

pub const MicrostructureMetrics = struct {
    // Kyle's Lambda: price impact per unit of order flow
    kyle_lambda: f64,
    // Roll spread estimate
    roll_spread: f64,
    // Amihud illiquidity
    amihud: f64,

    trade_prices: [256]f64,
    trade_sizes: [256]f64,
    trade_signs: [256]f64,
    idx: u8,
    count: u64,

    pub fn init() MicrostructureMetrics {
        return .{
            .kyle_lambda = 0,
            .roll_spread = 0,
            .amihud = 0,
            .trade_prices = [_]f64{0} ** 256,
            .trade_sizes = [_]f64{0} ** 256,
            .trade_signs = [_]f64{0} ** 256,
            .idx = 0,
            .count = 0,
        };
    }

    pub fn recordTrade(self: *MicrostructureMetrics, price: f64, size: f64, is_buy: bool) void {
        self.trade_prices[self.idx] = price;
        self.trade_sizes[self.idx] = size;
        self.trade_signs[self.idx] = if (is_buy) 1.0 else -1.0;
        self.idx +%= 1;
        self.count += 1;

        if (self.count > 20) {
            self.computeMetrics();
        }
    }

    fn computeMetrics(self: *MicrostructureMetrics) void {
        const n: usize = @intCast(@min(self.count, 256));
        if (n < 10) return;

        // Kyle's lambda: regression of price change on signed volume
        var sum_sv: f64 = 0;
        var sum_dp: f64 = 0;
        var sum_sv2: f64 = 0;
        var sum_sv_dp: f64 = 0;
        var pairs: u32 = 0;

        for (1..n) |i| {
            const ci = self.idx -% @as(u8, @intCast(n)) +% @as(u8, @intCast(i));
            const pi = self.idx -% @as(u8, @intCast(n)) +% @as(u8, @intCast(i - 1));
            const dp = self.trade_prices[ci] - self.trade_prices[pi];
            const sv = self.trade_signs[ci] * self.trade_sizes[ci];
            sum_sv += sv;
            sum_dp += dp;
            sum_sv2 += sv * sv;
            sum_sv_dp += sv * dp;
            pairs += 1;
        }

        if (pairs > 0) {
            const fn_p = @as(f64, @floatFromInt(pairs));
            const denom = fn_p * sum_sv2 - sum_sv * sum_sv;
            if (@abs(denom) > 1e-15) {
                self.kyle_lambda = (fn_p * sum_sv_dp - sum_sv * sum_dp) / denom;
            }
        }

        // Roll spread estimate: -2 * sqrt(-cov(dp_t, dp_{t-1}))
        var sum_dp1: f64 = 0;
        var sum_dp2: f64 = 0;
        var sum_dp12: f64 = 0;
        var roll_pairs: u32 = 0;

        for (2..n) |i| {
            const ci = self.idx -% @as(u8, @intCast(n)) +% @as(u8, @intCast(i));
            const pi = self.idx -% @as(u8, @intCast(n)) +% @as(u8, @intCast(i - 1));
            const ppi = self.idx -% @as(u8, @intCast(n)) +% @as(u8, @intCast(i - 2));
            const dp1 = self.trade_prices[ci] - self.trade_prices[pi];
            const dp2 = self.trade_prices[pi] - self.trade_prices[ppi];
            sum_dp1 += dp1;
            sum_dp2 += dp2;
            sum_dp12 += dp1 * dp2;
            roll_pairs += 1;
        }

        if (roll_pairs > 0) {
            const fn_rp = @as(f64, @floatFromInt(roll_pairs));
            const cov = sum_dp12 / fn_rp - (sum_dp1 / fn_rp) * (sum_dp2 / fn_rp);
            if (cov < 0) {
                self.roll_spread = 2.0 * @sqrt(-cov);
            }
        }

        // Amihud: avg |return| / volume
        var amihud_sum: f64 = 0;
        var amihud_n: u32 = 0;
        for (1..n) |i| {
            const ci = self.idx -% @as(u8, @intCast(n)) +% @as(u8, @intCast(i));
            const pi = self.idx -% @as(u8, @intCast(n)) +% @as(u8, @intCast(i - 1));
            if (self.trade_prices[pi] > 0 and self.trade_sizes[ci] > 0) {
                const ret = @abs(self.trade_prices[ci] - self.trade_prices[pi]) / self.trade_prices[pi];
                amihud_sum += ret / self.trade_sizes[ci];
                amihud_n += 1;
            }
        }
        if (amihud_n > 0) {
            self.amihud = amihud_sum / @as(f64, @floatFromInt(amihud_n));
        }
    }
};
