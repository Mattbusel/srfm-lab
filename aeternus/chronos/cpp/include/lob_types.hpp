#pragma once
/// lob_types.hpp — Core types for the Chronos LOB simulator.
///
/// Fixed-point price, quantities, timestamps, and cache-aligned structures
/// optimised for high-frequency L3 order book simulation.

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <limits>
#include <string>
#include <array>

// ── Alignment macros ──────────────────────────────────────────────────────────

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#define CACHE_ALIGN alignas(CACHE_LINE_SIZE)

// ── Fundamental Types ─────────────────────────────────────────────────────────

namespace chronos {

/// Nanosecond timestamp since epoch.
using Nanos      = uint64_t;
/// Order identifier (monotonically increasing, exchange-wide).
using OrderId    = uint64_t;
/// Instrument identifier.
using InstId     = uint32_t;
/// Agent/participant identifier.
using AgentId    = uint32_t;
/// Sequence number (for time priority within a price level).
using SeqNum     = uint64_t;

/// Fixed-point price: integer ticks. Divide by PRICE_SCALE to get float.
using TickPrice  = int64_t;
/// Quantity in base units (floating point for fractional qty support).
using Qty        = double;

/// Price scale: 1 = integer prices, 100 = 2 decimal places, 100000 = 5.
constexpr TickPrice PRICE_SCALE = 100'000LL;

/// Convert float price to tick price.
inline TickPrice to_tick(double price) noexcept {
    return static_cast<TickPrice>(price * PRICE_SCALE + 0.5);
}

/// Convert tick price to float.
inline double from_tick(TickPrice tick) noexcept {
    return static_cast<double>(tick) / static_cast<double>(PRICE_SCALE);
}

/// Sentinel invalid price.
constexpr TickPrice INVALID_PRICE = std::numeric_limits<TickPrice>::min();
constexpr OrderId   INVALID_ORDER_ID = 0;

// ── Side ─────────────────────────────────────────────────────────────────────

enum class Side : uint8_t {
    Bid = 0,
    Ask = 1,
};

inline Side opposite(Side s) noexcept {
    return s == Side::Bid ? Side::Ask : Side::Bid;
}

inline const char* side_str(Side s) noexcept {
    return s == Side::Bid ? "BID" : "ASK";
}

// ── Order Type ────────────────────────────────────────────────────────────────

enum class OrderType : uint8_t {
    Limit       = 0,
    Market      = 1,
    StopLimit   = 2,
    StopMarket  = 3,
    Iceberg     = 4,
    Peg         = 5,   // Mid-peg order
};

// ── Time in Force ─────────────────────────────────────────────────────────────

enum class TimeInForce : uint8_t {
    GTC  = 0,    ///< Good-till-cancel.
    IOC  = 1,    ///< Immediate-or-cancel.
    FOK  = 2,    ///< Fill-or-kill.
    Day  = 3,    ///< Day order.
    GTD  = 4,    ///< Good-till-date.
};

// ── Order Status ──────────────────────────────────────────────────────────────

enum class OrderStatus : uint8_t {
    New         = 0,
    PartialFill = 1,
    Filled      = 2,
    Cancelled   = 3,
    Rejected    = 4,
    Triggered   = 5,   // Stop triggered, now active.
};

// ── Order ─────────────────────────────────────────────────────────────────────

/// Cache-line aligned order structure.
/// Layout designed for cache efficiency: hot fields first.
struct CACHE_ALIGN Order {
    // ── Hot path fields (first 32 bytes, likely in first cache line) ──
    OrderId     id;               //  8 bytes
    TickPrice   price;            //  8 bytes (limit price)
    Qty         leaves_qty;       //  8 bytes (remaining visible qty)
    SeqNum      seq;              //  8 bytes
    // ── Fields accessed slightly less frequently ──
    Qty         orig_qty;         //  8 bytes
    Qty         filled_qty;       //  8 bytes
    Qty         hidden_qty;       //  8 bytes (iceberg hidden reserve)
    Nanos       timestamp_ns;     //  8 bytes
    // ── Classification fields ──
    TickPrice   stop_price;       //  8 bytes (for stop orders)
    OrderId     orig_order_id;    //  8 bytes (for cancel-replace tracking)
    AgentId     agent_id;         //  4 bytes
    InstId      instrument_id;    //  4 bytes
    Side        side;             //  1 byte
    OrderType   type;             //  1 byte
    TimeInForce tif;              //  1 byte
    OrderStatus status;           //  1 byte
    uint8_t     flags;            //  1 byte (custom flags)
    uint8_t     pad[3];           //  3 bytes padding

    // Total: ~88 bytes, fits in 2 cache lines.

    Order() noexcept {
        std::memset(this, 0, sizeof(Order));
        status = OrderStatus::New;
    }

    /// Is the order fully executed or cancelled?
    bool is_terminal() const noexcept {
        return status == OrderStatus::Filled
            || status == OrderStatus::Cancelled
            || status == OrderStatus::Rejected;
    }

    /// Is there remaining quantity to fill?
    bool has_leaves() const noexcept {
        return leaves_qty > 1e-9 || hidden_qty > 1e-9;
    }

    /// Apply a fill of given qty. Handles iceberg refill.
    /// Returns actual fill qty (capped at leaves_qty).
    Qty apply_fill(Qty fill_qty) noexcept {
        double actual = fill_qty < leaves_qty ? fill_qty : leaves_qty;
        leaves_qty -= actual;
        filled_qty += actual;

        // Iceberg refill from hidden reserve.
        if (leaves_qty < 1e-9 && hidden_qty > 1e-9) {
            if (type == OrderType::Iceberg) {
                // Refill to orig peak (stored in flags-encoded peak_qty not here;
                // simplified: refill by hidden up to orig_qty fraction).
                double peak = orig_qty * 0.1; // 10% visible (configurable)
                double refill = peak < hidden_qty ? peak : hidden_qty;
                leaves_qty = refill;
                hidden_qty -= refill;
            }
        }

        if (leaves_qty < 1e-9 && hidden_qty < 1e-9) {
            status = OrderStatus::Filled;
        } else {
            status = OrderStatus::PartialFill;
        }
        return actual;
    }

    double price_f64() const noexcept { return from_tick(price); }
};

static_assert(sizeof(Order) <= 2 * CACHE_LINE_SIZE, "Order should fit in 2 cache lines");

// ── Fill ─────────────────────────────────────────────────────────────────────

/// A fill record generated by order matching.
struct Fill {
    Nanos       timestamp_ns;
    OrderId     aggressor_id;
    OrderId     passive_id;
    TickPrice   price;
    Qty         qty;
    Side        side;           ///< Aggressor side.
    AgentId     aggressor_agent;
    AgentId     passive_agent;
    InstId      instrument_id;
    bool        is_partial;     ///< True if passive order has remaining qty.

    Fill() noexcept { std::memset(this, 0, sizeof(Fill)); }

    double price_f64() const noexcept { return from_tick(price); }
};

// ── Level ─────────────────────────────────────────────────────────────────────

/// Level-2 depth entry (price level summary, no individual orders).
struct Level {
    TickPrice   price;
    Qty         total_qty;
    uint32_t    order_count;
    uint32_t    pad;

    Level() noexcept : price(0), total_qty(0.0), order_count(0), pad(0) {}
    Level(TickPrice p, Qty q, uint32_t n) noexcept
        : price(p), total_qty(q), order_count(n), pad(0) {}

    double price_f64() const noexcept { return from_tick(price); }
    bool empty() const noexcept { return total_qty < 1e-9; }
};

// ── Market Data Snapshot ──────────────────────────────────────────────────────

constexpr size_t DEPTH_LEVELS = 10;

/// Level-2 market data snapshot.
struct MarketSnapshot {
    Nanos       timestamp_ns;
    InstId      instrument_id;
    TickPrice   best_bid;
    TickPrice   best_ask;
    TickPrice   last_price;
    Qty         last_qty;
    Qty         session_volume;
    uint32_t    bid_levels;     ///< Number of valid bid depth entries.
    uint32_t    ask_levels;     ///< Number of valid ask depth entries.
    Level       bids[DEPTH_LEVELS];
    Level       asks[DEPTH_LEVELS];
    double      imbalance;      ///< (bid_vol - ask_vol) / (bid_vol + ask_vol)

    MarketSnapshot() noexcept {
        std::memset(this, 0, sizeof(MarketSnapshot));
        best_bid = INVALID_PRICE;
        best_ask = INVALID_PRICE;
    }

    double mid_price() const noexcept {
        if (best_bid == INVALID_PRICE || best_ask == INVALID_PRICE) return 0.0;
        return (from_tick(best_bid) + from_tick(best_ask)) / 2.0;
    }

    double spread() const noexcept {
        if (best_bid == INVALID_PRICE || best_ask == INVALID_PRICE) return 0.0;
        return from_tick(best_ask - best_bid);
    }
};

// ── Order Flow Event ──────────────────────────────────────────────────────────

enum class EventType : uint8_t {
    OrderAdd     = 0,
    OrderCancel  = 1,
    OrderModify  = 2,
    Fill         = 3,
    Snapshot     = 4,
    Tick         = 5,
    Halt         = 6,
    Resume       = 7,
};

struct CACHE_ALIGN LobEvent {
    Nanos       timestamp_ns;
    EventType   type;
    uint8_t     pad[7];
    union {
        Order   order;     ///< For OrderAdd/Modify.
        struct {
            OrderId id;
            InstId  instrument_id;
        } cancel;          ///< For OrderCancel.
        Fill    fill;      ///< For Fill.
        uint8_t raw[128];  ///< Raw bytes (largest union member).
    };

    LobEvent() noexcept { std::memset(this, 0, sizeof(LobEvent)); }
};

// ── Agent Signal ──────────────────────────────────────────────────────────────

/// Signal from an agent to the exchange: what order to place.
struct AgentAction {
    enum class ActionType : uint8_t {
        NoOp     = 0,
        Submit   = 1,
        Cancel   = 2,
        Modify   = 3,
    };

    ActionType  action;
    Order       order;           ///< For Submit/Modify.
    OrderId     cancel_id;       ///< For Cancel.

    static AgentAction no_op() noexcept {
        AgentAction a;
        a.action = ActionType::NoOp;
        return a;
    }

    static AgentAction submit(const Order& o) noexcept {
        AgentAction a;
        a.action = ActionType::Submit;
        a.order = o;
        return a;
    }

    static AgentAction cancel(OrderId id) noexcept {
        AgentAction a;
        a.action = ActionType::Cancel;
        a.cancel_id = id;
        return a;
    }
};

} // namespace chronos
