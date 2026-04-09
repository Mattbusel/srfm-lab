#ifndef SRFM_EXCHANGE_MATCHING_ENGINE_HPP
#define SRFM_EXCHANGE_MATCHING_ENGINE_HPP

// ============================================================================
// matching_engine.hpp
// Order Matching Engine for synthetic exchange simulation
// Part of srfm-lab / signal-engine
// ============================================================================

#include <cstdint>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>
#include <memory>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <iostream>
#include <utility>

namespace srfm::exchange {

// ============================================================================
// Forward declarations
// ============================================================================
class OrderBook;
class MatchingEngine;

// ============================================================================
// Type aliases
// ============================================================================
using Timestamp   = std::chrono::steady_clock::time_point;
using Duration    = std::chrono::steady_clock::duration;
using OrderId     = uint64_t;
using AgentId     = uint64_t;
using FillId      = uint64_t;
using Price       = double;
using Quantity    = double;
using Symbol      = std::string;

// ============================================================================
// Constants
// ============================================================================
namespace constants {
    constexpr double    PRICE_EPSILON           = 1e-9;
    constexpr double    QTY_EPSILON             = 1e-12;
    constexpr size_t    DEFAULT_TRADE_LOG_SIZE  = 100000;
    constexpr size_t    MAX_BATCH_SIZE          = 10000;
    constexpr double    DEFAULT_CIRCUIT_BREAKER_PCT = 0.10;  // 10%
    constexpr int64_t   DEFAULT_CIRCUIT_BREAKER_WINDOW_SEC = 60;
    constexpr uint32_t  DEFAULT_RATE_LIMIT_PER_SEC = 1000;
    constexpr double    DEFAULT_MIN_LOT_SIZE    = 0.001;
    constexpr double    DEFAULT_TICK_SIZE       = 0.01;
    constexpr int       DEFAULT_PRICE_BAND_PCT  = 20;       // 20% from reference
    constexpr Price     NO_PRICE                = std::numeric_limits<double>::quiet_NaN();
    constexpr Quantity  NO_QTY                  = 0.0;
}

// ============================================================================
// Enumerations
// ============================================================================

enum class Side : uint8_t {
    BUY  = 0,
    SELL = 1
};

[[nodiscard]] inline const char* to_string(Side s) noexcept {
    return s == Side::BUY ? "BUY" : "SELL";
}

[[nodiscard]] inline Side opposite_side(Side s) noexcept {
    return s == Side::BUY ? Side::SELL : Side::BUY;
}

// ---------------------------------------------------------------------------
enum class OrderType : uint8_t {
    MARKET = 0,   // execute immediately at best available price
    LIMIT  = 1,   // execute at specified price or better
    IOC    = 2,   // immediate-or-cancel: fill what you can, cancel rest
    FOK    = 3    // fill-or-kill: fill entirely or not at all
};

[[nodiscard]] inline const char* to_string(OrderType t) noexcept {
    switch (t) {
        case OrderType::MARKET: return "MARKET";
        case OrderType::LIMIT:  return "LIMIT";
        case OrderType::IOC:    return "IOC";
        case OrderType::FOK:    return "FOK";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
enum class OrderStatus : uint8_t {
    NEW       = 0,
    PARTIAL   = 1,
    FILLED    = 2,
    CANCELLED = 3,
    REJECTED  = 4
};

[[nodiscard]] inline const char* to_string(OrderStatus s) noexcept {
    switch (s) {
        case OrderStatus::NEW:       return "NEW";
        case OrderStatus::PARTIAL:   return "PARTIAL";
        case OrderStatus::FILLED:    return "FILLED";
        case OrderStatus::CANCELLED: return "CANCELLED";
        case OrderStatus::REJECTED:  return "REJECTED";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
enum class RejectReason : uint8_t {
    NONE                    = 0,
    INVALID_PRICE           = 1,
    INVALID_QUANTITY        = 2,
    PRICE_BAND_VIOLATION    = 3,
    CIRCUIT_BREAKER_ACTIVE  = 4,
    RATE_LIMIT_EXCEEDED     = 5,
    UNKNOWN_SYMBOL          = 6,
    SELF_TRADE_PREVENTION   = 7,
    LOT_SIZE_VIOLATION      = 8,
    TICK_SIZE_VIOLATION      = 9,
    FOK_CANNOT_FILL         = 10,
    DUPLICATE_ORDER_ID      = 11,
    INVALID_SIDE            = 12
};

[[nodiscard]] inline const char* to_string(RejectReason r) noexcept {
    switch (r) {
        case RejectReason::NONE:                   return "NONE";
        case RejectReason::INVALID_PRICE:          return "INVALID_PRICE";
        case RejectReason::INVALID_QUANTITY:        return "INVALID_QUANTITY";
        case RejectReason::PRICE_BAND_VIOLATION:   return "PRICE_BAND_VIOLATION";
        case RejectReason::CIRCUIT_BREAKER_ACTIVE: return "CIRCUIT_BREAKER_ACTIVE";
        case RejectReason::RATE_LIMIT_EXCEEDED:    return "RATE_LIMIT_EXCEEDED";
        case RejectReason::UNKNOWN_SYMBOL:         return "UNKNOWN_SYMBOL";
        case RejectReason::SELF_TRADE_PREVENTION:  return "SELF_TRADE_PREVENTION";
        case RejectReason::LOT_SIZE_VIOLATION:     return "LOT_SIZE_VIOLATION";
        case RejectReason::TICK_SIZE_VIOLATION:     return "TICK_SIZE_VIOLATION";
        case RejectReason::FOK_CANNOT_FILL:        return "FOK_CANNOT_FILL";
        case RejectReason::DUPLICATE_ORDER_ID:     return "DUPLICATE_ORDER_ID";
        case RejectReason::INVALID_SIDE:           return "INVALID_SIDE";
    }
    return "UNKNOWN";
}

// ============================================================================
// Order
// ============================================================================

struct Order {
    OrderId     order_id        = 0;
    AgentId     agent_id        = 0;
    Symbol      symbol;
    Side        side            = Side::BUY;
    OrderType   type            = OrderType::LIMIT;
    Price       price           = 0.0;
    Quantity    quantity         = 0.0;
    Quantity    remaining_qty   = 0.0;
    Timestamp   timestamp       = {};
    OrderStatus status          = OrderStatus::NEW;
    RejectReason reject_reason  = RejectReason::NONE;

    // Convenience
    [[nodiscard]] bool is_buy()  const noexcept { return side == Side::BUY; }
    [[nodiscard]] bool is_sell() const noexcept { return side == Side::SELL; }
    [[nodiscard]] bool is_active() const noexcept {
        return status == OrderStatus::NEW || status == OrderStatus::PARTIAL;
    }
    [[nodiscard]] Quantity filled_qty() const noexcept {
        return quantity - remaining_qty;
    }
    [[nodiscard]] double fill_ratio() const noexcept {
        return quantity > constants::QTY_EPSILON ? filled_qty() / quantity : 0.0;
    }
    [[nodiscard]] bool is_fully_filled() const noexcept {
        return remaining_qty < constants::QTY_EPSILON;
    }

    // Builder pattern helpers
    Order& with_id(OrderId id)          { order_id = id; return *this; }
    Order& with_agent(AgentId id)       { agent_id = id; return *this; }
    Order& with_symbol(Symbol s)        { symbol = std::move(s); return *this; }
    Order& with_side(Side s)            { side = s; return *this; }
    Order& with_type(OrderType t)       { type = t; return *this; }
    Order& with_price(Price p)          { price = p; return *this; }
    Order& with_quantity(Quantity q)     { quantity = q; remaining_qty = q; return *this; }
    Order& with_timestamp(Timestamp t)  { timestamp = t; return *this; }

    // Comparison for price-time priority
    [[nodiscard]] bool has_better_price(const Order& other, Side s) const noexcept {
        if (s == Side::BUY) return price > other.price + constants::PRICE_EPSILON;
        return price < other.price - constants::PRICE_EPSILON;
    }

    [[nodiscard]] bool has_same_price(const Order& other) const noexcept {
        return std::abs(price - other.price) < constants::PRICE_EPSILON;
    }

    [[nodiscard]] std::string to_string() const;
};

// ============================================================================
// Fill (execution report)
// ============================================================================

struct Fill {
    FillId      fill_id         = 0;
    OrderId     order_id        = 0;     // the order that triggered this fill
    OrderId     maker_order_id  = 0;
    OrderId     taker_order_id  = 0;
    AgentId     maker_agent_id  = 0;
    AgentId     taker_agent_id  = 0;
    Symbol      symbol;
    Price       price           = 0.0;
    Quantity    quantity         = 0.0;
    Timestamp   timestamp       = {};
    Side        aggressor_side  = Side::BUY;
    bool        is_maker        = false;

    [[nodiscard]] double notional() const noexcept { return price * quantity; }
    [[nodiscard]] std::string to_string() const;
};

// ============================================================================
// MarketDataSnapshot
// ============================================================================

struct MarketDataSnapshot {
    Symbol      symbol;
    Price       best_bid        = constants::NO_PRICE;
    Price       best_ask        = constants::NO_PRICE;
    Quantity    best_bid_qty    = 0.0;
    Quantity    best_ask_qty    = 0.0;
    Price       last_price      = constants::NO_PRICE;
    Quantity    last_qty        = 0.0;
    Price       open            = constants::NO_PRICE;
    Price       high            = constants::NO_PRICE;
    Price       low             = constants::NO_PRICE;
    Price       close           = constants::NO_PRICE;
    Quantity    volume          = 0.0;
    double      turnover        = 0.0;   // sum of notional
    uint64_t    n_trades        = 0;
    Timestamp   timestamp       = {};

    [[nodiscard]] Price spread() const noexcept {
        if (std::isnan(best_bid) || std::isnan(best_ask)) return constants::NO_PRICE;
        return best_ask - best_bid;
    }
    [[nodiscard]] Price mid_price() const noexcept {
        if (std::isnan(best_bid) || std::isnan(best_ask)) return constants::NO_PRICE;
        return (best_bid + best_ask) / 2.0;
    }
    [[nodiscard]] double spread_bps() const noexcept {
        Price mid = mid_price();
        if (std::isnan(mid) || mid < constants::PRICE_EPSILON) return constants::NO_PRICE;
        return (spread() / mid) * 10000.0;
    }
    [[nodiscard]] std::string to_string() const;
};

// ============================================================================
// PerformanceCounters
// ============================================================================

struct PerformanceCounters {
    std::atomic<uint64_t> orders_submitted{0};
    std::atomic<uint64_t> orders_processed{0};
    std::atomic<uint64_t> orders_rejected{0};
    std::atomic<uint64_t> fills_generated{0};
    std::atomic<uint64_t> cancellations{0};
    std::atomic<uint64_t> self_trade_preventions{0};
    std::atomic<uint64_t> circuit_breaker_trips{0};
    std::atomic<uint64_t> rate_limit_rejections{0};

    // Latency tracking (nanoseconds)
    std::atomic<int64_t>  total_match_latency_ns{0};
    std::atomic<int64_t>  max_match_latency_ns{0};
    std::atomic<uint64_t> match_count{0};

    [[nodiscard]] double avg_match_latency_ns() const noexcept {
        uint64_t mc = match_count.load(std::memory_order_relaxed);
        if (mc == 0) return 0.0;
        return static_cast<double>(total_match_latency_ns.load(std::memory_order_relaxed))
               / static_cast<double>(mc);
    }

    void record_match_latency(int64_t ns) noexcept {
        total_match_latency_ns.fetch_add(ns, std::memory_order_relaxed);
        match_count.fetch_add(1, std::memory_order_relaxed);
        int64_t current_max = max_match_latency_ns.load(std::memory_order_relaxed);
        while (ns > current_max) {
            if (max_match_latency_ns.compare_exchange_weak(
                    current_max, ns, std::memory_order_relaxed)) break;
        }
    }

    void reset() noexcept {
        orders_submitted.store(0, std::memory_order_relaxed);
        orders_processed.store(0, std::memory_order_relaxed);
        orders_rejected.store(0, std::memory_order_relaxed);
        fills_generated.store(0, std::memory_order_relaxed);
        cancellations.store(0, std::memory_order_relaxed);
        self_trade_preventions.store(0, std::memory_order_relaxed);
        circuit_breaker_trips.store(0, std::memory_order_relaxed);
        rate_limit_rejections.store(0, std::memory_order_relaxed);
        total_match_latency_ns.store(0, std::memory_order_relaxed);
        max_match_latency_ns.store(0, std::memory_order_relaxed);
        match_count.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] std::string to_string() const;
};

// ============================================================================
// TradeRecord (for TradeLog)
// ============================================================================

struct TradeRecord {
    FillId      fill_id         = 0;
    Symbol      symbol;
    Price       price           = 0.0;
    Quantity    quantity         = 0.0;
    Side        aggressor_side  = Side::BUY;
    AgentId     maker_agent     = 0;
    AgentId     taker_agent     = 0;
    Timestamp   timestamp       = {};

    [[nodiscard]] double notional() const noexcept { return price * quantity; }
};

// ============================================================================
// TradeLog - circular buffer of recent trades
// ============================================================================

class TradeLog {
public:
    explicit TradeLog(size_t capacity = constants::DEFAULT_TRADE_LOG_SIZE);

    // Non-copyable, movable
    TradeLog(const TradeLog&) = delete;
    TradeLog& operator=(const TradeLog&) = delete;
    TradeLog(TradeLog&& other) noexcept;
    TradeLog& operator=(TradeLog&& other) noexcept;
    ~TradeLog() = default;

    void push(TradeRecord record);
    void push(const Fill& fill);

    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] size_t capacity() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    // Access recent trades (0 = most recent)
    [[nodiscard]] const TradeRecord& at(size_t index) const;
    [[nodiscard]] const TradeRecord& operator[](size_t index) const;

    // Get last N trades
    [[nodiscard]] std::vector<TradeRecord> last_n(size_t n) const;

    // Get trades within a time window
    [[nodiscard]] std::vector<TradeRecord> since(Timestamp t) const;

    // Aggregations for TCA
    [[nodiscard]] double vwap(size_t n_trades) const;
    [[nodiscard]] double vwap_since(Timestamp t) const;
    [[nodiscard]] double total_volume(size_t n_trades) const;
    [[nodiscard]] double total_turnover(size_t n_trades) const;
    [[nodiscard]] Price  last_price() const;
    [[nodiscard]] Quantity last_quantity() const;

    void clear() noexcept;

private:
    std::vector<TradeRecord> buffer_;
    size_t                   capacity_;
    size_t                   head_     = 0;   // next write position
    size_t                   count_    = 0;

    [[nodiscard]] size_t physical_index(size_t logical) const noexcept;
};

// ============================================================================
// PriceLevel - all orders at one price (FIFO)
// ============================================================================

class PriceLevel {
public:
    explicit PriceLevel(Price price = 0.0);

    PriceLevel(const PriceLevel&) = default;
    PriceLevel& operator=(const PriceLevel&) = default;
    PriceLevel(PriceLevel&&) noexcept = default;
    PriceLevel& operator=(PriceLevel&&) noexcept = default;
    ~PriceLevel() = default;

    void add_order(Order order);
    bool remove_order(OrderId order_id);
    [[nodiscard]] Order* front_order() noexcept;
    [[nodiscard]] const Order* front_order() const noexcept;
    void pop_front();

    [[nodiscard]] Price     price() const noexcept { return price_; }
    [[nodiscard]] size_t    order_count() const noexcept { return orders_.size(); }
    [[nodiscard]] bool      empty() const noexcept { return orders_.empty(); }
    [[nodiscard]] Quantity  total_quantity() const noexcept;

    // Iterators for inspection
    [[nodiscard]] auto begin() noexcept        { return orders_.begin(); }
    [[nodiscard]] auto end() noexcept          { return orders_.end(); }
    [[nodiscard]] auto begin() const noexcept  { return orders_.begin(); }
    [[nodiscard]] auto end() const noexcept    { return orders_.end(); }
    [[nodiscard]] auto cbegin() const noexcept { return orders_.cbegin(); }
    [[nodiscard]] auto cend() const noexcept   { return orders_.cend(); }

    [[nodiscard]] const std::deque<Order>& orders() const noexcept { return orders_; }
    [[nodiscard]] std::deque<Order>& orders() noexcept { return orders_; }

private:
    Price              price_;
    std::deque<Order>  orders_;
};

// ============================================================================
// DepthEntry - one level of aggregated book depth
// ============================================================================

struct DepthEntry {
    Price       price       = 0.0;
    Quantity    quantity    = 0.0;
    size_t      order_count = 0;
};

// ============================================================================
// BookSnapshot - full order book state
// ============================================================================

struct BookSnapshot {
    Symbol                   symbol;
    std::vector<DepthEntry>  bids;
    std::vector<DepthEntry>  asks;
    Timestamp                timestamp = {};

    [[nodiscard]] Price best_bid() const noexcept {
        return bids.empty() ? constants::NO_PRICE : bids.front().price;
    }
    [[nodiscard]] Price best_ask() const noexcept {
        return asks.empty() ? constants::NO_PRICE : asks.front().price;
    }
    [[nodiscard]] Price mid_price() const noexcept {
        Price b = best_bid(), a = best_ask();
        if (std::isnan(b) || std::isnan(a)) return constants::NO_PRICE;
        return (b + a) / 2.0;
    }
    [[nodiscard]] Price spread() const noexcept {
        Price b = best_bid(), a = best_ask();
        if (std::isnan(b) || std::isnan(a)) return constants::NO_PRICE;
        return a - b;
    }
};

// ============================================================================
// SymbolConfig - per-symbol trading parameters
// ============================================================================

struct SymbolConfig {
    Symbol  symbol;
    double  tick_size               = constants::DEFAULT_TICK_SIZE;
    double  min_lot_size            = constants::DEFAULT_MIN_LOT_SIZE;
    double  max_order_qty           = 1e9;
    double  price_band_pct          = constants::DEFAULT_PRICE_BAND_PCT;
    double  circuit_breaker_pct     = constants::DEFAULT_CIRCUIT_BREAKER_PCT;
    int64_t circuit_breaker_window_sec = constants::DEFAULT_CIRCUIT_BREAKER_WINDOW_SEC;
    Price   reference_price         = constants::NO_PRICE;   // for price bands
    bool    trading_enabled         = true;
};

// ============================================================================
// CircuitBreaker
// ============================================================================

class CircuitBreaker {
public:
    CircuitBreaker() = default;
    explicit CircuitBreaker(double threshold_pct, int64_t window_seconds);

    void record_trade(Price price, Timestamp ts);
    [[nodiscard]] bool is_tripped() const noexcept;
    void reset() noexcept;
    void set_threshold(double pct) noexcept;
    void set_window(int64_t seconds) noexcept;

    [[nodiscard]] Price   anchor_price() const noexcept { return anchor_price_; }
    [[nodiscard]] double  threshold_pct() const noexcept { return threshold_pct_; }
    [[nodiscard]] int64_t window_seconds() const noexcept { return window_seconds_; }

private:
    double  threshold_pct_  = constants::DEFAULT_CIRCUIT_BREAKER_PCT;
    int64_t window_seconds_ = constants::DEFAULT_CIRCUIT_BREAKER_WINDOW_SEC;
    Price   anchor_price_   = constants::NO_PRICE;
    Timestamp anchor_time_  = {};
    bool    tripped_        = false;
};

// ============================================================================
// RateLimiter
// ============================================================================

class RateLimiter {
public:
    explicit RateLimiter(uint32_t max_per_second = constants::DEFAULT_RATE_LIMIT_PER_SEC);

    [[nodiscard]] bool check_and_consume(AgentId agent, Timestamp now);
    void set_limit(uint32_t max_per_second) noexcept;
    void reset() noexcept;
    [[nodiscard]] uint32_t limit() const noexcept { return max_per_second_; }

private:
    struct AgentBucket {
        Timestamp window_start  = {};
        uint32_t  count         = 0;
    };

    uint32_t max_per_second_;
    std::unordered_map<AgentId, AgentBucket> buckets_;
};

// ============================================================================
// OrderBook
// ============================================================================

class OrderBook {
public:
    explicit OrderBook(SymbolConfig config);

    // Non-copyable, movable
    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;
    OrderBook(OrderBook&&) noexcept = default;
    OrderBook& operator=(OrderBook&&) noexcept = default;
    ~OrderBook() = default;

    // ---- Core operations ----
    std::vector<Fill> add_order(Order order);
    bool cancel_order(OrderId order_id);
    std::vector<Fill> match();

    // ---- Market data queries ----
    [[nodiscard]] Price     best_bid() const noexcept;
    [[nodiscard]] Price     best_ask() const noexcept;
    [[nodiscard]] Price     mid_price() const noexcept;
    [[nodiscard]] Price     spread() const noexcept;
    [[nodiscard]] Quantity  best_bid_qty() const noexcept;
    [[nodiscard]] Quantity  best_ask_qty() const noexcept;

    [[nodiscard]] std::vector<DepthEntry> depth(int n_levels, Side side) const;
    [[nodiscard]] double vwap_to_fill(Quantity qty, Side side) const;

    [[nodiscard]] BookSnapshot       get_snapshot() const;
    [[nodiscard]] MarketDataSnapshot get_market_data() const;

    // ---- Order lookup ----
    [[nodiscard]] const Order* find_order(OrderId id) const;
    [[nodiscard]] Order*       find_order(OrderId id);
    [[nodiscard]] bool         has_order(OrderId id) const;
    [[nodiscard]] size_t       total_orders() const noexcept;
    [[nodiscard]] size_t       bid_levels() const noexcept;
    [[nodiscard]] size_t       ask_levels() const noexcept;

    // ---- Configuration ----
    [[nodiscard]] const SymbolConfig& config() const noexcept { return config_; }
    [[nodiscard]] const Symbol& symbol() const noexcept { return config_.symbol; }
    void set_reference_price(Price p) noexcept { config_.reference_price = p; }
    void set_trading_enabled(bool enabled) noexcept { config_.trading_enabled = enabled; }

    // ---- Trade log access ----
    [[nodiscard]] const TradeLog& trade_log() const noexcept { return trade_log_; }
    [[nodiscard]] TradeLog& trade_log() noexcept { return trade_log_; }

    // ---- Session management ----
    [[nodiscard]] Price   last_price() const noexcept { return last_price_; }
    [[nodiscard]] Quantity session_volume() const noexcept { return session_volume_; }
    [[nodiscard]] double  session_turnover() const noexcept { return session_turnover_; }
    [[nodiscard]] uint64_t session_trade_count() const noexcept { return session_trade_count_; }
    [[nodiscard]] Price   session_high() const noexcept { return session_high_; }
    [[nodiscard]] Price   session_low() const noexcept { return session_low_; }
    [[nodiscard]] Price   session_open() const noexcept { return session_open_; }

    void reset_session() noexcept;

    // ---- Circuit breaker ----
    [[nodiscard]] const CircuitBreaker& circuit_breaker() const noexcept { return circuit_breaker_; }
    [[nodiscard]] bool is_halted() const noexcept;

private:
    // ---- Internal matching ----
    std::vector<Fill> match_order(Order& incoming);
    std::vector<Fill> try_match_limit(Order& incoming);
    std::vector<Fill> try_match_market(Order& incoming);
    bool can_fill_fok(const Order& incoming) const;

    Fill generate_fill(const Order& maker, const Order& taker,
                       Quantity qty, Price price);

    // ---- Validation ----
    [[nodiscard]] RejectReason validate_order(const Order& order) const;
    [[nodiscard]] bool is_valid_tick(Price price) const noexcept;
    [[nodiscard]] bool is_valid_lot(Quantity qty) const noexcept;
    [[nodiscard]] bool is_within_price_band(Price price) const noexcept;

    // ---- Helpers ----
    void insert_into_book(Order order);
    void remove_empty_levels();
    void update_session_stats(Price trade_price, Quantity trade_qty);

    // ---- Data ----
    SymbolConfig config_;

    // Bids: descending by price (best bid = begin)
    std::map<Price, PriceLevel, std::greater<Price>> bids_;
    // Asks: ascending by price (best ask = begin)
    std::map<Price, PriceLevel>                      asks_;

    // O(1) order lookup
    std::unordered_map<OrderId, Price>  order_price_index_;
    std::unordered_map<OrderId, Side>   order_side_index_;

    // Trade log
    TradeLog trade_log_;

    // Session stats
    Price    last_price_         = constants::NO_PRICE;
    Quantity last_qty_           = 0.0;
    Quantity session_volume_     = 0.0;
    double   session_turnover_   = 0.0;
    uint64_t session_trade_count_= 0;
    Price    session_high_       = constants::NO_PRICE;
    Price    session_low_        = constants::NO_PRICE;
    Price    session_open_       = constants::NO_PRICE;

    // Circuit breaker
    CircuitBreaker circuit_breaker_;

    // Fill ID generator (per book)
    FillId next_fill_id_ = 1;
};

// ============================================================================
// SubmitResult
// ============================================================================

struct SubmitResult {
    Order               order;
    std::vector<Fill>   fills;
    RejectReason        reject_reason = RejectReason::NONE;
    bool                accepted      = false;

    [[nodiscard]] bool rejected() const noexcept { return !accepted; }
    [[nodiscard]] bool has_fills() const noexcept { return !fills.empty(); }
    [[nodiscard]] Quantity filled_qty() const noexcept {
        Quantity total = 0.0;
        for (const auto& f : fills) total += f.quantity;
        return total;
    }
};

// ============================================================================
// BatchResult
// ============================================================================

struct BatchResult {
    std::vector<SubmitResult>  results;
    uint64_t                   accepted     = 0;
    uint64_t                   rejected     = 0;
    uint64_t                   total_fills  = 0;
    Duration                   elapsed      = {};
};

// ============================================================================
// MatchingEngine
// ============================================================================

class MatchingEngine {
public:
    MatchingEngine();
    ~MatchingEngine() = default;

    // Non-copyable, movable
    MatchingEngine(const MatchingEngine&) = delete;
    MatchingEngine& operator=(const MatchingEngine&) = delete;
    MatchingEngine(MatchingEngine&&) noexcept = default;
    MatchingEngine& operator=(MatchingEngine&&) noexcept = default;

    // ---- Symbol management ----
    void add_symbol(SymbolConfig config);
    void remove_symbol(const Symbol& symbol);
    [[nodiscard]] bool has_symbol(const Symbol& symbol) const;
    [[nodiscard]] std::vector<Symbol> symbols() const;
    [[nodiscard]] const SymbolConfig& symbol_config(const Symbol& symbol) const;

    // ---- Order operations ----
    SubmitResult submit_order(Order order);
    bool cancel_order(OrderId order_id);
    BatchResult process_batch(std::vector<Order> orders);

    // ---- Market data ----
    [[nodiscard]] MarketDataSnapshot market_data(const Symbol& symbol) const;
    [[nodiscard]] BookSnapshot book_snapshot(const Symbol& symbol) const;
    [[nodiscard]] std::vector<MarketDataSnapshot> all_market_data() const;

    // ---- Book access ----
    [[nodiscard]] const OrderBook* get_book(const Symbol& symbol) const;
    [[nodiscard]] OrderBook*       get_book(const Symbol& symbol);

    // ---- Trade log ----
    [[nodiscard]] std::vector<TradeRecord> recent_trades(const Symbol& symbol, size_t n) const;
    [[nodiscard]] std::vector<TradeRecord> trades_since(const Symbol& symbol, Timestamp t) const;

    // ---- Performance ----
    [[nodiscard]] const PerformanceCounters& counters() const noexcept { return counters_; }
    void reset_counters() noexcept { counters_.reset(); }

    // ---- Rate limiter ----
    void set_rate_limit(uint32_t max_per_second);
    void disable_rate_limiting() noexcept { rate_limiting_enabled_ = false; }
    void enable_rate_limiting() noexcept  { rate_limiting_enabled_ = true; }

    // ---- Self-trade prevention ----
    void enable_self_trade_prevention() noexcept  { stp_enabled_ = true; }
    void disable_self_trade_prevention() noexcept { stp_enabled_ = false; }
    [[nodiscard]] bool self_trade_prevention_enabled() const noexcept { return stp_enabled_; }

    // ---- Global halt ----
    void halt_all() noexcept;
    void resume_all() noexcept;
    [[nodiscard]] bool is_globally_halted() const noexcept { return global_halt_; }

    // ---- Order ID generation ----
    [[nodiscard]] OrderId next_order_id() noexcept { return next_order_id_++; }

    // ---- Session ----
    void reset_all_sessions();

    // ---- Timestamp helper ----
    [[nodiscard]] static Timestamp now() noexcept {
        return std::chrono::steady_clock::now();
    }

private:
    // ---- Validation ----
    [[nodiscard]] RejectReason pre_validate(const Order& order) const;

    // ---- Data ----
    std::unordered_map<Symbol, std::unique_ptr<OrderBook>>  books_;
    std::unordered_map<OrderId, Symbol>                     order_symbol_index_;

    PerformanceCounters counters_;
    RateLimiter         rate_limiter_;

    bool     rate_limiting_enabled_ = true;
    bool     stp_enabled_           = true;
    bool     global_halt_           = false;
    OrderId  next_order_id_         = 1;
};

// ============================================================================
// Utility functions
// ============================================================================

namespace util {

[[nodiscard]] inline bool price_equal(Price a, Price b) noexcept {
    return std::abs(a - b) < constants::PRICE_EPSILON;
}

[[nodiscard]] inline bool qty_zero(Quantity q) noexcept {
    return q < constants::QTY_EPSILON;
}

[[nodiscard]] inline double round_to_tick(double price, double tick_size) noexcept {
    if (tick_size < constants::PRICE_EPSILON) return price;
    return std::round(price / tick_size) * tick_size;
}

[[nodiscard]] inline double round_to_lot(double qty, double lot_size) noexcept {
    if (lot_size < constants::QTY_EPSILON) return qty;
    return std::floor(qty / lot_size) * lot_size;
}

[[nodiscard]] inline int64_t to_nanos(Duration d) noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

[[nodiscard]] inline int64_t to_micros(Duration d) noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

} // namespace util

} // namespace srfm::exchange

#endif // SRFM_EXCHANGE_MATCHING_ENGINE_HPP
