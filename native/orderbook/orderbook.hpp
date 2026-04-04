#pragma once
#include "order.hpp"
#include <map>
#include <unordered_map>
#include <vector>
#include <functional>
#include <optional>
#include <atomic>
#include <mutex>

namespace hft {

struct BookStats {
    Price    best_bid;
    Price    best_ask;
    Price    spread;
    double   mid_price;
    Quantity bid_depth;   // total qty on bid side
    Quantity ask_depth;   // total qty on ask side
    double   imbalance;   // (bid_depth - ask_depth) / (bid_depth + ask_depth)
    uint32_t bid_levels;
    uint32_t ask_levels;
    uint64_t total_trades;
    Quantity total_volume;
    double   vwap;
};

// Callbacks for event notifications
using TradeCallback  = std::function<void(const Trade&)>;
using OrderCallback  = std::function<void(const Order&)>;
using CancelCallback = std::function<void(OrderId, Quantity remaining)>;

class OrderBook {
public:
    explicit OrderBook(const std::string& symbol, size_t max_levels = 1000);
    ~OrderBook();

    // Disable copy, allow move
    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;
    OrderBook(OrderBook&&) noexcept = default;

    // --- Order lifecycle ---
    bool        add_order(Order* order);
    bool        cancel_order(OrderId id);
    bool        modify_order(OrderId id, Price new_price, Quantity new_qty);
    std::vector<Trade> match_order(Order* aggressor);

    // --- Queries ---
    std::optional<Price>    best_bid() const noexcept;
    std::optional<Price>    best_ask() const noexcept;
    std::optional<double>   mid_price() const noexcept;
    std::optional<double>   spread() const noexcept;
    BookStats               stats() const noexcept;

    Quantity    bid_qty_at(Price p) const noexcept;
    Quantity    ask_qty_at(Price p) const noexcept;
    uint32_t    bid_levels_count() const noexcept;
    uint32_t    ask_levels_count() const noexcept;

    // L2 snapshot: returns up to `depth` levels
    std::vector<std::pair<Price, Quantity>> bid_snapshot(size_t depth = 10) const;
    std::vector<std::pair<Price, Quantity>> ask_snapshot(size_t depth = 10) const;

    const Order* find_order(OrderId id) const noexcept;
    size_t       order_count() const noexcept { return orders_.size(); }

    // --- Callbacks ---
    void set_trade_callback(TradeCallback cb)  { trade_cb_  = std::move(cb); }
    void set_fill_callback(OrderCallback cb)   { fill_cb_   = std::move(cb); }
    void set_cancel_callback(CancelCallback cb){ cancel_cb_ = std::move(cb); }

    const std::string& symbol() const noexcept { return symbol_; }

    // --- Trade reporting ---
    const std::vector<Trade>& trade_history() const noexcept { return trades_; }
    void clear_trade_history() { trades_.clear(); }

private:
    std::string symbol_;
    size_t      max_levels_;

    // Bid side: sorted descending (highest price first)
    std::map<Price, PriceLevel, std::greater<Price>> bids_;
    // Ask side: sorted ascending  (lowest price first)
    std::map<Price, PriceLevel, std::less<Price>>    asks_;

    // Fast order lookup by id
    std::unordered_map<OrderId, Order*> orders_;

    // Trade history
    std::vector<Trade>  trades_;
    std::atomic<uint64_t> trade_id_gen_{1};
    std::atomic<uint64_t> seq_gen_{1};

    // Cumulative for VWAP
    double   vwap_num_{0.0};
    uint64_t vwap_den_{0};

    // Callbacks
    TradeCallback   trade_cb_;
    OrderCallback   fill_cb_;
    CancelCallback  cancel_cb_;

    // Internal helpers
    std::vector<Trade> match_limit(Order* aggressor);
    std::vector<Trade> match_market(Order* aggressor);
    void               execute_fill(Order* passive, Order* aggressor,
                                    Quantity fill_qty, Price fill_price,
                                    std::vector<Trade>& out);
    void               remove_empty_level(Side side, Price price);
    uint64_t           next_trade_id() noexcept { return trade_id_gen_.fetch_add(1, std::memory_order_relaxed); }
    uint64_t           next_seq()      noexcept { return seq_gen_.fetch_add(1, std::memory_order_relaxed); }
};

} // namespace hft
