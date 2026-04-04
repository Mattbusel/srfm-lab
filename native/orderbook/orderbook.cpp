#include "orderbook.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace hft {

static Timestamp now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
               .count();
}

OrderBook::OrderBook(const std::string& symbol, size_t max_levels)
    : symbol_(symbol), max_levels_(max_levels)
{
    orders_.reserve(65536);
    trades_.reserve(8192);
}

OrderBook::~OrderBook() {
    // Orders are owned externally; just clear our map
    orders_.clear();
}

bool OrderBook::add_order(Order* order) {
    if (!order) return false;
    if (orders_.count(order->id)) return false; // duplicate

    // Set sequence number and timestamp if not set
    if (order->timestamp == 0) order->timestamp = now_ns();
    order->sequence_num = next_seq();

    // Market orders go directly to matching
    if (order->order_type == OrderType::Market) {
        auto fills = match_order(order);
        // If IOC/FOK and not filled, cancel remainder
        if (order->is_active()) {
            order->status = OrderStatus::Cancelled;
        }
        return true;
    }

    // For iceberg orders, set display qty
    if (order->order_type == OrderType::Iceberg && order->display_qty == 0) {
        order->display_qty = order->qty / 10; // default: 10% visible
        if (order->display_qty == 0) order->display_qty = 1;
        order->hidden_qty  = order->qty - order->display_qty;
    }

    // Try to match first (for aggressive limit orders)
    auto fills = match_order(order);

    // If still active, add to book
    if (order->is_active()) {
        if (order->tif == TimeInForce::IOC) {
            order->status = OrderStatus::Cancelled;
            if (cancel_cb_) cancel_cb_(order->id, order->leaves_qty());
            return true;
        }
        // FOK: if not fully filled, cancel entirely (already handled in match_order)
        if (order->tif == TimeInForce::FOK) {
            order->status = OrderStatus::Cancelled;
            if (cancel_cb_) cancel_cb_(order->id, order->leaves_qty());
            return true;
        }

        // Rest order on book
        orders_[order->id] = order;
        if (order->side == Side::Buy) {
            auto& level = bids_[order->price];
            if (level.price == 0) level.price = order->price;
            level.push_back(order);
        } else {
            auto& level = asks_[order->price];
            if (level.price == 0) level.price = order->price;
            level.push_back(order);
        }
    }
    return true;
}

bool OrderBook::cancel_order(OrderId id) {
    auto it = orders_.find(id);
    if (it == orders_.end()) return false;

    Order* order = it->second;
    Quantity remaining = order->leaves_qty();
    order->status = OrderStatus::Cancelled;

    if (order->side == Side::Buy) {
        auto bit = bids_.find(order->price);
        if (bit != bids_.end()) {
            bit->second.remove(order);
            if (bit->second.empty()) bids_.erase(bit);
        }
    } else {
        auto ait = asks_.find(order->price);
        if (ait != asks_.end()) {
            ait->second.remove(order);
            if (ait->second.empty()) asks_.erase(ait);
        }
    }

    orders_.erase(it);
    if (cancel_cb_) cancel_cb_(id, remaining);
    return true;
}

bool OrderBook::modify_order(OrderId id, Price new_price, Quantity new_qty) {
    auto it = orders_.find(id);
    if (it == orders_.end()) return false;

    Order* order = it->second;
    if (!order->is_active()) return false;

    // Cancel and re-add (loses queue priority)
    Quantity filled = order->filled_qty;
    Side side = order->side;
    OrderType otype = order->order_type;
    TimeInForce tif = order->tif;
    Timestamp ts = now_ns();

    cancel_order(id);

    order->price      = new_price;
    order->qty        = new_qty;
    order->filled_qty = filled;
    order->status     = OrderStatus::New;
    order->timestamp  = ts;
    order->sequence_num = next_seq();

    return add_order(order);
}

std::vector<Trade> OrderBook::match_order(Order* aggressor) {
    std::vector<Trade> result;

    if (aggressor->order_type == OrderType::Market) {
        return match_market(aggressor);
    }
    return match_limit(aggressor);
}

std::vector<Trade> OrderBook::match_market(Order* aggressor) {
    std::vector<Trade> result;

    if (aggressor->side == Side::Buy) {
        // Buy market: match against asks (ascending price)
        while (aggressor->is_active() && !asks_.empty()) {
            auto ait = asks_.begin();
            PriceLevel& level = ait->second;
            while (aggressor->is_active() && !level.empty()) {
                Order* passive = level.head;
                Quantity fill = std::min(aggressor->leaves_qty(), passive->leaves_qty());
                execute_fill(passive, aggressor, fill, level.price, result);
                if (!passive->is_active()) {
                    level.remove(passive);
                    orders_.erase(passive->id);
                }
            }
            if (level.empty()) asks_.erase(ait);
        }
    } else {
        // Sell market: match against bids (descending price)
        while (aggressor->is_active() && !bids_.empty()) {
            auto bit = bids_.begin();
            PriceLevel& level = bit->second;
            while (aggressor->is_active() && !level.empty()) {
                Order* passive = level.head;
                Quantity fill = std::min(aggressor->leaves_qty(), passive->leaves_qty());
                execute_fill(passive, aggressor, fill, level.price, result);
                if (!passive->is_active()) {
                    level.remove(passive);
                    orders_.erase(passive->id);
                }
            }
            if (level.empty()) bids_.erase(bit);
        }
    }
    return result;
}

std::vector<Trade> OrderBook::match_limit(Order* aggressor) {
    std::vector<Trade> result;

    // FOK: check if fully fillable before executing
    if (aggressor->tif == TimeInForce::FOK) {
        Quantity available = 0;
        if (aggressor->side == Side::Buy) {
            for (auto& [p, lvl] : asks_) {
                if (p > aggressor->price) break;
                available += lvl.total_qty;
                if (available >= aggressor->leaves_qty()) break;
            }
        } else {
            for (auto& [p, lvl] : bids_) {
                if (p < aggressor->price) break;
                available += lvl.total_qty;
                if (available >= aggressor->leaves_qty()) break;
            }
        }
        if (available < aggressor->leaves_qty()) {
            aggressor->status = OrderStatus::Cancelled;
            return result;
        }
    }

    if (aggressor->side == Side::Buy) {
        // Match against asks at or below our limit price
        while (aggressor->is_active() && !asks_.empty()) {
            auto ait = asks_.begin();
            if (ait->first > aggressor->price) break; // no more fillable levels
            PriceLevel& level = ait->second;
            while (aggressor->is_active() && !level.empty()) {
                Order* passive = level.head;
                Quantity fill = std::min(aggressor->leaves_qty(), passive->leaves_qty());
                execute_fill(passive, aggressor, fill, level.price, result);
                if (!passive->is_active()) {
                    level.remove(passive);
                    orders_.erase(passive->id);
                }
            }
            if (level.empty()) asks_.erase(ait);
        }
    } else {
        // Match against bids at or above our limit price
        while (aggressor->is_active() && !bids_.empty()) {
            auto bit = bids_.begin();
            if (bit->first < aggressor->price) break;
            PriceLevel& level = bit->second;
            while (aggressor->is_active() && !level.empty()) {
                Order* passive = level.head;
                Quantity fill = std::min(aggressor->leaves_qty(), passive->leaves_qty());
                execute_fill(passive, aggressor, fill, level.price, result);
                if (!passive->is_active()) {
                    level.remove(passive);
                    orders_.erase(passive->id);
                }
            }
            if (level.empty()) bids_.erase(bit);
        }
    }
    return result;
}

void OrderBook::execute_fill(Order* passive, Order* aggressor,
                              Quantity fill_qty, Price fill_price,
                              std::vector<Trade>& out)
{
    passive->filled_qty  += fill_qty;
    aggressor->filled_qty += fill_qty;

    // Handle iceberg replenishment
    if (passive->order_type == OrderType::Iceberg && passive->hidden_qty > 0) {
        Quantity replenish = std::min(passive->hidden_qty, passive->display_qty);
        passive->hidden_qty -= replenish;
        passive->qty        += replenish;
    }

    if (passive->filled_qty >= passive->qty)
        passive->status = OrderStatus::Filled;
    else if (passive->filled_qty > 0)
        passive->status = OrderStatus::PartialFill;

    if (aggressor->filled_qty >= aggressor->qty)
        aggressor->status = OrderStatus::Filled;
    else if (aggressor->filled_qty > 0)
        aggressor->status = OrderStatus::PartialFill;

    // Update level quantity
    if (passive->side == Side::Buy) {
        auto bit = bids_.find(fill_price);
        if (bit != bids_.end()) bit->second.total_qty -= fill_qty;
    } else {
        auto ait = asks_.find(fill_price);
        if (ait != asks_.end()) ait->second.total_qty -= fill_qty;
    }

    Trade t;
    t.trade_id       = next_trade_id();
    t.aggressor_id   = aggressor->id;
    t.passive_id     = passive->id;
    std::strncpy(t.symbol, symbol_.c_str(), 15);
    t.symbol[15]     = '\0';
    t.aggressor_side = aggressor->side;
    t.price          = fill_price;
    t.qty            = fill_qty;
    t.timestamp      = now_ns();
    t.sequence_num   = next_seq();

    // Update VWAP
    vwap_num_ += price_to_double(fill_price) * fill_qty;
    vwap_den_ += fill_qty;

    trades_.push_back(t);
    out.push_back(t);
    if (trade_cb_) trade_cb_(t);
    if (fill_cb_) { fill_cb_(*passive); fill_cb_(*aggressor); }
}

std::optional<Price> OrderBook::best_bid() const noexcept {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<Price> OrderBook::best_ask() const noexcept {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<double> OrderBook::mid_price() const noexcept {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb || !ba) return std::nullopt;
    return (price_to_double(*bb) + price_to_double(*ba)) / 2.0;
}

std::optional<double> OrderBook::spread() const noexcept {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb || !ba) return std::nullopt;
    return price_to_double(*ba) - price_to_double(*bb);
}

Quantity OrderBook::bid_qty_at(Price p) const noexcept {
    auto it = bids_.find(p);
    return it != bids_.end() ? it->second.total_qty : 0;
}

Quantity OrderBook::ask_qty_at(Price p) const noexcept {
    auto it = asks_.find(p);
    return it != asks_.end() ? it->second.total_qty : 0;
}

uint32_t OrderBook::bid_levels_count() const noexcept {
    return static_cast<uint32_t>(bids_.size());
}

uint32_t OrderBook::ask_levels_count() const noexcept {
    return static_cast<uint32_t>(asks_.size());
}

std::vector<std::pair<Price,Quantity>> OrderBook::bid_snapshot(size_t depth) const {
    std::vector<std::pair<Price,Quantity>> out;
    out.reserve(depth);
    for (auto& [p, lvl] : bids_) {
        if (out.size() >= depth) break;
        out.emplace_back(p, lvl.total_qty);
    }
    return out;
}

std::vector<std::pair<Price,Quantity>> OrderBook::ask_snapshot(size_t depth) const {
    std::vector<std::pair<Price,Quantity>> out;
    out.reserve(depth);
    for (auto& [p, lvl] : asks_) {
        if (out.size() >= depth) break;
        out.emplace_back(p, lvl.total_qty);
    }
    return out;
}

const Order* OrderBook::find_order(OrderId id) const noexcept {
    auto it = orders_.find(id);
    return it != orders_.end() ? it->second : nullptr;
}

BookStats OrderBook::stats() const noexcept {
    BookStats s{};
    auto bb = best_bid();
    auto ba = best_ask();
    s.best_bid   = bb.value_or(0);
    s.best_ask   = ba.value_or(0);
    if (bb && ba) {
        s.spread    = s.best_ask - s.best_bid;
        s.mid_price = (price_to_double(s.best_bid) + price_to_double(s.best_ask)) / 2.0;
    }
    for (auto& [p, lvl] : bids_) s.bid_depth += lvl.total_qty;
    for (auto& [p, lvl] : asks_) s.ask_depth += lvl.total_qty;
    uint64_t tot = s.bid_depth + s.ask_depth;
    s.imbalance  = tot ? (static_cast<double>(s.bid_depth) - s.ask_depth) / tot : 0.0;
    s.bid_levels = bid_levels_count();
    s.ask_levels = ask_levels_count();
    s.total_trades = trades_.size();
    s.total_volume = vwap_den_;
    s.vwap         = vwap_den_ > 0 ? vwap_num_ / vwap_den_ : 0.0;
    return s;
}

} // namespace hft
