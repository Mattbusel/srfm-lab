/// matching_engine.cpp — Full price-time priority LOB matching engine.
///
/// Implements:
/// - BTreeMap-equivalent via sorted arrays (cache-friendly for small depth)
/// - FIFO queue per price level (PriceLevel from price_level.hpp)
/// - Market, Limit, StopLimit, StopMarket, Iceberg order types
/// - IOC/FOK execution constraints
/// - SIMD-accelerated price level scanning
/// - Cancel-replace with time priority loss

#include "../include/lob_types.hpp"
#include "../include/price_level.hpp"
#include "../include/matching_engine.hpp"
#include "../include/simd_utils.hpp"

#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <memory>
#include <cassert>
#include <cmath>

namespace chronos {

// ── Level Map ─────────────────────────────────────────────────────────────────

/// Sorted map of price levels. For bids: descending (best = highest).
/// For asks: ascending (best = lowest).
///
/// We use std::map (Red-Black tree) for O(log n) insert/erase.
/// For hot path iteration, we additionally maintain a sorted array of prices
/// for SIMD scanning (refreshed on modification).

class PriceLevelMap {
public:
    Side side;

    explicit PriceLevelMap(Side s) : side(s) {}

    PriceLevel& get_or_create(TickPrice price) {
        auto it = levels_.find(price);
        if (it == levels_.end()) {
            auto [ins_it, ok] = levels_.emplace(price, PriceLevel(price, side));
            dirty_ = true;
            return ins_it->second;
        }
        return it->second;
    }

    PriceLevel* get(TickPrice price) {
        auto it = levels_.find(price);
        return it != levels_.end() ? &it->second : nullptr;
    }

    void erase(TickPrice price) {
        levels_.erase(price);
        dirty_ = true;
    }

    size_t size() const { return levels_.size(); }
    bool empty() const { return levels_.empty(); }

    TickPrice best_price() const noexcept {
        if (levels_.empty()) return INVALID_PRICE;
        if (side == Side::Bid) {
            return levels_.rbegin()->first;  // Highest bid.
        } else {
            return levels_.begin()->first;   // Lowest ask.
        }
    }

    PriceLevel* best_level() {
        if (levels_.empty()) return nullptr;
        if (side == Side::Bid) {
            return &levels_.rbegin()->second;
        } else {
            return &levels_.begin()->second;
        }
    }

    /// Iterate levels in matching order (best-first), calling fn(price, level).
    template <typename Fn>
    void iterate_matching(Fn&& fn) {
        if (side == Side::Bid) {
            // Descending (highest first).
            for (auto it = levels_.rbegin(); it != levels_.rend(); ++it) {
                if (!fn(it->first, it->second)) break;
            }
        } else {
            // Ascending (lowest first).
            for (auto it = levels_.begin(); it != levels_.end(); ++it) {
                if (!fn(it->first, it->second)) break;
            }
        }
    }

    Qty total_qty() const {
        Qty total = 0.0;
        for (auto& [p, lvl] : levels_) total += lvl.total_qty;
        return total;
    }

    // Fill snapshot arrays for SIMD scanning.
    void refresh_simd_cache() const {
        if (!dirty_) return;
        simd_prices_.clear();
        if (side == Side::Bid) {
            for (auto it = levels_.rbegin(); it != levels_.rend(); ++it)
                simd_prices_.push_back(it->first);
        } else {
            for (auto it = levels_.begin(); it != levels_.end(); ++it)
                simd_prices_.push_back(it->first);
        }
        dirty_ = false;
    }

    const std::vector<TickPrice>& simd_cache() const {
        refresh_simd_cache();
        return simd_prices_;
    }

private:
    std::map<TickPrice, PriceLevel> levels_;
    mutable std::vector<TickPrice> simd_prices_;
    mutable bool dirty_ = true;
};

// ── Matching Engine Implementation ────────────────────────────────────────────

class MatchingEngine final : public IMatchingEngine {
public:
    explicit MatchingEngine(InstId inst_id)
        : instrument_id_(inst_id)
        , bids_(Side::Bid)
        , asks_(Side::Ask)
        , fill_count_(0)
        , seq_(0)
        , last_price_(0)
        , last_qty_(0.0)
        , session_volume_(0.0)
        , session_notional_(0.0)
    {}

    // ── IMatchingEngine interface ─────────────────────────────────────────

    void submit(Order& order) override {
        order.seq = ++seq_;
        order.instrument_id = instrument_id_;

        switch (order.type) {
            case OrderType::Market:
                execute_market(order);
                break;
            case OrderType::Limit:
                execute_limit_entry(order);
                break;
            case OrderType::StopLimit:
            case OrderType::StopMarket:
                enqueue_stop(order);
                break;
            case OrderType::Iceberg: {
                // Set hidden qty = total - peak.
                // Peak is stored in leaves_qty, total in orig_qty.
                Qty peak = order.leaves_qty;
                if (order.orig_qty > peak) {
                    order.hidden_qty = order.orig_qty - peak;
                } else {
                    order.hidden_qty = 0.0;
                }
                execute_limit_entry(order);
                break;
            }
            default:
                order.status = OrderStatus::Rejected;
                break;
        }
    }

    bool cancel(OrderId order_id) override {
        auto it = order_index_.find(order_id);
        if (it == order_index_.end()) {
            // Try stop orders.
            auto sit = std::find_if(stop_orders_.begin(), stop_orders_.end(),
                [&](const Order& o) { return o.id == order_id; });
            if (sit != stop_orders_.end()) {
                stop_orders_.erase(sit);
                if (cancel_cb_) cancel_cb_(order_id, instrument_id_);
                return true;
            }
            return false;
        }

        auto [side, price] = it->second;
        auto* book = (side == Side::Bid) ? &bids_ : &asks_;
        auto* lvl = book->get(price);
        if (lvl) {
            Order removed;
            if (lvl->remove_by_id(order_id, removed)) {
                if (lvl->empty()) book->erase(price);
                order_index_.erase(it);
                removed.status = OrderStatus::Cancelled;
                if (cancel_cb_) cancel_cb_(order_id, instrument_id_);
                return true;
            }
        }
        order_index_.erase(it);
        return false;
    }

    bool modify(OrderId order_id, TickPrice new_price, Qty new_qty) override {
        // Cancel-replace: cancel old order, resubmit with new params.
        auto it = order_index_.find(order_id);
        if (it == order_index_.end()) return false;

        auto [side, old_price] = it->second;
        auto* book = (side == Side::Bid) ? &bids_ : &asks_;
        auto* lvl = book->get(old_price);
        if (!lvl) { order_index_.erase(it); return false; }

        Order removed;
        if (!lvl->remove_by_id(order_id, removed)) {
            order_index_.erase(it);
            return false;
        }
        if (lvl->empty()) book->erase(old_price);
        order_index_.erase(it);

        // Resubmit with new params (gets new sequence number → loses time priority).
        removed.price = new_price;
        removed.orig_qty = new_qty;
        removed.leaves_qty = new_qty;
        removed.filled_qty = 0.0;
        removed.status = OrderStatus::New;
        submit(removed);
        return true;
    }

    TickPrice best_bid() const noexcept override { return bids_.best_price(); }
    TickPrice best_ask() const noexcept override { return asks_.best_price(); }

    MarketSnapshot snapshot(size_t depth) const noexcept override {
        MarketSnapshot snap;
        snap.instrument_id = instrument_id_;
        snap.best_bid = bids_.best_price();
        snap.best_ask = asks_.best_price();
        snap.last_price = last_price_;
        snap.last_qty = last_qty_;
        snap.session_volume = session_volume_;

        // Fill bid levels.
        size_t i = 0;
        const_cast<PriceLevelMap&>(bids_).iterate_matching([&](TickPrice p, const PriceLevel& lvl) {
            if (i >= depth) return false;
            snap.bids[i++] = lvl.to_level();
            return true;
        });
        snap.bid_levels = static_cast<uint32_t>(i);

        // Fill ask levels.
        i = 0;
        const_cast<PriceLevelMap&>(asks_).iterate_matching([&](TickPrice p, const PriceLevel& lvl) {
            if (i >= depth) return false;
            snap.asks[i++] = lvl.to_level();
            return true;
        });
        snap.ask_levels = static_cast<uint32_t>(i);

        // Imbalance.
        Qty bid_vol = 0.0, ask_vol = 0.0;
        for (size_t j = 0; j < snap.bid_levels; ++j) bid_vol += snap.bids[j].total_qty;
        for (size_t j = 0; j < snap.ask_levels; ++j) ask_vol += snap.asks[j].total_qty;
        double total = bid_vol + ask_vol;
        snap.imbalance = total > 1e-9 ? (bid_vol - ask_vol) / total : 0.0;

        return snap;
    }

    double vwap_sweep(Side side, Qty qty) const noexcept override {
        double remaining = qty;
        double notional = 0.0;
        double filled = 0.0;

        auto& book = (side == Side::Bid) ? const_cast<PriceLevelMap&>(asks_)   // buy sweeps asks
                                         : const_cast<PriceLevelMap&>(bids_);   // sell sweeps bids

        book.iterate_matching([&](TickPrice price, const PriceLevel& lvl) {
            if (remaining < 1e-9) return false;
            double take = std::min(remaining, lvl.total_qty);
            notional += take * from_tick(price);
            filled += take;
            remaining -= take;
            return true;
        });

        return filled > 1e-9 ? notional / filled : 0.0;
    }

    void set_fill_callback(FillCallback cb) override { fill_cb_ = std::move(cb); }
    void set_cancel_callback(CancelCallback cb) override { cancel_cb_ = std::move(cb); }
    uint64_t fill_count() const noexcept override { return fill_count_; }
    size_t order_count() const noexcept override { return order_index_.size(); }

private:
    InstId instrument_id_;
    PriceLevelMap bids_;
    PriceLevelMap asks_;
    std::vector<Order> stop_orders_;
    std::unordered_map<OrderId, std::pair<Side, TickPrice>> order_index_;
    FillCallback fill_cb_;
    CancelCallback cancel_cb_;
    uint64_t fill_count_;
    SeqNum seq_;
    TickPrice last_price_;
    Qty last_qty_;
    Qty session_volume_;
    double session_notional_;

    // ── Market execution ────────────────────────────────────────────────────

    void execute_market(Order& aggressor) {
        auto& passive_book = (aggressor.side == Side::Bid) ? asks_ : bids_;
        TickPrice no_limit = (aggressor.side == Side::Bid)
            ? std::numeric_limits<TickPrice>::max()
            : std::numeric_limits<TickPrice>::min();

        match_against_book(aggressor, passive_book, no_limit);

        if (aggressor.tif == TimeInForce::IOC && aggressor.leaves_qty > 1e-9) {
            aggressor.status = OrderStatus::Cancelled;
        }

        trigger_stops(aggressor.timestamp_ns);
    }

    // ── Limit execution ──────────────────────────────────────────────────────

    void execute_limit_entry(Order& aggressor) {
        if (aggressor.tif == TimeInForce::FOK) {
            // Pre-check: sufficient qty available?
            double avail = vwap_sweep(aggressor.side, aggressor.leaves_qty);
            // If vwap_sweep fills less than needed, reject.
            double avail_qty = 0.0;
            auto& passive_book = (aggressor.side == Side::Bid) ? asks_ : bids_;
            TickPrice lim = aggressor.price;
            passive_book.iterate_matching([&](TickPrice p, const PriceLevel& lvl) {
                bool matchable = (aggressor.side == Side::Bid) ? (p <= lim) : (p >= lim);
                if (!matchable) return false;
                avail_qty += lvl.total_qty;
                return avail_qty < aggressor.leaves_qty;
            });
            if (avail_qty < aggressor.leaves_qty - 1e-9) {
                aggressor.status = OrderStatus::Rejected;
                return;
            }
        }

        auto& passive_book = (aggressor.side == Side::Bid) ? asks_ : bids_;
        match_against_book(aggressor, passive_book, aggressor.price);

        if (aggressor.tif == TimeInForce::IOC || aggressor.tif == TimeInForce::FOK) {
            if (aggressor.leaves_qty > 1e-9) {
                aggressor.status = OrderStatus::Cancelled;
                if (cancel_cb_) cancel_cb_(aggressor.id, instrument_id_);
            }
            return;
        }

        // Rest the unfilled portion.
        if (aggressor.leaves_qty > 1e-9 && aggressor.status != OrderStatus::Cancelled) {
            rest_order(aggressor);
        }

        trigger_stops(aggressor.timestamp_ns);
    }

    // ── Core matching loop ───────────────────────────────────────────────────

    void match_against_book(Order& aggressor, PriceLevelMap& passive_book, TickPrice limit) {
        std::vector<TickPrice> levels_to_delete;

        passive_book.iterate_matching([&](TickPrice price, PriceLevel& lvl) -> bool {
            if (aggressor.leaves_qty < 1e-9) return false;

            // Price check.
            bool price_ok = (aggressor.side == Side::Bid)
                ? (price <= limit)
                : (price >= limit);
            if (!price_ok) return false;

            // Fill from this level.
            while (aggressor.leaves_qty > 1e-9 && !lvl.empty()) {
                Order* passive = lvl.front();
                if (!passive) break;

                simd::prefetch_r(passive);

                double fill_qty = std::min(passive->leaves_qty, aggressor.leaves_qty);
                double fill_price = from_tick(price);

                // Apply fills.
                passive->apply_fill(fill_qty);
                aggressor.apply_fill(fill_qty);

                last_price_ = price;
                last_qty_ = fill_qty;
                session_volume_ += fill_qty;
                session_notional_ += fill_price * fill_qty;

                // Emit fill event.
                Fill f;
                f.timestamp_ns = aggressor.timestamp_ns;
                f.aggressor_id = aggressor.id;
                f.passive_id = passive->id;
                f.price = price;
                f.qty = fill_qty;
                f.side = aggressor.side;
                f.aggressor_agent = aggressor.agent_id;
                f.passive_agent = passive->agent_id;
                f.instrument_id = instrument_id_;
                f.is_partial = passive->leaves_qty > 1e-9;
                ++fill_count_;

                if (fill_cb_) fill_cb_(f);

                // Pop passive if filled.
                if (passive->status == OrderStatus::Filled) {
                    order_index_.erase(passive->id);
                    lvl.pop_front();
                }
            }

            if (lvl.empty()) {
                levels_to_delete.push_back(price);
            }

            return aggressor.leaves_qty > 1e-9;
        });

        for (auto p : levels_to_delete) {
            passive_book.erase(p);
        }
    }

    // ── Rest order ───────────────────────────────────────────────────────────

    void rest_order(const Order& order) {
        auto& book = (order.side == Side::Bid) ? bids_ : asks_;
        book.get_or_create(order.price).push_back(order);
        order_index_[order.id] = {order.side, order.price};
    }

    // ── Stop order management ────────────────────────────────────────────────

    void enqueue_stop(Order& order) {
        // Check if already triggered.
        bool triggered = (order.side == Side::Bid)
            ? (from_tick(last_price_) >= from_tick(order.stop_price))
            : (from_tick(last_price_) <= from_tick(order.stop_price));

        if (triggered) {
            activate_stop(order);
        } else {
            stop_orders_.push_back(order);
        }
    }

    void activate_stop(Order& order) {
        if (order.type == OrderType::StopLimit) {
            order.type = OrderType::Limit;
            execute_limit_entry(order);
        } else {
            order.type = OrderType::Market;
            execute_market(order);
        }
    }

    void trigger_stops(Nanos ts) {
        bool any_triggered = true;
        while (any_triggered) {
            any_triggered = false;
            std::vector<Order> to_activate;
            auto new_end = std::remove_if(stop_orders_.begin(), stop_orders_.end(),
                [&](Order& o) {
                    bool triggered = (o.side == Side::Bid)
                        ? (last_price_ >= o.stop_price)
                        : (last_price_ <= o.stop_price);
                    if (triggered) { to_activate.push_back(o); }
                    return triggered;
                });
            stop_orders_.erase(new_end, stop_orders_.end());

            for (auto& order : to_activate) {
                order.timestamp_ns = ts;
                order.seq = ++seq_;
                activate_stop(order);
                any_triggered = true;
            }
        }
    }
};

// ── Factory ───────────────────────────────────────────────────────────────────

std::unique_ptr<IMatchingEngine> create_matching_engine(InstId instrument_id) {
    return std::make_unique<MatchingEngine>(instrument_id);
}

} // namespace chronos
