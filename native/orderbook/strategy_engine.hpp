#pragma once
// Strategy execution engine: connects order book, risk manager,
// and feed handler into a trading strategy framework.
// Provides signal → order translation with latency tracking.

#include "orderbook.hpp"
#include "risk_checks.hpp"
#include "feed_handler.hpp"
#include "matching_engine.hpp"
#include "market_impact.hpp"
#include <functional>
#include <vector>
#include <deque>
#include <string>
#include <memory>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <unordered_map>

namespace hft {

// ============================================================
// Signal types for strategy → order generation
// ============================================================
enum class SignalType : uint8_t {
    None      = 0,
    BuyLimit  = 1,
    SellLimit = 2,
    BuyMkt    = 3,
    SellMkt   = 4,
    Cancel    = 5,
    Flatten   = 6,   // close all positions
    ModifyQty = 7,
};

struct Signal {
    SignalType  type;
    char        symbol[16];
    Price       price;
    Quantity    qty;
    OrderId     cancel_id;   // for Cancel
    TimeInForce tif;
    double      confidence;  // [0,1]
    int64_t     signal_ts;   // when signal was generated (ns)
};

// ============================================================
// Market Microstructure Features
// These are computed from the order book and used by strategies.
// ============================================================
struct BookFeatures {
    double mid_price;
    double spread;
    double spread_bps;
    double bid_depth;
    double ask_depth;
    double order_imbalance;       // (bid - ask) / (bid + ask)
    double weighted_mid;          // volume-weighted mid
    double bid_slope;             // bid qty / bid price distance from mid
    double ask_slope;
    double book_pressure;         // net flow indicator
    double realized_spread;       // trade price vs mid at time of trade
    double effective_spread;      // 2 * |trade_price - pre_trade_mid|
    double price_impact;          // post-trade mid vs pre-trade mid
    double kyle_lambda;           // price impact per unit order flow
    int64_t timestamp;

    static BookFeatures from_book(const OrderBook& book,
                                   const MarketImpactModel& impact)
    {
        BookFeatures f{};
        f.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        auto bb = book.best_bid();
        auto ba = book.best_ask();
        if (!bb || !ba) return f;

        double bid_p = price_to_double(*bb);
        double ask_p = price_to_double(*ba);
        f.mid_price  = (bid_p + ask_p) / 2.0;
        f.spread     = ask_p - bid_p;
        f.spread_bps = f.mid_price > 0 ? f.spread / f.mid_price * 10000.0 : 0.0;

        auto stats = book.stats();
        f.bid_depth = static_cast<double>(stats.bid_depth);
        f.ask_depth = static_cast<double>(stats.ask_depth);
        double tot  = f.bid_depth + f.ask_depth;
        f.order_imbalance = tot > 0 ? (f.bid_depth - f.ask_depth) / tot : 0.0;

        // Weighted mid
        if (tot > 0)
            f.weighted_mid = (bid_p * f.ask_depth + ask_p * f.bid_depth) / tot;
        else
            f.weighted_mid = f.mid_price;

        // Slopes (depth per tick away from BBO)
        double tick = 0.01; // 1 cent default
        f.bid_slope = f.bid_depth / std::max(f.spread / 2.0, tick);
        f.ask_slope = f.ask_depth / std::max(f.spread / 2.0, tick);
        f.book_pressure = f.bid_slope - f.ask_slope;

        auto ie = impact.estimate(1000.0, f.mid_price);
        f.kyle_lambda = ie.kyle_lambda;

        return f;
    }
};

// ============================================================
// Abstract strategy base class
// ============================================================
class Strategy {
public:
    virtual ~Strategy() = default;

    struct Context {
        const BookFeatures& features;
        const OrderBook&    book;
        const RiskManager&  risk;
        int64_t             timestamp_ns;
    };

    // Called on every book update. Returns signals (may be empty).
    virtual std::vector<Signal> on_book_update(const Context& ctx) = 0;

    // Called on every trade.
    virtual void on_trade(const Trade& t, const BookFeatures& features) { (void)t; (void)features; }

    // Called when our own order is filled.
    virtual void on_fill(const Order& order, Quantity fill_qty, Price fill_price) {
        (void)order; (void)fill_qty; (void)fill_price;
    }

    virtual std::string name() const = 0;
    virtual void reset() {}
};

// ============================================================
// Market Making Strategy
// Quotes at +/- offset from mid, adjusting for inventory and flow
// ============================================================
class MarketMakerStrategy : public Strategy {
public:
    struct Params {
        double base_half_spread = 2.0;  // ticks half-spread
        double tick_size        = 0.01;
        double max_inventory    = 1000.0;
        double target_inventory = 0.0;
        double inventory_skew   = 0.5;  // quote adjustment per unit inventory
        double vol_adjustment   = 1.5;  // spread multiplier when vol is high
        double vol_threshold    = 0.30; // annualized vol above which to widen
        Quantity quote_qty      = 100;
        size_t  max_levels      = 3;    // how many price levels to quote
        double  level_spacing   = 1.0;  // ticks between levels
    };

    explicit MarketMakerStrategy(const Params& p = {}) : p_(p) {}

    std::string name() const override { return "MarketMaker"; }

    std::vector<Signal> on_book_update(const Context& ctx) override {
        const auto& f = ctx.features;
        if (f.mid_price == 0) return {};

        std::vector<Signal> signals;
        update_count_++;

        // Cancel stale quotes
        if (active_bid_id_ > 0) {
            signals.push_back({SignalType::Cancel, {}, 0, 0, active_bid_id_, TimeInForce::GTC, 1.0, ctx.timestamp_ns});
            active_bid_id_ = 0;
        }
        if (active_ask_id_ > 0) {
            signals.push_back({SignalType::Cancel, {}, 0, 0, active_ask_id_, TimeInForce::GTC, 1.0, ctx.timestamp_ns});
            active_ask_id_ = 0;
        }

        // Compute adjusted spread
        double half_spread = p_.base_half_spread * p_.tick_size;

        // Inventory skew: shift quotes toward reducing inventory
        double inv_ratio = net_inventory_ / p_.max_inventory;
        inv_ratio = std::max(-1.0, std::min(1.0, inv_ratio));
        double skew = -inv_ratio * p_.inventory_skew * p_.tick_size;

        // Vol adjustment
        double spread_mult = 1.0;
        if (f.spread_bps > p_.vol_threshold * 10000.0 / 252.0) {
            spread_mult = p_.vol_adjustment;
        }
        half_spread *= spread_mult;

        // Imbalance signal: if strong buy flow, skew ask up
        double imb_adj = f.order_imbalance * p_.tick_size * 0.5;

        double bid_px = f.mid_price - half_spread + skew - imb_adj;
        double ask_px = f.mid_price + half_spread + skew - imb_adj;

        // Round to tick
        bid_px = std::floor(bid_px / p_.tick_size) * p_.tick_size;
        ask_px = std::ceil (ask_px / p_.tick_size) * p_.tick_size;

        // Don't cross the market
        double cur_bid = price_to_double(ctx.book.best_bid().value_or(0));
        double cur_ask = price_to_double(ctx.book.best_ask().value_or(0));
        if (cur_bid > 0) bid_px = std::min(bid_px, cur_bid);
        if (cur_ask > 0) ask_px = std::max(ask_px, cur_ask);
        if (bid_px >= ask_px) return signals;

        // Quote bid
        Signal bid{};
        bid.type  = SignalType::BuyLimit;
        bid.price = double_to_price(bid_px);
        bid.qty   = p_.quote_qty;
        bid.tif   = TimeInForce::GTC;
        bid.confidence = 0.7;
        bid.signal_ts  = ctx.timestamp_ns;
        active_bid_id_ = ++id_gen_;
        bid_px_ = bid_px;
        signals.push_back(bid);

        // Quote ask
        Signal ask{};
        ask.type  = SignalType::SellLimit;
        ask.price = double_to_price(ask_px);
        ask.qty   = p_.quote_qty;
        ask.tif   = TimeInForce::GTC;
        ask.confidence = 0.7;
        ask.signal_ts  = ctx.timestamp_ns;
        active_ask_id_ = ++id_gen_;
        ask_px_ = ask_px;
        signals.push_back(ask);

        return signals;
    }

    void on_fill(const Order& order, Quantity fill_qty, Price /*fill_price*/) override {
        if (order.side == Side::Buy)
            net_inventory_ += static_cast<double>(fill_qty);
        else
            net_inventory_ -= static_cast<double>(fill_qty);

        total_fills_++;
        total_fill_qty_ += fill_qty;
    }

    double net_inventory() const noexcept { return net_inventory_; }
    uint64_t update_count() const noexcept { return update_count_; }
    uint64_t total_fills() const noexcept { return total_fills_; }

    void reset() override {
        net_inventory_ = 0;
        active_bid_id_ = active_ask_id_ = 0;
        update_count_ = total_fills_ = total_fill_qty_ = 0;
    }

private:
    Params  p_;
    double  net_inventory_  = 0.0;
    double  bid_px_         = 0.0;
    double  ask_px_         = 0.0;
    OrderId active_bid_id_  = 0;
    OrderId active_ask_id_  = 0;
    uint64_t id_gen_        = 10000000;
    uint64_t update_count_  = 0;
    uint64_t total_fills_   = 0;
    uint64_t total_fill_qty_= 0;
};

// ============================================================
// Statistical Arbitrage Strategy (pairs trading)
// Monitors spread between two correlated instruments
// ============================================================
class StatArbStrategy : public Strategy {
public:
    struct Params {
        std::string sym_a;
        std::string sym_b;
        double      beta         = 1.0;    // hedge ratio
        double      entry_z      = 2.0;    // z-score to enter
        double      exit_z       = 0.5;    // z-score to exit
        size_t      lookback     = 60;     // spread mean/std estimation window
        Quantity    trade_qty    = 100;
    };

    explicit StatArbStrategy(const Params& p) : p_(p) {}

    std::string name() const override { return "StatArb_" + p_.sym_a + "_" + p_.sym_b; }

    std::vector<Signal> on_book_update(const Context& /*ctx*/) override {
        return {}; // Uses two-book context; overridden in derived
    }

    // Two-book update
    std::vector<Signal> on_book_pair(double mid_a, double mid_b, int64_t ts) {
        if (mid_a == 0 || mid_b == 0) return {};

        // Update spread history
        double spread_val = mid_a - p_.beta * mid_b;
        spread_hist_.push_back(spread_val);
        if (spread_hist_.size() > p_.lookback) spread_hist_.pop_front();

        if (spread_hist_.size() < 20) return {};

        // Compute z-score
        double mean = 0.0, var = 0.0;
        for (auto v : spread_hist_) mean += v;
        mean /= spread_hist_.size();
        for (auto v : spread_hist_) var += (v - mean) * (v - mean);
        var /= spread_hist_.size();
        double std_dev = std::sqrt(var);
        double z = std_dev > 1e-8 ? (spread_val - mean) / std_dev : 0.0;

        std::vector<Signal> signals;

        if (!in_position_ && std::fabs(z) > p_.entry_z) {
            // Enter: sell A, buy B (or vice versa)
            Signal sig_a{}, sig_b{};
            if (z > 0) {
                // Spread too wide: sell A, buy B
                sig_a.type = SignalType::SellMkt;
                sig_b.type = SignalType::BuyMkt;
                position_dir_ = -1;
            } else {
                sig_a.type = SignalType::BuyMkt;
                sig_b.type = SignalType::SellMkt;
                position_dir_ = 1;
            }
            sig_a.qty = sig_b.qty = p_.trade_qty;
            sig_a.signal_ts = sig_b.signal_ts = ts;
            sig_a.confidence = sig_b.confidence = std::min(std::fabs(z) / p_.entry_z, 1.0);
            signals.push_back(sig_a);
            signals.push_back(sig_b);
            in_position_ = true;
            entry_z_ = z;
            trades_++;
        } else if (in_position_ && std::fabs(z) < p_.exit_z) {
            // Exit
            Signal sig_a{}, sig_b{};
            sig_a.type = (position_dir_ == -1) ? SignalType::BuyMkt : SignalType::SellMkt;
            sig_b.type = (position_dir_ == -1) ? SignalType::SellMkt : SignalType::BuyMkt;
            sig_a.qty = sig_b.qty = p_.trade_qty;
            sig_a.signal_ts = sig_b.signal_ts = ts;
            signals.push_back(sig_a);
            signals.push_back(sig_b);
            in_position_ = false;
            position_dir_ = 0;
        }
        return signals;
    }

    double current_z() const noexcept { return entry_z_; }
    bool   in_position() const noexcept { return in_position_; }
    uint64_t trades() const noexcept { return trades_; }

private:
    Params              p_;
    std::deque<double>  spread_hist_;
    bool                in_position_ = false;
    int                 position_dir_ = 0;
    double              entry_z_ = 0.0;
    uint64_t            trades_  = 0;
};

// ============================================================
// Latency tracker for strategy → order submission
// ============================================================
struct LatencyHistogram {
    static constexpr size_t BUCKETS = 64;
    std::array<uint64_t, BUCKETS> counts = {};
    uint64_t min_ns = UINT64_MAX;
    uint64_t max_ns = 0;
    uint64_t sum_ns = 0;
    uint64_t total  = 0;

    void record(uint64_t ns) {
        size_t bucket = 0;
        uint64_t v = ns;
        while (v > 0 && bucket < BUCKETS - 1) { v >>= 1; ++bucket; }
        counts[bucket]++;
        if (ns < min_ns) min_ns = ns;
        if (ns > max_ns) max_ns = ns;
        sum_ns += ns;
        ++total;
    }

    uint64_t mean_ns() const { return total > 0 ? sum_ns / total : 0; }

    uint64_t percentile_ns(double p) const {
        uint64_t target = static_cast<uint64_t>(p * total);
        uint64_t cum = 0;
        for (size_t i = 0; i < BUCKETS; ++i) {
            cum += counts[i];
            if (cum >= target) return 1ULL << i;
        }
        return max_ns;
    }

    void print(const std::string& label) const {
        if (total == 0) return;
        std::cout << label << ": n=" << total
                  << " min=" << min_ns << "ns"
                  << " avg=" << mean_ns() << "ns"
                  << " p50=" << percentile_ns(0.50) << "ns"
                  << " p99=" << percentile_ns(0.99) << "ns"
                  << " max=" << max_ns << "ns" << std::endl;
    }
};

// ============================================================
// Strategy Engine: orchestrates multiple strategies
// ============================================================
class StrategyEngine {
public:
    explicit StrategyEngine(const std::string& symbol,
                             RiskLimits limits = {})
        : risk_(limits), book_(symbol), impact_()
    {}

    void add_strategy(std::unique_ptr<Strategy> strat) {
        strategies_.push_back(std::move(strat));
    }

    void set_reference_price(Price p) {
        risk_.set_reference_price(book_.symbol().c_str(), p);
    }

    // Process a book update: compute features, run strategies, submit orders
    void on_book_update() {
        int64_t t0 = now_ns();
        auto features = BookFeatures::from_book(book_, impact_);

        Strategy::Context ctx{features, book_, risk_, t0};

        for (auto& strat : strategies_) {
            auto signals = strat->on_book_update(ctx);
            for (auto& sig : signals) {
                process_signal(sig, *strat);
            }
        }

        int64_t lat = now_ns() - t0;
        latency_.record(static_cast<uint64_t>(lat));
        book_updates_++;
    }

    void on_trade(const Trade& t) {
        auto features = BookFeatures::from_book(book_, impact_);
        for (auto& strat : strategies_) strat->on_trade(t, features);

        impact_.on_trade(
            features.mid_price - t.qty * 1e-6,
            features.mid_price,
            t.aggressor_side == Side::Buy ? t.qty : -(int64_t)t.qty,
            price_to_double(t.price) * t.qty);

        risk_.on_fill(t.symbol, t.aggressor_side, t.qty, t.price);
        trades_++;
    }

    OrderBook&      book()  { return book_; }
    RiskManager&    risk()  { return risk_; }
    const LatencyHistogram& latency_hist() const { return latency_; }

    uint64_t book_updates() const noexcept { return book_updates_; }
    uint64_t trades()       const noexcept { return trades_; }
    uint64_t orders_sent()  const noexcept { return orders_sent_; }
    uint64_t orders_rejected() const noexcept { return orders_rejected_; }

    void print_stats() const {
        std::cout << "=== Strategy Engine Stats ===" << std::endl;
        std::cout << "  Book updates: " << book_updates_ << std::endl;
        std::cout << "  Trades seen:  " << trades_ << std::endl;
        std::cout << "  Orders sent:  " << orders_sent_ << std::endl;
        std::cout << "  Rejected:     " << orders_rejected_ << std::endl;
        latency_.print("  Signal→order latency");
    }

private:
    RiskManager         risk_;
    OrderBook           book_;
    MarketImpactModel   impact_;
    std::vector<std::unique_ptr<Strategy>> strategies_;
    LatencyHistogram    latency_;
    uint64_t            book_updates_     = 0;
    uint64_t            trades_           = 0;
    uint64_t            orders_sent_      = 0;
    uint64_t            orders_rejected_  = 0;

    void process_signal(const Signal& sig, Strategy& strat) {
        if (sig.type == SignalType::None) return;
        if (sig.type == SignalType::Cancel) {
            book_.cancel_order(sig.cancel_id);
            return;
        }

        // Build order
        Order o;
        o.id         = ++order_id_gen_;
        o.side       = (sig.type == SignalType::BuyLimit || sig.type == SignalType::BuyMkt)
                       ? Side::Buy : Side::Sell;
        o.order_type = (sig.type == SignalType::BuyMkt || sig.type == SignalType::SellMkt)
                       ? OrderType::Market : OrderType::Limit;
        o.tif        = sig.tif;
        o.price      = sig.price;
        o.qty        = sig.qty;
        o.timestamp  = sig.signal_ts;
        std::strncpy(o.symbol, book_.symbol().c_str(), 15);

        auto rej = risk_.check_order(o);
        if (rej != RejectionReason::None) {
            ++orders_rejected_;
            return;
        }

        auto trades = book_.add_order(&o);
        ++orders_sent_;

        for (auto& t : trades) {
            strat.on_fill(o, t.qty, t.price);
        }
    }

    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    uint64_t order_id_gen_ = 1000000;
};

} // namespace hft
