// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// execution_gateway.cpp — Order routing and execution gateway
// =============================================================================
// Provides:
// - Virtual execution gateway interface
// - Simulated exchange (for backtesting and simulation)
// - Order book simulation with level-1 and level-2 data
// - FIX-protocol inspired message types
// - Gateway metrics and latency tracking

#include "rtel/ring_buffer.hpp"
#include "rtel/shm_bus.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtel {
namespace gateway {

static constexpr double kEps = 1e-12;
static std::atomic<uint64_t> g_order_id_counter{1};

// ---------------------------------------------------------------------------
// FIX-style message types
// ---------------------------------------------------------------------------
enum class MsgType {
    NewOrder,
    CancelOrder,
    ModifyOrder,
    ExecutionReport,
    OrderCancelReject,
    MarketData,
    Heartbeat,
    Reject,
};

enum class OrdStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
    PendingNew,
    PendingCancel,
};

enum class OrdType {
    Market,
    Limit,
    Stop,
    StopLimit,
    MarketOnClose,
    LimitOnClose,
};

enum class Side {
    Buy,
    Sell,
};

enum class TimeInForce {
    Day,
    GoodTillCancel,
    AtTheOpening,
    ImmediateOrCancel,
    FillOrKill,
    GoodTillDate,
};

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------
struct Order {
    uint64_t    order_id      = 0;
    uint32_t    asset_id      = 0;
    Side        side          = Side::Buy;
    OrdType     ord_type      = OrdType::Market;
    TimeInForce tif           = TimeInForce::GoodTillCancel;
    double      quantity      = 0.0;
    double      price         = 0.0;      // limit price (0 for market)
    double      stop_price    = 0.0;
    double      filled_qty    = 0.0;
    double      avg_px        = 0.0;
    OrdStatus   status        = OrdStatus::PendingNew;
    uint64_t    submit_ts_ns  = 0;
    uint64_t    update_ts_ns  = 0;
    char        client_tag[32]{};

    double remaining() const { return quantity - filled_qty; }
    bool   is_active() const {
        return status == OrdStatus::New || status == OrdStatus::PartiallyFilled ||
               status == OrdStatus::PendingNew;
    }
    bool   is_buy()    const { return side == Side::Buy; }
    double notional()  const { return avg_px * filled_qty; }
};

// ---------------------------------------------------------------------------
// Execution report
// ---------------------------------------------------------------------------
struct ExecutionReport {
    uint64_t    exec_id       = 0;
    uint64_t    order_id      = 0;
    uint32_t    asset_id      = 0;
    OrdStatus   ord_status    = OrdStatus::New;
    double      last_qty      = 0.0;
    double      last_px       = 0.0;
    double      cum_qty       = 0.0;
    double      avg_px        = 0.0;
    double      leaves_qty    = 0.0;
    double      commission    = 0.0;
    uint64_t    timestamp_ns  = 0;
    bool        is_maker      = false;
    char        exec_type     = 'F';   // 'F'=fill, 'C'=cancel, 'N'=new
};

// ---------------------------------------------------------------------------
// Market data snapshot
// ---------------------------------------------------------------------------
struct L2Snapshot {
    uint32_t asset_id;
    uint64_t timestamp_ns;
    double   bids[10][2];   // [price, size]
    double   asks[10][2];
    int      n_bid_levels;
    int      n_ask_levels;
    double   last_trade_price;
    double   last_trade_size;
    double   daily_volume;
};

// ---------------------------------------------------------------------------
// Gateway interface
// ---------------------------------------------------------------------------
class IExecutionGateway {
public:
    virtual ~IExecutionGateway() = default;

    virtual uint64_t submit_order(Order& order) = 0;
    virtual bool     cancel_order(uint64_t order_id) = 0;
    virtual bool     modify_order(uint64_t order_id, double new_qty,
                                   double new_price) = 0;
    virtual std::optional<Order> get_order(uint64_t order_id) const = 0;

    using ExecCallback = std::function<void(const ExecutionReport&)>;
    virtual void set_exec_callback(ExecCallback cb) = 0;

    virtual L2Snapshot get_market_data(uint32_t asset_id) const = 0;
};

// ---------------------------------------------------------------------------
// Simulated exchange (matching engine)
// ---------------------------------------------------------------------------
class SimulatedExchange : public IExecutionGateway {
    struct LevelEntry { double price; double size; };
    struct BookSide {
        std::map<double, double, std::greater<double>> bids;  // descending
        std::map<double, double>                       asks;  // ascending
    };

    mutable std::mutex mutex_;
    std::unordered_map<uint32_t, BookSide>          books_;
    std::unordered_map<uint64_t, Order>             orders_;
    std::unordered_map<uint32_t, double>            last_trade_px_;
    std::unordered_map<uint32_t, double>            daily_volume_;
    std::atomic<uint64_t>                           exec_id_counter_{1};

    ExecCallback exec_callback_;

    // Slippage parameters
    double spread_bps_   = 2.0;
    double impact_coeff_ = 0.1;
    double maker_fee_bps_= 1.0;
    double taker_fee_bps_= 3.0;

    static uint64_t now_ns() {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    }

    // Simulate a fill
    void fill_order(Order& order, double fill_px, double fill_qty, bool is_maker) {
        double commission = fill_px * fill_qty *
            (is_maker ? maker_fee_bps_ : taker_fee_bps_) / 1e4;

        double prev_notional = order.avg_px * order.filled_qty;
        order.filled_qty    += fill_qty;
        order.avg_px         = (order.filled_qty > kEps) ?
            (prev_notional + fill_px * fill_qty) / order.filled_qty : 0.0;
        order.status         = (order.remaining() < kEps) ?
            OrdStatus::Filled : OrdStatus::PartiallyFilled;
        order.update_ts_ns   = now_ns();

        last_trade_px_[order.asset_id] = fill_px;
        daily_volume_[order.asset_id] += fill_qty;

        ExecutionReport rpt{};
        rpt.exec_id    = exec_id_counter_.fetch_add(1);
        rpt.order_id   = order.order_id;
        rpt.asset_id   = order.asset_id;
        rpt.ord_status = order.status;
        rpt.last_qty   = fill_qty;
        rpt.last_px    = fill_px;
        rpt.cum_qty    = order.filled_qty;
        rpt.avg_px     = order.avg_px;
        rpt.leaves_qty = order.remaining();
        rpt.commission = commission;
        rpt.timestamp_ns = order.update_ts_ns;
        rpt.is_maker   = is_maker;
        rpt.exec_type  = 'F';

        if (exec_callback_) exec_callback_(rpt);
    }

    double simulate_fill_price(const Order& order, uint32_t asset_id) {
        auto it = books_.find(asset_id);
        if (it == books_.end()) {
            auto lt = last_trade_px_.find(asset_id);
            return (lt != last_trade_px_.end()) ? lt->second : 100.0;
        }
        auto& book = it->second;
        if (order.is_buy()) {
            double best_ask = book.asks.empty() ? 100.0 : book.asks.begin()->first;
            double impact   = best_ask * impact_coeff_ *
                              std::sqrt(order.quantity * best_ask / 1e6);
            return best_ask + impact;
        } else {
            double best_bid = book.bids.empty() ? 99.0 : book.bids.begin()->first;
            double impact   = best_bid * impact_coeff_ *
                              std::sqrt(order.quantity * best_bid / 1e6);
            return best_bid - impact;
        }
    }

    bool try_match_limit(Order& order) {
        auto it = books_.find(order.asset_id);
        if (it == books_.end()) return false;
        auto& book = it->second;

        if (order.is_buy() && !book.asks.empty()) {
            auto best_ask = book.asks.begin();
            if (best_ask->first <= order.price) {
                double fill_qty = std::min(order.remaining(), best_ask->second);
                fill_order(order, best_ask->first, fill_qty, true);
                best_ask->second -= fill_qty;
                if (best_ask->second < kEps) book.asks.erase(best_ask);
                return true;
            }
        } else if (!order.is_buy() && !book.bids.empty()) {
            auto best_bid = book.bids.begin();
            if (best_bid->first >= order.price) {
                double fill_qty = std::min(order.remaining(), best_bid->second);
                fill_order(order, best_bid->first, fill_qty, true);
                best_bid->second -= fill_qty;
                if (best_bid->second < kEps) book.bids.erase(best_bid);
                return true;
            }
        }
        return false;
    }

public:
    SimulatedExchange() = default;

    void set_spread(double bps) { spread_bps_ = bps; }
    void set_impact_coeff(double c) { impact_coeff_ = c; }

    // Seed the order book for asset
    void seed_book(uint32_t asset_id, double mid_price,
                   int levels = 10, double level_size = 1000.0)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto& book = books_[asset_id];
        book.bids.clear();
        book.asks.clear();
        double tick = mid_price * 0.0001;
        for (int i = 1; i <= levels; ++i) {
            book.bids[mid_price - i*tick] = level_size * (1.0 + 0.1*i);
            book.asks[mid_price + i*tick] = level_size * (1.0 + 0.1*i);
        }
        last_trade_px_[asset_id] = mid_price;
    }

    uint64_t submit_order(Order& order) override {
        std::lock_guard<std::mutex> lk(mutex_);
        order.order_id     = g_order_id_counter.fetch_add(1);
        order.submit_ts_ns = now_ns();
        order.update_ts_ns = order.submit_ts_ns;
        order.status       = OrdStatus::New;

        if (order.ord_type == OrdType::Market) {
            double fill_px = simulate_fill_price(order, order.asset_id);
            fill_order(order, fill_px, order.quantity, false);
        } else if (order.ord_type == OrdType::Limit) {
            // Try immediate match first
            if (!try_match_limit(order) && order.tif != TimeInForce::ImmediateOrCancel) {
                // Queue in book
                auto& book = books_[order.asset_id];
                if (order.is_buy()) book.bids[order.price] += order.remaining();
                else                book.asks[order.price] += order.remaining();
            } else if (order.tif == TimeInForce::ImmediateOrCancel && order.status == OrdStatus::New) {
                order.status = OrdStatus::Cancelled;
            }
        }

        orders_[order.order_id] = order;
        return order.order_id;
    }

    bool cancel_order(uint64_t order_id) override {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = orders_.find(order_id);
        if (it == orders_.end() || !it->second.is_active()) return false;

        Order& order = it->second;
        // Remove from book
        auto bit = books_.find(order.asset_id);
        if (bit != books_.end()) {
            if (order.is_buy()) bit->second.bids.erase(order.price);
            else                bit->second.asks.erase(order.price);
        }
        order.status       = OrdStatus::Cancelled;
        order.update_ts_ns = now_ns();

        ExecutionReport rpt{};
        rpt.exec_id    = exec_id_counter_.fetch_add(1);
        rpt.order_id   = order_id;
        rpt.asset_id   = order.asset_id;
        rpt.ord_status = OrdStatus::Cancelled;
        rpt.leaves_qty = 0.0;
        rpt.cum_qty    = order.filled_qty;
        rpt.avg_px     = order.avg_px;
        rpt.exec_type  = 'C';
        rpt.timestamp_ns = order.update_ts_ns;
        if (exec_callback_) exec_callback_(rpt);

        return true;
    }

    bool modify_order(uint64_t order_id, double new_qty, double new_price) override {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = orders_.find(order_id);
        if (it == orders_.end() || !it->second.is_active()) return false;
        // Cancel and resubmit (simplified)
        cancel_order(order_id);
        Order new_order = it->second;
        new_order.quantity = new_qty;
        new_order.price    = new_price;
        new_order.filled_qty = 0.0;
        new_order.avg_px     = 0.0;
        new_order.status     = OrdStatus::New;
        submit_order(new_order);
        return true;
    }

    std::optional<Order> get_order(uint64_t order_id) const override {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = orders_.find(order_id);
        if (it == orders_.end()) return std::nullopt;
        return it->second;
    }

    void set_exec_callback(ExecCallback cb) override {
        exec_callback_ = std::move(cb);
    }

    L2Snapshot get_market_data(uint32_t asset_id) const override {
        std::lock_guard<std::mutex> lk(mutex_);
        L2Snapshot snap{};
        snap.asset_id = asset_id;
        snap.timestamp_ns = now_ns();

        auto it = books_.find(asset_id);
        if (it != books_.end()) {
            int bi = 0;
            for (auto& [px, sz] : it->second.bids) {
                if (bi >= 10) break;
                snap.bids[bi][0] = px; snap.bids[bi][1] = sz; ++bi;
            }
            snap.n_bid_levels = bi;
            int ai = 0;
            for (auto& [px, sz] : it->second.asks) {
                if (ai >= 10) break;
                snap.asks[ai][0] = px; snap.asks[ai][1] = sz; ++ai;
            }
            snap.n_ask_levels = ai;
        }
        auto lt = last_trade_px_.find(asset_id);
        snap.last_trade_price = (lt != last_trade_px_.end()) ? lt->second : 0.0;
        auto dv = daily_volume_.find(asset_id);
        snap.daily_volume = (dv != daily_volume_.end()) ? dv->second : 0.0;
        return snap;
    }

    // Simulate market tick (update book with new mid price)
    void tick_market(uint32_t asset_id, double mid_price,
                     double shock_pct = 0.0)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto& book = books_[asset_id];
        // Clear and re-seed (simplified re-center)
        double new_mid = mid_price * (1.0 + shock_pct);
        book.bids.clear();
        book.asks.clear();
        double tick = new_mid * 0.0001;
        for (int i = 1; i <= 10; ++i) {
            book.bids[new_mid - i*tick] = 1000.0 + 50.0*i;
            book.asks[new_mid + i*tick] = 1000.0 + 50.0*i;
        }
        last_trade_px_[asset_id] = new_mid;
    }

    // Match all resting limit orders against current book
    void match_resting_orders() {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto& [oid, order] : orders_) {
            if (!order.is_active() || order.ord_type != OrdType::Limit) continue;
            try_match_limit(order);
        }
    }

    int n_active_orders() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return (int)std::count_if(orders_.begin(), orders_.end(),
            [](const auto& p) { return p.second.is_active(); });
    }
};

// ---------------------------------------------------------------------------
// Gateway metrics
// ---------------------------------------------------------------------------
struct GatewayMetrics {
    std::atomic<uint64_t> orders_sent    {0};
    std::atomic<uint64_t> orders_filled  {0};
    std::atomic<uint64_t> orders_rejected{0};
    std::atomic<uint64_t> orders_cancelled{0};
    std::atomic<uint64_t> n_fills        {0};
    std::atomic<uint64_t> total_latency_ns{0};

    void record_order() { orders_sent.fetch_add(1, std::memory_order_relaxed); }
    void record_fill(uint64_t latency_ns) {
        orders_filled.fetch_add(1, std::memory_order_relaxed);
        n_fills.fetch_add(1, std::memory_order_relaxed);
        total_latency_ns.fetch_add(latency_ns, std::memory_order_relaxed);
    }

    double fill_rate() const {
        uint64_t sent = orders_sent.load();
        return (sent > 0) ? (double)orders_filled.load() / sent : 0.0;
    }

    double mean_latency_us() const {
        uint64_t n = n_fills.load();
        return (n > 0) ? (double)total_latency_ns.load() / n / 1000.0 : 0.0;
    }

    std::string prometheus() const {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "rtel_gw_orders_sent %lu\n"
            "rtel_gw_orders_filled %lu\n"
            "rtel_gw_orders_cancelled %lu\n"
            "rtel_gw_fill_rate %.4f\n"
            "rtel_gw_mean_latency_us %.2f\n",
            orders_sent.load(), orders_filled.load(), orders_cancelled.load(),
            fill_rate(), mean_latency_us());
        return buf;
    }
};

// ---------------------------------------------------------------------------
// Strategy execution context
// ---------------------------------------------------------------------------
class StrategyContext {
    SimulatedExchange& exchange_;
    GatewayMetrics     metrics_;
    uint32_t           strategy_id_;
    std::unordered_map<uint32_t, double> positions_;  // asset → quantity
    std::unordered_map<uint32_t, double> avg_costs_;
    double             realized_pnl_ = 0.0;
    double             commission_   = 0.0;

public:
    StrategyContext(SimulatedExchange& ex, uint32_t sid)
        : exchange_(ex), strategy_id_(sid)
    {
        exchange_.set_exec_callback([this](const ExecutionReport& rpt) {
            on_exec_report(rpt);
        });
    }

    void on_exec_report(const ExecutionReport& rpt) {
        if (rpt.exec_type == 'F') {
            auto oid = rpt.order_id;
            auto opt = exchange_.get_order(oid);
            if (!opt) return;
            const Order& ord = *opt;

            double signed_qty = ord.is_buy() ? rpt.last_qty : -rpt.last_qty;
            double cur_qty    = positions_[ord.asset_id];
            double cur_cost   = avg_costs_[ord.asset_id];

            if (cur_qty == 0.0) {
                positions_[ord.asset_id] = signed_qty;
                avg_costs_[ord.asset_id] = rpt.last_px;
            } else if (cur_qty * signed_qty > 0) {
                // Same direction, update avg cost
                double new_qty = cur_qty + signed_qty;
                avg_costs_[ord.asset_id] = (cur_cost * std::abs(cur_qty) +
                    rpt.last_px * rpt.last_qty) / std::abs(new_qty);
                positions_[ord.asset_id] = new_qty;
            } else {
                // Reducing/flipping
                double close_qty = std::min(std::abs(cur_qty), std::abs(signed_qty));
                double sign = (cur_qty > 0) ? 1.0 : -1.0;
                realized_pnl_ += sign * close_qty * (rpt.last_px - cur_cost);
                positions_[ord.asset_id] = cur_qty + signed_qty;
                if (std::abs(positions_[ord.asset_id]) < kEps) {
                    avg_costs_[ord.asset_id] = 0.0;
                } else if ((cur_qty + signed_qty) * cur_qty < 0) {
                    avg_costs_[ord.asset_id] = rpt.last_px;
                }
            }
            commission_ += rpt.commission;
            metrics_.record_fill(0);
        }
    }

    uint64_t buy_market(uint32_t asset_id, double qty) {
        Order o{};
        o.asset_id = asset_id;
        o.side     = Side::Buy;
        o.ord_type = OrdType::Market;
        o.quantity = qty;
        metrics_.record_order();
        return exchange_.submit_order(o);
    }

    uint64_t sell_market(uint32_t asset_id, double qty) {
        Order o{};
        o.asset_id = asset_id;
        o.side     = Side::Sell;
        o.ord_type = OrdType::Market;
        o.quantity = qty;
        metrics_.record_order();
        return exchange_.submit_order(o);
    }

    uint64_t buy_limit(uint32_t asset_id, double qty, double price) {
        Order o{};
        o.asset_id = asset_id;
        o.side     = Side::Buy;
        o.ord_type = OrdType::Limit;
        o.quantity = qty;
        o.price    = price;
        metrics_.record_order();
        return exchange_.submit_order(o);
    }

    uint64_t sell_limit(uint32_t asset_id, double qty, double price) {
        Order o{};
        o.asset_id = asset_id;
        o.side     = Side::Sell;
        o.ord_type = OrdType::Limit;
        o.quantity = qty;
        o.price    = price;
        metrics_.record_order();
        return exchange_.submit_order(o);
    }

    double position(uint32_t asset_id) const {
        auto it = positions_.find(asset_id);
        return (it != positions_.end()) ? it->second : 0.0;
    }

    double unrealized_pnl(const std::unordered_map<uint32_t, double>& prices) const {
        double upnl = 0.0;
        for (auto& [aid, qty] : positions_) {
            auto pit = prices.find(aid);
            if (pit == prices.end()) continue;
            double sign = (qty > 0) ? 1.0 : -1.0;
            double cost = avg_costs_.count(aid) ? avg_costs_.at(aid) : 0.0;
            upnl += sign * std::abs(qty) * (pit->second - cost);
        }
        return upnl;
    }

    double total_pnl(const std::unordered_map<uint32_t, double>& prices) const {
        return realized_pnl_ + unrealized_pnl(prices) - commission_;
    }

    double realized_pnl() const { return realized_pnl_; }
    double commission()   const { return commission_; }

    const GatewayMetrics& metrics() const { return metrics_; }
};

}  // namespace gateway
}  // namespace rtel
