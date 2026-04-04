#pragma once
// Event-driven backtester: replays market data through order book
// and strategy, tracking PnL, positions, and execution quality.

#include "orderbook.hpp"
#include "order_pool.hpp"
#include "risk_checks.hpp"
#include "strategy_engine.hpp"
#include "feed_handler.hpp"
#include <vector>
#include <deque>
#include <string>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <chrono>

namespace hft {

// ============================================================
// Market event for backtesting
// ============================================================
enum class EventType : uint8_t {
    NewOrder     = 0,
    CancelOrder  = 1,
    ModifyOrder  = 2,
    Trade        = 3,
    Quote        = 4,
    Heartbeat    = 5,
    End          = 255
};

struct BacktestEvent {
    EventType  type;
    Timestamp  timestamp;
    OrderId    order_id;
    Side       side;
    Price      price;
    Quantity   qty;
    Price      bid_price;
    Quantity   bid_qty;
    Price      ask_price;
    Quantity   ask_qty;
    char       symbol[16];
};

// ============================================================
// Trade record for PnL computation
// ============================================================
struct BacktestTrade {
    Timestamp  timestamp;
    OrderId    aggressor;
    OrderId    passive;
    Side       side;
    Price      price;
    Quantity   qty;
    double     pnl;         // realized PnL from this trade
    double     cost;        // transaction cost (slippage + fees)
    double     mid_at_fill; // mid-price at time of fill
    double     effective_spread; // 2 * |fill_px - mid|
};

// ============================================================
// PnL tracker
// ============================================================
struct PnLState {
    double realized        = 0.0;
    double unrealized      = 0.0;
    double transaction_cost = 0.0;
    double gross_pnl       = 0.0;
    double net_pnl         = 0.0;
    int64_t net_position   = 0;
    double avg_cost        = 0.0;

    void on_fill(Side side, Quantity qty, Price price,
                 double mid_px, double fee_per_share = 0.002)
    {
        double p = price_to_double(price);
        double fill_val = p * qty;
        double fee = qty * fee_per_share;
        transaction_cost += fee;

        if (side == Side::Buy) {
            double total = avg_cost * std::max(int64_t(0), net_position) + fill_val;
            net_position += static_cast<int64_t>(qty);
            avg_cost = net_position > 0 ? total / net_position : 0.0;
        } else {
            if (net_position > 0) {
                int64_t close = std::min(static_cast<int64_t>(qty), net_position);
                realized += (p - avg_cost) * close;
            } else if (net_position < 0) {
                // Adding to short
            }
            net_position -= static_cast<int64_t>(qty);
            if (net_position < 0 && avg_cost == 0.0)
                avg_cost = p; // short entry
        }

        // Effective spread
        double eff_spread = 2.0 * std::fabs(p - mid_px);
        gross_pnl = realized;
        net_pnl   = realized - transaction_cost;
    }

    void mark_to_market(double mkt_price) {
        if (net_position != 0)
            unrealized = (mkt_price - avg_cost) * net_position;
        net_pnl = realized + unrealized - transaction_cost;
    }

    double total_pnl() const { return realized + unrealized; }
};

// ============================================================
// Backtest result
// ============================================================
struct BacktestResult {
    // Returns series (daily or per-trade)
    std::vector<double> returns;
    std::vector<double> equity_curve;

    // Summary stats
    double total_return;
    double annualized_return;
    double annualized_vol;
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double calmar_ratio;
    double win_rate;
    double profit_factor;
    double total_pnl;
    double realized_pnl;
    double transaction_cost;

    // Execution stats
    uint64_t orders_submitted;
    uint64_t orders_filled;
    uint64_t orders_cancelled;
    uint64_t orders_rejected;
    uint64_t trades_count;
    double   avg_fill_rate;
    double   avg_slippage_bps;

    // Market stats
    uint64_t events_processed;
    double   elapsed_sim_time_ms;

    void print() const {
        auto sep = std::string(50, '-');
        std::cout << "\n=== Backtest Results ===" << std::endl;
        std::cout << sep << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Total Return:      " << total_return * 100 << "%" << std::endl;
        std::cout << "  Annual Return:     " << annualized_return * 100 << "%" << std::endl;
        std::cout << "  Annual Vol:        " << annualized_vol * 100 << "%" << std::endl;
        std::cout << "  Sharpe Ratio:      " << sharpe_ratio << std::endl;
        std::cout << "  Sortino Ratio:     " << sortino_ratio << std::endl;
        std::cout << "  Max Drawdown:      " << max_drawdown * 100 << "%" << std::endl;
        std::cout << "  Calmar Ratio:      " << calmar_ratio << std::endl;
        std::cout << "  Win Rate:          " << win_rate * 100 << "%" << std::endl;
        std::cout << "  Profit Factor:     " << profit_factor << std::endl;
        std::cout << "  Net PnL:           $" << total_pnl << std::endl;
        std::cout << "  Transaction Cost:  $" << transaction_cost << std::endl;
        std::cout << sep << std::endl;
        std::cout << "  Orders Submitted:  " << orders_submitted << std::endl;
        std::cout << "  Orders Filled:     " << orders_filled << std::endl;
        std::cout << "  Trades:            " << trades_count << std::endl;
        std::cout << "  Fill Rate:         " << avg_fill_rate * 100 << "%" << std::endl;
        std::cout << "  Avg Slippage:      " << avg_slippage_bps << " bps" << std::endl;
        std::cout << "  Events Processed:  " << events_processed << std::endl;
        std::cout << "  Sim Time:          " << elapsed_sim_time_ms << " ms" << std::endl;
    }
};

// ============================================================
// Synthetic event generator for backtesting
// ============================================================
class SyntheticEventGenerator {
public:
    struct Config {
        std::string symbol = "TEST";
        double initial_price = 100.0;
        double daily_vol     = 0.20;
        double lambda_new    = 500.0;   // orders/sec
        double lambda_cancel = 400.0;
        double lambda_trade  = 100.0;
        double tick_size     = 0.01;
        double spread_bps    = 10.0;
        size_t num_events    = 100000;
        uint64_t seed        = 42;
    };

    explicit SyntheticEventGenerator(const Config& cfg) : cfg_(cfg) {}

    std::vector<BacktestEvent> generate() {
        std::vector<BacktestEvent> events;
        events.reserve(cfg_.num_events);

        uint64_t rng = cfg_.seed;
        auto lcg = [&]() -> uint64_t {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            return rng;
        };
        auto uniform = [&]() -> double {
            return static_cast<double>(lcg() >> 11) / static_cast<double>(1ULL << 53);
        };
        auto randn = [&]() -> double {
            double u1 = std::max(uniform(), 1e-10);
            double u2 = uniform();
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        };
        auto exp_sample = [&](double rate) -> double {
            return -std::log(std::max(uniform(), 1e-10)) / rate;
        };

        double price = cfg_.initial_price;
        Timestamp ts = 34200LL * 1000000000LL; // 9:30 AM
        double dt_sec = 1.0 / (cfg_.lambda_new + cfg_.lambda_cancel + cfg_.lambda_trade);
        double vol_per_dt = cfg_.daily_vol * std::sqrt(dt_sec / (252.0 * 6.5 * 3600.0));
        OrderId oid = 1;

        double next_new    = exp_sample(cfg_.lambda_new);
        double next_cancel = exp_sample(cfg_.lambda_cancel);
        double next_trade  = exp_sample(cfg_.lambda_trade);
        double t_sec = 0.0;

        std::vector<OrderId> active_orders;
        active_orders.reserve(10000);

        while (events.size() < cfg_.num_events) {
            // GBM price update
            price *= std::exp(-0.5 * vol_per_dt * vol_per_dt + vol_per_dt * randn());
            price = std::round(price / cfg_.tick_size) * cfg_.tick_size;
            price = std::max(price, cfg_.tick_size);
            t_sec += dt_sec;
            ts += static_cast<Timestamp>(dt_sec * 1e9);

            double half_spread = price * cfg_.spread_bps / 20000.0;
            double snap_bid = std::round((price - half_spread) / cfg_.tick_size) * cfg_.tick_size;
            double snap_ask = std::round((price + half_spread) / cfg_.tick_size) * cfg_.tick_size;

            if (t_sec >= next_new) {
                BacktestEvent ev{};
                ev.type      = EventType::NewOrder;
                ev.timestamp = ts;
                ev.order_id  = oid++;
                ev.side      = uniform() < 0.5 ? Side::Buy : Side::Sell;
                double offset = uniform() * 5.0 * cfg_.tick_size;
                if (ev.side == Side::Buy)
                    ev.price = double_to_price(snap_bid - offset);
                else
                    ev.price = double_to_price(snap_ask + offset);
                ev.qty = static_cast<Quantity>(std::round(1 + uniform() * 99) * 100);
                ev.bid_price = double_to_price(snap_bid);
                ev.ask_price = double_to_price(snap_ask);
                ev.bid_qty   = 1000;
                ev.ask_qty   = 1000;
                std::strncpy(ev.symbol, cfg_.symbol.c_str(), 15);
                events.push_back(ev);
                active_orders.push_back(ev.order_id);
                next_new = t_sec + exp_sample(cfg_.lambda_new);
            }

            if (t_sec >= next_cancel && !active_orders.empty()) {
                BacktestEvent ev{};
                ev.type     = EventType::CancelOrder;
                ev.timestamp = ts;
                size_t idx  = static_cast<size_t>(uniform() * active_orders.size());
                ev.order_id = active_orders[idx];
                active_orders.erase(active_orders.begin() + idx);
                std::strncpy(ev.symbol, cfg_.symbol.c_str(), 15);
                events.push_back(ev);
                next_cancel = t_sec + exp_sample(cfg_.lambda_cancel);
            }

            if (t_sec >= next_trade) {
                BacktestEvent ev{};
                ev.type      = EventType::Trade;
                ev.timestamp = ts;
                ev.order_id  = oid++;
                ev.side      = uniform() < 0.5 ? Side::Buy : Side::Sell;
                ev.price     = ev.side == Side::Buy ? double_to_price(snap_ask) : double_to_price(snap_bid);
                ev.qty       = static_cast<Quantity>(std::round(1 + uniform() * 9) * 100);
                ev.bid_price = double_to_price(snap_bid);
                ev.ask_price = double_to_price(snap_ask);
                std::strncpy(ev.symbol, cfg_.symbol.c_str(), 15);
                events.push_back(ev);
                next_trade = t_sec + exp_sample(cfg_.lambda_trade);
            }
        }

        // Sort by timestamp
        std::stable_sort(events.begin(), events.end(),
            [](const BacktestEvent& a, const BacktestEvent& b){ return a.timestamp < b.timestamp; });

        return events;
    }

private:
    Config cfg_;
};

// ============================================================
// Main Backtester
// ============================================================
class Backtester {
public:
    struct Config {
        double fee_per_share    = 0.002;
        double slippage_bps     = 0.5;
        double initial_capital  = 1000000.0;
        bool   verbose          = false;
        size_t snapshot_interval = 1000; // events between equity snapshots
    };

    Backtester(const std::string& symbol, const Config& cfg = {})
        : symbol_(symbol), cfg_(cfg),
          book_(symbol), risk_(), pool_()
    {}

    BacktestResult run(const std::vector<BacktestEvent>& events,
                       Strategy& strategy)
    {
        auto wall_t0 = std::chrono::high_resolution_clock::now();

        BacktestResult result{};
        PnLState pnl{};
        MarketImpactModel impact;
        double equity = cfg_.initial_capital;

        size_t event_idx = 0;
        for (const auto& ev : events) {
            ++event_idx;
            process_event(ev, strategy, pnl, impact, result);

            // Equity snapshot
            if (event_idx % cfg_.snapshot_interval == 0) {
                auto mid_opt = book_.mid_price();
                double mid   = mid_opt.value_or(0.0);
                if (mid > 0) pnl.mark_to_market(mid);
                result.equity_curve.push_back(cfg_.initial_capital + pnl.net_pnl);
            }
        }

        // Final snapshot
        auto mid_opt = book_.mid_price();
        if (mid_opt) pnl.mark_to_market(*mid_opt);
        result.equity_curve.push_back(cfg_.initial_capital + pnl.net_pnl);

        // Compute performance stats
        auto wall_t1 = std::chrono::high_resolution_clock::now();
        result.elapsed_sim_time_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
        result.events_processed    = events.size();
        result.total_pnl           = pnl.net_pnl;
        result.realized_pnl        = pnl.realized;
        result.transaction_cost    = pnl.transaction_cost;
        result.orders_submitted    = orders_submitted_;
        result.orders_filled       = orders_filled_;
        result.orders_cancelled    = orders_cancelled_;
        result.orders_rejected     = orders_rejected_;
        result.trades_count        = trades_count_;

        if (orders_submitted_ > 0)
            result.avg_fill_rate = static_cast<double>(orders_filled_) / orders_submitted_;

        // Returns from equity curve
        result.returns.reserve(result.equity_curve.size());
        for (size_t i = 1; i < result.equity_curve.size(); ++i) {
            double r = (result.equity_curve[i] - result.equity_curve[i-1]) /
                       result.equity_curve[i-1];
            result.returns.push_back(r);
        }
        compute_stats(result);

        return result;
    }

private:
    std::string symbol_;
    Config      cfg_;
    OrderBook   book_;
    RiskManager risk_;
    OrderPool<> pool_;

    uint64_t orders_submitted_ = 0;
    uint64_t orders_filled_    = 0;
    uint64_t orders_cancelled_ = 0;
    uint64_t orders_rejected_  = 0;
    uint64_t trades_count_     = 0;
    double   total_slippage_bps_ = 0.0;

    void process_event(const BacktestEvent& ev, Strategy& strategy,
                       PnLState& pnl, MarketImpactModel& impact,
                       BacktestResult& result)
    {
        switch (ev.type) {
        case EventType::NewOrder: {
            // Add to book
            Order* o = pool_.alloc(ev.order_id, ev.symbol, ev.side,
                                    OrderType::Limit, TimeInForce::GTC,
                                    ev.price, ev.qty, ev.timestamp);
            if (o) book_.add_order(o);
            break;
        }
        case EventType::CancelOrder:
            book_.cancel_order(ev.order_id);
            ++orders_cancelled_;
            break;
        case EventType::Trade: {
            double mid = book_.mid_price().value_or(0.0);
            // Update impact model
            impact.on_trade(mid - 0.001, mid,
                ev.side == Side::Buy ? ev.qty : -(int64_t)ev.qty,
                price_to_double(ev.price) * ev.qty);

            // Run strategy on book update
            auto features = BookFeatures::from_book(book_, impact);
            features.mid_price = mid;
            features.timestamp = ev.timestamp;

            Strategy::Context ctx{features, book_, risk_, ev.timestamp};
            auto signals = strategy.on_book_update(ctx);

            for (const auto& sig : signals) {
                submit_strategy_order(sig, pnl, features.mid_price, impact, result);
            }
            ++trades_count_;
            break;
        }
        default:
            break;
        }
    }

    void submit_strategy_order(const Signal& sig, PnLState& pnl,
                                double mid_price, MarketImpactModel& /*impact*/,
                                BacktestResult& /*result*/)
    {
        if (sig.type == SignalType::None || sig.type == SignalType::Cancel) return;

        ++orders_submitted_;
        auto rej = risk_.check_order(
            Order(++oid_gen_, sig.symbol, sig.type == SignalType::BuyLimit ? Side::Buy : Side::Sell,
                  OrderType::Limit, sig.tif, sig.price, sig.qty, sig.signal_ts));
        if (rej != RejectionReason::None) { ++orders_rejected_; return; }

        // Simulate fill: assume immediate fill at limit price + slippage
        double fill_px = price_to_double(sig.price);
        double slippage = mid_price * cfg_.slippage_bps / 10000.0;
        if (sig.type == SignalType::BuyLimit) fill_px += slippage;
        else                                  fill_px -= slippage;

        Price fp = double_to_price(std::max(fill_px, 0.01));
        pnl.on_fill(sig.type == SignalType::BuyLimit ? Side::Buy : Side::Sell,
                    sig.qty, fp, mid_price, cfg_.fee_per_share);
        risk_.on_fill(sig.symbol, sig.type == SignalType::BuyLimit ? Side::Buy : Side::Sell,
                      sig.qty, fp);

        total_slippage_bps_ += std::fabs(fill_px - price_to_double(sig.price)) /
                               (mid_price > 0 ? mid_price : 1.0) * 10000.0;
        ++orders_filled_;
    }

    void compute_stats(BacktestResult& r) const {
        if (r.returns.empty()) return;
        const size_t n = r.returns.size();
        double mean = std::accumulate(r.returns.begin(), r.returns.end(), 0.0) / n;

        double var = 0.0, down_var = 0.0;
        for (auto ret : r.returns) {
            var += (ret - mean) * (ret - mean);
            double d = std::min(ret, 0.0);
            down_var += d * d;
        }
        var      /= (n > 1 ? n-1 : 1);
        down_var /= (n > 1 ? n-1 : 1);

        double ann = 252.0 * 6.5 * 3600.0 / cfg_.snapshot_interval;
        r.annualized_vol    = std::sqrt(var * ann);
        r.annualized_return = (1.0 + mean) * ann - 1.0;
        r.total_return      = (r.equity_curve.empty() ? 0.0 :
            (r.equity_curve.back() - cfg_.initial_capital) / cfg_.initial_capital);

        double ann_down = std::sqrt(down_var * ann);
        r.sharpe_ratio  = r.annualized_vol  > 0 ? r.annualized_return / r.annualized_vol  : 0.0;
        r.sortino_ratio = ann_down > 0 ? r.annualized_return / ann_down : 0.0;

        // Max drawdown
        double peak = cfg_.initial_capital;
        double mdd  = 0.0;
        for (auto e : r.equity_curve) {
            if (e > peak) peak = e;
            double dd = (peak - e) / peak;
            if (dd > mdd) mdd = dd;
        }
        r.max_drawdown  = mdd;
        r.calmar_ratio  = mdd > 0 ? r.annualized_return / mdd : 0.0;

        // Win/loss
        size_t wins = 0, losses = 0;
        double gw = 0.0, gl = 0.0;
        for (auto ret : r.returns) {
            if (ret > 0) { wins++; gw += ret; }
            else if (ret < 0) { losses++; gl += std::fabs(ret); }
        }
        r.win_rate      = n > 0 ? static_cast<double>(wins) / n : 0.0;
        r.profit_factor = gl > 0 ? gw / gl : 0.0;
        r.avg_slippage_bps = orders_filled_ > 0 ? total_slippage_bps_ / orders_filled_ : 0.0;
    }

    OrderId oid_gen_ = 9000000;
};

} // namespace hft
