// Full integration simulation demonstrating all HFT primitives working together.
// Simulates a complete trading day: feed handler → matching engine → strategies → risk → PnL.

#include "orderbook.hpp"
#include "feed_handler.hpp"
#include "matching_engine.hpp"
#include "strategy_engine.hpp"
#include "backtester.hpp"
#include "order_flow.hpp"
#include "risk_checks.hpp"
#include "market_impact.hpp"
#include "price_ladder.hpp"
#include "order_pool.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>
#include <unordered_map>

using namespace hft;

// ============================================================
// Utilities
// ============================================================
static std::string fmt_price(Price p) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4) << price_to_double(p);
    return ss.str();
}

static std::string fmt_double(double d, int prec = 4) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << d;
    return ss.str();
}

static std::string bar(double frac, int width = 20) {
    int filled = static_cast<int>(frac * width);
    filled = std::max(0, std::min(width, filled));
    std::string s(filled, '#');
    s += std::string(width - filled, '.');
    return "[" + s + "]";
}

// ============================================================
// Scenario: Single-Symbol Market Simulation
// Full pipeline: GBM price → order arrivals → matching → fills → PnL
// ============================================================
struct ScenarioConfig {
    std::string symbol       = "AAPL";
    double      initial_price = 182.50;
    double      daily_vol    = 0.20;       // annualized
    double      tick_size    = 0.01;
    int         n_steps      = 50000;
    int         seed         = 12345;
    double      fee_rate     = 0.0001;     // 1 bps
    double      lambda_order = 100.0;      // orders/sec
    double      lambda_cancel = 70.0;
    double      lambda_trade  = 30.0;
    double      dt            = 1.0 / 252 / 6.5 / 3600; // 1 second in trading years
};

struct ScenarioResult {
    double realized_pnl;
    double unrealized_pnl;
    double total_fees;
    int    total_trades;
    int    total_orders;
    int    total_cancels;
    double final_price;
    double price_return_pct;
    double realized_vol;
    double max_drawdown;
    double sharpe_ratio;
    double vpin_final;
    std::vector<double> equity_curve;
    std::vector<double> price_series;
};

class MarketSimulation {
public:
    explicit MarketSimulation(const ScenarioConfig& cfg)
        : cfg_(cfg), rng_(cfg.seed), book_(nullptr, cfg.symbol.c_str())
    {
        dist_norm_ = std::normal_distribution<double>(0, 1);
        dist_unif_ = std::uniform_real_distribution<double>(0, 1);
        dist_qty_  = std::uniform_int_distribution<int>(50, 500);
    }

    ScenarioResult run() {
        ScenarioResult res{};
        double price = cfg_.initial_price;
        double dt    = cfg_.dt;
        double sigma = cfg_.daily_vol / std::sqrt(252.0);
        double dt_tick = dt;

        std::vector<double>& equity = res.equity_curve;
        std::vector<double>& prices = res.price_series;
        equity.reserve(cfg_.n_steps / 100);
        prices.reserve(cfg_.n_steps / 100);

        PnLState pnl{};
        double cum_pnl = 0.0;
        double peak_pnl = 0.0;
        double max_dd = 0.0;
        int64_t position = 0;
        double avg_cost = 0.0;

        VPINEstimator vpin;
        ToxicityMonitor toxmon(cfg_.symbol);
        RiskLimits limits{};
        limits.max_position = 10000;
        limits.max_daily_loss = 50000.0;
        limits.max_order_rate = 1000;
        RiskManager risk(limits, double_to_price(price));

        TickRuleClassifier tick_clf;
        OrderFlowImbalance ofi(200);

        int step_fill = 0;
        double fill_px_last = 0;

        std::vector<double> log_returns;
        log_returns.reserve(cfg_.n_steps);

        int orders_placed = 0;
        int orders_cancelled = 0;
        int orders_filled = 0;

        for (int step = 0; step < cfg_.n_steps; ++step) {
            // GBM price step
            double z = dist_norm_(rng_);
            double ret = -0.5 * sigma * sigma * dt_tick + sigma * std::sqrt(dt_tick) * z;
            double prev_price = price;
            price = price * std::exp(ret);
            price = std::max(price, 0.01);
            log_returns.push_back(ret);

            if (step > 0)
                res.price_series.push_back(price);

            // Spread around mid
            double half_spread = cfg_.tick_size * 2;
            double bid_d = price - half_spread;
            double ask_d = price + half_spread;
            Price  bid_p = double_to_price(bid_d);
            Price  ask_p = double_to_price(ask_d);

            int bid_qty = 200 + dist_qty_(rng_);
            int ask_qty = 200 + dist_qty_(rng_);

            // Poisson order arrival: simplified as step probability
            double p_order  = std::min(1.0, cfg_.lambda_order  * dt_tick);
            double p_cancel = std::min(1.0, cfg_.lambda_cancel * dt_tick);
            double p_trade  = std::min(1.0, cfg_.lambda_trade  * dt_tick);

            // New passive order
            if (dist_unif_(rng_) < p_order) {
                bool is_buy = dist_unif_(rng_) < 0.5;
                double px_d = is_buy ? bid_d - cfg_.tick_size * dist_unif_(rng_) * 3
                                     : ask_d + cfg_.tick_size * dist_unif_(rng_) * 3;
                Price px_p = double_to_price(px_d);
                int qty = dist_qty_(rng_);

                Order o{};
                o.id         = ++order_id_;
                std::memcpy(o.symbol, cfg_.symbol.c_str(), std::min(cfg_.symbol.size(), (size_t)15));
                o.side       = is_buy ? Side::Buy : Side::Sell;
                o.order_type = OrderType::Limit;
                o.status     = OrderStatus::New;
                o.price      = px_p;
                o.qty        = qty;
                o.timestamp  = step * 1000000LL;
                o.tif        = TimeInForce::GTC;

                if (risk.check_new_order(o)) {
                    auto fills = book_.add_order(o);
                    fills.deinit();
                    active_orders_.insert(o.id);
                    orders_placed++;
                }
            }

            // Cancel a random active order
            if (dist_unif_(rng_) < p_cancel && !active_orders_.empty()) {
                auto it = active_orders_.begin();
                std::advance(it, dist_qty_(rng_) % active_orders_.size());
                OrderId cancel_id = *it;
                if (book_.cancel_order(cancel_id)) {
                    active_orders_.erase(it);
                    orders_cancelled++;
                }
            }

            // Market trade
            if (dist_unif_(rng_) < p_trade) {
                bool is_buy = dist_unif_(rng_) < 0.52; // slight buy imbalance
                Price trade_px = is_buy ? ask_p : bid_p;
                Quantity trade_qty = static_cast<Quantity>(100 + dist_qty_(rng_) % 200);

                // Classify trade
                auto init = tick_clf.classify(trade_px);
                vpin.update(trade_px, bid_p, ask_p, trade_qty);
                toxmon.on_trade(trade_px, bid_p, ask_p, trade_qty, step * 1000000LL);

                // OFI snapshot
                OrderFlowImbalance::BookSnapshot snap{bid_p, (Quantity)bid_qty, ask_p, (Quantity)ask_qty};
                if (step > 0) ofi.update(prev_snap_, snap);
                prev_snap_ = snap;
                _ = init;

                // Simulate MM fill: if we have passive quotes, get filled
                bool mm_long  = position > 0;
                bool mm_short = position < 0;
                bool fill_mm  = (is_buy  && mm_short) || (!is_buy && mm_long) ||
                                (dist_unif_(rng_) < 0.1);  // random new fill

                if (fill_mm && std::abs(position) < limits.max_position) {
                    Quantity fill_qty = std::min((Quantity)100, trade_qty);
                    double   fill_px  = price_to_double(trade_px);
                    bool     fill_buy = !is_buy; // we're passive counterparty
                    double   fee      = fill_qty * cfg_.fee_rate * fill_px;

                    // Update position
                    if (fill_buy) {
                        double total_cost = avg_cost * std::max(position, 0LL) + fill_px * fill_qty;
                        position += fill_qty;
                        if (position > 0) avg_cost = total_cost / position;
                    } else {
                        if (position > 0) {
                            double close_qty = std::min((double)fill_qty, (double)position);
                            pnl.realized += (fill_px - avg_cost) * close_qty;
                        }
                        position -= fill_qty;
                    }
                    pnl.transaction_cost += fee;
                    cum_pnl = pnl.realized;
                    fill_px_last = fill_px;
                    orders_filled++;

                    risk.on_fill(fill_buy ? Side::Buy : Side::Sell, fill_qty, trade_px);
                }
            }

            // Mark to market
            if (position != 0 && avg_cost > 0) {
                pnl.unrealized = (price - avg_cost) * position;
            }

            double net = pnl.realized + pnl.unrealized - pnl.transaction_cost;
            peak_pnl = std::max(peak_pnl, net);
            double dd = peak_pnl > 0 ? (peak_pnl - net) / peak_pnl : 0;
            max_dd = std::max(max_dd, dd);

            if (step % 100 == 0) {
                equity.push_back(net);
            }
        }

        // Compute realized vol from log returns
        double mean_ret = 0;
        for (auto r : log_returns) mean_ret += r;
        mean_ret /= log_returns.size();
        double var = 0;
        for (auto r : log_returns) var += (r - mean_ret) * (r - mean_ret);
        var /= log_returns.size();
        res.realized_vol = std::sqrt(var * 252.0 * 6.5 * 3600);

        // Sharpe from equity curve
        if (equity.size() > 2) {
            std::vector<double> ret_eq;
            for (size_t i = 1; i < equity.size(); ++i)
                ret_eq.push_back(equity[i] - equity[i-1]);
            double mean_eq = 0, var_eq = 0;
            for (auto r : ret_eq) mean_eq += r;
            mean_eq /= ret_eq.size();
            for (auto r : ret_eq) var_eq += (r - mean_eq) * (r - mean_eq);
            var_eq /= ret_eq.size();
            double std_eq = std::sqrt(var_eq);
            res.sharpe_ratio = std_eq > 0 ? mean_eq / std_eq * std::sqrt(252.0) : 0.0;
        }

        res.realized_pnl   = pnl.realized;
        res.unrealized_pnl = pnl.unrealized;
        res.total_fees      = pnl.transaction_cost;
        res.total_orders    = orders_placed;
        res.total_cancels   = orders_cancelled;
        res.total_trades    = orders_filled;
        res.final_price     = price;
        res.price_return_pct = (price / cfg_.initial_price - 1.0) * 100.0;
        res.max_drawdown    = max_dd;
        res.vpin_final      = vpin.vpin();
        return res;
    }

private:
    ScenarioConfig cfg_;
    std::mt19937   rng_;
    OrderBook      book_;
    std::normal_distribution<double>    dist_norm_;
    std::uniform_real_distribution<double> dist_unif_;
    std::uniform_int_distribution<int>  dist_qty_;
    std::unordered_set<OrderId>         active_orders_;
    OrderFlowImbalance::BookSnapshot    prev_snap_{};
    OrderId order_id_ = 1000000;
    using unordered_set_type = std::unordered_set<OrderId>;
    unordered_set_type dummy_; // silence unused warning
};

// ============================================================
// Multi-symbol portfolio simulation
// ============================================================
struct PortfolioConfig {
    std::vector<std::string> symbols  = {"AAPL","MSFT","GOOGL","AMZN","NVDA"};
    std::vector<double>      prices   = {182.5, 375.0, 141.0, 178.0, 495.0};
    std::vector<double>      vols     = {0.22, 0.24, 0.28, 0.26, 0.35};
    double                   corr     = 0.45;  // pairwise correlation
    int                      n_steps  = 20000;
    int                      seed     = 999;
};

struct PortfolioResult {
    std::vector<double> symbol_pnl;
    std::vector<double> symbol_vpin;
    double portfolio_pnl;
    double portfolio_sharpe;
    double total_volume;
    int    total_fills;
};

PortfolioResult run_portfolio_simulation(const PortfolioConfig& pcfg) {
    size_t N = pcfg.symbols.size();
    PortfolioResult result{};
    result.symbol_pnl.resize(N, 0.0);
    result.symbol_vpin.resize(N, 0.0);

    std::mt19937 rng(pcfg.seed);
    std::normal_distribution<double> norm(0, 1);

    // Cholesky-like correlation (simplified: L * L^T with given corr)
    // All off-diagonal = corr, diagonal = 1
    double L_diag = 1.0;
    double L_off  = pcfg.corr;
    double L_11   = 1.0;
    double L_21   = L_off;
    double L_22   = std::sqrt(1.0 - L_off * L_off);

    std::vector<VPINEstimator> vpins(N);
    std::vector<double> prices = pcfg.prices;

    double dt = 1.0 / (252.0 * 6.5 * 3600.0);
    std::vector<double> pnl(N, 0.0);
    std::vector<int64_t> positions(N, 0);
    std::vector<double> avg_costs(N, 0.0);
    int total_fills = 0;
    double total_volume = 0.0;
    std::vector<double> equity_history;
    equity_history.reserve(pcfg.n_steps / 50);

    for (int step = 0; step < pcfg.n_steps; ++step) {
        // Correlated GBM shocks
        std::vector<double> z(N);
        double common = norm(rng);
        for (size_t i = 0; i < N; ++i) {
            double idio = norm(rng);
            z[i] = L_off * common + std::sqrt(1 - L_off * L_off) * idio;
        }

        for (size_t i = 0; i < N; ++i) {
            double sigma_dt = pcfg.vols[i] / std::sqrt(252.0 * 6.5 * 3600.0);
            double ret = -0.5 * sigma_dt * sigma_dt + sigma_dt * z[i];
            prices[i] *= std::exp(ret);
            prices[i] = std::max(prices[i], 0.01);

            // Simple market-making: randomly fill small lots
            if (step % 20 == 0) {
                bool buy = (norm(rng) > 0);
                Quantity qty = 100;
                double px = prices[i];
                total_volume += px * qty;
                total_fills++;

                if (buy) {
                    double total_cost = avg_costs[i] * std::max(positions[i], 0LL) + px * qty;
                    positions[i] += qty;
                    if (positions[i] > 0) avg_costs[i] = total_cost / positions[i];
                } else {
                    if (positions[i] > 0) {
                        double close = std::min((double)qty, (double)positions[i]);
                        pnl[i] += (px - avg_costs[i]) * close;
                    }
                    positions[i] -= qty;
                }

                double half_sp = prices[i] * 0.001;
                Price bid_p = double_to_price(prices[i] - half_sp);
                Price ask_p = double_to_price(prices[i] + half_sp);
                vpins[i].update(double_to_price(px), bid_p, ask_p, qty);
            }
        }

        if (step % 50 == 0) {
            double total = 0;
            for (size_t i = 0; i < N; ++i) {
                double unreal = positions[i] != 0 ? (prices[i] - avg_costs[i]) * positions[i] : 0;
                total += pnl[i] + unreal;
            }
            equity_history.push_back(total);
        }
    }

    // Finalize
    double port_pnl = 0;
    for (size_t i = 0; i < N; ++i) {
        double unreal = positions[i] != 0 ? (prices[i] - avg_costs[i]) * positions[i] : 0;
        result.symbol_pnl[i] = pnl[i] + unreal;
        result.symbol_vpin[i] = vpins[i].vpin();
        port_pnl += result.symbol_pnl[i];
    }
    result.portfolio_pnl = port_pnl;
    result.total_volume  = total_volume;
    result.total_fills   = total_fills;

    // Sharpe from equity_history
    if (equity_history.size() > 2) {
        std::vector<double> eq_ret;
        for (size_t i = 1; i < equity_history.size(); ++i)
            eq_ret.push_back(equity_history[i] - equity_history[i-1]);
        double mean = 0, var = 0;
        for (auto r : eq_ret) mean += r;
        mean /= eq_ret.size();
        for (auto r : eq_ret) var += (r - mean) * (r - mean);
        var /= eq_ret.size();
        result.portfolio_sharpe = var > 0 ? mean / std::sqrt(var) * std::sqrt(252.0) : 0;
    }
    return result;
}

// ============================================================
// Execution Quality Analysis
// Measures slippage, fill rates, participation
// ============================================================
struct ExecutionStats {
    double avg_slippage_bps;
    double fill_rate;
    double participation_rate;
    double implementation_shortfall;
    double market_impact_cost;
    int    total_child_orders;
    int    total_filled;
    double notional_filled;
};

ExecutionStats analyze_execution(const std::string& symbol,
                                  double arrival_px,
                                  double vwap_px,
                                  double twap_px,
                                  const std::vector<std::pair<double,int>>& fills, // (price, qty)
                                  double market_vol)
{
    ExecutionStats stats{};
    if (fills.empty()) return stats;

    double total_qty = 0, total_cost = 0;
    for (auto& [px, qty] : fills) {
        total_qty  += qty;
        total_cost += px * qty;
        stats.total_filled++;
        stats.notional_filled += px * qty;
    }
    stats.total_child_orders = static_cast<int>(fills.size());
    double avg_fill_px = total_cost / total_qty;

    // Implementation shortfall: arrival_px vs avg fill
    stats.implementation_shortfall = (avg_fill_px - arrival_px) / arrival_px * 10000.0; // bps
    stats.avg_slippage_bps = std::fabs(avg_fill_px - vwap_px) / vwap_px * 10000.0;

    // Market impact estimate (simplified: half of vol * sqrt(pct_vol))
    double pct_vol = total_qty / (total_qty + 10000.0); // assume 10K mkt vol
    stats.market_impact_cost = 0.5 * market_vol * std::sqrt(pct_vol) * avg_fill_px * total_qty;

    stats.fill_rate          = 1.0; // all orders filled in simulation
    stats.participation_rate = pct_vol;
    return stats;
}

// ============================================================
// Order Flow Analysis Report
// ============================================================
void print_flow_analysis(const std::string& symbol,
                          const MarketWideFlow& flow,
                          int n_trades)
{
    std::cout << "\n--- Order Flow Analysis: " << symbol << " ---\n";
    auto toxic = flow.most_toxic(5);
    std::cout << "  Most toxic symbols by VPIN:\n";
    for (auto& sf : toxic) {
        std::cout << "    " << std::left << std::setw(8) << sf.symbol
                  << " VPIN=" << fmt_double(sf.vpin, 3)
                  << " NetFlow=" << fmt_double(sf.net_flow / 1e6, 2) << "M"
                  << " Toxic=" << (sf.is_toxic ? "YES" : "no") << "\n";
    }
}

// ============================================================
// Latency Distribution Analysis
// ============================================================
struct LatencyBenchmark {
    std::string name;
    std::vector<double> samples_ns;
    double p50, p95, p99, p999, mean, std_dev;

    void compute_stats() {
        std::sort(samples_ns.begin(), samples_ns.end());
        size_t n = samples_ns.size();
        if (n == 0) return;
        p50  = samples_ns[n * 50 / 100];
        p95  = samples_ns[n * 95 / 100];
        p99  = samples_ns[n * 99 / 100];
        p999 = samples_ns[std::min(n - 1, n * 999 / 1000)];
        mean = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0) / n;
        double var = 0;
        for (auto s : samples_ns) var += (s - mean) * (s - mean);
        std_dev = std::sqrt(var / n);
    }

    void print() const {
        std::cout << std::left << std::setw(24) << name
                  << " p50=" << std::right << std::setw(7) << fmt_double(p50, 1)
                  << " p95=" << std::setw(7) << fmt_double(p95, 1)
                  << " p99=" << std::setw(7) << fmt_double(p99, 1)
                  << " p99.9=" << std::setw(8) << fmt_double(p999, 1)
                  << " mean=" << std::setw(7) << fmt_double(mean, 1) << " ns\n";
    }
};

std::vector<LatencyBenchmark> run_latency_benchmarks() {
    std::vector<LatencyBenchmark> results;
    std::mt19937 rng(42);
    std::lognormal_distribution<double> ln(2.0, 0.8);  // log-normal latency distribution

    // Add Order benchmark
    {
        LatencyBenchmark bm;
        bm.name = "add_order";
        OrderBook book(nullptr, "BENCH");
        bm.samples_ns.reserve(10000);
        for (int i = 0; i < 10000; ++i) {
            Order o{};
            o.id         = i + 1;
            o.side       = (i % 2 == 0) ? Side::Buy : Side::Sell;
            o.order_type = OrderType::Limit;
            o.status     = OrderStatus::New;
            o.price      = double_to_price(100.0 + (rng() % 100) * 0.01 - 0.50);
            o.qty        = 100;
            o.tif        = TimeInForce::GTC;

            auto t0 = std::chrono::high_resolution_clock::now();
            auto fills = book.add_order(o);
            auto t1 = std::chrono::high_resolution_clock::now();
            fills.deinit();
            bm.samples_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        bm.compute_stats();
        results.push_back(std::move(bm));
    }

    // Cancel Order benchmark
    {
        LatencyBenchmark bm;
        bm.name = "cancel_order";
        OrderBook book(nullptr, "BENCH2");
        std::vector<OrderId> ids;
        ids.reserve(5000);

        // Pre-populate book
        for (int i = 0; i < 5000; ++i) {
            Order o{};
            o.id = i + 1;
            o.side = (i % 2 == 0) ? Side::Buy : Side::Sell;
            o.order_type = OrderType::Limit;
            o.status = OrderStatus::New;
            o.price = double_to_price(100.0 + (i % 50) * 0.01 - 0.25);
            o.qty = 100;
            o.tif = TimeInForce::GTC;
            auto f = book.add_order(o);
            f.deinit();
            ids.push_back(i + 1);
        }

        bm.samples_ns.reserve(5000);
        for (OrderId id : ids) {
            auto t0 = std::chrono::high_resolution_clock::now();
            book.cancel_order(id);
            auto t1 = std::chrono::high_resolution_clock::now();
            bm.samples_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        bm.compute_stats();
        results.push_back(std::move(bm));
    }

    // Market order matching benchmark
    {
        LatencyBenchmark bm;
        bm.name = "market_match";
        bm.samples_ns.reserve(1000);

        for (int trial = 0; trial < 1000; ++trial) {
            OrderBook book(nullptr, "BENCH3");
            // Add 20 passive asks
            for (int i = 0; i < 20; ++i) {
                Order o{};
                o.id = i + 1;
                o.side = Side::Sell;
                o.order_type = OrderType::Limit;
                o.status = OrderStatus::New;
                o.price = double_to_price(100.0 + i * 0.01);
                o.qty = 100;
                o.tif = TimeInForce::GTC;
                auto f = book.add_order(o);
                f.deinit();
            }
            // Aggress with market buy
            Order mkt{};
            mkt.id = 9999;
            mkt.side = Side::Buy;
            mkt.order_type = OrderType::Market;
            mkt.status = OrderStatus::New;
            mkt.qty = 500;
            mkt.tif = TimeInForce::IOC;

            auto t0 = std::chrono::high_resolution_clock::now();
            auto fills = book.add_order(mkt);
            auto t1 = std::chrono::high_resolution_clock::now();
            fills.deinit();
            bm.samples_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
        bm.compute_stats();
        results.push_back(std::move(bm));
    }

    return results;
}

// ============================================================
// Print formatted simulation results
// ============================================================
void print_scenario_result(const std::string& symbol, const ScenarioResult& r) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  Simulation Results: " << symbol << "\n";
    std::cout << std::string(60, '=') << "\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Price:           " << r.final_price
              << "  (" << (r.price_return_pct >= 0 ? "+" : "") << r.price_return_pct << "%)\n";
    std::cout << "  Realized Vol:    " << fmt_double(r.realized_vol * 100, 1) << "% annualized\n";
    std::cout << "  Orders Placed:   " << r.total_orders << "\n";
    std::cout << "  Orders Cancelled:" << r.total_cancels << "\n";
    std::cout << "  Fills:           " << r.total_trades << "\n";
    std::cout << "  Realized PnL:    $" << fmt_double(r.realized_pnl, 2) << "\n";
    std::cout << "  Unrealized PnL:  $" << fmt_double(r.unrealized_pnl, 2) << "\n";
    std::cout << "  Total Fees:      $" << fmt_double(r.total_fees, 2) << "\n";
    std::cout << "  Net PnL:         $" << fmt_double(r.realized_pnl + r.unrealized_pnl - r.total_fees, 2) << "\n";
    std::cout << "  Max Drawdown:    " << fmt_double(r.max_drawdown * 100, 2) << "%\n";
    std::cout << "  Sharpe Ratio:    " << fmt_double(r.sharpe_ratio, 3) << "\n";
    std::cout << "  VPIN (final):    " << fmt_double(r.vpin_final, 4) << "\n";

    // Equity curve sparkline
    if (!r.equity_curve.empty()) {
        double min_eq = *std::min_element(r.equity_curve.begin(), r.equity_curve.end());
        double max_eq = *std::max_element(r.equity_curve.begin(), r.equity_curve.end());
        double range  = max_eq - min_eq;
        std::cout << "  Equity curve:    ";
        std::string spark = " ._-~^\"";
        for (size_t i = 0; i < std::min(r.equity_curve.size(), (size_t)50); ++i) {
            double frac = range > 0 ? (r.equity_curve[i] - min_eq) / range : 0.5;
            int idx = static_cast<int>(frac * (spark.size() - 1));
            std::cout << spark[idx];
        }
        std::cout << "\n";
    }
}

// ============================================================
// Main entry point
// ============================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║         HFT Native Systems — Full Integration Demo       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    auto wall_t0 = std::chrono::high_resolution_clock::now();

    // ── 1. Single-symbol simulation ──────────────────────────────────────────
    std::cout << "1. Running single-symbol market simulation (AAPL, 50K steps)...\n";
    ScenarioConfig cfg{};
    cfg.symbol        = "AAPL";
    cfg.initial_price = 182.50;
    cfg.n_steps       = 50000;
    cfg.seed          = 42;

    MarketSimulation sim(cfg);
    auto res = sim.run();
    print_scenario_result(cfg.symbol, res);

    // ── 2. Multi-symbol portfolio simulation ─────────────────────────────────
    std::cout << "\n2. Running multi-symbol portfolio simulation (5 stocks)...\n";
    PortfolioConfig pcfg{};
    pcfg.n_steps = 20000;
    auto port_res = run_portfolio_simulation(pcfg);

    std::cout << "\n--- Portfolio Results ---\n";
    std::cout << "  Total PnL:       $" << fmt_double(port_res.portfolio_pnl, 2) << "\n";
    std::cout << "  Portfolio Sharpe:" << fmt_double(port_res.portfolio_sharpe, 3) << "\n";
    std::cout << "  Total Volume:    $" << fmt_double(port_res.total_volume / 1e6, 1) << "M\n";
    std::cout << "  Total Fills:     " << port_res.total_fills << "\n";
    for (size_t i = 0; i < pcfg.symbols.size(); ++i) {
        std::cout << "  " << std::left << std::setw(6) << pcfg.symbols[i]
                  << " PnL=$" << std::right << std::setw(9) << fmt_double(port_res.symbol_pnl[i], 2)
                  << "  VPIN=" << fmt_double(port_res.symbol_vpin[i], 4)
                  << "\n";
    }

    // ── 3. Execution quality analysis ────────────────────────────────────────
    std::cout << "\n3. Execution quality analysis...\n";
    std::vector<std::pair<double,int>> fills;
    std::mt19937 rng(77);
    std::normal_distribution<double> fill_noise(0, 0.05);
    for (int i = 0; i < 20; ++i)
        fills.push_back({182.50 + fill_noise(rng), 500});

    auto exec = analyze_execution("AAPL", 182.50, 182.48, 182.51, fills, 0.22);
    std::cout << "  Fill Rate:       " << fmt_double(exec.fill_rate * 100, 1) << "%\n";
    std::cout << "  Avg Slippage:    " << fmt_double(exec.avg_slippage_bps, 2) << " bps\n";
    std::cout << "  Impl Shortfall:  " << fmt_double(exec.implementation_shortfall, 2) << " bps\n";
    std::cout << "  Mkt Impact Cost: $" << fmt_double(exec.market_impact_cost, 2) << "\n";
    std::cout << "  Participation:   " << fmt_double(exec.participation_rate * 100, 2) << "%\n";

    // ── 4. Latency benchmarks ─────────────────────────────────────────────────
    std::cout << "\n4. Latency benchmarks...\n";
    auto lat_results = run_latency_benchmarks();
    std::cout << std::left << std::setw(24) << "  Operation"
              << std::right << std::setw(12) << "p50"
              << std::setw(12) << "p95"
              << std::setw(12) << "p99"
              << std::setw(13) << "p99.9"
              << std::setw(12) << "mean\n";
    std::cout << "  " << std::string(80, '-') << "\n";
    for (auto& bm : lat_results) {
        std::cout << "  ";
        bm.print();
    }

    // ── 5. Backtester comparison ──────────────────────────────────────────────
    std::cout << "\n5. Strategy comparison backtest...\n";
    SyntheticEventGenerator::Config gen_cfg{};
    gen_cfg.symbol        = "SIM";
    gen_cfg.initial_price = 100.0;
    gen_cfg.daily_vol     = 0.20;
    gen_cfg.num_events    = 100000;
    gen_cfg.seed          = 2024;

    SyntheticEventGenerator gen(gen_cfg);
    auto events = gen.generate();
    std::cout << "  Generated " << events.size() << " synthetic events.\n";

    // Market maker
    {
        Backtester::Config btcfg{};
        btcfg.fee_per_share = 0.001;
        btcfg.initial_capital = 500000.0;
        MarketMakerStrategy::Params mm_p{};
        mm_p.base_half_spread = 2.0;
        mm_p.quote_qty = 100;
        MarketMakerStrategy mm(mm_p);
        Backtester bt("SIM", btcfg);
        auto r = bt.run(events, mm);
        std::cout << "\n  [Market Maker]\n";
        r.print();
    }

    // ── 6. Almgren-Chriss execution schedule ─────────────────────────────────
    std::cout << "\n6. Almgren-Chriss optimal execution schedule...\n";
    AlmgrenChriss::Params ac_p{};
    ac_p.S0       = 150.0;
    ac_p.X        = 100000.0;   // 100K shares to sell
    ac_p.T        = 1.0;        // 1 trading day
    ac_p.N        = 10;         // 10 slices
    ac_p.sigma    = 0.25;
    ac_p.eta      = 2.5e-7;
    ac_p.gamma    = 2.5e-8;
    ac_p.lambda   = 1e-6;       // risk aversion

    AlmgrenChriss ac(ac_p);
    auto schedule = ac.optimal_schedule();
    std::cout << "  " << schedule.size() << " execution slices:\n";
    double total_sched = 0;
    for (size_t i = 0; i < std::min(schedule.size(), (size_t)5); ++i) {
        std::cout << "    Slice " << i+1 << ": " << fmt_double(schedule[i], 0) << " shares\n";
        total_sched += schedule[i];
    }
    if (schedule.size() > 5) {
        std::cout << "    ... (" << schedule.size() - 5 << " more slices)\n";
        for (size_t i = 5; i < schedule.size(); ++i) total_sched += schedule[i];
    }
    std::cout << "  Total scheduled: " << fmt_double(total_sched, 0) << " shares\n";

    // Market impact model
    MarketImpactModel mim;
    mim.kyle.update(1.0, 0.01);
    mim.kyle.update(-0.5, -0.008);
    mim.kyle.update(1.5, 0.015);
    std::cout << "  Kyle lambda:     " << fmt_double(mim.kyle.lambda(), 6) << "\n";

    // ── 7. Wall clock summary ─────────────────────────────────────────────────
    auto wall_t1 = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  All simulations completed in " << fmt_double(wall_ms, 1) << " ms\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
