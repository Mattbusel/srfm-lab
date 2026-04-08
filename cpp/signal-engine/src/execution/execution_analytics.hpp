#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <numeric>

namespace srfm::execution {

enum class Side : uint8_t { Buy = 0, Sell = 1 };
enum class OrderType : uint8_t { Market = 0, Limit = 1, IOC = 2, FOK = 3 };

struct Fill {
    uint64_t order_id;
    uint64_t trade_id;
    Side side;
    double price;
    double quantity;
    uint64_t timestamp_ns;
    double quoted_spread;    // spread at time of fill
    double arrival_price;    // mid price at order submission
    double decision_price;   // price when decision was made
};

struct MarketBar {
    double open, high, low, close;
    double volume;
    double vwap;
    uint64_t timestamp_ns;
};

struct OrderInfo {
    uint64_t order_id;
    Side side;
    OrderType type;
    double target_quantity;
    double filled_quantity;
    double limit_price;
    uint64_t submit_time_ns;
    uint64_t first_fill_time_ns;
    uint64_t last_fill_time_ns;
    double arrival_mid;
    double decision_price;
    int num_fills;
    int num_rejects;
};

// ----------- Implementation Shortfall -----------
struct ISResult {
    double shortfall_bps;      // basis points
    double shortfall_dollars;
    double arrival_price;
    double avg_execution_price;
    double realized_gain_loss;
    double delay_cost_bps;
    double trading_cost_bps;
    double opportunity_cost_bps;
};

class ImplementationShortfall {
public:
    void add_fill(const Fill& fill, const OrderInfo& order);
    ISResult compute(const OrderInfo& order) const;
    void reset();

private:
    std::vector<Fill> fills_;
};

// ----------- VWAP Slippage -----------
struct VWAPResult {
    double execution_vwap;
    double market_vwap;
    double slippage_bps;
    double participation_rate;
    double total_quantity;
    double market_volume;
};

class VWAPTracker {
public:
    void add_fill(const Fill& fill);
    void add_market_bar(const MarketBar& bar);
    VWAPResult compute() const;
    void reset();

private:
    std::vector<Fill> fills_;
    std::vector<MarketBar> bars_;
};

// ----------- Market Impact -----------
struct ImpactResult {
    double total_impact_bps;
    double temporary_impact_bps;
    double permanent_impact_bps;
    double realized_impact_bps;
    double post_trade_reversion;  // price reversion after execution
    double participation_rate;
};

class MarketImpact {
public:
    struct Config {
        int reversion_window_bars = 10;
        double temporary_decay = 0.5;
    };

    MarketImpact();
    explicit MarketImpact(const Config& cfg);

    void add_fill(const Fill& fill);
    void add_market_bar(const MarketBar& bar);
    void set_pre_trade_mid(double mid) { pre_trade_mid_ = mid; }
    void set_post_trade_mid(double mid) { post_trade_mid_ = mid; }
    ImpactResult compute(Side side) const;
    void reset();

private:
    Config config_;
    std::vector<Fill> fills_;
    std::vector<MarketBar> bars_;
    double pre_trade_mid_;
    double post_trade_mid_;
};

// ----------- Spread Capture -----------
struct SpreadResult {
    double avg_realized_spread_bps;
    double avg_quoted_spread_bps;
    double spread_capture_ratio;    // realized / quoted
    double effective_spread_bps;
    int n_fills;
};

class SpreadCapture {
public:
    void add_fill(const Fill& fill, double mid_at_fill);
    SpreadResult compute() const;
    void reset();

private:
    struct FillWithMid {
        Fill fill;
        double mid;
    };
    std::vector<FillWithMid> fills_;
};

// ----------- Fill Rate Analysis -----------
struct FillRateResult {
    double fill_rate;              // filled / target
    double partial_fill_rate;      // orders with partial fills / total orders
    double rejection_rate;         // rejected / total
    double avg_fill_time_ms;       // time to fill
    double median_fill_time_ms;
    double avg_fills_per_order;
    int total_orders;
    int fully_filled;
    int partially_filled;
    int rejected;
};

class FillRateAnalyzer {
public:
    void add_order(const OrderInfo& order);
    FillRateResult compute() const;
    void reset();

private:
    std::vector<OrderInfo> orders_;
};

// ----------- Timing Analysis -----------
struct TimingResult {
    double time_weighted_slippage_bps;
    double urgency_score;           // 0=patient, 1=aggressive
    double front_loading_ratio;     // fraction of qty in first third of time
    double back_loading_ratio;
    double consistency_score;       // how evenly distributed fills are
};

class TimingAnalyzer {
public:
    void add_fill(const Fill& fill, uint64_t order_start_ns, uint64_t order_end_ns);
    void set_arrival_price(double p) { arrival_price_ = p; }
    TimingResult compute() const;
    void reset();

private:
    struct TimedFill {
        Fill fill;
        double normalized_time; // 0..1 within order lifetime
    };
    std::vector<TimedFill> fills_;
    double arrival_price_ = 0;
};

// ----------- Cost Decomposition -----------
struct CostDecomp {
    double total_cost_bps;
    double spread_cost_bps;
    double impact_cost_bps;
    double timing_cost_bps;
    double opportunity_cost_bps;
    double commission_bps;
};

class CostDecomposer {
public:
    struct Config {
        double commission_per_share = 0.005;
    };

    CostDecomposer();
    explicit CostDecomposer(const Config& cfg);

    void set_order(const OrderInfo& order);
    void add_fill(const Fill& fill, double mid_at_fill, double market_vwap_at_fill);
    void set_final_mid(double mid) { final_mid_ = mid; }
    CostDecomp compute() const;
    void reset();

private:
    Config config_;
    OrderInfo order_;
    struct CostFill {
        Fill fill;
        double mid;
        double market_vwap;
    };
    std::vector<CostFill> fills_;
    double final_mid_;
    bool has_order_;
};

// ----------- TCA Report -----------
struct TCATradeReport {
    uint64_t order_id;
    ISResult implementation_shortfall;
    VWAPResult vwap;
    SpreadResult spread;
    TimingResult timing;
    CostDecomp cost;
};

struct TCAAggregateReport {
    int n_trades;
    double avg_is_bps;
    double avg_vwap_slippage_bps;
    double avg_spread_capture;
    double avg_total_cost_bps;
    double total_dollar_cost;
    double avg_fill_rate;
    double worst_is_bps;
    double best_is_bps;
    double is_std_dev;
};

class TCAEngine {
public:
    void add_trade_report(const TCATradeReport& report);
    TCAAggregateReport aggregate() const;
    const std::vector<TCATradeReport>& reports() const { return reports_; }
    void reset();

private:
    std::vector<TCATradeReport> reports_;
};

} // namespace srfm::execution
