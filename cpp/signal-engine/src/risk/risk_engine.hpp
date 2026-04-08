///////////////////////////////////////////////////////////////////////////////
// risk_engine.hpp
// Real-Time Risk Engine in C++
// VaR, CVaR, Greeks, Stress, Drawdown, Concentration, Limits, Alerts
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <deque>
#include <cassert>
#include <limits>

namespace signal_engine {
namespace risk {

using Vec = std::vector<double>;
using Mat = std::vector<std::vector<double>>;
using TimePoint = std::chrono::steady_clock::time_point;

///////////////////////////////////////////////////////////////////////////////
// Forward Declarations
///////////////////////////////////////////////////////////////////////////////

struct Position;
struct PortfolioSnapshot;
struct RiskLimits;
struct Alert;
struct StressScenario;

///////////////////////////////////////////////////////////////////////////////
// Position
///////////////////////////////////////////////////////////////////////////////

struct Position {
    std::string symbol;
    double quantity;
    double price;
    double market_value;
    double weight;

    // Greeks (for derivatives)
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;

    // Sector / classification
    std::string sector;
    std::string asset_class;

    // Liquidity
    double avg_daily_volume;
    double bid_ask_spread;

    double notional() const { return std::abs(quantity * price); }
};

///////////////////////////////////////////////////////////////////////////////
// PortfolioSnapshot
///////////////////////////////////////////////////////////////////////////////

struct PortfolioSnapshot {
    std::vector<Position> positions;
    double total_value;
    double cash;
    TimePoint timestamp;

    size_t n_positions() const { return positions.size(); }

    Vec weights() const {
        Vec w(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            w[i] = positions[i].weight;
        }
        return w;
    }

    Vec market_values() const {
        Vec mv(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            mv[i] = positions[i].market_value;
        }
        return mv;
    }
};

///////////////////////////////////////////////////////////////////////////////
// VaR Results
///////////////////////////////////////////////////////////////////////////////

struct VaRResult {
    double var;
    double cvar;           // Expected Shortfall
    double alpha;
    std::string method;
    int computation_time_us;  // microseconds
};

///////////////////////////////////////////////////////////////////////////////
// Greeks Aggregate
///////////////////////////////////////////////////////////////////////////////

struct GreeksAggregate {
    double portfolio_delta;
    double portfolio_gamma;
    double portfolio_vega;
    double portfolio_theta;
    double portfolio_rho;

    // Per-sector aggregates
    std::unordered_map<std::string, double> sector_delta;
    std::unordered_map<std::string, double> sector_gamma;
    std::unordered_map<std::string, double> sector_vega;

    // Dollar greeks
    double dollar_delta;
    double dollar_gamma;
    double dollar_vega;
    double dollar_theta;
};

///////////////////////////////////////////////////////////////////////////////
// Drawdown State
///////////////////////////////////////////////////////////////////////////////

struct DrawdownState {
    double high_water_mark;
    double current_drawdown;
    double max_drawdown;
    int duration_days;      // days in current drawdown
    int max_duration;       // max drawdown duration
    bool deleverage_triggered;
    double deleverage_threshold;
};

///////////////////////////////////////////////////////////////////////////////
// Concentration Metrics
///////////////////////////////////////////////////////////////////////////////

struct ConcentrationMetrics {
    double hhi;                    // Herfindahl-Hirschman Index
    double effective_n;            // 1/HHI
    double top1_weight;
    double top5_weight;
    double top10_weight;
    std::unordered_map<std::string, double> sector_weights;
    double max_sector_weight;
    std::string largest_sector;
};

///////////////////////////////////////////////////////////////////////////////
// Liquidity Metrics
///////////////////////////////////////////////////////////////////////////////

struct LiquidityMetrics {
    double portfolio_participation_rate;
    double days_to_liquidate_95;   // 95% liquidation
    double days_to_liquidate_100;
    Vec per_asset_days;
    double illiquid_fraction;       // fraction > 5 days to liquidate
    double weighted_spread;         // weighted average bid-ask
};

///////////////////////////////////////////////////////////////////////////////
// Stress Test Result
///////////////////////////////////////////////////////////////////////////////

struct StressResult {
    std::string scenario_name;
    double portfolio_pnl;
    double portfolio_pnl_pct;
    Vec position_pnl;
    double worst_position_pnl;
    std::string worst_position_name;
};

///////////////////////////////////////////////////////////////////////////////
// Risk Limits
///////////////////////////////////////////////////////////////////////////////

struct RiskLimits {
    // Position limits
    double max_position_weight;      // e.g., 0.10 (10%)
    double max_position_value;       // absolute dollar cap

    // Sector limits
    double max_sector_weight;        // e.g., 0.25 (25%)
    std::unordered_map<std::string, double> sector_limits;

    // VaR limits
    double max_var_pct;              // max VaR as % of portfolio
    double max_var_dollar;           // max VaR in dollars

    // Drawdown limits
    double max_drawdown;             // max allowed drawdown
    double deleverage_threshold;     // drawdown level to start deleveraging
    double deleverage_target;        // target exposure after deleverage

    // Correlation limits
    double max_correlation;          // max pairwise correlation
    double max_avg_correlation;

    // Concentration limits
    double min_effective_n;          // minimum effective number of bets
    double max_hhi;

    // Leverage
    double max_gross_leverage;
    double max_net_leverage;
};

///////////////////////////////////////////////////////////////////////////////
// Alert
///////////////////////////////////////////////////////////////////////////////

enum class AlertSeverity {
    INFO,
    WARNING,
    CRITICAL,
    BREACH
};

struct Alert {
    AlertSeverity severity;
    std::string category;
    std::string message;
    double current_value;
    double limit_value;
    double utilization_pct;     // current/limit * 100
    TimePoint timestamp;
};

///////////////////////////////////////////////////////////////////////////////
// RiskEngine Class
///////////////////////////////////////////////////////////////////////////////

class RiskEngine {
public:
    explicit RiskEngine(size_t max_history = 2520);  // ~10 years daily
    ~RiskEngine() = default;

    // -----------------------------------------------------------------------
    // Portfolio Update
    // -----------------------------------------------------------------------

    void update_portfolio(const PortfolioSnapshot& snapshot);
    void add_return(double portfolio_return);
    void add_returns(const Vec& asset_returns);  // multi-asset

    // -----------------------------------------------------------------------
    // VaR / CVaR
    // -----------------------------------------------------------------------

    /// Historical VaR/CVaR - O(n log n) for sorting, O(1) lookup
    VaRResult historical_var(double alpha = 0.95) const;

    /// Parametric VaR (Normal)
    VaRResult parametric_var_normal(double alpha = 0.95) const;

    /// Parametric VaR (Student-t)
    VaRResult parametric_var_t(double alpha = 0.95, double nu = 5.0) const;

    /// Parametric VaR (Cornish-Fisher)
    VaRResult parametric_var_cf(double alpha = 0.95) const;

    /// Monte Carlo VaR (GBM simulation)
    VaRResult monte_carlo_var(double alpha = 0.95,
                               int n_sim = 10000) const;

    /// Portfolio VaR (multi-asset with correlation)
    VaRResult portfolio_var(const Vec& weights,
                             double alpha = 0.95) const;

    // -----------------------------------------------------------------------
    // Greeks
    // -----------------------------------------------------------------------

    GreeksAggregate aggregate_greeks() const;

    // -----------------------------------------------------------------------
    // Stress Testing
    // -----------------------------------------------------------------------

    StressResult run_stress_test(const StressScenario& scenario) const;
    std::vector<StressResult> run_all_stress_tests() const;

    void add_stress_scenario(const StressScenario& scenario);
    void add_predefined_scenarios();

    // -----------------------------------------------------------------------
    // Drawdown
    // -----------------------------------------------------------------------

    DrawdownState get_drawdown_state() const;
    bool check_deleverage() const;

    // -----------------------------------------------------------------------
    // Concentration
    // -----------------------------------------------------------------------

    ConcentrationMetrics compute_concentration() const;

    // -----------------------------------------------------------------------
    // Liquidity
    // -----------------------------------------------------------------------

    LiquidityMetrics compute_liquidity(
        double participation_rate = 0.10) const;

    // -----------------------------------------------------------------------
    // Risk Limits
    // -----------------------------------------------------------------------

    void set_risk_limits(const RiskLimits& limits);
    std::vector<Alert> check_all_limits() const;
    bool is_within_limits() const;

    // -----------------------------------------------------------------------
    // Alert System
    // -----------------------------------------------------------------------

    std::vector<Alert> get_active_alerts() const;
    void clear_alerts();

    // -----------------------------------------------------------------------
    // Full Risk Report
    // -----------------------------------------------------------------------

    struct RiskReport {
        VaRResult var_95;
        VaRResult var_99;
        VaRResult cvar_95;
        GreeksAggregate greeks;
        DrawdownState drawdown;
        ConcentrationMetrics concentration;
        LiquidityMetrics liquidity;
        std::vector<StressResult> stress_results;
        std::vector<Alert> alerts;
        int computation_time_us;
    };

    RiskReport generate_report() const;

private:
    // Portfolio state
    PortfolioSnapshot current_portfolio_;
    std::deque<double> return_history_;
    std::deque<Vec> asset_return_history_;
    size_t max_history_;

    // Pre-sorted returns for O(1) percentile lookup
    mutable Vec sorted_returns_;
    mutable bool sorted_dirty_ = true;

    // Running statistics for incremental computation
    double running_mean_ = 0.0;
    double running_m2_ = 0.0;
    double running_m3_ = 0.0;
    double running_m4_ = 0.0;
    size_t running_count_ = 0;

    // Drawdown tracking
    double hwm_ = 0.0;
    double current_equity_ = 1.0;
    double max_drawdown_ = 0.0;
    int dd_duration_ = 0;
    int max_dd_duration_ = 0;

    // Stress scenarios
    std::vector<StressScenario> stress_scenarios_;

    // Risk limits
    RiskLimits limits_;
    bool limits_set_ = false;

    // Alert buffer
    mutable std::vector<Alert> alerts_;

    // Internal helpers
    void update_running_stats(double r);
    void ensure_sorted() const;
    double percentile(double p) const;

    // Cornish-Fisher quantile
    double cornish_fisher_quantile(double alpha) const;

    // Cholesky for MC simulation
    Mat compute_correlation_matrix() const;
    Mat cholesky_decompose(const Mat& A) const;
};

///////////////////////////////////////////////////////////////////////////////
// Stress Scenario
///////////////////////////////////////////////////////////////////////////////

struct StressScenario {
    std::string name;
    std::unordered_map<std::string, double> sector_shocks;  // sector -> pct shock
    std::unordered_map<std::string, double> asset_shocks;   // symbol -> pct shock
    double market_shock;           // broad market shock (applied to all)
    double vol_multiplier;         // volatility scaling
    double rate_shock_bps;         // interest rate shock
    double spread_shock_bps;       // credit spread shock
};

} // namespace risk
} // namespace signal_engine
