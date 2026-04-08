///////////////////////////////////////////////////////////////////////////////
// risk_engine.cpp
// Real-Time Risk Engine - Implementation
///////////////////////////////////////////////////////////////////////////////

#include "risk_engine.hpp"
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

namespace signal_engine {
namespace risk {

///////////////////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////////////////

RiskEngine::RiskEngine(size_t max_history)
    : max_history_(max_history) {
    return_history_.clear();
    asset_return_history_.clear();
    sorted_returns_.clear();
    alerts_.clear();
    stress_scenarios_.clear();
}

///////////////////////////////////////////////////////////////////////////////
// Portfolio Update
///////////////////////////////////////////////////////////////////////////////

void RiskEngine::update_portfolio(const PortfolioSnapshot& snapshot) {
    current_portfolio_ = snapshot;

    // Update equity for drawdown tracking
    current_equity_ = snapshot.total_value;
    if (current_equity_ > hwm_) {
        hwm_ = current_equity_;
        dd_duration_ = 0;
    } else {
        dd_duration_++;
        max_dd_duration_ = std::max(max_dd_duration_, dd_duration_);
    }

    if (hwm_ > 0) {
        double dd = (current_equity_ - hwm_) / hwm_;
        if (dd < max_drawdown_) {
            max_drawdown_ = dd;
        }
    }
}

void RiskEngine::add_return(double r) {
    return_history_.push_back(r);
    if (return_history_.size() > max_history_) {
        return_history_.pop_front();
    }
    sorted_dirty_ = true;
    update_running_stats(r);
}

void RiskEngine::add_returns(const Vec& asset_returns) {
    asset_return_history_.push_back(asset_returns);
    if (asset_return_history_.size() > max_history_) {
        asset_return_history_.pop_front();
    }
}

///////////////////////////////////////////////////////////////////////////////
// Running Statistics (Welford's online algorithm)
///////////////////////////////////////////////////////////////////////////////

void RiskEngine::update_running_stats(double r) {
    running_count_++;
    double n = static_cast<double>(running_count_);
    double delta = r - running_mean_;
    double delta_n = delta / n;
    double delta_n2 = delta_n * delta_n;
    double term1 = delta * delta_n * (n - 1);

    running_mean_ += delta_n;
    running_m4_ += term1 * delta_n2 * (n * n - 3 * n + 3) +
        6 * delta_n2 * running_m2_ - 4 * delta_n * running_m3_;
    running_m3_ += term1 * delta_n * (n - 2) - 3 * delta_n * running_m2_;
    running_m2_ += term1;
}

///////////////////////////////////////////////////////////////////////////////
// Sorted Returns for O(1) Percentile
///////////////////////////////////////////////////////////////////////////////

void RiskEngine::ensure_sorted() const {
    if (sorted_dirty_) {
        sorted_returns_.assign(return_history_.begin(), return_history_.end());
        std::sort(sorted_returns_.begin(), sorted_returns_.end());
        sorted_dirty_ = false;
    }
}

double RiskEngine::percentile(double p) const {
    ensure_sorted();
    if (sorted_returns_.empty()) return 0.0;

    size_t n = sorted_returns_.size();
    double idx = p * (n - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));

    if (lo == hi || hi >= n) return sorted_returns_[lo];

    double frac = idx - lo;
    return sorted_returns_[lo] * (1.0 - frac) + sorted_returns_[hi] * frac;
}

///////////////////////////////////////////////////////////////////////////////
// Historical VaR / CVaR
///////////////////////////////////////////////////////////////////////////////

VaRResult RiskEngine::historical_var(double alpha) const {
    auto start = std::chrono::steady_clock::now();

    if (return_history_.empty()) {
        return {0.0, 0.0, alpha, "historical", 0};
    }

    ensure_sorted();
    size_t n = sorted_returns_.size();

    // VaR: negative of the (1-alpha) quantile
    double var_val = -percentile(1.0 - alpha);

    // CVaR: mean of returns below VaR threshold
    size_t cutoff = static_cast<size_t>(std::floor((1.0 - alpha) * n));
    if (cutoff == 0) cutoff = 1;

    double sum_tail = 0.0;
    for (size_t i = 0; i < cutoff; ++i) {
        sum_tail += sorted_returns_[i];
    }
    double cvar_val = -sum_tail / cutoff;

    auto end = std::chrono::steady_clock::now();
    int us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return {var_val, cvar_val, alpha, "historical", us};
}

///////////////////////////////////////////////////////////////////////////////
// Parametric VaR (Normal)
///////////////////////////////////////////////////////////////////////////////

VaRResult RiskEngine::parametric_var_normal(double alpha) const {
    auto start = std::chrono::steady_clock::now();

    if (running_count_ < 2) {
        return {0.0, 0.0, alpha, "parametric_normal", 0};
    }

    double mu = running_mean_;
    double var = running_m2_ / (running_count_ - 1);
    double sigma = std::sqrt(var);

    // Normal quantile (rational approximation)
    double p = alpha;
    double t = std::sqrt(-2.0 * std::log(1.0 - p));
    double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    double z = t - (c0 + c1 * t + c2 * t * t) /
        (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    double var_val = -(mu - z * sigma);

    // CVaR for normal: mu + sigma * phi(z) / (1-alpha)
    double phi_z = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
    double cvar_val = -(mu - sigma * phi_z / (1.0 - alpha));

    auto end = std::chrono::steady_clock::now();
    int us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return {var_val, cvar_val, alpha, "parametric_normal", us};
}

///////////////////////////////////////////////////////////////////////////////
// Parametric VaR (Student-t)
///////////////////////////////////////////////////////////////////////////////

VaRResult RiskEngine::parametric_var_t(double alpha, double nu) const {
    auto start = std::chrono::steady_clock::now();

    if (running_count_ < 2) {
        return {0.0, 0.0, alpha, "parametric_t", 0};
    }

    double mu = running_mean_;
    double var = running_m2_ / (running_count_ - 1);
    double sigma = std::sqrt(var);

    // Scale for t-distribution
    double s = sigma * std::sqrt((nu - 2.0) / nu);

    // t-quantile approximation (Cornish-Fisher on standard normal)
    double p = alpha;
    double t_val = std::sqrt(-2.0 * std::log(1.0 - p));
    double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    double z = t_val - (c0 + c1 * t_val + c2 * t_val * t_val) /
        (1.0 + d1 * t_val + d2 * t_val * t_val + d3 * t_val * t_val * t_val);

    // Adjust for heavier tails
    double g1 = (z * z + 1.0) / (4.0 * (nu - 2.0));
    double t_q = z + z * g1;  // First-order correction

    double var_val = -(mu - s * t_q);
    double cvar_val = var_val * (nu + t_q * t_q) / ((nu - 1.0) * (1.0 - alpha));

    auto end = std::chrono::steady_clock::now();
    int us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return {var_val, std::max(var_val, cvar_val), alpha, "parametric_t", us};
}

///////////////////////////////////////////////////////////////////////////////
// Cornish-Fisher VaR
///////////////////////////////////////////////////////////////////////////////

double RiskEngine::cornish_fisher_quantile(double alpha) const {
    if (running_count_ < 4) return 0.0;

    double n = static_cast<double>(running_count_);
    double var = running_m2_ / (n - 1);
    double sigma = std::sqrt(var);
    double skew = (running_m3_ / n) / (sigma * sigma * sigma);
    double kurt = (running_m4_ / n) / (sigma * sigma * sigma * sigma) - 3.0;

    // Normal quantile
    double p = alpha;
    double t = std::sqrt(-2.0 * std::log(1.0 - p));
    double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    double z = t - (c0 + c1 * t + c2 * t * t) /
        (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    // Cornish-Fisher expansion
    double z_cf = z
        + (z * z - 1.0) * skew / 6.0
        + (z * z * z - 3.0 * z) * kurt / 24.0
        - (2.0 * z * z * z - 5.0 * z) * skew * skew / 36.0;

    return z_cf;
}

VaRResult RiskEngine::parametric_var_cf(double alpha) const {
    auto start = std::chrono::steady_clock::now();

    if (running_count_ < 4) {
        return {0.0, 0.0, alpha, "cornish_fisher", 0};
    }

    double mu = running_mean_;
    double sigma = std::sqrt(running_m2_ / (running_count_ - 1));
    double z_cf = cornish_fisher_quantile(alpha);

    double var_val = -(mu - z_cf * sigma);

    // Approximate CVaR
    double cvar_val = var_val * 1.1;  // Rough estimate

    auto end = std::chrono::steady_clock::now();
    int us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return {var_val, cvar_val, alpha, "cornish_fisher", us};
}

///////////////////////////////////////////////////////////////////////////////
// Monte Carlo VaR
///////////////////////////////////////////////////////////////////////////////

VaRResult RiskEngine::monte_carlo_var(double alpha, int n_sim) const {
    auto start = std::chrono::steady_clock::now();

    if (running_count_ < 2) {
        return {0.0, 0.0, alpha, "monte_carlo", 0};
    }

    double mu = running_mean_;
    double sigma = std::sqrt(running_m2_ / (running_count_ - 1));

    // GBM simulation for 1-day returns
    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);

    Vec sim_returns(n_sim);
    for (int i = 0; i < n_sim; ++i) {
        double z = norm(rng);
        sim_returns[i] = mu + sigma * z;
    }

    std::sort(sim_returns.begin(), sim_returns.end());

    size_t var_idx = static_cast<size_t>(std::floor((1.0 - alpha) * n_sim));
    double var_val = -sim_returns[var_idx];

    double sum_tail = 0.0;
    for (size_t i = 0; i <= var_idx; ++i) {
        sum_tail += sim_returns[i];
    }
    double cvar_val = -sum_tail / (var_idx + 1);

    auto end = std::chrono::steady_clock::now();
    int us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return {var_val, cvar_val, alpha, "monte_carlo", us};
}

///////////////////////////////////////////////////////////////////////////////
// Portfolio VaR (Multi-Asset)
///////////////////////////////////////////////////////////////////////////////

VaRResult RiskEngine::portfolio_var(const Vec& weights, double alpha) const {
    auto start = std::chrono::steady_clock::now();

    if (asset_return_history_.empty()) {
        return {0.0, 0.0, alpha, "portfolio_historical", 0};
    }

    size_t n = asset_return_history_.size();
    size_t k = weights.size();

    // Compute portfolio returns
    Vec port_returns(n);
    for (size_t t = 0; t < n; ++t) {
        double r = 0.0;
        size_t m = std::min(k, asset_return_history_[t].size());
        for (size_t j = 0; j < m; ++j) {
            r += weights[j] * asset_return_history_[t][j];
        }
        port_returns[t] = r;
    }

    std::sort(port_returns.begin(), port_returns.end());

    size_t var_idx = static_cast<size_t>(std::floor((1.0 - alpha) * n));
    if (var_idx >= n) var_idx = n - 1;

    double var_val = -port_returns[var_idx];

    double sum_tail = 0.0;
    for (size_t i = 0; i <= var_idx; ++i) {
        sum_tail += port_returns[i];
    }
    double cvar_val = -sum_tail / (var_idx + 1);

    auto end = std::chrono::steady_clock::now();
    int us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return {var_val, cvar_val, alpha, "portfolio_historical", us};
}

///////////////////////////////////////////////////////////////////////////////
// Greeks Aggregation
///////////////////////////////////////////////////////////////////////////////

GreeksAggregate RiskEngine::aggregate_greeks() const {
    GreeksAggregate agg{};

    for (const auto& pos : current_portfolio_.positions) {
        double mv = pos.market_value;

        agg.portfolio_delta += pos.delta * pos.quantity;
        agg.portfolio_gamma += pos.gamma * pos.quantity;
        agg.portfolio_vega += pos.vega * pos.quantity;
        agg.portfolio_theta += pos.theta * pos.quantity;
        agg.portfolio_rho += pos.rho * pos.quantity;

        agg.dollar_delta += pos.delta * mv;
        agg.dollar_gamma += pos.gamma * mv;
        agg.dollar_vega += pos.vega * mv;
        agg.dollar_theta += pos.theta * mv;

        // Sector aggregation
        agg.sector_delta[pos.sector] += pos.delta * pos.quantity;
        agg.sector_gamma[pos.sector] += pos.gamma * pos.quantity;
        agg.sector_vega[pos.sector] += pos.vega * pos.quantity;
    }

    return agg;
}

///////////////////////////////////////////////////////////////////////////////
// Stress Testing
///////////////////////////////////////////////////////////////////////////////

StressResult RiskEngine::run_stress_test(const StressScenario& scenario) const {
    StressResult result;
    result.scenario_name = scenario.name;

    size_t n = current_portfolio_.positions.size();
    result.position_pnl.resize(n);

    double total_pnl = 0.0;
    double worst_pnl = 0.0;
    std::string worst_name;

    for (size_t i = 0; i < n; ++i) {
        const auto& pos = current_portfolio_.positions[i];
        double shock = scenario.market_shock;

        // Check for sector-specific shock
        auto it_sector = scenario.sector_shocks.find(pos.sector);
        if (it_sector != scenario.sector_shocks.end()) {
            shock = it_sector->second;
        }

        // Check for asset-specific shock
        auto it_asset = scenario.asset_shocks.find(pos.symbol);
        if (it_asset != scenario.asset_shocks.end()) {
            shock = it_asset->second;
        }

        double pnl = pos.market_value * shock;

        // Add second-order effects (gamma)
        if (pos.gamma != 0.0) {
            pnl += 0.5 * pos.gamma * pos.quantity * pos.price * pos.price *
                shock * shock;
        }

        // Volatility shock effect
        if (scenario.vol_multiplier != 0.0 && pos.vega != 0.0) {
            pnl += pos.vega * pos.quantity * (scenario.vol_multiplier - 1.0) * 0.2;
        }

        // Rate shock effect
        if (scenario.rate_shock_bps != 0.0 && pos.rho != 0.0) {
            pnl += pos.rho * pos.quantity * scenario.rate_shock_bps / 10000.0;
        }

        result.position_pnl[i] = pnl;
        total_pnl += pnl;

        if (pnl < worst_pnl) {
            worst_pnl = pnl;
            worst_name = pos.symbol;
        }
    }

    result.portfolio_pnl = total_pnl;
    result.portfolio_pnl_pct = (current_portfolio_.total_value > 0) ?
        total_pnl / current_portfolio_.total_value : 0.0;
    result.worst_position_pnl = worst_pnl;
    result.worst_position_name = worst_name;

    return result;
}

std::vector<StressResult> RiskEngine::run_all_stress_tests() const {
    std::vector<StressResult> results;
    results.reserve(stress_scenarios_.size());
    for (const auto& sc : stress_scenarios_) {
        results.push_back(run_stress_test(sc));
    }
    return results;
}

void RiskEngine::add_stress_scenario(const StressScenario& scenario) {
    stress_scenarios_.push_back(scenario);
}

void RiskEngine::add_predefined_scenarios() {
    // 2008 Financial Crisis
    StressScenario crisis_2008;
    crisis_2008.name = "2008 Financial Crisis";
    crisis_2008.market_shock = -0.40;
    crisis_2008.vol_multiplier = 3.0;
    crisis_2008.rate_shock_bps = -200;
    crisis_2008.spread_shock_bps = 500;
    crisis_2008.sector_shocks["Financials"] = -0.60;
    crisis_2008.sector_shocks["Real Estate"] = -0.45;
    crisis_2008.sector_shocks["Consumer Discretionary"] = -0.50;
    crisis_2008.sector_shocks["Utilities"] = -0.20;
    stress_scenarios_.push_back(crisis_2008);

    // COVID-19 Crash
    StressScenario covid;
    covid.name = "2020 COVID-19 Crash";
    covid.market_shock = -0.34;
    covid.vol_multiplier = 4.0;
    covid.rate_shock_bps = -150;
    covid.spread_shock_bps = 300;
    covid.sector_shocks["Energy"] = -0.50;
    covid.sector_shocks["Industrials"] = -0.40;
    covid.sector_shocks["Technology"] = -0.25;
    covid.sector_shocks["Healthcare"] = -0.15;
    stress_scenarios_.push_back(covid);

    // 2022 Rate Hiking
    StressScenario rate_hike;
    rate_hike.name = "2022 Rate Hiking Cycle";
    rate_hike.market_shock = -0.25;
    rate_hike.vol_multiplier = 1.5;
    rate_hike.rate_shock_bps = 300;
    rate_hike.spread_shock_bps = 150;
    rate_hike.sector_shocks["Technology"] = -0.35;
    rate_hike.sector_shocks["Real Estate"] = -0.30;
    rate_hike.sector_shocks["Energy"] = 0.10;
    stress_scenarios_.push_back(rate_hike);

    // Flash Crash
    StressScenario flash;
    flash.name = "Flash Crash";
    flash.market_shock = -0.10;
    flash.vol_multiplier = 5.0;
    flash.rate_shock_bps = -50;
    flash.spread_shock_bps = 200;
    stress_scenarios_.push_back(flash);

    // Stagflation
    StressScenario stagflation;
    stagflation.name = "Stagflation Scenario";
    stagflation.market_shock = -0.20;
    stagflation.vol_multiplier = 2.0;
    stagflation.rate_shock_bps = 200;
    stagflation.spread_shock_bps = 200;
    stagflation.sector_shocks["Consumer Staples"] = -0.10;
    stagflation.sector_shocks["Energy"] = 0.15;
    stress_scenarios_.push_back(stagflation);

    // Tail Risk
    StressScenario tail;
    tail.name = "5-Sigma Tail Event";
    tail.market_shock = -0.50;
    tail.vol_multiplier = 6.0;
    tail.rate_shock_bps = -300;
    tail.spread_shock_bps = 800;
    stress_scenarios_.push_back(tail);
}

///////////////////////////////////////////////////////////////////////////////
// Drawdown
///////////////////////////////////////////////////////////////////////////////

DrawdownState RiskEngine::get_drawdown_state() const {
    DrawdownState state;
    state.high_water_mark = hwm_;
    state.current_drawdown = (hwm_ > 0) ?
        (current_equity_ - hwm_) / hwm_ : 0.0;
    state.max_drawdown = max_drawdown_;
    state.duration_days = dd_duration_;
    state.max_duration = max_dd_duration_;

    state.deleverage_threshold = limits_set_ ?
        limits_.deleverage_threshold : -0.10;
    state.deleverage_triggered = state.current_drawdown < state.deleverage_threshold;

    return state;
}

bool RiskEngine::check_deleverage() const {
    if (!limits_set_) return false;
    double dd = (hwm_ > 0) ? (current_equity_ - hwm_) / hwm_ : 0.0;
    return dd < limits_.deleverage_threshold;
}

///////////////////////////////////////////////////////////////////////////////
// Concentration
///////////////////////////////////////////////////////////////////////////////

ConcentrationMetrics RiskEngine::compute_concentration() const {
    ConcentrationMetrics metrics{};
    size_t n = current_portfolio_.positions.size();
    if (n == 0) return metrics;

    // Weights (absolute)
    Vec abs_weights(n);
    double total_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        abs_weights[i] = std::abs(current_portfolio_.positions[i].weight);
        total_abs += abs_weights[i];
    }
    if (total_abs > 0) {
        for (size_t i = 0; i < n; ++i) abs_weights[i] /= total_abs;
    }

    // HHI
    metrics.hhi = 0.0;
    for (size_t i = 0; i < n; ++i) {
        metrics.hhi += abs_weights[i] * abs_weights[i];
    }
    metrics.effective_n = (metrics.hhi > 0) ? 1.0 / metrics.hhi : 0.0;

    // Top-N weights
    Vec sorted_w = abs_weights;
    std::sort(sorted_w.begin(), sorted_w.end(), std::greater<double>());

    metrics.top1_weight = (n >= 1) ? sorted_w[0] : 0.0;
    metrics.top5_weight = 0.0;
    for (size_t i = 0; i < std::min(n, size_t(5)); ++i) {
        metrics.top5_weight += sorted_w[i];
    }
    metrics.top10_weight = 0.0;
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
        metrics.top10_weight += sorted_w[i];
    }

    // Sector weights
    for (const auto& pos : current_portfolio_.positions) {
        metrics.sector_weights[pos.sector] += std::abs(pos.weight);
    }

    metrics.max_sector_weight = 0.0;
    for (const auto& [sector, weight] : metrics.sector_weights) {
        if (weight > metrics.max_sector_weight) {
            metrics.max_sector_weight = weight;
            metrics.largest_sector = sector;
        }
    }

    return metrics;
}

///////////////////////////////////////////////////////////////////////////////
// Liquidity
///////////////////////////////////////////////////////////////////////////////

LiquidityMetrics RiskEngine::compute_liquidity(double participation_rate) const {
    LiquidityMetrics metrics{};
    size_t n = current_portfolio_.positions.size();
    if (n == 0) return metrics;

    metrics.per_asset_days.resize(n);
    double total_value = current_portfolio_.total_value;
    double weighted_spread = 0.0;
    double illiquid_count = 0;
    double max_days = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const auto& pos = current_portfolio_.positions[i];
        double shares = std::abs(pos.quantity);
        double daily_capacity = pos.avg_daily_volume * participation_rate;

        if (daily_capacity > 0) {
            double days = std::ceil(shares / daily_capacity);
            metrics.per_asset_days[i] = days;
            max_days = std::max(max_days, days);
            if (days > 5) illiquid_count++;
        } else {
            metrics.per_asset_days[i] = 999.0;
            illiquid_count++;
        }

        weighted_spread += std::abs(pos.weight) * pos.bid_ask_spread;
    }

    metrics.days_to_liquidate_100 = max_days;
    metrics.days_to_liquidate_95 = max_days * 0.7;  // Approximate
    metrics.illiquid_fraction = illiquid_count / n;
    metrics.weighted_spread = weighted_spread;
    metrics.portfolio_participation_rate = participation_rate;

    return metrics;
}

///////////////////////////////////////////////////////////////////////////////
// Risk Limits
///////////////////////////////////////////////////////////////////////////////

void RiskEngine::set_risk_limits(const RiskLimits& limits) {
    limits_ = limits;
    limits_set_ = true;
}

std::vector<Alert> RiskEngine::check_all_limits() const {
    std::vector<Alert> new_alerts;
    if (!limits_set_) return new_alerts;

    auto now = std::chrono::steady_clock::now();

    // Position weight limits
    for (const auto& pos : current_portfolio_.positions) {
        double abs_w = std::abs(pos.weight);

        if (abs_w > limits_.max_position_weight) {
            Alert a;
            a.severity = AlertSeverity::BREACH;
            a.category = "position_weight";
            a.message = "Position " + pos.symbol + " weight " +
                std::to_string(abs_w) + " exceeds limit " +
                std::to_string(limits_.max_position_weight);
            a.current_value = abs_w;
            a.limit_value = limits_.max_position_weight;
            a.utilization_pct = abs_w / limits_.max_position_weight * 100.0;
            a.timestamp = now;
            new_alerts.push_back(a);
        } else if (abs_w > limits_.max_position_weight * 0.9) {
            Alert a;
            a.severity = AlertSeverity::WARNING;
            a.category = "position_weight";
            a.message = "Position " + pos.symbol + " approaching limit";
            a.current_value = abs_w;
            a.limit_value = limits_.max_position_weight;
            a.utilization_pct = abs_w / limits_.max_position_weight * 100.0;
            a.timestamp = now;
            new_alerts.push_back(a);
        }

        // Dollar position limit
        if (limits_.max_position_value > 0 &&
            pos.notional() > limits_.max_position_value) {
            Alert a;
            a.severity = AlertSeverity::BREACH;
            a.category = "position_value";
            a.message = "Position " + pos.symbol + " value exceeds limit";
            a.current_value = pos.notional();
            a.limit_value = limits_.max_position_value;
            a.utilization_pct = pos.notional() / limits_.max_position_value * 100.0;
            a.timestamp = now;
            new_alerts.push_back(a);
        }
    }

    // Sector limits
    auto concentration = compute_concentration();
    for (const auto& [sector, weight] : concentration.sector_weights) {
        double limit = limits_.max_sector_weight;

        // Check for sector-specific limit
        auto it = limits_.sector_limits.find(sector);
        if (it != limits_.sector_limits.end()) {
            limit = it->second;
        }

        if (weight > limit) {
            Alert a;
            a.severity = AlertSeverity::BREACH;
            a.category = "sector_weight";
            a.message = "Sector " + sector + " weight " +
                std::to_string(weight) + " exceeds limit " +
                std::to_string(limit);
            a.current_value = weight;
            a.limit_value = limit;
            a.utilization_pct = weight / limit * 100.0;
            a.timestamp = now;
            new_alerts.push_back(a);
        }
    }

    // VaR limits
    if (limits_.max_var_pct > 0 && !return_history_.empty()) {
        auto var = historical_var(0.95);
        if (var.var > limits_.max_var_pct) {
            Alert a;
            a.severity = AlertSeverity::BREACH;
            a.category = "var";
            a.message = "VaR " + std::to_string(var.var) +
                " exceeds limit " + std::to_string(limits_.max_var_pct);
            a.current_value = var.var;
            a.limit_value = limits_.max_var_pct;
            a.utilization_pct = var.var / limits_.max_var_pct * 100.0;
            a.timestamp = now;
            new_alerts.push_back(a);
        }
    }

    // Drawdown limits
    auto dd = get_drawdown_state();
    if (std::abs(dd.current_drawdown) > limits_.max_drawdown) {
        Alert a;
        a.severity = AlertSeverity::CRITICAL;
        a.category = "drawdown";
        a.message = "Drawdown " + std::to_string(dd.current_drawdown * 100.0) +
            "% exceeds limit " + std::to_string(limits_.max_drawdown * 100.0) + "%";
        a.current_value = dd.current_drawdown;
        a.limit_value = limits_.max_drawdown;
        a.utilization_pct = std::abs(dd.current_drawdown) / limits_.max_drawdown * 100.0;
        a.timestamp = now;
        new_alerts.push_back(a);
    }

    // Deleverage check
    if (dd.deleverage_triggered) {
        Alert a;
        a.severity = AlertSeverity::CRITICAL;
        a.category = "deleverage";
        a.message = "Deleverage triggered at drawdown " +
            std::to_string(dd.current_drawdown * 100.0) + "%";
        a.current_value = dd.current_drawdown;
        a.limit_value = limits_.deleverage_threshold;
        a.utilization_pct = 100.0;
        a.timestamp = now;
        new_alerts.push_back(a);
    }

    // Concentration limits
    if (limits_.min_effective_n > 0) {
        if (concentration.effective_n < limits_.min_effective_n) {
            Alert a;
            a.severity = AlertSeverity::WARNING;
            a.category = "concentration";
            a.message = "Effective N " + std::to_string(concentration.effective_n) +
                " below minimum " + std::to_string(limits_.min_effective_n);
            a.current_value = concentration.effective_n;
            a.limit_value = limits_.min_effective_n;
            a.utilization_pct = limits_.min_effective_n / concentration.effective_n * 100.0;
            a.timestamp = now;
            new_alerts.push_back(a);
        }
    }

    if (limits_.max_hhi > 0 && concentration.hhi > limits_.max_hhi) {
        Alert a;
        a.severity = AlertSeverity::WARNING;
        a.category = "hhi";
        a.message = "HHI " + std::to_string(concentration.hhi) +
            " exceeds limit " + std::to_string(limits_.max_hhi);
        a.current_value = concentration.hhi;
        a.limit_value = limits_.max_hhi;
        a.utilization_pct = concentration.hhi / limits_.max_hhi * 100.0;
        a.timestamp = now;
        new_alerts.push_back(a);
    }

    // Store alerts
    alerts_ = new_alerts;
    return new_alerts;
}

bool RiskEngine::is_within_limits() const {
    auto alerts = check_all_limits();
    for (const auto& a : alerts) {
        if (a.severity == AlertSeverity::BREACH ||
            a.severity == AlertSeverity::CRITICAL) {
            return false;
        }
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Alert System
///////////////////////////////////////////////////////////////////////////////

std::vector<Alert> RiskEngine::get_active_alerts() const {
    return alerts_;
}

void RiskEngine::clear_alerts() {
    alerts_.clear();
}

///////////////////////////////////////////////////////////////////////////////
// Correlation Matrix (for MC)
///////////////////////////////////////////////////////////////////////////////

Mat RiskEngine::compute_correlation_matrix() const {
    if (asset_return_history_.empty()) return {};

    size_t n = asset_return_history_.size();
    size_t k = asset_return_history_[0].size();

    // Means
    Vec mu(k, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < k; ++j)
            mu[j] += asset_return_history_[t][j];
    for (size_t j = 0; j < k; ++j) mu[j] /= n;

    // Covariance
    Mat cov(k, Vec(k, 0.0));
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < k; ++i) {
            double di = asset_return_history_[t][i] - mu[i];
            for (size_t j = i; j < k; ++j) {
                double dj = asset_return_history_[t][j] - mu[j];
                cov[i][j] += di * dj;
            }
        }
    }

    // Correlation
    Mat corr(k, Vec(k, 0.0));
    for (size_t i = 0; i < k; ++i) {
        cov[i][i] /= (n - 1);
        for (size_t j = i + 1; j < k; ++j) {
            cov[i][j] /= (n - 1);
            cov[j][i] = cov[i][j];
        }
    }

    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < k; ++j) {
            double denom = std::sqrt(cov[i][i] * cov[j][j]);
            corr[i][j] = (denom > 1e-15) ? cov[i][j] / denom : 0.0;
        }
        corr[i][i] = 1.0;
    }

    return corr;
}

Mat RiskEngine::cholesky_decompose(const Mat& A) const {
    size_t n = A.size();
    Mat L(n, Vec(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += L[i][k] * L[j][k];
            if (i == j) {
                double d = A[i][i] - sum;
                L[i][j] = std::sqrt(std::max(d, 1e-10));
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return L;
}

///////////////////////////////////////////////////////////////////////////////
// Full Risk Report
///////////////////////////////////////////////////////////////////////////////

RiskEngine::RiskReport RiskEngine::generate_report() const {
    auto start = std::chrono::steady_clock::now();

    RiskReport report;
    report.var_95 = historical_var(0.95);
    report.var_99 = historical_var(0.99);
    report.cvar_95 = parametric_var_normal(0.95);
    report.greeks = aggregate_greeks();
    report.drawdown = get_drawdown_state();
    report.concentration = compute_concentration();
    report.liquidity = compute_liquidity();
    report.stress_results = run_all_stress_tests();
    report.alerts = check_all_limits();

    auto end = std::chrono::steady_clock::now();
    report.computation_time_us = static_cast<int>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    return report;
}

} // namespace risk
} // namespace signal_engine
