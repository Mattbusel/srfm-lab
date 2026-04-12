// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// risk_engine.cpp — Real-time risk management and position sizing
// =============================================================================
// Implements: VaR/CVaR, Kelly sizing, portfolio optimization,
// drawdown controls, margin/leverage limits, PnL attribution.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtel {
namespace risk {

static constexpr double kEps = 1e-12;

// ---------------------------------------------------------------------------
// Return series statistics
// ---------------------------------------------------------------------------
struct ReturnStats {
    double mean        = 0.0;
    double variance    = 0.0;
    double skewness    = 0.0;
    double kurtosis    = 0.0;
    double min         = 0.0;
    double max         = 0.0;
    double sharpe      = 0.0;
    double sortino     = 0.0;
    double max_drawdown= 0.0;
    double calmar      = 0.0;
    int    n           = 0;

    static ReturnStats compute(const std::vector<double>& rets,
                                double rfr_per_step = 0.0)
    {
        ReturnStats s;
        s.n = (int)rets.size();
        if (s.n < 2) return s;

        // Mean
        for (double r : rets) s.mean += r;
        s.mean /= s.n;
        s.min = *std::min_element(rets.begin(), rets.end());
        s.max = *std::max_element(rets.begin(), rets.end());

        // Higher moments
        double m2=0, m3=0, m4=0, downside=0;
        for (double r : rets) {
            double d = r - s.mean;
            m2 += d*d; m3 += d*d*d; m4 += d*d*d*d;
            double neg = std::min(r - rfr_per_step, 0.0);
            downside += neg*neg;
        }
        s.variance = m2 / (s.n - 1);
        double std_dev = std::sqrt(s.variance);
        if (std_dev > kEps) {
            s.skewness = (m3/s.n) / std::pow(std_dev, 3);
            s.kurtosis = (m4/s.n) / (s.variance*s.variance) - 3.0;
        }

        // Sharpe
        s.sharpe = (std_dev > kEps) ? (s.mean - rfr_per_step) / std_dev : 0.0;

        // Sortino
        double downside_std = std::sqrt(downside / s.n);
        s.sortino = (downside_std > kEps) ? (s.mean - rfr_per_step) / downside_std : 0.0;

        // Max drawdown
        double peak = 0.0, cum = 0.0;
        for (double r : rets) {
            cum  += r;
            peak  = std::max(peak, cum);
            double dd = peak - cum;
            s.max_drawdown = std::max(s.max_drawdown, dd);
        }

        // Calmar
        s.calmar = (s.max_drawdown > kEps) ? s.mean * s.n / s.max_drawdown : 0.0;

        return s;
    }
};

// ---------------------------------------------------------------------------
// VaR and CVaR
// ---------------------------------------------------------------------------
double var_historical(const std::vector<double>& sorted_rets,
                      double confidence = 0.95)
{
    if (sorted_rets.empty()) return 0.0;
    int idx = (int)((1.0 - confidence) * sorted_rets.size());
    idx = std::max(0, std::min(idx, (int)sorted_rets.size()-1));
    return -sorted_rets[idx];  // VaR as positive loss
}

double cvar_historical(const std::vector<double>& sorted_rets,
                       double confidence = 0.95)
{
    if (sorted_rets.empty()) return 0.0;
    int cutoff = (int)((1.0 - confidence) * sorted_rets.size());
    cutoff = std::max(1, std::min(cutoff, (int)sorted_rets.size()));
    double sum = 0.0;
    for (int i = 0; i < cutoff; ++i) sum += sorted_rets[i];
    return -sum / cutoff;
}

// Parametric VaR (normal distribution assumption)
double var_parametric(double mean, double std_dev, double confidence = 0.95) {
    // Inverse normal: for 95% -> z ≈ 1.645
    auto inv_norm = [](double p) -> double {
        // Beasley-Springer-Moro approximation
        static const double a[] = {2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637};
        static const double b[] = {-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833};
        static const double c[] = {0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
                                    0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
                                    0.0000321767881768, 0.0000002888167364, 0.0000003960315187};
        double x = p - 0.5;
        if (std::abs(x) < 0.42) {
            double r = x*x;
            return x*(((a[3]*r+a[2])*r+a[1])*r+a[0]) /
                      ((((b[3]*r+b[2])*r+b[1])*r+b[0])*r+1.0);
        }
        double r = (x > 0) ? std::log(-std::log(1.0-p)) : std::log(-std::log(p));
        double s = c[0]+r*(c[1]+r*(c[2]+r*(c[3]+r*(c[4]+r*(c[5]+r*(c[6]+r*(c[7]+r*c[8])))))));
        return (x < 0) ? -s : s;
    };
    double z = inv_norm(confidence);
    return -(mean - z * std_dev);
}

// ---------------------------------------------------------------------------
// Kelly criterion
// ---------------------------------------------------------------------------
double kelly_fraction(double win_prob, double win_loss_ratio) {
    // Kelly = p - q/b where b = win/loss ratio
    double q = 1.0 - win_prob;
    return win_prob - q / win_loss_ratio;
}

double kelly_fraction_continuous(double mean_ret, double var_ret) {
    // f* = mu / sigma^2
    return (var_ret > kEps) ? mean_ret / var_ret : 0.0;
}

// Fractional Kelly (scale by fraction to reduce variance)
double fractional_kelly(double full_kelly, double fraction = 0.5) {
    return std::max(0.0, full_kelly * fraction);
}

// ---------------------------------------------------------------------------
// Mean-variance optimization (single period)
// ---------------------------------------------------------------------------
struct MVOResult {
    std::vector<double> weights;
    double expected_return = 0.0;
    double expected_vol    = 0.0;
    double sharpe_ratio    = 0.0;
};

// Simple equal-risk-contribution portfolio
MVOResult equal_risk_contribution(const std::vector<double>& vols,
                                   const std::vector<std::vector<double>>& corr)
{
    int n = (int)vols.size();
    MVOResult result;
    result.weights.assign(n, 0.0);
    if (n == 0) return result;

    // Start with equal weights and iterate
    std::vector<double> w(n, 1.0/n);
    for (int iter = 0; iter < 100; ++iter) {
        // Compute covariance-weighted sum for each asset
        std::vector<double> mrc(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                mrc[i] += corr[i][j] * vols[i] * vols[j] * w[j];
            }
        }
        // Portfolio variance
        double pvar = 0.0;
        for (int i = 0; i < n; ++i) pvar += w[i] * mrc[i];
        if (pvar < kEps) break;

        // Risk contribution = w[i] * mrc[i] / pvar
        // Update weights to equalize risk contributions
        for (int i = 0; i < n; ++i) {
            double rc = (pvar > kEps) ? w[i] * mrc[i] / pvar : 1.0/n;
            w[i] *= (1.0/n) / (rc + kEps);
        }
        // Normalize
        double sum = std::accumulate(w.begin(), w.end(), 0.0);
        for (double& wi : w) wi /= sum;
    }
    result.weights = w;

    // Compute expected vol
    double pvar = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            pvar += w[i] * w[j] * corr[i][j] * vols[i] * vols[j];
    result.expected_vol = std::sqrt(std::max(0.0, pvar));

    return result;
}

// Maximum Sharpe Ratio portfolio (Lagrangian approximation)
MVOResult max_sharpe_portfolio(const std::vector<double>& expected_rets,
                                const std::vector<double>& vols,
                                const std::vector<std::vector<double>>& corr,
                                double rfr = 0.0)
{
    int n = (int)expected_rets.size();
    MVOResult result;
    result.weights.assign(n, 0.0);
    if (n == 0) return result;

    // Tangency portfolio: w ∝ Sigma^{-1} * (mu - rf)
    // Approximate Sigma^{-1} using diagonal (ignores correlation)
    std::vector<double> excess(n);
    std::vector<double> w(n);
    double sum_w = 0.0;
    for (int i = 0; i < n; ++i) {
        excess[i] = expected_rets[i] - rfr;
        double var_i = vols[i] * vols[i];
        w[i] = (var_i > kEps) ? excess[i] / var_i : 0.0;
        if (w[i] < 0) w[i] = 0.0;
        sum_w += w[i];
    }
    if (sum_w < kEps) {
        // Fallback to equal weight
        for (double& wi : w) wi = 1.0/n;
    } else {
        for (double& wi : w) wi /= sum_w;
    }
    result.weights = w;

    // Portfolio stats
    double pret = 0.0, pvar = 0.0;
    for (int i = 0; i < n; ++i) pret += w[i] * expected_rets[i];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            pvar += w[i]*w[j]*corr[i][j]*vols[i]*vols[j];
    result.expected_return = pret;
    result.expected_vol    = std::sqrt(std::max(0.0, pvar));
    result.sharpe_ratio    = (result.expected_vol > kEps) ?
        (pret - rfr) / result.expected_vol : 0.0;
    return result;
}

// ---------------------------------------------------------------------------
// Position limits and margin
// ---------------------------------------------------------------------------
struct PositionLimits {
    double max_position_usd;      // absolute dollar limit
    double max_leverage;          // max gross leverage
    double max_concentration;     // max single asset weight
    double max_drawdown_limit;    // stop at this drawdown
    double daily_var_limit;       // daily 95% VaR limit
    double margin_rate;           // haircut / margin rate

    static PositionLimits conservative() {
        return {1e6, 2.0, 0.10, 0.05, 0.02, 0.10};
    }
    static PositionLimits moderate() {
        return {5e6, 4.0, 0.20, 0.10, 0.05, 0.05};
    }
    static PositionLimits aggressive() {
        return {20e6, 8.0, 0.30, 0.20, 0.10, 0.03};
    }
};

// ---------------------------------------------------------------------------
// RiskManager
// ---------------------------------------------------------------------------
class RiskManager {
    PositionLimits limits_;
    double equity_             = 1e6;
    double peak_equity_        = 1e6;
    double current_drawdown_   = 0.0;
    double daily_pnl_          = 0.0;
    double gross_exposure_     = 0.0;

    std::unordered_map<int, double> positions_;  // asset_id -> dollar value
    std::vector<double> return_history_;

    // Rolling VaR window
    static constexpr int kVarWindow = 252;

public:
    explicit RiskManager(const PositionLimits& limits = PositionLimits::moderate(),
                         double initial_equity = 1e6)
        : limits_(limits), equity_(initial_equity), peak_equity_(initial_equity) {}

    // Check if new trade is allowed
    struct TradeCheck {
        bool allowed;
        std::string reason;
        double approved_size;
    };

    TradeCheck check_trade(int asset_id, double trade_value_usd) const {
        // Drawdown check
        if (current_drawdown_ > limits_.max_drawdown_limit) {
            return {false, "max drawdown breached", 0.0};
        }

        // Leverage check
        double new_gross = gross_exposure_ + std::abs(trade_value_usd);
        if (new_gross / equity_ > limits_.max_leverage) {
            double allowed = limits_.max_leverage * equity_ - gross_exposure_;
            allowed = std::max(0.0, std::min(std::abs(trade_value_usd), allowed));
            double adj = (trade_value_usd > 0) ? allowed : -allowed;
            return {true, "leverage limited", adj};
        }

        // Concentration check
        double current_pos = 0.0;
        auto it = positions_.find(asset_id);
        if (it != positions_.end()) current_pos = it->second;
        double new_pos = current_pos + trade_value_usd;
        if (std::abs(new_pos) / equity_ > limits_.max_concentration) {
            double max_pos = limits_.max_concentration * equity_;
            double adj = (trade_value_usd > 0) ?
                std::max(0.0, max_pos - current_pos) :
                std::min(0.0, -max_pos - current_pos);
            return {true, "concentration limited", adj};
        }

        // Position limit check
        if (std::abs(current_pos + trade_value_usd) > limits_.max_position_usd) {
            double max_abs = limits_.max_position_usd;
            double adj = std::clamp(trade_value_usd, -max_abs - current_pos, max_abs - current_pos);
            return {true, "position limit", adj};
        }

        return {true, "ok", trade_value_usd};
    }

    void apply_trade(int asset_id, double trade_value_usd) {
        positions_[asset_id] += trade_value_usd;
        gross_exposure_ = 0.0;
        for (auto& [id, pos] : positions_) gross_exposure_ += std::abs(pos);
    }

    void mark_pnl(double pnl) {
        equity_          += pnl;
        daily_pnl_       += pnl;
        peak_equity_      = std::max(peak_equity_, equity_);
        current_drawdown_ = (peak_equity_ > kEps) ?
            (peak_equity_ - equity_) / peak_equity_ : 0.0;
        if (!return_history_.empty() || pnl != 0.0) {
            double ret = (equity_ - pnl > kEps) ? pnl / (equity_ - pnl) : 0.0;
            return_history_.push_back(ret);
            if ((int)return_history_.size() > kVarWindow)
                return_history_.erase(return_history_.begin());
        }
    }

    void reset_daily() { daily_pnl_ = 0.0; }

    double equity()          const { return equity_; }
    double drawdown()        const { return current_drawdown_; }
    double leverage()        const { return (equity_ > kEps) ? gross_exposure_ / equity_ : 0.0; }
    double daily_pnl()       const { return daily_pnl_; }
    double gross_exposure()  const { return gross_exposure_; }

    double current_var(double confidence = 0.95) const {
        if (return_history_.size() < 10) return 0.0;
        std::vector<double> sorted = return_history_;
        std::sort(sorted.begin(), sorted.end());
        return var_historical(sorted, confidence) * equity_;
    }

    double current_cvar(double confidence = 0.95) const {
        if (return_history_.size() < 10) return 0.0;
        std::vector<double> sorted = return_history_;
        std::sort(sorted.begin(), sorted.end());
        return cvar_historical(sorted, confidence) * equity_;
    }

    bool is_risk_breach() const {
        return current_drawdown_ > limits_.max_drawdown_limit ||
               leverage() > limits_.max_leverage;
    }

    std::string risk_summary() const {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "equity=%.0f dd=%.2f%% lev=%.2fx daily_pnl=%.0f var95=%.0f",
            equity_, current_drawdown_*100, leverage(),
            daily_pnl_, current_var(0.95));
        return buf;
    }
};

// ---------------------------------------------------------------------------
// PositionSizer — converts signal strength to position sizes
// ---------------------------------------------------------------------------
class PositionSizer {
public:
    struct Config {
        double base_size_usd     = 10000.0;  // base position size
        double max_size_usd      = 100000.0;
        double kelly_fraction    = 0.25;     // fractional Kelly multiplier
        double signal_threshold  = 0.5;      // minimum |signal| to trade
        double vol_target        = 0.02;     // daily vol target (2%)
    };

    Config config;

    // Fixed fractional sizing
    double fixed_fractional(double signal, double equity) const {
        if (std::abs(signal) < config.signal_threshold) return 0.0;
        double fraction = std::clamp(std::abs(signal), 0.0, 1.0);
        double size = fraction * equity * 0.02;  // 2% of equity per unit signal
        return std::clamp(size, 0.0, config.max_size_usd) * (signal > 0 ? 1.0 : -1.0);
    }

    // Vol-scaled sizing (position size inversely proportional to vol)
    double vol_scaled(double signal, double vol, double equity) const {
        if (std::abs(signal) < config.signal_threshold || vol < kEps) return 0.0;
        double vol_factor = config.vol_target / vol;
        double size = std::abs(signal) * config.base_size_usd * vol_factor;
        size = std::clamp(size, 0.0, config.max_size_usd);
        return size * (signal > 0 ? 1.0 : -1.0);
    }

    // Kelly-sized position
    double kelly_sized(double signal, double win_prob, double vol, double equity) const {
        if (std::abs(signal) < config.signal_threshold) return 0.0;
        double full_k = kelly_fraction_continuous(
            signal * 0.001,   // rough mean return estimate
            vol * vol
        );
        double fk = fractional_kelly(full_k, config.kelly_fraction);
        double size = fk * equity;
        size = std::clamp(std::abs(size), 0.0, config.max_size_usd);
        return size * (signal > 0 ? 1.0 : -1.0);
    }
};

// ---------------------------------------------------------------------------
// PnL attribution
// ---------------------------------------------------------------------------
struct PnLAttribution {
    double total_pnl     = 0.0;
    double realized_pnl  = 0.0;
    double unrealized_pnl= 0.0;
    double trading_costs = 0.0;
    double financing_cost= 0.0;
    double net_pnl       = 0.0;

    // Factor attributions
    double market_beta_pnl  = 0.0;
    double factor1_pnl      = 0.0;
    double factor2_pnl      = 0.0;
    double alpha_pnl        = 0.0;

    std::string to_string() const {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "total=%.2f real=%.2f unreal=%.2f costs=%.2f alpha=%.2f",
            total_pnl, realized_pnl, unrealized_pnl, trading_costs, alpha_pnl);
        return buf;
    }
};

class PnLTracker {
    struct Trade {
        int asset_id;
        double entry_price;
        double quantity;
        double commission;
        int64_t timestamp;
    };

    std::unordered_map<int, std::vector<Trade>> open_trades_;
    double total_realized_   = 0.0;
    double total_commissions_= 0.0;

public:
    void open_trade(int asset_id, double price, double qty,
                    double commission = 0.0, int64_t ts = 0) {
        open_trades_[asset_id].push_back({asset_id, price, qty, commission, ts});
        total_commissions_ += commission;
    }

    double close_trade(int asset_id, double close_price, double qty) {
        auto it = open_trades_.find(asset_id);
        if (it == open_trades_.end() || it->second.empty()) return 0.0;

        // FIFO matching
        double pnl = 0.0;
        double remaining = std::abs(qty);
        auto& trades = it->second;
        while (remaining > kEps && !trades.empty()) {
            Trade& t = trades.front();
            double fill = std::min(remaining, std::abs(t.quantity));
            double sign = (t.quantity > 0) ? 1.0 : -1.0;
            pnl += sign * fill * (close_price - t.entry_price);
            t.quantity -= sign * fill;
            remaining  -= fill;
            if (std::abs(t.quantity) < kEps) trades.erase(trades.begin());
        }
        total_realized_ += pnl;
        return pnl;
    }

    double unrealized_pnl(const std::unordered_map<int, double>& current_prices) const {
        double upnl = 0.0;
        for (auto& [aid, trades] : open_trades_) {
            auto pit = current_prices.find(aid);
            if (pit == current_prices.end()) continue;
            for (auto& t : trades) {
                double sign = (t.quantity > 0) ? 1.0 : -1.0;
                upnl += sign * std::abs(t.quantity) * (pit->second - t.entry_price);
            }
        }
        return upnl;
    }

    double realized_pnl()    const { return total_realized_; }
    double total_commissions() const { return total_commissions_; }
};

// ---------------------------------------------------------------------------
// Slippage model
// ---------------------------------------------------------------------------
struct SlippageModel {
    double spread_half_bps;       // half-spread cost in bps
    double market_impact_bps_per_pct_adv; // impact per % of ADV

    // Estimate round-trip cost
    double estimate_cost_bps(double trade_size_usd, double adv_usd) const {
        double spread_cost = 2.0 * spread_half_bps;
        double pct_adv     = (adv_usd > kEps) ? trade_size_usd / adv_usd : 0.0;
        double impact      = market_impact_bps_per_pct_adv * pct_adv;
        return spread_cost + impact;
    }

    static SlippageModel liquid_large_cap() { return {1.0, 10.0}; }
    static SlippageModel illiquid_small_cap() { return {10.0, 100.0}; }
    static SlippageModel crypto_spot() { return {2.5, 30.0}; }
};

// ---------------------------------------------------------------------------
// RiskReporter — generates periodic risk reports
// ---------------------------------------------------------------------------
class RiskReporter {
    RiskManager& rm_;

public:
    explicit RiskReporter(RiskManager& rm) : rm_(rm) {}

    std::string generate_report() const {
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
            "=== AETERNUS Risk Report ===\n"
            "  Equity:          $%.2f\n"
            "  Gross Exposure:  $%.2f\n"
            "  Leverage:        %.2fx\n"
            "  Drawdown:        %.2f%%\n"
            "  Daily PnL:       $%.2f\n"
            "  95%% VaR:        $%.2f\n"
            "  Risk Breach:     %s\n",
            rm_.equity(),
            rm_.gross_exposure(),
            rm_.leverage(),
            rm_.drawdown() * 100.0,
            rm_.daily_pnl(),
            rm_.current_var(0.95),
            rm_.is_risk_breach() ? "YES" : "no"
        );
        return buf;
    }

    std::string prometheus_metrics() const {
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
            "rtel_risk_equity %.2f\n"
            "rtel_risk_leverage %.4f\n"
            "rtel_risk_drawdown %.6f\n"
            "rtel_risk_daily_pnl %.2f\n"
            "rtel_risk_var95 %.2f\n",
            rm_.equity(),
            rm_.leverage(),
            rm_.drawdown(),
            rm_.daily_pnl(),
            rm_.current_var(0.95)
        );
        return buf;
    }
};

// ---------------------------------------------------------------------------
// StressTest — scenario analysis
// ---------------------------------------------------------------------------
struct StressScenario {
    std::string name;
    std::vector<double> asset_shocks;  // per-asset price shock (fraction)
    double vol_multiplier = 1.0;
};

struct StressTestResult {
    std::string scenario_name;
    double portfolio_pnl;
    double portfolio_ret;
    bool   exceeds_var_limit;
};

StressTestResult run_stress_test(
    const StressScenario& scenario,
    const std::unordered_map<int, double>& positions,
    const std::unordered_map<int, double>& current_prices,
    double equity,
    double var_limit)
{
    double pnl = 0.0;
    for (auto& [aid, pos] : positions) {
        auto pit = current_prices.find(aid);
        if (pit == current_prices.end()) continue;
        double shock = (aid < (int)scenario.asset_shocks.size()) ?
            scenario.asset_shocks[aid] : 0.0;
        pnl += pos * shock;  // pos in $ value, shock is fractional
    }
    double ret = (equity > kEps) ? pnl / equity : 0.0;
    return {
        scenario.name,
        pnl,
        ret,
        std::abs(pnl) > var_limit
    };
}

static std::vector<StressScenario> standard_stress_scenarios(int n_assets) {
    return {
        {"market_crash_10pct",
         std::vector<double>(n_assets, -0.10), 3.0},
        {"market_crash_20pct",
         std::vector<double>(n_assets, -0.20), 5.0},
        {"vol_spike",
         std::vector<double>(n_assets, -0.05), 4.0},
        {"flash_crash",
         [&]() {
             std::vector<double> s(n_assets, -0.02);
             // First asset has extreme shock
             if (!s.empty()) s[0] = -0.15;
             return s;
         }(), 8.0}
    };
}

}  // namespace risk
}  // namespace rtel
