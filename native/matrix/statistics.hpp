#pragma once
// Financial statistics computed from return matrices.
// Rolling statistics, correlation, beta, factor models.

#include "matrix.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <deque>

namespace linalg {
namespace stats {

// ============================================================
// Rolling statistics (online, O(1) update)
// Welford's method for numerically stable variance
// ============================================================
struct RollingStats {
    size_t window;
    std::deque<double> buf;
    double sum   = 0.0;
    double sum_sq = 0.0;
    uint64_t count = 0;

    explicit RollingStats(size_t w) : window(w) {}

    void update(double x) {
        buf.push_back(x);
        sum    += x;
        sum_sq += x * x;
        ++count;
        if (buf.size() > window) {
            double old = buf.front(); buf.pop_front();
            sum    -= old;
            sum_sq -= old * old;
        }
    }

    double mean() const { return buf.empty() ? 0.0 : sum / buf.size(); }
    double variance() const {
        size_t n = buf.size();
        if (n < 2) return 0.0;
        return (sum_sq - sum * sum / n) / (n - 1);
    }
    double std_dev() const { return std::sqrt(variance()); }
    double min_val() const { return buf.empty() ? 0.0 : *std::min_element(buf.begin(), buf.end()); }
    double max_val() const { return buf.empty() ? 0.0 : *std::max_element(buf.begin(), buf.end()); }

    // Coefficient of variation
    double cv() const {
        double m = mean();
        return m != 0.0 ? std_dev() / std::fabs(m) : 0.0;
    }
};

// ============================================================
// Exponential Moving Average and Variance (EWMA)
// Used for GARCH-style volatility estimation
// ============================================================
struct EWMAStats {
    double alpha;      // decay factor
    double ema   = 0.0;
    double emvar = 0.0;
    bool   init  = false;

    explicit EWMAStats(double lambda = 0.94) : alpha(1.0 - lambda) {}

    void update(double x) {
        if (!init) {
            ema   = x;
            emvar = 0.0;
            init  = true;
            return;
        }
        double diff  = x - ema;
        ema   = alpha * x + (1.0 - alpha) * ema;
        emvar = (1.0 - alpha) * (emvar + alpha * diff * diff);
    }

    double ewma() const  { return ema; }
    double ewmvar() const { return emvar; }
    double ewmstd() const { return std::sqrt(emvar); }
};

// ============================================================
// Correlation matrix from returns
// ============================================================
MatrixD correlation_matrix(const MatrixD& cov) {
    if (!cov.is_square()) throw std::invalid_argument("corr: non-square");
    const size_t n = cov.rows();
    MatrixD corr(n, n);
    for (size_t i = 0; i < n; ++i) {
        double di = std::sqrt(cov(i,i));
        for (size_t j = 0; j < n; ++j) {
            double dj = std::sqrt(cov(j,j));
            corr(i,j) = (di > 0 && dj > 0) ? cov(i,j) / (di * dj) : 0.0;
        }
    }
    return corr;
}

// ============================================================
// Beta estimation: beta = cov(r_i, r_mkt) / var(r_mkt)
// ============================================================
std::vector<double> betas(const MatrixD& returns, int mkt_col = 0) {
    const size_t n = returns.cols();
    const size_t T = returns.rows();
    if (T < 2) return std::vector<double>(n, 1.0);

    // Market returns
    std::vector<double> mkt(T);
    for (size_t t = 0; t < T; ++t) mkt[t] = returns(t, mkt_col);
    double mkt_mean = std::accumulate(mkt.begin(), mkt.end(), 0.0) / T;
    double mkt_var  = 0.0;
    for (auto r : mkt) mkt_var += (r - mkt_mean) * (r - mkt_mean);
    mkt_var /= (T - 1);
    if (mkt_var < 1e-14) return std::vector<double>(n, 1.0);

    std::vector<double> result(n);
    for (size_t j = 0; j < n; ++j) {
        double mean_j = 0.0;
        for (size_t t = 0; t < T; ++t) mean_j += returns(t, j);
        mean_j /= T;

        double cov = 0.0;
        for (size_t t = 0; t < T; ++t)
            cov += (returns(t, j) - mean_j) * (mkt[t] - mkt_mean);
        cov /= (T - 1);
        result[j] = cov / mkt_var;
    }
    return result;
}

// ============================================================
// Sharpe Ratio and related statistics
// ============================================================
struct PerformanceStats {
    double total_return;
    double annualized_return;
    double annualized_vol;
    double sharpe_ratio;
    double sortino_ratio;   // uses downside deviation
    double calmar_ratio;    // annualized return / max drawdown
    double max_drawdown;
    double max_drawdown_duration; // in periods
    double win_rate;
    double profit_factor;   // gross wins / gross losses
    double avg_win;
    double avg_loss;
};

PerformanceStats compute_performance(const std::vector<double>& returns,
                                      double rf_rate = 0.0,
                                      double periods_per_year = 252.0)
{
    if (returns.empty()) return {};
    const size_t n = returns.size();

    // Total return
    double total = 1.0;
    for (auto r : returns) total *= (1.0 + r);
    double total_return = total - 1.0;

    // Annualized
    double ann_ret = std::pow(1.0 + total_return, periods_per_year / n) - 1.0;

    // Volatility
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / n;
    double var  = 0.0;
    double down_var = 0.0;
    for (auto r : returns) {
        var += (r - mean) * (r - mean);
        double under = std::min(r - rf_rate / periods_per_year, 0.0);
        down_var += under * under;
    }
    var      /= (n - 1);
    down_var /= (n - 1);
    double ann_vol  = std::sqrt(var) * std::sqrt(periods_per_year);
    double ann_down = std::sqrt(down_var) * std::sqrt(periods_per_year);

    // Sharpe / Sortino
    double sharpe  = ann_vol  > 0 ? (ann_ret - rf_rate) / ann_vol  : 0.0;
    double sortino = ann_down > 0 ? (ann_ret - rf_rate) / ann_down : 0.0;

    // Max drawdown
    double peak = 1.0, nav = 1.0;
    double max_dd = 0.0;
    double dd_dur = 0.0, max_dd_dur = 0.0;
    double dd_start = 0.0;
    for (size_t i = 0; i < n; ++i) {
        nav *= (1.0 + returns[i]);
        if (nav > peak) { peak = nav; dd_start = i; dd_dur = 0; }
        else {
            double dd = (peak - nav) / peak;
            dd_dur = i - dd_start;
            if (dd > max_dd) max_dd = dd;
            if (dd_dur > max_dd_dur) max_dd_dur = dd_dur;
        }
    }

    // Calmar
    double calmar = max_dd > 0 ? ann_ret / max_dd : 0.0;

    // Win/loss stats
    size_t wins = 0, losses = 0;
    double gross_wins = 0.0, gross_losses = 0.0;
    for (auto r : returns) {
        if (r > 0) { ++wins; gross_wins += r; }
        else if (r < 0) { ++losses; gross_losses += std::fabs(r); }
    }
    double win_rate      = n > 0 ? static_cast<double>(wins) / n : 0.0;
    double profit_factor = gross_losses > 0 ? gross_wins / gross_losses : 0.0;
    double avg_win  = wins   > 0 ? gross_wins   / wins   : 0.0;
    double avg_loss = losses > 0 ? gross_losses / losses : 0.0;

    return {total_return, ann_ret, ann_vol, sharpe, sortino,
            calmar, max_dd, max_dd_dur, win_rate, profit_factor, avg_win, avg_loss};
}

// ============================================================
// Value at Risk (VaR) and Expected Shortfall (CVaR)
// Historical simulation approach
// ============================================================
struct VaRResult {
    double var_1pct;   // 1% VaR
    double var_5pct;   // 5% VaR
    double cvar_1pct;  // Expected Shortfall at 1%
    double cvar_5pct;  // Expected Shortfall at 5%
};

VaRResult historical_var(std::vector<double> returns) {
    if (returns.empty()) return {};
    std::sort(returns.begin(), returns.end());
    const size_t n = returns.size();

    auto percentile = [&](double p) -> double {
        size_t idx = static_cast<size_t>(p * n);
        if (idx >= n) idx = n - 1;
        return returns[idx];
    };

    double var1 = -percentile(0.01);
    double var5 = -percentile(0.05);

    // CVaR: average of returns below VaR threshold
    auto cvar = [&](double threshold) -> double {
        double sum = 0.0; size_t cnt = 0;
        for (auto r : returns) {
            if (r < -threshold) { sum += r; ++cnt; }
        }
        return cnt > 0 ? -sum / cnt : threshold;
    };

    return {var1, var5, cvar(var1), cvar(var5)};
}

// Parametric VaR (assumes normal distribution)
VaRResult parametric_var(const std::vector<double>& returns) {
    if (returns.empty()) return {};
    const size_t n = returns.size();
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / n;
    double var  = 0.0;
    for (auto r : returns) var += (r - mean) * (r - mean);
    var = n > 1 ? var / (n-1) : 0;
    double sigma = std::sqrt(var);

    // z-scores: 1% = -2.326, 5% = -1.645
    double var1 = -(mean - 2.326 * sigma);
    double var5 = -(mean - 1.645 * sigma);
    // CVaR for normal: E[X | X < -VaR] = mu - sigma * phi(z) / (1-p)
    // phi(2.326) ≈ 0.0267, phi(1.645) ≈ 0.1031
    double cvar1 = -(mean - sigma * 0.0267 / 0.01);
    double cvar5 = -(mean - sigma * 0.1031 / 0.05);

    return {var1, var5, cvar1, cvar5};
}

// ============================================================
// Factor model: returns = alpha + beta * F + epsilon
// Simple OLS regression of asset returns on factor returns
// ============================================================
struct FactorModelResult {
    double alpha;           // Jensen's alpha
    double beta;            // factor loading
    double r_squared;       // coefficient of determination
    double tracking_error;  // std dev of residuals
    double information_ratio; // alpha / tracking_error
};

FactorModelResult single_factor_regression(
    const std::vector<double>& asset_returns,
    const std::vector<double>& factor_returns)
{
    if (asset_returns.size() != factor_returns.size() || asset_returns.size() < 3)
        return {};

    const size_t n = asset_returns.size();
    double mean_a = std::accumulate(asset_returns.begin(), asset_returns.end(), 0.0) / n;
    double mean_f = std::accumulate(factor_returns.begin(), factor_returns.end(), 0.0) / n;

    double cov_af = 0.0, var_f = 0.0, var_a = 0.0;
    for (size_t i = 0; i < n; ++i) {
        cov_af += (asset_returns[i] - mean_a) * (factor_returns[i] - mean_f);
        var_f  += (factor_returns[i] - mean_f) * (factor_returns[i] - mean_f);
        var_a  += (asset_returns[i] - mean_a) * (asset_returns[i] - mean_a);
    }
    cov_af /= (n-1); var_f /= (n-1); var_a /= (n-1);

    double beta  = var_f > 0 ? cov_af / var_f : 0.0;
    double alpha = mean_a - beta * mean_f;
    double r2    = var_a > 0 ? (beta * beta * var_f) / var_a : 0.0;

    // Tracking error (std dev of residuals)
    double te2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double resid = asset_returns[i] - (alpha + beta * factor_returns[i]);
        te2 += resid * resid;
    }
    te2 /= (n - 2);
    double te = std::sqrt(te2);
    double ir = te > 0 ? alpha / te : 0.0;

    // Annualize alpha and TE
    alpha *= 252.0;
    te    *= std::sqrt(252.0);
    ir     = te > 0 ? alpha / te : 0.0;

    return {alpha, beta, r2, te, ir};
}

} // namespace stats
} // namespace linalg
