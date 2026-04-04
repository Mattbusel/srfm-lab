#pragma once
#include "order.hpp"
#include <vector>
#include <deque>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace hft {

// ---- Almgren-Chriss Optimal Execution ----
// Models the market impact of liquidating a position of size X
// over time horizon T with N time steps.
//
// Permanent impact:   g(v) = gamma * v
// Temporary impact:   h(v) = epsilon * sgn(v) + eta * v
//
// Parameters:
//   sigma   = daily volatility
//   gamma   = permanent impact coefficient
//   eta     = temporary impact coefficient
//   epsilon = bid-ask spread half-width
//   tau     = time step size (T/N)
//   lambda  = risk aversion parameter

struct AlmgrenChrissParams {
    double sigma;    // daily price volatility
    double gamma;    // permanent impact (price per unit)
    double eta;      // temporary impact (price per unit per unit time)
    double epsilon;  // fixed cost per transaction (half spread)
    double lambda;   // trader risk aversion
    double tau;      // time step in days
    int    N;        // number of time steps
};

struct ExecutionSchedule {
    std::vector<double> trade_list;    // shares to trade in each interval
    std::vector<double> inventory;     // inventory at each time step
    double expected_cost;              // E[cost]
    double variance_cost;              // Var[cost]
    double efficient_frontier_param;   // lambda used
};

class AlmgrenChriss {
public:
    explicit AlmgrenChriss(const AlmgrenChrissParams& p) : p_(p) {}

    // Compute optimal execution schedule for selling X shares over N steps
    ExecutionSchedule optimal_schedule(double X) const {
        const double& sigma   = p_.sigma;
        const double& gamma   = p_.gamma;
        const double& eta     = p_.eta;
        const double& lambda  = p_.lambda;
        const double& tau     = p_.tau;
        const int&    N       = p_.N;
        const double  T       = N * tau;

        // kappa^2 = lambda * sigma^2 / (eta * (1 - gamma*tau/(2*eta)))
        double tilde_eta   = eta - 0.5 * gamma * tau;
        if (tilde_eta <= 0) tilde_eta = 1e-10;
        double kappa2      = lambda * sigma * sigma / tilde_eta;
        double kappa       = std::sqrt(kappa2);

        ExecutionSchedule sched;
        sched.efficient_frontier_param = lambda;
        sched.trade_list.resize(N);
        sched.inventory.resize(N + 1);

        // x(t_j) = X * sinh(kappa*(T-t_j)) / sinh(kappa*T)
        for (int j = 0; j <= N; ++j) {
            double t = j * tau;
            double denom = std::sinh(kappa * T);
            if (denom < 1e-15) denom = 1e-15;
            sched.inventory[j] = X * std::sinh(kappa * (T - t)) / denom;
        }

        // Trade list: n_j = x(t_{j-1}) - x(t_j)
        for (int j = 0; j < N; ++j)
            sched.trade_list[j] = sched.inventory[j] - sched.inventory[j+1];

        // Expected cost
        double E = 0.0;
        double sinh_kT = std::sinh(kappa * T);
        if (sinh_kT < 1e-15) sinh_kT = 1e-15;
        double cosh_kT = std::cosh(kappa * T);
        E = p_.epsilon * X
          + 0.5 * gamma * X * X
          + (eta / tau) * X * X * (kappa * tau * std::cosh(kappa * tau))
            / (2.0 * sinh_kT)
            * (std::sinh(2.0 * kappa * T) / sinh_kT
               + 2.0 * kappa * T / sinh_kT - 2.0);
        sched.expected_cost = E;

        // Variance of cost
        double V = 0.5 * sigma * sigma * X * X / kappa2
                   * (std::cosh(kappa * T) / sinh_kT
                      * (kappa * T * std::cosh(kappa * T) / sinh_kT - 1.0)
                      + kappa * T / sinh_kT);
        sched.variance_cost = V;

        return sched;
    }

    // Efficient frontier: sweep lambda from lo to hi
    std::vector<std::pair<double,double>> efficient_frontier(
        double X, double lambda_lo, double lambda_hi, int steps) const
    {
        std::vector<std::pair<double,double>> frontier;
        frontier.reserve(steps);
        AlmgrenChrissParams p2 = p_;
        double step = (lambda_hi - lambda_lo) / (steps - 1);
        for (int i = 0; i < steps; ++i) {
            p2.lambda = lambda_lo + i * step;
            AlmgrenChriss ac2(p2);
            auto sched = ac2.optimal_schedule(X);
            frontier.emplace_back(sched.variance_cost, sched.expected_cost);
        }
        return frontier;
    }

private:
    AlmgrenChrissParams p_;
};

// ---- Kyle Lambda (price impact from order flow) ----
// Lambda = cov(delta_p, order_flow) / var(order_flow)
// Estimated from rolling window of price changes and signed order flow
class KyleLambda {
public:
    explicit KyleLambda(size_t window = 100) : window_(window) {}

    void update(double price_change, double signed_order_flow) {
        prices_.push_back(price_change);
        flows_.push_back(signed_order_flow);
        if (prices_.size() > window_) {
            prices_.pop_front();
            flows_.pop_front();
        }
    }

    double estimate() const {
        if (prices_.size() < 10) return 0.0;
        const size_t n = prices_.size();
        std::vector<double> dp(prices_.begin(), prices_.end());
        std::vector<double> of(flows_.begin(), flows_.end());

        double mean_dp = std::accumulate(dp.begin(), dp.end(), 0.0) / n;
        double mean_of = std::accumulate(of.begin(), of.end(), 0.0) / n;

        double cov = 0.0, var_of = 0.0;
        for (size_t i = 0; i < n; ++i) {
            cov    += (dp[i] - mean_dp) * (of[i] - mean_of);
            var_of += (of[i] - mean_of) * (of[i] - mean_of);
        }
        if (var_of < 1e-15) return 0.0;
        return cov / var_of;
    }

    size_t count() const noexcept { return prices_.size(); }

private:
    size_t window_;
    std::deque<double> prices_;
    std::deque<double> flows_;
};

// ---- Amihud Illiquidity Ratio ----
// ILLIQ = (1/T) * sum( |r_t| / Volume_t )
// Higher value → more illiquid (larger price impact per unit volume)
class AmihudIlliquidity {
public:
    explicit AmihudIlliquidity(size_t window = 252) : window_(window) {}

    // r = return for period, vol = dollar volume
    void update(double abs_return, double dollar_volume) {
        if (dollar_volume > 0)
            ratios_.push_back(abs_return / dollar_volume);
        else
            ratios_.push_back(0.0);
        if (ratios_.size() > window_) ratios_.pop_front();
    }

    double estimate() const {
        if (ratios_.empty()) return 0.0;
        return std::accumulate(ratios_.begin(), ratios_.end(), 0.0) / ratios_.size();
    }

    // Annualized illiquidity (assumes daily observations)
    double annualized() const { return estimate() * 1e6; } // scaled

    size_t count() const noexcept { return ratios_.size(); }

private:
    size_t window_;
    std::deque<double> ratios_;
};

// ---- Combined Market Impact Estimator ----
struct ImpactEstimate {
    double kyle_lambda;          // price per unit order flow
    double amihud_illiq;         // price per unit dollar volume
    double expected_slippage;    // for given trade size
    double market_impact_bps;    // basis points
};

class MarketImpactModel {
public:
    MarketImpactModel(size_t kyle_window = 100, size_t amihud_window = 252)
        : kyle_(kyle_window), amihud_(amihud_window) {}

    void on_trade(double price_before, double price_after,
                  double signed_flow, double dollar_volume) {
        double dp = price_after - price_before;
        double abs_ret = std::fabs(dp / (price_before > 0 ? price_before : 1.0));
        kyle_.update(dp, signed_flow);
        amihud_.update(abs_ret, dollar_volume);
    }

    ImpactEstimate estimate(double trade_size_shares, double price) const {
        ImpactEstimate ie{};
        ie.kyle_lambda   = kyle_.estimate();
        ie.amihud_illiq  = amihud_.estimate();
        // Expected slippage from Kyle model
        ie.expected_slippage = std::fabs(ie.kyle_lambda * trade_size_shares);
        // Convert to bps
        if (price > 0)
            ie.market_impact_bps = ie.expected_slippage / price * 10000.0;
        return ie;
    }

private:
    KyleLambda         kyle_;
    AmihudIlliquidity  amihud_;
};

} // namespace hft
