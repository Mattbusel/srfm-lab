"""
PortfolioSimulator.jl — Advanced Portfolio Simulation

Covers:
  - Correlated GBM for N assets (Cholesky-based)
  - Regime-switching simulation: different mu/sigma/corr per regime
  - Jump-diffusion portfolio simulation (Merton jump model)
  - Rebalancing strategies: calendar, threshold, volatility-targeting
  - Transaction cost integration: bid-ask spread + market impact
  - Drawdown simulation: empirical distribution of drawdown paths
  - Wealth process under different utility functions
  - Sequence-of-returns risk analysis (retirement simulation)
  - Monte Carlo bootstrap: resample historical daily returns

Pure Julia stdlib only. No external dependencies.
"""
module PortfolioSimulator

using Statistics, LinearAlgebra, Random

export GBMSimulator, simulate_gbm, simulate_gbm_portfolio
export RegimeSwitchingSimulator, simulate_regime_portfolio
export JumpDiffusionSimulator, simulate_jump_portfolio
export RebalanceStrategy, CalendarRebalance, ThresholdRebalance, VolTargetRebalance
export apply_rebalance!, portfolio_return_path
export TransactionCostModel, apply_transaction_costs
export DrawdownDistribution, simulate_drawdown_paths
export WealthUtility, crra_utility, log_utility, mean_variance_utility
export sequence_of_returns_risk, safe_withdrawal_rate
export bootstrap_portfolio, historical_bootstrap
export run_portfolio_simulator_demo

# ─────────────────────────────────────────────────────────────
# 1. CORRELATED GBM SIMULATION
# ─────────────────────────────────────────────────────────────

"""
    GBMSimulator

Multi-asset Geometric Brownian Motion simulator.

Fields:
  mu    — drift vector (annualized)
  sigma — volatility vector (annualized)
  corr  — correlation matrix (N × N)
  S0    — initial prices (N-vector)
"""
struct GBMSimulator
    mu::Vector{Float64}
    sigma::Vector{Float64}
    corr::Matrix{Float64}
    S0::Vector{Float64}
    n_assets::Int
end

function GBMSimulator(mu::Vector{Float64}, sigma::Vector{Float64},
                       corr::Matrix{Float64}, S0::Vector{Float64})
    n = length(mu)
    GBMSimulator(mu, sigma, corr, S0, n)
end

"""Default equal-weight 4-asset BTC/ETH/SPY-like simulator."""
function GBMSimulator()
    GBMSimulator(
        [0.50, 0.40, 0.10, 0.05],    # annualized drifts
        [0.80, 0.90, 0.15, 0.05],    # annualized vols
        [1.0 0.8 0.3 -0.1;
         0.8 1.0 0.2 -0.1;
         0.3 0.2 1.0  0.2;
        -0.1 -0.1 0.2  1.0],
        [40_000.0, 2_500.0, 450.0, 100.0],
        4
    )
end

"""
    simulate_gbm(sim, T, n_steps, n_paths; rng=...) -> Array{Float64, 3}

Simulate correlated GBM paths.
Returns n_paths × n_assets × (n_steps+1) array.
"""
function simulate_gbm(sim::GBMSimulator, T::Float64, n_steps::Int,
                       n_paths::Int; rng=MersenneTwister(42))::Array{Float64,3}
    dt   = T / n_steps
    n    = sim.n_assets
    paths = zeros(n_paths, n, n_steps + 1)
    for p in 1:n_paths
        paths[p, :, 1] = copy(sim.S0)
    end

    # Cholesky decomposition for correlated returns
    L = cholesky(sim.corr + 1e-8*I).L

    for p in 1:n_paths
        for t in 1:n_steps
            z = L * randn(rng, n)
            for i in 1:n
                drift   = (sim.mu[i] - 0.5 * sim.sigma[i]^2) * dt
                diffuse = sim.sigma[i] * sqrt(dt) * z[i]
                paths[p, i, t+1] = paths[p, i, t] * exp(drift + diffuse)
            end
        end
    end
    paths
end

"""
    simulate_gbm_portfolio(sim, weights, T, n_steps, n_paths; rng=...)
       -> Matrix{Float64}

Simulate portfolio wealth path from GBM.
Returns n_paths × (n_steps+1) matrix of portfolio values.
"""
function simulate_gbm_portfolio(sim::GBMSimulator, weights::Vector{Float64},
                                  T::Float64, n_steps::Int, n_paths::Int;
                                  initial_wealth::Float64=100_000.0,
                                  rng=MersenneTwister(42))::Matrix{Float64}
    dt     = T / n_steps
    n      = sim.n_assets
    wealth = fill(initial_wealth, n_paths, n_steps + 1)
    L      = cholesky(sim.corr + 1e-8*I).L

    for p in 1:n_paths
        w_path = copy(weights)
        for t in 1:n_steps
            z    = L * randn(rng, n)
            rets = [(sim.mu[i] - 0.5*sim.sigma[i]^2)*dt +
                    sim.sigma[i]*sqrt(dt)*z[i] for i in 1:n]
            port_ret = dot(w_path, exp.(rets) .- 1.0)
            wealth[p, t+1] = wealth[p, t] * (1 + port_ret)
        end
    end
    wealth
end

# ─────────────────────────────────────────────────────────────
# 2. REGIME-SWITCHING SIMULATION
# ─────────────────────────────────────────────────────────────

"""
    RegimeSwitchingSimulator

Markov-switching simulator with K regimes.
Each regime has its own (mu, sigma, corr) parameters.
"""
struct RegimeSwitchingSimulator
    n_regimes::Int
    n_assets::Int
    mu::Vector{Vector{Float64}}       # n_regimes × n_assets
    sigma::Vector{Vector{Float64}}    # n_regimes × n_assets
    corr::Vector{Matrix{Float64}}     # n_regimes correlation matrices
    S0::Vector{Float64}
    transition_matrix::Matrix{Float64}  # n_regimes × n_regimes (row-stochastic)
    initial_regime::Int
end

function RegimeSwitchingSimulator(n_assets::Int=3)
    # 2 regimes: bull (1) and bear (2)
    RegimeSwitchingSimulator(
        2, n_assets,
        [[0.30, 0.25, 0.08], [-0.40, -0.50, -0.05]],
        [[0.50, 0.55, 0.12], [0.90, 1.10, 0.25]],
        [Matrix{Float64}(I, n_assets, n_assets) * 0.3 + fill(0.7, n_assets, n_assets),
         Matrix{Float64}(I, n_assets, n_assets) * 0.2 + fill(0.8, n_assets, n_assets)],
        fill(30_000.0, n_assets),
        [0.97 0.03; 0.12 0.88],  # regime persistence
        1
    )
end

"""
    simulate_regime_portfolio(sim, weights, T, n_steps, n_paths; rng=...)
       -> (wealth_paths, regime_paths)
"""
function simulate_regime_portfolio(sim::RegimeSwitchingSimulator,
                                    weights::Vector{Float64},
                                    T::Float64, n_steps::Int,
                                    n_paths::Int;
                                    initial_wealth::Float64=100_000.0,
                                    rng=MersenneTwister(42))
    dt     = T / n_steps
    n      = sim.n_assets
    wealth = fill(initial_wealth, n_paths, n_steps + 1)
    regimes = ones(Int, n_paths, n_steps + 1)

    # Precompute Cholesky per regime
    Ls = [cholesky(sim.corr[k] + 1e-8*I).L for k in 1:sim.n_regimes]
    # Precompute cumulative transition rows
    cum_T = [cumsum(sim.transition_matrix[k,:]) for k in 1:sim.n_regimes]

    for p in 1:n_paths
        regime = sim.initial_regime
        for t in 1:n_steps
            # Regime switching
            u = rand(rng)
            for k in 1:sim.n_regimes
                if u <= cum_T[regime][k]; regime = k; break; end
            end
            regimes[p, t+1] = regime

            z    = Ls[regime] * randn(rng, n)
            rets = [(sim.mu[regime][i] - 0.5*sim.sigma[regime][i]^2)*dt +
                     sim.sigma[regime][i]*sqrt(dt)*z[i] for i in 1:n]
            port_ret = dot(weights, exp.(rets) .- 1.0)
            wealth[p, t+1] = max(wealth[p, t] * (1 + port_ret), 1e-6)
        end
    end
    (wealth=wealth, regimes=regimes)
end

# ─────────────────────────────────────────────────────────────
# 3. JUMP-DIFFUSION
# ─────────────────────────────────────────────────────────────

"""
    JumpDiffusionSimulator

Merton jump-diffusion model for each asset.
dS/S = (μ - λk̄)dt + σdW + JdN
where J ~ lognormal, N ~ Poisson(λ).
"""
struct JumpDiffusionSimulator
    mu::Vector{Float64}
    sigma::Vector{Float64}
    corr::Matrix{Float64}
    S0::Vector{Float64}
    lambda::Vector{Float64}    # jump intensity (avg jumps per year)
    jump_mu::Vector{Float64}   # mean log-jump size
    jump_sigma::Vector{Float64} # std log-jump size
    n_assets::Int
end

function JumpDiffusionSimulator(n_assets::Int=3)
    JumpDiffusionSimulator(
        fill(0.15, n_assets), fill(0.40, n_assets),
        [1.0 0.7 0.3; 0.7 1.0 0.2; 0.3 0.2 1.0],
        fill(30_000.0, n_assets),
        fill(2.0, n_assets),          # ~2 jumps per year
        fill(-0.10, n_assets),        # avg jump = -10%
        fill(0.15, n_assets),         # jump size std = 15%
        n_assets
    )
end

"""
    simulate_jump_portfolio(sim, weights, T, n_steps, n_paths; rng=...)
       -> Matrix{Float64}
"""
function simulate_jump_portfolio(sim::JumpDiffusionSimulator,
                                   weights::Vector{Float64},
                                   T::Float64, n_steps::Int, n_paths::Int;
                                   initial_wealth::Float64=100_000.0,
                                   rng=MersenneTwister(42))::Matrix{Float64}
    dt = T / n_steps; n = sim.n_assets
    L  = cholesky(sim.corr + 1e-8*I).L
    wealth = fill(initial_wealth, n_paths, n_steps + 1)

    # Mean jump correction: k̄ = E[J-1] = exp(mu_J + 0.5*sig_J^2) - 1
    kbar = [exp(sim.jump_mu[i] + 0.5*sim.jump_sigma[i]^2) - 1.0 for i in 1:n]

    for p in 1:n_paths
        for t in 1:n_steps
            z    = L * randn(rng, n)
            rets = zeros(n)
            for i in 1:n
                drift    = (sim.mu[i] - sim.lambda[i]*kbar[i] - 0.5*sim.sigma[i]^2)*dt
                diffuse  = sim.sigma[i]*sqrt(dt)*z[i]
                # Jump component: Poisson number of jumps
                n_jumps  = Int(rand(rng) < sim.lambda[i]*dt ? 1 : 0)
                jump_ret = n_jumps > 0 ?
                           sum(sim.jump_mu[i] + sim.jump_sigma[i]*randn(rng)
                               for _ in 1:n_jumps; init=0.0) : 0.0
                rets[i] = drift + diffuse + jump_ret
            end
            port_ret = dot(weights, exp.(rets) .- 1.0)
            wealth[p, t+1] = max(wealth[p, t] * (1 + port_ret), 1e-6)
        end
    end
    wealth
end

# ─────────────────────────────────────────────────────────────
# 4. REBALANCING STRATEGIES
# ─────────────────────────────────────────────────────────────

"""
    RebalanceStrategy

Abstract type for rebalancing strategies.
"""
abstract type RebalanceStrategy end

"""
    CalendarRebalance

Rebalance on a fixed schedule (daily, monthly, quarterly, annual).
"""
struct CalendarRebalance <: RebalanceStrategy
    frequency::Int   # steps between rebalancing
    target_weights::Vector{Float64}
end

"""
    ThresholdRebalance

Rebalance when any weight drifts beyond a threshold from target.
"""
struct ThresholdRebalance <: RebalanceStrategy
    target_weights::Vector{Float64}
    threshold::Float64   # rebalance when |w_i - target_i| > threshold
end

"""
    VolTargetRebalance

Volatility-targeting rebalance: scale position to hit target volatility.
"""
struct VolTargetRebalance <: RebalanceStrategy
    base_weights::Vector{Float64}
    target_vol::Float64       # annualized target vol
    vol_window::Int           # lookback for vol estimation
    max_leverage::Float64
end

"""
    apply_rebalance!(weights, strategy, step, return_history) -> Bool

Returns true if rebalance was triggered, updates weights in-place.
"""
function apply_rebalance!(weights::Vector{Float64},
                           strat::CalendarRebalance,
                           step::Int, return_history=nothing)::Bool
    if step % strat.frequency == 0
        weights .= strat.target_weights
        return true
    end
    false
end

function apply_rebalance!(weights::Vector{Float64},
                           strat::ThresholdRebalance,
                           step::Int, return_history=nothing)::Bool
    max_drift = maximum(abs.(weights .- strat.target_weights))
    if max_drift > strat.threshold
        weights .= strat.target_weights
        return true
    end
    false
end

function apply_rebalance!(weights::Vector{Float64},
                           strat::VolTargetRebalance,
                           step::Int, return_history)::Bool
    step < strat.vol_window + 1 && return false
    # Estimate portfolio vol from history
    n = strat.vol_window
    hist_rets = return_history[end-n+1:end]
    realized_vol = std(hist_rets) * sqrt(252)
    realized_vol < 1e-6 && return false
    # Scale weights
    scale = clamp(strat.target_vol / realized_vol, 0.1, strat.max_leverage)
    weights .= strat.base_weights .* scale
    norm_w = sum(abs.(weights))
    norm_w > 0 && (weights ./= norm_w)
    true
end

"""
    portfolio_return_path(returns_matrix, initial_weights, strategy, tc; rng=...)
       -> NamedTuple

Simulate portfolio return path with rebalancing and transaction costs.
returns_matrix: T × N matrix of asset log-returns.
"""
function portfolio_return_path(returns_matrix::Matrix{Float64},
                                 initial_weights::Vector{Float64},
                                 strategy::RebalanceStrategy,
                                 tc::Float64=0.001)
    T, N = size(returns_matrix)
    weights  = copy(initial_weights)
    portfolio = [1.0]
    portfolio_returns = Float64[]
    rebalance_count  = 0
    tc_total         = 0.0
    weight_history   = zeros(T, N)
    weight_history[1, :] = weights

    for t in 1:T
        # Daily asset returns
        asset_rets = returns_matrix[t, :]
        port_ret   = dot(weights, asset_rets)

        # Update weights based on returns (drift)
        w_new = weights .* exp.(asset_rets)
        w_sum = sum(w_new); w_new ./= (w_sum + 1e-10)

        # Transaction cost for this period
        tc_period = 0.0

        # Check if rebalance needed
        if apply_rebalance!(w_new, strategy, t, portfolio_returns)
            trades    = w_new .- weights
            tc_period = sum(abs.(trades)) * tc
            rebalance_count += 1
        end
        weights = w_new
        weight_history[t, :] = weights

        # Portfolio return after TC
        effective_ret = port_ret - tc_period
        push!(portfolio, portfolio[end] * exp(effective_ret))
        push!(portfolio_returns, effective_ret)
        tc_total += tc_period
    end

    total_ret = (portfolio[end] - 1.0)
    annual_ret = total_ret * (252 / T)
    annual_vol = std(portfolio_returns) * sqrt(252)
    sharpe  = annual_vol > 1e-10 ? annual_ret / annual_vol : 0.0
    peak    = portfolio[1]; mdd = 0.0
    for pv in portfolio; peak = max(peak, pv); mdd = max(mdd, (peak-pv)/peak); end

    (portfolio=portfolio, returns=portfolio_returns, weights=weight_history,
     total_return=total_ret, annualized_sharpe=sharpe, max_drawdown=mdd,
     rebalance_count=rebalance_count, total_tc=tc_total)
end

# ─────────────────────────────────────────────────────────────
# 5. TRANSACTION COST MODEL
# ─────────────────────────────────────────────────────────────

"""
    TransactionCostModel

Comprehensive transaction cost model including:
  - Fixed commission
  - Proportional bid-ask spread
  - Market impact (square-root)
"""
struct TransactionCostModel
    fixed_commission::Float64   # per trade (USD)
    spread_bps::Float64         # half-spread in basis points
    impact_coeff::Float64       # phi for sqrt impact
    avg_daily_volume::Float64   # shares/USD for normalizing impact
end

TransactionCostModel(; spread_bps=5.0, impact=0.3, adv=1e6) =
    TransactionCostModel(0.0, spread_bps, impact, adv)

"""
    apply_transaction_costs(tc_model, trade_size, price, sigma) -> Float64

Compute total transaction cost for a trade.
Returns cost as fraction of trade value.
"""
function apply_transaction_costs(tc::TransactionCostModel,
                                   trade_size::Float64,
                                   price::Float64,
                                   sigma::Float64)::Float64
    trade_value = abs(trade_size) * price
    # Spread cost
    spread_cost = tc.spread_bps / 10_000 * trade_value
    # Market impact
    participation = abs(trade_size) / max(tc.avg_daily_volume, 1.0)
    impact_cost   = tc.impact_coeff * sigma * sqrt(participation) * trade_value
    # Fixed
    fixed_cost = tc.fixed_commission

    total = spread_cost + impact_cost + fixed_cost
    trade_value > 0 ? total / trade_value : 0.0
end

# ─────────────────────────────────────────────────────────────
# 6. DRAWDOWN SIMULATION
# ─────────────────────────────────────────────────────────────

"""
    DrawdownDistribution

Empirical distribution of drawdown paths from Monte Carlo.
"""
struct DrawdownDistribution
    max_drawdowns::Vector{Float64}
    avg_duration::Vector{Float64}  # avg drawdown duration in steps
    recovery_times::Vector{Float64}
    ulcer_indices::Vector{Float64}
end

"""
    simulate_drawdown_paths(wealth_paths) -> DrawdownDistribution

Compute drawdown statistics across Monte Carlo paths.
"""
function simulate_drawdown_paths(wealth_paths::Matrix{Float64})::DrawdownDistribution
    n_paths, T = size(wealth_paths)
    max_dds  = zeros(n_paths)
    avg_durs = zeros(n_paths)
    rec_times = zeros(n_paths)
    ulcer    = zeros(n_paths)

    for p in 1:n_paths
        pv   = wealth_paths[p, :]
        peak = pv[1]; max_dd = 0.0; in_dd = false
        dd_start = 1; total_dd_dur = 0; n_dds = 0
        ui   = 0.0

        for t in 1:T
            peak = max(peak, pv[t])
            dd   = (peak - pv[t]) / (peak + 1e-10)
            max_dd = max(max_dd, dd)
            ui += dd^2

            if dd > 0.01 && !in_dd
                in_dd = true; dd_start = t; n_dds += 1
            elseif dd < 0.005 && in_dd
                in_dd = false
                total_dd_dur += (t - dd_start)
            end
        end
        max_dds[p]  = max_dd
        avg_durs[p] = n_dds > 0 ? total_dd_dur / n_dds : 0.0
        ulcer[p]    = sqrt(ui / T)
        # Recovery time: steps at peak (approx)
        last_peak_t = findlast(pv .== maximum(pv))
        rec_times[p] = isnothing(last_peak_t) ? Float64(T) : Float64(last_peak_t)
    end

    DrawdownDistribution(max_dds, avg_durs, rec_times, ulcer)
end

"""
    drawdown_var(dd_dist, confidence) -> Float64

VaR of maximum drawdown at given confidence level.
"""
function drawdown_var(dd_dist::DrawdownDistribution, confidence::Float64=0.95)::Float64
    sorted = sort(dd_dist.max_drawdowns)
    idx = Int(ceil(confidence * length(sorted)))
    sorted[min(idx, end)]
end

# ─────────────────────────────────────────────────────────────
# 7. UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────

"""
    crra_utility(wealth, gamma) -> Float64

Constant Relative Risk Aversion utility.
U(W) = W^(1-γ) / (1-γ)  for γ ≠ 1
U(W) = ln(W)              for γ = 1
"""
function crra_utility(wealth::Float64, gamma::Float64)::Float64
    wealth <= 0 && return -1e10
    abs(gamma - 1.0) < 1e-6 ? log(wealth) :
                               wealth^(1-gamma) / (1-gamma)
end

"""
    log_utility(wealth) -> Float64

Log utility = CRRA with γ=1. Equivalent to maximizing geometric growth.
"""
log_utility(wealth::Float64) = wealth > 0 ? log(wealth) : -1e10

"""
    mean_variance_utility(ret, var, risk_aversion) -> Float64

Mean-variance utility: U = E[R] - (λ/2)*Var[R].
"""
mean_variance_utility(ret::Float64, var::Float64, lam::Float64) = ret - 0.5*lam*var

"""
    expected_utility(wealth_paths, gamma) -> Float64

Expected utility over Monte Carlo wealth paths at terminal date.
"""
function expected_utility(wealth_paths::Matrix{Float64}, gamma::Float64)::Float64
    terminal_wealth = wealth_paths[:, end]
    mean(crra_utility.(terminal_wealth, gamma))
end

"""
    optimal_kelly_fraction(mu, sigma; risk_free=0.0) -> Float64

Kelly criterion: fraction of wealth to allocate to risky asset.
f* = (μ - r) / σ² for log-utility investor.
"""
optimal_kelly_fraction(mu::Float64, sigma::Float64; rf::Float64=0.0) =
    sigma > 0 ? (mu - rf) / sigma^2 : 0.0

"""
    fractional_kelly(mu, sigma, fraction; rf=0.0) -> Float64

Fractional Kelly with safety factor (typically 0.25 - 0.50).
"""
fractional_kelly(mu::Float64, sigma::Float64, fraction::Float64; rf::Float64=0.0) =
    fraction * optimal_kelly_fraction(mu, sigma; rf=rf)

# ─────────────────────────────────────────────────────────────
# 8. SEQUENCE-OF-RETURNS RISK
# ─────────────────────────────────────────────────────────────

"""
    sequence_of_returns_risk(wealth_paths, withdrawal_rate, n_years)
       -> NamedTuple

Simulate sequence-of-returns risk for retirement/withdrawal scenarios.
withdrawal_rate: annual fraction of initial wealth withdrawn.
"""
function sequence_of_returns_risk(wealth_paths::Matrix{Float64},
                                    withdrawal_rate::Float64=0.04,
                                    n_years::Int=30)
    n_paths, T = size(wealth_paths)
    initial_wealth = wealth_paths[1, 1]
    annual_withdrawal = withdrawal_rate * initial_wealth

    # Steps per year (assume T = n_years * 252)
    steps_per_year = max(1, T ÷ max(n_years, 1))

    ruin_paths = 0
    final_wealth = zeros(n_paths)
    remaining_after_30y = zeros(n_paths)

    for p in 1:n_paths
        W = wealth_paths[p, 1]
        ruined = false
        for yr in 1:n_years
            # Annual return from wealth path
            t_end = min(yr * steps_per_year, T)
            t_beg = min((yr-1) * steps_per_year + 1, T)
            port_growth = wealth_paths[p, t_end] / max(wealth_paths[p, t_beg], 1e-6)
            W *= port_growth
            W -= annual_withdrawal
            if W <= 0
                ruined = true; ruin_paths += 1; break
            end
        end
        final_wealth[p] = ruined ? 0.0 : W
        remaining_after_30y[p] = final_wealth[p] / initial_wealth
    end

    ruin_prob = ruin_paths / n_paths
    median_final = median(final_wealth)
    (ruin_probability=ruin_prob, median_final_wealth=median_final,
     final_wealth_dist=final_wealth,
     pct_10=quantile(final_wealth, 0.10),
     pct_90=quantile(final_wealth, 0.90))
end

"""
    safe_withdrawal_rate(wealth_paths; max_ruin_prob=0.05) -> Float64

Find the safe withdrawal rate that keeps ruin probability below threshold.
"""
function safe_withdrawal_rate(wealth_paths::Matrix{Float64};
                               max_ruin_prob::Float64=0.05,
                               n_years::Int=30)::Float64
    for wr in 0.01:0.005:0.15
        result = sequence_of_returns_risk(wealth_paths, wr, n_years)
        result.ruin_probability <= max_ruin_prob && return wr
    end
    0.01
end

# ─────────────────────────────────────────────────────────────
# 9. BOOTSTRAP PORTFOLIO SIMULATION
# ─────────────────────────────────────────────────────────────

"""
    bootstrap_portfolio(historical_returns, weights, n_paths, n_steps;
                         block_size=20, rng=...) -> Matrix{Float64}

Block bootstrap portfolio simulation.
Resamples blocks of consecutive returns from historical data.
"""
function bootstrap_portfolio(historical_returns::Matrix{Float64},
                               weights::Vector{Float64},
                               n_paths::Int, n_steps::Int;
                               block_size::Int=20,
                               initial_wealth::Float64=100_000.0,
                               rng=MersenneTwister(42))::Matrix{Float64}
    T_hist, N = size(historical_returns)
    wealth    = fill(initial_wealth, n_paths, n_steps + 1)

    for p in 1:n_paths
        t = 1
        while t <= n_steps
            # Sample a random starting block
            start = rand(rng, 1:max(1, T_hist - block_size + 1))
            block = historical_returns[start:min(start+block_size-1, T_hist), :]
            for step_in_block in 1:size(block, 1)
                t > n_steps && break
                ret = dot(weights, block[step_in_block, :])
                wealth[p, t+1] = max(wealth[p, t] * exp(ret), 1e-6)
                t += 1
            end
        end
    end
    wealth
end

"""
    historical_bootstrap(returns, n_paths, n_years; rng=...) -> Matrix{Float64}

Simple i.i.d. bootstrap: resample daily returns with replacement.
"""
function historical_bootstrap(portfolio_returns::Vector{Float64},
                                n_paths::Int, n_years::Int;
                                initial_wealth::Float64=100_000.0,
                                rng=MersenneTwister(42))::Matrix{Float64}
    n_steps = n_years * 252
    T_hist  = length(portfolio_returns)
    wealth  = fill(initial_wealth, n_paths, n_steps + 1)
    for p in 1:n_paths
        for t in 1:n_steps
            r = portfolio_returns[rand(rng, 1:T_hist)]
            wealth[p, t+1] = max(wealth[p, t] * exp(r), 1e-6)
        end
    end
    wealth
end

# ─────────────────────────────────────────────────────────────
# 10. DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_portfolio_simulator_demo() -> Nothing
"""
function run_portfolio_simulator_demo()
    println("=" ^ 60)
    println("ADVANCED PORTFOLIO SIMULATOR DEMO")
    println("=" ^ 60)
    rng = MersenneTwister(42)
    n_paths = 200; T_years = 2.0; n_steps = Int(T_years * 252)
    weights = [0.4, 0.3, 0.2, 0.1]

    println("\n1. Correlated GBM Portfolio Simulation")
    sim = GBMSimulator()
    wealth_gbm = simulate_gbm_portfolio(sim, weights, T_years, n_steps, n_paths;
                                         rng=rng)
    terminal   = wealth_gbm[:, end]
    println("  Paths: $n_paths, Steps: $n_steps")
    println("  Median terminal wealth: \$$(round(median(terminal),digits=0))")
    println("  5th percentile:         \$$(round(quantile(terminal,0.05),digits=0))")
    println("  95th percentile:        \$$(round(quantile(terminal,0.95),digits=0))")

    println("\n2. Drawdown Analysis")
    dd = simulate_drawdown_paths(wealth_gbm)
    println("  Mean max drawdown: $(round(mean(dd.max_drawdowns)*100,digits=2))%")
    println("  95th pct DD:       $(round(drawdown_var(dd,0.95)*100,digits=2))%")
    println("  Mean ulcer index:  $(round(mean(dd.ulcer_indices)*100,digits=3))%")

    println("\n3. Regime-Switching Portfolio")
    rs_sim = RegimeSwitchingSimulator(3)
    wts3   = [0.5, 0.3, 0.2]
    rs_result = simulate_regime_portfolio(rs_sim, wts3, T_years, n_steps, n_paths; rng=rng)
    term_rs = rs_result.wealth[:, end]
    # Regime occupancy
    regime_occ = [mean(rs_result.regimes .== k) for k in 1:2]
    println("  Median wealth: \$$(round(median(term_rs),digits=0))")
    println("  Regime occupancy: Bull=$(round(regime_occ[1]*100,digits=1))%, Bear=$(round(regime_occ[2]*100,digits=1))%")

    println("\n4. Jump-Diffusion Portfolio")
    jd_sim = JumpDiffusionSimulator(3)
    wealth_jd = simulate_jump_portfolio(jd_sim, wts3, T_years, n_steps, n_paths; rng=rng)
    term_jd = wealth_jd[:, end]
    println("  Median wealth:       \$$(round(median(term_jd),digits=0))")
    println("  Crash risk (P < 50k): $(round(mean(term_jd .< 50_000)*100,digits=1))%")

    println("\n5. Rebalancing Strategies")
    # Generate synthetic returns
    raw_returns = sim.sigma' .* randn(rng, n_steps, sim.n_assets) .* (1/sqrt(252))
    for (name, strat) in [
            ("Annual Calendar", CalendarRebalance(252, weights)),
            ("5% Threshold",    ThresholdRebalance(weights, 0.05)),
            ("VolTarget 15%",   VolTargetRebalance(weights, 0.15, 21, 2.0))]
        result = portfolio_return_path(raw_returns, copy(weights), strat, 0.001)
        println("  $name: Return=$(round(result.total_return*100,digits=2))%, Sharpe=$(round(result.annualized_sharpe,digits=3)), Rebalances=$(result.rebalance_count)")
    end

    println("\n6. CRRA Utility Analysis")
    for gamma in [0.5, 1.0, 2.0, 5.0]
        eu = expected_utility(wealth_gbm, gamma)
        println("  γ=$gamma: E[U] = $(round(eu,digits=4))")
    end
    mu_gbm = dot(weights, sim.mu); sig_gbm = dot(weights.^2, sim.sigma.^2)
    kelly = optimal_kelly_fraction(mu_gbm, sqrt(sig_gbm))
    println("  Kelly fraction (full): $(round(kelly,digits=3))")
    println("  Half Kelly:            $(round(kelly*0.5,digits=3))")

    println("\n7. Sequence-of-Returns Risk")
    sor = sequence_of_returns_risk(wealth_gbm, 0.04, 10)
    println("  4% withdrawal rate, 10-year horizon:")
    println("  Ruin probability: $(round(sor.ruin_probability*100,digits=1))%")
    println("  Median final wealth: \$$(round(sor.median_final_wealth,digits=0))")
    println("  10th pct final: \$$(round(sor.pct_10,digits=0))")

    println("\n8. Historical Bootstrap")
    hist_rets = [dot(weights, raw_returns[t,:]) for t in 1:n_steps]
    boot_wealth = historical_bootstrap(hist_rets, 100, 2; rng=rng)
    boot_term = boot_wealth[:, end]
    println("  Bootstrap median: \$$(round(median(boot_term),digits=0))")
    println("  Bootstrap std:    \$$(round(std(boot_term),digits=0))")

    println("\nDone.")
    nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Variance-Gamma and Heston Stochastic Vol Simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    VarianceGammaParams

Variance-Gamma process: S_T = S_0 * exp((r + ω)T + σ W_{G(T)} + θ G(T))
where G(T) ~ Gamma(T/ν, ν) is a Gamma subordinator.
  - theta: drift of Brownian motion
  - sigma: vol of Brownian motion
  - nu:    variance of Gamma increments (controls kurtosis)
"""
struct VarianceGammaParams
    S0::Float64; r::Float64; theta::Float64; sigma::Float64; nu::Float64
end

"""
    simulate_variance_gamma(params, T, n_steps, n_paths) -> Matrix{Float64}

Returns price matrix (n_steps+1) × n_paths.
"""
function simulate_variance_gamma(p::VarianceGammaParams,
                                  T::Float64, n_steps::Int, n_paths::Int)
    dt   = T / n_steps
    omega = (1.0 / p.nu) * log(1.0 - p.theta * p.nu - 0.5 * p.sigma^2 * p.nu)
    prices = zeros(n_steps + 1, n_paths)
    prices[1, :] .= p.S0
    # Gamma increments: shape = dt/nu, scale = nu
    shape = dt / p.nu; scale = p.nu
    for path in 1:n_paths
        log_s = log(p.S0)
        for step in 1:n_steps
            # gamma variate via Marsaglia-Tsang (approximation via sum of exponentials)
            g = gamma_sample(shape, scale)
            z = randn()
            dX = p.theta * g + p.sigma * sqrt(g) * z
            log_s += (p.r + omega) * dt + dX
            prices[step + 1, path] = exp(log_s)
        end
    end
    return prices
end

"""
    gamma_sample(shape, scale)

Sample from Gamma(shape, scale) via Marsaglia-Tsang (shape ≥ 1) or
Ahrens-Dieter boost (shape < 1) — pure stdlib.
"""
function gamma_sample(shape::Float64, scale::Float64)
    if shape < 1.0
        # boost: Gamma(α) = Gamma(α+1) * U^(1/α)
        return gamma_sample(shape + 1.0, scale) * (rand() ^ (1.0 / shape))
    end
    # Marsaglia-Tsang method
    d = shape - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    while true
        x = randn(); v = (1.0 + c * x)^3
        if v > 0.0
            u = rand()
            if u < 1.0 - 0.0331 * x^4
                return d * v * scale
            end
            if log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
                return d * v * scale
            end
        end
    end
end

"""
    HestonParams

Heston stochastic volatility model:
  dS = r S dt + √V S dW_S
  dV = κ(θ_v − V) dt + ξ √V dW_V
  corr(dW_S, dW_V) = ρ
"""
struct HestonParams
    S0::Float64; V0::Float64; r::Float64
    kappa::Float64; theta_v::Float64; xi::Float64; rho::Float64
end

"""
    simulate_heston(params, T, n_steps, n_paths) -> (prices, vols)

Full-truncation Euler discretisation of the Heston SDE.
Returns price matrix and variance matrix, each (n_steps+1) × n_paths.
"""
function simulate_heston(p::HestonParams, T::Float64, n_steps::Int, n_paths::Int)
    dt    = T / n_steps
    sqdt  = sqrt(dt)
    prices = zeros(n_steps+1, n_paths); vols = zeros(n_steps+1, n_paths)
    prices[1, :] .= p.S0; vols[1, :] .= p.V0
    for path in 1:n_paths
        S = p.S0; V = p.V0
        for step in 1:n_steps
            z1 = randn(); z2 = randn()
            w_S = z1
            w_V = p.rho * z1 + sqrt(1 - p.rho^2) * z2
            V_pos = max(0.0, V)
            S_new = S * exp((p.r - 0.5 * V_pos) * dt + sqrt(V_pos) * sqdt * w_S)
            V_new = V + p.kappa * (p.theta_v - V_pos) * dt +
                    p.xi * sqrt(V_pos) * sqdt * w_V
            S = S_new; V = V_new
            prices[step+1, path] = S
            vols[step+1, path]   = V
        end
    end
    return prices, vols
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 10 – Scenario Analysis and Stress Testing
# ─────────────────────────────────────────────────────────────────────────────

"""
    ScenarioSpec

A deterministic stress scenario applied as multiplicative shocks to prices
and correlations.
Fields:
- name:    description of scenario (e.g., "2020 COVID crash")
- shocks:  vector of multiplicative price shocks per asset
- vol_mult: volatility multiplier
- corr_bump: correlation increase (additive, clamped to [-1,1])
"""
struct ScenarioSpec
    name::String
    shocks::Vector{Float64}
    vol_mult::Float64
    corr_bump::Float64
end

function apply_scenario(returns::Matrix{Float64}, spec::ScenarioSpec)
    n_assets = size(returns, 2)
    n_shocks = min(length(spec.shocks), n_assets)
    shocked = copy(returns)
    shocked[:, 1:n_shocks] .*= spec.shocks[1:n_shocks]'
    shocked .*= spec.vol_mult
    return shocked
end

"""
    historical_scenario_pnl(weights, returns, scenario_idx, window)

Apply a historical window of returns as a scenario to portfolio `weights`.
`scenario_idx` is the start of the window.
"""
function historical_scenario_pnl(weights::Vector{Float64},
                                   returns::Matrix{Float64},
                                   scenario_idx::Int, window::Int=20)
    T = size(returns, 1)
    stop = min(scenario_idx + window - 1, T)
    w = copy(weights); w ./= sum(abs.(w))
    pnl = Float64[]
    for t in scenario_idx:stop
        push!(pnl, dot(w, returns[t, :]))
    end
    return pnl, cumsum(pnl)
end

"""
    stress_test_suite(weights, mean_rets, cov_mat)

Run canonical stress tests (crash, vol spike, correlation breakdown, liquidity)
and report portfolio loss for each.
"""
function stress_test_suite(weights::Vector{Float64},
                            mean_rets::Vector{Float64},
                            cov_mat::Matrix{Float64})
    n = length(weights)
    scenarios = [
        ScenarioSpec("Crypto crash −50%",    fill(0.5, n),  1.0,  0.2),
        ScenarioSpec("Vol spike 3×",          ones(n),       3.0,  0.3),
        ScenarioSpec("Correlation breakdown", ones(n),       1.5,  0.5),
        ScenarioSpec("Liquidity event",       fill(0.7, n), 2.0,  0.4),
    ]
    println("\nStress Test Suite")
    println("-" ^ 50)
    results = Dict{String, Float64}()
    for spec in scenarios
        s_rets = mean_rets .* spec.shocks[1:n] .* spec.vol_mult
        pnl = dot(weights, s_rets)
        results[spec.name] = pnl
        @printf("  %-30s %+.4f\n", spec.name, pnl)
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 11 – Factor Risk Decomposition and Attribution
# ─────────────────────────────────────────────────────────────────────────────

"""
    factor_risk_decomposition(weights, factor_loadings, factor_cov, idio_var)

Brinson-Hood-Beebower style risk decomposition:
  Total variance = w'B F B'w + w'D w
where B = factor_loadings (n×k), F = factor_cov (k×k), D = diag(idio_var).
Returns (total_var, factor_var, idio_var_contribution, factor_contribution_pct).
"""
function factor_risk_decomposition(weights::Vector{Float64},
                                    B::Matrix{Float64},
                                    F::Matrix{Float64},
                                    idio_var::Vector{Float64})
    factor_var_total = dot(weights, B * F * B' * weights)
    idio_total       = dot(weights .^ 2, idio_var)
    total_var        = factor_var_total + idio_total
    # per-factor contributions
    k       = size(B, 2)
    f_contribs = zeros(k)
    for j in 1:k
        bj = B[:, j]
        f_contribs[j] = weights' * (bj * bj') * weights * F[j, j]
    end
    factor_pct = factor_var_total / (total_var + 1e-12) * 100
    return (total_var=total_var, factor_var=factor_var_total,
            idio_var=idio_total, factor_contribs=f_contribs,
            factor_pct=factor_pct)
end

"""
    brinson_attribution(portfolio_weights, benchmark_weights,
                         asset_returns, sector_map)

Brinson-Hood-Beebower attribution into allocation, selection, interaction effects
at the sector level.

`sector_map`: vector of sector indices (Int) for each asset.
"""
function brinson_attribution(pw::Vector{Float64}, bw::Vector{Float64},
                               rets::Vector{Float64}, sector_map::Vector{Int})
    sectors = sort(unique(sector_map))
    alloc = Float64[]; select = Float64[]; interact = Float64[]
    for s in sectors
        mask = sector_map .== s
        wp = sum(pw[mask]); wb = sum(bw[mask])
        rp = wb > 0 ? dot(pw[mask], rets[mask]) / (wp + 1e-12) : 0.0
        rb = wb > 0 ? dot(bw[mask], rets[mask]) / (wb + 1e-12) : 0.0
        rb_total = dot(bw, rets)
        push!(alloc,    (wp - wb) * (rb - rb_total))
        push!(select,    wb * (rp - rb))
        push!(interact, (wp - wb) * (rp - rb))
    end
    return (sectors=sectors, allocation=alloc,
            selection=select, interaction=interact,
            total=sum(alloc) + sum(select) + sum(interact))
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 12 – Monte Carlo VaR, CVaR and Backtest
# ─────────────────────────────────────────────────────────────────────────────

"""
    mc_var_cvar(weights, mean_rets, cov_mat; n_sims, alpha, horizon)

Full Monte Carlo VaR and CVaR for a portfolio over `horizon` days.
Uses Cholesky decomposition for correlated simulation.
"""
function mc_var_cvar(weights::Vector{Float64},
                      mean_rets::Vector{Float64},
                      cov_mat::Matrix{Float64};
                      n_sims::Int=10_000,
                      alpha::Float64=0.05,
                      horizon::Int=1)
    n  = length(weights)
    L  = cholesky(Symmetric(cov_mat + I(n) * 1e-8)).L
    pnl = zeros(n_sims)
    for i in 1:n_sims
        z   = L * randn(n)
        ret_h = (mean_rets .+ z) .* sqrt(Float64(horizon))
        pnl[i] = dot(weights, ret_h)
    end
    sorted = sort(pnl)
    var_idx = max(1, floor(Int, alpha * n_sims))
    var  = -sorted[var_idx]
    cvar = -mean(sorted[1:var_idx])
    return (var=var, cvar=cvar, pnl_dist=pnl)
end

"""
    kupiec_test(n_violations, n_obs, alpha)

Kupiec POF backtest for VaR: test H₀: p = α using likelihood ratio.
Returns (LR statistic, p-value approximation via chi-squared with df=1).
"""
function kupiec_test(n_violations::Int, n_obs::Int, alpha::Float64)
    p_hat = n_violations / n_obs
    p_hat = clamp(p_hat, 1e-8, 1 - 1e-8)
    LR = 2 * (n_violations * log(p_hat / alpha) +
               (n_obs - n_violations) * log((1 - p_hat) / (1 - alpha)))
    # chi-squared CDF approximation (1 df)
    pval = 1 - chi2_cdf(LR, 1)
    return (LR=LR, pval=pval, reject=(LR > 3.841))  # 5% critical value
end

"""
    chi2_cdf(x, df)

Regularised incomplete gamma function P(df/2, x/2), approximated via
series expansion (good for df = 1 or 2).
"""
function chi2_cdf(x::Float64, df::Int)
    x <= 0 && return 0.0
    a = df / 2.0; z = x / 2.0
    # lower incomplete gamma via series  ∑ z^n / (a*(a+1)*…*(a+n)) * e^{-z}
    term = 1.0 / a; s = term
    for n in 1:100
        term *= z / (a + n)
        s    += term
        abs(term) < 1e-10 * abs(s) && break
    end
    return s * exp(-z + a * log(z) - lgamma(a))
end

function lgamma(x::Float64)
    # Stirling approximation for lgamma (accurate for x > 7)
    x < 7.0 && return lgamma(x + 1.0) - log(x)
    return 0.5 * log(2π / x) + x * log(x) - x +
           1/(12x) - 1/(360x^3) + 1/(1260x^5)
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 – Tail Risk Hedging and Option Replication
# ─────────────────────────────────────────────────────────────────────────────

"""
    black_scholes_put(S, K, r, sigma, T)

Analytic Black-Scholes put price for tail-risk hedge sizing.
"""
function black_scholes_put(S::Float64, K::Float64, r::Float64,
                             sigma::Float64, T::Float64)
    T <= 0 && return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
end

function norm_cdf(x::Float64)
    return 0.5 * erfc(-x / sqrt(2.0))
end

function erfc(x::Float64)
    # Abramowitz & Stegun approximation, max error 1.5e-7
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (0.254829592 +
           t * (-0.284496736 +
           t * (1.421413741 +
           t * (-1.453152027 +
           t * 1.061405429))))
    result = poly * exp(-x^2)
    return x >= 0 ? result : 2.0 - result
end

"""
    tail_hedge_cost(portfolio_value, hedge_pct, S, K, r, sigma, T)

Cost of buying put options to hedge `hedge_pct` of portfolio value.
`K = S * (1 - hedge_pct)` by default (OTM puts).
"""
function tail_hedge_cost(portfolio_value::Float64, hedge_notional_pct::Float64,
                          S::Float64, r::Float64, sigma::Float64, T::Float64;
                          otm_pct::Float64=0.1)
    K          = S * (1 - otm_pct)
    put_price  = black_scholes_put(S, K, r, sigma, T)
    n_contracts = portfolio_value * hedge_notional_pct / S
    return n_contracts * put_price
end

"""
    delta_hedge_pnl(prices, sigma, r, K, T_initial, hedge_freq)

Simulate P&L of dynamic delta-hedging a short put over time.
`hedge_freq`: rebalance every N steps.
Returns (pnl, delta_series, gamma_series).
"""
function delta_hedge_pnl(prices::Vector{Float64}, sigma::Float64,
                           r::Float64, K::Float64, T_initial::Float64,
                           hedge_freq::Int=5)
    n  = length(prices)
    dt = T_initial / n
    deltas = zeros(n); gammas = zeros(n); pnl = zeros(n)
    position = 0.0    # shares held (delta hedge)
    cash     = 0.0
    for i in 1:n
        T_rem = T_initial - (i - 1) * dt
        T_rem = max(T_rem, 1e-8)
        S  = prices[i]
        d1 = (log(S / K) + (r + 0.5 * sigma^2) * T_rem) / (sigma * sqrt(T_rem))
        delta = norm_cdf(d1) - 1.0   # put delta
        gamma = exp(-0.5 * d1^2) / (sqrt(2π) * S * sigma * sqrt(T_rem))
        deltas[i] = delta; gammas[i] = gamma
        if mod(i, hedge_freq) == 1
            trade   = delta - position
            cash   -= trade * S
            position = delta
        end
        put_val = black_scholes_put(S, K, r, sigma, T_rem)
        pnl[i]  = position * S + cash - put_val   # hedge P&L minus option
    end
    return (pnl=pnl, deltas=deltas, gammas=gammas)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 14 – Multi-Period Simulation with Liability Matching
# ─────────────────────────────────────────────────────────────────────────────

"""
    LiabilityMatchingProblem

Asset-Liability Management (ALM) for a fund with deterministic future liabilities.
Minimises shortfall probability subject to liability present value constraint.
"""
struct LiabilityMatchingProblem
    liabilities::Vector{Float64}   # cash flows at each period
    discount_rates::Vector{Float64} # period-specific discount rates
    n_periods::Int
end

function present_value_liabilities(alm::LiabilityMatchingProblem)
    pv = 0.0
    for (t, (cf, r)) in enumerate(zip(alm.liabilities, alm.discount_rates))
        pv += cf / (1 + r)^t
    end
    return pv
end

"""
    simulate_alm(alm, initial_assets, mu, sigma, n_sims) -> NamedTuple

Monte Carlo simulation of asset-liability surplus over time.
Returns (surplus_paths, shortfall_prob, expected_surplus, funding_ratio).
"""
function simulate_alm(alm::LiabilityMatchingProblem,
                       initial_assets::Float64,
                       mu::Float64, sigma::Float64,
                       n_sims::Int=5_000)
    T   = alm.n_periods
    surplus = zeros(n_sims)
    shortfall_count = 0
    pv_liab = present_value_liabilities(alm)
    for _ in 1:n_sims
        assets = initial_assets
        fund   = 0.0
        for t in 1:T
            # Grow assets
            ret   = mu + sigma * randn()
            assets *= exp(ret)
            # Pay liability
            assets -= alm.liabilities[t]
            assets  = max(assets, 0.0)
        end
        surplus_t = assets - 0.0  # remaining assets after all liabilities
        surplus[_ <= n_sims ? _ : n_sims] = surplus_t
        if surplus_t < 0; shortfall_count += 1; end
    end
    return (surplus_paths=surplus,
            shortfall_prob=shortfall_count / n_sims,
            expected_surplus=mean(surplus),
            funding_ratio=initial_assets / (pv_liab + 1e-8))
end

"""
    glide_path(start_equity, end_equity, n_periods)

Classic target-date fund glide path: linearly reduce equity allocation
from `start_equity` to `end_equity` over `n_periods`.
Returns vector of equity weights.
"""
function glide_path(start_equity::Float64=0.90, end_equity::Float64=0.20,
                     n_periods::Int=40)
    return range(start_equity, end_equity; length=n_periods) |> collect
end

end  # module PortfolioSimulator
