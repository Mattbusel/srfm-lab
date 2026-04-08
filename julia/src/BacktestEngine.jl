###############################################################################
# BacktestEngine.jl
#
# Vectorized backtesting engine with fill simulation, performance analytics,
# walk-forward validation, Monte Carlo significance, CPCV, deflated Sharpe,
# trade-level analytics, regime-conditional breakdown, factor attribution.
#
# Dependencies: LinearAlgebra, Statistics, Random  (stdlib only)
###############################################################################

module BacktestEngine

using LinearAlgebra, Statistics, Random

export BacktestConfig, BacktestResult, run_backtest, run_backtest_vectorized
export SlippageModel, FixedSlippage, ProportionalSlippage, SqrtImpact, AlmgrenChriss
export PositionSizer, FixedFraction, VolTarget, KellySizer, RiskParitySizer
export SignalMapper, ThresholdSignal, ProportionalSignal, RankSignal
export RebalanceRule, ThresholdRebalance, CalendarRebalance, RiskTargetRebalance
export PerformanceMetrics, compute_metrics, rolling_metrics
export sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio, max_drawdown
export walk_forward_validate, expanding_window_validate, purged_kfold
export monte_carlo_significance, permutation_test
export cpcv_validate, deflated_sharpe_ratio
export trade_analytics, TradeRecord
export regime_performance, factor_attribution

# ─────────────────────────────────────────────────────────────────────────────
# §1  Type Hierarchy: Slippage Models
# ─────────────────────────────────────────────────────────────────────────────

abstract type SlippageModel end

struct FixedSlippage{T<:Real} <: SlippageModel
    cost_per_share::T
end
FixedSlippage() = FixedSlippage(0.01)

struct ProportionalSlippage{T<:Real} <: SlippageModel
    bps::T  # basis points
end
ProportionalSlippage() = ProportionalSlippage(5.0)

struct SqrtImpact{T<:Real} <: SlippageModel
    eta::T       # temporary impact coefficient
    sigma::T     # daily volatility
    adv::T       # average daily volume (dollars)
end
SqrtImpact() = SqrtImpact(0.1, 0.02, 1e7)

struct AlmgrenChriss{T<:Real} <: SlippageModel
    eta::T       # temporary impact
    gamma::T     # permanent impact
    sigma::T     # volatility
    lambda::T    # risk aversion
    tau::T       # trading horizon (days)
end
AlmgrenChriss() = AlmgrenChriss(0.1, 0.05, 0.02, 1e-6, 5.0)

"""Compute slippage cost for a trade."""
function slippage_cost(model::FixedSlippage{T}, price::T, shares::T, volume::T) where T<:Real
    abs(shares) * model.cost_per_share
end

function slippage_cost(model::ProportionalSlippage{T}, price::T, shares::T, volume::T) where T<:Real
    abs(shares) * price * model.bps * T(1e-4)
end

function slippage_cost(model::SqrtImpact{T}, price::T, shares::T, volume::T) where T<:Real
    dollar_volume = abs(shares) * price
    participation = dollar_volume / max(model.adv, T(1.0))
    model.eta * model.sigma * price * sqrt(participation) * abs(shares)
end

function slippage_cost(model::AlmgrenChriss{T}, price::T, shares::T, volume::T) where T<:Real
    dollar_volume = abs(shares) * price
    participation = dollar_volume / max(volume * price, T(1.0))
    temp_cost = model.eta * model.sigma * sqrt(participation) * dollar_volume
    perm_cost = model.gamma * dollar_volume * participation
    temp_cost + perm_cost
end

"""Almgren-Chriss optimal execution trajectory."""
function ac_optimal_trajectory(model::AlmgrenChriss{T}, total_shares::T,
                               n_steps::Int) where T<:Real
    kappa = sqrt(model.lambda * model.sigma^2 / model.eta)
    tau = model.tau
    trajectory = Vector{T}(undef, n_steps)
    remaining = total_shares
    for i in 1:n_steps
        t = (i - 1) * tau / n_steps
        t_next = i * tau / n_steps
        x_t = total_shares * sinh(kappa * (tau - t)) / sinh(kappa * tau)
        x_next = total_shares * sinh(kappa * (tau - t_next)) / sinh(kappa * tau)
        trajectory[i] = x_t - x_next
        remaining -= trajectory[i]
    end
    trajectory[n_steps] += remaining
    return trajectory
end

# ─────────────────────────────────────────────────────────────────────────────
# §2  Signal-to-Position Mapping
# ─────────────────────────────────────────────────────────────────────────────

abstract type SignalMapper end

struct ThresholdSignal{T<:Real} <: SignalMapper
    long_threshold::T
    short_threshold::T
end
ThresholdSignal() = ThresholdSignal(1.0, -1.0)

struct ProportionalSignal{T<:Real} <: SignalMapper
    scale::T
    cap::T
end
ProportionalSignal() = ProportionalSignal(1.0, 1.0)

struct RankSignal <: SignalMapper
    n_long::Int
    n_short::Int
end
RankSignal() = RankSignal(10, 10)

"""Map raw signals to target positions."""
function map_signal(mapper::ThresholdSignal{T}, signals::AbstractVector{T}) where T<:Real
    n = length(signals)
    positions = zeros(T, n)
    for i in 1:n
        if signals[i] > mapper.long_threshold
            positions[i] = one(T)
        elseif signals[i] < mapper.short_threshold
            positions[i] = -one(T)
        end
    end
    positions
end

function map_signal(mapper::ProportionalSignal{T}, signals::AbstractVector{T}) where T<:Real
    positions = clamp.(signals .* mapper.scale, -mapper.cap, mapper.cap)
    return positions
end

function map_signal(mapper::RankSignal, signals::AbstractVector{T}) where T<:Real
    n = length(signals)
    positions = zeros(T, n)
    sorted_idx = sortperm(signals; rev=true)
    n_long = min(mapper.n_long, n)
    n_short = min(mapper.n_short, n)
    for i in 1:n_long
        positions[sorted_idx[i]] = one(T) / n_long
    end
    for i in 1:n_short
        positions[sorted_idx[n - i + 1]] = -one(T) / n_short
    end
    positions
end

"""Cross-sectional z-score normalization of signals."""
function zscore_signals(signals::AbstractMatrix{T}) where T<:Real
    n_obs, n_assets = size(signals)
    out = similar(signals)
    for t in 1:n_obs
        row = signals[t, :]
        mu = mean(row)
        s = std(row)
        if s > T(1e-16)
            out[t, :] = (row .- mu) ./ s
        else
            out[t, :] .= zero(T)
        end
    end
    out
end

"""Winsorize signals at percentiles."""
function winsorize(signals::AbstractVector{T}; pct::T=T(0.05)) where T<:Real
    sorted = sort(signals)
    n = length(sorted)
    lo = sorted[max(1, ceil(Int, pct * n))]
    hi = sorted[min(n, floor(Int, (1 - pct) * n))]
    clamp.(signals, lo, hi)
end

# ─────────────────────────────────────────────────────────────────────────────
# §3  Position Sizing
# ─────────────────────────────────────────────────────────────────────────────

abstract type PositionSizer end

struct FixedFraction{T<:Real} <: PositionSizer
    fraction::T
end
FixedFraction() = FixedFraction(0.02)

struct VolTarget{T<:Real} <: PositionSizer
    target_vol::T
    lookback::Int
end
VolTarget() = VolTarget(0.10, 20)

struct KellySizer{T<:Real} <: PositionSizer
    fraction::T  # fractional Kelly
    lookback::Int
end
KellySizer() = KellySizer(0.25, 60)

struct RiskParitySizer{T<:Real} <: PositionSizer
    target_vol::T
    lookback::Int
end
RiskParitySizer() = RiskParitySizer(0.10, 60)

"""Compute position sizes."""
function size_positions(sizer::FixedFraction{T}, signals::AbstractVector{T},
                        equity::T, prices::AbstractVector{T};
                        kwargs...) where T<:Real
    positions = signals .* sizer.fraction .* equity ./ prices
    return positions
end

function size_positions(sizer::VolTarget{T}, signals::AbstractVector{T},
                        equity::T, prices::AbstractVector{T};
                        recent_returns::AbstractMatrix{T}=zeros(T,0,0),
                        kwargs...) where T<:Real
    n_assets = length(signals)
    if size(recent_returns, 1) < 2
        return signals .* equity ./ (n_assets .* prices)
    end
    vols = vec(std(recent_returns; dims=1)) .* sqrt(T(252))
    target_per_asset = sizer.target_vol / sqrt(T(n_assets))
    sizes = zeros(T, n_assets)
    for i in 1:n_assets
        if abs(signals[i]) > T(1e-10) && vols[i] > T(1e-10)
            dollar_pos = signals[i] * equity * target_per_asset / vols[i]
            sizes[i] = dollar_pos / prices[i]
        end
    end
    sizes
end

function size_positions(sizer::KellySizer{T}, signals::AbstractVector{T},
                        equity::T, prices::AbstractVector{T};
                        recent_returns::AbstractMatrix{T}=zeros(T,0,0),
                        kwargs...) where T<:Real
    n_assets = length(signals)
    if size(recent_returns, 1) < 2
        return signals .* equity ./ (n_assets .* prices)
    end
    mu = vec(mean(recent_returns; dims=1))
    sigma2 = vec(var(recent_returns; dims=1))
    kelly_fracs = mu ./ (sigma2 .+ T(1e-10))
    scaled = sizer.fraction .* kelly_fracs .* signals
    dollar_pos = scaled .* equity
    dollar_pos ./ prices
end

function size_positions(sizer::RiskParitySizer{T}, signals::AbstractVector{T},
                        equity::T, prices::AbstractVector{T};
                        recent_returns::AbstractMatrix{T}=zeros(T,0,0),
                        kwargs...) where T<:Real
    n_assets = length(signals)
    if size(recent_returns, 1) < 5
        return signals .* equity ./ (n_assets .* prices)
    end
    vols = vec(std(recent_returns; dims=1))
    inv_vols = one(T) ./ (vols .+ T(1e-10))
    weights = inv_vols ./ sum(inv_vols)
    # Scale to target vol
    port_vol_est = sqrt(dot(weights, cov(recent_returns) * weights)) * sqrt(T(252))
    scale = sizer.target_vol / max(port_vol_est, T(1e-10))
    dollar_pos = signals .* weights .* equity .* min(scale, T(5.0))
    dollar_pos ./ prices
end

# ─────────────────────────────────────────────────────────────────────────────
# §4  Rebalance Rules
# ─────────────────────────────────────────────────────────────────────────────

abstract type RebalanceRule end

struct ThresholdRebalance{T<:Real} <: RebalanceRule
    threshold::T  # max drift from target
end
ThresholdRebalance() = ThresholdRebalance(0.05)

struct CalendarRebalance <: RebalanceRule
    frequency::Int  # days between rebalances
end
CalendarRebalance() = CalendarRebalance(21)

struct RiskTargetRebalance{T<:Real} <: RebalanceRule
    target_vol::T
    vol_band::T  # rebalance if vol outside target ± band
    lookback::Int
end
RiskTargetRebalance() = RiskTargetRebalance(0.10, 0.02, 20)

"""Check if rebalance is needed."""
function should_rebalance(rule::ThresholdRebalance{T}, w_current::AbstractVector{T},
                          w_target::AbstractVector{T}; bar::Int=0) where T<:Real
    max_drift = maximum(abs.(w_current .- w_target))
    return max_drift > rule.threshold
end

function should_rebalance(rule::CalendarRebalance, w_current::AbstractVector{T},
                          w_target::AbstractVector{T}; bar::Int=0) where T<:Real
    return bar > 0 && mod(bar, rule.frequency) == 0
end

function should_rebalance(rule::RiskTargetRebalance{T}, w_current::AbstractVector{T},
                          w_target::AbstractVector{T};
                          bar::Int=0,
                          recent_returns::AbstractMatrix{T}=zeros(T,0,0)) where T<:Real
    if size(recent_returns, 1) < rule.lookback
        return false
    end
    port_ret = recent_returns * w_current
    vol = std(port_ret) * sqrt(T(252))
    return abs(vol - rule.target_vol) > rule.vol_band
end

# ─────────────────────────────────────────────────────────────────────────────
# §5  Backtest Configuration and Results
# ─────────────────────────────────────────────────────────────────────────────

struct BacktestConfig{T<:Real}
    slippage::SlippageModel
    sizer::PositionSizer
    signal_mapper::SignalMapper
    rebalance_rule::RebalanceRule
    initial_capital::T
    commission_rate::T
    margin_rate::T
    max_leverage::T
    risk_free_rate::T
end

function BacktestConfig(; slippage::SlippageModel=ProportionalSlippage(5.0),
                        sizer::PositionSizer=VolTarget(0.10, 20),
                        signal_mapper::SignalMapper=ProportionalSignal(1.0, 1.0),
                        rebalance_rule::RebalanceRule=CalendarRebalance(21),
                        initial_capital::Float64=1e6,
                        commission_rate::Float64=0.001,
                        margin_rate::Float64=0.05,
                        max_leverage::Float64=2.0,
                        risk_free_rate::Float64=0.02)
    BacktestConfig{Float64}(slippage, sizer, signal_mapper, rebalance_rule,
                            initial_capital, commission_rate, margin_rate,
                            max_leverage, risk_free_rate)
end

struct TradeRecord{T<:Real}
    bar_enter::Int
    bar_exit::Int
    asset_id::Int
    direction::Int  # +1 or -1
    entry_price::T
    exit_price::T
    shares::T
    pnl::T
    cost::T
end

mutable struct BacktestState{T<:Real}
    equity::T
    cash::T
    positions::Vector{T}      # shares held
    weights::Vector{T}        # current portfolio weights
    entry_prices::Vector{T}
    entry_bars::Vector{Int}
    total_trades::Int
    total_costs::T
end

struct BacktestResult{T<:Real}
    equity_curve::Vector{T}
    returns::Vector{T}
    positions_history::Matrix{T}
    weights_history::Matrix{T}
    trades::Vector{TradeRecord{T}}
    total_costs::T
    metrics::Dict{Symbol, T}
end

# ─────────────────────────────────────────────────────────────────────────────
# §6  Bar-by-Bar Event Loop
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_backtest(prices, signals, config) -> BacktestResult

Run bar-by-bar backtest with fill simulation.
"""
function run_backtest(prices::AbstractMatrix{T}, signals::AbstractMatrix{T},
                      config::BacktestConfig{T}) where T<:Real
    n_bars, n_assets = size(prices)
    @assert size(signals) == (n_bars, n_assets)
    state = BacktestState{T}(
        config.initial_capital,
        config.initial_capital,
        zeros(T, n_assets),
        zeros(T, n_assets),
        zeros(T, n_assets),
        zeros(Int, n_assets),
        0, zero(T)
    )
    equity_curve = Vector{T}(undef, n_bars)
    returns_vec = Vector{T}(undef, n_bars)
    positions_hist = Matrix{T}(undef, n_bars, n_assets)
    weights_hist = Matrix{T}(undef, n_bars, n_assets)
    trades = Vector{TradeRecord{T}}()
    returns_matrix = zeros(T, n_bars, n_assets)
    for t in 2:n_bars
        returns_matrix[t, :] = (prices[t, :] .- prices[t-1, :]) ./ prices[t-1, :]
    end
    prev_equity = config.initial_capital
    for t in 1:n_bars
        # Mark to market
        port_value = zero(T)
        for i in 1:n_assets
            port_value += state.positions[i] * prices[t, i]
        end
        state.equity = state.cash + port_value
        equity_curve[t] = state.equity
        returns_vec[t] = t > 1 ? (state.equity - prev_equity) / max(prev_equity, T(1e-10)) : zero(T)
        prev_equity = state.equity
        positions_hist[t, :] = state.positions
        total_pos = sum(abs.(state.positions) .* prices[t, :])
        if total_pos > T(1e-10)
            state.weights = (state.positions .* prices[t, :]) ./ total_pos
        else
            state.weights .= zero(T)
        end
        weights_hist[t, :] = state.weights
        # Generate target positions from signals
        sig = signals[t, :]
        target_pos_raw = map_signal(config.signal_mapper, sig)
        lookback = min(t - 1, 60)
        recent_ret = lookback >= 2 ? returns_matrix[t-lookback+1:t, :] : zeros(T, 0, n_assets)
        target_shares = size_positions(config.sizer, target_pos_raw,
                                       state.equity, prices[t, :];
                                       recent_returns=recent_ret)
        # Leverage check
        target_dollar = sum(abs.(target_shares) .* prices[t, :])
        if target_dollar > config.max_leverage * state.equity
            scale = config.max_leverage * state.equity / target_dollar
            target_shares .*= scale
        end
        # Rebalance check
        target_weights = (target_shares .* prices[t, :]) ./ max(state.equity, T(1e-10))
        do_rebalance = should_rebalance(config.rebalance_rule, state.weights,
                                        target_weights; bar=t, recent_returns=recent_ret)
        if !do_rebalance && t > 1
            continue
        end
        # Execute trades
        for i in 1:n_assets
            delta = target_shares[i] - state.positions[i]
            if abs(delta) < T(1e-10)
                continue
            end
            # Record exit trade if flipping direction
            if state.positions[i] != zero(T) && sign(target_shares[i]) != sign(state.positions[i])
                pnl = state.positions[i] * (prices[t, i] - state.entry_prices[i])
                cost = slippage_cost(config.slippage, prices[t, i],
                                    abs(state.positions[i]), T(1e6))
                cost += abs(state.positions[i]) * prices[t, i] * config.commission_rate
                push!(trades, TradeRecord{T}(
                    state.entry_bars[i], t, i,
                    state.positions[i] > 0 ? 1 : -1,
                    state.entry_prices[i], prices[t, i],
                    abs(state.positions[i]), pnl - cost, cost
                ))
                state.cash += state.positions[i] * prices[t, i] - cost
                state.total_costs += cost
                state.positions[i] = zero(T)
            end
            delta = target_shares[i] - state.positions[i]
            if abs(delta) < T(1e-10)
                continue
            end
            cost = slippage_cost(config.slippage, prices[t, i], abs(delta), T(1e6))
            cost += abs(delta) * prices[t, i] * config.commission_rate
            state.cash -= delta * prices[t, i] + cost
            state.total_costs += cost
            state.total_trades += 1
            if abs(state.positions[i]) < T(1e-10)
                state.entry_prices[i] = prices[t, i]
                state.entry_bars[i] = t
            end
            state.positions[i] = target_shares[i]
        end
    end
    # Close remaining positions
    for i in 1:n_assets
        if abs(state.positions[i]) > T(1e-10)
            pnl = state.positions[i] * (prices[n_bars, i] - state.entry_prices[i])
            cost = slippage_cost(config.slippage, prices[n_bars, i],
                                abs(state.positions[i]), T(1e6))
            push!(trades, TradeRecord{T}(
                state.entry_bars[i], n_bars, i,
                state.positions[i] > 0 ? 1 : -1,
                state.entry_prices[i], prices[n_bars, i],
                abs(state.positions[i]), pnl - cost, cost
            ))
        end
    end
    metrics = compute_metrics(returns_vec; rf=config.risk_free_rate)
    BacktestResult{T}(equity_curve, returns_vec, positions_hist, weights_hist,
                      trades, state.total_costs, metrics)
end

"""
    run_backtest_vectorized(returns, weights_matrix; kwargs...) -> BacktestResult

Fast vectorized backtest given pre-computed weights.
"""
function run_backtest_vectorized(returns::AbstractMatrix{T},
                                 weights::AbstractMatrix{T};
                                 initial_capital::T=T(1e6),
                                 tc_bps::T=T(5.0),
                                 rf::T=T(0.02)) where T<:Real
    n_bars, n_assets = size(returns)
    @assert size(weights) == (n_bars, n_assets)
    port_returns = Vector{T}(undef, n_bars)
    total_costs = zero(T)
    for t in 1:n_bars
        gross_ret = dot(weights[t, :], returns[t, :])
        if t > 1
            turnover = sum(abs.(weights[t, :] .- weights[t-1, :]))
            tc = turnover * tc_bps * T(1e-4)
        else
            tc = zero(T)
        end
        port_returns[t] = gross_ret - tc
        total_costs += tc * initial_capital
    end
    equity_curve = initial_capital .* cumprod(one(T) .+ port_returns)
    metrics = compute_metrics(port_returns; rf=rf)
    BacktestResult{T}(equity_curve, port_returns, zeros(T, n_bars, n_assets),
                      weights, TradeRecord{T}[], total_costs, metrics)
end

# ─────────────────────────────────────────────────────────────────────────────
# §7  Performance Analytics
# ─────────────────────────────────────────────────────────────────────────────

"""Annualized Sharpe ratio."""
function sharpe_ratio(returns::AbstractVector{T}; rf::T=T(0.0), freq::Int=252) where T<:Real
    excess = returns .- rf / freq
    mu = mean(excess)
    s = std(excess)
    mu / max(s, T(1e-16)) * sqrt(T(freq))
end

"""Sortino ratio."""
function sortino_ratio(returns::AbstractVector{T}; rf::T=T(0.0), freq::Int=252) where T<:Real
    excess = returns .- rf / freq
    downside = sqrt(mean(max.(-excess, zero(T)) .^ 2))
    mean(excess) / max(downside, T(1e-16)) * sqrt(T(freq))
end

"""Calmar ratio."""
function calmar_ratio(returns::AbstractVector{T}; freq::Int=252) where T<:Real
    ann_ret = mean(returns) * freq
    mdd = max_drawdown(returns)
    ann_ret / max(mdd, T(1e-16))
end

"""Maximum drawdown."""
function max_drawdown(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    if n == 0
        return zero(T)
    end
    cum = cumprod(one(T) .+ returns)
    peak = accumulate(max, cum)
    dd = (peak .- cum) ./ peak
    maximum(dd)
end

"""Maximum drawdown duration (bars)."""
function max_drawdown_duration(returns::AbstractVector{T}) where T<:Real
    cum = cumprod(one(T) .+ returns)
    peak = accumulate(max, cum)
    in_dd = cum .< peak
    max_dur = 0
    cur_dur = 0
    for v in in_dd
        if v
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else
            cur_dur = 0
        end
    end
    max_dur
end

"""Omega ratio."""
function omega_ratio(returns::AbstractVector{T}; threshold::T=T(0.0)) where T<:Real
    gains = sum(max.(returns .- threshold, zero(T)))
    losses = sum(max.(threshold .- returns, zero(T)))
    gains / max(losses, T(1e-16))
end

"""Tail ratio: 95th percentile / |5th percentile|."""
function tail_ratio(returns::AbstractVector{T}) where T<:Real
    sorted = sort(returns)
    n = length(sorted)
    p95 = sorted[max(1, ceil(Int, 0.95 * n))]
    p05 = sorted[max(1, ceil(Int, 0.05 * n))]
    abs(p95) / max(abs(p05), T(1e-16))
end

"""Gain-to-pain ratio."""
function gain_to_pain(returns::AbstractVector{T}) where T<:Real
    total_gain = sum(returns)
    total_pain = sum(abs.(min.(returns, zero(T))))
    total_gain / max(total_pain, T(1e-16))
end

"""Common sense ratio: tail_ratio * gain_to_pain."""
function common_sense_ratio(returns::AbstractVector{T}) where T<:Real
    tail_ratio(returns) * gain_to_pain(returns)
end

"""Stability of returns: R² of cumulative return regression."""
function stability(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    cum = cumsum(returns)
    x = collect(T, 1:n)
    x_mean = mean(x)
    y_mean = mean(cum)
    ss_xy = dot(x .- x_mean, cum .- y_mean)
    ss_xx = dot(x .- x_mean, x .- x_mean)
    ss_yy = dot(cum .- y_mean, cum .- y_mean)
    if ss_xx < T(1e-16) || ss_yy < T(1e-16)
        return zero(T)
    end
    r2 = (ss_xy / ss_xx)^2 * ss_xx / ss_yy
    return r2
end

"""Skewness of returns."""
function return_skewness(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    mu = mean(returns)
    s = std(returns)
    if s < T(1e-16)
        return zero(T)
    end
    sum((returns .- mu) .^ 3) / (n * s^3)
end

"""Kurtosis of returns (excess)."""
function return_kurtosis(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    mu = mean(returns)
    s = std(returns)
    if s < T(1e-16)
        return zero(T)
    end
    sum((returns .- mu) .^ 4) / (n * s^4) - T(3.0)
end

"""Value at Risk (historical)."""
function var_historical(returns::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    sorted = sort(returns)
    n = length(sorted)
    idx = max(1, ceil(Int, alpha * n))
    -sorted[idx]
end

"""Conditional VaR (Expected Shortfall)."""
function cvar_historical(returns::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    sorted = sort(returns)
    n = length(sorted)
    cutoff = max(1, floor(Int, alpha * n))
    -mean(sorted[1:cutoff])
end

"""Ulcer index."""
function ulcer_index(returns::AbstractVector{T}) where T<:Real
    cum = cumprod(one(T) .+ returns)
    peak = accumulate(max, cum)
    dd_pct = (peak .- cum) ./ peak .* 100
    sqrt(mean(dd_pct .^ 2))
end

"""
    compute_metrics(returns; rf=0.02) -> Dict

Compute all performance metrics.
"""
function compute_metrics(returns::AbstractVector{T}; rf::T=T(0.02), freq::Int=252) where T<:Real
    n = length(returns)
    metrics = Dict{Symbol, T}()
    metrics[:total_return] = prod(one(T) .+ returns) - one(T)
    metrics[:ann_return] = mean(returns) * freq
    metrics[:ann_vol] = std(returns) * sqrt(T(freq))
    metrics[:sharpe] = sharpe_ratio(returns; rf=rf, freq=freq)
    metrics[:sortino] = sortino_ratio(returns; rf=rf, freq=freq)
    metrics[:calmar] = calmar_ratio(returns; freq=freq)
    metrics[:omega] = omega_ratio(returns)
    metrics[:max_dd] = max_drawdown(returns)
    metrics[:max_dd_duration] = T(max_drawdown_duration(returns))
    metrics[:var_95] = var_historical(returns; alpha=T(0.05))
    metrics[:cvar_95] = cvar_historical(returns; alpha=T(0.05))
    metrics[:skewness] = return_skewness(returns)
    metrics[:kurtosis] = return_kurtosis(returns)
    metrics[:tail_ratio] = tail_ratio(returns)
    metrics[:stability] = stability(returns)
    metrics[:ulcer_index] = ulcer_index(returns)
    metrics[:gain_to_pain] = gain_to_pain(returns)
    metrics[:n_observations] = T(n)
    return metrics
end

"""Rolling performance metrics."""
function rolling_metrics(returns::AbstractVector{T};
                         window::Int=63, step::Int=1) where T<:Real
    n = length(returns)
    n_windows = div(n - window, step) + 1
    sharpes = Vector{T}(undef, n_windows)
    vols = Vector{T}(undef, n_windows)
    dds = Vector{T}(undef, n_windows)
    sortinos = Vector{T}(undef, n_windows)
    for (k, t) in enumerate(window:step:n)
        r = returns[t-window+1:t]
        sharpes[k] = sharpe_ratio(r)
        vols[k] = std(r) * sqrt(T(252))
        dds[k] = max_drawdown(r)
        sortinos[k] = sortino_ratio(r)
    end
    return (sharpe=sharpes, vol=vols, max_dd=dds, sortino=sortinos)
end

# ─────────────────────────────────────────────────────────────────────────────
# §8  Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────

"""
    walk_forward_validate(returns, signal_func; kwargs...) -> oos_returns

Rolling walk-forward: train on window, test on next step.
signal_func(train_data) -> weights for test period.
"""
function walk_forward_validate(returns::AbstractMatrix{T},
                               signal_func::Function;
                               train_window::Int=252,
                               test_window::Int=21,
                               gap::Int=0) where T<:Real
    n_obs, n_assets = size(returns)
    oos_returns = T[]
    oos_weights = Matrix{T}(undef, 0, n_assets)
    t = train_window + gap + 1
    while t + test_window - 1 <= n_obs
        train_data = returns[t-train_window-gap:t-gap-1, :]
        w = signal_func(train_data)
        for s in 0:test_window-1
            if t + s <= n_obs
                r = dot(w, returns[t + s, :])
                push!(oos_returns, r)
                oos_weights = vcat(oos_weights, w')
            end
        end
        t += test_window
    end
    return oos_returns, oos_weights
end

"""Expanding window walk-forward."""
function expanding_window_validate(returns::AbstractMatrix{T},
                                   signal_func::Function;
                                   min_train::Int=252,
                                   test_window::Int=21,
                                   gap::Int=0) where T<:Real
    n_obs, n_assets = size(returns)
    oos_returns = T[]
    t = min_train + gap + 1
    while t + test_window - 1 <= n_obs
        train_data = returns[1:t-gap-1, :]
        w = signal_func(train_data)
        for s in 0:test_window-1
            if t + s <= n_obs
                push!(oos_returns, dot(w, returns[t + s, :]))
            end
        end
        t += test_window
    end
    return oos_returns
end

"""Purged K-fold cross-validation."""
function purged_kfold(returns::AbstractMatrix{T},
                      signal_func::Function;
                      n_folds::Int=5, purge::Int=5) where T<:Real
    n_obs = size(returns, 1)
    fold_size = div(n_obs, n_folds)
    oos_returns = Vector{T}()
    fold_metrics = Vector{Dict{Symbol, T}}()
    for fold in 1:n_folds
        test_start = (fold - 1) * fold_size + 1
        test_end = min(fold * fold_size, n_obs)
        # Purge gap before and after test set
        train_mask = trues(n_obs)
        for t in max(1, test_start - purge):min(n_obs, test_end + purge)
            train_mask[t] = false
        end
        train_idx = findall(train_mask)
        if length(train_idx) < 30
            continue
        end
        train_data = returns[train_idx, :]
        w = signal_func(train_data)
        test_rets = returns[test_start:test_end, :] * w
        append!(oos_returns, test_rets)
        push!(fold_metrics, compute_metrics(test_rets))
    end
    return oos_returns, fold_metrics
end

"""Anchored walk-forward (always start from beginning)."""
function anchored_walk_forward(returns::AbstractMatrix{T},
                               signal_func::Function;
                               min_train::Int=126,
                               test_window::Int=21) where T<:Real
    expanding_window_validate(returns, signal_func;
                             min_train=min_train, test_window=test_window)
end

# ─────────────────────────────────────────────────────────────────────────────
# §9  Monte Carlo & Permutation Tests
# ─────────────────────────────────────────────────────────────────────────────

"""
    permutation_test(returns, weights; n_perm=1000) -> p_value, null_sharpes

Permutation test: shuffle returns across time to test significance.
"""
function permutation_test(returns::AbstractMatrix{T},
                          weights::AbstractMatrix{T};
                          n_perm::Int=1000,
                          rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n_obs = size(returns, 1)
    actual_rets = vec(sum(returns .* weights; dims=2))
    actual_sharpe = sharpe_ratio(actual_rets)
    null_sharpes = Vector{T}(undef, n_perm)
    for p in 1:n_perm
        perm_idx = randperm(rng, n_obs)
        perm_returns = returns[perm_idx, :]
        perm_rets = vec(sum(perm_returns .* weights; dims=2))
        null_sharpes[p] = sharpe_ratio(perm_rets)
    end
    p_value = mean(null_sharpes .>= actual_sharpe)
    return p_value, null_sharpes, actual_sharpe
end

"""
    monte_carlo_significance(returns; n_sim=10000) -> p_value

Monte Carlo test: generate random portfolios and compare.
"""
function monte_carlo_significance(returns::AbstractMatrix{T};
                                   strategy_returns::AbstractVector{T},
                                   n_sim::Int=10000,
                                   rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n_obs, n_assets = size(returns)
    actual_sharpe = sharpe_ratio(strategy_returns)
    count_better = 0
    for _ in 1:n_sim
        w = rand(rng, T, n_assets)
        w ./= sum(w)
        sim_ret = returns * w
        sim_sharpe = sharpe_ratio(sim_ret)
        if sim_sharpe >= actual_sharpe
            count_better += 1
        end
    end
    return count_better / n_sim
end

"""Block bootstrap for time-series data."""
function block_bootstrap(returns::AbstractVector{T};
                         block_size::Int=20, n_bootstrap::Int=1000,
                         rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n = length(returns)
    n_blocks = div(n, block_size)
    boot_sharpes = Vector{T}(undef, n_bootstrap)
    for b in 1:n_bootstrap
        boot_ret = T[]
        for _ in 1:n_blocks
            start = rand(rng, 1:n-block_size+1)
            append!(boot_ret, returns[start:start+block_size-1])
        end
        boot_sharpes[b] = sharpe_ratio(boot_ret)
    end
    return boot_sharpes
end

"""Stationary bootstrap (Politis & Romano)."""
function stationary_bootstrap(returns::AbstractVector{T};
                               avg_block::Int=20, n_bootstrap::Int=1000,
                               rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n = length(returns)
    p = one(T) / avg_block
    boot_sharpes = Vector{T}(undef, n_bootstrap)
    for b in 1:n_bootstrap
        boot_ret = Vector{T}(undef, n)
        idx = rand(rng, 1:n)
        for t in 1:n
            boot_ret[t] = returns[idx]
            if rand(rng) < p
                idx = rand(rng, 1:n)
            else
                idx = mod1(idx + 1, n)
            end
        end
        boot_sharpes[b] = sharpe_ratio(boot_ret)
    end
    return boot_sharpes
end

# ─────────────────────────────────────────────────────────────────────────────
# §10  CPCV (Combinatorial Purged Cross-Validation)
# ─────────────────────────────────────────────────────────────────────────────

"""
    cpcv_validate(returns, signal_func; n_groups=10, n_test=2, purge=5) -> oos_sharpes

Combinatorial purged cross-validation (de Prado).
"""
function cpcv_validate(returns::AbstractMatrix{T},
                       signal_func::Function;
                       n_groups::Int=10, n_test::Int=2,
                       purge::Int=5, max_combos::Int=100) where T<:Real
    n_obs = size(returns, 1)
    group_size = div(n_obs, n_groups)
    # Generate group boundaries
    groups = [(g-1)*group_size+1 : min(g*group_size, n_obs) for g in 1:n_groups]
    # Generate combinations of test groups
    combos = _combinations(n_groups, n_test)
    if length(combos) > max_combos
        combos = combos[1:max_combos]
    end
    oos_sharpes = Vector{T}(undef, length(combos))
    for (ci, test_groups) in enumerate(combos)
        test_mask = falses(n_obs)
        for g in test_groups
            test_mask[groups[g]] .= true
        end
        # Purge around test boundaries
        train_mask = .!test_mask
        for g in test_groups
            lo = first(groups[g])
            hi = last(groups[g])
            for t in max(1, lo-purge):min(n_obs, hi+purge)
                train_mask[t] = false
            end
        end
        train_idx = findall(train_mask)
        test_idx = findall(test_mask)
        if length(train_idx) < 30 || length(test_idx) < 5
            oos_sharpes[ci] = zero(T)
            continue
        end
        w = signal_func(returns[train_idx, :])
        test_rets = returns[test_idx, :] * w
        oos_sharpes[ci] = sharpe_ratio(test_rets)
    end
    return oos_sharpes
end

"""Generate combinations C(n, k) as vectors of indices."""
function _combinations(n::Int, k::Int)
    result = Vector{Vector{Int}}()
    combo = collect(1:k)
    while true
        push!(result, copy(combo))
        i = k
        while i > 0 && combo[i] == n - k + i
            i -= 1
        end
        if i == 0
            break
        end
        combo[i] += 1
        for j in i+1:k
            combo[j] = combo[j-1] + 1
        end
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# §11  Deflated Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

"""
    deflated_sharpe_ratio(observed_sharpe, n_trials, n_obs; kwargs...) -> dsr, p_value

Bailey & Lopez de Prado Deflated Sharpe Ratio.
Accounts for multiple testing, skewness, kurtosis.
"""
function deflated_sharpe_ratio(observed_sharpe::T, n_trials::Int, n_obs::Int;
                                skewness::T=T(0.0), kurtosis::T=T(0.0),
                                var_sharpe::T=T(0.0)) where T<:Real
    # Expected max Sharpe under null (iid normal)
    gamma_euler = T(0.5772156649)
    e_max_sharpe = if n_trials > 1
        sqrt(T(2) * log(T(n_trials))) -
        (log(pi) + gamma_euler) / (T(2) * sqrt(T(2) * log(T(n_trials))))
    else
        zero(T)
    end
    if var_sharpe <= zero(T)
        # Estimate variance of Sharpe ratio
        var_sharpe = (one(T) + T(0.25) * observed_sharpe^2 * (kurtosis - one(T)) -
                     observed_sharpe * skewness) / T(n_obs)
    end
    std_sharpe = sqrt(max(var_sharpe, T(1e-16)))
    # PSR: probability that true Sharpe > 0, adjusted for max
    z = (observed_sharpe - e_max_sharpe) / std_sharpe
    # Standard normal CDF approximation
    p_value = _normal_cdf(z)
    dsr = p_value
    return dsr, one(T) - p_value
end

"""Standard normal CDF approximation (Abramowitz & Stegun)."""
function _normal_cdf(x::T) where T<:Real
    if x < T(-8)
        return zero(T)
    elseif x > T(8)
        return one(T)
    end
    # Horner form
    t = one(T) / (one(T) + T(0.2316419) * abs(x))
    d = T(0.3989422804) * exp(-x * x / 2)
    p = d * t * (T(0.3193815) + t * (T(-0.3565638) + t * (T(1.781478) +
        t * (T(-1.8212560) + t * T(1.3302744)))))
    x >= zero(T) ? one(T) - p : p
end

"""Probabilistic Sharpe Ratio."""
function probabilistic_sharpe(observed_sharpe::T, benchmark_sharpe::T,
                               n_obs::Int;
                               skewness::T=T(0.0), kurtosis::T=T(0.0)) where T<:Real
    var_sr = (one(T) - skewness * observed_sharpe +
              T(0.25) * (kurtosis - one(T)) * observed_sharpe^2) / T(n_obs)
    z = (observed_sharpe - benchmark_sharpe) / sqrt(max(var_sr, T(1e-16)))
    _normal_cdf(z)
end

"""Minimum track record length for Sharpe significance."""
function min_track_record(observed_sharpe::T, benchmark_sharpe::T;
                          confidence::T=T(0.95),
                          skewness::T=T(0.0), kurtosis::T=T(0.0)) where T<:Real
    z_alpha = T(1.6449)  # 95% confidence
    if observed_sharpe <= benchmark_sharpe
        return T(Inf)
    end
    numer = (one(T) - skewness * observed_sharpe +
             T(0.25) * (kurtosis - one(T)) * observed_sharpe^2)
    denom = (observed_sharpe - benchmark_sharpe)^2
    z_alpha^2 * numer / denom
end

# ─────────────────────────────────────────────────────────────────────────────
# §12  Trade-Level Analytics
# ─────────────────────────────────────────────────────────────────────────────

"""
    trade_analytics(trades) -> Dict

Compute trade-level statistics.
"""
function trade_analytics(trades::Vector{TradeRecord{T}}) where T<:Real
    if isempty(trades)
        return Dict{Symbol, T}()
    end
    n = length(trades)
    pnls = [t.pnl for t in trades]
    costs = [t.cost for t in trades]
    durations = [T(t.bar_exit - t.bar_enter) for t in trades]
    winners = filter(x -> x > zero(T), pnls)
    losers = filter(x -> x <= zero(T), pnls)
    metrics = Dict{Symbol, T}()
    metrics[:n_trades] = T(n)
    metrics[:total_pnl] = sum(pnls)
    metrics[:total_costs] = sum(costs)
    metrics[:win_rate] = T(length(winners)) / T(n)
    metrics[:avg_win] = isempty(winners) ? zero(T) : mean(winners)
    metrics[:avg_loss] = isempty(losers) ? zero(T) : mean(losers)
    metrics[:profit_factor] = if isempty(losers) || sum(abs.(losers)) < T(1e-16)
        T(Inf)
    else
        sum(winners) / sum(abs.(losers))
    end
    metrics[:payoff_ratio] = if isempty(losers) || abs(mean(losers)) < T(1e-16)
        T(Inf)
    else
        metrics[:avg_win] / abs(metrics[:avg_loss])
    end
    metrics[:avg_hold_time] = mean(durations)
    metrics[:median_hold_time] = sort(durations)[div(n, 2) + 1]
    metrics[:max_win] = maximum(pnls)
    metrics[:max_loss] = minimum(pnls)
    metrics[:avg_pnl] = mean(pnls)
    metrics[:std_pnl] = std(pnls)
    metrics[:sharpe_trades] = metrics[:avg_pnl] / max(metrics[:std_pnl], T(1e-16))
    # Expectancy
    metrics[:expectancy] = metrics[:win_rate] * metrics[:avg_win] +
                           (one(T) - metrics[:win_rate]) * metrics[:avg_loss]
    # Kelly from trade stats
    if metrics[:payoff_ratio] != T(Inf) && metrics[:payoff_ratio] > zero(T)
        metrics[:kelly_pct] = metrics[:win_rate] -
                              (one(T) - metrics[:win_rate]) / metrics[:payoff_ratio]
    else
        metrics[:kelly_pct] = metrics[:win_rate]
    end
    # Consecutive wins/losses
    max_consec_win = 0
    max_consec_loss = 0
    cur_win = 0
    cur_loss = 0
    for pnl in pnls
        if pnl > zero(T)
            cur_win += 1
            cur_loss = 0
            max_consec_win = max(max_consec_win, cur_win)
        else
            cur_loss += 1
            cur_win = 0
            max_consec_loss = max(max_consec_loss, cur_loss)
        end
    end
    metrics[:max_consec_wins] = T(max_consec_win)
    metrics[:max_consec_losses] = T(max_consec_loss)
    # Recovery factor
    cum_pnl = cumsum(pnls)
    peak_pnl = accumulate(max, cum_pnl)
    dd_pnl = peak_pnl .- cum_pnl
    max_dd_pnl = maximum(dd_pnl)
    metrics[:recovery_factor] = metrics[:total_pnl] / max(max_dd_pnl, T(1e-16))
    return metrics
end

"""Per-asset trade breakdown."""
function trade_analytics_by_asset(trades::Vector{TradeRecord{T}},
                                   n_assets::Int) where T<:Real
    results = Dict{Int, Dict{Symbol, T}}()
    for a in 1:n_assets
        asset_trades = filter(t -> t.asset_id == a, trades)
        if !isempty(asset_trades)
            results[a] = trade_analytics(asset_trades)
        end
    end
    results
end

"""Long vs short trade breakdown."""
function trade_analytics_by_direction(trades::Vector{TradeRecord{T}}) where T<:Real
    longs = filter(t -> t.direction > 0, trades)
    shorts = filter(t -> t.direction < 0, trades)
    return (long=trade_analytics(longs), short=trade_analytics(shorts))
end

# ─────────────────────────────────────────────────────────────────────────────
# §13  Regime-Conditional Performance
# ─────────────────────────────────────────────────────────────────────────────

"""
    regime_performance(returns, regime_labels) -> Dict per regime

Performance metrics conditional on market regime.
"""
function regime_performance(returns::AbstractVector{T},
                            regime_labels::AbstractVector{Int}) where T<:Real
    @assert length(returns) == length(regime_labels)
    regimes = unique(regime_labels)
    results = Dict{Int, Dict{Symbol, T}}()
    for r in regimes
        mask = regime_labels .== r
        r_ret = returns[mask]
        if length(r_ret) > 1
            results[r] = compute_metrics(r_ret)
            results[r][:n_days] = T(length(r_ret))
            results[r][:pct_time] = T(length(r_ret)) / T(length(returns))
        end
    end
    results
end

"""Detect regimes from market returns using rolling volatility."""
function detect_vol_regimes(market_returns::AbstractVector{T};
                            window::Int=63, n_regimes::Int=3) where T<:Real
    n = length(market_returns)
    labels = ones(Int, n)
    if n < window
        return labels
    end
    vols = Vector{T}(undef, n)
    for t in 1:n
        lo = max(1, t - window + 1)
        vols[t] = std(market_returns[lo:t])
    end
    # Simple quantile-based regime classification
    sorted_vols = sort(vols[window:end])
    thresholds = [sorted_vols[max(1, round(Int, k * length(sorted_vols) / n_regimes))]
                  for k in 1:n_regimes-1]
    for t in 1:n
        label = 1
        for th in thresholds
            if vols[t] > th
                label += 1
            end
        end
        labels[t] = label
    end
    labels
end

"""Performance attribution by market regime and momentum."""
function regime_momentum_performance(strategy_returns::AbstractVector{T},
                                      market_returns::AbstractVector{T};
                                      vol_window::Int=63,
                                      mom_window::Int=252) where T<:Real
    n = length(strategy_returns)
    vol_regimes = detect_vol_regimes(market_returns; window=vol_window)
    # Momentum regime: up/down based on trailing return
    mom_regimes = ones(Int, n)
    for t in mom_window+1:n
        cum_ret = sum(market_returns[t-mom_window+1:t])
        mom_regimes[t] = cum_ret > zero(T) ? 1 : 2
    end
    # Combined regime
    combined = vol_regimes .* 10 .+ mom_regimes
    regime_performance(strategy_returns, combined)
end

# ─────────────────────────────────────────────────────────────────────────────
# §14  Factor Attribution
# ─────────────────────────────────────────────────────────────────────────────

"""
    factor_attribution(strategy_returns, factor_returns, factor_names) -> Dict

Regress strategy returns on factor returns.
"""
function factor_attribution(strategy_returns::AbstractVector{T},
                            factor_returns::AbstractMatrix{T};
                            factor_names::Vector{String}=String[]) where T<:Real
    n = length(strategy_returns)
    n_f, k = size(factor_returns)
    @assert n == n_f
    X = hcat(ones(T, n), factor_returns)
    beta = (X' * X) \ (X' * strategy_returns)
    predicted = X * beta
    residuals = strategy_returns .- predicted
    ss_res = dot(residuals, residuals)
    ss_tot = dot(strategy_returns .- mean(strategy_returns),
                 strategy_returns .- mean(strategy_returns))
    r_squared = one(T) - ss_res / max(ss_tot, T(1e-16))
    adj_r_squared = one(T) - (one(T) - r_squared) * (n - 1) / max(n - k - 1, 1)
    # T-statistics
    mse = ss_res / max(n - k - 1, 1)
    XtX_inv = inv(X' * X)
    se = sqrt.(max.(diag(XtX_inv) .* mse, T(1e-16)))
    t_stats = beta ./ se
    # Factor contributions
    contributions = zeros(T, k)
    for j in 1:k
        contributions[j] = beta[j+1] * mean(factor_returns[:, j]) * 252
    end
    alpha = beta[1] * 252  # annualized alpha
    result = Dict{Symbol, Any}()
    result[:alpha] = alpha
    result[:betas] = beta[2:end]
    result[:t_stats] = t_stats
    result[:r_squared] = r_squared
    result[:adj_r_squared] = adj_r_squared
    result[:residual_vol] = std(residuals) * sqrt(T(252))
    result[:factor_contributions] = contributions
    result[:information_ratio] = alpha / max(result[:residual_vol], T(1e-16))
    if !isempty(factor_names)
        result[:factor_names] = factor_names
    end
    return result
end

"""Rolling factor attribution."""
function rolling_factor_attribution(strategy_returns::AbstractVector{T},
                                     factor_returns::AbstractMatrix{T};
                                     window::Int=252, step::Int=21) where T<:Real
    n = length(strategy_returns)
    k = size(factor_returns, 2)
    n_windows = div(n - window, step) + 1
    alphas = Vector{T}(undef, n_windows)
    betas = Matrix{T}(undef, n_windows, k)
    r2s = Vector{T}(undef, n_windows)
    for (idx, t) in enumerate(window:step:n)
        r = strategy_returns[t-window+1:t]
        f = factor_returns[t-window+1:t, :]
        attr = factor_attribution(r, f)
        alphas[idx] = attr[:alpha]
        betas[idx, :] = attr[:betas]
        r2s[idx] = attr[:r_squared]
    end
    return (alpha=alphas, betas=betas, r_squared=r2s)
end

"""Brinson-style attribution: allocation, selection, interaction."""
function brinson_attribution(w_port::AbstractVector{T}, w_bench::AbstractVector{T},
                              r_port::AbstractVector{T}, r_bench::AbstractVector{T},
                              sector_map::Dict{Int, Vector{Int}}) where T<:Real
    n_sectors = length(sector_map)
    alloc_effect = zeros(T, n_sectors)
    select_effect = zeros(T, n_sectors)
    interact_effect = zeros(T, n_sectors)
    r_bench_total = dot(w_bench, r_bench)
    for (s, assets) in sector_map
        w_p_s = sum(w_port[assets])
        w_b_s = sum(w_bench[assets])
        if w_b_s > T(1e-16)
            r_b_s = dot(w_bench[assets], r_bench[assets]) / w_b_s
        else
            r_b_s = zero(T)
        end
        if w_p_s > T(1e-16)
            r_p_s = dot(w_port[assets], r_port[assets]) / w_p_s
        else
            r_p_s = zero(T)
        end
        alloc_effect[s] = (w_p_s - w_b_s) * (r_b_s - r_bench_total)
        select_effect[s] = w_b_s * (r_p_s - r_b_s)
        interact_effect[s] = (w_p_s - w_b_s) * (r_p_s - r_b_s)
    end
    return (allocation=alloc_effect, selection=select_effect,
            interaction=interact_effect,
            total_allocation=sum(alloc_effect),
            total_selection=sum(select_effect),
            total_interaction=sum(interact_effect))
end

# ─────────────────────────────────────────────────────────────────────────────
# §15  Drawdown Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""Detailed drawdown table."""
function drawdown_analysis(returns::AbstractVector{T}; top_n::Int=10) where T<:Real
    n = length(returns)
    cum = cumprod(one(T) .+ returns)
    peak = accumulate(max, cum)
    dd_series = (peak .- cum) ./ peak
    # Find drawdown periods
    drawdowns = Vector{NamedTuple{(:start, :trough, :end_, :depth, :duration, :recovery),
                                   Tuple{Int,Int,Int,T,Int,Int}}}()
    in_dd = false
    dd_start = 0
    dd_trough = 0
    dd_max = zero(T)
    for t in 1:n
        if dd_series[t] > T(1e-10)
            if !in_dd
                in_dd = true
                dd_start = t
                dd_max = zero(T)
            end
            if dd_series[t] > dd_max
                dd_max = dd_series[t]
                dd_trough = t
            end
        else
            if in_dd
                push!(drawdowns, (start=dd_start, trough=dd_trough, end_=t,
                                  depth=dd_max, duration=t-dd_start,
                                  recovery=t-dd_trough))
                in_dd = false
            end
        end
    end
    if in_dd
        push!(drawdowns, (start=dd_start, trough=dd_trough, end_=n,
                          depth=dd_max, duration=n-dd_start, recovery=n-dd_trough))
    end
    sort!(drawdowns; by=x -> -x.depth)
    return drawdowns[1:min(top_n, length(drawdowns))]
end

"""Underwater equity curve (drawdown over time)."""
function underwater_curve(returns::AbstractVector{T}) where T<:Real
    cum = cumprod(one(T) .+ returns)
    peak = accumulate(max, cum)
    -(peak .- cum) ./ peak
end

"""Drawdown-at-risk: VaR of drawdown distribution."""
function drawdown_at_risk(returns::AbstractVector{T};
                          alpha::T=T(0.05), window::Int=252) where T<:Real
    n = length(returns)
    if n < window
        return max_drawdown(returns)
    end
    dds = Vector{T}()
    for t in window:n
        r = returns[t-window+1:t]
        push!(dds, max_drawdown(r))
    end
    sorted = sort(dds; rev=true)
    sorted[max(1, ceil(Int, alpha * length(sorted)))]
end

# ─────────────────────────────────────────────────────────────────────────────
# §16  Benchmark Comparison
# ─────────────────────────────────────────────────────────────────────────────

"""Compare strategy against benchmark."""
function benchmark_comparison(strategy_returns::AbstractVector{T},
                               benchmark_returns::AbstractVector{T};
                               rf::T=T(0.02)) where T<:Real
    n = min(length(strategy_returns), length(benchmark_returns))
    sr = strategy_returns[1:n]
    br = benchmark_returns[1:n]
    excess = sr .- br
    result = Dict{Symbol, T}()
    result[:strategy_sharpe] = sharpe_ratio(sr; rf=rf)
    result[:benchmark_sharpe] = sharpe_ratio(br; rf=rf)
    result[:active_return] = (mean(sr) - mean(br)) * 252
    result[:tracking_error] = std(excess) * sqrt(T(252))
    result[:information_ratio] = result[:active_return] / max(result[:tracking_error], T(1e-16))
    result[:beta] = cov(sr, br) / max(var(br), T(1e-16))
    result[:alpha] = (mean(sr) - result[:beta] * mean(br)) * 252
    result[:treynor] = mean(sr .- rf/252) * 252 / max(abs(result[:beta]), T(1e-16))
    result[:up_capture] = mean(sr[br .> 0]) / max(mean(br[br .> 0]), T(1e-16))
    result[:down_capture] = mean(sr[br .< 0]) / max(abs(mean(br[br .< 0])), T(1e-16))
    result[:capture_ratio] = result[:up_capture] / max(result[:down_capture], T(1e-16))
    result[:correlation] = cor(sr, br)
    result[:max_dd_strategy] = max_drawdown(sr)
    result[:max_dd_benchmark] = max_drawdown(br)
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# §17  Strategy Capacity Estimation
# ─────────────────────────────────────────────────────────────────────────────

"""Estimate strategy capacity from market impact."""
function estimate_capacity(returns::AbstractVector{T},
                           avg_daily_volume::T;
                           target_sharpe_decay::T=T(0.5),
                           impact_model::SqrtImpact{T}=SqrtImpact{T}(T(0.1), T(0.02), avg_daily_volume)) where T<:Real
    base_sharpe = sharpe_ratio(returns)
    if base_sharpe <= zero(T)
        return zero(T)
    end
    target_sharpe = base_sharpe * (one(T) - target_sharpe_decay)
    # Binary search for capacity
    lo, hi = T(1e3), T(1e12)
    for _ in 1:100
        mid = (lo + hi) / 2
        participation = mid / avg_daily_volume
        impact_bps = impact_model.eta * impact_model.sigma * sqrt(participation) * T(1e4)
        adj_ret = mean(returns) - impact_bps * T(1e-4) * 2  # round-trip
        adj_vol = std(returns)
        adj_sharpe = adj_ret / max(adj_vol, T(1e-16)) * sqrt(T(252))
        if adj_sharpe > target_sharpe
            lo = mid
        else
            hi = mid
        end
        if (hi - lo) / max(lo, T(1e-10)) < T(1e-4)
            break
        end
    end
    return lo
end

# ─────────────────────────────────────────────────────────────────────────────
# §18  Multi-Strategy Portfolio
# ─────────────────────────────────────────────────────────────────────────────

"""Combine multiple strategy return streams."""
function multi_strategy_backtest(strategy_returns::Matrix{T};
                                 method::Symbol=:equal_weight,
                                 lookback::Int=252) where T<:Real
    n_obs, n_strats = size(strategy_returns)
    weights = Matrix{T}(undef, n_obs, n_strats)
    for t in 1:n_obs
        if method == :equal_weight || t < lookback
            weights[t, :] .= one(T) / n_strats
        elseif method == :inv_vol
            vols = vec(std(strategy_returns[max(1,t-lookback):t-1, :]; dims=1))
            inv_v = one(T) ./ (vols .+ T(1e-10))
            weights[t, :] = inv_v ./ sum(inv_v)
        elseif method == :sharpe_weighted
            sharpes = [sharpe_ratio(strategy_returns[max(1,t-lookback):t-1, i])
                      for i in 1:n_strats]
            pos_sharpes = max.(sharpes, zero(T))
            s = sum(pos_sharpes)
            if s > T(1e-10)
                weights[t, :] = pos_sharpes ./ s
            else
                weights[t, :] .= one(T) / n_strats
            end
        elseif method == :risk_parity
            if t > lookback
                sub = strategy_returns[t-lookback:t-1, :]
                Sigma = cov(sub)
                vols = sqrt.(max.(diag(Sigma), T(1e-16)))
                inv_v = one(T) ./ vols
                weights[t, :] = inv_v ./ sum(inv_v)
            else
                weights[t, :] .= one(T) / n_strats
            end
        end
    end
    combined = vec(sum(strategy_returns .* weights; dims=2))
    return combined, weights
end

# ─────────────────────────────────────────────────────────────────────────────
# §19  Return Decomposition
# ─────────────────────────────────────────────────────────────────────────────

"""Decompose portfolio return into components."""
function return_decomposition(w_start::AbstractVector{T},
                               w_end::AbstractVector{T},
                               asset_returns::AbstractVector{T}) where T<:Real
    n = length(w_start)
    # Drift return: return from holding initial weights
    drift_return = dot(w_start, asset_returns)
    # Rebalance return: return from weight changes
    rebal_return = dot(w_end .- w_start, asset_returns)
    # Cross term
    total = dot(w_end, asset_returns)
    return (drift=drift_return, rebalance=rebal_return, total=total)
end

"""Contribution to portfolio return by asset."""
function return_contribution(w::AbstractVector{T},
                              asset_returns::AbstractVector{T}) where T<:Real
    w .* asset_returns
end

"""Risk contribution by asset."""
function risk_contribution_decomp(w::AbstractVector{T},
                                   Sigma::AbstractMatrix{T}) where T<:Real
    vol = sqrt(max(dot(w, Sigma * w), zero(eltype(w))))
    mrc = Sigma * w ./ max(vol, eltype(w)(1e-16))
    rc = w .* mrc
    rc_pct = rc ./ max(vol, eltype(w)(1e-16))
    return (marginal=mrc, contribution=rc, pct_contribution=rc_pct, total_vol=vol)
end

# ─────────────────────────────────────────────────────────────────────────────
# §20  Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

"""Convert price matrix to return matrix."""
function prices_to_returns(prices::AbstractMatrix{T}) where T<:Real
    n, p = size(prices)
    returns = zeros(T, n, p)
    for t in 2:n
        for i in 1:p
            returns[t, i] = (prices[t, i] - prices[t-1, i]) / prices[t-1, i]
        end
    end
    returns
end

"""Generate synthetic price data for testing."""
function generate_test_prices(n_bars::Int, n_assets::Int;
                               mu::Float64=0.0001, vol::Float64=0.02,
                               rng::AbstractRNG=Random.GLOBAL_RNG)
    returns = mu .+ vol .* randn(rng, n_bars, n_assets)
    prices = zeros(n_bars, n_assets)
    prices[1, :] .= 100.0
    for t in 2:n_bars
        for i in 1:n_assets
            prices[t, i] = prices[t-1, i] * (1.0 + returns[t, i])
        end
    end
    prices
end

"""Generate correlated returns."""
function generate_correlated_returns(n_bars::Int, n_assets::Int;
                                      mu::AbstractVector{Float64}=zeros(0),
                                      corr_strength::Float64=0.3,
                                      vol::Float64=0.02,
                                      rng::AbstractRNG=Random.GLOBAL_RNG)
    if isempty(mu)
        mu = fill(0.0001, n_assets)
    end
    C = fill(corr_strength, n_assets, n_assets)
    for i in 1:n_assets
        C[i,i] = 1.0
    end
    L = cholesky(Symmetric(C)).L
    Z = randn(rng, n_bars, n_assets)
    returns = (Z * L') .* vol .+ mu'
    returns
end

# ─────────────────────────────────────────────────────────────────────────────
# §21  Advanced Fill Models
# ─────────────────────────────────────────────────────────────────────────────

"""Limit order fill probability model."""
function limit_order_fill_prob(limit_price::T, mid_price::T, spread::T,
                                vol::T, duration::T) where T<:Real
    # Probability that price reaches limit within duration
    distance = abs(limit_price - mid_price) / max(vol, T(1e-16))
    # Approximate using Brownian motion first passage time
    if limit_price <= mid_price  # buy limit
        p = exp(-T(2) * distance^2 / max(duration, T(1e-8)))
    else  # sell limit
        p = exp(-T(2) * distance^2 / max(duration, T(1e-8)))
    end
    clamp(p, zero(T), one(T))
end

"""VWAP slippage model: deviation from volume-weighted average price."""
function vwap_slippage(shares::T, total_volume::T, vol::T;
                       participation_limit::T=T(0.1)) where T<:Real
    participation = abs(shares) / max(total_volume, T(1.0))
    if participation > participation_limit
        # Significant market impact
        excess = participation - participation_limit
        return vol * sqrt(participation) + T(2) * vol * excess
    end
    vol * sqrt(participation)
end

"""Implementation shortfall decomposition."""
function implementation_shortfall(decision_price::T, arrival_price::T,
                                   execution_price::T, close_price::T,
                                   shares::T, direction::Int) where T<:Real
    # Total IS = (execution - decision) * direction * shares
    total_is = (execution_price - decision_price) * T(direction) * shares
    # Decomposition
    delay_cost = (arrival_price - decision_price) * T(direction) * shares
    market_impact = (execution_price - arrival_price) * T(direction) * shares
    timing_cost = (close_price - execution_price) * T(direction) * shares
    opportunity_cost = zero(T)  # for unfilled portion
    return (total=total_is, delay=delay_cost, impact=market_impact,
            timing=timing_cost, opportunity=opportunity_cost)
end

# ─────────────────────────────────────────────────────────────────────────────
# §22  Multi-Asset Backtest
# ─────────────────────────────────────────────────────────────────────────────

"""Cross-asset correlation regime detection."""
function cross_asset_regime(returns::AbstractMatrix{T};
                            window::Int=63, n_regimes::Int=3) where T<:Real
    n, p = size(returns)
    labels = ones(Int, n)
    if n < window + 1
        return labels
    end
    avg_corrs = Vector{T}(undef, n)
    for t in window:n
        sub = returns[t-window+1:t, :]
        C = cor(sub)
        total = zero(T)
        count = 0
        for j in 1:p, i in j+1:p
            total += C[i,j]
            count += 1
        end
        avg_corrs[t] = count > 0 ? total / count : zero(T)
    end
    # Quantile-based classification
    valid = avg_corrs[window:end]
    sorted = sort(valid)
    thresholds = [sorted[max(1, round(Int, k * length(sorted) / n_regimes))]
                  for k in 1:n_regimes-1]
    for t in window:n
        label = 1
        for th in thresholds
            if avg_corrs[t] > th
                label += 1
            end
        end
        labels[t] = label
    end
    labels
end

"""Pair trading backtest."""
function pairs_backtest(prices1::AbstractVector{T}, prices2::AbstractVector{T};
                        lookback::Int=60, entry_z::T=T(2.0),
                        exit_z::T=T(0.5), max_hold::Int=20) where T<:Real
    n = length(prices1)
    spread = log.(prices1) .- log.(prices2)
    returns_out = zeros(T, n)
    position = 0  # +1: long spread, -1: short spread, 0: flat
    hold_count = 0
    for t in lookback+1:n
        window = spread[t-lookback:t-1]
        mu = mean(window)
        sigma = std(window)
        z = (spread[t] - mu) / max(sigma, T(1e-16))
        if position == 0
            if z > entry_z
                position = -1  # short spread
            elseif z < -entry_z
                position = 1   # long spread
            end
            hold_count = 0
        else
            hold_count += 1
            if (position == 1 && z >= -exit_z) || (position == -1 && z <= exit_z) || hold_count >= max_hold
                position = 0
            end
        end
        if position != 0
            ret1 = (prices1[t] - prices1[t-1]) / prices1[t-1]
            ret2 = (prices2[t] - prices2[t-1]) / prices2[t-1]
            returns_out[t] = T(position) * (ret1 - ret2)
        end
    end
    returns_out
end

"""Statistical arbitrage basket backtest."""
function stat_arb_basket(returns::AbstractMatrix{T},
                          hedge_ratios::AbstractVector{T},
                          signals::AbstractVector{T};
                          lookback::Int=60,
                          entry_z::T=T(2.0),
                          exit_z::T=T(0.5)) where T<:Real
    n, p = size(returns)
    spread_returns = returns * hedge_ratios
    pnl = zeros(T, n)
    position = zero(T)
    for t in lookback+1:n
        window = spread_returns[t-lookback:t-1]
        mu = mean(window)
        sigma = std(window)
        z = (spread_returns[t] - mu) / max(sigma, T(1e-16))
        signal = signals[t]
        if abs(position) < T(1e-10)
            if z > entry_z && signal < zero(T)
                position = -one(T)
            elseif z < -entry_z && signal > zero(T)
                position = one(T)
            end
        else
            if (position > zero(T) && z >= -exit_z) || (position < zero(T) && z <= exit_z)
                position = zero(T)
            end
        end
        pnl[t] = position * spread_returns[t]
    end
    pnl
end

# ─────────────────────────────────────────────────────────────────────────────
# §23  Risk-Adjusted Performance Attribution
# ─────────────────────────────────────────────────────────────────────────────

"""Fama-French alpha decomposition."""
function ff_alpha_decomposition(strategy_returns::AbstractVector{T},
                                 market_returns::AbstractVector{T};
                                 window::Int=252) where T<:Real
    n = length(strategy_returns)
    alphas = Vector{T}(undef, n)
    betas = Vector{T}(undef, n)
    fill!(alphas, zero(T))
    fill!(betas, zero(T))
    for t in window:n
        sr = strategy_returns[t-window+1:t]
        mr = market_returns[t-window+1:t]
        X = hcat(ones(T, window), mr)
        b = (X' * X) \ (X' * sr)
        alphas[t] = b[1] * 252  # annualized
        betas[t] = b[2]
    end
    return alphas, betas
end

"""Conditional alpha: alpha in different market regimes."""
function conditional_alpha(strategy_returns::AbstractVector{T},
                           market_returns::AbstractVector{T};
                           window::Int=252) where T<:Real
    n = length(strategy_returns)
    up_alpha = T[]
    down_alpha = T[]
    for t in window:n
        sr = strategy_returns[t-window+1:t]
        mr = market_returns[t-window+1:t]
        up_mask = mr .> zero(T)
        down_mask = mr .<= zero(T)
        if count(up_mask) > 5
            push!(up_alpha, mean(sr[up_mask]) - mean(mr[up_mask]))
        end
        if count(down_mask) > 5
            push!(down_alpha, mean(sr[down_mask]) - mean(mr[down_mask]))
        end
    end
    return (up_alpha=isempty(up_alpha) ? zero(T) : mean(up_alpha) * 252,
            down_alpha=isempty(down_alpha) ? zero(T) : mean(down_alpha) * 252)
end

"""Performance persistence test."""
function performance_persistence(returns::AbstractVector{T};
                                  period::Int=63) where T<:Real
    n = length(returns)
    n_periods = div(n, period)
    if n_periods < 3
        return zero(T), zero(T)
    end
    period_sharpes = Vector{T}(undef, n_periods)
    for p in 1:n_periods
        r = returns[(p-1)*period+1 : min(p*period, n)]
        period_sharpes[p] = mean(r) / max(std(r), T(1e-16)) * sqrt(T(252))
    end
    # Autocorrelation of period Sharpes
    mu_s = mean(period_sharpes)
    var_s = var(period_sharpes)
    if var_s < T(1e-16)
        return zero(T), zero(T)
    end
    autocorr = zero(T)
    for i in 2:n_periods
        autocorr += (period_sharpes[i] - mu_s) * (period_sharpes[i-1] - mu_s)
    end
    autocorr /= ((n_periods - 1) * var_s)
    # Hurst exponent approximation (R/S analysis)
    hurst = _hurst_exponent(returns)
    return autocorr, hurst
end

function _hurst_exponent(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    if n < 20
        return T(0.5)
    end
    log_rs = T[]
    log_n = T[]
    for w in [10, 20, 40, 80, 160]
        if w > n ÷ 2 break end
        rs_vals = T[]
        for start in 1:w:n-w+1
            chunk = returns[start:start+w-1]
            mu = mean(chunk)
            cum_dev = cumsum(chunk .- mu)
            R = maximum(cum_dev) - minimum(cum_dev)
            S = std(chunk)
            if S > T(1e-16)
                push!(rs_vals, R / S)
            end
        end
        if !isempty(rs_vals)
            push!(log_rs, log(mean(rs_vals)))
            push!(log_n, log(T(w)))
        end
    end
    if length(log_rs) < 2
        return T(0.5)
    end
    # Linear regression slope
    x = log_n
    y = log_rs
    mx = mean(x)
    my = mean(y)
    num = dot(x .- mx, y .- my)
    den = dot(x .- mx, x .- mx)
    den < T(1e-16) ? T(0.5) : num / den
end

# ─────────────────────────────────────────────────────────────────────────────
# §24  Order Book Simulation
# ─────────────────────────────────────────────────────────────────────────────

"""Simple order book state."""
mutable struct OrderBookState{T<:Real}
    bid_prices::Vector{T}
    bid_sizes::Vector{T}
    ask_prices::Vector{T}
    ask_sizes::Vector{T}
    mid_price::T
    spread::T
end

function OrderBookState(mid::T; spread::T=T(0.01), n_levels::Int=5,
                        level_size::T=T(100.0), tick::T=T(0.01)) where T<:Real
    bids = [mid - spread/2 - (i-1)*tick for i in 1:n_levels]
    asks = [mid + spread/2 + (i-1)*tick for i in 1:n_levels]
    sizes = fill(level_size, n_levels)
    OrderBookState{T}(bids, sizes, asks, copy(sizes), mid, spread)
end

"""Simulate market order execution against order book."""
function simulate_market_order(book::OrderBookState{T}, side::Int, size::T) where T<:Real
    remaining = size
    total_cost = zero(T)
    if side > 0  # buy: consume asks
        for i in eachindex(book.ask_prices)
            if remaining <= zero(T) break end
            fill_qty = min(remaining, book.ask_sizes[i])
            total_cost += fill_qty * book.ask_prices[i]
            remaining -= fill_qty
        end
    else  # sell: consume bids
        for i in eachindex(book.bid_prices)
            if remaining <= zero(T) break end
            fill_qty = min(remaining, book.bid_sizes[i])
            total_cost += fill_qty * book.bid_prices[i]
            remaining -= fill_qty
        end
    end
    filled = size - remaining
    avg_price = filled > zero(T) ? total_cost / filled : book.mid_price
    slippage = abs(avg_price - book.mid_price) / book.mid_price
    return avg_price, filled, slippage
end

"""Estimate market impact from order book depth."""
function order_book_impact(book::OrderBookState{T}, side::Int, size::T) where T<:Real
    avg_price, filled, slippage = simulate_market_order(book, side, size)
    impact_bps = slippage * T(10000)
    return impact_bps, avg_price
end

# ─────────────────────────────────────────────────────────────────────────────
# §25  Strategy Combination and Selection
# ─────────────────────────────────────────────────────────────────────────────

"""Optimal strategy combination via mean-variance of strategy returns."""
function optimal_strategy_combination(strategy_returns::AbstractMatrix{T};
                                       target_vol::T=T(0.10),
                                       lookback::Int=252) where T<:Real
    n_obs, n_strats = size(strategy_returns)
    weights = Matrix{T}(undef, n_obs, n_strats)
    fill!(weights, one(T) / n_strats)
    for t in lookback+1:n_obs
        sub = strategy_returns[t-lookback:t-1, :]
        mu = vec(mean(sub; dims=1))
        Sigma = cov(sub)
        # Maximum Sharpe weights
        Sigma_inv = try
            inv(Symmetric(Sigma))
        catch
            pinv(Sigma)
        end
        w = Sigma_inv * mu
        s = sum(w)
        if abs(s) > T(1e-10)
            w ./= s
        else
            w = fill(one(T)/n_strats, n_strats)
        end
        # Enforce non-negative
        for i in 1:n_strats
            w[i] = max(w[i], zero(T))
        end
        s = sum(w)
        if s > T(1e-10)
            w ./= s
        else
            w = fill(one(T)/n_strats, n_strats)
        end
        # Vol targeting
        port_ret = sub * w
        realized_vol = std(port_ret) * sqrt(T(252))
        scale = target_vol / max(realized_vol, T(1e-10))
        w .*= min(scale, T(3.0))
        weights[t, :] = w
    end
    combined = vec(sum(strategy_returns .* weights; dims=2))
    return combined, weights
end

"""Strategy selection via deflated Sharpe."""
function strategy_selection(strategy_returns::AbstractMatrix{T};
                            significance::T=T(0.05)) where T<:Real
    n_obs, n_strats = size(strategy_returns)
    sharpes = [sharpe_ratio(strategy_returns[:, i]) for i in 1:n_strats]
    # Deflated Sharpe for each strategy
    selected = Int[]
    for i in 1:n_strats
        sr = sharpes[i]
        sk = begin
            r = strategy_returns[:, i]
            mu = mean(r); s = std(r)
            s > T(1e-16) ? sum((r .- mu).^3) / (n_obs * s^3) : zero(T)
        end
        ku = begin
            r = strategy_returns[:, i]
            mu = mean(r); s = std(r)
            s > T(1e-16) ? sum((r .- mu).^4) / (n_obs * s^4) - T(3) : zero(T)
        end
        dsr, p = deflated_sharpe_ratio(sr, n_strats, n_obs; skewness=sk, kurtosis=ku)
        if p < significance
            push!(selected, i)
        end
    end
    return selected, sharpes
end

# ─────────────────────────────────────────────────────────────────────────────
# §26  Turnover Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""Compute turnover time series from weight history."""
function turnover_series(weights::AbstractMatrix{T}) where T<:Real
    n = size(weights, 1)
    turnovers = zeros(T, n)
    for t in 2:n
        turnovers[t] = sum(abs.(weights[t,:] .- weights[t-1,:])) / 2
    end
    turnovers
end

"""Net-of-cost Sharpe ratio."""
function net_sharpe(returns::AbstractVector{T}, turnovers::AbstractVector{T};
                    cost_bps::T=T(10.0), rf::T=T(0.02)) where T<:Real
    costs = turnovers .* cost_bps .* T(1e-4)
    net_returns = returns .- costs
    sharpe_ratio(net_returns; rf=rf)
end

"""Break-even transaction cost."""
function breakeven_cost(returns::AbstractVector{T},
                        turnovers::AbstractVector{T};
                        rf::T=T(0.02)) where T<:Real
    # Find cost_bps where Sharpe = 0
    lo, hi = T(0), T(1000)
    for _ in 1:50
        mid = (lo + hi) / 2
        ns = net_sharpe(returns, turnovers; cost_bps=mid, rf=rf)
        if ns > zero(T)
            lo = mid
        else
            hi = mid
        end
    end
    (lo + hi) / 2
end

"""Half-life of strategy alpha decay."""
function alpha_half_life(returns::AbstractVector{T}; max_lag::Int=252) where T<:Real
    n = length(returns)
    if n < max_lag + 1
        return T(Inf)
    end
    # Regress r_t on r_{t-1}: AR(1)
    y = returns[2:end]
    x = returns[1:end-1]
    beta = dot(x, y) / max(dot(x, x), T(1e-16))
    if beta <= zero(T) || beta >= one(T)
        return T(Inf)
    end
    -log(T(2)) / log(beta)
end

# ─────────────────────────────────────────────────────────────────────────────
# §27  Seasonal and Calendar Effects
# ─────────────────────────────────────────────────────────────────────────────

"""Day-of-week effect analysis."""
function day_of_week_analysis(returns::AbstractVector{T};
                               days_per_week::Int=5) where T<:Real
    n = length(returns)
    dow_returns = [T[] for _ in 1:days_per_week]
    for t in 1:n
        day = mod1(t, days_per_week)
        push!(dow_returns[day], returns[t])
    end
    means = [isempty(d) ? zero(T) : mean(d) for d in dow_returns]
    stds = [length(d) > 1 ? std(d) : zero(T) for d in dow_returns]
    t_stats = means ./ max.(stds ./ sqrt.(T.(max.(length.(dow_returns), 1))), T(1e-16))
    return means, stds, t_stats
end

"""Month-of-year effect analysis."""
function month_of_year_analysis(returns::AbstractVector{T};
                                 days_per_month::Int=21) where T<:Real
    n = length(returns)
    monthly_returns = [T[] for _ in 1:12]
    for t in 1:n
        month = mod1(div(t - 1, days_per_month) + 1, 12)
        push!(monthly_returns[month], returns[t])
    end
    means = [isempty(m) ? zero(T) : mean(m) for m in monthly_returns]
    t_stats = [length(m) > 1 ? mean(m) / (std(m) / sqrt(T(length(m))) + T(1e-16)) : zero(T)
               for m in monthly_returns]
    return means, t_stats
end

"""Turn-of-month effect."""
function turn_of_month_effect(returns::AbstractVector{T};
                               window::Int=3, month_length::Int=21) where T<:Real
    n = length(returns)
    tom_returns = T[]
    non_tom_returns = T[]
    for t in 1:n
        day_in_month = mod1(t, month_length)
        if day_in_month <= window || day_in_month > month_length - window
            push!(tom_returns, returns[t])
        else
            push!(non_tom_returns, returns[t])
        end
    end
    tom_mean = isempty(tom_returns) ? zero(T) : mean(tom_returns)
    non_tom_mean = isempty(non_tom_returns) ? zero(T) : mean(non_tom_returns)
    diff = tom_mean - non_tom_mean
    return (tom_mean=tom_mean, non_tom_mean=non_tom_mean, difference=diff)
end

# ─────────────────────────────────────────────────────────────────────────────
# §28  Market Microstructure Metrics
# ─────────────────────────────────────────────────────────────────────────────

"""Kyle's lambda (price impact coefficient)."""
function kyle_lambda(returns::AbstractVector{T}, volumes::AbstractVector{T}) where T<:Real
    n = length(returns)
    signed_volume = returns .* volumes  # Approximate order flow
    X = hcat(ones(T, n), signed_volume)
    beta = (X' * X) \ (X' * returns)
    beta[2]  # Kyle's lambda
end

"""Amihud illiquidity ratio."""
function amihud_illiquidity(returns::AbstractVector{T},
                            dollar_volumes::AbstractVector{T};
                            window::Int=21) where T<:Real
    n = length(returns)
    illiq = Vector{T}(undef, n)
    for t in 1:n
        lo = max(1, t - window + 1)
        illiq[t] = mean(abs.(returns[lo:t]) ./ max.(dollar_volumes[lo:t], T(1e-10)))
    end
    illiq
end

"""Roll's spread estimator."""
function roll_spread(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    if n < 3
        return zero(T)
    end
    autocovariance = zero(T)
    for t in 2:n
        autocovariance += returns[t] * returns[t-1]
    end
    autocovariance /= (n - 1)
    if autocovariance < zero(T)
        return T(2) * sqrt(-autocovariance)
    else
        return zero(T)
    end
end

"""Corwin-Schultz high-low spread estimator."""
function corwin_schultz_spread(highs::AbstractVector{T},
                                lows::AbstractVector{T}) where T<:Real
    n = length(highs)
    spreads = Vector{T}(undef, n)
    fill!(spreads, zero(T))
    for t in 2:n
        beta = (log(highs[t] / lows[t]))^2 + (log(highs[t-1] / lows[t-1]))^2
        gamma = (log(max(highs[t], highs[t-1]) / min(lows[t], lows[t-1])))^2
        alpha_cs = (sqrt(T(2) * beta) - sqrt(beta)) / (T(3) - T(2) * sqrt(T(2))) -
                   sqrt(gamma / (T(3) - T(2) * sqrt(T(2))))
        spreads[t] = T(2) * (exp(alpha_cs) - one(T)) / (one(T) + exp(alpha_cs))
        spreads[t] = max(spreads[t], zero(T))
    end
    spreads
end

end # module BacktestEngine
