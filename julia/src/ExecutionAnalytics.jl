module ExecutionAnalytics

# ============================================================
# ExecutionAnalytics.jl — VWAP/TWAP, slippage, market impact,
#   transaction cost analysis (pure stdlib Julia)
# ============================================================

using Statistics, LinearAlgebra

export Order, Fill, ExecutionReport
export vwap_benchmark, twap_benchmark, arrival_price_benchmark
export vwap_slippage, twap_slippage, implementation_shortfall
export linear_impact_model, sqrt_impact_model, almgren_chriss_impact
export optimal_vwap_schedule, optimal_twap_schedule
export ac_optimal_trajectory, ac_efficient_frontier
export post_trade_tca, pre_trade_estimate
export spread_cost, opportunity_cost, timing_cost
export market_impact_decay, temporary_impact, permanent_impact
export intraday_volume_profile, volume_participation_rate
export schedule_cost_analysis, reversion_alpha
export broker_performance_score, execution_quality_score
export simulate_twap_execution, simulate_vwap_execution
export shortfall_decomposition, slippage_attribution
export dollar_cost_analysis, fill_rate_analysis

# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

struct Order
    ticker::String
    side::Symbol          # :buy or :sell
    quantity::Float64
    decision_price::Float64  # price at order decision
    limit_price::Float64     # 0.0 = market order
    arrival_time::Float64    # seconds since market open
end

struct Fill
    price::Float64
    quantity::Float64
    timestamp::Float64
    venue::String
end

struct ExecutionReport
    order::Order
    fills::Vector{Fill}
    benchmark_price::Float64
    benchmark_type::Symbol   # :vwap, :twap, :arrival
end

# ──────────────────────────────────────────────────────────────
# Benchmark computations
# ──────────────────────────────────────────────────────────────

"""
    vwap_benchmark(prices, volumes) -> vwap

Volume-weighted average price over a trading window.
"""
function vwap_benchmark(prices::Vector{Float64}, volumes::Vector{Float64})
    total_vol = sum(volumes)
    if total_vol < 1e-12; return mean(prices); end
    return dot(prices, volumes) / total_vol
end

"""
    twap_benchmark(prices, timestamps) -> twap

Time-weighted average price over a trading window.
"""
function twap_benchmark(prices::Vector{Float64}, timestamps::Vector{Float64})
    n = length(prices)
    if n == 1; return prices[1]; end
    durations = diff(timestamps)
    total_time = sum(durations)
    if total_time < 1e-12; return mean(prices); end
    return sum(prices[i] * durations[i] for i in 1:n-1) / total_time
end

"""
    arrival_price_benchmark(order) -> decision price

The arrival price (decision price) at order submission.
"""
arrival_price_benchmark(order::Order) = order.decision_price

"""
    interval_vwap(prices, volumes, start_idx, end_idx) -> vwap

VWAP for a sub-interval [start, end] of the trading day.
"""
function interval_vwap(prices::Vector{Float64}, volumes::Vector{Float64},
                         start_idx::Int, end_idx::Int)
    p = prices[start_idx:end_idx]
    v = volumes[start_idx:end_idx]
    return vwap_benchmark(p, v)
end

# ──────────────────────────────────────────────────────────────
# Slippage and implementation shortfall
# ──────────────────────────────────────────────────────────────

"""
    vwap_slippage(fills, market_vwap) -> slippage_bps

Slippage of execution vs VWAP benchmark (in basis points).
"""
function vwap_slippage(fills::Vector{Fill}, market_vwap::Float64, side::Symbol=:buy)
    if isempty(fills); return 0.0; end
    exec_vwap = vwap_benchmark([f.price for f in fills], [f.quantity for f in fills])
    raw_slip = (exec_vwap - market_vwap) / market_vwap
    # For buy orders, paying above VWAP is negative; for sells, below is negative
    slip = side == :buy ? raw_slip : -raw_slip
    return slip * 10_000.0
end

"""
    twap_slippage(fills, market_twap, side) -> slippage_bps
"""
function twap_slippage(fills::Vector{Fill}, market_twap::Float64, side::Symbol=:buy)
    if isempty(fills); return 0.0; end
    exec_vwap = vwap_benchmark([f.price for f in fills], [f.quantity for f in fills])
    raw_slip = (exec_vwap - market_twap) / market_twap
    slip = side == :buy ? raw_slip : -raw_slip
    return slip * 10_000.0
end

"""
    implementation_shortfall(decision_price, exec_price, end_price,
                               quantity, filled_quantity, side)

Implementation shortfall decomposition (Perold 1988).
Returns (total_IS, realized_cost, opportunity_cost) in bps.
"""
function implementation_shortfall(decision_price::Float64, exec_price::Float64,
                                    end_price::Float64, quantity::Float64,
                                    filled_quantity::Float64, side::Symbol=:buy)
    sign = side == :buy ? 1.0 : -1.0
    # Realized cost: cost of filled portion vs decision price
    realized_cost = sign * (exec_price - decision_price) / decision_price

    # Opportunity cost: cost of unfilled portion (missed alpha)
    unfilled = quantity - filled_quantity
    opp_cost = sign * (end_price - decision_price) / decision_price * (unfilled / quantity)

    total_IS = realized_cost + opp_cost
    return total_IS * 10_000.0, realized_cost * 10_000.0, opp_cost * 10_000.0
end

"""
    spread_cost(bid::Float64, ask::Float64) -> half_spread_bps

Half bid-ask spread cost in basis points.
"""
function spread_cost(bid::Float64, ask::Float64)
    mid = (bid + ask) / 2.0
    half_spread = (ask - bid) / 2.0
    return half_spread / mid * 10_000.0
end

"""
    timing_cost(arrival_price, current_price, side) -> cost_bps

Cost from market drift during order execution.
"""
function timing_cost(arrival_price::Float64, current_price::Float64, side::Symbol=:buy)
    sign = side == :buy ? 1.0 : -1.0
    return sign * (current_price - arrival_price) / arrival_price * 10_000.0
end

"""
    opportunity_cost(decision_price, close_price, fill_rate, side) -> opp_cost_bps

Opportunity cost from partial fills.
"""
function opportunity_cost(decision_price::Float64, close_price::Float64,
                            fill_rate::Float64, side::Symbol=:buy)
    sign = side == :buy ? 1.0 : -1.0
    unfilled_fraction = 1.0 - fill_rate
    return sign * (close_price - decision_price) / decision_price * unfilled_fraction * 10_000.0
end

# ──────────────────────────────────────────────────────────────
# Market impact models
# ──────────────────────────────────────────────────────────────

"""
    linear_impact_model(quantity, avg_daily_volume, volatility, eta) -> impact_bps

Linear market impact: I = eta * (Q/ADV) * sigma.
"""
function linear_impact_model(quantity::Float64, adv::Float64,
                               volatility::Float64, eta::Float64=0.1)
    pov = quantity / max(adv, 1.0)  # participation rate
    return eta * pov * volatility * 10_000.0
end

"""
    sqrt_impact_model(quantity, adv, volatility, gamma) -> impact_bps

Square-root market impact (Almgren et al. 2005):
I = gamma * sigma * sqrt(Q/ADV)
"""
function sqrt_impact_model(quantity::Float64, adv::Float64,
                             volatility::Float64, gamma::Float64=0.314)
    pov = quantity / max(adv, 1.0)
    return gamma * volatility * sqrt(pov) * 10_000.0
end

"""
    almgren_chriss_impact(quantity, adv, volatility, T, n_slices, eta, gamma)

Almgren-Chriss (2000) total cost: temporary + permanent impact.
T = trading horizon (days), n_slices = number of child orders.
eta = temporary impact coefficient, gamma = permanent impact.
Returns (temporary_cost, permanent_cost, total_cost) in bps.
"""
function almgren_chriss_impact(quantity::Float64, adv::Float64,
                                  volatility::Float64, T::Float64=1.0,
                                  n_slices::Int=10, eta::Float64=0.1,
                                  gamma::Float64=0.314)
    pov = quantity / max(adv, 1.0)
    slice_size = quantity / n_slices
    slice_pov = slice_size / (adv / 390.0)  # per minute participation

    # Temporary impact per slice
    temp_per_slice = eta * volatility * sqrt(slice_pov)
    total_temp = temp_per_slice * n_slices * 10_000.0

    # Permanent impact (proportional to total quantity)
    perm = gamma * volatility * pov * 10_000.0

    return total_temp, perm, total_temp + perm
end

"""
    temporary_impact(rate, adv, sigma, eta) -> impact_bps

Temporary impact as a function of trading rate.
rate = shares per unit time, adv = average daily volume.
"""
function temporary_impact(rate::Float64, adv::Float64, sigma::Float64,
                            eta::Float64=0.1)
    v = rate / max(adv / 390.0, 1.0)  # normalized by per-minute volume
    return eta * sigma * sqrt(v) * 10_000.0
end

"""
    permanent_impact(quantity, adv, sigma, gamma) -> impact_bps

Permanent price impact (persistent after trade completion).
"""
function permanent_impact(quantity::Float64, adv::Float64, sigma::Float64,
                            gamma::Float64=0.5)
    pov = quantity / max(adv, 1.0)
    return gamma * sigma * pov * 10_000.0
end

"""
    market_impact_decay(impact_at_t0, t, decay_halflife) -> current_impact

Exponential decay of market impact over time.
"""
function market_impact_decay(impact0::Float64, t::Float64, halflife::Float64=30.0)
    return impact0 * exp(-log(2.0) / halflife * t)
end

# ──────────────────────────────────────────────────────────────
# Almgren-Chriss optimal execution
# ──────────────────────────────────────────────────────────────

"""
    ac_optimal_trajectory(X, T, n, sigma, eta, gamma, lambda) -> (holdings, costs)

Almgren-Chriss optimal liquidation trajectory.
X = initial holdings, T = horizon (days), n = number of periods,
sigma = daily vol, eta = temp impact, gamma = perm impact, lambda = risk aversion.
"""
function ac_optimal_trajectory(X::Float64, T::Float64, n::Int,
                                  sigma::Float64, eta::Float64,
                                  gamma::Float64, lambda::Float64)
    tau = T / n
    # Decay parameter kappa
    # kappa = sqrt(lambda * sigma^2 / (eta * (1 - tau * gamma/(2*eta))))
    temp = lambda * sigma^2 / max(eta, 1e-12)
    kappa2 = temp * max(1.0 - tau * gamma / (2*eta), 0.01)
    kappa = sqrt(max(kappa2, 0.0))

    # Optimal holdings trajectory
    holdings = zeros(n + 1)
    for j in 0:n
        t_j = j * tau
        # sinh-based optimal trajectory
        if kappa * T > 1e-8
            holdings[j+1] = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
        else
            holdings[j+1] = X * (T - t_j) / T
        end
    end

    # Compute trade sizes and costs
    trades = -diff(holdings)
    temp_costs = sum(eta * (t / tau)^2 * tau for t in trades) * 10_000.0 / X
    perm_costs = sum(gamma * abs(t) for t in trades) * 10_000.0 / X

    return holdings, temp_costs, perm_costs
end

"""
    ac_efficient_frontier(X, T, n, sigma, eta, gamma, lambdas) -> (risks, costs)

Almgren-Chriss efficient frontier of expected cost vs risk.
"""
function ac_efficient_frontier(X::Float64, T::Float64, n::Int,
                                  sigma::Float64, eta::Float64, gamma::Float64,
                                  lambdas::Vector{Float64}=collect(range(0.001, 1.0, length=50)))
    risks = zeros(length(lambdas))
    costs = zeros(length(lambdas))
    for (i, lambda) in enumerate(lambdas)
        holdings, tc, pc = ac_optimal_trajectory(X, T, n, sigma, eta, gamma, lambda)
        trades = -diff(holdings)
        # Variance of trajectory
        var_traj = sigma^2 * sum(h^2 for h in holdings) * T / n
        risks[i] = sqrt(var_traj) / X * 10_000.0
        costs[i] = tc + pc
    end
    return risks, costs
end

"""
    optimal_vwap_schedule(adv_profile, target_quantity) -> trade_schedule

Compute VWAP-optimal trade schedule proportional to volume profile.
adv_profile: expected volume at each time interval.
"""
function optimal_vwap_schedule(adv_profile::Vector{Float64}, target_quantity::Float64)
    total_vol = sum(adv_profile)
    if total_vol < 1e-12; return fill(target_quantity / length(adv_profile), length(adv_profile)); end
    return target_quantity .* adv_profile ./ total_vol
end

"""
    optimal_twap_schedule(n_periods, target_quantity) -> trade_schedule

Uniform time-slice TWAP schedule.
"""
function optimal_twap_schedule(n_periods::Int, target_quantity::Float64)
    return fill(target_quantity / n_periods, n_periods)
end

# ──────────────────────────────────────────────────────────────
# Intraday volume profile
# ──────────────────────────────────────────────────────────────

"""
    intraday_volume_profile(n_intervals, pattern) -> profile_weights

Generate typical U-shaped intraday volume profile.
pattern ∈ :u_shaped, :flat, :morning_heavy
"""
function intraday_volume_profile(n_intervals::Int=390, pattern::Symbol=:u_shaped)
    profile = zeros(n_intervals)
    if pattern == :u_shaped
        # Classic U-shape: high open, high close, lower midday
        for i in 1:n_intervals
            t = (i - 1) / (n_intervals - 1)  # 0 to 1
            # Mixture of two half-normals at endpoints
            profile[i] = 0.3 * exp(-((t - 0.0)^2) / 0.02) +
                          0.3 * exp(-((t - 1.0)^2) / 0.02) +
                          0.1 + 0.3 * exp(-((t - 0.5)^2) / 0.05)
        end
    elseif pattern == :morning_heavy
        for i in 1:n_intervals
            t = (i - 1) / (n_intervals - 1)
            profile[i] = exp(-3.0 * t) + 0.2
        end
    else  # flat
        fill!(profile, 1.0)
    end
    profile ./= sum(profile)
    return profile
end

"""
    volume_participation_rate(child_order_qty, market_volume) -> pov

Participation rate of a child order in market volume.
"""
volume_participation_rate(child_qty::Float64, market_vol::Float64) =
    child_qty / max(market_vol, 1.0)

# ──────────────────────────────────────────────────────────────
# Pre-trade and post-trade TCA
# ──────────────────────────────────────────────────────────────

"""
    pre_trade_estimate(quantity, adv, sigma, spread_bps, T_hours, lambda) -> estimated_cost_bps

Pre-trade cost estimate combining spread, impact, and timing risk.
"""
function pre_trade_estimate(quantity::Float64, adv::Float64, sigma::Float64,
                              spread_bps::Float64, T_hours::Float64=6.5,
                              lambda::Float64=1e-6)
    pov = quantity / adv
    # Market impact (sqrt model)
    impact = 0.314 * sigma * sqrt(pov) * 10_000.0
    # Spread cost
    spread = spread_bps / 2.0
    # Timing risk: sigma * sqrt(T) scaled
    T_days = T_hours / 6.5
    timing = sigma * sqrt(T_days * pov) * 10_000.0 * lambda
    return spread + impact + timing
end

"""
    post_trade_tca(report) -> TCA summary NamedTuple
"""
function post_trade_tca(report::ExecutionReport)
    fills = report.fills
    order = report.order
    if isempty(fills)
        return (total_cost_bps=0.0, fill_rate=0.0, avg_price=order.decision_price,
                benchmark_slippage_bps=0.0)
    end
    total_qty = sum(f.quantity for f in fills)
    avg_price = vwap_benchmark([f.price for f in fills], [f.quantity for f in fills])
    fill_rate = total_qty / order.quantity
    sign = order.side == :buy ? 1.0 : -1.0
    cost_bps = sign * (avg_price - report.benchmark_price) / report.benchmark_price * 10_000.0
    return (total_cost_bps=cost_bps, fill_rate=fill_rate,
            avg_price=avg_price, benchmark_slippage_bps=cost_bps)
end

"""
    shortfall_decomposition(decision_price, exec_prices, exec_quantities,
                             market_prices, side) -> decomposition

Decompose implementation shortfall into components.
"""
function shortfall_decomposition(decision_price::Float64,
                                   exec_prices::Vector{Float64},
                                   exec_quantities::Vector{Float64},
                                   market_prices::Vector{Float64},
                                   side::Symbol=:buy)
    n = length(exec_prices)
    sign = side == :buy ? 1.0 : -1.0
    total_qty = sum(exec_quantities)

    # Delay cost: market drift from decision to first fill
    delay_cost = 0.0
    if n > 0
        first_market = market_prices[1]
        delay_cost = sign * (first_market - decision_price) / decision_price
    end

    # Trading cost: execution vs contemporaneous market
    trading_cost = 0.0
    for i in 1:n
        w = exec_quantities[i] / max(total_qty, 1.0)
        trading_cost += sign * w * (exec_prices[i] - market_prices[i]) / market_prices[i]
    end

    # Market impact (permanent): price moved from first to last trade
    perm_impact = 0.0
    if n > 1
        perm_impact = sign * (market_prices[end] - market_prices[1]) / market_prices[1]
    end

    return (delay_bps=delay_cost*10_000, trading_bps=trading_cost*10_000,
            permanent_impact_bps=perm_impact*10_000,
            total_bps=(delay_cost+trading_cost+perm_impact)*10_000)
end

# ──────────────────────────────────────────────────────────────
# Execution simulation
# ──────────────────────────────────────────────────────────────

"""
    simulate_twap_execution(S0, sigma, target_qty, n_slices, spread_bps, impact_eta)

Simulate TWAP execution over n_slices uniform time intervals.
Returns (fills, avg_exec_price, total_cost_bps).
"""
function simulate_twap_execution(S0::Float64, sigma::Float64,
                                   target_qty::Float64, n_slices::Int=10,
                                   spread_bps::Float64=5.0, impact_eta::Float64=0.1,
                                   adv::Float64=1e6)
    slice_qty = target_qty / n_slices
    slice_sigma = sigma / sqrt(390.0)  # per minute vol
    prices = zeros(n_slices)
    state = UInt64(42)
    S = S0
    for i in 1:n_slices
        state = state * 6364136223846793005 + 1442695040888963407
        u1 = max((state >> 11) / Float64(2^53), 1e-15)
        state = state * 6364136223846793005 + 1442695040888963407
        u2 = (state >> 11) / Float64(2^53)
        z = sqrt(-2.0 * log(u1)) * cos(2π * u2)
        S *= exp(-0.5*slice_sigma^2 + slice_sigma * z)
        # Add spread and impact
        impact = impact_eta * sigma * sqrt(slice_qty / (adv / n_slices))
        prices[i] = S * (1.0 + spread_bps/20_000.0 + impact/10_000.0)
    end
    avg_price = mean(prices)
    cost_bps = (avg_price - S0) / S0 * 10_000.0
    return prices, avg_price, cost_bps
end

"""
    simulate_vwap_execution(S0, sigma, volume_profile, target_qty, spread_bps, impact_eta)

Simulate VWAP execution following the market volume profile.
"""
function simulate_vwap_execution(S0::Float64, sigma::Float64,
                                   volume_profile::Vector{Float64},
                                   target_qty::Float64,
                                   spread_bps::Float64=5.0,
                                   impact_eta::Float64=0.1,
                                   adv::Float64=1e6)
    n = length(volume_profile)
    slice_sigma = sigma / sqrt(Float64(n))
    prices = zeros(n)
    volumes = zeros(n)
    market_vwap_prices = zeros(n)
    state = UInt64(123)
    S = S0
    for i in 1:n
        state = state * 6364136223846793005 + 1442695040888963407
        u1 = max((state >> 11) / Float64(2^53), 1e-15)
        state = state * 6364136223846793005 + 1442695040888963407
        u2 = (state >> 11) / Float64(2^53)
        z = sqrt(-2.0 * log(u1)) * cos(2π * u2)
        S *= exp(-0.5*slice_sigma^2 + slice_sigma * z)
        market_vwap_prices[i] = S
        vol_i = adv * volume_profile[i]
        slice_qty = target_qty * volume_profile[i]
        impact = impact_eta * sigma * sqrt(slice_qty / max(vol_i, 1.0))
        prices[i] = S * (1.0 + spread_bps/20_000.0 + impact/10_000.0)
        volumes[i] = slice_qty
    end
    exec_vwap = vwap_benchmark(prices, volumes)
    market_vwap = vwap_benchmark(market_vwap_prices, adv .* volume_profile)
    slippage = (exec_vwap - market_vwap) / market_vwap * 10_000.0
    return prices, exec_vwap, market_vwap, slippage
end

# ──────────────────────────────────────────────────────────────
# Analytics and scoring
# ──────────────────────────────────────────────────────────────

"""
    broker_performance_score(tca_results, benchmark_type) -> score

Aggregate broker performance score based on historical TCA.
tca_results: vector of cost_bps per order.
"""
function broker_performance_score(costs_bps::Vector{Float64})
    if isempty(costs_bps); return 50.0; end
    avg_cost = mean(costs_bps)
    consistency = std(costs_bps)
    # Score: lower average cost and lower variability = higher score
    # Normalize to [0, 100]
    cost_score = max(0.0, 100.0 - 10.0 * avg_cost)  # 0 cost = 100, 10 bps = 0
    consistency_score = max(0.0, 100.0 - 5.0 * consistency)
    return 0.6 * cost_score + 0.4 * consistency_score
end

"""
    execution_quality_score(slippage_bps, fill_rate, speed_score) -> quality

Composite execution quality score [0, 100].
"""
function execution_quality_score(slippage_bps::Float64, fill_rate::Float64,
                                   speed_score::Float64=1.0)
    slip_component = max(0.0, 100.0 - 20.0 * abs(slippage_bps))
    fill_component = fill_rate * 100.0
    speed_component = clamp(speed_score * 100.0, 0.0, 100.0)
    return 0.5 * slip_component + 0.3 * fill_component + 0.2 * speed_component
end

"""
    schedule_cost_analysis(target_schedule, executed_schedule, prices) -> analysis

Compare intended vs actual execution schedule.
"""
function schedule_cost_analysis(target::Vector{Float64}, executed::Vector{Float64},
                                  prices::Vector{Float64})
    n = length(prices)
    target_vwap = vwap_benchmark(prices, target)
    exec_vwap = vwap_benchmark(prices, executed)
    tracking_error = norm(target .- executed) / max(sum(target), 1.0)
    schedule_slippage = (exec_vwap - target_vwap) / target_vwap * 10_000.0
    return (target_vwap=target_vwap, exec_vwap=exec_vwap,
            tracking_error=tracking_error, schedule_slippage_bps=schedule_slippage)
end

"""
    reversion_alpha(price_series, impact_series, decay_halflife) -> reversion_return

Expected alpha from post-impact price reversion.
"""
function reversion_alpha(price_series::Vector{Float64}, impact_series::Vector{Float64},
                           decay_halflife::Float64=30.0)
    n = length(price_series)
    lambda = log(2.0) / decay_halflife
    total_reversion = 0.0
    for i in 1:n
        # Future reversion = impact * (1 - already_decayed)
        time_remaining = n - i
        remaining_impact = impact_series[i] * exp(-lambda * time_remaining)
        total_reversion += remaining_impact
    end
    return -total_reversion / max(n, 1)  # negative: reversion captures previous impact
end

"""
    fill_rate_analysis(orders, fills_per_order) -> analysis

Analyze fill rates by order type, size, and market condition.
"""
function fill_rate_analysis(quantities::Vector{Float64},
                              filled_quantities::Vector{Float64},
                              spreads_bps::Vector{Float64})
    n = length(quantities)
    fill_rates = filled_quantities ./ max.(quantities, 1e-12)
    # Correlation of fill rate with spread
    r_spread = cor(fill_rates, spreads_bps)
    # Fill rate by size quartile
    size_quartiles = quantile(sort(quantities), [0.25, 0.50, 0.75])
    q1_mask = quantities .<= size_quartiles[1]
    q4_mask = quantities .> size_quartiles[3]
    q1_fill = mean(fill_rates[q1_mask])
    q4_fill = mean(fill_rates[q4_mask])
    return (mean_fill_rate=mean(fill_rates), std_fill_rate=std(fill_rates),
            small_order_fill=q1_fill, large_order_fill=q4_fill,
            spread_correlation=r_spread)
end

"""
    dollar_cost_analysis(quantities, exec_prices, decision_prices, side) -> stats

Dollar-cost analysis of a set of orders.
"""
function dollar_cost_analysis(quantities::Vector{Float64},
                                exec_prices::Vector{Float64},
                                decision_prices::Vector{Float64},
                                side::Symbol=:buy)
    sign = side == :buy ? 1.0 : -1.0
    dollar_costs = sign .* (exec_prices .- decision_prices) .* quantities
    cost_bps = sign .* (exec_prices .- decision_prices) ./ decision_prices .* 10_000.0
    return (total_dollar_cost=sum(dollar_costs), mean_cost_bps=mean(cost_bps),
            median_cost_bps=median(cost_bps), std_cost_bps=std(cost_bps),
            n_orders=length(quantities))
end

"""
    slippage_attribution(total_slippage_bps, spread_bps, impact_bps,
                           timing_bps) -> fractions

Attribute total slippage to components.
"""
function slippage_attribution(total::Float64, spread::Float64,
                                impact::Float64, timing::Float64)
    components = Dict(:spread=>spread, :impact=>impact, :timing=>timing)
    other = total - spread - impact - timing
    components[:other] = other
    total_abs = sum(abs.(values(components))) + 1e-12
    fractions = Dict(k => abs(v)/total_abs for (k,v) in components)
    return components, fractions
end

end # module ExecutionAnalytics
