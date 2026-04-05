module ExecutionAnalytics

# ============================================================
# execution_analytics.jl -- Trade Execution Analytics
# ============================================================
# Covers: VWAP/TWAP strategy design and comparison,
# implementation shortfall calculation, slippage analysis,
# market impact modelling (Almgren-Chriss, linear permanent),
# arrival price benchmark, participation rate analysis,
# execution quality metrics (vs POV, IS, VWAP benchmark),
# intraday volume patterns, schedule optimisation,
# pre-trade cost estimation, post-trade TCA.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

struct OrderDetails
    symbol::String
    side::Symbol              # :buy or :sell
    total_shares::Float64
    arrival_price::Float64
    decision_price::Float64   # previous close or trigger price
    horizon_minutes::Int
    urgency::Symbol           # :low, :medium, :high
end

struct VWAPSchedule
    time_buckets::Vector{Int}   # minutes from order start
    participation_rates::Vector{Float64}
    expected_volume::Vector{Float64}
    target_shares::Vector{Float64}
end

struct TWAPSchedule
    n_slices::Int
    slice_size::Float64
    interval_minutes::Int
    total_shares::Float64
end

struct ExecutionRecord
    timestamp::Float64
    shares::Float64
    price::Float64
    side::Symbol
    venue::String
    algo::Symbol
end

struct MarketImpactModel
    permanent_coef::Float64    # gamma in Almgren-Chriss
    temporary_coef::Float64    # eta
    volatility::Float64
    adv::Float64               # average daily volume
    symbol::String
end

struct ExecutionResult
    symbol::String
    total_shares_executed::Float64
    vwap_achieved::Float64
    twap_achieved::Float64
    arrival_price::Float64
    decision_price::Float64
    implementation_shortfall::Float64
    slippage_bps::Float64
    market_impact_bps::Float64
    timing_cost_bps::Float64
    opportunity_cost_bps::Float64
    execution_quality_score::Float64
end

struct AlmgrenChrissParams
    sigma::Float64     # daily vol
    eta::Float64       # temporary impact coefficient
    gamma::Float64     # permanent impact coefficient
    alpha_power::Float64  # power law for temporary impact
    rho::Float64       # autocorrelation of trades
end

# ---- 1. VWAP Schedule Construction ----

function intraday_volume_profile(n_buckets::Int=78)::Vector{Float64}
    # Stylised U-shaped intraday volume profile (78 5-min buckets in 6.5hr session)
    profile = zeros(n_buckets)
    for i in 1:n_buckets
        t = (i - 1) / (n_buckets - 1)   # 0 to 1
        # U-shape: high at open and close
        u_shape = 1.5 * exp(-5*t) + 1.5 * exp(-5*(1-t)) + 0.3
        profile[i] = u_shape
    end
    total = sum(profile)
    return profile ./ total
end

function vwap_schedule(order::OrderDetails, volume_profile::Vector{Float64},
                        participation_cap::Float64=0.3)::VWAPSchedule
    n = length(volume_profile)
    n_active = min(order.horizon_minutes div 5, n)
    buckets = collect(1:n_active) .* 5
    # Proportional to volume profile, capped at participation_cap
    adv_per_bucket = 1e6 / n  # simplified: 1M shares per day
    expected_vol = volume_profile[1:n_active] .* adv_per_bucket .* n
    participation = min.(order.total_shares ./ sum(expected_vol), participation_cap)
    target = expected_vol .* participation
    # Normalise to total order size
    if sum(target) > 0
        target = target ./ sum(target) .* order.total_shares
    end
    return VWAPSchedule(buckets, fill(participation, n_active), expected_vol, target)
end

function twap_schedule(order::OrderDetails)::TWAPSchedule
    n = max(1, order.horizon_minutes div 5)
    slice = order.total_shares / n
    return TWAPSchedule(n, slice, 5, order.total_shares)
end

function pov_schedule(order::OrderDetails, target_pov::Float64=0.1,
                       volume_profile::Vector{Float64}=intraday_volume_profile())::Vector{Float64}
    adv = 1e6
    n = length(volume_profile)
    n_active = min(order.horizon_minutes div 5, n)
    bucket_vol = volume_profile[1:n_active] .* adv
    return min.(bucket_vol .* target_pov, order.total_shares / n_active)
end

# ---- 2. Market Impact Models ----

function almgren_chriss_impact(order_size::Float64, model::MarketImpactModel,
                                 horizon_days::Float64)::NamedTuple
    participation = order_size / (model.adv * horizon_days + 1e-8)
    permanent = model.permanent_coef * model.volatility * participation
    temporary = model.temporary_coef * model.volatility * sqrt(participation / horizon_days)
    total = permanent + temporary
    return (
        permanent_bps  = permanent * 1e4,
        temporary_bps  = temporary * 1e4,
        total_bps      = total * 1e4,
        participation  = participation,
    )
end

function linear_market_impact(order_size::Float64, adv::Float64,
                               volatility::Float64, beta::Float64=0.6)::Float64
    participation = order_size / (adv + 1e-8)
    return beta * volatility * sqrt(participation) * 1e4
end

function almgren_chriss_optimal_trajectory(total_shares::Float64,
                                            horizon::Int,
                                            params::AlmgrenChrissParams,
                                            risk_aversion::Float64=1e-5)
    n = horizon
    kappa2 = risk_aversion * params.sigma^2 / (params.eta + 1e-12)
    kappa  = sqrt(kappa2)
    sinh_kT = sinh(kappa * n)
    trajectory = zeros(n+1)
    trajectory[1] = total_shares
    for i in 1:n
        remaining = total_shares * sinh(kappa*(n-i+1)) / (sinh_kT + 1e-12)
        trajectory[i+1] = remaining
    end
    trade_list = diff(trajectory)  # negative = selling
    return (trajectory=trajectory, trades=-trade_list,
            cost_est=sum(abs.(trade_list) .* params.eta .* abs.(trade_list) ./ total_shares))
end

function twap_vs_vwap_cost(executions::Vector{ExecutionRecord},
                             market_vwap::Float64, market_twap::Float64)::NamedTuple
    exec_vwap = sum(e.shares * e.price for e in executions) /
                (sum(e.shares for e in executions) + 1e-8)
    vwap_slippage = (exec_vwap - market_vwap) / market_vwap * 1e4
    twap_slippage = (exec_vwap - market_twap) / market_twap * 1e4
    side_sign = length(executions) > 0 && executions[1].side == :buy ? 1.0 : -1.0
    return (
        exec_vwap       = exec_vwap,
        vwap_slippage_bps = side_sign * vwap_slippage,
        twap_slippage_bps = side_sign * twap_slippage,
        outperformed_vwap = side_sign * vwap_slippage < 0,
    )
end

# ---- 3. Implementation Shortfall ----

function implementation_shortfall(executions::Vector{ExecutionRecord},
                                    order::OrderDetails)::NamedTuple
    total_shares = sum(e.shares for e in executions)
    exec_vwap = sum(e.shares * e.price for e in executions) / (total_shares + 1e-8)
    dp = order.decision_price; ap = order.arrival_price
    side_sign = order.side == :buy ? 1.0 : -1.0
    # IS components
    delay_cost      = side_sign * (ap - dp) / dp * 1e4
    mkt_impact_cost = side_sign * (exec_vwap - ap) / ap * 1e4
    total_is        = delay_cost + mkt_impact_cost
    slippage_bps    = side_sign * (exec_vwap - ap) / ap * 1e4
    return (
        total_is_bps   = total_is,
        delay_cost_bps = delay_cost,
        mkt_impact_bps = mkt_impact_cost,
        slippage_bps   = slippage_bps,
        exec_vwap      = exec_vwap,
        arrival_price  = ap,
    )
end

function opportunity_cost(unexecuted_shares::Float64, arrival_price::Float64,
                            eod_price::Float64, side::Symbol)::Float64
    sign_val = side == :buy ? 1.0 : -1.0
    price_move = sign_val * (eod_price - arrival_price) / arrival_price * 1e4
    return unexecuted_shares * arrival_price * price_move / 1e4
end

function price_reversion_impact(pre_trade_price::Float64,
                                  execution_prices::Vector{Float64},
                                  post_trade_price::Float64)::NamedTuple
    exec_avg = mean(execution_prices)
    impact      = exec_avg - pre_trade_price
    reversion   = post_trade_price - exec_avg
    permanent   = post_trade_price - pre_trade_price
    temporary   = impact - permanent
    return (
        total_impact_bps    = impact / pre_trade_price * 1e4,
        permanent_bps       = permanent / pre_trade_price * 1e4,
        temporary_bps       = temporary / pre_trade_price * 1e4,
        reversion_bps       = reversion / pre_trade_price * 1e4,
    )
end

# ---- 4. Slippage Analysis ----

function bid_ask_slippage(exec_price::Float64, bid::Float64, ask::Float64,
                           side::Symbol)::Float64
    mid = (bid + ask) / 2
    if side == :buy
        return (exec_price - mid) / mid * 1e4
    else
        return (mid - exec_price) / mid * 1e4
    end
end

function volume_weighted_slippage(executions::Vector{ExecutionRecord},
                                    benchmark_price::Float64, side::Symbol)::Float64
    total_shares = sum(e.shares for e in executions)
    exec_vwap = sum(e.shares * e.price for e in executions) / (total_shares + 1e-8)
    sign_val = side == :buy ? 1.0 : -1.0
    return sign_val * (exec_vwap - benchmark_price) / benchmark_price * 1e4
end

function slippage_by_size(exec_records::Vector{NamedTuple},
                           adv_fractions::Vector{Float64})::Vector{Float64}
    n = length(adv_fractions)
    grouped_slip = zeros(n)
    counts = zeros(Int, n)
    bins = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 1.0]
    for rec in exec_records
        frac = rec.adv_fraction
        bin_idx = searchsortedfirst(bins, frac) - 1
        bin_idx = clamp(bin_idx, 1, n)
        grouped_slip[bin_idx] += rec.slippage_bps
        counts[bin_idx] += 1
    end
    return [counts[i] > 0 ? grouped_slip[i]/counts[i] : NaN for i in 1:n]
end

# ---- 5. Execution Quality Metrics ----

function execution_quality_score(is_bps::Float64, vwap_slippage_bps::Float64,
                                   pov_achieved::Float64, target_pov::Float64)::Float64
    is_score    = clamp(1 - abs(is_bps) / 100.0, 0.0, 1.0)
    vwap_score  = clamp(1 - abs(vwap_slippage_bps) / 50.0, 0.0, 1.0)
    pov_score   = clamp(1 - abs(pov_achieved - target_pov) / target_pov, 0.0, 1.0)
    return (is_score + vwap_score + pov_score) / 3.0
end

function intraday_volume_participation(exec_shares::Vector{Float64},
                                        market_volume::Vector{Float64})::Vector{Float64}
    return exec_shares ./ (market_volume .+ 1e-8)
end

function volume_profile_deviation(actual_schedule::Vector{Float64},
                                   target_schedule::Vector{Float64})::Float64
    n = length(actual_schedule); total_actual = sum(actual_schedule)
    if total_actual < 1e-8; return Inf; end
    actual_pct = actual_schedule ./ total_actual
    target_pct = target_schedule ./ sum(target_schedule)
    return sqrt(mean((actual_pct .- target_pct).^2))
end

# ---- 6. Pre-Trade Cost Estimation ----

function pretrade_cost_estimate(order::OrderDetails, model::MarketImpactModel,
                                  volatility::Float64, adv::Float64)
    horizon_days = order.horizon_minutes / 390.0  # 6.5hr trading day
    pov = order.total_shares / (adv * horizon_days + 1e-8)
    impact = almgren_chriss_impact(order.total_shares, model, horizon_days)
    # Spread cost (half spread)
    spread_bps = 1.0  # assumed 1bp spread
    timing_risk = volatility * sqrt(horizon_days) * 1e4
    return (
        estimated_market_impact_bps = impact.total_bps,
        permanent_impact_bps        = impact.permanent_bps,
        temporary_impact_bps        = impact.temporary_bps,
        spread_cost_bps             = spread_bps,
        timing_risk_1sigma_bps      = timing_risk,
        participation_rate          = pov,
        urgency_adjustment          = order.urgency == :high ? 1.5 :
                                       order.urgency == :low ? 0.7 : 1.0,
    )
end

function optimal_horizon_estimate(order_size::Float64, adv::Float64,
                                    volatility::Float64, risk_aversion::Float64=1.0)::Float64
    # Almgren-Chriss optimal horizon: T* = sqrt(order_size / (adv * risk_aversion * vol^2))
    return sqrt(order_size / (adv * risk_aversion * volatility^2 + 1e-12))
end

function urgency_cost_tradeoff(order_size::Float64, adv::Float64,
                                 volatility::Float64,
                                 horizons_days::Vector{Float64})::Matrix{Float64}
    n = length(horizons_days)
    result = zeros(n, 3)
    for (i, T) in enumerate(horizons_days)
        pov = order_size / (adv * T)
        impact_bps = 0.6 * volatility * sqrt(pov) * 1e4
        timing_risk_bps = volatility * sqrt(T) * 1e4
        result[i, :] = [T, impact_bps, timing_risk_bps]
    end
    return result
end

# ---- 7. Post-Trade TCA ----

function post_trade_tca(executions::Vector{ExecutionRecord},
                          order::OrderDetails,
                          market_vwap::Float64,
                          market_twap::Float64)::ExecutionResult
    total_shares = sum(e.shares for e in executions)
    exec_vwap = sum(e.shares * e.price for e in executions) / (total_shares + 1e-8)
    exec_twap = mean(e.price for e in executions)
    ap = order.arrival_price; dp = order.decision_price
    sign_val = order.side == :buy ? 1.0 : -1.0
    is_bps = sign_val * (exec_vwap - ap) / ap * 1e4
    delay_bps = sign_val * (ap - dp) / dp * 1e4
    slippage = sign_val * (exec_vwap - ap) / ap * 1e4
    vwap_slip = sign_val * (exec_vwap - market_vwap) / market_vwap * 1e4
    qs = execution_quality_score(is_bps, vwap_slip, 0.1, 0.1)
    return ExecutionResult(
        order.symbol, total_shares, exec_vwap, exec_twap,
        ap, dp, is_bps + delay_bps, slippage, vwap_slip, delay_bps,
        0.0, qs
    )
end

function tca_benchmark_comparison(exec_price::Float64, benchmarks::Dict{Symbol,Float64},
                                    side::Symbol)::Dict{Symbol,Float64}
    sign_val = side == :buy ? 1.0 : -1.0
    result = Dict{Symbol,Float64}()
    for (name, bench) in benchmarks
        result[name] = sign_val * (exec_price - bench) / bench * 1e4
    end
    return result
end

function reversion_adjusted_shortfall(executions::Vector{ExecutionRecord},
                                       arrival_price::Float64,
                                       reversion_window_price::Float64,
                                       side::Symbol)::Float64
    exec_vwap = sum(e.shares * e.price for e in executions) /
                (sum(e.shares for e in executions) + 1e-8)
    sign_val = side == :buy ? 1.0 : -1.0
    raw_is       = sign_val * (exec_vwap - arrival_price) / arrival_price * 1e4
    reversion    = sign_val * (arrival_price - reversion_window_price) / arrival_price * 1e4
    return raw_is + 0.5 * reversion
end

# ---- Demo ----

function demo()
    println("=== ExecutionAnalytics Demo ===")

    order = OrderDetails("AAPL", :buy, 100000.0, 185.0, 184.5, 60, :medium)
    profile = intraday_volume_profile(12)
    vwap_sched = vwap_schedule(order, profile)
    twap_sched = twap_schedule(order)
    println("VWAP schedule (12 5-min buckets), total target shares: ",
            round(sum(vwap_sched.target_shares), digits=0))
    println("TWAP slices: ", twap_sched.n_slices, " x ",
            round(twap_sched.slice_size, digits=0), " shares each")

    model = MarketImpactModel(0.5, 1.0, 0.02, 5e6, "AAPL")
    impact = almgren_chriss_impact(order.total_shares, model, 0.5)
    println("\nMarket impact estimate (100k shares, 0.5d horizon):")
    println("  Permanent: ", round(impact.permanent_bps, digits=2), " bps")
    println("  Temporary: ", round(impact.temporary_bps, digits=2), " bps")
    println("  Total:     ", round(impact.total_bps, digits=2), " bps")
    println("  Participation: ", round(impact.participation*100, digits=2), "%")

    ac_params = AlmgrenChrissParams(0.02, 0.1, 0.05, 0.6, 0.0)
    traj = almgren_chriss_optimal_trajectory(100000.0, 10, ac_params, 1e-5)
    println("\nA-C optimal trajectory (10 periods):")
    println("  ", round.(traj.trajectory[1:5], digits=0))
    println("  Estimated cost: ", round(traj.cost_est, digits=2))

    executions = [
        ExecutionRecord(Float64(i)*60, 10000.0, 185.0+0.01*randn(), :buy, "NYSE", :vwap)
        for i in 1:10
    ]
    mkt_vwap = 185.02; mkt_twap = 185.01
    is_result = implementation_shortfall(executions, order)
    println("\nImplementation shortfall:")
    println("  Total IS: ", round(is_result.total_is_bps, digits=2), " bps")
    println("  Market impact component: ", round(is_result.mkt_impact_bps, digits=2), " bps")

    tca = post_trade_tca(executions, order, mkt_vwap, mkt_twap)
    println("\nPost-trade TCA:")
    println("  Exec VWAP: ", round(tca.vwap_achieved, digits=4))
    println("  Slippage vs market VWAP: ", round(tca.market_impact_bps, digits=2), " bps")
    println("  Quality score: ", round(tca.execution_quality_score, digits=4))

    pre = pretrade_cost_estimate(order, model, 0.02, 5e6)
    println("\nPre-trade estimate:")
    println("  Expected impact: ", round(pre.estimated_market_impact_bps, digits=2), " bps")
    println("  Timing risk (1sigma): ", round(pre.timing_risk_1sigma_bps, digits=2), " bps")

    oh = optimal_horizon_estimate(order.total_shares, 5e6, 0.02)
    println("Optimal execution horizon: ", round(oh, digits=3), " days")
end

# ---- Additional Execution Analytics Functions ----

function participation_rate_schedule(volume_forecast::Vector{Float64},
                                      total_shares::Float64,
                                      max_pov::Float64=0.25)::Vector{Float64}
    n = length(volume_forecast)
    target_pov = total_shares / (sum(volume_forecast) + 1e-8)
    pov = clamp(target_pov, 0.0, max_pov)
    return min.(volume_forecast .* pov, total_shares / n)
end

function arrival_price_benchmark(order_size::Float64, arrival_price::Float64,
                                   exec_vwap::Float64, side::Symbol)::Float64
    sign_val = side == :buy ? 1.0 : -1.0
    return sign_val * (exec_vwap - arrival_price) / arrival_price * 1e4
end

function spread_cost_estimate(bid::Float64, ask::Float64, order_size::Float64,
                                price::Float64)::Float64
    half_spread = (ask - bid) / 2
    return half_spread / price * order_size * 1e4
end

function price_path_simulation(S0::Float64, mu::Float64, sigma::Float64,
                                 n_steps::Int, dt::Float64; seed::Int=42)::Vector{Float64}
    state = UInt64(seed); prices = [S0]
    for _ in 1:n_steps
        state = 6364136223846793005*state + 1442695040888963407
        u1 = max(Float64(state>>33)/Float64(2^31), 1e-15)
        state = 6364136223846793005*state + 1442695040888963407
        u2 = max(Float64(state>>33)/Float64(2^31), 1e-15)
        z = sqrt(-2*log(u1))*cos(2pi*u2)
        push!(prices, prices[end]*exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z))
    end
    return prices
end

function simulated_execution_cost(order_size::Float64, adv::Float64,
                                    volatility::Float64, horizon::Float64,
                                    n_sim::Int=1000; seed::Int=42)::NamedTuple
    pov = order_size / (adv * horizon + 1e-8)
    base_impact = 0.6 * volatility * sqrt(pov) * 1e4
    sim_costs = Float64[]
    state = UInt64(seed)
    for _ in 1:n_sim
        state = 6364136223846793005*state + 1442695040888963407
        u1 = max(Float64(state>>33)/Float64(2^31), 1e-15)
        state = 6364136223846793005*state + 1442695040888963407
        u2 = max(Float64(state>>33)/Float64(2^31), 1e-15)
        noise = sqrt(-2*log(u1))*cos(2pi*u2) * volatility * sqrt(horizon) * 1e4 * 0.3
        push!(sim_costs, max(0.0, base_impact + noise))
    end
    return (mean_cost=mean(sim_costs), std_cost=std(sim_costs),
            p10=sort(sim_costs)[max(1,round(Int,0.1*n_sim))],
            p90=sort(sim_costs)[min(n_sim,round(Int,0.9*n_sim))],
            base_estimate=base_impact)
end

function execution_shortfall_decomposition(decision_px::Float64, arrival_px::Float64,
                                             exec_vwap::Float64, eod_px::Float64,
                                             side::Symbol)::NamedTuple
    sign_val = side == :buy ? 1.0 : -1.0
    delay_alpha   = sign_val*(arrival_px - decision_px)/decision_px*1e4
    trading_cost  = sign_val*(exec_vwap - arrival_px)/arrival_px*1e4
    opportunity   = sign_val*(eod_px - exec_vwap)/exec_vwap*1e4
    total         = delay_alpha + trading_cost
    return (delay_alpha=delay_alpha, trading_cost=trading_cost,
            opportunity=opportunity, total_is=total)
end

function market_timing_score(exec_prices::Vector{Float64},
                               market_prices::Vector{Float64},
                               side::Symbol)::Float64
    n = min(length(exec_prices), length(market_prices))
    sign_val = side == :buy ? -1.0 : 1.0
    scores = [sign_val*(exec_prices[i] - market_prices[i]) for i in 1:n]
    return mean(scores) / (std(scores) + 1e-8)
end

function venue_analysis(executions::Vector{ExecutionRecord})
    venue_map = Dict{String, Vector{Float64}}()
    for e in executions
        if !haskey(venue_map, e.venue); venue_map[e.venue] = Float64[]; end
        push!(venue_map[e.venue], e.price)
    end
    return Dict(k => (mean_price=mean(v), count=length(v)) for (k,v) in venue_map)
end

function fill_rate_analysis(exec_shares::Float64, target_shares::Float64)::NamedTuple
    fill_pct = exec_shares / (target_shares + 1e-8) * 100.0
    unfilled  = max(0.0, target_shares - exec_shares)
    return (fill_pct=fill_pct, unfilled_shares=unfilled,
            complete=fill_pct >= 99.9)
end

function transaction_cost_alpha(gross_alpha_bps::Float64,
                                  commission_bps::Float64,
                                  impact_bps::Float64,
                                  spread_bps::Float64,
                                  turnover_annual::Float64)::Float64
    total_cost_one_way = commission_bps + impact_bps + spread_bps
    annual_cost = total_cost_one_way * turnover_annual * 2
    return gross_alpha_bps - annual_cost
end

function vwap_benchmark_vs_close(exec_vwap::Float64, market_vwap::Float64,
                                   closing_price::Float64, side::Symbol)::NamedTuple
    sign_val = side == :buy ? 1.0 : -1.0
    vs_vwap  = sign_val*(exec_vwap - market_vwap)/market_vwap*1e4
    vs_close = sign_val*(exec_vwap - closing_price)/closing_price*1e4
    return (vs_market_vwap=vs_vwap, vs_close=vs_close,
            preferred_benchmark=abs(vs_vwap) < abs(vs_close) ? :vwap : :close)
end


# ---- Execution Analytics Utilities (continued) ----

function execution_lag_analysis(decision_time::Float64,
                                  first_exec_time::Float64,
                                  exec_price::Float64,
                                  decision_price::Float64,
                                  side::Symbol)::NamedTuple
    lag_seconds = first_exec_time - decision_time
    sign_val = side == :buy ? 1.0 : -1.0
    price_move_bps = sign_val*(exec_price - decision_price)/decision_price*1e4
    return (lag_seconds=lag_seconds, price_move_bps=price_move_bps,
            adverse=price_move_bps > 0)
end

function algo_performance_attribution(vwap_slippage::Float64,
                                       timing_slippage::Float64,
                                       mktimpact_slippage::Float64)::NamedTuple
    total = vwap_slippage + timing_slippage + mktimpact_slippage
    pct_timing = timing_slippage/(abs(total)+1e-8)*100
    pct_impact = mktimpact_slippage/(abs(total)+1e-8)*100
    pct_vwap   = vwap_slippage/(abs(total)+1e-8)*100
    return (total_bps=total, timing_pct=pct_timing, impact_pct=pct_impact,
            schedule_pct=pct_vwap)
end

function expected_shortfall_from_vol(sigma_daily::Float64, horizon_days::Int,
                                       confidence::Float64=0.99)::Float64
    z = sqrt(2.0) * erfinv(2*confidence - 1)
    phi_z = exp(-0.5*z^2)/sqrt(2pi)
    es_1d = sigma_daily * phi_z / (1 - confidence + 1e-12)
    return es_1d * sqrt(Float64(horizon_days))
end

function time_in_force_analysis(executions::Vector{ExecutionRecord},
                                  horizon_minutes::Int)::NamedTuple
    if isempty(executions)
        return (completion_pct=0.0, avg_exec_time=NaN, last_exec_time=NaN)
    end
    t_start = executions[1].timestamp; t_end = executions[end].timestamp
    elapsed_min = (t_end - t_start) / 60.0
    pct = min(elapsed_min / horizon_minutes * 100.0, 100.0)
    return (completion_pct=pct, avg_exec_time=elapsed_min, last_exec_time=t_end)
end

function cross_venue_price_dispersion(venues::Vector{SpotMarket})::Float64
    mids = [(v.bid + v.ask)/2 for v in venues]
    return isempty(mids) ? 0.0 : std(mids) / (mean(mids) + 1e-8) * 1e4
end

function order_splitting_optimal(total_shares::Float64, n_slices::Int,
                                   price_impact_per_slice::Float64)::Float64
    slice_size = total_shares / n_slices
    total_impact = n_slices * price_impact_per_slice * sqrt(slice_size / total_shares)
    return total_impact * 1e4
end


# ---- Execution Analytics Utilities (continued) ----

function execution_lag_analysis(decision_time::Float64,
                                  first_exec_time::Float64,
                                  exec_price::Float64,
                                  decision_price::Float64,
                                  side::Symbol)::NamedTuple
    lag_seconds = first_exec_time - decision_time
    sign_val = side == :buy ? 1.0 : -1.0
    price_move_bps = sign_val*(exec_price - decision_price)/decision_price*1e4
    return (lag_seconds=lag_seconds, price_move_bps=price_move_bps,
            adverse=price_move_bps > 0)
end

function algo_performance_attribution(vwap_slippage::Float64,
                                       timing_slippage::Float64,
                                       mktimpact_slippage::Float64)::NamedTuple
    total = vwap_slippage + timing_slippage + mktimpact_slippage
    pct_timing = timing_slippage/(abs(total)+1e-8)*100
    pct_impact = mktimpact_slippage/(abs(total)+1e-8)*100
    pct_vwap   = vwap_slippage/(abs(total)+1e-8)*100
    return (total_bps=total, timing_pct=pct_timing, impact_pct=pct_impact,
            schedule_pct=pct_vwap)
end

function expected_shortfall_from_vol(sigma_daily::Float64, horizon_days::Int,
                                       confidence::Float64=0.99)::Float64
    z = sqrt(2.0) * erfinv(2*confidence - 1)
    phi_z = exp(-0.5*z^2)/sqrt(2pi)
    es_1d = sigma_daily * phi_z / (1 - confidence + 1e-12)
    return es_1d * sqrt(Float64(horizon_days))
end

function time_in_force_analysis(executions::Vector{ExecutionRecord},
                                  horizon_minutes::Int)::NamedTuple
    if isempty(executions)
        return (completion_pct=0.0, avg_exec_time=NaN, last_exec_time=NaN)
    end
    t_start = executions[1].timestamp; t_end = executions[end].timestamp
    elapsed_min = (t_end - t_start) / 60.0
    pct = min(elapsed_min / horizon_minutes * 100.0, 100.0)
    return (completion_pct=pct, avg_exec_time=elapsed_min, last_exec_time=t_end)
end

function cross_venue_price_dispersion(venues::Vector{SpotMarket})::Float64
    mids = [(v.bid + v.ask)/2 for v in venues]
    return isempty(mids) ? 0.0 : std(mids) / (mean(mids) + 1e-8) * 1e4
end

function order_splitting_optimal(total_shares::Float64, n_slices::Int,
                                   price_impact_per_slice::Float64)::Float64
    slice_size = total_shares / n_slices
    total_impact = n_slices * price_impact_per_slice * sqrt(slice_size / total_shares)
    return total_impact * 1e4
end


# ---- Execution Analytics Utilities (continued) ----

function execution_lag_analysis(decision_time::Float64,
                                  first_exec_time::Float64,
                                  exec_price::Float64,
                                  decision_price::Float64,
                                  side::Symbol)::NamedTuple
    lag_seconds = first_exec_time - decision_time
    sign_val = side == :buy ? 1.0 : -1.0
    price_move_bps = sign_val*(exec_price - decision_price)/decision_price*1e4
    return (lag_seconds=lag_seconds, price_move_bps=price_move_bps,
            adverse=price_move_bps > 0)
end

function algo_performance_attribution(vwap_slippage::Float64,
                                       timing_slippage::Float64,
                                       mktimpact_slippage::Float64)::NamedTuple
    total = vwap_slippage + timing_slippage + mktimpact_slippage
    pct_timing = timing_slippage/(abs(total)+1e-8)*100
    pct_impact = mktimpact_slippage/(abs(total)+1e-8)*100
    pct_vwap   = vwap_slippage/(abs(total)+1e-8)*100
    return (total_bps=total, timing_pct=pct_timing, impact_pct=pct_impact,
            schedule_pct=pct_vwap)
end

function expected_shortfall_from_vol(sigma_daily::Float64, horizon_days::Int,
                                       confidence::Float64=0.99)::Float64
    z = sqrt(2.0) * erfinv(2*confidence - 1)
    phi_z = exp(-0.5*z^2)/sqrt(2pi)
    es_1d = sigma_daily * phi_z / (1 - confidence + 1e-12)
    return es_1d * sqrt(Float64(horizon_days))
end

function time_in_force_analysis(executions::Vector{ExecutionRecord},
                                  horizon_minutes::Int)::NamedTuple
    if isempty(executions)
        return (completion_pct=0.0, avg_exec_time=NaN, last_exec_time=NaN)
    end
    t_start = executions[1].timestamp; t_end = executions[end].timestamp
    elapsed_min = (t_end - t_start) / 60.0
    pct = min(elapsed_min / horizon_minutes * 100.0, 100.0)
    return (completion_pct=pct, avg_exec_time=elapsed_min, last_exec_time=t_end)
end

function cross_venue_price_dispersion(venues::Vector{SpotMarket})::Float64
    mids = [(v.bid + v.ask)/2 for v in venues]
    return isempty(mids) ? 0.0 : std(mids) / (mean(mids) + 1e-8) * 1e4
end

function order_splitting_optimal(total_shares::Float64, n_slices::Int,
                                   price_impact_per_slice::Float64)::Float64
    slice_size = total_shares / n_slices
    total_impact = n_slices * price_impact_per_slice * sqrt(slice_size / total_shares)
    return total_impact * 1e4
end

end # module ExecutionAnalytics
