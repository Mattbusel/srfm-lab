# ============================================================
# Notebook 30: Transaction Cost Analysis & Execution Quality
# ============================================================
# Topics:
#   1. Market impact model comparison (linear, sqrt, AC)
#   2. VWAP and TWAP simulation and benchmarking
#   3. Implementation shortfall decomposition
#   4. Almgren-Chriss optimal execution
#   5. Intraday volume profile modeling
#   6. Broker performance analysis
#   7. Liquidity-adjusted portfolio construction
#   8. Capacity analysis
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 30: Execution Cost Analysis")
println("="^60)

# ── RNG ───────────────────────────────────────────────────
state_rng = UInt64(42)
function rnd()
    global state_rng
    state_rng = state_rng * 6364136223846793005 + 1442695040888963407
    return (state_rng >> 11) / Float64(2^53)
end
function rndn()
    u1 = max(rnd(), 1e-15); u2 = rnd()
    return sqrt(-2.0*log(u1)) * cos(2π*u2)
end

# ── Section 1: Market Impact Models ──────────────────────

println("\n--- Section 1: Market Impact Model Comparison ---")

# Parameters
sigma_daily = 0.02       # 2% daily vol
adv = 10_000_000.0       # $10M ADV

quantities_pct_adv = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
quantities = quantities_pct_adv .* adv

println("Impact comparison (sigma=2%, ADV=\$10M):")
println("  Q/ADV  | Linear (bps) | Sqrt (bps) | AC Total (bps)")
println("  " * "-"^50)
for (q, pov) in zip(quantities, quantities_pct_adv)
    # Linear: I = eta * (Q/ADV) * sigma
    linear_impact = 0.10 * pov * sigma_daily * 10_000
    # Square-root: I = gamma * sigma * sqrt(Q/ADV)
    sqrt_impact = 0.314 * sigma_daily * sqrt(pov) * 10_000
    # Almgren-Chriss: temp + perm (with T=1 day)
    T_days = 1.0; n_slices = 10; eta = 0.1; gamma_ac = 0.314
    slice_pov = pov / n_slices / (1.0 / 390.0)  # per-minute participation
    temp = eta * sigma_daily * sqrt(slice_pov) * n_slices * 10_000
    perm = gamma_ac * sigma_daily * pov * 10_000
    ac_total = temp + perm
    println("  $(lpad(round(pov*100,digits=0),5))%  | $(lpad(round(linear_impact,digits=1),12)) | $(lpad(round(sqrt_impact,digits=1),10)) | $(round(ac_total, digits=1))")
end

# ── Section 2: VWAP Execution Simulation ─────────────────

println("\n--- Section 2: VWAP Execution Simulation ---")

# Simulate intraday volume profile (U-shaped, 390 minutes)
n_minutes = 390
function u_shaped_volume_profile(n)
    profile = zeros(n)
    for i in 1:n
        t = (i - 1) / (n - 1)
        profile[i] = 0.35 * exp(-t^2 / 0.01) +
                      0.35 * exp(-(t-1.0)^2 / 0.01) +
                      0.10 + 0.20 * exp(-(t-0.5)^2 / 0.05)
    end
    profile ./= sum(profile)
    return profile
end

vol_profile = u_shaped_volume_profile(n_minutes)

println("Volume profile summary (U-shape):")
println("  Open (first 30 min):  $(round(sum(vol_profile[1:30])*100, digits=1))% of daily volume")
println("  Midday (150-240 min): $(round(sum(vol_profile[150:240])*100, digits=1))% of daily volume")
println("  Close (last 30 min):  $(round(sum(vol_profile[361:390])*100, digits=1))% of daily volume")

# Simulate VWAP execution
S0 = 100.0; sigma = 0.02; target_qty = 0.05 * adv
eta_impact = 0.1; spread_bps = 5.0

function simulate_execution(S0, sigma, vol_prof, target_qty, schedule, spread_bps, impact_eta, adv)
    n = length(schedule)
    slice_sigma = sigma / sqrt(Float64(n))
    prices = zeros(n)
    market_prices = zeros(n)
    S = S0
    for i in 1:n
        S *= exp(-0.5*slice_sigma^2 + slice_sigma * rndn())
        market_prices[i] = S
        vol_i = adv * vol_prof[i]
        slice_qty = schedule[i]
        pov = slice_qty / max(vol_i, 1.0)
        impact = impact_eta * sigma * sqrt(pov) * 10_000  # bps
        prices[i] = S * (1.0 + spread_bps/20_000.0 + impact/10_000.0)
    end
    exec_vwap = dot(prices, schedule) / max(sum(schedule), 1e-12)
    mkt_vwap  = dot(market_prices, adv .* vol_prof) / max(sum(adv .* vol_prof), 1e-12)
    return prices, market_prices, exec_vwap, mkt_vwap
end

# VWAP schedule: proportional to volume profile
vwap_schedule = target_qty .* vol_profile
# TWAP schedule: uniform
twap_schedule = fill(target_qty / n_minutes, n_minutes)

n_sims = 500
vwap_slippages = zeros(n_sims)
twap_slippages = zeros(n_sims)
for s in 1:n_sims
    _, _, ev, mv = simulate_execution(S0, sigma, vol_profile, target_qty, vwap_schedule, spread_bps, eta_impact, adv)
    vwap_slippages[s] = (ev - mv) / mv * 10_000
    _, _, et, mt = simulate_execution(S0, sigma, vol_profile, target_qty, twap_schedule, spread_bps, eta_impact, adv)
    twap_slippages[s] = (et - mt) / mt * 10_000
end

println("\nExecution benchmark comparison (500 simulations, buy 5% ADV):")
println("  Strategy | Avg Slippage (bps) | Std (bps) | 95th pct")
println("  " * "-"^52)
println("  VWAP     | $(lpad(round(mean(vwap_slippages),digits=2),17))  | $(lpad(round(std(vwap_slippages),digits=2),9)) | $(round(quantile(sort(vwap_slippages), 0.95), digits=2))")
println("  TWAP     | $(lpad(round(mean(twap_slippages),digits=2),17))  | $(lpad(round(std(twap_slippages),digits=2),9)) | $(round(quantile(sort(twap_slippages), 0.95), digits=2))")

# ── Section 3: Implementation Shortfall ──────────────────

println("\n--- Section 3: Implementation Shortfall Decomposition ---")

# Simulate IS for a set of buy orders
n_orders = 200
decision_prices = 100.0 .+ 5.0 .* [rndn() for _ in 1:n_orders]
order_sizes_pct = [0.01 + 0.09 * rnd() for _ in 1:n_orders]
execution_delays_min = [rnd() * 30 for _ in 1:n_orders]

is_total = zeros(n_orders)
is_delay = zeros(n_orders)
is_trading = zeros(n_orders)
is_perm = zeros(n_orders)

for i in 1:n_orders
    S = decision_prices[i]
    pov = order_sizes_pct[i]
    delay_mins = execution_delays_min[i]

    # Market drift during delay
    delay_return = sigma_daily / sqrt(390.0) * sqrt(delay_mins) * rndn()
    market_at_exec = S * exp(delay_return)

    # Trading cost (impact + spread)
    impact_bps = 0.314 * sigma_daily * sqrt(pov) * 10_000
    spread_c = spread_bps / 2.0
    exec_price = market_at_exec * (1.0 + (impact_bps + spread_c) / 10_000.0)

    # Permanent impact
    perm_bps = 0.5 * 0.314 * sigma_daily * pov * 10_000

    # IS components
    is_delay[i] = (market_at_exec - S) / S * 10_000
    is_trading[i] = (exec_price - market_at_exec) / market_at_exec * 10_000
    is_perm[i] = perm_bps
    is_total[i] = is_delay[i] + is_trading[i] + is_perm[i]
end

println("IS Decomposition (200 buy orders):")
println("  Component       | Mean (bps) | Std (bps) | % of Total")
println("  " * "-"^52)
total_abs = mean(abs.(is_total))
for (name, arr) in [("Delay cost", is_delay), ("Trading cost", is_trading),
                     ("Perm impact", is_perm), ("Total IS", is_total)]
    pct = mean(abs.(arr)) / total_abs * 100
    println("  $(lpad(name, 15)) | $(lpad(round(mean(arr),digits=2),10)) | $(lpad(round(std(arr),digits=2),9)) | $(round(pct,digits=1))%")
end

# ── Section 4: Almgren-Chriss Optimal Execution ──────────

println("\n--- Section 4: Almgren-Chriss Optimal Trajectory ---")

X0 = 1_000_000.0  # shares to liquidate
sigma_ac = 0.02
eta_ac = 0.0001; gamma_ac_val = 0.0001
T_horizon = 5.0   # 5 days
n_periods = 20    # 4 intervals/day

risk_aversions = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
println("AC efficient frontier (varying risk aversion):")
println("  Lambda     | Expected Cost (bps) | Risk (daily shares) | Strategy")
println("  " * "-"^62)

for lambda_ac in risk_aversions
    tau = T_horizon / n_periods
    temp_term = lambda_ac * sigma_ac^2 / max(eta_ac, 1e-12)
    kappa2 = temp_term * max(1.0 - tau * gamma_ac_val / (2*eta_ac), 0.01)
    kappa = sqrt(max(kappa2, 0.0))

    # Holdings trajectory
    holdings = zeros(n_periods + 1)
    for j in 0:n_periods
        t_j = j * tau
        if kappa * T_horizon > 1e-8
            holdings[j+1] = X0 * sinh(kappa * (T_horizon - t_j)) / sinh(kappa * T_horizon)
        else
            holdings[j+1] = X0 * (T_horizon - t_j) / T_horizon
        end
    end
    trades = -diff(holdings)

    # Costs
    temp_cost_total = sum(eta_ac * (t / tau)^2 * tau for t in trades) / X0 * 10_000
    perm_cost_total = sum(gamma_ac_val * abs(t) for t in trades) / X0 * 10_000
    total_cost = temp_cost_total + perm_cost_total

    # Risk
    risk = sigma_ac * sqrt(sum(holdings.^2) * tau) / X0 * 10_000

    # Strategy description
    first_trade_pct = trades[1] / X0 * 100
    strategy = first_trade_pct > 15 ? "Front-heavy" :
                first_trade_pct < 5  ? "TWAP-like" : "Balanced"

    println("  $(lpad(lambda_ac, 10)) | $(lpad(round(total_cost,digits=1),20)) | $(lpad(round(risk,digits=1),20)) | $strategy ($(round(first_trade_pct,digits=1))% first)")
end

# ── Section 5: Spread and Liquidity Analysis ─────────────

println("\n--- Section 5: Spread and Liquidity Analysis ---")

# Simulate order book data for 50 stocks
n_stocks = 50
stock_caps = sort(10_000.0 .* [rnd() + 0.5 for _ in 1:n_stocks], rev=true)  # market cap $5-15B
stock_adv = stock_caps .* 0.001 .* (0.8 .+ 0.4 .* [rnd() for _ in 1:n_stocks])  # ADV ≈ 0.1% of mktcap

# Spread tends to decrease with size and ADV
stock_spreads_bps = 5.0 ./ (stock_adv ./ 1e6).^0.4 .+ 2.0 .* [rnd() for _ in 1:n_stocks]
stock_vols = 0.015 .+ 0.030 .* [rnd() for _ in 1:n_stocks]

println("Liquidity profile by size bucket:")
println("  Bucket       | Stocks | Avg ADV (\$M) | Avg Spread | Avg Impact (5% ADV)")
println("  " * "-"^65)
buckets = [("Large (>8B)", stock_caps .> 8000.0),
           ("Mid (3-8B)",   (stock_caps .>= 3000.0) .& (stock_caps .<= 8000.0)),
           ("Small (<3B)", stock_caps .< 3000.0)]
for (label, mask) in buckets
    n_in = sum(mask)
    if n_in == 0; continue; end
    avg_adv_m = mean(stock_adv[mask]) / 1e6
    avg_spr = mean(stock_spreads_bps[mask])
    avg_vol = mean(stock_vols[mask])
    avg_impact = mean(0.314 .* stock_vols[mask] .* sqrt(0.05)) * 10_000
    println("  $(lpad(label, 12)) | $(lpad(n_in,6)) | $(lpad(round(avg_adv_m,digits=1),12)) | $(lpad(round(avg_spr,digits=1),10)) bps | $(round(avg_impact,digits=1)) bps")
end

# ── Section 6: Broker Performance ────────────────────────

println("\n--- Section 6: Broker Performance Scorecard ---")

# Simulate broker execution data
brokers = ["BrokerA", "BrokerB", "BrokerC", "DMA", "Algorithm"]
n_per_broker = 100

broker_data = Dict{String, Vector{Float64}}()
true_skill = [2.5, 5.0, 8.0, -1.0, 3.0]  # higher = better (lower costs)
for (b, skill) in zip(brokers, true_skill)
    costs = [skill + 3.0 * rndn() for _ in 1:n_per_broker]
    broker_data[b] = costs
end

println("Broker performance (cost in bps, buy orders):")
println("  Broker    | Mean Cost | Std  | Sharpe | t-stat | Grade")
println("  " * "-"^55)
for b in brokers
    costs = broker_data[b]
    mu = mean(costs)
    sig = std(costs)
    sharpe = sig > 0 ? mu / sig : 0.0  # higher = better performance vs variability
    tstat = mean(costs) / (std(costs) / sqrt(length(costs)))
    grade = mu < 5.0 ? "A" : mu < 7.0 ? "B" : mu < 9.0 ? "C" : "D"
    println("  $(lpad(b, 9)) | $(lpad(round(mu,digits=2),9)) | $(lpad(round(sig,digits=2),4)) | $(lpad(round(sharpe,digits=3),6)) | $(lpad(round(tstat,digits=2),6)) | $grade")
end

# ── Section 7: Liquidity-Adjusted Portfolio Construction ──

println("\n--- Section 7: Liquidity-Adjusted Optimization ---")

n_assets = 20
exp_returns = 0.10 .+ 0.05 .* [rndn() for _ in 1:n_assets]
vols = 0.15 .+ 0.10 .* [rnd() for _ in 1:n_assets]
adv_assets = [rnd() * 50e6 + 5e6 for _ in 1:n_assets]

# Correlation matrix
rho = 0.30
Sigma = Matrix{Float64}(I, n_assets, n_assets)
for i in 1:n_assets
    for j in 1:n_assets
        if i != j
            Sigma[i,j] = rho * vols[i] * vols[j]
        else
            Sigma[i,j] = vols[i]^2
        end
    end
end

target_aum = 500e6  # $500M portfolio

# Market impact cost per unit weight
impact_per_unit = 0.314 .* vols .* sqrt.(target_aum ./ adv_assets) .* 10_000  # bps

# Liquidity-adjusted return
lambda_liq = 0.5  # liquidity penalty
adj_returns = exp_returns .- lambda_liq .* impact_per_unit ./ 10_000

println("Liquidity adjustment effect (top 5 most liquid vs least liquid):")
println("  Asset | Gross Ret | Impact (bps) | Net Ret | ADV (\$M)")
println("  " * "-"^52)
# Sort by ADV
sorted_by_adv = sortperm(adv_assets, rev=true)
for rank in [1:3; n_assets-2:n_assets]
    i = sorted_by_adv[rank]
    println("  $(lpad(rank,5)) | $(lpad(round(exp_returns[i]*100,digits=2),9))% | " *
            "$(lpad(round(impact_per_unit[i],digits=1),12)) | $(lpad(round(adj_returns[i]*100,digits=2),7))% | " *
            "$(round(adv_assets[i]/1e6,digits=1))")
end

# Mean-variance with liquidity adjustment
lambda_mv = 2.0
w_no_liq  = max.((Sigma + 1e-6*I) \ exp_returns / lambda_mv, 0.0)
w_no_liq ./= sum(w_no_liq)
w_with_liq = max.((Sigma + 1e-6*I) \ adj_returns / lambda_mv, 0.0)
w_with_liq ./= sum(w_with_liq)

ret_no_liq   = dot(w_no_liq, exp_returns) * 100
ret_with_liq = dot(w_with_liq, exp_returns) * 100
impact_no_liq   = dot(w_no_liq, impact_per_unit)
impact_with_liq = dot(w_with_liq, impact_per_unit)

println("\nPortfolio comparison:")
println("  Without liquidity adj: Return=$(round(ret_no_liq,digits=2))%, Impact=$(round(impact_no_liq,digits=1)) bps")
println("  With liquidity adj:    Return=$(round(ret_with_liq,digits=2))%, Impact=$(round(impact_with_liq,digits=1)) bps")
println("  Net improvement: $(round(ret_with_liq - impact_with_liq/100 - (ret_no_liq - impact_no_liq/100), digits=2))% net return")

# ── Section 8: Capacity Analysis ─────────────────────────

println("\n--- Section 8: Strategy Capacity Analysis ---")

# Estimate capacity as a function of AUM
function estimate_capacity(alpha_bps, adv_vec, vol_vec, gamma_impact=0.314)
    # At capacity: impact_cost = alpha/2
    # gamma * sigma * sqrt(w * AUM / ADV) = alpha/2
    # => w * AUM / ADV = (alpha / (2 * gamma * sigma))^2
    n = length(adv_vec)
    capacities = zeros(n)
    for i in 1:n
        threshold_pov = (alpha_bps / 10_000 / (2 * gamma_impact * vol_vec[i]))^2
        capacities[i] = threshold_pov * adv_vec[i]
    end
    return capacities
end

alpha_signal = 8.0  # 8 bps signal alpha
capacities = estimate_capacity(alpha_signal, adv_assets, vols)

println("Strategy capacity by asset (alpha=$(alpha_signal) bps):")
println("  AUM Levels   | Pct of Capacity | Expected Net Alpha")
println("  " * "-"^45)
portfolio_capacity = sum(w_with_liq .* capacities)
aum_test_levels = [50e6, 100e6, 200e6, 500e6, 1000e6]
for aum in aum_test_levels
    pct_capacity = aum / portfolio_capacity * 100
    impact_at_aum = mean(0.314 .* vols .* sqrt.(aum .* w_with_liq ./ adv_assets)) * 10_000
    net_alpha = alpha_signal - impact_at_aum
    println("  \$$(lpad(round(aum/1e6,digits=0),6))M     | $(lpad(round(pct_capacity,digits=1),15))%  | $(round(net_alpha, digits=1)) bps")
end

println("\nEstimated strategy capacity: \$$(round(portfolio_capacity/1e6, digits=0))M")

println("\n✓ Notebook 30 complete")
