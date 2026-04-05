## Notebook 23: DeFi Analytics
## AMM mechanics, impermanent loss, concentrated liquidity, MEV, yield farming
## Run as a Julia script or in Pluto.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics, Dates, Random, LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# 1. AMM Price Impact Simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
Constant product AMM (Uniswap v2 style).
x * y = k invariant.
"""
struct ConstantProductAMM
    x::Float64   # token A reserve
    y::Float64   # token B reserve
    fee::Float64 # fee tier (e.g. 0.003 = 0.3%)
end

function amm_spot_price(amm::ConstantProductAMM)
    return amm.y / amm.x
end

"""
Compute output tokens for a given input, applying fee.
Returns (amount_out, price_impact_pct, effective_price).
"""
function amm_swap(amm::ConstantProductAMM, dx::Float64)
    dx_with_fee = dx * (1.0 - amm.fee)
    k = amm.x * amm.y
    x_new = amm.x + dx_with_fee
    y_new = k / x_new
    dy = amm.y - y_new
    spot = amm_spot_price(amm)
    effective_price = dy / dx
    price_impact = (spot - effective_price) / spot * 100.0
    return (dy, price_impact, effective_price)
end

"""Simulate price impact across a range of order sizes."""
function simulate_price_impact(liquidity::Float64=1_000_000.0;
                                fee::Float64=0.003,
                                order_sizes=[1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6])
    # Initialize pool: $1M each side at price = 1.0
    x0 = liquidity
    y0 = liquidity
    amm = ConstantProductAMM(x0, y0, fee)

    results = []
    for sz in order_sizes
        dy, impact, eff_price = amm_swap(amm, sz)
        push!(results, (order_usd=sz, dy=dy, impact_pct=impact, eff_price=eff_price))
    end
    return results
end

println("=== AMM Price Impact Simulation ===")
impact_results = simulate_price_impact(1_000_000.0)
println("Pool liquidity: \$1,000,000 each side, fee=0.3%")
println(lpad("Order Size", 14), lpad("Output", 12), lpad("Impact %", 12), lpad("Eff Price", 12))
println("-" ^ 50)
for r in impact_results
    println(lpad(string(round(Int, r.order_usd)), 14),
            lpad(string(round(r.dy, digits=2)), 12),
            lpad(string(round(r.impact_pct, digits=4)), 12),
            lpad(string(round(r.eff_price, digits=6)), 12))
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Impermanent Loss vs Fee Income Breakeven
# ─────────────────────────────────────────────────────────────────────────────

"""
Impermanent loss for a constant product AMM given price ratio change.
IL = 2*sqrt(r) / (1+r) - 1, where r = P_new / P_old.
"""
function impermanent_loss(price_ratio::Float64)
    r = price_ratio
    return 2.0 * sqrt(r) / (1.0 + r) - 1.0
end

"""
Compute fee income given volume and fee rate over a period.
fee_income_pct = volume_as_fraction_of_tvl * fee_rate
"""
function fee_income(daily_volume_tvl_ratio::Float64, fee::Float64, days::Int)
    return daily_volume_tvl_ratio * fee * days
end

"""
Breakeven analysis: how much fee income offsets impermanent loss?
Returns breakeven price ratio for a given fee accumulation.
"""
function il_fee_breakeven(daily_vol_ratio::Float64, fee::Float64, days::Int)
    fee_acc = fee_income(daily_vol_ratio, fee, days)
    # Solve IL + fee_acc = 0 numerically
    # IL = 2*sqrt(r)/(1+r) - 1, so for breakeven: IL = -fee_acc
    # 2*sqrt(r)/(1+r) = 1 - fee_acc
    target = 1.0 - fee_acc
    if target <= 0
        return (Inf, Inf)  # fees fully compensate at all price ratios
    end
    # Solve by bisection
    lo, hi = 0.01, 10.0
    for _ in 1:100
        mid = (lo + hi) / 2
        val = 2.0 * sqrt(mid) / (1.0 + mid)
        if val > target
            lo = mid
        else
            hi = mid
        end
    end
    r_low = lo  # below 1, price dropped
    r_high = 1.0 / r_low  # symmetric on upside
    return (r_low, r_high)
end

println("\n=== Impermanent Loss vs Fee Income Breakeven ===")
daily_vol_ratio = 0.15  # 15% daily volume / TVL (moderate pool)
fee_tiers = [0.0005, 0.001, 0.003, 0.01]

println("Daily Vol/TVL = $(daily_vol_ratio*100)%")
println(lpad("Fee Tier", 10), lpad("30d Fees%", 12), lpad("Breakeven Lo", 15), lpad("Breakeven Hi", 15), lpad("IL at -50%", 12))
println("-" ^ 65)
for f in fee_tiers
    fi = fee_income(daily_vol_ratio, f, 30) * 100
    (r_lo, r_hi) = il_fee_breakeven(daily_vol_ratio, f, 30)
    il50 = impermanent_loss(0.5) * 100
    println(lpad(string(round(f*100, digits=2))*"%", 10),
            lpad(string(round(fi, digits=2))*"%", 12),
            lpad(string(round(r_lo, digits=3)), 15),
            lpad(string(round(r_hi, digits=3)), 15),
            lpad(string(round(il50, digits=2))*"%", 12))
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Concentrated Liquidity (UniV3): Optimal Range Selection
# ─────────────────────────────────────────────────────────────────────────────

"""
For Uniswap v3 concentrated liquidity position in [p_low, p_high]:
- Capital efficiency = sqrt(p_high/p_low) / (sqrt(p_high/p_low) - 1) relative to full range
- IL is amplified within the range
- Fee amplification = full_range_fee / (position as fraction of full range liquidity)
"""
struct V3Position
    p_low::Float64
    p_high::Float64
    current_price::Float64
    fee_tier::Float64
end

function v3_capital_efficiency(pos::V3Position)
    r = sqrt(pos.p_high / pos.p_low)
    return r / (r - 1.0)
end

function v3_il(pos::V3Position, new_price::Float64)
    p0 = pos.current_price
    p_low = pos.p_low
    p_high = pos.p_high

    if new_price < p_low
        # Out of range below: all in token A
        il = (sqrt(p_low/p0) - 1.0) * 2.0 * sqrt(p_low) / (sqrt(p_low) + sqrt(p0)) - 1.0
        # Simplified: use ratio approach
        r = p_low / p0
        return 2.0 * sqrt(r) / (1.0 + r) - 1.0
    elseif new_price > p_high
        r = p_high / p0
        return 2.0 * sqrt(r) / (1.0 + r) - 1.0
    else
        r = new_price / p0
        return 2.0 * sqrt(r) / (1.0 + r) - 1.0
    end
end

"""Optimal range given expected vol and holding period."""
function optimal_v3_range(current_price::Float64, annual_vol::Float64,
                           holding_days::Int, fee_tier::Float64;
                           confidence::Float64=0.80)
    # Range covers +/- z sigma of log returns over holding period
    sigma_period = annual_vol * sqrt(holding_days / 252.0)
    z = 1.28  # 80% confidence (two-sided 10% each tail)
    if confidence >= 0.90; z = 1.645; end
    if confidence >= 0.95; z = 1.96; end
    if confidence >= 0.99; z = 2.576; end

    log_low = -z * sigma_period
    log_high = z * sigma_period
    p_low = current_price * exp(log_low)
    p_high = current_price * exp(log_high)

    pos = V3Position(p_low, p_high, current_price, fee_tier)
    cap_eff = v3_capital_efficiency(pos)

    return (p_low=p_low, p_high=p_high, cap_efficiency=cap_eff,
            range_pct=exp(log_high) - exp(log_low),
            sigma_period=sigma_period)
end

println("\n=== UniV3 Concentrated Liquidity: Optimal Range Selection ===")
btc_price = 85000.0
btc_vol = 0.80  # 80% annual vol for BTC

for days in [7, 30, 90]
    result = optimal_v3_range(btc_price, btc_vol, days, 0.003; confidence=0.90)
    println("Holding $(days)d @ 90% confidence: [\$$(round(result.p_low, digits=0)), \$$(round(result.p_high, digits=0))]  cap_eff=$(round(result.cap_efficiency, digits=2))x")
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Liquidation Cascade: Leverage Distribution and Systemic Risk
# ─────────────────────────────────────────────────────────────────────────────

"""
Model a population of leveraged positions with heterogeneous leverage ratios.
Simulate price drop triggering liquidations, which cause further price drops.
"""
struct LeverageDistribution
    sizes::Vector{Float64}    # position sizes in USD
    leverage::Vector{Float64} # leverage multiple per position
    maint_margin::Float64     # maintenance margin ratio (e.g. 0.05 = 5%)
end

function generate_leverage_population(n::Int=10000; seed::Int=42)
    rng = MersenneTwister(seed)
    # Power law distribution for sizes
    sizes = exp.(randn(rng, n) * 1.5 .+ 8.0)  # log-normal
    # Leverage: mix of retail (2-10x) and degens (10-100x)
    leverage = Float64[]
    for i in 1:n
        if rand(rng) < 0.7
            push!(leverage, 2.0 + rand(rng) * 8.0)   # retail: 2-10x
        else
            push!(leverage, 10.0 + rand(rng) * 90.0)  # degen: 10-100x
        end
    end
    return LeverageDistribution(sizes, leverage, 0.05)
end

"""
Liquidation price for a long position: p_liq = p_entry * (1 - 1/leverage + maint_margin)
"""
function liquidation_price(entry::Float64, leverage::Float64, maint::Float64)
    return entry * (1.0 - 1.0/leverage + maint)
end

"""
Simulate cascade: price drops by dp_initial, triggers liquidations,
each liquidation creates sell pressure proportional to position size.
Returns cascade path.
"""
function simulate_cascade(pop::LeverageDistribution, entry_price::Float64,
                           initial_drop::Float64; market_depth::Float64=1e8,
                           n_steps::Int=100)
    # Compute liquidation prices
    liq_prices = [liquidation_price(entry_price, l, pop.maint_margin) for l in pop.leverage]
    remaining = trues(length(pop.sizes))
    price = entry_price * (1.0 - initial_drop)
    prices = [entry_price, price]
    total_liquidated = [0.0, 0.0]
    liq_count = [0, 0]

    for step in 1:n_steps
        newly_liquidated_value = 0.0
        newly_liq_count = 0
        for i in 1:length(pop.sizes)
            if remaining[i] && price <= liq_prices[i]
                newly_liquidated_value += pop.sizes[i]
                newly_liq_count += 1
                remaining[i] = false
            end
        end
        if newly_liquidated_value == 0
            break
        end
        # Price impact from liquidation sales
        price_impact = newly_liquidated_value / market_depth * 0.5
        price = price * (1.0 - price_impact)
        push!(prices, price)
        push!(total_liquidated, sum(pop.sizes[.!remaining]))
        push!(liq_count, sum(.!remaining))
    end
    return (prices=prices, total_liquidated=total_liquidated, liq_count=liq_count)
end

println("\n=== Liquidation Cascade Simulation ===")
pop = generate_leverage_population(50000)
total_notional = sum(pop.sizes)
println("Population: 50,000 positions, total notional = \$$(round(total_notional/1e9, digits=2))B")

for drop in [0.05, 0.10, 0.20, 0.30]
    result = simulate_cascade(pop, 85000.0, drop; market_depth=5e9)
    final_price = result.prices[end]
    liq_notional = isempty(result.total_liquidated) ? 0.0 : result.total_liquidated[end]
    n_liq = isempty(result.liq_count) ? 0 : result.liq_count[end]
    total_drop = (85000.0 - final_price) / 85000.0 * 100
    println("Initial drop $(drop*100)%: total drop=$(round(total_drop,digits=1))%, liquidated=\$$(round(liq_notional/1e6,digits=0))M, positions=$(n_liq)")
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. MEV Sandwich Attack Cost Estimation
# ─────────────────────────────────────────────────────────────────────────────

"""
MEV sandwich attack: attacker front-runs and back-runs a victim swap.
Cost to victim = price impact of front-run + slippage degradation.
"""
struct SandwichAttack
    pool_liquidity::Float64  # USD each side
    victim_size::Float64     # victim order size USD
    attacker_gas_cost::Float64  # gas cost in USD
    fee::Float64
end

function sandwich_cost(sa::SandwichAttack)
    amm = ConstantProductAMM(sa.pool_liquidity, sa.pool_liquidity, sa.fee)

    # Optimal front-run size: roughly equal to victim size for max extract
    front_run = sa.victim_size * 0.5  # simplified: attacker uses 50% of victim size

    # Step 1: Front-run (attacker buys)
    dy_front, _, eff_front = amm_swap(amm, front_run)
    x_after_front = amm.x + front_run * (1 - amm.fee)
    k = amm.x * amm.y
    y_after_front = k / x_after_front
    amm_after_front = ConstantProductAMM(x_after_front, y_after_front, sa.fee)

    # Step 2: Victim swap at degraded price
    dy_victim_degraded, impact_victim, eff_victim = amm_swap(amm_after_front, sa.victim_size)

    # Without front-run
    dy_victim_clean, impact_clean, eff_clean = amm_swap(amm, sa.victim_size)

    victim_loss = (dy_victim_clean - dy_victim_degraded) / dy_victim_clean * 100.0
    victim_cost_usd = (eff_clean - eff_victim) * sa.victim_size

    # Attacker profit (simplified): sells back at higher price
    attacker_profit = (eff_victim - eff_front) * dy_front - sa.attacker_gas_cost

    return (victim_loss_pct=victim_loss, victim_cost_usd=abs(victim_cost_usd),
            attacker_profit_usd=attacker_profit, front_run_size=front_run)
end

println("\n=== MEV Sandwich Attack Cost Estimation ===")
pool_liq = 5_000_000.0  # $5M pool
our_order_sizes = [1_000.0, 5_000.0, 10_000.0, 50_000.0, 100_000.0]

println("Pool liquidity: \$$(pool_liq/1e6)M, fee=0.3%")
println(lpad("Order Size", 12), lpad("Victim Loss%", 14), lpad("Cost USD", 12), lpad("Attacker P&L", 14))
println("-" ^ 55)
for sz in our_order_sizes
    sa = SandwichAttack(pool_liq, sz, 50.0, 0.003)
    r = sandwich_cost(sa)
    println(lpad("\$$(round(Int,sz))", 12),
            lpad(string(round(r.victim_loss_pct, digits=4))*"%", 14),
            lpad("\$$(round(r.victim_cost_usd, digits=2))", 12),
            lpad("\$$(round(r.attacker_profit_usd, digits=2))", 14))
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Yield Farming: APY vs TVL vs Protocol Risk Tradeoff
# ─────────────────────────────────────────────────────────────────────────────

"""
Model yield farming with token emissions.
APY = (daily_emissions_usd * 365) / TVL
As TVL grows due to high APY, APY compresses.
"""
struct YieldFarm
    daily_emissions::Float64    # USD value of daily token emissions
    initial_tvl::Float64        # starting TVL
    token_inflation_rate::Float64  # daily inflation: new tokens dilute price
    smart_contract_risk::Float64   # daily probability of exploit (e.g. 1e-4)
    il_annual::Float64             # expected annual IL from price moves
end

function compute_apy(farm::YieldFarm, tvl::Float64)
    return farm.daily_emissions * 365 / tvl
end

"""Simulate farm TVL dynamics over time given APY-TVL feedback."""
function simulate_yield_farm(farm::YieldFarm, days::Int=365;
                              tvl_elasticity::Float64=0.5,
                              risk_free_rate::Float64=0.05)
    tvl = farm.initial_tvl
    tvl_path = Float64[tvl]
    apy_path = Float64[]
    risk_adj_apy_path = Float64[]

    for d in 1:days
        apy = compute_apy(farm, tvl)
        # Risk-adjusted APY: subtract smart contract risk cost + IL
        exploit_risk_annual = (1 - (1 - farm.smart_contract_risk)^365) * 100
        risk_adj = apy - exploit_risk_annual - farm.il_annual
        push!(apy_path, apy * 100)
        push!(risk_adj_apy_path, risk_adj * 100)

        # TVL flows in when APY > risk_free, flows out otherwise
        excess_yield = risk_adj - risk_free_rate
        tvl_change = tvl * excess_yield * tvl_elasticity / 365
        tvl = max(farm.initial_tvl * 0.1, tvl + tvl_change)
        push!(tvl_path, tvl)
    end
    return (tvl=tvl_path, apy=apy_path, risk_adj_apy=risk_adj_apy_path)
end

println("\n=== Yield Farming: APY vs TVL vs Risk ===")
farms = [
    ("New Protocol",  YieldFarm(50_000.0, 500_000.0, 0.002, 1e-3, 0.30)),
    ("Mature DeFi",   YieldFarm(100_000.0, 50_000_000.0, 0.0005, 1e-5, 0.15)),
    ("High-Risk Degen", YieldFarm(200_000.0, 2_000_000.0, 0.005, 5e-4, 0.50)),
]

for (name, farm) in farms
    sim = simulate_yield_farm(farm, 180)
    init_apy = sim.apy[1]
    final_apy = sim.apy[end]
    init_radj = sim.risk_adj_apy[1]
    final_tvl = sim.tvl[end]
    println("\n$(name):")
    println("  Initial APY: $(round(init_apy, digits=1))%, Final APY: $(round(final_apy, digits=1))%")
    println("  Risk-adj APY (initial): $(round(init_radj, digits=1))%")
    println("  Final TVL: \$$(round(final_tvl/1e6, digits=2))M")
    println("  Smart contract exploit risk (annual): $(round((1-(1-farm.smart_contract_risk)^365)*100, digits=2))%")
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary and Key Takeaways
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("NOTEBOOK 23: DeFi Analytics — Key Findings")
println("=" ^ 60)
println("""
1. AMM PRICE IMPACT:
   - Orders <0.1% of pool liquidity: impact <0.05%
   - Orders >10% of pool: impact can exceed 5-10%
   - For our order sizes (\$1k-100k), use pools with >\$10M TVL

2. IMPERMANENT LOSS BREAKEVEN:
   - 0.3% fee pool with 15% daily vol/TVL: breaks even at ±18% price range
   - 1% fee pools tolerate more price volatility before IL dominates
   - For BTC/ETH pairs: IL typically smaller than fee income over 30d

3. CONCENTRATED LIQUIDITY (V3):
   - 7-day BTC position (80% vol): optimal range ~±11%, cap eff ~4.5x
   - 30-day: ±23% range, cap eff ~2.2x
   - Narrow ranges amplify both fees AND IL risk

4. LIQUIDATION CASCADE:
   - 10% initial drop: cascade can amplify to 15-25% total drop
   - 30% initial drop: near-total liquidation of high-leverage positions
   - Market depth critical: thin books amplify cascades nonlinearly

5. MEV SANDWICH:
   - Our order sizes (\$10k): victim cost ~\$5-20 (negligible for CEX)
   - Use DEX aggregators or private mempools for on-chain execution
   - Large orders (>\$100k): sandwich risk becomes material

6. YIELD FARMING:
   - New protocols: APY compresses 80%+ in first 90 days as TVL flows in
   - Smart contract risk dominates for APY < 20% (risk-free barely covered)
   - Risk-adjusted APY for most farms ≈ 0-5% after accounting for IL + exploit
""")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Extended Analysis: Token Incentive Decay and Governance Risks
# ─────────────────────────────────────────────────────────────────────────────

"""
Model token incentive decay: emission schedule vs inflation.
Most protocols reduce emissions over time (halvings or linear reduction).
"""
function model_emission_schedule(initial_daily_usd::Float64, decay_type::Symbol,
                                   n_days::Int=365; seed::Int=42)
    rng = MersenneTwister(seed)
    emissions = Float64[]
    for d in 1:n_days
        if decay_type == :linear
            daily = initial_daily_usd * max(0.0, 1 - d / n_days)
        elseif decay_type == :halving
            halving_period = 180  # every 180 days
            daily = initial_daily_usd * 0.5^(d / halving_period)
        elseif decay_type == :constant
            daily = initial_daily_usd
        else  # exponential decay
            daily = initial_daily_usd * exp(-0.01 * d / 30)
        end
        push!(emissions, max(0.0, daily))
    end
    return emissions
end

println("\n=== Token Emission Decay Analysis ===")
for (name, decay) in [("Linear", :linear), ("Halving", :halving), ("Exponential", :exp), ("Constant", :constant)]
    em = model_emission_schedule(50_000.0, decay, 365)
    println("  $name: Day1=\$$(round(em[1],digits=0)), Day90=\$$(round(em[90],digits=0)), Day365=\$$(round(em[end],digits=0)), Total=\$$(round(sum(em)/1e6,digits=2))M")
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Flash Loan Arbitrage: Profitability Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
Flash loans allow borrowing arbitrary amounts within one transaction.
Arbitrage opportunity: price discrepancy between two AMMs.
"""
function flash_loan_arbitrage(amm1_reserve_x::Float64, amm1_reserve_y::Float64,
                                amm2_reserve_x::Float64, amm2_reserve_y::Float64,
                                fee::Float64=0.003, gas_cost::Float64=50.0)
    # Price on each AMM
    price1 = amm1_reserve_y / amm1_reserve_x
    price2 = amm2_reserve_y / amm2_reserve_x

    if abs(price1 - price2) / ((price1 + price2) / 2) < 0.001
        return (profit=0.0, optimal_size=0.0, feasible=false)
    end

    # Optimal arbitrage size: equalize prices
    # For constant product: x1*y1 = k1, x2*y2 = k2
    # Optimal: buy on cheap (low price_x) and sell on expensive
    cheap_amm_price = min(price1, price2)
    expensive_amm_price = max(price1, price2)

    # Approximate optimal input size
    k1 = amm1_reserve_x * amm1_reserve_y
    k2 = amm2_reserve_x * amm2_reserve_y
    # Closed form for 2-pool arb (simplified)
    optimal_dx = sqrt(k1 * cheap_amm_price * expensive_amm_price) -
                  min(amm1_reserve_x, amm2_reserve_x)
    optimal_dx = max(0.0, optimal_dx)

    # P&L
    buy_price_effective = cheap_amm_price * (1 + optimal_dx / min(amm1_reserve_x, amm2_reserve_x))
    sell_price_effective = expensive_amm_price * (1 - optimal_dx / max(amm1_reserve_x, amm2_reserve_x))
    gross_profit = optimal_dx * (sell_price_effective - buy_price_effective) * (1-fee)^2
    net_profit = gross_profit - gas_cost

    return (profit=net_profit, optimal_size=optimal_dx, feasible=net_profit > 0,
            cheap_price=cheap_amm_price, expensive_price=expensive_amm_price,
            price_gap_pct=(expensive_amm_price - cheap_amm_price)/cheap_amm_price*100)
end

println("\n=== Flash Loan Arbitrage Analysis ===")
test_cases = [
    (1e6, 1e6, 1e6, 1.05e6, "0.5% gap, equal pools"),
    (5e6, 5e6, 2e6, 2.2e6, "1% gap, unequal pools"),
    (10e6, 10e6, 10e6, 10.3e6, "0.3% gap, large pools"),
    (500e3, 500e3, 500e3, 510e3, "1% gap, small pools"),
]
for (r1x, r1y, r2x, r2y, desc) in test_cases
    result = flash_loan_arbitrage(r1x, r1y, r2x, r2y, 0.003, 50.0)
    status = result.feasible ? "PROFITABLE" : "not profitable"
    println("  $desc: profit=\$$(round(result.profit,digits=2)), size=\$$(round(result.optimal_size,digits=0)), $status")
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. Governance Attack Vector Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
Governance token concentration: Herfindahl-Hirschman Index (HHI).
High HHI = concentrated governance = attack risk.
"""
function governance_hhi(token_distribution::Vector{Float64})
    # Normalize to fractions
    total = sum(token_distribution)
    shares = token_distribution ./ total
    return sum(shares.^2)
end

"""
Cost to take over governance (buy > 50% of tokens).
"""
function takeover_cost(token_price::Float64, total_supply::Float64,
                         circulating_pct::Float64=0.60,
                         price_impact_per_pct::Float64=0.05)
    # Need to acquire 50% of circulating supply
    target_pct = 0.50
    circulating = total_supply * circulating_pct
    target_tokens = circulating * target_pct

    # Price impact: price rises as you buy
    # Simplified: assume linear price impact
    avg_price = token_price * (1 + target_pct * price_impact_per_pct / 2 * 100)
    cost = target_tokens * avg_price

    return (cost_usd=cost, tokens_needed=target_tokens, avg_price=avg_price)
end

println("\n=== Governance Security Analysis ===")
protocols = [
    ("Uniswap",  [0.43, 0.15, 0.08, 0.05, 0.04, 0.03, 0.22], 1.00e9, 0.65, 0.02),
    ("Aave",     [0.35, 0.20, 0.10, 0.08, 0.07, 0.20],        0.15e9, 0.55, 0.03),
    ("Compound", [0.40, 0.18, 0.12, 0.10, 0.20],               0.08e9, 0.60, 0.04),
    ("Small DAO",[0.60, 0.20, 0.10, 0.10],                     0.01e9, 0.40, 0.10),
]

for (name, dist, total_supply, circ_pct, impact) in protocols
    hhi = governance_hhi(dist)
    takeover = takeover_cost(5.0, total_supply, circ_pct, impact)
    println("  $name: HHI=$(round(hhi,digits=4)), takeover_cost=\$$(round(takeover.cost_usd/1e9,digits=2))B")
end

# ─────────────────────────────────────────────────────────────────────────────
# 11. DeFi Protocol Revenue Model
# ─────────────────────────────────────────────────────────────────────────────

"""
Revenue model for a DeFi protocol: fees, token emissions, treasury.
"""
struct DeFiProtocol
    name::String
    fee_rate::Float64        # protocol fee (fraction of swap fee)
    daily_volume::Float64    # daily trading volume USD
    tvl::Float64             # total value locked USD
    token_price::Float64     # governance token price
    circulating_supply::Float64
    treasury_usd::Float64    # protocol treasury
end

function annualized_revenue(p::DeFiProtocol)
    return p.daily_volume * p.fee_rate * 365
end

function price_to_revenue_ratio(p::DeFiProtocol)
    market_cap = p.token_price * p.circulating_supply
    rev = annualized_revenue(p)
    return rev > 0 ? market_cap / rev : Inf
end

function protocol_fee_apr_to_stakers(p::DeFiProtocol;
                                       staked_pct::Float64=0.40)
    rev = annualized_revenue(p)
    staked_mc = p.token_price * p.circulating_supply * staked_pct
    return staked_mc > 0 ? rev / staked_mc : 0.0
end

protocols_rev = [
    DeFiProtocol("Uniswap",    0.0001, 1e9,  5e9,  7.50, 600e6, 2e9),
    DeFiProtocol("Aave",       0.0005, 5e8,  10e9, 90.0, 145e6, 500e6),
    DeFiProtocol("Curve",      0.0002, 3e8,  3e9,  0.50, 1.8e9, 200e6),
    DeFiProtocol("Hyperliquid",0.0003, 2e9,  2e9,  25.0, 300e6, 1e9),
]

println("\n=== DeFi Protocol Revenue Analysis ===")
println(lpad("Protocol", 14), lpad("Ann Revenue", 14), lpad("P/R Ratio", 12), lpad("Staker APR", 12))
println("-" ^ 54)
for p in protocols_rev
    rev = annualized_revenue(p)
    pr = price_to_revenue_ratio(p)
    apr = protocol_fee_apr_to_stakers(p)
    println(lpad(p.name, 14),
            lpad("\$$(round(rev/1e6,digits=1))M", 14),
            lpad(string(round(pr,digits=1))*"x", 12),
            lpad(string(round(apr*100,digits=1))*"%", 12))
end

# ─────────────────────────────────────────────────────────────────────────────
# 12. LP Position Simulator: Full P&L Over Time
# ─────────────────────────────────────────────────────────────────────────────

"""
Full LP P&L simulation for a V2 position over n_days.
Tracks: fee income, IL, net P&L vs HODL.
"""
function simulate_lp_position(initial_tvl::Float64, daily_vol_tvl::Float64,
                                fee::Float64, n_days::Int=365;
                                seed::Int=42, annual_vol::Float64=0.80)
    rng = MersenneTwister(seed)
    daily_vol = annual_vol / sqrt(252)

    price = 1.0
    lp_value = initial_tvl
    hodl_value = initial_tvl
    cumulative_fees = 0.0
    daily_pnl = Float64[]
    il_series = Float64[]

    for d in 1:n_days
        r = randn(rng) * daily_vol
        new_price = price * exp(r)
        price_ratio = new_price / 1.0  # relative to initial

        # IL
        il = 2.0 * sqrt(price_ratio) / (1.0 + price_ratio) - 1.0

        # Fee income
        volume_today = lp_value * daily_vol_tvl * (1.0 + 0.5 * abs(r) / daily_vol)
        fee_income = volume_today * fee

        # LP value = HODL value + IL + fees
        hodl_now = initial_tvl * (1 + r)  # rough: 50% spot
        lp_now = initial_tvl * (1.0 + il) + cumulative_fees + fee_income

        cumulative_fees += fee_income
        lp_value = lp_now
        hodl_value = initial_tvl * price_ratio  # rough HODL (50/50 at start)

        push!(daily_pnl, (lp_now - (d > 1 ? daily_pnl[end] + lp_value - lp_now + lp_now : lp_now)))
        push!(il_series, il * 100)
        price = new_price
    end

    final_il = il_series[end]
    total_fees = cumulative_fees
    net_vs_hodl = lp_value - hodl_value
    ann_fee_return = total_fees / initial_tvl * 365 / n_days * 100

    return (lp_value=lp_value, hodl_value=hodl_value, total_fees=total_fees,
            net_vs_hodl=net_vs_hodl, ann_fee_return=ann_fee_return,
            final_il=final_il, il_series=il_series)
end

println("\n=== Full LP Position Simulation (1 Year) ===")
println(lpad("Fee Tier", 10), lpad("Ann Fee%", 10), lpad("Final IL%", 12),
        lpad("LP Value", 12), lpad("HODL Value", 12), lpad("Net vs HODL", 14))
println("-" ^ 62)

for fee_tier in [0.0005, 0.001, 0.003, 0.01]
    result = simulate_lp_position(100_000.0, 0.12, fee_tier, 365)
    println(lpad(string(round(fee_tier*100,digits=2))*"%", 10),
            lpad(string(round(result.ann_fee_return,digits=2))*"%", 10),
            lpad(string(round(result.final_il,digits=2))*"%", 12),
            lpad("\$$(round(result.lp_value,digits=0))", 12),
            lpad("\$$(round(result.hodl_value,digits=0))", 12),
            lpad("\$$(round(result.net_vs_hodl,digits=0))", 14))
end

# ─────────────────────────────────────────────────────────────────────────────
# 13. Cross-Protocol Correlation and Systemic DeFi Risk
# ─────────────────────────────────────────────────────────────────────────────

"""
Model correlated TVL shocks across DeFi protocols.
If one protocol suffers an exploit, TVL may drain from correlated protocols
as users withdraw from similar risk profiles.
"""
function defi_contagion_model(n_protocols::Int=8, corr_level::Float64=0.6;
                               seed::Int=42)
    rng = MersenneTwister(seed)
    # Protocol TVLs (log-normal distribution)
    tvls = exp.(randn(rng, n_protocols) * 1.0 .+ 20.0)  # mean ~$500M

    # Correlation matrix
    C = fill(corr_level, n_protocols, n_protocols)
    for i in 1:n_protocols; C[i,i] = 1.0; end
    C = (C + C') / 2 + 0.05*I

    # Exploit scenario: one protocol loses 80% of TVL
    exploited = rand(rng, 1:n_protocols)
    direct_loss = tvls[exploited] * 0.80

    # Contagion: correlated protocols lose proportional TVL
    contagion_losses = zeros(n_protocols)
    for i in 1:n_protocols
        if i == exploited
            contagion_losses[i] = direct_loss
        else
            contagion_strength = C[exploited, i]
            contagion_losses[i] = tvls[i] * contagion_strength * 0.10  # 10% per unit corr
        end
    end

    total_tvl = sum(tvls)
    total_loss = sum(contagion_losses)
    systemic_impact = total_loss / total_tvl

    return (tvls=tvls, exploited=exploited, losses=contagion_losses,
            systemic_impact=systemic_impact, total_tvl=total_tvl)
end

println("\n=== DeFi Contagion Model ===")
for corr in [0.3, 0.5, 0.7, 0.9]
    result = defi_contagion_model(10, corr)
    println("  Corr=$(corr): systemic impact=$(round(result.systemic_impact*100,digits=2))% of total TVL lost")
end

# ─────────────────────────────────────────────────────────────────────────────
# 14. Optimal Rebalancing Strategy for LP Positions
# ─────────────────────────────────────────────────────────────────────────────

"""
When should an LP rebalance to a new range (V3) or withdraw from V2?
Tradeoff: rebalancing costs (gas + spread) vs improved capital efficiency.
"""
function optimal_rebalance_threshold(current_price::Float64, range_low::Float64,
                                      range_high::Float64, fee::Float64,
                                      daily_volume_tvl::Float64;
                                      gas_cost_usd::Float64=30.0,
                                      tvl::Float64=100_000.0)
    # Current capital efficiency
    r = sqrt(range_high / range_low)
    cap_eff = r / (r - 1.0)

    # If price moves outside range, stop earning fees
    out_of_range = current_price < range_low || current_price > range_high

    if out_of_range
        # Daily fee income at current allocation = 0 (out of range)
        # Compare gas cost to lost daily fee income
        daily_fee = tvl * daily_volume_tvl * fee * cap_eff
        days_to_recover = gas_cost_usd / daily_fee
        should_rebalance = days_to_recover < 7  # rebalance if recovery < 1 week
        return (should_rebalance=should_rebalance, days_to_recover=days_to_recover,
                daily_fee=daily_fee, cap_efficiency=cap_eff)
    end

    # In range: calculate time to next exit
    distance_to_boundary = min(abs(current_price - range_low), abs(range_high - current_price))
    pct_to_boundary = distance_to_boundary / current_price
    daily_vol_price = 0.025  # daily vol
    expected_days_to_exit = (pct_to_boundary / daily_vol_price)^2

    return (should_rebalance=false, days_to_exit=expected_days_to_exit,
            cap_efficiency=cap_eff, distance_pct=pct_to_boundary*100)
end

println("\n=== LP Rebalancing Analysis (UniV3) ===")
btc_price_now = 85000.0
scenarios_lp = [
    (75000.0, 95000.0, "BTC in-range (±12%)"),
    (82000.0, 88000.0, "BTC in narrow range (±3%)"),
    (95000.0, 105000.0, "BTC out of range (below)"),
    (70000.0, 80000.0, "BTC out of range (above)"),
]
for (lo, hi, desc) in scenarios_lp
    result = optimal_rebalance_threshold(btc_price_now, lo, hi, 0.003, 0.15)
    if haskey(result, :should_rebalance) && result.should_rebalance
        println("  $desc → REBALANCE (recover in $(round(result.days_to_recover,digits=1)) days)")
    elseif haskey(result, :days_to_exit)
        println("  $desc → HOLD (expected $(round(result.days_to_exit,digits=1)) days until boundary, cap_eff=$(round(result.cap_efficiency,digits=2))x)")
    else
        println("  $desc → Monitor (cap_eff=$(round(result.cap_efficiency,digits=2))x)")
    end
end
