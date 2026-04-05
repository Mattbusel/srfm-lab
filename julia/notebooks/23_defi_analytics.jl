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
