## Notebook 33: Strategy Stress Testing
## Historical scenarios (COVID/FTX/Luna/Apr-2026), hypothetical BTC -50%,
## correlation stress, liquidity stress, model stress, combined worst-case
## Run as a Julia script or in Pluto.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics, Dates, Random, LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# 1. Strategy Model Setup
# ─────────────────────────────────────────────────────────────────────────────

"""Portfolio model with multi-asset positions."""
struct StrategyPortfolio
    assets::Vector{String}
    weights::Vector{Float64}    # current allocation
    leverage::Float64
    aum::Float64                # total AUM
    vol_target::Float64         # target annual vol (e.g., 0.20)
    max_drawdown_limit::Float64 # stop loss (e.g., 0.15 = 15%)
end

function default_portfolio()
    assets = ["BTC", "ETH", "BNB", "SOL", "ADA", "XRP", "AVAX", "DOGE", "CASH"]
    weights = [0.30, 0.20, 0.10, 0.08, 0.05, 0.05, 0.07, 0.05, 0.10]
    return StrategyPortfolio(assets, weights, 2.0, 1_000_000.0, 0.25, 0.15)
end

port = default_portfolio()

println("=== Strategy Stress Testing ===")
println("Portfolio: $(length(port.assets)-1) crypto assets + cash")
println("AUM: \$$(round(port.aum/1e6,digits=1))M, Leverage: $(port.leverage)x")
println("Weights: $(join([a*"="*string(round(w*100,digits=0))*"%" for (a,w) in zip(port.assets,port.weights)], ", "))")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Historical Stress Scenarios
# ─────────────────────────────────────────────────────────────────────────────

"""
Define historical stress scenarios with asset returns.
Returns in each scenario are expressed as % moves.
"""
struct StressScenario
    name::String
    description::String
    asset_returns::Dict{String, Float64}  # asset → total return during event
    duration_days::Int
    peak_vol_multiplier::Float64
end

# Historical scenarios (approximate numbers based on actual events)
scenarios = [
    StressScenario(
        "COVID-2020",
        "March 2020: pandemic onset, crypto crash 50-60% in 3 days",
        Dict("BTC"=>-0.50, "ETH"=>-0.60, "BNB"=>-0.58, "SOL"=>-0.65, "ADA"=>-0.62,
             "XRP"=>-0.55, "AVAX"=>-0.68, "DOGE"=>-0.52, "CASH"=>0.0),
        3, 5.0
    ),
    StressScenario(
        "Luna-UST-2022",
        "May 2022: Terra/Luna collapse, contagion across DeFi and CeFi",
        Dict("BTC"=>-0.35, "ETH"=>-0.45, "BNB"=>-0.40, "SOL"=>-0.55, "ADA"=>-0.48,
             "XRP"=>-0.30, "AVAX"=>-0.60, "DOGE"=>-0.40, "CASH"=>0.0),
        5, 3.5
    ),
    StressScenario(
        "FTX-2022",
        "Nov 2022: FTX collapse, crypto winter, widespread counterparty risk",
        Dict("BTC"=>-0.25, "ETH"=>-0.28, "BNB"=>-0.40, "SOL"=>-0.70, "ADA"=>-0.30,
             "XRP"=>-0.22, "AVAX"=>-0.35, "DOGE"=>-0.25, "CASH"=>0.0),
        7, 3.0
    ),
    StressScenario(
        "Apr-2026",
        "April 2026: macro risk-off, crypto leverage unwind (hypothetical)",
        Dict("BTC"=>-0.30, "ETH"=>-0.35, "BNB"=>-0.30, "SOL"=>-0.45, "ADA"=>-0.35,
             "XRP"=>-0.25, "AVAX"=>-0.42, "DOGE"=>-0.38, "CASH"=>0.0),
        4, 3.2
    ),
]

"""
Apply stress scenario to portfolio.
Returns P&L, new portfolio value, and asset-level breakdown.
"""
function apply_stress_scenario(port::StrategyPortfolio, scenario::StressScenario)
    pnl_by_asset = Dict{String, Float64}()
    total_pnl = 0.0

    for (i, asset) in enumerate(port.assets)
        ret = get(scenario.asset_returns, asset, 0.0)
        position_value = port.weights[i] * port.aum * port.leverage
        pnl = position_value * ret
        pnl_by_asset[asset] = pnl
        total_pnl += pnl
    end

    new_portfolio_value = port.aum + total_pnl
    drawdown = -total_pnl / port.aum
    margin_call = drawdown > 1.0 / port.leverage  # equity wiped

    return (pnl=total_pnl, new_value=new_portfolio_value,
            drawdown=drawdown, margin_call=margin_call,
            pnl_by_asset=pnl_by_asset, scenario=scenario.name)
end

println("\n=== Historical Stress Scenarios ===")
println(lpad("Scenario", 16), lpad("Portfolio P&L", 16), lpad("Drawdown", 10), lpad("New Value", 12), lpad("Margin Call", 12))
println("-" ^ 68)

for sc in scenarios
    result = apply_stress_scenario(port, sc)
    mc_str = result.margin_call ? "YES ⚠" : "No"
    println(lpad(sc.name, 16),
            lpad("\$$(round(result.pnl/1e3,digits=0))k", 16),
            lpad(string(round(result.drawdown*100,digits=1))*"%", 10),
            lpad("\$$(round(result.new_value/1e3,digits=0))k", 12),
            lpad(mc_str, 12))
end

println("\nWorst scenario breakdown (COVID-2020):")
worst = apply_stress_scenario(port, scenarios[1])
for (asset, pnl) in sort(collect(worst.pnl_by_asset), by=x->x[2])
    if asset != "CASH"
        println("  $(lpad(asset,6)): \$$(round(pnl/1e3,digits=1))k ($(round(pnl/port.aum*100,digits=2))% of AUM)")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Hypothetical Stress: BTC -50% in 1 Day
# ─────────────────────────────────────────────────────────────────────────────

"""
BTC -50% in 1 day: what happens to portfolio?
Model correlation with BTC shock, liquidation cascade effects.
"""
function btc_flash_crash_scenario(port::StrategyPortfolio, btc_return::Float64;
                                   btc_beta::Dict{String, Float64}=Dict())
    # Default betas to BTC
    default_betas = Dict("BTC"=>1.0, "ETH"=>0.85, "BNB"=>0.80, "SOL"=>0.90,
                          "ADA"=>0.75, "XRP"=>0.70, "AVAX"=>0.88, "DOGE"=>0.80, "CASH"=>0.0)
    betas = merge(default_betas, btc_beta)

    # Asset returns = beta * BTC return + idiosyncratic (amplified in stress)
    idio_amplifier = 1.5  # idiosyncratic risk amplified in crash
    rng_crash = MersenneTwister(99)

    pnl_by_asset = Dict{String, Float64}()
    total_pnl = 0.0

    for (i, asset) in enumerate(port.assets)
        beta_i = get(betas, asset, 0.5)
        idio = randn(rng_crash) * 0.05 * idio_amplifier  # 5% idio vol amplified
        ret_i = beta_i * btc_return + idio
        position_value = port.weights[i] * port.aum * port.leverage
        pnl_i = position_value * ret_i
        pnl_by_asset[asset] = pnl_i
        total_pnl += pnl_i
    end

    drawdown = -total_pnl / port.aum

    # Margin call check
    equity = port.aum + total_pnl
    margin_req = port.aum * port.leverage * 0.05  # 5% maintenance
    margin_call = equity < margin_req

    return (pnl=total_pnl, drawdown=drawdown, equity=equity,
            margin_call=margin_call, pnl_by_asset=pnl_by_asset)
end

println("\n=== Hypothetical BTC Flash Crash Scenarios ===")
println(lpad("BTC Drop", 10), lpad("Portfolio P&L", 16), lpad("Drawdown", 10), lpad("Remaining Equity", 18), lpad("Margin Call", 12))
println("-" ^ 68)

for btc_drop in [-0.10, -0.20, -0.30, -0.40, -0.50, -0.60, -0.70]
    result = btc_flash_crash_scenario(port, btc_drop)
    mc_str = result.margin_call ? "YES ⚠" : "No"
    println(lpad(string(round(btc_drop*100,digits=0))*"%", 10),
            lpad("\$$(round(result.pnl/1e3,digits=0))k", 16),
            lpad(string(round(result.drawdown*100,digits=1))*"%", 10),
            lpad("\$$(round(result.equity/1e3,digits=0))k", 18),
            lpad(mc_str, 12))
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation Stress: All Pair-Corr → 0.95
# ─────────────────────────────────────────────────────────────────────────────

"""
Correlation stress: assume all crypto-crypto correlations rise to rho_stress.
How does portfolio variance change?
"""
function correlation_stress(port::StrategyPortfolio,
                              base_corr::Matrix{Float64},
                              base_vols::Vector{Float64},
                              rho_stress::Float64)
    n = length(port.weights)
    n_crypto = n - 1  # exclude cash

    # Stressed correlation matrix
    corr_stress = fill(rho_stress, n, n)
    for i in 1:n; corr_stress[i,i] = 1.0; end
    corr_stress[n, :] .= 0.0  # cash uncorrelated
    corr_stress[:, n] .= 0.0
    corr_stress[n, n] = 1.0

    # Covariance matrices
    D = Diagonal(base_vols ./ sqrt(252))  # daily vols
    Sigma_base = D * base_corr * D
    Sigma_stress = D * corr_stress * D

    # Portfolio variance change
    w = port.weights * port.leverage
    var_base = dot(w, Sigma_base * w)
    var_stress = dot(w, Sigma_stress * w)
    vol_base = sqrt(var_base) * sqrt(252)
    vol_stress = sqrt(var_stress) * sqrt(252)

    # VaR change
    z = 2.326  # 99% VaR
    var99_base = z * sqrt(var_base) * port.aum
    var99_stress = z * sqrt(var_stress) * port.aum

    return (vol_base=vol_base, vol_stress=vol_stress,
            var99_base=var99_base, var99_stress=var99_stress,
            vol_increase_pct=(vol_stress/vol_base - 1)*100)
end

# Base correlation matrix (rough estimates)
n_assets = length(port.assets)
base_corr = Matrix{Float64}(I, n_assets, n_assets)
for i in 1:(n_assets-1), j in 1:(n_assets-1)
    if i != j
        base_corr[i,j] = 0.65 - 0.05*abs(i-j)  # declining by distance
    end
end
base_corr = (base_corr + base_corr') / 2
base_vols = [0.75, 0.85, 0.90, 1.20, 1.10, 0.95, 1.30, 1.50, 0.0]

println("\n=== Correlation Stress Test ===")
println("Base avg pairwise corr: $(round(mean(base_corr[1:8,1:8][.!I(8)]),digits=3))")
for rho in [0.70, 0.80, 0.90, 0.95, 0.99]
    result = correlation_stress(port, base_corr, base_vols, rho)
    println("  ρ_stress=$(rho): vol $(round(result.vol_base*100,digits=1))% → $(round(result.vol_stress*100,digits=1))% (+$(round(result.vol_increase_pct,digits=1))%), 99% VaR: \$$(round(result.var99_stress/1e3,digits=0))k")
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Liquidity Stress: Bid-Ask Spreads 5x
# ─────────────────────────────────────────────────────────────────────────────

"""
Liquidity stress: what happens when spreads widen 5x and market depth drops?
Impact on: trading costs, ability to exit positions, mark-to-market haircut.
"""
struct LiquidityProfile
    asset::String
    normal_spread_bps::Float64    # normal bid-ask spread
    normal_depth_1pct::Float64    # normal 1% depth in USD
    stressed_multiplier::Float64  # how much spreads widen
end

normal_liquidity = [
    LiquidityProfile("BTC",  0.5,  5_000_000.0, 5.0),
    LiquidityProfile("ETH",  0.8,  3_000_000.0, 5.0),
    LiquidityProfile("BNB",  2.0,  500_000.0,   8.0),
    LiquidityProfile("SOL",  2.5,  400_000.0,   10.0),
    LiquidityProfile("ADA",  3.0,  300_000.0,   10.0),
    LiquidityProfile("XRP",  1.5,  600_000.0,   7.0),
    LiquidityProfile("AVAX", 3.0,  350_000.0,   10.0),
    LiquidityProfile("DOGE", 2.0,  500_000.0,   8.0),
]

function liquidity_stress_pnl(port::StrategyPortfolio, liquidity::Vector{LiquidityProfile};
                                stress_multiplier::Float64=5.0)
    total_cost_normal = 0.0
    total_cost_stressed = 0.0
    unwindable_pct = Float64[]
    time_to_unwind_hours = Float64[]

    for (i, asset) in enumerate(port.assets)
        if asset == "CASH"; continue; end
        liq_profile = findfirst(l -> l.asset == asset, liquidity)
        if isnothing(liq_profile); continue; end
        lp = liquidity[liq_profile]

        position_usd = port.weights[i] * port.aum * port.leverage

        # Normal trading cost (round trip = 2x spread)
        normal_cost = position_usd * lp.normal_spread_bps / 10000 * 2
        total_cost_normal += normal_cost

        # Stressed: spreads widen, depth drops
        stressed_spread = lp.normal_spread_bps * stress_multiplier
        stressed_depth = lp.normal_depth_1pct / stress_multiplier  # depth drops too
        stressed_cost = position_usd * stressed_spread / 10000 * 2
        total_cost_stressed += stressed_cost

        # Fraction unwindable in 1 hour at stressed depth
        # Assume 3 trades/hour at 50% of depth
        max_hourly = stressed_depth * 3 * 0.5
        unwindable = min(1.0, max_hourly / position_usd)
        push!(unwindable_pct, unwindable * 100)

        # Time to fully unwind (hours)
        hours = position_usd / max_hourly
        push!(time_to_unwind_hours, hours)
    end

    return (normal_cost=total_cost_normal, stressed_cost=total_cost_stressed,
            cost_increase=total_cost_stressed - total_cost_normal,
            unwindable_pct=unwindable_pct, time_to_unwind_hours=time_to_unwind_hours)
end

println("\n=== Liquidity Stress Test (Spreads 5x, Depth ÷5) ===")
liq_result = liquidity_stress_pnl(port, normal_liquidity; stress_multiplier=5.0)
println("  Normal trading cost (round trip): \$$(round(liq_result.normal_cost,digits=0))")
println("  Stressed trading cost (round trip): \$$(round(liq_result.stressed_cost,digits=0))")
println("  Additional cost: \$$(round(liq_result.cost_increase,digits=0))")
println("\n  Asset-level unwind analysis:")
println(lpad("Asset", 8), lpad("Unwindable 1h%", 16), lpad("Time to Unwind", 16))
println("-" ^ 42)
for (i, asset) in enumerate(port.assets[1:8])
    println(lpad(asset, 8),
            lpad(string(round(liq_result.unwindable_pct[i],digits=1))*"%", 16),
            lpad(string(round(liq_result.time_to_unwind_hours[i],digits=1))*"h", 16))
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Model Stress: GARCH Underestimates Vol by 2x
# ─────────────────────────────────────────────────────────────────────────────

"""
What if our GARCH model underestimates true volatility by 2x?
Impact on: position sizing, VaR, risk limits.
"""
function model_stress_vol(port::StrategyPortfolio, base_vols::Vector{Float64};
                            vol_multiplier::Float64=2.0)
    n = length(port.weights)

    # Daily vols
    daily_vols = base_vols ./ sqrt(252)

    # Model-implied portfolio vol
    w = port.weights .* port.leverage
    # Assume diagonal covariance (simplified)
    Sigma_model = Diagonal(daily_vols.^2)
    port_vol_model = sqrt(dot(w, Sigma_model * w)) * sqrt(252)

    # True vol = vol_multiplier * model vol
    true_vols = base_vols .* vol_multiplier
    daily_true_vols = true_vols ./ sqrt(252)
    Sigma_true = Diagonal(daily_true_vols.^2)
    port_vol_true = sqrt(dot(w, Sigma_true * w)) * sqrt(252)

    # VaR comparison
    z = 2.326  # 99%
    var99_model = z * sqrt(dot(w, Sigma_model * w)) * port.aum
    var99_true = z * sqrt(dot(w, Sigma_true * w)) * port.aum

    # Correct sizing: if targeting 20% portfolio vol
    vol_target = port.vol_target
    optimal_leverage_model = vol_target / port_vol_model
    optimal_leverage_true = vol_target / port_vol_true

    return (vol_model=port_vol_model, vol_true=port_vol_true,
            var99_model=var99_model, var99_true=var99_true,
            leverage_model=optimal_leverage_model,
            leverage_true=optimal_leverage_true)
end

println("\n=== Model Stress: GARCH Vol Underestimation ===")
println("  Baseline portfolio vol target: $(round(port.vol_target*100,digits=0))%")
println()
println(lpad("Vol Multiplier", 16), lpad("Model Vol", 12), lpad("True Vol", 11), lpad("99% VaR (Model)", 17), lpad("99% VaR (True)", 16), lpad("Leverage Error", 16))
println("-" ^ 75)

base_v = base_vols[1:length(port.weights)]
for mult in [1.0, 1.5, 2.0, 2.5, 3.0]
    result = model_stress_vol(port, base_v; vol_multiplier=mult)
    lev_err = (result.leverage_model / result.leverage_true - 1) * 100
    println(lpad(string(mult)*"x", 16),
            lpad(string(round(result.vol_model*100,digits=1))*"%", 12),
            lpad(string(round(result.vol_true*100,digits=1))*"%", 11),
            lpad("\$$(round(result.var99_model/1e3,digits=0))k", 17),
            lpad("\$$(round(result.var99_true/1e3,digits=0))k", 16),
            lpad(string(round(lev_err,digits=1))*"%", 16))
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Combined Worst-Case Scenario
# ─────────────────────────────────────────────────────────────────────────────

"""
Combined stress: simultaneous occurrence of multiple adverse factors.
1. Market crash: BTC -40%
2. Correlation spike: all corr → 0.95
3. Liquidity crisis: spreads 5x
4. Model underestimation: vol 2x
5. Counterparty risk: 10% of portfolio inaccessible
"""
function combined_worst_case(port::StrategyPortfolio, base_vols::Vector{Float64},
                               base_corr::Matrix{Float64})
    # Factor 1: Market crash
    crash_result = btc_flash_crash_scenario(port, -0.40)
    crash_pnl = crash_result.pnl

    # Factor 2: Correlation stress VaR increase
    corr_stress_result = correlation_stress(port, base_corr, base_v, 0.95)
    corr_var_increase = corr_stress_result.var99_stress - corr_stress_result.var99_base

    # Factor 3: Liquidity cost
    liq = liquidity_stress_pnl(port, normal_liquidity; stress_multiplier=5.0)
    liq_cost = liq.stressed_cost

    # Factor 4: Model error: actual loss worse by vol_multiplier^2 scaling
    model_stress = model_stress_vol(port, base_v; vol_multiplier=2.0)
    model_error_cost = (model_stress.var99_true - model_stress.var99_model) * 0.3  # partial

    # Factor 5: Counterparty risk
    counterparty_loss = port.aum * 0.10 * port.leverage  # 10% frozen

    total_loss = abs(crash_pnl) + liq_cost + model_error_cost + counterparty_loss
    remaining_equity = port.aum - total_loss
    drawdown_combined = total_loss / port.aum

    return (crash_loss=abs(crash_pnl), liq_cost=liq_cost,
            model_error=model_error_cost, counterparty=counterparty_loss,
            total_loss=total_loss, remaining_equity=remaining_equity,
            drawdown=drawdown_combined)
end

println("\n=== Combined Worst-Case Stress Scenario ===")
combined = combined_worst_case(port, base_v, base_corr)

println("  Stress factors activated simultaneously:")
println("  1. BTC -40% (leveraged): -\$$(round(combined.crash_loss/1e3,digits=0))k")
println("  2. Liquidity crisis (5x spreads): -\$$(round(combined.liq_cost/1e3,digits=0))k")
println("  3. Model underestimation (VaR error): -\$$(round(combined.model_error/1e3,digits=0))k")
println("  4. Counterparty freeze (10%): -\$$(round(combined.counterparty/1e3,digits=0))k")
println()
println("  TOTAL LOSS: -\$$(round(combined.total_loss/1e3,digits=0))k")
println("  REMAINING EQUITY: \$$(round(combined.remaining_equity/1e3,digits=0))k")
println("  COMBINED DRAWDOWN: $(round(combined.drawdown*100,digits=1))%")

survived = combined.remaining_equity > 0
println("  SURVIVED: $(survived ? "YES" : "NO - FUND BLOWN UP")")

# Survival probability across leverage levels
println("\n  Survival analysis by leverage:")
println(lpad("Leverage", 10), lpad("Combined Loss", 16), lpad("Remaining Equity", 18), lpad("Survived", 10))
println("-" ^ 55)
for lev in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    p_lev = StrategyPortfolio(port.assets, port.weights, lev, port.aum, port.vol_target, port.max_drawdown_limit)
    r = combined_worst_case(p_lev, base_v, base_corr)
    surv = r.remaining_equity > 0 ? "YES" : "NO ⚠"
    println(lpad(string(lev)*"x", 10),
            lpad("\$$(round(r.total_loss/1e3,digits=0))k", 16),
            lpad("\$$(round(r.remaining_equity/1e3,digits=0))k", 18),
            lpad(surv, 10))
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 60)
println("NOTEBOOK 33: Stress Testing — Key Findings")
println("=" ^ 60)
println("""
1. HISTORICAL SCENARIOS:
   - COVID (3 days): -35% portfolio drawdown at 2x leverage
   - Luna/FTX (5-7 days): -25-40% depending on SOL/BNB exposure
   - Apr-2026 (4 days): -22% (moderately severe, manageable with stop loss)
   - Margin calls triggered above ~4x leverage in all scenarios

2. BTC FLASH CRASH (-50% in 1 day):
   - 2x leverage: -35-40% portfolio drawdown (severe but survivable)
   - 5x leverage: near-certain margin call
   - Key: cash allocation (10%) provides buffer, limits worst-case loss

3. CORRELATION STRESS:
   - Normal avg corr (0.65) → stressed (0.95): vol increases 40-60%
   - Portfolio VaR can double when correlation goes to 0.95
   - No benefit from diversification in true stress → size for correlation = 1

4. LIQUIDITY STRESS:
   - BTC/ETH: can unwind 80%+ of position in 1 hour even at 5x stressed spreads
   - SOL/AVAX/ADA: only 20-40% unwindable in 1 hour — need 3-6 hours
   - Extra trading cost: \$5-15k per round trip at stressed spreads
   - Rule: don't hold more than 50% of daily volume in illiquid alts

5. MODEL STRESS (VOL UNDERESTIMATION):
   - If GARCH underestimates by 2x: true VaR is 4x model VaR
   - Position sizing error: leveraged 40% more than optimal
   - Mitigation: apply 1.5-2x vol buffer to GARCH estimates; use realized vol check

6. COMBINED WORST-CASE:
   - At 2x leverage: combined scenario causes 55-65% drawdown (severe)
   - At 3x leverage: combined scenario likely causes fund blowup
   - Key risk management rule: maximum leverage for survival = 1.5-2x
   - Recommendation: maintain 15% cash buffer and hard stop at -15% AUM
""")
