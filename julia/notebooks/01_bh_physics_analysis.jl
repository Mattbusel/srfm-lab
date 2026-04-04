## Notebook 01: BH Physics Analysis
## Analyse Black-Hole mass dynamics across instruments
## Run as a Julia script or in Pluto.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SRFMResearch
using SRFMResearch.BHPhysics
using DataFrames, Statistics, Dates, CSV, Plots

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic price generation for analysis
# ─────────────────────────────────────────────────────────────────────────────

"""Generate a synthetic trending + volatile price series."""
function synthetic_prices(n::Int=2000; mu=0.0002, sigma=0.015, seed=42)
    rng    = MersenneTwister(seed)
    prices = zeros(n)
    prices[1] = 100.0
    for i in 2:n
        regime_shift = mod(i, 500) == 0 ? randn(rng) * 0.05 : 0.0
        r = mu + sigma * randn(rng) + regime_shift
        prices[i] = prices[i-1] * exp(r)
    end
    return prices
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Basic mass series inspection
# ─────────────────────────────────────────────────────────────────────────────

println("=== BH Physics Analysis ===\n")

prices = synthetic_prices(3000)

config = BHConfig(cf=0.003, bh_form=0.20, bh_collapse=0.08,
                   bh_decay=0.97, ctl_req=3)

ms = mass_series(prices, config)

println("Price series: $(length(prices)) bars")
println("Active fraction: $(round(mean(ms.active) * 100, digits=1))%")
println("Timelike fraction: $(round(mean(ms.timelike) * 100, digits=1))%")
println("Mean mass (active): $(round(mean(ms.masses[ms.active]), digits=4))")
println("Mean mass (inactive): $(round(mean(ms.masses[.!ms.active]), digits=4))")
println("Peak mass: $(round(maximum(ms.masses), digits=4))")

# Regime breakdown
regimes = bh_regime(ms.masses, ms.active, ms.bh_dir)
regime_counts = Dict{String, Int}()
for r in regimes
    regime_counts[r] = get(regime_counts, r, 0) + 1
end
println("\nRegime distribution:")
for (r, cnt) in sort(collect(regime_counts), by=x->-x[2])
    println("  $r: $(round(cnt/length(regimes)*100, digits=1))%")
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Parameter sensitivity grid search
# ─────────────────────────────────────────────────────────────────────────────

println("\n=== Parameter Sensitivity ===")

cf_range   = 0.001:0.001:0.010
form_range = [0.10, 0.15, 0.20, 0.25, 0.30]

sens_df = parameter_sensitivity(prices, collect(cf_range), form_range)
println("Grid search: $(nrow(sens_df)) combinations")

# Top 5 by Sharpe
top5 = first(sort(sens_df, :sharpe, rev=true), 5)
println("\nTop 5 by Sharpe:")
show(top5[:, [:cf, :bh_form, :sharpe, :n_trades, :max_dd]])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Backtest with optimal config
# ─────────────────────────────────────────────────────────────────────────────

best_cf   = top5[1, :cf]
best_form = top5[1, :bh_form]
opt_config = BHConfig(cf=best_cf, bh_form=best_form)

println("\n=== Backtest with Optimal Config ===")
println("CF=$(best_cf), Form=$(best_form)")

result = run_backtest(prices, opt_config; long_only=false, commission=0.0004)
println("Total return: $(round(result.total_return * 100, digits=2))%")
println("Sharpe ratio: $(round(result.sharpe, digits=3))")
println("Max drawdown: $(round(result.max_dd * 100, digits=2))%")
println("Trades: $(result.n_trades)")

if !isempty(result.trades)
    trades_df = trades_to_dataframe(result.trades)
    win_rate  = mean(trades_df[!, :pnl] .> 0) * 100
    avg_pnl   = mean(trades_df[!, :pnl]) * 100
    avg_dur   = mean(trades_df[!, :duration])
    println("Win rate: $(round(win_rate, digits=1))%")
    println("Avg PnL per trade: $(round(avg_pnl, digits=3))%")
    println("Avg duration: $(round(avg_dur, digits=1)) bars")
    println("\nRegime breakdown of trades:")
    for (r, cnt) in countmap(trades_df[!, :regime])
        pnl_r = mean(trades_df[trades_df[!, :regime] .== r, :pnl]) * 100
        println("  $r: $(cnt) trades, avg PnL=$(round(pnl_r,digits=2))%")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-sectional mass analysis (multiple instruments)
# ─────────────────────────────────────────────────────────────────────────────

println("\n=== Cross-Sectional Analysis ===")

symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
prices_dict = Dict(sym => synthetic_prices(1000; mu=randn()*0.0002, sigma=0.01+rand()*0.02)
                   for sym in symbols)

cross_df = mass_cross_sectional(prices_dict, opt_config)

# Summary per symbol
for sym in symbols
    sym_df    = cross_df[cross_df[!, :symbol] .== sym, :]
    act_frac  = mean(sym_df[!, :active])
    avg_mass  = mean(sym_df[!, :mass])
    println("  $sym: active=$(round(act_frac*100,digits=1))%, mean_mass=$(round(avg_mass,digits=4))")
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Walk-forward analysis
# ─────────────────────────────────────────────────────────────────────────────

println("\n=== Walk-Forward Analysis ===")

wf_df = walk_forward_bh(prices, opt_config,
                         train_size=500, test_size=200, step=100)

println("Walk-forward windows: $(nrow(wf_df))")
if !isempty(wf_df)
    println("Mean OOS Sharpe: $(round(mean(filter(!isnan, wf_df[!,:sharpe])), digits=3))")
    println("% profitable windows: $(round(mean(wf_df[!,:total_ret] .> 0)*100, digits=1))%")
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

p1 = plot_bh_mass_series(prices[1:500], ms.masses[1:500], ms.active[1:500],
                          ms.bh_dir[1:500]; cf=opt_config.cf,
                          bh_form=opt_config.bh_form,
                          title="BH Mass Dynamics — Synthetic Prices")

p2 = plot_equity_curve(result.equity_curve, result.regime_series;
                        title="BH Backtest Equity Curve")

p3 = plot_drawdown(result.equity_curve; title="BH Backtest Drawdown")

p4 = if !isempty(result.trades)
    plot_trade_analysis(trades_to_dataframe(result.trades))
end

# Save or display
savefig(p1, joinpath(@__DIR__, "01_bh_mass_dynamics.png"))
savefig(p2, joinpath(@__DIR__, "01_equity_curve.png"))
savefig(p3, joinpath(@__DIR__, "01_drawdown.png"))
println("\nPlots saved to notebooks/")

# Multi-symbol heatmap
dates_vec = Date(2020,1,1) .+ Day.(0:999)
mass_mat  = zeros(length(symbols), 1000)
for (i, sym) in enumerate(symbols)
    m = mass_series(prices_dict[sym], opt_config)
    mass_mat[i, :] = m.masses
end
p_heat = plot_bh_heatmap(mass_mat, symbols, dates_vec)
savefig(p_heat, joinpath(@__DIR__, "01_bh_heatmap.png"))

println("\n01_bh_physics_analysis.jl complete.")
