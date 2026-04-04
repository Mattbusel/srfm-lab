## Notebook 06: Factor Model Analysis
## Full factor model on SRFM trades: regression, IC, IC-IR, attribution

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SRFMResearch
using SRFMResearch.BHPhysics
using SRFMResearch.SRFMStats
using SRFMResearch.SRFMOptimization
using SRFMResearch.SRFMViz
using DataFrames, Statistics, LinearAlgebra, GLM, Plots, Random

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate synthetic SRFM trade data
# ─────────────────────────────────────────────────────────────────────────────

println("=== Factor Model Analysis ===\n")

rng = MersenneTwister(42)

function generate_srfm_trades(n_trades::Int=500; seed::Int=42)
    rng = MersenneTwister(seed)

    # Synthetic factors
    tf_scores     = rand(rng, -3:3, n_trades)
    peak_masses   = 0.1 .+ rand(rng, n_trades) .* 0.4
    durations     = rand(rng, 2:50, n_trades)
    regimes       = rand(rng, ["BH_BULL","BH_BEAR","BH_NEUTRAL"], n_trades)
    directions    = [rand(rng) > 0.5 ? 1 : -1 for _ in 1:n_trades]
    vol_at_entry  = 0.008 .+ rand(rng, n_trades) .* 0.020
    bh_age        = rand(rng, 1:30, n_trades)
    market_trend  = randn(rng, n_trades) .* 0.01

    # True PnL model (linear combination + noise)
    pnl = (0.0015 .* tf_scores .+
           0.003  .* peak_masses .-
           0.0001 .* durations .+
           0.002  .* (regimes .== "BH_BULL") .-
           0.001  .* (regimes .== "BH_BEAR") .+
           0.005  .* market_trend .+
           0.003  .* randn(rng, n_trades))

    return DataFrame(
        trade_id    = 1:n_trades,
        pnl         = pnl,
        tf_score    = Float64.(tf_scores),
        peak_mass   = peak_masses,
        duration    = Float64.(durations),
        regime      = regimes,
        direction   = Float64.(directions),
        vol_entry   = vol_at_entry,
        bh_age      = Float64.(bh_age),
        market_trend= market_trend,
    )
end

trades_df = generate_srfm_trades(600)
println("Generated $(nrow(trades_df)) synthetic SRFM trades")
println("Win rate: $(round(mean(trades_df.pnl .> 0)*100,digits=1))%")
println("Mean PnL: $(round(mean(trades_df.pnl)*100,digits=3))%")
println("Std PnL:  $(round(std(trades_df.pnl)*100,digits=3))%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Univariate Factor Analysis (IC per factor)
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Univariate Factor IC ---")

factor_cols = [:tf_score, :peak_mass, :duration, :direction, :vol_entry, :bh_age, :market_trend]

println(@sprintf("%-18s  IC        t-stat    p-value   Significant", "Factor"))
for fcol in factor_cols
    ic = rank_ic(trades_df[!, fcol], trades_df[!, :pnl])
    n  = nrow(trades_df)
    t  = ic * sqrt(n - 2) / sqrt(max(1 - ic^2, 1e-10))
    p  = 2 * (1 - cdf(TDist(n-2), abs(t)))
    sig = p < 0.05 ? "YES" : "no"
    @printf("%-18s  %7.4f   %7.3f   %7.4f   %s\n", string(fcol), ic, t, p, sig)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Rolling IC Series
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Rolling IC (window=50 trades) ---")

window = 50
n = nrow(trades_df)

ic_series = Dict{Symbol, Vector{Float64}}()
for fcol in factor_cols
    ic_roll = fill(NaN, n)
    for i in window:n
        ic_roll[i] = rank_ic(trades_df[i-window+1:i, fcol],
                              trades_df[i-window+1:i, :pnl])
    end
    ic_series[fcol] = ic_roll
end

# IC-IR per factor
println(@sprintf("%-18s  IC_mean   IC_std   ICIR", "Factor"))
for fcol in factor_cols
    ic = filter(!isnan, ic_series[fcol])
    if length(ic) >= 10
        ic_m = mean(ic); ic_s = std(ic)
        ir   = ic_s > 1e-10 ? ic_m / ic_s * sqrt(252) : 0.0
        @printf("%-18s  %7.4f   %7.4f   %7.3f\n", string(fcol), ic_m, ic_s, ir)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Multivariate OLS Factor Regression
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- OLS Factor Regression ---")

# Encode regime dummies
trades_df[!, :is_bull]    = Float64.(trades_df[!, :regime] .== "BH_BULL")
trades_df[!, :is_bear]    = Float64.(trades_df[!, :regime] .== "BH_BEAR")

formula = @formula(pnl ~ tf_score + peak_mass + duration + vol_entry +
                          bh_age + market_trend + is_bull + is_bear)
ols_model = lm(formula, trades_df)

println("R²: $(round(r2(ols_model), digits=4))")
println("Adj R²: $(round(adjr2(ols_model), digits=4))")
println("\nFactor Coefficients:")
coef_table = coeftable(ols_model)

# Manual extraction for readable output
coef_names = coef_table.rownms
coef_vals  = coef_table.cols[1]
coef_se    = coef_table.cols[2]
coef_t     = coef_table.cols[3]
coef_p     = coef_table.cols[4]

for i in 1:length(coef_names)
    sig = coef_p[i] < 0.05 ? "*" : ""
    @printf("  %-18s  β=%8.5f  SE=%.5f  t=%.3f  p=%.4f %s\n",
            coef_names[i], coef_vals[i], coef_se[i], coef_t[i], coef_p[i], sig)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Factor Attribution (decompose PnL)
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Factor PnL Attribution ---")

fitted_vals = predict(ols_model)
residuals   = trades_df[!, :pnl] .- fitted_vals

# R² per factor (incremental)
factors_ordered = [:tf_score, :peak_mass, :market_trend, :is_bull, :is_bear,
                    :duration, :vol_entry, :bh_age]

function r2_single(y, x)
    n = length(y); mx = mean(x); my = mean(y)
    beta = cov(x, y) / max(var(x), 1e-12)
    alpha = my - beta * mx
    resid = y .- (alpha .+ beta .* x)
    return 1 - var(resid) / max(var(y), 1e-12)
end

println(@sprintf("%-18s  R²_single  Corr_PnL", "Factor"))
for fcol in factors_ordered
    r2_f  = r2_single(trades_df[!, :pnl], trades_df[!, fcol])
    corr_f = cor(trades_df[!, :pnl], trades_df[!, fcol])
    @printf("%-18s  %.4f     %.4f\n", string(fcol), r2_f, corr_f)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Regime-conditional performance
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Regime-Conditional Statistics ---")

for reg in ["BH_BULL", "BH_BEAR", "BH_NEUTRAL"]
    mask   = trades_df[!, :regime] .== reg
    n_reg  = sum(mask)
    n_reg < 3 && continue
    pnls   = trades_df[mask, :pnl]
    wr     = mean(pnls .> 0) * 100
    avg_p  = mean(pnls) * 100
    std_p  = std(pnls) * 100
    ic_r   = rank_ic(trades_df[mask, :tf_score], pnls)
    @printf("  %-12s  n=%3d  WR=%.1f%%  AvgPnL=%.3f%%  Std=%.3f%%  IC_tfs=%.3f\n",
            reg, n_reg, wr, avg_p, std_p, ic_r)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Partial Correlations
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Partial Correlations ---")

factor_matrix = Matrix(trades_df[!, [:pnl, :tf_score, :peak_mass, :duration,
                                      :vol_entry, :market_trend]])
pcorr = partial_correlation(factor_matrix)
labels_pc = ["pnl", "tf_score", "peak_mass", "duration", "vol_entry", "market_trend"]

println("Partial correlations with PnL:")
for (i, name) in enumerate(labels_pc[2:end])
    @printf("  %-18s  ρ_partial=%.4f\n", name, pcorr[1, i+1])
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

# Factor exposures (OLS betas, normalised)
betas_normed = coef_vals[2:end] .* std.(eachcol(Matrix(trades_df[!, factor_cols[1:end]])[1:end, 1:length(coef_vals)-1]))
exp_nt = NamedTuple(Symbol(coef_names[i+1]) => coef_vals[i+1] for i in 1:length(coef_names)-1)
p_exp  = plot_factor_exposures(exp_nt)
savefig(p_exp, joinpath(@__DIR__, "06_factor_exposures.png"))

# IC heatmap (factors × rolling windows)
ic_matrix = hcat([ic_series[f] for f in factors_ordered if haskey(ic_series, f)]...)

# Trade PnL distribution
p_dist = plot_returns_distribution(trades_df[!, :pnl]; title="SRFM Trade PnL Distribution")
savefig(p_dist, joinpath(@__DIR__, "06_trade_pnl_distribution.png"))

# Rolling IC for top 3 factors
p_ic = plot(
    title="Rolling IC (window=$window trades)",
    xlabel="Trade #", ylabel="IC",
    size=(1100, 500), legend=:topright
)
top_factors = [:tf_score, :peak_mass, :market_trend]
ic_colors   = [:steelblue, :seagreen, :crimson]

for (k, fcol) in enumerate(top_factors)
    ic_r = ic_series[fcol]
    valid = .!isnan.(ic_r)
    plot!(p_ic, findall(valid), ic_r[valid];
          label=string(fcol), color=ic_colors[k], linewidth=1.5)
end
hline!(p_ic, [0.0]; color=:black, linewidth=1, label="")
savefig(p_ic, joinpath(@__DIR__, "06_rolling_ic.png"))

# Residual plot
p_resid = scatter(
    fitted_vals .* 100, residuals .* 100;
    xlabel="Fitted PnL (%)", ylabel="Residual (%)",
    title="OLS Factor Model Residuals",
    markersize=3, markerstrokewidth=0, alpha=0.5,
    color=:steelblue, label="Residuals",
    size=(700, 500)
)
hline!(p_resid, [0.0]; color=:red, linewidth=2, label="")
savefig(p_resid, joinpath(@__DIR__, "06_ols_residuals.png"))

println("\n06_factor_model_analysis.jl complete.")
