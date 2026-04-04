## Notebook 03: Portfolio Optimization Study
## Compare MVO, Risk Parity, HRP, Black-Litterman, Equal Weight

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SRFMResearch
using SRFMResearch.SRFMOptimization
using SRFMResearch.SRFMStats
using SRFMResearch.SRFMViz
using DataFrames, Statistics, LinearAlgebra, Plots, Random

# ─────────────────────────────────────────────────────────────────────────────
# 1. Synthetic multi-asset universe
# ─────────────────────────────────────────────────────────────────────────────

function generate_asset_returns(n_assets::Int=10, n_bars::Int=2000; seed=42)
    rng = MersenneTwister(seed)

    # Factor model: 3 factors
    n_factors = 3
    F = 0.02 * randn(rng, n_bars, n_factors)
    B = 0.5 * randn(rng, n_assets, n_factors)   # factor loadings
    E = 0.015 * randn(rng, n_bars, n_assets)     # idiosyncratic

    mu_daily = randn(rng, n_assets) * 0.0002 .+ 0.0001
    returns  = mu_daily' .+ F * B' + E

    return returns, mu_daily
end

println("=== Portfolio Optimization Study ===\n")

n_assets = 8
n_bars   = 1500
symbols  = ["Asset_$i" for i in 1:n_assets]

returns_mat, true_mu = generate_asset_returns(n_assets, n_bars)

# Estimate moments
Σ = cov(returns_mat) * 252   # annualised
μ = vec(mean(returns_mat, dims=1)) * 252   # annualised

println("Assets: $n_assets")
println("Estimation window: $n_bars bars")
println("Expected returns range: $(round(minimum(μ)*100,digits=1))% to $(round(maximum(μ)*100,digits=1))%")
println("Volatility range: $(round(minimum(sqrt.(diag(Σ)))*100,digits=1))% to $(round(maximum(sqrt.(diag(Σ)))*100,digits=1))%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute all portfolio allocations
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Computing Portfolio Allocations ---")

w_eq   = equal_weight(n_assets)
w_iv   = inverse_vol_weight(Σ)
w_mv   = min_variance(Σ)
w_ms   = max_sharpe(μ, Σ, 0.0)
w_rp   = risk_parity(Σ)
w_hrp  = hierarchical_risk_parity(Σ)
w_md   = max_diversification(Σ)

# Black-Litterman: one view (Asset 1 outperforms by 2% ann.)
mu_prior = capm_implied_returns(w_eq, Σ / 252) * 252
P = zeros(1, n_assets); P[1, 1] = 1.0; P[1, 2] = -1.0
Q = [0.02]
Ω = Diagonal([0.005^2])
bl_result = black_litterman(mu_prior, Σ, P, Q, Ω)
w_bl = max.(bl_result.posterior_weights, 0.0)
w_bl ./= sum(w_bl)

# Collect
weights_dict = Dict(
    "Equal Weight"   => w_eq,
    "Inv Vol"        => w_iv,
    "Min Variance"   => w_mv,
    "Max Sharpe"     => w_ms,
    "Risk Parity"    => w_rp,
    "HRP"            => w_hrp,
    "Max Divers."    => w_md,
    "Black-Litterman"=> w_bl,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Performance evaluation (IS)
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- In-Sample Portfolio Performance ---")

Σ_daily = Σ / 252
μ_daily = μ / 252

results = NamedTuple[]
for (name, w) in sort(weights_dict)
    port_ret  = dot(w, μ)
    port_vol  = sqrt(max(dot(w, Σ * w), 1e-10))
    port_sharpe = port_ret / port_vol
    rc  = risk_contributions(w, Σ_daily)
    push!(results, (
        method     = name,
        weights    = w,
        exp_return = port_ret * 100,
        volatility = port_vol * 100,
        sharpe     = port_sharpe,
        max_rc     = maximum(rc) * 100,
    ))
end
res_df = DataFrame(results)
sort!(res_df, :sharpe, rev=true)

println("Portfolio Performance (IS):")
for row in eachrow(res_df)
    @printf("  %-18s  Return=%5.1f%%  Vol=%5.1f%%  Sharpe=%5.3f  MaxRC=%4.1f%%\n",
            row.method, row.exp_return, row.volatility, row.sharpe, row.max_rc)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Out-of-sample backtest (rolling rebalance)
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Rolling Out-of-Sample Backtest ---")

function portfolio_backtest(returns_mat::Matrix{Float64}, method::String;
                             est_window::Int=252, rebal_freq::Int=21,
                             commission::Float64=0.001)

    n_bars, n_assets = size(returns_mat)
    eq = fill(1.0, n_bars)
    w  = ones(n_assets) / n_assets
    prev_w = copy(w)

    for i in (est_window + 1):n_bars
        # Rebalance
        if mod(i - est_window, rebal_freq) == 1
            Σ_est = cov(returns_mat[i-est_window:i-1, :]) * 252
            μ_est = vec(mean(returns_mat[i-est_window:i-1, :], dims=1)) * 252

            w = try
                if method == "Equal Weight"
                    equal_weight(n_assets)
                elseif method == "Inv Vol"
                    inverse_vol_weight(Σ_est)
                elseif method == "Min Variance"
                    min_variance(Σ_est)
                elseif method == "Max Sharpe"
                    max_sharpe(μ_est, Σ_est, 0.0)
                elseif method == "Risk Parity"
                    risk_parity(Σ_est)
                elseif method == "HRP"
                    hierarchical_risk_parity(Σ_est)
                else
                    equal_weight(n_assets)
                end
            catch
                equal_weight(n_assets)
            end

            # Turnover cost
            turnover = sum(abs.(w .- prev_w)) / 2
            eq[i]  *= (1 - commission * turnover)
            prev_w  = copy(w)
        end

        # Bar return
        bar_ret = dot(w, returns_mat[i, :])
        eq[i] *= exp(bar_ret) * (i > est_window + 1 ? 1.0 : 1.0)
    end
    return eq[est_window+1:end]
end

oos_methods = ["Equal Weight", "Inv Vol", "Min Variance", "Max Sharpe",
                "Risk Parity", "HRP"]

oos_results = NamedTuple[]
for method in oos_methods
    eq = portfolio_backtest(returns_mat, method)
    rets = diff(log.(eq))
    report = returns_statistics_report(rets; equity=eq)
    push!(oos_results, (
        method  = method,
        sharpe  = report.sharpe,
        cagr    = report.cagr * 100,
        max_dd  = report.max_drawdown * 100,
        vol     = report.annualised_vol * 100,
    ))
    println("  $(rpad(method, 18)) Sharpe=$(round(report.sharpe,digits=3)), CAGR=$(round(report.cagr*100,digits=1))%, MaxDD=$(round(report.max_drawdown*100,digits=1))%")
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Computing Efficient Frontier ---")

frontier = efficient_frontier(μ, Σ, 80)
opt_pt   = frontier[argmax([f.sharpe for f in frontier])]

println("Optimal portfolio: Return=$(round(opt_pt.achieved_return*100,digits=1))%, Vol=$(round(opt_pt.risk*100,digits=1))%, Sharpe=$(round(opt_pt.sharpe,digits=3))")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Weight stability analysis
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Weight Stability Analysis ---")

est_window = 252
n_windows  = n_bars - est_window

w_mv_series  = zeros(n_windows, n_assets)
w_rp_series  = zeros(n_windows, n_assets)

for i in 1:n_windows
    Σ_est = cov(returns_mat[i:i+est_window-1, :]) * 252
    w_mv_series[i, :] = try min_variance(Σ_est) catch; equal_weight(n_assets); end
    w_rp_series[i, :] = try risk_parity(Σ_est) catch; equal_weight(n_assets); end
end

mv_turnover = mean(sum(abs.(diff(w_mv_series, dims=1)), dims=2) ./ 2)
rp_turnover = mean(sum(abs.(diff(w_rp_series, dims=1)), dims=2) ./ 2)
println("Min Variance turnover: $(round(mv_turnover*100,digits=2))% per rebalance")
println("Risk Parity turnover:  $(round(rp_turnover*100,digits=2))% per rebalance")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

# Efficient frontier
p_front = plot_efficient_frontier(frontier, opt_pt)
savefig(p_front, joinpath(@__DIR__, "03_efficient_frontier.png"))

# Correlation matrix
corr_mat = cor(returns_mat)
p_corr = plot_correlation(corr_mat, symbols)
savefig(p_corr, joinpath(@__DIR__, "03_correlation_matrix.png"))

# Weight comparison bar chart
p_weights = plot(
    title="Portfolio Weights Comparison",
    xlabel="Asset", ylabel="Weight",
    size=(1200, 500), legend=:topright
)
for (name, w) in sort(weights_dict)
    plot!(p_weights, 1:n_assets, w;
          label=name, marker=:circle, linewidth=2)
end
savefig(p_weights, joinpath(@__DIR__, "03_weights_comparison.png"))

# Factor exposure for max Sharpe
exp_data = NamedTuple(Symbol(sym) => w_ms[i] for (i, sym) in enumerate(symbols))
p_exp = plot_factor_exposures(exp_data)
savefig(p_exp, joinpath(@__DIR__, "03_factor_exposures.png"))

println("\n03_portfolio_optimization_study.jl complete.")
