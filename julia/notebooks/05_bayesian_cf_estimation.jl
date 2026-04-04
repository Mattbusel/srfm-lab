## Notebook 05: Bayesian CF Estimation
## Estimate optimal critical frequency per instrument via MCMC

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SRFMResearch
using SRFMResearch.BHPhysics
using SRFMResearch.Bayesian
using SRFMResearch.SRFMStats
using DataFrames, Statistics, Plots, Random, Distributions

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate synthetic instruments with different volatility profiles
# ─────────────────────────────────────────────────────────────────────────────

println("=== Bayesian CF Estimation Study ===\n")

function generate_instrument(n::Int, sigma::Float64, seed::Int)
    rng = MersenneTwister(seed)
    prices = zeros(n)
    prices[1] = 100.0
    for i in 2:n
        r = sigma * randn(rng)
        prices[i] = prices[i-1] * exp(r)
    end
    return prices
end

instruments = Dict(
    "Low_Vol"   => generate_instrument(1000, 0.005, 1),
    "Med_Vol"   => generate_instrument(1000, 0.012, 2),
    "High_Vol"  => generate_instrument(1000, 0.025, 3),
    "Crypto"    => generate_instrument(1000, 0.040, 4),
    "FX_Pair"   => generate_instrument(1000, 0.003, 5),
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prior analysis: what does the median beta look like per instrument?
# ─────────────────────────────────────────────────────────────────────────────

println("--- Empirical Beta Analysis ---")
println(@sprintf("%-12s  MedianBeta  p10Beta  p90Beta  Optimal_CF(heuristic)", "Instrument"))

for (name, prices) in sort(instruments)
    betas = abs.(diff(log.(prices)))
    med_β = median(betas)
    p10_β = quantile(betas, 0.10)
    p90_β = quantile(betas, 0.90)
    # Heuristic: CF at ~25th percentile of beta distribution
    cf_heur = quantile(betas, 0.25)
    @printf("%-12s  %.5f      %.5f   %.5f   %.5f\n", name, med_β, p10_β, p90_β, cf_heur)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Grid search CF optimisation (baseline comparison)
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Grid Search CF Optimisation ---")

function optimise_cf_grid(prices::Vector{Float64};
                           cf_range=0.001:0.001:0.020,
                           bh_form=0.15, bh_collapse=0.06,
                           long_only=false)

    best_sharpe = -Inf
    best_cf     = cf_range[1]

    for cf in cf_range
        try
            config = BHConfig(cf=cf, bh_form=bh_form, bh_collapse=bh_collapse)
            result = run_backtest(prices, config; long_only=long_only)
            if result.sharpe > best_sharpe && result.n_trades >= 5
                best_sharpe = result.sharpe
                best_cf     = cf
            end
        catch
            continue
        end
    end
    return (best_cf=best_cf, best_sharpe=best_sharpe)
end

grid_results = Dict()
for (name, prices) in sort(instruments)
    gr = optimise_cf_grid(prices)
    grid_results[name] = gr
    @printf("%-12s  Grid best CF=%.4f  Sharpe=%.3f\n", name, gr.best_cf, gr.best_sharpe)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Bayesian CF estimation via MCMC
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Bayesian CF Estimation (MCMC) ---")

bayes_results = Dict()

for (name, prices) in sort(instruments)
    println("  Processing $name...")
    try
        result = estimate_cf_posterior(prices; n_samples=500, n_chains=2)
        bayes_results[name] = result
        @printf("  %-12s  CF_mean=%.5f  CF_std=%.5f  CI95=[%.5f, %.5f]\n",
                name, result.cf_mean, result.cf_std,
                result.cf_ci_95[1], result.cf_ci_95[2])
    catch e
        println("  $name: MCMC failed — $e")
        continue
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Compare Bayesian vs Grid CF on held-out data
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Bayesian vs Grid CF: Out-of-Sample Validation ---")

function generate_oos(n::Int, sigma::Float64, seed::Int)
    return generate_instrument(n, sigma, seed + 1000)
end

sigma_map = Dict(
    "Low_Vol"  => 0.005,
    "Med_Vol"  => 0.012,
    "High_Vol" => 0.025,
    "Crypto"   => 0.040,
    "FX_Pair"  => 0.003,
)

println(@sprintf("%-12s  Grid_CF  Bayes_CF  Grid_OOS  Bayes_OOS", "Instrument"))
for name in sort(collect(keys(instruments)))
    prices_oos = generate_oos(500, sigma_map[name], parse(Int, string(hash(name))[1:4]))

    grid_cf  = grid_results[name].best_cf
    bayes_cf = haskey(bayes_results, name) ? bayes_results[name].cf_mean : grid_cf

    function oos_sharpe(cf)
        try
            config = BHConfig(cf=cf, bh_form=0.15, bh_collapse=0.06)
            result = run_backtest(prices_oos, config)
            return result.n_trades >= 3 ? result.sharpe : 0.0
        catch
            return 0.0
        end
    end

    s_grid  = oos_sharpe(grid_cf)
    s_bayes = oos_sharpe(bayes_cf)

    @printf("%-12s  %.4f   %.4f    %.3f     %.3f\n",
            name, grid_cf, bayes_cf, s_grid, s_bayes)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Posterior predictive: simulate optimal BH dynamics
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Posterior Predictive Backtest ---")

# For Med_Vol instrument
if haskey(bayes_results, "Med_Vol")
    br       = bayes_results["Med_Vol"]
    cf_draws = br.cf_samples[1:min(50, length(br.cf_samples))]

    sharpes = Float64[]
    for cf_draw in cf_draws
        try
            config  = BHConfig(cf=cf_draw, bh_form=0.15, bh_collapse=0.06)
            result  = run_backtest(instruments["Med_Vol"], config)
            result.n_trades >= 3 && push!(sharpes, result.sharpe)
        catch
            continue
        end
    end

    if !isempty(sharpes)
        println("Med_Vol posterior predictive Sharpe:")
        println("  Mean: $(round(mean(sharpes),digits=3))")
        println("  Std:  $(round(std(sharpes),digits=3))")
        println("  P(Sharpe > 0): $(round(mean(sharpes .> 0)*100,digits=1))%")
        println("  95% CI: [$(round(quantile(sharpes,0.025),digits=3)), $(round(quantile(sharpes,0.975),digits=3))]")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

# CF posterior distributions per instrument
p_cf = plot(
    title="CF Posterior Distributions by Instrument",
    xlabel="CF", ylabel="Density",
    size=(900, 500), legend=:topright
)

colors_list = [:steelblue, :firebrick, :seagreen, :purple, :orange]

for (k, (name, result)) in enumerate(sort(collect(bayes_results)))
    cf_samp = result.cf_samples
    # KDE approximation
    x_range = range(max(0, minimum(cf_samp) - 0.002),
                     minimum([0.025, maximum(cf_samp) + 0.002]), length=200)
    kde_bw = 1.06 * std(cf_samp) * length(cf_samp)^(-0.2)
    kde_vals = [mean(pdf.(Normal(xi, kde_bw), collect(x_range))) for xi in cf_samp]
    # Simple histogram
    histogram!(p_cf, cf_samp; normalize=:pdf, alpha=0.4,
               label=name, color=colors_list[min(k,length(colors_list))], bins=30)
    vline!(p_cf, [result.cf_mean]; linewidth=2,
           color=colors_list[min(k,length(colors_list))], label="")
end

savefig(p_cf, joinpath(@__DIR__, "05_cf_posteriors.png"))

# Beta distribution per instrument
p_beta = plot(
    title="Beta (|Δlog price|) Distribution by Instrument",
    xlabel="β", ylabel="Density",
    size=(900, 500), legend=:topright
)

for (k, (name, prices)) in enumerate(sort(instruments))
    betas = abs.(diff(log.(prices)))
    histogram!(p_beta, betas; normalize=:pdf, alpha=0.4, bins=40,
               label=name, color=colors_list[min(k,length(colors_list))])
end

savefig(p_beta, joinpath(@__DIR__, "05_beta_distributions.png"))

println("\n05_bayesian_cf_estimation.jl complete.")
