## Notebook 04: Stochastic Process Calibration
## Fit GBM, GARCH, Heston, OU, MJD, Hawkes to synthetic + real-like data

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SRFMResearch
using SRFMResearch.Stochastic
using SRFMResearch.SRFMStats
using SRFMResearch.SRFMViz
using DataFrames, Statistics, Plots, Random, Distributions

# ─────────────────────────────────────────────────────────────────────────────
# 0. Synthetic data with known parameters (ground truth test)
# ─────────────────────────────────────────────────────────────────────────────

println("=== Stochastic Process Calibration Study ===\n")

rng = MersenneTwister(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. GBM Calibration
# ─────────────────────────────────────────────────────────────────────────────

println("--- GBM ---")

true_gbm = GBM(0.10, 0.20, 100.0)   # 10% return, 20% vol
gbm_paths = simulate(true_gbm, 1.0, 252, 500; rng=rng)

# Fit to one path
sample_path = gbm_paths[1, :]
fitted_gbm  = fit(GBM, sample_path, 1.0/252)

println("True params:   μ=$(true_gbm.mu), σ=$(true_gbm.sigma)")
println("Fitted params: μ=$(round(fitted_gbm.mu,digits=4)), σ=$(round(fitted_gbm.sigma,digits=4))")

# Simulate MC under fitted params
mc_paths = simulate(fitted_gbm, 1.0, 252, 1000; rng=rng)
td = terminal_distribution(mc_paths)
println("Terminal distribution: mean=$(round(td.mean,digits=2)), std=$(round(td.std,digits=2))")
println("VaR 5%: $(round(td.var_5pct,digits=2)), ES 5%: $(round(td.es_5pct,digits=2))")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GARCH Calibration
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- GARCH(1,1) ---")

true_garch = GARCH(0.00001, [0.08], [0.90])
garch_paths = simulate(true_garch, 2000, 1; rng=rng)
sample_rets = vec(garch_paths)

fitted_garch = fit(sample_rets, 1, 1)

println("True params:   ω=$(true_garch.omega), α=$(true_garch.alpha[1]), β=$(true_garch.beta[1])")
println("Fitted params: ω=$(round(fitted_garch.omega,digits=7)), α=$(round(fitted_garch.alpha[1],digits=4)), β=$(round(fitted_garch.beta[1],digits=4))")
println("True persistence:   $(sum(true_garch.alpha) + sum(true_garch.beta))")
println("Fitted persistence: $(round(sum(fitted_garch.alpha) + sum(fitted_garch.beta), digits=4))")

# Volatility forecast
h_fcst = forecast(fitted_garch, sample_rets, 20)
println("20-step vol forecast: $(round(sqrt(h_fcst[20])*100*sqrt(252), digits=2))% ann.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Heston Model
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Heston Stochastic Volatility ---")

heston = Heston(100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.02)

# Check Feller condition: 2κθ vs σ²
feller_lhs = 2 * heston.kappa * heston.theta
feller_rhs = heston.sigma^2
println("Feller condition: 2κθ=$(feller_lhs) > σ²=$(feller_rhs): $(feller_lhs > feller_rhs)")

# Simulate
s_paths, v_paths = simulate(heston, 1.0, 252, 500; rng=rng)
println("Mean terminal spot: $(round(mean(s_paths[:,end]),digits=2))")
println("Mean terminal vol:  $(round(mean(sqrt.(v_paths[:,end]))*100,digits=2))%")

# Realised vol (annualised) from paths
realised_vols = [std(diff(log.(s_paths[i,:]))) * sqrt(252) for i in 1:500]
println("Realised vol distribution: mean=$(round(mean(realised_vols)*100,digits=2))%, std=$(round(std(realised_vols)*100,digits=2))%")

# Option pricing
K_list = [90.0, 95.0, 100.0, 105.0, 110.0]
println("\nEuropean call prices (T=1yr):")
for K in K_list
    call_px = price_option(heston, K, 1.0, true)
    intrinsic = max(heston.S0 - K, 0.0) * exp(heston.r)
    @printf("  K=%5.0f  Call=%6.3f  Intrinsic=%6.3f\n", K, call_px, intrinsic)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Ornstein-Uhlenbeck
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Ornstein-Uhlenbeck (Mean Reversion) ---")

true_ou = OrnsteinUhlenbeck(0.0, 5.0, 0.02)   # fast reversion
ou_paths = simulate(true_ou, 1.0, 252, 200; rng=rng, X0=0.10)

# Fit from one path
one_path   = ou_paths[1, :]
fitted_ou  = fit(OrnsteinUhlenbeck, one_path, 1.0/252)

println("True params:   κ=$(true_ou.kappa), μ=$(true_ou.mu), σ=$(true_ou.sigma)")
println("Fitted params: κ=$(round(fitted_ou.kappa,digits=3)), μ=$(round(fitted_ou.mu,digits=4)), σ=$(round(fitted_ou.sigma,digits=4))")
println("True half-life:   $(round(half_life(true_ou)*252,digits=1)) bars")
println("Fitted half-life: $(round(half_life(fitted_ou)*252,digits=1)) bars")

# Stat vol
println("True stat. vol:   $(round(stationary_std(true_ou)*100,digits=3))%")
println("Fitted stat. vol: $(round(stationary_std(fitted_ou)*100,digits=3))%")

# Entry/exit bands
lo_entry, lo_exit, hi_entry, hi_exit = ou_entry_exit_bands(fitted_ou, 1.5)
println("Trading bands: long_entry=$(round(lo_entry,digits=4)), short_entry=$(round(hi_entry,digits=4))")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Merton Jump Diffusion
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Merton Jump Diffusion ---")

true_mjd = MertonJumpDiffusion(0.05, 0.15, 3.0, -0.05, 0.10)
mjd_paths = simulate(true_mjd, 1.0, 252, 1000; rng=rng)

# Compute returns from paths
mjd_rets  = vec(diff(log.(mjd_paths), dims=2))

fitted_mjd = fit(MertonJumpDiffusion, mjd_rets, 1.0/252)
println("True params:   λ=$(true_mjd.lambda), μⱼ=$(true_mjd.mu_j), σⱼ=$(true_mjd.sigma_j)")
println("Fitted params: λ=$(round(fitted_mjd.lambda,digits=3)), μⱼ=$(round(fitted_mjd.mu_j,digits=4)), σⱼ=$(round(fitted_mjd.sigma_j,digits=4))")

# Excess kurtosis (jumps → fat tails)
jb = jarque_bera_test(mjd_rets)
println("Returns: skew=$(round(jb.skewness,digits=3)), excess kurtosis=$(round(jb.excess_kurtosis,digits=3))")

td_mjd = terminal_distribution(mjd_paths)
println("Terminal VaR 5%: $(round(td_mjd.var_5pct,digits=3)), ES 5%: $(round(td_mjd.es_5pct,digits=3))")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Hawkes Process
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Hawkes Self-Exciting Process ---")

true_hawkes = Hawkes(0.5, 0.8, 1.5)   # α < β → stationary
T_sim       = 100.0

events = simulate(true_hawkes, T_sim; rng=rng)
println("True params:  μ=$(true_hawkes.mu), α=$(true_hawkes.alpha), β=$(true_hawkes.beta)")
println("Simulated events: $(length(events)) in T=$(T_sim)")
println("True branching ratio: $(round(branching_ratio(true_hawkes),digits=3))")
println("True mean intensity: $(round(mean_intensity(true_hawkes),digits=3))")
println("Empirical rate: $(round(length(events)/T_sim,digits=3))")

fitted_hawkes = fit(Hawkes, events, T_sim)
println("Fitted params: μ=$(round(fitted_hawkes.mu,digits=4)), α=$(round(fitted_hawkes.alpha,digits=4)), β=$(round(fitted_hawkes.beta,digits=4))")
println("Fitted branching ratio: $(round(branching_ratio(fitted_hawkes),digits=3))")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Model comparison: Heston vs GBM via Sharpe degradation
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Model Comparison ---")

# Under GBM: perfect diversification
gbm_mc    = simulate(GBM(0.1, 0.2, 100.0), 1.0, 252, 5000; rng=rng)
heston_mc = simulate(heston, 1.0, 252, 5000; rng=rng)[1]

gbm_rets_mc    = [log(gbm_mc[i, end] / gbm_mc[i, 1]) for i in 1:5000]
heston_rets_mc = [log(heston_mc[i, end] / heston_mc[i, 1]) for i in 1:5000]

println("GBM terminal:    mean=$(round(mean(gbm_rets_mc)*100,digits=2))%, skew=$(round(cumulant(gbm_rets_mc,3),digits=3))")
println("Heston terminal: mean=$(round(mean(heston_rets_mc)*100,digits=2))%, skew=$(round(cumulant(heston_rets_mc,3),digits=3))")

function cumulant(x, k)
    n = length(x); m = mean(x); s = std(x)
    return mean(((x .- m) ./ s).^k)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

# GBM MC fan chart
p_gbm = plot_mc_paths(gbm_paths[1:200, :]; title="GBM Monte Carlo (200 paths)")
savefig(p_gbm, joinpath(@__DIR__, "04_gbm_mc.png"))

# Heston MC fan chart
p_heston = plot_mc_paths(s_paths[1:200, :]; title="Heston Stochastic Vol MC")
savefig(p_heston, joinpath(@__DIR__, "04_heston_mc.png"))

# OU path + bands
ou_plot_path = ou_paths[1, :]
p_ou = plot(
    ou_plot_path,
    title="OU Process with Trading Bands",
    label="OU path",
    linewidth=1.5,
    color=:steelblue
)
hline!(p_ou, [fitted_ou.mu]; color=:black, linewidth=1.5, linestyle=:dash, label="Mean")
hline!(p_ou, [lo_entry, hi_entry]; color=:green, linewidth=1.5, linestyle=:dot, label="Entry bands")
savefig(p_ou, joinpath(@__DIR__, "04_ou_process.png"))

# GARCH volatility
fitted_vol = sqrt.(garch_variance_series(fitted_garch, sample_rets)) .* sqrt(252) .* 100
p_garch = plot(
    fitted_vol,
    title="GARCH(1,1) Conditional Volatility",
    ylabel="Ann. Vol (%)",
    linewidth=1.5, color=:purple, label="GARCH vol"
)
hline!(p_garch, [std(sample_rets)*sqrt(252)*100];
       color=:red, linestyle=:dash, label="Unconditional vol")
savefig(p_garch, joinpath(@__DIR__, "04_garch_vol.png"))

# Returns distribution with fat tails
p_dist = plot_returns_distribution(mjd_rets[1:2000];
                                    title="MJD Returns Distribution (fat tails)")
savefig(p_dist, joinpath(@__DIR__, "04_mjd_returns.png"))

println("\n04_stochastic_process_calibration.jl complete.")
