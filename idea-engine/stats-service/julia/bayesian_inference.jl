# =============================================================================
# bayesian_inference.jl — Full Bayesian Inference for Strategy Parameters
# =============================================================================
# Provides:
#   - MetropolisHastings   MCMC sampler (from scratch)
#   - WinRatePosterior     Beta-Binomial conjugate update
#   - PnLStudentT          Student-t model for heavy-tailed P&L
#   - PosteriorPredictive  Simulate next N trades from posterior
#   - BayesFactor          Model comparison (hypothesis A vs B)
#   - CredibleInterval     Equal-tailed credible intervals
#   - run_full_inference   Top-level driver: writes JSON
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, Distributions, JSON3
# =============================================================================

module BayesianInference

using Statistics
using LinearAlgebra
using JSON3

# ── Distributions (conjugate helpers) ─────────────────────────────────────────
# We implement only what we need so this module has no heavy dependencies.

# Log-density of Beta(α, β) at x ∈ (0,1)
function _logpdf_beta(x::Float64, α::Float64, β::Float64)::Float64
    (x <= 0.0 || x >= 1.0) && return -Inf
    (α - 1.0) * log(x) + (β - 1.0) * log(1.0 - x) -
        (lgamma(α) + lgamma(β) - lgamma(α + β))
end

# Log-density of Student-t(ν, μ, σ) at x
function _logpdf_studentt(x::Float64, ν::Float64, μ::Float64, σ::Float64)::Float64
    σ <= 0.0 && return -Inf
    ν <= 0.0 && return -Inf
    z = (x - μ) / σ
    lgamma((ν + 1.0) / 2.0) - lgamma(ν / 2.0) -
        0.5 * log(ν * π) - log(σ) -
        ((ν + 1.0) / 2.0) * log(1.0 + z^2 / ν)
end

# Log-density of Normal(μ, σ) at x
function _logpdf_normal(x::Float64, μ::Float64, σ::Float64)::Float64
    σ <= 0.0 && return -Inf
    -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)^2
end

# ── Simple LCG RNG (deterministic, no seed dependency on external libs) ────────
mutable struct LCG
    state::UInt64
end
LCG(seed::Int) = LCG(UInt64(abs(seed)))

function _next!(rng::LCG)::Float64
    rng.state = rng.state * 6364136223846793005 + 1442695040888963407
    Float64(rng.state >> 11) / Float64(2^53)
end

function _randn!(rng::LCG)::Float64
    # Box-Muller
    u1 = max(_next!(rng), 1e-15)
    u2 = _next!(rng)
    sqrt(-2.0 * log(u1)) * cos(2π * u2)
end

# ── Credible Interval ─────────────────────────────────────────────────────────

"""
Compute equal-tailed credible interval at `level` from MCMC samples.
Returns (lower, upper, mean, median, std).
"""
function CredibleInterval(samples::AbstractVector{<:Real}; level::Float64=0.95)
    α = (1.0 - level) / 2.0
    s = sort(Float64.(samples))
    n = length(s)
    lo_idx = max(1, round(Int, α * n))
    hi_idx = min(n, round(Int, (1.0 - α) * n))
    (
        lower  = s[lo_idx],
        upper  = s[hi_idx],
        mean   = mean(s),
        median = s[n ÷ 2 + 1],
        std    = std(s),
        level  = level
    )
end

# ── Metropolis-Hastings MCMC ──────────────────────────────────────────────────

"""
Generic Metropolis-Hastings MCMC sampler.

# Arguments
- `log_posterior` : function θ::Vector{Float64} → log p(θ | data)  (unnormalised)
- `θ_init`        : initial parameter vector
- `n_samples`     : total iterations (including burn-in)
- `burn_in`       : number of initial samples to discard
- `proposal_std`  : per-parameter proposal standard deviation (or scalar)
- `seed`          : RNG seed for reproducibility

# Returns
NamedTuple: (samples, acceptance_rate, log_posterior_trace, param_names)
"""
function MetropolisHastings(
    log_posterior::Function,
    θ_init::Vector{Float64};
    n_samples::Int = 5_000,
    burn_in::Int   = 1_000,
    proposal_std::Union{Float64, Vector{Float64}} = 0.05,
    seed::Int      = 1337,
    param_names::Vector{String} = String[]
)
    n_params   = length(θ_init)
    prop_σ     = proposal_std isa Float64 ? fill(proposal_std, n_params) : proposal_std
    length(prop_σ) == n_params || error("proposal_std length must match θ_init")

    rng        = LCG(seed)
    chain      = Matrix{Float64}(undef, n_samples, n_params)
    lp_trace   = Vector{Float64}(undef, n_samples)
    θ_cur      = copy(θ_init)
    lp_cur     = log_posterior(θ_cur)
    n_accepted = 0

    for i in 1:n_samples
        # Random-walk proposal
        θ_prop = θ_cur .+ [_randn!(rng) * prop_σ[k] for k in 1:n_params]
        lp_prop = log_posterior(θ_prop)

        # Metropolis acceptance
        log_α = lp_prop - lp_cur
        if log(max(_next!(rng), 1e-300)) < log_α
            θ_cur  = θ_prop
            lp_cur = lp_prop
            n_accepted += 1
        end

        chain[i, :]  = θ_cur
        lp_trace[i]  = lp_cur
    end

    keep = (burn_in + 1):n_samples
    names = isempty(param_names) ? ["θ_$i" for i in 1:n_params] : param_names

    (
        samples          = chain[keep, :],
        acceptance_rate  = n_accepted / n_samples,
        log_posterior_trace = lp_trace[keep],
        param_names      = names,
        n_samples        = length(keep),
        burn_in          = burn_in
    )
end

# ── Win Rate: Beta-Binomial Conjugate Update ──────────────────────────────────

"""
Bayesian win-rate estimation with Beta(α, β) conjugate prior.

Prior: win_rate ~ Beta(α₀, β₀)
Likelihood: wins ~ Binomial(n_trades, win_rate)
Posterior: win_rate ~ Beta(α₀ + wins, β₀ + losses)

Returns full posterior characterisation including MCMC samples (via
direct Beta sampling using inverse-CDF approximation).
"""
function WinRatePosterior(
    wins::Int, losses::Int;
    α_prior::Float64 = 1.0,
    β_prior::Float64 = 1.0,
    n_mcmc::Int      = 4_000,
    seed::Int        = 42
)
    wins   >= 0 || error("wins must be ≥ 0")
    losses >= 0 || error("losses must be ≥ 0")
    wins + losses > 0 || error("need at least one trade")

    # Conjugate posterior parameters
    α_post = α_prior + wins
    β_post = β_prior + losses
    n_total = wins + losses

    # Posterior mean and mode
    post_mean = α_post / (α_post + β_post)
    post_mode = (α_post - 1.0) / (α_post + β_post - 2.0)  # valid if α,β > 1
    post_var  = (α_post * β_post) / ((α_post + β_post)^2 * (α_post + β_post + 1.0))
    post_std  = sqrt(post_var)

    # Sample from Beta posterior via M-H
    log_post(θ) = _logpdf_beta(θ[1], α_post, β_post)
    mcmc = MetropolisHastings(
        log_post,
        [post_mean];
        n_samples    = n_mcmc + 500,
        burn_in      = 500,
        proposal_std = [post_std * 2.0],
        seed         = seed,
        param_names  = ["win_rate"]
    )

    samples = vec(mcmc.samples[:, 1])
    ci95    = CredibleInterval(samples; level=0.95)
    ci80    = CredibleInterval(samples; level=0.80)

    # Posterior probability that true win_rate > 50%
    prob_above_50 = mean(samples .> 0.5)

    # Likelihood ratio vs flat prior: Bayes factor approximation
    prior_density_at_mean  = exp(_logpdf_beta(post_mean, α_prior, β_prior))
    post_density_at_mean   = exp(_logpdf_beta(post_mean, α_post, β_post))
    savage_dickey_bf       = prior_density_at_mean > 0.0 ?
                                 post_density_at_mean / prior_density_at_mean : NaN

    (
        α_prior        = α_prior,
        β_prior        = β_prior,
        α_posterior    = α_post,
        β_posterior    = β_post,
        posterior_mean = post_mean,
        posterior_mode = post_mode,
        posterior_std  = post_std,
        ci_95          = ci95,
        ci_80          = ci80,
        prob_above_50pct = prob_above_50,
        samples        = samples,
        acceptance_rate = mcmc.acceptance_rate,
        n_trades       = n_total,
        savage_dickey_bf = savage_dickey_bf
    )
end

# ── P&L per Trade: Student-t Model ────────────────────────────────────────────

"""
Fit a Student-t(ν, μ, σ) model to per-trade P&L using MCMC.

Prior:
  μ ~ Normal(0, 10 * σ_data)
  log(σ) ~ Normal(log(σ_data), 1)
  log(ν) ~ Normal(log(5), 0.5)   — weakly informative, allows ν ∈ (1, ∞)

This model captures heavy tails (ν < 30) that arise from occasional large wins/losses.
"""
function PnLStudentT(
    pnl::AbstractVector{<:Real};
    n_mcmc::Int = 5_000,
    seed::Int   = 99
)
    x      = Float64.(pnl)
    n_obs  = length(x)
    n_obs < 5 && error("Need at least 5 P&L observations")

    # Prior scale parameters
    σ_data = std(x)
    μ_data = mean(x)

    # Log-posterior: log p(μ, log_σ, log_ν | data)
    function log_post(θ::Vector{Float64})
        μ      = θ[1]
        log_σ  = θ[2]
        log_ν  = θ[3]
        σ      = exp(log_σ)
        ν      = exp(log_ν)

        # Log-likelihood
        ll = sum(_logpdf_studentt(xi, ν, μ, σ) for xi in x)
        isfinite(ll) || return -Inf

        # Priors (log-scale)
        lp_μ  = _logpdf_normal(μ,    μ_data, 10.0 * σ_data)
        lp_lσ = _logpdf_normal(log_σ, log(σ_data), 1.0)
        lp_lν = _logpdf_normal(log_ν, log(5.0), 0.5)

        ll + lp_μ + lp_lσ + lp_lν
    end

    θ_init = [μ_data, log(σ_data), log(5.0)]
    prop_σ = [σ_data * 0.1, 0.1, 0.1]

    mcmc = MetropolisHastings(
        log_post, θ_init;
        n_samples    = n_mcmc + 1_000,
        burn_in      = 1_000,
        proposal_std = prop_σ,
        seed         = seed,
        param_names  = ["mu", "log_sigma", "log_nu"]
    )

    mu_samp    = mcmc.samples[:, 1]
    sigma_samp = exp.(mcmc.samples[:, 2])
    nu_samp    = exp.(mcmc.samples[:, 3])

    (
        mu     = CredibleInterval(mu_samp),
        sigma  = CredibleInterval(sigma_samp),
        nu     = CredibleInterval(nu_samp),
        mu_samples    = mu_samp,
        sigma_samples = sigma_samp,
        nu_samples    = nu_samp,
        acceptance_rate = mcmc.acceptance_rate,
        n_obs  = n_obs,
        interpretation = begin
            ν_med = median(nu_samp)
            if ν_med < 5.0
                "Very heavy tails (ν≈$(round(ν_med; digits=1))): extreme events likely"
            elseif ν_med < 15.0
                "Moderate heavy tails (ν≈$(round(ν_med; digits=1))): elevated tail risk"
            else
                "Near-Gaussian tails (ν≈$(round(ν_med; digits=1))): tails manageable"
            end
        end
    )
end

# ── Posterior Predictive Distribution ─────────────────────────────────────────

"""
Generate posterior predictive draws for the next `n_future` trades.

Uses the Student-t P&L model posterior and win-rate posterior jointly.
Returns quantiles, expected cumulative P&L, and VaR / CVaR at 95%.
"""
function PosteriorPredictive(
    pnl_model,          # result of PnLStudentT
    winrate_model;      # result of WinRatePosterior
    n_future::Int = 100,
    n_draws::Int  = 2_000,
    seed::Int     = 7
)
    rng        = LCG(seed)
    n_post     = length(pnl_model.mu_samples)

    cum_pnl_matrix = Matrix{Float64}(undef, n_draws, n_future)

    for draw in 1:n_draws
        # Draw one posterior sample for model params
        idx   = 1 + (draw - 1) % n_post
        μ     = pnl_model.mu_samples[idx]
        σ     = pnl_model.sigma_samples[idx]
        ν     = max(pnl_model.nu_samples[idx], 1.01)
        wr    = winrate_model.samples[1 + (draw - 1) % length(winrate_model.samples)]

        # Simulate n_future trades
        pnl_path = Float64[]
        for _ in 1:n_future
            # Win/loss indicator
            win = _next!(rng) < wr

            # Draw from Student-t
            z     = _randn!(rng)
            u     = max(_next!(rng), 1e-10)
            chi2  = -2.0 * log(u)  # χ²(1) approx via -2*log(U)
            # Student-t via normal / sqrt(chi2/ν)
            t_draw = z / sqrt(chi2 / ν)
            trade_pnl = μ + σ * t_draw
            # Sign toward win/loss direction
            if (trade_pnl > 0.0) != win
                trade_pnl = -trade_pnl
            end
            push!(pnl_path, trade_pnl)
        end
        cum_pnl_matrix[draw, :] = cumsum(pnl_path)
    end

    # Quantiles of cumulative P&L at each horizon
    q_lo   = [quantile(cum_pnl_matrix[:, t], 0.05) for t in 1:n_future]
    q_med  = [quantile(cum_pnl_matrix[:, t], 0.50) for t in 1:n_future]
    q_hi   = [quantile(cum_pnl_matrix[:, t], 0.95) for t in 1:n_future]
    q_mean = [mean(cum_pnl_matrix[:, t])            for t in 1:n_future]

    # Final P&L distribution (at trade n_future)
    final_pnl = cum_pnl_matrix[:, n_future]
    var_95    = quantile(final_pnl, 0.05)    # 5th percentile = 95% VaR
    cvar_95   = mean(final_pnl[final_pnl .<= var_95])

    prob_profit = mean(final_pnl .> 0.0)

    (
        n_future          = n_future,
        n_draws           = n_draws,
        quantile_05       = q_lo,
        quantile_50       = q_med,
        quantile_95       = q_hi,
        mean_path         = q_mean,
        final_var_95      = var_95,
        final_cvar_95     = cvar_95,
        prob_profit       = prob_profit,
        expected_final_pnl = mean(final_pnl)
    )
end

# ── Bayes Factor ──────────────────────────────────────────────────────────────

"""
Compute Bayes factor B₁₂ = p(data | M₁) / p(data | M₂) via harmonic mean
estimator from MCMC log-posterior values.

B₁₂ > 10  → strong evidence for M₁
B₁₂ > 3   → moderate evidence for M₁
B₁₂ ≈ 1   → data equally consistent with both models
B₁₂ < 1/3 → moderate evidence for M₂

# Arguments
- `lp_trace_1` : vector of log-posterior values from model 1 MCMC run
- `lp_trace_2` : vector of log-posterior values from model 2 MCMC run

# Returns
NamedTuple: (bayes_factor, log_bf, interpretation, model_probs)
"""
function BayesFactor(
    lp_trace_1::AbstractVector{<:Real},
    lp_trace_2::AbstractVector{<:Real};
    prior_odds::Float64 = 1.0   # p(M1)/p(M2) prior
)
    # Harmonic mean estimator of marginal likelihood
    # log p(data|M) ≈ -log( mean(exp(-lp)) )  [Newton-Raftery estimator]
    function _log_marglik(lp::Vector{Float64})
        # Stabilise via logsumexp
        max_lp = maximum(lp)
        -log(mean(exp.(max_lp .- lp))) - max_lp
    end

    lp1 = Float64.(lp_trace_1)
    lp2 = Float64.(lp_trace_2)

    log_ml1 = _log_marglik(lp1)
    log_ml2 = _log_marglik(lp2)
    log_bf  = log_ml1 - log_ml2 + log(prior_odds)
    bf      = exp(clamp(log_bf, -50.0, 50.0))

    interp = if bf > 100.0
        "Decisive evidence for M1 (BF>100)"
    elseif bf > 10.0
        "Strong evidence for M1 (BF>10)"
    elseif bf > 3.0
        "Moderate evidence for M1 (BF>3)"
    elseif bf > 1.0/3.0
        "Inconclusive (BF ≈ 1)"
    elseif bf > 1.0/10.0
        "Moderate evidence for M2 (BF<1/3)"
    else
        "Strong evidence for M2 (BF<1/10)"
    end

    post_prob_m1 = bf * prior_odds / (1.0 + bf * prior_odds)

    (
        bayes_factor    = bf,
        log_bf          = log_bf,
        log_marglik_m1  = log_ml1,
        log_marglik_m2  = log_ml2,
        interpretation  = interp,
        posterior_prob_m1 = post_prob_m1,
        posterior_prob_m2 = 1.0 - post_prob_m1
    )
end

# ── Top-level driver ──────────────────────────────────────────────────────────

"""
Run full Bayesian inference pipeline from raw trade data.

# Arguments
- `wins`   : number of winning trades
- `losses` : number of losing trades
- `pnl`    : per-trade P&L vector (can include both wins and losses)

Writes `bayesian_inference_results.json` to `\$STATS_OUTPUT_DIR`.
"""
function run_full_inference(
    wins::Int,
    losses::Int,
    pnl::Vector{Float64};
    output_dir::String = get(ENV, "STATS_OUTPUT_DIR",
                             joinpath(@__DIR__, "..", "output")),
    n_mcmc::Int = 4_000
)
    println("[bayesian] Fitting win-rate posterior (Beta-Binomial)...")
    wr = WinRatePosterior(wins, losses; n_mcmc=n_mcmc, seed=1)

    println("[bayesian] Fitting P&L Student-t model via MCMC...")
    pnl_fit = PnLStudentT(pnl; n_mcmc=n_mcmc, seed=2)

    println("[bayesian] Computing posterior predictive (next 100 trades)...")
    pred = PosteriorPredictive(pnl_fit, wr; n_future=100, n_draws=1_000)

    # Compare two win-rate hypotheses via Bayes factor:
    # M1: win_rate ~ Beta(1,1) flat prior  vs  M2: win_rate ~ Beta(0.5, 0.5) Jeffreys
    println("[bayesian] Computing Bayes factor (flat vs Jeffreys prior)...")

    function lp_flat(θ)
        _logpdf_beta(θ[1], 1.0 + wins, 1.0 + losses)
    end
    function lp_jeff(θ)
        _logpdf_beta(θ[1], 0.5 + wins, 0.5 + losses)
    end

    mcmc_flat = MetropolisHastings(lp_flat, [wr.posterior_mean];
        n_samples=3_000, burn_in=500, proposal_std=[0.03], seed=10)
    mcmc_jeff = MetropolisHastings(lp_jeff, [wr.posterior_mean];
        n_samples=3_000, burn_in=500, proposal_std=[0.03], seed=11)

    bf = BayesFactor(mcmc_flat.log_posterior_trace, mcmc_jeff.log_posterior_trace)

    # Build JSON output
    result = Dict(
        "win_rate" => Dict(
            "posterior_mean"     => wr.posterior_mean,
            "posterior_std"      => wr.posterior_std,
            "ci_95_lower"        => wr.ci_95.lower,
            "ci_95_upper"        => wr.ci_95.upper,
            "ci_80_lower"        => wr.ci_80.lower,
            "ci_80_upper"        => wr.ci_80.upper,
            "prob_above_50pct"   => wr.prob_above_50pct,
            "alpha_posterior"    => wr.α_posterior,
            "beta_posterior"     => wr.β_posterior,
            "n_trades"           => wr.n_trades
        ),
        "pnl_model" => Dict(
            "mu_mean"            => pnl_fit.mu.mean,
            "mu_ci_95"           => [pnl_fit.mu.lower, pnl_fit.mu.upper],
            "sigma_mean"         => pnl_fit.sigma.mean,
            "sigma_ci_95"        => [pnl_fit.sigma.lower, pnl_fit.sigma.upper],
            "nu_mean"            => pnl_fit.nu.mean,
            "nu_ci_95"           => [pnl_fit.nu.lower, pnl_fit.nu.upper],
            "interpretation"     => pnl_fit.interpretation,
            "acceptance_rate"    => pnl_fit.acceptance_rate
        ),
        "posterior_predictive" => Dict(
            "n_future"           => pred.n_future,
            "quantile_05"        => pred.quantile_05,
            "quantile_50"        => pred.quantile_50,
            "quantile_95"        => pred.quantile_95,
            "mean_path"          => pred.mean_path,
            "var_95"             => pred.final_var_95,
            "cvar_95"            => pred.final_cvar_95,
            "prob_profit"        => pred.prob_profit,
            "expected_final_pnl" => pred.expected_final_pnl
        ),
        "bayes_factor" => Dict(
            "bf_flat_vs_jeffreys"  => bf.bayes_factor,
            "log_bf"               => bf.log_bf,
            "interpretation"       => bf.interpretation,
            "posterior_prob_flat"  => bf.posterior_prob_m1,
            "posterior_prob_jeff"  => bf.posterior_prob_m2
        )
    )

    mkpath(output_dir)
    out_path = joinpath(output_dir, "bayesian_inference_results.json")
    open(out_path, "w") do io
        write(io, JSON3.write(result))
    end
    println("[bayesian] Results written to $out_path")

    result
end

export MetropolisHastings, WinRatePosterior, PnLStudentT
export PosteriorPredictive, BayesFactor, CredibleInterval
export run_full_inference

end  # module BayesianInference

# ── CLI self-test ─────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    using .BayesianInference
    using Statistics

    println("[bayesian_inference] Running self-test...")

    # Synthetic trade history: 58% win rate, 500 trades
    wins   = 290
    losses = 210

    rng_seed = 42
    # Generate fake P&L: wins ≈ +1.2, losses ≈ -1.0, heavy-tailed
    rng = BayesianInference.LCG(rng_seed)
    pnl = Float64[]
    for i in 1:500
        r = BayesianInference.(_next!)(rng)
        z = BayesianInference.(_randn!)(rng)
        push!(pnl, r < 0.58 ? 1.2 + 0.5*z : -1.0 + 0.3*z)
    end

    result = run_full_inference(wins, losses, pnl; n_mcmc=3_000)

    wr = result["win_rate"]
    println("  Win-rate posterior mean: $(round(wr["posterior_mean"]; digits=4))")
    println("  95% CI: [$(round(wr["ci_95_lower"]; digits=4)), $(round(wr["ci_95_upper"]; digits=4))]")
    println("  P(WR>50%) = $(round(wr["prob_above_50pct"]; digits=4))")

    pm = result["pnl_model"]
    println("  P&L μ: $(round(pm["mu_mean"]; digits=4)), σ: $(round(pm["sigma_mean"]; digits=4))")
    println("  Student-t ν: $(round(pm["nu_mean"]; digits=2)) — $(pm["interpretation"])")

    pp = result["posterior_predictive"]
    println("  Next 100 trades: E[PnL]=$(round(pp["expected_final_pnl"]; digits=2)), P(profit)=$(round(pp["prob_profit"]; digits=3))")
    println("  VaR95: $(round(pp["var_95"]; digits=2)), CVaR95: $(round(pp["cvar_95"]; digits=2))")

    println("[bayesian_inference] Self-test complete.")
end
