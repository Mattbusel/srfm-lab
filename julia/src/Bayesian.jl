"""
Bayesian — Bayesian inference and probabilistic models using Turing.jl.

Implements: strategy evaluation, Markov regime switching, BH CF estimation,
GP-based Bayesian optimisation, and MCMC utilities.
"""
module Bayesian

using Turing
using Distributions
using LinearAlgebra
using Statistics
using Random
using MCMCChains

export strategy_returns_model, markov_switching, bh_cf_model
export run_mcmc, posterior_predictive, compute_bayes_factor
export gp_surrogate_model, thompson_sampling_step
export bayesian_sharpe_estimate, regime_posterior

# ─────────────────────────────────────────────────────────────────────────────
# 1. Strategy Returns Model (Student-t with unknown DOF)
# ─────────────────────────────────────────────────────────────────────────────

"""
    strategy_returns_model(returns)

Bayesian model for strategy returns assuming Student-t likelihood.
Priors chosen to be weakly informative for daily returns.

Hyperpriors:
  ν  ~ Exponential(30)          — degrees of freedom (fat tails)
  μ  ~ Normal(0, 0.01)          — mean daily return
  σ  ~ truncated Normal(0.02)   — daily volatility
"""
@model function strategy_returns_model(returns)
    # Priors
    nu    ~ Exponential(30.0)
    mu    ~ Normal(0.0, 0.01)
    sigma ~ truncated(Normal(0.0, 0.02); lower=1e-5)

    nu_clamped = max(nu, 2.01)   # ensure finite variance

    # Likelihood
    for r in returns
        r ~ LocationScale(mu, sigma, TDist(nu_clamped))
    end
end

"""
    bayesian_sharpe_estimate(returns; n_samples=2000, n_chains=4) → NamedTuple

Full posterior Sharpe distribution.
Returns posterior samples, credible interval, Bayesian p-value.
"""
function bayesian_sharpe_estimate(returns::Vector{Float64};
                                    n_samples::Int=2000,
                                    n_chains::Int=4)::NamedTuple

    model  = strategy_returns_model(returns)
    chains = run_mcmc(model, nothing; n_samples=n_samples, n_chains=n_chains)

    # Posterior Sharpe
    mu_samples    = vec(Array(chains[:mu]))
    sigma_samples = vec(Array(chains[:sigma]))
    nu_samples    = vec(Array(chains[:nu]))

    sharpe_samples = mu_samples ./ max.(sigma_samples, 1e-8) .* sqrt(252)

    # Bayesian p-value: P(Sharpe > 0 | data)
    prob_positive = mean(sharpe_samples .> 0)

    ci_lo = quantile(sharpe_samples, 0.025)
    ci_hi = quantile(sharpe_samples, 0.975)

    return (
        chains          = chains,
        sharpe_samples  = sharpe_samples,
        mean_sharpe     = mean(sharpe_samples),
        median_sharpe   = median(sharpe_samples),
        std_sharpe      = std(sharpe_samples),
        ci_95           = (ci_lo, ci_hi),
        prob_positive   = prob_positive,
        mean_mu         = mean(mu_samples),
        mean_sigma      = mean(sigma_samples),
        mean_nu         = mean(nu_samples),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Bayesian Markov Regime Switching
# ─────────────────────────────────────────────────────────────────────────────

"""
    markov_switching(returns, K=2)

Bayesian K-state Gaussian Markov switching model.

State-dependent emissions: r_t | s_t=k ~ Normal(μ_k, σ_k)
Transition: P(s_t = j | s_{t-1} = i) = A_{ij}  (Dirichlet prior per row)

Uses the forward algorithm for likelihood evaluation.
"""
@model function markov_switching(returns, K::Int=2)
    n = length(returns)

    # Transition matrix (one Dirichlet per row)
    A = Vector{Vector{Float64}}(undef, K)
    for k in 1:K
        A[k] ~ Dirichlet(fill(2.0, K))
    end

    # State-dependent parameters
    mu    = Vector{Float64}(undef, K)
    sigma = Vector{Float64}(undef, K)

    for k in 1:K
        mu[k]    ~ Normal(0.0, 0.02)
        sigma[k] ~ truncated(Normal(0.01, 0.01); lower=1e-5)
    end

    # Initial state distribution
    pi0 ~ Dirichlet(fill(1.0, K))

    # Forward algorithm: compute log-likelihood
    log_alpha = fill(-Inf, K)
    for k in 1:K
        log_alpha[k] = log(pi0[k] + 1e-12) +
                       logpdf(Normal(mu[k], sigma[k]), returns[1])
    end

    for t in 2:n
        log_alpha_new = fill(-Inf, K)
        for j in 1:K
            vals = [log_alpha[i] + log(A[i][j] + 1e-12) for i in 1:K]
            log_alpha_new[j] = _logsumexp(vals) +
                               logpdf(Normal(mu[j], sigma[j]), returns[t])
        end
        log_alpha = log_alpha_new
    end

    Turing.@addlogprob! _logsumexp(log_alpha)
end

function _logsumexp(xs::Vector{Float64})::Float64
    m = maximum(xs)
    isinf(m) && return -Inf
    return m + log(sum(exp(x - m) for x in xs))
end

"""
    regime_posterior(chains, returns, K) → Matrix{Float64}

Viterbi decoding (MAP state sequence) from posterior parameter samples.
Returns (n × K) matrix of smoothed state probabilities.
"""
function regime_posterior(chains::Chains, returns::Vector{Float64},
                           K::Int=2)::Matrix{Float64}
    n = length(returns)
    probs = zeros(n, K)

    # Use posterior mean parameters
    mu_post    = [mean(vec(Array(chains[Symbol("mu[$k]")]))) for k in 1:K]
    sigma_post = [mean(vec(Array(chains[Symbol("sigma[$k]")]))) for k in 1:K]

    # Simple assignment: each bar to most likely state
    for t in 1:n
        liks = [logpdf(Normal(mu_post[k], sigma_post[k]), returns[t]) for k in 1:K]
        probs[t, :] = softmax(liks)
    end

    return probs
end

function softmax(x::Vector{Float64})::Vector{Float64}
    e = exp.(x .- maximum(x))
    return e ./ sum(e)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Bayesian BH CF Estimation
# ─────────────────────────────────────────────────────────────────────────────

"""
    bh_cf_model(prices, n_timelike_obs)

Bayesian model for estimating the optimal critical frequency (CF) parameter.

The generative story:
  1. CF ~ truncated Normal(0.005, 0.003)  — weak prior on CF
  2. For each bar: β = |Δ log price|
  3. Observation: timelike_i ~ Bernoulli(p_i) where:
       p_i = P(β < CF) ≈ logistic(scale * (CF - β))

This lets MCMC infer the posterior distribution of CF given the observed
pattern of timelike/spacelike bars.
"""
@model function bh_cf_model(prices, n_timelike_obs)
    cf ~ truncated(Normal(0.005, 0.003); lower=0.0005, upper=0.05)

    betas = abs.(diff(log.(prices)))
    n     = length(betas)
    @assert length(n_timelike_obs) == n

    # Likelihood: each observation is Bernoulli
    # p = sigmoid(10 * (cf - beta))  — soft classification
    for i in 1:n
        p = logistic(10.0 * (cf - betas[i]))
        n_timelike_obs[i] ~ Bernoulli(p)
    end
end

function logistic(x::Float64)::Float64
    return 1.0 / (1.0 + exp(-x))
end

"""
    estimate_cf_posterior(prices; n_samples=2000) → NamedTuple

Estimate CF posterior from price series.
"""
function estimate_cf_posterior(prices::Vector{Float64};
                                 n_samples::Int=2000,
                                 n_chains::Int=2)::NamedTuple

    betas     = abs.(diff(log.(prices)))
    tl_obs    = betas .< median(betas)   # initial classification

    model  = bh_cf_model(prices, Int.(tl_obs))
    chains = run_mcmc(model, nothing; n_samples=n_samples, n_chains=n_chains)

    cf_samples = vec(Array(chains[:cf]))

    return (
        chains      = chains,
        cf_samples  = cf_samples,
        cf_mean     = mean(cf_samples),
        cf_median   = median(cf_samples),
        cf_std      = std(cf_samples),
        cf_ci_95    = (quantile(cf_samples, 0.025), quantile(cf_samples, 0.975)),
        cf_map      = cf_samples[argmax(pdf.(Normal(mean(cf_samples),
                                                     std(cf_samples)), cf_samples))],
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. GP Surrogate Model (Turing-based)
# ─────────────────────────────────────────────────────────────────────────────

"""
    gp_surrogate_model(X_obs, y_obs)

Gaussian Process regression model for Bayesian optimization surrogate.

Kernel: squared-exponential (RBF)
Hyperpriors:
  length_scale ~ truncated Normal(1, 1)  — per dimension
  signal_std   ~ truncated Normal(1, 1)
  noise_std    ~ truncated Normal(0.1, 0.05)
"""
@model function gp_surrogate_model(X_obs, y_obs)
    n, d = size(X_obs)

    # Hyperprior on kernel hyperparams
    ls  ~ filldist(truncated(Normal(1.0, 1.0); lower=0.01), d)
    eta ~ truncated(Normal(1.0, 0.5); lower=0.01)
    s   ~ truncated(Normal(0.1, 0.1); lower=1e-4)

    # Build covariance matrix
    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        d2 = sum(((X_obs[i, k] - X_obs[j, k]) / max(ls[k], 1e-4))^2 for k in 1:d)
        K[i, j] = eta^2 * exp(-0.5 * d2)
    end
    K_noise = K + (s^2 + 1e-6) * I(n)

    # Likelihood
    y_obs ~ MvNormal(zeros(n), K_noise)
end

"""
    thompson_sampling_step(X_obs, y_obs, bounds, n_cand) → Vector{Float64}

Thompson Sampling acquisition: sample a GP posterior, return argmax.
"""
function thompson_sampling_step(X_obs::Matrix{Float64},
                                  y_obs::Vector{Float64},
                                  bounds::Matrix{Float64},
                                  n_cand::Int=1000;
                                  rng::AbstractRNG=Random.default_rng())::Vector{Float64}

    n, d = size(X_obs)
    lo   = bounds[:, 1]; hi = bounds[:, 2]

    # Generate candidate points
    X_cand = [lo[j] + rand(rng) * (hi[j] - lo[j]) for _ in 1:n_cand, j in 1:d]

    # GP posterior prediction (using posterior mean parameters)
    ls  = fill(0.5, d)
    eta = 1.0
    s   = 0.1

    function se_kernel(x1, x2)
        d2 = sum(((x1[k] - x2[k]) / max(ls[k], 1e-4))^2 for k in 1:d)
        return eta^2 * exp(-0.5 * d2)
    end

    # Build K (training-training)
    K = [se_kernel(X_obs[i, :], X_obs[j, :]) for i in 1:n, j in 1:n]
    K += (s^2 + 1e-6) * I(n)

    # Cholesky
    K_chol = cholesky(K + 1e-6 * I(n))
    alpha  = K_chol \ y_obs

    # Posterior mean + draw Thompson sample
    best_val = -Inf
    best_x   = X_cand[1, :]

    for c in 1:n_cand
        x_c  = X_cand[c, :]
        k_star = [se_kernel(X_obs[i, :], x_c) for i in 1:n]

        mu_c   = dot(k_star, alpha)
        v_c    = K_chol.L \ k_star
        sig_c  = sqrt(max(eta^2 - dot(v_c, v_c), 1e-8))

        # Thompson: sample from posterior at this point
        sample = mu_c + sig_c * randn(rng)

        if sample > best_val
            best_val = sample
            best_x   = x_c
        end
    end

    return best_x
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. MCMC Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_mcmc(model, data; n_samples=2000, n_chains=4, sampler=NUTS()) → Chains

Run MCMC using Turing.jl with the specified sampler.
data can be nothing if already incorporated in the model.
"""
function run_mcmc(model::Turing.Model, data;
                   n_samples::Int=2000,
                   n_chains::Int=4,
                   warmup::Int=1000,
                   sampler=NUTS(warmup, 0.65),
                   progress::Bool=false)::Chains

    chains = sample(model, sampler, MCMCThreads(), n_samples, n_chains;
                    progress=progress, discard_initial=warmup)
    return chains
end

"""
    posterior_predictive(chains, param_name, n_samples) → Vector{Float64}

Draw samples from the posterior predictive distribution for a named parameter.
"""
function posterior_predictive(chains::Chains, param_name::Symbol,
                               n_samples::Int=1000;
                               rng::AbstractRNG=Random.default_rng())::Vector{Float64}
    param_samples = vec(Array(chains[param_name]))
    n_avail       = length(param_samples)
    if n_samples <= n_avail
        idx = sample(rng, 1:n_avail, n_samples; replace=false)
        return param_samples[idx]
    else
        return param_samples[sample(rng, 1:n_avail, n_samples; replace=true)]
    end
end

"""
    compute_bayes_factor(model_1, model_2, data; n_samples=10000) → Float64

Approximate Bayes factor using Harmonic Mean Estimator (Savage-Dickey for nested models).
BF > 1 favours model_1.

NOTE: For production use, prefer bridge sampling or thermodynamic integration.
"""
function compute_bayes_factor(model_1::Turing.Model, model_2::Turing.Model, data;
                               n_samples::Int=10000)::Float64

    # Run chains for both models
    chains_1 = sample(model_1, NUTS(500, 0.65), n_samples; progress=false)
    chains_2 = sample(model_2, NUTS(500, 0.65), n_samples; progress=false)

    # Harmonic mean log-likelihood estimator (Newton & Raftery 1994)
    # This is known to be numerically unstable — acceptable for approximate use
    function harmonic_mean_ll(chains, model)
        lls = pointwise_loglikelihoods(model, chains)
        if isempty(lls)
            return -Inf
        end
        total_ll = [sum(values(lls[i])) for i in 1:length(lls)]
        # HME: -log(mean(exp(-ll)))
        max_ll = maximum(total_ll)
        return max_ll + log(length(total_ll)) -
               log(sum(exp(max_ll - ll) for ll in total_ll))
    end

    log_ml_1 = harmonic_mean_ll(chains_1, model_1)
    log_ml_2 = harmonic_mean_ll(chains_2, model_2)

    return exp(log_ml_1 - log_ml_2)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Model Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

"""
    gelman_rubin(chains) → Dict{Symbol, Float64}

Compute Gelman-Rubin R-hat for each parameter. Values close to 1.0 indicate convergence.
"""
function gelman_rubin(chains::Chains)::Dict{Symbol, Float64}
    param_names = names(chains, :parameters)
    r_hats      = Dict{Symbol, Float64}()

    for p in param_names
        samples = Array(chains[p])   # n_samples × n_chains
        n, m    = size(samples)

        chain_means = mean(samples, dims=1)
        chain_vars  = var(samples, dims=1)

        grand_mean  = mean(chain_means)
        B           = n * var(vec(chain_means))   # between-chain variance
        W           = mean(vec(chain_vars))        # within-chain variance

        var_plus    = (n - 1) / n * W + B / n
        r_hats[p]   = sqrt(max(var_plus / max(W, 1e-12), 1.0))
    end
    return r_hats
end

"""
    effective_sample_size(chains) → Dict{Symbol, Float64}

Estimate effective sample size per parameter via autocorrelation.
"""
function effective_sample_size(chains::Chains)::Dict{Symbol, Float64}
    param_names = names(chains, :parameters)
    ess_dict    = Dict{Symbol, Float64}()

    for p in param_names
        samples = vec(Array(chains[p]))
        n       = length(samples)

        # Autocorrelation-based ESS
        acf_sum = 0.0
        for k in 1:min(100, n-1)
            ρ_k = cor(samples[1:n-k], samples[k+1:n])
            abs(ρ_k) < 0.05 && break
            acf_sum += ρ_k
        end
        ess_dict[p] = n / max(1 + 2 * acf_sum, 1.0)
    end
    return ess_dict
end

"""
    mcmc_summary(chains) → DataFrame

Full MCMC summary table: mean, std, 2.5%, 97.5%, R-hat, ESS.
"""
function mcmc_summary(chains::Chains)

    param_names = names(chains, :parameters)
    r_hats      = gelman_rubin(chains)
    ess_dict    = effective_sample_size(chains)

    rows = NamedTuple[]
    for p in param_names
        samples = vec(Array(chains[p]))
        push!(rows, (
            parameter = string(p),
            mean      = mean(samples),
            std       = std(samples),
            q025      = quantile(samples, 0.025),
            q975      = quantile(samples, 0.975),
            r_hat     = get(r_hats, p, NaN),
            ess       = get(ess_dict, p, NaN),
        ))
    end

    using DataFrames
    return DataFrames.DataFrame(rows)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Bayesian Portfolio Allocation Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    bayesian_portfolio_model(returns_matrix)

Bayesian multivariate model for portfolio of n assets.
Estimates posterior over expected returns (μ) and covariance (Σ).

Returns:
  μ ~ MvNormal(0, 0.01)
  Σ ~ InverseWishart(n+2, I)
  returns_t ~ MvNormal(μ, Σ)
"""
@model function bayesian_portfolio_model(returns_matrix)
    n_bars, n_assets = size(returns_matrix)

    # Priors
    mu    ~ MvNormal(zeros(n_assets), 0.01 * I(n_assets))
    Sigma ~ InverseWishart(n_assets + 2, Matrix{Float64}(I(n_assets)))

    # Likelihood
    for t in 1:n_bars
        returns_matrix[t, :] ~ MvNormal(mu, Symmetric(Sigma))
    end
end

"""
    posterior_optimal_weights(returns_matrix; n_samples=1000) → NamedTuple

Bayesian tangency portfolio: sample from posterior, compute weights per sample,
return distribution of optimal weights.
"""
function posterior_optimal_weights(returns_matrix::Matrix{Float64};
                                    n_samples::Int=1000,
                                    n_chains::Int=2,
                                    rf::Float64=0.0)::NamedTuple

    n_assets = size(returns_matrix, 2)
    model    = bayesian_portfolio_model(returns_matrix)
    chains   = run_mcmc(model, nothing; n_samples=n_samples, n_chains=n_chains)

    # Extract mu samples
    mu_samples    = hcat([vec(Array(chains[Symbol("mu[$k]")])) for k in 1:n_assets]...)
    n_draws       = size(mu_samples, 1)

    # Weight samples
    weight_samples = Matrix{Float64}(undef, n_draws, n_assets)
    sample_sharpe  = Vector{Float64}(undef, n_draws)

    # Use posterior mean covariance
    # (Sigma MCMC samples for MW are expensive — use point estimate)
    Sigma_hat = cov(returns_matrix) + 1e-6 * I(n_assets)

    for i in 1:n_draws
        mu_i   = mu_samples[i, :]
        excess = mu_i .- rf
        w      = Sigma_hat \ excess
        if any(!isfinite, w) || sum(abs.(w)) < 1e-8
            w = ones(n_assets) / n_assets
        else
            w = abs.(w) ./ sum(abs.(w))   # long-only projection
        end
        weight_samples[i, :] = w
        port_ret = dot(w, mu_i) - rf
        port_vol = sqrt(max(dot(w, Sigma_hat * w), 1e-12))
        sample_sharpe[i] = port_ret / port_vol * sqrt(252)
    end

    return (
        weight_samples   = weight_samples,
        mean_weights     = vec(mean(weight_samples, dims=1)),
        std_weights      = vec(std(weight_samples, dims=1)),
        weight_ci_lo     = vec(mapslices(x -> quantile(x, 0.025), weight_samples, dims=1)),
        weight_ci_hi     = vec(mapslices(x -> quantile(x, 0.975), weight_samples, dims=1)),
        sharpe_samples   = sample_sharpe,
        mean_sharpe      = mean(sample_sharpe),
        chains           = chains,
    )
end

end # module Bayesian
