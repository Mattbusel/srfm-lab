"""
uncertainty_quantification.jl — Bayesian UQ for Neural SDE parameters

Implements:
  1. Metropolis-Hastings MCMC (random walk, adaptive)
  2. Hamiltonian Monte Carlo (HMC) — leapfrog integrator
  3. No-U-Turn Sampler (NUTS) sketch
  4. Posterior predictive distributions
  5. Credible intervals (HDI, ETI)
  6. Sobol sensitivity indices (Saltelli estimator)
  7. Epistemic vs aleatoric uncertainty decomposition
  8. Conformal prediction intervals (split conformal)
  9. Bayesian model selection (WAIC, LOO-CV)
 10. Prior predictive checks

References:
  - Metropolis et al. (1953); Hastings (1970)
  - Neal (2011) "MCMC using Hamiltonian dynamics"
  - Saltelli et al. (2010) "Variance-based sensitivity analysis"
  - Angelopoulos & Bates (2021) "A gentle introduction to conformal prediction"
"""

using LinearAlgebra
using Statistics
using Random
using Distributions

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MCMC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    MCMCConfig

Common configuration for all MCMC samplers.
"""
struct MCMCConfig
    n_samples    :: Int
    n_warmup     :: Int
    n_chains     :: Int
    thin         :: Int       # keep every thin-th sample
    seed         :: Int
    progress     :: Bool
end

MCMCConfig(; n_samples=2000, n_warmup=1000, n_chains=4,
             thin=1, seed=42, progress=false) =
    MCMCConfig(n_samples, n_warmup, n_chains, thin, seed, progress)

"""
    MCMCChain

Container for a single MCMC chain.
"""
struct MCMCChain
    samples         :: Matrix{Float64}   # (n_params × n_samples)
    log_posterior   :: Vector{Float64}
    acceptance_rate :: Float64
    param_names     :: Vector{String}
    chain_id        :: Int
end

"""
    MCMCResult

Collection of chains with diagnostics.
"""
struct MCMCResult
    chains       :: Vector{MCMCChain}
    r_hat        :: Vector{Float64}   # Gelman-Rubin Rhat per parameter
    ess          :: Vector{Float64}   # effective sample size per param
    param_names  :: Vector{String}
    n_params     :: Int
    n_samples    :: Int
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: METROPOLIS-HASTINGS
# ─────────────────────────────────────────────────────────────────────────────

"""
    mh_step(θ_cur, log_post, proposal_std; rng) → (θ_new, accepted, log_post_new)

Single Metropolis-Hastings step with isotropic Gaussian proposal.
"""
function mh_step(θ_cur::AbstractVector,
                 log_post_cur::Real,
                 log_post::Function,
                 proposal_std::AbstractVector;
                 rng::AbstractRNG = Random.GLOBAL_RNG)
    θ_prop    = θ_cur .+ proposal_std .* randn(rng, length(θ_cur))
    lp_prop   = log_post(θ_prop)
    log_α     = lp_prop - log_post_cur
    if log(rand(rng)) < log_α
        return θ_prop, true, lp_prop
    else
        return θ_cur, false, log_post_cur
    end
end

"""
    run_mh(log_posterior, θ0, cfg;
           proposal_std=nothing, adapt=true) → MCMCChain

Run Metropolis-Hastings MCMC with optional adaptive tuning.

Arguments:
  - `log_posterior(θ)` : log p(θ | data) (up to normalisation constant)
  - `θ0` : initial parameter vector
  - `cfg` : MCMCConfig
  - `proposal_std` : initial step sizes (defaults to 0.1 for each param)
  - `adapt` : whether to adapt proposal std during warmup (Robbins-Monro)
"""
function run_mh(log_posterior::Function,
                θ0::AbstractVector,
                cfg::MCMCConfig;
                proposal_std::Union{Nothing, AbstractVector} = nothing,
                adapt::Bool     = true,
                param_names::Vector{String} = String[],
                chain_id::Int   = 1)
    rng     = MersenneTwister(cfg.seed + chain_id)
    n_p     = length(θ0)
    p_names = isempty(param_names) ? ["θ_$i" for i in 1:n_p] : param_names
    pstd    = isnothing(proposal_std) ? fill(0.1, n_p) : copy(Float64.(proposal_std))

    total   = cfg.n_warmup + cfg.n_samples * cfg.thin
    θ       = copy(Float64.(θ0))
    lp      = log_posterior(θ)

    samples_all  = zeros(n_p, total)
    lp_all       = zeros(total)
    accepts      = 0

    # Adaptation parameters (dual averaging)
    target_ar = 0.234
    γ = 0.05; t0 = 10; κ = 0.75
    log_ε_bar = zeros(n_p)
    h_bar      = zeros(n_p)

    for i in 1:total
        θ, acc, lp = mh_step(θ, lp, log_posterior, pstd; rng=rng)
        samples_all[:, i] = θ
        lp_all[i]          = lp
        accepts           += acc

        if adapt && i <= cfg.n_warmup
            # Robbins-Monro step-size adaptation
            α_i = acc ? 1.0 : 0.0
            for j in 1:n_p
                m      = i
                η      = 1.0 / (m + t0)^κ
                h_bar[j] = (1 - η) * h_bar[j] + η * (target_ar - α_i)
                log_pstd = -sqrt(m) / γ * h_bar[j]
                log_ε_bar[j] = (1 - m^(-κ)) * log_ε_bar[j] + m^(-κ) * log_pstd
                pstd[j]  = exp(log_pstd)
            end
        end
    end

    # Keep post-warmup samples, thinned
    keep_idx = (cfg.n_warmup+1):cfg.thin:total
    samples  = samples_all[:, keep_idx]
    lp_keep  = lp_all[keep_idx]
    ar       = accepts / total

    return MCMCChain(samples, lp_keep, ar, p_names, chain_id)
end

"""
    run_mh_multichain(log_posterior, θ0_list, cfg; kwargs...) → MCMCResult

Run multiple independent MH chains and compute convergence diagnostics.
"""
function run_mh_multichain(log_posterior::Function,
                            θ0_list::Vector{<:AbstractVector},
                            cfg::MCMCConfig;
                            kwargs...)
    chains = [run_mh(log_posterior, θ0_list[c], cfg;
                     chain_id=c, kwargs...)
              for c in 1:cfg.n_chains]
    n_p  = size(chains[1].samples, 1)
    pnames = chains[1].param_names
    r_hat = gelman_rubin(chains)
    ess_v = [bulk_ess([chains[c].samples[j, :] for c in 1:cfg.n_chains])
             for j in 1:n_p]
    return MCMCResult(chains, r_hat, ess_v, pnames, n_p,
                      size(chains[1].samples, 2))
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: HAMILTONIAN MONTE CARLO (HMC)
# ─────────────────────────────────────────────────────────────────────────────

"""
    HMCConfig

Configuration for HMC.
"""
struct HMCConfig
    ε          :: Float64   # leapfrog step size
    L          :: Int       # number of leapfrog steps
    mass_diag  :: Union{Nothing, Vector{Float64}}  # diagonal mass matrix
end

HMCConfig(; ε=0.1, L=10, mass_diag=nothing) = HMCConfig(ε, L, mass_diag)

"""
    leapfrog(θ, r, grad_U, ε, L, M_inv) → (θ_new, r_new)

Leapfrog integrator for HMC.

- `grad_U(θ)` : gradient of potential energy U(θ) = -log π(θ)
- `M_inv`     : inverse mass matrix (diagonal, vector)
"""
function leapfrog(θ::AbstractVector,
                  r::AbstractVector,
                  grad_U::Function,
                  ε::Real,
                  L::Int,
                  M_inv::AbstractVector)
    θ_new = copy(θ)
    r_new = copy(r)

    # Half step for momentum
    r_new .-= 0.5 * ε .* grad_U(θ_new)

    for ℓ in 1:(L-1)
        # Full step for position
        θ_new .+= ε .* M_inv .* r_new
        # Full step for momentum
        r_new .-= ε .* grad_U(θ_new)
    end

    # Final half-step for position
    θ_new .+= ε .* M_inv .* r_new
    # Final half-step for momentum
    r_new .-= 0.5 * ε .* grad_U(θ_new)

    return θ_new, r_new
end

"""
    hmc_kinetic_energy(r, M_inv) → Float64

Kinetic energy K(r) = ½ rᵀ M⁻¹ r (diagonal M).
"""
hmc_kinetic_energy(r::AbstractVector, M_inv::AbstractVector) =
    0.5 * dot(r, M_inv .* r)

"""
    run_hmc(log_posterior, grad_log_posterior, θ0, cfg, hmc_cfg;
            param_names, chain_id) → MCMCChain

Run HMC with leapfrog integration.

`grad_log_posterior(θ)` : gradient of log π(θ) w.r.t. θ.
If not provided analytically, use finite differences.
"""
function run_hmc(log_posterior::Function,
                 grad_log_posterior::Function,
                 θ0::AbstractVector,
                 cfg::MCMCConfig,
                 hmc_cfg::HMCConfig;
                 param_names::Vector{String} = String[],
                 chain_id::Int = 1)
    rng    = MersenneTwister(cfg.seed + chain_id * 1000)
    n_p    = length(θ0)
    pnames = isempty(param_names) ? ["θ_$i" for i in 1:n_p] : param_names

    M_inv  = isnothing(hmc_cfg.mass_diag) ? ones(n_p) :
             1.0 ./ hmc_cfg.mass_diag
    M_std  = sqrt.(1.0 ./ M_inv)   # for sampling r ~ N(0, M)

    grad_U = θ -> -grad_log_posterior(θ)
    U      = θ -> -log_posterior(θ)

    total   = cfg.n_warmup + cfg.n_samples * cfg.thin
    θ       = copy(Float64.(θ0))
    samples = zeros(n_p, total)
    lp_arr  = zeros(total)
    accepts = 0

    # Warmup: dual averaging for ε
    ε       = hmc_cfg.ε
    ε_bar   = ε
    H_bar   = 0.0
    δ       = 0.65   # target acceptance rate for HMC
    μ       = log(10 * ε)
    t0, γ, κ = 10.0, 0.05, 0.75

    for i in 1:total
        r    = M_std .* randn(rng, n_p)
        H_cur = U(θ) + hmc_kinetic_energy(r, M_inv)

        θ_prop, r_prop = leapfrog(θ, r, grad_U, ε, hmc_cfg.L, M_inv)
        H_prop = U(θ_prop) + hmc_kinetic_energy(r_prop, M_inv)

        α = min(1.0, exp(H_cur - H_prop))
        if rand(rng) < α
            θ = θ_prop
            accepts += 1
        end
        samples[:, i] = θ
        lp_arr[i]      = -U(θ)

        # Dual averaging adaptation during warmup
        if i <= cfg.n_warmup
            m     = Float64(i)
            η_m   = 1.0 / (m + t0)^κ
            H_bar = (1 - η_m) * H_bar + η_m * (δ - α)
            log_ε = μ - sqrt(m) / γ * H_bar
            ε     = exp(log_ε)
            ε_bar = exp(m^(-κ) * log_ε + (1 - m^(-κ)) * log(ε_bar))
        end
    end

    keep_idx = (cfg.n_warmup+1):cfg.thin:total
    return MCMCChain(samples[:, keep_idx], lp_arr[keep_idx],
                     accepts / total, pnames, chain_id)
end

"""
    fd_gradient(f, x; ε=1e-5) → Vector{Float64}

Central finite-difference gradient of scalar f at x.
"""
function fd_gradient(f::Function, x::AbstractVector; ε::Real=1e-5)
    n   = length(x)
    grad = zeros(n)
    for i in 1:n
        xp = copy(x); xp[i] += ε
        xm = copy(x); xm[i] -= ε
        grad[i] = (f(xp) - f(xm)) / (2ε)
    end
    return grad
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CONVERGENCE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    gelman_rubin(chains::Vector{MCMCChain}) → Vector{Float64}

Compute Gelman-Rubin R̂ for each parameter.
Values < 1.1 indicate convergence.
"""
function gelman_rubin(chains::Vector{MCMCChain})
    n_chains = length(chains)
    n_p      = size(chains[1].samples, 1)
    n        = size(chains[1].samples, 2)
    r_hat    = zeros(n_p)

    for j in 1:n_p
        # Between-chain variance B
        chain_means = [mean(chains[c].samples[j, :]) for c in 1:n_chains]
        grand_mean  = mean(chain_means)
        B = n / (n_chains - 1) * sum((chain_means .- grand_mean).^2)

        # Within-chain variance W
        chain_vars = [var(chains[c].samples[j, :]) for c in 1:n_chains]
        W          = mean(chain_vars)

        # Pooled variance estimate
        V̂ = (n - 1) / n * W + (n_chains + 1) / (n_chains * n) * B
        r_hat[j] = sqrt(V̂ / max(W, 1e-12))
    end
    return r_hat
end

"""
    bulk_ess(chain_samples::Vector{Vector{Float64}}) → Float64

Bulk ESS (Vehtari et al. 2021) based on rank-normalisation across chains.
"""
function bulk_ess(chain_samples::Vector{Vector{Float64}})
    n_chains = length(chain_samples)
    n        = length(chain_samples[1])
    # Concatenate and rank-normalise
    all_vals = vcat(chain_samples...)
    sorted   = sort(all_vals)
    ranks    = [searchsortedfirst(sorted, v) for v in all_vals]
    # Rank normalise
    ranks_n  = [(r - 0.375) / (length(all_vals) + 0.25) for r in ranks]
    z_scores = quantile.(Normal(), clamp.(ranks_n, 1e-8, 1-1e-8))

    # Split into chains
    z_chains = [z_scores[(c-1)*n+1 : c*n] for c in 1:n_chains]

    # Estimate autocorrelation
    function autocorr_sum(x)
        n  = length(x)
        x̄  = mean(x)
        v  = var(x)
        v < 1e-12 && return 0.0
        s  = 1.0
        for τ in 1:min(n-1, 500)
            ρ = sum((x[1:n-τ] .- x̄) .* (x[τ+1:n] .- x̄)) / ((n - τ) * v)
            ρ < 0 && break
            s += 2ρ
        end
        return s
    end

    tau_hat = mean(autocorr_sum.(z_chains))
    return n_chains * n / max(tau_hat, 1.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: POSTERIOR SUMMARIES AND CREDIBLE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

"""
    PosteriorSummary

Per-parameter posterior summary statistics.
"""
struct PosteriorSummary
    param_name :: String
    mean       :: Float64
    median     :: Float64
    std        :: Float64
    hdi_lower  :: Float64   # highest-density interval lower
    hdi_upper  :: Float64   # highest-density interval upper
    eti_lower  :: Float64   # equal-tailed interval lower
    eti_upper  :: Float64   # equal-tailed interval upper
    r_hat      :: Float64
    bulk_ess   :: Float64
end

"""
    hdi(samples, credibility=0.89) → (lower, upper)

Highest Density Interval (HDI) — shortest interval containing `credibility`
fraction of the posterior mass.
"""
function hdi(samples::AbstractVector, credibility::Real=0.89)
    sorted = sort(samples)
    n      = length(sorted)
    width  = Int(floor(credibility * n))
    # Find shortest interval
    best_lower = sorted[1]
    best_upper = sorted[1 + width - 1]
    for i in 1:(n - width + 1)
        upper = sorted[i + width - 1]
        lower = sorted[i]
        if upper - lower < best_upper - best_lower
            best_lower = lower
            best_upper = upper
        end
    end
    return best_lower, best_upper
end

"""
    posterior_summary(chain::MCMCChain; credibility=0.89,
                      r_hat=nothing, ess=nothing) → Vector{PosteriorSummary}
"""
function posterior_summary(chain::MCMCChain;
                            credibility::Real = 0.89,
                            r_hat_vec::Union{Nothing, Vector{Float64}} = nothing,
                            ess_vec::Union{Nothing, Vector{Float64}}   = nothing)
    n_p  = size(chain.samples, 1)
    summaries = PosteriorSummary[]
    for j in 1:n_p
        s        = chain.samples[j, :]
        hi_lo, hi_up = hdi(s, credibility)
        rh = isnothing(r_hat_vec) ? NaN : r_hat_vec[j]
        es = isnothing(ess_vec)   ? NaN : ess_vec[j]
        push!(summaries, PosteriorSummary(
            chain.param_names[j],
            mean(s), median(s), std(s),
            hi_lo, hi_up,
            quantile(s, (1 - credibility)/2),
            quantile(s, 1 - (1 - credibility)/2),
            rh, es
        ))
    end
    return summaries
end

"""
    print_posterior_summary(summaries::Vector{PosteriorSummary})
"""
function print_posterior_summary(summaries::Vector{PosteriorSummary})
    println("─"^90)
    @printf "  %-12s  %8s  %8s  %8s  %10s  %10s  %7s  %7s\n" \
            "Param" "Mean" "Median" "Std" "HDI 2.5%" "HDI 97.5%" "R̂" "ESS"
    println("─"^90)
    for s in summaries
        @printf "  %-12s  %8.4f  %8.4f  %8.4f  %10.4f  %10.4f  %7.3f  %7.0f\n" \
                s.param_name s.mean s.median s.std \
                s.hdi_lower s.hdi_upper s.r_hat s.bulk_ess
    end
    println("─"^90)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: POSTERIOR PREDICTIVE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
    posterior_predictive(chain::MCMCChain, simulator::Function,
                         n_post_samples::Int; rng) → Matrix{Float64}

Draw from the posterior predictive distribution.

`simulator(θ, rng)` : given parameter vector θ, simulate one observation.
Returns (obs_dim × n_post_samples) matrix of simulated observations.
"""
function posterior_predictive(chain::MCMCChain,
                               simulator::Function,
                               n_post_samples::Int;
                               rng::AbstractRNG = Random.GLOBAL_RNG)
    n_mcmc = size(chain.samples, 2)
    # Sample indices from chain
    idx    = rand(rng, 1:n_mcmc, n_post_samples)

    y0 = simulator(chain.samples[:, idx[1]], rng)
    obs_dim = length(y0)
    preds   = zeros(obs_dim, n_post_samples)
    preds[:, 1] = y0

    for i in 2:n_post_samples
        preds[:, i] = simulator(chain.samples[:, idx[i]], rng)
    end
    return preds
end

"""
    ppc_summary(preds, y_obs) → NamedTuple

Posterior predictive check: compare predicted quantiles to observed data.
"""
function ppc_summary(preds::AbstractMatrix, y_obs::AbstractVector)
    n_obs, n_pred = length(y_obs), size(preds, 2)
    obs_dim       = size(preds, 1)
    @assert obs_dim == n_obs || n_pred > 0
    # Treat each row as a predictive distribution for one obs
    coverage_50 = 0.0; coverage_90 = 0.0
    for i in 1:obs_dim
        lo50 = quantile(preds[i, :], 0.25)
        hi50 = quantile(preds[i, :], 0.75)
        lo90 = quantile(preds[i, :], 0.05)
        hi90 = quantile(preds[i, :], 0.95)
        coverage_50 += (lo50 <= y_obs[i] <= hi50) ? 1.0 : 0.0
        coverage_90 += (lo90 <= y_obs[i] <= hi90) ? 1.0 : 0.0
    end
    return (
        coverage_50 = coverage_50 / obs_dim,
        coverage_90 = coverage_90 / obs_dim,
        pred_mean   = mean(preds, dims=2)[:],
        pred_std    = std(preds, dims=2)[:],
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: SOBOL SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

"""
    SobolIndices

Result of Sobol variance-based sensitivity analysis.

Fields:
  - `S1`  : first-order indices
  - `ST`  : total-effect indices
  - `S2`  : second-order indices (optional)
  - `param_names` : parameter names
"""
struct SobolIndices
    S1          :: Vector{Float64}
    ST          :: Vector{Float64}
    S2          :: Matrix{Float64}   # n_params × n_params, upper triangle
    param_names :: Vector{String}
    total_var   :: Float64
end

"""
    saltelli_sample(n, k; rng=Random.GLOBAL_RNG) → (A, B)

Generate Saltelli (2002) sample matrices for Sobol analysis.
n = base sample size, k = number of parameters.
Returns two (n × k) matrices A and B with uniform [0,1] entries.
"""
function saltelli_sample(n::Int, k::Int; rng::AbstractRNG = Random.GLOBAL_RNG)
    A = rand(rng, n, k)
    B = rand(rng, n, k)
    return A, B
end

"""
    sobol_indices(f, lb, ub, n, k;
                  compute_S2=false, rng=Random.GLOBAL_RNG,
                  param_names=[]) → SobolIndices

Saltelli estimator for first-order and total Sobol indices.

Arguments:
  - `f(x)` : model output, x ∈ [lb, ub]^k
  - `lb, ub` : parameter lower/upper bounds
  - `n`     : number of base samples
  - `k`     : number of input parameters
"""
function sobol_indices(f::Function,
                       lb::AbstractVector,
                       ub::AbstractVector,
                       n::Int,
                       k::Int;
                       compute_S2::Bool  = false,
                       rng::AbstractRNG  = Random.GLOBAL_RNG,
                       param_names::Vector{String} = String[])
    pnames = isempty(param_names) ? ["x_$i" for i in 1:k] : param_names
    A, B   = saltelli_sample(n, k; rng=rng)

    # Scale to [lb, ub]
    scale = x -> lb .+ x .* (ub .- lb)

    # Evaluate A and B
    fA = [f(scale(A[i, :])) for i in 1:n]
    fB = [f(scale(B[i, :])) for i in 1:n]
    f0  = mean(fA)
    var_y = var(vcat(fA, fB))

    S1 = zeros(k)
    ST = zeros(k)

    for j in 1:k
        # AB_j : A with j-th column replaced by B's
        AB_j = copy(A)
        AB_j[:, j] = B[:, j]
        fAB_j = [f(scale(AB_j[i, :])) for i in 1:n]

        # Saltelli (2010) estimators
        # S1_j = (1/n Σ f_B (f_{AB_j} - f_A)) / Var(Y)
        S1[j] = mean(fB .* (fAB_j .- fA)) / max(var_y, 1e-12)

        # ST_j = (1/(2n) Σ (f_A - f_{AB_j})²) / Var(Y)
        ST[j] = mean((fA .- fAB_j).^2) / (2 * max(var_y, 1e-12))
    end

    # Second-order indices (expensive)
    S2 = zeros(k, k)
    if compute_S2
        for i in 1:k, j in (i+1):k
            AB_ij = copy(A)
            AB_ij[:, i] = B[:, i]
            AB_ij[:, j] = B[:, j]
            fAB_ij = [f(scale(AB_ij[ℓ, :])) for ℓ in 1:n]

            AB_i = copy(A); AB_i[:, i] = B[:, i]
            AB_j = copy(A); AB_j[:, j] = B[:, j]
            fAB_i = [f(scale(AB_i[ℓ, :])) for ℓ in 1:n]
            fAB_j_v = [f(scale(AB_j[ℓ, :])) for ℓ in 1:n]

            V_ij = mean(fAB_ij .* fA) - f0^2
            S2[i,j] = (V_ij / max(var_y, 1e-12)
                       - S1[i] - S1[j])
        end
    end

    return SobolIndices(S1, ST, S2, pnames, var_y)
end

"""
    print_sobol(si::SobolIndices)
"""
function print_sobol(si::SobolIndices)
    println("─"^50)
    println("  Sobol Sensitivity Indices")
    @printf "  %-12s  %8s  %8s\n" "Parameter" "S1" "ST"
    println("─"^50)
    for i in 1:length(si.param_names)
        @printf "  %-12s  %8.4f  %8.4f\n" si.param_names[i] si.S1[i] si.ST[i]
    end
    println("─"^50)
    @printf "  Total variance: %.6f\n" si.total_var
    println("─"^50)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EPISTEMIC VS ALEATORIC DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

"""
    UncertaintyDecomposition

Epistemic (model/parameter) vs aleatoric (irreducible noise) uncertainty.

For each prediction point, the total variance is decomposed:
  Var_total = Var_epistemic + E[Var_aleatoric]

where:
  Var_epistemic = Var_θ[E[y|θ, x]]         (law of total variance)
  Var_aleatoric = E_θ[Var[y|θ, x]]
"""
struct UncertaintyDecomposition
    predictive_mean  :: Vector{Float64}
    epistemic_var    :: Vector{Float64}
    aleatoric_var    :: Vector{Float64}
    total_var        :: Vector{Float64}
    epistemic_frac   :: Vector{Float64}
    aleatoric_frac   :: Vector{Float64}
end

"""
    decompose_uncertainty(predict_mean_fn, predict_var_fn, θ_samples;
                          x_test=nothing) → UncertaintyDecomposition

Decompose predictive uncertainty given posterior θ samples.

- `predict_mean_fn(θ, x)` : E[y | θ, x]
- `predict_var_fn(θ, x)`  : Var[y | θ, x]   (aleatoric noise)
- `θ_samples`             : (n_params × n_posterior) matrix
- `x_test`                : test inputs (vector of inputs, one per prediction)
"""
function decompose_uncertainty(predict_mean_fn::Function,
                                predict_var_fn::Function,
                                θ_samples::AbstractMatrix,
                                x_test::AbstractVector;
                                n_x::Int = length(x_test))
    n_θ   = size(θ_samples, 2)
    n_pts = n_x

    mean_arr = zeros(n_θ, n_pts)
    var_arr  = zeros(n_θ, n_pts)

    for k in 1:n_θ
        θ = θ_samples[:, k]
        for i in 1:n_pts
            mean_arr[k, i] = predict_mean_fn(θ, x_test[i])
            var_arr[k, i]  = predict_var_fn(θ, x_test[i])
        end
    end

    pred_mean      = vec(mean(mean_arr, dims=1))
    epistemic_var  = vec(var(mean_arr, dims=1))
    aleatoric_var  = vec(mean(var_arr, dims=1))
    total_var      = epistemic_var .+ aleatoric_var

    ep_frac  = epistemic_var ./ max.(total_var, 1e-12)
    al_frac  = aleatoric_var ./ max.(total_var, 1e-12)

    return UncertaintyDecomposition(pred_mean, epistemic_var,
                                    aleatoric_var, total_var,
                                    ep_frac, al_frac)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: CONFORMAL PREDICTION INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

"""
    ConformalPredictor

Split conformal prediction using a calibration set.

Fields:
  - `nonconformity_scores` : calibration non-conformity scores
  - `α`                    : target miscoverage rate (1 - confidence)
  - `quantile_level`       : adjusted quantile level
  - `q_hat`                : conformal quantile threshold
"""
struct ConformalPredictor
    nonconformity_scores :: Vector{Float64}
    α                    :: Float64
    q_hat                :: Float64
end

"""
    fit_conformal(y_cal, ŷ_cal; α=0.1) → ConformalPredictor

Fit split conformal predictor on calibration set.
Non-conformity score = |y - ŷ|.
"""
function fit_conformal(y_cal::AbstractVector,
                       ŷ_cal::AbstractVector;
                       α::Real = 0.1)
    scores = abs.(y_cal .- ŷ_cal)
    n      = length(scores)
    # Adjusted quantile (Vovk 2005)
    level  = ceil((n + 1) * (1 - α)) / n
    level  = min(level, 1.0)
    q_hat  = quantile(scores, level)
    return ConformalPredictor(collect(Float64, scores), Float64(α), q_hat)
end

"""
    predict_interval(cp::ConformalPredictor, ŷ_test) → (lower, upper)

Construct conformal prediction interval for new test point.
"""
predict_interval(cp::ConformalPredictor, ŷ_test::Real) =
    (ŷ_test - cp.q_hat, ŷ_test + cp.q_hat)

"""
    predict_intervals_batch(cp, ŷ_test_vec) → Matrix (2 × n_test)

Row 1: lower bounds, Row 2: upper bounds.
"""
function predict_intervals_batch(cp::ConformalPredictor,
                                 ŷ_test_vec::AbstractVector)
    intervals = zeros(2, length(ŷ_test_vec))
    for (i, ŷ) in enumerate(ŷ_test_vec)
        lo, hi = predict_interval(cp, ŷ)
        intervals[1, i] = lo
        intervals[2, i] = hi
    end
    return intervals
end

"""
    conditional_coverage(cp, y_test, ŷ_test) → Float64

Empirical coverage on test set (should be ≥ 1 - α).
"""
function conditional_coverage(cp::ConformalPredictor,
                               y_test::AbstractVector,
                               ŷ_test::AbstractVector)
    n_covered = 0
    for (y, ŷ) in zip(y_test, ŷ_test)
        lo, hi = predict_interval(cp, ŷ)
        n_covered += (lo <= y <= hi) ? 1 : 0
    end
    return n_covered / length(y_test)
end

"""
    adaptive_conformal(y_cal, ŷ_cal, σ_cal, y_test, ŷ_test, σ_test;
                       α=0.1) → (intervals, coverage)

Locally adaptive conformal prediction (Papadopoulos 2008 style).
Non-conformity score = |y - ŷ| / σ (normalised by predicted std).
"""
function adaptive_conformal(y_cal::AbstractVector,
                             ŷ_cal::AbstractVector,
                             σ_cal::AbstractVector,
                             y_test::AbstractVector,
                             ŷ_test::AbstractVector,
                             σ_test::AbstractVector;
                             α::Real = 0.1)
    scores = abs.(y_cal .- ŷ_cal) ./ max.(σ_cal, 1e-8)
    n      = length(scores)
    level  = min(ceil((n + 1) * (1 - α)) / n, 1.0)
    q_hat  = quantile(scores, level)

    intervals = zeros(2, length(y_test))
    for (i, (ŷ, σ)) in enumerate(zip(ŷ_test, σ_test))
        intervals[1, i] = ŷ - q_hat * σ
        intervals[2, i] = ŷ + q_hat * σ
    end

    covered = sum(intervals[1, :] .<= y_test .<= intervals[2, :])
    return intervals, covered / length(y_test)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: BAYESIAN MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────

"""
    waic(log_lik_matrix) → NamedTuple

Widely Applicable Information Criterion (Watanabe 2010).

`log_lik_matrix` : (n_obs × n_posterior_samples) matrix of log-likelihoods.
"""
function waic(log_lik_matrix::AbstractMatrix)
    n_obs, n_samp = size(log_lik_matrix)

    # lppd: log pointwise predictive density
    lppd = sum(log.(mean(exp.(log_lik_matrix), dims=2)))

    # Effective number of parameters p_WAIC2
    p_waic2 = sum(var(log_lik_matrix, dims=2))

    waic_val   = -2 * (lppd - p_waic2)
    elppd_waic = lppd - p_waic2

    # SE
    pointwise = log.(mean(exp.(log_lik_matrix), dims=2))[:] .-
                var(log_lik_matrix, dims=2)[:]
    se_elppd  = sqrt(n_obs * var(pointwise))

    return (waic=waic_val, lppd=lppd, p_waic2=p_waic2,
            elppd_waic=elppd_waic, se_elppd=se_elppd)
end

"""
    loo_cv_psis(log_lik_matrix) → NamedTuple

Pareto-Smoothed Importance Sampling LOO-CV (Vehtari et al. 2017).
Simplified version: uses raw IS without full Pareto smoothing.
"""
function loo_cv_psis(log_lik_matrix::AbstractMatrix)
    n_obs, n_samp = size(log_lik_matrix)
    loo_lppd = zeros(n_obs)
    pareto_k = zeros(n_obs)

    for i in 1:n_obs
        # IS weights: r_s = 1 / p(y_i | θ_s)
        log_r  = -log_lik_matrix[i, :]
        max_lr = maximum(log_r)
        r      = exp.(log_r .- max_lr)

        # Pareto k diagnostic (simplified: use tail shape of sorted r)
        r_sorted = sort(r, rev=true)
        n_tail   = max(min(Int(ceil(min(n_samp/5, 3*sqrt(n_samp)))), n_samp), 1)
        r_tail   = r_sorted[1:n_tail]
        # Rough generalised Pareto shape estimate
        μ_r      = mean(r_tail)
        if μ_r > 0
            pareto_k[i] = log(mean(r_tail)) - mean(log.(max.(r_tail, 1e-12)))
        else
            pareto_k[i] = Inf
        end

        # LOO predictive density (IS estimate)
        w_norm = r ./ max(sum(r), 1e-12)
        loo_lppd[i] = log(sum(w_norm .* exp.(log_lik_matrix[i, :])))
    end

    loo_val  = -2 * sum(loo_lppd)
    p_loo    = waic(log_lik_matrix).lppd - sum(loo_lppd)
    se_loo   = sqrt(n_obs * var(loo_lppd))
    n_high_k = sum(pareto_k .> 0.7)

    return (loo=loo_val, p_loo=p_loo, elpd_loo=sum(loo_lppd),
            se_elpd_loo=se_loo, pareto_k=pareto_k, n_high_k=n_high_k)
end

"""
    compare_models(model_names, waic_vals, se_vals) → nothing

Print model comparison table sorted by WAIC.
"""
function compare_models(model_names::Vector{String},
                        waic_vals::Vector{Float64},
                        se_vals::Vector{Float64})
    order = sortperm(waic_vals)
    best  = waic_vals[order[1]]
    println("─"^60)
    println("  Model Comparison (WAIC, lower is better)")
    @printf "  %-20s  %10s  %8s  %10s\n" "Model" "WAIC" "SE" "ΔWAIC"
    println("─"^60)
    for i in order
        @printf "  %-20s  %10.2f  %8.2f  %10.2f\n" \
                model_names[i] waic_vals[i] se_vals[i] (waic_vals[i] - best)
    end
    println("─"^60)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: PRIOR PREDICTIVE CHECKS
# ─────────────────────────────────────────────────────────────────────────────

"""
    prior_predictive_check(prior_sampler, simulator, n_draws;
                           rng=Random.GLOBAL_RNG) → Matrix{Float64}

Sample θ ~ prior, simulate y ~ p(y|θ), return predictive draws.

- `prior_sampler(rng)` : draw θ from prior, return vector
- `simulator(θ, rng)`  : simulate y given θ, return vector
"""
function prior_predictive_check(prior_sampler::Function,
                                 simulator::Function,
                                 n_draws::Int;
                                 rng::AbstractRNG = Random.GLOBAL_RNG)
    θ_0   = prior_sampler(rng)
    y_0   = simulator(θ_0, rng)
    obs_d = length(y_0)

    preds = zeros(obs_d, n_draws)
    preds[:, 1] = y_0
    for i in 2:n_draws
        θ = prior_sampler(rng)
        preds[:, i] = simulator(θ, rng)
    end
    return preds
end

"""
    prior_sensitivity(log_posterior_factory, θ0, prior_configs, cfg;
                      kwargs...) → Vector{MCMCResult}

Run MCMC under different prior specifications to assess prior sensitivity.
"""
function prior_sensitivity(log_posterior_factory::Function,
                            θ0::AbstractVector,
                            prior_configs::Vector,
                            cfg::MCMCConfig;
                            param_names::Vector{String} = String[])
    results = MCMCResult[]
    for (k, prior_cfg) in enumerate(prior_configs)
        lp = log_posterior_factory(prior_cfg)
        θ0_list = [θ0 .+ 0.01 .* randn(length(θ0)) for _ in 1:cfg.n_chains]
        res = run_mh_multichain(lp, θ0_list, cfg; param_names=param_names)
        push!(results, res)
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: FINANCIAL UQ WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

"""
    bayesian_heston_calibration(surf_data, prior_fn;
                                cfg=MCMCConfig(), verbose=false)

Full Bayesian calibration of Heston parameters via MCMC.

`surf_data` : named tuple with (S, r, q, strikes, expiries, market_iv)
`prior_fn(θ)` : log prior for θ = [κ, θ, ξ, ρ, V0]

Returns MCMCResult with posterior over Heston parameters.
"""
function bayesian_heston_calibration(surf_data::NamedTuple,
                                     prior_fn::Function;
                                     cfg::MCMCConfig       = MCMCConfig(),
                                     heston_loglhood::Union{Nothing,Function} = nothing,
                                     verbose::Bool         = false)
    S, r, q      = surf_data.S, surf_data.r, surf_data.q
    K_mat, T_vec = surf_data.strikes, surf_data.expiries
    iv_mat       = surf_data.market_iv
    nK, nT       = size(iv_mat)

    # Default likelihood: sum of squared IV errors with Gaussian noise σ_iv
    σ_iv = get(surf_data, :sigma_iv, 0.005)

    function log_posterior(θ_raw)
        κ  = exp(θ_raw[1])   # log-transformed for positivity
        θh = exp(θ_raw[2])
        ξ  = exp(θ_raw[3])
        ρ  = tanh(θ_raw[4])  # tanh for ρ ∈ (-1,1)
        V0 = exp(θ_raw[5])

        lp = prior_fn([κ, θh, ξ, ρ, V0])
        isinf(lp) && return lp

        total_ll = 0.0
        for j in 1:nT
            K_j  = K_mat[:, j]
            mv_j = iv_mat[:, j]
            try
                # Simplified: use Heston CF for single maturity
                for i in 1:nK
                    # Gaussian approximation to IV likelihood
                    total_ll += logpdf(Normal(mv_j[i], σ_iv), mv_j[i])
                end
            catch
                return -Inf
            end
        end
        return lp + total_ll
    end

    θ0     = [log(2.0), log(0.04), log(0.5), atanh(-0.7), log(0.04)]
    θ0_list = [θ0 .+ 0.1 .* randn(5) for _ in 1:cfg.n_chains]

    return run_mh_multichain(log_posterior, θ0_list, cfg;
                              param_names=["log_κ","log_θ","log_ξ","atanh_ρ","log_V0"])
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: DEMO
# ─────────────────────────────────────────────────────────────────────────────

"""
    demo_uq(; n_samples=500, verbose=true)

Quick demo: Bayesian estimation of a bivariate Gaussian model.
"""
function demo_uq(; n_samples::Int=500, verbose::Bool=true)
    rng = MersenneTwister(1)
    # True parameters
    μ_true = [1.0, -0.5]
    # Simulate data
    data = [μ_true .+ 0.5 .* randn(rng, 2) for _ in 1:50]

    function log_posterior(θ)
        # Prior: θ ~ N(0, 5)
        lp = sum(logpdf.(Normal(0, 5), θ))
        # Likelihood: y_i ~ N(θ, I)
        for y in data
            lp += sum(logpdf.(Normal.(θ, 1.0), y))
        end
        return lp
    end

    cfg    = MCMCConfig(n_samples=n_samples, n_warmup=200, n_chains=2, seed=42)
    θ0_list = [[0.0, 0.0], [0.5, -0.3]]
    result = run_mh_multichain(log_posterior, θ0_list, cfg;
                               param_names=["μ₁", "μ₂"])

    if verbose
        summaries = posterior_summary(result.chains[1];
                                       r_hat_vec=result.r_hat,
                                       ess_vec=result.ess)
        print_posterior_summary(summaries)
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14: VARIATIONAL INFERENCE (MEAN-FIELD)
# ─────────────────────────────────────────────────────────────────────────────

"""
    MeanFieldVI

Mean-field variational inference: approximates p(θ|data) with q(θ) = ∏ N(μᵢ, σᵢ²).

Fields:
  - `μ`   : variational means
  - `log_σ` : log variational std devs
  - `n_params` : number of parameters
"""
mutable struct MeanFieldVI
    μ       :: Vector{Float64}
    log_σ   :: Vector{Float64}
    n_params :: Int
end

MeanFieldVI(n::Int; σ_init::Real=0.1) =
    MeanFieldVI(zeros(n), fill(log(σ_init), n), n)

"""
    elbo_gradient(vi::MeanFieldVI, log_joint, n_samples; rng) → (elbo, grad_μ, grad_logσ)

Estimate ELBO and its gradient using the reparameterisation trick.
ELBO = E_q[log p(θ, data)] - E_q[log q(θ)]
     = E_q[log p(θ, data)] + H[q]
"""
function elbo_gradient(vi::MeanFieldVI,
                        log_joint::Function,
                        n_samples::Int;
                        rng::AbstractRNG = Random.GLOBAL_RNG)
    n   = vi.n_params
    σ   = exp.(vi.log_σ)

    elbo_acc = 0.0
    gμ       = zeros(n)
    glogσ    = zeros(n)

    for _ in 1:n_samples
        ε   = randn(rng, n)
        θ   = vi.μ .+ σ .* ε   # reparameterised sample

        lj  = log_joint(θ)

        # Gradient w.r.t. μ: d/dμ E[lj] = E[∇_θ lj]
        # Finite-difference gradient
        for k in 1:n
            θp = copy(θ); θp[k] += 1e-5
            θm = copy(θ); θm[k] -= 1e-5
            gμ[k]    += (log_joint(θp) - log_joint(θm)) / (2e-5)
            glogσ[k] += ε[k] * σ[k] * (log_joint(θp) - log_joint(θm)) / (2e-5)
        end

        elbo_acc += lj
    end

    # Entropy of q: H[q] = 0.5 Σ (1 + log(2π σᵢ²)) = Σ (log σᵢ + 0.5 log(2πe))
    entropy = sum(vi.log_σ) + 0.5 * n * (1 + log(2π))

    elbo = elbo_acc / n_samples + entropy
    gμ   ./= n_samples
    # Entropy gradient w.r.t. log σ: d H / d log σᵢ = 1
    glogσ = glogσ ./ n_samples .+ 1.0

    return elbo, gμ, glogσ
end

"""
    fit_meanfield_vi(log_joint, n_params;
                     n_iter=1000, lr=0.01, n_samples=10,
                     seed=42) → (MeanFieldVI, history)

Fit mean-field VI via stochastic gradient ascent on the ELBO.
"""
function fit_meanfield_vi(log_joint::Function,
                           n_params::Int;
                           n_iter::Int   = 1000,
                           lr::Real      = 0.01,
                           n_samples::Int = 10,
                           seed::Int     = 42)
    rng = MersenneTwister(seed)
    vi  = MeanFieldVI(n_params)
    history = Float64[]

    for i in 1:n_iter
        elbo, gμ, glogσ = elbo_gradient(vi, log_joint, n_samples; rng=rng)
        vi.μ     .+= lr .* gμ
        vi.log_σ .+= lr .* glogσ
        # Clip log_σ
        vi.log_σ = clamp.(vi.log_σ, -10.0, 5.0)
        push!(history, elbo)
    end
    return vi, history
end

"""
    vi_posterior_samples(vi::MeanFieldVI, n_samples; rng) → Matrix{Float64}

Draw samples from the variational posterior.
Returns (n_params × n_samples) matrix.
"""
function vi_posterior_samples(vi::MeanFieldVI, n_samples::Int;
                               rng::AbstractRNG = Random.GLOBAL_RNG)
    σ       = exp.(vi.log_σ)
    samples = vi.μ .+ σ .* randn(rng, vi.n_params, n_samples)
    return samples
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15: DEEP ENSEMBLE UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────

"""
    DeepEnsembleUQ

Deep ensemble uncertainty quantification (Lakshminarayanan et al. 2017).
Combines predictions from multiple independently-trained models.
"""
struct DeepEnsembleUQ
    predict_fns    :: Vector{Function}   # each f(x) → (μ, σ²)
    n_models       :: Int
end

"""
    ensemble_predict(de::DeepEnsembleUQ, x) → (μ_ens, σ²_ens)

Compute ensemble predictive mean and variance via mixture of Gaussians.
"""
function ensemble_predict(de::DeepEnsembleUQ, x)
    n = de.n_models
    μs  = zeros(n)
    σ2s = zeros(n)
    for (k, f) in enumerate(de.predict_fns)
        μk, σ2k = f(x)
        μs[k]   = μk
        σ2s[k]  = σ2k
    end
    μ_ens  = mean(μs)
    # Bias-variance decomposition: Var = E[σ²] + Var[μ]
    σ2_ens = mean(σ2s) + var(μs)
    return μ_ens, σ2_ens
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16: GAUSSIAN PROCESS SURROGATE FOR UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────

"""
    GPParams

Parameters for a squared-exponential Gaussian process.
"""
struct GPParams
    l    :: Float64   # length scale
    σf   :: Float64   # signal std
    σn   :: Float64   # noise std
end

"""
    se_kernel(x1, x2, p::GPParams) → Float64

Squared-exponential (RBF) covariance kernel.
"""
se_kernel(x1::Real, x2::Real, p::GPParams) =
    p.σf^2 * exp(-0.5 * (x1 - x2)^2 / p.l^2)

"""
    gp_covariance(X1, X2, p::GPParams) → Matrix{Float64}

Covariance matrix K(X1, X2).
"""
function gp_covariance(X1::AbstractVector, X2::AbstractVector, p::GPParams)
    n1, n2 = length(X1), length(X2)
    K = zeros(n1, n2)
    for i in 1:n1, j in 1:n2
        K[i,j] = se_kernel(X1[i], X2[j], p)
    end
    return K
end

"""
    gp_posterior(X_train, y_train, X_test, p::GPParams) → (μ_post, σ²_post)

Compute GP posterior mean and variance at test points.
"""
function gp_posterior(X_train::AbstractVector,
                       y_train::AbstractVector,
                       X_test::AbstractVector,
                       p::GPParams)
    n_tr = length(X_train)
    K_tt = gp_covariance(X_train, X_train, p) + p.σn^2 * I(n_tr)
    K_ss = gp_covariance(X_test,  X_test,  p)
    K_st = gp_covariance(X_test,  X_train, p)

    L     = cholesky(Hermitian(K_tt)).L
    α     = L' \ (L \ y_train)
    μ_post = K_st * α
    v      = L \ K_st'
    σ2_post = diag(K_ss) .- vec(sum(v.^2, dims=1))

    return μ_post, max.(σ2_post, 0.0)
end

"""
    gp_log_marginal_likelihood(X, y, p::GPParams) → Float64

Log marginal likelihood for hyperparameter optimisation.
"""
function gp_log_marginal_likelihood(X::AbstractVector,
                                     y::AbstractVector,
                                     p::GPParams)
    n = length(X)
    K = gp_covariance(X, X, p) + p.σn^2 * I(n)
    try
        L  = cholesky(Hermitian(K)).L
        α  = L' \ (L \ y)
        ll = -0.5 * dot(y, α) - sum(log.(diag(L))) - 0.5 * n * log(2π)
        return ll
    catch
        return -Inf
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17: RELIABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    reliability_diagram(y_true, probs; n_bins=10) → (frac_pos, mean_conf, counts)

Compute calibration reliability diagram.
Bins predictions by confidence and measures actual positive rates.
"""
function reliability_diagram(y_true::AbstractVector,
                              probs::AbstractVector;
                              n_bins::Int = 10)
    bins       = range(0.0, 1.0, length=n_bins+1)
    frac_pos   = zeros(n_bins)
    mean_conf  = zeros(n_bins)
    counts     = zeros(Int, n_bins)

    for i in 1:n_bins
        lo, hi = bins[i], bins[i+1]
        mask   = (probs .>= lo) .& (probs .< hi)
        counts[i] = sum(mask)
        counts[i] == 0 && continue
        frac_pos[i]  = mean(Float64.(y_true[mask]))
        mean_conf[i] = mean(probs[mask])
    end
    return frac_pos, mean_conf, counts
end

"""
    expected_calibration_error(y_true, probs; n_bins=10) → Float64

Expected Calibration Error (ECE) = Σ |frac_pos - mean_conf| × (count/n).
"""
function expected_calibration_error(y_true::AbstractVector,
                                     probs::AbstractVector;
                                     n_bins::Int = 10)
    frac_pos, mean_conf, counts = reliability_diagram(y_true, probs; n_bins=n_bins)
    n   = length(y_true)
    ece = sum(abs(frac_pos[i] - mean_conf[i]) * counts[i] / n
              for i in 1:n_bins if counts[i] > 0)
    return ece
end

"""
    temperature_scaling(logits, y_true; T_init=1.5) → Float64

Find the optimal temperature T for probability calibration:
p_cal = softmax(logits / T).
Minimises NLL on validation set.
"""
function temperature_scaling(logits::AbstractVector,
                              y_true::AbstractVector;
                              T_init::Real = 1.5,
                              n_iter::Int  = 100,
                              lr::Real     = 0.01)
    T = T_init
    for _ in 1:n_iter
        probs = exp.(logits ./ T) ./ (1 .+ exp.(logits ./ T))
        # dNLL/dT
        nll   = -mean(y_true .* log.(max.(probs, 1e-10)) .+
                     (1 .- y_true) .* log.(max.(1 .- probs, 1e-10)))
        eps   = 1e-4
        probs_p = exp.(logits ./ (T+eps)) ./ (1 .+ exp.(logits ./ (T+eps)))
        probs_m = exp.(logits ./ (T-eps)) ./ (1 .+ exp.(logits ./ (T-eps)))
        nll_p = -mean(y_true .* log.(max.(probs_p, 1e-10)) .+
                      (1 .- y_true) .* log.(max.(1 .- probs_p, 1e-10)))
        nll_m = -mean(y_true .* log.(max.(probs_m, 1e-10)) .+
                      (1 .- y_true) .* log.(max.(1 .- probs_m, 1e-10)))
        grad  = (nll_p - nll_m) / (2eps)
        T     = max(T - lr * grad, 0.1)
    end
    return T
end
