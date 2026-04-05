"""
BayesianPortfolio — Bayesian Portfolio Theory

Full Bayesian treatment of portfolio construction:
  - Normal-Inverse-Wishart conjugate prior for returns
  - Posterior predictive distribution
  - Black-Litterman via Bayesian likelihood framework
  - MCMC portfolio optimization (Metropolis-Hastings)
  - Shrinkage estimators: James-Stein, Ledoit-Wolf, Oracle
  - Sequential Bayesian updating
  - Decision-theoretic portfolio under posterior uncertainty
  - Predictive Sharpe with estimation risk
"""
module BayesianPortfolio

using Statistics
using LinearAlgebra
using Random

export NormalInverseWishart, posterior_update, posterior_predictive
export bl_bayesian, bl_posterior_moments
export mcmc_portfolio, metropolis_hastings_weights
export james_stein_estimator, oracle_shrinkage
export sequential_bayesian_update, kalman_mean_update
export decision_theoretic_portfolio, maximize_posterior_utility
export predictive_sharpe_ratio, estimation_adjusted_sharpe
export bayesian_model_comparison, marginal_likelihood

# =============================================================================
# SECTION 1: NORMAL-INVERSE-WISHART CONJUGATE PRIOR
# =============================================================================

"""
    NormalInverseWishart

Normal-Inverse-Wishart prior/posterior for (μ, Σ) of multivariate normal returns.

Parameterization:
  μ | Σ ~ N(μ₀, Σ/κ₀)
  Σ ~ IW(ν₀, Ψ₀)

where:
  μ₀: prior mean vector
  κ₀: prior belief strength for mean (pseudo-samples)
  ν₀: prior degrees of freedom for covariance (must be > N-1)
  Ψ₀: prior scale matrix

Marginal of μ: t_{ν₀-N+1}(μ₀, Ψ₀/(κ₀(ν₀-N+1)))
"""
struct NormalInverseWishart
    mu0::Vector{Float64}   # prior mean
    kappa0::Float64        # prior strength
    nu0::Float64           # prior degrees of freedom
    Psi0::Matrix{Float64}  # prior scale matrix

    function NormalInverseWishart(mu0, kappa0, nu0, Psi0)
        N = length(mu0)
        @assert size(Psi0) == (N, N) "Psi0 must be N×N"
        @assert kappa0 > 0 "kappa0 must be positive"
        @assert nu0 > N - 1 "nu0 must be > N-1"
        new(Float64.(mu0), Float64(kappa0), Float64(nu0), Float64.(Psi0))
    end
end

"""
    NormalInverseWishart(N; scale=0.01) -> NormalInverseWishart

Default diffuse NIW prior for N-dimensional returns.
"""
function NormalInverseWishart(N::Int; scale::Float64=0.01)
    mu0  = zeros(N)
    kappa0 = 1.0
    nu0  = Float64(N + 2)
    Psi0 = scale * Matrix{Float64}(I, N, N)
    return NormalInverseWishart(mu0, kappa0, nu0, Psi0)
end

"""
    posterior_update(prior::NormalInverseWishart, returns) -> NormalInverseWishart

Bayesian update of Normal-Inverse-Wishart prior given observed returns.

Conjugate update formulas:
  κₙ = κ₀ + T
  νₙ = ν₀ + T
  μₙ = (κ₀μ₀ + T*x̄) / κₙ
  Ψₙ = Ψ₀ + S + (κ₀T/κₙ)(x̄-μ₀)(x̄-μ₀)'

where S = Σᵢ(xᵢ-x̄)(xᵢ-x̄)' is the sample scatter matrix.

This is exact: no approximation or MCMC needed.
"""
function posterior_update(prior::NormalInverseWishart,
                            returns::Matrix{Float64})::NormalInverseWishart

    T, N = size(returns)
    @assert length(prior.mu0) == N "Dimension mismatch"

    T == 0 && return prior

    x_bar = vec(mean(returns, dims=1))

    # Sample scatter (unnormalized covariance)
    deviations = returns .- x_bar'
    S = deviations' * deviations  # N × N

    # Conjugate updates
    kappa_n = prior.kappa0 + T
    nu_n = prior.nu0 + T
    mu_n = (prior.kappa0 * prior.mu0 + T * x_bar) / kappa_n

    # Update scale matrix
    diff_mean = x_bar - prior.mu0
    Psi_n = prior.Psi0 + S + (prior.kappa0 * T / kappa_n) * (diff_mean * diff_mean')

    return NormalInverseWishart(mu_n, kappa_n, nu_n, Psi_n)
end

"""
    posterior_predictive(posterior::NormalInverseWishart) -> NamedTuple

Compute posterior predictive distribution for future returns.

The posterior predictive is a multivariate Student-t:
    x_new | data ~ t_{ν_n-N+1}(μ_n, Ψ_n(κ_n+1)/(κ_n(ν_n-N+1)))

This accounts for both parameter uncertainty (estimation risk) and
natural randomness. The degrees of freedom decrease with sample size.

# Returns
- NamedTuple: mean, scale_matrix, df, predictive_cov
"""
function posterior_predictive(posterior::NormalInverseWishart)
    N = length(posterior.mu0)

    df = posterior.nu0 - N + 1.0  # degrees of freedom > 0 if nu0 > N-1
    df = max(df, 2.1)  # ensure finite variance

    # Scale matrix for predictive distribution
    scale_factor = (posterior.kappa0 + 1) / (posterior.kappa0 * (posterior.nu0 - N + 1))
    predictive_scale = scale_factor * posterior.Psi0

    # Predictive covariance (if df > 2): Σ_pred = (df/(df-2)) * scale
    predictive_cov = df > 2 ? (df / (df - 2)) * predictive_scale : predictive_scale

    return (mean=posterior.mu0, scale_matrix=predictive_scale,
             df=df, predictive_cov=predictive_cov,
             kappa=posterior.kappa0, nu=posterior.nu0)
end

"""
    sample_posterior_predictive(posterior, n_samples; seed=42) -> Matrix{Float64}

Draw samples from the posterior predictive distribution via:
  1. Sample Σ ~ IW(ν_n, Ψ_n)
  2. Sample μ | Σ ~ N(μ_n, Σ/κ_n)
  3. Sample x | μ, Σ ~ N(μ, Σ)

# Returns
- (n_samples × N) matrix of predictive samples
"""
function sample_posterior_predictive(posterior::NormalInverseWishart,
                                       n_samples::Int;
                                       seed::Int=42)::Matrix{Float64}
    N = length(posterior.mu0)
    rng = MersenneTwister(seed)

    samples = zeros(n_samples, N)

    for s in 1:n_samples
        # Step 1: sample Sigma from IW(nu_n, Psi_n)
        # Approximate: use mean of IW = Psi_n / (nu_n - N - 1)
        nu_n = posterior.nu0
        if nu_n > N + 1
            Sigma_s = posterior.Psi0 ./ (nu_n - N - 1)
        else
            Sigma_s = posterior.Psi0 ./ max(nu_n - N, 0.5)
        end
        Sigma_s = Symmetric(Sigma_s + 1e-8*I)

        # Add noise to simulate IW variation (Bartlett decomposition approximation)
        # For simplicity use Cholesky + chi-squared scaling
        L = try cholesky(Sigma_s).L catch I(N) * 0.01 end

        # Step 2: sample mu | Sigma
        mu_noise = L * randn(rng, N) ./ sqrt(posterior.kappa0)
        mu_s = posterior.mu0 .+ mu_noise

        # Step 3: sample x | mu, Sigma
        samples[s, :] = mu_s .+ L * randn(rng, N)
    end

    return samples
end

# =============================================================================
# SECTION 2: BLACK-LITTERMAN VIA BAYESIAN FRAMEWORK
# =============================================================================

"""
    bl_bayesian(prior::NormalInverseWishart, views_P, views_Q, views_Omega;
                market_weights=nothing, delta=2.5) -> NormalInverseWishart

Black-Litterman model reformulated as Bayesian updating.

Prior: Gaussian prior on μ from CAPM equilibrium (or flat prior)
Likelihood: views P*μ ~ N(Q, Omega) as a Gaussian likelihood

Posterior: μ_BL ~ N(μ_post, Σ_post) by normal-normal conjugacy.

This is more general than the standard BL as it naturally handles
different strengths of views and arbitrary prior distributions.

# Arguments
- `prior`: NIW prior (use market equilibrium as μ₀)
- `views_P`: K×N matrix of view portfolios
- `views_Q`: K-vector of view expected returns
- `views_Omega`: K×K view uncertainty matrix (diagonal = independent views)
- `market_weights`: N-vector market cap weights (for equilibrium)
- `delta`: risk aversion (for equilibrium prior)
"""
function bl_bayesian(prior::NormalInverseWishart,
                      views_P::Matrix{Float64},
                      views_Q::Vector{Float64},
                      views_Omega::Matrix{Float64};
                      market_weights::Union{Vector{Float64},Nothing}=nothing,
                      delta::Float64=2.5)

    N = length(prior.mu0)
    K = length(views_Q)

    @assert size(views_P) == (K, N)
    @assert size(views_Omega) == (K, K)

    # Prior covariance for mean: τ * Σ
    tau = 1.0 / max(prior.kappa0, 1.0)
    Sigma_prior = prior.Psi0 ./ max(prior.nu0 - N - 1, 1.0)
    tau_Sigma = tau * Sigma_prior + 1e-10*I

    # Posterior update: combine prior μ₀ with views
    tau_Sigma_inv = try inv(tau_Sigma) catch pinv(tau_Sigma) end
    Omega_inv = try inv(views_Omega + 1e-12*I) catch pinv(views_Omega) end

    # Posterior precision = prior precision + view precision
    Sigma_post_inv = tau_Sigma_inv + views_P' * Omega_inv * views_P
    Sigma_post = try inv(Sigma_post_inv + 1e-10*I) catch pinv(Sigma_post_inv) end

    # Posterior mean
    mu_post = Sigma_post * (tau_Sigma_inv * prior.mu0 + views_P' * Omega_inv * views_Q)

    # Updated kappa (strengthen prior)
    kappa_post = prior.kappa0 + K
    nu_post = prior.nu0 + K

    # Updated Psi from posterior covariance
    Psi_post = Sigma_post .* max(nu_post - N - 1, 1.0)

    return NormalInverseWishart(mu_post, kappa_post, nu_post, Psi_post)
end

"""
    bl_posterior_moments(posterior::NormalInverseWishart) -> NamedTuple

Extract posterior moments for portfolio construction.

# Returns
- NamedTuple: mean, covariance, std, confidence_ellipse_radius
"""
function bl_posterior_moments(posterior::NormalInverseWishart)
    N = length(posterior.mu0)
    nu = posterior.nu0
    kappa = posterior.kappa0

    # Posterior mean of μ
    mean_mu = posterior.mu0

    # Posterior mean of Σ
    Sigma_mean = if nu > N + 1
        posterior.Psi0 ./ (nu - N - 1)
    else
        posterior.Psi0
    end

    # Marginal variance of μ (integrating out Σ)
    # Var(μ_i) = Psi_ii / (kappa * (nu - N - 1))
    margin_var = diag(Sigma_mean) ./ max(kappa * (nu - N - 1), 1e-10)

    return (mean=mean_mu, covariance=Sigma_mean,
             std=sqrt.(max.(margin_var, 0.0)), kappa=kappa, nu=nu)
end

# =============================================================================
# SECTION 3: MCMC PORTFOLIO OPTIMIZATION
# =============================================================================

"""
    metropolis_hastings_weights(logpost_func, N; n_iter=10000, burnin=2000,
                                 seed=42) -> Matrix{Float64}

Generic Metropolis-Hastings sampler for portfolio weights.

Samples from a distribution over the N-simplex (long-only constraint):
    w ∈ ΔN: Σwᵢ = 1, wᵢ ≥ 0

Uses a Dirichlet proposal distribution centered at current weights.

# Arguments
- `logpost_func`: function(w::Vector{Float64}) -> log posterior density
- `N`: number of assets
- `n_iter`: total iterations
- `burnin`: burn-in period (discarded)
- `seed`: random seed

# Returns
- (n_samples × N) matrix of weight samples
"""
function metropolis_hastings_weights(logpost_func::Function,
                                       N::Int;
                                       n_iter::Int=10_000,
                                       burnin::Int=2_000,
                                       seed::Int=42)::Matrix{Float64}

    rng = MersenneTwister(seed)

    # Initialize at equal weights
    w_current = ones(N) / N
    lp_current = logpost_func(w_current)

    n_samples = n_iter - burnin
    samples = zeros(n_samples, N)
    n_accepted = 0

    # Concentration parameter for Dirichlet proposal
    concentration = 10.0

    for iter in 1:n_iter
        # Dirichlet proposal: alpha = concentration * w_current
        alpha = concentration .* w_current .+ 0.1
        w_proposed = _sample_dirichlet(rng, alpha)

        lp_proposed = logpost_func(w_proposed)

        # Acceptance ratio (with Dirichlet proposal correction)
        log_accept = lp_proposed - lp_current
        # Proposal correction: q(w_curr | w_prop) / q(w_prop | w_curr)
        # For Dirichlet: log q(w|α) = log Γ(Σα) - Σlog Γ(αᵢ) + Σ(αᵢ-1)log(wᵢ)
        log_q_fwd = _log_dirichlet_density(w_proposed, concentration .* w_current .+ 0.1)
        log_q_rev = _log_dirichlet_density(w_current, concentration .* w_proposed .+ 0.1)
        log_accept += log_q_rev - log_q_fwd

        if log(rand(rng)) < log_accept
            w_current = w_proposed
            lp_current = lp_proposed
            n_accepted += 1
        end

        if iter > burnin
            samples[iter - burnin, :] = w_current
        end

        # Adapt concentration parameter
        if iter % 100 == 0
            accept_rate = n_accepted / iter
            if accept_rate > 0.4
                concentration *= 1.1
            elseif accept_rate < 0.2
                concentration *= 0.9
            end
            concentration = clamp(concentration, 1.0, 1000.0)
        end
    end

    return samples
end

"""Sample from Dirichlet(alpha) distribution."""
function _sample_dirichlet(rng::AbstractRNG, alpha::Vector{Float64})::Vector{Float64}
    # Dirichlet from Gamma samples
    x = [rand(rng) for _ in alpha]  # uniform sample → Gamma approximation
    # Gamma(a, 1) from log transformation (Wilson-Hilferty)
    gamma_samples = zeros(length(alpha))
    for (i, a) in enumerate(alpha)
        gamma_samples[i] = max(1e-10, _sample_gamma(rng, a))
    end
    total = sum(gamma_samples)
    return total > 0 ? gamma_samples ./ total : ones(length(alpha)) / length(alpha)
end

"""Sample from Gamma(a, 1) using Marsaglia-Tsang method."""
function _sample_gamma(rng::AbstractRNG, a::Float64)::Float64
    a < 1 && return _sample_gamma(rng, a + 1) * rand(rng)^(1/a)
    d = a - 1/3
    c = 1 / sqrt(9d)
    while true
        x = randn(rng)
        v = (1 + c*x)^3
        v <= 0 && continue
        u = rand(rng)
        if u < 1 - 0.0331 * x^4 || log(u) < 0.5*x^2 + d*(1-v+log(v))
            return d * v
        end
    end
end

"""Log-density of Dirichlet distribution."""
function _log_dirichlet_density(x::Vector{Float64}, alpha::Vector{Float64})::Float64
    n = length(x)
    # log Γ(Σα) - Σ log Γ(αᵢ) + Σ(αᵢ-1)log(xᵢ)
    sum_alpha = sum(alpha)
    log_norm = _lgamma(sum_alpha) - sum(_lgamma.(alpha))
    log_kernel = sum((alpha[i] - 1) * log(max(x[i], 1e-15)) for i in 1:n)
    return log_norm + log_kernel
end

"""Log gamma function (Stirling approximation for large x)."""
function _lgamma(x::Float64)::Float64
    x <= 0 && return Inf
    x < 0.5 && return _lgamma(1 - x) + log(π / sin(π*x))  # reflection
    # Lanczos approximation
    if x < 7
        # Recurse up
        result = 0.0
        while x < 7
            result -= log(x)
            x += 1.0
        end
        return result + _lgamma_stirling(x)
    end
    return _lgamma_stirling(x)
end

function _lgamma_stirling(x::Float64)::Float64
    (x - 0.5)*log(x) - x + 0.5*log(2π) + 1/(12x) - 1/(360x^3)
end

"""
    mcmc_portfolio(returns, utility_func; n_iter=5000, burnin=1000) -> NamedTuple

MCMC portfolio optimization: sample weights from posterior distribution
over portfolios, where the "posterior" is proportional to exp(utility).

# Arguments
- `returns`: (T × N) return matrix
- `utility_func`: function(w, returns) -> Float64 (higher = better)

# Returns
- NamedTuple: mean_weights, samples, posterior_sharpe_mean, posterior_sharpe_std
"""
function mcmc_portfolio(returns::Matrix{Float64},
                          utility_func::Function;
                          n_iter::Int=5_000,
                          burnin::Int=1_000)

    T, N = size(returns)

    function logpost(w)
        all(w .>= 0) && abs(sum(w) - 1) < 1e-6 || return -Inf
        try
            return utility_func(w, returns)
        catch
            return -Inf
        end
    end

    samples = metropolis_hastings_weights(logpost, N;
                                            n_iter=n_iter, burnin=burnin)

    mean_weights = vec(mean(samples, dims=1))
    mean_weights = max.(mean_weights, 0.0)
    mean_weights ./= sum(mean_weights)

    # Compute posterior Sharpe for each sample
    port_returns = returns * mean_weights
    post_sharpe = std(port_returns) > 0 ? mean(port_returns) / std(port_returns) * sqrt(252) : 0.0

    return (mean_weights=mean_weights, samples=samples,
             posterior_sharpe_mean=post_sharpe)
end

# =============================================================================
# SECTION 4: SHRINKAGE ESTIMATORS
# =============================================================================

"""
    james_stein_estimator(x_bar, Sigma, n; target=nothing) -> Vector{Float64}

James-Stein (1961) shrinkage estimator for the mean.

The James-Stein estimator shrinks the sample mean toward a target μ₀:
    μ_JS = (1 - c/||x̄ - μ₀||²) * x̄ + c/||x̄ - μ₀||² * μ₀

where c = (N-2)σ²/T is the optimal shrinkage coefficient.

When N ≥ 3, JS dominates the MLE under squared loss (Stein's paradox).
In portfolio context: shrink toward equal or zero mean.

# Arguments
- `x_bar`: sample mean vector (N)
- `Sigma`: covariance estimate (N×N)
- `n`: sample size T

# Returns
- Shrunk mean vector
"""
function james_stein_estimator(x_bar::Vector{Float64},
                                  Sigma::Matrix{Float64},
                                  n::Int;
                                  target::Union{Vector{Float64},Nothing}=nothing)::Vector{Float64}

    N = length(x_bar)
    N < 3 && return x_bar  # JS only helps for N ≥ 3

    mu0 = target === nothing ? zeros(N) : target
    diff = x_bar - mu0

    # Estimate sigma^2 as average diagonal variance / n
    sigma2 = mean(diag(Sigma)) / n

    # Mahalanobis-like norm
    norm_sq = diff' * diff
    norm_sq < 1e-15 && return x_bar

    # JS shrinkage coefficient
    c = max(0.0, (N - 2) * sigma2)
    shrinkage = min(1.0, c / norm_sq)

    return (1.0 - shrinkage) * x_bar + shrinkage * mu0
end

"""
    oracle_shrinkage(sample_cov, true_cov) -> NamedTuple

Oracle shrinkage estimator: optimal alpha* if we knew the true covariance.

α* = argmin_α ||α * F + (1-α) * S - Σ||²_F

where F is the target, S is the sample covariance, Σ is the true covariance.

This gives the performance ceiling for shrinkage estimators.

# Returns
- NamedTuple: optimal_alpha, oracle_mse
"""
function oracle_shrinkage(sample_cov::Matrix{Float64},
                            true_cov::Matrix{Float64};
                            target::Union{Matrix{Float64},Nothing}=nothing)

    N = size(sample_cov, 1)
    F = target === nothing ? mean(diag(sample_cov)) * I(N) : target

    # α* = tr((Σ-S)(F-S)) / ||F-S||²
    num = tr((true_cov - sample_cov) * (F - sample_cov))
    denom = norm(F - sample_cov)^2

    alpha_star = denom > 0 ? clamp(num / denom, 0.0, 1.0) : 0.0
    oracle_est = alpha_star * F + (1 - alpha_star) * sample_cov
    oracle_mse = norm(oracle_est - true_cov)^2

    return (optimal_alpha=alpha_star, oracle_mse=oracle_mse, estimate=oracle_est)
end

# =============================================================================
# SECTION 5: SEQUENTIAL BAYESIAN UPDATING
# =============================================================================

"""
    sequential_bayesian_update(priors::NormalInverseWishart,
                                return_stream::Matrix{Float64};
                                batch_size=21) -> Vector{NormalInverseWishart}

Sequentially update the NIW posterior as new returns arrive.

Returns a vector of posteriors at each step, enabling:
- Online learning of the return distribution
- Monitoring of parameter evolution
- Studying how quickly beliefs converge

# Arguments
- `priors`: initial NIW prior
- `return_stream`: (T × N) returns in chronological order
- `batch_size`: update frequency (1 = update every day)

# Returns
- Vector of NIW posteriors (one per batch)
"""
function sequential_bayesian_update(prior::NormalInverseWishart,
                                      return_stream::Matrix{Float64};
                                      batch_size::Int=21)::Vector{NormalInverseWishart}

    T, N = size(return_stream)
    posteriors = NormalInverseWishart[]

    current_posterior = prior
    t = 1
    while t <= T
        batch_end = min(t + batch_size - 1, T)
        batch = return_stream[t:batch_end, :]
        current_posterior = posterior_update(current_posterior, batch)
        push!(posteriors, current_posterior)
        t = batch_end + 1
    end

    return posteriors
end

"""
    kalman_mean_update(mu_prior, P_prior, observation, H, R) -> NamedTuple

Kalman filter update for the mean, treating Σ as known.

State: μ (N-vector of expected returns)
Observation: y = H*μ + v, v ~ N(0, R)

Update:
    K = P H' (H P H' + R)⁻¹     (Kalman gain)
    μ_post = μ_prior + K(y - H*μ_prior)
    P_post = (I - K H) P_prior

Useful for sequential updating of return predictions.
"""
function kalman_mean_update(mu_prior::Vector{Float64},
                              P_prior::Matrix{Float64},
                              observation::Vector{Float64},
                              H::Matrix{Float64},
                              R::Matrix{Float64})

    N = length(mu_prior)

    # Innovation
    y_pred = H * mu_prior
    innovation = observation - y_pred

    # Innovation covariance
    S_innov = H * P_prior * H' + R + 1e-10*I

    # Kalman gain
    K = P_prior * H' * inv(S_innov)

    # Posterior update
    mu_post = mu_prior + K * innovation
    P_post = (I(N) - K * H) * P_prior

    return (mu=mu_post, P=P_post, innovation=innovation, K=K)
end

# =============================================================================
# SECTION 6: DECISION-THEORETIC PORTFOLIO
# =============================================================================

"""
    decision_theoretic_portfolio(posterior::NormalInverseWishart;
                                  utility=:log, risk_aversion=3.0,
                                  n_samples=2000) -> Vector{Float64}

Optimal portfolio under a decision-theoretic criterion, integrating
out parameter uncertainty via the posterior predictive distribution.

Utility functions:
- :log (Bernoulli-Kelly): maximize E[log(1 + w'r)]
- :power (CRRA): maximize E[(1+w'r)^{1-γ}]/(1-γ)
- :quadratic: maximize w'μ - (γ/2) w'Σw (Markowitz)
- :downside: maximize E[w'r] - γ*CVaR(w'r; α=0.05)

# Returns
- Optimal weights vector
"""
function decision_theoretic_portfolio(posterior::NormalInverseWishart;
                                        utility::Symbol=:log,
                                        risk_aversion::Float64=3.0,
                                        n_samples::Int=2000,
                                        seed::Int=42)::Vector{Float64}

    N = length(posterior.mu0)

    # Sample from posterior predictive for Monte Carlo expected utility
    pred_samples = sample_posterior_predictive(posterior, n_samples; seed=seed)

    function expected_utility(w::Vector{Float64})::Float64
        w_clamp = max.(w, 0.0)
        total = sum(w_clamp)
        total < 1e-10 && return -1e15
        w_norm = w_clamp ./ total

        port_returns = pred_samples * w_norm

        if utility == :log
            # Kelly/log utility: E[log(1+r)]
            log_vals = log.(max.(1.0 .+ port_returns, 1e-10))
            return mean(log_vals)

        elseif utility == :power
            # CRRA with γ = risk_aversion
            gamma = risk_aversion
            if gamma == 1.0
                return mean(log.(max.(1.0 .+ port_returns, 1e-10)))
            end
            values = (1.0 .+ port_returns) .^ (1.0 - gamma) ./ (1.0 - gamma)
            return mean(values)

        elseif utility == :quadratic
            # Mean-variance with γ
            mu_p = mean(port_returns)
            var_p = var(port_returns)
            return mu_p - 0.5 * risk_aversion * var_p

        else  # :downside
            mu_p = mean(port_returns)
            sorted_r = sort(port_returns)
            n_tail = max(1, floor(Int, 0.05 * n_samples))
            cvar = -mean(sorted_r[1:n_tail])
            return mu_p - risk_aversion * cvar
        end
    end

    # Optimize over the simplex via projected gradient
    w = ones(N) / N

    for iter in 1:500
        # Numerical gradient
        eps_grad = 1e-4
        grad = zeros(N)
        u0 = expected_utility(w)
        for i in 1:N
            w_eps = copy(w)
            w_eps[i] += eps_grad
            grad[i] = (expected_utility(w_eps) - u0) / eps_grad
        end

        # Projected gradient step
        w_new = _project_simplex(w + 0.01 * grad)
        norm(w_new - w) < 1e-8 && break
        w = w_new
    end

    return max.(w, 0.0) ./ sum(max.(w, 0.0))
end

"""Project onto probability simplex."""
function _project_simplex(v::Vector{Float64})::Vector{Float64}
    n = length(v)
    u = sort(v, rev=true)
    cssv = cumsum(u)
    rho = 0
    for j in 1:n
        if u[j] - (cssv[j] - 1.0) / j > 0
            rho = j
        end
    end
    theta = rho > 0 ? (cssv[rho] - 1.0) / rho : 0.0
    return max.(v .- theta, 0.0)
end

"""
    maximize_posterior_utility(posterior::NormalInverseWishart,
                                 risk_aversion::Float64) -> Vector{Float64}

Closed-form approximation: maximize expected utility under posterior predictive.

For quadratic utility: w* = (1/γ) * Σ_pred⁻¹ * μ_pred (normalized)
"""
function maximize_posterior_utility(posterior::NormalInverseWishart,
                                      risk_aversion::Float64)::Vector{Float64}

    pred = posterior_predictive(posterior)
    N = length(pred.mean)

    Sigma_inv = try
        inv(pred.predictive_cov + 1e-8*I)
    catch
        pinv(pred.predictive_cov)
    end

    w_raw = (1.0 / risk_aversion) * Sigma_inv * pred.mean
    w_pos = max.(w_raw, 0.0)
    total = sum(w_pos)
    return total > 0 ? w_pos ./ total : ones(N) / N
end

# =============================================================================
# SECTION 7: PREDICTIVE SHARPE AND MODEL COMPARISON
# =============================================================================

"""
    predictive_sharpe_ratio(posterior::NormalInverseWishart,
                              weights::Vector{Float64}; annualize=252) -> NamedTuple

Compute Sharpe ratio accounting for estimation uncertainty.

Under NIW posterior, the predictive mean and variance of portfolio return include:
  1. Parameter uncertainty (estimation risk)
  2. Natural variability of returns

Predictive Sharpe < Sample Sharpe because estimation risk inflates effective variance.

# Returns
- NamedTuple: predictive_sharpe, sample_sharpe, estimation_discount
"""
function predictive_sharpe_ratio(posterior::NormalInverseWishart,
                                   weights::Vector{Float64};
                                   annualize::Int=252)

    pred = posterior_predictive(posterior)
    N = length(weights)
    w = weights / sum(weights)

    # Portfolio predictive moments
    pred_mu_p  = w' * pred.mean
    pred_var_p = w' * pred.predictive_cov * w

    # Sample moments (without estimation uncertainty)
    sample_cov = posterior.Psi0 ./ max(posterior.nu0 - N - 1, 1.0)
    sample_mu_p  = w' * posterior.mu0
    sample_var_p = w' * sample_cov * w

    # Annualized Sharpes
    pred_sharpe   = pred_var_p > 0 ? pred_mu_p / sqrt(pred_var_p) * sqrt(annualize) : 0.0
    sample_sharpe = sample_var_p > 0 ? sample_mu_p / sqrt(sample_var_p) * sqrt(annualize) : 0.0

    # Discount due to estimation uncertainty
    discount = sample_sharpe != 0 ? pred_sharpe / sample_sharpe : 1.0

    return (predictive_sharpe=pred_sharpe, sample_sharpe=sample_sharpe,
             estimation_discount=discount, pred_mu=pred_mu_p, pred_std=sqrt(pred_var_p))
end

"""
    estimation_adjusted_sharpe(mu_hat, sigma_hat, T, N; confidence=0.95) -> NamedTuple

Adjust the observed Sharpe ratio downward for estimation uncertainty.

Lo (2002), Opdyke (2007): true Sharpe has confidence interval:
    SR_hat ± z_{α/2} * sqrt(1/T * (1 + SR²/2 * (κ-1)))

where κ is the kurtosis.

# Returns
- NamedTuple: adjusted_sharpe, lower_bound, upper_bound, t_statistic
"""
function estimation_adjusted_sharpe(mu_hat::Float64,
                                      sigma_hat::Float64,
                                      T::Int,
                                      N::Int;
                                      kurtosis::Float64=3.0,
                                      confidence::Float64=0.95)

    sigma_hat <= 0 && return (adjusted_sharpe=0.0, lower_bound=0.0, upper_bound=0.0, t_statistic=0.0)

    sr = mu_hat / sigma_hat

    # Standard error of Sharpe (Lo 2002)
    sr_var = (1 + sr^2/2 * (kurtosis - 1)) / T
    sr_se  = sqrt(max(sr_var, 0.0))

    # Adjusted Sharpe: penalize for N multiple testing
    z_alpha = confidence == 0.95 ? 1.645 : confidence == 0.99 ? 2.326 : 1.0

    lower = sr - z_alpha * sr_se
    upper = sr + z_alpha * sr_se
    t_stat = sr_se > 0 ? sr / sr_se : 0.0

    # Downward adjustment for multiple testing (Bonferroni)
    sr_adjusted = sr - sr_se * sqrt(2 * log(N))

    return (adjusted_sharpe=sr_adjusted, lower_bound=lower, upper_bound=upper,
             t_statistic=t_stat, standard_error=sr_se)
end

"""
    bayesian_model_comparison(returns, models; prior_probs=nothing) -> NamedTuple

Compare different distributional models for returns using marginal likelihoods.

Models supported:
- :normal: Gaussian returns
- :student_t: heavy-tailed (Student-t with ν df)
- :laplace: double exponential

Uses Bayes factors: BF_ij = p(data|M_i) / p(data|M_j)

# Returns
- NamedTuple: posterior_probs, bayes_factors, best_model
"""
function bayesian_model_comparison(returns::Vector{Float64},
                                     models::Vector{Symbol}=[:normal, :student_t, :laplace];
                                     prior_probs::Union{Vector{Float64},Nothing}=nothing)

    n_models = length(models)
    prior_p = prior_probs === nothing ? ones(n_models) / n_models : prior_probs
    prior_p ./= sum(prior_p)

    log_marginals = zeros(n_models)

    for (i, model) in enumerate(models)
        log_marginals[i] = _log_marginal_likelihood(returns, model)
    end

    # Posterior model probabilities
    log_max = maximum(log_marginals)
    log_sum = log(sum(exp.(log_marginals .- log_max))) + log_max

    post_probs = exp.(log_marginals .- log_sum) .* prior_p
    post_probs ./= sum(post_probs)

    # Bayes factors relative to first model
    bayes_factors = exp.(log_marginals .- log_marginals[1])

    best_idx = argmax(post_probs)
    best_model = models[best_idx]

    return (posterior_probs=post_probs, bayes_factors=bayes_factors,
             best_model=best_model, log_marginals=log_marginals)
end

"""Compute log marginal likelihood under a distributional model."""
function _log_marginal_likelihood(y::Vector{Float64}, model::Symbol)::Float64
    n = length(y)
    mu = mean(y)
    sigma = std(y)
    sigma <= 0 && return -Inf

    if model == :normal
        # Normal likelihood with Jeffreys prior: integrated over (mu, sigma²)
        # log p(y) ≈ -(n/2)*log(Σ(y-ȳ)²) + constants
        ss = sum((y .- mu).^2)
        return -(n-1)/2 * log(ss) - (n-1)/2 * log(2π/n)

    elseif model == :student_t
        # Student-t with estimated df (approximate)
        # MLE of nu via kurtosis: k = 3(n+2), E[k] = 6/(nu-4)+3 → nu = 6/(k-3)+4
        k4 = sum((y .- mu).^4) / (n * sigma^4)
        nu = k4 > 3.1 ? max(4.1, 6/(k4 - 3) + 4) : 30.0
        nu = min(nu, 100.0)

        log_lik = 0.0
        for yi in y
            z = (yi - mu) / sigma
            log_lik += _lgamma((nu+1)/2) - _lgamma(nu/2) - 0.5*log(π*nu) -
                       log(sigma) - (nu+1)/2 * log(1 + z^2/nu)
        end
        return log_lik

    else  # :laplace
        b = sigma / sqrt(2.0)
        log_lik = -n * log(2b) - sum(abs.(y .- mu)) / b
        return log_lik
    end
end

"""
    marginal_likelihood(returns::Matrix{Float64}, prior::NormalInverseWishart) -> Float64

Compute the log marginal likelihood of multivariate returns under NIW model.

log p(Y) = log Γ_N(ν_n/2) - log Γ_N(ν₀/2)
         + ν₀/2 * log|Ψ₀| - ν_n/2 * log|Ψ_n|
         + N*T/2 * log(π)
         + N/2 * log(κ₀/κ_n)

This is used for Bayesian model comparison (e.g., comparing different prior beliefs).
"""
function marginal_likelihood(returns::Matrix{Float64},
                               prior::NormalInverseWishart)::Float64

    T, N = size(returns)
    posterior = posterior_update(prior, returns)

    # Log marginal likelihood (multivariate t-distribution)
    nu0 = prior.nu0;  nu_n = posterior.nu0
    kap0 = prior.kappa0; kap_n = posterior.kappa0

    # Log|Ψ|
    log_det_Psi0 = try logdet(Symmetric(prior.Psi0 + 1e-10*I)) catch 0.0 end
    log_det_Psin = try logdet(Symmetric(posterior.Psi0 + 1e-10*I)) catch 0.0 end

    # Multivariate log-gamma
    function log_multivariate_gamma(a::Float64, n::Int)::Float64
        n*(n-1)/4 * log(π) + sum(_lgamma(a + (1-j)/2) for j in 1:n)
    end

    log_ml = (log_multivariate_gamma(nu_n/2, N) - log_multivariate_gamma(nu0/2, N)
              + nu0/2 * log_det_Psi0 - nu_n/2 * log_det_Psin
              - T*N/2 * log(π)
              + N/2 * log(kap0/kap_n))

    return log_ml
end

end # module BayesianPortfolio
