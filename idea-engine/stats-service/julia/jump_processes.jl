# =============================================================================
# jump_processes.jl — Jump-Diffusion Models for Crypto
# =============================================================================
# Provides:
#   - MertonJumpDiffusion     Merton log-normal jump model (MLE + simulation)
#   - KouDoubleExp            Kou double-exponential jump model
#   - CompoundPoisson         Compound Poisson process simulation
#   - JumpDetection           Lee-Mykland test + bipower variation
#   - JumpSeparation          Realised variance vs bipower variation decomposition
#   - GARJI                   Jump-adjusted GARCH (GARCH with jumps)
#   - JumpDiffusionFFT        Option pricing via FFT (characteristic function)
#   - FlashCrashDetection     Crypto flash crash detection
#   - run_jump_processes       Top-level driver + JSON export
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, Random, JSON3
# =============================================================================

module JumpProcesses

using Statistics
using LinearAlgebra
using Random
using JSON3

export MertonJumpDiffusion, KouDoubleExp, CompoundPoisson
export JumpDetection, JumpSeparation, GARJI
export JumpDiffusionFFT, FlashCrashDetection
export run_jump_processes

# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Standard normal PDF."""
_φ(x) = exp(-0.5*x^2) / sqrt(2π)

"""Standard normal CDF."""
function _Φ(x::Float64)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))))
    p = 1.0 - _φ(x) * poly
    x >= 0.0 ? p : 1.0 - p
end

"""Log-sum-exp trick for numerically stable log(∑exp(a_i))."""
function _logsumexp(v::Vector{Float64})
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

"""Exponential random variate via inverse transform."""
_rexp(λ, rng) = -log(rand(rng)) / λ

"""Generate exponential random variates."""
function _rexpv(n::Int, λ::Float64, rng::AbstractRNG)
    [-log(rand(rng)) / λ for _ in 1:n]
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Merton Jump-Diffusion Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    MertonJumpDiffusion(returns; dt, n_terms, init) → NamedTuple

Fit the Merton (1976) jump-diffusion model to return data via MLE.

dS/S = (μ - λκ)dt + σ dW + (e^J - 1) dN_t
where J ~ N(μ_J, σ_J²) and N_t ~ Poisson(λ).

Log-returns: r_t = (μ - σ²/2 - λκ)dt + σ√dt·Z + J·P_t

# Arguments
- `returns`  : daily log-return vector
- `dt`       : time step (default 1/252)
- `n_terms`  : Poisson mixture terms (default 20)
- `init`     : optional (mu, sigma, lambda, mu_J, sigma_J) initial guess

# Returns
NamedTuple: (mu, sigma, lambda, mu_J, sigma_J, loglik, AIC, BIC, kappa)
"""
function MertonJumpDiffusion(returns::Vector{Float64}; dt::Float64=1/252,
                              n_terms::Int=20,
                              init::Union{Nothing, Tuple}=nothing)
    n = length(returns)
    mu0, sig0, lam0, muj0, sigj0 =
        if isnothing(init)
            m = mean(returns)/dt; s = std(returns)/sqrt(dt)
            (m, s*0.8, 5.0, mean(returns[abs.(returns) .> 2*std(returns)]; init=0.0),
             2*std(returns)/sqrt(dt))
        else
            init
        end

    θ = [mu0, max(sig0, 0.01), max(lam0, 0.1), muj0, max(sigj0, 0.005)]

    best_ll = _merton_loglik(returns, θ[1], θ[2], θ[3], θ[4], θ[5], dt, n_terms)
    best_θ  = copy(θ)

    # Gradient ascent with adaptive step
    step = [1e-4, 1e-5, 0.1, 1e-4, 1e-5]
    for iter in 1:2000
        improved = false
        for k in 1:5
            for δ in [step[k], -step[k]]
                θ_try = copy(best_θ)
                θ_try[k] += δ
                _merton_valid(θ_try) || continue
                ll = _merton_loglik(returns, θ_try..., dt, n_terms)
                if ll > best_ll
                    best_ll = ll; best_θ = copy(θ_try)
                    improved = true
                end
            end
        end
        improved ? (step .*= 1.05) : (step .*= 0.95)
        maximum(step) < 1e-10 && break
    end

    mu, sigma, lambda, mu_J, sigma_J = best_θ
    kappa = exp(mu_J + 0.5*sigma_J^2) - 1.0   # mean jump size - 1
    n_params = 5
    results = (mu=mu, sigma=sigma, lambda=lambda, mu_J=mu_J, sigma_J=sigma_J,
               loglik=best_ll, kappa=kappa,
               AIC=-2best_ll + 2n_params, BIC=-2best_ll + n_params*log(n))
    return results
end

function _merton_valid(θ)
    mu, sigma, lambda, mu_J, sigma_J = θ
    return sigma > 0.001 && lambda > 0.01 && sigma_J > 0.001
end

"""Merton log-likelihood (Poisson mixture of normals)."""
function _merton_loglik(returns::Vector{Float64}, mu::Float64, sigma::Float64,
                         lambda::Float64, mu_J::Float64, sigma_J::Float64,
                         dt::Float64, n_terms::Int)::Float64
    kappa = exp(mu_J + 0.5*sigma_J^2) - 1.0
    drift = (mu - 0.5*sigma^2 - lambda*kappa) * dt
    ll = 0.0
    # Precompute Poisson weights log P(N=k)
    log_pois = zeros(n_terms+1)
    ldt = lambda * dt
    log_pois[1] = -ldt   # P(N=0) = exp(-λdt)
    for k in 1:n_terms
        log_pois[k+1] = log_pois[k] + log(ldt) - log(Float64(k))
    end

    for r in returns
        # log p(r) = log ∑_k P(N=k) · φ_k(r)
        terms = zeros(n_terms+1)
        for k in 0:n_terms
            mu_k = drift + k * mu_J
            sigma2_k = sigma^2 * dt + k * sigma_J^2
            sigma2_k = max(sigma2_k, 1e-10)
            terms[k+1] = log_pois[k+1] - 0.5*log(2π*sigma2_k) -
                          0.5*(r - mu_k)^2 / sigma2_k
        end
        ll += _logsumexp(terms)
    end
    return ll
end

"""
Simulate Merton jump-diffusion paths.

# Arguments
- `params`  : fitted NamedTuple from MertonJumpDiffusion
- `T`       : horizon in years
- `n_steps` : number of time steps
- `n_paths` : number of paths
"""
function simulate_merton(params::NamedTuple, T::Float64, n_steps::Int, n_paths::Int;
                          rng::AbstractRNG=Random.default_rng())::Matrix{Float64}
    dt = T / n_steps
    mu, sigma, lambda = params.mu, params.sigma, params.lambda
    mu_J, sigma_J = params.mu_J, params.sigma_J
    kappa = exp(mu_J + 0.5*sigma_J^2) - 1.0

    log_S = zeros(n_paths, n_steps+1)  # store log-price paths
    for t in 2:(n_steps+1)
        Z = randn(rng, n_paths)
        # Diffusion component
        diff = (mu - 0.5*sigma^2 - lambda*kappa)*dt .+ sigma*sqrt(dt)*Z
        # Jump component
        n_jumps = [rand(rng) < lambda*dt ? 1 : 0 for _ in 1:n_paths]
        jump_sizes = [nj > 0 ? mu_J + sigma_J*randn(rng) : 0.0 for nj in n_jumps]
        log_S[:, t] = log_S[:, t-1] .+ diff .+ jump_sizes
    end
    return exp.(log_S)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Kou Double-Exponential Jump Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    KouDoubleExp(returns; dt, init) → NamedTuple

Fit the Kou (2002) double-exponential jump-diffusion model via MLE.

Jump sizes have asymmetric double-exponential distribution:
f_J(y) = p·η₁·exp(-η₁·y)·1_{y≥0} + (1-p)·η₂·exp(η₂·y)·1_{y<0}

Provides closed-form option prices and realistic asymmetric jump behaviour.

# Arguments
- `returns` : daily log-return vector
- `dt`      : time step (default 1/252)
- `init`    : (mu, sigma, lambda, p, eta1, eta2) or nothing

# Returns
NamedTuple: (mu, sigma, lambda, p, eta1, eta2, loglik, AIC)
"""
function KouDoubleExp(returns::Vector{Float64}; dt::Float64=1/252,
                       init::Union{Nothing,Tuple}=nothing)
    n = length(returns)
    pos_ret = filter(r -> r > 0, returns)
    neg_ret = filter(r -> r < 0, returns)

    mu0   = mean(returns) / dt
    sig0  = std(returns) / sqrt(dt) * 0.8
    lam0  = 10.0
    p0    = 0.4
    eta1  = length(pos_ret) > 0 ? 1.0 / max(mean(pos_ret), 0.001) : 50.0
    eta2  = length(neg_ret) > 0 ? 1.0 / max(-mean(neg_ret), 0.001) : 50.0

    θ = isnothing(init) ? [mu0, sig0, lam0, p0, eta1, eta2] :
                           collect(Float64, init)

    best_ll = _kou_loglik(returns, θ..., dt)
    best_θ  = copy(θ)

    step = [1e-3, 1e-4, 0.1, 0.01, 1.0, 1.0]
    for iter in 1:3000
        improved = false
        for k in 1:6
            for δ in [step[k], -step[k]]
                θ_try = copy(best_θ)
                θ_try[k] += δ
                _kou_valid(θ_try) || continue
                ll = _kou_loglik(returns, θ_try..., dt)
                if ll > best_ll
                    best_ll = ll; best_θ = copy(θ_try); improved = true
                end
            end
        end
        improved ? (step .*= 1.05) : (step .*= 0.92)
        maximum(step) < 1e-11 && break
    end

    mu, sigma, lambda, p, eta1f, eta2f = best_θ
    # Mean jump size under Kou: E[J] = p/η₁ - (1-p)/η₂
    mean_jump = p/eta1f - (1.0-p)/eta2f

    return (mu=mu, sigma=sigma, lambda=lambda, p=p, eta1=eta1f, eta2=eta2f,
            loglik=best_ll, mean_jump=mean_jump,
            AIC=-2best_ll + 12, BIC=-2best_ll + 6*log(n))
end

function _kou_valid(θ)
    mu, sigma, lambda, p, eta1, eta2 = θ
    return sigma > 0.001 && lambda > 0.01 && 0 < p < 1 && eta1 > 1 && eta2 > 1
end

"""Kou model log-likelihood (semi-analytical via convolution approximation)."""
function _kou_loglik(returns::Vector{Float64}, mu::Float64, sigma::Float64,
                      lambda::Float64, p::Float64, eta1::Float64, eta2::Float64,
                      dt::Float64)::Float64
    n_terms = 15
    # Mean jump: E[J] = p/η₁ - (1-p)/η₂
    mean_J = p/eta1 - (1.0-p)/eta2
    # E[e^J] = p*η₁/(η₁-1) + (1-p)*η₂/(η₂+1)
    # (needs η₁ > 1 and η₂ > 1 for finiteness)
    e_eJ = eta1 > 1 ? p*eta1/(eta1-1.0) : 2.0
    e_eJ += eta2 > 1 ? (1.0-p)*eta2/(eta2+1.0) : 1.5
    drift = (mu - 0.5*sigma^2 - lambda*(e_eJ - 1.0)) * dt

    ldt = lambda * dt
    log_pois = zeros(n_terms+1)
    log_pois[1] = -ldt
    for k in 1:n_terms
        log_pois[k+1] = log_pois[k] + log(ldt) - log(Float64(k))
    end

    ll = 0.0
    for r in returns
        terms = zeros(n_terms+1)
        # k=0: pure Gaussian
        sig2_0 = sigma^2 * dt
        terms[1] = log_pois[1] - 0.5*log(2π*sig2_0) - 0.5*(r-drift)^2/sig2_0
        # k>=1: sum over k jumps drawn from Kou distribution (via moment matching)
        for k in 1:n_terms
            mu_k   = drift + k * mean_J
            sigma2_k = sigma^2*dt + k*(p/eta1^2 + (1-p)/eta2^2)  # variance approx
            sigma2_k = max(sigma2_k, 1e-10)
            terms[k+1] = log_pois[k+1] - 0.5*log(2π*sigma2_k) -
                          0.5*(r - mu_k)^2 / sigma2_k
        end
        ll += _logsumexp(terms)
    end
    return ll
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compound Poisson Process Simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    CompoundPoisson(lambda_cpp, jump_dist, T; n_paths, rng) → NamedTuple

Simulate a compound Poisson process N_t with specified jump distribution.

# Arguments
- `lambda_cpp` : Poisson intensity (average jumps per unit time)
- `jump_dist`  : function `rng → Float64` generating jump sizes
- `T`          : time horizon
- `n_paths`    : number of sample paths (default 1000)

# Returns
NamedTuple: (paths, jump_times, jump_sizes, E_terminal, Var_terminal)
"""
function CompoundPoisson(lambda_cpp::Float64, jump_dist::Function, T::Float64;
                          n_paths::Int=1000, n_steps::Int=252,
                          rng::AbstractRNG=Random.default_rng())
    dt = T / n_steps
    paths = zeros(n_paths, n_steps+1)

    all_jump_times = Vector{Vector{Float64}}()
    all_jump_sizes = Vector{Vector{Float64}}()

    for i in 1:n_paths
        jt = Float64[]; js = Float64[]
        t = 0.0
        while true
            t += _rexp(lambda_cpp, rng)
            t > T && break
            push!(jt, t)
            push!(js, jump_dist(rng))
        end
        push!(all_jump_times, jt)
        push!(all_jump_sizes, js)

        # Build path by accumulating jumps
        for s in 1:n_steps
            t_s = s * dt
            paths[i, s+1] = sum(js[k] for k in eachindex(jt) if jt[k] <= t_s; init=0.0)
        end
    end

    terminal = paths[:, end]
    return (paths=paths, jump_times=all_jump_times, jump_sizes=all_jump_sizes,
            E_terminal=mean(terminal), Var_terminal=var(terminal),
            lambda=lambda_cpp, T=T)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Jump Detection: Lee-Mykland Test
# ─────────────────────────────────────────────────────────────────────────────

"""
    JumpDetection(prices; window, alpha) → NamedTuple

Detect jumps in a return series using the Lee-Mykland (2008) test statistic.

L_t = r_t / (BV^{1/2}_{t-window:t-1} / sqrt(dt))

where BV is bipower variation. Jumps flagged when |L_t| > threshold.

# Arguments
- `prices`  : price vector (or log-prices)
- `window`  : rolling window for BV estimation (default 20)
- `alpha`   : significance level (default 0.01)

# Returns
NamedTuple: (jump_times, jump_magnitudes, test_stats, threshold, n_jumps)
"""
function JumpDetection(prices::Vector{Float64}; window::Int=20, alpha::Float64=0.01)
    n = length(prices)
    returns = diff(log.(max.(prices, 1e-10)))
    m = length(returns)

    # Bipower variation (BV) over rolling window
    BV = zeros(m)
    for t in (window+1):m
        window_rets = returns[(t-window):(t-1)]
        BV[t] = (π/2) * mean(abs(window_rets[i]) * abs(window_rets[i-1])
                               for i in 2:length(window_rets))
        BV[t] = max(BV[t], 1e-10)
    end
    BV[1:window] .= var(returns[1:window])

    # Lee-Mykland test statistic
    dt = 1.0/252.0
    L = zeros(m)
    for t in (window+1):m
        L[t] = returns[t] / sqrt(BV[t])
    end

    # Critical value from extreme value theory (GEV for max of |L_t|)
    c_n = sqrt(2 * log(window))
    S_n = c_n - (log(π) + log(log(window))) / (2 * c_n)
    # Threshold: S_n - log(-log(1-alpha)) / c_n
    threshold = S_n - log(-log(1.0 - alpha)) / c_n

    jump_mask = abs.(L) .> threshold
    jump_times = findall(jump_mask)
    jump_magnitudes = returns[jump_times]

    return (jump_times=jump_times, jump_magnitudes=jump_magnitudes,
            test_stats=L, threshold=threshold, n_jumps=sum(jump_mask),
            BV=BV)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Realised Variance vs Bipower Variation
# ─────────────────────────────────────────────────────────────────────────────

"""
    JumpSeparation(returns; freq) → NamedTuple

Decompose realised variance into diffusion and jump components using
bipower variation (Barndorff-Nielsen & Shephard, 2004).

RV = IV + JV  (Realised Variance = Integrated Variance + Jump Variation)
IV ≈ BV = (π/2) ∑ |r_i||r_{i-1}|

# Arguments
- `returns`  : high-frequency return vector
- `freq`     : sampling frequency per day (e.g. 78 for 5-min returns)

# Returns
NamedTuple: (RV, BV, JV, jump_ratio, Z_stat, significant_jumps)
"""
function JumpSeparation(returns::Vector{Float64}; freq::Int=78)
    n = length(returns)
    n < 2 && error("Need at least 2 observations")

    # Realised Variance
    RV = sum(r^2 for r in returns)

    # Bipower Variation
    BV = (π/2) * sum(abs(returns[i]) * abs(returns[i-1]) for i in 2:n)

    # Jump Variation estimate
    JV = max(RV - BV, 0.0)
    jump_ratio = JV / max(RV, 1e-10)

    # BNS test statistic for significant jump variation
    # Z = (RV - BV) / sqrt(variance_of_difference)
    # Asymptotic variance ≈ ((π/2)² + π - 5) * (1/n) * TP
    # where TP = tripower variation
    TP = (2^(2/3) * gamma(7/6) / gamma(0.5))^3 *
         sum(abs(returns[i])^(2/3) * abs(returns[i-1])^(2/3) * abs(returns[max(i-2,1)])^(2/3)
             for i in 3:n)
    TP = max(TP, 1e-10)
    asymp_var = ((π/2)^2 + π - 5) / n * TP / BV^2
    asymp_var = max(asymp_var, 1e-10)

    Z_stat = (1.0 - BV/RV) / sqrt(asymp_var)

    # Significant at 99%: |Z| > 2.576
    significant_jumps = abs(Z_stat) > 2.576

    return (RV=RV, BV=BV, JV=JV, jump_ratio=jump_ratio,
            Z_stat=Z_stat, significant_jumps=significant_jumps,
            n_obs=n, freq=freq)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. GARJI Model (GARCH with Autoregressive Jump Intensity)
# ─────────────────────────────────────────────────────────────────────────────

"""
    GARJI(returns; dt, max_iter) → NamedTuple

Fit a GARJI (GARCH with autoregressive jump intensity) model.
Combines GARCH(1,1) volatility with time-varying Poisson intensity.

λ_t = λ_0 + α_J · J_{t-1} + β_J · λ_{t-1}
h_t = ω + α_G r²_{t-1} + β_G h_{t-1}

# Arguments
- `returns`  : daily return vector
- `dt`       : time step (default 1/252)
- `max_iter` : maximum gradient-ascent iterations (default 2000)

# Returns
NamedTuple: (omega, alpha_g, beta_g, lambda0, alpha_J, beta_J,
             mu_J, sigma_J, loglik, filtered_intensity, filtered_variance)
"""
function GARJI(returns::Vector{Float64}; dt::Float64=1/252, max_iter::Int=2000)
    n = length(returns)

    # Initial parameters
    θ0 = [mean(returns)/dt,         # mu
           var(returns)*0.1,         # omega
           0.10,                     # alpha_G
           0.85,                     # beta_G
           5.0,                      # lambda0 (avg jumps/year)
           0.3,                      # alpha_J (jump propagation)
           0.5,                      # beta_J (intensity persistence)
           0.0,                      # mu_J
           0.03]                     # sigma_J

    function valid(θ)
        mu, ω, αg, βg, λ0, αJ, βJ, μJ, σJ = θ
        return ω > 1e-8 && αg > 0 && βg > 0 && αg+βg < 0.999 &&
               λ0 > 0.1 && αJ ≥ 0 && βJ ≥ 0 && αJ+βJ < 0.999 &&
               σJ > 0.001
    end

    function loglik(θ)
        mu, ω, αg, βg, λ0, αJ, βJ, μJ, σJ = θ
        h = var(returns)
        λ = λ0
        ll = 0.0
        for t in 1:n
            r = returns[t]
            ldt = λ * dt
            # Mixture: P(r|θ) ≈ P(N=0)·φ_0 + P(N=1)·φ_1 + ...
            terms = zeros(8)
            log_p0 = -ldt
            terms[1] = log_p0 - 0.5*log(2π*h) - 0.5*(r - mu*dt)^2/h
            for k in 1:7
                lp_k = log_p0 + k*log(ldt) - sum(log(j) for j in 1:k)
                sigma2_k = max(h + k*σJ^2, 1e-10)
                mu_k = mu*dt + k*μJ
                terms[k+1] = lp_k - 0.5*log(2π*sigma2_k) - 0.5*(r-mu_k)^2/sigma2_k
            end
            ll += _logsumexp(terms)
            # Update h (GARCH)
            h = max(ω + αg*r^2 + βg*h, 1e-10)
            # Update λ (ARJI): use E[N_t|r_t] as proxy for J_{t-1}
            e_N = λ * dt  # simplified (full ARJI uses filtered J)
            λ = max(λ0 + αJ*e_N/dt + βJ*λ, 0.1)
        end
        return ll
    end

    best_ll = loglik(θ0)
    best_θ  = copy(θ0)

    step = [1e-4, 1e-7, 0.005, 0.005, 0.1, 0.01, 0.01, 1e-4, 0.001]
    for iter in 1:max_iter
        improved = false
        for k in 1:9
            for δ in [step[k], -step[k]]
                θ_try = copy(best_θ)
                θ_try[k] += δ
                valid(θ_try) || continue
                ll = loglik(θ_try)
                if ll > best_ll
                    best_ll = ll; best_θ = copy(θ_try); improved = true
                end
            end
        end
        improved ? (step .*= 1.05) : (step .*= 0.9)
        maximum(step) < 1e-12 && break
    end

    mu, ω, αg, βg, λ0, αJ, βJ, μJ, σJ = best_θ
    # Filtered series
    filt_h = zeros(n); filt_λ = zeros(n)
    h = var(returns); λ = λ0
    for t in 1:n
        filt_h[t] = h; filt_λ[t] = λ
        r = returns[t]
        h = max(ω + αg*r^2 + βg*h, 1e-10)
        e_N = λ*dt
        λ = max(λ0 + αJ*e_N/dt + βJ*λ, 0.1)
    end

    return (mu=mu, omega=ω, alpha_g=αg, beta_g=βg, lambda0=λ0,
            alpha_J=αJ, beta_J=βJ, mu_J=μJ, sigma_J=σJ,
            loglik=best_ll, filtered_variance=filt_h,
            filtered_intensity=filt_λ)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Option Pricing via FFT (Merton characteristic function)
# ─────────────────────────────────────────────────────────────────────────────

"""
    JumpDiffusionFFT(params, S0, r_rf, T, K_vec; n_fft) → NamedTuple

Price European call options under the Merton jump-diffusion model
using the FFT method (Carr & Madan, 1999).

Characteristic function:
φ(u) = exp{iuT[r - σ²/2 - λκ] - u²σ²T/2 + λT[exp(iuμ_J - u²σ_J²/2) - 1]}

# Arguments
- `params`  : fitted Merton NamedTuple
- `S0`      : current spot price
- `r_rf`    : risk-free rate
- `T`       : time to expiry (years)
- `K_vec`   : vector of strike prices

# Returns
NamedTuple: (call_prices, put_prices, K_vec, delta, implied_vol_approx)
"""
function JumpDiffusionFFT(params::NamedTuple, S0::Float64, r_rf::Float64,
                           T::Float64, K_vec::Vector{Float64}; n_fft::Int=1024)
    mu, sigma, lambda = params.mu, params.sigma, params.lambda
    mu_J, sigma_J = params.mu_J, params.sigma_J
    kappa = exp(mu_J + 0.5*sigma_J^2) - 1.0

    # FFT grid
    N = n_fft
    eta = 0.25        # spacing in frequency domain
    lambda_f = 2π / (N * eta)  # spacing in log-strike domain
    b = N * lambda_f / 2.0     # shift

    # Frequency grid
    v = eta * (0:(N-1))
    # Log-strike grid
    k_grid = -b .+ lambda_f * (0:(N-1))

    # Damping factor α (ensures integrability)
    alpha_damp = 1.5

    # Characteristic function of log(S_T/S_0)
    function cf(u::ComplexF64)::ComplexF64
        iu = im * u
        # Merton CF
        return exp(iu * (log(S0) + (r_rf - 0.5*sigma^2 - lambda*kappa)*T) -
                   0.5 * sigma^2 * T * u^2 +
                   lambda * T * (exp(iu*mu_J - 0.5*sigma_J^2*u^2) - 1.0))
    end

    # Carr-Madan integrand
    psi = zeros(ComplexF64, N)
    for j in 1:N
        v_j = v[j]
        u = v_j - (alpha_damp + 1)*im
        ψ = exp(-r_rf * T) * cf(u) /
            (alpha_damp^2 + alpha_damp - v_j^2 + im*(2*alpha_damp + 1)*v_j)
        # Simpson weights
        w = j == 1 ? 1/3.0 : (iseven(j) ? 4/3.0 : 2/3.0)
        psi[j] = exp(im * v_j * b) * ψ * eta * w
    end

    # FFT
    fft_result = _fft(psi)

    # Extract call prices at log-strike grid
    call_prices = zeros(length(K_vec))
    put_prices  = zeros(length(K_vec))
    for (m, K) in enumerate(K_vec)
        k = log(K)
        # Find nearest grid point
        j = clamp(round(Int, (k + b) / lambda_f) + 1, 1, N)
        C = real(exp(-alpha_damp * k) / π * fft_result[j])
        C = max(C, S0 - K * exp(-r_rf * T))  # call floor
        call_prices[m] = C
        # Put-call parity
        put_prices[m] = C - S0 + K * exp(-r_rf * T)
        put_prices[m] = max(put_prices[m], 0.0)
    end

    # Approximate Black-Scholes implied vol (Newton)
    impl_vols = [_bs_implied_vol(call_prices[m], S0, K_vec[m], r_rf, T)
                 for m in 1:length(K_vec)]

    # Approximate delta (finite difference)
    delta = zeros(length(K_vec))
    dS = S0 * 0.001
    params_up = (mu=mu, sigma=sigma, lambda=lambda, mu_J=mu_J, sigma_J=sigma_J, kappa=kappa)
    # Delta approximation: (C(S+dS) - C(S-dS)) / (2dS)
    for m in 1:length(K_vec)
        K = K_vec[m]
        d1 = (log(S0/K) + (r_rf + 0.5*sigma^2)*T) / (sigma*sqrt(T))
        delta[m] = _Φ(d1)  # BS delta as approximation
    end

    return (call_prices=call_prices, put_prices=put_prices, K_vec=K_vec,
            implied_vols=impl_vols, delta=delta, T=T, S0=S0)
end

"""Simple FFT (Cooley-Tukey, power-of-2)."""
function _fft(x::Vector{ComplexF64})::Vector{ComplexF64}
    n = length(x)
    n == 1 && return x
    if n & (n-1) != 0
        # Not power of 2 — use DFT directly
        N = n
        y = zeros(ComplexF64, N)
        for k in 0:(N-1), j in 0:(N-1)
            y[k+1] += x[j+1] * exp(-2π*im*k*j/N)
        end
        return y
    end
    # Recursive FFT
    even = _fft(x[1:2:end])
    odd  = _fft(x[2:2:end])
    T_arr = [exp(-2π*im*(k-1)/n) * odd[k] for k in 1:(n÷2)]
    return [even[k] + T_arr[k] for k in 1:(n÷2)] |> v ->
           vcat(v, [even[k] - T_arr[k] for k in 1:(n÷2)])
end

"""Black-Scholes implied volatility via Newton-Raphson."""
function _bs_implied_vol(C_mkt::Float64, S::Float64, K::Float64,
                          r::Float64, T::Float64)::Float64
    C_mkt <= 0.0 && return NaN
    σ = 0.3  # initial guess
    for _ in 1:50
        d1 = (log(S/K) + (r + 0.5σ^2)*T) / (σ*sqrt(T))
        d2 = d1 - σ*sqrt(T)
        C_bs = S*_Φ(d1) - K*exp(-r*T)*_Φ(d2)
        vega = S * _φ(d1) * sqrt(T)
        vega < 1e-10 && break
        σ -= (C_bs - C_mkt) / vega
        σ = clamp(σ, 0.001, 5.0)
        abs(C_bs - C_mkt) < 1e-8 && break
    end
    return σ
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Flash Crash Detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    FlashCrashDetection(prices; threshold_z, min_recovery, window) → NamedTuple

Detect crypto flash crashes from high-frequency price data.
A flash crash is identified by:
1. Large negative return (|r| > threshold_z × rolling σ)
2. Fast partial recovery (recovery ratio within min_recovery periods)
3. Unusual jump magnitude distribution (tail probability)

# Arguments
- `prices`        : price vector (intraday or daily)
- `threshold_z`   : z-score threshold for crash (default 4.0)
- `min_recovery`  : look-ahead window for recovery (default 10)
- `window`        : rolling vol window (default 50)

# Returns
NamedTuple: (crash_times, crash_magnitudes, recovery_ratios,
             jump_tail_probs, severity_scores)
"""
function FlashCrashDetection(prices::Vector{Float64}; threshold_z::Float64=4.0,
                              min_recovery::Int=10, window::Int=50)
    n = length(prices)
    returns = diff(log.(max.(prices, 1e-10)))
    m = length(returns)

    rolling_mu  = zeros(m)
    rolling_sig = zeros(m)
    for t in (window+1):m
        w = returns[(t-window):(t-1)]
        rolling_mu[t]  = mean(w)
        rolling_sig[t] = std(w)
    end
    rolling_mu[1:window]  .= mean(returns[1:window])
    rolling_sig[1:window] .= std(returns[1:window])

    crash_times      = Int[]
    crash_magnitudes = Float64[]
    recovery_ratios  = Float64[]
    jump_tail_probs  = Float64[]
    severity_scores  = Float64[]

    for t in (window+1):(m - min_recovery)
        # Standardised return
        z_t = (returns[t] - rolling_mu[t]) / max(rolling_sig[t], 1e-8)
        z_t > -threshold_z && continue  # only negative crashes

        push!(crash_times, t)
        push!(crash_magnitudes, returns[t])

        # Recovery: max recovery in next min_recovery steps
        future_ret = sum(returns[(t+1):min(t+min_recovery, m)])
        recovery = future_ret / max(abs(returns[t]), 1e-10)
        push!(recovery_ratios, recovery)

        # Tail probability (Gaussian): P(Z < z_t)
        tail_p = _Φ(z_t)
        push!(jump_tail_probs, tail_p)

        # Severity score: |z_score| * (1 - recovery) * log(1 + n_concurrent)
        n_concurrent = sum(abs(returns[s] - rolling_mu[s]) / max(rolling_sig[s],1e-8) > 2
                           for s in max(1,t-5):min(m,t+5))
        severity = abs(z_t) * max(1.0 - recovery, 0.0) * log1p(n_concurrent)
        push!(severity_scores, severity)
    end

    # Jump magnitude distribution fitting (for detected crashes)
    jump_dist_params = if length(crash_magnitudes) >= 5
        m_crash = mean(crash_magnitudes)
        s_crash = std(crash_magnitudes)
        # Fit GEV shape parameter via probability-weighted moments
        xi_hat = _evt_shape_estimator(crash_magnitudes)
        (mean=m_crash, std=s_crash, shape=xi_hat)
    else
        (mean=0.0, std=0.0, shape=0.0)
    end

    return (crash_times=crash_times, crash_magnitudes=crash_magnitudes,
            recovery_ratios=recovery_ratios, jump_tail_probs=jump_tail_probs,
            severity_scores=severity_scores, n_crashes=length(crash_times),
            jump_distribution=jump_dist_params)
end

"""Simple GEV shape (tail index) estimator via Hill estimator."""
function _evt_shape_estimator(x::Vector{Float64})::Float64
    length(x) < 3 && return 0.0
    sorted_abs = sort(abs.(x), rev=true)
    k = max(2, length(sorted_abs) ÷ 3)
    # Hill estimator: γ = (1/k) ∑ log(x_{(i)}/x_{(k+1)})
    if sorted_abs[k+1] <= 0.0
        return 0.0
    end
    hill = mean(log(sorted_abs[i]) - log(sorted_abs[k+1]) for i in 1:k)
    return 1.0 / max(hill, 1e-3)  # tail index (inverse of Hill)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-Level Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_jump_processes(prices; out_path) → Dict

Full jump-process analysis pipeline for crypto price data.

# Arguments
- `prices`   : price vector (daily OHLC close, or intraday)
- `out_path` : optional JSON export path

# Returns
Dict with all fitted models, detected jumps, and option prices.

# Example
```julia
using Random
rng = Random.MersenneTwister(42)
n = 500
prices = cumsum(randn(rng, n) .* 100.0) .+ 50_000.0
prices = max.(prices, 1.0)
results = run_jump_processes(prices)
println("Merton λ = ", results["merton"]["lambda"], " jumps/year")
println("Jumps detected: ", results["lee_mykland"]["n_jumps"])
```
"""
function run_jump_processes(prices::Vector{Float64};
                             out_path::Union{String,Nothing}=nothing)
    n = length(prices)
    log_returns = diff(log.(max.(prices, 1e-10)))
    rng = Random.default_rng()

    results = Dict{String, Any}()

    # ── Merton Jump-Diffusion ──────────────────────────────────────────────
    @info "Fitting Merton jump-diffusion model..."
    merton = MertonJumpDiffusion(log_returns; dt=1/252, n_terms=15)
    results["merton"] = Dict(
        "mu"=>"$(merton.mu)", "sigma"=>merton.sigma, "lambda"=>merton.lambda,
        "mu_J"=>merton.mu_J, "sigma_J"=>merton.sigma_J, "kappa"=>merton.kappa,
        "loglik"=>merton.loglik, "AIC"=>merton.AIC, "BIC"=>merton.BIC
    )

    # ── Kou Double-Exponential ─────────────────────────────────────────────
    @info "Fitting Kou double-exponential model..."
    kou = KouDoubleExp(log_returns; dt=1/252)
    results["kou"] = Dict(
        "mu"=>kou.mu, "sigma"=>kou.sigma, "lambda"=>kou.lambda,
        "p"=>kou.p, "eta1"=>kou.eta1, "eta2"=>kou.eta2,
        "mean_jump"=>kou.mean_jump, "loglik"=>kou.loglik, "AIC"=>kou.AIC
    )

    # ── Compound Poisson Simulation ────────────────────────────────────────
    @info "Simulating compound Poisson process..."
    cpp = CompoundPoisson(merton.lambda, r -> merton.mu_J + merton.sigma_J*randn(r),
                           1.0; n_paths=200, n_steps=252, rng=rng)
    results["compound_poisson"] = Dict(
        "E_terminal"   => cpp.E_terminal,
        "Var_terminal" => cpp.Var_terminal,
        "lambda"       => cpp.lambda
    )

    # ── Jump Detection (Lee-Mykland) ───────────────────────────────────────
    @info "Detecting jumps (Lee-Mykland test)..."
    lm = JumpDetection(prices; window=20, alpha=0.01)
    results["lee_mykland"] = Dict(
        "n_jumps"         => lm.n_jumps,
        "threshold"       => lm.threshold,
        "jump_times"      => lm.jump_times,
        "jump_magnitudes" => lm.jump_magnitudes
    )

    # ── Jump Separation (BV) ───────────────────────────────────────────────
    @info "Jump separation via bipower variation..."
    js = JumpSeparation(log_returns; freq=1)
    results["jump_separation"] = Dict(
        "RV"                => js.RV,
        "BV"                => js.BV,
        "JV"                => js.JV,
        "jump_ratio"        => js.jump_ratio,
        "Z_stat"            => js.Z_stat,
        "significant_jumps" => js.significant_jumps
    )

    # ── GARJI ─────────────────────────────────────────────────────────────
    @info "Fitting GARJI (jump-GARCH) model..."
    garji = GARJI(log_returns; dt=1/252, max_iter=1000)
    results["garji"] = Dict(
        "omega"    => garji.omega, "alpha_g" => garji.alpha_g,
        "beta_g"   => garji.beta_g, "lambda0" => garji.lambda0,
        "alpha_J"  => garji.alpha_J, "beta_J" => garji.beta_J,
        "mu_J"     => garji.mu_J, "sigma_J" => garji.sigma_J,
        "loglik"   => garji.loglik
    )

    # ── Option Pricing (FFT) ───────────────────────────────────────────────
    @info "Pricing options via FFT (Merton CF)..."
    S0 = prices[end]
    K_vec = S0 .* [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    opts = JumpDiffusionFFT(merton, S0, 0.04, 0.25, K_vec; n_fft=256)
    results["options"] = Dict(
        "S0"           => S0,
        "K_vec"        => K_vec,
        "call_prices"  => opts.call_prices,
        "put_prices"   => opts.put_prices,
        "implied_vols" => opts.implied_vols
    )

    # ── Flash Crash Detection ──────────────────────────────────────────────
    @info "Detecting flash crashes..."
    fc = FlashCrashDetection(prices; threshold_z=3.5, min_recovery=5)
    results["flash_crashes"] = Dict(
        "n_crashes"      => fc.n_crashes,
        "crash_times"    => fc.crash_times,
        "crash_magnitudes" => fc.crash_magnitudes,
        "severity_scores"  => fc.severity_scores,
        "avg_recovery"   => isempty(fc.recovery_ratios) ? 0.0 : mean(fc.recovery_ratios)
    )

    # ── Model Comparison ───────────────────────────────────────────────────
    results["model_comparison"] = Dict(
        "preferred" => merton.AIC < kou.AIC ? "merton" : "kou",
        "AIC_merton" => merton.AIC,
        "AIC_kou"    => kou.AIC
    )

    if !isnothing(out_path)
        open(out_path, "w") do io
            JSON3.write(io, results)
        end
        @info "Results written to $out_path"
    end

    return results
end

end  # module JumpProcesses
