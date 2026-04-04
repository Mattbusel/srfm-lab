"""
Stochastic — Stochastic processes and simulation for quantitative finance.

Implements: GBM, GARCH(p,q), Heston SV, Merton Jump Diffusion,
Ornstein-Uhlenbeck, and Hawkes process. Full simulation + estimation for each.
"""
module Stochastic

using LinearAlgebra
using Statistics
using Distributions
using Optim
using Random

export GBM, GARCH, Heston, MertonJumpDiffusion, OrnsteinUhlenbeck, Hawkes
export simulate, fit, forecast, price_option, half_life

# ─────────────────────────────────────────────────────────────────────────────
# 1. Geometric Brownian Motion
# ─────────────────────────────────────────────────────────────────────────────

"""
    GBM(mu, sigma, S0)

Geometric Brownian Motion: dS = μS dt + σS dW.
"""
struct GBM
    mu::Float64
    sigma::Float64
    S0::Float64

    function GBM(mu, sigma, S0)
        @assert sigma > 0 "sigma must be positive"
        @assert S0 > 0    "S0 must be positive"
        new(mu, sigma, S0)
    end
end

"""
    simulate(p::GBM, T, n_steps, n_paths) → Matrix{Float64}

Returns n_paths × (n_steps+1) matrix of price paths (exact solution).
"""
function simulate(p::GBM, T::Float64, n_steps::Int, n_paths::Int;
                  rng::AbstractRNG=Random.default_rng())::Matrix{Float64}
    dt = T / n_steps
    drift = (p.mu - 0.5 * p.sigma^2) * dt
    vol   = p.sigma * sqrt(dt)

    paths = Matrix{Float64}(undef, n_paths, n_steps + 1)
    paths[:, 1] .= p.S0

    for t in 2:(n_steps + 1)
        z = randn(rng, n_paths)
        @. paths[:, t] = paths[:, t-1] * exp(drift + vol * z)
    end
    return paths
end

"""
    log_returns(paths::Matrix) → Matrix{Float64}

Compute log-returns from price paths.
"""
function log_returns(paths::Matrix{Float64})::Matrix{Float64}
    return log.(paths[:, 2:end] ./ paths[:, 1:end-1])
end

"""
    gbm_fit(prices::Vector{Float64}, dt::Float64) → GBM

Estimate GBM parameters from a price series using MLE.
"""
function fit(::Type{GBM}, prices::Vector{Float64}, dt::Float64=1.0/252)::GBM
    rets = diff(log.(prices))
    mu_log = mean(rets) / dt
    sigma  = std(rets) / sqrt(dt)
    mu     = mu_log + 0.5 * sigma^2
    return GBM(mu, sigma, prices[1])
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. GARCH(p,q)
# ─────────────────────────────────────────────────────────────────────────────

"""
    GARCH(omega, alpha, beta)

GARCH(p,q) model: σ²_t = ω + Σ αᵢ ε²_{t-i} + Σ βⱼ σ²_{t-j}.
"""
struct GARCH
    omega::Float64
    alpha::Vector{Float64}   # ARCH coefficients (length p)
    beta::Vector{Float64}    # GARCH coefficients (length q)

    function GARCH(omega, alpha, beta)
        @assert omega > 0
        @assert all(x -> x >= 0, alpha)
        @assert all(x -> x >= 0, beta)
        @assert sum(alpha) + sum(beta) < 1 "GARCH persistence must be < 1"
        new(omega, alpha, beta)
    end
end

"""
    garch_variance_series(g, residuals) → Vector{Float64}

Compute conditional variance series given residuals ε_t.
"""
function garch_variance_series(g::GARCH, residuals::Vector{Float64})::Vector{Float64}
    n = length(residuals)
    p = length(g.alpha)
    q = length(g.beta)
    h = zeros(Float64, n)

    # Unconditional variance as warmup
    var_uncond = g.omega / max(1 - sum(g.alpha) - sum(g.beta), 1e-8)
    h[1:max(p,q)] .= var_uncond

    for t in (max(p,q)+1):n
        h[t] = g.omega
        for i in 1:p
            h[t] += g.alpha[i] * residuals[t-i]^2
        end
        for j in 1:q
            h[t] += g.beta[j] * h[t-j]
        end
        h[t] = max(h[t], 1e-12)
    end
    return h
end

"""
    garch_loglik(params, returns, p, q) → Float64

Gaussian GARCH(p,q) log-likelihood (negated for minimisation).
"""
function garch_loglik(params::Vector{Float64}, returns::Vector{Float64},
                      p::Int, q::Int)::Float64
    n_params = 1 + p + q
    if length(params) != n_params
        return Inf
    end
    omega = params[1]
    alpha = params[2:1+p]
    beta  = params[2+p:end]

    if omega <= 0 || any(x -> x < 0, alpha) || any(x -> x < 0, beta)
        return Inf
    end
    if sum(alpha) + sum(beta) >= 1
        return Inf
    end

    g = GARCH(omega, alpha, beta)
    h = garch_variance_series(g, returns)
    n = length(returns)

    ll = 0.0
    for t in (max(p,q)+1):n
        ll += -0.5 * (log(2π) + log(h[t]) + returns[t]^2 / h[t])
    end
    return -ll
end

"""
    fit(returns, p, q) → GARCH

MLE estimation of GARCH(p,q) via Optim.jl (L-BFGS-B with bounds).
"""
function fit(returns::Vector{Float64}, p::Int=1, q::Int=1)::GARCH
    n_params = 1 + p + q
    var_ret  = var(returns)

    # Initial guess: persistent GARCH(1,1)-like
    x0 = zeros(n_params)
    x0[1] = var_ret * 0.05
    for i in 1:p; x0[1+i] = 0.05; end
    for j in 1:q; x0[1+p+j] = 0.90 / q; end

    lower = fill(1e-8, n_params)
    upper = vcat([var_ret * 10], fill(0.99, p + q))

    result = optimize(
        x -> garch_loglik(x, returns, p, q),
        lower, upper, x0,
        Fminbox(LBFGS()),
        Optim.Options(iterations=2000, g_tol=1e-8)
    )

    params = Optim.minimizer(result)
    return GARCH(params[1], params[2:1+p], params[2+p:end])
end

"""
    simulate(g::GARCH, T, n_paths; burn=500) → Matrix{Float64}

Simulate GARCH return paths. Returns n_paths × T matrix.
"""
function simulate(g::GARCH, T::Int, n_paths::Int;
                  burn::Int=500,
                  rng::AbstractRNG=Random.default_rng())::Matrix{Float64}
    p = length(g.alpha)
    q = length(g.beta)
    total = T + burn

    paths = Matrix{Float64}(undef, n_paths, T)

    for path in 1:n_paths
        rets = zeros(total)
        h    = zeros(total)
        var0 = g.omega / max(1 - sum(g.alpha) - sum(g.beta), 1e-8)
        h[1:max(p,q)] .= var0

        for t in (max(p,q)+1):total
            h[t] = g.omega
            for i in 1:p; h[t] += g.alpha[i] * rets[t-i]^2; end
            for j in 1:q; h[t] += g.beta[j] * h[t-j]; end
            h[t] = max(h[t], 1e-12)
            rets[t] = sqrt(h[t]) * randn(rng)
        end
        paths[path, :] = rets[burn+1:end]
    end
    return paths
end

"""
    forecast(g::GARCH, returns, h) → Vector{Float64}

h-step ahead conditional variance forecast.
"""
function forecast(g::GARCH, returns::Vector{Float64}, h::Int)::Vector{Float64}
    p = length(g.alpha)
    q = length(g.beta)
    variances = garch_variance_series(g, returns)

    var_uncond = g.omega / max(1 - sum(g.alpha) - sum(g.beta), 1e-8)
    pers = sum(g.alpha) + sum(g.beta)

    forecasts = zeros(h)
    last_h  = variances[end]
    last_e2 = returns[end]^2

    # 1-step: direct recursion
    forecasts[1] = g.omega + g.alpha[1] * last_e2 + g.beta[1] * last_h

    # Multi-step: E[ε²_{t+k}] = E[h_{t+k}] for k > 1
    for k in 2:h
        forecasts[k] = g.omega + pers * forecasts[k-1]
        # Long-run: var_uncond as k → ∞
    end

    # Converge toward unconditional variance
    for k in 1:h
        forecasts[k] = var_uncond + pers^k * (last_h - var_uncond)
    end

    return forecasts
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Heston Stochastic Volatility Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    Heston(S0, V0, kappa, theta, sigma, rho, r)

Heston (1993) model:
  dS = r S dt + √V S dW_S
  dV = κ(θ - V) dt + σ √V dW_V
  corr(dW_S, dW_V) = ρ
"""
struct Heston
    S0::Float64
    V0::Float64
    kappa::Float64
    theta::Float64
    sigma::Float64
    rho::Float64
    r::Float64

    function Heston(S0, V0, kappa, theta, sigma, rho, r)
        @assert S0 > 0; @assert V0 > 0; @assert kappa > 0
        @assert theta > 0; @assert sigma > 0
        @assert -1 < rho < 1
        # Feller condition: 2 κ θ > σ²
        if 2 * kappa * theta < sigma^2
            @warn "Feller condition violated: variance may hit zero"
        end
        new(S0, V0, kappa, theta, sigma, rho, r)
    end
end

"""
    simulate(h::Heston, T, n_steps, n_paths) → Tuple{Matrix, Matrix}

Euler-Maruyama discretisation with full truncation for V.
Returns (price_paths [n_paths × (n_steps+1)], vol_paths [n_paths × (n_steps+1)]).
"""
function simulate(hest::Heston, T::Float64, n_steps::Int, n_paths::Int;
                  rng::AbstractRNG=Random.default_rng())::Tuple{Matrix{Float64}, Matrix{Float64}}

    dt = T / n_steps
    sqdt = sqrt(dt)

    S_paths = Matrix{Float64}(undef, n_paths, n_steps + 1)
    V_paths = Matrix{Float64}(undef, n_paths, n_steps + 1)

    S_paths[:, 1] .= hest.S0
    V_paths[:, 1] .= hest.V0

    rho2 = sqrt(1 - hest.rho^2)

    for t in 2:(n_steps + 1)
        Z1 = randn(rng, n_paths)
        Z2 = hest.rho .* Z1 + rho2 .* randn(rng, n_paths)

        V_prev = max.(V_paths[:, t-1], 0.0)   # full truncation
        sqrtV  = sqrt.(V_prev)

        # Variance update (Milstein for variance process)
        dV = hest.kappa .* (hest.theta .- V_prev) .* dt .+
             hest.sigma .* sqrtV .* sqdt .* Z2 .+
             0.25 * hest.sigma^2 * dt * (Z2.^2 .- 1)
        V_paths[:, t] = max.(V_prev .+ dV, 0.0)

        # Price update
        S_prev = S_paths[:, t-1]
        log_ret = (hest.r - 0.5 .* V_prev) .* dt .+ sqrtV .* sqdt .* Z1
        S_paths[:, t] = S_prev .* exp.(log_ret)
    end

    return S_paths, V_paths
end

"""
    heston_char_fn(u, t, kappa, theta, sigma, rho, r, V0) → ComplexF64

Log-characteristic function of log-price under Heston (Lewis 2001 formulation).
"""
function heston_char_fn(u::ComplexF64, t::Float64,
                         kappa::Float64, theta::Float64, sigma::Float64,
                         rho::Float64, r::Float64, V0::Float64)::ComplexF64

    d = sqrt((rho * sigma * im * u - kappa)^2 - sigma^2 * (-im * u - u^2))
    g = (kappa - rho * sigma * im * u - d) /
        (kappa - rho * sigma * im * u + d)

    C = r * im * u * t +
        (kappa * theta / sigma^2) * (
            (kappa - rho * sigma * im * u - d) * t -
            2 * log((1 - g * exp(-d * t)) / (1 - g))
        )
    D = ((kappa - rho * sigma * im * u - d) / sigma^2) *
        ((1 - exp(-d * t)) / (1 - g * exp(-d * t)))

    return exp(C + D * V0)
end

"""
    price_option(h::Heston, K, T, call) → Float64

Price a European option using the Lewis (2001) formula with Gauss-Laguerre quadrature.
"""
function price_option(hest::Heston, K::Float64, T::Float64, call::Bool=true)::Float64
    F = hest.S0 * exp(hest.r * T)
    k = log(K / F)

    # Gauss-Laguerre nodes/weights (64-point)
    n_quad = 64
    nodes, weights = gauss_laguerre_64()

    integral = 0.0
    for j in 1:n_quad
        u = nodes[j] - 0.5im
        cf = heston_char_fn(u, T, hest.kappa, hest.theta,
                             hest.sigma, hest.rho, hest.r, hest.V0)
        integrand = real(exp(-im * u * k) * cf / (u^2 + 0.25))
        integral += weights[j] * integrand
    end

    call_price = hest.S0 - sqrt(K * F) * exp(-hest.r * T) * integral / π

    if call
        return max(call_price, 0.0)
    else
        # Put-call parity
        return call_price - hest.S0 + K * exp(-hest.r * T)
    end
end

"""
    gauss_laguerre_64() → Tuple{Vector, Vector}

Return 64-point Gauss-Laguerre nodes and weights for integration over [0, ∞).
Uses the recursion-based algorithm.
"""
function gauss_laguerre_64()
    n = 64
    # Build tridiagonal symmetric matrix (Golub-Welsch)
    a = [2k - 1 for k in 1:n]       # diagonal
    b = [sqrt(Float64(k)) for k in 1:(n-1)]   # off-diagonal

    T_mat = SymTridiagonal(Float64.(a), Float64.(b))
    vals, vecs = eigen(T_mat)

    # Nodes = eigenvalues (already positive for Laguerre)
    # Weights = (first component of eigenvector)^2 * factorial(0) = 1
    nodes   = vals
    weights = [vecs[1, j]^2 for j in 1:n]
    return nodes, weights
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Merton Jump Diffusion
# ─────────────────────────────────────────────────────────────────────────────

"""
    MertonJumpDiffusion(mu, sigma, lambda, mu_j, sigma_j)

Merton (1976) model:
  dS/S = (μ - λ k̄) dt + σ dW + (eʲ - 1) dN
where N is Poisson(λ), j ~ Normal(μⱼ, σⱼ²), k̄ = exp(μⱼ + ½σⱼ²) - 1.
"""
struct MertonJumpDiffusion
    mu::Float64
    sigma::Float64
    lambda::Float64   # jump arrival rate
    mu_j::Float64     # mean log-jump size
    sigma_j::Float64  # std log-jump size

    function MertonJumpDiffusion(mu, sigma, lambda, mu_j, sigma_j)
        @assert sigma > 0; @assert lambda >= 0; @assert sigma_j >= 0
        new(mu, sigma, lambda, mu_j, sigma_j)
    end
end

"""
    simulate(m::MertonJumpDiffusion, T, n_steps, n_paths) → Matrix{Float64}
"""
function simulate(m::MertonJumpDiffusion, T::Float64, n_steps::Int, n_paths::Int;
                  rng::AbstractRNG=Random.default_rng())::Matrix{Float64}
    dt     = T / n_steps
    sqdt   = sqrt(dt)
    k_bar  = exp(m.mu_j + 0.5 * m.sigma_j^2) - 1   # E[eʲ - 1]
    drift  = (m.mu - m.lambda * k_bar - 0.5 * m.sigma^2) * dt

    paths = Matrix{Float64}(undef, n_paths, n_steps + 1)
    paths[:, 1] .= 1.0   # normalised to 1

    pois_dist = Poisson(m.lambda * dt)

    for t in 2:(n_steps + 1)
        Z  = randn(rng, n_paths)
        Nj = rand(rng, pois_dist, n_paths)

        # Jump component: sum of Nj log-normal jumps
        jump = zeros(Float64, n_paths)
        for i in 1:n_paths
            if Nj[i] > 0
                for _ in 1:Nj[i]
                    jump[i] += m.mu_j + m.sigma_j * randn(rng)
                end
            end
        end

        log_ret = drift .+ m.sigma * sqdt .* Z .+ jump
        paths[:, t] = paths[:, t-1] .* exp.(log_ret)
    end
    return paths
end

"""
    fit(::Type{MertonJumpDiffusion}, returns, dt) → MertonJumpDiffusion

Estimate MJD parameters from return series using method of moments.
"""
function fit(::Type{MertonJumpDiffusion}, returns::Vector{Float64},
             dt::Float64=1.0/252)::MertonJumpDiffusion

    # Method of moments: match first 4 cumulants
    m1 = mean(returns) / dt
    m2 = var(returns) / dt
    m3 = cumulant(returns, 3) / dt
    m4 = cumulant(returns, 4) / dt

    # λ σⱼ⁴ ≈ excess kurtosis * σ_total² / 3  (approximate)
    sigma_j2 = max(sqrt(abs(m4) / max(abs(m3) + 1e-10, 1e-10)), 1e-4)
    mu_j     = m3 / max(abs(m2) * 3, 1e-10)
    lambda   = max(m4 / max(3 * sigma_j2^2 * dt, 1e-10), 0.1)
    sigma    = sqrt(max(m2 - lambda * (mu_j^2 + sigma_j2), 1e-8))
    k_bar    = exp(mu_j + 0.5 * sigma_j2) - 1
    mu       = m1 + lambda * k_bar + 0.5 * sigma^2

    return MertonJumpDiffusion(mu, sigma, lambda, mu_j, sqrt(sigma_j2))
end

function cumulant(x::Vector{Float64}, k::Int)::Float64
    n = length(x)
    m = mean(x)
    if k == 3
        return mean((x .- m).^3)
    elseif k == 4
        s2 = var(x)
        return mean((x .- m).^4) - 3 * s2^2
    end
    return 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ornstein-Uhlenbeck (Mean-Reverting)
# ─────────────────────────────────────────────────────────────────────────────

"""
    OrnsteinUhlenbeck(mu, kappa, sigma)

OU process: dX = κ(μ - X) dt + σ dW.
"""
struct OrnsteinUhlenbeck
    mu::Float64
    kappa::Float64
    sigma::Float64

    function OrnsteinUhlenbeck(mu, kappa, sigma)
        @assert kappa > 0 "kappa (mean-reversion speed) must be positive"
        @assert sigma > 0
        new(mu, kappa, sigma)
    end
end

"""
    simulate(ou::OrnsteinUhlenbeck, T, n_steps, n_paths; X0=mu) → Matrix{Float64}

Exact simulation using the conditional Gaussian distribution.
"""
function simulate(ou::OrnsteinUhlenbeck, T::Float64, n_steps::Int, n_paths::Int;
                  X0::Float64=ou.mu,
                  rng::AbstractRNG=Random.default_rng())::Matrix{Float64}
    dt   = T / n_steps
    e_κ  = exp(-ou.kappa * dt)
    mean_incr = ou.mu * (1 - e_κ)
    std_incr  = ou.sigma * sqrt((1 - e_κ^2) / (2 * ou.kappa))

    paths = Matrix{Float64}(undef, n_paths, n_steps + 1)
    paths[:, 1] .= X0

    for t in 2:(n_steps + 1)
        Z = randn(rng, n_paths)
        @. paths[:, t] = paths[:, t-1] * e_κ + mean_incr + std_incr * Z
    end
    return paths
end

"""
    fit(::Type{OrnsteinUhlenbeck}, series, dt) → OrnsteinUhlenbeck

MLE estimation from a discretely observed OU path via OLS regression.

For small dt:  X_{t+Δt} ≈ α + β X_t + ε
Then: κ = -log(β)/Δt, μ = α/(1-β), σ from residual variance.
"""
function fit(::Type{OrnsteinUhlenbeck}, series::Vector{Float64},
             dt::Float64=1.0/252)::OrnsteinUhlenbeck
    n = length(series)
    @assert n >= 10

    X  = series[1:n-1]
    Y  = series[2:n]

    # OLS: Y = α + β X
    mx  = mean(X); my = mean(Y)
    ss_xx = sum((x - mx)^2 for x in X)
    ss_xy = sum((X[i] - mx) * (Y[i] - my) for i in 1:n-1)

    beta  = ss_xy / ss_xx
    alpha = my - beta * mx

    beta  = clamp(beta, 1e-6, 1 - 1e-6)
    kappa = -log(beta) / dt
    mu    = alpha / (1 - beta)

    # Residual variance → σ
    resid    = Y .- (alpha .+ beta .* X)
    var_resid = var(resid)
    sigma    = sqrt(max(var_resid * 2 * kappa / (1 - beta^2), 1e-8))

    return OrnsteinUhlenbeck(mu, kappa, sigma)
end

"""
    half_life(ou::OrnsteinUhlenbeck) → Float64

Time (in time units) for half the deviation to revert: log(2) / κ.
"""
function half_life(ou::OrnsteinUhlenbeck)::Float64
    return log(2) / ou.kappa
end

"""
    stationary_std(ou::OrnsteinUhlenbeck) → Float64

Long-run (stationary) standard deviation: σ / √(2κ).
"""
function stationary_std(ou::OrnsteinUhlenbeck)::Float64
    return ou.sigma / sqrt(2 * ou.kappa)
end

"""
    ou_entry_exit_bands(ou, n_sigma) → Tuple{Float64, Float64, Float64, Float64}

Return (long_entry, long_exit, short_entry, short_exit) z-score levels.
"""
function ou_entry_exit_bands(ou::OrnsteinUhlenbeck, n_sigma::Float64=1.5)
    s = stationary_std(ou)
    long_entry  = ou.mu - n_sigma * s
    long_exit   = ou.mu
    short_entry = ou.mu + n_sigma * s
    short_exit  = ou.mu
    return (long_entry, long_exit, short_entry, short_exit)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Hawkes Process (Self-Exciting)
# ─────────────────────────────────────────────────────────────────────────────

"""
    Hawkes(mu, alpha, beta)

Univariate Hawkes process with exponential kernel:
  λ(t) = μ + α Σ_{tᵢ < t} exp(-β(t - tᵢ))

Stationarity requires: α < β.
"""
struct Hawkes
    mu::Float64
    alpha::Float64
    beta::Float64

    function Hawkes(mu, alpha, beta)
        @assert mu > 0
        @assert alpha >= 0
        @assert beta > 0
        if alpha >= beta
            @warn "Hawkes process not stationary: alpha >= beta"
        end
        new(mu, alpha, beta)
    end
end

"""
    simulate(h::Hawkes, T) → Vector{Float64}

Ogata's modified thinning algorithm. Returns event times in [0, T].
"""
function simulate(h::Hawkes, T::Float64;
                  rng::AbstractRNG=Random.default_rng())::Vector{Float64}
    events = Float64[]
    t = 0.0

    # Intensity at start = mu
    lambda_bar = h.mu + 5 * h.alpha   # upper bound (liberal)

    while t < T
        # Time to next candidate event
        u1 = rand(rng)
        dt = -log(u1) / lambda_bar
        t_cand = t + dt

        if t_cand > T
            break
        end

        # Compute intensity at t_cand
        excitation = sum(h.alpha * exp(-h.beta * (t_cand - ti)) for ti in events;
                         init=0.0)
        lambda_cand = h.mu + excitation

        # Acceptance step
        u2 = rand(rng)
        if u2 <= lambda_cand / lambda_bar
            push!(events, t_cand)
            t = t_cand
            lambda_bar = lambda_cand + h.alpha   # update upper bound
        else
            t = t_cand
        end
    end
    return events
end

"""
    hawkes_intensity(h, events, t) → Float64

Compute intensity λ(t) given past events.
"""
function hawkes_intensity(h::Hawkes, events::Vector{Float64}, t::Float64)::Float64
    excit = sum(exp(-h.beta * (t - ti)) for ti in events if ti < t; init=0.0)
    return h.mu + h.alpha * excit
end

"""
    hawkes_loglik(params, events, T) → Float64

Log-likelihood of a Hawkes process given observed events in [0, T].
"""
function hawkes_loglik(params::Vector{Float64}, events::Vector{Float64},
                        T::Float64)::Float64
    mu, alpha, beta = params
    if mu <= 0 || alpha < 0 || beta <= 0 || alpha >= beta
        return -Inf
    end

    n = length(events)
    if n == 0
        return -mu * T
    end

    # Recursive calculation of A_i = Σ_{j<i} exp(-β(tᵢ - tⱼ))
    A = zeros(n)
    for i in 2:n
        A[i] = exp(-beta * (events[i] - events[i-1])) * (1 + A[i-1])
    end

    # Log-likelihood
    ll = -mu * T

    # Sum of log intensities
    for i in 1:n
        lambda_i = mu + alpha * A[i]
        ll += log(max(lambda_i, 1e-15))
    end

    # Compensator integral
    ll -= alpha / beta * sum(1 - exp(-beta * (T - ti)) for ti in events)

    return ll
end

"""
    fit(::Type{Hawkes}, event_times, T) → Hawkes

MLE estimation via Optim.jl.
"""
function fit(::Type{Hawkes}, event_times::Vector{Float64}, T::Float64)::Hawkes
    n_events = length(event_times)

    # Initial guess: baseline rate, modest excitation
    mu0    = n_events / T * 0.5
    alpha0 = 0.3
    beta0  = 1.0

    x0     = [mu0, alpha0, beta0]
    lower  = [1e-8, 0.0, 1e-8]
    upper  = [Inf,  0.999, 100.0]

    result = optimize(
        x -> -hawkes_loglik(x, event_times, T),
        lower, upper, x0,
        Fminbox(LBFGS()),
        Optim.Options(iterations=1000, g_tol=1e-7)
    )

    p = Optim.minimizer(result)
    return Hawkes(p[1], p[2], p[3])
end

"""
    branching_ratio(h::Hawkes) → Float64

n = α/β: expected number of offspring per event. n < 1 for stationarity.
"""
function branching_ratio(h::Hawkes)::Float64
    return h.alpha / h.beta
end

"""
    mean_intensity(h::Hawkes) → Float64

E[λ] = μ / (1 - α/β) for stationary process.
"""
function mean_intensity(h::Hawkes)::Float64
    br = branching_ratio(h)
    if br >= 1
        return Inf
    end
    return h.mu / (1 - br)
end

# ─────────────────────────────────────────────────────────────────────────────
# Utility: Monte Carlo path statistics
# ─────────────────────────────────────────────────────────────────────────────

"""
    path_quantiles(paths, probs) → Matrix{Float64}

Compute row-wise quantiles of paths.  Returns length(probs) × n_steps matrix.
"""
function path_quantiles(paths::Matrix{Float64},
                         probs::Vector{Float64}=[0.05, 0.25, 0.5, 0.75, 0.95])::Matrix{Float64}
    n_paths, n_steps = size(paths)
    result = Matrix{Float64}(undef, length(probs), n_steps)
    for t in 1:n_steps
        col = sort(paths[:, t])
        for (k, p) in enumerate(probs)
            idx = clamp(round(Int, p * n_paths), 1, n_paths)
            result[k, t] = col[idx]
        end
    end
    return result
end

"""
    terminal_distribution(paths) → NamedTuple

Return summary statistics of terminal values: mean, std, skew, kurt,
VaR_5, ES_5.
"""
function terminal_distribution(paths::Matrix{Float64})::NamedTuple
    terminal = paths[:, end]
    n = length(terminal)
    m = mean(terminal); s = std(terminal)
    sorted = sort(terminal)
    var5_idx = max(1, round(Int, 0.05 * n))
    var5 = sorted[var5_idx]
    es5  = mean(sorted[1:var5_idx])
    sk   = mean(((terminal .- m) ./ s).^3)
    ku   = mean(((terminal .- m) ./ s).^4) - 3

    return (mean=m, std=s, skew=sk, excess_kurtosis=ku,
            var_5pct=var5, es_5pct=es5,
            p5=sorted[max(1,round(Int,0.05*n))],
            p25=sorted[max(1,round(Int,0.25*n))],
            p50=sorted[max(1,round(Int,0.50*n))],
            p75=sorted[max(1,round(Int,0.75*n))],
            p95=sorted[min(n,round(Int,0.95*n))])
end

end # module Stochastic
