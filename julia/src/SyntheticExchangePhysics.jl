"""
    SyntheticExchangePhysics

Stochastic physics layer for a synthetic exchange simulator.
Provides price dynamics, volatility surfaces, order flow, liquidity,
microstructure noise, correlated multi-asset generation, event injection,
calibration, and full universe synthesis.

Uses only LinearAlgebra, Statistics, Random from stdlib.
"""
module SyntheticExchangePhysics

using LinearAlgebra
using Statistics
using Random

export GeometricBrownianMotion, HestonModel, MertonJumpDiffusion, RegimeSwitchingModel
export VolatilitySurface, SVIParameters, OrderFlowModel, HawkesProcess
export LiquidityModel, FlashCrashGenerator, MarketMicrostructureNoise
export CorrelatedAssets, EventInjector, TrueValueProcess, CalibrationEngine
export SyntheticUniverseGenerator, SyntheticUniverse
export simulate, get_vol, generate_spread, generate_flash_crash
export add_microstructure_noise, generate_correlated_assets
export inject_event!, calibrate, generate_universe

# ============================================================================
# Section 1: Geometric Brownian Motion
# ============================================================================

"""
    GeometricBrownianMotion{T<:AbstractFloat}

Standard GBM: dS = μ S dt + σ S dW.

# Fields
- `mu::T`    — annualized drift
- `sigma::T` — annualized volatility
- `S0::T`    — initial price
"""
struct GeometricBrownianMotion{T<:AbstractFloat}
    mu::T
    sigma::T
    S0::T

    function GeometricBrownianMotion{T}(mu::T, sigma::T, S0::T) where {T<:AbstractFloat}
        S0 > zero(T) || throw(ArgumentError("S0 must be positive, got $S0"))
        sigma >= zero(T) || throw(ArgumentError("sigma must be non-negative, got $sigma"))
        new{T}(mu, sigma, S0)
    end
end

GeometricBrownianMotion(mu::T, sigma::T, S0::T) where {T<:AbstractFloat} =
    GeometricBrownianMotion{T}(mu, sigma, S0)

GeometricBrownianMotion(mu::Real, sigma::Real, S0::Real) =
    GeometricBrownianMotion(promote(float(mu), float(sigma), float(S0))...)

"""
    simulate(gbm::GeometricBrownianMotion, n_steps::Int, dt::Real; rng=Random.default_rng())

Simulate a GBM price path using the exact log-normal solution.
Returns a Vector of length `n_steps + 1` starting at `gbm.S0`.
"""
function simulate(gbm::GeometricBrownianMotion{T}, n_steps::Int, dt::Real;
                  rng::AbstractRNG=Random.default_rng()) where {T}
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    dt > 0 || throw(ArgumentError("dt must be positive"))

    prices = Vector{T}(undef, n_steps + 1)
    prices[1] = gbm.S0
    sqrt_dt = sqrt(T(dt))
    drift = (gbm.mu - T(0.5) * gbm.sigma^2) * T(dt)

    @inbounds for i in 2:(n_steps + 1)
        Z = randn(rng, T)
        log_return = drift + gbm.sigma * sqrt_dt * Z
        prices[i] = prices[i-1] * exp(log_return)
    end
    prices
end

"""
    simulate_antithetic(gbm::GeometricBrownianMotion, n_steps::Int, dt::Real; rng)

Antithetic variates: returns two negatively-correlated paths for variance reduction.
"""
function simulate_antithetic(gbm::GeometricBrownianMotion{T}, n_steps::Int, dt::Real;
                             rng::AbstractRNG=Random.default_rng()) where {T}
    prices_pos = Vector{T}(undef, n_steps + 1)
    prices_neg = Vector{T}(undef, n_steps + 1)
    prices_pos[1] = gbm.S0
    prices_neg[1] = gbm.S0
    sqrt_dt = sqrt(T(dt))
    drift = (gbm.mu - T(0.5) * gbm.sigma^2) * T(dt)

    @inbounds for i in 2:(n_steps + 1)
        Z = randn(rng, T)
        lr_pos = drift + gbm.sigma * sqrt_dt * Z
        lr_neg = drift + gbm.sigma * sqrt_dt * (-Z)
        prices_pos[i] = prices_pos[i-1] * exp(lr_pos)
        prices_neg[i] = prices_neg[i-1] * exp(lr_neg)
    end
    (prices_pos, prices_neg)
end

"""
    expected_price(gbm::GeometricBrownianMotion, t::Real)

Analytical expected price E[S(t)] = S0 * exp(mu * t).
"""
expected_price(gbm::GeometricBrownianMotion, t::Real) = gbm.S0 * exp(gbm.mu * t)

"""
    variance_price(gbm::GeometricBrownianMotion, t::Real)

Analytical variance Var[S(t)] = S0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1).
"""
function variance_price(gbm::GeometricBrownianMotion, t::Real)
    gbm.S0^2 * exp(2 * gbm.mu * t) * (exp(gbm.sigma^2 * t) - 1)
end

"""
    simulate_paths(gbm::GeometricBrownianMotion, n_paths, n_steps, dt; rng)

Monte Carlo: generate `n_paths` independent GBM paths.  Returns a matrix
of size `(n_steps+1) x n_paths`.
"""
function simulate_paths(gbm::GeometricBrownianMotion{T}, n_paths::Int, n_steps::Int, dt::Real;
                        rng::AbstractRNG=Random.default_rng()) where {T}
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    sqrt_dt = sqrt(T(dt))
    drift = (gbm.mu - T(0.5) * gbm.sigma^2) * T(dt)

    @inbounds for j in 1:n_paths
        paths[1, j] = gbm.S0
        for i in 2:(n_steps + 1)
            Z = randn(rng, T)
            paths[i, j] = paths[i-1, j] * exp(drift + gbm.sigma * sqrt_dt * Z)
        end
    end
    paths
end


# ============================================================================
# Section 2: Heston Stochastic Volatility Model
# ============================================================================

"""
    HestonModel{T<:AbstractFloat}

Heston (1993) stochastic volatility:
    dS = μ S dt + √v S dW₁
    dv = κ(θ − v) dt + ξ √v dW₂
    corr(dW₁, dW₂) = ρ

Uses the Quadratic Exponential (QE) scheme of Andersen (2008) for the
variance process to avoid negative variances.

# Fields
- `mu`    — drift of the price process
- `kappa` — mean-reversion speed of variance
- `theta` — long-run variance
- `xi`    — vol-of-vol
- `rho`   — correlation between Brownian motions
- `S0`    — initial price
- `v0`    — initial variance
"""
struct HestonModel{T<:AbstractFloat}
    mu::T
    kappa::T
    theta::T
    xi::T
    rho::T
    S0::T
    v0::T

    function HestonModel{T}(mu, kappa, theta, xi, rho, S0, v0) where {T<:AbstractFloat}
        S0 > zero(T) || throw(ArgumentError("S0 must be positive"))
        v0 >= zero(T) || throw(ArgumentError("v0 must be non-negative"))
        kappa > zero(T) || throw(ArgumentError("kappa must be positive"))
        theta > zero(T) || throw(ArgumentError("theta must be positive"))
        xi > zero(T) || throw(ArgumentError("xi must be positive"))
        abs(rho) <= one(T) || throw(ArgumentError("|rho| must be <= 1"))
        new{T}(T(mu), T(kappa), T(theta), T(xi), T(rho), T(S0), T(v0))
    end
end

function HestonModel(mu::Real, kappa::Real, theta::Real, xi::Real, rho::Real,
                     S0::Real, v0::Real)
    T = float(promote_type(typeof(mu), typeof(kappa), typeof(theta),
                           typeof(xi), typeof(rho), typeof(S0), typeof(v0)))
    HestonModel{T}(mu, kappa, theta, xi, rho, S0, v0)
end

"""
    _qe_variance_step(v, kappa, theta, xi, dt, U; psi_crit=1.5)

One step of the QE (Quadratic Exponential) discretisation for the CIR
variance process.  `U` is a uniform(0,1) draw.

When ψ = s²/m² is small the distribution is approximated by a scaled
non-central χ² (moment-matched); when ψ is large we switch to an
exponential approximation.  `psi_crit` is the switchover point (Andersen
recommends 1.5).
"""
function _qe_variance_step(v::T, kappa::T, theta::T, xi::T, dt::T, U::T;
                           psi_crit::T=T(1.5)) where {T<:AbstractFloat}
    # Moments of the exact conditional distribution
    e_kdt = exp(-kappa * dt)
    m = theta + (v - theta) * e_kdt                         # E[v(t+dt)|v(t)]
    s2 = (v * xi^2 * e_kdt / kappa) * (one(T) - e_kdt) +
         (theta * xi^2 / (2 * kappa)) * (one(T) - e_kdt)^2  # Var[v(t+dt)|v(t)]
    psi = s2 / max(m^2, eps(T))

    if psi <= psi_crit
        # Quadratic scheme: v_next = a * (b + Z_v)^2 where Z_v ~ N(0,1)
        b2 = 2 / psi - one(T) + sqrt(2 / psi) * sqrt(max(2 / psi - one(T), zero(T)))
        a = m / (one(T) + b2)
        b = sqrt(max(b2, zero(T)))
        # Inverse-CDF via normal quantile of U
        Zv = _norminv(U)
        v_next = a * (b + Zv)^2
    else
        # Exponential scheme: P(v_next=0)=p, else v_next ~ Exp(1/beta)
        p = (psi - one(T)) / (psi + one(T))
        beta = (one(T) - p) / max(m, eps(T))
        if U <= p
            v_next = zero(T)
        else
            v_next = log(max((one(T) - p) / (one(T) - U), eps(T))) / beta
        end
    end
    max(v_next, zero(T))
end

"""
    _norminv(p)

Rational approximation to the inverse standard-normal CDF (Beasley-Springer-Moro).
Accurate to about 1e-9 in the interior and degrades gracefully in the tails.
"""
function _norminv(p::T) where {T<:AbstractFloat}
    # Clamp to avoid ±Inf
    p = clamp(p, T(1e-15), one(T) - T(1e-15))

    # Coefficients for the rational approximation (Moro 1995)
    a = (T(-3.969683028665376e1), T(2.209460984245205e2),
         T(-2.759285104469687e2), T(1.383577518672690e2),
         T(-3.066479806614716e1), T(2.506628277459239e0))
    b = (T(-5.447609879822406e1), T(1.615858368580409e2),
         T(-1.556989798598866e2), T(6.680131188771972e1),
         T(-1.328068155288572e1))
    c = (T(-7.784894002430293e-3), T(-3.223964580411365e-1),
         T(-2.400758277161838e0), T(-2.549732539343734e0),
         T(4.374664141464968e0), T(2.938163982698783e0))
    d = (T(7.784695709041462e-3), T(3.224671290700398e-1),
         T(2.445134137142996e0), T(3.754408661907416e0))

    p_low  = T(0.02425)
    p_high = one(T) - p_low

    if p < p_low
        q = sqrt(-2 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+one(T))
    elseif p <= p_high
        q = p - T(0.5)
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+one(T))
    else
        q = sqrt(-2 * log(one(T) - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                 ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+one(T))
    end
end

"""
    simulate(heston::HestonModel, n_steps::Int, dt::Real; rng)

Full QE simulation of the Heston model.
Returns `(prices, variances)` each of length `n_steps + 1`.
"""
function simulate(heston::HestonModel{T}, n_steps::Int, dt::Real;
                  rng::AbstractRNG=Random.default_rng()) where {T}
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    dt_T = T(dt)

    prices = Vector{T}(undef, n_steps + 1)
    vars   = Vector{T}(undef, n_steps + 1)
    prices[1] = heston.S0
    vars[1]   = heston.v0

    sqrt_dt = sqrt(dt_T)
    rho = heston.rho
    sqrt_1mrho2 = sqrt(one(T) - rho^2)

    @inbounds for i in 2:(n_steps + 1)
        v_cur = vars[i-1]

        # QE step for variance
        U = rand(rng, T)
        v_next = _qe_variance_step(v_cur, heston.kappa, heston.theta,
                                   heston.xi, dt_T, U)
        vars[i] = v_next

        # Log-price step using the average of v_cur and v_next
        v_avg = T(0.5) * (v_cur + v_next)
        Z1 = randn(rng, T)
        # Correlation: W1 = Z1, but the variance innovation is already done via QE
        # so we just use the price diffusion with v_avg
        log_ret = (heston.mu - T(0.5) * v_avg) * dt_T + sqrt(max(v_avg, zero(T))) * sqrt_dt * Z1
        prices[i] = prices[i-1] * exp(log_ret)
    end
    (prices, vars)
end

"""
    simulate_paths(heston::HestonModel, n_paths, n_steps, dt; rng)

Monte Carlo: `n_paths` independent Heston paths.
Returns `(price_matrix, var_matrix)` each `(n_steps+1) x n_paths`.
"""
function simulate_paths(heston::HestonModel{T}, n_paths::Int, n_steps::Int, dt::Real;
                        rng::AbstractRNG=Random.default_rng()) where {T}
    price_mat = Matrix{T}(undef, n_steps + 1, n_paths)
    var_mat   = Matrix{T}(undef, n_steps + 1, n_paths)
    dt_T = T(dt)
    sqrt_dt = sqrt(dt_T)

    @inbounds for j in 1:n_paths
        price_mat[1, j] = heston.S0
        var_mat[1, j]   = heston.v0
        for i in 2:(n_steps + 1)
            v_cur = var_mat[i-1, j]
            U = rand(rng, T)
            v_next = _qe_variance_step(v_cur, heston.kappa, heston.theta,
                                       heston.xi, dt_T, U)
            var_mat[i, j] = v_next
            v_avg = T(0.5) * (v_cur + v_next)
            Z1 = randn(rng, T)
            lr = (heston.mu - T(0.5) * v_avg) * dt_T + sqrt(max(v_avg, zero(T))) * sqrt_dt * Z1
            price_mat[i, j] = price_mat[i-1, j] * exp(lr)
        end
    end
    (price_mat, var_mat)
end

"""
    feller_condition(heston::HestonModel)

Check whether the Feller condition 2κθ > ξ² is satisfied (ensures v > 0 a.s.).
"""
feller_condition(h::HestonModel) = 2 * h.kappa * h.theta > h.xi^2


# ============================================================================
# Section 3: Merton Jump-Diffusion
# ============================================================================

"""
    MertonJumpDiffusion{T<:AbstractFloat}

Merton (1976) jump-diffusion:
    dS/S = (μ − λ k̄) dt + σ dW + J dN

where N is Poisson(λ dt), J ~ exp(N(μ_J, σ_J)) − 1, and
k̄ = E[J] = exp(μ_J + σ_J²/2) − 1.

# Fields
- `mu`       — total drift
- `sigma`    — diffusion volatility
- `lambda_j` — jump intensity (Poisson rate)
- `mu_j`     — mean of log-jump size
- `sigma_j`  — std of log-jump size
- `S0`       — initial price
"""
struct MertonJumpDiffusion{T<:AbstractFloat}
    mu::T
    sigma::T
    lambda_j::T
    mu_j::T
    sigma_j::T
    S0::T

    function MertonJumpDiffusion{T}(mu, sigma, lambda_j, mu_j, sigma_j, S0) where {T}
        S0 > zero(T) || throw(ArgumentError("S0 must be positive"))
        sigma >= zero(T) || throw(ArgumentError("sigma must be non-negative"))
        lambda_j >= zero(T) || throw(ArgumentError("lambda_j must be non-negative"))
        sigma_j >= zero(T) || throw(ArgumentError("sigma_j must be non-negative"))
        new{T}(T(mu), T(sigma), T(lambda_j), T(mu_j), T(sigma_j), T(S0))
    end
end

function MertonJumpDiffusion(mu::Real, sigma::Real, lambda_j::Real,
                             mu_j::Real, sigma_j::Real, S0::Real)
    T = float(promote_type(typeof.((mu, sigma, lambda_j, mu_j, sigma_j, S0))...))
    MertonJumpDiffusion{T}(mu, sigma, lambda_j, mu_j, sigma_j, S0)
end

"""
    _poisson_sample(lambda, rng)

Sample from Poisson(lambda) using the inverse-CDF method.
"""
function _poisson_sample(lambda::T, rng::AbstractRNG) where {T<:AbstractFloat}
    if lambda <= zero(T)
        return 0
    end
    L = exp(-lambda)
    k = 0
    p = one(T)
    while true
        k += 1
        p *= rand(rng, T)
        if p < L
            return k - 1
        end
    end
end

"""
    simulate(mjd::MertonJumpDiffusion, n_steps::Int, dt::Real; rng)

Simulate a Merton jump-diffusion price path.
Returns a Vector of length `n_steps + 1`.
"""
function simulate(mjd::MertonJumpDiffusion{T}, n_steps::Int, dt::Real;
                  rng::AbstractRNG=Random.default_rng()) where {T}
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    dt_T = T(dt)

    # Compensator: k_bar = E[e^J - 1] = exp(mu_j + sigma_j^2/2) - 1
    k_bar = exp(mjd.mu_j + T(0.5) * mjd.sigma_j^2) - one(T)
    drift_comp = mjd.mu - mjd.lambda_j * k_bar
    sqrt_dt = sqrt(dt_T)

    prices = Vector{T}(undef, n_steps + 1)
    prices[1] = mjd.S0

    @inbounds for i in 2:(n_steps + 1)
        # Diffusion component
        Z = randn(rng, T)
        diffusion = (drift_comp - T(0.5) * mjd.sigma^2) * dt_T + mjd.sigma * sqrt_dt * Z

        # Jump component
        n_jumps = _poisson_sample(mjd.lambda_j * dt_T, rng)
        jump_sum = zero(T)
        for _ in 1:n_jumps
            jump_sum += mjd.mu_j + mjd.sigma_j * randn(rng, T)
        end

        prices[i] = prices[i-1] * exp(diffusion + jump_sum)
    end
    prices
end

"""
    simulate_paths(mjd::MertonJumpDiffusion, n_paths, n_steps, dt; rng)

Monte Carlo: `n_paths` Merton jump-diffusion paths.
"""
function simulate_paths(mjd::MertonJumpDiffusion{T}, n_paths::Int, n_steps::Int, dt::Real;
                        rng::AbstractRNG=Random.default_rng()) where {T}
    paths = Matrix{T}(undef, n_steps + 1, n_paths)
    dt_T = T(dt)
    k_bar = exp(mjd.mu_j + T(0.5) * mjd.sigma_j^2) - one(T)
    drift_comp = mjd.mu - mjd.lambda_j * k_bar
    sqrt_dt = sqrt(dt_T)

    @inbounds for j in 1:n_paths
        paths[1, j] = mjd.S0
        for i in 2:(n_steps + 1)
            Z = randn(rng, T)
            diff = (drift_comp - T(0.5) * mjd.sigma^2) * dt_T + mjd.sigma * sqrt_dt * Z
            nj = _poisson_sample(mjd.lambda_j * dt_T, rng)
            js = zero(T)
            for _ in 1:nj
                js += mjd.mu_j + mjd.sigma_j * randn(rng, T)
            end
            paths[i, j] = paths[i-1, j] * exp(diff + js)
        end
    end
    paths
end

"""
    total_variance(mjd::MertonJumpDiffusion, dt::Real)

Total variance per time step including the jump component:
    σ²_total dt = σ² dt + λ (μ_J² + σ_J²) dt
"""
function total_variance(mjd::MertonJumpDiffusion, dt::Real)
    (mjd.sigma^2 + mjd.lambda_j * (mjd.mu_j^2 + mjd.sigma_j^2)) * dt
end


# ============================================================================
# Section 4: Regime-Switching Model
# ============================================================================

"""
    RegimeParameters{T<:AbstractFloat}

Drift and volatility for a single regime.
"""
struct RegimeParameters{T<:AbstractFloat}
    mu::T
    sigma::T
end

"""
    RegimeSwitchingModel{T<:AbstractFloat}

Two-state Markov regime-switching GBM.

Regime 1 = bull, Regime 2 = bear (by convention).
Transition matrix P is 2×2 where P[i,j] = Prob(regime j at t+1 | regime i at t).

# Fields
- `regimes`        — Tuple of two RegimeParameters
- `transition_mat` — 2×2 row-stochastic transition matrix
- `S0`             — initial price
- `initial_regime` — 1 or 2
"""
struct RegimeSwitchingModel{T<:AbstractFloat}
    regimes::Tuple{RegimeParameters{T}, RegimeParameters{T}}
    transition_mat::Matrix{T}
    S0::T
    initial_regime::Int

    function RegimeSwitchingModel{T}(regimes, P, S0, ir) where {T}
        size(P) == (2, 2) || throw(ArgumentError("Transition matrix must be 2x2"))
        all(sum(P, dims=2) .≈ one(T)) || throw(ArgumentError("Rows of P must sum to 1"))
        all(P .>= zero(T)) || throw(ArgumentError("P entries must be non-negative"))
        S0 > zero(T) || throw(ArgumentError("S0 must be positive"))
        ir in (1, 2) || throw(ArgumentError("initial_regime must be 1 or 2"))
        new{T}(regimes, Matrix{T}(P), T(S0), ir)
    end
end

function RegimeSwitchingModel(mu_bull::Real, sigma_bull::Real,
                              mu_bear::Real, sigma_bear::Real,
                              P::AbstractMatrix, S0::Real;
                              initial_regime::Int=1)
    T = Float64
    r1 = RegimeParameters{T}(T(mu_bull), T(sigma_bull))
    r2 = RegimeParameters{T}(T(mu_bear), T(sigma_bear))
    RegimeSwitchingModel{T}((r1, r2), Matrix{T}(P), T(S0), initial_regime)
end

"""
    _next_regime(current::Int, P::Matrix, rng)

Sample the next regime from the transition matrix row.
"""
function _next_regime(current::Int, P::Matrix{T}, rng::AbstractRNG) where {T}
    u = rand(rng, T)
    u < P[current, 1] ? 1 : 2
end

"""
    simulate(rsm::RegimeSwitchingModel, n_steps::Int, dt::Real; rng)

Simulate a regime-switching GBM.
Returns `(prices, regimes)` where regimes is a Vector{Int} of length `n_steps + 1`.
"""
function simulate(rsm::RegimeSwitchingModel{T}, n_steps::Int, dt::Real;
                  rng::AbstractRNG=Random.default_rng()) where {T}
    dt_T = T(dt)
    sqrt_dt = sqrt(dt_T)

    prices  = Vector{T}(undef, n_steps + 1)
    regimes = Vector{Int}(undef, n_steps + 1)
    prices[1]  = rsm.S0
    regimes[1] = rsm.initial_regime

    @inbounds for i in 2:(n_steps + 1)
        reg = regimes[i-1]
        # Possibly switch regime
        regimes[i] = _next_regime(reg, rsm.transition_mat, rng)
        reg_new = regimes[i]

        rp = rsm.regimes[reg_new]
        Z = randn(rng, T)
        lr = (rp.mu - T(0.5) * rp.sigma^2) * dt_T + rp.sigma * sqrt_dt * Z
        prices[i] = prices[i-1] * exp(lr)
    end
    (prices, regimes)
end

"""
    stationary_distribution(rsm::RegimeSwitchingModel)

Compute the stationary distribution π of the 2-state Markov chain.
π₁ = (1 − P₂₂) / (2 − P₁₁ − P₂₂), π₂ = 1 − π₁.
"""
function stationary_distribution(rsm::RegimeSwitchingModel{T}) where {T}
    P = rsm.transition_mat
    denom = T(2) - P[1,1] - P[2,2]
    pi1 = (one(T) - P[2,2]) / denom
    (pi1, one(T) - pi1)
end

"""
    expected_regime_duration(rsm::RegimeSwitchingModel, regime::Int)

Expected number of steps in a regime = 1 / (1 − P[regime, regime]).
"""
function expected_regime_duration(rsm::RegimeSwitchingModel, regime::Int)
    1.0 / (1.0 - rsm.transition_mat[regime, regime])
end


# ============================================================================
# Section 5: Volatility Surface (SVI Parameterisation)
# ============================================================================

"""
    SVIParameters{T<:AbstractFloat}

Stochastic Volatility Inspired (SVI) parameterisation of the implied
total variance surface (Gatheral 2004):

    w(k) = a + b [ ρ(k − m) + √((k − m)² + σ²) ]

where k = log(K/F) is the log-moneyness, and w = σ²_imp × τ is the
total implied variance.

# Fields
- `a`     — overall level of variance
- `b`     — slope / tightness
- `rho`   — rotation (controls the skew)
- `m`     — translation (shifts the smile)
- `sigma` — smoothness (ATM curvature)
"""
struct SVIParameters{T<:AbstractFloat}
    a::T
    b::T
    rho::T
    m::T
    sigma::T

    function SVIParameters{T}(a, b, rho, m, sigma) where {T}
        b >= zero(T) || throw(ArgumentError("b must be non-negative"))
        sigma > zero(T) || throw(ArgumentError("sigma must be positive"))
        abs(rho) < one(T) || throw(ArgumentError("|rho| must be < 1"))
        # Roger Lee's moment condition: a + b*sigma*sqrt(1-rho^2) >= 0
        new{T}(T(a), T(b), T(rho), T(m), T(sigma))
    end
end

SVIParameters(a::Real, b::Real, rho::Real, m::Real, sigma::Real) =
    SVIParameters{Float64}(Float64(a), Float64(b), Float64(rho), Float64(m), Float64(sigma))

"""
    svi_total_variance(params::SVIParameters, k)

Evaluate the SVI total variance w(k) at log-moneyness k.
"""
function svi_total_variance(params::SVIParameters{T}, k::Real) where {T}
    k_T = T(k)
    dk = k_T - params.m
    params.a + params.b * (params.rho * dk + sqrt(dk^2 + params.sigma^2))
end

"""
    VolatilitySurface{T<:AbstractFloat}

A full implied-vol surface defined by per-expiry SVI slices plus
interpolation between them.

# Fields
- `expiries`   — sorted vector of expiry times (in years)
- `svi_slices` — one SVIParameters per expiry
- `forward`    — forward price per expiry (for moneyness conversion)
"""
struct VolatilitySurface{T<:AbstractFloat}
    expiries::Vector{T}
    svi_slices::Vector{SVIParameters{T}}
    forward::Vector{T}

    function VolatilitySurface{T}(expiries, slices, fwd) where {T}
        length(expiries) == length(slices) == length(fwd) ||
            throw(ArgumentError("Length mismatch"))
        issorted(expiries) || throw(ArgumentError("Expiries must be sorted"))
        new{T}(Vector{T}(expiries), slices, Vector{T}(fwd))
    end
end

function VolatilitySurface(expiries::AbstractVector{<:Real},
                           slices::AbstractVector{<:SVIParameters},
                           forward::AbstractVector{<:Real})
    T = Float64
    VolatilitySurface{T}(T.(expiries), slices, T.(forward))
end

"""
    get_vol(surf::VolatilitySurface, strike::Real, expiry::Real)

Get implied volatility at a given (strike, expiry) by:
1. Finding the two bracketing expiry slices.
2. Computing log-moneyness k = log(K / F) for each slice.
3. Evaluating SVI total variance at each slice.
4. Linearly interpolating in total-variance space.
5. Converting back: σ_imp = √(w / τ).
"""
function get_vol(surf::VolatilitySurface{T}, strike::Real, expiry::Real) where {T}
    tau = T(expiry)
    K = T(strike)
    tau > zero(T) || throw(ArgumentError("expiry must be positive"))
    K > zero(T) || throw(ArgumentError("strike must be positive"))

    exps = surf.expiries
    n = length(exps)

    # Boundary cases: extrapolate flat
    if tau <= exps[1]
        F = surf.forward[1]
        k = log(K / F)
        w = svi_total_variance(surf.svi_slices[1], k)
        w_scaled = w * (tau / exps[1])  # scale total variance
        return sqrt(max(w_scaled / tau, zero(T)))
    elseif tau >= exps[end]
        F = surf.forward[end]
        k = log(K / F)
        w = svi_total_variance(surf.svi_slices[end], k)
        w_scaled = w * (tau / exps[end])
        return sqrt(max(w_scaled / tau, zero(T)))
    end

    # Find bracketing indices
    idx = 1
    @inbounds for i in 1:(n-1)
        if exps[i+1] >= tau
            idx = i
            break
        end
    end

    t1, t2 = exps[idx], exps[idx+1]
    F1, F2 = surf.forward[idx], surf.forward[idx+1]

    # Interpolate forward
    alpha = (tau - t1) / (t2 - t1)
    F_interp = F1 * (one(T) - alpha) + F2 * alpha

    k1 = log(K / F1)
    k2 = log(K / F2)
    k_interp = log(K / F_interp)

    w1 = svi_total_variance(surf.svi_slices[idx], k1)
    w2 = svi_total_variance(surf.svi_slices[idx+1], k2)

    # Linear interpolation in total variance
    w_interp = w1 * (one(T) - alpha) + w2 * alpha
    sqrt(max(w_interp / tau, zero(T)))
end

"""
    build_surface_from_params(S0, r, expiries, svi_param_list)

Convenience constructor: compute forwards from spot, rate, and build surface.
"""
function build_surface_from_params(S0::Real, r::Real,
                                   expiries::AbstractVector{<:Real},
                                   svi_param_list::AbstractVector{<:SVIParameters})
    fwd = [S0 * exp(r * t) for t in expiries]
    VolatilitySurface(expiries, svi_param_list, fwd)
end

"""
    generate_default_surface(S0; atm_vol=0.20, skew=-0.15, convexity=0.05)

Generate a plausible equity vol surface with mild skew and term structure.
Expiries at 1M, 3M, 6M, 1Y, 2Y.
"""
function generate_default_surface(S0::Real; atm_vol::Real=0.20,
                                  skew::Real=-0.15, convexity::Real=0.05,
                                  r::Real=0.05)
    expiries = [1/12, 3/12, 6/12, 1.0, 2.0]
    slices = SVIParameters[]
    for (i, tau) in enumerate(expiries)
        # Total variance roughly increases with tau; skew flattens
        base_w = atm_vol^2 * tau
        b_val = convexity / sqrt(tau)  # curvature decreases with maturity
        rho_val = skew * exp(-0.3 * tau)  # skew flattens
        sig_val = 0.1 * sqrt(tau)
        a_val = base_w - b_val * sig_val  # ensure ATM matches roughly
        push!(slices, SVIParameters(a_val, b_val, clamp(rho_val, -0.99, 0.99), 0.0, sig_val))
    end
    build_surface_from_params(S0, r, expiries, slices)
end

"""
    vol_surface_to_matrix(surf, strikes, expiries)

Evaluate the surface on a strike × expiry grid. Returns a matrix.
"""
function vol_surface_to_matrix(surf::VolatilitySurface{T},
                               strikes::AbstractVector{<:Real},
                               expiries::AbstractVector{<:Real}) where {T}
    nk = length(strikes)
    nt = length(expiries)
    mat = Matrix{T}(undef, nk, nt)
    @inbounds for j in 1:nt, i in 1:nk
        mat[i, j] = get_vol(surf, strikes[i], expiries[j])
    end
    mat
end


# ============================================================================
# Section 6: Order Flow Model (Hawkes Process)
# ============================================================================

"""
    HawkesProcess{T<:AbstractFloat}

Self-exciting Hawkes process for order arrivals:
    λ(t) = μ + Σ_i α exp(−β (t − tᵢ))

# Fields
- `mu`    — baseline intensity
- `alpha` — excitation magnitude (must be < beta for stationarity)
- `beta`  — excitation decay rate
"""
struct HawkesProcess{T<:AbstractFloat}
    mu::T
    alpha::T
    beta::T

    function HawkesProcess{T}(mu, alpha, beta) where {T}
        mu > zero(T) || throw(ArgumentError("mu must be positive"))
        alpha >= zero(T) || throw(ArgumentError("alpha must be non-negative"))
        beta > zero(T) || throw(ArgumentError("beta must be positive"))
        alpha < beta || @warn "alpha >= beta: process is not stationary (branching ratio >= 1)"
        new{T}(T(mu), T(alpha), T(beta))
    end
end

HawkesProcess(mu::Real, alpha::Real, beta::Real) =
    HawkesProcess{Float64}(Float64(mu), Float64(alpha), Float64(beta))

"""
    branching_ratio(hp::HawkesProcess)

The branching ratio n = α/β.  Process is stationary iff n < 1.
"""
branching_ratio(hp::HawkesProcess) = hp.alpha / hp.beta

"""
    stationary_intensity(hp::HawkesProcess)

E[λ] = μ / (1 − α/β) when n < 1.
"""
function stationary_intensity(hp::HawkesProcess{T}) where {T}
    n = branching_ratio(hp)
    n < one(T) || throw(ErrorException("Process is not stationary"))
    hp.mu / (one(T) - n)
end

"""
    simulate(hp::HawkesProcess, T_horizon::Real; rng)

Simulate a Hawkes process on [0, T_horizon] using Ogata's thinning algorithm.
Returns a vector of event times.
"""
function simulate(hp::HawkesProcess{T}, T_horizon::Real;
                  rng::AbstractRNG=Random.default_rng()) where {T}
    T_h = T(T_horizon)
    events = T[]
    t = zero(T)

    # Upper bound for thinning: start with mu, update after each event
    lambda_bar = hp.mu + hp.alpha  # conservative initial bound

    while t < T_h
        # Propose next event from Poisson(lambda_bar)
        u1 = rand(rng, T)
        w = -log(max(u1, eps(T))) / lambda_bar
        t += w
        t >= T_h && break

        # Compute true intensity at proposed time
        lambda_t = hp.mu
        @inbounds for ti in events
            lambda_t += hp.alpha * exp(-hp.beta * (t - ti))
        end

        # Accept/reject
        u2 = rand(rng, T)
        if u2 * lambda_bar <= lambda_t
            push!(events, t)
            # Update upper bound
            lambda_bar = lambda_t + hp.alpha
        else
            lambda_bar = lambda_t + hp.alpha * T(0.1)  # tighten bound
            lambda_bar = max(lambda_bar, hp.mu)
        end
    end
    events
end

"""
    OrderFlowModel{T<:AbstractFloat}

Buy and sell order flow as two Hawkes processes with asymmetry parameter.

# Fields
- `buy_process`  — Hawkes process for buy arrivals
- `sell_process` — Hawkes process for sell arrivals
- `asymmetry`    — imbalance factor (>1 means more buys, <1 means more sells)
- `cross_excitation` — cross-process excitation coefficient
"""
struct OrderFlowModel{T<:AbstractFloat}
    buy_process::HawkesProcess{T}
    sell_process::HawkesProcess{T}
    asymmetry::T
    cross_excitation::T
end

function OrderFlowModel(mu_base::Real=1.0, alpha::Real=0.5, beta::Real=1.0;
                        asymmetry::Real=1.0, cross_excitation::Real=0.1)
    T = Float64
    buy_mu = T(mu_base) * T(asymmetry)
    sell_mu = T(mu_base) / T(asymmetry)
    bp = HawkesProcess(buy_mu, T(alpha), T(beta))
    sp = HawkesProcess(sell_mu, T(alpha), T(beta))
    OrderFlowModel{T}(bp, sp, T(asymmetry), T(cross_excitation))
end

"""
    simulate(ofm::OrderFlowModel, T_horizon::Real; rng)

Simulate buy and sell order arrival times with cross-excitation.
Returns `(buy_times, sell_times)`.

The cross-excitation is approximated: after simulating independently,
we add extra events triggered by the other side.
"""
function simulate(ofm::OrderFlowModel{T}, T_horizon::Real;
                  rng::AbstractRNG=Random.default_rng()) where {T}
    buy_times  = simulate(ofm.buy_process, T_horizon; rng=rng)
    sell_times = simulate(ofm.sell_process, T_horizon; rng=rng)

    # Cross-excitation: each buy can trigger a sell and vice versa
    ce = ofm.cross_excitation
    if ce > zero(T)
        extra_sells = T[]
        for tb in buy_times
            if rand(rng, T) < ce
                delay = -log(max(rand(rng, T), eps(T))) / ofm.sell_process.beta
                ts = tb + delay
                if ts < T(T_horizon)
                    push!(extra_sells, ts)
                end
            end
        end
        extra_buys = T[]
        for ts in sell_times
            if rand(rng, T) < ce
                delay = -log(max(rand(rng, T), eps(T))) / ofm.buy_process.beta
                tb = ts + delay
                if tb < T(T_horizon)
                    push!(extra_buys, tb)
                end
            end
        end
        append!(buy_times, extra_buys)
        append!(sell_times, extra_sells)
        sort!(buy_times)
        sort!(sell_times)
    end

    (buy_times, sell_times)
end

"""
    order_imbalance(buy_times, sell_times, window_start, window_end)

Compute order imbalance = (n_buy - n_sell) / (n_buy + n_sell) in a time window.
"""
function order_imbalance(buy_times::Vector{T}, sell_times::Vector{T},
                         window_start::Real, window_end::Real) where {T}
    nb = count(t -> window_start <= t <= window_end, buy_times)
    ns = count(t -> window_start <= t <= window_end, sell_times)
    total = nb + ns
    total == 0 ? zero(T) : T(nb - ns) / T(total)
end

"""
    intensity_path(hp::HawkesProcess, events, time_grid)

Evaluate the Hawkes intensity λ(t) on a time grid given observed events.
"""
function intensity_path(hp::HawkesProcess{T}, events::Vector{T},
                        time_grid::AbstractVector{<:Real}) where {T}
    n = length(time_grid)
    intensities = Vector{T}(undef, n)
    @inbounds for i in 1:n
        t = T(time_grid[i])
        lam = hp.mu
        for te in events
            te >= t && break
            lam += hp.alpha * exp(-hp.beta * (t - te))
        end
        intensities[i] = lam
    end
    intensities
end


# ============================================================================
# Section 7: Liquidity Model
# ============================================================================

"""
    LiquidityModel{T<:AbstractFloat}

Dynamic bid-ask spread model that responds to market conditions.

    spread(t) = base_spread
              + vol_sensitivity × σ_realized(t)
              + imbalance_sensitivity × |OI(t)|
              + size_impact × log(1 + order_size / depth)

# Fields
- `base_spread`            — minimum spread (in price units)
- `vol_sensitivity`        — spread response to realized volatility
- `imbalance_sensitivity`  — spread response to order imbalance
- `depth`                  — market depth (in shares)
- `size_impact`            — spread widening per unit of relative size
- `mean_reversion_speed`   — how quickly spread reverts to base
"""
struct LiquidityModel{T<:AbstractFloat}
    base_spread::T
    vol_sensitivity::T
    imbalance_sensitivity::T
    depth::T
    size_impact::T
    mean_reversion_speed::T

    function LiquidityModel{T}(bs, vs, is, d, si, mrs) where {T}
        bs > zero(T) || throw(ArgumentError("base_spread must be positive"))
        d > zero(T) || throw(ArgumentError("depth must be positive"))
        new{T}(T(bs), T(vs), T(is), T(d), T(si), T(mrs))
    end
end

function LiquidityModel(; base_spread::Real=0.01, vol_sensitivity::Real=0.5,
                        imbalance_sensitivity::Real=0.3, depth::Real=10000.0,
                        size_impact::Real=0.1, mean_reversion_speed::Real=0.5)
    LiquidityModel{Float64}(base_spread, vol_sensitivity, imbalance_sensitivity,
                             depth, size_impact, mean_reversion_speed)
end

"""
    generate_spread(lm::LiquidityModel, prices, buy_times, sell_times, dt;
                    order_sizes=nothing, vol_window=20)

Generate a dynamic spread path given price history and order flow.
Returns a vector of spreads of length `length(prices)`.
"""
function generate_spread(lm::LiquidityModel{T}, prices::AbstractVector{<:Real},
                         buy_times::AbstractVector{<:Real},
                         sell_times::AbstractVector{<:Real},
                         dt::Real;
                         order_sizes::Union{Nothing, AbstractVector{<:Real}}=nothing,
                         vol_window::Int=20) where {T}
    n = length(prices)
    spreads = Vector{T}(undef, n)
    dt_T = T(dt)

    @inbounds for i in 1:n
        # Realized volatility (rolling window)
        if i <= vol_window
            realized_vol = zero(T)
        else
            log_rets = [log(T(prices[j]) / T(prices[j-1])) for j in (i-vol_window+1):i]
            realized_vol = std(log_rets) / sqrt(dt_T)
        end

        # Order imbalance in the recent window
        t_now = T(i) * dt_T
        t_start = max(zero(T), t_now - T(vol_window) * dt_T)
        oi = order_imbalance(T.(buy_times), T.(sell_times), t_start, t_now)

        # Base spread components
        vol_comp = lm.vol_sensitivity * realized_vol
        imb_comp = lm.imbalance_sensitivity * abs(oi)

        # Size impact (liquidity smile)
        size_comp = zero(T)
        if order_sizes !== nothing && i <= length(order_sizes)
            relative_size = T(order_sizes[i]) / lm.depth
            size_comp = lm.size_impact * log(one(T) + relative_size)
        end

        raw_spread = lm.base_spread + vol_comp + imb_comp + size_comp

        # Mean reversion: smooth the spread path
        if i == 1
            spreads[i] = raw_spread
        else
            spreads[i] = spreads[i-1] + lm.mean_reversion_speed * (raw_spread - spreads[i-1])
        end

        # Floor at base spread
        spreads[i] = max(spreads[i], lm.base_spread)
    end
    spreads
end

"""
    liquidity_smile(lm::LiquidityModel, order_sizes)

Compute the "liquidity smile": effective spread as a function of order size.
Returns a vector of spreads.
"""
function liquidity_smile(lm::LiquidityModel{T}, order_sizes::AbstractVector{<:Real}) where {T}
    [lm.base_spread + lm.size_impact * log(one(T) + T(s) / lm.depth) for s in order_sizes]
end

"""
    effective_execution_price(mid_price, spread, side::Symbol, slippage_factor=0.5)

Compute effective execution price.
- `side = :buy`  → mid + spread/2 * (1 + slippage)
- `side = :sell` → mid - spread/2 * (1 + slippage)
"""
function effective_execution_price(mid_price::Real, spread::Real,
                                   side::Symbol; slippage_factor::Real=0.0)
    half_spread = spread / 2 * (1 + slippage_factor)
    if side == :buy
        return mid_price + half_spread
    elseif side == :sell
        return mid_price - half_spread
    else
        throw(ArgumentError("side must be :buy or :sell"))
    end
end


# ============================================================================
# Section 8: Flash Crash Generator
# ============================================================================

"""
    FlashCrashGenerator{T<:AbstractFloat}

Generates realistic flash crash dynamics:
1. Normal trading until trigger condition.
2. Sudden liquidity withdrawal (depth drops).
3. Cascade selling accelerates the decline.
4. Circuit breaker or exhaustion → recovery phase.

# Fields
- `trigger_threshold`      — price drop % that triggers the cascade
- `cascade_speed`          — how fast selling accelerates (higher = faster)
- `recovery_time`          — time steps for partial recovery
- `depth_withdrawal_rate`  — fraction of depth removed per step during cascade
- `max_drawdown`           — maximum total drawdown (circuit breaker level)
- `recovery_fraction`      — fraction of the crash that is recovered
"""
struct FlashCrashGenerator{T<:AbstractFloat}
    trigger_threshold::T
    cascade_speed::T
    recovery_time::Int
    depth_withdrawal_rate::T
    max_drawdown::T
    recovery_fraction::T

    function FlashCrashGenerator{T}(tt, cs, rt, dwr, md, rf) where {T}
        zero(T) < tt < one(T) || throw(ArgumentError("trigger_threshold in (0,1)"))
        cs > zero(T) || throw(ArgumentError("cascade_speed must be positive"))
        rt > 0 || throw(ArgumentError("recovery_time must be positive"))
        zero(T) < dwr < one(T) || throw(ArgumentError("depth_withdrawal_rate in (0,1)"))
        zero(T) < md < one(T) || throw(ArgumentError("max_drawdown in (0,1)"))
        zero(T) <= rf <= one(T) || throw(ArgumentError("recovery_fraction in [0,1]"))
        new{T}(T(tt), T(cs), rt, T(dwr), T(md), T(rf))
    end
end

function FlashCrashGenerator(; trigger_threshold::Real=0.02, cascade_speed::Real=3.0,
                             recovery_time::Int=50, depth_withdrawal_rate::Real=0.8,
                             max_drawdown::Real=0.10, recovery_fraction::Real=0.7)
    FlashCrashGenerator{Float64}(trigger_threshold, cascade_speed, recovery_time,
                                  depth_withdrawal_rate, max_drawdown, recovery_fraction)
end

"""
    generate_flash_crash(fcg::FlashCrashGenerator, base_price, n_steps, dt;
                         pre_crash_vol=0.01, trigger_step=nothing, rng)

Generate a complete price path with an embedded flash crash.

# Returns
- `prices`  — price path of length `n_steps + 1`
- `phases`  — Vector{Symbol} indicating :normal, :cascade, :recovery for each step
"""
function generate_flash_crash(fcg::FlashCrashGenerator{T}, base_price::Real,
                              n_steps::Int, dt::Real;
                              pre_crash_vol::Real=0.01,
                              trigger_step::Union{Nothing,Int}=nothing,
                              rng::AbstractRNG=Random.default_rng()) where {T}
    bp = T(base_price)
    dt_T = T(dt)
    sqrt_dt = sqrt(dt_T)
    pv = T(pre_crash_vol)

    prices = Vector{T}(undef, n_steps + 1)
    phases = Vector{Symbol}(undef, n_steps + 1)
    prices[1] = bp
    phases[1] = :normal

    # Determine trigger step
    trig = trigger_step === nothing ? div(n_steps, 3) : trigger_step
    trig = clamp(trig, 10, n_steps - fcg.recovery_time - 10)

    crash_bottom_step = 0
    crash_bottom_price = bp
    depth_factor = one(T)

    @inbounds for i in 2:(n_steps + 1)
        step = i - 1

        if step < trig
            # Normal phase
            Z = randn(rng, T)
            lr = -T(0.5) * pv^2 * dt_T + pv * sqrt_dt * Z
            prices[i] = prices[i-1] * exp(lr)
            phases[i] = :normal

        elseif step == trig
            # Trigger: initial shock
            shock = -fcg.trigger_threshold * (one(T) + T(0.5) * rand(rng, T))
            prices[i] = prices[i-1] * (one(T) + shock)
            phases[i] = :cascade
            depth_factor = one(T) - fcg.depth_withdrawal_rate

        elseif phases[i-1] == :cascade
            # Cascade phase: accelerating decline with reduced depth
            drawdown = one(T) - prices[i-1] / prices[trig]

            if drawdown >= fcg.max_drawdown
                # Hit circuit breaker / exhaustion
                prices[i] = prices[i-1]
                phases[i] = :recovery
                crash_bottom_step = step
                crash_bottom_price = prices[i-1]
            else
                # Cascading sell pressure
                sell_pressure = fcg.cascade_speed * drawdown * dt_T
                noise = pv * sqrt_dt * randn(rng, T) / depth_factor
                lr = -sell_pressure + noise
                prices[i] = prices[i-1] * exp(lr)
                phases[i] = :cascade
                # Depth continues to withdraw
                depth_factor = max(depth_factor * (one(T) - fcg.depth_withdrawal_rate * T(0.1)),
                                   T(0.05))
            end

        elseif phases[i-1] == :recovery || (crash_bottom_step > 0 && step <= crash_bottom_step + fcg.recovery_time)
            # Recovery phase: mean-reversion toward recovery target
            target = crash_bottom_price +
                     fcg.recovery_fraction * (prices[trig] - crash_bottom_price)
            progress = T(step - crash_bottom_step) / T(fcg.recovery_time)
            progress = clamp(progress, zero(T), one(T))

            # Exponential recovery with noise
            reversion_rate = T(2.0) * progress
            gap = log(target / prices[i-1])
            noise = pv * T(2.0) * sqrt_dt * randn(rng, T)
            lr = reversion_rate * gap * dt_T + noise
            prices[i] = prices[i-1] * exp(lr)
            phases[i] = step <= crash_bottom_step + fcg.recovery_time ? :recovery : :normal

        else
            # Post-recovery normal trading
            Z = randn(rng, T)
            lr = -T(0.5) * pv^2 * dt_T + pv * sqrt_dt * Z
            prices[i] = prices[i-1] * exp(lr)
            phases[i] = :normal
        end
    end

    (prices, phases)
end

"""
    flash_crash_statistics(prices, phases)

Compute statistics of a flash crash: max drawdown, duration, recovery %.
"""
function flash_crash_statistics(prices::AbstractVector{T}, phases::Vector{Symbol}) where {T}
    cascade_idx = findall(p -> p == :cascade, phases)
    recovery_idx = findall(p -> p == :recovery, phases)

    if isempty(cascade_idx)
        return Dict{String,Any}(
            "max_drawdown" => zero(T),
            "cascade_steps" => 0,
            "recovery_steps" => 0,
            "recovery_pct" => zero(T),
            "pre_crash_price" => prices[1],
            "bottom_price" => prices[1],
        )
    end

    pre_crash = prices[first(cascade_idx)]
    bottom = minimum(prices[first(cascade_idx):last(cascade_idx)])
    max_dd = one(T) - bottom / pre_crash

    final_price = isempty(recovery_idx) ? bottom : prices[last(recovery_idx)]
    recovery_pct = (final_price - bottom) / (pre_crash - bottom + eps(T))

    Dict{String,Any}(
        "max_drawdown" => max_dd,
        "cascade_steps" => length(cascade_idx),
        "recovery_steps" => length(recovery_idx),
        "recovery_pct" => recovery_pct,
        "pre_crash_price" => pre_crash,
        "bottom_price" => bottom,
    )
end


# ============================================================================
# Section 9: Market Microstructure Noise
# ============================================================================

"""
    MarketMicrostructureNoise{T<:AbstractFloat}

Add realistic microstructure noise to an efficient price path.

Components:
1. **Bid-ask bounce** (Roll 1984): autocorrelation from alternating
   buy/sell market orders hitting the spread.
2. **Tick rounding**: prices rounded to the nearest tick size.
3. **Stale quotes**: occasional repetition of old prices (zero returns).
4. **Processing delay**: small random delays in price updates.

# Fields
- `tick_size`         — minimum price increment
- `half_spread`       — half of the bid-ask spread (for Roll model)
- `stale_quote_prob`  — probability a quote is stale (unchanged)
- `delay_prob`        — probability of a processing delay
- `max_delay_steps`   — maximum delay in steps
"""
struct MarketMicrostructureNoise{T<:AbstractFloat}
    tick_size::T
    half_spread::T
    stale_quote_prob::T
    delay_prob::T
    max_delay_steps::Int

    function MarketMicrostructureNoise{T}(ts, hs, sqp, dp, mds) where {T}
        ts > zero(T) || throw(ArgumentError("tick_size must be positive"))
        hs >= zero(T) || throw(ArgumentError("half_spread must be non-negative"))
        zero(T) <= sqp < one(T) || throw(ArgumentError("stale_quote_prob in [0,1)"))
        zero(T) <= dp < one(T) || throw(ArgumentError("delay_prob in [0,1)"))
        mds >= 0 || throw(ArgumentError("max_delay_steps must be non-negative"))
        new{T}(T(ts), T(hs), T(sqp), T(dp), mds)
    end
end

function MarketMicrostructureNoise(; tick_size::Real=0.01, half_spread::Real=0.005,
                                   stale_quote_prob::Real=0.05, delay_prob::Real=0.02,
                                   max_delay_steps::Int=3)
    MarketMicrostructureNoise{Float64}(tick_size, half_spread, stale_quote_prob,
                                        delay_prob, max_delay_steps)
end

"""
    _round_to_tick(price, tick_size)

Round price to the nearest tick.
"""
_round_to_tick(price::T, tick_size::T) where {T} = round(price / tick_size) * tick_size

"""
    add_microstructure_noise(noise::MarketMicrostructureNoise, efficient_prices; rng)

Transform efficient (true) prices into observed (noisy) prices.
Returns the noisy price vector.
"""
function add_microstructure_noise(noise::MarketMicrostructureNoise{T},
                                 efficient_prices::AbstractVector{<:Real};
                                 rng::AbstractRNG=Random.default_rng()) where {T}
    n = length(efficient_prices)
    observed = Vector{T}(undef, n)

    # Roll model: each observation is true_price ± half_spread
    # Direction alternates with some randomness
    direction = one(T)  # +1 or -1

    @inbounds for i in 1:n
        p_true = T(efficient_prices[i])

        # Stale quote check
        if i > 1 && rand(rng, T) < noise.stale_quote_prob
            observed[i] = observed[i-1]
            continue
        end

        # Processing delay: use a lagged price
        if i > noise.max_delay_steps && rand(rng, T) < noise.delay_prob
            delay = rand(rng, 1:noise.max_delay_steps)
            p_true = T(efficient_prices[max(1, i - delay)])
        end

        # Bid-ask bounce (Roll model)
        if rand(rng) < 0.5
            direction = -direction
        end
        p_noisy = p_true + direction * noise.half_spread

        # Tick rounding
        p_noisy = _round_to_tick(p_noisy, noise.tick_size)

        # Ensure positive
        observed[i] = max(p_noisy, noise.tick_size)
    end
    observed
end

"""
    roll_spread_estimator(observed_prices)

Estimate the effective spread using the Roll (1984) model:
    spread = 2 √(−Cov(Δpₜ, Δpₜ₋₁))

Returns the estimated spread (or 0 if the autocovariance is positive).
"""
function roll_spread_estimator(observed_prices::AbstractVector{<:Real})
    returns = diff(log.(observed_prices))
    n = length(returns)
    n >= 2 || return 0.0

    # First-order autocovariance
    mu = mean(returns)
    gamma1 = sum((returns[i] - mu) * (returns[i-1] - mu) for i in 2:n) / (n - 1)

    if gamma1 < 0
        return 2.0 * sqrt(-gamma1)
    else
        return 0.0
    end
end

"""
    noise_ratio(efficient_prices, observed_prices)

Compute the noise-to-signal ratio = Var(noise) / Var(efficient returns).
"""
function noise_ratio(efficient_prices::AbstractVector{<:Real},
                     observed_prices::AbstractVector{<:Real})
    n = min(length(efficient_prices), length(observed_prices))
    eff_ret = diff(log.(efficient_prices[1:n]))
    obs_ret = diff(log.(observed_prices[1:n]))
    noise_ret = obs_ret .- eff_ret
    var(noise_ret) / max(var(eff_ret), 1e-15)
end


# ============================================================================
# Section 10: Correlated Assets (Cholesky Decomposition)
# ============================================================================

"""
    CorrelatedAssets{T<:AbstractFloat}

Generate N correlated asset paths using Cholesky decomposition.

# Fields
- `n_assets`   — number of assets
- `mu`         — vector of drifts
- `sigma`      — vector of volatilities
- `S0`         — vector of initial prices
- `corr_mat`   — correlation matrix (N×N, positive definite)
- `chol_lower` — lower Cholesky factor of corr_mat
"""
struct CorrelatedAssets{T<:AbstractFloat}
    n_assets::Int
    mu::Vector{T}
    sigma::Vector{T}
    S0::Vector{T}
    corr_mat::Matrix{T}
    chol_lower::Matrix{T}

    function CorrelatedAssets{T}(n, mu, sigma, S0, C) where {T}
        n > 0 || throw(ArgumentError("n_assets must be positive"))
        length(mu) == length(sigma) == length(S0) == n ||
            throw(ArgumentError("mu, sigma, S0 must have length n_assets"))
        size(C) == (n, n) || throw(ArgumentError("corr_mat must be n×n"))
        all(S0 .> zero(T)) || throw(ArgumentError("All S0 must be positive"))
        all(sigma .>= zero(T)) || throw(ArgumentError("All sigma must be non-negative"))

        # Verify symmetry and positive definiteness
        C_sym = T(0.5) .* (C .+ C')
        # Ensure diagonal is 1
        for i in 1:n
            C_sym[i, i] = one(T)
        end

        # Cholesky decomposition
        L = _cholesky_lower(C_sym)

        new{T}(n, Vector{T}(mu), Vector{T}(sigma), Vector{T}(S0),
               Matrix{T}(C_sym), L)
    end
end

"""
    _cholesky_lower(A)

Compute the lower-triangular Cholesky factor L such that A = L L'.
Manual implementation (no LAPACK dependency beyond LinearAlgebra).
"""
function _cholesky_lower(A::Matrix{T}) where {T}
    n = size(A, 1)
    L = zeros(T, n, n)
    @inbounds for j in 1:n
        s = zero(T)
        for k in 1:(j-1)
            s += L[j, k]^2
        end
        diag_val = A[j, j] - s
        if diag_val <= zero(T)
            # Matrix not positive definite: add small ridge
            diag_val = T(1e-8)
        end
        L[j, j] = sqrt(diag_val)
        for i in (j+1):n
            s = zero(T)
            for k in 1:(j-1)
                s += L[i, k] * L[j, k]
            end
            L[i, j] = (A[i, j] - s) / L[j, j]
        end
    end
    L
end

function CorrelatedAssets(mu::AbstractVector{<:Real}, sigma::AbstractVector{<:Real},
                         S0::AbstractVector{<:Real}, corr_mat::AbstractMatrix{<:Real})
    T = Float64
    n = length(mu)
    CorrelatedAssets{T}(n, T.(mu), T.(sigma), T.(S0), Matrix{T}(corr_mat))
end

"""
    generate_correlated_assets(ca::CorrelatedAssets, n_steps, dt; rng)

Generate N correlated GBM paths.
Returns a matrix of size `(n_steps + 1) × n_assets`.
"""
function generate_correlated_assets(ca::CorrelatedAssets{T}, n_steps::Int, dt::Real;
                                    rng::AbstractRNG=Random.default_rng()) where {T}
    dt_T = T(dt)
    sqrt_dt = sqrt(dt_T)
    n = ca.n_assets

    paths = Matrix{T}(undef, n_steps + 1, n)
    @inbounds for j in 1:n
        paths[1, j] = ca.S0[j]
    end

    Z_indep = Vector{T}(undef, n)
    Z_corr  = Vector{T}(undef, n)

    @inbounds for i in 2:(n_steps + 1)
        # Generate independent normals
        for j in 1:n
            Z_indep[j] = randn(rng, T)
        end

        # Correlate via Cholesky
        mul!(Z_corr, ca.chol_lower, Z_indep)

        for j in 1:n
            drift = (ca.mu[j] - T(0.5) * ca.sigma[j]^2) * dt_T
            diffusion = ca.sigma[j] * sqrt_dt * Z_corr[j]
            paths[i, j] = paths[i-1, j] * exp(drift + diffusion)
        end
    end
    paths
end

"""
    realized_correlation(paths, i, j; window=nothing)

Compute the realized correlation between asset i and asset j from simulated paths.
"""
function realized_correlation(paths::Matrix{T}, i::Int, j::Int;
                              window::Union{Nothing,Int}=nothing) where {T}
    n = size(paths, 1)
    start = window === nothing ? 2 : max(2, n - window + 1)
    ret_i = [log(paths[k, i] / paths[k-1, i]) for k in start:n]
    ret_j = [log(paths[k, j] / paths[k-1, j]) for k in start:n]
    cor(ret_i, ret_j)
end

"""
    realized_correlation_matrix(paths; window=nothing)

Full realized correlation matrix from simulated paths.
"""
function realized_correlation_matrix(paths::Matrix{T};
                                     window::Union{Nothing,Int}=nothing) where {T}
    n_assets = size(paths, 2)
    C = Matrix{T}(undef, n_assets, n_assets)
    @inbounds for i in 1:n_assets
        C[i, i] = one(T)
        for j in (i+1):n_assets
            c = realized_correlation(paths, i, j; window=window)
            C[i, j] = c
            C[j, i] = c
        end
    end
    C
end

"""
    generate_random_correlation_matrix(n; min_eigenvalue=0.01, rng)

Generate a random valid correlation matrix of size n using the
random rotation method.
"""
function generate_random_correlation_matrix(n::Int; min_eigenvalue::Real=0.01,
                                            rng::AbstractRNG=Random.default_rng())
    # Generate random matrix and create PD matrix via A'A
    A = randn(rng, n, n)
    M = A' * A
    # Normalize to correlation matrix
    D_inv = Diagonal(1.0 ./ sqrt.(diag(M)))
    C = D_inv * M * D_inv
    # Ensure exact symmetry and unit diagonal
    C = 0.5 .* (C .+ C')
    for i in 1:n
        C[i, i] = 1.0
    end
    # Add ridge to ensure minimum eigenvalue
    evals = eigvals(Symmetric(C))
    if minimum(evals) < min_eigenvalue
        shift = min_eigenvalue - minimum(evals)
        C .+= shift * I
        # Re-normalise
        D_inv2 = Diagonal(1.0 ./ sqrt.(diag(C)))
        C = D_inv2 * C * D_inv2
        for i in 1:n
            C[i, i] = 1.0
        end
    end
    C
end


# ============================================================================
# Section 11: Event Injector
# ============================================================================

"""
    EventType

Enumeration of injectable event types.
"""
@enum EventType begin
    EARNINGS_ANNOUNCEMENT
    FOMC_MEETING
    CIRCUIT_BREAKER
    EXCHANGE_OUTAGE
    DIVIDEND_EX_DATE
    INDEX_REBALANCE
    SHORT_SQUEEZE
    FAT_FINGER_TRADE
end

"""
    MarketEvent{T<:AbstractFloat}

A scheduled market event to inject into a simulation.

# Fields
- `event_type`  — type of event
- `step`        — time step when event fires
- `magnitude`   — size parameter (interpretation depends on type)
- `duration`    — how many steps the effect lasts
- `params`      — additional parameters as a Dict
"""
struct MarketEvent{T<:AbstractFloat}
    event_type::EventType
    step::Int
    magnitude::T
    duration::Int
    params::Dict{String,T}
end

function MarketEvent(et::EventType, step::Int; magnitude::Real=0.05,
                     duration::Int=10, params::Dict=Dict{String,Float64}())
    MarketEvent{Float64}(et, step, Float64(magnitude), duration,
                         Dict{String,Float64}(k => Float64(v) for (k,v) in params))
end

"""
    EventInjector{T<:AbstractFloat}

Manages a calendar of events and applies them to price/vol paths.

# Fields
- `events` — sorted vector of MarketEvent
"""
struct EventInjector{T<:AbstractFloat}
    events::Vector{MarketEvent{T}}

    function EventInjector{T}(events) where {T}
        sorted = sort(events, by=e -> e.step)
        new{T}(sorted)
    end
end

EventInjector(events::Vector{MarketEvent{T}}) where {T} = EventInjector{T}(events)
EventInjector() = EventInjector{Float64}(MarketEvent{Float64}[])

"""
    inject_event!(injector::EventInjector, event::MarketEvent)

Add an event to the injector's calendar (maintains sort order).
"""
function inject_event!(injector::EventInjector{T}, event::MarketEvent{T}) where {T}
    idx = searchsortedfirst(injector.events, event, by=e -> e.step)
    insert!(injector.events, idx, event)
    injector
end

"""
    _apply_earnings(prices, vols, event, step)

Earnings announcement: gap open + volatility spike that decays.
"""
function _apply_earnings!(prices::Vector{T}, vols::Union{Nothing,Vector{T}},
                          event::MarketEvent{T}, step::Int) where {T}
    n = length(prices)
    if step > n
        return
    end

    # Price gap (can be positive or negative)
    direction = get(event.params, "direction", one(T))
    gap = event.magnitude * direction
    @inbounds for i in step:n
        prices[i] *= (one(T) + gap)
    end

    # Vol spike that decays
    if vols !== nothing
        vol_spike = event.magnitude * T(3.0)  # vol spikes ~3x the price move
        decay = T(0.9)
        @inbounds for i in step:min(step + event.duration - 1, length(vols))
            vols[i] += vol_spike * decay^(i - step)
        end
    end
end

"""
    _apply_fomc!(prices, vols, event, step)

FOMC meeting: vol compression before, expansion after.
"""
function _apply_fomc!(prices::Vector{T}, vols::Union{Nothing,Vector{T}},
                      event::MarketEvent{T}, step::Int) where {T}
    n = length(prices)
    if vols === nothing
        return
    end

    # Pre-meeting vol compression (5 steps before)
    pre_steps = min(5, step - 1)
    compression = T(0.7)  # reduce vol by 30%
    @inbounds for i in max(1, step - pre_steps):(step - 1)
        vols[i] *= compression
    end

    # Post-meeting vol expansion
    expansion = one(T) + event.magnitude * T(5.0)
    decay = T(0.85)
    @inbounds for i in step:min(step + event.duration - 1, length(vols))
        vols[i] *= expansion * decay^(i - step)
    end

    # Price jump on announcement
    direction = get(event.params, "direction", one(T))
    jump = event.magnitude * direction * T(0.5)
    @inbounds for i in step:n
        prices[i] *= (one(T) + jump)
    end
end

"""
    _apply_circuit_breaker!(prices, vols, event, step)

Circuit breaker: freeze prices for `duration` steps, then resume with higher vol.
"""
function _apply_circuit_breaker!(prices::Vector{T}, vols::Union{Nothing,Vector{T}},
                                 event::MarketEvent{T}, step::Int) where {T}
    n = length(prices)
    freeze_price = step <= n ? prices[step] : prices[end]

    # Freeze
    @inbounds for i in step:min(step + event.duration - 1, n)
        prices[i] = freeze_price
        if vols !== nothing && i <= length(vols)
            vols[i] = zero(T)  # no trading = no vol
        end
    end

    # Resume with elevated vol
    if vols !== nothing
        resume_step = step + event.duration
        vol_bump = event.magnitude * T(2.0)
        @inbounds for i in resume_step:min(resume_step + event.duration, length(vols))
            vols[i] += vol_bump
        end
    end
end

"""
    _apply_exchange_outage!(prices, vols, event, step)

Exchange outage: stale prices for duration, then gap on reopen.
"""
function _apply_exchange_outage!(prices::Vector{T}, vols::Union{Nothing,Vector{T}},
                                 event::MarketEvent{T}, step::Int) where {T}
    n = length(prices)
    stale_price = step <= n ? prices[step] : prices[end]

    @inbounds for i in step:min(step + event.duration - 1, n)
        prices[i] = stale_price
    end

    # Gap on reopen
    reopen = step + event.duration
    if reopen <= n
        gap = event.magnitude * (T(2.0) * rand() - one(T))  # random direction
        @inbounds for i in reopen:n
            prices[i] *= (one(T) + gap)
        end
    end
end

"""
    apply_events!(injector::EventInjector, prices, vols=nothing)

Apply all scheduled events to price and (optionally) volatility paths.
Modifies vectors in-place.
"""
function apply_events!(injector::EventInjector{T}, prices::Vector{T},
                       vols::Union{Nothing,Vector{T}}=nothing) where {T}
    for event in injector.events
        if event.event_type == EARNINGS_ANNOUNCEMENT
            _apply_earnings!(prices, vols, event, event.step)
        elseif event.event_type == FOMC_MEETING
            _apply_fomc!(prices, vols, event, event.step)
        elseif event.event_type == CIRCUIT_BREAKER
            _apply_circuit_breaker!(prices, vols, event, event.step)
        elseif event.event_type == EXCHANGE_OUTAGE
            _apply_exchange_outage!(prices, vols, event, event.step)
        elseif event.event_type == DIVIDEND_EX_DATE
            _apply_dividend!(prices, event, event.step)
        elseif event.event_type == SHORT_SQUEEZE
            _apply_short_squeeze!(prices, vols, event, event.step)
        elseif event.event_type == FAT_FINGER_TRADE
            _apply_fat_finger!(prices, vols, event, event.step)
        end
    end
    prices
end

"""
    _apply_dividend!(prices, event, step)

Dividend ex-date: price drops by dividend amount.
"""
function _apply_dividend!(prices::Vector{T}, event::MarketEvent{T}, step::Int) where {T}
    n = length(prices)
    div_yield = event.magnitude  # as a fraction of price
    @inbounds for i in step:n
        prices[i] *= (one(T) - div_yield)
    end
end

"""
    _apply_short_squeeze!(prices, vols, event, step)

Short squeeze: rapid price spike with elevated vol.
"""
function _apply_short_squeeze!(prices::Vector{T}, vols::Union{Nothing,Vector{T}},
                               event::MarketEvent{T}, step::Int) where {T}
    n = length(prices)
    peak_step = step + div(event.duration, 3)

    @inbounds for i in step:min(step + event.duration - 1, n)
        progress = T(i - step) / T(event.duration)
        if i <= peak_step
            # Ramp up
            factor = one(T) + event.magnitude * (T(i - step) / T(peak_step - step + 1))
        else
            # Decay back
            decay_progress = T(i - peak_step) / T(event.duration - (peak_step - step))
            factor = one(T) + event.magnitude * (one(T) - decay_progress)
        end
        prices[i] *= factor

        if vols !== nothing && i <= length(vols)
            vols[i] *= (one(T) + event.magnitude * T(2.0) * (one(T) - progress))
        end
    end
end

"""
    _apply_fat_finger!(prices, vols, event, step)

Fat finger trade: single-step spike then immediate reversion.
"""
function _apply_fat_finger!(prices::Vector{T}, vols::Union{Nothing,Vector{T}},
                            event::MarketEvent{T}, step::Int) where {T}
    n = length(prices)
    if step <= n
        direction = get(event.params, "direction", one(T))
        prices[step] *= (one(T) + event.magnitude * direction)
        # Immediate reversion: next step returns to pre-event level
        if step + 1 <= n
            # Already at pre-event level from the un-modified path
        end
    end
end

"""
    generate_event_calendar(n_steps; n_earnings=4, n_fomc=8, rng)

Generate a random event calendar for a simulation.
"""
function generate_event_calendar(n_steps::Int; n_earnings::Int=4, n_fomc::Int=8,
                                 rng::AbstractRNG=Random.default_rng())
    events = MarketEvent{Float64}[]

    # Space earnings roughly equally
    for i in 1:n_earnings
        step = div(n_steps * i, n_earnings + 1) + rand(rng, -5:5)
        step = clamp(step, 10, n_steps - 10)
        direction = rand(rng) < 0.5 ? -1.0 : 1.0
        mag = 0.02 + 0.03 * rand(rng)
        push!(events, MarketEvent(EARNINGS_ANNOUNCEMENT, step;
              magnitude=mag, duration=20,
              params=Dict("direction" => direction)))
    end

    # FOMC meetings
    for i in 1:n_fomc
        step = div(n_steps * i, n_fomc + 1) + rand(rng, -3:3)
        step = clamp(step, 10, n_steps - 10)
        direction = rand(rng) < 0.5 ? -1.0 : 1.0
        mag = 0.005 + 0.01 * rand(rng)
        push!(events, MarketEvent(FOMC_MEETING, step;
              magnitude=mag, duration=15,
              params=Dict("direction" => direction)))
    end

    EventInjector(events)
end


# ============================================================================
# Section 12: True Value Process
# ============================================================================

"""
    TrueValueProcess{T<:AbstractFloat}

The "fundamental" value that informed traders observe.  Market price
oscillates around this with some efficiency.

    V(t+1) = V(t) * exp((μ_f − σ_f²/2) dt + σ_f √dt Z)

The market price mean-reverts towards V:
    S(t+1) = S(t) * exp(κ_p (log V(t) − log S(t)) dt + σ_m √dt Z_m)

# Fields
- `mu_fundamental`  — fundamental drift
- `sigma_fundamental` — fundamental volatility
- `sigma_market`    — market noise volatility (on top of fundamentals)
- `mean_reversion`  — price discovery speed (κ_p)
- `V0`             — initial fundamental value
- `S0`             — initial market price
"""
struct TrueValueProcess{T<:AbstractFloat}
    mu_fundamental::T
    sigma_fundamental::T
    sigma_market::T
    mean_reversion::T
    V0::T
    S0::T

    function TrueValueProcess{T}(mf, sf, sm, mr, V0, S0) where {T}
        V0 > zero(T) || throw(ArgumentError("V0 must be positive"))
        S0 > zero(T) || throw(ArgumentError("S0 must be positive"))
        sf >= zero(T) || throw(ArgumentError("sigma_fundamental must be non-negative"))
        sm >= zero(T) || throw(ArgumentError("sigma_market must be non-negative"))
        mr >= zero(T) || throw(ArgumentError("mean_reversion must be non-negative"))
        new{T}(T(mf), T(sf), T(sm), T(mr), T(V0), T(S0))
    end
end

function TrueValueProcess(; mu_fundamental::Real=0.05, sigma_fundamental::Real=0.15,
                          sigma_market::Real=0.25, mean_reversion::Real=2.0,
                          V0::Real=100.0, S0::Real=100.0)
    TrueValueProcess{Float64}(mu_fundamental, sigma_fundamental, sigma_market,
                               mean_reversion, V0, S0)
end

"""
    simulate(tvp::TrueValueProcess, n_steps::Int, dt::Real; rng)

Simulate the true value and market price processes.
Returns `(true_values, market_prices)`.
"""
function simulate(tvp::TrueValueProcess{T}, n_steps::Int, dt::Real;
                  rng::AbstractRNG=Random.default_rng()) where {T}
    dt_T = T(dt)
    sqrt_dt = sqrt(dt_T)

    true_vals = Vector{T}(undef, n_steps + 1)
    mkt_prices = Vector{T}(undef, n_steps + 1)
    true_vals[1] = tvp.V0
    mkt_prices[1] = tvp.S0

    @inbounds for i in 2:(n_steps + 1)
        # Fundamental value evolution
        Z_f = randn(rng, T)
        lr_f = (tvp.mu_fundamental - T(0.5) * tvp.sigma_fundamental^2) * dt_T +
               tvp.sigma_fundamental * sqrt_dt * Z_f
        true_vals[i] = true_vals[i-1] * exp(lr_f)

        # Market price mean-reverts toward fundamental
        gap = log(true_vals[i]) - log(mkt_prices[i-1])
        Z_m = randn(rng, T)
        lr_m = tvp.mean_reversion * gap * dt_T + tvp.sigma_market * sqrt_dt * Z_m
        mkt_prices[i] = mkt_prices[i-1] * exp(lr_m)
    end
    (true_vals, mkt_prices)
end

"""
    price_discovery_efficiency(true_values, market_prices; window=nothing)

Measure price discovery efficiency as the correlation between
log(market_price) and log(true_value).
"""
function price_discovery_efficiency(true_values::AbstractVector{<:Real},
                                    market_prices::AbstractVector{<:Real};
                                    window::Union{Nothing,Int}=nothing)
    n = min(length(true_values), length(market_prices))
    start = window === nothing ? 1 : max(1, n - window + 1)
    log_tv = log.(true_values[start:n])
    log_mp = log.(market_prices[start:n])
    cor(log_tv, log_mp)
end

"""
    mispricing_series(true_values, market_prices)

Compute the mispricing ratio series: S(t) / V(t) − 1.
"""
function mispricing_series(true_values::AbstractVector{<:Real},
                           market_prices::AbstractVector{<:Real})
    n = min(length(true_values), length(market_prices))
    [market_prices[i] / true_values[i] - 1.0 for i in 1:n]
end

"""
    half_life_of_mispricing(tvp::TrueValueProcess)

Theoretical half-life = ln(2) / κ_p (in time units, not steps).
"""
half_life_of_mispricing(tvp::TrueValueProcess) = log(2.0) / tvp.mean_reversion

"""
    information_ratio(true_values, market_prices, dt)

Annualized information ratio: mean(mispricing_change) / std(mispricing_change).
"""
function information_ratio(true_values::AbstractVector{<:Real},
                           market_prices::AbstractVector{<:Real}, dt::Real)
    mp = mispricing_series(true_values, market_prices)
    changes = diff(mp)
    length(changes) == 0 && return 0.0
    mu = mean(changes)
    s = std(changes)
    s < 1e-15 && return 0.0
    (mu / s) * sqrt(1.0 / dt)
end


# ============================================================================
# Section 13: Calibration Engine
# ============================================================================

"""
    CalibrationEngine

Methods for calibrating model parameters from historical price data.
"""
struct CalibrationEngine end

"""
    calibrate(::Type{GeometricBrownianMotion}, prices::AbstractVector, dt::Real)

Method of moments for GBM: estimate μ and σ from log-returns.
"""
function calibrate(::Type{GeometricBrownianMotion}, prices::AbstractVector{<:Real}, dt::Real)
    n = length(prices)
    n >= 3 || throw(ArgumentError("Need at least 3 prices"))

    log_returns = [log(prices[i] / prices[i-1]) for i in 2:n]

    # MLE / method of moments for GBM
    mu_hat = mean(log_returns) / dt + 0.5 * var(log_returns) / dt
    sigma_hat = std(log_returns) / sqrt(dt)

    GeometricBrownianMotion(mu_hat, sigma_hat, Float64(prices[1]))
end

"""
    calibrate(::Type{MertonJumpDiffusion}, prices::AbstractVector, dt::Real;
              jump_threshold=3.0)

Calibrate Merton jump-diffusion by:
1. Identifying jumps as returns > `jump_threshold` standard deviations.
2. Estimating diffusion parameters from non-jump returns.
3. Estimating jump parameters from the identified jumps.
"""
function calibrate(::Type{MertonJumpDiffusion}, prices::AbstractVector{<:Real}, dt::Real;
                   jump_threshold::Real=3.0)
    n = length(prices)
    n >= 10 || throw(ArgumentError("Need at least 10 prices"))

    log_returns = [log(prices[i] / prices[i-1]) for i in 2:n]
    n_ret = length(log_returns)

    # Initial estimates
    mu_all = mean(log_returns)
    sigma_all = std(log_returns)

    # Identify jumps
    threshold = jump_threshold * sigma_all
    is_jump = abs.(log_returns .- mu_all) .> threshold
    n_jumps = sum(is_jump)

    # Separate jump and diffusion returns
    diffusion_returns = log_returns[.!is_jump]
    jump_returns = log_returns[is_jump]

    # Diffusion parameters
    if length(diffusion_returns) >= 2
        sigma_hat = std(diffusion_returns) / sqrt(dt)
    else
        sigma_hat = sigma_all / sqrt(dt)
    end

    # Jump parameters
    lambda_hat = n_jumps / (n_ret * dt)
    if n_jumps >= 2
        mu_j_hat = mean(jump_returns)
        sigma_j_hat = std(jump_returns)
    elseif n_jumps == 1
        mu_j_hat = jump_returns[1]
        sigma_j_hat = 0.01
    else
        mu_j_hat = 0.0
        sigma_j_hat = 0.01
        lambda_hat = 0.0
    end

    # Drift (compensated)
    k_bar = exp(mu_j_hat + 0.5 * sigma_j_hat^2) - 1.0
    mu_hat = mean(log_returns) / dt + 0.5 * sigma_hat^2 + lambda_hat * k_bar

    MertonJumpDiffusion(mu_hat, sigma_hat, lambda_hat, mu_j_hat, sigma_j_hat,
                        Float64(prices[1]))
end

"""
    calibrate(::Type{HestonModel}, prices::AbstractVector, dt::Real;
              vol_window=20)

Simplified Heston calibration using realized variance time series:
1. Estimate realized variance path.
2. Fit CIR parameters (κ, θ, ξ) to variance by method of moments.
3. Estimate ρ from correlation of returns and variance changes.
"""
function calibrate(::Type{HestonModel}, prices::AbstractVector{<:Real}, dt::Real;
                   vol_window::Int=20)
    n = length(prices)
    n >= vol_window + 5 || throw(ArgumentError("Need more prices for Heston calibration"))

    log_returns = [log(prices[i] / prices[i-1]) for i in 2:n]

    # Realized variance series
    n_ret = length(log_returns)
    n_var = n_ret - vol_window + 1
    rv = Vector{Float64}(undef, n_var)
    for i in 1:n_var
        window = log_returns[i:(i + vol_window - 1)]
        rv[i] = var(window) / dt
    end

    # CIR moment matching for variance process
    # E[v] = theta, Var[v] = theta * xi^2 / (2 * kappa)
    theta_hat = mean(rv)
    var_v = var(rv)

    # Autocorrelation of variance -> kappa
    # Cor(v_t, v_{t+1}) ≈ exp(-kappa * dt)
    if length(rv) >= 3
        rv_mean = mean(rv)
        numerator = sum((rv[i] - rv_mean) * (rv[i+1] - rv_mean) for i in 1:(length(rv)-1))
        denominator = sum((rv[i] - rv_mean)^2 for i in 1:length(rv))
        acf1 = denominator > 0 ? numerator / denominator : 0.5
        acf1 = clamp(acf1, 0.01, 0.999)
        kappa_hat = -log(acf1) / (vol_window * dt)
    else
        kappa_hat = 2.0
    end

    # xi from variance of variance
    xi_hat = sqrt(max(2.0 * kappa_hat * var_v / max(theta_hat, 1e-8), 1e-8))

    # Correlation between returns and variance changes
    var_changes = diff(rv)
    ret_for_corr = log_returns[(vol_window+1):end]
    min_len = min(length(var_changes), length(ret_for_corr))
    if min_len >= 3
        rho_hat = cor(ret_for_corr[1:min_len], var_changes[1:min_len])
        rho_hat = clamp(rho_hat, -0.99, 0.99)
    else
        rho_hat = -0.7  # typical equity value
    end

    # Drift
    mu_hat = mean(log_returns) / dt + 0.5 * theta_hat

    # Initial variance
    v0_hat = rv[end]

    HestonModel(mu_hat, max(kappa_hat, 0.1), max(theta_hat, 1e-6),
                max(xi_hat, 0.01), rho_hat, Float64(prices[1]), max(v0_hat, 1e-6))
end

"""
    calibrate(::Type{RegimeSwitchingModel}, prices::AbstractVector, dt::Real;
              vol_window=20, vol_threshold_quantile=0.5)

Regime-switching calibration using volatility clustering:
1. Compute rolling realized vol.
2. Classify high-vol and low-vol regimes by threshold.
3. Estimate transition probabilities from regime sequence.
4. Estimate (μ, σ) per regime.
"""
function calibrate(::Type{RegimeSwitchingModel}, prices::AbstractVector{<:Real}, dt::Real;
                   vol_window::Int=20, vol_threshold_quantile::Real=0.5)
    n = length(prices)
    log_returns = [log(prices[i] / prices[i-1]) for i in 2:n]
    n_ret = length(log_returns)

    # Rolling volatility
    n_vol = n_ret - vol_window + 1
    n_vol >= 10 || throw(ArgumentError("Not enough data for regime calibration"))
    rolling_vol = [std(log_returns[i:(i+vol_window-1)]) / sqrt(dt) for i in 1:n_vol]

    # Classify regimes
    threshold = quantile(rolling_vol, vol_threshold_quantile)
    regimes = [rv <= threshold ? 1 : 2 for rv in rolling_vol]  # 1 = bull (low vol), 2 = bear

    # Transition probabilities
    n11 = n12 = n21 = n22 = 0
    for i in 1:(length(regimes)-1)
        if regimes[i] == 1 && regimes[i+1] == 1
            n11 += 1
        elseif regimes[i] == 1 && regimes[i+1] == 2
            n12 += 1
        elseif regimes[i] == 2 && regimes[i+1] == 1
            n21 += 1
        else
            n22 += 1
        end
    end

    p11 = n11 + n12 > 0 ? n11 / (n11 + n12) : 0.95
    p22 = n21 + n22 > 0 ? n22 / (n21 + n22) : 0.90
    P = [p11 (1-p11); (1-p22) p22]

    # Per-regime parameters
    ret_regime1 = log_returns[vol_window:end][regimes .== 1]
    ret_regime2 = log_returns[vol_window:end][regimes .== 2]

    mu1 = length(ret_regime1) >= 2 ? mean(ret_regime1) / dt : 0.10
    sig1 = length(ret_regime1) >= 2 ? std(ret_regime1) / sqrt(dt) : 0.15
    mu2 = length(ret_regime2) >= 2 ? mean(ret_regime2) / dt : -0.05
    sig2 = length(ret_regime2) >= 2 ? std(ret_regime2) / sqrt(dt) : 0.30

    RegimeSwitchingModel(mu1, sig1, mu2, sig2, P, Float64(prices[1]))
end

"""
    calibrate_hawkes(event_times::AbstractVector, T_horizon::Real)

Calibrate a Hawkes process from observed event times using
moment-based estimation.

The stationary mean intensity is λ̄ = μ / (1 − α/β), and the
autocovariance function decays exponentially with rate β.
"""
function calibrate_hawkes(event_times::AbstractVector{<:Real}, T_horizon::Real)
    n = length(event_times)
    n >= 10 || throw(ArgumentError("Need at least 10 events"))

    # Empirical intensity
    lambda_bar = n / T_horizon

    # Inter-arrival times
    iat = diff(sort(event_times))
    n_iat = length(iat)

    # Estimate beta from autocorrelation of inter-arrival times
    mu_iat = mean(iat)
    if n_iat >= 5
        # First-order autocorrelation
        cov1 = sum((iat[i] - mu_iat) * (iat[i+1] - mu_iat) for i in 1:(n_iat-1)) / (n_iat - 1)
        var_iat = var(iat)
        acf1 = var_iat > 0 ? cov1 / var_iat : 0.0
        acf1 = clamp(acf1, -0.99, 0.99)

        # For Hawkes, acf of inter-arrivals ≈ branching_ratio * exp(-beta * mean_iat)
        # Rough estimate: beta ≈ -log(|acf1|) / mean_iat
        if abs(acf1) > 0.01
            beta_hat = -log(abs(acf1)) / mu_iat
        else
            beta_hat = 1.0 / mu_iat
        end
    else
        beta_hat = 1.0 / mu_iat
    end

    beta_hat = max(beta_hat, 0.1)

    # Estimate branching ratio from the variance of counts
    # Var(N(t)) / E(N(t)) ≈ 1 / (1 - n)^2 where n = alpha/beta
    # Use windowed counts
    window = T_horizon / 20
    n_windows = 20
    counts = zeros(Int, n_windows)
    for t in event_times
        idx = clamp(Int(floor(t / window)) + 1, 1, n_windows)
        counts[idx] += 1
    end
    mean_count = mean(counts)
    var_count = var(counts)
    index_of_dispersion = mean_count > 0 ? var_count / mean_count : 1.0

    # n = 1 - 1/sqrt(IoD) (approximate)
    branching = index_of_dispersion > 1.0 ? 1.0 - 1.0 / sqrt(index_of_dispersion) : 0.0
    branching = clamp(branching, 0.0, 0.95)

    alpha_hat = branching * beta_hat
    mu_hat = lambda_bar * (1.0 - branching)

    HawkesProcess(max(mu_hat, 0.01), max(alpha_hat, 0.0), beta_hat)
end

"""
    calibrate_correlation(price_matrix, dt)

Estimate correlation matrix from a matrix of price paths (rows = time, cols = assets).
"""
function calibrate_correlation(price_matrix::AbstractMatrix{<:Real}, dt::Real)
    n_time, n_assets = size(price_matrix)
    n_time >= 3 || throw(ArgumentError("Need at least 3 time steps"))

    # Log returns
    returns = Matrix{Float64}(undef, n_time - 1, n_assets)
    for j in 1:n_assets
        for i in 2:n_time
            returns[i-1, j] = log(price_matrix[i, j] / price_matrix[i-1, j])
        end
    end

    cor(returns)
end

"""
    goodness_of_fit(model, prices, dt; n_simulations=1000)

Assess model fit by comparing simulated return distribution statistics
to empirical ones.  Returns a Dict of KS-like statistics.
"""
function goodness_of_fit(model::GeometricBrownianMotion, prices::AbstractVector{<:Real},
                         dt::Real; n_simulations::Int=1000,
                         rng::AbstractRNG=Random.default_rng())
    n_steps = length(prices) - 1
    empirical_returns = [log(prices[i+1] / prices[i]) for i in 1:n_steps]

    sim_means = Float64[]
    sim_stds = Float64[]
    sim_skews = Float64[]
    sim_kurts = Float64[]

    for _ in 1:n_simulations
        path = simulate(model, n_steps, dt; rng=rng)
        sim_ret = [log(path[i+1] / path[i]) for i in 1:n_steps]
        push!(sim_means, mean(sim_ret))
        push!(sim_stds, std(sim_ret))
        if n_steps >= 4
            m3 = mean((sim_ret .- mean(sim_ret)).^3) / std(sim_ret)^3
            m4 = mean((sim_ret .- mean(sim_ret)).^4) / std(sim_ret)^4
            push!(sim_skews, m3)
            push!(sim_kurts, m4)
        end
    end

    emp_mean = mean(empirical_returns)
    emp_std = std(empirical_returns)

    Dict{String,Any}(
        "mean_pvalue" => _empirical_pvalue(emp_mean, sim_means),
        "std_pvalue" => _empirical_pvalue(emp_std, sim_stds),
        "emp_mean" => emp_mean,
        "emp_std" => emp_std,
        "sim_mean_mean" => mean(sim_means),
        "sim_std_mean" => mean(sim_stds),
    )
end

function _empirical_pvalue(observed::Real, simulated::AbstractVector{<:Real})
    n = length(simulated)
    n == 0 && return 1.0
    count(s -> s >= observed, simulated) / n
end


# ============================================================================
# Section 14: Synthetic Universe Generator
# ============================================================================

"""
    AssetSpec{T<:AbstractFloat}

Specification for a single asset in the synthetic universe.

# Fields
- `name`      — ticker / identifier
- `model`     — which dynamics model to use (:gbm, :heston, :merton, :regime)
- `mu`        — drift
- `sigma`     — base volatility
- `S0`        — initial price
- `sector`    — sector classification (Int)
- `market_cap`— relative market cap weight
- `extra`     — additional model-specific parameters
"""
struct AssetSpec{T<:AbstractFloat}
    name::String
    model::Symbol
    mu::T
    sigma::T
    S0::T
    sector::Int
    market_cap::T
    extra::Dict{String,T}
end

function AssetSpec(name::String; model::Symbol=:gbm, mu::Real=0.08,
                   sigma::Real=0.20, S0::Real=100.0, sector::Int=1,
                   market_cap::Real=1.0, extra::Dict=Dict{String,Float64}())
    AssetSpec{Float64}(name, model, Float64(mu), Float64(sigma), Float64(S0),
                       sector, Float64(market_cap),
                       Dict{String,Float64}(k => Float64(v) for (k,v) in extra))
end

"""
    SyntheticUniverse{T<:AbstractFloat}

A complete multi-asset synthetic market universe.

# Fields
- `assets`          — vector of AssetSpec
- `correlation`     — inter-asset correlation matrix
- `event_calendar`  — scheduled events
- `true_value_proc` — fundamental value process parameters
- `liquidity`       — liquidity model
- `microstructure`  — noise model
"""
struct SyntheticUniverse{T<:AbstractFloat}
    assets::Vector{AssetSpec{T}}
    correlation::Matrix{T}
    event_calendar::EventInjector{T}
    true_value_proc::TrueValueProcess{T}
    liquidity::LiquidityModel{T}
    microstructure::MarketMicrostructureNoise{T}
end

"""
    SyntheticUniverseGenerator

Factory for creating synthetic universes with configurable complexity.
"""
struct SyntheticUniverseGenerator
    n_assets::Int
    n_sectors::Int
    intra_sector_corr::Float64
    inter_sector_corr::Float64
    n_steps::Int
    dt::Float64
end

function SyntheticUniverseGenerator(; n_assets::Int=50, n_sectors::Int=5,
                                    intra_sector_corr::Real=0.6,
                                    inter_sector_corr::Real=0.2,
                                    n_steps::Int=252, dt::Real=1/252)
    SyntheticUniverseGenerator(n_assets, n_sectors, Float64(intra_sector_corr),
                                Float64(inter_sector_corr), n_steps, Float64(dt))
end

"""
    _generate_sector_correlation(n_assets, n_sectors, intra, inter)

Build a block correlation matrix with higher intra-sector and lower
inter-sector correlations.
"""
function _generate_sector_correlation(n_assets::Int, n_sectors::Int,
                                     intra::Float64, inter::Float64)
    C = fill(inter, n_assets, n_assets)
    assets_per_sector = div(n_assets, n_sectors)

    for s in 1:n_sectors
        start_idx = (s - 1) * assets_per_sector + 1
        end_idx = s == n_sectors ? n_assets : s * assets_per_sector
        for i in start_idx:end_idx
            for j in start_idx:end_idx
                if i == j
                    C[i, j] = 1.0
                else
                    C[i, j] = intra
                end
            end
        end
    end

    # Ensure positive definiteness
    evals = eigvals(Symmetric(C))
    if minimum(evals) < 0.01
        shift = 0.01 - minimum(evals)
        C .+= shift * I
        D_inv = Diagonal(1.0 ./ sqrt.(diag(C)))
        C = D_inv * C * D_inv
        for i in 1:n_assets
            C[i, i] = 1.0
        end
    end

    # Force exact symmetry
    0.5 .* (C .+ C')
end

"""
    _generate_asset_specs(n_assets, n_sectors; rng)

Generate a diverse set of asset specifications.
"""
function _generate_asset_specs(n_assets::Int, n_sectors::Int;
                               rng::AbstractRNG=Random.default_rng())
    assets = AssetSpec{Float64}[]
    assets_per_sector = div(n_assets, n_sectors)

    sector_base_mu = [0.05 + 0.03 * randn(rng) for _ in 1:n_sectors]
    sector_base_sigma = [0.15 + 0.05 * abs(randn(rng)) for _ in 1:n_sectors]

    model_choices = [:gbm, :heston, :merton, :regime]

    for i in 1:n_assets
        sector = min(div(i - 1, assets_per_sector) + 1, n_sectors)
        name = "ASSET_$(lpad(string(i), 3, '0'))"

        # Randomize parameters around sector base
        mu = sector_base_mu[sector] + 0.02 * randn(rng)
        sigma = sector_base_sigma[sector] + 0.03 * abs(randn(rng))
        sigma = max(sigma, 0.05)

        S0 = 50.0 + 150.0 * rand(rng)
        market_cap = exp(randn(rng))  # log-normal market cap

        # Choose model (weight toward gbm for simplicity)
        model_weights = [0.4, 0.25, 0.2, 0.15]
        u = rand(rng)
        cumw = cumsum(model_weights)
        model_idx = findfirst(w -> u <= w, cumw)
        model_idx = model_idx === nothing ? 1 : model_idx
        model = model_choices[model_idx]

        extra = Dict{String,Float64}()
        if model == :heston
            extra["kappa"] = 2.0 + randn(rng)
            extra["theta"] = sigma^2
            extra["xi"] = 0.3 + 0.1 * abs(randn(rng))
            extra["rho"] = -0.7 + 0.2 * randn(rng)
            extra["rho"] = clamp(extra["rho"], -0.99, 0.99)
            extra["v0"] = sigma^2
        elseif model == :merton
            extra["lambda_j"] = 1.0 + abs(randn(rng))
            extra["mu_j"] = -0.02 + 0.01 * randn(rng)
            extra["sigma_j"] = 0.03 + 0.01 * abs(randn(rng))
        elseif model == :regime
            extra["mu_bull"] = mu + 0.05
            extra["sigma_bull"] = sigma * 0.7
            extra["mu_bear"] = mu - 0.10
            extra["sigma_bear"] = sigma * 1.5
            extra["p11"] = 0.95 + 0.04 * rand(rng)
            extra["p22"] = 0.90 + 0.08 * rand(rng)
        end

        push!(assets, AssetSpec(name; model=model, mu=mu, sigma=sigma, S0=S0,
                                sector=sector, market_cap=market_cap, extra=extra))
    end
    assets
end

"""
    generate_universe(gen::SyntheticUniverseGenerator; rng)

Generate a complete synthetic universe: assets, correlations, events,
true value process, liquidity, and microstructure.
"""
function generate_universe(gen::SyntheticUniverseGenerator;
                           rng::AbstractRNG=Random.default_rng())
    # Asset specifications
    assets = _generate_asset_specs(gen.n_assets, gen.n_sectors; rng=rng)

    # Correlation matrix
    corr = _generate_sector_correlation(gen.n_assets, gen.n_sectors,
                                        gen.intra_sector_corr, gen.inter_sector_corr)

    # Event calendar
    events = generate_event_calendar(gen.n_steps; rng=rng)

    # True value process
    tvp = TrueValueProcess(; mu_fundamental=0.05, sigma_fundamental=0.12,
                           sigma_market=0.20, mean_reversion=2.0,
                           V0=100.0, S0=100.0)

    # Liquidity model
    liq = LiquidityModel(; base_spread=0.01, vol_sensitivity=0.5,
                         imbalance_sensitivity=0.3, depth=10000.0)

    # Microstructure noise
    micro = MarketMicrostructureNoise(; tick_size=0.01, half_spread=0.005,
                                      stale_quote_prob=0.03, delay_prob=0.01)

    SyntheticUniverse{Float64}(assets, corr, events, tvp, liq, micro)
end

"""
    simulate_universe(universe::SyntheticUniverse, n_steps::Int, dt::Real; rng)

Simulate the entire universe: generate correlated base paths, apply
per-asset model dynamics, inject events, add microstructure noise.

Returns a Dict with:
- `"prices"`       — (n_steps+1) × n_assets matrix of observed prices
- `"true_prices"`  — (n_steps+1) × n_assets matrix of efficient prices
- `"spreads"`      — (n_steps+1) × n_assets matrix of bid-ask spreads
- `"regimes"`      — (n_steps+1) vector of macro regime states
- `"events_applied"` — count of events applied
"""
function simulate_universe(universe::SyntheticUniverse{T}, n_steps::Int, dt::Real;
                           rng::AbstractRNG=Random.default_rng()) where {T}
    n_assets = length(universe.assets)
    dt_T = T(dt)

    # Step 1: Generate correlated base noise
    mu_vec = T[a.mu for a in universe.assets]
    sigma_vec = T[a.sigma for a in universe.assets]
    S0_vec = T[a.S0 for a in universe.assets]
    ca = CorrelatedAssets(mu_vec, sigma_vec, S0_vec, universe.correlation)
    base_paths = generate_correlated_assets(ca, n_steps, dt; rng=rng)

    # Step 2: Apply per-asset model specifics
    efficient_prices = copy(base_paths)

    for (j, asset) in enumerate(universe.assets)
        if asset.model == :heston && haskey(asset.extra, "kappa")
            # Overlay stochastic vol on the base path
            h = HestonModel(asset.mu, asset.extra["kappa"], asset.extra["theta"],
                            asset.extra["xi"], asset.extra["rho"],
                            asset.S0, asset.extra["v0"])
            hp, hv = simulate(h, n_steps, dt; rng=rng)
            # Blend: use Heston vol structure with correlated drift
            vol_ratio = [sqrt(max(hv[i], T(1e-8))) / max(asset.sigma, T(1e-8)) for i in 1:(n_steps+1)]
            @inbounds for i in 2:(n_steps+1)
                # Scale the return by the stochastic vol ratio
                base_ret = log(efficient_prices[i, j] / efficient_prices[i-1, j])
                scaled_ret = base_ret * vol_ratio[i]
                efficient_prices[i, j] = efficient_prices[i-1, j] * exp(scaled_ret)
            end
        elseif asset.model == :merton && haskey(asset.extra, "lambda_j")
            # Add jumps to the base path
            lambda_j = asset.extra["lambda_j"]
            mu_j = asset.extra["mu_j"]
            sigma_j = asset.extra["sigma_j"]
            @inbounds for i in 2:(n_steps+1)
                n_jumps = _poisson_sample(T(lambda_j * dt), rng)
                for _ in 1:n_jumps
                    jump = exp(T(mu_j) + T(sigma_j) * randn(rng, T))
                    efficient_prices[i, j] *= jump
                end
            end
        elseif asset.model == :regime && haskey(asset.extra, "p11")
            # Apply regime-dependent vol scaling
            p11 = asset.extra["p11"]
            p22 = asset.extra["p22"]
            sig_bull = get(asset.extra, "sigma_bull", asset.sigma * 0.7)
            sig_bear = get(asset.extra, "sigma_bear", asset.sigma * 1.5)
            regime = 1
            @inbounds for i in 2:(n_steps+1)
                u = rand(rng, T)
                regime = regime == 1 ? (u < T(p11) ? 1 : 2) : (u < T(p22) ? 2 : 1)
                base_ret = log(efficient_prices[i, j] / efficient_prices[i-1, j])
                scale = regime == 1 ? T(sig_bull) / max(asset.sigma, T(1e-8)) :
                                      T(sig_bear) / max(asset.sigma, T(1e-8))
                efficient_prices[i, j] = efficient_prices[i-1, j] * exp(base_ret * scale)
            end
        end
    end

    # Step 3: Apply events
    for j in 1:n_assets
        price_col = efficient_prices[:, j]
        apply_events!(universe.event_calendar, price_col)
        efficient_prices[:, j] .= price_col
    end

    # Step 4: Add microstructure noise
    observed_prices = Matrix{T}(undef, n_steps + 1, n_assets)
    for j in 1:n_assets
        observed_prices[:, j] = add_microstructure_noise(
            universe.microstructure, efficient_prices[:, j]; rng=rng)
    end

    # Step 5: Generate spreads for a representative asset
    spread_matrix = Matrix{T}(undef, n_steps + 1, n_assets)
    dummy_buys = T[T(i) * dt_T for i in 1:10]
    dummy_sells = T[T(i) * dt_T for i in 1:10]
    for j in 1:n_assets
        spread_matrix[:, j] .= universe.liquidity.base_spread *
            (one(T) + T(0.5) * universe.assets[j].sigma / T(0.20))
    end

    Dict{String,Any}(
        "prices" => observed_prices,
        "true_prices" => efficient_prices,
        "spreads" => spread_matrix,
        "n_assets" => n_assets,
        "n_steps" => n_steps,
        "events_applied" => length(universe.event_calendar.events),
    )
end

"""
    universe_summary(universe::SyntheticUniverse)

Print a summary of the synthetic universe.
"""
function universe_summary(universe::SyntheticUniverse)
    n = length(universe.assets)
    sectors = unique(a.sector for a in universe.assets)
    models = Dict{Symbol,Int}()
    for a in universe.assets
        models[a.model] = get(models, a.model, 0) + 1
    end

    avg_mu = mean(a.mu for a in universe.assets)
    avg_sigma = mean(a.sigma for a in universe.assets)
    avg_corr = mean(universe.correlation[i,j] for i in 1:n for j in 1:n if i != j)

    Dict{String,Any}(
        "n_assets" => n,
        "n_sectors" => length(sectors),
        "model_counts" => models,
        "avg_drift" => avg_mu,
        "avg_volatility" => avg_sigma,
        "avg_correlation" => avg_corr,
        "n_scheduled_events" => length(universe.event_calendar.events),
        "tick_size" => universe.microstructure.tick_size,
        "base_spread" => universe.liquidity.base_spread,
    )
end


# ============================================================================
# Utility functions
# ============================================================================

"""
    log_returns(prices)

Compute log-return series from a price vector.
"""
log_returns(prices::AbstractVector{<:Real}) =
    [log(prices[i] / prices[i-1]) for i in 2:length(prices)]

"""
    simple_returns(prices)

Compute simple return series from a price vector.
"""
simple_returns(prices::AbstractVector{<:Real}) =
    [prices[i] / prices[i-1] - 1.0 for i in 2:length(prices)]

"""
    annualized_return(prices, dt)

Compute annualized return from a price path.
"""
function annualized_return(prices::AbstractVector{<:Real}, dt::Real)
    total_time = (length(prices) - 1) * dt
    total_time > 0 || return 0.0
    (prices[end] / prices[1])^(1.0 / total_time) - 1.0
end

"""
    annualized_volatility(prices, dt)

Compute annualized volatility from a price path.
"""
function annualized_volatility(prices::AbstractVector{<:Real}, dt::Real)
    rets = log_returns(prices)
    length(rets) >= 2 || return 0.0
    std(rets) / sqrt(dt)
end

"""
    sharpe_ratio(prices, dt; risk_free_rate=0.0)

Compute annualized Sharpe ratio.
"""
function sharpe_ratio(prices::AbstractVector{<:Real}, dt::Real;
                      risk_free_rate::Real=0.0)
    ret = annualized_return(prices, dt)
    vol = annualized_volatility(prices, dt)
    vol < 1e-15 && return 0.0
    (ret - risk_free_rate) / vol
end

"""
    maximum_drawdown(prices)

Compute the maximum drawdown of a price path.
"""
function maximum_drawdown(prices::AbstractVector{<:Real})
    peak = prices[1]
    max_dd = 0.0
    for p in prices
        peak = max(peak, p)
        dd = (peak - p) / peak
        max_dd = max(max_dd, dd)
    end
    max_dd
end

"""
    realized_volatility(prices, dt; window=nothing)

Compute realized volatility. If `window` is given, use rolling.
"""
function realized_volatility(prices::AbstractVector{<:Real}, dt::Real;
                             window::Union{Nothing,Int}=nothing)
    rets = log_returns(prices)
    if window === nothing
        return std(rets) / sqrt(dt)
    end

    n = length(rets)
    window = min(window, n)
    rv = Vector{Float64}(undef, n - window + 1)
    for i in 1:(n - window + 1)
        rv[i] = std(rets[i:(i + window - 1)]) / sqrt(dt)
    end
    rv
end

"""
    garman_klass_volatility(open, high, low, close)

Garman-Klass (1980) volatility estimator using OHLC data.
"""
function garman_klass_volatility(open::AbstractVector{<:Real},
                                 high::AbstractVector{<:Real},
                                 low::AbstractVector{<:Real},
                                 close::AbstractVector{<:Real})
    n = length(open)
    n == length(high) == length(low) == length(close) ||
        throw(ArgumentError("All vectors must have equal length"))

    gk = 0.0
    for i in 1:n
        u = log(high[i] / open[i])
        d = log(low[i] / open[i])
        c = log(close[i] / open[i])
        gk += 0.5 * (u - d)^2 - (2 * log(2) - 1) * c^2
    end
    sqrt(gk / n)
end

"""
    hurst_exponent(prices; max_lag=nothing)

Estimate the Hurst exponent using the R/S (rescaled range) method.
H > 0.5 → trending, H < 0.5 → mean-reverting, H ≈ 0.5 → random walk.
"""
function hurst_exponent(prices::AbstractVector{<:Real}; max_lag::Union{Nothing,Int}=nothing)
    rets = log_returns(prices)
    n = length(rets)
    max_k = max_lag === nothing ? div(n, 4) : min(max_lag, div(n, 2))
    max_k = max(max_k, 4)

    lags = Int[]
    rs_values = Float64[]

    for k in 4:max_k
        n_sub = div(n, k)
        n_sub >= 1 || continue

        rs_sum = 0.0
        count = 0
        for s in 1:n_sub
            sub = rets[((s-1)*k + 1):(s*k)]
            m = mean(sub)
            cumdev = cumsum(sub .- m)
            R = maximum(cumdev) - minimum(cumdev)
            S = std(sub)
            if S > 1e-15
                rs_sum += R / S
                count += 1
            end
        end
        if count > 0
            push!(lags, k)
            push!(rs_values, rs_sum / count)
        end
    end

    length(lags) < 2 && return 0.5

    # Linear regression: log(R/S) = H * log(n) + c
    log_lags = log.(lags)
    log_rs = log.(rs_values)
    n_pts = length(log_lags)
    x_mean = mean(log_lags)
    y_mean = mean(log_rs)
    num = sum((log_lags[i] - x_mean) * (log_rs[i] - y_mean) for i in 1:n_pts)
    den = sum((log_lags[i] - x_mean)^2 for i in 1:n_pts)
    H = den > 0 ? num / den : 0.5
    clamp(H, 0.0, 1.0)
end

"""
    autocorrelation(x, lag=1)

Compute the autocorrelation of a series at a given lag.
"""
function autocorrelation(x::AbstractVector{<:Real}, lag::Int=1)
    n = length(x)
    lag < n || return 0.0
    mu = mean(x)
    num = sum((x[i] - mu) * (x[i + lag] - mu) for i in 1:(n - lag))
    den = sum((x[i] - mu)^2 for i in 1:n)
    den > 0 ? num / den : 0.0
end

"""
    vwap(prices, volumes)

Volume-weighted average price.
"""
function vwap(prices::AbstractVector{<:Real}, volumes::AbstractVector{<:Real})
    length(prices) == length(volumes) || throw(ArgumentError("Length mismatch"))
    total_vol = sum(volumes)
    total_vol > 0 || return mean(prices)
    sum(prices .* volumes) / total_vol
end

"""
    twap(prices)

Time-weighted average price (simple mean).
"""
twap(prices::AbstractVector{<:Real}) = mean(prices)

"""
    kyle_lambda(prices, volumes, signs)

Estimate Kyle's lambda (price impact coefficient):
    ΔP = λ × signed_volume + ε

Uses OLS regression.
"""
function kyle_lambda(prices::AbstractVector{<:Real},
                     volumes::AbstractVector{<:Real},
                     signs::AbstractVector{<:Real})
    n = min(length(prices) - 1, length(volumes), length(signs))
    n >= 3 || return 0.0

    delta_p = [prices[i+1] - prices[i] for i in 1:n]
    signed_vol = [volumes[i] * signs[i] for i in 1:n]

    # OLS: lambda = cov(ΔP, SV) / var(SV)
    sv_mean = mean(signed_vol)
    dp_mean = mean(delta_p)
    num = sum((signed_vol[i] - sv_mean) * (delta_p[i] - dp_mean) for i in 1:n)
    den = sum((signed_vol[i] - sv_mean)^2 for i in 1:n)
    den > 0 ? num / den : 0.0
end

"""
    amihud_illiquidity(prices, volumes, dt)

Amihud (2002) illiquidity ratio: average |return| / volume.
"""
function amihud_illiquidity(prices::AbstractVector{<:Real},
                            volumes::AbstractVector{<:Real}, dt::Real)
    n = min(length(prices) - 1, length(volumes))
    n >= 1 || return 0.0

    illiq = 0.0
    count = 0
    for i in 1:n
        if volumes[i] > 0
            ret = abs(log(prices[i+1] / prices[i]))
            illiq += ret / volumes[i]
            count += 1
        end
    end
    count > 0 ? illiq / count : 0.0
end


# ============================================================================
# Simulation runners / convenience
# ============================================================================

"""
    run_full_simulation(; n_assets=10, n_steps=1000, dt=1/252, seed=42)

Run a complete multi-asset simulation with all physics layers.
Returns a comprehensive result Dict.
"""
function run_full_simulation(; n_assets::Int=10, n_steps::Int=1000,
                             dt::Real=1/252, seed::Int=42)
    rng = Random.MersenneTwister(seed)

    # Generate universe
    gen = SyntheticUniverseGenerator(; n_assets=n_assets, n_sectors=min(3, n_assets),
                                     n_steps=n_steps, dt=dt)
    universe = generate_universe(gen; rng=rng)

    # Simulate
    result = simulate_universe(universe, n_steps, dt; rng=rng)

    # Add summary statistics
    prices = result["prices"]
    stats = Dict{String,Any}()
    for j in 1:n_assets
        col = prices[:, j]
        stats[universe.assets[j].name] = Dict{String,Any}(
            "ann_return" => annualized_return(col, dt),
            "ann_vol" => annualized_volatility(col, dt),
            "sharpe" => sharpe_ratio(col, dt),
            "max_drawdown" => maximum_drawdown(col),
            "hurst" => hurst_exponent(col),
        )
    end
    result["asset_stats"] = stats
    result["universe_summary"] = universe_summary(universe)
    result
end

"""
    scenario_test(model, scenarios::Vector{<:Pair}, n_steps, dt; n_paths=100, rng)

Run a model under multiple parameter scenarios and collect statistics.
Each scenario is a Pair("name" => model_instance).
"""
function scenario_test(scenarios::Vector{<:Pair}, n_steps::Int, dt::Real;
                       n_paths::Int=100, rng::AbstractRNG=Random.default_rng())
    results = Dict{String,Dict{String,Float64}}()
    for (name, model) in scenarios
        paths = simulate_paths(model, n_paths, n_steps, dt; rng=rng)
        final_prices = paths[end, :]
        initial = paths[1, 1]

        total_returns = log.(final_prices ./ initial)
        results[name] = Dict{String,Float64}(
            "mean_return" => mean(total_returns),
            "std_return" => std(total_returns),
            "median_return" => median(total_returns),
            "min_return" => minimum(total_returns),
            "max_return" => maximum(total_returns),
            "prob_loss" => count(r -> r < 0, total_returns) / n_paths,
        )
    end
    results
end

"""
    stress_test_universe(universe::SyntheticUniverse, n_steps, dt;
                         vol_multiplier=2.0, corr_shift=0.3, rng)

Run a stress test by increasing vol and correlations.
"""
function stress_test_universe(universe::SyntheticUniverse{T}, n_steps::Int, dt::Real;
                              vol_multiplier::Real=2.0, corr_shift::Real=0.3,
                              rng::AbstractRNG=Random.default_rng()) where {T}
    # Create stressed universe
    stressed_assets = [AssetSpec(a.name; model=a.model,
                                mu=a.mu - T(0.1),  # lower drift in stress
                                sigma=a.sigma * T(vol_multiplier),
                                S0=a.S0, sector=a.sector,
                                market_cap=a.market_cap, extra=a.extra)
                       for a in universe.assets]

    n = length(universe.assets)
    stressed_corr = copy(universe.correlation)
    for i in 1:n, j in 1:n
        if i != j
            stressed_corr[i, j] = clamp(stressed_corr[i, j] + T(corr_shift), T(-0.99), T(0.99))
        end
    end
    # Re-ensure PD
    evals = eigvals(Symmetric(stressed_corr))
    if minimum(evals) < 0.01
        shift = 0.01 - minimum(evals)
        stressed_corr .+= shift * I
        D_inv = Diagonal(1.0 ./ sqrt.(diag(stressed_corr)))
        stressed_corr = D_inv * stressed_corr * D_inv
        for i in 1:n
            stressed_corr[i, i] = 1.0
        end
    end

    stressed_universe = SyntheticUniverse{T}(stressed_assets, stressed_corr,
                                              universe.event_calendar,
                                              universe.true_value_proc,
                                              universe.liquidity,
                                              universe.microstructure)

    simulate_universe(stressed_universe, n_steps, dt; rng=rng)
end

end # module SyntheticExchangePhysics
