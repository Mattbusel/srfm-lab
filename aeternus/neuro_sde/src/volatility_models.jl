"""
volatility_models.jl — Stochastic volatility models with neural corrections

Models implemented:
  1. NeuralHeston   — Heston backbone + neural drift/vol-of-vol corrections
  2. NeuralSABR     — SABR with neural vol-of-vol correction
  3. RoughVol       — Rough volatility (H < 0.5) via fractional BM approximation
  4. JumpDiffusion  — Neural SDE + compound Poisson jumps (Merton backbone)
  5. RegimeSwitching— CTMC regime indicator + neural emissions per regime

Each model implements:
  - struct definition + constructor
  - simulate(model, T, n_paths, dt) → paths
  - log_likelihood(model, data) → scalar

References:
  - Heston (1993), SABR (Hagan et al. 2002)
  - El Euch & Rosenbaum (2019) rough Heston
  - Merton (1976) jump-diffusion
"""

using LinearAlgebra
using Statistics
using Random
using Distributions
using Flux

# ─────────────────────────────────────────────────────────────────────────────
# 1. NEURAL HESTON MODEL
# ─────────────────────────────────────────────────────────────────────────────

"""
    NeuralHeston

Heston stochastic volatility model with neural corrections:

  dS = μ·S·dt + √V·S·dW₁
  dV = κ(θ-V)dt + ξ·√V·dW₂ + f_nn(S,V,t)dt + g_nn(S,V,t)dW₂
  ⟨dW₁, dW₂⟩ = ρ·dt

where f_nn and g_nn are neural networks providing additive corrections to
the classical Heston drift and diffusion respectively.

State vector: (log S, V) — log price and variance.

Fields:
  - `κ`, `θ`, `ξ`, `ρ`, `μ` : classical Heston parameters
  - `drift_correction`  : DriftNet for correction to drift
  - `vol_correction`    : DiffusionNet for vol-of-vol correction
"""
struct NeuralHeston
    κ  :: Float32    # mean reversion speed
    θ  :: Float32    # long-run variance
    ξ  :: Float32    # vol of vol
    ρ  :: Float32    # correlation
    μ  :: Float32    # drift (risk-free rate - dividend)
    drift_correction :: Union{DriftNet, Nothing}
    vol_correction   :: Union{DiffusionNet, Nothing}
end

"""
    NeuralHeston(; κ, θ, ξ, ρ, μ, state_dim=2, use_corrections=true,
                  hidden_dim=32, n_layers=2)
"""
function NeuralHeston(; κ::Float32  = 2.0f0,
                        θ::Float32  = 0.04f0,
                        ξ::Float32  = 0.3f0,
                        ρ::Float32  = -0.7f0,
                        μ::Float32  = 0.05f0,
                        use_corrections::Bool = true,
                        hidden_dim::Int       = 32,
                        n_layers::Int         = 2)

    drift_corr = use_corrections ?
        build_drift_net(2; hidden_dim=hidden_dim, n_layers=n_layers,
                         time_emb_dim=8, use_batchnorm=false) : nothing
    vol_corr   = use_corrections ?
        build_diffusion_net(2; hidden_dim=hidden_dim, n_layers=n_layers,
                             time_emb_dim=8, diagonal=true) : nothing

    return NeuralHeston(κ, θ, ξ, ρ, μ, drift_corr, vol_corr)
end

Flux.@functor NeuralHeston (drift_correction, vol_correction)

"""
    heston_drift(m::NeuralHeston, state, t) → drift vector

state = [log_S, V]. Returns d(log_S)/dt and dV/dt.
"""
function heston_drift(m::NeuralHeston, state::AbstractVector, t::Real)
    log_S, V = state[1], state[2]
    V_pos = max(V, 0f0)  # Feller condition safety

    # Classical Heston drift (Itô form for log S)
    d_logS = Float32(m.μ) - 0.5f0 * V_pos
    d_V    = Float32(m.κ) * (Float32(m.θ) - V_pos)

    drift = Float32[d_logS, d_V]

    if m.drift_correction !== nothing
        corr = m.drift_correction(state, Float32(t))
        drift = drift .+ 0.1f0 .* corr  # scale corrections to avoid dominance
    end

    return drift
end

"""
    heston_diffusion(m::NeuralHeston, state, t) → (σ₁, σ₂, ρ)

Returns the Cholesky factored diffusion coefficients for the correlated 2D BM.
We use the decomposition:
  dW₁ = dB₁
  dW₂ = ρ·dB₁ + √(1-ρ²)·dB₂

Diffusion matrix (in terms of independent BMs B₁, B₂):
  [[σ₁,    0         ],
   [σ₂ρ,  σ₂√(1-ρ²)]]

where σ₁ = √V and σ₂ = ξ√V.
"""
function heston_diffusion(m::NeuralHeston, state::AbstractVector, t::Real)
    log_S, V = state[1], state[2]
    V_pos  = max(V, 1f-6)
    sqrtV  = sqrt(V_pos)
    ρ      = Float32(m.ρ)
    ξ      = Float32(m.ξ)

    σ₁     = sqrtV                    # diffusion of log S
    σ₂_row1 = ξ * sqrtV * ρ           # correlated part for V
    σ₂_row2 = ξ * sqrtV * sqrt(max(1f0 - ρ^2, 1f-6))  # independent part

    # Returns lower Cholesky factor L: [[σ₁, 0], [σ₂ρ, σ₂√(1-ρ²)]]
    L = Float32[σ₁  0f0; σ₂_row1  σ₂_row2]

    if m.vol_correction !== nothing
        vol_corr = m.vol_correction(state, Float32(t))
        # Add diagonal correction (must remain positive)
        L[1,1] = max(L[1,1] + 0.05f0 * vol_corr[1], 1f-4)
        L[2,2] = max(L[2,2] + 0.05f0 * vol_corr[2], 1f-4)
    end

    return L  # (2×2) lower-triangular
end

"""
    simulate(m::NeuralHeston, S0, V0, T, n_paths, dt; rng)

Simulate n_paths of the NeuralHeston model.
Returns (log_S_paths, V_paths) each of shape (n_timesteps, n_paths).
"""
function simulate_model(m::NeuralHeston, S0::Float64, V0::Float64,
                          T::Float64, n_paths::Int, dt::Float64;
                          rng = Random.GLOBAL_RNG)

    n_steps = ceil(Int, T / dt)
    log_S_paths = zeros(Float32, n_steps+1, n_paths)
    V_paths     = zeros(Float32, n_steps+1, n_paths)

    log_S_paths[1, :] .= Float32(log(S0))
    V_paths[1, :]     .= Float32(V0)

    for p in 1:n_paths
        log_S = Float32(log(S0))
        V     = Float32(V0)

        for k in 1:n_steps
            t      = Float32((k-1) * dt)
            dt_k   = Float32(min(dt, T - (k-1)*dt))
            state  = Float32[log_S, V]

            # Draw two independent standard normals
            Z1, Z2 = Float32(randn(rng)), Float32(randn(rng))

            # Drift
            μ_vec = heston_drift(m, state, t)

            # Diffusion: L·[Z1, Z2]
            L     = heston_diffusion(m, state, t)
            dW1   = sqrt(dt_k) * Z1
            dW2   = sqrt(dt_k) * Z2
            noise = L * Float32[dW1, dW2]  # correlated increments

            # Milstein-style update with full reflection for V
            log_S += μ_vec[1] * dt_k + noise[1]
            V     += μ_vec[2] * dt_k + noise[2]
            V      = abs(V)  # full reflection to enforce V ≥ 0

            log_S_paths[k+1, p] = log_S
            V_paths[k+1, p]     = V
        end
    end

    return log_S_paths, V_paths
end

"""
    log_likelihood_model(m::NeuralHeston, log_returns, V_obs; dt)

Approximate log-likelihood of observed log returns under NeuralHeston.
Uses the Euler-Maruyama transition density (Gaussian approximation).
"""
function log_likelihood_model(m::NeuralHeston, log_returns::AbstractVector,
                                V_obs::AbstractVector; dt::Float64=1.0/252)

    n   = length(log_returns)
    ll  = 0.0f0
    V   = Float32(V_obs[1])

    for k in 1:n
        t     = Float32(k * dt)
        logS  = sum(log_returns[1:k])  # running sum
        state = Float32[logS, V]

        d_vec = heston_drift(m, state, t)
        L     = heston_diffusion(m, state, t)

        # Transition mean and variance for log S
        μ_r    = d_vec[1] * Float32(dt)
        σ²_r   = (L[1,1]^2) * Float32(dt)

        # Gaussian log-likelihood for return k
        r_k    = Float32(log_returns[k])
        ll    += -0.5f0 * (r_k - μ_r)^2 / σ²_r - 0.5f0 * log(2π * σ²_r)

        # Update variance state
        if k < length(V_obs)
            V = Float32(V_obs[k+1])
        else
            d_V   = d_vec[2] * Float32(dt)
            dW_V  = Float32(sqrt(dt)) * Float32(randn())
            V    += d_V + L[2,1] * dW_V + L[2,2] * Float32(randn()) * Float32(sqrt(dt))
            V     = abs(V)
        end
    end

    return ll
end

"""
    heston_characteristic_fn(m::NeuralHeston, u, T) → Complex

Characteristic function of log(S_T/S_0) under the classical Heston model
(ignoring neural corrections, used for Carr-Madan FFT calibration).

Closed-form from Heston (1993):
  φ(u) = exp(iuμT + κθ/ξ² · [(κ-ρξiu-d)T - 2log((1-ge^{-dT})/(1-g))]
           + V₀/ξ² · (κ-ρξiu-d)(1-e^{-dT})/(1-ge^{-dT}))

where d = √((ρξiu - κ)² + ξ²(iu + u²)), g = (κ-ρξiu-d)/(κ-ρξiu+d).
"""
function heston_characteristic_fn(m::NeuralHeston, u::Complex, T::Float64,
                                    V0::Float64=0.04)
    κ, θ, ξ, ρ, μ = Float64(m.κ), Float64(m.θ), Float64(m.ξ), Float64(m.ρ), Float64(m.μ)

    iu  = im * u
    d   = sqrt((ρ * ξ * iu - κ)^2 + ξ^2 * (iu + u^2))
    g   = (κ - ρ * ξ * iu - d) / (κ - ρ * ξ * iu + d)

    e_dT = exp(-d * T)
    A  = iu * μ * T
    B  = κ * θ / ξ^2 * ((κ - ρ*ξ*iu - d)*T - 2*log((1 - g*e_dT)/(1 - g)))
    C  = V0 / ξ^2 * (κ - ρ*ξ*iu - d) * (1 - e_dT) / (1 - g*e_dT)

    return exp(A + B + C)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. NEURAL SABR MODEL
# ─────────────────────────────────────────────────────────────────────────────

"""
    NeuralSABR

SABR model with neural vol-of-vol correction:

  dF = σ·Fᵝ·dW₁                    (forward price dynamics)
  dσ = α·σ·dW₂                      (classical SABR vol dynamics)
       + η_nn(F,σ,t)·σ·dt            (neural drift correction)
  ⟨dW₁,dW₂⟩ = ρ·dt

State: [log F, log σ].
"""
struct NeuralSABR
    α   :: Float32   # vol-of-vol
    β   :: Float32   # CEV exponent
    ρ   :: Float32   # correlation
    F0  :: Float32   # initial forward
    σ0  :: Float32   # initial vol
    vol_correction :: Union{DriftNet, Nothing}
end

function NeuralSABR(; α::Float32=0.3f0, β::Float32=0.5f0, ρ::Float32=-0.3f0,
                      F0::Float32=100.0f0, σ0::Float32=0.2f0,
                      use_correction::Bool=true,
                      hidden_dim::Int=32)

    corr = use_correction ?
        build_drift_net(2; hidden_dim=hidden_dim, n_layers=2,
                         time_emb_dim=8, use_batchnorm=false) : nothing
    return NeuralSABR(α, β, ρ, F0, σ0, corr)
end

Flux.@functor NeuralSABR (vol_correction,)

function simulate_model(m::NeuralSABR, T::Float64, n_paths::Int, dt::Float64;
                          rng = Random.GLOBAL_RNG)
    n_steps = ceil(Int, T / dt)
    F_paths = zeros(Float32, n_steps+1, n_paths)
    σ_paths = zeros(Float32, n_steps+1, n_paths)

    F_paths[1, :] .= m.F0
    σ_paths[1, :] .= m.σ0

    for p in 1:n_paths
        F = m.F0
        σ = m.σ0

        for k in 1:n_steps
            t    = Float32((k-1) * dt)
            dt_k = Float32(min(dt, T - (k-1)*dt))
            sqdt = sqrt(dt_k)

            Z1, Z2 = Float32(randn(rng)), Float32(randn(rng))
            dW1 = sqdt * Z1
            dW2 = sqdt * (Float32(m.ρ) * Z1 + sqrt(max(1f0 - m.ρ^2, 0f0)) * Z2)

            F_β = abs(F)^Float32(m.β)
            dF  = σ * F_β * dW1

            # SABR vol dynamics
            state = Float32[log(max(F, 1f-8)), log(max(σ, 1f-8))]
            dσ_classical = Float32(m.α) * σ * dW2
            dσ_neural    = 0f0
            if m.vol_correction !== nothing
                corr = m.vol_correction(state, t)
                dσ_neural = 0.05f0 * corr[2] * σ * dt_k
            end

            F = max(F + dF,       1f-6)
            σ = max(σ + dσ_classical + dσ_neural, 1f-6)

            F_paths[k+1, p] = F
            σ_paths[k+1, p] = σ
        end
    end

    return F_paths, σ_paths
end

"""
    sabr_implied_vol(m::NeuralSABR, F, K, T) → σ_BSM

Hagan et al. (2002) approximation for SABR implied volatility.
"""
function sabr_implied_vol(m::NeuralSABR, F::Float64, K::Float64, T::Float64)
    α, β, ρ, ν = Float64(m.α), Float64(m.β), Float64(m.ρ), Float64(m.α)
    σ0 = Float64(m.σ0)

    FK_mid = sqrt(F * K)
    log_FK = log(F / K)

    if abs(log_FK) < 1e-10  # ATM case
        FK_β = FK_mid^(1-β)
        σ_atm = σ0 / FK_β * (
            1 + ((1-β)^2/24 * σ0^2/FK_β^2 +
                  ρ*β*ν*σ0/(4*FK_β) +
                  ν^2*(2-3ρ^2)/24) * T
        )
        return σ_atm
    end

    # General case
    z   = ν/σ0 * FK_mid^(1-β) * log_FK
    χ_z = log((sqrt(1 - 2ρ*z + z^2) + z - ρ) / (1 - ρ))

    num = σ0
    den = FK_mid^(1-β) * (1 + (1-β)^2/24 * log_FK^2 + (1-β)^4/1920 * log_FK^4)

    correction = (1 + ((1-β)^2/24 * σ0^2/FK_mid^(2*(1-β)) +
                        ρ*β*ν*σ0/(4*FK_mid^(1-β)) +
                        ν^2*(2-3ρ^2)/24) * T)

    return num / den * z / χ_z * correction
end

function log_likelihood_model(m::NeuralSABR, log_returns::AbstractVector;
                                dt::Float64=1.0/252)
    n  = length(log_returns)
    ll = 0.0f0
    F  = m.F0
    σ  = m.σ0

    for k in 1:n
        F_β    = abs(Float32(F))^m.β
        σ_eff  = σ * F_β
        μ_r    = 0f0  # SABR has zero drift
        σ²_r   = σ_eff^2 * Float32(dt)

        r_k    = Float32(log_returns[k])
        ll    += -0.5f0 * (r_k - μ_r)^2 / σ²_r - 0.5f0 * log(2π * σ²_r)

        # Propagate state
        dσ = Float32(m.α) * σ * Float32(sqrt(dt)) * Float32(randn())
        F  = max(F  + σ * F_β * Float32(sqrt(dt)) * Float32(randn()), 1f-6)
        σ  = max(σ  + dσ, 1f-6)
    end

    return ll
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. ROUGH VOLATILITY MODEL
# ─────────────────────────────────────────────────────────────────────────────

"""
    RoughVol

Rough volatility model using fractional Brownian motion (fBM) with Hurst
parameter H ∈ (0, 1/2). For H < 1/2 the vol process is rougher than
standard BM, matching empirical findings (Gatheral et al. 2018).

We approximate fBM via the Cholesky decomposition of its covariance matrix:
  Cov[B^H(s), B^H(t)] = (1/2)(|s|^{2H} + |t|^{2H} - |s-t|^{2H})

Price dynamics:
  d log S = -V²/2 dt + V dW
  V(t) = σ₀·exp(η·∫₀ᵗ (t-s)^{H-1/2} dW^H(s) - η²/2·t^{2H}/(2H))
"""
struct RoughVol
    H      :: Float32   # Hurst exponent (< 0.5 for rough)
    η      :: Float32   # vol-of-vol
    σ0     :: Float32   # initial vol
    ρ      :: Float32   # correlation between W and W^H
    μ      :: Float32   # drift
end

function RoughVol(; H::Float32=0.1f0, η::Float32=0.3f0, σ0::Float32=0.2f0,
                    ρ::Float32=-0.7f0, μ::Float32=0.0f0)
    0 < H < 0.5 || @warn "RoughVol: H=$H should be in (0, 0.5) for rough vol"
    return RoughVol(H, η, σ0, ρ, μ)
end

"""
    build_fbm_covariance(T, n_steps, H) → C

Build the n_steps × n_steps covariance matrix of the fBM increments.
Uses the standard fBM covariance kernel.
"""
function build_fbm_covariance(T::Float64, n_steps::Int, H::Float64)
    dt  = T / n_steps
    t   = [k * dt for k in 1:n_steps]
    C   = zeros(n_steps, n_steps)
    for i in 1:n_steps, j in 1:n_steps
        s, t_j = t[i], t[j]
        C[i,j] = 0.5 * (abs(s)^(2H) + abs(t_j)^(2H) - abs(s-t_j)^(2H))
    end
    return C
end

"""
    fbm_cholesky_sim(H, T, n_steps, n_paths; rng)

Simulate n_paths of fractional BM paths using Cholesky decomposition.
Returns (n_steps × n_paths) matrix of fBM values.
"""
function fbm_cholesky_sim(H::Float64, T::Float64, n_steps::Int, n_paths::Int;
                           rng = Random.GLOBAL_RNG)
    C    = build_fbm_covariance(T, n_steps, H)
    # Regularise for numerical stability
    C   .+= 1e-8 * I
    L    = cholesky(Symmetric(C)).L    # lower-triangular Cholesky

    Z    = randn(rng, n_steps, n_paths)
    BH   = L * Z    # (n_steps × n_paths) fBM values

    return BH
end

"""
    simulate_model(m::RoughVol, S0, T, n_paths, dt; rng)

Simulate rough volatility model paths.
"""
function simulate_model(m::RoughVol, S0::Float64, T::Float64,
                          n_paths::Int, dt::Float64;
                          rng = Random.GLOBAL_RNG)

    n_steps = ceil(Int, T / dt)
    T_actual = n_steps * dt

    H_f  = Float64(m.H)
    η_f  = Float64(m.η)
    ρ_f  = Float64(m.ρ)
    σ0_f = Float64(m.σ0)
    μ_f  = Float64(m.μ)

    # Simulate fBM paths
    BH = fbm_cholesky_sim(H_f, T_actual, n_steps, n_paths; rng=rng)

    # Correlated standard BM
    Z_W  = randn(rng, n_steps, n_paths)

    # Volatility path: V(t) = σ₀·exp(η·B^H(t) - η²/2·t^{2H}/(2H))
    t_vec = [k * dt for k in 1:n_steps]
    norm_factor = [η_f^2 / 2 * t^(2H_f) / (2H_f) for t in t_vec]
    V_paths = Float32.(σ0_f .* exp.(η_f .* BH .- norm_factor))

    # Price paths via Euler-Maruyama
    S_paths = zeros(Float32, n_steps+1, n_paths)
    S_paths[1, :] .= Float32(S0)

    for k in 1:n_steps
        dt_k   = Float32(dt)
        S_prev = S_paths[k, :]
        V_k    = V_paths[k, :]

        # Correlated BM: dW = ρ·dW^H_increment + √(1-ρ²)·Z_W
        dWH_incr = k == 1 ? BH[1, :] : BH[k, :] .- BH[k-1, :]
        dW       = Float32.(ρ_f .* dWH_incr ./ sqrt(dt) .* sqrt(dt) .+
                             sqrt(max(1.0 - ρ_f^2, 0.0)) .* Z_W[k, :] .* sqrt(dt))

        log_S_prev = log.(max.(S_prev, Float32(1e-8)))
        log_S_new  = log_S_prev .+ Float32(μ_f) .* dt_k .-
                     0.5f0 .* V_k.^2 .* dt_k .+ V_k .* dW
        S_paths[k+1, :] = exp.(log_S_new)
    end

    return S_paths, V_paths
end

function log_likelihood_model(m::RoughVol, log_returns::AbstractVector;
                                dt::Float64=1.0/252)
    n   = length(log_returns)
    ll  = 0.0
    H_f = Float64(m.H)
    η_f = Float64(m.η)
    σ0  = Float64(m.σ0)
    μ_f = Float64(m.μ)

    # Use σ0 as approximate vol (full fBM-aware LL is expensive)
    for k in 1:n
        # Approximate vol with σ₀ decay
        t_k   = k * dt
        V_approx = σ0 * exp(-0.5 * η_f^2 * t_k^(2*H_f) / (2*H_f))
        V_approx = max(V_approx, 1e-6)

        μ_r   = (μ_f - 0.5 * V_approx^2) * dt
        σ²_r  = V_approx^2 * dt
        r_k   = log_returns[k]
        ll   += -0.5 * (r_k - μ_r)^2 / σ²_r - 0.5 * log(2π * σ²_r)
    end
    return Float32(ll)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. JUMP-DIFFUSION MODEL (Merton + Neural Corrections)
# ─────────────────────────────────────────────────────────────────────────────

"""
    JumpDiffusion

Merton (1976) jump-diffusion model with neural corrections:

  d log S = (μ - σ²/2 - λk̄)dt + σ dW + Y dN

where:
  - N(t) is a Poisson process with intensity λ
  - Y ~ N(μ_J, σ_J²) is the jump size
  - k̄ = exp(μ_J + σ_J²/2) - 1 (compensator)
  - Neural networks correct drift and diffusion

Fields:
  - `μ`, `σ` : drift and diffusion
  - `λ`      : jump intensity (events per year)
  - `μ_J`    : mean log jump size
  - `σ_J`    : std of log jump size
  - `drift_net`, `vol_net` : neural corrections
"""
struct JumpDiffusion
    μ     :: Float32
    σ     :: Float32
    λ     :: Float32  # jump intensity
    μ_J   :: Float32  # mean log jump
    σ_J   :: Float32  # std log jump
    drift_net :: Union{DriftNet, Nothing}
    vol_net   :: Union{DiffusionNet, Nothing}
end

function JumpDiffusion(; μ::Float32=0.05f0, σ::Float32=0.15f0,
                         λ::Float32=10.0f0,   # ~10 jumps/year
                         μ_J::Float32=-0.1f0, σ_J::Float32=0.05f0,
                         use_neural::Bool=true, hidden_dim::Int=32)
    drift_n = use_neural ?
        build_drift_net(1; hidden_dim=hidden_dim, n_layers=2,
                         time_emb_dim=8, use_batchnorm=false) : nothing
    vol_n = use_neural ?
        build_diffusion_net(1; hidden_dim=hidden_dim, n_layers=2,
                             time_emb_dim=8, diagonal=true) : nothing
    return JumpDiffusion(μ, σ, λ, μ_J, σ_J, drift_n, vol_n)
end

Flux.@functor JumpDiffusion (drift_net, vol_net)

"""
    simulate_model(m::JumpDiffusion, S0, T, n_paths, dt; rng)
"""
function simulate_model(m::JumpDiffusion, S0::Float64, T::Float64,
                          n_paths::Int, dt::Float64;
                          rng = Random.GLOBAL_RNG)

    n_steps = ceil(Int, T / dt)
    S_paths = zeros(Float32, n_steps+1, n_paths)
    S_paths[1, :] .= Float32(S0)

    k_bar = exp(Float32(m.μ_J) + 0.5f0 * Float32(m.σ_J)^2) - 1f0

    for p in 1:n_paths
        log_S = Float32(log(S0))

        for k in 1:n_steps
            t     = Float32((k-1) * dt)
            dt_k  = Float32(min(dt, T - (k-1)*dt))
            sqdt  = sqrt(dt_k)

            # Neural corrections
            state = Float32[log_S]
            μ_corr = 0f0
            σ_eff  = Float32(m.σ)
            if m.drift_net !== nothing
                μ_corr += 0.05f0 * m.drift_net(state, t)[1]
            end
            if m.vol_net !== nothing
                σ_eff = max(σ_eff + 0.05f0 * m.vol_net(state, t)[1], 1f-4)
            end

            # Diffusion part
            μ_net  = Float32(m.μ) - 0.5f0 * σ_eff^2 - Float32(m.λ) * k_bar * dt_k + μ_corr * dt_k
            dW     = sqdt * Float32(randn(rng))
            log_S += μ_net + σ_eff * dW

            # Jump part (Poisson arrivals in [t, t+dt])
            n_jumps = rand(rng, Poisson(Float64(m.λ) * dt_k))
            for _ in 1:n_jumps
                Y      = Float32(m.μ_J) + Float32(m.σ_J) * Float32(randn(rng))
                log_S += Y
            end

            S_paths[k+1, p] = exp(log_S)
        end
    end

    return S_paths
end

"""
    merton_characteristic_fn(m::JumpDiffusion, u, T) → Complex

Characteristic function of log(S_T/S_0) under Merton jump-diffusion.
φ(u) = exp(iuμT - u²σ²T/2 + λT(exp(iuμ_J - u²σ_J²/2) - 1))
"""
function merton_characteristic_fn(m::JumpDiffusion, u::Complex, T::Float64)
    μ, σ, λ, μ_J, σ_J = Float64(m.μ), Float64(m.σ), Float64(m.λ),
                          Float64(m.μ_J), Float64(m.σ_J)
    k_bar = exp(μ_J + 0.5*σ_J^2) - 1

    iu = im * u
    φ  = exp(iu*(μ - 0.5*σ^2 - λ*k_bar)*T
             - 0.5*u^2*σ^2*T
             + λ*T*(exp(iu*μ_J - 0.5*u^2*σ_J^2) - 1))
    return φ
end

function log_likelihood_model(m::JumpDiffusion, log_returns::AbstractVector;
                                dt::Float64=1.0/252)
    n  = length(log_returns)
    ll = 0.0

    # Gauss-Hermite quadrature for jump component (up to n_jumps_max jumps)
    n_jumps_max = 20
    λ_dt = Float64(m.λ) * dt
    k_bar = exp(Float64(m.μ_J) + 0.5*Float64(m.σ_J)^2) - 1

    for k in 1:n
        r_k = log_returns[k]
        p   = 0.0

        for j in 0:n_jumps_max
            # Poisson probability of j jumps
            pois_j = exp(-λ_dt + j*log(λ_dt + 1e-100) - sum(log.(1:max(j,1))))

            # Conditional distribution: N(μ_dt + j·μ_J, σ²_dt + j·σ_J²)
            μ_cond  = (Float64(m.μ) - 0.5*Float64(m.σ)^2 - Float64(m.λ)*k_bar)*dt + j*Float64(m.μ_J)
            σ²_cond = Float64(m.σ)^2*dt + j*Float64(m.σ_J)^2

            p += pois_j * exp(-0.5*(r_k - μ_cond)^2/σ²_cond) / sqrt(2π*σ²_cond)
        end

        ll += log(max(p, 1e-300))
    end

    return Float32(ll)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. REGIME-SWITCHING MODEL
# ─────────────────────────────────────────────────────────────────────────────

"""
    RegimeSwitching

Continuous-time Markov chain (CTMC) regime model with neural emission
distributions per regime.

State: (price, regime) where regime ∈ {1, ..., K}.
In each regime k, log returns are distributed as N(μ_k, σ_k²).
Regime transitions occur at rate Q[i,j] (intensity matrix).

Neural enhancement: μ_k and σ_k can have neural state-dependent corrections.

Fields:
  - `K`        : number of regimes
  - `Q`        : K×K transition rate matrix (off-diagonal ≥ 0, rows sum to 0)
  - `μ_vec`    : K-vector of regime mean returns
  - `σ_vec`    : K-vector of regime volatilities
  - `regime_nets` : Vector{DriftNet} for per-regime neural corrections
"""
struct RegimeSwitching
    K     :: Int
    Q     :: Matrix{Float32}       # transition rate matrix
    μ_vec :: Vector{Float32}
    σ_vec :: Vector{Float32}
    regime_nets :: Vector{Union{DriftNet, Nothing}}
end

function RegimeSwitching(K::Int=2;
                          Q           = nothing,
                          μ_vec       = nothing,
                          σ_vec       = nothing,
                          use_neural  = true,
                          hidden_dim  = 16)

    # Default: 2-regime Bull/Bear
    if Q === nothing
        if K == 2
            Q_mat = Float32[-2.0  2.0; 0.5  -0.5]  # bull mean-rev 2/yr, bear 0.5/yr
        else
            Q_mat = Float32.(diagm(0 => fill(-2.0, K), 1 => fill(1.0, K-1), -1 => fill(1.0, K-1)))
            for i in 1:K; Q_mat[i,i] = -sum(Q_mat[i,:]) + Q_mat[i,i]; end
        end
    else
        Q_mat = Float32.(Q)
    end

    if μ_vec === nothing
        μ_vec = K == 2 ? Float32[0.15, -0.05] : Float32.(range(0.2, -0.1, K))
    end
    if σ_vec === nothing
        σ_vec = K == 2 ? Float32[0.10, 0.30] : Float32.(range(0.08, 0.40, K))
    end

    nets = [use_neural ?
        build_drift_net(1; hidden_dim=hidden_dim, n_layers=2,
                         time_emb_dim=4, use_batchnorm=false) :
        nothing for _ in 1:K]

    return RegimeSwitching(K, Q_mat, Float32.(μ_vec), Float32.(σ_vec), nets)
end

Flux.@functor RegimeSwitching (regime_nets,)

"""
    sample_regime_path(m::RegimeSwitching, T, dt; rng) → regime_indicators

Simulate discrete-time regime sequence using matrix exponentiation.
Returns integer vector of regimes at each time step.
"""
function sample_regime_path(m::RegimeSwitching, T::Float64, dt::Float64;
                              rng = Random.GLOBAL_RNG,
                              initial_regime::Int=1)
    n_steps = ceil(Int, T / dt)
    regimes = Vector{Int}(undef, n_steps+1)
    regimes[1] = initial_regime

    # Discrete-time transition matrix P = exp(Q·dt)
    P = exp(Matrix(Float64.(m.Q)) .* dt)
    # Ensure rows are valid probabilities
    P = max.(P, 0.0)
    for i in 1:m.K; P[i,:] ./= sum(P[i,:]); end

    regime = initial_regime
    for k in 1:n_steps
        probs  = P[regime, :]
        regime = sample(1:m.K, Weights(probs))
        regimes[k+1] = regime
    end

    return regimes
end

"""
    simulate_model(m::RegimeSwitching, S0, T, n_paths, dt; rng)
"""
function simulate_model(m::RegimeSwitching, S0::Float64, T::Float64,
                          n_paths::Int, dt::Float64;
                          rng = Random.GLOBAL_RNG)

    n_steps = ceil(Int, T / dt)
    S_paths = zeros(Float32, n_steps+1, n_paths)
    R_paths = zeros(Int,     n_steps+1, n_paths)  # regime indicators
    S_paths[1, :] .= Float32(S0)
    R_paths[1, :] .= 1

    for p in 1:n_paths
        regimes = sample_regime_path(m, T, dt; rng=rng)
        R_paths[:, p] = regimes
        log_S = Float32(log(S0))

        for k in 1:n_steps
            t        = Float32((k-1) * dt)
            dt_k     = Float32(min(dt, T - (k-1)*dt))
            regime_k = regimes[k]

            μ_k  = m.μ_vec[regime_k]
            σ_k  = m.σ_vec[regime_k]

            # Neural correction
            if m.regime_nets[regime_k] !== nothing
                state = Float32[log_S]
                corr  = m.regime_nets[regime_k](state, t)
                μ_k  += 0.01f0 * corr[1]
            end

            dW     = Float32(sqrt(dt_k) * randn(rng))
            log_S += (μ_k - 0.5f0 * σ_k^2) * dt_k + σ_k * dW
            S_paths[k+1, p] = exp(log_S)
        end
    end

    return S_paths, R_paths
end

"""
    stationary_distribution(m::RegimeSwitching) → π

Compute the stationary distribution of the CTMC regime process.
Solve π·Q = 0 with Σπᵢ = 1.
"""
function stationary_distribution(m::RegimeSwitching)
    Q    = Float64.(m.Q)
    K    = m.K
    # Set up linear system [Q'  1; ones'] [π; 0] = [0; 1]
    A    = [Q'[:, 1:K-1] ones(K); ones(1, K-1) 1.0]
    b    = [zeros(K); 1.0]
    x    = A \ b
    π    = x[1:K]
    return Float32.(abs.(π) ./ sum(abs.(π)))
end

function log_likelihood_model(m::RegimeSwitching, log_returns::AbstractVector;
                                dt::Float64=1.0/252)
    n = length(log_returns)
    K = m.K
    ll = 0.0

    # Hamilton filter: forward algorithm
    π_stat = stationary_distribution(m)
    P = Float64.(exp(Matrix(Float64.(m.Q)) .* dt))
    P = max.(P, 0.0)
    for i in 1:K; P[i,:] ./= sum(P[i,:]); end

    # α_k[i] = p(r_{1:k}, s_k=i) (unnormalised)
    α = Float64.(π_stat)

    for k in 1:n
        r_k = log_returns[k]
        # Emission probabilities in each regime
        emit = [pdf(Normal(Float64(m.μ_vec[i])*dt - 0.5*Float64(m.σ_vec[i])^2*dt,
                           Float64(m.σ_vec[i])*sqrt(dt)), r_k) for i in 1:K]

        # Filter update
        α_new = (P' * α) .* emit
        total = sum(α_new)
        ll   += log(max(total, 1e-300))
        α     = α_new ./ max(total, 1e-300)
    end

    return Float32(ll)
end

# Utility: forward simulate a single scalar SDE
function simulate_scalar_sde(f_fn, g_fn, x0::Float64, T::Float64, dt::Float64,
                               params=nothing; rng=Random.GLOBAL_RNG)
    n_steps = ceil(Int, T / dt)
    path = zeros(Float64, n_steps+1)
    path[1] = x0
    x = x0
    t = 0.0

    for k in 1:n_steps
        dt_k = min(dt, T - (k-1)*dt)
        dW   = sqrt(dt_k) * randn(rng)
        μ    = f_fn(x, t, params)
        σ    = g_fn(x, t, params)
        x    = x + μ*dt_k + σ*dW
        t   += dt_k
        path[k+1] = x
    end
    return path
end
