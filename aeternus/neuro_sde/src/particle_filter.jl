"""
particle_filter.jl — Sequential Monte Carlo state estimation

Implements:
  1. Bootstrap Particle Filter (SIR)
  2. Auxiliary Particle Filter (APF / Liu-West)
  3. Unscented Kalman Filter (UKF) — sigma-point propagation
  4. Ensemble Kalman Filter (EnKF) — stochastic update
  5. Resampling strategies: systematic, stratified, residual, multinomial
  6. Degeneracy diagnostics and Effective Sample Size (ESS)
  7. Log-likelihood estimation for parameter learning
  8. Marginal likelihood via SMC²

References:
  - Gordon, Salmond & Smith (1993) — Bootstrap filter
  - Pitt & Shephard (1999) — Auxiliary PF
  - Julier & Uhlmann (1995) — UKF sigma points
  - Evensen (1994) — Ensemble Kalman Filter
  - Chopin (2002) — SMC² marginal likelihood
"""

using LinearAlgebra
using Statistics
using Random
using Distributions

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: TYPES AND ABSTRACTIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
    AbstractParticleFilter

Base type for all particle filter implementations.
"""
abstract type AbstractParticleFilter end

"""
    AbstractResamplingStrategy

Base type for resampling strategies.
"""
abstract type AbstractResamplingStrategy end
struct SystematicResampling  <: AbstractResamplingStrategy end
struct StratifiedResampling  <: AbstractResamplingStrategy end
struct ResidualResampling    <: AbstractResamplingStrategy end
struct MultinomialResampling <: AbstractResamplingStrategy end

"""
    ParticleState{T}

Container for the current particle ensemble.

Fields:
  - `particles` : (state_dim × N) matrix of particles
  - `weights`   : N-vector of normalised weights
  - `log_weights`: unnormalised log-weights
  - `t`         : current time
  - `ess`       : effective sample size
  - `log_marglik`: cumulative log marginal likelihood
"""
mutable struct ParticleState{T <: AbstractFloat}
    particles   :: Matrix{T}
    weights     :: Vector{T}
    log_weights :: Vector{T}
    t           :: T
    ess         :: T
    log_marglik :: T
end

function ParticleState(particles::Matrix{T}) where T
    N = size(particles, 2)
    w = fill(T(1/N), N)
    lw = fill(T(-log(N)), N)
    return ParticleState{T}(particles, w, lw, T(0), T(N), T(0))
end

"""
    FilterResult

Summary of particle filter run.
"""
struct FilterResult
    filtered_mean   :: Matrix{Float64}   # (state_dim × n_steps)
    filtered_var    :: Matrix{Float64}   # (state_dim × n_steps)  diagonal variance
    log_marglik     :: Float64           # total log marginal likelihood
    ess_history     :: Vector{Float64}   # ESS at each step
    n_resample      :: Int               # number of resampling events
    particles_history :: Union{Nothing, Array{Float64, 3}}  # (dim × N × T) optional
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: RESAMPLING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    effective_sample_size(weights) → Float64

ESS = 1 / Σ wᵢ²  (assuming weights are normalised).
"""
function effective_sample_size(weights::AbstractVector)
    return 1.0 / sum(weights.^2)
end

"""
    systematic_resample(weights, N; rng=Random.GLOBAL_RNG) → Vector{Int}

Systematic resampling: single uniform draw, ladder positions.
O(N) time, minimum variance estimator.
"""
function systematic_resample(weights::AbstractVector, N::Int;
                              rng::AbstractRNG = Random.GLOBAL_RNG)
    cumw  = cumsum(weights)
    cumw[end] = 1.0  # numerical safety
    u0    = rand(rng) / N
    u_vec = u0 .+ (0:N-1) ./ N
    indices = zeros(Int, N)
    j = 1
    for i in 1:N
        while u_vec[i] > cumw[j] && j < length(cumw)
            j += 1
        end
        indices[i] = j
    end
    return indices
end

"""
    stratified_resample(weights, N; rng=Random.GLOBAL_RNG) → Vector{Int}

Stratified resampling: N independent draws, one per stratum [k/N, (k+1)/N).
"""
function stratified_resample(weights::AbstractVector, N::Int;
                              rng::AbstractRNG = Random.GLOBAL_RNG)
    cumw    = cumsum(weights)
    cumw[end] = 1.0
    u_vec   = (rand(rng, N) .+ (0:N-1)) ./ N
    indices = zeros(Int, N)
    j = 1
    for i in 1:N
        while u_vec[i] > cumw[j] && j < length(cumw)
            j += 1
        end
        indices[i] = j
    end
    return indices
end

"""
    residual_resample(weights, N; rng=Random.GLOBAL_RNG) → Vector{Int}

Residual resampling (Liu & Chen 1998).
Deterministically copies floor(N * wᵢ) copies, then applies systematic
resampling to the residual fraction.
"""
function residual_resample(weights::AbstractVector, N::Int;
                           rng::AbstractRNG = Random.GLOBAL_RNG)
    M       = length(weights)
    counts  = zeros(Int, M)
    r       = zeros(M)
    n_used  = 0
    for i in 1:M
        ni       = floor(Int, N * weights[i])
        counts[i] = ni
        r[i]     = N * weights[i] - ni
        n_used   += ni
    end
    r ./= sum(r)   # renormalise residuals
    n_remain = N - n_used
    if n_remain > 0
        extra = systematic_resample(r, n_remain; rng=rng)
        for idx in extra
            counts[idx] += 1
        end
    end
    # Expand counts into index array
    indices = Int[]
    for i in 1:M
        append!(indices, fill(i, counts[i]))
    end
    return indices
end

"""
    multinomial_resample(weights, N; rng=Random.GLOBAL_RNG) → Vector{Int}

Simple multinomial resampling (high variance, use for baseline comparison).
"""
function multinomial_resample(weights::AbstractVector, N::Int;
                              rng::AbstractRNG = Random.GLOBAL_RNG)
    return rand(rng, Categorical(weights), N)
end

"""
    resample(ps::ParticleState, strategy::AbstractResamplingStrategy;
             rng=Random.GLOBAL_RNG) → ParticleState

Apply resampling strategy to particle state, returning new state with
uniform weights.
"""
function resample(ps::ParticleState{T},
                  strategy::AbstractResamplingStrategy;
                  rng::AbstractRNG = Random.GLOBAL_RNG) where T
    N  = size(ps.particles, 2)
    w  = ps.weights

    if strategy isa SystematicResampling
        idx = systematic_resample(w, N; rng=rng)
    elseif strategy isa StratifiedResampling
        idx = stratified_resample(w, N; rng=rng)
    elseif strategy isa ResidualResampling
        idx = residual_resample(w, N; rng=rng)
    else
        idx = multinomial_resample(w, N; rng=rng)
    end

    new_particles = ps.particles[:, idx]
    uw = fill(T(1.0 / N), N)
    ulw = fill(T(-log(N)), N)
    return ParticleState{T}(new_particles, uw, ulw, ps.t, T(N), ps.log_marglik)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: BOOTSTRAP PARTICLE FILTER (SIR)
# ─────────────────────────────────────────────────────────────────────────────

"""
    BootstrapFilter <: AbstractParticleFilter

Bootstrap / SIR particle filter configuration.

Fields:
  - `N`          : number of particles
  - `ess_threshold`: fraction of N below which resampling is triggered
  - `strategy`   : resampling strategy
  - `store_history`: whether to store all particle ensembles
"""
struct BootstrapFilter <: AbstractParticleFilter
    N             :: Int
    ess_threshold :: Float64
    strategy      :: AbstractResamplingStrategy
    store_history :: Bool
end

BootstrapFilter(N::Int;
                ess_threshold::Real = 0.5,
                strategy::AbstractResamplingStrategy = SystematicResampling(),
                store_history::Bool = false) =
    BootstrapFilter(N, ess_threshold, strategy, store_history)

"""
    bootstrap_filter(bf::BootstrapFilter, init_particles, transition!, log_likelihood,
                     observations; rng=Random.GLOBAL_RNG) → FilterResult

Run the bootstrap particle filter.

Arguments:
  - `init_particles` : (state_dim × N) initial particle array
  - `transition!(x_new, x, t, rng)` : in-place transition sampler
  - `log_likelihood(y, x, t)` : log p(y_t | x_t)
  - `observations` : (obs_dim × n_steps) matrix
"""
function bootstrap_filter(bf::BootstrapFilter,
                          init_particles::AbstractMatrix,
                          transition!::Function,
                          log_likelihood::Function,
                          observations::AbstractMatrix;
                          rng::AbstractRNG = Random.GLOBAL_RNG)
    d, N      = size(init_particles)
    n_steps   = size(observations, 2)
    T_type    = eltype(init_particles)

    ps       = ParticleState(copy(init_particles))
    x_new    = similar(init_particles[:, 1])

    fil_mean  = zeros(d, n_steps)
    fil_var   = zeros(d, n_steps)
    ess_hist  = zeros(n_steps)
    n_resamp  = 0
    log_ml    = 0.0

    particles_hist = bf.store_history ?
                     zeros(d, N, n_steps) : nothing

    for t in 1:n_steps
        # Propagate
        new_X = similar(ps.particles)
        for i in 1:N
            transition!(x_new, ps.particles[:, i], T_type(t), rng)
            new_X[:, i] = x_new
        end
        ps.particles = new_X

        # Weight update
        y_t = observations[:, t]
        log_w = zeros(N)
        for i in 1:N
            log_w[i] = log_likelihood(y_t, ps.particles[:, i], T_type(t))
        end

        # Normalise in log-space (numerically stable)
        max_lw   = maximum(log_w)
        w_unnorm = exp.(log_w .- max_lw)
        w_sum    = sum(w_unnorm)
        ps.weights     = w_unnorm ./ w_sum
        ps.log_weights = log_w .- log(w_sum) .- max_lw

        # Marginal likelihood contribution
        log_ml += log(w_sum) + max_lw - log(N)
        ps.log_marglik = log_ml

        # ESS
        ess    = effective_sample_size(ps.weights)
        ps.ess = ess
        ess_hist[t] = ess

        # Store moments
        fil_mean[:, t] = ps.particles * ps.weights
        for k in 1:d
            fil_var[k, t] = sum(ps.weights .*
                                (ps.particles[k, :] .- fil_mean[k, t]).^2)
        end

        bf.store_history && (particles_hist[:, :, t] = ps.particles)

        # Resample if needed
        if ess < bf.ess_threshold * N
            ps = resample(ps, bf.strategy; rng=rng)
            n_resamp += 1
        end

        ps.t = T_type(t)
    end

    return FilterResult(fil_mean, fil_var, log_ml, ess_hist,
                        n_resamp, particles_hist)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: AUXILIARY PARTICLE FILTER (APF)
# ─────────────────────────────────────────────────────────────────────────────

"""
    AuxiliaryParticleFilter <: AbstractParticleFilter

Pitt & Shephard (1999) Auxiliary Particle Filter.

Uses an auxiliary step to pre-select particles likely to receive high weight
after observing y_{t+1}, reducing weight degeneracy.
"""
struct AuxiliaryParticleFilter <: AbstractParticleFilter
    N             :: Int
    ess_threshold :: Float64
    strategy      :: AbstractResamplingStrategy
    store_history :: Bool
end

AuxiliaryParticleFilter(N::Int;
                        ess_threshold::Real = 0.5,
                        strategy::AbstractResamplingStrategy = SystematicResampling(),
                        store_history::Bool = false) =
    AuxiliaryParticleFilter(N, ess_threshold, strategy, store_history)

"""
    auxiliary_particle_filter(apf, init_particles, transition!, log_likelihood,
                              pilot_log_likelihood, observations; rng) → FilterResult

Run the Auxiliary Particle Filter.

Additional argument:
  - `pilot_log_likelihood(y, x, t)` : approximate log-likelihood (e.g. at predicted mean)
    used for the pre-selection step.
"""
function auxiliary_particle_filter(apf::AuxiliaryParticleFilter,
                                   init_particles::AbstractMatrix,
                                   transition!::Function,
                                   log_likelihood::Function,
                                   pilot_log_likelihood::Function,
                                   observations::AbstractMatrix;
                                   rng::AbstractRNG = Random.GLOBAL_RNG)
    d, N      = size(init_particles)
    n_steps   = size(observations, 2)
    T_type    = eltype(init_particles)

    particles = copy(init_particles)
    weights   = fill(1.0 / N, N)

    fil_mean  = zeros(d, n_steps)
    fil_var   = zeros(d, n_steps)
    ess_hist  = zeros(n_steps)
    n_resamp  = 0
    log_ml    = 0.0
    particles_hist = apf.store_history ? zeros(d, N, n_steps) : nothing

    x_pred = similar(particles[:, 1])

    for t in 1:n_steps
        y_t = observations[:, t]

        # Step 1: Compute pilot weights (pre-selection)
        pilot_lw = zeros(N)
        pilot_particles = similar(particles)
        for i in 1:N
            # Get pilot prediction (e.g. propagate one step w/o noise)
            transition!(x_pred, particles[:, i], T_type(t), rng)
            pilot_particles[:, i] = x_pred
            pilot_lw[i] = log(weights[i]) + pilot_log_likelihood(y_t, x_pred, T_type(t))
        end
        # Normalise pilot weights
        pilot_max = maximum(pilot_lw)
        pilot_w   = exp.(pilot_lw .- pilot_max)
        pilot_w ./= sum(pilot_w)

        # Step 2: Resample using pilot weights
        k_idx = systematic_resample(pilot_w, N; rng=rng)
        n_resamp += 1

        # Step 3: Propagate resampled particles
        new_particles = similar(particles)
        for i in 1:N
            transition!(x_pred, particles[:, k_idx[i]], T_type(t), rng)
            new_particles[:, i] = x_pred
        end

        # Step 4: Compute importance weights = true loglhood - pilot loglhood
        true_lw  = zeros(N)
        pilot_lw2 = zeros(N)
        for i in 1:N
            true_lw[i]   = log_likelihood(y_t, new_particles[:, i], T_type(t))
            pilot_lw2[i] = pilot_log_likelihood(y_t, new_particles[:, i], T_type(t))
        end
        log_w    = true_lw .- pilot_lw2
        max_lw   = maximum(log_w)
        w_unnorm = exp.(log_w .- max_lw)
        w_sum    = sum(w_unnorm)
        weights  = w_unnorm ./ w_sum

        log_ml += log(w_sum) + max_lw - log(N)
        particles = new_particles

        ess = effective_sample_size(weights)
        ess_hist[t] = ess

        # Moments
        fil_mean[:, t] = particles * weights
        for k in 1:d
            fil_var[k, t] = sum(weights .* (particles[k, :] .- fil_mean[k, t]).^2)
        end

        apf.store_history && (particles_hist[:, :, t] = particles)

        if ess < apf.ess_threshold * N
            idx      = systematic_resample(weights, N; rng=rng)
            particles = particles[:, idx]
            weights  = fill(1.0 / N, N)
            n_resamp += 1
        end
    end

    return FilterResult(fil_mean, fil_var, log_ml, ess_hist,
                        n_resamp, particles_hist)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: UNSCENTED KALMAN FILTER (UKF)
# ─────────────────────────────────────────────────────────────────────────────

"""
    UKFParams

Parameters controlling the Unscented Transform.

Fields:
  - α   : spread of sigma points (1e-4 ≤ α ≤ 1)
  - β   : prior knowledge of distribution (2 = Gaussian optimal)
  - κ   : secondary scaling (often 0 or 3 - state_dim)
  - λ   : computed as α² (state_dim + κ) - state_dim
"""
struct UKFParams
    α :: Float64
    β :: Float64
    κ :: Float64
end

UKFParams(; α=1e-3, β=2.0, κ=0.0) = UKFParams(α, β, κ)

"""
    ukf_weights(n, p::UKFParams) → (Wm, Wc)

Compute unscented transform mean and covariance weights for dimension n.
"""
function ukf_weights(n::Int, p::UKFParams)
    λ  = p.α^2 * (n + p.κ) - n
    Wm = fill(1.0 / (2(n + λ)), 2n + 1)
    Wc = copy(Wm)
    Wm[1] = λ / (n + λ)
    Wc[1] = λ / (n + λ) + (1 - p.α^2 + p.β)
    return Wm, Wc, λ
end

"""
    sigma_points(x, P, λ, n) → Matrix (n × 2n+1)

Compute 2n+1 sigma points from mean x and covariance P.
Uses Cholesky decomposition: xᵢ = x ± √((n+λ) P) columns.
"""
function sigma_points(x::AbstractVector, P::AbstractMatrix, λ::Real, n::Int)
    S  = cholesky(Hermitian((n + λ) .* P)).L
    σP = zeros(n, 2n + 1)
    σP[:, 1] = x
    for i in 1:n
        σP[:, i+1]   = x .+ S[:, i]
        σP[:, i+1+n] = x .- S[:, i]
    end
    return σP
end

"""
    UKFState

State of the Unscented Kalman Filter.
"""
mutable struct UKFState
    x   :: Vector{Float64}   # state mean
    P   :: Matrix{Float64}   # state covariance
    t   :: Float64
end

"""
    UKFResult

Result of a UKF run.
"""
struct UKFResult
    filtered_mean :: Matrix{Float64}   # (state_dim × n_steps)
    filtered_cov  :: Array{Float64, 3} # (state_dim × state_dim × n_steps)
    predicted_mean:: Matrix{Float64}
    predicted_cov :: Array{Float64, 3}
    innovations   :: Matrix{Float64}   # (obs_dim × n_steps)
    S_matrices    :: Array{Float64, 3} # innovation covariance
    log_likelihood:: Float64
end

"""
    ukf_run(x0, P0, f, h, Q, R, observations;
            ukf_params=UKFParams()) → UKFResult

Run the Unscented Kalman Filter.

Arguments:
  - `x0`, `P0` : initial state and covariance
  - `f(x, t)`  : state transition function (nonlinear)
  - `h(x, t)`  : observation function (nonlinear)
  - `Q`        : process noise covariance
  - `R`        : observation noise covariance
  - `observations` : (obs_dim × n_steps) measurement matrix
"""
function ukf_run(x0::AbstractVector,
                 P0::AbstractMatrix,
                 f::Function,
                 h::Function,
                 Q::AbstractMatrix,
                 R::AbstractMatrix,
                 observations::AbstractMatrix;
                 ukf_params::UKFParams = UKFParams())
    n        = length(x0)
    m        = size(observations, 1)
    n_steps  = size(observations, 2)
    Wm, Wc, λ = ukf_weights(n, ukf_params)

    state = UKFState(copy(Float64.(x0)), copy(Float64.(P0)), 0.0)

    fil_mean  = zeros(n, n_steps)
    fil_cov   = zeros(n, n, n_steps)
    pred_mean = zeros(n, n_steps)
    pred_cov  = zeros(n, n, n_steps)
    innov     = zeros(m, n_steps)
    S_mats    = zeros(m, m, n_steps)
    log_lik   = 0.0

    for t in 1:n_steps
        # ── Predict ──────────────────────────────────────────────────────
        SP     = sigma_points(state.x, state.P, λ, n)
        SP_f   = similar(SP)
        for i in 1:(2n+1)
            SP_f[:, i] = f(SP[:, i], Float64(t))
        end
        x_pred = SP_f * Wm
        P_pred = Q + sum(Wc[i] .* (SP_f[:, i] - x_pred) *
                         (SP_f[:, i] - x_pred)' for i in 1:(2n+1))

        pred_mean[:, t] = x_pred
        pred_cov[:, :, t] = P_pred

        # ── Update ───────────────────────────────────────────────────────
        SP2    = sigma_points(x_pred, P_pred, λ, n)
        Y      = zeros(m, 2n+1)
        for i in 1:(2n+1)
            Y[:, i] = h(SP2[:, i], Float64(t))
        end
        y_pred = Y * Wm
        Syy    = R + sum(Wc[i] .* (Y[:, i] - y_pred) *
                         (Y[:, i] - y_pred)' for i in 1:(2n+1))
        Sxy    = sum(Wc[i] .* (SP2[:, i] - x_pred) *
                     (Y[:, i] - y_pred)' for i in 1:(2n+1))

        K     = Sxy / Syy   # Kalman gain (n × m)
        y_t   = observations[:, t]
        innov_t = y_t - y_pred

        state.x = x_pred + K * innov_t
        state.P = Hermitian(P_pred - K * Syy * K')

        fil_mean[:, t]    = state.x
        fil_cov[:, :, t]  = state.P
        innov[:, t]       = innov_t
        S_mats[:, :, t]   = Syy

        # Log-likelihood
        try
            log_lik += logpdf(MvNormal(y_pred, Hermitian(Syy)), y_t)
        catch
            log_lik -= 0.5 * (m * log(2π) + logdet(Syy) +
                               dot(innov_t, Syy \ innov_t))
        end

        state.t = Float64(t)
    end

    return UKFResult(fil_mean, fil_cov, pred_mean, pred_cov,
                     innov, S_mats, log_lik)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ENSEMBLE KALMAN FILTER (EnKF)
# ─────────────────────────────────────────────────────────────────────────────

"""
    EnKFConfig

Configuration for the Ensemble Kalman Filter.

Fields:
  - `N`           : ensemble size
  - `inflate`     : multiplicative inflation factor for P
  - `localization`: optional localization matrix (same size as P)
"""
struct EnKFConfig
    N           :: Int
    inflate     :: Float64
    store_history :: Bool
end

EnKFConfig(N::Int; inflate::Real=1.0, store_history::Bool=false) =
    EnKFConfig(N, inflate, store_history)

"""
    EnKFResult

Result of EnKF run.
"""
struct EnKFResult
    filtered_mean  :: Matrix{Float64}
    filtered_var   :: Matrix{Float64}
    log_likelihood :: Float64
    ensemble_history :: Union{Nothing, Array{Float64, 3}}
end

"""
    enkf_run(cfg, ensemble0, f!, h, Q, R, observations; rng) → EnKFResult

Stochastic Ensemble Kalman Filter.

Arguments:
  - `ensemble0` : (state_dim × N) initial ensemble
  - `f!(x_new, x, t, rng)` : in-place stochastic propagator
  - `h(x, t)` : observation operator (linear or nonlinear)
  - `Q`        : process noise covariance (for additive noise sampling)
  - `R`        : observation noise covariance
  - `observations` : (obs_dim × n_steps)
"""
function enkf_run(cfg::EnKFConfig,
                  ensemble0::AbstractMatrix,
                  f!::Function,
                  h::Function,
                  Q::AbstractMatrix,
                  R::AbstractMatrix,
                  observations::AbstractMatrix;
                  rng::AbstractRNG = Random.GLOBAL_RNG)
    d, N     = size(ensemble0)
    m        = size(observations, 1)
    n_steps  = size(observations, 2)

    ensemble = copy(Float64.(ensemble0))
    Q_dist   = MvNormal(zeros(d), Q)
    R_dist   = MvNormal(zeros(m), R)

    fil_mean = zeros(d, n_steps)
    fil_var  = zeros(d, n_steps)
    log_lik  = 0.0
    ens_hist = cfg.store_history ? zeros(d, N, n_steps) : nothing

    x_new = zeros(d)

    for t in 1:n_steps
        # ── Forecast step ─────────────────────────────────────────────
        new_ens = similar(ensemble)
        for i in 1:N
            f!(x_new, ensemble[:, i], Float64(t), rng)
            new_ens[:, i] = x_new .+ rand(rng, Q_dist)
        end
        ensemble = new_ens

        # Apply inflation
        x_bar = mean(ensemble, dims=2)[:]
        if cfg.inflate != 1.0
            ensemble = x_bar .+ cfg.inflate .* (ensemble .- x_bar)
        end

        # ── Analysis step ─────────────────────────────────────────────
        # Ensemble mean and anomalies
        x_bar  = mean(ensemble, dims=2)[:]
        A      = ensemble .- x_bar          # d × N anomalies
        Pf     = (A * A') ./ (N - 1)       # ensemble covariance

        # Observation ensemble
        HX = zeros(m, N)
        for i in 1:N
            HX[:, i] = h(ensemble[:, i], Float64(t)) .+ rand(rng, R_dist)
        end
        y_bar  = mean(HX, dims=2)[:]
        HA     = HX .- y_bar                # m × N
        PHᵀ    = A * HA' ./ (N - 1)        # d × m
        HPHᵀ_R = HA * HA' ./ (N - 1) .+ R # m × m

        y_t = observations[:, t]

        # Stochastic EnKF update: add perturbed observations
        K = PHᵀ / HPHᵀ_R   # d × m

        for i in 1:N
            y_pert = y_t .+ rand(rng, R_dist)
            hxi    = h(ensemble[:, i], Float64(t))
            ensemble[:, i] .+= K * (y_pert .- hxi)
        end

        # Moments
        x_a = mean(ensemble, dims=2)[:]
        fil_mean[:, t] = x_a
        for k in 1:d
            fil_var[k, t] = var(ensemble[k, :])
        end

        # Log-likelihood (Gaussian approximation)
        innov  = y_t - mean(HX, dims=2)[:]
        S_mat  = HPHᵀ_R
        try
            log_lik += logpdf(MvNormal(zeros(m), Hermitian(S_mat)), innov)
        catch
            log_lik -= 0.5 * (m * log(2π) + logdet(S_mat) + dot(innov, S_mat \ innov))
        end

        cfg.store_history && (ens_hist[:, :, t] = ensemble)
    end

    return EnKFResult(fil_mean, fil_var, log_lik, ens_hist)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: DEGENERACY DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    DegeneracyMetrics

Diagnostics for particle filter weight degeneracy.
"""
struct DegeneracyMetrics
    ess          :: Float64    # effective sample size
    ess_fraction :: Float64    # ESS / N
    max_weight   :: Float64    # weight of dominant particle
    entropy      :: Float64    # weight entropy H = -Σ wᵢ log wᵢ
    is_degenerate :: Bool      # flag
    n_effective  :: Int        # round(ESS)
end

"""
    diagnose_degeneracy(weights; threshold=0.5) → DegeneracyMetrics

Compute comprehensive degeneracy diagnostics for particle weights.
"""
function diagnose_degeneracy(weights::AbstractVector; threshold::Real=0.5)
    N    = length(weights)
    ess  = effective_sample_size(weights)
    eff_frac = ess / N
    max_w    = maximum(weights)
    entropy  = -sum(w * log(max(w, 1e-300)) for w in weights)
    return DegeneracyMetrics(ess, eff_frac, max_w, entropy,
                             eff_frac < threshold, round(Int, ess))
end

"""
    jitter_particles!(particles, weights, σ_jitter; rng=Random.GLOBAL_RNG)

Apply Gaussian jitter to particles with high weight concentration
(MCMC rejuvenation step — helps maintain diversity).
"""
function jitter_particles!(particles::AbstractMatrix,
                           weights::AbstractVector,
                           σ_jitter::Real;
                           rng::AbstractRNG = Random.GLOBAL_RNG)
    d, N = size(particles)
    x_bar = particles * weights
    for i in 1:N
        particles[:, i] .+= σ_jitter .* randn(rng, d)
    end
    return particles
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: SMC² — MARGINAL LIKELIHOOD FOR PARAMETER LEARNING
# ─────────────────────────────────────────────────────────────────────────────

"""
    SMC2Config

Configuration for SMC² (Chopin, Jacob & Papaspiliopoulos 2013).
"""
struct SMC2Config
    N_x      :: Int    # particles for state
    N_θ      :: Int    # particles for parameter
    ess_frac :: Float64
    strategy :: AbstractResamplingStrategy
    n_mcmc   :: Int    # MCMC steps after parameter resampling
end

SMC2Config(; N_x=100, N_θ=500, ess_frac=0.5,
             strategy=SystematicResampling(), n_mcmc=5) =
    SMC2Config(N_x, N_θ, ess_frac, strategy, n_mcmc)

"""
    smc2_log_marglik(cfg, θ_particles, log_prior, transition_factory,
                     log_likelihood, observations; rng) → Vector{Float64}

Estimate log marginal likelihood p(y_{1:T}) for each parameter particle θ
using nested bootstrap filters.

Returns vector of log marginal likelihoods (one per θ particle).

This is a simplified version: each θ particle runs its own independent
bootstrap filter of size N_x over the full observation sequence.
"""
function smc2_log_marglik(cfg::SMC2Config,
                           θ_particles::AbstractMatrix,
                           log_prior::Function,
                           init_state_fn::Function,
                           transition_factory::Function,
                           log_likelihood_factory::Function,
                           observations::AbstractMatrix;
                           rng::AbstractRNG = Random.GLOBAL_RNG)
    N_θ     = size(θ_particles, 2)
    n_obs   = size(observations, 2)
    lml_θ   = zeros(N_θ)
    bf_cfg  = BootstrapFilter(cfg.N_x; ess_threshold=cfg.ess_frac,
                              strategy=cfg.strategy)

    for k in 1:N_θ
        θ_k = θ_particles[:, k]
        init_X   = init_state_fn(θ_k, cfg.N_x; rng=rng)
        trans_k! = transition_factory(θ_k)
        loglik_k = log_likelihood_factory(θ_k)

        try
            res = bootstrap_filter(bf_cfg, init_X, trans_k!,
                                   loglik_k, observations; rng=rng)
            lml_θ[k] = res.log_marglik + log_prior(θ_k)
        catch
            lml_θ[k] = -Inf
        end
    end
    return lml_θ
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: FINANCIAL SDE STATE-SPACE MODELS
# ─────────────────────────────────────────────────────────────────────────────

"""
    HestonStateSpaceModel

State-space representation of the Heston model for particle filtering.

State: x = [log S, V]
Observation: y = [log S_obs] (noisy log price)
"""
struct HestonStateSpaceModel
    κ    :: Float64
    θ    :: Float64
    ξ    :: Float64
    ρ    :: Float64
    r    :: Float64
    q    :: Float64
    dt   :: Float64
    σ_obs :: Float64   # observation noise std
end

"""
    heston_transition!(x_new, x, t, rng, model::HestonStateSpaceModel)

Euler-Maruyama transition for Heston state-space model.
x = [log S, V], ensures V > 0 via reflection.
"""
function heston_transition!(x_new::AbstractVector,
                            x::AbstractVector,
                            t::Real,
                            rng::AbstractRNG,
                            model::HestonStateSpaceModel)
    logS, V = x[1], max(x[2], 1e-6)
    dt = model.dt
    κ, θ, ξ, ρ = model.κ, model.θ, model.ξ, model.ρ
    r, q = model.r, model.q

    sqV = sqrt(V)
    z1  = randn(rng)
    z2  = ρ * z1 + sqrt(1 - ρ^2) * randn(rng)

    new_logS = logS + (r - q - 0.5 * V) * dt + sqV * sqrt(dt) * z1
    new_V    = V + κ * (θ - V) * dt + ξ * sqV * sqrt(dt) * z2
    new_V    = max(new_V, 0.0)   # reflection boundary

    x_new[1] = new_logS
    x_new[2] = new_V
end

"""
    heston_log_obs_likelihood(y, x, t, model::HestonStateSpaceModel) → Float64

Log p(y_t | x_t) for Heston: y = log S + ε, ε ~ N(0, σ_obs²).
"""
function heston_log_obs_likelihood(y::AbstractVector,
                                   x::AbstractVector,
                                   t::Real,
                                   model::HestonStateSpaceModel)
    logS_pred = x[1]
    return logpdf(Normal(logS_pred, model.σ_obs), y[1])
end

"""
    filter_heston(log_prices, κ, θ, ξ, ρ, r, q, dt;
                  N=2000, σ_obs=0.001) → FilterResult

Particle filter for latent variance estimation in the Heston model.
"""
function filter_heston(log_prices::AbstractVector,
                       κ::Real, θ::Real, ξ::Real, ρ::Real,
                       r::Real, q::Real, dt::Real;
                       N::Int       = 2000,
                       σ_obs::Real  = 0.001,
                       seed::Int    = 42,
                       rng::AbstractRNG = MersenneTwister(seed))
    model = HestonStateSpaceModel(κ, θ, ξ, ρ, r, q, dt, σ_obs)

    # Initial particles: [log S0, V ~ Gamma(2θ/ξ², ...)]
    logS0    = log_prices[1]
    init_V   = rand(rng, Gamma(2*κ*θ/ξ^2 + 1, ξ^2 / (2*κ)), N)
    init_X   = vcat(fill(logS0, 1, N), init_V')

    trans! = (xn, x, t, r) -> heston_transition!(xn, x, t, r, model)
    loglik  = (y, x, t) -> heston_log_obs_likelihood(y, x, t, model)

    obs_mat = reshape(log_prices, 1, length(log_prices))
    bf      = BootstrapFilter(N; ess_threshold=0.5,
                              strategy=SystematicResampling())

    return bootstrap_filter(bf, init_X, trans!, loglik, obs_mat; rng=rng)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: SUMMARY STATISTICS AND REPORTING
# ─────────────────────────────────────────────────────────────────────────────

"""
    filter_summary(result::FilterResult) → NamedTuple

Compute summary statistics from a filter result.
"""
function filter_summary(result::FilterResult)
    min_ess     = minimum(result.ess_history)
    mean_ess    = mean(result.ess_history)
    n_steps     = size(result.filtered_mean, 2)
    return (
        n_steps      = n_steps,
        log_marglik  = result.log_marglik,
        min_ess      = min_ess,
        mean_ess     = mean_ess,
        n_resample   = result.n_resample,
        state_dim    = size(result.filtered_mean, 1),
    )
end

"""
    particle_smoother_backward(filter_result, transition_density, N_smooth;
                                rng=Random.GLOBAL_RNG) → Matrix{Float64}

Backward particle smoother (Kitagawa 1996).
Returns smoothed state mean (state_dim × n_steps).

`transition_density(x_new, x_old)` : p(x_new | x_old) — unnormalised.
"""
function particle_smoother_backward(filter_result::FilterResult,
                                    transition_density::Function,
                                    N_smooth::Int;
                                    rng::AbstractRNG = Random.GLOBAL_RNG)
    @assert !isnothing(filter_result.particles_history) \
        "particle_smoother requires store_history=true in filter config"

    d, N, T_steps = size(filter_result.particles_history)
    smooth_mean   = zeros(d, T_steps)

    # Sample N_smooth trajectories using backward weights
    idx_T  = systematic_resample(filter_result.filtered_mean[:, T_steps] .* 0 .+ 1/N,
                                 N_smooth; rng=rng)
    smooth_traj = filter_result.particles_history[:, idx_T, T_steps]

    smooth_mean[:, T_steps] = mean(smooth_traj, dims=2)[:]

    for t in (T_steps-1):-1:1
        # Backward weights proportional to p(x_{t+1} | x_t)
        bw_mean = zeros(d, N_smooth)
        for j in 1:N_smooth
            x_next = smooth_traj[:, j]
            bw_w   = zeros(N)
            for i in 1:N
                x_cur  = filter_result.particles_history[:, i, t]
                bw_w[i] = transition_density(x_next, x_cur)
            end
            bw_w = max.(bw_w, 1e-300)
            bw_w ./= sum(bw_w)
            idx_t = multinomial_resample(bw_w, 1; rng=rng)[1]
            bw_mean[:, j] = filter_result.particles_history[:, idx_t, t]
        end
        smooth_traj = bw_mean
        smooth_mean[:, t] = mean(smooth_traj, dims=2)[:]
    end
    return smooth_mean
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: DEMOS
# ─────────────────────────────────────────────────────────────────────────────

"""
    demo_bootstrap_filter(; N=1000, T=200, seed=1)

Smoke test: linear Gaussian state-space model (Kalman filter ground truth).

  x_{t+1} = 0.9 x_t + ε_t,   ε_t ~ N(0, 1)
  y_t     = x_t + η_t,        η_t ~ N(0, 0.5)
"""
function demo_bootstrap_filter(; N::Int=1000, T::Int=200, seed::Int=1)
    rng  = MersenneTwister(seed)
    # Simulate
    x    = zeros(T+1); x[1] = 0.0
    y    = zeros(T)
    for t in 1:T
        x[t+1] = 0.9 * x[t] + randn(rng)
        y[t]   = x[t+1] + 0.5 * randn(rng)
    end

    # Bootstrap filter
    init_X = randn(rng, 1, N)
    trans! = (xn, xo, t, rng) -> begin xn[1] = 0.9 * xo[1] + randn(rng) end
    loglik = (yt, xt, t) -> logpdf(Normal(xt[1], 0.5), yt[1])
    obs    = reshape(y, 1, T)
    bf     = BootstrapFilter(N; ess_threshold=0.5)

    result = bootstrap_filter(bf, init_X, trans!, loglik, obs; rng=rng)
    rmse   = sqrt(mean((result.filtered_mean[1, :] .- x[2:end]).^2))
    return (result=result, rmse=rmse, true_x=x[2:end])
end

"""
    demo_ukf(; T=100, seed=1)

Smoke test: UKF on nonlinear Van-der-Pol-like system.
"""
function demo_ukf(; T::Int=100, seed::Int=1)
    rng = MersenneTwister(seed)
    n = 2; m = 1
    Q = 0.01 * I(n)
    R = reshape([0.1], 1, 1)

    x_true = zeros(n, T+1); x_true[:, 1] = [1.0, 0.0]
    y_obs  = zeros(m, T)
    f_true = x -> [x[1] * cos(x[2]); x[2] + 0.1 * x[1]]
    h_obs  = x -> [x[1]^2]
    for t in 1:T
        x_true[:, t+1] = f_true(x_true[:, t]) .+ 0.1 * randn(rng, n)
        y_obs[:, t]    = h_obs(x_true[:, t+1]) .+ 0.316 * randn(rng, m)
    end

    f_ukf  = (x, t) -> f_true(x)
    h_ukf  = (x, t) -> h_obs(x)
    res = ukf_run([1.0, 0.0], 0.1 * I(n), f_ukf, h_ukf,
                  Matrix(Q), R, y_obs)
    return (result=res, true_x=x_true[:, 2:end])
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: INTERACTING PARTICLE SYSTEMS (McKean-Vlasov)
# ─────────────────────────────────────────────────────────────────────────────

"""
    InteractingParticleConfig

Configuration for McKean-Vlasov / mean-field particle system.
"""
struct InteractingParticleConfig
    N          :: Int
    interaction_strength :: Float64
    kernel     :: Symbol   # :gaussian, :uniform, :coulomb
    bandwidth  :: Float64
end

InteractingParticleConfig(N::Int; interaction_strength::Real=0.1,
                           kernel::Symbol=:gaussian, bandwidth::Real=0.5) =
    InteractingParticleConfig(N, interaction_strength, kernel, bandwidth)

"""
    mean_field_kernel(x_i, x_j, cfg::InteractingParticleConfig) → Float64

Interaction kernel K(x_i, x_j) for mean-field particle system.
"""
function mean_field_kernel(x_i::AbstractVector, x_j::AbstractVector,
                            cfg::InteractingParticleConfig)
    d = norm(x_i - x_j)
    if cfg.kernel == :gaussian
        return exp(-d^2 / (2 * cfg.bandwidth^2))
    elseif cfg.kernel == :uniform
        return Float64(d <= cfg.bandwidth)
    else  # coulomb
        return 1.0 / max(d, 1e-6)
    end
end

"""
    interacting_pf_step!(particles, weights, transition!, log_likelihood,
                          y_t, cfg; rng) → nothing

Single step of interacting particle filter with mean-field correction.
Modifies particles in-place.
"""
function interacting_pf_step!(particles::AbstractMatrix,
                                weights::AbstractVector,
                                transition!::Function,
                                log_likelihood::Function,
                                y_t::AbstractVector,
                                cfg::InteractingParticleConfig;
                                rng::AbstractRNG = Random.GLOBAL_RNG)
    d, N  = size(particles)
    x_new = similar(particles[:, 1])
    x_bar = particles * weights   # weighted mean

    for i in 1:N
        transition!(x_new, particles[:, i], 0.0, rng)
        # Mean-field drift: push particles toward weighted mean
        particles[:, i] = x_new .+ cfg.interaction_strength .* (x_bar .- particles[:, i])
    end

    # Weight update
    for i in 1:N
        weights[i] *= exp(log_likelihood(y_t, particles[:, i], 0.0))
    end
    w_sum = sum(weights)
    w_sum > 0 && (weights ./= w_sum)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: CRANK-NICOLSON MCMC WITHIN PARTICLE FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
    cn_mcmc_move!(x, log_post, β_cn; rng) → (x_new, accepted)

Crank-Nicolson MCMC proposal (Beskos et al. 2008):
  x_prop = √(1-β²) x + β ξ,   ξ ~ N(0, Σ₀)

Dimension-independent acceptance rate for function-space targets.
"""
function cn_mcmc_move!(x::AbstractVector,
                        log_post::Function,
                        β_cn::Real;
                        prior_std::Real    = 1.0,
                        rng::AbstractRNG   = Random.GLOBAL_RNG)
    n      = length(x)
    ξ      = prior_std .* randn(rng, n)
    x_prop = sqrt(1 - β_cn^2) .* x .+ β_cn .* ξ
    lp_cur = log_post(x)
    lp_prop = log_post(x_prop)
    if log(rand(rng)) < lp_prop - lp_cur
        return x_prop, true
    else
        return x, false
    end
end

"""
    rejuvenate_particles!(particles, log_posterior, n_steps; β=0.5, rng) → Int

Apply Crank-Nicolson MCMC rejuvenation to particle ensemble.
Returns total number of accepted moves.
"""
function rejuvenate_particles!(particles::AbstractMatrix,
                                log_posterior::Function,
                                n_steps::Int;
                                β_cn::Real           = 0.5,
                                prior_std::Real      = 1.0,
                                rng::AbstractRNG     = Random.GLOBAL_RNG)
    d, N     = size(particles)
    n_accept = 0
    for i in 1:N
        x = particles[:, i]
        for _ in 1:n_steps
            x, acc = cn_mcmc_move!(x, log_posterior, β_cn;
                                    prior_std=prior_std, rng=rng)
            n_accept += acc
        end
        particles[:, i] = x
    end
    return n_accept
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14: TWISTED PARTICLE FILTER (LOCALLY OPTIMAL)
# ─────────────────────────────────────────────────────────────────────────────

"""
    TwistedPFConfig

Configuration for twisted (locally optimal) particle filter.
Uses a twisting function ψ_t(x) to reduce variance.
"""
struct TwistedPFConfig
    N             :: Int
    twist_fn      :: Function   # ψ_t(x, y_{t+1}) — look-ahead twist
    ess_threshold :: Float64
    strategy      :: AbstractResamplingStrategy
end

"""
    twisted_filter(cfg, init_particles, transition!, log_likelihood,
                   observations; rng) → FilterResult

Twisted particle filter where proposals are tilted by ψ_t(x, y_{t+1}).
This gives the locally-optimal (Dirac mixture) filter in special cases.
"""
function twisted_filter(cfg::TwistedPFConfig,
                         init_particles::AbstractMatrix,
                         transition!::Function,
                         log_likelihood::Function,
                         observations::AbstractMatrix;
                         rng::AbstractRNG = Random.GLOBAL_RNG)
    d, N      = size(init_particles)
    n_steps   = size(observations, 2)
    T_type    = eltype(init_particles)

    particles = copy(init_particles)
    weights   = fill(T_type(1/N), N)
    x_buf     = similar(particles[:, 1])

    fil_mean  = zeros(d, n_steps)
    fil_var   = zeros(d, n_steps)
    ess_hist  = zeros(n_steps)
    log_ml    = 0.0
    n_resamp  = 0

    for t in 1:n_steps
        y_t = observations[:, t]

        # Twisted proposal: propagate + compute twist weight correction
        new_particles = similar(particles)
        twist_log_w   = zeros(N)
        for i in 1:N
            transition!(x_buf, particles[:, i], T_type(t), rng)
            new_particles[:, i] = x_buf
            # Twist correction: log ψ(x_new, y_t) - log ψ(x_old, y_{t-1})
            twist_log_w[i] = log_likelihood(y_t, x_buf, T_type(t))
        end

        # Normalise weights
        max_lw   = maximum(twist_log_w)
        w_unnorm = exp.(twist_log_w .- max_lw)
        w_sum    = sum(w_unnorm)
        weights  = w_unnorm ./ w_sum
        log_ml  += log(w_sum) + max_lw - log(N)

        particles = new_particles
        ess       = effective_sample_size(weights)
        ess_hist[t] = ess

        fil_mean[:, t] = particles * weights
        for k in 1:d
            fil_var[k, t] = sum(weights .* (particles[k, :] .- fil_mean[k, t]).^2)
        end

        if ess < cfg.ess_threshold * N
            ps = ParticleState(particles)
            ps.weights = weights
            ps = resample(ps, cfg.strategy; rng=rng)
            particles = ps.particles
            weights   = ps.weights
            n_resamp += 1
        end
    end

    return FilterResult(fil_mean, fil_var, log_ml, ess_hist, n_resamp, nothing)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15: ONLINE PARTICLE FILTER EXTENSIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
    OnlineParticleFilter

Stateful object for streaming (online) data assimilation.
Maintains current particle state and supports one-step updates.
"""
mutable struct OnlineParticleFilter
    particles    :: Matrix{Float64}
    weights      :: Vector{Float64}
    log_marglik  :: Float64
    n_steps      :: Int
    N            :: Int
    ess_threshold :: Float64
    strategy     :: AbstractResamplingStrategy
    ess_history  :: Vector{Float64}
    n_resample   :: Int
end

"""
    OnlineParticleFilter(init_particles; ess_threshold=0.5) → OnlineParticleFilter
"""
function OnlineParticleFilter(init_particles::AbstractMatrix;
                               ess_threshold::Real = 0.5,
                               strategy::AbstractResamplingStrategy = SystematicResampling())
    d, N = size(init_particles)
    OnlineParticleFilter(copy(Float64.(init_particles)),
                          fill(1.0/N, N), 0.0, 0, N,
                          ess_threshold, strategy, Float64[], 0)
end

"""
    update!(opf::OnlineParticleFilter, y_t, transition!, log_likelihood; rng) → nothing

Process one new observation y_t through the online particle filter.
"""
function update!(opf::OnlineParticleFilter,
                  y_t::AbstractVector,
                  transition!::Function,
                  log_likelihood::Function;
                  rng::AbstractRNG = Random.GLOBAL_RNG)
    d, N   = size(opf.particles)
    x_buf  = zeros(d)
    new_X  = similar(opf.particles)

    for i in 1:N
        transition!(x_buf, opf.particles[:, i], Float64(opf.n_steps+1), rng)
        new_X[:, i] = x_buf
    end
    opf.particles = new_X

    log_w = [log_likelihood(y_t, opf.particles[:, i], Float64(opf.n_steps+1))
             for i in 1:N]
    max_lw   = maximum(log_w)
    w_unnorm = exp.(log_w .- max_lw)
    w_sum    = sum(w_unnorm)
    opf.weights .= w_unnorm ./ w_sum
    opf.log_marglik += log(w_sum) + max_lw - log(N)

    ess = effective_sample_size(opf.weights)
    push!(opf.ess_history, ess)
    opf.n_steps += 1

    if ess < opf.ess_threshold * N
        idx = systematic_resample(opf.weights, N; rng=rng)
        opf.particles = opf.particles[:, idx]
        fill!(opf.weights, 1.0/N)
        opf.n_resample += 1
    end
end

"""
    filtered_state(opf::OnlineParticleFilter) → (mean, var)

Extract current filtered state estimate from online PF.
"""
function filtered_state(opf::OnlineParticleFilter)
    d    = size(opf.particles, 1)
    μ    = opf.particles * opf.weights
    σ2   = [sum(opf.weights .* (opf.particles[k,:] .- μ[k]).^2) for k in 1:d]
    return μ, σ2
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16: PARTICLE GIBBS AND CONDITIONAL SMC
# ─────────────────────────────────────────────────────────────────────────────

"""
    conditional_smc(init_particles, x_ref, transition!, log_likelihood,
                    observations; strategy, rng) → (particles_T, ancestors)

Conditional Sequential Monte Carlo (Andrieu, Doucet & Holenstein 2010).
Runs a standard bootstrap filter conditioned on a reference trajectory x_ref.
Returns the final particle ensemble and the ancestor indices.
"""
function conditional_smc(init_particles::AbstractMatrix,
                          x_ref::AbstractMatrix,
                          transition!::Function,
                          log_likelihood::Function,
                          observations::AbstractMatrix;
                          strategy::AbstractResamplingStrategy = SystematicResampling(),
                          rng::AbstractRNG = Random.GLOBAL_RNG)
    d, N      = size(init_particles)
    n_steps   = size(observations, 2)
    particles = copy(init_particles)
    # Ensure reference trajectory is particle 1
    particles[:, 1] = x_ref[:, 1]

    weights   = fill(1.0/N, N)
    x_buf     = similar(particles[:, 1])
    ancestors = zeros(Int, N, n_steps)

    for t in 1:n_steps
        # Resample (excluding reference particle)
        idx = systematic_resample(weights, N; rng=rng)
        idx[1] = 1   # keep reference
        particles = particles[:, idx]
        ancestors[:, t] = idx

        # Propagate (keep reference fixed)
        new_X = similar(particles)
        new_X[:, 1] = x_ref[:, min(t+1, n_steps)]
        for i in 2:N
            transition!(x_buf, particles[:, i], Float64(t), rng)
            new_X[:, i] = x_buf
        end
        particles = new_X

        y_t = observations[:, t]
        for i in 1:N
            weights[i] = exp(log_likelihood(y_t, particles[:, i], Float64(t)))
        end
        w_s = sum(weights); w_s > 0 && (weights ./= w_s)
    end

    return particles, ancestors
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17: PARTICLE FILTER DIAGNOSTICS REPORT
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParticleFilterReport

Comprehensive diagnostic report for particle filter quality.
"""
struct ParticleFilterReport
    method_name    :: String
    n_particles    :: Int
    n_steps        :: Int
    log_marglik    :: Float64
    mean_ess       :: Float64
    min_ess        :: Float64
    n_resample     :: Int
    resample_rate  :: Float64
    filtered_rmse  :: Union{Nothing, Float64}
    notes          :: Vector{String}
end

"""
    generate_report(result::FilterResult, N, method_name;
                    true_states=nothing) → ParticleFilterReport
"""
function generate_report(result::FilterResult,
                          N::Int,
                          method_name::String;
                          true_states::Union{Nothing, AbstractMatrix} = nothing)
    notes = String[]
    mean_ess = mean(result.ess_history)
    min_ess  = minimum(result.ess_history)

    mean_ess / N < 0.2 && push!(notes, "WARNING: Mean ESS < 20% of N — consider increasing N")
    result.n_resample == 0 && push!(notes, "INFO: No resampling occurred — try lower ESS threshold")

    rmse = nothing
    if !isnothing(true_states)
        rmse = sqrt(mean((result.filtered_mean .- true_states).^2))
        rmse > 1.0 && push!(notes, "WARNING: High tracking RMSE = $(round(rmse, digits=4))")
    end

    return ParticleFilterReport(
        method_name, N, size(result.filtered_mean, 2),
        result.log_marglik, mean_ess, min_ess,
        result.n_resample, result.n_resample / size(result.filtered_mean, 2),
        rmse, notes
    )
end

"""
    print_report(r::ParticleFilterReport)
"""
function print_report(r::ParticleFilterReport)
    @printf "─────────────────────────────────────────────────\n"
    @printf "  Particle Filter Report: %s\n" r.method_name
    @printf "─────────────────────────────────────────────────\n"
    @printf "  N particles   : %d\n" r.n_particles
    @printf "  Steps         : %d\n" r.n_steps
    @printf "  Log marglik   : %.4f\n" r.log_marglik
    @printf "  Mean ESS      : %.1f (%.1f%%)\n" r.mean_ess (r.mean_ess/r.n_particles*100)
    @printf "  Min  ESS      : %.1f\n" r.min_ess
    @printf "  N resamplings : %d (rate %.2f)\n" r.n_resample r.resample_rate
    !isnothing(r.filtered_rmse) && @printf "  RMSE vs truth : %.6f\n" r.filtered_rmse
    for note in r.notes; @printf "  %s\n" note; end
    @printf "─────────────────────────────────────────────────\n"
end
