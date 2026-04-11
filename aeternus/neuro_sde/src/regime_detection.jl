"""
regime_detection.jl — Online regime detection from Neural SDE latent states

Implements:
  1. Particle filter with systematic resampling (1000 particles)
  2. Kalman-Bucy filter as linear baseline
  3. Viterbi-like (MAP) decoding in continuous time
  4. Online regime detection using latent SDE posterior

The particle filter propagates a set of weighted particles {(zₖ^i, wₖ^i)}
under the prior dynamics (Neural SDE) and reweights them using the
observation likelihood. Systematic resampling prevents particle degeneracy.

References:
  - Gordon, Salmond & Smith (1993) — bootstrap particle filter
  - Doucet, de Freitas & Gordon (2001) — Sequential MC Methods
  - Viterbi (1967) — dynamic programming for MAP decoding
"""

using LinearAlgebra
using Statistics
using Random
using Distributions

# ─────────────────────────────────────────────────────────────────────────────
# PARTICLE FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParticleFilter

Bootstrap particle filter for regime state estimation.

Fields:
  - `n_particles`  : number of particles
  - `d_latent`     : latent state dimension
  - `n_regimes`    : number of discrete regimes
  - `transition_fn`: f(z, regime, t) → z_new (SDE step function)
  - `likelihood_fn`: g(observation, z, regime) → log p(obs | z, regime)
  - `init_fn`      : () → (z_0, regime_0) initial sample function
"""
struct ParticleFilter{F,G,H}
    n_particles    :: Int
    d_latent       :: Int
    n_regimes      :: Int
    transition_fn  :: F     # (z, regime, t, dt, rng) → z_new
    likelihood_fn  :: G     # (obs, z, regime) → log_weight
    init_fn        :: H     # (rng) → (z, regime)
    Q_matrix       :: Matrix{Float64}   # CTMC transition rate matrix
end

"""
    ParticleFilterState

Runtime state of the particle filter.

Fields:
  - `particles`  : (d_latent, n_particles) current state particles
  - `regimes`    : (n_particles,) current regime for each particle
  - `log_weights`: (n_particles,) unnormalised log weights
  - `t`          : current time
  - `ess`        : effective sample size at last step
"""
mutable struct ParticleFilterState
    particles   :: Matrix{Float64}
    regimes     :: Vector{Int}
    log_weights :: Vector{Float64}
    t           :: Float64
    ess         :: Float64
    history_z   :: Vector{Matrix{Float64}}    # stored for smoothing
    history_r   :: Vector{Vector{Int}}
    history_w   :: Vector{Vector{Float64}}
end

"""
    init_particle_filter(pf::ParticleFilter; rng) → ParticleFilterState
"""
function init_particle_filter(pf::ParticleFilter;
                               rng = Random.GLOBAL_RNG)
    n = pf.n_particles
    d = pf.d_latent

    particles   = zeros(d, n)
    regimes     = zeros(Int, n)
    log_weights = fill(-log(n), n)  # uniform initial weights

    for i in 1:n
        z, r = pf.init_fn(rng)
        particles[:, i] = z
        regimes[i]      = r
    end

    return ParticleFilterState(
        particles, regimes, log_weights, 0.0, Float64(n),
        Matrix{Float64}[], Vector{Int}[], Vector{Float64}[]
    )
end

"""
    systematic_resample(weights) → indices

Systematic resampling: O(N) resampling that minimises variance.

Algorithm:
  1. Compute cumulative sum of weights
  2. Draw single uniform u ~ U[0, 1/N]
  3. Select particle i when cumsum[i] ≥ u + (k-1)/N for k=1:N
"""
function systematic_resample(weights::AbstractVector{<:Real})
    n   = length(weights)
    w   = weights ./ sum(weights)  # normalise
    cdf = cumsum(w)
    indices = zeros(Int, n)

    u0  = rand() / n
    j   = 1
    for k in 1:n
        u_k = u0 + (k-1) / n
        while j < n && cdf[j] < u_k
            j += 1
        end
        indices[k] = j
    end

    return indices
end

"""
    multinomial_resample(weights) → indices

Multinomial resampling (simpler but higher variance than systematic).
"""
function multinomial_resample(weights::AbstractVector{<:Real}, rng=Random.GLOBAL_RNG)
    n   = length(weights)
    w   = weights ./ sum(weights)
    return rand(rng, Categorical(w), n)
end

"""
    regime_transition_sample(regime, Q_matrix, dt; rng) → new_regime

Sample next regime from continuous-time Markov chain over interval dt.
Uses the exact transition probability matrix P = exp(Q·dt).
"""
function regime_transition_sample(regime::Int, Q_matrix::Matrix{Float64},
                                    dt::Float64; rng=Random.GLOBAL_RNG)
    K  = size(Q_matrix, 1)
    P  = exp(Q_matrix .* dt)          # matrix exponentiation
    P  = max.(P, 0.0)
    for i in 1:K; P[i,:] ./= sum(P[i,:]); end  # normalise rows

    probs = P[regime, :]
    return sample(1:K, Weights(probs))
end

"""
    particle_filter_step!(state, pf, observation, dt; rng)

Run one step of the particle filter:
  1. Propagate particles under prior dynamics (SDE + regime transitions)
  2. Weight by observation likelihood
  3. Resample if ESS < threshold

Returns updated state (in-place).
"""
function particle_filter_step!(state::ParticleFilterState,
                                pf::ParticleFilter,
                                observation,
                                dt::Float64;
                                rng           = Random.GLOBAL_RNG,
                                ess_threshold = 0.5)

    n = pf.n_particles

    # 1. PROPAGATION: sample new particles from prior
    new_particles = similar(state.particles)
    new_regimes   = similar(state.regimes)

    for i in 1:n
        z_i = state.particles[:, i]
        r_i = state.regimes[i]

        # Regime transition
        r_new = regime_transition_sample(r_i, pf.Q_matrix, dt; rng=rng)

        # State transition (SDE step)
        z_new = pf.transition_fn(z_i, r_new, state.t, dt, rng)

        new_particles[:, i] = z_new
        new_regimes[i]      = r_new
    end

    state.particles = new_particles
    state.regimes   = new_regimes

    # 2. WEIGHTING: update log weights with observation likelihood
    for i in 1:n
        ll_i = pf.likelihood_fn(observation, state.particles[:, i], state.regimes[i])
        state.log_weights[i] += ll_i
    end

    # Normalise (log-sum-exp for stability)
    lw_max = maximum(state.log_weights)
    log_Z  = lw_max + log(sum(exp.(state.log_weights .- lw_max)))
    state.log_weights .-= log_Z

    # 3. COMPUTE ESS
    w_norm = exp.(state.log_weights)
    state.ess = 1.0 / sum(w_norm.^2)

    # 4. RESAMPLE if ESS too low
    if state.ess < ess_threshold * n
        indices             = systematic_resample(w_norm)
        state.particles     = state.particles[:, indices]
        state.regimes       = state.regimes[indices]
        state.log_weights   = fill(-log(n), n)
    end

    state.t += dt

    # Store history
    push!(state.history_z, copy(state.particles))
    push!(state.history_r, copy(state.regimes))
    push!(state.history_w, copy(exp.(state.log_weights)))

    return state
end

"""
    regime_probabilities(state::ParticleFilterState, n_regimes) → Vector{Float64}

Compute the filtered regime probabilities from current particle weights.
π_k = Σᵢ wᵢ · 𝟙[regimeᵢ = k]
"""
function regime_probabilities(state::ParticleFilterState, n_regimes::Int)
    w_norm = exp.(state.log_weights)
    w_norm ./= sum(w_norm)
    π = zeros(n_regimes)
    for (i, w) in enumerate(w_norm)
        r = state.regimes[i]
        r <= n_regimes && (π[r] += w)
    end
    return π
end

# ─────────────────────────────────────────────────────────────────────────────
# DETECT REGIMES (full time series)
# ─────────────────────────────────────────────────────────────────────────────

"""
    RegimeDetector

Combines a LatentDynamicsModel with a particle filter for online regime detection.

Fields:
  - `ldm`         : LatentDynamicsModel
  - `n_regimes`   : number of regimes
  - `Q_matrix`    : CTMC transition rates
  - `n_particles` : particle filter size
  - `regime_mu`   : mean return per regime
  - `regime_sigma`: vol per regime
"""
struct RegimeDetector
    ldm          :: Any   # LatentDynamicsModel (Any to avoid circular)
    n_regimes    :: Int
    Q_matrix     :: Matrix{Float64}
    n_particles  :: Int
    regime_mu    :: Vector{Float64}
    regime_sigma :: Vector{Float64}
end

function RegimeDetector(ldm, n_regimes::Int=2;
                          Q_matrix     = nothing,
                          n_particles  = 1000,
                          regime_mu    = nothing,
                          regime_sigma = nothing)

    if Q_matrix === nothing
        Q_matrix = n_regimes == 2 ?
            [-2.0 2.0; 0.5 -0.5] :
            Float64.(diagm(0 => fill(-2.0, n_regimes),
                            1 => fill(1.0, n_regimes-1),
                            -1 => fill(1.0, n_regimes-1)))
    end

    if regime_mu === nothing
        regime_mu = n_regimes == 2 ? [0.15/252, -0.05/252] :
                     Float64.(range(0.2/252, -0.1/252, n_regimes))
    end

    if regime_sigma === nothing
        regime_sigma = n_regimes == 2 ? [0.10/sqrt(252), 0.30/sqrt(252)] :
                        Float64.(range(0.08/sqrt(252), 0.40/sqrt(252), n_regimes))
    end

    return RegimeDetector(ldm, n_regimes, Float64.(Q_matrix),
                           n_particles, Float64.(regime_mu), Float64.(regime_sigma))
end

"""
    detect_regimes(rd::RegimeDetector, returns; dt, context_len, rng)

Run particle filter on full return series to obtain time-varying regime
probability estimates.

Returns:
  - `regime_probs` : (n_regimes, n_obs) matrix of filtered probabilities
  - `map_regimes`  : (n_obs,) MAP regime at each time step
  - `pf_state`     : final particle filter state
"""
function detect_regimes(rd::RegimeDetector, returns::AbstractVector;
                          dt::Float64       = 1.0/252,
                          context_len::Int  = 60,
                          rng               = Random.GLOBAL_RNG)

    n      = length(returns)
    d_lat  = rd.ldm.d_latent
    K      = rd.n_regimes

    # Build particle filter
    function transition_fn(z, regime, t, dt_k, rng_fn)
        z32 = Float32.(z)
        t32 = Float32(t)
        dt32 = Float32(dt_k)
        μ_sde = drift_at(rd.ldm.sde_model, z32, t32)
        σ_sde = diffusion_at(rd.ldm.sde_model, z32, t32)
        dW    = Float32.(sqrt(dt32) .* randn(rng_fn, d_lat))
        z_new = z32 .+ μ_sde .* dt32 .+ σ_sde .* dW
        return Float64.(z_new)
    end

    function likelihood_fn(obs, z, regime)
        r       = Float64(obs)
        μ_k     = rd.regime_mu[regime]
        σ_k     = rd.regime_sigma[regime]
        σ²      = max(σ_k^2, 1e-10)
        return -0.5*(r - μ_k)^2/σ² - 0.5*log(2π*σ²)
    end

    function init_fn(rng_fn)
        # Encode a small window as initial state
        if n >= context_len
            ctx = reshape(Float32.(returns[1:context_len]), 1, context_len)
            μ_q, σ_q = encode_returns(rd.ldm.encoder, ctx)
            ε = Float32.(randn(rng_fn, d_lat))
            z0 = Float64.(μ_q .+ σ_q .* ε)
        else
            z0 = zeros(d_lat)
        end
        regime0 = rand(rng_fn, 1:K)
        return z0, regime0
    end

    pf = ParticleFilter(rd.n_particles, d_lat, K,
                         transition_fn, likelihood_fn, init_fn,
                         Float64.(rd.Q_matrix))

    state = init_particle_filter(pf; rng=rng)

    regime_probs = zeros(K, n)
    map_regimes  = zeros(Int, n)

    start_idx = min(context_len + 1, n)

    for k in start_idx:n
        obs = returns[k]
        particle_filter_step!(state, pf, obs, dt; rng=rng)

        π_k = regime_probabilities(state, K)
        regime_probs[:, k] = π_k
        map_regimes[k]     = argmax(π_k)
    end

    # Fill in context period with stationary distribution
    π_stat = abs.(exp(rd.Q_matrix' * context_len*dt)[1,:])
    π_stat ./= sum(π_stat)
    for k in 1:start_idx-1
        regime_probs[:, k] = π_stat
        map_regimes[k]     = argmax(π_stat)
    end

    return regime_probs, map_regimes, state
end

# ─────────────────────────────────────────────────────────────────────────────
# VITERBI-LIKE MAP DECODING
# ─────────────────────────────────────────────────────────────────────────────

"""
    viterbi_decode(regime_probs, Q_matrix, dt) → map_path

Given filtered regime probabilities, compute the most likely regime path
using the Viterbi algorithm on the discretised CTMC.

This is a smoothing step: unlike the online regime_probs (filtered),
this looks at the full sequence and finds the globally most probable path.
"""
function viterbi_decode(regime_probs::AbstractMatrix,
                          Q_matrix::Matrix{Float64},
                          dt::Float64)

    K, T = size(regime_probs)

    # Discrete transition matrix P = exp(Q·dt)
    P = exp(Q_matrix .* dt)
    P = max.(P, 0.0)
    for i in 1:K; P[i,:] ./= sum(P[i,:]); end

    log_P    = log.(max.(P, 1e-300))
    log_emit = log.(max.(regime_probs, 1e-300))

    # Viterbi forward pass
    δ = zeros(K, T)       # max log-prob to reach (k, t)
    ψ = zeros(Int, K, T)  # backpointer

    # Initialise from stationary distribution
    π_stat = ones(K) ./ K
    δ[:, 1] = log.(π_stat) .+ log_emit[:, 1]

    for t in 2:T
        for j in 1:K
            best_prev_score = -Inf
            best_prev_state = 1
            for i in 1:K
                score = δ[i, t-1] + log_P[i, j]
                if score > best_prev_score
                    best_prev_score = score
                    best_prev_state = i
                end
            end
            δ[j, t] = best_prev_score + log_emit[j, t]
            ψ[j, t] = best_prev_state
        end
    end

    # Backtrack
    map_path    = zeros(Int, T)
    map_path[T] = argmax(δ[:, T])
    for t in T-1:-1:1
        map_path[t] = ψ[map_path[t+1], t+1]
    end

    return map_path
end

# ─────────────────────────────────────────────────────────────────────────────
# KALMAN-BUCY FILTER (linear baseline)
# ─────────────────────────────────────────────────────────────────────────────

"""
    KalmanBucy

Kalman-Bucy filter for linear Gaussian state-space model.

State dynamics:  dz = F·z·dt + G·dW_t     (linear SDE)
Observations:    r_t = H·z_t + noise_t    (linear observation)

This serves as a linear baseline to compare against the neural SDE + particle filter.

Fields:
  - `F` : drift matrix (d_latent × d_latent)
  - `G` : diffusion matrix (d_latent × d_noise)
  - `H` : observation matrix (d_obs × d_latent)
  - `Q` : process noise covariance (G·Gᵀ)
  - `R` : observation noise covariance
"""
struct KalmanBucy
    F :: Matrix{Float64}   # system matrix
    G :: Matrix{Float64}   # diffusion matrix
    H :: Matrix{Float64}   # observation matrix
    Q :: Matrix{Float64}   # process noise Q = G·Gᵀ
    R :: Matrix{Float64}   # observation noise
end

function KalmanBucy(d_latent::Int, d_obs::Int;
                     F = nothing, G = nothing, H = nothing,
                     q_scale::Float64 = 0.1, r_scale::Float64 = 0.01)

    F_mat = F === nothing ? -0.5 * I(d_latent)   |> Matrix{Float64} : Matrix{Float64}(F)
    G_mat = G === nothing ? sqrt(q_scale) * I(d_latent) |> Matrix{Float64} : Matrix{Float64}(G)
    H_mat = H === nothing ? [ones(1, d_latent) ./ d_latent] |> Matrix{Float64} : Matrix{Float64}(H)
    if H === nothing && size(H_mat, 1) != d_obs
        H_mat = randn(d_obs, d_latent) .* 0.1
    end
    Q_mat = G_mat * G_mat'
    R_mat = r_scale * I(d_obs) |> Matrix{Float64}

    return KalmanBucy(F_mat, G_mat, H_mat, Q_mat, R_mat)
end

"""
    KalmanBucyState

Runtime state of the Kalman-Bucy filter.
"""
mutable struct KalmanBucyState
    μ   :: Vector{Float64}    # filtered mean
    Σ   :: Matrix{Float64}    # filtered covariance
    t   :: Float64
    log_likelihoods :: Vector{Float64}
end

"""
    kalman_predict!(state, kb, dt)

Kalman-Bucy prediction step (continuous-time Euler discretisation):
  μ_{t+dt} = μ_t + F·μ_t·dt
  Σ_{t+dt} = Σ_t + (F·Σ_t + Σ_t·Fᵀ + Q)·dt
"""
function kalman_predict!(state::KalmanBucyState, kb::KalmanBucy, dt::Float64)
    F, Q = kb.F, kb.Q
    state.μ = state.μ .+ F * state.μ .* dt
    state.Σ = state.Σ .+ (F * state.Σ .+ state.Σ * F' .+ Q) .* dt
    state.Σ = Symmetric(state.Σ)  # enforce symmetry
    state.t += dt
end

"""
    kalman_update!(state, kb, observation)

Kalman-Bucy update step (discrete observation):
  Innovation:   ν = r_t - H·μ_t
  Innovation Σ: S = H·Σ_t·Hᵀ + R
  Kalman gain:  K = Σ_t·Hᵀ·S⁻¹
  Updated mean: μ_t = μ_t + K·ν
  Updated Σ:    Σ_t = (I - K·H)·Σ_t
"""
function kalman_update!(state::KalmanBucyState, kb::KalmanBucy,
                          observation::AbstractVector)
    H, R = kb.H, kb.R
    μ, Σ = state.μ, state.Σ

    ν  = Float64.(observation) .- H * μ           # innovation
    S  = H * Σ * H' .+ R                          # innovation covariance
    K  = Σ * H' * inv(S)                          # Kalman gain
    μ  = μ .+ K * ν
    Σ  = (I - K * H) * Σ

    # Log-likelihood contribution
    d  = length(observation)
    ll = -0.5 * (d*log(2π) + log(det(S) + 1e-30) + dot(ν, S \ ν))
    push!(state.log_likelihoods, ll)

    state.μ = μ
    state.Σ = Matrix(Symmetric(Σ))
end

"""
    run_kalman_filter(kb, observations; dt)

Run the Kalman-Bucy filter on a sequence of observations.
Returns (filtered_means, filtered_covs, total_log_likelihood).
"""
function run_kalman_filter(kb::KalmanBucy, observations::AbstractMatrix;
                             dt::Float64 = 1.0/252)

    d_lat = size(kb.F, 1)
    T     = size(observations, 2)

    state = KalmanBucyState(zeros(d_lat), Matrix(1.0*I(d_lat)), 0.0, Float64[])

    filtered_means = zeros(d_lat, T)
    filtered_covs  = Vector{Matrix{Float64}}(undef, T)

    for t in 1:T
        kalman_predict!(state, kb, dt)
        kalman_update!(state, kb, observations[:, t])
        filtered_means[:, t] = state.μ
        filtered_covs[t]     = copy(state.Σ)
    end

    total_ll = sum(state.log_likelihoods)

    return filtered_means, filtered_covs, total_ll
end

"""
    fit_kalman_bucy(observations; dt, n_iter)

Fit Kalman-Bucy filter parameters (F, H, Q, R) via EM algorithm.
"""
function fit_kalman_bucy(observations::AbstractMatrix;
                          dt::Float64    = 1.0/252,
                          n_iter::Int    = 50,
                          d_latent::Int  = 2)

    d_obs, T = size(observations)
    kb = KalmanBucy(d_latent, d_obs)

    for iter in 1:n_iter
        # E-step: run Kalman filter
        μs, Σs, ll = run_kalman_filter(kb, observations; dt=dt)

        # M-step: update parameters (simplified, no full smoother)
        # Update R from innovation sequence
        R_new = zeros(d_obs, d_obs)
        for t in 1:T
            r_pred = kb.H * μs[:, t]
            ν_t    = observations[:, t] .- r_pred
            R_new .+= ν_t * ν_t'
        end
        R_new ./= T
        R_new = Symmetric(max.(R_new, 1e-6 * I))

        kb = KalmanBucy(kb.F, kb.G, kb.H, kb.Q, Matrix(R_new))

        iter % 10 == 0 && @info "Kalman EM iter $iter: ll=$(round(ll,digits=2))"
    end

    return kb
end

# ─────────────────────────────────────────────────────────────────────────────
# REGIME PROBABILITY TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    smooth_regime_probs(regime_probs, window=10) → smoothed_probs

Apply a rolling average to smooth noisy particle filter output.
"""
function smooth_regime_probs(regime_probs::AbstractMatrix, window::Int=10)
    K, T = size(regime_probs)
    smoothed = similar(regime_probs)

    for t in 1:T
        t_start = max(1, t-window+1)
        smoothed[:, t] = mean(regime_probs[:, t_start:t], dims=2) |> vec
    end

    return smoothed
end

"""
    regime_transition_times(map_regimes) → Vector{Int}

Find time indices where the MAP regime changes.
"""
function regime_transition_times(map_regimes::AbstractVector{Int})
    T = length(map_regimes)
    transitions = Int[]
    for t in 2:T
        map_regimes[t] != map_regimes[t-1] && push!(transitions, t)
    end
    return transitions
end

"""
    regime_duration_stats(map_regimes, n_regimes) → Dict

Compute average and std of time spent in each regime.
"""
function regime_duration_stats(map_regimes::AbstractVector{Int}, n_regimes::Int)
    durations = [Int[] for _ in 1:n_regimes]
    current_regime = map_regimes[1]
    current_len    = 1

    for t in 2:length(map_regimes)
        if map_regimes[t] == current_regime
            current_len += 1
        else
            push!(durations[current_regime], current_len)
            current_regime = map_regimes[t]
            current_len    = 1
        end
    end
    push!(durations[current_regime], current_len)

    stats = Dict{Int, NamedTuple}()
    for k in 1:n_regimes
        d = durations[k]
        if isempty(d)
            stats[k] = (mean=0.0, std=0.0, min=0, max=0, count=0)
        else
            stats[k] = (mean=mean(d), std=std(d), min=minimum(d),
                         max=maximum(d), count=length(d))
        end
    end

    return stats
end

"""
    regime_conditional_moments(returns, map_regimes, n_regimes)

Compute per-regime return moments: mean, std, skewness, kurtosis.
"""
function regime_conditional_moments(returns::AbstractVector,
                                     map_regimes::AbstractVector{Int},
                                     n_regimes::Int)
    stats = Dict{Int, NamedTuple}()

    for k in 1:n_regimes
        idx = findall(==(k), map_regimes)
        if length(idx) < 2
            stats[k] = (mean=0.0, std=0.0, skew=0.0, kurt=0.0, count=0)
            continue
        end
        r_k = Float64.(returns[idx])
        stats[k] = (
            mean  = mean(r_k),
            std   = std(r_k),
            skew  = skewness(r_k),
            kurt  = kurtosis(r_k),
            count = length(idx)
        )
    end

    return stats
end

"""
    regime_aware_vol_forecast(returns, regime_probs, regime_sigma, h=1)

Compute regime-probability-weighted vol forecast for horizon h.

σ²_forecast(t) = Σₖ π_k(t) · σ²_k
"""
function regime_aware_vol_forecast(returns::AbstractVector,
                                    regime_probs::AbstractMatrix,
                                    regime_sigma::AbstractVector)
    K, T = size(regime_probs)
    forecast = zeros(T)

    for t in 1:T
        σ²_weighted = sum(regime_probs[k, t] * regime_sigma[k]^2 for k in 1:K)
        forecast[t] = sqrt(max(σ²_weighted, 1e-12))
    end

    return forecast
end
