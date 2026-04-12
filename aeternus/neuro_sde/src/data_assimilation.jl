"""
data_assimilation.jl — Continuous and discrete state estimation

Implements:
  1. Kalman-Bucy continuous filter (linear SDEs)
  2. Extended Kalman Filter (EKF) for nonlinear SDEs
  3. Discrete-time Kalman filter (standard)
  4. Online parameter updating via Recursive Least Squares (RLS)
  5. Forgetting factor / exponential weighting methods
  6. State-space model estimation (Expectation-Maximisation)
  7. Variational Bayes for state-space models
  8. Fixed-interval Rauch-Tung-Striebel (RTS) smoother
  9. Observation noise estimation (adaptive filtering)
 10. Innovation sequence diagnostics

References:
  - Kalman (1960) "A new approach to linear filtering and prediction"
  - Kalman & Bucy (1961) "New results in linear filtering"
  - Shumway & Stoffer (1982) — EM for state-space models
  - Ljung & Söderström (1983) — Recursive least squares
"""

using LinearAlgebra
using Statistics
using Random
using Distributions

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: STANDARD DISCRETE KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
    KalmanFilterState

State of the discrete Kalman filter at a single time step.
"""
mutable struct KalmanFilterState
    x   :: Vector{Float64}   # filtered state mean
    P   :: Matrix{Float64}   # filtered state covariance
    x_pred :: Vector{Float64}
    P_pred :: Matrix{Float64}
    K   :: Matrix{Float64}   # Kalman gain
    innov :: Vector{Float64} # innovation y - H x_pred
    S   :: Matrix{Float64}   # innovation covariance
    log_lik :: Float64       # log-likelihood contribution
end

"""
    KalmanModel

Linear Gaussian state-space model:
  x_{t+1} = F x_t + B u_t + q_t,   q_t ~ N(0, Q)
  y_t      = H x_t + D u_t + r_t,   r_t ~ N(0, R)
"""
struct KalmanModel
    F :: Matrix{Float64}   # state transition (n×n)
    H :: Matrix{Float64}   # observation matrix (m×n)
    Q :: Matrix{Float64}   # process noise (n×n)
    R :: Matrix{Float64}   # observation noise (m×m)
    B :: Union{Nothing, Matrix{Float64}}  # input matrix (n×p)
    D :: Union{Nothing, Matrix{Float64}}  # feedthrough (m×p)
end

KalmanModel(F, H, Q, R) = KalmanModel(F, H, Q, R, nothing, nothing)

"""
    kalman_filter(model, x0, P0, observations; inputs=nothing) → (states, log_lik)

Run the discrete Kalman filter.

Returns:
  - `states`  : Vector of KalmanFilterState for each time step
  - `log_lik` : total log marginal likelihood
"""
function kalman_filter(model::KalmanModel,
                       x0::AbstractVector,
                       P0::AbstractMatrix,
                       observations::AbstractMatrix;
                       inputs::Union{Nothing, AbstractMatrix} = nothing)
    n = size(model.F, 1)
    m = size(model.H, 1)
    T = size(observations, 2)

    x = copy(Float64.(x0))
    P = copy(Float64.(P0))

    states  = Vector{KalmanFilterState}()
    log_lik = 0.0

    for t in 1:T
        # Prediction
        u_t   = isnothing(inputs) ? zeros(0) : inputs[:, t]
        x_p   = model.F * x
        !isnothing(model.B) && (x_p += model.B * u_t)
        P_p   = model.F * P * model.F' + model.Q

        # Update
        y_t   = observations[:, t]
        Hx    = model.H * x_p
        !isnothing(model.D) && (Hx += model.D * u_t)
        innov = y_t - Hx
        S_mat = model.H * P_p * model.H' + model.R
        S_sym = Symmetric(S_mat)
        K     = P_p * model.H' / S_sym

        x  = x_p + K * innov
        P  = Symmetric((I(n) - K * model.H) * P_p)

        # Log-likelihood
        ll = -0.5 * (m * log(2π) + logdet(S_sym) +
                     dot(innov, S_sym \ innov))
        log_lik += ll

        push!(states, KalmanFilterState(copy(x), copy(Matrix(P)),
                                         copy(x_p), copy(P_p),
                                         copy(K), copy(innov),
                                         copy(Matrix(S_sym)), ll))
    end
    return states, log_lik
end

"""
    kalman_filtered_means(states) → Matrix{Float64}

Extract (state_dim × T) filtered mean matrix.
"""
kalman_filtered_means(states::Vector{KalmanFilterState}) =
    hcat([s.x for s in states]...)

"""
    kalman_filtered_covs(states) → Array{Float64,3}

Extract (state_dim × state_dim × T) filtered covariance array.
"""
function kalman_filtered_covs(states::Vector{KalmanFilterState})
    n = length(states[1].x)
    T = length(states)
    P = zeros(n, n, T)
    for t in 1:T
        P[:, :, t] = states[t].P
    end
    return P
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: RAUCH-TUNG-STRIEBEL (RTS) SMOOTHER
# ─────────────────────────────────────────────────────────────────────────────

"""
    SmootherResult

Output of the RTS smoother.
"""
struct SmootherResult
    smoothed_mean :: Matrix{Float64}   # (n × T)
    smoothed_cov  :: Array{Float64, 3} # (n × n × T)
    gain          :: Array{Float64, 3} # smoother gain (n × n × T)
end

"""
    rts_smoother(model, filter_states) → SmootherResult

Rauch-Tung-Striebel fixed-interval smoother.
Backward pass given forward filter states.
"""
function rts_smoother(model::KalmanModel,
                      filter_states::Vector{KalmanFilterState})
    n  = size(model.F, 1)
    T  = length(filter_states)
    xs = zeros(n, T)
    Ps = zeros(n, n, T)
    Gs = zeros(n, n, T)

    # Initialise at T
    xs[:, T]    = filter_states[T].x
    Ps[:, :, T] = filter_states[T].P

    for t in (T-1):-1:1
        fs = filter_states[t]
        # G_t = P_t F' (P_{t+1|t})^{-1}
        G  = fs.P * model.F' / Symmetric(filter_states[t+1].P_pred)
        xs[:, t]    = fs.x + G * (xs[:, t+1] - filter_states[t+1].x_pred)
        Ps[:, :, t] = Symmetric(fs.P +
                       G * (Ps[:, :, t+1] - filter_states[t+1].P_pred) * G')
        Gs[:, :, t] = G
    end
    return SmootherResult(xs, Ps, Gs)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: EXTENDED KALMAN FILTER (EKF)
# ─────────────────────────────────────────────────────────────────────────────

"""
    EKFResult

Result from running the Extended Kalman Filter.
"""
struct EKFResult
    filtered_mean :: Matrix{Float64}   # (n × T)
    filtered_cov  :: Array{Float64, 3} # (n × n × T)
    innovations   :: Matrix{Float64}   # (m × T)
    log_likelihood :: Float64
end

"""
    ekf_run(x0, P0, f, h, F_jac, H_jac, Q, R, observations) → EKFResult

Extended Kalman Filter for nonlinear state-space models.

- `f(x, t)` : state transition function
- `h(x, t)` : observation function
- `F_jac(x, t)` : Jacobian ∂f/∂x
- `H_jac(x, t)` : Jacobian ∂h/∂x
- `Q`, `R`  : process and observation noise covariances
"""
function ekf_run(x0::AbstractVector,
                 P0::AbstractMatrix,
                 f::Function,
                 h::Function,
                 F_jac::Function,
                 H_jac::Function,
                 Q::AbstractMatrix,
                 R::AbstractMatrix,
                 observations::AbstractMatrix)
    n  = length(x0)
    m  = size(observations, 1)
    T  = size(observations, 2)

    x        = copy(Float64.(x0))
    P        = copy(Float64.(P0))
    fil_mean = zeros(n, T)
    fil_cov  = zeros(n, n, T)
    innov    = zeros(m, T)
    log_lik  = 0.0

    for t in 1:T
        # ── Predict ──────────────────────────────────────────────────
        x_p  = f(x, Float64(t))
        Jf   = F_jac(x, Float64(t))
        P_p  = Jf * P * Jf' + Q
        P_p  = Symmetric(P_p)

        # ── Update ───────────────────────────────────────────────────
        y_t  = observations[:, t]
        Jh   = H_jac(x_p, Float64(t))
        S_m  = Jh * P_p * Jh' + R
        S_m  = Symmetric(S_m)
        K    = P_p * Jh' / S_m
        inn  = y_t - h(x_p, Float64(t))

        x    = x_p + K * inn
        P    = Symmetric((I(n) - K * Jh) * P_p)

        fil_mean[:, t]   = x
        fil_cov[:, :, t] = Matrix(P)
        innov[:, t]      = inn

        log_lik += -0.5 * (m * log(2π) + logdet(S_m) + dot(inn, S_m \ inn))
    end
    return EKFResult(fil_mean, fil_cov, innov, log_lik)
end

"""
    fd_jacobian(f, x, t; ε=1e-5) → Matrix{Float64}

Compute Jacobian of f at (x,t) via central finite differences.
"""
function fd_jacobian(f::Function, x::AbstractVector, t::Real; ε::Real=1e-5)
    y0 = f(x, t)
    n  = length(x)
    m  = length(y0)
    J  = zeros(m, n)
    for j in 1:n
        xp = copy(x); xp[j] += ε
        xm = copy(x); xm[j] -= ε
        J[:, j] = (f(xp, t) - f(xm, t)) / (2ε)
    end
    return J
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: KALMAN-BUCY CONTINUOUS FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
    KalmanBucyModel

Continuous-time linear state-space model:
  dX_t = A X_t dt + B dW_t
  dY_t = C X_t dt + D dV_t

where W, V are independent standard Brownian motions.
"""
struct KalmanBucyModel
    A :: Matrix{Float64}   # n×n drift
    B :: Matrix{Float64}   # n×p diffusion
    C :: Matrix{Float64}   # m×n observation
    D :: Matrix{Float64}   # m×q observation noise
end

"""
    kalman_bucy_filter(model, x0, P0, dt, observations) → (means, covs, log_lik)

Discrete approximation to the Kalman-Bucy continuous filter via
first-order (Euler) time-discretisation.

The continuous Riccati equation dP/dt = AP + PAᵀ + BBᵀ - PC'(DDᵀ)⁻¹CP
is integrated alongside the filter equations.

Returns:
  - `means` : (n × T) filtered means
  - `covs`  : (n × n × T) filtered covariances
  - `log_lik` : total log-likelihood
"""
function kalman_bucy_filter(model::KalmanBucyModel,
                             x0::AbstractVector,
                             P0::AbstractMatrix,
                             dt::Real,
                             observations::AbstractMatrix)
    n  = size(model.A, 1)
    m  = size(model.C, 1)
    T  = size(observations, 2)

    x   = copy(Float64.(x0))
    P   = copy(Float64.(P0))
    Q   = model.B * model.B'    # n×n
    R   = model.D * model.D'    # m×m

    means   = zeros(n, T)
    covs    = zeros(n, n, T)
    log_lik = 0.0

    for t in 1:T
        # Riccati ODE step: dP/dt = AP + PAᵀ + Q - PC'R⁻¹CP
        R_sym = Symmetric(R)
        dP    = (model.A * P + P * model.A' + Q -
                 P * model.C' * (R_sym \ (model.C * P)))
        P     = Symmetric(P + dP * dt)

        # Kalman gain: K = P C' R⁻¹
        K   = P * model.C' / R_sym

        # State ODE + innovation injection
        y_t   = observations[:, t]
        innov  = y_t - model.C * x
        x     = x + (model.A * x + K * innov) * dt

        # Log-likelihood contribution (discrete approximation)
        S_mat = Symmetric(model.C * P * model.C' + R / dt)
        log_lik += -0.5 * (m * log(2π) + logdet(S_mat) +
                            dot(innov, S_mat \ innov))

        means[:, t]    = x
        covs[:, :, t]  = Matrix(P)
    end
    return means, covs, log_lik
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: RECURSIVE LEAST SQUARES (RLS)
# ─────────────────────────────────────────────────────────────────────────────

"""
    RLSState

State of the recursive least squares estimator.
"""
mutable struct RLSState
    θ   :: Vector{Float64}   # parameter estimate
    P   :: Matrix{Float64}   # covariance matrix
    t   :: Int               # time step
    λ   :: Float64           # forgetting factor
    residuals :: Vector{Float64}
end

"""
    RLSState(n_params; λ=1.0, P_init=1e6) → RLSState

Initialise RLS state.
"""
RLSState(n_params::Int; λ::Real=1.0, P_init::Real=1e6) =
    RLSState(zeros(n_params), P_init * I(n_params) |> Matrix,
             0, Float64(λ), Float64[])

"""
    rls_update!(state::RLSState, φ, y) → (θ, gain)

Single RLS update:
  Gain:      K = P φ / (λ + φᵀ P φ)
  Update:    θ = θ + K (y - φᵀ θ)
  Covariance: P = (P - K φᵀ P) / λ

Arguments:
  - `φ` : regressor vector (feature vector at time t)
  - `y` : scalar observation
"""
function rls_update!(state::RLSState, φ::AbstractVector, y::Real)
    Pφ   = state.P * φ
    denom = state.λ + dot(φ, Pφ)
    K     = Pφ ./ denom
    innov = y - dot(φ, state.θ)
    state.θ .+= K .* innov
    state.P  = (state.P .- K * Pφ') ./ state.λ
    state.t  += 1
    push!(state.residuals, innov)
    return state.θ, K
end

"""
    rls_batch(φ_mat, y_vec; λ=1.0, P_init=1e6) → (θ_history, state)

Run RLS on a batch of observations.

- `φ_mat` : (n_params × T) regressor matrix
- `y_vec` : T observations
"""
function rls_batch(φ_mat::AbstractMatrix,
                   y_vec::AbstractVector;
                   λ::Real     = 1.0,
                   P_init::Real = 1e6)
    n_p  = size(φ_mat, 1)
    T    = size(φ_mat, 2)
    state = RLSState(n_p; λ=λ, P_init=P_init)
    θ_hist = zeros(n_p, T)
    for t in 1:T
        rls_update!(state, φ_mat[:, t], y_vec[t])
        θ_hist[:, t] = state.θ
    end
    return θ_hist, state
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FORGETTING FACTOR METHODS
# ─────────────────────────────────────────────────────────────────────────────

"""
    ForgettingFactorConfig

Configuration for forgetting-factor adaptive estimation.
"""
struct ForgettingFactorConfig
    λ_fixed     :: Float64   # fixed forgetting factor (0 < λ ≤ 1)
    adapt_λ     :: Bool      # whether to adapt λ
    λ_min       :: Float64
    λ_max       :: Float64
    target_cv   :: Float64   # target coefficient of variation for adaptation
end

ForgettingFactorConfig(; λ_fixed=0.98, adapt_λ=false,
                         λ_min=0.9, λ_max=1.0, target_cv=0.1) =
    ForgettingFactorConfig(λ_fixed, adapt_λ, λ_min, λ_max, target_cv)

"""
    variable_forgetting_rls(φ_mat, y_vec, cfg::ForgettingFactorConfig) → θ_history

RLS with variable (adaptive) forgetting factor.
λ is adapted to maintain a target coefficient of variation of the
residuals.
"""
function variable_forgetting_rls(φ_mat::AbstractMatrix,
                                  y_vec::AbstractVector,
                                  cfg::ForgettingFactorConfig)
    n_p   = size(φ_mat, 1)
    T     = size(φ_mat, 2)
    state = RLSState(n_p; λ=cfg.λ_fixed)
    θ_hist = zeros(n_p, T)

    resid_window = Float64[]
    window_size  = 50

    for t in 1:T
        rls_update!(state, φ_mat[:, t], y_vec[t])

        if cfg.adapt_λ && t > window_size
            recent = state.residuals[max(1,end-window_size+1):end]
            cv_now = std(recent) / (abs(mean(recent)) + 1e-12)
            if cv_now > cfg.target_cv
                state.λ = max(state.λ * 0.995, cfg.λ_min)
            else
                state.λ = min(state.λ * 1.001, cfg.λ_max)
            end
        end
        θ_hist[:, t] = state.θ
    end
    return θ_hist
end

"""
    exponential_smoothing(x, α; initial=nothing) → Vector{Float64}

Simple exponential smoothing: S_t = α x_t + (1-α) S_{t-1}.
α ∈ (0,1]: higher = more weight on recent obs.
"""
function exponential_smoothing(x::AbstractVector, α::Real;
                                initial::Union{Nothing,Real} = nothing)
    n   = length(x)
    S   = zeros(n)
    S[1] = isnothing(initial) ? x[1] : Float64(initial)
    for t in 2:n
        S[t] = α * x[t] + (1 - α) * S[t-1]
    end
    return S
end

"""
    holt_winters_es(x, α, β; trend0=nothing) → Vector{Float64}

Holt-Winters double exponential smoothing (trend correction).
"""
function holt_winters_es(x::AbstractVector, α::Real, β::Real;
                          trend0::Union{Nothing,Real} = nothing)
    n   = length(x)
    S   = zeros(n)
    b   = zeros(n)
    S[1] = x[1]
    b[1] = isnothing(trend0) ? (n > 1 ? x[2] - x[1] : 0.0) : Float64(trend0)
    for t in 2:n
        S[t] = α * x[t] + (1 - α) * (S[t-1] + b[t-1])
        b[t] = β * (S[t] - S[t-1]) + (1 - β) * b[t-1]
    end
    return S
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: STATE-SPACE EM ALGORITHM (SHUMWAY-STOFFER)
# ─────────────────────────────────────────────────────────────────────────────

"""
    SSMParams

Parameters of a linear Gaussian state-space model for EM estimation.
"""
mutable struct SSMParams
    F :: Matrix{Float64}
    H :: Matrix{Float64}
    Q :: Matrix{Float64}
    R :: Matrix{Float64}
    μ0 :: Vector{Float64}
    Σ0 :: Matrix{Float64}
end

"""
    em_ssm(observations, n_states; n_iters=100, tol=1e-6) → (params, log_lik_hist)

Expectation-Maximisation for linear Gaussian state-space model.

Estimates {F, H, Q, R, μ₀, Σ₀} by alternating:
  E-step: Kalman filter + RTS smoother
  M-step: closed-form parameter updates

Reference: Shumway & Stoffer (1982).
"""
function em_ssm(observations::AbstractMatrix,
                n_states::Int;
                n_iters::Int   = 100,
                tol::Real      = 1e-6,
                seed::Int      = 42)
    m, T = size(observations)
    n    = n_states
    rng  = MersenneTwister(seed)

    # Random initialisation
    p = SSMParams(
        0.95 * I(n) |> Matrix,
        randn(rng, m, n) .* 0.1,
        0.1 * I(n) |> Matrix,
        0.1 * I(m) |> Matrix,
        zeros(n),
        I(n) |> Matrix
    )

    log_lik_hist = Float64[]
    prev_ll      = -Inf

    for iter in 1:n_iters
        # ── E-step: Kalman filter + RTS smoother ──────────────────────
        model   = KalmanModel(p.F, p.H, p.Q, p.R)
        states, ll = kalman_filter(model, p.μ0, p.Σ0, observations)
        smooth  = rts_smoother(model, states)

        push!(log_lik_hist, ll)

        # Convergence check
        if abs(ll - prev_ll) < tol
            break
        end
        prev_ll = ll

        # Sufficient statistics
        xs   = smooth.smoothed_mean          # n × T
        Ps   = smooth.smoothed_cov           # n × n × T

        # Cross-covariance Σ_{t, t-1|T} for M-step
        # P_{t, t-1 | T} = G_{t-1} Σ_{t|T}
        cross_cov = zeros(n, n, T)
        for t in 2:T
            cross_cov[:, :, t] = smooth.gain[:, :, t-1] * Ps[:, :, t]
        end

        # Compute sums for M-step
        Σ11 = sum(Ps[:, :, t] + xs[:, t] * xs[:, t]' for t in 1:T)
        Σ10 = sum(cross_cov[:, :, t] + xs[:, t] * xs[:, t-1]' for t in 2:T)
        Σ00 = sum(Ps[:, :, t] + xs[:, t] * xs[:, t]' for t in 1:(T-1))

        # ── M-step ─────────────────────────────────────────────────────
        # F: (T-1) × Σ10 Σ00⁻¹
        p.F  = Σ10 / max.(Σ00, 1e-10 * I(n))

        # Q
        p.Q  = Symmetric((Σ11 - p.F * Σ10' - Σ10 * p.F' + p.F * Σ00 * p.F') / (T - 1))
        p.Q  = max.(Matrix(p.Q), 1e-8 * I(n))

        # H: Σ_yx Σ_xx⁻¹
        Σyx  = sum(observations[:, t] * xs[:, t]' for t in 1:T)
        p.H  = Σyx / max.(Σ11, 1e-10 * I(n))

        # R
        Σyy  = sum(observations[:, t] * observations[:, t]' for t in 1:T) ./ T
        p.R  = Symmetric(Σyy - p.H * Σyx' / T)
        p.R  = max.(Matrix(p.R), 1e-8 * I(m))

        # Initial conditions
        p.μ0 = xs[:, 1]
        p.Σ0 = Symmetric(Ps[:, :, 1])
    end

    return p, log_lik_hist
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: INNOVATION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    InnovationDiagnostics

Tests on the innovation sequence to assess filter validity.
"""
struct InnovationDiagnostics
    autocorr      :: Vector{Float64}   # lag-1 to lag-20
    mean_innov    :: Vector{Float64}   # per observation dim
    std_innov     :: Vector{Float64}
    normalised_nees :: Vector{Float64} # Normalised Estimation Error Squared
    ljung_box_stat :: Float64
    ljung_box_pval :: Float64
    is_white_noise :: Bool
end

"""
    diagnose_innovations(innovations, S_matrices; lags=20, α=0.05)

Test whether innovations form a white noise sequence (necessary condition
for optimal filter).

- `innovations` : (m × T) matrix
- `S_matrices`  : (m × m × T) innovation covariance matrices
"""
function diagnose_innovations(innovations::AbstractMatrix,
                               S_matrices::AbstractArray;
                               lags::Int  = 20,
                               α::Real    = 0.05)
    m, T = size(innovations)

    # Mean and std per dim
    μ_inn = [mean(innovations[i, :]) for i in 1:m]
    σ_inn = [std(innovations[i, :]) for i in 1:m]

    # NEES: normalised innovation squared
    nees = zeros(T)
    for t in 1:T
        inn  = innovations[:, t]
        S_t  = Symmetric(S_matrices[:, :, t])
        try
            nees[t] = dot(inn, S_t \ inn) / m
        catch
            nees[t] = NaN
        end
    end

    # Autocorrelation of first dimension innovations (proxy)
    r1  = innovations[1, :]
    n   = length(r1)
    r1c = r1 .- mean(r1)
    var_r1 = var(r1c)

    ac = zeros(lags)
    for lag in 1:lags
        ac[lag] = sum(r1c[1:n-lag] .* r1c[lag+1:n]) / (n * var_r1)
    end

    # Ljung-Box test statistic
    Q_lb = n * (n + 2) * sum(ac[k]^2 / (n - k) for k in 1:lags)
    # Chi-squared critical value (approximate)
    # p-value via chi-squared distribution with `lags` dof
    lbpval = 1.0 - cdf(Chisq(lags), Q_lb)
    white_noise = lbpval > α

    return InnovationDiagnostics(ac, μ_inn, σ_inn, nees,
                                  Q_lb, lbpval, white_noise)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: ADAPTIVE OBSERVATION NOISE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    AdaptiveNoiseKF

Kalman filter with adaptive observation noise estimation
via Innovation covariance matching (Mehra 1972).
"""
mutable struct AdaptiveNoiseKF
    base_model :: KalmanModel
    R_est      :: Matrix{Float64}   # current R estimate
    window     :: Int               # window for covariance estimation
    innov_hist :: Matrix{Float64}   # (m × window) innovation history
    t          :: Int
end

function AdaptiveNoiseKF(model::KalmanModel; window::Int=50)
    m = size(model.H, 1)
    AdaptiveNoiseKF(model, copy(model.R), window,
                    zeros(m, window), 0)
end

"""
    adaptive_kf_step!(akf, x, P, y_t) → (x_new, P_new, innov)

Single step of adaptive Kalman filter.
Updates R estimate based on empirical innovation covariance.
"""
function adaptive_kf_step!(akf::AdaptiveNoiseKF,
                             x::AbstractVector,
                             P::AbstractMatrix,
                             y_t::AbstractVector)
    model = akf.base_model
    n = length(x)
    m = length(y_t)

    # Predict
    x_p  = model.F * x
    P_p  = model.F * P * model.F' + model.Q

    # Update with current R estimate
    H    = model.H
    S_m  = Symmetric(H * P_p * H' + akf.R_est)
    K    = P_p * H' / S_m
    innov = y_t - H * x_p

    x_new = x_p + K * innov
    P_new = Symmetric((I(n) - K * H) * P_p)

    # Store innovation
    akf.t += 1
    t_idx = mod1(akf.t, akf.window)
    akf.innov_hist[:, t_idx] = innov

    # Update R estimate every `window` steps
    if akf.t >= akf.window && mod(akf.t, 10) == 0
        ε_mat  = akf.innov_hist
        C_inn  = ε_mat * ε_mat' ./ akf.window
        # Mehra: R = C_inn - H P_p Hᵀ  (clamp to PD)
        R_new  = C_inn - H * P_p * H'
        # Ensure positive definite
        R_new  = Symmetric(R_new)
        λ_min  = minimum(eigvals(R_new))
        if λ_min < 1e-6
            R_new += (abs(λ_min) + 1e-6) * I(m)
        end
        akf.R_est = Matrix(R_new)
    end

    return x_new, Matrix(P_new), innov
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: ONLINE PARAMETER UPDATING FOR SDE MODELS
# ─────────────────────────────────────────────────────────────────────────────

"""
    OnlineSDEEstimator

Online estimator for SDE drift and diffusion parameters via
locally-linear approximation + RLS.

Model: dX_t ≈ (θ₁ + θ₂ X_t) dt + σ dW_t
(Ornstein-Uhlenbeck type, estimated online)
"""
mutable struct OnlineSDEEstimator
    rls_state :: RLSState   # for drift parameters [θ₁, θ₂]
    σ_ema     :: Float64    # EWMA estimate of diffusion
    λ_ema     :: Float64    # EWMA decay for σ
    dt        :: Float64
    n_obs     :: Int
    x_prev    :: Float64
end

function OnlineSDEEstimator(dt::Real; λ_rls::Real=0.99, λ_ema::Real=0.95,
                              P_init::Real=1e4)
    OnlineSDEEstimator(RLSState(2; λ=λ_rls, P_init=P_init),
                        0.1, λ_ema, dt, 0, NaN)
end

"""
    update!(est::OnlineSDEEstimator, x_t) → (θ_drift, σ_est)

Update online SDE parameter estimates given new observation x_t.
"""
function update!(est::OnlineSDEEstimator, x_t::Real)
    if isnan(est.x_prev)
        est.x_prev = x_t
        est.n_obs  += 1
        return zeros(2), est.σ_ema
    end

    Δx = x_t - est.x_prev
    # Regressor: [dt, X_{t-1} * dt]
    φ  = [est.dt, est.x_prev * est.dt]

    # RLS for drift: Δx = φᵀ [θ₁, θ₂] + noise
    θ, _ = rls_update!(est.rls_state, φ, Δx)

    # EWMA for diffusion: σ² ≈ EWMA of (Δx - drift)² / dt
    drift_hat  = dot(φ, θ)
    residual   = (Δx - drift_hat)^2 / est.dt
    est.σ_ema  = sqrt(est.λ_ema * est.σ_ema^2 + (1 - est.λ_ema) * residual)

    est.x_prev = x_t
    est.n_obs  += 1
    return θ, est.σ_ema
end

"""
    fit_online_sde(prices; dt=1/252, λ_rls=0.99, λ_ema=0.95)

Fit OU-type SDE parameters online to a price series.
Returns parameter history as a matrix.
"""
function fit_online_sde(prices::AbstractVector;
                        dt::Real   = 1/252,
                        λ_rls::Real = 0.99,
                        λ_ema::Real = 0.95)
    n   = length(prices)
    est = OnlineSDEEstimator(dt; λ_rls=λ_rls, λ_ema=λ_ema)
    θ_hist = zeros(2, n)
    σ_hist = zeros(n)
    for t in 1:n
        θ, σ = update!(est, log(prices[t]))
        θ_hist[:, t] = θ
        σ_hist[t]    = σ
    end
    return θ_hist, σ_hist
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: DEMO
# ─────────────────────────────────────────────────────────────────────────────

"""
    demo_kalman_filter(; n=200, seed=1)

Demo: estimate OU process with standard Kalman filter.
"""
function demo_kalman_filter(; n::Int=200, seed::Int=1)
    rng = MersenneTwister(seed)
    # Simulate OU process: X_{t+1} = 0.95 X_t + 0.1 ε_t
    X = zeros(n+1)
    for t in 1:n
        X[t+1] = 0.95 * X[t] + 0.1 * randn(rng)
    end
    # Noisy observations
    Y = X[2:end]' .+ 0.05 * randn(rng, 1, n)

    F = reshape([0.95], 1, 1)
    H = reshape([1.0],  1, 1)
    Q = reshape([0.01], 1, 1)
    R = reshape([0.0025], 1, 1)

    model = KalmanModel(F, H, Q, R)
    states, ll = kalman_filter(model, [0.0], [1.0], Y)
    fil_means  = kalman_filtered_means(states)
    rmse = sqrt(mean((fil_means[1, :] .- X[2:end]).^2))
    return (states=states, filtered_mean=fil_means[1,:],
            true_x=X[2:end], rmse=rmse, log_lik=ll)
end

"""
    demo_online_sde(; n=1000, seed=2)

Demo: online estimation of OU parameters.
"""
function demo_online_sde(; n::Int=1000, seed::Int=2)
    rng  = MersenneTwister(seed)
    # True OU: dX = κ(μ-X) dt + σ dW  with κ=2, μ=1, σ=0.3, dt=1/252
    κ, μ, σ, dt = 2.0, 1.0, 0.3, 1/252
    X  = zeros(n)
    X[1] = 1.0
    for t in 2:n
        X[t] = X[t-1] + κ * (μ - X[t-1]) * dt + σ * sqrt(dt) * randn(rng)
    end
    θ_hist, σ_hist = fit_online_sde(exp.(X); dt=dt)
    return (θ_history=θ_hist, sigma_history=σ_hist, true_x=X)
end

"""
    demo_em_ssm(; m=2, n=3, T=300, seed=5)

Demo: EM estimation of a state-space model.
"""
function demo_em_ssm(; m::Int=2, n_states::Int=3, T::Int=300, seed::Int=5)
    rng = MersenneTwister(seed)
    # True model
    F_true = 0.9 * I(n_states) |> Matrix
    H_true = randn(rng, m, n_states)
    Q_true = 0.1 * I(n_states) |> Matrix
    R_true = 0.2 * I(m) |> Matrix

    # Simulate
    X  = zeros(n_states, T+1)
    Y  = zeros(m, T)
    for t in 1:T
        X[:, t+1] = F_true * X[:, t] + 0.316 * randn(rng, n_states)
        Y[:, t]   = H_true * X[:, t+1] + 0.447 * randn(rng, m)
    end

    params, ll_hist = em_ssm(Y, n_states; n_iters=50)
    return (params=params, log_lik_history=ll_hist,
            true_F=F_true, true_H=H_true)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: VARIATIONAL BAYES FOR STATE-SPACE MODELS
# ─────────────────────────────────────────────────────────────────────────────

"""
    VBSSMResult

Result of Variational Bayes state-space model estimation.
"""
struct VBSSMResult
    smoothed_mean :: Matrix{Float64}
    smoothed_cov  :: Array{Float64, 3}
    F_post_mean   :: Matrix{Float64}
    H_post_mean   :: Matrix{Float64}
    Q_post_mean   :: Matrix{Float64}
    R_post_mean   :: Matrix{Float64}
    free_energy   :: Vector{Float64}
end

"""
    vb_ssm(observations, n_states; n_iters=50) → VBSSMResult

Simplified Variational Bayes EM for linear Gaussian SSM.
Alternates between:
  - VB-E step: compute variational posterior over states (Kalman smoother)
  - VB-M step: update hyperparameters of matrix-Normal-Wishart priors
"""
function vb_ssm(observations::AbstractMatrix,
                n_states::Int;
                n_iters::Int = 50,
                seed::Int    = 42)
    m, T = size(observations)
    n    = n_states
    rng  = MersenneTwister(seed)

    # Initialise with EM result as warm start (simplified)
    F = 0.9 * I(n) |> Matrix
    H = randn(rng, m, n) .* 0.1
    Q = 0.1 * I(n) |> Matrix
    R = 0.1 * I(m) |> Matrix
    μ0 = zeros(n)
    Σ0 = I(n) |> Matrix

    # Prior hyperparameters (conjugate priors)
    ν_Q = Float64(n + 2)   # Wishart dof for Q
    Ψ_Q = I(n) |> Matrix
    ν_R = Float64(m + 2)
    Ψ_R = I(m) |> Matrix

    free_energy_hist = Float64[]
    xs_mean = zeros(n, T)
    xs_cov  = zeros(n, n, T)

    for iter in 1:n_iters
        # VB-E step: Kalman smoother
        model   = KalmanModel(F, H, Q, R)
        states, ll = kalman_filter(model, μ0, Σ0, observations)
        smooth  = rts_smoother(model, states)

        xs_mean = smooth.smoothed_mean
        xs_cov  = smooth.smoothed_cov

        # Sufficient statistics
        Σ11 = sum(xs_cov[:, :, t] + xs_mean[:, t]*xs_mean[:, t]' for t in 1:T)
        Σ10 = zeros(n, n)
        for t in 2:T
            G = smooth.gain[:, :, t-1]
            Σ10 += G * xs_cov[:, :, t] + xs_mean[:, t]*xs_mean[:, t-1]'
        end
        Σ00 = sum(xs_cov[:, :, t] + xs_mean[:, t]*xs_mean[:, t]' for t in 1:(T-1))

        # VB-M step
        F = Σ10 / max.(Σ00, 1e-10 * I(n))

        Q_s = (Σ11 - F*Σ10' - Σ10*F' + F*Σ00*F') / (T-1)
        Q   = Hermitian(Q_s + Ψ_Q / (ν_Q + n + 1))
        Q   = max.(Matrix(Q), 1e-8 * I(n))

        Σyx = sum(observations[:, t] * xs_mean[:, t]' for t in 1:T)
        H   = Σyx / max.(Σ11, 1e-10 * I(n))

        Σyy = sum(observations[:, t] * observations[:, t]' for t in 1:T) ./ T
        R_s = Σyy - H * (Σyx / T)'
        R   = Hermitian(R_s + Ψ_R / (ν_R + m + 1))
        R   = max.(Matrix(R), 1e-8 * I(m))

        μ0 = xs_mean[:, 1]
        Σ0 = xs_cov[:, :, 1]

        # Free energy (lower bound on log evidence)
        push!(free_energy_hist, ll - 0.5*(T-1)*logdet(Symmetric(Q)) - 0.5*T*logdet(Symmetric(R)))
    end

    return VBSSMResult(xs_mean, xs_cov, F, H, Q, R, free_energy_hist)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: FIXED-LAG SMOOTHER
# ─────────────────────────────────────────────────────────────────────────────

"""
    FixedLagSmootherResult

Result of fixed-lag Kalman smoother (real-time smoother with lag L).
"""
struct FixedLagSmootherResult
    smoothed_mean :: Matrix{Float64}   # (n × T)
    smoothed_cov  :: Array{Float64, 3} # (n × n × T)
    lag           :: Int
end

"""
    fixed_lag_smoother(model, x0, P0, observations, L) → FixedLagSmootherResult

Fixed-lag smoother with lag L.
At time t, uses observations up to t+L for the estimate at time t.
Suitable for online smoothing with bounded delay.
"""
function fixed_lag_smoother(model::KalmanModel,
                             x0::AbstractVector,
                             P0::AbstractMatrix,
                             observations::AbstractMatrix,
                             L::Int)
    n = size(model.F, 1)
    T = size(observations, 2)
    sm_mean = zeros(n, T)
    sm_cov  = zeros(n, n, T)

    for t in 1:T
        # Run filter from start to min(t+L, T)
        t_end = min(t + L, T)
        obs_sub = observations[:, 1:t_end]
        states_sub, _ = kalman_filter(model, x0, P0, obs_sub)

        if t_end > t
            # Smooth back to time t
            smooth_sub = rts_smoother(model, states_sub)
            sm_mean[:, t]    = smooth_sub.smoothed_mean[:, t]
            sm_cov[:, :, t]  = smooth_sub.smoothed_cov[:, :, t]
        else
            sm_mean[:, t]   = states_sub[end].x
            sm_cov[:, :, t] = states_sub[end].P
        end
    end
    return FixedLagSmootherResult(sm_mean, sm_cov, L)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14: CONTINUOUS-DISCRETE KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
    ContinuousDiscreteKF

Kalman filter for continuous state / discrete observation model:
  dX_t = A X_t dt + L dW_t
  y_k  = H X_{t_k} + r_k,   r_k ~ N(0, R)

Uses matrix exponential for exact discretisation.
"""
struct ContinuousDiscreteKF
    A :: Matrix{Float64}
    L :: Matrix{Float64}   # diffusion coefficient
    H :: Matrix{Float64}
    R :: Matrix{Float64}
end

"""
    discretise_system(cdkf::ContinuousDiscreteKF, dt) → KalmanModel

Discretise a continuous-time SDE using the Van Loan method for Q.
"""
function discretise_system(cdkf::ContinuousDiscreteKF, dt::Real)
    n = size(cdkf.A, 1)
    # F = expm(A dt) ≈ I + A dt + (A dt)² / 2 ...
    # Simplified: use Padé approximation order 3
    A_dt = cdkf.A .* dt
    F    = I(n) + A_dt + 0.5 .* A_dt^2 + (1/6) .* A_dt^3

    # Q via Van Loan: integral of exp(A s) L L' exp(A' s) ds from 0 to dt
    # Simplified: first-order Q = L L' dt
    Q    = cdkf.L * cdkf.L' .* dt

    return KalmanModel(Matrix(F), cdkf.H, Q, cdkf.R)
end

"""
    run_cdkf(cdkf, x0, P0, observations, dt_vec) → (states, log_lik)

Run continuous-discrete Kalman filter with possibly non-uniform time steps.
"""
function run_cdkf(cdkf::ContinuousDiscreteKF,
                   x0::AbstractVector,
                   P0::AbstractMatrix,
                   observations::AbstractMatrix,
                   dt_vec::AbstractVector)
    T    = size(observations, 2)
    n    = length(x0)
    x    = copy(Float64.(x0))
    P    = copy(Float64.(P0))
    states  = Vector{KalmanFilterState}()
    log_lik = 0.0
    m = size(cdkf.H, 1)

    for t in 1:T
        dt    = dt_vec[min(t, length(dt_vec))]
        model = discretise_system(cdkf, dt)

        # Predict
        x_p  = model.F * x
        P_p  = model.F * P * model.F' + model.Q

        # Update
        y_t  = observations[:, t]
        S_m  = Symmetric(cdkf.H * P_p * cdkf.H' + cdkf.R)
        K    = P_p * cdkf.H' / S_m
        innov = y_t - cdkf.H * x_p

        x   = x_p + K * innov
        P   = Symmetric((I(n) - K * cdkf.H) * P_p)
        ll  = -0.5*(m*log(2π) + logdet(S_m) + dot(innov, S_m \ innov))
        log_lik += ll

        push!(states, KalmanFilterState(copy(x), copy(Matrix(P)),
                                         copy(x_p), copy(P_p),
                                         copy(K), copy(innov),
                                         copy(Matrix(S_m)), ll))
    end
    return states, log_lik
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15: STOCHASTIC VOLATILITY STATE-SPACE
# ─────────────────────────────────────────────────────────────────────────────

"""
    SVStateSpaceModel

Stochastic volatility model in state-space form:
  h_{t+1} = μ + φ (h_t - μ) + η_t,   η_t ~ N(0, σ²)
  y_t = exp(h_t / 2) ε_t,             ε_t ~ N(0, 1)

where h_t = log variance at time t.
"""
struct SVStateSpaceModel
    μ  :: Float64   # long-run log-variance
    φ  :: Float64   # persistence
    σ  :: Float64   # vol-of-vol
end

"""
    sv_filter_linearised(returns, p::SVStateSpaceModel; seed=42) → (h_filtered, log_lik)

Approximate linear Gaussian filter for SV model via log |y_t|² transformation.
log y_t² = h_t + log ε_t²
log ε_t² ≈ -1.2704 + ε̃_t where ε̃_t ~ N(0, π²/2) (Kim, Shephard & Chib 1998)
"""
function sv_filter_linearised(returns::AbstractVector,
                               p::SVStateSpaceModel;
                               seed::Int = 42)
    n   = length(returns)
    # Transform: z_t = log(y_t² + 1e-8)
    z   = log.(returns.^2 .+ 1e-8)
    # Observation equation: z_t = h_t + c + ε_t, ε_t ~ N(0, π²/2)
    c_offset = -1.2704
    R_obs    = π^2 / 2

    # State equation: h_{t+1} = μ(1-φ) + φ h_t + σ η_t
    F_val = p.φ
    Q_val = p.σ^2
    μc    = p.μ * (1 - p.φ)

    # Run Kalman filter on transformed observations
    x   = p.μ
    P   = p.σ^2 / (1 - p.φ^2)
    log_lik = 0.0
    h_filt  = zeros(n)

    for t in 1:n
        # Predict
        x_p = μc + F_val * x
        P_p = F_val * P * F_val + Q_val

        # Update: z_t - c_offset = h_t + ε_t
        y_t  = z[t] - c_offset
        S    = P_p + R_obs
        K    = P_p / S
        innov = y_t - x_p

        x   = x_p + K * innov
        P   = (1 - K) * P_p

        h_filt[t] = x
        log_lik  += -0.5 * (log(2π) + log(S) + innov^2 / S)
    end
    return h_filt, log_lik
end

"""
    sv_vol_estimate(returns, p::SVStateSpaceModel; seed=42) → Vector{Float64}

Estimate instantaneous volatility σ_t = exp(h_t/2) from SV model.
"""
function sv_vol_estimate(returns::AbstractVector,
                          p::SVStateSpaceModel;
                          seed::Int = 42)
    h_filt, _ = sv_filter_linearised(returns, p; seed=seed)
    return exp.(h_filt ./ 2)
end
