"""
kalman_enhanced.jl

Enhanced Kalman filtering suite for financial state estimation.

Kalman filter variants:
  1. Standard Kalman Filter (KF)       — linear Gaussian systems
  2. Extended Kalman Filter (EKF)      — nonlinear via Jacobian linearisation
  3. Unscented Kalman Filter (UKF)     — nonlinear via sigma-point propagation
  4. Ensemble Kalman Filter (EnKF)     — Monte Carlo, high-dimensional state

State-space model (linear):
    xₜ = Fxₜ₋₁ + Buₜ + wₜ,    wₜ ~ N(0, Q)   [state transition]
    yₜ = Hxₜ + vₜ,              vₜ ~ N(0, R)   [observation]

Applications:
  - Volatility regime extraction
  - Trend/noise decomposition for alpha signals
  - Pairs-trading spread estimation (mean-reverting spread)
  - EM algorithm for Q/R parameter estimation

References:
  Kalman (1960) "A New Approach to Linear Filtering and Prediction Problems"
  Julier & Uhlmann (1997) "A New Extension of the Kalman Filter to Nonlinear Systems"
  Evensen (1994) "Sequential data assimilation with a nonlinear quasi-geostrophic model"
  Shumway & Stoffer (2000) "Time Series Analysis and Its Applications"
"""

using LinearAlgebra
using Statistics
using Random
using Distributions
using Optim
using Plots

# ─────────────────────────────────────────────────────────────────────────────
# TYPE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
Linear state-space model parameters.
"""
struct LinearStateSpace
    F::Matrix{Float64}   # state transition matrix (n×n)
    H::Matrix{Float64}   # observation matrix (m×n)
    Q::Matrix{Float64}   # process noise covariance (n×n)
    R::Matrix{Float64}   # observation noise covariance (m×m)
    B::Matrix{Float64}   # control input matrix (n×k), optional
    x0::Vector{Float64}  # initial state
    P0::Matrix{Float64}  # initial covariance
end

"""
Convenience constructor with no control input.
"""
function LinearStateSpace(F, H, Q, R, x0, P0)
    n = size(F, 1)
    LinearStateSpace(F, H, Q, R, zeros(n, 0), x0, P0)
end

"""
Kalman filter output: filtered state estimates and covariances.
"""
struct KalmanOutput
    x_filtered::Matrix{Float64}   # T×n filtered state means
    P_filtered::Array{Float64,3}  # T×n×n filtered covariances
    x_predicted::Matrix{Float64}  # T×n one-step-ahead predicted means
    P_predicted::Array{Float64,3} # T×n×n predicted covariances
    innovations::Matrix{Float64}  # T×m innovation sequence
    S::Array{Float64,3}           # T×m×m innovation covariances
    log_likelihood::Float64
    K::Array{Float64,3}           # T×n×m Kalman gains
end

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
Run the standard Kalman filter forward pass.

Predict step:
    x̂ₜ|ₜ₋₁ = F·x̂ₜ₋₁|ₜ₋₁
    Pₜ|ₜ₋₁  = F·Pₜ₋₁|ₜ₋₁·Fᵀ + Q

Update step:
    yₜ innovation: ẑₜ = yₜ - H·x̂ₜ|ₜ₋₁
    Sₜ = H·Pₜ|ₜ₋₁·Hᵀ + R
    Kₜ = Pₜ|ₜ₋₁·Hᵀ·Sₜ⁻¹           (Kalman gain)
    x̂ₜ|ₜ = x̂ₜ|ₜ₋₁ + Kₜ·ẑₜ
    Pₜ|ₜ = (I - Kₜ·H)·Pₜ|ₜ₋₁      (Joseph form for numerical stability)

Log-likelihood:
    log p(Y) = Σₜ [-½(m·log(2π) + log|Sₜ| + ẑₜᵀ·Sₜ⁻¹·ẑₜ)]
"""
function kalman_filter(model::LinearStateSpace, Y::Matrix{Float64})
    T, m = size(Y)
    n = length(model.x0)

    x_pred = zeros(T, n)
    P_pred = zeros(T, n, n)
    x_filt = zeros(T, n)
    P_filt = zeros(T, n, n)
    innov  = zeros(T, m)
    S_arr  = zeros(T, m, m)
    K_arr  = zeros(T, n, m)

    ll = 0.0
    x = copy(model.x0)
    P = copy(model.P0)

    for t in 1:T
        # Predict
        x_p = model.F * x
        P_p = model.F * P * model.F' + model.Q
        P_p = (P_p + P_p') / 2  # enforce symmetry

        x_pred[t,:] = x_p
        P_pred[t,:,:] = P_p

        # Innovation
        y = Y[t,:]
        z = y - model.H * x_p
        S = model.H * P_p * model.H' + model.R
        S = (S + S') / 2

        innov[t,:] = z
        S_arr[t,:,:] = S

        # Kalman gain
        S_chol = cholesky(Symmetric(S))
        K = (P_p * model.H') / S  # n×m

        K_arr[t,:,:] = K

        # Update
        x = x_p + K * z
        I_KH = I - K * model.H
        P = I_KH * P_p * I_KH' + K * model.R * K'  # Joseph form

        x_filt[t,:] = x
        P_filt[t,:,:] = P

        # Log-likelihood contribution
        ll += -0.5 * (m * log(2π) + logdet(S) + dot(z, S \ z))
    end

    return KalmanOutput(x_filt, P_filt, x_pred, P_pred, innov, S_arr, ll, K_arr)
end

"""
Kalman smoother (Rauch-Tung-Striebel backward pass).

Computes E[xₜ | Y₁,...,Yᵀ] using forward filtered estimates.

Backward pass:
    Lₜ = Pₜ|ₜ · Fᵀ · (Pₜ₊₁|ₜ)⁻¹         (smoother gain)
    x̂ₜ|ₜ = x̂ₜ|ₜ + Lₜ(x̂ₜ₊₁|ₜ - Fx̂ₜ|ₜ)
    Pₜ|ₜ = Pₜ|ₜ + Lₜ(Pₜ₊₁|ₜ - Pₜ₊₁|ₜ)Lₜᵀ
"""
function kalman_smoother(model::LinearStateSpace, kf_output::KalmanOutput)
    T, n = size(kf_output.x_filtered)

    x_smooth = copy(kf_output.x_filtered)
    P_smooth = copy(kf_output.P_filtered)

    for t in T-1:-1:1
        P_f = kf_output.P_filtered[t,:,:]
        P_p_next = kf_output.P_predicted[t+1,:,:]
        x_f = kf_output.x_filtered[t,:]
        x_p_next = kf_output.x_predicted[t+1,:]
        x_s_next = x_smooth[t+1,:]
        P_s_next = P_smooth[t+1,:,:]

        L = P_f * model.F' * inv(P_p_next)  # smoother gain
        x_smooth[t,:] = x_f + L * (x_s_next - x_p_next)
        P_smooth[t,:,:] = P_f + L * (P_s_next - P_p_next) * L'
        P_smooth[t,:,:] = (P_smooth[t,:,:] + P_smooth[t,:,:]') / 2
    end

    return x_smooth, P_smooth
end

# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
Extended Kalman Filter for nonlinear state-space models.

    xₜ = f(xₜ₋₁) + wₜ,   wₜ ~ N(0,Q)
    yₜ = h(xₜ) + vₜ,     vₜ ~ N(0,R)

Linearise f and h via Jacobians:
    Fₜ = ∂f/∂x |_{x̂ₜ₋₁|ₜ₋₁}
    Hₜ = ∂h/∂x |_{x̂ₜ|ₜ₋₁}

Uses forward-difference numerical Jacobians if analytical ones are not provided.
"""
function ekf(f::Function, h::Function, Q::Matrix, R::Matrix,
             Y::Matrix{Float64}, x0::Vector, P0::Matrix;
             Jf::Union{Function,Nothing}=nothing,
             Jh::Union{Function,Nothing}=nothing,
             ε_jac=1e-5)
    T, m = size(Y)
    n = length(x0)

    x_filt = zeros(T, n)
    P_filt = zeros(T, n, n)
    x_pred = zeros(T, n)
    P_pred = zeros(T, n, n)
    innov  = zeros(T, m)
    ll = 0.0

    x = copy(x0)
    P = copy(P0)

    # Numerical Jacobian via forward differences
    function jacobian(func, x_at, out_dim)
        n_in = length(x_at)
        J = zeros(out_dim, n_in)
        f0 = func(x_at)
        for j in 1:n_in
            xp = copy(x_at); xp[j] += ε_jac
            J[:,j] = (func(xp) - f0) ./ ε_jac
        end
        return J
    end

    for t in 1:T
        # Predict
        x_p = f(x)
        Ft = isnothing(Jf) ? jacobian(f, x, n) : Jf(x)
        P_p = Ft * P * Ft' + Q
        P_p = (P_p + P_p') / 2

        x_pred[t,:] = x_p
        P_pred[t,:,:] = P_p

        # Innovation
        h_x_p = h(x_p)
        Ht = isnothing(Jh) ? jacobian(h, x_p, m) : Jh(x_p)
        z = Y[t,:] .- h_x_p
        S = Ht * P_p * Ht' + R
        S = (S + S') / 2

        innov[t,:] = z

        # Update
        K = (P_p * Ht') / S
        x = x_p + K * z
        P = (I - K * Ht) * P_p
        P = (P + P') / 2

        x_filt[t,:] = x
        P_filt[t,:,:] = P

        ll += -0.5 * (m * log(2π) + logdet(S) + dot(z, S \ z))
    end

    return x_filt, P_filt, x_pred, P_pred, innov, ll
end

# ─────────────────────────────────────────────────────────────────────────────
# UNSCENTED KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
Unscented Transform: propagate a Gaussian through a nonlinear function.

Generates 2n+1 sigma points:
    χ₀ = x̄
    χᵢ = x̄ + (√((n+λ)P))ᵢ,   i=1,...,n
    χᵢ = x̄ - (√((n+λ)P))ᵢ₋ₙ, i=n+1,...,2n

where λ = α²(n+κ) - n is a scaling parameter.

Returns (mean, covariance, cross-covariance) after propagation.
"""
function unscented_transform(x::Vector, P::Matrix, func::Function;
                              α=1e-3, β=2.0, κ=0.0)
    n = length(x)
    λ = α^2 * (n + κ) - n

    # Weights
    Wm = vcat([λ/(n+λ)], fill(1/(2(n+λ)), 2n))
    Wc = vcat([λ/(n+λ) + (1 - α^2 + β)], fill(1/(2(n+λ)), 2n))

    # Sigma points
    sqrtP = cholesky(Hermitian((n+λ) * P)).L
    σ_pts = zeros(n, 2n+1)
    σ_pts[:,1] = x
    for i in 1:n
        σ_pts[:,i+1]   = x + sqrtP[:,i]
        σ_pts[:,i+1+n] = x - sqrtP[:,i]
    end

    # Propagate
    Y_pts = hcat([func(σ_pts[:,i]) for i in 1:2n+1]...)
    m_out = size(Y_pts, 1)

    # Weighted mean
    ȳ = Y_pts * Wm

    # Weighted covariance
    Pyy = zeros(m_out, m_out)
    Pxy = zeros(n, m_out)
    for i in 1:2n+1
        dy = Y_pts[:,i] .- ȳ
        dx = σ_pts[:,i] .- x
        Pyy += Wc[i] * (dy * dy')
        Pxy += Wc[i] * (dx * dy')
    end

    return ȳ, Pyy, Pxy, σ_pts
end

"""
Unscented Kalman Filter.
"""
function ukf(f::Function, h::Function, Q::Matrix, R::Matrix,
             Y::Matrix{Float64}, x0::Vector, P0::Matrix;
             α=1e-3, β=2.0, κ=0.0)
    T, m = size(Y)
    n = length(x0)

    x_filt = zeros(T, n)
    P_filt = zeros(T, n, n)
    innov  = zeros(T, m)
    ll = 0.0

    x = copy(x0)
    P = copy(P0)

    for t in 1:T
        # Predict via UT
        x_p, Pxx, _, _ = unscented_transform(x, P, f; α=α, β=β, κ=κ)
        P_p = Pxx + Q
        P_p = (P_p + P_p') / 2

        # Update via UT
        ȳ, Pyy, Pxy, _ = unscented_transform(x_p, P_p, h; α=α, β=β, κ=κ)
        S = Pyy + R
        S = (S + S') / 2

        z = Y[t,:] .- ȳ
        innov[t,:] = z

        K = Pxy / S
        x = x_p + K * z
        P = P_p - K * S * K'
        P = (P + P') / 2

        x_filt[t,:] = x
        P_filt[t,:,:] = P

        ll += -0.5 * (m * log(2π) + logdet(S) + dot(z, S \ z))
    end

    return x_filt, P_filt, innov, ll
end

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

"""
Ensemble Kalman Filter (EnKF) for high-dimensional state estimation.

Uses N ensemble members {xₜ⁽ⁱ⁾} to represent the posterior distribution.
Suitable when the state dimension n >> observation dimension m.

Prediction:
    x̂ₜ|ₜ₋₁⁽ⁱ⁾ = f(xₜ₋₁⁽ⁱ⁾) + ηₜ⁽ⁱ⁾,  ηₜ⁽ⁱ⁾ ~ N(0,Q)

Update (stochastic EnKF):
    yₜ⁽ⁱ⁾ = yₜ + eₜ⁽ⁱ⁾,  eₜ⁽ⁱ⁾ ~ N(0,R)
    xₜ|ₜ⁽ⁱ⁾ = x̂ₜ|ₜ₋₁⁽ⁱ⁾ + Kₜ(yₜ⁽ⁱ⁾ - h(x̂ₜ|ₜ₋₁⁽ⁱ⁾))

    Kₜ = P̂ₜ Hᵀ (H P̂ₜ Hᵀ + R)⁻¹   [empirical covariance from ensemble]
"""
function enkf(f::Function, h::Function, Q::Matrix, R::Matrix,
              Y::Matrix{Float64}, x0::Vector, P0::Matrix;
              N_ensemble=100, rng=Random.GLOBAL_RNG)
    T, m = size(Y)
    n = length(x0)

    # Initialise ensemble from prior
    X = rand(rng, MvNormal(x0, P0), N_ensemble)  # n × N

    x_mean = zeros(T, n)
    P_ens  = zeros(T, n, n)

    for t in 1:T
        # Predict each ensemble member
        Q_sqrt = cholesky(Hermitian(Q)).L
        for i in 1:N_ensemble
            X[:,i] = f(X[:,i]) + Q_sqrt * randn(rng, n)
        end

        # Ensemble mean and anomaly
        x̄ = mean(X, dims=2)[:]
        A = X .- x̄  # n × N, anomaly matrix

        # Ensemble covariance (sample)
        P̂ = (A * A') / (N_ensemble - 1)

        # Observation operator at each member
        HX = hcat([h(X[:,i]) for i in 1:N_ensemble]...)  # m × N
        Hx̄ = mean(HX, dims=2)[:]
        HA = HX .- Hx̄

        # Innovation covariance
        S = (HA * HA') / (N_ensemble - 1) + R

        # Kalman gain (using ensemble cross-covariance)
        PHᵀ = (A * HA') / (N_ensemble - 1)  # n × m
        K = PHᵀ / S

        # Perturbed observations
        R_sqrt = cholesky(Hermitian(R)).L
        for i in 1:N_ensemble
            y_perturbed = Y[t,:] + R_sqrt * randn(rng, m)
            X[:,i] += K * (y_perturbed .- HX[:,i])
        end

        x_mean[t,:] = mean(X, dims=2)[:]
        P_ens[t,:,:] = (X .- x_mean[t,:]') * (X .- x_mean[t,:]')' / (N_ensemble - 1)
    end

    return x_mean, P_ens
end

# ─────────────────────────────────────────────────────────────────────────────
# EM ALGORITHM FOR PARAMETER ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

"""
EM algorithm for estimating Q (process noise) and R (observation noise)
in a linear state-space model.

E-step: run Kalman smoother with current parameters
M-step: update Q and R using sufficient statistics:
    R̂ = (1/T) Σₜ (yₜ - Hx̂ₜ|ₜ)(yₜ - Hx̂ₜ|ₜ)ᵀ + H·Pₜ|ₜ·Hᵀ
    Q̂ = (1/T) Σₜ (Pₜ|ₜ + x̂ₜ|ₜx̂ₜ|ₜᵀ - Fₜ·cov(xₜ,xₜ₋₁|ₜ)ᵀ - ...)

Reference: Shumway & Stoffer (1982)
"""
function em_kalman(model::LinearStateSpace, Y::Matrix{Float64};
                   max_iter=100, tol=1e-6, verbose=false)
    T, m = size(Y)
    n = length(model.x0)
    F = model.F; H = model.H
    Q = copy(model.Q); R = copy(model.R)
    x0 = copy(model.x0); P0 = copy(model.P0)

    ll_prev = -Inf

    for iter in 1:max_iter
        curr_model = LinearStateSpace(F, H, Q, R, x0, P0)

        # E-step: forward filter + backward smoother
        kf_out = kalman_filter(curr_model, Y)
        x_s, P_s = kalman_smoother(curr_model, kf_out)

        # Lag-one covariance smoother (for Q update)
        # Pₜ,ₜ₋₁ = (I - Kₜ·H)·F·Pₜ₋₁|ₜ₋₁  (then backward pass)
        P_lag = zeros(T, n, n)
        # Compute final lag covariance
        L = zeros(T-1, n, n)
        for t in 1:T-1
            P_ft = kf_out.P_filtered[t,:,:]
            P_p_next = kf_out.P_predicted[t+1,:,:]
            L[t,:,:] = P_ft * F' * inv(P_p_next)
        end
        P_lag[T,:,:] = (I - kf_out.K[T,:,:] * H) * F * kf_out.P_filtered[T-1,:,:]
        for t in T-1:-1:2
            P_lag[t,:,:] = kf_out.P_filtered[t,:,:] * L[t-1,:,:]' +
                           L[t,:,:] * (P_lag[t+1,:,:] - F * kf_out.P_filtered[t,:,:]) * L[t-1,:,:]'
        end

        # M-step: update Q and R
        # E[xxᵀ]_t
        Exx  = [P_s[t,:,:] + x_s[t,:]*x_s[t,:]' for t in 1:T]
        Exx_lag = [P_lag[t,:,:] + x_s[t,:]*x_s[t-1,:]' for t in 2:T]

        # Update R
        R_new = zeros(m, m)
        for t in 1:T
            r = Y[t,:] - H * x_s[t,:]
            R_new += r*r' + H*P_s[t,:,:]*H'
        end
        R = R_new ./ T

        # Update Q
        Q_new = zeros(n, n)
        for t in 2:T
            Q_new += Exx[t] - F*Exx_lag[t-1]' - Exx_lag[t-1]*F' + F*Exx[t-1]*F'
        end
        Q = Q_new ./ (T-1)

        # Ensure positive definiteness
        Q = (Q + Q') / 2 + 1e-10*I
        R = (R + R') / 2 + 1e-10*I

        # Update initial conditions
        x0 = x_s[1,:]
        P0 = P_s[1,:,:]

        # Check convergence
        ll_new = kf_out.log_likelihood
        verbose && println("EM iter $iter: ll=$(round(ll_new, digits=4))")

        abs(ll_new - ll_prev) < tol && (verbose && println("Converged at iter $iter"); break)
        ll_prev = ll_new
    end

    return LinearStateSpace(F, H, Q, R, x0, P0)
end

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE Q/R COVARIANCE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Adaptive Kalman filter: estimate Q and R online using innovation statistics.

Innovation-based adaptation (Mehra 1970):
    Ŝₜ = (1/M) Σ_{k=t-M+1}^{t} zₖzₖᵀ    [empirical innovation covariance]
    R̂ₜ = Ŝₜ - H·Pₜ|ₜ₋₁·Hᵀ
    Q̂ₜ = Kₜ·Ŝₜ·Kₜᵀ

Runs the standard KF but periodically updates Q and R.
"""
function adaptive_kalman_filter(model::LinearStateSpace, Y::Matrix{Float64};
                                  window=30, adapt_freq=10)
    T, m = size(Y)
    n = length(model.x0)

    x_filt = zeros(T, n)
    P_filt = zeros(T, n, n)
    innov  = zeros(T, m)

    x = copy(model.x0)
    P = copy(model.P0)
    Q = copy(model.Q)
    R = copy(model.R)

    for t in 1:T
        # Predict
        x_p = model.F * x
        P_p = model.F * P * model.F' + Q
        P_p = (P_p + P_p') / 2

        # Update
        z = Y[t,:] - model.H * x_p
        S = model.H * P_p * model.H' + R
        K = (P_p * model.H') / S
        x = x_p + K * z
        P = (I - K * model.H) * P_p
        P = (P + P') / 2

        x_filt[t,:] = x
        P_filt[t,:,:] = P
        innov[t,:] = z

        # Adaptive update every adapt_freq steps
        if t >= window && t % adapt_freq == 0
            recent_innov = innov[t-window+1:t, :]
            S_hat = (recent_innov' * recent_innov) ./ window

            R_new = S_hat - model.H * P_p * model.H'
            R = (R_new + R_new') / 2 + 1e-8*I
            R = max.(R, 1e-8*I)  # ensure PD

            Q_new = K * S_hat * K'
            Q = (Q_new + Q_new') / 2 + 1e-10*I
        end
    end

    return x_filt, P_filt, innov
end

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
Trend + noise decomposition using a local linear trend (LLT) model.

State: xₜ = [μₜ, βₜ]ᵀ (level + trend)
    μₜ = μₜ₋₁ + βₜ₋₁ + wₜ¹
    βₜ = βₜ₋₁ + wₜ²
    yₜ = μₜ + vₜ

Used for de-noising price or P&L series.
"""
function local_linear_trend(y::Vector{Float64};
                              σ_level=1.0, σ_slope=0.1, σ_obs=1.0,
                              estimate_params=true, verbose=false)
    T = length(y)
    Y = reshape(y, T, 1)

    # Model
    F = [1.0 1.0; 0.0 1.0]
    H = [1.0 0.0]
    Q = Diagonal([σ_level^2, σ_slope^2])
    R = fill(σ_obs^2, 1, 1)
    x0 = [y[1], 0.0]
    P0 = Diagonal([100.0, 1.0])

    model = LinearStateSpace(F, H, Matrix(Q), R, x0, Matrix(P0))

    if estimate_params
        verbose && println("Running EM to estimate Q, R...")
        model = em_kalman(model, Y; max_iter=50, verbose=verbose)
    end

    kf_out = kalman_filter(model, Y)
    x_s, P_s = kalman_smoother(model, kf_out)

    trend = x_s[:,1]
    slope = x_s[:,2]
    trend_std = sqrt.([P_s[t,1,1] for t in 1:T])

    return (trend=trend, slope=slope, trend_std=trend_std,
            model=model, kf=kf_out)
end

"""
Latent volatility extraction via a stochastic volatility model.

Discrete-time log-SV model:
    yₜ = σₜ · εₜ,   εₜ ~ N(0,1)
    log σₜ² = φ · log σₜ₋₁² + ητₜ,  ητ ~ N(0,σ²_η)

Rewritten: let hₜ = log σₜ²:
    log yₜ² = hₜ + log εₜ²   [observation, log χ²(1) noise]
    hₜ = φ · hₜ₋₁ + ητₜ

Estimated via EKF (observation noise is non-Gaussian, approximated as N(-1.27, π²/2)).
"""
function stochastic_volatility_filter(returns::Vector{Float64};
                                       φ₀=0.97, ση₀=0.1)
    T = length(returns)
    Y_log = log.(returns.^2 .+ 1e-8)

    # Approx: log(χ²(1)) ≈ N(-1.2704, π²/2)
    μ_logchi = -1.2704
    σ_logchi = π^2 / 2

    F_sv = fill(φ₀, 1, 1)
    H_sv = fill(1.0, 1, 1)
    Q_sv = fill(ση₀^2, 1, 1)
    R_sv = fill(σ_logchi, 1, 1)
    x0_sv = [mean(Y_log)]
    P0_sv = fill(ση₀^2 / (1 - φ₀^2), 1, 1)

    model_sv = LinearStateSpace(F_sv, H_sv, Q_sv, R_sv, x0_sv, P0_sv)
    Y_sv = reshape(Y_log .- μ_logchi, T, 1)

    kf_sv = kalman_filter(model_sv, Y_sv)
    h_filtered = kf_sv.x_filtered[:,1] .+ μ_logchi
    σ_filtered  = exp.(h_filtered ./ 2)

    return σ_filtered, h_filtered, kf_sv
end

"""
Pairs trading spread estimation via Kalman filter.

Model the hedge ratio β as a time-varying state:
    y₁ₜ = βₜ · y₂ₜ + αₜ + vₜ    (spread = y₁ - β·y₂ - α)
    βₜ = βₜ₋₁ + wₜ^β
    αₜ = αₜ₋₁ + wₜ^α

State: xₜ = [βₜ, αₜ]ᵀ
F = I₂
H = [y₂ₜ, 1]   (time-varying observation matrix)
"""
function pairs_trading_spread(y1::Vector{Float64}, y2::Vector{Float64};
                               σ_β=0.001, σ_α=0.001, σ_obs=0.01)
    T = length(y1)
    length(y2) == T || throw(DimensionMismatch("y1 and y2 must have same length"))

    x = [y1[1]/y2[1], 0.0]
    P = Diagonal([1.0, 1.0])
    Q = Diagonal([σ_β^2, σ_α^2])
    R = fill(σ_obs^2, 1, 1)
    F = Matrix(1.0*I, 2, 2)

    β_filt = zeros(T)
    α_filt = zeros(T)
    spread  = zeros(T)

    for t in 1:T
        # Time-varying H
        H = reshape([y2[t], 1.0], 1, 2)

        # Predict
        x_p = F * x
        P_p = F * P * F' + Q

        # Update
        z = y1[t] - (H * x_p)[1]
        S = (H * P_p * H')[1,1] + R[1,1]
        K = (P_p * H') ./ S
        x = x_p + K .* z
        P = (Matrix(I, 2, 2) - K * H) * P_p

        β_filt[t] = x[1]
        α_filt[t] = x[2]
        spread[t]  = y1[t] - β_filt[t]*y2[t] - α_filt[t]
    end

    # Normalise spread by its rolling standard deviation
    σ_spread = std(spread)
    z_spread = spread ./ σ_spread

    return (β=β_filt, α=α_filt, spread=spread, z_spread=z_spread)
end

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Plot filtered trend with confidence bands.
"""
function plot_trend(y, trend, trend_std; title="Local Linear Trend")
    T = length(y)
    t = 1:T

    p = plot(t, y, label="Observations", color=:gray, alpha=0.5, linewidth=0.5)
    plot!(p, t, trend, label="Filtered trend", color=:blue, linewidth=2)
    plot!(p, t, trend .+ 2*trend_std, fillrange=trend .- 2*trend_std,
          alpha=0.2, color=:blue, label="±2σ band")
    title!(p, title)
    xlabel!(p, "Time")
    ylabel!(p, "Value")
    return p
end

"""
Plot latent volatility estimate.
"""
function plot_volatility(returns, σ_filtered; title="Latent Volatility (EKF)")
    T = length(returns)
    t = 1:T

    p1 = plot(t, returns, label="Returns", color=:gray, linewidth=0.5,
              ylabel="Return", title=title)
    p2 = plot(t, σ_filtered, label="σ̂ₜ (filtered)", color=:red, linewidth=1.5,
              ylabel="Volatility", xlabel="Time")
    plot!(p2, t, abs.(returns), label="|rₜ|", color=:gray, alpha=0.4, linewidth=0.5)

    return plot(p1, p2, layout=(2,1), size=(900, 500))
end

"""
Plot pairs trading spread with Z-score and entry/exit signals.
"""
function plot_spread(result; entry_z=2.0, exit_z=0.5, title="Pairs Trading Spread")
    T = length(result.spread)
    t = 1:T

    p1 = plot(t, result.β, label="β̂ₜ (hedge ratio)", color=:blue,
              ylabel="Hedge ratio", title=title)

    p2 = plot(t, result.z_spread, label="Z-spread", color=:purple, linewidth=1.5,
              ylabel="Z-score", xlabel="Time")
    hline!(p2, [entry_z, -entry_z], color=:red, linestyle=:dash, label="Entry ±$(entry_z)σ")
    hline!(p2, [exit_z, -exit_z], color=:green, linestyle=:dash, label="Exit ±$(exit_z)σ")
    hline!(p2, [0.0], color=:black, linestyle=:dot, label=nothing)

    # Mark entry/exit points
    long_entries  = findall(result.z_spread .< -entry_z)
    short_entries = findall(result.z_spread .> entry_z)
    isempty(long_entries)  || scatter!(p2, long_entries,  result.z_spread[long_entries],
                                        color=:green, markersize=4, label="Long entry")
    isempty(short_entries) || scatter!(p2, short_entries, result.z_spread[short_entries],
                                        color=:red, markersize=4, label="Short entry")

    return plot(p1, p2, layout=(2,1), size=(900, 500))
end

# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

function demo()
    Random.seed!(42)
    println("=" ^ 60)
    println("Enhanced Kalman Filtering Demo")
    println("=" ^ 60)

    T = 500

    # 1. Local linear trend model
    println("\n1. Local Linear Trend Decomposition")
    true_trend = cumsum(randn(T) .* 0.05)
    noise = randn(T) .* 0.5
    y_obs = true_trend .+ noise

    result_llt = local_linear_trend(y_obs; estimate_params=true, verbose=true)
    corr = cor(result_llt.trend, true_trend)
    println("  Correlation with true trend: $(round(corr, digits=4))")

    # 2. Stochastic volatility extraction
    println("\n2. Stochastic Volatility Extraction")
    # GARCH-like returns with changing volatility
    σ_true = zeros(T)
    σ_true[1] = 0.02
    for t in 2:T
        σ_true[t] = sqrt(0.00001 + 0.1 * σ_true[t-1]^2 * randn()^2 + 0.85 * σ_true[t-1]^2)
    end
    returns = randn(T) .* σ_true
    σ_est, h_est, _ = stochastic_volatility_filter(returns)
    corr_σ = cor(σ_est, σ_true)
    println("  Correlation with true σₜ: $(round(corr_σ, digits=4))")

    # 3. Pairs trading spread
    println("\n3. Pairs Trading Spread Estimation")
    β_true = 1.5 .+ 0.1 .* cumsum(randn(T) .* 0.01)  # slowly drifting hedge ratio
    y2_base = cumsum(randn(T) .* 0.01) .+ 100.0
    y1_base = β_true .* y2_base .+ cumsum(randn(T) .* 0.001) .+ randn(T) .* 0.02

    spread_result = pairs_trading_spread(y1_base, y2_base; σ_β=0.005, σ_obs=0.02)
    corr_β = cor(spread_result.β, β_true)
    println("  Correlation with true β: $(round(corr_β, digits=4))")
    println("  Spread Z-score: mean=$(round(mean(spread_result.z_spread),digits=3)), std=$(round(std(spread_result.z_spread),digits=3))")

    # 4. UKF on nonlinear system
    println("\n4. Unscented Kalman Filter (nonlinear)")
    # Logistic growth model for price
    f_nl(x) = [x[1] / (1 + 0.01 * x[1])]
    h_nl(x) = x
    Q_nl = fill(0.1, 1, 1)
    R_nl = fill(0.5, 1, 1)
    x0_nl = [1.0]
    P0_nl = fill(1.0, 1, 1)
    # Generate synthetic data
    y_nl = cumsum(randn(100) .* 0.3) .+ 5.0
    Y_nl = reshape(y_nl, 100, 1)
    x_ukf, P_ukf, innov_ukf, ll_ukf = ukf(f_nl, h_nl, Q_nl, R_nl, Y_nl, x0_nl, P0_nl)
    println("  UKF log-likelihood: $(round(ll_ukf, digits=2))")

    # 5. EnKF for higher-dimensional state
    println("\n5. Ensemble Kalman Filter")
    n_state = 5
    F_en = 0.95 * Matrix(I, n_state, n_state)
    H_en = [1.0 zeros(1, n_state-1)]
    Q_en = 0.01 * Matrix(I, n_state, n_state)
    R_en = fill(0.1, 1, 1)
    x0_en = zeros(n_state)
    P0_en = Matrix(I, n_state, n_state)

    f_en(x) = F_en * x
    h_en(x) = H_en * x

    Y_en = cumsum(randn(200, 1), dims=1) .* 0.1
    x_enkf, P_enkf = enkf(f_en, h_en, Q_en, R_en, Y_en, x0_en, P0_en;
                           N_ensemble=200)
    println("  EnKF final state: $(round.(x_enkf[end,:], digits=4))")

    # Plots
    p_trend = plot_trend(y_obs, result_llt.trend, result_llt.trend_std)
    p_vol   = plot_volatility(returns, σ_est)
    p_pairs = plot_spread(spread_result)

    savefig(p_trend, "kalman_trend.png")
    savefig(p_vol,   "kalman_vol.png")
    savefig(p_pairs, "kalman_pairs.png")
    println("\nSaved: kalman_trend.png, kalman_vol.png, kalman_pairs.png")

    return result_llt, spread_result
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
