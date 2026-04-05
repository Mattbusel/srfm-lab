"""
SignalProcessing — Signal processing for financial time series.

Implements: Hodrick-Prescott filter, Christiano-Fitzgerald band-pass filter,
Kalman filter & smoother, Unscented Kalman filter, Particle filter (SMC),
Empirical Mode Decomposition (EMD) / Hilbert-Huang Transform,
Singular Spectrum Analysis (SSA), Savitzky-Golay filter.
All applied to crypto price series for regime and signal extraction.
"""
module SignalProcessing

using Statistics
using LinearAlgebra
using Random

export HPFilter, CFFilter, KalmanFilter, KalmanSmoother
export UnscentedKalmanFilter, ParticleFilter
export EMD, SSA, SavitzkyGolay
export run_signal_processing

# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Construct second-difference matrix D2 of size (n-2)×n."""
function _diff2_matrix(n::Int)::Matrix{Float64}
    D = zeros(n-2, n)
    for i in 1:(n-2)
        D[i, i]   = 1.0
        D[i, i+1] = -2.0
        D[i, i+2] = 1.0
    end
    return D
end

"""Tridiagonal solve (Thomas algorithm)."""
function _tridiag_solve(a::Vector{Float64}, b::Vector{Float64},
                         c::Vector{Float64}, d::Vector{Float64})::Vector{Float64}
    n = length(d)
    cp = copy(c); dp = copy(d); x = zeros(n)
    cp[1] /= b[1]; dp[1] /= b[1]
    for i in 2:n
        m = b[i] - a[i]*cp[i-1]
        abs(m) < 1e-14 && (m = 1e-14)
        cp[i] = c[i]/m; dp[i] = (d[i] - a[i]*dp[i-1])/m
    end
    x[n] = dp[n]
    for i in (n-1):-1:1; x[i] = dp[i] - cp[i]*x[i+1]; end
    return x
end

"""Cholesky factor with fallback."""
function _safe_chol(Σ::AbstractMatrix)
    try cholesky(Symmetric(Matrix(Σ))).L
    catch; cholesky(Symmetric(Matrix(Σ) + 1e-6*I)).L
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Hodrick-Prescott Filter
# ─────────────────────────────────────────────────────────────────────────────

"""
    HPFilter(y; lambda) → NamedTuple

Hodrick-Prescott filter for trend-cycle decomposition.

Minimises: ∑(y_t - τ_t)² + λ ∑(Δ²τ_t)²

The trend τ is the solution to:
(I + λ D₂'D₂) τ = y

# Arguments
- `y`      : time series vector
- `lambda` : smoothing parameter (default 1600 for quarterly; 6.25 for annual)

# Returns
NamedTuple: (trend, cycle, lambda, cycle_stats)

# Example
```julia
prices = cumsum(randn(500) .* 0.01) .+ 10.0
hp = HPFilter(prices; lambda=1600.0)
println("Cycle std: ", std(hp.cycle))
```
"""
function HPFilter(y::Vector{Float64}; lambda::Float64=1600.0)
    n = length(y)
    n < 4 && error("HP filter requires n ≥ 4")

    # Build (I + λ D₂'D₂) using sparse-style tridiagonal structure
    # The matrix is banded with bandwidth 2
    D2 = _diff2_matrix(n)
    A  = I(n) + lambda * D2' * D2  # n × n dense matrix
    # Solve A τ = y via Cholesky (A is SPD)
    F = cholesky(Symmetric(Matrix(A) + 1e-10*I))
    trend = F \ y
    cycle = y .- trend

    # Cycle statistics
    cycle_std  = std(cycle)
    cycle_mean = mean(cycle)
    # Autocorrelation of cycle
    ac1 = length(cycle) > 2 ? cov(cycle[2:end], cycle[1:(end-1)]) / max(var(cycle), 1e-10) : 0.0

    return (trend=trend, cycle=cycle, lambda=lambda,
            cycle_std=cycle_std, cycle_mean=cycle_mean,
            cycle_autocorr=ac1)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Christiano-Fitzgerald Band-Pass Filter
# ─────────────────────────────────────────────────────────────────────────────

"""
    CFFilter(y; pl, pu, drift) → NamedTuple

Christiano-Fitzgerald (2003) band-pass filter.
Extracts cycles with period between pl and pu.

Uses optimal (in MSE sense) two-sided moving average approximation to the
ideal band-pass filter, with full-sample asymmetric end-point corrections.

# Arguments
- `y`     : time series
- `pl`    : lower period bound (default 6 = 1.5 years if quarterly)
- `pu`    : upper period bound (default 32 = 8 years if quarterly)
- `drift` : if true, remove linear drift before filtering (default true)

# Returns
NamedTuple: (cycle, trend, pl, pu, filter_weights)
"""
function CFFilter(y::Vector{Float64}; pl::Float64=6.0, pu::Float64=32.0,
                  drift::Bool=true)
    n = length(y)
    pl < 2.0 && error("pl must be ≥ 2")
    pu > pl  || error("pu must be > pl")

    # Remove drift if requested
    t_vec = Float64.(1:n)
    y_work = if drift
        slope = cov(t_vec, y) / var(t_vec)
        intercept = mean(y) - slope * mean(t_vec)
        y .- (intercept .+ slope .* t_vec)
    else
        copy(y)
    end

    # Ideal filter weights
    ω_l = 2π / pu; ω_u = 2π / pl
    function b(k::Int)
        k == 0 && return (ω_u - ω_l) / π
        (sin(k * ω_u) - sin(k * ω_l)) / (k * π)
    end

    # Full-sample implementation (asymmetric at ends)
    cycle = zeros(n)
    max_lag = min(n÷2, 50)

    for t in 1:n
        # Determine available lags
        lag_lo = -(t-1); lag_hi = n-t
        # Truncate to max_lag
        lag_lo = max(lag_lo, -max_lag); lag_hi = min(lag_hi, max_lag)

        # CF adjustment: modify endpoint weights
        c_t = 0.0
        for k in lag_lo:lag_hi
            w_k = b(k)
            c_t += w_k * y_work[t + k]
        end
        cycle[t] = c_t
    end

    trend = y_work .- cycle

    # Filter weights for documentation
    weights = [b(k) for k in (-max_lag):max_lag]

    return (cycle=cycle, trend=trend .+ (drift ? (y .- y_work) : 0.0),
            pl=pl, pu=pu, filter_weights=weights)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Kalman Filter and Smoother
# ─────────────────────────────────────────────────────────────────────────────

"""
    KalmanFilter(y, F, H, Q, R, x0, P0) → NamedTuple

Linear Gaussian Kalman filter for state-space model:
    x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
    y_t = H x_t + v_t,        v_t ~ N(0, R)

# Arguments
- `y`  : observation vector (n × p matrix or n-vector for p=1)
- `F`  : state transition matrix (d × d)
- `H`  : observation matrix (p × d)
- `Q`  : process noise covariance (d × d)
- `R`  : observation noise covariance (p × p)
- `x0` : initial state mean (d-vector)
- `P0` : initial state covariance (d × d)

# Returns
NamedTuple: (filtered_mean, filtered_cov, predicted_mean, predicted_cov,
             log_likelihood, innovations, innovation_cov)
"""
function KalmanFilter(y::Union{Vector{Float64}, Matrix{Float64}},
                       F::Matrix{Float64}, H::Matrix{Float64},
                       Q::Matrix{Float64}, R::Matrix{Float64},
                       x0::Vector{Float64}, P0::Matrix{Float64})
    if isa(y, Vector)
        y = reshape(y, :, 1)
    end
    n, p = size(y)
    d = length(x0)

    xf = zeros(d, n)     # filtered means
    Pf = zeros(d, d, n)  # filtered covariances
    xp = zeros(d, n)     # predicted means
    Pp = zeros(d, d, n)  # predicted covariances
    innovations   = zeros(p, n)
    innov_cov     = zeros(p, p, n)
    log_lik = 0.0

    x_prev = copy(x0)
    P_prev = copy(P0)

    for t in 1:n
        # Predict
        x_pred = F * x_prev
        P_pred = F * P_prev * F' + Q
        P_pred = Symmetric(P_pred + P_pred') ./ 2  # ensure symmetry

        xp[:, t] = x_pred
        Pp[:, :, t] = Matrix(P_pred)

        # Update
        S = H * P_pred * H' + R   # innovation covariance
        S = Symmetric(S + S') ./ 2
        innov_cov[:, :, t] = Matrix(S)

        # Observation (handle missing: NaN → skip update)
        y_t = y[t, :]
        any(isnan.(y_t)) && begin
            xf[:, t] = x_pred; Pf[:, :, t] = Matrix(P_pred)
            x_prev = x_pred; P_prev = Matrix(P_pred); continue
        end

        innov = y_t .- H * x_pred
        innovations[:, t] = innov

        K = P_pred * H' / S   # Kalman gain
        x_new = x_pred + K * innov
        P_new = (I(d) - K * H) * P_pred
        P_new = Symmetric(P_new + P_new') ./ 2

        xf[:, t] = x_new
        Pf[:, :, t] = Matrix(P_new)

        # Log-likelihood contribution
        try
            S_mat = Matrix(S)
            L = cholesky(S_mat + 1e-10*I).L
            log_det = 2*sum(log.(diag(L)))
            log_lik += -0.5*(p*log(2π) + log_det + dot(innov, S_mat\innov))
        catch; end

        x_prev = x_new; P_prev = Matrix(P_new)
    end

    return (filtered_mean=xf, filtered_cov=Pf,
            predicted_mean=xp, predicted_cov=Pp,
            log_likelihood=log_lik,
            innovations=innovations, innovation_cov=innov_cov)
end

"""
    KalmanSmoother(kf_result, F, Q) → NamedTuple

Rauch-Tung-Striebel smoother: backward pass over Kalman filter output.

# Arguments
- `kf_result` : output from KalmanFilter
- `F`         : state transition matrix
- `Q`         : process noise covariance

# Returns
NamedTuple: (smoothed_mean, smoothed_cov, smoother_gain)
"""
function KalmanSmoother(kf_result::NamedTuple, F::Matrix{Float64}, Q::Matrix{Float64})
    n = size(kf_result.filtered_mean, 2)
    d = size(kf_result.filtered_mean, 1)

    xs = copy(kf_result.filtered_mean)
    Ps = copy(kf_result.filtered_cov)
    Gs = zeros(d, d, n-1)

    for t in (n-1):-1:1
        Pf_t = kf_result.filtered_cov[:, :, t]
        Pp_next = kf_result.predicted_cov[:, :, t+1]
        Pp_next_reg = Pp_next + 1e-8*I

        G = Pf_t * F' * inv(Symmetric(Pp_next_reg))   # smoother gain
        Gs[:, :, t] = G

        xs[:, t] = kf_result.filtered_mean[:, t] +
                   G * (xs[:, t+1] - kf_result.predicted_mean[:, t+1])
        Ps[:, :, t] = Pf_t + G * (Ps[:, :, t+1] - Pp_next) * G'
        Ps[:, :, t] = (Ps[:, :, t] + Ps[:, :, t]') ./ 2
    end

    return (smoothed_mean=xs, smoothed_cov=Ps, smoother_gain=Gs)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Unscented Kalman Filter
# ─────────────────────────────────────────────────────────────────────────────

"""
    UnscentedKalmanFilter(y, f!, h!, Q, R, x0, P0; alpha, beta, kappa) → NamedTuple

Unscented Kalman filter for nonlinear state-space models:
    x_t = f(x_{t-1}) + w_t
    y_t = h(x_t) + v_t

Uses sigma-point propagation (Julier & Uhlmann, 1997).

# Arguments
- `y`       : n × p observation matrix
- `f!`      : state transition function (x_prev → x_next)
- `h!`      : observation function (x → y)
- `Q`       : process noise covariance
- `R`       : observation noise covariance
- `x0`, `P0`: initial state and covariance
- `alpha`   : sigma-point spread (default 1e-3)
- `beta`    : distribution parameter (default 2.0 for Gaussian)
- `kappa`   : secondary scaling (default 0.0)
"""
function UnscentedKalmanFilter(y::Union{Vector{Float64}, Matrix{Float64}},
                                 f!::Function, h!::Function,
                                 Q::Matrix{Float64}, R::Matrix{Float64},
                                 x0::Vector{Float64}, P0::Matrix{Float64};
                                 alpha::Float64=1e-3, beta::Float64=2.0,
                                 kappa::Float64=0.0)
    if isa(y, Vector); y = reshape(y, :, 1); end
    n, p = size(y)
    d = length(x0)

    # UKF parameters
    λ = alpha^2 * (d + kappa) - d
    W_m = [λ/(d+λ); fill(1.0/(2(d+λ)), 2d)]   # mean weights
    W_c = copy(W_m)
    W_c[1] += (1 - alpha^2 + beta)              # covariance weight for x0

    # Storage
    xf = zeros(d, n)
    Pf = zeros(d, d, n)
    xp = zeros(d, n)
    Pp = zeros(d, d, n)
    innovations = zeros(p, n)
    log_lik = 0.0

    x_prev = copy(x0)
    P_prev = copy(P0)

    for t in 1:n
        # ── Predict ────────────────────────────────────────────────────────
        # Generate sigma points
        S = try _safe_chol(Symmetric((d+λ) * P_prev))
        catch; sqrt(d+λ)*I(d) end

        sigma_pts = [x_prev x_prev .+ S x_prev .- S]   # d × (2d+1)
        n_sigma = size(sigma_pts, 2)

        # Propagate sigma points through f
        sigma_f = hcat([f!(sigma_pts[:, k]) for k in 1:n_sigma]...)
        x_pred = sigma_f * W_m
        P_pred = Q + sum(W_c[k] * (sigma_f[:,k]-x_pred)*(sigma_f[:,k]-x_pred)'
                         for k in 1:n_sigma)
        P_pred = Symmetric(P_pred + P_pred') ./ 2

        xp[:, t] = x_pred
        Pp[:, :, t] = Matrix(P_pred)

        # ── Update ─────────────────────────────────────────────────────────
        y_t = y[t, :]
        if any(isnan.(y_t))
            xf[:, t] = x_pred; Pf[:, :, t] = Matrix(P_pred)
            x_prev = x_pred; P_prev = Matrix(P_pred); continue
        end

        # Regenerate sigma points for prediction step
        S2 = try _safe_chol(Symmetric((d+λ) * P_pred))
        catch; sqrt(d+λ)*I(d) end
        sigma_p2 = hcat([x_pred x_pred .+ S2 x_pred .- S2]...)

        # Propagate through h
        sigma_h = hcat([h!(sigma_p2[:, k]) for k in 1:n_sigma]...)
        y_pred = sigma_h * W_m
        innov = y_t .- y_pred
        innovations[:, t] = innov

        # Innovation covariance and cross-covariance
        S_yy = R + sum(W_c[k] * (sigma_h[:,k]-y_pred)*(sigma_h[:,k]-y_pred)'
                       for k in 1:n_sigma)
        P_xy = sum(W_c[k] * (sigma_p2[:,k]-x_pred)*(sigma_h[:,k]-y_pred)'
                   for k in 1:n_sigma)

        K = P_xy * inv(Symmetric(S_yy + 1e-10*I))
        x_new = x_pred + K * innov
        P_new = P_pred - K * S_yy * K'
        P_new = Symmetric(P_new + P_new') ./ 2

        xf[:, t] = x_new
        Pf[:, :, t] = Matrix(P_new)

        # Log-likelihood
        try
            S_mat = Symmetric(S_yy)
            L_ll  = cholesky(S_mat + 1e-10*I).L
            log_lik += -0.5*(p*log(2π) + 2*sum(log.(diag(L_ll))) +
                              dot(innov, S_mat\innov))
        catch; end

        x_prev = x_new; P_prev = Matrix(P_new)
    end

    return (filtered_mean=xf, filtered_cov=Pf,
            predicted_mean=xp, predicted_cov=Pp,
            log_likelihood=log_lik, innovations=innovations)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Particle Filter (Sequential Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────

"""
    ParticleFilter(y, f!, loglik_fn, Q; n_particles, rng) → NamedTuple

Bootstrap particle filter (Sequential Monte Carlo) for non-linear /
non-Gaussian state-space models.

Algorithm:
1. Sample particles x_t^{(i)} ~ f(x_{t-1}^{(i)}, noise)
2. Weight w^{(i)} ∝ p(y_t | x_t^{(i)})
3. Resample using systematic resampling

# Arguments
- `y`          : n × p observation matrix
- `f!`         : state transition: (x_prev, noise) → x_new
- `loglik_fn`  : log p(y_t | x_t): (y_t, x_t) → Float64
- `Q`          : process noise covariance for generating transitions
- `n_particles`: number of particles (default 1000)
- `x0_sampler` : function () → initial particle (default N(0,I))
"""
function ParticleFilter(y::Union{Vector{Float64}, Matrix{Float64}},
                          f!::Function, loglik_fn::Function,
                          Q::Matrix{Float64};
                          n_particles::Int=1000,
                          x0_sampler::Function=rng -> zeros(size(Q,1)),
                          rng::AbstractRNG=Random.default_rng())
    if isa(y, Vector); y = reshape(y, :, 1); end
    n, p = size(y)
    d = size(Q, 1)
    L_Q = try cholesky(Q + 1e-8*I).L catch; I(d)*0.01 end

    # Initialise particles
    particles = hcat([x0_sampler(rng) .+ L_Q * randn(rng, d) for _ in 1:n_particles]...)
    log_weights = fill(-log(n_particles), n_particles)

    # Storage
    mean_est = zeros(d, n)
    log_lik  = 0.0
    ESS_vec  = Float64[]

    for t in 1:n
        # Propagate
        for i in 1:n_particles
            particles[:, i] = f!(particles[:, i]) .+ L_Q * randn(rng, d)
        end

        y_t = y[t, :]
        if any(isnan.(y_t))
            mean_est[:, t] = particles * exp.(log_weights .- maximum(log_weights))
            push!(ESS_vec, n_particles); continue
        end

        # Weight update
        for i in 1:n_particles
            log_weights[i] += loglik_fn(y_t, particles[:, i])
        end

        # Normalise weights
        max_lw = maximum(log_weights)
        log_weights .-= max_lw
        log_lik += max_lw + log(sum(exp.(log_weights)))
        log_weights .-= log(sum(exp.(log_weights)))

        # ESS
        ess = 1.0 / sum(exp.(2.0 .* log_weights))
        push!(ESS_vec, ess)

        # Weighted mean estimate
        w_norm = exp.(log_weights)
        mean_est[:, t] = particles * w_norm

        # Systematic resampling when ESS < n/2
        if ess < n_particles / 2.0
            cdf = cumsum(w_norm)
            u0 = rand(rng) / n_particles
            j = 1
            idx = zeros(Int, n_particles)
            for i in 1:n_particles
                u_i = u0 + (i-1.0)/n_particles
                while cdf[j] < u_i && j < n_particles; j += 1; end
                idx[i] = j
            end
            particles = particles[:, idx]
            log_weights .= -log(n_particles)
        end
    end

    return (filtered_mean=mean_est, log_likelihood=log_lik,
            ESS=ESS_vec, n_particles=n_particles)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Empirical Mode Decomposition (EMD)
# ─────────────────────────────────────────────────────────────────────────────

"""
    EMD(x; max_imfs, max_sift, stop_criterion) → NamedTuple

Empirical Mode Decomposition (Huang et al. 1998).
Decomposes non-stationary / non-linear signal into Intrinsic Mode Functions (IMFs).

Each IMF satisfies:
1. Number of extrema and zero-crossings differ by at most 1
2. Upper and lower envelopes are symmetric

# Arguments
- `x`              : signal vector
- `max_imfs`       : maximum number of IMFs to extract (default 8)
- `max_sift`       : maximum sifting iterations (default 20)
- `stop_criterion` : sifting stop criterion (default 0.2, SD-based)

# Returns
NamedTuple: (IMFs, residue, n_imfs, instantaneous_freq)
"""
function EMD(x::Vector{Float64}; max_imfs::Int=8, max_sift::Int=20,
             stop_criterion::Float64=0.2)
    n = length(x)
    IMFs = Vector{Vector{Float64}}()
    residue = copy(x)

    for imf_idx in 1:max_imfs
        h = copy(residue)
        # Sifting process
        for sift in 1:max_sift
            # Find extrema
            maxima_idx = _find_maxima(h)
            minima_idx = _find_minima(h)

            # Need at least 3 extrema to interpolate
            (length(maxima_idx) < 3 || length(minima_idx) < 3) && break

            # Interpolate envelopes with cubic spline
            upper_env = _cubic_interp(maxima_idx, h[maxima_idx], n)
            lower_env = _cubic_interp(minima_idx, h[minima_idx], n)

            mean_env = (upper_env .+ lower_env) ./ 2.0
            h_new = h .- mean_env

            # Stopping criterion: SD of consecutive h values
            SD = sum((h_new[t] - h[t])^2 for t in 1:n) /
                 max(sum(h[t]^2 for t in 1:n), 1e-10)
            h = h_new
            SD < stop_criterion && break
        end

        # Check if h is an IMF
        n_zero = _count_zero_crossings(h)
        n_ext  = length(_find_maxima(h)) + length(_find_minima(h))
        abs(n_zero - n_ext) > 1 && break  # not an IMF → stop

        push!(IMFs, h)
        residue = residue .- h

        # Stopping: residue is monotone
        if length(_find_maxima(residue)) < 2 || length(_find_minima(residue)) < 2
            break
        end
    end

    # Instantaneous frequency via Hilbert transform (finite difference phase)
    inst_freq = Vector{Vector{Float64}}()
    for imf in IMFs
        phase = _hilbert_phase(imf)
        freq  = max.(diff(phase) ./ (2π), 0.0)
        push!(inst_freq, freq)
    end

    return (IMFs=IMFs, residue=residue, n_imfs=length(IMFs),
            instantaneous_freq=inst_freq)
end

"""Find local maxima indices."""
function _find_maxima(x::Vector{Float64})::Vector{Int}
    n = length(x)
    idx = Int[]
    for i in 2:(n-1)
        x[i] > x[i-1] && x[i] > x[i+1] && push!(idx, i)
    end
    return idx
end

"""Find local minima indices."""
function _find_minima(x::Vector{Float64})::Vector{Int}
    n = length(x)
    idx = Int[]
    for i in 2:(n-1)
        x[i] < x[i-1] && x[i] < x[i+1] && push!(idx, i)
    end
    return idx
end

"""Count zero crossings."""
function _count_zero_crossings(x::Vector{Float64})::Int
    n = length(x)
    cnt = 0
    for i in 1:(n-1)
        x[i] * x[i+1] < 0 && (cnt += 1)
    end
    return cnt
end

"""Simple cubic spline interpolation via natural spline."""
function _cubic_interp(xi::Vector{Int}, yi::Vector{Float64}, n::Int)::Vector{Float64}
    m = length(xi)
    m < 2 && return fill(mean(yi), n)

    # Extend endpoints
    x_pts = vcat(1, xi, n)
    y_pts = vcat(yi[1], yi, yi[end])
    m2 = length(x_pts)

    # Linear interpolation (simplified from cubic for robustness)
    out = zeros(n)
    for t in 1:n
        # Find bracket
        j = 1
        while j < m2-1 && x_pts[j+1] < t; j += 1; end
        j = clamp(j, 1, m2-1)
        x0, x1 = x_pts[j], x_pts[j+1]
        y0, y1 = y_pts[j], y_pts[j+1]
        α = x1 == x0 ? 0.5 : (t - x0) / (x1 - x0)
        out[t] = (1-α)*y0 + α*y1
    end
    return out
end

"""Approximate Hilbert transform phase via analytic signal (DFT method)."""
function _hilbert_phase(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    # FFT
    X = _simple_fft(complex.(x))
    # Zero negative frequencies, double positive
    H = zeros(ComplexF64, n)
    H[1] = X[1]
    if iseven(n)
        H[n÷2+1] = X[n÷2+1]
        H[2:(n÷2)] = 2.0 .* X[2:(n÷2)]
    else
        H[2:((n+1)÷2)] = 2.0 .* X[2:((n+1)÷2)]
    end
    # IFFT
    analytic = _simple_ifft(H)
    return angle.(analytic)
end

"""Simple FFT (Cooley-Tukey for power-of-2, else DFT)."""
function _simple_fft(x::Vector{ComplexF64})::Vector{ComplexF64}
    n = length(x)
    n == 1 && return x
    if n & (n-1) != 0
        # DFT fallback
        y = zeros(ComplexF64, n)
        for k in 0:(n-1), j in 0:(n-1)
            y[k+1] += x[j+1] * cis(-2π*k*j/n)
        end
        return y
    end
    even = _simple_fft(x[1:2:end])
    odd  = _simple_fft(x[2:2:end])
    T    = [cis(-2π*(k-1)/n)*odd[k] for k in 1:(n÷2)]
    return vcat(even .+ T, even .- T)
end

function _simple_ifft(X::Vector{ComplexF64})::Vector{ComplexF64}
    n = length(X)
    conj_fft = _simple_fft(conj.(X))
    return conj.(conj_fft) ./ n
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Singular Spectrum Analysis (SSA)
# ─────────────────────────────────────────────────────────────────────────────

"""
    SSA(x; L, n_components) → NamedTuple

Singular Spectrum Analysis for time series decomposition.

Steps:
1. Embedding: form trajectory matrix X of lag-L embeddings
2. SVD decomposition of X
3. Grouping of components (trend + oscillations)
4. Diagonal averaging (Hankelisation) for reconstruction

# Arguments
- `x`            : time series
- `L`            : window length / embedding dimension (default n÷4)
- `n_components` : number of components to retain (default min(L, 10))

# Returns
NamedTuple: (trend, oscillations, noise, singular_values, components,
             variance_explained)
"""
function SSA(x::Vector{Float64}; L::Int=0, n_components::Int=0)
    n = length(x)
    L_eff = L > 0 ? L : n ÷ 4
    L_eff = clamp(L_eff, 2, n÷2)
    K = n - L_eff + 1          # number of lagged windows

    # ── Step 1: Trajectory matrix ──────────────────────────────────────────
    X = zeros(L_eff, K)
    for k in 1:K
        X[:, k] = x[k:(k+L_eff-1)]
    end

    # ── Step 2: SVD ────────────────────────────────────────────────────────
    F_svd = svd(X)
    U, σ, V = F_svd.U, F_svd.S, F_svd.V
    n_comp = n_components > 0 ? min(n_components, length(σ)) : min(length(σ), 10)

    # ── Step 3: Component reconstruction via diagonal averaging ────────────
    reconstructed = zeros(n_comp, n)
    for k in 1:n_comp
        Xi = σ[k] * U[:, k] * V[:, k]'
        # Hankelisation: average along anti-diagonals
        for t in 1:n
            count = 0; s = 0.0
            for i in 1:L_eff
                j = t - i + 1
                1 ≤ j ≤ K && (s += Xi[i, j]; count += 1)
            end
            reconstructed[k, t] = count > 0 ? s/count : 0.0
        end
    end

    # Variance explained
    σ2 = σ.^2
    var_explained = σ2[1:n_comp] ./ max(sum(σ2), 1e-10)

    # Group: first 1-2 components = trend, next = oscillations, rest = noise
    n_trend = min(2, n_comp)
    n_noise = max(0, n_comp - n_trend - 2)

    trend = vec(sum(reconstructed[1:n_trend, :], dims=1))
    osc_end = max(n_trend+1, n_comp - n_noise)
    oscillations = n_trend < n_comp ?
        vec(sum(reconstructed[(n_trend+1):osc_end, :], dims=1)) : zeros(n)
    noise_est = x .- trend .- oscillations

    return (trend=trend, oscillations=oscillations, noise=noise_est,
            singular_values=σ[1:n_comp], components=reconstructed,
            variance_explained=var_explained, L=L_eff)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Savitzky-Golay Filter
# ─────────────────────────────────────────────────────────────────────────────

"""
    SavitzkyGolay(y; window, poly_order, derivative) → NamedTuple

Savitzky-Golay smoothing filter: fits a polynomial of order `poly_order`
to successive sliding windows of `window` data points.
Preserves higher moments (peaks, valleys) better than moving average.

# Arguments
- `y`          : signal vector
- `window`     : window size (must be odd, default 11)
- `poly_order` : polynomial degree (default 3)
- `derivative` : compute k-th derivative (default 0 = smooth only)

# Returns
NamedTuple: (smoothed, derivative_series, coefficients)
"""
function SavitzkyGolay(y::Vector{Float64}; window::Int=11, poly_order::Int=3,
                        derivative::Int=0)
    n = length(y)
    iseven(window) && (window += 1)  # ensure odd
    window = max(window, poly_order + 2)
    half = window ÷ 2

    # Precompute SG coefficients via least squares
    t_vec = Float64.((-half):half)   # window indices
    m = length(t_vec)

    # Vandermonde matrix for polynomial fitting
    V = zeros(m, poly_order+1)
    for j in 0:poly_order
        V[:, j+1] = t_vec .^ j
    end
    # Pseudoinverse: (V'V)^{-1} V'
    VtV = V' * V + 1e-10*I
    coeffs_matrix = VtV \ V'   # (poly_order+1) × m

    # For derivative k: multiply by k! / t^k factors
    if derivative == 0
        sg_coeffs = coeffs_matrix[1, :]   # zeroth-order = smoothing
    elseif derivative == 1
        sg_coeffs = coeffs_matrix[2, :]   # first derivative (linear term)
    elseif derivative == 2
        sg_coeffs = 2.0 .* coeffs_matrix[3, :]
    else
        sg_coeffs = coeffs_matrix[min(derivative+1, poly_order+1), :]
    end

    # Apply filter with edge mirroring
    smoothed = zeros(n)
    deriv_out = zeros(n)

    for i in 1:n
        # Indices for window
        start_idx = max(1, i - half)
        end_idx   = min(n, i + half)

        # Pad with reflection if needed
        segment = zeros(window)
        for k in 1:window
            t_idx = i - half + k - 1
            if t_idx < 1
                seg_idx = 2 - t_idx   # reflect
            elseif t_idx > n
                seg_idx = 2n - t_idx  # reflect
            else
                seg_idx = t_idx
            end
            segment[k] = y[clamp(seg_idx, 1, n)]
        end

        smoothed[i] = dot(sg_coeffs, segment)
    end

    # Derivative series (finite difference if derivative > 0)
    if derivative > 0
        for i in 1:n
            segment = zeros(window)
            for k in 1:window
                t_idx = i - half + k - 1
                seg_idx = clamp(t_idx, 1, n)
                segment[k] = y[seg_idx]
            end
            deriv_out[i] = dot(sg_coeffs, segment)
        end
    end

    return (smoothed=smoothed, derivative_series=derivative > 0 ? deriv_out : smoothed,
            window=window, poly_order=poly_order, half_window=half)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-Level Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_signal_processing(prices; periods_per_day, out_path) → Dict

Apply full signal processing pipeline to crypto price series.

# Arguments
- `prices`          : price vector (daily or intraday)
- `periods_per_day` : for intraday periodicity (default 1 = daily)
- `out_path`        : optional JSON export path

# Returns
Dict with all filter outputs, regime signals, and complexity measures.

# Example
```julia
using Random
rng = Random.MersenneTwister(42)
n = 600
prices = cumsum(randn(rng, n) .* 200) .+ 50_000.0
prices = max.(prices, 1.0)
results = run_signal_processing(prices)
println("HP trend at end: ", results["hp_filter"]["trend"][end])
println("Hurst (SSA): ", results["ssa"]["variance_explained"][1])
```
"""
function run_signal_processing(prices::Vector{Float64};
                                periods_per_day::Int=1,
                                out_path::Union{String,Nothing}=nothing)
    n = length(prices)
    log_prices = log.(max.(prices, 1e-10))
    returns    = diff(log_prices)
    m = length(returns)

    results = Dict{String, Any}()

    # ── Hodrick-Prescott Filter ────────────────────────────────────────────
    @info "HP filter (λ=1600)..."
    hp = HPFilter(log_prices; lambda=1600.0)
    results["hp_filter"] = Dict(
        "trend"         => hp.trend,
        "cycle"         => hp.cycle,
        "cycle_std"     => hp.cycle_std,
        "cycle_autocorr"=> hp.cycle_autocorr
    )

    # ── Christiano-Fitzgerald Filter ───────────────────────────────────────
    @info "CF band-pass filter (6-32 periods)..."
    cf = CFFilter(log_prices; pl=6.0, pu=32.0, drift=true)
    results["cf_filter"] = Dict(
        "cycle"  => cf.cycle,
        "trend"  => cf.trend,
        "pl"     => cf.pl,
        "pu"     => cf.pu,
        "cycle_std" => std(cf.cycle),
        "cycle_mean" => mean(cf.cycle)
    )

    # ── Kalman Filter (local level model) ─────────────────────────────────
    @info "Kalman filter (local level model)..."
    σ_q  = std(returns) * 0.1   # process noise
    σ_r  = std(returns) * 0.9   # observation noise
    F_kf = reshape([1.0], 1, 1)
    H_kf = reshape([1.0], 1, 1)
    Q_kf = reshape([σ_q^2], 1, 1)
    R_kf = reshape([σ_r^2], 1, 1)

    kf = KalmanFilter(log_prices, F_kf, H_kf, Q_kf, R_kf, [log_prices[1]], reshape([1.0], 1, 1))
    ks = KalmanSmoother(kf, F_kf, Q_kf)
    results["kalman_filter"] = Dict(
        "filtered_trend"  => vec(kf.filtered_mean),
        "smoothed_trend"  => vec(ks.smoothed_mean),
        "log_likelihood"  => kf.log_likelihood,
        "innovation_std"  => std(vec(kf.innovations))
    )

    # ── UKF (log-normal dynamics) ─────────────────────────────────────────
    @info "Unscented Kalman filter..."
    if n >= 50
        f_ukf(x) = [x[1] + 0.0001]   # constant drift
        h_ukf(x) = [x[1]]
        Q_ukf = reshape([σ_q^2 * 2], 1, 1)
        R_ukf = reshape([σ_r^2], 1, 1)
        ukf = UnscentedKalmanFilter(log_prices, f_ukf, h_ukf, Q_ukf, R_ukf,
                                      [log_prices[1]], reshape([0.1], 1, 1))
        results["ukf"] = Dict(
            "filtered_trend" => vec(ukf.filtered_mean),
            "log_likelihood" => ukf.log_likelihood
        )
    end

    # ── Particle Filter (stochastic vol state) ────────────────────────────
    @info "Particle filter (stochastic vol model)..."
    if m >= 50 && n >= 50
        # State: [log-vol_t]; observation: log-return^2 (proxy)
        f_pf = x -> [0.95 * x[1] + 0.05 * log(var(returns))]  # AR(1) log-vol
        loglik_pf = (y_t, x_t) -> begin
            h = exp(x_t[1])
            -0.5 * (log(2π * h) + y_t[1]^2 / h)
        end
        Q_pf = reshape([0.05], 1, 1)
        y_pf = reshape(returns, :, 1)   # observed log-returns
        x0_sampler = r -> [log(var(returns))]

        pf = ParticleFilter(y_pf, f_pf, loglik_pf, Q_pf;
                             n_particles=300, x0_sampler=x0_sampler, rng=Random.default_rng())
        results["particle_filter"] = Dict(
            "log_vol_estimate" => vec(pf.filtered_mean),
            "vol_estimate"     => exp.(vec(pf.filtered_mean) ./ 2),
            "avg_ESS"          => mean(pf.ESS),
            "log_likelihood"   => pf.log_likelihood
        )
    end

    # ── EMD / Hilbert-Huang Transform ─────────────────────────────────────
    @info "Empirical Mode Decomposition..."
    n_emd = min(n, 500)  # limit for speed
    emd = EMD(log_prices[1:n_emd]; max_imfs=6, max_sift=10)
    results["emd"] = Dict(
        "n_imfs"         => emd.n_imfs,
        "imf_1_variance" => emd.n_imfs >= 1 ? var(emd.IMFs[1]) : 0.0,
        "residue_trend"  => emd.residue,
        "imf_summary"    => [Dict("idx"=>k, "var"=>var(emd.IMFs[k]),
                                   "mean_freq"=>isempty(emd.instantaneous_freq[k]) ? 0.0 :
                                                mean(emd.instantaneous_freq[k]))
                              for k in 1:emd.n_imfs]
    )

    # ── SSA ────────────────────────────────────────────────────────────────
    @info "Singular Spectrum Analysis..."
    n_ssa = min(n, 500)
    ssa = SSA(log_prices[1:n_ssa]; L=n_ssa÷4, n_components=8)
    results["ssa"] = Dict(
        "trend"              => ssa.trend,
        "oscillations"       => ssa.oscillations,
        "variance_explained" => ssa.variance_explained,
        "L"                  => ssa.L,
        "n_components"       => length(ssa.singular_values),
        "cumvar_top4"        => sum(ssa.variance_explained[1:min(4, length(ssa.variance_explained))])
    )

    # ── Savitzky-Golay ─────────────────────────────────────────────────────
    @info "Savitzky-Golay smoothing..."
    sg = SavitzkyGolay(log_prices; window=21, poly_order=3, derivative=0)
    sg_d1 = SavitzkyGolay(log_prices; window=21, poly_order=3, derivative=1)
    results["savitzky_golay"] = Dict(
        "smoothed"          => sg.smoothed,
        "first_derivative"  => sg_d1.derivative_series,
        "window"            => sg.window,
        "poly_order"        => sg.poly_order,
        "smoothing_residual"=> std(log_prices .- sg.smoothed)
    )

    # ── Regime Signal Extraction ───────────────────────────────────────────
    # Combine signals to classify regimes
    hp_cycle   = hp.cycle
    kf_innov   = vec(kf.innovations)
    sg_slope   = sg_d1.derivative_series

    regime_signal = zeros(n)
    for t in 1:n
        # Combine: positive trend + positive momentum = bull
        slope_sign = sg_slope[t] > 0 ? 1 : -1
        cycle_sign = hp_cycle[t] > 0.01*std(hp_cycle) ? 1 : -1
        regime_signal[t] = 0.5 * slope_sign + 0.5 * cycle_sign
    end

    results["regime_signal"] = Dict(
        "signal"          => regime_signal,
        "bull_fraction"   => mean(regime_signal .> 0),
        "bear_fraction"   => mean(regime_signal .< 0),
        "current_regime"  => regime_signal[end] > 0 ? "bull" : "bear"
    )

    # ── Summary ────────────────────────────────────────────────────────────
    results["summary"] = Dict(
        "n"                  => n,
        "hp_cycle_std"       => std(hp.cycle),
        "cf_cycle_std"       => std(cf.cycle),
        "kf_loglik"          => kf.log_likelihood,
        "emd_n_imfs"         => emd.n_imfs,
        "ssa_top1_var_exp"   => ssa.variance_explained[1],
        "sg_smoothing_err"   => std(log_prices .- sg.smoothed),
        "current_regime"     => regime_signal[end] > 0 ? "bull" : "bear"
    )

    if !isnothing(out_path)
        try
            using JSON3
            open(out_path, "w") do io
                JSON3.write(io, results)
            end
            @info "Signal processing results written to $out_path"
        catch
            @warn "JSON3 not available; skipping JSON export"
        end
    end

    return results
end

end  # module SignalProcessing
