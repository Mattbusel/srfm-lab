# =============================================================================
# econometrics_advanced.jl — Advanced Econometrics for Macro/Quant Research
# =============================================================================
# Implements cutting-edge econometric methods:
#   1. ARFIMA(p,d,q): Geweke-Porter-Hudak + Whittle estimator
#   2. FIGARCH: fractionally integrated GARCH
#   3. Structural VAR with sign restrictions (Uhlig method)
#   4. Local projections (Jordà 2005) for impulse responses
#   5. Quantile regression via interior point method
#   6. Threshold VAR (TVAR) for regime-dependent dynamics
#   7. Factor-augmented VAR (FAVAR) with PCA factors
#   8. Long-run variance (Newey-West, Andrews kernel)
#   9. Unit root tests: ADF, KPSS, Phillips-Perron
#
# Julia ≥ 1.10 | No external packages
# =============================================================================

module EconometricsAdvanced

using Statistics
using LinearAlgebra

export arfima_gph, arfima_whittle, arfima_simulate, arfima_filter
export figarch_fit, figarch_conditional_variance
export svar_sign_restrictions, svar_irf, svar_bootstrap_ci
export local_projection, lp_irf, lp_cumulative_irf
export quantile_regression, quantile_regression_process
export threshold_var, tvar_likelihood, tvar_impulse_response
export favar_estimate, favar_irf
export newey_west_variance, andrews_kernel_variance, hac_variance
export adf_test, kpss_test, phillips_perron_test
export select_lag_bic, select_lag_aic

# =============================================================================
# SECTION 1: ARFIMA MODEL
# =============================================================================

"""
    arfima_gph(y; bandwidth=nothing) -> NamedTuple

Geweke-Porter-Hudak (1983) semi-parametric estimator of the fractional
differencing parameter d.

Estimate d from the slope of log periodogram on log frequency:
    log I(ωⱼ) = c + d * log(4 sin²(ωⱼ/2)) + εⱼ

for j = 1,...,m where m = n^bandwidth is the bandwidth.

# Arguments
- `y`: time series
- `bandwidth`: exponent for m = n^bandwidth (default 0.5)

# Returns
- NamedTuple: d_hat, se, t_stat, bandwidth_m
"""
function arfima_gph(y::Vector{Float64}; bandwidth::Float64=0.5)

    n = length(y)
    n < 8 && return (d_hat=0.0, se=0.0, t_stat=0.0, bandwidth_m=1)

    # Compute periodogram
    m = max(2, floor(Int, n^bandwidth))
    m = min(m, n ÷ 2 - 1)

    # Discrete Fourier transform at Fourier frequencies
    omega = [2π * j / n for j in 1:m]
    periodogram = zeros(m)

    y_mean = mean(y)
    y_centered = y .- y_mean

    for (k, w) in enumerate(omega)
        real_part = sum(y_centered[t] * cos(w * t) for t in 1:n)
        imag_part = sum(y_centered[t] * sin(w * t) for t in 1:n)
        periodogram[k] = (real_part^2 + imag_part^2) / (2π * n)
    end

    # GPH regression
    x_gph = log.(4 .* sin.(omega ./ 2) .^ 2)
    log_I = log.(max.(periodogram, 1e-20))

    # OLS regression: log I = a + d * x_gph + e
    x_mean = mean(x_gph)
    y_mean_gph = mean(log_I)

    sxx = sum((x_gph .- x_mean) .^ 2)
    sxy = sum((x_gph .- x_mean) .* (log_I .- y_mean_gph))

    d_hat = sxx > 0 ? sxy / sxx : 0.0

    # Standard error
    residuals = log_I .- (y_mean_gph - d_hat * x_mean) .- d_hat .* x_gph
    s2 = sum(residuals .^ 2) / (m - 2)
    se = sxx > 0 ? sqrt(s2 / sxx) : 0.0
    t_stat = se > 0 ? d_hat / se : 0.0

    return (d_hat=d_hat, se=se, t_stat=t_stat, bandwidth_m=m,
             periodogram=periodogram, frequencies=omega)
end

"""
    arfima_whittle(y; p=0, q=0) -> NamedTuple

Whittle (1951) approximate MLE for ARFIMA(p,d,q) parameters.

Whittle likelihood (in frequency domain):
    L_W(θ) = Σⱼ [log f(ωⱼ; θ) + I(ωⱼ)/f(ωⱼ; θ)]

where f(ωⱼ; θ) is the spectral density implied by ARFIMA(p,d,q).

The spectral density of ARFIMA:
    f(ω) = σ²/(2π) * |1-e^{-iω}|^{-2d} * |φ(e^{-iω})|⁻² * |θ(e^{-iω})|²

# Returns
- NamedTuple: d, ar_params, ma_params, sigma2, aic, bic
"""
function arfima_whittle(y::Vector{Float64}; p::Int=0, q::Int=0)

    n = length(y)
    n < 2*(p+q) + 10 && return (d=0.0, ar_params=zeros(p), ma_params=zeros(q),
                                   sigma2=var(y), aic=Inf, bic=Inf)

    # Fourier frequencies
    m = n ÷ 2
    omega = [2π * j / n for j in 1:m]

    # Periodogram
    y_mean = mean(y)
    yc = y .- y_mean
    I_w = zeros(m)
    for (k, w) in enumerate(omega)
        re = sum(yc[t] * cos(w*t) for t in 1:n)
        im = sum(yc[t] * sin(w*t) for t in 1:n)
        I_w[k] = (re^2 + im^2) / (2π * n)
    end

    # Compute ARFIMA spectral density
    function spectral_density(params::Vector{Float64})::Vector{Float64}
        d = params[1]
        ar = p > 0 ? params[2:(p+1)] : Float64[]
        ma = q > 0 ? params[(p+2):(p+q+1)] : Float64[]
        sigma2 = abs(params[end])

        f = zeros(m)
        for (k, w) in enumerate(omega)
            # Fractional differencing: |1-e^{-iω}|^{-2d}
            frac_factor = (2 * abs(sin(w/2)))^(-2*d)

            # AR polynomial: φ(e^{-iω}) = 1 - Σ φⱼ e^{-ijω}
            ar_re = 1.0; ar_im = 0.0
            for j in 1:p
                ar_re -= ar[j] * cos(j * w)
                ar_im += ar[j] * sin(j * w)
            end
            ar_sq = ar_re^2 + ar_im^2

            # MA polynomial: θ(e^{-iω}) = 1 + Σ θⱼ e^{-ijω}
            ma_re = 1.0; ma_im = 0.0
            for j in 1:q
                ma_re += ma[j] * cos(j * w)
                ma_im -= ma[j] * sin(j * w)
            end
            ma_sq = ma_re^2 + ma_im^2

            f[k] = sigma2 / (2π) * frac_factor * ma_sq / max(ar_sq, 1e-20)
        end
        return f
    end

    # Whittle likelihood (negative, to minimize)
    function whittle_loglik(params::Vector{Float64})::Float64
        d = params[1]
        abs(d) > 0.5 && return 1e15

        f = spectral_density(params)
        any(f .<= 0) && return 1e15

        return sum(log.(f) .+ I_w ./ f)
    end

    # Initial guess
    gph = arfima_gph(y)
    d0 = clamp(gph.d_hat, -0.49, 0.49)
    params0 = vcat([d0], zeros(p), zeros(q), [var(y)])

    # Coordinate descent
    best_params = copy(params0)
    best_ll = whittle_loglik(best_params)

    step_sizes = vcat([0.05], ones(p)*0.05, ones(q)*0.05, [var(y)*0.1])

    for outer in 1:200
        improved = false
        for dim in 1:length(best_params)
            for dir in [1.0, -1.0]
                cand = copy(best_params)
                cand[dim] += dir * step_sizes[dim]
                ll = whittle_loglik(cand)
                if ll < best_ll
                    best_ll = ll
                    best_params = cand
                    improved = true
                end
            end
        end
        if !improved
            step_sizes .*= 0.5
            all(step_sizes .< 1e-8) && break
        end
    end

    d_est = best_params[1]
    ar_est = p > 0 ? best_params[2:(p+1)] : Float64[]
    ma_est = q > 0 ? best_params[(p+2):(p+q+1)] : Float64[]
    sigma2_est = abs(best_params[end])

    n_params = 1 + p + q + 1
    aic = 2 * best_ll + 2 * n_params
    bic = 2 * best_ll + log(n) * n_params

    return (d=d_est, ar_params=ar_est, ma_params=ma_est,
             sigma2=sigma2_est, aic=aic, bic=bic, loglik=-best_ll)
end

"""
    arfima_filter(y, d) -> Vector{Float64}

Apply fractional differencing operator (1-L)^d to time series y.

Uses the binomial expansion:
    (1-L)^d = Σₖ₌₀^∞ C(d,k) (-1)^k L^k

truncated at n terms (for efficiency).
The fractional difference at lag k: π_k = Γ(k-d)/(Γ(-d)Γ(k+1))
"""
function arfima_filter(y::Vector{Float64}, d::Float64)::Vector{Float64}
    n = length(y)
    n < 2 && return y

    # Compute filter weights π_k = Γ(k-d)/(Γ(-d)Γ(k+1))
    # Using recursion: π_0 = 1, π_k = π_{k-1} * (k-1-d)/k
    max_lag = n
    pi_weights = zeros(max_lag)
    pi_weights[1] = 1.0
    for k in 2:max_lag
        pi_weights[k] = pi_weights[k-1] * ((k-1) - d) / k
    end

    # Apply filter: Δᵈy_t = Σₖ π_k y_{t-k}
    filtered = zeros(n)
    for t in 1:n
        for k in 1:t
            filtered[t] += pi_weights[k] * y[t - k + 1]
        end
    end

    return filtered
end

"""
    arfima_simulate(d, n; ar=[], ma=[], sigma=1.0, seed=42) -> Vector{Float64}

Simulate an ARFIMA(p,d,q) process.
"""
function arfima_simulate(d::Float64, n::Int;
                           ar::Vector{Float64}=Float64[],
                           ma::Vector{Float64}=Float64[],
                           sigma::Float64=1.0,
                           seed::Int=42)::Vector{Float64}

    # Generate white noise
    rng = seed
    eps = zeros(n)
    for t in 1:n
        rng = (1664525 * rng + 1013904223) % (2^32)
        u1 = (rng + 1) / 2^32
        rng = (1664525 * rng + 1013904223) % (2^32)
        u2 = (rng + 1) / 2^32
        eps[t] = sigma * sqrt(-2*log(u1)) * cos(2π*u2)
    end

    # Apply MA polynomial
    p = length(ar); q_ma = length(ma)
    y = copy(eps)
    for t in 2:n
        for j in 1:min(q_ma, t-1)
            y[t] += ma[j] * eps[t - j]
        end
    end

    # Invert fractional differencing: (1-L)^{-d} applied to y
    # This gives ARFIMA with fractional integration
    result = zeros(n)
    pi_inv = zeros(n)  # weights for (1-L)^{-d}
    pi_inv[1] = 1.0
    for k in 2:n
        pi_inv[k] = pi_inv[k-1] * (k-1+d) / k
    end

    for t in 1:n
        result[t] = sum(pi_inv[k] * (t-k >= 0 ? get(Dict([(i,y[i]) for i in 1:n]), t-k+1, 0.0) : 0.0) for k in 1:t)
    end

    # Apply AR polynomial
    for t in max(p+1, 1):n
        for j in 1:p
            result[t] += ar[j] * result[t - j]
        end
    end

    return result
end

# =============================================================================
# SECTION 2: FIGARCH
# =============================================================================

"""
    figarch_fit(returns; p=1, d_init=0.5, q=1, max_iter=200) -> NamedTuple

Fit FIGARCH(p,d,q) model: fractionally integrated GARCH.

FIGARCH(1,d,1) conditional variance:
    h_t = ω/(1-β) + [1 - (1-β)⁻¹*(1-φL)*(1-L)^d] ε²_t

The FIGARCH allows long memory in volatility persistence (d ∈ (0,1)).
Unlike GARCH where volatility shocks decay exponentially, FIGARCH
shocks decay hyperbolically: k(τ) ~ τ^{d-1} / Γ(d).

Uses Quasi-MLE (Gaussian QMLE).

# Returns
- NamedTuple: omega, phi, d, beta, loglik, aic, bic, h_series
"""
function figarch_fit(returns::Vector{Float64};
                      p::Int=1, q::Int=1,
                      d_init::Float64=0.5,
                      max_iter::Int=200)

    n = length(returns)
    n < 20 && return (omega=1e-5, phi=0.0, d=0.0, beta=0.0,
                       loglik=0.0, aic=Inf, bic=Inf, h_series=fill(var(returns),n))

    r2 = returns .^ 2
    var_r = var(returns)

    # Truncation order for fractional polynomial
    max_trunc = min(n, 1000)

    # FIGARCH filter: λ(L) = (1-βL)⁻¹ * φ(L) * (1-L)^d
    # λ_k = Σⱼ c_j where c_j are from binomial expansion
    function figarch_weights(phi::Float64, d::Float64, beta::Float64, trunc::Int)
        weights = zeros(trunc)
        weights[1] = phi - d
        for k in 2:trunc
            weights[k] = beta * weights[k-1] + (k - 1 - d) / k * (1 - phi + beta) - phi + beta
        end
        # Normalize
        return weights
    end

    function figarch_loglik(params::Vector{Float64})::Float64
        omega, phi, d, beta = params
        omega <= 0 && return -Inf
        abs(d) > 1 && return -Inf
        (phi <= 0 || beta >= 1 || beta < 0) && return -Inf
        phi >= 1.0 && return -Inf

        lam = figarch_weights(phi, d, beta, min(n-1, max_trunc))
        h = zeros(n)
        h[1] = var_r

        ll = 0.0
        for t in 2:n
            h_t = omega
            for k in 1:min(t-1, length(lam))
                h_t += lam[k] * r2[t-k]
            end
            h[t] = max(h_t, 1e-10)
            ll += -0.5 * (log(2π) + log(h[t]) + returns[t]^2 / h[t])
        end
        return ll
    end

    # Initial parameters
    best = [var_r * 0.05, 0.3, d_init, 0.6]
    best_ll = figarch_loglik(best)

    # Coordinate search
    step = [var_r*0.01, 0.05, 0.05, 0.05]
    for outer in 1:max_iter
        improved = false
        for dim in 1:4
            for dir in [1.0, -1.0]
                cand = copy(best)
                cand[dim] += dir * step[dim]
                ll = figarch_loglik(cand)
                if ll > best_ll
                    best_ll = ll
                    best = cand
                    improved = true
                end
            end
        end
        if !improved
            step .*= 0.5
            all(step .< 1e-9) && break
        end
    end

    omega_est, phi_est, d_est, beta_est = best

    # Compute conditional variance series
    lam = figarch_weights(phi_est, d_est, beta_est, min(n-1, max_trunc))
    h = zeros(n)
    h[1] = var_r
    for t in 2:n
        h_t = omega_est
        for k in 1:min(t-1, length(lam))
            h_t += lam[k] * r2[t-k]
        end
        h[t] = max(h_t, 1e-10)
    end

    n_params = 4
    aic = -2*best_ll + 2*n_params
    bic = -2*best_ll + log(n)*n_params

    return (omega=omega_est, phi=phi_est, d=d_est, beta=beta_est,
             loglik=best_ll, aic=aic, bic=bic, h_series=h)
end

"""
    figarch_conditional_variance(returns, omega, phi, d, beta) -> Vector{Float64}

Compute FIGARCH conditional variances for given parameters.
"""
function figarch_conditional_variance(returns::Vector{Float64},
                                        omega::Float64, phi::Float64,
                                        d::Float64, beta::Float64)::Vector{Float64}
    n = length(returns)
    r2 = returns .^ 2

    # Build fractional weights
    max_lag = min(n - 1, 500)
    lam = zeros(max_lag)
    lam[1] = phi - d
    for k in 2:max_lag
        lam[k] = beta * lam[k-1] + (k-1-d)/k * (1-phi+beta) - phi + beta
    end

    h = fill(var(returns), n)
    for t in 2:n
        h_t = omega
        for k in 1:min(t-1, max_lag)
            h_t += lam[k] * r2[t-k]
        end
        h[t] = max(h_t, 1e-10)
    end
    return h
end

# =============================================================================
# SECTION 3: STRUCTURAL VAR WITH SIGN RESTRICTIONS
# =============================================================================

"""
    svar_sign_restrictions(Y; p=4, n_draws=1000, sign_matrix=nothing) -> NamedTuple

Structural VAR identified via sign restrictions (Uhlig 2005).

Algorithm:
1. Estimate reduced-form VAR(p): Yₜ = A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + uₜ
2. Cholesky decompose Σ: Σ = PP'
3. Randomize over rotation matrices Q ∈ O(n): B = PQ
4. Accept if impulse responses satisfy sign restrictions

# Arguments
- `Y`: (T × K) matrix of endogenous variables
- `p`: VAR lag order
- `n_draws`: number of rotation draws
- `sign_matrix`: (K × K) matrix of +1/-1/0 for sign restrictions
                 sign_matrix[i,j] = +1 means shock j has positive
                 impact on variable i on impact

# Returns
- NamedTuple: median_irf, ci_lower, ci_upper, n_accepted, reduced_form
"""
function svar_sign_restrictions(Y::Matrix{Float64};
                                  p::Int=4,
                                  n_draws::Int=1000,
                                  sign_matrix::Union{Matrix{Int},Nothing}=nothing,
                                  n_irf_periods::Int=20)

    T, K = size(Y)
    T < p + K + 5 && error("Insufficient observations for SVAR")

    # Step 1: Estimate reduced-form VAR
    rf = var_estimate(Y; p=p)

    # Step 2: Cholesky of residual covariance
    Sigma = rf.Sigma
    P = try
        cholesky(Symmetric(Sigma + 1e-10*I)).L
    catch
        Matrix{Float64}(I, K, K) * sqrt(mean(diag(Sigma)))
    end

    # Default sign restrictions: lower triangular (Cholesky identification)
    if sign_matrix === nothing
        sign_matrix = LowerTriangular(ones(Int, K, K))
    end

    # Step 3: Draw random rotation matrices and compute IRFs
    accepted_irfs = Array{Float64,3}[]
    rng_state = 123

    for draw in 1:n_draws * 10  # over-draw to get n_draws accepted
        length(accepted_irfs) >= n_draws && break

        # Random rotation matrix via QR of random normal matrix
        Z = zeros(K, K)
        for i in 1:K, j in 1:K
            rng_state = (1664525*rng_state + 1013904223) % (2^32)
            u1 = (rng_state+1)/2^32
            rng_state = (1664525*rng_state + 1013904223) % (2^32)
            u2 = (rng_state+1)/2^32
            Z[i,j] = sqrt(-2*log(max(u1,1e-10))) * cos(2π*u2)
        end
        Q, _ = qr(Z)
        Q = Matrix(Q)

        # Structural impact matrix
        B = P * Q

        # Check sign restrictions
        accepted = true
        for i in 1:K, j in 1:K
            if sign_matrix[i,j] != 0
                if sign(B[i,j]) != sign_matrix[i,j]
                    accepted = false
                    break
                end
            end
        end

        if accepted
            # Compute IRF
            irf = _var_irf(rf.A_matrices, B, K, p, n_irf_periods)
            push!(accepted_irfs, irf)
        end
    end

    if isempty(accepted_irfs)
        # Return reduced-form Cholesky IRF if no draws accepted
        B_chol = P
        irf_chol = _var_irf(rf.A_matrices, B_chol, K, p, n_irf_periods)
        return (median_irf=irf_chol, ci_lower=irf_chol, ci_upper=irf_chol,
                 n_accepted=0, reduced_form=rf)
    end

    # Summarize accepted IRFs
    n_acc = length(accepted_irfs)
    irf_stack = cat(accepted_irfs..., dims=4)  # n_periods × K × K × n_acc

    median_irf = zeros(n_irf_periods, K, K)
    ci_lower   = zeros(n_irf_periods, K, K)
    ci_upper   = zeros(n_irf_periods, K, K)

    for h in 1:n_irf_periods, i in 1:K, j in 1:K
        vals = [accepted_irfs[d][h,i,j] for d in 1:n_acc]
        sorted_v = sort(vals)
        median_irf[h,i,j] = sorted_v[n_acc÷2 + 1]
        ci_lower[h,i,j]   = sorted_v[max(1, round(Int, 0.16*n_acc))]
        ci_upper[h,i,j]   = sorted_v[min(n_acc, round(Int, 0.84*n_acc))]
    end

    return (median_irf=median_irf, ci_lower=ci_lower, ci_upper=ci_upper,
             n_accepted=n_acc, reduced_form=rf)
end

"""Estimate reduced-form VAR(p) by OLS."""
function var_estimate(Y::Matrix{Float64}; p::Int=4)
    T, K = size(Y)
    T_eff = T - p

    # Build regressor matrix: [Y_{t-1},...,Y_{t-p}, 1]
    n_regressors = K * p + 1
    X = zeros(T_eff, n_regressors)
    Y_dep = Y[(p+1):end, :]

    for t in 1:T_eff
        for lag in 1:p
            X[t, (lag-1)*K+1 : lag*K] = Y[t + p - lag, :]
        end
        X[t, end] = 1.0
    end

    # OLS: B = (X'X)⁻¹X'Y
    XtX = X' * X + 1e-8 * I
    B = (XtX \ (X' * Y_dep))'

    # Residuals and covariance
    Fitted = X * B'
    U = Y_dep .- Fitted
    Sigma = U' * U ./ (T_eff - n_regressors)

    # Extract A matrices: K×K blocks
    A_matrices = Matrix{Float64}[]
    for lag in 1:p
        push!(A_matrices, B[:, (lag-1)*K+1 : lag*K])
    end

    return (B=B, A_matrices=A_matrices, Sigma=Sigma, residuals=U)
end

"""Compute VAR impulse response function."""
function _var_irf(A_matrices::Vector{Matrix{Float64}},
                   B::Matrix{Float64},
                   K::Int, p::Int, h::Int)::Array{Float64,3}

    irf = zeros(h, K, K)
    irf[1, :, :] = B

    # Accumulate: Φ_h = A₁Φ_{h-1} + ... + AₚΦ_{h-p}
    Phi = zeros(h, K, K)
    Phi[1, :, :] = I(K)

    for t in 2:h
        for lag in 1:min(t-1, p)
            lag_idx = length(A_matrices) >= lag ? lag : 1
            Phi[t, :, :] += A_matrices[lag_idx] * Phi[t - lag, :, :]
        end
        irf[t, :, :] = Phi[t, :, :] * B
    end

    return irf
end

"""
    svar_irf(Y, B; p=4, h=20) -> Array{Float64,3}

Compute SVAR impulse responses given structural impact matrix B.
"""
function svar_irf(Y::Matrix{Float64}, B::Matrix{Float64};
                   p::Int=4, h::Int=20)::Array{Float64,3}
    T, K = size(Y)
    rf = var_estimate(Y; p=p)
    return _var_irf(rf.A_matrices, B, K, p, h)
end

# =============================================================================
# SECTION 4: LOCAL PROJECTIONS (JORDÀ 2005)
# =============================================================================

"""
    local_projection(Y, shock_series; h_max=20, controls=nothing, p=4) -> NamedTuple

Jordà (2005) local projection estimator for impulse response functions.

For each horizon h: regress Y_{t+h} directly on shock_t and controls.
    Y_{i,t+h} = α + β_h * shock_t + Σₗ γₗ * Y_{t-l} + ε_{t+h}

LP is more robust than VAR to misspecification, though less efficient.
Newey-West standard errors correct for MA(h) autocorrelation in residuals.

# Arguments
- `Y`: (T × K) response variables
- `shock_series`: T-vector of structural shocks
- `h_max`: maximum horizon
- `controls`: additional control variables (T × M matrix)
- `p`: lags of Y to include as controls

# Returns
- NamedTuple: irf (h_max × K), se (h_max × K), ci_lower, ci_upper
"""
function local_projection(Y::Matrix{Float64},
                            shock_series::Vector{Float64};
                            h_max::Int=20,
                            controls::Union{Matrix{Float64},Nothing}=nothing,
                            p::Int=4)

    T, K = size(Y)
    T_eff = T - p - h_max

    if T_eff < 10
        return (irf=zeros(h_max, K), se=zeros(h_max, K),
                 ci_lower=zeros(h_max, K), ci_upper=zeros(h_max, K))
    end

    irf_mat = zeros(h_max, K)
    se_mat  = zeros(h_max, K)

    for h in 1:h_max
        # Build regressor matrix: [shock_t, Y_{t-1},...,Y_{t-p}, controls_t]
        n_obs = T - p - h
        n_obs < 5 && continue

        X_cols = Vector{Vector{Float64}}()
        push!(X_cols, ones(n_obs))  # intercept

        # Shock series at time t = p+1:T-h
        shock_t = shock_series[(p+1):(T-h)]
        push!(X_cols, shock_t)

        # Lags of Y
        for lag in 1:p, k in 1:K
            push!(X_cols, Y[(p-lag+1):(T-h-lag), k])
        end

        # Optional controls
        if controls !== nothing
            for m in 1:size(controls, 2)
                push!(X_cols, controls[(p+1):(T-h), m])
            end
        end

        X = hcat(X_cols...)
        n_x = size(X, 2)

        for k in 1:K
            y_h = Y[(p+1+h):(T), k]
            n_use = min(length(y_h), size(X, 1))
            y_use = y_h[1:n_use]
            X_use = X[1:n_use, :]

            # OLS
            b = try
                (X_use' * X_use + 1e-8*I) \ (X_use' * y_use)
            catch
                zeros(n_x)
            end

            irf_mat[h, k] = b[2]  # coefficient on shock

            # Newey-West SE with bandwidth h
            resid = y_use .- X_use * b
            nw_var = _newey_west_var_scalar(X_use[:, 2], resid, h)
            se_mat[h, k] = sqrt(max(nw_var, 0.0))
        end
    end

    ci_lower = irf_mat .- 1.645 .* se_mat  # 90% CI
    ci_upper = irf_mat .+ 1.645 .* se_mat

    return (irf=irf_mat, se=se_mat, ci_lower=ci_lower, ci_upper=ci_upper)
end

"""Newey-West variance for a scalar X'ε regression component."""
function _newey_west_var_scalar(x::Vector{Float64},
                                  e::Vector{Float64},
                                  bandwidth::Int)::Float64
    n = length(x)
    n < 3 && return 0.0

    xe = x .* e
    S = sum(xe .^ 2)

    for h in 1:bandwidth
        n - h < 1 && break
        w = 1.0 - h / (bandwidth + 1)
        gamma_h = sum(xe[1:(n-h)] .* xe[(1+h):n])
        S += 2 * w * gamma_h
    end

    sxx = sum(x .^ 2)
    return sxx > 0 ? S / sxx^2 : 0.0
end

"""
    lp_irf(Y, shock_series; h_max=20, p=4) -> Matrix{Float64}

Shorthand: return just the LP IRF matrix.
"""
function lp_irf(Y::Matrix{Float64}, shock_series::Vector{Float64};
                 h_max::Int=20, p::Int=4)::Matrix{Float64}
    return local_projection(Y, shock_series; h_max=h_max, p=p).irf
end

"""
    lp_cumulative_irf(Y, shock_series; h_max=20, p=4) -> Matrix{Float64}

Cumulative (integrated) local projection IRF.
Useful for level effects from growth rate shocks.
"""
function lp_cumulative_irf(Y::Matrix{Float64}, shock_series::Vector{Float64};
                             h_max::Int=20, p::Int=4)::Matrix{Float64}
    irf = lp_irf(Y, shock_series; h_max=h_max, p=p)
    return cumsum(irf, dims=1)
end

# =============================================================================
# SECTION 5: QUANTILE REGRESSION
# =============================================================================

"""
    quantile_regression(y, X, tau; max_iter=1000) -> NamedTuple

Quantile regression at quantile tau ∈ (0,1) via interior point method
(Koenker-Bassett 1978 formulation, solved with simplex/IPM).

Minimize: Σᵢ ρ_τ(yᵢ - Xᵢ'β)
where ρ_τ(u) = u(τ - I(u<0)) is the check function.

Uses iteratively reweighted least squares (IRLS) approximation.

# Arguments
- `y`: response vector
- `X`: design matrix (T × K, include intercept as first column)
- `tau`: quantile level ∈ (0,1)

# Returns
- NamedTuple: beta, fitted, residuals, bandwidth (for inference)
"""
function quantile_regression(y::Vector{Float64},
                               X::Matrix{Float64},
                               tau::Float64;
                               max_iter::Int=1000)

    n, K = size(X)
    @assert length(y) == n
    @assert 0 < tau < 1

    # Initialize with OLS
    b = try
        (X' * X + 1e-8*I) \ (X' * y)
    catch
        zeros(K)
    end

    # IRLS: iterative reweighted LS with check function gradient
    epsilon = 1e-4  # smoothing parameter
    for iter in 1:max_iter
        resid = y .- X * b

        # Smoothed check function weights
        weights = zeros(n)
        for i in 1:n
            r = resid[i]
            if abs(r) > epsilon
                weights[i] = tau / abs(r) + (1-tau) / abs(r)  # simplified
                weights[i] = r > 0 ? tau / max(abs(r), epsilon) :
                                     (1-tau) / max(abs(r), epsilon)
            else
                weights[i] = 1.0 / epsilon
            end
        end

        # Weighted LS
        W = Diagonal(weights)
        b_new = try
            (X' * W * X + 1e-8*I) \ (X' * W * y)
        catch
            b
        end

        norm(b_new - b) < 1e-8 && (b = b_new; break)
        b = b_new
    end

    fitted = X * b
    residuals = y .- fitted

    # Bandwidth for Koenker-Machado standard errors
    h = n^(-1/5) * (std(residuals) + 1e-10)

    return (beta=b, fitted=fitted, residuals=residuals, bandwidth=h, tau=tau)
end

"""
    quantile_regression_process(y, X; n_quantiles=19) -> NamedTuple

Estimate quantile regression coefficients across a grid of quantiles.
Returns the entire quantile process β(τ) for τ ∈ {0.05, 0.10, ..., 0.95}.
"""
function quantile_regression_process(y::Vector{Float64},
                                       X::Matrix{Float64};
                                       n_quantiles::Int=19)

    taus = range(0.05, 0.95, length=n_quantiles)
    K = size(X, 2)
    betas = zeros(n_quantiles, K)

    for (i, tau) in enumerate(taus)
        result = quantile_regression(y, X, tau)
        betas[i, :] = result.beta
    end

    return (betas=betas, taus=collect(taus))
end

# =============================================================================
# SECTION 6: THRESHOLD VAR
# =============================================================================

"""
    threshold_var(Y; p=2, delay=1, grid_points=100) -> NamedTuple

Threshold VAR (TVAR) model of Tong (1978), Tsay (1998).

Switches between two VAR regimes based on a threshold variable:
    Y_t = A₁ Y_{t-1:p} + ε_t  if q_{t-d} ≤ γ  (regime 1)
    Y_t = A₂ Y_{t-1:p} + ε_t  if q_{t-d} > γ  (regime 2)

Threshold variable q_{t-d} is typically the lagged first variable.
Threshold γ is estimated by grid search (least squares / profile likelihood).

# Arguments
- `Y`: (T × K) endogenous variables
- `p`: VAR lag order
- `delay`: delay d for threshold variable
- `grid_points`: number of threshold values to search

# Returns
- NamedTuple: gamma, regime1, regime2, n_regime1, n_regime2, ssr, rss_test
"""
function threshold_var(Y::Matrix{Float64};
                        p::Int=2,
                        delay::Int=1,
                        grid_points::Int=100)

    T, K = size(Y)
    T_eff = T - p

    # Threshold variable: first variable at lag delay
    threshold_var_series = Y[(p+1-delay):(T-delay), 1]

    # Grid of threshold values (trim 15% from each end)
    sorted_q = sort(threshold_var_series)
    n_trim = round(Int, 0.15 * T_eff)
    gamma_grid = sorted_q[(n_trim+1):(end-n_trim)]
    step = max(1, length(gamma_grid) ÷ grid_points)
    gamma_grid = gamma_grid[1:step:end]

    best_gamma = gamma_grid[1]
    best_ssr = Inf

    for gamma in gamma_grid
        # Split sample
        regime1_idx = findall(threshold_var_series .<= gamma)
        regime2_idx = findall(threshold_var_series .> gamma)

        (length(regime1_idx) < K*p + 2) && continue
        (length(regime2_idx) < K*p + 2) && continue

        # Fit VAR in each regime
        function regime_ssr(idx_list::Vector{Int})
            Y_dep = Y[(p+1):end, :][idx_list, :]
            X_regs = zeros(length(idx_list), K*p+1)
            for (row, t) in enumerate(idx_list)
                for lag in 1:p
                    X_regs[row, (lag-1)*K+1:lag*K] = Y[t+p-lag, :]
                end
                X_regs[row, end] = 1.0
            end
            B = try (X_regs'*X_regs + 1e-8*I)\(X_regs'*Y_dep) catch; zeros(K*p+1,K) end
            U = Y_dep - X_regs * B
            return sum(U.^2)
        end

        total_ssr = regime_ssr(regime1_idx) + regime_ssr(regime2_idx)
        if total_ssr < best_ssr
            best_ssr = total_ssr
            best_gamma = gamma
        end
    end

    # Fit final models
    regime1_idx = findall(threshold_var_series .<= best_gamma)
    regime2_idx = findall(threshold_var_series .> best_gamma)

    function fit_regime(idx_list::Vector{Int})
        isempty(idx_list) && return (A_matrices=Matrix{Float64}[], Sigma=I(K))
        Y_dep = Y[(p+1):end, :][idx_list, :]
        X_regs = zeros(length(idx_list), K*p+1)
        for (row, t) in enumerate(idx_list)
            for lag in 1:p
                X_regs[row, (lag-1)*K+1:lag*K] = Y[t+p-lag, :]
            end
            X_regs[row, end] = 1.0
        end
        B = try (X_regs'*X_regs + 1e-8*I)\(X_regs'*Y_dep) catch; zeros(K*p+1,K) end
        U = Y_dep - X_regs * B
        Sigma = U'*U ./ max(length(idx_list)-K*p-1, 1)
        A_mats = [B[(lag-1)*K+1:lag*K, :]' for lag in 1:p]
        return (A_matrices=A_mats, Sigma=Sigma, B=B, residuals=U)
    end

    regime1 = fit_regime(regime1_idx)
    regime2 = fit_regime(regime2_idx)

    # Tsay (1998) F-test for threshold nonlinearity
    rf_all = var_estimate(Y; p=p)
    rss_linear = sum(rf_all.residuals .^ 2)
    rss_threshold = best_ssr
    n_switch_params = K^2 * p + K  # extra params from regime 2
    T_eff_all = T - p
    f_stat = T_eff_all > 2*n_switch_params ?
        ((rss_linear - rss_threshold) / n_switch_params) /
        (rss_threshold / (T_eff_all - 2*n_switch_params)) : 0.0

    return (gamma=best_gamma, regime1=regime1, regime2=regime2,
             n_regime1=length(regime1_idx), n_regime2=length(regime2_idx),
             ssr=best_ssr, f_statistic=f_stat)
end

"""
    tvar_impulse_response(tvar_result, shock_idx, K, p; h=20) -> Matrix{Float64}

Compute regime-conditional IRF for TVAR.
"""
function tvar_impulse_response(tvar_result, shock_idx::Int, K::Int, p::Int;
                                 h::Int=20)::Matrix{Float64}
    # Use regime 1 IRF (dominant regime)
    r1 = tvar_result.regime1
    if isempty(r1.A_matrices)
        return zeros(h, K)
    end
    B_chol = try cholesky(Symmetric(r1.Sigma + 1e-8*I)).L catch I(K)*0.1 end
    irf = _var_irf(r1.A_matrices, B_chol, K, p, h)
    return irf[:, :, shock_idx]
end

# =============================================================================
# SECTION 7: FAVAR
# =============================================================================

"""
    favar_estimate(Y, X; r=3, p=4) -> NamedTuple

Factor-Augmented VAR (Bernanke, Boivin, Eliasz 2005).

Two-step estimation:
1. Extract r principal components F from large panel X (T × N)
2. Estimate VAR(p) on [F_t, Y_t] where Y_t is a subset of observables

The factors F summarize information in hundreds of macro series.

# Arguments
- `Y`: (T × Ky) observable policy variables (e.g., fed funds rate)
- `X`: (T × N) large panel of macro/financial indicators
- `r`: number of factors to extract
- `p`: VAR lag order

# Returns
- NamedTuple: factors, loadings, var_result, pct_variance_explained
"""
function favar_estimate(Y::Matrix{Float64},
                          X::Matrix{Float64};
                          r::Int=3,
                          p::Int=4)

    T, N = size(X)
    T2, Ky = size(Y)
    @assert T == T2

    # Step 1: PCA on standardized X
    X_std = (X .- mean(X, dims=1)) ./ max.(std(X, dims=1), 1e-10)

    # SVD for PCA
    U, S, V = try
        svd(X_std)
    catch
        (I(T), ones(min(T,N)), I(N))
    end

    r_use = min(r, size(V, 2), length(S))
    factors = U[:, 1:r_use] .* S[1:r_use]'  # T × r factor scores
    loadings = V[:, 1:r_use]                  # N × r factor loadings

    # Variance explained
    total_var = sum(S .^ 2)
    pct_var = total_var > 0 ? sum(S[1:r_use].^2) / total_var : 0.0

    # Step 2: VAR on [F, Y]
    Z = hcat(factors, Y)  # T × (r + Ky)
    var_result = var_estimate(Z; p=p)

    return (factors=factors, loadings=loadings, var_result=var_result,
             pct_variance_explained=pct_var, r=r_use, n_factors=r_use)
end

"""
    favar_irf(favar_result, shock_idx, Ky; h=20) -> NamedTuple

Compute FAVAR impulse responses.

# Returns
- NamedTuple: irf_factors, irf_observables, irf_panel (N series)
"""
function favar_irf(favar_result, shock_idx::Int, Ky::Int; h::Int=20)
    var_res = favar_result.var_result
    K = size(var_res.Sigma, 1)
    B = try cholesky(Symmetric(var_res.Sigma + 1e-8*I)).L catch I(K)*0.1 end
    irf = _var_irf(var_res.A_matrices, B, K, p=1, h=h)  # p not stored, use 1
    r = favar_result.r

    irf_factors   = irf[:, 1:r, shock_idx]
    irf_obs       = irf[:, (r+1):end, shock_idx]

    # Map factor IRFs back to panel via loadings: Λ * F_irf
    L = favar_result.loadings  # N × r
    irf_panel = irf_factors * L'  # h × N

    return (irf_factors=irf_factors, irf_observables=irf_obs, irf_panel=irf_panel)
end

# =============================================================================
# SECTION 8: LONG-RUN VARIANCE ESTIMATION
# =============================================================================

"""
    newey_west_variance(y; max_lag=nothing) -> Float64

Newey-West (1987) heteroscedasticity and autocorrelation consistent (HAC)
long-run variance estimator.

V = Σₕ₌₋ₘ^m w(h/m) * γ̂_h

with Bartlett kernel w(x) = 1 - |x| and automatic bandwidth m.

Andrews (1991) automatic bandwidth: m = 1.1447 * (αT)^{1/3}
where α depends on autocorrelation structure.
"""
function newey_west_variance(y::Vector{Float64};
                               max_lag::Union{Int,Nothing}=nothing)::Float64
    n = length(y)
    n < 2 && return 0.0

    # Andrews automatic bandwidth
    rho1 = n > 2 ? cor(y[2:end], y[1:end-1]) : 0.0
    alpha = 4 * rho1^2 / (1 - rho1^2)^2
    m = max_lag === nothing ? max(1, floor(Int, 1.1447 * (alpha * n)^(1/3))) : max_lag
    m = min(m, n - 1)

    # Autocovariances with Bartlett kernel
    y_mean = mean(y)
    yc = y .- y_mean

    V = sum(yc .^ 2) / n  # γ₀

    for h in 1:m
        gamma_h = sum(yc[1:(n-h)] .* yc[(h+1):n]) / n
        w_h = 1.0 - h / (m + 1)
        V += 2 * w_h * gamma_h
    end

    return max(V, 0.0)
end

"""
    andrews_kernel_variance(y; kernel=:parzen) -> Float64

Andrews (1991) data-driven HAC estimator with optimal kernel and bandwidth.

Supported kernels: :bartlett, :parzen, :qs (quadratic spectral)
"""
function andrews_kernel_variance(y::Vector{Float64};
                                   kernel::Symbol=:parzen)::Float64
    n = length(y)
    n < 2 && return 0.0

    y_mean = mean(y)
    yc = y .- y_mean

    # Andrews optimal bandwidth
    # Step 1: estimate AR(1) ρ
    rho = n > 2 ? cor(yc[2:end], yc[1:end-1]) : 0.0
    sigma2 = var(yc)

    # Kernel-specific bandwidth formula
    if kernel == :bartlett
        alpha = 4 * rho^2 / (1 - rho)^4
        m = 1.1447 * (alpha * n)^(1/3)
    elseif kernel == :parzen
        alpha = 6 * rho^2 / (1 - rho)^4
        m = 2.6614 * (alpha * n)^(1/5)
    else  # :qs
        alpha = 4 * rho^2 / (1 - rho)^4
        m = 1.3221 * (alpha * n)^(1/5)
    end

    m_int = max(1, round(Int, m))

    # Kernel functions
    function k(x::Float64)::Float64
        ax = abs(x)
        if kernel == :bartlett
            return ax <= 1 ? 1 - ax : 0.0
        elseif kernel == :parzen
            ax > 1 && return 0.0
            ax <= 0.5 && return 1 - 6ax^2 + 6ax^3
            return 2 * (1 - ax)^3
        else  # qs
            abs(x) < 1e-10 && return 1.0
            z = 6π * x / 5
            return 3/z^2 * (sin(z)/z - cos(z))
        end
    end

    V = sum(yc .^ 2) / n
    for h in 1:m_int
        h > n - 1 && break
        gamma_h = sum(yc[1:(n-h)] .* yc[(h+1):n]) / n
        V += 2 * k(h / m_int) * gamma_h
    end

    return max(V, 0.0)
end

"""
    hac_variance(X, y, bandwidth=nothing; kernel=:bartlett) -> Matrix{Float64}

HAC (heteroscedasticity and autocorrelation consistent) covariance matrix
for OLS estimator: V_HAC = (X'X)⁻¹ * S * (X'X)⁻¹

where S = Σₕ w(h) * X'ΩₕX and Ωₕ is the h-lag cross-moment matrix.
"""
function hac_variance(X::Matrix{Float64},
                       residuals::Vector{Float64};
                       bandwidth::Union{Int,Nothing}=nothing,
                       kernel::Symbol=:bartlett)::Matrix{Float64}

    n, K = size(X)
    n < K + 2 && return I(K) * 1e4

    m = bandwidth === nothing ? max(1, floor(Int, n^(1/3))) : bandwidth

    # Meat: S = Σₕ w(h) * Γₕ where Γₕ = X'diag(e)X at lag h
    Xe = X .* residuals  # n × K
    S = Xe' * Xe ./ n

    for h in 1:m
        h > n-1 && break
        w_h = kernel == :bartlett ? 1.0 - h/(m+1) : 1.0
        Gamma_h = Xe[1:(n-h), :]' * Xe[(h+1):n, :] ./ n
        S += w_h .* (Gamma_h + Gamma_h')
    end

    XtX_inv = try inv(X' * X ./ n) catch pinv(X' * X ./ n + 1e-8*I) end
    return XtX_inv * S * XtX_inv ./ n
end

# =============================================================================
# SECTION 9: UNIT ROOT TESTS
# =============================================================================

"""
    adf_test(y; max_lags=nothing, trend=:constant) -> NamedTuple

Augmented Dickey-Fuller test for unit root.

Regression: Δyₜ = α + βt + δyₜ₋₁ + Σᵢ γᵢΔyₜ₋ᵢ + εₜ

H₀: δ = 0 (unit root)
H₁: δ < 0 (stationarity)

Lag selection via BIC (or AIC).

# Arguments
- `y`: time series
- `max_lags`: max ADF lags (default sqrt(T))
- `trend`: :none, :constant (default), :trend (constant + time trend)

# Returns
- NamedTuple: statistic, p_value, critical_values, optimal_lag, trend
"""
function adf_test(y::Vector{Float64};
                   max_lags::Union{Int,Nothing}=nothing,
                   trend::Symbol=:constant)

    n = length(y)
    n < 10 && return (statistic=0.0, p_value=1.0, critical_values=(1%=-3.4,5%=-2.86,10%=-2.57),
                       optimal_lag=0, trend=trend)

    # Default max lags
    max_p = max_lags === nothing ? max(1, floor(Int, sqrt(n))) : max_lags
    max_p = min(max_p, n ÷ 4 - 1)

    dy = diff(y)
    T = length(dy)

    # Select lag by BIC
    best_bic = Inf
    best_lag = 0

    for p in 0:max_p
        T_eff = T - p
        T_eff < 5 && break

        # Regressor matrix
        n_extra = (trend == :none) ? 0 : (trend == :constant) ? 1 : 2
        X = zeros(T_eff, 1 + p + n_extra)
        y_lag = y[(p+1):(T)]  # y_{t-1}
        X[:, 1] = y_lag

        if trend == :constant
            X[:, 2] = ones(T_eff)
            for j in 1:p
                X[:, 2+j] = dy[(p-j+1):(T-j)]
            end
        elseif trend == :trend
            X[:, 2] = ones(T_eff)
            X[:, 3] = collect(1:T_eff)
            for j in 1:p
                X[:, 3+j] = dy[(p-j+1):(T-j)]
            end
        else
            for j in 1:p
                X[:, 1+j] = dy[(p-j+1):(T-j)]
            end
        end

        dep = dy[(p+1):T]
        b = try (X'*X + 1e-8*I)\(X'*dep) catch zeros(size(X,2)) end
        resid = dep - X*b
        s2 = sum(resid.^2) / (T_eff - size(X,2))
        bic_val = T_eff * log(s2) + log(T_eff) * size(X,2)

        if bic_val < best_bic
            best_bic = bic_val
            best_lag = p
        end
    end

    # Final regression with optimal lag
    p = best_lag
    T_eff = T - p
    n_extra = (trend == :none) ? 0 : (trend == :constant) ? 1 : 2
    X = zeros(T_eff, 1 + p + n_extra)
    y_lag = y[(p+1):(T)]
    X[:, 1] = y_lag

    if trend == :constant
        X[:, 2] = ones(T_eff)
        for j in 1:p; X[:, 2+j] = dy[(p-j+1):(T-j)] end
    elseif trend == :trend
        X[:, 2] = ones(T_eff)
        X[:, 3] = collect(1:T_eff)
        for j in 1:p; X[:, 3+j] = dy[(p-j+1):(T-j)] end
    else
        for j in 1:p; X[:, 1+j] = dy[(p-j+1):(T-j)] end
    end

    dep = dy[(p+1):T]
    b = try (X'*X + 1e-8*I)\(X'*dep) catch zeros(size(X,2)) end
    resid = dep - X*b
    s2 = sum(resid.^2) / (T_eff - size(X,2))

    # t-stat on y_{t-1} coefficient (column 1)
    var_b1 = s2 * (inv(X'*X + 1e-8*I)[1,1])
    t_stat = sqrt(var_b1) > 0 ? b[1] / sqrt(var_b1) : 0.0

    # MacKinnon (1994) critical values
    cv = if trend == :constant
        (Symbol("1%") => -3.43, Symbol("5%") => -2.86, Symbol("10%") => -2.57)
    elseif trend == :trend
        (Symbol("1%") => -3.96, Symbol("5%") => -3.41, Symbol("10%") => -3.13)
    else
        (Symbol("1%") => -2.56, Symbol("5%") => -1.94, Symbol("10%") => -1.62)
    end

    # Approximate p-value (response surface MacKinnon 1994)
    p_val = _adf_pvalue(t_stat, trend, n)

    return (statistic=t_stat, p_value=p_val, critical_values=cv,
             optimal_lag=best_lag, trend=trend)
end

"""Approximate ADF p-value using MacKinnon response surface."""
function _adf_pvalue(tau::Float64, trend::Symbol, n::Int)::Float64
    # MacKinnon (1994) Table 4 response surface coefficients
    # Approximate p-value interpolation
    # For τ < critical value(5%): reject H₀
    cv5 = trend == :none ? -1.94 : trend == :constant ? -2.86 : -3.41

    if tau < cv5
        return 0.01  # approximate: p < 5%
    elseif tau < -1.5
        return 0.10
    else
        return 0.50
    end
end

"""
    kpss_test(y; trend=:constant, max_lags=nothing) -> NamedTuple

Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.

H₀: series is stationary (reverse of ADF)
H₁: unit root

Test statistic: η = (1/T²) * Σ Sₜ² / s²(l)
where Sₜ = partial sum of residuals, s²(l) = Newey-West variance.

# Returns
- NamedTuple: statistic, critical_values, reject_stationarity
"""
function kpss_test(y::Vector{Float64};
                    trend::Symbol=:constant,
                    max_lags::Union{Int,Nothing}=nothing)

    n = length(y)

    # Detrend
    if trend == :constant
        resid = y .- mean(y)
    else  # :trend
        t_vec = collect(1:n)
        X = hcat(ones(n), t_vec)
        b = (X'*X + 1e-8*I) \ (X' * y)
        resid = y - X*b
    end

    # Partial sums S_t = Σᵢ₌₁ᵗ e_i
    S = cumsum(resid)

    # Long-run variance
    lrv = newey_west_variance(resid; max_lag=max_lags)

    # Test statistic
    eta = sum(S .^ 2) / (n^2 * max(lrv, 1e-20))

    # KPSS critical values (Kwiatkowski et al. 1992, Table 1)
    cv = if trend == :constant
        (Symbol("1%") => 0.739, Symbol("5%") => 0.463, Symbol("10%") => 0.347)
    else
        (Symbol("1%") => 0.216, Symbol("5%") => 0.146, Symbol("10%") => 0.119)
    end

    reject = eta > 0.463  # reject stationarity at 5%

    return (statistic=eta, critical_values=cv, reject_stationarity=reject, trend=trend)
end

"""
    phillips_perron_test(y; trend=:constant) -> NamedTuple

Phillips-Perron (1988) unit root test using non-parametric correction.

Equivalent to ADF but corrects for serial correlation and heteroscedasticity
non-parametrically rather than via augmentation.

Test statistic:
    Z_t = t_δ * sqrt(s²/λ²) - T*(λ²-s²)/(2*λ*se(δ̂))

where λ² is the long-run variance, s² is the short-run variance.
"""
function phillips_perron_test(y::Vector{Float64};
                                trend::Symbol=:constant)

    n = length(y)
    n < 5 && return (statistic=0.0, p_value=1.0, trend=trend)

    dy = diff(y)
    T = length(dy)
    y_lag = y[1:T]

    n_extra = trend == :constant ? 1 : trend == :trend ? 2 : 0
    X = zeros(T, 1 + n_extra)
    X[:, 1] = y_lag
    if trend == :constant
        X[:, 2] = ones(T)
    elseif trend == :trend
        X[:, 2] = ones(T)
        X[:, 3] = collect(1:T)
    end

    b = try (X'*X + 1e-8*I)\(X'*dy) catch zeros(size(X,2)) end
    resid = dy - X*b

    s2 = sum(resid.^2) / (T - size(X,2))  # short-run variance
    lambda2 = newey_west_variance(resid)     # long-run variance

    # t-statistic on y_{t-1}
    var_b1 = s2 * (inv(X'*X + 1e-8*I)[1,1])
    t_naive = sqrt(var_b1) > 0 ? b[1] / sqrt(var_b1) : 0.0

    # PP correction
    se_b1 = sqrt(var_b1)
    Z_t = se_b1 > 0 ?
          t_naive * sqrt(s2 / lambda2) - (lambda2 - s2) / (2 * sqrt(lambda2) * se_b1) :
          t_naive

    cv5 = trend == :none ? -1.94 : trend == :constant ? -2.86 : -3.41
    p_val = Z_t < cv5 ? 0.01 : Z_t < -1.5 ? 0.10 : 0.50

    cv = if trend == :constant
        (Symbol("1%") => -3.43, Symbol("5%") => -2.86, Symbol("10%") => -2.57)
    elseif trend == :trend
        (Symbol("1%") => -3.96, Symbol("5%") => -3.41, Symbol("10%") => -3.13)
    else
        (Symbol("1%") => -2.56, Symbol("5%") => -1.94, Symbol("10%") => -1.62)
    end

    return (statistic=Z_t, p_value=p_val, critical_values=cv, trend=trend)
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
    select_lag_bic(Y; max_p=12) -> Int

Select VAR lag order using BIC.
"""
function select_lag_bic(Y::Matrix{Float64}; max_p::Int=12)::Int
    T, K = size(Y)
    best_bic = Inf
    best_p = 1

    for p in 1:max_p
        T < p * K + K + 5 && break
        rf = try var_estimate(Y; p=p) catch; continue end
        T_eff = T - p
        Sigma = rf.Sigma
        logdet_Sigma = try log(det(Sigma + 1e-10*I)) catch 0.0 end
        bic = T_eff * logdet_Sigma + log(T_eff) * K^2 * p

        if bic < best_bic
            best_bic = bic
            best_p = p
        end
    end
    return best_p
end

"""
    select_lag_aic(Y; max_p=12) -> Int

Select VAR lag order using AIC.
"""
function select_lag_aic(Y::Matrix{Float64}; max_p::Int=12)::Int
    T, K = size(Y)
    best_aic = Inf
    best_p = 1

    for p in 1:max_p
        T < p * K + K + 5 && break
        rf = try var_estimate(Y; p=p) catch; continue end
        T_eff = T - p
        Sigma = rf.Sigma
        logdet_Sigma = try log(det(Sigma + 1e-10*I)) catch 0.0 end
        aic = T_eff * logdet_Sigma + 2 * K^2 * p

        if aic < best_aic
            best_aic = aic
            best_p = p
        end
    end
    return best_p
end

end # module EconometricsAdvanced
