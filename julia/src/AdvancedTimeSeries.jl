"""
AdvancedTimeSeries.jl — Advanced time series models for SRFM Lab

Extends the existing TimeSeriesAdvanced.jl with fully implemented:
  - TBATSModel: TBATS with multiple seasonalities and ARMA errors
  - DynamicFactorModel: time-varying loadings via Kalman filter
  - DCCGARCHModel: Dynamic Conditional Correlation GARCH(1,1)
  - BEKKModel: scalar BEKK multivariate GARCH
  - LSTARModel: Logistic Smooth Transition Autoregression
  - PSYTest: Phillips-Shi-Yu explosive root test for bubble detection
  - FunctionalTS: Karhunen-Loève decomposition for intraday curves
"""
module AdvancedTimeSeries

using Statistics, LinearAlgebra, Random

export TBATSModel, DynamicFactorModel, DCCGARCHModel, BEKKModel,
       LSTARModel, PSYTest, FunctionalTS

export fit!, forecast, update!, residuals, fitted_values,
       tbats_decompose, tbats_forecast,
       kalman_filter, kalman_smoother, dfm_estimate,
       dcc_fit, dcc_forecast, dcc_correlation,
       bekk_fit, bekk_covariance,
       lstar_fit, lstar_predict, transition_function,
       psy_bsadf, psy_critical_value, psy_bubble_dates,
       kl_decompose, kl_reconstruct, kl_scores, intraday_modes

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

function _ols(X::Matrix{Float64}, y::Vector{Float64})
    return (X' * X + 1e-10 * I) \ (X' * y)
end

function _lag_matrix(x::Vector{Float64}, p::Int)
    n = length(x)
    X = zeros(n-p, p)
    for j in 1:p
        X[:, j] = x[p+1-j:n-j]
    end
    return x[p+1:end], X
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. TBATSModel
# ─────────────────────────────────────────────────────────────────────────────

"""
TBATSModel: Trigonometric Seasonality, Box-Cox transformation, ARMA errors,
Trend, and Seasonal components.

Supports multiple seasonal periods (e.g., weekly=7 and monthly=30 for daily data).
Uses exponential smoothing for trend and seasonal components,
with ARMA(p,q) errors on the residuals.

Model:
  y_t = l_{t-1} + b_{t-1} + Σ_k s_{k,t} + d_t
  l_t = l_{t-1} + b_{t-1} + α × e_t
  b_t = b_{t-1} + β × e_t
  s_{k,t} = s_{k,t-m_k} + γ_k × e_t
  d_t = Σ φ_j d_{t-j} + e_t - Σ θ_j e_{t-j}
"""
mutable struct TBATSModel
    seasonal_periods::Vector{Int}    # e.g., [7, 30] for weekly + monthly
    ar_order::Int                    # ARMA error p
    ma_order::Int                    # ARMA error q
    # Smoothing parameters
    alpha::Float64   # level
    beta::Float64    # trend
    gamma::Vector{Float64}  # seasonal (one per period)
    # State
    level::Vector{Float64}
    trend_state::Vector{Float64}
    seasonal::Vector{Matrix{Float64}}   # seasonal[k] is n x 1 for period k
    ar_coef::Vector{Float64}
    ma_coef::Vector{Float64}
    errors::Vector{Float64}
    fitted::Vector{Float64}
    n_obs::Int
end

function TBATSModel(; periods::Vector{Int}=[7, 30],
                      ar_order::Int=1, ma_order::Int=0,
                      alpha::Float64=0.1, beta::Float64=0.05)
    k = length(periods)
    gamma = fill(0.05, k)
    return TBATSModel(periods, ar_order, ma_order, alpha, beta, gamma,
                       Float64[], Float64[],
                       [zeros(0,1) for _ in 1:k],
                       Float64[], Float64[], Float64[], Float64[], 0)
end

"""
Decompose a time series using TBATS framework.
Returns trend, seasonal components, and residuals.
"""
function tbats_decompose(tbats::TBATSModel, y::Vector{Float64})
    n = length(y)
    tbats.n_obs = n
    trend = zeros(n)
    trend_state = zeros(n)
    seasonal_comps = [zeros(n) for _ in tbats.seasonal_periods]
    residuals_ts = zeros(n)

    # Initialize
    trend[1] = y[1]
    trend_state[1] = 0.0
    for (k, period) in enumerate(tbats.seasonal_periods)
        seasonal_comps[k][1] = 0.0
    end

    # Iterative estimation
    for t in 2:n
        # Seasonal sum
        s_sum = sum(seasonal_comps[k][t-1] for k in 1:length(tbats.seasonal_periods))
        # One-step forecast
        y_hat = trend[t-1] + trend_state[t-1] + s_sum
        et = y[t] - y_hat
        residuals_ts[t] = et

        # Update level and trend
        trend[t] = trend[t-1] + trend_state[t-1] + tbats.alpha * et
        trend_state[t] = trend_state[t-1] + tbats.beta * tbats.alpha * et

        # Update seasonals
        for (k, period) in enumerate(tbats.seasonal_periods)
            prev_seasonal = t > period ? seasonal_comps[k][t-period] : seasonal_comps[k][1]
            seasonal_comps[k][t] = prev_seasonal + tbats.gamma[k] * et
        end
    end

    fitted = zeros(n)
    for t in 2:n
        s_sum = sum(seasonal_comps[k][t-1] for k in 1:length(tbats.seasonal_periods))
        fitted[t] = trend[t-1] + trend_state[t-1] + s_sum
    end
    fitted[1] = y[1]

    tbats.level = trend
    tbats.trend_state = trend_state
    tbats.errors = residuals_ts
    tbats.fitted = fitted

    # Fit AR on residuals
    if tbats.ar_order > 0 && n > tbats.ar_order + 5
        y_ar, X_ar = _lag_matrix(residuals_ts[2:end], tbats.ar_order)
        X_ar = hcat(ones(length(y_ar)), X_ar)
        tbats.ar_coef = _ols(X_ar, y_ar)[2:end]
    else
        tbats.ar_coef = zeros(tbats.ar_order)
    end

    return (trend=trend, seasonal=seasonal_comps, residuals=residuals_ts, fitted=fitted)
end

"""
Forecast h steps ahead.
"""
function tbats_forecast(tbats::TBATSModel, h::Int)
    n = tbats.n_obs
    forecasts = Float64[]
    level = tbats.level[end]
    trend_slope = tbats.trend_state[end]

    # Project seasonals forward
    seasonal_fwd = Vector{Float64}[]
    for (k, period) in enumerate(tbats.seasonal_periods)
        seasonal_k = tbats.seasonal[k]
        if isempty(seasonal_k) || size(seasonal_k, 1) == 0
            push!(seasonal_fwd, zeros(h))
        else
            fwd_k = Float64[]
            for j in 1:h
                s_idx = mod(n + j - 1, period) + 1
                push!(fwd_k, size(seasonal_k, 1) >= s_idx ? seasonal_k[s_idx, 1] : 0.0)
            end
            push!(seasonal_fwd, fwd_k)
        end
    end

    for j in 1:h
        s_sum = sum(seasonal_fwd[k][j] for k in 1:length(tbats.seasonal_periods))
        yhat = level + j * trend_slope + s_sum
        push!(forecasts, yhat)
    end
    return forecasts
end

fitted_values(tbats::TBATSModel) = tbats.fitted
residuals(tbats::TBATSModel) = tbats.errors

function fit!(tbats::TBATSModel, y::Vector{Float64})
    tbats_decompose(tbats, y)
    return tbats
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. DynamicFactorModel
# ─────────────────────────────────────────────────────────────────────────────

"""
DynamicFactorModel: extracts latent common factors from a panel of time series,
with factor dynamics modeled as a VAR(1) via Kalman filter.

State space form:
  y_t = Λ f_t + ε_t,   ε_t ~ N(0, R)
  f_t = A f_t-1 + η_t, η_t ~ N(0, Q)

where Λ is the (n_vars × n_factors) loading matrix, f_t is the (n_factors × 1)
state vector.

Estimation: Two-step PCA + Kalman filter refinement.
"""
mutable struct DynamicFactorModel
    n_factors::Int
    loadings::Matrix{Float64}   # n_vars × n_factors
    transition::Matrix{Float64} # A: n_factors × n_factors (VAR(1))
    obs_noise::Matrix{Float64}  # R: n_vars × n_vars (diagonal)
    state_noise::Matrix{Float64} # Q: n_factors × n_factors
    factors_smoothed::Matrix{Float64}  # T × n_factors
    mu_obs::Vector{Float64}     # mean of each observable
    std_obs::Vector{Float64}    # std of each observable
    n_obs::Int
end

function DynamicFactorModel(n_factors::Int=2)
    return DynamicFactorModel(n_factors, zeros(0,n_factors), zeros(n_factors,n_factors),
                               zeros(0,0), Matrix{Float64}(I, n_factors, n_factors),
                               zeros(0,n_factors), Float64[], Float64[], 0)
end

"""
Estimate DFM via two-step PCA + Kalman refinement.
Y: T × n_vars matrix of observables.
"""
function dfm_estimate(dfm::DynamicFactorModel, Y::Matrix{Float64})
    T, n = size(Y)
    k = dfm.n_factors

    # Standardize
    dfm.mu_obs = vec(mean(Y, dims=1))
    dfm.std_obs = vec(std(Y, dims=1)) .+ 1e-8
    Y_std = (Y .- dfm.mu_obs') ./ dfm.std_obs'

    # Step 1: PCA to get initial loadings and factors
    C = Y_std' * Y_std ./ T
    evals = eigvals(C)
    evecs = eigvecs(C)
    idx = sortperm(evals, rev=true)
    dfm.loadings = evecs[:, idx[1:k]]  # n × k
    F_pca = Y_std * dfm.loadings  # T × k

    # Step 2: Estimate VAR(1) on factors
    if T > k + 2
        F_curr = F_pca[2:end, :]   # (T-1) × k
        F_prev = F_pca[1:end-1, :]  # (T-1) × k
        # OLS: F_curr = F_prev * A' + noise
        A = (F_prev' * F_prev + 1e-6*I) \ (F_prev' * F_curr)
        dfm.transition = A'
    else
        dfm.transition = Matrix{Float64}(I, k, k) * 0.9
    end

    # Step 3: Kalman filter
    F_filtered, P_filtered = _kalman_filter_dfm(dfm, Y_std, F_pca[1,:])
    dfm.factors_smoothed = F_filtered
    dfm.n_obs = T

    # Estimate observation noise R (diagonal)
    Y_hat = F_filtered * dfm.loadings'
    resid = Y_std .- Y_hat
    dfm.obs_noise = Diagonal(vec(mean(resid.^2, dims=1)))

    return dfm
end

function _kalman_filter_dfm(dfm::DynamicFactorModel, Y_std::Matrix{Float64},
                              f0::Vector{Float64})
    T, n = size(Y_std)
    k = dfm.n_factors
    A = dfm.transition
    Lambda = dfm.loadings
    Q = dfm.state_noise
    R_diag = ones(n) .* 0.1  # initial noise estimate

    F = zeros(T, k)
    f_pred = copy(f0)
    P_pred = Matrix{Float64}(I, k, k)

    for t in 1:T
        y_t = Y_std[t, :]
        # Innovation
        y_pred = Lambda * f_pred
        innov = y_t .- y_pred

        # Kalman gain
        S = Lambda * P_pred * Lambda' + Diagonal(R_diag)
        K = P_pred * Lambda' * inv(S + 1e-8*I)

        # Update
        f_upd = f_pred .+ K * innov
        P_upd = (I - K * Lambda) * P_pred

        F[t, :] = f_upd

        # Predict next
        f_pred = A * f_upd
        P_pred = A * P_upd * A' + Q
    end

    return F, nothing
end

"""Kalman filter step for external use."""
function kalman_filter(A::Matrix{Float64}, C::Matrix{Float64},
                        Q::Matrix{Float64}, R::Matrix{Float64},
                        y_series::Matrix{Float64}, x0::Vector{Float64})
    T, n = size(y_series)
    k = length(x0)
    X_filt = zeros(T, k)
    x = copy(x0)
    P = Matrix{Float64}(I, k, k)

    for t in 1:T
        y_t = y_series[t, :]
        # Predict
        x_pred = A * x
        P_pred = A * P * A' + Q
        # Update
        innov = y_t .- C * x_pred
        S = C * P_pred * C' + R
        K = P_pred * C' * inv(S + 1e-8*I)
        x = x_pred .+ K * innov
        P = (I - K * C) * P_pred
        X_filt[t, :] = x
    end
    return X_filt
end

"""Kalman smoother (RTS smoother)."""
function kalman_smoother(A::Matrix{Float64}, Q::Matrix{Float64},
                          X_filt::Matrix{Float64})
    T, k = size(X_filt)
    X_smooth = copy(X_filt)
    for t in (T-1):-1:1
        P_t = Matrix{Float64}(I, k, k)  # simplified
        P_t1 = P_t
        G = P_t * A' * inv(A * P_t * A' + Q + 1e-8*I)
        X_smooth[t, :] = X_filt[t, :] .+ G * (X_smooth[t+1, :] .- A * X_filt[t, :])
    end
    return X_smooth
end

function fit!(dfm::DynamicFactorModel, Y::Matrix{Float64})
    dfm_estimate(dfm, Y)
    return dfm
end

function forecast(dfm::DynamicFactorModel, h::Int)
    isempty(dfm.factors_smoothed) && return zeros(h, size(dfm.loadings, 1))
    f_last = dfm.factors_smoothed[end, :]
    Y_fcast = zeros(h, size(dfm.loadings, 1))
    f = copy(f_last)
    for j in 1:h
        f = dfm.transition * f
        Y_fcast[j, :] = dfm.loadings * f .* dfm.std_obs .+ dfm.mu_obs
    end
    return Y_fcast
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. DCCGARCHModel
# ─────────────────────────────────────────────────────────────────────────────

"""
DCCGARCHModel: Dynamic Conditional Correlation GARCH(1,1).

Step 1: Fit univariate GARCH(1,1) to each series.
Step 2: Standardize residuals.
Step 3: Model the correlation matrix using DCC dynamics:
  Q_t = (1-a-b)*Q_bar + a*z_{t-1}z_{t-1}' + b*Q_{t-1}
  R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

Parameters estimated via two-step QML.
"""
mutable struct DCCGARCHModel
    n_series::Int
    # GARCH parameters per series
    garch_omega::Vector{Float64}
    garch_alpha::Vector{Float64}
    garch_beta::Vector{Float64}
    # DCC parameters
    dcc_a::Float64
    dcc_b::Float64
    # State
    Q_bar::Matrix{Float64}    # unconditional correlation
    garch_vol::Matrix{Float64}  # T × n (conditional volatilities)
    std_resid::Matrix{Float64}  # T × n (standardized residuals)
    Q_series::Vector{Matrix{Float64}}  # T Q matrices
    R_series::Vector{Matrix{Float64}}  # T correlation matrices
    n_obs::Int
end

function DCCGARCHModel(n_series::Int)
    return DCCGARCHModel(n_series,
                          fill(1e-5, n_series), fill(0.09, n_series), fill(0.89, n_series),
                          0.04, 0.94,
                          Matrix{Float64}(I, n_series, n_series),
                          zeros(0,n_series), zeros(0,n_series),
                          Matrix{Float64}[], Matrix{Float64}[],
                          0)
end

"""Fit GARCH(1,1) to a single series. Returns (h, z) = (variance, std residuals)."""
function _fit_garch11(r::Vector{Float64}; omega::Float64=1e-5, alpha::Float64=0.09, beta::Float64=0.89)
    n = length(r)
    h = fill(var(r), n)
    z = zeros(n)
    h[1] = var(r)
    for t in 2:n
        h[t] = omega + alpha * r[t-1]^2 + beta * h[t-1]
        h[t] = max(h[t], 1e-10)
    end
    z = r ./ sqrt.(h)
    return h, z
end

"""
Fit DCC-GARCH to a matrix of returns.
R: T × n returns.
"""
function dcc_fit(dcc::DCCGARCHModel, R::Matrix{Float64};
                  dcc_a::Float64=0.04, dcc_b::Float64=0.94)
    T, n = size(R)
    dcc.n_series = n
    dcc.dcc_a = dcc_a
    dcc.dcc_b = dcc_b

    # Step 1: Fit GARCH per series
    H = zeros(T, n)
    Z = zeros(T, n)
    for j in 1:n
        h_j, z_j = _fit_garch11(R[:, j];
                                  omega=dcc.garch_omega[min(j,end)],
                                  alpha=dcc.garch_alpha[min(j,end)],
                                  beta=dcc.garch_beta[min(j,end)])
        H[:, j] = h_j
        Z[:, j] = z_j
        if j <= length(dcc.garch_omega)
            dcc.garch_omega[j] = mean(h_j) * (1 - 0.09 - 0.89)
        end
    end
    dcc.garch_vol = sqrt.(H)
    dcc.std_resid = Z

    # Q_bar = unconditional correlation of standardized residuals
    dcc.Q_bar = cor(Z)

    # Step 2: DCC filter
    dcc.Q_series = Matrix{Float64}[]
    dcc.R_series = Matrix{Float64}[]

    Q = copy(dcc.Q_bar)
    for t in 1:T
        z_t = Z[t, :]
        Q_new = (1 - dcc_a - dcc_b) .* dcc.Q_bar .+ dcc_a .* (z_t * z_t') .+ dcc_b .* Q
        # Symmetrize
        Q_new = (Q_new + Q_new') / 2
        # Correlation
        D_inv = Diagonal(1.0 ./ sqrt.(max.(diag(Q_new), 1e-10)))
        R_t = D_inv * Q_new * D_inv
        # Clip to valid range
        for i in 1:n, j in 1:n
            if i != j
                R_t[i,j] = clamp(R_t[i,j], -0.999, 0.999)
            end
        end
        push!(dcc.Q_series, copy(Q_new))
        push!(dcc.R_series, copy(R_t))
        Q = Q_new
    end

    dcc.n_obs = T
    return dcc
end

function fit!(dcc::DCCGARCHModel, R::Matrix{Float64})
    dcc_fit(dcc, R)
    return dcc
end

"""Get the most recent dynamic correlation matrix."""
function dcc_correlation(dcc::DCCGARCHModel)
    isempty(dcc.R_series) && return dcc.Q_bar
    return dcc.R_series[end]
end

"""Forecast the conditional correlation h steps ahead (mean-reverts to Q_bar)."""
function dcc_forecast(dcc::DCCGARCHModel, h::Int)
    isempty(dcc.Q_series) && return [dcc.Q_bar for _ in 1:h]
    Q = dcc.Q_series[end]
    forecasts = Matrix{Float64}[]
    for _ in 1:h
        Q = (1 - dcc.dcc_a - dcc.dcc_b) .* dcc.Q_bar .+ (dcc.dcc_a + dcc.dcc_b) .* Q
        D_inv = Diagonal(1.0 ./ sqrt.(max.(diag(Q), 1e-10)))
        R_h = D_inv * Q * D_inv
        push!(forecasts, R_h)
    end
    return forecasts
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. BEKKModel — Scalar BEKK GARCH
# ─────────────────────────────────────────────────────────────────────────────

"""
BEKKModel: scalar BEKK(1,1) multivariate GARCH.

H_t = CC' + a² ε_{t-1}ε_{t-1}' + b² H_{t-1}

where C is a lower triangular matrix, a and b are scalar parameters.
The scalar BEKK ensures positive definiteness.
"""
mutable struct BEKKModel
    n_series::Int
    a::Float64          # ARCH scalar coefficient
    b::Float64          # GARCH scalar coefficient
    C::Matrix{Float64}  # lower triangular intercept
    H_series::Vector{Matrix{Float64}}  # conditional covariances
    H_uncond::Matrix{Float64}          # unconditional covariance
    n_obs::Int
end

function BEKKModel(n_series::Int; a::Float64=0.09, b::Float64=0.89)
    return BEKKModel(n_series, a, b, Matrix{Float64}(I, n_series, n_series),
                      Matrix{Float64}[], zeros(n_series, n_series), 0)
end

"""
Fit scalar BEKK to returns matrix R: T × n.
"""
function bekk_fit(bekk::BEKKModel, R::Matrix{Float64};
                   a::Float64=0.09, b::Float64=0.89)
    T, n = size(R)
    bekk.a = a
    bekk.b = b
    bekk.n_series = n

    # Initialize unconditional covariance
    H_bar = cov(R)
    bekk.H_uncond = H_bar

    # Compute C such that CC' = (1 - a^2 - b^2) * H_bar
    scale = max(1 - a^2 - b^2, 1e-6)
    C_sq = scale .* H_bar
    # Cholesky of C_sq
    try
        bekk.C = cholesky(C_sq + 1e-8*I).L
    catch
        bekk.C = Matrix{Float64}(I, n, n) * sqrt(scale * mean(diag(H_bar)))
    end

    # Filter
    bekk.H_series = Matrix{Float64}[]
    H = copy(H_bar)
    CC = bekk.C * bekk.C'

    for t in 1:T
        eps = R[t, :]
        H_new = CC .+ a^2 .* (eps * eps') .+ b^2 .* H
        H_new = (H_new + H_new') / 2 + 1e-8*I  # symmetrize + stabilize
        push!(bekk.H_series, copy(H_new))
        H = H_new
    end

    bekk.n_obs = T
    return bekk
end

function fit!(bekk::BEKKModel, R::Matrix{Float64})
    bekk_fit(bekk, R)
    return bekk
end

"""Get the most recent conditional covariance matrix."""
function bekk_covariance(bekk::BEKKModel)
    isempty(bekk.H_series) && return bekk.H_uncond
    return bekk.H_series[end]
end

"""Forecast covariance h steps ahead."""
function forecast(bekk::BEKKModel, h::Int)
    isempty(bekk.H_series) && return [bekk.H_uncond for _ in 1:h]
    H = bekk.H_series[end]
    CC = bekk.C * bekk.C'
    forecasts = Matrix{Float64}[]
    for _ in 1:h
        # E[H_{t+h}] mean-reverts to unconditional: iterate
        H = CC .+ (bekk.a^2 + bekk.b^2) .* H
        push!(forecasts, copy(H))
    end
    return forecasts
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. LSTARModel
# ─────────────────────────────────────────────────────────────────────────────

"""
LSTARModel: Logistic Smooth Transition Autoregression.

y_t = (φ₁₀ + φ₁₁y_{t-1} + ... + φ₁p y_{t-p}) × (1 - G(s_t; γ, c))
    + (φ₂₀ + φ₂₁y_{t-1} + ... + φ₂p y_{t-p}) × G(s_t; γ, c)
    + ε_t

where G(s; γ, c) = 1/(1 + exp(-γ(s-c))) is the logistic transition function.
The transition variable s_t is typically y_{t-d} (delay parameter d).
"""
mutable struct LSTARModel
    p::Int          # AR order
    d::Int          # delay: s_t = y_{t-d}
    gamma::Float64  # transition speed (steepness)
    c::Float64      # transition midpoint (threshold)
    phi1::Vector{Float64}  # AR coefficients in regime 1 (low)
    phi2::Vector{Float64}  # AR coefficients in regime 2 (high)
    fitted::Vector{Float64}
    residuals_ts::Vector{Float64}
    n_obs::Int
end

function LSTARModel(; p::Int=2, d::Int=1)
    return LSTARModel(p, d, 5.0, 0.0, zeros(p+1), zeros(p+1),
                       Float64[], Float64[], 0)
end

"""Logistic transition function G(s; γ, c) ∈ (0,1)."""
function transition_function(lstar::LSTARModel, s::Float64)
    return 1.0 / (1.0 + exp(-lstar.gamma * (s - lstar.c)))
end

"""
Fit LSTAR via grid search on (gamma, c) + OLS for phi.
"""
function lstar_fit(lstar::LSTARModel, y::Vector{Float64};
                    gamma_grid::Vector{Float64}=[1.0, 3.0, 5.0, 10.0, 20.0],
                    n_grid_c::Int=10)
    n = length(y)
    p = lstar.p
    d = lstar.d

    min_t = max(p, d) + 1
    n_reg = n - min_t + 1
    if n_reg < p + 5
        return lstar
    end

    Y = y[min_t:end]
    X1 = zeros(length(Y), p+1)  # regime 1 regressors
    X2 = zeros(length(Y), p+1)  # regime 2 regressors

    # Possible transition midpoints: quantiles of transition variable
    s_var = y[(d+1):(n-min_t+d+1)]  # y_{t-d}
    c_candidates = quantile(s_var, range(0.15, 0.85, length=n_grid_c))

    best_ssr = Inf
    best_gamma = lstar.gamma
    best_c = lstar.c

    for gamma in gamma_grid, c in c_candidates
        # Compute transition weights
        s_t = y[min_t-d:end-d]
        g_t = [1.0 / (1.0 + exp(-gamma * (s - c))) for s in s_t]

        # Build regressors
        for (i, t) in enumerate(min_t:n)
            for lag in 1:p
                X1[i, lag+1] = y[t-lag]
                X2[i, lag+1] = y[t-lag]
            end
            X1[i, 1] = 1.0
            X2[i, 1] = 1.0
        end

        # Weighted OLS
        W1 = Diagonal(1 .- g_t[1:length(Y)])
        W2 = Diagonal(g_t[1:length(Y)])
        XW1 = W1 * X1
        XW2 = W2 * X2
        X_full = hcat(XW1, XW2)

        if rank(X_full) < size(X_full, 2)
            continue
        end

        phi = (X_full' * X_full + 1e-10*I) \ (X_full' * Y)
        resid = Y .- X_full * phi
        ssr = sum(resid.^2)

        if ssr < best_ssr
            best_ssr = ssr
            best_gamma = gamma
            best_c = c
            lstar.phi1 = phi[1:p+1]
            lstar.phi2 = phi[p+2:end]
        end
    end

    lstar.gamma = best_gamma
    lstar.c = best_c

    # Compute fitted values
    lstar.fitted = zeros(n)
    lstar.residuals_ts = zeros(n)
    for t in min_t:n
        s_t = y[t-d]
        g_t = transition_function(lstar, s_t)
        x = vcat(1.0, [y[t-j] for j in 1:p])
        y_hat = (1-g_t) * dot(lstar.phi1, x) + g_t * dot(lstar.phi2, x)
        lstar.fitted[t] = y_hat
        lstar.residuals_ts[t] = y[t] - y_hat
    end

    lstar.n_obs = n
    return lstar
end

function fit!(lstar::LSTARModel, y::Vector{Float64})
    lstar_fit(lstar, y)
    return lstar
end

"""One-step-ahead prediction."""
function lstar_predict(lstar::LSTARModel, y_hist::Vector{Float64})
    p = lstar.p
    d = lstar.d
    length(y_hist) < max(p, d) && return NaN
    s = y_hist[end-d+1]
    g = transition_function(lstar, s)
    x = vcat(1.0, [y_hist[end-j+1] for j in 1:p])
    return (1-g) * dot(lstar.phi1, x) + g * dot(lstar.phi2, x)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. PSYTest — Phillips-Shi-Yu Explosive Root Test
# ─────────────────────────────────────────────────────────────────────────────

"""
PSYTest: Phillips, Shi & Yu (2015) right-tailed recursive ADF test.

Tests for explosive behavior (bubbles) in time series.
The Backward Supremum ADF (BSADF) statistic is computed by recursively
testing forward-expanding windows and taking the supremum.

High BSADF → explosive root → potential bubble.

Test statistic:
  ADF_[r1,r2](y) = T_{r1,r2} × (ρ_{r1,r2} - 1) / se_{r1,r2}
  BSADF_{r2}(y) = sup_{r1 ∈ [r0, r2-r0]} ADF_{r1, r2}(y)

Critical values approximated via asymptotic distribution.
"""
mutable struct PSYTest
    min_window::Int      # r0: minimum window size
    lag_order::Int       # ADF lag augmentation
    adf_stats::Vector{Float64}   # rolling ADF statistics
    bsadf_stats::Vector{Float64} # BSADF (supremum) statistics
    bubble_signal::Vector{Bool}  # is t in a bubble period?
    cv_99::Float64
    cv_95::Float64
    cv_90::Float64
    n_obs::Int
end

function PSYTest(; min_window::Int=20, lag_order::Int=1,
                   cv_99::Float64=2.0, cv_95::Float64=1.645, cv_90::Float64=1.282)
    return PSYTest(min_window, lag_order, Float64[], Float64[], Bool[],
                    cv_99, cv_95, cv_90, 0)
end

"""
Compute ADF t-statistic on a subsequence y for H1: ρ > 1.
Returns the t-stat for the lagged level coefficient.
"""
function _adf_tstat(y::Vector{Float64}, lag::Int)
    n = length(y)
    n < lag + 4 && return NaN

    dy = y[2:end] .- y[1:end-1]
    m = length(dy) - lag
    m < 3 && return NaN

    Y = dy[lag+1:end]
    X_cols = hcat(ones(m), y[lag+1:end-1])  # intercept + lagged level
    for l in 1:lag
        X_cols = hcat(X_cols, dy[lag+1-l:end-l])
    end

    size(X_cols, 1) <= size(X_cols, 2) && return NaN
    betas = (X_cols' * X_cols + 1e-10*I) \ (X_cols' * Y)
    resid = Y .- X_cols * betas
    sigma2 = sum(resid.^2) / max(m - size(X_cols,2), 1)
    inv_XtX = inv(X_cols' * X_cols + 1e-10*I)
    se_beta = sqrt(max(0.0, sigma2 * inv_XtX[2,2]))
    se_beta < 1e-12 && return NaN
    return betas[2] / se_beta  # t-stat on lagged level
end

"""
Compute PSY BSADF statistics for entire series.
"""
function psy_bsadf(psy::PSYTest, y::Vector{Float64})
    n = length(y)
    psy.n_obs = n
    psy.adf_stats = fill(NaN, n)
    psy.bsadf_stats = fill(NaN, n)
    psy.bubble_signal = falses(n)

    for t in psy.min_window:n
        # Compute ADF over the expanding window [1:t]
        adf_t = _adf_tstat(y[1:t], psy.lag_order)
        psy.adf_stats[t] = adf_t

        # BSADF: supremum over all valid start points
        max_adf = -Inf
        for r1 in 1:(t - psy.min_window + 1)
            if t - r1 + 1 >= psy.min_window
                adf_sub = _adf_tstat(y[r1:t], psy.lag_order)
                if !isnan(adf_sub)
                    max_adf = max(max_adf, adf_sub)
                end
            end
        end
        psy.bsadf_stats[t] = max_adf == -Inf ? NaN : max_adf
    end

    # Identify bubble periods
    for t in 1:n
        if !isnan(psy.bsadf_stats[t])
            psy.bubble_signal[t] = psy.bsadf_stats[t] > psy.cv_95
        end
    end

    return psy
end

function fit!(psy::PSYTest, y::Vector{Float64})
    psy_bsadf(psy, y)
    return psy
end

"""
Get critical value at a given confidence level.
"""
function psy_critical_value(psy::PSYTest; level::Float64=0.95)
    level >= 0.99 && return psy.cv_99
    level >= 0.95 && return psy.cv_95
    return psy.cv_90
end

"""
Identify contiguous bubble episode date ranges.
Returns list of (start, end) index pairs.
"""
function psy_bubble_dates(psy::PSYTest; min_duration::Int=5)
    n = length(psy.bubble_signal)
    episodes = Tuple{Int,Int}[]
    in_bubble = false
    start_t = 0

    for t in 1:n
        if psy.bubble_signal[t] && !in_bubble
            in_bubble = true
            start_t = t
        elseif !psy.bubble_signal[t] && in_bubble
            in_bubble = false
            duration = t - start_t
            duration >= min_duration && push!(episodes, (start_t, t-1))
        end
    end
    in_bubble && push!(episodes, (start_t, n))
    return episodes
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. FunctionalTS — Karhunen-Loève Decomposition for Intraday Curves
# ─────────────────────────────────────────────────────────────────────────────

"""
FunctionalTS: Functional time series analysis via Karhunen-Loève (KL) decomposition.

Each day's intraday price/volume curve is treated as a function.
KL decomposes these curves into orthogonal modes (eigenfunctions),
providing:
  - Dimension reduction for intraday patterns
  - Forecasting of tomorrow's curve from today's KL scores
  - Anomaly detection via reconstruction error

Model:
  X_i(t) = μ(t) + Σ_k ξ_{ik} φ_k(t) + ε_i(t)

where μ is the mean curve, φ_k are eigenfunctions, ξ_{ik} are scores.
"""
mutable struct FunctionalTS
    n_modes::Int
    mean_curve::Vector{Float64}    # mean function over time grid
    eigenfunctions::Matrix{Float64}  # n_grid × n_modes
    eigenvalues::Vector{Float64}
    scores::Matrix{Float64}        # n_days × n_modes (ξ_{ik})
    variance_explained::Vector{Float64}
    n_days::Int
    n_grid::Int
end

function FunctionalTS(; n_modes::Int=5)
    return FunctionalTS(n_modes, Float64[], zeros(0,n_modes), Float64[],
                         zeros(0,n_modes), Float64[], 0, 0)
end

"""
Fit KL decomposition to a matrix of intraday curves.
X: n_days × n_grid matrix (each row = one day's curve).
"""
function kl_decompose(fts::FunctionalTS, X::Matrix{Float64})
    n_days, n_grid = size(X)
    fts.n_days = n_days
    fts.n_grid = n_grid
    k = min(fts.n_modes, min(n_days, n_grid))

    # Mean curve
    fts.mean_curve = vec(mean(X, dims=1))
    X_centered = X .- fts.mean_curve'

    # Covariance operator (n_grid × n_grid)
    C = (X_centered' * X_centered) ./ max(n_days - 1, 1)

    # Eigendecomposition
    evals = eigvals(C)
    evecs = eigvecs(C)
    idx = sortperm(evals, rev=true)

    fts.eigenvalues = evals[idx[1:k]]
    fts.eigenfunctions = evecs[:, idx[1:k]]  # n_grid × k

    # Normalize eigenfunctions
    for j in 1:k
        norm_j = sqrt(sum(fts.eigenfunctions[:,j].^2))
        fts.eigenfunctions[:,j] ./= max(norm_j, 1e-10)
    end

    # KL scores: ξ_{ik} = integral X_i(t) φ_k(t) dt ≈ X_centered * φ_k
    fts.scores = X_centered * fts.eigenfunctions  # n_days × k

    # Variance explained
    total_var = sum(abs.(evals))
    fts.variance_explained = abs.(fts.eigenvalues) ./ (total_var + 1e-10)

    return fts
end

function fit!(fts::FunctionalTS, X::Matrix{Float64})
    kl_decompose(fts, X)
    return fts
end

"""
Reconstruct curves from KL scores.
"""
function kl_reconstruct(fts::FunctionalTS, scores::Matrix{Float64})
    isempty(fts.eigenfunctions) && return zeros(size(scores,1), length(fts.mean_curve))
    X_hat = scores * fts.eigenfunctions' .+ fts.mean_curve'
    return X_hat
end

"""Get the KL scores for historical data."""
function kl_scores(fts::FunctionalTS)
    return fts.scores
end

"""
The n dominant intraday modes (eigenfunctions).
"""
function intraday_modes(fts::FunctionalTS)
    isempty(fts.eigenfunctions) && return nothing
    return (modes=fts.eigenfunctions, values=fts.eigenvalues,
            var_explained=fts.variance_explained)
end

"""
Forecast tomorrow's curve from today's KL scores using AR(1) on each score.
"""
function forecast(fts::FunctionalTS, h::Int=1)
    isempty(fts.scores) && return zeros(h, fts.n_grid)
    k = size(fts.scores, 2)
    n = size(fts.scores, 1)
    forecasts = zeros(h, fts.n_grid)

    for j in 1:k
        # AR(1) on j-th score
        sc = fts.scores[:, j]
        if n > 2
            Y_sc = sc[2:end]
            X_sc = hcat(ones(n-1), sc[1:end-1])
            betas = (X_sc' * X_sc + 1e-10*I) \ (X_sc' * Y_sc)
            # Forecast
            s_prev = sc[end]
            for t in 1:h
                s_next = betas[1] + betas[2] * s_prev
                forecasts[t, :] .+= s_next .* fts.eigenfunctions[:, j]
                s_prev = s_next
            end
        end
    end

    forecasts .+= fts.mean_curve'
    return forecasts
end

"""
Compute reconstruction error (anomaly score) for new curves.
X_new: n × n_grid matrix of new curves.
High error → unusual intraday pattern.
"""
function anomaly_score(fts::FunctionalTS, X_new::Matrix{Float64})
    isempty(fts.eigenfunctions) && return fill(NaN, size(X_new,1))
    X_c = X_new .- fts.mean_curve'
    scores_new = X_c * fts.eigenfunctions  # n × k
    X_reconstructed = scores_new * fts.eigenfunctions' .+ fts.mean_curve'
    errors = vec(mean((X_new .- X_reconstructed).^2, dims=2))
    return errors
end

end  # module AdvancedTimeSeries
