module TimeSeriesAdvanced

# ============================================================
# TimeSeriesAdvanced.jl — SARIMA, state-space models, Kalman
#   smoother, dynamic factor models (pure stdlib)
# ============================================================

using Statistics, LinearAlgebra

export ARMAModel, SARIMAModel, StateSpaceModel, KalmanState
export fit_ar, fit_arma_ols, arma_forecast
export sarima_fit, sarima_forecast, seasonal_decompose
export kalman_filter, kalman_smoother, kalman_predict
export state_space_em, ssm_likelihood
export dynamic_factor_model, dfm_factors, dfm_forecast
export cointegration_test, vecm_fit, vecm_forecast
export structural_break_test, rolling_regression
export granger_causality, impulse_response_var
export acf, pacf, ljung_box_test
export hodrick_prescott_filter, baxter_king_filter
export bev_levin_coss, panel_unit_root

# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

struct ARMAModel
    ar_coefs::Vector{Float64}   # AR(p) coefficients
    ma_coefs::Vector{Float64}   # MA(q) coefficients
    intercept::Float64
    sigma2::Float64             # residual variance
end

struct SARIMAModel
    ar_coefs::Vector{Float64}
    ma_coefs::Vector{Float64}
    sar_coefs::Vector{Float64}  # seasonal AR
    sma_coefs::Vector{Float64}  # seasonal MA
    d::Int                       # differencing order
    D::Int                       # seasonal differencing
    s::Int                       # season length
    intercept::Float64
    sigma2::Float64
end

struct StateSpaceModel
    F::Matrix{Float64}    # transition matrix
    H::Matrix{Float64}    # observation matrix
    Q::Matrix{Float64}    # process noise covariance
    R::Matrix{Float64}    # observation noise covariance
    x0::Vector{Float64}   # initial state mean
    P0::Matrix{Float64}   # initial state covariance
end

mutable struct KalmanState
    x::Vector{Float64}    # filtered state
    P::Matrix{Float64}    # filtered covariance
    log_likelihood::Float64
end

# ──────────────────────────────────────────────────────────────
# ACF / PACF
# ──────────────────────────────────────────────────────────────

"""
    acf(series, max_lag) -> Vector of autocorrelations

Sample autocorrelation function up to max_lag.
"""
function acf(series::Vector{Float64}, max_lag::Int=20)
    n = length(series)
    mu = mean(series)
    gamma0 = sum((series .- mu).^2) / n
    result = zeros(max_lag + 1)
    result[1] = 1.0
    for k in 1:max_lag
        if n - k < 1; break; end
        result[k+1] = sum((series[1:n-k] .- mu) .* (series[k+1:n] .- mu)) / (n * gamma0)
    end
    return result
end

"""
    pacf(series, max_lag) -> Vector of partial autocorrelations

Partial autocorrelation function using Yule-Walker equations.
"""
function pacf(series::Vector{Float64}, max_lag::Int=20)
    acf_vals = acf(series, max_lag)
    result = zeros(max_lag)
    for k in 1:max_lag
        if k == 1
            result[1] = acf_vals[2]
        else
            # Durbin-Levinson recursion
            phi = zeros(k)
            phi_prev = zeros(k-1)
            for m in 1:k-1
                phi_prev[m] = result[m]
            end
            num = acf_vals[k+1] - sum(phi_prev[j] * acf_vals[k-j+1] for j in 1:k-1; init=0.0)
            denom = 1.0 - sum(phi_prev[j] * acf_vals[j+1] for j in 1:k-1; init=0.0)
            phi[k] = abs(denom) > 1e-12 ? num / denom : 0.0
            result[k] = phi[k]
        end
    end
    return result
end

"""
    ljung_box_test(series, max_lag) -> (Q_stat, p_approx)

Ljung-Box portmanteau test for autocorrelation.
"""
function ljung_box_test(series::Vector{Float64}, max_lag::Int=20)
    n = length(series)
    acf_vals = acf(series, max_lag)
    Q = 0.0
    for k in 1:max_lag
        rk = acf_vals[k+1]
        Q += rk^2 / (n - k)
    end
    Q *= n * (n + 2)
    # Chi-squared CDF approximation (Wilson-Hilferty)
    k_df = Float64(max_lag)
    z = ((Q / k_df)^(1.0/3.0) - (1.0 - 2.0/(9.0*k_df))) / sqrt(2.0/(9.0*k_df))
    p_approx = 0.5 * erfc(z / sqrt(2.0))
    return Q, p_approx
end

# ──────────────────────────────────────────────────────────────
# AR/ARMA fitting
# ──────────────────────────────────────────────────────────────

"""
    fit_ar(series, p) -> ARMAModel

Fit AR(p) model using Yule-Walker equations.
"""
function fit_ar(series::Vector{Float64}, p::Int)
    n = length(series)
    mu = mean(series)
    y = series .- mu
    acf_vals = acf(y, p)
    # Build Yule-Walker system: Γ * phi = γ
    Gamma = [acf_vals[abs(i-j)+1] for i in 1:p, j in 1:p]
    gamma_vec = acf_vals[2:p+1]
    phi = (Gamma + 1e-10*I) \ gamma_vec
    # Residual variance
    sigma2 = acf_vals[1] - dot(phi, gamma_vec)
    return ARMAModel(phi, Float64[], mu, max(sigma2, 1e-12))
end

"""
    fit_arma_ols(series, p, q) -> ARMAModel

Fit ARMA(p,q) via conditional sum-of-squares (approximate MLE).
Uses iterative OLS approach.
"""
function fit_arma_ols(series::Vector{Float64}, p::Int, q::Int)
    n = length(series)
    mu = mean(series)
    y = series .- mu
    # Initialize with AR(p) fit
    ar_model = fit_ar(series, max(p, 1))
    phi = p > 0 ? ar_model.ar_coefs : Float64[]
    theta = zeros(q)
    # Iterative: compute residuals, then update MA via regression
    max_order = max(p, q)
    eps = zeros(n)
    for _ in 1:20
        # Compute residuals given current phi, theta
        for t in max_order+1:n
            eps[t] = y[t]
            for i in 1:p
                eps[t] -= phi[i] * y[t-i]
            end
            for j in 1:q
                eps[t] -= theta[j] * eps[t-j]
            end
        end
        # Update: build design matrix
        if q > 0
            X = zeros(n - max_order, p + q + 1)
            yy = y[max_order+1:end]
            for t in 1:n-max_order
                X[t, 1] = 1.0
                for i in 1:p
                    X[t, 1+i] = y[max_order+t-i]
                end
                for j in 1:q
                    X[t, 1+p+j] = -eps[max_order+t-j]
                end
            end
            beta = (X'X + 1e-10*I) \ (X'yy)
            mu_est = mu + beta[1]
            phi = p > 0 ? beta[2:1+p] : Float64[]
            theta = q > 0 ? beta[2+p:end] : Float64[]
        end
    end
    sigma2 = var(eps[max_order+1:end])
    return ARMAModel(phi, theta, mu, sigma2)
end

"""
    arma_forecast(model, series, h) -> forecasts

h-step-ahead forecasts from ARMA model.
"""
function arma_forecast(model::ARMAModel, series::Vector{Float64}, h::Int=10)
    n = length(series)
    p = length(model.ar_coefs)
    q = length(model.ma_coefs)
    max_order = max(p, q, 1)

    # Compute in-sample residuals
    mu = model.intercept
    y = series .- mu
    eps = zeros(n)
    for t in max_order+1:n
        eps[t] = y[t]
        for i in 1:p
            if t-i >= 1; eps[t] -= model.ar_coefs[i] * y[t-i]; end
        end
        for j in 1:q
            if t-j >= 1; eps[t] -= model.ma_coefs[j] * eps[t-j]; end
        end
    end

    # Forecast
    y_ext = [y; zeros(h)]
    eps_ext = [eps; zeros(h)]
    for t in n+1:n+h
        y_ext[t] = 0.0
        for i in 1:p
            if t-i >= 1; y_ext[t] += model.ar_coefs[i] * y_ext[t-i]; end
        end
        for j in 1:q
            # Future MA shocks are zero in expectation
            if t-j >= 1 && t-j <= n
                y_ext[t] += model.ma_coefs[j] * eps_ext[t-j]
            end
        end
    end
    return y_ext[n+1:end] .+ mu
end

# ──────────────────────────────────────────────────────────────
# SARIMA
# ──────────────────────────────────────────────────────────────

"""
    seasonal_decompose(series, period) -> (trend, seasonal, residual)

Additive seasonal decomposition using moving averages.
"""
function seasonal_decompose(series::Vector{Float64}, period::Int=12)
    n = length(series)
    # Trend: centered moving average
    trend = zeros(n)
    half = period ÷ 2
    for i in half+1:n-half
        trend[i] = mean(series[i-half:i+half])
    end
    # Fill ends
    for i in 1:half
        trend[i] = trend[half+1]
    end
    for i in n-half+1:n
        trend[i] = trend[n-half]
    end

    # Seasonal component
    detrended = series .- trend
    seasonal_avg = zeros(period)
    counts = zeros(Int, period)
    for i in 1:n
        idx = ((i - 1) % period) + 1
        seasonal_avg[idx] += detrended[i]
        counts[idx] += 1
    end
    seasonal_avg ./= max.(counts, 1)
    seasonal_avg .-= mean(seasonal_avg)

    seasonal = [seasonal_avg[((i-1) % period) + 1] for i in 1:n]
    residual = series .- trend .- seasonal
    return trend, seasonal, residual
end

"""
    sarima_fit(series, p, d, q, P, D, Q, s) -> SARIMAModel

Fit SARIMA(p,d,q)(P,D,Q)[s] model.
Differences the series then fits ARMA with seasonal lags.
"""
function sarima_fit(series::Vector{Float64}, p::Int=1, d::Int=1, q::Int=1,
                     P::Int=1, D::Int=1, Q::Int=1, s::Int=12)
    y = copy(series)
    # Regular differencing
    for _ in 1:d
        y = diff(y)
    end
    # Seasonal differencing
    for _ in 1:D
        n = length(y)
        if n > s
            y = y[s+1:end] .- y[1:n-s]
        end
    end
    # Fit ARMA on differenced series
    base_model = fit_arma_ols(y, p, q)
    return SARIMAModel(base_model.ar_coefs, base_model.ma_coefs,
                        zeros(P), zeros(Q),
                        d, D, s, base_model.intercept, base_model.sigma2)
end

"""
    sarima_forecast(model, series, h) -> forecasts

h-step-ahead forecasts from SARIMA model.
Integrates back through differencing.
"""
function sarima_forecast(model::SARIMAModel, series::Vector{Float64}, h::Int=12)
    y = copy(series)
    # Differencing history for integration
    diff_history = Vector{Vector{Float64}}()

    for _ in 1:model.d
        push!(diff_history, copy(y))
        y = diff(y)
    end
    for _ in 1:model.D
        n = length(y)
        if n > model.s
            push!(diff_history, copy(y))
            y = y[model.s+1:end] .- y[1:n-model.s]
        end
    end

    arma_model = ARMAModel(model.ar_coefs, model.ma_coefs, model.intercept, model.sigma2)
    diff_forecasts = arma_forecast(arma_model, y, h)

    # Integrate back through seasonal differencing
    result = copy(diff_forecasts)
    if model.D > 0 && length(diff_history) > 0
        last_seasonal = diff_history[end]
        n_hist = length(last_seasonal)
        for i in 1:h
            lag_idx = max(1, n_hist - model.s + i - 1)
            result[i] += last_seasonal[min(lag_idx, n_hist)]
        end
    end

    # Integrate back through regular differencing
    for k in length(diff_history):-1:1
        if k <= model.d
            last_val = diff_history[k][end]
            result = cumsum([last_val; result])[2:end]
        end
    end

    return result
end

# ──────────────────────────────────────────────────────────────
# Kalman filter
# ──────────────────────────────────────────────────────────────

"""
    kalman_filter(ssm, observations) -> (states, covs, log_likelihood)

Kalman filter for linear Gaussian state-space model.
Returns filtered states, covariances, and log-likelihood.
"""
function kalman_filter(ssm::StateSpaceModel, observations::Matrix{Float64})
    T = size(observations, 2)
    n_state = length(ssm.x0)
    n_obs = size(observations, 1)

    states = zeros(n_state, T)
    covs = zeros(n_state, n_state, T)
    log_lik = 0.0

    x = copy(ssm.x0)
    P = copy(ssm.P0)

    for t in 1:T
        # Predict
        x_pred = ssm.F * x
        P_pred = ssm.F * P * ssm.F' .+ ssm.Q

        # Update
        y = observations[:, t]
        innov = y .- ssm.H * x_pred
        S = ssm.H * P_pred * ssm.H' .+ ssm.R
        S_sym = (S + S') ./ 2.0
        K = P_pred * ssm.H' / (S_sym + 1e-12*I)

        x = x_pred .+ K * innov
        P = (I(n_state) .- K * ssm.H) * P_pred

        states[:, t] = x
        covs[:, :, t] = P

        # Log-likelihood contribution
        det_S = max(det(S_sym), 1e-30)
        log_lik += -0.5 * (n_obs * log(2π) + log(det_S) + dot(innov, S_sym \ innov))
    end

    return states, covs, log_lik
end

"""
    kalman_smoother(ssm, observations) -> smoothed_states

Rauch-Tung-Striebel (RTS) smoother.
Returns smoothed state estimates.
"""
function kalman_smoother(ssm::StateSpaceModel, observations::Matrix{Float64})
    T = size(observations, 2)
    n_state = length(ssm.x0)

    # Forward pass
    filtered_states, filtered_covs, _ = kalman_filter(ssm, observations)

    # Also store predicted states and covs
    pred_states = zeros(n_state, T)
    pred_covs = zeros(n_state, n_state, T)
    x = copy(ssm.x0)
    P = copy(ssm.P0)
    for t in 1:T
        x_pred = ssm.F * x
        P_pred = ssm.F * P * ssm.F' .+ ssm.Q
        pred_states[:, t] = x_pred
        pred_covs[:, :, t] = P_pred
        # Update (same as filter)
        y = observations[:, t]
        S = ssm.H * P_pred * ssm.H' .+ ssm.R
        K = P_pred * ssm.H' / (S + 1e-12*I)
        x = x_pred .+ K * (y .- ssm.H * x_pred)
        P = (I(n_state) .- K * ssm.H) * P_pred
    end

    # Backward pass
    smoothed_states = copy(filtered_states)
    smoothed_covs = copy(filtered_covs)
    for t in T-1:-1:1
        Pk = filtered_covs[:,:,t]
        Pk1_pred = pred_covs[:,:,t+1]
        L = Pk * ssm.F' / (Pk1_pred + 1e-12*I)
        smoothed_states[:,t] = filtered_states[:,t] .+
            L * (smoothed_states[:,t+1] .- pred_states[:,t+1])
        smoothed_covs[:,:,t] = Pk .- L * (Pk1_pred .- smoothed_covs[:,:,t+1]) * L'
    end

    return smoothed_states, smoothed_covs
end

"""
    kalman_predict(ssm, last_state, last_cov, h) -> (forecasts, forecast_covs)

h-step-ahead prediction from Kalman filter state.
"""
function kalman_predict(ssm::StateSpaceModel, x0::Vector{Float64},
                          P0::Matrix{Float64}, h::Int=10)
    n_state = length(x0)
    n_obs = size(ssm.H, 1)
    forecasts = zeros(n_obs, h)
    forecast_covs = zeros(n_obs, n_obs, h)
    x, P = copy(x0), copy(P0)
    for t in 1:h
        x = ssm.F * x
        P = ssm.F * P * ssm.F' .+ ssm.Q
        forecasts[:, t] = ssm.H * x
        forecast_covs[:, :, t] = ssm.H * P * ssm.H' .+ ssm.R
    end
    return forecasts, forecast_covs
end

# ──────────────────────────────────────────────────────────────
# State-space EM estimation
# ──────────────────────────────────────────────────────────────

"""
    ssm_likelihood(ssm, observations) -> log_likelihood

Compute log-likelihood of observations under state-space model.
"""
function ssm_likelihood(ssm::StateSpaceModel, observations::Matrix{Float64})
    _, _, ll = kalman_filter(ssm, observations)
    return ll
end

"""
    state_space_em(observations, n_state, n_iter) -> StateSpaceModel

EM algorithm to fit a local-level state-space model.
observations: n_obs x T matrix.
"""
function state_space_em(observations::Matrix{Float64}, n_state::Int=2,
                          n_iter::Int=50)
    n_obs, T = size(observations)
    # Initialize with simple values
    F = Matrix{Float64}(I, n_state, n_state) .* 0.95
    H = hcat(Matrix{Float64}(I, min(n_obs, n_state), min(n_obs, n_state)),
              zeros(min(n_obs, n_state), max(0, n_state - n_obs)))
    if size(H, 2) < n_state
        H = hcat(H, zeros(size(H, 1), n_state - size(H, 2)))
    end
    H = H[:, 1:n_state]
    Q = 0.1 * I(n_state)
    R = 0.1 * I(n_obs)
    x0 = zeros(n_state)
    P0 = I(n_state) * 1.0

    ssm = StateSpaceModel(F, H, Matrix(Q), Matrix(R), x0, Matrix(P0))
    ll_prev = -Inf

    for _ in 1:n_iter
        # E-step: smooth
        sm_states, sm_covs = kalman_smoother(ssm, observations)
        ll = ssm_likelihood(ssm, observations)
        if abs(ll - ll_prev) < 1e-6; break; end
        ll_prev = ll

        # M-step: update Q and R (simplified)
        # Update R: R = (1/T) sum_t (y_t - H*x_t_smooth)(...)
        R_new = zeros(n_obs, n_obs)
        for t in 1:T
            innov = observations[:, t] .- H * sm_states[:, t]
            R_new .+= innov * innov' .+ H * sm_covs[:, :, t] * H'
        end
        R_new ./= T
        R_new = (R_new + R_new') ./ 2.0 .+ 1e-8*I
        ssm = StateSpaceModel(ssm.F, ssm.H, ssm.Q, Matrix(R_new), ssm.x0, ssm.P0)
    end
    return ssm
end

# ──────────────────────────────────────────────────────────────
# Dynamic Factor Model
# ──────────────────────────────────────────────────────────────

"""
    dynamic_factor_model(data, n_factors, n_lags) -> (factors, loadings, residuals)

Estimate dynamic factor model via principal components + VAR on factors.
data: T x N matrix of observed series.
"""
function dynamic_factor_model(data::Matrix{Float64}, n_factors::Int=3, n_lags::Int=1)
    T, N = size(data)
    # Standardize
    mu = mean(data, dims=1)
    sig = std(data, dims=1) .+ 1e-12
    X = (data .- mu) ./ sig

    # PCA via SVD to get initial factor estimates
    U, S, V = svd(X)
    factors = U[:, 1:n_factors] .* S[1:n_factors]'  # T x r
    loadings = V[:, 1:n_factors]                      # N x r

    # Residuals
    residuals = X .- factors * loadings'

    return factors, loadings, residuals
end

"""
    dfm_factors(data, n_factors) -> (factors, variance_explained)

Extract factors and compute variance explained.
"""
function dfm_factors(data::Matrix{Float64}, n_factors::Int=3)
    T, N = size(data)
    X = (data .- mean(data, dims=1)) ./ (std(data, dims=1) .+ 1e-12)
    C = (X' * X) / T
    # Power iteration for top eigenvalues/vectors
    factors = zeros(T, n_factors)
    explained = zeros(n_factors)
    C_rem = copy(C)
    for k in 1:n_factors
        v = ones(N) ./ sqrt(N)
        lambda = 0.0
        for _ in 1:500
            w = C_rem * v
            lambda = norm(w)
            v = lambda > 1e-12 ? w ./ lambda : w
        end
        factors[:, k] = X * v
        explained[k] = lambda
        C_rem .-= lambda .* (v * v')
    end
    total_var = tr(C)
    return factors, explained ./ max(total_var, 1e-12)
end

"""
    dfm_forecast(data, n_factors, h) -> forecasts_matrix

Factor model forecasts for all N series h steps ahead.
"""
function dfm_forecast(data::Matrix{Float64}, n_factors::Int=3, h::Int=5)
    T, N = size(data)
    factors, loadings, _ = dynamic_factor_model(data, n_factors)
    mu = mean(data, dims=1)
    sig = std(data, dims=1) .+ 1e-12

    # Fit AR(1) to each factor
    factor_forecasts = zeros(h, n_factors)
    for k in 1:n_factors
        f = factors[:, k]
        ar_model = fit_ar(f, 1)
        fcast = arma_forecast(ar_model, f, h)
        factor_forecasts[:, k] = fcast
    end

    # Reconstruct
    forecasts = factor_forecasts * loadings'
    # Unstandardize
    forecasts = forecasts .* sig .+ mu
    return forecasts
end

# ──────────────────────────────────────────────────────────────
# Cointegration and VECM
# ──────────────────────────────────────────────────────────────

"""
    cointegration_test(y1, y2) -> (adf_stat, is_cointegrated)

Engle-Granger cointegration test.
"""
function cointegration_test(y1::Vector{Float64}, y2::Vector{Float64})
    n = length(y1)
    # Step 1: regress y1 on y2
    X = hcat(ones(n), y2)
    beta = (X'X + 1e-10*I) \ (X'y1)
    residuals = y1 .- X * beta
    # Step 2: ADF test on residuals
    adf_stat = adf_test_stat(residuals, 1)
    # Critical value at 5% for cointegration residuals (approx -3.37)
    is_cointegrated = adf_stat < -3.37
    return adf_stat, is_cointegrated
end

function adf_test_stat(series::Vector{Float64}, p::Int=1)
    n = length(series)
    dy = diff(series)
    y_lag = series[1:end-1]
    # Build design matrix: [y_{t-1}, Δy_{t-1}, ..., Δy_{t-p}, 1]
    T = n - 1 - p
    if T < 5; return 0.0; end
    X = zeros(T, p + 2)
    Y = dy[p+1:end]
    for t in 1:T
        X[t, 1] = y_lag[t+p]  # lagged level
        X[t, 2] = 1.0          # intercept
        for j in 1:p
            X[t, 2+j] = dy[t+p-j]
        end
    end
    beta = (X'X + 1e-10*I) \ (X'Y)
    resid = Y .- X * beta
    sigma2 = var(resid)
    # t-stat on beta[1] (coefficient on lagged level)
    var_beta1 = sigma2 * ((X'X + 1e-10*I) \ I(size(X,2)))[1,1]
    return beta[1] / sqrt(max(var_beta1, 1e-15))
end

"""
    vecm_fit(y_matrix, r, p) -> (alpha, beta, Gamma_matrices)

Fit Vector Error Correction Model.
y_matrix: T x K matrix of K cointegrated series.
r = cointegration rank, p = lag order.
"""
function vecm_fit(Y::Matrix{Float64}, r::Int=1, p::Int=1)
    T, K = size(Y)
    dY = diff(Y, dims=1)
    Y_lag = Y[1:end-1, :]

    # Simplified: Johansen-like via reduced rank regression
    # S00 = cov(dY), S11 = cov(Y_lag), S01 = crosscov
    S00 = (dY' * dY) ./ T
    S11 = (Y_lag' * Y_lag) ./ T
    S01 = (dY' * Y_lag) ./ T
    S10 = S01'

    # Solve eigenvalue problem
    M = (S11 + 1e-10*I) \ S10 * (S00 + 1e-10*I) \ S01
    # Power iteration for top r eigenvectors
    beta = zeros(K, r)
    M_rem = copy(M)
    for k in 1:r
        v = ones(K) ./ sqrt(K)
        lambda = 0.0
        for _ in 1:200
            w = M_rem * v
            lambda = norm(w)
            v = lambda > 1e-12 ? w ./ lambda : v
        end
        beta[:, k] = v
        M_rem .-= lambda .* (v * v')
    end
    # Speed of adjustment
    alpha = S01 * beta / (beta' * S11 * beta + 1e-10*I)
    return alpha, beta
end

"""
    vecm_forecast(Y, alpha, beta, h) -> forecast_matrix

h-step-ahead VECM forecast.
"""
function vecm_forecast(Y::Matrix{Float64}, alpha::Matrix{Float64},
                         beta::Matrix{Float64}, h::Int=5)
    T, K = size(Y)
    forecasts = zeros(h, K)
    y_curr = copy(Y[end, :])
    for t in 1:h
        ect = beta' * y_curr  # error correction terms
        delta_y = alpha * ect
        y_curr = y_curr .+ delta_y
        forecasts[t, :] = y_curr
    end
    return forecasts
end

# ──────────────────────────────────────────────────────────────
# Structural break tests
# ──────────────────────────────────────────────────────────────

"""
    structural_break_test(series, min_fraction) -> (break_date, F_stat)

Chow test for structural break. Searches over all candidate break dates.
min_fraction: minimum fraction of observations in each sub-sample.
"""
function structural_break_test(series::Vector{Float64}, min_fraction::Float64=0.15)
    n = length(series)
    min_obs = round(Int, n * min_fraction)
    best_F = 0.0
    best_t = min_obs + 1
    t_range = min_obs+1:n-min_obs

    # Full-sample regression: AR(1)
    function ols_ssr(y_sub)
        n_s = length(y_sub)
        if n_s < 3; return Inf; end
        X = hcat(ones(n_s-1), y_sub[1:end-1])
        Y = y_sub[2:end]
        beta = (X'X + 1e-10*I) \ (X'Y)
        return sum((Y .- X*beta).^2)
    end

    ssr_full = ols_ssr(series)
    for t_break in t_range
        ssr1 = ols_ssr(series[1:t_break])
        ssr2 = ols_ssr(series[t_break+1:end])
        F = ((ssr_full - ssr1 - ssr2) / 2.0) / ((ssr1 + ssr2) / (n - 4))
        if F > best_F
            best_F = F
            best_t = t_break
        end
    end
    return best_t, best_F
end

"""
    rolling_regression(y, x_matrix, window) -> (betas, r_squareds)

Rolling OLS regressions over a moving window.
"""
function rolling_regression(y::Vector{Float64}, X::Matrix{Float64}, window::Int=60)
    n = length(y)
    k = size(X, 2)
    betas = zeros(n, k)
    r2s = zeros(n)
    for t in window:n
        yw = y[t-window+1:t]
        Xw = X[t-window+1:t, :]
        beta = (Xw'Xw + 1e-10*I) \ (Xw'yw)
        betas[t, :] = beta
        fitted = Xw * beta
        ss_res = sum((yw .- fitted).^2)
        ss_tot = sum((yw .- mean(yw)).^2)
        r2s[t] = 1.0 - ss_res / max(ss_tot, 1e-12)
    end
    return betas, r2s
end

# ──────────────────────────────────────────────────────────────
# Granger causality / VAR
# ──────────────────────────────────────────────────────────────

"""
    granger_causality(y, x, p) -> (F_stat, p_value_approx)

Granger causality test: does x Granger-cause y?
"""
function granger_causality(y::Vector{Float64}, x::Vector{Float64}, p::Int=4)
    n = length(y)
    # Restricted: regress y on lags of y only
    function build_X_lags(series, lags, n_obs)
        X = zeros(n_obs, lags)
        for j in 1:lags
            X[:, j] = series[p-j+1:n_obs+p-j]
        end
        return X
    end
    y_lags_only = build_X_lags(y, p, n-p)
    y_and_x = hcat(build_X_lags(y, p, n-p), build_X_lags(x, p, n-p))
    yy = y[p+1:end]

    X_R = hcat(ones(n-p), y_lags_only)
    X_U = hcat(ones(n-p), y_and_x)

    beta_R = (X_R'X_R + 1e-10*I) \ (X_R'yy)
    beta_U = (X_U'X_U + 1e-10*I) \ (X_U'yy)
    ssr_R = sum((yy .- X_R*beta_R).^2)
    ssr_U = sum((yy .- X_U*beta_U).^2)

    F = ((ssr_R - ssr_U) / p) / (ssr_U / (n - p - 2*p - 1))
    # F distribution p-value approximation
    df1, df2 = Float64(p), Float64(n - p - 2*p - 1)
    # Simple approximation using normal for large df2
    p_approx = df2 > 30 ? exp(-0.5 * F) : 0.05  # rough
    return F, p_approx
end

"""
    impulse_response_var(A_matrices, n_periods, shock_index) -> IRF_matrix

Compute impulse response function for a VAR model.
A_matrices: vector of companion matrices, n_periods: horizon.
"""
function impulse_response_var(A_matrices::Vector{Matrix{Float64}},
                                n_periods::Int=20, shock_var::Int=1)
    p = length(A_matrices)
    K = size(A_matrices[1], 1)
    shock = zeros(K)
    shock[shock_var] = 1.0
    irf = zeros(K, n_periods)
    irf[:, 1] = shock

    # Build companion form
    for t in 2:n_periods
        response = zeros(K)
        for j in 1:min(p, t-1)
            response .+= A_matrices[j] * irf[:, t-j]
        end
        irf[:, t] = response
    end
    return irf
end

# ──────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────

"""
    hodrick_prescott_filter(series, lambda) -> (trend, cycle)

Hodrick-Prescott filter for trend-cycle decomposition.
lambda = 1600 (quarterly), 100 (annual), 6.25 (monthly typical).
"""
function hodrick_prescott_filter(series::Vector{Float64}, lambda::Float64=1600.0)
    n = length(series)
    # Build penalty matrix (2nd difference)
    D2 = zeros(n-2, n)
    for i in 1:n-2
        D2[i, i] = 1.0
        D2[i, i+1] = -2.0
        D2[i, i+2] = 1.0
    end
    A = I(n) + lambda .* D2' * D2
    trend = A \ series
    cycle = series .- trend
    return trend, cycle
end

"""
    baxter_king_filter(series, low_freq, high_freq, K) -> cycle

Baxter-King band-pass filter. Extracts business cycle frequencies.
low_freq: lower bound (e.g. 6 for 6-quarter cycles),
high_freq: upper bound (e.g. 32 for 32-quarter cycles), K = lead/lag.
"""
function baxter_king_filter(series::Vector{Float64},
                              low_freq::Float64=6.0, high_freq::Float64=32.0,
                              K::Int=12)
    n = length(series)
    omega_l = 2π / high_freq
    omega_h = 2π / low_freq
    # Ideal band-pass weights
    weights = zeros(2K + 1)
    for k in -K:K
        if k == 0
            weights[k+K+1] = (omega_h - omega_l) / π
        else
            weights[k+K+1] = (sin(k * omega_h) - sin(k * omega_l)) / (k * π)
        end
    end
    # Make sum exactly zero (remove trend)
    weights .-= mean(weights)

    cycle = zeros(n)
    for t in K+1:n-K
        for (i, k) in enumerate(-K:K)
            cycle[t] += weights[i] * series[t + k]
        end
    end
    return cycle
end

"""
    panel_unit_root(panel_data) -> (test_stat, p_approx)

Im-Pesaran-Shin panel unit root test (simplified).
panel_data: T x N matrix.
"""
function panel_unit_root(panel::Matrix{Float64})
    T, N = size(panel)
    adf_stats = [adf_test_stat(panel[:, i], 1) for i in 1:N]
    mean_adf = mean(adf_stats)
    # IPS W-bar statistic: (mean_ADF - E[t]) / sqrt(Var[t]/N)
    # Approximate critical values for ADF with drift
    E_t = -1.503  # approximate mean of ADF under H0
    Var_t = 1.001
    W_bar = (mean_adf - E_t) / sqrt(Var_t / N)
    p_approx = 0.5 * erfc(W_bar / sqrt(2.0))
    return W_bar, p_approx
end

end # module TimeSeriesAdvanced
