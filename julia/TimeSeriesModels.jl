module TimeSeriesModels

# Time series models: ARIMA, Cointegration (Engle-Granger, Johansen),
# Granger Causality, State Space (Kalman Filter/Smoother)
# For production use in the SRFM quant trading system.

using LinearAlgebra
using Statistics
using Random
using Test

export ARIMAModel, fit_arima, forecast, aic
export CointegrationResult, engle_granger
export JohansenResult, johansen_test
export GrangerResult, granger_causality
export KalmanState, SmoothedState, kalman_filter, kalman_smoother

# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

"""
    ARIMAModel

Fitted ARIMA(p, d, q) model.

Fields:
- `p`: AR order
- `d`: integration order
- `q`: MA order
- `phi`: AR coefficients (length p)
- `theta`: MA coefficients (length q)
- `sigma2`: residual variance
- `log_likelihood`: log-likelihood at optimum
- `n`: number of observations used in fitting
- `original_series`: original (un-differenced) series for back-integration
- `diff_series`: d-times differenced series
"""
@kwdef struct ARIMAModel
    p::Int
    d::Int
    q::Int
    phi::Vector{Float64}
    theta::Vector{Float64}
    sigma2::Float64
    log_likelihood::Float64
    n::Int
    original_series::Vector{Float64}
    diff_series::Vector{Float64}
end

"""
    _difference(series, d) -> Vector{Float64}

Apply d-th order differencing to series.
"""
function _difference(series::Vector{Float64}, d::Int)::Vector{Float64}
    result = copy(series)
    for _ in 1:d
        result = diff(result)
    end
    return result
end

"""
    _undifference(diff_series, original, d) -> Vector{Float64}

Invert d-th order differencing using the last d values of the original series
as initial conditions.
"""
function _undifference(diff_series::Vector{Float64}, original::Vector{Float64}, d::Int)::Vector{Float64}
    result = copy(diff_series)
    for i in 1:d
        # Use the d-i'th last element of original as starting value
        start_val = original[end - (d - i)]
        result = cumsum(vcat(start_val, result))[2:end] .+ start_val
        result = vcat(start_val, result)
        result = result[2:end]
    end
    return result
end

"""
    _arma_innovations(series, phi, theta) -> (residuals, ll)

Compute ARMA innovations and log-likelihood for given parameters.
Uses recursive computation of residuals.
"""
function _arma_innovations(series::Vector{Float64}, phi::Vector{Float64},
                            theta::Vector{Float64})::Tuple{Vector{Float64}, Float64}
    n = length(series)
    p = length(phi)
    q = length(theta)
    resid = zeros(Float64, n)

    for t in 1:n
        # AR part
        ar_part = 0.0
        for i in 1:min(p, t-1)
            ar_part += phi[i] * series[t - i]
        end
        # MA part
        ma_part = 0.0
        for j in 1:min(q, t-1)
            ma_part += theta[j] * resid[t - j]
        end
        resid[t] = series[t] - ar_part - ma_part
    end

    sigma2 = mean(resid .^ 2)
    sigma2 = max(sigma2, 1e-10)
    ll = -0.5 * n * (log(2 * pi * sigma2) + 1.0)

    return resid, ll
end

"""
    _fit_arma_ols(series, p, q) -> (phi, theta, sigma2, ll)

Fit ARMA(p,q) via OLS (Hannan-Rissanen approximate method).
Step 1: Fit high-order AR to get residuals.
Step 2: OLS regression of series on lagged series and lagged residuals.
"""
function _fit_arma_ols(series::Vector{Float64}, p::Int, q::Int)::Tuple{Vector{Float64}, Vector{Float64}, Float64, Float64}
    n = length(series)

    if p == 0 && q == 0
        sigma2 = var(series)
        _, ll = _arma_innovations(series, Float64[], Float64[])
        return Float64[], Float64[], sigma2, ll
    end

    # Step 1: Fit AR(m) where m = max(p, q) + some extra lags
    m = max(p, q, 5)
    m = min(m, div(n, 4))

    phi_long, _, _, _ = _fit_ar(series, m)
    resid_long, _ = _arma_innovations(series, phi_long, Float64[])

    # Step 2: Build design matrix [lagged_y | lagged_resid]
    max_lag = max(p, q)
    start = max_lag + 1

    if start >= n
        # Fallback to pure AR
        if p > 0
            phi_hat, sigma2, ll = _fit_ar_ols(series, p)
            return phi_hat, zeros(q), sigma2, ll
        else
            sigma2 = var(series)
            return Float64[], zeros(q), sigma2, -0.5 * n * log(2*pi*sigma2) - n/2
        end
    end

    nrows = n - start + 1
    ncols = p + q

    if ncols == 0
        sigma2 = var(series)
        return Float64[], Float64[], sigma2, -0.5*n*log(2*pi*sigma2) - n/2
    end

    X = zeros(Float64, nrows, ncols)
    y = series[start:n]

    for i in 1:nrows
        t = start + i - 1
        col = 1
        for lag in 1:p
            X[i, col] = t - lag >= 1 ? series[t - lag] : 0.0
            col += 1
        end
        for lag in 1:q
            X[i, col] = t - lag >= 1 ? resid_long[t - lag] : 0.0
            col += 1
        end
    end

    # OLS
    beta = if size(X, 1) > size(X, 2)
        (X' * X + 1e-8 * I(ncols)) \ (X' * y)
    else
        zeros(ncols)
    end

    phi_hat = p > 0 ? beta[1:p] : Float64[]
    theta_hat = q > 0 ? beta[(p+1):(p+q)] : Float64[]

    # Stationarity check: bound AR roots
    phi_hat = _bound_ar_coefs(phi_hat)

    resid, ll = _arma_innovations(series, phi_hat, theta_hat)
    sigma2 = mean(resid .^ 2)

    return phi_hat, theta_hat, max(sigma2, 1e-10), ll
end

"""
    _fit_ar(series, p) -> (phi, sigma2, ll, resid)

Fit AR(p) via OLS (Yule-Walker).
"""
function _fit_ar(series::Vector{Float64}, p::Int)::Tuple{Vector{Float64}, Float64, Float64, Vector{Float64}}
    if p <= 0
        return Float64[], var(series), -Inf, series .- mean(series)
    end
    phi, sigma2, ll = _fit_ar_ols(series, p)
    resid, _ = _arma_innovations(series, phi, Float64[])
    return phi, sigma2, ll, resid
end

function _fit_ar_ols(series::Vector{Float64}, p::Int)::Tuple{Vector{Float64}, Float64, Float64}
    n = length(series)
    if n <= p
        return zeros(p), var(series), -Inf
    end
    nrows = n - p
    X = zeros(Float64, nrows, p)
    y = series[(p+1):n]
    for i in 1:nrows
        for j in 1:p
            X[i, j] = series[p + i - j]
        end
    end
    beta = (X' * X + 1e-8 * I(p)) \ (X' * y)
    resid = y .- X * beta
    sigma2 = max(mean(resid .^ 2), 1e-10)
    ll = -0.5 * nrows * (log(2 * pi * sigma2) + 1.0)
    return beta, sigma2, ll
end

"""
    _bound_ar_coefs(phi) -> Vector{Float64}

Ensure AR coefficients do not produce explosive roots by bounding.
"""
function _bound_ar_coefs(phi::Vector{Float64})::Vector{Float64}
    return clamp.(phi, -0.99, 0.99)
end

"""
    fit_arima(series, p, d, q) -> ARIMAModel

Fit an ARIMA(p, d, q) model to the given time series using the Hannan-Rissanen
approximate MLE method via Kalman filter state-space representation.

# Arguments
- `series`: observed univariate time series
- `p`: AR order
- `d`: integration order (number of differences)
- `q`: MA order

# Returns
Fitted ARIMAModel struct.
"""
function fit_arima(series::Vector{Float64}, p::Int, d::Int, q::Int)::ARIMAModel
    @assert p >= 0 "AR order p must be >= 0"
    @assert d >= 0 "Integration order d must be >= 0"
    @assert q >= 0 "MA order q must be >= 0"
    @assert length(series) > p + d + q + 5 "Series too short for given orders"

    original = copy(series)
    diff_s = _difference(series, d)
    n = length(diff_s)

    phi, theta, sigma2, ll = _fit_arma_ols(diff_s, p, q)

    return ARIMAModel(
        p=p, d=d, q=q,
        phi=phi,
        theta=theta,
        sigma2=sigma2,
        log_likelihood=ll,
        n=n,
        original_series=original,
        diff_series=diff_s
    )
end

"""
    forecast(model, h) -> Vector{Float64}

Generate h-step ahead forecasts from a fitted ARIMAModel.

# Arguments
- `model`: fitted ARIMAModel
- `h`: number of steps ahead to forecast

# Returns
Vector of h forecasted values in original (non-differenced) scale.
"""
function forecast(model::ARIMAModel, h::Int)::Vector{Float64}
    @assert h >= 1 "Forecast horizon h must be >= 1"

    diff_s = model.diff_series
    phi = model.phi
    theta = model.theta
    p = model.p
    q = model.q
    n = length(diff_s)

    # Compute in-sample residuals
    resid, _ = _arma_innovations(diff_s, phi, theta)

    # Extend series with forecasts
    extended = copy(diff_s)
    ext_resid = copy(resid)

    for t in 1:h
        idx = n + t
        ar_part = 0.0
        for i in 1:p
            ar_part += phi[i] * extended[idx - i]
        end
        # MA part uses zeros for future innovations
        ma_part = 0.0
        for j in 1:q
            ridx = idx - j
            ma_part += theta[j] * (ridx <= n ? ext_resid[ridx] : 0.0)
        end
        push!(extended, ar_part + ma_part)
        push!(ext_resid, 0.0)
    end

    forecasts_diff = extended[(n+1):end]

    # Undo differencing
    if model.d == 0
        return forecasts_diff
    end

    # Reconstruct from differenced forecasts using original series tail
    result = forecasts_diff
    orig = model.original_series

    for step in 1:model.d
        # The last value before forecasting at this differencing level
        level_val = orig[end - (model.d - step)]
        result = cumsum(vcat(result)) .+ level_val
    end

    return result
end

"""
    aic(model) -> Float64

Compute Akaike Information Criterion for fitted ARIMAModel.
AIC = -2 * log_likelihood + 2 * k, where k = p + q + 1 (sigma2).
"""
function aic(model::ARIMAModel)::Float64
    k = model.p + model.q + 1
    return -2.0 * model.log_likelihood + 2.0 * k
end

# ---------------------------------------------------------------------------
# Cointegration -- Engle-Granger
# ---------------------------------------------------------------------------

"""
    CointegrationResult

Result of a cointegration test.

Fields:
- `cointegrating_vector`: [1, -beta] where beta is estimated
- `residuals`: cointegrating residuals
- `adf_stat`: ADF test statistic on residuals
- `p_value`: approximate p-value (MacKinnon critical values)
- `is_cointegrated`: Bool, true if reject null of no cointegration at 5%
- `beta`: cointegrating coefficient (y2 coefficient)
"""
@kwdef struct CointegrationResult
    cointegrating_vector::Vector{Float64}
    residuals::Vector{Float64}
    adf_stat::Float64
    p_value::Float64
    is_cointegrated::Bool
    beta::Float64
end

"""
    _adf_test(series, lags=1) -> (stat, p_value)

Augmented Dickey-Fuller test for unit root.
H0: unit root present (non-stationary).
Returns test statistic and approximate p-value.
"""
function _adf_test(series::Vector{Float64}, lags::Int=1)::Tuple{Float64, Float64}
    n = length(series)
    dy = diff(series)
    m = length(dy)

    # Build regression: dy_t = alpha + beta * y_{t-1} + sum_j gamma_j * dy_{t-j} + eps
    start = lags + 1
    nrows = m - lags
    if nrows <= lags + 3
        return 0.0, 0.5
    end

    ncols = 2 + lags  # [const, y_{t-1}, dy_{t-1}, ..., dy_{t-lags}]
    X = zeros(Float64, nrows, ncols)
    y_reg = dy[(lags+1):m]

    for i in 1:nrows
        t = lags + i
        X[i, 1] = 1.0  # constant
        X[i, 2] = series[t]  # y_{t-1}
        for j in 1:lags
            X[i, 2 + j] = dy[t - j]  # lagged differences
        end
    end

    beta = (X' * X + 1e-10 * I(ncols)) \ (X' * y_reg)
    resid = y_reg .- X * beta
    sigma2 = mean(resid .^ 2)
    sigma2 = max(sigma2, 1e-12)

    # Standard error of beta[2] (coefficient on y_{t-1})
    XtX_inv = inv(X' * X + 1e-10 * I(ncols))
    se_beta = sqrt(max(sigma2 * XtX_inv[2, 2], 1e-12))
    t_stat = beta[2] / se_beta

    # Approximate p-value using MacKinnon (1994) response surface for n>=25
    # Critical values for no-trend case: 1%=-3.51, 5%=-2.89, 10%=-2.58
    p_val = if t_stat < -3.51
        0.01
    elseif t_stat < -2.89
        0.05
    elseif t_stat < -2.58
        0.10
    else
        0.50
    end

    return t_stat, p_val
end

"""
    engle_granger(y1, y2) -> CointegrationResult

Engle-Granger two-step cointegration test for a bivariate system.

Step 1: OLS regression y1 = alpha + beta * y2 + e
Step 2: ADF test on residuals e

H0: no cointegration (residuals have unit root).

# Arguments
- `y1`: first time series
- `y2`: second time series (must be same length as y1)

# Returns
CointegrationResult struct.
"""
function engle_granger(y1::Vector{Float64}, y2::Vector{Float64})::CointegrationResult
    @assert length(y1) == length(y2) "Series must have equal length"
    n = length(y1)

    # Step 1: OLS y1 = alpha + beta * y2
    X = [ones(n) y2]
    beta_vec = (X' * X + 1e-12 * I(2)) \ (X' * y1)
    alpha = beta_vec[1]
    beta = beta_vec[2]
    resid = y1 .- alpha .- beta .* y2

    # Step 2: ADF on residuals (no constant in ADF since residuals are mean-zero by construction)
    adf_stat, p_val = _adf_test(resid, 1)

    is_coint = p_val <= 0.05

    return CointegrationResult(
        cointegrating_vector=[1.0, -beta],
        residuals=resid,
        adf_stat=adf_stat,
        p_value=p_val,
        is_cointegrated=is_coint,
        beta=beta
    )
end

# ---------------------------------------------------------------------------
# Johansen Test
# ---------------------------------------------------------------------------

"""
    JohansenResult

Result of the Johansen cointegration test.

Fields:
- `trace_stats`: trace test statistics (n_vars values)
- `crit_vals`: 5% critical values (n_vars values)
- `n_cointegrating_vectors`: estimated number of cointegrating relations
- `alpha_matrix`: loading matrix (n_vars x r) for error correction
- `beta_matrix`: cointegrating vectors (n_vars x r)
"""
@kwdef struct JohansenResult
    trace_stats::Vector{Float64}
    crit_vals::Vector{Float64}
    n_cointegrating_vectors::Int
    alpha_matrix::Matrix{Float64}
    beta_matrix::Matrix{Float64}
end

"""
    johansen_test(Y, lag=1) -> JohansenResult

Johansen trace test for cointegration in a multivariate system.

Uses the reduced rank regression approach:
1. Regress dY and Y_{t-1} on lagged dY to remove short-run dynamics
2. Canonical correlation analysis on residuals
3. Trace test: -T * sum(log(1 - lambda_i))

# Arguments
- `Y`: T x n matrix of n time series
- `lag`: number of lags in the VAR (default 1)

# Returns
JohansenResult struct with trace statistics and critical values.
"""
function johansen_test(Y::Matrix{Float64}, lag::Int=1)::JohansenResult
    T, n = size(Y)
    @assert T > 2*lag + n + 5 "Sample too small for Johansen test"

    dY = diff(Y, dims=1)
    Y_lag1 = Y[lag:(T-1), :]  # levels lagged 1
    m = size(dY, 1)

    # Align
    start = lag + 1
    nobs = m - lag + 1
    if nobs <= n + lag
        # Return degenerate result
        return JohansenResult(
            trace_stats=zeros(n),
            crit_vals=_johansen_crit_vals(n),
            n_cointegrating_vectors=0,
            alpha_matrix=zeros(n, 1),
            beta_matrix=zeros(n, 1)
        )
    end

    dY_t = dY[lag:m, :]          # T-lag x n
    Y_t1 = Y_lag1[1:size(dY_t,1), :]   # T-lag x n

    # Build lagged dY matrix
    nobs2 = size(dY_t, 1)
    Z = zeros(Float64, nobs2, n * lag)
    for k in 1:lag
        for i in 1:nobs2
            src = i - k
            if src >= 1
                Z[i, ((k-1)*n+1):(k*n)] = dY[lag + i - 1 - k, :]
            end
        end
    end

    # Partial out lagged differences from dY_t and Y_t1
    function partial_out(M, Z)
        if size(Z, 2) == 0
            return M
        end
        beta = (Z' * Z + 1e-10 * I(size(Z,2))) \ (Z' * M)
        return M .- Z * beta
    end

    R0 = partial_out(dY_t, Z)
    R1 = partial_out(Y_t1, Z)

    # Moment matrices
    S00 = (R0' * R0) ./ nobs2
    S11 = (R1' * R1) ./ nobs2
    S01 = (R0' * R1) ./ nobs2

    # Solve generalized eigenvalue problem: S11^{-1} S10 S00^{-1} S01 beta = lambda beta
    S11_reg = S11 + 1e-8 * I(n)
    S00_reg = S00 + 1e-8 * I(n)

    M_mat = try
        inv(S11_reg) * S01' * inv(S00_reg) * S01
    catch
        zeros(n, n)
    end

    eig_result = eigen(Symmetric(M_mat + M_mat') ./ 2)
    lambdas = sort(real.(eig_result.values), rev=true)
    lambdas = clamp.(lambdas, 0.0, 1.0 - 1e-10)
    evecs = real.(eig_result.vectors[:, sortperm(real.(eig_result.values), rev=true)])

    # Trace statistics: trace_i = -T * sum_{j=i+1}^{n} log(1 - lambda_j)
    trace_stats = zeros(n)
    for i in 0:(n-1)
        trace_stats[i+1] = -nobs2 * sum(log(1.0 - lambdas[j]) for j in (i+1):n)
    end

    crit_5pct = _johansen_crit_vals(n)

    # Count cointegrating vectors (number of trace stats exceeding critical values)
    r = 0
    for i in 1:n
        if trace_stats[i] > crit_5pct[i]
            r += 1
        else
            break
        end
    end

    r_use = max(r, 1)
    beta_mat = evecs[:, 1:r_use]
    # Loading matrix alpha = S01 * beta * inv(beta' * S11 * beta)
    alpha_mat = try
        S01 * beta_mat * inv(beta_mat' * S11_reg * beta_mat)
    catch
        zeros(n, r_use)
    end

    return JohansenResult(
        trace_stats=trace_stats,
        crit_vals=crit_5pct,
        n_cointegrating_vectors=r,
        alpha_matrix=alpha_mat,
        beta_matrix=beta_mat
    )
end

"""
    _johansen_crit_vals(n) -> Vector{Float64}

Approximate 5% critical values for Johansen trace test (intercept model).
From Osterwald-Lenum (1992) Table 1.
"""
function _johansen_crit_vals(n::Int)::Vector{Float64}
    # 5% critical values for r = 0, 1, ..., n-1 hypotheses
    # These are approximate values for n up to 10
    base_vals = [15.41, 29.68, 47.21, 68.52, 87.31, 110.60, 131.70, 156.00, 182.82, 215.00]
    cv = Vector{Float64}(undef, n)
    for i in 1:n
        cv[i] = i <= length(base_vals) ? base_vals[i] : base_vals[end] + (i - length(base_vals)) * 30.0
    end
    return cv
end

# ---------------------------------------------------------------------------
# Granger Causality
# ---------------------------------------------------------------------------

"""
    GrangerResult

Result of Granger causality test.

Fields:
- `f_stat`: F-statistic for the test
- `p_value`: approximate p-value
- `does_x_cause_y`: Bool, true if we reject H0 (x does not Granger-cause y) at 5%
- `lag`: number of lags used
"""
@kwdef struct GrangerResult
    f_stat::Float64
    p_value::Float64
    does_x_cause_y::Bool
    lag::Int
end

"""
    granger_causality(x, y, lag=5) -> GrangerResult

Test whether x Granger-causes y using a bivariate VAR(lag) F-test.

H0: lagged values of x do not help predict y (x does not Granger-cause y).
The F-statistic compares restricted VAR (y on lags of y only) vs.
unrestricted VAR (y on lags of y and x).

# Arguments
- `x`: potential causing series
- `y`: series to be predicted
- `lag`: number of lags (default 5)

# Returns
GrangerResult struct.
"""
function granger_causality(x::Vector{Float64}, y::Vector{Float64}, lag::Int=5)::GrangerResult
    @assert length(x) == length(y) "Series must have equal length"
    n = length(y)
    @assert n > 2*lag + 5 "Series too short for lag=$lag"

    start = lag + 1
    nobs = n - lag
    q = lag  -- number of restrictions

    # Restricted model: y ~ const + lags of y
    X_res = zeros(Float64, nobs, lag + 1)
    for i in 1:nobs
        t = lag + i
        X_res[i, 1] = 1.0
        for j in 1:lag
            X_res[i, j+1] = y[t - j]
        end
    end
    y_dep = y[start:n]

    # Unrestricted model: y ~ const + lags of y + lags of x
    X_unres = zeros(Float64, nobs, 2*lag + 1)
    X_unres[:, 1:(lag+1)] = X_res
    for i in 1:nobs
        t = lag + i
        for j in 1:lag
            X_unres[i, lag + 1 + j] = x[t - j]
        end
    end

    function ols_rss(X, y_vec)
        beta = (X' * X + 1e-10 * I(size(X,2))) \ (X' * y_vec)
        resid = y_vec .- X * beta
        return sum(resid .^ 2), beta
    end

    rss_res, _ = ols_rss(X_res, y_dep)
    rss_unres, _ = ols_rss(X_unres, y_dep)

    k_res = size(X_res, 2)
    k_unres = size(X_unres, 2)

    df1 = k_unres - k_res
    df2 = nobs - k_unres

    if df2 <= 0 || df1 <= 0
        return GrangerResult(f_stat=0.0, p_value=1.0, does_x_cause_y=false, lag=lag)
    end

    rss_diff = max(rss_res - rss_unres, 0.0)
    mse_unres = max(rss_unres / df2, 1e-12)
    f_stat = (rss_diff / df1) / mse_unres

    # Approximate p-value from F distribution using Wilson-Hilferty transformation
    p_val = _f_pvalue(f_stat, df1, df2)

    return GrangerResult(
        f_stat=f_stat,
        p_value=p_val,
        does_x_cause_y=p_val < 0.05,
        lag=lag
    )
end

"""
    _f_pvalue(f, df1, df2) -> Float64

Approximate p-value for F(df1, df2) distribution using Wilson-Hilferty approximation.
"""
function _f_pvalue(f::Float64, df1::Int, df2::Int)::Float64
    # Use chi-squared approximation: df1*F ~ chi2(df1) for large df2
    chi2 = df1 * f
    # Approximate chi2 CDF survival using normal approximation
    k = Float64(df1)
    # Wilson-Hilferty: ((chi2/k)^(1/3) - (1 - 2/(9k))) / sqrt(2/(9k)) ~ N(0,1)
    mu_wh = 1.0 - 2.0 / (9.0 * k)
    sigma_wh = sqrt(2.0 / (9.0 * k))
    z = ((chi2 / k)^(1.0/3.0) - mu_wh) / max(sigma_wh, 1e-12)
    # 1 - Phi(z) approximation
    return _normal_survival(z)
end

"""
    _normal_survival(z) -> Float64

Approximate survival function P(Z > z) for standard normal Z.
Uses rational approximation by Abramowitz and Stegun (7.1.26).
"""
function _normal_survival(z::Float64)::Float64
    if z > 8.0
        return 0.0
    elseif z < -8.0
        return 1.0
    end
    # Standard normal CDF approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    phi = exp(-0.5 * z^2) / sqrt(2 * pi)
    cdf = 1.0 - phi * poly
    return z >= 0 ? 1.0 - cdf : cdf
end

# ---------------------------------------------------------------------------
# State Space / Kalman Filter
# ---------------------------------------------------------------------------

"""
    KalmanState

Result of Kalman filter recursion.

Fields:
- `x_filtered`: filtered state means (state_dim x T matrix)
- `P_filtered`: filtered state covariances (state_dim x state_dim x T array)
- `x_predicted`: predicted state means
- `P_predicted`: predicted state covariances
- `innovations`: innovation sequence (obs_dim x T)
- `S_matrices`: innovation covariance matrices
- `log_likelihood`: total Kalman log-likelihood
- `F`, `H`, `Q`, `R`: stored system matrices
"""
@kwdef struct KalmanState
    x_filtered::Matrix{Float64}
    P_filtered::Array{Float64, 3}
    x_predicted::Matrix{Float64}
    P_predicted::Array{Float64, 3}
    innovations::Matrix{Float64}
    S_matrices::Array{Float64, 3}
    log_likelihood::Float64
    F::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}
end

"""
    SmoothedState

Result of Rauch-Tung-Striebel (RTS) Kalman smoother.

Fields:
- `x_smoothed`: smoothed state means (state_dim x T)
- `P_smoothed`: smoothed state covariances (state_dim x state_dim x T)
"""
@kwdef struct SmoothedState
    x_smoothed::Matrix{Float64}
    P_smoothed::Array{Float64, 3}
end

"""
    kalman_filter(y, F, H, Q, R, x0, P0) -> KalmanState

Run the Kalman filter on an observation sequence.

State space model:
    x_t = F * x_{t-1} + w_t,   w_t ~ N(0, Q)
    y_t = H * x_t + v_t,        v_t ~ N(0, R)

# Arguments
- `y`: obs_dim x T observation matrix
- `F`: state_dim x state_dim transition matrix
- `H`: obs_dim x state_dim observation matrix
- `Q`: state_dim x state_dim process noise covariance
- `R`: obs_dim x obs_dim observation noise covariance
- `x0`: state_dim initial state mean
- `P0`: state_dim x state_dim initial state covariance

# Returns
KalmanState with filtered means, covariances, and log-likelihood.
"""
function kalman_filter(y::Matrix{Float64},
                        F::Matrix{Float64},
                        H::Matrix{Float64},
                        Q::Matrix{Float64},
                        R::Matrix{Float64},
                        x0::Vector{Float64},
                        P0::Matrix{Float64})::KalmanState
    obs_dim, T = size(y)
    state_dim = length(x0)

    x_pred = Matrix{Float64}(undef, state_dim, T)
    P_pred = Array{Float64, 3}(undef, state_dim, state_dim, T)
    x_filt = Matrix{Float64}(undef, state_dim, T)
    P_filt = Array{Float64, 3}(undef, state_dim, state_dim, T)
    innov = Matrix{Float64}(undef, obs_dim, T)
    S_mats = Array{Float64, 3}(undef, obs_dim, obs_dim, T)

    log_lik = 0.0
    x_prev = x0
    P_prev = P0

    for t in 1:T
        # Predict
        x_p = F * x_prev
        P_p = F * P_prev * F' + Q
        P_p = (P_p + P_p') ./ 2  -- symmetrize

        x_pred[:, t] = x_p
        P_pred[:, :, t] = P_p

        # Innovation
        y_t = y[:, t]
        v_t = y_t .- H * x_p
        S_t = H * P_p * H' + R
        S_t = (S_t + S_t') ./ 2

        innov[:, t] = v_t
        S_mats[:, :, t] = S_t

        # Kalman gain
        S_t_reg = S_t + 1e-10 * I(obs_dim)
        K_t = P_p * H' * inv(S_t_reg)

        # Update
        x_u = x_p + K_t * v_t
        P_u = (I(state_dim) - K_t * H) * P_p
        P_u = (P_u + P_u') ./ 2

        x_filt[:, t] = x_u
        P_filt[:, :, t] = P_u

        # Log-likelihood contribution
        sign_det, log_det = logabsdet(S_t_reg)
        if sign_det > 0
            log_lik += -0.5 * (obs_dim * log(2*pi) + log_det + dot(v_t, S_t_reg \ v_t))
        end

        x_prev = x_u
        P_prev = P_u
    end

    return KalmanState(
        x_filtered=x_filt,
        P_filtered=P_filt,
        x_predicted=x_pred,
        P_predicted=P_pred,
        innovations=innov,
        S_matrices=S_mats,
        log_likelihood=log_lik,
        F=F, H=H, Q=Q, R=R
    )
end

"""
    kalman_smoother(filtered) -> SmoothedState

Rauch-Tung-Striebel (RTS) backward smoother.
Improves state estimates using all available data.

# Arguments
- `filtered`: KalmanState from kalman_filter

# Returns
SmoothedState with smoothed means and covariances.
"""
function kalman_smoother(filtered::KalmanState)::SmoothedState
    state_dim, T = size(filtered.x_filtered)
    F = filtered.F

    x_s = copy(filtered.x_filtered)
    P_s = copy(filtered.P_filtered)

    for t in (T-1):-1:1
        x_f_t = filtered.x_filtered[:, t]
        P_f_t = filtered.P_filtered[:, :, t]
        P_p_tp1 = filtered.P_predicted[:, :, t+1]

        P_p_reg = P_p_tp1 + 1e-10 * I(state_dim)
        G_t = P_f_t * F' * inv(P_p_reg)

        x_s[:, t] = x_f_t + G_t * (x_s[:, t+1] - filtered.x_predicted[:, t+1])
        P_s[:, :, t] = P_f_t + G_t * (P_s[:, :, t+1] - P_p_tp1) * G_t'
        P_s[:, :, t] = (P_s[:, :, t] + P_s[:, :, t]') ./ 2
    end

    return SmoothedState(x_smoothed=x_s, P_smoothed=P_s)
end

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

function run_tests()
    @testset "TimeSeriesModels Tests" begin

        rng = MersenneTwister(123)

        # -- ARIMA --
        @testset "fit_arima AR(1)" begin
            phi_true = 0.7
            n = 200
            x = zeros(n)
            for t in 2:n
                x[t] = phi_true * x[t-1] + 0.1 * randn(rng)
            end
            model = fit_arima(x, 1, 0, 0)
            @test model.p == 1
            @test model.d == 0
            @test model.q == 0
            @test length(model.phi) == 1
            @test abs(model.phi[1] - phi_true) < 0.15
            @test model.sigma2 > 0.0
        end

        @testset "fit_arima ARMA(1,1)" begin
            n = 300
            x = zeros(n)
            eps = 0.1 .* randn(rng, n)
            for t in 2:n
                x[t] = 0.6 * x[t-1] + eps[t] + 0.3 * eps[t-1]
            end
            model = fit_arima(x, 1, 0, 1)
            @test model.p == 1
            @test model.q == 1
            @test model.sigma2 > 0.0
            @test isfinite(model.log_likelihood)
        end

        @testset "fit_arima ARIMA(1,1,0)" begin
            n = 200
            eps = 0.1 .* randn(rng, n)
            dx = zeros(n)
            for t in 2:n
                dx[t] = 0.5 * dx[t-1] + eps[t]
            end
            x = cumsum(dx)
            model = fit_arima(x, 1, 1, 0)
            @test model.d == 1
            @test length(model.diff_series) == n - 1
        end

        @testset "forecast AR(1)" begin
            phi_true = 0.5
            n = 150
            x = zeros(n)
            for t in 2:n
                x[t] = phi_true * x[t-1] + 0.05 * randn(rng)
            end
            model = fit_arima(x, 1, 0, 0)
            fc = forecast(model, 5)
            @test length(fc) == 5
            @test all(isfinite.(fc))
        end

        @testset "aic decreases with better model" begin
            n = 300
            x = zeros(n)
            eps = 0.1 .* randn(rng, n)
            for t in 2:n
                x[t] = 0.7 * x[t-1] + eps[t]
            end
            m1 = fit_arima(x, 1, 0, 0)
            m0 = fit_arima(x, 0, 0, 0)
            @test aic(m1) < aic(m0)  -- AR(1) should fit better than AR(0) for AR(1) data
        end

        @testset "aic formula" begin
            n = 100
            x = randn(rng, n)
            model = fit_arima(x, 1, 0, 1)
            expected_k = model.p + model.q + 1
            @test isapprox(aic(model), -2 * model.log_likelihood + 2 * expected_k, atol=1e-8)
        end

        @testset "_difference and _undifference" begin
            x = cumsum(randn(rng, 50))
            dx = _difference(x, 1)
            @test length(dx) == 49
            @test isapprox(dx, diff(x), atol=1e-10)
        end

        # -- Engle-Granger --
        @testset "engle_granger cointegrated" begin
            n = 300
            z = cumsum(randn(rng, n))  -- common stochastic trend
            y1 = z .+ 0.1 .* randn(rng, n)
            y2 = 2.0 .* z .+ 0.1 .* randn(rng, n)
            result = engle_granger(y1, y2)
            @test result.is_cointegrated
            @test abs(result.beta - 0.5) < 0.1  -- y1 = 0.5 * y2 + noise
            @test length(result.residuals) == n
            @test length(result.cointegrating_vector) == 2
        end

        @testset "engle_granger non_cointegrated" begin
            n = 300
            y1 = cumsum(randn(rng, n))
            y2 = cumsum(randn(rng, n))  -- independent random walks
            result = engle_granger(y1, y2)
            @test isa(result, CointegrationResult)
            @test isfinite(result.adf_stat)
        end

        @testset "engle_granger residuals" begin
            n = 200
            z = cumsum(randn(rng, n))
            y1 = z .+ randn(rng, n) .* 0.05
            y2 = z
            result = engle_granger(y1, y2)
            # Residuals should be approximately stationary (near-zero mean)
            @test abs(mean(result.residuals)) < 0.5
        end

        # -- Johansen --
        @testset "johansen_test basic" begin
            n = 200
            z = cumsum(randn(rng, n))
            y1 = z .+ randn(rng, n) .* 0.1
            y2 = 2.0 .* z .+ randn(rng, n) .* 0.1
            y3 = -z .+ randn(rng, n) .* 0.1
            Y = hcat(y1, y2, y3)
            result = johansen_test(Y, 1)
            @test length(result.trace_stats) == 3
            @test length(result.crit_vals) == 3
            @test result.n_cointegrating_vectors >= 0
            @test all(isfinite.(result.trace_stats))
        end

        @testset "johansen_test output shapes" begin
            n = 150
            Y = cumsum(randn(rng, n, 4), dims=1)
            result = johansen_test(Y, 1)
            @test length(result.trace_stats) == 4
            @test size(result.beta_matrix, 1) == 4
            @test size(result.alpha_matrix, 1) == 4
        end

        # -- Granger Causality --
        @testset "granger_causality no causation" begin
            n = 300
            x = randn(rng, n)
            y = randn(rng, n)
            result = granger_causality(x, y, 3)
            @test isa(result, GrangerResult)
            @test isfinite(result.f_stat)
            @test 0.0 <= result.p_value <= 1.0
            @test result.lag == 3
        end

        @testset "granger_causality with causation" begin
            n = 400
            x = randn(rng, n)
            y = zeros(n)
            for t in 4:n
                y[t] = 0.8 * x[t-1] + 0.1 * randn(rng)  -- x causes y
            end
            result = granger_causality(x, y, 3)
            @test result.f_stat > 1.0
            @test result.does_x_cause_y  -- should detect causation
        end

        @testset "granger_causality asymmetry" begin
            n = 300
            x = randn(rng, n)
            y = zeros(n)
            for t in 3:n
                y[t] = 0.6 * x[t-1] + 0.1 * randn(rng)
            end
            r_xy = granger_causality(x, y, 2)
            r_yx = granger_causality(y, x, 2)
            -- x causes y but y should not cause x (strongly)
            @test r_xy.f_stat > r_yx.f_stat
        end

        # -- Kalman Filter --
        @testset "kalman_filter random walk" begin
            n = 100
            true_state = cumsum(randn(rng, n)) .* 0.5
            obs = true_state .+ randn(rng, n) .* 2.0

            F = reshape([1.0], 1, 1)
            H = reshape([1.0], 1, 1)
            Q = reshape([0.25], 1, 1)
            R = reshape([4.0], 1, 1)
            x0 = [0.0]
            P0 = reshape([1.0], 1, 1)
            y_mat = reshape(obs, 1, n)

            ks = kalman_filter(y_mat, F, H, Q, R, x0, P0)
            @test size(ks.x_filtered) == (1, n)
            @test size(ks.P_filtered) == (1, 1, n)
            @test isfinite(ks.log_likelihood)
        end

        @testset "kalman_filter innovations" begin
            n = 80
            obs = randn(rng, n) .* 2.0
            F = reshape([1.0], 1, 1)
            H = reshape([1.0], 1, 1)
            Q = reshape([0.1], 1, 1)
            R = reshape([4.0], 1, 1)
            x0 = [0.0]
            P0 = reshape([1.0], 1, 1)
            y_mat = reshape(obs, 1, n)

            ks = kalman_filter(y_mat, F, H, Q, R, x0, P0)
            @test size(ks.innovations) == (1, n)
            @test all(isfinite.(ks.innovations))
        end

        @testset "kalman_smoother output" begin
            n = 60
            obs = randn(rng, n)
            F = reshape([0.9], 1, 1)
            H = reshape([1.0], 1, 1)
            Q = reshape([0.1], 1, 1)
            R = reshape([1.0], 1, 1)
            x0 = [0.0]
            P0 = reshape([1.0], 1, 1)
            y_mat = reshape(obs, 1, n)

            ks = kalman_filter(y_mat, F, H, Q, R, x0, P0)
            ss = kalman_smoother(ks)
            @test size(ss.x_smoothed) == (1, n)
            @test size(ss.P_smoothed) == (1, 1, n)
        end

        @testset "kalman_smoother improves mse" begin
            n = 100
            true_x = zeros(n)
            for t in 2:n
                true_x[t] = 0.9 * true_x[t-1] + 0.1 * randn(rng)
            end
            obs = true_x .+ randn(rng, n) .* 2.0

            F = reshape([0.9], 1, 1)
            H = reshape([1.0], 1, 1)
            Q = reshape([0.01], 1, 1)
            R = reshape([4.0], 1, 1)
            x0 = [0.0]
            P0 = reshape([1.0], 1, 1)
            y_mat = reshape(obs, 1, n)

            ks = kalman_filter(y_mat, F, H, Q, R, x0, P0)
            ss = kalman_smoother(ks)

            mse_filt = mean((vec(ks.x_filtered) .- true_x) .^ 2)
            mse_smooth = mean((vec(ss.x_smoothed) .- true_x) .^ 2)
            @test mse_smooth <= mse_filt + 0.01  -- smoother should be at least as good
        end

        @testset "kalman_filter multivariate" begin
            n = 50
            F2 = [0.9 0.1; 0.0 0.8]
            H2 = [1.0 0.0; 0.0 1.0]
            Q2 = 0.1 .* I(2) |> Matrix
            R2 = 1.0 .* I(2) |> Matrix
            x0_2 = zeros(2)
            P0_2 = Matrix{Float64}(I(2))
            y2 = randn(rng, 2, n)

            ks2 = kalman_filter(y2, F2, H2, Q2, R2, x0_2, P0_2)
            @test size(ks2.x_filtered) == (2, n)
            @test isfinite(ks2.log_likelihood)
        end

        @testset "granger_causality returns struct" begin
            n = 200
            x = randn(rng, n)
            y = randn(rng, n)
            r = granger_causality(x, y, 1)
            @test r.lag == 1
            @test r.f_stat >= 0.0
            @test isa(r.does_x_cause_y, Bool)
        end

        @testset "johansen_crit_vals" begin
            cv = TimeSeriesModels._johansen_crit_vals(5)
            @test length(cv) == 5
            @test all(cv .> 0.0)
            @test issorted(cv)
        end

    end
end

end # module TimeSeriesModels
