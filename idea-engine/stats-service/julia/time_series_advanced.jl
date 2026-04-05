module TimeSeriesAdvanced

# ============================================================
# time_series_advanced.jl -- Advanced Time Series Analysis
# ============================================================
# Covers: SARIMA, TBATS, state-space models, Kalman filter and
# smoother, dynamic factor models, spectral analysis,
# structural breaks, ARFIMA long memory, threshold AR,
# cointegration, Engle-Granger and Johansen tests.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

# ----------------------------------------------------------------
# Structs
# ----------------------------------------------------------------

struct ARIMAModel
    p::Int; d::Int; q::Int
    ar_coef::Vector{Float64}
    ma_coef::Vector{Float64}
    constant::Float64
    sigma2::Float64
end

struct SARIMAModel
    p::Int; d::Int; q::Int
    P::Int; D::Int; Q::Int; s::Int
    ar_coef::Vector{Float64}
    ma_coef::Vector{Float64}
    sar_coef::Vector{Float64}
    sma_coef::Vector{Float64}
    constant::Float64
    sigma2::Float64
end

struct StateSpaceModel
    Z::Matrix{Float64}   # m x k observation matrix
    H::Matrix{Float64}   # m x m noise covariance
    T::Matrix{Float64}   # k x k transition
    R::Matrix{Float64}   # k x r selection
    Q::Matrix{Float64}   # r x r state noise
    a1::Vector{Float64}
    P1::Matrix{Float64}
end

struct KalmanState
    a::Vector{Float64}
    P::Matrix{Float64}
    v::Vector{Float64}
    F::Matrix{Float64}
    K::Matrix{Float64}
    ll::Float64
end

struct DynamicFactorModel
    n_series::Int
    n_factors::Int
    factor_loadings::Matrix{Float64}
    factor_ar::Vector{Float64}
    idiosyncratic_var::Vector{Float64}
end

struct TBATSModel
    alpha::Float64
    beta_smooth::Float64
    gamma::Vector{Float64}
    periods::Vector{Int}
    phi::Float64
    box_cox_lambda::Float64
end

struct ARFIMAModel
    p::Int; q::Int
    d_frac::Float64
    ar_coef::Vector{Float64}
    ma_coef::Vector{Float64}
    sigma2::Float64
end

# ----------------------------------------------------------------
# 1. Differencing and Transformations
# ----------------------------------------------------------------

function difference(x::Vector{Float64}, d::Int=1)::Vector{Float64}
    y = copy(x)
    for _ in 1:d
        y = diff(y)
    end
    return y
end

function seasonal_difference(x::Vector{Float64}, s::Int, D::Int=1)::Vector{Float64}
    y = copy(x)
    for _ in 1:D
        y = [y[i] - y[i-s] for i in (s+1):length(y)]
    end
    return y
end

function box_cox_transform(x::Vector{Float64}, lambda::Float64)::Vector{Float64}
    if abs(lambda) < 1e-10
        return log.(x)
    else
        return (x .^ lambda .- 1.0) ./ lambda
    end
end

function box_cox_inverse(y::Vector{Float64}, lambda::Float64)::Vector{Float64}
    if abs(lambda) < 1e-10
        return exp.(y)
    else
        return (y .* lambda .+ 1.0) .^ (1.0 / lambda)
    end
end

function guerrero_lambda(x::Vector{Float64}, s::Int)::Float64
    n = length(x)
    m = n div s
    if m < 2; return 1.0; end
    best_lambda = 1.0; best_cv = Inf
    for lam in -1.0:0.1:2.0
        yt = box_cox_transform(abs.(x) .+ 1e-8, lam)
        groups = [yt[(j-1)*s+1 : j*s] for j in 1:m]
        stds_g = std.(groups); means_g = mean.(groups)
        cv = std(stds_g ./ (abs.(means_g) .+ 1e-12))
        if cv < best_cv; best_cv = cv; best_lambda = lam; end
    end
    return best_lambda
end

# ----------------------------------------------------------------
# 2. Autocorrelation
# ----------------------------------------------------------------

function acf_vals(x::Vector{Float64}, max_lag::Int)::Vector{Float64}
    n = length(x)
    xm = x .- mean(x)
    gamma0 = dot(xm, xm) / n
    return [dot(xm[1:n-k], xm[k+1:n]) / (n * gamma0) for k in 0:max_lag]
end

function pacf_vals(x::Vector{Float64}, max_lag::Int)::Vector{Float64}
    r = acf_vals(x, max_lag)
    phi = zeros(max_lag, max_lag)
    result = zeros(max_lag)
    phi[1,1] = r[2]; result[1] = r[2]
    for k in 2:max_lag
        num = r[k+1] - sum(phi[k-1, j] * r[k-j+1] for j in 1:(k-1); init=0.0)
        den = 1.0  - sum(phi[k-1, j] * r[j+1]     for j in 1:(k-1); init=0.0)
        phi[k,k] = num / (den + 1e-18)
        for j in 1:(k-1)
            phi[k,j] = phi[k-1,j] - phi[k,k] * phi[k-1, k-j]
        end
        result[k] = phi[k,k]
    end
    return result
end

function ljung_box_test(x::Vector{Float64}, lags::Int)
    n = length(x)
    r = acf_vals(x, lags)
    q = n * (n + 2) * sum(r[k+1]^2 / (n - k) for k in 1:lags)
    chi2_95 = lags + 2.0 * sqrt(2.0 * lags)
    return (q_stat=q, lags=lags, reject_white_noise=q > chi2_95)
end

# ----------------------------------------------------------------
# 3. LCG Normal Sampler (no Distributions.jl)
# ----------------------------------------------------------------

function lcg_randn(state::UInt64)
    state = 6364136223846793005 * state + 1442695040888963407
    u1 = max(Float64(state >> 33) / Float64(2^31), 1e-15)
    state = 6364136223846793005 * state + 1442695040888963407
    u2 = max(Float64(state >> 33) / Float64(2^31), 1e-15)
    return sqrt(-2.0 * log(u1)) * cos(2 * pi * u2), state
end

# ----------------------------------------------------------------
# 4. ARIMA
# ----------------------------------------------------------------

function arima_simulate(model::ARIMAModel, n::Int; seed::Int=42)::Vector{Float64}
    p = model.p; q = model.q; sig = sqrt(model.sigma2)
    total = n + max(p, q) * 10
    eps = zeros(total)
    rng = UInt64(seed)
    for i in 1:total
        v, rng = lcg_randn(rng)
        eps[i] = v * sig
    end
    y = zeros(total)
    for t in (max(p,q)+1):total
        ar_part = sum(model.ar_coef[i] * y[t-i] for i in 1:p; init=0.0)
        ma_part = sum(model.ma_coef[i] * eps[t-i] for i in 1:q; init=0.0)
        y[t] = model.constant + ar_part + ma_part + eps[t]
    end
    return y[end-n+1:end]
end

function arima_forecast(model::ARIMAModel, history::Vector{Float64}, h::Int)::Vector{Float64}
    p = model.p; q = model.q
    extended = copy(history)
    eps_ext = zeros(length(history) + h)
    forecasts = Float64[]
    for _ in 1:h
        t = length(extended)
        ar_part = sum(model.ar_coef[i] * extended[max(1,t-i+1)] for i in 1:p; init=0.0)
        ma_part = sum(model.ma_coef[i] * eps_ext[max(1,t-i+1)] for i in 1:q; init=0.0)
        yhat = model.constant + ar_part + ma_part
        push!(forecasts, yhat); push!(extended, yhat)
    end
    return forecasts
end

function arima_aic(model::ARIMAModel, y::Vector{Float64})::Float64
    n = length(y); k = model.p + model.q + 2
    ll = -0.5 * n * log(2 * pi * model.sigma2) - 0.5 * n
    return -2 * ll + 2 * k
end

function arima_bic(model::ARIMAModel, y::Vector{Float64})::Float64
    n = length(y); k = model.p + model.q + 2
    ll = -0.5 * n * log(2 * pi * model.sigma2) - 0.5 * n
    return -2 * ll + k * log(n)
end

# ----------------------------------------------------------------
# 5. SARIMA
# ----------------------------------------------------------------

function sarima_forecast(model::SARIMAModel, history::Vector{Float64}, h::Int)::Vector{Float64}
    s = model.s
    y = copy(history)
    forecasts = Float64[]
    for _ in 1:h
        t = length(y)
        yhat = model.constant
        for i in 1:model.p
            t_i = t - i + 1
            yhat += (t_i >= 1) ? model.ar_coef[i] * y[t_i] : 0.0
        end
        for i in 1:model.P
            t_i = t - i * s + 1
            yhat += (t_i >= 1) ? model.sar_coef[i] * y[t_i] : 0.0
        end
        push!(forecasts, yhat); push!(y, yhat)
    end
    return forecasts
end

function seasonal_decompose(y::Vector{Float64}, s::Int)
    n = length(y); half = s div 2
    trend = fill(NaN, n)
    for i in (half+1):(n-half)
        trend[i] = mean(y[i-half:i+half])
    end
    detrended = y .- trend
    seasonal = zeros(n)
    for k in 0:(s-1)
        indices = [k+1+j*s for j in 0:(n div s) if k+1+j*s <= n]
        valid = filter(i -> !isnan(detrended[i]), indices)
        if !isempty(valid)
            avg = mean(detrended[valid])
            for i in indices; seasonal[i] = avg; end
        end
    end
    residual = y .- trend .- seasonal
    return (trend=trend, seasonal=seasonal, residual=residual)
end

# ----------------------------------------------------------------
# 6. Kalman Filter and Smoother
# ----------------------------------------------------------------

function kalman_filter(model::StateSpaceModel, observations::Matrix{Float64})::Vector{KalmanState}
    m, T_len = size(observations)
    k_dim = length(model.a1)
    states = KalmanState[]
    a = copy(model.a1); P = copy(model.P1)
    for t in 1:T_len
        y_t = observations[:, t]
        v = y_t - model.Z * a
        F = model.Z * P * model.Z' + model.H
        F_inv = inv(F + 1e-10 * I(m))
        K = P * model.Z' * F_inv
        a_upd = a + K * v
        P_upd = (I(k_dim) - K * model.Z) * P
        _, log_det_F = logabsdet(F + 1e-10 * I(m))
        ll = -0.5 * (m * log(2pi) + log_det_F + dot(v, F_inv * v))
        push!(states, KalmanState(a_upd, (P_upd + P_upd') ./ 2, v, F, K, ll))
        a = model.T * a_upd
        P = model.T * P_upd * model.T' + model.R * model.Q * model.R'
    end
    return states
end

function kalman_smoother(model::StateSpaceModel, filtered::Vector{KalmanState})::Vector{Vector{Float64}}
    T_len = length(filtered)
    k_dim = length(filtered[1].a)
    smoothed_a = [copy(filtered[t].a) for t in 1:T_len]
    smoothed_P = [copy(filtered[t].P) for t in 1:T_len]
    for t in (T_len-1):-1:1
        P_pred = model.T * filtered[t].P * model.T' + model.R * model.Q * model.R'
        G = filtered[t].P * model.T' * inv(P_pred + 1e-10 * I(k_dim))
        smoothed_a[t] = filtered[t].a + G * (smoothed_a[t+1] - model.T * filtered[t].a)
        smoothed_P[t] = filtered[t].P + G * (smoothed_P[t+1] - P_pred) * G'
    end
    return smoothed_a
end

function kalman_log_likelihood(model::StateSpaceModel, observations::Matrix{Float64})::Float64
    states = kalman_filter(model, observations)
    return sum(s.ll for s in states)
end

# ----------------------------------------------------------------
# 7. Dynamic Factor Model
# ----------------------------------------------------------------

function dfm_extract_factors(data::Matrix{Float64}, n_factors::Int)::Matrix{Float64}
    mu = mean(data, dims=2)
    sig = std(data, dims=2) .+ 1e-10
    std_data = (data .- mu) ./ sig
    U, S, V = svd(std_data)
    return Diagonal(S[1:n_factors]) * V[:, 1:n_factors]'
end

function dfm_loadings(data::Matrix{Float64}, factors::Matrix{Float64})::Matrix{Float64}
    n = size(data, 1)
    Lambda = zeros(n, size(factors, 1))
    for i in 1:n
        X = factors'
        Lambda[i, :] = (X'X + 1e-10*I(size(X,2))) \ (X' * data[i, :])
    end
    return Lambda
end

function dfm_forecast(model::DynamicFactorModel, factors::Matrix{Float64}, h::Int)::Matrix{Float64}
    T_len = size(factors, 2)
    projected = copy(factors)
    for _ in 1:h
        next_f = model.factor_ar .* projected[:, end]
        projected = hcat(projected, next_f)
    end
    return model.factor_loadings * projected[:, (T_len+1):end]
end

# ----------------------------------------------------------------
# 8. ARFIMA Long Memory
# ----------------------------------------------------------------

function arfima_weights(d::Float64, max_lag::Int)::Vector{Float64}
    w = ones(max_lag + 1)
    for k in 1:max_lag
        w[k+1] = -w[k] * (d - k + 1) / k
    end
    return w
end

function arfima_difference(x::Vector{Float64}, d::Float64)::Vector{Float64}
    n = length(x); trunc = min(n - 1, 200)
    w = arfima_weights(d, trunc)
    result = zeros(n)
    for t in 1:n
        for k in 0:min(t-1, trunc)
            result[t] += w[k+1] * x[t-k]
        end
    end
    return result
end

function hurst_exponent(x::Vector{Float64})::Float64
    n = length(x)
    scales = Int[]; rs_vals = Float64[]
    for s in 8:4:(n div 2)
        n_blocks = n div s
        rs_block = Float64[]
        for b in 1:n_blocks
            block = x[(b-1)*s+1 : b*s]
            deviations = cumsum(block .- mean(block))
            R = maximum(deviations) - minimum(deviations)
            push!(rs_block, R / (std(block) + 1e-12))
        end
        push!(scales, s); push!(rs_vals, mean(rs_block))
    end
    if length(scales) < 2; return 0.5; end
    log_s = log.(Float64.(scales)); log_rs = log.(rs_vals)
    x_bar = mean(log_s); y_bar = mean(log_rs)
    return sum((log_s .- x_bar) .* (log_rs .- y_bar)) / (sum((log_s .- x_bar).^2) + 1e-12)
end

# ----------------------------------------------------------------
# 9. Structural Breaks
# ----------------------------------------------------------------

function chow_test(y::Vector{Float64}, x_mat::Matrix{Float64}, break_point::Int)
    n, kk = size(x_mat)
    b_pool = (x_mat' * x_mat + 1e-10*I(kk)) \ (x_mat' * y)
    rss_pool = sum((y - x_mat * b_pool).^2)
    y1 = y[1:break_point]; x1 = x_mat[1:break_point, :]
    b1 = (x1' * x1 + 1e-10*I(kk)) \ (x1' * y1)
    rss1 = sum((y1 - x1 * b1).^2)
    y2 = y[break_point+1:end]; x2 = x_mat[break_point+1:end, :]
    b2 = (x2' * x2 + 1e-10*I(kk)) \ (x2' * y2)
    rss2 = sum((y2 - x2 * b2).^2)
    rss_unr = rss1 + rss2
    f = ((rss_pool - rss_unr) / kk) / ((rss_unr / (n - 2kk)) + 1e-12)
    return (f_stat=f, break_point=break_point, rss_restricted=rss_pool, rss_unrestricted=rss_unr)
end

function bai_perron_breaks(y::Vector{Float64}, max_breaks::Int=3, min_segment::Int=10)::Vector{Int}
    n = length(y); x_mat = ones(n, 1)
    break_points = Int[]
    for _ in 1:max_breaks
        best_f = -Inf; best_bp = -1
        for bp in min_segment:(n-min_segment)
            res = chow_test(y, x_mat, bp)
            if res.f_stat > best_f; best_f = res.f_stat; best_bp = bp; end
        end
        if best_f > 4.0 && best_bp > 0; push!(break_points, best_bp); else; break; end
    end
    return sort(break_points)
end

# ----------------------------------------------------------------
# 10. Threshold AR
# ----------------------------------------------------------------

function tar_estimate(y::Vector{Float64}, delay::Int=1, n_thresholds::Int=50)
    n = length(y)
    sorted_y = sort(y)
    q15 = sorted_y[max(1, round(Int, 0.15*n))]
    q85 = sorted_y[min(n, round(Int, 0.85*n))]
    thresholds = range(q15, q85, length=n_thresholds)
    best_ssr = Inf; best_thresh = NaN
    best_b1 = zeros(2); best_b2 = zeros(2)
    for tau in thresholds
        idx1 = findall(t -> t >= delay+1 && y[t-delay] <= tau, 1:n)
        idx2 = findall(t -> t >= delay+1 && y[t-delay] > tau,  1:n)
        if length(idx1) < 5 || length(idx2) < 5; continue; end
        function ols_ssr(idx)
            X = hcat(ones(length(idx)), y[idx .- 1])
            b = (X'X + 1e-10*I(2)) \ (X' * y[idx])
            return sum((y[idx] - X*b).^2), b
        end
        s1, b1 = ols_ssr(idx1); s2, b2 = ols_ssr(idx2)
        if s1 + s2 < best_ssr
            best_ssr = s1+s2; best_thresh = tau; best_b1 = b1; best_b2 = b2
        end
    end
    return (threshold=best_thresh, regime1_coef=best_b1, regime2_coef=best_b2, ssr=best_ssr)
end

# ----------------------------------------------------------------
# 11. Cointegration
# ----------------------------------------------------------------

function engle_granger_test(y1::Vector{Float64}, y2::Vector{Float64})
    n = length(y1)
    X = hcat(ones(n), y2)
    b = (X'X + 1e-10*I(2)) \ (X' * y1)
    residuals = y1 - X * b
    dr = diff(residuals); r_lag = residuals[1:end-1]
    X_adf = hcat(ones(n-1), r_lag)
    b_adf = (X_adf'X_adf + 1e-10*I(2)) \ (X_adf' * dr)
    rho = b_adf[2]
    resid_adf = dr - X_adf * b_adf
    se_rho = sqrt(sum(resid_adf.^2) / ((n-3) * sum((r_lag .- mean(r_lag)).^2) + 1e-12))
    t_stat = rho / (se_rho + 1e-12)
    return (beta=b[2], alpha=b[1], adf_t_stat=t_stat, rho=rho,
            cointegrated_5pct=t_stat < -3.34, residuals=residuals)
end

function johansen_trace_statistic(data::Matrix{Float64}, r::Int)::Float64
    T_len, n_vars = size(data)
    diff_data = diff(data, dims=1); lag_data = data[1:end-1, :]
    X = lag_data
    Mxx = X'X / T_len
    M_inv = inv(Mxx + 1e-10*I(n_vars))
    S11 = diff_data' * diff_data / T_len
    S01 = diff_data' * lag_data  / T_len
    A = inv(S11 + 1e-10*I(n_vars)) * S01 * M_inv * S01'
    eigenvalues = sort(real.(eigvals(A)), rev=true)
    trace_stat = -T_len * sum(log(1.0 - clamp(eigenvalues[i], 0.0, 0.9999))
                              for i in (r+1):min(n_vars, length(eigenvalues)))
    return trace_stat
end

# ----------------------------------------------------------------
# 12. Spectral Analysis
# ----------------------------------------------------------------

function periodogram(x::Vector{Float64})
    n = length(x); xm = x .- mean(x)
    freqs = Float64[]; power = Float64[]
    for k in 0:(n div 2)
        re = sum(xm[t] * cos(2pi * k * (t-1) / n) for t in 1:n)
        im = sum(xm[t] * sin(2pi * k * (t-1) / n) for t in 1:n)
        push!(freqs, k / n); push!(power, (re^2 + im^2) / n)
    end
    return freqs, power
end

function dominant_period(x::Vector{Float64})::Float64
    freqs, power = periodogram(x)
    best_k = argmax(power[2:end]) + 1
    f = freqs[best_k]
    return f > 0.0 ? 1.0 / f : Inf
end

function welch_spectrum(x::Vector{Float64}, seg_len::Int, overlap::Int)
    n = length(x); step = seg_len - overlap
    n_half = seg_len div 2 + 1
    sum_pwr = zeros(n_half); n_segs = 0; pos = 1
    while pos + seg_len - 1 <= n
        seg = x[pos:pos+seg_len-1]
        w = [0.5 * (1 - cos(2pi * (i-1) / (seg_len-1))) for i in 1:seg_len]
        _, pwr = periodogram(seg .* w)
        sum_pwr .+= pwr[1:n_half]; n_segs += 1; pos += step
    end
    psd = n_segs > 0 ? sum_pwr ./ n_segs : sum_pwr
    return collect(0:(seg_len div 2)) ./ seg_len, psd
end

# ----------------------------------------------------------------
# 13. TBATS
# ----------------------------------------------------------------

function tbats_smooth(model::TBATSModel, y::Vector{Float64}, h::Int)::Vector{Float64}
    n = length(y)
    s_len = isempty(model.periods) ? 1 : model.periods[1]
    level = y[1]
    trend_val = n > 1 ? y[2] - y[1] : 0.0
    seasonals = zeros(s_len)
    for k in 1:min(s_len, n); seasonals[k] = y[k] - level; end
    for t in 2:n
        s_idx = mod1(t, s_len)
        new_level = model.alpha * (y[t] - seasonals[s_idx]) +
                    (1 - model.alpha) * (level + model.phi * trend_val)
        trend_val = model.beta_smooth * (new_level - level) +
                    (1 - model.beta_smooth) * model.phi * trend_val
        g = isempty(model.gamma) ? 0.1 : model.gamma[1]
        seasonals[s_idx] = g * (y[t] - new_level) + (1 - g) * seasonals[s_idx]
        level = new_level
    end
    forecasts = Float64[]
    for i in 1:h
        s_idx = mod1(n + i, s_len)
        damp = abs(model.phi) < 1e-10 ? Float64(i) :
               (1.0 - model.phi^i) / (1.0 - model.phi + 1e-12)
        push!(forecasts, level + damp * trend_val + seasonals[s_idx])
    end
    return forecasts
end

# ----------------------------------------------------------------
# Demo
# ----------------------------------------------------------------

function demo()
    println("=== TimeSeriesAdvanced Demo ===\n")
    n = 200
    rng = UInt64(99)
    noise = zeros(n)
    for i in 1:n; v, rng = lcg_randn(rng); noise[i] = v; end
    y = cumsum(0.1 .* noise) .+ 0.5 .* sin.(2pi .* (1:n) ./ 12)

    println("Hurst exponent: ", round(hurst_exponent(y), digits=4))
    println("Dominant period: ", round(dominant_period(y), digits=1))
    lb = ljung_box_test(diff(y), 10)
    println("Ljung-Box Q(10): ", round(lb.q_stat, digits=3), " | reject? ", lb.reject_white_noise)

    decomp = seasonal_decompose(y, 12)
    valid_s = filter(!isnan, decomp.seasonal)
    println("Seasonal range: [", round(minimum(valid_s), digits=3), ", ",
            round(maximum(valid_s), digits=3), "]")

    ar_model = ARIMAModel(2, 1, 1, [0.5, -0.2], [0.3], 0.0, 1.0)
    sim = arima_simulate(ar_model, 100)
    println("\nARIMA(2,1,1) sim mean: ", round(mean(sim), digits=4))
    println("5-step forecast: ", round.(arima_forecast(ar_model, sim, 5), digits=3))
    println("AIC: ", round(arima_aic(ar_model, sim), digits=2))

    y1 = cumsum(noise[1:100])
    y2 = y1 .+ 0.1 .* noise[1:100]
    eg = engle_granger_test(y1, y2)
    println("\nEngle-Granger t-stat: ", round(eg.adf_t_stat, digits=3))
    println("Cointegrated (5%): ", eg.cointegrated_5pct)

    step_series = [fill(0.0, 50); fill(1.0, 50); fill(-0.5, 50)]
    bp = bai_perron_breaks(step_series, 3)
    println("\nStructural breaks detected at: ", bp)

    tbats_m = TBATSModel(0.2, 0.05, [0.1], [12], 0.95, 1.0)
    fc = tbats_smooth(tbats_m, y, 12)
    println("TBATS 12-step forecast (first 3): ", round.(fc[1:3], digits=3))
end


# ================================================================
# ADDITIONAL ADVANCED TIME SERIES METHODS
# ================================================================

# ----------------------------------------------------------------
# Multivariate GARCH: DCC-GARCH
# ----------------------------------------------------------------

struct DCCModel
    n::Int
    omega::Vector{Float64}
    alpha::Vector{Float64}
    beta::Vector{Float64}
    dcc_a::Float64
    dcc_b::Float64
    Q_bar::Matrix{Float64}
end

"""Fit GARCH(1,1) for a single series via moment matching."""
function fit_garch11_mom(r::Vector{Float64})
    T = length(r); mu = mean(r); rc = r .- mu; v = var(rc)
    # Grid search
    best = (1e-10, 0.1, 0.85); best_ll = -Inf
    for a in [0.05,0.10,0.15,0.20]
        for b in [0.80,0.85,0.88,0.90,0.93]
            a + b >= 1.0 && continue
            om = v * (1 - a - b); om <= 0 && continue
            s2 = zeros(T); s2[1] = v; ll = 0.0; ok = true
            for t in 2:T
                s2[t] = om + a*rc[t-1]^2 + b*s2[t-1]
                s2[t] <= 0 && (ok = false; break)
                ll += -0.5*(log(2π*s2[t]) + rc[t]^2/s2[t])
            end
            ok && ll > best_ll && ((best_ll, best) = (ll, (om, a, b)))
        end
    end
    return best
end

"""Filter GARCH(1,1) conditional variances."""
function filter_garch11(r::Vector{Float64}, om::Float64, a::Float64, b::Float64)
    T = length(r); mu = mean(r); rc = r .- mu
    s2 = zeros(T); s2[1] = var(rc)
    for t in 2:T
        s2[t] = max(om + a*rc[t-1]^2 + b*s2[t-1], 1e-12)
    end
    return s2
end

"""
DCC-GARCH: Dynamic Conditional Correlations.
Two-step estimation: (1) univariate GARCH, (2) DCC parameters.
"""
function fit_dcc(R::Matrix{Float64}; a::Float64=0.05, b::Float64=0.93)
    T, N = size(R)
    om_v = zeros(N); al_v = zeros(N); bt_v = zeros(N); s2_mat = zeros(T, N)
    for i in 1:N
        om_v[i], al_v[i], bt_v[i] = fit_garch11_mom(R[:, i])
        s2_mat[:, i] = filter_garch11(R[:, i], om_v[i], al_v[i], bt_v[i])
    end
    # Standardized residuals
    eps = (R .- mean(R, dims=1)) ./ sqrt.(s2_mat)
    Q_bar = cov(eps)

    # DCC Q_t updates
    Q_t = copy(Q_bar); R_t = zeros(T, N, N)
    for t in 1:T
        if t > 1
            e = eps[t-1, :]
            Q_t = (1-a-b).*Q_bar + a.*(e*e') + b.*Q_t
        end
        d = Diagonal(1.0 ./ sqrt.(max.(diag(Q_t), 1e-12)))
        Rt = d * Q_t * d
        for i in 1:N, j in 1:N
            R_t[t, i, j] = i==j ? 1.0 : clamp(Rt[i,j], -0.999, 0.999)
        end
    end
    return DCCModel(N, om_v, al_v, bt_v, a, b, Q_bar), R_t, s2_mat
end

"""DCC portfolio variance: w' H_t w."""
function dcc_portfolio_variance(R_t::Array{Float64,3}, s2_mat::Matrix{Float64},
                                  weights::Vector{Float64})
    T = size(R_t, 1); N = length(weights)
    pv = zeros(T)
    for t in 1:T
        D = Diagonal(sqrt.(s2_mat[t, :]))
        H_t = D * R_t[t, :, :] * D
        pv[t] = dot(weights, H_t * weights)
    end
    return pv
end

# ----------------------------------------------------------------
# BEKK-GARCH (Scalar)
# ----------------------------------------------------------------

"""
Scalar BEKK-GARCH: H_t = (1-a-b)*H_bar + a*(ε_{t-1}ε'_{t-1}) + b*H_{t-1}
"""
function scalar_bekk(R::Matrix{Float64}; a::Float64=0.10, b::Float64=0.85)
    T, N = size(R)
    H_bar = cov(R)
    H = zeros(T, N, N); H[1, :, :] = H_bar
    for t in 2:T
        e = R[t-1, :]
        H[t, :, :] = (1-a-b) .* H_bar + a .* (e*e') + b .* H[t-1, :, :]
    end
    return H
end

# ----------------------------------------------------------------
# GO-GARCH (Generalized Orthogonal GARCH)
# ----------------------------------------------------------------

"""
GO-GARCH: diagonalize via PCA then fit univariate GARCH.
H_t = A * Λ_t * A' where A is rotation matrix from PCA.
"""
function go_garch(R::Matrix{Float64})
    T, N = size(R)
    Rc = R .- mean(R, dims=1)
    C = Rc' * Rc / T
    F = eigen(Symmetric(C), sortby=x->-x)
    A = F.vectors  # rotation matrix

    # Orthogonal factors
    factors = Rc * A  # T x N

    # Fit GARCH to each factor
    factor_s2 = zeros(T, N)
    garch_params = []
    for i in 1:N
        om, a, b = fit_garch11_mom(factors[:, i])
        push!(garch_params, (om, a, b))
        factor_s2[:, i] = filter_garch11(factors[:, i], om, a, b)
    end

    # Reconstruct H_t = A * diag(sigma2_t) * A'
    H = zeros(T, N, N)
    for t in 1:T
        Lam = Diagonal(factor_s2[t, :])
        H[t, :, :] = A * Lam * A'
    end

    return H, A, garch_params, factor_s2
end

# ----------------------------------------------------------------
# EXPLOSIVE ROOT DETECTION: PSY TEST
# ----------------------------------------------------------------

"""
ADF test statistic (right-tailed).
H0: unit root. H1: explosive (ρ > 1).
"""
function adf_right_tailed(y::Vector{Float64}, lag::Int=0)
    T = length(y); T < lag+5 && return NaN
    dy = diff(y); yl = y[1:end-1]
    if lag == 0
        n = T-1; X = hcat(ones(n), yl); Y = dy
    else
        n = T-1-lag; n < 3 && return NaN
        X = hcat(ones(n), yl[lag+1:end]); Y = dy[lag+1:end]
        for l in 1:lag; X = hcat(X, dy[lag+1-l:end-l]); end
    end
    beta = (X'*X + 1e-10*I) \ (X'*Y)
    resid = Y - X*beta
    s2 = sum(resid.^2) / max(n - size(X,2), 1)
    Vb = s2 * inv(X'*X + 1e-10*I)
    return beta[2] / sqrt(max(Vb[2,2], 1e-12))
end

"""
BSADF sequence: sup ADF over backward-expanding windows.
Phillips-Shi-Yu (2015) bubble test.
"""
function bsadf_sequence(y::Vector{Float64}; r0::Float64=0.10, lag::Int=0)
    T = length(y)
    min_w = max(round(Int, r0*T), 8)
    bsadf = fill(NaN, T)
    for t in min_w:T
        sup_v = -Inf
        for t1 in 1:t-min_w+1
            v = adf_right_tailed(y[t1:t], lag)
            isnan(v) || (sup_v = max(sup_v, v))
        end
        bsadf[t] = sup_v
    end
    # Approximate 95% critical values (PWY 2015 finite sample)
    cv95 = 1.78 + 3.0 * log(T) / sqrt(T)
    bubble_dates = findall(x -> !isnan(x) && x > cv95, bsadf)
    return (bsadf=bsadf, cv95=cv95, bubble_dates=bubble_dates)
end

# ----------------------------------------------------------------
# FRACTIONAL INTEGRATION: ARFIMA
# ----------------------------------------------------------------

"""
GPH (Geweke-Porter-Hudak) estimator of fractional d.
"""
function gph_d(y::Vector{Float64}; bw::Float64=0.5)
    T = length(y); m = max(3, round(Int, T^bw))
    yd = y .- mean(y)
    freq = [2π*j/T for j in 1:m]
    I_lamb = zeros(m)
    for k in 1:m
        re_v = sum(yd[t]*cos(freq[k]*t) for t in 1:T)
        im_v = sum(yd[t]*sin(freq[k]*t) for t in 1:T)
        I_lamb[k] = (re_v^2 + im_v^2) / (2π*T)
    end
    logI = log.(max.(I_lamb, 1e-10))
    x = [-2*log(abs(2*sin(f/2))) for f in freq]
    X = hcat(ones(m), x)
    beta = (X'*X) \ (X'*logI)
    s2 = sum((logI - X*beta).^2) / max(m-2, 1)
    se = sqrt(s2 * inv(X'*X)[2,2])
    return (d=beta[2], se=se, t_stat=beta[2]/max(se, 1e-10))
end

"""
Fractional differencing operator (1-L)^d y.
"""
function frac_diff(y::Vector{Float64}, d::Float64; max_lag::Int=100, tol::Float64=1e-5)
    n = length(y)
    w = ones(max_lag+1); for k in 1:max_lag; w[k+1] = -w[k]*(d-k+1)/k; abs(w[k+1]) < tol && (max_lag=k; break); end
    w = w[1:max_lag+1]
    out = zeros(n - max_lag)
    for t in max_lag+1:n
        out[t-max_lag] = sum(w[k+1]*y[t-k] for k in 0:max_lag)
    end
    return out
end

# ----------------------------------------------------------------
# FUNCTIONAL TIME SERIES: KARHUNEN-LOEVE
# ----------------------------------------------------------------

"""
Karhunen-Loève expansion of intraday curves.
curves: T x n_grid (T days, each day has n_grid intraday points).
"""
function kl_expansion(curves::Matrix{Float64}; n_comps::Int=5)
    T, n = size(curves)
    mu = vec(mean(curves, dims=1))
    C = (curves .- mu')' * (curves .- mu') / (T-1)
    F = eigen(Symmetric(C), sortby=x->-x)
    ev = max.(F.values[1:n_comps], 0.0)
    ef = F.vectors[:, 1:n_comps]
    scores = (curves .- mu') * ef
    var_exp = cumsum(ev) / max(sum(max.(F.values, 0.0)), 1e-10)
    return (eigenfunctions=ef, eigenvalues=ev, scores=scores, mean_curve=mu, var_explained=var_exp)
end

"""Reconstruct curve from KL scores."""
function kl_reconstruct(scores::Vector{Float64}, kl::NamedTuple)
    return kl.mean_curve + kl.eigenfunctions * scores
end

"""
Forecast functional time series using VAR on KL scores.
"""
function forecast_functional_ts(curves::Matrix{Float64}, h::Int=5; n_comps::Int=3, p::Int=1)
    kl = kl_expansion(curves, n_comps=n_comps)
    scores = kl.scores  # T x n_comps

    # Fit VAR(p) on scores
    T, K = size(scores)
    n_obs = T - p
    Y = scores[p+1:end, :]
    X = hcat([scores[p+1-l:T-l, :] for l in 1:p]..., ones(n_obs))
    B = (X'*X + 1e-8*I) \ (X'*Y)

    # Forecast h steps
    score_hist = copy(scores)
    forecasted_scores = zeros(h, K)
    for step in 1:h
        Tt = size(score_hist, 1)
        x_new = vcat([score_hist[Tt+1-l, :] for l in 1:p]..., [1.0])
        forecasted_scores[step, :] = B' * x_new
        score_hist = vcat(score_hist, forecasted_scores[step:step, :])
    end

    # Reconstruct forecast curves
    forecast_curves = zeros(h, length(kl.mean_curve))
    for step in 1:h
        forecast_curves[step, :] = kl_reconstruct(forecasted_scores[step, :], kl)
    end

    return (forecast_curves=forecast_curves, forecast_scores=forecasted_scores, kl=kl)
end

# ----------------------------------------------------------------
# PERIODIC ARMA
# ----------------------------------------------------------------

"""
Periodic AR: AR coefficients vary with period s.
"""
function fit_periodic_ar(y::Vector{Float64}, s::Int, p::Int=1)
    T = length(y)
    coefs = zeros(s, p+1)  # intercept + p lags per season
    sigma2 = zeros(s)

    for season in 1:s
        t_idx = [t for t in (p+1):T if mod(t-1, s)+1 == season]
        isempty(t_idx) && continue
        Y_s = y[t_idx]
        X_s = hcat(ones(length(t_idx)), [y[t-l] for t in t_idx, l in 1:p])
        n_s = length(t_idx)
        n_s < p+2 && continue
        beta = (X_s'*X_s + 1e-8*I) \ (X_s'*Y_s)
        coefs[season, :] = beta
        resid = Y_s - X_s*beta
        sigma2[season] = sum(resid.^2) / max(n_s-p-1, 1)
    end
    return (coefs=coefs, sigma2=sigma2, s=s, p=p)
end

"""Forecast from periodic AR model."""
function forecast_periodic_ar(model::NamedTuple, y_hist::Vector{Float64}, h::Int)
    T = length(y_hist); s = model.s; p = model.p
    y_ext = copy(y_hist)
    forecasts = Float64[]
    for step in 1:h
        t_new = T + step
        season = mod(t_new-1, s) + 1
        x = vcat(1.0, [y_ext[end+1-l] for l in 1:p])
        yhat = dot(model.coefs[season, :], x)
        push!(forecasts, yhat); push!(y_ext, yhat)
    end
    return forecasts
end

# ----------------------------------------------------------------
# LSTAR (Logistic Smooth Transition AR)
# ----------------------------------------------------------------

struct LSTARModel
    ar_low::Vector{Float64}   # AR coefficients in low regime
    ar_high::Vector{Float64}  # AR coefficients in high regime
    gamma::Float64            # transition speed
    c::Float64               # transition threshold
    delay::Int               # delay of transition variable
    p::Int
end

"""Logistic transition function."""
logistic(x::Float64, gamma::Float64, c::Float64) = 1.0 / (1.0 + exp(-gamma * (x - c)))

"""Fit LSTAR(p, 1) model."""
function fit_lstar(y::Vector{Float64}, p::Int=2, delay::Int=1;
                   n_gamma::Int=10, n_c::Int=20)
    T = length(y); T < 2p+10 && error("Too short")
    Y = y[p+1:T]
    n = length(Y)
    X = hcat([y[p+1-l:T-l] for l in 1:p]...)
    q = y[p+1-delay:T-delay]

    gamma_grid = exp.(range(log(0.1), log(100.0), length=n_gamma))
    c_grid = quantile(q, range(0.1, 0.9, length=n_c))

    best_sse = Inf; best_params = (1.0, median(q))

    for gamma in gamma_grid
        for c in c_grid
            G = [logistic(q[t], gamma, c) for t in 1:n]
            X_full = hcat(X, X .* G, ones(n), G)
            beta = (X_full'*X_full + 1e-8*I) \ (X_full'*Y)
            sse = sum((Y - X_full*beta).^2)
            sse < best_sse && ((best_sse, best_params) = (sse, (gamma, c)))
        end
    end

    gamma_opt, c_opt = best_params
    G = [logistic(q[t], gamma_opt, c_opt) for t in 1:n]
    X_full = hcat(X, X .* G, ones(n), G)
    beta = (X_full'*X_full + 1e-8*I) \ (X_full'*Y)

    ar_low  = vcat(beta[end-1], beta[1:p])
    ar_high = vcat(beta[end-1]+beta[end], beta[1:p] + beta[p+1:2p])

    return LSTARModel(ar_low, ar_high, gamma_opt, c_opt, delay, p)
end

"""Predict from LSTAR model."""
function predict_lstar(model::LSTARModel, y_hist::Vector{Float64}, h::Int)
    y_ext = copy(y_hist); p = model.p
    forecasts = Float64[]
    for step in 1:h
        T_ext = length(y_ext)
        q_val = y_ext[T_ext - model.delay + 1]
        G = logistic(q_val, model.gamma, model.c)
        x_lag = [y_ext[T_ext-l+1] for l in 1:p]
        yhat = (1-G) * (model.ar_low[1] + dot(model.ar_low[2:end], x_lag)) +
                   G  * (model.ar_high[1] + dot(model.ar_high[2:end], x_lag))
        push!(forecasts, yhat); push!(y_ext, yhat)
    end
    return forecasts
end

# ----------------------------------------------------------------
# ROLLING STATISTICS / CHANGE POINT DETECTION
# ----------------------------------------------------------------

"""Rolling mean, std, skewness, kurtosis."""
function rolling_moments(y::Vector{Float64}, w::Int)
    T = length(y)
    mu  = zeros(T); sigma = zeros(T); sk = zeros(T); kurt = zeros(T)
    for t in w:T
        sub = y[t-w+1:t]
        mu[t] = mean(sub); sigma[t] = std(sub)
        if sigma[t] > 1e-10
            z = (sub .- mu[t]) ./ sigma[t]
            sk[t] = mean(z.^3)
            kurt[t] = mean(z.^4) - 3.0
        end
    end
    return (mean=mu, std=sigma, skewness=sk, excess_kurtosis=kurt)
end

"""CUSUM change-point test statistic."""
function cusum_test(y::Vector{Float64})
    T = length(y); mu_all = mean(y); s = std(y)
    s < 1e-10 && return (stat=0.0, break_date=T÷2)
    cusum = zeros(T)
    for t in 1:T
        cusum[t] = sum(y[1:t] .- mu_all) / (s * sqrt(T))
    end
    stat = maximum(abs.(cusum))
    break_date = argmax(abs.(cusum))
    return (stat=stat, break_date=break_date, cusum=cusum)
end

"""Zivot-Andrews unit root test with structural break."""
function zivot_andrews_test(y::Vector{Float64}, lag::Int=1)
    T = length(y); trim = round(Int, 0.15 * T)
    min_t = trim + lag + 2; max_t = T - trim
    best_t = min_t; best_tstat = Inf

    for tb in min_t:max_t
        DU = [t > tb ? 1.0 : 0.0 for t in 1:T-1]
        DT = [t > tb ? Float64(t - tb) : 0.0 for t in 1:T-1]
        dy = diff(y); yl = y[1:end-1]
        X = hcat(ones(T-1), yl, DU, DT)
        if lag > 0
            for l in 1:lag
                if length(dy)-l >= size(X,1)
                    X = hcat(X, dy[1:end])
                end
            end
        end
        n = size(X, 1)
        beta = (X'*X + 1e-10*I) \ (X'*dy)
        resid = dy - X*beta
        s2 = sum(resid.^2) / max(n - size(X,2), 1)
        Vb = s2 * inv(X'*X + 1e-10*I)
        t_rho = beta[2] / sqrt(max(Vb[2,2], 1e-12))
        if t_rho < best_tstat
            best_tstat = t_rho
            best_t = tb
        end
    end

    # Critical values (approximate): -5.34 (1%), -4.80 (5%), -4.58 (10%)
    reject_5pct = best_tstat < -4.80
    return (t_stat=best_tstat, break_date=best_t, reject_unit_root_5pct=reject_5pct)
end

# ----------------------------------------------------------------
# SPECTRAL ANALYSIS
# ----------------------------------------------------------------

"""
DFT (Discrete Fourier Transform) for spectral analysis.
Returns (frequencies, power_spectrum).
"""
function periodogram(y::Vector{Float64})
    T = length(y); yd = y .- mean(y)
    m = T ÷ 2
    freqs = collect(1:m) ./ T
    power = zeros(m)
    for k in 1:m
        freq = 2π * k / T
        re_v = sum(yd[t] * cos(freq * t) for t in 1:T)
        im_v = sum(yd[t] * sin(freq * t) for t in 1:T)
        power[k] = (re_v^2 + im_v^2) / T
    end
    return (freqs=freqs, power=power)
end

"""Smoothed periodogram using Daniell kernel."""
function smoothed_periodogram(y::Vector{Float64}, bandwidth::Int=5)
    prd = periodogram(y)
    m = length(prd.power)
    smooth = zeros(m)
    for k in 1:m
        lo = max(1, k-bandwidth); hi = min(m, k+bandwidth)
        smooth[k] = mean(prd.power[lo:hi])
    end
    return (freqs=prd.freqs, power=smooth)
end

"""Detect dominant frequencies."""
function dominant_frequencies(y::Vector{Float64}; n_peaks::Int=5)
    prd = periodogram(y)
    sorted_idx = sortperm(prd.power, rev=true)
    top_idx = sorted_idx[1:min(n_peaks, length(sorted_idx))]
    return [(freq=prd.freqs[i], period=1/prd.freqs[i], power=prd.power[i]) for i in top_idx]
end

# ----------------------------------------------------------------
# FORECASTING EVALUATION
# ----------------------------------------------------------------

"""Forecast evaluation metrics."""
function forecast_metrics(y_true::Vector{Float64}, y_pred::Vector{Float64})
    n = length(y_true); @assert length(y_pred) == n
    err = y_true - y_pred
    mae = mean(abs.(err))
    mse = mean(err.^2); rmse = sqrt(mse)
    mape = mean(abs.(err) ./ max.(abs.(y_true), 1e-10)) * 100
    # MASE vs naive
    naive_err = y_true[2:end] - y_true[1:end-1]
    mase = mae / max(mean(abs.(naive_err)), 1e-10)
    # Directional accuracy
    dir_acc = mean(sign.(diff(y_true)) .== sign.(diff(y_pred)))
    return (mae=mae, mse=mse, rmse=rmse, mape=mape, mase=mase, directional_acc=dir_acc)
end

"""Diebold-Mariano test: compare two forecasts."""
function diebold_mariano_test(y_true::Vector{Float64},
                               y_pred1::Vector{Float64}, y_pred2::Vector{Float64})
    d = (y_true - y_pred1).^2 - (y_true - y_pred2).^2  # differential loss
    n = length(d)
    d_bar = mean(d); d_std = std(d) / sqrt(n)
    t_stat = d_bar / max(d_std, 1e-10)
    # p-value via normal approximation
    p_val = 2 * (1 - min(abs(t_stat), 8) / 8)  # crude approx
    return (t_stat=t_stat, d_bar=d_bar, reject_equal_accuracy=(abs(t_stat) > 1.96))
end

"""Mincer-Zarnowitz regression: efficiency test."""
function mz_regression(y_true::Vector{Float64}, y_pred::Vector{Float64})
    n = length(y_true)
    X = hcat(ones(n), y_pred)
    beta = (X'*X + 1e-10*I) \ (X'*y_true)
    yhat = X*beta; resid = y_true - yhat
    ss_res = sum(resid.^2); ss_tot = sum((y_true .- mean(y_true)).^2)
    r2 = max(0.0, 1 - ss_res/max(ss_tot, 1e-10))
    # Efficient forecast: alpha=0, beta=1
    unbiased = abs(beta[1]) < 0.01
    efficient = abs(beta[2] - 1.0) < 0.1
    return (alpha=beta[1], beta=beta[2], r2=r2, unbiased=unbiased, efficient=efficient)
end

# ----------------------------------------------------------------
# COPULA-BASED TIME SERIES
# ----------------------------------------------------------------

"""Gaussian copula transform: ranks → normal scores."""
function to_normal_scores(x::Vector{Float64})
    n = length(x)
    ranks = ordinal_ranks(x)
    u = ranks ./ (n + 1)
    return quantile_normal.(u)
end

"""Ordinal ranks."""
function ordinal_ranks(x::Vector{Float64})
    n = length(x); sorted_idx = sortperm(x); ranks = zeros(n)
    for (r, i) in enumerate(sorted_idx); ranks[i] = Float64(r); end
    return ranks
end

"""Normal quantile (Beasley-Springer-Moro approximation)."""
function quantile_normal(p::Float64)
    p = clamp(p, 1e-10, 1-1e-10)
    if p == 0.5; return 0.0; end
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    q = p - 0.5
    if abs(q) < 0.42
        r = q^2
        return q * (((a[4]*r+a[3])*r+a[2])*r+a[1]) / ((((b[4]*r+b[3])*r+b[2])*r+b[1])*r+1)
    end
    r = p < 0.5 ? p : 1-p
    r = log(-log(r))
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    x = c[1]; for i in 2:9; x += c[i]*r^(i-1); end
    return p < 0.5 ? -x : x
end

"""Dynamic copula correlation via DCC on normal scores."""
function dynamic_copula_correlation(x::Matrix{Float64}; a::Float64=0.05, b::Float64=0.93)
    # Transform each column to normal scores
    ns = hcat([to_normal_scores(x[:, j]) for j in 1:size(x,2)]...)
    # DCC on normal scores
    _, R_t, _ = fit_dcc(ns, a=a, b=b)
    return R_t
end

# ----------------------------------------------------------------
# ADDITIONAL DEMO
# ----------------------------------------------------------------

"""Extended demo for additional methods."""
function demo_extended(; seed::Int=42)
    rng = MersenneTwister(seed)
    T = 300
    y = cumsum(randn(rng, T) * 0.02)

    println("=== Extended Time Series Demo ===")

    # PSY bubble test
    psy = bsadf_sequence(y)
    println("PSY BSADF: $(length(psy.bubble_dates)) bubble dates detected")

    # GPH long memory
    g = gph_d(y)
    println("GPH d estimate: $(round(g.d, digits=4))")

    # CUSUM
    cs = cusum_test(diff(y))
    println("CUSUM break date: $(cs.break_date), stat=$(round(cs.stat, digits=3))")

    # DCC
    R2 = hcat(y[1:end-1], y[2:end]) * 0.01
    dcc_m, R_t, _ = fit_dcc(R2)
    println("DCC avg correlation: $(round(mean(R_t[:, 1, 2]), digits=4))")

    # Functional TS
    intraday = randn(rng, 60, 48)  # 60 days, 48 half-hour intervals
    kl = kl_expansion(intraday, n_comps=3)
    println("KL var explained (3 comps): $(round(kl.var_explained[3]*100, digits=1))%")

    # Periodic AR
    seasonal_y = sin.(2π .* (1:T) ./ 12) + randn(rng, T) * 0.3
    par = fit_periodic_ar(seasonal_y, 12, 2)
    println("Periodic AR: $(size(par.coefs)) coefficient matrix")

    # Forecast metrics
    y2 = y[2:T]; y_pred = y[1:T-1]
    fm = forecast_metrics(y2, y_pred)
    println("Naive forecast RMSE: $(round(fm.rmse, digits=5)), Direction: $(round(fm.directional_acc*100, digits=1))%")
end

end # module TimeSeriesAdvanced
