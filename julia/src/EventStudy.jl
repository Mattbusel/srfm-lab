###############################################################################
# EventStudy.jl
#
# Event study methodology: CAR, BHAR, market model, test statistics,
# cross-sectional regressions, calendar-time portfolios, earnings surprise,
# FOMC, IPO, M&A, multi-event studies.
#
# Dependencies: LinearAlgebra, Statistics, Random  (stdlib only)
###############################################################################

module EventStudy

using LinearAlgebra, Statistics, Random

export EventWindow, EventResult, EventStudyConfig
export compute_car, compute_bhar, compute_abnormal_returns
export market_model_ols, market_model_garch, fama_french_model
export patell_test, bmp_test, rank_test, sign_test, generalized_sign_test
export cross_sectional_regression, weighted_cross_section
export calendar_time_portfolio, calendar_time_fama_french
export earnings_surprise, sue_score, analyst_revision_signal
export fomc_event_study, fomc_pre_announcement_drift, fomc_post_reversal
export ipo_event_study, ipo_first_day_return, ipo_long_run_performance
export ma_event_study, target_abnormal_return, acquirer_abnormal_return
export multi_event_study, compound_events, event_interaction
export EventCluster, cluster_adjusted_test

# ─────────────────────────────────────────────────────────────────────────────
# §1  Core Types
# ─────────────────────────────────────────────────────────────────────────────

"""Configuration for event windows."""
struct EventWindow
    estimation_start::Int   # days relative to event (e.g., -270)
    estimation_end::Int     # e.g., -21
    event_start::Int        # e.g., -5
    event_end::Int          # e.g., +5
    pre_event_start::Int    # for pre-event drift analysis
    pre_event_end::Int
    post_event_start::Int
    post_event_end::Int
end

function EventWindow(; est_start::Int=-270, est_end::Int=-21,
                     evt_start::Int=-5, evt_end::Int=5,
                     pre_start::Int=-20, pre_end::Int=-6,
                     post_start::Int=6, post_end::Int=20)
    EventWindow(est_start, est_end, evt_start, evt_end,
                pre_start, pre_end, post_start, post_end)
end

"""Result of an event study for a single event."""
struct EventResult{T<:Real}
    firm_id::Int
    event_date::Int
    alpha::T
    beta::T
    abnormal_returns::Vector{T}
    car::T
    bhar::T
    standardized_car::T
    sigma_ar::T
    t_stat::T
    n_estimation::Int
    event_window_returns::Vector{T}
    market_window_returns::Vector{T}
end

"""Configuration for event study."""
struct EventStudyConfig{T<:Real}
    window::EventWindow
    model::Symbol           # :market_model, :fama_french, :garch
    min_estimation_obs::Int
    significance_level::T
    cluster_adjust::Bool
    robust_se::Bool
end

function EventStudyConfig(; window::EventWindow=EventWindow(),
                          model::Symbol=:market_model,
                          min_estimation_obs::Int=100,
                          significance_level::Float64=0.05,
                          cluster_adjust::Bool=false,
                          robust_se::Bool=false)
    EventStudyConfig{Float64}(window, model, min_estimation_obs,
                              significance_level, cluster_adjust, robust_se)
end

# ─────────────────────────────────────────────────────────────────────────────
# §2  Market Model Estimation
# ─────────────────────────────────────────────────────────────────────────────

"""
    market_model_ols(firm_returns, market_returns) -> alpha, beta, sigma, residuals

OLS market model: R_i = alpha + beta * R_m + epsilon.
"""
function market_model_ols(firm_returns::AbstractVector{T},
                          market_returns::AbstractVector{T}) where T<:Real
    n = length(firm_returns)
    @assert n == length(market_returns)
    X = hcat(ones(T, n), market_returns)
    beta_vec = (X' * X) \ (X' * firm_returns)
    alpha = beta_vec[1]
    beta = beta_vec[2]
    predicted = X * beta_vec
    residuals = firm_returns .- predicted
    sigma = std(residuals)
    return alpha, beta, sigma, residuals
end

"""Market model with Newey-West HAC standard errors."""
function market_model_hac(firm_returns::AbstractVector{T},
                          market_returns::AbstractVector{T};
                          max_lag::Int=5) where T<:Real
    n = length(firm_returns)
    X = hcat(ones(T, n), market_returns)
    beta_vec = (X' * X) \ (X' * firm_returns)
    residuals = firm_returns .- X * beta_vec
    # Newey-West HAC estimator
    XtX_inv = inv(X' * X)
    S = zeros(T, 2, 2)
    for t in 1:n
        S .+= residuals[t]^2 .* (X[t,:] * X[t,:]')
    end
    for lag in 1:max_lag
        w = one(T) - T(lag) / (max_lag + 1)
        G = zeros(T, 2, 2)
        for t in lag+1:n
            G .+= residuals[t] * residuals[t-lag] .* (X[t,:] * X[t-lag,:]')
        end
        S .+= w .* (G .+ G')
    end
    V = XtX_inv * S * XtX_inv
    se = sqrt.(max.(diag(V), T(1e-16)))
    t_stats = beta_vec ./ se
    return beta_vec[1], beta_vec[2], std(residuals), residuals, se, t_stats
end

"""
    market_model_garch(firm_returns, market_returns; kwargs...) -> alpha, beta, sigma_t, residuals

GARCH(1,1)-adjusted market model.
"""
function market_model_garch(firm_returns::AbstractVector{T},
                            market_returns::AbstractVector{T};
                            omega::T=T(1e-6),
                            garch_alpha::T=T(0.1),
                            garch_beta::T=T(0.85),
                            max_iter::Int=200) where T<:Real
    n = length(firm_returns)
    alpha_mm, beta_mm, _, residuals = market_model_ols(firm_returns, market_returns)
    # GARCH(1,1) on residuals
    sigma2 = Vector{T}(undef, n)
    sigma2[1] = var(residuals)
    for t in 2:n
        sigma2[t] = omega + garch_alpha * residuals[t-1]^2 + garch_beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], T(1e-10))
    end
    # Re-estimate with WLS
    W = Diagonal(one(T) ./ sqrt.(sigma2))
    X = hcat(ones(T, n), market_returns)
    Xw = W * X
    yw = W * firm_returns
    beta_wls = (Xw' * Xw) \ (Xw' * yw)
    residuals_wls = firm_returns .- X * beta_wls
    # Update GARCH
    sigma2_new = Vector{T}(undef, n)
    sigma2_new[1] = var(residuals_wls)
    for t in 2:n
        sigma2_new[t] = omega + garch_alpha * residuals_wls[t-1]^2 + garch_beta * sigma2_new[t-1]
        sigma2_new[t] = max(sigma2_new[t], T(1e-10))
    end
    sigma_t = sqrt.(sigma2_new)
    return beta_wls[1], beta_wls[2], sigma_t, residuals_wls
end

"""GARCH(1,1) calibration via MLE."""
function garch_calibrate(returns::AbstractVector{T};
                         max_iter::Int=500, tol::T=T(1e-8)) where T<:Real
    n = length(returns)
    mu = mean(returns)
    eps = returns .- mu
    # Initialize
    omega = var(eps) * T(0.05)
    alpha = T(0.1)
    beta = T(0.85)
    sigma2 = Vector{T}(undef, n)
    for iter in 1:max_iter
        sigma2[1] = omega / (one(T) - alpha - beta + T(1e-8))
        for t in 2:n
            sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], T(1e-10))
        end
        # Negative log-likelihood gradient (numerical)
        ll = -T(0.5) * sum(log.(sigma2) .+ eps.^2 ./ sigma2)
        delta = T(1e-5)
        # Gradient for omega
        omega_p = omega + delta
        s2 = copy(sigma2)
        s2[1] = omega_p / (one(T) - alpha - beta + T(1e-8))
        for t in 2:n
            s2[t] = omega_p + alpha * eps[t-1]^2 + beta * s2[t-1]
            s2[t] = max(s2[t], T(1e-10))
        end
        ll_p = -T(0.5) * sum(log.(s2) .+ eps.^2 ./ s2)
        grad_omega = (ll_p - ll) / delta
        # Gradient for alpha
        alpha_p = alpha + delta
        s2[1] = omega / (one(T) - alpha_p - beta + T(1e-8))
        for t in 2:n
            s2[t] = omega + alpha_p * eps[t-1]^2 + beta * s2[t-1]
            s2[t] = max(s2[t], T(1e-10))
        end
        ll_p = -T(0.5) * sum(log.(s2) .+ eps.^2 ./ s2)
        grad_alpha = (ll_p - ll) / delta
        # Gradient for beta
        beta_p = min(beta + delta, one(T) - alpha - T(0.01))
        s2[1] = omega / (one(T) - alpha - beta_p + T(1e-8))
        for t in 2:n
            s2[t] = omega + alpha * eps[t-1]^2 + beta_p * s2[t-1]
            s2[t] = max(s2[t], T(1e-10))
        end
        ll_p = -T(0.5) * sum(log.(s2) .+ eps.^2 ./ s2)
        grad_beta = (ll_p - ll) / delta
        lr = T(1e-6)
        omega = max(omega + lr * grad_omega, T(1e-10))
        alpha = clamp(alpha + lr * grad_alpha, T(1e-4), T(0.5))
        beta = clamp(beta + lr * grad_beta, T(0.3), one(T) - alpha - T(0.01))
        if abs(grad_omega) + abs(grad_alpha) + abs(grad_beta) < tol
            break
        end
    end
    sigma2[1] = omega / (one(T) - alpha - beta + T(1e-8))
    for t in 2:n
        sigma2[t] = omega + alpha * eps[t-1]^2 + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], T(1e-10))
    end
    return omega, alpha, beta, sqrt.(sigma2)
end

"""Fama-French three-factor model."""
function fama_french_model(firm_returns::AbstractVector{T},
                           market_excess::AbstractVector{T},
                           smb::AbstractVector{T},
                           hml::AbstractVector{T}) where T<:Real
    n = length(firm_returns)
    X = hcat(ones(T, n), market_excess, smb, hml)
    beta_vec = (X' * X) \ (X' * firm_returns)
    predicted = X * beta_vec
    residuals = firm_returns .- predicted
    sigma = std(residuals)
    return beta_vec, sigma, residuals
end

"""Carhart four-factor model."""
function carhart_model(firm_returns::AbstractVector{T},
                       market_excess::AbstractVector{T},
                       smb::AbstractVector{T},
                       hml::AbstractVector{T},
                       mom::AbstractVector{T}) where T<:Real
    n = length(firm_returns)
    X = hcat(ones(T, n), market_excess, smb, hml, mom)
    beta_vec = (X' * X) \ (X' * firm_returns)
    residuals = firm_returns .- X * beta_vec
    sigma = std(residuals)
    return beta_vec, sigma, residuals
end

# ─────────────────────────────────────────────────────────────────────────────
# §3  Abnormal Return Computation
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_abnormal_returns(firm_returns, market_returns, window) -> EventResult

Compute abnormal returns for a single event.
"""
function compute_abnormal_returns(firm_returns::AbstractVector{T},
                                  market_returns::AbstractVector{T},
                                  event_idx::Int,
                                  window::EventWindow;
                                  firm_id::Int=0) where T<:Real
    n = length(firm_returns)
    est_start = event_idx + window.estimation_start
    est_end = event_idx + window.estimation_end
    evt_start = event_idx + window.event_start
    evt_end = event_idx + window.event_end
    if est_start < 1 || evt_end > n || est_end < est_start
        # Not enough data
        return nothing
    end
    # Estimation window
    firm_est = firm_returns[est_start:est_end]
    mkt_est = market_returns[est_start:est_end]
    n_est = length(firm_est)
    alpha, beta, sigma, _ = market_model_ols(firm_est, mkt_est)
    # Event window abnormal returns
    evt_len = evt_end - evt_start + 1
    ar = Vector{T}(undef, evt_len)
    firm_evt = firm_returns[evt_start:evt_end]
    mkt_evt = market_returns[evt_start:evt_end]
    for i in 1:evt_len
        ar[i] = firm_evt[i] - (alpha + beta * mkt_evt[i])
    end
    # CAR
    car = sum(ar)
    # BHAR
    bhar_firm = prod(one(T) .+ firm_evt) - one(T)
    bhar_mkt = prod(one(T) .+ mkt_evt) - one(T)
    bhar = bhar_firm - bhar_mkt
    # Standardized CAR
    sigma_ar = sigma * sqrt(T(evt_len))
    # Prediction error adjustment
    mkt_mean_est = mean(mkt_est)
    mkt_var_est = var(mkt_est) * n_est
    mkt_sum_evt = sum(mkt_evt)
    adj = sqrt(one(T) + T(evt_len)/n_est +
               (mkt_sum_evt - evt_len * mkt_mean_est)^2 / mkt_var_est)
    sigma_car_adj = sigma * adj
    scar = car / max(sigma_car_adj, T(1e-16))
    t_stat = car / max(sigma_ar, T(1e-16))
    return EventResult{T}(firm_id, event_idx, alpha, beta, ar, car, bhar,
                          scar, sigma_ar, t_stat, n_est, firm_evt, mkt_evt)
end

"""
    compute_car(firm_returns, market_returns, event_idx, window) -> car, ar_series

Cumulative Abnormal Return.
"""
function compute_car(firm_returns::AbstractVector{T},
                     market_returns::AbstractVector{T},
                     event_idx::Int,
                     window::EventWindow) where T<:Real
    result = compute_abnormal_returns(firm_returns, market_returns, event_idx, window)
    if result === nothing
        return zero(T), T[]
    end
    return result.car, result.abnormal_returns
end

"""
    compute_bhar(firm_returns, market_returns, event_idx, window) -> bhar

Buy-Hold Abnormal Return.
"""
function compute_bhar(firm_returns::AbstractVector{T},
                      market_returns::AbstractVector{T},
                      event_idx::Int,
                      window::EventWindow) where T<:Real
    result = compute_abnormal_returns(firm_returns, market_returns, event_idx, window)
    if result === nothing
        return zero(T)
    end
    return result.bhar
end

"""Cumulative abnormal return over custom sub-window."""
function car_subwindow(result::EventResult{T}, start_offset::Int, end_offset::Int) where T<:Real
    evt_start = -length(result.abnormal_returns) ÷ 2  # approximate
    # Map offsets to AR indices
    n = length(result.abnormal_returns)
    mid = n ÷ 2 + 1
    idx_start = mid + start_offset
    idx_end = mid + end_offset
    idx_start = clamp(idx_start, 1, n)
    idx_end = clamp(idx_end, 1, n)
    sum(result.abnormal_returns[idx_start:idx_end])
end

"""Multi-factor abnormal returns."""
function compute_abnormal_returns_ff(firm_returns::AbstractVector{T},
                                     market_excess::AbstractVector{T},
                                     smb::AbstractVector{T},
                                     hml::AbstractVector{T},
                                     event_idx::Int,
                                     window::EventWindow) where T<:Real
    n = length(firm_returns)
    est_start = event_idx + window.estimation_start
    est_end = event_idx + window.estimation_end
    evt_start = event_idx + window.event_start
    evt_end = event_idx + window.event_end
    if est_start < 1 || evt_end > n
        return nothing
    end
    firm_est = firm_returns[est_start:est_end]
    mkt_est = market_excess[est_start:est_end]
    smb_est = smb[est_start:est_end]
    hml_est = hml[est_start:est_end]
    betas, sigma, _ = fama_french_model(firm_est, mkt_est, smb_est, hml_est)
    evt_len = evt_end - evt_start + 1
    ar = Vector{T}(undef, evt_len)
    for i in 1:evt_len
        t = evt_start + i - 1
        predicted = betas[1] + betas[2]*market_excess[t] + betas[3]*smb[t] + betas[4]*hml[t]
        ar[i] = firm_returns[t] - predicted
    end
    car = sum(ar)
    return car, ar, betas, sigma
end

# ─────────────────────────────────────────────────────────────────────────────
# §4  Statistical Tests
# ─────────────────────────────────────────────────────────────────────────────

"""Standard normal CDF approximation."""
function _normal_cdf(x::T) where T<:Real
    if x < T(-8) return zero(T) end
    if x > T(8) return one(T) end
    t = one(T) / (one(T) + T(0.2316419) * abs(x))
    d = T(0.3989422804) * exp(-x * x / 2)
    p = d * t * (T(0.3193815) + t * (T(-0.3565638) + t * (T(1.781478) +
        t * (T(-1.8212560) + t * T(1.3302744)))))
    x >= zero(T) ? one(T) - p : p
end

"""Standard normal PDF."""
_normal_pdf(x::T) where T<:Real = exp(-x^2 / 2) / sqrt(T(2π))

"""Inverse normal CDF (Beasley-Springer-Moro)."""
function _normal_quantile(p::T) where T<:Real
    if p <= zero(T) return T(-8.0) end
    if p >= one(T) return T(8.0) end
    # Rational approximation
    a = [T(-3.969683028665376e+01), T(2.209460984245205e+02),
         T(-2.759285104469687e+02), T(1.383577518672690e+02),
         T(-3.066479806614716e+01), T(2.506628277459239e+00)]
    b = [T(-5.447609879822406e+01), T(1.615858368580409e+02),
         T(-1.556989798598866e+02), T(6.680131188771972e+01),
         T(-1.328068155288572e+01)]
    c = [T(-7.784894002430293e-03), T(-3.223964580411365e-01),
         T(-2.400758277161838e+00), T(-2.549732539343734e+00),
         T(4.374664141464968e+00), T(2.938163982698783e+00)]
    d = [T(7.784695709041462e-03), T(3.224671290700398e-01),
         T(2.445134137142996e+00), T(3.754408661907416e+00)]
    p_low = T(0.02425)
    p_high = one(T) - p_low
    if p < p_low
        q = sqrt(-2 * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= p_high
        q = p - T(0.5)
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2 * log(one(T) - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

"""Student's t CDF approximation."""
function _t_cdf(t_val::T, df::Int) where T<:Real
    if df > 200
        return _normal_cdf(t_val)
    end
    x = T(df) / (T(df) + t_val^2)
    # Regularized incomplete beta function approximation
    # Use normal approximation with correction
    g = sqrt(T(2) / (T(df) * pi)) * exp(lgamma(T(df+1)/2) - lgamma(T(df)/2))
    z = t_val * (one(T) - one(T) / (T(4) * T(df)))
    _normal_cdf(z)
end

"""
    patell_test(results) -> z_stat, p_value

Patell (1976) standardized residual test.
"""
function patell_test(results::Vector{EventResult{T}}) where T<:Real
    N = length(results)
    if N == 0
        return zero(T), one(T)
    end
    # Sum of standardized CARs
    sum_scar = sum(r.standardized_car for r in results)
    z = sum_scar / sqrt(T(N))
    p = 2 * (one(T) - _normal_cdf(abs(z)))
    return z, p
end

"""
    bmp_test(results) -> t_stat, p_value

Boehmer-Musumeci-Poulsen (1991) cross-sectional t-test with standardized residuals.
"""
function bmp_test(results::Vector{EventResult{T}}) where T<:Real
    N = length(results)
    if N < 2
        return zero(T), one(T)
    end
    scars = [r.standardized_car for r in results]
    mean_scar = mean(scars)
    std_scar = std(scars)
    t_stat = mean_scar * sqrt(T(N)) / max(std_scar, T(1e-16))
    p = 2 * (one(T) - _t_cdf(abs(t_stat), N - 1))
    return t_stat, p
end

"""
    rank_test(results, all_firm_returns, market_returns, event_windows) -> z_stat, p_value

Corrado (1989) rank test.
"""
function rank_test(results::Vector{EventResult{T}};
                   standardize::Bool=true) where T<:Real
    N = length(results)
    if N == 0
        return zero(T), one(T)
    end
    # Rank the abnormal returns within each event's combined estimation+event period
    K_sum = zero(T)
    var_sum = zero(T)
    for result in results
        n_est = result.n_estimation
        ar = result.abnormal_returns
        L = length(ar)
        total_n = n_est + L
        # Compute rank of event window AR among all estimation residuals + event AR
        # Use mean rank - (n+1)/2
        mid_rank = T(total_n + 1) / 2
        for a in ar
            # Approximate rank based on CDF of standardized residual
            z_a = a / max(result.sigma_ar / sqrt(T(L)), T(1e-16))
            rank_approx = _normal_cdf(z_a) * total_n
            K_sum += (rank_approx - mid_rank) / total_n
        end
        var_sum += T(L) / T(12)  # variance of uniform rank
    end
    z = K_sum / sqrt(max(var_sum, T(1e-16)))
    p = 2 * (one(T) - _normal_cdf(abs(z)))
    return z, p
end

"""
    sign_test(results) -> z_stat, p_value

Nonparametric sign test for CARs.
"""
function sign_test(results::Vector{EventResult{T}}) where T<:Real
    N = length(results)
    if N == 0
        return zero(T), one(T)
    end
    n_positive = count(r -> r.car > zero(T), results)
    p_hat = T(n_positive) / T(N)
    z = (p_hat - T(0.5)) / sqrt(T(0.25) / T(N))
    p = 2 * (one(T) - _normal_cdf(abs(z)))
    return z, p
end

"""
    generalized_sign_test(results, expected_positive_frac) -> z_stat, p_value

Generalized sign test (adjusts for expected fraction of positive ARs).
"""
function generalized_sign_test(results::Vector{EventResult{T}};
                                expected_frac::T=T(0.5)) where T<:Real
    N = length(results)
    if N == 0
        return zero(T), one(T)
    end
    n_positive = count(r -> r.car > zero(T), results)
    z = (T(n_positive) - T(N) * expected_frac) /
        sqrt(T(N) * expected_frac * (one(T) - expected_frac))
    p = 2 * (one(T) - _normal_cdf(abs(z)))
    return z, p
end

"""Wilcoxon signed-rank test for CARs."""
function wilcoxon_test(results::Vector{EventResult{T}}) where T<:Real
    N = length(results)
    if N == 0
        return zero(T), one(T)
    end
    cars = [r.car for r in results]
    abs_cars = abs.(cars)
    sorted_idx = sortperm(abs_cars)
    ranks = zeros(T, N)
    for (r, i) in enumerate(sorted_idx)
        ranks[i] = T(r)
    end
    W_plus = sum(ranks[i] for i in 1:N if cars[i] > zero(T); init=zero(T))
    E_W = T(N) * T(N + 1) / 4
    Var_W = T(N) * T(N + 1) * T(2N + 1) / 24
    z = (W_plus - E_W) / sqrt(max(Var_W, T(1e-16)))
    p = 2 * (one(T) - _normal_cdf(abs(z)))
    return z, p
end

"""Kolari-Pynnonen adjusted BMP test (accounts for cross-correlation)."""
function kp_test(results::Vector{EventResult{T}}) where T<:Real
    N = length(results)
    if N < 3
        return bmp_test(results)
    end
    t_bmp, _ = bmp_test(results)
    scars = [r.standardized_car for r in results]
    # Estimate average cross-correlation
    avg_corr = zero(T)
    count = 0
    for i in 1:N
        for j in i+1:N
            L = min(length(results[i].abnormal_returns),
                    length(results[j].abnormal_returns))
            if L > 0
                ari = results[i].abnormal_returns[1:L]
                arj = results[j].abnormal_returns[1:L]
                if std(ari) > T(1e-16) && std(arj) > T(1e-16)
                    avg_corr += cor(ari, arj)
                    count += 1
                end
            end
        end
    end
    if count > 0
        avg_corr /= count
    end
    adj = sqrt(one(T) + (T(N) - one(T)) * avg_corr)
    t_kp = t_bmp / adj
    p = 2 * (one(T) - _normal_cdf(abs(t_kp)))
    return t_kp, p
end

# ─────────────────────────────────────────────────────────────────────────────
# §5  Cross-Sectional Regression of CARs
# ─────────────────────────────────────────────────────────────────────────────

"""
    cross_sectional_regression(results, firm_chars) -> betas, t_stats, r_squared

Regress CARs on firm characteristics.
"""
function cross_sectional_regression(results::Vector{EventResult{T}},
                                     firm_chars::AbstractMatrix{T}) where T<:Real
    N = length(results)
    k = size(firm_chars, 2)
    cars = [r.car for r in results]
    Y = convert(Vector{T}, cars)
    X = hcat(ones(T, N), firm_chars)
    beta = (X' * X) \ (X' * Y)
    predicted = X * beta
    residuals = Y .- predicted
    ss_res = dot(residuals, residuals)
    ss_tot = dot(Y .- mean(Y), Y .- mean(Y))
    r_squared = one(T) - ss_res / max(ss_tot, T(1e-16))
    # Standard errors
    mse = ss_res / max(N - k - 1, 1)
    XtX_inv = inv(X' * X)
    se = sqrt.(max.(diag(XtX_inv) .* mse, T(1e-16)))
    t_stats = beta ./ se
    return beta, t_stats, r_squared, se
end

"""Weighted cross-sectional regression (WLS on CARs)."""
function weighted_cross_section(results::Vector{EventResult{T}},
                                firm_chars::AbstractMatrix{T}) where T<:Real
    N = length(results)
    cars = [r.car for r in results]
    weights = [one(T) / max(r.sigma_ar^2, T(1e-16)) for r in results]
    W = Diagonal(weights)
    Y = convert(Vector{T}, cars)
    X = hcat(ones(T, N), firm_chars)
    beta = (X' * W * X) \ (X' * W * Y)
    predicted = X * beta
    residuals = Y .- predicted
    mse = dot(residuals, W * residuals) / max(N - size(X, 2), 1)
    se = sqrt.(max.(diag(inv(X' * W * X)) .* mse, T(1e-16)))
    t_stats = beta ./ se
    return beta, t_stats, se
end

"""Fama-MacBeth style two-pass regression."""
function fama_macbeth_regression(panel_cars::Matrix{T},
                                 panel_chars::Array{T, 3}) where T<:Real
    # panel_cars: N_firms x N_periods
    # panel_chars: N_firms x N_chars x N_periods
    N, n_periods = size(panel_cars)
    n_chars = size(panel_chars, 2)
    betas = Matrix{T}(undef, n_periods, n_chars + 1)
    for t in 1:n_periods
        Y = panel_cars[:, t]
        X = hcat(ones(T, N), panel_chars[:, :, t])
        betas[t, :] = (X' * X) \ (X' * Y)
    end
    mean_beta = vec(mean(betas; dims=1))
    se_beta = vec(std(betas; dims=1)) ./ sqrt(T(n_periods))
    t_stats = mean_beta ./ max.(se_beta, T(1e-16))
    return mean_beta, t_stats, se_beta, betas
end

# ─────────────────────────────────────────────────────────────────────────────
# §6  Calendar-Time Portfolio Approach
# ─────────────────────────────────────────────────────────────────────────────

"""
    calendar_time_portfolio(returns, event_dates, event_firms; kwargs...) -> port_returns

Form calendar-time portfolios around events.
"""
function calendar_time_portfolio(returns::AbstractMatrix{T},
                                  event_dates::AbstractVector{Int},
                                  event_firms::AbstractVector{Int};
                                  hold_period::Int=21,
                                  weighting::Symbol=:equal) where T<:Real
    n_obs, n_firms = size(returns)
    port_returns = Vector{T}()
    for t in 1:n_obs
        # Find firms in event window at time t
        active_firms = Int[]
        active_weights = T[]
        for (k, (d, f)) in enumerate(zip(event_dates, event_firms))
            if d <= t <= d + hold_period && 1 <= f <= n_firms
                push!(active_firms, f)
                if weighting == :equal
                    push!(active_weights, one(T))
                elseif weighting == :value
                    push!(active_weights, T(k))  # placeholder
                end
            end
        end
        if !isempty(active_firms)
            w = active_weights ./ sum(active_weights)
            r = sum(w[i] * returns[t, active_firms[i]] for i in eachindex(active_firms))
            push!(port_returns, r)
        else
            push!(port_returns, zero(T))
        end
    end
    port_returns
end

"""Calendar-time with Fama-French regression."""
function calendar_time_fama_french(port_returns::AbstractVector{T},
                                    market_excess::AbstractVector{T},
                                    smb::AbstractVector{T},
                                    hml::AbstractVector{T}) where T<:Real
    n = min(length(port_returns), length(market_excess), length(smb), length(hml))
    pr = port_returns[1:n]
    me = market_excess[1:n]
    s = smb[1:n]
    h = hml[1:n]
    betas, sigma, residuals = fama_french_model(pr, me, s, h)
    alpha = betas[1]
    X = hcat(ones(T, n), me, s, h)
    XtX_inv = inv(X' * X)
    mse = sum(residuals .^ 2) / max(n - 4, 1)
    se_alpha = sqrt(max(XtX_inv[1,1] * mse, T(1e-16)))
    t_alpha = alpha / se_alpha
    return (alpha=alpha, t_stat=t_alpha, betas=betas, sigma=sigma)
end

"""DGTW-adjusted returns (Daniel, Grinblatt, Titman, Wermers)."""
function dgtw_adjustment(firm_returns::AbstractVector{T},
                         benchmark_returns::AbstractVector{T}) where T<:Real
    firm_returns .- benchmark_returns
end

# ─────────────────────────────────────────────────────────────────────────────
# §7  Event Clustering Adjustment
# ─────────────────────────────────────────────────────────────────────────────

struct EventCluster
    events::Vector{Tuple{Int, Int}}  # (event_date, firm_id) pairs
    cluster_start::Int
    cluster_end::Int
end

"""Detect event clusters (overlapping event windows)."""
function detect_clusters(event_dates::AbstractVector{Int},
                          event_firms::AbstractVector{Int};
                          window_size::Int=11) where T
    n = length(event_dates)
    sorted_idx = sortperm(event_dates)
    clusters = Vector{EventCluster}()
    used = falses(n)
    for i in 1:n
        if used[sorted_idx[i]]
            continue
        end
        cluster_events = [(event_dates[sorted_idx[i]], event_firms[sorted_idx[i]])]
        used[sorted_idx[i]] = true
        cluster_end = event_dates[sorted_idx[i]] + window_size
        for j in i+1:n
            if used[sorted_idx[j]]
                continue
            end
            if event_dates[sorted_idx[j]] <= cluster_end
                push!(cluster_events, (event_dates[sorted_idx[j]], event_firms[sorted_idx[j]]))
                used[sorted_idx[j]] = true
                cluster_end = max(cluster_end, event_dates[sorted_idx[j]] + window_size)
            else
                break
            end
        end
        if length(cluster_events) > 1
            cs = minimum(d for (d, _) in cluster_events)
            ce = maximum(d for (d, _) in cluster_events) + window_size
            push!(clusters, EventCluster(cluster_events, cs, ce))
        end
    end
    clusters
end

"""
    cluster_adjusted_test(results, clusters) -> z_stat, p_value

Adjust test statistic for event clustering.
"""
function cluster_adjusted_test(results::Vector{EventResult{T}},
                                clusters::Vector{EventCluster}) where T<:Real
    N = length(results)
    if N == 0
        return zero(T), one(T)
    end
    cars = [r.car for r in results]
    mean_car = mean(cars)
    # Cluster-robust variance
    # Group CARs by cluster membership
    clustered_firms = Set{Int}()
    for c in clusters
        for (_, f) in c.events
            push!(clustered_firms, f)
        end
    end
    # Variance estimation
    if isempty(clusters)
        se = std(cars) / sqrt(T(N))
    else
        # Cluster-level aggregation
        cluster_sums = T[]
        non_cluster_cars = T[]
        cluster_firm_ids = Set{Int}()
        for c in clusters
            s = zero(T)
            for (_, f) in c.events
                push!(cluster_firm_ids, f)
                # Find matching result
                for r in results
                    if r.firm_id == f
                        s += r.car
                        break
                    end
                end
            end
            push!(cluster_sums, s)
        end
        for r in results
            if !(r.firm_id in cluster_firm_ids)
                push!(non_cluster_cars, r.car)
            end
        end
        all_items = vcat(cluster_sums, non_cluster_cars)
        M = length(all_items)
        se = std(all_items) / sqrt(T(max(M, 1)))
    end
    t_stat = mean_car / max(se, T(1e-16))
    p = 2 * (one(T) - _normal_cdf(abs(t_stat)))
    return t_stat, p
end

# ─────────────────────────────────────────────────────────────────────────────
# §8  Earnings Surprise Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    sue_score(actual_eps, forecast_eps, std_surprise) -> SUE

Standardized Unexpected Earnings.
"""
function sue_score(actual_eps::T, forecast_eps::T, std_surprise::T) where T<:Real
    (actual_eps - forecast_eps) / max(std_surprise, T(1e-8))
end

"""Compute SUE from time series of earnings."""
function sue_timeseries(earnings::AbstractVector{T}; lookback::Int=8) where T<:Real
    n = length(earnings)
    sue = Vector{T}(undef, n)
    for t in 1:n
        if t <= lookback
            sue[t] = zero(T)
            continue
        end
        past = earnings[t-lookback:t-1]
        # Seasonal random walk: surprise = E_t - E_{t-4}
        if t > 4
            surprise = earnings[t] - earnings[t-4]
            sigma = std(diff(past))
            sue[t] = surprise / max(sigma, T(1e-8))
        else
            sue[t] = zero(T)
        end
    end
    sue
end

"""
    analyst_revision_signal(forecasts_old, forecasts_new, prices) -> signal

Signal from analyst forecast revisions.
"""
function analyst_revision_signal(forecasts_old::AbstractVector{T},
                                  forecasts_new::AbstractVector{T},
                                  prices::AbstractVector{T}) where T<:Real
    n = length(forecasts_old)
    signal = Vector{T}(undef, n)
    for i in 1:n
        revision = forecasts_new[i] - forecasts_old[i]
        signal[i] = revision / max(abs(prices[i]), T(1e-8))
    end
    signal
end

"""Post-earnings announcement drift (PEAD) analysis."""
function pead_analysis(firm_returns::AbstractMatrix{T},
                       market_returns::AbstractVector{T},
                       sue_scores::AbstractVector{T},
                       announcement_dates::AbstractVector{Int};
                       n_quantiles::Int=5,
                       hold_days::Int=63) where T<:Real
    n_events = length(announcement_dates)
    n_assets = size(firm_returns, 2)
    # Sort events by SUE
    sorted_idx = sortperm(sue_scores)
    q_size = div(n_events, n_quantiles)
    quantile_returns = Matrix{T}(undef, hold_days, n_quantiles)
    fill!(quantile_returns, zero(T))
    for q in 1:n_quantiles
        start_idx = (q - 1) * q_size + 1
        end_idx = q == n_quantiles ? n_events : q * q_size
        events_in_q = sorted_idx[start_idx:end_idx]
        n_q = length(events_in_q)
        for k in events_in_q
            date = announcement_dates[k]
            firm = min(k, n_assets)  # simplified mapping
            for d in 1:hold_days
                t = date + d
                if 1 <= t <= size(firm_returns, 1)
                    quantile_returns[d, q] += firm_returns[t, firm] / n_q
                end
            end
        end
    end
    # Long-short: Q5 - Q1
    long_short = cumsum(quantile_returns[:, n_quantiles] .- quantile_returns[:, 1])
    return quantile_returns, long_short
end

"""
    earnings_surprise(actual, consensus, price; method=:sue) -> score

Generic earnings surprise score.
"""
function earnings_surprise(actual::T, consensus::T, price::T;
                           method::Symbol=:sue) where T<:Real
    if method == :sue
        return (actual - consensus) / max(abs(price), T(1e-8))
    elseif method == :pct
        return (actual - consensus) / max(abs(consensus), T(1e-8))
    else
        return actual - consensus
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# §9  FOMC Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""
    fomc_event_study(market_returns, fomc_dates; kwargs...) -> results

Analyze market behavior around FOMC announcements.
"""
function fomc_event_study(market_returns::AbstractVector{T},
                          fomc_dates::AbstractVector{Int};
                          pre_window::Int=5,
                          post_window::Int=5,
                          rate_changes::Union{Nothing, AbstractVector{T}}=nothing) where T<:Real
    n = length(market_returns)
    n_events = length(fomc_dates)
    pre_returns = Matrix{T}(undef, n_events, pre_window)
    post_returns = Matrix{T}(undef, n_events, post_window)
    event_day_return = Vector{T}(undef, n_events)
    fill!(pre_returns, zero(T))
    fill!(post_returns, zero(T))
    for (k, d) in enumerate(fomc_dates)
        if d <= pre_window || d + post_window > n
            event_day_return[k] = zero(T)
            continue
        end
        event_day_return[k] = market_returns[d]
        for i in 1:pre_window
            pre_returns[k, i] = market_returns[d - pre_window + i - 1]
        end
        for i in 1:post_window
            post_returns[k, i] = market_returns[d + i]
        end
    end
    # Average across events
    avg_pre = vec(mean(pre_returns; dims=1))
    avg_post = vec(mean(post_returns; dims=1))
    avg_event = mean(event_day_return)
    # Pre-announcement drift
    pre_cum = cumsum(avg_pre)
    pre_drift = pre_cum[end]
    # Post-announcement reversal
    post_cum = cumsum(avg_post)
    post_reversal = post_cum[end]
    result = Dict{Symbol, Any}()
    result[:avg_pre_returns] = avg_pre
    result[:avg_post_returns] = avg_post
    result[:avg_event_day] = avg_event
    result[:pre_drift] = pre_drift
    result[:post_reversal] = post_reversal
    result[:pre_t_stat] = pre_drift / (std(vec(sum(pre_returns; dims=2))) / sqrt(T(n_events)) + T(1e-16))
    result[:event_t_stat] = avg_event / (std(event_day_return) / sqrt(T(n_events)) + T(1e-16))
    # Conditional on rate change direction
    if rate_changes !== nothing
        hike_idx = findall(x -> x > zero(T), rate_changes)
        cut_idx = findall(x -> x < zero(T), rate_changes)
        hold_idx = findall(x -> x == zero(T), rate_changes)
        result[:hike_avg_return] = isempty(hike_idx) ? zero(T) : mean(event_day_return[hike_idx])
        result[:cut_avg_return] = isempty(cut_idx) ? zero(T) : mean(event_day_return[cut_idx])
        result[:hold_avg_return] = isempty(hold_idx) ? zero(T) : mean(event_day_return[hold_idx])
    end
    return result
end

"""Pre-announcement drift analysis."""
function fomc_pre_announcement_drift(market_returns::AbstractVector{T},
                                      fomc_dates::AbstractVector{Int};
                                      window::Int=10) where T<:Real
    n = length(market_returns)
    n_events = length(fomc_dates)
    drift_curves = Matrix{T}(undef, n_events, window)
    fill!(drift_curves, zero(T))
    for (k, d) in enumerate(fomc_dates)
        if d - window < 1
            continue
        end
        cum = zero(T)
        for i in 1:window
            cum += market_returns[d - window + i]
            drift_curves[k, i] = cum
        end
    end
    avg_drift = vec(mean(drift_curves; dims=1))
    se_drift = vec(std(drift_curves; dims=1)) ./ sqrt(T(n_events))
    return avg_drift, se_drift
end

"""Post-announcement reversal analysis."""
function fomc_post_reversal(market_returns::AbstractVector{T},
                             fomc_dates::AbstractVector{Int};
                             window::Int=10) where T<:Real
    n = length(market_returns)
    n_events = length(fomc_dates)
    reversal_curves = Matrix{T}(undef, n_events, window)
    fill!(reversal_curves, zero(T))
    for (k, d) in enumerate(fomc_dates)
        if d + window > n
            continue
        end
        cum = zero(T)
        for i in 1:window
            cum += market_returns[d + i]
            reversal_curves[k, i] = cum
        end
    end
    avg_reversal = vec(mean(reversal_curves; dims=1))
    se_reversal = vec(std(reversal_curves; dims=1)) ./ sqrt(T(n_events))
    return avg_reversal, se_reversal
end

# ─────────────────────────────────────────────────────────────────────────────
# §10  IPO Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""
    ipo_first_day_return(offer_price, close_price) -> underpricing

IPO first-day return (underpricing).
"""
function ipo_first_day_return(offer_price::T, close_price::T) where T<:Real
    (close_price - offer_price) / max(offer_price, T(1e-8))
end

"""
    ipo_long_run_performance(ipo_returns, market_returns; horizon=756) -> bhar

IPO long-run abnormal performance (typically 3 years).
"""
function ipo_long_run_performance(ipo_returns::AbstractVector{T},
                                   market_returns::AbstractVector{T};
                                   horizon::Int=756) where T<:Real
    h = min(horizon, length(ipo_returns), length(market_returns))
    bhar_ipo = prod(one(T) .+ ipo_returns[1:h]) - one(T)
    bhar_mkt = prod(one(T) .+ market_returns[1:h]) - one(T)
    bhar_ipo - bhar_mkt
end

"""
    ipo_event_study(ipo_data; kwargs...) -> results

Full IPO event study.
"""
function ipo_event_study(ipo_returns::AbstractMatrix{T},
                          market_returns::AbstractVector{T},
                          offer_prices::AbstractVector{T},
                          close_prices::AbstractVector{T},
                          ipo_dates::AbstractVector{Int};
                          horizons::Vector{Int}=[21, 63, 126, 252, 504, 756]) where T<:Real
    n_ipos = size(ipo_returns, 1)
    result = Dict{Symbol, Any}()
    # First-day returns
    fd_returns = [ipo_first_day_return(offer_prices[i], close_prices[i]) for i in 1:n_ipos]
    result[:avg_first_day_return] = mean(fd_returns)
    result[:median_first_day_return] = sort(fd_returns)[div(n_ipos, 2) + 1]
    result[:pct_positive_first_day] = mean(fd_returns .> zero(T))
    # Long-run performance at each horizon
    for h in horizons
        bhars = T[]
        for i in 1:n_ipos
            d = ipo_dates[i]
            if d + h <= length(market_returns)
                ipo_cum = prod(one(T) .+ ipo_returns[i, 1:min(h, size(ipo_returns, 2))])
                mkt_cum = prod(one(T) .+ market_returns[d+1:d+h])
                push!(bhars, ipo_cum - mkt_cum)
            end
        end
        if !isempty(bhars)
            result[Symbol("bhar_$(h)d")] = mean(bhars)
            result[Symbol("bhar_$(h)d_tstat")] = mean(bhars) / (std(bhars) / sqrt(T(length(bhars))) + T(1e-16))
        end
    end
    # Cross-sectional: first-day return predicting long-run?
    result[:first_day_returns] = fd_returns
    return result
end

"""IPO size and sector effects."""
function ipo_size_effect(first_day_returns::AbstractVector{T},
                         market_caps::AbstractVector{T};
                         n_groups::Int=5) where T<:Real
    n = length(first_day_returns)
    sorted_idx = sortperm(market_caps)
    group_size = div(n, n_groups)
    group_means = Vector{T}(undef, n_groups)
    for g in 1:n_groups
        start_idx = (g - 1) * group_size + 1
        end_idx = g == n_groups ? n : g * group_size
        group_means[g] = mean(first_day_returns[sorted_idx[start_idx:end_idx]])
    end
    spread = group_means[1] - group_means[end]
    return group_means, spread
end

# ─────────────────────────────────────────────────────────────────────────────
# §11  M&A Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""
    target_abnormal_return(target_returns, market_returns, announce_date, window) -> car

Abnormal return for acquisition target.
"""
function target_abnormal_return(target_returns::AbstractVector{T},
                                 market_returns::AbstractVector{T},
                                 announce_date::Int;
                                 window::EventWindow=EventWindow()) where T<:Real
    result = compute_abnormal_returns(target_returns, market_returns, announce_date, window)
    result === nothing ? zero(T) : result.car
end

"""Abnormal return for acquirer."""
function acquirer_abnormal_return(acquirer_returns::AbstractVector{T},
                                   market_returns::AbstractVector{T},
                                   announce_date::Int;
                                   window::EventWindow=EventWindow()) where T<:Real
    result = compute_abnormal_returns(acquirer_returns, market_returns, announce_date, window)
    result === nothing ? zero(T) : result.car
end

"""
    ma_event_study(target_returns, acquirer_returns, market_returns, dates; kwargs...) -> results

Full M&A event study: target and acquirer abnormal returns.
"""
function ma_event_study(target_returns_all::AbstractMatrix{T},
                         acquirer_returns_all::AbstractMatrix{T},
                         market_returns::AbstractVector{T},
                         announce_dates::AbstractVector{Int};
                         window::EventWindow=EventWindow(),
                         deal_values::Union{Nothing, AbstractVector{T}}=nothing,
                         payment_methods::Union{Nothing, AbstractVector{Int}}=nothing) where T<:Real
    n_deals = length(announce_dates)
    target_cars = Vector{T}(undef, n_deals)
    acquirer_cars = Vector{T}(undef, n_deals)
    combined_cars = Vector{T}(undef, n_deals)
    target_results = Vector{EventResult{T}}()
    acquirer_results = Vector{EventResult{T}}()
    for i in 1:n_deals
        n_t = size(target_returns_all, 2)
        n_a = size(acquirer_returns_all, 2)
        t_col = min(i, n_t)
        a_col = min(i, n_a)
        tr = compute_abnormal_returns(target_returns_all[:, t_col], market_returns,
                                      announce_dates[i], window; firm_id=i)
        ar = compute_abnormal_returns(acquirer_returns_all[:, a_col], market_returns,
                                      announce_dates[i], window; firm_id=i + n_deals)
        target_cars[i] = tr === nothing ? zero(T) : tr.car
        acquirer_cars[i] = ar === nothing ? zero(T) : ar.car
        combined_cars[i] = target_cars[i] + acquirer_cars[i]
        if tr !== nothing push!(target_results, tr) end
        if ar !== nothing push!(acquirer_results, ar) end
    end
    result = Dict{Symbol, Any}()
    result[:avg_target_car] = mean(target_cars)
    result[:avg_acquirer_car] = mean(acquirer_cars)
    result[:avg_combined_car] = mean(combined_cars)
    result[:target_t_stat] = !isempty(target_results) ? first(patell_test(target_results)) : zero(T)
    result[:acquirer_t_stat] = !isempty(acquirer_results) ? first(patell_test(acquirer_results)) : zero(T)
    # Payment method analysis
    if payment_methods !== nothing
        cash_idx = findall(x -> x == 1, payment_methods)
        stock_idx = findall(x -> x == 2, payment_methods)
        if !isempty(cash_idx)
            result[:cash_acquirer_car] = mean(acquirer_cars[cash_idx])
        end
        if !isempty(stock_idx)
            result[:stock_acquirer_car] = mean(acquirer_cars[stock_idx])
        end
    end
    # Deal size analysis
    if deal_values !== nothing
        median_val = sort(deal_values)[div(n_deals, 2) + 1]
        large_idx = findall(x -> x >= median_val, deal_values)
        small_idx = findall(x -> x < median_val, deal_values)
        result[:large_deal_target_car] = mean(target_cars[large_idx])
        result[:small_deal_target_car] = mean(target_cars[small_idx])
    end
    result[:target_results] = target_results
    result[:acquirer_results] = acquirer_results
    return result
end

"""Merger arbitrage spread calculation."""
function merger_arb_spread(target_price::T, offer_price::T,
                           prob_completion::T; days_to_close::Int=60,
                           rf::T=T(0.02)) where T<:Real
    gross_spread = (offer_price - target_price) / target_price
    annualized = gross_spread * T(252) / T(days_to_close)
    expected = prob_completion * gross_spread - (one(T) - prob_completion) * gross_spread * T(2)
    return (gross_spread=gross_spread, annualized=annualized,
            expected_return=expected, excess_over_rf=annualized - rf)
end

# ─────────────────────────────────────────────────────────────────────────────
# §12  Multi-Event Studies
# ─────────────────────────────────────────────────────────────────────────────

"""
    multi_event_study(returns, market_returns, events; kwargs...) -> results

Run event study across multiple event types.
"""
function multi_event_study(returns::AbstractMatrix{T},
                           market_returns::AbstractVector{T},
                           events::Vector{Tuple{Int, Int, Symbol}};
                           window::EventWindow=EventWindow()) where T<:Real
    # events: (date, firm_id, event_type)
    results_by_type = Dict{Symbol, Vector{EventResult{T}}}()
    for (date, firm, etype) in events
        if !haskey(results_by_type, etype)
            results_by_type[etype] = EventResult{T}[]
        end
        n_firms = size(returns, 2)
        if 1 <= firm <= n_firms
            result = compute_abnormal_returns(returns[:, firm], market_returns,
                                              date, window; firm_id=firm)
            if result !== nothing
                push!(results_by_type[etype], result)
            end
        end
    end
    summary = Dict{Symbol, Dict{Symbol, T}}()
    for (etype, results) in results_by_type
        d = Dict{Symbol, T}()
        if !isempty(results)
            cars = [r.car for r in results]
            d[:n_events] = T(length(results))
            d[:mean_car] = mean(cars)
            d[:median_car] = sort(cars)[div(length(cars), 2) + 1]
            d[:std_car] = std(cars)
            z, p = patell_test(results)
            d[:patell_z] = z
            d[:patell_p] = p
            z2, p2 = bmp_test(results)
            d[:bmp_t] = z2
            d[:bmp_p] = p2
        end
        summary[etype] = d
    end
    return summary, results_by_type
end

"""
    compound_events(results1, results2) -> interaction_car

Analyze compound effect of overlapping events.
"""
function compound_events(results1::Vector{EventResult{T}},
                          results2::Vector{EventResult{T}}) where T<:Real
    # Match events by firm_id
    cars1 = Dict(r.firm_id => r.car for r in results1)
    cars2 = Dict(r.firm_id => r.car for r in results2)
    common = intersect(keys(cars1), keys(cars2))
    if isempty(common)
        return zero(T), T[], T[]
    end
    joint_cars = [cars1[f] + cars2[f] for f in common]
    solo1 = [cars1[f] for f in common]
    solo2 = [cars2[f] for f in common]
    interaction = mean(joint_cars) - mean(solo1) - mean(solo2)
    return interaction, joint_cars, solo1
end

"""
    event_interaction(cars, indicator1, indicator2) -> interaction_effect

Test interaction between two event characteristics.
"""
function event_interaction(cars::AbstractVector{T},
                           indicator1::AbstractVector{Bool},
                           indicator2::AbstractVector{Bool}) where T<:Real
    n = length(cars)
    # 2x2 design
    both = cars[indicator1 .& indicator2]
    only1 = cars[indicator1 .& .!indicator2]
    only2 = cars[.!indicator1 .& indicator2]
    neither = cars[.!indicator1 .& .!indicator2]
    means = Dict{Symbol, T}()
    means[:both] = isempty(both) ? zero(T) : mean(both)
    means[:only1] = isempty(only1) ? zero(T) : mean(only1)
    means[:only2] = isempty(only2) ? zero(T) : mean(only2)
    means[:neither] = isempty(neither) ? zero(T) : mean(neither)
    interaction = means[:both] - means[:only1] - means[:only2] + means[:neither]
    means[:interaction] = interaction
    # T-test on interaction
    all_data = vcat(both, only1, only2, neither)
    se = std(all_data) * sqrt(one(T)/max(length(both),1) + one(T)/max(length(only1),1) +
                               one(T)/max(length(only2),1) + one(T)/max(length(neither),1))
    means[:interaction_t] = interaction / max(se, T(1e-16))
    return means
end

# ─────────────────────────────────────────────────────────────────────────────
# §13  Long-Horizon Event Studies
# ─────────────────────────────────────────────────────────────────────────────

"""Long-horizon BHAR with bootstrapped confidence intervals."""
function long_horizon_bhar(firm_returns::AbstractVector{T},
                           benchmark_returns::AbstractVector{T},
                           horizon::Int;
                           n_bootstrap::Int=1000,
                           rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    h = min(horizon, length(firm_returns), length(benchmark_returns))
    bhar = prod(one(T) .+ firm_returns[1:h]) - prod(one(T) .+ benchmark_returns[1:h])
    # Bootstrap confidence interval
    boot_bhars = Vector{T}(undef, n_bootstrap)
    for b in 1:n_bootstrap
        idx = rand(rng, 1:h, h)
        boot_firm = prod(one(T) .+ firm_returns[idx])
        boot_bench = prod(one(T) .+ benchmark_returns[idx])
        boot_bhars[b] = boot_firm - boot_bench
    end
    sorted = sort(boot_bhars)
    ci_lo = sorted[max(1, round(Int, 0.025 * n_bootstrap))]
    ci_hi = sorted[min(n_bootstrap, round(Int, 0.975 * n_bootstrap))]
    return (bhar=bhar, ci_lower=ci_lo, ci_upper=ci_hi, se=std(boot_bhars))
end

"""Ibbotson RATS (Returns Across Time and Securities) method."""
function ibbotson_rats(firm_returns_panel::AbstractMatrix{T},
                       market_returns::AbstractVector{T},
                       event_dates::AbstractVector{Int};
                       max_horizon::Int=252) where T<:Real
    n_events = length(event_dates)
    n_obs = length(market_returns)
    monthly_alphas = Vector{T}(undef, max_horizon)
    monthly_t_stats = Vector{T}(undef, max_horizon)
    for tau in 1:max_horizon
        event_returns = T[]
        mkt_returns = T[]
        for (k, d) in enumerate(event_dates)
            t = d + tau
            if 1 <= t <= n_obs && k <= size(firm_returns_panel, 2)
                push!(event_returns, firm_returns_panel[t, k])
                push!(mkt_returns, market_returns[t])
            end
        end
        if length(event_returns) > 2
            X = hcat(ones(T, length(mkt_returns)), mkt_returns)
            beta = (X' * X) \ (X' * event_returns)
            residuals = event_returns .- X * beta
            se_alpha = std(residuals) / sqrt(T(length(event_returns)))
            monthly_alphas[tau] = beta[1]
            monthly_t_stats[tau] = beta[1] / max(se_alpha, T(1e-16))
        else
            monthly_alphas[tau] = zero(T)
            monthly_t_stats[tau] = zero(T)
        end
    end
    cum_alpha = cumsum(monthly_alphas)
    return (monthly_alphas=monthly_alphas, cum_alpha=cum_alpha,
            t_stats=monthly_t_stats)
end

# ─────────────────────────────────────────────────────────────────────────────
# §14  Event Study Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

"""Test for event-induced variance increase."""
function variance_ratio_test(result::EventResult{T}) where T<:Real
    # Compare event window variance to estimation window variance
    event_var = var(result.abnormal_returns)
    est_var = result.sigma_ar^2
    F_stat = event_var / max(est_var, T(1e-16))
    n1 = length(result.abnormal_returns)
    n2 = result.n_estimation
    # Approximate p-value using normal approx of log F
    z = (log(F_stat) + one(T) / (n1 - 1) - one(T) / (n2 - 1)) /
        sqrt(T(2) / (n1 - 1) + T(2) / (n2 - 1))
    p = 2 * (one(T) - _normal_cdf(abs(z)))
    return F_stat, p
end

"""Test for normality of abnormal returns (Jarque-Bera)."""
function jarque_bera_test(residuals::AbstractVector{T}) where T<:Real
    n = length(residuals)
    mu = mean(residuals)
    s = std(residuals)
    if s < T(1e-16)
        return zero(T), one(T)
    end
    centered = (residuals .- mu) ./ s
    skew = sum(centered .^ 3) / n
    kurt = sum(centered .^ 4) / n - T(3)
    jb = T(n) / T(6) * (skew^2 + kurt^2 / T(4))
    # Chi-squared(2) p-value approximation
    p = exp(-jb / 2)
    return jb, p
end

"""Power analysis: minimum detectable abnormal return."""
function power_analysis(sigma::T, n_events::Int;
                        significance::T=T(0.05),
                        power::T=T(0.80),
                        event_window::Int=11) where T<:Real
    z_alpha = _normal_quantile(one(T) - significance / 2)
    z_beta = _normal_quantile(power)
    se = sigma * sqrt(T(event_window)) / sqrt(T(n_events))
    min_car = (z_alpha + z_beta) * se
    return min_car
end

"""Sample size calculation for desired power."""
function required_sample_size(target_car::T, sigma::T;
                               significance::T=T(0.05),
                               power::T=T(0.80),
                               event_window::Int=11) where T<:Real
    z_alpha = _normal_quantile(one(T) - significance / 2)
    z_beta = _normal_quantile(power)
    n = ((z_alpha + z_beta) * sigma * sqrt(T(event_window)) / target_car)^2
    return ceil(Int, n)
end

# ─────────────────────────────────────────────────────────────────────────────
# §15  Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

"""Run complete event study pipeline."""
function run_event_study(firm_returns::AbstractMatrix{T},
                         market_returns::AbstractVector{T},
                         event_dates::AbstractVector{Int};
                         config::EventStudyConfig=EventStudyConfig()) where T<:Real
    n_events = length(event_dates)
    n_firms = size(firm_returns, 2)
    results = Vector{EventResult{T}}()
    for (k, date) in enumerate(event_dates)
        firm_col = min(k, n_firms)
        r = compute_abnormal_returns(firm_returns[:, firm_col], market_returns,
                                      date, config.window; firm_id=k)
        if r !== nothing
            push!(results, r)
        end
    end
    # Aggregate statistics
    if isempty(results)
        return Dict{Symbol, Any}(:n_events => 0)
    end
    cars = [r.car for r in results]
    summary = Dict{Symbol, Any}()
    summary[:n_events] = length(results)
    summary[:mean_car] = mean(cars)
    summary[:median_car] = sort(cars)[div(length(cars), 2) + 1]
    summary[:std_car] = std(cars)
    # Multiple test statistics
    z_patell, p_patell = patell_test(results)
    t_bmp, p_bmp = bmp_test(results)
    z_rank, p_rank = rank_test(results)
    z_sign, p_sign = sign_test(results)
    z_gsign, p_gsign = generalized_sign_test(results)
    summary[:patell_z] = z_patell
    summary[:patell_p] = p_patell
    summary[:bmp_t] = t_bmp
    summary[:bmp_p] = p_bmp
    summary[:rank_z] = z_rank
    summary[:rank_p] = p_rank
    summary[:sign_z] = z_sign
    summary[:sign_p] = p_sign
    summary[:gsign_z] = z_gsign
    summary[:gsign_p] = p_gsign
    summary[:results] = results
    return summary
end

"""Generate synthetic event study data for testing."""
function generate_event_data(n_obs::Int, n_events::Int, n_firms::Int;
                              abnormal_return::Float64=0.02,
                              sigma::Float64=0.02,
                              beta::Float64=1.0,
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    market = sigma .* randn(rng, n_obs)
    firm_returns = Matrix{Float64}(undef, n_obs, n_firms)
    for j in 1:n_firms
        alpha = 0.0001 * randn(rng)
        b = beta + 0.3 * randn(rng)
        eps = sigma * 0.8 .* randn(rng, n_obs)
        firm_returns[:, j] = alpha .+ b .* market .+ eps
    end
    # Inject abnormal returns
    event_dates = sort(rand(rng, 100:n_obs-100, n_events))
    for (k, d) in enumerate(event_dates)
        firm = min(k, n_firms)
        firm_returns[d, firm] += abnormal_return
    end
    return firm_returns, market, event_dates
end

# ─────────────────────────────────────────────────────────────────────────────
# §16  Abnormal Volume Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""Compute abnormal volume around events."""
function abnormal_volume(volumes::AbstractVector{T}, event_idx::Int,
                          window::EventWindow) where T<:Real
    n = length(volumes)
    est_start = event_idx + window.estimation_start
    est_end = event_idx + window.estimation_end
    evt_start = event_idx + window.event_start
    evt_end = event_idx + window.event_end
    if est_start < 1 || evt_end > n
        return zeros(T, 0)
    end
    mean_vol = mean(volumes[est_start:est_end])
    std_vol = std(volumes[est_start:est_end])
    evt_len = evt_end - evt_start + 1
    av = Vector{T}(undef, evt_len)
    for i in 1:evt_len
        t = evt_start + i - 1
        av[i] = (volumes[t] - mean_vol) / max(std_vol, T(1e-16))
    end
    av
end

"""Cumulative abnormal volume."""
function cumulative_abnormal_volume(volumes::AbstractMatrix{T},
                                     event_dates::AbstractVector{Int},
                                     window::EventWindow) where T<:Real
    n_events = length(event_dates)
    n_firms = size(volumes, 2)
    evt_len = window.event_end - window.event_start + 1
    cav = zeros(T, evt_len)
    count = 0
    for (k, d) in enumerate(event_dates)
        firm = min(k, n_firms)
        av = abnormal_volume(volumes[:, firm], d, window)
        if !isempty(av)
            cav .+= av
            count += 1
        end
    end
    if count > 0
        cav ./= count
    end
    cumsum(cav)
end

"""Joint test of abnormal returns and abnormal volume."""
function joint_ar_av_test(results::Vector{EventResult{T}},
                           av_series::Vector{Vector{T}}) where T<:Real
    N = min(length(results), length(av_series))
    if N < 2
        return zero(T), one(T)
    end
    cars = [r.car for r in results[1:N]]
    avg_avs = [isempty(av) ? zero(T) : mean(av) for av in av_series[1:N]]
    # Correlation between CAR and abnormal volume
    if std(cars) < T(1e-16) || std(avg_avs) < T(1e-16)
        return zero(T), one(T)
    end
    rho = cor(cars, avg_avs)
    t_stat = rho * sqrt(T(N - 2)) / sqrt(max(one(T) - rho^2, T(1e-16)))
    p = 2 * (one(T) - _normal_cdf(abs(t_stat)))
    return t_stat, p
end

# ─────────────────────────────────────────────────────────────────────────────
# §17  Abnormal Volatility
# ─────────────────────────────────────────────────────────────────────────────

"""Detect abnormal volatility around events."""
function abnormal_volatility(returns::AbstractVector{T}, event_idx::Int,
                              window::EventWindow;
                              vol_window::Int=5) where T<:Real
    n = length(returns)
    est_start = event_idx + window.estimation_start
    est_end = event_idx + window.estimation_end
    evt_start = event_idx + window.event_start
    evt_end = event_idx + window.event_end
    if est_start < 1 || evt_end > n
        return zeros(T, 0)
    end
    # Estimation period realized vol
    est_rets = returns[est_start:est_end]
    n_windows = length(est_rets) - vol_window + 1
    if n_windows < 1
        return zeros(T, 0)
    end
    est_vols = [std(est_rets[i:i+vol_window-1]) for i in 1:n_windows]
    mean_vol = mean(est_vols)
    std_vol = std(est_vols)
    # Event period rolling vol
    evt_rets = returns[max(1, evt_start-vol_window+1):evt_end]
    n_evt_vols = length(evt_rets) - vol_window + 1
    if n_evt_vols < 1
        return zeros(T, 0)
    end
    avol = [(std(evt_rets[i:i+vol_window-1]) - mean_vol) / max(std_vol, T(1e-16))
            for i in 1:n_evt_vols]
    avol
end

# ─────────────────────────────────────────────────────────────────────────────
# §18  Bid-Ask Spread Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""Abnormal bid-ask spread around events."""
function abnormal_spread(spreads::AbstractVector{T}, event_idx::Int,
                          window::EventWindow) where T<:Real
    n = length(spreads)
    est_start = event_idx + window.estimation_start
    est_end = event_idx + window.estimation_end
    evt_start = event_idx + window.event_start
    evt_end = event_idx + window.event_end
    if est_start < 1 || evt_end > n
        return zeros(T, 0)
    end
    mean_spread = mean(spreads[est_start:est_end])
    std_spread = std(spreads[est_start:est_end])
    evt_len = evt_end - evt_start + 1
    as = Vector{T}(undef, evt_len)
    for i in 1:evt_len
        t = evt_start + i - 1
        as[i] = (spreads[t] - mean_spread) / max(std_spread, T(1e-16))
    end
    as
end

# ─────────────────────────────────────────────────────────────────────────────
# §19  Dividend Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""Dividend announcement effect."""
function dividend_event_study(firm_returns::AbstractMatrix{T},
                               market_returns::AbstractVector{T},
                               div_announce_dates::AbstractVector{Int},
                               div_changes::AbstractVector{T};
                               window::EventWindow=EventWindow()) where T<:Real
    n_events = length(div_announce_dates)
    n_firms = size(firm_returns, 2)
    increase_results = EventResult{T}[]
    decrease_results = EventResult{T}[]
    no_change_results = EventResult{T}[]
    for (k, d) in enumerate(div_announce_dates)
        firm = min(k, n_firms)
        result = compute_abnormal_returns(firm_returns[:, firm], market_returns,
                                          d, window; firm_id=k)
        if result === nothing continue end
        if div_changes[k] > T(0.001)
            push!(increase_results, result)
        elseif div_changes[k] < T(-0.001)
            push!(decrease_results, result)
        else
            push!(no_change_results, result)
        end
    end
    summary = Dict{Symbol, Any}()
    if !isempty(increase_results)
        summary[:increase_mean_car] = mean(r.car for r in increase_results)
        summary[:increase_n] = length(increase_results)
        z, p = patell_test(increase_results)
        summary[:increase_z] = z
        summary[:increase_p] = p
    end
    if !isempty(decrease_results)
        summary[:decrease_mean_car] = mean(r.car for r in decrease_results)
        summary[:decrease_n] = length(decrease_results)
        z, p = patell_test(decrease_results)
        summary[:decrease_z] = z
        summary[:decrease_p] = p
    end
    if !isempty(no_change_results)
        summary[:no_change_mean_car] = mean(r.car for r in no_change_results)
        summary[:no_change_n] = length(no_change_results)
    end
    return summary
end

# ─────────────────────────────────────────────────────────────────────────────
# §20  Stock Split Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""Stock split announcement and execution effects."""
function stock_split_study(firm_returns::AbstractMatrix{T},
                           market_returns::AbstractVector{T},
                           announce_dates::AbstractVector{Int},
                           execution_dates::AbstractVector{Int},
                           split_ratios::AbstractVector{T};
                           window::EventWindow=EventWindow()) where T<:Real
    n_events = length(announce_dates)
    n_firms = size(firm_returns, 2)
    announce_results = EventResult{T}[]
    exec_results = EventResult{T}[]
    for (k, (ad, ed)) in enumerate(zip(announce_dates, execution_dates))
        firm = min(k, n_firms)
        ar = compute_abnormal_returns(firm_returns[:, firm], market_returns, ad, window; firm_id=k)
        er = compute_abnormal_returns(firm_returns[:, firm], market_returns, ed, window; firm_id=k + n_events)
        if ar !== nothing push!(announce_results, ar) end
        if er !== nothing push!(exec_results, er) end
    end
    summary = Dict{Symbol, Any}()
    if !isempty(announce_results)
        summary[:announce_mean_car] = mean(r.car for r in announce_results)
        z, p = patell_test(announce_results)
        summary[:announce_z] = z
        summary[:announce_p] = p
    end
    if !isempty(exec_results)
        summary[:exec_mean_car] = mean(r.car for r in exec_results)
        z, p = patell_test(exec_results)
        summary[:exec_z] = z
        summary[:exec_p] = p
    end
    # Split ratio effect
    if !isempty(announce_results) && n_events > 5
        cars = [r.car for r in announce_results]
        ratios = split_ratios[1:min(length(cars), length(split_ratios))]
        if length(cars) == length(ratios) && std(ratios) > T(1e-16)
            summary[:car_ratio_corr] = cor(cars, ratios)
        end
    end
    return summary
end

# ─────────────────────────────────────────────────────────────────────────────
# §21  Regulatory Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""Industry-wide regulatory event study."""
function regulatory_event_study(industry_returns::AbstractMatrix{T},
                                 market_returns::AbstractVector{T},
                                 event_date::Int;
                                 window::EventWindow=EventWindow()) where T<:Real
    n_obs, n_firms = size(industry_returns)
    results = EventResult{T}[]
    for firm in 1:n_firms
        r = compute_abnormal_returns(industry_returns[:, firm], market_returns,
                                      event_date, window; firm_id=firm)
        if r !== nothing
            push!(results, r)
        end
    end
    if isempty(results)
        return Dict{Symbol, Any}(:n_firms => 0)
    end
    cars = [r.car for r in results]
    summary = Dict{Symbol, Any}()
    summary[:n_firms] = length(results)
    summary[:mean_car] = mean(cars)
    summary[:median_car] = sort(cars)[div(length(cars), 2) + 1]
    summary[:pct_positive] = mean(cars .> zero(T))
    z_p, p_p = patell_test(results)
    z_b, p_b = bmp_test(results)
    summary[:patell_z] = z_p
    summary[:patell_p] = p_p
    summary[:bmp_t] = z_b
    summary[:bmp_p] = p_b
    # Wealth effect
    summary[:total_wealth_effect] = sum(cars)
    return summary
end

# ─────────────────────────────────────────────────────────────────────────────
# §22  Short-Selling Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""Short interest announcement effect."""
function short_interest_study(firm_returns::AbstractMatrix{T},
                               market_returns::AbstractVector{T},
                               si_announce_dates::AbstractVector{Int},
                               si_changes::AbstractVector{T};
                               window::EventWindow=EventWindow(evt_start=-2, evt_end=5)) where T<:Real
    n_events = length(si_announce_dates)
    n_firms = size(firm_returns, 2)
    high_si_results = EventResult{T}[]
    low_si_results = EventResult{T}[]
    for (k, d) in enumerate(si_announce_dates)
        firm = min(k, n_firms)
        r = compute_abnormal_returns(firm_returns[:, firm], market_returns,
                                      d, window; firm_id=k)
        if r === nothing continue end
        if si_changes[k] > zero(T)
            push!(high_si_results, r)
        else
            push!(low_si_results, r)
        end
    end
    summary = Dict{Symbol, Any}()
    if !isempty(high_si_results)
        summary[:high_si_mean_car] = mean(r.car for r in high_si_results)
        summary[:high_si_n] = length(high_si_results)
    end
    if !isempty(low_si_results)
        summary[:low_si_mean_car] = mean(r.car for r in low_si_results)
        summary[:low_si_n] = length(low_si_results)
    end
    return summary
end

# ─────────────────────────────────────────────────────────────────────────────
# §23  Insider Trading Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""Insider trading disclosure effect."""
function insider_trading_study(firm_returns::AbstractMatrix{T},
                                market_returns::AbstractVector{T},
                                filing_dates::AbstractVector{Int},
                                trade_types::AbstractVector{Int},  # 1=buy, -1=sell
                                trade_sizes::AbstractVector{T};
                                window::EventWindow=EventWindow(evt_start=-1, evt_end=10)) where T<:Real
    n_events = length(filing_dates)
    n_firms = size(firm_returns, 2)
    buy_results = EventResult{T}[]
    sell_results = EventResult{T}[]
    for (k, d) in enumerate(filing_dates)
        firm = min(k, n_firms)
        r = compute_abnormal_returns(firm_returns[:, firm], market_returns,
                                      d, window; firm_id=k)
        if r === nothing continue end
        if trade_types[k] > 0
            push!(buy_results, r)
        else
            push!(sell_results, r)
        end
    end
    summary = Dict{Symbol, Any}()
    if !isempty(buy_results)
        summary[:buy_mean_car] = mean(r.car for r in buy_results)
        summary[:buy_n] = length(buy_results)
        z, p = patell_test(buy_results)
        summary[:buy_z] = z
        summary[:buy_p] = p
    end
    if !isempty(sell_results)
        summary[:sell_mean_car] = mean(r.car for r in sell_results)
        summary[:sell_n] = length(sell_results)
        z, p = patell_test(sell_results)
        summary[:sell_z] = z
        summary[:sell_p] = p
    end
    return summary
end

# ─────────────────────────────────────────────────────────────────────────────
# §24  Credit Rating Change Event Study
# ─────────────────────────────────────────────────────────────────────────────

"""Credit rating change effect on equity and bonds."""
function rating_change_study(firm_returns::AbstractMatrix{T},
                              market_returns::AbstractVector{T},
                              change_dates::AbstractVector{Int},
                              rating_changes::AbstractVector{Int};  # +1=upgrade, -1=downgrade
                              window::EventWindow=EventWindow()) where T<:Real
    n_events = length(change_dates)
    n_firms = size(firm_returns, 2)
    upgrade_results = EventResult{T}[]
    downgrade_results = EventResult{T}[]
    for (k, d) in enumerate(change_dates)
        firm = min(k, n_firms)
        r = compute_abnormal_returns(firm_returns[:, firm], market_returns,
                                      d, window; firm_id=k)
        if r === nothing continue end
        if rating_changes[k] > 0
            push!(upgrade_results, r)
        else
            push!(downgrade_results, r)
        end
    end
    summary = Dict{Symbol, Any}()
    if !isempty(upgrade_results)
        summary[:upgrade_mean_car] = mean(r.car for r in upgrade_results)
        summary[:upgrade_n] = length(upgrade_results)
        z, p = patell_test(upgrade_results)
        summary[:upgrade_z] = z
        summary[:upgrade_p] = p
    end
    if !isempty(downgrade_results)
        summary[:downgrade_mean_car] = mean(r.car for r in downgrade_results)
        summary[:downgrade_n] = length(downgrade_results)
        z, p = patell_test(downgrade_results)
        summary[:downgrade_z] = z
        summary[:downgrade_p] = p
    end
    # Asymmetry: downgrades typically have larger effect
    if !isempty(upgrade_results) && !isempty(downgrade_results)
        summary[:asymmetry] = abs(mean(r.car for r in downgrade_results)) -
                              abs(mean(r.car for r in upgrade_results))
    end
    return summary
end

# ─────────────────────────────────────────────────────────────────────────────
# §25  Event Window Optimization
# ─────────────────────────────────────────────────────────────────────────────

"""Find optimal event window that maximizes test power."""
function optimal_event_window(firm_returns::AbstractMatrix{T},
                               market_returns::AbstractVector{T},
                               event_dates::AbstractVector{Int};
                               min_window::Int=1, max_window::Int=20,
                               est_start::Int=-270, est_end::Int=-21) where T<:Real
    best_z = zero(T)
    best_window = (0, 0)
    for w_pre in 0:max_window
        for w_post in 0:max_window
            if w_pre + w_post < min_window
                continue
            end
            win = EventWindow(est_start=est_start, est_end=est_end,
                             evt_start=-w_pre, evt_end=w_post)
            results = EventResult{T}[]
            for (k, d) in enumerate(event_dates)
                firm = min(k, size(firm_returns, 2))
                r = compute_abnormal_returns(firm_returns[:, firm], market_returns,
                                              d, win; firm_id=k)
                if r !== nothing
                    push!(results, r)
                end
            end
            if length(results) >= 5
                z, _ = patell_test(results)
                if abs(z) > abs(best_z)
                    best_z = z
                    best_window = (-w_pre, w_post)
                end
            end
        end
    end
    return best_window, best_z
end

"""Robustness check: sensitivity to window specification."""
function window_sensitivity(firm_returns::AbstractMatrix{T},
                            market_returns::AbstractVector{T},
                            event_dates::AbstractVector{Int};
                            windows::Vector{Tuple{Int,Int}}=[(-1,1),(-2,2),(-5,5),(-10,10),(-20,20)],
                            est_start::Int=-270, est_end::Int=-21) where T<:Real
    results_dict = Dict{Tuple{Int,Int}, Dict{Symbol, T}}()
    for (ws, we) in windows
        win = EventWindow(est_start=est_start, est_end=est_end,
                         evt_start=ws, evt_end=we)
        results = EventResult{T}[]
        for (k, d) in enumerate(event_dates)
            firm = min(k, size(firm_returns, 2))
            r = compute_abnormal_returns(firm_returns[:, firm], market_returns,
                                          d, win; firm_id=k)
            if r !== nothing push!(results, r) end
        end
        if !isempty(results)
            cars = [r.car for r in results]
            z_p, p_p = patell_test(results)
            z_b, p_b = bmp_test(results)
            results_dict[(ws, we)] = Dict{Symbol, T}(
                :mean_car => mean(cars),
                :patell_z => z_p,
                :patell_p => p_p,
                :bmp_t => z_b,
                :bmp_p => p_b,
                :n_events => T(length(results))
            )
        end
    end
    results_dict
end

# ─────────────────────────────────────────────────────────────────────────────
# §26  Bootstrap Event Study Inference
# ─────────────────────────────────────────────────────────────────────────────

"""Bootstrap confidence intervals for mean CAR."""
function bootstrap_car_ci(results::Vector{EventResult{T}};
                          n_bootstrap::Int=5000,
                          confidence::T=T(0.95),
                          rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    N = length(results)
    if N == 0
        return (mean=zero(T), ci_lower=zero(T), ci_upper=zero(T))
    end
    cars = [r.car for r in results]
    boot_means = Vector{T}(undef, n_bootstrap)
    for b in 1:n_bootstrap
        idx = rand(rng, 1:N, N)
        boot_means[b] = mean(cars[idx])
    end
    sorted = sort(boot_means)
    alpha = (one(T) - confidence) / 2
    lo = sorted[max(1, round(Int, alpha * n_bootstrap))]
    hi = sorted[min(n_bootstrap, round(Int, (one(T) - alpha) * n_bootstrap))]
    return (mean=mean(cars), ci_lower=lo, ci_upper=hi, se=std(boot_means))
end

"""Wild bootstrap for robust inference under heteroskedasticity."""
function wild_bootstrap_test(results::Vector{EventResult{T}};
                              n_bootstrap::Int=5000,
                              rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    N = length(results)
    if N == 0
        return zero(T), one(T)
    end
    cars = [r.car for r in results]
    observed_mean = mean(cars)
    observed_t = observed_mean / (std(cars) / sqrt(T(N)))
    boot_t_stats = Vector{T}(undef, n_bootstrap)
    for b in 1:n_bootstrap
        # Rademacher weights
        weights = [rand(rng) < 0.5 ? one(T) : -one(T) for _ in 1:N]
        boot_cars = cars .* weights
        boot_mean = mean(boot_cars)
        boot_se = std(boot_cars) / sqrt(T(N))
        boot_t_stats[b] = boot_mean / max(boot_se, T(1e-16))
    end
    p_value = mean(abs.(boot_t_stats) .>= abs(observed_t))
    return observed_t, p_value
end

end # module EventStudy
