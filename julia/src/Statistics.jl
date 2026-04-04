"""
SRFMStats — Full statistical toolkit for quantitative strategy research.

Covers: performance metrics, hypothesis testing, distribution fitting,
correlation analysis, and time-series statistics.
"""
module SRFMStats

using Statistics
using LinearAlgebra
using Distributions
using StatsBase
using Optim
using Random

export sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, cagr
export profit_factor, information_ratio, ulcer_index, omega_ratio, tail_ratio
export t_test_strategy, bootstrap_sharpe_ci, walk_forward_t_test
export fit_normal, fit_student_t, fit_skewed_normal
export kolmogorov_smirnov_test, jarque_bera_test, anderson_darling_test
export rolling_correlation, rank_ic, icir, partial_correlation
export rolling_sharpe, rolling_beta, rolling_alpha
export hurst_exponent, autocorrelation_test, variance_ratio_test
export returns_statistics_report

# ─────────────────────────────────────────────────────────────────────────────
# 1. Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

"""
    sharpe_ratio(returns; rf=0.0, annualize=252) → Float64

Annualised Sharpe ratio: (mean - rf) / std * sqrt(annualize).
Returns 0 if std ≈ 0.
"""
function sharpe_ratio(returns::Vector{Float64};
                       rf::Float64=0.0,
                       annualize::Int=252)::Float64
    n = length(returns)
    n < 2 && return 0.0
    excess = returns .- rf / annualize
    m = mean(excess)
    s = std(excess)
    s < 1e-10 && return 0.0
    return m / s * sqrt(annualize)
end

"""
    sortino_ratio(returns; target=0.0, annualize=252) → Float64

Sortino ratio using downside deviation below target.
"""
function sortino_ratio(returns::Vector{Float64};
                        target::Float64=0.0,
                        annualize::Int=252)::Float64
    n = length(returns)
    n < 2 && return 0.0
    excess  = returns .- target / annualize
    dd_sq   = [r^2 for r in excess if r < 0]
    downside_std = isempty(dd_sq) ? 1e-10 : sqrt(mean(dd_sq))
    downside_std < 1e-10 && return Inf
    return mean(returns) / downside_std * sqrt(annualize)
end

"""
    calmar_ratio(returns) → Float64

CAGR / |max_drawdown|. Annualises assuming 252-bar year.
"""
function calmar_ratio(returns::Vector{Float64})::Float64
    equity = cumprod(1 .+ returns)
    dd_val, _, _ = max_drawdown(equity)
    dd_val < 1e-10 && return Inf
    c = cagr(equity)
    return c / dd_val
end

"""
    max_drawdown(equity) → Tuple{Float64, Int, Int}

Returns (max_drawdown_fraction, peak_index, trough_index).
"""
function max_drawdown(equity::Vector{Float64})::Tuple{Float64, Int, Int}
    n = length(equity)
    n < 2 && return (0.0, 1, 1)

    max_dd     = 0.0
    peak_idx   = 1
    trough_idx = 1
    curr_peak  = equity[1]
    curr_peak_idx = 1

    for i in 2:n
        if equity[i] > curr_peak
            curr_peak = equity[i]
            curr_peak_idx = i
        end
        dd = (curr_peak - equity[i]) / curr_peak
        if dd > max_dd
            max_dd     = dd
            peak_idx   = curr_peak_idx
            trough_idx = i
        end
    end
    return (max_dd, peak_idx, trough_idx)
end

"""
    drawdown_series(equity) → Vector{Float64}

Fraction underwater at each point: (peak_so_far - equity) / peak_so_far.
"""
function drawdown_series(equity::Vector{Float64})::Vector{Float64}
    n = length(equity)
    dd = zeros(Float64, n)
    peak = equity[1]
    for i in 2:n
        peak = max(peak, equity[i])
        dd[i] = (peak - equity[i]) / peak
    end
    return dd
end

"""
    cagr(equity, n_bars_per_year=252) → Float64

Compound Annual Growth Rate from equity curve.
"""
function cagr(equity::Vector{Float64}, n_bars_per_year::Int=252)::Float64
    n = length(equity)
    n < 2 && return 0.0
    years = (n - 1) / n_bars_per_year
    years < 1e-8 && return 0.0
    return (equity[end] / equity[1])^(1 / years) - 1
end

"""
    profit_factor(returns) → Float64

Gross profit / gross loss. Inf if no losing trades.
"""
function profit_factor(returns::Vector{Float64})::Float64
    gross_profit = sum(r for r in returns if r > 0; init=0.0)
    gross_loss   = sum(abs(r) for r in returns if r < 0; init=0.0)
    gross_loss < 1e-10 && return Inf
    return gross_profit / gross_loss
end

"""
    information_ratio(portfolio_rets, benchmark_rets) → Float64

IR = mean(active_return) / std(active_return) * sqrt(252).
"""
function information_ratio(portfolio_rets::Vector{Float64},
                             benchmark_rets::Vector{Float64})::Float64
    n = min(length(portfolio_rets), length(benchmark_rets))
    n < 2 && return 0.0
    active = portfolio_rets[1:n] .- benchmark_rets[1:n]
    return sharpe_ratio(active)
end

"""
    ulcer_index(equity) → Float64

RMS of drawdowns: sqrt(mean(dd²)).
"""
function ulcer_index(equity::Vector{Float64})::Float64
    dd = drawdown_series(equity)
    return sqrt(mean(dd.^2))
end

"""
    omega_ratio(returns; threshold=0.0) → Float64

∫_{threshold}^∞ (1-F(r)) dr / ∫_{-∞}^{threshold} F(r) dr,
approximated by discrete sums.
"""
function omega_ratio(returns::Vector{Float64}; threshold::Float64=0.0)::Float64
    sorted_r = sort(returns)
    n = length(sorted_r)
    numerator   = sum(max(r - threshold, 0.0) for r in sorted_r)
    denominator = sum(max(threshold - r, 0.0) for r in sorted_r)
    denominator < 1e-10 && return Inf
    return numerator / denominator
end

"""
    tail_ratio(returns; q=0.95) → Float64

|q-th percentile| / |1-q-th percentile|.
"""
function tail_ratio(returns::Vector{Float64}; q::Float64=0.95)::Float64
    sorted = sort(returns)
    n = length(sorted)
    hi_idx = clamp(round(Int, q * n), 1, n)
    lo_idx = clamp(round(Int, (1 - q) * n), 1, n)
    hi = abs(sorted[hi_idx])
    lo = abs(sorted[lo_idx])
    lo < 1e-10 && return Inf
    return hi / lo
end

"""
    win_rate(returns) → Float64

Fraction of positive returns.
"""
function win_rate(returns::Vector{Float64})::Float64
    isempty(returns) && return 0.0
    return count(r -> r > 0, returns) / length(returns)
end

"""
    avg_win_loss_ratio(returns) → Float64

Mean winning return / mean absolute losing return.
"""
function avg_win_loss_ratio(returns::Vector{Float64})::Float64
    wins   = filter(r -> r > 0, returns)
    losses = filter(r -> r < 0, returns)
    isempty(wins) && return 0.0
    isempty(losses) && return Inf
    return mean(wins) / mean(abs.(losses))
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling Statistics
# ─────────────────────────────────────────────────────────────────────────────

"""
    rolling_sharpe(returns, window; annualize=252) → Vector{Float64}
"""
function rolling_sharpe(returns::Vector{Float64}, window::Int;
                          annualize::Int=252)::Vector{Float64}
    n = length(returns)
    out = fill(NaN, n)
    for i in window:n
        w = returns[i-window+1:i]
        out[i] = sharpe_ratio(w; annualize=annualize)
    end
    return out
end

"""
    rolling_beta(asset_rets, bench_rets, window) → Vector{Float64}
"""
function rolling_beta(asset_rets::Vector{Float64}, bench_rets::Vector{Float64},
                       window::Int)::Vector{Float64}
    n = min(length(asset_rets), length(bench_rets))
    out = fill(NaN, n)
    for i in window:n
        a = asset_rets[i-window+1:i]
        b = bench_rets[i-window+1:i]
        cov_ab = cov(a, b)
        var_b  = var(b)
        out[i] = var_b < 1e-12 ? 0.0 : cov_ab / var_b
    end
    return out
end

"""
    rolling_alpha(asset_rets, bench_rets, window; annualize=252) → Vector{Float64}
"""
function rolling_alpha(asset_rets::Vector{Float64}, bench_rets::Vector{Float64},
                        window::Int; annualize::Int=252)::Vector{Float64}
    betas  = rolling_beta(asset_rets, bench_rets, window)
    n = length(betas)
    out = fill(NaN, n)
    for i in window:n
        a = asset_rets[i-window+1:i]
        b = bench_rets[i-window+1:i]
        out[i] = (mean(a) - betas[i] * mean(b)) * annualize
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Hypothesis Testing
# ─────────────────────────────────────────────────────────────────────────────

"""
    t_test_strategy(returns) → NamedTuple

One-sample t-test: H₀: mean = 0. Returns t-stat, p-value, 95% CI.
"""
function t_test_strategy(returns::Vector{Float64})::NamedTuple
    n = length(returns)
    n < 2 && return (t_stat=NaN, p_value=NaN, ci_lo=NaN, ci_hi=NaN, reject_h0=false)

    m   = mean(returns)
    s   = std(returns)
    se  = s / sqrt(n)
    t   = m / max(se, 1e-10)
    dof = n - 1

    dist   = TDist(dof)
    p_val  = 2 * (1 - cdf(dist, abs(t)))
    t_crit = quantile(dist, 0.975)
    ci_lo  = m - t_crit * se
    ci_hi  = m + t_crit * se

    return (t_stat=t, p_value=p_val, ci_lo=ci_lo, ci_hi=ci_hi,
            reject_h0=p_val < 0.05, n=n, mean=m, std=s)
end

"""
    bootstrap_sharpe_ci(returns; n_boot=10000, alpha=0.05) → Tuple{Float64, Float64}

Bootstrap confidence interval for Sharpe ratio.
"""
function bootstrap_sharpe_ci(returns::Vector{Float64};
                               n_boot::Int=10000,
                               alpha::Float64=0.05,
                               rng::AbstractRNG=Random.default_rng())::Tuple{Float64, Float64}
    n = length(returns)
    boot_sharpes = Vector{Float64}(undef, n_boot)

    for b in 1:n_boot
        sample_idx = rand(rng, 1:n, n)
        boot_sharpes[b] = sharpe_ratio(returns[sample_idx])
    end

    sort!(boot_sharpes)
    lo_idx = max(1,    round(Int, (alpha / 2) * n_boot))
    hi_idx = min(n_boot, round(Int, (1 - alpha / 2) * n_boot))
    return (boot_sharpes[lo_idx], boot_sharpes[hi_idx])
end

"""
    walk_forward_t_test(in_sample_returns, out_sample_returns) → NamedTuple

Compare IS vs OOS performance: t-test for equality of means, and
Sharpe degradation ratio.
"""
function walk_forward_t_test(in_sample_returns::Vector{Float64},
                               out_sample_returns::Vector{Float64})::NamedTuple
    is_sharpe  = sharpe_ratio(in_sample_returns)
    oos_sharpe = sharpe_ratio(out_sample_returns)
    degrade    = oos_sharpe / max(abs(is_sharpe), 1e-10)

    # Welch's t-test
    n1 = length(in_sample_returns);  n2 = length(out_sample_returns)
    m1 = mean(in_sample_returns);    m2 = mean(out_sample_returns)
    s1 = std(in_sample_returns);     s2 = std(out_sample_returns)

    se = sqrt(s1^2 / n1 + s2^2 / n2)
    t  = (m1 - m2) / max(se, 1e-12)

    # Welch-Satterthwaite DOF
    num = (s1^2 / n1 + s2^2 / n2)^2
    den = (s1^2 / n1)^2 / (n1 - 1) + (s2^2 / n2)^2 / (n2 - 1)
    dof = max(1, num / max(den, 1e-12))

    dist  = TDist(dof)
    p_val = 2 * (1 - cdf(dist, abs(t)))

    return (is_sharpe=is_sharpe, oos_sharpe=oos_sharpe,
            degradation_ratio=degrade, t_stat=t, p_value=p_val,
            significant_difference=p_val < 0.05)
end

"""
    newey_west_se(returns, lags) → Float64

Newey-West HAC standard error for the mean, accounting for autocorrelation.
"""
function newey_west_se(returns::Vector{Float64}, lags::Int)::Float64
    n = length(returns)
    m = mean(returns)
    e = returns .- m

    # Variance term
    V = var(returns)

    # Covariance terms with Bartlett kernel
    for k in 1:lags
        w = 1 - k / (lags + 1)
        cov_k = dot(e[1:n-k], e[k+1:n]) / (n - k)
        V += 2 * w * cov_k
    end

    return sqrt(max(V, 0.0) / n)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Distribution Fitting
# ─────────────────────────────────────────────────────────────────────────────

"""
    fit_normal(data) → Tuple{Float64, Float64}

Return (μ, σ) MLE estimates.
"""
function fit_normal(data::Vector{Float64})::Tuple{Float64, Float64}
    return (mean(data), std(data))
end

"""
    fit_student_t(data) → Tuple{Float64, Float64, Float64}

Return (μ, σ, ν) via MLE for Student-t using Optim.jl.
"""
function fit_student_t(data::Vector{Float64})::Tuple{Float64, Float64, Float64}
    n = length(data)
    m0 = mean(data); s0 = std(data)

    function neg_ll(params)
        mu, log_sigma, log_nu = params
        sigma = exp(log_sigma)
        nu    = exp(log_nu) + 2   # ensure nu > 2
        d     = LocationScale(mu, sigma, TDist(nu))
        ll    = sum(logpdf(d, x) for x in data)
        return -ll
    end

    x0 = [m0, log(s0), log(5.0)]
    result = optimize(neg_ll, x0, BFGS(), Optim.Options(iterations=2000))
    p = Optim.minimizer(result)

    mu    = p[1]
    sigma = exp(p[2])
    nu    = exp(p[3]) + 2
    return (mu, sigma, nu)
end

"""
    fit_skewed_normal(data) → Tuple{Float64, Float64, Float64}

Return (ξ location, ω scale, α skewness) for the skew-normal distribution.
"""
function fit_skewed_normal(data::Vector{Float64})::Tuple{Float64, Float64, Float64}
    m0 = mean(data); s0 = std(data)
    sk = mean(((data .- m0) ./ s0).^3)

    # Method of moments starting point for α
    alpha_init = sign(sk) * min(abs(sk) * 0.8, 0.98) / sqrt(1 - (abs(sk) * 0.8)^(2/3))

    function neg_ll(params)
        xi, log_omega, alpha = params
        omega = exp(log_omega)
        d = SkewNormal(xi, omega, alpha)
        ll = sum(logpdf(d, x) for x in data)
        return -ll
    end

    x0 = [m0, log(s0), alpha_init]
    result = optimize(neg_ll, x0, BFGS(), Optim.Options(iterations=2000))
    p = Optim.minimizer(result)
    return (p[1], exp(p[2]), p[3])
end

"""
    kolmogorov_smirnov_test(data, dist) → NamedTuple

Two-sample KS test against a theoretical distribution.
Returns D statistic, p-value, and rejection at 5%.
"""
function kolmogorov_smirnov_test(data::Vector{Float64},
                                   dist::Distribution)::NamedTuple
    n      = length(data)
    sorted = sort(data)
    D      = 0.0

    for (i, x) in enumerate(sorted)
        F_emp  = i / n
        F_the  = cdf(dist, x)
        D = max(D, abs(F_emp - F_the), abs((i-1)/n - F_the))
    end

    # Kolmogorov distribution approximation
    lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D
    p_val  = kolmogorov_p(lambda)

    return (statistic=D, p_value=p_val, reject_h0=p_val < 0.05, n=n)
end

function kolmogorov_p(lambda::Float64)::Float64
    # Series expansion of P(K > λ)
    sum_val = 0.0
    for j in 1:100
        term = (-1)^(j-1) * exp(-2 * j^2 * lambda^2)
        sum_val += term
        abs(term) < 1e-12 && break
    end
    return clamp(2 * sum_val, 0.0, 1.0)
end

"""
    jarque_bera_test(data) → NamedTuple

Tests normality via skewness and excess kurtosis.
JB = n/6 * (S² + K²/4) ~ χ²(2) under H₀.
"""
function jarque_bera_test(data::Vector{Float64})::NamedTuple
    n = length(data)
    m = mean(data); s = std(data)
    S = mean(((data .- m) ./ s).^3)
    K = mean(((data .- m) ./ s).^4) - 3

    JB    = n / 6 * (S^2 + K^2 / 4)
    p_val = 1 - cdf(Chisq(2), JB)

    return (statistic=JB, p_value=p_val, skewness=S, excess_kurtosis=K,
            reject_normality=p_val < 0.05)
end

"""
    anderson_darling_test(data) → NamedTuple

Anderson-Darling test for normality.
"""
function anderson_darling_test(data::Vector{Float64})::NamedTuple
    n = length(data)
    m = mean(data); s = std(data)
    z = sort((data .- m) ./ s)

    A2 = -n - sum((2i - 1) / n * (log(cdf(Normal(), z[i])) +
                                    log(1 - cdf(Normal(), z[n+1-i])))
                   for i in 1:n)

    # Adjusted statistic
    A2_star = A2 * (1 + 4/n - 25/n^2)

    # Critical values at 5%
    reject = A2_star > 0.787

    # Approximate p-value
    p_val = if A2_star >= 0.6
        exp(1.2937 - 5.709 * A2_star + 0.0186 * A2_star^2)
    elseif A2_star >= 0.34
        exp(0.9177 - 4.279 * A2_star - 1.38 * A2_star^2)
    else
        1 - exp(-13.436 + 101.14 * A2_star - 223.73 * A2_star^2)
    end

    return (statistic=A2, adjusted_statistic=A2_star,
            p_value=clamp(p_val, 0.0, 1.0), reject_normality=reject)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    rolling_correlation(a, b, window) → Vector{Float64}

Pearson correlation over rolling window. NaN for first window-1 bars.
"""
function rolling_correlation(a::Vector{Float64}, b::Vector{Float64},
                               window::Int)::Vector{Float64}
    n = min(length(a), length(b))
    out = fill(NaN, n)
    for i in window:n
        wa = a[i-window+1:i]
        wb = b[i-window+1:i]
        sa = std(wa); sb = std(wb)
        if sa < 1e-10 || sb < 1e-10
            out[i] = NaN
        else
            out[i] = cov(wa, wb) / (sa * sb)
        end
    end
    return out
end

"""
    rank_ic(predictions, returns) → Float64

Rank Information Coefficient: Spearman correlation between predictions and returns.
"""
function rank_ic(predictions::Vector{Float64}, returns::Vector{Float64})::Float64
    n = min(length(predictions), length(returns))
    p = predictions[1:n]; r = returns[1:n]

    rp = ordinalrank(p); rr = ordinalrank(r)
    return cor(Float64.(rp), Float64.(rr))
end

"""
    icir(ic_series) → Float64

IC Information Ratio: mean(IC) / std(IC). Annualised by sqrt(252).
"""
function icir(ic_series::Vector{Float64})::Float64
    s = std(ic_series)
    s < 1e-10 && return 0.0
    return mean(ic_series) / s * sqrt(252)
end

"""
    partial_correlation(X) → Matrix{Float64}

Compute partial correlation matrix from data matrix X (n_obs × n_vars).
Via precision matrix: P = -D^{-1/2} Σ^{-1} D^{-1/2} (off-diagonal).
"""
function partial_correlation(X::Matrix{Float64})::Matrix{Float64}
    Σ = cov(X)
    n = size(Σ, 1)

    # Regularise for invertibility
    Σ_reg = Σ + 1e-8 * I(n)

    try
        Θ = inv(Σ_reg)   # precision matrix
        D_inv_sqrt = Diagonal(1.0 ./ sqrt.(diag(Θ)))
        P = -D_inv_sqrt * Θ * D_inv_sqrt
        # Set diagonal to 1
        for i in 1:n; P[i, i] = 1.0; end
        return P
    catch
        # Fallback: return regular correlation
        return cor(X)
    end
end

"""
    spearman_correlation(X) → Matrix{Float64}

Rank correlation matrix for n_vars × n_obs matrix.
"""
function spearman_correlation(X::Matrix{Float64})::Matrix{Float64}
    n, p = size(X)
    R = Matrix{Float64}(undef, p, p)
    for i in 1:p, j in 1:p
        R[i,j] = cor(Float64.(ordinalrank(X[:,i])),
                     Float64.(ordinalrank(X[:,j])))
    end
    return R
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Time-Series Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

"""
    hurst_exponent(series; max_lag=100) → Float64

Estimate Hurst exponent via R/S analysis.
H ≈ 0.5: random walk; H > 0.5: trending; H < 0.5: mean-reverting.
"""
function hurst_exponent(series::Vector{Float64}; max_lag::Int=100)::Float64
    n = length(series)
    lags = unique(round.(Int, exp.(range(log(10), log(min(n÷2, max_lag)), length=20))))
    filter!(l -> l >= 2, lags)

    log_lags = Float64[]
    log_rs   = Float64[]

    for lag in lags
        n_chunks = n ÷ lag
        n_chunks < 1 && continue
        rs_vals = Float64[]
        for k in 1:n_chunks
            chunk = series[(k-1)*lag+1 : k*lag]
            m     = mean(chunk)
            dev   = cumsum(chunk .- m)
            R     = maximum(dev) - minimum(dev)
            S     = std(chunk)
            S < 1e-12 && continue
            push!(rs_vals, R / S)
        end
        isempty(rs_vals) && continue
        push!(log_lags, log(lag))
        push!(log_rs,   log(mean(rs_vals)))
    end

    length(log_lags) < 2 && return 0.5

    # OLS slope
    mx = mean(log_lags); my = mean(log_rs)
    h  = sum((log_lags .- mx) .* (log_rs .- my)) /
         sum((log_lags .- mx).^2)
    return clamp(h, 0.0, 1.0)
end

"""
    autocorrelation_test(returns, max_lag=20) → NamedTuple

Ljung-Box test for autocorrelation in returns.
H₀: no autocorrelation up to max_lag.
"""
function autocorrelation_test(returns::Vector{Float64};
                                max_lag::Int=20)::NamedTuple
    n = length(returns)
    acf_vals = Float64[]
    for k in 1:max_lag
        rho_k = cor(returns[1:n-k], returns[k+1:n])
        push!(acf_vals, rho_k)
    end

    # Ljung-Box statistic
    Q = n * (n + 2) * sum(acf_vals[k]^2 / (n - k) for k in 1:max_lag)
    p_val = 1 - cdf(Chisq(max_lag), Q)

    return (Q_stat=Q, p_value=p_val, reject_h0=p_val < 0.05,
            acf=acf_vals, max_lag=max_lag)
end

"""
    variance_ratio_test(prices, lags=[2,5,10,20]) → NamedTuple

Lo-MacKinlay variance ratio test. VR > 1 → positive autocorrelation.
"""
function variance_ratio_test(prices::Vector{Float64};
                               lags::Vector{Int}=[2,5,10,20])::NamedTuple
    returns = diff(log.(prices))
    n = length(returns)
    var1 = var(returns)

    vr_stats = Dict{Int, NamedTuple}()

    for q in lags
        # q-period returns
        q_rets = [sum(returns[i:i+q-1]) for i in 1:n-q+1]
        var_q  = var(q_rets) / q
        VR     = var_q / max(var1, 1e-12)

        # Asymptotic z-statistic (under homoskedasticity)
        z = (VR - 1) / sqrt(2 * (2q - 1) * (q - 1) / (3q * n))
        p_val = 2 * (1 - cdf(Normal(), abs(z)))

        vr_stats[q] = (VR=VR, z_stat=z, p_value=p_val)
    end

    return (variance_ratios=vr_stats, returns_used=length(returns))
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Report Generator
# ─────────────────────────────────────────────────────────────────────────────

"""
    returns_statistics_report(returns; equity=nothing, annualize=252) → NamedTuple

Comprehensive statistics report for a returns series.
"""
function returns_statistics_report(returns::Vector{Float64};
                                    equity::Union{Vector{Float64}, Nothing}=nothing,
                                    annualize::Int=252)::NamedTuple
    if isnothing(equity)
        equity = cumprod(1 .+ returns)
        pushfirst!(equity, 1.0)
    end

    sr   = sharpe_ratio(returns; annualize=annualize)
    so   = sortino_ratio(returns; annualize=annualize)
    cal  = calmar_ratio(returns)
    pf   = profit_factor(returns)
    wr   = win_rate(returns)
    wl   = avg_win_loss_ratio(returns)
    ul   = ulcer_index(equity)
    om   = omega_ratio(returns)
    tr   = tail_ratio(returns)
    c    = cagr(equity, annualize)
    dd_val, pk, tr_idx = max_drawdown(equity)
    jb   = jarque_bera_test(returns)
    ttest= t_test_strategy(returns)
    h    = hurst_exponent(equity)

    return (
        n_returns      = length(returns),
        mean_return    = mean(returns),
        std_return     = std(returns),
        annualised_ret = mean(returns) * annualize,
        annualised_vol = std(returns) * sqrt(annualize),
        cagr           = c,
        sharpe         = sr,
        sortino        = so,
        calmar         = cal,
        max_drawdown   = dd_val,
        max_dd_peak    = pk,
        max_dd_trough  = tr_idx,
        profit_factor  = pf,
        win_rate       = wr,
        avg_wl_ratio   = wl,
        ulcer_index    = ul,
        omega_ratio    = om,
        tail_ratio     = tr,
        hurst          = h,
        skewness       = jb.skewness,
        excess_kurtosis= jb.excess_kurtosis,
        jb_p_value     = jb.p_value,
        is_normal      = !jb.reject_normality,
        t_stat         = ttest.t_stat,
        t_p_value      = ttest.p_value,
        significant    = ttest.reject_h0,
    )
end

end # module SRFMStats
