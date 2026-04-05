"""
robust_statistics.jl — Robust Statistical Methods for Finance

Covers:
  - M-estimators: Huber, Tukey biweight, Andrews wave
  - Robust regression via IRLS (Iteratively Reweighted Least Squares)
  - Minimum Covariance Determinant (MCD) for outlier detection
  - Robust PCA (RPCA): sparse + low-rank decomposition via ADMM
  - Breakdown point analysis
  - Winsorization and trimming with optimal cutoffs
  - Robust Sharpe ratio and robust VaR estimation
  - Influence function analysis
  - Application: robust BH signal detection despite outlier bars

Pure Julia stdlib only. No external dependencies.
"""

module RobustStatistics

using Statistics, LinearAlgebra, Random

export HuberEstimator, TukeyBiweight, AndrewsWave
export m_estimate_location, m_estimate_scale
export robust_regression_irls, huber_regression, bisquare_regression
export mcd_covariance, mcd_mahalanobis, mcd_outlier_detection
export rpca_admm, rpca_decompose
export winsorize, winsorize_optimal, trimmed_mean, trimmed_std
export robust_sharpe, robust_var, robust_cvar
export influence_function, sensitivity_curve
export robust_bh_signal
export run_robust_statistics_demo

# ─────────────────────────────────────────────────────────────
# 1. M-ESTIMATOR LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────

"""
    HuberEstimator

Huber M-estimator: quadratic for |r| ≤ k, linear for |r| > k.
Breakdown point: approaches 50% as k → 0, but loses efficiency.
Standard choice: k = 1.345 (95% efficiency for normal data).
"""
struct HuberEstimator
    k::Float64
end
HuberEstimator() = HuberEstimator(1.345)

"""Huber loss ρ(r)."""
function rho(e::HuberEstimator, r::Float64)::Float64
    abs(r) <= e.k ? 0.5 * r^2 : e.k * abs(r) - 0.5 * e.k^2
end

"""Huber influence function ψ(r) = ρ'(r)."""
function psi(e::HuberEstimator, r::Float64)::Float64
    abs(r) <= e.k ? r : e.k * sign(r)
end

"""Huber weight function w(r) = ψ(r)/r."""
function weight(e::HuberEstimator, r::Float64)::Float64
    abs(r) < 1e-10 ? 1.0 : min(1.0, e.k / abs(r))
end

"""
    TukeyBiweight

Tukey bisquare (biweight) M-estimator.
Fully redescends to zero for |r| > k: complete rejection of outliers.
Standard choice: k = 4.685 (95% efficiency for normal data).
Breakdown point: 50%.
"""
struct TukeyBiweight
    k::Float64
end
TukeyBiweight() = TukeyBiweight(4.685)

function rho(e::TukeyBiweight, r::Float64)::Float64
    if abs(r) > e.k; return e.k^2 / 6; end
    u = r / e.k
    e.k^2 / 6 * (1 - (1 - u^2)^3)
end

function psi(e::TukeyBiweight, r::Float64)::Float64
    abs(r) > e.k ? 0.0 : r * (1 - (r/e.k)^2)^2
end

function weight(e::TukeyBiweight, r::Float64)::Float64
    abs(r) > e.k ? 0.0 : (1 - (r/e.k)^2)^2
end

"""
    AndrewsWave

Andrews' sine wave M-estimator.
ψ(r) = sin(r/k) for |r| ≤ kπ, 0 otherwise.
Standard choice: k = 1.339.
"""
struct AndrewsWave
    k::Float64
end
AndrewsWave() = AndrewsWave(1.339)

function psi(e::AndrewsWave, r::Float64)::Float64
    abs(r) > e.k * π ? 0.0 : sin(r / e.k)
end

function weight(e::AndrewsWave, r::Float64)::Float64
    abs(r) < 1e-10 && return 1.0
    abs(r) > e.k * π ? 0.0 : sin(r / e.k) / r
end

# ─────────────────────────────────────────────────────────────
# 2. M-ESTIMATORS FOR LOCATION AND SCALE
# ─────────────────────────────────────────────────────────────

"""
    m_estimate_location(x, estimator; scale=nothing, tol=1e-6, maxiter=100)
       -> Float64

Robust location estimate using M-estimator via IRLS.
If scale=nothing, uses MAD estimate.
"""
function m_estimate_location(x::Vector{Float64}, estimator;
                               scale::Union{Float64,Nothing}=nothing,
                               tol::Float64=1e-6, maxiter::Int=100)::Float64
    n = length(x)
    # Initial estimate: median
    mu = median(x)
    # Scale estimate: MAD / 0.6745
    s = isnothing(scale) ? median(abs.(x .- mu)) / 0.6745 : scale
    s < 1e-10 && return mu

    for _ in 1:maxiter
        r = (x .- mu) ./ s
        w = [weight(estimator, ri) for ri in r]
        sum_w = sum(w)
        sum_w < 1e-10 && break
        mu_new = sum(w .* x) / sum_w
        abs(mu_new - mu) < tol * s && return mu_new
        mu = mu_new
    end
    mu
end

"""
    m_estimate_scale(x, estimator; mu=nothing, tol=1e-6, maxiter=100)
       -> Float64

Robust scale estimate (M-estimator for variance).
Uses τ-estimator approach for scale.
"""
function m_estimate_scale(x::Vector{Float64}, estimator;
                            mu::Union{Float64,Nothing}=nothing,
                            tol::Float64=1e-6, maxiter::Int=100)::Float64
    n = length(x)
    loc = isnothing(mu) ? median(x) : mu
    s = median(abs.(x .- loc)) / 0.6745
    s < 1e-10 && return s

    # Iterative scale estimation
    for _ in 1:maxiter
        r = (x .- loc) ./ s
        w = [weight(estimator, ri) for ri in r]
        # Normalized: E[w * r^2] = constant (normalized for normal dist)
        s_new = s * sqrt(sum(w .* r.^2) / (n * 0.5))
        abs(s_new - s) < tol * s && return s_new
        s = s_new
    end
    s
end

"""
    mad_scale(x) -> Float64

Median Absolute Deviation (MAD) scale estimate.
Consistent estimator for σ under normality: σ̂ = MAD / 0.6745.
"""
function mad_scale(x::Vector{Float64})::Float64
    med = median(x)
    median(abs.(x .- med)) / 0.6745
end

# ─────────────────────────────────────────────────────────────
# 3. ROBUST REGRESSION (IRLS)
# ─────────────────────────────────────────────────────────────

"""
    robust_regression_irls(X, y, estimator; tol=1e-6, maxiter=50)
       -> (coefficients, weights, residuals)

Robust linear regression via Iteratively Reweighted Least Squares.
X: n × p design matrix, y: n-vector of responses.
"""
function robust_regression_irls(X::Matrix{Float64}, y::Vector{Float64},
                                  estimator;
                                  tol::Float64=1e-6, maxiter::Int=50)
    n, p = size(X)
    # Initial OLS estimate
    beta = (X'X + 1e-8*I) \ (X'y)
    residuals = y .- X * beta
    s = mad_scale(residuals)
    s < 1e-10 && (s = std(residuals))

    for iter in 1:maxiter
        r = residuals ./ (s + 1e-10)
        w = [weight(estimator, ri) for ri in r]
        # Weighted least squares
        W    = Diagonal(w)
        XtWX = X' * W * X + 1e-8*I
        XtWy = X' * W * y
        beta_new = XtWX \ XtWy
        res_new  = y .- X * beta_new
        s        = mad_scale(res_new)

        if norm(beta_new - beta) < tol * (norm(beta) + 1e-10)
            return (coefficients=beta_new, weights=w, residuals=res_new,
                    scale=s, n_iter=iter)
        end
        beta = beta_new; residuals = res_new
    end
    (coefficients=beta, weights=[weight(estimator, (y[i]-dot(X[i,:],beta))/(s+1e-10)) for i in 1:n],
     residuals=residuals, scale=s, n_iter=maxiter)
end

"""
    huber_regression(X, y; k=1.345, ...) -> NamedTuple

Convenience wrapper for Huber M-regression.
"""
huber_regression(X::Matrix{Float64}, y::Vector{Float64}; k::Float64=1.345, kwargs...) =
    robust_regression_irls(X, y, HuberEstimator(k); kwargs...)

"""
    bisquare_regression(X, y; k=4.685, ...) -> NamedTuple

Convenience wrapper for Tukey bisquare M-regression.
"""
bisquare_regression(X::Matrix{Float64}, y::Vector{Float64}; k::Float64=4.685, kwargs...) =
    robust_regression_irls(X, y, TukeyBiweight(k); kwargs...)

"""
    regression_breakdown_point(estimator) -> Float64

Theoretical breakdown point of M-estimator in regression.
"""
function regression_breakdown_point(estimator)::Float64
    if isa(estimator, HuberEstimator)
        return 1.0 / (2 + 2 * estimator.k^2 / pi)  # approximate
    elseif isa(estimator, TukeyBiweight)
        return 0.5  # 50% breakdown point
    else
        return 0.5
    end
end

# ─────────────────────────────────────────────────────────────
# 4. MINIMUM COVARIANCE DETERMINANT (MCD)
# ─────────────────────────────────────────────────────────────

"""
    mcd_covariance(X; h=nothing, n_trials=10, rng=MersenneTwister(42))
       -> NamedTuple

Minimum Covariance Determinant estimator.
Finds the subset of h observations with smallest covariance determinant.

Returns:
  - robust_mean: location estimate
  - robust_cov: covariance estimate
  - support: indices of the h inliers
"""
function mcd_covariance(X::Matrix{Float64};
                          h::Union{Int,Nothing}=nothing,
                          n_trials::Int=10,
                          rng=MersenneTwister(42))
    n, p = size(X)
    h_val = isnothing(h) ? Int(ceil(0.75 * n)) : h
    h_val = max(p + 1, min(h_val, n))

    best_det   = Inf
    best_mu    = zeros(p)
    best_Sigma = Matrix{Float64}(I, p, p)
    best_support = collect(1:h_val)

    for trial in 1:n_trials
        # Random initial subset
        support = sort(randperm(rng, n)[1:h_val])

        # C-step iterations
        for _ in 1:20
            subset = X[support, :]
            mu     = vec(mean(subset, dims=1))
            Sigma  = cov(subset) + 1e-8*I

            # Compute Mahalanobis distances for all points
            Sigma_inv = inv(Sigma)
            d2 = [begin
                      diff = X[i,:] - mu
                      dot(diff, Sigma_inv * diff)
                  end for i in 1:n]

            new_support = partialsortperm(d2, 1:h_val)
            sort!(new_support)

            new_support == support && break
            support = new_support
        end

        subset = X[support, :]
        mu     = vec(mean(subset, dims=1))
        Sigma  = cov(subset) + 1e-8*I

        d = det(Sigma)
        if d < best_det
            best_det     = d
            best_mu      = mu
            best_Sigma   = Sigma
            best_support = support
        end
    end

    # Consistency factor (for 75% trimming under normality)
    alpha_q = 1.0 - h_val / n
    c_factor = 1.0 / (1.0 - 2.0 * alpha_q * pdf_normal_quantile(alpha_q))

    (robust_mean=best_mu, robust_cov=best_Sigma * c_factor,
     support=best_support, det=best_det)
end

"""Approximate: c(alpha) for MCD consistency correction."""
function pdf_normal_quantile(alpha::Float64)::Float64
    # chi^2(p) quantile approximation — simplified
    # For p=1: use normal quantile
    q = sqrt(2) * erfinv_approx(2*alpha - 1)
    exp(-0.5 * q^2) / sqrt(2π) * q
end

"""Approximate erfinv via Newton's method."""
function erfinv_approx(y::Float64)::Float64
    y = clamp(y, -0.9999, 0.9999)
    a = 0.147
    ln_term = log(1 - y^2)
    t1 = 2/(π*a) + ln_term/2
    sign(y) * sqrt(sqrt(t1^2 - ln_term/a) - t1)
end

"""
    mcd_mahalanobis(X, mcd_result) -> Vector{Float64}

Robust Mahalanobis distances based on MCD estimates.
"""
function mcd_mahalanobis(X::Matrix{Float64}, mcd)::Vector{Float64}
    n = size(X, 1)
    mu = mcd.robust_mean
    Sigma_inv = inv(mcd.robust_cov)
    [begin
         d = X[i,:] - mu
         sqrt(max(dot(d, Sigma_inv * d), 0.0))
     end for i in 1:n]
end

"""
    mcd_outlier_detection(X; threshold=2.5, ...) -> NamedTuple

Detect multivariate outliers using MCD Mahalanobis distance.
"""
function mcd_outlier_detection(X::Matrix{Float64};
                                 threshold::Float64=2.5,
                                 kwargs...)
    mcd = mcd_covariance(X; kwargs...)
    dist = mcd_mahalanobis(X, mcd)
    outlier_idx = findall(dist .> threshold * sqrt(size(X,2)))
    inlier_idx  = findall(dist .<= threshold * sqrt(size(X,2)))
    (outlier_indices=outlier_idx, inlier_indices=inlier_idx,
     distances=dist, threshold=threshold * sqrt(size(X,2)),
     n_outliers=length(outlier_idx), mcd=mcd)
end

# ─────────────────────────────────────────────────────────────
# 5. ROBUST PCA (RPCA) — ADMM
# ─────────────────────────────────────────────────────────────

"""
    rpca_admm(M; lambda=nothing, mu=nothing, tol=1e-5, maxiter=200)
       -> (L, S, n_iter)

Robust PCA via ADMM (Alternating Direction Method of Multipliers).
Decomposes M = L + S where:
  L is low-rank (underlying structure)
  S is sparse (outliers / anomalies)

Solves: min_{L,S} ||L||_* + λ||S||_1  s.t. L + S = M
Reference: Candès, Li, Ma & Wright (2011)
"""
function rpca_admm(M::Matrix{Float64};
                    lambda::Union{Float64,Nothing}=nothing,
                    mu::Union{Float64,Nothing}=nothing,
                    tol::Float64=1e-5, maxiter::Int=200)
    m, n = size(M)
    lam = isnothing(lambda) ? 1.0 / sqrt(max(m, n)) : lambda
    rho = isnothing(mu)     ? 1.25 / norm(M, 2) : mu

    L = zeros(m, n)
    S = zeros(m, n)
    Y = zeros(m, n)  # dual variable

    for iter in 1:maxiter
        # L-update: singular value thresholding
        Z = M - S + Y / rho
        U, sv, Vt = svd(Z)
        sv_thresh = max.(sv .- 1.0/rho, 0.0)
        L_new = U * Diagonal(sv_thresh) * Vt'

        # S-update: soft thresholding (L1 proximal)
        Z2 = M - L_new + Y / rho
        S_new = sign.(Z2) .* max.(abs.(Z2) .- lam/rho, 0.0)

        # Dual update
        residual = M - L_new - S_new
        Y .+= rho .* residual

        # Check convergence
        primal_res = norm(residual, "fro") / max(norm(M, "fro"), 1.0)
        if primal_res < tol && iter > 5
            return (L=L_new, S=S_new, n_iter=iter, converged=true)
        end

        L, S = L_new, S_new
    end
    (L=L, S=S, n_iter=maxiter, converged=false)
end

"""Frobenius norm."""
norm(A::Matrix, ::String) = sqrt(sum(A.^2))

"""
    rpca_decompose(returns_matrix; lambda=nothing) -> NamedTuple

Apply RPCA to a return matrix to separate clean signal from outlier noise.
Useful for removing flash crashes, data errors, etc.

returns_matrix: T × N matrix of asset returns.
"""
function rpca_decompose(returns_matrix::Matrix{Float64}; lambda=nothing)
    result = rpca_admm(returns_matrix; lambda=lambda, maxiter=100)
    n_sparse = sum(abs.(result.S) .> 1e-10)
    sparse_frac = n_sparse / length(result.S)
    # Low-rank structure: effective rank
    sv = svd(result.L).S
    sv_cum = cumsum(sv) ./ (sum(sv) + 1e-10)
    eff_rank = findfirst(>=(0.90), sv_cum)

    (low_rank=result.L, sparse=result.S,
     sparse_fraction=sparse_frac,
     effective_rank=eff_rank,
     n_outliers=n_sparse,
     converged=result.converged)
end

# ─────────────────────────────────────────────────────────────
# 6. WINSORIZATION AND TRIMMING
# ─────────────────────────────────────────────────────────────

"""
    winsorize(x, lower, upper) -> Vector{Float64}

Winsorize vector at [lower, upper] quantile levels.
"""
function winsorize(x::Vector{Float64}, lower::Float64=0.01,
                    upper::Float64=0.99)::Vector{Float64}
    n = length(x)
    lo = quantile(x, lower)
    hi = quantile(x, upper)
    clamp.(x, lo, hi)
end

"""
    winsorize_optimal(x; criterion=:efficiency) -> Vector{Float64}

Adaptive winsorization with data-driven cutoff selection.
Criterion: :efficiency (maximize efficiency relative to OLS) or :mse.
"""
function winsorize_optimal(x::Vector{Float64};
                             criterion::Symbol=:efficiency)::Vector{Float64}
    n = length(x)
    best_w = copy(x)
    best_score = Inf

    # Try various quantile levels and pick best
    for alpha in [0.01, 0.02, 0.05, 0.10, 0.15]
        w = winsorize(x, alpha, 1-alpha)
        if criterion == :mse
            # Minimize MSE relative to trimmed mean
            score = mad_scale(w)
        else
            # Efficiency: compare winsorized std to trimmed std
            score = std(w) / (mad_scale(x) + 1e-10)
        end
        if score < best_score
            best_score = score
            best_w     = w
        end
    end
    best_w
end

"""
    trimmed_mean(x, alpha) -> Float64

Alpha-trimmed mean: remove top and bottom alpha fraction.
"""
function trimmed_mean(x::Vector{Float64}, alpha::Float64=0.1)::Float64
    n = length(x)
    k = Int(floor(alpha * n))
    sorted = sort(x)
    mean(sorted[k+1:end-k])
end

"""
    trimmed_std(x, alpha) -> Float64

Alpha-trimmed standard deviation.
"""
function trimmed_std(x::Vector{Float64}, alpha::Float64=0.1)::Float64
    n = length(x)
    k = Int(floor(alpha * n))
    sorted = sort(x)
    trimmed = sorted[k+1:end-k]
    std(trimmed) / sqrt(1 - 2*alpha)  # consistency correction
end

# ─────────────────────────────────────────────────────────────
# 7. ROBUST RISK MEASURES
# ─────────────────────────────────────────────────────────────

"""
    robust_sharpe(returns; risk_free=0.0, annualize=252) -> Float64

Robust Sharpe ratio using M-estimator for location and scale.
Uses Huber estimator for robustness against outlier returns.
"""
function robust_sharpe(returns::Vector{Float64};
                        risk_free::Float64=0.0,
                        annualize::Int=252)::Float64
    est = HuberEstimator(1.345)
    excess = returns .- risk_free / annualize
    mu     = m_estimate_location(excess, est)
    sigma  = m_estimate_scale(excess, est; mu=mu)
    sigma < 1e-10 && return 0.0
    mu / sigma * sqrt(annualize)
end

"""
    robust_var(returns, confidence; method=:filtered) -> Float64

Robust VaR estimate. Methods:
  :filtered  — winsorize then standard VaR
  :trimmed   — use trimmed distribution
  :mcd       — use MCD scatter for multivariate (single asset: use scale estimate)
"""
function robust_var(returns::Vector{Float64}, confidence::Float64=0.99;
                     method::Symbol=:filtered)::Float64
    n = length(returns)
    if method == :filtered
        w = winsorize(returns, 0.02, 0.98)
        return -quantile(w, 1 - confidence)
    elseif method == :trimmed
        k = Int(floor(0.05 * n))
        sorted = sort(returns)
        trimmed = sorted[k+1:end-k]
        idx = Int(ceil((1 - confidence) * n))
        return -sorted[max(1, idx)]
    else
        # Parametric with robust scale
        est = HuberEstimator()
        mu  = m_estimate_location(returns, est)
        sig = m_estimate_scale(returns, est; mu=mu)
        # Normal quantile
        z   = -1.6449  # 99% VaR
        return -(mu + z * sig)
    end
end

"""
    robust_cvar(returns, confidence) -> Float64

Robust CVaR (Expected Shortfall) using winsorized returns.
"""
function robust_cvar(returns::Vector{Float64}, confidence::Float64=0.99)::Float64
    n = length(returns)
    var_val = robust_var(returns, confidence; method=:filtered)
    tail    = returns[returns .<= -var_val]
    isempty(tail) && return var_val
    -mean(tail)
end

"""
    robust_beta(asset_returns, market_returns) -> Float64

Robust market beta using Huber regression.
"""
function robust_beta(asset_returns::Vector{Float64},
                      market_returns::Vector{Float64})::Float64
    n = min(length(asset_returns), length(market_returns))
    X = reshape(market_returns[1:n], n, 1)
    y = asset_returns[1:n]
    result = huber_regression(X, y)
    result.coefficients[1]
end

# ─────────────────────────────────────────────────────────────
# 8. INFLUENCE FUNCTION ANALYSIS
# ─────────────────────────────────────────────────────────────

"""
    influence_function(x, estimator_fn, point; h=1e-4) -> Float64

Numerical influence function of an estimator at a data point.
IF(x₀; T, F) = d/dε T(F + ε δ_{x₀}) |_{ε=0}
"""
function influence_function(x::Vector{Float64},
                              estimator_fn::Function,
                              point::Float64;
                              h::Float64=1e-4)::Float64
    t_base = estimator_fn(x)
    # Add contamination: replace one observation with `point`
    x_plus = copy(x)
    n = length(x)
    # Average influence via leave-one-add-one
    ifs = zeros(n)
    for i in 1:n
        x_mod = copy(x); x_mod[i] = point
        t_mod = estimator_fn(x_mod)
        ifs[i] = (t_mod - t_base) / h
    end
    mean(ifs)
end

"""
    sensitivity_curve(x, estimator_fn; n_grid=100, range_mult=3.0) -> NamedTuple

Compute the sensitivity curve SC(x₀) = n * [T_n(x₁,...,xₙ₋₁,x₀) - T_{n-1}(x₁,...,xₙ₋₁)]
over a grid of contamination points x₀.
"""
function sensitivity_curve(x::Vector{Float64},
                             estimator_fn::Function;
                             n_grid::Int=100,
                             range_mult::Float64=3.0)
    n   = length(x)
    mu  = mean(x)
    sig = std(x)
    grid = range(mu - range_mult*sig, mu + range_mult*sig, length=n_grid)

    x_base = x[1:end-1]
    t_base = estimator_fn(x_base)

    sc = map(grid) do x0
        x_mod = vcat(x_base, [x0])
        t_mod = estimator_fn(x_mod)
        (n - 1) * (t_mod - t_base)
    end

    (grid=collect(grid), sensitivity=collect(sc),
     max_influence=maximum(abs.(sc)),
     bounded=maximum(abs.(sc)) < 5 * sig)
end

"""
    cook_distance(X, y, beta, residuals, mse) -> Vector{Float64}

Cook's distance for OLS regression: influence of each observation.
"""
function cook_distance(X::Matrix{Float64}, y::Vector{Float64},
                        beta::Vector{Float64},
                        residuals::Vector{Float64},
                        mse::Float64)::Vector{Float64}
    n, p = size(X)
    XtX_inv = inv(X'X + 1e-8*I)
    h = [X[i,:]' * XtX_inv * X[i,:] for i in 1:n]  # leverage scores
    [h[i] * residuals[i]^2 / ((1 - h[i])^2 * p * mse) for i in 1:n]
end

# ─────────────────────────────────────────────────────────────
# 9. ROBUST BH SIGNAL DETECTION
# ─────────────────────────────────────────────────────────────

"""
    RobustBHSignal

Robust version of a BH (Brownian/Harmonic) trend signal that is insensitive
to price spikes, flash crashes, and microstructure noise.
"""
struct RobustBHSignal
    lookback::Int          # signal lookback window
    outlier_threshold::Float64  # Mahalanobis distance for outlier rejection
    winsorize_alpha::Float64    # winsorization quantile
end

RobustBHSignal(lookback::Int=20) = RobustBHSignal(lookback, 2.5, 0.05)

"""
    robust_bh_signal(prices, sig::RobustBHSignal) -> Vector{Float64}

Compute a robust BH-inspired trend signal from noisy price data.
Steps:
  1. Winsorize log-returns to remove extreme outliers
  2. Apply Huber M-estimator for location
  3. Compute signal as normalized robust mean return
  4. Adjust sign based on robust trend direction
"""
function robust_bh_signal(prices::Vector{Float64},
                            sig::RobustBHSignal)::Vector{Float64}
    n = length(prices)
    returns = [0.0; diff(log.(max.(prices, 1.0)))]
    signal  = zeros(n)
    est     = HuberEstimator(1.345)

    for t in (sig.lookback+1):n
        window = returns[t-sig.lookback+1:t]
        # Step 1: Winsorize
        w_ret  = winsorize(window, sig.winsorize_alpha, 1-sig.winsorize_alpha)
        # Step 2: Robust location
        mu_r   = m_estimate_location(w_ret, est)
        # Step 3: Robust scale for normalization
        sig_r  = m_estimate_scale(w_ret, est; mu=mu_r)
        sig_r  < 1e-10 && continue
        # Step 4: Signal = normalized robust mean (Z-score style)
        z      = mu_r / (sig_r / sqrt(sig.lookback))
        signal[t] = clamp(z, -3.0, 3.0)
    end
    signal
end

"""
    compare_signals(prices; lookback=20) -> NamedTuple

Compare naive momentum signal vs robust BH signal on noisy price data.
"""
function compare_signals(prices::Vector{Float64}; lookback::Int=20)
    returns = [0.0; diff(log.(max.(prices, 1.0)))]
    n = length(returns)
    naive   = zeros(n)
    robust  = zeros(n)
    rb_sig  = RobustBHSignal(lookback)

    for t in (lookback+1):n
        window    = returns[t-lookback+1:t]
        naive[t]  = mean(window) / (std(window) + 1e-10)
    end
    robust = robust_bh_signal(prices, rb_sig)

    # Predictive correlation with future 1-day return
    future_ret = [t < n ? returns[t+1] : 0.0 for t in 1:n]
    valid = (lookback+1):(n-1)
    corr_naive  = length(valid) > 1 ?
                  cor(naive[valid], future_ret[valid])  : 0.0
    corr_robust = length(valid) > 1 ?
                  cor(robust[valid], future_ret[valid]) : 0.0

    (naive_signal=naive, robust_signal=robust,
     naive_ic=corr_naive, robust_ic=corr_robust,
     ic_improvement=corr_robust - corr_naive)
end

# ─────────────────────────────────────────────────────────────
# 10. DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_robust_statistics_demo() -> Nothing
"""
function run_robust_statistics_demo()
    println("=" ^ 60)
    println("ROBUST STATISTICS DEMO")
    println("=" ^ 60)

    rng = MersenneTwister(42)
    n   = 200

    # Generate contaminated data
    clean   = randn(rng, n) .* 2 .+ 5.0  # N(5, 4)
    contam  = copy(clean)
    outlier_idx = rand(rng, 1:n, 20)
    contam[outlier_idx] .+= randn(rng, 20) .* 30  # 10% contamination

    println("\n1. M-Estimators for Location")
    est_huber  = HuberEstimator(1.345)
    est_tukey  = TukeyBiweight(4.685)
    est_wave   = AndrewsWave(1.339)
    println("  True mean: 5.0")
    println("  OLS mean:             $(round(mean(contam), digits=3))")
    println("  Huber location:       $(round(m_estimate_location(contam, est_huber), digits=3))")
    println("  Tukey Biweight:       $(round(m_estimate_location(contam, est_tukey), digits=3))")
    println("  Andrews Wave:         $(round(m_estimate_location(contam, est_wave),  digits=3))")
    println("  Median:               $(round(median(contam), digits=3))")

    println("\n2. Robust Regression")
    X = randn(rng, n, 3)
    true_beta = [1.5, -0.8, 2.0]
    y = X * true_beta .+ randn(rng, n)
    y[rand(rng, 1:n, 15)] .+= randn(rng, 15) .* 20  # outliers

    ols   = (X'X + 1e-8*I) \ (X'y)
    hr    = huber_regression(X, y)
    bis   = bisquare_regression(X, y)
    println("  True beta:         $(true_beta)")
    println("  OLS beta:          $(round.(ols, digits=3))")
    println("  Huber beta:        $(round.(hr.coefficients, digits=3))")
    println("  Bisquare beta:     $(round.(bis.coefficients, digits=3))")
    println("  Tukey BP:          $(round(regression_breakdown_point(TukeyBiweight()), digits=2))")

    println("\n3. Minimum Covariance Determinant")
    X2 = randn(rng, n, 4)
    X2[rand(rng, 1:n, 10), :] .+= 10.0  # multivariate outliers
    od = mcd_outlier_detection(X2; threshold=2.5, n_trials=5, rng=rng)
    println("  Detected $(od.n_outliers) outliers (planted: 10)")
    println("  Mean MCD distance: $(round(mean(od.distances), digits=3))")

    println("\n4. Robust PCA (RPCA)")
    T_rpca  = 50; N_rpca = 5
    L_true  = randn(rng, T_rpca, 2) * randn(rng, 2, N_rpca)  # low-rank
    S_true  = sprand_custom(rng, T_rpca, N_rpca, 0.05) .* 5.0  # sparse
    M_rpca  = L_true .+ S_true
    rpca    = rpca_decompose(M_rpca)
    println("  Sparse fraction:  $(round(rpca.sparse_fraction*100, digits=1))%")
    println("  Effective rank:   $(rpca.effective_rank)")
    println("  Converged:        $(rpca.converged)")
    # Recovery error
    L_err = norm(rpca.low_rank - L_true, "fro") / (norm(L_true, "fro") + 1e-10)
    println("  Low-rank error:   $(round(L_err*100, digits=1))%")

    println("\n5. Winsorization")
    w_std   = winsorize(contam, 0.05, 0.95)
    w_opt   = winsorize_optimal(contam)
    println("  Raw std:          $(round(std(contam), digits=3))")
    println("  Winsorized (5%) std: $(round(std(w_std), digits=3))")
    println("  Optimal winsorized: $(round(std(w_opt), digits=3))")
    println("  Trimmed mean (10%): $(round(trimmed_mean(contam, 0.1), digits=3))")

    println("\n6. Robust Risk Measures")
    returns_noisy = randn(rng, 500) .* 0.02 .+ 0.001
    returns_noisy[rand(rng, 1:500, 10)] .-= 0.15  # fat tail events
    println("  Standard Sharpe: $(round(mean(returns_noisy)/std(returns_noisy)*sqrt(252),digits=3))")
    println("  Robust Sharpe:   $(round(robust_sharpe(returns_noisy),digits=3))")
    println("  Standard VaR 99%: $(round(quantile(returns_noisy,0.01)*(-1)*100,digits=3))%")
    println("  Robust VaR 99%:  $(round(robust_var(returns_noisy,0.99)*100,digits=3))%")
    println("  Robust CVaR 99%: $(round(robust_cvar(returns_noisy,0.99)*100,digits=3))%")

    println("\n7. Influence Function")
    est_fn = x -> m_estimate_location(x, HuberEstimator())
    sc     = sensitivity_curve(contam[1:50], est_fn; n_grid=20)
    println("  Max sensitivity:     $(round(sc.max_influence, digits=3))")
    println("  Bounded influence:   $(sc.bounded)")

    println("\n8. Robust BH Signal")
    prices_noisy = cumsum(randn(rng, 300) .* 0.015) .+ 100.0
    # Inject price spikes
    prices_noisy[rand(rng, 1:300, 10)] .*= (1 .+ randn(rng, 10) .* 0.3)
    prices_noisy = max.(prices_noisy, 1.0)
    cmp = compare_signals(prices_noisy; lookback=15)
    println("  Naive IC:   $(round(cmp.naive_ic, digits=4))")
    println("  Robust IC:  $(round(cmp.robust_ic, digits=4))")
    println("  Improvement: $(round(cmp.ic_improvement, digits=4))")

    println("\nDone.")
    nothing
end

"""Generate sparse random matrix."""
function sprand_custom(rng, m::Int, n::Int, density::Float64)::Matrix{Float64}
    S = zeros(m, n)
    for i in 1:m, j in 1:n
        rand(rng) < density && (S[i,j] = randn(rng))
    end
    S
end

# ─────────────────────────────────────────────────────────────
# 11. ROBUST COVARIANCE ESTIMATION
# ─────────────────────────────────────────────────────────────

"""
    shrinkage_covariance(X; method=:ledoit_wolf) -> Matrix{Float64}

Shrinkage covariance estimator (Ledoit-Wolf or Oracle shrinkage).
Shrinks sample covariance toward a structured target.
"""
function shrinkage_covariance(X::Matrix{Float64};
                               method::Symbol=:ledoit_wolf)::Matrix{Float64}
    n, p = size(X)
    S = cov(X)  # sample covariance

    if method == :diagonal
        target = Diagonal(diag(S))
        # Optimal shrinkage intensity (simplified)
        alpha = min(0.5, p / n)
        return (1 - alpha) .* S .+ alpha .* Matrix(target)
    end

    # Ledoit-Wolf oracle approximation
    mu = tr(S) / p  # shrinkage target = mu * I
    target = mu * I
    # Analytical shrinkage coefficient
    delta_sq = norm(S - Matrix(target), "fro")^2
    alpha_lw  = min(delta_sq / (norm(S, "fro")^2 + 1e-10), 1.0)
    (1 - alpha_lw) .* S .+ alpha_lw .* mu .* Matrix{Float64}(I, p, p)
end

norm(A::Matrix, ::String) = sqrt(sum(A.^2))  # Frobenius

"""
    tyler_m_estimator(X; tol=1e-5, maxiter=100) -> Matrix{Float64}

Tyler's M-estimator of scatter (distribution-free, robust shape estimate).
Iterative: Σ_{k+1} = (p/n) Σ_i (x_i x_i') / (x_i' Σ_k^{-1} x_i)
"""
function tyler_m_estimator(X::Matrix{Float64};
                             tol::Float64=1e-5, maxiter::Int=100)::Matrix{Float64}
    n, p = size(X)
    Sigma = Matrix{Float64}(I, p, p)
    for _ in 1:maxiter
        Sigma_inv = inv(Sigma + 1e-8*I)
        # Reweighted scatter
        Sigma_new = zeros(p, p)
        for i in 1:n
            xi = X[i, :]
            w = p / max(dot(xi, Sigma_inv * xi), 1e-10)
            Sigma_new .+= w .* xi * xi'
        end
        Sigma_new ./= n
        # Normalize for identifiability: tr(Sigma) = p
        Sigma_new .*= p / tr(Sigma_new)
        norm(Sigma_new - Sigma, "fro") < tol && (Sigma = Sigma_new; break)
        Sigma = Sigma_new
    end
    Sigma
end

# ─────────────────────────────────────────────────────────────
# 12. ROBUST HYPOTHESIS TESTING
# ─────────────────────────────────────────────────────────────

"""
    robust_t_test(x1, x2) -> NamedTuple

Robust two-sample test using trimmed means and Winsorized variance.
"""
function robust_t_test(x1::Vector{Float64}, x2::Vector{Float64};
                        alpha::Float64=0.1)
    n1 = length(x1); n2 = length(x2)
    mu1 = trimmed_mean(x1, alpha); mu2 = trimmed_mean(x2, alpha)
    s1  = trimmed_std(x1, alpha);  s2  = trimmed_std(x2, alpha)
    se  = sqrt(s1^2/n1 + s2^2/n2 + 1e-10)
    t_stat = (mu1 - mu2) / se
    df = min(n1, n2) - 1
    # Approximate p-value via normal (large sample)
    p_val = 2.0 * norm_cdf(-abs(t_stat))
    (t_statistic=t_stat, p_value=p_val, df=df, robust_mean_diff=mu1-mu2,
     significant_5pct=p_val < 0.05)
end

"""
    wilcoxon_signed_rank(x; mu0=0.0) -> NamedTuple

Wilcoxon signed-rank test for median = mu0.
"""
function wilcoxon_signed_rank(x::Vector{Float64}; mu0::Float64=0.0)
    d = x .- mu0
    d = d[abs.(d) .> 1e-10]
    n = length(d)
    n == 0 && return (W=0.0, p_value=1.0, n=0)
    # Rank absolute values
    abs_d = abs.(d)
    ranks = sortperm(sortperm(abs_d)) .+ 0.0
    W_plus  = sum(ranks[d .> 0]; init=0.0)
    W_minus = sum(ranks[d .< 0]; init=0.0)
    W = min(W_plus, W_minus)
    # Normal approximation
    mu_W = n*(n+1)/4.0; sigma_W = sqrt(n*(n+1)*(2n+1)/24.0 + 1e-10)
    z = (W - mu_W) / sigma_W
    p_val = 2.0 * norm_cdf(-abs(z))
    (W_statistic=W, p_value=p_val, n=n, significant_5pct=p_val < 0.05)
end

# ─────────────────────────────────────────────────────────────
# 13. QUANTILE REGRESSION
# ─────────────────────────────────────────────────────────────

"""
    quantile_regression(X, y, tau; maxiter=200) -> Vector{Float64}

Quantile regression at quantile level tau ∈ (0,1) via interior point / IRLS.
Minimizes Σ ρ_τ(y_i - x_i'β) where ρ_τ(u) = u*(τ - I(u<0)).
"""
function quantile_regression(X::Matrix{Float64}, y::Vector{Float64},
                               tau::Float64=0.5;
                               maxiter::Int=200, tol::Float64=1e-6)::Vector{Float64}
    n, p = size(X)
    beta = (X'X + 1e-8*I) \ (X'y)  # OLS init

    for _ in 1:maxiter
        resid = y .- X * beta
        # Asymmetric weights
        w = [abs(r) < 1e-6 ? 1e4 :
             (r > 0 ? tau / abs(r) : (1-tau) / abs(r)) for r in resid]
        W = Diagonal(w)
        beta_new = (X'*W*X + 1e-8*I) \ (X'*W*y)
        norm(beta_new - beta) < tol && (beta = beta_new; break)
        beta = beta_new
    end
    beta
end

"""
    quantile_regression_band(X, y; quantiles=[0.1,0.5,0.9]) -> Matrix{Float64}

Fit quantile regression at multiple quantile levels.
Returns n × length(quantiles) matrix of fitted values.
"""
function quantile_regression_band(X::Matrix{Float64}, y::Vector{Float64};
                                    quantiles::Vector{Float64}=[0.1,0.5,0.9])::Matrix{Float64}
    n = size(X, 1); q = length(quantiles)
    fitted = zeros(n, q)
    for (j, tau) in enumerate(quantiles)
        beta = quantile_regression(X, y, tau)
        fitted[:, j] = X * beta
    end
    fitted
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 13 – Robust Trend and Changepoint Detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_trend_filter(y, lam)

ℓ1 Trend Filter (Kim et al. 2009): minimise ½||y - x||² + λ||D²x||₁
where D² is the second-difference operator.  Solved via ADMM.
Produces a piecewise-linear robust trend extraction.
"""
function robust_trend_filter(y::Vector{Float64}, lam::Float64=10.0;
                               max_iter::Int=500, tol::Float64=1e-4)
    n = length(y)
    # D2: (n-2) × n second difference matrix rows
    D2 = zeros(n-2, n)
    for i in 1:(n-2)
        D2[i, i] = 1; D2[i, i+1] = -2; D2[i, i+2] = 1
    end
    m  = n - 2
    # ADMM variables
    x  = copy(y)
    z  = D2 * y
    u  = zeros(m)
    rho = 1.0
    A  = I(n) + rho * D2' * D2
    # Cholesky of A (tridiagonal-like, use direct solve)
    for iter in 1:max_iter
        # x update
        rhs = y + rho * D2' * (z .- u)
        x   = A \ rhs
        # z update: soft threshold
        Dx  = D2 * x
        v   = Dx .+ u
        z_new = sign.(v) .* max.(abs.(v) .- lam / rho, 0.0)
        r_norm = norm(Dx .- z_new)
        z = z_new
        # dual update
        u .+= Dx .- z
        r_norm < tol && break
    end
    return x
end

"""
    robust_changepoint_detect(y, penalty, min_seg)

Binary segmentation for mean-shift changepoint detection using
robust median-based test statistic.
Returns vector of changepoint indices.
"""
function robust_changepoint_detect(y::Vector{Float64},
                                    penalty::Float64=10.0,
                                    min_seg::Int=10)
    n = length(y)
    changepoints = Int[]
    # iterative binary segmentation
    segments = [(1, n)]
    while !isempty(segments)
        (l, r) = popfirst!(segments)
        r - l < 2 * min_seg && continue
        best_stat = 0.0; best_cp = -1
        for cp in (l + min_seg):(r - min_seg)
            left  = y[l:cp]; right = y[cp+1:r]
            stat  = abs(median(left) - median(right))
            if stat > best_stat
                best_stat = stat; best_cp = cp
            end
        end
        if best_stat > penalty
            push!(changepoints, best_cp)
            push!(segments, (l, best_cp))
            push!(segments, (best_cp+1, r))
        end
    end
    return sort(changepoints)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 14 – Robust Portfolio Construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_minimum_variance(returns; max_weight, reg)

Minimum-variance portfolio using MCD covariance estimate.
Weights found via projected gradient descent on w'Σw subject to
∑w=1, w∈[0, max_weight].
"""
function robust_minimum_variance(returns::Matrix{Float64};
                                  max_weight::Float64=0.4,
                                  reg::Float64=1e-4,
                                  max_iter::Int=1000)
    n_assets = size(returns, 2)
    Sigma, _ = mcd_covariance(returns)
    Sigma   += I(n_assets) * reg
    # projected gradient descent
    w = ones(n_assets) / n_assets
    lr = 1.0 / (2 * opnorm(Sigma))
    for _ in 1:max_iter
        grad = 2 * Sigma * w
        w   -= lr * grad
        # project onto simplex with box constraints [0, max_weight]
        w    = clamp.(w, 0.0, max_weight)
        s    = sum(w)
        s > 0 && (w ./= s)
    end
    return (weights=w, variance=dot(w, Sigma * w))
end

"""
    robust_risk_parity(returns; reg)

Equal Risk Contribution (ERC) portfolio using MCD covariance.
Solves w* = argmin ∑ᵢ (wᵢ (Σw)ᵢ - 1/n * w'Σw)² via gradient descent.
"""
function robust_risk_parity(returns::Matrix{Float64};
                              reg::Float64=1e-4, max_iter::Int=2000,
                              lr::Float64=0.01)
    n = size(returns, 2)
    Sigma, _ = mcd_covariance(returns)
    Sigma   += I(n) * reg
    w = ones(n) / n
    for _ in 1:max_iter
        Sw  = Sigma * w
        wSw = dot(w, Sw)
        target = wSw / n
        rc  = w .* Sw              # risk contributions
        loss_grad = 2 * (rc .- target) .* Sw
        w  -= lr * loss_grad
        w   = max.(w, 1e-8)
        w ./= sum(w)
    end
    return (weights=w, risk_contributions=w .* (Sigma * w))
end

"""
    robust_mean_variance(returns, gamma; reg)

Mean-variance optimisation using Huber location and MCD covariance.
max μ'w - (γ/2) w'Σw  s.t. ∑w=1, w≥0
"""
function robust_mean_variance(returns::Matrix{Float64}, gamma::Float64=1.0;
                                reg::Float64=1e-4, max_iter::Int=2000)
    n  = size(returns, 2)
    mu = [m_estimate_location(returns[:, i], HuberEstimator()) for i in 1:n]
    Sigma, _ = mcd_covariance(returns)
    Sigma   += I(n) * reg
    w = ones(n) / n
    lr = 1e-2
    for _ in 1:max_iter
        grad = mu - gamma * Sigma * w
        w   += lr * grad
        w    = max.(w, 0.0)
        s    = sum(w); s > 0 && (w ./= s)
    end
    ret  = dot(mu, w)
    risk = sqrt(dot(w, Sigma * w))
    return (weights=w, expected_return=ret, volatility=risk,
            sharpe=ret / (risk + 1e-8))
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 15 – Robust Evaluation Metrics for Crypto Returns
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_information_ratio(returns, benchmark_returns, window)

Rolling information ratio using Huber-estimated mean and MAD scale
for active return (returns - benchmark_returns).
"""
function robust_information_ratio(returns::Vector{Float64},
                                   benchmark::Vector{Float64},
                                   window::Int=60)
    n      = length(returns)
    active = returns .- benchmark
    ir     = fill(NaN, n)
    for i in (window+1):n
        sub  = active[i-window+1:i]
        mu_h = m_estimate_location(sub, HuberEstimator())
        sg_h = mad_scale(sub)
        ir[i] = sg_h < 1e-8 ? 0.0 : mu_h / sg_h * sqrt(window)
    end
    return ir
end

"""
    robust_tail_ratio(returns, percentile)

Tail ratio: abs(p-th percentile) / abs((100-p)-th percentile).
Values > 1 → right tail dominates (positive skew), < 1 → left tail.
"""
function robust_tail_ratio(returns::Vector{Float64}, percentile::Float64=5.0)
    sorted = sort(returns)
    n      = length(sorted)
    lo_idx = max(1, floor(Int, percentile / 100 * n))
    hi_idx = min(n, ceil(Int, (1 - percentile/100) * n))
    lo     = abs(sorted[lo_idx])
    hi     = abs(sorted[hi_idx])
    return hi / (lo + 1e-8)
end

end  # module RobustStatistics
