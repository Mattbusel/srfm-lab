"""
RobustStatistics.jl
====================
Robust statistics module for financial data analysis.

Exports:
  huber_loss           — Huber M-estimator loss/derivative
  tukey_biweight       — Tukey's bisquare M-estimator
  andrews_wave         — Andrews' sine wave M-estimator
  robust_regression    — IRLS with user-selected M-estimator
  mcd_covariance       — Minimum Covariance Determinant
  rpca                 — Robust PCA via ADMM
  robust_sharpe        — Sharpe using median and MAD
  robust_var           — Trimmed Value-at-Risk
  influence_function   — Empirical influence function
  breakdown_point      — Theoretical breakdown point
"""
module RobustStatistics

using Statistics, LinearAlgebra, Random

export huber_loss, tukey_biweight, andrews_wave,
       robust_regression, mcd_covariance, rpca,
       robust_sharpe, robust_var, influence_function, breakdown_point,
       robust_mean, robust_std, robust_covariance,
       robust_regression_irls, mad, trimmed_mean, winsorized_mean,
       robust_correlation, theil_sen_slope, lms_regression,
       robust_pca_huber

# ─────────────────────────────────────────────────────────────────────────────
# 1. M-ESTIMATOR LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
    huber_loss(r, k) → (loss, weight, derivative)

Huber M-estimator.
  ρ(r) = r²/2           if |r| ≤ k  (quadratic zone)
  ρ(r) = k|r| - k²/2   if |r| > k  (linear zone)

`k`: tuning constant (default 1.345 for 95% efficiency at Gaussian)
"""
function huber_loss(r::Float64, k::Float64=1.345)
    ar = abs(r)
    if ar <= k
        loss = 0.5 * r^2
        psi  = r           # derivative ψ(r) = ρ'(r)
        w    = 1.0         # IRLS weight = ψ(r)/r
    else
        loss = k * ar - 0.5 * k^2
        psi  = k * sign(r)
        w    = k / ar
    end
    return (loss=loss, psi=psi, weight=w)
end

function huber_loss(r::Vector{Float64}, k::Float64=1.345)
    results = [huber_loss(ri, k) for ri in r]
    return (
        loss   = [res.loss   for res in results],
        psi    = [res.psi    for res in results],
        weight = [res.weight for res in results],
    )
end

"""
    tukey_biweight(r, c) → (loss, weight, derivative)

Tukey's bisquare / biweight M-estimator.
  ρ(r) = (c²/6)[1 - (1 - (r/c)²)³]  if |r| ≤ c
  ρ(r) = c²/6                          if |r| > c

`c`: tuning constant (default 4.685 for 95% efficiency at Gaussian)
"""
function tukey_biweight(r::Float64, c::Float64=4.685)
    u = r / c
    if abs(u) <= 1.0
        loss = (c^2/6) * (1 - (1 - u^2)^3)
        psi  = r * (1 - u^2)^2
        w    = (1 - u^2)^2
    else
        loss = c^2 / 6
        psi  = 0.0
        w    = 0.0
    end
    return (loss=loss, psi=psi, weight=w)
end

function tukey_biweight(r::Vector{Float64}, c::Float64=4.685)
    results = [tukey_biweight(ri, c) for ri in r]
    return (
        loss   = [res.loss   for res in results],
        psi    = [res.psi    for res in results],
        weight = [res.weight for res in results],
    )
end

"""
    andrews_wave(r, c) → (loss, weight, derivative)

Andrews' sine wave M-estimator.
  ψ(r) = sin(r/c)  if |r| ≤ cπ
  ψ(r) = 0          otherwise

`c`: tuning constant (default 1.339 for 95% efficiency)
"""
function andrews_wave(r::Float64, c::Float64=1.339)
    if abs(r) <= c * π
        psi  = sin(r / c)
        loss = c * (1 - cos(r / c))
        w    = abs(r) > 1e-10 ? psi / r : 1.0 / c
    else
        psi  = 0.0
        loss = 2 * c
        w    = 0.0
    end
    return (loss=loss, psi=psi, weight=w)
end

function andrews_wave(r::Vector{Float64}, c::Float64=1.339)
    results = [andrews_wave(ri, c) for ri in r]
    return (
        loss   = [res.loss   for res in results],
        psi    = [res.psi    for res in results],
        weight = [res.weight for res in results],
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. ROBUST LOCATION / SCALE ESTIMATORS
# ─────────────────────────────────────────────────────────────────────────────

"""
    mad(x) → Float64

Median Absolute Deviation (MAD).
mad(x) = median(|x_i - median(x)|)
Scaled by 1/Φ^{-1}(3/4) ≈ 1.4826 for consistency with σ at Gaussian.
"""
function mad(x::Vector{Float64}; consistency_factor::Float64=1.4826)
    med = median(x)
    return consistency_factor * median(abs.(x .- med))
end

"""
    robust_mean(x, method) → Float64

Robust location estimators.
`method`: :median, :trimmed (10%), :huber, :bisquare
"""
function robust_mean(x::Vector{Float64}; method::Symbol=:huber,
                      trim_frac::Float64=0.10, k::Float64=1.345, max_iter::Int=50)
    if method == :median
        return median(x)
    elseif method == :trimmed
        return trimmed_mean(x, trim_frac)
    elseif method == :huber
        return _irls_location(x, k, :huber, max_iter)
    elseif method == :bisquare
        return _irls_location(x, 4.685, :bisquare, max_iter)
    else
        error("Unknown method: $method")
    end
end

function _irls_location(x::Vector{Float64}, k::Float64, estimator::Symbol,
                          max_iter::Int)
    mu = median(x)
    s  = mad(x) + 1e-8

    for _ in 1:max_iter
        r = (x .- mu) ./ s
        if estimator == :huber
            ws = huber_loss(r, k).weight
        else
            ws = tukey_biweight(r, k).weight
        end
        ws = max.(ws, 1e-8)
        mu_new = sum(ws .* x) / sum(ws)
        abs(mu_new - mu) < 1e-6 * s && return mu_new
        mu = mu_new
    end
    return mu
end

"""
    robust_std(x, method) → Float64

Robust scale estimators.
`method`: :mad, :qn, :sn, :iqr
"""
function robust_std(x::Vector{Float64}; method::Symbol=:mad)
    if method == :mad
        return mad(x)
    elseif method == :iqr
        q75, q25 = quantile(x, 0.75), quantile(x, 0.25)
        return (q75 - q25) / 1.3490   # consistency factor
    elseif method == :qn
        # Qn estimator (Rousseeuw & Croux 1993)
        n = length(x)
        h = n ÷ 2 + 1
        k = h * (h - 1) ÷ 2
        diffs = sort([abs(x[i] - x[j]) for i in 1:n for j in i+1:n])
        isempty(diffs) && return 0.0
        cn = length(x) <= 9 ? [0.399, 0.994, 0.512, 0.844, 0.611, 0.857, 0.669, 0.872][n-1] : 1.0
        return cn * 2.2219 * diffs[min(k, length(diffs))]
    else
        return mad(x)
    end
end

"""
    trimmed_mean(x, alpha) → Float64

Alpha-trimmed mean: remove alpha fraction from each tail.
"""
function trimmed_mean(x::Vector{Float64}, alpha::Float64=0.10)
    n = length(x)
    k = floor(Int, n * alpha)
    xs = sort(x)
    return mean(xs[k+1:n-k])
end

"""
    winsorized_mean(x, alpha) → Float64

Winsorized mean: replace tail values with quantile boundaries.
"""
function winsorized_mean(x::Vector{Float64}, alpha::Float64=0.10)
    lo = quantile(x, alpha)
    hi = quantile(x, 1 - alpha)
    return mean(clamp.(x, lo, hi))
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. ROBUST REGRESSION (IRLS)
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_regression(X, y, estimator; kwargs) → NamedTuple

Iteratively Reweighted Least Squares (IRLS) with M-estimator.
`estimator`: :huber (default), :bisquare, :andrews

Returns coefficient vector, residuals, weights, scale, and convergence info.
"""
function robust_regression(X::Matrix{Float64}, y::Vector{Float64};
                             estimator::Symbol=:huber,
                             k_huber::Float64=1.345,
                             k_bisquare::Float64=4.685,
                             k_andrews::Float64=1.339,
                             max_iter::Int=50,
                             tol::Float64=1e-6,
                             add_intercept::Bool=true)
    n, p = size(X)
    Xd   = add_intercept ? hcat(ones(n), X) : X
    p_aug = size(Xd, 2)

    # Initial OLS estimate
    beta = (Xd'Xd + 1e-8*I(p_aug)) \ (Xd'y)
    resid = y .- Xd * beta

    k = estimator == :huber    ? k_huber    :
        estimator == :bisquare ? k_bisquare :
        estimator == :andrews  ? k_andrews  : k_huber

    for iter in 1:max_iter
        # Robust scale estimate
        scale = mad(resid) + 1e-10
        r_std = resid ./ scale

        # Compute IRLS weights
        if estimator == :huber
            ws = huber_loss(r_std, k).weight
        elseif estimator == :bisquare
            ws = tukey_biweight(r_std, k).weight
        elseif estimator == :andrews
            ws = andrews_wave(r_std, k).weight
        else
            ws = ones(n)
        end
        ws = max.(ws, 1e-8)

        # Weighted OLS
        W       = Diagonal(ws)
        XtWX    = Xd'W*Xd + 1e-8*I(p_aug)
        XtWy    = Xd'(W*y)
        beta_new = XtWX \ XtWy

        delta = norm(beta_new - beta)
        beta  = beta_new
        resid = y .- Xd * beta

        if delta < tol * (norm(beta) + 1e-8)
            return (beta=beta, residuals=resid, weights=ws,
                    scale=scale, converged=true, n_iter=iter,
                    r2=_r2(y, resid))
        end
    end

    scale = mad(resid) + 1e-10
    r_std = resid ./ scale
    ws = estimator == :huber    ? huber_loss(r_std, k).weight :
         estimator == :bisquare ? tukey_biweight(r_std, k).weight :
                                  andrews_wave(r_std, k).weight

    return (beta=beta, residuals=resid, weights=ws,
            scale=scale, converged=false, n_iter=max_iter,
            r2=_r2(y, resid))
end

function _r2(y::Vector{Float64}, resid::Vector{Float64})
    ss_tot = sum((y .- mean(y)).^2)
    ss_res = sum(resid.^2)
    return max(1 - ss_res / (ss_tot + 1e-10), 0.0)
end

"""
    theil_sen_slope(x, y) → (slope, intercept)

Theil-Sen non-parametric regression: slope = median of all pairwise slopes.
Breakdown point ≈ 29.3%.
"""
function theil_sen_slope(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    length(y) == n || error("x and y must have same length")

    slopes = Float64[]
    for i in 1:n, j in i+1:n
        dx = x[j] - x[i]
        abs(dx) > 1e-10 && push!(slopes, (y[j] - y[i]) / dx)
    end
    isempty(slopes) && return (slope=NaN, intercept=NaN)

    slope     = median(slopes)
    intercept = median(y .- slope .* x)
    return (slope=slope, intercept=intercept)
end

"""
    lms_regression(X, y, n_subsets) → NamedTuple

Least Median of Squares regression.
Minimizes the median of squared residuals.
"""
function lms_regression(X::Matrix{Float64}, y::Vector{Float64};
                          n_subsets::Int=500, seed::Int=42)
    rng = MersenneTwister(seed)
    n, p = size(X)
    Xd   = hcat(ones(n), X)
    p_aug = p + 1

    best_meds = Inf
    best_beta = zeros(p_aug)

    for _ in 1:n_subsets
        # Random minimal subset (p+1 observations)
        idx  = sort(randperm(rng, n)[1:min(p_aug, n)])
        Xs   = Xd[idx, :]
        ys   = y[idx]
        rank_ok = rank(Xs) >= p_aug
        rank_ok || continue

        beta  = (Xs'Xs + 1e-8*I(p_aug)) \ (Xs'ys)
        resid = y .- Xd * beta
        meds  = median(resid.^2)

        if meds < best_meds
            best_meds = meds
            best_beta = beta
        end
    end

    resid = y .- Xd * best_beta
    scale = 1.4826 * sqrt(best_meds)
    return (beta=best_beta, residuals=resid, scale=scale, lms=best_meds)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. MINIMUM COVARIANCE DETERMINANT
# ─────────────────────────────────────────────────────────────────────────────

"""
    mcd_covariance(X, h) → (location, scatter, support_set)

Minimum Covariance Determinant estimator.
`h`: number of observations to use (default = floor(0.75*n)).
Returns robust location (mean), scatter (covariance), and support set indices.
"""
function mcd_covariance(X::Matrix{Float64};
                          h_frac::Float64=0.75,
                          n_trials::Int=10,
                          max_iter::Int=100,
                          seed::Int=42)
    rng = MersenneTwister(seed)
    n, p = size(X)
    h    = floor(Int, n * h_frac)
    h    = max(h, p + 1)

    best_det  = Inf
    best_loc  = zeros(p)
    best_scat = I(p) * 1.0
    best_h_set = collect(1:h)

    for trial in 1:n_trials
        # Random initial subset
        h_set = sort(randperm(rng, n)[1:h])

        for iter in 1:max_iter
            Xh     = X[h_set, :]
            loc_h  = vec(mean(Xh, dims=1))
            scat_h = cov(Xh) + 1e-8 * I(p)

            # Mahalanobis distances to all n observations
            Sinv    = inv(scat_h)
            diffs   = X .- loc_h'
            mah_sq  = [dot(diffs[i,:], Sinv * diffs[i,:]) for i in 1:n]

            # New h-subset = observations with smallest Mahalanobis distance
            new_h_set = sort(sortperm(mah_sq)[1:h])
            new_h_set == h_set && break
            h_set = new_h_set
        end

        Xh    = X[h_set, :]
        scat  = cov(Xh) + 1e-8 * I(p)
        det_v = det(scat)

        if det_v < best_det
            best_det   = det_v
            best_loc   = vec(mean(Xh, dims=1))
            best_scat  = scat
            best_h_set = h_set
        end
    end

    # Consistency correction for normal distribution
    chi2_q = quantile_chi2(p, h_frac)
    correction = h_frac / chi2_q
    best_scat .= correction .* best_scat

    return (location=best_loc, scatter=best_scat, support=best_h_set,
            breakdown_point=1 - h_frac)
end

"""
    quantile_chi2(p, alpha) → Float64

Approximate chi-squared quantile via Wilson-Hilferty.
"""
function quantile_chi2(p::Int, alpha::Float64)
    # Normal quantile approximation
    z = quantile_normal(alpha)
    return p * (1 + z * sqrt(2/p) - 1/(3p)) + sqrt(2p) * z
end

function quantile_normal(p::Float64)
    # Rational approximation for N^{-1}(p)
    p = clamp(p, 1e-6, 1 - 1e-6)
    if p < 0.5
        t = sqrt(-2 * log(p))
        return -(t - (2.515517 + 0.802853*t + 0.010328*t^2) /
                     (1 + 1.432788*t + 0.189269*t^2 + 0.001308*t^3))
    else
        return -quantile_normal(1 - p)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. ROBUST PCA VIA ADMM
# ─────────────────────────────────────────────────────────────────────────────

"""
    rpca(M, lambda) → (L, S)

Robust PCA via ADMM: decompose M = L + S
  L: low-rank component
  S: sparse component (outliers)

minimize rank(L) + λ * ‖S‖₁  subject to  L + S = M
Solved via nuclear norm + L1 norm relaxation using ADMM.

`lambda`: sparse regularization (default 1/√max(m,n))
"""
function rpca(M::Matrix{Float64}; lambda::Union{Float64,Nothing}=nothing,
               max_iter::Int=500, tol::Float64=1e-6, rho::Float64=1.0)
    m, n  = size(M)
    λ     = lambda === nothing ? 1.0 / sqrt(max(m, n)) : lambda

    L = copy(M)
    S = zeros(m, n)
    Y = zeros(m, n)   # Lagrange multipliers

    frobenius_norm_M = norm(M, :F) + 1e-10

    for iter in 1:max_iter
        # Update L: nuclear norm prox (SVD soft-threshold)
        L = _svd_threshold(M - S + Y / rho, 1.0 / rho)

        # Update S: L1 norm prox (element-wise soft-threshold)
        R = M - L + Y / rho
        S = _soft_threshold.(R, λ / rho)

        # Update Lagrange multiplier
        residual = M - L - S
        Y .+= rho .* residual

        # Convergence check
        err = norm(residual, :F) / frobenius_norm_M
        err < tol && break
    end

    return (L=L, S=S, rank_L=_approx_rank(L), sparsity_S=mean(abs.(S) .> 1e-6))
end

"""
    _svd_threshold(A, tau) → Matrix{Float64}

Nuclear norm proximal operator: soft-threshold singular values.
"""
function _svd_threshold(A::Matrix{Float64}, tau::Float64)
    F = svd(A)
    sigma_thresh = max.(F.S .- tau, 0.0)
    return F.U * Diagonal(sigma_thresh) * F.Vt
end

"""
    _soft_threshold(x, tau) → Float64

L1 proximal operator.
"""
_soft_threshold(x::Float64, tau::Float64) = sign(x) * max(abs(x) - tau, 0.0)

function _approx_rank(A::Matrix{Float64}; threshold::Float64=1e-3)
    sv = svdvals(A)
    return sum(sv .> threshold * sv[1])
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. ROBUST FINANCIAL RISK MEASURES
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_sharpe(returns) → Float64

Robust Sharpe ratio using median and MAD instead of mean/std.
rs = median(r) / MAD(r)
Annualized by sqrt(252).
"""
function robust_sharpe(returns::Vector{Float64}; ann_factor::Float64=252.0)
    isempty(returns) && return NaN
    mu   = median(returns)
    sig  = mad(returns)
    sig < 1e-10 && return NaN
    return (mu / sig) * sqrt(ann_factor)
end

"""
    robust_var(returns, alpha; method) → Float64

Robust Value-at-Risk.
`method`:
  :trimmed  — use trimmed distribution
  :huber    — fit Huber regression to tail
  :mcd      — MCD-based elliptical VaR
"""
function robust_var(returns::Vector{Float64}, alpha::Float64=0.05;
                     method::Symbol=:trimmed)
    isempty(returns) && return NaN

    if method == :trimmed
        # Trim 1% from the top (right tail) before computing VaR
        xs = sort(returns)
        n  = length(xs)
        n_top_trim = floor(Int, n * 0.01)
        xs_trim    = xs[1:n-n_top_trim]
        return -quantile(xs_trim, alpha)

    elseif method == :huber
        # Fit Huber estimator to returns, compute VaR from robust fit
        mu_h  = robust_mean(returns; method=:huber)
        sig_h = mad(returns)
        # Use normal approximation with robust params
        z_alpha = -1.6449   # approximate N^{-1}(0.05)
        return -(mu_h + z_alpha * sig_h)

    else  # default trimmed
        return -quantile(returns, alpha)
    end
end

"""
    robust_cvar(returns, alpha) → Float64

Robust Conditional Value-at-Risk (Expected Shortfall).
Uses Winsorized tail estimate.
"""
function robust_cvar(returns::Vector{Float64}, alpha::Float64=0.05)
    isempty(returns) && return NaN
    xs    = sort(returns)
    n     = length(xs)
    k     = ceil(Int, n * alpha)
    k     = max(k, 1)
    # Winsorize: replace extreme tail with 5th percentile of tail
    tail  = xs[1:k]
    return -mean(tail)
end

"""
    robust_covariance(X; method) → Matrix{Float64}

Robust covariance estimation.
`method`: :mcd, :ledoit_wolf_robust, :ogk (Orthogonalized Gnanadesikan-Kettenring)
"""
function robust_covariance(X::Matrix{Float64}; method::Symbol=:mcd)
    n, p = size(X)
    if method == :mcd
        mcd = mcd_covariance(X)
        return mcd.scatter
    elseif method == :ogk
        return _ogk_covariance(X)
    else
        return cov(X) + 1e-8 * I(p)
    end
end

"""
    _ogk_covariance(X) → Matrix{Float64}

Orthogonalized Gnanadesikan-Kettenring (OGK) robust covariance.
"""
function _ogk_covariance(X::Matrix{Float64})
    n, p = size(X)
    D    = zeros(p)
    for j in 1:p
        D[j] = mad(X[:,j]) + 1e-10
    end
    Z = X ./ D'   # standardize with MAD

    # Robust correlation using tau-scale
    U = zeros(p, p)
    for i in 1:p, j in 1:p
        if i == j
            U[i,j] = 1.0
        else
            s_plus  = mad(Z[:,i] .+ Z[:,j]) + 1e-10
            s_minus = mad(Z[:,i] .- Z[:,j]) + 1e-10
            U[i,j]  = (s_plus^2 - s_minus^2) / (s_plus^2 + s_minus^2)
        end
    end

    # Eigendecomposition and reassemble
    U_sym = (U + U') / 2 + 1e-6 * I(p)
    U_sym = max.(U_sym, -0.999); for i in 1:p; U_sym[i,i] = 1.0; end

    D_mat = Diagonal(D)
    return D_mat * U_sym * D_mat
end

"""
    robust_correlation(x, y; method) → Float64

Robust correlation between two vectors.
`method`: :pearson_huber, :spearman, :tau_b
"""
function robust_correlation(x::Vector{Float64}, y::Vector{Float64};
                              method::Symbol=:spearman)
    n = length(x); n == length(y) || error("x, y must be same length")

    if method == :spearman
        # Spearman rank correlation
        rx = invperm(sortperm(x)) |> float
        ry = invperm(sortperm(y)) |> float
        return cor(rx, ry)

    elseif method == :tau_b
        # Kendall's tau-b
        nc = nd = 0
        for i in 1:n, j in i+1:n
            dx = x[i] - x[j]; dy = y[i] - y[j]
            sign_prod = sign(dx * dy)
            nc += sign_prod > 0 ? 1 : 0
            nd += sign_prod < 0 ? 1 : 0
        end
        total_pairs = n * (n-1) ÷ 2
        return total_pairs > 0 ? (nc - nd) / total_pairs : 0.0

    else  # pearson with Huber-adjusted location
        mu_x = robust_mean(x; method=:huber)
        mu_y = robust_mean(y; method=:huber)
        s_x  = mad(x) + 1e-10
        s_y  = mad(y) + 1e-10
        return cov(x .- mu_x, y .- mu_y) / (s_x * s_y)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. INFLUENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

"""
    influence_function(estimator, data, points; epsilon) → Vector{Float64}

Empirical influence function for a given estimator.
IF(x; T, F) ≈ [T(data + ε*point) - T(data)] / ε for each point.

`estimator`: function from Vector{Float64} to Float64
`points`: grid of contamination points
`epsilon`: contamination mass (default 0.01)
"""
function influence_function(estimator::Function, data::Vector{Float64},
                              points::Vector{Float64};
                              epsilon::Float64=0.01)
    T_data = estimator(data)
    n = length(data)
    IF = Float64[]

    for x in points
        # Add contaminating point to data
        data_new = vcat(data, x)
        # Weighted estimator: (1-ε) * T(data) + ε * T({x}) approximately
        T_new = estimator(data_new)
        push!(IF, (T_new - T_data) / epsilon)
    end

    return IF
end

"""
    influence_function_mean(x_grid) → Vector{Float64}

Analytical influence function for the mean: IF(x; mean, F) = x - μ.
"""
function influence_function_mean(data::Vector{Float64}, x_grid::Vector{Float64})
    mu = mean(data)
    return x_grid .- mu
end

"""
    influence_function_median(x_grid, data) → Vector{Float64}

Analytical influence function for the median: IF(x; median, F) = sign(x-m) / (2f(m))
where f(m) is the density at the median (estimated via KDE).
"""
function influence_function_median(data::Vector{Float64}, x_grid::Vector{Float64})
    med = median(data)
    h   = 1.06 * std(data) * length(data)^(-0.2) + 1e-8   # Silverman's rule
    # KDE at median
    f_med = mean(exp.(-0.5 .* ((data .- med) ./ h).^2)) / (h * sqrt(2π))
    return sign.(x_grid .- med) ./ (2 * f_med + 1e-10)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. BREAKDOWN POINT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

"""
    breakdown_point(estimator_name, n) → Float64

Theoretical asymptotic breakdown point for standard estimators.

Breakdown point: the fraction of observations that can be adversarially
corrupted before the estimator becomes arbitrarily bad.
"""
function breakdown_point(estimator_name::Symbol, n::Int=Inf)
    if estimator_name == :mean
        return n == Inf ? 0.0 : 1/n          # breaks down with 1 outlier
    elseif estimator_name == :median
        return 0.5                            # 50% breakdown
    elseif estimator_name == :trimmed_10
        return 0.10                           # 10% breakdown
    elseif estimator_name == :trimmed_25
        return 0.25                           # 25% breakdown
    elseif estimator_name == :mad
        return 0.5                            # 50% breakdown
    elseif estimator_name == :huber
        return 0.0                            # location: 0 (but scale-adjusted)
    elseif estimator_name == :bisquare
        return 0.5                            # with high-breakdown scale
    elseif estimator_name == :mcd_50
        return 0.5                            # h = 50%
    elseif estimator_name == :mcd_75
        return 0.25                           # h = 75%
    elseif estimator_name == :lms
        return 0.5                            # LMS: 50%
    elseif estimator_name == :lts_50
        return 0.5                            # LTS with h=50%: 50%
    elseif estimator_name == :theil_sen
        return 0.293                          # Theil-Sen: ~29.3%
    elseif estimator_name == :rpca
        return 0.5                            # theoretical for RPCA
    else
        return NaN
    end
end

"""
    breakdown_comparison() → Vector{NamedTuple}

Return breakdown point table for common estimators.
"""
function breakdown_comparison()
    estimators = [:mean, :median, :trimmed_10, :trimmed_25, :mad,
                  :mcd_75, :mcd_50, :lms, :theil_sen, :bisquare, :rpca]
    return [(name=e, breakdown_pct=breakdown_point(e)*100) for e in estimators]
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. ROBUST HUBER PCA
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_pca_huber(X, k; k_huber) → (components, scores, explained_var)

Robust PCA using Huber-weighted covariance matrix.
More resistant to outliers than standard PCA.
"""
function robust_pca_huber(X::Matrix{Float64}, k::Int;
                            k_huber::Float64=1.345, max_iter::Int=50)
    n, p = size(X)

    # Robust centering
    mu = [robust_mean(X[:,j]; method=:huber) for j in 1:p]
    Xc = X .- mu'

    # Iterative robust covariance estimation
    S = cov(Xc) + 1e-8 * I(p)

    for _ in 1:max_iter
        # Mahalanobis distances
        Sinv = inv(S)
        mah  = [sqrt(dot(Xc[i,:], Sinv * Xc[i,:])) for i in 1:n]

        # Huber weights based on Mahalanobis distance
        ws   = [huber_loss(m / sqrt(p + 0.0), k_huber).weight for m in mah]
        ws   = max.(ws, 1e-8)

        # Weighted covariance
        S_new = zeros(p, p)
        w_sum = sum(ws)
        for i in 1:n
            S_new .+= (ws[i] / w_sum) .* (Xc[i,:] * Xc[i,:]')
        end
        S_new .+= 1e-8 * I(p)

        delta = norm(S_new - S) / (norm(S) + 1e-8)
        S = S_new
        delta < 1e-5 && break
    end

    # Eigendecomposition
    S_sym = (S + S') / 2
    eig   = eigen(Symmetric(S_sym))

    # Sort by descending eigenvalue
    order  = sortperm(eig.values, rev=true)
    lambda = eig.values[order]
    V      = eig.vectors[:, order]

    # Return top k components
    V_k    = V[:, 1:min(k, p)]
    scores = Xc * V_k
    total_var   = sum(abs.(lambda))
    explained   = cumsum(abs.(lambda[1:min(k,p)])) ./ (total_var + 1e-10)

    return (components=V_k, scores=scores, eigenvalues=lambda[1:min(k,p)],
            explained_variance=explained, center=mu)
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. ROBUST PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_calmar(returns; lookback) → Float64

Robust Calmar ratio: median annual return / maximum drawdown.
"""
function robust_calmar(returns::Vector{Float64}; ann_factor::Float64=252.0)
    isempty(returns) && return NaN
    ann_ret  = median(returns) * ann_factor
    pv       = cumprod(1.0 .+ returns)
    peak     = pv[1]; mdd = 0.0
    for v in pv; peak = max(peak,v); mdd = max(mdd, (peak-v)/peak); end
    return mdd < 1e-6 ? NaN : ann_ret / mdd
end

"""
    robust_sortino(returns, target; method) → Float64

Robust Sortino ratio using robust downside deviation.
"""
function robust_sortino(returns::Vector{Float64}, target::Float64=0.0;
                         ann_factor::Float64=252.0)
    excess     = returns .- target
    downside   = excess[excess .< 0]
    isempty(downside) && return Inf
    dd_vol     = mad(downside) * sqrt(ann_factor)  # robust downside vol
    ann_excess = median(excess) * ann_factor
    return dd_vol < 1e-10 ? NaN : ann_excess / dd_vol
end

"""
    omega_ratio(returns, threshold) → Float64

Omega ratio: ∫_threshold^∞ (1-F(r)) dr / ∫_{-∞}^threshold F(r) dr
Empirical estimate from sorted returns.
"""
function omega_ratio(returns::Vector{Float64}, threshold::Float64=0.0)
    gains  = sum(max.(returns .- threshold, 0.0))
    losses = sum(max.(threshold .- returns, 0.0))
    return losses < 1e-10 ? Inf : gains / losses
end

"""
    robust_information_ratio(active_returns; window) → Float64

Information ratio using robust tracking error (MAD-based).
"""
function robust_information_ratio(active_returns::Vector{Float64};
                                   ann_factor::Float64=252.0)
    isempty(active_returns) && return NaN
    active_ret = median(active_returns) * ann_factor
    te         = mad(active_returns)    * sqrt(ann_factor)
    return te < 1e-10 ? NaN : active_ret / te
end

end  # module RobustStatistics
