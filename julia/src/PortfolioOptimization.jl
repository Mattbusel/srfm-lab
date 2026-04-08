###############################################################################
# PortfolioOptimization.jl
#
# Full portfolio optimization suite: mean-variance, Black-Litterman, HRP,
# ERC, CVaR, max diversification, robust Markowitz, Kelly, risk budgeting,
# transaction-cost-aware, tax-lot-aware, multi-period, efficient frontier.
#
# Dependencies: LinearAlgebra, Statistics, Random  (stdlib only)
###############################################################################

module PortfolioOptimization

using LinearAlgebra, Statistics, Random

export mean_variance_optimize, active_set_qp, efficient_frontier
export black_litterman, black_litterman_tau_calibrate
export hrp_portfolio, hrp_distance_matrix, quasi_diag, recursive_bisection
export erc_portfolio, erc_newton
export cvar_optimize, cvar_lp_relaxation
export max_diversification_portfolio
export min_correlation_portfolio
export ledoit_wolf_shrinkage, resampled_efficient_frontier, robust_markowitz
export kelly_criterion, fractional_kelly, constrained_kelly
export risk_budget_portfolio, factor_risk_budget
export transaction_cost_optimize, tc_aware_rebalance
export tax_lot_rebalance, tax_lot_harvest
export multi_period_mv, dp_multi_period
export parametric_efficient_frontier

# ─────────────────────────────────────────────────────────────────────────────
# §1  Core Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Annualise return and covariance given observation frequency."""
function annualize(mu::AbstractVector{T}, Sigma::AbstractMatrix{T};
                   freq::Int=252) where T<:Real
    mu_ann = mu .* freq
    Sigma_ann = Sigma .* freq
    return mu_ann, Sigma_ann
end

"""Clamp weights to [lb, ub] element-wise."""
function clamp_weights!(w::AbstractVector{T}, lb::T, ub::T) where T<:Real
    @inbounds for i in eachindex(w)
        w[i] = clamp(w[i], lb, ub)
    end
    w
end

"""Normalize weights to sum to target (default 1)."""
function normalize_weights!(w::AbstractVector{T}; target::T=one(T)) where T<:Real
    s = sum(w)
    if abs(s) > eps(T)
        w .*= target / s
    end
    w
end

"""Compute portfolio return."""
port_return(w::AbstractVector, mu::AbstractVector) = dot(w, mu)

"""Compute portfolio variance."""
port_variance(w::AbstractVector, Sigma::AbstractMatrix) = dot(w, Sigma * w)

"""Compute portfolio volatility."""
port_vol(w::AbstractVector, Sigma::AbstractMatrix) = sqrt(max(port_variance(w, Sigma), 0.0))

"""Nearest symmetric positive-definite matrix via Higham's algorithm."""
function nearest_psd(A::AbstractMatrix{T}; max_iter::Int=100, tol::T=T(1e-10)) where T<:Real
    n = size(A, 1)
    Y = copy(A)
    dS = zeros(T, n, n)
    for _ in 1:max_iter
        R = Y .- dS
        F = eigen(Symmetric(R))
        vals = max.(F.values, zero(T))
        X = F.vectors * Diagonal(vals) * F.vectors'
        dS = X .- R
        Y = (X .+ X') ./ 2
        if norm(Y - X) < tol
            break
        end
    end
    return Symmetric(Y)
end

"""Cholesky with fallback to nearest PSD."""
function safe_cholesky(Sigma::AbstractMatrix{T}) where T<:Real
    try
        return cholesky(Symmetric(Sigma))
    catch
        S = nearest_psd(Sigma)
        return cholesky(S)
    end
end

"""Solve linear system Ax = b robustly."""
function robust_solve(A::AbstractMatrix{T}, b::AbstractVector{T}) where T<:Real
    try
        return A \ b
    catch
        return pinv(A) * b
    end
end

"""Build a block-diagonal matrix from a vector of square matrices."""
function block_diag(blocks::Vector{<:AbstractMatrix{T}}) where T<:Real
    n = sum(size(B, 1) for B in blocks)
    M = zeros(T, n, n)
    offset = 0
    for B in blocks
        s = size(B, 1)
        M[offset+1:offset+s, offset+1:offset+s] .= B
        offset += s
    end
    M
end

"""Exponentially weighted covariance matrix."""
function ewm_cov(returns::AbstractMatrix{T}; span::Int=60) where T<:Real
    n, p = size(returns)
    lambda = 1.0 - 2.0 / (span + 1)
    weights = [lambda^(n - t) for t in 1:n]
    weights ./= sum(weights)
    mu = returns' * weights
    demeaned = returns .- mu'
    Sigma = zeros(T, p, p)
    @inbounds for t in 1:n
        for j in 1:p
            for i in j:p
                Sigma[i, j] += weights[t] * demeaned[t, i] * demeaned[t, j]
            end
        end
    end
    for j in 1:p
        for i in 1:j-1
            Sigma[i, j] = Sigma[j, i]
        end
    end
    Symmetric(Sigma)
end

"""De-noise covariance via Marchenko-Pastur filtering."""
function denoise_cov(Sigma::AbstractMatrix{T}, n_obs::Int;
                     bandwidth::T=T(0.01)) where T<:Real
    p = size(Sigma, 1)
    q = n_obs / p
    F = eigen(Symmetric(Sigma))
    vals = F.values
    vecs = F.vectors
    sigma2 = median(vals)
    lambda_plus = sigma2 * (1.0 + 1.0/q + 2.0*sqrt(1.0/q))
    cleaned = copy(vals)
    for i in eachindex(cleaned)
        if cleaned[i] < lambda_plus
            cleaned[i] = sigma2
        end
    end
    tr_orig = sum(vals)
    cleaned .*= tr_orig / sum(cleaned)
    Sigma_clean = vecs * Diagonal(cleaned) * vecs'
    Symmetric((Sigma_clean .+ Sigma_clean') ./ 2)
end

"""Detone covariance by removing first k eigenvectors (market modes)."""
function detone_cov(Sigma::AbstractMatrix{T}; k::Int=1) where T<:Real
    F = eigen(Symmetric(Sigma))
    vals = copy(F.values)
    n = length(vals)
    for i in (n-k+1):n
        vals[i] = zero(T)
    end
    Sigma_det = F.vectors * Diagonal(vals) * F.vectors'
    Symmetric((Sigma_det .+ Sigma_det') ./ 2)
end

# ─────────────────────────────────────────────────────────────────────────────
# §2  Quadratic Programming – Active Set Method
# ─────────────────────────────────────────────────────────────────────────────

"""
    ActiveSetQP

Solve:  min  0.5 x' H x + f' x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             lb <= x <= ub
using primal active-set method.
"""
struct ActiveSetQP{T<:Real}
    H::Matrix{T}
    f::Vector{T}
    A_eq::Matrix{T}
    b_eq::Vector{T}
    A_ineq::Matrix{T}
    b_ineq::Vector{T}
    lb::Vector{T}
    ub::Vector{T}
end

function ActiveSetQP(H::AbstractMatrix{T}, f::AbstractVector{T};
                     A_eq::AbstractMatrix{T}=zeros(T,0,length(f)),
                     b_eq::AbstractVector{T}=zeros(T,0),
                     A_ineq::AbstractMatrix{T}=zeros(T,0,length(f)),
                     b_ineq::AbstractVector{T}=zeros(T,0),
                     lb::AbstractVector{T}=fill(T(-Inf), length(f)),
                     ub::AbstractVector{T}=fill(T(Inf), length(f))) where T<:Real
    ActiveSetQP{T}(Matrix(H), Vector(f), Matrix(A_eq), Vector(b_eq),
                   Matrix(A_ineq), Vector(b_ineq), Vector(lb), Vector(ub))
end

"""Convert bound constraints to inequality constraints."""
function _bounds_to_ineq(qp::ActiveSetQP{T}) where T<:Real
    n = length(qp.f)
    bound_rows_lower = Matrix{T}(undef, 0, n)
    bound_rhs_lower = Vector{T}()
    bound_rows_upper = Matrix{T}(undef, 0, n)
    bound_rhs_upper = Vector{T}()
    for i in 1:n
        if isfinite(qp.lb[i])
            row = zeros(T, 1, n)
            row[1, i] = -one(T)
            bound_rows_lower = vcat(bound_rows_lower, row)
            push!(bound_rhs_lower, -qp.lb[i])
        end
        if isfinite(qp.ub[i])
            row = zeros(T, 1, n)
            row[1, i] = one(T)
            bound_rows_upper = vcat(bound_rows_upper, row)
            push!(bound_rhs_upper, qp.ub[i])
        end
    end
    A_all = vcat(qp.A_ineq, bound_rows_lower, bound_rows_upper)
    b_all = vcat(qp.b_ineq, bound_rhs_lower, bound_rhs_upper)
    return A_all, b_all
end

"""Find feasible starting point via simple projection."""
function _feasible_start(qp::ActiveSetQP{T}) where T<:Real
    n = length(qp.f)
    x = zeros(T, n)
    if length(qp.b_eq) > 0 && size(qp.A_eq, 1) > 0
        x = pinv(qp.A_eq) * qp.b_eq
    end
    for i in 1:n
        x[i] = clamp(x[i], qp.lb[i], qp.ub[i])
    end
    if length(qp.b_eq) > 0 && size(qp.A_eq, 1) > 0
        res = qp.A_eq * x .- qp.b_eq
        if norm(res) > T(1e-8)
            for iter in 1:50
                dx = pinv(qp.A_eq) * (qp.b_eq .- qp.A_eq * x)
                x .+= dx
                for i in 1:n
                    x[i] = clamp(x[i], qp.lb[i], qp.ub[i])
                end
                if norm(qp.A_eq * x .- qp.b_eq) < T(1e-10)
                    break
                end
            end
        end
    end
    x
end

"""
    active_set_qp(H, f; kwargs...) -> x, obj_val, converged

Solve convex QP via active-set method.
"""
function active_set_qp(H::AbstractMatrix{T}, f::AbstractVector{T};
                       A_eq::AbstractMatrix{T}=zeros(T,0,length(f)),
                       b_eq::AbstractVector{T}=zeros(T,0),
                       A_ineq::AbstractMatrix{T}=zeros(T,0,length(f)),
                       b_ineq::AbstractVector{T}=zeros(T,0),
                       lb::AbstractVector{T}=fill(T(-Inf), length(f)),
                       ub::AbstractVector{T}=fill(T(Inf), length(f)),
                       max_iter::Int=5000,
                       tol::T=T(1e-10)) where T<:Real
    qp = ActiveSetQP(H, f; A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq,
                     lb=lb, ub=ub)
    A_all, b_all = _bounds_to_ineq(qp)
    n = length(f)
    m_eq = size(qp.A_eq, 1)
    m_ineq = size(A_all, 1)
    x = _feasible_start(qp)
    active = Set{Int}()
    for i in 1:m_ineq
        if abs(dot(A_all[i,:], x) - b_all[i]) < tol
            push!(active, i)
        end
    end
    converged = false
    for iter in 1:max_iter
        g = H * x .+ f
        n_active = length(active)
        active_idx = sort(collect(active))
        n_constr = m_eq + n_active
        if n_constr == 0
            d = -(H \ g)
        else
            A_work = zeros(T, n_constr, n)
            if m_eq > 0
                A_work[1:m_eq, :] .= qp.A_eq
            end
            for (k, idx) in enumerate(active_idx)
                A_work[m_eq + k, :] .= A_all[idx, :]
            end
            KKT = vcat(hcat(H, A_work'), hcat(A_work, zeros(T, n_constr, n_constr)))
            rhs = vcat(-g, zeros(T, n_constr))
            sol = robust_solve(KKT, rhs)
            d = sol[1:n]
            lambdas = sol[n+1:end]
            if norm(d) < tol
                if n_active == 0
                    converged = true
                    break
                end
                ineq_lambdas = lambdas[m_eq+1:end]
                min_lam, min_idx = findmin(ineq_lambdas)
                if min_lam >= -tol
                    converged = true
                    break
                end
                delete!(active, active_idx[min_idx])
                continue
            end
        end
        alpha = one(T)
        blocking = -1
        for i in 1:m_ineq
            if i in active
                continue
            end
            ad = dot(A_all[i,:], d)
            if ad > tol
                slack = b_all[i] - dot(A_all[i,:], x)
                ratio = slack / ad
                if ratio < alpha - tol
                    alpha = ratio
                    blocking = i
                end
            end
        end
        alpha = max(alpha, zero(T))
        x .+= alpha .* d
        if blocking >= 0
            push!(active, blocking)
        end
        if alpha < tol && blocking < 0
            converged = true
            break
        end
    end
    for i in 1:n
        x[i] = clamp(x[i], qp.lb[i], qp.ub[i])
    end
    obj_val = T(0.5) * dot(x, H * x) + dot(f, x)
    return x, obj_val, converged
end

"""Gradient-projection QP solver for box-constrained problems."""
function gradient_projection_qp(H::AbstractMatrix{T}, f::AbstractVector{T},
                                lb::AbstractVector{T}, ub::AbstractVector{T};
                                max_iter::Int=10000, tol::T=T(1e-10)) where T<:Real
    n = length(f)
    x = (lb .+ ub) ./ 2
    for i in 1:n
        if !isfinite(x[i])
            x[i] = zero(T)
        end
    end
    alpha = T(1.0) / (opnorm(H) + T(1e-8))
    for iter in 1:max_iter
        g = H * x .+ f
        x_new = x .- alpha .* g
        for i in 1:n
            x_new[i] = clamp(x_new[i], lb[i], ub[i])
        end
        if norm(x_new .- x) < tol
            x = x_new
            break
        end
        x = x_new
    end
    obj = T(0.5) * dot(x, H * x) + dot(f, x)
    return x, obj
end

# ─────────────────────────────────────────────────────────────────────────────
# §3  Mean-Variance Optimization
# ─────────────────────────────────────────────────────────────────────────────

"""
    mean_variance_optimize(mu, Sigma; kwargs...) -> weights

Standard Markowitz mean-variance optimization.
"""
function mean_variance_optimize(mu::AbstractVector{T}, Sigma::AbstractMatrix{T};
                                target_return::Union{Nothing,T}=nothing,
                                risk_aversion::T=T(1.0),
                                lb::T=T(0.0), ub::T=T(1.0),
                                long_only::Bool=true,
                                max_iter::Int=5000) where T<:Real
    n = length(mu)
    H = risk_aversion .* Sigma
    f = -mu
    A_eq = ones(T, 1, n)
    b_eq = [one(T)]
    lb_vec = fill(long_only ? lb : T(-Inf), n)
    ub_vec = fill(ub, n)
    A_ineq = zeros(T, 0, n)
    b_ineq = zeros(T, 0)
    if target_return !== nothing
        A_ineq = reshape(-mu, 1, n)
        b_ineq = [-target_return]
    end
    w, _, _ = active_set_qp(H, f; A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq,
                            b_ineq=b_ineq, lb=lb_vec, ub=ub_vec, max_iter=max_iter)
    normalize_weights!(w)
    return w
end

"""Global minimum variance portfolio."""
function min_variance_portfolio(Sigma::AbstractMatrix{T};
                                lb::T=T(0.0), ub::T=T(1.0)) where T<:Real
    n = size(Sigma, 1)
    mu = zeros(T, n)
    mean_variance_optimize(mu, Sigma; risk_aversion=T(1.0), lb=lb, ub=ub)
end

"""Maximum return portfolio subject to volatility constraint."""
function max_return_portfolio(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                             max_vol::T; lb::T=T(0.0), ub::T=T(1.0)) where T<:Real
    n = length(mu)
    # Bisection on risk aversion
    gamma_lo, gamma_hi = T(1e-6), T(1e6)
    w_best = fill(one(T)/n, n)
    for _ in 1:100
        gamma = (gamma_lo + gamma_hi) / 2
        w = mean_variance_optimize(mu, Sigma; risk_aversion=gamma, lb=lb, ub=ub)
        vol = port_vol(w, Sigma)
        if vol > max_vol
            gamma_lo = gamma
        else
            gamma_hi = gamma
            w_best = w
        end
        if abs(vol - max_vol) / max(max_vol, T(1e-8)) < T(1e-6)
            break
        end
    end
    w_best
end

"""Tangency portfolio (maximum Sharpe ratio)."""
function tangency_portfolio(mu::AbstractVector{T}, Sigma::AbstractMatrix{T};
                            rf::T=T(0.0), lb::T=T(0.0), ub::T=T(1.0)) where T<:Real
    excess = mu .- rf
    n = length(mu)
    gamma_lo, gamma_hi = T(1e-6), T(1e4)
    best_sharpe = T(-Inf)
    w_best = fill(one(T)/n, n)
    for _ in 1:200
        gamma = (gamma_lo + gamma_hi) / 2
        w = mean_variance_optimize(mu, Sigma; risk_aversion=gamma, lb=lb, ub=ub)
        ret = dot(w, excess)
        vol = port_vol(w, Sigma)
        sharpe = vol > T(1e-12) ? ret / vol : T(-Inf)
        if sharpe > best_sharpe
            best_sharpe = sharpe
            w_best = copy(w)
        end
        # Heuristic: lower gamma -> more aggressive -> higher return
        gamma_hi = gamma  # narrow from above
        if abs(gamma_hi - gamma_lo) < T(1e-10)
            break
        end
    end
    # More precise: grid search
    for log_gamma in range(-6.0, 4.0, length=500)
        gamma = T(10.0^log_gamma)
        w = mean_variance_optimize(mu, Sigma; risk_aversion=gamma, lb=lb, ub=ub)
        ret = dot(w, excess)
        vol = port_vol(w, Sigma)
        sharpe = vol > T(1e-12) ? ret / vol : T(-Inf)
        if sharpe > best_sharpe
            best_sharpe = sharpe
            w_best = copy(w)
        end
    end
    w_best
end

# ─────────────────────────────────────────────────────────────────────────────
# §4  Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

"""
    efficient_frontier(mu, Sigma; n_points=50) -> returns, vols, weights_matrix

Compute the efficient frontier by tracing target returns.
"""
function efficient_frontier(mu::AbstractVector{T}, Sigma::AbstractMatrix{T};
                            n_points::Int=50, lb::T=T(0.0), ub::T=T(1.0)) where T<:Real
    n = length(mu)
    w_min = min_variance_portfolio(Sigma; lb=lb, ub=ub)
    ret_min = dot(w_min, mu)
    ret_max = maximum(mu)
    targets = range(ret_min, ret_max, length=n_points)
    rets = Vector{T}(undef, n_points)
    vols = Vector{T}(undef, n_points)
    W = Matrix{T}(undef, n, n_points)
    for (k, tgt) in enumerate(targets)
        w = mean_variance_optimize(mu, Sigma; target_return=tgt, risk_aversion=T(1e-6),
                                   lb=lb, ub=ub)
        rets[k] = dot(w, mu)
        vols[k] = port_vol(w, Sigma)
        W[:, k] = w
    end
    return rets, vols, W
end

"""Parametric efficient frontier using two-fund theorem (unconstrained)."""
function parametric_efficient_frontier(mu::AbstractVector{T}, Sigma::AbstractMatrix{T};
                                       n_points::Int=100) where T<:Real
    n = length(mu)
    ones_n = ones(T, n)
    Sigma_inv = inv(Symmetric(Sigma))
    a = dot(ones_n, Sigma_inv * mu)
    b = dot(mu, Sigma_inv * mu)
    c = dot(ones_n, Sigma_inv * ones_n)
    d = b * c - a^2
    g = (Sigma_inv * (b .* ones_n .- a .* mu)) ./ d
    h = (Sigma_inv * (c .* mu .- a .* ones_n)) ./ d
    ret_min = a / c
    ret_max = maximum(mu) * T(1.2)
    targets = range(ret_min, ret_max, length=n_points)
    rets = Vector{T}(undef, n_points)
    vols = Vector{T}(undef, n_points)
    W = Matrix{T}(undef, n, n_points)
    for (k, tgt) in enumerate(targets)
        w = g .+ h .* tgt
        rets[k] = dot(w, mu)
        vols[k] = port_vol(w, Sigma)
        W[:, k] = w
    end
    return rets, vols, W
end

# ─────────────────────────────────────────────────────────────────────────────
# §5  Black-Litterman
# ─────────────────────────────────────────────────────────────────────────────

"""
    black_litterman(Sigma, market_cap_weights, P, Q; kwargs...) -> posterior_mu, posterior_Sigma

Black-Litterman model with view matrix P, view returns Q.
"""
function black_litterman(Sigma::AbstractMatrix{T},
                         w_mkt::AbstractVector{T},
                         P::AbstractMatrix{T},
                         Q::AbstractVector{T};
                         tau::T=T(0.05),
                         risk_aversion::T=T(2.5),
                         Omega::Union{Nothing, AbstractMatrix{T}}=nothing,
                         confidence::Union{Nothing, AbstractVector{T}}=nothing) where T<:Real
    n = size(Sigma, 1)
    pi_eq = risk_aversion .* Sigma * w_mkt
    tau_Sigma = tau .* Sigma
    if Omega === nothing
        if confidence !== nothing
            k = length(Q)
            omega_diag = [(1.0 / confidence[i] - 1.0) * dot(P[i,:], tau_Sigma * P[i,:])
                         for i in 1:k]
            Omega = Diagonal(omega_diag)
        else
            Omega = Diagonal(diag(P * tau_Sigma * P'))
        end
    end
    M = tau_Sigma * P' * inv(P * tau_Sigma * P' .+ Omega)
    mu_bl = pi_eq .+ M * (Q .- P * pi_eq)
    Sigma_bl = (I(n) .- M * P) * tau_Sigma
    Sigma_posterior = Sigma .+ Sigma_bl
    return mu_bl, Symmetric(Sigma_posterior)
end

"""Calibrate tau via cross-validation on held-out returns."""
function black_litterman_tau_calibrate(returns::AbstractMatrix{T},
                                       w_mkt::AbstractVector{T},
                                       P::AbstractMatrix{T},
                                       Q::AbstractVector{T};
                                       tau_grid::AbstractVector{T}=T.(0.01:0.01:0.2),
                                       risk_aversion::T=T(2.5),
                                       n_folds::Int=5) where T<:Real
    n_obs, n_assets = size(returns)
    fold_size = div(n_obs, n_folds)
    best_tau = tau_grid[1]
    best_score = T(-Inf)
    for tau in tau_grid
        total_score = zero(T)
        for fold in 1:n_folds
            test_start = (fold - 1) * fold_size + 1
            test_end = min(fold * fold_size, n_obs)
            train_idx = vcat(1:test_start-1, test_end+1:n_obs)
            if length(train_idx) < n_assets + 1
                continue
            end
            mu_train = vec(mean(returns[train_idx, :]; dims=1))
            Sigma_train = cov(returns[train_idx, :])
            mu_bl, Sigma_bl = black_litterman(Sigma_train, w_mkt, P, Q;
                                              tau=tau, risk_aversion=risk_aversion)
            w_bl = mean_variance_optimize(mu_bl, Sigma_bl; risk_aversion=risk_aversion)
            test_rets = returns[test_start:test_end, :] * w_bl
            total_score += mean(test_rets) / (std(test_rets) + T(1e-10))
        end
        if total_score > best_score
            best_score = total_score
            best_tau = tau
        end
    end
    return best_tau
end

"""Black-Litterman with tilts from an alpha signal."""
function bl_alpha_tilt(Sigma::AbstractMatrix{T}, w_mkt::AbstractVector{T},
                       alpha::AbstractVector{T};
                       tau::T=T(0.05), risk_aversion::T=T(2.5),
                       tilt_strength::T=T(0.1)) where T<:Real
    n = length(w_mkt)
    P = Matrix{T}(I, n, n)
    pi_eq = risk_aversion .* Sigma * w_mkt
    Q = pi_eq .+ tilt_strength .* alpha
    return black_litterman(Sigma, w_mkt, P, Q; tau=tau, risk_aversion=risk_aversion)
end

# ─────────────────────────────────────────────────────────────────────────────
# §6  Hierarchical Risk Parity (HRP)
# ─────────────────────────────────────────────────────────────────────────────

"""Correlation-based distance matrix."""
function hrp_distance_matrix(corr::AbstractMatrix{T}) where T<:Real
    n = size(corr, 1)
    D = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n
        for i in 1:n
            D[i,j] = sqrt(max(T(0.5) * (1 - corr[i,j]), zero(T)))
        end
    end
    D
end

"""Single-linkage hierarchical clustering."""
function _single_linkage(D::AbstractMatrix{T}) where T<:Real
    n = size(D, 1)
    cluster_id = collect(1:n)
    clusters = Dict{Int, Vector{Int}}(i => [i] for i in 1:n)
    next_id = n + 1
    merge_order = Vector{Tuple{Int,Int,T}}()
    active = Set(1:n)
    dist = copy(D)
    for i in 1:n
        dist[i,i] = T(Inf)
    end
    for _ in 1:n-1
        min_d = T(Inf)
        mi, mj = 0, 0
        for i in active
            for j in active
                if i < j && dist[i,j] < min_d
                    min_d = dist[i,j]
                    mi, mj = i, j
                end
            end
        end
        if mi == 0
            break
        end
        push!(merge_order, (mi, mj, min_d))
        new_cluster = vcat(clusters[mi], clusters[mj])
        clusters[next_id] = new_cluster
        delete!(active, mi)
        delete!(active, mj)
        for k in active
            d_new = min(dist[mi, k], dist[mj, k])
            dist[next_id > size(dist,1) ? mi : next_id, k] = d_new
            dist[k, next_id > size(dist,1) ? mi : next_id] = d_new
        end
        # Reuse mi slot for new cluster
        for k in active
            dist[mi, k] = min(dist[mi, k], dist[mj, k])
            dist[k, mi] = dist[mi, k]
        end
        dist[mj, :] .= T(Inf)
        dist[:, mj] .= T(Inf)
        clusters[mi] = new_cluster
        delete!(clusters, mj)
        push!(active, mi)  # keep mi as the merged cluster
        delete!(active, mj)
    end
    return merge_order, clusters
end

"""Quasi-diagonalization: seriation of a dendrogram."""
function quasi_diag(corr::AbstractMatrix{T}) where T<:Real
    n = size(corr, 1)
    D = hrp_distance_matrix(corr)
    merge_order, clusters = _single_linkage(D)
    # Extract leaf order from the last surviving cluster
    if length(clusters) == 1
        return first(values(clusters))
    end
    # Fallback: return keys of remaining active cluster
    all_items = Int[]
    for (_, v) in clusters
        append!(all_items, v)
    end
    return unique(all_items)[1:n]
end

"""Recursive bisection for HRP weight allocation."""
function recursive_bisection(Sigma::AbstractMatrix{T},
                             sorted_idx::AbstractVector{Int}) where T<:Real
    n = size(Sigma, 1)
    w = ones(T, n)
    cluster_items = [sorted_idx]
    while !isempty(cluster_items)
        next_clusters = Vector{Vector{Int}}()
        for items in cluster_items
            if length(items) <= 1
                continue
            end
            mid = div(length(items), 2)
            left = items[1:mid]
            right = items[mid+1:end]
            # Inverse-variance allocation within each cluster
            var_left = _cluster_var(Sigma, left)
            var_right = _cluster_var(Sigma, right)
            alpha = one(T) - var_left / (var_left + var_right + T(1e-16))
            for i in left
                w[i] *= alpha
            end
            for i in right
                w[i] *= (one(T) - alpha)
            end
            if length(left) > 1
                push!(next_clusters, left)
            end
            if length(right) > 1
                push!(next_clusters, right)
            end
        end
        cluster_items = next_clusters
    end
    normalize_weights!(w)
    return w
end

function _cluster_var(Sigma::AbstractMatrix{T}, idx::AbstractVector{Int}) where T<:Real
    sub_Sigma = Sigma[idx, idx]
    n = length(idx)
    if n == 1
        return sub_Sigma[1,1]
    end
    ivp = T(1.0) ./ diag(sub_Sigma)
    ivp ./= sum(ivp)
    return dot(ivp, sub_Sigma * ivp)
end

"""
    hrp_portfolio(returns) -> weights

Full HRP pipeline.
"""
function hrp_portfolio(returns::AbstractMatrix{T}) where T<:Real
    Sigma = cov(returns)
    corr = cor(returns)
    sorted_idx = quasi_diag(corr)
    w = recursive_bisection(Sigma, sorted_idx)
    return w
end

"""HRP with custom distance metric."""
function hrp_portfolio(returns::AbstractMatrix{T},
                       dist_func::Function) where T<:Real
    Sigma = cov(returns)
    corr = cor(returns)
    n = size(corr, 1)
    D = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        D[i,j] = dist_func(corr[i,:], corr[j,:])
    end
    merge_order, clusters = _single_linkage(D)
    all_items = Int[]
    for (_, v) in clusters
        append!(all_items, v)
    end
    sorted_idx = unique(all_items)[1:n]
    recursive_bisection(Sigma, sorted_idx)
end

# ─────────────────────────────────────────────────────────────────────────────
# §7  Equal Risk Contribution (ERC)
# ─────────────────────────────────────────────────────────────────────────────

"""Risk contribution of each asset."""
function risk_contributions(w::AbstractVector{T}, Sigma::AbstractMatrix{T}) where T<:Real
    sigma_p = port_vol(w, Sigma)
    mrc = Sigma * w ./ max(sigma_p, T(1e-16))
    rc = w .* mrc
    return rc
end

"""
    erc_newton(Sigma; kwargs...) -> weights

Equal Risk Contribution portfolio via Newton's method.
"""
function erc_newton(Sigma::AbstractMatrix{T};
                    max_iter::Int=500, tol::T=T(1e-10),
                    lb::T=T(1e-6)) where T<:Real
    n = size(Sigma, 1)
    w = fill(one(T) / n, n)
    target_rc = one(T) / n
    for iter in 1:max_iter
        sigma_p = port_vol(w, Sigma)
        if sigma_p < T(1e-16)
            break
        end
        Sigma_w = Sigma * w
        rc = w .* Sigma_w ./ sigma_p
        rc_frac = rc ./ (sum(rc) + T(1e-16))
        grad = rc_frac .- target_rc
        if norm(grad) < tol
            break
        end
        # Approximate Jacobian
        J = Matrix{T}(undef, n, n)
        for j in 1:n
            for i in 1:n
                if i == j
                    J[i,j] = (Sigma_w[i] + w[i] * Sigma[i,i]) / sigma_p -
                              w[i] * Sigma_w[i] * Sigma_w[j] / sigma_p^3
                else
                    J[i,j] = w[i] * Sigma[i,j] / sigma_p -
                              w[i] * Sigma_w[i] * Sigma_w[j] / sigma_p^3
                end
            end
        end
        J_norm = J ./ (sum(rc) + T(1e-16))
        dw = robust_solve(J_norm, -grad)
        # Line search
        alpha = one(T)
        for _ in 1:20
            w_new = w .+ alpha .* dw
            if all(x -> x > lb, w_new)
                break
            end
            alpha *= T(0.5)
        end
        w .+= alpha .* dw
        clamp_weights!(w, lb, T(Inf))
        normalize_weights!(w)
    end
    normalize_weights!(w)
    return w
end

"""ERC portfolio via cyclical coordinate descent."""
function erc_coordinate_descent(Sigma::AbstractMatrix{T};
                                max_iter::Int=5000, tol::T=T(1e-10)) where T<:Real
    n = size(Sigma, 1)
    x = fill(one(T), n)  # Work in log space: w_i = exp(x_i) / sum(exp(x_j))
    for iter in 1:max_iter
        w = exp.(x .- maximum(x))
        w ./= sum(w)
        sigma_p = port_vol(w, Sigma)
        if sigma_p < T(1e-16)
            break
        end
        Sigma_w = Sigma * w
        rc = w .* Sigma_w ./ sigma_p
        target = sigma_p / n
        max_diff = zero(T)
        for i in 1:n
            diff = rc[i] - target
            max_diff = max(max_diff, abs(diff))
            x[i] -= T(0.1) * diff / (target + T(1e-16))
        end
        if max_diff < tol
            break
        end
    end
    w = exp.(x .- maximum(x))
    w ./= sum(w)
    return w
end

"""
    erc_portfolio(Sigma) -> weights

ERC with fallback methods.
"""
function erc_portfolio(Sigma::AbstractMatrix{T}) where T<:Real
    w = erc_newton(Sigma)
    rc = risk_contributions(w, Sigma)
    rc_frac = rc ./ (sum(rc) + T(1e-16))
    target = one(T) / size(Sigma, 1)
    if maximum(abs.(rc_frac .- target)) > T(1e-4)
        w = erc_coordinate_descent(Sigma)
    end
    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# §8  CVaR Optimization
# ─────────────────────────────────────────────────────────────────────────────

"""
Compute CVaR (Expected Shortfall) of portfolio at confidence level alpha.
"""
function portfolio_cvar(w::AbstractVector{T}, returns::AbstractMatrix{T};
                        alpha::T=T(0.05)) where T<:Real
    port_rets = returns * w
    sorted = sort(port_rets)
    n = length(sorted)
    cutoff = max(1, floor(Int, alpha * n))
    -mean(sorted[1:cutoff])
end

"""
    cvar_lp_relaxation(mu, returns; alpha=0.05, target_return=nothing) -> weights

CVaR minimization via LP relaxation (Rockafellar-Uryasev).
Solved iteratively with gradient descent.
"""
function cvar_lp_relaxation(mu::AbstractVector{T}, returns::AbstractMatrix{T};
                            alpha::T=T(0.05),
                            target_return::Union{Nothing,T}=nothing,
                            max_iter::Int=2000, lr::T=T(0.01)) where T<:Real
    n_obs, n_assets = size(returns)
    w = fill(one(T) / n_assets, n_assets)
    zeta = zero(T)  # VaR threshold
    for iter in 1:max_iter
        port_rets = returns * w
        losses = -port_rets
        exceedances = losses .- zeta
        cvar_grad_w = zeros(T, n_assets)
        cvar_grad_zeta = zero(T)
        count_exceed = 0
        for t in 1:n_obs
            if exceedances[t] > 0
                cvar_grad_w .-= returns[t, :] ./ (alpha * n_obs)
                count_exceed += 1
            end
        end
        cvar_grad_zeta = one(T) - count_exceed / (alpha * n_obs)
        # Project gradient for sum-to-one constraint
        cvar_grad_w .-= mean(cvar_grad_w)
        w .-= lr .* cvar_grad_w
        zeta -= lr * cvar_grad_zeta
        # Project onto simplex
        for i in 1:n_assets
            w[i] = max(w[i], T(1e-8))
        end
        normalize_weights!(w)
        if target_return !== nothing
            curr_ret = dot(w, mu)
            if curr_ret < target_return
                # Tilt toward higher return assets
                excess = mu .- mean(mu)
                w .+= T(0.01) .* excess .* (target_return - curr_ret)
                for i in 1:n_assets
                    w[i] = max(w[i], T(1e-8))
                end
                normalize_weights!(w)
            end
        end
    end
    return w
end

"""
    cvar_optimize(mu, returns; kwargs...) -> weights

Main CVaR optimization entry point.
"""
function cvar_optimize(mu::AbstractVector{T}, returns::AbstractMatrix{T};
                       alpha::T=T(0.05),
                       target_return::Union{Nothing,T}=nothing,
                       max_iter::Int=3000) where T<:Real
    cvar_lp_relaxation(mu, returns; alpha=alpha, target_return=target_return,
                       max_iter=max_iter)
end

# ─────────────────────────────────────────────────────────────────────────────
# §9  Max Diversification Portfolio
# ─────────────────────────────────────────────────────────────────────────────

"""Diversification ratio: sum(w*sigma) / portfolio_sigma."""
function diversification_ratio(w::AbstractVector{T}, Sigma::AbstractMatrix{T}) where T<:Real
    sigma_assets = sqrt.(diag(Sigma))
    port_sigma = port_vol(w, Sigma)
    dot(w, sigma_assets) / max(port_sigma, T(1e-16))
end

"""
    max_diversification_portfolio(Sigma; kwargs...) -> weights

Maximize diversification ratio.
"""
function max_diversification_portfolio(Sigma::AbstractMatrix{T};
                                       lb::T=T(0.0), ub::T=T(1.0),
                                       max_iter::Int=5000) where T<:Real
    n = size(Sigma, 1)
    sigma_assets = sqrt.(diag(Sigma))
    # Max diversification = min w'Σw / (w'σ)^2
    # Equivalent to min-var on correlation matrix
    D_inv = Diagonal(one(T) ./ sigma_assets)
    corr = D_inv * Sigma * D_inv
    w = min_variance_portfolio(corr; lb=lb, ub=ub)
    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# §10  Minimum Correlation Portfolio
# ─────────────────────────────────────────────────────────────────────────────

"""
    min_correlation_portfolio(Sigma) -> weights

Minimize average pairwise correlation of portfolio constituents.
"""
function min_correlation_portfolio(Sigma::AbstractMatrix{T};
                                   max_iter::Int=3000, tol::T=T(1e-8)) where T<:Real
    n = size(Sigma, 1)
    sigma_assets = sqrt.(diag(Sigma))
    corr = Sigma ./ (sigma_assets * sigma_assets')
    for i in 1:n
        corr[i,i] = zero(T)
    end
    # Iterative reweighting: penalize highly correlated assets
    w = fill(one(T) / n, n)
    for iter in 1:max_iter
        avg_corr = corr * w
        penalty = avg_corr ./ (sum(abs.(avg_corr)) + T(1e-16))
        w_new = w .- T(0.01) .* penalty
        for i in 1:n
            w_new[i] = max(w_new[i], T(1e-8))
        end
        normalize_weights!(w_new)
        if norm(w_new .- w) < tol
            w = w_new
            break
        end
        w = w_new
    end
    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# §11  Robust Markowitz
# ─────────────────────────────────────────────────────────────────────────────

"""
    ledoit_wolf_shrinkage(returns) -> Sigma_shrunk, shrinkage_intensity

Ledoit-Wolf linear shrinkage toward scaled identity.
"""
function ledoit_wolf_shrinkage(returns::AbstractMatrix{T}) where T<:Real
    n, p = size(returns)
    S = cov(returns)
    mu_S = tr(S) / p
    delta = zero(T)
    for j in 1:p, i in 1:p
        delta += (S[i,j] - (i == j ? mu_S : zero(T)))^2
    end
    delta /= p
    # Estimate squared Frobenius norm of S - mu*I
    beta_sum = zero(T)
    X = returns .- mean(returns; dims=1)
    for k in 1:n
        xk = X[k, :]
        Mk = xk * xk' .- S
        beta_sum += sum(Mk .^ 2)
    end
    beta = beta_sum / (n^2 * p)
    alpha = min(beta / (delta + T(1e-16)), one(T))
    F = mu_S .* Matrix{T}(I, p, p)
    Sigma_shrunk = alpha .* F .+ (one(T) - alpha) .* S
    return Symmetric(Sigma_shrunk), alpha
end

"""Oracle approximating shrinkage (OAS) estimator."""
function oas_shrinkage(returns::AbstractMatrix{T}) where T<:Real
    n, p = size(returns)
    S = cov(returns)
    tr_S = tr(S)
    tr_S2 = tr(S * S)
    mu = tr_S / p
    rho_num = (1 - 2/p) * tr_S2 + tr_S^2
    rho_den = (n + 1 - 2/p) * (tr_S2 - tr_S^2 / p)
    rho = clamp(rho_num / (rho_den + T(1e-16)), zero(T), one(T))
    F = mu .* Matrix{T}(I, p, p)
    Sigma_oas = rho .* F .+ (one(T) - rho) .* S
    return Symmetric(Sigma_oas), rho
end

"""
    resampled_efficient_frontier(mu, Sigma, n_obs; n_resample=500, n_points=50) -> avg_weights

Michaud resampled efficient frontier.
"""
function resampled_efficient_frontier(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                                      n_obs::Int;
                                      n_resample::Int=500, n_points::Int=50,
                                      lb::T=T(0.0), ub::T=T(1.0),
                                      rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n = length(mu)
    L = safe_cholesky(Sigma).L
    W_accum = zeros(T, n, n_points)
    for s in 1:n_resample
        Z = randn(rng, T, n_obs, n)
        sim_returns = repeat(mu', n_obs) .+ Z * L'
        mu_s = vec(mean(sim_returns; dims=1))
        Sigma_s = cov(sim_returns)
        _, _, W_s = efficient_frontier(mu_s, Sigma_s; n_points=n_points, lb=lb, ub=ub)
        W_accum .+= W_s
    end
    W_avg = W_accum ./ n_resample
    for k in 1:n_points
        col = @view W_avg[:, k]
        normalize_weights!(col)
    end
    return W_avg
end

"""
    robust_markowitz(mu, Sigma; kwargs...) -> weights

Robust Markowitz: Ledoit-Wolf shrinkage + optional resampling.
"""
function robust_markowitz(returns::AbstractMatrix{T};
                          risk_aversion::T=T(1.0),
                          shrinkage::Symbol=:ledoit_wolf,
                          resample::Bool=false,
                          n_resample::Int=200,
                          lb::T=T(0.0), ub::T=T(1.0)) where T<:Real
    mu = vec(mean(returns; dims=1))
    Sigma_shrunk, alpha = if shrinkage == :ledoit_wolf
        ledoit_wolf_shrinkage(returns)
    elseif shrinkage == :oas
        oas_shrinkage(returns)
    else
        cov(returns), zero(T)
    end
    if resample
        n_obs = size(returns, 1)
        W = resampled_efficient_frontier(mu, Sigma_shrunk, n_obs;
                                         n_resample=n_resample, lb=lb, ub=ub)
        # Pick portfolio near tangency
        n_points = size(W, 2)
        best_idx = 1
        best_sharpe = T(-Inf)
        for k in 1:n_points
            w = W[:, k]
            ret = dot(w, mu)
            vol = port_vol(w, Sigma_shrunk)
            sr = vol > T(1e-12) ? ret / vol : T(-Inf)
            if sr > best_sharpe
                best_sharpe = sr
                best_idx = k
            end
        end
        return W[:, best_idx]
    else
        return mean_variance_optimize(mu, Sigma_shrunk; risk_aversion=risk_aversion,
                                     lb=lb, ub=ub)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# §12  Kelly Criterion
# ─────────────────────────────────────────────────────────────────────────────

"""
    kelly_criterion(mu, Sigma) -> weights

Full Kelly: maximize E[log(1 + w'r)] ≈ w'μ - 0.5 w'Σw.
"""
function kelly_criterion(mu::AbstractVector{T}, Sigma::AbstractMatrix{T}) where T<:Real
    w = Sigma \ mu
    return w
end

"""Fractional Kelly."""
function fractional_kelly(mu::AbstractVector{T}, Sigma::AbstractMatrix{T};
                          fraction::T=T(0.5)) where T<:Real
    w_full = kelly_criterion(mu, Sigma)
    return fraction .* w_full
end

"""Constrained Kelly (long-only, sum-to-one, bounded)."""
function constrained_kelly(mu::AbstractVector{T}, Sigma::AbstractMatrix{T};
                           fraction::T=T(0.5),
                           lb::T=T(0.0), ub::T=T(1.0),
                           max_leverage::T=T(1.0)) where T<:Real
    n = length(mu)
    # Kelly objective: max w'μ - 0.5 w'Σw  =>  min 0.5 w'Σw - w'μ
    # With fraction: min 0.5/f w'Σw - w'μ
    H = Sigma ./ fraction
    f_vec = -mu
    A_eq = ones(T, 1, n)
    b_eq = [max_leverage]
    lb_vec = fill(lb, n)
    ub_vec = fill(ub, n)
    w, _, _ = active_set_qp(H, f_vec; A_eq=A_eq, b_eq=b_eq,
                            lb=lb_vec, ub=ub_vec)
    return w
end

"""Kelly with transaction costs."""
function kelly_with_costs(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                          w_current::AbstractVector{T};
                          fraction::T=T(0.5), cost_rate::T=T(0.001)) where T<:Real
    n = length(mu)
    # Adjust expected returns for transaction costs
    mu_adj = copy(mu)
    for i in 1:n
        mu_adj[i] -= cost_rate * abs(one(T)/n - w_current[i])  # rough estimate
    end
    constrained_kelly(mu_adj, Sigma; fraction=fraction)
end

"""Geometric growth rate of a portfolio."""
function geometric_growth_rate(w::AbstractVector{T}, mu::AbstractVector{T},
                               Sigma::AbstractMatrix{T}) where T<:Real
    dot(w, mu) - T(0.5) * dot(w, Sigma * w)
end

# ─────────────────────────────────────────────────────────────────────────────
# §13  Risk Budgeting with Factor Exposure Constraints
# ─────────────────────────────────────────────────────────────────────────────

"""
    risk_budget_portfolio(Sigma, budgets; kwargs...) -> weights

Risk budgeting: each asset contributes a specified fraction of total risk.
"""
function risk_budget_portfolio(Sigma::AbstractMatrix{T},
                               budgets::AbstractVector{T};
                               max_iter::Int=1000, tol::T=T(1e-10)) where T<:Real
    n = size(Sigma, 1)
    @assert length(budgets) == n
    budgets = budgets ./ sum(budgets)
    # Spinu (2013) formulation: min 0.5 x'Σx s.t. sum(b_i ln x_i) >= c
    # Equivalent: x_i = b_i / (Σx)_i, then rescale
    x = copy(budgets)
    for i in 1:n
        x[i] = max(x[i], T(1e-8))
    end
    for iter in 1:max_iter
        Sigma_x = Sigma * x
        x_new = budgets ./ (Sigma_x .+ T(1e-16))
        # Rescale
        sigma_p = sqrt(dot(x_new, Sigma * x_new))
        x_new ./= max(sigma_p, T(1e-16))
        if norm(x_new .- x) / (norm(x) + T(1e-16)) < tol
            x = x_new
            break
        end
        x = x_new
    end
    w = x ./ sum(x)
    return w
end

"""
    factor_risk_budget(Sigma, B, factor_budgets; kwargs...) -> weights

Risk budgeting with factor exposure constraints.
B is n_assets x n_factors factor loading matrix.
"""
function factor_risk_budget(Sigma::AbstractMatrix{T},
                            B::AbstractMatrix{T},
                            factor_budgets::AbstractVector{T};
                            factor_cov::Union{Nothing, AbstractMatrix{T}}=nothing,
                            max_iter::Int=2000,
                            tol::T=T(1e-10)) where T<:Real
    n_assets, n_factors = size(B)
    @assert length(factor_budgets) == n_factors
    factor_budgets = factor_budgets ./ sum(factor_budgets)
    if factor_cov === nothing
        factor_cov = B' * Sigma * B
    end
    # Start with inverse-vol weights
    sigma_assets = sqrt.(diag(Sigma))
    w = (one(T) ./ sigma_assets)
    w ./= sum(w)
    for iter in 1:max_iter
        sigma_p = port_vol(w, Sigma)
        if sigma_p < T(1e-16)
            break
        end
        # Factor risk contributions
        Sigma_w = Sigma * w
        factor_rc = zeros(T, n_factors)
        for f in 1:n_factors
            factor_rc[f] = dot(B[:, f] .* w, Sigma_w) / sigma_p
        end
        factor_rc_frac = factor_rc ./ (sum(abs.(factor_rc)) + T(1e-16))
        # Adjust weights to move factor risk contributions toward budget
        grad = zeros(T, n_assets)
        for f in 1:n_factors
            diff = factor_rc_frac[f] - factor_budgets[f]
            grad .+= diff .* B[:, f]
        end
        w .-= T(0.01) .* grad
        for i in 1:n_assets
            w[i] = max(w[i], T(1e-8))
        end
        normalize_weights!(w)
        if norm(grad) < tol
            break
        end
    end
    return w
end

"""Risk parity across factor groups."""
function factor_group_risk_parity(Sigma::AbstractMatrix{T},
                                   groups::Vector{Vector{Int}};
                                   max_iter::Int=1000) where T<:Real
    n = size(Sigma, 1)
    n_groups = length(groups)
    # Equal risk contribution across groups
    w = fill(one(T) / n, n)
    for iter in 1:max_iter
        sigma_p = port_vol(w, Sigma)
        if sigma_p < T(1e-16)
            break
        end
        Sigma_w = Sigma * w
        group_rc = zeros(T, n_groups)
        for (g, idx) in enumerate(groups)
            for i in idx
                group_rc[g] += w[i] * Sigma_w[i] / sigma_p
            end
        end
        target_rc = sigma_p / n_groups
        for (g, idx) in enumerate(groups)
            scale = target_rc / (group_rc[g] + T(1e-16))
            scale = clamp(scale, T(0.5), T(2.0))
            for i in idx
                w[i] *= scale^T(0.1)
            end
        end
        for i in 1:n
            w[i] = max(w[i], T(1e-8))
        end
        normalize_weights!(w)
    end
    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# §14  Transaction Cost-Aware Optimization
# ─────────────────────────────────────────────────────────────────────────────

"""
    transaction_cost_optimize(mu, Sigma, w_current; kwargs...) -> weights

Mean-variance optimization penalized by transaction costs.
"""
function transaction_cost_optimize(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                                   w_current::AbstractVector{T};
                                   risk_aversion::T=T(1.0),
                                   cost_linear::T=T(0.001),
                                   cost_quadratic::T=T(0.0),
                                   turnover_limit::T=T(Inf),
                                   lb::T=T(0.0), ub::T=T(1.0),
                                   max_iter::Int=5000) where T<:Real
    n = length(mu)
    # Penalize turnover in objective
    # min 0.5γ w'Σw - w'μ + c_lin * |w - w_curr| + c_quad * (w - w_curr)'(w - w_curr)
    # Linearize |w - w_curr| with auxiliary variables
    # Expand: w = w+ - w-  where w+ = max(w - w_curr, 0), w- = max(w_curr - w, 0)
    H = risk_aversion .* Sigma .+ T(2.0) .* cost_quadratic .* I(n)
    f_vec = -mu .+ T(2.0) .* cost_quadratic .* w_current
    # Add linear cost penalty via iterative linearization
    w = copy(w_current)
    for outer in 1:20
        delta = w .- w_current
        subgrad = cost_linear .* sign.(delta)
        f_adj = f_vec .+ subgrad
        A_eq = ones(T, 1, n)
        b_eq = [one(T)]
        lb_vec = fill(lb, n)
        ub_vec = fill(ub, n)
        # Turnover constraint
        A_ineq = zeros(T, 0, n)
        b_ineq = zeros(T, 0)
        if isfinite(turnover_limit)
            # Approximate: upper bound on turnover
            # |w_i - w_curr_i| <= t_i, sum(t_i) <= turnover_limit
            # With linearization: (w_i - w_curr_i) <= turnover_limit/n
            for i in 1:n
                row_pos = zeros(T, 1, n)
                row_pos[1, i] = one(T)
                row_neg = zeros(T, 1, n)
                row_neg[1, i] = -one(T)
                bound = turnover_limit / n + w_current[i]
                bound_neg = -turnover_limit / n + w_current[i]
                A_ineq = vcat(A_ineq, row_pos)
                push!(b_ineq, bound)
                A_ineq = vcat(A_ineq, row_neg)
                push!(b_ineq, -bound_neg)
            end
        end
        w_new, _, _ = active_set_qp(H, f_adj; A_eq=A_eq, b_eq=b_eq,
                                    A_ineq=A_ineq, b_ineq=b_ineq,
                                    lb=lb_vec, ub=ub_vec, max_iter=max_iter)
        if norm(w_new .- w) < T(1e-8)
            w = w_new
            break
        end
        w = w_new
    end
    normalize_weights!(w)
    return w
end

"""
    tc_aware_rebalance(mu, Sigma, w_current, threshold; kwargs...) -> weights, do_rebalance

Rebalance only if benefit exceeds transaction costs.
"""
function tc_aware_rebalance(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                            w_current::AbstractVector{T};
                            risk_aversion::T=T(1.0),
                            cost_rate::T=T(0.001),
                            min_benefit::T=T(0.0001)) where T<:Real
    w_opt = mean_variance_optimize(mu, Sigma; risk_aversion=risk_aversion)
    utility_current = dot(w_current, mu) - T(0.5) * risk_aversion * port_variance(w_current, Sigma)
    utility_opt = dot(w_opt, mu) - T(0.5) * risk_aversion * port_variance(w_opt, Sigma)
    tc = cost_rate * sum(abs.(w_opt .- w_current))
    benefit = utility_opt - utility_current - tc
    if benefit > min_benefit
        w_final = transaction_cost_optimize(mu, Sigma, w_current;
                                           risk_aversion=risk_aversion,
                                           cost_linear=cost_rate)
        return w_final, true
    else
        return w_current, false
    end
end

"""Optimal trade schedule (Almgren-Chriss style)."""
function optimal_trade_schedule(w_target::AbstractVector{T},
                                w_current::AbstractVector{T};
                                n_periods::Int=5,
                                urgency::T=T(1.0),
                                impact_coeff::T=T(0.1)) where T<:Real
    n = length(w_target)
    delta = w_target .- w_current
    # Exponential decay schedule
    schedule = Matrix{T}(undef, n, n_periods)
    remaining = copy(delta)
    for t in 1:n_periods
        frac = one(T) - exp(-urgency * t / n_periods)
        frac_prev = t > 1 ? one(T) - exp(-urgency * (t-1) / n_periods) : zero(T)
        trade = delta .* (frac - frac_prev)
        schedule[:, t] = trade
        remaining .-= trade
    end
    # Adjust last period for rounding
    schedule[:, n_periods] .+= remaining
    return schedule
end

# ─────────────────────────────────────────────────────────────────────────────
# §15  Tax-Lot Aware Rebalancing
# ─────────────────────────────────────────────────────────────────────────────

"""Tax lot representation."""
struct TaxLot{T<:Real}
    asset_id::Int
    shares::T
    cost_basis::T
    purchase_date::Int  # days since epoch
    is_short_term::Bool
end

"""
    tax_lot_rebalance(lots, prices, w_target; kwargs...) -> trades

Tax-aware rebalancing: minimize taxes while approaching target.
"""
function tax_lot_rebalance(lots::Vector{TaxLot{T}}, prices::AbstractVector{T},
                           w_target::AbstractVector{T};
                           short_term_rate::T=T(0.37),
                           long_term_rate::T=T(0.20),
                           current_day::Int=0,
                           lt_threshold::Int=365) where T<:Real
    n_assets = length(prices)
    n_lots = length(lots)
    # Compute current positions
    positions = zeros(T, n_assets)
    for lot in lots
        positions[lot.asset_id] += lot.shares * prices[lot.asset_id]
    end
    total_value = sum(positions)
    w_current = positions ./ max(total_value, T(1e-16))
    trades_needed = (w_target .- w_current) .* total_value ./ prices
    # For sells, prioritize lots with losses or long-term gains
    sell_schedule = Vector{Tuple{Int, T, T}}()  # (lot_index, shares_to_sell, tax)
    for asset_id in 1:n_assets
        if trades_needed[asset_id] >= 0
            continue  # buying, no tax lot selection needed
        end
        shares_to_sell = -trades_needed[asset_id]
        # Find lots for this asset, sorted by tax efficiency
        asset_lots = [(i, lot) for (i, lot) in enumerate(lots) if lot.asset_id == asset_id]
        # Sort: losses first, then long-term gains, then short-term gains
        lot_scores = map(asset_lots) do (i, lot)
            gain_per_share = prices[asset_id] - lot.cost_basis
            is_lt = !lot.is_short_term && (current_day - lot.purchase_date >= lt_threshold)
            tax_rate = is_lt ? long_term_rate : short_term_rate
            tax_cost = max(gain_per_share * tax_rate, zero(T))
            (i, lot, tax_cost, gain_per_share)
        end
        sort!(lot_scores; by=x -> x[3])  # lowest tax first
        remaining = shares_to_sell
        for (lot_idx, lot, tax_cost, gain) in lot_scores
            if remaining <= 0
                break
            end
            sell_qty = min(remaining, lot.shares)
            tax = sell_qty * max(gain * (gain > 0 ? (lot.is_short_term ? short_term_rate : long_term_rate) : short_term_rate), zero(T))
            push!(sell_schedule, (lot_idx, sell_qty, tax))
            remaining -= sell_qty
        end
    end
    return sell_schedule, trades_needed
end

"""
    tax_lot_harvest(lots, prices; threshold=0.03) -> harvest_candidates

Identify tax-loss harvesting opportunities.
"""
function tax_lot_harvest(lots::Vector{TaxLot{T}}, prices::AbstractVector{T};
                         threshold::T=T(0.03),
                         wash_sale_days::Int=30,
                         current_day::Int=0) where T<:Real
    candidates = Vector{Tuple{Int, T, T}}()  # (lot_idx, shares, loss)
    for (i, lot) in enumerate(lots)
        current_value = lot.shares * prices[lot.asset_id]
        basis_value = lot.shares * lot.cost_basis
        loss = basis_value - current_value
        loss_pct = loss / max(basis_value, T(1e-16))
        if loss_pct > threshold
            push!(candidates, (i, lot.shares, loss))
        end
    end
    sort!(candidates; by=x -> -x[3])  # largest losses first
    return candidates
end

"""Compute after-tax return of a portfolio."""
function after_tax_return(w::AbstractVector{T}, mu::AbstractVector{T},
                          div_yields::AbstractVector{T};
                          income_rate::T=T(0.37),
                          ltcg_rate::T=T(0.20),
                          turnover::T=T(0.5),
                          pct_ltcg::T=T(0.7)) where T<:Real
    # Total return = price appreciation + dividends
    # Tax: dividends at income_rate, realized gains at blended rate
    cap_gains = mu .- div_yields
    blended_cg_rate = pct_ltcg * ltcg_rate + (one(T) - pct_ltcg) * income_rate
    after_tax_cg = cap_gains .* (one(T) .- turnover .* blended_cg_rate)
    after_tax_div = div_yields .* (one(T) .- income_rate)
    after_tax_mu = after_tax_cg .+ after_tax_div
    return dot(w, after_tax_mu)
end

# ─────────────────────────────────────────────────────────────────────────────
# §16  Multi-Period Mean-Variance (Dynamic Programming)
# ─────────────────────────────────────────────────────────────────────────────

"""
    multi_period_mv(mu_t, Sigma_t; kwargs...) -> weights_per_period

Multi-period mean-variance via backward induction.
mu_t: vector of T expected return vectors
Sigma_t: vector of T covariance matrices
"""
function multi_period_mv(mu_t::Vector{Vector{T}}, Sigma_t::Vector{Matrix{T}};
                         risk_aversion::T=T(1.0),
                         discount::T=T(0.99),
                         tc_rate::T=T(0.001)) where T<:Real
    n_periods = length(mu_t)
    n_assets = length(mu_t[1])
    weights = Vector{Vector{T}}(undef, n_periods)
    # Terminal period
    weights[n_periods] = mean_variance_optimize(mu_t[n_periods], Sigma_t[n_periods];
                                                 risk_aversion=risk_aversion)
    # Backward induction
    for t in (n_periods-1):-1:1
        w_next = weights[t + 1]
        # Adjust expected return for transaction costs to reach next period target
        mu_adj = mu_t[t] .- tc_rate .* abs.(mu_t[t] .- dot(w_next, mu_t[t]) .* ones(T, n_assets))
        # Value function: current utility + discounted future
        gamma_eff = risk_aversion / (one(T) + discount)
        weights[t] = mean_variance_optimize(mu_adj, Sigma_t[t]; risk_aversion=gamma_eff)
    end
    return weights
end

"""
    dp_multi_period(mu_scenarios, Sigma, n_periods; kwargs...) -> policy

Dynamic programming multi-period optimization with scenario tree.
"""
function dp_multi_period(mu_scenarios::Matrix{T},
                         Sigma::AbstractMatrix{T},
                         n_periods::Int;
                         risk_aversion::T=T(1.0),
                         n_grid::Int=20) where T<:Real
    n_scenarios, n_assets = size(mu_scenarios)
    # Discretize wealth levels
    wealth_grid = range(T(0.5), T(2.0), length=n_grid)
    # Value function table
    V = zeros(T, n_grid, n_periods + 1)
    policy = Vector{Matrix{T}}(undef, n_periods)  # n_grid x n_assets
    # Terminal value function
    for iw in 1:n_grid
        V[iw, n_periods + 1] = wealth_grid[iw]
    end
    # Backward induction
    for t in n_periods:-1:1
        policy[t] = zeros(T, n_grid, n_assets)
        for iw in 1:n_grid
            W = wealth_grid[iw]
            best_val = T(-Inf)
            best_w = fill(one(T) / n_assets, n_assets)
            # Try several candidate portfolios
            for trial in 1:50
                if trial == 1
                    w_try = fill(one(T) / n_assets, n_assets)
                else
                    w_try = rand(T, n_assets)
                    w_try ./= sum(w_try)
                end
                # Expected value across scenarios
                exp_val = zero(T)
                for s in 1:n_scenarios
                    r = dot(w_try, mu_scenarios[s, :])
                    vol = port_vol(w_try, Sigma)
                    W_next = W * (one(T) + r)
                    utility = W_next - T(0.5) * risk_aversion * W * vol^2
                    # Interpolate future value
                    idx_f = (W_next - wealth_grid[1]) / (wealth_grid[end] - wealth_grid[1]) * (n_grid - 1) + 1
                    idx_f = clamp(idx_f, one(T), T(n_grid))
                    idx_lo = max(1, floor(Int, idx_f))
                    idx_hi = min(n_grid, idx_lo + 1)
                    frac = idx_f - idx_lo
                    future_val = (one(T) - frac) * V[idx_lo, t+1] + frac * V[idx_hi, t+1]
                    exp_val += (utility + T(0.99) * future_val) / n_scenarios
                end
                if exp_val > best_val
                    best_val = exp_val
                    best_w = w_try
                end
            end
            V[iw, t] = best_val
            policy[t][iw, :] = best_w
        end
    end
    return policy, V[:, 1:n_periods], wealth_grid
end

# ─────────────────────────────────────────────────────────────────────────────
# §17  Additional Portfolio Metrics
# ─────────────────────────────────────────────────────────────────────────────

"""Herfindahl-Hirschman Index of portfolio concentration."""
hhi(w::AbstractVector) = sum(w .^ 2)

"""Effective number of bets."""
effective_n(w::AbstractVector) = one(eltype(w)) / hhi(w)

"""Portfolio entropy."""
function portfolio_entropy(w::AbstractVector{T}) where T<:Real
    s = zero(T)
    for wi in w
        if wi > T(1e-16)
            s -= wi * log(wi)
        end
    end
    s
end

"""Marginal risk contribution."""
function marginal_risk_contribution(w::AbstractVector{T}, Sigma::AbstractMatrix{T}) where T<:Real
    vol = port_vol(w, Sigma)
    Sigma * w ./ max(vol, T(1e-16))
end

"""Component VaR at given confidence."""
function component_var(w::AbstractVector{T}, Sigma::AbstractMatrix{T};
                       alpha::T=T(0.05), z::T=T(-1.6449)) where T<:Real
    vol = port_vol(w, Sigma)
    total_var = -z * vol
    mrc = marginal_risk_contribution(w, Sigma)
    cvar = w .* mrc .* (-z)
    return cvar, total_var
end

"""Tracking error relative to benchmark."""
function tracking_error(w::AbstractVector{T}, w_bench::AbstractVector{T},
                        Sigma::AbstractMatrix{T}) where T<:Real
    delta = w .- w_bench
    sqrt(max(dot(delta, Sigma * delta), zero(T)))
end

"""Information ratio."""
function information_ratio(w::AbstractVector{T}, w_bench::AbstractVector{T},
                           mu::AbstractVector{T}, Sigma::AbstractMatrix{T}) where T<:Real
    excess_ret = dot(w .- w_bench, mu)
    te = tracking_error(w, w_bench, Sigma)
    excess_ret / max(te, T(1e-16))
end

"""Maximum drawdown from a return series."""
function max_drawdown(returns::AbstractVector{T}) where T<:Real
    cum = cumprod(one(T) .+ returns)
    peak = accumulate(max, cum)
    dd = (peak .- cum) ./ peak
    maximum(dd)
end

"""Omega ratio at threshold."""
function omega_ratio(returns::AbstractVector{T}; threshold::T=T(0.0)) where T<:Real
    gains = sum(max.(returns .- threshold, zero(T)))
    losses = sum(max.(threshold .- returns, zero(T)))
    gains / max(losses, T(1e-16))
end

"""Sortino ratio."""
function sortino_ratio(returns::AbstractVector{T}; rf::T=T(0.0)) where T<:Real
    excess = returns .- rf
    downside = sqrt(mean(max.(-excess, zero(T)) .^ 2))
    mean(excess) / max(downside, T(1e-16))
end

"""Calmar ratio."""
function calmar_ratio(returns::AbstractVector{T}) where T<:Real
    ann_ret = mean(returns) * 252
    mdd = max_drawdown(returns)
    ann_ret / max(mdd, T(1e-16))
end

# ─────────────────────────────────────────────────────────────────────────────
# §18  Inverse Optimization (Implied Views)
# ─────────────────────────────────────────────────────────────────────────────

"""Reverse-engineer implied returns from portfolio weights."""
function implied_returns(w::AbstractVector{T}, Sigma::AbstractMatrix{T};
                         risk_aversion::T=T(2.5)) where T<:Real
    risk_aversion .* Sigma * w
end

"""Implied risk aversion from market portfolio and expected return."""
function implied_risk_aversion(w_mkt::AbstractVector{T}, Sigma::AbstractMatrix{T},
                                mu_mkt::T) where T<:Real
    mu_mkt / dot(w_mkt, Sigma * w_mkt)
end

"""Implied views: what P, Q would produce given weights from prior?"""
function implied_views(w_target::AbstractVector{T}, w_prior::AbstractVector{T},
                       Sigma::AbstractMatrix{T};
                       tau::T=T(0.05), risk_aversion::T=T(2.5)) where T<:Real
    n = length(w_target)
    pi_eq = risk_aversion .* Sigma * w_prior
    mu_implied = risk_aversion .* Sigma * w_target
    diff = mu_implied .- pi_eq
    # Infer views: use identity P (asset-level views)
    P = Matrix{T}(I, n, n)
    Q = pi_eq .+ diff ./ tau
    return P, Q
end

# ─────────────────────────────────────────────────────────────────────────────
# §19  Scenario Analysis & Stress Testing
# ─────────────────────────────────────────────────────────────────────────────

"""Apply scenario shocks and compute portfolio impact."""
function scenario_analysis(w::AbstractVector{T},
                           scenarios::Matrix{T}) where T<:Real
    # scenarios: n_scenarios x n_assets matrix of returns
    port_rets = scenarios * w
    return port_rets
end

"""Stress test: worst-case portfolio loss under parameter uncertainty."""
function stress_test(w::AbstractVector{T}, Sigma::AbstractMatrix{T},
                     mu::AbstractVector{T};
                     mu_uncertainty::T=T(0.02),
                     cov_uncertainty::T=T(0.2),
                     n_scenarios::Int=1000,
                     rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n = length(w)
    worst_loss = zero(T)
    worst_vol = zero(T)
    for _ in 1:n_scenarios
        mu_shock = mu .+ mu_uncertainty .* randn(rng, T, n)
        # Random perturbation of covariance
        Z = randn(rng, T, n, n) .* cov_uncertainty
        Sigma_shock = Sigma .+ (Z .+ Z') ./ 2
        # Ensure PSD
        F = eigen(Symmetric(Sigma_shock))
        vals = max.(F.values, T(1e-8))
        Sigma_shock = F.vectors * Diagonal(vals) * F.vectors'
        ret = dot(w, mu_shock)
        vol = port_vol(w, Sigma_shock)
        loss = -ret + T(2.33) * vol  # ~99% VaR
        if loss > worst_loss
            worst_loss = loss
            worst_vol = vol
        end
    end
    return worst_loss, worst_vol
end

"""Marginal contribution to scenario loss."""
function marginal_scenario_impact(w::AbstractVector{T},
                                  scenario::AbstractVector{T}) where T<:Real
    # scenario: n_assets vector of asset returns in stress scenario
    port_ret = dot(w, scenario)
    contributions = w .* scenario
    return contributions, port_ret
end

# ─────────────────────────────────────────────────────────────────────────────
# §20  Constraint Handling Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Build sector exposure constraints."""
function sector_constraints(n_assets::Int, sector_map::Dict{Int, Vector{Int}},
                            sector_bounds::Dict{Int, Tuple{T, T}}) where T<:Real
    n_constraints = 2 * length(sector_bounds)
    A = zeros(T, n_constraints, n_assets)
    b = zeros(T, n_constraints)
    row = 1
    for (sector_id, (lb, ub)) in sector_bounds
        assets = sector_map[sector_id]
        for a in assets
            A[row, a] = one(T)      # sum <= ub
            A[row+1, a] = -one(T)   # -sum <= -lb
        end
        b[row] = ub
        b[row+1] = -lb
        row += 2
    end
    return A[1:row-1, :], b[1:row-1]
end

"""Build turnover constraint matrix."""
function turnover_constraint(n_assets::Int, w_current::AbstractVector{T},
                             max_turnover::T) where T<:Real
    # |w - w_curr|_1 <= max_turnover
    # Linearize: introduce slack variables u_i >= |w_i - w_curr_i|
    # w_i - w_curr_i <= u_i  => w_i - u_i <= w_curr_i
    # w_curr_i - w_i <= u_i  => -w_i - u_i <= -w_curr_i
    # sum u_i <= max_turnover
    A = zeros(T, 2 * n_assets + 1, 2 * n_assets)
    b = zeros(T, 2 * n_assets + 1)
    for i in 1:n_assets
        A[2i-1, i] = one(T)
        A[2i-1, n_assets+i] = -one(T)
        b[2i-1] = w_current[i]
        A[2i, i] = -one(T)
        A[2i, n_assets+i] = -one(T)
        b[2i] = -w_current[i]
    end
    for i in 1:n_assets
        A[2*n_assets+1, n_assets+i] = one(T)
    end
    b[2*n_assets+1] = max_turnover
    return A, b
end

"""Cardinality constraint via iterative thresholding."""
function cardinality_constrained_mv(mu::AbstractVector{T}, Sigma::AbstractMatrix{T},
                                     max_assets::Int;
                                     risk_aversion::T=T(1.0),
                                     n_rounds::Int=10) where T<:Real
    n = length(mu)
    mask = trues(n)
    w = mean_variance_optimize(mu, Sigma; risk_aversion=risk_aversion)
    for round in 1:n_rounds
        active_count = count(mask)
        if active_count <= max_assets
            break
        end
        # Zero out smallest weight
        min_idx = 0
        min_val = T(Inf)
        for i in 1:n
            if mask[i] && abs(w[i]) < min_val
                min_val = abs(w[i])
                min_idx = i
            end
        end
        if min_idx > 0
            mask[min_idx] = false
        end
        # Re-optimize with remaining assets
        active = findall(mask)
        if isempty(active)
            break
        end
        mu_sub = mu[active]
        Sigma_sub = Sigma[active, active]
        w_sub = mean_variance_optimize(mu_sub, Sigma_sub; risk_aversion=risk_aversion)
        w = zeros(T, n)
        for (k, idx) in enumerate(active)
            w[idx] = w_sub[k]
        end
    end
    normalize_weights!(w)
    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# §21  Covariance Estimation Methods
# ─────────────────────────────────────────────────────────────────────────────

"""Constant correlation model."""
function constant_correlation_cov(returns::AbstractMatrix{T}) where T<:Real
    n, p = size(returns)
    S = cov(returns)
    sigma = sqrt.(diag(S))
    corr = S ./ (sigma * sigma')
    avg_corr = zero(T)
    count = 0
    for j in 1:p, i in j+1:p
        avg_corr += corr[i,j]
        count += 1
    end
    avg_corr /= count
    Sigma_cc = sigma * sigma' .* avg_corr
    for i in 1:p
        Sigma_cc[i,i] = S[i,i]
    end
    Symmetric(Sigma_cc)
end

"""Single-factor covariance model."""
function single_factor_cov(returns::AbstractMatrix{T},
                           factor_returns::AbstractVector{T}) where T<:Real
    n, p = size(returns)
    betas = zeros(T, p)
    alphas = zeros(T, p)
    resid_var = zeros(T, p)
    factor_var = var(factor_returns)
    factor_mean = mean(factor_returns)
    for i in 1:p
        beta = cov(returns[:, i], factor_returns) / max(factor_var, T(1e-16))
        alpha = mean(returns[:, i]) - beta * factor_mean
        resids = returns[:, i] .- alpha .- beta .* factor_returns
        betas[i] = beta
        alphas[i] = alpha
        resid_var[i] = var(resids)
    end
    Sigma = betas * betas' .* factor_var .+ Diagonal(resid_var)
    Symmetric(Sigma)
end

"""Multi-factor covariance model."""
function multi_factor_cov(returns::AbstractMatrix{T},
                          factors::AbstractMatrix{T}) where T<:Real
    n, p = size(returns)
    n_f, k = size(factors)
    @assert n == n_f
    # OLS: returns = factors * B' + epsilon
    B = (factors' * factors) \ (factors' * returns)  # k x p
    residuals = returns .- factors * B
    D = Diagonal(vec(var(residuals; dims=1)))
    F = cov(factors)
    Sigma = B' * F * B .+ D
    Symmetric(Sigma)
end

"""Gerber statistic-based covariance."""
function gerber_cov(returns::AbstractMatrix{T}; threshold::T=T(0.5)) where T<:Real
    n, p = size(returns)
    sigma = vec(std(returns; dims=1))
    H = zeros(T, p, p)
    for j in 1:p, i in j:p
        n_concordant = 0
        n_discordant = 0
        for t in 1:n
            ri = returns[t, i] / max(sigma[i], T(1e-16))
            rj = returns[t, j] / max(sigma[j], T(1e-16))
            if abs(ri) > threshold && abs(rj) > threshold
                if ri * rj > 0
                    n_concordant += 1
                else
                    n_discordant += 1
                end
            end
        end
        total = n_concordant + n_discordant
        if total > 0
            H[i, j] = (n_concordant - n_discordant) / total
        end
        H[j, i] = H[i, j]
    end
    Sigma = Diagonal(sigma) * H * Diagonal(sigma)
    Symmetric((Sigma .+ Sigma') ./ 2)
end

# ─────────────────────────────────────────────────────────────────────────────
# §22  Regime-Aware Portfolio Optimization
# ─────────────────────────────────────────────────────────────────────────────

"""Regime-switching portfolio: blend portfolios based on regime probabilities."""
function regime_portfolio(mu_regimes::Vector{Vector{T}},
                          Sigma_regimes::Vector{Matrix{T}},
                          regime_probs::AbstractVector{T};
                          risk_aversion::T=T(1.0),
                          lb::T=T(0.0), ub::T=T(1.0)) where T<:Real
    n_regimes = length(mu_regimes)
    n_assets = length(mu_regimes[1])
    # Blended moments
    mu_blend = zeros(T, n_assets)
    Sigma_blend = zeros(T, n_assets, n_assets)
    for r in 1:n_regimes
        mu_blend .+= regime_probs[r] .* mu_regimes[r]
        Sigma_blend .+= regime_probs[r] .* Sigma_regimes[r]
    end
    # Add cross-regime variance
    for r in 1:n_regimes
        diff = mu_regimes[r] .- mu_blend
        Sigma_blend .+= regime_probs[r] .* (diff * diff')
    end
    mean_variance_optimize(mu_blend, Sigma_blend; risk_aversion=risk_aversion,
                          lb=lb, ub=ub)
end

"""Worst-case regime portfolio (minimax)."""
function worst_case_portfolio(mu_regimes::Vector{Vector{T}},
                              Sigma_regimes::Vector{Matrix{T}};
                              risk_aversion::T=T(1.0),
                              lb::T=T(0.0), ub::T=T(1.0)) where T<:Real
    n_regimes = length(mu_regimes)
    n_assets = length(mu_regimes[1])
    # Find portfolio that maximizes worst-case utility
    best_w = fill(one(T) / n_assets, n_assets)
    best_worst_util = T(-Inf)
    for trial in 1:200
        gamma = T(10.0^(rand() * 4 - 2))
        for r in 1:n_regimes
            w = mean_variance_optimize(mu_regimes[r], Sigma_regimes[r];
                                      risk_aversion=gamma, lb=lb, ub=ub)
            worst_util = T(Inf)
            for r2 in 1:n_regimes
                util = dot(w, mu_regimes[r2]) - T(0.5) * risk_aversion * port_variance(w, Sigma_regimes[r2])
                worst_util = min(worst_util, util)
            end
            if worst_util > best_worst_util
                best_worst_util = worst_util
                best_w = copy(w)
            end
        end
    end
    return best_w
end

# ─────────────────────────────────────────────────────────────────────────────
# §23  Bayesian Portfolio Optimization
# ─────────────────────────────────────────────────────────────────────────────

"""Bayesian shrinkage of expected returns toward grand mean."""
function bayesian_shrinkage_returns(returns::AbstractMatrix{T};
                                     prior_strength::T=T(1.0)) where T<:Real
    n, p = size(returns)
    mu = vec(mean(returns; dims=1))
    grand_mean = mean(mu)
    sigma2 = vec(var(returns; dims=1))
    # Shrinkage toward grand mean
    tau = var(mu)
    shrinkage = sigma2 ./ (sigma2 .+ n .* tau .* prior_strength)
    mu_shrunk = (one(T) .- shrinkage) .* mu .+ shrinkage .* grand_mean
    return mu_shrunk
end

"""Bayesian mean-variance with parameter uncertainty."""
function bayesian_mv(returns::AbstractMatrix{T};
                     risk_aversion::T=T(1.0),
                     prior_strength::T=T(1.0),
                     lb::T=T(0.0), ub::T=T(1.0)) where T<:Real
    n, p = size(returns)
    mu = bayesian_shrinkage_returns(returns; prior_strength=prior_strength)
    Sigma, _ = ledoit_wolf_shrinkage(returns)
    # Inflate covariance for parameter uncertainty
    Sigma_adj = Sigma .* (one(T) + one(T) / n)
    mean_variance_optimize(mu, Sigma_adj; risk_aversion=risk_aversion, lb=lb, ub=ub)
end

# ─────────────────────────────────────────────────────────────────────────────
# §24  Mean-CVaR Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

"""Compute mean-CVaR efficient frontier."""
function mean_cvar_frontier(mu::AbstractVector{T}, returns::AbstractMatrix{T};
                            n_points::Int=30, alpha::T=T(0.05)) where T<:Real
    n_assets = length(mu)
    ret_min = minimum(mu)
    ret_max = maximum(mu)
    targets = range(ret_min * T(0.5), ret_max, length=n_points)
    rets = Vector{T}(undef, n_points)
    cvars = Vector{T}(undef, n_points)
    W = Matrix{T}(undef, n_assets, n_points)
    for (k, tgt) in enumerate(targets)
        w = cvar_optimize(mu, returns; alpha=alpha, target_return=tgt)
        rets[k] = dot(w, mu)
        cvars[k] = portfolio_cvar(w, returns; alpha=alpha)
        W[:, k] = w
    end
    return rets, cvars, W
end

# ─────────────────────────────────────────────────────────────────────────────
# §25  Portfolio Construction Pipelines
# ─────────────────────────────────────────────────────────────────────────────

"""Full pipeline: estimate moments -> optimize -> post-process."""
function portfolio_pipeline(returns::AbstractMatrix{T};
                            method::Symbol=:mean_variance,
                            risk_aversion::T=T(1.0),
                            shrinkage::Symbol=:ledoit_wolf,
                            lb::T=T(0.0), ub::T=T(1.0),
                            max_assets::Int=0) where T<:Real
    mu = vec(mean(returns; dims=1))
    Sigma = if shrinkage == :ledoit_wolf
        first(ledoit_wolf_shrinkage(returns))
    elseif shrinkage == :oas
        first(oas_shrinkage(returns))
    elseif shrinkage == :constant_corr
        constant_correlation_cov(returns)
    else
        Symmetric(cov(returns))
    end
    w = if method == :mean_variance
        mean_variance_optimize(mu, Sigma; risk_aversion=risk_aversion, lb=lb, ub=ub)
    elseif method == :min_variance
        min_variance_portfolio(Sigma; lb=lb, ub=ub)
    elseif method == :hrp
        hrp_portfolio(returns)
    elseif method == :erc
        erc_portfolio(Sigma)
    elseif method == :max_div
        max_diversification_portfolio(Sigma; lb=lb, ub=ub)
    elseif method == :min_corr
        min_correlation_portfolio(Sigma)
    elseif method == :kelly
        constrained_kelly(mu, Sigma; lb=lb, ub=ub)
    elseif method == :cvar
        cvar_optimize(mu, returns)
    else
        fill(one(T) / size(returns, 2), size(returns, 2))
    end
    if max_assets > 0 && count(x -> x > T(1e-6), w) > max_assets
        w = cardinality_constrained_mv(mu, Sigma, max_assets;
                                       risk_aversion=risk_aversion)
    end
    return w, mu, Sigma
end

"""Ensemble portfolio: blend multiple optimization methods."""
function ensemble_portfolio(returns::AbstractMatrix{T};
                            methods::Vector{Symbol}=[:mean_variance, :hrp, :erc, :min_variance],
                            blend_weights::Union{Nothing, AbstractVector{T}}=nothing) where T<:Real
    n_methods = length(methods)
    n_assets = size(returns, 2)
    if blend_weights === nothing
        blend_weights = fill(one(T) / n_methods, n_methods)
    end
    w_ensemble = zeros(T, n_assets)
    for (i, m) in enumerate(methods)
        w, _, _ = portfolio_pipeline(returns; method=m)
        w_ensemble .+= blend_weights[i] .* w
    end
    normalize_weights!(w_ensemble)
    return w_ensemble
end

"""Rolling portfolio optimization."""
function rolling_optimize(returns::AbstractMatrix{T};
                          window::Int=252, step::Int=21,
                          method::Symbol=:mean_variance,
                          risk_aversion::T=T(1.0)) where T<:Real
    n_obs, n_assets = size(returns)
    n_rebalances = div(n_obs - window, step) + 1
    W = Matrix{T}(undef, n_assets, n_rebalances)
    dates = Vector{Int}(undef, n_rebalances)
    for (k, t) in enumerate(window:step:n_obs)
        r = returns[t-window+1:t, :]
        w, _, _ = portfolio_pipeline(r; method=method, risk_aversion=risk_aversion)
        W[:, k] = w
        dates[k] = t
    end
    return W, dates
end

# ─────────────────────────────────────────────────────────────────────────────
# §26  Optimization Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

"""Check if portfolio satisfies KKT conditions for mean-variance."""
function check_kkt(w::AbstractVector{T}, mu::AbstractVector{T},
                   Sigma::AbstractMatrix{T};
                   risk_aversion::T=T(1.0), tol::T=T(1e-6)) where T<:Real
    n = length(w)
    grad = risk_aversion .* Sigma * w .- mu
    # For active constraints (w_i = 0 or w_i = 1), gradient can be positive
    violations = 0
    for i in 1:n
        if w[i] > tol && w[i] < one(T) - tol
            # Interior point: gradient should be equal (up to Lagrange multiplier)
            if abs(grad[i] - mean(grad[findall(x -> x > tol && x < one(T) - tol, w)])) > tol
                violations += 1
            end
        end
    end
    sum_violation = abs(sum(w) - one(T))
    return violations, sum_violation
end

"""Portfolio turnover between two weight vectors."""
turnover(w_old::AbstractVector, w_new::AbstractVector) = sum(abs.(w_new .- w_old)) / 2

"""Active share relative to benchmark."""
active_share(w::AbstractVector, w_bench::AbstractVector) = sum(abs.(w .- w_bench)) / 2

end # module PortfolioOptimization
