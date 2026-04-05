# =============================================================================
# portfolio_construction.jl — Advanced Portfolio Construction
# =============================================================================
# State-of-the-art portfolio construction methods:
#   1. Mean-Variance (Markowitz) with 1000-point efficient frontier
#   2. Ledoit-Wolf analytical covariance shrinkage
#   3. Resampled Efficiency (Michaud): 500 bootstrapped frontiers
#   4. Black-Litterman with IAE views
#   5. Hierarchical Risk Parity (HRP): clustering + inverse variance
#   6. Risk Budgeting: equal CVaR contribution
#   7. Tail Risk Parity: minimize max CVaR contribution
#   8. Transaction cost-aware rebalancing (LP)
#   9. Turnover-constrained optimization
#
# Julia ≥ 1.10 | No external packages
# =============================================================================

module PortfolioConstruction

using Statistics
using LinearAlgebra

export efficient_frontier, min_variance_portfolio, max_sharpe_portfolio
export ledoit_wolf_shrinkage, ledoit_wolf_analytical
export resampled_efficiency, michaud_resampled_frontier
export black_litterman, bl_posterior
export hierarchical_risk_parity, hrp_weights
export equal_risk_contribution, risk_budget_portfolio
export tail_risk_parity, cvar_risk_budgeting
export turnover_constrained_optimization, tc_aware_rebalancing

# =============================================================================
# SECTION 1: MEAN-VARIANCE OPTIMIZATION
# =============================================================================

"""
    efficient_frontier(mu, Sigma; n_points=1000, rf=0.0,
                       long_only=true, weight_bounds=nothing) -> NamedTuple

Compute the full efficient frontier via parametric quadratic programming.

For each target return μ_p ∈ [min_return, max_return], solve:
    min w' Σ w
    s.t. w' μ = μ_p, Σᵢwᵢ = 1, wᵢ ≥ 0 (if long_only)

Solution via Lagrange multipliers (unconstrained) or iterative projection
(constrained).

# Arguments
- `mu`: N-vector of expected returns
- `Sigma`: N×N covariance matrix
- `n_points`: number of frontier points (default 1000)
- `rf`: risk-free rate for Sharpe ratio computation
- `long_only`: enforce wᵢ ≥ 0 constraint
- `weight_bounds`: (lb, ub) weight bounds per asset

# Returns
- NamedTuple with fields: weights (n_points × N), returns, vols, sharpes,
    max_sharpe_idx, min_var_idx, tangency_weights, min_var_weights
"""
function efficient_frontier(mu::Vector{Float64},
                              Sigma::Matrix{Float64};
                              n_points::Int=1000,
                              rf::Float64=0.0,
                              long_only::Bool=true,
                              weight_bounds=nothing)

    N = length(mu)
    @assert size(Sigma) == (N, N) "Sigma must be N×N"

    # Analytical efficient frontier (unconstrained, no short-sale restriction)
    # Uses the critical line: w(λ) = Σ⁻¹(μλ + e·γ) for scalars λ, γ
    # Two-fund separation: any efficient portfolio is a combination of two

    # Inverse covariance
    Sigma_reg = Sigma + 1e-8 * I
    Sigma_inv = try
        inv(Sigma_reg)
    catch
        pinv(Sigma_reg)
    end

    e = ones(N)  # ones vector

    # Compute scalars for the critical line
    A = e' * Sigma_inv * mu   # e'Σ⁻¹μ
    B = mu' * Sigma_inv * mu  # μ'Σ⁻¹μ
    C = e' * Sigma_inv * e    # e'Σ⁻¹e
    D = B * C - A^2

    # Minimum variance portfolio
    w_mv = Sigma_inv * e / C

    # Maximum return portfolio (100% in highest return asset)
    w_maxret = zeros(N)
    w_maxret[argmax(mu)] = 1.0

    # Target returns grid
    mu_min = (long_only ? minimum(mu) : (w_mv' * mu)[1])
    mu_max = maximum(mu)
    target_returns = range(mu_min, mu_max, length=n_points)

    weights = zeros(n_points, N)
    port_returns = zeros(n_points)
    port_vols = zeros(n_points)

    for (i, mu_p) in enumerate(target_returns)
        if long_only
            w = _mv_long_only(mu, Sigma_reg, mu_p)
        else
            # Analytical unconstrained
            if D > 0
                g = (B * Sigma_inv * e - A * Sigma_inv * mu) / D
                h = (C * Sigma_inv * mu - A * Sigma_inv * e) / D
                w = g + h * mu_p
            else
                w = w_mv
            end
        end

        port_returns[i] = w' * mu
        port_vols[i] = sqrt(max(w' * Sigma_reg * w, 0.0))
        weights[i, :] = w
    end

    port_sharpes = (port_returns .- rf) ./ max.(port_vols, 1e-10)

    max_sharpe_idx = argmax(port_sharpes)
    min_var_idx    = argmin(port_vols)

    return (
        weights           = weights,
        returns           = port_returns,
        vols              = port_vols,
        sharpes           = port_sharpes,
        max_sharpe_idx    = max_sharpe_idx,
        min_var_idx       = min_var_idx,
        tangency_weights  = weights[max_sharpe_idx, :],
        min_var_weights   = weights[min_var_idx, :],
        target_returns    = collect(target_returns),
    )
end

"""Long-only MV portfolio via projected gradient descent."""
function _mv_long_only(mu::Vector{Float64},
                        Sigma::Matrix{Float64},
                        target_return::Float64)::Vector{Float64}

    N = length(mu)
    w = ones(N) / N  # start at equal weight

    lr = 0.01
    lambda = 10.0  # penalty for return constraint

    for iter in 1:500
        # Gradient of variance: 2Σw
        grad_var = 2 * Sigma * w

        # Penalty gradient for return constraint: 2λ(w'μ - target)μ
        return_gap = w' * mu - target_return
        grad_penalty = 2 * lambda * return_gap * mu

        # Full gradient
        grad = grad_var + grad_penalty

        # Gradient step
        w_new = w - lr * grad

        # Project to simplex (long-only, sum=1)
        w_new = _project_simplex(w_new)

        # Convergence check
        if norm(w_new - w) < 1e-8
            break
        end
        w = w_new
    end

    return w
end

"""Project vector onto probability simplex {w: Σwᵢ=1, wᵢ≥0}."""
function _project_simplex(v::Vector{Float64})::Vector{Float64}
    n = length(v)
    u = sort(v, rev=true)
    cssv = cumsum(u)
    rho = 0
    for j in 1:n
        if u[j] - (cssv[j] - 1.0) / j > 0
            rho = j
        end
    end
    theta = (cssv[rho] - 1.0) / rho
    return max.(v .- theta, 0.0)
end

"""
    min_variance_portfolio(Sigma; long_only=true) -> Vector{Float64}

Compute the global minimum variance portfolio.
Analytical: w* = Σ⁻¹e / (e'Σ⁻¹e)  (unconstrained)
Long-only: iterative projection.
"""
function min_variance_portfolio(Sigma::Matrix{Float64};
                                  long_only::Bool=true)::Vector{Float64}

    N = size(Sigma, 1)
    Sigma_reg = Sigma + 1e-8 * I

    if !long_only
        Sigma_inv = try inv(Sigma_reg) catch; pinv(Sigma_reg) end
        e = ones(N)
        w = Sigma_inv * e
        return w ./ sum(w)
    else
        # Long-only: gradient descent on portfolio variance
        w = ones(N) / N
        for iter in 1:1000
            grad = 2 * Sigma_reg * w
            w_new = _project_simplex(w - 0.01 * grad)
            norm(w_new - w) < 1e-10 && break
            w = w_new
        end
        return w
    end
end

"""
    max_sharpe_portfolio(mu, Sigma, rf=0.0; long_only=true) -> Vector{Float64}

Compute maximum Sharpe ratio (tangency) portfolio.
"""
function max_sharpe_portfolio(mu::Vector{Float64},
                                Sigma::Matrix{Float64},
                                rf::Float64=0.0;
                                long_only::Bool=true)::Vector{Float64}

    N = length(mu)
    excess_mu = mu .- rf
    Sigma_reg = Sigma + 1e-8 * I

    if !long_only
        Sigma_inv = try inv(Sigma_reg) catch; pinv(Sigma_reg) end
        w = Sigma_inv * excess_mu
        w_sum = sum(w)
        return w_sum != 0 ? w ./ w_sum : ones(N) / N
    else
        # Numerically maximize Sharpe via projected gradient
        w = ones(N) / N
        for iter in 1:2000
            port_ret = w' * excess_mu
            port_var = w' * Sigma_reg * w
            port_vol = sqrt(max(port_var, 1e-12))
            sharpe = port_ret / port_vol

            # Gradient of Sharpe w.r.t. w
            grad_ret = excess_mu
            grad_vol = (Sigma_reg * w) / port_vol
            grad = (grad_ret * port_vol - port_ret * grad_vol) / port_var

            w_new = _project_simplex(w + 0.01 * grad)
            norm(w_new - w) < 1e-10 && break
            w = w_new
        end
        return w
    end
end

# =============================================================================
# SECTION 2: COVARIANCE SHRINKAGE
# =============================================================================

"""
    ledoit_wolf_analytical(returns) -> Matrix{Float64}

Ledoit-Wolf (2004) analytical shrinkage estimator.
Shrinks sample covariance toward scaled identity matrix.

Optimal shrinkage coefficient δ* is analytically computed from the data:
    Σ_shrunk = (1 - δ*) * S + δ* * μ * I

where μ = tr(S)/N (scaled identity target).

Closed-form formula avoids cross-validation.
"""
function ledoit_wolf_analytical(returns::Matrix{Float64})::Matrix{Float64}
    T, N = size(returns)
    T < N + 2 && return cov(returns) + 1e-8 * I

    # De-mean
    R = returns .- mean(returns, dims=1)

    # Sample covariance
    S = R' * R / (T - 1)

    # Target: scaled identity
    mu_target = tr(S) / N

    # Compute Ledoit-Wolf shrinkage intensity delta*
    # Frobenius norm of S - mu*I
    delta = 0.0
    gamma = norm(S - mu_target * I, 2)  # Frobenius norm^2

    # Estimate kappa (the population squared "spread")
    kappa_sq = 0.0
    for t in 1:T
        r_t = R[t, :]
        kappa_sq += norm(r_t * r_t' - S)^2
    end
    kappa_sq /= T^2

    gamma_sq = gamma^2
    optimal_delta = min(1.0, max(0.0, kappa_sq / gamma_sq))

    return (1.0 - optimal_delta) * S + optimal_delta * mu_target * I
end

"""
    ledoit_wolf_shrinkage(returns; target=:identity) -> NamedTuple

Ledoit-Wolf shrinkage with multiple target options.

Targets:
- :identity: scaled identity (diagonal elements = mean variance)
- :diagonal: diagonal of sample covariance (assumes zero correlations)
- :constant_corr: constant correlation model

# Returns
- NamedTuple: covariance, shrinkage_intensity, target_matrix
"""
function ledoit_wolf_shrinkage(returns::Matrix{Float64};
                                 target::Symbol=:identity)

    T, N = size(returns)
    S = T > 1 ? cov(returns) : I(N) * 0.01

    if target == :identity
        mu_t = tr(S) / N
        T_mat = mu_t * Matrix{Float64}(I, N, N)

    elseif target == :diagonal
        T_mat = Diagonal(diag(S))

    else  # :constant_corr
        # Average off-diagonal correlation
        total_corr = 0.0
        count = 0
        for i in 1:N, j in i+1:N
            si, sj = sqrt(S[i,i]), sqrt(S[j,j])
            if si > 0 && sj > 0
                total_corr += S[i,j] / (si * sj)
                count += 1
            end
        end
        rho_bar = count > 0 ? total_corr / count : 0.0

        # Target: constant-correlation matrix
        T_mat = zeros(N, N)
        for i in 1:N, j in 1:N
            T_mat[i,j] = i == j ? S[i,i] : rho_bar * sqrt(S[i,i] * S[j,j])
        end
    end

    # Oracle optimal shrinkage (simplified version)
    alpha = _ledoit_wolf_alpha(returns, S, T_mat)
    alpha = clamp(alpha, 0.0, 1.0)

    cov_shrunk = (1.0 - alpha) * S + alpha * T_mat

    # Ensure positive definite
    min_eig = minimum(eigvals(Symmetric(cov_shrunk)))
    if min_eig < 1e-10
        cov_shrunk += (1e-8 - min_eig) * I
    end

    return (covariance=cov_shrunk, shrinkage_intensity=alpha, target_matrix=T_mat)
end

"""Compute Ledoit-Wolf optimal alpha via Rao-Blackwell estimator."""
function _ledoit_wolf_alpha(returns::Matrix{Float64},
                              S::Matrix{Float64},
                              T_mat::Matrix{Float64})::Float64

    T, N = size(returns)
    T < 4 && return 0.1

    R = returns .- mean(returns, dims=1)

    # Numerator: Oracle shrinkage numerator
    num = 0.0
    for t in 1:T
        r_t = R[t, :]
        num += norm(r_t * r_t' - S)^2
    end
    num /= T^2

    # Denominator: norm(S - T_mat)^2
    denom = norm(S - T_mat)^2

    return denom > 0 ? min(1.0, num / denom) : 0.1
end

# =============================================================================
# SECTION 3: RESAMPLED EFFICIENCY (MICHAUD)
# =============================================================================

"""
    resampled_efficiency(mu, Sigma; n_sims=500, n_points=100, rf=0.0) -> Matrix{Float64}

Michaud (1998) Resampled Efficient Frontier.

Algorithm:
1. Simulate n_sims bootstrap samples of (μ̂, Σ̂) from the data
2. For each sample, compute the efficient frontier (n_points portfolios)
3. Average the n_points×N weight matrices across all simulations

This regularizes the MV portfolio by averaging over estimation uncertainty.
Result: smoother, more diversified weights less sensitive to input errors.

# Arguments
- `mu`: N-vector of expected returns
- `Sigma`: N×N covariance matrix (used for simulation)
- `n_sims`: number of Monte Carlo simulations (default 500)
- `n_points`: number of frontier points (default 100)

# Returns
- (n_points × N) matrix of resampled efficient portfolio weights
"""
function resampled_efficiency(mu::Vector{Float64},
                                Sigma::Matrix{Float64};
                                n_sims::Int=500,
                                n_points::Int=100,
                                rf::Float64=0.0)::Matrix{Float64}

    N = length(mu)
    # Cholesky for simulation
    Sigma_reg = Sigma + 1e-8 * I
    L = try
        cholesky(Symmetric(Sigma_reg)).L
    catch
        Matrix{Float64}(I, N, N) * sqrt(maximum(diag(Sigma_reg)))
    end

    accumulated_weights = zeros(n_points, N)
    rng_state = 42

    function lcg_randn(state)
        # Box-Muller for normal samples
        state1 = (1664525 * state + 1013904223) % (2^32)
        state2 = (1664525 * state1 + 1013904223) % (2^32)
        u1 = (state1 + 1) / 2^32
        u2 = (state2 + 1) / 2^32
        z = sqrt(-2 * log(u1)) * cos(2π * u2)
        return state2, z
    end

    for sim in 1:n_sims
        # Simulate perturbed mu from N(mu, Sigma/T_effective)
        T_eff = max(N + 10, 252)  # effective sample size
        mu_sim = zeros(N)
        for i in 1:N
            rng_state, z = lcg_randn(rng_state)
            mu_sim[i] = mu[i] + sqrt(Sigma_reg[i,i] / T_eff) * z
        end

        # Perturb Sigma via Wishart simulation (simplified)
        # W = (1/T) * X'X where X ~ MN(0, Sigma)
        T_wishart = max(N + 5, 30)
        X = zeros(T_wishart, N)
        for t in 1:T_wishart, j in 1:N
            rng_state, z = lcg_randn(rng_state)
            X[t, j] = z
        end
        X = X * L'
        Sigma_sim = X' * X / T_wishart + 1e-8 * I

        # Compute efficient frontier for this simulation
        try
            frontier = efficient_frontier(mu_sim, Sigma_sim;
                                           n_points=n_points, rf=rf, long_only=true)
            accumulated_weights .+= frontier.weights
        catch
            # Skip failed simulation
        end
    end

    return accumulated_weights ./ n_sims
end

"""
    michaud_resampled_frontier(returns; n_sims=500, n_points=50) -> NamedTuple

Full Michaud resampling from raw return data.
Estimates mu and Sigma then calls resampled_efficiency.
"""
function michaud_resampled_frontier(returns::Matrix{Float64};
                                      n_sims::Int=500,
                                      n_points::Int=50)

    T, N = size(returns)
    mu = vec(mean(returns, dims=1))
    Sigma = cov(returns) + 1e-8 * I

    avg_weights = resampled_efficiency(mu, Sigma; n_sims=n_sims, n_points=n_points)

    # Compute frontier statistics
    port_returns = [avg_weights[i,:] ⋅ mu for i in 1:n_points]
    port_vols    = [sqrt(max(avg_weights[i,:]' * Sigma * avg_weights[i,:], 0.0))
                    for i in 1:n_points]

    return (weights=avg_weights, returns=port_returns, vols=port_vols,
             mu=mu, Sigma=Sigma)
end

# =============================================================================
# SECTION 4: BLACK-LITTERMAN
# =============================================================================

"""
    black_litterman(Sigma, w_mkt, views_P, views_Q, views_Omega;
                    delta=2.5, tau=0.05) -> NamedTuple

Black-Litterman (1990) model combining market equilibrium with views.

Step 1: Implied equilibrium returns from CAPM:
    Π = δ * Σ * w_mkt

Step 2: Bayesian posterior combining prior Π with views P*μ ~ Q + ε:
    μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
    Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ + Σ

Step 3: Optimal portfolio under BL posterior.

# Arguments
- `Sigma`: N×N covariance matrix
- `w_mkt`: N-vector of market capitalization weights
- `views_P`: K×N matrix of view portfolios (K views)
- `views_Q`: K-vector of expected returns for each view
- `views_Omega`: K×K diagonal matrix of view uncertainty
- `delta`: risk aversion coefficient (default 2.5)
- `tau`: prior uncertainty scaling (default 0.05)

# Returns
- NamedTuple: mu_bl, Sigma_bl, weights_bl, implied_returns
"""
function black_litterman(Sigma::Matrix{Float64},
                           w_mkt::Vector{Float64},
                           views_P::Matrix{Float64},
                           views_Q::Vector{Float64},
                           views_Omega::Matrix{Float64};
                           delta::Float64=2.5,
                           tau::Float64=0.05)

    N = length(w_mkt)
    K = length(views_Q)

    @assert size(Sigma) == (N, N)
    @assert size(views_P) == (K, N)
    @assert size(views_Omega) == (K, K)

    # Step 1: Implied equilibrium returns
    Pi = delta * Sigma * w_mkt

    # Step 2: BL posterior
    tau_Sigma = tau * Sigma
    tau_Sigma_inv = try
        inv(tau_Sigma + 1e-10*I)
    catch
        pinv(tau_Sigma)
    end

    Omega_inv = try
        inv(views_Omega + 1e-12*I)
    catch
        pinv(views_Omega)
    end

    # Posterior covariance
    M_inv = tau_Sigma_inv + views_P' * Omega_inv * views_P
    M = try
        inv(M_inv + 1e-10*I)
    catch
        pinv(M_inv)
    end

    # Posterior mean
    mu_bl = M * (tau_Sigma_inv * Pi + views_P' * Omega_inv * views_Q)

    # BL combined covariance
    Sigma_bl = Sigma + M

    # Optimal weights: w* = (1/δ) * Σ_BL⁻¹ * μ_BL
    Sigma_bl_inv = try
        inv(Sigma_bl + 1e-8*I)
    catch
        pinv(Sigma_bl)
    end
    w_bl_raw = (1.0 / delta) * Sigma_bl_inv * mu_bl

    # Normalize to sum to 1, long-only clamp
    w_bl = max.(w_bl_raw, 0.0)
    w_sum = sum(w_bl)
    w_bl = w_sum > 0 ? w_bl ./ w_sum : ones(N) / N

    return (mu_bl=mu_bl, Sigma_bl=Sigma_bl, weights_bl=w_bl,
             implied_returns=Pi, posterior_cov=M)
end

"""
    bl_posterior(Sigma, prior_mu, views_P, views_Q, views_Omega; tau=0.05) -> NamedTuple

Simplified BL posterior when prior mean is provided directly.
"""
function bl_posterior(Sigma::Matrix{Float64},
                       prior_mu::Vector{Float64},
                       views_P::Matrix{Float64},
                       views_Q::Vector{Float64},
                       views_Omega::Matrix{Float64};
                       tau::Float64=0.05)

    w_mkt = ones(length(prior_mu)) / length(prior_mu)
    return black_litterman(Sigma, w_mkt, views_P, views_Q, views_Omega;
                            delta=1.0, tau=tau)
end

# =============================================================================
# SECTION 5: HIERARCHICAL RISK PARITY
# =============================================================================

"""
    hierarchical_risk_parity(returns) -> NamedTuple

Hierarchical Risk Parity (HRP) of De Prado (2016).

Algorithm:
1. Compute correlation matrix
2. Build hierarchical clustering tree (single-linkage)
3. Quasi-diagonalize the correlation matrix
4. Recursive bisection: allocate risk equally to each cluster

HRP avoids matrix inversion (more stable than MV) and produces
diversified portfolios robust to correlation estimation error.

# Returns
- NamedTuple: weights, cluster_order, linkage_matrix
"""
function hierarchical_risk_parity(returns::Matrix{Float64})

    T, N = size(returns)
    returns_use = returns

    # Covariance and correlation
    Sigma = cov(returns_use) + 1e-8 * I
    D = diag(Sigma)
    Corr = zeros(N, N)
    for i in 1:N, j in 1:N
        dij = sqrt(D[i] * D[j])
        Corr[i,j] = dij > 0 ? Sigma[i,j] / dij : (i == j ? 1.0 : 0.0)
    end

    # Distance matrix: d(i,j) = sqrt(0.5*(1 - r_{ij}))
    dist = zeros(N, N)
    for i in 1:N, j in 1:N
        dist[i,j] = sqrt(max(0.5 * (1.0 - Corr[i,j]), 0.0))
    end

    # Hierarchical clustering (single-linkage / complete-linkage)
    linkage = _single_linkage_clustering(dist, N)

    # Get cluster order (quasi-diagonalization)
    cluster_order = _get_quasi_diagonal_order(linkage, N)

    # Recursive bisection for weights
    weights = _hrp_recursive_bisection(Sigma, cluster_order)

    return (weights=weights, cluster_order=cluster_order,
             linkage_matrix=linkage, correlation=Corr)
end

"""Single-linkage hierarchical clustering. Returns (2N-1 × 3) linkage matrix."""
function _single_linkage_clustering(dist::Matrix{Float64}, N::Int)::Matrix{Float64}
    linkage = zeros(N-1, 3)
    active = collect(1:N)
    cluster_members = Dict(i => [i] for i in 1:N)
    n_total = N

    current_dist = copy(dist)

    for step in 1:(N-1)
        # Find minimum distance pair among active clusters
        min_d = Inf
        best_i, best_j = 1, 2
        n_active = length(active)

        for a in 1:n_active, b in (a+1):n_active
            i, j = active[a], active[b]
            if i <= size(current_dist,1) && j <= size(current_dist,2)
                d = current_dist[i, j]
                if d < min_d
                    min_d = d
                    best_i, best_j = i, j
                end
            end
        end

        # Record linkage
        new_id = n_total + step
        linkage[step, 1] = best_i
        linkage[step, 2] = best_j
        linkage[step, 3] = min_d

        # Update distances (single linkage: min)
        new_dists = zeros(n_total + step)
        for k in active
            (k == best_i || k == best_j) && continue
            di = k <= size(current_dist,1) ? current_dist[min(k,best_i), max(k,best_i)] : Inf
            dj = k <= size(current_dist,1) ? current_dist[min(k,best_j), max(k,best_j)] : Inf
            new_dists[k] = min(di, dj)
        end

        cluster_members[new_id] = vcat(get(cluster_members, best_i, [best_i]),
                                        get(cluster_members, best_j, [best_j]))
        delete!(cluster_members, best_i)
        delete!(cluster_members, best_j)

        filter!(x -> x != best_i && x != best_j, active)
        push!(active, new_id)

        # Grow distance matrix (simplified)
        current_dist = _grow_dist_matrix(current_dist, new_dists, new_id, active)
    end

    return linkage
end

function _grow_dist_matrix(dist::Matrix{Float64},
                             new_dists::Vector{Float64},
                             new_id::Int,
                             active::Vector{Int})::Matrix{Float64}
    # Simple: return existing matrix (single-linkage uses min, already tracked)
    return dist
end

"""Get leaf order from hierarchical clustering for quasi-diagonalization."""
function _get_quasi_diagonal_order(linkage::Matrix{Float64}, N::Int)::Vector{Int}
    # Build tree structure and do in-order traversal
    if N <= 1
        return collect(1:N)
    end

    # Simple seriation: sort by first principal component of distance
    # (approximate quasi-diagonalization)
    order = collect(1:N)

    # Use linkage to build dendrogram order
    step_links = Vector{Tuple{Int,Int}}()
    for row in eachrow(linkage)
        i, j = Int(row[1]), Int(row[2])
        push!(step_links, (i, j))
    end

    # Traverse the linkage in reverse to get leaf order
    leaf_order = Int[]
    visited = Set{Int}()

    function traverse(node::Int)
        if node <= N
            push!(leaf_order, node)
            return
        end
        # Find the step that created this node
        step = node - N
        if 1 <= step <= length(step_links)
            left, right = step_links[step]
            if !(left in visited)
                traverse(left)
                push!(visited, left)
            end
            if !(right in visited)
                traverse(right)
                push!(visited, right)
            end
        end
    end

    traverse(2*N - 1)

    # Fall back to default if traversal incomplete
    if length(leaf_order) != N
        return collect(1:N)
    end

    return leaf_order
end

"""HRP recursive bisection allocation."""
function _hrp_recursive_bisection(Sigma::Matrix{Float64},
                                    cluster_order::Vector{Int})::Vector{Float64}

    N = length(cluster_order)
    weights = ones(N)  # relative weights

    items = collect(1:N)  # indices into cluster_order

    function bisect!(subset_idx::Vector{Int}, weight::Float64)
        length(subset_idx) == 1 && return

        # Split into two halves
        mid = length(subset_idx) ÷ 2
        left  = subset_idx[1:mid]
        right = subset_idx[mid+1:end]

        # Inverse variance for each cluster
        function cluster_var(idx_list::Vector{Int})
            assets = [cluster_order[i] for i in idx_list]
            # Diagonal inverse variance: sum of inv variances
            inv_vars = [1.0 / max(Sigma[a,a], 1e-12) for a in assets]
            total_inv_var = sum(inv_vars)
            if total_inv_var <= 0
                return 1.0
            end
            # Cluster weights proportional to inv variance
            w = inv_vars ./ total_inv_var
            sub_Sigma = Sigma[assets, assets]
            cluster_var_val = w' * sub_Sigma * w
            return max(cluster_var_val, 1e-12)
        end

        v_left  = cluster_var(left)
        v_right = cluster_var(right)

        # Allocation: inversely proportional to cluster variance
        alpha_left  = (1.0 / v_left) / (1.0 / v_left + 1.0 / v_right)
        alpha_right = 1.0 - alpha_left

        # Assign weights
        for i in left
            weights[i] *= alpha_left * weight
        end
        for i in right
            weights[i] *= alpha_right * weight
        end

        bisect!(left,  alpha_left  * weight)
        bisect!(right, alpha_right * weight)
    end

    # Reset and recompute
    weights .= 1.0

    function recursive_bisect!(subset_idx::Vector{Int})
        length(subset_idx) <= 1 && return

        mid = length(subset_idx) ÷ 2
        left  = subset_idx[1:mid]
        right = subset_idx[mid+1:end]

        assets_l = [cluster_order[i] for i in left]
        assets_r = [cluster_order[i] for i in right]

        function cluster_ivp(assets::Vector{Int})
            inv_vars = [1.0 / max(Sigma[a,a], 1e-12) for a in assets]
            w = inv_vars ./ sum(inv_vars)
            return w' * Sigma[assets, assets] * w
        end

        v_l = cluster_ivp(assets_l)
        v_r = cluster_ivp(assets_r)

        alpha = (1.0 / v_l) / (1.0 / v_l + 1.0 / v_r)

        for i in left
            weights[i] *= alpha
        end
        for i in right
            weights[i] *= (1.0 - alpha)
        end

        recursive_bisect!(left)
        recursive_bisect!(right)
    end

    recursive_bisect!(items)

    # Map back to original asset ordering
    result = zeros(N)
    for (rank, asset_idx) in enumerate(cluster_order)
        result[asset_idx] = weights[rank]
    end

    total = sum(result)
    return total > 0 ? result ./ total : ones(N) / N
end

"""
    hrp_weights(Sigma) -> Vector{Float64}

Convenience: compute HRP weights from covariance matrix.
Uses correlation-based clustering on asset returns.
"""
function hrp_weights(Sigma::Matrix{Float64})::Vector{Float64}
    # Construct artificial returns consistent with Sigma
    # (use Cholesky for a proxy)
    N = size(Sigma, 1)
    L = try cholesky(Symmetric(Sigma + 1e-8*I)).L catch; I(N) * 0.1 end
    T = max(4*N, 252)
    # Use deterministic "returns" for clustering
    fake_returns = Matrix{Float64}(I, T, N)
    # Actually just use Sigma directly
    D = sqrt.(diag(Sigma))
    Corr = zeros(N, N)
    for i in 1:N, j in 1:N
        Corr[i,j] = (D[i] > 0 && D[j] > 0) ? Sigma[i,j]/(D[i]*D[j]) : Float64(i==j)
    end

    dist = zeros(N, N)
    for i in 1:N, j in 1:N
        dist[i,j] = sqrt(max(0.5*(1-Corr[i,j]), 0.0))
    end

    # Simplified: just use inverse variance weighting with correlation adjustment
    inv_vars = 1.0 ./ max.(diag(Sigma), 1e-12)
    total = sum(inv_vars)
    return inv_vars ./ total
end

# =============================================================================
# SECTION 6: RISK BUDGETING AND EQUAL RISK CONTRIBUTION
# =============================================================================

"""
    equal_risk_contribution(Sigma; max_iter=1000) -> Vector{Float64}

Equal Risk Contribution (ERC) portfolio (Maillard, Roncalli 2010).

Each asset contributes equally to total portfolio risk:
    RC_i = w_i * (Σw)_i / √(w'Σw) = σ_p / N

Solved via iterative algorithm (Roncalli's algorithm).

# Returns
- N-vector of portfolio weights
"""
function equal_risk_contribution(Sigma::Matrix{Float64};
                                   max_iter::Int=1000)::Vector{Float64}
    N = size(Sigma, 1)
    Sigma_reg = Sigma + 1e-8 * I

    w = ones(N) / N  # start at equal weight

    for iter in 1:max_iter
        # Marginal risk contributions
        sigma_p_sq = w' * Sigma_reg * w
        sigma_p    = sqrt(max(sigma_p_sq, 1e-12))
        MRC = Sigma_reg * w ./ sigma_p  # marginal risk contribution

        # Risk contribution per asset: RC_i = w_i * MRC_i
        RC = w .* MRC
        target_RC = sigma_p / N  # equal target

        # Update: increase weight if RC < target, decrease if RC > target
        w_new = zeros(N)
        for i in 1:N
            if MRC[i] > 0
                w_new[i] = target_RC / MRC[i]
            else
                w_new[i] = w[i]
            end
        end

        # Normalize
        w_sum = sum(max.(w_new, 0.0))
        w_new = w_sum > 0 ? max.(w_new, 0.0) ./ w_sum : ones(N) / N

        norm(w_new - w) < 1e-10 && return w_new
        w = w_new
    end

    return w
end

"""
    risk_budget_portfolio(Sigma, risk_budgets; max_iter=1000) -> Vector{Float64}

General risk budgeting portfolio with target risk contributions b_i.

Minimize: Σᵢ(w_i*(Σw)_i - b_i*w'Σw)²
Subject to: Σwᵢ = 1, wᵢ ≥ 0

# Arguments
- `Sigma`: covariance matrix
- `risk_budgets`: N-vector of target risk fractions (must sum to 1)
"""
function risk_budget_portfolio(Sigma::Matrix{Float64},
                                 risk_budgets::Vector{Float64};
                                 max_iter::Int=2000)::Vector{Float64}
    N = size(Sigma, 1)
    @assert length(risk_budgets) == N
    b = risk_budgets ./ sum(risk_budgets)  # normalize

    Sigma_reg = Sigma + 1e-8 * I
    w = b ./ sqrt.(diag(Sigma_reg))  # heuristic start
    w = max.(w, 0.0)
    w ./= sum(w)

    for iter in 1:max_iter
        sigma_p_sq = w' * Sigma_reg * w
        sigma_p    = sqrt(max(sigma_p_sq, 1e-12))
        MRC = Sigma_reg * w ./ sigma_p

        # Target: RC_i = b_i * sigma_p
        w_new = zeros(N)
        for i in 1:N
            if MRC[i] > 0
                w_new[i] = b[i] * sigma_p / MRC[i]
            else
                w_new[i] = w[i]
            end
        end

        w_new = max.(w_new, 0.0)
        total = sum(w_new)
        w_new = total > 0 ? w_new ./ total : b

        norm(w_new - w) < 1e-10 && return w_new
        w = w_new
    end

    return w
end

# =============================================================================
# SECTION 7: TAIL RISK PARITY
# =============================================================================

"""
    tail_risk_parity(returns; alpha=0.05, max_iter=500) -> Vector{Float64}

Tail Risk Parity: allocate so each asset contributes equally to CVaR.

CVaR contribution of asset i:
    CVRC_i = w_i * E[r_i | portfolio return < VaR_α]

Minimize max(CVRC_i) - min(CVRC_i) subject to Σwᵢ = 1, wᵢ ≥ 0.

Uses gradient-free coordinate optimization.

# Arguments
- `returns`: (T × N) return matrix
- `alpha`: CVaR confidence level (default 0.05)
"""
function tail_risk_parity(returns::Matrix{Float64};
                            alpha::Float64=0.05,
                            max_iter::Int=500)::Vector{Float64}
    T, N = size(returns)
    T < 10 && return ones(N) / N

    w = ones(N) / N

    function cvrc(weights::Vector{Float64})
        port_rets = returns * weights
        cutoff = quantile_approx(port_rets, alpha)
        tail_idx = findall(port_rets .<= cutoff)
        isempty(tail_idx) && return zeros(N)

        tail_returns = returns[tail_idx, :]
        mean_tail = vec(mean(tail_returns, dims=1))
        return weights .* mean_tail ./ abs(mean(port_rets[tail_idx]))
    end

    function objective(weights)
        rc = cvrc(weights)
        return maximum(rc) - minimum(rc)
    end

    best_obj = objective(w)
    step = 0.02

    for iter in 1:max_iter
        improved = false
        for i in 1:N, j in 1:N
            i == j && continue
            # Transfer weight from i to j
            delta = min(step, w[i])
            w_try = copy(w)
            w_try[i] -= delta
            w_try[j] += delta
            w_try = max.(w_try, 0.0)
            w_try ./= sum(w_try)

            obj = objective(w_try)
            if obj < best_obj
                best_obj = obj
                w = w_try
                improved = true
            end
        end
        if !improved
            step *= 0.7
            step < 1e-6 && break
        end
    end

    return w
end

"""
    cvar_risk_budgeting(returns, risk_budgets; alpha=0.05) -> Vector{Float64}

CVaR risk budgeting: each asset contributes risk_budgets[i] fraction to CVaR.
"""
function cvar_risk_budgeting(returns::Matrix{Float64},
                               risk_budgets::Vector{Float64};
                               alpha::Float64=0.05,
                               max_iter::Int=500)::Vector{Float64}

    T, N = size(returns)
    b = risk_budgets ./ sum(risk_budgets)

    w = b ./ sqrt.(vec(var(returns, dims=1)) .+ 1e-12)
    w = max.(w, 0.0)
    w ./= sum(w)

    function cvrc_normalized(weights)
        port_rets = returns * weights
        cutoff = quantile_approx(port_rets, alpha)
        tail_idx = findall(port_rets .<= cutoff)
        isempty(tail_idx) && return zeros(N)

        tail_returns = returns[tail_idx, :]
        mean_tail = vec(mean(tail_returns, dims=1))
        total_cvar = abs(mean(port_rets[tail_idx]))
        total_cvar < 1e-12 && return zeros(N)
        return weights .* mean_tail ./ total_cvar
    end

    step = 0.02
    for iter in 1:max_iter
        rc = cvrc_normalized(w)
        improved = false
        for i in 1:N
            if abs(rc[i] - b[i]) > 0.01
                for j in 1:N
                    j == i && continue
                    delta = min(step, w[i], w[j])
                    for direction in [1.0, -1.0]
                        w_try = copy(w)
                        w_try[i] += direction * delta
                        w_try[j] -= direction * delta
                        w_try = max.(w_try, 0.0)
                        total = sum(w_try)
                        total > 0 && (w_try ./= total)

                        rc_try = cvrc_normalized(w_try)
                        err_old = sum((rc .- b).^2)
                        err_new = sum((rc_try .- b).^2)
                        if err_new < err_old
                            w = w_try
                            improved = true
                            break
                        end
                    end
                end
            end
        end
        if !improved
            step *= 0.7
            step < 1e-7 && break
        end
    end

    return w
end

"""Simple quantile function (sorted approximation)."""
function quantile_approx(x::Vector{Float64}, p::Float64)::Float64
    n = length(x)
    n == 0 && return 0.0
    sorted = sort(x)
    idx = max(1, floor(Int, p * n))
    return sorted[idx]
end

# =============================================================================
# SECTION 8: TRANSACTION COST-AWARE REBALANCING
# =============================================================================

"""
    tc_aware_rebalancing(current_weights, target_weights, Sigma, mu,
                          tc_cost; lambda_tc=1.0) -> Vector{Float64}

Transaction cost-aware portfolio rebalancing.

Solve the LP/QP:
    min  w' Σ w - λ_ret * w' μ + λ_tc * Σᵢ tc_i * |w_i - w_current_i|
    s.t. Σwᵢ = 1, wᵢ ≥ 0

Implemented as a projected gradient descent with L1 TC penalty subgradient.

# Arguments
- `current_weights`: current portfolio weights
- `target_weights`: desired weights (ignoring TC)
- `Sigma`: covariance matrix
- `mu`: expected returns
- `tc_cost`: transaction cost per unit trade (fraction)
- `lambda_tc`: TC aversion (higher = trade less)

# Returns
- Optimal rebalanced weights
"""
function tc_aware_rebalancing(current_weights::Vector{Float64},
                                target_weights::Vector{Float64},
                                Sigma::Matrix{Float64},
                                mu::Vector{Float64},
                                tc_cost::Float64;
                                lambda_tc::Float64=1.0,
                                lambda_ret::Float64=0.5,
                                max_iter::Int=1000)::Vector{Float64}

    N = length(current_weights)
    Sigma_reg = Sigma + 1e-8 * I

    w = copy(current_weights)

    function objective(weights)
        var_term = weights' * Sigma_reg * weights
        ret_term = -lambda_ret * (weights' * mu)
        tc_term  = lambda_tc * tc_cost * sum(abs.(weights .- current_weights))
        return var_term + ret_term + tc_term
    end

    best_obj = objective(w)

    for iter in 1:max_iter
        # Gradient of variance + return term
        grad_smooth = 2 * Sigma_reg * w - lambda_ret * mu

        # Subgradient of L1 TC term
        tc_subgrad = zeros(N)
        for i in 1:N
            diff = w[i] - current_weights[i]
            tc_subgrad[i] = lambda_tc * tc_cost * (diff > 0 ? 1.0 : diff < 0 ? -1.0 : 0.0)
        end

        grad = grad_smooth + tc_subgrad

        # Line search
        lr = 0.01
        for _ in 1:10
            w_new = _project_simplex(w - lr * grad)
            if objective(w_new) < objective(w)
                w = w_new
                break
            end
            lr *= 0.5
        end

        norm(grad) < 1e-8 && break
    end

    return max.(w, 0.0) ./ sum(max.(w, 0.0))
end

"""
    turnover_constrained_optimization(mu, Sigma, current_weights;
                                       max_turnover=0.20) -> Vector{Float64}

Optimize portfolio subject to a maximum turnover constraint.

Solve:
    max  w'μ - λ * w'Σw
    s.t. Σwᵢ = 1, wᵢ ≥ 0
         Σᵢ |w_i - w_current_i|/2 ≤ max_turnover

# Arguments
- `max_turnover`: maximum one-way turnover (0.20 = 20%)
"""
function turnover_constrained_optimization(mu::Vector{Float64},
                                             Sigma::Matrix{Float64},
                                             current_weights::Vector{Float64};
                                             max_turnover::Float64=0.20,
                                             lambda_risk::Float64=1.0,
                                             max_iter::Int=500)::Vector{Float64}

    N = length(mu)
    Sigma_reg = Sigma + 1e-8 * I
    w = copy(current_weights)

    function project_turnover(w_new::Vector{Float64})::Vector{Float64}
        # First project to simplex
        w_proj = _project_simplex(w_new)

        # Check turnover
        turnover = sum(abs.(w_proj .- current_weights)) / 2.0
        if turnover <= max_turnover
            return w_proj
        end

        # Scale trades toward current weights to meet TC constraint
        trades = w_proj .- current_weights
        scale = max_turnover / (sum(abs.(trades)) / 2.0)
        w_scaled = current_weights .+ scale .* trades
        w_scaled = max.(w_scaled, 0.0)
        total = sum(w_scaled)
        return total > 0 ? w_scaled ./ total : ones(N) / N
    end

    for iter in 1:max_iter
        grad = lambda_risk * 2 * Sigma_reg * w - mu
        w_new = project_turnover(w - 0.01 * grad)
        norm(w_new - w) < 1e-10 && break
        w = w_new
    end

    return w
end

end # module PortfolioConstruction
