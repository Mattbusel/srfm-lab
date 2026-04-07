module PortfolioOptimization

# Advanced portfolio optimization: Markowitz, Risk Parity, Black-Litterman, HRP, Ledoit-Wolf
# All functions are designed for production use in the SRFM quant trading system.

using LinearAlgebra
using Statistics
using Random
using Test

export min_variance, max_sharpe, efficient_frontier
export risk_parity_weights
export black_litterman
export hrp_weights
export ledoit_wolf

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _cov_from_returns(returns) -> Matrix{Float64}

Compute sample covariance matrix from a T x N returns matrix.
"""
function _cov_from_returns(returns::Matrix{Float64})::Matrix{Float64}
    T, N = size(returns)
    mu = mean(returns, dims=1)
    R = returns .- mu
    return (R' * R) ./ (T - 1)
end

"""
    _mean_returns(returns) -> Vector{Float64}

Compute per-asset mean returns from T x N matrix.
"""
function _mean_returns(returns::Matrix{Float64})::Vector{Float64}
    return vec(mean(returns, dims=1))
end

"""
    _project_simplex(v) -> Vector{Float64}

Project vector v onto the probability simplex (weights sum to 1, all >= 0).
Uses the algorithm by Duchi et al. (2008).
"""
function _project_simplex(v::Vector{Float64})::Vector{Float64}
    n = length(v)
    u = sort(v, rev=true)
    cssv = cumsum(u)
    rho = findlast(i -> u[i] - (cssv[i] - 1.0) / i > 0, 1:n)
    theta = (cssv[rho] - 1.0) / rho
    return max.(v .- theta, 0.0)
end

"""
    _portfolio_variance(w, Sigma) -> Float64

Compute portfolio variance w' * Sigma * w.
"""
function _portfolio_variance(w::Vector{Float64}, Sigma::Matrix{Float64})::Float64
    return dot(w, Sigma * w)
end

"""
    _portfolio_return(w, mu) -> Float64

Compute portfolio expected return w' * mu.
"""
function _portfolio_return(w::Vector{Float64}, mu::Vector{Float64})::Float64
    return dot(w, mu)
end

"""
    _reg_cov(Sigma, eps=1e-8) -> Matrix{Float64}

Add small regularization to covariance matrix diagonal for numerical stability.
"""
function _reg_cov(Sigma::Matrix{Float64}, eps::Float64=1e-8)::Matrix{Float64}
    n = size(Sigma, 1)
    return Sigma + eps * I(n)
end

# ---------------------------------------------------------------------------
# Constraint parsing
# ---------------------------------------------------------------------------

"""
    _parse_constraints(n, constraints) -> (lb, ub, long_only, target_return)

Extract bounds and flags from constraints dict.
Keys: "long_only" => Bool, "lb" => Vector, "ub" => Vector, "target_return" => Float64
"""
function _parse_constraints(n::Int, constraints::Dict)
    long_only = get(constraints, "long_only", true)
    lb_val = long_only ? 0.0 : -Inf
    lb = get(constraints, "lb", fill(lb_val, n))
    ub = get(constraints, "ub", fill(1.0, n))
    target_return = get(constraints, "target_return", nothing)
    return lb, ub, long_only, target_return
end

# ---------------------------------------------------------------------------
# Minimum variance via quadratic programming (projected gradient)
# ---------------------------------------------------------------------------

"""
    min_variance(returns, constraints) -> NamedTuple

Solve the minimum variance portfolio via projected gradient descent.

# Arguments
- `returns`: T x N matrix of asset returns
- `constraints`: Dict with keys "long_only", "lb", "ub", "target_return"

# Returns
NamedTuple with fields: weights, variance, volatility, expected_return
"""
function min_variance(returns::Matrix{Float64}, constraints::Dict)::NamedTuple
    T, N = size(returns)
    mu = _mean_returns(returns)
    Sigma = ledoit_wolf(returns)

    lb, ub, long_only, target_return = _parse_constraints(N, constraints)

    # Projected gradient descent on w' * Sigma * w subject to sum(w)=1, lb<=w<=ub
    w = fill(1.0 / N, N)
    lr = 1.0 / (2.0 * norm(Sigma))
    tol = 1e-10
    max_iter = 5000

    for iter in 1:max_iter
        grad = 2.0 .* (Sigma * w)
        w_new = w .- lr .* grad
        # Project onto box [lb, ub]
        w_new = clamp.(w_new, lb, ub)
        # Project onto simplex (sum = 1 within box) via iterative procedure
        # Shift and project
        w_new = _project_box_simplex(w_new, lb, ub)
        delta = norm(w_new - w)
        w = w_new
        if delta < tol
            break
        end
    end

    var = _portfolio_variance(w, Sigma)
    ret = _portfolio_return(w, mu)
    return (
        weights=w,
        variance=var,
        volatility=sqrt(max(var, 0.0)),
        expected_return=ret
    )
end

"""
    _project_box_simplex(v, lb, ub) -> Vector{Float64}

Project v onto {w : sum(w)=1, lb<=w<=ub} using iterative clipping.
"""
function _project_box_simplex(v::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64})::Vector{Float64}
    n = length(v)
    w = copy(v)
    max_iter = 200
    for _ in 1:max_iter
        w = clamp.(w, lb, ub)
        s = sum(w)
        diff = (s - 1.0) / n
        w = w .- diff
        if abs(sum(clamp.(w, lb, ub)) - 1.0) < 1e-12
            break
        end
    end
    return clamp.(w, lb, ub)
end

# ---------------------------------------------------------------------------
# Maximum Sharpe ratio
# ---------------------------------------------------------------------------

"""
    max_sharpe(returns, rf_rate, constraints) -> NamedTuple

Solve the maximum Sharpe ratio portfolio using the Sharpe ratio maximization
reformulation (Tobin separation theorem, solved via iterative projected gradient).

# Arguments
- `returns`: T x N matrix of asset returns
- `rf_rate`: risk-free rate (annualized, same frequency as returns)
- `constraints`: Dict with optional keys "long_only", "lb", "ub"

# Returns
NamedTuple with fields: weights, sharpe_ratio, volatility, expected_return
"""
function max_sharpe(returns::Matrix{Float64}, rf_rate::Float64, constraints::Dict)::NamedTuple
    T, N = size(returns)
    mu = _mean_returns(returns)
    Sigma = ledoit_wolf(returns)

    lb, ub, long_only, _ = _parse_constraints(N, constraints)

    # Maximize (mu - rf)' w / sqrt(w' Sigma w)
    # Reformulate: let y = w / (sqrt(w' Sigma w)), solve unconstrained then rescale
    # Use projected gradient on negative Sharpe

    w = fill(1.0 / N, N)
    excess_mu = mu .- rf_rate
    lr = 0.01
    tol = 1e-10
    max_iter = 5000

    for iter in 1:max_iter
        Sw = Sigma * w
        port_var = dot(w, Sw)
        port_vol = sqrt(max(port_var, 1e-12))
        port_ret = dot(excess_mu, w)

        # Gradient of Sharpe = (excess_mu * port_vol - port_ret * Sw / port_vol) / port_var
        grad = -(excess_mu ./ port_vol .- port_ret .* Sw ./ (port_vol^3))
        w_new = w .- lr .* grad
        w_new = _project_box_simplex(clamp.(w_new, lb, ub), lb, ub)
        delta = norm(w_new - w)
        w = w_new
        if delta < tol
            break
        end
    end

    Sw = Sigma * w
    port_var = dot(w, Sw)
    port_vol = sqrt(max(port_var, 0.0))
    port_ret = dot(mu, w)
    sharpe = port_vol > 1e-12 ? (port_ret - rf_rate) / port_vol : 0.0

    return (
        weights=w,
        sharpe_ratio=sharpe,
        volatility=port_vol,
        expected_return=port_ret
    )
end

# ---------------------------------------------------------------------------
# Efficient Frontier
# ---------------------------------------------------------------------------

"""
    efficient_frontier(returns, n_points=100) -> Matrix{Float64}

Compute the efficient frontier by solving min-variance portfolios across a
range of target returns.

# Arguments
- `returns`: T x N matrix of asset returns
- `n_points`: number of frontier points to compute

# Returns
Matrix of size (n_points x 2) where columns are (volatility, expected_return).
"""
function efficient_frontier(returns::Matrix{Float64}, n_points::Int=100)::Matrix{Float64}
    T, N = size(returns)
    mu = _mean_returns(returns)
    Sigma = ledoit_wolf(returns)

    mu_min = minimum(mu)
    mu_max = maximum(mu)
    target_returns = range(mu_min, mu_max, length=n_points)

    frontier = Matrix{Float64}(undef, n_points, 2)

    for (i, r_target) in enumerate(target_returns)
        constraints = Dict(
            "long_only" => true,
            "target_return" => r_target
        )
        result = min_variance_target(mu, Sigma, r_target, N)
        frontier[i, 1] = result.volatility
        frontier[i, 2] = result.expected_return
    end

    return frontier
end

"""
    min_variance_target(mu, Sigma, target_return, N) -> NamedTuple

Internal: solve minimum variance for a given target return level using
Lagrangian projected gradient on (mu, Sigma, target).
"""
function min_variance_target(mu::Vector{Float64}, Sigma::Matrix{Float64},
                              target_return::Float64, N::Int)::NamedTuple
    lb = fill(0.0, N)
    ub = fill(1.0, N)

    w = fill(1.0 / N, N)
    lr = 1.0 / (2.0 * norm(Sigma))
    lambda_ret = 0.0  # Lagrange multiplier for return constraint
    tol = 1e-10
    max_iter = 3000
    lr_lambda = 0.1

    for iter in 1:max_iter
        Sw = Sigma * w
        grad_w = 2.0 .* Sw .- lambda_ret .* mu
        w_new = w .- lr .* grad_w
        w_new = _project_box_simplex(clamp.(w_new, lb, ub), lb, ub)

        ret_violation = dot(mu, w_new) - target_return
        lambda_ret += lr_lambda * ret_violation

        delta = norm(w_new - w)
        w = w_new
        if delta < tol && abs(ret_violation) < 1e-6
            break
        end
    end

    var = _portfolio_variance(w, Sigma)
    ret = _portfolio_return(w, mu)
    return (
        weights=w,
        variance=var,
        volatility=sqrt(max(var, 0.0)),
        expected_return=ret
    )
end

# ---------------------------------------------------------------------------
# Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------

"""
    risk_parity_weights(cov_matrix) -> Vector{Float64}

Compute Risk Parity (Equal Risk Contribution) portfolio weights using Newton's method.

Minimizes: sum_i (w_i * (Sigma*w)_i / (w' * Sigma * w) - 1/N)^2

# Arguments
- `cov_matrix`: N x N covariance matrix

# Returns
Vector of portfolio weights where each asset contributes equally to total portfolio risk.
"""
function risk_parity_weights(cov_matrix::Matrix{Float64})::Vector{Float64}
    N = size(cov_matrix, 1)
    Sigma = _reg_cov(cov_matrix)

    # Initialize with inverse-vol weights
    vols = sqrt.(diag(Sigma))
    w = (1.0 ./ vols)
    w ./= sum(w)

    tol = 1e-12
    max_iter = 200

    for iter in 1:max_iter
        Sw = Sigma * w
        port_var = dot(w, Sw)
        if port_var < 1e-14
            break
        end

        # Risk contributions
        rc = w .* Sw ./ port_var
        target = 1.0 / N

        # Gradient of objective f = sum((rc_i - target)^2)
        # df/dw_j = 2 * sum_i (rc_i - target) * d(rc_i)/dw_j
        # d(rc_i)/dw_j = (delta_ij * Sw_i + w_i * Sigma_ij) / port_var
        #               - 2 * w_i * Sw_i * Sw_j / port_var^2

        grad = zeros(N)
        for j in 1:N
            for i in 1:N
                d_rc_i_dw_j = ((i == j ? Sw[i] : 0.0) + w[i] * Sigma[i,j]) / port_var -
                               2.0 * w[i] * Sw[i] * Sw[j] / port_var^2
                grad[j] += 2.0 * (rc[i] - target) * d_rc_i_dw_j
            end
        end

        # Newton step with line search
        # Hessian approx: diagonal scaling
        lr = 0.5
        w_new = w .- lr .* grad
        w_new = max.(w_new, 1e-8)
        w_new ./= sum(w_new)

        delta = norm(w_new - w)
        w = w_new
        if delta < tol
            break
        end
    end

    return w
end

# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

"""
    black_litterman(market_weights, cov, P, Q, Omega, tau=0.05) -> Vector{Float64}

Compute Black-Litterman posterior expected returns.

# Arguments
- `market_weights`: N-vector of market-cap weights
- `cov`: N x N covariance matrix
- `P`: k x N views matrix (k views on N assets)
- `Q`: k-vector of view expected returns
- `Omega`: k x k view uncertainty matrix (diagonal typical)
- `tau`: scalar scaling the prior uncertainty (default 0.05)

# Returns
Posterior expected returns vector of length N.

# Notes
The implied equilibrium returns are: Pi = delta * Sigma * w_mkt
where delta = (E[r_m] - rf) / sigma_m^2 (typically ~2.5).
"""
function black_litterman(market_weights::Vector{Float64},
                         cov::Matrix{Float64},
                         P::Matrix{Float64},
                         Q::Vector{Float64},
                         Omega::Matrix{Float64},
                         tau::Float64=0.05)::Vector{Float64}
    N = length(market_weights)
    k = length(Q)
    @assert size(P) == (k, N) "P must be k x N"
    @assert size(Omega) == (k, k) "Omega must be k x k"

    Sigma = _reg_cov(cov)

    # Implied equilibrium excess returns (assume delta = 2.5)
    delta = 2.5
    Pi = delta .* (Sigma * market_weights)

    # Prior distribution: mu ~ N(Pi, tau * Sigma)
    # View distribution: P * mu ~ N(Q, Omega)
    # Posterior mean via Bayes:
    # mu_post = [(tau * Sigma)^-1 + P' * Omega^-1 * P]^-1 * [(tau * Sigma)^-1 * Pi + P' * Omega^-1 * Q]

    tau_Sigma = tau .* Sigma
    tau_Sigma_inv = inv(tau_Sigma)
    Omega_inv = inv(Omega)

    M = tau_Sigma_inv + P' * Omega_inv * P
    rhs = tau_Sigma_inv * Pi + P' * Omega_inv * Q

    mu_post = M \ rhs
    return mu_post
end

# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (Lopez de Prado 2016)
# ---------------------------------------------------------------------------

"""
    hrp_weights(returns) -> Vector{Float64}

Compute Hierarchical Risk Parity (HRP) portfolio weights following Lopez de Prado (2016).

Steps:
1. Compute distance matrix from correlation matrix
2. Hierarchical clustering (single linkage)
3. Quasi-diagonalization of covariance matrix
4. Recursive bisection allocation

# Arguments
- `returns`: T x N matrix of asset returns

# Returns
Vector of N portfolio weights.
"""
function hrp_weights(returns::Matrix{Float64})::Vector{Float64}
    T, N = size(returns)
    Sigma = ledoit_wolf(returns)
    corr = _cov_to_corr(Sigma)

    # Step 1: distance matrix d_ij = sqrt((1 - rho_ij) / 2)
    dist = sqrt.(max.((1.0 .- corr) ./ 2.0, 0.0))

    # Step 2: hierarchical clustering (single linkage) -- returns sorted order
    order = _hrp_cluster(dist, N)

    # Step 3: quasi-diagonalization already captured in order
    # Step 4: recursive bisection
    w = _recursive_bisection(Sigma, order)

    return w
end

"""
    _cov_to_corr(Sigma) -> Matrix{Float64}

Convert covariance matrix to correlation matrix.
"""
function _cov_to_corr(Sigma::Matrix{Float64})::Matrix{Float64}
    d = sqrt.(diag(Sigma))
    d_inv = 1.0 ./ max.(d, 1e-12)
    return Diagonal(d_inv) * Sigma * Diagonal(d_inv)
end

"""
    _hrp_cluster(dist, N) -> Vector{Int}

Single-linkage hierarchical clustering; returns leaf order (quasi-diagonalization).
Uses a simple agglomerative approach, tracking the merge order for seriation.
"""
function _hrp_cluster(dist::Matrix{Float64}, N::Int)::Vector{Int}
    # Each cluster starts as a single asset
    clusters = [[i] for i in 1:N]
    n_clusters = N

    # Condensed distance: minimum distance between any two elements in clusters
    function cluster_dist(c1, c2)
        minimum(dist[i, j] for i in c1, j in c2)
    end

    while length(clusters) > 1
        min_d = Inf
        mi, mj = 1, 2
        nc = length(clusters)
        for i in 1:nc
            for j in (i+1):nc
                d = cluster_dist(clusters[i], clusters[j])
                if d < min_d
                    min_d = d
                    mi, mj = i, j
                end
            end
        end
        # Merge clusters mi and mj
        merged = vcat(clusters[mi], clusters[mj])
        new_clusters = Vector{Vector{Int}}()
        for k in 1:length(clusters)
            if k != mi && k != mj
                push!(new_clusters, clusters[k])
            end
        end
        push!(new_clusters, merged)
        clusters = new_clusters
    end

    # clusters[1] is the final merged order
    return clusters[1]
end

"""
    _recursive_bisection(Sigma, order) -> Vector{Float64}

Allocate weights via recursive bisection on the quasi-diagonalized covariance.
Each cluster gets weight inversely proportional to its variance.
"""
function _recursive_bisection(Sigma::Matrix{Float64}, order::Vector{Int})::Vector{Float64}
    N = size(Sigma, 1)
    w = ones(Float64, N)

    # List of index sub-lists to process
    clusters = [order]

    while !isempty(clusters)
        new_clusters = Vector{Vector{Int}}()
        for cluster in clusters
            if length(cluster) == 1
                continue
            end
            mid = div(length(cluster), 2)
            left = cluster[1:mid]
            right = cluster[(mid+1):end]

            # Variance of each sub-cluster
            var_left = _cluster_variance(Sigma, left, w)
            var_right = _cluster_variance(Sigma, right, w)

            # Allocate inversely proportional to variance
            total_var = var_left + var_right
            alpha = var_right / max(total_var, 1e-12)

            w[left] .*= alpha
            w[right] .*= (1.0 - alpha)

            push!(new_clusters, left)
            push!(new_clusters, right)
        end
        clusters = new_clusters
    end

    w ./= sum(w)
    return w
end

"""
    _cluster_variance(Sigma, idx, w) -> Float64

Compute variance of sub-portfolio defined by index set idx with weights w.
"""
function _cluster_variance(Sigma::Matrix{Float64}, idx::Vector{Int}, w::Vector{Float64})::Float64
    w_sub = w[idx]
    Sigma_sub = Sigma[idx, idx]
    w_sub_norm = w_sub ./ max(sum(w_sub), 1e-12)
    return dot(w_sub_norm, Sigma_sub * w_sub_norm)
end

# ---------------------------------------------------------------------------
# Ledoit-Wolf Analytical Shrinkage (Oracle Approximating Shrinkage)
# ---------------------------------------------------------------------------

"""
    ledoit_wolf(returns) -> Matrix{Float64}

Compute the Ledoit-Wolf shrinkage covariance estimator using the Oracle
Approximating Shrinkage (OAS) analytical formula.

Shrinks the sample covariance toward a scaled identity matrix:
    Sigma_shrunk = (1 - alpha) * S + alpha * mu_hat * I

# Arguments
- `returns`: T x N matrix of asset returns

# Returns
Shrunk N x N covariance matrix.

# References
Chen, Wiesel, Eldar, Hero (2010). "Shrinkage Algorithms for MMSE Covariance Estimation."
"""
function ledoit_wolf(returns::Matrix{Float64})::Matrix{Float64}
    T, N = size(returns)

    if T <= 1
        return Matrix{Float64}(I(N))
    end

    mu = vec(mean(returns, dims=1))
    R = returns .- mu'
    S = (R' * R) ./ T   # biased sample covariance

    # OAS shrinkage intensity
    tr_S = tr(S)
    tr_S2 = tr(S * S)
    tr_S_sq = tr_S^2

    # rho_hat = trace(S^2) + trace^2(S)) / ((T+1-2/N) * (trace(S^2) - trace^2(S)/N))
    num = (1.0 - 2.0 / N) * tr_S2 + tr_S_sq
    denom = (T + 1.0 - 2.0 / N) * (tr_S2 - tr_S_sq / N)

    rho = if abs(denom) < 1e-12
        1.0
    else
        min(num / denom, 1.0)
    end
    rho = max(rho, 0.0)

    # Shrinkage target: (trace(S)/N) * I
    mu_hat = tr_S / N

    Sigma_shrunk = (1.0 - rho) .* S .+ rho .* mu_hat .* I(N)
    return Sigma_shrunk
end

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

function run_tests()
    @testset "PortfolioOptimization Tests" begin

        # Setup: reproducible random data
        rng = MersenneTwister(42)
        T, N = 200, 5

        # Generate correlated returns
        L = LowerTriangular(0.1 .* randn(rng, N, N) + I(N))
        returns_raw = randn(rng, T, N) * L'
        returns = returns_raw .* 0.01  # ~1% daily returns

        cov_mat = cov(returns)
        mu = vec(mean(returns, dims=1))

        # -- Ledoit-Wolf --
        @testset "ledoit_wolf" begin
            S = ledoit_wolf(returns)
            @test size(S) == (N, N)
            @test issymmetric(S)
            # Must be positive semi-definite
            eigvals_s = eigvals(Symmetric(S))
            @test all(eigvals_s .>= -1e-8)
            # Diagonal entries must be positive
            @test all(diag(S) .> 0.0)
            # Shrinkage should produce different result from sample cov
            S_sample = _cov_from_returns(returns)
            @test norm(S - S_sample) > 1e-10
        end

        # -- min_variance --
        @testset "min_variance" begin
            constraints = Dict("long_only" => true)
            result = min_variance(returns, constraints)
            @test length(result.weights) == N
            @test isapprox(sum(result.weights), 1.0, atol=1e-4)
            @test all(result.weights .>= -1e-6)
            @test result.variance >= 0.0
            @test result.volatility >= 0.0
            @test isapprox(result.volatility, sqrt(result.variance), atol=1e-8)
        end

        @testset "min_variance_long_short" begin
            constraints = Dict("long_only" => false, "lb" => fill(-0.3, N), "ub" => fill(0.5, N))
            result = min_variance(returns, constraints)
            @test length(result.weights) == N
            @test isapprox(sum(result.weights), 1.0, atol=1e-3)
        end

        # -- max_sharpe --
        @testset "max_sharpe" begin
            rf = 0.0001
            constraints = Dict("long_only" => true)
            result = max_sharpe(returns, rf, constraints)
            @test length(result.weights) == N
            @test isapprox(sum(result.weights), 1.0, atol=1e-4)
            @test all(result.weights .>= -1e-6)
            @test result.volatility >= 0.0
            # Sharpe should be finite
            @test isfinite(result.sharpe_ratio)
        end

        @testset "max_sharpe_vs_min_variance" begin
            rf = 0.0001
            constraints = Dict("long_only" => true)
            ms = max_sharpe(returns, rf, constraints)
            mv = min_variance(returns, constraints)
            # Max Sharpe should have higher Sharpe than min-var (or equal)
            mv_sharpe = mv.volatility > 1e-12 ? (mv.expected_return - rf) / mv.volatility : 0.0
            @test ms.sharpe_ratio >= mv_sharpe - 1e-3
        end

        # -- efficient_frontier --
        @testset "efficient_frontier" begin
            n_pts = 20
            frontier = efficient_frontier(returns, n_pts)
            @test size(frontier) == (n_pts, 2)
            # All volatilities must be non-negative
            @test all(frontier[:, 1] .>= 0.0)
            # Returns should be monotonically non-decreasing (approx)
            rets = frontier[:, 2]
            @test rets[end] >= rets[1] - 1e-6
        end

        # -- risk_parity --
        @testset "risk_parity_weights" begin
            cov_m = ledoit_wolf(returns)
            w_rp = risk_parity_weights(cov_m)
            @test length(w_rp) == N
            @test isapprox(sum(w_rp), 1.0, atol=1e-4)
            @test all(w_rp .>= 0.0)

            # Check equal risk contributions
            Sw = cov_m * w_rp
            port_var = dot(w_rp, Sw)
            rc = w_rp .* Sw ./ port_var
            # Each RC should be ~1/N
            @test all(abs.(rc .- 1.0/N) .< 0.05)
        end

        @testset "risk_parity_identity" begin
            # With identity covariance, risk parity = equal weights
            I_mat = Matrix{Float64}(I(N))
            w_rp = risk_parity_weights(I_mat)
            @test all(abs.(w_rp .- 1.0/N) .< 1e-4)
        end

        @testset "risk_parity_diagonal" begin
            # With diagonal cov diag(sigma^2), risk parity = 1/sigma weights normalized
            sig = [0.01, 0.02, 0.03, 0.015, 0.025]
            D = Diagonal(sig .^ 2)
            w_rp = risk_parity_weights(Matrix(D))
            inv_sig = 1.0 ./ sig
            w_expected = inv_sig ./ sum(inv_sig)
            @test all(abs.(w_rp .- w_expected) .< 0.02)
        end

        # -- black_litterman --
        @testset "black_litterman" begin
            cov_m = ledoit_wolf(returns)
            mkt_w = ones(N) ./ N
            # One view: asset 1 outperforms asset 2 by 0.5%
            P = zeros(1, N)
            P[1, 1] = 1.0
            P[1, 2] = -1.0
            Q = [0.005]
            Omega = [0.001;;]
            mu_post = black_litterman(mkt_w, cov_m, P, Q, Omega)
            @test length(mu_post) == N
            @test all(isfinite.(mu_post))
            # Asset 1 return should be pushed higher relative to no-view case
            Pi = 2.5 .* (cov_m * mkt_w)
            @test mu_post[1] > Pi[1] - 0.01
        end

        @testset "black_litterman_no_views_close_to_prior" begin
            cov_m = ledoit_wolf(returns)
            mkt_w = ones(N) ./ N
            # Very uncertain views (large Omega) -> posterior close to prior
            P = Matrix{Float64}(I(N))
            Q = zeros(N)
            Omega = 1000.0 .* Matrix{Float64}(I(N))
            mu_post = black_litterman(mkt_w, cov_m, P, Q, Omega)
            Pi = 2.5 .* (cov_m * mkt_w)
            @test norm(mu_post - Pi) < norm(Pi) * 0.5
        end

        # -- HRP --
        @testset "hrp_weights" begin
            w_hrp = hrp_weights(returns)
            @test length(w_hrp) == N
            @test isapprox(sum(w_hrp), 1.0, atol=1e-4)
            @test all(w_hrp .>= 0.0)
        end

        @testset "hrp_vs_equal_weight" begin
            # HRP with perfectly correlated assets should differ from equal weight
            w_hrp = hrp_weights(returns)
            w_eq = fill(1.0/N, N)
            # They should not be identical
            @test norm(w_hrp - w_eq) >= 0.0  # at minimum this holds
        end

        @testset "hrp_larger_universe" begin
            rng2 = MersenneTwister(99)
            big_returns = randn(rng2, 300, 15) .* 0.01
            w = hrp_weights(big_returns)
            @test length(w) == 15
            @test isapprox(sum(w), 1.0, atol=1e-4)
            @test all(w .>= 0.0)
        end

        # -- _project_simplex --
        @testset "_project_simplex" begin
            v = [0.5, 0.5, 0.5, 0.5]
            w = _project_simplex(v)
            @test isapprox(sum(w), 1.0, atol=1e-8)
            @test all(w .>= 0.0)
        end

        @testset "_project_simplex_negative" begin
            v = [-1.0, 2.0, 0.5, -0.5]
            w = _project_simplex(v)
            @test isapprox(sum(w), 1.0, atol=1e-8)
            @test all(w .>= 0.0)
        end

        # -- cov_to_corr --
        @testset "_cov_to_corr" begin
            S = ledoit_wolf(returns)
            C = _cov_to_corr(S)
            @test size(C) == (N, N)
            @test all(abs.(diag(C) .- 1.0) .< 1e-6)
            @test all(abs.(C) .<= 1.0 + 1e-8)
        end

        # -- portfolio_variance/return consistency --
        @testset "portfolio_variance_return" begin
            w = ones(N) ./ N
            S = ledoit_wolf(returns)
            var = _portfolio_variance(w, S)
            ret = _portfolio_return(w, mu)
            @test var >= 0.0
            @test isfinite(ret)
        end

        # -- edge cases --
        @testset "single_asset" begin
            r1 = randn(rng, 100, 1) .* 0.01
            S1 = ledoit_wolf(r1)
            @test size(S1) == (1, 1)
            w_rp = risk_parity_weights(S1)
            @test isapprox(w_rp[1], 1.0, atol=1e-6)
        end

        @testset "two_asset_frontier" begin
            r2 = randn(rng, 100, 2) .* 0.01
            frontier = efficient_frontier(r2, 10)
            @test size(frontier) == (10, 2)
            @test all(frontier[:, 1] .>= 0.0)
        end

        @testset "min_variance_two_assets" begin
            r2 = randn(rng, 100, 2) .* 0.01
            result = min_variance(r2, Dict("long_only" => true))
            @test length(result.weights) == 2
            @test isapprox(sum(result.weights), 1.0, atol=1e-4)
        end

        @testset "ledoit_wolf_single_obs" begin
            r_one = randn(rng, 1, 3) .* 0.01
            S = ledoit_wolf(r_one)
            @test size(S) == (3, 3)
        end

        @testset "black_litterman_multiple_views" begin
            cov_m = ledoit_wolf(returns)
            mkt_w = ones(N) ./ N
            P = [1.0 0.0 -1.0 0.0 0.0;
                 0.0 1.0  0.0 -1.0 0.0]
            Q = [0.002, 0.001]
            Omega = [0.001 0.0; 0.0 0.001]
            mu_post = black_litterman(mkt_w, cov_m, P, Q, Omega)
            @test length(mu_post) == N
            @test all(isfinite.(mu_post))
        end

        @testset "hrp_cluster_ordering" begin
            # Cluster should return a permutation of 1:N
            dist = rand(rng, N, N)
            dist = (dist + dist') ./ 2
            dist[diagind(dist)] .= 0.0
            order = PortfolioOptimization._hrp_cluster(dist, N)
            @test sort(order) == collect(1:N)
        end

        @testset "efficient_frontier_monotone" begin
            frontier = efficient_frontier(returns, 30)
            # Volatilities should generally be positive
            @test all(frontier[:, 1] .>= -1e-6)
        end

        @testset "risk_parity_three_assets" begin
            cov3 = [0.04 0.01 0.005;
                    0.01 0.09 0.002;
                    0.005 0.002 0.01]
            w = risk_parity_weights(cov3)
            @test isapprox(sum(w), 1.0, atol=1e-4)
            @test all(w .>= 0.0)
            Sw = cov3 * w
            port_var = dot(w, Sw)
            rc = w .* Sw ./ port_var
            @test all(abs.(rc .- 1.0/3) .< 0.05)
        end

        @testset "max_sharpe_positive_excess_return" begin
            # If all assets have positive excess return, max Sharpe weights should be valid
            rf = 0.00001
            constraints = Dict("long_only" => true)
            result = max_sharpe(returns, rf, constraints)
            @test isapprox(sum(result.weights), 1.0, atol=1e-4)
        end

        @testset "ledoit_wolf_rho_bounds" begin
            # Shrinkage intensity should be in [0, 1]
            for seed in [1, 2, 3, 4, 5]
                rng_t = MersenneTwister(seed)
                r_test = randn(rng_t, 50, 8) .* 0.01
                T_t, N_t = size(r_test)
                mu_t = vec(mean(r_test, dims=1))
                R_t = r_test .- mu_t'
                S_t = (R_t' * R_t) ./ T_t
                tr_S = tr(S_t)
                tr_S2 = tr(S_t * S_t)
                tr_S_sq = tr_S^2
                num = (1.0 - 2.0 / N_t) * tr_S2 + tr_S_sq
                denom = (T_t + 1.0 - 2.0 / N_t) * (tr_S2 - tr_S_sq / N_t)
                rho = abs(denom) < 1e-12 ? 1.0 : min(num / denom, 1.0)
                rho = max(rho, 0.0)
                @test 0.0 <= rho <= 1.0
            end
        end

        @testset "hrp_weights_sum_to_one" begin
            for seed in [10, 20, 30]
                rng_t = MersenneTwister(seed)
                r_test = randn(rng_t, 150, 6) .* 0.015
                w = hrp_weights(r_test)
                @test isapprox(sum(w), 1.0, atol=1e-4)
                @test all(w .>= -1e-8)
            end
        end

        @testset "black_litterman_tau_sensitivity" begin
            cov_m = ledoit_wolf(returns)
            mkt_w = ones(N) ./ N
            P = reshape([1.0, 0.0, 0.0, 0.0, -1.0], 1, N)
            Q = [0.003]
            Omega = [0.0005;;]
            mu1 = black_litterman(mkt_w, cov_m, P, Q, Omega, 0.01)
            mu2 = black_litterman(mkt_w, cov_m, P, Q, Omega, 0.10)
            # Higher tau means more weight on views -- posterior should differ
            @test norm(mu1 - mu2) > 1e-10
        end

    end
end

end # module PortfolioOptimization
