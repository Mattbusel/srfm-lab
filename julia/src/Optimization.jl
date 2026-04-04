"""
SRFMOptimization — Portfolio and parameter optimization for quant research.

Implements: Mean-Variance (MVO), Risk Parity, Hierarchical Risk Parity,
Black-Litterman, grid search, Bayesian optimization, walk-forward optimization.
"""
module SRFMOptimization

using LinearAlgebra
using Statistics
using Optim
using DataFrames
using Clustering
using Distances
using Random

export min_variance, max_sharpe, efficient_frontier
export risk_parity, hierarchical_risk_parity
export black_litterman
export grid_search, bayesian_optimize, walk_forward_optimize
export equal_weight, inverse_vol_weight, max_diversification

# ─────────────────────────────────────────────────────────────────────────────
# 1. Mean-Variance Optimization
# ─────────────────────────────────────────────────────────────────────────────

"""
    min_variance(Σ; constraints=nothing) → Vector{Float64}

Minimum variance portfolio. Constraints: nothing = long-only sum-to-1.
"""
function min_variance(Σ::Matrix{Float64};
                       constraints::Union{Nothing, Dict}=nothing)::Vector{Float64}
    n = size(Σ, 1)
    @assert size(Σ, 2) == n "Σ must be square"

    # Analytic solution for unconstrained (long-only handled by projection)
    long_only = true
    if !isnothing(constraints) && haskey(constraints, :long_only)
        long_only = constraints[:long_only]
    end

    if long_only
        return _min_var_constrained(Σ, n)
    else
        # Analytic: w ∝ Σ⁻¹ 1
        ones_vec = ones(n)
        Σ_inv = inv(Σ + 1e-8 * I(n))
        w = Σ_inv * ones_vec
        return w ./ sum(w)
    end
end

function _min_var_constrained(Σ::Matrix{Float64}, n::Int)::Vector{Float64}
    # Quadratic programming via Optim: minimize w'Σw subject to sum(w)=1, w≥0
    function portfolio_var(w)
        return dot(w, Σ * w)
    end

    # Start: equal weight
    w0 = ones(n) / n

    result = optimize(
        portfolio_var,
        zeros(n), ones(n), w0,
        Fminbox(LBFGS()),
        Optim.Options(iterations=5000, g_tol=1e-10)
    )

    w = Optim.minimizer(result)
    # Project to sum = 1
    w = max.(w, 0.0)
    s = sum(w)
    s < 1e-10 && return ones(n) / n
    return w ./ s
end

"""
    max_sharpe(μ, Σ, rf; long_only=true) → Vector{Float64}

Maximum Sharpe ratio portfolio. Tangency portfolio on the CML.
"""
function max_sharpe(μ::Vector{Float64}, Σ::Matrix{Float64}, rf::Float64=0.0;
                     long_only::Bool=true)::Vector{Float64}
    n = length(μ)
    excess = μ .- rf

    if long_only
        function neg_sharpe(w)
            w_pos = max.(w, 0.0)
            s = sum(w_pos)
            s < 1e-10 && return 0.0
            w_norm = w_pos ./ s
            port_ret = dot(w_norm, μ) - rf
            port_var = dot(w_norm, Σ * w_norm)
            port_var < 1e-12 && return -1e6
            return -(port_ret / sqrt(port_var))
        end

        w0 = ones(n) / n
        result = optimize(neg_sharpe, zeros(n), ones(n), w0,
                          Fminbox(LBFGS()),
                          Optim.Options(iterations=5000, g_tol=1e-10))
        w = max.(Optim.minimizer(result), 0.0)
    else
        # Analytic tangency: w ∝ Σ⁻¹ (μ - rf)
        Σ_reg = Σ + 1e-8 * I(n)
        w = inv(Σ_reg) * excess
    end

    s = sum(w)
    s < 1e-10 && return ones(n) / n
    return w ./ s
end

"""
    efficient_frontier(μ, Σ, n_points=100) → Vector{NamedTuple}

Trace the Markowitz efficient frontier from min-variance to max-return.
Each point: (target_return, risk, weights, sharpe).
"""
function efficient_frontier(μ::Vector{Float64}, Σ::Matrix{Float64},
                              n_points::Int=100)::Vector{NamedTuple}
    n = length(μ)

    # Range: from min-var return to max-return portfolio
    w_min = min_variance(Σ)
    ret_min = dot(w_min, μ)
    ret_max = maximum(μ)

    target_rets = range(ret_min, ret_max, length=n_points)
    frontier    = NamedTuple[]

    Σ_reg = Σ + 1e-8 * I(n)

    for target in target_rets
        # Minimize variance subject to: w'μ = target, sum(w) = 1, w ≥ 0
        function obj(w)
            return dot(w, Σ_reg * w)
        end

        # Start from equal weight, nudge toward high-return assets
        w0 = (μ ./ sum(μ)) .* 0.5 .+ ones(n) / n .* 0.5
        w0 .= max.(w0, 0.0); w0 ./= sum(w0)

        result = optimize(
            obj, zeros(n), ones(n), w0,
            Fminbox(LBFGS()),
            Optim.Options(iterations=2000, g_tol=1e-9)
        )

        w    = max.(Optim.minimizer(result), 0.0)
        s    = sum(w)
        s > 1e-10 && (w ./= s)

        port_ret = dot(w, μ)
        port_var = dot(w, Σ_reg * w)
        port_std = sqrt(max(port_var, 0.0))
        sharpe   = port_std > 1e-10 ? port_ret / port_std : 0.0

        push!(frontier, (target_return=target, achieved_return=port_ret,
                          risk=port_std, variance=port_var,
                          sharpe=sharpe, weights=copy(w)))
    end

    return frontier
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk Parity
# ─────────────────────────────────────────────────────────────────────────────

"""
    risk_parity(Σ; max_iter=1000, tol=1e-10) → Vector{Float64}

Equal Risk Contribution (ERC) portfolio via Newton's method.
"""
function risk_parity(Σ::Matrix{Float64};
                      max_iter::Int=1000,
                      tol::Float64=1e-10)::Vector{Float64}
    n = size(Σ, 1)
    w = ones(Float64, n) / n
    Σ_reg = Σ + 1e-8 * I(n)

    for iter in 1:max_iter
        σ_p  = sqrt(max(dot(w, Σ_reg * w), 1e-12))
        mrc  = Σ_reg * w ./ σ_p          # marginal risk contribution
        rc   = w .* mrc                  # risk contribution per asset
        rc_target = σ_p / n              # equal target

        # Gradient of ERC objective: Σ (RC_i - RC_target)²
        grad = 2 .* (rc .- rc_target) .* mrc

        # Newton step (approximate Hessian as diagonal)
        diag_H = 2 .* mrc.^2 .+ 1e-6
        step   = grad ./ diag_H

        w_new = w .- 0.1 .* step
        w_new = max.(w_new, 1e-8)
        w_new ./= sum(w_new)

        if norm(w_new - w) < tol
            return w_new
        end
        w = w_new
    end
    return w
end

"""
    risk_contributions(w, Σ) → Vector{Float64}

Return risk contribution of each asset as a fraction of total portfolio risk.
"""
function risk_contributions(w::Vector{Float64}, Σ::Matrix{Float64})::Vector{Float64}
    Σ_reg = Σ + 1e-8 * I(size(Σ,1))
    σ_p   = sqrt(max(dot(w, Σ_reg * w), 1e-12))
    mrc   = Σ_reg * w ./ σ_p
    rc    = w .* mrc
    return rc ./ σ_p
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Hierarchical Risk Parity (HRP)
# ─────────────────────────────────────────────────────────────────────────────

"""
    hierarchical_risk_parity(Σ; linkage=:single) → Vector{Float64}

Lopez de Prado (2016) HRP: cluster → quasi-diagonalise → recursive bisection.
"""
function hierarchical_risk_parity(Σ::Matrix{Float64};
                                   linkage::Symbol=:ward)::Vector{Float64}
    n = size(Σ, 1)

    # 1. Correlation → distance matrix
    corr  = cov2cor(Σ)
    dist  = sqrt.(max.(0.5 .* (1 .- corr), 0.0))

    # 2. Hierarchical clustering
    D_vec = vec([dist[i,j] for j in 1:n for i in 1:n if i < j])
    hc    = hclust(pairwise(Euclidean(), dist'), linkage=linkage)
    order = hc.order   # leaf ordering

    # 3. Quasi-diagonal matrix
    Σ_qd  = Σ[order, order]

    # 4. Recursive bisection
    w_ordered = _hrp_recursive_bisection(Σ_qd, collect(1:n))

    # Re-map to original order
    w = zeros(n)
    for (i, orig_idx) in enumerate(order)
        w[orig_idx] = w_ordered[i]
    end
    return w ./ sum(w)
end

function _cluster_variance(Σ::Matrix{Float64}, idx::Vector{Int})::Float64
    n = length(idx)
    if n == 1
        return Σ[idx[1], idx[1]]
    end
    Σ_sub = Σ[idx, idx]
    # Minimum-variance weight in sub-cluster (equal weight approximation)
    w = ones(n) / n
    return dot(w, Σ_sub * w)
end

function _hrp_recursive_bisection(Σ::Matrix{Float64},
                                    items::Vector{Int})::Vector{Float64}
    n = length(items)
    w = ones(Float64, n)

    if n == 1
        return w
    end

    # Split in half
    mid   = n ÷ 2
    left  = items[1:mid]
    right = items[mid+1:end]

    var_l = _cluster_variance(Σ, left)
    var_r = _cluster_variance(Σ, right)

    # Allocate inversely proportional to variance
    alpha = 1 - var_l / max(var_l + var_r, 1e-12)

    # Recurse
    w_l = _hrp_recursive_bisection(Σ, left)
    w_r = _hrp_recursive_bisection(Σ, right)

    w_full = zeros(n)
    w_full[1:mid]     = alpha .* w_l ./ max(sum(w_l), 1e-10)
    w_full[mid+1:end] = (1-alpha) .* w_r ./ max(sum(w_r), 1e-10)
    return w_full
end

function cov2cor(Σ::Matrix{Float64})::Matrix{Float64}
    d = sqrt.(diag(Σ))
    corr = Σ ./ (d * d')
    clamp!(corr, -1.0, 1.0)
    for i in 1:size(corr,1); corr[i,i] = 1.0; end
    return corr
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Black-Litterman
# ─────────────────────────────────────────────────────────────────────────────

"""
    black_litterman(μ_prior, Σ, P, Q, Ω, tau) → NamedTuple

Black-Litterman posterior.

Args:
  μ_prior — prior expected returns (e.g. CAPM implied)
  Σ       — covariance matrix
  P       — views matrix (k × n), each row is a view portfolio
  Q       — view returns (k-vector)
  Ω       — view uncertainty (k × k diagonal)
  tau     — confidence in prior (typically 0.01–0.05)

Returns posterior_mu, posterior_Sigma, posterior_weights.
"""
function black_litterman(μ_prior::Vector{Float64},
                          Σ::Matrix{Float64},
                          P::Matrix{Float64},
                          Q::Vector{Float64},
                          Ω::Matrix{Float64},
                          tau::Float64=0.025)::NamedTuple

    n = length(μ_prior)
    k = length(Q)
    @assert size(P) == (k, n)
    @assert size(Ω) == (k, k)

    τΣ = tau .* Σ
    τΣ_reg = τΣ + 1e-8 * I(n)

    # BL master formula
    M = inv(inv(τΣ_reg) + P' * inv(Ω) * P)
    posterior_mu  = M * (inv(τΣ_reg) * μ_prior + P' * inv(Ω) * Q)
    posterior_Sigma = Σ + M   # uncertainty in posterior mean

    # Optimal weights: max Sharpe with posterior
    Σ_reg  = posterior_Sigma + 1e-8 * I(n)
    w_opt  = inv(Σ_reg) * posterior_mu
    w_norm = w_opt ./ max(sum(abs.(w_opt)), 1e-10)

    return (posterior_mu    = posterior_mu,
            posterior_Sigma = posterior_Sigma,
            posterior_weights = w_norm,
            prior_mu        = μ_prior,
            views_P         = P,
            views_Q         = Q)
end

"""
    capm_implied_returns(w_market, Σ, rf, risk_aversion) → Vector{Float64}

Compute CAPM-implied equilibrium returns: μ = rf + δ Σ w_market.
"""
function capm_implied_returns(w_market::Vector{Float64}, Σ::Matrix{Float64},
                                rf::Float64=0.0,
                                risk_aversion::Float64=2.5)::Vector{Float64}
    return rf .+ risk_aversion .* (Σ * w_market)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Simple Weighting Schemes
# ─────────────────────────────────────────────────────────────────────────────

"""
    equal_weight(n) → Vector{Float64}
"""
equal_weight(n::Int)::Vector{Float64} = ones(n) / n

"""
    inverse_vol_weight(Σ) → Vector{Float64}

1/σᵢ weighting, normalised.
"""
function inverse_vol_weight(Σ::Matrix{Float64})::Vector{Float64}
    vols = sqrt.(diag(Σ))
    w    = 1.0 ./ max.(vols, 1e-8)
    return w ./ sum(w)
end

"""
    max_diversification(Σ) → Vector{Float64}

Maximise the Diversification Ratio: (w'σ) / √(w'Σw).
"""
function max_diversification(Σ::Matrix{Float64})::Vector{Float64}
    n   = size(Σ, 1)
    vols = sqrt.(max.(diag(Σ), 1e-12))
    Σ_reg = Σ + 1e-8 * I(n)

    function neg_dr(w)
        w_pos = max.(w, 1e-8)
        w_pos ./= sum(w_pos)
        denom = sqrt(max(dot(w_pos, Σ_reg * w_pos), 1e-12))
        numer = dot(w_pos, vols)
        return -(numer / denom)
    end

    w0 = inverse_vol_weight(Σ)
    result = optimize(neg_dr, zeros(n), ones(n), w0,
                      Fminbox(LBFGS()),
                      Optim.Options(iterations=2000))
    w = max.(Optim.minimizer(result), 0.0)
    return w ./ sum(w)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Grid Search
# ─────────────────────────────────────────────────────────────────────────────

"""
    grid_search(f, param_grid; n_jobs=4) → DataFrame

Parallel grid search. f receives a Dict of params and returns a scalar (the objective).
param_grid: Dict{String, Vector} of parameter names → values.

Returns a DataFrame with all param combinations + "objective" column,
sorted descending.
"""
function grid_search(f::Function, param_grid::Dict{String, <:AbstractVector};
                      n_jobs::Int=4)::DataFrame
    param_names = collect(keys(param_grid))
    param_vals  = [param_grid[k] for k in param_names]

    # Build all combos
    combos = _cartesian_product(param_vals)
    n_combos = length(combos)

    objectives = Vector{Float64}(undef, n_combos)

    # Thread-parallel evaluation
    Threads.@threads for i in 1:n_combos
        combo = combos[i]
        params = Dict(param_names[j] => combo[j] for j in 1:length(param_names))
        try
            objectives[i] = f(params)
        catch
            objectives[i] = -Inf
        end
    end

    # Build DataFrame
    rows = Vector{Dict{String, Any}}(undef, n_combos)
    for i in 1:n_combos
        d = Dict{String, Any}(param_names[j] => combos[i][j] for j in 1:length(param_names))
        d["objective"] = objectives[i]
        rows[i] = d
    end

    df = DataFrame(rows)
    sort!(df, :objective, rev=true)
    return df
end

function _cartesian_product(vecs::Vector)
    result = [Any[]]
    for v in vecs
        result = [vcat(r, [x]) for r in result for x in v]
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Bayesian Optimization (GP Surrogate)
# ─────────────────────────────────────────────────────────────────────────────

"""
    bayesian_optimize(f, param_bounds, n_iter=100; n_initial=10) → NamedTuple

Gaussian Process-based Bayesian optimization via Expected Improvement.

param_bounds: Dict{String, Tuple{Float64, Float64}} of (lo, hi) bounds.
Returns best_params, best_value, history.
"""
function bayesian_optimize(f::Function,
                             param_bounds::Dict{String, Tuple{Float64, Float64}},
                             n_iter::Int=100;
                             n_initial::Int=10,
                             rng::AbstractRNG=Random.default_rng())::NamedTuple

    param_names = collect(keys(param_bounds))
    n_params    = length(param_names)
    lo  = [param_bounds[k][1] for k in param_names]
    hi  = [param_bounds[k][2] for k in param_names]

    # Initial random samples (Latin Hypercube)
    X_obs = Matrix{Float64}(undef, n_initial, n_params)
    for j in 1:n_params
        for i in 1:n_initial
            X_obs[i, j] = lo[j] + (rand(rng) + (i-1)) / n_initial * (hi[j] - lo[j])
        end
        X_obs[:, j] = X_obs[shuffle(rng, 1:n_initial), j]
    end

    y_obs = Vector{Float64}(undef, n_initial)
    for i in 1:n_initial
        params = Dict(param_names[j] => X_obs[i, j] for j in 1:n_params)
        try
            y_obs[i] = f(params)
        catch
            y_obs[i] = -Inf
        end
    end

    history = DataFrame(iteration=Int[], objective=Float64[])
    for i in 1:n_initial
        push!(history, (iteration=i, objective=y_obs[i]))
    end

    # GP hyperparameters: RBF kernel, length scales
    for iter in (n_initial+1):(n_initial+n_iter)
        # Fit GP (simplified: squared-exponential kernel, noise σ²=0.01)
        ls    = _estimate_length_scales(X_obs, y_obs, lo, hi)
        K     = _se_kernel_matrix(X_obs, X_obs, ls) + 1e-4 * I(size(X_obs, 1))
        K_chol = cholesky(K + 1e-6 * I(size(K,1)))
        alpha  = K_chol \ y_obs

        # Acquisition: Expected Improvement
        best_so_far = maximum(filter(isfinite, y_obs))

        function acq_neg(x_new)
            k_star  = _se_kernel_vec(X_obs, x_new, ls)
            mu_star = dot(k_star, alpha)
            v       = K_chol.L \ k_star
            sigma_star = sqrt(max(1.0 - dot(v, v), 1e-8))
            z = (mu_star - best_so_far) / sigma_star
            ei = (mu_star - best_so_far) * _norm_cdf(z) + sigma_star * _norm_pdf(z)
            return -ei
        end

        # Random restarts for acquisition maximisation
        best_acq_x   = X_obs[1, :]
        best_acq_val = Inf
        for _ in 1:20
            x_init = lo .+ rand(rng, n_params) .* (hi .- lo)
            res = optimize(acq_neg, lo, hi, x_init, Fminbox(LBFGS()),
                           Optim.Options(iterations=200, g_tol=1e-6))
            if Optim.minimum(res) < best_acq_val
                best_acq_val = Optim.minimum(res)
                best_acq_x   = Optim.minimizer(res)
            end
        end

        # Evaluate
        params = Dict(param_names[j] => best_acq_x[j] for j in 1:n_params)
        y_new  = try f(params) catch; -Inf; end

        X_obs = vcat(X_obs, best_acq_x')
        push!(y_obs, y_new)
        push!(history, (iteration=iter, objective=y_new))
    end

    best_idx    = argmax(y_obs)
    best_params = Dict(param_names[j] => X_obs[best_idx, j] for j in 1:n_params)

    return (best_params=best_params, best_value=y_obs[best_idx],
            history=history, n_evaluations=length(y_obs))
end

function _se_kernel_matrix(X1::Matrix{Float64}, X2::Matrix{Float64},
                             ls::Vector{Float64})::Matrix{Float64}
    n1 = size(X1, 1); n2 = size(X2, 1)
    K  = Matrix{Float64}(undef, n1, n2)
    for i in 1:n1, j in 1:n2
        d2 = sum(((X1[i, k] - X2[j, k]) / max(ls[k], 1e-8))^2 for k in 1:length(ls))
        K[i, j] = exp(-0.5 * d2)
    end
    return K
end

function _se_kernel_vec(X::Matrix{Float64}, x::Vector{Float64},
                         ls::Vector{Float64})::Vector{Float64}
    n = size(X, 1)
    k = Vector{Float64}(undef, n)
    for i in 1:n
        d2 = sum(((X[i, j] - x[j]) / max(ls[j], 1e-8))^2 for j in 1:length(ls))
        k[i] = exp(-0.5 * d2)
    end
    return k
end

function _estimate_length_scales(X::Matrix{Float64}, y::Vector{Float64},
                                   lo::Vector{Float64}, hi::Vector{Float64})::Vector{Float64}
    # Use range as default length scale
    return max.(hi .- lo, 1e-4) .* 0.5
end

_norm_cdf(z::Float64)::Float64 = 0.5 * (1 + erf(z / sqrt(2)))
_norm_pdf(z::Float64)::Float64 = exp(-0.5 * z^2) / sqrt(2π)

function erf(x::Float64)::Float64
    # Abramowitz & Stegun approximation
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                t * (-1.453152027 + t * 1.061405429))))
    return sign(x) * (1 - poly * exp(-x^2))
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Walk-Forward Optimization
# ─────────────────────────────────────────────────────────────────────────────

"""
    walk_forward_optimize(data, strategy_fn, param_grid, train_size, test_size, step) → DataFrame

Walk-forward optimization:
  1. For each window: run grid_search on train slice
  2. Apply best params to test slice
  3. Collect OOS performance metrics

strategy_fn(data_slice, params) → returns::Vector{Float64}

Returns DataFrame with columns:
  window, train_start, train_end, test_start, test_end,
  best_params, is_sharpe, oos_sharpe, oos_n_trades
"""
function walk_forward_optimize(
    data::DataFrame,
    strategy_fn::Function,
    param_grid::Dict{String, <:AbstractVector},
    train_size::Int,
    test_size::Int,
    step::Int;
    metric::String="sharpe"
)::DataFrame

    n = nrow(data)
    results = NamedTuple[]
    window  = 1
    i       = 1

    while i + train_size + test_size <= n
        train_end  = i + train_size - 1
        test_start = train_end + 1
        test_end   = min(test_start + test_size - 1, n)

        train_data = data[i:train_end, :]
        test_data  = data[test_start:test_end, :]

        # In-sample grid search
        is_df = grid_search(
            params -> begin
                rets = strategy_fn(train_data, params)
                isempty(rets) && return -Inf
                _metric_value(rets, metric)
            end,
            param_grid
        )

        # Best params
        best_row    = first(is_df)
        best_params = Dict(k => best_row[k] for k in keys(param_grid))
        is_sharpe   = best_row[:objective]

        # OOS test
        oos_rets = try
            strategy_fn(test_data, best_params)
        catch
            Float64[]
        end

        oos_sharpe = isempty(oos_rets) ? NaN :
                     _metric_value(oos_rets, "sharpe")

        push!(results, (
            window      = window,
            train_start = i,
            train_end   = train_end,
            test_start  = test_start,
            test_end    = test_end,
            best_params = best_params,
            is_sharpe   = is_sharpe,
            oos_sharpe  = oos_sharpe,
            oos_n       = length(oos_rets),
        ))

        i      += step
        window += 1
    end

    return isempty(results) ? DataFrame() : DataFrame(results)
end

function _metric_value(rets::Vector{Float64}, metric::String)::Float64
    isempty(rets) && return -Inf
    if metric == "sharpe"
        n = length(rets); n < 2 && return 0.0
        m = mean(rets); s = std(rets)
        s < 1e-10 && return 0.0
        return m / s * sqrt(252)
    elseif metric == "total_return"
        return sum(rets)
    elseif metric == "calmar"
        eq = cumprod(1 .+ rets)
        dd_val = first(max_drawdown_val(eq))
        dd_val < 1e-10 && return 1000.0
        c = (eq[end])^(252 / length(rets)) - 1
        return c / dd_val
    else
        return mean(rets)
    end
end

function max_drawdown_val(eq::Vector{Float64})::Tuple{Float64, Int, Int}
    n = length(eq); peak = eq[1]; max_dd = 0.0; pk = 1; tr = 1; cpk = 1
    for i in 2:n
        if eq[i] > peak; peak = eq[i]; cpk = i; end
        dd = (peak - eq[i]) / max(peak, 1e-12)
        if dd > max_dd; max_dd = dd; pk = cpk; tr = i; end
    end
    return (max_dd, pk, tr)
end


# ─────────────────────────────────────────────────────────────────────────────
# 9. Kelly Criterion Sizing
# ─────────────────────────────────────────────────────────────────────────────

"""
    kelly_fraction(mu, sigma2; f_max=0.25) → Float64

Kelly optimal fraction: f* = μ / σ² (continuous-time Kelly).
Capped at f_max to avoid excessive leverage.
"""
function kelly_fraction(mu::Float64, sigma2::Float64; f_max::Float64=0.25)::Float64
    sigma2 < 1e-10 && return 0.0
    f = mu / sigma2
    return clamp(f, 0.0, f_max)
end

"""
    fractional_kelly(mu, sigma2, fraction; f_max=0.25) → Float64

Fractional Kelly sizing (e.g., half-Kelly: fraction=0.5).
"""
function fractional_kelly(mu::Float64, sigma2::Float64, fraction::Float64=0.5;
                            f_max::Float64=0.25)::Float64
    return kelly_fraction(mu, sigma2; f_max=f_max) * fraction
end

"""
    kelly_portfolio(mu, Sigma; fraction=0.5) → Vector{Float64}

Multi-asset Kelly allocation: w* = Σ⁻¹ μ, long-only projected, normalised.
"""
function kelly_portfolio(mu::Vector{Float64}, Sigma::Matrix{Float64};
                          fraction::Float64=0.5)::Vector{Float64}
    n = length(mu)
    Sigma_reg = Sigma + 1e-8 * I(n)
    w = inv(Sigma_reg) * mu
    w = max.(w, 0.0)
    s = sum(w)
    s < 1e-10 && return ones(n) / n
    return w ./ s .* fraction
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. CVaR (Expected Shortfall) Optimisation
# ─────────────────────────────────────────────────────────────────────────────

"""
    min_cvar(returns_matrix, alpha; max_iter=500) → Vector{Float64}

Minimum Conditional Value-at-Risk portfolio via sub-gradient descent.
"""
function min_cvar(returns_matrix::Matrix{Float64}, alpha::Float64=0.05;
                   max_iter::Int=500)::Vector{Float64}

    T, N = size(returns_matrix)
    @assert 0 < alpha < 1

    tail_n = max(1, round(Int, alpha * T))
    w = ones(N) / N

    for iter in 1:max_iter
        port_rets   = returns_matrix * w
        sorted_idx  = sortperm(port_rets)
        tail_idx    = sorted_idx[1:tail_n]

        grad        = vec(mean(returns_matrix[tail_idx, :], dims=1))
        step_size   = 0.5 / (iter^0.6)

        w_new       = w .- step_size .* grad   # minimise CVaR
        w_new       = max.(w_new, 0.0)
        s           = sum(w_new)
        w           = s > 1e-10 ? w_new ./ s : ones(N) / N
    end
    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# 11. Transaction Cost-Aware Rebalancing
# ─────────────────────────────────────────────────────────────────────────────

"""
    rebalance_cost(w_current, w_target, transaction_cost) → Float64

Total one-way transaction cost to rebalance.
"""
function rebalance_cost(w_current::Vector{Float64}, w_target::Vector{Float64},
                         transaction_cost::Float64=0.001)::Float64
    return sum(abs.(w_target - w_current)) / 2 * transaction_cost
end

"""
    no_trade_zone(w_target, tol) → Tuple{Vector, Vector}

Return (lo, hi) bounds around each weight; don't trade if w stays inside.
"""
function no_trade_zone(w_target::Vector{Float64}, tol::Float64=0.05)
    return w_target .- tol, w_target .+ tol
end

# ─────────────────────────────────────────────────────────────────────────────
# 12. Robust Optimisation
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_min_variance(mu, Sigma, uncertainty; gamma=1.0) → Vector{Float64}

Robust Markowitz: penalise for estimation uncertainty in μ.
"""
function robust_min_variance(mu::Vector{Float64}, Sigma::Matrix{Float64},
                               uncertainty::Float64=0.01;
                               gamma::Float64=1.0)::Vector{Float64}
    n = length(mu)
    Sigma_reg = Sigma + 1e-8 * I(n)
    kap       = uncertainty
    mu_robust = mu .* max.(1 .- kap ./ max.(abs.(mu), 1e-10), 0.0)

    function neg_utility(w)
        w_pos = max.(w, 0.0); s = sum(w_pos)
        s < 1e-10 && return 1e6
        w_n   = w_pos ./ s
        vari  = dot(w_n, Sigma_reg * w_n)
        ret   = dot(w_n, mu_robust)
        pen   = kap * norm(w_n)
        return vari - gamma * (ret - pen)
    end

    w0     = ones(n) / n
    result = optimize(neg_utility, zeros(n), ones(n), w0,
                       Fminbox(LBFGS()), Optim.Options(iterations=2000))
    w      = max.(Optim.minimizer(result), 0.0)
    s      = sum(w); s < 1e-10 && return ones(n) / n
    return w ./ s
end

# ─────────────────────────────────────────────────────────────────────────────
# 13. Momentum & Mean-Reversion Weights
# ─────────────────────────────────────────────────────────────────────────────

"""
    momentum_weight(returns_matrix, lookback; top_n=nothing) → Vector{Float64}

Long-only momentum: weight by positive cumulative return over lookback.
"""
function momentum_weight(returns_matrix::Matrix{Float64}, lookback::Int;
                          top_n::Union{Nothing, Int}=nothing)::Vector{Float64}
    T, N = size(returns_matrix)
    lb   = min(lookback, T)
    mom  = [sum(returns_matrix[end-lb+1:end, j]) for j in 1:N]

    if !isnothing(top_n)
        sorted = sortperm(mom, rev=true)
        w = zeros(N)
        for i in 1:min(top_n, N)
            w[sorted[i]] = max(mom[sorted[i]], 0.0)
        end
    else
        w = max.(mom, 0.0)
    end
    s = sum(w); s < 1e-10 && return ones(N) / N
    return w ./ s
end

"""
    mean_reversion_weight(returns_matrix, lookback) → Vector{Float64}

Contrarian: weight inversely to recent returns (long losers).
"""
function mean_reversion_weight(returns_matrix::Matrix{Float64},
                                 lookback::Int)::Vector{Float64}
    T, N = size(returns_matrix)
    lb   = min(lookback, T)
    mom  = [sum(returns_matrix[end-lb+1:end, j]) for j in 1:N]
    w    = max.(-mom, 0.0)
    s    = sum(w); s < 1e-10 && return ones(N) / N
    return w ./ s
end

# ─────────────────────────────────────────────────────────────────────────────
# 14. Cluster-Based Portfolio
# ─────────────────────────────────────────────────────────────────────────────

"""
    cluster_portfolio(Sigma, n_clusters; linkage=:ward) → Vector{Float64}

Cluster assets by correlation → equal-weight within cluster →
inverse-variance across clusters.
"""
function cluster_portfolio(Sigma::Matrix{Float64}, n_clusters::Int=3;
                             linkage::Symbol=:ward)::Vector{Float64}
    n    = size(Sigma, 1)
    corr = cov2cor(Sigma)
    dist = sqrt.(max.(0.5 .* (1 .- corr), 0.0))

    hc     = hclust(pairwise(Euclidean(), dist'), linkage=linkage)
    labels = cutree(hc; k=n_clusters)

    cluster_vars  = zeros(n_clusters)
    cluster_sizes = zeros(Int, n_clusters)

    for k in 1:n_clusters
        members = findall(labels .== k)
        isempty(members) && continue
        cluster_sizes[k] = length(members)
        nc = length(members)
        wc = ones(nc) / nc
        cluster_vars[k] = dot(wc, Sigma[members, members] * wc)
    end

    cluster_weights = 1.0 ./ max.(cluster_vars, 1e-12)
    cluster_weights ./= sum(cluster_weights)

    w = zeros(n)
    for k in 1:n_clusters
        members = findall(labels .== k)
        isempty(members) && continue
        for m in members
            w[m] = cluster_weights[k] / length(members)
        end
    end
    s = sum(w); s > 1e-10 && (w ./= s)
    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# 15. Portfolio Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

"""
    portfolio_summary(w, mu, Sigma; rf=0.0, annualize=252) → NamedTuple

Comprehensive portfolio analytics.
"""
function portfolio_summary(w::Vector{Float64}, mu::Vector{Float64},
                             Sigma::Matrix{Float64};
                             rf::Float64=0.0,
                             annualize::Int=252)::NamedTuple

    n        = length(w)
    Sigma_r  = Sigma + 1e-8 * I(n)
    vols     = sqrt.(diag(Sigma_r))

    port_ret  = dot(w, mu) * annualize
    port_var  = dot(w, Sigma_r * w) * annualize
    port_vol  = sqrt(max(port_var, 0.0))
    sharpe    = port_vol > 1e-10 ? (port_ret - rf) / port_vol : 0.0

    mrc  = Sigma_r * w
    sigp = sqrt(max(dot(w, mrc), 1e-12))
    rc   = w .* mrc ./ sigp
    rcp  = rc ./ sigp

    hhi    = sum(w.^2)
    eff_n  = 1.0 / hhi
    dr     = dot(w, vols .* sqrt(annualize)) / max(port_vol, 1e-10)

    return (
        weights             = w,
        port_return         = port_ret,
        port_vol            = port_vol,
        sharpe              = sharpe,
        risk_contribs       = rcp,
        max_rc              = maximum(rcp),
        hhi                 = hhi,
        effective_n         = eff_n,
        diversification_ratio = dr,
    )
end

"""
    attribution(w, factor_returns, factor_betas) → NamedTuple

Return decomposition: systematic (factor) + idiosyncratic.
"""
function attribution(w::Vector{Float64},
                      factor_returns::Matrix{Float64},
                      factor_betas::Matrix{Float64})::NamedTuple
    # factor_returns: T × K
    # factor_betas: N × K
    port_beta   = vec(w' * factor_betas)            # K-vector
    factor_pnl  = factor_returns * port_beta        # T-vector
    total_ret   = sum(w)                             # simplified scalar

    return (
        port_factor_betas = port_beta,
        factor_contribution = factor_pnl,
        systematic_var      = var(factor_pnl),
    )
end

export kelly_fraction, fractional_kelly, kelly_portfolio
export min_cvar, rebalance_cost, no_trade_zone
export robust_min_variance
export momentum_weight, mean_reversion_weight
export cluster_portfolio, portfolio_summary, attribution

end # module SRFMOptimization
