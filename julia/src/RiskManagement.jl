"""
RiskManagement — Comprehensive risk management for crypto quantitative strategies.

Covers:
  - CVaR optimisation (linear programming formulation)
  - Maximum drawdown constraint in portfolio optimisation
  - Kelly criterion variants: full Kelly, half Kelly, fractional Kelly
  - Risk parity with CVaR instead of variance
  - Tail risk hedging: when to hold cash vs trade
  - Regime-conditional VaR: different limits in stress vs normal
  - Correlation stress testing: what if crypto correlation spikes to 0.95?
  - Dynamic stop-loss calibration via EVT
"""
module RiskManagement

using LinearAlgebra
using Statistics
using Random

export CVaROptimizer, optimize_cvar_portfolio, cvar_weights
export MaxDrawdownPortfolio, mdp_weights
export KellyFraction, full_kelly, half_kelly, fractional_kelly, kelly_growth_optimal
export CVaRRiskParity, cvar_risk_parity_weights
export TailRiskHedge, compute_hedge_ratio, cash_vs_trade_rule
export regime_conditional_var, var_stress_by_regime
export CorrelationStressTest, stress_correlation, stressed_portfolio_var
export DynamicStopLoss, calibrate_stop_loss, gpd_stop_loss
export PortfolioRiskReport, generate_risk_report
export var_historical, cvar_historical, es_historical

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: CVaR Optimisation
# ─────────────────────────────────────────────────────────────────────────────

"""
    CVaROptimizer

Conditional Value-at-Risk portfolio optimiser.
Implements the Rockafellar-Uryasev (2000) LP formulation:

Minimise CVaR_α(w) = z + (1/((1-α)*T)) * sum max(-r_t'w - z, 0)
subject to: w ≥ 0, sum(w) = 1

This is a linear program in (w, z, u) where u_t = max(-r_t'w - z, 0).
"""
struct CVaROptimizer
    alpha::Float64      # confidence level (e.g. 0.95)
    target_return::Float64
    max_weight::Float64
    long_only::Bool
end

"""
    CVaROptimizer(; alpha, target_return, max_weight, long_only) -> CVaROptimizer
"""
function CVaROptimizer(; alpha::Float64=0.95, target_return::Float64=0.0,
                         max_weight::Float64=0.5, long_only::Bool=true)::CVaROptimizer
    return CVaROptimizer(alpha, target_return, max_weight, long_only)
end

"""
    optimize_cvar_portfolio(R, opt; n_iter, tol) -> NamedTuple

Minimise portfolio CVaR using the coordinate descent approach.
R: T × d matrix of scenario returns.
Returns optimal weights and portfolio risk metrics.
"""
function optimize_cvar_portfolio(R::Matrix{Float64}, opt::CVaROptimizer;
                                   n_iter::Int=500, tol::Float64=1e-6)::NamedTuple
    T, d = size(R)
    alpha = opt.alpha

    # Initialise with equal weights
    w = fill(1/d, d)

    best_cvar = Inf
    best_w    = copy(w)

    # Projected gradient descent with CVaR objective
    for iter in 1:n_iter
        port_ret = R * w
        cvar_val, z_star = _cvar_and_threshold(port_ret, alpha)

        if cvar_val < best_cvar
            best_cvar = cvar_val
            best_w    = copy(w)
        end

        # Gradient of CVaR w.r.t. w
        # ∂CVaR/∂w = -(1/((1-α)T)) * sum_{t: r_t'w < -z*} r_t
        tail_mask = port_ret .< -z_star
        n_tail    = max(sum(tail_mask), 1)
        grad      = -(1 / ((1 - alpha) * T)) .* (R[tail_mask, :]' * ones(n_tail))

        # Update with projected gradient
        step = 0.1 / sqrt(iter)
        w_new = w .- step .* grad

        # Project onto simplex (long-only, sum=1, max weight constraint)
        w_new = opt.long_only ? project_simplex(w_new, opt.max_weight) :
                                w_new ./ max(sum(abs.(w_new)), 1e-10)

        # Check convergence
        norm(w_new - w) < tol && break
        w = w_new
    end

    port_rets = R * best_w
    cvar_opt, var_opt = _cvar_and_threshold(port_rets, alpha)

    return (weights=best_w, cvar=cvar_opt, var=var_opt,
            expected_return=mean(port_rets),
            sharpe=mean(port_rets)/max(std(port_rets), 1e-10)*sqrt(252))
end

"""
    project_simplex(w, max_w) -> Vector{Float64}

Project weight vector onto the simplex {w : sum=1, 0≤w_i≤max_w}.
Uses the Duchi et al. (2008) algorithm adapted for box constraints.
"""
function project_simplex(w::Vector{Float64}, max_w::Float64=1.0)::Vector{Float64}
    d = length(w)
    # Clip to [0, max_w]
    w_clip = clamp.(w, 0.0, max_w)
    s = sum(w_clip)

    # If already sums to 1, done
    abs(s - 1.0) < 1e-8 && return w_clip

    # Adjust: project to sum=1 while keeping box constraints
    # Use iterative scaling
    w_proj = copy(w_clip)
    for _ in 1:50
        excess = sum(w_proj) - 1.0
        abs(excess) < 1e-8 && break
        # Reduce by distributing excess uniformly
        w_proj .-= excess / d
        w_proj   = clamp.(w_proj, 0.0, max_w)
    end

    # Final normalisation
    s2 = sum(w_proj)
    s2 < 1e-10 && return fill(1/d, d)
    return w_proj ./ s2
end

"""
    _cvar_and_threshold(port_rets, alpha) -> Tuple{Float64, Float64}

Compute CVaR and VaR threshold from portfolio return vector.
"""
function _cvar_and_threshold(port_rets::Vector{Float64}, alpha::Float64)::Tuple{Float64,Float64}
    sorted = sort(port_rets)
    n = length(sorted)
    var_idx = clamp(round(Int, (1-alpha) * n), 1, n)
    z_star  = -sorted[var_idx]  # VaR (positive = loss)

    tail_rets = sorted[1:var_idx]
    cvar = -mean(tail_rets)  # CVaR (positive = loss)

    return cvar, z_star
end

"""
    cvar_weights(R; alpha, target_return, max_weight) -> Vector{Float64}

Convenience function: return CVaR-optimal weights.
"""
function cvar_weights(R::Matrix{Float64}; alpha::Float64=0.95,
                       target_return::Float64=0.0,
                       max_weight::Float64=0.5)::Vector{Float64}
    opt = CVaROptimizer(alpha=alpha, target_return=target_return, max_weight=max_weight)
    result = optimize_cvar_portfolio(R, opt)
    return result.weights
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: VaR and CVaR Measures
# ─────────────────────────────────────────────────────────────────────────────

"""
    var_historical(returns; alpha) -> Float64

Historical Value-at-Risk at confidence level alpha.
Returns the loss (positive) at the alpha quantile.
"""
function var_historical(returns::Vector{Float64}; alpha::Float64=0.99)::Float64
    -quantile_v(sort(returns), 1 - alpha)
end

"""
    cvar_historical(returns; alpha) -> Float64

Historical Conditional VaR (Expected Shortfall) at confidence level alpha.
"""
function cvar_historical(returns::Vector{Float64}; alpha::Float64=0.99)::Float64
    n = length(returns)
    sorted = sort(returns)
    cutoff = round(Int, (1-alpha) * n)
    cutoff = clamp(cutoff, 1, n)
    -mean(sorted[1:cutoff])
end

"""
    es_historical(returns; alpha) -> Float64

Alias for cvar_historical (Expected Shortfall = CVaR).
"""
es_historical(returns; alpha=0.99) = cvar_historical(returns; alpha=alpha)

function quantile_v(sorted::Vector{Float64}, p::Float64)::Float64
    n = length(sorted)
    idx = clamp(p * n, 1.0, Float64(n))
    lo = floor(Int, idx)
    hi = min(ceil(Int, idx), n)
    lo == hi && return sorted[lo]
    return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo])
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Maximum Drawdown Constraint
# ─────────────────────────────────────────────────────────────────────────────

"""
    MaxDrawdownPortfolio

Portfolio optimiser with maximum drawdown constraint.
Implements a Monte Carlo approach: simulate paths and select weights
that achieve target return with drawdown ≤ max_dd.
"""
struct MaxDrawdownPortfolio
    max_dd::Float64      # maximum allowed drawdown (e.g. 0.20 = 20%)
    target_sharpe::Float64
    n_sim::Int
end

"""
    mdp_weights(R, mdp; rng) -> NamedTuple

Find weights that maximise Sharpe subject to max drawdown constraint.
Uses a random search over the weight simplex.
"""
function mdp_weights(R::Matrix{Float64}, mdp::MaxDrawdownPortfolio;
                      rng::AbstractRNG=Random.default_rng())::NamedTuple
    T, d = size(R)

    best_sharpe = -Inf
    best_weights = fill(1/d, d)
    best_mdd = 1.0

    for _ in 1:mdp.n_sim
        # Random Dirichlet-like weights
        raw = -log.(rand(rng, d))
        w   = raw ./ sum(raw)

        port_ret = R * w
        eq = cumprod(1 .+ port_ret)

        # Max drawdown
        mdd = _max_drawdown_equity(eq)
        mdd > mdp.max_dd && continue

        sr = mean(port_ret) / max(std(port_ret), 1e-10) * sqrt(252)
        if sr > best_sharpe
            best_sharpe  = sr
            best_weights = copy(w)
            best_mdd     = mdd
        end
    end

    port_rets = R * best_weights
    return (weights=best_weights, sharpe=best_sharpe, max_dd=best_mdd,
            expected_return=mean(port_rets))
end

function _max_drawdown_equity(equity::Vector{Float64})::Float64
    mdd  = 0.0
    peak = equity[1]
    for e in equity
        e > peak && (peak = e)
        dd = (peak - e) / peak
        dd > mdd && (mdd = dd)
    end
    return mdd
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Kelly Criterion Variants
# ─────────────────────────────────────────────────────────────────────────────

"""
    KellyFraction

Stores Kelly fraction parameters and computed fractions.
"""
struct KellyFraction
    full_kelly::Float64
    half_kelly::Float64
    quarter_kelly::Float64
    fractional::Float64   # arbitrary fraction
    mu::Float64
    sigma::Float64
    win_prob::Float64
    win_loss_ratio::Float64
end

"""
    full_kelly(mu, sigma; r) -> Float64

Continuous Kelly fraction: f* = (μ - r) / σ²
(Derived from maximising E[log(wealth)] in GBM framework.)
"""
function full_kelly(mu::Float64, sigma::Float64; r::Float64=0.0)::Float64
    sigma < 1e-10 && return 0.0
    return (mu - r) / sigma^2
end

"""
    half_kelly(mu, sigma; r) -> Float64

Half-Kelly: f = f_Kelly / 2.
Reduces drawdowns significantly at the cost of ~25% lower CAGR.
"""
half_kelly(mu, sigma; r=0.0) = full_kelly(mu, sigma; r=r) / 2

"""
    fractional_kelly(mu, sigma, kappa; r) -> Float64

Fractional Kelly with fraction κ ∈ [0, 1]: f = κ * f_Kelly.
"""
fractional_kelly(mu, sigma, kappa; r=0.0) = kappa * full_kelly(mu, sigma; r=r)

"""
    kelly_discrete(p, b) -> Float64

Discrete Kelly fraction for a binary bet.
p: probability of winning, b: net payoff per unit staked (win amount).
f* = p - (1-p)/b
"""
kelly_discrete(p::Float64, b::Float64)::Float64 = max(p - (1-p)/b, 0.0)

"""
    kelly_growth_optimal(mu, sigma; r, n_fracs) -> NamedTuple

Find the growth-optimal Kelly fraction via expected log-return simulation.
Computes E[log(1 + f*(r_t - r))] for a range of f values.
"""
function kelly_growth_optimal(mu::Float64, sigma::Float64;
                                r::Float64=0.0,
                                n_fracs::Int=100,
                                seed::Int=42)::NamedTuple
    rng = MersenneTwister(seed)
    n_sim = 5000

    rets = mu .+ sigma .* randn(rng, n_sim)  # simulated excess returns

    f_grid  = range(0.0, 3.0, length=n_fracs)
    growth  = zeros(n_fracs)

    for (i, f) in enumerate(f_grid)
        log_growths = log.(max.(1 .+ f .* (rets .- r), 1e-300))
        growth[i]   = mean(log_growths)
    end

    best_idx  = argmax(growth)
    f_optimal = f_grid[best_idx]

    return (f_optimal=f_optimal, growth_curve=(f_grid=collect(f_grid), growth=growth),
            f_theoretical=full_kelly(mu, sigma; r=r))
end

"""
    kelly_variants_summary(mu, sigma; r) -> KellyFraction

Compute all Kelly variants for given drift/vol parameters.
"""
function kelly_variants_summary(mu::Float64, sigma::Float64;
                                  r::Float64=0.0)::KellyFraction
    fk  = full_kelly(mu, sigma; r=r)
    hk  = half_kelly(mu, sigma; r=r)
    p_win = 0.5 + mu / (2 * sigma * sqrt(2/π))  # rough estimate
    win_loss = (mu + sigma) / max(abs(mu - sigma), 1e-10)
    return KellyFraction(fk, hk, fk/4, fk/2, mu, sigma, clamp(p_win, 0.1, 0.9), win_loss)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: CVaR Risk Parity
# ─────────────────────────────────────────────────────────────────────────────

"""
    CVaRRiskParity

Risk parity using CVaR as the risk measure instead of variance.
Goal: each asset contributes equally to total portfolio CVaR.
"""
struct CVaRRiskParity
    alpha::Float64
    n_iter::Int
    tol::Float64
end

"""
    cvar_risk_parity_weights(R; alpha, n_iter, tol) -> Vector{Float64}

Compute CVaR risk parity weights.
Uses iterative proportional scaling (Maillard et al. 2010 adapted to CVaR).
"""
function cvar_risk_parity_weights(R::Matrix{Float64};
                                    alpha::Float64=0.95,
                                    n_iter::Int=300,
                                    tol::Float64=1e-6)::Vector{Float64}
    T, d = size(R)
    w = fill(1/d, d)

    for iter in 1:n_iter
        port_ret = R * w
        cvar_total, z_star = _cvar_and_threshold(port_ret, alpha)
        cvar_total < 1e-12 && break

        # Marginal CVaR: ∂CVaR/∂w_i = -E[R_i | R_portfolio < -VaR]
        tail_mask = port_ret .< -z_star
        n_tail    = max(sum(tail_mask), 1)
        marginal_cvar = -mean(R[tail_mask, :]; dims=1)[:]

        # CVaR contribution of each asset
        contrib = w .* marginal_cvar
        contrib_total = sum(contrib)

        # Target: equal contribution → w_i ∝ 1/marginal_cvar_i
        target_contrib = fill(cvar_total / d, d)
        w_new = w .* target_contrib ./ max.(contrib, 1e-10)
        w_new = clamp.(w_new, 1e-4, 1.0)
        w_new ./= sum(w_new)

        norm(w_new - w) < tol && return w_new
        w = w_new
    end

    return w
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Tail Risk Hedging
# ─────────────────────────────────────────────────────────────────────────────

"""
    TailRiskHedge

Parameters for a tail risk hedging decision rule.
Determines when to reduce exposure to cash based on tail risk signals.
"""
struct TailRiskHedge
    var_threshold::Float64   # reduce to cash when rolling VaR > threshold
    cvar_threshold::Float64  # reduce to cash when CVaR > threshold
    vol_multiple::Float64    # reduce when vol > multiple * long-run average
    window::Int              # lookback window for rolling estimates
    cash_fraction::Float64   # fraction to move to cash when triggered
end

"""
    TailRiskHedge(; ...) -> TailRiskHedge
"""
function TailRiskHedge(; var_threshold::Float64=0.03,
                         cvar_threshold::Float64=0.05,
                         vol_multiple::Float64=2.0,
                         window::Int=60,
                         cash_fraction::Float64=0.50)::TailRiskHedge
    return TailRiskHedge(var_threshold, cvar_threshold, vol_multiple, window, cash_fraction)
end

"""
    compute_hedge_ratio(returns, hedge; t) -> Float64

Compute the recommended hedge ratio (fraction of portfolio in cash) at time t.
Returns value in [0, 1]. 0 = fully invested, 1 = fully in cash.
"""
function compute_hedge_ratio(returns::Vector{Float64}, hedge::TailRiskHedge;
                               t::Int=length(returns))::Float64
    t < hedge.window + 1 && return 0.0
    w = returns[max(1, t-hedge.window+1):t]

    rolling_var  = var_historical(w; alpha=0.95)
    rolling_cvar = cvar_historical(w; alpha=0.95)
    rolling_vol  = std(w)
    avg_vol      = std(returns[1:t])

    triggers = Float64[]
    rolling_var > hedge.var_threshold   && push!(triggers, hedge.cash_fraction)
    rolling_cvar > hedge.cvar_threshold && push!(triggers, hedge.cash_fraction)
    rolling_vol > hedge.vol_multiple * avg_vol && push!(triggers, hedge.cash_fraction * 0.7)

    isempty(triggers) && return 0.0
    return minimum(triggers)  # conservative: use smallest cash fraction
end

"""
    cash_vs_trade_rule(returns, hedge) -> Vector{Float64}

Apply tail risk hedging rule across full return series.
Returns position sizes in [0, 1] at each bar.
"""
function cash_vs_trade_rule(returns::Vector{Float64}, hedge::TailRiskHedge)::Vector{Float64}
    n = length(returns)
    positions = ones(n)

    for t in (hedge.window+1):n
        hedge_ratio = compute_hedge_ratio(returns, hedge; t=t)
        positions[t] = 1 - hedge_ratio
    end
    return positions
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Regime-Conditional VaR
# ─────────────────────────────────────────────────────────────────────────────

"""
    regime_conditional_var(returns, regimes; alpha) -> Dict{Int,Float64}

Compute VaR separately for each regime.
Returns Dict{regime_id => VaR_in_that_regime}.
"""
function regime_conditional_var(returns::Vector{Float64}, regimes::Vector{Int};
                                  alpha::Float64=0.99)::Dict{Int,Float64}
    n = min(length(returns), length(regimes))
    K = maximum(regimes[1:n])

    var_by_regime = Dict{Int,Float64}()
    for k in 1:K
        idx_k = findall(==(k), regimes[1:n])
        isempty(idx_k) && continue
        r_k = returns[idx_k]
        var_by_regime[k] = var_historical(r_k; alpha=alpha)
    end
    return var_by_regime
end

"""
    var_stress_by_regime(returns, regimes; alpha, stress_regimes) -> NamedTuple

Compute VaR limits conditioned on regime, with stressed limits for
high-risk regimes.
stress_regimes: set of regime IDs considered "stress" (get 2× VaR limit).
"""
function var_stress_by_regime(returns::Vector{Float64}, regimes::Vector{Int};
                                alpha::Float64=0.99,
                                stress_regimes::Set{Int}=Set([3]))::NamedTuple
    var_by_regime = regime_conditional_var(returns, regimes; alpha=alpha)
    normal_regimes = setdiff(Set(unique(regimes)), stress_regimes)

    normal_var = isempty(normal_regimes) ? 0.0 :
                 mean(get(var_by_regime, k, 0.0) for k in normal_regimes)
    stress_var = isempty(stress_regimes) ? normal_var :
                 mean(get(var_by_regime, k, 0.0) for k in stress_regimes if haskey(var_by_regime, k))

    # Sizing limit: risk-budget-based
    size_limit_normal = normal_var > 1e-8 ? 0.02 / normal_var : 1.0  # target 2% daily var
    size_limit_stress = stress_var  > 1e-8 ? 0.01 / stress_var  : 0.5  # target 1% in stress

    return (var_by_regime=var_by_regime, normal_var=normal_var, stress_var=stress_var,
            size_limit_normal=min(size_limit_normal, 2.0),
            size_limit_stress=min(size_limit_stress, 1.0))
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Correlation Stress Testing
# ─────────────────────────────────────────────────────────────────────────────

"""
    CorrelationStressTest

Stress test portfolio under hypothetical correlation regimes.
"""
struct CorrelationStressTest
    base_correlation::Matrix{Float64}
    stressed_correlation::Matrix{Float64}
    stress_label::String
end

"""
    stress_correlation(R_base, target_rho; method) -> Matrix{Float64}

Create a stressed correlation matrix where all pairwise correlations
are shifted towards target_rho.
method: :blend (interpolate toward target) or :override (set all off-diagonals).
"""
function stress_correlation(R_base::Matrix{Float64}, target_rho::Float64;
                              method::Symbol=:blend, blend_factor::Float64=1.0)::Matrix{Float64}
    d = size(R_base, 1)
    R_stress = copy(R_base)

    for i in 1:d
        for j in (i+1):d
            if method == :override
                R_stress[i, j] = target_rho
                R_stress[j, i] = target_rho
            else  # :blend
                R_stress[i, j] = (1 - blend_factor) * R_base[i, j] + blend_factor * target_rho
                R_stress[j, i] = R_stress[i, j]
            end
        end
    end

    # Ensure PSD via eigenvalue clipping
    R_stress = _ensure_psd(R_stress)
    return R_stress
end

function _ensure_psd(R::Matrix{Float64})::Matrix{Float64}
    vals, vecs = eigen(Symmetric(R))
    vals = max.(vals, 1e-6)
    R_psd = vecs * Diagonal(vals) * vecs'
    # Re-normalise to correlation matrix
    d = size(R_psd, 1)
    D = Diagonal(1 ./ sqrt.(diag(R_psd)))
    return D * R_psd * D
end

"""
    stressed_portfolio_var(weights, vols, R_stressed; alpha, n_sim) -> Float64

Compute portfolio VaR under a stressed correlation matrix.
Uses Monte Carlo simulation with given vols and stressed correlations.
"""
function stressed_portfolio_var(weights::Vector{Float64}, vols::Vector{Float64},
                                  R_stressed::Matrix{Float64};
                                  alpha::Float64=0.99,
                                  n_sim::Int=10000,
                                  rng::AbstractRNG=Random.default_rng())::Float64
    d = length(weights)
    # Covariance = D * R * D where D = Diagonal(vols)
    D   = Diagonal(vols)
    Cov = D * R_stressed * D

    # Cholesky
    L   = try cholesky(Symmetric(Cov + 1e-10 * I)).L catch; D end

    # Simulate returns
    Z = randn(rng, n_sim, d) * L'
    port_rets = Z * weights

    return var_historical(port_rets; alpha=alpha)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Dynamic Stop-Loss via EVT
# ─────────────────────────────────────────────────────────────────────────────

"""
    DynamicStopLoss

Dynamic stop-loss level calibrated using Extreme Value Theory.
The stop is set at the estimated T-day return level from the GPD tail fit.
"""
struct DynamicStopLoss
    lookback::Int         # bars for tail estimation
    quantile_u::Float64   # POT threshold quantile
    T_years::Float64      # recurrence interval for stop level
    min_stop::Float64     # minimum stop loss (floor)
    max_stop::Float64     # maximum stop loss (cap)
end

"""
    DynamicStopLoss(; ...) -> DynamicStopLoss
"""
function DynamicStopLoss(; lookback::Int=252, quantile_u::Float64=0.90,
                           T_years::Float64=1.0, min_stop::Float64=0.02,
                           max_stop::Float64=0.15)::DynamicStopLoss
    return DynamicStopLoss(lookback, quantile_u, T_years, min_stop, max_stop)
end

"""
    calibrate_stop_loss(returns, stop_params; t) -> Float64

Calibrate dynamic stop-loss level at time t using EVT.
Returns stop loss as a positive fraction (e.g. 0.05 = 5% stop).
"""
function calibrate_stop_loss(returns::Vector{Float64}, params::DynamicStopLoss;
                               t::Int=length(returns))::Float64
    t < params.lookback + 20 && return params.max_stop

    w_rets = returns[max(1, t-params.lookback+1):t]
    losses = -w_rets  # losses are positive

    # GPD fit
    n  = length(losses)
    sl = sort(losses)
    u_idx = clamp(round(Int, params.quantile_u * n), 1, n)
    u  = sl[u_idx]
    excesses = filter(x -> x > 0, losses .- u)

    isempty(excesses) && return params.max_stop

    xi, beta = _fit_gpd_simple(excesses)

    # T-year return level
    p_stop = 1 / (params.T_years * 252)
    n_u    = length(excesses)

    if abs(xi) < 1e-6
        stop_level = u + beta * log(n / n_u * (1 - (1 - p_stop)))
    else
        stop_level = u + (beta/xi) * ((n / n_u * (1 - (1-p_stop)))^(-xi) - 1)
    end

    return clamp(stop_level, params.min_stop, params.max_stop)
end

"""
    _fit_gpd_simple(excesses) -> Tuple{Float64, Float64}

Simple GPD fitting (xi, beta) via method of moments.
"""
function _fit_gpd_simple(excesses::Vector{Float64})::Tuple{Float64,Float64}
    n = length(excesses)
    n < 4 && return (0.0, mean(excesses))

    m1 = mean(excesses)
    m2 = var(excesses)

    xi   = 0.5 * (1 - m1^2 / (m2/2 - m1^2))
    beta = m1 * (1 - xi)

    # Sanity bounds
    xi   = clamp(xi,   -0.5, 1.0)
    beta = max(beta, mean(excesses) * 0.1)

    return (xi, beta)
end

"""
    gpd_stop_loss(returns; quantile_u, T_years) -> Float64

One-call function: calibrate GPD stop-loss from a return series.
"""
function gpd_stop_loss(returns::Vector{Float64};
                         quantile_u::Float64=0.90, T_years::Float64=1.0,
                         min_stop::Float64=0.02, max_stop::Float64=0.15)::Float64
    params = DynamicStopLoss(lookback=length(returns), quantile_u=quantile_u,
                               T_years=T_years, min_stop=min_stop, max_stop=max_stop)
    return calibrate_stop_loss(returns, params; t=length(returns))
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Portfolio Risk Report
# ─────────────────────────────────────────────────────────────────────────────

"""
    PortfolioRiskReport

Summary risk report for a portfolio.
"""
struct PortfolioRiskReport
    weights::Vector{Float64}
    names::Vector{String}
    var_99::Float64
    cvar_99::Float64
    max_dd::Float64
    sharpe::Float64
    expected_return::Float64
    expected_vol::Float64
    diversification_ratio::Float64
    tail_risk_regime::Symbol
end

"""
    generate_risk_report(R, weights, names; regimes) -> PortfolioRiskReport

Generate a comprehensive risk report for a portfolio.
"""
function generate_risk_report(R::Matrix{Float64}, weights::Vector{Float64},
                                names::Vector{String};
                                regimes::Union{Vector{Int},Nothing}=nothing)::PortfolioRiskReport
    T, d = size(R)
    port_rets = R * weights

    var_99  = var_historical(port_rets; alpha=0.99)
    cvar_99 = cvar_historical(port_rets; alpha=0.99)

    eq = cumprod(1 .+ port_rets)
    mdd = _max_drawdown_equity(eq)

    m = mean(port_rets)
    s = std(port_rets)
    sharpe = s > 1e-10 ? m / s * sqrt(252) : 0.0

    # Diversification ratio: weighted avg vol / portfolio vol
    indiv_vols = [std(R[:, i]) for i in 1:d]
    wtd_avg_vol = dot(weights, indiv_vols)
    port_vol = s
    dr = port_vol > 1e-10 ? wtd_avg_vol / port_vol : 1.0

    # Tail risk regime
    tail_regime = cvar_99 > var_99 * 1.5 ? :Heavy_Tail :
                  cvar_99 > var_99 * 1.2 ? :Moderate_Tail : :Light_Tail

    return PortfolioRiskReport(weights, names, var_99, cvar_99, mdd, sharpe,
                                m * 252, s * sqrt(252), dr, tail_regime)
end

"""
    Base.show(io, report::PortfolioRiskReport)

Print a formatted risk report.
"""
function Base.show(io::IO, r::PortfolioRiskReport)
    println(io, "=== Portfolio Risk Report ===")
    println(io, @sprintf("  Expected Return (ann): %+.2f%%", r.expected_return*100))
    println(io, @sprintf("  Expected Vol (ann):     %.2f%%", r.expected_vol*100))
    println(io, @sprintf("  Sharpe Ratio:           %.3f",  r.sharpe))
    println(io, @sprintf("  VaR@99%%:               %.4f%%", r.var_99*100))
    println(io, @sprintf("  CVaR@99%%:              %.4f%%", r.cvar_99*100))
    println(io, @sprintf("  Max Drawdown:           %.2f%%", r.max_dd*100))
    println(io, @sprintf("  Diversification Ratio:  %.4f",  r.diversification_ratio))
    println(io, @sprintf("  Tail Risk Regime:       %s",    string(r.tail_risk_regime)))
    println(io, "\n  Weights:")
    for (name, w) in zip(r.names, r.weights)
        println(io, @sprintf("    %-12s: %.4f", name, w))
    end
end

end  # module RiskManagement
