# =============================================================================
# risk_decomposition.jl — Portfolio Risk Decomposition
# =============================================================================
# Provides:
#   - ComponentVaR            Marginal contribution VaR / CVaR
#   - RiskAttribution         Systematic vs idiosyncratic decomposition
#   - FactorRiskDecomp        PCA + named factor (BTC-beta, size, momentum)
#   - DrawdownDecomp          Duration, magnitude, recovery distribution
#   - StressTesting           Historical scenarios (COVID, FTX, Luna, Apr-2026)
#   - CornishFisherVaR        Skewness/kurtosis adjusted VaR
#   - ESDecomposition         Expected Shortfall by instrument
#   - MarginalRiskTimeSeries  Rolling marginal risk contribution
#   - run_risk_decomposition   Top-level driver + JSON export
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, Random, JSON3
# =============================================================================

module RiskDecomposition

using Statistics
using LinearAlgebra
using Random
using JSON3

export ComponentVaR, RiskAttribution, FactorRiskDecomp
export DrawdownDecomp, StressTesting, CornishFisherVaR
export ESDecomposition, MarginalRiskTimeSeries
export run_risk_decomposition

# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Standard normal CDF."""
function _Φ(x::Float64)::Float64
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    c = 0.319381530; c1 = -0.356563782; c2 = 1.781477937
    c3 = -1.821255978; c4 = 1.330274429
    poly = t*(c + t*(c1 + t*(c2 + t*(c3 + t*c4))))
    p = 1.0 - exp(-0.5*x^2)/sqrt(2π) * poly
    x >= 0.0 ? p : 1.0 - p
end

"""Inverse normal CDF (Beasley-Springer-Moro)."""
function _Φinv(p::Float64)::Float64
    p = clamp(p, 1e-10, 1.0-1e-10)
    a = (0.3374754822726869, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187)
    b = (1.0, -0.1532080194741711, -0.2722878986169206,
         0.0519274593571442, 0.0136929880922736)
    r = p < 0.5 ? sqrt(-2log(p)) : sqrt(-2log(1-p))
    num = a[1]+r*(a[2]+r*(a[3]+r*(a[4]+r*(a[5]+r*(a[6]+r*(a[7]+r*(a[8]+r*a[9])))))))
    den = b[1]+r*(b[2]+r*(b[3]+r*(b[4]+r*b[5])))
    p < 0.5 ? -(num/den) : num/den
end

"""Regularise covariance matrix."""
_reg(Σ::Matrix{Float64}; ε=1e-6) = Σ + ε * I

"""Compute portfolio VaR at level alpha from return distribution."""
function _portfolio_var(port_returns::Vector{Float64}, alpha::Float64)::Float64
    sorted = sort(port_returns)
    idx = max(1, round(Int, (1-alpha)*length(sorted)))
    return -sorted[idx]
end

"""Compute portfolio CVaR (Expected Shortfall) at level alpha."""
function _portfolio_cvar(port_returns::Vector{Float64}, alpha::Float64)::Float64
    sorted = sort(port_returns)
    idx = max(1, round(Int, (1-alpha)*length(sorted)))
    return -mean(sorted[1:idx])
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Component VaR and CVaR
# ─────────────────────────────────────────────────────────────────────────────

"""
    ComponentVaR(returns, weights; alpha, n_sims) → NamedTuple

Compute component VaR: the marginal contribution of each asset to total VaR.

Component VaR_i = ρ_{i,p} · VaR_i
where ρ_{i,p} is the correlation between asset i and portfolio.

# Arguments
- `returns` : n×d return matrix
- `weights` : portfolio weights (d-vector)
- `alpha`   : confidence level (default 0.99)
- `n_sims`  : Monte Carlo paths for simulation (default 5000)

# Returns
NamedTuple: (component_VaR, component_CVaR, pct_contribution,
             portfolio_VaR, portfolio_CVaR, marginal_VaR)
"""
function ComponentVaR(returns::Matrix{Float64}, weights::Vector{Float64};
                       alpha::Float64=0.99, n_sims::Int=5000,
                       rng::AbstractRNG=Random.default_rng())
    n, d = size(returns)
    length(weights) == d || error("weights must match assets")
    w = weights / sum(weights)

    Σ = cov(returns)
    mu = mean(returns, dims=1)[:]
    Σ_reg = _reg(Σ)

    # Portfolio statistics
    mu_p    = dot(w, mu)
    sigma_p = sqrt(max(dot(w, Σ_reg * w), 1e-12))

    # Parametric (normal) component VaR
    z_alpha = _Φinv(alpha)
    VaR_parametric = -mu_p + z_alpha * sigma_p

    # Marginal VaR_i = ∂VaR_p/∂w_i = z_alpha * (Σw)_i / σ_p
    grad_sigma = Σ_reg * w / sigma_p
    marginal_VaR = -mu .+ z_alpha .* grad_sigma

    # Component VaR_i = w_i * marginal_VaR_i
    component_VaR = w .* marginal_VaR
    pct_contribution = component_VaR ./ sum(component_VaR)

    # Monte Carlo CVaR decomposition
    L = cholesky(Σ_reg).L
    Z = randn(rng, n_sims, d)
    sim_returns = Z * L' .+ mu'   # n_sims × d
    port_returns = sim_returns * w

    VaR_mc   = _portfolio_var(port_returns, alpha)
    CVaR_mc  = _portfolio_cvar(port_returns, alpha)

    # Component CVaR: E[r_i | r_p ≤ -VaR]
    tail_mask = port_returns .≤ -VaR_mc
    n_tail = sum(tail_mask)

    component_CVaR = if n_tail > 0
        [-mean(sim_returns[tail_mask, j]) * w[j] for j in 1:d]
    else
        zeros(d)
    end

    return (component_VaR=component_VaR, component_CVaR=component_CVaR,
            pct_contribution=pct_contribution, marginal_VaR=marginal_VaR,
            portfolio_VaR=VaR_mc, portfolio_CVaR=CVaR_mc,
            VaR_parametric=VaR_parametric)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk Attribution: Systematic vs Idiosyncratic
# ─────────────────────────────────────────────────────────────────────────────

"""
    RiskAttribution(returns, weights; market_col) → NamedTuple

Decompose portfolio risk into systematic (market beta) and idiosyncratic.

σ²_p = β'Σ_f β + Σ_ε   (factor decomposition)

# Arguments
- `returns`    : n×d asset return matrix
- `weights`    : portfolio weights
- `market_col` : which column is the market / BTC (default 1)

# Returns
NamedTuple: (systematic_var, idiosyncratic_var, betas, R2, tracking_error)
"""
function RiskAttribution(returns::Matrix{Float64}, weights::Vector{Float64};
                          market_col::Int=1)
    n, d = size(returns)
    w = weights / sum(weights)

    market_ret = returns[:, market_col]
    port_ret   = returns * w

    # Beta of portfolio vs market
    σ2_mkt = var(market_ret)
    β_port = cov(port_ret, market_ret) / max(σ2_mkt, 1e-10)

    # Individual asset betas
    betas = [cov(returns[:,j], market_ret) / max(σ2_mkt, 1e-10) for j in 1:d]

    # Systematic vs idiosyncratic variance
    σ2_port       = var(port_ret)
    systematic_var = β_port^2 * σ2_mkt
    idiosyn_var    = max(σ2_port - systematic_var, 0.0)

    # R² of portfolio on market
    R2 = systematic_var / max(σ2_port, 1e-10)

    # Tracking error vs market (annualised)
    tracking_err = std(port_ret .- market_ret) * sqrt(252)

    # Information ratio
    excess_ret = mean(port_ret .- market_ret) * 252
    info_ratio = excess_ret / max(tracking_err, 1e-10)

    # Asset-level systematic and idiosyncratic contributions
    asset_systematic = [betas[j]^2 * σ2_mkt * w[j]^2 for j in 1:d]
    residuals = returns .- market_ret .* betas'
    asset_idiosyn = [var(residuals[:,j]) * w[j]^2 for j in 1:d]

    return (systematic_var=systematic_var, idiosyncratic_var=idiosyn_var,
            portfolio_beta=β_port, asset_betas=betas, R2=R2,
            tracking_error=tracking_err, information_ratio=info_ratio,
            asset_systematic=asset_systematic, asset_idiosyn=asset_idiosyn)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Factor Risk Decomposition
# ─────────────────────────────────────────────────────────────────────────────

"""
    FactorRiskDecomp(returns, weights; n_pca, factor_names) → NamedTuple

Decompose portfolio risk using PCA-based factors and named crypto factors.

# Arguments
- `returns`      : n×d return matrix
- `weights`      : portfolio weights
- `n_pca`        : number of PCA factors (default min(d,3))
- `factor_names` : names for any external factors passed in `factors` kwarg

# Returns
NamedTuple: (pca_contribution, factor_loadings, factor_variance_explained,
             residual_var, factor_VaR_contribution)
"""
function FactorRiskDecomp(returns::Matrix{Float64}, weights::Vector{Float64};
                           n_pca::Int=0, factor_names::Vector{String}=String[],
                           factors::Union{Nothing, Matrix{Float64}}=nothing)
    n, d = size(returns)
    w = weights / sum(weights)
    n_f = n_pca > 0 ? n_pca : min(d, 3)

    # Demean returns
    mu = mean(returns, dims=1)
    R_dm = returns .- mu

    # PCA via eigendecomposition of covariance
    Σ = cov(returns)
    Σ_reg = _reg(Σ)
    F = eigen(Symmetric(Σ_reg))
    # Sort by descending eigenvalue
    order = sortperm(F.values, rev=true)
    eigenvals = F.values[order]
    eigenvecs = F.vectors[:, order]

    # Factor loadings (first n_f eigenvectors)
    B = eigenvecs[:, 1:n_f]                  # d × n_f loadings
    F_scores = R_dm * B                      # n × n_f factor realisations
    total_var = sum(eigenvals)
    pct_var   = eigenvals[1:n_f] ./ max(total_var, 1e-10)

    # Portfolio factor exposures
    port_exposures = B' * w                  # n_f-vector

    # Variance decomposition
    factor_var = [port_exposures[k]^2 * eigenvals[k] for k in 1:n_f]
    total_factor_var = sum(factor_var)
    residual_var = max(dot(w, Σ * w) - total_factor_var, 0.0)

    # Named crypto factors: construct BTC-beta, momentum, size proxies
    crypto_factors = _build_crypto_factors(returns, n)
    n_cf = length(crypto_factors)
    crypto_loadings = zeros(d, n_cf)
    crypto_var_contribs = zeros(n_cf)

    for (k, fname) in enumerate(keys(crypto_factors))
        f_series = crypto_factors[fname]
        length(f_series) != n && continue
        for j in 1:d
            crypto_loadings[j, k] = cov(returns[:,j], f_series) /
                                     max(var(f_series), 1e-10)
        end
        port_exp = dot(crypto_loadings[:,k], w)
        crypto_var_contribs[k] = port_exp^2 * var(f_series)
    end

    # Factor VaR contributions (normal approximation)
    z = _Φinv(0.99)
    factor_VaR = z .* sqrt.(max.(factor_var, 0.0))
    total_VaR  = z * sqrt(max(dot(w, Σ * w), 1e-12))

    return (pca_contribution=pct_var, factor_loadings=B,
            factor_variance_explained=factor_var, residual_var=residual_var,
            total_var=dot(w, Σ*w), factor_VaR_contribution=factor_VaR,
            total_VaR=total_VaR, crypto_loadings=crypto_loadings,
            crypto_var_contribs=crypto_var_contribs,
            eigenvalues=eigenvals[1:n_f])
end

"""Build named crypto factor series from returns matrix."""
function _build_crypto_factors(returns::Matrix{Float64}, n::Int)::Dict{String,Vector{Float64}}
    factors = Dict{String,Vector{Float64}}()
    # BTC-beta: first asset (assumed BTC)
    factors["btc_market"] = returns[:,1]
    # Momentum: trailing 20-day cumulative return
    if n > 22
        mom = zeros(n)
        for t in 22:n
            mom[t] = sum(returns[(t-20):(t-1), 1])
        end
        factors["momentum"] = mom
    end
    # Mean-reversion: negative of short-term return
    factors["mean_reversion"] = -returns[:,1]
    # Volatility factor: rolling std of returns
    vol_f = zeros(n)
    win = 10
    for t in (win+1):n
        vol_f[t] = std(returns[(t-win):(t-1), 1])
    end
    vol_f[1:win] .= std(returns[1:win, 1])
    factors["volatility"] = vol_f
    return factors
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Drawdown Decomposition
# ─────────────────────────────────────────────────────────────────────────────

"""
    DrawdownDecomp(returns; weights) → NamedTuple

Decompose portfolio drawdown into duration, magnitude, and recovery statistics.

# Arguments
- `returns` : n-vector of portfolio returns (or n×d matrix with weights)
- `weights` : optional weights for multi-asset case

# Returns
NamedTuple: (max_drawdown, drawdown_series, durations, magnitudes,
             recovery_times, avg_duration, avg_magnitude, calmar_ratio)
"""
function DrawdownDecomp(returns::Union{Vector{Float64}, Matrix{Float64}};
                         weights::Vector{Float64}=Float64[])
    if isa(returns, Matrix)
        isempty(weights) && (weights = fill(1.0/size(returns,2), size(returns,2)))
        port_ret = returns * (weights / sum(weights))
    else
        port_ret = returns
    end
    n = length(port_ret)

    # Compute cumulative return path
    cum_ret = cumprod(1.0 .+ port_ret)

    # Running maximum
    running_max = accumulate(max, cum_ret)

    # Drawdown series
    dd = (cum_ret .- running_max) ./ running_max

    # Identify drawdown episodes
    durations  = Int[]
    magnitudes = Float64[]
    recovery_times = Int[]

    in_drawdown = false
    dd_start = 0
    dd_max   = 0.0

    for t in 1:n
        if dd[t] < -1e-6
            if !in_drawdown
                in_drawdown = true
                dd_start = t
                dd_max = dd[t]
            else
                dd_max = min(dd_max, dd[t])
            end
        else
            if in_drawdown
                in_drawdown = false
                dur = t - dd_start
                push!(durations, dur)
                push!(magnitudes, -dd_max)
                # Recovery time: from peak to trough to new peak
                rec = t - dd_start
                push!(recovery_times, rec)
            end
        end
    end
    if in_drawdown
        push!(durations, n - dd_start)
        push!(magnitudes, -dd_max)
        push!(recovery_times, n - dd_start)
    end

    max_dd = minimum(dd)
    ann_return = mean(port_ret) * 252
    calmar = ann_return / max(abs(max_dd), 1e-10)

    # Distribution stats
    avg_dur = isempty(durations) ? 0.0 : mean(durations)
    avg_mag = isempty(magnitudes) ? 0.0 : mean(magnitudes)
    max_dur = isempty(durations) ? 0 : maximum(durations)

    # Drawdown at risk (95th percentile of magnitude)
    dd_at_risk = isempty(magnitudes) ? 0.0 : quantile(magnitudes, 0.95)

    return (max_drawdown=max_dd, drawdown_series=dd,
            durations=durations, magnitudes=magnitudes,
            recovery_times=recovery_times, avg_duration=avg_dur,
            avg_magnitude=avg_mag, max_duration=max_dur,
            calmar_ratio=calmar, dd_at_risk=dd_at_risk,
            cumulative_returns=cum_ret)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Stress Testing: Historical Scenarios
# ─────────────────────────────────────────────────────────────────────────────

"""
    StressTesting(returns, dates; weights) → NamedTuple

Apply historical stress scenarios to portfolio.

Scenarios included:
- COVID crash (Feb-Mar 2020): ~50% crypto drawdown
- FTX collapse (Nov 2022): ~25% drawdown
- Luna/Terra (May 2022): ~60% drawdown
- April 2026: +30% BTC rally / alt-coin divergence
- Generic: 1-day -20%, -30%, -40% shocks

# Arguments
- `returns`  : n×d return matrix
- `dates`    : optional date labels for each row
- `weights`  : portfolio weights

# Returns
NamedTuple: (scenario_losses, scenario_VaR, worst_day, stress_CVaR)
"""
function StressTesting(returns::Matrix{Float64},
                        dates::Vector{String}=String[];
                        weights::Vector{Float64}=Float64[])
    n, d = size(returns)
    isempty(weights) && (weights = fill(1.0/d, d))
    w = weights / sum(weights)
    port_ret = returns * w

    # ── Historical percentile scenarios ───────────────────────────────────
    sorted_port = sort(port_ret)
    scenarios = Dict{String, Float64}()

    # Worst single day in history
    scenarios["worst_day_actual"]     = minimum(port_ret)
    scenarios["best_day_actual"]      = maximum(port_ret)
    scenarios["5th_percentile"]       = quantile(port_ret, 0.05)
    scenarios["1st_percentile"]       = quantile(port_ret, 0.01)
    scenarios["0.1th_percentile"]     = sorted_port[max(1, round(Int, 0.001*n))]

    # ── Parametric shock scenarios ─────────────────────────────────────────
    μ = mean(port_ret); σ = std(port_ret)

    # COVID-like crash: -15σ shock (systemic correlation jump)
    scenarios["covid_crash_equiv"]    = μ - 15σ

    # FTX-like: institutional withdrawal, credit contraction
    scenarios["ftx_collapse_equiv"]   = μ - 8σ

    # Luna/Terra: stablecoin de-peg, contagion spiral
    scenarios["luna_terra_equiv"]     = μ - 12σ

    # April 2026 rally scenario
    scenarios["apr2026_rally_equiv"]  = μ + 10σ

    # Generic shocks
    scenarios["shock_minus_20pct"]    = -0.20
    scenarios["shock_minus_30pct"]    = -0.30
    scenarios["shock_minus_40pct"]    = -0.40

    # ── Multi-asset stress: correlation spike ──────────────────────────────
    Σ = cov(returns)
    Σ_stress = 0.5 * Σ + 0.5 * (ones(d,d) * maximum(Σ))   # all correlations → 1
    Σ_stress = _reg(Σ_stress)
    σ_stress = sqrt(max(dot(w, Σ_stress * w), 1e-12))
    scenarios["correlation_crisis_VaR99"] = _Φinv(0.99) * σ_stress

    # ── Scenario P&L by asset ──────────────────────────────────────────────
    shock_vec = [-0.10, -0.20, -0.30]
    asset_pnl = Dict{String, Vector{Float64}}()
    for shock in shock_vec
        asset_pnl["shock_$(round(Int, shock*100))pct"] =
            [shock * w[j] for j in 1:d]
    end

    # ── Summary statistics ─────────────────────────────────────────────────
    worst_day = minimum(port_ret)
    stress_CVaR = mean(port_ret[port_ret .≤ quantile(port_ret, 0.01)])

    return (scenarios=scenarios, asset_pnl=asset_pnl,
            worst_day=worst_day, stress_CVaR=stress_CVaR,
            portfolio_mean=μ, portfolio_std=σ,
            correlation_stress_VaR=scenarios["correlation_crisis_VaR99"])
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Cornish-Fisher VaR
# ─────────────────────────────────────────────────────────────────────────────

"""
    CornishFisherVaR(returns, weights; alpha) → NamedTuple

Compute Cornish-Fisher adjusted VaR incorporating skewness and excess kurtosis.

z_CF = z + (z²-1)/6 · S + (z³-3z)/24 · K - (2z³-5z)/36 · S²

where S = skewness, K = excess kurtosis, z = Φ⁻¹(α).

# Arguments
- `returns` : n×d return matrix
- `weights` : portfolio weights
- `alpha`   : confidence level (default 0.99)

# Returns
NamedTuple: (VaR_CF, VaR_normal, adjustment, skewness, excess_kurtosis)
"""
function CornishFisherVaR(returns::Matrix{Float64}, weights::Vector{Float64};
                           alpha::Float64=0.99)
    n, d = size(returns)
    w = weights / sum(weights)
    port_ret = returns * w

    μ = mean(port_ret); σ = std(port_ret)
    n_r = length(port_ret)

    # Sample moments
    S = sum((r - μ)^3 for r in port_ret) / (n_r * σ^3)   # skewness
    K = sum((r - μ)^4 for r in port_ret) / (n_r * σ^4) - 3.0  # excess kurtosis

    z = _Φinv(alpha)

    # Cornish-Fisher expansion
    z_cf = z + (z^2 - 1)/6 * S + (z^3 - 3z)/24 * K - (2z^3 - 5z)/36 * S^2

    VaR_CF     = -(μ - z_cf * σ)
    VaR_normal = -(μ - z * σ)
    adjustment = VaR_CF - VaR_normal

    # ES from Cornish-Fisher
    # Approximate: ES_CF ≈ ES_normal · (1 + adjustment_factor)
    φ_z = exp(-0.5*z^2)/sqrt(2π)
    ES_normal = -(μ - σ * φ_z / (1.0 - alpha))
    CF_ratio = z_cf / z
    ES_CF = ES_normal * CF_ratio

    # Per-asset contributions (assuming normal returns at margin)
    Σ = cov(returns)
    Σ_reg = _reg(Σ)
    σ_p = sqrt(max(dot(w, Σ_reg * w), 1e-12))
    marginal_sigma = Σ_reg * w / σ_p
    cf_contribution = w .* z_cf .* marginal_sigma

    return (VaR_CF=VaR_CF, VaR_normal=VaR_normal, adjustment=adjustment,
            ES_CF=ES_CF, ES_normal=ES_normal,
            skewness=S, excess_kurtosis=K,
            z_normal=z, z_CF=z_cf,
            cf_contribution=cf_contribution, alpha=alpha)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Expected Shortfall Decomposition by Instrument
# ─────────────────────────────────────────────────────────────────────────────

"""
    ESDecomposition(returns, weights; alpha) → NamedTuple

Decompose Expected Shortfall (CVaR) across instruments.

ES_i = w_i · E[r_i | r_p ≤ -VaR_p]

# Arguments
- `returns`  : n×d return matrix
- `weights`  : portfolio weights
- `alpha`    : confidence level (default 0.99)

# Returns
NamedTuple: (ES_contribution, pct_ES, portfolio_ES, conditional_means,
             diversification_ratio)
"""
function ESDecomposition(returns::Matrix{Float64}, weights::Vector{Float64};
                          alpha::Float64=0.99,
                          rng::AbstractRNG=Random.default_rng())
    n, d = size(returns)
    w = weights / sum(weights)
    port_ret = returns * w

    # Portfolio VaR at alpha
    VaR_val = _portfolio_var(port_ret, alpha)

    # Tail events
    tail_mask = port_ret .≤ -VaR_val
    n_tail = sum(tail_mask)

    ES_contribution = zeros(d)
    conditional_means = zeros(d)

    if n_tail > 0
        for j in 1:d
            cond_mean = mean(returns[tail_mask, j])
            conditional_means[j] = cond_mean
            ES_contribution[j] = -w[j] * cond_mean
        end
    end

    portfolio_ES = sum(ES_contribution)
    pct_ES = portfolio_ES > 0 ? ES_contribution ./ portfolio_ES : zeros(d)

    # Diversification ratio: sum(w_i * VaR_i) / VaR_portfolio
    individual_VaR = [-quantile(returns[:,j], 1.0-alpha) for j in 1:d]
    undiversified_VaR = dot(w, abs.(individual_VaR))
    div_ratio = undiversified_VaR / max(VaR_val, 1e-10)

    # Marginal ES: ∂ES_p/∂w_i
    marginal_ES = conditional_means .* (-1.0)

    return (ES_contribution=ES_contribution, pct_ES=pct_ES,
            portfolio_ES=portfolio_ES, conditional_means=conditional_means,
            portfolio_VaR=VaR_val, n_tail_events=n_tail,
            diversification_ratio=div_ratio, marginal_ES=marginal_ES)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Marginal Risk Contribution Time Series
# ─────────────────────────────────────────────────────────────────────────────

"""
    MarginalRiskTimeSeries(returns, weights; window, alpha) → NamedTuple

Compute rolling marginal risk contributions of each asset over time.

# Arguments
- `returns` : n×d return matrix
- `weights` : portfolio weights
- `window`  : rolling window (default 60)
- `alpha`   : VaR confidence level (default 0.99)

# Returns
NamedTuple: (marginal_VaR_ts, component_VaR_ts, portfolio_VaR_ts, dates_idx)
"""
function MarginalRiskTimeSeries(returns::Matrix{Float64}, weights::Vector{Float64};
                                 window::Int=60, alpha::Float64=0.99)
    n, d = size(returns)
    w = weights / sum(weights)
    n_roll = n - window + 1

    marginal_VaR_ts   = zeros(n_roll, d)
    component_VaR_ts  = zeros(n_roll, d)
    portfolio_VaR_ts  = zeros(n_roll)

    z = _Φinv(alpha)

    for t in 1:n_roll
        window_data = returns[t:(t+window-1), :]
        Σ_w = cov(window_data)
        Σ_reg = _reg(Σ_w)
        mu_w = mean(window_data, dims=1)[:]
        mu_p = dot(w, mu_w)
        σ_p  = sqrt(max(dot(w, Σ_reg * w), 1e-12))

        # Marginal VaR
        grad_σ = Σ_reg * w / σ_p
        mVaR = -mu_w .+ z .* grad_σ
        cVaR = w .* mVaR

        marginal_VaR_ts[t, :]  = mVaR
        component_VaR_ts[t, :] = cVaR
        portfolio_VaR_ts[t]    = -mu_p + z * σ_p
    end

    # Trend in risk contributions
    trend_cVaR = zeros(d)
    for j in 1:d
        ts = component_VaR_ts[:, j]
        t_idx = 1:n_roll
        trend_cVaR[j] = cov(Float64.(t_idx), ts) / var(Float64.(t_idx))
    end

    return (marginal_VaR_ts=marginal_VaR_ts, component_VaR_ts=component_VaR_ts,
            portfolio_VaR_ts=portfolio_VaR_ts, window=window,
            trend_component_VaR=trend_cVaR, n_windows=n_roll)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-Level Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_risk_decomposition(returns; weights, asset_names, out_path) → Dict

Complete portfolio risk decomposition pipeline.

# Arguments
- `returns`     : n×d return matrix
- `weights`     : portfolio weights (default equal-weight)
- `asset_names` : optional string labels for assets
- `out_path`    : optional JSON export path

# Returns
Dict with all risk metrics, decompositions, and stress tests.

# Example
```julia
using Random
rng = Random.MersenneTwister(42)
n, d = 500, 4
returns = randn(rng, n, d) .* 0.03
returns[:, 1] .+= 0.001   # BTC-like positive drift
w = [0.4, 0.3, 0.2, 0.1]
results = run_risk_decomposition(returns; weights=w,
           asset_names=["BTC","ETH","SOL","AVAX"],
           out_path="risk_decomp.json")
println("Portfolio VaR(99%): ", results["component_VaR"]["portfolio_VaR"])
println("Max drawdown: ", results["drawdown"]["max_drawdown"])
```
"""
function run_risk_decomposition(returns::Matrix{Float64};
                                 weights::Vector{Float64}=Float64[],
                                 asset_names::Vector{String}=String[],
                                 out_path::Union{String,Nothing}=nothing)
    n, d = size(returns)
    isempty(weights) && (weights = fill(1.0/d, d))
    w = weights / sum(weights)
    isempty(asset_names) && (asset_names = ["asset_$(j)" for j in 1:d])

    rng = Random.default_rng()
    results = Dict{String, Any}()

    # ── Component VaR ──────────────────────────────────────────────────────
    @info "Computing component VaR/CVaR..."
    cvr = ComponentVaR(returns, w; alpha=0.99, n_sims=3000, rng=rng)
    results["component_VaR"] = Dict(
        "component_VaR"       => Dict(asset_names[j] => cvr.component_VaR[j] for j in 1:d),
        "component_CVaR"      => Dict(asset_names[j] => cvr.component_CVaR[j] for j in 1:d),
        "pct_contribution"    => Dict(asset_names[j] => cvr.pct_contribution[j] for j in 1:d),
        "portfolio_VaR"       => cvr.portfolio_VaR,
        "portfolio_CVaR"      => cvr.portfolio_CVaR,
        "VaR_parametric"      => cvr.VaR_parametric
    )

    # ── Risk Attribution ───────────────────────────────────────────────────
    @info "Risk attribution (systematic vs idiosyncratic)..."
    ra = RiskAttribution(returns, w; market_col=1)
    results["risk_attribution"] = Dict(
        "systematic_var"    => ra.systematic_var,
        "idiosyncratic_var" => ra.idiosyncratic_var,
        "portfolio_beta"    => ra.portfolio_beta,
        "asset_betas"       => Dict(asset_names[j] => ra.asset_betas[j] for j in 1:d),
        "R2"                => ra.R2,
        "tracking_error"    => ra.tracking_error,
        "information_ratio" => ra.information_ratio
    )

    # ── Factor Risk Decomposition ──────────────────────────────────────────
    @info "Factor risk decomposition (PCA + crypto factors)..."
    frd = FactorRiskDecomp(returns, w; n_pca=min(d, 3))
    results["factor_risk"] = Dict(
        "pca_variance_explained" => frd.pca_contribution,
        "factor_VaR_contribution" => frd.factor_VaR_contribution,
        "total_VaR"              => frd.total_VaR,
        "residual_var"           => frd.residual_var,
        "eigenvalues"            => frd.eigenvalues
    )

    # ── Drawdown Decomposition ─────────────────────────────────────────────
    @info "Drawdown decomposition..."
    dd = DrawdownDecomp(returns; weights=w)
    results["drawdown"] = Dict(
        "max_drawdown"   => dd.max_drawdown,
        "avg_duration"   => dd.avg_duration,
        "avg_magnitude"  => dd.avg_magnitude,
        "max_duration"   => dd.max_duration,
        "calmar_ratio"   => dd.calmar_ratio,
        "dd_at_risk_95"  => dd.dd_at_risk,
        "n_episodes"     => length(dd.durations)
    )

    # ── Stress Testing ─────────────────────────────────────────────────────
    @info "Stress testing (historical + parametric scenarios)..."
    st = StressTesting(returns; weights=w)
    results["stress_testing"] = Dict(
        "scenarios"           => st.scenarios,
        "worst_day"           => st.worst_day,
        "stress_CVaR"         => st.stress_CVaR,
        "portfolio_mean"      => st.portfolio_mean,
        "portfolio_std"       => st.portfolio_std
    )

    # ── Cornish-Fisher VaR ─────────────────────────────────────────────────
    @info "Cornish-Fisher adjusted VaR..."
    cf = CornishFisherVaR(returns, w; alpha=0.99)
    results["cornish_fisher"] = Dict(
        "VaR_CF"           => cf.VaR_CF,
        "VaR_normal"       => cf.VaR_normal,
        "adjustment"       => cf.adjustment,
        "skewness"         => cf.skewness,
        "excess_kurtosis"  => cf.excess_kurtosis,
        "ES_CF"            => cf.ES_CF
    )

    # ── ES Decomposition ──────────────────────────────────────────────────
    @info "Expected Shortfall decomposition..."
    es = ESDecomposition(returns, w; alpha=0.99, rng=rng)
    results["ES_decomposition"] = Dict(
        "ES_contribution"     => Dict(asset_names[j] => es.ES_contribution[j] for j in 1:d),
        "pct_ES"              => Dict(asset_names[j] => es.pct_ES[j] for j in 1:d),
        "portfolio_ES"        => es.portfolio_ES,
        "portfolio_VaR"       => es.portfolio_VaR,
        "diversification_ratio" => es.diversification_ratio,
        "n_tail_events"       => es.n_tail_events
    )

    # ── Marginal Risk Time Series ──────────────────────────────────────────
    @info "Rolling marginal risk contributions..."
    if n >= 80
        mrs = MarginalRiskTimeSeries(returns, w; window=min(60, n÷4))
        results["marginal_risk_ts"] = Dict(
            "portfolio_VaR_ts"      => mrs.portfolio_VaR_ts,
            "current_marginal_VaR"  => Dict(asset_names[j] => mrs.marginal_VaR_ts[end, j] for j in 1:d),
            "trend_component_VaR"   => Dict(asset_names[j] => mrs.trend_component_VaR[j] for j in 1:d),
            "window"                => mrs.window
        )
    end

    # ── Summary ────────────────────────────────────────────────────────────
    results["summary"] = Dict(
        "n_assets"           => d,
        "n_observations"     => n,
        "weights"            => Dict(asset_names[j] => w[j] for j in 1:d),
        "VaR_99"             => cvr.portfolio_VaR,
        "CVaR_99"            => cvr.portfolio_CVaR,
        "VaR_99_CF"          => cf.VaR_CF,
        "max_drawdown"       => dd.max_drawdown,
        "portfolio_beta"     => ra.portfolio_beta,
        "R2_vs_market"       => ra.R2,
        "calmar_ratio"       => dd.calmar_ratio,
        "worst_scenario"     => minimum(values(st.scenarios))
    )

    if !isnothing(out_path)
        open(out_path, "w") do io
            JSON3.write(io, results)
        end
        @info "Results written to $out_path"
    end

    return results
end

end  # module RiskDecomposition
