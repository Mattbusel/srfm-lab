"""
ExtremeValueTheory — Extreme Value Theory for crypto tail risk.

Implements: GEV distribution (Gumbel/Fréchet/Weibull), GPD for
peaks-over-threshold, Hill estimator, Pickands-Balkema-de Haan,
block maxima, return levels, multivariate EVT spectral measure,
extreme VaR/ES, and per-asset worst-case loss estimation.
"""
module ExtremeValueTheory

using Statistics
using LinearAlgebra
using Random

export GEVFit, GPDFit, HillEstimator, BlockMaxima
export ReturnLevel, MultivariateEVT, ExtremeVaR, ExtremeES
export run_evt_analysis

# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Regularised lower incomplete gamma via series (for GEV likelihood)."""
function _lgamma_approx(x::Float64)::Float64
    # Stirling approximation
    x ≤ 0.0 && return 1e10
    x + (x - 0.5)*log(x) - x + 0.5*log(2π) +
    1/(12x) - 1/(360x^3) + 1/(1260x^5)
end

"""Natural log of the GEV density."""
function _gev_logpdf(x::Float64, μ::Float64, σ::Float64, ξ::Float64)::Float64
    σ ≤ 0.0 && return -Inf
    z = (x - μ) / σ
    if abs(ξ) < 1e-8  # Gumbel limit
        -log(σ) - z - exp(-z)
    else
        t = 1.0 + ξ * z
        t ≤ 0.0 && return -Inf
        -log(σ) - (1.0 + 1.0/ξ) * log(t) - t^(-1.0/ξ)
    end
end

"""GEV CDF."""
function _gev_cdf(x::Float64, μ::Float64, σ::Float64, ξ::Float64)::Float64
    z = (x - μ) / σ
    if abs(ξ) < 1e-8
        exp(-exp(-z))
    else
        t = 1.0 + ξ * z
        t ≤ 0.0 && return ξ < 0 ? 1.0 : 0.0
        exp(-t^(-1.0/ξ))
    end
end

"""GPD log-density."""
function _gpd_logpdf(x::Float64, σ::Float64, ξ::Float64)::Float64
    σ ≤ 0.0 && return -Inf
    x < 0.0  && return -Inf
    if abs(ξ) < 1e-8  # exponential limit
        -log(σ) - x/σ
    else
        t = 1.0 + ξ * x / σ
        t ≤ 0.0 && return -Inf
        -log(σ) - (1.0 + 1.0/ξ) * log(t)
    end
end

"""GPD CDF."""
function _gpd_cdf(x::Float64, σ::Float64, ξ::Float64)::Float64
    x < 0.0 && return 0.0
    if abs(ξ) < 1e-8
        1.0 - exp(-x/σ)
    else
        t = 1.0 + ξ * x / σ
        t ≤ 0.0 && return ξ < 0 ? 1.0 : 0.0
        1.0 - t^(-1.0/ξ)
    end
end

"""Basic gradient-free optimiser (coordinate ascent with restarts)."""
function _maximise(f::Function, θ0::Vector{Float64}, lower::Vector{Float64},
                    upper::Vector{Float64}; max_iter::Int=3000)
    best_val = f(θ0)
    best_θ   = copy(θ0)
    step     = (upper .- lower) .* 0.05

    for iter in 1:max_iter
        improved = false
        for k in eachindex(θ0)
            for δ in [step[k], -step[k]]
                θ_try = copy(best_θ)
                θ_try[k] = clamp(θ_try[k] + δ, lower[k], upper[k])
                v = f(θ_try)
                if v > best_val
                    best_val = v; best_θ = copy(θ_try); improved = true
                end
            end
        end
        improved ? (step .*= 1.05) : (step .*= 0.9)
        maximum(step) < 1e-10 && break
    end
    return best_θ, best_val
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. GEV Distribution Fit (Block Maxima)
# ─────────────────────────────────────────────────────────────────────────────

"""
    GEVFit(maxima) → NamedTuple

Fit the Generalized Extreme Value distribution to block maxima via MLE.

GEV(μ, σ, ξ): F(x) = exp{-[1 + ξ(x-μ)/σ]^{-1/ξ}}
- ξ > 0: Fréchet (heavy-tailed)
- ξ = 0: Gumbel (exponential tails)
- ξ < 0: Weibull (bounded upper tail)

# Arguments
- `maxima` : vector of block maxima (e.g. monthly maximums of absolute returns)

# Returns
NamedTuple: (mu, sigma, xi, loglik, AIC, type, CI_xi)
"""
function GEVFit(maxima::Vector{Float64})
    n = length(maxima)
    n < 4 && error("Need at least 4 observations for GEV fit")

    # Method of moments initial estimate
    mu0  = mean(maxima) - 0.5772 * std(maxima) * sqrt(6) / π
    sig0 = std(maxima) * sqrt(6) / π
    xi0  = 0.1

    function ll(θ)
        μ, σ, ξ = θ
        σ ≤ 0.0 && return -Inf
        s = 0.0
        for x in maxima
            lp = _gev_logpdf(x, μ, σ, ξ)
            isfinite(lp) ? (s += lp) : return -Inf
        end
        return s
    end

    θ_opt, ll_opt = _maximise(ll, [mu0, sig0, xi0],
                               [minimum(maxima) - 10*sig0, 1e-4, -0.5],
                               [maximum(maxima), 10*sig0, 1.5])
    μ, σ, ξ = θ_opt

    # Approximate 95% CI for ξ via profile likelihood (grid)
    xi_grid = range(-0.4, 1.2, length=50) |> collect
    profile_ll = Float64[]
    for xi_try in xi_grid
        opt_inner, ll_inner = _maximise(
            θ -> ll([θ[1], θ[2], xi_try]),
            [μ, σ],
            [minimum(maxima) - 10σ, 1e-4],
            [maximum(maxima), 10σ])
        push!(profile_ll, ll_inner)
    end
    # 95% CI: within 1.92 log-units of maximum
    max_pll = maximum(profile_ll)
    ci_mask = profile_ll .≥ max_pll - 1.921
    ci_xi = (xi_grid[findfirst(ci_mask)], xi_grid[findlast(ci_mask)])

    type = if ξ > 0.05
        "Fréchet (heavy-tailed, ξ=$(round(ξ,digits=3)))"
    elseif ξ < -0.05
        "Weibull (bounded, ξ=$(round(ξ,digits=3)))"
    else
        "Gumbel (light-tailed)"
    end

    AIC = -2*ll_opt + 6  # 3 params × 2

    return (mu=μ, sigma=σ, xi=ξ, loglik=ll_opt, AIC=AIC,
            type=type, CI_xi=ci_xi, n=n)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Block Maxima Method
# ─────────────────────────────────────────────────────────────────────────────

"""
    BlockMaxima(returns; block_size, loss) → NamedTuple

Extract block maxima from return series and fit GEV distribution.

# Arguments
- `returns`    : return vector
- `block_size` : observations per block (default 21 = trading month)
- `loss`       : if true, work with losses (-returns) (default true)

# Returns
NamedTuple: (maxima, gev_fit, return_levels, block_size)
"""
function BlockMaxima(returns::Vector{Float64}; block_size::Int=21, loss::Bool=true)
    x = loss ? -returns : returns
    n = length(x)
    n_blocks = n ÷ block_size
    n_blocks < 4 && error("Not enough blocks (need block_size ≤ n/4)")

    # Extract maxima
    maxima = [maximum(x[(k*block_size+1):min((k+1)*block_size, n)])
              for k in 0:(n_blocks-1)]

    # Fit GEV
    gev = GEVFit(maxima)

    # Return levels for T = 2, 5, 10, 20, 50, 100 blocks
    T_vec = [2, 5, 10, 20, 50, 100]
    rl = ReturnLevel(gev, T_vec)

    return (maxima=maxima, gev_fit=gev, return_levels=rl,
            block_size=block_size, n_blocks=n_blocks)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. GPD Peaks-Over-Threshold
# ─────────────────────────────────────────────────────────────────────────────

"""
    GPDFit(x; threshold, method, n_threshold_candidates) → NamedTuple

Fit the Generalized Pareto Distribution to exceedances over a threshold.
Implements the Pickands-Balkema-de Haan theorem:
P(X - u > y | X > u) → GPD(σ, ξ) as u → x_F.

# Arguments
- `x`                    : data vector (losses, positive)
- `threshold`            : threshold u (default: 90th percentile)
- `method`               : :mle or :pwm (probability-weighted moments)
- `n_threshold_candidates`: try this many thresholds for stability plot

# Returns
NamedTuple: (sigma, xi, threshold, exceedances, loglik, mean_excess,
             stability_plot, extreme_quantile_99_9)
"""
function GPDFit(x::Vector{Float64}; threshold::Float64=NaN,
                method::Symbol=:mle, n_threshold_candidates::Int=20)
    n = length(x)
    u = isnan(threshold) ? quantile(x, 0.90) : threshold

    # Exceedances
    exc = x[x .> u] .- u
    n_exc = length(exc)
    n_exc < 5 && error("Too few exceedances (n_exc=$n_exc); lower threshold")

    if method == :pwm
        # Probability-weighted moments
        x_s = sort(exc)
        M0 = mean(x_s)
        M1 = mean(x_s .* ((1:n_exc) .- 1) ./ max.(n_exc .- 1, 1))
        σ_pwm = 2M0*M1 / (M0 - 2M1)
        ξ_pwm = 2 - M0 / (M0 - 2M1)
        σ_est, ξ_est = σ_pwm, ξ_pwm
        ll_opt = sum(_gpd_logpdf(e, σ_est, ξ_est) for e in exc)
    else
        # MLE
        function ll(θ)
            σ, ξ = θ
            σ ≤ 0 && return -Inf
            s = 0.0
            for e in exc
                lp = _gpd_logpdf(e, σ, ξ)
                isfinite(lp) ? (s += lp) : return -Inf
            end
            return s
        end
        σ0 = mean(exc); ξ0 = 0.1
        θ_opt, ll_opt = _maximise(ll, [σ0, ξ0], [1e-5, -0.5], [10σ0, 1.5])
        σ_est, ξ_est = θ_opt
    end

    # Mean excess function (should be linear for GPD)
    sorted_x = sort(x)
    thresholds_me = sorted_x[max(1,n÷10):n÷10:end]
    mean_excess = [mean(x[x .> t] .- t) for t in thresholds_me]

    # Threshold stability plot
    thresholds_stab = quantile.(Ref(x), range(0.80, 0.97, length=n_threshold_candidates))
    σ_stab = Float64[]; ξ_stab = Float64[]
    for u_t in thresholds_stab
        exc_t = x[x .> u_t] .- u_t
        length(exc_t) < 5 && continue
        function ll_t(θ)
            σ, ξ = θ; σ ≤ 0 && return -Inf
            s = sum(_gpd_logpdf(e, σ, ξ) for e in exc_t)
            isfinite(s) ? s : -Inf
        end
        θ_t, _ = _maximise(ll_t, [mean(exc_t), 0.1],
                             [1e-5, -0.5], [5mean(exc_t), 1.5];
                             max_iter=500)
        push!(σ_stab, θ_t[1]); push!(ξ_stab, θ_t[2])
    end

    # Extreme quantile at 99.9%
    p = 0.999
    ζ_u = n_exc / n  # fraction exceeding threshold
    q_extreme = if abs(ξ_est) < 1e-8
        u + σ_est * log((1-p)/(ζ_u))
    else
        u + σ_est / ξ_est * (((1-p)/ζ_u)^(-ξ_est) - 1.0)
    end

    return (sigma=σ_est, xi=ξ_est, threshold=u, exceedances=exc,
            n_exceedances=n_exc, loglik=ll_opt,
            mean_excess=(thresholds=thresholds_me, values=mean_excess),
            stability=(thresholds=thresholds_stab[1:length(σ_stab)],
                       sigma=σ_stab, xi=ξ_stab),
            extreme_quantile_99_9=q_extreme)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Hill Estimator
# ─────────────────────────────────────────────────────────────────────────────

"""
    HillEstimator(x; k_range) → NamedTuple

Estimate the tail index using the Hill (1975) estimator.

γ̂_k = (1/k) ∑_{i=1}^{k} log(X_{(n-i+1)}) - log(X_{(n-k)})

where X_{(1)} ≤ X_{(2)} ≤ ... are the order statistics.

# Arguments
- `x`       : loss data (positive values)
- `k_range` : range of k values for stability plot (default 5:5:n÷2)

# Returns
NamedTuple: (gamma, alpha_tail, k_optimal, stability_plot, CI_95)
"""
function HillEstimator(x::Vector{Float64}; k_range::AbstractVector=Int[])
    n = length(x)
    x_pos = abs.(x)
    sorted = sort(x_pos, rev=true)

    ks = isempty(k_range) ? (5:5:(n÷2)) |> collect : collect(k_range)
    filter!(k -> k < n, ks)

    hill_vals = Float64[]
    for k in ks
        γ = mean(log.(sorted[1:k])) - log(sorted[k+1])
        push!(hill_vals, γ)
    end

    # Optimal k: minimum of AMSE-optimal selector (simplified: Drees-Kaufmann)
    # Use k that minimises variance (plateau region)
    k_opt_idx = if length(hill_vals) >= 5
        # Find the longest stable plateau
        window_stab = 5
        min_var_idx = 1
        min_var = Inf
        for i in 1:(length(hill_vals)-window_stab+1)
            v = var(hill_vals[i:(i+window_stab-1)])
            if v < min_var
                min_var = v; min_var_idx = i + window_stab÷2
            end
        end
        min_var_idx
    else
        length(hill_vals) ÷ 2 + 1
    end

    γ_opt = hill_vals[min(k_opt_idx, length(hill_vals))]
    α_opt = 1.0 / max(γ_opt, 1e-3)   # tail index α = 1/γ

    # 95% CI: asymptotic normality of Hill estimator
    k_opt = ks[min(k_opt_idx, length(ks))]
    se_γ = γ_opt / sqrt(k_opt)
    CI = (γ_opt - 1.96*se_γ, γ_opt + 1.96*se_γ)

    return (gamma=γ_opt, alpha_tail=α_opt, k_optimal=k_opt,
            stability_plot=(k=ks, gamma=hill_vals), CI_95=CI,
            interpretation=α_opt > 4 ? "finite kurtosis" :
                           α_opt > 2 ? "infinite kurtosis" : "infinite variance")
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Return Level and Return Period
# ─────────────────────────────────────────────────────────────────────────────

"""
    ReturnLevel(gev_fit, T_vec) → NamedTuple

Compute return levels x_T such that P(X > x_T) = 1/T.
Inversion of the GEV CDF: x_T = μ - σ/ξ · [1 - (-log(1-1/T))^{-ξ}]

# Arguments
- `gev_fit` : fitted GEV NamedTuple from GEVFit
- `T_vec`   : return periods (e.g. [10, 50, 100, 200, 500])
"""
function ReturnLevel(gev_fit::NamedTuple, T_vec::Vector{Int})
    μ, σ, ξ = gev_fit.mu, gev_fit.sigma, gev_fit.xi
    levels = Float64[]

    for T in T_vec
        p = 1.0 - 1.0/T  # non-exceedance probability
        y_T = -log(p)     # reduced variate
        if abs(ξ) < 1e-8
            x_T = μ - σ * log(y_T)
        else
            x_T = μ + σ / ξ * (y_T^(-ξ) - 1.0)
        end
        push!(levels, x_T)
    end

    return (T=T_vec, levels=levels, mu=μ, sigma=σ, xi=ξ)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Multivariate EVT: Spectral Measure
# ─────────────────────────────────────────────────────────────────────────────

"""
    MultivariateEVT(returns; radial_threshold, n_bins) → NamedTuple

Estimate the spectral measure for bivariate extreme value theory.
The spectral measure H on the unit simplex characterises tail dependence.

# Arguments
- `returns`           : n×2 bivariate return matrix
- `radial_threshold`  : quantile threshold for exceedances (default 0.95)
- `n_bins`            : angular bins for spectral density (default 20)

# Returns
NamedTuple: (spectral_measure, angular_density, tail_dependence_coeff,
             extremal_index, bivariate_extremal_index)
"""
function MultivariateEVT(returns::Matrix{Float64}; radial_threshold::Float64=0.95,
                          n_bins::Int=20)
    size(returns, 2) < 2 && error("MultivariateEVT requires at least bivariate data")
    n, d = size(returns)

    # Work with losses
    x = -returns[:, 1]; y = -returns[:, 2]

    # Standardise margins to unit Pareto (rank transform)
    rank_x = sortperm(sortperm(x)) ./ (n+1)
    rank_y = sortperm(sortperm(y)) ./ (n+1)
    # Unit Pareto: Z = 1/(1-F)
    z_x = 1.0 ./ max.(1.0 .- rank_x, 1e-6)
    z_y = 1.0 ./ max.(1.0 .- rank_y, 1e-6)

    # Pseudo-polar coordinates
    r = z_x .+ z_y
    w = z_x ./ max.(r, 1e-10)  # angular component ∈ [0,1]

    # Select exceedances
    r_threshold = quantile(r, radial_threshold)
    exc_mask = r .> r_threshold
    w_exc = w[exc_mask]

    # Spectral measure: histogram of angles for extreme observations
    bins = range(0.0, 1.0, length=n_bins+1) |> collect
    h_counts = zeros(n_bins)
    for wi in w_exc
        b = clamp(floor(Int, wi * n_bins) + 1, 1, n_bins)
        h_counts[b] += 1
    end
    h_counts ./= max(sum(h_counts), 1)  # normalise

    # Tail dependence coefficient from spectral measure
    # λ = 1 - E[max(w, 1-w)] for symmetric angular measure
    if !isempty(w_exc)
        lambda_tail = 2.0 - 2.0 * mean(max.(w_exc, 1.0 .- w_exc))
        lambda_tail = clamp(lambda_tail, 0.0, 1.0)
    else
        lambda_tail = 0.0
    end

    # Extremal index (univariate) via blocks estimator
    θ_x = _extremal_index(x, 20)
    θ_y = _extremal_index(y, 20)

    # Bivariate extremal index
    θ_biv = 1.0 - lambda_tail  # asymptotic independence → θ_biv = 1

    return (spectral_measure=h_counts, angular_bins=bins[1:n_bins],
            angular_density=w_exc, tail_dependence_coeff=lambda_tail,
            extremal_index_x=θ_x, extremal_index_y=θ_y,
            bivariate_extremal_index=θ_biv, n_exceedances=sum(exc_mask))
end

"""Extremal index via blocks estimator."""
function _extremal_index(x::Vector{Float64}, block_size::Int)::Float64
    n = length(x)
    u = quantile(x, 0.95)
    n_exceed = sum(x .> u)
    n_exceed == 0 && return 1.0

    # Block exceedances
    n_blocks = n ÷ block_size
    n_block_exceed = sum(
        any(x[(k*block_size+1):min((k+1)*block_size,n)] .> u)
        for k in 0:(n_blocks-1))

    return n_block_exceed / max(n_exceed, 1)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Extreme VaR and ES
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExtremeVaR(returns; alpha, method) → NamedTuple

Compute extreme VaR and ES at high confidence levels (99%, 99.9%, 99.99%)
using GPD extrapolation beyond the empirical distribution.

# Arguments
- `returns` : return vector (or loss = -returns)
- `alpha`   : confidence levels (default [0.99, 0.999, 0.9999])
- `method`  : :gpd, :gev, or :both (default :gpd)

# Returns
NamedTuple: (VaR, ES, method, gpd_fit, confidence_levels)
"""
function ExtremeVaR(returns::Vector{Float64};
                    alpha::Vector{Float64}=Float64[0.99, 0.999, 0.9999],
                    method::Symbol=:gpd)
    losses = -returns  # work with losses

    VaR_dict = Dict{Float64, Float64}()
    ES_dict  = Dict{Float64, Float64}()

    # Historical VaR (for comparison)
    for a in alpha
        q_idx = clamp(round(Int, (1-a) * length(losses)), 1, length(losses))
        VaR_dict[a] = sort(losses, rev=true)[q_idx]
    end

    # GPD extrapolation
    gpd = nothing
    if method == :gpd || method == :both
        try
            gpd = GPDFit(losses; threshold=quantile(losses, 0.90))
            n = length(losses)
            ζ_u = gpd.n_exceedances / n

            for a in alpha
                # GPD quantile
                p_excess = (1.0 - a) / ζ_u  # conditional exceedance probability
                p_excess = clamp(p_excess, 1e-10, 1.0)
                if abs(gpd.xi) < 1e-8
                    VaR_dict[a] = gpd.threshold + gpd.sigma * log(1.0/p_excess)
                else
                    VaR_dict[a] = gpd.threshold + gpd.sigma/gpd.xi * (p_excess^(-gpd.xi) - 1.0)
                end

                # ES under GPD
                if gpd.xi < 1.0
                    ES_dict[a] = VaR_dict[a] / (1-gpd.xi) + (gpd.sigma - gpd.xi * gpd.threshold) / (1-gpd.xi)
                else
                    ES_dict[a] = Inf  # undefined for ξ ≥ 1
                end
            end
        catch e
            @warn "GPD fit failed: $e; falling back to empirical quantiles"
        end
    end

    # Fill ES for missing entries
    for a in alpha
        if !haskey(ES_dict, a)
            tail_idx = round(Int, (1-a) * length(losses))
            sorted_l = sort(losses, rev=true)
            ES_dict[a] = mean(sorted_l[1:max(1,tail_idx)])
        end
    end

    # Annualised figures (daily losses → annual by √252)
    annual_factor = sqrt(252)

    return (VaR=VaR_dict, ES=ES_dict, gpd_fit=gpd,
            confidence_levels=alpha,
            VaR_annual=Dict(a => VaR_dict[a]*annual_factor for a in alpha),
            ES_annual=Dict(a => ES_dict[a]*annual_factor for a in alpha))
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Extreme ES (analytical for GPD)
# ─────────────────────────────────────────────────────────────────────────────

"""
    ExtremeES(gpd_fit, alpha) → Float64

Compute Expected Shortfall from fitted GPD.
ES_α = (VaR_α + σ - ξ·u) / (1-ξ)   for ξ < 1.
"""
function ExtremeES(gpd_fit::NamedTuple, alpha::Float64)::Float64
    σ, ξ, u = gpd_fit.sigma, gpd_fit.xi, gpd_fit.threshold
    abs(ξ - 1.0) < 1e-6 && return Inf

    # VaR_α from GPD
    n = length(gpd_fit.exceedances)
    ζ_u = gpd_fit.n_exceedances / (n + gpd_fit.n_exceedances)
    p_exc = (1-alpha) / max(ζ_u, 1e-10)
    p_exc = clamp(p_exc, 1e-10, 1.0)

    var_a = if abs(ξ) < 1e-8
        u + σ * log(1.0/p_exc)
    else
        u + σ/ξ * (p_exc^(-ξ) - 1.0)
    end

    es = (var_a + σ - ξ * u) / (1.0 - ξ)
    return max(es, var_a)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-Level Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_evt_analysis(returns_matrix; asset_names, out_path) → Dict

Full EVT analysis pipeline for crypto assets.

# Arguments
- `returns_matrix` : n×d matrix of daily returns
- `asset_names`    : optional asset labels
- `out_path`       : optional JSON output path (requires JSON3)

# Returns
Dict with GEV, GPD, Hill, extreme VaR, and multivariate EVT results.

# Example
```julia
using Random
rng = Random.MersenneTwister(42)
n, d = 1000, 3
returns = randn(rng, n, d) .* 0.03
returns[:, 1] .-= 0.001  # slight negative drift (BTC-like crisis)
results = run_evt_analysis(returns;
    asset_names=["BTC","ETH","SOL"])
println("BTC 99.9% extreme VaR (daily): ", results["BTC"]["extreme_VaR"]["VaR"][0.999])
```
"""
function run_evt_analysis(returns_matrix::Matrix{Float64};
                           asset_names::Vector{String}=String[],
                           block_size::Int=21,
                           out_path::Union{String,Nothing}=nothing)
    n, d = size(returns_matrix)
    isempty(asset_names) && (asset_names = ["asset_$j" for j in 1:d])

    results = Dict{String, Any}()

    for j in 1:d
        ret = returns_matrix[:, j]
        losses = -ret
        name = asset_names[j]
        @info "EVT analysis for $name..."

        asset_res = Dict{String, Any}()

        # Block Maxima + GEV
        try
            bm = BlockMaxima(losses; block_size=block_size, loss=false)
            asset_res["block_maxima"] = Dict(
                "n_blocks"      => bm.n_blocks,
                "gev_mu"        => bm.gev_fit.mu,
                "gev_sigma"     => bm.gev_fit.sigma,
                "gev_xi"        => bm.gev_fit.xi,
                "gev_type"      => bm.gev_fit.type,
                "return_levels" => Dict(string(T) => rl
                    for (T, rl) in zip(bm.return_levels.T, bm.return_levels.levels))
            )
        catch e
            asset_res["block_maxima"] = Dict("error" => string(e))
        end

        # GPD Peaks-over-Threshold
        try
            gpd = GPDFit(losses; threshold=quantile(losses, 0.90))
            asset_res["gpd"] = Dict(
                "sigma"                   => gpd.sigma,
                "xi"                      => gpd.xi,
                "threshold"               => gpd.threshold,
                "n_exceedances"           => gpd.n_exceedances,
                "extreme_quantile_99_9"   => gpd.extreme_quantile_99_9,
                "loglik"                  => gpd.loglik
            )
        catch e
            asset_res["gpd"] = Dict("error" => string(e))
        end

        # Hill Estimator
        try
            hill = HillEstimator(losses)
            asset_res["hill"] = Dict(
                "gamma"          => hill.gamma,
                "alpha_tail"     => hill.alpha_tail,
                "k_optimal"      => hill.k_optimal,
                "CI_95_lower"    => hill.CI_95[1],
                "CI_95_upper"    => hill.CI_95[2],
                "interpretation" => hill.interpretation
            )
        catch e
            asset_res["hill"] = Dict("error" => string(e))
        end

        # Extreme VaR / ES
        try
            evar = ExtremeVaR(ret; alpha=[0.99, 0.999, 0.9999])
            asset_res["extreme_VaR"] = Dict(
                "VaR"  => Dict(string(a) => evar.VaR[a] for a in keys(evar.VaR)),
                "ES"   => Dict(string(a) => evar.ES[a]  for a in keys(evar.ES))
            )
        catch e
            asset_res["extreme_VaR"] = Dict("error" => string(e))
        end

        results[name] = asset_res
    end

    # Multivariate EVT (first two assets)
    if d >= 2
        try
            @info "Multivariate EVT ($(asset_names[1]), $(asset_names[2]))..."
            mv_evt = MultivariateEVT(returns_matrix[:, 1:2]; radial_threshold=0.95)
            results["multivariate_evt"] = Dict(
                "tail_dependence"        => mv_evt.tail_dependence_coeff,
                "extremal_index_1"       => mv_evt.extremal_index_x,
                "extremal_index_2"       => mv_evt.extremal_index_y,
                "bivariate_extremal_idx" => mv_evt.bivariate_extremal_index,
                "n_exceedances"          => mv_evt.n_exceedances,
                "spectral_measure"       => mv_evt.spectral_measure
            )
        catch e
            results["multivariate_evt"] = Dict("error" => string(e))
        end
    end

    # Portfolio-level worst case
    port_ret = mean(returns_matrix, dims=2)[:,1]
    try
        port_evar = ExtremeVaR(port_ret; alpha=[0.99, 0.999])
        results["portfolio_extreme_risk"] = Dict(
            "VaR_99"    => port_evar.VaR[0.99],
            "VaR_999"   => port_evar.VaR[0.999],
            "ES_99"     => port_evar.ES[0.99],
            "ES_999"    => port_evar.ES[0.999],
            "annual_ES_99" => port_evar.ES_annual[0.99]
        )
    catch e
        results["portfolio_extreme_risk"] = Dict("error" => string(e))
    end

    # Optional JSON export
    if !isnothing(out_path)
        try
            using JSON3
            open(out_path, "w") do io
                JSON3.write(io, results)
            end
            @info "EVT results written to $out_path"
        catch
            @warn "JSON3 not available; skipping JSON export"
        end
    end

    return results
end

end  # module ExtremeValueTheory
