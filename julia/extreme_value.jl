"""
extreme_value.jl

Extreme Value Theory (EVT) for tail risk quantification in crypto trading.

Two fundamental approaches:
  1. Block Maxima (BM): Model periodwise maxima via GEV distribution
  2. Peak-over-Threshold (POT): Model exceedances via GPD

Generalized Extreme Value (GEV) distribution:
    F(x; μ, σ, ξ) = exp(-(1 + ξ(x-μ)/σ)^{-1/ξ})
  ξ > 0: Fréchet (heavy tails, crypto returns)
  ξ = 0: Gumbel (light tails, e.g. normal maxima)
  ξ < 0: Weibull (bounded upper tail)

Generalised Pareto Distribution (GPD):
    G(y; σ, ξ) = 1 - (1 + ξy/σ)^{-1/ξ}  for y > 0
  where y = x - u is the exceedance over threshold u.

References:
  Embrechts, Klüppelberg & Mikosch (1997) "Modelling Extremal Events"
  McNeil, Frey & Embrechts (2015) "Quantitative Risk Management"
  de Haan & Ferreira (2006) "Extreme Value Theory: An Introduction"
"""

using Optim
using Distributions
using Statistics
using LinearAlgebra
using Random
using Plots
using HypothesisTests

# ─────────────────────────────────────────────────────────────────────────────
# GEV DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

"""
GEV CDF: F(x; μ, σ, ξ).
"""
function gev_cdf(x::Real, μ::Real, σ::Real, ξ::Real)
    σ > 0 || return NaN
    if abs(ξ) < 1e-8
        # Gumbel limit
        z = (x - μ) / σ
        return exp(-exp(-z))
    else
        t = 1 + ξ * (x - μ) / σ
        t <= 0 && return ξ > 0 ? 0.0 : 1.0
        return exp(-t^(-1/ξ))
    end
end

"""
GEV PDF: f(x; μ, σ, ξ).
"""
function gev_pdf(x::Real, μ::Real, σ::Real, ξ::Real)
    σ > 0 || return NaN
    if abs(ξ) < 1e-8
        z = (x - μ) / σ
        return (1/σ) * exp(-z - exp(-z))
    else
        t = 1 + ξ * (x - μ) / σ
        t <= 0 && return 0.0
        return (1/σ) * t^(-1/ξ - 1) * exp(-t^(-1/ξ))
    end
end

"""
GEV quantile function (inverse CDF).
For return level computation.
"""
function gev_quantile(p::Real, μ::Real, σ::Real, ξ::Real)
    0 < p < 1 || return NaN
    if abs(ξ) < 1e-8
        return μ - σ * log(-log(p))
    else
        return μ + σ * ((-log(p))^(-ξ) - 1) / ξ
    end
end

"""
Log-likelihood for GEV distribution.
"""
function gev_loglikelihood(params::Vector{Float64}, maxima::Vector{Float64})
    μ, σ_raw, ξ = params
    σ = exp(σ_raw)  # ensure σ > 0

    n = length(maxima)
    ll = -n * log(σ)

    for x in maxima
        if abs(ξ) < 1e-8
            z = (x - μ) / σ
            ll -= z + exp(-z)
        else
            t = 1 + ξ * (x - μ) / σ
            t <= 0 && return -Inf
            ll -= (1/ξ + 1) * log(t) + t^(-1/ξ)
        end
    end

    return ll
end

"""
Fit GEV distribution to block maxima via MLE.
Returns (μ̂, σ̂, ξ̂) and standard errors.
"""
function fit_gev(maxima::Vector{Float64}; verbose=false)
    n = length(maxima)
    n < 10 && @warn "Small sample ($n maxima). GEV estimates may be unreliable."

    # Method of moments initialisation
    x̄ = mean(maxima)
    s² = var(maxima)
    # GEV moments: E[X] ≈ μ + σ(Γ(1-ξ)-1)/ξ for ξ≠0
    # Initial: assume Gumbel
    σ₀ = sqrt(6 * s²) / π
    μ₀ = x̄ - 0.5772 * σ₀
    ξ₀ = 0.1

    θ₀ = [μ₀, log(σ₀), ξ₀]

    neg_ll(θ) = -gev_loglikelihood(θ, maxima)

    result = optimize(neg_ll, θ₀, LBFGS(),
                      Optim.Options(iterations=2000, g_tol=1e-8, show_trace=verbose))

    θ_opt = Optim.minimizer(result)
    μ̂ = θ_opt[1]
    σ̂ = exp(θ_opt[2])
    ξ̂ = θ_opt[3]

    # Approximate standard errors via Hessian inverse
    H = Optim.hessian!(result)
    se = try
        sqrt.(abs.(diag(inv(H))))
    catch
        fill(NaN, 3)
    end

    verbose && begin
        println("GEV MLE: μ=$(round(μ̂,digits=4)), σ=$(round(σ̂,digits=4)), ξ=$(round(ξ̂,digits=4))")
        println("SE:      μ=$(round(se[1],digits=4)), σ=$(round(se[2],digits=4)), ξ=$(round(se[3],digits=4))")
        tail_type = ξ̂ > 0.05 ? "Fréchet (heavy tail)" : ξ̂ < -0.05 ? "Weibull (bounded)" : "Gumbel"
        println("Tail type: $tail_type")
    end

    return (μ=μ̂, σ=σ̂, ξ=ξ̂, se=se, result=result)
end

"""
Extract block maxima from a time series.
block_size: number of observations per block (e.g., 22 trading days = 1 month).
"""
function block_maxima(returns::Vector{Float64}, block_size::Int; loss=true)
    # Optionally work with losses (-returns)
    data = loss ? -returns : returns
    n = length(data)
    n_blocks = n ÷ block_size
    maxima = [maximum(data[(b-1)*block_size+1 : b*block_size]) for b in 1:n_blocks]
    return maxima
end

# ─────────────────────────────────────────────────────────────────────────────
# GPD AND PEAK-OVER-THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

"""
GPD CDF: G(y; σ, ξ) = 1 - (1 + ξy/σ)^{-1/ξ}, y > 0.
"""
function gpd_cdf(y::Real, σ::Real, ξ::Real)
    σ > 0 && y >= 0 || return NaN
    if abs(ξ) < 1e-8
        return 1 - exp(-y/σ)
    else
        t = 1 + ξ * y / σ
        t <= 0 && return 1.0
        return 1 - t^(-1/ξ)
    end
end

"""
GPD PDF.
"""
function gpd_pdf(y::Real, σ::Real, ξ::Real)
    σ > 0 && y >= 0 || return NaN
    if abs(ξ) < 1e-8
        return exp(-y/σ) / σ
    else
        t = 1 + ξ * y / σ
        t <= 0 && return 0.0
        return t^(-1/ξ - 1) / σ
    end
end

"""
GPD quantile function.
"""
function gpd_quantile(p::Real, σ::Real, ξ::Real)
    0 <= p < 1 || return NaN
    if abs(ξ) < 1e-8
        return -σ * log(1 - p)
    else
        return σ * ((1-p)^(-ξ) - 1) / ξ
    end
end

"""
GPD log-likelihood.
"""
function gpd_loglikelihood(σ_raw::Float64, ξ::Float64, exceedances::Vector{Float64})
    σ = exp(σ_raw)
    n = length(exceedances)
    ll = -n * log(σ)

    for y in exceedances
        if abs(ξ) < 1e-8
            ll -= y / σ
        else
            t = 1 + ξ * y / σ
            t <= 0 && return -Inf
            ll -= (1/ξ + 1) * log(t)
        end
    end

    return ll
end

"""
Fit GPD to exceedances above threshold u.
"""
function fit_gpd(returns::Vector{Float64}, u::Float64;
                  loss=true, verbose=false)
    data = loss ? -returns : returns
    exceedances = data[data .> u] .- u

    isempty(exceedances) && error("No exceedances above threshold u=$u")
    n_u = length(exceedances)
    n   = length(data)

    verbose && println("GPD: $n_u exceedances above u=$(round(u,digits=4)) ($(round(100*n_u/n,digits=2))% of data)")

    # Initial parameters
    σ₀ = mean(exceedances)
    ξ₀ = 0.1

    neg_ll(θ) = -gpd_loglikelihood(θ[1], θ[2], exceedances)

    result = optimize(neg_ll, [log(σ₀), ξ₀], LBFGS(),
                      Optim.Options(iterations=2000, g_tol=1e-8, show_trace=verbose))

    θ_opt = Optim.minimizer(result)
    σ̂ = exp(θ_opt[1])
    ξ̂ = θ_opt[2]

    verbose && println("GPD MLE: σ=$(round(σ̂,digits=4)), ξ=$(round(ξ̂,digits=4))")

    return (σ=σ̂, ξ=ξ̂, u=u, n_u=n_u, n=n, exceedances=exceedances, result=result)
end

# ─────────────────────────────────────────────────────────────────────────────
# HILL ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

"""
Hill estimator for the extreme value index (tail index 1/ξ).

For X with Pareto-like tail: P(X > x) ~ x^{-α}, the Hill estimator is:
    Ĥ_k = (1/k) Σᵢ₌₁ᵏ log(X_(n-i+1) / X_(n-k))

where X_(1) ≤ ... ≤ X_(n) are order statistics.
The tail index α̂ = 1/Ĥ_k, so ξ̂ = Ĥ_k.

Returns (k_values, hill_estimates) for the Hill plot.
"""
function hill_estimator(data::Vector{Float64}; loss=true, k_min=10)
    X = loss ? sort(-data) : sort(data)
    X = reverse(X)  # descending order
    n = length(X)

    k_vals = k_min:n-1
    hill = zeros(length(k_vals))

    for (idx, k) in enumerate(k_vals)
        hill[idx] = mean(log(X[i] / X[k+1]) for i in 1:k)
    end

    return collect(k_vals), hill
end

"""
Select optimal k for Hill estimator using minimum variance criterion.
"""
function optimal_hill_k(k_vals::Vector{Int}, hill::Vector{Float64}; window=20)
    n = length(hill)
    var_vals = Float64[]

    for i in window:n
        seg = hill[max(1,i-window+1):i]
        push!(var_vals, var(seg))
    end

    k_star_idx = argmin(var_vals) + window - 1
    k_star = k_vals[k_star_idx]
    ξ_star = hill[k_star_idx]

    return k_star, ξ_star
end

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD SELECTION
# ─────────────────────────────────────────────────────────────────────────────

"""
Mean excess function (mean residual life plot) for threshold selection.

e(u) = E[X - u | X > u] = σ_u / (1 - ξ)  [GPD mean excess]

For GPD, e(u) is LINEAR in u with slope ξ/(1-ξ).
Choose threshold u* where the plot becomes linear.
"""
function mean_excess_function(data::Vector{Float64};
                               loss=true,
                               n_thresholds=50,
                               quantile_range=(0.80, 0.99))
    X = loss ? -data : data
    q_lo, q_hi = quantile_range

    u_vals = quantile(X, range(q_lo, q_hi, length=n_thresholds))
    me_vals = Float64[]
    ci_lo   = Float64[]
    ci_hi   = Float64[]

    for u in u_vals
        exc = X[X .> u] .- u
        if length(exc) < 5
            push!(me_vals, NaN); push!(ci_lo, NaN); push!(ci_hi, NaN)
            continue
        end
        me = mean(exc)
        se = std(exc) / sqrt(length(exc))
        push!(me_vals, me)
        push!(ci_lo, me - 1.96*se)
        push!(ci_hi, me + 1.96*se)
    end

    return u_vals, me_vals, ci_lo, ci_hi
end

# ─────────────────────────────────────────────────────────────────────────────
# RETURN LEVELS
# ─────────────────────────────────────────────────────────────────────────────

"""
Compute return level xₘ: value exceeded on average once every m blocks.

For GEV with n blocks per year:
    xₘ = μ + σ[((-log(1-1/m))^{-ξ} - 1)/ξ]

For POT with λ exceedances per year:
    xₘ = u + (σ/ξ)[(mλ)^ξ - 1]

Returns a vector of (return_period, return_level) pairs.
"""
function return_levels_gev(gev_params; return_periods=[1, 5, 10, 25, 50, 100],
                             blocks_per_year=52)  # weekly blocks
    μ, σ, ξ = gev_params.μ, gev_params.σ, gev_params.ξ

    println("\nGEV Return Levels:")
    println("  μ=$(round(μ,digits=4)), σ=$(round(σ,digits=4)), ξ=$(round(ξ,digits=4))")
    println("  ─" ^ 25)
    println("  Period (yr) │ Return Level")
    println("  ─" ^ 25)

    levels = Float64[]
    for T_yr in return_periods
        # Exceedance probability = 1/(T_yr × blocks_per_year)
        p = 1 - 1/(T_yr * blocks_per_year)
        rl = gev_quantile(p, μ, σ, ξ)
        push!(levels, rl)
        println("  $(lpad(T_yr, 10)) │ $(round(rl, digits=4))")
    end

    return collect(zip(return_periods, levels))
end

"""
Return levels for POT/GPD.
"""
function return_levels_gpd(gpd_params; return_periods=[1, 5, 10, 25, 50, 100],
                             obs_per_year=252)
    σ, ξ, u = gpd_params.σ, gpd_params.ξ, gpd_params.u
    n_u, n = gpd_params.n_u, gpd_params.n

    λ = n_u / n * obs_per_year  # exceedances per year

    println("\nGPD Return Levels (POT method):")
    println("  u=$(round(u,digits=4)), σ=$(round(σ,digits=4)), ξ=$(round(ξ,digits=4))")
    println("  λ=$(round(λ,digits=2)) exceedances/year")
    println("  ─" ^ 30)
    println("  Period (yr) │ Return Level │ % Loss")
    println("  ─" ^ 30)

    levels = Float64[]
    for T_yr in return_periods
        if abs(ξ) < 1e-8
            rl = u + σ * log(T_yr * λ)
        else
            rl = u + (σ/ξ) * ((T_yr * λ)^ξ - 1)
        end
        push!(levels, rl)
        println("  $(lpad(T_yr, 10)) │ $(lpad(round(rl,digits=4), 12)) │ $(round(100*rl,digits=2))%")
    end

    return collect(zip(return_periods, levels))
end

# ─────────────────────────────────────────────────────────────────────────────
# MULTIVARIATE EXTREMES
# ─────────────────────────────────────────────────────────────────────────────

"""
Tail dependence coefficients between two random variables X₁, X₂.

Upper tail dependence: λᵤ = lim_{u→1} P(X₂ > F₂⁻¹(u) | X₁ > F₁⁻¹(u))
Lower tail dependence: λₗ = lim_{u→0} P(X₂ < F₂⁻¹(u) | X₁ < F₁⁻¹(u))

Estimated non-parametrically:
    λ̂ᵤ = #{X₁ > q_u AND X₂ > q_u} / #{X₁ > q_u}
"""
function tail_dependence(X1::Vector{Float64}, X2::Vector{Float64};
                          u=0.95)
    n = length(X1)
    q1 = quantile(X1, u)
    q2 = quantile(X2, u)

    # Upper tail
    idx_upper = X1 .> q1
    λ_upper = sum(X2[idx_upper] .> q2) / sum(idx_upper)

    # Lower tail
    q1_lo = quantile(X1, 1-u)
    q2_lo = quantile(X2, 1-u)
    idx_lower = X1 .< q1_lo
    λ_lower = sum(X2[idx_lower] .< q2_lo) / sum(idx_lower)

    return (upper=λ_upper, lower=λ_lower)
end

"""
Spectral measure estimation for bivariate extremes.

For (X₁, X₂) in the max-domain of attraction of a bivariate extreme distribution,
the spectral measure H on [0,1] captures the dependence structure.

Estimated via the empirical spectral measure at the quantile level u:
    W = X₁/(X₁+X₂) conditional on X₁+X₂ > some large quantile
"""
function spectral_measure(X1::Vector{Float64}, X2::Vector{Float64};
                           u_prob=0.90, n_bins=20)
    # Standardise to unit Fréchet margins
    F1 = ecdf(X1)
    F2 = ecdf(X2)
    Z1 = -1 ./ log.(F1.(X1))
    Z2 = -1 ./ log.(F2.(X2))

    # Threshold on radial component R = Z1 + Z2
    R = Z1 .+ Z2
    u_r = quantile(R, u_prob)

    # Angular component W = Z1/R conditional on R > u
    W = Z1[R .> u_r] ./ R[R .> u_r]

    # Histogram estimate of spectral measure
    bins = range(0, 1, length=n_bins+1)
    hist = fit(Histogram, W, bins)
    H_weights = hist.weights ./ sum(hist.weights)

    return W, H_weights, midpoints(bins)
end

function midpoints(r)
    n = length(r)
    [(r[i] + r[i+1])/2 for i in 1:n-1]
end

# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL AND PORTFOLIO TAIL RISK
# ─────────────────────────────────────────────────────────────────────────────

"""
Compute tail risk metrics for a single symbol's returns.
Returns: VaR, CVaR/ES, GPD fit parameters.
"""
function symbol_tail_risk(returns::Vector{Float64}, symbol::String;
                           u_prob=0.95, confidence_levels=[0.99, 0.995, 0.999])
    println("\n─" ^ 50)
    println("Tail Risk: $symbol ($(length(returns)) observations)")
    println("─" ^ 50)

    # Threshold: 95th percentile of losses
    losses = -returns
    u = quantile(losses, u_prob)

    # Fit GPD
    gpd = fit_gpd(returns, u; loss=true, verbose=true)

    # Return levels
    rl_data = return_levels_gpd(gpd; return_periods=[1, 5, 10])

    # VaR and CVaR at specified confidence levels
    n = length(returns)
    n_u = gpd.n_u

    println("\n  Tail Risk Metrics:")
    for α in confidence_levels
        # VaR: GPD quantile
        p_excess = (1-α) * n / n_u  # probability of exceedance given above threshold
        if 0 < p_excess < 1
            y_α = gpd_quantile(1 - p_excess, gpd.σ, gpd.ξ)
            VaR_α = u + y_α

            # CVaR = E[X | X > VaR] = VaR + E[GPD exceedance above y_α]
            if abs(gpd.ξ) < 1e-8
                CVaR_α = VaR_α + gpd.σ
            elseif gpd.ξ < 1
                CVaR_α = (VaR_α + gpd.σ - gpd.ξ * u) / (1 - gpd.ξ)
            else
                CVaR_α = Inf
            end
            println("    α=$(α): VaR=$(round(100*VaR_α,digits=2))%, CVaR=$(round(100*CVaR_α,digits=2))%")
        end
    end

    return gpd
end

"""
Portfolio tail risk using the Pickands-Balkema-de Haan theorem.
Portfolio loss = w'·losses, then apply POT.
"""
function portfolio_tail_risk(returns_matrix::Matrix{Float64},
                              weights::Vector{Float64};
                              u_prob=0.95)
    n_assets = size(returns_matrix, 2)
    length(weights) == n_assets || throw(DimensionMismatch("Weight dimension mismatch"))
    abs(sum(weights) - 1.0) < 1e-10 || throw(ArgumentError("Weights must sum to 1"))

    portfolio_returns = returns_matrix * weights
    gpd = fit_gpd(portfolio_returns, quantile(-portfolio_returns, u_prob);
                   loss=true, verbose=true)

    println("\nPortfolio Tail Risk (weights: $(round.(weights, digits=3)))")
    return_levels_gpd(gpd)

    return gpd
end

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Return level plot for GEV fit.
"""
function plot_return_levels_gev(gev_params, maxima::Vector{Float64};
                                  blocks_per_year=52, title="GEV Return Levels")
    μ, σ, ξ = gev_params.μ, gev_params.σ, gev_params.ξ

    T_range = range(1/(blocks_per_year), 100, length=500)
    rl_fitted = [gev_quantile(1 - 1/(T * blocks_per_year), μ, σ, ξ) for T in T_range]

    # Empirical return periods
    n = length(maxima)
    sorted_max = sort(maxima)
    emp_probs = (1:n) ./ (n+1)
    emp_T = -1 ./ (blocks_per_year .* log.(emp_probs))

    p = plot(T_range, rl_fitted, label="GEV fit", color=:blue, linewidth=2,
             xscale=:log10, xlabel="Return period (years)",
             ylabel="Return level", title=title)
    scatter!(p, emp_T, sorted_max, label="Empirical", markersize=3, color=:black)

    return p
end

"""
Plot GPD fit with diagnostic plots.
"""
function plot_gpd_diagnostics(gpd_params, title="GPD Diagnostics")
    σ, ξ = gpd_params.σ, gpd_params.ξ
    exc = gpd_params.exceedances
    n = length(exc)

    # Probability plot
    sorted_exc = sort(exc)
    emp_probs = (1:n) ./ (n+1)
    fitted_probs = [gpd_cdf(y, σ, ξ) for y in sorted_exc]
    p1 = scatter(emp_probs, fitted_probs, label="", markersize=3,
                 xlabel="Empirical probability", ylabel="Fitted probability",
                 title="Probability Plot")
    plot!(p1, [0,1], [0,1], color=:red, linestyle=:dash, label="y=x")

    # Quantile plot
    emp_quantiles = sorted_exc
    fit_quantiles = [gpd_quantile(p, σ, ξ) for p in emp_probs]
    p2 = scatter(fit_quantiles, emp_quantiles, label="", markersize=3,
                 xlabel="Fitted quantiles", ylabel="Empirical quantiles",
                 title="Quantile Plot")
    qq_max = max(maximum(fit_quantiles[isfinite.(fit_quantiles)]), maximum(emp_quantiles))
    plot!(p2, [0, qq_max], [0, qq_max], color=:red, linestyle=:dash, label="y=x")

    # Density
    x_range = range(0, quantile(Exponential(σ), 0.99), length=200)
    p3 = histogram(exc, normalize=:pdf, bins=30, alpha=0.5, label="Data", title="GPD Fit")
    plot!(p3, x_range, [gpd_pdf(y, σ, ξ) for y in x_range],
          color=:red, linewidth=2, label="GPD(σ=$(round(σ,digits=3)),ξ=$(round(ξ,digits=3)))")

    return plot(p1, p2, p3, layout=(1,3), size=(1050, 320), suptitle=title)
end

"""
Hill plot for tail index estimation.
"""
function plot_hill(k_vals, hill; title="Hill Plot")
    p = plot(k_vals, hill, label="Hill estimate", color=:blue, linewidth=1.5,
             xlabel="Number of order statistics k",
             ylabel="ξ̂ (tail index)",
             title=title)
    hline!(p, [0.0], color=:black, linestyle=:dash, label="ξ=0")
    return p
end

"""
Mean excess plot for threshold selection.
"""
function plot_mean_excess(u_vals, me_vals, ci_lo, ci_hi; title="Mean Excess Plot")
    valid = .!isnan.(me_vals)
    p = plot(u_vals[valid], me_vals[valid], label="Mean excess e(u)",
             color=:blue, linewidth=2, xlabel="Threshold u",
             ylabel="Mean excess", title=title)
    plot!(p, u_vals[valid], ci_lo[valid], color=:blue, linestyle=:dash,
          alpha=0.5, label="95% CI")
    plot!(p, u_vals[valid], ci_hi[valid], color=:blue, linestyle=:dash,
          alpha=0.5, label=nothing)
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

function demo()
    Random.seed!(42)
    println("=" ^ 60)
    println("Extreme Value Theory Demo")
    println("=" ^ 60)

    # Synthetic crypto-like returns: t-distribution (heavy tails)
    n = 1000
    ν = 4.0  # degrees of freedom → ξ ≈ 1/ν = 0.25
    returns = rand(TDist(ν), n) .* 0.02  # ≈ 2% daily volatility

    println("\nSynthetic data: t($ν) distributed returns, n=$n")
    println("Expected ξ ≈ $(round(1/ν, digits=3))")

    # 1. Block Maxima + GEV
    println("\n" * "─" ^ 40)
    println("1. Block Maxima Method (GEV)")
    println("─" ^ 40)

    maxima = block_maxima(returns, 20; loss=true)  # monthly blocks
    println("$(length(maxima)) blocks of 20 observations")

    gev = fit_gev(maxima; verbose=true)
    rl_gev = return_levels_gev(gev; blocks_per_year=12)

    # 2. POT with Hill estimator for threshold
    println("\n" * "─" ^ 40)
    println("2. Peak-over-Threshold Method (GPD)")
    println("─" ^ 40)

    k_vals, hill = hill_estimator(returns; loss=true)
    k_star, ξ_hill = optimal_hill_k(k_vals, hill)
    println("Hill estimate of ξ: $(round(ξ_hill, digits=4)) (at k*=$k_star)")

    u = quantile(-returns, 0.95)
    gpd = fit_gpd(returns, u; loss=true, verbose=true)
    rl_gpd = return_levels_gpd(gpd)

    # 3. Threshold selection
    u_vals, me_vals, ci_lo, ci_hi = mean_excess_function(returns; loss=true)

    # 4. Tail dependence
    println("\n" * "─" ^ 40)
    println("3. Bivariate Tail Dependence")
    println("─" ^ 40)

    # Two correlated t-distributed series
    Z = rand(MvNormal([0;0], [1 0.6; 0.6 1]), n)'
    X1 = quantile.(TDist(ν), cdf.(Normal(), Z[:,1])) .* 0.02
    X2 = quantile.(TDist(ν), cdf.(Normal(), Z[:,2])) .* 0.02
    tdep = tail_dependence(X1, X2; u=0.95)
    println("Upper tail dependence λᵤ: $(round(tdep.upper, digits=4))")
    println("Lower tail dependence λₗ: $(round(tdep.lower, digits=4))")

    W, H_wts, H_bins = spectral_measure(X1, X2)
    println("Spectral measure mean: $(round(mean(W), digits=4)) (0.5 = symmetric)")

    # 5. Multi-asset analysis
    println("\n" * "─" ^ 40)
    println("4. Portfolio Tail Risk")
    println("─" ^ 40)
    R_matrix = hcat(returns, X1, X2)
    weights = [0.4, 0.35, 0.25]
    gpd_port = portfolio_tail_risk(R_matrix, weights)

    # Plots
    p_hill = plot_hill(k_vals, hill; title="Hill Plot: ξ̂ vs k")
    p_me   = plot_mean_excess(u_vals, me_vals, ci_lo, ci_hi)
    p_gpd  = plot_gpd_diagnostics(gpd)
    p_rl   = plot_return_levels_gev(gev, maxima; blocks_per_year=12)

    diag_plot = plot(p_hill, p_me, p_rl, layout=(1,3), size=(1200, 350))
    savefig(diag_plot, "evt_diagnostics.png")
    println("\nSaved evt_diagnostics.png")
    savefig(p_gpd, "evt_gpd.png")
    println("Saved evt_gpd.png")

    return gev, gpd
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
