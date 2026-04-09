"""
    BacktestFarmAnalytics

Statistical analysis engine for processing thousands of backtest results.
Provides multiple testing corrections, alpha landscape analysis, parameter sensitivity,
regime-conditional analysis, optimal portfolio construction, validation methods,
report generation, and Gaussian process surrogate modeling.

Dependencies: LinearAlgebra, Statistics, Random (stdlib only).
"""
module BacktestFarmAnalytics

using LinearAlgebra
using Statistics
using Random

export BacktestConfig, BacktestMetrics, BacktestResult, BacktestResultSet,
       load_result_set, filter_results, sort_by_sharpe, summary_stats

export bonferroni_correction, holm_stepdown, benjamini_hochberg,
       deflated_sharpe_ratio, minimum_backtest_length, probability_of_backtest_overfitting

export kernel_density_estimate, pca_returns, kmeans_strategies,
       factor_decomposition, alpha_capacity_estimate

export sobol_first_order, sobol_total_order, morris_elementary_effects,
       local_sensitivity, parameter_interactions

export fit_hmm, detect_regimes, per_regime_sharpe, classify_specialist_generalist,
       regime_transition_cost

export markowitz_mvo, risk_parity_portfolio, max_diversification_portfolio,
       kelly_criterion, black_litterman

export walk_forward_consistency, ts_bootstrap_ci, permutation_test,
       combinatorial_purged_cv

export format_latex_table, summary_statistics_report, top_strategies_report,
       sensitivity_heatmap_data, regime_performance_table

export GaussianProcessModel, fit_gp!, predict_gp, rbf_kernel,
       expected_improvement, bayesian_optimization_step, active_learning_suggest

# ============================================================================
# Section 1: Core Data Structures
# ============================================================================

"""
    BacktestConfig{T<:Real}

Configuration parameters for a single backtest run.
Stores parameter names and their values as a dictionary-like structure.
"""
struct BacktestConfig{T<:Real}
    name::String
    params::Dict{String, T}
    metadata::Dict{String, String}

    function BacktestConfig{T}(name::String, params::Dict{String, T},
                                metadata::Dict{String, String}) where T<:Real
        new{T}(name, params, metadata)
    end
end

BacktestConfig(name::String, params::Dict{String, T},
               metadata::Dict{String, String}=Dict{String,String}()) where T<:Real =
    BacktestConfig{T}(name, params, metadata)

BacktestConfig(name::String, pairs::Pair{String,T}...) where T<:Real =
    BacktestConfig(name, Dict(pairs...), Dict{String,String}())

function Base.show(io::IO, c::BacktestConfig)
    print(io, "BacktestConfig(\"", c.name, "\", ", length(c.params), " params)")
end

"""
    BacktestMetrics{T<:Real}

Performance metrics from a single backtest run.
"""
struct BacktestMetrics{T<:Real}
    sharpe::T
    sortino::T
    calmar::T
    max_drawdown::T
    total_return::T
    annual_return::T
    annual_volatility::T
    skewness::T
    kurtosis::T
    var_95::T
    cvar_95::T
    win_rate::T
    profit_factor::T
    num_trades::Int
    returns::Vector{T}

    function BacktestMetrics{T}(; sharpe::T=zero(T), sortino::T=zero(T),
                                  calmar::T=zero(T), max_drawdown::T=zero(T),
                                  total_return::T=zero(T), annual_return::T=zero(T),
                                  annual_volatility::T=zero(T), skewness::T=zero(T),
                                  kurtosis::T=zero(T), var_95::T=zero(T),
                                  cvar_95::T=zero(T), win_rate::T=zero(T),
                                  profit_factor::T=zero(T), num_trades::Int=0,
                                  returns::Vector{T}=T[]) where T<:Real
        new{T}(sharpe, sortino, calmar, max_drawdown, total_return, annual_return,
               annual_volatility, skewness, kurtosis, var_95, cvar_95, win_rate,
               profit_factor, num_trades, returns)
    end
end

BacktestMetrics(; kwargs...) = BacktestMetrics{Float64}(; kwargs...)

function Base.show(io::IO, m::BacktestMetrics)
    print(io, "BacktestMetrics(sharpe=", round(m.sharpe, digits=3),
          ", ret=", round(m.total_return * 100, digits=2), "%)")
end

"""
    BacktestResult{T<:Real}

A single backtest result pairing configuration with metrics.
"""
struct BacktestResult{T<:Real}
    config::BacktestConfig{T}
    metrics::BacktestMetrics{T}
    id::Int
    timestamp::Float64
end

BacktestResult(config::BacktestConfig{T}, metrics::BacktestMetrics{T};
               id::Int=0, timestamp::Float64=0.0) where T =
    BacktestResult{T}(config, metrics, id, timestamp)

function Base.show(io::IO, r::BacktestResult)
    print(io, "BacktestResult(\"", r.config.name, "\", sharpe=",
          round(r.metrics.sharpe, digits=3), ")")
end

"""
    BacktestResultSet{T<:Real}

Collection of backtest results with indexing and iteration support.
"""
struct BacktestResultSet{T<:Real}
    results::Vector{BacktestResult{T}}
    param_names::Vector{String}
    metric_names::Vector{String}

    function BacktestResultSet{T}(results::Vector{BacktestResult{T}},
                                   param_names::Vector{String},
                                   metric_names::Vector{String}) where T<:Real
        new{T}(results, param_names, metric_names)
    end
end

function BacktestResultSet(results::Vector{BacktestResult{T}}) where T<:Real
    pnames = if isempty(results)
        String[]
    else
        collect(keys(results[1].config.params))
    end
    mnames = ["sharpe", "sortino", "calmar", "max_drawdown", "total_return",
              "annual_return", "annual_volatility", "skewness", "kurtosis",
              "var_95", "cvar_95", "win_rate", "profit_factor"]
    BacktestResultSet{T}(results, pnames, mnames)
end

Base.length(rs::BacktestResultSet) = length(rs.results)
Base.getindex(rs::BacktestResultSet, i::Int) = rs.results[i]
Base.getindex(rs::BacktestResultSet, r::UnitRange) = BacktestResultSet(rs.results[r])
Base.iterate(rs::BacktestResultSet) = iterate(rs.results)
Base.iterate(rs::BacktestResultSet, state) = iterate(rs.results, state)
Base.eltype(::Type{BacktestResultSet{T}}) where T = BacktestResult{T}
Base.firstindex(rs::BacktestResultSet) = 1
Base.lastindex(rs::BacktestResultSet) = length(rs.results)

function Base.show(io::IO, rs::BacktestResultSet)
    n = length(rs.results)
    print(io, "BacktestResultSet(", n, " results, ",
          length(rs.param_names), " params)")
end

"""
    load_result_set(data::Vector{Dict{String,Any}}) -> BacktestResultSet

Load a BacktestResultSet from a JSON-like vector of dictionaries.
Each dict should have "config" and "metrics" keys.
"""
function load_result_set(data::Vector{Dict{String, Any}})
    results = BacktestResult{Float64}[]
    for (i, d) in enumerate(data)
        cfg_data = get(d, "config", Dict{String,Any}())
        met_data = get(d, "metrics", Dict{String,Any}())

        name = get(cfg_data, "name", "strategy_$i")
        params = Dict{String, Float64}()
        for (k, v) in get(cfg_data, "params", Dict{String,Any}())
            params[k] = Float64(v)
        end
        metadata = Dict{String, String}()
        for (k, v) in get(cfg_data, "metadata", Dict{String,Any}())
            metadata[k] = string(v)
        end

        config = BacktestConfig(name, params, metadata)

        rets_raw = get(met_data, "returns", Float64[])
        rets = Float64.(rets_raw)

        metrics = BacktestMetrics{Float64}(
            sharpe        = Float64(get(met_data, "sharpe", 0.0)),
            sortino       = Float64(get(met_data, "sortino", 0.0)),
            calmar        = Float64(get(met_data, "calmar", 0.0)),
            max_drawdown  = Float64(get(met_data, "max_drawdown", 0.0)),
            total_return  = Float64(get(met_data, "total_return", 0.0)),
            annual_return = Float64(get(met_data, "annual_return", 0.0)),
            annual_volatility = Float64(get(met_data, "annual_volatility", 0.0)),
            skewness      = Float64(get(met_data, "skewness", 0.0)),
            kurtosis      = Float64(get(met_data, "kurtosis", 0.0)),
            var_95        = Float64(get(met_data, "var_95", 0.0)),
            cvar_95       = Float64(get(met_data, "cvar_95", 0.0)),
            win_rate      = Float64(get(met_data, "win_rate", 0.0)),
            profit_factor = Float64(get(met_data, "profit_factor", 0.0)),
            num_trades    = Int(get(met_data, "num_trades", 0)),
            returns       = rets
        )

        push!(results, BacktestResult(config, metrics; id=i, timestamp=Float64(get(d, "timestamp", 0.0))))
    end
    BacktestResultSet(results)
end

"""
    filter_results(rs::BacktestResultSet, pred::Function) -> BacktestResultSet

Filter results by a predicate on BacktestResult.
"""
function filter_results(rs::BacktestResultSet{T}, pred::Function) where T
    filtered = filter(pred, rs.results)
    BacktestResultSet(collect(filtered))
end

"""
    sort_by_sharpe(rs::BacktestResultSet; descending=true) -> BacktestResultSet

Sort results by Sharpe ratio.
"""
function sort_by_sharpe(rs::BacktestResultSet{T}; descending::Bool=true) where T
    sorted = sort(rs.results, by=r -> r.metrics.sharpe, rev=descending)
    BacktestResultSet(sorted)
end

"""
    extract_sharpes(rs::BacktestResultSet) -> Vector

Extract Sharpe ratios from all results.
"""
function extract_sharpes(rs::BacktestResultSet{T}) where T
    T[r.metrics.sharpe for r in rs.results]
end

"""
    extract_param_matrix(rs::BacktestResultSet) -> Matrix, Vector{String}

Extract parameter values as a matrix (rows=strategies, cols=params).
"""
function extract_param_matrix(rs::BacktestResultSet{T}) where T
    pnames = rs.param_names
    n = length(rs.results)
    p = length(pnames)
    X = zeros(T, n, p)
    for (i, r) in enumerate(rs.results)
        for (j, pn) in enumerate(pnames)
            X[i, j] = get(r.config.params, pn, zero(T))
        end
    end
    X, pnames
end

"""
    extract_return_matrix(rs::BacktestResultSet) -> Matrix

Extract returns as matrix (rows=time, cols=strategies). Pads shorter series with zeros.
"""
function extract_return_matrix(rs::BacktestResultSet{T}) where T
    max_len = maximum(length(r.metrics.returns) for r in rs.results; init=0)
    n = length(rs.results)
    R = zeros(T, max_len, n)
    for (j, r) in enumerate(rs.results)
        rets = r.metrics.returns
        R[1:length(rets), j] .= rets
    end
    R
end

"""
    compute_sharpe(returns::AbstractVector{T}; rf::T=zero(T), periods::Int=252) where T

Compute annualized Sharpe ratio from a return series.
"""
function compute_sharpe(returns::AbstractVector{T}; rf::T=zero(T), periods::Int=252) where T<:Real
    n = length(returns)
    n < 2 && return zero(T)
    mu = mean(returns) - rf / periods
    sigma = std(returns; corrected=true)
    sigma < eps(T) && return zero(T)
    mu / sigma * sqrt(T(periods))
end

"""
    compute_sortino(returns::AbstractVector{T}; rf::T=zero(T), periods::Int=252) where T

Compute annualized Sortino ratio.
"""
function compute_sortino(returns::AbstractVector{T}; rf::T=zero(T), periods::Int=252) where T<:Real
    n = length(returns)
    n < 2 && return zero(T)
    mu = mean(returns) - rf / periods
    downside = returns[returns .< zero(T)]
    isempty(downside) && return T(Inf)
    dd = sqrt(mean(downside .^ 2))
    dd < eps(T) && return zero(T)
    mu / dd * sqrt(T(periods))
end

"""
    compute_max_drawdown(returns::AbstractVector{T}) where T

Compute maximum drawdown from a return series.
"""
function compute_max_drawdown(returns::AbstractVector{T}) where T<:Real
    n = length(returns)
    n == 0 && return zero(T)
    cumret = cumprod(one(T) .+ returns)
    peak = accumulate(max, cumret)
    dd = (peak .- cumret) ./ peak
    maximum(dd)
end

"""
    compute_skewness(x::AbstractVector{T}) where T

Compute sample skewness.
"""
function compute_skewness(x::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 3 && return zero(T)
    mu = mean(x)
    s = std(x; corrected=true)
    s < eps(T) && return zero(T)
    m3 = mean((x .- mu) .^ 3)
    m3 / s^3
end

"""
    compute_kurtosis(x::AbstractVector{T}) where T

Compute excess kurtosis.
"""
function compute_kurtosis(x::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 4 && return zero(T)
    mu = mean(x)
    s = std(x; corrected=true)
    s < eps(T) && return zero(T)
    m4 = mean((x .- mu) .^ 4)
    m4 / s^4 - T(3)
end

"""
    summary_stats(rs::BacktestResultSet) -> Dict

Compute summary statistics of Sharpe distribution across all results.
"""
function summary_stats(rs::BacktestResultSet{T}) where T
    sharpes = extract_sharpes(rs)
    n = length(sharpes)
    n == 0 && return Dict{String, T}()
    sorted = sort(sharpes)
    Dict{String, T}(
        "count"    => T(n),
        "mean"     => mean(sharpes),
        "median"   => sorted[div(n+1, 2)],
        "std"      => std(sharpes; corrected=true),
        "min"      => minimum(sharpes),
        "max"      => maximum(sharpes),
        "q25"      => sorted[max(1, div(n, 4))],
        "q75"      => sorted[max(1, div(3*n, 4))],
        "skewness" => compute_skewness(sharpes),
        "kurtosis" => compute_kurtosis(sharpes),
    )
end

# ============================================================================
# Section 2: Multiple Testing Corrections
# ============================================================================

"""
    MultipleTesting

Module for multiple hypothesis testing corrections applied to backtest results.
Implements Bonferroni, Holm, Benjamini-Hochberg, Deflated Sharpe Ratio,
Minimum Backtest Length, and Probability of Backtest Overfitting.
"""
module MultipleTesting

using Statistics
using LinearAlgebra
using Random

"""
    bonferroni_correction(pvalues::AbstractVector{T}; alpha::T=T(0.05)) where T

Apply Bonferroni correction for family-wise error rate.
Returns adjusted p-values and a boolean vector of which tests remain significant.

The Bonferroni correction multiplies each p-value by the number of tests,
providing strong FWER control at the cost of reduced power.
"""
function bonferroni_correction(pvalues::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    m = length(pvalues)
    m == 0 && return (adjusted=T[], significant=Bool[], threshold=alpha)

    adjusted = clamp.(pvalues .* m, zero(T), one(T))
    significant = adjusted .< alpha
    (adjusted=adjusted, significant=significant, threshold=alpha / m)
end

"""
    holm_stepdown(pvalues::AbstractVector{T}; alpha::T=T(0.05)) where T

Apply Holm step-down procedure for FWER control.
More powerful than Bonferroni while maintaining strong FWER control.

Procedure:
1. Sort p-values in ascending order.
2. For the k-th smallest p-value, compare against alpha/(m-k+1).
3. Reject hypotheses until the first non-rejection.
"""
function holm_stepdown(pvalues::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    m = length(pvalues)
    m == 0 && return (adjusted=T[], significant=Bool[], rejected_count=0)

    order = sortperm(pvalues)
    sorted_p = pvalues[order]

    adjusted = similar(pvalues)
    significant = falses(m)

    # Compute adjusted p-values (enforcing monotonicity)
    running_max = zero(T)
    for k in 1:m
        adj_p = sorted_p[k] * (m - k + 1)
        running_max = max(running_max, adj_p)
        adjusted[order[k]] = clamp(running_max, zero(T), one(T))
    end

    significant .= adjusted .< alpha
    rejected_count = sum(significant)

    (adjusted=adjusted, significant=significant, rejected_count=rejected_count)
end

"""
    benjamini_hochberg(pvalues::AbstractVector{T}; alpha::T=T(0.05)) where T

Apply Benjamini-Hochberg procedure for False Discovery Rate (FDR) control.
Controls the expected proportion of false positives among rejected hypotheses.

Procedure:
1. Sort p-values in ascending order.
2. Find the largest k such that p_(k) <= k*alpha/m.
3. Reject all hypotheses with rank <= k.
"""
function benjamini_hochberg(pvalues::AbstractVector{T}; alpha::T=T(0.05)) where T<:Real
    m = length(pvalues)
    m == 0 && return (adjusted=T[], significant=Bool[], fdr_threshold=alpha, discoveries=0)

    order = sortperm(pvalues)
    sorted_p = pvalues[order]

    adjusted = similar(pvalues)

    # Compute adjusted p-values (step-up, enforcing monotonicity from the right)
    running_min = one(T)
    for k in m:-1:1
        adj_p = sorted_p[k] * m / k
        running_min = min(running_min, adj_p)
        adjusted[order[k]] = clamp(running_min, zero(T), one(T))
    end

    significant = adjusted .< alpha
    discoveries = sum(significant)

    # Find the BH threshold
    bh_threshold = zero(T)
    for k in m:-1:1
        if sorted_p[k] <= k * alpha / m
            bh_threshold = sorted_p[k]
            break
        end
    end

    (adjusted=adjusted, significant=significant, fdr_threshold=bh_threshold, discoveries=discoveries)
end

"""
    _standard_normal_cdf(x::T) where T<:Real

Approximate the standard normal CDF using the rational approximation.
Abramowitz and Stegun formula 26.2.17, max error ~7.5e-8.
"""
function _standard_normal_cdf(x::T) where T<:Real
    # Constants for the approximation
    a1 = T(0.254829592)
    a2 = T(-0.284496736)
    a3 = T(1.421413741)
    a4 = T(-1.453152027)
    a5 = T(1.061405429)
    p  = T(0.3275911)

    sign_x = x < zero(T) ? -one(T) : one(T)
    x_abs = abs(x)
    t = one(T) / (one(T) + p * x_abs)
    y = one(T) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x_abs^2 / 2)

    T(0.5) * (one(T) + sign_x * y)
end

"""
    _standard_normal_pdf(x::T) where T<:Real

Standard normal PDF.
"""
function _standard_normal_pdf(x::T) where T<:Real
    T(1.0 / sqrt(2.0 * pi)) * exp(-x^2 / 2)
end

"""
    _standard_normal_quantile(p::T) where T<:Real

Approximate inverse normal CDF (quantile function) using rational approximation.
Beasley-Springer-Moro algorithm.
"""
function _standard_normal_quantile(p::T) where T<:Real
    p <= zero(T) && return T(-Inf)
    p >= one(T) && return T(Inf)
    abs(p - T(0.5)) < eps(T) && return zero(T)

    # Rational approximation for central region
    a = [T(-3.969683028665376e+01), T(2.209460984245205e+02),
         T(-2.759285104469687e+02), T(1.383577518672690e+02),
         T(-3.066479806614716e+01), T(2.506628277459239e+00)]
    b = [T(-5.447609879822406e+01), T(1.615858368580409e+02),
         T(-1.556989798598866e+02), T(6.680131188771972e+01),
         T(-1.328068155288572e+01)]
    c = [T(-7.784894002430293e-03), T(-3.223964580411365e-01),
         T(-2.400758277161838e+00), T(-2.549732539343734e+00),
         T(4.374664141464968e+00), T(2.938163982698783e+00)]
    d = [T(7.784695709041462e-03), T(3.224671290700398e-01),
         T(2.445134137142996e+00), T(3.754408661907416e+00)]

    p_low  = T(0.02425)
    p_high = one(T) - p_low

    if p < p_low
        q = sqrt(-T(2) * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+one(T))
    elseif p <= p_high
        q = p - T(0.5)
        r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6]) * q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+one(T))
    else
        q = sqrt(-T(2) * log(one(T) - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+one(T)))
    end
end

"""
    sharpe_to_pvalue(sr::T, n::Int; sr0::T=zero(T)) where T

Convert a Sharpe ratio to a p-value under the null hypothesis that true SR = sr0.
Uses the standard error of the Sharpe ratio estimator.
"""
function sharpe_to_pvalue(sr::T, n::Int; sr0::T=zero(T)) where T<:Real
    n < 2 && return one(T)
    se = sqrt((one(T) + sr^2 / T(4)) / T(n))  # Approximate SE of SR
    z = (sr - sr0) / se
    one(T) - _standard_normal_cdf(z)
end

"""
    deflated_sharpe_ratio(observed_sr::T, n::Int, num_trials::Int;
                          sr_std::T=one(T), skew::T=zero(T), kurt::T=zero(T)) where T

Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Adjusts the observed Sharpe ratio for the number of strategies tested,
accounting for higher moments of the return distribution.

Arguments:
- `observed_sr`: The Sharpe ratio of the best (selected) strategy
- `n`: Number of return observations
- `num_trials`: Number of strategies tested (selection bias correction)
- `sr_std`: Standard deviation of Sharpe ratios across all trials
- `skew`: Skewness of the return series
- `kurt`: Excess kurtosis of the return series

Returns named tuple with deflated SR, p-value, and expected maximum SR.
"""
function deflated_sharpe_ratio(observed_sr::T, n::Int, num_trials::Int;
                                sr_std::T=one(T), skew::T=zero(T),
                                kurt::T=zero(T)) where T<:Real
    num_trials < 1 && error("num_trials must be >= 1")
    n < 2 && return (dsr=zero(T), pvalue=one(T), expected_max_sr=zero(T))

    # Expected maximum Sharpe ratio under the null (i.i.d. trials)
    # E[max(Z_1,...,Z_N)] approximation using Euler-Mascheroni constant
    euler_mascheroni = T(0.5772156649015329)
    v = T(num_trials)

    # Expected max of v i.i.d. standard normals (approximation)
    if v > one(T)
        expected_max_z = (one(T) - euler_mascheroni) * _standard_normal_quantile(one(T) - one(T)/v) +
                         euler_mascheroni * _standard_normal_quantile(one(T) - one(T)/(v * T(exp(1))))
    else
        expected_max_z = zero(T)
    end

    expected_max_sr = expected_max_z * sr_std

    # Standard error of the Sharpe ratio with skewness/kurtosis correction
    se_sr = sqrt((one(T) - skew * observed_sr + (kurt - one(T)) / T(4) * observed_sr^2) / T(n))
    se_sr = max(se_sr, eps(T))

    # Deflated test statistic
    z_deflated = (observed_sr - expected_max_sr) / se_sr

    # P-value (one-sided test)
    pvalue = one(T) - _standard_normal_cdf(z_deflated)

    (dsr=z_deflated, pvalue=pvalue, expected_max_sr=expected_max_sr,
     se_sr=se_sr, significant_5pct=pvalue < T(0.05))
end

"""
    deflated_sharpe_ratio(rs::BacktestResultSet) -> NamedTuple

Convenience method: compute DSR for the best strategy in the result set.
"""
function deflated_sharpe_ratio(rs)
    results = rs.results
    isempty(results) && error("Empty result set")

    sharpes = [r.metrics.sharpe for r in results]
    best_idx = argmax(sharpes)
    best = results[best_idx]

    rets = best.metrics.returns
    n = length(rets)
    sr = best.metrics.sharpe
    sr_std = std(sharpes; corrected=true)
    skew = compute_skewness_internal(rets)
    kurt = compute_kurtosis_internal(rets)
    num_trials = length(results)

    deflated_sharpe_ratio(sr, n, num_trials; sr_std=sr_std, skew=skew, kurt=kurt)
end

function compute_skewness_internal(x::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 3 && return zero(T)
    mu = mean(x)
    s = std(x; corrected=true)
    s < eps(T) && return zero(T)
    mean((x .- mu) .^ 3) / s^3
end

function compute_kurtosis_internal(x::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 4 && return zero(T)
    mu = mean(x)
    s = std(x; corrected=true)
    s < eps(T) && return zero(T)
    mean((x .- mu) .^ 4) / s^4 - T(3)
end

"""
    minimum_backtest_length(sr::T; sr0::T=zero(T), skew::T=zero(T),
                            kurt::T=zero(T), alpha::T=T(0.05)) where T

Compute the minimum number of observations needed for a backtest to provide
reliable inference about the Sharpe ratio (Bailey & Lopez de Prado).

Arguments:
- `sr`: Observed Sharpe ratio (annualized)
- `sr0`: Null hypothesis Sharpe ratio
- `skew`: Skewness of returns
- `kurt`: Excess kurtosis of returns
- `alpha`: Significance level

Returns the minimum number of daily observations required.
"""
function minimum_backtest_length(sr::T; sr0::T=zero(T), skew::T=zero(T),
                                  kurt::T=zero(T), alpha::T=T(0.05)) where T<:Real
    z_alpha = _standard_normal_quantile(one(T) - alpha)

    # De-annualize Sharpe (assume 252 trading days)
    sr_daily = sr / sqrt(T(252))
    sr0_daily = sr0 / sqrt(T(252))

    delta = sr_daily - sr0_daily
    abs(delta) < eps(T) && return typemax(Int)

    # Variance of the SR estimator with higher-moment corrections
    # Var(SR_hat) = (1 - skew*SR + (kurt-1)/4 * SR^2) / n
    # We need z_alpha^2 * Var(SR_hat) = delta^2
    # => n = z_alpha^2 * (1 - skew*sr + (kurt-1)/4 * sr^2) / delta^2

    var_factor = one(T) - skew * sr_daily + (kurt - one(T)) / T(4) * sr_daily^2
    var_factor = max(var_factor, eps(T))

    n_min = ceil(Int, z_alpha^2 * var_factor / delta^2)
    max(n_min, 10)
end

"""
    probability_of_backtest_overfitting(returns_matrix::AbstractMatrix{T},
                                        num_groups::Int=10;
                                        metric::Symbol=:sharpe) where T

Compute the Probability of Backtest Overfitting (PBO) using
Combinatorial Purged Cross-Validation (CPCV).

The method:
1. Partition the time series into S groups.
2. For each combination of S/2 groups (in-sample), the remaining are out-of-sample.
3. For each combination, find the best in-sample strategy.
4. Measure its rank in the OOS performance.
5. PBO = fraction of combinations where the IS-best ranks below median OOS.

Arguments:
- `returns_matrix`: T x N matrix (T time periods, N strategies)
- `num_groups`: Number of time partitions (S)
- `metric`: Performance metric to use (:sharpe, :total_return, :sortino)

Returns PBO probability and detailed combination results.
"""
function probability_of_backtest_overfitting(returns_matrix::AbstractMatrix{T},
                                              num_groups::Int=10;
                                              metric::Symbol=:sharpe,
                                              max_combinations::Int=1000,
                                              rng::AbstractRNG=Random.default_rng()) where T<:Real
    ntime, nstrats = size(returns_matrix)
    nstrats < 2 && error("Need at least 2 strategies")
    num_groups < 4 && error("Need at least 4 groups")
    num_groups = min(num_groups, ntime)

    # Partition time indices into groups
    group_size = ntime ÷ num_groups
    groups = Vector{UnitRange{Int}}(undef, num_groups)
    for g in 1:num_groups
        start_idx = (g - 1) * group_size + 1
        end_idx = g == num_groups ? ntime : g * group_size
        groups[g] = start_idx:end_idx
    end

    # Number of IS groups = S/2
    s_half = num_groups ÷ 2

    # Generate combinations (subset of all C(S, S/2) if too many)
    all_combos = _generate_combinations(num_groups, s_half)
    if length(all_combos) > max_combinations
        all_combos = all_combos[randperm(rng, length(all_combos))[1:max_combinations]]
    end

    # Evaluate metric function
    metric_fn = if metric == :sharpe
        rets -> _compute_sharpe_simple(rets)
    elseif metric == :total_return
        rets -> sum(rets)
    elseif metric == :sortino
        rets -> _compute_sortino_simple(rets)
    else
        error("Unknown metric: $metric")
    end

    num_overfit = 0
    oos_ranks = Int[]
    logit_values = T[]

    for combo in all_combos
        is_indices = Int[]
        oos_indices = Int[]
        oos_groups = setdiff(1:num_groups, combo)

        for g in combo
            append!(is_indices, collect(groups[g]))
        end
        for g in oos_groups
            append!(oos_indices, collect(groups[g]))
        end

        # Compute IS and OOS performance for each strategy
        is_scores = T[metric_fn(returns_matrix[is_indices, j]) for j in 1:nstrats]
        oos_scores = T[metric_fn(returns_matrix[oos_indices, j]) for j in 1:nstrats]

        # Best IS strategy
        best_is = argmax(is_scores)

        # Rank of IS-best in OOS (1 = worst, N = best)
        oos_rank = sum(oos_scores[best_is] .>= oos_scores)
        push!(oos_ranks, oos_rank)

        # Logit: log(rank / (N+1 - rank))
        adjusted_rank = clamp(oos_rank, 1, nstrats)
        denom = max(nstrats + 1 - adjusted_rank, 1)
        logit_val = log(T(adjusted_rank) / T(denom))
        push!(logit_values, logit_val)

        # Overfit if rank is below median
        if oos_rank <= nstrats ÷ 2
            num_overfit += 1
        end
    end

    pbo = T(num_overfit) / T(length(all_combos))

    (pbo=pbo,
     num_combinations=length(all_combos),
     oos_ranks=oos_ranks,
     logit_values=logit_values,
     mean_rank=mean(T.(oos_ranks)),
     median_rank=T(sort(oos_ranks)[max(1, length(oos_ranks)÷2)]),
     overfit_flag=pbo > T(0.5))
end

"""
    _generate_combinations(n::Int, k::Int) -> Vector{Vector{Int}}

Generate all combinations of k elements from 1:n.
"""
function _generate_combinations(n::Int, k::Int)
    combos = Vector{Int}[]
    k > n && return combos
    combo = collect(1:k)
    while true
        push!(combos, copy(combo))
        # Find rightmost element that can be incremented
        i = k
        while i > 0 && combo[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        combo[i] += 1
        for j in (i+1):k
            combo[j] = combo[j-1] + 1
        end
    end
    combos
end

function _compute_sharpe_simple(rets::AbstractVector{T}) where T<:Real
    n = length(rets)
    n < 2 && return zero(T)
    mu = mean(rets)
    sigma = std(rets; corrected=true)
    sigma < eps(T) && return zero(T)
    mu / sigma * sqrt(T(252))
end

function _compute_sortino_simple(rets::AbstractVector{T}) where T<:Real
    n = length(rets)
    n < 2 && return zero(T)
    mu = mean(rets)
    downside = rets[rets .< zero(T)]
    isempty(downside) && return T(Inf)
    dd = sqrt(mean(downside .^ 2))
    dd < eps(T) && return zero(T)
    mu / dd * sqrt(T(252))
end

"""
    multiple_testing_analysis(rs; alpha=0.05) -> Dict

Run all multiple testing corrections on a BacktestResultSet.
Converts Sharpe ratios to p-values and applies Bonferroni, Holm, and BH corrections.
"""
function multiple_testing_analysis(rs; alpha::Float64=0.05)
    results = rs.results
    n_strategies = length(results)

    # Convert Sharpe ratios to p-values
    pvalues = Float64[sharpe_to_pvalue(Float64(r.metrics.sharpe),
                                        length(r.metrics.returns))
                       for r in results]

    bonf = bonferroni_correction(pvalues; alpha=alpha)
    holm = holm_stepdown(pvalues; alpha=alpha)
    bh = benjamini_hochberg(pvalues; alpha=alpha)

    Dict(
        "pvalues" => pvalues,
        "bonferroni" => bonf,
        "holm" => holm,
        "benjamini_hochberg" => bh,
        "n_strategies" => n_strategies,
        "n_bonferroni_significant" => sum(bonf.significant),
        "n_holm_significant" => holm.rejected_count,
        "n_bh_discoveries" => bh.discoveries,
    )
end

end # module MultipleTesting

# ============================================================================
# Section 3: Alpha Landscape Analysis
# ============================================================================

"""
    AlphaLandscapeAnalysis

Tools for understanding the distribution and structure of alpha across
a large set of backtested strategies.
"""
module AlphaLandscapeAnalysis

using Statistics
using LinearAlgebra
using Random

"""
    kernel_density_estimate(data::AbstractVector{T}; bandwidth::T=T(-1),
                            n_points::Int=200, kernel::Symbol=:gaussian) where T

Compute kernel density estimate of a distribution.

Arguments:
- `data`: Vector of observations (e.g., Sharpe ratios)
- `bandwidth`: Bandwidth parameter (negative = auto via Silverman's rule)
- `n_points`: Number of evaluation points
- `kernel`: Kernel function (:gaussian, :epanechnikov, :uniform)

Returns (x_grid, density) tuple.
"""
function kernel_density_estimate(data::AbstractVector{T}; bandwidth::T=T(-1),
                                  n_points::Int=200, kernel::Symbol=:gaussian) where T<:Real
    n = length(data)
    n == 0 && return (zeros(T, 0), zeros(T, 0))

    # Silverman's rule of thumb
    if bandwidth < zero(T)
        sigma = std(data; corrected=true)
        iqr_est = _approximate_iqr(data)
        bandwidth = T(0.9) * min(sigma, iqr_est / T(1.34)) * T(n)^(T(-1)/T(5))
        bandwidth = max(bandwidth, eps(T))
    end

    data_min = minimum(data) - T(3) * bandwidth
    data_max = maximum(data) + T(3) * bandwidth
    x_grid = range(data_min, data_max, length=n_points)
    density = zeros(T, n_points)

    kernel_fn = if kernel == :gaussian
        (u::T) -> exp(-u^2 / 2) / sqrt(T(2) * T(pi))
    elseif kernel == :epanechnikov
        (u::T) -> abs(u) <= one(T) ? T(0.75) * (one(T) - u^2) : zero(T)
    elseif kernel == :uniform
        (u::T) -> abs(u) <= one(T) ? T(0.5) : zero(T)
    else
        error("Unknown kernel: $kernel")
    end

    for (i, x) in enumerate(x_grid)
        for d in data
            u = (x - d) / bandwidth
            density[i] += kernel_fn(u)
        end
        density[i] /= (n * bandwidth)
    end

    (collect(x_grid), density)
end

function _approximate_iqr(data::AbstractVector{T}) where T<:Real
    sorted = sort(data)
    n = length(sorted)
    q25 = sorted[max(1, div(n, 4))]
    q75 = sorted[max(1, div(3*n, 4))]
    q75 - q25
end

"""
    pca_returns(return_matrix::AbstractMatrix{T}; n_components::Int=5) where T

Perform Principal Component Analysis on the strategy return correlation matrix.

Arguments:
- `return_matrix`: T x N matrix (T time periods, N strategies)
- `n_components`: Number of principal components to retain

Returns eigenvalues, eigenvectors, explained variance ratios, and loadings.
"""
function pca_returns(return_matrix::AbstractMatrix{T}; n_components::Int=5) where T<:Real
    ntime, nstrats = size(return_matrix)
    nstrats < 2 && error("Need at least 2 strategies for PCA")
    n_components = min(n_components, nstrats)

    # Standardize returns
    means = vec(mean(return_matrix, dims=1))
    stds = vec(std(return_matrix, dims=1; corrected=true))
    stds[stds .< eps(T)] .= one(T)  # avoid division by zero

    standardized = (return_matrix .- means') ./ stds'

    # Correlation matrix
    corr_mat = (standardized' * standardized) / T(ntime - 1)

    # Eigendecomposition (symmetric)
    eigen_result = eigen(Symmetric(corr_mat))
    eigenvalues = reverse(eigen_result.values)
    eigenvectors = eigen_result.vectors[:, end:-1:1]

    total_var = sum(eigenvalues)
    explained_ratio = eigenvalues ./ total_var
    cumulative_ratio = cumsum(explained_ratio)

    # Loadings (eigenvectors scaled by sqrt of eigenvalues)
    loadings = eigenvectors[:, 1:n_components] .* sqrt.(eigenvalues[1:n_components])'

    (eigenvalues=eigenvalues[1:n_components],
     eigenvectors=eigenvectors[:, 1:n_components],
     explained_variance_ratio=explained_ratio[1:n_components],
     cumulative_variance_ratio=cumulative_ratio[1:n_components],
     loadings=loadings,
     correlation_matrix=corr_mat,
     n_components_for_90pct=findfirst(cumulative_ratio .>= T(0.9)))
end

"""
    kmeans_strategies(return_matrix::AbstractMatrix{T}, k::Int;
                      max_iter::Int=100, n_init::Int=10,
                      rng::AbstractRNG=Random.default_rng()) where T

Cluster strategies by return correlation using k-means.

Uses correlation distance (1 - correlation) as the distance metric.
Runs multiple initializations and returns the best clustering.

Returns cluster assignments, centroids, inertia, and silhouette scores.
"""
function kmeans_strategies(return_matrix::AbstractMatrix{T}, k::Int;
                           max_iter::Int=100, n_init::Int=10,
                           rng::AbstractRNG=Random.default_rng()) where T<:Real
    ntime, nstrats = size(return_matrix)
    k = min(k, nstrats)
    k < 1 && error("k must be >= 1")

    # Compute correlation matrix
    corr_mat = cor(return_matrix)

    # Distance matrix: 1 - |correlation|
    dist_mat = one(T) .- abs.(corr_mat)
    for i in 1:nstrats
        dist_mat[i, i] = zero(T)
    end

    best_labels = zeros(Int, nstrats)
    best_inertia = T(Inf)
    best_centroids = zeros(T, k, nstrats)

    for init in 1:n_init
        # K-means++ initialization on distance matrix
        centroids_idx = _kmeans_pp_init(dist_mat, k, rng)
        labels = zeros(Int, nstrats)

        for iter in 1:max_iter
            # Assignment step
            old_labels = copy(labels)
            for i in 1:nstrats
                min_dist = T(Inf)
                for c in 1:k
                    d = dist_mat[i, centroids_idx[c]]
                    if d < min_dist
                        min_dist = d
                        labels[i] = c
                    end
                end
            end

            # Check convergence
            labels == old_labels && break

            # Update step: find medoid of each cluster
            for c in 1:k
                members = findall(labels .== c)
                isempty(members) && continue
                # Find member with smallest total distance to other members
                min_total = T(Inf)
                best_member = members[1]
                for m in members
                    total = sum(dist_mat[m, m2] for m2 in members)
                    if total < min_total
                        min_total = total
                        best_member = m
                    end
                end
                centroids_idx[c] = best_member
            end
        end

        # Compute inertia
        inertia = zero(T)
        for i in 1:nstrats
            inertia += dist_mat[i, centroids_idx[labels[i]]]^2
        end

        if inertia < best_inertia
            best_inertia = inertia
            best_labels .= labels
            for c in 1:k
                best_centroids[c, :] .= corr_mat[centroids_idx[c], :]
            end
        end
    end

    # Compute silhouette scores
    silhouettes = _compute_silhouette(dist_mat, best_labels, k)

    # Cluster sizes
    cluster_sizes = [sum(best_labels .== c) for c in 1:k]

    (labels=best_labels,
     centroids=best_centroids,
     inertia=best_inertia,
     silhouette_scores=silhouettes,
     mean_silhouette=mean(silhouettes),
     cluster_sizes=cluster_sizes,
     correlation_matrix=corr_mat,
     distance_matrix=dist_mat)
end

function _kmeans_pp_init(dist_mat::AbstractMatrix{T}, k::Int, rng::AbstractRNG) where T<:Real
    n = size(dist_mat, 1)
    centroids = Int[]

    # First centroid: random
    push!(centroids, rand(rng, 1:n))

    for _ in 2:k
        # Distance from each point to nearest existing centroid
        min_dists = T[minimum(dist_mat[i, c] for c in centroids) for i in 1:n]
        min_dists .^= 2
        total = sum(min_dists)
        total < eps(T) && (push!(centroids, rand(rng, 1:n)); continue)

        # Weighted random selection
        r = rand(rng) * total
        cumulative = zero(T)
        selected = n
        for i in 1:n
            cumulative += min_dists[i]
            if cumulative >= r
                selected = i
                break
            end
        end
        push!(centroids, selected)
    end
    centroids
end

function _compute_silhouette(dist_mat::AbstractMatrix{T}, labels::Vector{Int}, k::Int) where T<:Real
    n = size(dist_mat, 1)
    silhouettes = zeros(T, n)

    for i in 1:n
        ci = labels[i]
        # a(i) = mean distance to same-cluster members
        same = findall(labels .== ci)
        if length(same) > 1
            a_i = sum(dist_mat[i, j] for j in same if j != i) / (length(same) - 1)
        else
            a_i = zero(T)
        end

        # b(i) = min mean distance to any other cluster
        b_i = T(Inf)
        for c in 1:k
            c == ci && continue
            others = findall(labels .== c)
            isempty(others) && continue
            mean_dist = mean(dist_mat[i, j] for j in others)
            b_i = min(b_i, mean_dist)
        end

        denom = max(a_i, b_i)
        silhouettes[i] = denom > eps(T) ? (b_i - a_i) / denom : zero(T)
    end
    silhouettes
end

"""
    factor_decomposition(return_matrix::AbstractMatrix{T}; n_factors::Int=3) where T

Decompose strategy returns into common factors using PCA-based factor model.

Model: R = B * F + epsilon
where R is returns, B is factor loadings, F is factor returns.

Returns factor returns, loadings, residuals, and R-squared for each strategy.
"""
function factor_decomposition(return_matrix::AbstractMatrix{T}; n_factors::Int=3) where T<:Real
    ntime, nstrats = size(return_matrix)
    n_factors = min(n_factors, nstrats, ntime)

    # Demean
    means = vec(mean(return_matrix, dims=1))
    demeaned = return_matrix .- means'

    # SVD for factor extraction
    U, S, Vt = svd(demeaned)

    # Factor returns (T x K)
    factor_returns = U[:, 1:n_factors] .* S[1:n_factors]'

    # Factor loadings (N x K): regression of each strategy on factors
    # B = (F'F)^(-1) F'R
    FtF = factor_returns' * factor_returns
    FtR = factor_returns' * demeaned
    loadings = (FtF \ FtR)'  # N x K

    # Residuals
    fitted = factor_returns * loadings'
    residuals = demeaned - fitted

    # R-squared for each strategy
    r_squared = zeros(T, nstrats)
    for j in 1:nstrats
        ss_total = sum(demeaned[:, j] .^ 2)
        ss_resid = sum(residuals[:, j] .^ 2)
        r_squared[j] = ss_total > eps(T) ? one(T) - ss_resid / ss_total : zero(T)
    end

    # Variance explained by each factor
    total_var = sum(S .^ 2)
    factor_var_explained = S[1:n_factors] .^ 2 ./ total_var

    (factor_returns=factor_returns,
     loadings=loadings,
     residuals=residuals,
     r_squared=r_squared,
     factor_variance_explained=factor_var_explained,
     mean_r_squared=mean(r_squared),
     specific_variance=vec(var(residuals, dims=1)))
end

"""
    alpha_capacity_estimate(returns::AbstractVector{T}, aum_levels::AbstractVector{T};
                            impact_coefficient::T=T(0.1),
                            market_volume::T=T(1e9)) where T

Estimate alpha capacity: at what AUM does alpha decay to zero?

Uses a simple market impact model where slippage is proportional to
(trade_size / market_volume)^0.5, reducing effective Sharpe.

Arguments:
- `returns`: Strategy return series at baseline AUM
- `aum_levels`: Vector of AUM levels to evaluate
- `impact_coefficient`: Market impact scaling factor
- `market_volume`: Average daily market volume

Returns AUM levels, effective Sharpe at each level, and estimated capacity.
"""
function alpha_capacity_estimate(returns::AbstractVector{T}, aum_levels::AbstractVector{T};
                                  impact_coefficient::T=T(0.1),
                                  market_volume::T=T(1e9),
                                  turnover::T=T(1.0)) where T<:Real
    n = length(returns)
    n < 2 && error("Need at least 2 return observations")

    base_sharpe = _sharpe_simple(returns)
    base_mean = mean(returns)
    base_std = std(returns; corrected=true)

    effective_sharpes = similar(aum_levels)
    effective_returns = similar(aum_levels)
    impact_costs = similar(aum_levels)

    for (i, aum) in enumerate(aum_levels)
        # Daily trade volume = AUM * turnover
        daily_trade = aum * turnover

        # Market impact cost (square root model)
        participation_rate = daily_trade / market_volume
        impact = impact_coefficient * sqrt(max(participation_rate, zero(T)))
        impact_costs[i] = impact

        # Effective mean return = base return - impact cost
        eff_mean = base_mean - impact
        effective_returns[i] = eff_mean

        # Effective Sharpe
        effective_sharpes[i] = base_std > eps(T) ? eff_mean / base_std * sqrt(T(252)) : zero(T)
    end

    # Estimate capacity: AUM where Sharpe drops to 0
    capacity = zero(T)
    for i in 1:(length(aum_levels)-1)
        if effective_sharpes[i] > zero(T) && effective_sharpes[i+1] <= zero(T)
            # Linear interpolation
            frac = effective_sharpes[i] / (effective_sharpes[i] - effective_sharpes[i+1])
            capacity = aum_levels[i] + frac * (aum_levels[i+1] - aum_levels[i])
            break
        end
    end

    # Half-life: AUM where Sharpe drops to half
    half_sharpe = base_sharpe / T(2)
    half_life_aum = zero(T)
    for i in 1:(length(aum_levels)-1)
        if effective_sharpes[i] > half_sharpe && effective_sharpes[i+1] <= half_sharpe
            frac = (effective_sharpes[i] - half_sharpe) /
                   (effective_sharpes[i] - effective_sharpes[i+1])
            half_life_aum = aum_levels[i] + frac * (aum_levels[i+1] - aum_levels[i])
            break
        end
    end

    (aum_levels=aum_levels,
     effective_sharpes=effective_sharpes,
     effective_returns=effective_returns,
     impact_costs=impact_costs,
     base_sharpe=base_sharpe,
     estimated_capacity=capacity,
     half_life_aum=half_life_aum)
end

function _sharpe_simple(rets::AbstractVector{T}) where T<:Real
    n = length(rets)
    n < 2 && return zero(T)
    mu = mean(rets)
    sigma = std(rets; corrected=true)
    sigma < eps(T) && return zero(T)
    mu / sigma * sqrt(T(252))
end

"""
    elbow_criterion(return_matrix::AbstractMatrix{T}; max_k::Int=15,
                    rng::AbstractRNG=Random.default_rng()) where T

Determine optimal number of clusters using the elbow criterion.
Computes inertia for each k and finds the elbow point.
"""
function elbow_criterion(return_matrix::AbstractMatrix{T}; max_k::Int=15,
                          rng::AbstractRNG=Random.default_rng()) where T<:Real
    _, nstrats = size(return_matrix)
    max_k = min(max_k, nstrats)

    inertias = zeros(T, max_k)
    silhouettes = zeros(T, max_k)

    for k in 1:max_k
        result = kmeans_strategies(return_matrix, k; rng=rng, n_init=5)
        inertias[k] = result.inertia
        silhouettes[k] = k > 1 ? result.mean_silhouette : zero(T)
    end

    # Find elbow using second derivative
    best_k = 2
    if max_k >= 3
        max_second_deriv = T(-Inf)
        for k in 2:(max_k-1)
            second_deriv = inertias[k-1] - T(2) * inertias[k] + inertias[k+1]
            if second_deriv > max_second_deriv
                max_second_deriv = second_deriv
                best_k = k
            end
        end
    end

    (optimal_k=best_k, inertias=inertias, silhouettes=silhouettes)
end

end # module AlphaLandscapeAnalysis

# ============================================================================
# Section 4: Parameter Sensitivity Analysis
# ============================================================================

"""
    ParameterSensitivity

Methods for understanding how backtest performance depends on parameter choices.
Implements Sobol indices, Morris screening, local gradients, and interaction effects.
"""
module ParameterSensitivity

using Statistics
using LinearAlgebra
using Random

"""
    sobol_first_order(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                      n_bootstrap::Int=100,
                      rng::AbstractRNG=Random.default_rng()) where T

Estimate first-order Sobol sensitivity indices for each parameter.

Uses the Saltelli estimator: S_i = V[E[Y|X_i]] / V[Y]

The first-order index measures the fraction of output variance explained
by each parameter individually (main effect).

Arguments:
- `param_matrix`: N x p matrix of parameter values
- `responses`: N-vector of performance metric values (e.g., Sharpe ratios)
- `n_bootstrap`: Number of bootstrap samples for confidence intervals

Returns indices, confidence intervals, and ranking.
"""
function sobol_first_order(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                           n_bootstrap::Int=100,
                           rng::AbstractRNG=Random.default_rng()) where T<:Real
    n, p = size(param_matrix)
    n < 10 && error("Need at least 10 samples for Sobol analysis")

    total_var = var(responses; corrected=true)
    total_var < eps(T) && return (indices=zeros(T, p), ci_low=zeros(T, p),
                                  ci_high=zeros(T, p), ranking=collect(1:p))

    # Estimate conditional variance using binning approach
    indices = zeros(T, p)
    n_bins = max(5, isqrt(n))

    for j in 1:p
        # Sort by parameter j and bin
        order = sortperm(param_matrix[:, j])
        sorted_resp = responses[order]
        bin_size = max(1, n ÷ n_bins)

        conditional_means = T[]
        for b in 1:n_bins
            start_idx = (b-1) * bin_size + 1
            end_idx = min(b * bin_size, n)
            start_idx > n && break
            push!(conditional_means, mean(sorted_resp[start_idx:end_idx]))
        end

        # V[E[Y|X_j]] / V[Y]
        if length(conditional_means) > 1
            indices[j] = var(conditional_means; corrected=true) / total_var
        end
    end

    # Clamp to [0, 1]
    indices .= clamp.(indices, zero(T), one(T))

    # Bootstrap confidence intervals
    ci_low = zeros(T, p)
    ci_high = zeros(T, p)
    boot_indices = zeros(T, n_bootstrap, p)

    for b in 1:n_bootstrap
        boot_idx = rand(rng, 1:n, n)
        boot_params = param_matrix[boot_idx, :]
        boot_resp = responses[boot_idx]
        boot_var = var(boot_resp; corrected=true)
        boot_var < eps(T) && continue

        for j in 1:p
            order = sortperm(boot_params[:, j])
            sorted_resp = boot_resp[order]
            bin_size = max(1, n ÷ n_bins)

            cond_means = T[]
            for bb in 1:n_bins
                s = (bb-1) * bin_size + 1
                e = min(bb * bin_size, n)
                s > n && break
                push!(cond_means, mean(sorted_resp[s:e]))
            end

            if length(cond_means) > 1
                boot_indices[b, j] = clamp(var(cond_means; corrected=true) / boot_var,
                                           zero(T), one(T))
            end
        end
    end

    for j in 1:p
        sorted_boot = sort(boot_indices[:, j])
        ci_low[j] = sorted_boot[max(1, round(Int, 0.025 * n_bootstrap))]
        ci_high[j] = sorted_boot[max(1, round(Int, 0.975 * n_bootstrap))]
    end

    ranking = sortperm(indices, rev=true)

    (indices=indices, ci_low=ci_low, ci_high=ci_high, ranking=ranking,
     total_variance=total_var, sum_first_order=sum(indices))
end

"""
    sobol_total_order(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                      n_bootstrap::Int=100,
                      rng::AbstractRNG=Random.default_rng()) where T

Estimate total-order Sobol sensitivity indices.

The total index S_Ti = 1 - V[E[Y|X_~i]] / V[Y] captures the total effect
of parameter i including all interactions.

Arguments:
- `param_matrix`: N x p matrix of parameter values
- `responses`: N-vector of performance metric values

Returns total-order indices and interaction contributions.
"""
function sobol_total_order(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                           n_bootstrap::Int=100,
                           rng::AbstractRNG=Random.default_rng()) where T<:Real
    n, p = size(param_matrix)
    n < 10 && error("Need at least 10 samples")

    total_var = var(responses; corrected=true)
    total_var < eps(T) && return (indices=zeros(T, p), interaction_effects=zeros(T, p))

    # Estimate V[E[Y|X_~j]] by conditioning on all parameters except j
    total_indices = zeros(T, p)
    n_bins = max(3, isqrt(isqrt(n)))  # Fewer bins for higher-dimensional conditioning

    # For total order, we use the complementary approach:
    # Bin by parameter j, compute within-bin variance, average it
    # E[V[Y|X_j]] / V[Y] gives the fraction NOT explained by X_j alone
    # But for total order: S_Ti = E[V[Y|X_~i]] / V[Y]
    # Approximation: use leave-one-out conditioning via nearest neighbors

    for j in 1:p
        # Group samples with similar X_j values, compute residual variance
        order = sortperm(param_matrix[:, j])
        sorted_resp = responses[order]
        bin_size = max(2, n ÷ n_bins)

        within_var = zero(T)
        n_effective = 0
        for b in 1:n_bins
            s = (b-1) * bin_size + 1
            e = min(b * bin_size, n)
            s >= e && continue
            bv = var(sorted_resp[s:e]; corrected=true)
            count = e - s + 1
            within_var += bv * count
            n_effective += count
        end

        if n_effective > 0
            within_var /= n_effective
            # V[E[Y|X_j]] = V[Y] - E[V[Y|X_j]]
            # S_j = V[E[Y|X_j]] / V[Y] = 1 - E[V[Y|X_j]] / V[Y]
            # S_Tj >= S_j; approximate: S_Tj = within_var / total_var
            # (This captures both main effect and interactions)
            total_indices[j] = within_var / total_var
        end
    end

    # Total order should be >= first order, and the residual is interactions
    # Here we use the complementary definition
    total_indices .= one(T) .- (one(T) .- total_indices)

    # Get first-order for comparison
    first_order = sobol_first_order(param_matrix, responses; n_bootstrap=0, rng=rng)
    interaction_effects = total_indices .- first_order.indices
    interaction_effects .= max.(interaction_effects, zero(T))

    (indices=total_indices, interaction_effects=interaction_effects,
     first_order=first_order.indices, ranking=sortperm(total_indices, rev=true))
end

"""
    morris_elementary_effects(eval_fn::Function, bounds::Matrix{T},
                               param_names::Vector{String};
                               n_trajectories::Int=20, n_levels::Int=6,
                               rng::AbstractRNG=Random.default_rng()) where T

Morris method for screening parameters in high-dimensional spaces.

Generates random OAT (one-at-a-time) trajectories through parameter space,
computing elementary effects for each parameter at each step.

Arguments:
- `eval_fn`: Function mapping parameter vector -> scalar metric
- `bounds`: p x 2 matrix of [lower upper] bounds per parameter
- `param_names`: Names of parameters
- `n_trajectories`: Number of random trajectories
- `n_levels`: Number of grid levels for the design

Returns mu (mean effect), mu_star (mean absolute effect), sigma (std of effect)
for each parameter, enabling screening of important parameters.
"""
function morris_elementary_effects(eval_fn::Function, bounds::Matrix{T},
                                    param_names::Vector{String};
                                    n_trajectories::Int=20, n_levels::Int=6,
                                    rng::AbstractRNG=Random.default_rng()) where T<:Real
    p = size(bounds, 1)
    @assert size(bounds, 2) == 2 "bounds must be p x 2"
    @assert length(param_names) == p "param_names must match bounds"

    delta = T(n_levels) / T(2 * (n_levels - 1))
    effects = zeros(T, n_trajectories, p)

    for t in 1:n_trajectories
        # Generate random starting point on the grid
        x0 = zeros(T, p)
        for j in 1:p
            level = rand(rng, 0:(n_levels-2))
            x0[j] = T(level) / T(n_levels - 1)
        end

        # Convert to actual parameter values
        x_actual = _scale_to_bounds(x0, bounds)
        y0 = eval_fn(x_actual)

        # Random order of perturbation
        perm = randperm(rng, p)

        x_current = copy(x0)
        y_current = y0

        for j in perm
            x_new = copy(x_current)
            # Perturb parameter j by +/- delta
            if x_current[j] + delta <= one(T)
                x_new[j] = x_current[j] + delta
            else
                x_new[j] = x_current[j] - delta
            end

            x_new_actual = _scale_to_bounds(x_new, bounds)
            y_new = eval_fn(x_new_actual)

            # Elementary effect
            ee = (y_new - y_current) / delta
            effects[t, j] = ee

            x_current = x_new
            y_current = y_new
        end
    end

    # Compute Morris statistics
    mu = vec(mean(effects, dims=1))
    mu_star = vec(mean(abs.(effects), dims=1))
    sigma = vec(std(effects, dims=1; corrected=true))

    # Classification
    # High mu_star + low sigma => linear effect
    # High mu_star + high sigma => nonlinear or interaction effects
    # Low mu_star => negligible parameter
    classification = String[]
    median_mu_star = sort(mu_star)[max(1, p÷2)]
    median_sigma = sort(sigma)[max(1, p÷2)]

    for j in 1:p
        if mu_star[j] > median_mu_star
            if sigma[j] > median_sigma
                push!(classification, "nonlinear/interactive")
            else
                push!(classification, "linear")
            end
        else
            push!(classification, "negligible")
        end
    end

    (mu=mu, mu_star=mu_star, sigma=sigma,
     param_names=param_names, classification=classification,
     effects_matrix=effects, ranking=sortperm(mu_star, rev=true))
end

function _scale_to_bounds(x::AbstractVector{T}, bounds::AbstractMatrix{T}) where T<:Real
    p = length(x)
    result = similar(x)
    for j in 1:p
        result[j] = bounds[j, 1] + x[j] * (bounds[j, 2] - bounds[j, 1])
    end
    result
end

"""
    local_sensitivity(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                      h_fraction::T=T(0.01)) where T

Compute local sensitivity: gradient of performance metric w.r.t. each parameter.

Uses finite difference approximation with local polynomial regression
to estimate the gradient at the mean parameter values.

Returns gradient vector, standardized gradient, and elasticities.
"""
function local_sensitivity(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                           h_fraction::T=T(0.01)) where T<:Real
    n, p = size(param_matrix)
    n < 5 && error("Need at least 5 samples")

    # Compute gradient via linear regression (multivariate)
    # Y = X * beta + epsilon => beta ≈ gradient at centroid
    X_centered = param_matrix .- mean(param_matrix, dims=1)
    y_centered = responses .- mean(responses)

    # OLS: beta = (X'X)^(-1) X'y
    XtX = X_centered' * X_centered
    Xty = X_centered' * y_centered

    # Regularized solve for numerical stability
    lambda = T(1e-8) * tr(XtX) / p
    beta = (XtX + lambda * I) \ Xty

    # Standardized gradient: d(Y)/d(X_j) * std(X_j) / std(Y)
    param_stds = vec(std(param_matrix, dims=1; corrected=true))
    resp_std = std(responses; corrected=true)

    standardized = zeros(T, p)
    if resp_std > eps(T)
        for j in 1:p
            standardized[j] = beta[j] * param_stds[j] / resp_std
        end
    end

    # Elasticity: (dY/dX_j) * (mean(X_j) / mean(Y))
    param_means = vec(mean(param_matrix, dims=1))
    resp_mean = mean(responses)
    elasticity = zeros(T, p)
    if abs(resp_mean) > eps(T)
        for j in 1:p
            elasticity[j] = beta[j] * param_means[j] / resp_mean
        end
    end

    # R-squared of the linear model
    predicted = X_centered * beta
    ss_res = sum((y_centered .- predicted) .^ 2)
    ss_tot = sum(y_centered .^ 2)
    r_squared = ss_tot > eps(T) ? one(T) - ss_res / ss_tot : zero(T)

    (gradient=beta, standardized_gradient=standardized, elasticity=elasticity,
     r_squared=r_squared, param_means=param_means,
     ranking=sortperm(abs.(standardized), rev=true))
end

"""
    parameter_interactions(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                           param_names::Vector{String}=String[]) where T

Detect parameter interaction effects: which parameter pairs have synergies?

Fits a second-order polynomial model and examines cross-term coefficients.
Also computes correlation between parameter effects.

Returns interaction matrix, significant interactions, and synergy/conflict flags.
"""
function parameter_interactions(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                                param_names::Vector{String}=String[]) where T<:Real
    n, p = size(param_matrix)
    n_terms = p + p * (p - 1) ÷ 2  # linear + interaction terms
    n < n_terms + 2 && error("Need more samples than model terms (got $n, need $(n_terms+2))")

    if isempty(param_names)
        param_names = ["p$j" for j in 1:p]
    end

    # Standardize parameters
    pmeans = vec(mean(param_matrix, dims=1))
    pstds = vec(std(param_matrix, dims=1; corrected=true))
    pstds[pstds .< eps(T)] .= one(T)
    X_std = (param_matrix .- pmeans') ./ pstds'

    # Build design matrix with interaction terms
    # Columns: [X1, X2, ..., Xp, X1*X2, X1*X3, ..., X(p-1)*Xp]
    X_design = zeros(T, n, n_terms)
    X_design[:, 1:p] .= X_std

    col = p + 1
    interaction_pairs = Tuple{Int,Int}[]
    for i in 1:p
        for j in (i+1):p
            X_design[:, col] .= X_std[:, i] .* X_std[:, j]
            push!(interaction_pairs, (i, j))
            col += 1
        end
    end

    # OLS
    y = responses .- mean(responses)
    XtX = X_design' * X_design
    Xty = X_design' * y
    lambda = T(1e-8) * tr(XtX) / n_terms
    beta = (XtX + lambda * I) \ Xty

    # Extract interaction coefficients
    interaction_matrix = zeros(T, p, p)
    for (idx, (i, j)) in enumerate(interaction_pairs)
        coef = beta[p + idx]
        interaction_matrix[i, j] = coef
        interaction_matrix[j, i] = coef
    end

    # Significance: compare interaction coefficient to its standard error
    predicted = X_design * beta
    residuals = y .- predicted
    mse = sum(residuals .^ 2) / max(1, n - n_terms)
    cov_beta = mse * inv(XtX + lambda * I)

    significant_interactions = Tuple{String, String, T, T}[]
    for (idx, (i, j)) in enumerate(interaction_pairs)
        coef = beta[p + idx]
        se = sqrt(max(cov_beta[p + idx, p + idx], eps(T)))
        t_stat = coef / se
        # Approximate p-value using normal approximation
        pval = T(2) * (one(T) - _normal_cdf_approx(abs(t_stat)))
        if pval < T(0.05)
            push!(significant_interactions, (param_names[i], param_names[j], coef, pval))
        end
    end

    # Sort by absolute coefficient
    sort!(significant_interactions, by=x -> abs(x[3]), rev=true)

    # R-squared
    ss_res = sum(residuals .^ 2)
    ss_tot = sum(y .^ 2)
    r_squared = ss_tot > eps(T) ? one(T) - ss_res / ss_tot : zero(T)

    (interaction_matrix=interaction_matrix,
     significant_interactions=significant_interactions,
     linear_coefficients=beta[1:p],
     interaction_coefficients=beta[p+1:end],
     interaction_pairs=interaction_pairs,
     r_squared=r_squared,
     param_names=param_names)
end

function _normal_cdf_approx(x::T) where T<:Real
    a1 = T(0.254829592)
    a2 = T(-0.284496736)
    a3 = T(1.421413741)
    a4 = T(-1.453152027)
    a5 = T(1.061405429)
    pp = T(0.3275911)
    sign_x = x < zero(T) ? -one(T) : one(T)
    x_abs = abs(x)
    t = one(T) / (one(T) + pp * x_abs)
    y = one(T) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x_abs^2 / 2)
    T(0.5) * (one(T) + sign_x * y)
end

"""
    sensitivity_summary(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                        param_names::Vector{String}=String[],
                        rng::AbstractRNG=Random.default_rng()) where T

Run all sensitivity analyses and produce a unified summary.
"""
function sensitivity_summary(param_matrix::AbstractMatrix{T}, responses::AbstractVector{T};
                              param_names::Vector{String}=String[],
                              rng::AbstractRNG=Random.default_rng()) where T<:Real
    _, p = size(param_matrix)
    if isempty(param_names)
        param_names = ["param_$j" for j in 1:p]
    end

    sobol_1st = sobol_first_order(param_matrix, responses; rng=rng)
    sobol_tot = sobol_total_order(param_matrix, responses; rng=rng)
    local_sens = local_sensitivity(param_matrix, responses)
    interactions = parameter_interactions(param_matrix, responses; param_names=param_names)

    Dict(
        "sobol_first_order" => sobol_1st,
        "sobol_total_order" => sobol_tot,
        "local_sensitivity" => local_sens,
        "interactions" => interactions,
        "param_names" => param_names,
        "most_important" => param_names[sobol_1st.ranking[1]],
        "strongest_interaction" => isempty(interactions.significant_interactions) ?
            "none" : "$(interactions.significant_interactions[1][1]) x $(interactions.significant_interactions[1][2])"
    )
end

end # module ParameterSensitivity

# ============================================================================
# Section 5: Regime-Conditional Analysis
# ============================================================================

"""
    RegimeConditionalAnalysis

Hidden Markov Model-based regime detection and per-regime strategy evaluation.
"""
module RegimeConditionalAnalysis

using Statistics
using LinearAlgebra
using Random

"""
    HMMParameters{T<:Real}

Parameters for a Gaussian Hidden Markov Model with K states.
"""
mutable struct HMMParameters{T<:Real}
    k::Int                          # Number of states
    initial_probs::Vector{T}        # Initial state distribution (K,)
    transition_matrix::Matrix{T}    # Transition matrix (K x K)
    means::Vector{T}                # Emission means (K,)
    variances::Vector{T}            # Emission variances (K,)

    function HMMParameters{T}(k::Int) where T<:Real
        new{T}(k,
               fill(one(T)/k, k),
               fill(one(T)/k, k, k),
               zeros(T, k),
               ones(T, k))
    end
end

"""
    fit_hmm(observations::AbstractVector{T}; k::Int=3, max_iter::Int=100,
            tol::T=T(1e-6), n_init::Int=5,
            rng::AbstractRNG=Random.default_rng()) where T

Fit a Gaussian Hidden Markov Model with k states using the Baum-Welch algorithm
(a special case of EM for HMMs).

Arguments:
- `observations`: Time series of returns
- `k`: Number of hidden states (regimes)
- `max_iter`: Maximum EM iterations
- `tol`: Convergence tolerance on log-likelihood
- `n_init`: Number of random initializations

Returns the fitted HMMParameters and log-likelihood.
"""
function fit_hmm(observations::AbstractVector{T}; k::Int=3, max_iter::Int=100,
                  tol::T=T(1e-6), n_init::Int=5,
                  rng::AbstractRNG=Random.default_rng()) where T<:Real
    n = length(observations)
    n < k && error("Need at least $k observations")

    best_params = HMMParameters{T}(k)
    best_ll = T(-Inf)

    obs_mean = mean(observations)
    obs_std = std(observations; corrected=true)

    for init in 1:n_init
        params = HMMParameters{T}(k)

        # Initialize means spread across the data range
        sorted_obs = sort(observations)
        for i in 1:k
            idx = max(1, round(Int, (2*i - 1) * n / (2*k)))
            params.means[i] = sorted_obs[idx] + T(0.1) * randn(rng) * obs_std
        end
        sort!(params.means)

        params.variances .= obs_std^2 .* (T(0.5) .+ rand(rng, k))

        # Random transition matrix (encourage persistence)
        for i in 1:k
            for j in 1:k
                params.transition_matrix[i, j] = i == j ? T(0.9) + T(0.05) * rand(rng) : T(0.05) * rand(rng)
            end
            params.transition_matrix[i, :] ./= sum(params.transition_matrix[i, :])
        end

        params.initial_probs .= one(T) / k

        # Baum-Welch iterations
        alpha = zeros(T, n, k)
        beta = zeros(T, n, k)
        gamma = zeros(T, n, k)
        xi = zeros(T, n - 1, k, k)
        scale = zeros(T, n)

        prev_ll = T(-Inf)

        for iter in 1:max_iter
            # Forward pass (scaled)
            for j in 1:k
                alpha[1, j] = params.initial_probs[j] * _gauss_pdf(observations[1], params.means[j], params.variances[j])
            end
            scale[1] = sum(alpha[1, :])
            scale[1] = max(scale[1], eps(T))
            alpha[1, :] ./= scale[1]

            for t in 2:n
                for j in 1:k
                    alpha[t, j] = zero(T)
                    for i in 1:k
                        alpha[t, j] += alpha[t-1, i] * params.transition_matrix[i, j]
                    end
                    alpha[t, j] *= _gauss_pdf(observations[t], params.means[j], params.variances[j])
                end
                scale[t] = sum(alpha[t, :])
                scale[t] = max(scale[t], eps(T))
                alpha[t, :] ./= scale[t]
            end

            # Log-likelihood
            ll = sum(log.(scale))

            # Check convergence
            if abs(ll - prev_ll) < tol
                break
            end
            prev_ll = ll

            # Backward pass (scaled)
            beta[n, :] .= one(T)
            for t in (n-1):-1:1
                for i in 1:k
                    beta[t, i] = zero(T)
                    for j in 1:k
                        beta[t, i] += params.transition_matrix[i, j] *
                                       _gauss_pdf(observations[t+1], params.means[j], params.variances[j]) *
                                       beta[t+1, j]
                    end
                end
                s = max(scale[t+1], eps(T))
                beta[t, :] ./= s
            end

            # Compute gamma and xi
            for t in 1:n
                denom = sum(alpha[t, j] * beta[t, j] for j in 1:k)
                denom = max(denom, eps(T))
                for j in 1:k
                    gamma[t, j] = alpha[t, j] * beta[t, j] / denom
                end
            end

            for t in 1:(n-1)
                denom = zero(T)
                for i in 1:k
                    for j in 1:k
                        xi[t, i, j] = alpha[t, i] * params.transition_matrix[i, j] *
                                       _gauss_pdf(observations[t+1], params.means[j], params.variances[j]) *
                                       beta[t+1, j]
                        denom += xi[t, i, j]
                    end
                end
                denom = max(denom, eps(T))
                xi[t, :, :] ./= denom
            end

            # M-step: update parameters
            params.initial_probs .= gamma[1, :]
            params.initial_probs ./= sum(params.initial_probs)

            for i in 1:k
                gamma_sum = sum(gamma[t, i] for t in 1:(n-1))
                gamma_sum = max(gamma_sum, eps(T))
                for j in 1:k
                    params.transition_matrix[i, j] = sum(xi[t, i, j] for t in 1:(n-1)) / gamma_sum
                end
                # Normalize row
                row_sum = sum(params.transition_matrix[i, :])
                params.transition_matrix[i, :] ./= max(row_sum, eps(T))
            end

            for j in 1:k
                gamma_sum = sum(gamma[t, j] for t in 1:n)
                gamma_sum = max(gamma_sum, eps(T))
                params.means[j] = sum(gamma[t, j] * observations[t] for t in 1:n) / gamma_sum
                params.variances[j] = sum(gamma[t, j] * (observations[t] - params.means[j])^2 for t in 1:n) / gamma_sum
                params.variances[j] = max(params.variances[j], eps(T))
            end
        end

        ll = sum(log.(scale))
        if ll > best_ll
            best_ll = ll
            best_params = params
        end
    end

    # Sort states by mean (low -> high: bear, neutral, bull)
    order = sortperm(best_params.means)
    best_params.means .= best_params.means[order]
    best_params.variances .= best_params.variances[order]
    best_params.initial_probs .= best_params.initial_probs[order]
    best_params.transition_matrix .= best_params.transition_matrix[order, order]

    (params=best_params, log_likelihood=best_ll)
end

function _gauss_pdf(x::T, mu::T, var::T) where T<:Real
    var = max(var, eps(T))
    T(1.0 / sqrt(2.0 * pi * var)) * exp(-(x - mu)^2 / (T(2) * var))
end

"""
    detect_regimes(observations::AbstractVector{T}, params::HMMParameters{T}) where T

Run Viterbi algorithm to find the most likely state sequence.

Returns the state sequence and state probabilities at each time step.
"""
function detect_regimes(observations::AbstractVector{T}, params::HMMParameters{T}) where T<:Real
    n = length(observations)
    k = params.k

    # Viterbi algorithm (log space)
    log_delta = zeros(T, n, k)
    psi = zeros(Int, n, k)

    for j in 1:k
        log_delta[1, j] = log(max(params.initial_probs[j], eps(T))) +
                           _log_gauss_pdf(observations[1], params.means[j], params.variances[j])
    end

    for t in 2:n
        for j in 1:k
            best_val = T(-Inf)
            best_state = 1
            for i in 1:k
                val = log_delta[t-1, i] + log(max(params.transition_matrix[i, j], eps(T)))
                if val > best_val
                    best_val = val
                    best_state = i
                end
            end
            log_delta[t, j] = best_val + _log_gauss_pdf(observations[t], params.means[j], params.variances[j])
            psi[t, j] = best_state
        end
    end

    # Backtrack
    states = zeros(Int, n)
    states[n] = argmax(log_delta[n, :])
    for t in (n-1):-1:1
        states[t] = psi[t+1, states[t+1]]
    end

    # Forward-backward for state probabilities
    state_probs = zeros(T, n, k)
    alpha = zeros(T, n, k)

    for j in 1:k
        alpha[1, j] = params.initial_probs[j] * _gauss_pdf(observations[1], params.means[j], params.variances[j])
    end
    s = sum(alpha[1, :])
    alpha[1, :] ./= max(s, eps(T))

    for t in 2:n
        for j in 1:k
            for i in 1:k
                alpha[t, j] += alpha[t-1, i] * params.transition_matrix[i, j]
            end
            alpha[t, j] *= _gauss_pdf(observations[t], params.means[j], params.variances[j])
        end
        s = sum(alpha[t, :])
        alpha[t, :] ./= max(s, eps(T))
    end

    # Approximate state probs from forward only (smoothing omitted for brevity)
    state_probs .= alpha

    # Regime labels
    regime_labels = ["bear", "neutral", "bull"]
    if k > 3
        regime_labels = ["regime_$i" for i in 1:k]
    end

    # Regime durations
    durations = Dict{Int, Vector{Int}}()
    for s in 1:k
        durations[s] = Int[]
    end
    current_state = states[1]
    current_dur = 1
    for t in 2:n
        if states[t] == current_state
            current_dur += 1
        else
            push!(durations[current_state], current_dur)
            current_state = states[t]
            current_dur = 1
        end
    end
    push!(durations[current_state], current_dur)

    mean_durations = Dict{Int, Float64}()
    for s in 1:k
        d = durations[s]
        mean_durations[s] = isempty(d) ? 0.0 : mean(Float64.(d))
    end

    (states=states, state_probs=state_probs, regime_labels=regime_labels,
     durations=durations, mean_durations=mean_durations)
end

function _log_gauss_pdf(x::T, mu::T, var::T) where T<:Real
    var = max(var, eps(T))
    -T(0.5) * log(T(2) * T(pi) * var) - (x - mu)^2 / (T(2) * var)
end

"""
    per_regime_sharpe(return_matrix::AbstractMatrix{T}, states::AbstractVector{Int};
                      k::Int=3) where T

Compute Sharpe ratio for each strategy in each regime.

Arguments:
- `return_matrix`: T x N return matrix
- `states`: T-vector of regime labels (1..k)
- `k`: Number of regimes

Returns a k x N matrix of per-regime Sharpe ratios.
"""
function per_regime_sharpe(return_matrix::AbstractMatrix{T}, states::AbstractVector{Int};
                           k::Int=3) where T<:Real
    ntime, nstrats = size(return_matrix)
    @assert length(states) == ntime "States must match time dimension"

    regime_sharpes = zeros(T, k, nstrats)
    regime_counts = zeros(Int, k)

    for s in 1:k
        mask = states .== s
        regime_counts[s] = sum(mask)
        if regime_counts[s] < 2
            continue
        end
        regime_rets = return_matrix[mask, :]
        for j in 1:nstrats
            r = regime_rets[:, j]
            mu = mean(r)
            sigma = std(r; corrected=true)
            regime_sharpes[s, j] = sigma > eps(T) ? mu / sigma * sqrt(T(252)) : zero(T)
        end
    end

    (sharpes=regime_sharpes, regime_counts=regime_counts)
end

"""
    classify_specialist_generalist(regime_sharpes::AbstractMatrix{T};
                                    threshold::T=T(0.5)) where T

Classify strategies as regime specialists or generalists.

A specialist has high Sharpe in one regime but low/negative in others.
A generalist has moderate Sharpe across all regimes.

Arguments:
- `regime_sharpes`: k x N matrix of per-regime Sharpe ratios
- `threshold`: Minimum average Sharpe to be considered a generalist

Returns classification labels and specialist regime assignments.
"""
function classify_specialist_generalist(regime_sharpes::AbstractMatrix{T};
                                         threshold::T=T(0.5)) where T<:Real
    k, nstrats = size(regime_sharpes)
    classifications = String[]
    specialist_regimes = Int[]

    for j in 1:nstrats
        sharpes = regime_sharpes[:, j]
        avg_sharpe = mean(sharpes)
        max_sharpe = maximum(sharpes)
        min_sharpe = minimum(sharpes)
        best_regime = argmax(sharpes)

        # Specialist: high variance across regimes, strong in one
        sharpe_range = max_sharpe - min_sharpe
        cv = std(sharpes; corrected=true) / max(abs(avg_sharpe), eps(T))

        if cv > T(1.5) && max_sharpe > threshold
            push!(classifications, "specialist")
            push!(specialist_regimes, best_regime)
        elseif avg_sharpe > threshold && cv < T(0.8)
            push!(classifications, "generalist")
            push!(specialist_regimes, 0)
        elseif avg_sharpe > zero(T)
            push!(classifications, "moderate")
            push!(specialist_regimes, best_regime)
        else
            push!(classifications, "poor")
            push!(specialist_regimes, 0)
        end
    end

    # Summary counts
    n_specialist = count(c -> c == "specialist", classifications)
    n_generalist = count(c -> c == "generalist", classifications)
    n_moderate = count(c -> c == "moderate", classifications)
    n_poor = count(c -> c == "poor", classifications)

    (classifications=classifications, specialist_regimes=specialist_regimes,
     n_specialist=n_specialist, n_generalist=n_generalist,
     n_moderate=n_moderate, n_poor=n_poor)
end

"""
    regime_transition_cost(return_matrix::AbstractMatrix{T},
                           states::AbstractVector{Int},
                           specialist_map::Dict{Int, Vector{Int}};
                           transaction_cost::T=T(0.001)) where T

Estimate the cost of switching between specialist strategies as regimes change.

Arguments:
- `return_matrix`: T x N return matrix
- `states`: T-vector of regime states
- `specialist_map`: Dict mapping regime -> vector of strategy indices specialized in it
- `transaction_cost`: Cost per unit of turnover

Returns total transition cost, number of transitions, and net benefit of switching.
"""
function regime_transition_cost(return_matrix::AbstractMatrix{T},
                                states::AbstractVector{Int},
                                specialist_map::Dict{Int, Vector{Int}};
                                transaction_cost::T=T(0.001)) where T<:Real
    ntime, nstrats = size(return_matrix)
    @assert length(states) == ntime

    total_switch_cost = zero(T)
    n_transitions = 0
    specialist_return = zero(T)
    generalist_return = zero(T)

    # Identify generalist: best average Sharpe strategy
    avg_returns = vec(mean(return_matrix, dims=1))
    generalist_idx = argmax(avg_returns)

    prev_state = states[1]
    current_specialist = get(specialist_map, prev_state, Int[])

    for t in 2:ntime
        new_state = states[t]

        # Generalist return (always hold the same strategy)
        generalist_return += return_matrix[t, generalist_idx]

        # Specialist return
        if !isempty(current_specialist)
            best_specialist = current_specialist[argmax([return_matrix[t, s] for s in current_specialist])]
            specialist_return += return_matrix[t, best_specialist]
        else
            specialist_return += return_matrix[t, generalist_idx]
        end

        if new_state != prev_state
            n_transitions += 1
            total_switch_cost += transaction_cost * T(2)  # Buy new + sell old

            new_specialists = get(specialist_map, new_state, Int[])
            current_specialist = new_specialists
            prev_state = new_state
        end
    end

    net_specialist = specialist_return - total_switch_cost
    benefit = net_specialist - generalist_return

    (total_switch_cost=total_switch_cost,
     n_transitions=n_transitions,
     specialist_return_gross=specialist_return,
     specialist_return_net=net_specialist,
     generalist_return=generalist_return,
     switching_benefit=benefit,
     switching_beneficial=benefit > zero(T),
     avg_cost_per_transition=n_transitions > 0 ? total_switch_cost / n_transitions : zero(T))
end

"""
    full_regime_analysis(return_matrix::AbstractMatrix{T};
                         k::Int=3, rng::AbstractRNG=Random.default_rng()) where T

Run complete regime-conditional analysis pipeline.
"""
function full_regime_analysis(return_matrix::AbstractMatrix{T};
                               k::Int=3, rng::AbstractRNG=Random.default_rng()) where T<:Real
    ntime, nstrats = size(return_matrix)

    # Aggregate return series (equal-weight portfolio)
    agg_returns = vec(mean(return_matrix, dims=2))

    # Fit HMM
    hmm_result = fit_hmm(agg_returns; k=k, rng=rng)

    # Detect regimes
    regime_result = detect_regimes(agg_returns, hmm_result.params)

    # Per-regime Sharpe
    pr_sharpe = per_regime_sharpe(return_matrix, regime_result.states; k=k)

    # Classify strategies
    classification = classify_specialist_generalist(pr_sharpe.sharpes)

    # Build specialist map
    specialist_map = Dict{Int, Vector{Int}}()
    for s in 1:k
        specialist_map[s] = findall(i -> classification.specialist_regimes[i] == s &&
                                          classification.classifications[i] == "specialist",
                                    1:nstrats)
    end

    # Transition cost
    tc = regime_transition_cost(return_matrix, regime_result.states, specialist_map)

    Dict(
        "hmm_params" => hmm_result.params,
        "log_likelihood" => hmm_result.log_likelihood,
        "states" => regime_result.states,
        "regime_labels" => regime_result.regime_labels,
        "mean_durations" => regime_result.mean_durations,
        "regime_sharpes" => pr_sharpe.sharpes,
        "regime_counts" => pr_sharpe.regime_counts,
        "classifications" => classification.classifications,
        "specialist_map" => specialist_map,
        "transition_cost" => tc,
    )
end

end # module RegimeConditionalAnalysis

# ============================================================================
# Section 6: Optimal Portfolio Construction
# ============================================================================

"""
    OptimalPortfolioConstruction

Portfolio optimization across STRATEGIES (not assets), treating each backtest
strategy as an investable return stream.
"""
module OptimalPortfolioConstruction

using Statistics
using LinearAlgebra

"""
    markowitz_mvo(return_matrix::AbstractMatrix{T};
                  target_return::Union{T, Nothing}=nothing,
                  risk_aversion::T=T(1.0),
                  long_only::Bool=true,
                  max_weight::T=T(0.3)) where T

Mean-Variance Optimization across strategies.

If target_return is specified, minimizes variance subject to return constraint.
Otherwise, maximizes utility = mu'w - (risk_aversion/2) * w'Sigma*w.

Arguments:
- `return_matrix`: T x N return matrix (T periods, N strategies)
- `target_return`: Target portfolio return (optional)
- `risk_aversion`: Risk aversion parameter for utility maximization
- `long_only`: Whether to enforce non-negative weights
- `max_weight`: Maximum weight per strategy

Returns optimal weights, expected return, volatility, and Sharpe ratio.
"""
function markowitz_mvo(return_matrix::AbstractMatrix{T};
                       target_return::Union{T, Nothing}=nothing,
                       risk_aversion::T=T(1.0),
                       long_only::Bool=true,
                       max_weight::T=T(0.3)) where T<:Real
    ntime, n = size(return_matrix)
    n < 2 && error("Need at least 2 strategies")

    mu = vec(mean(return_matrix, dims=1))
    Sigma = cov(return_matrix)

    # Regularize covariance matrix
    lambda_reg = T(1e-6) * tr(Sigma) / n
    Sigma_reg = Sigma + lambda_reg * I

    if isnothing(target_return)
        # Maximize utility: max mu'w - (gamma/2) w'Sigma w
        # Unconstrained: w* = (1/gamma) * Sigma^(-1) * mu
        w = Sigma_reg \ mu
        w ./= risk_aversion

        if long_only
            w .= max.(w, zero(T))
        end
        w .= min.(w, max_weight)

        # Normalize to sum to 1
        w_sum = sum(abs.(w))
        w_sum > eps(T) && (w ./= w_sum)
    else
        # Minimize variance subject to mu'w = target_return, 1'w = 1
        # Lagrangian solution
        ones_vec = ones(T, n)
        Sigma_inv = inv(Sigma_reg)

        A = mu' * Sigma_inv * mu
        B = mu' * Sigma_inv * ones_vec
        C = ones_vec' * Sigma_inv * ones_vec
        D = A * C - B^2

        if abs(D) < eps(T)
            w = fill(one(T) / n, n)
        else
            g = (A * (Sigma_inv * ones_vec) - B * (Sigma_inv * mu)) / D
            h = (C * (Sigma_inv * mu) - B * (Sigma_inv * ones_vec)) / D
            w = g .+ h .* target_return
        end

        if long_only
            w .= max.(w, zero(T))
            w_sum = sum(w)
            w_sum > eps(T) && (w ./= w_sum)
        end
        w .= min.(w, max_weight)
        w_sum = sum(w)
        w_sum > eps(T) && (w ./= w_sum)
    end

    # Portfolio statistics
    port_return = dot(mu, w) * T(252)
    port_vol = sqrt(dot(w, Sigma_reg * w)) * sqrt(T(252))
    port_sharpe = port_vol > eps(T) ? port_return / port_vol : zero(T)

    # Effective number of strategies
    hhi = sum(w .^ 2)
    eff_n = hhi > eps(T) ? one(T) / hhi : T(n)

    (weights=w, expected_return=port_return, volatility=port_vol,
     sharpe=port_sharpe, effective_n=eff_n, covariance=Sigma)
end

"""
    risk_parity_portfolio(return_matrix::AbstractMatrix{T};
                          max_iter::Int=200, tol::T=T(1e-8)) where T

Compute the risk parity (equal risk contribution) portfolio across strategies.

Each strategy contributes equally to total portfolio risk:
w_i * (Sigma * w)_i = sigma_p^2 / N for all i.

Uses iterative reweighting algorithm.
"""
function risk_parity_portfolio(return_matrix::AbstractMatrix{T};
                               max_iter::Int=200, tol::T=T(1e-8)) where T<:Real
    ntime, n = size(return_matrix)
    Sigma = cov(return_matrix)
    lambda_reg = T(1e-8) * tr(Sigma) / n
    Sigma_reg = Sigma + lambda_reg * I

    # Initialize equal weights
    w = fill(one(T) / n, n)

    for iter in 1:max_iter
        # Marginal risk contribution: MRC_i = (Sigma * w)_i
        mrc = Sigma_reg * w
        port_var = dot(w, mrc)
        port_var < eps(T) && break

        # Risk contribution: RC_i = w_i * MRC_i
        rc = w .* mrc

        # Target: equal risk contribution
        target_rc = port_var / n

        # Update weights: w_i = target_rc / MRC_i (inverse volatility weighting variant)
        w_new = similar(w)
        for i in 1:n
            w_new[i] = mrc[i] > eps(T) ? target_rc / mrc[i] : w[i]
        end
        w_new .= max.(w_new, eps(T))
        w_new ./= sum(w_new)

        # Check convergence
        if maximum(abs.(w_new .- w)) < tol
            w = w_new
            break
        end
        w = w_new
    end

    # Final statistics
    mu = vec(mean(return_matrix, dims=1))
    port_return = dot(mu, w) * T(252)
    port_vol = sqrt(dot(w, Sigma_reg * w)) * sqrt(T(252))
    port_sharpe = port_vol > eps(T) ? port_return / port_vol : zero(T)

    # Risk contributions
    mrc_final = Sigma_reg * w
    rc_final = w .* mrc_final
    rc_pct = rc_final ./ max(sum(rc_final), eps(T))

    (weights=w, expected_return=port_return, volatility=port_vol,
     sharpe=port_sharpe, risk_contributions=rc_final,
     risk_contribution_pct=rc_pct, risk_contribution_std=std(rc_pct; corrected=true))
end

"""
    max_diversification_portfolio(return_matrix::AbstractMatrix{T};
                                  max_iter::Int=200, tol::T=T(1e-8)) where T

Compute the maximum diversification portfolio across strategies.

Maximizes the diversification ratio: DR = (w' * sigma) / sqrt(w' * Sigma * w)
where sigma is the vector of individual volatilities.

Uses iterative optimization.
"""
function max_diversification_portfolio(return_matrix::AbstractMatrix{T};
                                       max_iter::Int=200, tol::T=T(1e-8)) where T<:Real
    ntime, n = size(return_matrix)
    Sigma = cov(return_matrix)
    lambda_reg = T(1e-8) * tr(Sigma) / n
    Sigma_reg = Sigma + lambda_reg * I

    sigmas = sqrt.(diag(Sigma_reg))

    # Correlation matrix
    D_inv = Diagonal(one(T) ./ max.(sigmas, eps(T)))
    C = D_inv * Sigma_reg * D_inv

    # Maximum diversification = minimum variance in correlation space
    # w_corr = C^(-1) * 1 / (1' * C^(-1) * 1)
    C_reg = C + T(1e-8) * I
    C_inv = inv(C_reg)
    ones_vec = ones(T, n)
    w_corr = C_inv * ones_vec
    w_corr ./= sum(w_corr)

    # Transform back: w = D_inv * w_corr
    w = D_inv * w_corr
    w .= max.(w, zero(T))
    w ./= max(sum(w), eps(T))

    # Compute diversification ratio
    weighted_vol = dot(w, sigmas)
    port_vol_pre = sqrt(max(dot(w, Sigma_reg * w), eps(T)))
    div_ratio = weighted_vol / port_vol_pre

    # Portfolio stats
    mu = vec(mean(return_matrix, dims=1))
    port_return = dot(mu, w) * T(252)
    port_vol = port_vol_pre * sqrt(T(252))
    port_sharpe = port_vol > eps(T) ? port_return / port_vol : zero(T)

    (weights=w, expected_return=port_return, volatility=port_vol,
     sharpe=port_sharpe, diversification_ratio=div_ratio,
     individual_volatilities=sigmas .* sqrt(T(252)))
end

"""
    kelly_criterion(return_matrix::AbstractMatrix{T};
                    fraction::T=T(0.5),
                    max_leverage::T=T(2.0)) where T

Compute Kelly criterion optimal allocation across strategies.

Full Kelly: f* = Sigma^(-1) * mu
Half Kelly (fraction=0.5) is commonly used for more conservative sizing.

Arguments:
- `return_matrix`: T x N return matrix
- `fraction`: Kelly fraction (0.5 = half Kelly)
- `max_leverage`: Maximum total leverage
"""
function kelly_criterion(return_matrix::AbstractMatrix{T};
                         fraction::T=T(0.5),
                         max_leverage::T=T(2.0)) where T<:Real
    ntime, n = size(return_matrix)
    mu = vec(mean(return_matrix, dims=1))
    Sigma = cov(return_matrix)
    lambda_reg = T(1e-6) * tr(Sigma) / n
    Sigma_reg = Sigma + lambda_reg * I

    # Full Kelly
    f_full = Sigma_reg \ mu

    # Apply fraction
    f = f_full .* fraction

    # Cap leverage
    total_leverage = sum(abs.(f))
    if total_leverage > max_leverage
        f .*= max_leverage / total_leverage
    end

    # Normalize to weights
    w = f ./ max(sum(abs.(f)), eps(T))

    # Expected geometric growth rate (Kelly criterion)
    port_mu = dot(mu, f)
    port_var = dot(f, Sigma_reg * f)
    growth_rate = port_mu - port_var / T(2)

    # Portfolio stats
    port_return = dot(mu, w) * T(252)
    port_vol = sqrt(max(dot(w, Sigma_reg * w), eps(T))) * sqrt(T(252))
    port_sharpe = port_vol > eps(T) ? port_return / port_vol : zero(T)

    (weights=w, kelly_fractions=f, full_kelly=f_full,
     expected_return=port_return, volatility=port_vol,
     sharpe=port_sharpe, growth_rate=growth_rate * T(252),
     leverage=sum(abs.(f)))
end

"""
    black_litterman(return_matrix::AbstractMatrix{T},
                    views::Vector{Tuple{Vector{T}, T, T}};
                    tau::T=T(0.05),
                    risk_aversion::T=T(2.5),
                    market_weights::Union{Vector{T}, Nothing}=nothing) where T

Black-Litterman model for strategy allocation with views on regime outlook.

Arguments:
- `return_matrix`: T x N return matrix
- `views`: Vector of (pick_vector, view_return, view_confidence) tuples.
  Each pick_vector is length N, view_return is the expected return of that view,
  view_confidence is the uncertainty (variance) of the view.
- `tau`: Uncertainty scaling of the prior
- `risk_aversion`: Risk aversion parameter
- `market_weights`: Prior equilibrium weights (default: equal weight)

Returns posterior expected returns, optimal weights, and diagnostics.
"""
function black_litterman(return_matrix::AbstractMatrix{T},
                         views::Vector{Tuple{Vector{T}, T, T}};
                         tau::T=T(0.05),
                         risk_aversion::T=T(2.5),
                         market_weights::Union{Vector{T}, Nothing}=nothing) where T<:Real
    ntime, n = size(return_matrix)
    Sigma = cov(return_matrix)
    lambda_reg = T(1e-6) * tr(Sigma) / n
    Sigma_reg = Sigma + lambda_reg * I

    # Market equilibrium weights
    if isnothing(market_weights)
        market_weights = fill(one(T) / n, n)
    end

    # Implied equilibrium returns: Pi = delta * Sigma * w_mkt
    Pi = risk_aversion * Sigma_reg * market_weights

    if isempty(views)
        # No views: return equilibrium
        w = market_weights
        port_return = dot(Pi, w) * T(252)
        port_vol = sqrt(dot(w, Sigma_reg * w)) * sqrt(T(252))
        port_sharpe = port_vol > eps(T) ? port_return / port_vol : zero(T)
        return (weights=w, posterior_returns=Pi .* T(252),
                prior_returns=Pi .* T(252),
                expected_return=port_return, volatility=port_vol,
                sharpe=port_sharpe)
    end

    # Build view matrices
    k_views = length(views)
    P = zeros(T, k_views, n)   # Pick matrix
    Q = zeros(T, k_views)       # View returns
    Omega = zeros(T, k_views, k_views)  # View uncertainty

    for (i, (pick, ret, conf)) in enumerate(views)
        @assert length(pick) == n "Pick vector must be length $n"
        P[i, :] .= pick
        Q[i] = ret
        Omega[i, i] = conf
    end

    # Black-Litterman posterior
    # mu_BL = [(tau*Sigma)^(-1) + P'*Omega^(-1)*P]^(-1) *
    #         [(tau*Sigma)^(-1)*Pi + P'*Omega^(-1)*Q]
    tau_Sigma_inv = inv(tau * Sigma_reg)
    Omega_inv = inv(Omega + T(1e-12) * I)

    posterior_precision = tau_Sigma_inv + P' * Omega_inv * P
    posterior_cov = inv(posterior_precision)
    posterior_mean = posterior_cov * (tau_Sigma_inv * Pi + P' * Omega_inv * Q)

    # Optimal weights from posterior
    w_bl = Sigma_reg \ posterior_mean
    w_bl ./= risk_aversion
    w_bl .= max.(w_bl, zero(T))
    w_sum = sum(w_bl)
    w_sum > eps(T) && (w_bl ./= w_sum)

    port_return = dot(posterior_mean, w_bl) * T(252)
    port_vol = sqrt(max(dot(w_bl, Sigma_reg * w_bl), eps(T))) * sqrt(T(252))
    port_sharpe = port_vol > eps(T) ? port_return / port_vol : zero(T)

    (weights=w_bl, posterior_returns=posterior_mean .* T(252),
     prior_returns=Pi .* T(252),
     posterior_covariance=posterior_cov,
     expected_return=port_return, volatility=port_vol,
     sharpe=port_sharpe, view_impact=posterior_mean .- Pi)
end

"""
    compare_portfolios(return_matrix::AbstractMatrix{T}) where T

Run all portfolio construction methods and compare results.
"""
function compare_portfolios(return_matrix::AbstractMatrix{T}) where T<:Real
    mvo = markowitz_mvo(return_matrix)
    rp = risk_parity_portfolio(return_matrix)
    mdp = max_diversification_portfolio(return_matrix)
    kelly = kelly_criterion(return_matrix)

    # Equal weight benchmark
    _, n = size(return_matrix)
    ew = fill(one(T) / n, n)
    mu = vec(mean(return_matrix, dims=1))
    Sigma = cov(return_matrix)
    ew_ret = dot(mu, ew) * T(252)
    ew_vol = sqrt(dot(ew, Sigma * ew)) * sqrt(T(252))
    ew_sharpe = ew_vol > eps(T) ? ew_ret / ew_vol : zero(T)

    Dict(
        "mvo" => (weights=mvo.weights, sharpe=mvo.sharpe, ret=mvo.expected_return, vol=mvo.volatility),
        "risk_parity" => (weights=rp.weights, sharpe=rp.sharpe, ret=rp.expected_return, vol=rp.volatility),
        "max_diversification" => (weights=mdp.weights, sharpe=mdp.sharpe, ret=mdp.expected_return, vol=mdp.volatility),
        "kelly" => (weights=kelly.weights, sharpe=kelly.sharpe, ret=kelly.expected_return, vol=kelly.volatility),
        "equal_weight" => (weights=ew, sharpe=ew_sharpe, ret=ew_ret, vol=ew_vol),
    )
end

end # module OptimalPortfolioConstruction

# ============================================================================
# Section 7: Backtest Validation
# ============================================================================

"""
    BacktestValidation

Statistical validation methods for backtest results including walk-forward
analysis, bootstrap, permutation tests, and CPCV.
"""
module BacktestValidation

using Statistics
using LinearAlgebra
using Random

"""
    walk_forward_consistency(returns::AbstractVector{T}; n_folds::Int=5,
                             is_fraction::T=T(0.6)) where T

Evaluate walk-forward consistency: compare in-sample vs out-of-sample Sharpe
across multiple folds.

Splits the time series into n_folds sequential folds. For each fold,
uses is_fraction as in-sample and the remainder as out-of-sample.

Returns IS/OOS Sharpe for each fold, correlation, and degradation ratio.
"""
function walk_forward_consistency(returns::AbstractVector{T}; n_folds::Int=5,
                                   is_fraction::T=T(0.6)) where T<:Real
    n = length(returns)
    fold_size = n ÷ n_folds
    fold_size < 10 && error("Folds too small (size=$fold_size)")

    is_sharpes = zeros(T, n_folds)
    oos_sharpes = zeros(T, n_folds)

    for fold in 1:n_folds
        start_idx = (fold - 1) * fold_size + 1
        end_idx = fold == n_folds ? n : fold * fold_size

        fold_data = returns[start_idx:end_idx]
        is_end = round(Int, length(fold_data) * is_fraction)
        is_end = max(is_end, 2)

        is_data = fold_data[1:is_end]
        oos_data = fold_data[(is_end+1):end]

        is_sharpes[fold] = _sharpe(is_data)
        oos_sharpes[fold] = length(oos_data) >= 2 ? _sharpe(oos_data) : zero(T)
    end

    # Correlation between IS and OOS Sharpe
    correlation = _pearson_correlation(is_sharpes, oos_sharpes)

    # Degradation ratio: OOS Sharpe / IS Sharpe
    mean_is = mean(is_sharpes)
    mean_oos = mean(oos_sharpes)
    degradation = abs(mean_is) > eps(T) ? mean_oos / mean_is : zero(T)

    # Consistency: fraction of folds where OOS Sharpe > 0
    consistency = mean(oos_sharpes .> zero(T))

    (is_sharpes=is_sharpes, oos_sharpes=oos_sharpes,
     correlation=correlation, degradation_ratio=degradation,
     consistency=consistency, mean_is_sharpe=mean_is,
     mean_oos_sharpe=mean_oos, n_folds=n_folds,
     pass=degradation > T(0.5) && consistency > T(0.5))
end

function _sharpe(rets::AbstractVector{T}) where T<:Real
    n = length(rets)
    n < 2 && return zero(T)
    mu = mean(rets)
    sigma = std(rets; corrected=true)
    sigma < eps(T) && return zero(T)
    mu / sigma * sqrt(T(252))
end

function _pearson_correlation(x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 2 && return zero(T)
    mx = mean(x)
    my = mean(y)
    sx = std(x; corrected=true)
    sy = std(y; corrected=true)
    (sx < eps(T) || sy < eps(T)) && return zero(T)
    mean((x .- mx) .* (y .- my)) / (sx * sy) * T(n) / T(n - 1)
end

"""
    ts_bootstrap_ci(returns::AbstractVector{T}; n_bootstrap::Int=1000,
                    confidence::T=T(0.95), block_length::Int=-1,
                    method::Symbol=:stationary,
                    rng::AbstractRNG=Random.default_rng()) where T

Time-series bootstrap confidence intervals for the Sharpe ratio.

Supports:
- :stationary - Politis & Romano stationary bootstrap (geometric block lengths)
- :circular - Circular block bootstrap (fixed block lengths)

Arguments:
- `returns`: Return series
- `n_bootstrap`: Number of bootstrap replications
- `confidence`: Confidence level
- `block_length`: Expected block length (auto if -1)
- `method`: :stationary or :circular
"""
function ts_bootstrap_ci(returns::AbstractVector{T}; n_bootstrap::Int=1000,
                          confidence::T=T(0.95), block_length::Int=-1,
                          method::Symbol=:stationary,
                          rng::AbstractRNG=Random.default_rng()) where T<:Real
    n = length(returns)
    n < 10 && error("Need at least 10 observations")

    # Auto block length (cube root rule)
    if block_length < 1
        block_length = max(1, round(Int, n^(1/3)))
    end

    sharpe_original = _sharpe(returns)
    boot_sharpes = zeros(T, n_bootstrap)

    for b in 1:n_bootstrap
        boot_sample = if method == :stationary
            _stationary_bootstrap(returns, block_length, rng)
        elseif method == :circular
            _circular_bootstrap(returns, block_length, rng)
        else
            error("Unknown method: $method")
        end
        boot_sharpes[b] = _sharpe(boot_sample)
    end

    sort!(boot_sharpes)
    alpha = one(T) - confidence
    ci_low = boot_sharpes[max(1, round(Int, alpha / 2 * n_bootstrap))]
    ci_high = boot_sharpes[max(1, round(Int, (one(T) - alpha / 2) * n_bootstrap))]

    boot_mean = mean(boot_sharpes)
    boot_std = std(boot_sharpes; corrected=true)

    # Bias-corrected estimate
    bias = boot_mean - sharpe_original
    bc_sharpe = sharpe_original - bias

    (original_sharpe=sharpe_original,
     ci_low=ci_low, ci_high=ci_high,
     confidence=confidence,
     boot_mean=boot_mean, boot_std=boot_std,
     bias=bias, bias_corrected_sharpe=bc_sharpe,
     significant=ci_low > zero(T),
     method=method, block_length=block_length)
end

function _stationary_bootstrap(data::AbstractVector{T}, expected_block::Int,
                                rng::AbstractRNG) where T<:Real
    n = length(data)
    p = one(T) / expected_block  # Probability of starting new block
    result = similar(data)
    idx = rand(rng, 1:n)

    for i in 1:n
        result[i] = data[idx]
        if rand(rng) < p
            idx = rand(rng, 1:n)  # New random start
        else
            idx = mod1(idx + 1, n)  # Continue block
        end
    end
    result
end

function _circular_bootstrap(data::AbstractVector{T}, block_length::Int,
                              rng::AbstractRNG) where T<:Real
    n = length(data)
    result = similar(data)
    pos = 1
    while pos <= n
        start = rand(rng, 1:n)
        for j in 0:(block_length-1)
            pos > n && break
            result[pos] = data[mod1(start + j, n)]
            pos += 1
        end
    end
    result
end

"""
    permutation_test(returns::AbstractVector{T}; n_permutations::Int=5000,
                     rng::AbstractRNG=Random.default_rng()) where T

Permutation test for strategy significance.

Tests whether the strategy's Sharpe ratio is significantly different from
what would be expected by random chance (i.e., under random reordering of returns).

Under the null hypothesis, the temporal ordering of returns doesn't matter,
so we permute the return series and recompute Sharpe each time.

Returns p-value, permutation distribution, and significance flag.
"""
function permutation_test(returns::AbstractVector{T}; n_permutations::Int=5000,
                           rng::AbstractRNG=Random.default_rng()) where T<:Real
    n = length(returns)
    n < 5 && error("Need at least 5 observations")

    observed_sharpe = _sharpe(returns)
    perm_sharpes = zeros(T, n_permutations)

    for p in 1:n_permutations
        perm_returns = returns[randperm(rng, n)]
        perm_sharpes[p] = _sharpe(perm_returns)
    end

    # One-sided p-value: fraction of permutations with Sharpe >= observed
    pvalue = mean(perm_sharpes .>= observed_sharpe)

    # Two-sided p-value
    pvalue_two_sided = mean(abs.(perm_sharpes) .>= abs(observed_sharpe))

    sort!(perm_sharpes)
    perm_mean = mean(perm_sharpes)
    perm_std = std(perm_sharpes; corrected=true)

    # Effect size: how many stds above permutation mean
    effect_size = perm_std > eps(T) ? (observed_sharpe - perm_mean) / perm_std : zero(T)

    (observed_sharpe=observed_sharpe,
     pvalue=pvalue, pvalue_two_sided=pvalue_two_sided,
     perm_mean=perm_mean, perm_std=perm_std,
     effect_size=effect_size,
     significant_5pct=pvalue < T(0.05),
     significant_1pct=pvalue < T(0.01),
     n_permutations=n_permutations,
     percentile_rank=T(1) - pvalue)
end

"""
    combinatorial_purged_cv(return_matrix::AbstractMatrix{T};
                            n_groups::Int=10, purge_length::Int=5,
                            embargo_length::Int=5,
                            rng::AbstractRNG=Random.default_rng()) where T

Combinatorial Purged Cross-Validation (CPCV) for strategy selection.

Addresses leakage issues in standard cross-validation by:
1. Purging: removing observations near the IS/OOS boundary
2. Embargoing: preventing OOS observations that follow IS too closely
3. Combinatorial: testing all C(S, S/2) IS/OOS splits

Arguments:
- `return_matrix`: T x N matrix of strategy returns
- `n_groups`: Number of time partitions
- `purge_length`: Number of observations to purge at boundaries
- `embargo_length`: Embargo period after IS ends

Returns IS/OOS performance comparison and overfitting probability.
"""
function combinatorial_purged_cv(return_matrix::AbstractMatrix{T};
                                  n_groups::Int=10, purge_length::Int=5,
                                  embargo_length::Int=5,
                                  max_combinations::Int=500,
                                  rng::AbstractRNG=Random.default_rng()) where T<:Real
    ntime, nstrats = size(return_matrix)
    n_groups = min(n_groups, ntime ÷ 10)
    n_groups = max(n_groups, 4)
    s_half = n_groups ÷ 2

    group_size = ntime ÷ n_groups
    groups = Vector{UnitRange{Int}}(undef, n_groups)
    for g in 1:n_groups
        s = (g-1) * group_size + 1
        e = g == n_groups ? ntime : g * group_size
        groups[g] = s:e
    end

    # Generate combinations
    combos = _gen_combos(n_groups, s_half)
    if length(combos) > max_combinations
        combos = combos[randperm(rng, length(combos))[1:max_combinations]]
    end

    is_best_indices = Int[]
    oos_ranks = Int[]
    is_sharpes_all = zeros(T, length(combos), nstrats)
    oos_sharpes_all = zeros(T, length(combos), nstrats)

    for (ci, combo) in enumerate(combos)
        oos_groups = setdiff(1:n_groups, combo)

        # Collect IS and OOS indices with purging and embargo
        is_indices = Int[]
        oos_indices = Int[]

        for g in combo
            append!(is_indices, collect(groups[g]))
        end

        is_set = Set(is_indices)

        for g in oos_groups
            for idx in groups[g]
                # Check purge: not within purge_length of any IS boundary
                purged = false
                for is_g in combo
                    boundary_start = first(groups[is_g])
                    boundary_end = last(groups[is_g])
                    if abs(idx - boundary_start) <= purge_length ||
                       abs(idx - boundary_end) <= purge_length
                        purged = true
                        break
                    end
                end

                # Check embargo: not within embargo_length after IS ends
                embargoed = false
                for is_g in combo
                    is_end = last(groups[is_g])
                    if idx > is_end && idx <= is_end + embargo_length
                        embargoed = true
                        break
                    end
                end

                if !purged && !embargoed
                    push!(oos_indices, idx)
                end
            end
        end

        length(oos_indices) < 5 && continue

        # Compute IS and OOS Sharpes
        for j in 1:nstrats
            is_sharpes_all[ci, j] = _sharpe(return_matrix[is_indices, j])
            oos_sharpes_all[ci, j] = _sharpe(return_matrix[oos_indices, j])
        end

        best_is = argmax(is_sharpes_all[ci, :])
        push!(is_best_indices, best_is)
        oos_rank = sum(oos_sharpes_all[ci, best_is] .>= oos_sharpes_all[ci, :])
        push!(oos_ranks, oos_rank)
    end

    # PBO: fraction where IS-best ranks below median OOS
    n_combos = length(oos_ranks)
    pbo = n_combos > 0 ? mean(oos_ranks .<= nstrats ÷ 2) : one(T)

    # IS vs OOS Sharpe correlation across combinations
    if n_combos > 1
        is_best_sharpes = T[is_sharpes_all[c, is_best_indices[c]] for c in 1:n_combos]
        oos_best_sharpes = T[oos_sharpes_all[c, is_best_indices[c]] for c in 1:n_combos]
        is_oos_corr = _pearson_correlation(is_best_sharpes, oos_best_sharpes)
    else
        is_oos_corr = zero(T)
    end

    # Degradation
    mean_is = n_combos > 0 ? mean(T[maximum(is_sharpes_all[c, :]) for c in 1:n_combos]) : zero(T)
    mean_oos = n_combos > 0 ? mean(T[oos_sharpes_all[c, is_best_indices[c]] for c in 1:n_combos]) : zero(T)

    (pbo=pbo, n_combinations=n_combos,
     oos_ranks=oos_ranks, is_oos_correlation=is_oos_corr,
     mean_is_sharpe=mean_is, mean_oos_sharpe=mean_oos,
     degradation_ratio=abs(mean_is) > eps(T) ? mean_oos / mean_is : zero(T),
     overfitting_flag=pbo > T(0.5),
     purge_length=purge_length, embargo_length=embargo_length)
end

function _gen_combos(n::Int, k::Int)
    combos = Vector{Int}[]
    k > n && return combos
    combo = collect(1:k)
    while true
        push!(combos, copy(combo))
        i = k
        while i > 0 && combo[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        combo[i] += 1
        for j in (i+1):k
            combo[j] = combo[j-1] + 1
        end
    end
    combos
end

"""
    comprehensive_validation(returns::AbstractVector{T};
                             rng::AbstractRNG=Random.default_rng()) where T

Run all validation methods on a single strategy's returns.
"""
function comprehensive_validation(returns::AbstractVector{T};
                                   rng::AbstractRNG=Random.default_rng()) where T<:Real
    wf = walk_forward_consistency(returns)
    boot_stat = ts_bootstrap_ci(returns; method=:stationary, rng=rng)
    boot_circ = ts_bootstrap_ci(returns; method=:circular, rng=rng)
    perm = permutation_test(returns; rng=rng)

    overall_pass = wf.pass && boot_stat.significant && perm.significant_5pct

    Dict(
        "walk_forward" => wf,
        "bootstrap_stationary" => boot_stat,
        "bootstrap_circular" => boot_circ,
        "permutation_test" => perm,
        "overall_pass" => overall_pass,
        "original_sharpe" => _sharpe(returns),
    )
end

end # module BacktestValidation

# ============================================================================
# Section 8: Report Generator
# ============================================================================

"""
    ReportGenerator

Generate formatted reports from backtest analysis results.
Produces LaTeX-style tables (as strings), summary statistics,
and structured data for heatmaps and charts.
"""
module ReportGenerator

using Statistics

"""
    format_latex_table(headers::Vector{String}, data::AbstractMatrix;
                       caption::String="", label::String="",
                       alignment::String="", precision::Int=4) -> String

Generate a LaTeX-formatted table as a string.

Arguments:
- `headers`: Column header strings
- `data`: Matrix of values (rows x columns)
- `caption`: Table caption
- `label`: LaTeX label for cross-referencing
- `alignment`: Column alignment string (e.g., "lccc")
- `precision`: Decimal precision for floating point numbers
"""
function format_latex_table(headers::Vector{String}, data::AbstractMatrix;
                            caption::String="", label::String="",
                            alignment::String="", precision::Int=4)
    nrows, ncols = size(data)
    @assert length(headers) == ncols "Headers must match number of columns"

    if isempty(alignment)
        alignment = "l" * repeat("c", ncols - 1)
    end

    lines = String[]
    push!(lines, "\\begin{table}[htbp]")
    push!(lines, "\\centering")
    if !isempty(caption)
        push!(lines, "\\caption{$caption}")
    end
    if !isempty(label)
        push!(lines, "\\label{$label}")
    end
    push!(lines, "\\begin{tabular}{$alignment}")
    push!(lines, "\\toprule")

    # Header row
    header_str = join(headers, " & ") * " \\\\"
    push!(lines, header_str)
    push!(lines, "\\midrule")

    # Data rows
    for i in 1:nrows
        row_strs = String[]
        for j in 1:ncols
            val = data[i, j]
            if val isa AbstractFloat
                push!(row_strs, _format_number(val, precision))
            elseif val isa Integer
                push!(row_strs, string(val))
            else
                push!(row_strs, string(val))
            end
        end
        push!(lines, join(row_strs, " & ") * " \\\\")
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    join(lines, "\n")
end

function _format_number(x::Real, precision::Int)
    if abs(x) < 1e-10
        return "0." * repeat("0", precision)
    elseif abs(x) >= 1e6
        return string(round(x / 1e6, digits=2)) * "M"
    elseif abs(x) >= 1e3
        return string(round(x / 1e3, digits=2)) * "K"
    else
        return string(round(x, digits=precision))
    end
end

"""
    summary_statistics_report(sharpes::AbstractVector{T};
                               strategy_names::Vector{String}=String[]) where T -> String

Generate a text summary of Sharpe ratio distribution statistics.
"""
function summary_statistics_report(sharpes::AbstractVector{T};
                                    strategy_names::Vector{String}=String[]) where T<:Real
    n = length(sharpes)
    n == 0 && return "No strategies to report."

    sorted = sort(sharpes)
    mu = mean(sharpes)
    med = sorted[max(1, (n+1)÷2)]
    sigma = n > 1 ? std(sharpes; corrected=true) : zero(T)
    q25 = sorted[max(1, n÷4)]
    q75 = sorted[max(1, 3*n÷4)]

    # Skewness and kurtosis
    skew = _skewness(sharpes)
    kurt = _kurtosis(sharpes)

    lines = String[]
    push!(lines, "=" ^ 60)
    push!(lines, "  BACKTEST FARM SUMMARY STATISTICS")
    push!(lines, "=" ^ 60)
    push!(lines, "")
    push!(lines, "  Number of strategies:  $n")
    push!(lines, "  Mean Sharpe:           $(round(mu, digits=4))")
    push!(lines, "  Median Sharpe:         $(round(med, digits=4))")
    push!(lines, "  Std Dev:               $(round(sigma, digits=4))")
    push!(lines, "  Min Sharpe:            $(round(minimum(sharpes), digits=4))")
    push!(lines, "  Max Sharpe:            $(round(maximum(sharpes), digits=4))")
    push!(lines, "  Q25:                   $(round(q25, digits=4))")
    push!(lines, "  Q75:                   $(round(q75, digits=4))")
    push!(lines, "  IQR:                   $(round(q75 - q25, digits=4))")
    push!(lines, "  Skewness:              $(round(skew, digits=4))")
    push!(lines, "  Excess Kurtosis:       $(round(kurt, digits=4))")
    push!(lines, "")
    push!(lines, "  Sharpe > 0:   $(sum(sharpes .> zero(T))) ($(round(100*mean(sharpes .> zero(T)), digits=1))%)")
    push!(lines, "  Sharpe > 0.5: $(sum(sharpes .> T(0.5))) ($(round(100*mean(sharpes .> T(0.5)), digits=1))%)")
    push!(lines, "  Sharpe > 1.0: $(sum(sharpes .> one(T))) ($(round(100*mean(sharpes .> one(T)), digits=1))%)")
    push!(lines, "  Sharpe > 2.0: $(sum(sharpes .> T(2))) ($(round(100*mean(sharpes .> T(2)), digits=1))%)")
    push!(lines, "=" ^ 60)

    join(lines, "\n")
end

function _skewness(x::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 3 && return zero(T)
    mu = mean(x)
    s = std(x; corrected=true)
    s < eps(T) && return zero(T)
    mean((x .- mu) .^ 3) / s^3
end

function _kurtosis(x::AbstractVector{T}) where T<:Real
    n = length(x)
    n < 4 && return zero(T)
    mu = mean(x)
    s = std(x; corrected=true)
    s < eps(T) && return zero(T)
    mean((x .- mu) .^ 4) / s^4 - T(3)
end

"""
    top_strategies_report(rs; n_top::Int=10, pvalues::AbstractVector=Float64[],
                          dsr_pvalue::Float64=NaN) -> String

Generate a report of the top N strategies with statistical significance markers.
"""
function top_strategies_report(rs; n_top::Int=10, pvalues::AbstractVector=Float64[],
                               dsr_pvalue::Float64=NaN)
    results = rs.results
    n = length(results)
    n == 0 && return "No strategies to report."
    n_top = min(n_top, n)

    # Sort by Sharpe
    order = sortperm([r.metrics.sharpe for r in results], rev=true)

    lines = String[]
    push!(lines, "=" ^ 80)
    push!(lines, "  TOP $n_top STRATEGIES (out of $n)")
    push!(lines, "=" ^ 80)
    push!(lines, "")

    header = rpad("Rank", 6) * rpad("Name", 25) * rpad("Sharpe", 10) *
             rpad("Return", 10) * rpad("MaxDD", 10) * rpad("Sig", 10)
    push!(lines, header)
    push!(lines, "-" ^ 80)

    for rank in 1:n_top
        idx = order[rank]
        r = results[idx]
        name = length(r.config.name) > 22 ? r.config.name[1:22] * "..." : r.config.name

        sig_marker = ""
        if !isempty(pvalues) && idx <= length(pvalues)
            p = pvalues[idx]
            if p < 0.001
                sig_marker = "***"
            elseif p < 0.01
                sig_marker = "**"
            elseif p < 0.05
                sig_marker = "*"
            else
                sig_marker = "ns"
            end
        end

        line = rpad(string(rank), 6) *
               rpad(name, 25) *
               rpad(string(round(r.metrics.sharpe, digits=3)), 10) *
               rpad(string(round(r.metrics.total_return * 100, digits=1)) * "%", 10) *
               rpad(string(round(r.metrics.max_drawdown * 100, digits=1)) * "%", 10) *
               rpad(sig_marker, 10)
        push!(lines, line)
    end

    push!(lines, "-" ^ 80)

    if !isnan(dsr_pvalue)
        push!(lines, "")
        push!(lines, "  Deflated Sharpe Ratio p-value: $(round(dsr_pvalue, digits=4))")
        if dsr_pvalue < 0.05
            push!(lines, "  => Best strategy PASSES the DSR test (significant after multiple testing)")
        else
            push!(lines, "  => Best strategy FAILS the DSR test (likely overfitting)")
        end
    end

    push!(lines, "")
    push!(lines, "  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    push!(lines, "=" ^ 80)

    join(lines, "\n")
end

"""
    sensitivity_heatmap_data(param_names::Vector{String},
                             interaction_matrix::AbstractMatrix{T}) where T -> Dict

Prepare parameter sensitivity interaction data for heatmap visualization.

Returns a Dict with x_labels, y_labels, z_values, and formatted text annotations.
"""
function sensitivity_heatmap_data(param_names::Vector{String},
                                  interaction_matrix::AbstractMatrix{T}) where T<:Real
    p = length(param_names)
    @assert size(interaction_matrix) == (p, p)

    # Format text annotations
    text_annotations = Matrix{String}(undef, p, p)
    for i in 1:p
        for j in 1:p
            val = interaction_matrix[i, j]
            if abs(val) < T(0.001)
                text_annotations[i, j] = "-"
            else
                text_annotations[i, j] = string(round(val, digits=3))
            end
        end
    end

    # Find strongest interactions
    strong_interactions = Tuple{String, String, T}[]
    for i in 1:p
        for j in (i+1):p
            val = interaction_matrix[i, j]
            if abs(val) > T(0.01)
                push!(strong_interactions, (param_names[i], param_names[j], val))
            end
        end
    end
    sort!(strong_interactions, by=x -> abs(x[3]), rev=true)

    Dict(
        "x_labels" => param_names,
        "y_labels" => param_names,
        "z_values" => interaction_matrix,
        "text_annotations" => text_annotations,
        "strong_interactions" => strong_interactions,
    )
end

"""
    regime_performance_table(regime_sharpes::AbstractMatrix{T},
                             regime_labels::Vector{String},
                             strategy_names::Vector{String};
                             n_top::Int=10) where T -> String

Generate a regime performance table showing per-regime Sharpe for top strategies.
"""
function regime_performance_table(regime_sharpes::AbstractMatrix{T},
                                   regime_labels::Vector{String},
                                   strategy_names::Vector{String};
                                   n_top::Int=10) where T<:Real
    k, nstrats = size(regime_sharpes)
    n_top = min(n_top, nstrats)

    # Sort by average Sharpe across regimes
    avg_sharpes = vec(mean(regime_sharpes, dims=1))
    order = sortperm(avg_sharpes, rev=true)

    lines = String[]
    push!(lines, "=" ^ (30 + 12 * (k + 1)))
    push!(lines, "  REGIME PERFORMANCE TABLE")
    push!(lines, "=" ^ (30 + 12 * (k + 1)))
    push!(lines, "")

    # Header
    header = rpad("Strategy", 28)
    for label in regime_labels
        header *= rpad(label, 12)
    end
    header *= rpad("Average", 12)
    push!(lines, header)
    push!(lines, "-" ^ (30 + 12 * (k + 1)))

    for rank in 1:n_top
        idx = order[rank]
        name = length(strategy_names[idx]) > 25 ?
               strategy_names[idx][1:25] * "..." : strategy_names[idx]
        line = rpad(name, 28)
        for s in 1:k
            val = regime_sharpes[s, idx]
            marker = val > T(1) ? "+" : (val < zero(T) ? "-" : " ")
            line *= rpad(marker * string(round(val, digits=2)), 12)
        end
        line *= rpad(string(round(avg_sharpes[idx], digits=2)), 12)
        push!(lines, line)
    end

    push!(lines, "-" ^ (30 + 12 * (k + 1)))

    # Regime averages
    avg_line = rpad("REGIME AVERAGE", 28)
    for s in 1:k
        avg_line *= rpad(string(round(mean(regime_sharpes[s, :]), digits=2)), 12)
    end
    avg_line *= rpad(string(round(mean(avg_sharpes), digits=2)), 12)
    push!(lines, avg_line)

    push!(lines, "=" ^ (30 + 12 * (k + 1)))
    push!(lines, "  + = Sharpe > 1.0, - = Sharpe < 0")

    join(lines, "\n")
end

"""
    full_report(rs; rng=Random.default_rng()) -> String

Generate a comprehensive analysis report for a BacktestResultSet.
"""
function full_report(rs; pvalues::Vector{Float64}=Float64[],
                     dsr_pvalue::Float64=NaN)
    results = rs.results
    n = length(results)
    sharpes = Float64[r.metrics.sharpe for r in results]
    names = [r.config.name for r in results]

    report = String[]
    push!(report, summary_statistics_report(sharpes; strategy_names=names))
    push!(report, "")
    push!(report, top_strategies_report(rs; pvalues=pvalues, dsr_pvalue=dsr_pvalue))
    push!(report, "")

    # Distribution analysis
    push!(report, "SHARPE DISTRIBUTION ANALYSIS")
    push!(report, "-" ^ 40)
    n_bins = min(20, max(5, n ÷ 10))
    push!(report, "  Histogram bins: $n_bins")

    sorted = sort(sharpes)
    bin_width = (sorted[end] - sorted[1]) / n_bins
    if bin_width > 0
        for b in 1:n_bins
            lo = sorted[1] + (b-1) * bin_width
            hi = sorted[1] + b * bin_width
            count = sum(lo .<= sharpes .< hi)
            bar = repeat("#", min(count, 50))
            push!(report, "  [$(round(lo, digits=2)), $(round(hi, digits=2))): $bar ($count)")
        end
    end

    join(report, "\n")
end

end # module ReportGenerator

# ============================================================================
# Section 9: Gaussian Process Surrogate
# ============================================================================

"""
    GaussianProcessSurrogate

Gaussian Process regression for predicting Sharpe ratio from parameter vectors.
Includes acquisition functions for Bayesian optimization and active learning.
"""
module GaussianProcessSurrogate

using Statistics
using LinearAlgebra
using Random

"""
    KernelFunction{T<:Real}

Abstract type for GP kernel functions.
"""
abstract type KernelFunction{T<:Real} end

"""
    RBFKernel{T<:Real} <: KernelFunction{T}

Radial Basis Function (Squared Exponential) kernel.
k(x, x') = signal_variance * exp(-||x - x'||^2 / (2 * length_scale^2))
"""
mutable struct RBFKernel{T<:Real} <: KernelFunction{T}
    signal_variance::T
    length_scales::Vector{T}  # Per-dimension (ARD)

    function RBFKernel{T}(signal_variance::T, length_scales::Vector{T}) where T<:Real
        new{T}(signal_variance, length_scales)
    end
end

RBFKernel(signal_variance::T, length_scales::Vector{T}) where T<:Real =
    RBFKernel{T}(signal_variance, length_scales)

RBFKernel(p::Int; signal_variance::Float64=1.0, length_scale::Float64=1.0) =
    RBFKernel(signal_variance, fill(length_scale, p))

"""
    rbf_kernel(x1::AbstractVector{T}, x2::AbstractVector{T},
               kernel::RBFKernel{T}) where T

Evaluate the RBF kernel between two points.
"""
function rbf_kernel(x1::AbstractVector{T}, x2::AbstractVector{T},
                    kernel::RBFKernel{T}) where T<:Real
    @assert length(x1) == length(x2) == length(kernel.length_scales)
    sq_dist = zero(T)
    for d in eachindex(x1)
        sq_dist += ((x1[d] - x2[d]) / kernel.length_scales[d])^2
    end
    kernel.signal_variance * exp(-sq_dist / T(2))
end

"""
    MaternKernel{T<:Real} <: KernelFunction{T}

Matern 5/2 kernel for less smooth functions.
"""
mutable struct MaternKernel{T<:Real} <: KernelFunction{T}
    signal_variance::T
    length_scales::Vector{T}

    function MaternKernel{T}(signal_variance::T, length_scales::Vector{T}) where T<:Real
        new{T}(signal_variance, length_scales)
    end
end

MaternKernel(signal_variance::T, length_scales::Vector{T}) where T<:Real =
    MaternKernel{T}(signal_variance, length_scales)

function matern52_kernel(x1::AbstractVector{T}, x2::AbstractVector{T},
                          kernel::MaternKernel{T}) where T<:Real
    sq_dist = zero(T)
    for d in eachindex(x1)
        sq_dist += ((x1[d] - x2[d]) / kernel.length_scales[d])^2
    end
    r = sqrt(sq_dist)
    sqrt5_r = sqrt(T(5)) * r
    kernel.signal_variance * (one(T) + sqrt5_r + T(5)/T(3) * sq_dist) * exp(-sqrt5_r)
end

"""
    GaussianProcessModel{T<:Real}

Gaussian Process regression model.

Fields:
- `kernel`: Kernel function
- `noise_variance`: Observation noise variance
- `X_train`: Training inputs (N x p)
- `y_train`: Training outputs (N,)
- `K_inv`: Inverse of (K + noise*I) for prediction
- `alpha`: K_inv * y_train (cached for predictions)
- `fitted`: Whether the model has been fit
"""
mutable struct GaussianProcessModel{T<:Real}
    kernel::RBFKernel{T}
    noise_variance::T
    X_train::Matrix{T}
    y_train::Vector{T}
    K_inv::Matrix{T}
    alpha::Vector{T}
    L::Matrix{T}           # Cholesky factor
    log_marginal_likelihood::T
    fitted::Bool

    function GaussianProcessModel{T}(kernel::RBFKernel{T};
                                      noise_variance::T=T(0.01)) where T<:Real
        new{T}(kernel, noise_variance,
               zeros(T, 0, 0), T[], zeros(T, 0, 0), T[], zeros(T, 0, 0),
               T(-Inf), false)
    end
end

GaussianProcessModel(kernel::RBFKernel{T}; noise_variance::T=T(0.01)) where T<:Real =
    GaussianProcessModel{T}(kernel; noise_variance=noise_variance)

GaussianProcessModel(p::Int; signal_variance::Float64=1.0, length_scale::Float64=1.0,
                     noise_variance::Float64=0.01) =
    GaussianProcessModel(RBFKernel(p; signal_variance=signal_variance,
                                   length_scale=length_scale);
                         noise_variance=noise_variance)

"""
    _compute_kernel_matrix(X::AbstractMatrix{T}, kernel::RBFKernel{T}) where T

Compute the kernel matrix K[i,j] = k(X[i,:], X[j,:]).
"""
function _compute_kernel_matrix(X::AbstractMatrix{T}, kernel::RBFKernel{T}) where T<:Real
    n = size(X, 1)
    K = zeros(T, n, n)
    for i in 1:n
        K[i, i] = kernel.signal_variance
        for j in (i+1):n
            K[i, j] = rbf_kernel(X[i, :], X[j, :], kernel)
            K[j, i] = K[i, j]
        end
    end
    K
end

"""
    _compute_cross_kernel(X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                          kernel::RBFKernel{T}) where T

Compute cross-kernel matrix K[i,j] = k(X1[i,:], X2[j,:]).
"""
function _compute_cross_kernel(X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                                kernel::RBFKernel{T}) where T<:Real
    n1 = size(X1, 1)
    n2 = size(X2, 1)
    K = zeros(T, n1, n2)
    for i in 1:n1
        for j in 1:n2
            K[i, j] = rbf_kernel(X1[i, :], X2[j, :], kernel)
        end
    end
    K
end

"""
    fit_gp!(gp::GaussianProcessModel{T}, X::AbstractMatrix{T},
            y::AbstractVector{T}) where T

Fit the Gaussian Process model to training data.

Computes the kernel matrix, its Cholesky decomposition, and caches
the solution vector alpha = K^(-1) * y for fast predictions.
"""
function fit_gp!(gp::GaussianProcessModel{T}, X::AbstractMatrix{T},
                 y::AbstractVector{T}) where T<:Real
    n, p = size(X)
    @assert length(y) == n "X and y dimensions must match"
    @assert length(gp.kernel.length_scales) == p "Kernel dimension must match data"

    gp.X_train = copy(X)
    gp.y_train = copy(y)

    # Compute kernel matrix
    K = _compute_kernel_matrix(X, gp.kernel)

    # Add noise
    K_noisy = K + gp.noise_variance * I

    # Cholesky decomposition (with jitter for numerical stability)
    jitter = T(1e-8)
    local L
    for attempt in 1:5
        try
            L = cholesky(Symmetric(K_noisy + jitter * I)).L
            break
        catch
            jitter *= T(10)
            if attempt == 5
                # Fall back to eigendecomposition
                eig = eigen(Symmetric(K_noisy))
                eig_vals = max.(eig.values, eps(T))
                K_noisy = eig.vectors * Diagonal(eig_vals) * eig.vectors'
                L = cholesky(Symmetric(K_noisy)).L
            end
        end
    end

    gp.L = Matrix(L)
    gp.alpha = L' \ (L \ y)
    gp.K_inv = L' \ (L \ Matrix{T}(I, n, n))

    # Log marginal likelihood
    # log p(y|X) = -0.5 * y' * alpha - sum(log(diag(L))) - n/2 * log(2pi)
    gp.log_marginal_likelihood = -T(0.5) * dot(y, gp.alpha) -
                                  sum(log.(diag(L))) -
                                  T(n) / T(2) * log(T(2) * T(pi))

    gp.fitted = true
    gp
end

"""
    predict_gp(gp::GaussianProcessModel{T}, X_test::AbstractMatrix{T};
               return_std::Bool=true) where T

Predict using the fitted GP at test points.

Returns (mean, std) predictions. The mean is the posterior mean and std
is the posterior standard deviation (epistemic uncertainty).
"""
function predict_gp(gp::GaussianProcessModel{T}, X_test::AbstractMatrix{T};
                    return_std::Bool=true) where T<:Real
    @assert gp.fitted "Model must be fitted first"

    n_test = size(X_test, 1)

    # Cross-kernel between test and training points
    K_star = _compute_cross_kernel(X_test, gp.X_train, gp.kernel)

    # Posterior mean: mu* = K* * alpha
    mu = K_star * gp.alpha

    if !return_std
        return (mean=mu, std=zeros(T, n_test))
    end

    # Posterior variance: var* = k** - K* * K_inv * K*'
    sigma = zeros(T, n_test)
    for i in 1:n_test
        k_star_star = gp.kernel.signal_variance  # k(x*, x*)
        v = gp.L \ K_star[i, :]
        sigma[i] = sqrt(max(k_star_star - dot(v, v), eps(T)))
    end

    (mean=mu, std=sigma)
end

"""
    predict_gp(gp::GaussianProcessModel{T}, x_test::AbstractVector{T}) where T

Predict at a single test point.
"""
function predict_gp(gp::GaussianProcessModel{T}, x_test::AbstractVector{T}) where T<:Real
    result = predict_gp(gp, reshape(x_test, 1, :))
    (mean=result.mean[1], std=result.std[1])
end

"""
    expected_improvement(gp::GaussianProcessModel{T}, x::AbstractVector{T};
                         f_best::Union{T, Nothing}=nothing,
                         xi::T=T(0.01)) where T

Compute Expected Improvement acquisition function at point x.

EI(x) = (f_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
where z = (f_best - mu(x) - xi) / sigma(x)

Arguments:
- `gp`: Fitted GP model
- `x`: Point to evaluate
- `f_best`: Best observed value (default: max of training y)
- `xi`: Exploration-exploitation trade-off parameter
"""
function expected_improvement(gp::GaussianProcessModel{T}, x::AbstractVector{T};
                               f_best::Union{T, Nothing}=nothing,
                               xi::T=T(0.01)) where T<:Real
    @assert gp.fitted "Model must be fitted first"

    pred = predict_gp(gp, x)
    mu = pred.mean
    sigma = pred.std

    if isnothing(f_best)
        f_best = maximum(gp.y_train)
    end

    sigma < eps(T) && return zero(T)

    z = (f_best - mu - xi) / sigma

    # We want to MAXIMIZE, but EI is typically for minimization
    # For maximization: EI(x) = (mu - f_best - xi) * Phi(z_max) + sigma * phi(z_max)
    z_max = (mu - f_best - xi) / sigma
    ei = (mu - f_best - xi) * _normal_cdf(z_max) + sigma * _normal_pdf(z_max)

    max(ei, zero(T))
end

"""
    expected_improvement(gp::GaussianProcessModel{T}, X::AbstractMatrix{T};
                         f_best::Union{T, Nothing}=nothing,
                         xi::T=T(0.01)) where T

Compute Expected Improvement at multiple points.
"""
function expected_improvement(gp::GaussianProcessModel{T}, X::AbstractMatrix{T};
                               f_best::Union{T, Nothing}=nothing,
                               xi::T=T(0.01)) where T<:Real
    n = size(X, 1)
    ei = zeros(T, n)
    for i in 1:n
        ei[i] = expected_improvement(gp, X[i, :]; f_best=f_best, xi=xi)
    end
    ei
end

function _normal_cdf(x::T) where T<:Real
    a1 = T(0.254829592)
    a2 = T(-0.284496736)
    a3 = T(1.421413741)
    a4 = T(-1.453152027)
    a5 = T(1.061405429)
    p  = T(0.3275911)
    sign_x = x < zero(T) ? -one(T) : one(T)
    x_abs = abs(x)
    t = one(T) / (one(T) + p * x_abs)
    y = one(T) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x_abs^2 / 2)
    T(0.5) * (one(T) + sign_x * y)
end

function _normal_pdf(x::T) where T<:Real
    T(1.0 / sqrt(2.0 * pi)) * exp(-x^2 / 2)
end

"""
    UCB(gp::GaussianProcessModel{T}, x::AbstractVector{T};
        beta::T=T(2.0)) where T

Upper Confidence Bound acquisition function.
UCB(x) = mu(x) + beta * sigma(x)
"""
function UCB(gp::GaussianProcessModel{T}, x::AbstractVector{T};
             beta::T=T(2.0)) where T<:Real
    pred = predict_gp(gp, x)
    pred.mean + beta * pred.std
end

"""
    probability_of_improvement(gp::GaussianProcessModel{T}, x::AbstractVector{T};
                                f_best::Union{T, Nothing}=nothing,
                                xi::T=T(0.01)) where T

Probability of Improvement acquisition function.
PI(x) = Phi((mu(x) - f_best - xi) / sigma(x))
"""
function probability_of_improvement(gp::GaussianProcessModel{T}, x::AbstractVector{T};
                                     f_best::Union{T, Nothing}=nothing,
                                     xi::T=T(0.01)) where T<:Real
    pred = predict_gp(gp, x)
    if isnothing(f_best)
        f_best = maximum(gp.y_train)
    end
    pred.std < eps(T) && return zero(T)
    z = (pred.mean - f_best - xi) / pred.std
    _normal_cdf(z)
end

"""
    bayesian_optimization_step(gp::GaussianProcessModel{T},
                                bounds::Matrix{T};
                                n_candidates::Int=1000,
                                acquisition::Symbol=:ei,
                                xi::T=T(0.01), beta::T=T(2.0),
                                rng::AbstractRNG=Random.default_rng()) where T

Perform one step of Bayesian optimization:
1. Generate candidate points
2. Compute acquisition function at each candidate
3. Return the point with highest acquisition value

Arguments:
- `gp`: Fitted GP model
- `bounds`: p x 2 matrix of [lower, upper] bounds per parameter
- `n_candidates`: Number of random candidate points to evaluate
- `acquisition`: Acquisition function (:ei, :ucb, :pi)
- `xi`: EI/PI exploration parameter
- `beta`: UCB exploration parameter

Returns the suggested next point and its acquisition value.
"""
function bayesian_optimization_step(gp::GaussianProcessModel{T},
                                     bounds::Matrix{T};
                                     n_candidates::Int=1000,
                                     acquisition::Symbol=:ei,
                                     xi::T=T(0.01), beta::T=T(2.0),
                                     rng::AbstractRNG=Random.default_rng()) where T<:Real
    @assert gp.fitted "Model must be fitted first"
    p = size(bounds, 1)

    # Generate random candidates within bounds
    X_candidates = zeros(T, n_candidates, p)
    for i in 1:n_candidates
        for j in 1:p
            X_candidates[i, j] = bounds[j, 1] + rand(rng) * (bounds[j, 2] - bounds[j, 1])
        end
    end

    # Also add training points perturbed slightly (local search)
    n_local = min(size(gp.X_train, 1) * 5, n_candidates ÷ 2)
    X_local = zeros(T, n_local, p)
    for i in 1:n_local
        base_idx = rand(rng, 1:size(gp.X_train, 1))
        for j in 1:p
            scale = (bounds[j, 2] - bounds[j, 1]) * T(0.1)
            X_local[i, j] = clamp(gp.X_train[base_idx, j] + randn(rng) * scale,
                                  bounds[j, 1], bounds[j, 2])
        end
    end

    X_all = vcat(X_candidates, X_local)
    n_all = size(X_all, 1)

    # Evaluate acquisition function
    acq_values = zeros(T, n_all)
    f_best = maximum(gp.y_train)

    if acquisition == :ei
        acq_values = expected_improvement(gp, X_all; f_best=f_best, xi=xi)
    elseif acquisition == :ucb
        for i in 1:n_all
            acq_values[i] = UCB(gp, X_all[i, :]; beta=beta)
        end
    elseif acquisition == :pi
        for i in 1:n_all
            acq_values[i] = probability_of_improvement(gp, X_all[i, :]; f_best=f_best, xi=xi)
        end
    else
        error("Unknown acquisition function: $acquisition")
    end

    best_idx = argmax(acq_values)
    best_point = X_all[best_idx, :]

    # GP prediction at the suggested point
    pred = predict_gp(gp, best_point)

    (next_point=best_point,
     acquisition_value=acq_values[best_idx],
     predicted_mean=pred.mean,
     predicted_std=pred.std,
     f_best=f_best,
     acquisition_function=acquisition)
end

"""
    active_learning_suggest(gp::GaussianProcessModel{T},
                            bounds::Matrix{T};
                            n_suggestions::Int=5,
                            n_candidates::Int=2000,
                            strategy::Symbol=:uncertainty,
                            rng::AbstractRNG=Random.default_rng()) where T

Suggest which parameter combinations to backtest next using active learning.

Strategies:
- :uncertainty - Points with highest predictive uncertainty
- :ei - Points with highest Expected Improvement
- :exploration - Space-filling design targeting underexplored regions
- :mixed - Combination of EI and uncertainty

Returns a list of suggested parameter vectors, ordered by priority.
"""
function active_learning_suggest(gp::GaussianProcessModel{T},
                                  bounds::Matrix{T};
                                  n_suggestions::Int=5,
                                  n_candidates::Int=2000,
                                  strategy::Symbol=:uncertainty,
                                  rng::AbstractRNG=Random.default_rng()) where T<:Real
    @assert gp.fitted "Model must be fitted first"
    p = size(bounds, 1)

    # Generate candidates
    X_candidates = zeros(T, n_candidates, p)
    for i in 1:n_candidates
        for j in 1:p
            X_candidates[i, j] = bounds[j, 1] + rand(rng) * (bounds[j, 2] - bounds[j, 1])
        end
    end

    # Predict at all candidates
    preds = predict_gp(gp, X_candidates)

    # Score candidates
    scores = zeros(T, n_candidates)

    if strategy == :uncertainty
        scores .= preds.std
    elseif strategy == :ei
        f_best = maximum(gp.y_train)
        scores = expected_improvement(gp, X_candidates; f_best=f_best)
    elseif strategy == :exploration
        # Distance to nearest training point
        for i in 1:n_candidates
            min_dist = T(Inf)
            for j in 1:size(gp.X_train, 1)
                d = zero(T)
                for dim in 1:p
                    d += ((X_candidates[i, dim] - gp.X_train[j, dim]) /
                          (bounds[dim, 2] - bounds[dim, 1]))^2
                end
                min_dist = min(min_dist, sqrt(d))
            end
            scores[i] = min_dist
        end
    elseif strategy == :mixed
        # 50% EI + 50% uncertainty (normalized)
        f_best = maximum(gp.y_train)
        ei_scores = expected_improvement(gp, X_candidates; f_best=f_best)
        max_ei = maximum(ei_scores)
        max_std = maximum(preds.std)
        if max_ei > eps(T) && max_std > eps(T)
            scores .= T(0.5) .* ei_scores ./ max_ei .+ T(0.5) .* preds.std ./ max_std
        else
            scores .= preds.std
        end
    else
        error("Unknown strategy: $strategy")
    end

    # Select top suggestions (with diversity: reject points too close to each other)
    suggestions = Vector{Vector{T}}()
    pred_means = T[]
    pred_stds = T[]
    acq_scores = T[]
    used = falses(n_candidates)
    min_dist_threshold = T(0.1)  # Minimum normalized distance between suggestions

    order = sortperm(scores, rev=true)

    for idx in order
        length(suggestions) >= n_suggestions && break

        point = X_candidates[idx, :]

        # Check distance to already selected points
        too_close = false
        for existing in suggestions
            d = zero(T)
            for dim in 1:p
                range = bounds[dim, 2] - bounds[dim, 1]
                d += ((point[dim] - existing[dim]) / max(range, eps(T)))^2
            end
            if sqrt(d) < min_dist_threshold
                too_close = true
                break
            end
        end

        if !too_close
            push!(suggestions, point)
            push!(pred_means, preds.mean[idx])
            push!(pred_stds, preds.std[idx])
            push!(acq_scores, scores[idx])
        end
    end

    (suggestions=suggestions,
     predicted_means=pred_means,
     predicted_stds=pred_stds,
     acquisition_scores=acq_scores,
     strategy=strategy,
     n_training_points=size(gp.X_train, 1))
end

"""
    optimize_hyperparameters!(gp::GaussianProcessModel{T};
                               n_restarts::Int=5,
                               max_iter::Int=50,
                               rng::AbstractRNG=Random.default_rng()) where T

Optimize GP hyperparameters (length scales, signal variance, noise variance)
by maximizing the log marginal likelihood.

Uses random restarts with gradient-free optimization (coordinate descent).
"""
function optimize_hyperparameters!(gp::GaussianProcessModel{T};
                                    n_restarts::Int=5,
                                    max_iter::Int=50,
                                    rng::AbstractRNG=Random.default_rng()) where T<:Real
    @assert gp.fitted "Model must be fitted with initial data first"

    p = length(gp.kernel.length_scales)
    n_params = p + 2  # length_scales + signal_variance + noise_variance

    best_lml = gp.log_marginal_likelihood
    best_ls = copy(gp.kernel.length_scales)
    best_sv = gp.kernel.signal_variance
    best_nv = gp.noise_variance

    for restart in 1:n_restarts
        # Random initialization
        if restart > 1
            for d in 1:p
                gp.kernel.length_scales[d] = exp(randn(rng)) * best_ls[d]
            end
            gp.kernel.signal_variance = exp(randn(rng) * T(0.5)) * best_sv
            gp.noise_variance = exp(randn(rng) * T(0.5)) * best_nv
        end

        # Coordinate descent on log-transformed parameters
        for iter in 1:max_iter
            improved = false

            # Optimize each length scale
            for d in 1:p
                orig = gp.kernel.length_scales[d]
                fit_gp!(gp, gp.X_train, gp.y_train)
                base_lml = gp.log_marginal_likelihood

                for factor in [T(1.5), T(0.67), T(1.2), T(0.83)]
                    gp.kernel.length_scales[d] = orig * factor
                    fit_gp!(gp, gp.X_train, gp.y_train)
                    if gp.log_marginal_likelihood > base_lml
                        base_lml = gp.log_marginal_likelihood
                        orig = gp.kernel.length_scales[d]
                        improved = true
                    else
                        gp.kernel.length_scales[d] = orig
                    end
                end
            end

            # Optimize signal variance
            orig_sv = gp.kernel.signal_variance
            for factor in [T(1.5), T(0.67)]
                gp.kernel.signal_variance = orig_sv * factor
                fit_gp!(gp, gp.X_train, gp.y_train)
                if gp.log_marginal_likelihood > best_lml
                    orig_sv = gp.kernel.signal_variance
                    improved = true
                else
                    gp.kernel.signal_variance = orig_sv
                end
            end

            # Optimize noise variance
            orig_nv = gp.noise_variance
            for factor in [T(1.5), T(0.67), T(2.0), T(0.5)]
                gp.noise_variance = orig_nv * factor
                fit_gp!(gp, gp.X_train, gp.y_train)
                if gp.log_marginal_likelihood > best_lml
                    orig_nv = gp.noise_variance
                    improved = true
                else
                    gp.noise_variance = orig_nv
                end
            end

            !improved && break
        end

        fit_gp!(gp, gp.X_train, gp.y_train)
        if gp.log_marginal_likelihood > best_lml
            best_lml = gp.log_marginal_likelihood
            best_ls .= gp.kernel.length_scales
            best_sv = gp.kernel.signal_variance
            best_nv = gp.noise_variance
        end
    end

    # Restore best
    gp.kernel.length_scales .= best_ls
    gp.kernel.signal_variance = best_sv
    gp.noise_variance = best_nv
    fit_gp!(gp, gp.X_train, gp.y_train)

    (log_marginal_likelihood=gp.log_marginal_likelihood,
     length_scales=copy(gp.kernel.length_scales),
     signal_variance=gp.kernel.signal_variance,
     noise_variance=gp.noise_variance)
end

"""
    bayesian_optimization_loop(eval_fn::Function, bounds::Matrix{T};
                                n_initial::Int=10, n_iterations::Int=50,
                                acquisition::Symbol=:ei,
                                noise_variance::T=T(0.01),
                                optimize_hypers::Bool=true,
                                rng::AbstractRNG=Random.default_rng()) where T

Run a full Bayesian optimization loop:
1. Initialize with random points
2. Fit GP
3. Select next point via acquisition function
4. Evaluate and update
5. Repeat

Arguments:
- `eval_fn`: Function mapping parameter vector -> scalar metric (e.g., Sharpe)
- `bounds`: p x 2 matrix of parameter bounds
- `n_initial`: Number of initial random evaluations
- `n_iterations`: Number of BO iterations
- `acquisition`: Acquisition function to use

Returns optimization history and best found point.
"""
function bayesian_optimization_loop(eval_fn::Function, bounds::Matrix{T};
                                     n_initial::Int=10, n_iterations::Int=50,
                                     acquisition::Symbol=:ei,
                                     noise_variance::T=T(0.01),
                                     optimize_hypers::Bool=true,
                                     rng::AbstractRNG=Random.default_rng()) where T<:Real
    p = size(bounds, 1)

    # Initial random design (Latin Hypercube-like)
    X_history = zeros(T, n_initial, p)
    y_history = zeros(T, n_initial)

    for i in 1:n_initial
        for j in 1:p
            # Stratified random: divide range into n_initial strata
            lo = bounds[j, 1] + (i-1) * (bounds[j, 2] - bounds[j, 1]) / n_initial
            hi = bounds[j, 1] + i * (bounds[j, 2] - bounds[j, 1]) / n_initial
            X_history[i, j] = lo + rand(rng) * (hi - lo)
        end
        y_history[i] = eval_fn(X_history[i, :])
    end

    # Initialize GP
    gp = GaussianProcessModel(RBFKernel(p; signal_variance=var(y_history),
                                         length_scale=1.0);
                               noise_variance=noise_variance)
    fit_gp!(gp, X_history, y_history)

    best_y_trace = T[maximum(y_history)]
    best_x = X_history[argmax(y_history), :]

    for iter in 1:n_iterations
        # Optionally optimize hyperparameters
        if optimize_hypers && iter % 5 == 0
            optimize_hyperparameters!(gp; rng=rng, n_restarts=2)
        end

        # Get next point
        step = bayesian_optimization_step(gp, bounds; acquisition=acquisition, rng=rng)
        next_x = step.next_point

        # Evaluate
        next_y = eval_fn(next_x)

        # Update data
        X_history = vcat(X_history, next_x')
        push!(y_history, next_y)

        # Update best
        if next_y > best_y_trace[end]
            push!(best_y_trace, next_y)
            best_x = next_x
        else
            push!(best_y_trace, best_y_trace[end])
        end

        # Refit GP
        fit_gp!(gp, X_history, y_history)
    end

    (best_x=best_x, best_y=best_y_trace[end],
     X_history=X_history, y_history=y_history,
     best_y_trace=best_y_trace,
     gp=gp, n_evaluations=n_initial + n_iterations)
end

"""
    gp_model_diagnostics(gp::GaussianProcessModel{T}) where T

Compute diagnostic statistics for a fitted GP model.

Returns leave-one-out cross-validation error, mean standardized residual,
and calibration metrics.
"""
function gp_model_diagnostics(gp::GaussianProcessModel{T}) where T<:Real
    @assert gp.fitted "Model must be fitted"

    n = length(gp.y_train)

    # LOO predictions (analytic from K_inv)
    loo_means = zeros(T, n)
    loo_vars = zeros(T, n)

    for i in 1:n
        loo_vars[i] = one(T) / max(gp.K_inv[i, i], eps(T))
        loo_means[i] = gp.y_train[i] - gp.alpha[i] / max(gp.K_inv[i, i], eps(T))
    end

    loo_errors = gp.y_train .- loo_means
    loo_mse = mean(loo_errors .^ 2)
    loo_rmse = sqrt(loo_mse)

    # Standardized residuals
    std_residuals = loo_errors ./ sqrt.(max.(loo_vars, eps(T)))
    mean_std_resid = mean(abs.(std_residuals))

    # Calibration: fraction of observations within 1, 2, 3 sigma
    within_1sigma = mean(abs.(std_residuals) .< one(T))
    within_2sigma = mean(abs.(std_residuals) .< T(2))
    within_3sigma = mean(abs.(std_residuals) .< T(3))

    # Expected: 68.3%, 95.4%, 99.7%
    cal_1 = within_1sigma - T(0.683)
    cal_2 = within_2sigma - T(0.954)

    (loo_rmse=loo_rmse, loo_mse=loo_mse,
     mean_abs_standardized_residual=mean_std_resid,
     within_1sigma=within_1sigma,
     within_2sigma=within_2sigma,
     within_3sigma=within_3sigma,
     calibration_1sigma_error=cal_1,
     calibration_2sigma_error=cal_2,
     log_marginal_likelihood=gp.log_marginal_likelihood,
     n_training=n,
     well_calibrated=abs(cal_1) < T(0.15) && abs(cal_2) < T(0.1))
end

end # module GaussianProcessSurrogate

# ============================================================================
# Section 10: Top-level convenience API
# ============================================================================

# Re-export submodule functions via delegation

"""
    bonferroni_correction(pvalues; alpha=0.05)

Apply Bonferroni correction. See `MultipleTesting.bonferroni_correction`.
"""
bonferroni_correction(pv; kwargs...) = MultipleTesting.bonferroni_correction(pv; kwargs...)

"""
    holm_stepdown(pvalues; alpha=0.05)

Apply Holm step-down procedure. See `MultipleTesting.holm_stepdown`.
"""
holm_stepdown(pv; kwargs...) = MultipleTesting.holm_stepdown(pv; kwargs...)

"""
    benjamini_hochberg(pvalues; alpha=0.05)

Apply Benjamini-Hochberg FDR correction. See `MultipleTesting.benjamini_hochberg`.
"""
benjamini_hochberg(pv; kwargs...) = MultipleTesting.benjamini_hochberg(pv; kwargs...)

"""
    deflated_sharpe_ratio(args...; kwargs...)

Compute Deflated Sharpe Ratio. See `MultipleTesting.deflated_sharpe_ratio`.
"""
deflated_sharpe_ratio(args...; kwargs...) = MultipleTesting.deflated_sharpe_ratio(args...; kwargs...)

"""
    minimum_backtest_length(sr; kwargs...)

Compute minimum backtest length. See `MultipleTesting.minimum_backtest_length`.
"""
minimum_backtest_length(sr; kwargs...) = MultipleTesting.minimum_backtest_length(sr; kwargs...)

"""
    probability_of_backtest_overfitting(returns_matrix; kwargs...)

Compute PBO. See `MultipleTesting.probability_of_backtest_overfitting`.
"""
probability_of_backtest_overfitting(rm; kwargs...) =
    MultipleTesting.probability_of_backtest_overfitting(rm; kwargs...)

"""
    kernel_density_estimate(data; kwargs...)

KDE of distribution. See `AlphaLandscapeAnalysis.kernel_density_estimate`.
"""
kernel_density_estimate(d; kwargs...) = AlphaLandscapeAnalysis.kernel_density_estimate(d; kwargs...)

"""
    pca_returns(return_matrix; kwargs...)

PCA on strategy returns. See `AlphaLandscapeAnalysis.pca_returns`.
"""
pca_returns(rm; kwargs...) = AlphaLandscapeAnalysis.pca_returns(rm; kwargs...)

"""
    kmeans_strategies(return_matrix, k; kwargs...)

K-means clustering. See `AlphaLandscapeAnalysis.kmeans_strategies`.
"""
kmeans_strategies(rm, k; kwargs...) = AlphaLandscapeAnalysis.kmeans_strategies(rm, k; kwargs...)

"""
    factor_decomposition(return_matrix; kwargs...)

Factor decomposition. See `AlphaLandscapeAnalysis.factor_decomposition`.
"""
factor_decomposition(rm; kwargs...) = AlphaLandscapeAnalysis.factor_decomposition(rm; kwargs...)

"""
    alpha_capacity_estimate(returns, aum_levels; kwargs...)

Alpha capacity estimation. See `AlphaLandscapeAnalysis.alpha_capacity_estimate`.
"""
alpha_capacity_estimate(r, a; kwargs...) = AlphaLandscapeAnalysis.alpha_capacity_estimate(r, a; kwargs...)

"""
    sobol_first_order(param_matrix, responses; kwargs...)

Sobol first-order indices. See `ParameterSensitivity.sobol_first_order`.
"""
sobol_first_order(pm, r; kwargs...) = ParameterSensitivity.sobol_first_order(pm, r; kwargs...)

"""
    sobol_total_order(param_matrix, responses; kwargs...)

Sobol total-order indices. See `ParameterSensitivity.sobol_total_order`.
"""
sobol_total_order(pm, r; kwargs...) = ParameterSensitivity.sobol_total_order(pm, r; kwargs...)

"""
    morris_elementary_effects(eval_fn, bounds, param_names; kwargs...)

Morris screening. See `ParameterSensitivity.morris_elementary_effects`.
"""
morris_elementary_effects(f, b, p; kwargs...) =
    ParameterSensitivity.morris_elementary_effects(f, b, p; kwargs...)

"""
    local_sensitivity(param_matrix, responses; kwargs...)

Local sensitivity analysis. See `ParameterSensitivity.local_sensitivity`.
"""
local_sensitivity(pm, r; kwargs...) = ParameterSensitivity.local_sensitivity(pm, r; kwargs...)

"""
    parameter_interactions(param_matrix, responses; kwargs...)

Parameter interaction analysis. See `ParameterSensitivity.parameter_interactions`.
"""
parameter_interactions(pm, r; kwargs...) = ParameterSensitivity.parameter_interactions(pm, r; kwargs...)

"""
    fit_hmm(observations; kwargs...)

Fit Hidden Markov Model. See `RegimeConditionalAnalysis.fit_hmm`.
"""
fit_hmm(obs; kwargs...) = RegimeConditionalAnalysis.fit_hmm(obs; kwargs...)

"""
    detect_regimes(observations, params)

Viterbi decoding. See `RegimeConditionalAnalysis.detect_regimes`.
"""
detect_regimes(obs, params) = RegimeConditionalAnalysis.detect_regimes(obs, params)

"""
    per_regime_sharpe(return_matrix, states; kwargs...)

Per-regime Sharpe. See `RegimeConditionalAnalysis.per_regime_sharpe`.
"""
per_regime_sharpe(rm, s; kwargs...) = RegimeConditionalAnalysis.per_regime_sharpe(rm, s; kwargs...)

"""
    classify_specialist_generalist(regime_sharpes; kwargs...)

Strategy classification. See `RegimeConditionalAnalysis.classify_specialist_generalist`.
"""
classify_specialist_generalist(rs; kwargs...) =
    RegimeConditionalAnalysis.classify_specialist_generalist(rs; kwargs...)

"""
    regime_transition_cost(return_matrix, states, specialist_map; kwargs...)

Regime transition cost. See `RegimeConditionalAnalysis.regime_transition_cost`.
"""
regime_transition_cost(rm, s, sm; kwargs...) =
    RegimeConditionalAnalysis.regime_transition_cost(rm, s, sm; kwargs...)

"""
    markowitz_mvo(return_matrix; kwargs...)

Mean-Variance Optimization. See `OptimalPortfolioConstruction.markowitz_mvo`.
"""
markowitz_mvo(rm; kwargs...) = OptimalPortfolioConstruction.markowitz_mvo(rm; kwargs...)

"""
    risk_parity_portfolio(return_matrix; kwargs...)

Risk parity. See `OptimalPortfolioConstruction.risk_parity_portfolio`.
"""
risk_parity_portfolio(rm; kwargs...) = OptimalPortfolioConstruction.risk_parity_portfolio(rm; kwargs...)

"""
    max_diversification_portfolio(return_matrix; kwargs...)

Max diversification. See `OptimalPortfolioConstruction.max_diversification_portfolio`.
"""
max_diversification_portfolio(rm; kwargs...) =
    OptimalPortfolioConstruction.max_diversification_portfolio(rm; kwargs...)

"""
    kelly_criterion(return_matrix; kwargs...)

Kelly criterion. See `OptimalPortfolioConstruction.kelly_criterion`.
"""
kelly_criterion(rm; kwargs...) = OptimalPortfolioConstruction.kelly_criterion(rm; kwargs...)

"""
    black_litterman(return_matrix, views; kwargs...)

Black-Litterman. See `OptimalPortfolioConstruction.black_litterman`.
"""
black_litterman(rm, v; kwargs...) = OptimalPortfolioConstruction.black_litterman(rm, v; kwargs...)

"""
    walk_forward_consistency(returns; kwargs...)

Walk-forward analysis. See `BacktestValidation.walk_forward_consistency`.
"""
walk_forward_consistency(r; kwargs...) = BacktestValidation.walk_forward_consistency(r; kwargs...)

"""
    ts_bootstrap_ci(returns; kwargs...)

Bootstrap confidence intervals. See `BacktestValidation.ts_bootstrap_ci`.
"""
ts_bootstrap_ci(r; kwargs...) = BacktestValidation.ts_bootstrap_ci(r; kwargs...)

"""
    permutation_test(returns; kwargs...)

Permutation test. See `BacktestValidation.permutation_test`.
"""
permutation_test(r; kwargs...) = BacktestValidation.permutation_test(r; kwargs...)

"""
    combinatorial_purged_cv(return_matrix; kwargs...)

CPCV. See `BacktestValidation.combinatorial_purged_cv`.
"""
combinatorial_purged_cv(rm; kwargs...) = BacktestValidation.combinatorial_purged_cv(rm; kwargs...)

"""
    format_latex_table(headers, data; kwargs...)

LaTeX table. See `ReportGenerator.format_latex_table`.
"""
format_latex_table(h, d; kwargs...) = ReportGenerator.format_latex_table(h, d; kwargs...)

"""
    summary_statistics_report(sharpes; kwargs...)

Summary report. See `ReportGenerator.summary_statistics_report`.
"""
summary_statistics_report(s; kwargs...) = ReportGenerator.summary_statistics_report(s; kwargs...)

"""
    top_strategies_report(rs; kwargs...)

Top strategies report. See `ReportGenerator.top_strategies_report`.
"""
top_strategies_report(rs; kwargs...) = ReportGenerator.top_strategies_report(rs; kwargs...)

"""
    sensitivity_heatmap_data(param_names, interaction_matrix)

Heatmap data. See `ReportGenerator.sensitivity_heatmap_data`.
"""
sensitivity_heatmap_data(pn, im) = ReportGenerator.sensitivity_heatmap_data(pn, im)

"""
    regime_performance_table(regime_sharpes, regime_labels, strategy_names; kwargs...)

Regime table. See `ReportGenerator.regime_performance_table`.
"""
regime_performance_table(rs, rl, sn; kwargs...) =
    ReportGenerator.regime_performance_table(rs, rl, sn; kwargs...)

# GP exports
const GaussianProcessModel = GaussianProcessSurrogate.GaussianProcessModel
const RBFKernel = GaussianProcessSurrogate.RBFKernel

"""
    fit_gp!(gp, X, y)

Fit GP model. See `GaussianProcessSurrogate.fit_gp!`.
"""
fit_gp!(gp, X, y) = GaussianProcessSurrogate.fit_gp!(gp, X, y)

"""
    predict_gp(gp, X; kwargs...)

GP prediction. See `GaussianProcessSurrogate.predict_gp`.
"""
predict_gp(gp, X; kwargs...) = GaussianProcessSurrogate.predict_gp(gp, X; kwargs...)

"""
    rbf_kernel(x1, x2, kernel)

RBF kernel evaluation. See `GaussianProcessSurrogate.rbf_kernel`.
"""
rbf_kernel(x1, x2, k) = GaussianProcessSurrogate.rbf_kernel(x1, x2, k)

"""
    expected_improvement(gp, x; kwargs...)

Expected Improvement. See `GaussianProcessSurrogate.expected_improvement`.
"""
expected_improvement(gp, x; kwargs...) = GaussianProcessSurrogate.expected_improvement(gp, x; kwargs...)

"""
    bayesian_optimization_step(gp, bounds; kwargs...)

BO step. See `GaussianProcessSurrogate.bayesian_optimization_step`.
"""
bayesian_optimization_step(gp, b; kwargs...) =
    GaussianProcessSurrogate.bayesian_optimization_step(gp, b; kwargs...)

"""
    active_learning_suggest(gp, bounds; kwargs...)

Active learning suggestions. See `GaussianProcessSurrogate.active_learning_suggest`.
"""
active_learning_suggest(gp, b; kwargs...) =
    GaussianProcessSurrogate.active_learning_suggest(gp, b; kwargs...)

# ============================================================================
# Section 11: Full Pipeline
# ============================================================================

"""
    run_full_analysis(rs::BacktestResultSet{T};
                      n_clusters::Int=5,
                      n_regimes::Int=3,
                      rng::AbstractRNG=Random.default_rng()) where T -> Dict

Run the complete backtest farm analysis pipeline:
1. Multiple testing corrections on Sharpe ratios
2. Alpha landscape analysis (KDE, PCA, clustering, factors)
3. Parameter sensitivity (Sobol, local, interactions)
4. Regime-conditional analysis (HMM, per-regime Sharpe, specialist classification)
5. Optimal portfolio construction (MVO, risk parity, max diversification, Kelly)
6. Backtest validation (walk-forward, bootstrap, permutation) on top strategy
7. GP surrogate model fit
8. Report generation

Returns a comprehensive Dict of all results.
"""
function run_full_analysis(rs::BacktestResultSet{T};
                            n_clusters::Int=5,
                            n_regimes::Int=3,
                            rng::AbstractRNG=Random.default_rng()) where T<:Real
    n = length(rs)
    n < 3 && error("Need at least 3 results for analysis")

    results_dict = Dict{String, Any}()

    # Extract data
    sharpes = extract_sharpes(rs)
    return_mat = extract_return_matrix(rs)
    param_mat, param_names = extract_param_matrix(rs)

    # 1. Multiple Testing
    pvalues = Float64[MultipleTesting.sharpe_to_pvalue(Float64(r.metrics.sharpe),
                       length(r.metrics.returns)) for r in rs.results]
    mt = MultipleTesting.multiple_testing_analysis(rs)
    dsr = MultipleTesting.deflated_sharpe_ratio(rs)
    results_dict["multiple_testing"] = mt
    results_dict["deflated_sharpe"] = dsr

    # 2. Alpha Landscape
    kde = AlphaLandscapeAnalysis.kernel_density_estimate(sharpes)
    results_dict["sharpe_kde"] = kde

    if size(return_mat, 1) > 5 && size(return_mat, 2) > 2
        pca = AlphaLandscapeAnalysis.pca_returns(return_mat)
        results_dict["pca"] = pca

        nc = min(n_clusters, n)
        if nc >= 2
            clusters = AlphaLandscapeAnalysis.kmeans_strategies(return_mat, nc; rng=rng)
            results_dict["clusters"] = clusters
        end

        factors = AlphaLandscapeAnalysis.factor_decomposition(return_mat)
        results_dict["factors"] = factors
    end

    # 3. Parameter Sensitivity
    if size(param_mat, 2) > 0 && size(param_mat, 1) > size(param_mat, 2) + 2
        sobol = ParameterSensitivity.sobol_first_order(param_mat, sharpes; rng=rng)
        results_dict["sobol"] = sobol

        local_sens = ParameterSensitivity.local_sensitivity(param_mat, sharpes)
        results_dict["local_sensitivity"] = local_sens

        if size(param_mat, 1) > size(param_mat, 2) * (size(param_mat, 2) + 1) ÷ 2 + 5
            interactions = ParameterSensitivity.parameter_interactions(param_mat, sharpes;
                                                                       param_names=param_names)
            results_dict["interactions"] = interactions
        end
    end

    # 4. Regime Analysis
    if size(return_mat, 1) > 50
        regime = RegimeConditionalAnalysis.full_regime_analysis(return_mat; k=n_regimes, rng=rng)
        results_dict["regime_analysis"] = regime
    end

    # 5. Portfolio Construction
    if size(return_mat, 1) > 10 && size(return_mat, 2) > 1
        portfolios = OptimalPortfolioConstruction.compare_portfolios(return_mat)
        results_dict["portfolios"] = portfolios
    end

    # 6. Validation of top strategy
    best_idx = argmax(sharpes)
    best_returns = rs.results[best_idx].metrics.returns
    if length(best_returns) > 50
        validation = BacktestValidation.comprehensive_validation(best_returns; rng=rng)
        results_dict["top_strategy_validation"] = validation
    end

    # 7. GP Surrogate
    if size(param_mat, 1) > 10 && size(param_mat, 2) > 0
        gp = GaussianProcessSurrogate.GaussianProcessModel(size(param_mat, 2);
              signal_variance=max(var(sharpes), 0.01), noise_variance=0.01)
        GaussianProcessSurrogate.fit_gp!(gp, param_mat, sharpes)

        bounds = hcat(vec(minimum(param_mat, dims=1)), vec(maximum(param_mat, dims=1)))
        # Expand bounds slightly
        for j in 1:size(bounds, 1)
            range = bounds[j, 2] - bounds[j, 1]
            if range < eps(Float64)
                bounds[j, 1] -= 1.0
                bounds[j, 2] += 1.0
            else
                bounds[j, 1] -= 0.1 * range
                bounds[j, 2] += 0.1 * range
            end
        end

        suggestions = GaussianProcessSurrogate.active_learning_suggest(gp, bounds; rng=rng)
        results_dict["gp_model"] = gp
        results_dict["gp_suggestions"] = suggestions
    end

    # 8. Reports
    report = ReportGenerator.full_report(rs; pvalues=pvalues, dsr_pvalue=Float64(dsr.pvalue))
    results_dict["report"] = report

    results_dict["summary_stats"] = summary_stats(rs)

    results_dict
end

"""
    generate_synthetic_results(n_strategies::Int=100, n_periods::Int=500;
                                n_params::Int=5,
                                rng::AbstractRNG=Random.default_rng()) -> BacktestResultSet

Generate synthetic backtest results for testing the analysis pipeline.
"""
function generate_synthetic_results(n_strategies::Int=100, n_periods::Int=500;
                                     n_params::Int=5,
                                     rng::AbstractRNG=Random.default_rng())
    param_names = ["param_$j" for j in 1:n_params]
    results = BacktestResult{Float64}[]

    for i in 1:n_strategies
        # Random parameters
        params = Dict{String, Float64}()
        param_vec = zeros(n_params)
        for (j, pn) in enumerate(param_names)
            val = randn(rng) * 2.0
            params[pn] = val
            param_vec[j] = val
        end

        config = BacktestConfig("strategy_$i", params)

        # Generate returns that depend on parameters
        true_sharpe = 0.3 * param_vec[1] - 0.1 * param_vec[1]^2 +
                      0.2 * param_vec[min(2, n_params)] + 0.05 * randn(rng)
        daily_mu = true_sharpe / sqrt(252.0) * 0.01
        daily_vol = 0.01

        rets = daily_mu .+ daily_vol .* randn(rng, n_periods)

        sr = compute_sharpe(rets)
        so = compute_sortino(rets)
        md = compute_max_drawdown(rets)
        tr = prod(1.0 .+ rets) - 1.0
        sk = compute_skewness(rets)
        ku = compute_kurtosis(rets)

        metrics = BacktestMetrics{Float64}(
            sharpe=sr, sortino=so, calmar=abs(md) > 0 ? (mean(rets)*252) / md : 0.0,
            max_drawdown=md, total_return=tr, annual_return=mean(rets)*252,
            annual_volatility=std(rets)*sqrt(252.0), skewness=sk, kurtosis=ku,
            var_95=sort(rets)[max(1, round(Int, 0.05*n_periods))],
            cvar_95=mean(sort(rets)[1:max(1, round(Int, 0.05*n_periods))]),
            win_rate=mean(rets .> 0), profit_factor=abs(sum(rets[rets .< 0])) > 0 ?
                sum(rets[rets .> 0]) / abs(sum(rets[rets .< 0])) : 0.0,
            num_trades=n_periods, returns=rets
        )

        push!(results, BacktestResult(config, metrics; id=i))
    end

    BacktestResultSet(results)
end

end # module BacktestFarmAnalytics
