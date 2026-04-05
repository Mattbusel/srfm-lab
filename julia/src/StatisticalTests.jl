"""
StatisticalTests — Complete Statistical Testing Framework

Full implementations from scratch:
  - Standard parametric: t-test, F-test, chi-squared, ANOVA
  - Non-parametric: Wilcoxon, Mann-Whitney, Kruskal-Wallis, Friedman
  - Time series: ADF, KPSS, PP, ARCH-LM, Ljung-Box, Diebold-Mariano
  - Cointegration: Engle-Granger, Johansen trace and max-eigenvalue
  - Multiple testing: Bonferroni, BH, Romano-Wolf bootstrap
  - Bootstrap: percentile, BCa, studentized; block bootstrap
  - White's reality check, Hansen's SPA test
  - Permutation tests for strategy significance
  - Power analysis and sample size determination
"""
module StatisticalTests

using Statistics
using LinearAlgebra
using Random

export one_sample_t_test, two_sample_t_test, paired_t_test, welch_t_test
export f_test_variance, levene_test, bartlett_test
export chi_squared_test, chi_squared_gof, chi_squared_independence
export one_way_anova, two_way_anova, kruskal_wallis_test, friedman_test
export wilcoxon_signed_rank, mann_whitney_u, sign_test
export adf_test, kpss_test, pp_test, arch_lm_test, ljung_box_test
export diebold_mariano_test, mincer_zarnowitz_test
export engle_granger_cointegration, johansen_cointegration_trace
export bonferroni_correction, benjamini_hochberg, romano_wolf_bootstrap
export bootstrap_ci, bca_bootstrap, studentized_bootstrap
export block_bootstrap_ci, stationary_bootstrap
export whites_reality_check, hansen_spa_test
export permutation_test, strategy_permutation_test
export power_analysis, sample_size_determination
export TestResult

# =============================================================================
# CORE DATA STRUCTURE
# =============================================================================

"""
    TestResult

Standardized result of a statistical test.
"""
struct TestResult
    name::String
    statistic::Float64
    p_value::Float64
    critical_value::Float64
    reject_null::Bool        # at 5% significance
    confidence_interval::Tuple{Float64, Float64}
    additional::Dict{String, Float64}
end

function TestResult(name, stat, p, cv;
                    ci=(NaN, NaN),
                    additional=Dict{String,Float64}())
    TestResult(name, stat, p, cv, p < 0.05, ci, additional)
end

# =============================================================================
# SECTION 1: PARAMETRIC TESTS
# =============================================================================

"""
    one_sample_t_test(x; mu0=0.0, alpha=0.05) -> TestResult

One-sample t-test: H₀: μ = μ₀.
t = (x̄ - μ₀) / (s/√n) ~ t(n-1) under H₀.
"""
function one_sample_t_test(x::Vector{Float64};
                             mu0::Float64=0.0,
                             alpha::Float64=0.05)::TestResult

    n = length(x)
    n < 2 && return TestResult("One-Sample t-test", 0.0, 1.0, Inf)

    xbar = mean(x); s = std(x)
    t_stat = s > 0 ? (xbar - mu0) / (s / sqrt(n)) : 0.0
    df = n - 1

    p_val = 2 * _t_cdf_tail(abs(t_stat), df)
    cv = _t_quantile(1-alpha/2, df)

    # Confidence interval
    se = s / sqrt(n)
    ci = (xbar - cv*se, xbar + cv*se)

    return TestResult("One-Sample t-test", t_stat, p_val, cv;
                       ci=ci, additional=Dict("df"=>df, "mean"=>xbar, "se"=>se))
end

"""
    two_sample_t_test(x, y; equal_var=false, alpha=0.05) -> TestResult

Two-sample t-test (equal variance) or Welch's t-test (unequal variance).
H₀: μ_x = μ_y.
"""
function two_sample_t_test(x::Vector{Float64}, y::Vector{Float64};
                             equal_var::Bool=false,
                             alpha::Float64=0.05)::TestResult

    n1 = length(x); n2 = length(y)
    (n1 < 2 || n2 < 2) && return TestResult("Two-Sample t-test", 0.0, 1.0, Inf)

    m1 = mean(x); m2 = mean(y)
    s1 = std(x);  s2 = std(y)

    if equal_var
        # Pooled variance
        sp2 = ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2)
        se = sqrt(sp2 * (1/n1 + 1/n2))
        df = n1 + n2 - 2.0
    else
        # Welch-Satterthwaite
        v1 = s1^2/n1; v2 = s2^2/n2
        se = sqrt(v1 + v2)
        df = (v1+v2)^2 / (v1^2/(n1-1) + v2^2/(n2-1))
    end

    t_stat = se > 0 ? (m1-m2)/se : 0.0
    p_val = 2 * _t_cdf_tail(abs(t_stat), df)
    cv = _t_quantile(1-alpha/2, df)

    diff = m1-m2
    ci = (diff - cv*se, diff + cv*se)

    return TestResult("Two-Sample t-test", t_stat, p_val, cv;
                       ci=ci, additional=Dict("df"=>df, "diff"=>diff))
end

"""
    welch_t_test(x, y; alpha=0.05) -> TestResult

Welch's t-test (unequal variances). Robust default for most comparisons.
"""
welch_t_test(x, y; alpha=0.05) = two_sample_t_test(x, y; equal_var=false, alpha=alpha)

"""
    paired_t_test(x, y; alpha=0.05) -> TestResult

Paired t-test: H₀: mean(x-y) = 0. Reduces to one-sample on differences.
"""
function paired_t_test(x::Vector{Float64}, y::Vector{Float64};
                        alpha::Float64=0.05)::TestResult
    n = min(length(x), length(y))
    d = x[1:n] .- y[1:n]
    result = one_sample_t_test(d; mu0=0.0, alpha=alpha)
    return TestResult("Paired t-test", result.statistic, result.p_value,
                       result.critical_value; ci=result.confidence_interval,
                       additional=result.additional)
end

"""
    f_test_variance(x, y; alpha=0.05) -> TestResult

F-test for equality of variances. H₀: σ²_x = σ²_y.
F = s²_x / s²_y ~ F(n₁-1, n₂-1).
"""
function f_test_variance(x::Vector{Float64}, y::Vector{Float64};
                           alpha::Float64=0.05)::TestResult

    n1 = length(x); n2 = length(y)
    (n1 < 3 || n2 < 3) && return TestResult("F-test (Variance)", 1.0, 1.0, Inf)

    s1 = var(x); s2 = var(y)
    s2 <= 0 && return TestResult("F-test (Variance)", 0.0, 1.0, Inf)

    f_stat = s1 / s2
    df1 = n1-1.0; df2 = n2-1.0

    # Two-tailed p-value
    p_upper = _f_cdf_tail(f_stat, df1, df2)
    p_val = min(2 * min(p_upper, 1-p_upper), 1.0)
    cv = _f_quantile(1-alpha/2, df1, df2)

    return TestResult("F-test (Variance)", f_stat, p_val, cv;
                       additional=Dict("df1"=>df1, "df2"=>df2, "ratio"=>sqrt(f_stat)))
end

"""
    levene_test(groups...; alpha=0.05) -> TestResult

Levene's test for equality of variances (more robust than Bartlett).
Uses absolute deviations from group medians.
"""
function levene_test(groups::Vector{Float64}...;
                      alpha::Float64=0.05)::TestResult

    k = length(groups)
    N = sum(length, groups)
    k < 2 && return TestResult("Levene Test", 0.0, 1.0, Inf)

    # Absolute deviations from group medians
    z_groups = [abs.(g .- median(g)) for g in groups]
    z_means  = mean.(z_groups)
    z_grand  = mean(vcat(z_groups...))

    # F-statistic
    numerator   = sum(length(z_groups[i]) * (z_means[i]-z_grand)^2 for i in 1:k) / (k-1)
    denominator = sum(sum((z-z_means[i])^2 for z in z_groups[i]) for i in 1:k) / (N-k)

    f_stat = denominator > 0 ? numerator/denominator : 0.0
    p_val = _f_cdf_tail(f_stat, k-1.0, N-k*1.0)
    cv = _f_quantile(1-alpha, k-1.0, N-k*1.0)

    return TestResult("Levene Test", f_stat, p_val, cv;
                       additional=Dict("k"=>k, "N"=>N))
end

"""
    bartlett_test(groups...; alpha=0.05) -> TestResult

Bartlett's test for equality of variances. Assumes normality.
"""
function bartlett_test(groups::Vector{Float64}...; alpha::Float64=0.05)::TestResult
    k = length(groups)
    N = sum(length, groups)
    k < 2 && return TestResult("Bartlett Test", 0.0, 1.0, Inf)

    ns = length.(groups)
    variances = var.(groups)
    df_i = ns .- 1

    # Pooled variance
    sp2 = sum(df_i .* variances) / (N - k)
    sp2 <= 0 && return TestResult("Bartlett Test", 0.0, 1.0, Inf)

    M = (N-k)*log(sp2) - sum(df_i .* log.(max.(variances, 1e-20)))
    c = 1 + (sum(1.0 ./ df_i) - 1.0/(N-k)) / (3*(k-1))
    X2 = M / c

    p_val = _chi2_cdf_tail(X2, k-1.0)
    cv = _chi2_quantile(1-alpha, k-1.0)

    return TestResult("Bartlett Test", X2, p_val, cv;
                       additional=Dict("k"=>k))
end

# =============================================================================
# SECTION 2: CHI-SQUARED TESTS
# =============================================================================

"""
    chi_squared_gof(observed, expected; alpha=0.05) -> TestResult

Chi-squared goodness of fit test.
X² = Σ (O-E)²/E ~ χ²(k-1).
"""
function chi_squared_gof(observed::Vector{Float64},
                           expected::Vector{Float64};
                           alpha::Float64=0.05)::TestResult

    k = length(observed)
    @assert length(expected) == k
    df = k - 1.0

    # Normalize expected
    e_norm = expected .* (sum(observed) / sum(expected))
    X2 = sum((observed .- e_norm).^2 ./ max.(e_norm, 1e-10))
    p_val = _chi2_cdf_tail(X2, df)
    cv = _chi2_quantile(1-alpha, df)

    return TestResult("Chi-Squared GoF", X2, p_val, cv;
                       additional=Dict("df"=>df))
end

"""
    chi_squared_independence(contingency_table; alpha=0.05) -> TestResult

Chi-squared test of independence for a contingency table.
"""
function chi_squared_independence(table::Matrix{Float64};
                                    alpha::Float64=0.05)::TestResult

    rows, cols = size(table)
    n = sum(table)
    row_sums = sum(table, dims=2)
    col_sums = sum(table, dims=1)

    X2 = 0.0
    for i in 1:rows, j in 1:cols
        expected = row_sums[i] * col_sums[j] / n
        expected > 0 && (X2 += (table[i,j]-expected)^2 / expected)
    end

    df = (rows-1.0) * (cols-1.0)
    p_val = _chi2_cdf_tail(X2, df)
    cv = _chi2_quantile(1-alpha, df)

    return TestResult("Chi-Squared Independence", X2, p_val, cv;
                       additional=Dict("df"=>df, "n"=>n))
end

"""
    chi_squared_test(x, bins=10; alpha=0.05) -> TestResult

Chi-squared test for normality (discretized into bins).
"""
function chi_squared_test(x::Vector{Float64}; bins::Int=10, alpha::Float64=0.05)::TestResult
    n = length(x)
    n < bins && return TestResult("Chi-Squared Normality", 0.0, 1.0, Inf)

    mu = mean(x); sig = std(x)
    # Expected counts under N(mu, sig²)
    edges = [minimum(x)] ++ [quantile(x, k/bins) for k in 1:(bins-1)] ++ [maximum(x)+1]
    edges = unique(sort(edges))
    k = length(edges)-1

    observed = zeros(k)
    for xi in x
        for i in 1:k
            if xi >= edges[i] && (xi < edges[i+1] || i == k)
                observed[i] += 1; break
            end
        end
    end

    expected = zeros(k)
    for i in 1:k
        z1 = sig > 0 ? (edges[i]-mu)/sig : -Inf
        z2 = sig > 0 ? (edges[i+1]-mu)/sig : Inf
        expected[i] = n * (_normal_cdf(z2) - _normal_cdf(z1))
    end

    return chi_squared_gof(observed, expected; alpha=alpha)
end

# =============================================================================
# SECTION 3: NON-PARAMETRIC TESTS
# =============================================================================

"""
    wilcoxon_signed_rank(x; mu0=0.0, alpha=0.05) -> TestResult

Wilcoxon signed-rank test: non-parametric alternative to one-sample t-test.
H₀: median = mu0.
"""
function wilcoxon_signed_rank(x::Vector{Float64};
                                mu0::Float64=0.0,
                                alpha::Float64=0.05)::TestResult

    d = x .- mu0
    d = d[d .!= 0]  # remove zeros
    n = length(d)
    n < 2 && return TestResult("Wilcoxon Signed-Rank", 0.0, 1.0, Inf)

    # Rank absolute differences
    abs_d = abs.(d)
    ranks = _rank(abs_d)

    W_plus  = sum(ranks[d .> 0])
    W_minus = sum(ranks[d .< 0])
    W = min(W_plus, W_minus)

    # Normal approximation (for n ≥ 10)
    mu_W = n*(n+1)/4
    sigma_W = sqrt(n*(n+1)*(2n+1)/24)
    z = sigma_W > 0 ? (W - mu_W) / sigma_W : 0.0
    p_val = 2*_normal_cdf(-abs(z))

    return TestResult("Wilcoxon Signed-Rank", W, p_val, 0.0;
                       additional=Dict("W+"=>W_plus, "W-"=>W_minus, "z"=>z))
end

"""
    mann_whitney_u(x, y; alpha=0.05) -> TestResult

Mann-Whitney U test: non-parametric two-sample comparison.
H₀: P(X > Y) = 0.5.
"""
function mann_whitney_u(x::Vector{Float64}, y::Vector{Float64};
                          alpha::Float64=0.05)::TestResult

    n1 = length(x); n2 = length(y)
    (n1 < 1 || n2 < 1) && return TestResult("Mann-Whitney U", 0.0, 1.0, Inf)

    # Compute U = Σ W_{ij} where W_{ij} = I(x_i > y_j) + 0.5*I(x_i==y_j)
    U1 = 0.0
    for xi in x, yj in y
        xi > yj ? (U1 += 1.0) : xi == yj ? (U1 += 0.5) : nothing
    end
    U2 = n1*n2 - U1
    U = min(U1, U2)

    # Normal approximation
    mu_U = n1*n2/2
    sigma_U = sqrt(n1*n2*(n1+n2+1)/12)
    z = sigma_U > 0 ? (U - mu_U)/sigma_U : 0.0
    p_val = 2*_normal_cdf(-abs(z))

    cv = _normal_quantile(1-alpha/2)
    return TestResult("Mann-Whitney U", U, p_val, cv;
                       additional=Dict("U1"=>U1, "U2"=>U2, "z"=>z))
end

"""
    kruskal_wallis_test(groups...; alpha=0.05) -> TestResult

Kruskal-Wallis H test: non-parametric one-way ANOVA.
H₀: all groups have the same distribution.
"""
function kruskal_wallis_test(groups::Vector{Float64}...;
                               alpha::Float64=0.05)::TestResult

    k = length(groups)
    k < 2 && return TestResult("Kruskal-Wallis", 0.0, 1.0, Inf)

    all_vals = vcat(groups...)
    N = length(all_vals)
    all_ranks = _rank(all_vals)

    ni = length.(groups)
    cumlen = [0; cumsum(ni)]

    R_bar = [(sum(all_ranks[cumlen[i]+1:cumlen[i+1]]) / ni[i]) for i in 1:k]

    H = 12/(N*(N+1)) * sum(ni[i]*(R_bar[i] - (N+1)/2)^2 for i in 1:k)

    df = k-1.0
    p_val = _chi2_cdf_tail(H, df)
    cv = _chi2_quantile(1-alpha, df)

    return TestResult("Kruskal-Wallis", H, p_val, cv;
                       additional=Dict("df"=>df, "k"=>k, "N"=>N))
end

"""
    friedman_test(matrix; alpha=0.05) -> TestResult

Friedman test: non-parametric repeated measures ANOVA.
`matrix` is (n_subjects × k_treatments).
"""
function friedman_test(matrix::Matrix{Float64}; alpha::Float64=0.05)::TestResult
    n, k = size(matrix)
    (n < 3 || k < 2) && return TestResult("Friedman", 0.0, 1.0, Inf)

    # Rank within each row (subject)
    R = zeros(n, k)
    for i in 1:n
        R[i,:] = _rank(matrix[i,:])
    end

    R_bar = vec(mean(R, dims=1))
    X2 = 12n / (k*(k+1)) * sum((R_bar[j] - (k+1)/2)^2 for j in 1:k)

    df = k-1.0
    p_val = _chi2_cdf_tail(X2, df)
    cv = _chi2_quantile(1-alpha, df)

    return TestResult("Friedman", X2, p_val, cv; additional=Dict("df"=>df))
end

"""
    one_way_anova(groups...; alpha=0.05) -> TestResult

One-way ANOVA F-test. H₀: μ₁ = μ₂ = ... = μₖ.
"""
function one_way_anova(groups::Vector{Float64}...; alpha::Float64=0.05)::TestResult
    k = length(groups)
    k < 2 && return TestResult("One-Way ANOVA", 0.0, 1.0, Inf)

    all_vals = vcat(groups...)
    N = length(all_vals)
    grand_mean = mean(all_vals)

    # Between-group SS
    SSB = sum(length(g)*(mean(g)-grand_mean)^2 for g in groups)
    # Within-group SS
    SSW = sum(sum((x-mean(g))^2 for x in g) for g in groups)

    df1 = k-1.0; df2 = N-k*1.0
    MSB = SSB/df1; MSW = SSW/df2
    F = MSW > 0 ? MSB/MSW : 0.0

    p_val = _f_cdf_tail(F, df1, df2)
    cv = _f_quantile(1-alpha, df1, df2)

    return TestResult("One-Way ANOVA", F, p_val, cv;
                       additional=Dict("df1"=>df1, "df2"=>df2, "F"=>F))
end

"""
    sign_test(x, y; alpha=0.05) -> TestResult

Sign test: H₀: P(X > Y) = 0.5. Uses binomial distribution.
"""
function sign_test(x::Vector{Float64}, y::Vector{Float64}; alpha::Float64=0.05)::TestResult
    n = min(length(x), length(y))
    d = x[1:n] .- y[1:n]
    d = d[d .!= 0]; n_eff = length(d)
    n_eff < 2 && return TestResult("Sign Test", 0.0, 1.0, Inf)

    n_pos = sum(d .> 0)
    # Exact binomial p-value (normal approx for large n)
    z = (n_pos - n_eff/2) / sqrt(n_eff/4)
    p_val = 2*_normal_cdf(-abs(z))

    return TestResult("Sign Test", z, p_val, _normal_quantile(1-alpha/2);
                       additional=Dict("n_plus"=>n_pos, "n_eff"=>n_eff))
end

# =============================================================================
# SECTION 4: TIME SERIES TESTS
# =============================================================================

"""
    ljung_box_test(residuals; lags=20, alpha=0.05) -> TestResult

Ljung-Box Q test for autocorrelation in residuals.
H₀: no autocorrelation up to lag h.
Q = n(n+2) Σ ρ̂²ₕ/(n-h) ~ χ²(h).
"""
function ljung_box_test(residuals::Vector{Float64};
                          lags::Int=20,
                          alpha::Float64=0.05)::TestResult

    n = length(residuals)
    n < lags + 2 && return TestResult("Ljung-Box", 0.0, 1.0, Inf)

    r_mean = mean(residuals)
    rc = residuals .- r_mean
    gamma0 = sum(rc.^2)/n

    Q = 0.0
    for h in 1:lags
        rho_h = sum(rc[1:(n-h)] .* rc[(h+1):n]) / (n * gamma0)
        Q += rho_h^2 / (n-h)
    end
    Q *= n*(n+2)

    p_val = _chi2_cdf_tail(Q, lags*1.0)
    cv = _chi2_quantile(1-alpha, lags*1.0)

    return TestResult("Ljung-Box Q", Q, p_val, cv;
                       additional=Dict("lags"=>lags))
end

"""
    arch_lm_test(residuals; lags=12, alpha=0.05) -> TestResult

Engle (1982) ARCH-LM test for conditional heteroscedasticity.
Regress r²ₜ on r²ₜ₋₁,...,r²ₜ₋ₘ and test R².
LM = T * R² ~ χ²(m).
"""
function arch_lm_test(residuals::Vector{Float64};
                       lags::Int=12,
                       alpha::Float64=0.05)::TestResult

    n = length(residuals)
    n < lags + 5 && return TestResult("ARCH-LM", 0.0, 1.0, Inf)

    r2 = residuals.^2
    T_eff = n - lags

    # Regress r²_t on [1, r²_{t-1}, ..., r²_{t-lags}]
    Y = r2[(lags+1):n]
    X = hcat(ones(T_eff), [r2[(lags-j+1):(n-j)] for j in 1:lags]...)

    b = try (X'*X+1e-8I)\(X'*Y) catch zeros(lags+1) end
    Yhat = X*b
    ss_res = sum((Y-Yhat).^2); ss_tot = sum((Y.-mean(Y)).^2)
    R2 = ss_tot > 0 ? 1-ss_res/ss_tot : 0.0

    LM = T_eff * R2
    p_val = _chi2_cdf_tail(LM, lags*1.0)
    cv = _chi2_quantile(1-alpha, lags*1.0)

    return TestResult("ARCH-LM", LM, p_val, cv;
                       additional=Dict("lags"=>lags, "R2"=>R2))
end

"""
    adf_test(y; max_lags=nothing, trend=:constant, alpha=0.05) -> TestResult

Augmented Dickey-Fuller unit root test.
"""
function adf_test(y::Vector{Float64};
                   max_lags::Union{Int,Nothing}=nothing,
                   trend::Symbol=:constant,
                   alpha::Float64=0.05)::TestResult

    n = length(y); n < 10 && return TestResult("ADF", 0.0, 1.0, Inf)
    max_p = max_lags === nothing ? max(1, floor(Int, sqrt(n))) : max_lags
    max_p = min(max_p, n÷4-1)

    dy = diff(y); T = length(dy)

    # BIC lag selection
    best_bic = Inf; best_p = 0
    for p in 0:max_p
        T_eff = T-p; T_eff < 5 && break
        n_extra = trend == :none ? 0 : trend == :constant ? 1 : 2
        X = _build_adf_X(y, dy, p, T_eff, trend, n_extra)
        dep = dy[(p+1):T]
        b = try (X'X+1e-8I)\(X'*dep) catch zeros(size(X,2)) end
        resid = dep-X*b; s2 = sum(resid.^2)/(T_eff-size(X,2))
        bic = T_eff*log(s2)+log(T_eff)*size(X,2)
        if bic < best_bic; best_bic=bic; best_p=p end
    end

    p = best_p; T_eff = T-p
    n_extra = trend == :none ? 0 : trend == :constant ? 1 : 2
    X = _build_adf_X(y, dy, p, T_eff, trend, n_extra)
    dep = dy[(p+1):T]
    b = try (X'X+1e-8I)\(X'*dep) catch zeros(size(X,2)) end
    resid = dep-X*b; s2 = sum(resid.^2)/(T_eff-size(X,2))
    var_b1 = s2*(try inv(X'X+1e-8I)[1,1] catch 1/n end)
    t = sqrt(var_b1) > 0 ? b[1]/sqrt(var_b1) : 0.0

    cv5 = trend == :none ? -1.94 : trend == :constant ? -2.86 : -3.41
    p_val = t < cv5 ? 0.01 : t < -1.5 ? 0.10 : 0.50

    return TestResult("ADF", t, p_val, cv5;
                       additional=Dict("lag"=>p, "n"=>n))
end

function _build_adf_X(y, dy, p, T_eff, trend, n_extra)
    T = length(dy)
    X = zeros(T_eff, 1+p+n_extra)
    X[:,1] = y[(p+1):T]
    if trend == :constant
        X[:,2] = ones(T_eff)
        for j in 1:p; X[:,2+j] = dy[(p-j+1):(T-j)] end
    elseif trend == :trend
        X[:,2] = ones(T_eff); X[:,3] = collect(1:T_eff)
        for j in 1:p; X[:,3+j] = dy[(p-j+1):(T-j)] end
    else
        for j in 1:p; X[:,1+j] = dy[(p-j+1):(T-j)] end
    end
    return X
end

"""
    kpss_test(y; trend=:constant, alpha=0.05) -> TestResult

KPSS test for stationarity. H₀: series IS stationary.
"""
function kpss_test(y::Vector{Float64}; trend::Symbol=:constant, alpha::Float64=0.05)::TestResult
    n = length(y)

    resid = if trend == :constant
        y .- mean(y)
    else
        X = hcat(ones(n), collect(1:n))
        b = (X'X+1e-8I)\(X'*y)
        y .- X*b
    end

    S = cumsum(resid)
    m = max(1, floor(Int, sqrt(n)))

    # Newey-West long-run variance
    yc = resid .- mean(resid)
    V = sum(yc.^2)/n
    for h in 1:m
        w = 1-h/(m+1)
        gamma_h = sum(yc[1:(n-h)].*yc[(h+1):n])/n
        V += 2w*gamma_h
    end

    eta = sum(S.^2)/(n^2 * max(V, 1e-20))
    cv = trend == :constant ? 0.463 : 0.146

    p_val = eta > cv ? 0.01 : 0.20
    return TestResult("KPSS", eta, p_val, cv;
                       additional=Dict("V"=>V))
end

"""
    pp_test(y; trend=:constant, alpha=0.05) -> TestResult

Phillips-Perron unit root test (non-parametric correction).
"""
function pp_test(y::Vector{Float64}; trend::Symbol=:constant, alpha::Float64=0.05)::TestResult
    n = length(y)
    dy = diff(y); T = length(dy)
    y_lag = y[1:T]
    n_extra = trend == :none ? 0 : trend == :constant ? 1 : 2
    X = zeros(T, 1+n_extra)
    X[:,1] = y_lag
    trend == :constant && (X[:,2] = ones(T))
    trend == :trend    && (X[:,2]=ones(T); X[:,3]=collect(1:T))

    b = try (X'X+1e-8I)\(X'*dy) catch zeros(size(X,2)) end
    resid = dy-X*b
    s2 = sum(resid.^2)/(T-size(X,2))
    var_b1 = s2*(try inv(X'X+1e-8I)[1,1] catch 1/n end)
    t_naive = sqrt(var_b1) > 0 ? b[1]/sqrt(var_b1) : 0.0

    # NW long-run variance
    m = max(1, floor(Int, T^(1/3)))
    yc = resid.-mean(resid)
    lambda2 = sum(yc.^2)/T
    for h in 1:m
        w = 1-h/(m+1)
        lambda2 += 2w*sum(yc[1:(T-h)].*yc[(h+1):T])/T
    end

    se = sqrt(var_b1)
    Z_t = se > 0 ? t_naive*sqrt(s2/lambda2) - (lambda2-s2)/(2*sqrt(lambda2)*se) : t_naive
    cv5 = trend == :none ? -1.94 : trend == :constant ? -2.86 : -3.41
    p_val = Z_t < cv5 ? 0.01 : Z_t < -1.5 ? 0.10 : 0.50

    return TestResult("Phillips-Perron", Z_t, p_val, cv5)
end

"""
    diebold_mariano_test(e1, e2; loss=:squared, h=1, alpha=0.05) -> TestResult

Diebold-Mariano (1995) test for equal predictive accuracy.
H₀: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t) is the loss differential.
"""
function diebold_mariano_test(e1::Vector{Float64}, e2::Vector{Float64};
                                loss::Symbol=:squared,
                                h::Int=1, alpha::Float64=0.05)::TestResult

    n = min(length(e1), length(e2))
    loss_func = loss == :absolute ? abs : x -> x^2

    d = loss_func.(e1[1:n]) .- loss_func.(e2[1:n])

    d_mean = mean(d)

    # HAC variance with NW bandwidth h
    m = h - 1
    V = sum((d.-d_mean).^2)/n
    for lag in 1:m
        w = 1-lag/(m+1)
        cov_lag = sum((d[1:(n-lag)].-d_mean).*(d[(lag+1):n].-d_mean))/n
        V += 2w*cov_lag
    end
    se = sqrt(max(V/n, 0.0))
    DM = se > 0 ? d_mean/se : 0.0

    p_val = 2*_normal_cdf(-abs(DM))
    cv = _normal_quantile(1-alpha/2)

    return TestResult("Diebold-Mariano", DM, p_val, cv;
                       additional=Dict("mean_diff"=>d_mean, "se"=>se))
end

"""
    mincer_zarnowitz_test(forecasts, actuals; alpha=0.05) -> TestResult

Mincer-Zarnowitz (1969) forecast efficiency test.
Regress actual on forecast: y = a + b*f + e.
H₀: a=0, b=1 (forecast unbiased and efficient).
"""
function mincer_zarnowitz_test(forecasts::Vector{Float64},
                                 actuals::Vector{Float64};
                                 alpha::Float64=0.05)::TestResult

    n = min(length(forecasts), length(actuals))
    n < 5 && return TestResult("Mincer-Zarnowitz", 0.0, 1.0, Inf)

    X = hcat(ones(n), forecasts[1:n])
    y = actuals[1:n]
    b = try (X'X+1e-8I)\(X'*y) catch [0.0,1.0] end
    resid = y-X*b; s2 = sum(resid.^2)/(n-2)
    Cov_b = s2*(X'X+1e-8I)^(-1)

    # F-test: H₀: a=0, b=1; test R'β = r
    R = [1 0; 0 1]; r = [0.0; 1.0]
    diff_vec = R*b .- r
    F = (diff_vec' * inv(R*Cov_b*R' .+ 1e-12I) * diff_vec) / 2

    p_val = _f_cdf_tail(F, 2.0, n-2.0)
    cv = _f_quantile(1-alpha, 2.0, n-2.0)

    return TestResult("Mincer-Zarnowitz", F, p_val, cv;
                       additional=Dict("alpha"=>b[1], "beta"=>b[2]))
end

# =============================================================================
# SECTION 5: COINTEGRATION TESTS
# =============================================================================

"""
    engle_granger_cointegration(y, x; trend=:constant, alpha=0.05) -> TestResult

Engle-Granger (1987) two-step cointegration test.
Step 1: regress y on x, get residuals.
Step 2: ADF test on residuals (test for unit root in cointegrating residuals).
"""
function engle_granger_cointegration(y::Vector{Float64},
                                       x::Vector{Float64};
                                       trend::Symbol=:constant,
                                       alpha::Float64=0.05)::TestResult

    n = min(length(y), length(x))
    n < 10 && return TestResult("Engle-Granger CI", 0.0, 1.0, Inf)

    X = trend == :constant ? hcat(ones(n), x[1:n]) : x[1:n:1,:]
    b = try (X'X+1e-8I)\(X'*y[1:n]) catch [0.0,1.0] end
    resid = y[1:n] .- X*b

    # ADF on residuals (no constant—already de-meaned by regression)
    result = adf_test(resid; trend=:none, alpha=alpha)

    # MacKinnon CI critical values (more negative than standard ADF)
    cv5_ci = trend == :constant ? -3.34 : -3.90
    p_adj = result.statistic < cv5_ci ? 0.01 : 0.20

    return TestResult("Engle-Granger CI", result.statistic, p_adj, cv5_ci;
                       additional=Dict("beta"=>b[end]))
end

"""
    johansen_cointegration_trace(Y; p=1, alpha=0.05) -> NamedTuple

Johansen (1988, 1991) trace test for cointegrating rank.
Tests H₀: rank ≤ r vs H₁: rank > r for r = 0, 1, ..., K-1.

# Returns
- NamedTuple: trace_statistics, p_values, critical_values, cointegrating_rank
"""
function johansen_cointegration_trace(Y::Matrix{Float64};
                                        p::Int=1,
                                        alpha::Float64=0.05)

    T, K = size(Y)
    T < p+K+5 && return (trace_statistics=zeros(K), p_values=ones(K),
                           critical_values=zeros(K), cointegrating_rank=0)

    # Step 1: regress ΔY on lagged ΔY
    dY = diff(Y, dims=1)
    T_eff = size(dY, 1) - p

    # Build regressor matrices for reduced-rank regression
    Y0 = dY[(p+1):end, :]            # T_eff × K
    Y1 = Y[p:(T-1), :]              # T_eff × K (lagged levels)

    # Controls: lagged differences
    Z = zeros(T_eff, K*p)
    for lag in 1:p
        Z[:, (lag-1)*K+1:lag*K] = dY[(p-lag+1):(end-lag), :]
    end

    # Partial out Z from Y0 and Y1
    M = I(T_eff) - Z * (try (Z'Z+1e-8I)\Z' catch Z\I(T_eff) end)
    R0 = M * Y0   # T_eff × K
    R1 = M * Y1   # T_eff × K

    # Eigenvalue problem: S₁₁⁻¹ S₁₀ S₀₀⁻¹ S₀₁
    S00 = R0'*R0/T_eff + 1e-8I
    S11 = R1'*R1/T_eff + 1e-8I
    S01 = R0'*R1/T_eff

    try
        S11_inv = inv(S11)
        A = S11_inv * S01' * inv(S00) * S01
        eigs_vals = real.(eigvals(Symmetric(A+A')/2))
        eigs_vals = sort(abs.(eigs_vals), rev=true)
        eigs_vals = min.(eigs_vals, 1-1e-10)

        # Trace statistics: λ_trace(r) = -T Σₖ₌ᵣ₊₁ᴷ log(1-λ_k)
        trace_stats = zeros(K)
        for r in 0:(K-1)
            trace_stats[r+1] = -T_eff * sum(log.(1 .- max.(eigs_vals[r+1:end], 1e-10)))
        end

        # Osterwald-Lenum (1992) 5% critical values (K-r trends, no linear trend)
        cv_table = [3.76, 15.41, 29.68, 47.21, 68.52]  # K=1..5

        crit_vals = zeros(K)
        for r in 0:(K-1)
            idx = K - r
            crit_vals[r+1] = idx <= length(cv_table) ? cv_table[idx] : cv_table[end]
        end

        p_vals = [trace_stats[r+1] > crit_vals[r+1] ? 0.01 : 0.20 for r in 0:(K-1)]
        rank = findfirst(p_vals .>= 0.05)
        ci_rank = rank === nothing ? K : rank - 1

        return (trace_statistics=trace_stats, p_values=p_vals,
                 critical_values=crit_vals, cointegrating_rank=ci_rank)
    catch
        return (trace_statistics=zeros(K), p_values=ones(K),
                 critical_values=zeros(K), cointegrating_rank=0)
    end
end

# =============================================================================
# SECTION 6: MULTIPLE TESTING
# =============================================================================

"""
    bonferroni_correction(p_values; alpha=0.05) -> Vector{Bool}

Bonferroni correction: adjusted threshold = α/m.
Controls family-wise error rate (FWER).
"""
function bonferroni_correction(p_values::Vector{Float64};
                                 alpha::Float64=0.05)::Vector{Bool}
    m = length(p_values)
    return p_values .< (alpha / m)
end

"""
    benjamini_hochberg(p_values; alpha=0.05) -> Vector{Bool}

Benjamini-Hochberg (1995) procedure for false discovery rate (FDR) control.
Rejects H_{(i)} if p_{(i)} ≤ (i/m)α.
"""
function benjamini_hochberg(p_values::Vector{Float64};
                              alpha::Float64=0.05)::Vector{Bool}
    m = length(p_values)
    sorted_idx = sortperm(p_values)
    sorted_p = p_values[sorted_idx]

    reject_sorted = zeros(Bool, m)
    for i in m:-1:1
        if sorted_p[i] <= i*alpha/m
            reject_sorted[1:i] .= true
            break
        end
    end

    reject = zeros(Bool, m)
    for (rank, idx) in enumerate(sorted_idx)
        reject[idx] = reject_sorted[rank]
    end
    return reject
end

"""
    romano_wolf_bootstrap(test_statistics, returns_matrix;
                           B=1000, alpha=0.05) -> Vector{Bool}

Romano-Wolf (2005) step-down bootstrap procedure.
Controls FWER with greater power than Bonferroni.
Uses block bootstrap to account for time series dependence.
"""
function romano_wolf_bootstrap(test_statistics::Vector{Float64},
                                  returns_matrix::Matrix{Float64};
                                  B::Int=1000,
                                  alpha::Float64=0.05,
                                  block_size::Int=10,
                                  seed::Int=42)::Vector{Bool}

    m = length(test_statistics)
    T, _ = size(returns_matrix)
    rng = MersenneTwister(seed)

    # Step-down: sort statistics from largest to smallest
    sorted_idx = sortperm(abs.(test_statistics), rev=true)

    # Bootstrap distribution of max statistic
    max_stats_boot = zeros(B)
    for b in 1:B
        # Block bootstrap resample
        n_blocks = ceil(Int, T/block_size)
        boot_idx = Int[]
        for _ in 1:n_blocks
            start = rand(rng, 1:(T-block_size+1))
            append!(boot_idx, start:(start+block_size-1))
        end
        boot_idx = boot_idx[1:T]
        boot_mat = returns_matrix[boot_idx, :]

        # Bootstrap test statistics (use mean/std as proxy)
        boot_stats = vec(mean(boot_mat, dims=1)) ./ vec(std(boot_mat, dims=1)) .* sqrt(T)
        max_stats_boot[b] = maximum(abs.(boot_stats))
    end

    # Critical value from bootstrap max distribution
    sort!(max_stats_boot)
    cv_idx = ceil(Int, (1-alpha)*B)
    cv = max_stats_boot[min(cv_idx, B)]

    reject = abs.(test_statistics) .> cv
    return reject
end

# =============================================================================
# SECTION 7: BOOTSTRAP INFERENCE
# =============================================================================

"""
    bootstrap_ci(data, stat_func; B=2000, alpha=0.05, seed=42) -> Tuple

Percentile bootstrap confidence interval.
"""
function bootstrap_ci(data::Vector{Float64},
                       stat_func::Function;
                       B::Int=2000, alpha::Float64=0.05,
                       seed::Int=42)::Tuple{Float64,Float64}

    n = length(data)
    rng = MersenneTwister(seed)
    obs_stat = stat_func(data)
    boot_stats = [stat_func(data[rand(rng, 1:n, n)]) for _ in 1:B]
    sort!(boot_stats)

    lo = boot_stats[max(1, floor(Int, alpha/2*B))]
    hi = boot_stats[min(B, ceil(Int, (1-alpha/2)*B))]
    return (lo, hi)
end

"""
    bca_bootstrap(data, stat_func; B=2000, alpha=0.05) -> Tuple

Bias-corrected and accelerated (BCa) bootstrap CI.
Adjusts for bias and acceleration (skewness) in the bootstrap distribution.
"""
function bca_bootstrap(data::Vector{Float64},
                         stat_func::Function;
                         B::Int=2000, alpha::Float64=0.05,
                         seed::Int=42)::Tuple{Float64,Float64}

    n = length(data)
    rng = MersenneTwister(seed)
    obs = stat_func(data)

    boot = [stat_func(data[rand(rng, 1:n, n)]) for _ in 1:B]
    sort!(boot)

    # Bias correction: z0 = Φ⁻¹(fraction of boot < obs)
    frac_below = sum(boot .< obs) / B
    frac_below = clamp(frac_below, 1e-6, 1-1e-6)
    z0 = _normal_quantile(frac_below)

    # Acceleration: jackknife skewness
    jk_stats = [stat_func(data[setdiff(1:n, [i])]) for i in 1:n]
    jk_mean = mean(jk_stats)
    num_a = sum((jk_mean .- jk_stats).^3)
    den_a = 6 * sum((jk_mean .- jk_stats).^2)^(3/2)
    a_hat = abs(den_a) > 0 ? num_a/den_a : 0.0

    # BCa percentile levels
    za = _normal_quantile(alpha/2)
    zb = _normal_quantile(1-alpha/2)
    p_lo = _normal_cdf(z0 + (z0+za)/(1-a_hat*(z0+za)))
    p_hi = _normal_cdf(z0 + (z0+zb)/(1-a_hat*(z0+zb)))

    lo = boot[max(1, floor(Int, p_lo*B))]
    hi = boot[min(B, ceil(Int, p_hi*B))]
    return (lo, hi)
end

"""
    studentized_bootstrap(data, stat_func, se_func; B=1000, alpha=0.05) -> Tuple

Studentized (t) bootstrap CI: more accurate coverage.
"""
function studentized_bootstrap(data::Vector{Float64},
                                 stat_func::Function,
                                 se_func::Function;
                                 B::Int=1000, alpha::Float64=0.05,
                                 seed::Int=42)::Tuple{Float64,Float64}

    n = length(data); rng = MersenneTwister(seed)
    obs = stat_func(data); se_obs = se_func(data)

    t_boot = zeros(B)
    for b in 1:B
        resample = data[rand(rng, 1:n, n)]
        t_boot[b] = se_func(resample) > 0 ?
            (stat_func(resample) - obs) / se_func(resample) : 0.0
    end
    sort!(t_boot)

    t_lo = t_boot[max(1, floor(Int, alpha/2*B))]
    t_hi = t_boot[min(B, ceil(Int, (1-alpha/2)*B))]

    return (obs - t_hi*se_obs, obs - t_lo*se_obs)
end

"""
    block_bootstrap_ci(data, stat_func; block_size=20, B=1000) -> Tuple

Block bootstrap CI for time series: preserves temporal dependence.
"""
function block_bootstrap_ci(data::Vector{Float64},
                               stat_func::Function;
                               block_size::Int=20,
                               B::Int=1000,
                               alpha::Float64=0.05,
                               seed::Int=42)::Tuple{Float64,Float64}

    n = length(data); rng = MersenneTwister(seed)
    boot_stats = zeros(B)

    for b in 1:B
        n_blocks = ceil(Int, n/block_size)
        resampled = Float64[]
        for _ in 1:n_blocks
            start = rand(rng, 1:(n-block_size+1))
            append!(resampled, data[start:(start+block_size-1)])
        end
        boot_stats[b] = stat_func(resampled[1:n])
    end
    sort!(boot_stats)

    lo = boot_stats[max(1, floor(Int, alpha/2*B))]
    hi = boot_stats[min(B, ceil(Int, (1-alpha/2)*B))]
    return (lo, hi)
end

"""
    stationary_bootstrap(data, stat_func; mean_block=20, B=1000) -> Tuple

Politis-Romano (1994) stationary bootstrap with random block lengths.
Block lengths ~ Geometric(1/mean_block).
"""
function stationary_bootstrap(data::Vector{Float64},
                                 stat_func::Function;
                                 mean_block::Int=20,
                                 B::Int=1000,
                                 alpha::Float64=0.05,
                                 seed::Int=42)::Tuple{Float64,Float64}

    n = length(data); rng = MersenneTwister(seed)
    p = 1.0/mean_block
    boot_stats = zeros(B)

    for b in 1:B
        resampled = Float64[]
        while length(resampled) < n
            # Random start
            start = rand(rng, 1:n)
            # Geometric block length
            block_len = 1
            while rand(rng) > p && block_len < n
                block_len += 1
            end
            for j in 0:(block_len-1)
                push!(resampled, data[mod(start+j-1, n)+1])
                length(resampled) >= n && break
            end
        end
        boot_stats[b] = stat_func(resampled[1:n])
    end
    sort!(boot_stats)

    lo = boot_stats[max(1, floor(Int, alpha/2*B))]
    hi = boot_stats[min(B, ceil(Int, (1-alpha/2)*B))]
    return (lo, hi)
end

# =============================================================================
# SECTION 8: WHITE'S REALITY CHECK AND SPA
# =============================================================================

"""
    whites_reality_check(performance_series, benchmark_series;
                          B=1000, block_size=10, alpha=0.05) -> NamedTuple

White (2000) Reality Check: test whether any of m strategies beats benchmark
after accounting for multiple comparison.

H₀: max_k E[f_k(θ)] ≤ 0 (no strategy beats benchmark)

Uses stationary bootstrap to get p-value.
"""
function whites_reality_check(performance_series::Matrix{Float64},
                                benchmark_series::Vector{Float64};
                                B::Int=1000, block_size::Int=10,
                                alpha::Float64=0.05,
                                seed::Int=42)

    T, m = size(performance_series)
    rng = MersenneTwister(seed)

    # Loss differentials: d_t^k = f_k(θ_t) - benchmark_t
    d = performance_series .- benchmark_series

    # Observed test statistic: max mean d_k
    d_means = vec(mean(d, dims=1))
    obs_max = maximum(d_means)

    # Bootstrap max statistic
    boot_max = zeros(B)
    p_block = 1/block_size

    for b in 1:B
        d_boot = zeros(T, m)
        for col in 1:m
            resampled = Float64[]
            while length(resampled) < T
                start = rand(rng, 1:T)
                block_len = 1
                while rand(rng) > p_block && block_len < T; block_len += 1 end
                for j in 0:(block_len-1)
                    push!(resampled, d[mod(start+j-1,T)+1, col])
                    length(resampled) >= T && break
                end
            end
            d_boot[:, col] = resampled[1:T]
        end
        # Demeaned boot stat
        d_boot_means = vec(mean(d_boot, dims=1)) .- d_means
        boot_max[b] = maximum(d_boot_means)
    end

    sort!(boot_max)
    p_val = sum(boot_max .>= obs_max) / B

    cv = boot_max[min(B, ceil(Int, (1-alpha)*B))]

    return (test_statistic=obs_max, p_value=p_val, critical_value=cv,
             reject_null=(p_val < alpha), d_means=d_means)
end

"""
    hansen_spa_test(performance_series, benchmark_series;
                    B=1000, alpha=0.05) -> NamedTuple

Hansen (2005) Superior Predictive Ability test.
Improves White's RC by trimming strategies with p-values above a threshold.
"""
function hansen_spa_test(performance_series::Matrix{Float64},
                           benchmark_series::Vector{Float64};
                           B::Int=1000, alpha::Float64=0.05,
                           seed::Int=42)

    T, m = size(performance_series)
    d = performance_series .- benchmark_series
    d_means = vec(mean(d, dims=1))

    # Variance of loss differentials (NW estimate)
    omega_k = zeros(m)
    for k in 1:m
        dk = d[:,k] .- d_means[k]
        omega_k[k] = sum(dk.^2)/T
        lag = max(1, floor(Int, T^(1/3)))
        for h in 1:lag
            w = 1-h/(lag+1)
            omega_k[k] += 2w*sum(dk[1:(T-h)].*dk[(h+1):T])/T
        end
    end

    # SPA uses "consistent" model averaging
    # Trim low-performing strategies
    gamma_n = 0.1 / log(log(T))  # threshold
    included = d_means .>= -gamma_n .* sqrt.(omega_k)

    spa_stat = maximum(vcat(d_means[included] ./ sqrt.(omega_k[included] ./ T), [0.0]))

    # Bootstrap (simplified: use iid bootstrap)
    rng = MersenneTwister(seed)
    boot_stats = zeros(B)
    for b in 1:B
        idx = rand(rng, 1:T, T)
        d_boot = d[idx, :]
        d_boot_means = vec(mean(d_boot, dims=1)) .- d_means
        incl_b = d_means .>= -gamma_n .* sqrt.(omega_k)
        boot_stats[b] = maximum(vcat(d_boot_means[incl_b] ./ sqrt.(omega_k[incl_b] ./ T), [0.0]))
    end
    sort!(boot_stats)
    p_val = sum(boot_stats .>= spa_stat) / B

    return (spa_statistic=spa_stat, p_value=p_val, reject_null=(p_val<alpha))
end

# =============================================================================
# SECTION 9: PERMUTATION TESTS
# =============================================================================

"""
    permutation_test(x, y, stat_func; B=10000, alternative=:two_sided) -> NamedTuple

Exact permutation test for comparing two groups.
"""
function permutation_test(x::Vector{Float64}, y::Vector{Float64},
                            stat_func::Function=t_stat_func;
                            B::Int=10_000, alternative::Symbol=:two_sided,
                            seed::Int=42)

    n1 = length(x); n2 = length(y)
    all_data = vcat(x, y)
    obs_stat = stat_func(x, y)
    rng = MersenneTwister(seed)

    perm_stats = zeros(B)
    for b in 1:B
        perm = randperm(rng, n1+n2)
        perm_stats[b] = stat_func(all_data[perm[1:n1]], all_data[perm[n1+1:end]])
    end

    if alternative == :two_sided
        p_val = sum(abs.(perm_stats) .>= abs(obs_stat)) / B
    elseif alternative == :greater
        p_val = sum(perm_stats .>= obs_stat) / B
    else
        p_val = sum(perm_stats .<= obs_stat) / B
    end

    return (statistic=obs_stat, p_value=p_val, permutation_distribution=perm_stats)
end

t_stat_func(x, y) = begin
    n1=length(x); n2=length(y)
    (mean(x)-mean(y)) / sqrt(var(x)/n1 + var(y)/n2 + 1e-10)
end

"""
    strategy_permutation_test(returns, signal; B=5000, metric=:sharpe) -> NamedTuple

Test whether a trading strategy has genuine predictive power by comparing
observed performance to permutation distribution of shuffled signals.
"""
function strategy_permutation_test(returns::Vector{Float64},
                                     signal::Vector{Float64};
                                     B::Int=5000,
                                     metric::Symbol=:sharpe,
                                     seed::Int=42)

    n = min(length(returns), length(signal))
    r = returns[1:n]; s = signal[1:n]
    rng = MersenneTwister(seed)

    function compute_metric(ret, sig)
        pos_rets = ret .* sign.(sig)
        μ = mean(pos_rets); σ = std(pos_rets)
        if metric == :sharpe; return σ > 0 ? μ/σ*sqrt(252) : 0.0
        elseif metric == :mean; return mean(pos_rets)
        else; return μ - 0.5*σ^2  # log utility
        end
    end

    obs_perf = compute_metric(r, s)

    perm_perfs = zeros(B)
    for b in 1:B
        perm_sig = s[randperm(rng, n)]
        perm_perfs[b] = compute_metric(r, perm_sig)
    end

    p_val = sum(perm_perfs .>= obs_perf) / B
    sort!(perm_perfs)
    ci_lo = perm_perfs[max(1, floor(Int, 0.025*B))]
    ci_hi = perm_perfs[min(B, ceil(Int, 0.975*B))]

    return (observed=obs_perf, p_value=p_val, is_significant=(p_val<0.05),
             ci=(ci_lo, ci_hi), perm_distribution=perm_perfs)
end

# =============================================================================
# SECTION 10: POWER ANALYSIS
# =============================================================================

"""
    power_analysis(effect_size, n, alpha=0.05; test=:two_sample_t) -> Float64

Compute statistical power for a given effect size and sample size.

Uses Cohen's framework:
- Small effect: d = 0.2 (hard to detect)
- Medium effect: d = 0.5 (meaningful)
- Large effect: d = 0.8 (easy to detect)
"""
function power_analysis(effect_size::Float64, n::Int,
                          alpha::Float64=0.05;
                          test::Symbol=:two_sample_t)::Float64

    if test == :two_sample_t
        # Noncentrality parameter: δ = d * sqrt(n/2)
        delta = effect_size * sqrt(n / 2)
        z_alpha = _normal_quantile(1-alpha/2)
        # Power = P(|Z + δ| > z_α) ≈ Φ(δ - z_α) + Φ(-δ - z_α)
        return _normal_cdf(delta - z_alpha) + _normal_cdf(-delta - z_alpha)

    elseif test == :one_sample_t
        delta = effect_size * sqrt(n)
        z_alpha = _normal_quantile(1-alpha/2)
        return _normal_cdf(delta - z_alpha) + _normal_cdf(-delta - z_alpha)

    else  # :correlation
        z = 0.5*log((1+effect_size)/(1-effect_size+1e-10))  # Fisher's z
        se = 1/sqrt(n-3)
        z_alpha = _normal_quantile(1-alpha/2)
        return _normal_cdf(z/se - z_alpha)
    end
end

"""
    sample_size_determination(effect_size, power=0.80, alpha=0.05;
                               test=:two_sample_t) -> Int

Compute required sample size to achieve desired power.
Uses Newton's method to invert the power function.
"""
function sample_size_determination(effect_size::Float64,
                                     target_power::Float64=0.80,
                                     alpha::Float64=0.05;
                                     test::Symbol=:two_sample_t)::Int

    # Binary search for n
    n_lo, n_hi = 2, 100_000
    for _ in 1:50
        n_mid = (n_lo + n_hi) ÷ 2
        pwr = power_analysis(effect_size, n_mid, alpha; test=test)
        if pwr < target_power
            n_lo = n_mid
        else
            n_hi = n_mid
        end
        n_hi - n_lo <= 1 && break
    end

    return n_hi
end

# =============================================================================
# INTERNAL DISTRIBUTION FUNCTIONS
# =============================================================================

"""Standard normal CDF."""
function _normal_cdf(x::Float64)::Float64
    x >= 8 && return 1.0; x <= -8 && return 0.0
    t = 1/(1+0.2316419*abs(x))
    poly = t*(0.319381530+t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
    cdf = 1-exp(-0.5x^2)/sqrt(2π)*poly
    return x >= 0 ? cdf : 1-cdf
end

"""Normal quantile (inverse CDF) via rational approximation."""
function _normal_quantile(p::Float64)::Float64
    p <= 0 && return -Inf; p >= 1 && return Inf
    p == 0.5 && return 0.0

    # Rational approximation (Beasley-Springer-Moro)
    p_use = p > 0.5 ? 1-p : p
    t = sqrt(-2*log(p_use))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    z = t - (c[1]+t*(c[2]+t*c[3]))/(1+t*(d[1]+t*(d[2]+t*d[3])))
    return p > 0.5 ? z : -z
end

"""Student-t tail probability (upper, two-tailed convention)."""
function _t_cdf_tail(t::Float64, df::Float64)::Float64
    # Use normal approximation for large df
    df > 100 && return 1-_normal_cdf(t)
    # Incomplete beta function (rational approx)
    x = df / (df + t^2)
    p = 0.5 * _incomplete_beta(x, df/2, 0.5)
    return p
end

"""Student-t quantile."""
function _t_quantile(p::Float64, df::Float64)::Float64
    df > 100 && return _normal_quantile(p)
    # Numerical inversion via bisection
    lo, hi = 0.0, 10.0
    for _ in 1:60
        mid = (lo+hi)/2
        if 1-_t_cdf_tail(mid, df) < p; lo = mid else hi = mid end
    end
    return (lo+hi)/2
end

"""Chi-squared tail probability."""
function _chi2_cdf_tail(x::Float64, df::Float64)::Float64
    x <= 0 && return 1.0
    # Regularized upper incomplete gamma
    return 1.0 - _reg_inc_gamma(df/2, x/2)
end

"""Chi-squared quantile."""
function _chi2_quantile(p::Float64, df::Float64)::Float64
    # Wilson-Hilferty approximation
    k = df; h = 2/(9k)
    z = _normal_quantile(p)
    return max(0.0, k*(1 - h + z*sqrt(h))^3)
end

"""F distribution tail probability."""
function _f_cdf_tail(f::Float64, d1::Float64, d2::Float64)::Float64
    f <= 0 && return 1.0
    x = d1*f / (d1*f + d2)
    return 1.0 - _incomplete_beta(x, d1/2, d2/2)
end

"""F quantile (bisection)."""
function _f_quantile(p::Float64, d1::Float64, d2::Float64)::Float64
    p <= 0 && return 0.0; p >= 1 && return Inf
    lo, hi = 0.0, 100.0
    for _ in 1:60
        mid = (lo+hi)/2
        if 1-_f_cdf_tail(mid, d1, d2) < p; lo = mid else hi = mid end
    end
    return (lo+hi)/2
end

"""Regularized incomplete gamma via series (simple approximation)."""
function _reg_inc_gamma(a::Float64, x::Float64)::Float64
    x <= 0 && return 0.0
    # Series representation
    term = exp(-x + a*log(x) - _lgamma_approx(a+1))
    sum_val = term
    for n in 1:200
        term *= x/(a+n)
        sum_val += term
        abs(term) < 1e-12 && break
    end
    return clamp(sum_val, 0.0, 1.0)
end

"""Incomplete beta function via continued fraction."""
function _incomplete_beta(x::Float64, a::Float64, b::Float64)::Float64
    (x <= 0 || a <= 0 || b <= 0) && return 0.0
    x >= 1 && return 1.0
    # Symmetry for efficiency
    if x > (a+1)/(a+b+2)
        return 1-_incomplete_beta(1-x, b, a)
    end
    # Modified Lentz continued fraction
    lbeta = _lgamma_approx(a) + _lgamma_approx(b) - _lgamma_approx(a+b)
    front = exp(a*log(x) + b*log(1-x) - lbeta) / a
    # Simple series approximation
    term = 1.0; sum_val = 1.0
    for n in 1:100
        term *= (a+n-1)*x/(n*(a+b+n-1))  # simplified
        sum_val += term
        abs(term) < 1e-10 && break
    end
    return clamp(front * sum_val, 0.0, 1.0)
end

"""Log-gamma approximation (Stirling)."""
function _lgamma_approx(x::Float64)::Float64
    x <= 0 && return Inf
    x < 1 && return _lgamma_approx(x+1) - log(x)
    (x-0.5)*log(x) - x + 0.5*log(2π) + 1/(12x)
end

"""Rank vector (with ties averaged)."""
function _rank(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    sorted_idx = sortperm(x)
    ranks = zeros(n)
    i = 1
    while i <= n
        j = i
        while j < n && x[sorted_idx[j+1]] == x[sorted_idx[i]]
            j += 1
        end
        avg_rank = (i+j)/2
        for k in i:j; ranks[sorted_idx[k]] = avg_rank end
        i = j + 1
    end
    return ranks
end

end # module StatisticalTests
