# systemic_risk.jl
# Systemic risk measurement for crypto/quant trading lab
# Pure Julia stdlib implementation

module SystemicRiskMeasures

using Statistics, LinearAlgebra, Random

# ============================================================
# DATA STRUCTURES
# ============================================================

struct SystemicRiskResult
    firm_names::Vector{String}
    covar::Vector{Float64}         # ΔCoVaR per firm
    mes::Vector{Float64}           # Marginal Expected Shortfall
    srisk::Vector{Float64}         # SRISK capital shortfall
    lrmes::Vector{Float64}         # Long-Run MES
    dip::Vector{Float64}           # Distressed Insurance Premium
    centrality::Vector{Float64}    # Network eigenvector centrality
    contagion_index::Vector{Float64}
end

struct GrangerRiskNetwork
    firms::Vector{String}
    adjacency::Matrix{Float64}     # directed: [i,j] = i Granger-causes j
    p_values::Matrix{Float64}
    threshold::Float64
end

struct ContagionScenario
    initial_defaults::Vector{Int}
    default_probabilities::Vector{Float64}
    rounds::Int
    total_affected_fraction::Float64
end

struct CryptoExchangeRisk
    exchanges::Vector{String}
    shared_token_matrix::Matrix{Float64}
    interconnectedness_score::Vector{Float64}
    systemic_importance::Vector{Float64}
end

# ============================================================
# QUANTILE REGRESSION (for CoVaR)
# ============================================================

"""
Quantile regression via interior point / simplex.
Minimizes sum of check function rho_tau(y - X*beta).
Uses iteratively reweighted least squares (IRLS) approximation.
"""
function quantile_regression(X::Matrix{Float64}, y::Vector{Float64}, tau::Float64;
                              max_iter::Int=500, tol::Float64=1e-8)
    n, p = size(X)
    beta = X \ y  # OLS starting point

    for iter in 1:max_iter
        resid = y - X * beta
        # Huber-like weights for quantile check function
        h = max(1e-6, quantile(abs.(resid[resid .!= 0]), 0.01))
        weights = map(r -> begin
            if abs(r) < h
                1.0 / (2 * h)
            elseif r > 0
                tau / abs(r + 1e-10)
            else
                (1 - tau) / abs(r - 1e-10)
            end
        end, resid)

        W = Diagonal(max.(weights, 1e-8))
        beta_new = (X' * W * X + 1e-8 * I) \ (X' * W * (y - (resid .* (weights .< 1/(2*h)) .* (tau - 0.5))))

        # Standard IRLS for quantile regression
        adj_y = y - resid .* ifelse.(resid .> 0, 1 - tau, -tau) ./ max.(abs.(resid), h)
        beta_new = (X' * W * X + 1e-8 * I) \ (X' * W * adj_y)

        if norm(beta_new - beta) < tol
            return beta_new
        end
        beta = beta_new
    end
    return beta
end

"""
Simple quantile regression via linear programming (Barrodale-Roberts).
Uses the simplex-based approach for small problems.
"""
function quantile_regression_simplex(X::Matrix{Float64}, y::Vector{Float64}, tau::Float64;
                                      max_iter::Int=1000)
    n, p = size(X)
    # Initialize with OLS
    beta = X \ y
    step_size = 0.1

    for iter in 1:max_iter
        resid = y - X * beta
        grad = -X' * (tau .- (resid .< 0))
        beta_new = beta - step_size * grad / n

        if norm(grad) / n < 1e-7
            break
        end
        # Armijo line search
        for ls in 1:20
            r_new = y - X * beta_new
            obj_new = sum(r_new .* (tau .- (r_new .< 0)))
            r_old = resid
            obj_old = sum(r_old .* (tau .- (r_old .< 0)))
            if obj_new < obj_old
                break
            end
            step_size *= 0.5
            beta_new = beta - step_size * grad / n
        end
        beta = beta_new
        step_size = min(step_size * 1.05, 1.0)
    end
    return beta
end

# ============================================================
# CoVaR (Adrian-Brunnermeier 2016)
# ============================================================

"""
Compute ΔCoVaR for each firm.
ΔCoVaR_i = CoVaR_{system|firm_i at VaR} - CoVaR_{system|firm_i at median}

Inputs:
- returns: T x N matrix of returns (columns = firms, last column = system/market)
- q: quantile level (e.g., 0.05 for 5% VaR)
"""
function compute_covar(returns::Matrix{Float64}, q::Float64=0.05;
                       system_col::Int=size(returns,2))
    T, N = size(returns)
    n_firms = N - 1  # exclude system column
    system_ret = returns[:, system_col]
    delta_covar = zeros(N - 1)

    # State variables: lagged system return, lagged volatility proxy
    for i in 1:(N-1)
        firm_ret = returns[:, i]

        # Build feature matrix for quantile regression
        # System return ~ alpha + beta1*firm_ret + controls
        X = hcat(ones(T-1), firm_ret[1:T-1], system_ret[1:T-1])
        y = system_ret[2:T]

        # Quantile regression at q level
        beta_q = quantile_regression_simplex(X, y, q)
        # Quantile regression at 0.5 level (median)
        beta_med = quantile_regression_simplex(X, y, 0.5)

        # VaR of firm i at quantile q
        X_firm = hcat(ones(T-1), firm_ret[1:T-1])
        beta_firm = quantile_regression_simplex(X_firm, firm_ret[2:T], q)

        firm_var_q = mean(X_firm * beta_firm)
        firm_var_med = mean(X_firm * quantile_regression_simplex(X_firm, firm_ret[2:T], 0.5))

        # CoVaR at VaR vs at median
        x_var = [1.0, firm_var_q, mean(system_ret[1:T-1])]
        x_med = [1.0, firm_var_med, mean(system_ret[1:T-1])]

        covar_at_var = dot(beta_q, x_var)
        covar_at_med = dot(beta_med, x_med)

        delta_covar[i] = covar_at_var - covar_at_med
    end

    return delta_covar
end

# ============================================================
# MES (Marginal Expected Shortfall)
# ============================================================

"""
Marginal Expected Shortfall: expected loss of firm i when
market return falls below threshold C (e.g., -2%).
MES_i = E[r_i | r_m < C]
"""
function compute_mes(returns::Matrix{Float64}, threshold::Float64=-0.02;
                     system_col::Int=size(returns,2))
    T, N = size(returns)
    system_ret = returns[:, system_col]
    mes = zeros(N - 1)

    stress_idx = findall(system_ret .< threshold)

    if length(stress_idx) == 0
        # Use bottom 5% if no observations below threshold
        cutoff = quantile(system_ret, 0.05)
        stress_idx = findall(system_ret .< cutoff)
    end

    for i in 1:(N-1)
        firm_ret = returns[:, i]
        mes[i] = mean(firm_ret[stress_idx])
    end

    return mes
end

"""
Bivariate DCC-GARCH estimate of MES using dynamic conditional correlation.
Uses rolling window approach for time-varying MES.
"""
function compute_mes_rolling(returns::Matrix{Float64}, threshold::Float64=-0.02;
                              window::Int=252, system_col::Int=size(returns,2))
    T, N = size(returns)
    system_ret = returns[:, system_col]
    mes_rolling = zeros(T, N-1)

    for t in (window+1):T
        w_start = t - window
        w_end = t - 1
        sub_returns = returns[w_start:w_end, :]
        mes_rolling[t, :] = compute_mes(sub_returns, threshold, system_col=system_col)
    end

    return mes_rolling
end

# ============================================================
# SRISK (Brownlees-Engle 2017)
# ============================================================

"""
SRISK: capital shortfall in a crisis scenario.
SRISK_i = max(0, k*(Debt_i + Equity_i*(1 - LRMES_i)) - Equity_i*(1 - LRMES_i))
        = max(0, k*Debt_i - (1-k)*Equity_i*(1 - LRMES_i))

where k is the prudential capital fraction (typically 8% for banks,
5.5% for broker-dealers), LRMES is the Long-Run MES over 6 months.
"""
function compute_srisk(equity::Vector{Float64}, debt::Vector{Float64},
                       lrmes::Vector{Float64}, k::Float64=0.08)
    N = length(equity)
    srisk = zeros(N)

    for i in 1:N
        # Expected equity in crisis
        equity_crisis = equity[i] * (1 - lrmes[i])
        # Capital shortfall
        srisk[i] = max(0.0, k * (debt[i] + equity_crisis) - equity_crisis)
    end

    return srisk
end

"""
SRISK as fraction of total system SRISK.
"""
function srisk_share(srisk::Vector{Float64})
    total = sum(max.(srisk, 0))
    total == 0 && return zeros(length(srisk))
    return max.(srisk, 0) ./ total
end

# ============================================================
# LRMES (Long-Run MES)
# ============================================================

"""
Long-Run Marginal Expected Shortfall via simulation.
Estimates expected firm loss if market falls by h% over 6 months.

Uses bivariate GJR-GARCH dynamics with DCC correlation.
Simplified: uses historical simulation with volatility scaling.
"""
function compute_lrmes(returns::Matrix{Float64};
                       horizon::Int=126,           # 6 months trading days
                       market_decline::Float64=0.40, # 40% market decline threshold
                       n_sim::Int=10000,
                       system_col::Int=size(returns,2),
                       rng::AbstractRNG=Random.default_rng())
    T, N = size(returns)
    n_firms = N - 1
    system_ret = returns[:, system_col]
    lrmes = zeros(n_firms)

    # Estimate GARCH(1,1) parameters for each series
    function garch11_filter(r::Vector{Float64})
        mu = mean(r)
        omega = var(r) * 0.05
        alpha = 0.1
        beta = 0.85
        T_r = length(r)
        sigma2 = zeros(T_r)
        sigma2[1] = var(r)
        for t in 2:T_r
            sigma2[t] = omega + alpha * (r[t-1] - mu)^2 + beta * sigma2[t-1]
        end
        return sigma2
    end

    sigma2_sys = garch11_filter(system_ret)
    last_sigma_sys = sqrt(sigma2_sys[end])

    for i in 1:n_firms
        firm_ret = returns[:, i]
        sigma2_firm = garch11_filter(firm_ret)
        last_sigma_firm = sqrt(sigma2_firm[end])

        # Correlation from recent window
        recent_T = min(60, T)
        rho = cor(system_ret[end-recent_T+1:end], firm_ret[end-recent_T+1:end])
        rho = clamp(rho, -0.999, 0.999)

        # Simulate joint paths using Cholesky decomposition
        L = [1.0 0.0; rho sqrt(1 - rho^2)]

        stress_losses = Float64[]
        for sim in 1:n_sim
            # Simulate horizon-day path
            cum_sys = 0.0
            cum_firm = 0.0
            sig_s = last_sigma_sys
            sig_f = last_sigma_firm

            for d in 1:horizon
                z = L * randn(rng, 2)
                ret_s = z[1] * sig_s
                ret_f = z[2] * sig_f
                cum_sys += ret_s
                cum_firm += ret_f
                # GARCH update
                sig_s = sqrt(max(1e-10, var(system_ret) * 0.05 + 0.1 * ret_s^2 + 0.85 * sig_s^2))
                sig_f = sqrt(max(1e-10, var(firm_ret) * 0.05 + 0.1 * ret_f^2 + 0.85 * sig_f^2))
            end

            if cum_sys < -market_decline
                push!(stress_losses, -cum_firm)
            end
        end

        lrmes[i] = isempty(stress_losses) ? 0.0 : mean(stress_losses)
    end

    return lrmes
end

# ============================================================
# DIP (Distressed Insurance Premium)
# ============================================================

"""
Distressed Insurance Premium (Huang, Zhou, Zhu 2009).
DIP measures the cost of insuring against systemic financial distress.
Approximated as: DIP = π * (1-R) * LGD_contribution

Using CDS spread proxy and default correlations.
"""
function compute_dip(returns::Matrix{Float64}, default_barrier::Float64=-0.5;
                     recovery_rate::Float64=0.4,
                     risk_free_rate::Float64=0.02,
                     T_horizon::Float64=1.0)
    n_obs, N = size(returns)
    n_firms = N - 1
    dip = zeros(n_firms)

    for i in 1:n_firms
        r = returns[:, i]

        # Estimate default probability via equity return dynamics
        mu_hat = mean(r) * 252
        sigma_hat = std(r) * sqrt(252)

        # Black-Scholes implied default prob (simplified Merton)
        # P(default) = N(-d2) where d2 = (log(V/B) + (r - σ²/2)*T) / (σ*√T)
        # Use return distribution as proxy: P(r < barrier)
        if sigma_hat > 0
            d2 = (mu_hat - risk_free_rate - 0.5 * sigma_hat^2) * T_horizon / (sigma_hat * sqrt(T_horizon))
            # Normal CDF approximation
            p_default = normal_cdf(-d2 + log(1.0) / (sigma_hat * sqrt(T_horizon)))
        else
            p_default = 0.0
        end

        p_default = clamp(p_default, 0.0, 1.0)
        lgd = 1.0 - recovery_rate

        # DIP = risk-neutral default probability * LGD * notional (normalized)
        dip[i] = p_default * lgd * exp(-risk_free_rate * T_horizon)
    end

    return dip
end

"""Normal CDF approximation (Abramowitz & Stegun)."""
function normal_cdf(x::Float64)
    if x < -8.0; return 0.0; end
    if x > 8.0; return 1.0; end
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    t = 1.0 / (1.0 + p * abs(x))
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    pdf_val = exp(-0.5 * x^2) / sqrt(2π)
    result = 1.0 - pdf_val * poly
    return x >= 0 ? result : 1.0 - result
end

# ============================================================
# NETWORK-BASED SYSTEMIC RISK
# ============================================================

"""
Build loss network from return correlations.
Edge weight = max(0, negative correlation * magnitude).
"""
function build_loss_network(returns::Matrix{Float64}, threshold::Float64=0.3)
    T, N = size(returns)
    C = cor(returns)
    # Loss network: strong negative correlations indicate systemic links
    # Use absolute correlation for undirected loss propagation
    W = zeros(N, N)
    for i in 1:N
        for j in 1:N
            if i != j && abs(C[i,j]) > threshold
                W[i,j] = abs(C[i,j])
            end
        end
    end
    return W
end

"""
Eigenvector centrality of the loss network.
Power iteration method.
"""
function eigenvector_centrality(W::Matrix{Float64}; max_iter::Int=1000, tol::Float64=1e-10)
    N = size(W, 1)
    v = ones(N) / N

    for iter in 1:max_iter
        v_new = W * v
        norm_v = norm(v_new)
        norm_v < 1e-10 && break
        v_new ./= norm_v
        if norm(v_new - v) < tol
            return v_new
        end
        v = v_new
    end

    return v
end

"""
PageRank-style systemic importance score.
Accounts for directionality of contagion.
"""
function systemic_pagerank(W::Matrix{Float64}; damping::Float64=0.85, max_iter::Int=500)
    N = size(W, 1)

    # Normalize columns
    col_sums = sum(W, dims=1)[:]
    W_norm = copy(W)
    for j in 1:N
        if col_sums[j] > 0
            W_norm[:, j] ./= col_sums[j]
        else
            W_norm[:, j] .= 1.0 / N
        end
    end

    pr = ones(N) / N
    teleport = ones(N) / N

    for iter in 1:max_iter
        pr_new = damping * W_norm * pr + (1 - damping) * teleport
        if norm(pr_new - pr) < 1e-10
            return pr_new
        end
        pr = pr_new
    end

    return pr
end

# ============================================================
# GRANGER-CAUSALITY RISK NETWORK
# ============================================================

"""
Test Granger causality from series x to series y at lag p.
H0: x does not Granger-cause y.
Returns F-statistic and approximate p-value.
"""
function granger_test(x::Vector{Float64}, y::Vector{Float64}, lag::Int=5)
    T = length(y)
    T < 2 * lag + 10 && return (0.0, 1.0)

    # Restricted model: y ~ lags of y only
    Y = y[lag+1:T]
    n = length(Y)

    # Build Y lag matrix
    X_R = ones(n, lag + 1)
    for l in 1:lag
        X_R[:, l+1] = y[lag+1-l:T-l]
    end

    # Unrestricted model: y ~ lags of y + lags of x
    X_U = hcat(X_R, zeros(n, lag))
    for l in 1:lag
        X_U[:, lag+1+l] = x[lag+1-l:T-l]
    end

    function ssr(X::Matrix{Float64}, Y::Vector{Float64})
        beta = (X' * X + 1e-10 * I) \ (X' * Y)
        resid = Y - X * beta
        return sum(resid.^2)
    end

    RSS_R = ssr(X_R, Y)
    RSS_U = ssr(X_U, Y)

    p_r = size(X_R, 2)
    p_u = size(X_U, 2)
    df1 = p_u - p_r
    df2 = n - p_u

    df2 <= 0 && return (0.0, 1.0)
    RSS_U < 1e-15 && return (0.0, 1.0)

    F = ((RSS_R - RSS_U) / df1) / (RSS_U / df2)
    F = max(0.0, F)

    # Approximate p-value using F-distribution tail
    p_val = f_pvalue(F, df1, df2)

    return (F, p_val)
end

"""Approximate p-value for F-distribution using beta function approximation."""
function f_pvalue(F::Float64, d1::Int, d2::Int)
    F <= 0 && return 1.0
    x = d1 * F / (d1 * F + d2)
    # Regularized incomplete beta function (approximation)
    return 1.0 - regularized_beta(x, d1 / 2.0, d2 / 2.0)
end

"""Regularized incomplete beta function via continued fraction."""
function regularized_beta(x::Float64, a::Float64, b::Float64)
    x <= 0 && return 0.0
    x >= 1 && return 1.0

    # Use symmetry relation for better convergence
    if x > (a + 1) / (a + b + 2)
        return 1.0 - regularized_beta(1 - x, b, a)
    end

    log_beta_ab = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * log(x) + b * log(1 - x) - log_beta_ab) / a

    # Lentz continued fraction
    f = 1.0
    C = 1.0
    D = 1.0 - (a + b) * x / (a + 1)
    abs(D) < 1e-30 && (D = 1e-30)
    D = 1.0 / D
    f = D

    for m in 1:200
        # Even step
        aa = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        D = 1.0 + aa * D
        abs(D) < 1e-30 && (D = 1e-30)
        C = 1.0 + aa / C
        abs(C) < 1e-30 && (C = 1e-30)
        D = 1.0 / D
        f *= C * D

        # Odd step
        aa = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        D = 1.0 + aa * D
        abs(D) < 1e-30 && (D = 1e-30)
        C = 1.0 + aa / C
        abs(C) < 1e-30 && (C = 1e-30)
        D = 1.0 / D
        delta = C * D
        f *= delta

        abs(delta - 1.0) < 1e-10 && break
    end

    return front * f
end

"""
Build full Granger-causality risk network.
Returns adjacency matrix where A[i,j] = F-stat if i significantly causes j.
"""
function build_granger_network(returns::Matrix{Float64}, lag::Int=5,
                                significance::Float64=0.05)
    T, N = size(returns)
    adj = zeros(N, N)
    pvals = ones(N, N)

    for i in 1:N
        for j in 1:N
            if i == j; continue; end
            F, pv = granger_test(returns[:, i], returns[:, j], lag)
            pvals[i, j] = pv
            if pv < significance
                adj[i, j] = F
            end
        end
    end

    return GrangerRiskNetwork(
        ["Asset_$k" for k in 1:N],
        adj, pvals, significance
    )
end

# ============================================================
# CONTAGION INDEX
# ============================================================

"""
Simulate contagion from initial defaulting firms.
Uses a threshold model: firm j defaults if fraction of its
neighbors that have defaulted exceeds threshold phi.
"""
function simulate_contagion(W::Matrix{Float64},
                             initial_defaults::Vector{Int},
                             phi::Float64=0.3;
                             max_rounds::Int=100)
    N = size(W, 1)
    defaulted = falses(N)
    for i in initial_defaults
        1 <= i <= N && (defaulted[i] = true)
    end

    rounds = 0
    for round in 1:max_rounds
        new_defaults = false
        for j in 1:N
            defaulted[j] && continue

            # Fraction of weighted neighbors that have defaulted
            total_exposure = sum(W[:, j])
            total_exposure < 1e-10 && continue

            defaulted_exposure = sum(W[i, j] for i in 1:N if defaulted[i])
            if defaulted_exposure / total_exposure > phi
                defaulted[j] = true
                new_defaults = true
            end
        end
        rounds += 1
        !new_defaults && break
    end

    return ContagionScenario(
        initial_defaults,
        Float64.(defaulted),
        rounds,
        mean(defaulted)
    )
end

"""
Contagion index: average fraction of system affected when each firm defaults alone.
"""
function contagion_index(returns::Matrix{Float64}; phi::Float64=0.3)
    T, N = size(returns)
    W = build_loss_network(returns)
    indices = zeros(N)

    for i in 1:N
        scenario = simulate_contagion(W, [i], phi)
        indices[i] = scenario.total_affected_fraction
    end

    return indices
end

# ============================================================
# STRESS TESTING: SIMULTANEOUS DEFAULT SCENARIOS
# ============================================================

struct StressTestResult
    scenario_name::String
    defaulted_firms::Vector{Int}
    contagion_fraction::Float64
    system_loss::Float64
    capital_shortfall::Float64
end

"""
Run simultaneous default stress scenarios.
"""
function stress_test_simultaneous(returns::Matrix{Float64},
                                   equity::Vector{Float64},
                                   debt::Vector{Float64};
                                   k::Float64=0.08,
                                   scenarios::Vector{Vector{Int}}=Vector{Int}[])
    T, N = size(returns)
    n_firms = N - 1
    W = build_loss_network(returns)

    # If no scenarios provided, generate automatic scenarios
    if isempty(scenarios)
        # Single firm defaults (each firm)
        for i in 1:n_firms
            push!(scenarios, [i])
        end
        # Top 3 firms by centrality default together
        centrality = eigenvector_centrality(W[1:n_firms, 1:n_firms])
        top3 = sortperm(centrality, rev=true)[1:min(3, n_firms)]
        push!(scenarios, top3)
    end

    results = StressTestResult[]

    for scenario in scenarios
        contagion = simulate_contagion(W[1:n_firms, 1:n_firms], scenario)

        # Estimate system loss
        defaulted_set = Set(scenario)
        # Add contagion defaults (stochastic threshold exceeded)
        for j in 1:n_firms
            contagion.default_probabilities[j] > 0.5 && push!(defaulted_set, j)
        end

        total_loss = sum(equity[j] for j in defaulted_set if j <= length(equity))
        cap_shortfall = sum(max(0.0, k * debt[j] - (1-k) * equity[j])
                           for j in defaulted_set if j <= length(equity))

        push!(results, StressTestResult(
            "Scenario_$(join(scenario, '_'))",
            collect(defaulted_set),
            contagion.total_affected_fraction,
            total_loss,
            cap_shortfall
        ))
    end

    return results
end

# ============================================================
# CRYPTO EXCHANGE INTERCONNECTEDNESS
# ============================================================

"""
Measure exchange interconnectedness from shared token holdings.
Exchanges sharing large fractions of the same tokens are more interconnected.
"""
function crypto_exchange_interconnectedness(
    exchanges::Vector{String},
    token_holdings::Matrix{Float64}  # n_exchanges x n_tokens, portfolio weights
)
    n_ex = length(exchanges)
    n_tok = size(token_holdings, 2)

    # Normalize rows to get portfolio weights
    holdings_norm = copy(token_holdings)
    for i in 1:n_ex
        s = sum(holdings_norm[i, :])
        s > 0 && (holdings_norm[i, :] ./= s)
    end

    # Shared token exposure matrix
    shared = zeros(n_ex, n_ex)
    for i in 1:n_ex
        for j in 1:n_ex
            # Cosine similarity of token portfolios
            dot_ij = dot(holdings_norm[i, :], holdings_norm[j, :])
            norm_i = norm(holdings_norm[i, :])
            norm_j = norm(holdings_norm[j, :])
            if norm_i > 0 && norm_j > 0
                shared[i, j] = dot_ij / (norm_i * norm_j)
            end
        end
    end

    # Eigenvector centrality as systemic importance
    centrality = eigenvector_centrality(shared)

    # Systemic importance: centrality * total holdings
    total_holdings = vec(sum(token_holdings, dims=2))
    total_holdings_norm = total_holdings / max(sum(total_holdings), 1e-10)
    systemic_importance = centrality .* total_holdings_norm
    systemic_importance ./= max(sum(systemic_importance), 1e-10)

    return CryptoExchangeRisk(exchanges, shared, centrality, systemic_importance)
end

"""
Exchange failure cascade: what happens when exchange i fails?
"""
function exchange_failure_cascade(
    exchange_risk::CryptoExchangeRisk,
    failing_exchange::Int;
    contagion_threshold::Float64=0.5
)
    n_ex = length(exchange_risk.exchanges)
    W = exchange_risk.shared_token_matrix

    failed = falses(n_ex)
    failed[failing_exchange] = true

    cascade_rounds = 0
    for round in 1:n_ex
        new_failure = false
        for j in 1:n_ex
            failed[j] && continue
            # Exposure to failed exchanges
            failed_exposure = sum(W[i, j] for i in 1:n_ex if failed[i])
            total_exposure = sum(W[:, j])
            total_exposure < 1e-10 && continue

            if failed_exposure / total_exposure > contagion_threshold
                failed[j] = true
                new_failure = true
            end
        end
        cascade_rounds += 1
        !new_failure && break
    end

    return (
        failed_exchanges=findall(failed),
        fraction_affected=mean(failed),
        rounds=cascade_rounds
    )
end

# ============================================================
# UNIFIED SYSTEMIC RISK FRAMEWORK
# ============================================================

"""
Compute all systemic risk measures for a set of firms.
Returns a SystemicRiskResult with all measures.
"""
function compute_all_systemic_risk(
    returns::Matrix{Float64},
    equity::Vector{Float64},
    debt::Vector{Float64};
    firm_names::Union{Vector{String}, Nothing}=nothing,
    var_level::Float64=0.05,
    mes_threshold::Float64=-0.02,
    k::Float64=0.08,
    n_lrmes_sim::Int=5000,
    rng::AbstractRNG=Random.default_rng()
)
    T, N = size(returns)
    n_firms = N - 1

    names = firm_names === nothing ? ["Firm_$i" for i in 1:n_firms] : firm_names

    println("Computing CoVaR...")
    delta_covar = compute_covar(returns, var_level)

    println("Computing MES...")
    mes = compute_mes(returns, mes_threshold)

    println("Computing LRMES...")
    lrmes = compute_lrmes(returns, n_sim=n_lrmes_sim, rng=rng)

    println("Computing SRISK...")
    srisk = compute_srisk(equity, debt, lrmes, k)

    println("Computing DIP...")
    dip = compute_dip(returns)

    println("Building loss network...")
    W = build_loss_network(returns)
    centrality = eigenvector_centrality(W[1:n_firms, 1:n_firms])

    println("Computing contagion index...")
    ci = contagion_index(returns)

    return SystemicRiskResult(names, delta_covar, mes, srisk, lrmes, dip,
                               centrality, ci)
end

"""
Print summary table of systemic risk measures.
"""
function print_systemic_risk_summary(result::SystemicRiskResult)
    n = length(result.firm_names)
    println("\n=== SYSTEMIC RISK SUMMARY ===")
    println(rpad("Firm", 12), rpad("ΔCoVaR", 10), rpad("MES", 10),
            rpad("SRISK", 12), rpad("LRMES", 10), rpad("DIP", 10),
            rpad("Centrality", 12), rpad("Contagion", 10))
    println("-"^86)

    for i in 1:n
        println(
            rpad(result.firm_names[i], 12),
            rpad(round(result.covar[i], digits=4), 10),
            rpad(round(result.mes[i], digits=4), 10),
            rpad(round(result.srisk[i], digits=2), 12),
            rpad(round(result.lrmes[i], digits=4), 10),
            rpad(round(result.dip[i], digits=4), 10),
            rpad(round(result.centrality[i], digits=4), 12),
            rpad(round(result.contagion_index[i], digits=4), 10)
        )
    end
end

# ============================================================
# RANKING AND AGGREGATION
# ============================================================

"""
Composite systemic risk score using equal-weighted z-scores of all measures.
"""
function composite_systemic_score(result::SystemicRiskResult)
    n = length(result.firm_names)

    function zscore(v::Vector{Float64})
        mu = mean(v)
        s = std(v)
        s < 1e-10 && return zeros(length(v))
        return (v .- mu) ./ s
    end

    # Higher values = more systemic risk
    scores = (
        zscore(-result.covar) +     # more negative = more systemic
        zscore(-result.mes) +        # more negative = more systemic
        zscore(result.srisk) +
        zscore(result.lrmes) +
        zscore(result.dip) +
        zscore(result.centrality) +
        zscore(result.contagion_index)
    ) / 7.0

    return scores
end

"""
Rank firms by composite systemic risk score.
"""
function rank_systemic_risk(result::SystemicRiskResult)
    scores = composite_systemic_score(result)
    ranking = sortperm(scores, rev=true)

    println("\n=== SYSTEMIC RISK RANKING ===")
    for (rank, idx) in enumerate(ranking)
        println("$rank. $(result.firm_names[idx]) — score: $(round(scores[idx], digits=4))")
    end

    return ranking, scores
end

# ============================================================
# DEMO / TEST FUNCTIONS
# ============================================================

"""Generate synthetic returns for testing."""
function generate_test_returns(n_firms::Int=5, T::Int=1000;
                                rng::AbstractRNG=Random.default_rng())
    # Generate correlated returns: firms + market factor
    factor = randn(rng, T) * 0.01  # market factor
    returns = zeros(T, n_firms + 1)  # last column = market

    betas = 0.5 .+ rand(rng, n_firms)  # loadings between 0.5 and 1.5
    for i in 1:n_firms
        idio = randn(rng, T) * 0.015
        returns[:, i] = betas[i] * factor + idio
    end
    returns[:, end] = factor + randn(rng, T) * 0.005  # market

    return returns
end

"""Run a complete systemic risk analysis demo."""
function demo_systemic_risk(; seed::Int=42)
    rng = MersenneTwister(seed)
    println("Generating test data...")
    returns = generate_test_returns(5, 500, rng=rng)

    n_firms = size(returns, 2) - 1
    equity = 1e9 .* (1.0 .+ rand(rng, n_firms))
    debt = 5e9 .* (1.0 .+ rand(rng, n_firms))

    result = compute_all_systemic_risk(returns, equity, debt,
                                        firm_names=["BTC_ex", "ETH_ex", "BNB_ex",
                                                    "SOL_ex", "ADA_ex"],
                                        n_lrmes_sim=500, rng=rng)

    print_systemic_risk_summary(result)
    rank_systemic_risk(result)

    # Granger network
    println("\nBuilding Granger causality network...")
    gn = build_granger_network(returns, 3)
    n_edges = count(gn.adjacency .> 0)
    println("Number of significant causal links: $n_edges / $(n_firms*(n_firms-1))")

    # Stress test
    println("\nRunning stress tests...")
    stress = stress_test_simultaneous(returns, equity, debt)
    println("Stress scenarios completed: $(length(stress))")
    for s in stress[1:min(3, length(stress))]
        println("  $(s.scenario_name): contagion=$(round(s.contagion_fraction, digits=3)), loss=$(round(s.system_loss/1e9, digits=2))B")
    end

    return result
end


# ============================================================
# ADDITIONAL SYSTEMIC RISK MODULES
# ============================================================

# ============================================================
# ABSORBING MARKOV CHAIN CONTAGION MODEL
# ============================================================

"""
Model financial system as absorbing Markov chain.
States: solvent (1..N), default (N+1 = absorbing).
Transition probabilities from correlation structure.
"""
function absorbing_markov_contagion(
    returns::Matrix{Float64},
    default_barriers::Vector{Float64};
    dt::Float64=1.0/252
)
    T, N = size(returns)
    C = cor(returns)
    vols = vec(std(returns, dims=1)) .* sqrt(252)

    # Default probability per firm (Black-Scholes Merton)
    mu_annual = vec(mean(returns, dims=1)) .* 252
    default_probs = zeros(N)
    for i in 1:N
        sigma = vols[i]
        mu = mu_annual[i]
        d2 = (log(1.0 / max(default_barriers[i], 1e-10)) + (mu - 0.5*sigma^2)) /
              max(sigma, 1e-10)
        default_probs[i] = normal_cdf(-d2)
    end

    # Transition matrix (simplified: independent defaults + correlation)
    P = zeros(N+1, N+1)
    for i in 1:N
        P[i, N+1] = default_probs[i]
        P[i, i] = 1.0 - default_probs[i]
        # Small contagion transitions
        contagion = 0.01 * (C[i, :] .* default_probs)
        for j in 1:N
            if j != i && contagion[j] > 0
                transfer = min(contagion[j], P[i, i])
                P[i, j] += transfer
                P[i, i] -= transfer
            end
        end
    end
    P[N+1, N+1] = 1.0  # absorbing state

    # Expected time to absorption (default)
    # Q = P[1:N, 1:N] (transient submatrix)
    Q = P[1:N, 1:N]
    N_matrix = inv(I - Q + 1e-8*I)  # fundamental matrix
    expected_time_to_default = vec(sum(N_matrix, dims=2))

    return (transition_matrix=P, default_probs=default_probs,
            expected_time_to_default=expected_time_to_default,
            fundamental_matrix=N_matrix)
end

# ============================================================
# FINANCIAL STABILITY INDEX (FSI)
# ============================================================

"""
Financial Stability Index: composite measure combining
multiple systemic risk dimensions.
"""
struct FSI
    date_idx::Vector{Int}
    fsi::Vector{Float64}         # composite index
    credit_component::Vector{Float64}
    equity_component::Vector{Float64}
    fx_component::Vector{Float64}
    funding_component::Vector{Float64}
    weights::Vector{Float64}
end

"""
Compute Financial Stability Index from multiple market indicators.
"""
function compute_fsi(
    credit_spreads::Vector{Float64},       # e.g., CDS spreads, OAS
    equity_vol::Vector{Float64},           # realized/implied equity vol
    fx_vol::Vector{Float64},               # FX volatility
    funding_stress::Vector{Float64};       # Libor-OIS or FRA-OIS
    window::Int=60,
    weights::Vector{Float64}=[0.35, 0.30, 0.15, 0.20]
)
    T = minimum(length.([credit_spreads, equity_vol, fx_vol, funding_stress]))

    function rolling_zscore(x::Vector{Float64})
        result = zeros(T)
        for t in window+1:T
            sub = x[max(1,t-window):t]
            mu, sigma = mean(sub), std(sub)
            result[t] = (x[t] - mu) / max(sigma, 1e-10)
        end
        return result
    end

    credit_z = rolling_zscore(credit_spreads[1:T])
    equity_z = rolling_zscore(equity_vol[1:T])
    fx_z = rolling_zscore(fx_vol[1:T])
    funding_z = rolling_zscore(funding_stress[1:T])

    fsi = weights[1]*credit_z + weights[2]*equity_z + weights[3]*fx_z + weights[4]*funding_z

    return FSI(1:T, fsi, credit_z, equity_z, fx_z, funding_z, weights)
end

"""
Identify stress episodes from FSI.
"""
function fsi_stress_episodes(fsi::FSI; threshold::Float64=1.5)
    in_stress = fsi.fsi .> threshold
    episodes = Tuple{Int,Int}[]
    start_t = 0

    for t in 1:length(fsi.fsi)
        if in_stress[t] && start_t == 0
            start_t = t
        elseif !in_stress[t] && start_t > 0
            push!(episodes, (start_t, t-1))
            start_t = 0
        end
    end
    start_t > 0 && push!(episodes, (start_t, length(fsi.fsi)))

    return (episodes=episodes, n_episodes=length(episodes),
            avg_duration=isempty(episodes) ? 0.0 : mean(e[2]-e[1]+1 for e in episodes),
            max_fsi=maximum(fsi.fsi))
end

# ============================================================
# INTERCONNECTEDNESS MEASURES
# ============================================================

"""
Absorption ratio: fraction of total variance explained by first K eigenvectors.
High AR = high interconnectedness = fragile system.
"""
function absorption_ratio(returns::Matrix{Float64}; k_fraction::Float64=0.2)
    T, N = size(returns)
    C = cov(returns)
    F = eigen(Symmetric(C), sortby=x->-x)
    k = max(1, round(Int, k_fraction * N))
    total_var = sum(max.(F.values, 0.0))
    top_var = sum(max.(F.values[1:k], 0.0))
    return top_var / max(total_var, 1e-10)
end

"""
Rolling absorption ratio time series.
"""
function rolling_absorption_ratio(returns::Matrix{Float64}; window::Int=252, k_fraction::Float64=0.2)
    T, N = size(returns)
    ar = zeros(T)
    for t in window:T
        ar[t] = absorption_ratio(returns[t-window+1:t, :], k_fraction=k_fraction)
    end
    return ar
end

"""
Turbulence index (Kritzman-Li 2010).
Measures abnormality of current returns relative to historical covariance.
d_t = (r_t - mu)' * Sigma^{-1} * (r_t - mu)
"""
function turbulence_index(returns::Matrix{Float64}; window::Int=252)
    T, N = size(returns)
    ti = zeros(T)

    for t in window:T
        hist = returns[t-window+1:t-1, :]
        mu = vec(mean(hist, dims=1))
        Sigma = cov(hist) + 1e-8*I
        r_t = returns[t, :] - mu
        try
            ti[t] = dot(r_t, Sigma \ r_t) / N
        catch
            ti[t] = NaN
        end
    end

    return ti
end

"""
Systemic turbulence: fraction of assets in simultaneous distress.
"""
function systemic_turbulence(returns::Matrix{Float64}; window::Int=252,
                               distress_quantile::Float64=0.05)
    T, N = size(returns)
    st = zeros(T)

    for t in window:T
        hist_returns = returns[t-window+1:t-1, :]
        current = returns[t, :]
        distress_count = sum(current[i] < quantile(hist_returns[:, i], distress_quantile) for i in 1:N)
        st[t] = distress_count / N
    end

    return st
end

# ============================================================
# TAIL DEPENDENCE AND EXTREME EVENT ANALYSIS
# ============================================================

"""
Compute tail dependence coefficient between two series.
Lower tail: λ_L = lim_{u→0} P(X≤F^{-1}(u) | Y≤G^{-1}(u))
"""
function tail_dependence(x::Vector{Float64}, y::Vector{Float64}; threshold::Float64=0.05)
    n = length(x)
    rank_x = ordinal_ranks_sys(x) ./ (n + 1)
    rank_y = ordinal_ranks_sys(y) ./ (n + 1)

    # Lower tail
    joint_lower = sum((rank_x .<= threshold) .& (rank_y .<= threshold))
    lambda_L = joint_lower / max(sum(rank_x .<= threshold), 1)

    # Upper tail
    joint_upper = sum((rank_x .> 1-threshold) .& (rank_y .> 1-threshold))
    lambda_U = joint_upper / max(sum(rank_x .> 1-threshold), 1)

    return (lambda_lower=lambda_L, lambda_upper=lambda_U)
end

function ordinal_ranks_sys(x::Vector{Float64})
    n = length(x); idx = sortperm(x); ranks = zeros(n)
    for (r, i) in enumerate(idx); ranks[i] = Float64(r); end
    return ranks
end

"""
CoES (Co-Expected Shortfall): joint extreme loss measure.
"""
function co_expected_shortfall(returns::Matrix{Float64};
                                q::Float64=0.05,
                                system_col::Int=size(returns,2))
    T, N = size(returns)
    sys = returns[:, system_col]
    threshold = quantile(sys, q)
    stress_idx = findall(sys .<= threshold)

    co_es = zeros(N-1)
    for i in 1:N-1
        co_es[i] = isempty(stress_idx) ? 0.0 : mean(returns[stress_idx, i])
    end

    return co_es
end

"""
Extreme event clustering: time between tail events (exceedances).
"""
function extreme_clustering(returns::Vector{Float64}; threshold_q::Float64=0.05)
    n = length(returns)
    thr = quantile(returns, threshold_q)
    exceedance_times = findall(returns .<= thr)

    if length(exceedance_times) < 2
        return (mean_duration=NaN, clustering_coef=NaN, exceedances=exceedance_times)
    end

    inter_arrival = diff(exceedance_times)
    mean_dur = mean(inter_arrival)
    # Clustering: fraction of back-to-back exceedances
    clustering = sum(inter_arrival .== 1) / length(inter_arrival)

    return (mean_duration=mean_dur, clustering_coef=clustering,
            exceedances=exceedance_times, inter_arrival_times=inter_arrival)
end

# ============================================================
# SYSTEMIC RISK FORECASTING
# ============================================================

"""
Predict systemic risk one period ahead using lagged indicators.
Uses quantile regression to model CoVaR dynamics.
"""
function forecast_systemic_risk(
    covar_history::Vector{Float64},
    mes_history::Vector{Float64},
    srisk_history::Vector{Float64},
    market_returns::Vector{Float64};
    lag::Int=5,
    horizon::Int=1,
    quantile_level::Float64=0.05
)
    T = minimum(length.([covar_history, mes_history, srisk_history, market_returns]))
    T < lag + 10 && return zeros(3)

    n = T - lag - horizon + 1
    Y_covar = covar_history[lag+horizon:T]
    Y_mes = mes_history[lag+horizon:T]

    # Feature matrix: lagged values
    X = hcat(
        ones(n),
        covar_history[lag:T-horizon],
        mes_history[lag:T-horizon],
        srisk_history[lag:T-horizon],
        market_returns[lag:T-horizon]
    )

    # OLS prediction (can be replaced with quantile regression)
    beta_covar = (X'*X + 1e-8*I) \ (X'*Y_covar)
    beta_mes = (X'*X + 1e-8*I) \ (X'*Y_mes)

    # Forecast
    x_last = [1.0, covar_history[T], mes_history[T], srisk_history[T], market_returns[T]]
    pred_covar = dot(beta_covar, x_last)
    pred_mes = dot(beta_mes, x_last)
    pred_srisk = srisk_history[T] * 1.0  # random walk as baseline

    return [pred_covar, pred_mes, pred_srisk]
end

# ============================================================
# CROSS-BORDER CONTAGION
# ============================================================

"""
Cross-country contagion model using market return correlations.
Measures how distress in one country propagates to others.
"""
struct CrossBorderContagion
    countries::Vector{String}
    normal_corr::Matrix{Float64}    # correlation in normal times
    stress_corr::Matrix{Float64}    # correlation during stress
    excess_corr::Matrix{Float64}    # stress - normal
    contagion_indices::Vector{Float64}
end

function compute_cross_border_contagion(
    returns::Matrix{Float64},
    country_names::Vector{String};
    stress_quantile::Float64=0.10,
    normal_quantile_range::Tuple{Float64,Float64}=(0.25, 0.75)
)
    T, N = size(returns)
    @assert length(country_names) == N

    # Split into stress and normal periods based on average return
    avg_return = vec(mean(returns, dims=2))
    stress_thr = quantile(avg_return, stress_quantile)
    lo_q, hi_q = normal_quantile_range
    normal_lo = quantile(avg_return, lo_q)
    normal_hi = quantile(avg_return, hi_q)

    stress_idx = findall(avg_return .<= stress_thr)
    normal_idx = findall(normal_lo .<= avg_return .<= normal_hi)

    stress_corr = length(stress_idx) >= 5 ? cor(returns[stress_idx, :]) : Matrix(I*1.0, N, N)
    normal_corr = length(normal_idx) >= 5 ? cor(returns[normal_idx, :]) : cor(returns)
    excess_corr = stress_corr - normal_corr

    # Contagion index: average excess correlation (outward)
    contagion_indices = vec(mean(excess_corr, dims=2))

    return CrossBorderContagion(country_names, normal_corr, stress_corr,
                                 excess_corr, contagion_indices)
end

"""Print cross-border contagion summary."""
function print_cross_border_contagion(cbc::CrossBorderContagion)
    N = length(cbc.countries)
    println("\n=== Cross-Border Contagion ===")
    println(rpad("Country", 14), rpad("Normal Avg Corr", 18), rpad("Stress Avg Corr", 18), rpad("Contagion Index", 18))
    println("-"^68)
    for i in 1:N
        normal_avg = (sum(cbc.normal_corr[i, :]) - 1) / max(N-1, 1)
        stress_avg = (sum(cbc.stress_corr[i, :]) - 1) / max(N-1, 1)
        println(
            rpad(cbc.countries[i], 14),
            rpad(round(normal_avg, digits=4), 18),
            rpad(round(stress_avg, digits=4), 18),
            round(cbc.contagion_indices[i], digits=4)
        )
    end
end

# ============================================================
# CRYPTO-SPECIFIC SYSTEMIC RISK
# ============================================================

"""
Crypto systemic risk: measure interconnectedness from
correlated liquidations and shared oracle price feeds.
"""
struct CryptoSystemicRisk
    protocols::Vector{String}
    liquidation_correlation::Matrix{Float64}
    shared_oracle_risk::Matrix{Float64}
    composite_risk::Vector{Float64}
    defi_srisk::Vector{Float64}
end

function crypto_systemic_risk(
    protocol_names::Vector{String},
    liquidation_volumes::Matrix{Float64},  # T x N liquidation events per protocol
    oracle_providers::Vector{Vector{String}};  # shared oracle providers
    capital_at_risk::Vector{Float64}=ones(length(protocol_names))
)
    N = length(protocol_names)
    T = size(liquidation_volumes, 1)

    # Liquidation correlation
    liq_corr = T >= 5 ? cor(liquidation_volumes) : Matrix(I*1.0, N, N)

    # Shared oracle risk matrix
    shared_oracle = zeros(N, N)
    for i in 1:N
        for j in 1:N
            shared = length(intersect(oracle_providers[i], oracle_providers[j]))
            total = length(union(oracle_providers[i], oracle_providers[j]))
            shared_oracle[i, j] = total > 0 ? shared / total : 0.0
        end
    end

    # Composite risk score
    composite = vec(mean(liq_corr + shared_oracle, dims=2)) / 2

    # DeFi SRISK: capital at risk weighted by composite risk
    total_risk = sum(composite .* capital_at_risk)
    defi_srisk = composite .* capital_at_risk ./ max(total_risk, 1e-10)

    return CryptoSystemicRisk(protocol_names, liq_corr, shared_oracle,
                               composite, defi_srisk)
end

"""
Stablecoin depeg contagion model.
Models how a stablecoin depeg propagates through DeFi.
"""
function stablecoin_depeg_contagion(
    stablecoins::Vector{String},
    tvl_by_stablecoin::Vector{Float64},   # TVL per stablecoin in protocols
    depeg_magnitudes::Vector{Float64};    # 1.0 = fully depegged
    collateral_overlap::Matrix{Float64}=Matrix(I*1.0, length(stablecoins), length(stablecoins))
)
    N = length(stablecoins)

    # Contagion: protocols using depegged stablecoin as collateral suffer IL
    tvl_at_risk = zeros(N)
    for i in 1:N
        # Direct loss from own depeg
        tvl_at_risk[i] = tvl_by_stablecoin[i] * depeg_magnitudes[i]
        # Indirect: other stablecoins using i as collateral
        for j in 1:N
            j == i && continue
            tvl_at_risk[j] += tvl_by_stablecoin[i] * depeg_magnitudes[i] *
                               collateral_overlap[i, j] * 0.5
        end
    end

    total_tvl = sum(tvl_by_stablecoin)
    systemic_loss = sum(tvl_at_risk)
    systemic_loss_pct = systemic_loss / max(total_tvl, 1e-10) * 100

    return (stablecoins=stablecoins, tvl_at_risk=tvl_at_risk,
            systemic_loss=systemic_loss, systemic_loss_pct=systemic_loss_pct,
            fraction_affected=sum(tvl_at_risk .> 0.01 .* tvl_by_stablecoin) / N)
end

# ============================================================
# REGULATORY CAPITAL AND BUFFER ANALYSIS
# ============================================================

"""
Compute regulatory capital requirements under stress.
Basel III-inspired: minimum capital + buffers.
"""
function regulatory_capital_analysis(
    equity::Vector{Float64},
    risk_weighted_assets::Vector{Float64},
    srisk::Vector{Float64};
    min_tier1::Float64=0.06,         # 6% min Tier 1
    conservation_buffer::Float64=0.025,
    countercyclical_buffer::Float64=0.025,
    g_sib_surcharge::Float64=0.02
)
    N = length(equity)
    total_buffer = min_tier1 + conservation_buffer + countercyclical_buffer

    tier1_ratio = equity ./ max.(risk_weighted_assets, 1e-10)
    required_capital = risk_weighted_assets .* total_buffer
    capital_surplus = equity - required_capital

    # G-SIB surcharge for systemically important firms
    g_sib_flag = srisk ./ max(sum(srisk), 1e-10) .> 0.05  # >5% of total SRISK
    g_sib_requirement = g_sib_flag .* (risk_weighted_assets .* g_sib_surcharge)

    total_requirement = required_capital + g_sib_requirement
    shortfall = max.(total_requirement - equity, 0.0)

    return (tier1_ratio=tier1_ratio, capital_surplus=capital_surplus,
            shortfall=shortfall, g_sib_flag=g_sib_flag,
            total_system_shortfall=sum(shortfall),
            leverage_ratio=equity ./ max.(risk_weighted_assets .* 12.5, 1e-10))
end

end # module SystemicRiskMeasures
