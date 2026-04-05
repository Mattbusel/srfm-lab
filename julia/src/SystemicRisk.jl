module SystemicRisk

# ============================================================
# SystemicRisk.jl — Contagion models, CoVaR, MES, SRISK,
#                    network centrality for financial stability
# ============================================================

using Statistics, LinearAlgebra

export covar, delta_covar, mes, srisk_estimate
export network_contagion_model, debtrank, eisenberg_noe
export systemic_expected_shortfall, marginal_expected_shortfall
export betweenness_centrality_approx, eigenvector_centrality
export pagerank, katz_centrality, hub_authority
export default_cascade, fireale_model
export contingent_claims_systemic, bank_distance_to_default
export macroprudential_indicator, buffer_capital_requirement
export connectivity_matrix, exposure_network
export tail_dependence_matrix, extreme_downside_correlation
export stress_test_contagion, reverse_stress_test
export systemic_risk_decomposition, shapley_systemic

# ──────────────────────────────────────────────────────────────
# CoVaR: Adrian & Brunnermeier (2011)
# ──────────────────────────────────────────────────────────────

"""
    covar(returns_i, returns_system, q_system, q_i)

Compute CoVaR of the financial system conditional on institution i
being at its q_i quantile. Uses quantile regression via simplex method.
"""
function covar(returns_i::Vector{Float64}, returns_system::Vector{Float64},
                q_system::Float64=0.01, q_i::Float64=0.01)
    n = length(returns_i)
    # Quantile regression: minimize sum rho_q(y - X*beta)
    # where rho_q(u) = u*(q - 1_{u<0})
    # Simple approach: sort and compute empirical quantile conditional on i being in tail

    # Find observations where i is at its quantile
    var_i = quantile(sort(returns_i), q_i)
    tail_idx = findall(x -> x <= var_i, returns_i)
    if length(tail_idx) < 5
        # Fallback: use bottom decile
        n_tail = max(5, round(Int, n * 0.10))
        sorted_idx = sortperm(returns_i)
        tail_idx = sorted_idx[1:n_tail]
    end

    # CoVaR = quantile of system returns in the stress state
    system_in_stress = returns_system[tail_idx]
    covar_val = quantile(sort(system_in_stress), q_system)
    return covar_val
end

"""
    delta_covar(returns_i, returns_system, q_system, q_i, q_median)

ΔCoVaR = CoVaR(system|i at q_i) - CoVaR(system|i at median).
Measures marginal contribution of institution i to systemic risk.
"""
function delta_covar(returns_i::Vector{Float64}, returns_system::Vector{Float64},
                      q_system::Float64=0.01, q_i::Float64=0.01, q_median::Float64=0.50)
    cvar_stress = covar(returns_i, returns_system, q_system, q_i)
    cvar_normal = covar(returns_i, returns_system, q_system, q_median)
    return cvar_stress - cvar_normal
end

# ──────────────────────────────────────────────────────────────
# MES: Marginal Expected Shortfall (Acharya et al.)
# ──────────────────────────────────────────────────────────────

"""
    mes(returns_i, returns_market, threshold_quantile)

Marginal Expected Shortfall: expected loss of institution i
conditional on the market being in its worst q fraction of days.
"""
function mes(returns_i::Vector{Float64}, returns_market::Vector{Float64},
              threshold_quantile::Float64=0.05)
    n = length(returns_market)
    n_tail = max(1, round(Int, n * threshold_quantile))
    sorted_idx = sortperm(returns_market)
    tail_idx = sorted_idx[1:n_tail]
    return mean(returns_i[tail_idx])
end

"""
    systemic_expected_shortfall(returns_matrix, market_returns, q)

SES vector: MES for each institution given market tail event.
"""
function systemic_expected_shortfall(returns_matrix::Matrix{Float64},
                                       market_returns::Vector{Float64},
                                       q::Float64=0.05)
    n_inst = size(returns_matrix, 2)
    return [mes(returns_matrix[:, i], market_returns, q) for i in 1:n_inst]
end

"""
    marginal_expected_shortfall(returns_i, returns_system, q)

MES of institution i with respect to the system return.
"""
function marginal_expected_shortfall(returns_i::Vector{Float64},
                                      returns_system::Vector{Float64},
                                      q::Float64=0.05)
    return mes(returns_i, returns_system, q)
end

# ──────────────────────────────────────────────────────────────
# SRISK: Brownlees & Engle
# ──────────────────────────────────────────────────────────────

"""
    srisk_estimate(equity, debt, lrmes, k)

SRISK = max(0, k*(Debt + Equity) - Equity*(1 - LRMES))
where k = prudential capital ratio (e.g. 0.08),
LRMES = Long-Run Marginal Expected Shortfall (over a crisis period).
"""
function srisk_estimate(equity::Float64, debt::Float64,
                          lrmes::Float64, k::Float64=0.08)
    return max(0.0, k * (debt + equity) - equity * (1.0 - lrmes))
end

"""
    lrmes_from_mes(mes_daily, crisis_horizon_days, market_return_crisis)

Approximate LRMES from short-run MES using:
LRMES ≈ 1 - exp(18 * MES)  [Brownlees-Engle approximation]
"""
function lrmes_from_mes(mes_daily::Float64, crisis_horizon_days::Int=180,
                          market_return_crisis::Float64=-0.40)
    # Proportional scaling
    return 1.0 - exp(crisis_horizon_days / 22.0 * mes_daily)
end

"""
    srisk_vector(equities, debts, mes_vec, k)

Compute SRISK for a vector of financial institutions.
"""
function srisk_vector(equities::Vector{Float64}, debts::Vector{Float64},
                       mes_vec::Vector{Float64}, k::Float64=0.08)
    lrmes_vec = lrmes_from_mes.(mes_vec)
    return srisk_estimate.(equities, debts, lrmes_vec, k)
end

# ──────────────────────────────────────────────────────────────
# Network contagion models
# ──────────────────────────────────────────────────────────────

"""
    eisenberg_noe(L, e) -> (p_star, defaulted)

Eisenberg-Noe (2001) clearing payment vector.
L[i,j] = nominal liability of i to j.
e[i] = external asset value of institution i.
Returns clearing payment vector p* and default indicators.
"""
function eisenberg_noe(L::Matrix{Float64}, e::Vector{Float64};
                        tol::Float64=1e-10, maxiter::Int=500)
    n = length(e)
    p_bar = vec(sum(L, dims=2))  # total liabilities of each institution
    # Relative liability matrix
    Pi = zeros(n, n)
    for i in 1:n
        if p_bar[i] > 1e-12
            Pi[i, :] = L[i, :] ./ p_bar[i]
        end
    end

    # Iterate: p^{k+1} = min(p_bar, Pi'*p^k + e)
    p = copy(p_bar)
    for _ in 1:maxiter
        p_new = min.(p_bar, Pi' * p .+ e)
        if norm(p_new - p, Inf) < tol
            p = p_new
            break
        end
        p = p_new
    end
    defaulted = p .< p_bar .- tol
    return p, defaulted
end

"""
    default_cascade(adj_matrix, shocked_nodes, threshold)

Simple threshold cascade model. Each node defaults if fraction
of defaulted neighbors exceeds threshold.
Returns final set of defaulted nodes and cascade rounds.
"""
function default_cascade(adj_matrix::Matrix{Float64},
                           initially_defaulted::Vector{Int},
                           threshold::Float64=0.3)
    n = size(adj_matrix, 1)
    defaulted = falses(n)
    for i in initially_defaulted
        defaulted[i] = true
    end
    cascade_sizes = [sum(defaulted)]
    prev_count = -1
    while sum(defaulted) != prev_count
        prev_count = sum(defaulted)
        for i in 1:n
            if defaulted[i]; continue; end
            neighbors = adj_matrix[:, i]
            total_exposure = sum(neighbors)
            if total_exposure < 1e-12; continue; end
            defaulted_exposure = dot(neighbors, Float64.(defaulted))
            if defaulted_exposure / total_exposure >= threshold
                defaulted[i] = true
            end
        end
        push!(cascade_sizes, sum(defaulted))
    end
    return findall(defaulted), cascade_sizes
end

"""
    debtrank(adj_matrix, shocked_nodes, shock_fraction)

DebtRank algorithm (Battiston et al. 2012).
Measures fraction of total economic value lost due to distress.
adj_matrix[i,j] = fraction of j's equity impacted by i's distress.
"""
function debtrank(adj_matrix::Matrix{Float64},
                   shocked_nodes::Vector{Int},
                   shock_fraction::Float64=1.0)
    n = size(adj_matrix, 1)
    h = zeros(n)  # distress level
    for i in shocked_nodes
        h[i] = shock_fraction
    end
    s = ones(Bool, n)  # active status
    economic_loss = sum(h)
    prev_loss = -1.0
    round_count = 0
    while abs(economic_loss - prev_loss) > 1e-12 && round_count < n
        prev_loss = economic_loss
        h_new = copy(h)
        for j in 1:n
            if !s[j]; continue; end
            delta_h = sum(adj_matrix[i,j] * h[i] for i in 1:n if s[i] && i != j)
            h_new[j] = min(1.0, h[j] + delta_h)
            if h_new[j] >= 1.0; s[j] = false; end
        end
        h = h_new
        economic_loss = sum(h)
        round_count += 1
    end
    return economic_loss / n, h
end

"""
    fireale_model(prices, holdings_matrix, lambda, total_supply)

Fire sale contagion model. Institutions sell assets when losses
exceed threshold, depressing prices and causing further losses.
holdings_matrix[i,j] = fraction of asset j held by institution i.
lambda = price impact coefficient.
"""
function fireale_model(prices::Vector{Float64},
                        holdings_matrix::Matrix{Float64},
                        equity_ratios::Vector{Float64},
                        lambda::Float64=0.1)
    n_inst, n_assets = size(holdings_matrix)
    p = copy(prices)
    equity = copy(equity_ratios)
    total_loss = 0.0

    for round in 1:20
        # Identify distressed institutions
        distressed = findall(equity .< 0.0)
        if isempty(distressed); break; end

        # Fire sales
        delta_p = zeros(n_assets)
        for i in distressed
            sell_fraction = min(1.0, abs(equity[i]))
            for j in 1:n_assets
                delta_p[j] -= lambda * holdings_matrix[i,j] * sell_fraction * p[j]
            end
        end

        p = max.(p .+ delta_p, 0.0)
        # Update equity based on price changes
        for i in 1:n_inst
            portfolio_loss = dot(holdings_matrix[i,:], delta_p)
            equity[i] += portfolio_loss
        end
        total_loss += abs(sum(delta_p))
    end
    return p, equity, total_loss
end

# ──────────────────────────────────────────────────────────────
# Network centrality measures
# ──────────────────────────────────────────────────────────────

"""
    eigenvector_centrality(adj_matrix; tol, maxiter) -> centrality_vector

Power iteration to compute eigenvector centrality.
"""
function eigenvector_centrality(A::Matrix{Float64};
                                  tol::Float64=1e-10, maxiter::Int=500)
    n = size(A, 1)
    v = ones(n) ./ sqrt(n)
    for _ in 1:maxiter
        w = A * v
        lambda = norm(w, Inf)
        if lambda < 1e-12; break; end
        v_new = w ./ lambda
        if norm(v_new - v) < tol; return v_new; end
        v = v_new
    end
    return v
end

"""
    pagerank(adj_matrix, d, tol, maxiter) -> rank_vector

PageRank with damping factor d (typically 0.85).
"""
function pagerank(A::Matrix{Float64}, d::Float64=0.85;
                   tol::Float64=1e-10, maxiter::Int=200)
    n = size(A, 1)
    # Column-normalize A
    col_sums = max.(vec(sum(A, dims=1)), 1e-15)
    M = A ./ col_sums'
    r = ones(n) ./ n
    for _ in 1:maxiter
        r_new = (1.0 - d) / n .* ones(n) .+ d .* (M * r)
        if norm(r_new - r) < tol; return r_new; end
        r = r_new
    end
    return r
end

"""
    katz_centrality(A, alpha) -> centrality_vector

Katz centrality: c = (I - alpha*A')^{-1} * 1
"""
function katz_centrality(A::Matrix{Float64}, alpha::Float64=0.1)
    n = size(A, 1)
    M = Matrix{Float64}(I, n, n) .- alpha .* A'
    return M \ ones(n)
end

"""
    hub_authority(A, n_iter) -> (hub_scores, authority_scores)

HITS algorithm: hub and authority scores.
"""
function hub_authority(A::Matrix{Float64}, n_iter::Int=100)
    n = size(A, 1)
    h = ones(n)
    a = ones(n)
    for _ in 1:n_iter
        a_new = A' * h
        norm_a = norm(a_new)
        a = norm_a > 1e-12 ? a_new ./ norm_a : a_new
        h_new = A * a
        norm_h = norm(h_new)
        h = norm_h > 1e-12 ? h_new ./ norm_h : h_new
    end
    return h, a
end

"""
    betweenness_centrality_approx(A, n_samples) -> centrality_vector

Approximate betweenness centrality via random BFS/DFS sampling.
A[i,j] > 0 means edge exists.
"""
function betweenness_centrality_approx(A::Matrix{Float64}, n_samples::Int=50)
    n = size(A, 1)
    adj = [findall(A[i,:] .> 0) for i in 1:n]
    centrality = zeros(n)

    function bfs_paths(source::Int)
        dist = fill(-1, n)
        pred = [Int[] for _ in 1:n]
        sigma = zeros(Int, n)
        dist[source] = 0
        sigma[source] = 1
        queue = [source]
        order = Int[]
        qi = 1
        while qi <= length(queue)
            v = queue[qi]; qi += 1
            push!(order, v)
            for w in adj[v]
                if dist[w] < 0
                    dist[w] = dist[v] + 1
                    push!(queue, w)
                end
                if dist[w] == dist[v] + 1
                    sigma[w] += sigma[v]
                    push!(pred[w], v)
                end
            end
        end
        return order, pred, sigma
    end

    sources = n <= n_samples ? (1:n) : (round.(Int, range(1, n, length=n_samples)))
    for s in sources
        order, pred, sigma = bfs_paths(s)
        delta = zeros(n)
        for w in reverse(order)
            for v in pred[w]
                if sigma[w] > 0
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                end
            end
            if w != s
                centrality[w] += delta[w]
            end
        end
    end

    scale = length(sources) == n ? 1.0 / ((n-1)*(n-2)) : n / (2.0 * length(sources) * (n-1))
    return centrality .* scale
end

# ──────────────────────────────────────────────────────────────
# Tail risk and dependence
# ──────────────────────────────────────────────────────────────

"""
    tail_dependence_matrix(returns_matrix, q) -> TDM

Estimate lower tail dependence coefficient between each pair.
λ_L(i,j) ≈ P(X_i < q | X_j < q)
"""
function tail_dependence_matrix(returns_matrix::Matrix{Float64}, q::Float64=0.05)
    n_obs, n = size(returns_matrix)
    TDM = zeros(n, n)
    n_tail = max(1, round(Int, n_obs * q))
    for i in 1:n
        qi = sort(returns_matrix[:, i])[n_tail]
        mask_i = returns_matrix[:, i] .<= qi
        for j in 1:n
            qj = sort(returns_matrix[:, j])[n_tail]
            mask_j = returns_matrix[:, j] .<= qj
            joint = sum(mask_i .& mask_j)
            TDM[i,j] = joint / max(sum(mask_j), 1)
        end
    end
    return TDM
end

"""
    extreme_downside_correlation(returns_matrix, q) -> correlation matrix

Correlation matrix computed only on joint tail observations.
"""
function extreme_downside_correlation(returns_matrix::Matrix{Float64}, q::Float64=0.10)
    n_obs, n = size(returns_matrix)
    mkt = vec(mean(returns_matrix, dims=2))
    threshold = quantile(sort(mkt), q)
    tail_obs = returns_matrix[mkt .<= threshold, :]
    if size(tail_obs, 1) < 5
        return cor(returns_matrix)
    end
    return cor(tail_obs)
end

# ──────────────────────────────────────────────────────────────
# Contingent claims & distance to default
# ──────────────────────────────────────────────────────────────

"""
    bank_distance_to_default(asset_value, debt, asset_vol, T)

Merton model distance to default:
DD = (ln(V/D) + (μ - σ²/2)*T) / (σ*√T)
"""
function bank_distance_to_default(asset_value::Float64, debt::Float64,
                                    asset_vol::Float64, T::Float64=1.0,
                                    drift::Float64=0.05)
    if asset_value <= 0 || debt <= 0; return -Inf; end
    dd = (log(asset_value / debt) + (drift - 0.5*asset_vol^2)*T) / (asset_vol * sqrt(T))
    return dd
end

"""
    contingent_claims_systemic(asset_values, debts, asset_vols, correlations)

System-wide default probability using multivariate Merton model.
Returns vector of individual default probabilities and joint default prob.
"""
function contingent_claims_systemic(asset_values::Vector{Float64},
                                     debts::Vector{Float64},
                                     asset_vols::Vector{Float64},
                                     T::Float64=1.0, drift::Float64=0.05)
    n = length(asset_values)
    dds = [bank_distance_to_default(asset_values[i], debts[i], asset_vols[i], T, drift)
           for i in 1:n]
    # Standard normal CDF approximation
    norm_cdf(x) = 0.5 * (1.0 + erf(x / sqrt(2.0)))
    # erf approximation
    erf_approx(x) = sign(x) * (1.0 - 1.0/(1.0 + 0.278393*abs(x) + 0.230389*abs(x)^2 + 0.000972*abs(x)^3 + 0.078108*abs(x)^4)^4)
    norm_cdf2(x) = 0.5 * (1.0 + erf_approx(x / sqrt(2.0)))
    default_probs = norm_cdf2.(-dds)
    return default_probs, dds
end

# ──────────────────────────────────────────────────────────────
# Macroprudential indicators
# ──────────────────────────────────────────────────────────────

"""
    connectivity_matrix(bilateral_exposures) -> normalized adjacency

Normalize bilateral exposure matrix to get network connectivity.
"""
function connectivity_matrix(exposures::Matrix{Float64})
    n = size(exposures, 1)
    total_assets = vec(sum(exposures, dims=2))
    conn = zeros(n, n)
    for i in 1:n
        if total_assets[i] > 1e-12
            conn[i, :] = exposures[i, :] ./ total_assets[i]
        end
    end
    return conn
end

"""
    exposure_network(balance_sheets, interbank_fraction)

Build simplified interbank exposure network from balance sheet data.
balance_sheets: matrix where rows are banks, columns are [assets, liabilities, equity]
"""
function exposure_network(balance_sheets::Matrix{Float64}, interbank_fraction::Float64=0.3)
    n = size(balance_sheets, 1)
    assets = balance_sheets[:, 1]
    liabilities = balance_sheets[:, 2]
    # Simplified: distribute interbank claims proportionally
    total_claims = interbank_fraction * sum(assets)
    L = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                L[i,j] = interbank_fraction * assets[i] * (liabilities[j] / sum(liabilities))
            end
        end
    end
    return L
end

"""
    macroprudential_indicator(returns_matrix, q_sys, lookback)

Composite macroprudential risk indicator combining:
- Average tail correlation
- System-level VaR
- Concentration (Herfindahl index)
"""
function macroprudential_indicator(returns_matrix::Matrix{Float64},
                                    q_sys::Float64=0.01, lookback::Int=252)
    n_obs, n = size(returns_matrix)
    recent = min(lookback, n_obs)
    R = returns_matrix[end-recent+1:end, :]
    system_returns = vec(mean(R, dims=2))
    # System VaR
    sys_var = quantile(sort(system_returns), q_sys)
    # Average pairwise tail correlation
    tail_corr = mean(tail_dependence_matrix(R, 0.10))
    # Herfindahl on absolute mean returns
    mean_ret = abs.(vec(mean(R, dims=1)))
    total = sum(mean_ret)
    hhi = total > 0 ? sum((mean_ret ./ total).^2) : 1.0/n
    # Composite score
    indicator = -sys_var * (1.0 + tail_corr) * (1.0 + hhi)
    return indicator, sys_var, tail_corr, hhi
end

"""
    buffer_capital_requirement(srisk_vec, total_capital, alpha)

Countercyclical buffer: fraction of additional capital required
based on SRISK. alpha = scaling factor.
"""
function buffer_capital_requirement(srisk_vec::Vector{Float64},
                                     total_capital::Float64,
                                     alpha::Float64=0.10)
    total_srisk = sum(max.(srisk_vec, 0.0))
    buffer = alpha * total_srisk / max(total_capital, 1.0)
    return clamp(buffer, 0.0, 0.25)  # Cap at 25% additional buffer
end

# ──────────────────────────────────────────────────────────────
# Stress testing
# ──────────────────────────────────────────────────────────────

"""
    stress_test_contagion(L_matrix, e_vector, shock_institutions, shock_size)

Run Eisenberg-Noe contagion starting from an external shock.
Returns (total_loss, fraction_defaulted, rounds_to_clear).
"""
function stress_test_contagion(L::Matrix{Float64}, e::Vector{Float64},
                                 shocked::Vector{Int}, shock_size::Float64=0.5)
    e_stressed = copy(e)
    for i in shocked
        e_stressed[i] *= (1.0 - shock_size)
    end
    p_star, defaulted = eisenberg_noe(L, e_stressed)
    p_bar = vec(sum(L, dims=2))
    total_loss = sum(p_bar .- p_star)
    frac_default = mean(defaulted)
    return total_loss, frac_default, p_star, defaulted
end

"""
    reverse_stress_test(L, e, target_default_fraction) -> min_shock

Find minimum shock size that causes target fraction of defaults.
Binary search over shock sizes.
"""
function reverse_stress_test(L::Matrix{Float64}, e::Vector{Float64},
                               target_fraction::Float64=0.20,
                               institution_idx::Int=1)
    lo, hi = 0.0, 1.0
    for _ in 1:50
        mid = (lo + hi) / 2.0
        _, frac, _, _ = stress_test_contagion(L, e, [institution_idx], mid)
        if frac < target_fraction
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2.0
end

# ──────────────────────────────────────────────────────────────
# Systemic risk decomposition
# ──────────────────────────────────────────────────────────────

"""
    systemic_risk_decomposition(returns_matrix, system_returns, q)

Decompose system VaR into institution-level contributions.
Returns (contributions, fractions).
"""
function systemic_risk_decomposition(returns_matrix::Matrix{Float64},
                                      system_returns::Vector{Float64},
                                      q::Float64=0.01)
    n = size(returns_matrix, 2)
    n_tail = max(1, round(Int, length(system_returns) * q))
    tail_idx = sortperm(system_returns)[1:n_tail]
    contributions = [mean(returns_matrix[tail_idx, i]) for i in 1:n]
    total = sum(abs.(contributions))
    fractions = total > 0 ? abs.(contributions) ./ total : fill(1.0/n, n)
    return contributions, fractions
end

"""
    shapley_systemic(returns_matrix, system_returns, q, n_permutations)

Approximate Shapley value decomposition of systemic VaR.
Computationally intensive for large n; uses sampling.
"""
function shapley_systemic(returns_matrix::Matrix{Float64},
                            system_returns::Vector{Float64},
                            q::Float64=0.01, n_perm::Int=200)
    n = size(returns_matrix, 2)
    system_var = quantile(sort(system_returns), q)
    shapley = zeros(n)

    function coalition_var(members::Vector{Int})
        if isempty(members); return 0.0; end
        coal_returns = vec(mean(returns_matrix[:, members], dims=2))
        return quantile(sort(coal_returns), q)
    end

    # Simple permutation sampling
    state = UInt64(42)
    for _ in 1:n_perm
        # Generate random permutation using LCG
        perm = collect(1:n)
        for i in n:-1:2
            state = state * 6364136223846793005 + 1442695040888963407
            j = (state % i) + 1
            perm[i], perm[j] = perm[j], perm[i]
        end
        for k in 1:n
            with_k = sort(perm[1:k])
            without_k = sort(perm[1:k-1])
            marginal = coalition_var(with_k) - coalition_var(without_k)
            shapley[perm[k]] += marginal
        end
    end
    return shapley ./ n_perm
end


# ============================================================
# SECTION 2: ADVANCED SYSTEMIC RISK MEASURES
# ============================================================

function delta_covar_time_varying(returns_i::Vector{Float64},
                                    returns_system::Vector{Float64};
                                    tau::Float64=0.05,
                                    garch_omega::Float64=1e-6,
                                    garch_alpha::Float64=0.1,
                                    garch_beta::Float64=0.85)
    # Time-varying CoVaR via rolling quantile regression with GARCH volatility
    n = length(returns_i)
    h = ones(n)
    for t in 2:n
        h[t] = garch_omega + garch_alpha*returns_i[t-1]^2 + garch_beta*h[t-1]
    end
    sigma = sqrt.(h)
    std_returns = returns_i ./ (sigma .+ 1e-10)
    # Rolling VaR of standardized returns
    covar_series = zeros(n)
    window = 60
    for t in window:n
        w_i = std_returns[t-window+1:t]
        w_s = returns_system[t-window+1:t]
        var_i = quantile(w_i, tau)
        # CoVaR: quantile of system given institution at its VaR
        in_tail = findall(r -> r <= var_i, w_i)
        if length(in_tail) >= 3
            covar_series[t] = quantile(w_s[in_tail], tau)
        end
    end
    return (covar=covar_series, garch_vol=sigma)
end

function network_systemic_risk(adjacency::Matrix{Float64}, exposures::Vector{Float64})
    n = size(adjacency, 1)
    # DebtRank algorithm
    h = copy(exposures); h0 = copy(h)
    distress_before = copy(h)
    impact_total = zeros(n)
    for _ in 1:n  # propagation rounds
        h_new = copy(h)
        for i in 1:n
            for j in 1:n
                if adjacency[j,i] > 0
                    h_new[i] = min(1.0, h_new[i] + adjacency[j,i] * h[j])
                end
            end
        end
        delta = h_new .- h
        all(abs.(delta) .< 1e-8) && break
        h = h_new
    end
    total_loss = sum((h .- h0) .* exposures)
    return (final_distress=h, total_loss=total_loss, loss_amplification=total_loss/sum(exposures))
end

function systemic_loss_distribution(returns::Matrix{Float64}, weights::Vector{Float64};
                                      n_scenarios::Int=10000, seed::Int=42)
    T, N = size(returns)
    # Bootstrap scenarios
    rng_state = UInt64(seed)
    lcg() = (rng_state = rng_state*6364136223846793005+1442695040888963407; rng_state)
    scenario_losses = zeros(n_scenarios)
    for s in 1:n_scenarios
        idx = (lcg() % T) + 1
        row = returns[idx,:]
        scenario_losses[s] = -dot(weights, row)
    end
    sort!(scenario_losses)
    var95 = quantile(scenario_losses, 0.95)
    var99 = quantile(scenario_losses, 0.99)
    es95  = mean([l for l in scenario_losses if l >= var95])
    es99  = mean([l for l in scenario_losses if l >= var99])
    return (losses=scenario_losses, var95=var95, var99=var99, es95=es95, es99=es99)
end

function absorption_ratio_extended(returns::Matrix{Float64}; n_factors::Int=1)
    T, N = size(returns)
    # Covariance matrix
    mu = [mean(returns[:,i]) for i in 1:N]
    X = returns .- mu'
    C = X'*X ./ (T-1)
    total_var = sum(diag(C))
    # Power iteration for top eigenvalues
    absorbed = 0.0
    B = copy(C)
    for _ in 1:n_factors
        v = randn(N); v ./= norm(v)
        for _ in 1:500
            vn = B*v; vn ./= (norm(vn)+1e-15)
            norm(vn-v)<1e-10 && break; v=vn
        end
        lam = dot(v, B*v)
        absorbed += lam
        B .-= lam .* (v*v')
    end
    return absorbed / (total_var + 1e-10)
end

function financial_stress_index(credit_spreads::Vector{Float64},
                                  equity_vol::Vector{Float64},
                                  fx_vol::Vector{Float64},
                                  funding_spread::Vector{Float64};
                                  window::Int=252)
    n = min(length(credit_spreads), length(equity_vol),
            length(fx_vol), length(funding_spread))
    fsi = zeros(n)
    for i in window:n
        ws = window
        z_credit  = (credit_spreads[i] - mean(credit_spreads[max(1,i-ws+1):i])) /
                    (std(credit_spreads[max(1,i-ws+1):i]) + 1e-10)
        z_eqvol   = (equity_vol[i] - mean(equity_vol[max(1,i-ws+1):i])) /
                    (std(equity_vol[max(1,i-ws+1):i]) + 1e-10)
        z_fxvol   = (fx_vol[i] - mean(fx_vol[max(1,i-ws+1):i])) /
                    (std(fx_vol[max(1,i-ws+1):i]) + 1e-10)
        z_funding = (funding_spread[i] - mean(funding_spread[max(1,i-ws+1):i])) /
                    (std(funding_spread[max(1,i-ws+1):i]) + 1e-10)
        fsi[i] = (z_credit + z_eqvol + z_fxvol + z_funding) / 4.0
    end
    return fsi
end

function turbulence_index(returns::Matrix{Float64}; window::Int=252)
    T, N = size(returns)
    turbulence = zeros(T)
    for t in window+1:T
        hist = returns[t-window:t-1,:]
        mu   = [mean(hist[:,i]) for i in 1:N]
        Sigma = cov(hist)
        r = returns[t,:] .- mu
        # Pseudo-inverse for robustness
        try
            Sigma_inv = inv(Sigma + 1e-6*I(N))
            turbulence[t] = dot(r, Sigma_inv * r)
        catch
            turbulence[t] = dot(r, r)
        end
    end
    return turbulence
end

function cross_border_contagion(domestic_returns::Vector{Float64},
                                  foreign_returns::Vector{Float64};
                                  crisis_threshold::Float64=-0.02,
                                  window::Int=60)
    n = min(length(domestic_returns), length(foreign_returns))
    normal_corr = Float64[]; crisis_corr = Float64[]
    for t in window:n
        d = domestic_returns[t-window+1:t]
        f = foreign_returns[t-window+1:t]
        in_crisis = mean(f) < crisis_threshold
        c = cor(d, f)
        in_crisis ? push!(crisis_corr, c) : push!(normal_corr, c)
    end
    return (normal_correlation = isempty(normal_corr) ? 0.0 : mean(normal_corr),
            crisis_correlation = isempty(crisis_corr) ? 0.0 : mean(crisis_corr),
            contagion_excess    = isempty(crisis_corr) ? 0.0 :
                                  mean(crisis_corr) - mean(normal_corr))
end

function srisk_panel(returns_i::Matrix{Float64},
                      returns_market::Vector{Float64},
                      equity_values::Vector{Float64},
                      debt_values::Vector{Float64};
                      k::Float64=0.08, h::Int=22, simulation_n::Int=10000)
    N = size(returns_i, 2)
    srisk_vals = zeros(N)
    T = length(returns_market)
    for i in 1:N
        ret_i = returns_i[:,i]
        beta_i = cov(ret_i, returns_market) / (var(returns_market) + 1e-10)
        # LRMES approximation: 1 - exp(-18 * beta * sigma_m * sqrt(h))
        sigma_m = std(returns_market)
        lrmes = 1 - exp(-18 * beta_i * sigma_m * sqrt(h))
        lrmes = clamp(lrmes, 0.0, 1.0)
        srisk_vals[i] = max(k*(equity_values[i]+debt_values[i]) -
                            (1-lrmes)*equity_values[i], 0.0)
    end
    return (srisk=srisk_vals, total_srisk=sum(srisk_vals),
            systemic_share=srisk_vals./max(sum(srisk_vals),1e-10))
end

function tail_dependence_copula(u::Vector{Float64}, v::Vector{Float64};
                                  threshold::Float64=0.95)
    n = length(u)
    # Lower tail
    lo_idx = findall(i -> u[i] < 1-threshold && v[i] < 1-threshold, 1:n)
    lo_expected = n * (1-threshold)^2
    lambda_lo = length(lo_idx) / (lo_expected + 1e-10)
    # Upper tail
    hi_idx = findall(i -> u[i] > threshold && v[i] > threshold, 1:n)
    hi_expected = n * (1-threshold)^2
    lambda_hi = length(hi_idx) / (hi_expected + 1e-10)
    return (lower_tail=lambda_lo, upper_tail=lambda_hi)
end

function bank_capital_buffer(tier1_capital::Float64, rwas::Float64,
                               systemic_surcharge::Float64=0.0)
    # Basel III: minimum CET1 = 4.5%, conservation = 2.5%, countercyclical = 0-2.5%
    min_cet1 = 0.045; conservation = 0.025; ccb = 0.025
    total_requirement = min_cet1 + conservation + ccb + systemic_surcharge
    actual_ratio = tier1_capital / (rwas + 1e-10)
    buffer = actual_ratio - total_requirement
    meets_requirement = buffer >= 0
    return (capital_ratio=actual_ratio, requirement=total_requirement,
            buffer=buffer, meets_minimum=meets_requirement,
            stressed_buffer=buffer - 0.02)  # 2% stress buffer
end

function systemic_risk_ranking(institutions::Vector{String},
                                 covar_values::Vector{Float64},
                                 mes_values::Vector{Float64},
                                 srisk_values::Vector{Float64})
    n = length(institutions)
    # Z-score each measure
    z(v) = (v .- mean(v)) ./ (std(v) .+ 1e-10)
    composite = (z(covar_values) .+ z(mes_values) .+ z(srisk_values)) ./ 3
    order = sortperm(composite; rev=true)
    return (institutions=institutions[order], composite_score=composite[order],
            covar_rank=sortperm(sortperm(covar_values; rev=true)),
            mes_rank=sortperm(sortperm(mes_values; rev=true)),
            srisk_rank=sortperm(sortperm(srisk_values; rev=true)))
end

function interconnectedness_index(exposure_matrix::Matrix{Float64})
    N = size(exposure_matrix, 1)
    # Normalize by total assets
    row_sums = sum(exposure_matrix, dims=2)[:,1]
    col_sums = sum(exposure_matrix, dims=1)[1,:]
    # In-degree and out-degree centrality
    in_degree  = col_sums ./ (sum(col_sums) + 1e-10)
    out_degree = row_sums ./ (sum(row_sums) + 1e-10)
    # Betweenness approximation: product of in- and out-degree
    betweenness = in_degree .* out_degree
    # Overall index: top 10% concentration
    threshold = quantile(betweenness, 0.9)
    top_nodes = findall(b -> b >= threshold, betweenness)
    return (in_degree=in_degree, out_degree=out_degree,
            betweenness=betweenness, top_systemic_nodes=top_nodes,
            concentration_index=sum(betweenness[top_nodes]))
end

function liquidity_mismatch_index(asset_liquidity::Vector{Float64},
                                    liability_maturity::Vector{Float64},
                                    values::Vector{Float64})
    # Positive LMI = funding risk (short liabilities, illiquid assets)
    weighted_asset_liq = sum(asset_liquidity .* values) / (sum(values) + 1e-10)
    weighted_liab_mat  = sum(liability_maturity .* values) / (sum(values) + 1e-10)
    lmi = weighted_liab_mat - weighted_asset_liq
    return (lmi=lmi, asset_liquidity_score=weighted_asset_liq,
            liability_maturity_score=weighted_liab_mat,
            is_at_risk=(lmi > 0))
end

function regulatory_stress_test(portfolio_weights::Vector{Float64},
                                   asset_shocks::Vector{Float64},
                                   correlations::Matrix{Float64},
                                   scenario_name::String="Adverse")
    n = length(portfolio_weights)
    # Direct shock
    direct_loss = dot(portfolio_weights, asset_shocks)
    # Second round: correlated losses
    shock_cov = correlations .* (asset_shocks * asset_shocks')
    portfolio_vol = sqrt(max(dot(portfolio_weights, shock_cov*portfolio_weights), 0.0))
    tail_loss = direct_loss - 1.645 * portfolio_vol  # 5% tail
    println("Stress Test: ", scenario_name)
    println("  Direct loss:    ", round(direct_loss*100, digits=2), "%")
    println("  Portfolio vol:  ", round(portfolio_vol*100, digits=2), "%")
    println("  Tail loss (5%): ", round(tail_loss*100, digits=2), "%")
    return (scenario=scenario_name, direct_loss=direct_loss,
            portfolio_vol=portfolio_vol, tail_loss=tail_loss)
end

# ============================================================
# EXTENDED DEMO
# ============================================================

function demo_systemic_risk_extended()
    println("=== Systemic Risk Extended Demo ===")

    # FSI
    n=500; cs=randn(n).*0.02.+0.005; ev=abs.(randn(n)).*0.15.+0.15
    fxv=abs.(randn(n)).*0.05.+0.05; fs=randn(n).*0.01.+0.002
    fsi=financial_stress_index(cs,ev,fxv,fs)
    println("FSI last 5: ", round.(fsi[end-4:end],digits=2))

    # Turbulence
    rets=randn(300,5).*0.01
    turb=turbulence_index(rets; window=60)
    println("Max turbulence: ", round(maximum(turb),sigdigits=3))

    # Absorption ratio
    ar=absorption_ratio_extended(rets; n_factors=1)
    println("Absorption ratio (1 factor): ", round(ar,digits=3))

    # Network systemic risk
    adj=rand(5,5).*0.1; adj-=Diagonal(diag(adj))
    exp_=rand(5).*1e9
    net=network_systemic_risk(adj,exp_./maximum(exp_))
    println("Network loss amplification: ", round(net.loss_amplification,sigdigits=3))

    # SRISK panel
    ret_mat=randn(252,5).*0.01; mkt_r=randn(252).*0.01
    eq_v=rand(5).*1e10; d_v=rand(5).*2e10
    sr=srisk_panel(ret_mat,mkt_r,eq_v,d_v)
    println("Total SRISK: \$", round(sum(sr.srisk)/1e9,digits=2),"B")

    # Capital buffer
    cap=bank_capital_buffer(50e9, 400e9; systemic_surcharge=0.02)
    println("Capital ratio: ", round(cap.capital_ratio*100,digits=2),
            "% Buffer: ", round(cap.buffer*100,digits=2),"%")
end

end # module SystemicRisk
