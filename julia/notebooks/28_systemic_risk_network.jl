# ============================================================
# Notebook 28: Systemic Risk Measurement & Network Contagion
# ============================================================
# Topics:
#   1. Building a financial network from balance sheets
#   2. CoVaR and ΔCoVaR calculation
#   3. Marginal Expected Shortfall (MES) and SRISK
#   4. Eisenberg-Noe contagion model
#   5. DebtRank algorithm
#   6. Network centrality for systemic importance
#   7. Stress testing the financial network
#   8. Macroprudential buffer requirements
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 28: Systemic Risk & Network Contagion")
println("="^60)

# ── Section 1: Simulate Financial Network ─────────────────

println("\n--- Section 1: Financial Network Setup ---")

n_banks = 10
# Bank balance sheets: [assets, liabilities, equity, market_cap]
# All in billions USD
bank_names = ["BankA", "BankB", "BankC", "BankD", "BankE",
               "BankF", "BankG", "BankH", "BankI", "BankJ"]
assets     = [800.0, 650.0, 500.0, 400.0, 300.0, 200.0, 150.0, 100.0, 75.0, 50.0]
debts      = [740.0, 600.0, 460.0, 368.0, 276.0, 184.0, 138.0,  92.0, 69.0, 46.0]
equities   = assets .- debts
market_cap = equities .* (1.0 .+ randn_like(10, 1.0))  # equity + premium

function randn_like(n, scale)
    state = UInt64(99)
    result = zeros(n)
    for i in 1:n
        state = state * 6364136223846793005 + 1442695040888963407
        u1 = max((state >> 11) / Float64(2^53), 1e-15)
        state = state * 6364136223846793005 + 1442695040888963407
        u2 = (state >> 11) / Float64(2^53)
        result[i] = sqrt(-2.0*log(u1)) * cos(2π*u2) * scale
    end
    return result
end

market_cap = max.(equities .+ randn_like(n_banks, 10.0), equities .* 0.5)
leverage = assets ./ equities

println("Financial network participants:")
println("  Bank  | Assets(\$B) | Equity(\$B) | Leverage | Market Cap")
println("  " * "-"^57)
for i in 1:n_banks
    println("  $(lpad(bank_names[i],5)) | $(lpad(round(assets[i],digits=0),9))   | " *
            "$(lpad(round(equities[i],digits=0),9))   | $(lpad(round(leverage[i],digits=1),8)) | \$$(round(market_cap[i],digits=0))B")
end

# Build interbank exposure matrix
# L[i,j] = nominal liability of bank i to bank j (in billions)
state_rng = UInt64(42)
function rand_float()
    global state_rng
    state_rng = state_rng * 6364136223846793005 + 1442695040888963407
    return (state_rng >> 11) / Float64(2^53)
end

L = zeros(n_banks, n_banks)
for i in 1:n_banks
    for j in 1:n_banks
        if i != j && rand_float() < 0.4  # 40% connectivity
            L[i,j] = rand_float() * equities[i] * 0.2
        end
    end
end
# Scale to make liabilities reasonable
for i in 1:n_banks
    total_ib = sum(L[i,:])
    if total_ib > 0.3 * debts[i]
        L[i,:] *= 0.3 * debts[i] / total_ib
    end
end

println("\nInterbank exposure matrix (top 5×5, \$B):")
for i in 1:5
    row = join([lpad(round(L[i,j], digits=1), 6) for j in 1:5], "  ")
    println("  $(bank_names[i]): $row  ...")
end

# ── Section 2: CoVaR Analysis ─────────────────────────────

println("\n--- Section 2: CoVaR and ΔCoVaR ---")

# Simulate correlated bank returns
n_obs = 1000
# Factor model: common factor + idiosyncratic
n_factors = 3
factor_betas = [randn_like(n_banks, 0.6) .+ 0.3,
                randn_like(n_banks, 0.3),
                randn_like(n_banks, 0.2)]

factor_returns = zeros(n_obs, n_factors)
state_rng = UInt64(11)
for t in 1:n_obs
    for k in 1:n_factors
        state_rng = state_rng * 6364136223846793005 + 1442695040888963407
        u1 = max((state_rng >> 11) / Float64(2^53), 1e-15)
        state_rng = state_rng * 6364136223846793005 + 1442695040888963407
        u2 = (state_rng >> 11) / Float64(2^53)
        factor_returns[t, k] = sqrt(-2.0*log(u1)) * cos(2π*u2) * 0.01
    end
end

bank_returns = zeros(n_obs, n_banks)
for j in 1:n_banks
    idio_vol = 0.005 + 0.003 * (n_banks - j) / n_banks
    for t in 1:n_obs
        systematic = sum(factor_betas[k][j] * factor_returns[t, k] for k in 1:n_factors)
        state_rng = state_rng * 6364136223846793005 + 1442695040888963407
        u1 = max((state_rng >> 11) / Float64(2^53), 1e-15)
        state_rng = state_rng * 6364136223846793005 + 1442695040888963407
        u2 = (state_rng >> 11) / Float64(2^53)
        idio = sqrt(-2.0*log(u1)) * cos(2π*u2) * idio_vol
        bank_returns[t, j] = systematic + idio
    end
end

# Market/system returns
system_returns = vec(mean(bank_returns, dims=2))

function compute_covar(returns_i, returns_system, q_i=0.01, q_sys=0.01)
    n = length(returns_i)
    var_i = quantile(sort(returns_i), q_i)
    tail_mask = returns_i .<= var_i
    if sum(tail_mask) < 3; return quantile(sort(returns_system), q_sys); end
    system_conditional = returns_system[tail_mask]
    return quantile(sort(system_conditional), q_sys)
end

function compute_mes(returns_i, returns_system, q=0.05)
    n = length(returns_system)
    n_tail = max(1, round(Int, n * q))
    tail_idx = sortperm(returns_system)[1:n_tail]
    return mean(returns_i[tail_idx])
end

println("CoVaR and MES by bank (system = equal-weight portfolio):")
println("  Bank  | VaR(1%) | CoVaR   | ΔCoVaR  | MES(5%) | Contribution")
println("  " * "-"^62)
dcovar_vec = zeros(n_banks)
mes_vec = zeros(n_banks)
for j in 1:n_banks
    var_1pct = quantile(sort(bank_returns[:,j]), 0.01)
    cvar_stress = compute_covar(bank_returns[:,j], system_returns, 0.01, 0.01)
    cvar_normal = compute_covar(bank_returns[:,j], system_returns, 0.50, 0.01)
    delta_cvar = cvar_stress - cvar_normal
    mes = compute_mes(bank_returns[:,j], system_returns, 0.05)
    dcovar_vec[j] = delta_cvar
    mes_vec[j] = mes
    contrib = abs(delta_cvar) / (sum(abs.(dcovar_vec[1:j])) + 1e-12) * 100
    println("  $(lpad(bank_names[j],5)) | $(lpad(round(var_1pct*100,digits=2),7))% | $(lpad(round(cvar_stress*100,digits=2),7))% | " *
            "$(lpad(round(delta_cvar*100,digits=2),7))% | $(lpad(round(mes*100,digits=2),7))% | $(round(contrib,digits=1))%")
end

# ── Section 3: SRISK ─────────────────────────────────────

println("\n--- Section 3: SRISK Computation ---")

k_prudential = 0.08  # 8% capital ratio requirement
crisis_horizon = 180
market_crash = -0.40  # -40% crash assumption

function lrmes_approx(mes_daily)
    return 1.0 - exp(crisis_horizon / 22.0 * mes_daily)
end

println("SRISK by institution:")
println("  Bank  | LRMES   | SRISK(\$B) | % of Total | Capital Shortage")
println("  " * "-"^58)
srisk_vec = zeros(n_banks)
for j in 1:n_banks
    lrmes = lrmes_approx(mes_vec[j])
    srisk = max(0.0, k_prudential * (debts[j] + equities[j]) - equities[j] * (1.0 - lrmes))
    srisk_vec[j] = srisk
end
total_srisk = sum(srisk_vec)
for j in 1:n_banks
    lrmes = lrmes_approx(mes_vec[j])
    pct = srisk_vec[j] / max(total_srisk, 1e-12) * 100
    shortfall = srisk_vec[j] > 0 ? "YES" : "no"
    println("  $(lpad(bank_names[j],5)) | $(lpad(round(lrmes_approx(mes_vec[j])*100,digits=1),7))% | " *
            "$(lpad(round(srisk_vec[j],digits=1),9))  | $(lpad(round(pct,digits=1),10))%  | $shortfall")
end
println("  Total SRISK: \$$(round(total_srisk, digits=1))B")

# ── Section 4: Eisenberg-Noe Contagion ───────────────────

println("\n--- Section 4: Eisenberg-Noe Clearing ---")

# External assets = equity (simplified)
e_external = copy(equities)

function eisenberg_noe(L_mat, e_vec; tol=1e-10, maxiter=500)
    n = length(e_vec)
    p_bar = vec(sum(L_mat, dims=2))
    Pi = zeros(n, n)
    for i in 1:n
        if p_bar[i] > 1e-12
            Pi[i, :] = L_mat[i, :] ./ p_bar[i]
        end
    end
    p = copy(p_bar)
    for _ in 1:maxiter
        p_new = min.(p_bar, Pi' * p .+ e_vec)
        if maximum(abs.(p_new .- p)) < tol
            p = p_new; break
        end
        p = p_new
    end
    defaulted = p .< p_bar .- tol
    return p, defaulted
end

# No shock scenario
p_star, defaulted = eisenberg_noe(L, e_external)
println("No-shock clearing: $(sum(defaulted)) defaults")

# Shock BankA (largest)
shock_sizes = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
println("\nContagion from shocking BankA (asset loss):")
println("  Shock | Defaults | Total Loss (\$B) | System Losses (\$B)")
println("  " * "-"^52)
for s in shock_sizes
    e_shocked = copy(e_external)
    e_shocked[1] *= (1.0 - s)
    p_s, def_s = eisenberg_noe(L, e_shocked)
    p_bar_local = vec(sum(L, dims=2))
    loss = sum(p_bar_local .- p_s)
    n_defaults = sum(def_s)
    sys_equity_loss = sum(equities) - sum(max.(e_shocked .+ Pi' * p_s .- p_bar_local, 0.0) for _ in 1:1)
    println("  $(lpad(round(s*100,digits=0),5))% | $(lpad(n_defaults,8)) | $(lpad(round(loss,digits=1),15))   | $(round(sum(equities) - sum(e_shocked .+ Pi' * p_s for Pi in [zeros(n_banks,n_banks)]),digits=1))")
    println("  $(lpad(round(s*100,digits=0),5))% | $(lpad(n_defaults,8)) | $(lpad(round(loss,digits=1),15))   |")
end

# Clean output
for s in shock_sizes
    e_shocked = copy(e_external)
    e_shocked[1] *= (1.0 - s)
    p_s, def_s = eisenberg_noe(L, e_shocked)
    p_bar_local = vec(sum(L, dims=2))
    loss = sum(p_bar_local .- p_s)
    println("  Shock $(round(s*100,digits=0))%: $(sum(def_s)) defaults, total loss = \$$(round(loss,digits=1))B")
end

# ── Section 5: DebtRank ──────────────────────────────────

println("\n--- Section 5: DebtRank ---")

# Build impact matrix: adj[i,j] = fraction of bank j's equity impacted by bank i
adj_debtrank = zeros(n_banks, n_banks)
for i in 1:n_banks
    for j in 1:n_banks
        if i != j && equities[j] > 0
            adj_debtrank[i,j] = L[i,j] / equities[j]
        end
    end
end

function debtrank_algo(adj, shocked, shock_frac=1.0)
    n = size(adj, 1)
    h = zeros(n)
    for i in shocked
        h[i] = shock_frac
    end
    s = trues(n)  # active status
    round_count = 0
    while true
        h_new = copy(h)
        changed = false
        for j in 1:n
            if !s[j]; continue; end
            delta = sum(adj[i,j] * h[i] for i in 1:n if s[i] && i != j; init=0.0)
            h_new[j] = min(1.0, h[j] + delta)
            if h_new[j] >= 1.0; s[j] = false; end
            if abs(h_new[j] - h[j]) > 1e-10; changed = true; end
        end
        h = h_new
        round_count += 1
        if !changed || round_count > n; break; end
    end
    return sum(h) / n, h
end

println("DebtRank by initial shocked bank:")
println("  Shocked | Economic Loss | Rounds | Defaulted")
println("  " * "-"^45)
for i in 1:n_banks
    loss, h = debtrank_algo(adj_debtrank, [i], 1.0)
    n_full = sum(h .>= 1.0)
    println("  $(lpad(bank_names[i],7)) | $(lpad(round(loss*100,digits=1),13))%  | N/A    | $(n_full) fully distressed")
end

# ── Section 6: Network Centrality ────────────────────────

println("\n--- Section 6: Network Centrality Measures ---")

# Build adjacency matrix for centrality (binary or weighted)
A_net = (L .+ L') ./ 2.0  # symmetrize

function power_iteration_centrality(A, maxiter=200)
    n = size(A, 1)
    v = ones(n) ./ sqrt(n)
    lambda = 0.0
    for _ in 1:maxiter
        w = A * v
        lambda = norm(w, Inf)
        v = lambda > 1e-12 ? w ./ lambda : w
    end
    return v ./ sum(v)
end

centrality = power_iteration_centrality(A_net)
# PageRank-like centrality
function pagerank_net(A, d=0.85, maxiter=200)
    n = size(A, 1)
    col_sums = max.(vec(sum(A, dims=1)), 1e-12)
    M = A ./ col_sums'
    r = ones(n) ./ n
    for _ in 1:maxiter
        r_new = (1-d)/n .* ones(n) .+ d .* (M * r)
        if norm(r_new - r) < 1e-10; return r_new; end
        r = r_new
    end
    return r
end
pr = pagerank_net(A_net)

println("Network centrality and SRISK:")
println("  Bank  | Eigenvec | PageRank | SRISK(\$B) | SIFIness Score")
println("  " * "-"^58)
sifi_scores = centrality .* 0.3 .+ pr ./ maximum(pr) .* 0.3 .+ srisk_vec ./ max(maximum(srisk_vec), 1e-12) .* 0.4
for j in 1:n_banks
    println("  $(lpad(bank_names[j],5)) | $(lpad(round(centrality[j],digits=4),8)) | " *
            "$(lpad(round(pr[j],digits=4),8)) | $(lpad(round(srisk_vec[j],digits=1),9))  | " *
            "$(lpad(round(sifi_scores[j],digits=3),14))")
end

# Top systemic risk contributors
println("\nTop 3 SIFIs by composite score:")
sifi_rank = sortperm(sifi_scores, rev=true)
for k in 1:3
    j = sifi_rank[k]
    println("  #$k: $(bank_names[j]) (SIFI score: $(round(sifi_scores[j],digits=3)))")
end

# ── Section 7: Stress Testing ────────────────────────────

println("\n--- Section 7: Macro Stress Testing ---")

scenarios = [
    ("Base",          0.0,  0.0,   0.0),
    ("Mild recession",-0.10, 0.05,  0.5),
    ("Severe shock",  -0.30, 0.10,  1.0),
    ("Systemic crisis",-0.50, 0.20, 2.0),
]

println("Stress test results:")
for (name, market_shock, spread_shock, funding_stress) in scenarios
    # Apply shock to external assets
    e_stressed = equities .* (1.0 .+ market_shock) .-
                  assets .* spread_shock .* 0.05  # credit losses
    e_stressed = max.(e_stressed, 0.0)
    p_s, def_s = eisenberg_noe(L, e_stressed)
    p_bar_local = vec(sum(L, dims=2))
    contagion_loss = sum(p_bar_local .- p_s)
    total_equity_loss = sum(equities) - sum(max.(e_stressed, 0.0))
    n_def = sum(def_s)
    println("  $name: $(n_def) defaults, equity loss = \$$(round(total_equity_loss,digits=1))B, " *
            "contagion = \$$(round(contagion_loss,digits=1))B")
end

# ── Section 8: Buffer Requirements ───────────────────────

println("\n--- Section 8: Macroprudential Buffer Requirements ---")

total_capital = sum(equities)
println("System capital: \$$(round(total_capital, digits=0))B")
println("Total SRISK: \$$(round(total_srisk, digits=1))B")

# Countercyclical buffer
ccyb_fraction = 0.10 * total_srisk / max(total_capital, 1e-12)
ccyb_fraction = min(ccyb_fraction, 0.025)  # cap at 2.5%
println("Countercyclical Buffer (CCyB): $(round(ccyb_fraction*100, digits=2))%")

# SIFI surcharges
println("\nSIFI surcharge requirements:")
for j in sifi_rank[1:5]
    surcharge = min(0.035, sifi_scores[j] * 0.05)
    required_capital = equities[j] * surcharge
    println("  $(bank_names[j]): +$(round(surcharge*100,digits=1))% → additional capital = \$$(round(required_capital,digits=1))B")
end

println("\n✓ Notebook 28 complete")
