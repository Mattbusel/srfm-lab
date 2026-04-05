# =============================================================================
# causal_discovery.jl — Causal Discovery Algorithms
# =============================================================================
# Provides:
#   - PartialCorrelation     Partial correlation controlling for a conditioning set
#   - PCAlgorithm            Peter-Clark constraint-based causal graph discovery
#   - LiNGAM                 Linear Non-Gaussian Acyclic Model (ICA-based)
#   - CausalSummary          Summarise discovered causes of strategy win/loss
#   - run_causal_discovery   Top-level driver
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, JSON3
# =============================================================================

module CausalDiscovery

using Statistics
using LinearAlgebra
using JSON3

export PartialCorrelation, PCAlgorithm, LiNGAM
export CausalSummary, run_causal_discovery

# ── Partial Correlation ───────────────────────────────────────────────────────

"""
Compute partial correlation between variables i and j conditioning on set S.

Uses the inverse covariance (precision) matrix approach:
  ρ(i,j|S) = -P[i,j] / sqrt(P[i,i] * P[j,j])
where P = Σ⁻¹ is the precision matrix restricted to {i,j} ∪ S.

# Arguments
- `X`   : n × p data matrix (n observations, p variables)
- `i`   : first variable index (1-based)
- `j`   : second variable index (1-based)
- `S`   : conditioning set (vector of indices; can be empty)

# Returns
NamedTuple: (partial_corr, t_stat, p_value, n_df)
"""
function PartialCorrelation(
    X::Matrix{Float64},
    i::Int, j::Int,
    S::Vector{Int} = Int[]
)
    n, p = size(X)
    idx  = unique(vcat([i, j], S))
    all(1 .<= idx .<= p) || error("Variable indices out of range")

    if isempty(S)
        # Simple correlation
        xi = X[:, i] .- mean(X[:, i])
        xj = X[:, j] .- mean(X[:, j])
        si = std(xi)
        sj = std(xj)
        ρ  = (si > 1e-12 && sj > 1e-12) ? dot(xi, xj) / (n * si * sj) : 0.0
        n_df = n - 2
    else
        # Partial correlation via precision submatrix
        X_sub = X[:, idx]
        # Standardise
        X_sub = (X_sub .- mean(X_sub; dims=1)) ./ (std(X_sub; dims=1) .+ 1e-12)

        Σ_sub = (X_sub' * X_sub) ./ (n - 1)
        Σ_sub += I * 1e-8   # regularise

        P_sub = inv(Σ_sub)

        # Positions of i and j in the sub-matrix
        pos_i = findfirst(==(i), idx)
        pos_j = findfirst(==(j), idx)

        denom = sqrt(abs(P_sub[pos_i, pos_i] * P_sub[pos_j, pos_j]))
        ρ = denom > 1e-12 ? -P_sub[pos_i, pos_j] / denom : 0.0
        ρ = clamp(ρ, -1.0, 1.0)
        n_df = n - 2 - length(S)
    end

    n_df = max(n_df, 1)
    # Fisher z-transform for t-statistic
    ρ_clamped = clamp(ρ, -0.9999, 0.9999)
    z_stat    = 0.5 * log((1.0 + ρ_clamped) / (1.0 - ρ_clamped))
    se        = 1.0 / sqrt(max(n_df - 1, 1))
    t_stat    = z_stat / se

    # Two-tailed p-value via normal approximation
    p_value   = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))

    (partial_corr=ρ, t_stat=t_stat, p_value=p_value, n_df=n_df)
end

function _normal_cdf(z::Float64)
    # Abramowitz & Stegun approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    p = 1.0 - 0.3989422801 * exp(-z^2 / 2) *
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
             t * (-1.821255978 + t * 1.330274429))))
    z >= 0 ? p : 1.0 - p
end

# ── PC Algorithm ─────────────────────────────────────────────────────────────

"""
PC (Peter-Clark) algorithm for constraint-based causal structure learning.

Algorithm:
1. Start with complete undirected graph.
2. For each pair (i,j): test conditional independence given subsets of neighbours.
   Remove edge if any conditioning set renders them independent (p > α).
3. Orient colliders (v-structures): i → k ← j if i-k-j and i,j not adjacent.
4. Apply Meek orientation rules to propagate orientations.

# Arguments
- `X`           : n × p data matrix
- `var_names`   : variable name strings (length p)
- `alpha`       : significance level for CI tests (default 0.05)
- `max_cond_set`: maximum conditioning set size (default 3; larger = slower)

# Returns
NamedTuple: (adjacency_matrix, edge_list, skeleton, v_structures, var_names)
"""
function PCAlgorithm(
    X::Matrix{Float64};
    var_names::Vector{String}  = String[],
    alpha::Float64              = 0.05,
    max_cond_set::Int           = 3
)
    n, p = size(X)
    n < p + 5 && @warn "Few observations relative to variables; results may be unreliable"
    names = isempty(var_names) ? ["V$i" for i in 1:p] : var_names

    # Step 1: Complete skeleton (adjacency = 1 everywhere except diagonal)
    adj = ones(Bool, p, p)
    for i in 1:p; adj[i, i] = false; end

    sep_sets = Dict{Tuple{Int,Int}, Vector{Int}}()   # separation sets

    # Step 2: Remove edges using conditional independence tests
    for ord in 0:max_cond_set
        changed = false
        for i in 1:p, j in (i+1):p
            !adj[i, j] && continue

            # Neighbours of i (excluding j)
            nbrs_i = [k for k in 1:p if k != i && k != j && adj[i, k]]

            length(nbrs_i) < ord && continue

            # Test all subsets of size ord
            found_sep = false
            for S in _subsets(nbrs_i, ord)
                res = PartialCorrelation(X, i, j, S)
                if res.p_value > alpha
                    adj[i, j] = false
                    adj[j, i] = false
                    sep_sets[(i,j)] = S
                    sep_sets[(j,i)] = S
                    found_sep = true
                    changed = true
                    break
                end
            end
            found_sep && continue

            # Also try conditioning on neighbours of j
            nbrs_j = [k for k in 1:p if k != i && k != j && adj[j, k]]
            for S in _subsets(nbrs_j, ord)
                res = PartialCorrelation(X, i, j, S)
                if res.p_value > alpha
                    adj[i, j] = false
                    adj[j, i] = false
                    sep_sets[(i,j)] = S
                    sep_sets[(j,i)] = S
                    changed = true
                    break
                end
            end
        end
        !changed && break
    end

    # Step 3: Orient v-structures
    # direction[i,j] = true means i → j
    direction = falses(p, p)

    for k in 1:p
        # Find all pairs i,j both adjacent to k but not adjacent to each other
        parents = [v for v in 1:p if v != k && adj[v, k]]
        for a in 1:length(parents), b in (a+1):length(parents)
            i, j = parents[a], parents[b]
            !adj[i, j] || continue   # must be non-adjacent

            # Check if k is in the separation set of i and j
            sep = get(sep_sets, (i, j), nothing)
            if isnothing(sep) || !(k in sep)
                # Orient as collider: i → k ← j
                direction[i, k] = true
                direction[j, k] = true
                adj[k, i] = false   # k does not point back to i
                adj[k, j] = false
            end
        end
    end

    # Step 4: Meek rule 1 — avoid new colliders
    for _ in 1:p
        for i in 1:p, j in 1:p
            direction[i, j] || continue
            for k in 1:p
                k == i || k == j || !adj[j, k] || continue
                !direction[k, j] && !direction[j, k] && !adj[i, k] || continue
                direction[j, k] = true
                adj[k, j]       = false
            end
        end
    end

    # Build output edge list
    edges = NamedTuple[]
    for i in 1:p, j in 1:p
        i >= j && continue
        if direction[i, j]
            push!(edges, (from=names[i], to=names[j], type="directed"))
        elseif direction[j, i]
            push!(edges, (from=names[j], to=names[i], type="directed"))
        elseif adj[i, j]
            push!(edges, (from=names[i], to=names[j], type="undirected"))
        end
    end

    # Adjacency as Float64 matrix for JSON export
    adj_matrix = Float64.(adj)
    for i in 1:p, j in 1:p
        if direction[i, j]; adj_matrix[i, j] = 2.0; end  # 2 = directed
    end

    v_structures = filter(e -> e.type == "directed", edges)

    (
        adjacency_matrix = adj_matrix,
        edge_list        = edges,
        n_edges          = length(edges),
        n_directed       = length(v_structures),
        var_names        = names,
        alpha            = alpha
    )
end

function _subsets(v::Vector{Int}, k::Int)
    k == 0 && return [Int[]]
    k > length(v) && return Vector{Vector{Int}}()
    k == length(v) && return [v]
    result = Vector{Vector{Int}}()
    _subsets_helper!(result, v, k, 1, Int[])
    result
end

function _subsets_helper!(result, v, k, start, current)
    length(current) == k && (push!(result, copy(current)); return)
    for i in start:length(v)
        push!(current, v[i])
        _subsets_helper!(result, v, k, i+1, current)
        pop!(current)
    end
end

# ── LiNGAM ───────────────────────────────────────────────────────────────────

"""
LiNGAM (Linear Non-Gaussian Acyclic Model) for causal direction inference.

For each pair (Xi, Xj): regress Xi on Xj and test residuals for non-Gaussianity.
The causal direction is X_cause → X_effect when residuals of effect ~ non-Gaussian.

Uses the pairwise LiNGAM heuristic (DirectLiNGAM simplified):
  - Fit Xi = β Xj + εᵢ and Xj = γ Xi + εⱼ
  - Test each residual set for non-Gaussianity via kurtosis-based test
  - Direction: Xi → Xj if kurtosis of εᵢ > kurtosis of εⱼ (normalised)

Returns causal direction scores per pair and a full causal order.
"""
function LiNGAM(
    X::Matrix{Float64};
    var_names::Vector{String} = String[],
    n_bootstrap::Int = 100
)
    n, p = size(X)
    names = isempty(var_names) ? ["V$i" for i in 1:p] : var_names

    # Standardise
    X_std = (X .- mean(X; dims=1)) ./ (std(X; dims=1) .+ 1e-12)

    # Excess kurtosis: kurt = E[(x-μ)^4]/σ^4 - 3
    function excess_kurtosis(v::Vector{Float64})
        n_v = length(v)
        m   = mean(v)
        s   = std(v)
        s < 1e-12 && return 0.0
        sum(((v .- m) ./ s).^4) / n_v - 3.0
    end

    # Pairwise causal scores
    pair_results = NamedTuple[]

    for i in 1:p, j in (i+1):p
        xi = X_std[:, i]
        xj = X_std[:, j]

        # Regression xi = β xj + ε_i
        β_ij  = dot(xi, xj) / (dot(xj, xj) + 1e-12)
        ε_ij  = xi .- β_ij .* xj

        # Regression xj = γ xi + ε_j
        γ_ji  = dot(xj, xi) / (dot(xi, xi) + 1e-12)
        ε_ji  = xj .- γ_ji .* xi

        kurt_ij = excess_kurtosis(ε_ij)   # residuals if Xi caused by Xj
        kurt_ji = excess_kurtosis(ε_ji)   # residuals if Xj caused by Xi

        # In LiNGAM: lower |kurtosis| of residuals → true causal direction
        # (residuals of effect on cause are more independent of cause)
        r_ij = abs(cov(xj, ε_ij)) / (std(xj) * std(ε_ij) + 1e-12)
        r_ji = abs(cov(xi, ε_ji)) / (std(xi) * std(ε_ji) + 1e-12)

        # Direction score: positive → Xi causes Xj
        direction_score = r_ji - r_ij

        causal_direction = if direction_score > 0.02
            "$(names[i]) → $(names[j])"
        elseif direction_score < -0.02
            "$(names[j]) → $(names[i])"
        else
            "$(names[i]) ↔ $(names[j]) (indeterminate)"
        end

        push!(pair_results, (
            var_i            = names[i],
            var_j            = names[j],
            direction_score  = direction_score,
            causal_direction = causal_direction,
            residual_dep_ij  = r_ij,
            residual_dep_ji  = r_ji,
            kurt_ij          = kurt_ij,
            kurt_ji          = kurt_ji,
            confident        = abs(direction_score) > 0.05
        ))
    end

    # Sort by confidence
    sort!(pair_results, by=r -> -abs(r.direction_score))

    # Estimate causal order: Klingenberg topological sort heuristic
    causal_scores = Dict{String,Float64}()
    for nm in names; causal_scores[nm] = 0.0; end
    for pr in pair_results
        if pr.direction_score > 0
            causal_scores[pr.var_i] += abs(pr.direction_score)
        else
            causal_scores[pr.var_j] += abs(pr.direction_score)
        end
    end
    causal_order = sort(names; by=nm -> -get(causal_scores, nm, 0.0))

    (
        pair_results   = pair_results,
        causal_order   = causal_order,
        n_pairs        = length(pair_results),
        n_confident    = count(r -> r.confident, pair_results),
        var_names      = names
    )
end

# ── Causal Summary ────────────────────────────────────────────────────────────

"""
Given PC and LiNGAM results, produce human-readable causal summary for
strategy_pnl variable.

Identifies what CAUSES the strategy to win or lose (not just correlates).
"""
function CausalSummary(
    pc_result,
    lingam_result;
    target_var::String = "strategy_pnl"
)
    # Find direct parents of target in PC DAG
    pc_parents = [e.from for e in pc_result.edge_list
                  if e.type == "directed" && e.to == target_var]
    pc_children = [e.to for e in pc_result.edge_list
                   if e.type == "directed" && e.from == target_var]

    # Find LiNGAM-confirmed causes
    lingam_causes = [r.var_i for r in lingam_result.pair_results
                     if r.confident &&
                        r.causal_direction == "$(r.var_i) → $target_var"]
    lingam_causes_j = [r.var_j for r in lingam_result.pair_results
                       if r.confident &&
                          r.causal_direction == "$(r.var_j) → $target_var"]
    append!(lingam_causes, lingam_causes_j)

    # Consensus: variables that appear in both PC parents and LiNGAM causes
    consensus = intersect(pc_parents, lingam_causes)

    findings = String[]
    if !isempty(consensus)
        push!(findings, "Confirmed causes (PC + LiNGAM): $(join(consensus, ", "))")
    end
    if !isempty(setdiff(pc_parents, lingam_causes))
        push!(findings, "PC-only parents: $(join(setdiff(pc_parents, lingam_causes), ", "))")
    end
    if !isempty(lingam_causes) && isempty(consensus)
        push!(findings, "LiNGAM-only causes: $(join(lingam_causes, ", "))")
    end
    if !isempty(pc_children)
        push!(findings, "Variables caused BY strategy P&L: $(join(pc_children, ", "))")
    end
    isempty(findings) && push!(findings,
        "No clear causal parents detected for $target_var at current significance level")

    (
        target_var      = target_var,
        pc_parents      = pc_parents,
        lingam_causes   = unique(lingam_causes),
        consensus_causes = consensus,
        effects_of_pnl  = pc_children,
        findings        = findings,
        causal_order    = lingam_result.causal_order
    )
end

# ── Top-level driver ──────────────────────────────────────────────────────────

"""
Run full causal discovery pipeline on a feature matrix.

Expected variables: BTC_return, ETH_return, SOL_return, ..., strategy_pnl,
                    hour_of_day, day_of_week

Writes `causal_discovery_results.json` to `\$STATS_OUTPUT_DIR`.
"""
function run_causal_discovery(
    X::Matrix{Float64},
    var_names::Vector{String};
    alpha::Float64   = 0.05,
    target_var::String = "strategy_pnl",
    output_dir::String = get(ENV, "STATS_OUTPUT_DIR",
                              joinpath(@__DIR__, "..", "output"))
)
    n, p = size(X)
    println("[causal] Running PC algorithm on $p variables, $n observations (α=$alpha)...")
    pc = PCAlgorithm(X; var_names=var_names, alpha=alpha, max_cond_set=3)

    println("[causal] Running pairwise LiNGAM...")
    lingam = LiNGAM(X; var_names=var_names)

    println("[causal] Generating causal summary for '$target_var'...")
    summary = CausalSummary(pc, lingam; target_var=target_var)

    result = Dict(
        "variables"   => var_names,
        "n_obs"       => n,
        "alpha"       => alpha,
        "pc_algorithm" => Dict(
            "adjacency_matrix" => [pc.adjacency_matrix[i, :] for i in 1:p],
            "edges" => map(e -> Dict("from"=>e.from, "to"=>e.to, "type"=>e.type),
                           pc.edge_list),
            "n_edges"     => pc.n_edges,
            "n_directed"  => pc.n_directed
        ),
        "lingam" => Dict(
            "causal_order"  => lingam.causal_order,
            "n_confident"   => lingam.n_confident,
            "pair_results"  => map(r -> Dict(
                "from"            => r.var_i,
                "to"              => r.var_j,
                "direction"       => r.causal_direction,
                "direction_score" => r.direction_score,
                "confident"       => r.confident
            ), filter(r -> r.confident, lingam.pair_results))
        ),
        "causal_summary" => Dict(
            "target"            => summary.target_var,
            "confirmed_causes"  => summary.consensus_causes,
            "pc_parents"        => summary.pc_parents,
            "lingam_causes"     => summary.lingam_causes,
            "effects"           => summary.effects_of_pnl,
            "findings"          => summary.findings
        )
    )

    mkpath(output_dir)
    out_path = joinpath(output_dir, "causal_discovery_results.json")
    open(out_path, "w") do io
        write(io, JSON3.write(result))
    end
    println("[causal] Results written to $out_path")

    result
end

end  # module CausalDiscovery

# ── CLI self-test ─────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    using .CausalDiscovery
    using Statistics, LinearAlgebra

    println("[causal_discovery] Running self-test...")

    n = 200
    rng_state = UInt64(12345)
    function lcg_rand()
        rng_state = rng_state * 6364136223846793005 + 1442695040888963407
        Float64(rng_state >> 11) / Float64(2^53)
    end
    function lcg_randn()
        u1 = max(lcg_rand(), 1e-15)
        u2 = lcg_rand()
        sqrt(-2*log(u1)) * cos(2π*u2)
    end

    # Generate causal structure:
    # BTC_ret → ETH_ret, BTC_ret → strategy_pnl, hour_of_day → strategy_pnl
    btc_ret     = [lcg_randn() * 0.03 for _ in 1:n]
    eth_ret     = btc_ret .* 0.85 .+ [lcg_randn() * 0.015 for _ in 1:n]
    sol_ret     = btc_ret .* 0.70 .+ [lcg_randn() * 0.025 for _ in 1:n]
    hour_of_day = [Float64(floor(lcg_rand() * 24)) for _ in 1:n]
    day_of_week = [Float64(floor(lcg_rand() * 7))  for _ in 1:n]
    strategy_pnl = 0.4 .* btc_ret .+ 0.1 .* (hour_of_day ./ 24.0 .- 0.5) .+
                   [lcg_randn() * 0.01 for _ in 1:n]

    X = hcat(btc_ret, eth_ret, sol_ret, strategy_pnl, hour_of_day, day_of_week)
    vars = ["BTC_return", "ETH_return", "SOL_return", "strategy_pnl",
            "hour_of_day", "day_of_week"]

    result = run_causal_discovery(X, vars; alpha=0.05)

    pc = result["pc_algorithm"]
    println("  PC: $(pc["n_edges"]) edges, $(pc["n_directed"]) directed")
    println("  PC edges: $([(e["from"]," → ",e["to"]) for e in pc["edges"] if e["type"]=="directed"])")

    lg = result["lingam"]
    println("  LiNGAM causal order: $(join(lg["causal_order"], " → "))")
    println("  LiNGAM confident pairs: $(lg["n_confident"])")

    cs = result["causal_summary"]
    println("  Confirmed causes of strategy_pnl: $(cs["confirmed_causes"])")
    for f in cs["findings"]
        println("    → $f")
    end

    println("[causal_discovery] Self-test complete.")
end
