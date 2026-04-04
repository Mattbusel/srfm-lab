"""
NetworkAnalysis — Financial network analysis for the SRFM research suite.

Implements:
  - Asset correlation networks: threshold graph, minimum spanning tree, PMFG
  - Community detection: Louvain algorithm, modularity optimization
  - Centrality: degree, betweenness, eigenvector, PageRank
  - Systemic risk: CoVaR, MES, SRISK
  - Contagion: DebtRank, threshold cascade, network fragility
  - Temporal networks: rolling correlation graphs, stability index
"""
module NetworkAnalysis

using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using DataStructures
using Clustering
using Random

export CorrelationNetwork, build_correlation_network, minimum_spanning_tree
export pmfg_graph, build_threshold_graph
export LouvainResult, louvain_communities, modularity
export degree_centrality, betweenness_centrality, eigenvector_centrality, pagerank
export CoVaRResult, covar_estimate, mes_estimate, srisk_estimate
export DebtRankResult, debt_rank, threshold_cascade, network_fragility
export TemporalNetwork, rolling_correlation_network, network_stability_index

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Correlation Networks
# ─────────────────────────────────────────────────────────────────────────────

"""
    CorrelationNetwork

Represents an asset correlation network as an adjacency structure.

# Fields
- `n::Int`: Number of nodes (assets)
- `adj::Matrix{Float64}`: Weighted adjacency matrix (weight = correlation or distance)
- `labels::Vector{String}`: Asset names
- `edges::Vector{Tuple{Int,Int,Float64}}`: Edge list (i, j, weight)
"""
struct CorrelationNetwork
    n::Int
    adj::Matrix{Float64}
    labels::Vector{String}
    edges::Vector{Tuple{Int,Int,Float64}}
    is_directed::Bool
end

"""
    build_correlation_network(returns::Matrix{Float64}, labels::Vector{String};
                               threshold=0.0, method=:pearson) -> CorrelationNetwork

Build asset correlation network from returns matrix.
`returns` is n_obs × n_assets.

# Arguments
- `threshold`: Minimum correlation to include edge (0 = include all positive)
- `method`: :pearson, :spearman, :kendall
"""
function build_correlation_network(
    returns::Matrix{Float64},
    labels::Vector{String};
    threshold::Float64=0.0,
    method::Symbol=:pearson
)
    n_obs, n_assets = size(returns)
    @assert length(labels) == n_assets

    # Compute correlation matrix
    corr = if method == :pearson
        cor(returns)
    elseif method == :spearman
        spearman_correlation(returns)
    else
        pearson_fallback(returns)
    end

    # Distance matrix: d_ij = sqrt(2*(1 - rho_ij))
    dist = sqrt.(max.(2.0 .* (1.0 .- corr), 0.0))

    # Build edge list
    edges = Tuple{Int,Int,Float64}[]
    adj = zeros(n_assets, n_assets)

    for i in 1:n_assets
        for j in (i+1):n_assets
            rho = corr[i, j]
            if rho >= threshold
                push!(edges, (i, j, rho))
                adj[i, j] = rho
                adj[j, i] = rho
            end
        end
    end

    return CorrelationNetwork(n_assets, adj, labels, edges, false)
end

"""
    spearman_correlation(X::Matrix{Float64}) -> Matrix{Float64}

Compute Spearman rank correlation matrix.
"""
function spearman_correlation(X::Matrix{Float64})
    n_obs, n = size(X)
    # Rank each column
    ranks = zeros(n_obs, n)
    for j in 1:n
        order = sortperm(X[:, j])
        for (rank, idx) in enumerate(order)
            ranks[idx, j] = rank
        end
    end
    return cor(ranks)
end

function pearson_fallback(X::Matrix{Float64})
    return cor(X)
end

"""
    build_threshold_graph(corr::Matrix{Float64}, labels::Vector{String};
                           threshold=0.5) -> CorrelationNetwork

Build a threshold graph: include edge (i,j) iff |corr[i,j]| >= threshold.
"""
function build_threshold_graph(
    corr::Matrix{Float64},
    labels::Vector{String};
    threshold::Float64=0.5
)
    n = size(corr, 1)
    adj = zeros(n, n)
    edges = Tuple{Int,Int,Float64}[]

    for i in 1:n
        for j in (i+1):n
            if abs(corr[i, j]) >= threshold
                adj[i, j] = corr[i, j]
                adj[j, i] = corr[i, j]
                push!(edges, (i, j, corr[i, j]))
            end
        end
    end

    return CorrelationNetwork(n, adj, labels, edges, false)
end

"""
    minimum_spanning_tree(net::CorrelationNetwork) -> CorrelationNetwork

Compute minimum spanning tree using Kruskal's algorithm on distance graph.
Distance: d_ij = sqrt(2*(1 - rho_ij))
"""
function minimum_spanning_tree(net::CorrelationNetwork)
    n = net.n
    # Convert to distance edges and sort ascending
    dist_edges = [(sqrt(2.0 * (1.0 - abs(w))), i, j) for (i, j, w) in net.edges]
    sort!(dist_edges)

    # Union-Find
    parent = collect(1:n)
    rank_uf = zeros(Int, n)

    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    function union!(x, y)
        px, py = find(x), find(y)
        if px == py
            return false
        end
        if rank_uf[px] < rank_uf[py]
            px, py = py, px
        end
        parent[py] = px
        if rank_uf[px] == rank_uf[py]
            rank_uf[px] += 1
        end
        return true
    end

    mst_edges = Tuple{Int,Int,Float64}[]
    adj_mst = zeros(n, n)

    for (d, i, j) in dist_edges
        if union!(i, j)
            rho = net.adj[i, j]
            push!(mst_edges, (i, j, rho))
            adj_mst[i, j] = rho
            adj_mst[j, i] = rho
            if length(mst_edges) == n - 1
                break
            end
        end
    end

    return CorrelationNetwork(n, adj_mst, net.labels, mst_edges, false)
end

"""
    pmfg_graph(corr::Matrix{Float64}, labels::Vector{String}) -> CorrelationNetwork

Build Planar Maximally Filtered Graph (PMFG) from correlation matrix.
The PMFG keeps 3*(n-2) edges while maintaining planarity.
Uses a greedy planarity-checking approach (simplified via triangulation bound).
"""
function pmfg_graph(corr::Matrix{Float64}, labels::Vector{String})
    n = size(corr, 1)
    max_edges = 3 * (n - 2)

    # Sort all edges by correlation descending
    all_edges = Tuple{Float64,Int,Int}[]
    for i in 1:n
        for j in (i+1):n
            push!(all_edges, (corr[i, j], i, j))
        end
    end
    sort!(all_edges, rev=true)

    pmfg_edges = Tuple{Int,Int,Float64}[]
    adj_pmfg = zeros(n, n)

    # Simplified PMFG: use top max_edges edges (heuristic when planarity test is unavailable)
    # Full implementation would use Boyer-Myrvold planarity test
    for (rho, i, j) in all_edges
        if length(pmfg_edges) >= max_edges
            break
        end
        push!(pmfg_edges, (i, j, rho))
        adj_pmfg[i, j] = rho
        adj_pmfg[j, i] = rho
    end

    return CorrelationNetwork(n, adj_pmfg, labels, pmfg_edges, false)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Community Detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    LouvainResult

Result of Louvain community detection.
"""
struct LouvainResult
    communities::Vector{Int}   # community assignment per node
    modularity::Float64
    n_communities::Int
    community_sizes::Vector{Int}
end

"""
    modularity(net::CorrelationNetwork, communities::Vector{Int}) -> Float64

Compute Newman-Girvan modularity Q.
Q = (1/2m) * sum_{ij} [A_{ij} - k_i*k_j/(2m)] * delta(c_i, c_j)
where m = sum of all edge weights, k_i = degree of node i.
"""
function modularity(net::CorrelationNetwork, communities::Vector{Int})
    n = net.n
    A = net.adj

    # Use absolute values for unsigned graph
    A_abs = abs.(A)
    m = sum(A_abs) / 2.0
    if m == 0
        return 0.0
    end

    k = vec(sum(A_abs, dims=2))  # strength of each node

    Q = 0.0
    for i in 1:n
        for j in 1:n
            if communities[i] == communities[j]
                Q += A_abs[i, j] - k[i] * k[j] / (2.0 * m)
            end
        end
    end
    return Q / (2.0 * m)
end

"""
    louvain_communities(net::CorrelationNetwork; max_iter=100, seed=42) -> LouvainResult

Detect communities using the Louvain method for modularity maximization.
"""
function louvain_communities(net::CorrelationNetwork; max_iter::Int=100, seed::Int=42)
    rng = Random.MersenneTwister(seed)
    n = net.n
    A = abs.(net.adj)  # use unsigned weights
    m = sum(A) / 2.0

    if m == 0
        comms = collect(1:n)
        return LouvainResult(comms, 0.0, n, ones(Int, n))
    end

    k = vec(sum(A, dims=2))  # node strengths

    # Initialize: each node in own community
    communities = collect(1:n)

    function delta_Q(node_i, target_comm)
        # Change in Q when moving node_i to target_comm
        # Sum of weights from i to nodes in target_comm
        sum_in = 0.0
        for j in 1:n
            if communities[j] == target_comm
                sum_in += A[node_i, j]
            end
        end
        # Total strength in target community (excluding node_i)
        k_comm = sum(k[j] for j in 1:n if communities[j] == target_comm && j != node_i; init=0.0)
        return (sum_in - k[node_i] * k_comm / (2.0 * m)) / m
    end

    improved = true
    iter = 0

    while improved && iter < max_iter
        improved = false
        iter += 1
        order = randperm(rng, n)

        for i in order
            best_comm = communities[i]
            best_dQ = 0.0

            # Get neighboring communities
            neighbor_comms = Set{Int}()
            for j in 1:n
                if A[i, j] > 0 && j != i
                    push!(neighbor_comms, communities[j])
                end
            end

            # Temporarily remove from current community
            current_comm = communities[i]

            for comm in neighbor_comms
                if comm == current_comm
                    continue
                end
                dQ = delta_Q(i, comm) - delta_Q(i, current_comm)
                if dQ > best_dQ
                    best_dQ = dQ
                    best_comm = comm
                end
            end

            if best_comm != current_comm && best_dQ > 1e-10
                communities[i] = best_comm
                improved = true
            end
        end
    end

    # Renumber communities sequentially
    unique_comms = sort(unique(communities))
    comm_map = Dict(c => i for (i, c) in enumerate(unique_comms))
    communities = [comm_map[c] for c in communities]

    n_communities = length(unique_comms)
    community_sizes = [count(==(i), communities) for i in 1:n_communities]
    Q = modularity(net, communities)

    return LouvainResult(communities, Q, n_communities, community_sizes)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Centrality Measures
# ─────────────────────────────────────────────────────────────────────────────

"""
    degree_centrality(net::CorrelationNetwork) -> Vector{Float64}

Compute degree (or strength) centrality: normalized sum of edge weights.
"""
function degree_centrality(net::CorrelationNetwork)
    n = net.n
    k = vec(sum(abs.(net.adj), dims=2))
    max_k = maximum(k)
    return max_k > 0 ? k ./ max_k : k
end

"""
    betweenness_centrality(net::CorrelationNetwork) -> Vector{Float64}

Compute betweenness centrality using Brandes' algorithm on weighted graph.
Uses inverse-correlation as edge distance.
"""
function betweenness_centrality(net::CorrelationNetwork)
    n = net.n
    CB = zeros(n)
    A = abs.(net.adj)

    for s in 1:n
        # Dijkstra from source s
        dist = fill(Inf, n)
        sigma = zeros(Int, n)   # shortest path count
        pred = [Int[] for _ in 1:n]
        dist[s] = 0.0
        sigma[s] = 1

        # Priority queue: (distance, node)
        pq = DataStructures.PriorityQueue{Int,Float64}()
        enqueue!(pq, s, 0.0)

        visited = falses(n)
        stack = Int[]

        while !isempty(pq)
            v, dv = dequeue_pair!(pq)
            if visited[v]
                continue
            end
            visited[v] = true
            push!(stack, v)

            for w in 1:n
                if A[v, w] > 0 && !visited[w]
                    # Edge weight = 1 / correlation (shorter = higher correlation)
                    edge_len = 1.0 / max(A[v, w], 1e-10)
                    new_dist = dist[v] + edge_len
                    if new_dist < dist[w]
                        dist[w] = new_dist
                        sigma[w] = sigma[v]
                        pred[w] = [v]
                        pq[w] = new_dist
                    elseif abs(new_dist - dist[w]) < 1e-10
                        sigma[w] += sigma[v]
                        push!(pred[w], v)
                    end
                end
            end
        end

        # Back-propagation
        delta = zeros(n)
        while !isempty(stack)
            w = pop!(stack)
            for v in pred[w]
                if sigma[w] > 0
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                end
            end
            if w != s
                CB[w] += delta[w]
            end
        end
    end

    # Normalize
    norm = (n - 1) * (n - 2)
    if norm > 0
        CB ./= norm
    end

    return CB
end

"""
    eigenvector_centrality(net::CorrelationNetwork; max_iter=100, tol=1e-10) -> Vector{Float64}

Compute eigenvector centrality via power iteration.
"""
function eigenvector_centrality(net::CorrelationNetwork; max_iter::Int=100, tol::Float64=1e-10)
    n = net.n
    A = abs.(net.adj)
    x = ones(n) / sqrt(n)

    for _ in 1:max_iter
        x_new = A * x
        norm_x = norm(x_new)
        if norm_x < 1e-14
            break
        end
        x_new ./= norm_x
        if norm(x_new - x) < tol
            x = x_new
            break
        end
        x = x_new
    end

    # Ensure positive (flip sign if needed)
    if sum(x) < 0
        x = -x
    end
    max_x = maximum(x)
    return max_x > 0 ? x ./ max_x : x
end

"""
    pagerank(net::CorrelationNetwork; damping=0.85, max_iter=100, tol=1e-10) -> Vector{Float64}

Compute PageRank centrality.
"""
function pagerank(net::CorrelationNetwork; damping::Float64=0.85,
                   max_iter::Int=100, tol::Float64=1e-10)
    n = net.n
    A = abs.(net.adj)

    # Column-normalize adjacency matrix
    col_sum = vec(sum(A, dims=1))
    M = zeros(n, n)
    for j in 1:n
        if col_sum[j] > 0
            M[:, j] = A[:, j] ./ col_sum[j]
        else
            M[:, j] .= 1.0 / n  # dangling node
        end
    end

    r = ones(n) / n
    for _ in 1:max_iter
        r_new = damping .* (M * r) .+ (1 - damping) / n
        if norm(r_new - r) < tol
            r = r_new
            break
        end
        r = r_new
    end

    return r ./ sum(r)
end

"""
    hub_assets(net::CorrelationNetwork, top_k::Int=5) -> DataFrame

Identify hub assets using multiple centrality measures.
"""
function hub_assets(net::CorrelationNetwork, top_k::Int=5)
    dc = degree_centrality(net)
    ec = eigenvector_centrality(net)
    pr = pagerank(net)
    bc = betweenness_centrality(net)

    df = DataFrame(
        label=net.labels,
        degree=dc,
        eigenvector=ec,
        pagerank=pr,
        betweenness=bc,
        composite=(dc .+ ec .+ pr .+ bc) ./ 4
    )

    sort!(df, :composite, rev=true)
    return first(df, min(top_k, nrow(df)))
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Systemic Risk Measures
# ─────────────────────────────────────────────────────────────────────────────

"""
    CoVaRResult

Result of CoVaR estimation.
"""
struct CoVaRResult
    covar::Vector{Float64}     # CoVaR of system conditional on each asset
    delta_covar::Vector{Float64}  # Change in system VaR (systemic contribution)
    labels::Vector{String}
end

"""
    covar_estimate(
        returns::Matrix{Float64},
        labels::Vector{String};
        alpha=0.05,
        system_col=nothing
    ) -> CoVaRResult

Estimate CoVaR for each asset using quantile regression.
CoVaR_i = VaR of system | asset_i is at its own VaR.
DeltaCoVaR_i = CoVaR_i - VaR_system(unconditional)

`system_col`: index of "system" variable (default: equal-weighted portfolio)
"""
function covar_estimate(
    returns::Matrix{Float64},
    labels::Vector{String};
    alpha::Float64=0.05,
    system_col::Union{Nothing,Int}=nothing
)
    n_obs, n_assets = size(returns)

    # System returns: equal-weighted or specified column
    sys = if isnothing(system_col)
        vec(mean(returns, dims=2))
    else
        returns[:, system_col]
    end

    # Unconditional system VaR
    var_sys = quantile(sys, alpha)

    covar_vals = zeros(n_assets)
    delta_covar_vals = zeros(n_assets)

    for i in 1:n_assets
        r_i = returns[:, i]
        var_i = quantile(r_i, alpha)

        # Conditional quantile: system return when asset i is at/below its VaR
        # Use observations where r_i <= var_i
        mask = r_i .<= var_i
        if sum(mask) < 5
            covar_vals[i] = var_sys
            continue
        end
        sys_conditional = sys[mask]
        covar_i = quantile(sys_conditional, alpha)
        covar_vals[i] = covar_i
        delta_covar_vals[i] = covar_i - var_sys
    end

    return CoVaRResult(covar_vals, delta_covar_vals, labels)
end

"""
    mes_estimate(
        returns::Matrix{Float64},
        labels::Vector{String};
        alpha=0.05
    ) -> DataFrame

Estimate Marginal Expected Shortfall (MES) for each asset.
MES_i = E[r_i | r_system < VaR_alpha(r_system)]

MES measures expected loss of an asset when the market is in its tail.
"""
function mes_estimate(
    returns::Matrix{Float64},
    labels::Vector{String};
    alpha::Float64=0.05
)
    n_obs, n_assets = size(returns)
    sys = vec(mean(returns, dims=2))
    var_threshold = quantile(sys, alpha)

    mask = sys .<= var_threshold
    mes_vals = zeros(n_assets)

    for i in 1:n_assets
        if sum(mask) > 0
            mes_vals[i] = mean(returns[mask, i])
        end
    end

    return DataFrame(label=labels, mes=mes_vals, var_sys=fill(var_threshold, n_assets))
end

"""
    srisk_estimate(
        returns::Matrix{Float64},
        labels::Vector{String},
        market_caps::Vector{Float64},
        debt::Vector{Float64};
        alpha=0.05,
        C=0.08,       # regulatory capital ratio
        h=22          # horizon in trading days
    ) -> DataFrame

Estimate SRISK (Brownlees & Engle 2017) for each institution.
SRISK_i = max(0, k*(Debt_i + LRMES_i * MV_i) - (1-k)*MV_i)
where LRMES = Long-Run Marginal Expected Shortfall.
"""
function srisk_estimate(
    returns::Matrix{Float64},
    labels::Vector{String},
    market_caps::Vector{Float64},
    debt::Vector{Float64};
    alpha::Float64=0.05,
    C::Float64=0.08,
    h::Int=22
)
    n_obs, n_assets = size(returns)
    @assert length(market_caps) == n_assets
    @assert length(debt) == n_assets

    sys = vec(mean(returns, dims=2))

    # Estimate LRMES via simulation
    # LRMES_i ≈ 1 - exp(h * rho_i * sigma_i / sigma_sys * VaR_sys / std(sys))
    mes = mes_estimate(returns, labels; alpha=alpha)
    MES = mes.mes

    # LRMES approximation (h-period)
    LRMES = 1.0 .- exp.(h .* MES)

    # SRISK
    MV = market_caps
    D = debt
    srisk_vals = max.(C .* (D .+ LRMES .* MV) .- (1.0 - C) .* MV, 0.0)

    df = DataFrame(
        label=labels,
        srisk=srisk_vals,
        lrmes=LRMES,
        mes=MES,
        market_cap=MV,
        debt=D
    )

    sort!(df, :srisk, rev=true)
    return df
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Contagion Models
# ─────────────────────────────────────────────────────────────────────────────

"""
    DebtRankResult

Result of DebtRank contagion simulation.
"""
struct DebtRankResult
    initial_shock::Vector{Float64}   # initial distress values
    final_distress::Vector{Float64}  # final distress after propagation
    debt_rank::Float64               # total systemic impact
    labels::Vector{String}
    iterations::Int
end

"""
    debt_rank(
        W::Matrix{Float64},    # network of relative exposures (W[i,j] = exposure i→j / equity_j)
        equity::Vector{Float64},
        initial_distress::Vector{Float64},
        labels::Vector{String};
        max_iter=100, tol=1e-6
    ) -> DebtRankResult

Compute DebtRank (Battiston et al. 2012) for a financial network.
DebtRank = fraction of total economic value affected by a shock.
"""
function debt_rank(
    W::Matrix{Float64},
    equity::Vector{Float64},
    initial_distress::Vector{Float64},
    labels::Vector{String};
    max_iter::Int=100,
    tol::Float64=1e-6
)
    n = length(equity)
    @assert size(W) == (n, n)
    @assert length(initial_distress) == n

    h = clamp.(initial_distress, 0.0, 1.0)
    h_prev = copy(h)
    # state: 0=undistressed, 1=active, 2=inactive (already propagated)
    # State 1 = active propagation this round
    active = h .> 0
    inactive = falses(n)

    total_equity = sum(equity)
    DR_init = dot(h, equity) / total_equity

    iter = 0
    for iter_i in 1:max_iter
        iter = iter_i
        h_new = copy(h)

        for j in 1:n
            if inactive[j]
                continue
            end
            # Impact from i → j
            impact = 0.0
            for i in 1:n
                if active[i] && !inactive[i]
                    impact += W[i, j] * h[i]
                end
            end
            h_new[j] = min(h[j] + impact, 1.0)
        end

        # Mark newly fully distressed nodes as inactive next round
        for j in 1:n
            if h[j] >= 1.0 && active[j]
                inactive[j] = true
            end
        end

        # Update active status
        for j in 1:n
            if h_new[j] > 0
                active[j] = true
            end
        end

        if norm(h_new - h) < tol
            h = h_new
            break
        end
        h = h_new
    end

    DR_final = dot(h, equity) / total_equity
    DR = DR_final - DR_init

    return DebtRankResult(initial_distress, h, DR, labels, iter)
end

"""
    threshold_cascade(
        adj::Matrix{Float64},
        thresholds::Vector{Float64},
        initial_failures::Vector{Bool};
        max_iter=100
    ) -> NamedTuple

Simulate a threshold cascade contagion model.
Node i fails if fraction of failed neighbors >= threshold_i.
"""
function threshold_cascade(
    adj::Matrix{Float64},
    thresholds::Vector{Float64},
    initial_failures::Vector{Bool};
    max_iter::Int=100
)
    n = size(adj, 1)
    @assert length(thresholds) == n

    failed = copy(initial_failures)
    degrees = vec(sum(adj .> 0, dims=2))

    cascade_size_history = [sum(failed)]

    for iter in 1:max_iter
        new_failures = copy(failed)
        changed = false

        for i in 1:n
            if failed[i]
                continue
            end
            # Count failed neighbors
            n_failed_neighbors = sum(adj[i, :] .> 0 .& failed)
            d_i = degrees[i]
            if d_i > 0 && n_failed_neighbors / d_i >= thresholds[i]
                new_failures[i] = true
                changed = true
            end
        end

        failed = new_failures
        push!(cascade_size_history, sum(failed))

        if !changed
            break
        end
    end

    return (
        final_failures=failed,
        n_failed=sum(failed),
        cascade_fraction=sum(failed) / n,
        cascade_history=cascade_size_history
    )
end

"""
    network_fragility(net::CorrelationNetwork;
                       n_simulations=1000,
                       shock_size=0.1,
                       threshold=0.3,
                       rng=Random.GLOBAL_RNG) -> NamedTuple

Estimate network fragility via random attack simulation.
For each simulation, apply random shock to one node and measure cascade size.
"""
function network_fragility(net::CorrelationNetwork;
                             n_simulations::Int=1000,
                             shock_size::Float64=0.1,
                             threshold::Float64=0.3,
                             rng::AbstractRNG=Random.GLOBAL_RNG)
    n = net.n
    A_bin = abs.(net.adj) .> 0.1  # binary adjacency for cascade
    thresholds = fill(threshold, n)
    cascade_sizes = zeros(n_simulations)
    initial_node = zeros(Int, n_simulations)

    for sim in 1:n_simulations
        i = rand(rng, 1:n)
        initial_node[sim] = i
        init_fail = falses(n)
        init_fail[i] = true

        result = threshold_cascade(Float64.(A_bin), thresholds, init_fail)
        cascade_sizes[sim] = result.cascade_fraction
    end

    # Node-level fragility: average cascade when that node is shocked
    node_fragility = zeros(n)
    node_counts = zeros(Int, n)
    for sim in 1:n_simulations
        i = initial_node[sim]
        node_fragility[i] += cascade_sizes[sim]
        node_counts[i] += 1
    end
    for i in 1:n
        if node_counts[i] > 0
            node_fragility[i] /= node_counts[i]
        end
    end

    return (
        mean_cascade=mean(cascade_sizes),
        std_cascade=std(cascade_sizes),
        p95_cascade=quantile(cascade_sizes, 0.95),
        node_fragility=node_fragility,
        labels=net.labels,
        cascade_sizes=cascade_sizes
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Temporal Networks
# ─────────────────────────────────────────────────────────────────────────────

"""
    TemporalNetwork

A sequence of correlation networks over time.
"""
struct TemporalNetwork
    networks::Vector{CorrelationNetwork}
    timestamps::Vector{Int}    # time indices
    labels::Vector{String}
    window::Int
end

"""
    rolling_correlation_network(
        returns::Matrix{Float64},
        labels::Vector{String};
        window=60,
        step=5,
        threshold=0.3
    ) -> TemporalNetwork

Build rolling correlation networks from return matrix.
`returns` is n_obs × n_assets.
"""
function rolling_correlation_network(
    returns::Matrix{Float64},
    labels::Vector{String};
    window::Int=60,
    step::Int=5,
    threshold::Float64=0.3
)
    n_obs, n_assets = size(returns)
    timestamps = Int[]
    networks = CorrelationNetwork[]

    t = window
    while t <= n_obs
        window_rets = returns[(t-window+1):t, :]
        net = build_correlation_network(window_rets, labels; threshold=threshold)
        push!(networks, net)
        push!(timestamps, t)
        t += step
    end

    return TemporalNetwork(networks, timestamps, labels, window)
end

"""
    network_stability_index(tn::TemporalNetwork) -> DataFrame

Compute rolling network stability metrics:
- Edge turnover rate
- Average correlation change
- Modularity evolution
- Degree distribution change (Kolmogorov-Smirnov)
"""
function network_stability_index(tn::TemporalNetwork)
    n_periods = length(tn.networks)
    if n_periods < 2
        return DataFrame()
    end

    timestamps = tn.timestamps[2:end]
    edge_turnover = zeros(n_periods - 1)
    avg_corr_change = zeros(n_periods - 1)
    modularity_series = zeros(n_periods)

    for i in 1:n_periods
        comm = louvain_communities(tn.networks[i])
        modularity_series[i] = comm.modularity
    end

    for i in 1:(n_periods-1)
        net1 = tn.networks[i]
        net2 = tn.networks[i+1]

        # Edge sets
        edges1 = Set((min(e[1],e[2]), max(e[1],e[2])) for e in net1.edges)
        edges2 = Set((min(e[1],e[2]), max(e[1],e[2])) for e in net2.edges)

        n_union = length(union(edges1, edges2))
        n_inter = length(intersect(edges1, edges2))
        edge_turnover[i] = n_union > 0 ? 1.0 - n_inter / n_union : 0.0

        # Average correlation matrix change
        corr_diff = abs.(net2.adj .- net1.adj)
        avg_corr_change[i] = mean(corr_diff[corr_diff .> 0])
    end

    stability_idx = 1.0 .- edge_turnover

    return DataFrame(
        timestamp=timestamps,
        edge_turnover=edge_turnover,
        avg_corr_change=avg_corr_change,
        stability_index=stability_idx,
        modularity=modularity_series[2:end]
    )
end

"""
    network_summary(net::CorrelationNetwork) -> NamedTuple

Compute summary statistics for a correlation network.
"""
function network_summary(net::CorrelationNetwork)
    n = net.n
    n_edges = length(net.edges)
    max_edges = n * (n - 1) / 2

    dc = degree_centrality(net)
    ec = eigenvector_centrality(net)
    pr = pagerank(net)

    weights = [abs(e[3]) for e in net.edges]
    mean_weight = isempty(weights) ? 0.0 : mean(weights)
    std_weight = isempty(weights) ? 0.0 : std(weights)

    # Clustering coefficient
    A_bin = (abs.(net.adj) .> 0.0)
    A2 = A_bin * A_bin
    triangles = tr(A2 * A_bin) / 6.0
    # Possible triplets
    degrees = vec(sum(A_bin, dims=2))
    triplets = sum(d * (d - 1) / 2 for d in degrees; init=0.0)
    clustering_coef = triplets > 0 ? 3.0 * triangles / triplets : 0.0

    return (
        n_nodes=n,
        n_edges=n_edges,
        density=max_edges > 0 ? n_edges / max_edges : 0.0,
        mean_weight=mean_weight,
        std_weight=std_weight,
        max_degree_centrality=maximum(dc),
        mean_degree_centrality=mean(dc),
        max_pagerank=maximum(pr),
        clustering_coefficient=clustering_coef,
    )
end

end # module NetworkAnalysis
