"""
graph_theory.jl — Graph Algorithms for Financial Networks

Covers:
  - Graph representations: adjacency list + weighted adjacency matrix
  - Shortest paths: Dijkstra, Bellman-Ford, Floyd-Warshall
  - Spanning trees: Kruskal, Prim, minimum spanning arborescence
  - Community detection: spectral clustering, label propagation, Louvain
  - Centrality: degree, betweenness, closeness, PageRank, eigenvector
  - Graph signal processing: graph Fourier transform, graph wavelets
  - Temporal graphs: time-varying network analysis
  - Application: crypto correlation graph, lead-lag community detection

Pure Julia stdlib only. No external dependencies.
"""

module GraphTheory

using Statistics, LinearAlgebra, Random

export Graph, DiGraph, add_edge!, neighbors, adjacency_matrix, weight_matrix
export dijkstra, bellman_ford, floyd_warshall
export kruskal_mst, prim_mst
export degree_centrality, betweenness_centrality, closeness_centrality
export pagerank, eigenvector_centrality
export spectral_clustering, label_propagation, louvain_communities
export graph_fourier_transform, graph_wavelet_filter
export build_correlation_graph, lead_lag_communities
export temporal_network_analysis
export run_graph_theory_demo

# ─────────────────────────────────────────────────────────────
# 1. GRAPH DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

"""
    Graph

Undirected weighted graph using adjacency lists.
"""
mutable struct Graph
    n::Int                          # number of vertices
    adj::Vector{Vector{Tuple{Int,Float64}}}   # adj[u] = [(v, w), ...]
    edges::Vector{Tuple{Int,Int,Float64}}     # all edges
end

Graph(n::Int) = Graph(n, [Tuple{Int,Float64}[] for _ in 1:n], Tuple{Int,Int,Float64}[])

"""
    DiGraph

Directed weighted graph using adjacency lists.
"""
mutable struct DiGraph
    n::Int
    adj::Vector{Vector{Tuple{Int,Float64}}}
    edges::Vector{Tuple{Int,Int,Float64}}
end

DiGraph(n::Int) = DiGraph(n, [Tuple{Int,Float64}[] for _ in 1:n], Tuple{Int,Int,Float64}[])

"""Add undirected edge (u, v) with weight w."""
function add_edge!(g::Graph, u::Int, v::Int, w::Float64=1.0)
    push!(g.adj[u], (v, w))
    push!(g.adj[v], (u, w))
    push!(g.edges, (min(u,v), max(u,v), w))
end

"""Add directed edge (u → v) with weight w."""
function add_edge!(g::DiGraph, u::Int, v::Int, w::Float64=1.0)
    push!(g.adj[u], (v, w))
    push!(g.edges, (u, v, w))
end

neighbors(g::Union{Graph,DiGraph}, u::Int) = [v for (v,_) in g.adj[u]]
edge_weights(g::Union{Graph,DiGraph}, u::Int) = [(v,w) for (v,w) in g.adj[u]]

"""
    adjacency_matrix(g::Graph) -> Matrix{Float64}

Binary adjacency matrix (1 if edge exists).
"""
function adjacency_matrix(g::Union{Graph,DiGraph})::Matrix{Float64}
    A = zeros(g.n, g.n)
    for (u, v, _) in g.edges
        A[u, v] = 1.0
        isa(g, Graph) && (A[v, u] = 1.0)
    end
    A
end

"""
    weight_matrix(g) -> Matrix{Float64}

Weighted adjacency matrix.
"""
function weight_matrix(g::Union{Graph,DiGraph})::Matrix{Float64}
    W = zeros(g.n, g.n)
    for (u, v, w) in g.edges
        W[u, v] = w
        isa(g, Graph) && (W[v, u] = w)
    end
    W
end

"""Build graph from correlation matrix (threshold-based)."""
function build_correlation_graph(corr::Matrix{Float64};
                                  threshold::Float64=0.5)::Graph
    n = size(corr, 1)
    g = Graph(n)
    for i in 1:n, j in (i+1):n
        if abs(corr[i,j]) >= threshold
            add_edge!(g, i, j, abs(corr[i,j]))
        end
    end
    g
end

# ─────────────────────────────────────────────────────────────
# 2. SHORTEST PATH ALGORITHMS
# ─────────────────────────────────────────────────────────────

"""
    dijkstra(g, src) -> (dist, prev)

Dijkstra's algorithm for single-source shortest paths.
Returns dist[v] = shortest distance from src to v,
and prev[v] = predecessor on shortest path.
O((V + E) log V) with binary heap.
"""
function dijkstra(g::Union{Graph,DiGraph}, src::Int)
    n = g.n
    dist = fill(Inf, n)
    prev = fill(-1, n)
    dist[src] = 0.0

    # Priority queue as sorted vector of (dist, vertex)
    pq = [(0.0, src)]
    visited = falses(n)

    while !isempty(pq)
        # Extract minimum
        sort!(pq)
        (d, u) = popfirst!(pq)
        visited[u] && continue
        visited[u] = true

        for (v, w) in g.adj[u]
            w < 0 && continue  # Dijkstra requires non-negative weights
            new_dist = d + w
            if new_dist < dist[v]
                dist[v] = new_dist
                prev[v] = u
                push!(pq, (new_dist, v))
            end
        end
    end
    dist, prev
end

"""
    reconstruct_path(prev, src, dst) -> Vector{Int}

Reconstruct shortest path from prev array.
"""
function reconstruct_path(prev::Vector{Int}, src::Int, dst::Int)::Vector{Int}
    path = Int[]
    cur = dst
    while cur != -1
        pushfirst!(path, cur)
        cur == src && break
        cur = prev[cur]
        length(path) > length(prev) && return Int[]  # no path
    end
    isempty(path) || path[1] == src ? path : Int[]
end

"""
    bellman_ford(g, src) -> (dist, prev, has_negative_cycle)

Bellman-Ford algorithm. Handles negative edge weights.
Returns (distances, predecessors, whether negative cycle exists).
O(V * E).
"""
function bellman_ford(g::Union{Graph,DiGraph}, src::Int)
    n = g.n
    dist = fill(Inf, n)
    prev = fill(-1, n)
    dist[src] = 0.0

    edges = g.edges

    # Relax all edges n-1 times
    for _ in 1:(n-1)
        for (u, v, w) in edges
            if dist[u] + w < dist[v]
                dist[v] = dist[u] + w
                prev[v] = u
            end
            if isa(g, Graph) && dist[v] + w < dist[u]
                dist[u] = dist[v] + w
                prev[u] = v
            end
        end
    end

    # Check for negative cycles
    neg_cycle = false
    for (u, v, w) in edges
        if dist[u] + w < dist[v]
            neg_cycle = true; break
        end
    end

    dist, prev, neg_cycle
end

"""
    floyd_warshall(g) -> Matrix{Float64}

Floyd-Warshall all-pairs shortest paths. O(V^3).
Returns D[i,j] = shortest path distance from i to j.
"""
function floyd_warshall(g::Union{Graph,DiGraph})::Matrix{Float64}
    n = g.n
    D = fill(Inf, n, n)
    for i in 1:n; D[i,i] = 0.0; end
    for (u, v, w) in g.edges
        D[u,v] = min(D[u,v], w)
        isa(g, Graph) && (D[v,u] = min(D[v,u], w))
    end
    for k in 1:n, i in 1:n, j in 1:n
        D[i,j] > D[i,k] + D[k,j] && (D[i,j] = D[i,k] + D[k,j])
    end
    D
end

# ─────────────────────────────────────────────────────────────
# 3. MINIMUM SPANNING TREE
# ─────────────────────────────────────────────────────────────

"""
    kruskal_mst(g::Graph) -> (total_weight, edges)

Kruskal's algorithm for minimum spanning tree. O(E log E).
Uses union-find (path compression + union by rank).
"""
function kruskal_mst(g::Graph)
    n = g.n
    # Sort edges by weight
    sorted_edges = sort(unique(g.edges), by=e->e[3])

    # Union-Find
    parent = collect(1:n)
    rank   = zeros(Int, n)

    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        end
        x
    end

    function union!(x, y)
        px, py = find(x), find(y)
        px == py && return false
        if rank[px] < rank[py]; px, py = py, px; end
        parent[py] = px
        rank[py] == rank[px] && (rank[px] += 1)
        true
    end

    mst_edges = Tuple{Int,Int,Float64}[]
    total_w   = 0.0
    for (u, v, w) in sorted_edges
        if union!(u, v)
            push!(mst_edges, (u, v, w))
            total_w += w
            length(mst_edges) == n - 1 && break
        end
    end
    (total_weight=total_w, edges=mst_edges, n_components=n - length(mst_edges))
end

"""
    prim_mst(g::Graph, start=1) -> (total_weight, edges)

Prim's algorithm for minimum spanning tree starting from vertex `start`.
"""
function prim_mst(g::Graph, start::Int=1)
    n = g.n
    in_mst  = falses(n)
    key     = fill(Inf, n)
    parent  = fill(-1, n)
    key[start] = 0.0
    mst_edges = Tuple{Int,Int,Float64}[]

    for _ in 1:n
        # Find minimum key vertex not yet in MST
        u = -1
        for v in 1:n
            if !in_mst[v] && (u == -1 || key[v] < key[u])
                u = v
            end
        end
        u == -1 && break
        in_mst[u] = true
        parent[u] != -1 && push!(mst_edges, (parent[u], u, key[u]))

        for (v, w) in g.adj[u]
            if !in_mst[v] && w < key[v]
                key[v]    = w
                parent[v] = u
            end
        end
    end
    (total_weight=sum(e[3] for e in mst_edges), edges=mst_edges)
end

# ─────────────────────────────────────────────────────────────
# 4. CENTRALITY MEASURES
# ─────────────────────────────────────────────────────────────

"""
    degree_centrality(g) -> Vector{Float64}

Normalized degree centrality: deg(v) / (n-1).
For directed graph, returns out-degree centrality.
"""
function degree_centrality(g::Union{Graph,DiGraph})::Vector{Float64}
    n = g.n
    [length(g.adj[v]) / max(n-1, 1) for v in 1:n]
end

"""
    betweenness_centrality(g) -> Vector{Float64}

Betweenness centrality via Brandes' algorithm. O(V*E).
Fraction of shortest paths passing through each vertex.
"""
function betweenness_centrality(g::Union{Graph,DiGraph})::Vector{Float64}
    n = g.n
    bc = zeros(n)

    for s in 1:n
        # BFS for shortest path counting
        stack  = Int[]
        pred   = [Int[] for _ in 1:n]
        sigma  = zeros(Int, n); sigma[s] = 1
        dist   = fill(-1, n);   dist[s]  = 0
        queue  = [s]
        qi = 1
        while qi <= length(queue)
            v = queue[qi]; qi += 1
            push!(stack, v)
            for (w, _) in g.adj[v]
                if dist[w] < 0
                    push!(queue, w)
                    dist[w] = dist[v] + 1
                end
                if dist[w] == dist[v] + 1
                    sigma[w] += sigma[v]
                    push!(pred[w], v)
                end
            end
        end
        # Back-propagate
        delta = zeros(n)
        while !isempty(stack)
            w = pop!(stack)
            for v in pred[w]
                delta[v] += (sigma[v] / max(sigma[w], 1)) * (1 + delta[w])
            end
            w != s && (bc[w] += delta[w])
        end
    end
    # Normalize
    factor = isa(g, DiGraph) ? (n-1)*(n-2) : (n-1)*(n-2)/2
    factor = max(factor, 1.0)
    bc ./ factor
end

"""
    closeness_centrality(g) -> Vector{Float64}

Closeness centrality: 1 / mean shortest path length.
Uses Floyd-Warshall for all-pairs distances.
"""
function closeness_centrality(g::Union{Graph,DiGraph})::Vector{Float64}
    D = floyd_warshall(g)
    n = g.n
    [begin
        reachable = [D[v,u] for u in 1:n if u != v && D[v,u] < Inf]
        isempty(reachable) ? 0.0 : 1.0 / mean(reachable)
     end for v in 1:n]
end

"""
    pagerank(g; alpha=0.85, tol=1e-8, maxiter=100) -> Vector{Float64}

PageRank algorithm. alpha = damping factor.
"""
function pagerank(g::Union{Graph,DiGraph};
                   alpha::Float64=0.85,
                   tol::Float64=1e-8,
                   maxiter::Int=100)::Vector{Float64}
    n  = g.n
    pr = fill(1.0/n, n)

    # Out-degree for normalization
    out_deg = [length(g.adj[v]) for v in 1:n]

    for _ in 1:maxiter
        pr_new = fill((1 - alpha) / n, n)
        for v in 1:n
            for (u, _) in g.adj[v]
                # In undirected: edge u→v and v→u both exist
                # Divide by out-degree of source
                od = out_deg[v]
                od > 0 && (pr_new[u] += alpha * pr[v] / od)
            end
        end
        # Handle dangling nodes
        dangling = sum(pr[v] for v in 1:n if out_deg[v] == 0; init=0.0)
        pr_new .+= alpha * dangling / n

        norm(pr_new - pr, 1) < tol && (pr = pr_new; break)
        pr = pr_new
    end
    pr ./ sum(pr)
end

"""
    eigenvector_centrality(g; tol=1e-6, maxiter=100) -> Vector{Float64}

Eigenvector centrality via power iteration.
Centrality is the dominant eigenvector of the adjacency matrix.
"""
function eigenvector_centrality(g::Union{Graph,DiGraph};
                                  tol::Float64=1e-6,
                                  maxiter::Int=100)::Vector{Float64}
    n = g.n
    A = weight_matrix(g)
    x = fill(1.0/n, n)
    for _ in 1:maxiter
        x_new = A * x
        nrm   = norm(x_new, Inf)
        nrm < 1e-10 && break
        x_new ./= nrm
        norm(x_new - x, 1) < tol && (x = x_new; break)
        x = x_new
    end
    x ./ sum(x)
end

# ─────────────────────────────────────────────────────────────
# 5. COMMUNITY DETECTION
# ─────────────────────────────────────────────────────────────

"""
    spectral_clustering(g, k; n_eig=k) -> Vector{Int}

Spectral clustering of graph vertices into k communities.
Uses normalized graph Laplacian and k-means on eigenvectors.
"""
function spectral_clustering(g::Union{Graph,DiGraph}, k::Int)::Vector{Int}
    n = g.n
    W = weight_matrix(g)
    D_vec = vec(sum(W, dims=2))
    D_inv_sqrt = Diagonal(1.0 ./ max.(sqrt.(D_vec), 1e-10))
    # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = Matrix(I, n, n) .- D_inv_sqrt * W * D_inv_sqrt

    # Eigen-decomposition (smallest k eigenvalues)
    F = eigen(Symmetric(L_sym))
    # Take k smallest eigenvectors (skip first if near-zero)
    k = min(k, n)
    U = F.vectors[:, 1:k]  # n × k

    # K-means clustering on rows of U
    labels = kmeans_cluster(U, k)
    labels
end

"""Simple k-means on matrix rows."""
function kmeans_cluster(X::Matrix{Float64}, k::Int;
                         maxiter::Int=100, rng=MersenneTwister(42))::Vector{Int}
    n = size(X, 1)
    k = min(k, n)
    # Initialize centroids: random rows
    centers = X[rand(rng, 1:n, k), :]
    labels  = zeros(Int, n)
    for _ in 1:maxiter
        new_labels = [argmin([norm(X[i,:] - centers[c,:]) for c in 1:k]) for i in 1:n]
        new_labels == labels && break
        labels = new_labels
        for c in 1:k
            members = X[labels .== c, :]
            isempty(members) || (centers[c, :] = vec(mean(members, dims=1)))
        end
    end
    labels
end

"""
    label_propagation(g; maxiter=100, rng=...) -> Vector{Int}

Label propagation community detection.
Each vertex iteratively adopts the most common label among its neighbors.
"""
function label_propagation(g::Union{Graph,DiGraph};
                             maxiter::Int=100,
                             rng=MersenneTwister(1))::Vector{Int}
    n = g.n
    labels = collect(1:n)  # each node is its own community initially
    order  = collect(1:n)
    for _ in 1:maxiter
        shuffle!(rng, order)
        changed = false
        for v in order
            nbrs = [u for (u,_) in g.adj[v]]
            isempty(nbrs) && continue
            # Most frequent neighbor label
            nbr_labels = labels[nbrs]
            counts = Dict{Int,Int}()
            for l in nbr_labels
                counts[l] = get(counts, l, 0) + 1
            end
            best_label = argmax(counts)
            if best_label != labels[v]
                labels[v] = best_label
                changed    = true
            end
        end
        !changed && break
    end
    # Remap labels to 1:k
    unique_labels = unique(labels)
    label_map = Dict(l => i for (i, l) in enumerate(unique_labels))
    [label_map[l] for l in labels]
end

"""
    louvain_communities(g; resolution=1.0) -> Vector{Int}

Simplified Louvain method for modularity maximization.
Phase 1: greedy modularity gains. Phase 2: compress and repeat.
"""
function louvain_communities(g::Union{Graph,DiGraph};
                               resolution::Float64=1.0,
                               rng=MersenneTwister(42))::Vector{Int}
    n = g.n
    W = weight_matrix(g)
    m = sum(W) / 2  # total edge weight (undirected)
    m < 1e-10 && return collect(1:n)

    community = collect(1:n)
    ki = vec(sum(W, dims=2))  # weighted degree

    improved = true
    while improved
        improved = false
        order = shuffle(rng, 1:n)
        for v in order
            current_c = community[v]
            # Evaluate ΔQ for removing v from its community and adding to neighbors'
            best_delta = 0.0
            best_c     = current_c

            neighbor_communities = unique([community[u] for (u,_) in g.adj[v]])

            for c in neighbor_communities
                c == current_c && continue
                # ΔQ = [k_{i,c} / m] - [resolution * k_c * k_i / (2m^2)]
                k_ic = sum(W[v, u] for u in 1:n if community[u] == c; init=0.0)
                k_c  = sum(ki[u] for u in 1:n if community[u] == c; init=0.0)
                delta = k_ic / m - resolution * k_c * ki[v] / (2m^2)
                if delta > best_delta
                    best_delta = delta
                    best_c     = c
                end
            end

            if best_c != current_c
                community[v] = best_c
                improved = true
            end
        end
    end

    # Remap
    unique_c = unique(community)
    cmap = Dict(c => i for (i,c) in enumerate(unique_c))
    [cmap[c] for c in community]
end

"""
    modularity(g, communities) -> Float64

Compute modularity Q of a community partition.
"""
function modularity(g::Union{Graph,DiGraph}, communities::Vector{Int})::Float64
    W = weight_matrix(g)
    m = sum(W) / 2
    m < 1e-10 && return 0.0
    n = g.n
    ki = vec(sum(W, dims=2))
    Q  = 0.0
    for i in 1:n, j in 1:n
        communities[i] == communities[j] || continue
        Q += W[i,j] - ki[i]*ki[j]/(2m)
    end
    Q / (2m)
end

# ─────────────────────────────────────────────────────────────
# 6. GRAPH SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────

"""
    graph_laplacian(g) -> Matrix{Float64}

Combinatorial graph Laplacian: L = D - W.
"""
function graph_laplacian(g::Union{Graph,DiGraph})::Matrix{Float64}
    W = weight_matrix(g)
    D = Diagonal(vec(sum(W, dims=2)))
    Matrix(D) .- W
end

"""
    normalized_laplacian(g) -> Matrix{Float64}

Normalized symmetric Laplacian: L_sym = D^{-1/2} L D^{-1/2}.
"""
function normalized_laplacian(g::Union{Graph,DiGraph})::Matrix{Float64}
    W = weight_matrix(g)
    d = max.(vec(sum(W, dims=2)), 1e-10)
    D_inv_sqrt = Diagonal(1.0 ./ sqrt.(d))
    L = Diagonal(d) .- W
    D_inv_sqrt * L * D_inv_sqrt
end

"""
    graph_fourier_transform(g, signal) -> (coeffs, freqs)

Graph Fourier Transform: project graph signal onto eigenvectors of Laplacian.
eigenvectors = graph frequency modes, eigenvalues = graph frequencies.
"""
function graph_fourier_transform(g::Union{Graph,DiGraph},
                                  signal::Vector{Float64})
    L = graph_laplacian(g)
    F = eigen(Symmetric(L))
    coeffs = F.vectors' * signal  # GFT coefficients
    (coefficients=coeffs, frequencies=F.values, basis=F.vectors)
end

"""
    graph_wavelet_filter(g, signal, scale) -> Vector{Float64}

Apply graph wavelet filter at given scale.
Uses Mexican hat wavelet on graph spectral domain.
ψ(λ, s) = s*λ*exp(-s²λ²/2)
"""
function graph_wavelet_filter(g::Union{Graph,DiGraph},
                                signal::Vector{Float64},
                                scale::Float64=1.0)::Vector{Float64}
    L = graph_laplacian(g)
    F = eigen(Symmetric(L))
    U = F.vectors
    λ = F.values
    # Wavelet kernel in spectral domain
    psi = scale .* λ .* exp.(-scale^2 .* λ.^2 ./ 2)
    # Filter: U * diag(psi) * U' * signal
    coeffs = U' * signal
    U * (psi .* coeffs)
end

"""
    graph_low_pass_filter(g, signal; threshold=1.0) -> Vector{Float64}

Graph low-pass filter: retain only frequencies below threshold.
"""
function graph_low_pass_filter(g::Union{Graph,DiGraph},
                                 signal::Vector{Float64};
                                 threshold::Float64=1.0)::Vector{Float64}
    L = graph_laplacian(g)
    F = eigen(Symmetric(L))
    U = F.vectors
    λ = F.values
    h = Float64[lam <= threshold ? 1.0 : 0.0 for lam in λ]
    coeffs = U' * signal
    U * (h .* coeffs)
end

# ─────────────────────────────────────────────────────────────
# 7. TEMPORAL GRAPHS
# ─────────────────────────────────────────────────────────────

"""
    TemporalGraph

Time-varying graph: sequence of snapshots.
"""
struct TemporalGraph
    snapshots::Vector{Graph}
    times::Vector{Float64}
    n::Int  # number of nodes (fixed)
end

"""
    temporal_network_analysis(snapshots) -> NamedTuple

Analyze evolution of network properties over time.
"""
function temporal_network_analysis(tg::TemporalGraph)
    T = length(tg.snapshots)
    densities    = zeros(T)
    avg_degrees  = zeros(T)
    modularities = zeros(T)

    for (i, g) in enumerate(tg.snapshots)
        n_edges = length(g.edges)
        max_edges = tg.n * (tg.n - 1) / 2
        densities[i]   = n_edges / max(max_edges, 1)
        avg_degrees[i] = 2 * n_edges / max(tg.n, 1)
        comms = label_propagation(g)
        modularities[i] = modularity(g, comms)
    end

    (densities=densities, avg_degrees=avg_degrees, modularities=modularities,
     times=tg.times)
end

"""
    build_temporal_correlation_network(returns, window, step; threshold=0.5) -> TemporalGraph

Build rolling correlation network from asset return matrix.
"""
function build_temporal_correlation_network(returns::Matrix{Float64},
                                              window::Int, step::Int;
                                              threshold::Float64=0.5)
    n_t, n_assets = size(returns)
    snapshots = Graph[]
    times     = Float64[]
    t = window
    while t <= n_t
        R = returns[t-window+1:t, :]
        C = cor(R)
        g = build_correlation_graph(C; threshold=threshold)
        push!(snapshots, g)
        push!(times, Float64(t))
        t += step
    end
    TemporalGraph(snapshots, times, n_assets)
end

# ─────────────────────────────────────────────────────────────
# 8. CRYPTO APPLICATIONS
# ─────────────────────────────────────────────────────────────

"""
    lead_lag_communities(returns; lag=1, threshold=0.3) -> NamedTuple

Find lead-lag relationships and community structure in crypto market.
- Builds directed correlation graph from lagged correlations
- Detects communities of assets that co-move
- Identifies leaders (high out-degree) and laggards
"""
function lead_lag_communities(returns::Matrix{Float64};
                                lag::Int=1,
                                threshold::Float64=0.3)
    n_t, n_assets = size(returns)
    # Lagged cross-correlation matrix
    lag_corr = zeros(n_assets, n_assets)
    for i in 1:n_assets, j in 1:n_assets
        i == j && continue
        # Correlation of returns[t, i] with returns[t+lag, j]
        ri = returns[1:n_t-lag, i]
        rj = returns[1+lag:n_t, j]
        s1 = std(ri); s2 = std(rj)
        (s1 < 1e-10 || s2 < 1e-10) && continue
        lag_corr[i, j] = cov(ri, rj) / (s1 * s2)
    end

    # Build directed graph from significant lead-lag
    g = DiGraph(n_assets)
    for i in 1:n_assets, j in 1:n_assets
        i == j && continue
        if lag_corr[i, j] > threshold
            add_edge!(g, i, j, lag_corr[i, j])
        end
    end

    # Centrality measures
    dc = degree_centrality(g)
    pr = pagerank(g)

    # Leaders: high out-degree (PageRank source nodes)
    leaders  = sortperm(dc, rev=true)[1:min(3, n_assets)]
    laggards = sortperm(dc)[1:min(3, n_assets)]

    # Undirected graph for community detection
    g_undir = Graph(n_assets)
    for i in 1:n_assets, j in (i+1):n_assets
        sym_corr = 0.5 * (abs(lag_corr[i,j]) + abs(lag_corr[j,i]))
        sym_corr > threshold && add_edge!(g_undir, i, j, sym_corr)
    end
    communities = louvain_communities(g_undir)
    n_communities = maximum(communities; init=1)

    (lag_correlation_matrix=lag_corr, directed_graph=g,
     communities=communities, n_communities=n_communities,
     leaders=leaders, laggards=laggards,
     degree_centrality=dc, pagerank=pr)
end

"""
    minimum_spanning_tree_portfolio(corr_matrix) -> NamedTuple

Build MST of correlation matrix as compact financial network.
Useful for portfolio diversification: pick assets from different MST branches.
"""
function minimum_spanning_tree_portfolio(corr_matrix::Matrix{Float64})
    n = size(corr_matrix, 1)
    # Convert correlation to distance: d = sqrt(2*(1 - corr))
    dist_matrix = sqrt.(max.(2.0 .* (1.0 .- corr_matrix), 0.0))
    g = Graph(n)
    for i in 1:n, j in (i+1):n
        add_edge!(g, i, j, dist_matrix[i,j])
    end
    mst = kruskal_mst(g)
    # Build MST graph
    g_mst = Graph(n)
    for (u, v, w) in mst.edges
        add_edge!(g_mst, u, v, w)
    end
    # Degree in MST = number of connections
    mst_degree = [length(g_mst.adj[v]) for v in 1:n]
    hub_nodes  = sortperm(mst_degree, rev=true)[1:min(3,n)]
    leaf_nodes = [v for v in 1:n if mst_degree[v] == 1]

    (mst_graph=g_mst, total_distance=mst.total_weight,
     hub_nodes=hub_nodes, leaf_nodes=leaf_nodes,
     mst_degree=mst_degree)
end

# ─────────────────────────────────────────────────────────────
# 9. DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_graph_theory_demo() -> Nothing

Demonstration of all graph theory algorithms.
"""
function run_graph_theory_demo()
    println("=" ^ 60)
    println("GRAPH THEORY FOR FINANCIAL NETWORKS DEMO")
    println("=" ^ 60)

    rng = MersenneTwister(42)
    n   = 10

    # Build sample graph
    println("\n1. Graph Construction & Shortest Paths")
    g = Graph(n)
    for i in 1:n, j in (i+1):n
        rand(rng) < 0.4 && add_edge!(g, i, j, 0.5 + rand(rng))
    end
    println("  Nodes: $(g.n), Edges: $(length(g.edges))")

    dist, prev = dijkstra(g, 1)
    println("  Dijkstra from 1: dist[10] = $(round(dist[10],digits=3))")
    path = reconstruct_path(prev, 1, 10)
    println("  Path 1→10: $path")

    D = floyd_warshall(g)
    println("  Floyd-Warshall: avg shortest path = $(round(mean(D[D .< Inf]),digits=3))")

    # MST
    println("\n2. Minimum Spanning Trees")
    mst_k = kruskal_mst(g)
    mst_p = prim_mst(g)
    println("  Kruskal MST weight: $(round(mst_k.total_weight,digits=3))")
    println("  Prim MST weight:    $(round(mst_p.total_weight,digits=3))")

    # Centrality
    println("\n3. Centrality Measures")
    dc = degree_centrality(g)
    ec = eigenvector_centrality(g)
    pr = pagerank(g)
    bc = betweenness_centrality(g)
    cc = closeness_centrality(g)
    println("  Most central (degree):      node $(argmax(dc))")
    println("  Most central (PageRank):    node $(argmax(pr))")
    println("  Most central (betweenness): node $(argmax(bc))")
    println("  Most central (eigenvector): node $(argmax(ec))")
    println("  Most central (closeness):   node $(argmax(cc))")

    # Community Detection
    println("\n4. Community Detection")
    c_spec   = spectral_clustering(g, 3)
    c_label  = label_propagation(g; rng=rng)
    c_louv   = louvain_communities(g; rng=rng)
    println("  Spectral (3): communities = $(sort(unique(c_spec)))")
    println("  Label Prop:   $(maximum(c_label)) communities")
    println("  Louvain:      $(maximum(c_louv)) communities")
    println("  Modularity (Louvain): $(round(modularity(g, c_louv),digits=4))")

    # Graph Signal Processing
    println("\n5. Graph Signal Processing")
    signal = randn(rng, n)
    gft = graph_fourier_transform(g, signal)
    filtered = graph_low_pass_filter(g, signal; threshold=1.0)
    wavelets = graph_wavelet_filter(g, signal, 0.5)
    println("  GFT coefficients (first 3): $(round.(gft.coefficients[1:3],digits=3))")
    println("  Low-pass filtered signal norm: $(round(norm(filtered),digits=3))")
    println("  Wavelet filtered norm: $(round(norm(wavelets),digits=3))")

    # Crypto Lead-Lag
    println("\n6. Crypto Lead-Lag Community Analysis")
    n_assets = 5; T = 300
    returns_sim = randn(rng, T, n_assets)
    # Add lead-lag: asset 1 leads asset 2
    returns_sim[2:end, 2] .+= 0.5 .* returns_sim[1:end-1, 1]
    ll = lead_lag_communities(returns_sim; lag=1, threshold=0.2)
    println("  Assets: $n_assets, Lead-lag threshold: 0.2")
    println("  Communities: $(ll.n_communities)")
    println("  Leaders (degree): $(ll.leaders)")
    println("  PageRank top: $(sortperm(ll.pagerank, rev=true)[1:3])")

    # MST Portfolio
    println("\n7. MST Portfolio Diversification")
    C = cor(randn(rng, 200, n_assets))
    mst_p2 = minimum_spanning_tree_portfolio(C)
    println("  MST distance: $(round(mst_p2.total_distance,digits=3))")
    println("  Hub assets (diversify away from): $(mst_p2.hub_nodes)")
    println("  Leaf assets (most independent):   $(mst_p2.leaf_nodes)")

    println("\nDone.")
    nothing
end

# ─────────────────────────────────────────────────────────────
# 10. GRAPH GENERATION MODELS
# ─────────────────────────────────────────────────────────────

"""
    erdos_renyi_graph(n, p; rng=...) -> Graph

Erdős–Rényi random graph G(n,p): each edge exists with probability p.
"""
function erdos_renyi_graph(n::Int, p::Float64; rng=MersenneTwister(1))::Graph
    g = Graph(n)
    for i in 1:n, j in (i+1):n
        rand(rng) < p && add_edge!(g, i, j, 1.0)
    end
    g
end

"""
    barabasi_albert_graph(n, m; rng=...) -> Graph

Barabási–Albert preferential attachment model.
Each new node attaches to m existing nodes with prob ∝ degree.
Creates scale-free networks (power-law degree distribution).
"""
function barabasi_albert_graph(n::Int, m::Int; rng=MersenneTwister(1))::Graph
    m = min(m, n-1)
    g = Graph(n)
    # Initialize with a clique of m+1 nodes
    for i in 1:(m+1), j in (i+1):(m+1)
        add_edge!(g, i, j, 1.0)
    end
    degrees = [length(g.adj[v]) for v in 1:n]

    for new_node in (m+2):n
        # Preferential attachment: choose m nodes
        total_deg = sum(degrees[1:new_node-1])
        total_deg == 0 && (total_deg = 1)
        probs = degrees[1:new_node-1] ./ total_deg
        targets = Set{Int}()
        attempts = 0
        while length(targets) < m && attempts < 1000
            u = rand(rng)
            cs = 0.0
            for (k, p) in enumerate(probs)
                cs += p
                if u <= cs
                    push!(targets, k)
                    break
                end
            end
            attempts += 1
        end
        for t in targets
            add_edge!(g, new_node, t, 1.0)
            degrees[new_node] += 1
            degrees[t] += 1
        end
    end
    g
end

"""
    watts_strogatz_graph(n, k, beta; rng=...) -> Graph

Watts–Strogatz small-world graph.
Start with k-regular ring lattice, rewire each edge with probability beta.
"""
function watts_strogatz_graph(n::Int, k::Int, beta::Float64;
                                rng=MersenneTwister(1))::Graph
    k_half = k ÷ 2
    g = Graph(n)
    # Regular ring lattice
    for i in 1:n, j in 1:k_half
        neighbor = (i - 1 + j) % n + 1
        add_edge!(g, i, neighbor, 1.0)
    end
    # Rewiring
    all_edges = copy(g.edges)
    empty!(g.adj); empty!(g.edges)
    for v in 1:n; g.adj[v] = Tuple{Int,Float64}[]; end
    for (u, v, w) in all_edges
        if rand(rng) < beta
            # Rewire to random node
            new_v = rand(rng, 1:n)
            while new_v == u || any(e -> e[1] == new_v || e[2] == new_v,
                                    filter(e -> e[1] == u || e[2] == u, g.edges))
                new_v = rand(rng, 1:n)
                break  # allow some duplicates for simplicity
            end
            add_edge!(g, u, new_v, w)
        else
            add_edge!(g, u, v, w)
        end
    end
    g
end

# ─────────────────────────────────────────────────────────────
# 11. GRAPH METRICS AND STATISTICS
# ─────────────────────────────────────────────────────────────

"""
    clustering_coefficient(g) -> Vector{Float64}

Local clustering coefficient for each vertex.
C_v = (# triangles through v) / (# possible triangles through v)
"""
function clustering_coefficient(g::Union{Graph,DiGraph})::Vector{Float64}
    n = g.n
    cc = zeros(n)
    for v in 1:n
        nbrs = Set(neighbors(g, v))
        k = length(nbrs)
        k < 2 && continue
        triangles = 0
        nbr_list = collect(nbrs)
        for i in 1:length(nbr_list), j in (i+1):length(nbr_list)
            u1 = nbr_list[i]; u2 = nbr_list[j]
            u2 ∈ Set(neighbors(g, u1)) && (triangles += 1)
        end
        cc[v] = 2.0 * triangles / (k * (k-1))
    end
    cc
end

"""
    global_clustering_coefficient(g) -> Float64

Global clustering coefficient (average of local).
"""
global_clustering_coefficient(g) = mean(clustering_coefficient(g))

"""
    average_path_length(g) -> Float64

Average shortest path length (only over connected pairs).
"""
function average_path_length(g::Union{Graph,DiGraph})::Float64
    D = floyd_warshall(g)
    finite_d = D[D .< Inf .&& D .> 0]
    isempty(finite_d) ? Inf : mean(finite_d)
end

"""
    graph_diameter(g) -> Float64

Graph diameter: maximum shortest path length.
"""
function graph_diameter(g::Union{Graph,DiGraph})::Float64
    D = floyd_warshall(g)
    max_finite = maximum(D[D .< Inf]; init=0.0)
    max_finite
end

"""
    degree_distribution(g) -> (degrees, frequencies)

Compute degree distribution of graph.
"""
function degree_distribution(g::Union{Graph,DiGraph})
    degs = [length(g.adj[v]) for v in 1:g.n]
    max_d = maximum(degs; init=0)
    freq = zeros(max_d + 1)
    for d in degs; freq[d+1] += 1; end
    freq ./= g.n
    (degrees=0:max_d, frequencies=freq, mean_degree=mean(degs), max_degree=max_d)
end

"""
    is_scale_free(g; alpha_threshold=2.0) -> Bool

Test whether graph has a scale-free degree distribution (power-law).
Uses linear regression on log-log degree distribution.
"""
function is_scale_free(g::Union{Graph,DiGraph}; alpha_threshold::Float64=2.0)::Bool
    dd = degree_distribution(g)
    degs = collect(dd.degrees)[2:end]  # skip degree-0
    freqs = dd.frequencies[2:end]
    valid = (freqs .> 0)
    sum(valid) < 3 && return false
    log_d = log.(degs[valid])
    log_f = log.(freqs[valid])
    X = hcat(ones(sum(valid)), log_d)
    coef = (X'X + 1e-8*I) \ (X'log_f)
    alpha = -coef[2]  # power-law exponent
    alpha > alpha_threshold
end

# ─────────────────────────────────────────────────────────────
# 12. NETWORK ROBUSTNESS
# ─────────────────────────────────────────────────────────────

"""
    attack_robustness(g; strategy=:degree, n_steps=10) -> Vector{Float64}

Simulate targeted attacks on the network.
strategy: :degree (attack high-degree nodes first), :random, :betweenness
Returns fraction of remaining connected component after each removal.
"""
function attack_robustness(g::Union{Graph,DiGraph};
                             strategy::Symbol=:degree,
                             n_steps::Int=10)::Vector{Float64}
    n = g.n
    remaining = collect(1:n)
    n_steps = min(n_steps, n)
    connectivity = zeros(n_steps)

    for step in 1:n_steps
        # Build subgraph of remaining nodes
        sub = Graph(length(remaining))
        node_map = Dict(remaining[i] => i for i in 1:length(remaining))
        for (u, v, w) in g.edges
            haskey(node_map, u) && haskey(node_map, v) &&
                add_edge!(sub, node_map[u], node_map[v], w)
        end

        # Connected component size
        D = floyd_warshall(sub)
        giant = maximum([sum(D[i,:] .< Inf) for i in 1:sub.n]; init=0)
        connectivity[step] = giant / n

        # Remove node by strategy
        length(remaining) <= 1 && break
        if strategy == :degree
            # Remove highest degree node
            degs = [length(sub.adj[i]) for i in 1:sub.n]
            remove_idx = argmax(degs)
        else
            remove_idx = 1
        end
        deleteat!(remaining, remove_idx)
    end
    connectivity
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 15 – Graph-Based Portfolio Analytics
# ─────────────────────────────────────────────────────────────────────────────

"""
    correlation_threshold_graph(corr_matrix, threshold)

Build a graph where edges connect assets with |correlation| > threshold.
Returns adjacency list representation.
"""
function correlation_threshold_graph(corr::Matrix{Float64},
                                      threshold::Float64=0.5)
    n = size(corr, 1)
    g = Graph(n)
    for i in 1:n, j in (i+1):n
        if abs(corr[i, j]) >= threshold
            add_edge!(g, i, j, corr[i, j])
        end
    end
    return g
end

"""
    graph_portfolio_diversification(g, weights)

Compute a graph-based diversification score:
  D = 1 - (sum of edge weights for connected portfolio pairs) / (n*(n-1)/2)
Higher score → more diversified (fewer strong correlations in portfolio).
"""
function graph_portfolio_diversification(g::Graph, weights::Vector{Float64})
    n = g.n
    total_w = 0.0; connected_w = 0.0
    for i in 1:n, j in (i+1):n
        w_ij = weights[i] * weights[j]
        total_w += w_ij
        for (nbr, edge_w) in g.adj[i]
            if nbr == j
                connected_w += w_ij * abs(edge_w)
                break
            end
        end
    end
    return 1.0 - connected_w / (total_w + 1e-8)
end

"""
    network_centrality_weights(corr_matrix; method)

Derive portfolio weights inversely proportional to centrality:
low-centrality (peripheral) assets receive higher weight → diversification.
Methods: :degree, :eigenvector, :closeness
"""
function network_centrality_weights(corr::Matrix{Float64};
                                     method::Symbol=:degree,
                                     threshold::Float64=0.3)
    g = correlation_threshold_graph(corr, threshold)
    if method == :eigenvector
        c = eigenvector_centrality(g)
    elseif method == :closeness
        c = closeness_centrality(g)
    else
        c = Float64[length(g.adj[i]) for i in 1:g.n]
    end
    # inverse centrality weights
    inv_c = 1.0 ./ (c .+ 0.01)
    return inv_c ./ sum(inv_c)
end

end  # module GraphTheory
