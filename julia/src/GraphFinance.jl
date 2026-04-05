"""
GraphFinance.jl
================
Financial graph theory module for correlation/network analysis.

Exports:
  FinancialGraph           — struct with adjacency, weights, node metadata
  minimum_spanning_tree    — Prim's algorithm MST
  maximum_spanning_tree    — Max spanning tree
  spectral_clustering      — Normalized Laplacian, k eigenvectors
  label_propagation        — Semi-supervised community detection
  pagerank                 — Power iteration PageRank
  graph_fourier_transform  — Via Laplacian eigenvectors
  temporal_graph_analysis  — How community structure evolves
  contagion_simulation     — Network spread model
"""
module GraphFinance

using Statistics, LinearAlgebra, Random

export FinancialGraph, minimum_spanning_tree, maximum_spanning_tree,
       spectral_clustering, label_propagation, pagerank,
       graph_fourier_transform, temporal_graph_analysis, contagion_simulation,
       build_correlation_graph, build_partial_correlation_graph,
       planar_maximally_filtered_graph, graph_metrics,
       community_modularity, louvain_step, centrality_measures

# ─────────────────────────────────────────────────────────────────────────────
# 1. FINANCIAL GRAPH STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

"""
    FinancialGraph{T}

Undirected weighted graph for financial assets.
Fields:
  n         — number of nodes
  adj       — adjacency matrix (n × n), zeros on diagonal
  weights   — weight matrix (n × n)
  node_names — asset names or identifiers
  metadata  — Dict for additional node properties
"""
mutable struct FinancialGraph{T}
    n           ::Int
    adj         ::Matrix{Float64}
    weights     ::Matrix{Float64}
    node_names  ::Vector{T}
    metadata    ::Dict{Symbol, Any}
end

function FinancialGraph(n::Int; names::Vector=["Node_$i" for i in 1:n])
    FinancialGraph{eltype(names)}(n, zeros(n,n), zeros(n,n), names,
                                   Dict{Symbol,Any}())
end

function FinancialGraph(adj::Matrix{Float64};
                         names::Vector=["Node_$i" for i in 1:size(adj,1)])
    n = size(adj, 1)
    FinancialGraph{eltype(names)}(n, adj, adj, names, Dict{Symbol,Any}())
end

function Base.show(io::IO, G::FinancialGraph)
    n_edges = sum(G.adj .> 0) ÷ 2
    println(io, "FinancialGraph: $(G.n) nodes, $n_edges edges")
    println(io, "  Nodes: $(join(G.node_names[1:min(5,G.n)], ", "))",
            G.n > 5 ? "..." : "")
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. GRAPH CONSTRUCTION FROM RETURNS
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_correlation_graph(returns, threshold) → FinancialGraph

Build graph where edge weight = |correlation|.
Only connect edges where |ρ| > threshold.
"""
function build_correlation_graph(returns::Matrix{Float64};
                                  threshold::Float64=0.3,
                                  names::Vector=["Asset_$i" for i in 1:size(returns,2)])
    n   = size(returns, 2)
    C   = cor(returns)
    adj = zeros(n, n)
    wgt = zeros(n, n)

    for i in 1:n, j in i+1:n
        rho = abs(C[i,j])
        if rho > threshold
            adj[i,j] = adj[j,i] = 1.0
            wgt[i,j] = wgt[j,i] = rho
        end
    end

    G = FinancialGraph{String}(n, adj, wgt, names, Dict{Symbol,Any}())
    G.metadata[:correlation_matrix] = C
    G.metadata[:threshold] = threshold
    return G
end

"""
    build_partial_correlation_graph(returns, threshold) → FinancialGraph

Partial correlation graph: condition on all other assets.
Useful for identifying direct connections vs spurious correlations.
"""
function build_partial_correlation_graph(returns::Matrix{Float64};
                                          threshold::Float64=0.2,
                                          names::Vector=["Asset_$i" for i in 1:size(returns,2)])
    n    = size(returns, 2)
    C    = cov(returns) + 1e-8 * I(n)
    Cinv = inv(C)

    # Partial correlation from precision matrix
    pc_mat = zeros(n, n)
    for i in 1:n, j in 1:n
        pc_mat[i,j] = -Cinv[i,j] / sqrt(abs(Cinv[i,i] * Cinv[j,j]) + 1e-10)
    end
    for i in 1:n; pc_mat[i,i] = 1.0; end

    adj = zeros(n, n)
    wgt = zeros(n, n)
    for i in 1:n, j in i+1:n
        if abs(pc_mat[i,j]) > threshold
            adj[i,j] = adj[j,i] = 1.0
            wgt[i,j] = wgt[j,i] = abs(pc_mat[i,j])
        end
    end

    G = FinancialGraph{String}(n, adj, wgt, names, Dict{Symbol,Any}())
    G.metadata[:partial_correlation] = pc_mat
    return G
end

"""
    planar_maximally_filtered_graph(C) → FinancialGraph

PMFG: removes edges to maintain planarity, keeping most important correlations.
Simplified: keeps the top 3*(n-2) edges (Euler formula for planar graphs).
"""
function planar_maximally_filtered_graph(returns::Matrix{Float64};
                                          names::Vector=["Asset_$i" for i in 1:size(returns,2)])
    n       = size(returns, 2)
    C       = cor(returns)
    n_edges = 3 * (n - 2)    # planar graph edge limit

    # Sort all edges by |correlation| descending
    edges = [(abs(C[i,j]), i, j) for i in 1:n for j in i+1:n]
    sort!(edges, rev=true)

    adj = zeros(n, n)
    wgt = zeros(n, n)
    count = 0
    for (c, i, j) in edges
        count >= n_edges && break
        adj[i,j] = adj[j,i] = 1.0
        wgt[i,j] = wgt[j,i] = c
        count += 1
    end

    return FinancialGraph{String}(n, adj, wgt, names, Dict{Symbol,Any}())
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. MINIMUM / MAXIMUM SPANNING TREE (PRIM'S ALGORITHM)
# ─────────────────────────────────────────────────────────────────────────────

"""
    minimum_spanning_tree(G) → FinancialGraph

Prim's algorithm for minimum spanning tree.
For correlation networks, MST keeps the essential connections.
"""
function minimum_spanning_tree(G::FinancialGraph)
    n       = G.n
    in_tree = falses(n)
    mst_adj = zeros(n, n)
    mst_wgt = zeros(n, n)

    in_tree[1] = true
    for _ in 1:n-1
        best_w = Inf
        best_i = best_j = -1
        for i in 1:n
            in_tree[i] || continue
            for j in 1:n
                in_tree[j] && continue
                G.adj[i,j] == 0 && continue
                w = G.weights[i,j] == 0.0 ? G.adj[i,j] : G.weights[i,j]
                if w < best_w
                    best_w = w; best_i = i; best_j = j
                end
            end
        end
        best_i == -1 && break
        in_tree[best_j] = true
        mst_adj[best_i, best_j] = mst_adj[best_j, best_i] = 1.0
        mst_wgt[best_i, best_j] = mst_wgt[best_j, best_i] = best_w
    end

    mst = FinancialGraph{eltype(G.node_names)}(n, mst_adj, mst_wgt,
                                                copy(G.node_names),
                                                Dict{Symbol,Any}())
    mst.metadata[:type] = "MST"
    return mst
end

"""
    maximum_spanning_tree(G) → FinancialGraph

Maximum spanning tree via Prim's on negated weights.
For correlation graphs: keeps the strongest correlations.
"""
function maximum_spanning_tree(G::FinancialGraph)
    n       = G.n
    in_tree = falses(n)
    mst_adj = zeros(n, n)
    mst_wgt = zeros(n, n)

    in_tree[1] = true
    for _ in 1:n-1
        best_w = -Inf
        best_i = best_j = -1
        for i in 1:n
            in_tree[i] || continue
            for j in 1:n
                in_tree[j] && continue
                G.adj[i,j] == 0 && continue
                w = G.weights[i,j] == 0.0 ? G.adj[i,j] : G.weights[i,j]
                if w > best_w
                    best_w = w; best_i = i; best_j = j
                end
            end
        end
        best_i == -1 && break
        in_tree[best_j] = true
        mst_adj[best_i, best_j] = mst_adj[best_j, best_i] = 1.0
        mst_wgt[best_i, best_j] = mst_wgt[best_j, best_i] = best_w
    end

    max_st = FinancialGraph{eltype(G.node_names)}(n, mst_adj, mst_wgt,
                                                   copy(G.node_names),
                                                   Dict{Symbol,Any}())
    max_st.metadata[:type] = "MaxST"
    return max_st
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPECTRAL CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

"""
    spectral_clustering(G, k) → Vector{Int}

Normalized spectral clustering using the graph Laplacian.
Returns cluster assignment for each node.
"""
function spectral_clustering(G::FinancialGraph, k::Int;
                               rng::AbstractRNG=MersenneTwister(42),
                               max_iter::Int=100)
    n  = G.n
    W  = G.weights
    # Degree matrix
    d  = vec(sum(W, dims=2))
    D  = Diagonal(d)

    # Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    d_sqrt_inv = 1.0 ./ sqrt.(max.(d, 1e-10))
    D_inv_sqrt = Diagonal(d_sqrt_inv)
    L_sym      = I(n) - D_inv_sqrt * W * D_inv_sqrt

    L_sym = (L_sym + L_sym') / 2 + 1e-8 * I(n)

    # Eigen-decomposition: smallest k eigenvectors
    eig       = eigen(Symmetric(L_sym))
    U         = eig.vectors[:, 1:k]   # n × k embedding

    # Normalize rows
    row_norms = sqrt.(sum(U.^2, dims=2)) .+ 1e-10
    U_norm    = U ./ row_norms

    # k-means on embedding
    labels = _kmeans(U_norm, k; rng=rng, max_iter=max_iter)
    return labels
end

"""
    _kmeans(X, k) → Vector{Int}

Simple k-means clustering on matrix X (n × d).
"""
function _kmeans(X::Matrix{Float64}, k::Int;
                  rng::AbstractRNG=MersenneTwister(42),
                  max_iter::Int=100, tol::Float64=1e-4)
    n, d = size(X)
    k    = min(k, n)

    # Random initialization
    centers = X[randperm(rng, n)[1:k], :]
    labels  = zeros(Int, n)

    for _ in 1:max_iter
        # Assignment step
        new_labels = [argmin([sum((X[i,:] .- centers[j,:]).^2) for j in 1:k])
                      for i in 1:n]

        # Update step
        new_centers = zeros(k, d)
        counts      = zeros(Int, k)
        for (i, label) in enumerate(new_labels)
            new_centers[label, :] .+= X[i, :]
            counts[label] += 1
        end
        for j in 1:k
            if counts[j] > 0
                new_centers[j, :] ./= counts[j]
            else
                new_centers[j, :] = X[rand(rng, 1:n), :]
            end
        end

        delta = maximum(abs.(new_centers .- centers))
        labels  = new_labels
        centers = new_centers
        delta < tol && break
    end

    return labels
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. LABEL PROPAGATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    label_propagation(G, seeds, max_iter) → Vector{Int}

Semi-supervised community detection via label propagation.
`seeds`: Dict{Int, Int} mapping node index → initial label.
Unlabeled nodes have label 0.
Returns final label for each node.
"""
function label_propagation(G::FinancialGraph, seeds::Dict{Int,Int};
                             max_iter::Int=50, rng::AbstractRNG=MersenneTwister(42))
    n      = G.n
    labels = zeros(Int, n)

    # Initialize seeds
    for (node, label) in seeds
        labels[node] = label
    end
    # Initialize unlabeled with random labels from seeds
    seed_labels = collect(values(seeds))
    isempty(seed_labels) && return labels

    for i in 1:n
        labels[i] == 0 && (labels[i] = rand(rng, seed_labels))
    end

    for _ in 1:max_iter
        changed = false
        order   = randperm(rng, n)

        for i in order
            haskey(seeds, i) && continue  # keep seed labels fixed

            # Count neighbor labels weighted by edge weight
            label_counts = Dict{Int, Float64}()
            for j in 1:n
                G.adj[i,j] == 0 && continue
                w = G.weights[i,j] > 0 ? G.weights[i,j] : 1.0
                lj = labels[j]
                label_counts[lj] = get(label_counts, lj, 0.0) + w
            end

            if !isempty(label_counts)
                best_label = argmax(label_counts)
                if best_label != labels[i]
                    labels[i] = best_label
                    changed    = true
                end
            end
        end

        changed || break
    end

    return labels
end

"""
    label_propagation_unsupervised(G, max_iter) → Vector{Int}

Unsupervised label propagation: initialize each node with unique label,
propagate to convergence (community detection).
"""
function label_propagation_unsupervised(G::FinancialGraph;
                                         max_iter::Int=100,
                                         rng::AbstractRNG=MersenneTwister(42))
    n      = G.n
    labels = collect(1:n)  # each node starts with unique label

    for _ in 1:max_iter
        changed = false
        order   = randperm(rng, n)

        for i in order
            label_counts = Dict{Int, Float64}()
            label_counts[labels[i]] = 0.0  # include self

            for j in 1:n
                G.adj[i,j] == 0 && continue
                w  = G.weights[i,j] > 0 ? G.weights[i,j] : 1.0
                lj = labels[j]
                label_counts[lj] = get(label_counts, lj, 0.0) + w
            end

            best_label = argmax(label_counts)
            if best_label != labels[i]
                labels[i] = best_label
                changed    = true
            end
        end

        changed || break
    end

    # Relabel to consecutive integers
    unique_labels = unique(labels)
    label_map     = Dict(l => i for (i,l) in enumerate(unique_labels))
    return [label_map[l] for l in labels]
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. PAGERANK
# ─────────────────────────────────────────────────────────────────────────────

"""
    pagerank(G, d; max_iter, tol) → Vector{Float64}

Power iteration PageRank.
`d`: damping factor (default 0.85).
Returns PageRank score for each node.
"""
function pagerank(G::FinancialGraph, d::Float64=0.85;
                   max_iter::Int=100, tol::Float64=1e-8,
                   use_weights::Bool=true)
    n   = G.n
    W   = use_weights ? G.weights : G.adj
    # Column-stochastic transition matrix
    out_strength = vec(sum(W, dims=2)) .+ 1e-10
    T = (W ./ out_strength)'  # n × n column stochastic

    pr  = fill(1.0/n, n)
    for _ in 1:max_iter
        pr_new = (1-d)/n .+ d .* (T * pr)
        delta  = maximum(abs.(pr_new .- pr))
        pr     = pr_new
        delta < tol && break
    end

    return pr ./ sum(pr)  # normalize
end

"""
    hub_authority(G; max_iter, tol) → (hubs, authorities)

HITS algorithm: hub and authority scores.
"""
function hub_authority(G::FinancialGraph; max_iter::Int=50, tol::Float64=1e-8)
    n   = G.n
    A   = G.weights
    h   = ones(n)
    a   = ones(n)

    for _ in 1:max_iter
        a_new = A' * h;  a_new ./= max(norm(a_new), 1e-10)
        h_new = A  * a;  h_new ./= max(norm(h_new), 1e-10)
        delta = max(norm(a_new - a), norm(h_new - h))
        a = a_new; h = h_new
        delta < tol && break
    end

    return (hubs=h ./ sum(h), authorities=a ./ sum(a))
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAPH FOURIER TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

"""
    graph_fourier_transform(G, signal) → (freqs, spectrum, eigenvecs)

Graph Fourier Transform via Laplacian eigenvectors.
`signal`: node-level signal (length n).
`freqs`: Laplacian eigenvalues (graph frequencies).
`spectrum`: GFT coefficients.
"""
function graph_fourier_transform(G::FinancialGraph, signal::Vector{Float64})
    length(signal) == G.n || error("Signal length must equal number of nodes")
    n  = G.n
    W  = G.weights
    d  = vec(sum(W, dims=2))
    L  = Diagonal(d) - W    # unnormalized Laplacian

    # Eigendecomposition
    L_sym = (L + L') / 2 + 1e-8 * I(n)
    eig   = eigen(Symmetric(L_sym))
    U     = eig.vectors      # columns are eigenvectors
    lambda = eig.values       # eigenvalues = graph frequencies

    # GFT: project signal onto eigenvectors
    spectrum = U' * signal

    return (frequencies=lambda, spectrum=spectrum, eigenvectors=U)
end

"""
    inverse_gft(U, spectrum) → Vector{Float64}

Inverse Graph Fourier Transform: reconstruct signal from spectrum.
"""
function inverse_gft(U::Matrix{Float64}, spectrum::Vector{Float64})
    return U * spectrum
end

"""
    graph_filter(G, signal, cutoff_k) → Vector{Float64}

Low-pass graph filter: keep only the k lowest-frequency components.
`cutoff_k`: number of frequency components to retain.
"""
function graph_filter(G::FinancialGraph, signal::Vector{Float64}, cutoff_k::Int)
    gft = graph_fourier_transform(G, signal)
    filtered_spectrum = copy(gft.spectrum)
    filtered_spectrum[cutoff_k+1:end] .= 0.0
    return inverse_gft(gft.eigenvectors, filtered_spectrum)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. TEMPORAL GRAPH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

"""
    temporal_graph_analysis(Gs, timestamps) → NamedTuple

Analyze how community structure and graph properties evolve over time.
`Gs`: Vector of FinancialGraph (one per time window)
`timestamps`: corresponding timestamps or labels
"""
function temporal_graph_analysis(Gs::Vector{<:FinancialGraph},
                                  timestamps::Vector)
    n_periods = length(Gs)
    n_periods == length(timestamps) || error("Length mismatch")

    # Metrics per period
    density        = Float64[]
    avg_clustering = Float64[]
    n_communities  = Int[]
    avg_weight     = Float64[]
    pageranks      = Vector{Vector{Float64}}()

    for G in Gs
        push!(density,  graph_density(G))
        push!(avg_clustering, mean_clustering_coefficient(G))

        # Community count via label propagation
        labels = label_propagation_unsupervised(G)
        push!(n_communities, length(unique(labels)))

        valid_w = G.weights[G.weights .> 0]
        push!(avg_weight, isempty(valid_w) ? 0.0 : mean(valid_w))

        push!(pageranks, pagerank(G))
    end

    # Community stability: how much do communities change?
    stability = Float64[]
    for t in 2:n_periods
        labels_prev = label_propagation_unsupervised(Gs[t-1])
        labels_curr = label_propagation_unsupervised(Gs[t])
        # Normalized mutual information proxy: fraction of same-cluster pairs
        n = length(labels_prev)
        same_pairs = sum([labels_prev[i] == labels_prev[j] && labels_curr[i] == labels_curr[j]
                          for i in 1:n for j in i+1:n])
        total_pairs = n * (n-1) ÷ 2
        push!(stability, total_pairs > 0 ? same_pairs / total_pairs : 0.0)
    end

    # PageRank stability: correlation of PR vectors between periods
    pr_stability = Float64[]
    for t in 2:n_periods
        if length(pageranks[t-1]) == length(pageranks[t])
            push!(pr_stability, cor(pageranks[t-1], pageranks[t]))
        end
    end

    return (
        timestamps        = timestamps,
        density           = density,
        avg_clustering    = avg_clustering,
        n_communities     = n_communities,
        avg_weight        = avg_weight,
        community_stability = stability,
        pagerank_stability  = pr_stability,
        pageranks           = pageranks,
    )
end

function graph_density(G::FinancialGraph)
    n_edges    = sum(G.adj) / 2
    max_edges  = G.n * (G.n - 1) / 2
    return max_edges > 0 ? n_edges / max_edges : 0.0
end

function clustering_coefficient(G::FinancialGraph, i::Int)
    neighbors = findall(G.adj[i, :] .> 0)
    k = length(neighbors)
    k < 2 && return 0.0
    # Count edges among neighbors
    n_tri = 0
    for a in neighbors, b in neighbors
        a == b && continue
        G.adj[a, b] > 0 && (n_tri += 1)
    end
    return n_tri / (k * (k-1))
end

function mean_clustering_coefficient(G::FinancialGraph)
    n = G.n
    cs = [clustering_coefficient(G, i) for i in 1:n]
    return mean(cs)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. CONTAGION SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    contagion_simulation(G, initial_shock, alpha; n_steps) → NamedTuple

Simulate financial contagion spreading through a network.
`initial_shock`: Dict{Int, Float64} mapping node → initial loss fraction (0-1)
`alpha`: contagion transmission parameter
`n_steps`: max simulation steps

Returns evolution of distress levels, defaults, and cascade timing.
"""
function contagion_simulation(G::FinancialGraph,
                               initial_shock::Dict{Int, Float64},
                               alpha::Float64=0.2;
                               n_steps::Int=10,
                               default_threshold::Float64=0.8)
    n        = G.n
    distress = zeros(n)
    defaulted = falses(n)

    # Apply initial shocks
    for (node, loss) in initial_shock
        distress[node] = clamp(loss, 0.0, 1.0)
        distress[node] >= default_threshold && (defaulted[node] = true)
    end

    history     = [copy(distress)]
    default_hist = [copy(defaulted)]
    new_defaults_per_step = Int[]

    for step in 1:n_steps
        new_distress = copy(distress)
        new_def      = 0

        for i in 1:n
            defaulted[i] && continue   # already defaulted

            # Receive contagion from neighbors
            contagion_in = 0.0
            for j in 1:n
                G.adj[i,j] == 0 && continue
                # Weight by edge weight and distress level of j
                w = G.weights[i,j] > 0 ? G.weights[i,j] : 1.0
                contagion_in += alpha * w * distress[j]
            end

            new_distress[i] = min(distress[i] + contagion_in, 1.0)

            if new_distress[i] >= default_threshold && !defaulted[i]
                defaulted[i] = true
                new_def += 1
            end
        end

        push!(new_defaults_per_step, new_def)
        distress = new_distress
        push!(history, copy(distress))
        push!(default_hist, copy(defaulted))

        new_def == 0 && all(distress .≈ history[end-1]) && break
    end

    # System-level metrics
    total_defaulted = sum(defaulted)
    system_stress   = mean(distress)
    cascade_depth   = findfirst(new_defaults_per_step .== 0)

    return (
        distress_history   = history,
        default_history    = default_hist,
        final_distress     = distress,
        defaulted          = defaulted,
        total_defaulted    = total_defaulted,
        system_stress      = system_stress,
        cascade_depth      = cascade_depth,
        new_defaults_per_step = new_defaults_per_step,
    )
end

"""
    contagion_analysis(G, alpha_range) → Vector{NamedTuple}

Analyze contagion severity across a range of transmission parameters.
"""
function contagion_analysis(G::FinancialGraph, alpha_range::AbstractRange;
                             shock_node::Int=1, shock_level::Float64=1.0)
    results = NamedTuple[]
    for alpha in alpha_range
        shock = Dict(shock_node => shock_level)
        sim   = contagion_simulation(G, shock, alpha)
        push!(results, (
            alpha           = alpha,
            total_defaulted = sim.total_defaulted,
            system_stress   = sim.system_stress,
            cascade_depth   = sim.cascade_depth,
        ))
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. GRAPH METRICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    graph_metrics(G) → NamedTuple

Compute comprehensive graph metrics.
"""
function graph_metrics(G::FinancialGraph)
    n        = G.n
    degrees  = vec(sum(G.adj, dims=2))
    strengths = vec(sum(G.weights, dims=2))
    pr       = pagerank(G)
    cc       = [clustering_coefficient(G, i) for i in 1:n]

    return (
        n_nodes          = n,
        n_edges          = Int(sum(G.adj) ÷ 2),
        density          = graph_density(G),
        mean_degree      = mean(degrees),
        max_degree       = maximum(degrees),
        degree_std       = std(degrees),
        mean_strength    = mean(strengths),
        mean_clustering  = mean(cc),
        max_clustering   = maximum(cc),
        pagerank         = pr,
        max_pagerank     = maximum(pr),
        pr_entropy       = -sum(pr .* log.(pr .+ 1e-10)),   # diversity of centrality
    )
end

"""
    centrality_measures(G) → NamedTuple

Compute multiple centrality measures for all nodes.
"""
function centrality_measures(G::FinancialGraph)
    n  = G.n
    # Degree centrality
    degree    = vec(sum(G.adj, dims=2)) ./ (n - 1)
    # Strength centrality
    strength  = vec(sum(G.weights, dims=2)) ./ (n - 1)
    # PageRank
    pr        = pagerank(G)
    # Eigenvector centrality
    eig_cent  = _eigenvector_centrality(G)
    # Hub/authority scores
    hubs, auth = hub_authority(G)
    # Betweenness centrality (approximate via random paths)
    between   = _approx_betweenness(G)

    return (
        node_names  = G.node_names,
        degree      = degree,
        strength    = strength,
        pagerank    = pr,
        eigenvector = eig_cent,
        hubs        = hubs,
        authorities = auth,
        betweenness = between,
    )
end

function _eigenvector_centrality(G::FinancialGraph; max_iter::Int=100, tol::Float64=1e-8)
    n = G.n
    x = ones(n)
    for _ in 1:max_iter
        x_new = G.weights * x
        norm_x = norm(x_new)
        norm_x < 1e-10 && break
        x_new ./= norm_x
        norm(x_new - x) < tol && (x = x_new; break)
        x = x_new
    end
    return x ./ (sum(x) + 1e-10)
end

function _approx_betweenness(G::FinancialGraph; n_samples::Int=100,
                               rng::AbstractRNG=MersenneTwister(42))
    n = G.n
    between = zeros(n)

    for _ in 1:n_samples
        s, t = randperm(rng, n)[1:2]
        path = _bfs_path(G, s, t)
        path === nothing && continue
        for node in path[2:end-1]
            between[node] += 1.0
        end
    end

    return between ./ (n_samples + 1e-10)
end

function _bfs_path(G::FinancialGraph, s::Int, t::Int)
    n      = G.n
    visited = falses(n)
    parent  = zeros(Int, n)
    queue   = [s]
    visited[s] = true

    while !isempty(queue)
        curr = popfirst!(queue)
        curr == t && break
        for j in 1:n
            G.adj[curr, j] == 0 && continue
            visited[j] && continue
            visited[j] = true
            parent[j]  = curr
            push!(queue, j)
        end
    end

    visited[t] || return nothing

    path = Int[t]
    curr = t
    while curr != s
        curr = parent[curr]
        curr == 0 && return nothing
        pushfirst!(path, curr)
    end
    return path
end

# ─────────────────────────────────────────────────────────────────────────────
# 11. COMMUNITY MODULARITY
# ─────────────────────────────────────────────────────────────────────────────

"""
    community_modularity(G, labels) → Float64

Compute modularity Q of a community partition.
Q ∈ [-0.5, 1]: higher = better community structure.
"""
function community_modularity(G::FinancialGraph, labels::Vector{Int})
    length(labels) == G.n || error("labels must have length G.n")
    W  = G.weights
    m  = sum(W) / 2 + 1e-10
    k  = vec(sum(W, dims=2))

    Q = 0.0
    for i in 1:G.n, j in 1:G.n
        labels[i] == labels[j] || continue
        Q += W[i,j] - k[i]*k[j] / (2m)
    end

    return Q / (2m)
end

"""
    louvain_step(G) → (labels, delta_Q)

Single Louvain pass: greedy modularity optimization.
Each node tries to join the community of its highest-gain neighbor.
"""
function louvain_step(G::FinancialGraph; rng::AbstractRNG=MersenneTwister(42))
    n      = G.n
    labels = collect(1:n)
    W      = G.weights
    k      = vec(sum(W, dims=2))
    m      = sum(W) / 2 + 1e-10

    Q_before = community_modularity(G, labels)
    improved = true

    while improved
        improved = false
        for i in randperm(rng, n)
            # Try moving node i to each neighbor's community
            neighbors = findall(G.adj[i, :] .> 0)
            isempty(neighbors) && continue

            neighbor_communities = unique([labels[j] for j in neighbors])
            best_gain   = 0.0
            best_comm   = labels[i]

            for c in neighbor_communities
                c == labels[i] && continue
                # Gain = 2 * [k_{i→c} - k[i] * Σ_{j in c} k[j] / (2m)]
                k_i_c  = sum(W[i,j] for j in 1:n if labels[j] == c)
                sum_k_c = sum(k[j] for j in 1:n if labels[j] == c)
                gain   = k_i_c - k[i] * sum_k_c / (2m)
                if gain > best_gain
                    best_gain = gain; best_comm = c
                end
            end

            if best_comm != labels[i]
                labels[i] = best_comm
                improved   = true
            end
        end
    end

    # Renumber communities
    unique_c  = unique(labels)
    c_map     = Dict(c => i for (i,c) in enumerate(unique_c))
    labels    = [c_map[l] for l in labels]

    Q_after = community_modularity(G, labels)
    return labels, Q_after - Q_before
end

end  # module GraphFinance
