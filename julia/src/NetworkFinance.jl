module NetworkFinance

using LinearAlgebra
using Statistics
using Random

# ============================================================================
# SECTION 1: Network Construction
# ============================================================================

"""
    FinancialNetwork

Weighted directed/undirected financial network.
"""
struct FinancialNetwork
    num_nodes::Int
    adjacency::Matrix{Float64}   # Weighted adjacency matrix
    node_names::Vector{String}
    directed::Bool
end

"""
    correlation_threshold_network(returns_matrix, threshold; method=:pearson)

Build network from correlation matrix with threshold.
Edge exists if |corr(i,j)| >= threshold.
"""
function correlation_threshold_network(returns_matrix::Matrix{Float64},
                                        threshold::Float64;
                                        method::Symbol=:pearson)::FinancialNetwork
    T, N = size(returns_matrix)
    C = _compute_correlation(returns_matrix, method)

    adj = zeros(N, N)
    for i in 1:N
        for j in (i+1):N
            if abs(C[i, j]) >= threshold
                adj[i, j] = C[i, j]
                adj[j, i] = C[i, j]
            end
        end
    end

    names = ["Asset_$i" for i in 1:N]
    return FinancialNetwork(N, adj, names, false)
end

"""
    _compute_correlation(returns_matrix, method)

Compute correlation matrix using specified method.
"""
function _compute_correlation(returns_matrix::Matrix{Float64}, method::Symbol)::Matrix{Float64}
    T, N = size(returns_matrix)

    if method == :pearson
        return cor(returns_matrix)
    elseif method == :spearman
        # Rank correlation
        ranked = zeros(T, N)
        for j in 1:N
            order = sortperm(returns_matrix[:, j])
            for i in 1:T
                ranked[order[i], j] = Float64(i)
            end
        end
        return cor(ranked)
    elseif method == :kendall
        C = zeros(N, N)
        for i in 1:N
            C[i, i] = 1.0
            for j in (i+1):N
                concordant = 0
                discordant = 0
                for s in 1:T
                    for t in (s+1):T
                        sign_i = sign(returns_matrix[t, i] - returns_matrix[s, i])
                        sign_j = sign(returns_matrix[t, j] - returns_matrix[s, j])
                        prod_sign = sign_i * sign_j
                        if prod_sign > 0
                            concordant += 1
                        elseif prod_sign < 0
                            discordant += 1
                        end
                    end
                end
                tau = (concordant - discordant) / max(concordant + discordant, 1)
                C[i, j] = tau
                C[j, i] = tau
            end
        end
        return C
    end

    return cor(returns_matrix)
end

"""
    distance_matrix_from_correlation(C)

Convert correlation to distance: d(i,j) = sqrt(2*(1 - rho(i,j))).
"""
function distance_matrix_from_correlation(C::Matrix{Float64})::Matrix{Float64}
    N = size(C, 1)
    D = zeros(N, N)
    for i in 1:N
        for j in 1:N
            D[i, j] = sqrt(max(2.0 * (1.0 - C[i, j]), 0.0))
        end
    end
    return D
end

"""
    minimum_spanning_tree(distance_matrix)

Prim's algorithm for MST. Returns adjacency matrix.
"""
function minimum_spanning_tree(D::Matrix{Float64})::Matrix{Float64}
    N = size(D, 1)
    in_tree = falses(N)
    adj = zeros(N, N)
    min_edge = fill(Inf, N)
    parent = zeros(Int, N)

    in_tree[1] = true
    for j in 2:N
        min_edge[j] = D[1, j]
        parent[j] = 1
    end

    for _ in 2:N
        # Find minimum edge to tree
        u = 0
        min_val = Inf
        for j in 1:N
            if !in_tree[j] && min_edge[j] < min_val
                min_val = min_edge[j]
                u = j
            end
        end

        if u == 0
            break
        end

        in_tree[u] = true
        adj[u, parent[u]] = D[u, parent[u]]
        adj[parent[u], u] = D[parent[u], u]

        # Update minimum edges
        for j in 1:N
            if !in_tree[j] && D[u, j] < min_edge[j]
                min_edge[j] = D[u, j]
                parent[j] = u
            end
        end
    end

    return adj
end

"""
    mst_network(returns_matrix)

Build MST-based financial network from returns.
"""
function mst_network(returns_matrix::Matrix{Float64})::FinancialNetwork
    C = cor(returns_matrix)
    D = distance_matrix_from_correlation(C)
    adj = minimum_spanning_tree(D)
    N = size(returns_matrix, 2)
    names = ["Asset_$i" for i in 1:N]
    return FinancialNetwork(N, adj, names, false)
end

"""
    pmfg_network(returns_matrix)

Planar Maximally Filtered Graph (Tumminello et al. 2005).
Greedy algorithm: add edges in decreasing correlation order,
only if planarity is maintained. Simplified: allow up to 3(N-2) edges.
"""
function pmfg_network(returns_matrix::Matrix{Float64})::FinancialNetwork
    T, N = size(returns_matrix)
    C = cor(returns_matrix)

    # Sort all edges by correlation (descending)
    edges = Tuple{Int, Int, Float64}[]
    for i in 1:N
        for j in (i+1):N
            push!(edges, (i, j, C[i, j]))
        end
    end
    sort!(edges, by=x -> -x[3])

    max_edges = 3 * (N - 2)
    adj = zeros(N, N)
    num_edges = 0

    # Degree tracking for planarity approximation
    degrees = zeros(Int, N)

    for (i, j, w) in edges
        if num_edges >= max_edges
            break
        end

        # Simplified planarity check: max degree constraint
        # True planarity testing is complex; we use degree heuristic
        if degrees[i] < 2 * sqrt(N) && degrees[j] < 2 * sqrt(N)
            adj[i, j] = w
            adj[j, i] = w
            degrees[i] += 1
            degrees[j] += 1
            num_edges += 1
        end
    end

    names = ["Asset_$i" for i in 1:N]
    return FinancialNetwork(N, adj, names, false)
end

"""
    partial_correlation_network(returns_matrix, threshold)

Network based on partial correlations (conditional independence).
"""
function partial_correlation_network(returns_matrix::Matrix{Float64},
                                      threshold::Float64)::FinancialNetwork
    T, N = size(returns_matrix)
    C = cor(returns_matrix)

    # Partial correlation via precision matrix
    P = inv(C + 1e-6 * I)

    # Convert precision to partial correlation
    partial_C = zeros(N, N)
    for i in 1:N
        for j in 1:N
            if i != j
                partial_C[i, j] = -P[i, j] / sqrt(max(P[i, i] * P[j, j], 1e-15))
            else
                partial_C[i, i] = 1.0
            end
        end
    end

    adj = zeros(N, N)
    for i in 1:N
        for j in (i+1):N
            if abs(partial_C[i, j]) >= threshold
                adj[i, j] = partial_C[i, j]
                adj[j, i] = partial_C[i, j]
            end
        end
    end

    names = ["Asset_$i" for i in 1:N]
    return FinancialNetwork(N, adj, names, false)
end

"""
    glasso_network(returns_matrix, lambda_reg)

Graphical LASSO network: sparse precision matrix estimation.
Uses coordinate descent.
"""
function glasso_network(returns_matrix::Matrix{Float64},
                        lambda_reg::Float64)::FinancialNetwork
    T, N = size(returns_matrix)
    S = cov(returns_matrix)

    # Initialize with diagonal
    W = copy(S) + lambda_reg * I
    Theta = inv(W)

    for outer_iter in 1:50
        W_old = copy(W)

        for j in 1:N
            # Partition: W_{11}, w_{12}, w_{22}
            idx = [i for i in 1:N if i != j]
            W_11 = W[idx, idx]
            s_12 = S[idx, j]

            # Coordinate descent for LASSO
            beta = zeros(N - 1)
            for cd_iter in 1:100
                for k in 1:(N-1)
                    r_k = s_12[k]
                    for l in 1:(N-1)
                        if l != k
                            r_k -= W_11[k, l] * beta[l]
                        end
                    end

                    # Soft threshold
                    if r_k > lambda_reg
                        beta[k] = (r_k - lambda_reg) / W_11[k, k]
                    elseif r_k < -lambda_reg
                        beta[k] = (r_k + lambda_reg) / W_11[k, k]
                    else
                        beta[k] = 0.0
                    end
                end
            end

            # Update W
            w_12 = W_11 * beta
            for (ki, k) in enumerate(idx)
                W[k, j] = w_12[ki]
                W[j, k] = w_12[ki]
            end
        end

        if norm(W - W_old) < 1e-6
            break
        end
    end

    Theta = inv(W)

    # Build adjacency from precision matrix
    adj = zeros(N, N)
    for i in 1:N
        for j in (i+1):N
            if abs(Theta[i, j]) > 1e-6
                partial_corr = -Theta[i, j] / sqrt(max(Theta[i, i] * Theta[j, j], 1e-15))
                adj[i, j] = partial_corr
                adj[j, i] = partial_corr
            end
        end
    end

    names = ["Asset_$i" for i in 1:N]
    return FinancialNetwork(N, adj, names, false)
end

# ============================================================================
# SECTION 2: Centrality Measures
# ============================================================================

"""
    degree_centrality(net::FinancialNetwork)

Degree centrality: number of connections normalized by max possible.
"""
function degree_centrality(net::FinancialNetwork)::Vector{Float64}
    N = net.num_nodes
    degrees = [sum(net.adjacency[i, j] != 0.0 for j in 1:N if j != i) for i in 1:N]
    max_degree = N - 1
    return degrees / max_degree
end

"""
    strength_centrality(net::FinancialNetwork)

Strength (weighted degree) centrality.
"""
function strength_centrality(net::FinancialNetwork)::Vector{Float64}
    N = net.num_nodes
    strengths = [sum(abs(net.adjacency[i, j]) for j in 1:N if j != i) for i in 1:N]
    max_s = maximum(strengths)
    return max_s > 0 ? strengths / max_s : strengths
end

"""
    betweenness_centrality(net::FinancialNetwork)

Brandes (2001) algorithm for betweenness centrality.
O(NM) for unweighted, O(NM + N^2 log N) for weighted.
"""
function betweenness_centrality(net::FinancialNetwork)::Vector{Float64}
    N = net.num_nodes
    CB = zeros(N)

    for s in 1:N
        # BFS/Dijkstra from source s
        S = Int[]          # Stack of nodes in order of non-increasing distance
        P = [Int[] for _ in 1:N]  # Predecessors
        sigma = zeros(N)   # Number of shortest paths
        sigma[s] = 1.0
        d = fill(-1.0, N)  # Distance
        d[s] = 0.0
        delta = zeros(N)

        # Use Dijkstra for weighted graphs
        # Priority queue as sorted list (simplified)
        Q = [(0.0, s)]

        while !isempty(Q)
            sort!(Q, by=x -> x[1])
            dist_v, v = popfirst!(Q)

            if dist_v > d[v] && d[v] >= 0.0 && dist_v > d[v] + 1e-10
                continue
            end

            push!(S, v)

            for w in 1:N
                if w != v && abs(net.adjacency[v, w]) > 0.0
                    edge_weight = 1.0 / max(abs(net.adjacency[v, w]), 1e-10)
                    new_dist = d[v] + edge_weight

                    if d[w] < 0.0 || new_dist < d[w] - 1e-10
                        d[w] = new_dist
                        sigma[w] = sigma[v]
                        P[w] = [v]
                        push!(Q, (new_dist, w))
                    elseif abs(new_dist - d[w]) < 1e-10
                        sigma[w] += sigma[v]
                        push!(P[w], v)
                    end
                end
            end
        end

        # Back-propagation
        while !isempty(S)
            w = pop!(S)
            for v in P[w]
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
    norm_factor = (N - 1) * (N - 2)
    if !net.directed
        norm_factor /= 2.0
    end
    if norm_factor > 0
        CB ./= norm_factor
    end

    return CB
end

"""
    closeness_centrality(net::FinancialNetwork)

Closeness centrality: inverse of average shortest path length.
"""
function closeness_centrality(net::FinancialNetwork)::Vector{Float64}
    N = net.num_nodes
    CC = zeros(N)

    for s in 1:N
        # Dijkstra
        dist = fill(Inf, N)
        dist[s] = 0.0
        visited = falses(N)

        for _ in 1:N
            u = 0
            min_d = Inf
            for v in 1:N
                if !visited[v] && dist[v] < min_d
                    min_d = dist[v]
                    u = v
                end
            end
            if u == 0
                break
            end
            visited[u] = true

            for v in 1:N
                if abs(net.adjacency[u, v]) > 0.0
                    w = 1.0 / max(abs(net.adjacency[u, v]), 1e-10)
                    if dist[u] + w < dist[v]
                        dist[v] = dist[u] + w
                    end
                end
            end
        end

        reachable = sum(dist .< Inf) - 1
        if reachable > 0
            total_dist = sum(d for d in dist if d < Inf && d > 0.0)
            CC[s] = reachable / max(total_dist, 1e-15)
        end
    end

    return CC
end

"""
    eigenvector_centrality(net::FinancialNetwork; max_iter=200, tol=1e-8)

Eigenvector centrality: dominant eigenvector of adjacency matrix.
"""
function eigenvector_centrality(net::FinancialNetwork;
                                 max_iter::Int=200, tol::Float64=1e-8)::Vector{Float64}
    N = net.num_nodes
    A = abs.(net.adjacency)
    x = ones(N) / sqrt(N)

    for iter in 1:max_iter
        x_new = A * x
        norm_x = norm(x_new)
        if norm_x < 1e-15
            break
        end
        x_new ./= norm_x

        if norm(x_new - x) < tol
            x = x_new
            break
        end
        x = x_new
    end

    # Ensure non-negative
    x = abs.(x)
    return x / max(maximum(x), 1e-15)
end

"""
    pagerank(net::FinancialNetwork; damping=0.85, max_iter=100, tol=1e-8)

PageRank centrality.
"""
function pagerank(net::FinancialNetwork; damping::Float64=0.85,
                   max_iter::Int=100, tol::Float64=1e-8)::Vector{Float64}
    N = net.num_nodes
    A = abs.(net.adjacency)

    # Out-degree
    out_degree = vec(sum(A, dims=2))
    pr = fill(1.0 / N, N)

    for iter in 1:max_iter
        pr_new = fill((1.0 - damping) / N, N)

        for j in 1:N
            if out_degree[j] > 0
                contribution = damping * pr[j] / out_degree[j]
                for i in 1:N
                    if A[j, i] > 0
                        pr_new[i] += contribution * A[j, i] / max(sum(A[j, :]), 1e-15) * out_degree[j]
                    end
                end
            else
                # Dangling node: distribute evenly
                for i in 1:N
                    pr_new[i] += damping * pr[j] / N
                end
            end
        end

        # Normalize
        s = sum(pr_new)
        pr_new ./= max(s, 1e-15)

        if norm(pr_new - pr) < tol
            return pr_new
        end
        pr = pr_new
    end

    return pr
end

"""
    katz_centrality(net::FinancialNetwork; alpha=0.1, beta=1.0)

Katz centrality: x = (I - alpha*A)^{-1} * beta * 1.
"""
function katz_centrality(net::FinancialNetwork; alpha::Float64=0.1,
                          beta::Float64=1.0)::Vector{Float64}
    N = net.num_nodes
    A = abs.(net.adjacency)

    # Ensure alpha < 1/spectral_radius
    eig_vals = eigvals(A)
    spectral_radius = maximum(abs.(eig_vals))
    alpha_safe = min(alpha, 0.99 / max(spectral_radius, 1.0))

    M = I - alpha_safe * A
    b = fill(beta, N)
    x = M \ b

    x = abs.(x)
    return x / max(maximum(x), 1e-15)
end

# ============================================================================
# SECTION 3: Community Detection
# ============================================================================

"""
    spectral_clustering(net::FinancialNetwork, num_clusters)

Spectral clustering using graph Laplacian.
"""
function spectral_clustering(net::FinancialNetwork, num_clusters::Int)::Vector{Int}
    N = net.num_nodes
    A = abs.(net.adjacency)
    D = Diagonal(vec(sum(A, dims=2)))

    # Normalized Laplacian: L_rw = I - D^{-1}A
    D_inv = Diagonal([D[i,i] > 0 ? 1.0 / D[i,i] : 0.0 for i in 1:N])
    L_rw = I - D_inv * A

    # Eigendecomposition
    eig = eigen(Symmetric(Matrix(L_rw)))
    # Use smallest K eigenvectors (excluding first which is all-zeros eigenvalue)
    K = min(num_clusters, N)
    V = eig.vectors[:, 1:K]

    # Normalize rows
    for i in 1:N
        row_norm = norm(V[i, :])
        if row_norm > 1e-10
            V[i, :] ./= row_norm
        end
    end

    # K-means clustering on V
    labels = _kmeans(V, K)
    return labels
end

"""
    _kmeans(X, K; max_iter=100, seed=42)

Simple K-means clustering.
"""
function _kmeans(X::Matrix{Float64}, K::Int; max_iter::Int=100, seed::Int=42)::Vector{Int}
    rng = Random.MersenneTwister(seed)
    N, D = size(X)

    # Initialize centroids randomly
    indices = randperm(rng, N)[1:min(K, N)]
    centroids = X[indices, :]

    labels = zeros(Int, N)

    for iter in 1:max_iter
        # Assign
        old_labels = copy(labels)
        for i in 1:N
            min_dist = Inf
            for k in 1:K
                d = sum((X[i, j] - centroids[k, j])^2 for j in 1:D)
                if d < min_dist
                    min_dist = d
                    labels[i] = k
                end
            end
        end

        if labels == old_labels
            break
        end

        # Update centroids
        for k in 1:K
            members = findall(labels .== k)
            if !isempty(members)
                for j in 1:D
                    centroids[k, j] = mean(X[m, j] for m in members)
                end
            end
        end
    end

    return labels
end

"""
    modularity(net::FinancialNetwork, communities)

Newman-Girvan modularity: Q = (1/2m) * sum (A_ij - k_i*k_j/(2m)) * delta(c_i, c_j).
"""
function modularity(net::FinancialNetwork, communities::Vector{Int})::Float64
    N = net.num_nodes
    A = abs.(net.adjacency)
    m2 = sum(A)  # 2m

    if m2 < 1e-15
        return 0.0
    end

    k = vec(sum(A, dims=2))
    Q = 0.0

    for i in 1:N
        for j in 1:N
            if communities[i] == communities[j]
                Q += A[i, j] - k[i] * k[j] / m2
            end
        end
    end

    return Q / m2
end

"""
    modularity_optimization(net::FinancialNetwork; max_iter=100, seed=42)

Greedy modularity optimization (Clauset, Newman, Moore 2004).
"""
function modularity_optimization(net::FinancialNetwork;
                                  max_iter::Int=100, seed::Int=42)::Vector{Int}
    N = net.num_nodes
    communities = collect(1:N)  # Each node starts in own community

    A = abs.(net.adjacency)
    m2 = sum(A)
    if m2 < 1e-15
        return communities
    end

    k = vec(sum(A, dims=2))

    for iter in 1:max_iter
        improved = false

        for i in 1:N
            best_comm = communities[i]
            best_dQ = 0.0

            # Try moving i to each neighbor's community
            neighbor_comms = Set{Int}()
            for j in 1:N
                if A[i, j] > 0 && j != i
                    push!(neighbor_comms, communities[j])
                end
            end

            for c in neighbor_comms
                if c == communities[i]
                    continue
                end

                # Compute delta Q for moving i from current to c
                sum_in_c = 0.0
                sum_in_old = 0.0
                for j in 1:N
                    if communities[j] == c && j != i
                        sum_in_c += A[i, j]
                    end
                    if communities[j] == communities[i] && j != i
                        sum_in_old += A[i, j]
                    end
                end

                k_c = sum(k[j] for j in 1:N if communities[j] == c)
                k_old = sum(k[j] for j in 1:N if communities[j] == communities[i]) - k[i]

                dQ = (sum_in_c - sum_in_old) / m2 +
                     k[i] * (k_old - k_c) / (m2^2 / 4.0)

                if dQ > best_dQ
                    best_dQ = dQ
                    best_comm = c
                end
            end

            if best_dQ > 1e-10
                communities[i] = best_comm
                improved = true
            end
        end

        if !improved
            break
        end
    end

    # Relabel communities consecutively
    unique_comms = unique(communities)
    label_map = Dict(c => i for (i, c) in enumerate(unique_comms))
    return [label_map[c] for c in communities]
end

"""
    label_propagation(net::FinancialNetwork; max_iter=100, seed=42)

Label propagation community detection.
"""
function label_propagation(net::FinancialNetwork;
                            max_iter::Int=100, seed::Int=42)::Vector{Int}
    rng = Random.MersenneTwister(seed)
    N = net.num_nodes
    labels = collect(1:N)
    A = abs.(net.adjacency)

    for iter in 1:max_iter
        order = randperm(rng, N)
        changed = false

        for i in order
            # Count weighted neighbor labels
            label_weights = Dict{Int, Float64}()
            for j in 1:N
                if A[i, j] > 0 && j != i
                    lbl = labels[j]
                    label_weights[lbl] = get(label_weights, lbl, 0.0) + A[i, j]
                end
            end

            if !isempty(label_weights)
                best_label = labels[i]
                best_weight = 0.0
                for (lbl, w) in label_weights
                    if w > best_weight
                        best_weight = w
                        best_label = lbl
                    end
                end

                if best_label != labels[i]
                    labels[i] = best_label
                    changed = true
                end
            end
        end

        if !changed
            break
        end
    end

    # Relabel
    unique_labels = unique(labels)
    label_map = Dict(l => i for (i, l) in enumerate(unique_labels))
    return [label_map[l] for l in labels]
end

# ============================================================================
# SECTION 4: Dynamic Networks
# ============================================================================

"""
    rolling_correlation_network(returns_matrix, window, step, threshold)

Build time-varying network from rolling correlation windows.
"""
function rolling_correlation_network(returns_matrix::Matrix{Float64},
                                      window::Int, step::Int, threshold::Float64)
    T, N = size(returns_matrix)
    num_snapshots = (T - window) ÷ step + 1

    networks = Vector{FinancialNetwork}(undef, num_snapshots)
    densities = Vector{Float64}(undef, num_snapshots)
    avg_correlations = Vector{Float64}(undef, num_snapshots)

    for s in 1:num_snapshots
        start_t = (s - 1) * step + 1
        end_t = start_t + window - 1

        sub_returns = returns_matrix[start_t:end_t, :]
        net = correlation_threshold_network(sub_returns, threshold)
        networks[s] = net

        # Network statistics
        num_edges = sum(net.adjacency .!= 0) / 2
        max_edges = N * (N - 1) / 2
        densities[s] = num_edges / max_edges

        C = cor(sub_returns)
        upper_tri = [C[i, j] for i in 1:N for j in (i+1):N]
        avg_correlations[s] = mean(upper_tri)
    end

    return (networks=networks, densities=densities,
            avg_correlations=avg_correlations)
end

"""
    network_distance(net1::FinancialNetwork, net2::FinancialNetwork)

Distance between two networks (Frobenius norm of adjacency difference).
"""
function network_distance(net1::FinancialNetwork, net2::FinancialNetwork)::Float64
    return norm(net1.adjacency - net2.adjacency)
end

"""
    regime_conditional_networks(returns_matrix, regime_labels, threshold)

Build separate networks for each regime.
"""
function regime_conditional_networks(returns_matrix::Matrix{Float64},
                                      regime_labels::Vector{Int},
                                      threshold::Float64)
    regimes = unique(regime_labels)
    networks = Dict{Int, FinancialNetwork}()

    for r in regimes
        idx = findall(regime_labels .== r)
        if length(idx) > 10
            sub_returns = returns_matrix[idx, :]
            networks[r] = correlation_threshold_network(sub_returns, threshold)
        end
    end

    return networks
end

"""
    network_persistence(networks::Vector{FinancialNetwork})

Measure edge persistence across time windows.
"""
function network_persistence(networks::Vector{FinancialNetwork})::Matrix{Float64}
    if isempty(networks)
        return zeros(0, 0)
    end

    N = networks[1].num_nodes
    T_net = length(networks)
    persistence = zeros(N, N)

    for net in networks
        for i in 1:N
            for j in 1:N
                if net.adjacency[i, j] != 0.0
                    persistence[i, j] += 1.0
                end
            end
        end
    end

    persistence ./= T_net
    return persistence
end

# ============================================================================
# SECTION 5: Contagion Models
# ============================================================================

"""
    sir_epidemic(net::FinancialNetwork, initial_infected, beta, gamma, num_steps;
                 seed=42)

SIR epidemic model on financial network.
S -> I with probability beta per infected neighbor.
I -> R with probability gamma.
"""
function sir_epidemic(net::FinancialNetwork, initial_infected::Vector{Int},
                      beta::Float64, gamma_rec::Float64, num_steps::Int;
                      seed::Int=42)
    rng = Random.MersenneTwister(seed)
    N = net.num_nodes
    A = abs.(net.adjacency)

    # States: 0=S, 1=I, 2=R
    state = zeros(Int, N)
    for i in initial_infected
        state[i] = 1
    end

    S_count = Vector{Int}(undef, num_steps + 1)
    I_count = Vector{Int}(undef, num_steps + 1)
    R_count = Vector{Int}(undef, num_steps + 1)

    S_count[1] = sum(state .== 0)
    I_count[1] = sum(state .== 1)
    R_count[1] = sum(state .== 2)

    for t in 1:num_steps
        new_state = copy(state)

        for i in 1:N
            if state[i] == 0  # Susceptible
                # Check infected neighbors
                for j in 1:N
                    if state[j] == 1 && A[i, j] > 0
                        infection_prob = beta * abs(A[i, j]) / max(maximum(abs.(A)), 1e-10)
                        if rand(rng) < infection_prob
                            new_state[i] = 1
                            break
                        end
                    end
                end
            elseif state[i] == 1  # Infected
                if rand(rng) < gamma_rec
                    new_state[i] = 2
                end
            end
        end

        state = new_state
        S_count[t+1] = sum(state .== 0)
        I_count[t+1] = sum(state .== 1)
        R_count[t+1] = sum(state .== 2)
    end

    return (susceptible=S_count, infected=I_count, recovered=R_count,
            final_state=state, total_infected=R_count[end] + I_count[end])
end

"""
    debtrank_network(net::FinancialNetwork, equities, initial_shocks)

DebtRank (Battiston et al. 2012) on financial network.
"""
function debtrank_network(net::FinancialNetwork, equities::Vector{Float64},
                           initial_shocks::Vector{Float64})
    N = net.num_nodes
    A = abs.(net.adjacency)

    h = copy(initial_shocks)
    h = clamp.(h, 0.0, 1.0)

    # Leverage-weighted exposure
    W = zeros(N, N)
    for i in 1:N
        for j in 1:N
            if equities[i] > 0 && A[j, i] > 0
                W[i, j] = min(A[j, i] / equities[i], 1.0)
            end
        end
    end

    state = ones(Int, N)  # 1=undistressed
    for i in 1:N
        if h[i] > 0.0
            state[i] = 2  # distressed
        end
    end

    for round in 1:N
        h_new = copy(h)
        changed = false

        for i in 1:N
            if state[i] == 1
                stress = sum(W[i, j] * h[j] for j in 1:N if state[j] == 2)
                if stress > 0.0
                    h_new[i] = min(h[i] + stress, 1.0)
                    state[i] = 2
                    changed = true
                end
            end
        end

        for i in 1:N
            if state[i] == 2 && !changed
                state[i] = 3
            end
        end

        h = h_new
        if !changed
            break
        end
    end

    total_assets = sum(equities)
    dr = sum(h[i] * equities[i] for i in 1:N) / max(total_assets, 1e-15)

    return (debtrank=dr, distress_levels=h, states=state)
end

"""
    fire_sale_cascade(net::FinancialNetwork, asset_holdings, initial_shock_asset,
                      price_impact_param, leverage_ratios)

Fire-sale cascade model.
Banks sell assets -> price drops -> other banks lose value -> forced selling.
"""
function fire_sale_cascade(net::FinancialNetwork, asset_holdings::Matrix{Float64},
                            initial_shock_asset::Int, price_impact_param::Float64,
                            leverage_ratios::Vector{Float64})
    N = net.num_nodes  # Banks
    M = size(asset_holdings, 2)  # Assets

    prices = ones(M)
    equity = zeros(N)
    for i in 1:N
        equity[i] = sum(asset_holdings[i, :] .* prices) / max(leverage_ratios[i], 1.0)
    end

    # Initial shock
    prices[initial_shock_asset] *= 0.8  # 20% drop

    rounds = 0
    max_rounds = 50

    for round in 1:max_rounds
        rounds = round
        sales = zeros(N, M)
        any_sale = false

        for i in 1:N
            # Mark to market
            portfolio_value = sum(asset_holdings[i, a] * prices[a] for a in 1:M)
            current_equity = portfolio_value / max(leverage_ratios[i], 1.0)

            if current_equity < equity[i] * 0.5  # Distressed
                # Fire sale: sell proportionally
                for a in 1:M
                    sales[i, a] = asset_holdings[i, a] * 0.1
                    any_sale = true
                end
            end
        end

        if !any_sale
            break
        end

        # Price impact from sales
        for a in 1:M
            total_sales = sum(sales[i, a] for i in 1:N)
            prices[a] *= exp(-price_impact_param * total_sales)
            prices[a] = max(prices[a], 0.01)
        end

        # Update holdings
        for i in 1:N
            for a in 1:M
                asset_holdings[i, a] -= sales[i, a]
            end
        end
    end

    # Final equity
    final_equity = zeros(N)
    for i in 1:N
        final_equity[i] = sum(asset_holdings[i, a] * prices[a] for a in 1:M) /
                           max(leverage_ratios[i], 1.0)
    end

    return (final_prices=prices, final_equity=final_equity,
            rounds=rounds, price_decline=1.0 .- prices)
end

# ============================================================================
# SECTION 6: Bipartite Networks
# ============================================================================

"""
    BipartiteNetwork

Bipartite network (e.g., fund-stock holdings).
"""
struct BipartiteNetwork
    num_type_a::Int    # e.g., funds
    num_type_b::Int    # e.g., stocks
    weights::Matrix{Float64}  # A x B weight matrix
    names_a::Vector{String}
    names_b::Vector{String}
end

"""
    bipartite_projection_a(bn::BipartiteNetwork)

Project bipartite network onto type-A nodes.
A_ij = sum_k w_ik * w_jk (common neighbors in B).
"""
function bipartite_projection_a(bn::BipartiteNetwork)::FinancialNetwork
    W = bn.weights
    proj = W * W'

    # Normalize
    for i in 1:bn.num_type_a
        proj[i, i] = 0.0
    end

    return FinancialNetwork(bn.num_type_a, proj, bn.names_a, false)
end

"""
    bipartite_projection_b(bn::BipartiteNetwork)

Project bipartite network onto type-B nodes.
"""
function bipartite_projection_b(bn::BipartiteNetwork)::FinancialNetwork
    W = bn.weights
    proj = W' * W

    for i in 1:bn.num_type_b
        proj[i, i] = 0.0
    end

    return FinancialNetwork(bn.num_type_b, proj, bn.names_b, false)
end

"""
    portfolio_overlap(holdings_matrix)

Compute portfolio overlap matrix.
holdings_matrix: F x S (funds x stocks), entry = weight or shares held.
overlap(i,j) = cos similarity of holdings vectors.
"""
function portfolio_overlap(holdings_matrix::Matrix{Float64})::Matrix{Float64}
    F = size(holdings_matrix, 1)
    overlap = zeros(F, F)

    norms = [norm(holdings_matrix[i, :]) for i in 1:F]

    for i in 1:F
        for j in i:F
            if norms[i] > 0 && norms[j] > 0
                overlap[i, j] = dot(holdings_matrix[i, :], holdings_matrix[j, :]) /
                                (norms[i] * norms[j])
            end
            overlap[j, i] = overlap[i, j]
        end
    end

    return overlap
end

"""
    crowding_index(holdings_matrix)

Measure of portfolio crowding for each stock.
HHI of ownership: crowding_j = sum (w_ij / sum_i w_ij)^2
"""
function crowding_index(holdings_matrix::Matrix{Float64})::Vector{Float64}
    F, S = size(holdings_matrix)
    crowding = Vector{Float64}(undef, S)

    for j in 1:S
        total = sum(holdings_matrix[:, j])
        if total > 0
            crowding[j] = sum((holdings_matrix[i, j] / total)^2 for i in 1:F)
        else
            crowding[j] = 0.0
        end
    end

    return crowding
end

# ============================================================================
# SECTION 7: Granger Causality Network
# ============================================================================

"""
    granger_causality_test(x, y, lags)

Pairwise Granger causality test: does x Granger-cause y?
F-test comparing restricted and unrestricted models.
"""
function granger_causality_test(x::Vector{Float64}, y::Vector{Float64},
                                 lags::Int)
    n = min(length(x), length(y))
    T = n - lags

    if T < lags + 5
        return (f_stat=0.0, p_value=1.0, granger_causes=false)
    end

    # Restricted model: y_t = sum a_i * y_{t-i} + eps
    Y = y[(lags+1):n]
    X_r = zeros(T, lags)
    for l in 1:lags
        X_r[:, l] = y[(lags+1-l):(n-l)]
    end
    X_r = hcat(ones(T), X_r)

    beta_r = (X_r' * X_r) \ (X_r' * Y)
    e_r = Y - X_r * beta_r
    ssr_r = sum(e_r.^2)

    # Unrestricted model: y_t = sum a_i * y_{t-i} + sum b_i * x_{t-i} + eps
    X_u = hcat(X_r, zeros(T, lags))
    for l in 1:lags
        X_u[:, lags + 1 + l] = x[(lags+1-l):(n-l)]
    end

    beta_u = (X_u' * X_u) \ (X_u' * Y)
    e_u = Y - X_u * beta_u
    ssr_u = sum(e_u.^2)

    # F-test
    df1 = lags
    df2 = T - 2 * lags - 1
    if df2 <= 0 || ssr_u < 1e-15
        return (f_stat=0.0, p_value=1.0, granger_causes=false)
    end

    f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)

    # Approximate p-value using F-distribution via beta function
    p_value = _f_distribution_pvalue(f_stat, df1, df2)

    return (f_stat=f_stat, p_value=p_value, granger_causes=p_value < 0.05)
end

"""
    _f_distribution_pvalue(f, d1, d2)

Approximate p-value for F-distribution using incomplete beta.
"""
function _f_distribution_pvalue(f::Float64, d1::Int, d2::Int)::Float64
    if f <= 0.0
        return 1.0
    end
    x = d2 / (d2 + d1 * f)
    # Incomplete beta approximation via continued fraction
    # Use normal approximation for large df
    if d1 + d2 > 30
        z = ((f / d1 - 1.0 / d2) * sqrt(d2)) / sqrt(2.0 / d1 + 2.0 / d2)
        return 1.0 - _normal_cdf_approx(z)
    end
    # Simple approximation
    chi_sq = f * d1
    z = sqrt(2.0 * chi_sq) - sqrt(2.0 * d1 - 1.0)
    return 1.0 - _normal_cdf_approx(z)
end

function _normal_cdf_approx(x::Float64)::Float64
    if x < -8.0
        return 0.0
    elseif x > 8.0
        return 1.0
    end
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    pdf_val = exp(-0.5 * x * x) / sqrt(2.0 * pi)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
           t * (-1.821255978 + t * 1.330274429))))
    cdf_val = 1.0 - pdf_val * poly
    return x >= 0.0 ? cdf_val : 1.0 - cdf_val
end

"""
    granger_causality_network(returns_matrix, lags; significance=0.05)

Build directed Granger causality network from returns.
"""
function granger_causality_network(returns_matrix::Matrix{Float64}, lags::Int;
                                    significance::Float64=0.05)::FinancialNetwork
    T, N = size(returns_matrix)
    adj = zeros(N, N)

    for i in 1:N
        for j in 1:N
            if i != j
                result = granger_causality_test(returns_matrix[:, i],
                                                 returns_matrix[:, j], lags)
                if result.granger_causes
                    adj[i, j] = result.f_stat
                end
            end
        end
    end

    names = ["Asset_$i" for i in 1:N]
    return FinancialNetwork(N, adj, names, true)
end

# ============================================================================
# SECTION 8: Transfer Entropy Network
# ============================================================================

"""
    transfer_entropy(x, y, lags; num_bins=10)

Transfer entropy from X to Y: TE_{X->Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k}).
Estimated via binning.
"""
function transfer_entropy(x::Vector{Float64}, y::Vector{Float64}, lags::Int;
                           num_bins::Int=10)::Float64
    n = min(length(x), length(y))
    T = n - lags

    if T < 20
        return 0.0
    end

    # Bin data
    x_binned = _bin_data(x, num_bins)
    y_binned = _bin_data(y, num_bins)

    # Compute conditional entropies
    # H(Y_t | Y_past) and H(Y_t | Y_past, X_past)

    # For simplicity, use lag=1
    # Count joint probabilities
    # P(y_t, y_{t-1}) and P(y_t, y_{t-1}, x_{t-1})
    counts_yy = zeros(num_bins, num_bins)
    counts_yyx = zeros(num_bins, num_bins, num_bins)
    counts_y = zeros(num_bins)
    counts_yx = zeros(num_bins, num_bins)

    for t in (lags+1):n
        yt = y_binned[t]
        yt1 = y_binned[t-1]
        xt1 = x_binned[t-1]

        counts_yy[yt, yt1] += 1.0
        counts_yyx[yt, yt1, xt1] += 1.0
        counts_y[yt1] += 1.0
        counts_yx[yt1, xt1] += 1.0
    end

    total = Float64(T)

    # H(Y_t | Y_{t-1}) = -sum P(y_t, y_{t-1}) * log(P(y_t | y_{t-1}))
    h_y_given_ypast = 0.0
    for yt in 1:num_bins
        for yt1 in 1:num_bins
            p_joint = counts_yy[yt, yt1] / total
            p_cond = counts_y[yt1] > 0 ? counts_yy[yt, yt1] / counts_y[yt1] : 0.0
            if p_joint > 0 && p_cond > 0
                h_y_given_ypast -= p_joint * log(p_cond)
            end
        end
    end

    # H(Y_t | Y_{t-1}, X_{t-1})
    h_y_given_ypast_xpast = 0.0
    for yt in 1:num_bins
        for yt1 in 1:num_bins
            for xt1 in 1:num_bins
                p_joint = counts_yyx[yt, yt1, xt1] / total
                p_cond = counts_yx[yt1, xt1] > 0 ?
                         counts_yyx[yt, yt1, xt1] / counts_yx[yt1, xt1] : 0.0
                if p_joint > 0 && p_cond > 0
                    h_y_given_ypast_xpast -= p_joint * log(p_cond)
                end
            end
        end
    end

    te = h_y_given_ypast - h_y_given_ypast_xpast
    return max(te, 0.0)
end

"""
    _bin_data(x, num_bins)

Bin data into num_bins equal-frequency bins.
"""
function _bin_data(x::Vector{Float64}, num_bins::Int)::Vector{Int}
    n = length(x)
    sorted_idx = sortperm(x)
    bins = zeros(Int, n)
    bin_size = max(1, n ÷ num_bins)

    for i in 1:n
        bins[sorted_idx[i]] = min((i - 1) ÷ bin_size + 1, num_bins)
    end

    return bins
end

"""
    transfer_entropy_network(returns_matrix, lags; num_bins=10, threshold=0.01)

Build directed transfer entropy network.
"""
function transfer_entropy_network(returns_matrix::Matrix{Float64}, lags::Int;
                                   num_bins::Int=10,
                                   threshold::Float64=0.01)::FinancialNetwork
    T, N = size(returns_matrix)
    adj = zeros(N, N)

    for i in 1:N
        for j in 1:N
            if i != j
                te = transfer_entropy(returns_matrix[:, i], returns_matrix[:, j],
                                       lags; num_bins=num_bins)
                if te > threshold
                    adj[i, j] = te
                end
            end
        end
    end

    names = ["Asset_$i" for i in 1:N]
    return FinancialNetwork(N, adj, names, true)
end

"""
    net_information_flow(te_network::FinancialNetwork)

Net information flow for each node: outgoing TE - incoming TE.
"""
function net_information_flow(te_network::FinancialNetwork)::Vector{Float64}
    N = te_network.num_nodes
    A = te_network.adjacency
    outflow = vec(sum(A, dims=2))
    inflow = vec(sum(A, dims=1))
    return outflow - inflow
end

# ============================================================================
# SECTION 9: Network-Based Portfolio
# ============================================================================

"""
    mst_portfolio_weights(returns_matrix)

MST-based portfolio: weights inversely proportional to centrality.
Peripheral (less connected) assets get higher weight for diversification.
"""
function mst_portfolio_weights(returns_matrix::Matrix{Float64})::Vector{Float64}
    net = mst_network(returns_matrix)
    centrality = degree_centrality(net)

    # Inverse centrality weights
    inv_centrality = [1.0 / max(c, 0.01) for c in centrality]
    total = sum(inv_centrality)
    return inv_centrality / total
end

"""
    centrality_weighted_portfolio(returns_matrix, threshold; centrality_type=:eigenvector)

Portfolio with weights based on network centrality.
"""
function centrality_weighted_portfolio(returns_matrix::Matrix{Float64},
                                        threshold::Float64;
                                        centrality_type::Symbol=:eigenvector)::Vector{Float64}
    net = correlation_threshold_network(returns_matrix, threshold)

    if centrality_type == :eigenvector
        centrality = eigenvector_centrality(net)
    elseif centrality_type == :betweenness
        centrality = betweenness_centrality(net)
    elseif centrality_type == :pagerank
        centrality = pagerank(net)
    elseif centrality_type == :closeness
        centrality = closeness_centrality(net)
    else
        centrality = degree_centrality(net)
    end

    # Inverse centrality for diversification
    inv_c = [1.0 / max(c, 0.01) for c in centrality]
    total = sum(inv_c)
    return inv_c / total
end

"""
    network_diversification_score(returns_matrix, weights, threshold)

Score portfolio diversification using network structure.
"""
function network_diversification_score(returns_matrix::Matrix{Float64},
                                        weights::Vector{Float64},
                                        threshold::Float64)::Float64
    net = correlation_threshold_network(returns_matrix, threshold)
    N = net.num_nodes

    # Weighted average clustering coefficient
    cluster_coeff = _clustering_coefficient(net)

    # Cross-community holding
    communities = spectral_clustering(net, max(2, round(Int, sqrt(N))))
    num_communities = maximum(communities)

    comm_weights = zeros(num_communities)
    for i in 1:N
        comm_weights[communities[i]] += weights[i]
    end

    # HHI of community weights
    hhi = sum(w^2 for w in comm_weights)

    return 1.0 - hhi  # Higher is more diversified
end

"""
    _clustering_coefficient(net::FinancialNetwork)

Local clustering coefficient for each node.
"""
function _clustering_coefficient(net::FinancialNetwork)::Vector{Float64}
    N = net.num_nodes
    A = Float64.(net.adjacency .!= 0)
    cc = zeros(N)

    for i in 1:N
        neighbors = findall(A[i, :] .> 0)
        k = length(neighbors)
        if k < 2
            cc[i] = 0.0
            continue
        end

        triangles = 0.0
        for a in neighbors
            for b in neighbors
                if a < b && A[a, b] > 0
                    triangles += 1.0
                end
            end
        end

        cc[i] = 2.0 * triangles / (k * (k - 1))
    end

    return cc
end

# ============================================================================
# SECTION 10: Systemic Risk Indicators
# ============================================================================

"""
    absorption_ratio(returns_matrix, num_eigenvectors)

Kritzman, Li, Page, Rigobon (2011) Absorption Ratio.
AR = fraction of variance explained by top eigenvectors.
"""
function absorption_ratio(returns_matrix::Matrix{Float64},
                           num_eigenvectors::Int)::Float64
    C = cov(returns_matrix)
    eig_vals = eigvals(Symmetric(C))
    sorted_vals = sort(eig_vals, rev=true)

    K = min(num_eigenvectors, length(sorted_vals))
    total_var = sum(sorted_vals)

    return sum(sorted_vals[1:K]) / max(total_var, 1e-15)
end

"""
    turbulence_index(returns, mu_history, sigma_history)

Kritzman-Li (2010) turbulence index.
d_t = (r_t - mu)' * Sigma^{-1} * (r_t - mu)
"""
function turbulence_index(returns::Vector{Float64}, mu_history::Vector{Float64},
                           sigma_history::Matrix{Float64})::Float64
    diff_vec = returns - mu_history
    return dot(diff_vec, sigma_history \ diff_vec)
end

"""
    turbulence_index_series(returns_matrix, lookback)

Rolling turbulence index.
"""
function turbulence_index_series(returns_matrix::Matrix{Float64},
                                  lookback::Int)::Vector{Float64}
    T, N = size(returns_matrix)
    turb = Vector{Float64}(undef, max(T - lookback, 0))

    for t in (lookback+1):T
        historical = returns_matrix[(t-lookback):(t-1), :]
        mu = vec(mean(historical, dims=1))
        sigma = cov(historical)
        sigma += 1e-6 * I  # Regularize

        current = returns_matrix[t, :]
        turb[t - lookback] = turbulence_index(current, mu, sigma)
    end

    return turb
end

"""
    network_density_indicator(returns_matrix, window, step, threshold)

Network density as crisis indicator.
High density = high systemic risk.
"""
function network_density_indicator(returns_matrix::Matrix{Float64},
                                    window::Int, step::Int,
                                    threshold::Float64)::Vector{Float64}
    result = rolling_correlation_network(returns_matrix, window, step, threshold)
    return result.densities
end

"""
    systemic_risk_dashboard(returns_matrix, lookback, threshold)

Comprehensive systemic risk metrics.
"""
function systemic_risk_dashboard(returns_matrix::Matrix{Float64},
                                  lookback::Int, threshold::Float64)
    T, N = size(returns_matrix)

    # Absorption ratio
    ar = absorption_ratio(returns_matrix, max(1, N ÷ 5))

    # Average correlation
    C = cor(returns_matrix)
    upper_tri = [C[i, j] for i in 1:N for j in (i+1):N]
    avg_corr = mean(upper_tri)

    # Network density
    net = correlation_threshold_network(returns_matrix, threshold)
    num_edges = sum(net.adjacency .!= 0) / 2
    density = num_edges / (N * (N - 1) / 2)

    # Clustering
    cc = _clustering_coefficient(net)
    avg_clustering = mean(cc)

    # Eigenvalue concentration
    eig_vals = eigvals(Symmetric(C))
    sorted_eigs = sort(eig_vals, rev=true)
    top_eig_ratio = sorted_eigs[1] / max(sum(sorted_eigs), 1e-15)

    return (absorption_ratio=ar, avg_correlation=avg_corr,
            network_density=density, avg_clustering=avg_clustering,
            eigenvalue_concentration=top_eig_ratio,
            risk_score=0.3 * ar + 0.3 * avg_corr + 0.2 * density + 0.2 * top_eig_ratio)
end

# ============================================================================
# SECTION 11: Multiplex Networks
# ============================================================================

"""
    MultiplexNetwork

Multi-layer financial network.
"""
struct MultiplexNetwork
    layers::Vector{FinancialNetwork}
    layer_names::Vector{String}
    num_nodes::Int
end

"""
    create_multiplex_network(returns_matrix, lags, threshold)

Create multiplex with correlation, Granger causality, and partial correlation layers.
"""
function create_multiplex_network(returns_matrix::Matrix{Float64},
                                   lags::Int, threshold::Float64)::MultiplexNetwork
    N = size(returns_matrix, 2)

    # Layer 1: Correlation
    corr_net = correlation_threshold_network(returns_matrix, threshold)

    # Layer 2: Granger causality
    gc_net = granger_causality_network(returns_matrix, lags)

    # Layer 3: Partial correlation
    pc_net = partial_correlation_network(returns_matrix, threshold)

    return MultiplexNetwork([corr_net, gc_net, pc_net],
                             ["Correlation", "Granger", "Partial"],
                             N)
end

"""
    multiplex_pagerank(mn::MultiplexNetwork; damping=0.85, layer_weights=nothing)

PageRank on multiplex network: aggregate across layers.
"""
function multiplex_pagerank(mn::MultiplexNetwork; damping::Float64=0.85,
                             layer_weights::Union{Nothing, Vector{Float64}}=nothing)::Vector{Float64}
    L = length(mn.layers)
    N = mn.num_nodes

    if layer_weights === nothing
        layer_weights = fill(1.0 / L, L)
    end

    # Aggregate adjacency
    A_agg = zeros(N, N)
    for (l, net) in enumerate(mn.layers)
        A_agg .+= layer_weights[l] * abs.(net.adjacency)
    end

    agg_net = FinancialNetwork(N, A_agg, mn.layers[1].node_names, false)
    return pagerank(agg_net; damping=damping)
end

"""
    multiplex_overlap(mn::MultiplexNetwork)

Edge overlap across layers: fraction of edges present in multiple layers.
"""
function multiplex_overlap(mn::MultiplexNetwork)::Float64
    L = length(mn.layers)
    N = mn.num_nodes

    edge_counts = zeros(Int, N, N)
    for net in mn.layers
        for i in 1:N
            for j in 1:N
                if net.adjacency[i, j] != 0.0
                    edge_counts[i, j] += 1
                end
            end
        end
    end

    total_edges = sum(edge_counts .> 0)
    multi_layer_edges = sum(edge_counts .> 1)

    return total_edges > 0 ? multi_layer_edges / total_edges : 0.0
end

"""
    interlayer_correlation(mn::MultiplexNetwork)

Correlation between layers based on edge presence.
"""
function interlayer_correlation(mn::MultiplexNetwork)::Matrix{Float64}
    L = length(mn.layers)
    N = mn.num_nodes

    # Flatten adjacency to vectors
    vectors = [vec(mn.layers[l].adjacency .!= 0.0) for l in 1:L]

    corr_matrix = zeros(L, L)
    for i in 1:L
        for j in 1:L
            corr_matrix[i, j] = cor(Float64.(vectors[i]), Float64.(vectors[j]))
        end
    end

    return corr_matrix
end

"""
    network_summary_statistics(net::FinancialNetwork)

Comprehensive network summary.
"""
function network_summary_statistics(net::FinancialNetwork)
    N = net.num_nodes
    A = abs.(net.adjacency)

    num_edges = sum(A .> 0) / (net.directed ? 1 : 2)
    max_edges = net.directed ? N * (N - 1) : N * (N - 1) / 2
    density = num_edges / max(max_edges, 1.0)

    degrees = [sum(A[i, :] .> 0) for i in 1:N]
    avg_degree = mean(degrees)
    max_degree = maximum(degrees)

    strengths = vec(sum(A, dims=2))
    avg_strength = mean(strengths)

    cc = _clustering_coefficient(net)
    avg_clustering = mean(cc)

    # Assortativity (degree correlation)
    assort = _assortativity(net)

    return (num_nodes=N, num_edges=Int(num_edges), density=density,
            avg_degree=avg_degree, max_degree=max_degree,
            avg_strength=avg_strength, avg_clustering=avg_clustering,
            assortativity=assort)
end

"""
    _assortativity(net::FinancialNetwork)

Degree assortativity coefficient.
"""
function _assortativity(net::FinancialNetwork)::Float64
    N = net.num_nodes
    A = Float64.(net.adjacency .!= 0)
    degrees = vec(sum(A, dims=2))

    edges = Tuple{Int, Int}[]
    for i in 1:N
        for j in (i+1):N
            if A[i, j] > 0
                push!(edges, (i, j))
            end
        end
    end

    M = length(edges)
    if M < 2
        return 0.0
    end

    sum_jk = 0.0
    sum_j_plus_k = 0.0
    sum_j2_plus_k2 = 0.0

    for (j, k) in edges
        dj = degrees[j]
        dk = degrees[k]
        sum_jk += dj * dk
        sum_j_plus_k += dj + dk
        sum_j2_plus_k2 += dj^2 + dk^2
    end

    numerator = sum_jk / M - (sum_j_plus_k / (2.0 * M))^2
    denominator = sum_j2_plus_k2 / (2.0 * M) - (sum_j_plus_k / (2.0 * M))^2

    return abs(denominator) > 1e-15 ? numerator / denominator : 0.0
end

"""
    shortest_path_distribution(net::FinancialNetwork)

Distribution of shortest path lengths.
"""
function shortest_path_distribution(net::FinancialNetwork)
    N = net.num_nodes
    A = abs.(net.adjacency)

    all_paths = Float64[]

    for s in 1:N
        dist = fill(Inf, N)
        dist[s] = 0.0
        visited = falses(N)

        for _ in 1:N
            u = 0
            min_d = Inf
            for v in 1:N
                if !visited[v] && dist[v] < min_d
                    min_d = dist[v]
                    u = v
                end
            end
            if u == 0
                break
            end
            visited[u] = true

            for v in 1:N
                if A[u, v] > 0
                    w = 1.0 / max(A[u, v], 1e-10)
                    if dist[u] + w < dist[v]
                        dist[v] = dist[u] + w
                    end
                end
            end
        end

        for v in 1:N
            if v != s && dist[v] < Inf
                push!(all_paths, dist[v])
            end
        end
    end

    if isempty(all_paths)
        return (mean_path=0.0, median_path=0.0, max_path=0.0, diameter=0.0)
    end

    return (mean_path=mean(all_paths), median_path=median(all_paths),
            max_path=maximum(all_paths), diameter=maximum(all_paths),
            num_paths=length(all_paths))
end

# ============================================================================
# SECTION 12: Additional Network Utilities
# ============================================================================

"""
    k_core_decomposition(net::FinancialNetwork)

K-core decomposition: maximal subgraph where every node has degree >= k.
Returns core number for each node.
"""
function k_core_decomposition(net::FinancialNetwork)::Vector{Int}
    N = net.num_nodes
    A = Float64.(net.adjacency .!= 0)
    degrees = [sum(A[i, :]) for i in 1:N]
    core = zeros(Int, N)
    remaining = trues(N)

    k = 0
    while any(remaining)
        k += 1
        changed = true
        while changed
            changed = false
            for i in 1:N
                if remaining[i]
                    deg = sum(A[i, j] > 0 && remaining[j] for j in 1:N)
                    if deg < k
                        remaining[i] = false
                        core[i] = k - 1
                        changed = true
                    end
                end
            end
        end
    end

    # Remaining nodes get highest core
    for i in 1:N
        if core[i] == 0
            core[i] = k
        end
    end

    return core
end

"""
    rich_club_coefficient(net::FinancialNetwork, k_threshold)

Rich-club coefficient: fraction of edges among high-degree nodes.
phi(k) = 2*E_{>k} / (N_{>k} * (N_{>k} - 1))
"""
function rich_club_coefficient(net::FinancialNetwork, k_threshold::Int)::Float64
    N = net.num_nodes
    A = Float64.(net.adjacency .!= 0)
    degrees = [Int(sum(A[i, :])) for i in 1:N]

    rich_nodes = findall(degrees .>= k_threshold)
    n_rich = length(rich_nodes)
    if n_rich < 2
        return 0.0
    end

    edges_among_rich = 0
    for i in rich_nodes
        for j in rich_nodes
            if i < j && A[i, j] > 0
                edges_among_rich += 1
            end
        end
    end

    return 2.0 * edges_among_rich / (n_rich * (n_rich - 1))
end

"""
    network_entropy(net::FinancialNetwork)

Shannon entropy of degree distribution.
"""
function network_entropy(net::FinancialNetwork)::Float64
    N = net.num_nodes
    A = Float64.(net.adjacency .!= 0)
    degrees = [Int(sum(A[i, :])) for i in 1:N]

    max_deg = maximum(degrees)
    if max_deg == 0
        return 0.0
    end

    # Degree distribution
    counts = zeros(max_deg + 1)
    for d in degrees
        counts[d + 1] += 1.0
    end
    probs = counts / N

    H = 0.0
    for p in probs
        if p > 0
            H -= p * log(p)
        end
    end

    return H
end

"""
    small_world_coefficient(net::FinancialNetwork; num_random=10, seed=42)

Small-world coefficient: sigma = (C/C_rand) / (L/L_rand).
sigma > 1 indicates small-world network.
"""
function small_world_coefficient(net::FinancialNetwork;
                                  num_random::Int=10, seed::Int=42)::Float64
    rng = Random.MersenneTwister(seed)
    N = net.num_nodes
    A = Float64.(net.adjacency .!= 0)

    # Clustering coefficient of real network
    cc_real = mean(_clustering_coefficient(net))

    # Average path length of real network
    sp = shortest_path_distribution(net)
    L_real = sp.mean_path

    # Generate random networks with same degree sequence
    num_edges = Int(sum(A) / 2)
    cc_rand_sum = 0.0
    L_rand_sum = 0.0

    for trial in 1:num_random
        # Erdos-Renyi random graph with same density
        rand_adj = zeros(N, N)
        p_edge = 2.0 * num_edges / (N * (N - 1))
        for i in 1:N
            for j in (i+1):N
                if rand(rng) < p_edge
                    rand_adj[i, j] = 1.0
                    rand_adj[j, i] = 1.0
                end
            end
        end

        rand_net = FinancialNetwork(N, rand_adj, net.node_names, false)
        cc_rand_sum += mean(_clustering_coefficient(rand_net))
        sp_rand = shortest_path_distribution(rand_net)
        L_rand_sum += sp_rand.mean_path
    end

    cc_rand = cc_rand_sum / num_random
    L_rand = L_rand_sum / num_random

    if cc_rand < 1e-10 || L_real < 1e-10
        return 0.0
    end

    sigma = (cc_real / cc_rand) / (L_real / max(L_rand, 1e-10))
    return sigma
end

"""
    network_robustness(net::FinancialNetwork; attack_type=:targeted, fraction=0.1)

Measure network robustness to node removal.
attack_type: :random or :targeted (remove highest degree first).
Returns fraction of nodes in largest connected component after removal.
"""
function network_robustness(net::FinancialNetwork;
                             attack_type::Symbol=:targeted,
                             fraction::Float64=0.1)::Float64
    N = net.num_nodes
    A = copy(Float64.(net.adjacency .!= 0))
    num_remove = max(1, round(Int, fraction * N))

    if attack_type == :targeted
        degrees = [sum(A[i, :]) for i in 1:N]
        remove_order = sortperm(degrees, rev=true)
    else
        remove_order = randperm(N)
    end

    removed = falses(N)
    for i in 1:num_remove
        node = remove_order[i]
        removed[node] = true
        A[node, :] .= 0.0
        A[:, node] .= 0.0
    end

    # Find largest connected component via BFS
    visited = falses(N)
    max_component = 0

    for start in 1:N
        if visited[start] || removed[start]
            continue
        end

        # BFS
        queue = [start]
        visited[start] = true
        component_size = 0

        while !isempty(queue)
            v = popfirst!(queue)
            component_size += 1

            for w in 1:N
                if !visited[w] && !removed[w] && A[v, w] > 0
                    visited[w] = true
                    push!(queue, w)
                end
            end
        end

        max_component = max(max_component, component_size)
    end

    return max_component / (N - num_remove)
end

"""
    effective_resistance(net::FinancialNetwork, i, j)

Effective resistance between nodes i and j.
R_ij = (e_i - e_j)' * L^+ * (e_i - e_j)
where L^+ is pseudoinverse of Laplacian.
"""
function effective_resistance(net::FinancialNetwork, i::Int, j::Int)::Float64
    N = net.num_nodes
    A = abs.(net.adjacency)
    D = Diagonal(vec(sum(A, dims=2)))
    L = D - A

    # Pseudoinverse via eigendecomposition
    eig = eigen(Symmetric(Matrix(L)))
    L_pinv = zeros(N, N)
    for k in 1:N
        if eig.values[k] > 1e-10
            L_pinv += (1.0 / eig.values[k]) * eig.vectors[:, k] * eig.vectors[:, k]'
        end
    end

    e_ij = zeros(N)
    e_ij[i] = 1.0
    e_ij[j] = -1.0

    return dot(e_ij, L_pinv * e_ij)
end

"""
    total_effective_resistance(net::FinancialNetwork)

Kirchhoff index: sum of all pairwise effective resistances.
"""
function total_effective_resistance(net::FinancialNetwork)::Float64
    N = net.num_nodes
    A = abs.(net.adjacency)
    D = Diagonal(vec(sum(A, dims=2)))
    L = D - A

    eig = eigen(Symmetric(Matrix(L)))
    # Kirchhoff index = N * sum(1/lambda_i for nonzero eigenvalues)
    kirchhoff = 0.0
    for k in 1:N
        if eig.values[k] > 1e-10
            kirchhoff += N / eig.values[k]
        end
    end

    return kirchhoff
end

"""
    spectral_gap(net::FinancialNetwork)

Spectral gap: difference between two largest eigenvalues of adjacency.
"""
function spectral_gap(net::FinancialNetwork)::Float64
    A = abs.(net.adjacency)
    eig_vals = eigvals(Symmetric(A))
    sorted = sort(eig_vals, rev=true)
    if length(sorted) >= 2
        return sorted[1] - sorted[2]
    end
    return 0.0
end

"""
    algebraic_connectivity(net::FinancialNetwork)

Fiedler value: second-smallest eigenvalue of Laplacian.
Measures how well-connected the network is.
"""
function algebraic_connectivity(net::FinancialNetwork)::Float64
    N = net.num_nodes
    A = abs.(net.adjacency)
    D = Diagonal(vec(sum(A, dims=2)))
    L = D - A

    eig_vals = eigvals(Symmetric(Matrix(L)))
    sorted = sort(eig_vals)

    # Second smallest (first is ~0)
    for v in sorted
        if v > 1e-8
            return v
        end
    end

    return 0.0
end

"""
    network_flow_betweenness(net::FinancialNetwork)

Flow betweenness: based on maximum flow through each node.
Approximation using current-flow betweenness.
"""
function network_flow_betweenness(net::FinancialNetwork)::Vector{Float64}
    N = net.num_nodes
    A = abs.(net.adjacency)
    D = Diagonal(vec(sum(A, dims=2)))
    L = D - A

    # Pseudoinverse of Laplacian
    eig = eigen(Symmetric(Matrix(L)))
    L_pinv = zeros(N, N)
    for k in 1:N
        if eig.values[k] > 1e-10
            L_pinv += (1.0 / eig.values[k]) * eig.vectors[:, k] * eig.vectors[:, k]'
        end
    end

    fb = zeros(N)

    for s in 1:N
        for t in (s+1):N
            if s == t
                continue
            end

            # Current flow from s to t
            e_st = zeros(N)
            e_st[s] = 1.0
            e_st[t] = -1.0
            potentials = L_pinv * e_st

            # Current through each node
            for v in 1:N
                if v != s && v != t
                    throughput = 0.0
                    for w in 1:N
                        if A[v, w] > 0
                            current = A[v, w] * abs(potentials[v] - potentials[w])
                            throughput += current
                        end
                    end
                    fb[v] += throughput / 2.0
                end
            end
        end
    end

    # Normalize
    pairs = N * (N - 1) / 2.0
    if pairs > 0
        fb ./= pairs
    end

    return fb
end

"""
    link_prediction_scores(net::FinancialNetwork)

Link prediction: common neighbors, Jaccard, Adamic-Adar scores.
Returns matrix of scores for non-existing edges.
"""
function link_prediction_scores(net::FinancialNetwork)
    N = net.num_nodes
    A = Float64.(net.adjacency .!= 0)

    cn_scores = zeros(N, N)      # Common neighbors
    jaccard_scores = zeros(N, N)  # Jaccard
    aa_scores = zeros(N, N)       # Adamic-Adar

    degrees = [sum(A[i, :]) for i in 1:N]

    for i in 1:N
        neighbors_i = Set(findall(A[i, :] .> 0))
        for j in (i+1):N
            if A[i, j] == 0  # Only predict missing edges
                neighbors_j = Set(findall(A[j, :] .> 0))

                # Common neighbors
                common = intersect(neighbors_i, neighbors_j)
                cn_scores[i, j] = length(common)
                cn_scores[j, i] = cn_scores[i, j]

                # Jaccard
                union_size = length(union(neighbors_i, neighbors_j))
                if union_size > 0
                    jaccard_scores[i, j] = length(common) / union_size
                    jaccard_scores[j, i] = jaccard_scores[i, j]
                end

                # Adamic-Adar
                aa = 0.0
                for z in common
                    if degrees[z] > 1
                        aa += 1.0 / log(degrees[z])
                    end
                end
                aa_scores[i, j] = aa
                aa_scores[j, i] = aa
            end
        end
    end

    return (common_neighbors=cn_scores, jaccard=jaccard_scores, adamic_adar=aa_scores)
end

"""
    network_motif_census(net::FinancialNetwork)

Count 3-node motifs (triads) in directed network.
"""
function network_motif_census(net::FinancialNetwork)
    N = net.num_nodes
    A = Float64.(net.adjacency .!= 0)

    # Triad types for undirected: triangle, path, empty
    triangles = 0
    paths = 0
    empty_triads = 0

    for i in 1:N
        for j in (i+1):N
            for k in (j+1):N
                e_ij = A[i, j] > 0 || A[j, i] > 0
                e_jk = A[j, k] > 0 || A[k, j] > 0
                e_ik = A[i, k] > 0 || A[k, i] > 0

                num_edges = Int(e_ij) + Int(e_jk) + Int(e_ik)

                if num_edges == 3
                    triangles += 1
                elseif num_edges == 2
                    paths += 1
                elseif num_edges == 1
                    # Open triad with one edge
                elseif num_edges == 0
                    empty_triads += 1
                end
            end
        end
    end

    # Global clustering coefficient
    gcc = triangles > 0 || paths > 0 ?
          3.0 * triangles / (3.0 * triangles + paths) : 0.0

    return (triangles=triangles, paths=paths, empty_triads=empty_triads,
            global_clustering=gcc)
end

"""
    temporal_network_metrics(networks::Vector{FinancialNetwork})

Compute temporal metrics across network snapshots.
"""
function temporal_network_metrics(networks::Vector{FinancialNetwork})
    T_net = length(networks)
    if T_net < 2
        return (jaccard_similarity=Float64[], density_change=Float64[])
    end

    jaccard_sim = Vector{Float64}(undef, T_net - 1)
    density_change = Vector{Float64}(undef, T_net - 1)
    centrality_stability = Vector{Float64}(undef, T_net - 1)

    for t in 1:(T_net - 1)
        A1 = Float64.(networks[t].adjacency .!= 0)
        A2 = Float64.(networks[t+1].adjacency .!= 0)

        # Jaccard similarity of edge sets
        intersection = sum(A1 .> 0 .&& A2 .> 0)
        union_count = sum(A1 .> 0 .|| A2 .> 0)
        jaccard_sim[t] = union_count > 0 ? intersection / union_count : 1.0

        # Density change
        N = networks[t].num_nodes
        max_edges = N * (N - 1)
        d1 = sum(A1) / max_edges
        d2 = sum(A2) / max_edges
        density_change[t] = d2 - d1

        # Centrality stability (rank correlation of degree centrality)
        deg1 = degree_centrality(networks[t])
        deg2 = degree_centrality(networks[t+1])
        centrality_stability[t] = cor(deg1, deg2)
    end

    return (jaccard_similarity=jaccard_sim, density_change=density_change,
            centrality_stability=centrality_stability)
end

"""
    network_resilience_score(net::FinancialNetwork)

Composite resilience score based on multiple network properties.
"""
function network_resilience_score(net::FinancialNetwork)::Float64
    N = net.num_nodes

    # 1. Algebraic connectivity (higher = more resilient)
    ac = algebraic_connectivity(net)
    ac_score = min(ac / 1.0, 1.0)

    # 2. Inverse HHI of degree distribution (more equal = more resilient)
    degrees = degree_centrality(net)
    degree_hhi = sum(d^2 for d in degrees) / max(sum(degrees)^2, 1e-15) * N
    hhi_score = 1.0 - min(degree_hhi, 1.0)

    # 3. Clustering (moderate clustering is good)
    cc = mean(_clustering_coefficient(net))
    cc_score = 4.0 * cc * (1.0 - cc)  # Peaks at 0.5

    # 4. Robustness to targeted attack
    rob = network_robustness(net; attack_type=:targeted, fraction=0.1)
    rob_score = rob

    return 0.3 * ac_score + 0.2 * hhi_score + 0.2 * cc_score + 0.3 * rob_score
end

end # module NetworkFinance
