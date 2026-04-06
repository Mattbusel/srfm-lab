"""
topological_data.jl

Topological Data Analysis (TDA) for market microstructure.

TDA extracts shape features from data that are robust to noise and
coordinate changes. The core idea:
  1. Build a nested family of simplicial complexes (filtration)
  2. Track birth/death of topological features (connected components,
     loops, voids) as the scale parameter grows
  3. Features with long lifetimes are "real"; short ones are noise

Key invariants:
  H₀: connected components (clusters)
  H₁: 1-cycles / loops (market cycles, correlation rings)
  H₂: 2-cycles / voids (gaps in return space)

References:
  Edelsbrunner & Harer (2010) "Computational Topology"
  Carlsson (2009) "Topology and Data"
  Gidea & Katz (2018) "Topological Data Analysis of Financial Time Series"
"""

import Pkg
let _required = ["Distances", "Plots", "StatsBase"]
    _installed = keys(Pkg.project().dependencies)
    for pkg in _required
        pkg ∉ _installed && Pkg.add(pkg; io=devnull)
    end
end

using LinearAlgebra
using Statistics
using Random
using Distances
using Plots
using StatsBase

# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

"""
Compute pairwise distance matrix for a set of return vectors.
Supports: :euclidean, :correlation, :dtw (simplified), :cosine
"""
function distance_matrix(X::Matrix{Float64}; metric=:correlation)
    n = size(X, 1)
    D = zeros(n, n)

    if metric == :euclidean
        for i in 1:n, j in i+1:n
            d = norm(X[i,:] .- X[j,:])
            D[i,j] = d; D[j,i] = d
        end
    elseif metric == :correlation
        # Distance = sqrt(2(1 - ρ)) — standard financial TDA metric
        for i in 1:n, j in i+1:n
            ρ = cor(X[i,:], X[j,:])
            d = sqrt(2 * (1 - clamp(ρ, -1, 1)))
            D[i,j] = d; D[j,i] = d
        end
    elseif metric == :cosine
        for i in 1:n, j in i+1:n
            norm_i = norm(X[i,:]); norm_j = norm(X[j,:])
            if norm_i == 0 || norm_j == 0
                d = 1.0
            else
                cosine_sim = dot(X[i,:], X[j,:]) / (norm_i * norm_j)
                d = 1 - clamp(cosine_sim, -1, 1)
            end
            D[i,j] = d; D[j,i] = d
        end
    else
        error("Unknown metric: $metric")
    end

    return D
end

# ─────────────────────────────────────────────────────────────────────────────
# VIETORIS-RIPS COMPLEX
# ─────────────────────────────────────────────────────────────────────────────

"""
Build the Vietoris-Rips complex at scale ε from distance matrix D.

VR(ε): include k-simplex {v₀,...,vₖ} iff d(vᵢ,vⱼ) ≤ ε for all i,j.

Returns adjacency (edge) list at scale ε.
"""
struct RipsComplex
    n::Int                              # number of points
    filtration_values::Vector{Float64}  # sorted edge weights (births)
    edges::Vector{Tuple{Int,Int}}       # sorted edge list
    D::Matrix{Float64}                  # distance matrix
end

function build_rips_complex(D::Matrix{Float64})
    n = size(D, 1)
    edge_list = Tuple{Float64,Int,Int}[]

    for i in 1:n, j in i+1:n
        push!(edge_list, (D[i,j], i, j))
    end

    sort!(edge_list, by=x->x[1])

    filt_vals = [e[1] for e in edge_list]
    edges = [(e[2], e[3]) for e in edge_list]

    return RipsComplex(n, filt_vals, edges, D)
end

# ─────────────────────────────────────────────────────────────────────────────
# UNION-FIND (DISJOINT SET UNION) FOR H₀
# ─────────────────────────────────────────────────────────────────────────────

mutable struct UnionFind
    parent::Vector{Int}
    rank::Vector{Int}
    birth::Vector{Float64}  # birth time of each component
end

function UnionFind(n::Int)
    UnionFind(collect(1:n), zeros(Int, n), zeros(n))
end

function find!(uf::UnionFind, x::Int)
    if uf.parent[x] != x
        uf.parent[x] = find!(uf, uf.parent[x])  # path compression
    end
    return uf.parent[x]
end

function union!(uf::UnionFind, x::Int, y::Int, ε::Float64)
    rx, ry = find!(uf, x), find!(uf, y)
    rx == ry && return false, -1, -1  # already connected, creates cycle

    # Union by rank; younger component dies
    if uf.rank[rx] < uf.rank[ry]
        uf.parent[rx] = ry
        return true, rx, ry
    elseif uf.rank[rx] > uf.rank[ry]
        uf.parent[ry] = rx
        return true, ry, rx
    else
        uf.parent[ry] = rx
        uf.rank[rx] += 1
        return true, ry, rx
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT HOMOLOGY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Persistence pair: (birth, death, dimension).
A feature is born at scale `birth` and dies at `death`.
Lifetime = death - birth.
"""
struct PersistencePair
    birth::Float64
    death::Float64
    dim::Int
    generator_i::Int  # index of generator (vertex or edge)
    generator_j::Int
end

lifetime(p::PersistencePair) = p.death - p.birth

"""
Compute H₀ (connected components) via incremental union-find.

As ε increases, each edge {i,j} either:
  - Merges two components → kills a H₀ feature (death of younger component)
  - Creates a loop → births a H₁ feature

Returns H₀ persistence pairs.
"""
function compute_H0(rc::RipsComplex)
    uf = UnionFind(rc.n)
    pairs = PersistencePair[]

    # All components born at ε=0
    for v in 1:rc.n
        uf.birth[v] = 0.0
    end

    for (k, (i, j)) in enumerate(rc.edges)
        ε = rc.filtration_values[k]
        merged, dying_root, surviving_root = union!(uf, i, j, ε)

        if merged
            # Younger component (born later) dies
            b_i = uf.birth[find!(uf, i)]
            b_j = uf.birth[find!(uf, j)]
            # After merge, dying component is the one with later birth
            push!(pairs, PersistencePair(0.0, ε, 0, dying_root, 0))
        end
    end

    # Last remaining component lives forever (death = Inf)
    # Keep only the essential component
    return pairs
end

"""
Simplified persistent homology for H₁ (cycles) via boundary matrix reduction.

For H₁ in the Rips complex:
  - Each triangle {i,j,k} with d(i,j), d(j,k), d(i,k) ≤ ε kills a loop.
  - Loops are born when an edge creates a cycle.

This implements a simplified version tracking cycle birth/death.
Full implementation would use the persistence algorithm (Edelsbrunner et al. 2002).
"""
function compute_H1_simplified(rc::RipsComplex, D::Matrix{Float64};
                                 ε_max=Inf)
    n = rc.n
    uf = UnionFind(n)
    pairs_H0 = PersistencePair[]
    pairs_H1 = PersistencePair[]

    cycle_births = Dict{Int, Float64}()  # edge_idx → birth_ε for H₁
    cycle_gen    = Dict{Int, Tuple{Int,Int}}()

    for (k, (i, j)) in enumerate(rc.edges)
        ε = rc.filtration_values[k]
        ε > ε_max && break

        ri, rj = find!(uf, i), find!(uf, j)

        if ri == rj
            # Creates a cycle: H₁ born at ε
            cycle_births[k] = ε
            cycle_gen[k] = (i, j)
        else
            # Merges components: H₀ dies
            push!(pairs_H0, PersistencePair(0.0, ε, 0, i, j))
            union!(uf, i, j, ε)

            # Check if any existing cycle is killed by triangles
            # (simplified: kill cycles whose generators are now in same component)
            to_kill = Int[]
            for (ck, birth_ε) in cycle_births
                ci, cj = cycle_gen[ck]
                if find!(uf, ci) == find!(uf, cj)
                    push!(pairs_H1, PersistencePair(birth_ε, ε, 1, ci, cj))
                    push!(to_kill, ck)
                end
            end
            for ck in to_kill
                delete!(cycle_births, ck)
                delete!(cycle_gen, ck)
            end
        end
    end

    # Surviving cycles live to ε_max
    ε_end = isempty(rc.filtration_values) ? 1.0 : rc.filtration_values[end]
    for (ck, birth_ε) in cycle_births
        ci, cj = cycle_gen[ck]
        push!(pairs_H1, PersistencePair(birth_ε, ε_end, 1, ci, cj))
    end

    return pairs_H0, pairs_H1
end

"""
Compute Betti numbers β₀, β₁ as a function of ε.
β₀ = number of connected components
β₁ = number of independent cycles
"""
function betti_numbers(rc::RipsComplex; ε_range=nothing, n_steps=100)
    ε_max = isnothing(ε_range) ? rc.filtration_values[end] : ε_range[2]
    ε_min = isnothing(ε_range) ? 0.0 : ε_range[1]
    ε_grid = range(ε_min, ε_max, length=n_steps) |> collect

    β0 = zeros(Int, n_steps)
    β1 = zeros(Int, n_steps)

    pairs_H0, pairs_H1 = compute_H1_simplified(rc, rc.D)

    for (k, ε) in enumerate(ε_grid)
        # β₀: count components alive at ε
        β0[k] = rc.n - count(p -> p.birth <= ε && p.death <= ε, pairs_H0)
        β0[k] = max(β0[k], 1)  # at least one component

        # β₁: count cycles alive at ε
        β1[k] = count(p -> p.birth <= ε && p.death > ε, pairs_H1)
    end

    return ε_grid, β0, β1
end

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────

"""
Persistence landscape λₖ(t): a functional summary of the persistence diagram.

For each persistence pair (b, d), define a tent function:
    f_{b,d}(t) = max(0, min(t-b, d-t))

The k-th landscape is the k-th largest tent value at each t:
    λₖ(t) = k-th max over all pairs of f_{b,d}(t)

Landscapes live in L² and allow statistical testing.
"""
function persistence_landscape(pairs::Vector{PersistencePair};
                                n_points=200, t_min=nothing, t_max=nothing,
                                n_landscapes=3)
    isempty(pairs) && return zeros(n_points, n_landscapes), Float64[]

    t_lo = isnothing(t_min) ? minimum(p.birth for p in pairs) : t_min
    t_hi = isnothing(t_max) ? maximum(p.death for p in pairs if isfinite(p.death)) : t_max

    t_grid = range(t_lo, t_hi, length=n_points) |> collect
    landscapes = zeros(n_points, n_landscapes)

    for (k, t) in enumerate(t_grid)
        # Evaluate tent function for each pair
        tent_vals = [max(0.0, min(t - p.birth, p.death - t)) for p in pairs
                     if isfinite(p.death)]
        sort!(tent_vals, rev=true)

        for l in 1:min(n_landscapes, length(tent_vals))
            landscapes[k, l] = tent_vals[l]
        end
    end

    return landscapes, t_grid
end

"""
L² norm of persistence landscape (summary statistic).
"""
function landscape_norm(landscape::Matrix{Float64}, t_grid::Vector{Float64})
    n_landscapes = size(landscape, 2)
    norms = zeros(n_landscapes)
    dt = diff(t_grid)[1]
    for l in 1:n_landscapes
        norms[l] = sqrt(sum(landscape[:,l].^2) * dt)
    end
    return norms
end

# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK AND WASSERSTEIN DISTANCE BETWEEN DIAGRAMS
# ─────────────────────────────────────────────────────────────────────────────

"""
Wasserstein distance between two persistence diagrams.

d_W(Dgm₁, Dgm₂) = inf_{γ: Dgm₁→Dgm₂} (Σ ||p - γ(p)||_∞^q)^{1/q}

The infimum is over partial matchings; unmatched points are matched to
the diagonal (birth = death), with cost equal to half-lifetime.

Implemented via linear assignment (Hungarian algorithm approximation).
"""
function diagram_wasserstein(pairs1::Vector{PersistencePair},
                              pairs2::Vector{PersistencePair};
                              q=2, dim=nothing)

    # Filter by dimension if requested
    p1 = isnothing(dim) ? pairs1 : filter(p -> p.dim == dim, pairs1)
    p2 = isnothing(dim) ? pairs2 : filter(p -> p.dim == dim, pairs2)

    # Remove infinite bars for distance computation
    p1 = filter(p -> isfinite(p.death), p1)
    p2 = filter(p -> isfinite(p.death), p2)

    n, m = length(p1), length(p2)

    # Points on diagram as (birth, death)
    pts1 = [(p.birth, p.death) for p in p1]
    pts2 = [(p.birth, p.death) for p in p2]

    # Diagonal projections: Δ(b,d) = ((b+d)/2, (b+d)/2)
    diag1 = [((b+d)/2, (b+d)/2) for (b,d) in pts1]
    diag2 = [((b+d)/2, (b+d)/2) for (b,d) in pts2]

    # Cost between two diagram points (L∞ norm)
    cost(a, b) = max(abs(a[1] - b[1]), abs(a[2] - b[2]))

    # Build augmented cost matrix (n+m) × (n+m)
    N = n + m
    C = zeros(N, N)

    for i in 1:n, j in 1:m
        C[i,j] = cost(pts1[i], pts2[j])
    end
    # Unmatched in p1 → diagonal of p2
    for i in 1:n, j in m+1:N
        C[i,j] = cost(pts1[i], diag1[i])
    end
    # Unmatched in p2 → diagonal of p1
    for i in n+1:N, j in 1:m
        C[i,j] = cost(diag2[j-n], pts2[j-n])
    end

    # Greedy matching (exact would use Hungarian algorithm)
    used_row = falses(N)
    used_col = falses(N)
    total_cost = 0.0
    pairs_matched = 0

    for _ in 1:N
        best_val = Inf
        best_r, best_c = -1, -1
        for r in 1:N, c in 1:N
            !used_row[r] && !used_col[c] && C[r,c] < best_val || continue
            best_val = C[r,c]
            best_r, best_c = r, c
        end
        best_r == -1 && break
        total_cost += best_val^q
        used_row[best_r] = true
        used_col[best_c] = true
        pairs_matched += 1
    end

    return total_cost^(1/q)
end

"""
Bottleneck distance between diagrams:
d_B(Dgm₁, Dgm₂) = inf_{γ} sup_p ||p - γ(p)||_∞
"""
function diagram_bottleneck(pairs1::Vector{PersistencePair},
                             pairs2::Vector{PersistencePair}; dim=nothing)
    p1 = isnothing(dim) ? pairs1 : filter(p -> p.dim == dim, pairs1)
    p2 = isnothing(dim) ? pairs2 : filter(p -> p.dim == dim, pairs2)
    p1 = filter(p -> isfinite(p.death), p1)
    p2 = filter(p -> isfinite(p.death), p2)

    isempty(p1) && isempty(p2) && return 0.0

    # Binary search on bottleneck distance
    all_distances = Float64[]
    for a in p1, b in p2
        push!(all_distances, max(abs(a.birth - b.birth), abs(a.death - b.death)))
    end
    # Distance to diagonal
    for a in p1
        push!(all_distances, (a.death - a.birth) / 2)
    end
    for b in p2
        push!(all_distances, (b.death - b.birth) / 2)
    end

    sort!(unique!(all_distances))

    # Binary search: smallest δ such that a δ-matching exists
    lo, hi = 0, length(all_distances)
    while lo < hi
        mid = (lo + hi) ÷ 2
        δ = all_distances[mid + 1]
        if has_matching(p1, p2, δ)
            hi = mid
        else
            lo = mid + 1
        end
    end

    return lo < length(all_distances) ? all_distances[lo + 1] : all_distances[end]
end

function has_matching(p1, p2, δ)
    # Check if bipartite matching exists with all edges ≤ δ
    # (simplified greedy check)
    used = falses(length(p2))
    for a in p1
        matched = false
        # Try to match to diagonal
        if (a.death - a.birth) / 2 <= δ
            matched = true
        else
            # Try to match to some b in p2
            for (j, b) in enumerate(p2)
                !used[j] || continue
                if max(abs(a.birth - b.birth), abs(a.death - b.death)) <= δ
                    used[j] = true
                    matched = true
                    break
                end
            end
        end
        matched || return false
    end
    return true
end

# ─────────────────────────────────────────────────────────────────────────────
# MARKET TOPOLOGY: CORRELATION SPACE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

"""
Analyse structural holes in the asset correlation space using TDA.

Assets are modelled as points in return space. Clusters (H₀) represent
groups of correlated assets; loops (H₁) represent cyclic dependencies.

Returns persistence diagram and Betti numbers.
"""
function market_topology(returns_matrix::Matrix{Float64}, asset_names::Vector{String};
                          metric=:correlation, ε_max=2.0)
    n_assets, T = size(returns_matrix)
    println("Market Topology Analysis: $n_assets assets × $T time steps")

    # Distance matrix
    D = distance_matrix(returns_matrix'; metric=metric)

    # Build Rips filtration
    rc = build_rips_complex(D)

    # Compute persistence
    pairs_H0, pairs_H1 = compute_H1_simplified(rc, D; ε_max=ε_max)

    println("\nH₀ pairs (connected components): $(length(pairs_H0))")
    println("H₁ pairs (cycles): $(length(pairs_H1))")

    # Significant features
    threshold = median([lifetime(p) for p in pairs_H1 if isfinite(lifetime(p))]) * 2.0
    sig_H1 = filter(p -> isfinite(lifetime(p)) && lifetime(p) > threshold, pairs_H1)
    println("Significant H₁ features (lifetime > 2×median): $(length(sig_H1))")

    # Betti numbers
    ε_vals, β0, β1 = betti_numbers(rc; n_steps=200)

    return rc, pairs_H0, pairs_H1, ε_vals, β0, β1, D
end

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL TDA: SLIDING WINDOW
# ─────────────────────────────────────────────────────────────────────────────

"""
Temporal TDA: compute persistence over a sliding window of a 1D time series.

Embeds a time series x(t) into a point cloud using delay embedding:
    Xᵢ = (x(i), x(i+τ), x(i+2τ), ..., x(i+(d-1)τ))

Then computes TDA on the point cloud. Loops in H₁ indicate periodic structure.
"""
function temporal_tda(series::Vector{Float64};
                       window=50, step=10, delay=1, embed_dim=3,
                       ε_max=3.0)

    T = length(series)
    n_windows = floor(Int, (T - window) / step) + 1

    β1_sequence = Float64[]
    landscape_norms = Float64[]

    for w in 1:n_windows
        t_start = (w-1) * step + 1
        t_end   = t_start + window - 1
        t_end > T && break

        seg = series[t_start:t_end]

        # Delay embedding
        n_pts = window - (embed_dim - 1) * delay
        n_pts <= 0 && continue

        X = zeros(n_pts, embed_dim)
        for i in 1:n_pts, d in 1:embed_dim
            X[i,d] = seg[i + (d-1)*delay]
        end

        # Normalise
        X .-= mean(X, dims=1)
        s = std(X[:])
        s > 0 && (X ./= s)

        # TDA
        D = distance_matrix(X; metric=:euclidean)
        rc = build_rips_complex(D)
        _, pairs_H1 = compute_H1_simplified(rc, D; ε_max=ε_max)

        # H₁ Betti number at midpoint ε
        ε_mid = ε_max / 2
        b1 = count(p -> p.birth <= ε_mid && p.death > ε_mid, pairs_H1)
        push!(β1_sequence, Float64(b1))

        # Landscape norm
        ls, t_grid = persistence_landscape(pairs_H1)
        push!(landscape_norms, sum(landscape_norm(ls, t_grid)))
    end

    return β1_sequence, landscape_norms
end

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Persistence diagram: scatter plot of (birth, death) pairs.
Points far from the diagonal have long lifetimes and are significant.
"""
function plot_persistence_diagram(pairs_H0::Vector{PersistencePair},
                                   pairs_H1::Vector{PersistencePair};
                                   title="Persistence Diagram")
    max_val = 0.0
    for p in vcat(pairs_H0, pairs_H1)
        isfinite(p.death) && (max_val = max(max_val, p.death))
    end
    max_val = max(max_val, 1.0)

    p = plot([0, max_val], [0, max_val], color=:black, linestyle=:dash,
             label="Diagonal", xlabel="Birth", ylabel="Death", title=title)

    # H₀ points
    b0 = [pair.birth for pair in pairs_H0 if isfinite(pair.death)]
    d0 = [pair.death for pair in pairs_H0 if isfinite(pair.death)]
    isempty(b0) || scatter!(p, b0, d0, color=:blue, markersize=6, label="H₀ (components)")

    # H₁ points
    b1 = [pair.birth for pair in pairs_H1 if isfinite(pair.death)]
    d1 = [pair.death for pair in pairs_H1 if isfinite(pair.death)]
    isempty(b1) || scatter!(p, b1, d1, color=:red, markershape=:diamond,
                             markersize=6, label="H₁ (cycles)")

    return p
end

"""
Barcode plot: horizontal bars for each persistence pair.
"""
function plot_barcode(pairs_H0::Vector{PersistencePair},
                      pairs_H1::Vector{PersistencePair};
                      ε_max=nothing, title="Persistence Barcode")
    all_pairs = vcat(
        [(p, 0) for p in pairs_H0 if isfinite(p.death)],
        [(p, 1) for p in pairs_H1]
    )
    sort!(all_pairs, by=x -> -lifetime(x[1]))

    max_ε = isnothing(ε_max) ? maximum(p.death for (p,_) in all_pairs if isfinite(p.death)) : ε_max

    p = plot(title=title, xlabel="ε (filtration parameter)", ylabel="Feature index",
             yticks=nothing, legend=:topright)

    colors = Dict(0 => :blue, 1 => :red)
    labels_used = Set{Int}()

    for (k, (pair, dim)) in enumerate(all_pairs)
        death = isfinite(pair.death) ? pair.death : max_ε * 1.05
        lbl = dim ∈ labels_used ? nothing : "H$dim"
        plot!(p, [pair.birth, death], [k, k], color=colors[dim],
              linewidth=2, label=lbl, alpha=0.8)
        push!(labels_used, dim)
    end

    return p
end

"""
Betti number curves.
"""
function plot_betti_curves(ε_vals, β0, β1; title="Betti Numbers")
    p = plot(ε_vals, β0, label="β₀ (components)", color=:blue, linewidth=2,
             xlabel="ε", ylabel="Betti number", title=title)
    plot!(p, ε_vals, β1, label="β₁ (cycles)", color=:red, linewidth=2, linestyle=:dash)
    return p
end

"""
Plot persistence landscape.
"""
function plot_landscape(landscapes, t_grid; title="Persistence Landscape")
    p = plot(title=title, xlabel="t", ylabel="λ(t)")
    colors_ls = [:blue, :red, :green]
    for l in 1:size(landscapes, 2)
        plot!(p, t_grid, landscapes[:,l], label="λ$l",
              color=colors_ls[min(l, length(colors_ls))], linewidth=1.5)
    end
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

function demo()
    Random.seed!(42)
    println("=" ^ 60)
    println("Topological Data Analysis Demo")
    println("=" ^ 60)

    # 1. Simple point cloud: circle (should have β₁ = 1)
    println("\n1. TDA of circular point cloud (expect β₁ = 1)")
    n_pts = 30
    θ = range(0, 2π, length=n_pts+1)[1:n_pts]
    circle = hcat(cos.(θ) .+ 0.1 .* randn(n_pts),
                  sin.(θ) .+ 0.1 .* randn(n_pts))

    D_circle = distance_matrix(circle; metric=:euclidean)
    rc_circle = build_rips_complex(D_circle)
    pairs_H0_c, pairs_H1_c = compute_H1_simplified(rc_circle, D_circle)

    println("  H₀ pairs: $(length(pairs_H0_c))")
    println("  H₁ pairs: $(length(pairs_H1_c))")

    ε_vals_c, β0_c, β1_c = betti_numbers(rc_circle; n_steps=100)
    max_β1 = maximum(β1_c)
    println("  Max β₁ over filtration: $max_β1 (expect ≥ 1)")

    # 2. Synthetic asset correlation TDA
    println("\n2. Market Topology: Synthetic asset returns")
    n_assets = 8
    T_steps  = 252

    # Create 3 clusters of correlated assets
    factor1 = randn(T_steps)
    factor2 = randn(T_steps)
    factor3 = randn(T_steps)

    R = zeros(n_assets, T_steps)
    for i in 1:3;   R[i,:] = 0.7 * factor1 + 0.3 * randn(T_steps); end
    for i in 4:6;   R[i,:] = 0.7 * factor2 + 0.3 * randn(T_steps); end
    for i in 7:n_assets; R[i,:] = 0.7 * factor3 + 0.3 * randn(T_steps); end

    asset_names = ["A$i" for i in 1:n_assets]
    rc_mkt, pairs_H0_m, pairs_H1_m, ε_vals_m, β0_m, β1_m, D_mkt =
        market_topology(R, asset_names; ε_max=2.0)

    # 3. Temporal TDA on synthetic BH signal
    println("\n3. Temporal TDA on a trending signal")
    t = range(0, 4π, length=200)
    signal = sin.(t) .+ 0.5 .* sin.(2 .* t) .+ 0.2 .* randn(200)
    β1_seq, ls_norms = temporal_tda(signal; window=40, step=5, embed_dim=2)
    println("  Mean β₁ over windows: $(round(mean(β1_seq), digits=2))")

    # 4. Diagram distances
    println("\n4. Persistence Diagram Distances")
    # Diagram 1: circle
    # Diagram 2: slightly noisier circle
    θ2 = range(0, 2π, length=n_pts+1)[1:n_pts]
    circle2 = hcat(cos.(θ2) .+ 0.2 .* randn(n_pts), sin.(θ2) .+ 0.2 .* randn(n_pts))
    D2 = distance_matrix(circle2; metric=:euclidean)
    rc2 = build_rips_complex(D2)
    _, pairs_H1_2 = compute_H1_simplified(rc2, D2)

    d_W = diagram_wasserstein(pairs_H1_c, pairs_H1_2; q=2)
    d_B = diagram_bottleneck(pairs_H1_c, pairs_H1_2)
    println("  Wasserstein distance between H₁ diagrams: $(round(d_W, digits=4))")
    println("  Bottleneck distance between H₁ diagrams:  $(round(d_B, digits=4))")

    # 5. Persistence landscape
    println("\n5. Persistence Landscape")
    ls, t_grid_ls = persistence_landscape(pairs_H1_c; n_points=100)
    norms = landscape_norm(ls, t_grid_ls)
    println("  Landscape L² norms: $(round.(norms, digits=4))")

    # Plots
    p_diag = plot_persistence_diagram(pairs_H0_c, pairs_H1_c;
                                       title="Circle Persistence Diagram")
    p_bar  = plot_barcode(pairs_H0_c, pairs_H1_c; title="Circle Barcode")
    p_bet  = plot_betti_curves(ε_vals_c, β0_c, β1_c; title="Betti Curves: Circle")
    p_land = plot_landscape(ls, t_grid_ls; title="H₁ Persistence Landscape")

    full_plot = plot(p_diag, p_bar, p_bet, p_land,
                     layout=(2,2), size=(1100, 800))
    savefig(full_plot, "tda_demo.png")
    println("\nSaved tda_demo.png")

    p_mkt_diag = plot_persistence_diagram(pairs_H0_m, pairs_H1_m;
                                           title="Market Topology: Persistence Diagram")
    p_mkt_bet  = plot_betti_curves(ε_vals_m, β0_m, β1_m;
                                    title="Market Topology: Betti Curves")
    mkt_plot = plot(p_mkt_diag, p_mkt_bet, layout=(1,2), size=(900, 400))
    savefig(mkt_plot, "tda_market.png")
    println("Saved tda_market.png")

    return pairs_H0_c, pairs_H1_c
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
