# =============================================================================
# causal_fast.jl — Fast Causal Discovery in Julia
# =============================================================================
# Provides:
#   - GrangerCausalityMatrix  (pairwise Granger tests, parallel)
#   - TransferEntropy         (non-linear causality measure)
#   - PCAlgorithm             (PC causal graph discovery)
#   - FCI                     (FCI algorithm for hidden confounders)
#
# All methods output adjacency matrices / edge lists serialisable as JSON.
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, JSON3
# =============================================================================

module CausalFast

using Statistics
using LinearAlgebra
using JSON3
using Base.Threads

export GrangerCausalityMatrix, TransferEntropy, PCAlgorithm, FCI
export CausalGraph, EdgeType

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"""
Edge types for the causal graph.
"""
@enum EdgeType begin
    NO_EDGE    = 0
    DIRECTED   = 1    # X → Y
    REVERSED   = 2    # X ← Y
    UNDIRECTED = 3    # X — Y  (PC skeleton)
    BIDIRECTED = 4    # X ↔ Y  (hidden confounder, FCI)
    CIRCLE     = 5    # X ∘—∘ Y (FCI partial ancestral graph)
end

"""
Causal graph: stores the adjacency matrix and metadata.
"""
struct CausalGraph
    n_vars      ::Int
    var_names   ::Vector{String}
    adj_matrix  ::Matrix{Int}      # EdgeType values (Int for JSON serialisation)
    pvalue_matrix ::Matrix{Float64}
    method      ::String
    metadata    ::Dict{String, Any}
end

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

"""
OLS residuals for regression y ~ X (no intercept assumed if intercept col included).
"""
function _ols_residuals(y::Vector{Float64}, X::Matrix{Float64})
    β = (X'X) \ (X'y)
    y .- X * β
end

"""
F-test for the joint significance of added regressors.
Compares restricted model (X_r) to unrestricted (X_u = [X_r, X_extra]).
Returns (F_stat, p_value).
"""
function _f_test(y::Vector{Float64},
                  X_r::Matrix{Float64},
                  X_u::Matrix{Float64})
    n    = length(y)
    k_r  = size(X_r, 2)
    k_u  = size(X_u, 2)
    q    = k_u - k_r

    r_r  = _ols_residuals(y, X_r)
    r_u  = _ols_residuals(y, X_u)

    RSS_r = sum(r_r.^2)
    RSS_u = sum(r_u.^2)

    if RSS_u < 1e-14
        return (Inf, 0.0)
    end

    F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k_u))

    # p-value via F-distribution CDF approximation (incomplete beta)
    p = _f_pvalue(F, q, n - k_u)
    (F, p)
end

"""
F-distribution p-value via regularised incomplete beta function (Abramowitz & Stegun).
P(F > f) = I_{x}(d2/2, d1/2) where x = d2 / (d2 + d1*f)
"""
function _f_pvalue(f::Float64, d1::Int, d2::Int)
    f <= 0.0 && return 1.0
    x  = Float64(d2) / (Float64(d2) + Float64(d1) * f)
    _incomplete_beta_regularised(x, d2 / 2.0, d1 / 2.0)
end

"""
Regularised incomplete beta function via continued fraction (Lentz method).
"""
function _incomplete_beta_regularised(x::Float64, a::Float64, b::Float64)
    (x < 0.0 || x > 1.0) && error("x must be in [0,1]")
    x == 0.0 && return 0.0
    x == 1.0 && return 1.0

    # Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0)
        return 1.0 - _incomplete_beta_regularised(1.0 - x, b, a)
    end

    # Log beta function via log-gamma (Stirling approximation)
    log_beta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front    = exp(log(x) * a + log(1.0 - x) * b - log_beta) / a

    # Continued fraction expansion (max 200 iterations)
    f = 1.0; C = 1.0; D = 1.0 - (a + b) * x / (a + 1.0)
    D = abs(D) < 1e-30 ? 1e-30 : 1.0 / D
    f = D

    for m in 1:200
        m_f = Float64(m)

        # Even step
        numerator = m_f * (b - m_f) * x / ((a + 2m_f - 1.0) * (a + 2m_f))
        D = 1.0 + numerator * D
        C = 1.0 + numerator / C
        D = abs(D) < 1e-30 ? 1e-30 : 1.0 / D
        delta = C * D; f *= delta

        # Odd step
        numerator = -(a + m_f) * (a + b + m_f) * x / ((a + 2m_f) * (a + 2m_f + 1.0))
        D = 1.0 + numerator * D
        C = 1.0 + numerator / C
        D = abs(D) < 1e-30 ? 1e-30 : 1.0 / D
        delta = C * D; f *= delta

        abs(delta - 1.0) < 1e-12 && break
    end

    front * f
end

"""
Log-gamma via Lanczos approximation.
"""
function _lgamma(z::Float64)
    g    = 7.0
    c    = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    z -= 1.0
    x   = c[1]
    for i in 2:9
        x += c[i] / (z + Float64(i - 1))
    end
    t = z + g + 0.5
    0.5 * log(2π) + (z + 0.5) * log(t) - t + log(x)
end

"""
Partial correlation between x and y given conditioning set Z.
Uses recursive formula. Returns (r, p_value).
"""
function _partial_correlation(x::Vector{Float64}, y::Vector{Float64},
                                Z::Matrix{Float64})
    if size(Z, 2) == 0
        return _pearson_correlation(x, y)
    end

    # Regress x and y on Z, take correlation of residuals
    Z_design = hcat(ones(length(x)), Z)
    rx = _ols_residuals(x, Z_design)
    ry = _ols_residuals(y, Z_design)
    _pearson_correlation(rx, ry)
end

"""
Pearson correlation with t-test p-value.
Returns (r, p_value).
"""
function _pearson_correlation(x::Vector{Float64}, y::Vector{Float64})
    n  = length(x)
    r  = cor(x, y)
    isnan(r) && return (0.0, 1.0)
    r  = clamp(r, -1.0 + 1e-10, 1.0 - 1e-10)
    t  = r * sqrt(n - 2) / sqrt(1 - r^2)
    # Two-sided t-test: p = 2 * P(T > |t|) ≈ 2 * _f_pvalue(t^2, 1, n-2)
    p  = _f_pvalue(t^2, 1, n - 2)
    (r, min(1.0, p))
end

# ---------------------------------------------------------------------------
# Granger Causality Matrix
# ---------------------------------------------------------------------------

"""
Pairwise Granger causality test for all variable pairs.

For each pair (X → Y): fit VAR(max_lag) on Y with and without lags of X,
then F-test the null that X's lags add no predictive power.

# Arguments
- `data_matrix` : n × m matrix (rows = time, cols = variables)
- `max_lag`     : maximum lag order to include
- `alpha`       : significance level

# Returns
CausalGraph with DIRECTED edges where X Granger-causes Y at level alpha.
"""
function GrangerCausalityMatrix(data_matrix::AbstractMatrix{<:Real};
                                  max_lag::Int=5,
                                  alpha::Float64=0.05,
                                  var_names::Vector{String}=String[])
    n, m = size(data_matrix)
    n > max_lag + 5 || error("Need more observations than max_lag")

    names = isempty(var_names) ? ["X$i" for i in 1:m] : var_names
    X     = Float64.(data_matrix)

    adj   = zeros(Int,     m, m)
    pvals = ones(Float64,  m, m)

    # Build lagged design matrix for a single variable
    function lagged_cols(col_idx::Int, lags::Int)
        [X[max_lag+1-lag : n-lag, col_idx] for lag in 1:lags]
    end

    # Parallel execution over all pairs
    results = Vector{Tuple{Int,Int,Float64,Bool}}(undef, m*m)
    pair_idx = 1
    pairs = [(i, j) for i in 1:m, j in 1:m if i != j]

    Threads.@threads for pair in pairs
        i, j = pair
        y    = X[max_lag+1:n, j]

        # Restricted: Y lags only
        r_cols = lagged_cols(j, max_lag)
        X_r    = hcat(ones(length(y)), r_cols...)

        # Unrestricted: Y lags + X lags
        x_cols = lagged_cols(i, max_lag)
        X_u    = hcat(X_r, x_cols...)

        F_stat, p_val = _f_test(y, X_r, X_u)

        # Thread-safe write via atomic or sequential (use lock for dict-of-results)
        # For simplicity use per-(i,j) storage
        pvals[i, j] = p_val
        adj[i, j]   = p_val < alpha ? Int(DIRECTED) : Int(NO_EDGE)
    end

    CausalGraph(
        m, names, adj, pvals,
        "granger",
        Dict{String, Any}("max_lag" => max_lag, "alpha" => alpha)
    )
end

# ---------------------------------------------------------------------------
# Transfer Entropy
# ---------------------------------------------------------------------------

"""
Compute Transfer Entropy from X to Y.

TE(X→Y) = H(Yₜ | Yₜ₋₁) − H(Yₜ | Yₜ₋₁, Xₜ₋ₗₐg)

Uses a kernel density / symbolisation approach.  For computational tractability
we discretise into n_bins bins and use plug-in entropy estimation.

# Returns
NamedTuple: (te_xy, te_yx, net_te, dominant_direction)
"""
function TransferEntropy(x::AbstractVector{<:Real},
                          y::AbstractVector{<:Real};
                          lag::Int=1,
                          n_bins::Int=10)
    n = length(x)
    n == length(y) || error("x and y must have the same length")
    n > lag + 10   || error("Series too short for Transfer Entropy at lag=$lag")

    xf = Float64.(x)
    yf = Float64.(y)

    # Symbolise via equal-frequency binning
    function symbolise(v::Vector{Float64}, k::Int)
        sorted = sort(v)
        n_v    = length(v)
        breaks = [sorted[max(1, round(Int, i*n_v/k))] for i in 0:k]
        breaks[end] = Inf
        [searchsortedfirst(breaks, vi) - 1 for vi in v] .|> v -> clamp(v, 1, k)
    end

    sx = symbolise(xf, n_bins)
    sy = symbolise(yf, n_bins)

    # Build symbol sequences with lag
    y_t    = sy[lag+1:n]
    y_lag  = sy[1:n-lag]
    x_lag  = sx[1:n-lag]

    n2 = length(y_t)

    # Plug-in entropy estimator
    function entropy1(v)
        cnt = Dict{Int, Int}()
        for vi in v; cnt[vi] = get(cnt, vi, 0) + 1; end
        -sum(c/n2 * log2(c/n2) for c in values(cnt))
    end

    function entropy2(v1, v2)
        cnt = Dict{Tuple{Int,Int}, Int}()
        for i in 1:n2
            k_ = (v1[i], v2[i])
            cnt[k_] = get(cnt, k_, 0) + 1
        end
        -sum(c/n2 * log2(c/n2) for c in values(cnt))
    end

    function entropy3(v1, v2, v3)
        cnt = Dict{Tuple{Int,Int,Int}, Int}()
        for i in 1:n2
            k_ = (v1[i], v2[i], v3[i])
            cnt[k_] = get(cnt, k_, 0) + 1
        end
        -sum(c/n2 * log2(c/n2) for c in values(cnt))
    end

    # TE(X→Y) = H(Yt, Yt-1) + H(Yt-1, Xt-1) − H(Yt-1) − H(Yt, Yt-1, Xt-1)
    H_yt_yl   = entropy2(y_t, y_lag)
    H_yl_xl   = entropy2(y_lag, x_lag)
    H_yl      = entropy1(y_lag)
    H_yt_yl_xl = entropy3(y_t, y_lag, x_lag)

    te_xy = H_yt_yl + H_yl_xl - H_yl - H_yt_yl_xl

    # TE(Y→X) (reverse direction)
    x_t   = sx[lag+1:n]
    H_xt_xl   = entropy2(x_t, x_lag)
    H_xl_yl   = entropy2(x_lag, y_lag)
    H_xl      = entropy1(x_lag)
    H_xt_xl_yl = entropy3(x_t, x_lag, y_lag)

    te_yx = H_xt_xl + H_xl_yl - H_xl - H_xt_xl_yl

    net_te = te_xy - te_yx

    dom = if abs(net_te) < 0.01
        "BIDIRECTIONAL"
    elseif net_te > 0
        "X→Y"
    else
        "Y→X"
    end

    (
        te_xy               = max(0.0, te_xy),
        te_yx               = max(0.0, te_yx),
        net_te              = net_te,
        dominant_direction  = dom,
        lag                 = lag,
        n_bins              = n_bins
    )
end

# ---------------------------------------------------------------------------
# PC Algorithm
# ---------------------------------------------------------------------------

"""
PC algorithm for causal graph discovery from observational data.

Phases:
  1. Skeleton discovery via conditional independence tests (partial correlation)
  2. V-structure (collider) orientation
  3. Remaining edge orientation via Meek rules

# Arguments
- `data_matrix` : n × m matrix
- `alpha`        : significance level for independence tests
- `max_cond_set` : maximum conditioning set size (default 3)

# Returns CausalGraph
"""
function PCAlgorithm(data_matrix::AbstractMatrix{<:Real};
                      alpha::Float64=0.05,
                      max_cond_set::Int=3,
                      var_names::Vector{String}=String[])
    n, m = size(data_matrix)
    names = isempty(var_names) ? ["X$i" for i in 1:m] : var_names
    X     = Float64.(data_matrix)
    # Standardise
    for j in 1:m
        μ = mean(X[:, j]); σ = std(X[:, j])
        σ > 1e-12 && (X[:, j] = (X[:, j] .- μ) ./ σ)
    end

    # Separation sets: sep_set[i,j] = conditioning set making i⊥j
    sep_set = Dict{Tuple{Int,Int}, Vector{Int}}()

    # Phase 1: Build skeleton (complete undirected graph, then remove edges)
    adj_undirected = trues(m, m)
    for i in 1:m; adj_undirected[i, i] = false; end

    pval_matrix = ones(Float64, m, m)

    cond_set_size = 0
    while cond_set_size <= max_cond_set
        changed = false
        for i in 1:m, j in (i+1):m
            !adj_undirected[i, j] && continue

            # Get adjacency-set neighbours of i (excluding j)
            adj_i = [k for k in 1:m if k != i && k != j && adj_undirected[i, k]]

            length(adj_i) < cond_set_size && continue

            # Test all conditioning sets of the given size
            removed = false
            for cond in _combinations(adj_i, cond_set_size)
                Z    = X[:, cond]
                r, p = _partial_correlation(X[:, i], X[:, j],
                                             size(Z, 2) > 0 ? Z : Matrix{Float64}(undef, n, 0))
                pval_matrix[i, j] = pval_matrix[j, i] = p

                if p >= alpha
                    adj_undirected[i, j] = adj_undirected[j, i] = false
                    sep_set[(i,j)] = sep_set[(j,i)] = collect(cond)
                    removed = true
                    changed = true
                    break
                end
            end
            removed && break
        end
        cond_set_size += 1
        !changed && cond_set_size > 0 && break
    end

    # Phase 2: Orient V-structures (colliders)
    adj = zeros(Int, m, m)
    for i in 1:m, j in (i+1):m
        if adj_undirected[i, j]
            adj[i, j] = Int(UNDIRECTED)
            adj[j, i] = Int(UNDIRECTED)
        end
    end

    for i in 1:m
        neighbours = [k for k in 1:m if adj_undirected[i, k]]
        for (a_idx, b_idx) in [(a,b) for (idx_a, a) in enumerate(neighbours)
                                      for b in neighbours[idx_a+1:end]]
            # Orient a → i ← b if a-i-b and a not adjacent to b
            # and i ∉ sep_set(a, b)
            !adj_undirected[a_idx, b_idx] || continue
            sep = get(sep_set, (a_idx, b_idx), nothing)
            if sep !== nothing && !(i in sep)
                adj[a_idx, i] = Int(DIRECTED)
                adj[b_idx, i] = Int(DIRECTED)
                adj[i, a_idx] = Int(NO_EDGE)
                adj[i, b_idx] = Int(NO_EDGE)
            end
        end
    end

    # Phase 3: Meek orientation rules (simplified R1 only)
    # R1: If a → b — c and a not adjacent to c, orient b → c
    changed = true
    while changed
        changed = false
        for b in 1:m, c in 1:m
            b == c && continue
            adj[b, c] != Int(UNDIRECTED) && continue
            for a in 1:m
                a == b || a == c || continue
                adj[a, b] == Int(DIRECTED) &&
                !adj_undirected[a, c] || continue
                adj[b, c] = Int(DIRECTED)
                adj[c, b] = Int(NO_EDGE)
                changed = true
            end
        end
    end

    CausalGraph(m, names, adj, pval_matrix, "PC",
                Dict{String, Any}("alpha" => alpha, "max_cond_set" => max_cond_set))
end

# ---------------------------------------------------------------------------
# FCI Algorithm
# ---------------------------------------------------------------------------

"""
FCI (Fast Causal Inference) algorithm — extends PC to handle hidden confounders.

FCI produces a Partial Ancestral Graph (PAG) where:
  - X → Y : X is an ancestor of Y (direct cause)
  - X ↔ Y : hidden common cause
  - X ∘→ Y : X may or may not be an ancestor of Y
  - X ∘—∘ Y : no orientation information

Implementation: simplified FCI using the PC skeleton + bidirected edge marking
for pairs with correlated residuals after regressing on all observed variables.

# Returns CausalGraph
"""
function FCI(data_matrix::AbstractMatrix{<:Real};
              alpha::Float64=0.05,
              var_names::Vector{String}=String[])
    n, m = size(data_matrix)
    names = isempty(var_names) ? ["X$i" for i in 1:m] : var_names
    X     = Float64.(data_matrix)

    # Start with PC skeleton
    pc_graph = PCAlgorithm(data_matrix; alpha=alpha, var_names=names,
                            max_cond_set=2)

    adj   = copy(pc_graph.adj_matrix)
    pvals = copy(pc_graph.pvalue_matrix)

    # FCI extension: detect hidden confounders
    # Heuristic: if X and Y are d-separated in the PC graph but their
    # residuals (after regressing on all other vars) are still correlated,
    # mark as bidirected (hidden confounder).
    for i in 1:m, j in (i+1):m
        adj[i, j] != Int(NO_EDGE) && continue   # already connected

        # Regress X_i and X_j on all other vars
        other = [k for k in 1:m if k != i && k != j]
        if isempty(other)
            continue
        end

        Z  = X[:, other]
        Z_d = hcat(ones(n), Z)
        ri = _ols_residuals(X[:, i], Z_d)
        rj = _ols_residuals(X[:, j], Z_d)

        r, p = _pearson_correlation(ri, rj)
        pvals[i, j] = pvals[j, i] = p

        if p < alpha
            # Correlated residuals → hidden confounder
            adj[i, j] = Int(BIDIRECTED)
            adj[j, i] = Int(BIDIRECTED)
        end
    end

    # Mark remaining undirected edges as circle-circle (FCI PAG convention)
    for i in 1:m, j in (i+1):m
        if adj[i, j] == Int(UNDIRECTED)
            adj[i, j] = Int(CIRCLE)
            adj[j, i] = Int(CIRCLE)
        end
    end

    CausalGraph(m, names, adj, pvals, "FCI",
                Dict{String, Any}("alpha" => alpha))
end

# ---------------------------------------------------------------------------
# Utility: combinations generator
# ---------------------------------------------------------------------------

function _combinations(v::Vector{Int}, k::Int)
    isempty(v) || k == 0 && return [Int[]]
    k > length(v) && return Vector{Int}[]
    result = Vector{Int}[]
    _combo_helper!(result, v, k, 1, Int[])
    result
end

function _combo_helper!(result::Vector{Vector{Int}}, v::Vector{Int},
                         k::Int, start::Int, current::Vector{Int})
    if length(current) == k
        push!(result, copy(current))
        return
    end
    for i in start:length(v)
        push!(current, v[i])
        _combo_helper!(result, v, k, i+1, current)
        pop!(current)
    end
end

# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

function graph_to_json(g::CausalGraph)::String
    JSON3.write(Dict(
        "n_vars"         => g.n_vars,
        "var_names"      => g.var_names,
        "adj_matrix"     => g.adj_matrix,
        "pvalue_matrix"  => g.pvalue_matrix,
        "method"         => g.method,
        "edge_types"     => Dict(
            "0" => "NO_EDGE",
            "1" => "DIRECTED (→)",
            "2" => "REVERSED (←)",
            "3" => "UNDIRECTED (—)",
            "4" => "BIDIRECTED (↔)",
            "5" => "CIRCLE (∘)"
        ),
        "metadata"       => g.metadata
    ))
end

function te_to_json(te_result)::String
    JSON3.write(te_result)
end

end  # module CausalFast

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using .CausalFast
    using Statistics
    using Random

    println("[causal_fast] Running self-tests...")

    rng = MersenneTwister(42)
    n   = 300
    m   = 5

    # Synthetic data with known causal structure: X1 → X2 → X3, X1 → X4
    X1 = randn(rng, n)
    X2 = 0.7 * X1 + 0.5 * randn(rng, n)
    X3 = 0.6 * X2 + 0.5 * randn(rng, n)
    X4 = 0.5 * X1 + 0.5 * randn(rng, n)
    X5 = randn(rng, n)   # independent noise
    data = hcat(X1, X2, X3, X4, X5)

    # 1. Granger causality
    granger = GrangerCausalityMatrix(data; max_lag=3, alpha=0.05,
                                      var_names=["X1","X2","X3","X4","X5"])
    n_granger_edges = sum(granger.adj_matrix .> 0) ÷ 2
    println("  Granger: $n_granger_edges significant causal links found")

    # 2. Transfer entropy (X1 → X2)
    te = TransferEntropy(X1, X2; lag=1)
    println("  Transfer Entropy X1→X2 = $(round(te.te_xy; digits=4)), " *
            "dominant: $(te.dominant_direction)")

    # 3. PC algorithm
    pc = PCAlgorithm(data; alpha=0.05, var_names=["X1","X2","X3","X4","X5"])
    n_pc_edges = sum(pc.adj_matrix .> 0) ÷ 2
    println("  PC graph: $n_pc_edges edges in skeleton")

    # 4. FCI
    fci = FCI(data; alpha=0.05, var_names=["X1","X2","X3","X4","X5"])
    n_fci_edges = sum(fci.adj_matrix .> 0) ÷ 2
    n_bidir     = sum(fci.adj_matrix .== Int(CausalFast.BIDIRECTED)) ÷ 2
    println("  FCI graph: $n_fci_edges edges, $n_bidir bidirected (hidden confounders)")

    # Write output
    out_dir = get(ENV, "STATS_OUTPUT_DIR", joinpath(@__DIR__, "..", "output"))
    mkpath(out_dir)
    open(joinpath(out_dir, "granger_adjacency.json"), "w") do io
        write(io, CausalFast.graph_to_json(granger))
    end
    open(joinpath(out_dir, "pc_adjacency.json"), "w") do io
        write(io, CausalFast.graph_to_json(pc))
    end
    open(joinpath(out_dir, "fci_adjacency.json"), "w") do io
        write(io, CausalFast.graph_to_json(fci))
    end

    println("[causal_fast] Self-tests complete.")
end
