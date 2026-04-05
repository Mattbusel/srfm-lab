# =============================================================================
# information_theory.jl — Information-Theoretic Signal Analysis
# =============================================================================
# Provides:
#   - ShannonEntropy          Discrete and continuous Shannon entropy
#   - DifferentialEntropy     KDE-based differential entropy estimation
#   - MutualInformation       KSG estimator (k-NN based mutual information)
#   - TransferEntropy         Directional information flow (TE: X→Y)
#   - PermutationEntropy      Complexity measure via ordinal patterns
#   - SampleEntropy           Sample entropy (SampEn) for time series
#   - ApproximateEntropy      ApEn for physiological / financial signals
#   - LempelZivComplexity     Lempel-Ziv complexity of binary sequence
#   - FeatureSelection        Max mutual info / min redundancy feature selector
#   - EffectiveTransferEntropy ETE with bootstrap significance testing
#   - SignalInfoContent       Which signals carry most info about future returns
#   - run_information_theory  Top-level driver with JSON export
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, Random, JSON3
# =============================================================================

module InformationTheory

using Statistics
using LinearAlgebra
using Random
using JSON3

export ShannonEntropy, DifferentialEntropy, MutualInformation
export TransferEntropy, PermutationEntropy, SampleEntropy, ApproximateEntropy
export LempelZivComplexity, FeatureSelection, EffectiveTransferEntropy
export SignalInfoContent, run_information_theory

# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Digamma function via series expansion (for KSG estimator)."""
function _digamma(x::Float64)::Float64
    x < 1.0 && return _digamma(x + 1.0) - 1.0/x
    # Asymptotic expansion for large x
    result = log(x) - 1.0/(2x)
    x2 = x^2
    result -= 1.0/(12x2) - 1.0/(120x2^2) + 1.0/(252x2^3)
    return result
end

"""Euclidean distance between two vectors."""
_dist(x::AbstractVector, y::AbstractVector) = sqrt(sum((x[i]-y[i])^2 for i in eachindex(x)))

"""L∞ (Chebyshev) distance."""
_linf(x::AbstractVector, y::AbstractVector) = maximum(abs.(x .- y))

"""Gaussian kernel."""
_gauss_kernel(u::Float64) = exp(-0.5*u^2) / sqrt(2π)

"""Silverman's bandwidth rule."""
function _silverman_bw(x::Vector{Float64})::Float64
    n = length(x); s = std(x)
    iqr = quantile(x, 0.75) - quantile(x, 0.25)
    s_eff = min(s, iqr / 1.349)
    return 1.06 * s_eff * n^(-0.2)
end

"""KDE evaluation at point x0."""
function _kde(data::Vector{Float64}, x0::Float64, h::Float64)::Float64
    n = length(data)
    s = 0.0
    for xi in data
        s += _gauss_kernel((x0 - xi) / h)
    end
    return s / (n * h)
end

"""Bin data into k equal-width bins, return probabilities."""
function _bin_probs(x::Vector{Float64}, k::Int)::Vector{Float64}
    lo, hi = minimum(x), maximum(x)
    hi == lo && return [1.0]
    edges = range(lo, hi, length=k+1) |> collect
    counts = zeros(k)
    for xi in x
        b = clamp(floor(Int, (xi - lo)/(hi - lo) * k) + 1, 1, k)
        counts[b] += 1.0
    end
    return counts ./ length(x)
end

"""Joint bin probabilities for two series."""
function _joint_bin_probs(x::Vector{Float64}, y::Vector{Float64}, k::Int)
    n = length(x); lo_x, hi_x = minimum(x), maximum(x); lo_y, hi_y = minimum(y), maximum(y)
    (hi_x == lo_x || hi_y == lo_y) && return ones(k,k) ./ k^2
    p = zeros(k, k)
    for i in 1:n
        bx = clamp(floor(Int, (x[i]-lo_x)/(hi_x-lo_x)*k)+1, 1, k)
        by = clamp(floor(Int, (y[i]-lo_y)/(hi_y-lo_y)*k)+1, 1, k)
        p[bx, by] += 1.0
    end
    return p ./ n
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Shannon Entropy
# ─────────────────────────────────────────────────────────────────────────────

"""
    ShannonEntropy(x; bins, base) → Float64

Compute Shannon entropy of a time series via histogram binning.

H(X) = -∑ p_i log(p_i)

# Arguments
- `x`    : data vector
- `bins` : number of histogram bins (default: Sturges rule)
- `base` : logarithm base (default 2.0 → bits; use ℯ for nats)

# Returns
Estimated Shannon entropy in bits (or nats).

# Example
```julia
x = randn(1000)
H = ShannonEntropy(x; bins=20)   # ≈ 4.16 bits for standard normal
```
"""
function ShannonEntropy(x::Vector{Float64}; bins::Int=0, base::Float64=2.0)::Float64
    n = length(x)
    k = bins > 0 ? bins : max(5, round(Int, 1 + log2(n)))
    p = _bin_probs(x, k)
    H = 0.0
    log_b = log(base)
    for pi in p
        pi > 0.0 && (H -= pi * log(pi) / log_b)
    end
    return H
end

"""
    ShannonEntropy(probs; base) → Float64

Compute Shannon entropy directly from probability vector.
"""
function ShannonEntropy(probs::Vector{Float64}; base::Float64=2.0)::Float64
    log_b = log(base)
    H = 0.0
    for p in probs
        p > 0.0 && (H -= p * log(p) / log_b)
    end
    return H
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Differential Entropy (KDE-based)
# ─────────────────────────────────────────────────────────────────────────────

"""
    DifferentialEntropy(x; h, n_eval) → Float64

Estimate differential (continuous) entropy via kernel density estimation.

h(X) = -∫ f(x) log f(x) dx ≈ -(1/n) ∑ log f̂(x_i)

# Arguments
- `x`      : data vector
- `h`      : bandwidth (default: Silverman's rule)
- `n_eval` : number of evaluation points for integration (default 200)

# Returns
Estimated differential entropy in nats.
"""
function DifferentialEntropy(x::Vector{Float64}; h::Float64=0.0, n_eval::Int=200)::Float64
    n = length(x)
    bw = h > 0.0 ? h : _silverman_bw(x)

    # Evaluate KDE at each data point (leave-one-out for bias reduction)
    H = 0.0
    for i in 1:n
        f_hat = 0.0
        for j in 1:n
            j == i && continue
            f_hat += _gauss_kernel((x[i] - x[j]) / bw)
        end
        f_hat /= (n - 1) * bw
        f_hat > 0.0 && (H -= log(f_hat))
    end
    return H / n
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Mutual Information (KSG Estimator)
# ─────────────────────────────────────────────────────────────────────────────

"""
    MutualInformation(x, y; k, estimator) → Float64

Estimate mutual information I(X;Y) using the Kraskov-Stögbauer-Grassberger
(KSG) k-nearest-neighbour estimator (Algorithm 1).

I(X;Y) ≈ ψ(k) - <ψ(n_x+1) + ψ(n_y+1)> + ψ(n)

where n_x, n_y are point counts in marginal boxes of size ε.

# Arguments
- `x`         : first variable (n-vector)
- `y`         : second variable (n-vector)
- `k`         : number of nearest neighbours (default 5)
- `estimator` : :ksg1 or :ksg2 (default :ksg1)

# Returns
Estimated mutual information in nats.
"""
function MutualInformation(x::Vector{Float64}, y::Vector{Float64};
                            k::Int=5, estimator::Symbol=:ksg1)::Float64
    n = length(x); length(y) == n || error("x and y must have same length")

    # Standardise
    x_s = (x .- mean(x)) ./ max(std(x), 1e-10)
    y_s = (y .- mean(y)) ./ max(std(y), 1e-10)

    data = hcat(x_s, y_s)   # n×2 joint space

    MI = 0.0
    for i in 1:n
        # Find k-th nearest neighbour in joint space (L∞ norm)
        dists = [_linf(data[i,:], data[j,:]) for j in 1:n if j != i]
        sort!(dists)
        ε = dists[min(k, length(dists))]

        if estimator == :ksg1
            # Count points within ε in each marginal
            n_x = sum(abs(x_s[j] - x_s[i]) < ε for j in 1:n if j != i)
            n_y = sum(abs(y_s[j] - y_s[i]) < ε for j in 1:n if j != i)
            MI += _digamma(k) - _digamma(n_x + 1.0) - _digamma(n_y + 1.0)
        else
            # KSG Algorithm 2: use separate radii per marginal
            d_x = sort([abs(x_s[j] - x_s[i]) for j in 1:n if j != i])
            d_y = sort([abs(y_s[j] - y_s[i]) for j in 1:n if j != i])
            ε_x = d_x[min(k, length(d_x))]
            ε_y = d_y[min(k, length(d_y))]
            n_x = sum(abs(x_s[j] - x_s[i]) ≤ ε_x for j in 1:n if j != i)
            n_y = sum(abs(y_s[j] - y_s[i]) ≤ ε_y for j in 1:n if j != i)
            MI += _digamma(k) + _digamma(n) - _digamma(n_x + 0.5) - _digamma(n_y + 0.5)
        end
    end
    MI = if estimator == :ksg1
        _digamma(n) + MI / n
    else
        MI / n
    end
    return max(MI, 0.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Transfer Entropy
# ─────────────────────────────────────────────────────────────────────────────

"""
    TransferEntropy(x, y; lag, bins, k) → NamedTuple

Compute Transfer Entropy from X to Y:
TE(X→Y) = I(Y_{t+1}; X_t | Y_t)
         = H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t, X_t)

Measures the additional information X provides about Y's future
beyond Y's own past.

# Arguments
- `x`    : source series
- `y`    : target series
- `lag`  : time lag (default 1)
- `bins` : histogram bins for density estimation (default 10)

# Returns
NamedTuple: (TE_xy, TE_yx, net_flow, direction)
"""
function TransferEntropy(x::Vector{Float64}, y::Vector{Float64};
                          lag::Int=1, bins::Int=10)
    n = length(x); n == length(y) || error("x and y must match in length")
    n_eff = n - lag

    # Build shifted series
    y_future = y[(lag+1):end]
    y_past   = y[1:(n-lag)]
    x_past   = x[1:(n-lag)]

    # TE(X→Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
    #         = H(Y_{t+1}, Y_t) - H(Y_t) - H(Y_{t+1}, Y_t, X_t) + H(Y_t, X_t)
    H_yf_yp   = _joint_entropy_2d(y_future, y_past, bins)
    H_yp      = ShannonEntropy(y_past; bins=bins)
    H_yf_yp_xp = _joint_entropy_3d(y_future, y_past, x_past, bins)
    H_yp_xp   = _joint_entropy_2d(y_past, x_past, bins)

    TE_xy = H_yf_yp - H_yp - H_yf_yp_xp + H_yp_xp
    TE_xy = max(TE_xy, 0.0)

    # Reverse direction
    H_xf_xp   = _joint_entropy_2d(x[(lag+1):end], x[1:(n-lag)], bins)
    H_xp      = ShannonEntropy(x[1:(n-lag)]; bins=bins)
    H_xf_xp_yp = _joint_entropy_3d(x[(lag+1):end], x[1:(n-lag)], y_past, bins)
    H_xp_yp   = _joint_entropy_2d(x[1:(n-lag)], y_past, bins)

    TE_yx = H_xf_xp - H_xp - H_xf_xp_yp + H_xp_yp
    TE_yx = max(TE_yx, 0.0)

    net_flow = TE_xy - TE_yx
    direction = net_flow > 0 ? "X→Y" : "Y→X"

    return (TE_xy=TE_xy, TE_yx=TE_yx, net_flow=net_flow, direction=direction)
end

"""2D joint entropy via histogram."""
function _joint_entropy_2d(x::Vector{Float64}, y::Vector{Float64}, k::Int)::Float64
    p = _joint_bin_probs(x, y, k)
    H = 0.0
    for pij in p
        pij > 0.0 && (H -= pij * log2(pij))
    end
    return H
end

"""3D joint entropy via histogram."""
function _joint_entropy_3d(x::Vector{Float64}, y::Vector{Float64},
                             z::Vector{Float64}, k::Int)::Float64
    n = length(x)
    lo_x, hi_x = minimum(x), maximum(x)
    lo_y, hi_y = minimum(y), maximum(y)
    lo_z, hi_z = minimum(z), maximum(z)
    (hi_x==lo_x || hi_y==lo_y || hi_z==lo_z) && return log2(k^3)
    p = zeros(k, k, k)
    for i in 1:n
        bx = clamp(floor(Int,(x[i]-lo_x)/(hi_x-lo_x)*k)+1, 1, k)
        by = clamp(floor(Int,(y[i]-lo_y)/(hi_y-lo_y)*k)+1, 1, k)
        bz = clamp(floor(Int,(z[i]-lo_z)/(hi_z-lo_z)*k)+1, 1, k)
        p[bx,by,bz] += 1.0
    end
    p ./= n
    H = 0.0
    for pijk in p
        pijk > 0.0 && (H -= pijk * log2(pijk))
    end
    return H
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Permutation Entropy
# ─────────────────────────────────────────────────────────────────────────────

"""
    PermutationEntropy(x; m, tau, normalize) → Float64

Compute permutation entropy (Bandt & Pompe, 2002).
Encodes local ordinal structure of a time series into symbols.

H_p = -∑ p(π) log p(π)

# Arguments
- `x`         : time series
- `m`         : embedding dimension (pattern length, default 3)
- `tau`       : time delay (default 1)
- `normalize` : if true, divide by log(m!) (default true)

# Returns
Permutation entropy in nats (or normalised to [0,1]).
"""
function PermutationEntropy(x::Vector{Float64}; m::Int=3, tau::Int=1,
                             normalize::Bool=true)::Float64
    n = length(x)
    n_patterns = n - (m-1)*tau
    n_patterns < 1 && error("Series too short for given m and tau")

    # Count ordinal patterns
    perm_counts = Dict{Vector{Int}, Int}()
    for i in 1:n_patterns
        segment = [x[i + (j-1)*tau] for j in 1:m]
        perm = sortperm(segment)
        perm_counts[perm] = get(perm_counts, perm, 0) + 1
    end

    # Compute entropy
    H = 0.0
    for (_, cnt) in perm_counts
        p = cnt / n_patterns
        H -= p * log(p)
    end

    if normalize
        log_fact_m = sum(log(j) for j in 1:m)  # log(m!)
        H /= log_fact_m
    end
    return H
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Sample Entropy
# ─────────────────────────────────────────────────────────────────────────────

"""
    SampleEntropy(x; m, r) → Float64

Compute Sample Entropy (SampEn): measures regularity/complexity.
Low SampEn = high regularity; high SampEn = high complexity/randomness.

SampEn(m, r) = -log(A / B)
where A = #templates of length m+1 that match (within r)
      B = #templates of length m that match

# Arguments
- `x` : time series
- `m` : template length (default 2)
- `r` : tolerance (default 0.2 * std(x))

# Returns
Sample entropy (nats). Returns NaN if B = 0.
"""
function SampleEntropy(x::Vector{Float64}; m::Int=2, r::Float64=0.0)::Float64
    n = length(x)
    tol = r > 0.0 ? r : 0.2 * std(x)

    A = 0; B = 0
    for i in 1:(n-m)
        for j in (i+1):(n-m)
            # Check m-length match
            match_m = true
            for k in 0:(m-1)
                abs(x[i+k] - x[j+k]) > tol && (match_m = false; break)
            end
            if match_m
                B += 1
                # Check m+1 length
                if i+m ≤ n && j+m ≤ n && abs(x[i+m] - x[j+m]) ≤ tol
                    A += 1
                end
            end
        end
    end
    (A == 0 || B == 0) && return NaN
    return -log(A / B)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Approximate Entropy
# ─────────────────────────────────────────────────────────────────────────────

"""
    ApproximateEntropy(x; m, r) → Float64

Compute Approximate Entropy (ApEn) for a time series.
Measures regularity; lower = more predictable.

# Arguments
- `x` : time series
- `m` : template length (default 2)
- `r` : tolerance (default 0.2 * std(x))
"""
function ApproximateEntropy(x::Vector{Float64}; m::Int=2, r::Float64=0.0)::Float64
    n = length(x)
    tol = r > 0.0 ? r : 0.2 * std(x)

    function _phi(m_val::Int)::Float64
        cnt = 0.0
        for i in 1:(n-m_val+1)
            matches = 0
            for j in 1:(n-m_val+1)
                ok = true
                for k in 0:(m_val-1)
                    abs(x[i+k] - x[j+k]) > tol && (ok = false; break)
                end
                ok && (matches += 1)
            end
            cnt += log(matches / (n - m_val + 1.0))
        end
        return cnt / (n - m_val + 1.0)
    end

    return _phi(m) - _phi(m+1)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Lempel-Ziv Complexity
# ─────────────────────────────────────────────────────────────────────────────

"""
    LempelZivComplexity(x; threshold) → NamedTuple

Compute Lempel-Ziv complexity of a time series (binarised at median).

# Arguments
- `x`         : time series
- `threshold` : binarisation threshold (default: median(x))

# Returns
NamedTuple: (complexity, n_words, normalised_complexity)
"""
function LempelZivComplexity(x::Vector{Float64}; threshold::Float64=NaN)
    thr = isnan(threshold) ? median(x) : threshold
    bits = x .> thr  # binary string

    n = length(bits)
    # Lempel-Ziv parsing: find the minimum number of distinct words
    i = 1
    complexity = 1
    l = 1
    k = 1
    k_max = 1

    while true
        if bits[i + k - 1] == bits[i + l - 1]  # careful indexing
            k += 1
            if i + k - 1 > n
                complexity += 1
                break
            end
        else
            k_max = max(k_max, k)
            i += 1
            if i > n - l
                complexity += 1
                break
            end
            k = 1
        end
        if k_max ≥ l + 1
            complexity += 1
            l += k_max
            if l + 1 > n
                break
            end
            i = 1; k = 1; k_max = 1
        end
    end

    # Alternative: simple word-based LZ
    # Normalise by theoretical maximum: c_max ≈ n / log2(n)
    n_words = complexity
    c_max = n / max(log2(n), 1.0)
    normalised = n_words / c_max

    return (complexity=n_words, normalised_complexity=normalised, n=n)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Information-Theoretic Feature Selection (mRMR)
# ─────────────────────────────────────────────────────────────────────────────

"""
    FeatureSelection(X, y; n_features, method, bins) → NamedTuple

Select features using Max-Relevance Min-Redundancy (mRMR) criterion.

Score = I(X_j; Y) - (1/|S|) ∑_{X_i ∈ S} I(X_j; X_i)

# Arguments
- `X`          : n×p feature matrix
- `y`          : n-vector target variable
- `n_features` : number of features to select (default: all, ranked)
- `method`     : :mrmr or :maxmi (default :mrmr)
- `bins`       : histogram bins for MI estimation (default 10)

# Returns
NamedTuple: (selected_indices, scores, mi_with_target, mi_redundancy)
"""
function FeatureSelection(X::Matrix{Float64}, y::Vector{Float64};
                           n_features::Int=0, method::Symbol=:mrmr,
                           bins::Int=10, k_nn::Int=5)
    n, p = size(X)
    n_sel = n_features > 0 ? min(n_features, p) : p

    # Compute MI of each feature with target
    mi_target = Float64[MutualInformation(X[:,j], y; k=k_nn) for j in 1:p]

    if method == :maxmi
        # Simple: rank by MI with target
        order = sortperm(mi_target, rev=true)
        return (selected_indices=order[1:n_sel],
                scores=mi_target[order[1:n_sel]],
                mi_with_target=mi_target,
                mi_redundancy=zeros(p))
    end

    # mRMR: greedy forward selection
    selected = Int[]
    remaining = collect(1:p)
    scores = Float64[]
    redundancy = zeros(p)

    # Precompute inter-feature MI matrix (expensive for large p)
    max_pairs = min(p, 20)  # limit for performance
    MI_feat = zeros(p, p)
    for i in 1:min(p, max_pairs), j in (i+1):min(p, max_pairs)
        MI_feat[i,j] = MI_feat[j,i] = MutualInformation(X[:,i], X[:,j]; k=k_nn)
    end

    for _ in 1:n_sel
        best_score = -Inf
        best_feat  = remaining[1]

        for j in remaining
            red = isempty(selected) ? 0.0 :
                  mean(MI_feat[j, s] for s in selected)
            score = mi_target[j] - red
            if score > best_score
                best_score = score
                best_feat = j
            end
        end

        push!(selected, best_feat)
        push!(scores, best_score)
        filter!(x -> x != best_feat, remaining)
    end

    return (selected_indices=selected, scores=scores,
            mi_with_target=mi_target, mi_matrix=MI_feat)
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. Effective Transfer Entropy with Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

"""
    EffectiveTransferEntropy(x, y; lag, bins, n_bootstrap, alpha) → NamedTuple

Compute Effective Transfer Entropy with bootstrap significance testing.

ETE(X→Y) = TE(X→Y) - TE_shuffle(X→Y)
Corrects for finite-sample bias using shuffled surrogates.

# Arguments
- `x`           : source time series
- `y`           : target time series
- `lag`         : time lag (default 1)
- `bins`        : histogram bins (default 8)
- `n_bootstrap` : number of bootstrap samples (default 100)
- `alpha`       : significance level (default 0.05)

# Returns
NamedTuple: (ETE_xy, ETE_yx, p_value_xy, p_value_yx, significant_xy, significant_yx)
"""
function EffectiveTransferEntropy(x::Vector{Float64}, y::Vector{Float64};
                                   lag::Int=1, bins::Int=8,
                                   n_bootstrap::Int=100, alpha::Float64=0.05,
                                   rng::AbstractRNG=Random.default_rng())
    # Observed TE
    te_obs = TransferEntropy(x, y; lag=lag, bins=bins)
    TE_xy_obs = te_obs.TE_xy
    TE_yx_obs = te_obs.TE_yx

    # Shuffle surrogates for X (destroy causal structure X→Y)
    null_xy = Float64[]
    null_yx = Float64[]
    for _ in 1:n_bootstrap
        x_shuf = shuffle(rng, x)
        y_shuf = shuffle(rng, y)
        te_shuf_xy = TransferEntropy(x_shuf, y; lag=lag, bins=bins).TE_xy
        te_shuf_yx = TransferEntropy(x, y_shuf; lag=lag, bins=bins).TE_yx
        push!(null_xy, te_shuf_xy)
        push!(null_yx, te_shuf_yx)
    end

    # Bias correction
    ETE_xy = TE_xy_obs - mean(null_xy)
    ETE_yx = TE_yx_obs - mean(null_yx)

    # P-values: fraction of surrogates ≥ observed
    p_xy = mean(null_xy .≥ TE_xy_obs)
    p_yx = mean(null_yx .≥ TE_yx_obs)

    return (ETE_xy=ETE_xy, ETE_yx=ETE_yx,
            TE_xy_raw=TE_xy_obs, TE_yx_raw=TE_yx_obs,
            p_value_xy=p_xy, p_value_yx=p_yx,
            significant_xy=p_xy < alpha,
            significant_yx=p_yx < alpha,
            null_mean_xy=mean(null_xy), null_mean_yx=mean(null_yx))
end

# ─────────────────────────────────────────────────────────────────────────────
# 11. Signal Information Content for Future Returns
# ─────────────────────────────────────────────────────────────────────────────

"""
    SignalInfoContent(signals, returns; lags, bins, n_bootstrap) → NamedTuple

Determine which signals carry the most information about future returns
using mutual information and transfer entropy.

# Arguments
- `signals`     : n×p matrix of signal values (e.g. volume, spread, RSI)
- `returns`     : n-vector of asset returns
- `lags`        : vector of forecast lags to test (default [1,2,5,10])
- `bins`        : histogram bins for MI (default 10)
- `n_bootstrap` : bootstrap repetitions for significance (default 50)

# Returns
NamedTuple: (mi_scores, te_scores, ranking, summary_table)
"""
function SignalInfoContent(signals::Matrix{Float64}, returns::Vector{Float64};
                            lags::Vector{Int}=Int[1,2,5,10], bins::Int=10,
                            n_bootstrap::Int=50, k_nn::Int=5,
                            rng::AbstractRNG=Random.default_rng())
    n, p = size(signals)
    n_lags = length(lags)

    mi_scores  = zeros(p, n_lags)
    te_scores  = zeros(p, n_lags)
    p_values   = ones(p, n_lags)

    for (li, lag) in enumerate(lags)
        if lag >= n; continue; end
        y_future = returns[(lag+1):end]
        for j in 1:p
            sig_past = signals[1:(n-lag), j]
            # Mutual information between lagged signal and future returns
            mi_scores[j, li] = MutualInformation(sig_past, y_future; k=k_nn)
            # Transfer entropy
            te_result = TransferEntropy(sig_past, y_future; lag=1, bins=bins)
            te_scores[j, li] = te_result.TE_xy
            # Bootstrap p-value
            null = Float64[]
            for _ in 1:n_bootstrap
                s_shuf = shuffle(rng, sig_past)
                push!(null, MutualInformation(s_shuf, y_future; k=k_nn))
            end
            p_values[j, li] = mean(null .≥ mi_scores[j, li])
        end
    end

    # Overall ranking: average MI across lags
    avg_mi = mean(mi_scores, dims=2)[:,1]
    ranking = sortperm(avg_mi, rev=true)

    # Summary: top-5 signals at each lag
    summary = Dict{String, Any}()
    for (li, lag) in enumerate(lags)
        top_k = min(5, p)
        top_idx = sortperm(mi_scores[:, li], rev=true)[1:top_k]
        summary["lag_$(lag)"] = Dict(
            "top_signals"  => top_idx,
            "mi_values"    => mi_scores[top_idx, li],
            "te_values"    => te_scores[top_idx, li],
            "p_values"     => p_values[top_idx, li]
        )
    end

    return (mi_scores=mi_scores, te_scores=te_scores,
            p_values=p_values, ranking=ranking,
            avg_mi=avg_mi, summary=summary)
end

# ─────────────────────────────────────────────────────────────────────────────
# 12. Top-Level Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_information_theory(data; target_col, out_path) → Dict

Full information-theoretic analysis pipeline for crypto signals.

# Arguments
- `data`       : n×p matrix (last column = target returns by default)
- `target_col` : which column to use as the target (default last)
- `out_path`   : optional JSON export path

# Returns
Dict with entropy, MI, TE, permutation entropy, and feature rankings.

# Example
```julia
using Random
rng = Random.MersenneTwister(42)
n = 500; p = 6
data = randn(rng, n, p)
results = run_information_theory(data; out_path="it_results.json")
println("Top signal: ", results["feature_ranking"][1])
```
"""
function run_information_theory(data::Matrix{Float64};
                                 target_col::Int=0,
                                 out_path::Union{String,Nothing}=nothing)
    n, p = size(data)
    tc = target_col > 0 ? target_col : p
    returns = data[:, tc]
    sig_cols = setdiff(1:p, [tc])
    signals = isempty(sig_cols) ? data : data[:, sig_cols]

    results = Dict{String, Any}()

    # ── Entropy of Each Series ─────────────────────────────────────────────
    @info "Computing Shannon and differential entropy..."
    entropies = Dict{String, Float64}()
    diff_entropies = Dict{String, Float64}()
    for j in 1:p
        entropies["col_$(j)"]      = ShannonEntropy(data[:,j]; bins=20)
        diff_entropies["col_$(j)"] = DifferentialEntropy(data[:,j])
    end
    results["shannon_entropy"] = entropies
    results["differential_entropy"] = diff_entropies

    # ── Permutation Entropy ────────────────────────────────────────────────
    @info "Permutation entropy (complexity)..."
    perm_ents = Dict{String, Float64}()
    for j in 1:p
        perm_ents["col_$(j)"] = PermutationEntropy(data[:,j]; m=4, tau=1)
    end
    results["permutation_entropy"] = perm_ents

    # ── Sample / Approximate Entropy ──────────────────────────────────────
    n_sub = min(n, 300)  # limit for O(n²) algos
    results["sample_entropy_target"]  = SampleEntropy(returns[1:n_sub])
    results["approx_entropy_target"]  = ApproximateEntropy(returns[1:n_sub])

    # ── Lempel-Ziv Complexity ─────────────────────────────────────────────
    @info "Lempel-Ziv complexity..."
    lz_result = LempelZivComplexity(returns)
    results["lempel_ziv"] = Dict(
        "complexity"   => lz_result.complexity,
        "normalised"   => lz_result.normalised_complexity
    )

    # ── Mutual Information with Future Returns ─────────────────────────────
    @info "Mutual information: signals → future returns..."
    mi_1d = Float64[]
    for j in 1:size(signals,2)
        lag = 1
        if lag < n
            push!(mi_1d, MutualInformation(signals[1:(n-lag),j], returns[(lag+1):end]; k=3))
        end
    end
    results["mi_1d_lag"] = mi_1d

    # ── Transfer Entropy ──────────────────────────────────────────────────
    @info "Transfer entropy: pairwise signal–return flows..."
    te_results = Dict{String, Any}()
    for j in 1:min(size(signals,2), 4)
        te = TransferEntropy(signals[:,j], returns; lag=1, bins=8)
        te_results["signal_$(j)_to_returns"] = Dict(
            "TE_xy" => te.TE_xy, "TE_yx" => te.TE_yx,
            "net_flow" => te.net_flow, "direction" => te.direction
        )
    end
    results["transfer_entropy"] = te_results

    # ── Effective Transfer Entropy ─────────────────────────────────────────
    @info "Effective TE (bootstrap, n=50)..."
    rng = Random.default_rng()
    if size(signals,2) >= 1
        ete = EffectiveTransferEntropy(signals[:,1], returns;
                                        lag=1, bins=8, n_bootstrap=50, rng=rng)
        results["effective_TE"] = Dict(
            "ETE_xy"         => ete.ETE_xy,
            "ETE_yx"         => ete.ETE_yx,
            "p_value_xy"     => ete.p_value_xy,
            "significant_xy" => ete.significant_xy
        )
    end

    # ── Feature Selection ──────────────────────────────────────────────────
    @info "mRMR feature selection..."
    if size(signals, 2) >= 2
        fs = FeatureSelection(signals, returns; n_features=min(5, size(signals,2)),
                               method=:mrmr, bins=8, k_nn=3)
        results["feature_selection"] = Dict(
            "selected_indices" => fs.selected_indices,
            "scores"           => fs.scores,
            "mi_with_target"   => fs.mi_with_target
        )
        results["feature_ranking"] = fs.selected_indices
    end

    # ── Signal Info Content ────────────────────────────────────────────────
    @info "Signal information content at multiple lags..."
    sic = SignalInfoContent(signals, returns; lags=[1,2,5], bins=8,
                             n_bootstrap=30, k_nn=3, rng=rng)
    results["signal_info_content"] = Dict(
        "ranking"  => sic.ranking,
        "avg_mi"   => sic.avg_mi,
        "summary"  => sic.summary
    )

    if !isnothing(out_path)
        open(out_path, "w") do io
            JSON3.write(io, results)
        end
        @info "Results written to $out_path"
    end

    return results
end

end  # module InformationTheory
