"""
optimal_transport.jl

Optimal transport (Wasserstein distances) for financial distribution analysis.

Optimal transport seeks the minimum-cost way to "move mass" from one
probability distribution to another. The Wasserstein-p distance is:

    Wₚ(μ, ν) = (inf_{γ ∈ Γ(μ,ν)} ∫ c(x,y)^p dγ(x,y))^{1/p}

where Γ(μ,ν) is the set of couplings with marginals μ and ν,
and c(x,y) is the ground cost (typically Euclidean distance).

Applications:
  - Compare live P&L distribution vs backtest
  - Detect distribution drift sequentially (CUSUM on OT distance)
  - Compute barycenters for multi-period portfolio blending
  - Compare regime-conditional return distributions

References:
  Villani (2009) "Optimal Transport: Old and New"
  Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
  Peyré & Cuturi (2019) "Computational Optimal Transport"
"""

using LinearAlgebra
using Statistics
using Random
using Distributions
using Plots
using StatsBase

# ─────────────────────────────────────────────────────────────────────────────
# 1D WASSERSTEIN DISTANCE (EXACT)
# ─────────────────────────────────────────────────────────────────────────────

"""
Exact Wasserstein-1 distance for 1D distributions.

For 1D measures, W₁(μ,ν) = ∫|F_μ(x) - F_ν(x)| dx
where F_μ, F_ν are CDFs.

With empirical measures on sorted samples:
    W₁ = (1/n) Σᵢ |X_(i) - Y_(i)|  when n = m

Returns the W₁ distance.
"""
function wasserstein_1d(μ_samples::Vector{Float64}, ν_samples::Vector{Float64};
                         p::Int=1)
    n = length(μ_samples)
    m = length(ν_samples)

    if n == m
        # Equal-weight empirical measures: sort and match
        X_sorted = sort(μ_samples)
        Y_sorted = sort(ν_samples)
        return mean(abs.(X_sorted .- Y_sorted) .^ p) ^ (1/p)
    else
        # Unequal sizes: interpolate CDFs on common quantile grid
        n_q = max(n, m, 1000)
        quantiles = range(0, 1, length=n_q)

        F_μ_inv = quantile.(Ref(μ_samples), quantiles)
        F_ν_inv = quantile.(Ref(ν_samples), quantiles)

        return mean(abs.(F_μ_inv .- F_ν_inv) .^ p) ^ (1/p)
    end
end

"""
Exact Wasserstein-2 distance for 1D distributions.
W₂(μ,ν) = (∫(F_μ⁻¹(t) - F_ν⁻¹(t))² dt)^{1/2}
"""
wasserstein_2_1d(μ_samples, ν_samples) = wasserstein_1d(μ_samples, ν_samples; p=2)

"""
Energy distance (related to W₁ for 1D):
ED(μ,ν) = 2E|X-Y| - E|X-X'| - E|Y-Y'|

This is equivalent to 2·W₁ for 1D distributions.
"""
function energy_distance(X::Vector{Float64}, Y::Vector{Float64})
    n, m = length(X), length(Y)
    E_XY = mean(abs(x - y) for x in X, y in Y)
    E_XX = n > 1 ? mean(abs(X[i] - X[j]) for i in 1:n, j in 1:n if i != j) : 0.0
    E_YY = m > 1 ? mean(abs(Y[i] - Y[j]) for i in 1:m, j in 1:m if i != j) : 0.0
    return 2 * E_XY - E_XX - E_YY
end

# ─────────────────────────────────────────────────────────────────────────────
# 2D / MULTIDIMENSIONAL: SINKHORN REGULARISED OT
# ─────────────────────────────────────────────────────────────────────────────

"""
Compute pairwise squared Euclidean cost matrix C[i,j] = ||xᵢ - yⱼ||².
"""
function cost_matrix(X::Matrix{Float64}, Y::Matrix{Float64})
    n, d = size(X)
    m = size(Y, 1)
    C = zeros(n, m)
    for i in 1:n, j in 1:m
        C[i,j] = sum((X[i,:] .- Y[j,:]).^2)
    end
    return C
end

"""
Sinkhorn algorithm for regularised optimal transport.

Solves: min_{P ∈ Γ(a,b)} <C, P> - ε · H(P)
where H(P) = -Σᵢⱼ Pᵢⱼ(log Pᵢⱼ - 1) is the entropy of P.

The entropic regularisation makes the problem strictly convex with unique
solution P* = diag(u) · K · diag(v), where K = exp(-C/ε).

Iterative Sinkhorn scaling:
    u ← a / (K v)
    v ← b / (Kᵀ u)

Returns (W_reg, transport_plan P).
"""
function sinkhorn(a::Vector{Float64}, b::Vector{Float64},
                  C::Matrix{Float64}, ε::Float64;
                  max_iter=1000, tol=1e-9, log_domain=true)
    n, m = length(a), length(b)
    size(C) == (n, m) || throw(DimensionMismatch("C must be n×m"))

    if log_domain
        # Log-domain Sinkhorn for numerical stability
        # u, v are log-scaling vectors
        log_K = -C ./ ε
        log_a = log.(a)
        log_b = log.(b)
        log_u = zeros(n)
        log_v = zeros(m)

        for iter in 1:max_iter
            log_u_new = log_a .- vec(logsumexp_cols(log_K .+ log_v', 2))
            log_v_new = log_b .- vec(logsumexp_cols((log_K .+ log_u_new)', 2))

            err = maximum(abs.(log_u_new .- log_u)) + maximum(abs.(log_v_new .- log_v))
            log_u = log_u_new
            log_v = log_v_new
            err < tol && break
        end

        # Transport plan in log domain
        log_P = log_u .+ log_K .+ log_v'
        P = exp.(log_P)
    else
        # Standard Sinkhorn (may overflow for small ε)
        K = exp.(-C ./ ε)
        u = ones(n)
        v = ones(m)

        for iter in 1:max_iter
            u_new = a ./ (K * v)
            v_new = b ./ (K' * u_new)
            err = maximum(abs.(u_new .- u)) + maximum(abs.(v_new .- v))
            u = u_new
            v = v_new
            err < tol && break
        end

        P = Diagonal(u) * K * Diagonal(v)
    end

    # Regularised Wasserstein distance
    W_reg = dot(C, P)

    return W_reg, P
end

# Log-sum-exp along dimension dim
function logsumexp_cols(A::Matrix{Float64}, dim::Int)
    if dim == 1
        return [logsumexp(A[:,j]) for j in 1:size(A,2)]'
    else
        return [logsumexp(A[i,:]) for i in 1:size(A,1)]
    end
end

function logsumexp(x::Vector{Float64})
    m = maximum(x)
    return m + log(sum(exp.(x .- m)))
end

"""
Wasserstein-2 distance for 2D/multidimensional distributions using Sinkhorn.

X: n×d matrix of samples from distribution μ
Y: m×d matrix of samples from distribution ν
reg: regularisation parameter ε (smaller → closer to true OT, but less stable)
"""
function wasserstein_2d(X::Matrix{Float64}, Y::Matrix{Float64}, reg::Float64=0.01;
                         max_iter=500, tol=1e-8)
    n, d = size(X)
    m = size(Y, 1)

    # Uniform weights
    a = fill(1.0/n, n)
    b = fill(1.0/m, m)

    # Squared Euclidean cost
    C = cost_matrix(X, Y)

    W_reg, P = sinkhorn(a, b, C, reg; max_iter=max_iter, tol=tol)

    # Debiased Wasserstein: remove self-transport terms
    C_XX = cost_matrix(X, X)
    C_YY = cost_matrix(Y, Y)
    W_XX, _ = sinkhorn(a, a, C_XX, reg)
    W_YY, _ = sinkhorn(b, b, C_YY, reg)

    W_debiased = max(0.0, W_reg - 0.5 * (W_XX + W_YY))

    return sqrt(W_debiased), P
end

# ─────────────────────────────────────────────────────────────────────────────
# SLICED WASSERSTEIN DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

"""
Sliced Wasserstein distance: average W₁ over random 1D projections.

SW₂(μ,ν) = (∫_{S^{d-1}} W₂²(θ#μ, θ#ν) dσ(θ))^{1/2}

where θ#μ is the push-forward of μ under projection onto direction θ.

This is O(n log n · n_proj) vs O(n² · n_iter) for Sinkhorn,
making it practical for large d and n.
"""
function sliced_wasserstein(X::Matrix{Float64}, Y::Matrix{Float64},
                             n_proj::Int=200; p::Int=2, rng=Random.GLOBAL_RNG)
    d = size(X, 2)
    size(Y, 2) == d || throw(DimensionMismatch("X and Y must have same dimension"))

    W_sq = 0.0
    for _ in 1:n_proj
        # Sample random direction on unit sphere
        θ = randn(rng, d)
        θ ./= norm(θ)

        # Project samples
        X_proj = X * θ
        Y_proj = Y * θ

        # 1D Wasserstein
        w = wasserstein_1d(X_proj, Y_proj; p=p)
        W_sq += w^p
    end

    return (W_sq / n_proj)^(1/p)
end

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO DISTRIBUTION COMPARISON: LIVE vs BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

"""
Compare live P&L distribution to backtest P&L distribution.
Returns W₁ distance and diagnostic statistics.

Inputs:
  live_pnl   : vector of daily/hourly P&L from live trading
  backtest_pnl: vector of P&L from backtest simulation
"""
function compare_pnl_distributions(live_pnl::Vector{Float64},
                                    backtest_pnl::Vector{Float64};
                                    n_bootstrap=500, α=0.05)
    W1 = wasserstein_1d(live_pnl, backtest_pnl)
    W2 = wasserstein_1d(live_pnl, backtest_pnl; p=2)

    # Bootstrap confidence interval for W₁
    n_live = length(live_pnl)
    n_bt   = length(backtest_pnl)
    boot_W1 = zeros(n_bootstrap)
    for b in 1:n_bootstrap
        boot_live = live_pnl[rand(1:n_live, n_live)]
        boot_bt   = backtest_pnl[rand(1:n_bt, n_bt)]
        boot_W1[b] = wasserstein_1d(boot_live, boot_bt)
    end
    ci = quantile(boot_W1, [α/2, 1 - α/2])

    # Summary statistics
    println("P&L Distribution Comparison:")
    println("  Live:     mean=$(round(mean(live_pnl),digits=4)), std=$(round(std(live_pnl),digits=4))")
    println("  Backtest: mean=$(round(mean(backtest_pnl),digits=4)), std=$(round(std(backtest_pnl),digits=4))")
    println("  W₁ distance: $(round(W1, digits=4))")
    println("  W₂ distance: $(round(W2, digits=4))")
    println("  $(round(Int,(1-α)*100))% Bootstrap CI for W₁: [$(round(ci[1],digits=4)), $(round(ci[2],digits=4))]")

    # Normalised distance (relative to scale)
    scale = max(std(live_pnl), std(backtest_pnl))
    W1_norm = W1 / scale
    println("  Normalised W₁/σ: $(round(W1_norm, digits=4))")
    println("  Interpretation: $(W1_norm < 0.1 ? "distributions are similar" : W1_norm < 0.5 ? "moderate divergence" : "substantial divergence")")

    return (W1=W1, W2=W2, bootstrap_ci=ci, W1_normalised=W1_norm)
end

# ─────────────────────────────────────────────────────────────────────────────
# REGIME COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

"""
Compare return distributions across market regimes.
regimes: vector of regime labels (e.g., 1, 2, 3 for bull/bear/sideways)
returns: aligned vector of returns

Returns pairwise OT distance matrix between regime-conditional distributions.
"""
function regime_ot_distances(returns::Vector{Float64}, regimes::Vector{Int})
    unique_regimes = sort(unique(regimes))
    K = length(unique_regimes)
    D = zeros(K, K)

    regime_returns = [returns[regimes .== r] for r in unique_regimes]

    for i in 1:K, j in i+1:K
        d = wasserstein_1d(regime_returns[i], regime_returns[j])
        D[i,j] = d
        D[j,i] = d
    end

    println("Regime OT Distance Matrix:")
    println("  Regimes: $unique_regimes")
    for i in 1:K
        println("  $(unique_regimes[i]): $(round.(D[i,:], digits=4))")
    end

    return D, regime_returns
end

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL DRIFT DETECTION: CUSUM ON OT DISTANCES
# ─────────────────────────────────────────────────────────────────────────────

"""
Sequential distribution drift detection using CUSUM on Wasserstein distances.

At each time step t, compute W₁ between a reference window and a sliding
current window. CUSUM statistic:
    S_t = max(0, S_{t-1} + W₁(t) - k)

where k is the allowable slack. Alarm when S_t > h.

Returns: (ot_distances, cusum_stats, alarm_times)
"""
function drift_detection(data::Vector{Float64};
                          ref_window=100, test_window=50, step=10,
                          slack_k=nothing, threshold_h=nothing)

    n = length(data)
    ref = data[1:ref_window]

    # Estimate baseline OT noise level
    baseline_distances = Float64[]
    for _ in 1:50
        perm = randperm(ref_window)
        w = wasserstein_1d(ref[perm[1:ref_window÷2]], ref[perm[ref_window÷2+1:end]])
        push!(baseline_distances, w)
    end
    baseline_W1 = mean(baseline_distances)
    baseline_std = std(baseline_distances)

    k = isnothing(slack_k) ? baseline_W1 + baseline_std : slack_k
    h = isnothing(threshold_h) ? 5 * baseline_std : threshold_h

    println("Drift Detection (CUSUM on W₁):")
    println("  Reference window: $ref_window, Test window: $test_window")
    println("  Baseline W₁: $(round(baseline_W1, digits=4))")
    println("  Slack k=$(round(k,digits=4)), Threshold h=$(round(h,digits=4))")

    # Sliding window computation
    t_starts = ref_window+1 : step : n-test_window
    ot_dists = Float64[]
    cusum    = Float64[]
    S = 0.0

    for t in t_starts
        window = data[t:t+test_window-1]
        w1 = wasserstein_1d(ref, window)
        push!(ot_dists, w1)

        S = max(0.0, S + w1 - k)
        push!(cusum, S)
    end

    alarms = findall(cusum .> h)
    alarm_times = isempty(alarms) ? Int[] : [t_starts[a] for a in alarms]

    println("  Detected $(length(alarm_times)) drift alarms")
    isempty(alarm_times) || println("  First alarm at observation $(alarm_times[1])")

    return ot_dists, cusum, alarm_times, collect(t_starts)
end

# ─────────────────────────────────────────────────────────────────────────────
# WASSERSTEIN BARYCENTER
# ─────────────────────────────────────────────────────────────────────────────

"""
Wasserstein barycenter for 1D distributions (exact via quantile averaging).

For 1D empirical measures with equal weights λ₁,...,λₖ (summing to 1),
the Wasserstein-2 barycenter is:
    F_bar⁻¹(t) = Σₖ λₖ · Fₖ⁻¹(t)

i.e., average the quantile functions (inverse CDFs).

This is the multi-period portfolio blend: mix return distributions
from different time periods or strategies.
"""
function wasserstein_barycenter_1d(distributions::Vector{Vector{Float64}},
                                    weights::Vector{Float64}=Float64[];
                                    n_quantiles=1000)
    K = length(distributions)
    isempty(weights) && (weights = fill(1.0/K, K))
    abs(sum(weights) - 1.0) < 1e-10 || throw(ArgumentError("Weights must sum to 1"))

    # Evaluate quantile functions at common grid
    q_grid = range(0.001, 0.999, length=n_quantiles) |> collect
    Q = zeros(n_quantiles, K)
    for k in 1:K
        Q[:,k] = quantile.(Ref(distributions[k]), q_grid)
    end

    # Barycenter quantile function: weighted average
    Q_bar = Q * weights

    return Q_bar, q_grid
end

"""
2D Wasserstein barycenter via iterative Sinkhorn algorithm (Cuturi & Doucet 2014).
Returns barycenter distribution as weighted mixture.
"""
function wasserstein_barycenter_2d(distributions::Vector{Matrix{Float64}},
                                    weights::Vector{Float64}=Float64[],
                                    reg::Float64=0.01;
                                    max_iter=100, n_support=100)
    K = length(distributions)
    isempty(weights) && (weights = fill(1.0/K, K))
    d = size(distributions[1], 2)

    # Initialise barycenter support as pooled sample
    all_data = vcat(distributions...)
    idx = rand(1:size(all_data,1), n_support)
    Z = all_data[idx, :]  # n_support × d

    b = fill(1.0/n_support, n_support)

    for iter in 1:max_iter
        Z_new = zeros(n_support, d)
        total_weight = 0.0

        for k in 1:K
            n_k = size(distributions[k], 1)
            a_k = fill(1.0/n_k, n_k)

            C = cost_matrix(Z, distributions[k])
            _, P = sinkhorn(b, a_k, C, reg)

            # Displacement interpolation: push Z towards distributions[k]
            T_k = (P * distributions[k]) ./ (P * ones(n_k))  # barycentric projection
            Z_new .+= weights[k] .* T_k
        end

        Z = Z_new
    end

    return Z
end

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Plot transport plan as heatmap for 2D OT.
"""
function plot_transport_plan(X::Matrix{Float64}, Y::Matrix{Float64},
                              P::Matrix{Float64};
                              title="Optimal Transport Plan")
    p = heatmap(P, xlabel="Source index", ylabel="Target index",
                title=title, color=:viridis, colorbar_title="Mass")
    return p
end

"""
Compare two 1D distributions: histogram overlay + CDF + transport map.
"""
function plot_distribution_comparison(X::Vector{Float64}, Y::Vector{Float64};
                                       label_X="Distribution 1",
                                       label_Y="Distribution 2",
                                       title="Distribution Comparison")
    W1 = wasserstein_1d(X, Y)
    W2 = wasserstein_1d(X, Y; p=2)

    # Histogram
    x_min = min(minimum(X), minimum(Y))
    x_max = max(maximum(X), maximum(Y))
    bins = range(x_min, x_max, length=50)

    p1 = histogram(X, normalize=:pdf, bins=bins, alpha=0.5, label=label_X,
                   title="$title\nW₁=$(round(W1,digits=4)), W₂=$(round(W2,digits=4))")
    histogram!(p1, Y, normalize=:pdf, bins=bins, alpha=0.5, label=label_Y)

    # CDF comparison
    x_grid = range(x_min, x_max, length=300)
    F_X = [mean(X .<= x) for x in x_grid]
    F_Y = [mean(Y .<= x) for x in x_grid]

    p2 = plot(x_grid, F_X, label=label_X, linewidth=2, title="CDF Comparison")
    plot!(p2, x_grid, F_Y, label=label_Y, linewidth=2, linestyle=:dash)
    fill_between!(p2, x_grid, F_X, F_Y, alpha=0.2, color=:gray, label="Gap")

    # Quantile-quantile plot (OT map)
    n_q = 200
    q_grid = range(0.01, 0.99, length=n_q)
    qX = quantile.(Ref(X), q_grid)
    qY = quantile.(Ref(Y), q_grid)

    p3 = scatter(qX, qY, markersize=2, color=:blue, label="Q-Q",
                 xlabel=label_X, ylabel=label_Y, title="Q-Q Plot (OT Map)")
    plot!(p3, [minimum(qX), maximum(qX)], [minimum(qX), maximum(qX)],
          color=:red, linestyle=:dash, label="y=x")

    return plot(p1, p2, p3, layout=(1,3), size=(1200, 350))
end

# Approximate fill_between using ribbon
function fill_between!(p, x, y1, y2; kwargs...)
    mid = (y1 .+ y2) ./ 2
    half_ribbon = abs.(y2 .- y1) ./ 2
    plot!(p, x, mid; ribbon=half_ribbon, kwargs...)
end

"""
Plot drift detection results.
"""
function plot_drift_detection(data, ot_dists, cusum, alarm_times, t_starts;
                               threshold_h, title="Distribution Drift Detection")
    p1 = plot(t_starts, ot_dists, label="W₁(t)", color=:blue, linewidth=1.5,
              title="Wasserstein Distance", ylabel="W₁", xlabel="Time")

    p2 = plot(t_starts, cusum, label="CUSUM", color=:orange, linewidth=1.5,
              title="CUSUM Statistic", ylabel="S_t", xlabel="Time")
    hline!(p2, [threshold_h], color=:red, linestyle=:dash, label="Threshold h")
    for at in alarm_times
        vline!(p2, [at], color=:red, alpha=0.4, label=nothing)
    end

    p3 = plot(data, label="Data", color=:gray, linewidth=0.5,
              title="Time Series", ylabel="Value", xlabel="Observation")
    for at in alarm_times
        vline!(p3, [at], color=:red, alpha=0.6, label=nothing)
    end

    return plot(p3, p1, p2, layout=(3,1), size=(900, 700), suptitle=title)
end

"""
Visualise Wasserstein barycenter for 1D distributions.
"""
function plot_barycenter(distributions, weights=Float64[];
                          labels=String[])
    K = length(distributions)
    isempty(weights) && (weights = fill(1.0/K, K))
    isempty(labels) && (labels = ["Dist $k" for k in 1:K])

    Q_bar, q_grid = wasserstein_barycenter_1d(distributions, weights)

    p = plot(title="Wasserstein Barycenter", xlabel="Value", ylabel="Density")
    colors = palette(:tab10, K)
    for k in 1:K
        Q_k = quantile.(Ref(distributions[k]), q_grid)
        # Numerical density from quantile function
        dQ = diff(Q_k) ./ diff(q_grid)
        x_mid = (Q_k[1:end-1] .+ Q_k[2:end]) ./ 2
        plot!(p, x_mid, 1 ./ dQ, label="$(labels[k]) (w=$(round(weights[k],digits=2)))",
              color=colors[k], alpha=0.5, linewidth=1.5)
    end

    # Barycenter density
    dQ_bar = diff(Q_bar) ./ diff(q_grid)
    x_bar_mid = (Q_bar[1:end-1] .+ Q_bar[2:end]) ./ 2
    plot!(p, x_bar_mid, 1 ./ dQ_bar, label="Barycenter",
          color=:black, linewidth=2.5)

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

function demo()
    Random.seed!(42)
    println("=" ^ 60)
    println("Optimal Transport Demo")
    println("=" ^ 60)

    # 1. Basic 1D comparison
    println("\n1. 1D Wasserstein Distances")
    X = randn(500)
    Y = randn(500) .* 1.5 .+ 0.5  # shifted, scaled normal
    W1 = wasserstein_1d(X, Y)
    W2 = wasserstein_1d(X, Y; p=2)
    println("   N(0,1) vs N(0.5,1.5²): W₁=$(round(W1,digits=4)), W₂=$(round(W2,digits=4))")

    # 2. P&L comparison: live vs backtest
    println("\n2. Live vs Backtest P&L Comparison")
    backtest_pnl = randn(252) .* 1000 .+ 50   # strategy returns
    live_pnl     = randn(252) .* 1200 .+ 30   # live: more volatile, slightly lower mean
    result = compare_pnl_distributions(live_pnl, backtest_pnl)

    # 3. Regime comparison
    println("\n3. Regime-Conditional Distribution Comparison")
    returns_all = vcat(randn(200) .* 0.01,        # calm regime
                       randn(100) .* 0.03 .- 0.005, # crash regime
                       randn(150) .* 0.015)         # recovery
    regime_labels = vcat(fill(1, 200), fill(2, 100), fill(3, 150))
    D_regime, _ = regime_ot_distances(returns_all, regime_labels)

    # 4. Sinkhorn 2D
    println("\n4. Sinkhorn Regularised OT (2D)")
    X2d = randn(100, 2)
    Y2d = randn(100, 2) .+ [1.0 0.5]
    W_sink, P = wasserstein_2d(X2d, Y2d, 0.05)
    println("   W₂ (Sinkhorn, ε=0.05): $(round(W_sink, digits=4))")

    # 5. Sliced Wasserstein
    println("\n5. Sliced Wasserstein Distance")
    SW = sliced_wasserstein(X2d, Y2d, 500)
    println("   Sliced W₂ (500 proj): $(round(SW, digits=4))")

    # 6. Drift detection
    println("\n6. Sequential Drift Detection")
    # Introduce distribution change at observation 300
    data_stationary = randn(500)
    data_shifted    = randn(200) .+ 2.0
    data_combined   = vcat(data_stationary, data_shifted)
    ot_d, cusum, alarms, t_starts = drift_detection(data_combined;
                                                      ref_window=100,
                                                      test_window=50,
                                                      step=5)

    # 7. Barycenter
    println("\n7. Wasserstein Barycenter")
    dists = [randn(200) .+ k for k in [-2, 0, 2]]
    wts = [0.3, 0.4, 0.3]
    Q_bar, _ = wasserstein_barycenter_1d(dists, wts)
    println("   Barycenter median: $(round(median(Q_bar), digits=3))")
    println("   Expected (weighted): $(round(dot([-2,0,2], wts), digits=3))")

    # Plots
    p1 = plot_distribution_comparison(X, Y; label_X="N(0,1)", label_Y="N(0.5,2.25)")
    savefig(p1, "ot_comparison.png")
    println("\nSaved ot_comparison.png")

    p2 = plot_barycenter(dists, wts; labels=["μ=-2", "μ=0", "μ=2"])
    savefig(p2, "ot_barycenter.png")
    println("Saved ot_barycenter.png")

    if !isempty(alarms)
        threshold_h = 5 * std([wasserstein_1d(data_combined[1:50], data_combined[51:100])])
        p3 = plot_drift_detection(data_combined, ot_d, cusum, alarms, t_starts;
                                   threshold_h=threshold_h)
        savefig(p3, "ot_drift.png")
        println("Saved ot_drift.png")
    end

    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
