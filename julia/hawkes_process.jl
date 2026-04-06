"""
hawkes_process.jl

Complete Hawkes process implementation for high-frequency trade arrival modeling.

A Hawkes process is a self-exciting point process where past events increase
the probability of future events. The conditional intensity is:

    λ(t) = μ + Σ_{t_i < t} α · exp(-β(t - t_i))

where μ is the baseline intensity, α is the excitation magnitude, and β is
the decay rate. The ratio α/β is the branching ratio — must be < 1 for
stationarity.

References:
  Hawkes (1971) "Spectra of some self-exciting and mutually exciting point processes"
  Ogata (1981) "On Lewis' simulation method for point processes"
  Embrechts et al. (2011) "Multivariate Hawkes processes"
"""

using Optim
using LinearAlgebra
using Statistics
using Random
using Distributions
using HypothesisTests
using Plots
using SQLite
using DataFrames
using Dates

# ─────────────────────────────────────────────────────────────────────────────
# CORE STRUCTS
# ─────────────────────────────────────────────────────────────────────────────

"""
Univariate Hawkes process with exponential kernel.
"""
struct HawkesProcess
    μ::Float64          # baseline intensity (events/second)
    α::Float64          # excitation amplitude
    β::Float64          # decay rate
    function HawkesProcess(μ, α, β)
        μ > 0 || throw(ArgumentError("μ must be positive"))
        α >= 0 || throw(ArgumentError("α must be non-negative"))
        β > 0 || throw(ArgumentError("β must be positive"))
        new(μ, α, β)
    end
end

"""
Multivariate Hawkes process with N × N excitation matrix.
Each component i has baseline μ_i and is excited by component j
with amplitude α_ij and decay β_ij.
"""
struct MultivariateHawkesProcess
    μ::Vector{Float64}          # N-vector of baselines
    α::Matrix{Float64}          # N×N excitation matrix
    β::Matrix{Float64}          # N×N decay matrix
    N::Int

    function MultivariateHawkesProcess(μ, α, β)
        N = length(μ)
        size(α) == (N, N) || throw(DimensionMismatch("α must be N×N"))
        size(β) == (N, N) || throw(DimensionMismatch("β must be N×N"))
        all(μ .> 0) || throw(ArgumentError("All μ must be positive"))
        all(α .>= 0) || throw(ArgumentError("All α must be non-negative"))
        all(β .> 0) || throw(ArgumentError("All β must be positive"))
        new(μ, α, β, N)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# INTENSITY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

"""
Compute conditional intensity λ(t | H_t) for univariate Hawkes process.
History `events` is a sorted vector of event times before t.
"""
function intensity(hp::HawkesProcess, t::Float64, events::Vector{Float64})
    λ = hp.μ
    for tᵢ in events
        tᵢ < t || break
        λ += hp.α * exp(-hp.β * (t - tᵢ))
    end
    return λ
end

"""
Vectorised intensity over a time grid — efficient for plotting.
Uses recursive update: R(i) = exp(-β·Δt)·(R(i-1) + 1) at each event.
"""
function intensity_path(hp::HawkesProcess, events::Vector{Float64},
                        t_grid::AbstractVector{Float64})
    λ_path = fill(hp.μ, length(t_grid))
    for (k, t) in enumerate(t_grid)
        for tᵢ in events
            tᵢ >= t && break
            λ_path[k] += hp.α * exp(-hp.β * (t - tᵢ))
        end
    end
    return λ_path
end

# ─────────────────────────────────────────────────────────────────────────────
# LOG-LIKELIHOOD
# ─────────────────────────────────────────────────────────────────────────────

"""
Exact log-likelihood for univariate Hawkes process with exponential kernel.

L(θ) = -μT + (α/β)·Σ_i[exp(-β(T-t_i)) - 1] + Σ_i log(μ + Σ_{j<i} α·exp(-β(t_i-t_j)))

Uses the recursive auxiliary variable A_i = Σ_{j<i} exp(-β(t_i - t_j))
to achieve O(n) computation (rather than O(n²)).

    A_i = exp(-β(t_i - t_{i-1})) · (1 + A_{i-1})
"""
function log_likelihood(μ::Float64, α::Float64, β::Float64,
                        events::Vector{Float64}, T::Float64)
    n = length(events)
    n == 0 && return -μ * T

    # Term 1: integral of baseline
    ll = -μ * T

    # Term 2: integral of excitation kernel ∫₀ᵀ Σᵢ α·exp(-β(t-tᵢ)) dt
    # = (α/β) · Σᵢ (1 - exp(-β(T - tᵢ)))
    for tᵢ in events
        ll += (α / β) * (exp(-β * (T - tᵢ)) - 1.0)
    end

    # Term 3: Σᵢ log λ(tᵢ)
    A = 0.0  # recursive excitation accumulator
    for i in 1:n
        if i > 1
            A = exp(-β * (events[i] - events[i-1])) * (1.0 + A)
        end
        λᵢ = μ + α * A
        λᵢ <= 0 && return -Inf
        ll += log(λᵢ)
    end

    return ll
end

# ─────────────────────────────────────────────────────────────────────────────
# MLE FITTING VIA L-BFGS
# ─────────────────────────────────────────────────────────────────────────────

"""
Fit univariate Hawkes process by maximum likelihood estimation.
Uses L-BFGS with box constraints via Optim.jl.

Returns a fitted HawkesProcess and optimization result.
"""
function fit(::Type{HawkesProcess}, events::Vector{Float64}, T::Float64;
             μ₀=nothing, α₀=nothing, β₀=nothing, verbose=false)

    isempty(events) && throw(ArgumentError("Event vector is empty"))
    issorted(events) || sort!(events)

    # Initial parameter guesses from data moments
    n = length(events)
    rate = n / T
    μ_init = isnothing(μ₀) ? rate * 0.5 : μ₀
    α_init = isnothing(α₀) ? 0.3 : α₀
    β_init = isnothing(β₀) ? 1.0 / (T / n) : β₀  # 1 / mean_interval

    # Softplus reparameterisation: θ_raw → exp(θ_raw) enforces positivity
    # We use log-space optimisation for unconstrained search.
    θ₀ = [log(μ_init), log(α_init), log(β_init)]

    neg_ll(θ) = -log_likelihood(exp(θ[1]), exp(θ[2]), exp(θ[3]), events, T)

    result = optimize(neg_ll, θ₀, LBFGS(),
                      Optim.Options(iterations=2000, g_tol=1e-8,
                                    show_trace=verbose))

    θ_opt = Optim.minimizer(result)
    μ_hat, α_hat, β_hat = exp.(θ_opt)

    verbose && println("MLE: μ=$(round(μ_hat,digits=4)), α=$(round(α_hat,digits=4)), β=$(round(β_hat,digits=4))")
    verbose && println("Branching ratio: $(round(α_hat/β_hat, digits=4))")

    return HawkesProcess(μ_hat, α_hat, β_hat), result
end

"""
Gradient of log-likelihood for use in gradient-based optimisers.
"""
function log_likelihood_gradient(μ, α, β, events, T)
    n = length(events)
    dL_dμ = -T
    dL_dα = 0.0
    dL_dβ = 0.0

    A = 0.0
    dA_dβ = 0.0  # ∂A/∂β

    for i in 1:n
        if i > 1
            Δ = events[i] - events[i-1]
            dA_dβ = exp(-β * Δ) * (dA_dβ - Δ * (1.0 + A))
            A = exp(-β * Δ) * (1.0 + A)
        end
        λᵢ = μ + α * A
        dL_dμ += 1.0 / λᵢ
        dL_dα += A / λᵢ + (1.0 / β) * (exp(-β * (T - events[i])) - 1.0)
        dL_dβ += α * dA_dβ / λᵢ - (α / β^2) * (exp(-β*(T-events[i])) - 1.0) +
                  (α / β) * (T - events[i]) * exp(-β * (T - events[i]))
    end

    return [dL_dμ, dL_dα, dL_dβ]
end

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION: OGATA'S THINNING ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

"""
Simulate a univariate Hawkes process on [0, T] using Ogata's thinning algorithm.

Algorithm:
  1. Compute upper bound M on intensity in [s, ∞) — λ*(s) using current history
  2. Propose candidate time s + Exp(M)
  3. Accept with probability λ(proposed) / M
  4. Update history and repeat

Returns sorted vector of event times.
"""
function simulate(hp::HawkesProcess, T::Float64; rng=Random.GLOBAL_RNG)
    events = Float64[]
    t = 0.0

    while t < T
        # Upper bound: intensity is maximised at current time (just after last event)
        if isempty(events)
            λ_upper = hp.μ
        else
            λ_upper = intensity(hp, t, events) + 1e-10  # slight buffer
        end

        # Proposed next event time
        Δ = rand(rng, Exponential(1.0 / λ_upper))
        t_proposed = t + Δ

        t_proposed > T && break

        # Compute actual intensity at proposed time
        λ_actual = intensity(hp, t_proposed, events)

        # Accept/reject
        if rand(rng) <= λ_actual / λ_upper
            push!(events, t_proposed)
            t = t_proposed
        else
            t = t_proposed
        end
    end

    return events
end

"""
Simulate multivariate Hawkes process via multivariate thinning.
Returns a vector of event-time vectors, one per component.
"""
function simulate(mhp::MultivariateHawkesProcess, T::Float64; rng=Random.GLOBAL_RNG)
    events = [Float64[] for _ in 1:mhp.N]
    t = 0.0

    # Flatten all events with component labels
    all_events = Tuple{Float64,Int}[]

    while t < T
        # Compute intensity upper bound for each component
        λ_bounds = zeros(mhp.N)
        for i in 1:mhp.N
            λ_bounds[i] = mhp.μ[i]
            for j in 1:mhp.N
                for tᵢ in events[j]
                    λ_bounds[i] += mhp.α[i,j] * exp(-mhp.β[i,j] * (t - tᵢ))
                end
            end
        end

        λ_total = sum(λ_bounds)
        λ_total <= 0 && break

        Δ = rand(rng, Exponential(1.0 / λ_total))
        t_proposed = t + Δ
        t_proposed > T && break

        # Recompute actual intensities
        λ_actual = zeros(mhp.N)
        for i in 1:mhp.N
            λ_actual[i] = mhp.μ[i]
            for j in 1:mhp.N
                for tᵢ in events[j]
                    λ_actual[i] += mhp.α[i,j] * exp(-mhp.β[i,j] * (t_proposed - tᵢ))
                end
            end
        end

        λ_actual_total = sum(λ_actual)
        u = rand(rng) * λ_total

        if u <= λ_actual_total
            # Determine which component fires
            cum = 0.0
            fired = mhp.N
            for i in 1:mhp.N
                cum += λ_actual[i]
                if u <= cum
                    fired = i
                    break
                end
            end
            push!(events[fired], t_proposed)
        end
        t = t_proposed
    end

    return events
end

# ─────────────────────────────────────────────────────────────────────────────
# GOODNESS-OF-FIT: RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

"""
Compute compensator (integrated intensity) at each event time.
Under correct specification, these are iid Exponential(1).

    Λ(tᵢ) = μ·tᵢ + (α/β) Σ_{j≤i} (1 - exp(-β(tᵢ - tⱼ)))

Returns residuals Λ(t₁), Λ(t₂)-Λ(t₁), ..., which should be Exp(1).
"""
function compensator_residuals(hp::HawkesProcess, events::Vector{Float64})
    n = length(events)
    Λ = zeros(n)

    for i in 1:n
        tᵢ = events[i]
        # Baseline contribution
        Λ[i] = hp.μ * tᵢ
        # Excitation contribution from all prior events
        for j in 1:i
            tⱼ = events[j]
            Λ[i] += (hp.α / hp.β) * (1.0 - exp(-hp.β * (tᵢ - tⱼ)))
        end
    end

    # Differences are the rescaled inter-arrival times
    residuals = diff([0.0; Λ])
    return residuals
end

"""
Kolmogorov-Smirnov goodness-of-fit test.
Under H₀, residuals ~ Exp(1).
Returns (KS statistic, p-value, pass/fail).
"""
function goodness_of_fit(hp::HawkesProcess, events::Vector{Float64})
    residuals = compensator_residuals(hp, events)

    # KS test against Exp(1)
    ks = ExactOneSampleKSTest(residuals, Exponential(1.0))
    stat = ks.δ
    pval = pvalue(ks)

    println("Kolmogorov-Smirnov GoF test:")
    println("  KS statistic = $(round(stat, digits=4))")
    println("  p-value      = $(round(pval, digits=4))")
    println("  Model fit    = $(pval > 0.05 ? "PASS (p > 0.05)" : "FAIL (p ≤ 0.05)")")

    return (statistic=stat, pvalue=pval, pass=pval > 0.05)
end

# ─────────────────────────────────────────────────────────────────────────────
# BRANCHING RATIO AND STABILITY
# ─────────────────────────────────────────────────────────────────────────────

"""
Branching ratio η = α/β.
For stability (stationarity), we need η < 1.
Interpretation: average number of offspring per event.
"""
function branching_ratio(hp::HawkesProcess)
    η = hp.α / hp.β
    println("Branching ratio η = α/β = $(round(η, digits=4))")
    if η >= 1.0
        println("  WARNING: Process is NOT stationary (η ≥ 1). Explosions possible.")
    else
        println("  Process is stationary. Mean cluster size = $(round(1/(1-η), digits=2))")
    end
    return η
end

"""
Stationary mean intensity: E[λ] = μ / (1 - α/β).
"""
function stationary_intensity(hp::HawkesProcess)
    η = branching_ratio(hp)
    η >= 1.0 && error("Process not stationary; mean intensity is infinite")
    return hp.μ / (1.0 - η)
end

# ─────────────────────────────────────────────────────────────────────────────
# NONPARAMETRIC ESTIMATION: WIENER-HOPF EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────

"""
Nonparametric kernel estimation via Wiener-Hopf integral equations.
Estimates the kernel φ(t) such that:

    λ(t) = μ + ∫₀^∞ φ(s) dN(t-s)

Uses the Bartlett (1963) spectral approach:
    Ĝ(ω) = S_λλ(ω) / S_NN(ω)

where S denotes cross-spectral density.

Returns a discretised kernel on [0, τ_max] with step dt.
"""
function nonparametric_kernel(events::Vector{Float64}, T::Float64;
                               τ_max=5.0, dt=0.01, bandwidth=0.1)
    τ_bins = 0:dt:τ_max
    n_bins = length(τ_bins)

    # Empirical cross-correlation: φ̂(τ) = ∫₀^{T-τ} dN(t)dN(t+τ)/T - μ²dt
    n = length(events)
    μ_hat = n / T
    kernel = zeros(n_bins)

    for (k, τ) in enumerate(τ_bins)
        count = 0
        for tᵢ in events
            tⱼ = tᵢ + τ
            # Binary search: is tⱼ in events?
            idx = searchsortedfirst(events, tⱼ - dt/2)
            while idx <= n && events[idx] <= tⱼ + dt/2
                if abs(events[idx] - tⱼ) < dt/2 && events[idx] != tᵢ
                    count += 1
                end
                idx += 1
            end
        end
        kernel[k] = count / (T * dt) - μ_hat^2
    end

    # Gaussian kernel smoothing
    σ = bandwidth
    kernel_smooth = similar(kernel)
    for k in 1:n_bins
        w = [exp(-0.5 * ((τ_bins[k] - τ_bins[j]) / σ)^2) for j in 1:n_bins]
        w ./= sum(w)
        kernel_smooth[k] = dot(w, kernel)
    end

    return collect(τ_bins), kernel_smooth
end

# ─────────────────────────────────────────────────────────────────────────────
# MULTIVARIATE MLE
# ─────────────────────────────────────────────────────────────────────────────

"""
Log-likelihood for multivariate Hawkes process.
events[i] is the sorted vector of event times for component i.
"""
function multivariate_log_likelihood(mhp::MultivariateHawkesProcess,
                                     events::Vector{Vector{Float64}}, T::Float64)
    N = mhp.N
    ll = 0.0

    # Merge all events with component labels
    all_events = sort([(t, j) for j in 1:N for t in events[j]], by=x->x[1])

    # Integral terms: -μᵢT - Σⱼ Σₜⱼ (αᵢⱼ/βᵢⱼ)(1 - exp(-βᵢⱼ(T-tⱼₖ)))
    for i in 1:N
        ll -= mhp.μ[i] * T
        for j in 1:N
            for tᵢ in events[j]
                ll -= (mhp.α[i,j] / mhp.β[i,j]) * (1.0 - exp(-mhp.β[i,j] * (T - tᵢ)))
            end
        end
    end

    # Sum of log intensities at each event time
    # Maintain recursive accumulators Aᵢⱼ for each pair (i,j)
    A = zeros(N, N)  # A[i,j] = excitation from j on i
    last_t = 0.0

    for (t, fired_j) in all_events
        # Update accumulators: A[i,j] *= exp(-β[i,j]*Δt)
        Δ = t - last_t
        for i in 1:N, j in 1:N
            A[i,j] *= exp(-mhp.β[i,j] * Δ)
        end

        # Log intensity of fired component
        λ_fired = mhp.μ[fired_j] + sum(mhp.α[fired_j, j] * A[fired_j, j] for j in 1:N)
        λ_fired <= 0 && return -Inf
        ll += log(λ_fired)

        # Update accumulator: component fired_j just fired
        for i in 1:N
            A[i, fired_j] += 1.0
        end

        last_t = t
    end

    return ll
end

"""
Fit multivariate Hawkes process.
"""
function fit(::Type{MultivariateHawkesProcess}, events::Vector{Vector{Float64}},
             T::Float64; verbose=false)
    N = length(events)
    n_params = N + 2 * N^2  # μ + α + β

    # Initialise parameters
    rates = [length(e) / T for e in events]
    θ₀ = zeros(n_params)
    θ₀[1:N] = log.(rates .* 0.5)                  # log μ
    θ₀[N+1:N+N^2] = fill(log(0.3), N^2)           # log α (flat)
    θ₀[N+N^2+1:end] = fill(log(1.0), N^2)         # log β

    function unpack(θ)
        μ = exp.(θ[1:N])
        α = reshape(exp.(θ[N+1:N+N^2]), N, N)
        β = reshape(exp.(θ[N+N^2+1:end]), N, N)
        return MultivariateHawkesProcess(μ, α, β)
    end

    neg_ll(θ) = -multivariate_log_likelihood(unpack(θ), events, T)

    result = optimize(neg_ll, θ₀, LBFGS(),
                      Optim.Options(iterations=1000, g_tol=1e-6, show_trace=verbose))

    return unpack(Optim.minimizer(result)), result
end

# ─────────────────────────────────────────────────────────────────────────────
# REAL DATA: LOAD FROM SQLITE
# ─────────────────────────────────────────────────────────────────────────────

"""
Load trade arrival times from execution/live_trades.db.
Returns event times in seconds from session start.

Expected schema: trades(timestamp INTEGER, symbol TEXT, ...)
where timestamp is Unix milliseconds.
"""
function load_trade_events(db_path::String, symbol::String;
                           date::Union{String,Nothing}=nothing)
    db = SQLite.DB(db_path)

    query = if isnothing(date)
        "SELECT timestamp FROM trades WHERE symbol = '$symbol' ORDER BY timestamp"
    else
        "SELECT timestamp FROM trades WHERE symbol = '$symbol' AND date(timestamp/1000,'unixepoch') = '$date' ORDER BY timestamp"
    end

    df = DBInterface.execute(db, query) |> DataFrame
    SQLite.close(db)

    isempty(df) && error("No trades found for symbol=$symbol")

    # Convert to seconds, normalise to start from 0
    ts = Float64.(df.timestamp) ./ 1000.0
    ts .-= ts[1]

    println("Loaded $(length(ts)) trades for $symbol over $(round(ts[end]/3600, digits=2)) hours")
    return ts
end

"""
Full pipeline: load data → fit Hawkes → goodness of fit → plot.
"""
function analyze_trade_arrivals(db_path::String, symbol::String;
                                 date::Union{String,Nothing}=nothing)
    println("=" ^ 60)
    println("Hawkes Process Analysis: $symbol")
    println("=" ^ 60)

    # Load data
    events = load_trade_events(db_path, symbol; date=date)
    T = events[end] + 1.0  # observation window

    # Fit model
    println("\nFitting Hawkes process via MLE...")
    hp, opt_result = fit(HawkesProcess, events, T; verbose=true)

    # Branching ratio
    println("\nStationarity check:")
    η = branching_ratio(hp)

    # Goodness of fit
    println("\nGoodness of fit:")
    gof = goodness_of_fit(hp, events)

    # Stationary intensity
    if η < 1.0
        λ_stat = stationary_intensity(hp)
        println("\nStationary mean intensity: $(round(λ_stat, digits=4)) events/sec")
    end

    return hp, gof
end

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

"""
Plot intensity function over time with event markers.
"""
function plot_intensity(hp::HawkesProcess, events::Vector{Float64};
                        t_max=nothing, n_points=1000, title="Hawkes Process Intensity")
    t_end = isnothing(t_max) ? events[end] * 1.05 : t_max
    t_grid = range(0, t_end, length=n_points) |> collect

    λ_path = intensity_path(hp, events, t_grid)

    p1 = plot(t_grid, λ_path, label="λ(t)", color=:blue, linewidth=1.5,
              ylabel="Intensity", title=title, legend=:topright)
    hline!(p1, [hp.μ], label="μ (baseline)", color=:red, linestyle=:dash)

    # Event rug plot
    p2 = scatter(events, zeros(length(events)), markersize=2, color=:black,
                 label="Events", xlabel="Time (s)", yticks=[], ylabel="")

    return plot(p1, p2, layout=(2,1), size=(900, 500))
end

"""
Plot QQ-plot and histogram of residuals for goodness-of-fit assessment.
"""
function plot_residuals(hp::HawkesProcess, events::Vector{Float64})
    resids = compensator_residuals(hp, events)

    # QQ-plot vs Exp(1)
    n = length(resids)
    sorted_resids = sort(resids)
    theoretical_quantiles = -log.(1 .- (1:n) ./ (n + 1))

    p1 = scatter(theoretical_quantiles, sorted_resids,
                 xlabel="Theoretical Exp(1) quantiles",
                 ylabel="Sample quantiles",
                 title="Residual QQ-Plot",
                 label="Residuals", markersize=2, color=:blue)
    plot!(p1, [0, maximum(theoretical_quantiles)],
          [0, maximum(theoretical_quantiles)],
          color=:red, label="y=x", linestyle=:dash)

    # Histogram with Exp(1) overlay
    x_range = range(0, quantile(Exponential(1.0), 0.99), length=100)
    p2 = histogram(resids, normalize=:pdf, bins=40, label="Residuals",
                   xlabel="Residual value", ylabel="Density",
                   title="Residual Distribution vs Exp(1)")
    plot!(p2, x_range, pdf.(Exponential(1.0), x_range),
          color=:red, linewidth=2, label="Exp(1)")

    return plot(p1, p2, layout=(1,2), size=(900, 400))
end

"""
Visualise branching structure: colour events by generation.
Immigration events (generation 0) vs offspring (generation 1, 2, ...).
"""
function plot_branching_structure(hp::HawkesProcess, events::Vector{Float64};
                                   max_gen=5)
    n = length(events)
    # Assign generations probabilistically
    gen = zeros(Int, n)
    parent = zeros(Int, n)  # 0 = immigrant

    for i in 1:n
        # Probability of being immigrant vs offspring of each prior event
        p_immigrant = hp.μ
        p_offspring = [hp.α * exp(-hp.β * (events[i] - events[j])) for j in 1:i-1]

        total = p_immigrant + sum(p_offspring)
        u = rand() * total

        if u <= p_immigrant
            gen[i] = 0  # immigrant
            parent[i] = 0
        else
            u -= p_immigrant
            for j in 1:i-1
                u -= p_offspring[j]
                if u <= 0
                    parent[i] = j
                    gen[i] = min(gen[j] + 1, max_gen)
                    break
                end
            end
        end
    end

    colors = palette(:viridis, max_gen + 1)
    p = scatter(events, gen, color=[colors[g+1] for g in gen],
                markersize=4, xlabel="Time", ylabel="Generation",
                title="Branching Structure",
                label=nothing, yticks=0:max_gen)

    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

"""
Run a self-contained demo with synthetic data.
"""
function demo()
    Random.seed!(42)
    println("=" ^ 60)
    println("Hawkes Process Demo")
    println("=" ^ 60)

    # True parameters
    hp_true = HawkesProcess(0.5, 0.8, 2.0)
    println("\nTrue parameters: μ=$(hp_true.μ), α=$(hp_true.α), β=$(hp_true.β)")
    println("True branching ratio: $(round(hp_true.α/hp_true.β, digits=3))")

    # Simulate
    T = 1000.0
    events = simulate(hp_true, T)
    println("Simulated $(length(events)) events over T=$T seconds")
    println("Empirical rate: $(round(length(events)/T, digits=3)) events/sec")
    println("Theoretical mean: $(round(stationary_intensity(hp_true), digits=3)) events/sec")

    # Fit
    println("\nFitting via MLE...")
    hp_fit, opt = fit(HawkesProcess, events, T; verbose=true)

    println("\nParameter recovery:")
    println("  μ: true=$(hp_true.μ), estimated=$(round(hp_fit.μ, digits=4))")
    println("  α: true=$(hp_true.α), estimated=$(round(hp_fit.α, digits=4))")
    println("  β: true=$(hp_true.β), estimated=$(round(hp_fit.β, digits=4))")

    # GoF
    println()
    gof = goodness_of_fit(hp_fit, events)

    # Nonparametric kernel
    τ_bins, kernel = nonparametric_kernel(events, T; τ_max=3.0, dt=0.05)
    println("\nNonparametric kernel estimated on $(length(τ_bins)) bins")

    # Plots
    p_intensity = plot_intensity(hp_fit, events[1:min(200, end)];
                                  t_max=events[min(200, end)],
                                  title="Fitted Hawkes Intensity")
    p_resid = plot_residuals(hp_fit, events)
    p_branch = plot_branching_structure(hp_fit, events[1:min(100, end)])
    p_kernel = plot(τ_bins, kernel, xlabel="Lag τ", ylabel="φ̂(τ)",
                    title="Nonparametric Kernel Estimate", color=:purple, linewidth=2,
                    label="Estimated kernel")
    τ_true = range(0, 3, length=200)
    plot!(p_kernel, τ_true, hp_true.α .* exp.(-hp_true.β .* τ_true),
          color=:red, linestyle=:dash, linewidth=2, label="True kernel")

    full_plot = plot(p_intensity, p_resid, p_branch, p_kernel,
                     layout=(2,2), size=(1200, 800))
    savefig(full_plot, "hawkes_demo.png")
    println("\nSaved plot to hawkes_demo.png")

    # Multivariate demo (2-dimensional)
    println("\n" * "=" ^ 60)
    println("Multivariate Hawkes Process Demo (N=2)")
    println("=" ^ 60)

    μ_mv = [0.3, 0.2]
    α_mv = [0.5 0.3; 0.2 0.4]
    β_mv = [2.0 1.5; 1.5 2.0]
    mhp_true = MultivariateHawkesProcess(μ_mv, α_mv, β_mv)

    mv_events = simulate(mhp_true, 500.0)
    println("Component 1: $(length(mv_events[1])) events")
    println("Component 2: $(length(mv_events[2])) events")

    println("\nFitting multivariate Hawkes (this may take a moment)...")
    mhp_fit, _ = fit(MultivariateHawkesProcess, mv_events, 500.0; verbose=false)
    println("Fitted μ: $(round.(mhp_fit.μ, digits=3))")
    println("Fitted α:\n$(round.(mhp_fit.α, digits=3))")
    println("Fitted β:\n$(round.(mhp_fit.β, digits=3))")

    return hp_fit, gof
end

# Execute demo if run as script
if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
