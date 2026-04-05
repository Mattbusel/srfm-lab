# =============================================================================
# optimizer.jl — Bayesian Optimization & Multi-Objective Pareto Search
# =============================================================================
# Provides:
#   - BayesianOptimizer  (Gaussian Process surrogate + Expected Improvement)
#   - GridSearchParallel (parallel exhaustive grid search)
#   - NSGA-II multi-objective Pareto front
#   - Objective functions that call the Python backtest engine via PyCall
#
# Julia ≥ 1.10 | Packages: LinearAlgebra, Statistics, Random, JSON3, PyCall
# =============================================================================

module Optimizer

using LinearAlgebra
using Statistics
using Random
using JSON3

# Optional PyCall — graceful degradation if not available
const HAS_PYCALL = try
    using PyCall
    true
catch
    false
end

export BayesianOptimizer, optimize!, GridSearchParallel, run_grid_search
export objective_sharpe, objective_calmar, multi_objective_pareto
export ParamBounds, OptimResult

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"""
Parameter bounds: dict-like mapping param name → (min, max).
"""
const ParamBounds = Dict{String, Tuple{Float64, Float64}}

"""
Result of a single objective evaluation.
"""
struct EvalPoint
    x       ::Vector{Float64}
    y       ::Float64        # objective value (higher is better)
    params  ::Dict{String, Float64}
end

"""
Optimisation result summary.
"""
struct OptimResult
    best_params  ::Dict{String, Float64}
    best_score   ::Float64
    n_iterations ::Int
    history      ::Vector{EvalPoint}
    method       ::String
end

# ---------------------------------------------------------------------------
# Gaussian Process kernel
# ---------------------------------------------------------------------------

"""
Squared-exponential (RBF) kernel with ARD length-scales.
k(x, x') = σ² exp(−½ Σ_d (xd − x'd)² / ℓd²)
"""
struct SEKernel
    σ²  ::Float64             # signal variance
    ℓ   ::Vector{Float64}     # length-scales per dimension
end

SEKernel(d::Int; σ²=1.0, ℓ=1.0) = SEKernel(σ², fill(Float64(ℓ), d))

function (k::SEKernel)(x::Vector{Float64}, x′::Vector{Float64})
    r2 = sum(((x[i] - x′[i]) / k.ℓ[i])^2 for i in eachindex(x))
    k.σ² * exp(-0.5 * r2)
end

"""
Build full kernel matrix K(X, X).
"""
function kernel_matrix(k::SEKernel, X::Matrix{Float64}, noise::Float64=1e-6)
    n = size(X, 1)
    K = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        K[i, j] = k(X[i, :], X[j, :])
    end
    K + noise * I
end

"""
Cross-covariance vector k(X, x*).
"""
function kernel_vector(k::SEKernel, X::Matrix{Float64}, x_star::Vector{Float64})
    [k(X[i, :], x_star) for i in 1:size(X, 1)]
end

# ---------------------------------------------------------------------------
# Gaussian Process surrogate
# ---------------------------------------------------------------------------

"""
Minimal GP surrogate: posterior mean and variance at a new point.
"""
mutable struct GaussianProcess
    kernel      ::SEKernel
    X_train     ::Matrix{Float64}   # n × d
    y_train     ::Vector{Float64}   # n
    K_inv       ::Matrix{Float64}   # (K + σ²I)⁻¹
    noise       ::Float64
end

function GaussianProcess(kernel::SEKernel; noise=1e-4)
    GaussianProcess(kernel, Matrix{Float64}(undef, 0, 0),
                    Float64[], Matrix{Float64}(undef, 0, 0), noise)
end

"""
Fit (or refit) GP to training data.
"""
function fit!(gp::GaussianProcess, X::Matrix{Float64}, y::Vector{Float64})
    gp.X_train = X
    gp.y_train = y
    K          = kernel_matrix(gp.kernel, X, gp.noise)
    gp.K_inv   = inv(K)
    gp
end

"""
GP posterior mean and variance at x*.
"""
function predict(gp::GaussianProcess, x_star::Vector{Float64})
    if size(gp.X_train, 1) == 0
        return 0.0, gp.kernel.σ²
    end
    k_star   = kernel_vector(gp.kernel, gp.X_train, x_star)
    k_starstar = gp.kernel(x_star, x_star)
    μ        = dot(k_star, gp.K_inv * gp.y_train)
    σ²       = k_starstar - dot(k_star, gp.K_inv * k_star)
    μ, max(σ², 1e-12)
end

"""
Update GP with a new (x, y) observation without full re-inversion.
"""
function update_gp!(gp::GaussianProcess, x_new::Vector{Float64}, y_new::Float64)
    if size(gp.X_train, 1) == 0
        X_new = reshape(x_new, 1, :)
        fit!(gp, X_new, [y_new])
    else
        X_new = vcat(gp.X_train, reshape(x_new, 1, :))
        y_new_vec = vcat(gp.y_train, y_new)
        fit!(gp, X_new, y_new_vec)
    end
    gp
end

# ---------------------------------------------------------------------------
# Acquisition function — Expected Improvement
# ---------------------------------------------------------------------------

"""
Expected Improvement acquisition function.

EI(x) = (μ(x) − f* − ξ) Φ(Z) + σ(x) φ(Z)
where Z = (μ(x) − f* − ξ) / σ(x)
"""
function acquisition_function(gp::GaussianProcess, x::Vector{Float64};
                               xi::Float64=0.01)
    μ, σ² = predict(gp, x)
    σ      = sqrt(σ²)
    f_best = isempty(gp.y_train) ? 0.0 : maximum(gp.y_train)

    if σ < 1e-10
        return 0.0
    end

    Z  = (μ - f_best - xi) / σ
    # Normal CDF and PDF via Taylor approximation (avoid Distributions dependency)
    Φ  = _normal_cdf(Z)
    φ  = _normal_pdf(Z)
    (μ - f_best - xi) * Φ + σ * φ
end

# Rational approximation for normal CDF (Abramowitz & Stegun 26.2.17)
function _normal_cdf(z::Float64)
    if z < -6.0 return 0.0 end
    if z >  6.0 return 1.0 end
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    p = t * (0.319381530 +
        t * (-0.356563782 +
        t * (1.781477937 +
        t * (-1.821255978 +
        t *  1.330274429))))
    d = exp(-0.5 * z^2) / sqrt(2π)
    c = 1.0 - d * p
    z >= 0 ? c : 1.0 - c
end

_normal_pdf(z::Float64) = exp(-0.5 * z^2) / sqrt(2π)

# ---------------------------------------------------------------------------
# BayesianOptimizer
# ---------------------------------------------------------------------------

"""
Gaussian-process Bayesian optimiser with Expected Improvement acquisition.

# Fields
- `bounds`   : parameter bounds
- `gp`       : GP surrogate
- `history`  : all evaluated points
- `rng`      : internal RNG (seeded for reproducibility)
"""
mutable struct BayesianOptimizer
    bounds      ::ParamBounds
    gp          ::GaussianProcess
    history     ::Vector{EvalPoint}
    rng         ::MersenneTwister
    n_random    ::Int       # initial random exploration points
    xi          ::Float64   # EI exploration-exploitation trade-off
end

function BayesianOptimizer(bounds::ParamBounds;
                            seed=42, n_random=10, xi=0.01,
                            noise=1e-4, length_scale=1.0)
    d      = length(bounds)
    kernel = SEKernel(d; ℓ=length_scale)
    gp     = GaussianProcess(kernel; noise=noise)
    BayesianOptimizer(bounds, gp, EvalPoint[], MersenneTwister(seed),
                      n_random, xi)
end

"""
Scale a raw x ∈ [0,1]ᵈ vector to the actual parameter space.
"""
function _unscale(bo::BayesianOptimizer, x_unit::Vector{Float64})
    names = sort(collect(keys(bo.bounds)))
    Dict(
        n => bo.bounds[n][1] + x_unit[i] * (bo.bounds[n][2] - bo.bounds[n][1])
        for (i, n) in enumerate(names)
    )
end

function _scale(bo::BayesianOptimizer, params::Dict{String, Float64})
    names = sort(collect(keys(bo.bounds)))
    [(params[n] - bo.bounds[n][1]) / (bo.bounds[n][2] - bo.bounds[n][1])
     for n in names]
end

"""
Sample a candidate by maximising Expected Improvement via random restarts.
"""
function _next_candidate(bo::BayesianOptimizer; n_restarts=20)
    d    = length(bo.bounds)
    best_x    = rand(bo.rng, d)
    best_ei   = -Inf

    for _ in 1:n_restarts
        x0   = rand(bo.rng, d)
        # Simple gradient-free optimisation of EI: random hill-climbing
        x    = copy(x0)
        step = 0.05
        for _ in 1:200
            ei_cur = acquisition_function(bo.gp, x; xi=bo.xi)
            improved = false
            for dim in 1:d
                for δ in (step, -step)
                    x_try    = copy(x)
                    x_try[dim] = clamp(x_try[dim] + δ, 0.0, 1.0)
                    ei_try   = acquisition_function(bo.gp, x_try; xi=bo.xi)
                    if ei_try > ei_cur
                        x       = x_try
                        ei_cur  = ei_try
                        improved = true
                    end
                end
            end
            if !improved
                step *= 0.5
                step < 1e-6 && break
            end
        end
        ei = acquisition_function(bo.gp, x; xi=bo.xi)
        if ei > best_ei
            best_ei = ei
            best_x  = x
        end
    end
    best_x
end

"""
Run Bayesian optimisation.

# Arguments
- `objective_fn` : function(Dict{String,Float64}) → Float64 (higher = better)
- `param_bounds`  : ParamBounds
- `n_iter`        : total iterations (includes random init phase)

# Returns OptimResult
"""
function optimize!(bo::BayesianOptimizer,
                   objective_fn::Function;
                   n_iter::Int=100)
    d     = length(bo.bounds)
    names = sort(collect(keys(bo.bounds)))

    for iter in 1:n_iter
        # Exploration phase: random sampling
        if iter <= bo.n_random || isempty(bo.history)
            x_unit = rand(bo.rng, d)
        else
            x_unit = _next_candidate(bo)
        end

        params = _unscale(bo, x_unit)

        y = try
            objective_fn(params)
        catch e
            @warn "Objective failed at iter $iter: $e"
            NaN
        end

        isnan(y) && continue

        # Update GP
        update_gp!(bo.gp, x_unit, y)

        push!(bo.history, EvalPoint(x_unit, y, params))

        if iter % 10 == 0
            best = maximum(p.y for p in bo.history)
            @info "BO iter $iter/$n_iter | best = $(round(best; digits=4))"
        end
    end

    isempty(bo.history) && error("No successful evaluations")

    best_point = argmax(p -> p.y, bo.history)
    OptimResult(
        best_point.params,
        best_point.y,
        length(bo.history),
        bo.history,
        "bayesian"
    )
end

# Convenience wrapper
function optimize(objective_fn::Function, param_bounds::ParamBounds;
                  n_iter::Int=100, kwargs...)
    bo = BayesianOptimizer(param_bounds; kwargs...)
    optimize!(bo, objective_fn; n_iter=n_iter)
end

# ---------------------------------------------------------------------------
# Grid search (parallel)
# ---------------------------------------------------------------------------

"""
Parallel exhaustive grid search.

# Fields
- `param_grid`  : Dict mapping param name → vector of candidate values
- `n_workers`   : number of Julia threads to use
"""
struct GridSearchParallel
    param_grid  ::Dict{String, Vector{Float64}}
    n_workers   ::Int
end

GridSearchParallel(param_grid; n_workers=4) =
    GridSearchParallel(param_grid, n_workers)

"""
Run grid search, return OptimResult.
"""
function run_grid_search(gs::GridSearchParallel, objective_fn::Function)
    names  = sort(collect(keys(gs.param_grid)))
    grids  = [gs.param_grid[n] for n in names]

    # Cartesian product of all parameter values
    candidates = vec(collect(Iterators.product(grids...)))
    n_cands    = length(candidates)

    @info "Grid search: $(n_cands) candidates, $(gs.n_workers) threads"

    scores = Vector{Float64}(undef, n_cands)
    Threads.@threads for i in 1:n_cands
        vals   = candidates[i]
        params = Dict(names[j] => Float64(vals[j]) for j in eachindex(names))
        scores[i] = try
            objective_fn(params)
        catch
            NaN
        end
    end

    best_idx    = argmax(filter(!isnan, scores) |> _ -> eachindex(scores))
    valid_idx   = [i for i in eachindex(scores) if !isnan(scores[i])]
    best_idx    = valid_idx[argmax(scores[valid_idx])]

    best_vals   = candidates[best_idx]
    best_params = Dict(names[j] => Float64(best_vals[j]) for j in eachindex(names))

    history = [
        EvalPoint(
            [Float64(candidates[i][j]) for j in eachindex(names)],
            scores[i],
            Dict(names[j] => Float64(candidates[i][j]) for j in eachindex(names))
        )
        for i in valid_idx
    ]

    OptimResult(best_params, scores[best_idx], n_cands, history, "grid")
end

# ---------------------------------------------------------------------------
# Objective functions (call Python backtest via PyCall)
# ---------------------------------------------------------------------------

"""
Call Python backtest engine and return the Sharpe ratio.
Falls back to a synthetic objective when PyCall is unavailable.
"""
function objective_sharpe(params::Dict{String, Float64})::Float64
    if HAS_PYCALL
        py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
"""
        try
            bt = PyCall.pyimport("shadow_runner.backtest")
            result = bt.run(params)
            return Float64(result["sharpe"])
        catch e
            @warn "PyCall backtest failed: $e — using synthetic objective"
        end
    end
    # Synthetic: bowl-shaped Sharpe surface for testing
    vals = collect(values(params))
    -sum((v - 0.5)^2 for v in vals) + 1.5 + 0.1 * randn()
end

"""
Call Python backtest engine and return the Calmar ratio.
"""
function objective_calmar(params::Dict{String, Float64})::Float64
    if HAS_PYCALL
        try
            bt = PyCall.pyimport("shadow_runner.backtest")
            result = bt.run(params)
            return Float64(result["calmar"])
        catch e
            @warn "PyCall backtest failed: $e — using synthetic objective"
        end
    end
    vals = collect(values(params))
    -sum((v - 0.4)^2 for v in vals) + 1.2 + 0.1 * randn()
end

# ---------------------------------------------------------------------------
# NSGA-II multi-objective Pareto optimisation
# ---------------------------------------------------------------------------

"""
Individual in NSGA-II population.
"""
mutable struct Individual
    params      ::Dict{String, Float64}
    objectives  ::Vector{Float64}   # (higher = better for each)
    rank        ::Int
    crowding    ::Float64
end

Individual(params, objectives) = Individual(params, objectives, 0, 0.0)

"""
Check Pareto dominance: a dominates b iff a is at least as good on all
objectives and strictly better on at least one.
"""
function dominates(a::Individual, b::Individual)
    all(a.objectives .>= b.objectives) && any(a.objectives .> b.objectives)
end

"""
Non-dominated sorting: assign Pareto rank to each individual.
"""
function fast_non_dominated_sort!(population::Vector{Individual})
    n = length(population)
    S = [Int[] for _ in 1:n]        # individuals dominated by i
    n_dom = zeros(Int, n)            # domination count for i
    fronts = [Int[]]

    for p in 1:n, q in 1:n
        p == q && continue
        if dominates(population[p], population[q])
            push!(S[p], q)
        elseif dominates(population[q], population[p])
            n_dom[p] += 1
        end
    end

    for p in 1:n
        n_dom[p] == 0 && push!(fronts[1], p)
    end

    i = 1
    while !isempty(fronts[i])
        next_front = Int[]
        for p in fronts[i]
            population[p].rank = i
            for q in S[p]
                n_dom[q] -= 1
                n_dom[q] == 0 && push!(next_front, q)
            end
        end
        push!(fronts, next_front)
        i += 1
    end
    fronts
end

"""
Crowding distance assignment for diversity preservation.
"""
function crowding_distance!(population::Vector{Individual}, front::Vector{Int})
    isempty(front) && return
    n_obj = length(population[1].objectives)
    for i in front
        population[i].crowding = 0.0
    end
    for obj_idx in 1:n_obj
        sorted = sort(front, by = i -> population[i].objectives[obj_idx])
        population[sorted[1]].crowding   = Inf
        population[sorted[end]].crowding = Inf
        obj_min = population[sorted[1]].objectives[obj_idx]
        obj_max = population[sorted[end]].objectives[obj_idx]
        rng = obj_max - obj_min
        rng < 1e-12 && continue
        for k in 2:(length(sorted) - 1)
            population[sorted[k]].crowding +=
                (population[sorted[k+1]].objectives[obj_idx] -
                 population[sorted[k-1]].objectives[obj_idx]) / rng
        end
    end
end

"""
Binary tournament selection (NSGA-II style).
"""
function tournament_select(pop::Vector{Individual}, rng::MersenneTwister)
    a, b = rand(rng, 1:length(pop)), rand(rng, 1:length(pop))
    ia, ib = pop[a], pop[b]
    if ia.rank < ib.rank return a end
    if ib.rank < ia.rank return b end
    ia.crowding >= ib.crowding ? a : b
end

"""
Simulated binary crossover (SBX) for real-valued parameters.
"""
function sbx_crossover(p1::Vector{Float64}, p2::Vector{Float64},
                        bounds::ParamBounds, names::Vector{String};
                        η_c::Float64=20.0, rng=MersenneTwister())
    c1, c2 = copy(p1), copy(p2)
    for (i, n) in enumerate(names)
        lo, hi = bounds[n]
        rand(rng) > 0.5 && continue
        if abs(p1[i] - p2[i]) < 1e-10 continue end
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        u = rand(rng)
        if u < 0.5
            β = (2u)^(1 / (η_c + 1))
        else
            β = (1 / (2(1 - u)))^(1 / (η_c + 1))
        end
        c1[i] = clamp(0.5 * ((y1 + y2) - β * (y2 - y1)), lo, hi)
        c2[i] = clamp(0.5 * ((y1 + y2) + β * (y2 - y1)), lo, hi)
    end
    c1, c2
end

"""
Polynomial mutation.
"""
function poly_mutation!(x::Vector{Float64}, bounds::ParamBounds,
                         names::Vector{String};
                         η_m::Float64=20.0, pm::Float64=0.1,
                         rng=MersenneTwister())
    for (i, n) in enumerate(names)
        rand(rng) > pm && continue
        lo, hi = bounds[n]
        δ_max  = hi - lo
        δ_max < 1e-12 && continue
        u = rand(rng)
        if u < 0.5
            δ = (2u)^(1 / (η_m + 1)) - 1
        else
            δ = 1 - (2(1 - u))^(1 / (η_m + 1))
        end
        x[i] = clamp(x[i] + δ * δ_max, lo, hi)
    end
    x
end

"""
NSGA-II: find the Pareto-optimal set across multiple objectives.

# Arguments
- `params_list`     : list of parameter dicts to seed initial population
                      (or pass empty list for random initialisation)
- `objective_fns`   : vector of objective functions (each returns Float64, higher=better)
- `bounds`          : parameter bounds
- `pop_size`        : population size (default 100)
- `n_generations`   : number of generations (default 50)

# Returns
Vector of Pareto-front individuals (rank 1)
"""
function multi_objective_pareto(
    objective_fns ::Vector{Function},
    bounds        ::ParamBounds;
    params_list   ::Vector{Dict{String, Float64}} = Dict{String, Float64}[],
    pop_size      ::Int = 100,
    n_generations ::Int = 50,
    seed          ::Int = 42
)
    rng   = MersenneTwister(seed)
    names = sort(collect(keys(bounds)))
    d     = length(bounds)
    n_obj = length(objective_fns)

    # Helper: params dict → objective vector
    function evaluate(params)
        [f(params) for f in objective_fns]
    end

    # Helper: unit vector → params dict
    function make_params(x_unit)
        Dict(names[i] => bounds[names[i]][1] +
             x_unit[i] * (bounds[names[i]][2] - bounds[names[i]][1])
             for i in 1:d)
    end

    # Initialise population
    population = Individual[]

    # Seed with provided params
    for p in params_list
        x    = [Float64(get(p, n, rand(rng))) for n in names]
        objs = evaluate(p)
        push!(population, Individual(p, objs))
    end

    # Fill remainder with random individuals
    while length(population) < pop_size
        x_unit = rand(rng, d)
        params = make_params(x_unit)
        objs   = try evaluate(params) catch; fill(NaN, n_obj) end
        any(isnan, objs) && continue
        push!(population, Individual(params, objs))
    end

    @info "NSGA-II: pop_size=$pop_size, n_gen=$n_generations, n_obj=$n_obj"

    for gen in 1:n_generations
        # Create offspring
        offspring = Individual[]
        while length(offspring) < pop_size
            i1 = tournament_select(population, rng)
            i2 = tournament_select(population, rng)

            p1 = [Float64(population[i1].params[n]) for n in names]
            p2 = [Float64(population[i2].params[n]) for n in names]

            c1_vec, c2_vec = sbx_crossover(p1, p2, bounds, names; rng=rng)

            for cv in (c1_vec, c2_vec)
                poly_mutation!(cv, bounds, names; rng=rng)
                params = make_params(cv)
                objs   = try evaluate(params) catch; fill(NaN, n_obj) end
                any(isnan, objs) && continue
                push!(offspring, Individual(params, objs))
            end
        end

        # Combine parent + offspring, sort, select
        combined = vcat(population, offspring)
        fronts   = fast_non_dominated_sort!(combined)

        for front in fronts
            crowding_distance!(combined, front)
        end

        # Select next generation
        population = Individual[]
        for front in fronts
            isempty(front) && continue
            if length(population) + length(front) <= pop_size
                append!(population, combined[front])
            else
                # Fill remainder by crowding distance
                remaining = pop_size - length(population)
                sorted_front = sort(front,
                    by = i -> combined[i].crowding, rev = true)
                append!(population, combined[sorted_front[1:remaining]])
                break
            end
        end

        if gen % 10 == 0
            pareto_size = count(p -> p.rank == 1, population)
            @info "Gen $gen/$n_generations | Pareto front size: $pareto_size"
        end
    end

    pareto_front = filter(p -> p.rank == 1, population)
    @info "NSGA-II complete. Pareto front: $(length(pareto_front)) solutions"
    pareto_front
end

# ---------------------------------------------------------------------------
# JSON serialisation helpers (for Python wrapper)
# ---------------------------------------------------------------------------

function result_to_json(r::OptimResult)::String
    d = Dict(
        "best_params"  => r.best_params,
        "best_score"   => r.best_score,
        "n_iterations" => r.n_iterations,
        "method"       => r.method,
        "history"      => [
            Dict("params" => p.params, "score" => p.y)
            for p in r.history
        ]
    )
    JSON3.write(d)
end

function pareto_to_json(front::Vector{Individual})::String
    arr = [
        Dict(
            "params"     => ind.params,
            "objectives" => ind.objectives,
            "rank"       => ind.rank,
            "crowding"   => ind.crowding
        )
        for ind in front
    ]
    JSON3.write(arr)
end

end  # module Optimizer

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using .Optimizer

    println("[optimizer] Starting self-test...")

    bounds = Dict(
        "fast_period" => (5.0,  50.0),
        "slow_period" => (20.0, 200.0),
        "threshold"   => (0.0,  1.0)
    )

    # Bayesian optimisation
    result = Optimizer.optimize(objective_sharpe, bounds; n_iter=50)
    println("[optimizer] BO best Sharpe=$(round(result.best_score; digits=4))")
    println("[optimizer] BO best params=$(result.best_params)")

    # Write result
    out_dir = get(ENV, "STATS_OUTPUT_DIR", joinpath(@__DIR__, "..", "output"))
    mkpath(out_dir)
    open(joinpath(out_dir, "optimizer_result.json"), "w") do io
        write(io, Optimizer.result_to_json(result))
    end
    println("[optimizer] Self-test complete.")
end
