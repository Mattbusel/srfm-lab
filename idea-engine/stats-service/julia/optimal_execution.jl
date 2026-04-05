"""
optimal_execution.jl

Almgren-Chriss optimal execution framework.
  - Model: minimize E[cost] + λ·Var[cost] over a trading trajectory
  - Closed-form solution for linear (permanent + temporary) market impact
  - Efficient frontier: trade-off curves between expected cost and variance
  - TWAP schedule generation for different urgency levels
  - Numerical solution via gradient descent when closed-form unavailable
  - Utility: for a given risk aversion λ, choose optimal schedule
"""

using Statistics
using LinearAlgebra
using JSON3
using Dates

# ─── Model Parameters ─────────────────────────────────────────────────────────

"""
Almgren-Chriss market impact parameters for a single instrument.

Fields:
  X       : total shares to liquidate (positive = sell)
  T       : total time horizon (e.g., 1.0 = 1 day)
  N       : number of trading periods
  σ       : daily volatility of the asset
  η       : temporary market impact coefficient (linear)
             cost_temp = η · v  where v = trading rate (shares/period)
  γ       : permanent market impact coefficient (linear)
             cost_perm = γ · x  where x = signed volume
  ε       : bid-ask spread (half-spread as fraction of price)
"""
struct ACParams
    X::Float64     # total position to liquidate (shares)
    T::Float64     # total time (days)
    N::Int         # number of time steps
    σ::Float64     # volatility per unit time
    η::Float64     # temporary impact coefficient
    γ::Float64     # permanent impact coefficient
    ε::Float64     # half bid-ask spread coefficient
end

"""Trading trajectory: sequence of shares held at each time step."""
struct Trajectory
    times::Vector{Float64}       # time points t_0..t_N
    holdings::Vector{Float64}    # shares held x_0..x_N  (x_0 = X, x_N = 0)
    trades::Vector{Float64}      # shares sold per period n_k = x_{k-1} - x_k
    expected_cost::Float64
    variance::Float64
    risk_adjusted_cost::Float64  # E[cost] + λ·Var[cost]
    lambda::Float64              # risk aversion used
end

# ─── Almgren-Chriss Analytical Solution ──────────────────────────────────────

"""
    ac_kappa(params)

Compute the characteristic decay rate κ for the Almgren-Chriss solution.

κ satisfies:  cosh(κ·τ) = 1 + (λ·σ²·τ²) / (2·η)  approximately,
but the exact expression (from the paper) is:

    κ = (1/τ) · arccosh(1 + τ²·λ·σ² / (2·η))

where τ = T/N (period length).
"""
function ac_kappa(params::ACParams, lambda::Float64)::Float64
    τ = params.T / params.N
    # Avoid numerical issues when λ·σ² is very small
    arg = 1.0 + τ^2 * lambda * params.σ^2 / (2.0 * max(params.η, 1e-12))
    arg <= 1.0 && return 0.0
    return acosh(arg) / τ
end

"""
    ac_closed_form(params, lambda)

Almgren-Chriss closed-form optimal execution trajectory.

The optimal trading schedule is:
    x_j = X · sinh(κ·(T - t_j)) / sinh(κ·T)

where κ is the characteristic rate depending on λ (risk aversion).

Special case κ→0 (risk-neutral): x_j = X · (T - t_j)/T  (TWAP)

Returns a Trajectory struct with the optimal schedule.
"""
function ac_closed_form(params::ACParams, lambda::Float64)::Trajectory
    N = params.N
    τ = params.T / N
    times = [j * τ for j in 0:N]

    κ = ac_kappa(params, lambda)

    holdings = Float64[]
    if κ < 1e-8
        # Risk-neutral limit: TWAP
        for j in 0:N
            push!(holdings, params.X * (1.0 - j / N))
        end
    else
        sinh_kT = sinh(κ * params.T)
        sinh_kT < 1e-15 && (sinh_kT = 1e-15)
        for j in 0:N
            t_j = j * τ
            push!(holdings, params.X * sinh(κ * (params.T - t_j)) / sinh_kT)
        end
    end

    # Enforce boundary conditions
    holdings[1] = params.X
    holdings[end] = 0.0

    trades = [holdings[j] - holdings[j+1] for j in 1:N]

    E_cost, var_cost = compute_cost(params, trades, τ)
    rac = E_cost + lambda * var_cost

    return Trajectory(times, holdings, trades, E_cost, var_cost, rac, lambda)
end

# ─── Cost Computation ─────────────────────────────────────────────────────────

"""
    compute_cost(params, trades, tau)

Compute expected cost and variance for a given trading schedule.

Almgren-Chriss cost model:
  Temporary impact: η · (n_k/τ)² · τ  for each period k
  Permanent impact: γ · n_k · x̄_k     where x̄_k is average holding
  Spread cost:      ε · |n_k|

Variance (simplified linear model):
  Var[cost] = σ² · τ · sum( x̄_k² )
  where x̄_k = (x_{k-1} + x_k)/2

Returns (E_cost, Var_cost).
"""
function compute_cost(params::ACParams, trades::Vector{Float64}, τ::Float64)::Tuple{Float64, Float64}
    N = length(trades)
    x = zeros(N+1)
    x[1] = params.X
    for k in 1:N
        x[k+1] = x[k] - trades[k]
    end

    E_cost = 0.0
    var_cost = 0.0

    for k in 1:N
        v_k = trades[k] / τ           # trading rate
        x_bar = (x[k] + x[k+1]) / 2  # average holding

        # Temporary impact cost
        temp_cost = params.η * v_k^2 * τ
        # Permanent impact cost
        perm_cost = params.γ * trades[k] * x_bar
        # Spread cost
        spread_cost = params.ε * abs(trades[k])

        E_cost += temp_cost + perm_cost + spread_cost

        # Variance contribution (holdings × volatility)
        var_cost += x_bar^2
    end

    var_cost *= params.σ^2 * τ
    return (E_cost, var_cost)
end

# ─── Efficient Frontier ───────────────────────────────────────────────────────

"""
    efficient_frontier(params; n_lambdas=50, lambda_min=1e-6, lambda_max=1e-2)

Compute the Almgren-Chriss efficient frontier: a set of optimal trajectories
for different risk aversion levels λ.

Returns a vector of (lambda, E_cost, Var_cost, trajectory) tuples.
The frontier shows the trade-off between expected cost and variance.
"""
function efficient_frontier(
    params::ACParams;
    n_lambdas::Int = 50,
    lambda_min::Float64 = 1e-6,
    lambda_max::Float64 = 1e-2
)::Vector{NamedTuple}
    # Log-spaced lambda grid
    lambdas = exp.(range(log(lambda_min), log(lambda_max), length=n_lambdas))
    results = NamedTuple[]

    for λ in lambdas
        traj = ac_closed_form(params, λ)
        push!(results, (
            lambda          = λ,
            expected_cost   = traj.expected_cost,
            variance        = traj.variance,
            std_cost        = sqrt(max(traj.variance, 0.0)),
            rac             = traj.risk_adjusted_cost,
            trajectory      = traj,
        ))
    end

    return results
end

# ─── TWAP Schedules ───────────────────────────────────────────────────────────

"""
    twap_schedule(params)

Generate a standard TWAP (Time-Weighted Average Price) schedule.
Trades uniformly: n_k = X/N for all k.
This is the risk-neutral (λ=0) Almgren-Chriss solution.
"""
function twap_schedule(params::ACParams)::Trajectory
    return ac_closed_form(params, 0.0)
end

"""
    urgency_schedule(params, urgency)

Generate trading schedules for different urgency levels.
Urgency ∈ [0, 1]:
  - 0.0: VWAP-like (spread over full horizon, TWAP)
  - 0.5: Balanced (moderate front-loading)
  - 1.0: Immediate (trade as fast as possible)

Higher urgency → more front-loaded → higher expected cost but lower variance.
"""
function urgency_schedule(params::ACParams, urgency::Float64)::Trajectory
    urgency = clamp(urgency, 0.0, 1.0)

    # Map urgency to lambda: higher urgency → higher risk aversion → more aggressive
    lambda_min = 1e-7
    lambda_max = 5e-3
    lambda = lambda_min * (lambda_max / lambda_min)^urgency

    return ac_closed_form(params, lambda)
end

"""
    urgency_schedules(params; levels=[0.0, 0.25, 0.5, 0.75, 1.0])

Compute optimal schedules for a set of urgency levels.
Returns a Dict mapping urgency label to trajectory.
"""
function urgency_schedules(
    params::ACParams;
    levels::Vector{Float64} = [0.0, 0.25, 0.5, 0.75, 1.0]
)::Dict{String, Trajectory}
    labels = ["PASSIVE", "BALANCED_LOW", "BALANCED", "AGGRESSIVE", "IMMEDIATE"]
    n = min(length(levels), length(labels))
    out = Dict{String, Trajectory}()
    for i in 1:n
        out[labels[i]] = urgency_schedule(params, levels[i])
    end
    return out
end

# ─── Numerical Solution (Gradient Descent) ────────────────────────────────────

"""
    ac_numerical(params, lambda; lr=1e-4, max_iter=5000, tol=1e-8)

Numerical optimization of the Almgren-Chriss objective via gradient descent
on the constrained problem (sum of trades = X, trades ≥ 0).

The objective is:
    min_{n_1,...,n_N} E[cost] + λ·Var[cost]
    s.t. sum(n_k) = X,  n_k ≥ 0

Uses projected gradient descent with simplex projection.
"""
function ac_numerical(
    params::ACParams,
    lambda::Float64;
    lr::Float64 = 1e-4,
    max_iter::Int = 5000,
    tol::Float64 = 1e-8
)::Trajectory
    N = params.N
    τ = params.T / N

    # Initialize with TWAP
    trades = fill(params.X / N, N)
    objective_history = Float64[]

    function objective_and_grad(trades_::Vector{Float64})
        E, V = compute_cost(params, trades_, τ)
        obj = E + lambda * V
        push!(objective_history, obj)

        # Numerical gradient
        eps = 1e-7
        grad = zeros(N)
        for k in 1:N
            trades_plus = copy(trades_)
            trades_plus[k] += eps
            E_p, V_p = compute_cost(params, trades_plus, τ)
            grad[k] = ((E_p + lambda * V_p) - obj) / eps
        end
        return (obj, grad)
    end

    prev_obj = Inf
    for iter in 1:max_iter
        obj, grad = objective_and_grad(trades)

        # Gradient step
        trades_new = trades .- lr .* grad

        # Project onto constraint: sum = X, all ≥ 0
        # Simple projection: clip negatives, then rescale to sum = X
        trades_new = max.(trades_new, 0.0)
        s = sum(trades_new)
        if s > 1e-10
            trades_new .*= params.X / s
        else
            trades_new = fill(params.X / N, N)
        end

        trades = trades_new

        # Convergence
        if abs(prev_obj - obj) < tol
            break
        end
        prev_obj = obj
    end

    E_cost, var_cost = compute_cost(params, trades, τ)

    # Reconstruct holdings
    holdings = zeros(N+1)
    holdings[1] = params.X
    for k in 1:N
        holdings[k+1] = holdings[k] - trades[k]
    end
    holdings[end] = 0.0

    times = [j * τ for j in 0:N]
    rac = E_cost + lambda * var_cost

    return Trajectory(times, holdings, trades, E_cost, var_cost, rac, lambda)
end

# ─── Trajectory Comparison ────────────────────────────────────────────────────

"""
    compare_trajectories(trajectories, names)

Compare a set of trajectories by their cost/variance characteristics.
Returns a summary Dict.
"""
function compare_trajectories(
    trajectories::Vector{Trajectory},
    names::Vector{String}
)::Vector{Dict}
    [
        Dict(
            "name"          => names[i],
            "lambda"        => trajectories[i].lambda,
            "expected_cost" => trajectories[i].expected_cost,
            "std_cost"      => sqrt(max(trajectories[i].variance, 0.0)),
            "variance"      => trajectories[i].variance,
            "rac"           => trajectories[i].risk_adjusted_cost,
            "front_loading" => trajectories[i].trades[1] / max(trajectories[i].holdings[1], 1e-10),
        )
        for i in eachindex(trajectories)
    ]
end

# ─── Market Impact Estimation ─────────────────────────────────────────────────

"""
    estimate_impact_params(trades, price_changes; min_obs=20)

Estimate the temporary and permanent market impact coefficients from
historical execution data using OLS regression.

Δp_t = γ · n_t + ε_t          (permanent impact)
cost_t = η · (n_t/τ)² · τ     (temporary impact, fitted by spread)
"""
function estimate_impact_params(
    trades::Vector{Float64},      # historical trade sizes (signed)
    price_changes::Vector{Float64},  # corresponding price changes
    τ::Float64 = 1.0             # time step
)::NamedTuple
    n = length(trades)
    n == length(price_changes) || throw(ArgumentError("lengths must match"))
    n < 20 && return (gamma=NaN, eta=NaN, r_squared=NaN)

    # Estimate permanent impact via OLS
    x_mean = mean(trades)
    y_mean = mean(price_changes)
    ss_xx = sum((t - x_mean)^2 for t in trades) + 1e-10
    cov_xy = sum((trades[i] - x_mean) * (price_changes[i] - y_mean) for i in 1:n)
    γ = cov_xy / ss_xx

    # Residuals → estimate temporary impact from execution cost
    y_hat = γ .* trades .+ y_mean
    residuals = price_changes .- y_hat
    ss_res = sum(r^2 for r in residuals)
    ss_tot = sum((y - y_mean)^2 for y in price_changes) + 1e-10
    r2 = 1.0 - ss_res / ss_tot

    # Temporary impact coefficient from residual variance and trade rate
    trade_rates = trades ./ τ
    rate_mean = mean(trade_rates)
    if std(trade_rates) < 1e-10
        η = NaN
    else
        ss_rate = sum((r - rate_mean)^2 for r in trade_rates) + 1e-10
        cov_rate_res = sum((trade_rates[i] - rate_mean) * residuals[i] for i in 1:n)
        η = abs(cov_rate_res / ss_rate)
    end

    return (gamma=γ, eta=η, r_squared=r2)
end

# ─── Serialization ────────────────────────────────────────────────────────────

"""
    trajectory_to_dict(traj)

Convert a Trajectory to a Dict for JSON export.
"""
function trajectory_to_dict(traj::Trajectory)::Dict
    Dict(
        "lambda"          => traj.lambda,
        "expected_cost"   => traj.expected_cost,
        "variance"        => traj.variance,
        "std_cost"        => sqrt(max(traj.variance, 0.0)),
        "risk_adj_cost"   => traj.risk_adjusted_cost,
        "times"           => traj.times,
        "holdings"        => traj.holdings,
        "trades"          => traj.trades,
        "front_load_pct"  => (traj.trades[1] / max(traj.holdings[1], 1e-10)) * 100,
    )
end

"""
    export_execution_results(params, frontier, urgency_scheds, filepath)

Write all execution results to a JSON file.
"""
function export_execution_results(
    params::ACParams,
    frontier::Vector{NamedTuple},
    urgency_scheds::Dict{String, Trajectory},
    filepath::String
)
    frontier_data = [
        Dict(
            "lambda"        => f.lambda,
            "expected_cost" => f.expected_cost,
            "std_cost"      => f.std_cost,
            "variance"      => f.variance,
        )
        for f in frontier
    ]

    urgency_data = Dict(k => trajectory_to_dict(v) for (k, v) in urgency_scheds)

    output = Dict(
        "params" => Dict(
            "X"  => params.X,
            "T"  => params.T,
            "N"  => params.N,
            "sigma" => params.σ,
            "eta"   => params.η,
            "gamma" => params.γ,
            "epsilon" => params.ε,
        ),
        "efficient_frontier" => frontier_data,
        "urgency_schedules"  => urgency_data,
    )

    open(filepath, "w") do io
        JSON3.write(io, output)
    end
    @info "Execution results written to $filepath"
end

# ─── Demo / Entry Point ───────────────────────────────────────────────────────

function run_optimal_execution_demo()
    @info "Running Almgren-Chriss optimal execution demo..."

    # Example: liquidate 100 BTC over 1 day in 48 half-hour periods
    params = ACParams(
        100.0,    # X: 100 BTC to liquidate
        1.0,      # T: 1 day
        48,       # N: 48 periods (30-min bars)
        0.025,    # σ: 2.5% daily vol
        0.01,     # η: temporary impact (0.01 per unit rate)
        0.005,    # γ: permanent impact (0.005 per unit volume)
        0.0002,   # ε: half spread (2 bps)
    )

    @info "Parameters: X=$(params.X) BTC, T=$(params.T)d, N=$(params.N) periods"

    # TWAP baseline
    twap = twap_schedule(params)
    @info "TWAP: E[cost]=$(round(twap.expected_cost, sigdigits=4)), σ[cost]=$(round(sqrt(twap.variance), sigdigits=4))"

    # Efficient frontier
    @info "Computing efficient frontier (50 λ values)..."
    frontier = efficient_frontier(params; n_lambdas=50)
    min_var = frontier[1]
    max_util = frontier[end]
    @info "  Min-variance end: E=$(round(min_var.expected_cost, sigdigits=4)), σ=$(round(min_var.std_cost, sigdigits=4))"
    @info "  Max-utility end:  E=$(round(max_util.expected_cost, sigdigits=4)), σ=$(round(max_util.std_cost, sigdigits=4))"

    # Urgency schedules
    @info "Computing urgency schedules..."
    urgency = urgency_schedules(params)
    for (name, traj) in sort(urgency)
        fl = traj.trades[1] / params.X * 100
        @info "  $name: front-load=$(round(fl, digits=1))%, E=$(round(traj.expected_cost, sigdigits=4))"
    end

    # Numerical solution (verification)
    @info "Numerical solution (gradient descent, λ=0.001)..."
    num_traj = ac_numerical(params, 0.001; max_iter=2000)
    closed_traj = ac_closed_form(params, 0.001)
    @info "  Closed-form: E=$(round(closed_traj.expected_cost, sigdigits=4))"
    @info "  Numerical:   E=$(round(num_traj.expected_cost, sigdigits=4))"
    @info "  Difference:  $(round(abs(num_traj.expected_cost - closed_traj.expected_cost) / max(closed_traj.expected_cost, 1e-10) * 100, digits=2))%"

    # Export
    outfile = joinpath(@__DIR__, "optimal_execution_results.json")
    export_execution_results(params, frontier, urgency, outfile)

    return (params, frontier, urgency)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_optimal_execution_demo()
end
