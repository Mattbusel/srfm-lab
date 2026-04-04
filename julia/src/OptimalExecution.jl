"""
OptimalExecution — Optimal trade execution algorithms for the SRFM research suite.

Implements:
  - Almgren-Chriss model (permanent/temporary impact, efficient frontier, closed-form trajectory)
  - Obizhaeva-Wang model (limit order book dynamics)
  - IS (Implementation Shortfall) minimization
  - VWAP schedule construction
  - Transaction cost analysis (pre-trade vs post-trade attribution)
  - Multi-asset execution with correlation-adjusted impact
"""
module OptimalExecution

using LinearAlgebra
using Statistics
using Distributions
using Optim
using DataFrames

export AlmgrenChrissParams, ACTrajectory, ac_optimal_trajectory, ac_efficient_frontier
export ac_expected_cost, ac_variance_cost, ac_closed_form
export OWParams, ow_optimal_trajectory, ow_spread_cost
export ISMinimizer, minimize_is, vwap_schedule, twap_schedule
export TCAResult, pre_trade_estimate, post_trade_attribution
export MultiAssetExecution, multi_asset_optimal

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Almgren-Chriss Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    AlmgrenChrissParams

Parameters for the Almgren-Chriss optimal execution model.

# Fields
- `X::Float64`: Total shares to liquidate (positive = sell)
- `T::Float64`: Total time horizon (in days or other unit)
- `N::Int`: Number of time intervals
- `sigma::Float64`: Daily volatility of asset price
- `eta::Float64`: Temporary impact coefficient (price per share^2 per unit time)
- `gamma::Float64`: Permanent impact coefficient (price per share)
- `epsilon::Float64`: Fixed half-spread cost per share
- `tau::Float64`: Time step (T/N)
- `lambda::Float64`: Risk-aversion parameter
"""
struct AlmgrenChrissParams
    X::Float64        # total shares to liquidate
    T::Float64        # horizon
    N::Int            # number of steps
    sigma::Float64    # volatility per unit time
    eta::Float64      # temporary impact
    gamma::Float64    # permanent impact
    epsilon::Float64  # fixed half-spread
    tau::Float64      # time step
    lambda::Float64   # risk aversion

    function AlmgrenChrissParams(X, T, N, sigma, eta, gamma, epsilon, lambda)
        tau = T / N
        new(X, T, N, sigma, eta, gamma, epsilon, tau, lambda)
    end
end

"""
    ACTrajectory

Holds the result of Almgren-Chriss optimal execution trajectory.

# Fields
- `times::Vector{Float64}`: Time grid
- `holdings::Vector{Float64}`: Share holdings at each time
- `trades::Vector{Float64}`: Shares traded in each interval
- `expected_cost::Float64`: Expected implementation shortfall
- `variance_cost::Float64`: Variance of implementation shortfall
- `utility::Float64`: E[cost] + lambda * Var[cost]
"""
struct ACTrajectory
    times::Vector{Float64}
    holdings::Vector{Float64}
    trades::Vector{Float64}
    expected_cost::Float64
    variance_cost::Float64
    utility::Float64
end

"""
    ac_kappa(p::AlmgrenChrissParams)

Compute the key parameter kappa from AC model parameters.
kappa satisfies: cosh(kappa*tau)/sinh(kappa*tau) = (1 + lambda*sigma^2*tau^2/(2*eta_tilde))
where eta_tilde = eta - gamma*tau/2
"""
function ac_kappa(p::AlmgrenChrissParams)
    eta_tilde = p.eta - 0.5 * p.gamma * p.tau
    if eta_tilde <= 0
        # degenerate: treat as zero permanent impact
        eta_tilde = p.eta
    end
    kappa_sq = p.lambda * p.sigma^2 / eta_tilde
    return sqrt(kappa_sq)
end

"""
    ac_optimal_trajectory(p::AlmgrenChrissParams) -> ACTrajectory

Compute the closed-form Almgren-Chriss optimal execution trajectory.

The optimal strategy minimizes E[IS] + lambda * Var[IS].
Closed form: x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
"""
function ac_optimal_trajectory(p::AlmgrenChrissParams)
    kappa = ac_kappa(p)
    N = p.N
    tau = p.tau
    T = p.T
    X = p.X

    # Time grid at trading decision points: t_0=0, t_1=tau, ..., t_N=T
    times = [i * tau for i in 0:N]

    # Holdings at each time point (continuous formula)
    holdings = zeros(N + 1)
    for i in 0:N
        t = i * tau
        if kappa * T < 1e-10
            # Near-zero kappa: linear liquidation
            holdings[i+1] = X * (1.0 - t / T)
        else
            holdings[i+1] = X * sinh(kappa * (T - t)) / sinh(kappa * T)
        end
    end
    # Enforce boundary conditions
    holdings[1] = X
    holdings[end] = 0.0

    # Trades in each interval
    trades = diff(holdings)  # negative = selling

    # Compute expected cost
    E_cost = ac_expected_cost(p, trades)
    V_cost = ac_variance_cost(p, holdings)
    utility = E_cost + p.lambda * V_cost

    return ACTrajectory(times, holdings, trades, E_cost, V_cost, utility)
end

"""
    ac_expected_cost(p::AlmgrenChrissParams, trades::Vector{Float64}) -> Float64

Compute expected implementation shortfall for a given trade schedule.

E[IS] = epsilon * sum(|n_j|) + (gamma/2)*X^2 + eta * sum(n_j^2 / tau)
where n_j = shares traded in interval j (negative for sells)
"""
function ac_expected_cost(p::AlmgrenChrissParams, trades::Vector{Float64})
    N = length(trades)
    tau = p.tau

    # Fixed cost (half-spread)
    fixed_cost = p.epsilon * sum(abs.(trades))

    # Permanent impact: linear in total traded = gamma/2 * X^2 (independent of trajectory)
    permanent_cost = 0.5 * p.gamma * p.X^2

    # Temporary impact: sum eta * (n_j/tau)^2 * tau = eta/tau * sum(n_j^2)
    temp_cost = (p.eta / tau) * sum(trades .^ 2)

    return fixed_cost + permanent_cost + temp_cost
end

"""
    ac_variance_cost(p::AlmgrenChrissParams, holdings::Vector{Float64}) -> Float64

Compute variance of implementation shortfall for a given holdings trajectory.

Var[IS] = sigma^2 * tau * sum(x_j^2)
"""
function ac_variance_cost(p::AlmgrenChrissParams, holdings::Vector{Float64})
    # holdings[1] = X (start), holdings[end] = 0 (end)
    # sum over mid-period holdings
    x_mid = holdings[1:end-1]
    return p.sigma^2 * p.tau * sum(x_mid .^ 2)
end

"""
    ac_closed_form(p::AlmgrenChrissParams) -> NamedTuple

Return closed-form expressions for E[IS] and Var[IS] for the optimal trajectory.

From Almgren & Chriss (2000), equations (14)–(17).
"""
function ac_closed_form(p::AlmgrenChrissParams)
    kappa = ac_kappa(p)
    X = p.X
    T = p.T
    N = p.N
    tau = p.tau
    sigma = p.sigma
    eta = p.eta
    gamma = p.gamma
    epsilon = p.epsilon

    if kappa * T < 1e-10
        # Degenerate case: linear liquidation
        E_IS = epsilon * X + 0.5 * gamma * X^2 + eta * X^2 / T
        Var_IS = sigma^2 * T * X^2 / 3.0
        return (expected_cost=E_IS, variance_cost=Var_IS, kappa=kappa)
    end

    # Closed-form expected cost
    # From AC2000: E[IS] = 0.5*gamma*X^2 + epsilon*X
    #              + eta*X^2*kappa / (sinh(kappa*T)) *
    #                (cosh(kappa*(T - tau)) - 1) / (2*sinh(kappa*tau/2))^2 ...
    # Use discrete approximation via summation for accuracy
    traj = ac_optimal_trajectory(p)
    E_IS = traj.expected_cost
    Var_IS = traj.variance_cost

    # Analytical approximation (continuous limit)
    sinh_kT = sinh(kappa * T)
    E_IS_cont = epsilon * X + 0.5 * gamma * X^2 +
                eta * kappa * X^2 * (1 / tanh(kappa * tau) - 1) *
                cosh(kappa * (T - tau / 2)) / sinh(kappa * T)

    Var_IS_cont = sigma^2 * X^2 * (T * cosh(kappa * T) - sinh(kappa * T) / kappa) /
                  (2.0 * kappa * sinh(kappa * T)^2) * (kappa * T)

    return (expected_cost=E_IS, variance_cost=Var_IS,
            expected_cost_cont=E_IS_cont, variance_cost_cont=Var_IS_cont,
            kappa=kappa)
end

"""
    ac_efficient_frontier(p::AlmgrenChrissParams; n_points=100) -> DataFrame

Compute the Almgren-Chriss efficient frontier: set of (E[IS], Var[IS]) pairs
for the optimal trajectory across a range of risk-aversion parameters.

Returns a DataFrame with columns: lambda, expected_cost, variance_cost, utility, kappa.
"""
function ac_efficient_frontier(p::AlmgrenChrissParams; n_points::Int=100)
    lambdas = exp.(range(log(1e-8), log(1e2), length=n_points))

    results = DataFrame(
        lambda=Float64[],
        expected_cost=Float64[],
        variance_cost=Float64[],
        utility=Float64[],
        kappa=Float64[]
    )

    for lam in lambdas
        p_lam = AlmgrenChrissParams(p.X, p.T, p.N, p.sigma, p.eta, p.gamma, p.epsilon, lam)
        traj = ac_optimal_trajectory(p_lam)
        push!(results, (
            lambda=lam,
            expected_cost=traj.expected_cost,
            variance_cost=traj.variance_cost,
            utility=traj.utility,
            kappa=ac_kappa(p_lam)
        ))
    end

    sort!(results, :variance_cost)
    return results
end

"""
    ac_discrete_trajectory(p::AlmgrenChrissParams; lambda_override=nothing) -> ACTrajectory

Compute optimal discrete trajectory by solving the tridiagonal system.
This matches the exact discrete-time Bellman recursion from Almgren & Chriss (2000).
"""
function ac_discrete_trajectory(p::AlmgrenChrissParams; lambda_override=nothing)
    lam = isnothing(lambda_override) ? p.lambda : lambda_override
    N = p.N
    tau = p.tau
    sigma = p.sigma
    eta = p.eta
    gamma = p.gamma
    X = p.X
    epsilon = p.epsilon

    eta_tilde = eta - 0.5 * gamma * tau

    # Build tridiagonal system for optimal holdings
    # From first-order conditions: 2*eta_tilde*x_{j-1} - (4*eta_tilde + 2*lam*sigma^2*tau^2)*x_j
    #                               + 2*eta_tilde*x_{j+1} = 0
    # with x_0 = X, x_N = 0

    if N <= 1
        return ac_optimal_trajectory(p)
    end

    # Coefficient for diagonal
    diag_coef = 2.0 * (2.0 * eta_tilde + lam * sigma^2 * tau^2)
    off_coef = -2.0 * eta_tilde

    # Interior points x_1, ..., x_{N-1}
    n_interior = N - 1
    A = zeros(n_interior, n_interior)
    b = zeros(n_interior)

    for i in 1:n_interior
        A[i, i] = diag_coef
        if i > 1
            A[i, i-1] = off_coef
        end
        if i < n_interior
            A[i, i+1] = off_coef
        end
    end

    # Boundary: x_0 = X contributes to first row
    b[1] = 2.0 * eta_tilde * X

    # Solve
    x_interior = A \ b
    holdings = vcat([X], x_interior, [0.0])
    times = [i * tau for i in 0:N]
    trades = diff(holdings)

    E_cost = ac_expected_cost(p, trades)
    V_cost = ac_variance_cost(p, holdings)
    utility = E_cost + lam * V_cost

    return ACTrajectory(times, holdings, trades, E_cost, V_cost, utility)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Obizhaeva-Wang Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    OWParams

Parameters for the Obizhaeva-Wang limit order book model.

# Fields
- `X::Float64`: Total shares to execute
- `T::Float64`: Time horizon
- `rho::Float64`: Order book resilience (recovery rate)
- `q::Float64`: Order book depth at best quote (shares per unit price)
- `sigma::Float64`: Asset volatility
- `lambda::Float64`: Risk aversion
"""
struct OWParams
    X::Float64
    T::Float64
    rho::Float64     # resilience / recovery speed
    q::Float64       # LOB depth
    sigma::Float64
    lambda::Float64
end

"""
    ow_price_impact(params::OWParams, trade_size::Float64, t_since_last::Float64) -> Float64

Compute temporary price impact in OW model.
After a trade n, the price impact decays as: I(t) = (n/q) * exp(-rho*t)
"""
function ow_price_impact(params::OWParams, trade_size::Float64, t_since_last::Float64)
    return (trade_size / params.q) * exp(-params.rho * t_since_last)
end

"""
    ow_optimal_trajectory(params::OWParams, N::Int) -> NamedTuple

Compute the Obizhaeva-Wang optimal trajectory for block-shaped LOB.

The OW model has a different optimal structure from AC: it favors trading
in continuous bursts with discrete blocks at beginning/end.

For a block order book with depth q, the optimal strategy is:
  - Initial block: n_0 = C_0 * X (fraction at t=0)
  - Continuous trading at rate v(t) in (0,T)
  - Final block: n_T = C_T * X (fraction at t=T)
"""
function ow_optimal_trajectory(params::OWParams, N::Int)
    X = params.X
    T = params.T
    rho = params.rho
    q = params.q
    sigma = params.sigma
    lambda = params.lambda
    tau = T / N

    # Compute key parameter
    phi = lambda * sigma^2 * q  # phi = lambda * sigma^2 / (1/q)

    # For the continuous OW problem, optimal solution (no risk aversion limit):
    # v(t) = constant trading rate
    # With risk aversion, includes initial/final blocks

    # Numerically compute optimal strategy via discrete approximation
    # Minimize sum of (impact costs) + lambda * variance
    # Subject to sum of trades = X

    # Simplification: use equal time slices with front/back-loading
    # determined by resilience parameter

    # Compute optimal initial block fraction
    # From OW (2013), optimal initial/final blocks:
    alpha = rho / (rho + phi / q)

    # Clamp alpha to valid range
    alpha = clamp(alpha, 0.0, 0.5)

    n0 = alpha * X       # initial block trade
    nT = alpha * X       # terminal block trade
    n_cont = (1.0 - 2.0 * alpha) * X  # to be spread over (0,T)

    # Continuous portion: uniform (optimal for linear impact with resilience)
    v = n_cont / T  # trading rate
    trades_continuous = fill(v * tau, N - 1)

    # Full trade schedule
    trades = vcat([n0], trades_continuous, [nT])
    # Adjust last trade to ensure sum = X
    trades[end] = X - sum(trades[1:end-1])

    times = vcat([0.0], [i * tau for i in 1:N])
    holdings = X .- cumsum(trades)
    holdings = vcat([X], X .- cumsum(trades))

    # Compute spread/impact cost
    total_cost = ow_spread_cost(params, trades, tau)
    variance = sigma^2 * tau * sum(max.(holdings[1:end-1], 0.0) .^ 2)

    return (
        times=times,
        holdings=holdings,
        trades=trades,
        spread_cost=total_cost,
        variance=variance,
        utility=total_cost + lambda * variance,
        initial_block=n0,
        terminal_block=nT,
        continuous_rate=v
    )
end

"""
    ow_spread_cost(params::OWParams, trades::Vector{Float64}, tau::Float64) -> Float64

Compute total spread/impact cost for OW model given trade schedule.
"""
function ow_spread_cost(params::OWParams, trades::Vector{Float64}, tau::Float64)
    q = params.q
    rho = params.rho
    N = length(trades)
    total_cost = 0.0
    impact_state = 0.0  # current accumulated impact

    for j in 1:N
        n_j = trades[j]
        # Cost of this trade: (impact_state + n_j / q) * n_j / 2
        avg_impact = impact_state + n_j / (2.0 * q)
        total_cost += avg_impact * abs(n_j)
        # Update impact state after trade
        impact_state = (impact_state + n_j / q) * exp(-rho * tau)
    end

    return total_cost
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Implementation Shortfall Minimization and VWAP
# ─────────────────────────────────────────────────────────────────────────────

"""
    ISMinimizer

Configuration for IS minimization.
"""
struct ISMinimizer
    X::Float64          # total shares
    T::Float64          # horizon
    N::Int              # steps
    volume_profile::Vector{Float64}   # intraday volume fractions (length N)
    sigma::Float64      # volatility
    bid_ask_spread::Float64
    market_impact_alpha::Float64  # price impact = alpha * (n/ADV)^beta
    market_impact_beta::Float64
    ADV::Float64        # average daily volume
end

"""
    minimize_is(ism::ISMinimizer; lambda=0.5) -> NamedTuple

Minimize implementation shortfall using numerical optimization.
IS = sum_j [ bid_ask/2 * |n_j| + alpha * (|n_j|/ADV)^beta * S_j * n_j ] + market risk

Uses Optim.jl for constrained optimization.
"""
function minimize_is(ism::ISMinimizer; lambda::Float64=0.5)
    N = ism.N
    X = ism.X
    sigma = ism.sigma
    tau = ism.T / N

    # Objective: minimize E[IS] + lambda * Var[IS]
    function objective(n_vec)
        # IS cost components
        cost = 0.0
        holdings = X
        variance_sum = 0.0

        for j in 1:N
            n_j = n_vec[j]
            # Fixed spread cost
            cost += ism.bid_ask_spread * abs(n_j)
            # Market impact cost (power law)
            adv_frac = abs(n_j) / (ism.ADV * tau)
            impact = ism.market_impact_alpha * adv_frac^ism.market_impact_beta
            cost += impact * abs(n_j)
            # Position risk
            variance_sum += holdings^2 * tau
            holdings -= n_j
        end

        # Market risk contribution
        variance_sum *= sigma^2

        return cost + lambda * variance_sum
    end

    # Initial: VWAP schedule
    vol_frac = ism.volume_profile ./ sum(ism.volume_profile)
    n0 = vol_frac .* X

    result = optimize(objective, n0, LBFGS();
        autodiff=:forward,
        iterations=1000
    )

    n_opt = Optim.minimizer(result)
    # Normalize to ensure sum = X
    scale = X / sum(n_opt)
    n_opt .*= scale

    holdings = vcat([X], X .- cumsum(n_opt))

    return (
        trades=n_opt,
        holdings=holdings,
        expected_cost=objective(n_opt),
        converged=Optim.converged(result),
        iterations=Optim.iterations(result)
    )
end

"""
    vwap_schedule(volume_profile::Vector{Float64}, total_shares::Float64) -> Vector{Float64}

Compute VWAP-optimal execution schedule (proportional to volume profile).
"""
function vwap_schedule(volume_profile::Vector{Float64}, total_shares::Float64)
    total_vol = sum(volume_profile)
    @assert total_vol > 0 "Volume profile must be positive"
    return volume_profile ./ total_vol .* total_shares
end

"""
    twap_schedule(N::Int, total_shares::Float64) -> Vector{Float64}

Compute TWAP execution schedule (equal slices).
"""
function twap_schedule(N::Int, total_shares::Float64)
    return fill(total_shares / N, N)
end

"""
    participation_rate_schedule(volume_profile::Vector{Float64}, total_shares::Float64,
                                 participation_rate::Float64) -> Vector{Float64}

Compute execution schedule constrained to a maximum participation rate.
"""
function participation_rate_schedule(
    volume_profile::Vector{Float64},
    total_shares::Float64,
    participation_rate::Float64
)
    N = length(volume_profile)
    max_trades = participation_rate .* volume_profile
    remaining = total_shares
    trades = zeros(N)

    for i in 1:N
        trades[i] = min(max_trades[i], remaining)
        remaining -= trades[i]
        if remaining <= 0
            break
        end
    end

    # If not fully executed, scale up proportionally
    if remaining > 0
        @warn "Cannot fully execute within participation rate; executed $(total_shares - remaining) / $total_shares"
    end

    return trades
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Transaction Cost Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    TCAResult

Holds transaction cost analysis results.
"""
struct TCAResult
    # Pre-trade estimates
    pretrade_is_estimate::Float64
    pretrade_spread_cost::Float64
    pretrade_impact_cost::Float64
    pretrade_timing_risk::Float64

    # Post-trade actuals
    posttrade_is_realized::Float64
    posttrade_spread_cost::Float64
    posttrade_impact_cost::Float64

    # Attribution
    decision_price::Float64
    arrival_price::Float64
    avg_execution_price::Float64
    benchmark_price::Float64  # e.g. VWAP

    # Slippage decomposition
    spread_slippage::Float64
    impact_slippage::Float64
    timing_slippage::Float64
    alpha_decay::Float64

    # Metrics
    is_bps::Float64         # IS in basis points
    vwap_slippage_bps::Float64
end

"""
    pre_trade_estimate(params::AlmgrenChrissParams, S0::Float64) -> NamedTuple

Estimate pre-trade transaction costs using AC model.
Returns expected IS, spread cost, impact cost, and timing risk in basis points.
"""
function pre_trade_estimate(params::AlmgrenChrissParams, S0::Float64)
    traj = ac_optimal_trajectory(params)
    X = params.X

    # Expected IS in dollars
    E_IS = traj.expected_cost

    # Spread component
    spread_cost = params.epsilon * X

    # Market impact component
    impact_cost = E_IS - spread_cost - 0.5 * params.gamma * X^2

    # Timing risk (1-sigma range)
    timing_risk_1sigma = sqrt(traj.variance_cost)

    # Convert to bps
    notional = X * S0
    is_bps = (E_IS / notional) * 10000
    spread_bps = (spread_cost / notional) * 10000
    impact_bps = (impact_cost / notional) * 10000
    risk_bps = (timing_risk_1sigma / notional) * 10000

    return (
        expected_is=E_IS,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
        timing_risk_1sigma=timing_risk_1sigma,
        is_bps=is_bps,
        spread_bps=spread_bps,
        impact_bps=impact_bps,
        risk_bps=risk_bps,
        trajectory=traj
    )
end

"""
    post_trade_attribution(
        exec_prices::Vector{Float64},
        exec_shares::Vector{Float64},
        exec_times::Vector{Float64},
        decision_price::Float64,
        market_prices::Vector{Float64},
        volume_profile::Vector{Float64}
    ) -> TCAResult

Attribute post-trade implementation shortfall into components.

IS = Spread Slippage + Market Impact + Timing/Alpha Decay

# Arguments
- `exec_prices`: Vector of execution prices for each trade slice
- `exec_shares`: Vector of shares executed in each slice
- `exec_times`: Vector of timestamps for each slice
- `decision_price`: Price at time of decision (arrival price)
- `market_prices`: Market mid prices at each execution time
- `volume_profile`: Market volume in each interval
"""
function post_trade_attribution(
    exec_prices::Vector{Float64},
    exec_shares::Vector{Float64},
    exec_times::Vector{Float64},
    decision_price::Float64,
    market_prices::Vector{Float64},
    volume_profile::Vector{Float64}
)
    @assert length(exec_prices) == length(exec_shares) == length(exec_times)

    total_shares = sum(exec_shares)
    notional_decision = total_shares * decision_price

    # Average execution price (quantity-weighted)
    avg_exec_price = sum(exec_prices .* exec_shares) / total_shares

    # VWAP benchmark
    vwap_benchmark = sum(market_prices .* volume_profile) / sum(volume_profile)

    # Total IS in bps
    is_dollars = (avg_exec_price - decision_price) * total_shares
    is_bps = (is_dollars / notional_decision) * 10000

    # VWAP slippage
    vwap_slip_dollars = (avg_exec_price - vwap_benchmark) * total_shares
    vwap_slip_bps = (vwap_slip_dollars / notional_decision) * 10000

    # Spread slippage: half the bid-ask spread
    # Estimated from price impact at each trade
    spread_slippage = 0.0
    impact_slippage = 0.0
    timing_slippage = 0.0
    alpha_decay = 0.0

    for i in eachindex(exec_prices)
        n_i = exec_shares[i]
        mid = market_prices[i]
        exec = exec_prices[i]

        # Spread: exec vs mid
        spread_cost_i = abs(exec - mid) * n_i
        spread_slippage += spread_cost_i

        # Impact: change in mid price attributed to our order
        # (simplified: price move beyond spread)
        if i > 1
            price_move = (market_prices[i] - market_prices[i-1])
            # Adjust for market drift vs impact
            impact_slippage += max(0.0, price_move) * n_i * 0.5
        end
    end

    # Timing slippage: arrival price vs VWAP
    timing_slippage = abs(decision_price - vwap_benchmark) * total_shares
    # Alpha decay: opportunity cost of not trading all at once
    alpha_decay = is_dollars - spread_slippage - impact_slippage

    return TCAResult(
        # Pre-trade (not computed here, set to 0)
        0.0, 0.0, 0.0, 0.0,
        # Post-trade
        is_dollars,
        spread_slippage,
        impact_slippage,
        # Prices
        decision_price,
        decision_price,
        avg_exec_price,
        vwap_benchmark,
        # Attribution
        spread_slippage,
        impact_slippage,
        timing_slippage,
        alpha_decay,
        # Metrics
        is_bps,
        vwap_slip_bps
    )
end

"""
    tca_summary(result::TCAResult) -> DataFrame

Return a formatted DataFrame summary of TCA results.
"""
function tca_summary(result::TCAResult)
    notional = result.decision_price > 0 ? result.decision_price : 1.0
    DataFrame(
        metric=["IS (bps)", "VWAP Slippage (bps)",
                "Spread Slippage (\$)", "Impact Slippage (\$)",
                "Timing Slippage (\$)", "Alpha Decay (\$)",
                "Avg Exec Price", "VWAP Benchmark", "Decision Price"],
        value=[result.is_bps, result.vwap_slippage_bps,
               result.spread_slippage, result.impact_slippage,
               result.timing_slippage, result.alpha_decay,
               result.avg_execution_price, result.benchmark_price,
               result.decision_price]
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Multi-Asset Execution with Correlation-Adjusted Impact
# ─────────────────────────────────────────────────────────────────────────────

"""
    MultiAssetExecution

Parameters for multi-asset correlated execution.

# Fields
- `X::Vector{Float64}`: Target trade vector (shares per asset)
- `T::Float64`: Execution horizon
- `N::Int`: Number of time steps
- `sigma::Vector{Float64}`: Per-asset volatilities
- `Sigma::Matrix{Float64}`: Covariance matrix of returns
- `eta::Vector{Float64}`: Temporary impact coefficients per asset
- `gamma::Vector{Float64}`: Permanent impact coefficients per asset
- `Lambda::Matrix{Float64}`: Cross-impact matrix (temporary)
- `lambda::Float64`: Risk aversion
"""
struct MultiAssetExecution
    X::Vector{Float64}
    T::Float64
    N::Int
    sigma::Vector{Float64}
    Sigma::Matrix{Float64}  # return covariance
    eta::Vector{Float64}
    gamma::Vector{Float64}
    Lambda::Matrix{Float64}  # cross-impact (temporary)
    lambda::Float64

    function MultiAssetExecution(X, T, N, sigma, Sigma, eta, gamma, lambda;
                                  cross_impact_frac=0.1)
        n_assets = length(X)
        # Build cross-impact matrix: diagonal = eta, off-diagonal = cross_impact_frac * sqrt(eta_i * eta_j)
        Lambda = zeros(n_assets, n_assets)
        for i in 1:n_assets
            for j in 1:n_assets
                if i == j
                    Lambda[i, j] = eta[i]
                else
                    Lambda[i, j] = cross_impact_frac * sqrt(eta[i] * eta[j])
                end
            end
        end
        new(X, T, N, sigma, Sigma, eta, gamma, Lambda, lambda)
    end
end

"""
    multi_asset_optimal(mae::MultiAssetExecution) -> NamedTuple

Compute optimal multi-asset execution trajectory accounting for cross-asset impact.

Uses matrix generalization of Almgren-Chriss:
Minimize E[IS] + lambda * Var[IS] where:
  E[IS] = sum_j { n_j' * (Lambda/tau) * n_j } + permanent impact
  Var[IS] = tau * sum_j { x_j' * Sigma * x_j }

Solved via matrix Riccati equation / numerical optimization.
"""
function multi_asset_optimal(mae::MultiAssetExecution)
    n_assets = length(mae.X)
    N = mae.N
    tau = mae.T / N
    X = mae.X
    Sigma = mae.Sigma
    Lambda = mae.Lambda
    lambda = mae.lambda

    # Per-step variance contribution matrix
    Q = lambda * tau * Sigma

    # Per-step impact cost matrix
    R = (1.0 / tau) * Lambda

    # Solve via dynamic programming / matrix Riccati
    # For each step j: x_{j+1} = x_j - n_j
    # Cost-to-go: V_j(x) = x' * P_j * x + q_j
    # with P_N = 0, and backward recursion:
    # P_{j-1} = Q + P_j - P_j * (R + P_j)^{-1} * P_j

    P = zeros(n_assets, n_assets)
    P_history = [copy(P)]

    for j in N:-1:1
        # Backward Riccati step
        M = R + P
        M_inv = inv(M)
        P_new = Q + P - P * M_inv * P
        P = P_new
        pushfirst!(P_history, copy(P))
    end

    # Forward pass: compute optimal trades
    holdings = zeros(n_assets, N + 1)
    holdings[:, 1] = X
    trades = zeros(n_assets, N)

    for j in 1:N
        P_j = P_history[j+1]
        M = Lambda / tau + P_j
        n_opt = -inv(M) * P_j * holdings[:, j]
        # Clamp to not overshoot
        for i in 1:n_assets
            if X[i] > 0
                n_opt[i] = clamp(n_opt[i], -holdings[i, j], 0.0)
            else
                n_opt[i] = clamp(n_opt[i], 0.0, -holdings[i, j])
            end
        end
        trades[:, j] = n_opt
        holdings[:, j+1] = holdings[:, j] + n_opt
    end
    # Ensure liquidation
    holdings[:, end] = zeros(n_assets)

    # Compute costs
    total_impact_cost = 0.0
    total_variance = 0.0

    for j in 1:N
        n_j = trades[:, j]
        x_j = holdings[:, j]
        total_impact_cost += dot(n_j, (Lambda / tau) * n_j)
        total_variance += tau * dot(x_j, Sigma * x_j)
    end

    permanent_cost = 0.5 * dot(X, diagm(mae.gamma) * X)

    return (
        holdings=holdings,
        trades=trades,
        impact_cost=total_impact_cost,
        permanent_cost=permanent_cost,
        variance=total_variance,
        utility=total_impact_cost + permanent_cost + lambda * total_variance,
        P_history=P_history
    )
end

"""
    correlation_adjusted_impact(
        eta::Vector{Float64},
        corr::Matrix{Float64},
        trade_vec::Vector{Float64}
    ) -> Vector{Float64}

Compute cross-asset price impact adjusted for correlation.
Cross-impact coefficient between assets i and j scales with sqrt(corr[i,j]).
"""
function correlation_adjusted_impact(
    eta::Vector{Float64},
    corr::Matrix{Float64},
    trade_vec::Vector{Float64}
)
    n = length(eta)
    impact = zeros(n)
    for i in 1:n
        for j in 1:n
            rho_ij = abs(corr[i, j])
            impact[i] += sqrt(eta[i] * eta[j]) * sqrt(rho_ij) * trade_vec[j]
        end
    end
    return impact
end

"""
    execution_shortfall(
        initial_price::Vector{Float64},
        exec_prices::Matrix{Float64},   # n_assets × n_steps
        exec_shares::Matrix{Float64}    # n_assets × n_steps
    ) -> Vector{Float64}

Compute realized implementation shortfall per asset.
"""
function execution_shortfall(
    initial_price::Vector{Float64},
    exec_prices::Matrix{Float64},
    exec_shares::Matrix{Float64}
)
    n_assets = size(exec_prices, 1)
    is = zeros(n_assets)

    for i in 1:n_assets
        total_shares = sum(exec_shares[i, :])
        avg_price = sum(exec_prices[i, :] .* exec_shares[i, :]) / total_shares
        is[i] = (avg_price - initial_price[i]) * total_shares
    end

    return is
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Intraday Volume Profile Models
# ─────────────────────────────────────────────────────────────────────────────

"""
    u_shaped_volume_profile(N::Int; open_weight=2.5, close_weight=2.0) -> Vector{Float64}

Generate a U-shaped intraday volume profile (higher volume near open/close).
"""
function u_shaped_volume_profile(N::Int; open_weight::Float64=2.5, close_weight::Float64=2.0)
    t = range(0, 1, length=N)
    # U-shape: beta distribution-like mixture
    profile = @. 1.0 + open_weight * exp(-20.0 * t^2) + close_weight * exp(-20.0 * (1.0 - t)^2)
    return profile ./ sum(profile)
end

"""
    fit_volume_profile(historical_volumes::Matrix{Float64}) -> Vector{Float64}

Fit an average intraday volume profile from historical data.
`historical_volumes` is n_days × n_intervals matrix.
"""
function fit_volume_profile(historical_volumes::Matrix{Float64})
    n_days, n_intervals = size(historical_volumes)
    # Normalize each day
    daily_sums = sum(historical_volumes, dims=2)
    normalized = historical_volumes ./ daily_sums
    # Average across days
    profile = vec(mean(normalized, dims=1))
    return profile ./ sum(profile)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Impact Model Calibration
# ─────────────────────────────────────────────────────────────────────────────

"""
    calibrate_impact_params(
        trade_sizes::Vector{Float64},
        price_impacts::Vector{Float64},
        ADV::Float64
    ) -> NamedTuple

Calibrate power-law temporary impact model: impact = alpha * (size/ADV)^beta
using OLS in log-log space.
"""
function calibrate_impact_params(
    trade_sizes::Vector{Float64},
    price_impacts::Vector{Float64},
    ADV::Float64
)
    # Filter positive values
    mask = (trade_sizes .> 0) .& (price_impacts .> 0)
    x = log.(trade_sizes[mask] ./ ADV)
    y = log.(price_impacts[mask])

    n = length(x)
    if n < 2
        return (alpha=0.0, beta=0.6, r_squared=NaN)
    end

    # OLS: y = a + beta * x
    x_mean = mean(x)
    y_mean = mean(y)
    beta = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean) .^ 2)
    a = y_mean - beta * x_mean
    alpha = exp(a)

    # R-squared
    y_pred = a .+ beta .* x
    ss_res = sum((y .- y_pred) .^ 2)
    ss_tot = sum((y .- y_mean) .^ 2)
    r2 = 1.0 - ss_res / ss_tot

    return (alpha=alpha, beta=beta, r_squared=r2)
end

"""
    square_root_impact(sigma::Float64, ADV::Float64, shares::Float64;
                       alpha=0.1, beta=0.5) -> Float64

Square-root market impact model (Almgren et al. 2005):
MI = alpha * sigma * (shares / ADV)^beta

Returns impact in price units (fraction of price).
"""
function square_root_impact(sigma::Float64, ADV::Float64, shares::Float64;
                             alpha::Float64=0.1, beta::Float64=0.5)
    return alpha * sigma * (shares / ADV)^beta
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Liquidation Strategies Under Constraints
# ─────────────────────────────────────────────────────────────────────────────

"""
    constrained_liquidation(
        p::AlmgrenChrissParams;
        max_trade_rate::Float64=Inf,
        min_trade_rate::Float64=0.0,
        must_liquidate_by::Int=0
    ) -> ACTrajectory

Compute constrained optimal liquidation with bounds on trade rates.
Uses quadratic programming structure solved iteratively.
"""
function constrained_liquidation(
    p::AlmgrenChrissParams;
    max_trade_rate::Float64=Inf,
    min_trade_rate::Float64=0.0,
    must_liquidate_by::Int=0
)
    N = p.N
    X = p.X
    tau = p.tau

    # Start with unconstrained solution
    base_traj = ac_optimal_trajectory(p)
    trades = copy(base_traj.trades)

    # Apply constraints
    for i in eachindex(trades)
        rate = abs(trades[i]) / tau
        if rate > max_trade_rate
            trades[i] = sign(trades[i]) * max_trade_rate * tau
        end
        if abs(trades[i]) < min_trade_rate * tau
            trades[i] = sign(trades[i]) * min_trade_rate * tau
        end
    end

    # Re-normalize to ensure X is fully executed
    total_traded = sum(abs.(trades))
    if total_traded < X - 1e-8
        # Add remaining to last step
        remaining = X - total_traded
        trades[end] -= remaining  # selling: negative trade
    end

    # Recompute holdings
    holdings = vcat([X], X .+ cumsum(trades))
    holdings[end] = 0.0

    times = base_traj.times
    E_cost = ac_expected_cost(p, trades)
    V_cost = ac_variance_cost(p, holdings)

    return ACTrajectory(times, holdings, trades, E_cost, V_cost, E_cost + p.lambda * V_cost)
end

"""
    risk_limit_trajectory(
        p::AlmgrenChrissParams,
        var_limit::Float64
    ) -> ACTrajectory

Find minimum-cost trajectory subject to VaR limit: sqrt(Var[IS]) <= var_limit.
Binary search on lambda to find the constraint-satisfying solution.
"""
function risk_limit_trajectory(p::AlmgrenChrissParams, var_limit::Float64)
    # Binary search for lambda such that sqrt(Var) = var_limit
    lam_lo = 1e-10
    lam_hi = 1e6

    for _ in 1:100
        lam_mid = sqrt(lam_lo * lam_hi)
        p_mid = AlmgrenChrissParams(p.X, p.T, p.N, p.sigma, p.eta, p.gamma, p.epsilon, lam_mid)
        traj = ac_optimal_trajectory(p_mid)
        std_cost = sqrt(traj.variance_cost)

        if abs(std_cost - var_limit) / (var_limit + 1e-10) < 1e-4
            return traj
        elseif std_cost > var_limit
            lam_lo = lam_mid
        else
            lam_hi = lam_mid
        end
    end

    # Return solution at midpoint
    lam_mid = sqrt(lam_lo * lam_hi)
    p_mid = AlmgrenChrissParams(p.X, p.T, p.N, p.sigma, p.eta, p.gamma, p.epsilon, lam_mid)
    return ac_optimal_trajectory(p_mid)
end

end # module OptimalExecution
