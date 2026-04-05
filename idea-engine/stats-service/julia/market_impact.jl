"""
market_impact.jl — Market Impact Models for Trading

Covers:
  - Almgren-Chriss linear impact (permanent + temporary)
  - Square-root market impact (empirical law)
  - Propagator / Transient Impact model with decay kernel
  - Kyle's lambda: market depth from order flow
  - Obizhaeva-Wang model with resilience
  - Cross-asset impact: trading one asset affects another
  - Intraday impact: time-of-day scaling (U-shaped volume)
  - Implementation shortfall decomposition
  - Optimal execution under uncertain impact

Pure Julia stdlib only. No external dependencies.
"""

module MarketImpact

using Statistics, LinearAlgebra, Random

export AlmgrenChriss, ac_optimal_trajectory, ac_cost
export sqrt_impact_model, sqrt_impact_cost
export PropagatorModel, propagator_impact, propagator_trajectory
export kyle_lambda, kyle_optimal_trade
export ObizhaevaNang, ow_optimal_trajectory
export cross_impact_matrix, cross_impact_cost
export intraday_volume_profile, intraday_impact_scaling
export implementation_shortfall, shortfall_decompose
export optimal_execution_uncertain_impact
export run_market_impact_demo

# ─────────────────────────────────────────────────────────────
# 1. ALMGREN-CHRISS MODEL
# ─────────────────────────────────────────────────────────────

"""
    AlmgrenChriss

Almgren-Chriss optimal liquidation model.

Fields:
  X     — initial inventory (shares to sell)
  T     — liquidation horizon (time steps)
  sigma — price volatility (per step)
  eta   — temporary impact coefficient
  gamma — permanent impact coefficient
  rho   — risk aversion parameter
"""
struct AlmgrenChriss
    X::Float64       # shares to liquidate
    T::Int           # number of periods
    sigma::Float64   # price vol per step
    eta::Float64     # temporary impact coeff ($/share per share/step)
    gamma::Float64   # permanent impact coeff ($/share per share/step)
    rho::Float64     # risk aversion
end

"""
    ac_optimal_trajectory(model) -> Vector{Float64}

Compute optimal liquidation trajectory (shares remaining at each step).
Closed-form solution from Almgren & Chriss (2001).

x_j = X * sinh(κ(T-j)*Δt) / sinh(κ*T*Δt)
where κ = sqrt(rho*sigma^2 / eta * Δt) approximately.
"""
function ac_optimal_trajectory(m::AlmgrenChriss)::Vector{Float64}
    T = m.T
    # Characteristic time scale κ
    # For continuous limit: κ² = ρσ²/η
    kappa = sqrt(m.rho * m.sigma^2 / (m.eta + 1e-15))

    trajectory = zeros(T + 1)
    trajectory[1] = m.X

    for j in 0:T-1
        # Optimal inventory at time j
        num = sinh(kappa * (T - j))
        den = sinh(kappa * T)
        den < 1e-10 && (den = 1e-10)
        trajectory[j + 1] = m.X * num / den
    end
    trajectory[end] = 0.0
    trajectory
end

"""
    ac_cost(model, trajectory) -> NamedTuple

Compute expected cost and variance for a given trading trajectory.
"""
function ac_cost(m::AlmgrenChriss, trajectory::Vector{Float64})
    n = length(trajectory) - 1
    trades    = -diff(trajectory)  # shares sold each period (positive)
    exp_cost  = 0.0
    variance  = 0.0

    for j in 1:n
        v_j = trades[j]  # trade rate
        # Permanent impact cost (paid on remaining inventory)
        exp_cost += m.gamma * v_j * trajectory[j]
        # Temporary impact cost (bid-ask + urgency)
        exp_cost += m.eta * v_j^2
        # Variance contribution from remaining inventory
        variance += m.sigma^2 * trajectory[j]^2
    end

    (expected_cost=exp_cost, variance=variance,
     risk_adjusted_cost=exp_cost + m.rho * variance,
     total_trades=sum(abs.(trades)))
end

"""
    ac_efficient_frontier(X, T, sigma, eta, gamma; n_rho=20) -> Vector{NamedTuple}

Compute efficient frontier of (expected cost, variance) pairs
by varying risk aversion rho.
"""
function ac_efficient_frontier(X::Float64, T::Int, sigma::Float64,
                                 eta::Float64, gamma::Float64;
                                 n_rho::Int=20)
    rhos = exp.(range(log(1e-6), log(1e2), length=n_rho))
    frontier = map(rhos) do rho
        m   = AlmgrenChriss(X, T, sigma, eta, gamma, rho)
        traj = ac_optimal_trajectory(m)
        cost = ac_cost(m, traj)
        (rho=rho, expected_cost=cost.expected_cost, variance=cost.variance)
    end
    frontier
end

# ─────────────────────────────────────────────────────────────
# 2. SQUARE-ROOT MARKET IMPACT
# ─────────────────────────────────────────────────────────────

"""
    sqrt_impact_model(Q, V, sigma, phi=0.314) -> Float64

Square-root market impact model (empirical law):
  MI = phi * sigma * sqrt(Q / V)
where:
  Q   — order quantity (shares)
  V   — average daily volume (shares)
  sigma — daily volatility
  phi — universal constant ≈ 0.314 (Almgren et al. 2005)
"""
function sqrt_impact_model(Q::Float64, V::Float64, sigma::Float64,
                             phi::Float64=0.314)::Float64
    phi * sigma * sqrt(Q / max(V, 1.0))
end

"""
    sqrt_impact_cost(Q, V, sigma, S0, phi=0.314) -> Float64

Total cost of executing order Q given square-root impact.
MI is in price units (fraction of price), so cost = Q * S0 * MI.
"""
function sqrt_impact_cost(Q::Float64, V::Float64, sigma::Float64,
                            S0::Float64, phi::Float64=0.314)::Float64
    mi = sqrt_impact_model(Q, V, sigma, phi)
    Q * S0 * mi
end

"""
    sqrt_impact_schedule(total_Q, n_slices, V_daily, sigma, S0; phi=0.314)
       -> NamedTuple

Optimal child order schedule to minimize total square-root impact.
For sqrt impact, equal splitting is suboptimal; use V-shaped schedule.
"""
function sqrt_impact_schedule(total_Q::Float64, n_slices::Int,
                                V_daily::Float64, sigma::Float64,
                                S0::Float64; phi::Float64=0.314)
    # For concave impact (MI ∝ Q^0.5), split evenly is approximately optimal
    # More sophisticated: use time-weighted average price (TWAP) with volume weighting
    v_intraday = intraday_volume_profile(n_slices)
    slices     = total_Q .* v_intraday ./ sum(v_intraday)  # volume-weighted

    costs   = [sqrt_impact_cost(slices[i], V_daily * v_intraday[i], sigma, S0, phi)
               for i in 1:n_slices]
    total_cost = sum(costs)
    (slices=slices, costs=costs, total_cost=total_cost,
     avg_impact=total_cost / (total_Q * S0))
end

# ─────────────────────────────────────────────────────────────
# 3. PROPAGATOR MODEL (TRANSIENT IMPACT)
# ─────────────────────────────────────────────────────────────

"""
    PropagatorModel

Transient/propagator model for market impact decay.
Price impact of a trade decays over time according to a kernel G(t).

Fields:
  kernel  — function G(t) ≥ 0, impact decay kernel
  lambda  — impact coefficient (impact per unit trade)
  sigma   — volatility of idiosyncratic price moves
"""
struct PropagatorModel
    kernel::Function
    lambda::Float64
    sigma::Float64
end

"""Power-law decay kernel: G(t) = (1 + t/t0)^{-beta}."""
power_law_kernel(beta::Float64=0.5, t0::Float64=1.0) =
    t -> (1.0 + t / t0)^(-beta)

"""Exponential decay kernel: G(t) = exp(-t/tau)."""
exp_decay_kernel(tau::Float64=10.0) = t -> exp(-t / tau)

"""
    propagator_impact(model, trades, times) -> Vector{Float64}

Compute price impact path given a sequence of trades at given times.
Impact at time t = λ * Σ_{s≤t} n_s * G(t - s)
"""
function propagator_impact(model::PropagatorModel,
                             trades::Vector{Float64},
                             times::Vector{Float64})::Vector{Float64}
    T = length(trades)
    impact = zeros(T)
    for t in 1:T
        s = 0.0
        for s_idx in 1:t
            dt = times[t] - times[s_idx]
            s += trades[s_idx] * model.kernel(dt)
        end
        impact[t] = model.lambda * s
    end
    impact
end

"""
    propagator_trajectory(model, total_Q, T, dt=1.0) -> NamedTuple

Optimal execution trajectory minimizing expected cost under propagator model.
Uses numerical optimization (gradient descent on slice sizes).
"""
function propagator_trajectory(model::PropagatorModel,
                                 total_Q::Float64, T::Int,
                                 dt::Float64=1.0;
                                 maxiter::Int=200, lr::Float64=1e-4)
    times  = Float64[dt * t for t in 1:T]
    # Initialize: equal slicing
    trades = fill(total_Q / T, T)

    # Gradient descent to minimize total cost = Σ_t n_t * I_t
    for _ in 1:maxiter
        impact = propagator_impact(model, trades, times)
        cost   = dot(trades, impact)
        # Numerical gradient
        grad = zeros(T)
        eps  = 1e-5
        for i in 1:T
            trades_p = copy(trades); trades_p[i] += eps
            trades_m = copy(trades); trades_m[i] -= eps
            cost_p = dot(trades_p, propagator_impact(model, trades_p, times))
            cost_m = dot(trades_m, propagator_impact(model, trades_m, times))
            grad[i] = (cost_p - cost_m) / (2eps)
        end
        trades .-= lr .* grad
        # Project onto simplex: trades ≥ 0 and sum = total_Q
        trades = max.(trades, 0.0)
        s = sum(trades)
        s > 0 && (trades .*= total_Q / s)
    end

    impact = propagator_impact(model, trades, times)
    total_cost = dot(trades, impact)
    (trades=trades, impact=impact, total_cost=total_cost, times=times)
end

# ─────────────────────────────────────────────────────────────
# 4. KYLE'S LAMBDA
# ─────────────────────────────────────────────────────────────

"""
    kyle_lambda(price_changes, order_flow) -> Float64

Estimate Kyle's lambda (market depth parameter) via OLS regression:
  Δp_t = λ * OF_t + ε_t
where OF_t = signed order flow (buy vol - sell vol).

Higher λ → less liquid market, more price impact per unit order flow.
"""
function kyle_lambda(price_changes::Vector{Float64},
                      order_flow::Vector{Float64})::Float64
    n = min(length(price_changes), length(order_flow))
    x = order_flow[1:n]
    y = price_changes[1:n]
    # OLS: λ = cov(x,y) / var(x)
    vx = var(x)
    vx < 1e-15 && return 0.0
    cov(x, y) / vx
end

"""
    kyle_optimal_trade(lambda, sigma_info, sigma_noise) -> Float64

Kyle equilibrium: optimal trade size for informed trader.
sigma_info = std of private information signal
sigma_noise = std of noise trading
"""
function kyle_optimal_trade(lambda::Float64, sigma_info::Float64,
                              sigma_noise::Float64)::Float64
    # Kyle (1985) equilibrium: informed trades β = σ_u / (2λσ_v)
    # where σ_u = noise vol, σ_v = fundamental vol
    lambda <= 0 && return 0.0
    sigma_noise / (2 * lambda * sigma_info + 1e-10)
end

"""
    estimate_kyle_lambda_rolling(prices, volumes, window=60) -> Vector{Float64}

Rolling estimation of Kyle's lambda over time.
"""
function estimate_kyle_lambda_rolling(prices::Vector{Float64},
                                       signed_volumes::Vector{Float64},
                                       window::Int=60)::Vector{Float64}
    n = length(prices)
    rets = [0.0; diff(log.(prices))]
    lambdas = zeros(n)
    for t in window:n
        dp = rets[t-window+1:t]
        of = signed_volumes[t-window+1:t]
        lambdas[t] = kyle_lambda(dp, of)
    end
    lambdas
end

# ─────────────────────────────────────────────────────────────
# 5. OBIZHAEVA-WANG MODEL (LIMIT ORDER BOOK RESILIENCE)
# ─────────────────────────────────────────────────────────────

"""
    ObizhaevaWang

Obizhaeva-Wang model with LOB resilience.

Fields:
  rho   — resilience rate (LOB replenishment)
  Q0    — initial LOB depth (shares available at best)
  eta   — temporary impact per unit trade
  sigma — price volatility
  X     — total inventory to execute
  T     — execution horizon
"""
struct ObizhaevaWang
    rho::Float64    # resilience (mean-reversion rate of LOB)
    Q0::Float64     # LOB depth
    eta::Float64    # temporary impact
    sigma::Float64  # volatility
    X::Float64      # total order
    T::Float64      # execution horizon
end

"""
    ow_optimal_trajectory(model, n_steps) -> NamedTuple

Optimal execution trajectory under OW model.
In the OW model, the optimal strategy has block trades at the start/end
with a continuous trading in between.
"""
function ow_optimal_trajectory(m::ObizhaevaWang, n_steps::Int=50)
    dt  = m.T / n_steps
    times = [dt * t for t in 0:n_steps]

    # OW optimal: triangular rate (linear in time for simple case)
    # More generally: solve continuous-time HJB equation
    # Simplified: rate proportional to remaining inventory with resilience adjustment
    rho  = m.rho
    eta  = m.eta

    # Characteristic rate: nu* satisfying the HJB FOC
    # nu = nu_0 * exp(-rho * t) approximately
    nu_0 = m.X * rho / (1 - exp(-rho * m.T) + 1e-10)

    trades   = [nu_0 * exp(-rho * t) * dt for t in times[1:end-1]]
    trades  .*= m.X / sum(trades)  # normalize

    inventory = [m.X; m.X .- cumsum(trades)]

    # Cost computation (temporary + resilience effect)
    impact_cost = 0.0
    D = 0.0  # current LOB displacement
    for (i, v) in enumerate(trades)
        D = D * exp(-rho * dt) + v / (m.Q0 + 1e-10)
        impact_cost += eta * v^2 / dt + 0.5 * D * v
    end

    (trades=trades, inventory=inventory, times=times,
     impact_cost=impact_cost, avg_rate=mean(trades)/dt)
end

# ─────────────────────────────────────────────────────────────
# 6. CROSS-IMPACT
# ─────────────────────────────────────────────────────────────

"""
    cross_impact_matrix(returns, volumes, n_assets; lag=0) -> Matrix{Float64}

Estimate cross-impact matrix: Λ[i,j] = impact of trading asset j on asset i's price.
Uses multivariate regression of price changes on signed order flows.
"""
function cross_impact_matrix(returns::Matrix{Float64},
                               signed_volumes::Matrix{Float64};
                               lag::Int=0)::Matrix{Float64}
    n_t, n_assets = size(returns)
    Lambda = zeros(n_assets, n_assets)
    for i in 1:n_assets
        # Regress returns[i] on all signed volumes (with lag)
        y = returns[lag+1:end, i]
        X = signed_volumes[1:end-lag, :]
        # OLS: Λ[i,:] = (X'X)^{-1} X'y
        XtX = X' * X + 1e-6 * I
        Xty = X' * y
        Lambda[i, :] = XtX \ Xty
    end
    Lambda
end

"""
    cross_impact_cost(Lambda, trades_vector, price_vector) -> Float64

Total cross-impact cost of executing a basket of trades.
trades_vector: signed order sizes per asset
"""
function cross_impact_cost(Lambda::Matrix{Float64},
                             trades::Vector{Float64},
                             prices::Vector{Float64})::Float64
    # Price changes due to cross-impact
    price_changes = Lambda * trades
    # Cost = sum of trade_i * price_change_i
    dot(trades, price_changes .* prices)
end

"""
    optimal_basket_execution(Lambda, target_trades, prices; rho=1.0) -> Vector{Float64}

Optimal execution order for a basket trade considering cross-impact.
Minimizes (E[cost] + rho * Var[cost]).
Simple closed form: trades = (Lambda + Lambda') \ (-rho * diag_term)
"""
function optimal_basket_execution(Lambda::Matrix{Float64},
                                    target_trades::Vector{Float64},
                                    prices::Vector{Float64};
                                    rho::Float64=1.0)::Vector{Float64}
    n = length(target_trades)
    # Symmetrize impact matrix
    Sym = 0.5 .* (Lambda .+ Lambda')
    # Optimal proportional split (simplified: scale to minimize impact)
    # Full problem requires solving Sym * trades = target_trades
    # subject to constraint that total exposure matches
    try
        return (Sym + rho * 1e-4 * I) \ target_trades
    catch
        return target_trades
    end
end

# ─────────────────────────────────────────────────────────────
# 7. INTRADAY IMPACT PROFILE
# ─────────────────────────────────────────────────────────────

"""
    intraday_volume_profile(n_steps=48) -> Vector{Float64}

U-shaped intraday volume profile (normalized to sum to 1).
Higher volume at open and close, lower at midday.
"""
function intraday_volume_profile(n_steps::Int=48)::Vector{Float64}
    times = range(0.0, 1.0, length=n_steps)
    # U-shaped: vol ∝ exp(-((t - 0.5) / 0.3)^2) is inverse of what we want
    # U-shape: high at t=0 and t=1, low at t=0.5
    profile = [(t - 0.5)^2 * 4 + 0.3 for t in times]
    profile ./= sum(profile)
    profile
end

"""
    intraday_impact_scaling(n_steps=48) -> Vector{Float64}

Impact scaling factor by time of day.
Impact is higher when volume is lower (thin market).
"""
function intraday_impact_scaling(n_steps::Int=48)::Vector{Float64}
    vol_profile = intraday_volume_profile(n_steps)
    # Impact ∝ 1/sqrt(volume) (from square-root model)
    scaling = 1.0 ./ sqrt.(vol_profile .* n_steps)
    scaling ./ mean(scaling)  # normalize so mean = 1
end

"""
    vwap_schedule(total_Q, n_steps) -> Vector{Float64}

VWAP execution schedule: trade proportional to volume profile.
"""
function vwap_schedule(total_Q::Float64, n_steps::Int=48)::Vector{Float64}
    vol = intraday_volume_profile(n_steps)
    total_Q .* vol
end

"""
    twap_schedule(total_Q, n_steps) -> Vector{Float64}

TWAP execution schedule: equal slices over time.
"""
twap_schedule(total_Q::Float64, n_steps::Int=48) = fill(total_Q / n_steps, n_steps)

# ─────────────────────────────────────────────────────────────
# 8. IMPLEMENTATION SHORTFALL DECOMPOSITION
# ─────────────────────────────────────────────────────────────

"""
    implementation_shortfall(arrival_price, executed_prices, quantities,
                               decision_price) -> NamedTuple

Compute implementation shortfall (IS) and decompose into components.

IS = paper_return - actual_return
   = timing_cost + market_impact_cost + opportunity_cost
"""
function implementation_shortfall(arrival_price::Float64,
                                    executed_prices::Vector{Float64},
                                    quantities::Vector{Float64},
                                    decision_price::Float64,
                                    final_price::Float64)
    total_Q    = sum(abs.(quantities))
    total_Q < 1e-10 && return (is=0.0, timing=0.0, impact=0.0, opportunity=0.0)

    # Weighted average execution price
    wavg_price = dot(abs.(quantities), executed_prices) / total_Q

    # IS relative to arrival price
    is = (wavg_price - arrival_price) / arrival_price

    # Decomposition:
    # Timing cost: price drift from decision to arrival
    timing = (arrival_price - decision_price) / decision_price

    # Market impact: executed price vs arrival
    market_impact = (wavg_price - arrival_price) / arrival_price

    # Opportunity cost: unexecuted portion × price drift after execution
    executed_frac = total_Q  # (assume full execution here)
    opportunity   = 0.0  # simplified

    (is=is, timing=timing, market_impact=market_impact,
     opportunity=opportunity, wavg_execution_price=wavg_price,
     arrival_price=arrival_price, final_price=final_price)
end

"""
    shortfall_decompose(pre_trade_price, trades, prices, vwap_ref) -> NamedTuple

Detailed IS decomposition following Kissell & Glantz (2003).
"""
function shortfall_decompose(pre_trade_price::Float64,
                               trades::Vector{Float64},
                               prices::Vector{Float64},
                               vwap_ref::Float64)
    total_Q = sum(abs.(trades))
    total_Q < 1e-10 && return nothing

    wavg   = dot(abs.(trades), prices) / total_Q
    is_bps = (wavg - pre_trade_price) / pre_trade_price * 10_000
    vs_vwap = (wavg - vwap_ref) / vwap_ref * 10_000

    # Timing: price change in first period
    timing_bps = length(prices) > 1 ?
                 (prices[1] - pre_trade_price) / pre_trade_price * 10_000 : 0.0

    # Impact: WAVG vs first execution price
    impact_bps = (wavg - prices[1]) / prices[1] * 10_000

    (is_bps=is_bps, vs_vwap_bps=vs_vwap, timing_bps=timing_bps,
     impact_bps=impact_bps, wavg=wavg)
end

# ─────────────────────────────────────────────────────────────
# 9. OPTIMAL EXECUTION UNDER UNCERTAIN IMPACT
# ─────────────────────────────────────────────────────────────

"""
    optimal_execution_uncertain_impact(X, T, sigma, eta_mean, eta_vol,
                                        gamma, rho; n_sim=1000, rng=...)
       -> NamedTuple

Robust optimal execution when impact parameters are uncertain.
Monte Carlo over impact coefficient draws, optimize expected risk-adjusted cost.
"""
function optimal_execution_uncertain_impact(X::Float64, T::Int,
                                              sigma::Float64,
                                              eta_mean::Float64,
                                              eta_vol::Float64,
                                              gamma::Float64,
                                              rho::Float64;
                                              n_sim::Int=1000,
                                              rng=MersenneTwister(42))
    # For each eta draw, compute optimal trajectory and cost
    eta_samples = max.(eta_mean .+ eta_vol .* randn(rng, n_sim), 1e-8)

    all_costs     = zeros(n_sim)
    all_variances = zeros(n_sim)

    # Use the mean-eta trajectory as a robust baseline
    m_mean   = AlmgrenChriss(X, T, sigma, eta_mean, gamma, rho)
    traj_mean = ac_optimal_trajectory(m_mean)

    for (i, eta_i) in enumerate(eta_samples)
        m_i   = AlmgrenChriss(X, T, sigma, eta_i, gamma, rho)
        cost_i = ac_cost(m_i, traj_mean)  # cost of mean trajectory under eta_i
        all_costs[i]     = cost_i.expected_cost
        all_variances[i] = cost_i.variance
    end

    # Worst-case (95th percentile) and robust cost
    sorted_costs = sort(all_costs)
    robust_cost  = sorted_costs[Int(ceil(0.95 * n_sim))]

    # Compare with per-scenario optimal
    opt_costs = zeros(n_sim)
    for (i, eta_i) in enumerate(eta_samples)
        m_i  = AlmgrenChriss(X, T, sigma, eta_i, gamma, rho)
        tj   = ac_optimal_trajectory(m_i)
        c    = ac_cost(m_i, tj)
        opt_costs[i] = c.expected_cost
    end

    (mean_cost=mean(all_costs), std_cost=std(all_costs),
     robust_cost_95=robust_cost,
     optimal_per_scenario_cost=mean(opt_costs),
     robust_overhead=robust_cost - mean(opt_costs),
     trajectory=traj_mean)
end

# ─────────────────────────────────────────────────────────────
# 10. DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_market_impact_demo() -> Nothing

Demonstration of all market impact models.
"""
function run_market_impact_demo()
    println("=" ^ 60)
    println("MARKET IMPACT MODELS DEMO")
    println("=" ^ 60)

    # ── Almgren-Chriss ──
    println("\n1. Almgren-Chriss Optimal Liquidation")
    X = 10_000.0   # shares to sell
    T = 20         # periods (e.g., 20 minutes)
    m = AlmgrenChriss(X, T, 0.02, 1e-4, 5e-6, 1e-4)
    traj = ac_optimal_trajectory(m)
    cost = ac_cost(m, traj)
    println("  Inventory: $(m.X) shares over $T periods")
    println("  Trajectory[0:5]: $(round.(traj[1:6], digits=0))")
    println("  Expected Cost:   \$$(round(cost.expected_cost, digits=2))")
    println("  Risk-Adj Cost:   \$$(round(cost.risk_adjusted_cost, digits=2))")

    # TWAP comparison
    twap_traj = vcat(X .- cumsum(twap_schedule(X, T)), [0.0])
    cost_twap = ac_cost(m, twap_traj)
    println("  TWAP Cost:       \$$(round(cost_twap.expected_cost, digits=2))")
    println("  Savings vs TWAP: \$$(round(cost_twap.expected_cost - cost.expected_cost, digits=2))")

    # ── Square-Root Model ──
    println("\n2. Square-Root Market Impact")
    Q = 50_000.0; V = 5_000_000.0; sigma = 0.015; S0 = 50_000.0
    mi = sqrt_impact_model(Q, V, sigma)
    cost_sqrt = sqrt_impact_cost(Q, V, sigma, S0)
    println("  Order: $(Q) shares, ADV: $(V) shares")
    println("  Market Impact: $(round(mi * 100, digits=4))% of price")
    println("  Impact Cost:   \$$(round(cost_sqrt, digits=2))")
    schedule = sqrt_impact_schedule(Q, 10, V, sigma, S0)
    println("  10-slice schedule total cost: \$$(round(schedule.total_cost, digits=2))")
    println("  Avg impact per slice: $(round(schedule.avg_impact * 100, digits=4))%")

    # ── Propagator Model ──
    println("\n3. Propagator / Transient Impact Model")
    kern = power_law_kernel(0.5, 1.0)
    prop = PropagatorModel(kern, 1e-4, 0.01)
    trades_unif = fill(500.0, 20)
    times_p = Float64.(1:20)
    imp = propagator_impact(prop, trades_unif, times_p)
    println("  Final impact (uniform schedule): $(round(imp[end] * 100, digits=4))%")
    ptraj = propagator_trajectory(prop, 10_000.0, 10, 1.0; maxiter=50)
    println("  Propagator optimal cost: $(round(ptraj.total_cost, digits=4))")

    # ── Kyle Lambda ──
    println("\n4. Kyle's Lambda")
    rng_k = MersenneTwister(1)
    true_lambda = 0.001
    n_obs = 200
    of = randn(rng_k, n_obs)
    dp = true_lambda .* of .+ 0.01 .* randn(rng_k, n_obs)
    lambda_hat = kyle_lambda(dp, of)
    println("  True lambda: $true_lambda, Estimated: $(round(lambda_hat, digits=5))")
    optimal_trade = kyle_optimal_trade(lambda_hat, 0.02, 0.5)
    println("  Kyle optimal informed trade size: $(round(optimal_trade, digits=1)) shares")

    # ── Obizhaeva-Wang ──
    println("\n5. Obizhaeva-Wang (LOB Resilience)")
    ow = ObizhaevaWang(0.1, 10_000.0, 1e-4, 0.02, 10_000.0, 1.0)
    ow_result = ow_optimal_trajectory(ow, 20)
    println("  OW impact cost: $(round(ow_result.impact_cost, digits=4))")
    println("  Avg trade rate: $(round(ow_result.avg_rate, digits=1)) shares/step")

    # ── Cross-Impact ──
    println("\n6. Cross-Impact Matrix")
    rng_c = MersenneTwister(2)
    n_assets = 4; T_data = 500
    rets_cross = randn(rng_c, T_data, n_assets) .* 0.01
    vols_cross = randn(rng_c, T_data, n_assets) .* 1000
    Lambda = cross_impact_matrix(rets_cross, vols_cross)
    println("  Cross-impact matrix (diagonal): $(round.(diag(Lambda) .* 1e4, digits=2)) bps/(lot)")
    basket_trades = [1000.0, -500.0, 200.0, -300.0]
    prices_basket = [50000.0, 40000.0, 30000.0, 20000.0]
    cross_cost = cross_impact_cost(Lambda, basket_trades, prices_basket)
    println("  Cross-impact cost of basket: \$$(round(cross_cost, digits=2))")

    # ── Intraday Profile ──
    println("\n7. Intraday Volume & Impact Profile")
    vol_prof  = intraday_volume_profile(8)
    imp_scale = intraday_impact_scaling(8)
    println("  Vol profile (8 slots): $(round.(vol_prof .* 100, digits=1))%")
    println("  Impact scaling:        $(round.(imp_scale, digits=2))x")

    # ── Implementation Shortfall ──
    println("\n8. Implementation Shortfall")
    arr_price = 50_000.0
    exec_prices = arr_price .* (1 .+ cumsum(randn(MersenneTwister(3), 5) .* 0.001))
    quants = fill(200.0, 5)
    is_res = implementation_shortfall(arr_price, exec_prices, quants, 49_900.0, exec_prices[end])
    println("  IS (bps): $(round(is_res.is * 10000, digits=2))")
    println("  Timing (bps): $(round(is_res.timing * 10000, digits=2))")
    println("  Market Impact (bps): $(round(is_res.market_impact * 10000, digits=2))")

    println("\nDone.")
    nothing
end

# ─────────────────────────────────────────────────────────────
# 11. OPTIMAL SPLIT ORDER SCHEDULING
# ─────────────────────────────────────────────────────────────

"""
    optimal_split_order(total_Q, n_slices, eta, gamma, price, sigma) -> Vector{Float64}

Compute optimal child order schedule that minimizes expected total
market impact cost under linear temporary + permanent impact.
Uses Almgren-Chriss closed form adapted for discrete slices.
"""
function optimal_split_order(total_Q::Float64, n_slices::Int,
                               eta::Float64, gamma::Float64,
                               price::Float64, sigma::Float64)::Vector{Float64}
    uniform = fill(total_Q / n_slices, n_slices)
    denom = max(sinh(sqrt(max(gamma * sigma^2 / (eta + 1e-15), 0.0)) * n_slices), 1e-10)
    kappa = sqrt(max(gamma * sigma^2 / (eta + 1e-15), 0.0))
    T = Float64(n_slices)
    schedule = zeros(n_slices)
    for j in 1:n_slices
        t_j = j - 0.5
        v1 = cosh(kappa * (T - t_j))
        v2 = cosh(kappa * (T - t_j - 1))
        schedule[j] = (v1 - v2) / denom * total_Q
    end
    s_sum = sum(abs.(schedule))
    s_sum > 0 ? schedule .* (total_Q / s_sum) : uniform
end

"""
    participation_rate_constraint(total_Q, adv, max_prate) -> Float64

Maximum order size per period respecting a participation rate constraint.
"""
participation_rate_constraint(total_Q::Float64, adv::Float64, max_prate::Float64=0.15) =
    min(total_Q, max_prate * adv)

# ─────────────────────────────────────────────────────────────
# 12. MARKET MICROSTRUCTURE MEASURES
# ─────────────────────────────────────────────────────────────

"""
    roll_spread_estimator(prices) -> Float64

Roll (1984) implicit bid-ask spread estimator.
S = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
"""
function roll_spread_estimator(prices::Vector{Float64})::Float64
    n = length(prices); n < 3 && return 0.0
    dp = diff(prices)
    c  = cov(dp[1:end-1], dp[2:end])
    c >= 0 ? 0.0 : 2.0 * sqrt(-c)
end

"""
    amihud_illiquidity(returns, volumes) -> Vector{Float64}

Amihud (2002) illiquidity: |r_t| / Dollar_Volume_t.
Higher = less liquid.
"""
function amihud_illiquidity(returns::Vector{Float64},
                              volumes::Vector{Float64})::Vector{Float64}
    n = min(length(returns), length(volumes))
    abs.(returns[1:n]) ./ (abs.(volumes[1:n]) .+ 1e-10) .* 1e6
end

"""
    amihud_rolling(returns, volumes, window) -> Vector{Float64}

Rolling Amihud illiquidity ratio.
"""
function amihud_rolling(returns::Vector{Float64}, volumes::Vector{Float64},
                          window::Int=20)::Vector{Float64}
    illiq = amihud_illiquidity(returns, volumes)
    n = length(illiq)
    rolling = zeros(n)
    for t in window:n
        rolling[t] = mean(illiq[t-window+1:t])
    end
    rolling
end

"""
    bid_ask_spread_model(sigma, informed_fraction, avg_trade_size, inventory_cost) -> Float64

Glosten-Milgrom-style spread decomposition.
Spread = adverse_selection_component + order_processing_component.
"""
function bid_ask_spread_model(sigma::Float64, informed_fraction::Float64,
                               avg_trade_size::Float64,
                               inventory_cost::Float64=0.0005)::Float64
    adverse = 2.0 * informed_fraction * sigma * sqrt(avg_trade_size)
    adverse + inventory_cost
end

# ─────────────────────────────────────────────────────────────
# 13. VENUE ANALYSIS AND SMART ORDER ROUTING
# ─────────────────────────────────────────────────────────────

"""
    VenueProfile

Characteristics of a trading venue (exchange, dark pool, lit market).
"""
struct VenueProfile
    name::String
    fee_bps::Float64
    fill_rate::Float64
    avg_spread_bps::Float64
    latency_ms::Float64
    adv_fraction::Float64
end

"""
    smart_order_route(order_size, venues, sigma, urgency) -> Vector{Float64}

Allocate order across venues minimizing cost subject to fill probability.
"""
function smart_order_route(order_size::Float64, venues::Vector{VenueProfile},
                             sigma::Float64, urgency::Float64=0.5)::Vector{Float64}
    n_v = length(venues); n_v == 0 && return Float64[]
    scores = map(venues) do v
        fill_b  = urgency * v.fill_rate
        cost_p  = (1 - urgency) * (v.fee_bps + v.avg_spread_bps/2) / 10_000
        lat_p   = urgency * v.latency_ms / 100.0 * sigma
        fill_b - cost_p - lat_p
    end
    max_s = maximum(scores)
    w = exp.(scores .- max_s); w ./= sum(w)
    w .* order_size
end

"""
    execution_quality_metrics(executed_price, mid_price, decision_price, direction) -> NamedTuple

Compute execution quality (effective spread, price improvement, slippage).
"""
function execution_quality_metrics(executed_price::Float64, mid_price::Float64,
                                    decision_price::Float64, direction::Int=1)
    eff_spread = 2.0 * direction * (executed_price - mid_price) / mid_price * 10_000
    improvement = direction * (mid_price - executed_price) / mid_price * 10_000
    slippage   = direction * (executed_price - decision_price) / decision_price * 10_000
    (effective_spread_bps=eff_spread, price_improvement_bps=improvement, slippage_bps=slippage)
end

# ─────────────────────────────────────────────────────────────
# 14. IMPACT CALIBRATION FROM CRYPTO DATA
# ─────────────────────────────────────────────────────────────

"""
    calibrate_impact_parameters(trade_sizes, price_changes, volumes) -> NamedTuple

Calibrate linear and square-root impact model parameters from historical data.
"""
function calibrate_impact_parameters(trade_sizes::Vector{Float64},
                                       price_changes::Vector{Float64},
                                       volumes::Vector{Float64})
    n = min(length.([trade_sizes, price_changes, volumes])...)
    participation = trade_sizes[1:n] ./ (volumes[1:n] .+ 1.0)

    # Linear: MI = eta * Q
    X_lin = reshape(trade_sizes[1:n], n, 1)
    XtX = (X_lin'X_lin)[1,1]
    eta_lin = XtX > 0 ? (X_lin'price_changes[1:n])[1] / XtX : 0.0
    eta_lin = max(eta_lin, 0.0)

    # Sqrt: MI = phi * sigma * sqrt(Q/V)
    x_sqrt = sqrt.(max.(participation, 0.0))
    X_sqrt = reshape(x_sqrt, n, 1)
    sigma_est = std(price_changes[1:n])
    XtX_s = (X_sqrt'X_sqrt)[1,1]
    phi_hat = XtX_s > 0 ? (X_sqrt'price_changes[1:n])[1] / XtX_s : 0.0
    phi_hat = phi_hat / (sigma_est + 1e-10)

    y = price_changes[1:n]; ss_tot = sum((y .- mean(y)).^2) + 1e-15
    r2_lin  = 1 - sum((y .- eta_lin .* trade_sizes[1:n]).^2) / ss_tot
    r2_sqrt = 1 - sum((y .- phi_hat * sigma_est .* x_sqrt).^2) / ss_tot

    (eta_linear=eta_lin, phi_sqrt=phi_hat, sigma=sigma_est,
     r2_linear=r2_lin, r2_sqrt=r2_sqrt,
     preferred_model=r2_sqrt > r2_lin ? :sqrt : :linear)
end

"""
    impact_decay_empirical(prices, trade_times, trade_sizes, decay_window=60)
       -> NamedTuple

Estimate empirical impact decay: how quickly does price impact revert?
"""
function impact_decay_empirical(prices::Vector{Float64},
                                  trade_times::Vector{Int},
                                  trade_sizes::Vector{Float64},
                                  decay_window::Int=60)
    n_trades = length(trade_times)
    n_prices = length(prices)
    decay_profiles = zeros(decay_window)
    n_events = 0

    for (idx, t0) in enumerate(trade_times)
        t0 + decay_window > n_prices && continue
        direction = trade_sizes[idx] > 0 ? 1.0 : -1.0
        for tau in 1:decay_window
            ret = log(prices[t0+tau] / prices[t0])
            decay_profiles[tau] += direction * ret
        end
        n_events += 1
    end

    n_events > 0 && (decay_profiles ./= n_events)

    # Fit exponential decay: impact(tau) = impact_0 * exp(-tau/theta)
    t_vals = Float64.(1:decay_window)
    log_decay = log.(max.(abs.(decay_profiles), 1e-10))
    # OLS fit of log(impact) on t
    X = hcat(ones(decay_window), t_vals)
    coef = (X'X + 1e-8*I) \ (X'log_decay)
    theta_decay = coef[2] != 0 ? -1.0 / coef[2] : Inf  # decay time constant

    (avg_decay_profile=decay_profiles, decay_time_constant=theta_decay,
     n_events=n_events)
end

"""
    intraday_impact_pattern(prices, volumes, n_buckets) -> NamedTuple

Estimate intraday pattern of market impact from tick data.
Returns impact scaling by time bucket (U-shaped is typical).
"""
function intraday_impact_pattern(prices::Vector{Float64},
                                   volumes::Vector{Float64},
                                   n_buckets::Int=24)
    n = length(prices); bucket_size = max(n ÷ n_buckets, 1)
    impact_b = zeros(n_buckets); vol_b = zeros(n_buckets)
    for b in 1:n_buckets
        t1 = (b-1)*bucket_size+1; t2 = min(b*bucket_size, n)
        t2 <= t1 && continue
        seg_p = prices[t1:t2]; seg_v = volumes[t1:t2]
        length(seg_p) < 2 && continue
        rets = diff(log.(max.(seg_p, 1e-10)))
        vol_b[b] = mean(seg_v)
        impact_b[b] = mean(abs.(rets)) / (mean(seg_v)+1e-10) * 1e6
    end
    v_s = sum(vol_b); v_s > 0 && (vol_b ./= v_s)
    i_m = mean(impact_b[impact_b .> 0])
    i_m > 0 && (impact_b ./= i_m)
    (impact_by_bucket=impact_b, volume_fraction=vol_b, n_buckets=n_buckets)
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 15 – Optimal Liquidation Under Uncertainty
# ─────────────────────────────────────────────────────────────────────────────

"""
    robust_optimal_liquidation(X0, T, n_steps, eta_mean, eta_std,
                                sigma, lam; n_mc)

Robust Almgren-Chriss: optimise trajectory under uncertainty about η (impact).
Uses Monte Carlo over η draws and maximises worst-case CVaR of execution cost.
Returns the trajectory that minimises expected cost under ambiguity.
"""
function robust_optimal_liquidation(X0::Float64, T::Float64, n_steps::Int,
                                     eta_mean::Float64, eta_std::Float64,
                                     sigma::Float64, lam::Float64;
                                     n_mc::Int=500)
    best_traj  = zeros(n_steps + 1)
    best_cost  = Inf
    # Search over a family of linear + sinh interpolated trajectories
    for tau_factor in [0.5, 1.0, 2.0, 4.0]
        traj = zeros(n_steps + 1); traj[1] = X0
        for i in 2:(n_steps+1)
            frac = (n_steps - (i-1)) / n_steps
            traj[i] = X0 * (sinh(tau_factor * frac) / (sinh(tau_factor) + 1e-8))
        end
        traj[end] = 0.0
        # evaluate expected cost across η samples
        costs = Float64[]
        for _ in 1:n_mc
            eta = max(0.0, eta_mean + eta_std * randn())
            cost = 0.0
            for i in 1:n_steps
                dx     = traj[i] - traj[i+1]
                rate   = dx / (T / n_steps)
                impact = eta * rate
                risk   = lam * sigma^2 * traj[i+1]^2 * (T / n_steps)
                cost  += impact * dx + risk
            end
            push!(costs, cost)
        end
        # CVaR at 90th percentile
        sorted_c = sort(costs)
        cvar_90  = mean(sorted_c[floor(Int, 0.9*n_mc):end])
        if cvar_90 < best_cost
            best_cost = cvar_90; best_traj = copy(traj)
        end
    end
    return (trajectory=best_traj, cvar_cost=best_cost)
end

"""
    participation_schedule(volume_profile, target_participation, total_quantity)

Generate a VWAP participation schedule: trade `target_participation` fraction
of expected volume in each bucket.
Returns per-bucket quantities (may need to be capped to remaining inventory).
"""
function participation_schedule(volume_profile::Vector{Float64},
                                  target_participation::Float64,
                                  total_quantity::Float64)
    frac_vol   = volume_profile ./ (sum(volume_profile) + 1e-8)
    scheduled  = frac_vol .* total_quantity ./ target_participation
    # cap cumulative to total
    cumq = cumsum(scheduled)
    for i in 1:length(scheduled)
        if cumq[i] > total_quantity
            scheduled[i] -= cumq[i] - total_quantity
            scheduled[i]  = max(0.0, scheduled[i])
            scheduled[i+1:end] .= 0.0
            break
        end
    end
    return scheduled
end

"""
    execution_shortfall_decompose(decision_price, arrival_price,
                                   fill_prices, quantities, benchmark)

Decompose implementation shortfall into:
- Timing cost (arrival vs decision)
- Market impact (fills vs arrival)
- Missed trade (unfilled residual)
- Spread cost (half-spread times fills)
"""
function execution_shortfall_decompose(decision_price::Float64,
                                        arrival_price::Float64,
                                        fill_prices::Vector{Float64},
                                        quantities::Vector{Float64},
                                        benchmark_price::Float64,
                                        half_spread::Float64)
    total_qty = sum(quantities)
    avg_fill  = total_qty > 0 ? dot(quantities, fill_prices) / total_qty : arrival_price
    timing    = (arrival_price - decision_price) * total_qty
    impact    = (avg_fill - arrival_price) * total_qty
    missed    = (benchmark_price - arrival_price) * (0.0)   # assume fully filled
    spread_c  = half_spread * total_qty
    total_is  = timing + impact + missed + spread_c
    return (total_is=total_is, timing_cost=timing,
            impact_cost=impact, spread_cost=spread_c)
end

"""
    intraday_liquidity_forecast(historical_volumes, day_of_week, hour_of_day)

Forecast intraday liquidity using seasonal decomposition:
base_vol × dow_factor × hod_factor, estimated from historical data.
"""
function intraday_liquidity_forecast(historical_volumes::Matrix{Float64},
                                      day_of_week::Int, hour_of_day::Int)
    # historical_volumes: n_days × n_hours
    n_days, n_hours = size(historical_volumes)
    total_vol   = sum(historical_volumes, dims=2)[:]
    base_vol    = mean(total_vol)
    dow_factor  = mean(total_vol[mod1.((1:n_days) .- (day_of_week - 1), 7) .== 1]) / base_vol
    hourly_avg  = vec(mean(historical_volumes, dims=1))
    daily_total = mean(hourly_avg) * n_hours
    hod_factor  = hourly_avg[hour_of_day] * n_hours / (daily_total + 1e-8)
    return base_vol * dow_factor * hod_factor
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 16 – Transaction Cost Analytics Summary
# ─────────────────────────────────────────────────────────────────────────────

"""
    tca_summary(fills, quantities, arrival_price, benchmark_price,
                 half_spread, sigma, adv)

Comprehensive Transaction Cost Analysis report.
Returns named tuple with all standard TCA metrics.
"""
function tca_summary(fills::Vector{Float64}, quantities::Vector{Float64},
                      arrival_price::Float64, benchmark_price::Float64,
                      half_spread::Float64, sigma::Float64, adv::Float64)
    total_qty = sum(quantities)
    avg_fill  = total_qty > 0 ? dot(quantities, fills) / total_qty : arrival_price
    is        = (avg_fill - arrival_price) / arrival_price * 1e4   # bps
    spread_c  = half_spread / arrival_price * 1e4                   # bps
    mkt_move  = (benchmark_price - arrival_price) / arrival_price * 1e4
    # Normalised participation rate
    part_rate = total_qty / (adv + 1e-8)
    # Square-root model estimated impact
    sqrt_impact = sigma * sqrt(part_rate) * 1e4
    println("=" ^ 55)
    println("TCA Summary")
    println("=" ^ 55)
    @printf("  Arrival price:      %.6f\n", arrival_price)
    @printf("  Avg fill price:     %.6f\n", avg_fill)
    @printf("  Implementation IS:  %+.2f bps\n", is)
    @printf("  Spread cost:        %.2f bps\n", spread_c)
    @printf("  Market move:        %+.2f bps\n", mkt_move)
    @printf("  Participation rate: %.4f\n", part_rate)
    @printf("  Est. sqrt impact:   %.2f bps\n", sqrt_impact)
    println("=" ^ 55)
    return (is_bps=is, spread_bps=spread_c, market_move_bps=mkt_move,
            participation_rate=part_rate, estimated_sqrt_impact=sqrt_impact)
end

"""
    market_impact_budget(total_quantity, risk_budget_bps, sigma, phi, adv)

Given a total order and impact budget in bps, compute the maximum feasible
participation rate and recommended execution horizon.
phi: square-root impact coefficient.
"""
function market_impact_budget(total_quantity::Float64,
                               risk_budget_bps::Float64,
                               sigma::Float64, phi::Float64,
                               adv::Float64)
    # phi * sigma * sqrt(Q/V) <= budget_bps / 1e4
    max_part  = (risk_budget_bps / (1e4 * phi * sigma + 1e-8))^2
    rec_horizon_days = (total_quantity / adv) / max_part
    return (max_participation=max_part, horizon_days=rec_horizon_days)
end


"""
    brokerage_cost_model(quantity, price, commission_per_share,
                          exchange_fee_bps, sec_fee_bps)

Model total brokerage costs: commission + exchange fee + SEC fee.
Returns (total_cost, breakdown).
"""
function brokerage_cost_model(quantity::Float64, price::Float64;
                               commission_per_share::Float64=0.005,
                               exchange_fee_bps::Float64=0.3,
                               sec_fee_bps::Float64=0.00229)
    notional   = quantity * price
    commission = quantity * commission_per_share
    exch_fee   = notional * exchange_fee_bps / 1e4
    sec_fee    = notional * sec_fee_bps / 1e4
    total      = commission + exch_fee + sec_fee
    return (total=total, commission=commission,
            exchange_fee=exch_fee, sec_fee=sec_fee,
            total_bps=total / (notional + 1e-8) * 1e4)
end

"""
    expected_shortfall_from_impact(impact_model_cost, sigma, T, confidence)

Combine deterministic impact cost with stochastic market-move risk
to estimate expected shortfall at `confidence` level for an execution.
Uses normal approximation for market uncertainty.
"""
function expected_shortfall_from_impact(impact_cost::Float64,
                                         sigma::Float64, T::Float64,
                                         confidence::Float64=0.95)
    vol_T    = sigma * sqrt(T)
    z_alpha  = -qnorm_mi(1 - confidence)
    # standard normal CVaR = phi(z)/alpha
    cvar_z   = exp(-0.5 * z_alpha^2) / (sqrt(2π) * (1 - confidence))
    market_es = vol_T * cvar_z
    return impact_cost + market_es
end

function qnorm_mi(p::Float64)
    p = clamp(p, 1e-10, 1-1e-10)
    t = p < 0.5 ? sqrt(-2log(p)) : sqrt(-2log(1-p))
    c0=2.515517; c1=0.802853; c2=0.010328
    d1=1.432788; d2=0.189269; d3=0.001308
    x = t - (c0+c1*t+c2*t^2)/(1+d1*t+d2*t^2+d3*t^3)
    return p < 0.5 ? -x : x
end

end  # module MarketImpact
