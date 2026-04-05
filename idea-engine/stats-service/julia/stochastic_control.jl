# =============================================================================
# stochastic_control.jl — Stochastic Optimal Control for Trading
# =============================================================================
# Provides:
#   - HJBSolver               Hamilton-Jacobi-Bellman via finite differences
#   - AlmgrenChriss           Optimal liquidation PDE (price impact)
#   - MeanVarianceDP          Mean-variance dynamic programming (discrete)
#   - CARAUtility             CARA utility maximisation with GARCH volatility
#   - MertonProblem           Merton's continuous-time portfolio (closed + num)
#   - ValueIteration          General value function iteration
#   - PolicyExtraction        Optimal policy from value function
#   - run_stochastic_control  Top-level driver + JSON export
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, Random, JSON3
# =============================================================================

module StochasticControl

using Statistics
using LinearAlgebra
using Random
using JSON3

export HJBSolver, AlmgrenChriss, MeanVarianceDP, CARAUtility
export MertonProblem, ValueIteration, PolicyExtraction
export run_stochastic_control

# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Tridiagonal matrix solve (Thomas algorithm)."""
function _tridiag_solve(a::Vector{Float64}, b::Vector{Float64},
                         c::Vector{Float64}, d::Vector{Float64})::Vector{Float64}
    n = length(d)
    cp = zeros(n); dp = zeros(n); x = zeros(n)
    cp[1] = c[1] / b[1]; dp[1] = d[1] / b[1]
    for i in 2:n
        m = b[i] - a[i] * cp[i-1]
        abs(m) < 1e-14 && (m = 1e-14)
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i-1]) / m
    end
    x[n] = dp[n]
    for i in (n-1):-1:1
        x[i] = dp[i] - cp[i] * x[i+1]
    end
    return x
end

"""Sigmoid function."""
_sigmoid(x) = 1.0 / (1.0 + exp(-x))

"""Soft-threshold operator for L1 proximal."""
_soft(x, λ) = sign(x) * max(abs(x) - λ, 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Hamilton-Jacobi-Bellman Solver via Finite Differences
# ─────────────────────────────────────────────────────────────────────────────

"""
    HJBSolver(mu, sigma, r, gamma; x_min, x_max, t_max, nx, nt) → NamedTuple

Solve the Hamilton-Jacobi-Bellman PDE for a portfolio optimization problem
with CARA utility U(x) = -exp(-γx) / γ.

PDE: ∂V/∂t + max_π { (r + π(μ-r)) ∂V/∂x + ½π²σ²x² ∂²V/∂x² } = 0
with terminal condition V(x, T) = U(x).

# Arguments
- `mu`    : expected return of risky asset
- `sigma` : volatility of risky asset
- `r`     : risk-free rate
- `gamma` : risk aversion coefficient
- `x_min, x_max` : wealth grid bounds
- `t_max` : investment horizon
- `nx`    : spatial grid points (default 100)
- `nt`    : time grid points (default 200)

# Returns
NamedTuple: (V, x_grid, t_grid, policy, merton_fraction)
"""
function HJBSolver(mu::Float64, sigma::Float64, r::Float64, gamma::Float64;
                   x_min::Float64=0.1, x_max::Float64=10.0,
                   t_max::Float64=1.0, nx::Int=100, nt::Int=200)
    dx = (x_max - x_min) / (nx - 1)
    dt = t_max / nt
    x_grid = range(x_min, x_max, length=nx) |> collect
    t_grid = range(0.0, t_max, length=nt+1) |> collect

    # Terminal condition: CARA utility
    V = [-exp(-gamma * x) / gamma for x in x_grid]

    # Merton closed-form fraction (used as initial policy guess)
    merton_pi = (mu - r) / (gamma * sigma^2)

    # Store policy (optimal fraction at each (x, t))
    policy = fill(merton_pi, nx, nt+1)

    # Backward in time (Crank-Nicolson scheme)
    for k in nt:-1:1
        # For each spatial point, find optimal π via pointwise maximisation
        pi_opt = zeros(nx)
        for i in 2:(nx-1)
            xi = x_grid[i]
            Vx = (V[i+1] - V[i-1]) / (2dx)
            Vxx = (V[i+1] - 2V[i] + V[i-1]) / dx^2
            # FOC: (mu-r)*xi*Vx + sigma^2*xi^2*pi*Vxx = 0
            # => pi* = -(mu-r)*xi*Vx / (sigma^2*xi^2*Vxx)
            denom = sigma^2 * xi^2 * Vxx
            if abs(denom) > 1e-12
                pi_opt[i] = -(mu - r) * xi * Vx / denom
            else
                pi_opt[i] = merton_pi
            end
            pi_opt[i] = clamp(pi_opt[i], -5.0, 5.0)
        end
        pi_opt[1] = pi_opt[2]; pi_opt[nx] = pi_opt[nx-1]
        policy[:, k] = pi_opt

        # Build tridiagonal system for implicit time step
        a = zeros(nx); b = zeros(nx); c = zeros(nx); rhs = zeros(nx)
        for i in 2:(nx-1)
            xi = x_grid[i]
            pi = pi_opt[i]
            drift  = (r + pi * (mu - r)) * xi
            diffus = 0.5 * (pi * sigma * xi)^2

            # Crank-Nicolson coefficients
            alpha = diffus / dx^2 - drift / (2dx)
            beta  = -2diffus / dx^2 - 0.0   # no discount term in value function
            gam_c = diffus / dx^2 + drift / (2dx)

            a[i] = -0.5 * dt * alpha
            b[i] = 1.0 - 0.5 * dt * beta
            c[i] = -0.5 * dt * gam_c
            rhs[i] = 0.5 * dt * alpha * V[i-1] +
                     (1.0 + 0.5 * dt * beta) * V[i] +
                     0.5 * dt * gam_c * V[i+1]
        end
        # Boundary conditions
        b[1] = 1.0; rhs[1] = V[1]
        b[nx] = 1.0; rhs[nx] = V[nx]
        a[1] = 0.0; c[1] = 0.0
        a[nx] = 0.0; c[nx] = 0.0

        V_new = _tridiag_solve(a, b, c, rhs)
        V = V_new
    end

    return (V=V, x_grid=x_grid, t_grid=t_grid, policy=policy,
            merton_fraction=merton_pi, nx=nx, nt=nt)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Almgren-Chriss Optimal Liquidation
# ─────────────────────────────────────────────────────────────────────────────

"""
    AlmgrenChriss(X0, T, sigma, eta, gamma_ac; n_steps, risk_aversion) → NamedTuple

Solve the Almgren-Chriss optimal liquidation problem.

Minimise E[C] + λ·Var[C] subject to selling X0 shares over [0, T].
Price impact: temporary impact η·v, permanent impact γ·v.

Closed-form solution:
x*(t) = X0 · sinh(κ(T-t)) / sinh(κT)
where κ = sqrt(λσ²/η)

# Arguments
- `X0`          : initial position (shares)
- `T`           : liquidation horizon (days)
- `sigma`       : daily volatility (price)
- `eta`         : temporary impact coefficient
- `gamma_ac`    : permanent impact coefficient
- `n_steps`     : number of trading intervals (default 10)
- `risk_aversion`: λ risk/variance tradeoff (default 1e-6)

# Returns
NamedTuple: (trajectory, trade_schedule, expected_cost, variance_cost,
             efficient_frontier, execution_shortfall)
"""
function AlmgrenChriss(X0::Float64, T::Float64, sigma::Float64,
                        eta::Float64, gamma_ac::Float64;
                        n_steps::Int=10,
                        risk_aversion::Float64=1e-6)
    n = n_steps
    dt = T / n
    tau_grid = range(0.0, T, length=n+1) |> collect

    # Almgren-Chriss parameter
    kappa2 = risk_aversion * sigma^2 / eta
    kappa  = sqrt(max(kappa2, 1e-12))

    # Optimal trajectory: x*(t) = X0 * sinh(κ(T-t)) / sinh(κT)
    sinh_kT = sinh(kappa * T)
    abs(sinh_kT) < 1e-10 && (sinh_kT = 1e-10)

    trajectory = [X0 * sinh(kappa * (T - t)) / sinh_kT for t in tau_grid]

    # Trade schedule: n_j = x(t_{j-1}) - x(t_j)
    trades = diff(-trajectory)  # positive = selling

    # Expected cost
    E_cost = 0.5 * gamma_ac * X0^2 +
             eta * X0^2 * (kappa / tanh(kappa * T/2)) / T *
             (1.0 + dt / T)

    # Variance of cost
    Var_cost = 0.5 * sigma^2 * X0^2 *
               (T - (1/kappa) * tanh(kappa * T/2)) / (kappa^2 * tanh(kappa*T/2)^2 * T)

    # Efficient frontier (vary risk_aversion from 1e-8 to 1e-4)
    lambdas = exp.(range(log(1e-8), log(1e-4), length=20))
    frontier_E = Float64[]
    frontier_V = Float64[]
    for λ in lambdas
        κ2 = λ * sigma^2 / eta
        κ  = sqrt(max(κ2, 1e-12))
        sh = sinh(κ * T); sh < 1e-10 && (sh = 1e-10)
        E = 0.5 * gamma_ac * X0^2 + eta * X0^2 * κ / tanh(κ * T) / T
        V = 0.5 * sigma^2 * X0^2 * T / 3.0  # simplified
        push!(frontier_E, E); push!(frontier_V, V)
    end

    # Execution shortfall vs TWAP
    twap_trades = fill(X0 / n, n)
    twap_cost   = eta * sum(twap_trades .^ 2) / dt + gamma_ac * X0^2 / 2
    shortfall   = E_cost - twap_cost

    return (trajectory=trajectory, trade_schedule=trades,
            expected_cost=E_cost, variance_cost=Var_cost,
            efficient_frontier=(E=frontier_E, V=frontier_V),
            kappa=kappa, execution_shortfall=shortfall,
            n_steps=n_steps, tau_grid=tau_grid)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Mean-Variance Dynamic Programming
# ─────────────────────────────────────────────────────────────────────────────

"""
    MeanVarianceDP(mu, Sigma, rf; n_periods, n_wealth, lambda_risk) → NamedTuple

Solve the mean-variance dynamic programming problem over n_periods.
At each period, choose portfolio weights to maximise:
E[W_{t+1}] - λ · Var[W_{t+1}]

Uses backward induction on a discretised wealth grid.

# Arguments
- `mu`        : d-vector of expected returns
- `Sigma`     : d×d covariance matrix of returns
- `rf`        : risk-free rate per period
- `n_periods` : investment horizon (default 4)
- `n_wealth`  : wealth grid points (default 50)
- `lambda_risk`: risk aversion parameter (default 1.0)

# Returns
NamedTuple: (value_function, policy_weights, efficient_weights, sharpe)
"""
function MeanVarianceDP(mu::Vector{Float64}, Sigma::Matrix{Float64},
                         rf::Float64=0.0; n_periods::Int=4,
                         n_wealth::Int=50, lambda_risk::Float64=1.0)
    d = length(mu)
    W_min = 0.5; W_max = 3.0
    W_grid = range(W_min, W_max, length=n_wealth) |> collect
    dW = W_grid[2] - W_grid[1]

    # Value function V(w, t) and policy π*(w, t)
    V = zeros(n_wealth, n_periods+1)
    policy = zeros(n_wealth, n_periods, d)

    # Terminal condition: V(w, T) = w (linear utility at terminal)
    V[:, n_periods+1] = W_grid

    # Solve analytically for unconstrained MV policy at each period
    # π* = (2λ Σ)^{-1} (μ - rf·1)
    ones_d = ones(d)
    excess = mu .- rf
    Sigma_reg = Sigma + 1e-6 * I
    pi_myopic = inv(2.0 * lambda_risk * Sigma_reg) * excess

    # Project onto simplex constraints [0,1] with sum ≤ 1
    pi_clamped = clamp.(pi_myopic, 0.0, 1.0)
    pi_sum = sum(pi_clamped)
    pi_sum > 1.0 && (pi_clamped ./= pi_sum)

    # Expected return and variance of myopic policy
    mu_port   = dot(pi_clamped, mu) + (1.0 - sum(pi_clamped)) * rf
    var_port  = dot(pi_clamped, Sigma * pi_clamped)
    sharpe    = (mu_port - rf) / sqrt(max(var_port, 1e-10))

    # Backward induction
    for t in n_periods:-1:1
        for (iw, w) in enumerate(W_grid)
            # Discretise next-period wealth distribution
            best_val = -Inf
            best_pi  = pi_clamped

            # Search over a set of candidate allocations
            n_search = 10
            for s in 1:n_search
                # Perturb allocation
                pi_try = pi_clamped .* (0.5 + rand())
                pi_try = clamp.(pi_try, 0.0, 1.0)
                ps = sum(pi_try); ps > 1.0 && (pi_try ./= ps)

                r_port = dot(pi_try, mu) + (1.0 - sum(pi_try)) * rf
                v_port = dot(pi_try, Sigma * pi_try)

                # Approximate next-period value: E[V(W', t+1)]
                E_W = w * (1.0 + r_port)
                # Find nearest grid point
                idx_next = clamp(round(Int, (E_W - W_min)/dW + 1), 1, n_wealth)
                EV = V[idx_next, t+1]

                # Objective: E[V] - λ·w²·var_port (risk penalty)
                obj = EV - lambda_risk * w^2 * v_port
                if obj > best_val
                    best_val = obj
                    best_pi = copy(pi_try)
                end
            end
            V[iw, t] = best_val
            policy[iw, t, :] = best_pi
        end
    end

    # Efficient frontier: vary lambda
    lambdas = exp.(range(-2, 3, length=30))
    ef_return = Float64[]; ef_vol = Float64[]
    for λ in lambdas
        pi_ef = inv(2.0 * λ * Sigma_reg) * excess
        pi_ef = clamp.(pi_ef, 0.0, 1.0)
        ps = sum(pi_ef); ps > 1.0 && (pi_ef ./= ps)
        push!(ef_return, dot(pi_ef, mu) + (1.0 - sum(pi_ef)) * rf)
        push!(ef_vol, sqrt(dot(pi_ef, Sigma * pi_ef)))
    end

    return (value_function=V, policy_weights=policy,
            myopic_weights=pi_clamped, sharpe=sharpe,
            efficient_frontier=(returns=ef_return, vols=ef_vol),
            mu_port=mu_port, var_port=var_port)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. CARA Utility with GARCH Volatility
# ─────────────────────────────────────────────────────────────────────────────

"""
    CARAUtility(returns, gamma_cara; n_periods, garch_params) → NamedTuple

CARA utility maximisation with GARCH(1,1) volatility forecasting.

U(c) = -exp(-γ·c) / γ

# Arguments
- `returns`      : vector of historical returns for GARCH fitting
- `gamma_cara`   : absolute risk aversion coefficient
- `n_periods`    : forecast horizon
- `garch_params` : optional (omega, alpha, beta) tuple; fit if not provided

# Returns
NamedTuple: (optimal_allocation, utility, vol_forecast, certainty_equivalent)
"""
function CARAUtility(returns::Vector{Float64}, gamma_cara::Float64;
                     n_periods::Int=10, garch_params::Union{Nothing,Tuple}=nothing)
    n = length(returns)

    # ── GARCH(1,1) Estimation ──────────────────────────────────────────────
    omega, alpha_g, beta_g = if isnothing(garch_params)
        _fit_garch(returns)
    else
        garch_params
    end

    # GARCH variance forecast
    h0 = var(returns)
    h_forecast = _garch_forecast(omega, alpha_g, beta_g, h0, returns[end], n_periods)

    # ── CARA Optimal Allocation ────────────────────────────────────────────
    # Closed form for normally distributed returns with GARCH vol:
    # π* = (μ - r) / (γ · σ²)
    mu_ret = mean(returns)
    rf     = 0.0
    allocations = Float64[]

    for t in 1:n_periods
        sigma2_t = h_forecast[t]
        pi_t = (mu_ret - rf) / (gamma_cara * sigma2_t)
        pi_t = clamp(pi_t, -2.0, 2.0)  # leverage constraint
        push!(allocations, pi_t)
    end

    # Certainty equivalent: the fixed consumption c such that U(c) = E[U(W)]
    # For CARA with normal returns: CE = μ_π - γ/2 · σ²_π
    avg_alloc = mean(allocations)
    mu_port   = avg_alloc * mu_ret
    var_port  = avg_alloc^2 * mean(h_forecast)
    ce        = mu_port - gamma_cara / 2.0 * var_port

    # Expected utility
    E_utility = -exp(-gamma_cara * ce) / gamma_cara

    # Risk premium over cash
    risk_premium = mu_port - rf

    return (optimal_allocation=allocations,
            certainty_equivalent=ce,
            expected_utility=E_utility,
            vol_forecast=sqrt.(h_forecast),
            garch_params=(omega=omega, alpha=alpha_g, beta=beta_g),
            risk_premium=risk_premium,
            gamma=gamma_cara)
end

"""Fit GARCH(1,1) via quasi-MLE (Berndt-Hall-Hall-Hausman gradient search)."""
function _fit_garch(returns::Vector{Float64})::Tuple{Float64,Float64,Float64}
    n = length(returns)
    mu = mean(returns)
    eps = returns .- mu

    # Initial parameter guess
    omega0 = var(returns) * 0.1
    alpha0 = 0.1
    beta0  = 0.8

    best_ll = -Inf
    best_params = (omega0, alpha0, beta0)

    # Grid search for initialisation
    for ω in [1e-6, 1e-5, 1e-4], α in [0.05, 0.10, 0.15], β in [0.75, 0.85, 0.90]
        α + β >= 1.0 && continue
        ll = _garch_loglik(eps, ω, α, β)
        if ll > best_ll
            best_ll = ll
            best_params = (ω, α, β)
        end
    end

    # Local gradient ascent
    ω, α, β = best_params
    step = 1e-5
    for _ in 1:500
        ll0 = _garch_loglik(eps, ω, α, β)
        g_ω = (_garch_loglik(eps, ω+step, α, β) - ll0) / step
        g_α = (_garch_loglik(eps, ω, α+step, β) - ll0) / step
        g_β = (_garch_loglik(eps, ω, α, β+step) - ll0) / step
        lr = 1e-8
        ω += lr * g_ω; α += lr * g_α; β += lr * g_β
        # Project onto stationary region
        ω = max(ω, 1e-8)
        α = max(α, 0.001)
        β = max(β, 0.001)
        if α + β >= 0.999
            total = α + β
            α /= total / 0.999
            β /= total / 0.999
        end
    end
    return (ω, α, β)
end

"""GARCH(1,1) log-likelihood."""
function _garch_loglik(eps::Vector{Float64}, omega::Float64,
                        alpha::Float64, beta::Float64)::Float64
    n = length(eps)
    h = var(eps)
    ll = 0.0
    for t in 1:n
        h = omega + alpha * eps[max(t-1,1)]^2 + beta * h
        h = max(h, 1e-10)
        ll += -0.5 * (log(2π) + log(h) + eps[t]^2 / h)
    end
    return ll
end

"""GARCH(1,1) multi-step variance forecast."""
function _garch_forecast(omega::Float64, alpha::Float64, beta::Float64,
                          h0::Float64, eps_last::Float64, n_steps::Int)::Vector{Float64}
    sigma_bar2 = omega / (1.0 - alpha - beta)  # unconditional variance
    h_next = omega + alpha * eps_last^2 + beta * h0
    forecasts = zeros(n_steps)
    forecasts[1] = h_next
    for t in 2:n_steps
        # Multi-step: E[h_{t+k}] = σ̄² + (α+β)^k (h_{t+1} - σ̄²)
        forecasts[t] = sigma_bar2 + (alpha + beta)^(t-1) * (h_next - sigma_bar2)
        forecasts[t] = max(forecasts[t], 1e-10)
    end
    return forecasts
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Merton's Continuous-Time Portfolio Problem
# ─────────────────────────────────────────────────────────────────────────────

"""
    MertonProblem(mu, sigma, r, gamma_m; T, W0, rho, closed_form) → NamedTuple

Solve Merton's continuous-time portfolio optimisation problem.

Maximise E[∫₀ᵀ e^{-ρt} U(c_t) dt + e^{-ρT} B(W_T)]
with U(c) = c^{1-γ} / (1-γ) (CRRA utility).

Closed-form solution for CRRA utility:
π* = (μ - r) / (γ σ²)
c* = ρ/γ · W (approximately for long horizon)

# Arguments
- `mu`         : expected return of risky asset
- `sigma`      : volatility
- `r`          : risk-free rate
- `gamma_m`    : relative risk aversion (CRRA parameter)
- `T`          : investment horizon (years)
- `W0`         : initial wealth
- `rho`        : subjective discount rate (default 0.05)
- `closed_form`: if true, return analytical solution (default true)

# Returns
NamedTuple: (pi_star, c_star, V0, growth_rate, simulation)
"""
function MertonProblem(mu::Float64, sigma::Float64, r::Float64,
                        gamma_m::Float64; T::Float64=1.0, W0::Float64=1.0,
                        rho::Float64=0.05, closed_form::Bool=true,
                        n_paths::Int=1000, rng::AbstractRNG=Random.default_rng())
    # ── Closed-Form Solution ───────────────────────────────────────────────
    pi_star = (mu - r) / (gamma_m * sigma^2)    # optimal risky fraction
    pi_star = clamp(pi_star, -3.0, 3.0)

    # Optimal consumption rate
    nu = (rho - (1.0-gamma_m)*(r + 0.5*(mu-r)^2/(gamma_m*sigma^2))) / gamma_m
    c_star_rate = nu  # c* = nu * W

    # Value function at W0
    # V(W,t) = f(t) W^{1-γ} / (1-γ)
    # f(t) = [ν exp(ν(T-t)) + exp(-ν t)·something...]  (complex in general)
    # For infinite horizon / simplified version:
    if abs(c_star_rate) > 1e-10
        V0 = W0^(1.0 - gamma_m) / ((1.0 - gamma_m) * c_star_rate)
    else
        V0 = W0^(1.0 - gamma_m) / (1.0 - gamma_m + 1e-10)
    end

    # Expected portfolio return under optimal policy
    r_port = r + pi_star * (mu - r)
    var_port = (pi_star * sigma)^2
    # Long-run growth rate of wealth (log-utility is γ=1 limiting case)
    g = r_port - 0.5 * var_port

    # ── Numerical Simulation ───────────────────────────────────────────────
    n_steps = 252
    dt = T / n_steps
    W_paths = zeros(n_paths, n_steps+1)
    W_paths[:, 1] .= W0

    dW_drift = (r_port - c_star_rate) * dt
    dW_vol   = pi_star * sigma * sqrt(dt)

    for t in 2:(n_steps+1)
        z = randn(rng, n_paths)
        @. W_paths[:, t] = W_paths[:, t-1] * exp(dW_drift - 0.5*dW_vol^2 + dW_vol*z)
        W_paths[:, t] = max.(W_paths[:, t], 0.0)
    end

    W_terminal = W_paths[:, end]
    E_terminal = mean(W_terminal)
    P_ruin = mean(W_terminal .< 0.1 * W0)

    # Certainty equivalent terminal wealth (CRRA)
    if gamma_m != 1.0
        CE = (mean(W_terminal .^ (1.0 - gamma_m)) / ((1.0 - gamma_m) * max(V0, 1e-10)))^(1.0/(1.0-gamma_m))
    else
        CE = exp(mean(log.(max.(W_terminal, 1e-10))))
    end

    return (pi_star=pi_star, c_star_rate=c_star_rate, V0=V0,
            growth_rate=g, r_portfolio=r_port,
            E_terminal_wealth=E_terminal, P_ruin=P_ruin,
            certainty_equivalent=CE, W_paths=W_paths)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Value Function Iteration (General DP)
# ─────────────────────────────────────────────────────────────────────────────

"""
    ValueIteration(reward_fn, transition_fn, state_grid, action_grid;
                   beta, max_iter, tol) → NamedTuple

General value function iteration for discounted infinite-horizon DP.

V(s) = max_a { R(s,a) + β ∑_{s'} P(s'|s,a) V(s') }

# Arguments
- `reward_fn`    : (s_idx, a_idx) → Float64 reward
- `transition_fn`: (s_idx, a_idx) → Vector of (s'_idx, prob) pairs
- `state_grid`   : vector of state values
- `action_grid`  : vector of action values
- `beta`         : discount factor (default 0.99)
- `max_iter`     : maximum iterations (default 1000)
- `tol`          : convergence tolerance (default 1e-8)

# Returns
NamedTuple: (V, policy, n_iter, converged)
"""
function ValueIteration(reward_fn::Function, transition_fn::Function,
                         state_grid::Vector{Float64}, action_grid::Vector{Float64};
                         beta::Float64=0.99, max_iter::Int=1000, tol::Float64=1e-8)
    ns = length(state_grid)
    na = length(action_grid)

    V = zeros(ns)
    policy = ones(Int, ns)
    n_iter = 0
    converged = false

    for iter in 1:max_iter
        V_new = zeros(ns)
        for si in 1:ns
            best_val = -Inf
            best_ai  = 1
            for ai in 1:na
                r = reward_fn(si, ai)
                # Expected value over transitions
                EV = 0.0
                for (sp_idx, prob) in transition_fn(si, ai)
                    1 ≤ sp_idx ≤ ns && (EV += prob * V[sp_idx])
                end
                val = r + beta * EV
                if val > best_val
                    best_val = val
                    best_ai = ai
                end
            end
            V_new[si] = best_val
            policy[si] = best_ai
        end
        delta = maximum(abs.(V_new - V))
        V = V_new
        n_iter = iter
        if delta < tol
            converged = true
            break
        end
    end

    return (V=V, policy=policy, policy_values=action_grid[policy],
            n_iter=n_iter, converged=converged)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Policy Extraction from Value Function
# ─────────────────────────────────────────────────────────────────────────────

"""
    PolicyExtraction(V, state_grid, action_grid, reward_fn, transition_fn;
                     beta) → NamedTuple

Extract optimal policy from a converged value function.
Performs one greedy step: π*(s) = argmax_a Q(s,a).

# Arguments
- `V`            : converged value function (ns-vector)
- `state_grid`   : state values
- `action_grid`  : action values
- `reward_fn`    : reward function (si, ai) → Float64
- `transition_fn`: transition function
- `beta`         : discount factor

# Returns
NamedTuple: (policy_indices, policy_values, Q_values)
"""
function PolicyExtraction(V::Vector{Float64}, state_grid::Vector{Float64},
                           action_grid::Vector{Float64}, reward_fn::Function,
                           transition_fn::Function; beta::Float64=0.99)
    ns = length(state_grid)
    na = length(action_grid)
    Q = zeros(ns, na)
    policy_idx = ones(Int, ns)

    for si in 1:ns
        best_val = -Inf
        for ai in 1:na
            r = reward_fn(si, ai)
            EV = 0.0
            for (sp_idx, prob) in transition_fn(si, ai)
                1 ≤ sp_idx ≤ ns && (EV += prob * V[sp_idx])
            end
            Q[si, ai] = r + beta * EV
            if Q[si, ai] > best_val
                best_val = Q[si, ai]
                policy_idx[si] = ai
            end
        end
    end

    return (policy_indices=policy_idx, policy_values=action_grid[policy_idx],
            Q_values=Q)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Transaction-Cost Penalised DP
# ─────────────────────────────────────────────────────────────────────────────

"""
    TransactionCostDP(mu, sigma, r, gamma_tc; lambda_tc, n_periods, n_grid) → NamedTuple

Dynamic programming with transaction cost penalty.
At each period, choose π to maximise:
E[W] - λ·Var[W] - c_tc · |π - π_{t-1}|

# Arguments
- `mu`       : expected return
- `sigma`    : volatility
- `r`        : risk-free rate
- `gamma_tc` : risk aversion
- `lambda_tc`: transaction cost per unit traded (default 0.001)
- `n_periods`: investment horizon (default 5)
- `n_grid`   : action grid size (default 20)

# Returns
NamedTuple: (V, policy, rebalancing_schedule, total_cost)
"""
function TransactionCostDP(mu::Float64, sigma::Float64, r::Float64,
                            gamma_tc::Float64; lambda_tc::Float64=0.001,
                            n_periods::Int=5, n_grid::Int=20)
    # State: (period, current_allocation), Action: new_allocation
    pi_grid = range(-1.0, 2.0, length=n_grid) |> collect  # allow some leverage
    n_s = n_grid

    # Terminal value function: V(π, T) = 0 (no bequest)
    V = zeros(n_s, n_periods+1)
    policy = zeros(Int, n_s, n_periods)

    # Backward induction
    for t in n_periods:-1:1
        for si in 1:n_s
            pi_cur = pi_grid[si]
            best_val = -Inf
            best_ai  = si
            for ai in 1:n_s
                pi_new = pi_grid[ai]
                # Expected portfolio return
                r_port = r + pi_new * (mu - r)
                v_port = (pi_new * sigma)^2
                # One-period utility (mean-variance)
                one_step = r_port - gamma_tc * v_port
                # Transaction cost
                tc = lambda_tc * abs(pi_new - pi_cur)
                # Continuation value (interpolate)
                next_val = V[ai, t+1]
                val = one_step - tc + next_val
                if val > best_val
                    best_val = val
                    best_ai = ai
                end
            end
            V[si, t] = best_val
            policy[si, t] = best_ai
        end
    end

    # Simulate policy from central initial position
    start_si = div(n_s, 2)
    schedule = Float64[pi_grid[start_si]]
    total_tc = 0.0
    si = start_si
    for t in 1:n_periods
        ai = policy[si, t]
        total_tc += lambda_tc * abs(pi_grid[ai] - pi_grid[si])
        si = ai
        push!(schedule, pi_grid[si])
    end

    return (V=V, policy=policy, pi_grid=pi_grid,
            rebalancing_schedule=schedule, total_cost=total_tc)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Top-Level Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_stochastic_control(returns; out_path) → Dict

Full stochastic control pipeline for a crypto trading account.

# Arguments
- `returns`  : n×d return matrix (or n-vector for single asset)
- `out_path` : optional JSON output path

# Returns
Dict with all control problem solutions.

# Example
```julia
using Random
rng = Random.MersenneTwister(42)
returns = randn(rng, 252) .* 0.03 .+ 0.0005   # daily BTC-like returns
results = run_stochastic_control(returns)
println("Merton π*: ", results["merton"]["pi_star"])
println("AC schedule: ", results["almgren_chriss"]["trade_schedule"])
```
"""
function run_stochastic_control(returns::Union{Vector{Float64}, Matrix{Float64}};
                                 out_path::Union{String,Nothing}=nothing)
    # Normalise to vector if univariate
    if isa(returns, Matrix)
        r_vec = mean(returns, dims=2)[:,1]
        r_mat = returns
    else
        r_vec = returns
        r_mat = reshape(returns, :, 1)
    end
    d = size(r_mat, 2)

    mu_r  = mean(r_vec) * 252.0    # annualise
    sig_r = std(r_vec)  * sqrt(252.0)
    rf    = 0.04  # 4% annual risk-free

    results = Dict{String, Any}()

    # ── HJB Solver ─────────────────────────────────────────────────────────
    @info "Solving HJB PDE..."
    hjb = HJBSolver(mu_r, sig_r, rf, 3.0; nx=80, nt=100, t_max=1.0)
    results["hjb"] = Dict(
        "merton_fraction" => hjb.merton_fraction,
        "terminal_V"      => hjb.V[end÷2],    # value at median wealth
        "policy_at_1"     => hjb.policy[end÷2, 1],
        "nx" => hjb.nx, "nt" => hjb.nt
    )

    # ── Almgren-Chriss ─────────────────────────────────────────────────────
    @info "Computing Almgren-Chriss liquidation schedule..."
    X0 = 1_000_000.0  # 1M position
    ac = AlmgrenChriss(X0, 5.0, sig_r * sqrt(X0/1e6),
                        1e-7, 1e-8; n_steps=20)
    results["almgren_chriss"] = Dict(
        "trade_schedule"    => ac.trade_schedule,
        "expected_cost"     => ac.expected_cost,
        "kappa"             => ac.kappa,
        "execution_shortfall" => ac.execution_shortfall,
        "n_steps"           => ac.n_steps
    )

    # ── Mean-Variance DP ───────────────────────────────────────────────────
    @info "Running mean-variance DP..."
    mu_vec  = [mean(r_mat[:,j]) * 252 for j in 1:d]
    sig_mat = cov(r_mat) * 252
    mvdp = MeanVarianceDP(mu_vec, sig_mat, rf; n_periods=4, lambda_risk=2.0)
    results["mv_dp"] = Dict(
        "myopic_weights" => mvdp.myopic_weights,
        "sharpe"         => mvdp.sharpe,
        "mu_port"        => mvdp.mu_port,
        "var_port"       => mvdp.var_port
    )

    # ── CARA + GARCH ───────────────────────────────────────────────────────
    @info "CARA utility with GARCH volatility..."
    cara = CARAUtility(r_vec, 5.0; n_periods=10)
    results["cara_garch"] = Dict(
        "optimal_allocation"    => cara.optimal_allocation,
        "certainty_equivalent"  => cara.certainty_equivalent,
        "expected_utility"      => cara.expected_utility,
        "vol_forecast_10d"      => cara.vol_forecast,
        "garch_alpha"           => cara.garch_params.alpha,
        "garch_beta"            => cara.garch_params.beta
    )

    # ── Merton Problem ─────────────────────────────────────────────────────
    @info "Solving Merton's portfolio problem..."
    rng = Random.default_rng()
    merton = MertonProblem(mu_r, sig_r, rf, 3.0; T=1.0, n_paths=500, rng=rng)
    results["merton"] = Dict(
        "pi_star"               => merton.pi_star,
        "c_star_rate"           => merton.c_star_rate,
        "growth_rate"           => merton.growth_rate,
        "E_terminal_wealth"     => merton.E_terminal_wealth,
        "P_ruin"                => merton.P_ruin,
        "certainty_equivalent"  => merton.certainty_equivalent
    )

    # ── Transaction-Cost DP ────────────────────────────────────────────────
    @info "Transaction cost DP..."
    tc_dp = TransactionCostDP(mu_r, sig_r, rf, 3.0; lambda_tc=0.002, n_periods=5)
    results["transaction_cost_dp"] = Dict(
        "rebalancing_schedule" => tc_dp.rebalancing_schedule,
        "total_cost"           => tc_dp.total_cost
    )

    # ── Summary ────────────────────────────────────────────────────────────
    results["summary"] = Dict(
        "mu_annualized"   => mu_r,
        "sigma_annualized"=> sig_r,
        "rf"              => rf,
        "optimal_fraction_merton" => merton.pi_star,
        "optimal_fraction_hjb"    => hjb.merton_fraction
    )

    if !isnothing(out_path)
        open(out_path, "w") do io
            JSON3.write(io, results)
        end
        @info "Results written to $out_path"
    end

    return results
end

end  # module StochasticControl
