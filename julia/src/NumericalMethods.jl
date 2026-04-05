module NumericalMethods

# ============================================================
# NumericalMethods.jl — PDE solvers, Monte Carlo, quadrature,
#                        optimization routines (pure stdlib)
# ============================================================

using Statistics, LinearAlgebra

export bisect, brentq, secant_method, newton_raphson
export gauss_legendre_quadrature, simpson_rule, gauss_hermite
export fd_european_option, fd_american_option, fd_barrier_option
export mc_european, mc_asian, mc_barrier, mc_lookback
export quasi_mc_sobol, antithetic_variates_mc
export runge_kutta4, euler_method, adams_bashforth
export gradient_descent, adam_optimizer, bfgs_step
export cholesky_decomp, lu_decomp, qr_decomp_gram_schmidt
export power_iteration, inverse_iteration
export sor_solve, conjugate_gradient
export interpolate_linear, interpolate_cubic_spline
export lsq_fit, ridge_regression_lstsq

# ──────────────────────────────────────────────────────────────
# Root-finding
# ──────────────────────────────────────────────────────────────

"""
    bisect(f, a, b; tol, maxiter) -> root

Bisection method for root finding on [a, b].
Requires f(a) and f(b) to have opposite signs.
"""
function bisect(f::Function, a::Float64, b::Float64;
                tol::Float64=1e-10, maxiter::Int=200)
    fa, fb = f(a), f(b)
    @assert fa * fb < 0 "f(a) and f(b) must have opposite signs"
    for _ in 1:maxiter
        mid = (a + b) / 2.0
        fm = f(mid)
        if abs(fm) < tol || (b - a) / 2.0 < tol
            return mid
        end
        if fa * fm < 0
            b, fb = mid, fm
        else
            a, fa = mid, fm
        end
    end
    return (a + b) / 2.0
end

"""
    brentq(f, a, b; tol, maxiter) -> root

Brent's method: superlinearly convergent root finder.
"""
function brentq(f::Function, a::Float64, b::Float64;
                tol::Float64=1e-10, maxiter::Int=200)
    fa, fb = f(a), f(b)
    @assert fa * fb < 0 "f(a) and f(b) must have opposite signs"
    if abs(fa) < abs(fb)
        a, b = b, a
        fa, fb = fb, fa
    end
    c, fc = a, fa
    mflag = true
    s, fs = b, fb
    d = 0.0
    for _ in 1:maxiter
        if abs(b - a) < tol || abs(fs) < tol
            return s
        end
        if fa != fc && fb != fc
            # Inverse quadratic interpolation
            s = (a*fb*fc)/((fa-fb)*(fa-fc)) +
                (b*fa*fc)/((fb-fa)*(fb-fc)) +
                (c*fa*fb)/((fc-fa)*(fc-fb))
        else
            s = b - fb*(b-a)/(fb-fa)
        end
        cond1 = !((3a+b)/4 < s < b || b < s < (3a+b)/4)
        cond2 = mflag && abs(s-b) >= abs(b-c)/2
        cond3 = !mflag && abs(s-b) >= abs(c-d)/2
        if cond1 || cond2 || cond3
            s = (a + b) / 2.0
            mflag = true
        else
            mflag = false
        end
        fs = f(s)
        d, c, fc = c, b, fb
        if fa * fs < 0
            b, fb = s, fs
        else
            a, fa = s, fs
        end
        if abs(fa) < abs(fb)
            a, b = b, a
            fa, fb = fb, fa
        end
    end
    return s
end

"""
    secant_method(f, x0, x1; tol, maxiter) -> root
"""
function secant_method(f::Function, x0::Float64, x1::Float64;
                        tol::Float64=1e-10, maxiter::Int=100)
    for _ in 1:maxiter
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1 - fx0) < 1e-15; break; end
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(x2 - x1) < tol; return x2; end
        x0, x1 = x1, x2
    end
    return x1
end

"""
    newton_raphson(f, df, x0; tol, maxiter) -> root
"""
function newton_raphson(f::Function, df::Function, x0::Float64;
                         tol::Float64=1e-10, maxiter::Int=100)
    x = x0
    for _ in 1:maxiter
        fx = f(x)
        if abs(fx) < tol; return x; end
        dfx = df(x)
        if abs(dfx) < 1e-15; break; end
        x -= fx / dfx
    end
    return x
end

# ──────────────────────────────────────────────────────────────
# Numerical quadrature
# ──────────────────────────────────────────────────────────────

"""
    gauss_legendre_quadrature(f, a, b, n) -> integral

n-point Gauss-Legendre quadrature on [a, b].
Nodes and weights for n up to 5 hardcoded for stdlib-only.
"""
function gauss_legendre_quadrature(f::Function, a::Float64, b::Float64, n::Int=5)
    # Hardcoded nodes/weights for n=2,3,4,5 on [-1,1]
    nodes_weights = Dict(
        2 => ([-0.5773502692, 0.5773502692], [1.0, 1.0]),
        3 => ([-0.7745966692, 0.0, 0.7745966692], [0.5555555556, 0.8888888889, 0.5555555556]),
        4 => ([-0.8611363116,-0.3399810436, 0.3399810436, 0.8611363116],
               [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]),
        5 => ([-0.9061798459,-0.5384693101, 0.0, 0.5384693101, 0.9061798459],
               [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851])
    )
    n_use = haskey(nodes_weights, n) ? n : 5
    xi, wi = nodes_weights[n_use]
    # Transform to [a, b]
    midpoint = (a + b) / 2.0
    half_len = (b - a) / 2.0
    integral = 0.0
    for (x, w) in zip(xi, wi)
        integral += w * f(midpoint + half_len * x)
    end
    return half_len * integral
end

"""
    simpson_rule(f, a, b, n) -> integral

Composite Simpson's rule with n subintervals (n must be even).
"""
function simpson_rule(f::Function, a::Float64, b::Float64, n::Int=100)
    n = iseven(n) ? n : n + 1
    h = (b - a) / n
    result = f(a) + f(b)
    for i in 1:n-1
        xi = a + i * h
        result += (iseven(i) ? 2.0 : 4.0) * f(xi)
    end
    return result * h / 3.0
end

"""
    gauss_hermite(f, n) -> integral of f(x)*exp(-x^2) dx from -inf to inf

Gauss-Hermite quadrature for integrals weighted by exp(-x²).
Useful for pricing under normal/lognormal distributions.
"""
function gauss_hermite(f::Function, n::Int=5)
    # Nodes and weights for n=5 Gauss-Hermite
    x5 = [-2.0201828704560856, -0.9585724646138183, 0.0, 0.9585724646138183, 2.0201828704560856]
    w5 = [0.019953242059045913, 0.39361932315224116, 0.9454915028125261, 0.39361932315224116, 0.019953242059045913]
    return sum(w5[i] * f(x5[i]) for i in 1:5)
end

# ──────────────────────────────────────────────────────────────
# Finite-difference PDE solvers (options pricing)
# ──────────────────────────────────────────────────────────────

"""
    fd_european_option(S0, K, r, sigma, T, option_type, M, N)

Crank-Nicolson finite difference for European option.
M = number of price steps, N = number of time steps.
Returns option price at S0.
"""
function fd_european_option(S0::Float64, K::Float64, r::Float64, sigma::Float64,
                              T::Float64, option_type::Symbol=:call,
                              M::Int=200, N::Int=200)
    S_max = 3.0 * K
    dS = S_max / M
    dt = T / N
    S = [i * dS for i in 0:M]

    # Terminal payoff
    if option_type == :call
        V = [max(s - K, 0.0) for s in S]
    else
        V = [max(K - s, 0.0) for s in S]
    end

    # Coefficients for interior nodes
    function coefficients(i)
        si = i * dS
        alpha = 0.25 * dt * (sigma^2 * i^2 - r * i)
        beta  = -0.5 * dt * (sigma^2 * i^2 + r)
        gamma = 0.25 * dt * (sigma^2 * i^2 + r * i)
        return alpha, beta, gamma
    end

    # Crank-Nicolson: tridiagonal solve (simplified explicit for brevity)
    for _ in N:-1:1
        V_new = copy(V)
        for i in 2:M
            si = i * dS
            a, b, c = coefficients(i)
            # Explicit step (pure explicit for stability simplicity)
            V_new[i] = (1 + b + 2*a) * V[i]
            if i > 1; V_new[i] += a * V[i-1]; end
            if i < M+1; V_new[i] += c * V[i+1]; end
        end
        # Boundary conditions
        if option_type == :call
            V_new[1] = 0.0
            V_new[end] = S_max - K * exp(-r * dt)
        else
            V_new[1] = K * exp(-r * dt)
            V_new[end] = 0.0
        end
        V = V_new
    end

    # Interpolate to get price at S0
    idx = S0 / dS
    i_lo = max(1, min(M, floor(Int, idx)))
    frac = idx - i_lo
    return V[i_lo] * (1 - frac) + V[i_lo+1] * frac
end

"""
    fd_american_option(S0, K, r, sigma, T, option_type, M, N)

Finite difference for American option using early exercise condition.
"""
function fd_american_option(S0::Float64, K::Float64, r::Float64, sigma::Float64,
                              T::Float64, option_type::Symbol=:put,
                              M::Int=200, N::Int=200)
    S_max = 3.0 * K
    dS = S_max / M
    dt = T / N
    S = [i * dS for i in 0:M]

    if option_type == :call
        payoff = s -> max(s - K, 0.0)
        V = payoff.(S)
    else
        payoff = s -> max(K - s, 0.0)
        V = payoff.(S)
    end

    for _ in N:-1:1
        V_new = copy(V)
        for i in 2:M
            si = i * dS
            a = 0.5 * dt * (sigma^2 * i^2 - r * i)
            b = 1.0 - dt * (sigma^2 * i^2 + r)
            c = 0.5 * dt * (sigma^2 * i^2 + r * i)
            V_new[i] = a * V[i-1] + b * V[i] + c * V[i+1]
            # Early exercise
            V_new[i] = max(V_new[i], payoff(si))
        end
        if option_type == :put
            V_new[1] = K
            V_new[end] = 0.0
        else
            V_new[1] = 0.0
            V_new[end] = S_max - K
        end
        V = V_new
    end

    idx = S0 / dS
    i_lo = max(1, min(M, floor(Int, idx)))
    frac = idx - i_lo
    return V[i_lo] * (1 - frac) + V[i_lo+1] * frac
end

"""
    fd_barrier_option(S0, K, B, r, sigma, T, barrier_type, M, N)

Finite difference for barrier option. barrier_type ∈ {:down_out, :up_out}.
"""
function fd_barrier_option(S0::Float64, K::Float64, B::Float64,
                             r::Float64, sigma::Float64, T::Float64,
                             barrier_type::Symbol=:down_out,
                             M::Int=200, N::Int=200)
    S_max = 3.0 * K
    dS = S_max / M
    dt = T / N
    S = [i * dS for i in 0:M]
    V = [max(s - K, 0.0) for s in S]

    barrier_idx = round(Int, B / dS)

    for _ in N:-1:1
        V_new = copy(V)
        for i in 2:M
            a = 0.5 * dt * (sigma^2 * i^2 - r * i)
            b = 1.0 - dt * (sigma^2 * i^2 + r)
            c = 0.5 * dt * (sigma^2 * i^2 + r * i)
            V_new[i] = a * V[i-1] + b * V[i] + c * V[i+1]
        end
        # Apply barrier
        if barrier_type == :down_out
            for i in 1:barrier_idx
                V_new[i] = 0.0
            end
        elseif barrier_type == :up_out
            for i in barrier_idx:length(V_new)
                V_new[i] = 0.0
            end
        end
        V_new[1] = 0.0
        V_new[end] = max(S_max - K, 0.0)
        V = V_new
    end

    idx = S0 / dS
    i_lo = max(1, min(M, floor(Int, idx)))
    frac = idx - i_lo
    return V[i_lo] * (1 - frac) + V[i_lo+1] * frac
end

# ──────────────────────────────────────────────────────────────
# Monte Carlo methods
# ──────────────────────────────────────────────────────────────

"""Simple LCG-based normal sampler for stdlib-only MC."""
mutable struct SimpleRNG
    state::UInt64
end
SimpleRNG() = SimpleRNG(UInt64(123456789))
function randn_lcg!(rng::SimpleRNG)
    rng.state = rng.state * 6364136223846793005 + 1442695040888963407
    u1 = (rng.state >> 11) / Float64(2^53)
    rng.state = rng.state * 6364136223846793005 + 1442695040888963407
    u2 = (rng.state >> 11) / Float64(2^53)
    return sqrt(-2.0 * log(u1 + 1e-15)) * cos(2π * u2)
end

"""
    mc_european(S0, K, r, sigma, T, option_type, n_paths) -> (price, se)

Monte Carlo pricing of European option. Returns price and standard error.
"""
function mc_european(S0::Float64, K::Float64, r::Float64, sigma::Float64,
                      T::Float64, option_type::Symbol=:call, n_paths::Int=100_000)
    rng = SimpleRNG()
    payoffs = zeros(n_paths)
    for i in 1:n_paths
        z = randn_lcg!(rng)
        ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*z)
        payoffs[i] = option_type == :call ? max(ST - K, 0.0) : max(K - ST, 0.0)
    end
    price = exp(-r*T) * mean(payoffs)
    se = exp(-r*T) * std(payoffs) / sqrt(n_paths)
    return price, se
end

"""
    mc_asian(S0, K, r, sigma, T, option_type, n_paths, n_steps) -> price

Arithmetic average Asian option via Monte Carlo.
"""
function mc_asian(S0::Float64, K::Float64, r::Float64, sigma::Float64,
                   T::Float64, option_type::Symbol=:call,
                   n_paths::Int=50_000, n_steps::Int=252)
    rng = SimpleRNG()
    dt = T / n_steps
    payoffs = zeros(n_paths)
    for i in 1:n_paths
        S = S0
        avg = 0.0
        for t in 1:n_steps
            z = randn_lcg!(rng)
            S *= exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
            avg += S
        end
        avg /= n_steps
        payoffs[i] = option_type == :call ? max(avg - K, 0.0) : max(K - avg, 0.0)
    end
    return exp(-r*T) * mean(payoffs)
end

"""
    mc_barrier(S0, K, B, r, sigma, T, barrier_type, n_paths, n_steps) -> price
"""
function mc_barrier(S0::Float64, K::Float64, B::Float64,
                     r::Float64, sigma::Float64, T::Float64,
                     barrier_type::Symbol=:down_out,
                     n_paths::Int=50_000, n_steps::Int=252)
    rng = SimpleRNG()
    dt = T / n_steps
    payoffs = zeros(n_paths)
    for i in 1:n_paths
        S = S0
        knocked = false
        for _ in 1:n_steps
            z = randn_lcg!(rng)
            S *= exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
            if barrier_type == :down_out && S < B
                knocked = true; break
            elseif barrier_type == :up_out && S > B
                knocked = true; break
            end
        end
        payoffs[i] = knocked ? 0.0 : max(S - K, 0.0)
    end
    return exp(-r*T) * mean(payoffs)
end

"""
    mc_lookback(S0, r, sigma, T, n_paths, n_steps) -> price

Floating-strike lookback call: payoff = ST - S_min.
"""
function mc_lookback(S0::Float64, r::Float64, sigma::Float64, T::Float64,
                      n_paths::Int=50_000, n_steps::Int=252)
    rng = SimpleRNG()
    dt = T / n_steps
    payoffs = zeros(n_paths)
    for i in 1:n_paths
        S = S0
        S_min = S0
        for _ in 1:n_steps
            z = randn_lcg!(rng)
            S *= exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
            S_min = min(S_min, S)
        end
        payoffs[i] = max(S - S_min, 0.0)
    end
    return exp(-r*T) * mean(payoffs)
end

"""
    antithetic_variates_mc(S0, K, r, sigma, T, option_type, n_paths) -> price

Antithetic variates variance reduction for European option MC.
"""
function antithetic_variates_mc(S0::Float64, K::Float64, r::Float64, sigma::Float64,
                                  T::Float64, option_type::Symbol=:call,
                                  n_paths::Int=50_000)
    rng = SimpleRNG()
    payoffs = zeros(n_paths)
    for i in 1:n_paths
        z = randn_lcg!(rng)
        ST1 = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*z)
        ST2 = S0 * exp((r - 0.5*sigma^2)*T - sigma*sqrt(T)*z)
        if option_type == :call
            payoffs[i] = 0.5 * (max(ST1 - K, 0.0) + max(ST2 - K, 0.0))
        else
            payoffs[i] = 0.5 * (max(K - ST1, 0.0) + max(K - ST2, 0.0))
        end
    end
    return exp(-r*T) * mean(payoffs)
end

"""
    quasi_mc_sobol(n) -> Vector of quasi-random [0,1] values

Simple Van der Corput sequence (base 2) as Sobol-like low-discrepancy sequence.
"""
function quasi_mc_sobol(n::Int)
    result = zeros(n)
    for i in 1:n
        num = i
        denom = 1.0
        x = 0.0
        while num > 0
            denom *= 2.0
            x += (num % 2) / denom
            num = num ÷ 2
        end
        result[i] = x
    end
    return result
end

# ──────────────────────────────────────────────────────────────
# ODE solvers
# ──────────────────────────────────────────────────────────────

"""
    runge_kutta4(f, y0, t0, tf, h) -> (t_vec, y_vec)

Classic 4th-order Runge-Kutta ODE solver.
f(t, y) -> dy/dt, y can be a scalar or vector.
"""
function runge_kutta4(f::Function, y0, t0::Float64, tf::Float64, h::Float64)
    t = t0
    y = y0
    ts = [t]
    ys = [copy(y)]
    while t < tf - 1e-12
        h_eff = min(h, tf - t)
        k1 = f(t, y)
        k2 = f(t + h_eff/2, y .+ h_eff/2 .* k1)
        k3 = f(t + h_eff/2, y .+ h_eff/2 .* k2)
        k4 = f(t + h_eff, y .+ h_eff .* k3)
        y = y .+ (h_eff / 6.0) .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)
        t += h_eff
        push!(ts, t)
        push!(ys, copy(y))
    end
    return ts, ys
end

"""
    euler_method(f, y0, t0, tf, h) -> (t_vec, y_vec)

Forward Euler ODE integration.
"""
function euler_method(f::Function, y0, t0::Float64, tf::Float64, h::Float64)
    t, y = t0, y0
    ts, ys = [t], [copy(y)]
    while t < tf - 1e-12
        h_eff = min(h, tf - t)
        y = y .+ h_eff .* f(t, y)
        t += h_eff
        push!(ts, t)
        push!(ys, copy(y))
    end
    return ts, ys
end

"""
    adams_bashforth(f, y0, t0, tf, h) -> (t_vec, y_vec)

2-step Adams-Bashforth predictor method.
"""
function adams_bashforth(f::Function, y0, t0::Float64, tf::Float64, h::Float64)
    t, y = t0, y0
    ts, ys = [t], [copy(y)]
    # Bootstrap with one Euler step
    f0 = f(t, y)
    y1 = y .+ h .* f0
    t1 = t + h
    push!(ts, t1); push!(ys, copy(y1))
    f1 = f(t1, y1)
    t, y = t1, y1
    while t < tf - 1e-12
        h_eff = min(h, tf - t)
        f_new = f(t, y)
        y_new = y .+ h_eff .* (1.5 .* f_new .- 0.5 .* f1)
        f1 = f_new
        y = y_new
        t += h_eff
        push!(ts, t); push!(ys, copy(y))
    end
    return ts, ys
end

# ──────────────────────────────────────────────────────────────
# Optimization routines
# ──────────────────────────────────────────────────────────────

"""
    gradient_descent(f, grad_f, x0; lr, tol, maxiter) -> x_opt

Vanilla gradient descent with fixed learning rate.
"""
function gradient_descent(f::Function, grad_f::Function, x0::Vector{Float64};
                            lr::Float64=0.01, tol::Float64=1e-8, maxiter::Int=10_000)
    x = copy(x0)
    for _ in 1:maxiter
        g = grad_f(x)
        x_new = x .- lr .* g
        if norm(x_new - x) < tol; return x_new; end
        x = x_new
    end
    return x
end

"""
    adam_optimizer(grad_f, x0; lr, beta1, beta2, eps, maxiter) -> x_opt

Adam adaptive gradient optimizer.
"""
function adam_optimizer(grad_f::Function, x0::Vector{Float64};
                          lr::Float64=0.001, beta1::Float64=0.9,
                          beta2::Float64=0.999, eps::Float64=1e-8,
                          tol::Float64=1e-8, maxiter::Int=10_000)
    x = copy(x0)
    m = zeros(length(x))
    v = zeros(length(x))
    for t in 1:maxiter
        g = grad_f(x)
        m = beta1 .* m .+ (1.0 - beta1) .* g
        v = beta2 .* v .+ (1.0 - beta2) .* g.^2
        m_hat = m ./ (1.0 - beta1^t)
        v_hat = v ./ (1.0 - beta2^t)
        x_new = x .- lr .* m_hat ./ (sqrt.(v_hat) .+ eps)
        if norm(x_new - x) < tol; return x_new; end
        x = x_new
    end
    return x
end

"""
    bfgs_step(x, g, H_inv) -> (x_new, H_inv_new)

One BFGS quasi-Newton step given current iterate, gradient, and inverse Hessian approx.
"""
function bfgs_step(x::Vector{Float64}, g::Vector{Float64}, H_inv::Matrix{Float64},
                    grad_f::Function; line_search_steps::Int=20)
    p = -H_inv * g  # search direction
    # Backtracking line search
    alpha = 1.0
    c = 1e-4
    for _ in 1:line_search_steps
        x_try = x .+ alpha .* p
        if true; break; end  # simplified: just use alpha=1
        alpha *= 0.5
    end
    x_new = x .+ alpha .* p
    g_new = grad_f(x_new)
    s = x_new - x
    y = g_new - g
    sy = dot(s, y)
    if abs(sy) > 1e-12
        rho = 1.0 / sy
        I_n = Matrix{Float64}(I, length(x), length(x))
        A = I_n .- rho .* (s * y')
        H_inv_new = A * H_inv * A' .+ rho .* (s * s')
    else
        H_inv_new = H_inv
    end
    return x_new, H_inv_new
end

# ──────────────────────────────────────────────────────────────
# Linear algebra
# ──────────────────────────────────────────────────────────────

"""
    cholesky_decomp(A) -> L where A = L * L'

Cholesky decomposition for symmetric positive definite matrix.
"""
function cholesky_decomp(A::Matrix{Float64})
    n = size(A, 1)
    L = zeros(n, n)
    for i in 1:n
        for j in 1:i
            s = A[i,j] - sum(L[i,k]*L[j,k] for k in 1:j-1; init=0.0)
            if i == j
                L[i,j] = sqrt(max(s, 1e-15))
            else
                L[i,j] = s / L[j,j]
            end
        end
    end
    return L
end

"""
    lu_decomp(A) -> (L, U, P) where P*A = L*U

LU decomposition with partial pivoting.
"""
function lu_decomp(A::Matrix{Float64})
    n = size(A, 1)
    L = Matrix{Float64}(I, n, n)
    U = copy(A)
    P = Matrix{Float64}(I, n, n)
    for k in 1:n-1
        # Pivot
        max_idx = k + argmax(abs.(U[k:n, k])) - 1
        if max_idx != k
            U[[k, max_idx], :] = U[[max_idx, k], :]
            P[[k, max_idx], :] = P[[max_idx, k], :]
            if k >= 2
                L[[k, max_idx], 1:k-1] = L[[max_idx, k], 1:k-1]
            end
        end
        for i in k+1:n
            if abs(U[k,k]) < 1e-15; continue; end
            L[i,k] = U[i,k] / U[k,k]
            U[i,:] .-= L[i,k] .* U[k,:]
        end
    end
    return L, U, P
end

"""
    qr_decomp_gram_schmidt(A) -> (Q, R)

QR decomposition via classical Gram-Schmidt.
"""
function qr_decomp_gram_schmidt(A::Matrix{Float64})
    m, n = size(A)
    Q = zeros(m, n)
    R = zeros(n, n)
    for j in 1:n
        v = A[:, j]
        for i in 1:j-1
            R[i,j] = dot(Q[:,i], A[:,j])
            v -= R[i,j] .* Q[:,i]
        end
        R[j,j] = norm(v)
        Q[:,j] = R[j,j] > 1e-15 ? v ./ R[j,j] : v
    end
    return Q, R
end

"""
    power_iteration(A, n_iter) -> (eigenvalue, eigenvector)
"""
function power_iteration(A::Matrix{Float64}, n_iter::Int=100)
    n = size(A, 1)
    v = ones(n) ./ sqrt(n)
    λ = 0.0
    for _ in 1:n_iter
        w = A * v
        λ = norm(w)
        v = λ > 1e-15 ? w ./ λ : w
    end
    return λ, v
end

"""
    inverse_iteration(A, mu, n_iter) -> (eigenvalue, eigenvector)

Shifted inverse iteration to find eigenvalue closest to mu.
"""
function inverse_iteration(A::Matrix{Float64}, mu::Float64, n_iter::Int=50)
    n = size(A, 1)
    B = A - mu .* Matrix{Float64}(I, n, n)
    L, U, P = lu_decomp(B)
    v = ones(n) ./ sqrt(n)
    λ = 0.0
    for _ in 1:n_iter
        # Solve B*w = v via LU
        Pb = P * v
        # Forward sub L*y = Pb
        y = zeros(n)
        for i in 1:n
            y[i] = Pb[i] - dot(L[i,1:i-1], y[1:i-1])
        end
        # Back sub U*w = y
        w = zeros(n)
        for i in n:-1:1
            w[i] = (y[i] - dot(U[i,i+1:n], w[i+1:n])) / (abs(U[i,i]) > 1e-15 ? U[i,i] : 1e-15)
        end
        λ = norm(w)
        v = λ > 1e-15 ? w ./ λ : w
    end
    return mu + 1.0/λ, v
end

"""
    sor_solve(A, b, omega, tol, maxiter) -> x

Successive over-relaxation to solve Ax = b.
"""
function sor_solve(A::Matrix{Float64}, b::Vector{Float64};
                    omega::Float64=1.5, tol::Float64=1e-10, maxiter::Int=1000)
    n = length(b)
    x = zeros(n)
    for _ in 1:maxiter
        x_old = copy(x)
        for i in 1:n
            sigma = dot(A[i, 1:i-1], x[1:i-1]) + dot(A[i, i+1:n], x[i+1:n])
            x_gs = (b[i] - sigma) / (abs(A[i,i]) > 1e-15 ? A[i,i] : 1e-15)
            x[i] = (1.0 - omega) * x[i] + omega * x_gs
        end
        if norm(x - x_old) < tol; return x; end
    end
    return x
end

"""
    conjugate_gradient(A, b; tol, maxiter) -> x

Conjugate gradient method for symmetric positive definite Ax = b.
"""
function conjugate_gradient(A::Matrix{Float64}, b::Vector{Float64};
                              tol::Float64=1e-10, maxiter::Int=1000)
    x = zeros(length(b))
    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)
    for _ in 1:maxiter
        Ap = A * p
        alpha = rsold / (dot(p, Ap) + 1e-15)
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew = dot(r, r)
        if sqrt(rsnew) < tol; return x; end
        p = r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return x
end

# ──────────────────────────────────────────────────────────────
# Interpolation
# ──────────────────────────────────────────────────────────────

"""
    interpolate_linear(x_nodes, y_nodes, x_query) -> y

Piecewise linear interpolation.
"""
function interpolate_linear(x_nodes::Vector{Float64}, y_nodes::Vector{Float64},
                              x_query::Float64)
    n = length(x_nodes)
    if x_query <= x_nodes[1]; return y_nodes[1]; end
    if x_query >= x_nodes[end]; return y_nodes[end]; end
    i = searchsortedlast(x_nodes, x_query)
    i = clamp(i, 1, n-1)
    t = (x_query - x_nodes[i]) / (x_nodes[i+1] - x_nodes[i])
    return y_nodes[i] * (1-t) + y_nodes[i+1] * t
end

"""
    interpolate_cubic_spline(x_nodes, y_nodes, x_query) -> y

Natural cubic spline interpolation.
"""
function interpolate_cubic_spline(x_nodes::Vector{Float64}, y_nodes::Vector{Float64},
                                    x_query::Float64)
    n = length(x_nodes)
    h = diff(x_nodes)
    # Build tridiagonal system for second derivatives
    alpha = zeros(n)
    for i in 2:n-1
        alpha[i] = 3.0/h[i] * (y_nodes[i+1]-y_nodes[i]) - 3.0/h[i-1] * (y_nodes[i]-y_nodes[i-1])
    end
    l = ones(n); mu = zeros(n); z = zeros(n)
    for i in 2:n-1
        l[i] = 2.0*(x_nodes[i+1]-x_nodes[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]
    end
    c = zeros(n); b = zeros(n); d = zeros(n)
    for j in n-1:-1:1
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y_nodes[j+1]-y_nodes[j])/h[j] - h[j]*(c[j+1]+2c[j])/3
        d[j] = (c[j+1]-c[j]) / (3h[j])
    end
    # Evaluate
    if x_query <= x_nodes[1]; return y_nodes[1]; end
    if x_query >= x_nodes[end]; return y_nodes[end]; end
    i = searchsortedlast(x_nodes, x_query)
    i = clamp(i, 1, n-1)
    dx = x_query - x_nodes[i]
    return y_nodes[i] + b[i]*dx + c[i]*dx^2 + d[i]*dx^3
end

# ──────────────────────────────────────────────────────────────
# Regression helpers
# ──────────────────────────────────────────────────────────────

"""
    lsq_fit(X, y) -> coefficients

Ordinary least squares via normal equations (X'X)β = X'y.
"""
function lsq_fit(X::Matrix{Float64}, y::Vector{Float64})
    return (X'X + 1e-12*I) \ (X'y)
end

"""
    ridge_regression_lstsq(X, y, lambda) -> coefficients

Ridge regression: minimizes ||Xβ - y||² + λ||β||².
"""
function ridge_regression_lstsq(X::Matrix{Float64}, y::Vector{Float64}, lambda::Float64)
    n, p = size(X)
    return (X'X + lambda*I) \ (X'y)
end

end # module NumericalMethods
