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


# ============================================================
# SECTION 2: ADVANCED ROOT FINDING & OPTIMIZATION
# ============================================================

function bisection(f::Function, a::Float64, b::Float64; tol::Float64=1e-10, maxiter::Int=200)
    fa=f(a); fb=f(b)
    fa*fb > 0 && error("sign change required")
    for _ in 1:maxiter
        c=(a+b)/2; fc=f(c)
        abs(fc)<tol && return c
        fa*fc<0 ? (b=c;fb=fc) : (a=c;fa=fc)
    end
    return (a+b)/2
end

function newton_raphson(f::Function, df::Function, x0::Float64;
                         tol::Float64=1e-10, maxiter::Int=100)
    x=x0
    for i in 1:maxiter
        fx=f(x); dfx=df(x)
        abs(dfx)<1e-15 && break
        xn=x-fx/dfx; abs(xn-x)<tol && return xn; x=xn
    end
    return x
end

function brent_method(f::Function, a::Float64, b::Float64;
                       tol::Float64=1e-10, maxiter::Int=100)
    fa=f(a); fb=f(b); fa*fb > 0 && error("sign change required")
    c=a; fc=fa; d=b-a; e=d
    for _ in 1:maxiter
        if fb*fc>0; c=a; fc=fa; d=b-a; e=d; end
        abs(fc)<abs(fb) && (a=b;b=c;c=a; fa=fb;fb=fc;fc=fa)
        tol1=2*eps()*abs(b)+0.5*tol; xm=(c-b)/2
        (abs(xm)<=tol1||abs(fb)<1e-15) && return b
        if abs(e)>=tol1 && abs(fa)>abs(fb)
            s=fb/fa; p=2*xm*s; q=1-s
            if a!=c; t=fb/fc; r=fb/fa; p=s*(2*xm*t*(t-r)-(b-a)*(r-1)); q=(t-1)*(r-1)*(s-1); end
            p>0 && (q=-q); p=abs(p)
            if 2*p<min(3*xm*q-abs(tol1*q), abs(e*q))
                e=d; d=p/q
            else; d=xm; e=d; end
        else; d=xm; e=d; end
        a=b; fa=fb
        b+=abs(d)>tol1 ? d : sign(xm)*tol1
        fb=f(b)
    end
    return b
end

function secant(f::Function, x0::Float64, x1::Float64; tol::Float64=1e-10, maxiter::Int=100)
    for _ in 1:maxiter
        fx0=f(x0); fx1=f(x1)
        abs(fx1-fx0)<1e-15 && break
        x2=x1-fx1*(x1-x0)/(fx1-fx0)
        abs(x2-x1)<tol && return x2; x0=x1; x1=x2
    end
    return x1
end

function gradient_descent_wolfe(f_grad::Function, x0::Vector{Float64};
                                  c1::Float64=1e-4, c2::Float64=0.9,
                                  max_iter::Int=1000, tol::Float64=1e-6)
    x=copy(x0); f_val,g=f_grad(x)
    for i in 1:max_iter
        p=-g; alpha=1.0
        # Backtracking line search
        for _ in 1:50
            xn=x.+alpha.*p; fn,_=f_grad(xn)
            fn<=f_val+c1*alpha*dot(g,p) && break; alpha*=0.5
        end
        x_new=x.+alpha.*p; f_new,g_new=f_grad(x_new)
        norm(g_new)<tol && return (x=x_new,f=f_new,iter=i,converged=true)
        x=x_new; f_val=f_new; g=g_new
    end
    return (x=x, f=f_val, iter=max_iter, converged=false)
end

function conjugate_gradient_linear(A::Matrix{Float64}, b::Vector{Float64};
                                     tol::Float64=1e-10, maxiter::Int=1000)
    n=length(b); x=zeros(n); r=copy(b); p=copy(r)
    rs_old=dot(r,r)
    for _ in 1:maxiter
        Ap=A*p; alpha=rs_old/(dot(p,Ap)+1e-15)
        x.+=alpha.*p; r.-=alpha.*Ap
        rs_new=dot(r,r)
        rs_new<tol^2 && break
        p=r.+(rs_new/rs_old).*p; rs_old=rs_new
    end
    return x
end

function nelder_mead(f::Function, x0::Vector{Float64};
                      tol::Float64=1e-8, maxiter::Int=5000)
    n=length(x0)
    s=[copy(x0) for _ in 1:n+1]
    for i in 2:n+1; s[i][i-1]+=max(0.05,0.05*abs(x0[i-1])); end
    v=[f(si) for si in s]
    for _ in 1:maxiter
        ord=sortperm(v); s=s[ord]; v=v[ord]
        xo=mean(s[1:n])
        xr=xo.+1.0.*(xo.-s[n+1]); fr=f(xr)
        if fr<v[1]
            xe=xo.+2.0.*(xr.-xo); fe=f(xe)
            if fe<fr; s[n+1]=xe; v[n+1]=fe
            else; s[n+1]=xr; v[n+1]=fr; end
        elseif fr<v[n]
            s[n+1]=xr; v[n+1]=fr
        else
            xc=xo.+0.5.*(s[n+1].-xo); fc=f(xc)
            if fc<v[n+1]; s[n+1]=xc; v[n+1]=fc
            else
                for i in 2:n+1
                    s[i]=s[1].+0.5.*(s[i].-s[1]); v[i]=f(s[i])
                end
            end
        end
        std([f(si) for si in s])<tol && return (x=s[1],f=v[1],converged=true)
    end
    return (x=s[1],f=v[1],converged=false)
end

function simulated_annealing_opt(f::Function, x0::Vector{Float64};
                                   T0::Float64=10.0, Tf::Float64=1e-5,
                                   cool::Float64=0.995, maxiter::Int=50000,
                                   step::Float64=0.1)
    x=copy(x0); fx=f(x); best_x=copy(x); best_f=fx; T=T0
    for _ in 1:maxiter
        xn=x.+randn(length(x)).*step; fn=f(xn)
        delta=fn-fx
        (delta<0||rand()<exp(-delta/T)) && (x=xn; fx=fn)
        fx<best_f && (best_f=fx; best_x=copy(x))
        T=max(T*cool,Tf); T<=Tf && break
    end
    return (x=best_x, f=best_f)
end

# ============================================================
# SECTION 3: ADVANCED INTEGRATION
# ============================================================

function gauss_legendre_quadrature(f::Function, a::Float64, b::Float64; n::Int=10)
    # Gauss-Legendre nodes and weights for n up to 10
    nodes_weights = Dict(
        2 => ([-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]),
        3 => ([-0.7745966692414834, 0.0, 0.7745966692414834],
               [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
        5 => ([-0.9061798459386640, -0.5384693101056831, 0.0,
                0.5384693101056831, 0.9061798459386640],
               [0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                0.4786286704993665, 0.2369268850561891])
    )
    n_use = n in keys(nodes_weights) ? n : 5
    xi, wi = nodes_weights[n_use]
    mid = (a+b)/2; half = (b-a)/2
    return half * sum(wi[i]*f(mid+half*xi[i]) for i in eachindex(xi))
end

function adaptive_quadrature(f::Function, a::Float64, b::Float64;
                               tol::Float64=1e-8, maxdepth::Int=50)
    function quad_rec(a, b, fa, fm, fb, tol, depth)
        m=(a+b)/2
        lm=(a+m)/2; rm=(m+b)/2
        flm=f(lm); frm=f(rm)
        s1=( b-a)/6*(fa+4*fm+fb)
        s2=(b-a)/12*(fa+4*flm+2*fm+4*frm+fb)
        if depth>=maxdepth||abs(s2-s1)<15*tol
            return s2+(s2-s1)/15
        end
        return quad_rec(a,m,fa,flm,fm,tol/2,depth+1)+
               quad_rec(m,b,fm,frm,fb,tol/2,depth+1)
    end
    fa=f(a); fm=f((a+b)/2); fb=f(b)
    return quad_rec(a,b,fa,fm,fb,tol,0)
end

function romberg(f::Function, a::Float64, b::Float64; max_order::Int=8, tol::Float64=1e-10)
    R=zeros(max_order,max_order); h=b-a
    R[1,1]=h*(f(a)+f(b))/2
    for j in 2:max_order
        h/=2
        R[j,1]=R[j-1,1]/2+h*sum(f(a+(2k-1)*h) for k in 1:2^(j-2))
        for k in 2:j; R[j,k]=R[j,k-1]+(R[j,k-1]-R[j-1,k-1])/(4^(k-1)-1); end
        j>2 && abs(R[j,j]-R[j-1,j-1])<tol && return R[j,j]
    end
    return R[max_order,max_order]
end

function monte_carlo_integrate_qmc(f::Function, a::Float64, b::Float64, n::Int=10000)
    # Halton sequence for QMC
    function halton(k, base)
        r=0.0; f_=1.0; n_=k
        while n_>0; f_/=base; r+=f_*(n_%base); n_÷=base; end
        r
    end
    xs = [a+(b-a)*halton(i,2) for i in 1:n]
    return (b-a)*mean(f.(xs))
end

# ============================================================
# SECTION 4: ODE & PDE SOLVERS
# ============================================================

function euler_ode(f::Function, y0::Vector{Float64},
                    t0::Float64, t1::Float64, n::Int=1000)
    h=(t1-t0)/n; t=t0; y=copy(y0)
    ts=[t]; ys=[copy(y)]
    for _ in 1:n
        y=y.+h.*f(t,y); t+=h
        push!(ts,t); push!(ys,copy(y))
    end
    return (t=ts,y=ys)
end

function rk4_ode(f::Function, y0::Vector{Float64},
                  t0::Float64, t1::Float64, n::Int=1000)
    h=(t1-t0)/n; t=t0; y=copy(y0)
    ts=[t]; ys=[copy(y)]
    for _ in 1:n
        k1=f(t,y); k2=f(t+h/2,y.+h/2.*k1)
        k3=f(t+h/2,y.+h/2.*k2); k4=f(t+h,y.+h.*k3)
        y=y.+h/6.*(k1.+2*k2.+2*k3.+k4); t+=h
        push!(ts,t); push!(ys,copy(y))
    end
    return (t=ts,y=ys)
end

function crank_nicolson_heat(u0::Vector{Float64}, dx::Float64, dt::Float64,
                               D::Float64, T::Float64)
    n=length(u0); nsteps=round(Int,T/dt); r=D*dt/dx^2/2
    u=copy(u0); ni=n-2
    # Build matrices once
    A_lo=fill(-r,ni-1); A_md=fill(1+2r,ni); A_up=fill(-r,ni-1)
    B_lo=fill(r,ni-1);  B_md=fill(1-2r,ni); B_up=fill(r,ni-1)
    for _ in 1:nsteps
        rhs=zeros(ni)
        for i in 1:ni
            rhs[i]=B_md[i]*u[i+1]
            i>1 && (rhs[i]+=B_lo[i-1]*u[i])
            i<ni && (rhs[i]+=B_up[i]*u[i+2])
        end
        # Thomas algorithm
        c=zeros(ni); d=zeros(ni)
        c[1]=A_up[1]/A_md[1]; d[1]=rhs[1]/A_md[1]
        for i in 2:ni
            denom=A_md[i]-A_lo[i-1]*c[i-1]
            c[i]=i<ni ? A_up[i]/denom : 0.0
            d[i]=(rhs[i]-A_lo[i-1]*d[i-1])/denom
        end
        x=zeros(ni); x[ni]=d[ni]
        for i in ni-1:-1:1; x[i]=d[i]-c[i]*x[i+1]; end
        u[2:n-1]=x
    end
    return u
end

function thomas_algorithm(lo::Vector{Float64}, md::Vector{Float64},
                            up::Vector{Float64}, rhs::Vector{Float64})
    n=length(md); c=zeros(n); d=zeros(n)
    c[1]=up[1]/md[1]; d[1]=rhs[1]/md[1]
    for i in 2:n
        denom=md[i]-lo[i-1]*c[i-1]
        c[i]=i<n ? up[i]/denom : 0.0
        d[i]=(rhs[i]-lo[i-1]*d[i-1])/denom
    end
    x=zeros(n); x[n]=d[n]
    for i in n-1:-1:1; x[i]=d[i]-c[i]*x[i+1]; end
    return x
end

# ============================================================
# SECTION 5: QUASI-MONTE CARLO & VARIANCE REDUCTION
# ============================================================

function sobol_sequence_2d(n::Int)
    # Simple bit-reversal Sobol for 2D
    xs=zeros(n); ys=zeros(n)
    for i in 1:n
        r=0.0; f_=0.5; k=i
        while k>0; r+=f_*(k&1); k>>=1; f_*=0.5; end; xs[i]=r
        r=0.0; f_=0.5; k=i
        # Second dimension with different scramble
        while k>0; r+=f_*((k>>1)&1+k&1)%2; k>>=1; f_*=0.5; end; ys[i]=r
    end
    return xs,ys
end

function antithetic_variates_mc(f::Function, n::Int=50000)
    samples = randn(n)
    y1=[f(z) for z in samples]; y2=[f(-z) for z in samples]
    paired=[(y1[i]+y2[i])/2 for i in 1:n]
    return (estimate=mean(paired), se=std(paired)/sqrt(n),
            variance_reduction=1-var(paired)/var(y1))
end

function control_variate_option(S0::Float64, K::Float64, r::Float64,
                                  sigma::Float64, T::Float64, n::Int=100000)
    dt=T; sqrt_T=sqrt(dt)
    zs=randn(n)
    STs=S0.*exp.((r-0.5*sigma^2).*T.+sigma.*sqrt_T.*zs)
    payoffs=max.(STs.-K,0.0).*exp(-r*T)
    # Control: E[S_T] = S0*exp(r*T)
    control=STs; E_control=S0*exp(r*T)
    c_opt=cov(payoffs,control)/(var(control)+1e-10)
    cv_payoffs=payoffs.-c_opt.*(control.-E_control)
    return (price=mean(cv_payoffs), se=std(cv_payoffs)/sqrt(n))
end

function stratified_sampling(f::Function, a::Float64, b::Float64,
                               n_strata::Int=100, samps::Int=10)
    w=(b-a)/n_strata; total=0.0; var_total=0.0
    for k in 1:n_strata
        lo=a+(k-1)*w; vals=[f(lo+rand()*w) for _ in 1:samps]
        total+=mean(vals)*w; var_total+=var(vals)/samps*w^2
    end
    return (estimate=total, se=sqrt(var_total))
end

# ============================================================
# SECTION 6: INTERPOLATION & REGRESSION
# ============================================================

function cubic_spline_interpolate(xs::Vector{Float64}, ys::Vector{Float64},
                                    x_query::Vector{Float64})
    n=length(xs); h=diff(xs)
    # Natural cubic spline: tridiagonal system for second derivatives
    n_int=n-2; rhs=zeros(n_int)
    lo=zeros(n_int-1); md=zeros(n_int); up=zeros(n_int-1)
    for i in 1:n_int
        md[i]=2*(h[i]+h[i+1])
        rhs[i]=6*((ys[i+2]-ys[i+1])/h[i+1]-(ys[i+1]-ys[i])/h[i])
        i<n_int && (up[i]=h[i+1]; lo[i]=h[i+1])
    end
    sigma=thomas_algorithm(lo, md, up, rhs)
    sigma_all=vcat(0.0, sigma, 0.0)
    result=zeros(length(x_query))
    for (qi,xq) in enumerate(x_query)
        seg=searchsortedlast(xs, xq); seg=clamp(seg,1,n-1)
        dx=xs[seg+1]-xs[seg]; t=(xq-xs[seg])/dx
        a_=sigma_all[seg]*dx^2/6; b_=sigma_all[seg+1]*dx^2/6
        result[qi]=((1-t)*ys[seg]+t*ys[seg+1] +
                    t*(1-t)*((1-t)*(a_) + t*(b_)))
    end
    return result
end

function polynomial_regression(x::Vector{Float64}, y::Vector{Float64}, degree::Int=2)
    n=length(x)
    X=hcat([x.^k for k in 0:degree]...)
    beta=(X'*X+1e-8*I(degree+1))\(X'*y)
    fitted=X*beta; resid=y.-fitted
    r2=1-var(resid)/(var(y)+1e-10)
    return (beta=beta, fitted=fitted, r2=r2)
end

function local_polynomial_regression(x::Vector{Float64}, y::Vector{Float64},
                                       x_grid::Vector{Float64}; h::Float64=1.0)
    n=length(x); m=length(x_grid); fitted=zeros(m)
    for i in 1:m
        xg=x_grid[i]
        w=[exp(-0.5*((x[j]-xg)/h)^2) for j in 1:n]
        W=Diagonal(w)
        X=hcat(ones(n), x.-xg)
        beta=(X'*W*X+1e-8*I(2))\(X'*W*y)
        fitted[i]=beta[1]
    end
    return fitted
end

# ============================================================
# SECTION 7: LINEAR ALGEBRA & DECOMPOSITIONS
# ============================================================

function gram_schmidt(A::Matrix{Float64})
    m,n=size(A); Q=zeros(m,n); R=zeros(n,n)
    for j in 1:n
        v=A[:,j]
        for i in 1:j-1
            R[i,j]=dot(Q[:,i],A[:,j]); v.-=R[i,j].*Q[:,i]
        end
        R[j,j]=norm(v); Q[:,j]=v./(R[j,j]+1e-15)
    end
    return Q,R
end

function power_iteration(A::Matrix{Float64}; maxiter::Int=1000, tol::Float64=1e-10)
    n=size(A,1); v=randn(n); v./=norm(v)
    lambda=0.0
    for _ in 1:maxiter
        Av=A*v; lambda=dot(v,Av)
        v_new=Av./norm(Av)
        norm(v_new-v)<tol && (v=v_new; break); v=v_new
    end
    return (eigenvalue=lambda, eigenvector=v)
end

function inverse_iteration(A::Matrix{Float64}, mu::Float64=0.0;
                             maxiter::Int=500, tol::Float64=1e-10)
    n=size(A,1); v=randn(n); v./=norm(v)
    Amu=A-mu*I(n)
    for _ in 1:maxiter
        w=Amu\v; v_new=w./norm(w)
        norm(v_new-v)<tol && (v=v_new; break); v=v_new
    end
    Av=A*v; lambda=dot(v,Av)
    return (eigenvalue=lambda, eigenvector=v)
end

function householder_qr(A::Matrix{Float64})
    m,n=size(A); Q=Matrix{Float64}(I,m,m); R=copy(A)
    for k in 1:min(m-1,n)
        x=R[k:m,k]; e=zeros(m-k+1); e[1]=norm(x)
        v=x.-e; nv=norm(v)
        nv<1e-14 && continue; v./=nv
        R[k:m,k:n].-=2*v*(v'*R[k:m,k:n])
        Q[:,k:m].=(Q[:,k:m]*(I(m-k+1)-2*v*v'))'
    end
    return Q,R
end

# ============================================================
# SECTION 8: FINANCIAL NUMERICS
# ============================================================

function black_scholes_call_price(S::Float64, K::Float64, r::Float64,
                                    sigma::Float64, T::Float64)
    T<=0 && return max(S-K,0.0)
    d1=(log(S/K)+(r+0.5*sigma^2)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    N(x)=0.5*(1+erf(x/sqrt(2)))
    return S*N(d1)-K*exp(-r*T)*N(d2)
end

function black_scholes_put_price(S::Float64, K::Float64, r::Float64,
                                   sigma::Float64, T::Float64)
    T<=0 && return max(K-S,0.0)
    d1=(log(S/K)+(r+0.5*sigma^2)*T)/(sigma*sqrt(T))
    d2=d1-sigma*sqrt(T)
    N(x)=0.5*(1+erf(x/sqrt(2)))
    return K*exp(-r*T)*N(-d2)-S*N(-d1)
end

function implied_volatility_brent(market_price::Float64, S::Float64, K::Float64,
                                    r::Float64, T::Float64; is_call::Bool=true)
    T<=0 && return NaN
    f = is_call ? black_scholes_call_price : black_scholes_put_price
    try
        return brent_method(v -> f(S,K,r,v,T) - market_price, 0.001, 10.0)
    catch; return NaN; end
end

function bond_duration(cash_flows::Vector{Float64}, times::Vector{Float64}, ytm::Float64)
    pv = [cf*exp(-ytm*t) for (cf,t) in zip(cash_flows,times)]
    price = sum(pv)
    duration = sum(pv[i]*times[i] for i in eachindex(pv)) / (price+1e-10)
    modified = duration/(1+ytm)
    convexity = sum(pv[i]*times[i]^2 for i in eachindex(pv)) / (price+1e-10)
    return (price=price, duration=duration, modified_duration=modified, convexity=convexity)
end

function bond_ytm(price::Float64, cash_flows::Vector{Float64}, times::Vector{Float64})
    f(y) = sum(cf*exp(-y*t) for (cf,t) in zip(cash_flows,times)) - price
    try; return brent_method(f, 0.001, 0.5); catch; return NaN; end
end

function var_historical(returns::Vector{Float64}, confidence::Float64=0.95)
    return -quantile(returns, 1-confidence)
end

function var_parametric(returns::Vector{Float64}, confidence::Float64=0.95)
    mu=mean(returns); sigma=std(returns)
    # z for 95% is 1.645, for 99% is 2.326
    z_map=Dict(0.90=>1.282, 0.95=>1.645, 0.99=>2.326)
    z=get(z_map, round(confidence,digits=2), 1.645)
    return -(mu-z*sigma)
end

function expected_shortfall(returns::Vector{Float64}, confidence::Float64=0.95)
    threshold=quantile(returns, 1-confidence)
    tail=[r for r in returns if r<=threshold]
    return isempty(tail) ? var_historical(returns,confidence) : -mean(tail)
end

# ============================================================
# EXTENDED DEMO
# ============================================================

function demo_numerical_methods_extended()
    println("=== Numerical Methods Extended Demo ===")

    # Root finding
    f3(x) = x^3-2x-5
    println("Bisection: ", round(bisection(f3,2.0,3.0),digits=8))
    println("Brent:     ", round(brent_method(f3,2.0,3.0),digits=8))

    # Integration
    fi(x) = sin(x)^2/(1+x^2)
    println("Romberg ∫: ", round(romberg(fi,0.0,π),digits=8))
    println("GL 5pt  ∫: ", round(gauss_legendre_quadrature(fi,0.0,π;n=5),digits=6))

    # ODE
    sir(t,y)=begin β=0.3;γ=0.1;N=1e6; [-β*y[1]*y[2]/N, β*y[1]*y[2]/N-γ*y[2], γ*y[2]] end
    sol=rk4_ode(sir,[999000.0,1000.0,0.0],0.0,160.0,160)
    println("SIR peak I: ", round(maximum([y[2] for y in sol.y])/1e6,digits=3),"M")

    # BS option
    println("BS call: ", round(black_scholes_call_price(100.0,100.0,0.05,0.2,1.0),digits=4))

    # IV
    mkt=black_scholes_call_price(100.0,100.0,0.05,0.2,1.0)
    iv=implied_volatility_brent(mkt,100.0,100.0,0.05,1.0)
    println("Round-trip IV: ", round(iv,digits=4))

    # VaR
    rets=randn(1000).*0.01
    println("Historical VaR 95%: ", round(var_historical(rets)*100,digits=3),"%")
    println("ES 95%: ", round(expected_shortfall(rets)*100,digits=3),"%")

    # Spline
    xs=[0.0,1.0,2.0,3.0,4.0]; ys=sin.(xs)
    interp=cubic_spline_interpolate(xs,ys,[0.5,1.5,2.5])
    println("Cubic spline at [0.5,1.5,2.5]: ", round.(interp,digits=4))

    # Bond
    cfs=[5.0,5.0,5.0,105.0]; ts=[1.0,2.0,3.0,4.0]
    bd=bond_duration(cfs,ts,0.05)
    println("Bond duration: ", round(bd.duration,digits=4),
            " modified: ", round(bd.modified_duration,digits=4))
end

end # module NumericalMethods
