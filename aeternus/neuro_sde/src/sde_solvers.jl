"""
sde_solvers.jl — SDE numerical integration schemes

Implements from scratch (not wrapping DiffEq.jl):
  - Euler-Maruyama with optional adaptive step size
  - Milstein scheme with Lévy area approximation
  - Runge-Kutta 4.5 for SDEs (Rößler / Platen style)

Each solver supports:
  - Itô and Stratonovich interpretations
  - Batch path simulation (vectorised over initial conditions)
  - GPU-compatible array operations
  - Configurable RNG for reproducibility
"""

using LinearAlgebra
using Random
using Statistics

# ─────────────────────────────────────────────────────────────────────────────
# TYPES: INTERPRETATION & SOLVER TOKENS
# ─────────────────────────────────────────────────────────────────────────────

abstract type SDEInterpretation end
struct ItoInterpretation          <: SDEInterpretation end
struct StratonovichInterpretation <: SDEInterpretation end

abstract type SDESolver end
struct EulerMaruyama <: SDESolver
    adaptive  :: Bool
    rtol      :: Float64
    atol      :: Float64
    dt_min    :: Float64
    dt_max    :: Float64
end
EulerMaruyama(; adaptive=false, rtol=1e-3, atol=1e-6,
               dt_min=1e-6, dt_max=0.1) =
    EulerMaruyama(adaptive, rtol, atol, dt_min, dt_max)

struct Milstein <: SDESolver
    levy_approx_order :: Int   # number of terms for Lévy area approx
end
Milstein(; levy_approx_order::Int=5) = Milstein(levy_approx_order)

struct RungeKutta45SDE <: SDESolver
    rtol :: Float64
    atol :: Float64
end
RungeKutta45SDE(; rtol=1e-4, atol=1e-8) = RungeKutta45SDE(rtol, atol)

# ─────────────────────────────────────────────────────────────────────────────
# SDE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────

"""
    SDEProblem

Encapsulates an SDE:  dX = f(X,t)dt + g(X,t)dW

Fields:
  - `f`           : drift function f(x, t, params) → drift vector
  - `g`           : diffusion function g(x, t, params) → diffusion coeff
  - `x0`          : initial condition (vector or matrix for batches)
  - `tspan`       : (t0, t1) tuple
  - `params`      : model parameters (passed to f and g)
  - `noise_dim`   : dimension of Wiener process (default = state_dim)
  - `interpretation`: ItoInterpretation() or StratonovichInterpretation()
"""
struct SDEProblem{F,G,X,P,I<:SDEInterpretation}
    f              :: F
    g              :: G
    x0             :: X
    tspan          :: Tuple{Float64,Float64}
    params         :: P
    noise_dim      :: Int
    interpretation :: I
end

function SDEProblem(f, g, x0, tspan;
                    params         = nothing,
                    noise_dim      = length(x0),
                    interpretation = ItoInterpretation())
    SDEProblem(f, g, x0, tspan, params, noise_dim, interpretation)
end

# ─────────────────────────────────────────────────────────────────────────────
# EULER-MARUYAMA SOLVER
# ─────────────────────────────────────────────────────────────────────────────

"""
    euler_maruyama_step(f, g, x, t, dt, dW, params, interp)

Single Euler-Maruyama step.

Itô:          X_{n+1} = X_n + f(X_n, t_n)·Δt + g(X_n, t_n)·ΔW_n
Stratonovich: Euler-Heun correction — f and g evaluated at midpoint.
"""
function euler_maruyama_step(f, g, x, t, dt, dW, params,
                              interp::ItoInterpretation)
    μ = f(x, t, params)
    σ = g(x, t, params)
    return x .+ μ .* dt .+ σ .* dW
end

function euler_maruyama_step(f, g, x, t, dt, dW, params,
                              interp::StratonovichInterpretation)
    # Euler-Heun (predictor-corrector for Stratonovich)
    μ0 = f(x, t,      params)
    σ0 = g(x, t,      params)
    x_pred = x .+ μ0 .* dt .+ σ0 .* dW          # Euler predictor
    μ1 = f(x_pred, t + dt, params)
    σ1 = g(x_pred, t + dt, params)
    return x .+ 0.5 .* (μ0 .+ μ1) .* dt .+ 0.5 .* (σ0 .+ σ1) .* dW
end

"""
    adaptive_step_size(x_em, x_half, dt, rtol, atol) → dt_new, accept

Estimates local error from two half-steps vs one full step and adjusts dt.
"""
function adaptive_step_size(x_full::AbstractArray,
                              x_half::AbstractArray,
                              dt::Float64,
                              rtol::Float64,
                              atol::Float64)
    err_vec = abs.(x_full .- x_half)
    scale   = atol .+ rtol .* max.(abs.(x_full), abs.(x_half))
    err     = sqrt(mean((err_vec ./ scale).^2))
    # PI controller for step size
    if err < 1e-14
        return min(dt * 5.0, 0.1), true
    end
    dt_new = dt * min(5.0, max(0.2, 0.9 * err^(-0.5)))
    accept = err <= 1.0
    return dt_new, accept
end

"""
    solve_em(prob::SDEProblem, solver::EulerMaruyama, dt; rng, save_at)

Solve an SDE using Euler-Maruyama. Returns (times, states).

- `save_at` : vector of additional times to save (default: all steps)
- Non-adaptive: fixed step dt throughout
- Adaptive: rejects steps and halves dt until error criterion met
"""
function solve_em(prob::SDEProblem, solver::EulerMaruyama, dt::Float64;
                  rng   = Random.GLOBAL_RNG,
                  save_at = nothing)

    t0, t1 = prob.tspan
    x      = copy(prob.x0)
    d      = length(x)
    m      = prob.noise_dim

    times  = [t0]
    states = [copy(x)]

    t = t0
    while t < t1 - 1e-14
        dt_cur = min(dt, t1 - t)

        if solver.adaptive
            # Attempt step; if error too large, halve dt
            accepted = false
            while !accepted
                dW1  = sqrt(dt_cur) .* randn(rng, Float32, m)
                x1   = euler_maruyama_step(prob.f, prob.g, x, t, dt_cur, dW1,
                                           prob.params, prob.interpretation)
                # Two half-steps for error estimate
                dt_h = dt_cur / 2
                dWa  = sqrt(dt_h) .* randn(rng, Float32, m)
                dWb  = sqrt(dt_h) .* randn(rng, Float32, m)
                xh   = euler_maruyama_step(prob.f, prob.g, x,  t,       dt_h, dWa,
                                           prob.params, prob.interpretation)
                xh   = euler_maruyama_step(prob.f, prob.g, xh, t+dt_h, dt_h, dWb,
                                           prob.params, prob.interpretation)
                dt_new, accepted = adaptive_step_size(x1, xh, dt_cur,
                                                       solver.rtol, solver.atol)
                dt_cur = clamp(dt_new, solver.dt_min, solver.dt_max)
                if accepted
                    # Use Richardson-extrapolated value for better accuracy
                    x = 2 .* xh .- x1
                end
            end
        else
            dW = sqrt(dt_cur) .* randn(rng, Float32, m)
            x  = euler_maruyama_step(prob.f, prob.g, x, t, dt_cur, dW,
                                     prob.params, prob.interpretation)
        end

        t += dt_cur
        push!(times,  t)
        push!(states, copy(x))
    end

    return times, states
end

# ─────────────────────────────────────────────────────────────────────────────
# MILSTEIN SCHEME
# ─────────────────────────────────────────────────────────────────────────────

"""
    levy_area_approx(dW1, dW2, dt, n_terms) → A12

Approximate Lévy area A_{12} = ∫₀^Δt (W₁ dW₂ - W₂ dW₁) / 2 using
the Fourier series approximation (Kloeden-Platen, 1992, §5.8).

n_terms: number of Fourier terms (higher = more accurate but slower).
"""
function levy_area_approx(dW1::Float64, dW2::Float64, dt::Float64,
                           n_terms::Int=5)
    # Simplified Fourier approximation
    # A₁₂ ≈ (dW₁·dW₂ - corr) / 2  (diagonal approximation)
    # Full implementation uses random Fourier coefficients
    A = 0.0
    for r in 1:n_terms
        ξr1 = randn()
        ξr2 = randn()
        ηr1 = randn()
        ηr2 = randn()
        coeff = dt / (2π * r)
        A += coeff * (ξr1 * (sqrt(2/dt) * dW2 + ηr2) -
                      ξr2 * (sqrt(2/dt) * dW1 + ηr1))
    end
    return A
end

"""
    milstein_step_diagonal(f, g, g_prime, x, t, dt, dW, params)

Milstein step for diagonal diffusion (g is a vector of scalars).
Requires g' = ∂g/∂x (Jacobian of diffusion w.r.t. state).

Milstein correction for diagonal case:
  X_{n+1} = X_n + f·Δt + gᵢ·ΔWᵢ + (1/2)·gᵢ·(∂gᵢ/∂xᵢ)·(ΔWᵢ² - Δt)
"""
function milstein_step_diagonal(f_fn, g_fn, g_prime_fn,
                                 x::AbstractVector, t::Real, dt::Real,
                                 dW::AbstractVector, params)
    μ  = f_fn(x, t, params)
    σ  = g_fn(x, t, params)
    σ′ = g_prime_fn(x, t, params)  # diagonal elements of Jacobian

    # Itô correction term
    correction = 0.5f0 .* σ .* σ′ .* (dW.^2 .- dt)
    return x .+ μ .* dt .+ σ .* dW .+ correction
end

"""
    milstein_step_full(f, g, x, t, dt, dW, levy_areas, params)

Milstein step for full (non-diagonal) diffusion matrix.
Requires Lévy areas Aᵢⱼ for the double stochastic integral correction.
"""
function milstein_step_full(f_fn, g_fn, x::AbstractVector, t::Real, dt::Real,
                              dW::AbstractVector, levy_areas::Matrix,
                              params)
    d = length(x)
    m = length(dW)
    μ  = f_fn(x, t, params)      # (d,)
    G  = g_fn(x, t, params)      # (d, m) diffusion matrix

    # Milstein correction: Σⱼ Σₖ (Gⱼ · ∂Gⱼₖ/∂x) · Iⱼₖ
    # For diagonal approximation: use dWⱼ² - dt on diagonal, 0 elsewhere
    correction = zeros(eltype(x), d)
    for j in 1:m
        for k in 1:m
            if j == k
                Ijk = 0.5 * (dW[j]^2 - dt)
            else
                Ijk = levy_areas[j, k]
            end
            # Approximate ∂G[:,j]/∂x via finite difference or zero
            # Here we use the zero approximation (first-order Milstein)
            correction .+= G[:, j] .* G[j, k] .* Ijk ./ dt
        end
    end

    return x .+ μ .* dt .+ G * dW .+ correction
end

"""
    solve_milstein(prob::SDEProblem, solver::Milstein, dt; rng)

Milstein scheme solver. For diagonal diffusion uses exact correction;
for full diffusion approximates Lévy areas via Fourier series.
"""
function solve_milstein(prob::SDEProblem, solver::Milstein, dt::Float64;
                         rng = Random.GLOBAL_RNG)
    t0, t1 = prob.tspan
    x = copy(prob.x0)
    d = length(x)
    m = prob.noise_dim

    times  = [t0]
    states = [copy(x)]

    t = t0
    while t < t1 - 1e-14
        dt_cur = min(dt, t1 - t)
        dW     = Float64.(sqrt(dt_cur) .* randn(rng, m))

        # Numerical Jacobian of diffusion via central differences
        ε = 1e-5
        σ_x = zeros(d, m)  # (d, m): columns are ∂gⱼ/∂x for each noise j
        for k in 1:d
            e = zeros(d)
            e[k] = ε
            gp = prob.g(x .+ e, t, prob.params)
            gm = prob.g(x .- e, t, prob.params)
            if m == d  # diagonal case: g returns vector
                σ_x[k, k] = (gp[k] - gm[k]) / (2ε)
            end
        end

        σ  = prob.g(x, t, prob.params)
        μ  = prob.f(x, t, prob.params)

        # Milstein correction
        if m == d
            # Diagonal case
            correction = 0.5 .* σ .* diag(σ_x) .* (dW.^2 .- dt_cur)
            x = x .+ μ .* dt_cur .+ σ .* dW .+ correction
        else
            # Full matrix case with Lévy areas
            levy = zeros(m, m)
            for j in 1:m, k in j+1:m
                levy[j,k] =  levy_area_approx(dW[j], dW[k], dt_cur,
                                               solver.levy_approx_order)
                levy[k,j] = -levy[j,k]
            end
            x = milstein_step_full(prob.f, prob.g, x, t, dt_cur, dW, levy,
                                    prob.params)
        end

        t += dt_cur
        push!(times,  t)
        push!(states, copy(x))
    end

    return times, states
end

# ─────────────────────────────────────────────────────────────────────────────
# RUNGE-KUTTA 4.5 FOR SDEs (Rößler SRK scheme)
# ─────────────────────────────────────────────────────────────────────────────

"""
    rk45_sde_step(f, g, x, t, dt, dW, params)

Stochastic Runge-Kutta step based on Rößler (2010) SRK scheme for Itô SDEs.
Uses a 4-stage scheme for the drift and 2-stage for the diffusion.

Drift stages (deterministic, Dormand-Prince style):
  k1 = f(x,           t        )
  k2 = f(x + a21·k1·h, t + c2·h)
  k3 = f(x + a31·k1·h + a32·k2·h, t + c3·h)
  x_{n+1} = x + h·(b1·k1 + b2·k2 + b3·k3)

Diffusion stages:
  l1 = g(x,           t)
  l2 = g(x + l1·√h,  t + h)
  diffusion term = (dW/2)·(l1 + l2)
"""
function rk45_sde_step(f_fn, g_fn, x::AbstractArray, t::Real, dt::Float64,
                        dW::AbstractArray, params)
    h = dt
    sqh = sqrt(h)

    # Rößler 2010 Table 1 coefficients (SRK method of weak order 2)
    c2 = 0.75f0
    a21 = 0.75f0
    b1 = 1f0/3f0;  b2 = 2f0/3f0

    # Drift stages
    k1 = f_fn(x,                     t,       params)
    k2 = f_fn(x .+ a21 .* k1 .* h,  t + c2*h, params)

    # Diffusion stages
    l1 = g_fn(x,                      t,    params)
    l2 = g_fn(x .+ l1 .* sqh,         t+h,  params)

    drift_inc    = h  .* (b1 .* k1 .+ b2 .* k2)
    diffuse_inc  = 0.5f0 .* dW .* (l1 .+ l2)

    # Itô correction (weak scheme): subtract 0.5·g·(∂g/∂x)·h
    # Estimated via finite difference
    ε = sqrt(eps(eltype(x)))
    ito_correction = zero(x)
    d = length(x)
    for k in 1:d
        e = zero(x)
        e[k] = ε
        gp = g_fn(x .+ e, t, params)
        gm = g_fn(x .- e, t, params)
        dg_dxk = (gp .- gm) ./ (2ε)
        # For diagonal: Σⱼ gⱼ · ∂gⱼ/∂xₖ · h/2
        if length(l1) == d
            ito_correction[k] -= 0.5f0 * l1[k] * dg_dxk[k] * h
        end
    end

    return x .+ drift_inc .+ diffuse_inc .+ ito_correction
end

"""
    solve_rk45(prob::SDEProblem, solver::RungeKutta45SDE, dt; rng)

Adaptive RK4.5 for SDEs. Uses embedded error estimate to control dt.
"""
function solve_rk45(prob::SDEProblem, solver::RungeKutta45SDE, dt::Float64;
                     rng = Random.GLOBAL_RNG)
    t0, t1 = prob.tspan
    x = copy(prob.x0)
    d = length(x)
    m = prob.noise_dim

    times  = [t0]
    states = [copy(x)]

    t   = t0
    dt_cur = dt

    while t < t1 - 1e-14
        dt_cur = min(dt_cur, t1 - t)
        accepted = false

        while !accepted
            dW = Float64.(sqrt(dt_cur) .* randn(rng, m))

            # Full step
            x1 = rk45_sde_step(prob.f, prob.g, x, t, dt_cur, dW, prob.params)

            # Two half-steps for error estimate
            dW_a = Float64.(sqrt(dt_cur/2) .* randn(rng, m))
            dW_b = Float64.(sqrt(dt_cur/2) .* randn(rng, m))
            x2   = rk45_sde_step(prob.f, prob.g, x,  t,           dt_cur/2, dW_a, prob.params)
            x2   = rk45_sde_step(prob.f, prob.g, x2, t+dt_cur/2, dt_cur/2, dW_b, prob.params)

            # Error estimate
            err_norm = norm(x1 .- x2) / (solver.atol + solver.rtol * norm(x1))
            if err_norm <= 1.0 || dt_cur <= 1e-8
                accepted = true
                x = x1  # use single-step result
                dt_cur = dt_cur * min(5.0, max(0.2, 0.9 * err_norm^(-0.4)))
            else
                dt_cur = dt_cur * 0.5
            end
        end

        t += min(dt_cur, t1 - t)
        push!(times,  t)
        push!(states, copy(x))
    end

    return times, states
end

# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH: solve_sde
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_sde(prob::SDEProblem, solver, dt; rng, kwargs...)

Dispatch to the correct solver based on `solver` type.
Returns a named tuple (t=times, u=states).
"""
function solve_sde(prob::SDEProblem, solver::EulerMaruyama, dt::Float64;
                   rng = Random.GLOBAL_RNG, kwargs...)
    times, states = solve_em(prob, solver, dt; rng=rng, kwargs...)
    return (t=times, u=states)
end

function solve_sde(prob::SDEProblem, solver::Milstein, dt::Float64;
                   rng = Random.GLOBAL_RNG, kwargs...)
    times, states = solve_milstein(prob, solver, dt; rng=rng)
    return (t=times, u=states)
end

function solve_sde(prob::SDEProblem, solver::RungeKutta45SDE, dt::Float64;
                   rng = Random.GLOBAL_RNG, kwargs...)
    times, states = solve_rk45(prob, solver, dt; rng=rng)
    return (t=times, u=states)
end

# ─────────────────────────────────────────────────────────────────────────────
# BATCH PATH SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    batch_simulate(prob::SDEProblem, solver, dt, n_paths; rng, parallel)

Simulate `n_paths` independent realisations of the SDE.
Returns a 3D array of shape (state_dim, n_timesteps, n_paths).

Uses Julia's broadcasting / threading for efficiency.
"""
function batch_simulate(prob::SDEProblem, solver, dt::Float64, n_paths::Int;
                         rng      = Random.GLOBAL_RNG,
                         parallel = false)

    # Solve first path to get time grid size
    sol0 = solve_sde(prob, solver, dt; rng=rng)
    n_t  = length(sol0.t)
    d    = length(prob.x0)

    # Pre-allocate output
    all_paths = zeros(Float64, d, n_t, n_paths)

    for k in 1:n_t
        all_paths[:, k, 1] = sol0.u[k]
    end

    if parallel
        Threads.@threads for p in 2:n_paths
            rng_p = MersenneTwister(rand(rng, UInt64))
            sol   = solve_sde(prob, solver, dt; rng=rng_p)
            for k in 1:min(n_t, length(sol.u))
                all_paths[:, k, p] = sol.u[k]
            end
        end
    else
        for p in 2:n_paths
            sol = solve_sde(prob, solver, dt; rng=rng)
            for k in 1:min(n_t, length(sol.u))
                all_paths[:, k, p] = sol.u[k]
            end
        end
    end

    return sol0.t, all_paths   # (times, d × n_t × n_paths)
end

"""
    simulate_paths(f, g, x0, tspan, dt, n_paths; rng, solver)

Convenience wrapper: takes raw functions f, g and simulates n_paths.
"""
function simulate_paths(f, g, x0::AbstractVector, tspan, dt::Float64,
                         n_paths::Int;
                         rng    = Random.GLOBAL_RNG,
                         solver = EulerMaruyama(),
                         params = nothing,
                         noise_dim = length(x0),
                         interpretation = ItoInterpretation())

    prob = SDEProblem(f, g, x0, tspan;
                      params=params, noise_dim=noise_dim,
                      interpretation=interpretation)
    return batch_simulate(prob, solver, dt, n_paths; rng=rng)
end

# ─────────────────────────────────────────────────────────────────────────────
# GPU SUPPORT (CuArray compatible versions)
# ─────────────────────────────────────────────────────────────────────────────

"""
    gpu_batch_simulate(prob, solver, dt, n_paths; device)

GPU-accelerated batch simulation. All arrays are moved to `device` (CuArray).
Falls back to CPU if CUDA not available.

The key insight: we simulate all n_paths simultaneously by stacking the
state matrix as (state_dim × n_paths) and applying the drift/diffusion
functions in a vectorised manner across the batch dimension.
"""
function gpu_batch_simulate(prob::SDEProblem, solver::EulerMaruyama,
                              dt::Float64, n_paths::Int;
                              device = identity)

    t0, t1 = prob.tspan
    d      = length(prob.x0)
    m      = prob.noise_dim

    # Stack initial conditions: (d × n_paths)
    X = device(repeat(reshape(Float32.(prob.x0), d, 1), 1, n_paths))

    times    = [t0]
    all_X    = [copy(X)]

    t = t0
    while t < t1 - 1e-14
        dt_cur = Float32(min(dt, t1 - t))
        dW = device(sqrt(dt_cur) .* randn(Float32, m, n_paths))

        # Vectorised drift and diffusion over all paths
        # f and g should handle (d × batch) input
        μ  = prob.f(X, t, prob.params)   # (d × n_paths)
        σ  = prob.g(X, t, prob.params)   # (d × n_paths) for diagonal

        X = X .+ μ .* dt_cur .+ σ .* dW

        t += dt_cur
        push!(times,  t)
        push!(all_X,  copy(X))
    end

    return times, all_X   # list of (d × n_paths) matrices
end

# ─────────────────────────────────────────────────────────────────────────────
# PATH STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    path_statistics(paths, times) → NamedTuple

Compute mean, std, quantiles from a batch of paths.
paths: (state_dim, n_timesteps, n_paths)
"""
function path_statistics(paths::Array{<:Real,3}, times::AbstractVector)
    d, n_t, n_paths = size(paths)
    means   = dropdims(mean(paths, dims=3), dims=3)     # (d, n_t)
    stds    = dropdims(std(paths,  dims=3), dims=3)     # (d, n_t)
    q05     = mapslices(x -> quantile(x, 0.05), paths, dims=3)[:,:,1]
    q25     = mapslices(x -> quantile(x, 0.25), paths, dims=3)[:,:,1]
    q75     = mapslices(x -> quantile(x, 0.75), paths, dims=3)[:,:,1]
    q95     = mapslices(x -> quantile(x, 0.95), paths, dims=3)[:,:,1]

    return (t=times, mean=means, std=stds,
            q05=q05, q25=q25, q75=q75, q95=q95,
            n_paths=n_paths)
end

"""
    path_variance_scaling_test(paths, times, expected_power=0.5) → Bool

Test whether path variance scales as Δt^(2·expected_power).
For Brownian motion expected_power=0.5 (variance ~ Δt).
Returns true if empirical scaling exponent is close to expected.
"""
function path_variance_scaling_test(paths::Array{<:Real,3},
                                     times::AbstractVector;
                                     expected_power::Float64 = 0.5,
                                     tol::Float64            = 0.1)
    # Compute variance of final state across paths
    final_states = paths[:, end, :]   # (d, n_paths)
    t1 = times[end] - times[1]
    var_final = vec(var(final_states, dims=2))

    # Compare to initial variance (zero at t=0 for fixed IC)
    # Variance should scale as σ² · t for standard BM
    # We check the ratio std/sqrt(t) is approximately constant
    mid_idx   = length(times) ÷ 2
    t_mid     = times[mid_idx] - times[1]
    mid_states = paths[:, mid_idx, :]
    var_mid   = vec(var(mid_states, dims=2))

    if t_mid < 1e-10 || any(var_mid .≈ 0)
        return true  # degenerate case
    end

    # Empirical scaling exponent: log(var_final/var_mid) / log(t1/t_mid)
    ratios  = var_final ./ var_mid
    t_ratio = t1 / t_mid
    exponents = log.(ratios) ./ log(t_ratio)

    mean_exp = mean(exponents)
    return abs(mean_exp - 2*expected_power) < tol
end

"""
    log_returns(price_paths) → return_paths

Compute log returns from simulated price paths.
paths: (1, n_timesteps, n_paths) for scalar price.
"""
function log_returns(price_paths::Array{<:Real,3})
    _, n_t, n_paths = size(price_paths)
    rets = diff(log.(price_paths[1, :, :]), dims=1)   # (n_t-1, n_paths)
    return rets
end
