"""
adjoint_sensitivity.jl — Gradient computation through SDE solvers

Implements:
  1. Continuous adjoint method (neural ODE / SDE adjoint)
  2. Discrete adjoint via reverse-mode AD (Zygote)
  3. Variance reduction via control variates
  4. Importance sampling for tail-event gradients
  5. Stochastic adjoint interpolation

References:
  - Li et al. (2020) "Scalable gradients for stochastic differential equations"
  - Kidger et al. (2021) "Neural SDEs as infinite-dimensional GANs"
  - Chen et al. (2018) "Neural ordinary differential equations"
"""

using LinearAlgebra
using Statistics
using Random
using Zygote

# ─────────────────────────────────────────────────────────────────────────────
# CONTINUOUS ADJOINT FOR SDEs
# ─────────────────────────────────────────────────────────────────────────────

"""
    SDEAdjointState

Augmented state for the continuous adjoint ODE:
  (x, a, θ_grad) where:
  - x      : forward SDE state at time t
  - a      : adjoint variable a(t) = dL/dx(t)
  - θ_grad : accumulated gradient w.r.t. parameters
"""
struct SDEAdjointState{T}
    x      :: Vector{T}
    a      :: Vector{T}
    θ_grad :: Vector{T}
end

"""
    continuous_adjoint(f, g, θ, x_path, t_path, dL_dxT; n_params)

Compute dL/dθ via the continuous stochastic adjoint method.

Arguments:
  - f       : drift function f(x, t, θ)
  - g       : diffusion function g(x, t, θ)
  - θ       : parameter vector (flattened)
  - x_path  : forward path {x(t₀), ..., x(T)} from the SDE solver
  - t_path  : corresponding times
  - dL_dxT  : terminal loss gradient ∂L/∂x(T)

The adjoint SDE (backwards in time):
  da = -[∂f/∂x]ᵀ a dt - [∂g/∂x]ᵀ a dW̄  (reversed Brownian)

Parameter gradient:
  dθ_grad = -[∂f/∂θ]ᵀ a dt - [∂g/∂θ]ᵀ a dW̄
"""
function continuous_adjoint(f, g, θ::AbstractVector,
                              x_path::Vector{<:AbstractVector},
                              t_path::AbstractVector,
                              dL_dxT::AbstractVector;
                              dt_adjoint::Float64 = 1e-3,
                              rng = Random.GLOBAL_RNG)

    T      = t_path[end]
    t0     = t_path[1]
    n_θ    = length(θ)
    d      = length(x_path[1])

    # Initialise adjoint at terminal time
    a      = copy(dL_dxT)
    θ_grad = zeros(n_θ)

    # Backward integration using Euler-Maruyama on adjoint SDE
    # We use the stored forward path for interpolation
    n_steps = length(t_path) - 1

    for k in n_steps:-1:1
        t1_k  = t_path[k+1]
        t0_k  = t_path[k]
        Δt    = t1_k - t0_k
        x_k   = x_path[k]   # forward state at t_k (used for Jacobians)

        # Jacobians via finite differences
        ε = 1e-5

        # ∂f/∂x at (x_k, t1_k)
        df_dx = zeros(d, d)
        for j in 1:d
            e = zeros(d); e[j] = ε
            df_dx[:, j] = (f(x_k .+ e, t1_k, θ) .- f(x_k .- e, t1_k, θ)) ./ (2ε)
        end

        # ∂g/∂x at (x_k, t1_k) — diagonal case assumed for efficiency
        dg_dx = zeros(d, d)
        for j in 1:d
            e = zeros(d); e[j] = ε
            dg_dx[:, j] = (g(x_k .+ e, t1_k, θ) .- g(x_k .- e, t1_k, θ)) ./ (2ε)
        end

        # ∂f/∂θ at (x_k, t1_k)
        df_dθ = zeros(d, n_θ)
        for j in 1:n_θ
            e = zeros(n_θ); e[j] = ε
            θp = θ .+ e; θm = θ .- e
            df_dθ[:, j] = (f(x_k, t1_k, θp) .- f(x_k, t1_k, θm)) ./ (2ε)
        end

        # ∂g/∂θ at (x_k, t1_k)
        dg_dθ = zeros(d, n_θ)
        for j in 1:n_θ
            e = zeros(n_θ); e[j] = ε
            θp = θ .+ e; θm = θ .- e
            dg_dθ[:, j] = (g(x_k, t1_k, θp) .- g(x_k, t1_k, θm)) ./ (2ε)
        end

        # Brownian motion increment (reversed — use fresh noise)
        dW = sqrt(Δt) .* randn(rng, d)

        # Adjoint update (backward Euler-Maruyama on adjoint SDE)
        # da = -(∂f/∂x)ᵀ a dt - (∂g/∂x)ᵀ a dW
        da_drift = -df_dx' * a
        da_diff  = -dg_dx' * a

        a = a .+ da_drift .* Δt .+ da_diff .* dW

        # Accumulate parameter gradient
        # dθ/dt = -(∂f/∂θ)ᵀ a - (∂g/∂θ)ᵀ a · dW/dt
        θ_grad .+= -(df_dθ' * a) .* Δt .- (dg_dθ' * a) .* dW
    end

    return θ_grad, a  # a(t0) = dL/dx(t0)
end

# ─────────────────────────────────────────────────────────────────────────────
# DISCRETE ADJOINT VIA ZYGOTE
# ─────────────────────────────────────────────────────────────────────────────

"""
    discrete_adjoint(loss_fn, model, x0, tspan, dt; solver)

Compute gradients of `loss_fn` w.r.t. model parameters using Zygote's
reverse-mode AD, differentiating through the discrete SDE steps.

This approach stores all intermediate states (checkpointing) and backpropagates
through each Euler-Maruyama step. Memory cost is O(n_steps · d).

Arguments:
  - loss_fn : function(trajectory) → scalar loss
  - model   : LatentSDE or similar Flux model
  - x0      : initial state
  - tspan   : (t0, t1)
  - dt      : time step

Returns (loss_value, gradients) where gradients is a Zygote gradient dict.
"""
function discrete_adjoint(loss_fn, model, x0::AbstractVector,
                            tspan::Tuple, dt::Float64;
                            rng         = Random.GLOBAL_RNG,
                            n_samples   = 1)

    t0, t1 = tspan
    n_steps = ceil(Int, (t1 - t0) / dt)
    d = length(x0)

    # Pre-generate Brownian increments (outside AD tape for efficiency)
    noise = [sqrt(dt) .* randn(rng, Float32, d) for _ in 1:n_steps]

    function forward_pass(ps)
        Flux.loadparams!(model, ps)
        x = Float32.(x0)
        t = Float32(t0)
        trajectory = [x]
        for k in 1:n_steps
            dt_k = Float32(min(dt, t1 - t))
            μ    = drift_at(model, x, t)
            σ    = diffusion_at(model, x, t)
            dW   = noise[k][1:d]
            x    = x .+ μ .* dt_k .+ σ .* dW
            t   += dt_k
            push!(trajectory, x)
        end
        return loss_fn(trajectory)
    end

    ps_flat = Flux.params(model)
    loss_val, grads = Zygote.withgradient(() -> begin
        x   = Float32.(x0)
        t   = Float32(t0)
        traj = [x]
        for k in 1:n_steps
            dt_k = Float32(min(dt, t1 - t))
            μ    = drift_at(model, x, t)
            σ    = diffusion_at(model, x, t)
            dW   = noise[k][1:d]
            x    = x .+ μ .* dt_k .+ σ .* dW
            t   += dt_k
            push!(traj, x)
        end
        loss_fn(traj)
    end, ps_flat)

    return loss_val, grads
end

"""
    discrete_adjoint_mc(loss_fn, model, x0, tspan, dt, n_mc; rng)

Monte Carlo version: average gradient over n_mc independent paths.
Reduces gradient variance at the cost of n_mc forward/backward passes.
"""
function discrete_adjoint_mc(loss_fn, model, x0::AbstractVector,
                               tspan::Tuple, dt::Float64, n_mc::Int;
                               rng = Random.GLOBAL_RNG)

    total_loss = 0.0
    accum_grads = nothing

    ps_flat = Flux.params(model)

    for _ in 1:n_mc
        lv, grads = discrete_adjoint(loss_fn, model, x0, tspan, dt; rng=rng)
        total_loss += lv

        if accum_grads === nothing
            accum_grads = grads
        else
            for p in ps_flat
                if grads[p] !== nothing
                    accum_grads[p] .+= grads[p]
                end
            end
        end
    end

    # Average
    total_loss /= n_mc
    for p in ps_flat
        if accum_grads[p] !== nothing
            accum_grads[p] ./= n_mc
        end
    end

    return total_loss, accum_grads
end

# ─────────────────────────────────────────────────────────────────────────────
# VARIANCE REDUCTION: CONTROL VARIATES
# ─────────────────────────────────────────────────────────────────────────────

"""
    control_variate_gradient(loss_fn, baseline_fn, model, x0, tspan, dt,
                              n_samples; rng)

Reduce gradient variance using control variates.

The control variate is h(τ) where E[h(τ)] is known (zero for the standard
choice h = f(x) - E[f(x)] over the prior).

Gradient estimate:
  ∇̂L = (1/N) Σᵢ [(L(τᵢ) - c·h(τᵢ)) · ∇ log p(τᵢ | θ)]

Optimal c = Cov(L, h·∇logp) / Var(h·∇logp) (estimated from samples).

Here we use a simpler baseline: subtract the running mean of losses.
"""
function control_variate_gradient(loss_fn, model, x0::AbstractVector,
                                   tspan::Tuple, dt::Float64,
                                   n_samples::Int;
                                   rng        = Random.GLOBAL_RNG,
                                   baseline_decay = 0.9)

    d = length(x0)
    t0, t1 = tspan
    n_steps = ceil(Int, (t1 - t0) / dt)
    ps_flat = Flux.params(model)

    # Collect loss values and raw gradients
    losses     = zeros(n_samples)
    all_grads  = Vector{Any}(undef, n_samples)

    for i in 1:n_samples
        noise = [sqrt(dt) .* randn(rng, Float32, d) for _ in 1:n_steps]
        lv, grads = Zygote.withgradient(() -> begin
            x   = Float32.(x0)
            t   = Float32(t0)
            for k in 1:n_steps
                dt_k = Float32(min(dt, t1 - t))
                μ = drift_at(model, x, t)
                σ = diffusion_at(model, x, t)
                x = x .+ μ .* dt_k .+ σ .* noise[k][1:d]
                t += dt_k
            end
            loss_fn([x])
        end, ps_flat)

        losses[i]    = lv
        all_grads[i] = grads
    end

    # Control variate: subtract mean loss (reduces variance in policy gradient)
    baseline = mean(losses)
    adjusted_losses = losses .- baseline

    # Average adjusted gradients
    final_grads_dict = Dict{Any, Any}()
    for p in ps_flat
        g_stack = []
        for i in 1:n_samples
            if all_grads[i][p] !== nothing
                push!(g_stack, adjusted_losses[i] .* all_grads[i][p])
            end
        end
        if !isempty(g_stack)
            final_grads_dict[p] = mean(g_stack)
        end
    end

    return mean(losses), final_grads_dict, baseline
end

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANCE SAMPLING FOR TAIL EVENTS
# ─────────────────────────────────────────────────────────────────────────────

"""
    ImportanceSampler

Importance sampler that tilts the Brownian motion distribution toward
rare events (large negative returns, vol spikes, etc.).

The tilted measure Q shifts the drift by a vector h(x,t):
  dX = [f(x,t) + h(x,t)]dt + g(x,t)dŴ
where Ŵ is Brownian under Q.

The Radon-Nikodym derivative (likelihood ratio) is:
  dP/dQ = exp(-∫₀ᵀ h·g⁻¹dW - (1/2)∫₀ᵀ|h·g⁻¹|²dt)

Fields:
  - `shift_fn` : h(x, t) → shift vector
  - `target_percentile` : percentile of loss to target (e.g. 0.05 for VaR 5%)
"""
struct ImportanceSampler{F}
    shift_fn           :: F
    target_percentile  :: Float64
end

ImportanceSampler(shift_fn; target_percentile=0.05) =
    ImportanceSampler(shift_fn, target_percentile)

"""
    importance_sampling_gradient(loss_fn, model, sampler, x0, tspan, dt,
                                  n_paths; rng)

Estimate gradient focusing on tail events using importance sampling.

Returns (weighted_loss, gradients, effective_sample_size).
"""
function importance_sampling_gradient(loss_fn, model,
                                       sampler::ImportanceSampler,
                                       x0::AbstractVector,
                                       tspan::Tuple, dt::Float64,
                                       n_paths::Int;
                                       rng = Random.GLOBAL_RNG)
    d = length(x0)
    t0, t1 = tspan
    n_steps = ceil(Int, (t1 - t0) / dt)
    ps_flat = Flux.params(model)

    log_weights = zeros(n_paths)
    all_losses  = zeros(n_paths)
    all_grads   = Vector{Any}(undef, n_paths)

    for i in 1:n_paths
        t   = Float32(t0)
        x   = Float32.(x0)
        log_w = 0.0  # log Radon-Nikodym derivative accumulator

        noise_path = Vector{Vector{Float32}}(undef, n_steps)
        for k in 1:n_steps
            dt_k    = Float32(min(dt, t1 - t))
            h       = Float32.(sampler.shift_fn(x, t))
            σ       = diffusion_at(model, x, t)
            σ_inv   = 1f0 ./ (σ .+ 1f-8)
            # Tilted noise: dW_Q = dW_P + h·σ⁻¹·dt
            dW_P    = sqrt(dt_k) .* randn(rng, Float32, d)
            dW_Q    = dW_P .+ h .* σ_inv .* dt_k
            noise_path[k] = dW_Q
            # Log-weight update: Girsanov kernel
            log_w  -= dot(h .* σ_inv, dW_P) + 0.5 * sum((h .* σ_inv).^2) * dt_k
            # Update state under Q
            μ    = drift_at(model, x, t)
            x    = x .+ (μ .+ h) .* dt_k .+ σ .* dW_Q
            t   += dt_k
        end
        log_weights[i] = log_w

        # Compute loss and gradient at this path
        lv, grads = Zygote.withgradient(() -> begin
            xr  = Float32.(x0)
            tr  = Float32(t0)
            for k in 1:n_steps
                dt_k = Float32(min(dt, t1 - tr))
                μ    = drift_at(model, xr, tr)
                σ    = diffusion_at(model, xr, tr)
                xr   = xr .+ μ .* dt_k .+ σ .* noise_path[k][1:d]
                tr  += dt_k
            end
            loss_fn([xr])
        end, ps_flat)

        all_losses[i] = lv
        all_grads[i]  = grads
    end

    # Normalised importance weights (log-sum-exp for numerical stability)
    log_w_max  = maximum(log_weights)
    weights    = exp.(log_weights .- log_w_max)
    weights  ./= sum(weights)

    # Effective sample size (Kish formula)
    ess = 1.0 / sum(weights.^2)

    # Weighted loss and gradients
    weighted_loss = dot(weights, all_losses)
    final_grads   = Dict{Any, Any}()
    for p in ps_flat
        g_acc = nothing
        for i in 1:n_paths
            if all_grads[i][p] !== nothing
                contrib = weights[i] .* all_grads[i][p]
                g_acc = g_acc === nothing ? contrib : g_acc .+ contrib
            end
        end
        final_grads[p] = g_acc
    end

    return weighted_loss, final_grads, ess
end

# ─────────────────────────────────────────────────────────────────────────────
# ADJOINT INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    AdjointInterpolant

Stores forward path states between checkpoints, enabling
memory-efficient adjoint computation via checkpointed recomputation.

Between checkpoints, the forward trajectory is re-simulated from the
checkpoint state using the stored Brownian motion increments.
"""
struct AdjointInterpolant
    checkpoints :: Vector{Tuple{Float64, Vector{Float64}}}  # (t, x) pairs
    noise_cache :: Vector{Vector{Float64}}                   # dW for each step
    dt          :: Float64
    n_checkpoints :: Int
end

"""
    build_interpolant(prob, solver, dt, n_checkpoints; rng)

Run the forward SDE, store checkpoints and Brownian increments.
"""
function build_interpolant(prob::SDEProblem, dt::Float64,
                            n_checkpoints::Int;
                            rng = Random.GLOBAL_RNG)
    t0, t1 = prob.tspan
    d = length(prob.x0)
    n_steps = ceil(Int, (t1 - t0) / dt)
    checkpoint_freq = max(1, n_steps ÷ n_checkpoints)

    x = Float64.(prob.x0)
    t = t0
    checkpoints = [(t0, copy(x))]
    noise_cache = Vector{Float64}[]

    for k in 1:n_steps
        dt_k = min(dt, t1 - t)
        dW   = sqrt(dt_k) .* randn(rng, d)
        push!(noise_cache, dW)

        μ = prob.f(x, t, prob.params)
        σ = prob.g(x, t, prob.params)
        x = x .+ μ .* dt_k .+ σ .* dW
        t += dt_k

        if k % checkpoint_freq == 0 || k == n_steps
            push!(checkpoints, (t, copy(x)))
        end
    end

    return AdjointInterpolant(checkpoints, noise_cache, dt, n_checkpoints)
end

"""
    adjoint_interpolate(interp, t_query) → x(t_query)

Reconstruct the forward state at time t_query by rerunning from the
nearest earlier checkpoint.
"""
function adjoint_interpolate(interp::AdjointInterpolant, prob::SDEProblem,
                               t_query::Float64)
    # Find the last checkpoint before t_query
    cp_idx = findlast(cp -> cp[1] <= t_query, interp.checkpoints)
    cp_idx === nothing && (cp_idx = 1)

    t_cp, x_cp = interp.checkpoints[cp_idx]
    x = copy(x_cp)
    t = t_cp
    dt = interp.dt

    # Step index corresponding to checkpoint
    t0 = interp.checkpoints[1][1]
    step_start = round(Int, (t_cp - t0) / dt) + 1

    # Rerun from checkpoint to t_query using cached noise
    k = step_start
    while t < t_query - 1e-14 && k <= length(interp.noise_cache)
        dt_k = min(dt, t_query - t)
        dW   = interp.noise_cache[k][1:length(x)]
        μ    = prob.f(x, t, prob.params)
        σ    = prob.g(x, t, prob.params)
        x    = x .+ μ .* dt_k .+ σ .* dW .* sqrt(dt_k/dt)
        t   += dt_k
        k   += 1
    end
    return x
end

# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT CLIPPING & NORM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    clip_gradients!(grads, max_norm)

Clip gradient norms to prevent exploding gradients during adjoint backprop.
Uses global norm clipping: if ‖g‖ > max_norm, scale all gradients by max_norm/‖g‖.
"""
function clip_gradients!(grads, ps_flat, max_norm::Float64)
    total_norm = 0.0
    for p in ps_flat
        if grads[p] !== nothing
            total_norm += sum(grads[p].^2)
        end
    end
    total_norm = sqrt(total_norm)

    if total_norm > max_norm
        scale = max_norm / total_norm
        for p in ps_flat
            if grads[p] !== nothing
                grads[p] .*= scale
            end
        end
    end
    return total_norm
end

"""
    gradient_variance(grads_list, ps_flat) → Dict

Compute per-parameter gradient variance across a list of gradient dicts.
Useful for diagnosing high-variance gradient estimators.
"""
function gradient_variance(grads_list::Vector, ps_flat)
    result = Dict{Any, Float64}()
    for p in ps_flat
        samples = [g[p] for g in grads_list if g[p] !== nothing]
        isempty(samples) && continue
        flat_samples = hcat([vec(s) for s in samples]...)  # (n_params, n_samples)
        result[p] = mean(var(flat_samples, dims=2))
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# PATH LIKELIHOOD AND SCORE FUNCTION ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

"""
    path_log_likelihood(f, g, x_path, t_path, params, dt)

Approximate log p(x_{0:T} | θ) under the Euler-Maruyama discretisation.

log p ≈ Σₙ log N(x_{n+1}; x_n + f·Δt, g²·Δt)
       = -Σₙ [(x_{n+1} - x_n - f·Δt)² / (2g²Δt) + 0.5·log(2π·g²·Δt)]
"""
function path_log_likelihood(f, g, x_path::Vector{<:AbstractVector},
                               t_path::AbstractVector, params, dt::Float64)
    ll = 0.0
    n  = length(x_path) - 1

    for k in 1:n
        xk  = x_path[k]
        xk1 = x_path[k+1]
        tk  = t_path[k]
        Δt  = t_path[k+1] - tk

        μk  = f(xk, tk, params)
        σk  = g(xk, tk, params)

        residual = xk1 .- (xk .+ μk .* Δt)
        var_k    = (σk .^ 2) .* Δt

        ll += -0.5 * sum(residual.^2 ./ var_k) - 0.5 * sum(log.(2π .* var_k))
    end

    return ll
end

"""
    score_function_gradient(loss_fn, log_p_fn, model, trajectories, weights)

Score function (REINFORCE) gradient estimator.

∇_θ E[L(τ)] = E[L(τ) · ∇_θ log p(τ|θ)]

Arguments:
  - trajectories : vector of (x_path, t_path) tuples
  - weights      : optional importance weights
"""
function score_function_gradient(loss_fn, model,
                                  trajectories::Vector,
                                  t_paths::Vector;
                                  weights = nothing,
                                  rng     = Random.GLOBAL_RNG)

    n = length(trajectories)
    w = weights === nothing ? fill(1.0/n, n) : weights

    ps_flat    = Flux.params(model)
    total_grad = Dict{Any, Any}()

    for (i, (x_path, t_path)) in enumerate(zip(trajectories, t_paths))
        L_i = loss_fn(x_path)
        dt_i = t_path[2] - t_path[1]

        # Score: ∇_θ log p(τ|θ)
        _, score_grads = Zygote.withgradient(ps_flat) do
            path_log_likelihood(
                (x, t, p) -> drift_at(model, x, t),
                (x, t, p) -> diffusion_at(model, x, t),
                x_path, t_path, nothing, dt_i
            )
        end

        for p in ps_flat
            if score_grads[p] !== nothing
                contrib = w[i] * L_i .* score_grads[p]
                if haskey(total_grad, p)
                    total_grad[p] .+= contrib
                else
                    total_grad[p]  = contrib
                end
            end
        end
    end

    return total_grad
end
