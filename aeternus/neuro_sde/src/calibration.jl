"""
calibration.jl — Model calibration methods for Neural SDE models

Implements:
  1. Maximum Likelihood Estimation via adjoint-based gradient descent
  2. Generalised Method of Moments (GMM)
  3. Characteristic function matching via Carr-Madan FFT
  4. Training loop with Adam, LR scheduling, early stopping
  5. Validation loss tracking and overfitting detection
  6. Bootstrap confidence intervals

References:
  - Carr & Madan (1999) "Option valuation using the fast Fourier transform"
  - Hansen (1982) "Large sample properties of GMM estimators"
"""

using LinearAlgebra
using Statistics
using Random
using Distributions
using Flux
using Optimisers
using FFTW

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    TrainingConfig

Hyperparameters for the training loop.

Fields:
  - `n_epochs`       : total epochs
  - `batch_size`     : number of paths per gradient step
  - `learning_rate`  : initial learning rate
  - `lr_schedule`    : one of :constant, :cosine, :exponential, :reduce_on_plateau
  - `lr_decay`       : decay factor for exponential schedule
  - `patience`       : epochs without improvement before early stopping
  - `min_delta`      : minimum improvement to count as progress
  - `grad_clip`      : gradient clipping norm (Inf = no clip)
  - `print_every`    : print loss every N epochs
  - `n_mc_paths`     : Monte Carlo paths for loss estimation
  - `dt`             : SDE time step
"""
struct TrainingConfig
    n_epochs      :: Int
    batch_size    :: Int
    learning_rate :: Float64
    lr_schedule   :: Symbol
    lr_decay      :: Float64
    patience      :: Int
    min_delta     :: Float64
    grad_clip     :: Float64
    print_every   :: Int
    n_mc_paths    :: Int
    dt            :: Float64
end

function TrainingConfig(;
    n_epochs      = 200,
    batch_size    = 32,
    learning_rate = 1e-3,
    lr_schedule   = :cosine,
    lr_decay      = 0.95,
    patience      = 20,
    min_delta     = 1e-5,
    grad_clip     = 5.0,
    print_every   = 10,
    n_mc_paths    = 50,
    dt            = 1.0/252
)
    TrainingConfig(n_epochs, batch_size, learning_rate, lr_schedule,
                   lr_decay, patience, min_delta, grad_clip, print_every,
                   n_mc_paths, dt)
end

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION TRACKER
# ─────────────────────────────────────────────────────────────────────────────

"""
    ValidationTracker

Tracks train/validation losses, detects overfitting, manages early stopping.
"""
mutable struct ValidationTracker
    train_losses :: Vector{Float64}
    val_losses   :: Vector{Float64}
    best_val     :: Float64
    best_epoch   :: Int
    no_improve   :: Int
    patience     :: Int
    min_delta    :: Float64
end

function ValidationTracker(patience::Int=20, min_delta::Float64=1e-5)
    ValidationTracker(Float64[], Float64[], Inf, 0, 0, patience, min_delta)
end

"""
    update!(tracker, train_loss, val_loss, epoch) → should_stop
"""
function update!(tracker::ValidationTracker, train_loss::Real,
                  val_loss::Real, epoch::Int)
    push!(tracker.train_losses, Float64(train_loss))
    push!(tracker.val_losses,   Float64(val_loss))

    if val_loss < tracker.best_val - tracker.min_delta
        tracker.best_val   = val_loss
        tracker.best_epoch = epoch
        tracker.no_improve = 0
        return false
    else
        tracker.no_improve += 1
        return tracker.no_improve >= tracker.patience
    end
end

"""
    overfitting_score(tracker) → Float64

Returns (val_loss / train_loss) - 1.
Values > 0.1 indicate potential overfitting.
"""
function overfitting_score(tracker::ValidationTracker)
    isempty(tracker.val_losses) && return 0.0
    n = min(10, length(tracker.val_losses))
    recent_val   = mean(tracker.val_losses[end-n+1:end])
    recent_train = mean(tracker.train_losses[end-n+1:end])
    recent_train ≈ 0.0 && return 0.0
    return (recent_val / recent_train) - 1.0
end

# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULING
# ─────────────────────────────────────────────────────────────────────────────

"""
    lr_at_epoch(config, epoch) → lr

Compute learning rate at given epoch according to schedule.
"""
function lr_at_epoch(config::TrainingConfig, epoch::Int)
    lr0 = config.learning_rate
    T   = config.n_epochs

    if config.lr_schedule == :constant
        return lr0
    elseif config.lr_schedule == :cosine
        # Cosine annealing: lr = lr_min + 0.5(lr_max-lr_min)(1+cos(πt/T))
        lr_min = lr0 * 0.01
        return lr_min + 0.5 * (lr0 - lr_min) * (1 + cos(π * epoch / T))
    elseif config.lr_schedule == :exponential
        return lr0 * config.lr_decay^epoch
    elseif config.lr_schedule == :warmup_cosine
        warmup = div(T, 10)
        if epoch < warmup
            return lr0 * epoch / warmup
        else
            t     = epoch - warmup
            T_cos = T - warmup
            lr_min = lr0 * 0.01
            return lr_min + 0.5 * (lr0 - lr_min) * (1 + cos(π * t / T_cos))
        end
    else
        return lr0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# MAXIMUM LIKELIHOOD TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

"""
    train_model!(model, data, config; val_data, rng)

Main training loop using MLE via negative log-likelihood minimisation.

data: named tuple with fields depending on model type:
  - For return series: (returns=..., dt=..., T=...)
  - For option prices: (strikes=..., maturities=..., prices=..., S0=...)

Returns (final_loss, tracker).
"""
function train_model!(model, data, config::TrainingConfig;
                       val_data = nothing,
                       rng      = Random.GLOBAL_RNG,
                       loss_fn  = nothing)

    ps_flat  = Flux.params(model)
    optim    = Optimisers.Adam(config.learning_rate)
    opt_state = Optimisers.setup(optim, model)
    tracker  = ValidationTracker(config.patience, config.min_delta)

    best_params = deepcopy(Flux.params(model))

    for epoch in 1:config.n_epochs
        # Update learning rate
        lr_now = lr_at_epoch(config, epoch)
        Optimisers.adjust!(opt_state, lr_now)

        # Compute loss and gradients
        train_loss, grads = Zygote.withgradient(ps_flat) do
            if loss_fn !== nothing
                loss_fn(model, data, config, rng)
            else
                default_mle_loss(model, data, config, rng)
            end
        end

        # Gradient clipping
        if isfinite(config.grad_clip)
            total_norm = 0.0
            for p in ps_flat
                grads[p] !== nothing && (total_norm += sum(grads[p].^2))
            end
            total_norm = sqrt(total_norm)
            if total_norm > config.grad_clip
                scale = config.grad_clip / total_norm
                for p in ps_flat
                    grads[p] !== nothing && (grads[p] .*= scale)
                end
            end
        end

        # Parameter update
        Optimisers.update!(opt_state, model, grads)

        # Validation
        val_loss = if val_data !== nothing
            with_no_grad() do
                if loss_fn !== nothing
                    loss_fn(model, val_data, config, rng)
                else
                    default_mle_loss(model, val_data, config, rng)
                end
            end
        else
            train_loss
        end

        should_stop = update!(tracker, train_loss, val_loss, epoch)

        # Save best parameters
        if tracker.no_improve == 0
            best_params = deepcopy(Flux.params(model))
        end

        if epoch % config.print_every == 0
            @info "Epoch $epoch/$( config.n_epochs): train=$(round(train_loss,digits=4)), val=$(round(val_loss,digits=4)), lr=$(round(lr_now,sigdigits=3))"
            ov = overfitting_score(tracker)
            ov > 0.15 && @warn "  Possible overfitting: score=$(round(ov,digits=3))"
        end

        if should_stop
            @info "Early stopping at epoch $epoch (best val=$(round(tracker.best_val,digits=5)) at epoch $(tracker.best_epoch))"
            # Restore best parameters
            Flux.loadparams!(model, best_params)
            break
        end
    end

    return tracker.train_losses[end], tracker
end

"""
    with_no_grad(f)

Run f() without tracking gradients (for validation).
"""
function with_no_grad(f)
    Zygote.ignore() do
        f()
    end
end

"""
    default_mle_loss(model, data, config, rng) → Float32

Default MLE loss: negative path log-likelihood averaged over n_mc_paths.
"""
function default_mle_loss(model, data, config::TrainingConfig, rng)
    returns = Float32.(data.returns)
    dt      = Float64(get(data, :dt, config.dt))
    n       = length(returns)
    T       = n * dt
    tspan   = (0.0, T)

    # Simulate paths and compute average negative log-likelihood
    total_nll = 0.0f0
    n_paths   = config.n_mc_paths

    for _ in 1:n_paths
        x   = hasfield(typeof(data), :x0) ? Float32.(data.x0) : Float32[0.0]
        t   = Float32(0.0)

        path_ll = 0.0f0
        for k in 1:n
            dt_k = Float32(dt)
            μ_k  = drift_at(model, x, t)
            σ_k  = diffusion_at(model, x, t)

            # Gaussian log-likelihood for log return k
            r_k   = returns[k]
            μ_r   = (length(μ_k) == 1 ? μ_k[1] : μ_k[1]) * dt_k
            σ²_r  = (length(σ_k) == 1 ? σ_k[1] : σ_k[1])^2 * dt_k
            path_ll += -0.5f0 * (r_k - μ_r)^2 / σ²_r - 0.5f0 * log(2π * σ²_r)

            # Euler step for state update
            dW = Float32(sqrt(dt)) * Float32(randn(rng))
            if length(μ_k) == 1
                x = x .+ μ_k .* dt_k .+ σ_k .* dW
            else
                x = x .+ μ_k .* dt_k .+ σ_k .* dW
            end
            t += dt_k
        end
        total_nll -= path_ll
    end

    return total_nll / n_paths
end

# ─────────────────────────────────────────────────────────────────────────────
# MAXIMUM LIKELIHOOD ESTIMATION (standalone)
# ─────────────────────────────────────────────────────────────────────────────

"""
    mle_calibrate(model, returns; dt, n_epochs, lr, rng)

Calibrate model by maximising log-likelihood of observed return series.
Uses Adam optimiser with cosine LR schedule.

Returns (calibrated_model, loss_history).
"""
function mle_calibrate(model, returns::AbstractVector;
                         dt::Float64      = 1.0/252,
                         n_epochs::Int    = 300,
                         lr::Float64      = 1e-3,
                         batch_size::Int  = 64,
                         val_frac::Float64 = 0.2,
                         rng              = Random.GLOBAL_RNG)

    n       = length(returns)
    n_val   = round(Int, n * val_frac)
    n_train = n - n_val

    train_returns = returns[1:n_train]
    val_returns   = returns[n_train+1:end]

    data_train = (returns=train_returns, dt=dt)
    data_val   = (returns=val_returns,   dt=dt)

    config = TrainingConfig(n_epochs=n_epochs, learning_rate=lr,
                             batch_size=batch_size, lr_schedule=:cosine,
                             print_every=50, n_mc_paths=20, dt=dt)

    final_loss, tracker = train_model!(model, data_train, config;
                                       val_data=data_val, rng=rng)

    return model, tracker.train_losses
end

# ─────────────────────────────────────────────────────────────────────────────
# GMM CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    empirical_moments(returns, lags) → (moments, moment_names)

Compute moment conditions for GMM calibration:
  - Mean return
  - Variance of returns
  - Skewness of returns
  - Excess kurtosis
  - Autocovariance of squared returns (at given lags)
  - Autocovariance of absolute returns
"""
function empirical_moments(returns::AbstractVector, lags::Vector{Int}=[1,5,10,21])
    n    = length(returns)
    r    = Float64.(returns)
    r²   = r.^2
    |r|  = abs.(r)

    m = Float64[]
    names = String[]

    push!(m, mean(r));     push!(names, "mean_ret")
    push!(m, var(r));      push!(names, "var_ret")
    push!(m, skewness(r)); push!(names, "skew_ret")
    push!(m, kurtosis(r)); push!(names, "kurt_ret")

    for lag in lags
        if lag < n
            cov_r² = cov(r²[1:n-lag], r²[lag+1:n])
            cov_|r| = cov(|r|[1:n-lag], |r|[lag+1:n])
            push!(m, cov_r²);   push!(names, "acov_r²_$lag")
            push!(m, cov_|r|);  push!(names, "acov_|r|_$lag")
        end
    end

    return m, names
end

"""
    model_moments(model, T, dt, n_paths; rng) → Vector{Float64}

Simulate model moments matching those in empirical_moments.
"""
function model_moments(model, T::Float64, dt::Float64, n_paths::Int;
                        rng  = Random.GLOBAL_RNG,
                        lags = [1,5,10,21])
    all_returns = Float64[]

    for _ in 1:n_paths
        x = Float32[0.0]
        t = Float32(0.0)
        n_steps = ceil(Int, T / dt)
        for k in 1:n_steps
            dt_k = Float32(min(dt, T - (k-1)*dt))
            μ_k  = drift_at(model, x, t)
            σ_k  = diffusion_at(model, x, t)
            dW   = Float32(sqrt(dt_k)) * Float32(randn(rng))
            r_k  = μ_k[1] * dt_k + σ_k[1] * dW
            push!(all_returns, Float64(r_k))
            x = x .+ μ_k .* dt_k .+ σ_k .* dW
            t += dt_k
        end
    end

    return empirical_moments(all_returns, lags)[1]
end

"""
    gmm_calibrate(model, returns; dt, n_epochs, lr, W, rng)

Calibrate model by minimising GMM objective:
  J(θ) = [m_emp - m_model(θ)]' W [m_emp - m_model(θ)]

where W is the weighting matrix (identity, optimal, or diagonal).

Returns (calibrated_model, gmm_obj_history).
"""
function gmm_calibrate(model, returns::AbstractVector;
                         dt::Float64    = 1.0/252,
                         n_epochs::Int  = 100,
                         lr::Float64    = 5e-4,
                         W              = nothing,
                         n_sim_paths::Int = 200,
                         rng            = Random.GLOBAL_RNG)

    T   = length(returns) * dt
    lags = [1, 5, 10, 21]

    # Empirical moments
    m_emp, mnames = empirical_moments(returns, lags)
    n_moments = length(m_emp)

    # Weighting matrix (identity if not specified)
    W_mat = W === nothing ? Matrix{Float64}(I, n_moments, n_moments) : W

    ps_flat   = Flux.params(model)
    optim     = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optim, model)
    obj_hist  = Float64[]

    for epoch in 1:n_epochs
        loss_val, grads = Zygote.withgradient(ps_flat) do
            m_sim = model_moments(model, T, dt, min(n_sim_paths, 50); rng=rng, lags=lags)
            diff  = Float32.(m_emp .- m_sim)
            # GMM objective (scalar)
            Float32(dot(diff, Float32.(W_mat) * diff))
        end

        # Handle NaN/Inf gradients
        all_finite = all(p -> grads[p] === nothing || all(isfinite.(grads[p])), ps_flat)
        if !all_finite
            @warn "GMM: non-finite gradient at epoch $epoch, skipping update"
            push!(obj_hist, loss_val)
            continue
        end

        Optimisers.update!(opt_state, model, grads)
        push!(obj_hist, loss_val)

        epoch % 10 == 0 && @info "GMM epoch $epoch: J=$(round(loss_val,digits=6))"
    end

    return model, obj_hist
end

# ─────────────────────────────────────────────────────────────────────────────
# CARR-MADAN FFT CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    carr_madan_price(char_fn, K_vec, S0, r, T; N, η, α)

Price European call options using the Carr-Madan (1999) FFT method.

The modified call price:
  c_T(k) = exp(-αk)/π · ∫₀^∞ exp(-iuk) · ψ_T(u) du

where ψ_T(u) = exp(-rT)φ_T(u-(α+1)i) / (α² + α - u² + i(2α+1)u)
and φ_T is the characteristic function of log(S_T/S_0).

k = log(K/S0).

Returns vector of call prices at K_vec.
"""
function carr_madan_price(char_fn::Function, K_vec::AbstractVector,
                            S0::Float64, r::Float64, T::Float64;
                            N::Int = 4096,   # FFT grid size
                            η::Float64 = 0.25,  # integration step
                            α::Float64 = 1.5)   # damping factor

    # Grid setup
    λ_grid = 2π / (N * η)        # log-strike grid spacing
    b      = N * λ_grid / 2      # log-strike range bound
    k_vec  = -b .+ λ_grid .* (0:N-1)  # log-strike grid

    # Characteristic function evaluated on FFT grid
    u_vec  = η .* (0:N-1)
    ψ      = Vector{ComplexF64}(undef, N)

    for j in 1:N
        u = u_vec[j]
        v = u - (α + 1) * im  # shifted argument
        φ = char_fn(v)
        denom = α^2 + α - u^2 + im * (2α + 1) * u
        ψ[j]  = exp(-r*T) * φ / denom
    end

    # Apply Simpson's rule weights
    w = ones(N)
    w[1] = 1/3; w[end] = 1/3
    for j in 2:N-1; w[j] = j % 2 == 0 ? 2/3 : 4/3; end

    # FFT
    x  = ψ .* exp.(im .* b .* u_vec) .* w .* η
    fft_vals = real.(fft(x))

    # Undamp
    C_grid = exp.(-α .* k_vec) ./ π .* fft_vals

    # Interpolate to requested strikes
    log_K = log.(K_vec ./ S0)
    C_out = zeros(length(K_vec))
    for (i, lk) in enumerate(log_K)
        idx = searchsortedfirst(k_vec, lk)
        if idx == 1
            C_out[i] = C_grid[1]
        elseif idx > N
            C_out[i] = max(S0 - K_vec[i] * exp(-r*T), 0.0)  # intrinsic value
        else
            # Linear interpolation
            t_interp = (lk - k_vec[idx-1]) / (k_vec[idx] - k_vec[idx-1])
            C_out[i] = (1-t_interp) * C_grid[idx-1] + t_interp * C_grid[idx]
        end
        C_out[i] = max(C_out[i], 0.0)
    end

    return C_out
end

"""
    carr_madan_calibrate(model, strikes, maturities, market_prices, S0, r;
                          n_epochs, lr, rng)

Calibrate model by minimising squared error between model and market option prices,
using the Carr-Madan FFT pricer.

For models with known characteristic functions (NeuralHeston, JumpDiffusion).
"""
function carr_madan_calibrate(model, strikes::AbstractVector,
                                maturities::AbstractVector,
                                market_prices::AbstractMatrix,  # (n_strikes, n_maturities)
                                S0::Float64, r::Float64;
                                n_epochs::Int = 200,
                                lr::Float64   = 1e-3,
                                rng           = Random.GLOBAL_RNG)

    @assert size(market_prices) == (length(strikes), length(maturities))

    ps_flat   = Flux.params(model)
    optim     = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optim, model)
    loss_hist = Float64[]

    for epoch in 1:n_epochs
        loss_val, grads = Zygote.withgradient(ps_flat) do
            total_mse = 0.0f0
            for (j, T) in enumerate(maturities)
                # Get characteristic function for this model
                if model isa NeuralHeston
                    char_fn = u -> heston_characteristic_fn(model, u, T)
                elseif model isa JumpDiffusion
                    char_fn = u -> merton_characteristic_fn(model, u, T)
                else
                    break  # skip if no characteristic function
                end

                model_prices = carr_madan_price(char_fn, strikes, S0, r, T)
                diff = Float32.(model_prices) .- Float32.(market_prices[:, j])
                total_mse += sum(diff.^2) / length(diff)
            end
            total_mse / length(maturities)
        end

        Optimisers.update!(opt_state, model, grads)
        push!(loss_hist, loss_val)

        epoch % 20 == 0 && @info "FFT calib epoch $epoch: MSE=$(round(loss_val,sigdigits=4))"
    end

    return model, loss_hist
end

# ─────────────────────────────────────────────────────────────────────────────
# HESTON PARAMETER CALIBRATION (classical, closed-form)
# ─────────────────────────────────────────────────────────────────────────────

"""
    calibrate_heston_params(returns, V_proxy; dt, n_epochs, lr)

Calibrate classical Heston parameters (κ, θ, ξ, ρ) from observed return
and proxy variance series using MLE.

Uses log-likelihood of the Euler-Maruyama discretisation.
"""
function calibrate_heston_params(returns::AbstractVector,
                                   V_proxy::AbstractVector;
                                   dt::Float64   = 1.0/252,
                                   n_epochs::Int = 500,
                                   lr::Float64   = 1e-2,
                                   rng           = Random.GLOBAL_RNG)

    n = length(returns)
    @assert length(V_proxy) == n

    # Parameterised in log-space for positivity constraints
    # θ_raw = [log_κ, log_θ, log_ξ, atanh_ρ, log_μ]
    θ = [log(2.0), log(0.04), log(0.3), -0.85, log(0.05)]  # initial guess

    function neg_ll(θ_raw)
        κ   = exp(θ_raw[1])
        θ_v = exp(θ_raw[2])
        ξ   = exp(θ_raw[3])
        ρ   = tanh(θ_raw[4])
        μ_r = exp(θ_raw[5])

        ll = 0.0
        for k in 1:n-1
            V_k   = max(V_proxy[k],   1e-8)
            V_k1  = max(V_proxy[k+1], 1e-8)

            # Log return conditional mean and variance
            μ_logS  = (μ_r - 0.5*V_k) * dt
            σ²_logS = V_k * dt

            r_k     = returns[k]
            ll += -0.5*(r_k - μ_logS)^2/σ²_logS - 0.5*log(2π*σ²_logS)

            # Variance process LL
            μ_V    = κ*(θ_v - V_k)*dt
            σ²_V   = ξ^2 * V_k * dt
            σ²_V   = max(σ²_V, 1e-10)
            dV_obs = V_k1 - V_k
            ll += -0.5*(dV_obs - μ_V)^2/σ²_V - 0.5*log(2π*σ²_V)
        end
        return -ll
    end

    # Gradient descent
    optim  = Optimisers.Adam(lr)
    opt_st = Optimisers.setup(optim, θ)
    hist   = Float64[]

    for epoch in 1:n_epochs
        loss_val, grad = Zygote.withgradient(neg_ll, θ)
        Optimisers.update!(opt_st, θ, grad[1])
        push!(hist, loss_val)
        epoch % 50 == 0 && @info "Heston MLE epoch $epoch: NLL=$(round(loss_val,digits=2))"
    end

    κ_hat   = exp(θ[1])
    θ_hat   = exp(θ[2])
    ξ_hat   = exp(θ[3])
    ρ_hat   = tanh(θ[4])
    μ_hat   = exp(θ[5])

    @info "Calibrated Heston: κ=$(round(κ_hat,digits=3)), θ=$(round(θ_hat,digits=4)), ξ=$(round(ξ_hat,digits=3)), ρ=$(round(ρ_hat,digits=3)), μ=$(round(μ_hat,digits=4))"

    return (κ=κ_hat, θ=θ_hat, ξ=ξ_hat, ρ=ρ_hat, μ=μ_hat), hist
end

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

"""
    bootstrap_ci(model, returns, n_bootstrap; dt, ci_level, n_epochs, rng)

Compute bootstrap confidence intervals for calibrated model parameters.

Algorithm:
  1. Calibrate model on original data → θ̂
  2. For b=1:B:
     a. Resample returns with replacement (block bootstrap for time series)
     b. Recalibrate model → θ̂_b
  3. Compute percentile CI: [quantile(θ̂_b, α/2), quantile(θ̂_b, 1-α/2)]

Returns DataFrame with parameter estimates and confidence bounds.
"""
function bootstrap_ci(model, returns::AbstractVector, n_bootstrap::Int=100;
                       dt::Float64       = 1.0/252,
                       ci_level::Float64 = 0.95,
                       n_epochs::Int     = 100,
                       block_size::Int   = 21,
                       rng               = Random.GLOBAL_RNG)

    n  = length(returns)
    α  = 1 - ci_level

    # Original calibration
    @info "Calibrating on full dataset..."
    model_orig  = deepcopy(model)
    _, loss_hist = mle_calibrate(model_orig, returns; dt=dt, n_epochs=n_epochs, rng=rng)

    # Extract original parameters
    orig_params  = Float64.(vcat([vec(p) for p in Flux.params(model_orig)]...))
    n_params     = length(orig_params)

    bootstrap_params = zeros(n_bootstrap, n_params)

    for b in 1:n_bootstrap
        b % 10 == 0 && @info "Bootstrap sample $b/$n_bootstrap"

        # Block bootstrap resampling
        n_blocks  = ceil(Int, n / block_size)
        block_starts = rand(rng, 1:max(1, n-block_size+1), n_blocks)
        boot_returns = Float64[]
        for s in block_starts
            append!(boot_returns, returns[s:min(s+block_size-1, n)])
        end
        boot_returns = boot_returns[1:n]

        # Recalibrate
        model_b = deepcopy(model)
        try
            mle_calibrate(model_b, boot_returns; dt=dt, n_epochs=n_epochs, rng=rng)
            bootstrap_params[b, :] = Float64.(vcat([vec(p) for p in Flux.params(model_b)]...))
        catch e
            @warn "Bootstrap $b failed: $e"
            bootstrap_params[b, :] = orig_params  # fallback to original
        end
    end

    # Compute confidence intervals (percentile method)
    lower_α = α / 2
    upper_α = 1 - α / 2
    ci_lower = [quantile(bootstrap_params[:, j], lower_α) for j in 1:n_params]
    ci_upper = [quantile(bootstrap_params[:, j], upper_α) for j in 1:n_params]

    results = DataFrame(
        param_idx  = 1:n_params,
        estimate   = orig_params,
        ci_lower   = ci_lower,
        ci_upper   = ci_upper,
        ci_width   = ci_upper .- ci_lower,
        ci_level   = fill(ci_level, n_params)
    )

    @info "Bootstrap CI ($(round(100*ci_level))%) computed for $n_params parameters"
    return results, bootstrap_params
end

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON METRICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    aic(model, returns, ll; dt)

Akaike Information Criterion: AIC = 2k - 2 log L
where k = number of parameters, L = maximum likelihood.
"""
function aic(model, ll::Float64)
    k = count_params(model)
    return 2k - 2ll
end

"""
    bic(model, returns, ll; dt)

Bayesian Information Criterion: BIC = k log n - 2 log L.
"""
function bic(model, ll::Float64, n_obs::Int)
    k = count_params(model)
    return k * log(n_obs) - 2ll
end

"""
    diebold_mariano_test(errors1, errors2) → (stat, pvalue)

Diebold-Mariano test for equal predictive accuracy.
H₀: E[d_t] = 0 where d_t = e₁_t² - e₂_t².
"""
function diebold_mariano_test(errors1::AbstractVector, errors2::AbstractVector)
    d    = errors1.^2 .- errors2.^2
    n    = length(d)
    d̄    = mean(d)
    # HAC variance with Newey-West bandwidth
    bw   = ceil(Int, 4*(n/100)^(2/9))
    γ0   = var(d)
    γ_ac = sum(abs.(autocov(d, collect(1:bw))))
    σ²_d = (γ0 + 2*γ_ac) / n
    σ²_d = max(σ²_d, 1e-12)
    stat = d̄ / sqrt(σ²_d)
    # Approximate p-value using normal distribution
    pval = 2 * (1 - cdf(Normal(0,1), abs(stat)))
    return (stat=stat, pvalue=pval)
end

"""
    autocov(x, lags) → Vector{Float64}

Autocovariances of x at given lags.
"""
function autocov(x::AbstractVector, lags::AbstractVector{Int})
    n  = length(x)
    x̄  = mean(x)
    xc = x .- x̄
    [dot(xc[1:n-lag], xc[lag+1:n]) / n for lag in lags]
end
