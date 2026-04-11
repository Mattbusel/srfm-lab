"""
market_calibration.jl — End-to-end market calibration pipeline

Calibrates NeuralHeston and classical Heston to synthetic LOB return data,
performs rolling recalibration with regime-aware parameter switching, and
reports in-sample/out-of-sample accuracy.

Data format: CSV with columns [timestamp, open, high, low, close, volume]
Compatible with Chronos LOB simulator output.
"""

using DataFrames
using CSV
using Statistics
using LinearAlgebra
using Random
using Distributions
using Flux

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING AND PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_lob_data(path; symbol, freq)

Load price data from CSV file (Chronos LOB format).
Computes log returns and realised volatility.

Expected columns: timestamp, close (and optionally open, high, low, volume).
Returns a DataFrame with additional columns: log_ret, realized_vol.
"""
function load_lob_data(path::String;
                         symbol::Union{String,Nothing} = nothing,
                         vol_window::Int               = 21)

    df = CSV.read(path, DataFrame)

    # Normalise column names
    rename!(df, lowercase.(names(df)))

    # Filter by symbol if present
    if symbol !== nothing && "symbol" in names(df)
        df = filter(row -> row.symbol == symbol, df)
    end

    # Ensure sorted by time
    if "timestamp" in names(df)
        sort!(df, :timestamp)
    end

    # Compute log returns
    close_col = "close" in names(df) ? :close :
                "price" in names(df) ? :price : names(df)[end]

    prices = Float64.(df[:, close_col])
    n      = length(prices)

    log_rets = [k == 1 ? 0.0 : log(prices[k] / prices[k-1]) for k in 1:n]

    df[!, :log_ret] = log_rets

    # Realised volatility (rolling std)
    rv = zeros(n)
    for k in 1:n
        start_k = max(1, k - vol_window + 1)
        window  = log_rets[start_k:k]
        rv[k]   = length(window) > 1 ? std(window) * sqrt(252) : 0.0
    end
    df[!, :realized_vol] = rv

    @info "Loaded $(n) observations from $path"
    @info "Return stats: mean=$(round(mean(log_rets[2:end])*252,digits=4)), ann_vol=$(round(std(log_rets[2:end])*sqrt(252),digits=4))"

    return df
end

"""
    generate_synthetic_lob_data(; n_obs, dt, heston_params, seed)

Generate synthetic return data from a classical Heston model.
Used for testing when real Chronos LOB data is not available.

Returns DataFrame with columns: timestamp, close, log_ret, realized_vol.
"""
function generate_synthetic_lob_data(;
    n_obs::Int        = 2520,          # 10 years of daily data
    dt::Float64       = 1.0/252,
    κ::Float64        = 2.0,
    θ::Float64        = 0.04,
    ξ::Float64        = 0.3,
    ρ::Float64        = -0.7,
    μ::Float64        = 0.07,
    V0::Float64       = 0.04,
    S0::Float64       = 100.0,
    seed::Int         = 42,
    vol_window::Int   = 21
)
    rng = MersenneTwister(seed)

    # Simulate Heston using Euler-Maruyama
    log_S = log(S0)
    V     = V0
    prices    = zeros(n_obs)
    log_rets  = zeros(n_obs)
    V_path    = zeros(n_obs)

    prices[1]   = S0
    V_path[1]   = V0

    for k in 2:n_obs
        sqdt  = sqrt(dt)
        Z1, Z2 = randn(rng), randn(rng)
        dW1   = sqdt * Z1
        dW2   = sqdt * (ρ * Z1 + sqrt(max(1 - ρ^2, 0)) * Z2)

        sqrtV  = sqrt(max(V, 0))
        d_logS = (μ - 0.5*V)*dt + sqrtV*dW1
        d_V    = κ*(θ - V)*dt + ξ*sqrtV*dW2
        V      = abs(V + d_V)
        log_S += d_logS

        prices[k]   = exp(log_S)
        log_rets[k] = d_logS
        V_path[k]   = V
    end

    # Realised vol
    rv = zeros(n_obs)
    for k in 1:n_obs
        start_k = max(1, k - vol_window + 1)
        rv[k]   = length(log_rets[start_k:k]) > 1 ?
                   std(log_rets[start_k:k]) * sqrt(252) : 0.0
    end

    df = DataFrame(
        timestamp    = 1:n_obs,
        close        = prices,
        log_ret      = log_rets,
        realized_vol = rv,
        true_var     = V_path
    )

    return df, V_path
end

# ─────────────────────────────────────────────────────────────────────────────
# MARKET CALIBRATOR
# ─────────────────────────────────────────────────────────────────────────────

"""
    MarketCalibrator

Encapsulates the full calibration workflow for a single fixed window.

Fields:
  - `neural_heston` : NeuralHeston model
  - `classical_params` : calibrated classical Heston params (NamedTuple)
  - `config`        : TrainingConfig
  - `data`          : calibration DataFrame
  - `dt`            : time step
"""
mutable struct MarketCalibrator
    neural_heston    :: NeuralHeston
    classical_params :: Union{NamedTuple, Nothing}
    config           :: TrainingConfig
    dt               :: Float64
    in_sample_ll     :: Float64
    oos_rmse         :: Float64
end

function MarketCalibrator(; dt::Float64=1.0/252,
                             hidden_dim::Int=32,
                             n_layers::Int=2,
                             n_epochs::Int=200,
                             lr::Float64=1e-3)

    model  = NeuralHeston(use_corrections=true, hidden_dim=hidden_dim, n_layers=n_layers)
    config = TrainingConfig(n_epochs=n_epochs, learning_rate=lr,
                             lr_schedule=:cosine, print_every=50,
                             n_mc_paths=20, dt=dt)

    return MarketCalibrator(model, nothing, config, dt, -Inf, Inf)
end

"""
    calibrate_to_market(mc::MarketCalibrator, df; val_frac, rng)

Calibrate NeuralHeston and classical Heston to data in `df`.

Steps:
  1. Extract return and variance proxy series
  2. Calibrate classical Heston parameters (closed-form MLE)
  3. Initialise NeuralHeston with classical parameters
  4. Fine-tune NeuralHeston via gradient descent
  5. Report in-sample log-likelihoods
"""
function calibrate_to_market(mc::MarketCalibrator, df::DataFrame;
                               val_frac::Float64 = 0.2,
                               rng               = Random.GLOBAL_RNG)

    returns  = Float64.(df.log_ret)
    V_proxy  = (Float64.(df.realized_vol) ./ sqrt(252)).^2  # annualised → daily variance

    n        = length(returns)
    dt       = mc.dt
    n_val    = round(Int, n * val_frac)
    n_train  = n - n_val

    @info "Calibration: n_train=$n_train, n_val=$n_val, dt=$dt"

    # 1. Classical Heston calibration
    @info "Calibrating classical Heston..."
    classical_params, ll_hist = calibrate_heston_params(
        returns[2:n_train], V_proxy[2:n_train];
        dt=dt, n_epochs=300, lr=1e-2, rng=rng
    )
    mc.classical_params = classical_params

    # Initialise NeuralHeston with classical parameters
    mc.neural_heston = NeuralHeston(
        κ = Float32(classical_params.κ),
        θ = Float32(classical_params.θ),
        ξ = Float32(classical_params.ξ),
        ρ = Float32(classical_params.ρ),
        μ = Float32(classical_params.μ),
        use_corrections = true,
        hidden_dim = 32
    )

    # 2. Neural Heston calibration (MLE fine-tuning)
    @info "Fine-tuning NeuralHeston..."

    function neural_heston_loss(model, data, config, rng_local)
        returns_d = Float32.(data.returns)
        V_d       = Float32.(data.V_proxy)
        n_d       = length(returns_d)
        nll       = 0.0f0

        log_S = 0.0f0
        V     = Float32(mean(V_d[1:min(5, n_d)]))

        for k in 1:n_d
            state = Float32[log_S, V]
            t     = Float32(k * config.dt)

            # Drift and diffusion from neural model
            d_vec = heston_drift(model, state, t)
            L     = heston_diffusion(model, state, t)

            μ_r    = d_vec[1] * Float32(config.dt)
            σ²_r   = L[1,1]^2 * Float32(config.dt)
            σ²_r   = max(σ²_r, 1f-8)

            r_k    = returns_d[k]
            nll   += 0.5f0*(r_k - μ_r)^2/σ²_r + 0.5f0*log(2π*σ²_r)

            # Update state with neural dynamics
            dW1 = Float32(sqrt(config.dt)) * Float32(randn(rng_local))
            dW2 = Float32(sqrt(config.dt)) * Float32(randn(rng_local))
            log_S += d_vec[1]*Float32(config.dt) + L[1,1]*dW1
            V     += d_vec[2]*Float32(config.dt) + L[2,1]*dW1 + L[2,2]*dW2
            V      = abs(V)
        end

        return nll / n_d
    end

    data_train = (returns=returns[2:n_train], V_proxy=V_proxy[2:n_train])
    data_val   = (returns=returns[n_train+1:end], V_proxy=V_proxy[n_train+1:end])

    _, tracker = train_model!(mc.neural_heston, data_train, mc.config;
                               val_data=data_val, rng=rng,
                               loss_fn=neural_heston_loss)

    # In-sample log-likelihood comparison
    mc.in_sample_ll = -tracker.train_losses[end] * n_train

    # OOS RMSE
    mc.oos_rmse = oos_prediction_error(mc, data_val.returns)

    @info "Calibration complete: IS_LL=$(round(mc.in_sample_ll,digits=2)), OOS_RMSE=$(round(mc.oos_rmse,digits=6))"

    return mc
end

"""
    oos_prediction_error(mc::MarketCalibrator, oos_returns) → rmse

Compute out-of-sample 1-step ahead return prediction RMSE.
"""
function oos_prediction_error(mc::MarketCalibrator,
                                oos_returns::AbstractVector)
    n       = length(oos_returns)
    n > 0 || return Inf

    # Neural Heston predictions
    errors  = zeros(n)
    log_S   = 0.0f0
    V       = Float32(mc.classical_params === nothing ? 0.04 : mc.classical_params.θ)

    for k in 1:n
        state  = Float32[log_S, V]
        t      = Float32(k * mc.dt)
        d_vec  = heston_drift(mc.neural_heston, state, t)
        μ_pred = Float64(d_vec[1]) * mc.dt
        errors[k] = Float64(oos_returns[k]) - μ_pred

        # Update state
        L      = heston_diffusion(mc.neural_heston, state, t)
        log_S += d_vec[1] * Float32(mc.dt)
        V      = abs(V + d_vec[2] * Float32(mc.dt))
    end

    return sqrt(mean(errors.^2))
end

# ─────────────────────────────────────────────────────────────────────────────
# ROLLING CALIBRATOR
# ─────────────────────────────────────────────────────────────────────────────

"""
    RollingCalibrator

Performs rolling-window recalibration with regime-aware parameter switching.

Each window recalibrates the model on a fresh data window.
Regime detection informs which parameter set to apply.

Fields:
  - `window_size`   : calibration window length (e.g. 504 = 2 years daily)
  - `step_size`     : recalibration frequency (e.g. 63 = quarterly)
  - `dt`            : time step
  - `calibrators`   : Vector of fitted MarketCalibrators, one per window
  - `window_starts` : start indices of each window
"""
mutable struct RollingCalibrator
    window_size   :: Int
    step_size     :: Int
    dt            :: Float64
    calibrators   :: Vector{MarketCalibrator}
    window_starts :: Vector{Int}
    oos_errors    :: Vector{Float64}
end

function RollingCalibrator(; window_size::Int=504, step_size::Int=63,
                              dt::Float64=1.0/252)
    return RollingCalibrator(window_size, step_size, dt,
                              MarketCalibrator[], Int[], Float64[])
end

"""
    rolling_recalibrate(rc::RollingCalibrator, df; rng)

Run rolling recalibration on the full dataset.

For each window [t, t+window_size]:
  1. Calibrate models on window data
  2. Predict on next step_size observations
  3. Record OOS errors
  4. Slide window forward

Returns updated RollingCalibrator with all window results.
"""
function rolling_recalibrate(rc::RollingCalibrator, df::DataFrame;
                               rng = Random.GLOBAL_RNG,
                               verbose::Bool = true)

    n       = nrow(df)
    t_start = 1
    window  = rc.window_size
    step    = rc.step_size

    returns  = Float64.(df.log_ret)
    V_proxy  = (Float64.(df.realized_vol) ./ sqrt(252)).^2

    all_oos_errors  = Float64[]
    all_classical_oos = Float64[]

    window_idx = 1
    while t_start + window + step <= n + 1
        t_end   = t_start + window - 1
        t_pred  = min(t_end + step, n)

        verbose && @info "Window $window_idx: train=[$(t_start):$(t_end)], predict=[$(t_end+1):$(t_pred)]"

        # Build calibration DataFrame for this window
        df_win = df[t_start:t_end, :]

        # Fit calibrators
        mc = MarketCalibrator(; dt=rc.dt, n_epochs=100, lr=1e-3)
        try
            calibrate_to_market(mc, df_win; rng=rng)
        catch e
            @warn "Window $window_idx calibration failed: $e"
            t_start += step
            window_idx += 1
            continue
        end

        push!(rc.calibrators,   mc)
        push!(rc.window_starts, t_start)

        # OOS evaluation
        oos_rets    = returns[t_end+1:t_pred]
        oos_V       = V_proxy[t_end+1:t_pred]

        if !isempty(oos_rets)
            # Neural Heston OOS RMSE
            neural_rmse = oos_prediction_error(mc, oos_rets)
            push!(all_oos_errors, neural_rmse)

            # Classical Heston benchmark
            if mc.classical_params !== nothing
                cp = mc.classical_params
                class_errors = zeros(length(oos_rets))
                V_cl = Float64(cp.θ)
                for k in eachindex(oos_rets)
                    μ_pred = (Float64(cp.μ) - 0.5*V_cl) * rc.dt
                    class_errors[k] = oos_rets[k] - μ_pred
                    V_cl = abs(V_cl + Float64(cp.κ)*(Float64(cp.θ)-V_cl)*rc.dt)
                end
                push!(all_classical_oos, sqrt(mean(class_errors.^2)))
            end
        end

        t_start   += step
        window_idx += 1
    end

    rc.oos_errors = all_oos_errors

    if !isempty(all_oos_errors)
        @info "Rolling calibration complete:"
        @info "  Neural Heston avg OOS RMSE: $(round(mean(all_oos_errors)*1e4,digits=3)) bps"
        if !isempty(all_classical_oos)
            @info "  Classical Heston avg OOS RMSE: $(round(mean(all_classical_oos)*1e4,digits=3)) bps"
            improvement = (mean(all_classical_oos) - mean(all_oos_errors)) / mean(all_classical_oos) * 100
            @info "  Neural improvement: $(round(improvement,digits=2))%"
        end
    end

    return rc, all_oos_errors, all_classical_oos
end

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON: NEURAL SDE vs CLASSICAL HESTON
# ─────────────────────────────────────────────────────────────────────────────

"""
    compare_models(mc::MarketCalibrator, df; rng)

Comprehensive in-sample and out-of-sample comparison of:
  - Classical Heston (calibrated via MLE)
  - Neural Heston (calibrated via gradient descent)

Metrics:
  - In-sample log-likelihood (per observation)
  - Out-of-sample 1-step RMSE
  - Out-of-sample directional accuracy
  - Volatility forecasting RMSE (vs realised vol)
  - AIC and BIC

Returns DataFrame with comparison results.
"""
function compare_models(mc::MarketCalibrator, df::DataFrame;
                          test_frac::Float64 = 0.2,
                          rng               = Random.GLOBAL_RNG)

    n      = nrow(df)
    n_test = round(Int, n * test_frac)
    n_train = n - n_test

    returns  = Float64.(df.log_ret)
    V_proxy  = (Float64.(df.realized_vol) ./ sqrt(252)).^2

    train_ret = returns[2:n_train]
    test_ret  = returns[n_train+1:end]
    test_V    = V_proxy[n_train+1:end]

    # ── Classical Heston ──────────────────────────────────────────────────────
    cp = mc.classical_params
    if cp !== nothing
        # In-sample LL
        ll_class_is = sum(begin
            μ_r   = (cp.μ - 0.5*V_proxy[k])*mc.dt
            σ²_r  = max(V_proxy[k]*mc.dt, 1e-10)
            -0.5*(train_ret[k-1] - μ_r)^2/σ²_r - 0.5*log(2π*σ²_r)
        end for k in 2:n_train) / n_train

        # OOS predictions (classical)
        class_oos_ret_errs = Float64[]
        class_oos_vol_errs = Float64[]
        V_cl = Float64(cp.θ)
        for k in eachindex(test_ret)
            μ_pred     = (cp.μ - 0.5*V_cl)*mc.dt
            vol_pred   = sqrt(max(V_cl, 1e-8))
            push!(class_oos_ret_errs,  test_ret[k] - μ_pred)
            push!(class_oos_vol_errs,  vol_pred - sqrt(max(test_V[k], 1e-8)))
            V_cl = abs(V_cl + cp.κ*(cp.θ - V_cl)*mc.dt)
        end
    else
        ll_class_is = -Inf
        class_oos_ret_errs = zeros(n_test)
        class_oos_vol_errs = zeros(n_test)
    end

    # ── Neural Heston ─────────────────────────────────────────────────────────
    log_S  = 0.0f0
    V_nh   = Float32(mc.classical_params !== nothing ? mc.classical_params.θ : 0.04)
    ll_neural_is = 0.0f0

    for k in 1:n_train-1
        state = Float32[log_S, V_nh]
        t     = Float32(k * mc.dt)
        d_vec = heston_drift(mc.neural_heston, state, t)
        L     = heston_diffusion(mc.neural_heston, state, t)

        μ_r   = d_vec[1] * Float32(mc.dt)
        σ²_r  = max(L[1,1]^2 * Float32(mc.dt), 1f-8)
        r_k   = Float32(train_ret[k])
        ll_neural_is += -0.5f0*(r_k - μ_r)^2/σ²_r - 0.5f0*log(2π*σ²_r)

        dW1 = Float32(sqrt(mc.dt)) * Float32(randn(rng))
        dW2 = Float32(sqrt(mc.dt)) * Float32(randn(rng))
        log_S += d_vec[1]*Float32(mc.dt) + L[1,1]*dW1
        V_nh  += d_vec[2]*Float32(mc.dt) + L[2,1]*dW1 + L[2,2]*dW2
        V_nh   = abs(V_nh)
    end
    ll_neural_is = Float64(ll_neural_is) / (n_train-1)

    # OOS neural predictions
    neural_oos_ret_errs = Float64[]
    neural_oos_vol_errs = Float64[]
    for k in eachindex(test_ret)
        state     = Float32[log_S, V_nh]
        t         = Float32((n_train + k) * mc.dt)
        d_vec     = heston_drift(mc.neural_heston, state, t)
        L         = heston_diffusion(mc.neural_heston, state, t)

        μ_pred    = Float64(d_vec[1]) * mc.dt
        vol_pred  = Float64(sqrt(max(L[1,1]^2 * Float32(mc.dt), 1f-8))) / sqrt(mc.dt)

        push!(neural_oos_ret_errs,  test_ret[k] - μ_pred)
        push!(neural_oos_vol_errs,  vol_pred - sqrt(max(test_V[k], 1e-8)))

        dW1 = Float32(sqrt(mc.dt)) * Float32(randn(rng))
        dW2 = Float32(sqrt(mc.dt)) * Float32(randn(rng))
        log_S += d_vec[1]*Float32(mc.dt) + L[1,1]*dW1
        V_nh  += d_vec[2]*Float32(mc.dt) + L[2,1]*dW1 + L[2,2]*dW2
        V_nh   = abs(V_nh)
    end

    # Directional accuracy
    dir_acc_class  = mean(sign.(class_oos_ret_errs  .+ mean(test_ret)) .== sign.(test_ret))
    dir_acc_neural = mean(sign.(neural_oos_ret_errs .+ mean(test_ret)) .== sign.(test_ret))

    # AIC / BIC (approximate: classical Heston has 5 params)
    aic_class  = 2*5  - 2*ll_class_is*n_train
    bic_class  = 5*log(n_train) - 2*ll_class_is*n_train
    aic_neural = 2*count_params(mc.neural_heston) - 2*ll_neural_is*n_train
    bic_neural = count_params(mc.neural_heston)*log(n_train) - 2*ll_neural_is*n_train

    results = DataFrame(
        metric = [
            "IS_log_likelihood_per_obs",
            "OOS_return_RMSE",
            "OOS_vol_RMSE",
            "OOS_directional_accuracy",
            "AIC",
            "BIC",
            "n_params"
        ],
        classical_heston = [
            ll_class_is,
            sqrt(mean(class_oos_ret_errs.^2)),
            sqrt(mean(class_oos_vol_errs.^2)),
            dir_acc_class,
            aic_class,
            bic_class,
            5.0
        ],
        neural_heston = [
            ll_neural_is,
            sqrt(mean(neural_oos_ret_errs.^2)),
            sqrt(mean(neural_oos_vol_errs.^2)),
            dir_acc_neural,
            Float64(aic_neural),
            Float64(bic_neural),
            Float64(count_params(mc.neural_heston))
        ]
    )

    @info "\n" * "─"^60
    @info "Model Comparison: NeuralHeston vs Classical Heston"
    @info "─"^60
    for row in eachrow(results)
        @info "  $(row.metric):"
        @info "    Classical = $(round(row.classical_heston, sigdigits=4))"
        @info "    Neural    = $(round(row.neural_heston,    sigdigits=4))"
    end

    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# REGIME-AWARE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    RegimeAwareCalibrator

Maintains separate NeuralHeston models for each regime, switching between
them based on the particle filter regime probabilities.

Fields:
  - `regime_models` : Vector{NeuralHeston}, one per regime
  - `detector`      : RegimeDetector
  - `n_regimes`     : number of regimes
"""
mutable struct RegimeAwareCalibrator
    regime_models  :: Vector{NeuralHeston}
    detector       :: RegimeDetector
    n_regimes      :: Int
    dt             :: Float64
end

function RegimeAwareCalibrator(ldm, n_regimes::Int=2;
                                 dt::Float64=1.0/252,
                                 hidden_dim::Int=32)

    models = [NeuralHeston(use_corrections=true, hidden_dim=hidden_dim)
              for _ in 1:n_regimes]
    detector = RegimeDetector(ldm, n_regimes)

    return RegimeAwareCalibrator(models, detector, n_regimes, dt)
end

"""
    calibrate_regime_aware(rac, df; rng)

Calibrate separate NeuralHeston models for each detected regime.

Steps:
  1. Run regime detection on full dataset
  2. Split observations by dominant regime
  3. Calibrate one NeuralHeston per regime
  4. Combine predictions using regime probabilities
"""
function calibrate_regime_aware(rac::RegimeAwareCalibrator, df::DataFrame;
                                  rng = Random.GLOBAL_RNG)

    returns  = Float64.(df.log_ret)
    n        = length(returns)

    # Detect regimes
    @info "Running regime detection..."
    regime_probs, map_regimes, _ = detect_regimes(rac.detector, returns;
                                                    dt=rac.dt, rng=rng)

    # Split data by regime
    for k in 1:rac.n_regimes
        regime_idx = findall(==(k), map_regimes[2:end]) .+ 1
        if length(regime_idx) < 50
            @warn "Regime $k has only $(length(regime_idx)) observations, skipping"
            continue
        end

        @info "Calibrating regime $k (n=$(length(regime_idx)) obs)..."
        regime_returns = returns[regime_idx]

        # Build a mini DataFrame for this regime
        rv_k = [std(regime_returns[max(1,i-20):i]) * sqrt(252)
                for i in eachindex(regime_returns)]
        df_k = DataFrame(log_ret=regime_returns, realized_vol=rv_k)

        mc_k = MarketCalibrator(; dt=rac.dt, n_epochs=100)
        try
            calibrate_to_market(mc_k, df_k; rng=rng)
            rac.regime_models[k] = mc_k.neural_heston
        catch e
            @warn "Regime $k calibration failed: $e"
        end
    end

    return rac, regime_probs, map_regimes
end

"""
    predict_regime_aware(rac, state, t, regime_probs)

Generate drift/vol prediction weighted by regime probabilities.
"""
function predict_regime_aware(rac::RegimeAwareCalibrator,
                                state::AbstractVector,
                                t::Real,
                                regime_probs::AbstractVector)

    d_vec_total = zeros(Float32, 2)
    L_total     = zeros(Float32, 2, 2)

    for k in 1:rac.n_regimes
        w     = Float32(regime_probs[k])
        d_k   = heston_drift(rac.regime_models[k],     state, t)
        L_k   = heston_diffusion(rac.regime_models[k], state, t)
        d_vec_total .+= w .* d_k
        L_total     .+= w .* L_k
    end

    return d_vec_total, L_total
end
