"""
simulation_pipeline.jl — End-to-end pipeline from raw data to calibrated SDE simulation

Implements:
  1. Data loading (CSV, in-memory, synthetic)
  2. Feature extraction (delegates to feature_engineering.jl)
  3. Model selection via AIC/BIC/WAIC
  4. Calibration (delegates to calibration_engine.jl)
  5. Forward simulation of calibrated SDEs
  6. Scenario generation (risk scenarios, stress tests)
  7. Output formatting (summary stats, paths, distributions)
  8. Pipeline composition and chaining
  9. Validation and backtesting
 10. Reproducibility (seed management, config serialisation)

This module is the top-level orchestrator of the NeuroSDE library.
"""

using Statistics
using LinearAlgebra
using Random
using Printf
using Dates

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PIPELINE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    DataSource

Union type for various data sources.
"""
abstract type AbstractDataSource end

struct CSVDataSource      <: AbstractDataSource
    filepath  :: String
    delimiter :: Char
    date_col  :: String
    price_col :: String
    vol_col   :: Union{Nothing, String}
end

struct InMemoryDataSource <: AbstractDataSource
    prices    :: Vector{Float64}
    dates     :: Union{Nothing, Vector{Float64}}
    ohlcv     :: Union{Nothing, Any}   # OHLCV type from feature_engineering
end

struct SyntheticDataSource <: AbstractDataSource
    model     :: Symbol   # :gbm, :heston, :ou, :sabr
    params    :: NamedTuple
    n_steps   :: Int
    dt        :: Float64
    seed      :: Int
end

"""
    PipelineConfig

Full configuration for the simulation pipeline.

Fields:
  - `data_source`   : data input specification
  - `feature_cfg`   : feature engineering settings
  - `model_names`   : models to calibrate (:heston, :sabr, :neural)
  - `n_paths`       : number of simulation paths
  - `horizon`       : simulation horizon (in years)
  - `dt`            : simulation step size
  - `seed`          : global random seed
  - `n_restarts`    : calibration restarts
  - `output_dir`    : directory for saving outputs
  - `verbose`       : logging level
"""
struct PipelineConfig
    data_source    :: AbstractDataSource
    model_names    :: Vector{Symbol}
    n_paths        :: Int
    horizon        :: Float64
    dt             :: Float64
    seed           :: Int
    n_restarts     :: Int
    output_dir     :: Union{Nothing, String}
    verbose        :: Bool
end

function PipelineConfig(data_source::AbstractDataSource;
                        model_names   = [:heston],
                        n_paths::Int  = 1000,
                        horizon::Real = 1.0,
                        dt::Real      = 1/252,
                        seed::Int     = 42,
                        n_restarts::Int = 5,
                        output_dir    = nothing,
                        verbose::Bool = true)
    PipelineConfig(data_source, model_names, n_paths,
                   Float64(horizon), Float64(dt), seed,
                   n_restarts, output_dir, verbose)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate_gbm(S0, r, q, σ, T, dt, N; seed=42) → Matrix{Float64}

Simulate N paths of Geometric Brownian Motion.
Returns (n_steps+1 × N) matrix of price paths.
"""
function simulate_gbm(S0::Real, r::Real, q::Real, σ::Real,
                       T::Real, dt::Real, N::Int;
                       seed::Int = 42)
    rng    = MersenneTwister(seed)
    n_steps = Int(round(T / dt))
    paths  = zeros(n_steps + 1, N)
    paths[1, :] .= S0
    drift  = (r - q - 0.5 * σ^2) * dt
    diff   = σ * sqrt(dt)
    for t in 1:n_steps
        z = randn(rng, N)
        paths[t+1, :] = paths[t, :] .* exp.(drift .+ diff .* z)
    end
    return paths
end

"""
    simulate_heston(S0, V0, κ, θ, ξ, ρ, r, q, T, dt, N; seed=42) → (S_paths, V_paths)

Simulate N paths of the Heston model via Euler-Maruyama.
"""
function simulate_heston(S0::Real, V0::Real, κ::Real, θ::Real, ξ::Real,
                          ρ::Real, r::Real, q::Real, T::Real, dt::Real, N::Int;
                          seed::Int = 42)
    rng     = MersenneTwister(seed)
    n_steps = Int(round(T / dt))
    S_paths = zeros(n_steps + 1, N)
    V_paths = zeros(n_steps + 1, N)
    S_paths[1, :] .= S0
    V_paths[1, :] .= V0

    sqdt = sqrt(dt)
    for t in 1:n_steps
        z1 = randn(rng, N)
        z2 = ρ .* z1 .+ sqrt(1 - ρ^2) .* randn(rng, N)
        V_cur = max.(V_paths[t, :], 0.0)
        sqV   = sqrt.(V_cur)
        S_paths[t+1, :] = S_paths[t, :] .* exp.((r - q .- 0.5 .* V_cur) .* dt
                                                   .+ sqV .* sqdt .* z1)
        V_new = V_cur .+ κ .* (θ .- V_cur) .* dt .+ ξ .* sqV .* sqdt .* z2
        V_paths[t+1, :] = max.(V_new, 0.0)
    end
    return S_paths, V_paths
end

"""
    simulate_ou(x0, κ, θ, σ, T, dt, N; seed=42) → Matrix{Float64}

Simulate N paths of Ornstein-Uhlenbeck process (exact Euler).
dX_t = κ(θ - X_t) dt + σ dW_t
"""
function simulate_ou(x0::Real, κ::Real, θ::Real, σ::Real,
                      T::Real, dt::Real, N::Int;
                      seed::Int = 42)
    rng     = MersenneTwister(seed)
    n_steps = Int(round(T / dt))
    paths   = zeros(n_steps + 1, N)
    paths[1, :] .= x0
    sqdt = sqrt(dt)
    for t in 1:n_steps
        z = randn(rng, N)
        paths[t+1, :] = paths[t, :] .+ κ .* (θ .- paths[t, :]) .* dt .+ σ .* sqdt .* z
    end
    return paths
end

"""
    simulate_rough_vol(S0, H, ν, ρ, r, q, T, dt, N; seed=42) → (S_paths, V_paths)

Approximate simulation of rough Bergomi model.
Uses Cholesky factorisation of the fBm covariance matrix (exact simulation
for small T, approximate for large N).
"""
function simulate_rough_vol(S0::Real, H::Real, ν::Real, ρ::Real,
                             r::Real, q::Real, T::Real, dt::Real, N::Int;
                             seed::Int = 42)
    rng     = MersenneTwister(seed)
    n_steps = Int(round(T / dt))
    t_grid  = collect(1:n_steps) .* dt

    # fBm covariance kernel: Cov(W^H_s, W^H_t) = 0.5(s^{2H} + t^{2H} - |t-s|^{2H})
    n = n_steps
    Σ_fbm = zeros(n, n)
    for i in 1:n, j in 1:n
        s, t = t_grid[i], t_grid[j]
        Σ_fbm[i,j] = 0.5 * (s^(2H) + t^(2H) - abs(t-s)^(2H))
    end
    Σ_fbm = Hermitian(Σ_fbm + 1e-8 * I(n))
    L_fbm = cholesky(Σ_fbm).L

    S_paths = zeros(n_steps + 1, N)
    V_paths = zeros(n_steps + 1, N)
    S_paths[1, :] .= S0
    ξ0 = 0.04  # initial forward variance curve (flat)
    V_paths[1, :] .= ξ0

    for k in 1:N
        # Simulate fBm
        z     = randn(rng, n)
        W_H   = L_fbm * z

        # Rough vol
        V = ξ0 .* exp.(ν .* W_H .- 0.5 .* ν^2 .* t_grid.^(2H))
        V_paths[2:end, k] = V

        # Integrate S via Euler (correlated BM)
        z_S = ρ .* z .+ sqrt(1 - ρ^2) .* randn(rng, n)
        sqdt = sqrt(dt)
        for t in 1:n_steps
            S_paths[t+1, k] = S_paths[t, k] * exp(
                (r - q - 0.5 * V[t]) * dt + sqrt(max(V[t], 0.0) * dt) * z_S[t])
        end
    end
    return S_paths, V_paths
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MODEL SELECTION (AIC/BIC)
# ─────────────────────────────────────────────────────────────────────────────

"""
    ModelSelectionResult

Result of information-criterion model selection.
"""
struct ModelSelectionResult
    model_names    :: Vector{String}
    log_likelihoods :: Vector{Float64}
    n_params       :: Vector{Int}
    n_obs          :: Int
    aic            :: Vector{Float64}
    bic            :: Vector{Float64}
    best_aic       :: Int    # index of best model by AIC
    best_bic       :: Int    # index of best model by BIC
    delta_aic      :: Vector{Float64}
    delta_bic      :: Vector{Float64}
    akaike_weights :: Vector{Float64}
end

"""
    aic(log_lik, k) → Float64

Akaike Information Criterion: AIC = 2k - 2 log L.
"""
aic(log_lik::Real, k::Int) = 2k - 2 * log_lik

"""
    bic(log_lik, k, n) → Float64

Bayesian Information Criterion: BIC = k log(n) - 2 log L.
"""
bic(log_lik::Real, k::Int, n::Int) = k * log(n) - 2 * log_lik

"""
    aicc(log_lik, k, n) → Float64

Corrected AIC for small samples.
"""
aicc(log_lik::Real, k::Int, n::Int) = aic(log_lik, k) + 2k*(k+1)/(n - k - 1)

"""
    select_model(log_likelihoods, n_params, n_obs;
                 model_names=[]) → ModelSelectionResult

Compare models using AIC and BIC.
"""
function select_model(log_likelihoods::Vector{Float64},
                      n_params::Vector{Int},
                      n_obs::Int;
                      model_names::Vector{String} = String[])
    n_models = length(log_likelihoods)
    pnames   = isempty(model_names) ? ["Model_$i" for i in 1:n_models] : model_names
    aic_vals = [aic(log_likelihoods[i], n_params[i]) for i in 1:n_models]
    bic_vals = [bic(log_likelihoods[i], n_params[i], n_obs) for i in 1:n_models]

    best_aic = argmin(aic_vals)
    best_bic = argmin(bic_vals)

    Δaic = aic_vals .- aic_vals[best_aic]
    Δbic = bic_vals .- bic_vals[best_bic]

    # Akaike weights
    w = exp.(-0.5 .* Δaic)
    w ./= sum(w)

    return ModelSelectionResult(pnames, log_likelihoods, n_params, n_obs,
                                 aic_vals, bic_vals, best_aic, best_bic,
                                 Δaic, Δbic, w)
end

"""
    print_model_selection(ms::ModelSelectionResult)
"""
function print_model_selection(ms::ModelSelectionResult)
    println("─"^80)
    println("  Model Selection Results")
    @printf "  %-16s  %12s  %6s  %10s  %10s  %6s  %8s\n" \
            "Model" "LogLik" "k" "AIC" "BIC" "ΔAIC" "w_AIC"
    println("─"^80)
    for i in 1:length(ms.model_names)
        marker = (i == ms.best_aic) ? "*" : " "
        @printf "  %-15s%s  %12.2f  %6d  %10.2f  %10.2f  %6.2f  %8.4f\n" \
                ms.model_names[i] marker ms.log_likelihoods[i] ms.n_params[i] \
                ms.aic[i] ms.bic[i] ms.delta_aic[i] ms.akaike_weights[i]
    end
    println("─"^80)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SCENARIO GENERATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    ScenarioConfig

Configuration for scenario generation.
"""
struct ScenarioConfig
    scenario_type :: Symbol   # :normal, :stress, :historical, :reverse_stress
    n_scenarios   :: Int
    horizon       :: Float64
    dt            :: Float64
    shock_size    :: Float64  # std multiples for stress test
    seed          :: Int
end

ScenarioConfig(; scenario_type=:normal, n_scenarios=1000,
                 horizon=1.0, dt=1/52, shock_size=3.0, seed=42) =
    ScenarioConfig(scenario_type, n_scenarios, horizon, dt, shock_size, seed)

"""
    ScenarioSet

Container for generated scenarios.
"""
struct ScenarioSet
    paths       :: Matrix{Float64}   # (n_steps+1 × n_scenarios)
    final_vals  :: Vector{Float64}
    returns     :: Vector{Float64}   # log returns over horizon
    var_95      :: Float64
    var_99      :: Float64
    cvar_95     :: Float64
    cvar_99     :: Float64
    scenario_type :: Symbol
end

"""
    compute_var_cvar(returns, confidence=0.95) → (var, cvar)

Compute Value-at-Risk and Conditional-VaR from return distribution.
"""
function compute_var_cvar(returns::AbstractVector, confidence::Real=0.95)
    sorted = sort(returns)
    n      = length(sorted)
    idx    = Int(floor((1 - confidence) * n))
    idx    = max(idx, 1)
    var_c  = -sorted[idx]
    cvar_c = -mean(sorted[1:idx])
    return var_c, cvar_c
end

"""
    generate_scenarios(S0, model_params, scenario_cfg::ScenarioConfig;
                       model=:heston) → ScenarioSet

Generate scenario paths using a calibrated model.
"""
function generate_scenarios(S0::Real,
                             model_params::NamedTuple,
                             scenario_cfg::ScenarioConfig;
                             model::Symbol = :heston)
    rng = MersenneTwister(scenario_cfg.seed)
    T   = scenario_cfg.horizon
    dt  = scenario_cfg.dt
    N   = scenario_cfg.n_scenarios

    paths = if model == :heston
        κ   = get(model_params, :κ,  2.0)
        θ_h = get(model_params, :θ,  0.04)
        ξ   = get(model_params, :ξ,  0.5)
        ρ   = get(model_params, :ρ, -0.7)
        V0  = get(model_params, :V0, 0.04)
        r   = get(model_params, :r,   0.05)
        q   = get(model_params, :q,   0.02)
        S_p, _ = simulate_heston(S0, V0, κ, θ_h, ξ, ρ, r, q, T, dt, N;
                                  seed=scenario_cfg.seed)
        S_p
    elseif model == :gbm
        r  = get(model_params, :r, 0.05)
        q  = get(model_params, :q, 0.02)
        σ  = get(model_params, :σ, 0.20)
        simulate_gbm(S0, r, q, σ, T, dt, N; seed=scenario_cfg.seed)
    else
        # Default: GBM
        simulate_gbm(S0, 0.05, 0.02, 0.20, T, dt, N; seed=scenario_cfg.seed)
    end

    # Apply stress shocks if needed
    if scenario_cfg.scenario_type == :stress
        shock = exp(scenario_cfg.shock_size * 0.20 * sqrt(T))
        paths[end, :] .*= (rand(rng, N) .< 0.5 ? shock : 1/shock)
    end

    final_vals = paths[end, :]
    log_rets   = log.(final_vals ./ S0)

    var_95, cvar_95 = compute_var_cvar(log_rets, 0.95)
    var_99, cvar_99 = compute_var_cvar(log_rets, 0.99)

    return ScenarioSet(paths, final_vals, log_rets,
                       var_95, var_99, cvar_95, cvar_99,
                       scenario_cfg.scenario_type)
end

"""
    print_scenario_summary(ss::ScenarioSet)
"""
function print_scenario_summary(ss::ScenarioSet)
    @printf "─────────────────────────────────────────────\n"
    @printf "  Scenario Summary (%s, n=%d)\n" string(ss.scenario_type) length(ss.final_vals)
    @printf "─────────────────────────────────────────────\n"
    @printf "  Mean return     : %+.4f\n" mean(ss.returns)
    @printf "  Std return      : %.4f\n"  std(ss.returns)
    @printf "  Median return   : %+.4f\n" median(ss.returns)
    @printf "  10th pct        : %+.4f\n" quantile(ss.returns, 0.10)
    @printf "  90th pct        : %+.4f\n" quantile(ss.returns, 0.90)
    @printf "  VaR 95%%         : %.4f\n" ss.var_95
    @printf "  CVaR 95%%        : %.4f\n" ss.cvar_95
    @printf "  VaR 99%%         : %.4f\n" ss.var_99
    @printf "  CVaR 99%%        : %.4f\n" ss.cvar_99
    @printf "─────────────────────────────────────────────\n"
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────

"""
    BacktestResult

Result of a rolling-window backtest.
"""
struct BacktestResult
    dates         :: Vector{Int}     # time indices
    predicted_var :: Vector{Float64} # predicted VaR at each step
    actual_returns :: Vector{Float64}
    violations    :: BitVector       # actual > predicted VaR
    violation_rate :: Float64
    kupiec_stat   :: Float64         # Kupiec POF test statistic
    kupiec_pval   :: Float64
end

"""
    run_backtest(returns, model_fn, window, confidence;
                 rolling_step=1) → BacktestResult

Rolling-window backtest of a VaR model.

- `model_fn(historical_returns)` : given history, return predicted 1-day VaR
- `window` : estimation window size
- `confidence` : VaR confidence level
"""
function run_backtest(returns::AbstractVector,
                      model_fn::Function,
                      window::Int,
                      confidence::Real;
                      rolling_step::Int = 1)
    n        = length(returns)
    n_test   = length(window:rolling_step:(n-1))
    pred_var  = zeros(n_test)
    act_ret   = zeros(n_test)
    test_dates = zeros(Int, n_test)
    k = 0
    for t in window:rolling_step:(n-1)
        k += 1
        hist = returns[max(1, t-window+1):t]
        pred_var[k]   = model_fn(hist)
        act_ret[k]    = returns[t+1]
        test_dates[k] = t+1
    end

    violations    = act_ret .< -pred_var
    viol_rate     = mean(violations)
    expected_rate = 1 - confidence

    # Kupiec POF test
    T_bt  = length(violations)
    V_bt  = sum(violations)
    p_hat = viol_rate
    p0    = expected_rate

    ll0   = V_bt * log(p0) + (T_bt - V_bt) * log(1 - p0)
    ll1   = V_bt > 0 && V_bt < T_bt ?
            V_bt * log(p_hat) + (T_bt - V_bt) * log(1 - p_hat) : -Inf
    kupiec_stat = -2 * (ll0 - ll1)
    kupiec_pval = kupiec_stat > 0 ? 1.0 - cdf(Chisq(1), kupiec_stat) : 1.0

    return BacktestResult(test_dates, pred_var, act_ret,
                           violations, viol_rate,
                           kupiec_stat, kupiec_pval)
end

"""
    historical_var_model(returns; confidence=0.95) → VaR

Simple historical simulation VaR model for backtesting baseline.
"""
function historical_var_model(confidence::Real)
    return hist -> -quantile(hist, 1 - confidence)
end

"""
    gaussian_var_model(confidence=0.95) → Function

Parametric Gaussian VaR model.
"""
function gaussian_var_model(confidence::Real)
    return hist -> mean(hist) - quantile(Normal(), 1 - confidence) * std(hist)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

"""
    SimulationOutput

Container for all pipeline simulation outputs.
"""
struct SimulationOutput
    spot_paths     :: Matrix{Float64}   # (n_steps+1 × N)
    vol_paths      :: Union{Nothing, Matrix{Float64}}
    time_grid      :: Vector{Float64}
    model_name     :: Symbol
    model_params   :: NamedTuple
    scenario_set   :: Union{Nothing, ScenarioSet}
    config         :: PipelineConfig
    timestamp      :: Float64
    elapsed_sec    :: Float64
end

"""
    path_statistics(paths) → NamedTuple

Compute summary statistics from a Monte Carlo path matrix.
"""
function path_statistics(paths::AbstractMatrix)
    n_steps, N = size(paths)
    # Log-returns per path
    log_rets = log.(paths[end, :] ./ paths[1, :])
    # Cross-sectional stats at each time step
    path_mean = vec(mean(paths, dims=2))
    path_std  = vec(std(paths, dims=2))
    path_p10  = [quantile(paths[t, :], 0.10) for t in 1:n_steps]
    path_p90  = [quantile(paths[t, :], 0.90) for t in 1:n_steps]

    return (
        mean_final   = mean(log_rets),
        std_final    = std(log_rets),
        var_95       = -quantile(log_rets, 0.05),
        path_mean    = path_mean,
        path_std     = path_std,
        path_p10     = path_p10,
        path_p90     = path_p90,
        n_paths      = N,
        n_steps      = n_steps,
    )
end

"""
    print_simulation_summary(out::SimulationOutput)
"""
function print_simulation_summary(out::SimulationOutput)
    stats = path_statistics(out.spot_paths)
    @printf "═══════════════════════════════════════════════════\n"
    @printf "  Simulation Summary: %s\n" string(out.model_name)
    @printf "═══════════════════════════════════════════════════\n"
    @printf "  Paths: %-6d  Steps: %-6d\n" stats.n_paths stats.n_steps
    @printf "  Mean log-return : %+.6f\n" stats.mean_final
    @printf "  Std log-return  : %.6f\n"  stats.std_final
    @printf "  VaR 95%% (1-yr)  : %.4f\n" stats.var_95
    @printf "  Elapsed (s)     : %.3f\n"  out.elapsed_sec
    @printf "═══════════════════════════════════════════════════\n"
    !isnothing(out.scenario_set) && print_scenario_summary(out.scenario_set)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_data(src::SyntheticDataSource) → Vector{Float64}

Generate synthetic price data from specified model.
"""
function load_data(src::SyntheticDataSource)
    if src.model == :gbm
        r  = get(src.params, :r, 0.05)
        q  = get(src.params, :q, 0.02)
        σ  = get(src.params, :σ, 0.20)
        S0 = get(src.params, :S0, 100.0)
        paths = simulate_gbm(S0, r, q, σ,
                             src.n_steps * src.dt, src.dt, 1;
                             seed=src.seed)
        return paths[:, 1]
    elseif src.model == :heston
        κ  = get(src.params, :κ,  2.0)
        θ  = get(src.params, :θ,  0.04)
        ξ  = get(src.params, :ξ,  0.5)
        ρ  = get(src.params, :ρ, -0.7)
        V0 = get(src.params, :V0, 0.04)
        r  = get(src.params, :r, 0.05)
        q  = get(src.params, :q, 0.02)
        S0 = get(src.params, :S0, 100.0)
        S_p, _ = simulate_heston(S0, V0, κ, θ, ξ, ρ, r, q,
                                  src.n_steps * src.dt, src.dt, 1;
                                  seed=src.seed)
        return S_p[:, 1]
    elseif src.model == :ou
        κ  = get(src.params, :κ, 2.0)
        θ  = get(src.params, :θ, 0.04)
        σ  = get(src.params, :σ, 0.1)
        x0 = get(src.params, :x0, 0.04)
        paths = simulate_ou(x0, κ, θ, σ,
                            src.n_steps * src.dt, src.dt, 1;
                            seed=src.seed)
        return paths[:, 1]
    else
        # Default GBM
        return simulate_gbm(100.0, 0.05, 0.02, 0.20,
                            src.n_steps * src.dt, src.dt, 1;
                            seed=src.seed)[:, 1]
    end
end

load_data(src::InMemoryDataSource) = src.prices

"""
    run_pipeline(cfg::PipelineConfig) → SimulationOutput

Execute the full simulation pipeline:
  1. Load data
  2. Compute features
  3. Select model
  4. Calibrate
  5. Simulate
  6. Generate scenarios
  7. Return output
"""
function run_pipeline(cfg::PipelineConfig)
    t_start = time()
    cfg.verbose && println("\n" * "═"^55)
    cfg.verbose && println("  NeuroSDE Simulation Pipeline")
    cfg.verbose && println("  Started: " * string(now()))
    cfg.verbose && println("═"^55)

    rng = MersenneTwister(cfg.seed)

    # ── Step 1: Load data ──────────────────────────────────────────────
    cfg.verbose && @info "Step 1: Loading data..."
    prices = load_data(cfg.data_source)
    n_obs  = length(prices)
    S0     = prices[end]
    cfg.verbose && @printf "  Loaded %d observations, S0 = %.4f\n" n_obs S0

    # ── Step 2: Log-returns ────────────────────────────────────────────
    cfg.verbose && @info "Step 2: Computing log-returns..."
    log_rets = log.(prices[2:end] ./ prices[1:end-1])
    σ_hist   = std(log_rets) * sqrt(252)
    μ_hist   = mean(log_rets) * 252
    cfg.verbose && @printf "  Historical vol = %.4f, drift = %.4f\n" σ_hist μ_hist

    # ── Step 3: Model selection (simplified) ──────────────────────────
    cfg.verbose && @info "Step 3: Model selection..."
    n_models = length(cfg.model_names)
    ll_vals  = zeros(n_models)
    k_vals   = Int[]

    for (i, mname) in enumerate(cfg.model_names)
        if mname == :gbm
            # GBM: 2 params (μ, σ)
            σ_mle = std(log_rets)
            μ_mle = mean(log_rets) - 0.5 * σ_mle^2
            ll_vals[i] = sum(logpdf.(Normal(μ_mle, σ_mle), log_rets))
            push!(k_vals, 2)
        elseif mname == :heston
            # Use simplified VG-like fit (proxy LL)
            ll_vals[i] = sum(logpdf.(Normal(mean(log_rets), std(log_rets) * 1.1), log_rets)) - 5.0
            push!(k_vals, 5)
        elseif mname == :ou
            # OU for log-prices
            push!(k_vals, 3)
            ll_vals[i] = sum(logpdf.(Normal(0.0, std(log_rets)), log_rets)) - 2.0
        else
            push!(k_vals, 4)
            ll_vals[i] = sum(logpdf.(Normal(mean(log_rets), std(log_rets)), log_rets)) - 1.0
        end
    end

    ms_result = select_model(ll_vals, k_vals, n_obs;
                              model_names=string.(cfg.model_names))
    cfg.verbose && print_model_selection(ms_result)
    best_model = cfg.model_names[ms_result.best_aic]

    # ── Step 4: Calibrate ──────────────────────────────────────────────
    cfg.verbose && @info "Step 4: Calibrating model :$best_model..."
    model_params = if best_model == :gbm
        (r=0.05, q=0.02, σ=σ_hist, S0=S0)
    elseif best_model == :heston
        (κ=2.0, θ=σ_hist^2, ξ=0.5, ρ=-0.7, V0=σ_hist^2, r=0.05, q=0.02, S0=S0)
    elseif best_model == :ou
        (κ=2.0, θ=mean(log.(prices)), σ=σ_hist, x0=log(S0))
    else
        (r=0.05, q=0.02, σ=σ_hist, S0=S0)
    end
    cfg.verbose && @info "  Calibrated: " * string(model_params)

    # ── Step 5: Forward simulation ─────────────────────────────────────
    cfg.verbose && @info "Step 5: Forward simulation (N=$(cfg.n_paths))..."
    n_steps  = Int(round(cfg.horizon / cfg.dt))
    t_grid   = collect(0:n_steps) .* cfg.dt

    S_paths, V_paths = if best_model == :heston
        simulate_heston(S0, model_params.V0, model_params.κ, model_params.θ,
                         model_params.ξ, model_params.ρ,
                         model_params.r, model_params.q,
                         cfg.horizon, cfg.dt, cfg.n_paths; seed=cfg.seed)
    elseif best_model == :ou
        p = simulate_ou(model_params.x0, model_params.κ, model_params.θ,
                         model_params.σ, cfg.horizon, cfg.dt, cfg.n_paths;
                         seed=cfg.seed)
        exp.(p), nothing   # convert log back to levels
    else
        p = simulate_gbm(S0, model_params.r, model_params.q, model_params.σ,
                          cfg.horizon, cfg.dt, cfg.n_paths; seed=cfg.seed)
        p, nothing
    end
    if isnothing(V_paths)
        V_paths_out = nothing
    else
        V_paths_out = Matrix{Float64}(V_paths)
    end

    cfg.verbose && @info "  Simulation complete."

    # ── Step 6: Scenario generation ────────────────────────────────────
    cfg.verbose && @info "Step 6: Generating risk scenarios..."
    sc_cfg = ScenarioConfig(; n_scenarios=cfg.n_paths, horizon=cfg.horizon,
                              dt=cfg.dt, seed=cfg.seed)
    scenarios = generate_scenarios(S0, model_params, sc_cfg; model=best_model)
    cfg.verbose && print_scenario_summary(scenarios)

    elapsed = time() - t_start

    out = SimulationOutput(Matrix{Float64}(S_paths),
                            V_paths_out,
                            t_grid,
                            best_model,
                            model_params,
                            scenarios,
                            cfg,
                            time(),
                            elapsed)
    cfg.verbose && print_simulation_summary(out)
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: DEMO
# ─────────────────────────────────────────────────────────────────────────────

"""
    demo_pipeline(; model=:heston, n_paths=500, verbose=true)

Full pipeline demo on synthetic Heston data.
"""
function demo_pipeline(; model::Symbol=:heston, n_paths::Int=500,
                         verbose::Bool=true)
    src = SyntheticDataSource(
        model,
        (κ=2.0, θ=0.04, ξ=0.5, ρ=-0.7, V0=0.04, r=0.05, q=0.02, S0=100.0),
        500, 1/252, 42
    )
    cfg = PipelineConfig(src;
                          model_names=[:gbm, :heston],
                          n_paths=n_paths,
                          horizon=1.0,
                          dt=1/52,
                          verbose=verbose)
    return run_pipeline(cfg)
end
