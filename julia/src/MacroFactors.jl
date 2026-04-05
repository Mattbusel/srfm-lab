"""
MacroFactors.jl — Macro Factor Modeling for Crypto

Covers:
  - Fed funds rate sensitivity: crypto beta to rate changes
  - Dollar index (DXY) factor: crypto vs USD moves
  - VIX regime factor: crypto vol scaling with equity vol
  - Inflation factor: crypto as inflation hedge (CPI sensitivity)
  - Liquidity factor: M2 money supply growth correlation
  - Risk-on/risk-off factor: SPY beta of crypto
  - Multi-factor model: regress crypto returns on macro factors
  - Factor timing: when to use macro signals for sizing
  - Nowcasting: real-time macro regime from high-freq data

Pure Julia stdlib only. No external dependencies.
"""
module MacroFactors

using Statistics, LinearAlgebra, Random

export MacroFactor, MacroFactorModel
export rate_sensitivity_factor, dollar_factor, vix_factor
export inflation_factor, liquidity_factor, risk_on_off_factor
export fit_macro_model!, macro_factor_returns
export factor_timing_signal, conditional_factor_betas
export NowcastingModel, fit_nowcasting!, nowcast_regime
export macro_regime_classification, regime_factor_loadings
export factor_contribution_decomposition
export macro_risk_parity, macro_factor_portfolio
export run_macro_factors_demo

# ─────────────────────────────────────────────────────────────
# 1. MACRO FACTOR DEFINITIONS
# ─────────────────────────────────────────────────────────────

"""
    MacroFactor

A named macro factor with its time series and metadata.
"""
struct MacroFactor
    name::String
    series::Vector{Float64}   # factor return/change series
    frequency::Symbol         # :daily, :weekly, :monthly
    sign_convention::Int      # +1 = positive is "risk-on", -1 = positive is "risk-off"
end

"""
    MacroFactorModel

Linear factor model: r_crypto = α + β₁F₁ + β₂F₂ + ... + ε
"""
mutable struct MacroFactorModel
    alpha::Float64
    betas::Vector{Float64}
    residual_vol::Float64
    r_squared::Float64
    factor_names::Vector{String}
    t_stats::Vector{Float64}
    fitted::Bool
    n_obs::Int
end

function MacroFactorModel(n_factors::Int, names::Vector{String}=String[])
    ns = isempty(names) ? ["factor_$i" for i in 1:n_factors] : names
    MacroFactorModel(0.0, zeros(n_factors), 0.0, 0.0, ns, zeros(n_factors), false, 0)
end

# ─────────────────────────────────────────────────────────────
# 2. INDIVIDUAL FACTOR SERIES CONSTRUCTION
# ─────────────────────────────────────────────────────────────

"""
    rate_sensitivity_factor(fed_funds_rate, crypto_returns; window=60)
       -> NamedTuple

Compute crypto sensitivity to Fed funds rate changes.
Returns beta (rate change beta), rolling beta, and hedged residual.
"""
function rate_sensitivity_factor(fed_funds_rate::Vector{Float64},
                                   crypto_returns::Vector{Float64};
                                   window::Int=60)
    n = min(length(fed_funds_rate), length(crypto_returns))
    rate_changes = [0.0; diff(fed_funds_rate[1:n])]

    # Full-sample OLS
    x = rate_changes; y = crypto_returns[1:n]
    beta_full = cov(x, y) / (var(x) + 1e-15)
    alpha_full = mean(y) - beta_full * mean(x)
    residuals  = y .- alpha_full .- beta_full .* x
    r2 = 1 - var(residuals) / (var(y) + 1e-15)

    # Rolling beta
    rolling_beta = zeros(n)
    for t in window:n
        xw = x[t-window+1:t]; yw = y[t-window+1:t]
        rolling_beta[t] = cov(xw, yw) / (var(xw) + 1e-15)
    end

    # Factor series: rate-change surprises (deviations from rolling mean)
    rate_surprise = x .- [t > window ?
                           mean(x[t-window+1:t]) : mean(x[1:t]) for t in 1:n]

    (beta=beta_full, alpha=alpha_full, r_squared=r2, residuals=residuals,
     rolling_beta=rolling_beta, rate_changes=rate_changes,
     factor_series=rate_surprise)
end

"""
    dollar_factor(dxy_changes, crypto_returns; window=60) -> NamedTuple

Compute crypto sensitivity to USD index (DXY) changes.
BTC is often anti-correlated with DXY (weaker USD → stronger BTC).
"""
function dollar_factor(dxy_changes::Vector{Float64},
                        crypto_returns::Vector{Float64};
                        window::Int=60)
    n = min(length(dxy_changes), length(crypto_returns))
    x = dxy_changes[1:n]; y = crypto_returns[1:n]

    beta  = cov(x, y) / (var(x) + 1e-15)
    alpha = mean(y) - beta * mean(x)
    resid = y .- alpha .- beta .* x
    r2    = 1 - var(resid) / (var(y) + 1e-15)

    # DXY factor: negative beta means crypto is anti-USD
    # Signal: DXY falling (positive for crypto if beta < 0)
    signal = -sign(beta) .* x  # flip to make positive = good for crypto

    rolling_beta = zeros(n)
    for t in window:n
        xw = x[t-window+1:t]; yw = y[t-window+1:t]
        rolling_beta[t] = cov(xw, yw) / (var(xw) + 1e-15)
    end

    (beta=beta, alpha=alpha, r_squared=r2, residuals=resid,
     signal=signal, rolling_beta=rolling_beta,
     anti_dollar=beta < -0.1)
end

"""
    vix_factor(vix_levels, crypto_returns; window=60) -> NamedTuple

VIX regime factor for crypto.
High VIX → risk-off → crypto underperforms.
VIX change beta + VIX level regime indicator.
"""
function vix_factor(vix_levels::Vector{Float64},
                     crypto_returns::Vector{Float64};
                     window::Int=60,
                     high_vix_threshold::Float64=25.0,
                     extreme_vix::Float64=35.0)
    n = min(length(vix_levels), length(crypto_returns))
    vix = vix_levels[1:n]; ret = crypto_returns[1:n]
    vix_changes = [0.0; diff(vix)]

    # Beta to VIX changes
    beta_vix = cov(vix_changes, ret) / (var(vix_changes) + 1e-15)

    # Regime indicator
    low_vix    = findall(vix .< high_vix_threshold)
    high_vix   = findall((vix .>= high_vix_threshold) .& (vix .< extreme_vix))
    extreme    = findall(vix .>= extreme_vix)

    # Average returns by regime
    ret_low  = isempty(low_vix)  ? 0.0 : mean(ret[low_vix])
    ret_high = isempty(high_vix) ? 0.0 : mean(ret[high_vix])
    ret_ext  = isempty(extreme)  ? 0.0 : mean(ret[extreme])

    # VIX scaling factor: scale positions inversely with VIX
    vix_scale = [min(high_vix_threshold / max(v, 1.0), 2.0) for v in vix]

    # Signal: sell when VIX spikes (negative for crypto)
    signal = -vix_changes ./ (std(vix_changes) + 1e-10)
    signal = clamp.(signal, -2.0, 2.0)

    rolling_beta = zeros(n)
    for t in window:n
        xw = vix_changes[t-window+1:t]; yw = ret[t-window+1:t]
        rolling_beta[t] = cov(xw, yw) / (var(xw) + 1e-15)
    end

    (beta_vix=beta_vix, ret_low_vix=ret_low, ret_high_vix=ret_high,
     ret_extreme_vix=ret_ext, signal=signal, vix_scale=vix_scale,
     rolling_beta=rolling_beta)
end

"""
    inflation_factor(cpi_changes, crypto_returns; window=60) -> NamedTuple

Inflation factor: BTC as digital gold / inflation hedge.
Tests whether BTC outperforms during high-inflation regimes.
"""
function inflation_factor(cpi_changes::Vector{Float64},
                            crypto_returns::Vector{Float64};
                            window::Int=60)
    n = min(length(cpi_changes), length(crypto_returns))
    x = cpi_changes[1:n]; y = crypto_returns[1:n]

    beta = cov(x, y) / (var(x) + 1e-15)
    # Inflation hedge ratio: positive beta = inflation hedge
    hedge_effectiveness = max(beta / (std(y)/std(x) + 1e-10), 0.0)

    # Rolling 12m correlation (inflation works on slow cycle)
    corr_rolling = zeros(n)
    for t in window:n
        xw = x[t-window+1:t]; yw = y[t-window+1:t]
        corr_rolling[t] = cor(xw, yw)
    end

    # Inflation regime signal: buy crypto when inflation rising
    signal = x ./ (std(x) + 1e-10)
    signal = clamp.(signal, -2.0, 2.0) .* sign(beta)

    (inflation_beta=beta, hedge_effectiveness=hedge_effectiveness,
     rolling_correlation=corr_rolling, signal=signal,
     is_inflation_hedge=beta > 0.1)
end

"""
    liquidity_factor(m2_growth, crypto_returns; window=60) -> NamedTuple

M2 money supply growth factor.
QE / monetary expansion → more liquidity → bullish for crypto.
"""
function liquidity_factor(m2_growth::Vector{Float64},
                            crypto_returns::Vector{Float64};
                            window::Int=60)
    n = min(length(m2_growth), length(crypto_returns))
    # M2 growth leads crypto by ~1-3 months (use lagged signal)
    x_lag1 = [0.0; m2_growth[1:n-1]]
    y = crypto_returns[1:n]

    beta_contemp = cov(m2_growth[1:n], y) / (var(m2_growth[1:n]) + 1e-15)
    beta_lag1    = cov(x_lag1, y) / (var(x_lag1) + 1e-15)

    # Use lagged beta for timing signal
    signal = x_lag1 .* sign(beta_lag1)
    signal = clamp.(signal ./ (std(x_lag1)+1e-10), -2.0, 2.0)

    (beta_contemporaneous=beta_contemp, beta_lagged=beta_lag1,
     signal=signal, liquidity_driven=abs(beta_lag1) > 0.1)
end

"""
    risk_on_off_factor(spy_returns, crypto_returns; window=60) -> NamedTuple

Risk-on/risk-off factor: SPY beta of crypto.
High SPY beta → crypto behaves like risk asset.
During risk-off: both crypto and equities fall.
"""
function risk_on_off_factor(spy_returns::Vector{Float64},
                              crypto_returns::Vector{Float64};
                              window::Int=60)
    n = min(length(spy_returns), length(crypto_returns))
    x = spy_returns[1:n]; y = crypto_returns[1:n]

    beta   = cov(x, y) / (var(x) + 1e-15)
    alpha  = mean(y) - beta * mean(x)
    resid  = y .- alpha .- beta .* x
    r2     = 1 - var(resid) / (var(y) + 1e-15)

    # Risk-on signal: buy when SPY is rising and beta is positive
    signal = x .* sign(beta)  # directional exposure to equity factor

    # Conditional beta: bull vs bear
    bull_mask = x .> 0; bear_mask = x .<= 0
    beta_bull = sum(bull_mask) > 2 ? cov(x[bull_mask], y[bull_mask]) / (var(x[bull_mask])+1e-15) : beta
    beta_bear = sum(bear_mask) > 2 ? cov(x[bear_mask], y[bear_mask]) / (var(x[bear_mask])+1e-15) : beta

    rolling_beta = zeros(n)
    for t in window:n
        xw = x[t-window+1:t]; yw = y[t-window+1:t]
        rolling_beta[t] = cov(xw, yw) / (var(xw) + 1e-15)
    end

    (spy_beta=beta, alpha=alpha, r_squared=r2,
     beta_bull_market=beta_bull, beta_bear_market=beta_bear,
     signal=signal, rolling_beta=rolling_beta, residuals=resid)
end

# ─────────────────────────────────────────────────────────────
# 3. MULTI-FACTOR MODEL
# ─────────────────────────────────────────────────────────────

"""
    fit_macro_model!(model, factors_matrix, crypto_returns; ridge=1e-4)
       -> MacroFactorModel

Fit OLS factor model with optional ridge regularization.
factors_matrix: T × K matrix of factor returns.
"""
function fit_macro_model!(model::MacroFactorModel,
                            factors_matrix::Matrix{Float64},
                            crypto_returns::Vector{Float64};
                            ridge::Float64=1e-4)
    T, K = size(factors_matrix)
    n    = min(T, length(crypto_returns))
    X    = hcat(ones(n), factors_matrix[1:n, :])  # n × (K+1), with intercept
    y    = crypto_returns[1:n]

    # Ridge OLS
    coef = (X'X + ridge * Diagonal(vcat(0.0, ones(K)))) \ (X'y)
    model.alpha = coef[1]
    model.betas = coef[2:end]

    # Residuals and R²
    y_hat = X * coef
    resid = y .- y_hat
    ss_res = sum(resid.^2); ss_tot = sum((y .- mean(y)).^2)
    model.r_squared    = 1 - ss_res / (ss_tot + 1e-15)
    model.residual_vol = std(resid) * sqrt(252)
    model.n_obs        = n
    model.fitted       = true

    # t-statistics
    sigma2  = ss_res / max(n - K - 1, 1)
    XtX_inv = inv(X'X + ridge * Diagonal(vcat(0.0, ones(K))))
    se_coef = sqrt.(max.(sigma2 .* diag(XtX_inv), 0.0))
    model.t_stats = coef ./ (se_coef .+ 1e-15)

    model
end

"""
    macro_factor_returns(model, factors_matrix) -> NamedTuple

Decompose crypto returns into factor contributions.
"""
function macro_factor_returns(model::MacroFactorModel,
                                factors_matrix::Matrix{Float64})
    model.fitted || error("Model not fitted")
    T, K = size(factors_matrix)
    alpha_contrib   = fill(model.alpha, T)
    factor_contribs = zeros(T, K)
    for k in 1:K
        factor_contribs[:, k] = model.betas[k] .* factors_matrix[:, k]
    end
    total_fitted = alpha_contrib .+ vec(sum(factor_contribs, dims=2))
    (alpha=alpha_contrib, factor_contributions=factor_contribs,
     total_fitted=total_fitted)
end

"""
    factor_contribution_decomposition(model, factors, returns) -> NamedTuple

Attribution of realized returns to each factor.
"""
function factor_contribution_decomposition(model::MacroFactorModel,
                                             factors::Matrix{Float64},
                                             returns::Vector{Float64})
    model.fitted || error("Model not fitted")
    T = min(size(factors, 1), length(returns))
    decomp = macro_factor_returns(model, factors[1:T, :])
    residual = returns[1:T] .- decomp.total_fitted

    # Cumulative contributions
    cum_alpha   = cumsum(decomp.alpha)
    cum_factors = cumsum(decomp.factor_contributions, dims=1)
    cum_resid   = cumsum(residual)

    # Variance decomposition
    total_var = var(returns[1:T])
    var_alpha = var(decomp.alpha)
    var_factors = [var(decomp.factor_contributions[:, k]) for k in 1:model.n_obs == 0 ? 0 : size(factors,2)]
    var_resid = var(residual)

    (cumulative_alpha=cum_alpha, cumulative_factor_contribs=cum_factors,
     cumulative_residual=cum_resid, variance_alpha=var_alpha,
     variance_residual=var_resid, r_squared=model.r_squared)
end

# ─────────────────────────────────────────────────────────────
# 4. FACTOR TIMING
# ─────────────────────────────────────────────────────────────

"""
    factor_timing_signal(factor_series, returns; fast=20, slow=60) -> Vector{Float64}

Generate a timing signal: when is a factor expected to be predictive?
Uses momentum of factor IC (information coefficient) to time factor exposure.
"""
function factor_timing_signal(factor_series::Vector{Float64},
                                returns::Vector{Float64};
                                fast::Int=20, slow::Int=60)::Vector{Float64}
    n = min(length(factor_series), length(returns))
    ic_rolling = zeros(n)
    timing = zeros(n)

    # Rolling IC = rank correlation of factor with next-period return
    for t in slow:n-1
        window_f = factor_series[t-slow+1:t]
        window_r = returns[t-slow+2:t+1]
        ic_rolling[t] = cor(window_f, window_r)
    end

    # Fast MA of IC (recent IC)
    ic_fast = zeros(n)
    for t in fast:n
        ic_fast[t] = mean(ic_rolling[t-fast+1:t])
    end

    # Timing signal: use factor when IC is positive and improving
    for t in slow:n
        ic_trend = ic_fast[t] - (t > slow ? ic_fast[t-1] : ic_fast[t])
        timing[t] = clamp(ic_fast[t] + 0.5*ic_trend, -1.0, 1.0)
    end
    timing
end

"""
    conditional_factor_betas(factors_matrix, returns, conditioning_var;
                              n_quantiles=5) -> Matrix{Float64}

Estimate factor betas conditioned on macro state (VIX, rate level, etc.).
Returns n_quantiles × n_factors matrix of conditional betas.
"""
function conditional_factor_betas(factors_matrix::Matrix{Float64},
                                    returns::Vector{Float64},
                                    conditioning_var::Vector{Float64};
                                    n_quantiles::Int=5)::Matrix{Float64}
    T, K = size(factors_matrix)
    n    = min(T, length(returns), length(conditioning_var))

    quantile_levels = range(0.0, 1.0, length=n_quantiles+1)
    cond_betas = zeros(n_quantiles, K)

    for q in 1:n_quantiles
        lo = quantile(conditioning_var[1:n], quantile_levels[q])
        hi = quantile(conditioning_var[1:n], quantile_levels[q+1])
        mask = (conditioning_var[1:n] .>= lo) .& (conditioning_var[1:n] .<= hi)
        sum(mask) < K + 2 && continue

        X = factors_matrix[mask, :]
        y = returns[mask]
        coef = (X'X + 1e-6*I) \ (X'y)
        cond_betas[q, :] = coef
    end
    cond_betas
end

# ─────────────────────────────────────────────────────────────
# 5. MACRO REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────

"""
    macro_regime_classification(rate_level, vix_level, inflation, m2_growth)
       -> Vector{Int}

Classify macro regime at each observation.
Returns integer codes:
  1 = Risk-On (low rates, low VIX, moderate inflation, high M2)
  2 = Risk-Off (high VIX, falling market)
  3 = Inflationary (high inflation, rising rates)
  4 = Deflationary (falling inflation, falling rates, high VIX)
"""
function macro_regime_classification(rate_level::Vector{Float64},
                                       vix_level::Vector{Float64},
                                       inflation::Vector{Float64},
                                       m2_growth::Vector{Float64})::Vector{Int}
    n = min(length.([rate_level, vix_level, inflation, m2_growth])...)

    # Normalize each to [0,1] using rolling percentile
    function percentile_rank(x::Vector{Float64}, window::Int=60)::Vector{Float64}
        n = length(x); rank = zeros(n)
        for t in window:n
            seg = x[t-window+1:t]
            rank[t] = sum(seg .< x[t]) / window
        end
        rank
    end

    r_pct  = percentile_rank(rate_level[1:n])
    v_pct  = percentile_rank(vix_level[1:n])
    inf_pct = percentile_rank(inflation[1:n])
    m2_pct  = percentile_rank(m2_growth[1:n])

    regimes = zeros(Int, n)
    for t in 60:n
        rp = r_pct[t]; vp = v_pct[t]; ip = inf_pct[t]; mp = m2_pct[t]

        if vp < 0.3 && mp > 0.5    # low VIX, high M2
            regimes[t] = 1   # Risk-On
        elseif vp > 0.7             # high VIX
            regimes[t] = 2   # Risk-Off
        elseif ip > 0.7 && rp > 0.5  # high inflation + rising rates
            regimes[t] = 3   # Inflationary
        else
            regimes[t] = 4   # Deflationary / uncertain
        end
    end
    regimes
end

"""
    regime_factor_loadings(regimes, factors_matrix, returns) -> Dict

Compute factor betas for each macro regime.
"""
function regime_factor_loadings(regimes::Vector{Int},
                                  factors_matrix::Matrix{Float64},
                                  returns::Vector{Float64})::Dict
    n = min(length(regimes), size(factors_matrix,1), length(returns))
    unique_regimes = sort(unique(regimes[regimes .> 0]))
    regime_betas = Dict{Int, Vector{Float64}}()

    for r in unique_regimes
        mask = (regimes[1:n] .== r)
        sum(mask) < size(factors_matrix,2) + 2 && continue
        X = factors_matrix[mask, :]
        y = returns[mask]
        coef = (X'X + 1e-6*I) \ (X'y)
        regime_betas[r] = coef
    end
    regime_betas
end

# ─────────────────────────────────────────────────────────────
# 6. NOWCASTING
# ─────────────────────────────────────────────────────────────

"""
    NowcastingModel

Real-time macro regime estimation from high-frequency market data.
Uses high-frequency proxies for slow-moving macro variables.
"""
mutable struct NowcastingModel
    factor_weights::Matrix{Float64}  # K_hf × K_macro
    regime_centroids::Matrix{Float64}  # n_regimes × K_macro
    n_regimes::Int
    n_hf_factors::Int
    n_macro_factors::Int
    fitted::Bool
end

function NowcastingModel(n_regimes::Int=4, n_hf::Int=5, n_macro::Int=4)
    NowcastingModel(zeros(n_hf, n_macro), zeros(n_regimes, n_macro),
                    n_regimes, n_hf, n_macro, false)
end

"""
    fit_nowcasting!(model, hf_data, macro_data; n_iter=50) -> NowcastingModel

Fit nowcasting model to align high-frequency proxies with macro variables.
hf_data:    T × K_hf matrix (high-freq: VIX, equity vol, credit spreads, etc.)
macro_data: T × K_macro matrix (slow macro: rates, CPI, M2, etc.)
"""
function fit_nowcasting!(model::NowcastingModel,
                           hf_data::Matrix{Float64},
                           macro_data::Matrix{Float64};
                           n_iter::Int=50)
    T, K_hf   = size(hf_data)
    T2, K_mac = size(macro_data)
    n = min(T, T2)

    # Step 1: Learn weights to map HF → macro via OLS
    model.factor_weights = zeros(K_hf, K_mac)
    for k in 1:K_mac
        X = hf_data[1:n, :]
        y = macro_data[1:n, k]
        model.factor_weights[:, k] = (X'X + 1e-4*I) \ (X'y)
    end

    # Step 2: K-means on mapped macro factors to find regime centroids
    predicted_macro = hf_data[1:n, :] * model.factor_weights  # n × K_mac
    model.regime_centroids = _kmeans_centroids(predicted_macro, model.n_regimes)
    model.fitted = true
    model
end

"""Simple K-means centroids."""
function _kmeans_centroids(X::Matrix{Float64}, k::Int;
                             rng=MersenneTwister(42))::Matrix{Float64}
    n, d = size(X)
    k = min(k, n)
    idx = rand(rng, 1:n, k)
    centroids = X[idx, :]
    for _ in 1:100
        labels = [argmin([norm(X[i,:] - centroids[c,:]) for c in 1:k]) for i in 1:n]
        new_c  = similar(centroids)
        for c in 1:k
            members = X[labels .== c, :]
            new_c[c, :] = isempty(members) ? centroids[c,:] : vec(mean(members, dims=1))
        end
        norm(new_c - centroids) < 1e-6 && break
        centroids = new_c
    end
    centroids
end

"""
    nowcast_regime(model, hf_observation) -> (regime, probabilities)

Nowcast current macro regime from high-frequency market data.
"""
function nowcast_regime(model::NowcastingModel,
                          hf_observation::Vector{Float64})
    model.fitted || return (1, fill(1.0/model.n_regimes, model.n_regimes))

    # Map HF → macro
    macro_pred = model.factor_weights' * hf_observation

    # Soft assignment via inverse distance weighting
    dists = [norm(macro_pred .- model.regime_centroids[k, :])
             for k in 1:model.n_regimes]
    inv_d = 1.0 ./ (dists .+ 1e-10)
    probs = inv_d ./ sum(inv_d)
    (argmax(probs), probs)
end

# ─────────────────────────────────────────────────────────────
# 7. MACRO-DRIVEN PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────

"""
    macro_risk_parity(factor_betas, factor_vols, target_vol=0.15) -> Vector{Float64}

Macro factor risk parity: allocate weights so each macro factor
contributes equally to portfolio volatility.
"""
function macro_risk_parity(factor_betas::Vector{Float64},
                             factor_vols::Vector{Float64},
                             target_vol::Float64=0.15)::Vector{Float64}
    K = length(factor_betas)
    # Marginal risk contribution = beta_k * vol_k
    mrc = abs.(factor_betas) .* factor_vols
    total_mrc = sum(mrc)
    total_mrc < 1e-10 && return fill(1.0/K, K)

    # Equal risk contribution weights
    weights = (target_vol / K) ./ (mrc .+ 1e-10)
    weights ./= sum(abs.(weights))
    weights
end

"""
    macro_factor_portfolio(model, regime, assets_betas_per_regime)
       -> Vector{Float64}

Build a portfolio that is neutral to undesired macro factors in a given regime.
assets_betas_per_regime: n_assets × n_factors matrix for the current regime.
"""
function macro_factor_portfolio(model::MacroFactorModel,
                                  regime::Int,
                                  asset_factor_betas::Matrix{Float64};
                                  target_factor_exposure::Vector{Float64}=Float64[])
    n_assets, n_factors = size(asset_factor_betas)
    # Default: zero exposure to all macro factors (factor-neutral)
    target = isempty(target_factor_exposure) ? zeros(n_factors) : target_factor_exposure

    # Minimum variance portfolio subject to factor neutrality
    # min w'w  s.t. B'w = target, 1'w = 1
    B = asset_factor_betas
    # Augmented system: [B' ; 1'] * w = [target; 1]
    n_constraints = n_factors + 1
    A = vcat(B', ones(1, n_assets))
    b = vcat(target, [1.0])

    # Least-norm solution: w = A' * (A*A')^{-1} * b
    AAt = A * A'
    try
        w = A' * ((AAt + 1e-6*I) \ b)
        return w
    catch
        return fill(1.0/n_assets, n_assets)
    end
end

# ─────────────────────────────────────────────────────────────
# 8. DEMO
# ─────────────────────────────────────────────────────────────

"""
    generate_synthetic_macro(n; rng=...) -> NamedTuple

Generate synthetic macro + crypto data for demo.
"""
function generate_synthetic_macro(n::Int=500; rng=MersenneTwister(42))
    # Fed funds rate: slow mean-reverting
    fed_rate = Float64[0.05]
    for t in 2:n
        dr = 0.002 * (0.04 - fed_rate[end]) + 0.005 * randn(rng)
        push!(fed_rate, max(fed_rate[end] + dr, 0.0))
    end

    # DXY: random walk with drift
    dxy = cumsum(randn(rng, n) .* 0.005)
    dxy_changes = [0.0; diff(dxy)]

    # VIX: mean-reverting to 20
    vix = Float64[20.0]
    for t in 2:n
        dv = 0.1*(20.0 - vix[end]) + 2.0*randn(rng)
        push!(vix, max(vix[end] + dv, 10.0))
    end

    # CPI: slow-moving inflation
    cpi = cumsum(0.002 .+ 0.001 .* randn(rng, n))
    cpi_changes = [0.0; diff(cpi)]

    # M2: step-function with gradual growth
    m2_growth = 0.005 .+ 0.002 .* randn(rng, n)

    # SPY returns: correlated with macro
    spy = randn(rng, n) .* 0.012 .- 0.001 .* (vix .- 20) / 10

    # BTC returns: higher beta, anti-USD, inflation-sensitive
    btc = 3.0 .* spy .+ (-2.0) .* dxy_changes .+
          1.5 .* m2_growth .+ 0.5 .* cpi_changes .+
          randn(rng, n) .* 0.025

    (fed_rate=fed_rate, dxy_changes=dxy_changes, vix=vix,
     cpi_changes=cpi_changes, m2_growth=m2_growth,
     spy_returns=spy, btc_returns=btc)
end

"""
    run_macro_factors_demo() -> Nothing
"""
function run_macro_factors_demo()
    println("=" ^ 60)
    println("MACRO FACTOR MODELING FOR CRYPTO DEMO")
    println("=" ^ 60)

    rng = MersenneTwister(42)
    d = generate_synthetic_macro(500; rng=rng)
    n = length(d.btc_returns)

    println("\n1. Rate Sensitivity Factor")
    rs = rate_sensitivity_factor(d.fed_rate, d.btc_returns)
    println("  Rate beta:    $(round(rs.beta,digits=3))")
    println("  Rate R²:      $(round(rs.r_squared,digits=3))")
    println("  Alpha (ann): $(round(rs.alpha*252*100,digits=2)) bps/day × 252")

    println("\n2. Dollar (DXY) Factor")
    df = dollar_factor(d.dxy_changes, d.btc_returns)
    println("  DXY beta:     $(round(df.beta,digits=3))")
    println("  Anti-dollar:  $(df.anti_dollar)")
    println("  DXY R²:       $(round(df.r_squared,digits=3))")

    println("\n3. VIX Regime Factor")
    vf = vix_factor(d.vix, d.btc_returns)
    println("  VIX change beta:    $(round(vf.beta_vix,digits=3))")
    println("  Return in low VIX:  $(round(vf.ret_low_vix*252*100,digits=2)) bps ann.")
    println("  Return in high VIX: $(round(vf.ret_high_vix*252*100,digits=2)) bps ann.")

    println("\n4. Inflation Factor")
    inf_f = inflation_factor(d.cpi_changes, d.btc_returns)
    println("  Inflation beta:     $(round(inf_f.inflation_beta,digits=3))")
    println("  Hedge effectiveness: $(round(inf_f.hedge_effectiveness,digits=3))")
    println("  Is inflation hedge:  $(inf_f.is_inflation_hedge)")

    println("\n5. Liquidity (M2) Factor")
    liq = liquidity_factor(d.m2_growth, d.btc_returns)
    println("  M2 beta (contemp):  $(round(liq.beta_contemporaneous,digits=3))")
    println("  M2 beta (lagged):   $(round(liq.beta_lagged,digits=3))")
    println("  Liquidity driven:   $(liq.liquidity_driven)")

    println("\n6. Risk-On/Off (SPY Beta)")
    ro = risk_on_off_factor(d.spy_returns, d.btc_returns)
    println("  SPY beta:           $(round(ro.spy_beta,digits=3))")
    println("  Beta (bull market): $(round(ro.beta_bull_market,digits=3))")
    println("  Beta (bear market): $(round(ro.beta_bear_market,digits=3))")
    println("  SPY R²:             $(round(ro.r_squared,digits=3))")

    println("\n7. Multi-Factor Model")
    F_matrix = hcat(
        rs.factor_series,
        df.signal,
        vf.signal,
        inf_f.signal,
        liq.signal,
        ro.signal
    )
    factor_names = ["rate","dxy","vix","inflation","m2","spy"]
    macro_model = MacroFactorModel(6, factor_names)
    fit_macro_model!(macro_model, F_matrix, d.btc_returns; ridge=1e-4)
    println("  Factor betas:")
    for (name, b, t) in zip(factor_names, macro_model.betas, macro_model.t_stats[2:end])
        println("    $name: β=$(round(b,digits=4)) (t=$(round(t,digits=2)))")
    end
    println("  Multi-factor R²: $(round(macro_model.r_squared,digits=4))")
    println("  Residual vol (ann): $(round(macro_model.residual_vol*100,digits=2))%")

    println("\n8. Macro Regime Classification")
    regimes = macro_regime_classification(d.fed_rate, d.vix, d.cpi_changes, d.m2_growth)
    regime_labels = Dict(0=>"Unknown",1=>"Risk-On",2=>"Risk-Off",3=>"Inflationary",4=>"Deflationary")
    counts = [sum(regimes .== r) for r in 1:4]
    for (r, c) in enumerate(counts)
        println("  $(regime_labels[r]): $(c) days ($(round(c/n*100,digits=1))%)")
    end

    # Returns by regime
    for r in 1:4
        mask = regimes .== r
        sum(mask) < 5 && continue
        avg_ret = mean(d.btc_returns[mask]) * 252 * 100
        println("  BTC avg return in $(regime_labels[r]): $(round(avg_ret,digits=2)) bps×252")
    end

    println("\n9. Nowcasting Model")
    hf_data = hcat(d.vix, abs.(d.spy_returns), abs.(d.dxy_changes),
                   d.m2_growth, d.cpi_changes)
    macro_data = hcat(d.fed_rate, d.vix, d.cpi_changes, d.m2_growth)
    nc_model = NowcastingModel(4, 5, 4)
    fit_nowcasting!(nc_model, hf_data, macro_data)
    last_obs = vec(hf_data[end, :])
    regime_now, probs = nowcast_regime(nc_model, last_obs)
    println("  Nowcast regime: $(regime_labels[regime_now])")
    println("  Regime probs:   $(round.(probs,digits=3))")

    println("\n10. Factor Timing Signal")
    timing = factor_timing_signal(rs.factor_series, d.btc_returns; fast=20, slow=60)
    timing_ic = cor(timing[61:end-1], d.btc_returns[62:end])
    println("  Rate factor timing IC: $(round(timing_ic,digits=4))")
    println("  Mean timing signal:    $(round(mean(timing),digits=4))")

    println("\nDone.")
    nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Cross-Asset Contagion and Spillover Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    spillover_index(returns, lag, forecast_horizon)

Diebold-Yilmaz spillover index via rolling VAR(lag) approximation.
Uses simplified FEVD (forecast error variance decomposition) from OLS impulse
responses (Cholesky ordering).
Returns: (total_spillover, from_table, to_table)
"""
function spillover_index(returns::Matrix{Float64}, lag::Int=1,
                          forecast_horizon::Int=10)
    T, n = size(returns)
    # Estimate VAR(lag) equation by equation
    A = zeros(n, n * lag)    # companion coefficient matrix row
    A_list = Matrix{Float64}[]
    for i in 1:n
        Y = returns[(lag+1):T, i]
        X = hcat([returns[(lag+1-l):(T-l), :] for l in 1:lag]...)
        # OLS
        A_row = (X' * X + I(size(X,2)) * 1e-6) \ (X' * Y)
        push!(A_list, reshape(A_row, n, lag))
    end
    # Sigma (residual covariance)
    resid = zeros(T - lag, n)
    for i in 1:n
        Y = returns[(lag+1):T, i]
        X = hcat([returns[(lag+1-l):(T-l), :] for l in 1:lag]...)
        resid[:, i] = Y - X * vec(A_list[i])
    end
    Sigma = resid' * resid / (T - lag - n * lag)

    # IRF via recursive substitution (H steps)
    Psi = [zeros(n, n) for _ in 1:forecast_horizon]
    Psi[1] = Matrix{Float64}(I(n))
    for h in 2:forecast_horizon
        for l in 1:min(h-1, lag)
            for i in 1:n
                Psi[h][i, :] .+= A_list[i][:, l]' * Psi[h-l]
            end
        end
    end

    # Generalised FEVD (normalised)
    FEVD = zeros(n, n)
    sigma_sq = diag(Sigma)
    for i in 1:n, j in 1:n
        num = sum((Psi[h][i, :] ⋅ Sigma[:, j])^2 / Sigma[j, j] for h in 1:forecast_horizon)
        den = sum(Psi[h][i, :] ⋅ (Sigma * Psi[h][i, :]) for h in 1:forecast_horizon)
        FEVD[i, j] = num / (den + 1e-8)
    end
    FEVD ./= sum(FEVD, dims=2)   # row-normalise

    total = (sum(FEVD) - tr(FEVD)) / n * 100
    from_v = [sum(FEVD[i, :]) - FEVD[i, i] for i in 1:n] ./ n .* 100
    to_v   = [sum(FEVD[:, j]) - FEVD[j, j] for j in 1:n] ./ n .* 100
    return (total=total, from_spillover=from_v, to_spillover=to_v, fevd=FEVD)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 10 – Yield Curve Factor Sensitivities
# ─────────────────────────────────────────────────────────────────────────────

"""
    yield_curve_factors(yields_matrix)

Decompose yield curve movements into 3 PCA factors:
  PC1 ≈ level shift, PC2 ≈ slope (twist), PC3 ≈ curvature (butterfly).

`yields_matrix`: T × M where M = number of tenors.
Returns (factor_scores T×3, loadings M×3, explained_var 3-vector).
"""
function yield_curve_factors(yields_matrix::Matrix{Float64})
    T, M = size(yields_matrix)
    mu   = mean(yields_matrix, dims=1)[1, :]
    Y    = yields_matrix .- mu'
    C    = Y' * Y ./ T                         # M×M covariance
    # Power iteration for top 3 eigenvectors
    k    = min(3, M)
    vecs = zeros(M, k)
    Y_work = copy(C)
    for j in 1:k
        v = randn(M); v ./= norm(v)
        for _ in 1:200
            v = Y_work * v; v ./= norm(v)
        end
        vecs[:, j] = v
        lam = dot(v, C * v)
        Y_work -= lam * v * v'
    end
    scores     = Y * vecs
    total_var  = tr(C)
    explained  = [dot(vecs[:, j], C * vecs[:, j]) / total_var for j in 1:k]
    return (scores=scores, loadings=vecs, explained_var=explained)
end

"""
    rate_surprise_beta(asset_returns, rate_surprises, window)

Rolling regression of asset returns on rate surprises (e.g., Fed meeting
day changes in 2-year yield) over a rolling window.
Returns time series of betas and t-statistics.
"""
function rate_surprise_beta(asset_returns::Vector{Float64},
                              rate_surprises::Vector{Float64},
                              window::Int=60)
    n = length(asset_returns)
    @assert length(rate_surprises) == n
    betas = fill(NaN, n); tstats = fill(NaN, n)
    for i in (window+1):n
        y = asset_returns[i-window+1:i]
        x = rate_surprises[i-window+1:i]
        xm = mean(x); ym = mean(y)
        beta = sum((x .- xm) .* (y .- ym)) / (sum((x .- xm).^2) + 1e-12)
        resid = y .- (ym + beta .* (x .- xm))
        se = sqrt(sum(resid.^2) / (window - 2) / (sum((x .- xm).^2) + 1e-12))
        betas[i]  = beta
        tstats[i] = beta / (se + 1e-12)
    end
    return (betas=betas, tstats=tstats)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 11 – Global Macro Momentum and Carry Strategies
# ─────────────────────────────────────────────────────────────────────────────

"""
    macro_momentum_signal(factor_returns, lookback, skip)

Time-series momentum applied to macro factor returns:
  signal = sign(mean return over [t-lookback-skip, t-skip])
Avoids short-term reversal by skipping the last `skip` days.
"""
function macro_momentum_signal(factor_returns::Vector{Float64},
                                 lookback::Int=252, skip::Int=21)
    n   = length(factor_returns)
    sig = fill(0.0, n)
    for i in (lookback + skip + 1):n
        past = factor_returns[i-lookback-skip:i-skip-1]
        sig[i] = sign(mean(past))
    end
    return sig
end

"""
    carry_signal(forward_rate, spot_rate) -> Vector{Float64}

Carry = forward_rate - spot_rate (in log-price units).
Positive carry → long, negative → short.
"""
function carry_signal(forward_rate::Vector{Float64},
                       spot_rate::Vector{Float64})
    carry = forward_rate .- spot_rate
    return sign.(carry)
end

"""
    macro_carry_momentum_portfolio(factor_signals, factor_returns,
                                    carry_weights, mom_weights) -> Vector{Float64}

Combine carry and momentum signals into a diversified macro factor portfolio.
Returns time series of portfolio returns.
"""
function macro_carry_momentum_portfolio(signals_carry::Matrix{Float64},
                                         signals_mom::Matrix{Float64},
                                         factor_returns::Matrix{Float64};
                                         carry_w::Float64=0.5,
                                         mom_w::Float64=0.5)
    T, k = size(factor_returns)
    port_rets = zeros(T)
    for t in 1:T
        combined = carry_w .* signals_carry[t, :] + mom_w .* signals_mom[t, :]
        # normalise to unit exposure
        w = combined ./ (sum(abs.(combined)) + 1e-8)
        port_rets[t] = dot(w, factor_returns[t, :])
    end
    return port_rets
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 12 – Inflation Nowcasting and Real Rate Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    RealRateModel

Decomposes nominal rates into real rate and expected inflation:
  r_nominal = r_real + π_expected + term_premium
Estimated via rolling OLS of breakeven inflation on CPI surprises.
"""
mutable struct RealRateModel
    window::Int
    real_rate::Vector{Float64}
    expected_inflation::Vector{Float64}
    term_premium::Vector{Float64}
end

function RealRateModel(window::Int=60)
    RealRateModel(window, Float64[], Float64[], Float64[])
end

function fit_real_rate_model!(model::RealRateModel,
                               nominal_rate::Vector{Float64},
                               tips_rate::Vector{Float64},
                               cpi_yoy::Vector{Float64})
    n = length(nominal_rate)
    rr = fill(NaN, n); pi_exp = fill(NaN, n); tp = fill(NaN, n)
    for i in (model.window+1):n
        brk  = mean(nominal_rate[i-model.window:i] .- tips_rate[i-model.window:i])
        rr[i]    = tips_rate[i]
        pi_exp[i]= brk
        tp[i]    = nominal_rate[i] - rr[i] - pi_exp[i]
    end
    model.real_rate          = rr
    model.expected_inflation = pi_exp
    model.term_premium       = tp
    return model
end

"""
    inflation_nowcast(commodity_rets, trade_balance, pmi_manuf, window)

Simple nowcast of monthly CPI via rolling regression on high-frequency proxies.
Returns predicted CPI change and residual.
"""
function inflation_nowcast(commodity_rets::Vector{Float64},
                             trade_balance::Vector{Float64},
                             pmi_manuf::Vector{Float64},
                             cpi_actual::Vector{Float64},
                             window::Int=24)
    n = length(cpi_actual)
    predicted = fill(NaN, n); residuals = fill(NaN, n)
    for i in (window+1):n
        y = cpi_actual[i-window+1:i]
        X = hcat(ones(window),
                  commodity_rets[i-window+1:i],
                  trade_balance[i-window+1:i],
                  pmi_manuf[i-window+1:i])
        beta = (X' * X + I(4) * 1e-4) \ (X' * y)
        x_new = [1.0, commodity_rets[i], trade_balance[i], pmi_manuf[i]]
        predicted[i] = dot(beta, x_new)
        residuals[i] = cpi_actual[i] - predicted[i]
    end
    return (predicted=predicted, residuals=residuals)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 13 – Macro Factor Dashboard
# ─────────────────────────────────────────────────────────────────────────────

"""
    macro_dashboard(asset_name, asset_returns, factor_matrix, factor_names;
                     window)

Print a concise macro factor dashboard showing rolling betas, t-stats,
R², and IC for each factor.
"""
function macro_dashboard(asset_name::String,
                          asset_returns::Vector{Float64},
                          factor_matrix::Matrix{Float64},
                          factor_names::Vector{String};
                          window::Int=60)
    T, k = size(factor_matrix)
    n    = length(asset_returns)
    @assert T == n "asset_returns and factor_matrix must have same length"

    println("=" ^ 70)
    println("Macro Factor Dashboard: $asset_name  (rolling window = $window)")
    println("=" ^ 70)
    @printf("%-20s  %8s  %8s  %8s  %8s\n",
            "Factor", "Beta", "t-stat", "R²", "IC")
    println("-" ^ 70)
    for j in 1:k
        f   = factor_matrix[:, j]
        mu_f = mean(f[end-window+1:end])
        beta_vec, tstat_vec = rate_surprise_beta(asset_returns, f, window)
        beta  = isnan(beta_vec[end]) ? 0.0 : beta_vec[end]
        tstat = isnan(tstat_vec[end]) ? 0.0 : tstat_vec[end]
        # Pearson correlation as IC proxy
        y = asset_returns[end-window+1:end]
        x = f[end-window+1:end]
        ic = cor(x, y)
        # R² from beta
        r2 = isnan(ic) ? 0.0 : ic^2
        @printf("%-20s  %8.4f  %8.3f  %8.3f  %8.3f\n",
                factor_names[j], beta, tstat, r2, ic)
    end
    println("=" ^ 70)
    nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 14 – Cross-Sectional Macro Factor Ranking
# ─────────────────────────────────────────────────────────────────────────────

"""
    cross_sectional_factor_rank(factor_exposures, returns, n_portfolios)

Sort assets into `n_portfolios` quantile buckets by factor exposure,
compute equal-weighted return of each bucket, and return the long-short spread.

`factor_exposures`: T × n matrix (time × assets)
`returns`:          T × n matrix of asset returns
"""
function cross_sectional_factor_rank(factor_exposures::Matrix{Float64},
                                      returns::Matrix{Float64},
                                      n_portfolios::Int=5)
    T, n = size(returns)
    @assert size(factor_exposures) == (T, n)
    bucket_rets = zeros(T, n_portfolios)
    for t in 1:T
        exp_t = factor_exposures[t, :]
        ret_t = returns[t, :]
        order = sortperm(exp_t)
        bucket_size = n ÷ n_portfolios
        for b in 1:n_portfolios
            start_i = (b-1) * bucket_size + 1
            stop_i  = b == n_portfolios ? n : b * bucket_size
            idxs = order[start_i:stop_i]
            bucket_rets[t, b] = mean(ret_t[idxs])
        end
    end
    ls_spread = bucket_rets[:, n_portfolios] .- bucket_rets[:, 1]
    ic_series = Float64[]
    for t in 1:T
        push!(ic_series, cor(factor_exposures[t, :], returns[t, :]))
    end
    return (bucket_returns=bucket_rets, ls_spread=ls_spread,
            ic_series=ic_series, mean_ic=mean(filter(!isnan, ic_series)))
end

"""
    factor_decay_analysis(signal, returns, horizons)

Measure IC at multiple forward horizons to understand how quickly a
macro signal's predictive power decays.
Returns a vector of mean ICs one per horizon.
"""
function factor_decay_analysis(signal::Vector{Float64},
                                 returns::Vector{Float64},
                                 horizons::Vector{Int}=[1,5,10,21,63])
    n = length(signal)
    ics = Float64[]
    for h in horizons
        forward_rets = vcat(fill(NaN, h), returns[1:n-h] .* 0.0 .+
                            [sum(returns[i:min(i+h-1,n)]) for i in 1:(n-h)])
        valid = .!isnan.(signal) .& .!isnan.(forward_rets)
        if sum(valid) < 10
            push!(ics, NaN)
        else
            push!(ics, cor(signal[valid], forward_rets[valid]))
        end
    end
    return (horizons=horizons, ics=ics)
end

"""
    macro_factor_zscore(factor_series, window)

Rolling z-score of a macro factor: (x - μ_window) / σ_window.
Converts level series to standardised signal for cross-asset comparison.
"""
function macro_factor_zscore(factor_series::Vector{Float64}, window::Int=60)
    n  = length(factor_series)
    zs = fill(NaN, n)
    for i in (window+1):n
        sub = factor_series[i-window+1:i]
        mu  = mean(sub); sg = std(sub)
        sg > 0 && (zs[i] = (factor_series[i] - mu) / sg)
    end
    return zs
end

"""
    macro_factor_composite(z_scores_matrix, ic_weights)

Build a composite macro score as IC-weighted average of standardised signals.
`z_scores_matrix`: T × k, each column a zscore'd factor.
`ic_weights`:      k-vector of information coefficients (or equal if not provided).
"""
function macro_factor_composite(z_scores::Matrix{Float64},
                                  ic_weights::Union{Vector{Float64},Nothing}=nothing)
    T, k = size(z_scores)
    w = ic_weights === nothing ? ones(k) ./ k : ic_weights ./ (sum(abs.(ic_weights)) + 1e-8)
    composite = zeros(T)
    for t in 1:T
        row = z_scores[t, :]
        valid = .!isnan.(row)
        if any(valid)
            composite[t] = dot(w[valid], row[valid]) / (sum(abs.(w[valid])) + 1e-8)
        end
    end
    return composite
end

end  # module MacroFactors
