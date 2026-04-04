"""
FactorModel — Full factor model implementation for SRFM trades.

Implements: Fama-French style factor construction, cross-sectional regression,
IC/ICIR, factor portfolio construction, alpha attribution, and multi-factor
combination via Elastic Net and PCA.
"""
module FactorModel

using LinearAlgebra
using Statistics
using StatsBase
using Distributions
using DataFrames
using GLM
using Optim
using Random

export Factor, FactorUniverse, compute_factors, cross_sectional_regression
export factor_ic, factor_icir, factor_portfolio
export barra_risk_model, factor_attribution
export pca_factors, elastic_net_factor_selection
export factor_decay_analysis, factor_turnover
export alpha_model, predict_returns

# ─────────────────────────────────────────────────────────────────────────────
# 1. Factor Data Structures
# ─────────────────────────────────────────────────────────────────────────────

"""
    Factor

A single factor with exposures (loadings) per asset.
"""
struct Factor
    name::String
    exposures::Vector{Float64}    # per-asset z-scores
    raw_values::Vector{Float64}   # raw factor values before standardisation
    date::Int                      # bar/date index
end

"""
    FactorUniverse

Collection of factors evaluated across all assets and time.
"""
struct FactorUniverse
    factors::Dict{String, Matrix{Float64}}   # factor_name → (T × N) matrix
    factor_names::Vector{String}
    n_assets::Int
    n_bars::Int
    returns::Matrix{Float64}                  # (T × N) forward returns
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Factor Construction from OHLCV / BH data
# ─────────────────────────────────────────────────────────────────────────────

"""
    standardise_factor(raw, method=:cross_section) → Vector{Float64}

Standardise a factor cross-sectionally to z-scores. Winsorise at ±3σ.
"""
function standardise_factor(raw::Vector{Float64};
                              method::Symbol=:cross_section)::Vector{Float64}
    valid = filter(!isnan, raw)
    isempty(valid) && return fill(0.0, length(raw))

    m = mean(valid); s = std(valid)
    s < 1e-10 && return fill(0.0, length(raw))

    z = (raw .- m) ./ s
    # Winsorise
    z = clamp.(z, -3.0, 3.0)
    # Re-normalise after winsorising
    mz = mean(filter(!isnan, z)); sz = std(filter(!isnan, z))
    sz < 1e-10 && return fill(0.0, length(raw))
    return (z .- mz) ./ sz
end

"""
    compute_factors(prices_matrix, volumes_matrix; lookbacks=[5,21,63]) → FactorUniverse

Compute a standard set of factors from price + volume data.

Factors computed:
  Momentum_5, Momentum_21, Momentum_63   — log returns over lookback
  Reversal_1                              — 1-bar return reversal
  Vol_21                                  — 21-bar realised volatility
  Vol_ratio                               — short-vol / long-vol ratio
  Skewness_21                             — 21-bar return skewness
  Volume_trend                            — log(avg_vol_5 / avg_vol_21)
  Price_range                             — (high-low) / close
  RSI_14                                  — z-scored RSI
"""
function compute_factors(prices_matrix::Matrix{Float64},
                          volumes_matrix::Union{Matrix{Float64}, Nothing}=nothing;
                          lookbacks::Vector{Int}=[5, 21, 63])::FactorUniverse

    T, N = size(prices_matrix)
    @assert T > maximum(lookbacks) + 1

    rets = diff(log.(prices_matrix), dims=1)    # (T-1) × N
    T_r  = size(rets, 1)

    factor_dict = Dict{String, Matrix{Float64}}()

    # Momentum factors
    for lb in lookbacks
        fname = "Momentum_$lb"
        fmat  = fill(NaN, T_r, N)
        for t in lb:T_r
            raw = vec(sum(rets[t-lb+1:t, :], dims=1))
            fmat[t, :] = standardise_factor(raw)
        end
        factor_dict[fname] = fmat
    end

    # Short-term reversal
    fmat_rev = fill(NaN, T_r, N)
    for t in 1:T_r
        fmat_rev[t, :] = standardise_factor(-rets[t, :])
    end
    factor_dict["Reversal_1"] = fmat_rev

    # Volatility
    vol_lb = 21
    fmat_vol = fill(NaN, T_r, N)
    for t in vol_lb:T_r
        raw = [std(rets[t-vol_lb+1:t, j]) for j in 1:N]
        fmat_vol[t, :] = standardise_factor(raw)
    end
    factor_dict["Vol_21"] = fmat_vol

    # Volatility ratio (short / long)
    fmat_volr = fill(NaN, T_r, N)
    for t in 63:T_r
        short_vol = [std(rets[t-4:t, j]) for j in 1:N]
        long_vol  = [std(rets[t-62:t, j]) for j in 1:N]
        ratio     = short_vol ./ max.(long_vol, 1e-10)
        fmat_volr[t, :] = standardise_factor(ratio)
    end
    factor_dict["Vol_ratio"] = fmat_volr

    # Skewness
    fmat_skew = fill(NaN, T_r, N)
    for t in vol_lb:T_r
        raw = [begin
            w = rets[t-vol_lb+1:t, j]
            m = mean(w); s = std(w)
            s < 1e-10 ? 0.0 : mean(((w .- m) ./ s).^3)
        end for j in 1:N]
        fmat_skew[t, :] = standardise_factor(raw)
    end
    factor_dict["Skewness_21"] = fmat_skew

    # Volume trend
    if !isnothing(volumes_matrix) && size(volumes_matrix, 1) >= T
        log_vol = log.(max.(volumes_matrix[1:end-1, :], 1.0))
        fmat_vtend = fill(NaN, T_r, N)
        for t in 21:T_r
            short_v = vec(mean(log_vol[t-4:t, :], dims=1))
            long_v  = vec(mean(log_vol[t-20:t, :], dims=1))
            fmat_vtend[t, :] = standardise_factor(short_v .- long_v)
        end
        factor_dict["Volume_trend"] = fmat_vtend
    end

    # Price range (proxy for intraday vol / liquidity)
    fmat_range = fill(NaN, T_r, N)
    for t in 1:T_r
        # Approximate from returns: |ret| as range proxy
        fmat_range[t, :] = standardise_factor(abs.(rets[t, :]))
    end
    factor_dict["Price_range"] = fmat_range

    # RSI proxy: smoothed over-sold/-bought
    fmat_rsi = fill(NaN, T_r, N)
    rsi_lb = 14
    for t in rsi_lb:T_r
        raw_rsi = zeros(N)
        for j in 1:N
            w = rets[t-rsi_lb+1:t, j]
            gains  = sum(max(r, 0.0) for r in w)
            losses = sum(max(-r, 0.0) for r in w)
            raw_rsi[j] = losses < 1e-10 ? 100.0 :
                         100.0 - 100.0 / (1.0 + gains / losses)
        end
        fmat_rsi[t, :] = standardise_factor(raw_rsi)
    end
    factor_dict["RSI_14"] = fmat_rsi

    # 1-bar forward returns as the target (for IC computation)
    fwd_rets = fill(NaN, T_r, N)
    for t in 1:T_r-1
        fwd_rets[t, :] = rets[t+1, :]
    end

    factor_names = collect(keys(factor_dict))
    return FactorUniverse(factor_dict, factor_names, N, T_r, fwd_rets)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Information Coefficient Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    factor_ic(fu, factor_name) → Vector{Float64}

Compute cross-sectional IC (Spearman rank correlation between factor exposures
and forward returns) for each period.
"""
function factor_ic(fu::FactorUniverse, factor_name::String)::Vector{Float64}
    @assert haskey(fu.factors, factor_name)
    F = fu.factors[factor_name]    # T × N
    R = fu.returns                  # T × N
    T = fu.n_bars

    ic_series = fill(NaN, T)
    for t in 1:T-1
        f = F[t, :]; r = R[t, :]
        valid = .!isnan.(f) .& .!isnan.(r)
        sum(valid) < 5 && continue

        rf = ordinalrank(Float64.(f[valid]))
        rr = ordinalrank(Float64.(r[valid]))
        ic_series[t] = cor(rf, rr)
    end
    return ic_series
end

"""
    factor_icir(ic_series; annualize=252) → Float64

IC Information Ratio: mean(IC) / std(IC) * sqrt(annualize).
"""
function factor_icir(ic_series::Vector{Float64}; annualize::Int=252)::Float64
    valid = filter(!isnan, ic_series)
    length(valid) < 5 && return 0.0
    s = std(valid)
    s < 1e-10 && return 0.0
    return mean(valid) / s * sqrt(annualize)
end

"""
    factor_decay(fu, factor_name, max_lag=20) → Vector{Float64}

IC at different forward horizons (lag = 1, 2, ..., max_lag bars).
"""
function factor_decay_analysis(fu::FactorUniverse, factor_name::String,
                                 max_lag::Int=20)::Vector{Float64}
    F = fu.factors[factor_name]
    T = fu.n_bars
    N = fu.n_assets
    rets = diff(log.(ones(T+1, N)), dims=1)   # placeholder — use fu.returns

    ic_by_lag = zeros(max_lag)
    for lag in 1:max_lag
        ic_vals = Float64[]
        for t in 1:T-lag
            f = F[t, :]
            # lag-period cumulative return
            r_fwd = if t+lag <= T; fu.returns[t+lag-1, :]; else fill(NaN, N); end
            valid = .!isnan.(f) .& .!isnan.(r_fwd)
            sum(valid) < 5 && continue
            push!(ic_vals, cor(ordinalrank(f[valid]), ordinalrank(r_fwd[valid])))
        end
        ic_by_lag[lag] = isempty(ic_vals) ? NaN : mean(ic_vals)
    end
    return ic_by_lag
end

"""
    factor_turnover(fu, factor_name, window=5) → Float64

Mean cross-sectional rank correlation between factor at t and t-window.
High turnover → low correlation → costly to trade.
"""
function factor_turnover(fu::FactorUniverse, factor_name::String,
                           window::Int=5)::Float64
    F = fu.factors[factor_name]
    T = fu.n_bars
    corr_vals = Float64[]
    for t in window+1:T
        f_now  = F[t, :]; f_lag = F[t-window, :]
        valid  = .!isnan.(f_now) .& .!isnan.(f_lag)
        sum(valid) < 5 && continue
        push!(corr_vals, cor(ordinalrank(f_now[valid]), ordinalrank(f_lag[valid])))
    end
    return isempty(corr_vals) ? NaN : mean(corr_vals)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-Sectional Regression
# ─────────────────────────────────────────────────────────────────────────────

"""
    cross_sectional_regression(returns_t, factor_exposures_t) → NamedTuple

Fama-MacBeth style cross-sectional regression at a single point in time.
Returns factor realised returns (γ), t-stats, and residuals.
"""
function cross_sectional_regression(returns_t::Vector{Float64},
                                     factor_exposures_t::Matrix{Float64})::NamedTuple

    N, K = size(factor_exposures_t)   # N assets, K factors
    @assert length(returns_t) == N

    valid = .!isnan.(returns_t) .& all(.!isnan.(factor_exposures_t), dims=2)[:]
    N_v   = sum(valid)
    N_v < K + 2 && return (gammas=fill(NaN, K), t_stats=fill(NaN, K),
                             r2=NaN, residuals=fill(NaN, N))

    R_v = returns_t[valid]
    F_v = factor_exposures_t[valid, :]

    # Add intercept
    X  = hcat(ones(N_v), F_v)
    XtX = X' * X + 1e-8 * I(K+1)
    beta = XtX \ (X' * R_v)

    fitted   = X * beta
    resid_v  = R_v - fitted
    sigma2   = var(resid_v)
    se       = sqrt.(sigma2 .* diag(inv(XtX)))

    t_stats = beta ./ max.(se, 1e-12)
    r2      = 1 - var(resid_v) / max(var(R_v), 1e-12)

    # Reconstruct full residuals
    resid_full = fill(NaN, N)
    r_idx = findall(valid)
    for (k, i) in enumerate(r_idx)
        resid_full[i] = resid_v[k]
    end

    return (gammas = beta[2:end], t_stats = t_stats[2:end],
            intercept = beta[1], r2 = r2, residuals = resid_full)
end

"""
    fama_macbeth(fu; min_ic=0.0) → DataFrame

Full Fama-MacBeth regression over all time periods.
Returns time series of factor realised returns + summary statistics.
"""
function fama_macbeth(fu::FactorUniverse; min_ic::Float64=0.0)::DataFrame
    T = fu.n_bars
    K = length(fu.factor_names)

    gamma_series = fill(NaN, T, K)

    for t in 1:T-1
        f_exposures = hcat([fu.factors[fname][t, :] for fname in fu.factor_names]...)
        rets_t      = fu.returns[t, :]

        valid_rows = .!any(isnan.(f_exposures), dims=2)[:] .& .!isnan.(rets_t)
        sum(valid_rows) < K + 2 && continue

        reg = cross_sectional_regression(rets_t, f_exposures)
        gamma_series[t, :] = reg.gammas
    end

    # Build summary
    rows = NamedTuple[]
    for (k, fname) in enumerate(fu.factor_names)
        gs      = filter(!isnan, gamma_series[:, k])
        isempty(gs) && continue
        m       = mean(gs); s = std(gs)
        t_stat  = length(gs) > 1 ? m / (s / sqrt(length(gs))) : 0.0
        p_val   = length(gs) > 1 ? 2 * (1 - cdf(TDist(length(gs)-1), abs(t_stat))) : 1.0
        push!(rows, (
            factor      = fname,
            mean_gamma  = m,
            std_gamma   = s,
            t_stat      = t_stat,
            p_value     = p_val,
            significant = p_val < 0.05,
            n_obs       = length(gs),
        ))
    end
    return isempty(rows) ? DataFrame() : DataFrame(rows)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Factor Portfolio Construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    factor_portfolio(factor_exposures, returns; method=:long_short) → NamedTuple

Construct a factor-mimicking portfolio.

Methods:
  :long_short  — long top quintile, short bottom quintile
  :signal      — weight proportional to exposure z-score
  :rank_weight — weight proportional to rank
"""
function factor_portfolio(factor_exposures::Vector{Float64},
                           returns::Vector{Float64};
                           method::Symbol=:long_short,
                           n_quintile::Int=5)::NamedTuple

    N = length(factor_exposures)
    valid = .!isnan.(factor_exposures) .& .!isnan.(returns)
    N_v   = sum(valid)
    N_v < n_quintile * 2 && return (weights=zeros(N), portfolio_return=NaN,
                                     long_ret=NaN, short_ret=NaN)

    f_v = factor_exposures[valid]
    r_v = returns[valid]
    sorted_idx = sortperm(f_v)

    q_size = N_v ÷ n_quintile

    if method == :long_short
        top_idx    = sorted_idx[end-q_size+1:end]
        bottom_idx = sorted_idx[1:q_size]

        long_ret  = mean(r_v[top_idx])
        short_ret = mean(r_v[bottom_idx])
        port_ret  = long_ret - short_ret

        # Weights in full space
        w = zeros(N)
        valid_idx = findall(valid)
        for i in top_idx;    w[valid_idx[i]] = 1.0 / q_size;  end
        for i in bottom_idx; w[valid_idx[i]] = -1.0 / q_size; end

    elseif method == :signal
        # Normalised signal weights
        f_norm = f_v ./ max(std(f_v), 1e-10)
        port_ret = dot(f_norm, r_v) / max(sum(abs.(f_norm)), 1e-10)
        long_ret = mean(r_v[f_v .> 0])
        short_ret = mean(r_v[f_v .<= 0])

        w = zeros(N)
        valid_idx = findall(valid)
        for (k, i) in enumerate(valid_idx); w[i] = f_norm[k]; end
        s = sum(abs.(w)); s > 1e-10 && (w ./= s)

    else  # :rank_weight
        ranks  = ordinalrank(f_v)
        ranks_c = ranks .- mean(ranks)   # demean
        port_ret = dot(ranks_c, r_v) / max(sum(abs.(ranks_c)), 1e-10)
        long_ret = mean(r_v[ranks .> median(ranks)])
        short_ret = mean(r_v[ranks .<= median(ranks)])

        w = zeros(N)
        valid_idx = findall(valid)
        for (k, i) in enumerate(valid_idx); w[i] = ranks_c[k]; end
        s = sum(abs.(w)); s > 1e-10 && (w ./= s)
    end

    return (weights=w, portfolio_return=port_ret,
            long_ret=long_ret, short_ret=short_ret)
end

"""
    quintile_analysis(fu, factor_name; n_quintile=5) → DataFrame

Compute per-quintile mean return and hit rate over all time periods.
"""
function quintile_analysis(fu::FactorUniverse, factor_name::String;
                             n_quintile::Int=5)::DataFrame

    @assert haskey(fu.factors, factor_name)
    F = fu.factors[factor_name]
    R = fu.returns
    T = fu.n_bars

    quintile_rets = zeros(n_quintile, T)
    q_counts      = zeros(Int, n_quintile, T)

    for t in 1:T-1
        f = F[t, :]; r = R[t, :]
        valid = .!isnan.(f) .& .!isnan.(r)
        sum(valid) < n_quintile && continue

        f_v = f[valid]; r_v = r[valid]
        n_v = length(f_v)

        qs = [quantile(f_v, (k-1)/n_quintile) for k in 1:n_quintile+1]
        for q in 1:n_quintile
            mask = f_v .>= qs[q] .& f_v .< qs[q+1]
            q == n_quintile && (mask .|= f_v .>= qs[end])
            if sum(mask) > 0
                quintile_rets[q, t] = mean(r_v[mask])
                q_counts[q, t] = sum(mask)
            end
        end
    end

    rows = NamedTuple[]
    for q in 1:n_quintile
        valid_t = q_counts[q, :] .> 0
        qr      = quintile_rets[q, valid_t]
        push!(rows, (
            quintile   = q,
            mean_ret   = mean(qr),
            std_ret    = std(qr),
            hit_rate   = mean(qr .> 0),
            sharpe     = std(qr) > 1e-10 ? mean(qr) / std(qr) * sqrt(252) : 0.0,
            n_obs      = sum(valid_t),
        ))
    end
    df = DataFrame(rows)

    # Monotonicity score: Spearman of quintile vs mean_ret
    if nrow(df) > 2
        mono_score = cor(Float64.(df.quintile), Float64.(df.mean_ret))
    else
        mono_score = NaN
    end

    @info "Factor: $factor_name, Monotonicity: $(round(mono_score, digits=3))"
    return df
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Barra-style Risk Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    barra_risk_model(returns_matrix, factor_exposures) → NamedTuple

Simplified Barra-style fundamental factor risk model.

returns_matrix   : T × N
factor_exposures : N × K (fixed factor loadings)

Returns:
  F   — factor covariance matrix (K × K)
  D   — diagonal idiosyncratic variances (N-vector)
  B   — factor exposure matrix (N × K)
  Sigma — full covariance: B F B' + diag(D)
"""
function barra_risk_model(returns_matrix::Matrix{Float64},
                           factor_exposures::Matrix{Float64})::NamedTuple

    T, N = size(returns_matrix)
    _, K = size(factor_exposures)
    B = factor_exposures   # N × K

    # OLS: R_t = B * f_t + ε_t  →  f_t = (B'B)⁻¹ B' R_t
    BtB     = B' * B + 1e-8 * I(K)
    BtB_inv = inv(BtB)

    # Factor realised returns
    F_realised = zeros(T, K)
    residuals  = zeros(T, N)
    for t in 1:T
        r_t = returns_matrix[t, :]
        valid = .!isnan.(r_t)
        if sum(valid) < K
            continue
        end
        B_v = B[valid, :]; r_v = r_t[valid]
        f_t = (B_v' * B_v + 1e-8 * I(K)) \ (B_v' * r_v)
        F_realised[t, :] = f_t
        residuals[t, valid] = r_v - B_v * f_t
    end

    # Factor covariance
    F_cov = cov(F_realised) + 1e-8 * I(K)

    # Idiosyncratic variances
    D = vec(var(residuals, dims=1))
    D = max.(D, 1e-10)

    # Full covariance
    Sigma_full = B * F_cov * B' + Diagonal(D)

    return (
        factor_returns    = F_realised,
        factor_cov        = F_cov,
        idio_vars         = D,
        loadings          = B,
        total_cov         = Sigma_full,
        factor_names      = ["Factor_$k" for k in 1:K],
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. PCA Factors
# ─────────────────────────────────────────────────────────────────────────────

"""
    pca_factors(returns_matrix; n_components=5) → NamedTuple

Extract PCA factors from returns. Returns eigenvectors, scores, variance explained.
"""
function pca_factors(returns_matrix::Matrix{Float64};
                      n_components::Int=5)::NamedTuple

    T, N = size(returns_matrix)

    # Demean
    mu  = vec(mean(returns_matrix, dims=1))
    R_c = returns_matrix .- mu'

    # SVD (more numerically stable than eigen on cross-product)
    U, S, Vt = svd(R_c)

    n_comp = min(n_components, size(Vt, 1))

    # Factor loadings: N × n_comp (Vt' = V)
    loadings = Vt[1:n_comp, :]'   # N × n_comp

    # Factor scores: T × n_comp
    scores = U[:, 1:n_comp] .* S[1:n_comp]'

    # Variance explained
    total_var = sum(S.^2)
    var_expl  = S[1:n_comp].^2 ./ total_var

    return (
        loadings        = loadings,
        scores          = scores,
        singular_values = S[1:n_comp],
        variance_explained = var_expl,
        cumulative_var  = cumsum(var_expl),
        n_components    = n_comp,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Elastic Net Factor Selection
# ─────────────────────────────────────────────────────────────────────────────

"""
    elastic_net_factor_selection(X, y; alpha=0.5, lambda=0.001, max_iter=1000)
        → NamedTuple

Elastic Net regression for factor selection (combines L1 + L2 penalties).

α = 1: Lasso; α = 0: Ridge.
Returns selected factors (nonzero coefficients).
"""
function elastic_net_factor_selection(X::Matrix{Float64}, y::Vector{Float64};
                                        alpha::Float64=0.5,
                                        lambda::Float64=0.001,
                                        max_iter::Int=1000)::NamedTuple

    n, p = size(X)
    beta  = zeros(p)
    beta0 = mean(y)

    # Coordinate descent
    r = y .- beta0 .- X * beta

    for iter in 1:max_iter
        beta_old = copy(beta)

        for j in 1:p
            rj  = r .+ X[:, j] .* beta[j]
            zj  = dot(X[:, j], rj) / n
            xj2 = dot(X[:, j], X[:, j]) / n + lambda * (1 - alpha)

            # Soft threshold
            if zj > lambda * alpha
                beta[j] = (zj - lambda * alpha) / xj2
            elseif zj < -lambda * alpha
                beta[j] = (zj + lambda * alpha) / xj2
            else
                beta[j] = 0.0
            end

            r = rj .- X[:, j] .* beta[j]
        end

        if norm(beta - beta_old) < 1e-8
            break
        end
    end

    selected    = findall(abs.(beta) .> 1e-8)
    fitted_vals = beta0 .+ X * beta
    residuals   = y - fitted_vals
    r2          = 1 - var(residuals) / max(var(y), 1e-12)

    return (
        coefficients = beta,
        intercept    = beta0,
        selected     = selected,
        n_selected   = length(selected),
        r2           = r2,
        residuals    = residuals,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Alpha Model
# ─────────────────────────────────────────────────────────────────────────────

"""
    alpha_model(fu; combination=:icir_weighted) → Matrix{Float64}

Combine factor signals into a composite alpha (predicted returns).

Methods:
  :equal         — equal weight factors
  :icir_weighted — weight by rolling ICIR
  :regression    — OLS combination
"""
function alpha_model(fu::FactorUniverse;
                     combination::Symbol=:icir_weighted,
                     window::Int=63)::Matrix{Float64}

    T = fu.n_bars; N = fu.n_assets; K = length(fu.factor_names)
    alpha_mat = fill(NaN, T, N)

    for t in window+1:T-1
        # Compute rolling ICIR for each factor
        weights = zeros(K)
        for (k, fname) in enumerate(fu.factor_names)
            ic_series = Float64[]
            for s in (t - window + 1):(t-1)
                f = fu.factors[fname][s, :]; r = fu.returns[s, :]
                valid = .!isnan.(f) .& .!isnan.(r)
                sum(valid) < 5 && continue
                push!(ic_series, cor(ordinalrank(f[valid]), ordinalrank(r[valid])))
            end
            if length(ic_series) >= 5
                m = mean(ic_series); s = std(ic_series)
                if combination == :icir_weighted
                    weights[k] = s > 1e-10 ? abs(m) / s * sign(m) : 0.0
                elseif combination == :equal
                    weights[k] = 1.0
                end
            end
        end

        # Normalise weights
        sw = sum(abs.(weights)); sw < 1e-10 && (weights = ones(K))
        weights ./= sw

        # Combine factor signals
        alpha_t = zeros(N)
        for (k, fname) in enumerate(fu.factor_names)
            f = fu.factors[fname][t, :]
            valid = .!isnan.(f)
            alpha_t[valid] .+= weights[k] .* f[valid]
        end
        alpha_mat[t, :] = standardise_factor(alpha_t)
    end

    return alpha_mat
end

"""
    predict_returns(alpha_mat, t) → Vector{Float64}

Return the alpha (predicted return) vector at time t.
"""
function predict_returns(alpha_mat::Matrix{Float64}, t::Int)::Vector{Float64}
    return alpha_mat[t, :]
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. Factor Attribution
# ─────────────────────────────────────────────────────────────────────────────

"""
    factor_attribution(portfolio_weights, factor_exposures,
                        factor_returns_series) → NamedTuple

Brinson-Hood-Beebower style attribution decomposed by factor.

Returns per-factor PnL contribution and R² of factor model for the portfolio.
"""
function factor_attribution(portfolio_weights::Vector{Float64},
                              factor_exposures::Matrix{Float64},
                              factor_returns_series::Matrix{Float64})::NamedTuple

    T, K = size(factor_returns_series)
    N    = length(portfolio_weights)
    @assert size(factor_exposures) == (N, K)

    # Portfolio factor betas: N × K → scalar
    port_betas = vec(portfolio_weights' * factor_exposures)   # K-vector

    # Factor PnL
    factor_pnl = factor_returns_series * port_betas   # T-vector

    # Total portfolio return (approximate: using weights × factor model)
    total_pnl  = [dot(portfolio_weights,
                      factor_exposures * factor_returns_series[t, :]) for t in 1:T]

    # Idiosyncratic: total - systematic
    idio_pnl   = total_pnl .- factor_pnl

    # Attribution per factor
    per_factor  = Dict{String, Vector{Float64}}()
    for k in 1:K
        per_factor["Factor_$k"] = factor_returns_series[:, k] .* port_betas[k]
    end

    return (
        total_pnl        = total_pnl,
        factor_pnl       = factor_pnl,
        idiosyncratic_pnl= idio_pnl,
        per_factor       = per_factor,
        port_betas       = port_betas,
        systematic_frac  = var(factor_pnl) / max(var(total_pnl), 1e-12),
    )
end

end # module FactorModel
