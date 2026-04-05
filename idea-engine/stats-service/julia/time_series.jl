# =============================================================================
# time_series.jl — Advanced Time-Series Analysis
# =============================================================================
# Provides:
#   - HurstExponent     (R/S analysis for long memory)
#   - FractionalDifferencing  (achieve stationarity while preserving memory)
#   - ARIMA_fit         (auto ARIMA order selection via information criteria)
#   - WaveletDecomposition   (multi-scale trend/noise separation)
#   - InformationCriteria    (AIC / BIC / HQIC)
#   - CointegratedPairs      (Engle-Granger test for pairs trading)
#   - SpectralDensity        (periodogram for cyclical detection)
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, FFTW, JSON3
# =============================================================================

module TimeSeries

using Statistics
using LinearAlgebra
using JSON3

# Optional FFTW for spectral density
const HAS_FFTW = try
    using FFTW; true
catch; false end

export HurstExponent, FractionalDifferencing, ARIMA_fit
export WaveletDecomposition, InformationCriteria, CointegratedPairs, SpectralDensity

# ---------------------------------------------------------------------------
# Hurst Exponent (R/S analysis)
# ---------------------------------------------------------------------------

"""
Compute the Hurst exponent of a time series using Rescaled Range (R/S) analysis.

The Hurst exponent H characterises long-range memory:
  H ≈ 0.5  → random walk (no memory)
  H > 0.5  → persistent (trending)
  H < 0.5  → mean-reverting (anti-persistent)

# Returns
NamedTuple: (H, rs_pairs, log_n, log_rs, intercept)
"""
function HurstExponent(ts::AbstractVector{<:Real};
                        min_chunk::Int=10,
                        n_chunks::Int=20)
    x = Float64.(ts)
    n = length(x)
    n < 2 * min_chunk && error("Series too short for R/S analysis (need ≥ $(2*min_chunk))")

    # Logarithmically spaced chunk sizes
    max_chunk = n ÷ 2
    chunk_sizes = unique(round.(Int,
        exp.(range(log(min_chunk), log(max_chunk); length=n_chunks))
    ))
    filter!(s -> s >= min_chunk && s <= max_chunk, chunk_sizes)

    log_n  = Float64[]
    log_rs = Float64[]

    for chunk_size in chunk_sizes
        n_sub = n ÷ chunk_size
        n_sub < 2 && continue

        rs_vals = Float64[]
        for k in 1:n_sub
            sub = x[(k-1)*chunk_size + 1 : k*chunk_size]

            # Mean-adjust
            sub_adj = sub .- mean(sub)

            # Cumulative sum
            cum_dev = cumsum(sub_adj)

            # Range
            R = maximum(cum_dev) - minimum(cum_dev)

            # Standard deviation
            S = std(sub; corrected=true)
            S < 1e-12 && continue

            push!(rs_vals, R / S)
        end

        isempty(rs_vals) && continue
        push!(log_n,  log(chunk_size))
        push!(log_rs, log(mean(rs_vals)))
    end

    length(log_n) < 2 && error("Not enough valid chunk sizes for OLS fit")

    # OLS: log(R/S) = H * log(n) + c
    X   = hcat(ones(length(log_n)), log_n)
    β   = (X'X) \ (X'log_rs)

    (
        H         = β[2],
        intercept = β[1],
        rs_pairs  = collect(zip(log_n, log_rs)),
        n_chunks  = length(log_n),
        interpretation = _hurst_label(β[2])
    )
end

function _hurst_label(H)
    H > 0.55 && return "PERSISTENT (trending)"
    H < 0.45 && return "ANTI-PERSISTENT (mean-reverting)"
    "RANDOM WALK"
end

# ---------------------------------------------------------------------------
# Fractional Differencing
# ---------------------------------------------------------------------------

"""
Apply fractional differencing of order d to a time series.

Uses the fixed-window FFD (Fracdiff) approach:
  Δᵈ xₜ = Σₖ₌₀^∞ ωₖ xₜ₋ₖ
  ωₖ = Πⱼ₌₀^{k-1} (j - d) / (j + 1)

# Arguments
- `ts`      : input price/equity series
- `d`       : differencing order ∈ (0, 1) for fractional; 1 = standard diff
- `thresh`  : weight threshold to truncate the infinite series (default 1e-5)

# Returns
NamedTuple: (differenced, weights, d, stationarity_note)
"""
function FractionalDifferencing(ts::AbstractVector{<:Real}, d::Float64;
                                  thresh::Float64=1e-5)
    0.0 <= d <= 2.0 || error("d must be in [0, 2]")

    x = Float64.(ts)
    n = length(x)

    # Compute weights until they drop below threshold
    weights = Float64[1.0]
    k = 1
    while true
        w = weights[end] * (k - 1 - d) / k
        abs(w) < thresh && break
        push!(weights, w)
        k += 1
        k > n && break
    end

    L     = length(weights)
    n_out = n - L + 1
    n_out < 1 && error("Series too short for fractional differencing with d=$d")

    out = zeros(Float64, n_out)
    for t in 1:n_out
        # Dot product with reversed window
        s = 0.0
        for j in 1:L
            s += weights[j] * x[t + L - j]
        end
        out[t] = s
    end

    # ADF-style stationarity check via variance ratio
    var_original = var(x)
    var_diff     = var(out)
    note = var_diff < var_original ?
        "Variance reduced (consistent with stationarity)" :
        "Variance increased — check d value"

    (
        differenced       = out,
        weights           = weights,
        d                 = d,
        n_weights         = L,
        stationarity_note = note
    )
end

"""
Find the minimum d ∈ [0.0, 1.0] that achieves approximate stationarity
(variance stabilisation relative to a first-differenced benchmark).
"""
function find_stationary_d(ts::AbstractVector{<:Real};
                             d_grid=0.0:0.05:1.0,
                             thresh=1e-5)
    x      = Float64.(ts)
    var_fd = var(diff(x))   # variance of standard (d=1) differenced series

    for d in d_grid
        d == 0.0 && continue
        result = try FractionalDifferencing(x, d; thresh=thresh) catch; nothing end
        isnothing(result) && continue
        # Accept d where variance is close to that of the fully-differenced series
        if result.differenced |> var |> v -> v <= var_fd * 1.1
            return d
        end
    end
    1.0   # fall back to full differencing
end

# ---------------------------------------------------------------------------
# ARIMA fitting (auto order selection)
# ---------------------------------------------------------------------------

"""
AR(p) coefficient estimation via Yule-Walker equations.
"""
function _yule_walker(x::Vector{Float64}, p::Int)
    n    = length(x)
    x_c  = x .- mean(x)
    # Autocorrelations
    γ    = [sum(x_c[1:n-k] .* x_c[1+k:n]) / n for k in 0:p]
    R    = [γ[abs(i-j)+1] for i in 1:p, j in 1:p]
    r    = γ[2:p+1]
    φ    = R \ r
    σ²   = γ[1] - dot(r, φ)
    φ, max(σ², 1e-12)
end

"""
MA residuals via innovations algorithm (approximate).
"""
function _ma_residuals(x::Vector{Float64}, φ::Vector{Float64})
    n      = length(x)
    p      = length(φ)
    x_c    = x .- mean(x)
    errors = zeros(Float64, n)
    x_hat  = zeros(Float64, n)

    for t in (p+1):n
        x_hat[t] = dot(φ, x_c[t-1:-1:t-p])
        errors[t] = x_c[t] - x_hat[t]
    end
    errors
end

"""
Log-likelihood of AR(p) model (Gaussian innovations).
"""
function _ar_loglik(x::Vector{Float64}, p::Int)
    p == 0 && return -0.5 * length(x) * log(var(x))
    n = length(x)
    φ, σ² = _yule_walker(x, p)
    resid = _ma_residuals(x, φ)
    n_eff = n - p
    -0.5 * n_eff * (log(2π * σ²) + 1.0)
end

"""
Auto ARIMA: selects optimal (p, d, q) by minimising BIC.

Searches p ∈ 0:max_p, d ∈ {0,1,2}, q ∈ 0:max_q.
Returns the best model specification and diagnostics.

# Returns
NamedTuple: (p, d, q, aic, bic, ar_coeffs, residuals, fitted)
"""
function ARIMA_fit(ts::AbstractVector{<:Real};
                    max_p::Int=5, max_q::Int=5,
                    max_d::Int=2, criterion::Symbol=:bic)
    x      = Float64.(ts)
    n      = length(x)
    n < 20 && error("Series too short for ARIMA (need ≥ 20 points)")

    best_ic  = Inf
    best_spec = (p=0, d=0, q=0)
    best_φ   = Float64[]
    best_resid = Float64[]

    for d_ord in 0:max_d
        xs = d_ord == 0 ? x :
             d_ord == 1 ? diff(x) :
             diff(diff(x))

        length(xs) < max_p + 5 && continue
        xs_c = xs .- mean(xs)

        for p_ord in 0:max_p
            φ = p_ord > 0 ? _yule_walker(xs, p_ord)[1] : Float64[]
            resid = p_ord > 0 ? _ma_residuals(xs, φ) : xs_c

            # Simple MA(q) via residuals autocorrelation (approx)
            for q_ord in 0:min(max_q, p_ord + 2)
                n_params = p_ord + q_ord + 1   # +1 for variance
                n_eff    = length(resid) - p_ord

                σ²  = var(resid)
                ll  = -0.5 * n_eff * (log(2π * σ²) + 1.0)
                aic = -2ll + 2 * n_params
                bic = -2ll + n_params * log(n_eff)
                hqic= -2ll + 2 * n_params * log(log(n_eff))

                ic  = criterion == :aic ? aic :
                      criterion == :hqic ? hqic : bic

                if ic < best_ic
                    best_ic   = ic
                    best_spec = (p=p_ord, d=d_ord, q=q_ord)
                    best_φ    = φ
                    best_resid = resid
                end
            end
        end
    end

    n_eff = length(best_resid) - best_spec.p
    σ²    = var(best_resid)
    ll    = -0.5 * n_eff * (log(2π * σ²) + 1.0)
    k     = best_spec.p + best_spec.q + 1

    (
        p           = best_spec.p,
        d           = best_spec.d,
        q           = best_spec.q,
        aic         = -2ll + 2k,
        bic         = -2ll + k * log(max(n_eff, 1)),
        ar_coeffs   = best_φ,
        residuals   = best_resid,
        sigma2      = σ²,
        criterion   = criterion
    )
end

# ---------------------------------------------------------------------------
# Wavelet Decomposition (Haar, db4 via lifting scheme)
# ---------------------------------------------------------------------------

"""
Haar wavelet decomposition: iterative dyadic filter bank.
Returns approximation coefficients and detail coefficients per level.
"""
function _haar_decompose(x::Vector{Float64}, levels::Int)
    n       = length(x)
    approx  = copy(x)
    details = Vector{Vector{Float64}}()

    for lvl in 1:levels
        L = length(approx)
        L_out = L ÷ 2
        L_out < 1 && break

        a = [(approx[2i-1] + approx[2i]) / √2 for i in 1:L_out]
        d = [(approx[2i-1] - approx[2i]) / √2 for i in 1:L_out]

        pushfirst!(details, d)
        approx = a
    end
    approx, details
end

"""
Haar wavelet reconstruction from approximation + detail coefficients.
"""
function _haar_reconstruct(approx::Vector{Float64}, details::Vector{Vector{Float64}})
    a = copy(approx)
    for d in details
        L = length(d)
        a_new = zeros(Float64, 2L)
        for i in 1:L
            a_new[2i-1] = (a[i] + d[i]) / √2
            a_new[2i]   = (a[i] - d[i]) / √2
        end
        a = a_new
    end
    a
end

"""
Multi-scale wavelet decomposition of a time series.

Separates signal into trend (low-frequency approximation) and noise components
at each scale level.  Uses Haar wavelets by default.

# Returns
NamedTuple: (trend, details, reconstructed, scales, energy_by_level)
"""
function WaveletDecomposition(ts::AbstractVector{<:Real}; levels::Int=5)
    x = Float64.(ts)
    n = length(x)
    n < 2^levels && @warn "Series length $n may be too short for $levels levels"

    # Pad to next power of 2 if needed
    target_len = nextpow(2, n)
    x_padded   = vcat(x, fill(x[end], target_len - n))

    approx, details = _haar_decompose(x_padded, levels)

    # Reconstruct trend (zero out detail coefficients)
    trend_full = _haar_reconstruct(approx, [zeros(length(d)) for d in details])
    trend      = trend_full[1:n]

    # Energy at each detail level
    energy_by_level = [sum(d.^2) for d in details]
    total_energy    = sum(energy_by_level) + sum(approx.^2)
    energy_frac     = energy_by_level ./ total_energy

    # Residual noise = original - trend
    noise = x .- trend

    (
        trend           = trend,
        noise           = noise,
        approx          = approx,
        details         = [d[1:min(length(d), n)] for d in details],
        n_levels        = length(details),
        energy_by_level = energy_by_level,
        energy_fraction = energy_frac,
        snr             = var(trend) / max(var(noise), 1e-12)
    )
end

# ---------------------------------------------------------------------------
# Information Criteria
# ---------------------------------------------------------------------------

"""
Compute AIC, BIC, and HQIC for a fitted model.

# Arguments
- `ts`        : original time series
- `residuals` : model residuals
- `n_params`  : number of estimated parameters

# Returns
NamedTuple: (aic, bic, hqic, log_likelihood)
"""
function InformationCriteria(ts::AbstractVector{<:Real},
                               residuals::AbstractVector{<:Real},
                               n_params::Int)
    n    = length(residuals)
    σ²   = var(residuals; corrected=false)
    σ²   = max(σ², 1e-12)
    ll   = -0.5 * n * (log(2π * σ²) + 1.0)

    aic  = -2ll + 2 * n_params
    bic  = -2ll + n_params * log(n)
    hqic = -2ll + 2 * n_params * log(log(max(n, 3)))

    (
        aic            = aic,
        bic            = bic,
        hqic           = hqic,
        log_likelihood = ll,
        n_obs          = n,
        n_params       = n_params,
        sigma2         = σ²
    )
end

# ---------------------------------------------------------------------------
# Cointegrated Pairs (Engle-Granger)
# ---------------------------------------------------------------------------

"""
Engle-Granger two-step cointegration test.

Step 1: OLS regression y = α + β x + ε
Step 2: ADF test on residuals ε

A significant ADF test indicates the pair is cointegrated (stationary spread).

# Returns
NamedTuple: (cointegrated, beta, alpha, spread, adf_stat, adf_critical_5pct,
             half_life_days)
"""
function _engle_granger_pair(y::Vector{Float64}, x::Vector{Float64})
    n    = length(y)
    n == length(x) || error("y and x must have the same length")

    # OLS: y = α + β x
    X    = hcat(ones(n), x)
    β_ols = (X'X) \ (X'y)
    α, β = β_ols[1], β_ols[2]
    ε    = y .- α .- β .* x

    # ADF test on residuals (Dickey-Fuller with no trend)
    # H₀: unit root (non-stationary)  H₁: stationary
    Δε   = diff(ε)
    n2   = length(Δε)
    ε_lag = ε[1:n2]
    ols_y = Δε
    ols_X = hcat(ones(n2), ε_lag)
    β2    = (ols_X'ols_X) \ (ols_X'ols_y)
    res2  = ols_y .- ols_X * β2
    σ²2   = sum(res2.^2) / (n2 - 2)
    se_δ  = sqrt(σ²2 / sum((ε_lag .- mean(ε_lag)).^2))
    δ     = β2[2]    # coefficient on lagged residual
    t_stat = δ / se_δ

    # MacKinnon (1991) approximate critical values for EG test (n→∞)
    crit_1pct  = -3.90
    crit_5pct  = -3.34
    crit_10pct = -3.05

    cointegrated = t_stat < crit_5pct

    # Half-life: mean reversion speed from AR(1) on spread
    # ε_t = φ ε_{t-1} + noise  →  half_life = -log(2) / log(|φ|)
    φ_est   = δ + 1.0   # since δ = φ - 1
    half_life = if abs(φ_est) < 1.0 && φ_est > 0.0
        -log(2) / log(φ_est)
    else
        Inf
    end

    (
        cointegrated    = cointegrated,
        beta            = β,
        alpha           = α,
        spread          = ε,
        adf_stat        = t_stat,
        adf_critical_5pct = crit_5pct,
        half_life_days  = half_life,
        mean_spread     = mean(ε),
        std_spread      = std(ε)
    )
end

"""
Find all cointegrated pairs in a price matrix.

# Arguments
- `price_matrix` : n × m matrix, each column is a price series for one asset
- `asset_names`  : optional vector of asset name strings

# Returns
Vector of NamedTuples for pairs passing the cointegration test, sorted by |ADF|
"""
function CointegratedPairs(price_matrix::AbstractMatrix{<:Real};
                             asset_names::Vector{String}=String[])
    n, m = size(price_matrix)
    n < 30 && error("Need at least 30 observations for cointegration test")

    names = isempty(asset_names) ? ["Asset_$i" for i in 1:m] : asset_names
    length(names) == m || error("asset_names length must equal number of columns")

    X = Float64.(price_matrix)
    results = NamedTuple[]

    for i in 1:m, j in (i+1):m
        y = X[:, i]
        x = X[:, j]

        # Check both directions, keep lower ADF stat
        r1 = _engle_granger_pair(y, x)
        r2 = _engle_granger_pair(x, y)
        r  = abs(r1.adf_stat) >= abs(r2.adf_stat) ? r1 : r2

        push!(results, merge(r, (
            asset_a = names[i],
            asset_b = names[j]
        )))
    end

    # Sort by ADF statistic (most negative = most strongly cointegrated)
    sort!(results, by = r -> r.adf_stat)

    filter(r -> r.cointegrated, results)
end

# ---------------------------------------------------------------------------
# Spectral Density (Periodogram)
# ---------------------------------------------------------------------------

"""
Compute the periodogram (raw spectral density estimate) of a time series.

Uses FFT if FFTW is available; falls back to DFT otherwise.

# Returns
NamedTuple: (frequencies, power, dominant_cycles_days, nyquist_freq)
"""
function SpectralDensity(ts::AbstractVector{<:Real};
                          fs::Float64=1.0,       # sampling frequency (default 1/day)
                          smooth_window::Int=3)
    x = Float64.(ts) .- mean(ts)   # remove mean
    n = length(x)
    n < 10 && error("Series too short for spectral analysis")

    # Apply Hann window to reduce spectral leakage
    hann = [0.5 * (1 - cos(2π * (i-1) / (n-1))) for i in 1:n]
    x_win = x .* hann

    # FFT
    n_fft   = nextpow(2, n)
    x_pad   = vcat(x_win, zeros(n_fft - n))

    if HAS_FFTW
        X_fft = FFTW.rfft(x_pad)
    else
        # Manual DFT (slow but correct for moderate n)
        m      = n_fft ÷ 2 + 1
        X_fft  = [sum(x_pad[t] * exp(-2π*im*(t-1)*(k-1)/n_fft)
                       for t in 1:n_fft)
                  for k in 1:m]
    end

    # Power spectral density (one-sided)
    power_raw = abs2.(X_fft)
    power_raw[2:end-1] .*= 2   # double-sided correction

    m_out     = length(power_raw)
    freqs     = [fs * (k-1) / n_fft for k in 1:m_out]

    # Smooth with moving average
    power = copy(power_raw)
    if smooth_window > 1
        hw = smooth_window ÷ 2
        for i in (hw+1):(m_out - hw)
            power[i] = mean(power_raw[i-hw:i+hw])
        end
    end

    # Identify dominant cycles (peaks in power spectrum; exclude DC component)
    nyquist   = fs / 2.0
    nz_idx    = findall(freqs .> 0)
    peak_idxs = Int[]
    for i in 2:(length(nz_idx) - 1)
        ii = nz_idx[i]
        if power[ii] > power[ii-1] && power[ii] > power[ii+1]
            push!(peak_idxs, ii)
        end
    end

    # Top 5 peaks by power
    sort!(peak_idxs, by = i -> -power[i])
    top_peaks   = peak_idxs[1:min(5, length(peak_idxs))]
    dom_freqs   = freqs[top_peaks]
    dom_periods = [f > 0 ? 1.0/f : Inf for f in dom_freqs]   # in days if fs=1

    (
        frequencies          = freqs,
        power                = power,
        dominant_cycles_days = dom_periods,
        dominant_frequencies = dom_freqs,
        nyquist_freq         = nyquist,
        n_samples            = n
    )
end

# ---------------------------------------------------------------------------
# JSON output helper
# ---------------------------------------------------------------------------

function to_json(x)::String
    JSON3.write(x)
end

end  # module TimeSeries

# ---------------------------------------------------------------------------
# CLI entry point (self-test)
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using .TimeSeries
    using Statistics

    println("[time_series] Running self-tests...")

    rng = Random.MersenneTwister(42)
    n   = 512

    # Synthetic GBM price series
    log_rets  = 0.0002 .+ 0.012 .* randn(rng, n)
    prices    = exp.(cumsum(log_rets)) .* 100.0

    # 1. Hurst exponent
    h = HurstExponent(log_rets)
    println("  Hurst H = $(round(h.H; digits=4)) — $(h.interpretation)")

    # 2. Fractional differencing
    fd = FractionalDifferencing(prices, 0.4)
    println("  FracDiff d=0.4: $(length(fd.differenced)) output points, $(fd.n_weights) weights")

    # 3. ARIMA
    arima = ARIMA_fit(log_rets)
    println("  ARIMA($(arima.p),$(arima.d),$(arima.q))  BIC=$(round(arima.bic; digits=2))")

    # 4. Wavelet
    wt = WaveletDecomposition(prices; levels=4)
    println("  Wavelet: SNR=$(round(wt.snr; digits=2)), energy fractions=$(round.(wt.energy_fraction; digits=3))")

    # 5. Spectral density
    sd = SpectralDensity(log_rets)
    println("  Dominant cycles (days): $(round.(sd.dominant_cycles_days; digits=1))")

    # 6. Cointegration
    prices2 = prices .* 1.5 .+ 0.3 .* randn(rng, n)   # cointegrated synthetic pair
    pairs = CointegratedPairs(hcat(prices, prices2); asset_names=["A", "B"])
    println("  Cointegrated pairs found: $(length(pairs))")

    # Write results
    out_dir = get(ENV, "STATS_OUTPUT_DIR", joinpath(@__DIR__, "..", "output"))
    mkpath(out_dir)
    open(joinpath(out_dir, "time_series_results.json"), "w") do io
        write(io, JSON3.write(Dict(
            "hurst"        => h.H,
            "arima_order"  => (arima.p, arima.d, arima.q),
            "dom_cycles"   => sd.dominant_cycles_days,
            "n_coint_pairs"=> length(pairs)
        )))
    end

    println("[time_series] Self-tests complete.")
end
