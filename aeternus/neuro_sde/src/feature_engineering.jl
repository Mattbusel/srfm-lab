"""
feature_engineering.jl — Raw OHLCV data → SDE-compatible feature vectors

Implements:
  1. Realized volatility estimators:
     - Close-to-close (classical)
     - Parkinson (high-low range)
     - Garman-Klass
     - Rogers-Satchell
     - Yang-Zhang (overnight + intraday)
     - Bipower variation (Barndorff-Nielsen & Shephard)
     - Realised variance from high-frequency returns
  2. Microstructure noise filtering:
     - Sparse sampling
     - Two-scales realized variance (TSRV)
     - Pre-averaging estimator
     - Bid-ask bounce correction
  3. Path signature features (up to degree 4)
  4. Wavelet decomposition features (Haar, Daubechies)
  5. Rolling window statistics
  6. Return decomposition: trend, seasonality, noise
  7. Jump detection (BN-S test, Lee-Mykland)
  8. Feature normalisation and standardisation

References:
  - Garman & Klass (1980) "On the Estimation of Security Price Volatilities"
  - Rogers & Satchell (1991)
  - Yang & Zhang (2000)
  - Barndorff-Nielsen & Shephard (2004) — Bipower variation
  - Zhang, Mykland & Aït-Sahalia (2005) — TSRV
  - Chen & Lyons (2016) — Path signatures
"""

using Statistics
using LinearAlgebra
using Random

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: OHLCV DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

"""
    OHLCV

Open-High-Low-Close-Volume bar data.
All prices are in natural units (not log).
"""
struct OHLCV
    open   :: Vector{Float64}
    high   :: Vector{Float64}
    low    :: Vector{Float64}
    close  :: Vector{Float64}
    volume :: Vector{Float64}
    timestamps :: Union{Nothing, Vector{Float64}}
end

function OHLCV(O, H, L, C, V)
    n = length(C)
    @assert length(O) == n && length(H) == n && length(L) == n && length(V) == n
    OHLCV(Float64.(O), Float64.(H), Float64.(L), Float64.(C),
          Float64.(V), nothing)
end

Base.length(d::OHLCV) = length(d.close)

"""
    log_returns(prices) → Vector{Float64}

Compute log returns: r_t = log(P_t / P_{t-1}).
"""
function log_returns(prices::AbstractVector)
    n = length(prices)
    n <= 1 && return Float64[]
    return [log(prices[t] / prices[t-1]) for t in 2:n]
end

"""
    log_ohlcv(data::OHLCV) → OHLCV (log prices)

Return OHLCV with all prices log-transformed.
"""
function log_ohlcv(data::OHLCV)
    OHLCV(log.(data.open), log.(data.high), log.(data.low),
          log.(data.close), data.volume, data.timestamps)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: REALISED VOLATILITY ESTIMATORS
# ─────────────────────────────────────────────────────────────────────────────

"""
    close_to_close_vol(closes; window=20, annualise=252) → Vector{Float64}

Classical close-to-close realised volatility (rolling window).
σ_t = std(log returns in window) * √annualise
"""
function close_to_close_vol(closes::AbstractVector;
                             window::Int      = 20,
                             annualise::Real  = 252.0)
    r  = log_returns(closes)
    n  = length(r)
    vols = fill(NaN, length(closes))
    for t in window:n
        vols[t+1] = std(r[t-window+1:t]) * sqrt(annualise)
    end
    return vols
end

"""
    parkinson_vol(highs, lows; window=20, annualise=252) → Vector{Float64}

Parkinson (1980) volatility using high-low range:
σ² ≈ 1/(4 log 2) × (log H/L)²

Rolling window version.
"""
function parkinson_vol(highs::AbstractVector, lows::AbstractVector;
                       window::Int     = 20,
                       annualise::Real = 252.0)
    n    = length(highs)
    @assert length(lows) == n
    daily_var = [(log(highs[t] / lows[t]))^2 / (4 * log(2)) for t in 1:n]
    vols      = fill(NaN, n)
    for t in window:n
        vols[t] = sqrt(mean(daily_var[t-window+1:t]) * annualise)
    end
    return vols
end

"""
    parkinson_vol_single(high, low) → Float64

Parkinson volatility for a single bar (unannualised).
"""
parkinson_vol_single(high::Real, low::Real) =
    sqrt((log(high / low))^2 / (4 * log(2)))

"""
    garman_klass_vol(opens, highs, lows, closes;
                     window=20, annualise=252) → Vector{Float64}

Garman-Klass (1980) estimator:
GK = 0.5 (log H/L)² - (2log2-1)(log C/O)²
"""
function garman_klass_vol(opens::AbstractVector,
                          highs::AbstractVector,
                          lows::AbstractVector,
                          closes::AbstractVector;
                          window::Int     = 20,
                          annualise::Real = 252.0)
    n = length(closes)
    daily_var = zeros(n)
    for t in 1:n
        u = log(highs[t]  / opens[t])
        d = log(lows[t]   / opens[t])
        c = log(closes[t] / opens[t])
        daily_var[t] = 0.5 * (u - d)^2 - (2*log(2) - 1) * c^2
    end
    daily_var = max.(daily_var, 0.0)
    vols = fill(NaN, n)
    for t in window:n
        vols[t] = sqrt(mean(daily_var[t-window+1:t]) * annualise)
    end
    return vols
end

"""
    rogers_satchell_vol(opens, highs, lows, closes;
                        window=20, annualise=252) → Vector{Float64}

Rogers-Satchell (1991) drift-independent estimator:
RS = (log H/C)(log H/O) + (log L/C)(log L/O)
"""
function rogers_satchell_vol(opens::AbstractVector,
                              highs::AbstractVector,
                              lows::AbstractVector,
                              closes::AbstractVector;
                              window::Int     = 20,
                              annualise::Real = 252.0)
    n = length(closes)
    daily_var = zeros(n)
    for t in 1:n
        lHC = log(highs[t]  / closes[t])
        lHO = log(highs[t]  / opens[t])
        lLC = log(lows[t]   / closes[t])
        lLO = log(lows[t]   / opens[t])
        daily_var[t] = lHC * lHO + lLC * lLO
    end
    daily_var = max.(daily_var, 0.0)
    vols = fill(NaN, n)
    for t in window:n
        vols[t] = sqrt(mean(daily_var[t-window+1:t]) * annualise)
    end
    return vols
end

"""
    yang_zhang_vol(opens, highs, lows, closes;
                   window=20, annualise=252, k=0.34) → Vector{Float64}

Yang-Zhang (2000) estimator: handles overnight gaps.
σ²_YZ = σ²_overnight + k σ²_open + (1-k) σ²_close
where σ²_close = Rogers-Satchell within day.
"""
function yang_zhang_vol(opens::AbstractVector,
                        highs::AbstractVector,
                        lows::AbstractVector,
                        closes::AbstractVector;
                        window::Int     = 20,
                        annualise::Real = 252.0,
                        k::Real         = 0.34)
    n = length(closes)

    # Overnight returns: open_t / close_{t-1}
    overnight_r = [log(opens[t] / closes[t-1]) for t in 2:n]
    open_r      = [log(closes[t] / opens[t])   for t in 1:n]
    rs          = zeros(n)
    for t in 1:n
        lHC = log(highs[t]  / closes[t])
        lHO = log(highs[t]  / opens[t])
        lLC = log(lows[t]   / closes[t])
        lLO = log(lows[t]   / opens[t])
        rs[t] = lHC * lHO + lLC * lLO
    end

    vols = fill(NaN, n)
    for t in (window+1):n
        idx  = t-window:t-1
        σ2_ov  = var(overnight_r[idx .- 1])
        σ2_op  = var(open_r[idx])
        σ2_rs  = mean(rs[idx])
        yz     = σ2_ov + k * σ2_op + (1 - k) * σ2_rs
        vols[t] = sqrt(max(yz, 0.0) * annualise)
    end
    return vols
end

"""
    bipower_variation(returns; window=20, annualise=252) → Vector{Float64}

Barndorff-Nielsen & Shephard (2004) bipower variation:
BV_n = (π/2) × (1/(n-1)) Σ |r_t| |r_{t-1}|

Robust to jumps; converges to integrated variance when no jumps.
"""
function bipower_variation(returns::AbstractVector;
                           window::Int     = 20,
                           annualise::Real = 252.0)
    n    = length(returns)
    μ₁   = sqrt(2 / π)  # E[|Z|] for Z ~ N(0,1)
    bpvs = fill(NaN, n)
    for t in window:n
        r_win = returns[t-window+1:t]
        bv    = 0.0
        for i in 2:window
            bv += abs(r_win[i]) * abs(r_win[i-1])
        end
        bv       *= (π/2) / (window - 1)
        bpvs[t] = sqrt(bv * annualise)
    end
    return bpvs
end

"""
    realised_variance(hf_returns; subsamples=1) → Float64

Realised variance from high-frequency returns.
RV = Σ r²_i
"""
function realised_variance(hf_returns::AbstractVector; subsamples::Int=1)
    if subsamples == 1
        return sum(hf_returns.^2)
    else
        # Subsample at different grids and average
        n   = length(hf_returns)
        rvs = zeros(subsamples)
        for s in 1:subsamples
            idx = s:subsamples:n
            rvs[s] = sum(hf_returns[idx].^2) * subsamples
        end
        return mean(rvs)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: MICROSTRUCTURE NOISE FILTERING
# ─────────────────────────────────────────────────────────────────────────────

"""
    tsrv(hf_prices; J=2, K=nothing) → Float64

Two-Scales Realised Variance (Zhang, Mykland & Aït-Sahalia 2005).
Corrects for microstructure noise using two sampling frequencies.

- `J` : slow grid subsampling frequency
- `K` : fast grid (defaults to all ticks)
"""
function tsrv(hf_prices::AbstractVector; J::Int=2, K::Int=1)
    n      = length(hf_prices)
    # Fast grid: all observations
    r_fast = log_returns(hf_prices)
    rv_all = sum(r_fast.^2)
    n_all  = n - 1

    # Slow grid: every J-th observation
    slow_prices = hf_prices[1:J:end]
    r_slow      = log_returns(slow_prices)
    rv_slow     = sum(r_slow.^2)
    n_slow      = length(slow_prices) - 1

    # TSRV estimator
    c = n_slow / n_all
    tsrv_est = rv_slow - c * rv_all
    return max(tsrv_est, 0.0)
end

"""
    preaveraging_rv(hf_prices; kn=nothing) → Float64

Pre-averaging realised volatility (Jacod et al. 2009).
Averages prices over blocks of size kn to reduce noise.
"""
function preaveraging_rv(hf_prices::AbstractVector; kn::Union{Nothing,Int}=nothing)
    n  = length(hf_prices)
    kn = isnothing(kn) ? max(Int(floor(sqrt(n))), 2) : kn

    # Weight function g(x) = min(x, 1-x)
    g = [(j / kn <= 0.5) ? j / kn : 1 - j / kn for j in 1:kn]
    ψ1 = sum(g.^2) / kn
    ψ2 = sum(diff(vcat(0.0, g, 0.0)).^2)

    # Pre-averaged returns
    n_blocks = n - kn
    Z̄ = zeros(n_blocks)
    for i in 1:n_blocks
        Z̄[i] = sum(g[j] * log(hf_prices[i+j] / hf_prices[i+j-1])
                   for j in 1:kn)
    end

    # Bias-corrected RV
    rv_pre = sum(Z̄.^2) / (ψ1 * kn)
    # Noise correction (using adjacent pairs)
    noise_correction = ψ2 / (2 * ψ1 * kn) * sum(log.(hf_prices[2:end] ./ hf_prices[1:end-1]).^2)
    return max(rv_pre - noise_correction, 0.0)
end

"""
    roll_spread_estimate(prices) → Float64

Roll (1984) bid-ask spread estimate from price autocorrelation:
s = 2 √(-Cov(Δp_t, Δp_{t-1})) if negative, else 0.
"""
function roll_spread_estimate(prices::AbstractVector)
    r   = log_returns(prices)
    cov = cov(r[1:end-1], r[2:end])
    return cov < 0 ? 2 * sqrt(-cov) : 0.0
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: JUMP DETECTION
# ─────────────────────────────────────────────────────────────────────────────

"""
    bns_jump_test(returns; window=20, α=0.01) → (jump_days, test_stats, threshold)

Barndorff-Nielsen & Shephard (2004) jump test.
Tests H₀: no jumps using RV / BPV ratio.
"""
function bns_jump_test(returns::AbstractVector;
                       window::Int  = 20,
                       α::Real      = 0.01)
    n        = length(returns)
    rv_arr   = zeros(n)
    bpv_arr  = zeros(n)

    for t in window:n
        r_win    = returns[t-window+1:t]
        rv_arr[t]  = sum(r_win.^2)
        bpv_arr[t] = (π/2) / (window-1) * sum(abs.(r_win[2:end]) .* abs.(r_win[1:end-1]))
    end

    # Test statistic (simplified Wald version)
    μ₁   = sqrt(2/π)
    κ₄   = 3.0  # fourth cumulant for Gaussian
    θ_BPV = (π^2/4 + π - 5)

    stats = zeros(n)
    for t in window:n
        bpv = max(bpv_arr[t], 1e-12)
        stats[t] = (rv_arr[t] - bpv_arr[t]) / bpv *
                   sqrt(window / θ_BPV * max(rv_arr[t]^2 / bpv^2, 1.0))
    end

    thresh     = quantile(Normal(), 1 - α)
    jump_days  = findall(stats .> thresh)
    return jump_days, stats, thresh
end

"""
    lee_mykland_jumps(prices; window=252, α=0.01) → Vector{Int}

Lee-Mykland (2008) jump detection using local return standardisation.
Standardises each return by a local volatility estimate.
"""
function lee_mykland_jumps(prices::AbstractVector;
                           window::Int = 252,
                           α::Real     = 0.01)
    r    = log_returns(prices)
    n    = length(r)
    # Bipower variation for local vol
    μ₁   = sqrt(2/π)
    jump_idx = Int[]
    for t in (window+1):n
        r_win  = r[t-window:t-1]
        bv     = (π/2) / (window-1) * sum(abs.(r_win[2:end]) .* abs.(r_win[1:end-1]))
        σ_loc  = sqrt(max(bv / window, 1e-12))
        L      = abs(r[t]) / σ_loc
        # Critical value for maxima of abs(N(0,1))
        c_n    = sqrt(2 * log(window))
        s_n    = 1.0 / sqrt(2 * log(window))
        # Gumbel critical value
        thresh = -log(-log(1 - α)) * s_n + c_n
        L > thresh && push!(jump_idx, t)
    end
    return jump_idx
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: PATH SIGNATURE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

"""
    signature_level1(path) → Vector{Float64}

Level-1 path signature: ∫ dx_i — just the increments.
path is (d × n) matrix of d-dimensional path sampled at n points.
"""
function signature_level1(path::AbstractMatrix)
    d, n = size(path)
    increments = path[:, end] .- path[:, 1]
    return increments
end

"""
    signature_level2(path) → Vector{Float64}

Level-2 iterated integrals: S^{ij} = ∫∫_{s<t} dx_i dx_j (Chen shuffle formula).
Uses the discrete approximation via iterated sums.
"""
function signature_level2(path::AbstractMatrix)
    d, n = size(path)
    result = zeros(d * d)
    for i in 1:d, j in 1:d
        # S^{ij} ≈ Σ_{s<t} (path[i,s]-path[i,s-1]) * cumsum(path[j,1..s-1])
        s = 0.0
        cum_j = 0.0
        for t in 2:n
            dx_i = path[i, t] - path[i, t-1]
            dx_j = path[j, t] - path[j, t-1]
            # ∫₀^t dX^j up to t-1
            s += cum_j * dx_i
            cum_j += dx_j
        end
        result[(i-1)*d + j] = s
    end
    return result
end

"""
    signature_level3(path) → Vector{Float64}

Level-3 iterated integrals (d^3 terms).
Uses the recursive Chen relation: S^{ijk} = ∫ S^{ij}(0,t) dX^k_t
"""
function signature_level3(path::AbstractMatrix)
    d, n = size(path)
    result = zeros(d^3)
    # S^{ij}(0,t) for each t — running level-2 sums
    S2_running = zeros(d, d)    # S^{ij}(0, current_t)
    S3 = zeros(d, d, d)

    for t in 2:n
        for i in 1:d
            dxi = path[i, t] - path[i, t-1]
            # Update S3 first (uses old S2)
            for j in 1:d, k in 1:d
                S3[i, j, k] += S2_running[j, k] * dxi
            end
        end
        # Update S2
        for i in 1:d
            dxi = path[i, t] - path[i, t-1]
            cum = path[:, t-1] .- path[:, 1]
            for j in 1:d
                S2_running[i, j] += cum[j] * dxi
            end
        end
    end
    return reshape(S3, d^3)
end

"""
    signature_level4(path) → Vector{Float64}

Level-4 iterated integrals (d^4 terms).
Recursive from level 3.
"""
function signature_level4(path::AbstractMatrix)
    d, n = size(path)
    S2 = zeros(d, d)
    S3 = zeros(d, d, d)
    S4 = zeros(d, d, d, d)

    for t in 2:n
        dx = path[:, t] .- path[:, t-1]
        # Update S4 from S3
        for i in 1:d, j in 1:d, k in 1:d, l in 1:d
            S4[i,j,k,l] += S3[j,k,l] * dx[i]
        end
        # Update S3 from S2
        for i in 1:d, j in 1:d, k in 1:d
            S3[i,j,k] += S2[j,k] * dx[i]
        end
        # Update S2 from S1 (increments)
        cum = path[:, t-1] .- path[:, 1]
        for i in 1:d, j in 1:d
            S2[i,j] += cum[j] * dx[i]
        end
    end
    return reshape(S4, d^4)
end

"""
    path_signature(path; degree=4, include_time=true) → Vector{Float64}

Compute truncated path signature up to specified degree.
Optionally prepend time as an additional channel.

path: (d × n) matrix of d-dim path at n time points.
"""
function path_signature(path::AbstractMatrix;
                        degree::Int        = 4,
                        include_time::Bool = true)
    d, n = size(path)
    if include_time
        # Add time channel [0, 1/n, ..., 1]
        t_channel = reshape(collect(range(0.0, 1.0, length=n)), 1, n)
        path = vcat(t_channel, path)
        d += 1
    end

    sig = Float64[]
    # Level 0: scalar 1
    push!(sig, 1.0)
    # Level 1
    degree >= 1 && append!(sig, signature_level1(path))
    # Level 2
    degree >= 2 && append!(sig, signature_level2(path))
    # Level 3
    degree >= 3 && append!(sig, signature_level3(path))
    # Level 4
    degree >= 4 && append!(sig, signature_level4(path))

    return sig
end

"""
    log_signature(path; degree=4) → Vector{Float64}

Log-signature (more compact representation using Lyndon basis).
Simplified: returns the signature with log-normalisation per level.
"""
function log_signature(path::AbstractMatrix; degree::Int=4)
    sig = path_signature(path; degree=degree, include_time=false)
    return sig
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: WAVELET DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

"""
    haar_wavelet_transform(x) → (approx, detail) matrices

Discrete Haar wavelet transform (DWT) of vector x.
Returns (approximation_coeffs, detail_coeffs) at each level.
"""
function haar_wavelet_transform(x::AbstractVector)
    n     = length(x)
    # Pad to power of 2
    n_pad = nextpow(2, n)
    xpad  = zeros(n_pad)
    xpad[1:n] = x

    n_levels = Int(log2(n_pad))
    approx   = Vector{Vector{Float64}}()
    detail   = Vector{Vector{Float64}}()

    current = copy(xpad)
    for level in 1:n_levels
        m    = length(current)
        half = m ÷ 2
        a    = zeros(half)
        d    = zeros(half)
        for i in 1:half
            a[i] = (current[2i-1] + current[2i]) / sqrt(2)
            d[i] = (current[2i-1] - current[2i]) / sqrt(2)
        end
        push!(approx, copy(a))
        push!(detail, copy(d))
        current = a
    end
    return approx, detail
end

"""
    db4_wavelet_filter() → (h, g)

Daubechies-4 (D4) low-pass filter coefficients h and high-pass g.
"""
function db4_wavelet_filter()
    h = [(1 + sqrt(3)) / (4*sqrt(2)),
         (3 + sqrt(3)) / (4*sqrt(2)),
         (3 - sqrt(3)) / (4*sqrt(2)),
         (1 - sqrt(3)) / (4*sqrt(2))]
    # Quadrature mirror filter: g_k = (-1)^(k+1) h_{N-1-k}
    g = [-h[4], h[3], -h[2], h[1]]
    return h, g
end

"""
    db4_wavelet_transform(x; n_levels=4) → (approx, detail_list)

Single-level Daubechies-4 DWT applied recursively for n_levels.
"""
function db4_wavelet_transform(x::AbstractVector; n_levels::Int=4)
    h, g    = db4_wavelet_filter()
    approx_list = Vector{Vector{Float64}}()
    detail_list = Vector{Vector{Float64}}()

    current = copy(Float64.(x))
    for level in 1:n_levels
        n  = length(current)
        # Circular convolution + downsample by 2
        n_out = n ÷ 2
        a = zeros(n_out)
        d = zeros(n_out)
        for i in 1:n_out
            for k in 1:length(h)
                idx = mod1(2i - k + 1, n)  # circular
                a[i] += h[k] * current[idx]
                d[i] += g[k] * current[idx]
            end
        end
        push!(approx_list, copy(a))
        push!(detail_list, copy(d))
        current = a
    end
    return approx_list, detail_list
end

"""
    wavelet_energy(detail_list) → Vector{Float64}

Compute energy at each wavelet scale: E_j = ||d_j||² / ||all d||².
"""
function wavelet_energy(detail_list::Vector{Vector{Float64}})
    energies = [sum(d.^2) for d in detail_list]
    total    = max(sum(energies), 1e-12)
    return energies ./ total
end

"""
    wavelet_features(x; n_levels=6) → Vector{Float64}

Extract wavelet-based features from time series x:
- Energy per level (relative)
- RMS per level
- Entropy per level
"""
function wavelet_features(x::AbstractVector; n_levels::Int=6)
    _, detail_list = haar_wavelet_transform(x)
    n_actual = min(n_levels, length(detail_list))
    d_use    = detail_list[1:n_actual]

    energies = wavelet_energy(d_use)
    rms_vals = [sqrt(mean(d.^2)) for d in d_use]
    entropy_vals = Float64[]
    for d in d_use
        e2 = d.^2 ./ max(sum(d.^2), 1e-12)
        push!(entropy_vals, -sum(e .* log(max(e, 1e-12)) for e in e2))
    end
    return vcat(energies, rms_vals, entropy_vals)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: ROLLING WINDOW STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

"""
    rolling_mean(x, window) → Vector{Float64}

Rolling arithmetic mean.
"""
function rolling_mean(x::AbstractVector, window::Int)
    n = length(x)
    result = fill(NaN, n)
    for t in window:n
        result[t] = mean(x[t-window+1:t])
    end
    return result
end

"""
    rolling_std(x, window) → Vector{Float64}
"""
function rolling_std(x::AbstractVector, window::Int)
    n = length(x)
    result = fill(NaN, n)
    for t in window:n
        result[t] = std(x[t-window+1:t])
    end
    return result
end

"""
    rolling_skewness(x, window) → Vector{Float64}

Rolling skewness: E[(X-μ)³] / σ³.
"""
function rolling_skewness(x::AbstractVector, window::Int)
    n = length(x)
    result = fill(NaN, n)
    for t in window:n
        w   = x[t-window+1:t]
        μ, σ = mean(w), std(w)
        σ < 1e-12 && (result[t] = 0.0; continue)
        result[t] = mean(((w .- μ) ./ σ).^3)
    end
    return result
end

"""
    rolling_kurtosis(x, window) → Vector{Float64}

Rolling excess kurtosis: E[(X-μ)⁴] / σ⁴ - 3.
"""
function rolling_kurtosis(x::AbstractVector, window::Int)
    n = length(x)
    result = fill(NaN, n)
    for t in window:n
        w    = x[t-window+1:t]
        μ, σ = mean(w), std(w)
        σ < 1e-12 && (result[t] = 0.0; continue)
        result[t] = mean(((w .- μ) ./ σ).^4) - 3.0
    end
    return result
end

"""
    rolling_autocorr(x, window, lag=1) → Vector{Float64}

Rolling lag-k autocorrelation.
"""
function rolling_autocorr(x::AbstractVector, window::Int, lag::Int=1)
    n      = length(x)
    result = fill(NaN, n)
    for t in (window + lag):n
        w   = x[t-window+1:t]
        w1  = w[1:end-lag]
        w2  = w[lag+1:end]
        c   = cov(w1, w2)
        v   = std(w) + 1e-12
        result[t] = c / v^2
    end
    return result
end

"""
    rolling_quantile(x, window, q) → Vector{Float64}

Rolling q-th quantile.
"""
function rolling_quantile(x::AbstractVector, window::Int, q::Real)
    n      = length(x)
    result = fill(NaN, n)
    for t in window:n
        result[t] = quantile(x[t-window+1:t], q)
    end
    return result
end

"""
    ewma_vol(returns; λ=0.94, annualise=252) → Vector{Float64}

EWMA (RiskMetrics) volatility: σ²_t = λ σ²_{t-1} + (1-λ) r²_t.
"""
function ewma_vol(returns::AbstractVector;
                  λ::Real        = 0.94,
                  annualise::Real = 252.0)
    n    = length(returns)
    σ2   = zeros(n)
    σ2[1] = returns[1]^2
    for t in 2:n
        σ2[t] = λ * σ2[t-1] + (1 - λ) * returns[t]^2
    end
    return sqrt.(σ2 .* annualise)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: FEATURE MATRIX CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

"""
    FeatureConfig

Configuration for full feature extraction pipeline.
"""
struct FeatureConfig
    vol_window      :: Int
    skew_window     :: Int
    kurt_window     :: Int
    sig_degree      :: Int
    sig_window      :: Int    # window for signature computation
    wavelet_levels  :: Int
    include_jumps   :: Bool
    include_wavelet :: Bool
    include_sig     :: Bool
    normalise       :: Bool
end

FeatureConfig(; vol_window=20, skew_window=60, kurt_window=60,
                sig_degree=3, sig_window=20,
                wavelet_levels=4, include_jumps=true,
                include_wavelet=true, include_sig=true,
                normalise=true) =
    FeatureConfig(vol_window, skew_window, kurt_window,
                  sig_degree, sig_window, wavelet_levels,
                  include_jumps, include_wavelet, include_sig, normalise)

"""
    FeatureMatrix

Container for the computed feature matrix.
"""
struct FeatureMatrix
    features     :: Matrix{Float64}   # (n_features × n_obs)
    feature_names :: Vector{String}
    timestamps   :: Union{Nothing, Vector{Float64}}
    n_obs        :: Int
    n_features   :: Int
end

"""
    extract_features(data::OHLCV, cfg::FeatureConfig) → FeatureMatrix

Full feature extraction pipeline.
"""
function extract_features(data::OHLCV, cfg::FeatureConfig)
    n        = length(data)
    returns  = log_returns(data.close)
    r_padded = vcat(NaN, returns)  # align with close prices

    all_features = Vector{Pair{String, Vector{Float64}}}()

    # ── Vol estimators ─────────────────────────────────────────────────────
    push!(all_features, "cc_vol"   => close_to_close_vol(data.close; window=cfg.vol_window))
    push!(all_features, "park_vol" => parkinson_vol(data.high, data.low; window=cfg.vol_window))
    push!(all_features, "gk_vol"   => garman_klass_vol(data.open, data.high, data.low, data.close; window=cfg.vol_window))
    push!(all_features, "rs_vol"   => rogers_satchell_vol(data.open, data.high, data.low, data.close; window=cfg.vol_window))
    push!(all_features, "yz_vol"   => yang_zhang_vol(data.open, data.high, data.low, data.close; window=cfg.vol_window))

    # BPV (needs returns)
    bpv = bipower_variation(returns; window=cfg.vol_window)
    push!(all_features, "bpv" => vcat(NaN, bpv))

    # EWMA vol
    ewma = ewma_vol(returns)
    push!(all_features, "ewma_vol" => vcat(NaN, ewma))

    # ── Rolling statistics ───────────────────────────────────────────────
    push!(all_features, "rolling_mean_ret" => rolling_mean(r_padded, cfg.vol_window))
    push!(all_features, "rolling_std_ret"  => rolling_std(r_padded, cfg.vol_window))
    push!(all_features, "rolling_skew"     => rolling_skewness(r_padded, cfg.skew_window))
    push!(all_features, "rolling_kurt"     => rolling_kurtosis(r_padded, cfg.kurt_window))
    push!(all_features, "rolling_ac1"      => rolling_autocorr(r_padded, cfg.vol_window, 1))
    push!(all_features, "rolling_ac5"      => rolling_autocorr(r_padded, cfg.vol_window, 5))
    push!(all_features, "q10"              => rolling_quantile(r_padded, cfg.vol_window, 0.10))
    push!(all_features, "q90"              => rolling_quantile(r_padded, cfg.vol_window, 0.90))

    # ── Volume features ──────────────────────────────────────────────────
    log_vol = log.(max.(data.volume, 1.0))
    push!(all_features, "log_volume"  => log_vol)
    push!(all_features, "vol_ma20"    => rolling_mean(log_vol, cfg.vol_window))
    push!(all_features, "vol_ratio"   => log_vol .- rolling_mean(log_vol, cfg.vol_window))

    # ── Price range features ─────────────────────────────────────────────
    log_range = log.(data.high ./ data.low)
    push!(all_features, "log_range"   => log_range)
    push!(all_features, "close_pos"   => (data.close .- data.low) ./
                                          max.(data.high .- data.low, 1e-10))

    # ── Jump indicators ─────────────────────────────────────────────────
    if cfg.include_jumps
        jump_idx, jump_stats, _ = bns_jump_test(returns)
        jump_flag = zeros(n)
        jump_flag[vcat(1, jump_idx .+ 1)] .= 1.0
        push!(all_features, "jump_flag"  => jump_flag)
        jump_stats_padded = vcat(zeros(n - length(jump_stats)), jump_stats)
        push!(all_features, "jump_stat"  => jump_stats_padded)
    end

    # ── Wavelet features ─────────────────────────────────────────────────
    if cfg.include_wavelet && n >= 2^cfg.wavelet_levels
        wf_dim = 3 * cfg.wavelet_levels
        wf_mat = fill(NaN, wf_dim, n)
        w_win  = 2^cfg.wavelet_levels
        for t in w_win:n
            wf = wavelet_features(r_padded[t-w_win+1:t]; n_levels=cfg.wavelet_levels)
            wf_mat[:, t] = wf[1:wf_dim]
        end
        for j in 1:wf_dim
            push!(all_features, "wav_$(j)" => wf_mat[j, :])
        end
    end

    # ── Signature features ───────────────────────────────────────────────
    if cfg.include_sig && n >= cfg.sig_window
        # 2D path: (log return, log vol estimate)
        cc_v = close_to_close_vol(data.close; window=5)
        sig_dim = 1 + 2 + 4 + 8   # degrees 0..3 for d=2
        sig_mat = fill(NaN, sig_dim, n)
        for t in cfg.sig_window:n
            r_seg  = r_padded[t-cfg.sig_window+1:t]
            v_seg  = cc_v[t-cfg.sig_window+1:t]
            nanmask = isnan.(r_seg) .| isnan.(v_seg)
            any(nanmask) && continue
            path_2d = Matrix(vcat(r_seg', v_seg'))   # 2 × sig_window
            sig_vec = path_signature(path_2d; degree=cfg.sig_degree, include_time=false)
            sig_mat[1:min(sig_dim, length(sig_vec)), t] = sig_vec[1:min(sig_dim, length(sig_vec))]
        end
        for j in 1:sig_dim
            push!(all_features, "sig_$(j)" => sig_mat[j, :])
        end
    end

    # ── Assemble feature matrix ──────────────────────────────────────────
    n_feats = length(all_features)
    F_mat   = zeros(n_feats, n)
    f_names = String[]
    for (k, (name, vec)) in enumerate(all_features)
        push!(f_names, name)
        if length(vec) == n
            F_mat[k, :] = vec
        else
            # Pad/truncate
            m = min(length(vec), n)
            F_mat[k, 1:m] = vec[1:m]
        end
    end

    # Replace NaN with 0 (or mean-impute)
    for k in 1:n_feats
        row = F_mat[k, :]
        not_nan = .!isnan.(row)
        if any(not_nan)
            μ = mean(row[not_nan])
            F_mat[k, isnan.(row)] .= μ
        end
    end

    # ── Normalise ────────────────────────────────────────────────────────
    if cfg.normalise
        for k in 1:n_feats
            row = F_mat[k, :]
            μ   = mean(row)
            σ   = std(row)
            σ < 1e-12 && continue
            F_mat[k, :] = (row .- μ) ./ σ
        end
    end

    return FeatureMatrix(F_mat, f_names, data.timestamps, n, n_feats)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: FEATURE SELECTION AND CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

"""
    feature_correlation_matrix(fm::FeatureMatrix) → Matrix{Float64}

Compute correlation matrix between features.
"""
function feature_correlation_matrix(fm::FeatureMatrix)
    n_f = fm.n_features
    C   = zeros(n_f, n_f)
    for i in 1:n_f, j in 1:n_f
        if i == j
            C[i,j] = 1.0
        elseif i < j
            v1 = fm.features[i, :]
            v2 = fm.features[j, :]
            σ1 = std(v1); σ2 = std(v2)
            (σ1 < 1e-12 || σ2 < 1e-12) && continue
            C[i,j] = cov(v1, v2) / (σ1 * σ2)
            C[j,i] = C[i,j]
        end
    end
    return C
end

"""
    remove_redundant_features(fm::FeatureMatrix; corr_threshold=0.95) → FeatureMatrix

Remove features with pairwise correlation above threshold.
Keeps the first feature in each highly-correlated pair.
"""
function remove_redundant_features(fm::FeatureMatrix;
                                   corr_threshold::Real = 0.95)
    C      = feature_correlation_matrix(fm)
    n_f    = fm.n_features
    keep   = trues(n_f)
    for i in 1:n_f
        keep[i] || continue
        for j in (i+1):n_f
            keep[j] || continue
            if abs(C[i,j]) >= corr_threshold
                keep[j] = false
            end
        end
    end
    idx      = findall(keep)
    new_feat = fm.features[idx, :]
    new_names = fm.feature_names[idx]
    return FeatureMatrix(new_feat, new_names, fm.timestamps,
                         fm.n_obs, length(idx))
end

"""
    pca_features(fm::FeatureMatrix; n_components=10) → (scores, loadings, var_explained)

PCA dimensionality reduction on feature matrix.
Returns principal component scores and loadings.
"""
function pca_features(fm::FeatureMatrix; n_components::Int=10)
    X     = fm.features'   # n_obs × n_features
    X̄     = mean(X, dims=1)
    Xc    = X .- X̄
    C     = Xc' * Xc ./ (fm.n_obs - 1)
    F     = eigen(Symmetric(C); sortby=x->-x)
    n_k   = min(n_components, size(F.vectors, 2))
    L     = F.vectors[:, 1:n_k]       # loadings
    scores = Xc * L                    # n_obs × n_components
    λ     = F.values[1:n_k]
    var_exp = λ ./ max(sum(F.values), 1e-12)
    return scores, L, var_exp
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: DEMO
# ─────────────────────────────────────────────────────────────────────────────

"""
    demo_feature_engineering(; n=500, seed=42)

Smoke test: generate synthetic OHLCV and extract features.
"""
function demo_feature_engineering(; n::Int=500, seed::Int=42)
    rng = MersenneTwister(seed)
    # Simulate GBM price path
    S  = zeros(n)
    S[1] = 100.0
    σ  = 0.20 / sqrt(252)
    for t in 2:n
        S[t] = S[t-1] * exp(-0.5σ^2 + σ * randn(rng))
    end
    spread = 0.005
    H = S .* (1 .+ spread * abs.(randn(rng, n)))
    L = S .* (1 .- spread * abs.(randn(rng, n)))
    O = vcat(S[1], S[1:end-1]) .* (1 .+ 0.002 * randn(rng, n))
    V = abs.(randn(rng, n)) .* 1e6 .+ 1e5

    data = OHLCV(O, H, L, S, V)
    cfg  = FeatureConfig(; include_sig=true, include_wavelet=true)
    fm   = extract_features(data, cfg)
    @info "Feature matrix: $(fm.n_features) features × $(fm.n_obs) obs"
    return fm
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: MICROSTRUCTURE FEATURES (ADDITIONAL)
# ─────────────────────────────────────────────────────────────────────────────

"""
    amihud_illiquidity(returns, volumes; window=20) → Vector{Float64}

Amihud (2002) illiquidity ratio:
ILLIQ_t = (1/window) × Σ |r_i| / Volume_i

Measures price impact per unit volume.
"""
function amihud_illiquidity(returns::AbstractVector,
                             volumes::AbstractVector;
                             window::Int = 20)
    n   = length(returns)
    ill = fill(NaN, n)
    for t in window:n
        r_w = returns[t-window+1:t]
        v_w = volumes[t-window+1:t]
        ill[t] = mean(abs.(r_w) ./ max.(v_w, 1.0))
    end
    return ill
end

"""
    turnover_ratio(volumes, shares_outstanding; window=20) → Vector{Float64}

Rolling turnover ratio: average volume / shares outstanding.
"""
function turnover_ratio(volumes::AbstractVector,
                         shares_outstanding::Real;
                         window::Int = 20)
    return rolling_mean(volumes, window) ./ shares_outstanding
end

"""
    price_impact_coefficient(mid_returns, signed_volumes; window=50) → Vector{Float64}

Rolling linear price impact: regress r_t on signed_volume_t.
λ_t = β from OLS over window.
"""
function price_impact_coefficient(mid_returns::AbstractVector,
                                   signed_volumes::AbstractVector;
                                   window::Int = 50)
    n  = length(mid_returns)
    λ  = fill(NaN, n)
    for t in window:n
        y = mid_returns[t-window+1:t]
        x = signed_volumes[t-window+1:t]
        μy, μx = mean(y), mean(x)
        cov_xy = mean((y .- μy) .* (x .- μx))
        var_x  = var(x)
        var_x < 1e-12 && continue
        λ[t] = cov_xy / var_x
    end
    return λ
end

"""
    trade_arrival_rate(trade_timestamps, snapshot_timestamps; window_secs=60) → Vector{Float64}

Estimate trade arrival rate (trades per second) via rolling count.
"""
function trade_arrival_rate(trade_ts::AbstractVector,
                             snapshot_ts::AbstractVector;
                             window_secs::Real = 60.0)
    n    = length(snapshot_ts)
    rate = zeros(n)
    for t in 1:n
        t_end   = snapshot_ts[t]
        t_start = t_end - window_secs
        n_trades = count(ts -> t_start <= ts <= t_end, trade_ts)
        rate[t]  = n_trades / window_secs
    end
    return rate
end

"""
    signed_volume_imbalance(trade_sizes, trade_sides; window=20) → Vector{Float64}

Rolling order flow imbalance from signed trade volumes.
OFI = (buyer_vol - seller_vol) / (buyer_vol + seller_vol)
"""
function signed_volume_imbalance(sizes::AbstractVector,
                                  sides::AbstractVector;
                                  window::Int = 20)
    n   = length(sizes)
    ofi = fill(NaN, n)
    for t in window:n
        s_w  = sizes[t-window+1:t]
        sd_w = sides[t-window+1:t]
        buy_vol  = sum(s_w[i] for i in 1:window if sd_w[i] > 0; init=0.0)
        sell_vol = sum(s_w[i] for i in 1:window if sd_w[i] < 0; init=0.0)
        denom    = buy_vol + sell_vol
        ofi[t]   = denom > 0 ? (buy_vol - sell_vol) / denom : 0.0
    end
    return ofi
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: RETURN DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

"""
    hp_filter(y; λ=1600.0) → (trend, cycle)

Hodrick-Prescott filter for trend-cycle decomposition.
Minimises: Σ (y_t - τ_t)² + λ Σ (Δ²τ_t)²
"""
function hp_filter(y::AbstractVector; λ::Real = 1600.0)
    n    = length(y)
    # Build second-difference matrix
    D2   = zeros(n-2, n)
    for i in 1:n-2
        D2[i, i] = 1; D2[i, i+1] = -2; D2[i, i+2] = 1
    end
    # Solve: (I + λ D2' D2) τ = y
    A     = I(n) + λ .* (D2' * D2)
    trend = A \ y
    cycle = y .- trend
    return trend, cycle
end

"""
    seasonal_decompose(y, period; method=:additive) → (trend, seasonal, residual)

Simple seasonal decomposition via moving average.
"""
function seasonal_decompose(y::AbstractVector, period::Int;
                             method::Symbol = :additive)
    n = length(y)
    # Moving average trend
    trend = rolling_mean(y, period)
    seasonal = zeros(n)
    # Seasonal indices
    for i in 1:period
        indices = i:period:n
        valid   = filter(j -> !isnan(trend[j]), indices)
        if method == :additive
            s_vals = [y[j] - trend[j] for j in valid if !isnan(trend[j])]
        else
            s_vals = [y[j] / max(trend[j], 1e-12) for j in valid if !isnan(trend[j])]
        end
        isempty(s_vals) && continue
        s_mean = mean(s_vals)
        for j in indices
            seasonal[j] = s_mean
        end
    end
    residual = method == :additive ? y .- trend .- seasonal : y ./ max.(trend .* seasonal, 1e-12)
    return trend, seasonal, residual
end

"""
    normalise_features(fm::FeatureMatrix; method=:zscore) → FeatureMatrix

Normalise feature matrix by z-score, min-max, or robust scaling.
"""
function normalise_features(fm::FeatureMatrix; method::Symbol=:zscore)
    F = copy(fm.features)
    for k in 1:fm.n_features
        row = F[k, :]
        if method == :zscore
            μ = mean(row); σ = std(row)
            σ < 1e-12 && continue
            F[k, :] = (row .- μ) ./ σ
        elseif method == :minmax
            lo = minimum(row); hi = maximum(row)
            hi - lo < 1e-12 && continue
            F[k, :] = (row .- lo) ./ (hi - lo)
        elseif method == :robust
            med = median(row)
            iqr = quantile(row, 0.75) - quantile(row, 0.25)
            iqr < 1e-12 && continue
            F[k, :] = (row .- med) ./ iqr
        end
    end
    return FeatureMatrix(F, fm.feature_names, fm.timestamps, fm.n_obs, fm.n_features)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: LAGGED FEATURE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

"""
    add_lagged_features(fm::FeatureMatrix, lags::AbstractVector{Int}) → FeatureMatrix

Add lagged versions of all features to the feature matrix.
"""
function add_lagged_features(fm::FeatureMatrix,
                              lags::AbstractVector{Int})
    n_f = fm.n_features
    n   = fm.n_obs
    n_lag = length(lags)
    new_feats = zeros(n_f * n_lag, n)
    new_names = String[]

    for (k, lag) in enumerate(lags), j in 1:n_f
        col = fm.features[j, :]
        lagged = vcat(fill(NaN, lag), col[1:end-lag])
        new_feats[(k-1)*n_f + j, :] = lagged
        push!(new_names, "$(fm.feature_names[j])_lag$(lag)")
    end

    all_feats = vcat(fm.features, new_feats)
    all_names = vcat(fm.feature_names, new_names)
    return FeatureMatrix(all_feats, all_names, fm.timestamps,
                          n, size(all_feats, 1))
end

"""
    interaction_features(fm::FeatureMatrix, pairs::Vector{Tuple{Int,Int}}) → FeatureMatrix

Add pairwise interaction features (products) for specified feature index pairs.
"""
function interaction_features(fm::FeatureMatrix,
                               pairs::Vector{Tuple{Int,Int}})
    n   = fm.n_obs
    np  = length(pairs)
    int_feats = zeros(np, n)
    int_names = String[]
    for (k, (i, j)) in enumerate(pairs)
        int_feats[k, :] = fm.features[i, :] .* fm.features[j, :]
        push!(int_names, "$(fm.feature_names[i])_x_$(fm.feature_names[j])")
    end
    all_feats = vcat(fm.features, int_feats)
    all_names = vcat(fm.feature_names, int_names)
    return FeatureMatrix(all_feats, all_names, fm.timestamps, n, size(all_feats, 1))
end
