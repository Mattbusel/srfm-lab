# =============================================================================
# wavelet_analysis.jl — Wavelet-Based Signal Analysis
# =============================================================================
# Provides:
#   - HaarTransform        Haar DWT forward and inverse
#   - Daubechies4Transform D4 wavelet (lifting scheme)
#   - MultiResolutionAnalysis  Decompose P&L into trend + cycle + noise
#   - DominantCycles       Identify periodic components (weekly/monthly)
#   - WaveletCoherence     Co-movement between BTC price and strategy P&L
#   - Scalogram            Time-frequency energy map (JSON for dashboard)
#   - run_wavelet_analysis Top-level driver
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, JSON3
# =============================================================================

module WaveletAnalysis

using Statistics
using LinearAlgebra
using JSON3

export HaarTransform, Daubechies4Transform, MultiResolutionAnalysis
export DominantCycles, WaveletCoherence, Scalogram
export run_wavelet_analysis

# ── Utility ───────────────────────────────────────────────────────────────────

"""Pad vector to next power of 2 (reflect padding to reduce boundary effects)."""
function _pad_pow2(x::Vector{Float64})
    n      = length(x)
    n_pad  = nextpow(2, n)
    if n_pad == n
        return x, n
    end
    # Reflect padding
    extra = n_pad - n
    vcat(x, reverse(x[end-extra+1:end]))
end

"""Next power of two ≥ n."""
function nextpow(base::Int, n::Int)
    p = 1
    while p < n; p *= base; end
    p
end

# ── Haar Wavelet Transform ────────────────────────────────────────────────────

"""
In-place Haar DWT on a power-of-2 length vector.
Uses the lifting scheme: predict step (detail) + update step (smooth).
Returns (approximation_coeffs, detail_coeffs_by_level).
"""
function HaarTransform(x::AbstractVector{<:Real}; levels::Int = -1)
    v  = Float64.(x)
    n  = length(v)
    ispow2(n) || error("HaarTransform requires power-of-2 length; got $n")

    max_levels = round(Int, log2(n))
    lv         = levels < 1 ? max_levels : min(levels, max_levels)

    details = Vector{Vector{Float64}}()
    approx  = copy(v)

    for _ in 1:lv
        L   = length(approx)
        L2  = L ÷ 2
        a   = Vector{Float64}(undef, L2)
        d   = Vector{Float64}(undef, L2)
        for i in 1:L2
            s      = (approx[2i-1] + approx[2i]) * 0.5
            t      = (approx[2i-1] - approx[2i]) * 0.5
            a[i]   = s
            d[i]   = t
        end
        push!(details, d)
        approx = a
    end

    # details[1] = finest scale, details[end] = coarsest detail
    (approx = approx, details = details, levels = lv, original_length = n)
end

"""Inverse Haar DWT: reconstruct signal from approximation + details."""
function HaarInverse(approx::Vector{Float64}, details::Vector{Vector{Float64}})
    a = copy(approx)
    for d in reverse(details)
        L    = length(d)
        out  = Vector{Float64}(undef, 2L)
        for i in 1:L
            out[2i-1] = a[i] + d[i]
            out[2i]   = a[i] - d[i]
        end
        a = out
    end
    a
end

# ── Daubechies-4 Wavelet (D4) ─────────────────────────────────────────────────

"""
Daubechies-4 wavelet filter coefficients (Mallat convention).
"""
const D4_LO_DEC = let
    s3 = sqrt(3.0)
    r8 = sqrt(8.0)
    [(1+s3)/r8, (3+s3)/r8, (3-s3)/r8, (1-s3)/r8]
end

const D4_HI_DEC = let
    lo = D4_LO_DEC
    [lo[4], -lo[3], lo[2], -lo[1]]
end

"""
Apply D4 convolution-based DWT (one level).
Periodic boundary conditions.
"""
function _d4_level(x::Vector{Float64})
    n   = length(x)
    n2  = n ÷ 2
    a   = zeros(Float64, n2)
    d   = zeros(Float64, n2)
    for i in 1:n2
        idx = [(2i-1 - 1 + k) % n + 1 for k in 0:3]
        for k in 1:4
            a[i] += D4_LO_DEC[k] * x[idx[k]]
            d[i] += D4_HI_DEC[k] * x[idx[k]]
        end
    end
    a, d
end

"""
Multi-level Daubechies-4 DWT.
Input must have power-of-2 length.
"""
function Daubechies4Transform(x::AbstractVector{<:Real}; levels::Int = -1)
    v  = Float64.(x)
    n  = length(v)
    ispow2(n) || error("Daubechies4Transform requires power-of-2 length; got $n")

    max_levels = round(Int, log2(n)) - 1   # D4 needs ≥ 4 samples per level
    lv         = levels < 1 ? max_levels : min(levels, max_levels)

    details = Vector{Vector{Float64}}()
    approx  = copy(v)
    for _ in 1:lv
        length(approx) < 4 && break
        a, d = _d4_level(approx)
        push!(details, d)
        approx = a
    end

    (approx = approx, details = details, levels = length(details), original_length = n)
end

"""
D4 inverse (one level) — transpose convolution.
"""
function _d4_inv_level(a::Vector{Float64}, d::Vector{Float64})
    n2  = length(a)
    n   = 2n2
    x   = zeros(Float64, n)
    lo  = D4_LO_DEC
    hi  = D4_HI_DEC
    for i in 1:n2
        for (k, flo, fhi) in zip(0:3, lo, hi)
            idx = (2i - 1 - 1 + k) % n + 1
            x[idx] += a[i] * flo + d[i] * fhi
        end
    end
    x
end

function Daubechies4Inverse(approx::Vector{Float64}, details::Vector{Vector{Float64}})
    a = copy(approx)
    for d in reverse(details)
        a = _d4_inv_level(a, d)
    end
    a
end

# ── Multi-Resolution Analysis ─────────────────────────────────────────────────

"""
Decompose a P&L curve into trend + cycle components + noise using Haar wavelets.

Level assignments (assuming daily data, 1 sample/day):
  Level 1: 2-day oscillations (noise)
  Level 2: 4-day cycles
  Level 3: 8-day cycles
  Level 4: 16-day cycles (~2 weeks)
  Level 5: 32-day cycles (~monthly)
  Level 6+: trend

# Returns
NamedTuple with trend, weekly_cycle, monthly_cycle, noise, energy_fractions
"""
function MultiResolutionAnalysis(
    pnl::AbstractVector{<:Real};
    fs_days::Int   = 1,   # sampling rate: 1 observation per day
    n_levels::Int  = 6,
    wavelet::Symbol = :haar
)
    x       = Float64.(pnl)
    n_orig  = length(x)
    x_pad, n_orig2 = _pad_pow2(x)
    n_pad   = length(x_pad)

    # Forward transform
    wt = wavelet == :d4 ?
        Daubechies4Transform(x_pad; levels=n_levels) :
        HaarTransform(x_pad; levels=n_levels)

    n_lv = wt.levels

    # Reconstruct each component by zeroing other levels
    function _recon_level(lvl_idx::Int)
        z_details = [zeros(length(d)) for d in wt.details]
        z_details[lvl_idx] = wt.details[lvl_idx]
        recon = if wavelet == :d4
            Daubechies4Inverse(zeros(length(wt.approx)), z_details)
        else
            HaarInverse(zeros(length(wt.approx)), z_details)
        end
        recon[1:n_orig]
    end

    # Reconstruct trend (approximation only, all details zeroed)
    trend_full = if wavelet == :d4
        Daubechies4Inverse(wt.approx, [zeros(length(d)) for d in wt.details])
    else
        HaarInverse(wt.approx, [zeros(length(d)) for d in wt.details])
    end
    trend = trend_full[1:n_orig]

    # Energy per level
    energy_levels = [sum(d.^2) for d in wt.details]
    energy_trend  = sum(wt.approx.^2)
    total_energy  = sum(energy_levels) + energy_trend
    energy_frac   = vcat(energy_levels ./ total_energy, energy_trend / total_energy)

    # Weekly cycle: levels 3 (8-day) and 4 (16-day) if enough levels
    weekly_cycle  = n_lv >= 3 ? _recon_level(3) : zeros(n_orig)
    monthly_cycle = n_lv >= 5 ? _recon_level(5) : zeros(n_orig)

    # Noise = original − all reconstructed components
    noise = x .- trend .- weekly_cycle .- monthly_cycle

    # Variance decomposition
    var_total   = var(x)
    var_trend   = var(trend)
    var_weekly  = var(weekly_cycle)
    var_monthly = var(monthly_cycle)
    var_noise   = var(noise)

    snr_db = var_noise > 1e-12 ?
        10.0 * log10(var_trend / var_noise) : 0.0

    (
        original        = x,
        trend           = trend,
        weekly_cycle    = weekly_cycle,
        monthly_cycle   = monthly_cycle,
        noise           = noise,
        energy_fraction = energy_frac,
        variance_trend  = var_trend / var_total,
        variance_weekly = var_weekly / var_total,
        variance_monthly= var_monthly / var_total,
        variance_noise  = var_noise / var_total,
        snr_db          = snr_db,
        n_levels        = n_lv,
        wavelet         = wavelet
    )
end

# ── Dominant Cycles ───────────────────────────────────────────────────────────

"""
Identify dominant cycles in a P&L series via wavelet energy analysis.

For each wavelet level, the scale corresponds to a frequency band.
Peak energy levels indicate dominant periodicities.

# Returns
NamedTuple: cycles sorted by energy (period_days, energy_fraction, label)
"""
function DominantCycles(
    pnl::AbstractVector{<:Real};
    fs_days::Int   = 1,
    n_levels::Int  = 7,
    wavelet::Symbol = :haar
)
    x      = Float64.(pnl)
    n_orig = length(x)
    n_orig < 32 && error("Need ≥ 32 data points for DominantCycles analysis")

    x_pad, _ = _pad_pow2(x)

    wt = wavelet == :d4 ?
        Daubechies4Transform(x_pad; levels=n_levels) :
        HaarTransform(x_pad; levels=n_levels)

    energy_levels  = [sum(d.^2) / length(d) for d in wt.details]
    total_energy   = sum(energy_levels)

    # Period (days) for each level: level k → 2^k day period
    cycles = map(enumerate(energy_levels)) do (k, e)
        period_days = 2^k / fs_days
        label = if period_days < 4
            "Noise ($(round(period_days; digits=1))d)"
        elseif period_days < 10
            "Weekly (~$(round(period_days; digits=0))d)"
        elseif period_days < 20
            "Biweekly (~$(round(period_days; digits=0))d)"
        elseif period_days < 45
            "Monthly (~$(round(period_days; digits=0))d)"
        elseif period_days < 100
            "Quarterly (~$(round(period_days; digits=0))d)"
        else
            "Long-term (~$(round(period_days; digits=0))d)"
        end
        (level=k, period_days=period_days, energy=e,
         energy_fraction=total_energy>0 ? e/total_energy : 0.0,
         label=label)
    end

    # Sort by energy descending
    sort!(cycles, by=c -> -c.energy)

    dominant = filter(c -> c.energy_fraction > 0.05, cycles)
    isempty(dominant) && (dominant = [cycles[1]])

    (
        all_cycles     = cycles,
        dominant_cycles = dominant,
        dominant_period_days = dominant[1].period_days,
        dominant_label  = dominant[1].label,
        n_levels        = wt.levels
    )
end

# ── Wavelet Coherence ─────────────────────────────────────────────────────────

"""
Estimate wavelet coherence between BTC price and strategy P&L.

Coherence at scale j: |Σ_t Wₓ(t,j) * conj(Wy(t,j))| / sqrt(|Wₓ|² * |Wy|²)
Value ∈ [0,1]; 1 = perfectly in phase, 0 = decorrelated.

Uses Haar transform; coherence computed per level (scale).

# Returns
NamedTuple: coherence_by_level, phase_lead_days, interpretation
"""
function WaveletCoherence(
    btc_returns::AbstractVector{<:Real},
    strategy_pnl::AbstractVector{<:Real};
    n_levels::Int = 6
)
    length(btc_returns) == length(strategy_pnl) ||
        error("btc_returns and strategy_pnl must have equal length")
    n = length(btc_returns)
    n < 32 && error("Need ≥ 32 observations for WaveletCoherence")

    x_pad, _ = _pad_pow2(Float64.(btc_returns))
    y_pad, _ = _pad_pow2(Float64.(strategy_pnl))
    n_pad    = min(length(x_pad), length(y_pad))
    x_pad    = x_pad[1:n_pad]
    y_pad    = y_pad[1:n_pad]

    wt_x = HaarTransform(x_pad; levels=n_levels)
    wt_y = HaarTransform(y_pad; levels=n_levels)

    coherence_levels = Float64[]
    phase_levels     = Float64[]

    for k in 1:min(length(wt_x.details), length(wt_y.details))
        dx = wt_x.details[k]
        dy = wt_y.details[k]
        len = min(length(dx), length(dy))

        # Cross-wavelet spectrum
        cws     = dot(dx[1:len], dy[1:len])
        power_x = sqrt(sum(dx[1:len].^2))
        power_y = sqrt(sum(dy[1:len].^2))

        coh = (power_x > 1e-12 && power_y > 1e-12) ?
            abs(cws) / (power_x * power_y) : 0.0
        push!(coherence_levels, min(coh, 1.0))

        # Phase: positive = BTC leads strategy
        phase = atan(sum(dx[1:len] .* dy[1:len]), sum(dx[1:len].^2))
        push!(phase_levels, phase)
    end

    period_days = [2^k for k in 1:length(coherence_levels)]

    # Average coherence weighted by period
    avg_coh = mean(coherence_levels)

    interp = if avg_coh > 0.7
        "HIGH coherence: strategy P&L strongly tracks BTC price movements"
    elseif avg_coh > 0.4
        "MODERATE coherence: partial co-movement with BTC; some decorrelation"
    else
        "LOW coherence: strategy largely independent of BTC price; good diversification"
    end

    (
        coherence_by_level   = coherence_levels,
        phase_by_level       = phase_levels,
        period_days          = period_days,
        average_coherence    = avg_coh,
        interpretation       = interp,
        btc_dominates_at_scales = period_days[coherence_levels .>= 0.6]
    )
end

# ── Scalogram ─────────────────────────────────────────────────────────────────

"""
Compute scalogram: time-frequency energy map of the P&L series.

Returns a 2D matrix (scale × time) of wavelet coefficient magnitudes squared.
Suitable for dashboard visualisation as a heat map.

# Returns
NamedTuple: scalogram_matrix (n_levels × n_time), time_axis, scale_axis_days
"""
function Scalogram(
    pnl::AbstractVector{<:Real};
    n_levels::Int  = 5,
    wavelet::Symbol = :haar
)
    x      = Float64.(pnl)
    n_orig = length(x)
    x_pad, _ = _pad_pow2(x)

    wt = wavelet == :d4 ?
        Daubechies4Transform(x_pad; levels=n_levels) :
        HaarTransform(x_pad; levels=n_levels)

    n_lv    = min(length(wt.details), n_levels)

    # For each level, upsample detail coefficients back to original length
    # to get a time-aligned representation
    scalogram = Matrix{Float64}(undef, n_lv, n_orig)

    for k in 1:n_lv
        d       = wt.details[k]
        d_abs2  = d.^2
        # Upsample by repeating each coefficient 2^k times
        repeat_factor = 2^k
        upsampled = Float64[]
        for v in d_abs2
            append!(upsampled, fill(v, repeat_factor))
        end
        # Trim or pad to n_orig
        if length(upsampled) >= n_orig
            scalogram[k, :] = upsampled[1:n_orig]
        else
            scalogram[k, :] = vcat(upsampled, zeros(n_orig - length(upsampled)))
        end
    end

    scale_days = [Float64(2^k) for k in 1:n_lv]

    (
        scalogram_matrix  = scalogram,
        n_levels          = n_lv,
        n_time            = n_orig,
        scale_axis_days   = scale_days,
        time_axis         = collect(1:n_orig),
        max_energy        = maximum(scalogram),
        normalized_scalogram = scalogram ./ max(maximum(scalogram), 1e-12)
    )
end

# ── Top-level driver ──────────────────────────────────────────────────────────

"""
Run full wavelet analysis pipeline.

# Arguments
- `pnl`         : daily strategy P&L vector
- `btc_returns` : daily BTC returns (same length)

Writes `wavelet_analysis_results.json` to `\$STATS_OUTPUT_DIR`.
"""
function run_wavelet_analysis(
    pnl::Vector{Float64},
    btc_returns::Vector{Float64};
    output_dir::String = get(ENV, "STATS_OUTPUT_DIR",
                              joinpath(@__DIR__, "..", "output"))
)
    println("[wavelet] Running multi-resolution analysis (Haar)...")
    mra_haar = MultiResolutionAnalysis(pnl; wavelet=:haar, n_levels=6)

    println("[wavelet] Running multi-resolution analysis (D4)...")
    mra_d4   = MultiResolutionAnalysis(pnl; wavelet=:d4,  n_levels=5)

    println("[wavelet] Identifying dominant cycles...")
    cycles   = DominantCycles(pnl; n_levels=7)

    println("[wavelet] Computing wavelet coherence with BTC...")
    coher    = WaveletCoherence(btc_returns, pnl; n_levels=6)

    println("[wavelet] Building scalogram...")
    scalo    = Scalogram(pnl; n_levels=5, wavelet=:haar)

    result = Dict(
        "multi_resolution_haar" => Dict(
            "variance_trend"   => mra_haar.variance_trend,
            "variance_weekly"  => mra_haar.variance_weekly,
            "variance_monthly" => mra_haar.variance_monthly,
            "variance_noise"   => mra_haar.variance_noise,
            "snr_db"           => mra_haar.snr_db,
            "trend"            => mra_haar.trend,
            "weekly_cycle"     => mra_haar.weekly_cycle,
            "monthly_cycle"    => mra_haar.monthly_cycle,
            "noise"            => mra_haar.noise
        ),
        "multi_resolution_d4" => Dict(
            "variance_trend"   => mra_d4.variance_trend,
            "variance_weekly"  => mra_d4.variance_weekly,
            "variance_monthly" => mra_d4.variance_monthly,
            "snr_db"           => mra_d4.snr_db
        ),
        "dominant_cycles" => map(c -> Dict(
            "period_days"      => c.period_days,
            "energy_fraction"  => c.energy_fraction,
            "label"            => c.label
        ), cycles.dominant_cycles),
        "dominant_period_days" => cycles.dominant_period_days,
        "wavelet_coherence" => Dict(
            "coherence_by_level"  => coher.coherence_by_level,
            "period_days"         => coher.period_days,
            "average_coherence"   => coher.average_coherence,
            "interpretation"      => coher.interpretation
        ),
        "scalogram" => Dict(
            "matrix"             => [scalo.scalogram_matrix[k, :] for k in 1:scalo.n_levels],
            "scale_axis_days"    => scalo.scale_axis_days,
            "n_levels"           => scalo.n_levels,
            "max_energy"         => scalo.max_energy
        )
    )

    mkpath(output_dir)
    out_path = joinpath(output_dir, "wavelet_analysis_results.json")
    open(out_path, "w") do io
        write(io, JSON3.write(result))
    end
    println("[wavelet] Results written to $out_path")

    result
end

end  # module WaveletAnalysis

# ── CLI self-test ─────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    using .WaveletAnalysis
    using Statistics

    println("[wavelet_analysis] Running self-test...")

    n = 256
    # Synthetic P&L: trend + weekly cycle + noise
    t = collect(1:n)
    pnl_synth = 0.005 .* t .+
                2.0 .* sin.(2π .* t ./ 7) .+   # weekly cycle
                0.8 .* sin.(2π .* t ./ 30) .+  # monthly cycle
                randn(n) .* 0.5

    btc_synth = 0.003 .* t .+ randn(n) .* 1.0

    result = run_wavelet_analysis(pnl_synth, btc_synth)

    mra = result["multi_resolution_haar"]
    println("  Trend explains $(round(mra["variance_trend"]*100; digits=1))% of variance")
    println("  Weekly cycle: $(round(mra["variance_weekly"]*100; digits=1))%")
    println("  Noise: $(round(mra["variance_noise"]*100; digits=1))%")
    println("  SNR: $(round(mra["snr_db"]; digits=2)) dB")

    dom = result["dominant_cycles"]
    println("  Dominant cycle: $(dom[1]["label"]) (energy=$(round(dom[1]["energy_fraction"]*100; digits=1))%)")

    coh = result["wavelet_coherence"]
    println("  BTC coherence: $(round(coh["average_coherence"]; digits=3)) — $(coh["interpretation"])")

    println("[wavelet_analysis] Self-test complete.")
end
