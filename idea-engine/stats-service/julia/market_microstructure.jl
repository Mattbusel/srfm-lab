"""
market_microstructure.jl

Advanced market microstructure analytics:
  - Roll spread estimator
  - Kyle's lambda (price impact)
  - Amihud illiquidity
  - Intraday U-shape (volume and volatility by hour)
  - Tick-by-tick signature plot (realized variance vs sampling frequency)
  - Optimal sampling frequency for realized variance
  - Per-symbol microstructure report (JSON export)
"""

using Statistics
using LinearAlgebra
using JSON3
using Dates

# ─── Data Structures ─────────────────────────────────────────────────────────

"""Tick-level trade record."""
struct Tick
    ts::DateTime
    price::Float64
    volume::Float64       # in base units (e.g. BTC)
    side::Symbol          # :buy or :sell
end

"""OHLCV bar."""
struct Bar
    ts::DateTime
    open::Float64
    high::Float64
    low::Float64
    close::Float64
    volume::Float64
end

"""Per-symbol microstructure report."""
struct MicrostructureReport
    symbol::String
    roll_spread_bps::Float64
    kyle_lambda::Float64            # price impact per unit volume
    amihud_ratio::Float64           # daily avg
    amihud_series::Vector{Float64}  # per-day time series
    intraday_volume::Vector{Float64} # by hour 0-23
    intraday_volatility::Vector{Float64}
    signature_frequencies::Vector{Float64}  # samples per day
    signature_rv::Vector{Float64}           # realized variance at each freq
    optimal_sampling_freq::Float64          # samples per day
    noise_variance::Float64
    computed_at::String
end

# ─── Roll Spread Estimator ────────────────────────────────────────────────────

"""
    roll_spread(prices; min_obs=30)

Estimate the effective bid-ask spread using the Roll (1984) model.

The model assumes:
    Δp_t = c·q_t + u_t
where c = half-spread, q_t ∈ {-1,+1} is trade direction (unobserved).

Cov(Δp_t, Δp_{t-1}) = -c²

Roll spread = 2·sqrt(-Cov(Δp_t, Δp_{t-1}))

Returns spread in same units as prices. Returns NaN if covariance is positive
(indicates the model assumptions are violated, e.g. trending market).
"""
function roll_spread(prices::Vector{Float64}; min_obs::Int=30)::Float64
    n = length(prices)
    n >= min_obs + 1 || return NaN

    dp = diff(prices)
    n_dp = length(dp)
    n_dp < 2 && return NaN

    # Compute first-order autocovariance of price changes
    dp_mean = mean(dp)
    cov_lag1 = sum((dp[i] - dp_mean) * (dp[i-1] - dp_mean) for i in 2:n_dp) / (n_dp - 1)

    # Roll model: spread = 2 * sqrt(-cov)
    cov_lag1 >= 0 && return 0.0   # positive autocov → no spread estimate
    return 2.0 * sqrt(-cov_lag1)
end

"""
    roll_spread_bps(prices; min_obs=30)

Roll spread expressed in basis points relative to mean price.
"""
function roll_spread_bps(prices::Vector{Float64}; min_obs::Int=30)::Float64
    s = roll_spread(prices; min_obs)
    isnan(s) && return NaN
    midprice = mean(prices)
    midprice == 0 && return NaN
    return (s / midprice) * 10_000
end

"""
    roll_spread_rolling(prices, window; min_obs=30)

Rolling window Roll spread estimation. Returns vector of spread estimates.
"""
function roll_spread_rolling(prices::Vector{Float64}, window::Int; min_obs::Int=30)::Vector{Float64}
    n = length(prices)
    out = fill(NaN, n)
    for i in window:n
        window_prices = prices[max(1, i-window+1):i]
        out[i] = roll_spread_bps(window_prices; min_obs)
    end
    return out
end

# ─── Kyle's Lambda ────────────────────────────────────────────────────────────

"""
    kyle_lambda(price_changes, signed_volumes; min_obs=30)

Estimate Kyle's lambda: the price impact per unit of signed order flow.

Model: Δp_t = λ · x_t + ε_t
where x_t = signed volume (positive = buy, negative = sell)

Estimated via OLS: λ = Cov(Δp, x) / Var(x)

A higher lambda means the market is less liquid (more price impact per unit vol).

Returns lambda in price units per volume unit.
"""
function kyle_lambda(
    price_changes::Vector{Float64},
    signed_volumes::Vector{Float64};
    min_obs::Int=30
)::Float64
    n = length(price_changes)
    n == length(signed_volumes) || throw(ArgumentError("lengths must match"))
    n >= min_obs || return NaN

    # OLS: β = (X'X)^{-1} X'y
    x = signed_volumes
    y = price_changes
    x_mean = mean(x)
    y_mean = mean(y)

    var_x = sum((xi - x_mean)^2 for xi in x)
    var_x < 1e-12 && return NaN

    cov_xy = sum((x[i] - x_mean) * (y[i] - y_mean) for i in 1:n)
    λ = cov_xy / var_x
    return λ
end

"""
    kyle_lambda_robust(price_changes, signed_volumes; min_obs=30, trim=0.01)

Robust Kyle lambda estimation with winsorization to handle outliers.
"""
function kyle_lambda_robust(
    price_changes::Vector{Float64},
    signed_volumes::Vector{Float64};
    min_obs::Int=30,
    trim::Float64=0.01
)::Float64
    n = length(price_changes)
    n >= min_obs || return NaN

    # Winsorize
    dp_lo, dp_hi = quantile(price_changes, [trim, 1-trim])
    sv_lo, sv_hi = quantile(signed_volumes, [trim, 1-trim])

    dp_w = clamp.(price_changes, dp_lo, dp_hi)
    sv_w = clamp.(signed_volumes, sv_lo, sv_hi)

    return kyle_lambda(dp_w, sv_w; min_obs)
end

"""
    kyle_lambda_ols_full(price_changes, signed_volumes)

Full OLS with R² and t-stat. Returns (lambda, r_squared, t_stat).
"""
function kyle_lambda_ols_full(
    price_changes::Vector{Float64},
    signed_volumes::Vector{Float64}
)::NamedTuple
    n = length(price_changes)
    n >= 10 || return (lambda=NaN, r_squared=NaN, t_stat=NaN, se=NaN)

    x = signed_volumes .- mean(signed_volumes)
    y = price_changes .- mean(price_changes)

    ss_xx = sum(xi^2 for xi in x)
    ss_xx < 1e-12 && return (lambda=NaN, r_squared=NaN, t_stat=NaN, se=NaN)

    λ = sum(x[i]*y[i] for i in 1:n) / ss_xx
    y_hat = λ .* x
    residuals = y .- y_hat
    ss_res = sum(r^2 for r in residuals)
    ss_tot = sum(yi^2 for yi in y)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    se = sqrt(ss_res / (n - 2)) / sqrt(ss_xx)
    t_stat = λ / max(se, 1e-12)

    return (lambda=λ, r_squared=r2, t_stat=t_stat, se=se)
end

# ─── Amihud Illiquidity ───────────────────────────────────────────────────────

"""
    amihud_ratio(bars; scale=1e6)

Amihud (2002) illiquidity ratio: daily |return| / dollar volume.

ILLIQ_d = |R_d| / (P_d · V_d)

Higher ILLIQ → lower liquidity (each dollar of volume moves the price more).

scale: multiply by this factor for readability (default 1e6 = per million USD).
Returns per-day time series.
"""
function amihud_ratio(bars::Vector{Bar}; scale::Float64=1e6)::Vector{Float64}
    out = Float64[]
    for bar in bars
        dollar_vol = bar.close * bar.volume
        dollar_vol < 1.0 && push!(out, NaN) && continue
        abs_return = abs(bar.close - bar.open) / max(bar.open, 1e-10)
        push!(out, (abs_return / dollar_vol) * scale)
    end
    return out
end

"""
    amihud_moving_average(bars, window; scale=1e6)

Rolling mean Amihud ratio over a given window of days.
"""
function amihud_moving_average(bars::Vector{Bar}, window::Int; scale::Float64=1e6)::Vector{Float64}
    daily = amihud_ratio(bars; scale)
    n = length(daily)
    out = fill(NaN, n)
    for i in window:n
        vals = filter(!isnan, daily[max(1, i-window+1):i])
        isempty(vals) || (out[i] = mean(vals))
    end
    return out
end

# ─── Intraday U-Shape ─────────────────────────────────────────────────────────

"""
    intraday_volume_profile(ticks)

Compute mean volume traded per hour-of-day across all ticks.
Returns a 24-element vector (hours 0-23).
"""
function intraday_volume_profile(ticks::Vector{Tick})::Vector{Float64}
    counts = zeros(Float64, 24)
    totals = zeros(Float64, 24)
    for tick in ticks
        h = hour(tick.ts)
        totals[h+1] += tick.volume
        counts[h+1] += 1.0
    end
    # Return mean volume per tick in each hour (handles unequal day counts)
    return [counts[h] > 0 ? totals[h] / counts[h] : 0.0 for h in 1:24]
end

"""
    intraday_volatility_profile(bars_1min)

Compute mean absolute return per minute aggregated by hour-of-day.
Expects 1-minute bars. Returns a 24-element vector.
"""
function intraday_volatility_profile(bars_1min::Vector{Bar})::Vector{Float64}
    totals = zeros(Float64, 24)
    counts = zeros(Int, 24)
    for bar in bars_1min
        h = hour(bar.ts)
        bar.open > 0 || continue
        abs_ret = abs(bar.close - bar.open) / bar.open
        totals[h+1] += abs_ret
        counts[h+1] += 1
    end
    return [counts[h] > 0 ? totals[h] / counts[h] : 0.0 for h in 1:24]
end

"""
    u_shape_score(profile)

Measure how U-shaped a profile is: ratio of (avg endpoints) / (avg middle).
Values > 1 indicate a U-shape (elevated at open/close).
"""
function u_shape_score(profile::Vector{Float64})::Float64
    length(profile) < 8 && return NaN
    # Use "trading hours" heuristic for crypto: all hours, but open/close peaks
    # morning session: hours 0-3, afternoon: hours 12-15, night: hours 20-23
    endpoint_hours = vcat(1:4, 21:24)
    middle_hours = 8:16
    avg_ends = mean(profile[endpoint_hours])
    avg_mid = mean(profile[middle_hours])
    avg_mid < 1e-10 && return NaN
    return avg_ends / avg_mid
end

# ─── Signature Plot ───────────────────────────────────────────────────────────

"""
    realized_variance(prices, n_subsamples)

Compute realized variance (sum of squared returns) at a given sampling frequency.
prices: tick-level price series (high frequency).
n_subsamples: number of observations to use (sub-samples from the full series).

Returns the realized variance (annualized if desired by caller).
"""
function realized_variance(prices::Vector{Float64}, n_subsamples::Int)::Float64
    n = length(prices)
    n_subsamples >= 2 || return NaN
    n_subsamples > n && return NaN

    # Evenly spaced subsampling
    step = (n - 1) / (n_subsamples - 1)
    indices = [max(1, round(Int, 1 + (i-1) * step)) for i in 1:n_subsamples]
    unique!(sort!(indices))
    length(indices) < 2 && return NaN

    subprices = prices[indices]
    returns = diff(log.(max.(subprices, 1e-10)))
    return sum(r^2 for r in returns)
end

"""
    signature_plot(prices; freq_grid=nothing)

Compute the signature plot: realized variance as a function of sampling frequency.
This reveals whether microstructure noise is present:
  - At high freq: RV inflated by bid-ask bounce (noise dominates)
  - At low freq:  RV approaches true variance

Returns (frequencies, realized_variances).
The optimal sampling frequency is where the curve "bends" (second derivative changes sign).
"""
function signature_plot(
    prices::Vector{Float64};
    freq_grid::Union{Nothing, Vector{Int}} = nothing
)::Tuple{Vector{Int}, Vector{Float64}}
    n = length(prices)

    if isnothing(freq_grid)
        # Default: log-spaced from 5 to n/2
        n_points = min(40, n ÷ 2)
        freq_grid = unique(round.(Int, 10 .^ range(log10(5), log10(n ÷ 2), length=n_points)))
    end

    rvs = [realized_variance(prices, k) for k in freq_grid]
    return (freq_grid, rvs)
end

# ─── Optimal Sampling Frequency ───────────────────────────────────────────────

"""
    estimate_noise_variance(prices; n_high_freq=500)

Estimate microstructure noise variance using the high-frequency limit of the
signature plot. As Δt → 0, RV → 2nσ_ε² where n is the number of observations
and σ_ε² is the per-observation noise variance.
"""
function estimate_noise_variance(prices::Vector{Float64}; n_high_freq::Int=500)::Float64
    n = length(prices)
    n_sample = min(n_high_freq, n)
    rv_high = realized_variance(prices, n_sample)
    isnan(rv_high) && return NaN
    # Each tick contributes 2·σ_ε² to RV at high freq
    return rv_high / (2 * n_sample)
end

"""
    optimal_sampling_frequency(prices)

Estimate the optimal sampling frequency for realized variance estimation using
the Bandi-Russell (2008) approach.

The optimal frequency minimizes MSE of the RV estimator:
    MSE = (2·n·σ_ε²)² / n + (IV/n)²  (simplified)

Optimal n* ≈ (IV / (4·σ_ε²))^(2/3)

where IV = integrated variance (approximated by low-frequency RV).

Returns the optimal number of samples per period.
"""
function optimal_sampling_frequency(prices::Vector{Float64})::NamedTuple
    n = length(prices)
    n < 20 && return (optimal_n=NaN, noise_var=NaN, iv_estimate=NaN)

    # Estimate noise variance from highest frequency
    σ_ε² = estimate_noise_variance(prices; n_high_freq=min(1000, n))
    isnan(σ_ε²) && return (optimal_n=NaN, noise_var=NaN, iv_estimate=NaN)

    # Estimate IV from low-frequency (use ~20 observations)
    n_low = min(20, n ÷ 5)
    iv = realized_variance(prices, n_low)
    isnan(iv) && return (optimal_n=NaN, noise_var=σ_ε², iv_estimate=NaN)

    # Bandi-Russell formula
    if σ_ε² < 1e-15
        return (optimal_n=n, noise_var=σ_ε², iv_estimate=iv)
    end
    n_star = (iv / (4.0 * σ_ε²))^(2/3)
    n_star = clamp(round(Int, n_star), 5, n)

    return (optimal_n=n_star, noise_var=σ_ε², iv_estimate=iv)
end

"""
    find_signature_kink(freqs, rvs)

Find the kink point in the signature plot where the slope changes from
steeply decreasing (noise-dominated) to flat (noise-free).
This is done by finding the minimum curvature point.
"""
function find_signature_kink(freqs::Vector{Int}, rvs::Vector{Float64})::Int
    valid = .!isnan.(rvs)
    sum(valid) < 4 && return freqs[1]

    f = freqs[valid]
    r = rvs[valid]
    n = length(f)

    # Normalize
    f_norm = (f .- minimum(f)) ./ (maximum(f) - minimum(f) + 1e-10)
    r_norm = (r .- minimum(r)) ./ (maximum(r) - minimum(r) + 1e-10)

    # Second derivative (curvature) via finite differences
    curvatures = Float64[]
    for i in 2:n-1
        d1 = (r_norm[i] - r_norm[i-1]) / (f_norm[i] - f_norm[i-1] + 1e-10)
        d2_num = (r_norm[i+1] - r_norm[i]) / (f_norm[i+1] - f_norm[i] + 1e-10) - d1
        d2_den = (f_norm[i+1] - f_norm[i-1]) / 2.0 + 1e-10
        push!(curvatures, abs(d2_num / d2_den))
    end

    kink_idx = argmax(curvatures) + 1   # +1 for offset from finite diff
    return f[min(kink_idx, length(f))]
end

# ─── Full Report ──────────────────────────────────────────────────────────────

"""
    compute_microstructure_report(symbol, ticks, bars_daily, bars_1min)

Compute a full microstructure report for a symbol.

Arguments:
  - symbol: ticker string
  - ticks: tick-level trade data
  - bars_daily: daily OHLCV bars
  - bars_1min: 1-minute OHLCV bars

Returns a MicrostructureReport struct.
"""
function compute_microstructure_report(
    symbol::String,
    ticks::Vector{Tick},
    bars_daily::Vector{Bar},
    bars_1min::Vector{Bar}
)::MicrostructureReport
    # Prices from ticks
    tick_prices = [t.price for t in ticks]

    # Roll spread
    rs_bps = isempty(tick_prices) ? NaN : roll_spread_bps(tick_prices)

    # Kyle's lambda
    if length(ticks) >= 30
        dp = diff(tick_prices)
        signed_vols = [(t.side == :buy ? 1.0 : -1.0) * t.volume for t in ticks[2:end]]
        kl = kyle_lambda_robust(dp, signed_vols)
    else
        kl = NaN
    end

    # Amihud ratio
    amihud_daily = amihud_ratio(bars_daily)
    amihud_avg = mean(filter(!isnan, amihud_daily))

    # Intraday profiles
    hourly_vol = intraday_volume_profile(ticks)
    hourly_vola = intraday_volatility_profile(bars_1min)

    # Signature plot
    freqs, rvs = signature_plot(tick_prices)
    opt = optimal_sampling_frequency(tick_prices)

    MicrostructureReport(
        symbol,
        rs_bps,
        kl,
        amihud_avg,
        amihud_daily,
        hourly_vol,
        hourly_vola,
        Float64.(freqs),
        rvs,
        Float64(opt.optimal_n),
        opt.noise_var,
        string(now(UTC))
    )
end

"""
    report_to_dict(r)

Convert a MicrostructureReport to a plain Dict for JSON serialization.
"""
function report_to_dict(r::MicrostructureReport)::Dict
    Dict(
        "symbol"                => r.symbol,
        "roll_spread_bps"       => isnan(r.roll_spread_bps) ? nothing : r.roll_spread_bps,
        "kyle_lambda"           => isnan(r.kyle_lambda) ? nothing : r.kyle_lambda,
        "amihud_ratio_avg"      => isnan(r.amihud_ratio) ? nothing : r.amihud_ratio,
        "amihud_series"         => r.amihud_series,
        "intraday_volume"       => r.intraday_volume,
        "intraday_volatility"   => r.intraday_volatility,
        "signature_frequencies" => r.signature_frequencies,
        "signature_rv"          => r.signature_rv,
        "optimal_sampling_freq" => isnan(r.optimal_sampling_freq) ? nothing : r.optimal_sampling_freq,
        "noise_variance"        => isnan(r.noise_variance) ? nothing : r.noise_variance,
        "computed_at"           => r.computed_at,
    )
end

"""
    export_report_json(report, filepath)

Write a MicrostructureReport to a JSON file.
"""
function export_report_json(report::MicrostructureReport, filepath::String)
    open(filepath, "w") do io
        JSON3.write(io, report_to_dict(report))
    end
    @info "Microstructure report written to $filepath"
end

"""
    export_reports_json(reports, filepath)

Write multiple reports (one per symbol) to a JSON array.
"""
function export_reports_json(reports::Vector{MicrostructureReport}, filepath::String)
    dicts = [report_to_dict(r) for r in reports]
    open(filepath, "w") do io
        JSON3.write(io, dicts)
    end
    @info "$(length(reports)) microstructure reports written to $filepath"
end

# ─── Synthetic Demo ───────────────────────────────────────────────────────────

"""
    synthetic_ticks(n; spread_bps=2.0, lambda=0.001, seed=42)

Generate synthetic tick data for testing. Simulates a random walk with
microstructure noise (bid-ask bounce) and order flow imbalance.
"""
function synthetic_ticks(n::Int; spread_bps::Float64=2.0, lambda::Float64=0.001, seed::Int=42)::Vector{Tick}
    rng = nothing  # would use Random.seed!(seed) in practice
    price = 50_000.0
    half_spread = price * spread_bps / 20_000.0
    ticks = Tick[]
    t0 = DateTime(2024, 1, 1, 0, 0, 0)

    for i in 1:n
        dt = Second(round(Int, 86_400 / n))
        ts = t0 + dt * (i - 1)

        side = rand() > 0.48 ? :buy : :sell
        vol = rand() * 2.0 + 0.01

        # Kyle price impact
        impact = lambda * vol * (side == :buy ? 1.0 : -1.0)
        # Random walk
        price += impact + randn() * 5.0
        # Observed price has bid-ask noise
        observed = price + (side == :buy ? half_spread : -half_spread)

        push!(ticks, Tick(ts, max(1.0, observed), vol, side))
    end
    return ticks
end

"""
    synthetic_daily_bars(n; seed=42)

Generate n synthetic daily OHLCV bars.
"""
function synthetic_daily_bars(n::Int; seed::Int=42)::Vector{Bar}
    price = 50_000.0
    t0 = DateTime(2024, 1, 1)
    bars = Bar[]
    for i in 1:n
        vol_daily = 1_000_000.0 + rand() * 500_000
        ret = randn() * 0.02
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(randn()) * 0.005)
        l = min(o, c) * (1 - abs(randn()) * 0.005)
        push!(bars, Bar(t0 + Day(i-1), o, h, l, c, vol_daily))
        price = c
    end
    return bars
end

# ─── Example / Entry Point ────────────────────────────────────────────────────

function run_microstructure_demo()
    @info "Running microstructure analytics demo..."

    symbols = ["BTC", "ETH", "SOL"]
    reports = MicrostructureReport[]

    for sym in symbols
        @info "Processing $sym..."
        ticks = synthetic_ticks(5_000; spread_bps=sym == "BTC" ? 0.8 : sym == "ETH" ? 1.2 : 2.5)
        daily_bars = synthetic_daily_bars(90)
        min_bars = Bar[]  # would be populated in production
        report = compute_microstructure_report(sym, ticks, daily_bars, min_bars)
        push!(reports, report)

        @info "  Roll spread: $(round(report.roll_spread_bps, digits=2)) bps"
        @info "  Kyle lambda: $(round(report.kyle_lambda, sigdigits=4))"
        @info "  Amihud (avg): $(round(report.amihud_ratio, sigdigits=4))"
        @info "  Optimal sampling: $(round(report.optimal_sampling_freq)) ticks"
        @info "  Noise variance: $(round(report.noise_variance, sigdigits=4))"
    end

    outfile = joinpath(@__DIR__, "microstructure_reports.json")
    export_reports_json(reports, outfile)
    return reports
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_microstructure_demo()
end
