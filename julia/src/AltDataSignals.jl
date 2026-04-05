"""
AltDataSignals.jl — Advanced alternative data signal construction

Extends the existing AlternativeData.jl with deeper signal engineering:
  - WebTrafficSignal: trend momentum with adaptive lookback
  - OptionsMarketSignal: PCR, skew, VIX term structure signals
  - OnChainWhaleDetector: large tx classification, flow impact model
  - FuturesTermStructure: contango/backwardation carry + roll yield
  - AltDataComposite: PCA, IC², EWMA-weighted combination
  - Rolling IC monitoring and signal performance attribution
"""
module AltDataSignals

using Statistics, LinearAlgebra, Random

export WebTrafficSignal, OptionsMarketSignal, OnChainWhaleDetector,
       FuturesTermStructure, AltDataComposite, SignalICTracker

export fit!, update!, predict, compute_ic, rolling_ic_series,
       normalized, composite_weights, signal_half_life,
       detect_whale_events, flow_score, carry_signal, term_slope_signal,
       pca_components, ic_squared_weights, ewma_ic_weights,
       tracker_sharpe, tracker_decay, synthetic_altdata

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Z-score of a vector, optionally using rolling window."""
function normalized(x::AbstractVector{Float64}; window::Int=0)
    n = length(x)
    if window <= 0 || window >= n
        valid = filter(!isnan, x)
        mu = isempty(valid) ? 0.0 : mean(valid)
        sg = isempty(valid) ? 1.0 : std(valid)
        return (x .- mu) ./ (sg + 1e-8)
    end
    out = fill(NaN, n)
    for t in window:n
        w = filter(!isnan, x[max(1,t-window+1):t])
        if length(w) < 3; continue; end
        out[t] = (x[t] - mean(w)) / (std(w) + 1e-8)
    end
    return out
end

"""Compute IC between signal (at t) and return (at t+lag)."""
function compute_ic(signal::Vector{Float64}, returns::Vector{Float64}; lag::Int=1)
    n = min(length(signal), length(returns)) - lag
    if n < 10; return NaN; end
    s = signal[1:n]
    r = returns[lag+1:lag+n]
    valid = .!isnan.(s) .& .!isnan.(r)
    sum(valid) < 10 && return NaN
    c = cor(s[valid], r[valid])
    return isnan(c) ? NaN : c
end

"""Rolling IC over sliding window."""
function rolling_ic_series(signal::Vector{Float64}, returns::Vector{Float64};
                            window::Int=60, lag::Int=1)
    n = min(length(signal), length(returns))
    ics = fill(NaN, n)
    for t in (window+lag):n
        s = signal[t-window+1:t-lag]
        r = returns[t-window+lag+1:t]
        valid = .!isnan.(s) .& .!isnan.(r)
        sum(valid) < 10 && continue
        c = cor(s[valid], r[valid])
        ics[t] = isnan(c) ? NaN : c
    end
    return ics
end

"""Exponentially weighted moving average."""
function _ewma(x::Vector{Float64}, halflife::Float64)
    alpha = 1 - exp(-log(2) / max(halflife, 0.1))
    result = fill(NaN, length(x))
    ema = NaN
    for (t, val) in enumerate(x)
        isnan(val) && continue
        ema = isnan(ema) ? val : alpha * val + (1-alpha) * ema
        result[t] = ema
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. WebTrafficSignal
# ─────────────────────────────────────────────────────────────────────────────

"""
WebTrafficSignal models search interest momentum as a crypto leading indicator.

Captures the phenomenon where retail search volume surges 1-5 days before
price moves driven by attention effects. Uses a short-term vs long-term
baseline comparison with adaptive smoothing.

Fields:
  lookback_short  : short momentum window (days)
  lookback_long   : long baseline window (days)
  smooth_hl       : EWMA smoothing half-life
  trend_data      : stored raw search interest [0,100]
  signal_values   : computed signal values
"""
mutable struct WebTrafficSignal
    lookback_short::Int
    lookback_long::Int
    smooth_hl::Float64
    trend_data::Vector{Float64}
    signal_values::Vector{Float64}
    ic_log::Vector{Float64}   # running IC log
end

function WebTrafficSignal(; lookback_short::Int=7, lookback_long::Int=28,
                            smooth_hl::Float64=3.0)
    return WebTrafficSignal(lookback_short, lookback_long, smooth_hl,
                             Float64[], Float64[], Float64[])
end

function _web_signal_raw(wts::WebTrafficSignal)
    n = length(wts.trend_data)
    sig = fill(NaN, n)
    lb_s, lb_l = wts.lookback_short, wts.lookback_long
    for t in (lb_l+1):n
        short_win = wts.trend_data[max(1,t-lb_s+1):t]
        long_win  = wts.trend_data[max(1,t-lb_l+1):t]
        sig[t] = (mean(short_win) - mean(long_win)) / (std(long_win) + 1e-8)
    end
    return _ewma(sig, wts.smooth_hl)
end

"""Fit the web traffic signal on historical trend data."""
function fit!(wts::WebTrafficSignal, trend_data::Vector{Float64},
               returns::Vector{Float64}=Float64[])
    append!(wts.trend_data, trend_data)
    wts.signal_values = _web_signal_raw(wts)
    if !isempty(returns)
        ic = compute_ic(wts.signal_values[1:min(end,length(returns))], returns)
        push!(wts.ic_log, isnan(ic) ? 0.0 : ic)
    end
    return wts
end

"""Update with a single new observation."""
function update!(wts::WebTrafficSignal, new_trend::Float64,
                  realized_return::Union{Float64,Nothing}=nothing)
    push!(wts.trend_data, new_trend)
    wts.signal_values = _web_signal_raw(wts)
    if !isnothing(realized_return) && !isempty(wts.signal_values)
        ic_sign = wts.signal_values[end] * realized_return > 0 ? 1.0 : -1.0
        push!(wts.ic_log, ic_sign)
    end
end

"""Current signal value."""
predict(wts::WebTrafficSignal) = isempty(wts.signal_values) ? NaN : wts.signal_values[end]

"""Signal half-life: number of periods before IC decays to half."""
function signal_half_life(wts::WebTrafficSignal; min_obs::Int=30)
    ics = filter(!isnan, wts.ic_log)
    n = length(ics)
    n < min_obs && return NaN
    half = n ÷ 2
    early = mean(ics[1:half])
    late  = mean(ics[half+1:end])
    (early <= 0 || late <= 0) && return NaN
    return n * log(2) / (log(early) - log(late) + 1e-10)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. OptionsMarketSignal
# ─────────────────────────────────────────────────────────────────────────────

"""
OptionsMarketSignal extracts three types of signals from the crypto options market:

1. PCR (Put/Call Ratio): Contrarian — when fear (puts) dominates, mean reversion likely.
2. Skew: 25-delta put IV minus call IV. High skew = fear. Contrarian signal.
3. Term Structure: IV slope across maturities. Inverted = stress = reduce risk.

Each signal can be used standalone or combined.
"""
mutable struct OptionsMarketSignal
    pcr_lb::Int                   # lookback for PCR z-score
    skew_lb::Int                  # lookback for skew normalization
    tenors_days::Vector{Int}      # available tenors (days)
    pcr_history::Vector{Float64}
    atm_iv_history::Matrix{Float64}   # n_obs × n_tenors
    skew_history::Vector{Float64}
    n_obs::Int
end

function OptionsMarketSignal(; pcr_lb::Int=10, skew_lb::Int=20,
                               tenors::Vector{Int}=[7,14,30,60,90,180])
    return OptionsMarketSignal(pcr_lb, skew_lb, tenors,
                                Float64[], zeros(0, length(tenors)),
                                Float64[], 0)
end

"""Ingest one observation of options data."""
function update!(ops::OptionsMarketSignal, pcr::Float64,
                  atm_ivs::Vector{Float64}, skew::Float64)
    push!(ops.pcr_history, pcr)
    push!(ops.skew_history, skew)
    n_t = length(ops.tenors_days)
    row = length(atm_ivs) >= n_t ? atm_ivs[1:n_t] : vcat(atm_ivs, fill(NaN, n_t-length(atm_ivs)))
    if ops.n_obs == 0
        ops.atm_iv_history = reshape(row, 1, n_t)
    else
        ops.atm_iv_history = vcat(ops.atm_iv_history, reshape(row, 1, n_t))
    end
    ops.n_obs += 1
end

"""PCR contrarian signal: high PCR (fear) → buy signal."""
function pcr_signal(ops::OptionsMarketSignal)
    isempty(ops.pcr_history) && return Float64[]
    # Negative because contrarian
    return -normalized(ops.pcr_history; window=ops.pcr_lb)
end

"""Skew contrarian signal."""
function skew_signal(ops::OptionsMarketSignal)
    isempty(ops.skew_history) && return Float64[]
    return -normalized(ops.skew_history; window=ops.skew_lb)
end

"""
Term structure signal: slope of IV curve.
Positive slope (normal): calm → neutral/long
Inverted (front > back): stress → reduce risk
"""
function term_slope_signal(ops::OptionsMarketSignal; front_idx::Int=1, back_idx::Int=3)
    ops.n_obs < 5 && return Float64[]
    n_t = length(ops.tenors_days)
    (front_idx > n_t || back_idx > n_t) && return fill(NaN, ops.n_obs)
    slope = ops.atm_iv_history[:, back_idx] .- ops.atm_iv_history[:, front_idx]
    # Positive slope = normal = bullish
    return normalized(slope; window=20)
end

"""Combined options signal: weighted average of PCR + skew + term."""
function combined_options_signal(ops::OptionsMarketSignal;
                                   w_pcr::Float64=0.40,
                                   w_skew::Float64=0.30,
                                   w_term::Float64=0.30)
    pcr = pcr_signal(ops)
    sk  = skew_signal(ops)
    ts  = term_slope_signal(ops)
    n   = min(length(pcr), length(sk), length(ts))
    n == 0 && return Float64[]

    return w_pcr .* coalesce.(pcr[1:n], 0.0) .+
           w_skew .* coalesce.(sk[1:n], 0.0) .+
           w_term .* coalesce.(ts[1:n], 0.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. OnChainWhaleDetector
# ─────────────────────────────────────────────────────────────────────────────

"""A single on-chain transaction record."""
struct WhaleEvent
    day_idx::Int
    address::String
    amount_usd::Float64
    direction::Int            # +1 = exchange outflow (accumulating), -1 = inflow (selling)
    asset::String
end

"""
OnChainWhaleDetector identifies large wallet movements and measures their
impact on short-term price dynamics.

Algorithm:
1. Filter transactions above threshold (default \$1M).
2. Classify as exchange inflow (sell pressure) vs outflow (buy pressure).
3. Aggregate net flow with exponential decay weighting.
4. Score is proportional to log-size × direction.
"""
mutable struct OnChainWhaleDetector
    threshold_usd::Float64      # minimum USD to qualify
    decay_halflife::Float64     # signal decay in days
    event_log::Vector{WhaleEvent}
    daily_flow::Vector{Float64} # daily net flow score
    n_days::Int
end

function OnChainWhaleDetector(; threshold::Float64=1e6, decay_halflife::Float64=2.0)
    return OnChainWhaleDetector(threshold, decay_halflife, WhaleEvent[], Float64[], 0)
end

"""
Add raw transactions for one day; returns whale events.
transactions: vector of (address, amount_usd, is_outflow::Bool, asset) tuples.
"""
function detect_whale_events(wdet::OnChainWhaleDetector,
                               transactions::Vector{<:NamedTuple};
                               day::Int=1)
    events = WhaleEvent[]
    for tx in transactions
        tx.amount_usd < wdet.threshold_usd && continue
        dir = tx.is_outflow ? 1 : -1
        ev = WhaleEvent(day, tx.address, tx.amount_usd, dir, tx.asset)
        push!(events, ev)
        push!(wdet.event_log, ev)
    end
    return events
end

"""
Compute daily flow score from a list of WhaleEvents on a given day.
Score = Σ direction_i × log(size_i / threshold).
"""
function flow_score(wdet::OnChainWhaleDetector, events::Vector{WhaleEvent})
    isempty(events) && return 0.0
    return sum(ev.direction * log(ev.amount_usd / wdet.threshold_usd) for ev in events)
end

"""
Update daily flow signal with today's events.
"""
function update!(wdet::OnChainWhaleDetector, events::Vector{WhaleEvent})
    score = flow_score(wdet, events)
    push!(wdet.daily_flow, score)
    wdet.n_days += 1
end

"""
Aggregate whale signal over recent window with exponential decay.
"""
function predict(wdet::OnChainWhaleDetector; window::Int=5)
    isempty(wdet.daily_flow) && return 0.0
    n = length(wdet.daily_flow)
    w = window == 0 ? n : min(window, n)
    recent = wdet.daily_flow[end-w+1:end]
    weights = [exp(-log(2)/wdet.decay_halflife * (w-i)) for i in 1:w]
    weights ./= sum(weights)
    return dot(weights, recent)
end

"""
Estimate the permanent price impact of a given flow score.
Using Kyle's lambda model: impact ∝ order_size / market_depth.
"""
function price_impact(wdet::OnChainWhaleDetector, flow_usd::Float64;
                       market_depth_usd::Float64=5e9, permanent_frac::Float64=0.4)
    lambda = permanent_frac / market_depth_usd
    return lambda * flow_usd  # fractional price move
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. FuturesTermStructure
# ─────────────────────────────────────────────────────────────────────────────

"""
FuturesTermStructure models the shape of the futures curve and extracts
carry and slope signals.

- Contango (F > S, basis > 0): cost of carry is negative for long holders;
  indicates strong demand for leveraged longs.
- Backwardation (F < S, basis < 0): positive carry for longs;
  often signals supply overhang or short-term spot demand.

Signals:
  carry_signal     : -basis_short (positive = backwardation = bullish)
  slope_signal     : basis_long - basis_short (positive = normal = calm)
  roll_yield       : annualized roll benefit from front to back
"""
mutable struct FuturesTermStructure
    tenors_days::Vector{Int}
    basis_history::Matrix{Float64}    # n_obs × n_tenors (annualized basis)
    carry_signal_hist::Vector{Float64}
    slope_signal_hist::Vector{Float64}
    roll_yield_hist::Vector{Float64}
    n_obs::Int
end

function FuturesTermStructure(; tenors::Vector{Int}=[7, 30, 90, 180])
    return FuturesTermStructure(tenors, zeros(0, length(tenors)),
                                 Float64[], Float64[], Float64[], 0)
end

"""
Ingest new observation: spot price + futures prices for each tenor.
Computes annualized basis for each tenor.
"""
function update!(fts::FuturesTermStructure, spot::Float64,
                  futures_prices::Vector{Float64})
    n_t = length(fts.tenors_days)
    basis = Float64[]
    for (i, T_d) in enumerate(fts.tenors_days)
        i > length(futures_prices) && break
        F = futures_prices[i]
        T = T_d / 365.0
        push!(basis, (F - spot) / (spot * T + 1e-10))  # annualized
    end

    # Pad to n_t
    while length(basis) < n_t; push!(basis, NaN); end
    row = basis[1:n_t]

    if fts.n_obs == 0
        fts.basis_history = reshape(row, 1, n_t)
    else
        fts.basis_history = vcat(fts.basis_history, reshape(row, 1, n_t))
    end
    fts.n_obs += 1

    # Carry signal: negative short-term basis → backwardation → positive signal
    push!(fts.carry_signal_hist, isnan(row[1]) ? 0.0 : -row[1])

    # Slope: back - front (positive = normal contango curve)
    slope = n_t >= 3 ? (row[3] - row[1]) : NaN
    push!(fts.slope_signal_hist, isnan(slope) ? 0.0 : slope)

    # Roll yield: annualized benefit of rolling from front to next tenor
    if n_t >= 2 && !isnan(row[1]) && !isnan(row[2])
        roll = row[1] - row[2]  # basis drops as you move further out
        push!(fts.roll_yield_hist, roll)
    else
        push!(fts.roll_yield_hist, 0.0)
    end
end

"""Return the current carry signal (z-scored)."""
function carry_signal(fts::FuturesTermStructure; window::Int=20)
    isempty(fts.carry_signal_hist) && return NaN
    z = normalized(fts.carry_signal_hist; window=window)
    return z[end]
end

"""Return current term slope signal (z-scored)."""
function term_slope_signal(fts::FuturesTermStructure; window::Int=20)
    isempty(fts.slope_signal_hist) && return NaN
    z = normalized(fts.slope_signal_hist; window=window)
    return z[end]
end

"""Classify term structure regime."""
function classify_regime(fts::FuturesTermStructure)
    fts.n_obs < 5 && return :unknown
    avg = mean(fts.carry_signal_hist[max(1,end-4):end])
    abs(avg) < 0.01 && return :neutral
    avg > 0.01 && return :backwardation
    return :contango
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. AltDataComposite
# ─────────────────────────────────────────────────────────────────────────────

"""
AltDataComposite combines multiple alternative data signals into a single
composite predictor using one of three methods:
  :equal       — equal weighting
  :ic_squared  — weights proportional to IC² (Markowitz for signals)
  :pca         — projects onto first PCA component

Provides rolling performance tracking and adaptive weight updates.
"""
mutable struct AltDataComposite
    method::Symbol
    signal_names::Vector{String}
    weights::Vector{Float64}
    pca_loadings::Vector{Float64}   # first PC
    ic_history::Matrix{Float64}     # n_updates × n_signals
    composite_hist::Vector{Float64}
    n_updates::Int
    lookback_ic::Int
end

function AltDataComposite(names::Vector{String};
                            method::Symbol=:ic_squared, lookback_ic::Int=60)
    n = length(names)
    return AltDataComposite(method, names, fill(1.0/n, n),
                             fill(1.0/n, n), zeros(0, n),
                             Float64[], 0, lookback_ic)
end

"""
Fit weights from a matrix of signal values and forward returns.
signals: T × n_signals; returns: length-T vector.
"""
function fit!(comp::AltDataComposite, signals::Matrix{Float64},
               returns::Vector{Float64})
    n_sig = size(signals, 2)

    if comp.method == :equal
        comp.weights = fill(1.0/n_sig, n_sig)

    elseif comp.method == :ic_squared
        ics = [compute_ic(signals[:,j], returns) for j in 1:n_sig]
        ics = [isnan(x) ? 0.0 : x for x in ics]
        ic2 = max.(ics, 0.0).^2
        total = sum(ic2)
        comp.weights = total > 1e-10 ? ic2 ./ total : fill(1.0/n_sig, n_sig)
        comp.ic_history = vcat(comp.ic_history, ics')

    elseif comp.method == :pca
        valid_mask = .!any(isnan.(signals), dims=2)[:, 1]
        if sum(valid_mask) < 10
            comp.weights = fill(1.0/n_sig, n_sig)
        else
            S = signals[valid_mask, :]
            S_c = S .- mean(S, dims=1)
            C = (S_c' * S_c) ./ max(sum(valid_mask)-1, 1)
            evals = eigvals(C)
            evecs = eigvecs(C)
            idx = sortperm(evals, rev=true)
            first_pc = abs.(evecs[:, idx[1]])
            comp.pca_loadings = first_pc
            comp.weights = first_pc ./ max(sum(first_pc), 1e-10)
        end
    end

    comp.n_updates += 1
    return comp
end

"""
Compute composite from a vector of current signal values.
"""
function predict(comp::AltDataComposite, signal_vals::Vector{Float64})
    n = min(length(signal_vals), length(comp.weights))
    return dot(comp.weights[1:n], [isnan(v) ? 0.0 : v for v in signal_vals[1:n]])
end

"""
Compute IC²-based weights from a signals matrix and returns.
Pure function (no mutation).
"""
function ic_squared_weights(signals::Matrix{Float64}, returns::Vector{Float64})
    n_sig = size(signals, 2)
    ics = Float64[]
    for j in 1:n_sig
        ic = compute_ic(signals[:,j], returns)
        push!(ics, isnan(ic) ? 0.0 : ic)
    end
    ic2 = max.(ics, 0.0).^2
    total = sum(ic2)
    weights = total > 1e-10 ? ic2 ./ total : fill(1.0/n_sig, n_sig)
    return (weights=weights, ics=ics, ic2=ic2)
end

"""
Extract PCA components from signals matrix.
Returns scores, loadings, and explained variance by component.
"""
function pca_components(signals::Matrix{Float64}; n_components::Int=3)
    n_obs, n_sig = size(signals)
    # Replace NaNs with column means
    S = copy(signals)
    for j in 1:n_sig
        col_mean = mean(filter(!isnan, S[:, j]))
        S[:, j] = [isnan(x) ? col_mean : x for x in S[:, j]]
    end
    S_c = S .- mean(S, dims=1)
    C = (S_c' * S_c) ./ max(n_obs-1, 1)
    evals = eigvals(C)
    evecs = eigvecs(C)
    idx = sortperm(evals, rev=true)
    k = min(n_components, n_sig)
    loadings = evecs[:, idx[1:k]]
    scores = S_c * loadings
    total_var = sum(evals)
    explained = [evals[idx[i]] / total_var for i in 1:k]
    return (scores=scores, loadings=loadings, explained_var=explained)
end

"""
EWMA IC-weighted combination: adapts signal weights over rolling IC history.
"""
function ewma_ic_weights(signals::Matrix{Float64}, returns::Vector{Float64};
                          window::Int=60, halflife::Float64=30.0)
    n_obs, n_sig = size(signals)
    composite = fill(NaN, n_obs)
    alpha = 1 - exp(-log(2)/halflife)
    ema_ic2 = fill(1.0/n_sig, n_sig)

    for t in (window+1):n_obs
        # Rolling IC for each signal over past `window` observations
        ics_t = Float64[]
        for j in 1:n_sig
            ic = compute_ic(signals[max(1,t-window):t-1, j], returns[max(1,t-window):t-1])
            push!(ics_t, isnan(ic) ? 0.0 : max(ic, 0.0))
        end
        ic2_t = ics_t.^2
        # EMA update
        ema_ic2 = alpha .* ic2_t .+ (1-alpha) .* ema_ic2
        total = sum(ema_ic2)
        weights = total > 1e-10 ? ema_ic2 ./ total : fill(1.0/n_sig, n_sig)
        sig_row = [isnan(signals[t,j]) ? 0.0 : signals[t,j] for j in 1:n_sig]
        composite[t] = dot(weights, sig_row)
    end
    return composite
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. SignalICTracker
# ─────────────────────────────────────────────────────────────────────────────

"""
SignalICTracker: live monitoring of signal IC, Sharpe, and decay.

Tracks:
  - Rolling hit rate (sign accuracy)
  - EWMA IC
  - Strategy P&L (simple long/short based on signal threshold)
  - Decay detection: is IC declining over time?
"""
mutable struct SignalICTracker
    name::String
    threshold::Float64        # signal threshold for taking positions
    tcost::Float64            # round-trip transaction cost (fraction)
    ic_window::Int
    signal_log::Vector{Float64}
    return_log::Vector{Float64}
    pnl_log::Vector{Float64}
    position_log::Vector{Float64}
    n_obs::Int
end

function SignalICTracker(name::String; threshold::Float64=0.5,
                          tcost::Float64=0.001, ic_window::Int=60)
    return SignalICTracker(name, threshold, tcost, ic_window,
                            Float64[], Float64[], Float64[], Float64[], 0)
end

"""Add one observation."""
function update!(tracker::SignalICTracker, signal::Float64, ret::Float64)
    push!(tracker.signal_log, signal)
    push!(tracker.return_log, ret)
    tracker.n_obs += 1

    pos = 0.0
    if !isnan(signal)
        if signal > tracker.threshold; pos = 1.0
        elseif signal < -tracker.threshold; pos = -1.0
        end
    end
    prev_pos = isempty(tracker.position_log) ? 0.0 : tracker.position_log[end]
    trade = abs(pos - prev_pos)
    pnl = pos * ret - trade * tracker.tcost

    push!(tracker.pnl_log, pnl)
    push!(tracker.position_log, pos)
end

"""Annualized Sharpe ratio over most recent `window` observations."""
function tracker_sharpe(tracker::SignalICTracker; window::Int=0)
    pnl = tracker.pnl_log
    isempty(pnl) && return NaN
    w = window == 0 ? length(pnl) : min(window, length(pnl))
    r = pnl[end-w+1:end]
    ann_ret = mean(r) * 252
    ann_vol = std(r) * sqrt(252)
    return ann_vol > 1e-10 ? ann_ret / ann_vol : NaN
end

"""Detect IC decay. Returns (half_life_days, is_decaying)."""
function tracker_decay(tracker::SignalICTracker; split_pct::Float64=0.5)
    n = tracker.n_obs
    n < 20 && return (half_life=NaN, decaying=false)
    split = round(Int, n * split_pct)
    s1 = tracker.signal_log[1:split]
    r1 = tracker.return_log[1:split]
    s2 = tracker.signal_log[split+1:end]
    r2 = tracker.return_log[split+1:end]
    ic1 = compute_ic(s1, r1)
    ic2 = compute_ic(s2, r2)
    (isnan(ic1) || isnan(ic2) || ic1 <= 0) && return (half_life=NaN, decaying=false)

    decaying = ic2 < ic1 * 0.7
    half_life = decaying ? (n/2) * log(2) / log(ic1 / max(ic2, 1e-4)) : NaN
    return (half_life=half_life, decaying=decaying, ic_early=ic1, ic_late=ic2)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Synthetic data generator and end-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────

"""
Generate a synthetic alt-data dataset for testing.
Returns NamedTuple with returns, trend, pcr, whale_flow, basis, skew, atm_iv.
"""
function synthetic_altdata(n::Int=600; seed::Int=42)
    rng = MersenneTwister(seed)
    returns = randn(rng, n) .* 0.025 .+ 0.0003

    # Trend: 3-day lead on price
    trend = 50.0 .+ 20.0 .* vcat(zeros(3), returns[1:end-3]) ./ 0.025 .+
            randn(rng, n) .* 10.0
    trend = clamp.(trend, 0.0, 100.0)

    # PCR: contrarian (negative correlation with +1 day return)
    pcr = 1.0 .+ 0.5 .* (-returns) ./ 0.025 .+ randn(rng, n) .* 0.2
    pcr = max.(0.3, pcr)

    # Whale flow: informed, +1 day lead
    whale = randn(rng, n) .* 0.5 .+ vcat(0.0, returns[1:end-1]) ./ 0.025 .* 0.4

    # Basis: OU process with momentum driven by returns
    basis = zeros(n)
    basis[1] = 0.001
    for t in 2:n
        basis[t] = basis[t-1] * 0.9 + 0.0003 * returns[t] + randn(rng) * 0.0005
    end

    # Skew: fear indicator, rises when returns are negative
    skew = 0.05 .- returns ./ 0.025 .* 0.02 .+ randn(rng, n) .* 0.01
    skew = max.(0.0, skew)

    # ATM IV (3 tenors): front end more volatile
    atm_iv_1w  = 0.80 .+ returns ./ 0.025 .* (-0.05) .+ randn(rng, n) .* 0.03
    atm_iv_1m  = 0.75 .+ randn(rng, n) .* 0.02
    atm_iv_3m  = 0.70 .+ randn(rng, n) .* 0.015
    atm_iv = hcat(atm_iv_1w, atm_iv_1m, atm_iv_3m)

    return (returns=returns, trend=trend, pcr=pcr, whale=whale,
            basis=basis, skew=skew, atm_iv=atm_iv)
end

"""
Run a complete alt-data signal pipeline on a dataset.
Returns IC for each signal and composite.
"""
function run_altdata_pipeline(data::NamedTuple; verbose::Bool=true)
    n = length(data.returns)
    fwd = vcat(data.returns[2:end], [NaN])

    # 1. Web traffic
    wts = WebTrafficSignal()
    fit!(wts, data.trend, data.returns)
    ic_trend = compute_ic(wts.signal_values, fwd)

    # 2. PCR signal
    pcr_sig = -normalized(data.pcr; window=10)
    ic_pcr = compute_ic(pcr_sig, fwd)

    # 3. Whale flow
    whale_sig = normalized(data.whale; window=20)
    ic_whale = compute_ic(whale_sig, fwd)

    # 4. Carry (basis)
    carry_sig = normalized(-data.basis; window=20)
    ic_carry = compute_ic(carry_sig, fwd)

    # 5. Skew (contrarian)
    skew_sig = normalized(-data.skew; window=20)
    ic_skew = compute_ic(skew_sig, fwd)

    # 6. Composite
    all_sigs = hcat(coalesce.(wts.signal_values, 0.0),
                     coalesce.(pcr_sig, 0.0),
                     coalesce.(whale_sig, 0.0),
                     coalesce.(carry_sig, 0.0),
                     coalesce.(skew_sig, 0.0))
    w = ic_squared_weights(all_sigs[1:n-1,:], fwd[1:n-1])
    composite = all_sigs * w.weights
    ic_composite = compute_ic(composite, fwd)

    if verbose
        println("=== AltDataSignals Pipeline ===")
        println("  Web Trend IC:  $(round(ic_trend, digits=5))")
        println("  PCR IC:        $(round(ic_pcr, digits=5))")
        println("  Whale Flow IC: $(round(ic_whale, digits=5))")
        println("  Carry IC:      $(round(ic_carry, digits=5))")
        println("  Skew IC:       $(round(ic_skew, digits=5))")
        println("  Composite IC:  $(round(ic_composite, digits=5))")
        println("  IC²-Weights: trend=$(round(w.weights[1],digits=3)), " *
                "pcr=$(round(w.weights[2],digits=3)), whale=$(round(w.weights[3],digits=3)), " *
                "carry=$(round(w.weights[4],digits=3)), skew=$(round(w.weights[5],digits=3))")
    end

    return (ic_trend=ic_trend, ic_pcr=ic_pcr, ic_whale=ic_whale,
            ic_carry=ic_carry, ic_skew=ic_skew, ic_composite=ic_composite,
            weights=w.weights, composite=composite)
end


# ── Extension: Additional Alt-Data Infrastructure ────────────────────────────

"""
    SentimentScoreSignal

NLP-based sentiment scoring from news headlines and social media.
Uses a simple bag-of-words model with configurable positive/negative lexicons.
"""
mutable struct SentimentScoreSignal
    pos_lexicon::Vector{String}
    neg_lexicon::Vector{String}
    decay::Float64
    score_ewma::Float64
    history::Vector{Float64}
    count::Int
end

function SentimentScoreSignal(decay=0.85)
    pos_words = ["bullish", "surge", "rally", "breakout", "adoption", "institutional",
                 "upgrade", "partnership", "launch", "record", "growth", "positive",
                 "buy", "strong", "gain", "rise", "moon", "accumulate", "hodl", "green"]
    neg_words = ["bearish", "crash", "dump", "hack", "exploit", "ban", "regulation",
                 "investigation", "fraud", "lawsuit", "loss", "decline", "fear", "sell",
                 "weak", "drop", "collapse", "scam", "rug", "red", "capitulate"]
    return SentimentScoreSignal(pos_words, neg_words, decay, 0.0, Float64[], 0)
end

function score_headline(sig::SentimentScoreSignal, text::String)
    words = split(lowercase(text), r"[^a-z]+")
    pos_count = sum(w in sig.pos_lexicon for w in words)
    neg_count = sum(w in sig.neg_lexicon for w in words)
    total = pos_count + neg_count
    total == 0 && return 0.0
    return (pos_count - neg_count) / total
end

function update!(sig::SentimentScoreSignal, text::String)
    raw = score_headline(sig, text)
    if sig.count == 0
        sig.score_ewma = raw
    else
        sig.score_ewma = sig.decay * sig.score_ewma + (1 - sig.decay) * raw
    end
    push!(sig.history, raw)
    sig.count += 1
    return sig
end

predict(sig::SentimentScoreSignal) = sig.score_ewma

function batch_score(sig::SentimentScoreSignal, texts::Vector{String})
    isempty(texts) && return 0.0
    return mean(score_headline(sig, t) for t in texts)
end

"""
    GithubActivitySignal

Developer activity tracking: commit velocity and star growth as leading
indicators of protocol health and adoption momentum.
"""
mutable struct GithubActivitySignal
    alpha::Float64
    beta::Float64
    commit_ewma::Float64
    star_ewma::Float64
    history::Vector{NamedTuple}
    count::Int
end

GithubActivitySignal(alpha=0.3, beta=0.2) = GithubActivitySignal(alpha, beta, 0.0, 0.0, [], 0)

function update!(sig::GithubActivitySignal, commits_today::Int, stars_today::Int)
    if sig.count == 0
        sig.commit_ewma = commits_today
        sig.star_ewma   = stars_today
    else
        sig.commit_ewma = sig.alpha * commits_today + (1 - sig.alpha) * sig.commit_ewma
        sig.star_ewma   = sig.beta  * stars_today   + (1 - sig.beta)  * sig.star_ewma
    end
    push!(sig.history, (commits=commits_today, stars=stars_today))
    sig.count += 1
end

function dev_activity_score(sig::GithubActivitySignal)
    sig.count < 2 && return 0.0
    recent = min(sig.count, 30)
    hist = sig.history[(end-recent+1):end]
    avg_commits = mean(h.commits for h in hist)
    avg_stars   = mean(h.stars   for h in hist)
    avg_commits == 0 && return 0.0
    commit_score = (sig.commit_ewma - avg_commits) / max(avg_commits, 1)
    star_score   = (sig.star_ewma   - avg_stars)   / max(avg_stars, 1)
    return 0.6 * commit_score + 0.4 * star_score
end

"""
    ExchangeFlowSignal

Net exchange inflow/outflow tracking as a proxy for selling pressure
(positive net = inflows = bearish) or accumulation (negative = bullish).
"""
mutable struct ExchangeFlowSignal
    window::Int
    net_flows::Vector{Float64}
    ewma_flow::Float64
    smoothing::Float64
end

ExchangeFlowSignal(window=14, smoothing=0.2) = ExchangeFlowSignal(window, Float64[], 0.0, smoothing)

function update!(sig::ExchangeFlowSignal, inflow::Float64, outflow::Float64)
    net = inflow - outflow
    push!(sig.net_flows, net)
    if length(sig.net_flows) == 1
        sig.ewma_flow = net
    else
        sig.ewma_flow = sig.smoothing * net + (1 - sig.smoothing) * sig.ewma_flow
    end
    return sig
end

function exchange_pressure_signal(sig::ExchangeFlowSignal)
    length(sig.net_flows) < sig.window && return 0.0
    recent = sig.net_flows[(end-sig.window+1):end]
    avg_abs = mean(abs.(recent))
    avg_abs == 0 && return 0.0
    return clamp(sig.ewma_flow / avg_abs, -1.0, 1.0)
end

function rolling_exchange_ic(sig::ExchangeFlowSignal, forward_returns::Vector{Float64}, window::Int)
    n = min(length(sig.net_flows), length(forward_returns))
    n < window && return Float64[]
    ics = Float64[]
    for t in window:n
        s_w = sig.net_flows[(t-window+1):t]
        r_w = forward_returns[(t-window+1):t]
        push!(ics, cor(s_w, r_w))
    end
    return ics
end

"""
    AltDataPortfolio

Meta-learner that combines multiple alt-data signals with adaptive
weight allocation via online gradient ascent on IC.
"""
mutable struct AltDataPortfolio
    signals::Vector{Any}
    signal_names::Vector{String}
    weights::Vector{Float64}
    ic_history::Vector{Vector{Float64}}
    learning_rate::Float64
    regularization::Float64
end

function AltDataPortfolio(signals, names; lr=0.01, reg=0.01)
    n = length(signals)
    AltDataPortfolio(signals, names, ones(n)/n, [Float64[] for _ in 1:n], lr, reg)
end

function portfolio_signal(port::AltDataPortfolio, predictions::Vector{Float64})
    return dot(port.weights, predictions)
end

function update_weights!(port::AltDataPortfolio, predictions::Vector{Float64}, forward_return::Float64)
    n = length(port.weights)
    grad = predictions .* forward_return
    port.weights .+= port.learning_rate .* grad
    port.weights .-= port.regularization .* port.weights
    port.weights = max.(port.weights, 0)
    total = sum(port.weights)
    total > 0 && (port.weights ./= total)
    return port
end

function portfolio_ic_stats(port::AltDataPortfolio)
    stats = []
    for (i, name) in enumerate(port.signal_names)
        hist = port.ic_history[i]
        length(hist) >= 5 || continue
        push!(stats, (
            name    = name,
            mean_ic = mean(hist),
            std_ic  = std(hist),
            ir      = mean(hist) / max(std(hist), 1e-10),
            weight  = port.weights[i],
        ))
    end
    return stats
end

"""
    synthetic_sentiment_data(n)

Generate synthetic news headline data with known sentiment polarity for testing.
"""
function synthetic_sentiment_data(n=100)
    pos_texts = [
        "BTC surges to new all-time high on institutional adoption",
        "Ethereum upgrade boosts ecosystem growth and developer activity",
        "Major bank announces crypto custody services in record move",
        "DeFi protocol launches with strong early traction and partnership",
    ]
    neg_texts = [
        "Crypto exchange hacked millions in losses reported",
        "Regulatory ban on crypto trading announced by government",
        "Bitcoin drops sharply amid market sell-off and fear",
        "DeFi exploit drains protocol funds in major fraud incident",
    ]
    neutral_texts = [
        "Crypto market sees mixed trading volume today",
        "Bitcoin price consolidates near key support level",
        "Ethereum developers release new technical specification",
    ]
    texts   = String[]
    signals = Float64[]
    for _ in 1:n
        r = rand()
        if r < 0.35
            push!(texts, rand(pos_texts));    push!(signals,  1.0)
        elseif r < 0.65
            push!(texts, rand(neg_texts));    push!(signals, -1.0)
        else
            push!(texts, rand(neutral_texts)); push!(signals,  0.0)
        end
    end
    return texts, signals
end

"""
    run_sentiment_pipeline(n_obs)

Run a complete sentiment signal test pipeline with IC evaluation.
"""
function run_sentiment_pipeline(n_obs=200)
    sig = SentimentScoreSignal(0.90)
    texts, true_signals = synthetic_sentiment_data(n_obs)
    predictions = Float64[]
    for text in texts
        update!(sig, text)
        push!(predictions, predict(sig))
    end
    n_eval = min(length(predictions), length(true_signals)) - 5
    ic = n_eval > 0 ? cor(predictions[6:(n_eval+5)], true_signals[6:(n_eval+5)]) : NaN
    return (signal=sig, predictions=predictions, true_signals=true_signals, ic=ic)
end

export SentimentScoreSignal, GithubActivitySignal, ExchangeFlowSignal, AltDataPortfolio
export score_headline, batch_score, dev_activity_score
export exchange_pressure_signal, rolling_exchange_ic
export portfolio_signal, update_weights!, portfolio_ic_stats
export synthetic_sentiment_data, run_sentiment_pipeline

end  # module AltDataSignals
