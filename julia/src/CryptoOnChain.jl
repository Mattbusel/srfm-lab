"""
CryptoOnChain.jl — On-Chain Analytics for Crypto Valuation & Signal Generation

Covers:
  - MVRV-Z score: market value vs realized value
  - SOPR (Spent Output Profit Ratio)
  - NVT (Network Value to Transactions) ratio
  - Puell Multiple: miner revenue vs historical average
  - Stock-to-Flow model (S2F and S2FX cross-asset)
  - Exchange net flow signals (accumulation/distribution)
  - HODL waves: age distribution proxy
  - Hash ribbon: miner capitulation indicator
  - Composite on-chain signal with learned/fixed weights
  - Backtesting on-chain signals against price history

Pure Julia stdlib only. No external dependencies.
"""
module CryptoOnChain

using Statistics, LinearAlgebra, Random

export MVRVSignal, compute_mvrv_z, mvrv_signal
export SOPRSignal, compute_sopr, sopr_signal
export NVTSignal, compute_nvt, nvt_signal
export PuellMultiple, compute_puell, puell_signal
export StockToFlow, s2f_model, s2fx_model, s2f_predicted_price
export ExchangeFlowSignal, exchange_flow_signal
export HODLWaveSignal, hodl_wave_proxy
export HashRibbonSignal, hash_ribbon_signal, miner_capitulation
export OnChainComposite, composite_signal, fit_composite_weights!
export backtest_onchain_signal, compare_onchain_signals
export simulate_onchain_data, run_onchain_demo

# ─────────────────────────────────────────────────────────────
# 1. MVRV-Z SCORE
# ─────────────────────────────────────────────────────────────

"""
    MVRVSignal

MVRV (Market Value to Realized Value) Z-Score.

The MVRV ratio = Market Cap / Realized Cap
  - Realized cap weights each coin by the price when it last moved
  - MVRV > 3.7 historically marks cycle tops
  - MVRV < 1.0 historically marks cycle bottoms
  - Z-score normalizes: (MV - RV) / σ(MV - RV)
"""
struct MVRVSignal
    window::Int  # lookback for std normalization
end

MVRVSignal() = MVRVSignal(365)

"""
    compute_mvrv_z(market_cap, realized_cap, window) -> Vector{Float64}

Compute MVRV Z-score time series.
"""
function compute_mvrv_z(market_cap::Vector{Float64},
                          realized_cap::Vector{Float64},
                          window::Int=365)::Vector{Float64}
    n = length(market_cap)
    mvrv_diff = market_cap .- realized_cap
    z = zeros(n)
    for t in window:n
        seg = mvrv_diff[t-window+1:t]
        mu  = mean(seg)
        sig = std(seg)
        sig < 1e-10 && continue
        z[t] = (mvrv_diff[t] - mu) / sig
    end
    z
end

"""
    mvrv_signal(market_cap, realized_cap; window=365, overbought=2.0, oversold=-0.5)
       -> NamedTuple

Generate trading signal from MVRV Z-score.
  +1 = accumulate (cheap), 0 = neutral, -1 = distribute (expensive)
"""
function mvrv_signal(market_cap::Vector{Float64},
                      realized_cap::Vector{Float64};
                      window::Int=365,
                      overbought::Float64=2.0,
                      oversold::Float64=-0.5)
    z      = compute_mvrv_z(market_cap, realized_cap, window)
    ratio  = market_cap ./ (realized_cap .+ 1e-10)
    signal = zeros(length(z))
    for (i, zi) in enumerate(z)
        if zi < oversold;     signal[i] =  1.0  # buy zone
        elseif zi > overbought; signal[i] = -1.0  # sell zone
        end
    end
    (z_score=z, ratio=ratio, signal=signal,
     overbought_threshold=overbought, oversold_threshold=oversold)
end

# ─────────────────────────────────────────────────────────────
# 2. SOPR — SPENT OUTPUT PROFIT RATIO
# ─────────────────────────────────────────────────────────────

"""
    SOPRSignal

SOPR = realized value of all coins moved today / cost basis of those coins.
SOPR > 1: coins being sold at profit (holders are profitable)
SOPR < 1: coins being sold at loss (capitulation signal → buy)
SOPR = 1: support/resistance (holders at breakeven)

Adjusted SOPR (aSOPR) removes short-term holders (< 1hr outputs).
"""
struct SOPRSignal
    smoothing::Int   # EMA window for smoothing
end

SOPRSignal() = SOPRSignal(7)

"""
    compute_sopr(realized_prices, cost_basis) -> Vector{Float64}

Compute raw SOPR from realized prices and cost basis of moved coins.
"""
function compute_sopr(realized_prices::Vector{Float64},
                       cost_basis::Vector{Float64})::Vector{Float64}
    max.(realized_prices ./ (cost_basis .+ 1e-10), 1e-10)
end

"""
    sopr_signal(sopr; smoothing=7) -> NamedTuple

Generate trading signal from SOPR.
  buy  signal when SOPR dips below 1 and recovers (capitulation)
  sell signal when SOPR is well above 1 (euphoria)
"""
function sopr_signal(sopr::Vector{Float64}; smoothing::Int=7)
    n = length(sopr)
    # Exponential smoothing
    alpha  = 2.0 / (smoothing + 1)
    sopr_s = similar(sopr); sopr_s[1] = sopr[1]
    for t in 2:n; sopr_s[t] = alpha * sopr[t] + (1-alpha) * sopr_s[t-1]; end

    signal = zeros(n)
    for t in 2:n
        if sopr_s[t] < 1.0 && sopr_s[t] > sopr_s[t-1]  # recovering from below 1
            signal[t] = 1.0   # bullish: capitulation ending
        elseif sopr_s[t] > 1.1
            signal[t] = -0.5  # mild bearish: profit-taking zone
        elseif sopr_s[t] > 1.3
            signal[t] = -1.0  # strong bearish: euphoria
        end
    end
    (sopr_raw=sopr, sopr_smooth=sopr_s, signal=signal)
end

# ─────────────────────────────────────────────────────────────
# 3. NVT RATIO
# ─────────────────────────────────────────────────────────────

"""
    NVTSignal

NVT (Network Value to Transactions) Ratio.
NVT = Market Cap / Daily On-Chain Volume (USD)
Analogous to P/E ratio: high NVT = overvalued relative to usage.
NVT Signal uses a 90-day MA of transaction volume.
"""
struct NVTSignal
    volume_window::Int   # for NVT Signal (smoothed volume)
    signal_window::Int   # for NVT Signal MA
end

NVTSignal() = NVTSignal(90, 14)

"""
    compute_nvt(market_cap, tx_volume; volume_window=90) -> Vector{Float64}

Compute NVT ratio (raw) and NVT Signal.
"""
function compute_nvt(market_cap::Vector{Float64},
                      tx_volume::Vector{Float64};
                      volume_window::Int=90,
                      signal_window::Int=14)
    n = length(market_cap)
    nvt_raw    = zeros(n)
    nvt_signal = zeros(n)

    # NVT = market cap / tx volume
    for t in 1:n
        nvt_raw[t] = market_cap[t] / max(tx_volume[t], 1.0)
    end

    # NVT Signal = market cap / MA(tx volume)
    for t in volume_window:n
        avg_vol = mean(tx_volume[t-volume_window+1:t])
        nvt_signal[t] = market_cap[t] / max(avg_vol, 1.0)
    end

    (nvt_raw=nvt_raw, nvt_signal=nvt_signal)
end

"""
    nvt_signal(market_cap, tx_volume; thresholds=(65, 100)) -> NamedTuple

Generate trading signal from NVT.
"""
function nvt_signal(market_cap::Vector{Float64},
                     tx_volume::Vector{Float64};
                     thresholds::Tuple{Float64,Float64}=(65.0, 100.0))
    nvt = compute_nvt(market_cap, tx_volume)
    n   = length(nvt.nvt_signal)
    signal = zeros(n)
    for t in 1:n
        nvts = nvt.nvt_signal[t]
        if nvts > thresholds[2]; signal[t] = -1.0   # overvalued
        elseif nvts < thresholds[1]; signal[t] = 1.0 # undervalued
        end
    end
    (nvt_raw=nvt.nvt_raw, nvt_signal_series=nvt.nvt_signal, signal=signal,
     thresholds=thresholds)
end

# ─────────────────────────────────────────────────────────────
# 4. PUELL MULTIPLE
# ─────────────────────────────────────────────────────────────

"""
    PuellMultiple

Puell Multiple = Daily miner revenue (USD) / 365-day MA of daily revenue.
Captures miner capitulation and extreme undervaluation/overvaluation.
  > 4: historically extreme tops (miners very profitable)
  < 0.5: historically extreme bottoms (miners barely breaking even)
"""
struct PuellMultiple
    window::Int   # MA window (default 365)
end

PuellMultiple() = PuellMultiple(365)

"""
    compute_puell(daily_miner_revenue; window=365) -> Vector{Float64}
"""
function compute_puell(daily_miner_revenue::Vector{Float64};
                        window::Int=365)::Vector{Float64}
    n = length(daily_miner_revenue)
    puell = zeros(n)
    for t in window:n
        ma = mean(daily_miner_revenue[t-window+1:t])
        puell[t] = daily_miner_revenue[t] / max(ma, 1.0)
    end
    puell
end

"""
    puell_signal(daily_miner_revenue; buy=0.5, sell=4.0, window=365) -> NamedTuple
"""
function puell_signal(daily_miner_revenue::Vector{Float64};
                       buy::Float64=0.5, sell::Float64=4.0, window::Int=365)
    puell = compute_puell(daily_miner_revenue; window=window)
    n     = length(puell)
    signal = zeros(n)
    for t in 1:n
        puell[t] > 1e-10 || continue
        if puell[t] < buy;  signal[t] =  1.0  # extreme bottom
        elseif puell[t] > sell; signal[t] = -1.0  # extreme top
        end
    end
    (puell=puell, signal=signal, buy_threshold=buy, sell_threshold=sell)
end

# ─────────────────────────────────────────────────────────────
# 5. STOCK-TO-FLOW MODEL
# ─────────────────────────────────────────────────────────────

"""
    StockToFlow

Stock-to-Flow model for Bitcoin valuation.
S2F = Stock (total circulating supply) / Flow (annual new supply mined)
ln(Market Cap) = a * ln(S2F) + b  (linear in log-log space)

S2FX (cross-asset model) adds gold and silver as data points.
"""
struct StockToFlow
    a::Float64  # regression coefficient
    b::Float64  # intercept
end

StockToFlow() = StockToFlow(3.31819, 14.6227)  # PlanB's original estimates

"""
    s2f_model(supply, annual_issuance) -> Float64

Compute Stock-to-Flow ratio and predicted market cap.
"""
function s2f_model(m::StockToFlow, supply::Float64, annual_issuance::Float64)::Float64
    s2f = supply / max(annual_issuance, 1.0)
    exp(m.a * log(max(s2f, 1e-10)) + m.b)
end

"""
    s2f_predicted_price(m::StockToFlow, supply, annual_issuance) -> Float64

Predicted BTC price from S2F model.
"""
function s2f_predicted_price(m::StockToFlow, supply::Float64,
                               annual_issuance::Float64)::Float64
    mc = s2f_model(m, supply, annual_issuance)
    mc / max(supply, 1.0)
end

"""
    s2fx_model(supply, annual_issuance; phase="btc") -> Float64

S2FX model: maps BTC phases to gold/silver cluster.
Returns model-implied market cap.
"""
function s2fx_model(supply::Float64, annual_issuance::Float64;
                     phase::String="btc")::Float64
    s2f = supply / max(annual_issuance, 1.0)
    # S2FX clusters (from PlanB's 2020 paper, log-log space)
    if phase == "gold";   return exp(3.31 * log(s2f) + 14.0)
    elseif phase == "silver"; return exp(3.31 * log(s2f) + 13.0)
    else; return exp(3.31 * log(s2f) + 14.6)
    end
end

"""
    fit_s2f!(m, supply_series, issuance_series, price_series) -> StockToFlow

Fit S2F model via OLS regression on log-log data.
"""
function fit_s2f!(m::StockToFlow, supply_series::Vector{Float64},
                   issuance_series::Vector{Float64},
                   market_cap_series::Vector{Float64})::StockToFlow
    n = min(length(supply_series), length(issuance_series), length(market_cap_series))
    log_s2f = log.(max.(supply_series[1:n] ./ max.(issuance_series[1:n], 1.0), 1e-10))
    log_mc  = log.(max.(market_cap_series[1:n], 1.0))
    # OLS: log_mc = a * log_s2f + b
    x = hcat(log_s2f, ones(n))
    coef = (x'x + 1e-8*I) \ (x' * log_mc)
    StockToFlow(coef[1], coef[2])
end

"""
    s2f_signal(prices, model, supply_series, issuance_series) -> NamedTuple

Generate trading signal from S2F fair value deviation.
"""
function s2f_signal(prices::Vector{Float64}, m::StockToFlow,
                     supply_series::Vector{Float64},
                     issuance_series::Vector{Float64})
    n = min(length(prices), length(supply_series), length(issuance_series))
    predicted = [s2f_predicted_price(m, supply_series[t], issuance_series[t])
                 for t in 1:n]
    deviation = log.(max.(prices[1:n], 1.0)) .- log.(max.(predicted, 1.0))
    signal = clamp.(-deviation, -1.5, 1.5) ./ 1.5  # normalize to [-1,1]
    (predicted_price=predicted, log_deviation=deviation, signal=signal)
end

# ─────────────────────────────────────────────────────────────
# 6. EXCHANGE FLOW SIGNALS
# ─────────────────────────────────────────────────────────────

"""
    ExchangeFlowSignal

Exchange net flow = Exchange inflows - Exchange outflows.
  Net positive (coins moving to exchanges) → bearish (sell pressure)
  Net negative (coins leaving exchanges)   → bullish (HODLing, accumulation)

Exchange balance = cumsum of net flows. Declining balance is bullish.
"""
struct ExchangeFlowSignal
    smoothing::Int  # smoothing window for net flow
end

ExchangeFlowSignal() = ExchangeFlowSignal(7)

"""
    exchange_flow_signal(inflows, outflows; smoothing=7) -> NamedTuple
"""
function exchange_flow_signal(inflows::Vector{Float64},
                               outflows::Vector{Float64};
                               smoothing::Int=7)
    n          = min(length(inflows), length(outflows))
    net_flow   = inflows[1:n] .- outflows[1:n]
    exch_balance = cumsum(net_flow)

    # Smooth net flow
    alpha = 2.0 / (smoothing + 1)
    net_flow_s = similar(net_flow); net_flow_s[1] = net_flow[1]
    for t in 2:n
        net_flow_s[t] = alpha * net_flow[t] + (1-alpha) * net_flow_s[t-1]
    end

    # Signal: based on flow momentum
    signal = zeros(n)
    for t in smoothing:n
        seg_flow  = net_flow_s[t-smoothing+1:t]
        trend     = seg_flow[end] - seg_flow[1]  # flow trend
        # Negative trend (outflows increasing) = bullish
        signal[t] = -sign(trend) * min(abs(trend) / (std(seg_flow)+1e-10), 1.0)
    end

    # Exchange balance change rate
    balance_change = [t > 1 ? exch_balance[t] - exch_balance[t-1] : 0.0
                      for t in 1:n]

    (net_flow=net_flow, net_flow_smooth=net_flow_s,
     exchange_balance=exch_balance, balance_change=balance_change,
     signal=signal)
end

# ─────────────────────────────────────────────────────────────
# 7. HODL WAVE PROXY
# ─────────────────────────────────────────────────────────────

"""
    HODLWaveSignal

HODL Waves track what fraction of the circulating supply
last moved at different time intervals (1d, 1w, 1m, 3m, 6m, 1y, 2y, 3y, 5y+).

Long-term HODLing increases are bullish (diamond hands accumulating).
Short-term holder dominance near cycle tops is bearish.
"""
struct HODLWaveSignal
    lt_threshold::Float64  # fraction of long-term (>1y) holders = bullish
    st_threshold::Float64  # fraction of short-term (<1m) holders = bearish
end

HODLWaveSignal() = HODLWaveSignal(0.60, 0.30)

"""
    hodl_wave_proxy(prices, window=365) -> NamedTuple

Proxy HODL wave from price data using holder behavior patterns.
Long-term holders tend to buy dips (price below MA) and hold through tops.
"""
function hodl_wave_proxy(prices::Vector{Float64}; window::Int=365)
    n = length(prices)
    lt_proxy = zeros(n)  # long-term holder proxy (>1y behavior)
    st_proxy = zeros(n)  # short-term holder proxy

    for t in window:n
        price_ma = mean(prices[t-window+1:t])
        # Long-term holders: price below MA (accumulation) → increasing
        lt_proxy[t] = clamp((price_ma - prices[t]) / price_ma, 0.0, 1.0) * 0.3 + 0.5
        # Short-term holders: price above MA (distribution) → increasing
        st_proxy[t] = clamp((prices[t] - price_ma) / price_ma, 0.0, 1.0) * 0.3 + 0.1
    end

    # Normalize to sum to 1
    total = lt_proxy .+ st_proxy .+ 0.2  # 20% medium-term constant proxy
    lt_pct = lt_proxy ./ total
    st_pct = st_proxy ./ total

    signal = zeros(n)
    for t in 1:n
        lt_pct[t] > 0.6 && (signal[t] = 1.0)    # HODLer accumulation
        st_pct[t] > 0.35 && (signal[t] = -1.0)  # retail speculation
    end

    (lt_fraction=lt_pct, st_fraction=st_pct, signal=signal)
end

# ─────────────────────────────────────────────────────────────
# 8. HASH RIBBON
# ─────────────────────────────────────────────────────────────

"""
    HashRibbonSignal

Hash Ribbon uses the 30-day and 60-day MA of Bitcoin's hash rate.
  30MA crosses below 60MA → miner capitulation (short → buy signal)
  30MA recovers above 60MA → capitulation over (buy signal confirmed)
  Miners selling inventory during capitulation → short-term bearish pressure
"""
struct HashRibbonSignal
    ma_short::Int  # default 30
    ma_long::Int   # default 60
end

HashRibbonSignal() = HashRibbonSignal(30, 60)

"""
    hash_ribbon_signal(hash_rate) -> NamedTuple

Compute hash ribbon signal from hash rate time series.
"""
function hash_ribbon_signal(hash_rate::Vector{Float64};
                              ma_short::Int=30, ma_long::Int=60)
    n = length(hash_rate)
    ma_s = zeros(n); ma_l = zeros(n)
    for t in ma_short:n
        ma_s[t] = mean(hash_rate[t-ma_short+1:t])
    end
    for t in ma_long:n
        ma_l[t] = mean(hash_rate[t-ma_long+1:t])
    end

    signal = zeros(n)
    # Capitulation: when 30MA was below 60MA and crosses back above
    in_capitulation = false
    for t in 2:n
        ma_s[t] <= 0 || ma_l[t] <= 0 && continue
        prev_below = ma_s[t-1] < ma_l[t-1]
        now_above  = ma_s[t]   >= ma_l[t]
        now_below  = ma_s[t]   < ma_l[t]

        if now_below && !prev_below; in_capitulation = true; end
        if now_above && in_capitulation
            signal[t] = 1.5  # strong buy: capitulation over
            in_capitulation = false
        elseif in_capitulation
            signal[t] = -0.5  # mild bearish during capitulation
        end
    end

    # Ribbon compression index: how tight are the MAs?
    spread = [(ma_l[t] > 0 ? abs(ma_s[t] - ma_l[t]) / ma_l[t] : 0.0) for t in 1:n]

    (ma_30=ma_s, ma_60=ma_l, signal=signal, spread=spread)
end

"""
    miner_capitulation(hash_rate, miner_revenue, prices) -> NamedTuple

Detect miner capitulation events using hash rate decline + revenue stress.
"""
function miner_capitulation(hash_rate::Vector{Float64},
                              miner_revenue::Vector{Float64},
                              prices::Vector{Float64};
                              hash_decline_threshold::Float64=0.10)
    n = min(length(hash_rate), length(miner_revenue), length(prices))
    events = Int[]
    severity = zeros(n)

    for t in 31:n
        # Hash rate 30-day decline
        hash_30d_ago = hash_rate[t-30]
        hash_decline = (hash_30d_ago - hash_rate[t]) / (hash_30d_ago + 1e-10)

        # Revenue stress: compare to 1-year MA
        rev_ma = mean(miner_revenue[max(1,t-365):t])
        rev_stress = 1.0 - miner_revenue[t] / (rev_ma + 1e-10)

        severity[t] = hash_decline + rev_stress
        if hash_decline > hash_decline_threshold && rev_stress > 0.1
            push!(events, t)
        end
    end

    (capitulation_events=events, severity=severity,
     n_events=length(events))
end

# ─────────────────────────────────────────────────────────────
# 9. COMPOSITE ON-CHAIN SIGNAL
# ─────────────────────────────────────────────────────────────

"""
    OnChainComposite

Weighted composite of multiple on-chain signals.
"""
mutable struct OnChainComposite
    weights::Vector{Float64}
    signal_names::Vector{String}
    n_signals::Int
end

function OnChainComposite(n_signals::Int, names::Vector{String}=String[])
    wts = fill(1.0/n_signals, n_signals)
    ns  = isempty(names) ? ["signal_$i" for i in 1:n_signals] : names
    OnChainComposite(wts, ns, n_signals)
end

"""
    composite_signal(comp, signals_matrix) -> Vector{Float64}

Compute weighted composite from n_signals × T matrix of individual signals.
Returns T-vector of composite signal values in [-1, 1].
"""
function composite_signal(comp::OnChainComposite,
                            signals_matrix::Matrix{Float64})::Vector{Float64}
    # signals_matrix: n_signals × T
    T = size(signals_matrix, 2)
    result = zeros(T)
    for i in 1:comp.n_signals
        result .+= comp.weights[i] .* signals_matrix[i, :]
    end
    clamp.(result, -1.0, 1.0)
end

"""
    fit_composite_weights!(comp, signals_matrix, returns; method=:ridge)

Fit composite weights to maximize correlation with future returns.
method: :ridge (L2 regularized OLS) or :rank (rank correlation)
"""
function fit_composite_weights!(comp::OnChainComposite,
                                  signals_matrix::Matrix{Float64},
                                  returns::Vector{Float64};
                                  method::Symbol=:ridge,
                                  lambda::Float64=0.01)
    n_sig, T = size(signals_matrix)
    n = min(T - 1, length(returns) - 1)
    # Use signal at t to predict return at t+1
    X = signals_matrix[:, 1:n]'   # n × n_sig
    y = returns[2:n+1]

    if method == :ridge
        # Ridge regression: w = (X'X + λI)^{-1} X'y
        w = (X'X + lambda*I) \ (X'y)
    else
        # Simple rank correlation weights
        w = [cor(X[:,i], y) for i in 1:n_sig]
    end

    # Normalize to positive weights summing to 1
    w = abs.(w)
    w_sum = sum(w)
    w_sum > 1e-10 && (w ./= w_sum)
    comp.weights .= w
    comp
end

# ─────────────────────────────────────────────────────────────
# 10. BACKTESTING ON-CHAIN SIGNALS
# ─────────────────────────────────────────────────────────────

"""
    backtest_onchain_signal(prices, signal; tc=0.001, position_scale=1.0)
       -> NamedTuple

Simple backtest: position = signal value (clipped to [-1, 1]).
"""
function backtest_onchain_signal(prices::Vector{Float64},
                                   signal::Vector{Float64};
                                   tc::Float64=0.001,
                                   position_scale::Float64=1.0)
    n = min(length(prices), length(signal))
    log_returns = [0.0; diff(log.(max.(prices[1:n], 1e-10)))]

    portfolio = [10_000.0]
    position  = 0.0
    daily_ret = zeros(n)

    for t in 2:n
        desired  = clamp(signal[t-1] * position_scale, -1.0, 1.0)
        trade    = desired - position
        cost     = abs(trade) * tc
        position = desired
        ret      = position * log_returns[t] - cost
        pv       = portfolio[end] * exp(ret)
        push!(portfolio, pv)
        daily_ret[t] = ret
    end

    total_ret = (portfolio[end] - portfolio[1]) / portfolio[1]
    annual_ret = total_ret * (252 / n)
    annual_vol = std(daily_ret) * sqrt(252)
    sharpe  = annual_vol > 1e-10 ? annual_ret / annual_vol : 0.0

    # Max drawdown
    peak = portfolio[1]; mdd = 0.0
    for pv in portfolio
        peak = max(peak, pv); mdd = max(mdd, (peak-pv)/peak)
    end

    # BH comparison
    bh_ret = (prices[n] - prices[1]) / prices[1]

    (portfolio=portfolio, total_return=total_ret, annual_return=annual_ret,
     annual_volatility=annual_vol, sharpe=sharpe, max_drawdown=mdd,
     bh_return=bh_ret, alpha=total_ret - bh_ret, n_days=n)
end

"""
    compare_onchain_signals(prices, signals_dict) -> Dict

Backtest multiple signals and return comparative metrics.
"""
function compare_onchain_signals(prices::Vector{Float64},
                                   signals_dict::Dict{String,Vector{Float64}})
    results = Dict{String, NamedTuple}()
    for (name, sig) in signals_dict
        results[name] = backtest_onchain_signal(prices, sig)
    end
    results
end

# ─────────────────────────────────────────────────────────────
# 11. SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────

"""
    simulate_onchain_data(n; rng=...) -> NamedTuple

Generate synthetic on-chain data for demonstration.
All series are plausible BTC-like values.
"""
function simulate_onchain_data(n::Int=1000; rng=MersenneTwister(42))
    # Price with 4-year halving cycle
    prices = Float64[20_000.0]
    for t in 2:n
        cycle_factor = 1.0 + 0.5 * sin(2π * t / (4 * 365))
        mu   = 0.0005 * cycle_factor
        sig  = 0.025
        push!(prices, max(prices[end] * exp(mu + sig * randn(rng)), 1.0))
    end

    # Market cap = price * supply (gradually growing supply)
    supply = [19_000_000.0 + 6.25/144 * (t * 144) for t in 1:n]  # ~6.25 BTC/block
    supply = min.(supply, 21_000_000.0)
    market_cap = prices .* supply

    # Realized cap: lags market cap with mean-reversion
    realized_cap = similar(market_cap)
    realized_cap[1] = market_cap[1] * 0.7
    for t in 2:n
        # Slow mean-reversion toward market cap
        alpha = 0.005
        realized_cap[t] = (1-alpha)*realized_cap[t-1] + alpha*market_cap[t]
    end

    # Annual issuance (BTC halving roughly every 210k blocks = ~4yr)
    halving_idx = [365*4, 365*8]
    issuance = fill(6.25 * 144 * 365.0, n)
    for hi in halving_idx
        hi < n && (issuance[hi:end] ./= 2)
    end
    daily_issuance = issuance ./ 365

    # Transaction volume: correlated with price with noise
    tx_volume = market_cap .* (0.005 .+ 0.003 .* randn(rng, n))
    tx_volume = max.(tx_volume, 1.0)

    # Miner revenue: block subsidy + fees
    block_reward_usd = daily_issuance .* prices
    fees = market_cap .* 0.0001 .* (1 .+ 0.5 .* randn(rng, n).^2)
    miner_revenue = max.(block_reward_usd .+ fees, 1.0)

    # SOPR proxy: mean-reverting around 1.0
    sopr = 1.0 .+ 0.1 .* randn(rng, n) .+ 0.05 .* sin.(2π .* (1:n) ./ 365)
    cost_basis = prices ./ max.(sopr, 0.1)

    # Hash rate: correlated with price
    hash_rate = [80.0]
    for t in 2:n
        mu_h = 0.0003 * log(prices[t]/prices[t-1] + 1)
        push!(hash_rate, max(hash_rate[end] * exp(mu_h + 0.01*randn(rng)), 1.0))
    end

    # Exchange flows
    inflows  = max.(market_cap .* 0.001 .* abs.(randn(rng, n)), 0.0)
    outflows = max.(market_cap .* 0.001 .* abs.(randn(rng, n)), 0.0)
    # Net flows anti-correlated with realized cap changes
    rc_change = [t > 1 ? realized_cap[t] - realized_cap[t-1] : 0.0 for t in 1:n]
    inflows  .+= max.(rc_change .* 0.01, 0.0)
    outflows .+= max.(.-rc_change .* 0.01, 0.0)

    (prices=prices, market_cap=market_cap, realized_cap=realized_cap,
     supply=supply, issuance=daily_issuance, miner_revenue=miner_revenue,
     tx_volume=tx_volume, sopr=sopr, cost_basis=cost_basis,
     hash_rate=hash_rate, inflows=inflows, outflows=outflows)
end

# ─────────────────────────────────────────────────────────────
# 12. DEMO
# ─────────────────────────────────────────────────────────────

"""
    run_onchain_demo() -> Nothing
"""
function run_onchain_demo()
    println("=" ^ 60)
    println("CRYPTO ON-CHAIN ANALYTICS DEMO")
    println("=" ^ 60)

    rng = MersenneTwister(42)
    d = simulate_onchain_data(500; rng=rng)
    n = length(d.prices)
    println("Simulated $(n) days of on-chain data")
    println("Price range: \$$(round(minimum(d.prices),digits=0)) – \$$(round(maximum(d.prices),digits=0))")

    println("\n1. MVRV-Z Score")
    mvrv = mvrv_signal(d.market_cap, d.realized_cap; window=100)
    current_z = round(mvrv.z_score[end], digits=3)
    println("  Current MVRV ratio: $(round(mvrv.ratio[end],digits=3))x")
    println("  Current Z-score:    $(current_z)")
    println("  Signal:             $(mvrv.signal[end] > 0 ? "BUY" : mvrv.signal[end] < 0 ? "SELL" : "NEUTRAL")")
    n_buy  = sum(mvrv.signal .> 0)
    n_sell = sum(mvrv.signal .< 0)
    println("  Buy/Sell days: $n_buy / $n_sell")

    println("\n2. SOPR Signal")
    sopr = sopr_signal(d.sopr; smoothing=7)
    println("  Current SOPR (raw):    $(round(d.sopr[end],digits=4))")
    println("  Current SOPR (smooth): $(round(sopr.sopr_smooth[end],digits=4))")
    println("  Signal: $(sopr.signal[end])")

    println("\n3. NVT Signal")
    nvt = nvt_signal(d.market_cap, d.tx_volume; thresholds=(65.0, 100.0))
    valid_nvt = nvt.nvt_signal_series[nvt.nvt_signal_series .> 0]
    !isempty(valid_nvt) && println("  Avg NVT Signal: $(round(mean(valid_nvt),digits=1))")
    println("  Current signal: $(nvt.signal[end])")

    println("\n4. Puell Multiple")
    p = puell_signal(d.miner_revenue; buy=0.5, sell=4.0, window=100)
    println("  Current Puell: $(round(p.puell[end],digits=3))")
    println("  Signal: $(p.signal[end])")
    n_buy_p = sum(p.signal .> 0)
    println("  Buy signals: $n_buy_p days out of $(n-100)")

    println("\n5. Stock-to-Flow Model")
    s2f = StockToFlow()
    s2f_fitted = fit_s2f!(s2f, d.supply, d.issuance .* 365, d.market_cap)
    println("  S2F params: a=$(round(s2f_fitted.a,digits=3)), b=$(round(s2f_fitted.b,digits=3))")
    last_supply   = d.supply[end]
    last_issuance = d.issuance[end] * 365
    pred_price    = s2f_predicted_price(s2f_fitted, last_supply, last_issuance)
    actual_price  = d.prices[end]
    println("  Predicted price: \$$(round(pred_price,digits=0))")
    println("  Actual price:    \$$(round(actual_price,digits=0))")
    println("  Deviation:       $(round((actual_price/pred_price-1)*100,digits=1))%")
    sig_s2f = s2f_signal(d.prices, s2f_fitted, d.supply, d.issuance .* 365)
    println("  S2F signal (current): $(round(sig_s2f.signal[end],digits=3))")

    println("\n6. Exchange Flow Signal")
    efs = exchange_flow_signal(d.inflows, d.outflows; smoothing=7)
    println("  Current exchange balance: $(round(efs.exchange_balance[end]/1e9,digits=2))B USD")
    println("  Signal: $(round(efs.signal[end],digits=3))")

    println("\n7. HODL Wave Proxy")
    hodl = hodl_wave_proxy(d.prices; window=100)
    println("  LT holder fraction (current): $(round(hodl.lt_fraction[end]*100,digits=1))%")
    println("  ST holder fraction (current): $(round(hodl.st_fraction[end]*100,digits=1))%")
    println("  Signal: $(hodl.signal[end])")

    println("\n8. Hash Ribbon")
    hr = hash_ribbon_signal(d.hash_rate; ma_short=30, ma_long=60)
    n_buy_hr = sum(hr.signal .> 1.0)
    println("  Hash ribbon buy signals: $n_buy_hr")
    println("  Current spread: $(round(hr.spread[end]*100,digits=2))%")

    println("\n9. Composite Signal & Backtest")
    # Build signal matrix
    valid_start = 101
    n_valid = n - valid_start + 1
    sig_matrix = zeros(6, n)
    sig_matrix[1, :] = mvrv.signal
    sig_matrix[2, :] = sopr.signal
    sig_matrix[3, :] = nvt.signal
    sig_matrix[4, :] = p.signal
    sig_matrix[5, :] = efs.signal
    sig_matrix[6, :] = hodl.signal

    comp = OnChainComposite(6, ["mvrv","sopr","nvt","puell","exchange","hodl"])
    returns_for_fit = [t > 1 ? log(d.prices[t]/d.prices[t-1]) : 0.0 for t in 1:n]
    fit_composite_weights!(comp, sig_matrix, returns_for_fit; method=:ridge)
    println("  Fitted weights:")
    for (name, w) in zip(comp.signal_names, comp.weights)
        println("    $name: $(round(w*100,digits=1))%")
    end

    comp_signal = composite_signal(comp, sig_matrix)
    bt = backtest_onchain_signal(d.prices, comp_signal)
    println("\n  Backtest Results:")
    println("    Total Return:  $(round(bt.total_return*100,digits=2))%")
    println("    Annual Return: $(round(bt.annual_return*100,digits=2))%")
    println("    Sharpe Ratio:  $(round(bt.sharpe,digits=3))")
    println("    Max Drawdown:  $(round(bt.max_drawdown*100,digits=2))%")
    println("    BH Return:     $(round(bt.bh_return*100,digits=2))%")
    println("    Alpha:         $(round(bt.alpha*100,digits=2))%")

    println("\nDone.")
    nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 – Realised Volatility and ATR On-Chain Signals
# ─────────────────────────────────────────────────────────────────────────────

"""
    realised_volatility(prices, window)

Annualised realised volatility using log-returns over a rolling window.
"""
function realised_volatility(prices::Vector{Float64}, window::Int=30)
    n  = length(prices)
    rv = fill(NaN, n)
    for i in (window+1):n
        rets = diff(log.(prices[i-window:i]))
        rv[i] = std(rets) * sqrt(365)
    end
    return rv
end

"""
    average_true_range(high, low, close, period)

ATR: rolling mean of max(H-L, |H-Cₚ|, |L-Cₚ|) over `period` days.
Used as a volatility-adjusted position sizing signal.
"""
function average_true_range(high::Vector{Float64}, low::Vector{Float64},
                             close::Vector{Float64}, period::Int=14)
    n  = length(close)
    tr = fill(NaN, n)
    for i in 2:n
        hl  = high[i] - low[i]
        hcp = abs(high[i] - close[i-1])
        lcp = abs(low[i]  - close[i-1])
        tr[i] = max(hl, hcp, lcp)
    end
    atr = fill(NaN, n)
    for i in (period+1):n
        atr[i] = mean(tr[i-period+1:i])
    end
    return atr
end

"""
    vol_adjusted_position(price, atr, risk_pct, portfolio_value)

Kelly-style position sizing: risk `risk_pct` of `portfolio_value` per ATR unit.
Returns number of units to hold.
"""
function vol_adjusted_position(price::Float64, atr::Float64,
                                risk_pct::Float64=0.01,
                                portfolio_value::Float64=100_000.0)
    atr <= 0.0 && return 0.0
    dollar_risk = portfolio_value * risk_pct
    return dollar_risk / (atr * price)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 12 – Funding Rate Momentum and Basis Signal
# ─────────────────────────────────────────────────────────────────────────────

"""
    FundingRateSignal

Perpetual funding rate signal: sustained positive funding implies longs pay
shorts (crowded long), bearish contrarian; negative → bearish crowding, bullish.
Uses 8-hour funding rates (3 per day).
"""
struct FundingRateSignal
    smooth_window::Int   # EMA window for signal smoothing
    threshold::Float64   # abs(signal) > threshold → trade
end

FundingRateSignal(; smooth_window=24, threshold=0.0005) =
    FundingRateSignal(smooth_window, threshold)

"""
    funding_rate_signal(fr, rates) -> (signal, position)

`rates`: vector of periodic funding rates (e.g., 8-hourly).
Returns smoothed signal and contrarian position (+1 long, -1 short, 0 flat).
"""
function funding_rate_signal(fr::FundingRateSignal, rates::Vector{Float64})
    n  = length(rates)
    alpha = 2.0 / (fr.smooth_window + 1)
    ema   = zeros(n); ema[1] = rates[1]
    for i in 2:n
        ema[i] = alpha * rates[i] + (1 - alpha) * ema[i-1]
    end
    # contrarian: high positive funding → short signal
    pos = zeros(n)
    for i in 1:n
        if ema[i] >  fr.threshold; pos[i] = -1.0
        elseif ema[i] < -fr.threshold; pos[i] = 1.0
        end
    end
    return ema, pos
end

"""
    basis_signal(spot, futures) -> Vector{Float64}

Annualised basis (futures premium) = (F - S)/S * 365/days_to_expiry.
High positive basis signals contango; approaching zero signals sell pressure.
"""
function basis_signal(spot::Vector{Float64}, futures::Vector{Float64};
                      days_to_expiry::Int=30)
    return (futures .- spot) ./ spot .* (365.0 / days_to_expiry)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 13 – Glassnode-Style Whale and Cohort Analytics
# ─────────────────────────────────────────────────────────────────────────────

"""
    WhaleAccumulationSignal

Tracks net change in large-holder balances (proxy: addresses holding >1000 BTC).
Whale accumulation during dips historically precedes uptrends.
"""
mutable struct WhaleAccumulationSignal
    window::Int
    threshold::Float64   # net change fraction triggering signal
end

WhaleAccumulationSignal(; window=7, threshold=0.002) =
    WhaleAccumulationSignal(window, threshold)

"""
    whale_accumulation_signal(ws, whale_balance, price)

`whale_balance`: vector of aggregate whale holdings (USD or coin units).
Returns (+1, -1, 0) position signal and rolling net change.
"""
function whale_accumulation_signal(ws::WhaleAccumulationSignal,
                                    whale_balance::Vector{Float64},
                                    price::Vector{Float64})
    n     = length(whale_balance)
    delta = zeros(n)
    sig   = zeros(n)
    for i in (ws.window+1):n
        pct_chg = (whale_balance[i] - whale_balance[i-ws.window]) /
                   abs(whale_balance[i-ws.window] + 1e-8)
        delta[i] = pct_chg
        if pct_chg >  ws.threshold; sig[i] =  1.0   # accumulating
        elseif pct_chg < -ws.threshold; sig[i] = -1.0  # distributing
        end
    end
    return delta, sig
end

"""
    spent_output_age_distribution(ages, n_bins)

Histogram approximation of UTXO spent-output age bands.
Useful for on-chain lifespan analysis.
"""
function spent_output_age_distribution(ages::Vector{Float64},
                                        n_bins::Int=10)
    mn, mx = minimum(ages), maximum(ages)
    edges = range(mn, mx; length=n_bins+1)
    counts = zeros(Int, n_bins)
    for a in ages
        bin = min(n_bins, max(1, floor(Int, (a - mn) / (mx - mn + 1e-8) * n_bins) + 1))
        counts[bin] += 1
    end
    return collect(edges), counts
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 14 – On-Chain Regime Model and Signal Aggregation Dashboard
# ─────────────────────────────────────────────────────────────────────────────

"""
    OnChainRegime

Four regimes based on MVRV-Z and SOPR:
  1. Accumulation  — MVRV-Z < -1, SOPR < 1
  2. Early Bull    — MVRV-Z ∈ [-1, 2], SOPR ≥ 1
  3. Late Bull     — MVRV-Z > 2, SOPR > 1.05
  4. Distribution  — MVRV-Z > 3.5 or SOPR > 1.15
"""
@enum OnChainRegime AccumulationRegime EarlyBullRegime LateBullRegime DistributionRegime

function classify_onchain_regime(mvrv_z::Float64, sopr::Float64)::OnChainRegime
    if mvrv_z > 3.5 || sopr > 1.15;  return DistributionRegime
    elseif mvrv_z > 2.0 && sopr > 1.05; return LateBullRegime
    elseif mvrv_z >= -1.0;               return EarlyBullRegime
    else;                                 return AccumulationRegime
    end
end

"""
    regime_series(mvrv_z_vec, sopr_vec)

Apply `classify_onchain_regime` element-wise to produce a regime time series.
"""
function regime_series(mvrv_z_vec::Vector{Float64},
                        sopr_vec::Vector{Float64})
    @assert length(mvrv_z_vec) == length(sopr_vec)
    return [classify_onchain_regime(mvrv_z_vec[i], sopr_vec[i])
            for i in 1:length(mvrv_z_vec)]
end

"""
    regime_conditional_returns(returns, regimes)

Dictionary of mean and std return stratified by regime.
"""
function regime_conditional_returns(returns::Vector{Float64},
                                     regimes::Vector{OnChainRegime})
    results = Dict{OnChainRegime, NamedTuple}()
    for r in instances(OnChainRegime)
        mask = [reg == r for reg in regimes]
        if any(mask)
            rs = returns[mask]
            results[r] = (mean=mean(rs), std=std(rs), n=sum(mask))
        end
    end
    return results
end

"""
    onchain_dashboard(prices, market_cap, realised_cap, tx_volume,
                       supply, hash_rate, miner_revenue)

Convenience wrapper computing the core on-chain signal suite and printing
a summary table.  All input vectors assumed aligned daily.
"""
function onchain_dashboard(prices::Vector{Float64},
                            market_cap::Vector{Float64},
                            realised_cap::Vector{Float64},
                            tx_volume::Vector{Float64},
                            supply::Vector{Float64},
                            hash_rate::Vector{Float64},
                            miner_revenue::Vector{Float64})
    n = length(prices)

    # MVRV-Z
    mz  = compute_mvrv_z(market_cap, realised_cap)
    # SOPR (proxy: tx_volume / market_cap)
    sopr_proxy = tx_volume ./ (market_cap .+ 1e-8)

    # NVT
    nvt  = compute_nvt(market_cap, tx_volume)

    # Puell
    daily_issuance = diff(vcat(supply[1], supply)) .* prices
    puell = compute_puell(miner_revenue, daily_issuance)

    # Hash ribbon
    hr_sig = hash_ribbon_signal(hash_rate)

    println("=" ^ 60)
    println("On-Chain Dashboard  (n=$(n) days)")
    println("=" ^ 60)
    println("MVRV-Z     last=$(round(mz[end], digits=2))  " *
            "min=$(round(minimum(filter(!isnan, mz)), digits=2))  " *
            "max=$(round(maximum(filter(!isnan, mz)), digits=2))")
    println("NVT ratio  last=$(round(nvt[end], digits=1))")
    println("Puell Mult last=$(round(puell[end], digits=2))")
    println("Hash Ribbon signal=$(hr_sig[end])")
    println("=" ^ 60)

    return (mvrv_z=mz, sopr=sopr_proxy, nvt=nvt, puell=puell, hr=hr_sig)
end

end  # module CryptoOnChain
