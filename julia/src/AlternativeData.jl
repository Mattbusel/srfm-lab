module AlternativeData

# ============================================================
# AlternativeData.jl — Satellite signals, options flow,
#   dark pool analysis, social metrics, alt-data processing
# ============================================================

using Statistics, LinearAlgebra

export SatelliteSignal, OptionsFlowRecord, DarkPoolPrint, SocialMetric
export satellite_retail_traffic, satellite_cargo_activity
export options_flow_signal, unusual_options_activity, put_call_skew_signal
export dark_pool_imbalance, dark_pool_price_signal, lit_dark_ratio
export social_sentiment_score, social_volume_signal, trending_score
export news_sentiment_decay, event_signal_decay
export altdata_zscore, altdata_rank_signal, composite_alt_signal
export feature_engineering_pipeline, signal_orthogonalization
export information_coefficient, signal_turnover, signal_decay_halflife
export alt_data_factor_model, altdata_pca_factors
export backtest_alt_signal, signal_capacity_estimate
export web_traffic_signal, job_posting_signal, credit_card_signal

# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

struct SatelliteSignal
    ticker::String
    date::Int        # YYYYMMDD
    location_count::Float64
    visit_count::Float64
    car_count::Float64
    baseline_avg::Float64
end

struct OptionsFlowRecord
    ticker::String
    date::Int
    call_premium::Float64
    put_premium::Float64
    call_oi_change::Float64
    put_oi_change::Float64
    unusual_flag::Bool
end

struct DarkPoolPrint
    ticker::String
    date::Int
    dark_volume::Float64
    lit_volume::Float64
    dark_price::Float64
    lit_price::Float64
end

struct SocialMetric
    ticker::String
    date::Int
    mention_count::Float64
    positive_count::Float64
    negative_count::Float64
    total_count::Float64
end

# ──────────────────────────────────────────────────────────────
# Satellite / geolocation signals
# ──────────────────────────────────────────────────────────────

"""
    satellite_retail_traffic(signals, lookback, min_obs) -> z_score

Convert satellite foot traffic observations to a z-score signal.
Positive = above average traffic, likely positive for retail earnings.
"""
function satellite_retail_traffic(signals::Vector{SatelliteSignal},
                                    lookback::Int=52, min_obs::Int=10)
    n = length(signals)
    if n < min_obs; return 0.0; end
    recent = min(lookback, n)
    visits = [s.visit_count for s in signals]
    baselines = [s.baseline_avg for s in signals]
    # Compute yoy deviation
    deviations = (visits .- baselines) ./ max.(baselines, 1.0)
    recent_dev = mean(deviations[max(1,end-recent+1):end])
    historical_std = std(deviations) + 1e-12
    return recent_dev / historical_std
end

"""
    satellite_cargo_activity(port_counts, baseline_counts) -> signal

Shipping/cargo satellite signal: positive means supply chain is active.
"""
function satellite_cargo_activity(port_counts::Vector{Float64},
                                    baseline_counts::Vector{Float64})
    n = length(port_counts)
    if n == 0; return 0.0; end
    yoy_change = (port_counts .- baseline_counts) ./ max.(baseline_counts, 1.0)
    return mean(yoy_change)
end

"""
    satellite_parking_lot(car_counts, capacity, seasonality_factors) -> occupancy_signal

Parking lot signal normalized for seasonality and capacity.
"""
function satellite_parking_lot(car_counts::Vector{Float64},
                                 capacity::Float64,
                                 seasonality_factors::Vector{Float64})
    occupancy = car_counts ./ max(capacity, 1.0)
    adjusted = occupancy ./ max.(seasonality_factors, 0.01)
    return mean(adjusted) - 1.0  # deviation from expected occupancy
end

# ──────────────────────────────────────────────────────────────
# Options flow signals
# ──────────────────────────────────────────────────────────────

"""
    options_flow_signal(records, lookback) -> (bullish_score, bearish_score)

Convert options flow data to directional signal.
Unusual call buying → bullish; unusual put buying → bearish.
"""
function options_flow_signal(records::Vector{OptionsFlowRecord}, lookback::Int=20)
    n = min(lookback, length(records))
    if n == 0; return 0.0, 0.0; end
    recent = records[end-n+1:end]
    call_flow = sum(r.call_premium for r in recent)
    put_flow = sum(r.put_premium for r in recent)
    unusual_calls = sum(r.call_premium for r in recent if r.unusual_flag)
    unusual_puts = sum(r.put_premium for r in recent if r.unusual_flag)
    total = call_flow + put_flow + 1e-12
    bullish = (call_flow + 2.0 * unusual_calls) / total
    bearish = (put_flow + 2.0 * unusual_puts) / total
    return bullish, bearish
end

"""
    unusual_options_activity(record, avg_daily_volume, threshold) -> is_unusual

Flag unusual options activity based on premium relative to average.
"""
function unusual_options_activity(record::OptionsFlowRecord,
                                    avg_call_premium::Float64,
                                    avg_put_premium::Float64,
                                    threshold::Float64=3.0)
    call_unusual = record.call_premium > threshold * avg_call_premium
    put_unusual = record.put_premium > threshold * avg_put_premium
    return call_unusual || put_unusual
end

"""
    put_call_skew_signal(records, short_window, long_window) -> signal

Put/call ratio signal: ratio of put to call premium flow.
Negative signal when puts dominate (bearish flow).
"""
function put_call_skew_signal(records::Vector{OptionsFlowRecord},
                                short_window::Int=5, long_window::Int=20)
    n = length(records)
    if n < short_window; return 0.0; end
    function pc_ratio(rs)
        calls = sum(r.call_premium for r in rs) + 1e-12
        puts = sum(r.put_premium for r in rs)
        return puts / calls
    end
    short_r = records[max(1,end-short_window+1):end]
    long_r = records[max(1,end-long_window+1):end]
    short_pc = pc_ratio(short_r)
    long_pc = pc_ratio(long_r)
    # Signal: short PC below long PC → bullish (less put buying recently)
    return -(short_pc - long_pc)
end

"""
    oi_change_signal(records, lookback) -> directional_signal

Open interest change signal: rising call OI → bullish positioning.
"""
function oi_change_signal(records::Vector{OptionsFlowRecord}, lookback::Int=10)
    n = min(lookback, length(records))
    if n == 0; return 0.0; end
    recent = records[end-n+1:end]
    net_oi = sum(r.call_oi_change - r.put_oi_change for r in recent)
    normalizer = sum(abs(r.call_oi_change) + abs(r.put_oi_change) for r in recent) + 1e-12
    return net_oi / normalizer
end

# ──────────────────────────────────────────────────────────────
# Dark pool signals
# ──────────────────────────────────────────────────────────────

"""
    dark_pool_imbalance(prints, lookback) -> imbalance_score

Net buy/sell imbalance from dark pool prints.
Positive = net buying in dark (potentially bullish).
"""
function dark_pool_imbalance(prints::Vector{DarkPoolPrint}, lookback::Int=20)
    n = min(lookback, length(prints))
    if n == 0; return 0.0; end
    recent = prints[end-n+1:end]
    # Infer direction from price vs VWAP
    imbalances = Float64[]
    for p in recent
        if p.dark_price > p.lit_price
            push!(imbalances, p.dark_volume)   # bought above market → buy imbalance
        elseif p.dark_price < p.lit_price
            push!(imbalances, -p.dark_volume)  # sold below market → sell imbalance
        end
    end
    if isempty(imbalances); return 0.0; end
    total_vol = sum(p.dark_volume for p in recent) + 1e-12
    return sum(imbalances) / total_vol
end

"""
    dark_pool_price_signal(prints) -> premium_signal

Dark pool price premium relative to lit market.
Consistent premium → institutional buying pressure.
"""
function dark_pool_price_signal(prints::Vector{DarkPoolPrint})
    if isempty(prints); return 0.0; end
    premiums = [(p.dark_price - p.lit_price) / max(p.lit_price, 1e-12) for p in prints]
    return mean(premiums)
end

"""
    lit_dark_ratio(prints, lookback) -> ratio_signal

Ratio of dark to lit volume. High ratio can indicate large institutional interest.
"""
function lit_dark_ratio(prints::Vector{DarkPoolPrint}, lookback::Int=20)
    n = min(lookback, length(prints))
    if n == 0; return 0.5; end
    recent = prints[end-n+1:end]
    dark_vol = sum(p.dark_volume for p in recent)
    lit_vol = sum(p.lit_volume for p in recent)
    return dark_vol / max(dark_vol + lit_vol, 1e-12)
end

"""
    dark_pool_block_signal(prints, block_threshold) -> signal

Signal from large block trades in dark pools. Large blocks → informed trading.
"""
function dark_pool_block_signal(prints::Vector{DarkPoolPrint},
                                  block_threshold::Float64=1e6)
    block_buys = sum(p.dark_volume for p in prints
                     if p.dark_volume >= block_threshold && p.dark_price >= p.lit_price)
    block_sells = sum(p.dark_volume for p in prints
                      if p.dark_volume >= block_threshold && p.dark_price < p.lit_price)
    total = block_buys + block_sells + 1e-12
    return (block_buys - block_sells) / total
end

# ──────────────────────────────────────────────────────────────
# Social / sentiment signals
# ──────────────────────────────────────────────────────────────

"""
    social_sentiment_score(metrics, lookback) -> sentiment_score

Net sentiment score from social metrics.
Range approximately [-1, 1]; positive = net positive.
"""
function social_sentiment_score(metrics::Vector{SocialMetric}, lookback::Int=7)
    n = min(lookback, length(metrics))
    if n == 0; return 0.0; end
    recent = metrics[end-n+1:end]
    total_pos = sum(m.positive_count for m in recent)
    total_neg = sum(m.negative_count for m in recent)
    total = total_pos + total_neg + 1e-12
    return (total_pos - total_neg) / total
end

"""
    social_volume_signal(metrics, short_window, long_window) -> volume_surge

Ratio of recent mention volume to historical average.
Surge in mentions often precedes price movement.
"""
function social_volume_signal(metrics::Vector{SocialMetric},
                                short_window::Int=3, long_window::Int=30)
    n = length(metrics)
    if n < short_window; return 0.0; end
    short_vol = mean(m.mention_count for m in metrics[max(1,end-short_window+1):end])
    long_vol = mean(m.mention_count for m in metrics[max(1,end-long_window+1):end])
    return long_vol > 1e-12 ? short_vol / long_vol - 1.0 : 0.0
end

"""
    trending_score(metrics, decay_half_life) -> score

Exponentially weighted trending score for social activity.
"""
function trending_score(metrics::Vector{SocialMetric}, decay_half_life::Float64=3.0)
    n = length(metrics)
    if n == 0; return 0.0; end
    lambda = log(2.0) / decay_half_life
    weighted_sum = 0.0
    weight_sum = 0.0
    for (k, m) in enumerate(reverse(metrics))
        w = exp(-lambda * (k - 1))
        weighted_sum += w * m.mention_count
        weight_sum += w
    end
    return weight_sum > 0 ? weighted_sum / weight_sum : 0.0
end

"""
    news_sentiment_decay(sentiment_series, halflife) -> current_signal

Exponentially decay news sentiment over time.
sentiment_series: time-ordered vector of sentiment scores [-1, 1].
"""
function news_sentiment_decay(sentiment_series::Vector{Float64}, halflife::Float64=5.0)
    n = length(sentiment_series)
    if n == 0; return 0.0; end
    lambda = log(2.0) / halflife
    weights = [exp(-lambda * (n - t)) for t in 1:n]
    return dot(weights, sentiment_series) / sum(weights)
end

"""
    event_signal_decay(event_magnitude, days_since_event, halflife) -> current_impact
"""
function event_signal_decay(event_magnitude::Float64, days_since::Float64,
                              halflife::Float64=10.0)
    return event_magnitude * exp(-log(2.0) / halflife * days_since)
end

# ──────────────────────────────────────────────────────────────
# Signal processing / normalization
# ──────────────────────────────────────────────────────────────

"""
    altdata_zscore(signal_series, lookback, winsorize_std) -> z_scores

Rolling z-score normalization of a signal series with optional winsorization.
"""
function altdata_zscore(signal_series::Vector{Float64},
                          lookback::Int=252, winsorize_std::Float64=3.0)
    n = length(signal_series)
    z = zeros(n)
    for i in 1:n
        window = signal_series[max(1, i-lookback+1):i]
        mu = mean(window)
        sigma = std(window) + 1e-12
        z[i] = (signal_series[i] - mu) / sigma
        z[i] = clamp(z[i], -winsorize_std, winsorize_std)
    end
    return z
end

"""
    altdata_rank_signal(signal_series, lookback) -> rank_scores [-0.5, 0.5]

Cross-sectional rank transform for a panel of signals.
"""
function altdata_rank_signal(signal_matrix::Matrix{Float64})
    n_time, n_assets = size(signal_matrix)
    ranked = zeros(n_time, n_assets)
    for t in 1:n_time
        row = signal_matrix[t, :]
        sorted_idx = sortperm(row)
        ranks = zeros(n_assets)
        for (r, i) in enumerate(sorted_idx)
            ranks[i] = (r - 1) / max(n_assets - 1, 1) - 0.5
        end
        ranked[t, :] = ranks
    end
    return ranked
end

"""
    composite_alt_signal(signals_matrix, weights) -> composite

Weighted combination of multiple alternative data signals.
signals_matrix: T x K matrix of signals, weights: K-vector.
"""
function composite_alt_signal(signals_matrix::Matrix{Float64},
                                weights::Vector{Float64})
    w = weights ./ max(sum(abs.(weights)), 1e-12)
    return signals_matrix * w
end

# ──────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────

"""
    feature_engineering_pipeline(raw_signal, lags, ma_windows) -> feature_matrix

Build feature matrix from a raw alt-data signal.
Includes: lags, moving averages, momentum, volatility.
"""
function feature_engineering_pipeline(raw::Vector{Float64},
                                        lags::Vector{Int}=[1,2,5],
                                        ma_windows::Vector{Int}=[5,10,20])
    n = length(raw)
    features = Dict{String, Vector{Float64}}()

    # Raw signal
    features["raw"] = raw

    # Lags
    for lag in lags
        feat = zeros(n)
        for i in lag+1:n
            feat[i] = raw[i-lag]
        end
        features["lag_$lag"] = feat
    end

    # Moving averages and MA-diff
    for w in ma_windows
        ma = zeros(n)
        for i in w:n
            ma[i] = mean(raw[i-w+1:i])
        end
        features["ma_$w"] = ma
        # Signal minus MA (mean-reversion indicator)
        features["diff_ma_$w"] = raw .- ma
    end

    # 20-day rolling volatility
    vol = zeros(n)
    for i in 21:n
        vol[i] = std(raw[i-20:i])
    end
    features["vol_20"] = vol

    # Momentum (rate of change)
    mom = zeros(n)
    for i in 6:n
        mom[i] = (raw[i] - raw[i-5]) / max(abs(raw[i-5]), 1e-12)
    end
    features["mom_5"] = mom

    # Assemble matrix
    keys_sorted = sort(collect(keys(features)))
    F = hcat([features[k] for k in keys_sorted]...)
    return F, keys_sorted
end

"""
    signal_orthogonalization(signal_matrix, factor_matrix) -> residual_signals

Remove exposure to known factors from alt-data signals.
Uses OLS residualization.
"""
function signal_orthogonalization(signals::Matrix{Float64},
                                    factors::Matrix{Float64})
    n, k_sig = size(signals)
    _, k_fac = size(factors)
    X = hcat(ones(n), factors)
    residuals = zeros(n, k_sig)
    for j in 1:k_sig
        beta = (X'X + 1e-10*I) \ (X' * signals[:, j])
        residuals[:, j] = signals[:, j] .- X * beta
    end
    return residuals
end

# ──────────────────────────────────────────────────────────────
# Signal evaluation metrics
# ──────────────────────────────────────────────────────────────

"""
    information_coefficient(signal, forward_returns, method) -> IC

Information coefficient between signal and forward returns.
method ∈ :pearson or :spearman
"""
function information_coefficient(signal::Vector{Float64},
                                   forward_returns::Vector{Float64},
                                   method::Symbol=:spearman)
    n = length(signal)
    if n < 5; return 0.0; end
    if method == :spearman
        rank_s = Float64.(sortperm(sortperm(signal)))
        rank_r = Float64.(sortperm(sortperm(forward_returns)))
        return cor(rank_s, rank_r)
    else
        return cor(signal, forward_returns)
    end
end

"""
    signal_turnover(signal_series) -> avg_turnover

Average daily turnover of the signal (fraction of portfolio rebalanced).
"""
function signal_turnover(signal_series::Matrix{Float64})
    n_time, n_assets = size(signal_series)
    if n_time < 2; return 0.0; end
    turnovers = zeros(n_time - 1)
    for t in 2:n_time
        delta = abs.(signal_series[t,:] .- signal_series[t-1,:])
        turnovers[t-1] = sum(delta) / 2.0
    end
    return mean(turnovers)
end

"""
    signal_decay_halflife(ic_series) -> halflife_days

Estimate signal decay half-life by fitting exponential to IC vs lag.
"""
function signal_decay_halflife(ic_at_lags::Vector{Float64})
    n = length(ic_at_lags)
    if n < 3; return Inf; end
    ic0 = max(ic_at_lags[1], 1e-12)
    # Find lag where IC drops to half
    for i in 2:n
        if ic_at_lags[i] <= ic0 / 2.0
            return Float64(i - 1)
        end
    end
    return Float64(n)
end

# ──────────────────────────────────────────────────────────────
# Factor model integration
# ──────────────────────────────────────────────────────────────

"""
    alt_data_factor_model(signal_matrix, returns_matrix, n_factors) -> (loadings, factors, alpha)

Extract latent factors from alt-data signals and compute return predictability.
"""
function alt_data_factor_model(signals::Matrix{Float64},
                                 returns::Matrix{Float64},
                                 n_factors::Int=3)
    T, N = size(signals)
    # Standardize signals
    sig_std = (signals .- mean(signals, dims=1)) ./ (std(signals, dims=1) .+ 1e-12)
    # PCA on signals via SVD
    F = svd(sig_std / sqrt(T))
    factors = F.U[:, 1:n_factors] .* sqrt(T)
    loadings = F.V[:, 1:n_factors] .* F.S[1:n_factors]' ./ sqrt(T)
    # Predict returns from factors
    X = hcat(ones(T), factors)
    betas = (X'X + 1e-10*I) \ (X' * returns)
    alpha = betas[1:1, :]  # intercepts
    return loadings, factors, alpha
end

"""
    altdata_pca_factors(signal_matrix, n_components) -> (components, explained_var)

PCA decomposition of alternative data signal matrix.
"""
function altdata_pca_factors(signals::Matrix{Float64}, n_components::Int=5)
    T, N = size(signals)
    sig_centered = signals .- mean(signals, dims=1)
    C = (sig_centered' * sig_centered) / (T - 1)
    # Power iteration for top components
    components = zeros(N, n_components)
    explained = zeros(n_components)
    C_remaining = copy(C)
    for k in 1:n_components
        v = randn(N); v ./= norm(v)
        for _ in 1:200
            v_new = C_remaining * v
            lambda = norm(v_new)
            v = lambda > 1e-12 ? v_new ./ lambda : v_new
        end
        lambda_k = dot(v, C_remaining * v)
        components[:, k] = v
        explained[k] = lambda_k
        C_remaining .-= lambda_k .* (v * v')
    end
    total_var = tr(C)
    return components, explained ./ max(total_var, 1e-12)
end

# ──────────────────────────────────────────────────────────────
# Specialized alt-data signals
# ──────────────────────────────────────────────────────────────

"""
    web_traffic_signal(visit_counts, unique_visitors, bounce_rates, lookback) -> signal

Web traffic alternative data signal. High traffic with low bounce = positive.
"""
function web_traffic_signal(visits::Vector{Float64}, unique_visitors::Vector{Float64},
                              bounce_rates::Vector{Float64}, lookback::Int=30)
    n = min(lookback, length(visits))
    if n == 0; return 0.0; end
    recent_v = visits[end-n+1:end]
    recent_uv = unique_visitors[end-n+1:end]
    recent_br = bounce_rates[end-n+1:end]
    # Quality-adjusted traffic
    quality_traffic = recent_v .* recent_uv ./ (visits .+ 1.0)[end-n+1:end] .* (1.0 .- recent_br)
    hist_mean = length(visits) > n ? mean(visits[1:end-n]) : mean(visits)
    return (mean(quality_traffic) - hist_mean) / max(hist_mean, 1e-12)
end

"""
    job_posting_signal(postings_by_dept, total_employees, lookback) -> signal

Job posting alt-data. Surge in hiring → corporate expansion signal.
"""
function job_posting_signal(tech_postings::Vector{Float64},
                              total_postings::Vector{Float64},
                              total_employees::Float64,
                              lookback::Int=30)
    n = min(lookback, length(total_postings))
    if n == 0; return 0.0; end
    recent_postings = total_postings[end-n+1:end]
    hist_postings = total_postings[1:max(1, end-n)]
    hiring_rate = mean(recent_postings) / max(total_employees, 1.0)
    hist_hiring = mean(hist_postings) / max(total_employees, 1.0)
    # Tech ratio (R&D intensity signal)
    tech_ratio = mean(tech_postings[end-n+1:end]) / max(mean(total_postings[end-n+1:end]), 1.0)
    return (hiring_rate - hist_hiring) / max(hist_hiring, 1e-12) + 0.5 * (tech_ratio - 0.3)
end

"""
    credit_card_signal(sales_data, baseline, seasonality, lookback) -> signal

Credit card spending alt-data signal. Normalized for seasonality.
"""
function credit_card_signal(sales::Vector{Float64}, baseline::Vector{Float64},
                              seasonality_idx::Vector{Float64}, lookback::Int=4)
    n = min(lookback, length(sales))
    if n == 0; return 0.0; end
    recent_s = sales[end-n+1:end]
    recent_b = baseline[end-n+1:end]
    recent_sea = seasonality_idx[end-n+1:end]
    # Seasonality-adjusted deviation from baseline
    adj_deviation = ((recent_s ./ max.(recent_sea, 0.01)) .- recent_b) ./ max.(recent_b, 1.0)
    return mean(adj_deviation)
end

# ──────────────────────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────────────────────

"""
    backtest_alt_signal(signal_series, forward_returns, holding_period, transaction_cost)

Simple long-short backtest of an alt-data signal.
Returns (annualized_return, sharpe, ic_series, max_drawdown).
"""
function backtest_alt_signal(signal::Vector{Float64}, returns::Vector{Float64},
                               holding_period::Int=5,
                               transaction_cost::Float64=0.001)
    n = length(signal)
    strategy_returns = zeros(n)
    ic_series = zeros(n - holding_period)

    for t in 1:n-holding_period
        fwd_ret = mean(returns[t+1:t+holding_period])
        # Long if signal > 0, short if < 0
        position = sign(signal[t])
        strategy_returns[t] = position * fwd_ret - transaction_cost * abs(position)
        ic_series[t] = cor([signal[t]], [fwd_ret])  # single point IC = ±1
    end

    # More meaningful IC: rolling 60-day window
    ic_rolling = Float64[]
    window_ic = 60
    for t in window_ic:n-holding_period
        sigs = signal[t-window_ic+1:t]
        fwds = [mean(returns[s+1:min(s+holding_period,n)]) for s in t-window_ic+1:t]
        if std(sigs) > 1e-12 && std(fwds) > 1e-12
            push!(ic_rolling, cor(sigs, fwds))
        end
    end

    active_returns = strategy_returns[strategy_returns .!= 0]
    ann_ret = mean(active_returns) * 252.0
    ann_std = std(active_returns) * sqrt(252.0)
    sharpe = ann_std > 1e-12 ? ann_ret / ann_std : 0.0

    # Max drawdown
    cum_ret = cumsum(strategy_returns)
    max_drawdown = 0.0
    peak = cum_ret[1]
    for r in cum_ret
        peak = max(peak, r)
        max_drawdown = max(max_drawdown, peak - r)
    end

    mean_ic = isempty(ic_rolling) ? 0.0 : mean(ic_rolling)
    return ann_ret, sharpe, mean_ic, max_drawdown
end

"""
    signal_capacity_estimate(signal, returns, avg_daily_volume, max_impact) -> capacity_usd

Estimate signal capacity: max AUM before signal is arbitraged away.
"""
function signal_capacity_estimate(signal::Vector{Float64}, returns::Vector{Float64},
                                    avg_daily_volume::Float64,
                                    max_impact_bps::Float64=5.0)
    n = length(signal)
    # Expected alpha per unit of signal
    if std(signal) < 1e-12; return 0.0; end
    alpha = cor(signal[1:end-1], returns[2:end]) * std(returns[2:end]) / std(signal[1:end-1])
    # Capacity = volume * (alpha / max_impact)
    max_impact = max_impact_bps / 10_000.0
    capacity = avg_daily_volume * (alpha / max(max_impact, 1e-6)) * 252.0
    return max(capacity, 0.0)
end

end # module AlternativeData
