# ============================================================
# Notebook 29: Alternative Data Sources & Signal Construction
# ============================================================
# Topics:
#   1. Satellite foot traffic data simulation and processing
#   2. Options flow signal construction
#   3. Dark pool analysis
#   4. Social sentiment signal
#   5. Web traffic and job posting data
#   6. Signal combination and IC analysis
#   7. Orthogonalization against traditional factors
#   8. Backtest of combined alt-data strategy
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 29: Alternative Data Integration")
println("="^60)

# ── Utility: RNG ──────────────────────────────────────────
state_rng = UInt64(42)
function next_rand()
    global state_rng
    state_rng = state_rng * 6364136223846793005 + 1442695040888963407
    return (state_rng >> 11) / Float64(2^53)
end
function next_randn()
    u1 = max(next_rand(), 1e-15)
    u2 = next_rand()
    return sqrt(-2.0*log(u1)) * cos(2π*u2)
end

# ── Section 1: Satellite Foot Traffic ─────────────────────

println("\n--- Section 1: Satellite Foot Traffic ---")

n_weeks = 104  # 2 years of weekly data
n_retailers = 20

# Simulate weekly foot traffic for retailers
# Some have predictable seasonal patterns; some have earnings surprises
true_alphas = [0.05 * next_randn() for _ in 1:n_retailers]
seasonality_amp = [0.10 + 0.05 * next_rand() for _ in 1:n_retailers]

foot_traffic = zeros(n_weeks, n_retailers)
weekly_returns = zeros(n_weeks, n_retailers)
for j in 1:n_retailers
    for t in 1:n_weeks
        # Seasonal pattern (holiday peaks)
        week_of_year = ((t - 1) % 52) + 1
        season = seasonality_amp[j] * sin(2π * week_of_year / 52) +
                  0.15 * sin(4π * week_of_year / 52)
        # Trend + alpha
        trend = 1.0 + true_alphas[j] * t / n_weeks
        noise = 0.05 * next_randn()
        foot_traffic[t, j] = 100.0 * trend * (1.0 + season + noise)
        # Stock return has weak correlation with foot traffic
        alpha_signal = 0.3 * (season + noise)
        weekly_returns[t, j] = alpha_signal * 0.01 + 0.02 * next_randn()
    end
end

println("Simulated $n_weeks weeks × $n_retailers retailers")
println("Foot traffic stats:")
println("  Mean: $(round(mean(foot_traffic), digits=1))")
println("  Std:  $(round(std(foot_traffic), digits=1))")

# Process satellite signal: year-over-year deviation z-score
function yoy_zscore_signal(traffic_matrix, lookback=52)
    T, N = size(traffic_matrix)
    signal = zeros(T, N)
    for t in lookback+1:T
        for j in 1:N
            recent = traffic_matrix[t, j]
            yoy_base = traffic_matrix[t-lookback, j]
            historical = [traffic_matrix[s, j] / traffic_matrix[max(1,s-lookback), j]
                          for s in lookback+1:t-1]
            if length(historical) < 4
                signal[t, j] = (recent - yoy_base) / max(yoy_base, 1.0)
            else
                mu_ratio = mean(historical)
                sig_ratio = std(historical) + 1e-12
                signal[t, j] = ((recent / yoy_base) - mu_ratio) / sig_ratio
            end
        end
    end
    return signal
end

satellite_signal = yoy_zscore_signal(foot_traffic)

# IC of satellite signal vs 4-week forward return
fwd_returns_4w = zeros(n_weeks, n_retailers)
for t in 1:n_weeks-4
    for j in 1:n_retailers
        fwd_returns_4w[t, j] = sum(weekly_returns[t+1:t+4, j])
    end
end

ic_per_week = zeros(n_weeks)
for t in 54:n_weeks-4
    sig = satellite_signal[t, :]
    fwd = fwd_returns_4w[t, :]
    valid = abs.(sig) .> 1e-12
    if sum(valid) > 5
        ic_per_week[t] = cor(sig[valid], fwd[valid])
    end
end

valid_ics = ic_per_week[ic_per_week .!= 0]
println("Satellite Signal IC Analysis:")
println("  Mean IC: $(round(mean(valid_ics), digits=4))")
println("  IC Std:  $(round(std(valid_ics), digits=4))")
println("  IR (IC/std): $(round(mean(valid_ics)/std(valid_ics)*sqrt(52), digits=3)) annualized")

# ── Section 2: Options Flow Signal ────────────────────────

println("\n--- Section 2: Options Flow Signal ---")

n_days = 252
n_stocks = 15

# Simulate daily options flow data
# call_premium, put_premium, unusual activity flag
call_prem = zeros(n_days, n_stocks)
put_prem  = zeros(n_days, n_stocks)
unusual_flag = falses(n_days, n_stocks)
stock_returns = zeros(n_days, n_stocks)
informed_signal = zeros(n_days, n_stocks)

for j in 1:n_stocks
    for t in 1:n_days
        # Unusual activity on ~5% of days
        is_unusual = next_rand() < 0.05
        unusual_flag[t, j] = is_unusual
        call_base = 50000.0 + 20000.0 * next_randn()
        put_base  = 45000.0 + 18000.0 * next_randn()
        if is_unusual
            # Unusual flow: directional bet
            direction = next_rand() > 0.5 ? 1.0 : -1.0
            call_prem[t, j] = call_base + (direction > 0 ? 3.0 : 1.0) * abs(call_base) * 0.5
            put_prem[t, j]  = put_base  + (direction < 0 ? 3.0 : 1.0) * abs(put_base)  * 0.5
            informed_signal[t, j] = direction  # true direction
        else
            call_prem[t, j] = max(call_base, 1000.0)
            put_prem[t, j]  = max(put_base,  1000.0)
            informed_signal[t, j] = 0.0
        end
        # Returns: partially predict by informed flow
        fwd_noise = 0.02 * next_randn()
        stock_returns[t, j] = 0.015 * informed_signal[t, j] + fwd_noise
    end
end

# Construct put/call ratio signal
function options_flow_signal(calls, puts, unusual, lookback=10)
    T, N = size(calls)
    signal = zeros(T, N)
    for t in lookback:T
        for j in 1:N
            recent_calls = calls[t-lookback+1:t, j]
            recent_puts  = puts[t-lookback+1:t, j]
            recent_uniq  = unusual[t-lookback+1:t, j]
            total_calls = sum(recent_calls)
            total_puts  = sum(recent_puts)
            unusual_call = sum(recent_calls[recent_uniq])
            unusual_put  = sum(recent_puts[recent_uniq])
            pc_ratio = total_puts / max(total_calls, 1.0)
            # Negative pc_ratio anomaly → bullish
            # Enhanced by unusual call activity
            signal[t, j] = -(pc_ratio - 1.0) + 2.0 * unusual_call / max(total_calls, 1.0) -
                             2.0 * unusual_put / max(total_puts, 1.0)
        end
    end
    return signal
end

opt_signal = options_flow_signal(call_prem, put_prem, unusual_flag)

# IC of options signal vs 5-day forward returns
fwd_5d = zeros(n_days, n_stocks)
for t in 1:n_days-5
    fwd_5d[t, :] = sum(stock_returns[t+1:t+5, :], dims=1)[:]
end

ic_opt = zeros(n_days)
for t in 12:n_days-5
    sig = opt_signal[t, :]
    fwd = fwd_5d[t, :]
    if std(sig) > 1e-12 && std(fwd) > 1e-12
        ic_opt[t] = cor(sig, fwd)
    end
end
valid_opt_ics = ic_opt[ic_opt .!= 0]
println("Options Flow Signal:")
println("  Mean IC: $(round(mean(valid_opt_ics), digits=4))")
println("  IC IR:   $(round(mean(valid_opt_ics)/std(valid_opt_ics)*sqrt(252), digits=3)) annualized")

# ── Section 3: Dark Pool Signal ───────────────────────────

println("\n--- Section 3: Dark Pool Analysis ---")

# Simulate dark pool prints
dark_vol_frac = 0.30 .+ 0.10 .* randn.(n_days)

function simulate_dark_pool(stock_rets, n_days, n_stocks)
    dark_premium = zeros(n_days, n_stocks)
    dark_vol_ratio = zeros(n_days, n_stocks)
    for j in 1:n_stocks
        for t in 1:n_days
            # Dark pool premium correlates with informed buying
            future_ret = t < n_days ? stock_rets[t+1, j] : 0.0
            dark_premium[t, j] = 0.3 * future_ret + 0.0001 * next_randn()
            dark_vol_ratio[t, j] = 0.25 + 0.10 * next_randn() + 0.2 * abs(future_ret) * 50
            dark_vol_ratio[t, j] = clamp(dark_vol_ratio[t, j], 0.05, 0.80)
        end
    end
    return dark_premium, dark_vol_ratio
end

dp_premium, dp_vol_ratio = simulate_dark_pool(stock_returns, n_days, n_stocks)

# Dark pool signal: premium and volume surge
function dark_pool_signal_matrix(premium, vol_ratio, lookback=5)
    T, N = size(premium)
    signal = zeros(T, N)
    for t in lookback+1:T
        for j in 1:N
            avg_prem = mean(premium[t-lookback:t, j])
            avg_ratio = mean(vol_ratio[t-lookback:t, j])
            hist_ratio = mean(vol_ratio[max(1,t-lookback*4):t-lookback, j])
            signal[t, j] = avg_prem * 10_000 + (avg_ratio - hist_ratio) * 5
        end
    end
    return signal
end

dp_signal = dark_pool_signal_matrix(dp_premium, dp_vol_ratio)

ic_dp = zeros(n_days)
for t in 10:n_days-5
    sig = dp_signal[t, :]
    fwd = fwd_5d[t, :]
    if std(sig) > 1e-12 && std(fwd) > 1e-12
        ic_dp[t] = cor(sig, fwd)
    end
end
valid_dp = ic_dp[ic_dp .!= 0]
println("Dark Pool Signal:")
println("  Mean IC: $(round(mean(valid_dp), digits=4))")
println("  IC IR:   $(round(mean(valid_dp)/std(valid_dp)*sqrt(252), digits=3)) annualized")

# ── Section 4: Social Sentiment ───────────────────────────

println("\n--- Section 4: Social Sentiment Signal ---")

function simulate_social_data(returns, n_days, n_stocks)
    sentiment = zeros(n_days, n_stocks)
    volume = zeros(n_days, n_stocks)
    for j in 1:n_stocks
        for t in 1:n_days
            base_vol = 1000.0 + 200.0 * next_randn()
            future_ret = t < n_days ? returns[t+1, j] : 0.0
            sentiment[t, j] = 0.4 * future_ret * 100 + 0.05 * next_randn()
            volume[t, j] = max(base_vol + 500 * abs(future_ret) * 50, 10.0)
        end
    end
    return sentiment, volume
end

soc_sent, soc_vol = simulate_social_data(stock_returns, n_days, n_stocks)

# Sentiment z-score signal
function sentiment_zscore(sent, vol, lookback=20)
    T, N = size(sent)
    signal = zeros(T, N)
    for t in lookback+1:T
        for j in 1:N
            hist_sent = sent[t-lookback:t-1, j]
            mu = mean(hist_sent)
            sig = std(hist_sent) + 1e-12
            signal[t, j] = (sent[t, j] - mu) / sig
            # Volume surge amplification
            hist_vol = vol[t-lookback:t-1, j]
            vol_zscore = (vol[t, j] - mean(hist_vol)) / (std(hist_vol) + 1e-12)
            signal[t, j] *= (1.0 + 0.3 * max(vol_zscore, 0.0))
        end
    end
    return signal
end

soc_signal = sentiment_zscore(soc_sent, soc_vol)
ic_soc = zeros(n_days)
for t in 22:n_days-5
    sig = soc_signal[t, :]
    fwd = fwd_5d[t, :]
    if std(sig) > 1e-12
        ic_soc[t] = cor(sig, fwd)
    end
end
valid_soc = ic_soc[ic_soc .!= 0]
println("Social Sentiment Signal:")
println("  Mean IC: $(round(mean(valid_soc), digits=4))")
println("  IC IR:   $(round(mean(valid_soc)/std(valid_soc)*sqrt(252), digits=3)) annualized")

# ── Section 5: Signal Combination ────────────────────────

println("\n--- Section 5: Signal Combination ---")

# Standardize signals to z-scores
function zscore_signal(signal_matrix, lookback=60)
    T, N = size(signal_matrix)
    result = zeros(T, N)
    for t in lookback+1:T
        for j in 1:N
            window = signal_matrix[t-lookback:t-1, j]
            mu = mean(window)
            sig_v = std(window) + 1e-12
            result[t, j] = clamp((signal_matrix[t, j] - mu) / sig_v, -3.0, 3.0)
        end
    end
    return result
end

# Use daily signals for combination
opt_z = zscore_signal(opt_signal)
dp_z  = zscore_signal(dp_signal)
soc_z = zscore_signal(soc_signal)

# Compute ICs for each signal over evaluation period
eval_start = 70  # after lookback warmup

function rolling_ic(signal, fwd_ret, window=40)
    T = size(signal, 1)
    ics = zeros(T)
    for t in window+1:T-5
        sig = signal[t, :]
        fwd = fwd_ret[t, :]
        valid = abs.(sig) .> 1e-12
        if sum(valid) >= 5
            ics[t] = cor(sig[valid], fwd[valid])
        end
    end
    return ics
end

ic_opt2  = rolling_ic(opt_z,  fwd_5d)
ic_dp2   = rolling_ic(dp_z,   fwd_5d)
ic_soc2  = rolling_ic(soc_z,  fwd_5d)

signals_combined = [opt_z, dp_z, soc_z]
signal_names = ["Options Flow", "Dark Pool", "Social Sentiment"]
ics_combined = [ic_opt2, ic_dp2, ic_soc2]

# Mean IC for equal-weight combination
combo_signal = (opt_z .+ dp_z .+ soc_z) ./ 3.0
ic_combo = rolling_ic(combo_signal, fwd_5d)

println("Signal IC Summary (mean over evaluation period):")
println("  Signal         | Mean IC  | IC t-stat")
println("  " * "-"^42)
for (name, ics) in zip(signal_names, ics_combined)
    valid_ic = ics[ics .!= 0]
    tstat = isempty(valid_ic) ? 0.0 : mean(valid_ic) / std(valid_ic) * sqrt(length(valid_ic))
    println("  $(lpad(name, 14)) | $(lpad(round(mean(valid_ic), digits=4), 8)) | $(round(tstat, digits=2))")
end
valid_combo = ic_combo[ic_combo .!= 0]
combo_tstat = isempty(valid_combo) ? 0.0 : mean(valid_combo)/std(valid_combo)*sqrt(length(valid_combo))
println("  $(lpad("Combo (equal)",14)) | $(lpad(round(mean(valid_combo),digits=4),8)) | $(round(combo_tstat, digits=2))")

# ── Section 6: Factor Orthogonalization ──────────────────

println("\n--- Section 6: Factor Orthogonalization ---")

# Simulate traditional factors (momentum, value, quality)
function simulate_factors(T, N)
    mom = zeros(T, N)
    val = zeros(T, N)
    qual = zeros(T, N)
    for j in 1:N
        for t in 1:T
            mom[t, j] = 0.5 * next_randn()
            val[t, j] = 0.3 * next_randn()
            qual[t, j] = 0.4 * next_randn()
        end
    end
    return mom, val, qual
end

mom_factor, val_factor, qual_factor = simulate_factors(n_days, n_stocks)

# Orthogonalize combo signal against factors
function orthogonalize(signal, factors_list)
    T, N = size(signal)
    residual = copy(signal)
    for t in 1:T
        y = signal[t, :]
        X = hcat([f[t, :] for f in factors_list]...)
        if size(X, 1) < size(X, 2) + 2; continue; end
        beta = (X'X + 1e-10*I) \ (X'y)
        residual[t, :] = y .- X * beta
    end
    return residual
end

factors_to_remove = [mom_factor, val_factor, qual_factor]
combo_ortho = orthogonalize(combo_signal, factors_to_remove)

ic_ortho = rolling_ic(combo_ortho, fwd_5d)
valid_ortho = ic_ortho[ic_ortho .!= 0]
ortho_tstat = isempty(valid_ortho) ? 0.0 : mean(valid_ortho)/std(valid_ortho)*sqrt(length(valid_ortho))

println("Before orthogonalization: IC=$(round(mean(valid_combo),digits=4)), t-stat=$(round(combo_tstat,digits=2))")
println("After orthogonalization:  IC=$(round(mean(valid_ortho),digits=4)), t-stat=$(round(ortho_tstat,digits=2))")
println("Signal decay from orthogonalization: $(round((1 - mean(valid_ortho)/max(abs(mean(valid_combo)),1e-12))*100, digits=1))%")

# ── Section 7: Backtest ───────────────────────────────────

println("\n--- Section 7: Alt-Data Strategy Backtest ---")

function backtest_ls_signal(signal, returns, n_long=5, n_short=5, tc_bps=10.0)
    T, N = size(signal)
    strategy_rets = zeros(T)
    turnover = zeros(T)
    prev_long = Int[]
    prev_short = Int[]

    for t in 1:T-1
        sig = signal[t, :]
        sorted_idx = sortperm(sig)
        n_valid = sum(abs.(sig) .> 1e-12)
        if n_valid < n_long + n_short; continue; end

        long_idx  = sorted_idx[end-n_long+1:end]
        short_idx = sorted_idx[1:n_short]

        fwd = returns[t+1, :]
        ls_ret = mean(fwd[long_idx]) - mean(fwd[short_idx])

        # Transaction cost from turnover
        new_long  = Set(long_idx)
        new_short = Set(short_idx)
        tover = length(symdiff(new_long, Set(prev_long))) / max(2*n_long, 1) +
                 length(symdiff(new_short, Set(prev_short))) / max(2*n_short, 1)
        tc = tover * tc_bps / 10_000.0
        strategy_rets[t] = ls_ret - tc

        prev_long  = long_idx
        prev_short = short_idx
        turnover[t] = tover
    end
    return strategy_rets, turnover
end

println("Long-short backtest (5L/5S, 10 bps TC):")
println("  Signal         | Ann Ret | Sharpe | Max DD | Turnover")
println("  " * "-"^55)

signals_to_test = [opt_z, dp_z, soc_z, combo_signal, combo_ortho]
names_to_test = ["Options Flow", "Dark Pool", "Social Sent", "Combo", "Combo Ortho"]

for (sig, name) in zip(signals_to_test, names_to_test)
    rets, tover = backtest_ls_signal(sig, stock_returns, 4, 4, 10.0)
    active = rets[rets .!= 0]
    if isempty(active); continue; end
    ann_ret = mean(active) * 252.0 * 100
    ann_std = std(active) * sqrt(252.0) * 100
    sharpe  = ann_std > 0 ? ann_ret / ann_std : 0.0
    cum = cumsum(rets)
    peak = -Inf; max_dd = 0.0
    for r in cum; peak = max(peak, r); max_dd = max(max_dd, peak - r); end
    avg_to = mean(tover[tover .> 0]) * 100
    println("  $(lpad(name, 14)) | $(lpad(round(ann_ret,digits=1),7))% | $(lpad(round(sharpe,digits=2),6)) | $(lpad(round(max_dd*100,digits=1),6))% | $(round(avg_to,digits=0))%/day")
end

println("\n✓ Notebook 29 complete")
