# ============================================================
# Notebook 31: Crypto Basis & Funding Rate Arbitrage
# ============================================================
# Topics:
#   1. BTC/ETH futures basis analysis
#   2. Funding rate dynamics and persistence
#   3. Delta-neutral basis trade construction
#   4. Cross-exchange spread arbitrage
#   5. Calendar spread trading
#   6. Perpetual vs fixed expiry arb
#   7. Liquidation cascade risk
#   8. Strategy simulation and risk analysis
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 31: Crypto Basis & Funding Rate Trading")
println("="^60)

# ── RNG ───────────────────────────────────────────────────
rng_s = UInt64(42)
function rnd()
    global rng_s
    rng_s = rng_s * 6364136223846793005 + 1442695040888963407
    (rng_s >> 11) / Float64(2^53)
end
function rndn()
    u1 = max(rnd(), 1e-15); u2 = rnd()
    sqrt(-2.0*log(u1)) * cos(2π*u2)
end

# ── Section 1: Futures Basis Analysis ────────────────────

println("\n--- Section 1: BTC Futures Basis ---")

# Simulate BTC price and futures prices at different expiries
n_days = 365
S0 = 45000.0
mu_btc = 0.20     # 20% annual drift
sigma_btc = 0.80  # 80% annual vol
r_btc = 0.05      # BTC financing rate

# Generate spot price path
btc_prices = zeros(n_days)
btc_prices[1] = S0
for t in 2:n_days
    btc_prices[t] = btc_prices[t-1] * exp((mu_btc/365 - 0.5*(sigma_btc/sqrt(365))^2) +
                                            sigma_btc/sqrt(365) * rndn())
end

# Futures prices with term structure + noise
function futures_price(spot, T_years, r, convenience_yield=0.0, noise=0.0)
    fair = spot * exp((r - convenience_yield) * T_years)
    return fair * (1.0 + noise)
end

expiries_days = [7, 14, 28, 90, 180, 365]
expiries_years = expiries_days ./ 365.0

# Compute basis for each expiry at each day
println("Term structure at day 1 (S=\$$(round(btc_prices[1],digits=0))):")
println("  Expiry | Futures   | Basis   | Ann. Basis | vs Risk-free")
println("  " * "-"^55)
for (d, T) in zip(expiries_days, expiries_years)
    noise = 0.005 * rndn()  # ±50 bps noise
    F = futures_price(btc_prices[1], T, r_btc, 0.0, noise)
    basis_pct = (F - btc_prices[1]) / btc_prices[1] * 100
    ann_basis = basis_pct / T
    vs_rf = ann_basis - r_btc * 100
    println("  $(lpad(d,6))d | \$$(lpad(round(F,digits=0),8)) | $(lpad(round(basis_pct,digits=2),7))% | " *
            "$(lpad(round(ann_basis,digits=2),10))% | $(round(vs_rf,digits=2))%")
end

# Rolling basis for quarterly contract
quarterly_basis = zeros(n_days)
for t in 1:n_days
    T_remaining = max((90 - (t % 90)) / 365.0, 1.0/365.0)
    F_q = futures_price(btc_prices[t], T_remaining, r_btc, 0.0, 0.003 * rndn())
    quarterly_basis[t] = (F_q - btc_prices[t]) / btc_prices[t] * 365.0 / T_remaining * 100
end

println("\nQuarterly basis statistics over $(n_days) days:")
println("  Mean ann. basis: $(round(mean(quarterly_basis),digits=2))%")
println("  Std:             $(round(std(quarterly_basis),digits=2))%")
println("  Max:             $(round(maximum(quarterly_basis),digits=2))%")
println("  Min:             $(round(minimum(quarterly_basis),digits=2))%")
println("  % above 10%:     $(round(mean(quarterly_basis .> 10.0)*100, digits=1))%")
println("  % above 20%:     $(round(mean(quarterly_basis .> 20.0)*100, digits=1))%")

# ── Section 2: Funding Rate Dynamics ─────────────────────

println("\n--- Section 2: Perpetual Funding Rates ---")

# Simulate funding rates (paid every 8 hours, 3x per day)
n_periods = n_days * 3  # 8-hour funding periods
funding_rates = zeros(n_periods)
# Funding mean-reverts to a positive value when price is trending up
funding_mean = 0.0001   # 0.01% per 8h = ~11% annualized
funding_kappa = 0.05    # mean reversion speed
funding_vol = 0.0003

funding_rates[1] = funding_mean
for t in 2:n_periods
    # Mean-reverting + drift linked to price momentum
    price_t = btc_prices[min((t-1) ÷ 3 + 1, n_days)]
    price_lag = btc_prices[max((t-5) ÷ 3 + 1, 1)]
    price_mom = (price_t - price_lag) / price_lag
    drift = funding_kappa * (funding_mean + 0.5 * price_mom - funding_rates[t-1])
    funding_rates[t] = funding_rates[t-1] + drift + funding_vol * rndn()
    funding_rates[t] = clamp(funding_rates[t], -0.005, 0.005)
end

ann_funding = funding_rates .* 3 .* 365 .* 100

println("Funding rate statistics:")
println("  Mean (per 8h):        $(round(mean(funding_rates)*100, digits=4))%")
println("  Ann. equivalent:      $(round(mean(ann_funding), digits=2))%")
println("  % positive periods:   $(round(mean(funding_rates .> 0)*100, digits=1))%")
println("  % above 0.05%/8h:     $(round(mean(funding_rates .> 0.0005)*100, digits=1))%")
println("  Autocorrelation(1):   $(round(cor(funding_rates[1:end-1], funding_rates[2:end]), digits=4))")
println("  Autocorrelation(3):   $(round(cor(funding_rates[1:end-3], funding_rates[4:end]), digits=4))")

# Funding signal: predict funding from recent history
function funding_signal(rates, lookback_short=7, lookback_long=30)
    n = length(rates)
    signal = zeros(n)
    for t in max(lookback_long+1, 1):n
        short_avg = mean(rates[t-lookback_short+1:t])
        long_avg = mean(rates[t-lookback_long+1:t])
        signal[t] = short_avg - long_avg
    end
    return signal
end

fund_signal = funding_signal(funding_rates)
# IC of signal with next-period funding
fund_ic = cor(fund_signal[32:end-1], funding_rates[33:end])
println("  Funding momentum IC:  $(round(fund_ic, digits=4))")

# ── Section 3: Delta-Neutral Basis Trade ─────────────────

println("\n--- Section 3: Delta-Neutral Basis Trade ---")

# Strategy: Long spot + Short perp at daily settlement
# Revenue: funding income
# Risk: basis moves, margin requirements

initial_capital = 1_000_000.0  # $1M
leverage = 2.0
position_btc = initial_capital / btc_prices[1]
transaction_cost = 0.001  # 10 bps each side

# Simulate daily P&L
daily_pnl = zeros(n_days)
daily_funding_pnl = zeros(n_days)
daily_basis_pnl = zeros(n_days)

for t in 2:n_days
    # Funding income (3 payments per day, short perp → receive positive funding)
    t_start = (t-2) * 3 + 1
    t_end = (t-1) * 3
    daily_fund = sum(funding_rates[min(t_start:t_end, 1:n_periods)])
    funding_income = daily_fund * position_btc * btc_prices[t]
    daily_funding_pnl[t] = funding_income

    # Basis P&L: long spot, short perp
    spot_return = (btc_prices[t] - btc_prices[t-1]) / btc_prices[t-1]
    # Perp price slightly deviates from spot
    perp_premium = 0.0001 * rndn()  # small random premium
    perp_return = spot_return + perp_premium
    basis_pnl = (spot_return - perp_return) * position_btc * btc_prices[t-1]
    daily_basis_pnl[t] = basis_pnl

    daily_pnl[t] = funding_income + basis_pnl
end

# Subtract entry/exit costs
daily_pnl[2] -= transaction_cost * initial_capital
daily_pnl[end] -= transaction_cost * initial_capital

cum_pnl = cumsum(daily_pnl)
total_return = cum_pnl[end] / initial_capital * 100
ann_return = total_return / n_days * 365
sharpe = std(daily_pnl[2:end]) > 0 ?
    mean(daily_pnl[2:end]) / std(daily_pnl[2:end]) * sqrt(252) : 0.0

println("Delta-neutral basis trade results (1 year):")
println("  Funding income:      \$$(round(sum(daily_funding_pnl)/1000, digits=1))K")
println("  Basis P&L:           \$$(round(sum(daily_basis_pnl)/1000, digits=1))K")
println("  Total P&L:           \$$(round(cum_pnl[end]/1000, digits=1))K")
println("  Annualized return:   $(round(ann_return, digits=2))%")
println("  Sharpe ratio:        $(round(sharpe, digits=3))")

# Drawdown
peak = -Inf; max_dd = 0.0
for r in cum_pnl
    peak = max(peak, r)
    max_dd = max(max_dd, peak - r)
end
println("  Max drawdown:        \$$(round(max_dd/1000, digits=1))K ($(round(max_dd/initial_capital*100, digits=2))%)")

# ── Section 4: Cross-Exchange Arb ─────────────────────────

println("\n--- Section 4: Cross-Exchange Spread Arbitrage ---")

# Simulate bid/ask on 3 exchanges
exchanges = ["Binance", "Bybit", "OKX"]
n_snapshots = 10000
spread_data = zeros(n_snapshots, 3)  # bid-ask spreads
price_diff = zeros(n_snapshots, 3)   # price vs reference

for t in 1:n_snapshots
    for j in 1:3
        # Price difference from reference (random walk with mean-reversion)
        spread_data[t, j] = 5.0 + 3.0 * abs(rndn())  # bps spread
        price_diff[t, j]  = 10.0 * rndn()  # bps price difference
    end
end

# Identify arbitrage opportunities (cross-exchange spread > 10 bps)
arb_threshold_bps = 10.0
n_arb_opps = 0
arb_profits = Float64[]
for t in 1:n_snapshots
    # Best bid on each exchange: reference + price_diff/2
    bids = price_diff[t, :] ./ 2
    asks = price_diff[t, :] ./ 2 .+ spread_data[t, :]
    best_bid_idx = argmax(bids)
    best_ask_idx = argmin(asks)
    if best_bid_idx != best_ask_idx
        spread_bps = (bids[best_bid_idx] - asks[best_ask_idx])
        if spread_bps >= arb_threshold_bps
            n_arb_opps += 1
            # Net after execution costs (5 bps each side)
            net_profit = (spread_bps - 10.0) / 10_000.0 * 100_000  # $100K position
            push!(arb_profits, net_profit)
        end
    end
end

println("Cross-exchange arb analysis ($n_snapshots snapshots):")
println("  Opportunities (>10 bps spread): $(n_arb_opps) ($(round(n_arb_opps/n_snapshots*100,digits=2))%)")
if !isempty(arb_profits)
    println("  Avg profit per arb:             \$$(round(mean(arb_profits),digits=2))")
    println("  Median profit:                  \$$(round(median(arb_profits),digits=2))")
    daily_arbs = n_arb_opps / n_snapshots * 3 * 390  # rough daily count
    println("  Est. daily arb income:          \$$(round(daily_arbs * mean(arb_profits), digits=0))")
end

# ── Section 5: Calendar Spread Trading ───────────────────

println("\n--- Section 5: Calendar Spread Analysis ---")

# Calendar spread = nearby futures - far futures
# Positive = backwardation (near > far), negative = contango

# Simulate spread dynamics
n_rolls = 12  # monthly rolls
println("Calendar spread returns by regime:")

contango_spreads = Float64[]
backwardation_spreads = Float64[]

for m in 1:n_rolls
    t_idx = round(Int, (m-0.5) * n_days / n_rolls)
    S = btc_prices[min(t_idx, n_days)]
    T_near = 30.0 / 365.0
    T_far = 90.0 / 365.0
    # Compute near and far futures with stochastic basis
    near_basis_noise = 0.003 * rndn()
    far_basis_noise = 0.005 * rndn()
    F_near = S * exp(r_btc * T_near) * (1 + near_basis_noise)
    F_far  = S * exp(r_btc * T_far) * (1 + far_basis_noise)

    # Calendar spread return: long near, short far
    spread_return = (F_near / S - F_far / S)
    # Roll cost: losing the time value when near expires
    roll_cost = abs(F_far - F_near) / S

    println("  Month $m: S=\$$(round(S,digits=0)), F_near=\$$(round(F_near,digits=0)), " *
            "F_far=\$$(round(F_far,digits=0)), Spread=$(round(spread_return*100,digits=3))%")

    if spread_return > 0; push!(backwardation_spreads, spread_return)
    else; push!(contango_spreads, spread_return); end
end

println("\nSpread regime summary:")
println("  Contango months:      $(length(contango_spreads))/$(n_rolls)")
println("  Backwardation months: $(length(backwardation_spreads))/$(n_rolls)")
if !isempty(contango_spreads)
    println("  Avg contango spread:  $(round(mean(contango_spreads)*100, digits=3))%")
end

# ── Section 6: Risk Management ────────────────────────────

println("\n--- Section 6: Basis Trade Risk Management ---")

# VaR and CVaR of the basis trade
daily_returns = [daily_pnl[t] / initial_capital for t in 2:n_days]
sort!(daily_returns)
n_ret = length(daily_returns)
var_1pct  = daily_returns[round(Int, 0.01 * n_ret)]
var_5pct  = daily_returns[round(Int, 0.05 * n_ret)]
cvar_1pct = mean(daily_returns[1:max(1,round(Int, 0.01 * n_ret))])
cvar_5pct = mean(daily_returns[1:max(1,round(Int, 0.05 * n_ret))])

println("Risk metrics:")
println("  VaR (1%):  $(round(var_1pct*100, digits=4))% → \$$(round(var_1pct*initial_capital, digits=0))")
println("  VaR (5%):  $(round(var_5pct*100, digits=4))% → \$$(round(var_5pct*initial_capital, digits=0))")
println("  CVaR (1%): $(round(cvar_1pct*100, digits=4))% → \$$(round(cvar_1pct*initial_capital, digits=0))")
println("  CVaR (5%): $(round(cvar_5pct*100, digits=4))% → \$$(round(cvar_5pct*initial_capital, digits=0))")

# Margin requirements
maintenance_margin = 0.05  # 5% of position value
position_value = position_btc * btc_prices[end]
required_margin = maintenance_margin * position_value
excess_capital = initial_capital - required_margin
println("\nMargin analysis (at current BTC price):")
println("  Position value:  \$$(round(position_value/1e6, digits=2))M")
println("  Required margin: \$$(round(required_margin/1e3, digits=0))K")
println("  Excess capital:  \$$(round(excess_capital/1e3, digits=0))K")
println("  Available yield: $(round(sum(daily_funding_pnl)/initial_capital*100, digits=2))%")

# ── Section 7: Liquidation Cascade Risk ──────────────────

println("\n--- Section 7: Liquidation Cascade Risk ---")

# Model OI distribution by leverage
oi_total = 5e9  # $5B open interest
leverage_buckets = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
oi_weights = [0.30, 0.25, 0.20, 0.15, 0.07, 0.03]  # distribution

println("Liquidation risk analysis:")
println("  Price Drop | Liq'd OI (\$B) | Cascade Mult | Total Impact")
println("  " * "-"^50)

function norm_cdf_approx(x)
    sign_x = x >= 0 ? 1.0 : -1.0
    ax = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return (1.0 - sign_x * p * exp(-ax^2)) / 2.0 + (sign_x > 0 ? 0.0 : 0.0)
end

for price_drop in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    liq_frac = 0.0
    for (lev, wt) in zip(leverage_buckets, oi_weights)
        liq_dist = 1.0 / lev
        if price_drop >= liq_dist
            liq_frac += wt
        end
    end
    liq_oi = oi_total * liq_frac / 1e9
    # Cascade: liquidation selling creates further 0.5x amplification
    cascade_mult = 1.0 + 0.5 * liq_frac
    total_impact = price_drop * cascade_mult * 100
    println("  $(lpad(round(price_drop*100,digits=0),9))%  | $(lpad(round(liq_oi,digits=2),13)) | $(lpad(round(cascade_mult,digits=2),12)) | $(round(total_impact, digits=1))%")
end

# ── Section 8: Full Strategy Backtest ─────────────────────

println("\n--- Section 8: Strategy Summary ---")

ann_funding_yield = sum(daily_funding_pnl) / initial_capital * 365.0 / n_days * 100
ann_basis_pnl = sum(daily_basis_pnl) / initial_capital * 365.0 / n_days * 100

println("Annual performance attribution:")
println("  Funding carry:     $(round(ann_funding_yield, digits=2))%")
println("  Basis P&L:         $(round(ann_basis_pnl, digits=2))%")
println("  Transaction costs: $(round(-2 * transaction_cost * 100, digits=2))%")
println("  Net return:        $(round(ann_return, digits=2))%")
println("  Sharpe ratio:      $(round(sharpe, digits=3))")
println("  Max drawdown:      $(round(max_dd / initial_capital * 100, digits=2))%")
println("  Calmar ratio:      $(round(ann_return / max(max_dd/initial_capital*100, 0.01), digits=3))")

println("\n✓ Notebook 31 complete")
