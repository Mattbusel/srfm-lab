module CryptoMechanics

# ============================================================
# CryptoMechanics.jl — Basis trading, funding rate arb,
#   cross-exchange spreads, calendar spreads, crypto derivatives
# ============================================================

using Statistics, LinearAlgebra

export BasisTrade, FundingRateData, ExchangeQuote, CalendarSpread
export compute_basis, annualized_basis, basis_carry
export funding_rate_signal, cumulative_funding, funding_arb_pnl
export cross_exchange_spread, spread_zscore, arb_opportunity
export calendar_spread_value, roll_yield, roll_cost
export perp_fair_value, perp_implied_rate, cash_carry_return
export optimal_hedge_ratio_basis, basis_risk
export funding_rate_predict, funding_momentum, oi_weighted_funding
export cross_margin_arb, triangular_arb_crypto
export liquidation_cascade_risk, open_interest_signal
export term_structure_crypto, term_premium
export delta_neutral_basis_pnl, simulate_basis_strategy
export slippage_adjusted_arb, net_arb_profit

# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

struct BasisTrade
    spot_price::Float64
    futures_price::Float64
    expiry_days::Float64
    financing_rate::Float64   # annualized
    storage_cost::Float64     # annualized (negligible for crypto, ~0)
end

struct FundingRateData
    timestamp::Int            # Unix timestamp
    funding_rate::Float64     # per 8-hour period
    mark_price::Float64
    index_price::Float64
    open_interest::Float64
end

struct ExchangeQuote
    exchange::String
    bid::Float64
    ask::Float64
    bid_size::Float64
    ask_size::Float64
    timestamp::Int
end

struct CalendarSpread
    near_expiry_days::Float64
    far_expiry_days::Float64
    near_price::Float64
    far_price::Float64
    spot_price::Float64
    risk_free_rate::Float64
end

# ──────────────────────────────────────────────────────────────
# Basis mechanics
# ──────────────────────────────────────────────────────────────

"""
    compute_basis(trade) -> basis_percent

Compute futures basis as percentage of spot price.
Basis = (Futures - Spot) / Spot
"""
function compute_basis(trade::BasisTrade)
    return (trade.futures_price - trade.spot_price) / trade.spot_price
end

"""
    annualized_basis(trade) -> annualized_basis_rate

Convert futures basis to annualized rate.
"""
function annualized_basis(trade::BasisTrade)
    raw_basis = compute_basis(trade)
    days = max(trade.expiry_days, 1.0)
    return raw_basis * 365.0 / days
end

"""
    basis_carry(trade) -> net_carry

Net carry from basis trade: annualized basis minus financing cost.
Positive = profitable cash-and-carry.
"""
function basis_carry(trade::BasisTrade)
    return annualized_basis(trade) - trade.financing_rate - trade.storage_cost
end

"""
    perp_fair_value(spot, funding_rate_8h, periods_elapsed) -> fair_value

Theoretical perpetual swap price given spot and funding accumulation.
"""
function perp_fair_value(spot::Float64, funding_rate_8h::Float64,
                           periods_elapsed::Float64=1.0)
    # Fair value converges to spot; funding is the cost of holding
    return spot * (1.0 + funding_rate_8h * periods_elapsed)
end

"""
    perp_implied_rate(mark_price, index_price) -> implied_funding_8h

Back out implied funding rate from perp vs index deviation.
Standard formula: clamp( (Mark - Index)/Index, -0.05%, 0.05% )
"""
function perp_implied_rate(mark_price::Float64, index_price::Float64)
    raw = (mark_price - index_price) / max(index_price, 1e-12)
    return clamp(raw, -0.0005, 0.0005)
end

"""
    cash_carry_return(spot_buy, futures_sell, expiry_days, financing_rate)

Return of a cash-and-carry trade (buy spot, sell futures at expiry).
"""
function cash_carry_return(spot_buy::Float64, futures_sell::Float64,
                             expiry_days::Float64, financing_rate::Float64)
    gross_return = (futures_sell - spot_buy) / spot_buy
    financing_cost = financing_rate * expiry_days / 365.0
    return gross_return - financing_cost
end

"""
    optimal_hedge_ratio_basis(spot_returns, futures_returns) -> hedge_ratio

Minimum variance hedge ratio for basis trading using OLS.
"""
function optimal_hedge_ratio_basis(spot_returns::Vector{Float64},
                                    futures_returns::Vector{Float64})
    if std(futures_returns) < 1e-12; return 1.0; end
    return cov(spot_returns, futures_returns) / var(futures_returns)
end

"""
    basis_risk(spot_returns, futures_returns, hedge_ratio) -> basis_risk_vol

Residual volatility (basis risk) of a hedged position.
"""
function basis_risk(spot_returns::Vector{Float64}, futures_returns::Vector{Float64},
                     hedge_ratio::Float64)
    hedged_pnl = spot_returns .- hedge_ratio .* futures_returns
    return std(hedged_pnl)
end

# ──────────────────────────────────────────────────────────────
# Funding rate signals and arbitrage
# ──────────────────────────────────────────────────────────────

"""
    funding_rate_signal(funding_data, lookback_periods) -> signal

Generate directional signal from funding rate history.
Persistently high (positive) funding → overheated longs → mean-reversion.
"""
function funding_rate_signal(data::Vector{FundingRateData}, lookback::Int=21)
    n = min(lookback, length(data))
    if n == 0; return 0.0; end
    recent = data[end-n+1:end]
    rates = [d.funding_rate for d in recent]
    avg_rate = mean(rates)
    std_rate = std(rates) + 1e-12
    # Positive signal when funding is unusually negative (longs being paid)
    # Negative signal when funding is unusually high (shorts profitable to hold)
    return -avg_rate / std_rate
end

"""
    cumulative_funding(data, horizon_periods) -> total_funding_return

Total funding paid/received over a horizon for a short position.
(Short perp = receive funding when funding is positive.)
"""
function cumulative_funding(data::Vector{FundingRateData}, horizon::Int=21)
    n = min(horizon, length(data))
    if n == 0; return 0.0; end
    recent = data[end-n+1:end]
    # Funding paid every 8 hours; compound
    value = 1.0
    for d in recent
        value *= (1.0 + d.funding_rate)
    end
    return value - 1.0
end

"""
    funding_arb_pnl(funding_data, spot_returns, perp_returns,
                    position_size, transaction_cost) -> pnl_series

PnL of delta-neutral funding rate arbitrage:
Long spot + Short perp, collect funding.
"""
function funding_arb_pnl(funding_data::Vector{FundingRateData},
                           spot_returns::Vector{Float64},
                           perp_returns::Vector{Float64},
                           position_size::Float64=1.0,
                           transaction_cost::Float64=0.0005)
    n = min(length(spot_returns), length(perp_returns), length(funding_data))
    pnl = zeros(n)
    for i in 1:n
        # Delta-neutral: long spot, short perp
        delta_pnl = (spot_returns[i] - perp_returns[i]) * position_size
        # Funding income (short perp receives positive funding)
        funding_income = funding_data[i].funding_rate * position_size
        pnl[i] = delta_pnl + funding_income
    end
    # Net of entry/exit transaction costs
    pnl[1] -= transaction_cost * position_size
    pnl[end] -= transaction_cost * position_size
    return pnl
end

"""
    funding_momentum(data, short_window, long_window) -> momentum_signal

Funding rate momentum: short-run funding minus long-run average.
"""
function funding_momentum(data::Vector{FundingRateData},
                            short_window::Int=7, long_window::Int=30)
    n = length(data)
    if n < short_window; return 0.0; end
    short_avg = mean(d.funding_rate for d in data[end-min(short_window,n)+1:end])
    long_avg = mean(d.funding_rate for d in data[end-min(long_window,n)+1:end])
    return short_avg - long_avg
end

"""
    oi_weighted_funding(funding_data) -> weighted_rate

Open-interest-weighted average funding rate.
"""
function oi_weighted_funding(data::Vector{FundingRateData})
    if isempty(data); return 0.0; end
    total_oi = sum(d.open_interest for d in data)
    if total_oi < 1e-12
        return mean(d.funding_rate for d in data)
    end
    return sum(d.funding_rate * d.open_interest for d in data) / total_oi
end

# ──────────────────────────────────────────────────────────────
# Cross-exchange spread arbitrage
# ──────────────────────────────────────────────────────────────

"""
    cross_exchange_spread(quote_a, quote_b) -> (spread, direction)

Compute cross-exchange spread.
Returns: spread in bps, direction (:buy_a_sell_b or :buy_b_sell_a).
"""
function cross_exchange_spread(qa::ExchangeQuote, qb::ExchangeQuote)
    # Buy on A, sell on B
    spread_ab = (qb.bid - qa.ask) / qa.ask
    # Buy on B, sell on A
    spread_ba = (qa.bid - qb.ask) / qb.ask
    if spread_ab > spread_ba
        return spread_ab, :buy_a_sell_b
    else
        return spread_ba, :buy_b_sell_a
    end
end

"""
    spread_zscore(spread_history, current_spread, lookback) -> z

Z-score of current spread relative to recent history.
"""
function spread_zscore(history::Vector{Float64}, current::Float64, lookback::Int=100)
    n = min(lookback, length(history))
    if n < 5; return 0.0; end
    window = history[end-n+1:end]
    mu = mean(window)
    sigma = std(window) + 1e-12
    return (current - mu) / sigma
end

"""
    arb_opportunity(quotes, min_spread_bps, max_execution_time_ms) -> (is_arb, profit_bps)

Identify and quantify a cross-exchange arbitrage opportunity.
"""
function arb_opportunity(quotes::Vector{ExchangeQuote},
                           min_spread_bps::Float64=3.0,
                           max_latency_ms::Float64=200.0)
    n = length(quotes)
    best_bid = -Inf
    best_ask = Inf
    best_bid_exchange = ""
    best_ask_exchange = ""

    for q in quotes
        if q.bid > best_bid
            best_bid = q.bid
            best_bid_exchange = q.exchange
        end
        if q.ask < best_ask
            best_ask = q.ask
            best_ask_exchange = q.exchange
        end
    end

    if best_bid_exchange == best_ask_exchange; return false, 0.0; end
    spread_bps = (best_bid - best_ask) / max(best_ask, 1e-12) * 10_000.0
    is_arb = spread_bps >= min_spread_bps
    return is_arb, spread_bps
end

"""
    slippage_adjusted_arb(raw_spread_bps, size, avg_daily_volume,
                           market_impact_coeff) -> net_spread_bps

Adjust arb spread for market impact and slippage.
"""
function slippage_adjusted_arb(raw_spread::Float64, size::Float64,
                                  daily_volume::Float64,
                                  impact_coeff::Float64=0.1)
    participation_rate = size / max(daily_volume, 1e-12)
    slippage_bps = impact_coeff * sqrt(participation_rate) * 10_000.0
    return raw_spread - 2.0 * slippage_bps  # slippage on both legs
end

"""
    net_arb_profit(spread_bps, position_size, transaction_cost_bps,
                   financing_cost_bps) -> profit_usd

Net dollar profit from a cross-exchange arb.
"""
function net_arb_profit(spread_bps::Float64, position_size::Float64,
                          tx_cost_bps::Float64=2.0, financing_bps::Float64=0.5)
    net_bps = spread_bps - tx_cost_bps - financing_bps
    return net_bps / 10_000.0 * position_size
end

# ──────────────────────────────────────────────────────────────
# Calendar spreads and term structure
# ──────────────────────────────────────────────────────────────

"""
    calendar_spread_value(cs) -> spread_value

Value of a calendar spread (long near, short far or vice versa).
"""
function calendar_spread_value(cs::CalendarSpread)
    return cs.far_price - cs.near_price
end

"""
    roll_yield(cs) -> annualized_roll

Annualized roll yield from rolling a futures position from near to far contract.
Negative when term structure is in contango (far > near).
"""
function roll_yield(cs::CalendarSpread)
    spread = cs.near_price - cs.far_price
    time_diff = (cs.far_expiry_days - cs.near_expiry_days) / 365.0
    return (spread / cs.near_price) / max(time_diff, 1e-12 / 365.0)
end

"""
    roll_cost(cs, contract_size, roll_frequency_days) -> annual_roll_cost

Annual cost of rolling futures contracts.
"""
function roll_cost(cs::CalendarSpread, contract_size::Float64=1.0,
                    roll_freq_days::Float64=30.0)
    roll_per_period = abs(cs.far_price - cs.near_price) / cs.near_price
    periods_per_year = 365.0 / roll_freq_days
    return roll_per_period * periods_per_year * contract_size * cs.near_price
end

"""
    term_structure_crypto(expiries_days, futures_prices, spot_price) -> (implied_rates, slope)

Extract implied interest rate term structure from futures prices.
F = S * exp(r * T) => r = ln(F/S) / T
"""
function term_structure_crypto(expiries::Vector{Float64}, futures_prices::Vector{Float64},
                                  spot_price::Float64)
    n = length(expiries)
    implied_rates = zeros(n)
    for i in 1:n
        T = expiries[i] / 365.0
        if T > 0 && futures_prices[i] > 0 && spot_price > 0
            implied_rates[i] = log(futures_prices[i] / spot_price) / T
        end
    end
    # Fit slope via OLS
    if n >= 2
        X = hcat(ones(n), expiries)
        beta = (X'X + 1e-10*I) \ (X' * implied_rates)
        slope = beta[2]
    else
        slope = 0.0
    end
    return implied_rates, slope
end

"""
    term_premium(short_rate, long_rate, horizon_days) -> premium

Crypto term premium: long-maturity rate minus short-maturity rate.
"""
function term_premium(short_rate::Float64, long_rate::Float64,
                        horizon_days::Float64=90.0)
    return long_rate - short_rate
end

# ──────────────────────────────────────────────────────────────
# Open interest and liquidation signals
# ──────────────────────────────────────────────────────────────

"""
    open_interest_signal(oi_series, price_series, lookback) -> signal

Combine OI and price direction for sentiment signal.
OI rising + price rising = bullish confirmation.
OI rising + price falling = bearish divergence.
"""
function open_interest_signal(oi::Vector{Float64}, prices::Vector{Float64},
                                lookback::Int=14)
    n = min(lookback, min(length(oi), length(prices)))
    if n < 2; return 0.0; end
    oi_change = (oi[end] - oi[end-n+1]) / max(oi[end-n+1], 1e-12)
    price_change = (prices[end] - prices[end-n+1]) / max(prices[end-n+1], 1e-12)
    # Both same sign = trending; opposite sign = divergence
    return oi_change * price_change
end

"""
    liquidation_cascade_risk(open_interest, leverage_distribution,
                               volatility, liquidation_threshold) -> cascade_risk_score

Estimate risk of a liquidation cascade.
leverage_distribution: histogram of leverage levels [2x, 5x, 10x, 20x, ...]
"""
function liquidation_cascade_risk(open_interest::Float64,
                                    leverage_levels::Vector{Float64},
                                    leverage_weights::Vector{Float64},
                                    volatility::Float64,
                                    threshold::Float64=0.01)
    # Fraction of positions liquidated = fraction with liquidation price within threshold
    # Liquidation distance = 1/leverage
    n = length(leverage_levels)
    liquidated_fraction = 0.0
    for (i, lev) in enumerate(leverage_levels)
        liq_distance = 1.0 / max(lev, 1.0)
        # Probability price hits liquidation level given current vol
        # P(|ΔP/P| >= liq_distance) ≈ 2*Φ(-liq_distance/vol) [normal approximation]
        z = liq_distance / max(volatility, 1e-12)
        prob_liq = 2.0 * (0.5 * erfc(z / sqrt(2.0)))
        liquidated_fraction += leverage_weights[i] * prob_liq
    end
    # erfc approximation
    function erfc_approx(x)
        t = 1.0 / (1.0 + 0.3275911 * x)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return poly * exp(-x^2)
    end

    liquidated_oi = open_interest * liquidated_fraction
    # Cascade multiplier: liquidation selling suppresses price further
    cascade_multiplier = 1.0 + 0.1 * log1p(liquidated_oi / max(open_interest, 1e-12))
    return liquidated_fraction * cascade_multiplier
end

# ──────────────────────────────────────────────────────────────
# Triangular arbitrage
# ──────────────────────────────────────────────────────────────

"""
    triangular_arb_crypto(price_ab, price_bc, price_ac) -> (profit_fraction, direction)

Triangular arbitrage between three crypto pairs.
e.g., BTC/USD, ETH/USD, ETH/BTC
"""
function triangular_arb_crypto(price_ab::Float64, price_bc::Float64, price_ac::Float64)
    # Path 1: buy A with C, sell A for B, sell B for C
    # A -> B at price_ab, B -> C at price_bc, C -> A at 1/price_ac
    path1 = price_ab * price_bc / price_ac - 1.0
    # Path 2: reverse
    path2 = price_ac / (price_ab * price_bc) - 1.0
    if path1 > path2
        return path1, :path1
    else
        return path2, :path2
    end
end

"""
    cross_margin_arb(margin_rate_a, margin_rate_b, position_size) -> annualized_spread

Arbitrage between different margin lending rates across exchanges.
"""
function cross_margin_arb(rate_a::Float64, rate_b::Float64,
                            position_size::Float64=1.0)
    # Borrow on cheaper exchange, lend on expensive
    spread = abs(rate_a - rate_b)
    return spread * position_size
end

# ──────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────

"""
    delta_neutral_basis_pnl(spot_prices, futures_prices, funding_rates,
                             entry_basis, exit_basis, transaction_cost)

Simulate P&L of a delta-neutral basis trade over a price series.
"""
function delta_neutral_basis_pnl(spot_prices::Vector{Float64},
                                   futures_prices::Vector{Float64},
                                   funding_rates::Vector{Float64},
                                   entry_basis::Float64=0.02,
                                   exit_basis::Float64=0.005,
                                   transaction_cost::Float64=0.001)
    n = length(spot_prices)
    in_trade = false
    entry_idx = 0
    total_pnl = 0.0
    trade_count = 0
    pnl_series = zeros(n)

    for i in 1:n
        current_basis = (futures_prices[i] - spot_prices[i]) / spot_prices[i]

        if !in_trade && current_basis >= entry_basis
            in_trade = true
            entry_idx = i
            total_pnl -= transaction_cost
        end

        if in_trade
            # Collect funding (short futures)
            idx = min(i, length(funding_rates))
            total_pnl += funding_rates[idx]
            pnl_series[i] = total_pnl

            if current_basis <= exit_basis || i == n
                in_trade = false
                total_pnl -= transaction_cost
                trade_count += 1
            end
        end
    end

    ann_return = total_pnl * 365.0 / max(n, 1)
    return pnl_series, total_pnl, ann_return, trade_count
end

"""
    simulate_basis_strategy(S0, mu, sigma, funding_mean, funding_std,
                             T_days, n_paths) -> (mean_return, sharpe, max_dd)

Monte Carlo simulation of basis trading strategy.
"""
function simulate_basis_strategy(S0::Float64=40000.0, mu::Float64=0.0,
                                   sigma::Float64=0.60, funding_mean::Float64=0.0001,
                                   funding_std::Float64=0.0003,
                                   T_days::Int=365, n_paths::Int=500)
    state = UInt64(42)
    function randz()
        state = state * 6364136223846793005 + 1442695040888963407
        u1 = max((state >> 11) / Float64(2^53), 1e-15)
        state = state * 6364136223846793005 + 1442695040888963407
        u2 = (state >> 11) / Float64(2^53)
        return sqrt(-2.0 * log(u1)) * cos(2π * u2)
    end

    path_returns = zeros(n_paths)
    for p in 1:n_paths
        cum_pnl = 0.0
        S = S0
        for t in 1:T_days
            # Spot return
            S *= exp((mu/365 - 0.5*(sigma/sqrt(365))^2) + sigma/sqrt(365) * randz())
            # Funding income (short perp, long spot)
            funding = funding_mean + funding_std * randz() * sqrt(1/3)  # 3 per day
            cum_pnl += funding * 3.0  # 3 periods per day
        end
        path_returns[p] = cum_pnl
    end

    mean_ret = mean(path_returns)
    ann_std = std(path_returns)
    sharpe = ann_std > 1e-12 ? mean_ret / ann_std : 0.0

    # Max drawdown
    cum = cumsum(path_returns)
    peak = -Inf; max_dd = 0.0
    for r in sort(path_returns, rev=true)[1:min(10,n_paths)]
        peak = max(peak, r)
        max_dd = max(max_dd, peak - r)
    end

    return mean_ret, sharpe, max_dd
end

end # module CryptoMechanics
