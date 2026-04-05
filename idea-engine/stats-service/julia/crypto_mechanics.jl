module CryptoMechanics

# ============================================================
# crypto_mechanics.jl -- Crypto Trading Mechanics & Strategies
# ============================================================
# Covers: funding rate arbitrage, cash-and-carry basis trading,
# cross-exchange spread capture, perpetual-spot convergence,
# calendar spreads, options basis, delta-neutral hedging,
# liquidation cascade risk, PnL attribution, position sizing,
# multi-leg strategy construction, Greeks for crypto options.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

struct PerpContract
    symbol::String
    mark_price::Float64
    index_price::Float64
    funding_rate_8h::Float64
    open_interest::Float64
    volume_24h::Float64
    max_leverage::Float64
end

struct SpotMarket
    symbol::String
    bid::Float64
    ask::Float64
    last::Float64
    volume_24h::Float64
    exchange::String
end

struct FundingArb
    long_spot::Float64
    short_perp::Float64
    net_notional::Float64
    funding_rate_8h::Float64
    holding_days::Float64
    borrow_rate_annual::Float64
end

struct CalendarSpread
    near_symbol::String
    far_symbol::String
    near_price::Float64
    far_price::Float64
    near_expiry_days::Int
    far_expiry_days::Int
    basis_pct::Float64
end

struct CrossExchangeSpread
    buy_exchange::String
    sell_exchange::String
    buy_price::Float64
    sell_price::Float64
    transfer_fee::Float64
    transfer_time_hrs::Float64
    spread_pct::Float64
end

struct CryptoOption
    underlying::String
    strike::Float64
    expiry_days::Int
    call_or_put::Symbol
    iv::Float64
    premium::Float64
    delta::Float64
    gamma::Float64
    vega::Float64
    theta::Float64
end

struct LiquidationLevel
    position_size::Float64
    entry_price::Float64
    leverage::Float64
    side::Symbol
    maintenance_margin::Float64
    liq_price::Float64
end

struct FundingRateHistory
    timestamps::Vector{Float64}
    rates_8h::Vector{Float64}
    mark_prices::Vector{Float64}
    index_prices::Vector{Float64}
end

# ---- 1. Funding Rate Analysis ----

function funding_rate_premium(mark::Float64, index::Float64)::Float64
    return (mark - index) / (index + 1e-8)
end

function funding_8h_annualised(rate_8h::Float64)::Float64
    return rate_8h * 3 * 365 * 100.0
end

function funding_arb_pnl(arb::FundingArb)::NamedTuple
    daily_funding = arb.funding_rate_8h * 3 * arb.net_notional
    total_funding = daily_funding * arb.holding_days
    borrow_cost   = arb.net_notional * arb.borrow_rate_annual * arb.holding_days / 365.0
    net_pnl       = total_funding - borrow_cost
    ann_yield     = net_pnl / (arb.net_notional + 1e-8) * (365.0 / arb.holding_days) * 100.0
    breakeven_fr  = arb.borrow_rate_annual / (3*365*100)
    return (
        daily_funding_income = daily_funding,
        total_funding_income = total_funding,
        borrow_cost          = borrow_cost,
        net_pnl              = net_pnl,
        annualised_yield_pct = ann_yield,
        breakeven_rate_8h    = breakeven_fr,
        is_profitable        = net_pnl > 0,
    )
end

function funding_rate_signal(hist::FundingRateHistory, lookback::Int=30)
    n = length(hist.rates_8h)
    recent = hist.rates_8h[max(1,n-lookback*3):n]
    avg_rate = mean(recent)
    std_rate = std(recent) + 1e-8
    latest = hist.rates_8h[end]
    z = (latest - avg_rate) / std_rate
    signal = z > 2 ? :extreme_longs_paying : z < -2 ? :extreme_shorts_paying : :normal
    cumulative_annual = sum(recent) * 3 * 365 / length(recent) * 100.0
    return (
        latest_rate = latest,
        avg_rate    = avg_rate,
        z_score     = z,
        signal      = signal,
        implied_annual_pct = cumulative_annual,
        percentile  = count(r -> r < latest, recent) / length(recent) * 100.0,
    )
end

function optimal_funding_entry(hist::FundingRateHistory, z_threshold::Float64=2.0)
    n = length(hist.rates_8h)
    entries = Int[]
    rates = hist.rates_8h
    for i in 21:n
        window = rates[i-20:i-1]
        z = (rates[i] - mean(window)) / (std(window) + 1e-8)
        if z > z_threshold
            push!(entries, i)
        end
    end
    return entries
end

# ---- 2. Basis Trading ----

function perp_spot_basis(perp_price::Float64, spot_price::Float64)::Float64
    return (perp_price - spot_price) / spot_price * 100.0
end

function basis_trade_pnl(perp_price::Float64, spot_price::Float64,
                          size::Float64, days::Float64,
                          funding_rate_daily::Float64, exit_basis_pct::Float64)
    entry_basis = perp_spot_basis(perp_price, spot_price)
    funding_income = funding_rate_daily * perp_price * size * days
    basis_convergence = (entry_basis - exit_basis_pct) / 100.0 * spot_price * size
    total_pnl = funding_income + basis_convergence
    ann_yield = total_pnl / (spot_price * size + 1e-8) * (365.0/days) * 100.0
    return (
        entry_basis_pct    = entry_basis,
        funding_income     = funding_income,
        basis_pnl          = basis_convergence,
        total_pnl          = total_pnl,
        annualised_yield   = ann_yield,
    )
end

function carry_adjusted_basis(basis_pct::Float64, funding_rate_8h::Float64,
                               holding_days::Float64)::Float64
    funding_carry = funding_rate_8h * 3 * holding_days * 100.0
    return basis_pct - funding_carry
end

function roll_yield(near_price::Float64, far_price::Float64,
                     days_to_near_expiry::Int, days_to_far_expiry::Int)::Float64
    spread = far_price - near_price
    days_diff = far_expiry_days - days_to_near_expiry
    return (spread / near_price) * (365.0 / (days_diff + 1e-8)) * 100.0
end

# ---- 3. Calendar Spreads ----

function calendar_spread_value(spread::CalendarSpread)::Float64
    return spread.far_price - spread.near_price
end

function calendar_implied_rate(spread::CalendarSpread)::Float64
    fwd_rate = (spread.far_price / spread.near_price - 1.0)
    days_diff = spread.far_expiry_days - spread.near_expiry_days
    return fwd_rate * (365.0 / (days_diff + 1e-8)) * 100.0
end

function optimal_calendar_leg_ratio(near_vega::Float64, far_vega::Float64)::Float64
    return far_vega / (near_vega + 1e-8)
end

function calendar_theta_decay(near_theta::Float64, far_theta::Float64,
                                leg_ratio::Float64)::Float64
    return far_theta * leg_ratio - near_theta
end

# ---- 4. Cross-Exchange Arbitrage ----

function cross_exchange_arb_profit(spread::CrossExchangeSpread,
                                    size::Float64)::NamedTuple
    gross_profit = (spread.sell_price - spread.buy_price) * size
    transfer_fees = spread.transfer_fee * size
    execution_slippage = (spread.sell_price + spread.buy_price) * 0.0005 * size
    net_profit = gross_profit - transfer_fees - execution_slippage
    return (
        gross_profit        = gross_profit,
        transfer_fees       = transfer_fees,
        slippage_cost       = execution_slippage,
        net_profit          = net_profit,
        spread_pct          = spread.spread_pct,
        is_profitable       = net_profit > 0,
        min_size_profitable = (transfer_fees + execution_slippage) /
                              max(gross_profit / size - 1e-8, 1e-8),
    )
end

function triangular_arb(price_ab::Float64, price_bc::Float64,
                          price_ac::Float64, fee_pct::Float64=0.001)::Float64
    # A -> B -> C -> A path profit
    fee = 1 - fee_pct
    return price_ab * fee * (1/price_bc) * fee * price_ac * fee - 1.0
end

function stat_arb_zscore(spread::Vector{Float64}, window::Int=30)::Vector{Float64}
    n = length(spread); z = fill(NaN, n)
    for i in (window+1):n
        h = spread[i-window:i-1]
        z[i] = (spread[i] - mean(h)) / (std(h) + 1e-8)
    end
    return z
end

# ---- 5. Liquidation Mechanics ----

function liquidation_price(liq::LiquidationLevel)::Float64
    mm = liq.maintenance_margin
    if liq.side == :long
        return liq.entry_price * (1.0 - 1.0/liq.leverage + mm)
    else
        return liq.entry_price * (1.0 + 1.0/liq.leverage - mm)
    end
end

function liquidation_cascade_volume(positions::Vector{LiquidationLevel},
                                     price_drop_pct::Float64)::Float64
    total_liq = 0.0
    for pos in positions
        lp = liquidation_price(pos)
        price = pos.entry_price * (1 - price_drop_pct)
        if pos.side == :long && price <= lp
            total_liq += pos.position_size * pos.entry_price
        elseif pos.side == :short && price >= lp
            total_liq += pos.position_size * pos.entry_price
        end
    end
    return total_liq
end

function max_position_size(account_equity::Float64, leverage::Float64,
                            price::Float64, risk_pct::Float64=0.02)::Float64
    risk_dollars = account_equity * risk_pct
    return risk_dollars * leverage / (price + 1e-8)
end

function kelly_position_size(win_rate::Float64, win_size::Float64,
                              loss_size::Float64)::Float64
    return win_rate/loss_size - (1-win_rate)/win_size
end

# ---- 6. Delta-Neutral Hedging ----

function delta_hedge_ratio(option::CryptoOption, position_size::Float64)::Float64
    return -option.delta * position_size
end

function gamma_scalp_pnl(gamma::Float64, price_move::Float64,
                           position_size::Float64)::Float64
    return 0.5 * gamma * price_move^2 * position_size
end

function vega_pnl(vega::Float64, vol_move::Float64, position_size::Float64)::Float64
    return vega * vol_move * position_size
end

function theta_decay_daily(theta::Float64, position_size::Float64)::Float64
    return theta * position_size / 365.0
end

function options_pnl_components(option::CryptoOption, price_move::Float64,
                                  vol_move::Float64, time_elapsed_days::Float64,
                                  position_size::Float64)::NamedTuple
    delta_pnl = option.delta * price_move * position_size
    gamma_pnl = 0.5 * option.gamma * price_move^2 * position_size
    vega_pnl_val = option.vega * vol_move * position_size
    theta_pnl = option.theta * time_elapsed_days * position_size
    total = delta_pnl + gamma_pnl + vega_pnl_val + theta_pnl
    return (delta=delta_pnl, gamma=gamma_pnl, vega=vega_pnl_val,
            theta=theta_pnl, total=total)
end

# ---- 7. Crypto BS Option Pricing ----

function crypto_bs_call(S::Float64, K::Float64, r::Float64, q::Float64,
                         sigma::Float64, T::Float64)::Float64
    d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
    d2 = d1 - sigma*sqrt(T)
    nd1 = 0.5*(1 + erf(d1/sqrt(2))); nd2 = 0.5*(1 + erf(d2/sqrt(2)))
    return S*exp(-q*T)*nd1 - K*exp(-r*T)*nd2
end

function crypto_bs_put(S::Float64, K::Float64, r::Float64, q::Float64,
                        sigma::Float64, T::Float64)::Float64
    d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
    d2 = d1 - sigma*sqrt(T)
    nd1m = 0.5*(1 - erf(d1/sqrt(2))); nd2m = 0.5*(1 - erf(d2/sqrt(2)))
    return K*exp(-r*T)*nd2m - S*exp(-q*T)*nd1m
end

function crypto_bs_greeks(S::Float64, K::Float64, r::Float64, q::Float64,
                           sigma::Float64, T::Float64, opt::Symbol=:call)::CryptoOption
    d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T) + 1e-12)
    d2 = d1 - sigma*sqrt(T)
    nd1 = 0.5*(1 + erf(d1/sqrt(2))); nd2 = 0.5*(1 + erf(d2/sqrt(2)))
    phi_d1 = exp(-0.5*d1^2)/sqrt(2pi)
    delta = opt==:call ? exp(-q*T)*nd1 : exp(-q*T)*(nd1-1)
    gamma = exp(-q*T)*phi_d1 / (S*sigma*sqrt(T) + 1e-12)
    vega  = S*exp(-q*T)*phi_d1*sqrt(T)
    theta = opt==:call ?
        (-S*exp(-q*T)*phi_d1*sigma/(2*sqrt(T)+1e-12) - r*K*exp(-r*T)*nd2 + q*S*exp(-q*T)*nd1) :
        (-S*exp(-q*T)*phi_d1*sigma/(2*sqrt(T)+1e-12) + r*K*exp(-r*T)*(1-nd2) - q*S*exp(-q*T)*(1-nd1))
    price = opt==:call ? crypto_bs_call(S,K,r,q,sigma,T) : crypto_bs_put(S,K,r,q,sigma,T)
    return CryptoOption("CRYPTO", K, round(Int,T*365), opt, sigma, price, delta, gamma, vega, theta)
end

# ---- 8. Market Microstructure (Crypto) ----

function bid_ask_spread_bps(bid::Float64, ask::Float64)::Float64
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 10000.0
end

function effective_spread(executed_price::Float64, bid::Float64, ask::Float64)::Float64
    mid = (bid + ask) / 2.0
    return 2.0 * abs(executed_price - mid) / mid * 10000.0
end

function order_book_imbalance(bid_qty::Float64, ask_qty::Float64)::Float64
    return (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)
end

function kyle_lambda(price_changes::Vector{Float64}, order_flows::Vector{Float64})::Float64
    n = length(price_changes)
    x_bar = mean(order_flows); y_bar = mean(price_changes)
    cov_val = sum((order_flows .- x_bar) .* (price_changes .- y_bar)) / (n-1+1e-8)
    var_x = var(order_flows) + 1e-12
    return cov_val / var_x
end

# ---- 9. Portfolio Risk for Crypto ----

function crypto_portfolio_var(prices::Matrix{Float64}, weights::Vector{Float64},
                               confidence::Float64=0.99, horizon_days::Int=1)::Float64
    log_rets = diff(log.(prices), dims=1)
    port_rets = log_rets * weights
    sorted = sort(port_rets)
    cutoff_idx = max(1, round(Int, (1-confidence)*length(sorted)))
    var_1d = -sorted[cutoff_idx]
    return var_1d * sqrt(horizon_days)
end

function max_loss_estimate(prices::Matrix{Float64}, weights::Vector{Float64},
                            window::Int=252)::Float64
    log_rets = diff(log.(prices), dims=1)
    n = size(log_rets, 1)
    if n < window; return NaN; end
    port_rets = log_rets[end-window+1:end, :] * weights
    return -minimum(port_rets)
end

# ---- 10. Strategy Backtesting Helpers ----

function funding_arb_backtest(hist::FundingRateHistory,
                               entry_z::Float64=2.0, exit_z::Float64=0.5)
    n = length(hist.rates_8h)
    pnl = zeros(n); in_trade = false; entry_rate = 0.0
    window = 30
    for i in (window+1):n
        recent = hist.rates_8h[i-window:i-1]
        z = (hist.rates_8h[i] - mean(recent)) / (std(recent)+1e-8)
        if !in_trade && z > entry_z
            in_trade = true; entry_rate = hist.rates_8h[i]
        elseif in_trade
            pnl[i] = hist.rates_8h[i]  # collect funding
            if abs(z) < exit_z
                in_trade = false
            end
        end
    end
    total_pnl = sum(pnl) * 3 * 365  # annualised
    return (pnl_series=pnl, total_annualised=total_pnl,
            n_trades=count(p -> p > 0, pnl))
end

function stat_arb_backtest(spread::Vector{Float64}, entry_z::Float64=2.0,
                             exit_z::Float64=0.0, window::Int=30)
    n = length(spread); pnl = zeros(n); in_trade = false; entry_spread = 0.0; side = 1
    for i in (window+1):n
        h = spread[i-window:i-1]
        mu = mean(h); sig = std(h)+1e-8; z = (spread[i]-mu)/sig
        if !in_trade
            if z > entry_z; in_trade=true; entry_spread=spread[i]; side=-1
            elseif z < -entry_z; in_trade=true; entry_spread=spread[i]; side=1
            end
        else
            pnl[i] = side * (spread[i] - entry_spread)
            if abs(z) < exit_z; in_trade=false; entry_spread=0.0; end
        end
    end
    return (pnl=pnl, total=sum(pnl), sharpe=mean(pnl[pnl.!=0])/(std(pnl[pnl.!=0])+1e-8)*sqrt(252))
end

# ---- Demo ----

function demo()
    println("=== CryptoMechanics Demo ===")

    arb = FundingArb(100.0, 100.0, 10000.0, 0.0003, 30.0, 0.05)
    res = funding_arb_pnl(arb)
    println("Funding arb (30d, FR=0.03% 8h):")
    println("  Total funding income: \$", round(res.total_funding_income, digits=2))
    println("  Borrow cost: \$", round(res.borrow_cost, digits=2))
    println("  Net PnL: \$", round(res.net_pnl, digits=2))
    println("  Annualised yield: ", round(res.annualised_yield_pct, digits=2), "%")

    spread_obj = CrossExchangeSpread("Binance","OKX",30000.0,30120.0,5.0,2.0,0.4)
    arb_res = cross_exchange_arb_profit(spread_obj, 1.0)
    println("\nCross-exchange arb:")
    println("  Gross profit: \$", round(arb_res.gross_profit, digits=2))
    println("  Net profit: \$", round(arb_res.net_profit, digits=2))
    println("  Profitable: ", arb_res.is_profitable)

    opt = crypto_bs_greeks(30000.0, 30000.0, 0.05, 0.0, 0.80, 30/365, :call)
    println("\nATM call (BTC 30d, vol=80%):")
    println("  Premium: \$", round(opt.premium, digits=2))
    println("  Delta: ", round(opt.delta, digits=4))
    println("  Gamma: ", round(opt.gamma, digits=6))
    println("  Vega:  ", round(opt.vega, digits=2))
    println("  Theta: ", round(opt.theta, digits=2))

    liq = LiquidationLevel(1.0, 30000.0, 10.0, :long, 0.005, 0.0)
    lp = liquidation_price(liq)
    println("\nLong 1 BTC at 30000 with 10x leverage:")
    println("  Liquidation price: \$", round(lp, digits=2))

    fr_times = collect(Float64, 1:100)
    fr_rates = 0.0001 .+ 0.0002 .* sin.(fr_times ./ 10)
    fr_rates[90] = 0.0015  # spike
    hist = FundingRateHistory(fr_times, fr_rates, 30000.0 .* ones(100), 29900.0 .* ones(100))
    sig = funding_rate_signal(hist, 20)
    println("\nFunding rate signal: ", sig.signal, " (z=", round(sig.z_score, digits=2), ")")

    basis = perp_spot_basis(30120.0, 30000.0)
    println("\nPerp-spot basis: ", round(basis, digits=3), "%")
    println("Annualised funding (0.03% 8h): ", round(funding_8h_annualised(0.0003), digits=2), "%")
end

# ---- Additional Crypto Mechanics Functions ----

function funding_rate_prediction(hist::FundingRateHistory, horizon::Int=3)::Float64
    n = length(hist.rates_8h)
    if n < 10; return hist.rates_8h[end]; end
    rates = hist.rates_8h
    slope = (rates[end] - rates[end-min(9,n-1)]) / min(9.0,Float64(n-1))
    return rates[end] + slope * horizon
end

function cross_exchange_volume_ratio(vol_a::Float64, vol_b::Float64)::Float64
    total = vol_a + vol_b + 1e-8
    return vol_a / total
end

function perp_premium_index(perp_price::Float64, spot_price::Float64,
                              risk_free::Float64=0.05)::Float64
    theo_premium = risk_free / (3*365) * spot_price
    actual_premium = perp_price - spot_price
    return (actual_premium - theo_premium) / spot_price * 1e4
end

function liquidation_heatmap(price_levels::Vector{Float64},
                               positions::Vector{LiquidationLevel})::Vector{Float64}
    n = length(price_levels); heatmap = zeros(n)
    for pos in positions
        lp = liquidation_price(pos)
        idx = argmin(abs.(price_levels .- lp))
        heatmap[idx] += pos.position_size * pos.entry_price
    end
    return heatmap
end

function portfolio_delta_neutral(options::Vector{CryptoOption},
                                   spot_holdings::Float64)::Float64
    total_delta = sum(o.delta for o in options) + spot_holdings
    return -total_delta
end

function gamma_scalping_threshold(gamma::Float64, theta::Float64,
                                    position_size::Float64)::Float64
    daily_theta = abs(theta) * position_size / 365.0
    min_move = sqrt(2 * daily_theta / (gamma * position_size + 1e-12))
    return min_move
end

function volatility_carry_trade(realised_vol::Float64, implied_vol::Float64,
                                  vega::Float64, position_size::Float64)::NamedTuple
    carry = implied_vol - realised_vol
    daily_pnl = vega * carry / sqrt(252.0) * position_size
    ann_carry = daily_pnl * 252.0
    breakeven_vol = implied_vol
    return (vol_carry=carry, daily_pnl=daily_pnl, ann_carry_pnl=ann_carry,
            breakeven_realised_vol=breakeven_vol)
end

function spot_futures_convergence_rate(basis_now::Float64, basis_at_expiry::Float64,
                                        days_to_expiry::Float64)::Float64
    if days_to_expiry < 0.5; return 0.0; end
    return (basis_now - basis_at_expiry) / days_to_expiry
end

function cross_margin_efficiency(margin_a::Float64, margin_b::Float64,
                                   correlation::Float64)::Float64
    combined_margin = sqrt(margin_a^2 + margin_b^2 + 2*correlation*margin_a*margin_b)
    sum_margins = margin_a + margin_b
    return 1.0 - combined_margin / (sum_margins + 1e-8)
end

function funding_carry_breakeven_vol(funding_rate_annual::Float64,
                                      delta::Float64, vega::Float64)::Float64
    return sqrt(abs(funding_rate_annual / (vega * delta + 1e-8)))
end

function multi_exchange_best_execution(exchanges::Vector{SpotMarket},
                                        order_size::Float64)::NamedTuple
    best_buy = argmin([ex.ask for ex in exchanges])
    best_sell = argmax([ex.bid for ex in exchanges])
    best_ask = exchanges[best_buy].ask
    best_bid = exchanges[best_sell].bid
    spread = best_bid - best_ask
    return (best_buy_exchange=exchanges[best_buy].exchange,
            best_sell_exchange=exchanges[best_sell].exchange,
            best_ask=best_ask, best_bid=best_bid,
            spread_bps=spread/best_ask*1e4,
            can_arb=best_bid > best_ask)
end

function perpetual_swap_pnl(entry_price::Float64, exit_price::Float64,
                              size::Float64, side::Symbol,
                              funding_payments::Vector{Float64})::NamedTuple
    sign_val = side == :long ? 1.0 : -1.0
    price_pnl = sign_val * (exit_price - entry_price) * size
    funding_pnl = side == :long ? -sum(funding_payments)*size :
                                    sum(funding_payments)*size
    total = price_pnl + funding_pnl
    return (price_pnl=price_pnl, funding_pnl=funding_pnl, total_pnl=total,
            return_pct=total/(entry_price*size)*100)
end


# ---- Crypto Mechanics Utilities (continued) ----

function basis_arbitrage_signal(basis_history::Vector{Float64}, window::Int=14)
    n = length(basis_history); if n < window + 1; return (z=NaN, signal=:insufficient); end
    hist = basis_history[end-window:end-1]; cur = basis_history[end]
    z = (cur - mean(hist)) / (std(hist) + 1e-8)
    sig = z > 2 ? :wide_basis_sell_perp : z < -2 ? :narrow_basis_buy_perp : :neutral
    return (z=z, signal=sig, current_basis=cur, hist_avg=mean(hist))
end

function cross_coin_correlation(prices_a::Vector{Float64}, prices_b::Vector{Float64},
                                  window::Int=30)::Float64
    n = min(length(prices_a), length(prices_b))
    if n < window + 1; return NaN; end
    ra = diff(log.(prices_a[end-window:end]))
    rb = diff(log.(prices_b[end-window:end]))
    return cor(ra, rb)
end

function volatility_regime(price_series::Vector{Float64},
                             window_short::Int=7, window_long::Int=30)
    n = length(price_series)
    if n < window_long + 1; return :insufficient_data; end
    rets = diff(log.(price_series))
    vol_s = std(rets[end-window_short+1:end]) * sqrt(365.0)
    vol_l = std(rets[end-window_long+1:end])  * sqrt(365.0)
    ratio = vol_s / (vol_l + 1e-8)
    return ratio > 1.5 ? :high_vol_regime : ratio < 0.67 ? :low_vol_regime : :normal_regime
end

function defi_vs_cefi_spread(defi_rate::Float64, cefi_rate::Float64)::Float64
    return (defi_rate - cefi_rate) * 100.0
end

function options_market_stress_indicator(skew_25d::Float64,
                                          term_slope::Float64,
                                          vol_level::Float64)::Float64
    skew_z = skew_25d / 0.05
    slope_z = -term_slope / 0.02
    vol_z   = (vol_level - 0.6) / 0.3
    return (skew_z + slope_z + vol_z) / 3.0
end

function miner_capitulation_signal(hash_rate::Vector{Float64},
                                    btc_price::Vector{Float64},
                                    window::Int=14)::Symbol
    n = min(length(hash_rate), length(btc_price))
    if n < window + 1; return :insufficient_data; end
    hr_mom = (hash_rate[n] - hash_rate[n-window]) / (hash_rate[n-window] + 1e-8)
    px_mom = (btc_price[n] - btc_price[n-window]) / (btc_price[n-window] + 1e-8)
    if hr_mom < -0.1 && px_mom < -0.1; return :capitulation
    elseif hr_mom > 0.1; return :expansion
    else; return :stable
    end
end

function stablecoin_depeg_risk(peg_price::Float64, current_price::Float64,
                                 backing_ratio::Float64)::NamedTuple
    deviation_bps = (current_price - peg_price) / peg_price * 1e4
    backing_buffer = (backing_ratio - 1.0) * 100.0
    risk = abs(deviation_bps) > 50 || backing_ratio < 1.05 ? :high :
           abs(deviation_bps) > 20 ? :medium : :low
    return (deviation_bps=deviation_bps, backing_buffer_pct=backing_buffer, risk=risk)
end

function crypto_tax_lot_pnl(purchase_price::Float64, current_price::Float64,
                              quantity::Float64, holding_days::Int)::NamedTuple
    unrealised_pnl = (current_price - purchase_price) * quantity
    pct_return = (current_price - purchase_price) / (purchase_price + 1e-8) * 100.0
    lt_eligible = holding_days >= 365
    return (unrealised_pnl=unrealised_pnl, pct_return=pct_return,
            long_term_eligible=lt_eligible, holding_days=holding_days)
end

end # module CryptoMechanics
