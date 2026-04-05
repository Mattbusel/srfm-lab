## crypto_mechanics.R
## Basis trading, funding rate arb, cross-exchange spread, calendar spreads
## Pure base R -- no library() calls

# ============================================================
# 1. BASIS TRADING
# ============================================================

compute_basis <- function(futures_price, spot_price) {
  basis     <- futures_price - spot_price
  basis_pct <- basis / spot_price * 100
  list(basis = basis, basis_pct = basis_pct,
       annualized_basis = basis_pct * 365 / 30)  # approx for 30-day futures
}

rolling_basis <- function(futures_prices, spot_prices, window = 24) {
  basis <- futures_prices - spot_prices
  n     <- length(basis)
  ma    <- as.numeric(stats::filter(basis, rep(1/window, window), sides = 1))
  z     <- (basis - ma) / (sd(basis, na.rm = TRUE) + 1e-8)
  list(basis = basis, ma = ma, zscore = z,
       signal = ifelse(z > 2, -1, ifelse(z < -2, 1, 0)))  # mean reversion
}

basis_carry_pnl <- function(entry_basis, exit_basis, n_contracts,
                             contract_size, holding_hours) {
  # Long spot / short futures strategy
  basis_change <- exit_basis - entry_basis
  pnl          <- basis_change * n_contracts * contract_size
  annualized   <- pnl / (holding_hours / 8760) / (n_contracts * contract_size)
  list(pnl = pnl, basis_change = basis_change,
       annualized_return = annualized)
}

# ============================================================
# 2. FUNDING RATE ARBITRAGE
# ============================================================

funding_arb_pnl <- function(funding_rates, position_size,
                             entry_price, price_series,
                             long_perp = FALSE) {
  # Long spot + short perp (or vice versa): collect funding
  n         <- length(funding_rates)
  cum_fund  <- cumsum(funding_rates)
  price_ret <- diff(log(price_series))

  if (long_perp) {
    # Long perp, short spot: pay funding if positive
    hedge_pnl  <- -price_ret * position_size
    fund_pnl   <- -funding_rates[-1] * position_size * entry_price
  } else {
    # Short perp, long spot: receive funding if positive
    hedge_pnl  <- price_ret * position_size
    fund_pnl   <- funding_rates[-1] * position_size * entry_price
  }

  total_pnl <- hedge_pnl + fund_pnl
  list(
    funding_pnl   = fund_pnl,
    hedge_pnl     = hedge_pnl,
    total_pnl     = total_pnl,
    cum_total_pnl = cumsum(total_pnl),
    sharpe        = mean(total_pnl) / (sd(total_pnl) + 1e-8) * sqrt(365*3)  # 8h
  )
}

optimal_funding_entry <- function(funding_rates, n_periods = 3,
                                  min_rate = 0.001) {
  n    <- length(funding_rates)
  roll <- rep(NA_real_, n)
  for (i in seq(n_periods, n))
    roll[i] <- mean(funding_rates[seq(i - n_periods + 1, i)])
  entry_signal <- roll > min_rate
  exit_signal  <- roll < min_rate / 2
  list(rolling_avg = roll, entry = entry_signal, exit = exit_signal,
       expected_8h_rate = roll)
}

# ============================================================
# 3. CROSS-EXCHANGE SPREAD
# ============================================================

cross_exchange_spread <- function(prices_a, prices_b,
                                  fee_a = 0.001, fee_b = 0.001) {
  spread     <- prices_a - prices_b
  spread_pct <- spread / ((prices_a + prices_b) / 2) * 100
  round_trip_cost_pct <- (fee_a + fee_b) * 100
  profitable <- abs(spread_pct) > round_trip_cost_pct
  direction  <- ifelse(spread > 0, "buy_B_sell_A", "buy_A_sell_B")
  list(spread = spread, spread_pct = spread_pct,
       profitable = profitable, direction = direction,
       net_spread_pct = abs(spread_pct) - round_trip_cost_pct)
}

arb_execution_window <- function(spread_series, threshold_pct,
                                 latency_ms = 10) {
  # Estimate time windows where spread exceeds threshold
  n      <- length(spread_series)
  above  <- abs(spread_series) > threshold_pct
  starts <- which(diff(c(FALSE, above)) == 1)
  ends   <- which(diff(c(above, FALSE)) == -1)
  windows <- data.frame(start = starts, end = ends,
                        duration = ends - starts + 1)
  viable <- windows[windows$duration * latency_ms / 1000 < 1, ]  # sub-second
  list(all_windows = windows, viable = viable,
       arb_frequency = length(starts) / n)
}

triangular_arb <- function(price_ab, price_bc, price_ac,
                            fee = 0.001) {
  # A -> B -> C -> A cycle
  profit1 <- (1 - fee)^3 * price_ab * price_bc / price_ac - 1
  # A -> C -> B -> A cycle
  profit2 <- (1 - fee)^3 * price_ac / (price_bc * price_ab) - 1
  list(cycle1_profit = profit1, cycle2_profit = profit2,
       best_profit    = max(profit1, profit2),
       best_direction = if (profit1 > profit2) "ABC" else "ACB",
       profitable     = max(profit1, profit2) > 0)
}

# ============================================================
# 4. CALENDAR SPREADS
# ============================================================

calendar_spread <- function(near_price, far_price, near_expiry_days,
                            far_expiry_days, rf = 0.02) {
  # Fair value spread based on cost of carry
  fair_near <- near_price  # spot is reference
  fair_far  <- near_price * exp(rf * (far_expiry_days - near_expiry_days) / 365)
  spread_actual <- far_price - near_price
  spread_fair   <- fair_far - near_price
  richness      <- spread_actual - spread_fair
  list(actual_spread = spread_actual, fair_spread = spread_fair,
       richness = richness, roll_yield = -richness / near_price)
}

roll_yield_curve <- function(futures_prices, expiries_days, spot) {
  n     <- length(futures_prices)
  yields <- sapply(seq_len(n), function(i) {
    (futures_prices[i] / spot - 1) / (expiries_days[i] / 365)
  })
  contango  <- all(diff(futures_prices) > 0)
  backwardo <- all(diff(futures_prices) < 0)
  list(yields = yields, contango = contango, backwardation = backwardo,
       curve_slope = if (n > 1) coef(lm(futures_prices ~ expiries_days))[2] else NA)
}

# ============================================================
# 5. PERPETUAL SWAP MECHANICS
# ============================================================

perp_mark_price <- function(index_price, impact_bid, impact_ask,
                             premium_interval = 0.01) {
  fair_price <- (impact_bid + impact_ask) / 2
  premium    <- (fair_price - index_price) / index_price
  mark       <- index_price * (1 + premium)
  list(mark_price = mark, premium = premium,
       clamped_premium = max(-premium_interval, min(premium_interval, premium)))
}

funding_rate_calc <- function(mark_price, index_price,
                               interest_rate_8h = 0.0001,
                               premium_clamp = 0.005) {
  premium  <- (mark_price - index_price) / index_price
  clamped  <- max(-premium_clamp, min(premium_clamp, premium))
  rate     <- clamped + interest_rate_8h
  list(rate = rate, premium = premium, direction = sign(rate),
       annualized = rate * 3 * 365)
}

liquidation_price <- function(entry_price, leverage, side = "long",
                               maintenance_margin = 0.005) {
  initial_margin <- 1 / leverage
  if (side == "long")
    entry_price * (1 - initial_margin + maintenance_margin)
  else
    entry_price * (1 + initial_margin - maintenance_margin)
}

# ============================================================
# 6. DEFI YIELD COMPARISONS
# ============================================================

cefi_vs_defi_yield <- function(cefi_apr, defi_apy_gross,
                                gas_cost_usd, position_size_usd,
                                smart_contract_risk_pct = 0.02) {
  defi_apr_net <- log(1 + defi_apy_gross) - gas_cost_usd / position_size_usd -
                  smart_contract_risk_pct
  excess_yield <- defi_apr_net - cefi_apr
  list(cefi_apr = cefi_apr, defi_apr_net = defi_apr_net,
       excess_yield = excess_yield,
       prefer_defi  = defi_apr_net > cefi_apr)
}

# ============================================================
# 7. VOLATILITY REGIME FOR CRYPTO
# ============================================================

crypto_vol_regime <- function(returns, short_w = 7, long_w = 30) {
  n      <- length(returns)
  vol_s  <- rep(NA_real_, n); vol_l <- rep(NA_real_, n)
  for (i in seq(long_w, n)) {
    vol_s[i] <- sd(returns[seq(i - short_w  + 1, i)]) * sqrt(365)
    vol_l[i] <- sd(returns[seq(i - long_w   + 1, i)]) * sqrt(365)
  }
  regime <- ifelse(vol_s > vol_l * 1.5, "high_vol",
             ifelse(vol_s < vol_l * 0.7, "low_vol", "normal"))
  list(short_vol = vol_s, long_vol = vol_l, regime = regime)
}

realized_crypto_vol <- function(prices, window = 24, annualize_hrs = 8760) {
  ret    <- diff(log(prices))
  n      <- length(ret)
  rv     <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx   <- seq(i - window + 1, i)
    rv[i] <- sqrt(sum(ret[idx]^2) / window * annualize_hrs)
  }
  rv
}

# ============================================================
# 8. PORTFOLIO METRICS CRYPTO
# ============================================================

crypto_portfolio_stats <- function(returns_matrix, weights) {
  port_ret <- as.vector(returns_matrix %*% weights)
  # Annualize assuming daily data
  ann <- 365
  mu  <- mean(port_ret) * ann
  sg  <- sd(port_ret) * sqrt(ann)
  cum <- cumprod(1 + port_ret)
  dd  <- (cum - cummax(cum)) / cummax(cum)
  list(
    ann_return = mu, ann_vol = sg,
    sharpe     = mu / (sg + 1e-8),
    max_dd     = min(dd),
    calmar     = mu / abs(min(dd) + 1e-8),
    skew       = mean((port_ret-mean(port_ret))^3) / sd(port_ret)^3,
    kurt_excess= mean((port_ret-mean(port_ret))^4) / sd(port_ret)^4 - 3
  )
}

correlation_regime_shift <- function(returns_matrix, window = 30) {
  T_   <- nrow(returns_matrix); N <- ncol(returns_matrix)
  mean_corr <- rep(NA_real_, T_)
  for (i in seq(window, T_)) {
    idx <- seq(i - window + 1, i)
    C   <- cor(returns_matrix[idx, ])
    mean_corr[i] <- mean(C[upper.tri(C)])
  }
  list(mean_corr = mean_corr,
       stress     = mean_corr > quantile(mean_corr, 0.9, na.rm = TRUE))
}


# ============================================================
# ADDITIONAL: DEFI INTEGRATION
# ============================================================

amm_to_cex_arb <- function(amm_price, cex_bid, cex_ask,
                             amm_fee = 0.003, gas_cost_pct = 0.001) {
  buy_on_amm  <- cex_bid - amm_price * (1 + amm_fee) - gas_cost_pct * amm_price
  sell_on_amm <- amm_price * (1 - amm_fee) - cex_ask - gas_cost_pct * amm_price
  list(buy_amm_profit = buy_on_amm, sell_amm_profit = sell_on_amm,
       best = max(buy_on_amm, sell_on_amm),
       direction = if (buy_on_amm > sell_on_amm) "buy_amm_sell_cex" else "sell_amm_buy_cex")
}

flash_loan_arb <- function(price_a, price_b, fee_rate = 0.0009,
                            loan_amount = 1e6) {
  # Simple flash loan arb between two prices
  gross_profit <- loan_amount * abs(price_b / price_a - 1)
  flash_fee    <- loan_amount * fee_rate
  net_profit   <- gross_profit - flash_fee
  list(gross = gross_profit, fee = flash_fee, net = net_profit,
       profitable = net_profit > 0,
       min_spread = fee_rate * 100)  # minimum spread in pct to be profitable
}

# ============================================================
# ADDITIONAL: OPTIONS ON PERPS
# ============================================================

deribit_style_settlement <- function(index_prices, expiry_window = 30) {
  # Deribit settles at 30min TWAP before expiry
  n   <- length(index_prices)
  twap_settle <- mean(index_prices[max(1, n - expiry_window + 1):n])
  list(settlement_price = twap_settle,
       vs_last = twap_settle - index_prices[n],
       manipulation_resistance = sd(index_prices[max(1,n-expiry_window+1):n]) /
                                  twap_settle)
}

perp_basis_carry <- function(perp_price, spot_price,
                              funding_rate, holding_days) {
  basis_pct    <- (perp_price - spot_price) / spot_price * 100
  annualized_b <- basis_pct * 365 / holding_days
  funding_ann  <- funding_rate * 3 * 365 * 100  # 8h periods
  total_carry  <- annualized_b + funding_ann
  list(basis_pct = basis_pct, annualized_basis = annualized_b,
       funding_ann = funding_ann, total_carry = total_carry)
}

# ============================================================
# ADDITIONAL: RISK MANAGEMENT FOR CRYPTO
# ============================================================

crypto_var_historical <- function(returns, confidence = 0.95,
                                   window = 100, decay = 0.94) {
  n      <- length(returns)
  var_hs <- rep(NA, n)
  var_ew <- rep(NA, n)
  for (i in seq(window, n)) {
    idx    <- seq(i - window + 1, i)
    var_hs[i] <- quantile(returns[idx], 1 - confidence)
    # EWMA-weighted historical simulation
    w      <- decay^rev(seq_along(idx) - 1)
    w      <- w / sum(w)
    ord    <- order(returns[idx])
    cumw   <- cumsum(w[ord])
    var_ew[i] <- returns[idx[ord[which(cumw >= 1 - confidence)[1]]]]
  }
  list(historical = var_hs, ewma_weighted = var_ew,
       es = sapply(seq_along(var_hs), function(i) {
         if (is.na(var_hs[i])) return(NA)
         mean(returns[pmax(1, i-window+1):i][
           returns[pmax(1, i-window+1):i] <= var_hs[i]])
       }))
}

cross_margin_risk <- function(positions_by_asset, prices, margins,
                               correlation_matrix) {
  N           <- length(positions_by_asset)
  notionals   <- positions_by_asset * prices
  port_var    <- as.numeric(
    t(notionals) %*% (correlation_matrix * (margins %o% margins)) %*% notionals)
  margin_req  <- sqrt(port_var)
  margin_util <- margin_req / (sum(abs(notionals)) + 1e-8)
  list(portfolio_var = port_var, required_margin = margin_req,
       margin_utilization = margin_util,
       by_asset = abs(notionals) * margins)
}

# ============================================================
# ADDITIONAL: MARKET STRUCTURE ANALYSIS
# ============================================================

exchange_dominance <- function(volume_by_exchange) {
  total <- sum(volume_by_exchange)
  share <- volume_by_exchange / total
  hhi   <- sum(share^2)
  list(shares = share, hhi = hhi,
       top_exchange_pct = max(share) * 100,
       n_significant = sum(share > 0.05))
}

bid_ask_depth_analysis <- function(bid_prices, bid_sizes,
                                    ask_prices, ask_sizes, levels = 5) {
  # Order book imbalance at multiple levels
  levels  <- min(levels, length(bid_prices), length(ask_prices))
  bid_val <- sum(bid_prices[1:levels] * bid_sizes[1:levels])
  ask_val <- sum(ask_prices[1:levels] * ask_sizes[1:levels])
  imb     <- (bid_val - ask_val) / (bid_val + ask_val + 1e-8)

  # Weighted mid
  wmid <- (bid_prices[1] * ask_sizes[1] + ask_prices[1] * bid_sizes[1]) /
          (bid_sizes[1] + ask_sizes[1] + 1e-8)

  list(bid_depth = bid_val, ask_depth = ask_val,
       imbalance = imb, weighted_mid = wmid,
       spread = ask_prices[1] - bid_prices[1],
       spread_pct = (ask_prices[1] - bid_prices[1]) /
                    ((ask_prices[1] + bid_prices[1])/2) * 100)
}

order_book_resilience <- function(mid_price_before, mid_price_after,
                                   trade_size, recovery_time) {
  impact      <- (mid_price_after - mid_price_before) / mid_price_before * 1e4
  resilience  <- abs(impact) / (trade_size + 1e-8)
  list(impact_bps = impact, resilience = resilience,
       recovery_time = recovery_time,
       resilience_score = 1 / (resilience * recovery_time + 1e-8))
}

# ============================================================
# ADDITIONAL: DEFI PROTOCOL METRICS
# ============================================================
protocol_revenue_analysis <- function(fee_revenue, tvl, token_price,
                                       token_supply, window=30) {
  pe_ratio     <- token_price * token_supply / (fee_revenue * 365 + 1e-8)
  ps_ratio     <- token_price * token_supply / (tvl + 1e-8)
  rev_yield    <- fee_revenue / (token_price * token_supply + 1e-8) * 365
  rev_ma       <- as.numeric(stats::filter(fee_revenue, rep(1/window,window), sides=1))
  list(pe=pe_ratio, ps=ps_ratio, revenue_yield=rev_yield,
       revenue_growth=c(NA, diff(log(fee_revenue+1))),
       smoothed_revenue=rev_ma)
}

token_inflation_impact <- function(circulating_supply, total_supply,
                                    emission_rate_daily, price) {
  inflation_annual <- emission_rate_daily * 365 / circulating_supply
  dilution_pct     <- emission_rate_daily / (circulating_supply + 1e-8)
  revenue_needed   <- emission_rate_daily * price  # buy pressure needed to offset
  list(inflation_annual=inflation_annual, dilution_daily=dilution_pct,
       sell_pressure=emission_rate_daily*price,
       fdv=price*total_supply, mc=price*circulating_supply)
}

vetoken_model <- function(locked_supply, total_supply, lock_period_years,
                           base_boost=1, max_boost=2.5) {
  lock_ratio <- locked_supply / (total_supply + 1e-8)
  boost_avg  <- base_boost + (max_boost-base_boost) * lock_ratio
  voting_pct <- locked_supply / total_supply
  list(lock_ratio=lock_ratio, avg_boost=boost_avg,
       voting_power=voting_pct, governance_concentration=lock_ratio^2)
}

# ============================================================
# ADDITIONAL: MEV / ORDERFLOW
# ============================================================
sandwich_attack_cost <- function(victim_size_usd, pool_liquidity_usd,
                                  fee_rate=0.003, gas_cost_usd=5) {
  # Estimate sandwich attack profitability
  price_impact  <- victim_size_usd / (pool_liquidity_usd + 1e-8)
  front_run_profit <- victim_size_usd * price_impact * 0.5
  back_run_profit  <- victim_size_usd * price_impact * 0.5
  total_mev     <- front_run_profit + back_run_profit - 2*gas_cost_usd -
                   2*fee_rate*victim_size_usd
  list(victim_impact=price_impact, mev_profit=total_mev,
       mev_positive=total_mev>0, victim_cost=victim_size_usd*price_impact)
}

block_builder_revenue <- function(priority_fees, mev_revenue, base_fee,
                                   blocks_per_day=7200) {
  total_per_block <- priority_fees + mev_revenue
  daily_revenue   <- total_per_block * blocks_per_day
  list(per_block=total_per_block, daily=daily_revenue,
       mev_fraction=mev_revenue/(total_per_block+1e-8),
       annualized=daily_revenue*365)
}

# ============================================================
# ADDITIONAL: STABLECOIN ANALYTICS
# ============================================================
stablecoin_depeg_model <- function(prices, peg=1.0, confidence=0.99) {
  deviation   <- prices - peg
  vol_dev     <- sd(deviation)
  var_depeg   <- qnorm(1-confidence) * vol_dev
  es_depeg    <- mean(deviation[deviation < quantile(deviation, 1-confidence)])
  list(mean_dev=mean(deviation), vol=vol_dev, var=var_depeg, es=es_depeg,
       pct_off_peg=mean(abs(deviation)>0.005)*100,
       worst_depeg=min(deviation))
}

stablecoin_collateral_ratio <- function(collateral_value, stablecoin_supply,
                                         target_cr=1.5) {
  cr    <- collateral_value / (stablecoin_supply + 1e-8)
  safe  <- cr > target_cr
  buffer <- (cr - target_cr) / target_cr
  list(cr=cr, safe=safe, buffer_pct=buffer*100,
       max_mintable=collateral_value/target_cr - stablecoin_supply,
       liquidation_price=stablecoin_supply*target_cr/collateral_value)
}


# ============================================================
# ADDITIONAL: ON-CHAIN & ADVANCED CRYPTO MECHANICS
# ============================================================

nvt_ratio <- function(market_cap, transaction_volume, window = 28) {
  nvt     <- market_cap / (transaction_volume + 1)
  nvt_sma <- as.numeric(stats::filter(nvt, rep(1/window, window), sides=1))
  z       <- (nvt - nvt_sma) / (sd(nvt, na.rm=TRUE) + 1e-8)
  list(nvt = nvt, smoothed = nvt_sma, z_score = z,
       signal = ifelse(z > 2, -1, ifelse(z < -2, 1, 0)),
       overvalued = nvt > quantile(nvt, 0.9, na.rm=TRUE))
}

sopr_signal <- function(realized_price_by_coin, current_price) {
  sopr   <- current_price / (realized_price_by_coin + 1e-8)
  sopr_ma <- as.numeric(stats::filter(sopr, rep(1/14, 14), sides=1))
  list(sopr = sopr, smoothed = sopr_ma,
       in_profit = sopr > 1,
       capitulation = sopr < 0.95,
       signal = ifelse(sopr_ma > 1.05, -1, ifelse(sopr_ma < 0.98, 1, 0)))
}

active_addresses_signal <- function(active_addrs, price, window = 30) {
  n  <- length(active_addrs)
  aa_ma <- as.numeric(stats::filter(active_addrs, rep(1/window, window), sides=1))
  z     <- (active_addrs - aa_ma) / (sd(active_addrs, na.rm=TRUE) + 1e-8)
  price_growth  <- c(NA, diff(log(price)))
  addr_growth   <- c(NA, diff(log(active_addrs)))
  divergence    <- sign(price_growth) != sign(addr_growth) & !is.na(price_growth)
  list(active = active_addrs, smoothed = aa_ma, z_score = z,
       divergence = divergence,
       signal = ifelse(z > 1 & !divergence, 1, ifelse(z < -1, -1, 0)))
}

hash_rate_signal <- function(hash_rate, difficulty, price, window = 14) {
  miner_revenue <- price / (hash_rate + 1e-12)
  diff_adj_hr   <- hash_rate / (difficulty + 1e-12)
  hr_ma         <- as.numeric(stats::filter(hash_rate, rep(1/window, window), sides=1))
  security_z    <- (hash_rate - hr_ma) / (sd(hash_rate, na.rm=TRUE) + 1e-8)
  list(hash_rate = hash_rate, miner_revenue_per_hash = miner_revenue,
       diff_adj = diff_adj_hr, smoothed = hr_ma, security_z = security_z,
       strong_network = security_z > 1,
       signal = ifelse(security_z > 1.5, 1, ifelse(security_z < -2, -1, 0)))
}

realized_vs_market_cap <- function(market_cap, realized_cap) {
  mvrv  <- market_cap / (realized_cap + 1e-8)
  n     <- length(mvrv)
  mvrv_mean <- mean(mvrv, na.rm=TRUE)
  mvrv_sd   <- sd(mvrv, na.rm=TRUE)
  z <- (mvrv - mvrv_mean) / (mvrv_sd + 1e-8)
  list(mvrv = mvrv, z_score = z,
       overvalued = mvrv > 3.5, undervalued = mvrv < 1,
       signal = ifelse(z > 2, -1, ifelse(z < -1, 1, 0)))
}

coin_days_destroyed <- function(coins_moved, days_since_last_move,
                                  price, window = 30) {
  cdd      <- coins_moved * days_since_last_move
  cdd_ma   <- as.numeric(stats::filter(cdd, rep(1/window, window), sides=1))
  dormancy <- cdd / (coins_moved + 1e-12)
  z        <- (cdd - cdd_ma) / (sd(cdd, na.rm=TRUE) + 1e-8)
  list(cdd = cdd, smoothed = cdd_ma, dormancy_days = dormancy,
       z_score = z,
       long_term_holder_sell = z > 2,
       signal = ifelse(z > 2, -1, 0))
}
