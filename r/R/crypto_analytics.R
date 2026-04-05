# =============================================================================
# crypto_analytics.R
# Crypto-specific analytics and on-chain metric modeling
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Crypto markets have unique on-chain data that equity
# markets lack. MVRV-Z measures if the market is overvalued relative to
# realized value; NVT (network value to transactions) is crypto's P/E ratio;
# funding rates reveal leverage in the perpetual futures market.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. MVRV-Z SCORE
# -----------------------------------------------------------------------------

#' MVRV-Z Score: (Market Cap - Realized Cap) / StdDev(Market Cap)
#' Values > 7: historically overvalued (sell signal)
#' Values < 0: undervalued (buy signal)
#' @param market_cap time series of market cap (in USD)
#' @param realized_cap time series of realized cap (cost basis aggregate)
#' @return data.frame with date-indexed MVRV, Z-score, and signals
mvrv_z_score <- function(market_cap, realized_cap, dates = NULL) {
  n <- length(market_cap)
  if (is.null(dates)) dates <- seq_len(n)

  # MVRV ratio
  mvrv <- market_cap / realized_cap

  # Rolling Z-score over all available history
  mvrv_z <- numeric(n)
  for (t in seq_len(n)) {
    mu_t  <- mean(market_cap[1:t])
    sd_t  <- if (t > 1) sd(market_cap[1:t]) else 1
    mvrv_z[t] <- (market_cap[t] - realized_cap[t]) / (sd_t + 1e-10)
  }

  # Signal: buy when z < 0, sell when z > 7
  signal <- ifelse(mvrv_z < 0, "BUY",
            ifelse(mvrv_z > 7, "SELL", "NEUTRAL"))

  result <- data.frame(date=dates, market_cap=market_cap,
                        realized_cap=realized_cap,
                        MVRV=mvrv, MVRV_Z=mvrv_z, signal=signal)

  cat("=== MVRV-Z Score Analysis ===\n")
  cat(sprintf("Latest MVRV: %.3f\n", tail(mvrv, 1)))
  cat(sprintf("Latest MVRV-Z: %.3f\n", tail(mvrv_z, 1)))
  cat(sprintf("Current signal: %s\n", tail(signal, 1)))
  cat(sprintf("Days in BUY zone: %d (%.1f%%)\n",
              sum(signal=="BUY"), 100*mean(signal=="BUY")))
  cat(sprintf("Days in SELL zone: %d (%.1f%%)\n",
              sum(signal=="SELL"), 100*mean(signal=="SELL")))

  invisible(result)
}

# -----------------------------------------------------------------------------
# 2. SOPR (Spent Output Profit Ratio)
# -----------------------------------------------------------------------------

#' SOPR: ratio of realized value to creation value of spent UTXOs
#' SOPR > 1: coins sold at a profit (holders taking gains)
#' SOPR < 1: coins sold at a loss (capitulation)
#' SOPR ~= 1 acts as support/resistance
#' @param sopr_ts time series of SOPR values
sopr_analysis <- function(sopr_ts, dates = NULL) {
  n <- length(sopr_ts)
  if (is.null(dates)) dates <- seq_len(n)

  # Smoothed SOPR (7-day)
  sopr_smooth <- stats::filter(sopr_ts, rep(1/7, 7), sides=1)
  sopr_smooth <- as.numeric(sopr_smooth)

  # Deviation from 1.0 (equilibrium)
  sopr_dev <- sopr_ts - 1

  # Signal generation
  # SOPR crossing above 1 from below = selling pressure resuming
  sopr_cross_up   <- which(sopr_ts[-1] > 1 & sopr_ts[-n] <= 1) + 1
  sopr_cross_down <- which(sopr_ts[-1] < 1 & sopr_ts[-n] >= 1) + 1

  # Accumulation (SOPR < 0.99 for extended period)
  accumulation <- rle(sopr_ts < 0.99)
  acc_periods <- which(accumulation$values & accumulation$lengths >= 5)

  cat("=== SOPR Analysis ===\n")
  cat(sprintf("Latest SOPR: %.4f (%s)\n",
              tail(sopr_ts, 1),
              ifelse(tail(sopr_ts,1) > 1, "profit-taking", "capitulation")))
  cat(sprintf("Upward crossings (profit-taking phases): %d\n", length(sopr_cross_up)))
  cat(sprintf("Downward crossings (loss phases): %d\n", length(sopr_cross_down)))

  invisible(list(sopr=sopr_ts, sopr_smooth=sopr_smooth,
                 cross_up=sopr_cross_up, cross_down=sopr_cross_down,
                 current_regime=ifelse(tail(sopr_ts,1)>1,"profit","loss")))
}

# -----------------------------------------------------------------------------
# 3. HASH RATE MODELING
# -----------------------------------------------------------------------------

#' Hash rate analysis: miner behavior and difficulty ribbon
#' Hash rate = total computational power securing the network
#' Hash rate decline = miners capitulating (bearish)
#' Hash rate growth after correction = miner confidence (bullish)
#' @param hash_rate time series of hash rate (EH/s)
#' @param price bitcoin price series (aligned)
hash_rate_analysis <- function(hash_rate, price, dates=NULL) {
  n <- length(hash_rate)
  if (is.null(dates)) dates <- seq_len(n)

  # Hash ribbons: short (30d) and long (60d) moving averages
  hash_ma30 <- stats::filter(hash_rate, rep(1/30, 30), sides=1)
  hash_ma60 <- stats::filter(hash_rate, rep(1/60, 60), sides=1)
  hash_ma30 <- as.numeric(hash_ma30); hash_ma60 <- as.numeric(hash_ma60)

  # Miner capitulation: MA30 crosses below MA60
  cap_signal  <- !is.na(hash_ma30) & !is.na(hash_ma60) & hash_ma30 < hash_ma60
  cap_start   <- which(diff(cap_signal * 1) == 1)
  cap_end     <- which(diff(cap_signal * 1) == -1)

  # Price during capitulation events
  if (length(cap_start) > 0 && length(cap_end) > 0) {
    cap_events <- data.frame(
      start = cap_start,
      end   = if (length(cap_end) >= length(cap_start)) cap_end[seq_along(cap_start)]
              else c(cap_end, rep(n, length(cap_start)-length(cap_end))),
      duration = numeric(length(cap_start))
    )
    cap_events$duration <- cap_events$end - cap_events$start
    cap_events$price_change <- sapply(seq_len(nrow(cap_events)), function(i) {
      s <- cap_events$start[i]; e <- min(cap_events$end[i], n)
      if (s < 1 || e > n) return(NA)
      (price[e] - price[s]) / price[s]
    })
  } else {
    cap_events <- NULL
  }

  # Hash rate growth rate
  hash_growth <- c(NA, diff(log(hash_rate)))

  # Correlation between hash rate growth and future price returns (1-30 day)
  hash_price_corr <- sapply(1:30, function(h) {
    if (h >= n) return(NA)
    cor(hash_growth[1:(n-h)], diff(log(price))[1:(n-h)], use="complete")
  })

  cat("=== Hash Rate Analysis ===\n")
  cat(sprintf("Latest hash rate: %.2f (rel to 30d MA: %.2f%%)\n",
              tail(hash_rate,1),
              100*(tail(hash_rate,1)/tail(hash_ma30,1)-1)))
  cat(sprintf("Miner capitulation events: %d\n", length(cap_start)))
  cat(sprintf("Best predictive lag (hash->price): %d days (corr=%.3f)\n",
              which.max(abs(hash_price_corr)),
              max(abs(hash_price_corr), na.rm=TRUE)))

  invisible(list(hash_ma30=hash_ma30, hash_ma60=hash_ma60,
                 cap_signal=cap_signal, cap_events=cap_events,
                 hash_price_corr=hash_price_corr))
}

# -----------------------------------------------------------------------------
# 4. NVT RATIO ANALYSIS
# -----------------------------------------------------------------------------

#' NVT (Network Value to Transactions) ratio
#' NVT = Market Cap / Daily On-Chain Transaction Volume (USD)
#' Like P/E for crypto: high NVT = overvalued relative to utility
#' NVT Signal uses 90d MA of volume to smooth noise
#' @param market_cap market cap time series
#' @param tx_volume daily on-chain transaction volume (USD)
nvt_analysis <- function(market_cap, tx_volume, dates=NULL) {
  n <- length(market_cap)
  if (is.null(dates)) dates <- seq_len(n)

  # Basic NVT
  nvt <- market_cap / (tx_volume + 1)

  # NVT Signal: use 90-day MA of volume
  tx_ma90 <- as.numeric(stats::filter(tx_volume, rep(1/90, 90), sides=1))
  nvt_signal <- market_cap / (tx_ma90 + 1)

  # Rolling z-score of NVT signal (zscore over 2-year window)
  window <- min(730, n)
  nvt_zscore <- numeric(n)
  for (t in window:n) {
    win_vals <- nvt_signal[max(1, t-window+1):t]
    nvt_zscore[t] <- (nvt_signal[t] - mean(win_vals, na.rm=TRUE)) /
                     (sd(win_vals, na.rm=TRUE) + 1e-10)
  }

  # Overbought/oversold thresholds (empirical for BTC)
  overbought <- 2.0; oversold <- -1.5
  signal_state <- ifelse(nvt_zscore > overbought, "OVERBOUGHT",
                  ifelse(nvt_zscore < oversold, "OVERSOLD", "NEUTRAL"))

  cat("=== NVT Analysis ===\n")
  cat(sprintf("Latest NVT:        %.2f\n", tail(nvt, 1)))
  cat(sprintf("Latest NVT Signal: %.2f\n", tail(nvt_signal, 1)))
  cat(sprintf("NVT Z-Score:       %.2f (%s)\n",
              tail(nvt_zscore, 1), tail(signal_state, 1)))
  cat(sprintf("% time overbought: %.1f%%\n", 100*mean(signal_state=="OVERBOUGHT")))
  cat(sprintf("% time oversold:   %.1f%%\n", 100*mean(signal_state=="OVERSOLD")))

  invisible(list(nvt=nvt, nvt_signal=nvt_signal, nvt_zscore=nvt_zscore,
                 signal=signal_state))
}

# -----------------------------------------------------------------------------
# 5. EXCHANGE FLOW ANALYSIS (SUPPLY SHOCK RATIO)
# -----------------------------------------------------------------------------

#' Supply Shock Ratio analysis
#' Supply Shock = Illiquid Supply / Liquid Supply
#' Rising SSR = coins moving to cold storage (supply shock = bullish)
#' Falling SSR = coins moving to exchanges (sell pressure = bearish)
#' @param illiquid_supply coins in long-term/cold storage
#' @param liquid_supply coins on exchanges / in short-term hands
supply_shock_analysis <- function(illiquid_supply, liquid_supply, dates=NULL) {
  n <- length(illiquid_supply)
  if (is.null(dates)) dates <- seq_len(n)

  ssr <- illiquid_supply / (liquid_supply + 1e-10)
  ssr_change <- c(NA, diff(ssr) / ssr[-n])

  # Exchange netflow: negative = coins leaving exchanges (bullish)
  exchange_netflow <- c(NA, diff(liquid_supply))
  outflow_events <- which(exchange_netflow < quantile(exchange_netflow, 0.10, na.rm=T))
  inflow_events  <- which(exchange_netflow > quantile(exchange_netflow, 0.90, na.rm=T))

  # 30-day momentum of SSR
  ssr_mom30 <- c(rep(NA,30), ssr[31:n] / ssr[1:(n-30)] - 1)

  cat("=== Supply Shock Analysis ===\n")
  cat(sprintf("Latest SSR: %.3f\n", tail(ssr,1)))
  cat(sprintf("30-day SSR change: %.2f%%\n", 100*tail(ssr_mom30,1)))
  cat(sprintf("Exchange outflow events (>1-sigma): %d\n", length(outflow_events)))
  cat(sprintf("Exchange inflow events (>1-sigma):  %d\n", length(inflow_events)))
  regime <- ifelse(tail(ssr_mom30,1) > 0.02, "SUPPLY_SHOCK (bullish)",
            ifelse(tail(ssr_mom30,1) < -0.02, "SUPPLY_GLUT (bearish)", "NEUTRAL"))
  cat(sprintf("Current regime: %s\n", regime))

  invisible(list(ssr=ssr, ssr_change=ssr_change, ssr_mom30=ssr_mom30,
                 outflow_events=outflow_events, inflow_events=inflow_events))
}

# -----------------------------------------------------------------------------
# 6. FUNDING RATE ARBITRAGE MODELING
# -----------------------------------------------------------------------------

#' Funding rate analysis for perpetual futures
#' Positive funding: longs pay shorts (market is long-leveraged = overbought signal)
#' Negative funding: shorts pay longs (market is short-leveraged = oversold signal)
#' Funding arbitrage: long spot, short perp when funding is positive
#' @param funding_rates 8-hour funding rate series
#' @param price spot price series
funding_rate_analysis <- function(funding_rates, price, dates=NULL) {
  n <- length(funding_rates)
  if (is.null(dates)) dates <- seq_len(n)

  # Annualized funding rate
  # 3 funding periods per day * 365 days = 1095 periods per year
  funding_annualized <- funding_rates * 1095

  # Cumulative funding over rolling 7-day window (21 periods)
  cumfund_7d <- as.numeric(stats::filter(funding_rates, rep(1, 21), sides=1))

  # Arbitrage P&L: go long spot, short perp
  # P&L_t = funding_t - (spot_return_t - 0)  [assuming no basis cost]
  spot_returns <- c(NA, diff(log(price)))
  arb_pnl <- funding_rates - abs(spot_returns) * 0  # funding received; basis risk ignored
  arb_cumulative <- cumsum(ifelse(is.na(arb_pnl), 0, arb_pnl))

  # Signal: enter arb when 7-day cumulative funding > threshold
  # Exit when funding reverts to neutral
  threshold_enter <- quantile(cumfund_7d, 0.80, na.rm=TRUE)
  threshold_exit  <- quantile(cumfund_7d, 0.50, na.rm=TRUE)

  # Sharpe of funding arb strategy
  valid_pnl <- arb_pnl[!is.na(arb_pnl)]
  arb_sharpe <- mean(valid_pnl) / (sd(valid_pnl)+1e-10) * sqrt(1095)

  cat("=== Funding Rate Analysis ===\n")
  cat(sprintf("Latest funding rate: %.5f (%.2f%% ann)\n",
              tail(funding_rates,1), 100*tail(funding_annualized,1)))
  cat(sprintf("7-day cumulative funding: %.5f\n", tail(cumfund_7d,1)))
  cat(sprintf("Arb strategy Sharpe (annualized): %.3f\n", arb_sharpe))
  cat(sprintf("Funding regime: %s\n",
              ifelse(tail(funding_annualized,1) > 0.50, "LONGS PAYING (overbought)",
              ifelse(tail(funding_annualized,1) < -0.20, "SHORTS PAYING (oversold)",
                     "NEUTRAL"))))

  invisible(list(funding_ann=funding_annualized, cumfund_7d=cumfund_7d,
                 arb_pnl=arb_pnl, arb_cumulative=arb_cumulative,
                 arb_sharpe=arb_sharpe))
}

# -----------------------------------------------------------------------------
# 7. TOKEN UNLOCK IMPACT QUANTIFICATION
# -----------------------------------------------------------------------------

#' Analyze price impact of token unlocks
#' Large cliff unlocks create sell pressure; vested unlocks are gradual
#' @param unlock_schedule data.frame: date, unlock_amount (tokens), total_supply
#' @param price price series aligned by date
token_unlock_analysis <- function(unlock_schedule, price) {
  n_unlocks <- nrow(unlock_schedule)
  n_price   <- length(price)

  # Compute unlock as % of circulating supply
  unlock_pct <- unlock_schedule$unlock_amount / unlock_schedule$total_supply

  # Event study: return around unlock dates
  window_pre  <- 7   # days before unlock
  window_post <- 30  # days after unlock

  events <- lapply(seq_len(n_unlocks), function(i) {
    t_event <- unlock_schedule$date[i]
    if (t_event <= window_pre || t_event > n_price - window_post) return(NULL)
    pre_returns  <- diff(log(price[(t_event-window_pre):t_event]))
    post_returns <- diff(log(price[t_event:(t_event+window_post)]))
    list(
      unlock_pct   = unlock_pct[i],
      pre_cum_ret  = sum(pre_returns),
      post_cum_ret = sum(post_returns),
      t_event      = t_event
    )
  })
  events <- Filter(Negate(is.null), events)

  if (length(events) == 0) {
    cat("No events in sample range\n"); return(invisible(NULL))
  }

  pre_rets  <- sapply(events, `[[`, "pre_cum_ret")
  post_rets <- sapply(events, `[[`, "post_cum_ret")
  unlk_pcts <- sapply(events, `[[`, "unlock_pct")

  # Regression: post_return ~ unlock_pct
  if (length(unlk_pcts) > 2) {
    X_reg <- cbind(1, unlk_pcts)
    b_reg <- solve(t(X_reg)%*%X_reg) %*% t(X_reg) %*% post_rets
    pred_impact <- b_reg[1] + b_reg[2] * unlk_pcts
    impact_per_pct <- b_reg[2]  # expected return per 1% unlock
  } else {
    impact_per_pct <- NA
  }

  cat("=== Token Unlock Impact Analysis ===\n")
  cat(sprintf("Events analyzed: %d\n", length(events)))
  cat(sprintf("Avg pre-unlock return (7d): %.3f%%\n", 100*mean(pre_rets)))
  cat(sprintf("Avg post-unlock return (30d): %.3f%%\n", 100*mean(post_rets)))
  if (!is.na(impact_per_pct)) {
    cat(sprintf("Price impact per 1%% unlock: %.3f%%\n", 100*impact_per_pct))
  }

  invisible(list(events=events, pre_rets=pre_rets, post_rets=post_rets,
                 impact_per_pct=impact_per_pct))
}

# -----------------------------------------------------------------------------
# 8. CRYPTO DOMINANCE REGIME ANALYSIS
# -----------------------------------------------------------------------------

#' Bitcoin dominance regime analysis
#' Rising BTC dominance = risk-off (money flowing to safety of BTC)
#' Falling BTC dominance = altcoin season
#' @param btc_dom_ts BTC market cap dominance (0-1 scale)
#' @param btc_price BTC price series
#' @param alt_index altcoin index price series
btc_dominance_regimes <- function(btc_dom_ts, btc_price, alt_index=NULL, dates=NULL) {
  n <- length(btc_dom_ts)
  if (is.null(dates)) dates <- seq_len(n)

  # Dominance trend: 30d vs 90d MA
  dom_ma30 <- as.numeric(stats::filter(btc_dom_ts, rep(1/30, 30), sides=1))
  dom_ma90 <- as.numeric(stats::filter(btc_dom_ts, rep(1/90, 90), sides=1))

  # Regime classification
  regime <- rep("NEUTRAL", n)
  valid  <- !is.na(dom_ma30) & !is.na(dom_ma90)
  regime[valid & dom_ma30 > dom_ma90] <- "BTC_DOMINANCE"   # BTC outperforming
  regime[valid & dom_ma30 < dom_ma90] <- "ALT_SEASON"       # alts outperforming

  # BTC returns by regime
  btc_rets <- c(NA, diff(log(btc_price)))
  btc_by_regime <- tapply(btc_rets, regime, mean, na.rm=TRUE)

  cat("=== BTC Dominance Regime Analysis ===\n")
  cat(sprintf("Current dominance: %.2f%%\n", 100*tail(btc_dom_ts,1)))
  cat(sprintf("Current regime: %s\n", tail(regime,1)))
  cat(sprintf("BTC dominance regime: %d days (%.1f%%)\n",
              sum(regime=="BTC_DOMINANCE"), 100*mean(regime=="BTC_DOMINANCE")))
  cat(sprintf("Alt season regime: %d days (%.1f%%)\n",
              sum(regime=="ALT_SEASON"), 100*mean(regime=="ALT_SEASON")))
  cat("Mean BTC return by regime:\n")
  print(btc_by_regime)

  if (!is.null(alt_index)) {
    alt_rets <- c(NA, diff(log(alt_index)))
    # Altcoin performance in each regime
    alt_by_regime <- tapply(alt_rets, regime, mean, na.rm=TRUE)
    cat("Mean alt return by regime:\n")
    print(alt_by_regime)
  }

  invisible(list(regime=regime, dom_ma30=dom_ma30, dom_ma90=dom_ma90))
}

# -----------------------------------------------------------------------------
# 9. STABLECOIN SUPPLY AS LIQUIDITY INDICATOR
# -----------------------------------------------------------------------------

#' Stablecoin supply growth as crypto market liquidity indicator
#' Rising stablecoin supply = dry powder waiting to enter crypto = bullish
#' @param stablecoin_supply time series of total stablecoin market cap
#' @param price crypto price to compare
stablecoin_liquidity <- function(stablecoin_supply, price, dates=NULL) {
  n <- length(stablecoin_supply)
  if (is.null(dates)) dates <- seq_len(n)

  # Stablecoin growth rates
  sc_growth_30d <- c(rep(NA,30), stablecoin_supply[31:n]/stablecoin_supply[1:(n-30)] - 1)
  sc_growth_7d  <- c(rep(NA,7),  stablecoin_supply[8:n]/stablecoin_supply[1:(n-7)] - 1)

  # Stablecoin supply ratio: stablecoins / total crypto market cap (approx)
  # Higher ratio = more liquidity relative to market cap = potential for rally
  price_returns <- c(NA, diff(log(price)))

  # Lead-lag: does SC growth predict future price?
  sc_price_corr <- sapply(1:30, function(h) {
    valid <- !is.na(sc_growth_30d) & !is.na(price_returns)
    if (sum(valid) < 50) return(NA)
    sc_h <- sc_growth_30d[valid][1:(sum(valid)-h)]
    pr_h <- price_returns[valid][(h+1):sum(valid)]
    if (length(sc_h) != length(pr_h)) return(NA)
    cor(sc_h, pr_h)
  })

  best_lag <- which.max(abs(sc_price_corr))

  cat("=== Stablecoin Liquidity Analysis ===\n")
  cat(sprintf("Latest stablecoin supply: %.2f B\n", tail(stablecoin_supply,1)/1e9))
  cat(sprintf("30-day growth: %.2f%%\n", 100*tail(sc_growth_30d,1)))
  cat(sprintf("Best predictive lag: %d days (corr=%.3f)\n",
              best_lag, sc_price_corr[best_lag]))
  cat(sprintf("Liquidity signal: %s\n",
              ifelse(tail(sc_growth_30d,1) > 0.05, "BULLISH (rising liquidity)",
              ifelse(tail(sc_growth_30d,1) < -0.05, "BEARISH (declining liquidity)",
                     "NEUTRAL"))))

  invisible(list(sc_growth_30d=sc_growth_30d, sc_growth_7d=sc_growth_7d,
                 sc_price_corr=sc_price_corr, best_lag=best_lag))
}

# -----------------------------------------------------------------------------
# 10. FEAR & GREED INDEX SIGNAL TESTING
# -----------------------------------------------------------------------------

#' Fear & Greed Index signal analysis
#' Extreme fear (0-20) = historically good buy signal (contrarian)
#' Extreme greed (80-100) = historically good sell signal (contrarian)
#' @param fng_index Fear & Greed index series (0-100)
#' @param price corresponding price series
#' @param forward_window days ahead to measure return
fear_greed_signal <- function(fng_index, price, forward_window = 30) {
  n <- length(fng_index)
  price_returns_fw <- c(rep(NA, forward_window),
                         diff(log(price), lag=forward_window))
  # Align: return_{t+fw} against fng_t
  if (length(price_returns_fw) != n) {
    price_returns_fw <- c(diff(log(price), lag=forward_window), rep(NA, forward_window))
  }

  # Classify regimes
  extreme_fear  <- fng_index <= 20
  fear          <- fng_index > 20 & fng_index <= 40
  neutral       <- fng_index > 40 & fng_index <= 60
  greed         <- fng_index > 60 & fng_index <= 80
  extreme_greed <- fng_index > 80

  regimes <- c("Extreme Fear"=sum(extreme_fear), "Fear"=sum(fear),
                "Neutral"=sum(neutral), "Greed"=sum(greed),
                "Extreme Greed"=sum(extreme_greed))

  # Forward returns by regime
  mean_ret <- function(mask) mean(price_returns_fw[mask], na.rm=TRUE)
  sr_ret   <- function(mask) {
    x <- price_returns_fw[mask]; x <- x[!is.na(x)]
    if (length(x) < 3) return(NA)
    mean(x) / sd(x) * sqrt(252/forward_window)
  }

  returns_by_regime <- data.frame(
    regime = names(regimes),
    n_obs  = as.numeric(regimes),
    mean_fw_return = c(mean_ret(extreme_fear), mean_ret(fear), mean_ret(neutral),
                       mean_ret(greed), mean_ret(extreme_greed)),
    sharpe = c(sr_ret(extreme_fear), sr_ret(fear), sr_ret(neutral),
               sr_ret(greed), sr_ret(extreme_greed))
  )

  cat("=== Fear & Greed Signal Analysis ===\n")
  cat(sprintf("Current index: %d (%s)\n", tail(fng_index,1),
              ifelse(tail(fng_index,1) <= 20, "Extreme Fear",
              ifelse(tail(fng_index,1) <= 40, "Fear",
              ifelse(tail(fng_index,1) <= 60, "Neutral",
              ifelse(tail(fng_index,1) <= 80, "Greed", "Extreme Greed"))))))
  cat(sprintf("\n%d-day forward returns by sentiment:\n", forward_window))
  print(returns_by_regime)

  invisible(returns_by_regime)
}

# -----------------------------------------------------------------------------
# 11. COMPREHENSIVE CRYPTO ANALYTICS PIPELINE
# -----------------------------------------------------------------------------

#' Run all crypto-specific analytics on available data
#' @param price_btc BTC price series
#' @param market_cap BTC market cap series
#' @param realized_cap BTC realized cap series
#' @param funding_rates perpetual funding rate series (optional)
#' @param btc_dominance BTC dominance series (optional)
run_crypto_analytics <- function(price_btc, market_cap=NULL, realized_cap=NULL,
                                  funding_rates=NULL, btc_dominance=NULL,
                                  fng_index=NULL) {
  n <- length(price_btc)
  cat("=============================================================\n")
  cat("CRYPTO ANALYTICS SUITE\n")
  cat(sprintf("Observations: %d\n\n", n))

  results <- list()

  # MVRV-Z
  if (!is.null(market_cap) && !is.null(realized_cap)) {
    cat("--- MVRV-Z Score ---\n")
    results$mvrv <- mvrv_z_score(market_cap, realized_cap)
    cat("\n")
  }

  # NVT (approximate: use market cap / proxy volume)
  if (!is.null(market_cap)) {
    cat("--- On-Chain Hash Rate Proxy ---\n")
    # Estimate realized cap from price if not provided
    if (is.null(realized_cap)) {
      cat("(Realized cap not provided, skipping NVT)\n")
    }
  }

  # Funding rates
  if (!is.null(funding_rates)) {
    cat("--- Funding Rate Analysis ---\n")
    results$funding <- funding_rate_analysis(funding_rates, price_btc)
    cat("\n")
  }

  # BTC Dominance
  if (!is.null(btc_dominance)) {
    cat("--- BTC Dominance Regimes ---\n")
    results$dominance <- btc_dominance_regimes(btc_dominance, price_btc)
    cat("\n")
  }

  # Fear & Greed
  if (!is.null(fng_index)) {
    cat("--- Fear & Greed Signal ---\n")
    results$fng <- fear_greed_signal(fng_index, price_btc)
    cat("\n")
  }

  # Return attribution
  cat("--- Price Return Summary ---\n")
  log_rets <- diff(log(price_btc))
  cat(sprintf("Total return: %.1f%%\n",  100*(tail(price_btc,1)/price_btc[1]-1)))
  cat(sprintf("Daily vol: %.4f (%.1f%% ann)\n", sd(log_rets), 100*sd(log_rets)*sqrt(365)))
  cat(sprintf("Max drawdown: %.2f%%\n",
              100*(min(price_btc)/cummax(c(price_btc[1],price_btc[1:n-1]))[which.min(price_btc)] - 1)))

  invisible(results)
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 500
# price_btc <- cumsum(c(30000, diff(exp(cumsum(rnorm(n-1, 0.001, 0.03))*30000))))
# price_btc <- pmax(price_btc, 1000)
# market_cap <- price_btc * 19e6  # ~19M BTC in circulation
# realized_cap <- price_btc * 19e6 * runif(n, 0.6, 0.9)  # typically below mkt cap
# fng_index <- pmin(100, pmax(0, round(50 + 30*sin(seq(0,6*pi,length.out=n)) + rnorm(n,0,10))))
# result <- run_crypto_analytics(price_btc, market_cap, realized_cap, fng_index=fng_index)

# =============================================================================
# EXTENDED CRYPTO ANALYTICS: On-Chain Valuation, Altcoin Beta, Momentum,
# Liquidity-Adjusted Returns, Puell Multiple, NUPL, Cross-Asset Crypto Factors
# =============================================================================

# -----------------------------------------------------------------------------
# NUPL (Net Unrealized Profit/Loss): fraction of market cap in profit
# NUPL = (market cap - realized cap) / market cap
# NUPL > 0.75: euphoria/greed; NUPL < 0: capitulation
# -----------------------------------------------------------------------------
nupl_analysis <- function(price, market_cap, realized_cap) {
  nupl <- (market_cap - realized_cap) / market_cap

  # Classify market phase
  phase <- cut(nupl,
               breaks = c(-Inf, -0.25, 0, 0.25, 0.5, 0.75, Inf),
               labels = c("capitulation","hope","optimism","belief","euphoria","greed"),
               right  = TRUE)

  # Cyclical analysis: identify tops and bottoms
  returns <- diff(log(price))
  nupl_returns_corr <- cor(nupl[-length(nupl)], returns, use = "complete.obs")

  # Rolling NUPL regime
  window <- 30
  rolling_nupl <- filter(nupl, rep(1/window, window), sides=1)

  list(
    nupl = nupl,
    phase = as.character(phase),
    phase_table = table(phase) / length(nupl),
    rolling_nupl = rolling_nupl,
    current_nupl = nupl[length(nupl)],
    current_phase = as.character(phase[length(phase)]),
    nupl_price_corr = nupl_returns_corr,
    # Historical signal stats
    mean_nupl_when_positive_return = mean(nupl[!is.na(nupl) & returns > 0], na.rm=TRUE),
    mean_nupl_when_negative_return = mean(nupl[!is.na(nupl) & returns < 0], na.rm=TRUE)
  )
}

# -----------------------------------------------------------------------------
# Puell Multiple: miner revenue relative to historical average
# High Puell = miners selling pressure; Low Puell = miner capitulation / buy signal
# Puell Multiple = daily issuance USD / 365-day MA of daily issuance USD
# -----------------------------------------------------------------------------
puell_multiple <- function(price, block_reward, n_blocks_per_day = 144,
                             window = 365) {
  # Daily miner revenue in USD
  daily_issuance_usd <- price * block_reward * n_blocks_per_day

  # Moving average of daily issuance
  ma_issuance <- filter(daily_issuance_usd, rep(1/window, window), sides=1)

  puell <- daily_issuance_usd / (ma_issuance + 1e-10)

  # Signal zones
  signal <- ifelse(puell < 0.5, "buy",
            ifelse(puell > 4.0, "sell", "neutral"))

  # Forward return analysis by Puell zone
  fwd_ret <- c(diff(log(price)), NA)
  buy_fwd  <- mean(fwd_ret[signal == "buy"  & !is.na(fwd_ret)], na.rm=TRUE)
  sell_fwd <- mean(fwd_ret[signal == "sell" & !is.na(fwd_ret)], na.rm=TRUE)

  list(
    puell_multiple = puell,
    signal = signal,
    current_puell = puell[length(puell)],
    current_signal = signal[length(signal)],
    avg_fwd_return_buy_zone  = buy_fwd  * 252,
    avg_fwd_return_sell_zone = sell_fwd * 252,
    pct_time_in_buy_zone  = mean(signal == "buy",  na.rm=TRUE),
    pct_time_in_sell_zone = mean(signal == "sell", na.rm=TRUE)
  )
}

# -----------------------------------------------------------------------------
# Altcoin Beta Matrix: compute each altcoin's beta to BTC across regimes
# Altcoin betas are time-varying and regime-dependent:
# Bull market: high beta (alts amplify BTC moves)
# Bear market: even higher beta (alts crash harder)
# Sideways: lower correlation, independent idiosyncratic moves
# -----------------------------------------------------------------------------
altcoin_beta_matrix <- function(returns_mat, btc_col = 1, window = 60) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)
  btc_ret <- returns_mat[, btc_col]
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:p)

  # Full-sample betas
  betas <- apply(returns_mat, 2, function(r) {
    cov(r, btc_ret) / var(btc_ret)
  })

  # Rolling betas
  rolling_betas <- matrix(NA, n, p)
  for (t in window:n) {
    idx <- (t-window+1):t
    btc_sub <- btc_ret[idx]
    var_btc  <- var(btc_sub)
    if (var_btc == 0) next
    for (j in 1:p) {
      rolling_betas[t, j] <- cov(returns_mat[idx, j], btc_sub) / var_btc
    }
  }
  colnames(rolling_betas) <- asset_names

  # Bull/bear conditional betas
  bull_days <- btc_ret > 0
  bear_days <- btc_ret < 0

  bull_betas <- apply(returns_mat[bull_days, , drop=FALSE], 2, function(r) {
    cov(r, btc_ret[bull_days]) / var(btc_ret[bull_days])
  })
  bear_betas <- apply(returns_mat[bear_days, , drop=FALSE], 2, function(r) {
    cov(r, btc_ret[bear_days]) / var(btc_ret[bear_days])
  })

  beta_df <- data.frame(
    asset = asset_names,
    full_beta = betas,
    bull_beta = bull_betas,
    bear_beta = bear_betas,
    asymmetry = bear_betas - bull_betas,  # positive = more downside sensitivity
    idiosyncratic_vol = apply(returns_mat, 2, function(r) {
      resid <- r - betas[which(asset_names == asset_names)] * btc_ret
      sd(resid) * sqrt(252)
    })
  )

  list(
    beta_matrix = beta_df,
    rolling_betas = rolling_betas,
    btc_vol = sd(btc_ret) * sqrt(252),
    avg_altcoin_beta = mean(betas[-btc_col])
  )
}

# -----------------------------------------------------------------------------
# Crypto Momentum Factor: cross-sectional momentum among top crypto assets
# Rank coins by 1-month return, long winners, short losers
# Crypto momentum has historically been stronger than equity momentum
# -----------------------------------------------------------------------------
crypto_momentum_factor <- function(returns_mat, formation_window = 21,
                                    holding_period = 5, top_pct = 0.25) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)
  n_top <- max(1, floor(p * top_pct))

  factor_returns <- rep(NA, n)

  for (t in (formation_window + 1):(n - holding_period)) {
    # Formation return: cumulative return over past [formation_window] days
    form_ret <- apply(returns_mat[(t-formation_window):(t-1), , drop=FALSE], 2,
                      function(r) prod(1 + r) - 1)

    # Rank: winners = top, losers = bottom
    ranks <- rank(-form_ret)
    longs  <- ranks <= n_top
    shorts <- ranks >= (p - n_top + 1)

    # Holding period return
    hold_ret <- apply(returns_mat[t:(t+holding_period-1), , drop=FALSE], 2,
                      function(r) prod(1 + r) - 1)

    long_ret  <- if(sum(longs)>0)  mean(hold_ret[longs])  else 0
    short_ret <- if(sum(shorts)>0) mean(hold_ret[shorts]) else 0

    factor_returns[t + holding_period - 1] <- (long_ret - short_ret) / 2
  }

  valid <- !is.na(factor_returns)
  r <- factor_returns[valid]
  ann_ret <- mean(r) * (252 / holding_period)
  ann_vol <- sd(r) * sqrt(252 / holding_period)

  list(
    factor_returns = factor_returns,
    ann_return = ann_ret,
    ann_vol = ann_vol,
    sharpe = if (ann_vol > 0) ann_ret / ann_vol else NA,
    win_rate = mean(r > 0),
    max_drawdown = min((cumprod(1 + r) - cummax(cumprod(1 + r))) / cummax(cumprod(1 + r)))
  )
}

# -----------------------------------------------------------------------------
# Funding Rate-Adjusted Sharpe Ratio
# Perpetual futures carry a funding rate that must be paid/received
# Longs pay shorts when funding > 0 (bullish sentiment); adjust returns accordingly
# -----------------------------------------------------------------------------
funding_adjusted_sharpe <- function(spot_returns, funding_rates,
                                     position = "long", annualize = TRUE) {
  # funding_rates: 8-hourly funding rates (typical for perps); convert to daily
  daily_funding <- if (length(funding_rates) == length(spot_returns)) {
    funding_rates
  } else {
    # Aggregate to daily if 8-hourly
    n_daily <- floor(length(funding_rates) / 3)
    sapply(1:n_daily, function(d) sum(funding_rates[((d-1)*3+1):(d*3)]))
  }

  n <- min(length(spot_returns), length(daily_funding))
  sr <- spot_returns[1:n]; fr <- daily_funding[1:n]

  # Adjust returns: long position pays funding when positive
  if (position == "long") {
    net_returns <- sr - fr  # pay funding
  } else {
    net_returns <- -sr + fr  # short: receive funding, lose from price appreciation
  }

  ann_factor <- if (annualize) sqrt(252) else 1
  sharpe_gross <- mean(sr) / sd(sr) * ann_factor
  sharpe_net   <- mean(net_returns) / sd(net_returns) * ann_factor

  list(
    gross_sharpe = sharpe_gross,
    net_sharpe = sharpe_net,
    funding_drag_annualized = mean(fr) * 252,
    avg_daily_funding_bps = mean(fr) * 1e4,
    pct_positive_funding = mean(fr > 0),
    sharpe_degradation = sharpe_gross - sharpe_net,
    position = position
  )
}

# -----------------------------------------------------------------------------
# Liquidity-Adjusted Returns: adjust raw returns for bid-ask cost
# In crypto, round-trip transaction costs vary by exchange and market cap tier
# Large-cap (BTC/ETH): ~2-5 bps; mid-cap: ~10-30 bps; small-cap: ~50-200 bps
# -----------------------------------------------------------------------------
liquidity_adjusted_returns <- function(returns, volume_vec, market_cap_vec,
                                         holding_period = 1,
                                         fee_tier = "mid") {
  fee_map <- c(large = 0.0005, mid = 0.0020, small = 0.0100, micro = 0.0250)
  if (!fee_tier %in% names(fee_map)) {
    stop("fee_tier must be: large, mid, small, or micro")
  }
  base_fee <- fee_map[fee_tier]

  # Scale fee by relative volume (illiquidity premium)
  rel_vol <- volume_vec / (mean(volume_vec) + 1)
  illiquidity_multiplier <- 1 / sqrt(pmax(rel_vol, 0.01))

  effective_fee <- base_fee * illiquidity_multiplier

  # Round-trip cost per holding period
  roundtrip_cost <- 2 * effective_fee / holding_period

  adj_returns <- returns - roundtrip_cost

  list(
    raw_returns = returns,
    adj_returns = adj_returns,
    effective_fees = effective_fee,
    avg_fee_bps = mean(effective_fee) * 1e4,
    ann_raw = mean(returns) * 252,
    ann_adj = mean(adj_returns) * 252,
    fee_drag_ann = mean(roundtrip_cost) * 252,
    sharpe_raw = mean(returns) / sd(returns) * sqrt(252),
    sharpe_adj = mean(adj_returns) / sd(adj_returns) * sqrt(252)
  )
}

# -----------------------------------------------------------------------------
# BTC Dominance Momentum Signal: predict altseason/bitcoin season transitions
# When BTC dominance falls rapidly, alts tend to outperform (alt season)
# When BTC dominance rises, capital flows back to BTC
# -----------------------------------------------------------------------------
btc_dominance_signal <- function(btc_dominance, altcoin_returns,
                                   btc_returns, window = 14) {
  n <- length(btc_dominance)

  # Dominance momentum
  dom_mom <- c(rep(NA, window), diff(btc_dominance, lag = window))

  # Classification
  alt_season <- dom_mom < -2  # dominance fell > 2% in window
  btc_season <- dom_mom > 2   # dominance rose > 2%

  # Forward return analysis
  fwd_btc  <- c(btc_returns[-1],  NA)
  fwd_alt  <- c(altcoin_returns[-1], NA)

  results <- list(
    btc_in_alt_season  = mean(fwd_btc[alt_season & !is.na(fwd_btc)],  na.rm=TRUE) * 252,
    alt_in_alt_season  = mean(fwd_alt[alt_season & !is.na(fwd_alt)],  na.rm=TRUE) * 252,
    btc_in_btc_season  = mean(fwd_btc[btc_season & !is.na(fwd_btc)],  na.rm=TRUE) * 252,
    alt_in_btc_season  = mean(fwd_alt[btc_season & !is.na(fwd_alt)],  na.rm=TRUE) * 252,
    pct_alt_season     = mean(alt_season, na.rm=TRUE),
    pct_btc_season     = mean(btc_season, na.rm=TRUE),
    dominance_momentum = dom_mom,
    current_regime     = ifelse(alt_season[n], "alt_season",
                          ifelse(btc_season[n], "btc_season", "neutral"))
  )
  results
}

# -----------------------------------------------------------------------------
# Stablecoin Dominance Signal: proxy for risk-off sentiment in crypto
# When stablecoin supply grows (or stablecoin dominance rises), traders
# are de-risking and moving to cash -- bearish for crypto overall
# -----------------------------------------------------------------------------
stablecoin_dominance_signal <- function(stablecoin_supply, total_crypto_mcap,
                                          crypto_returns, window = 7) {
  n <- length(stablecoin_supply)

  # Stablecoin dominance (pct of total crypto in stablecoins)
  sc_dominance <- stablecoin_supply / total_crypto_mcap * 100

  # Rate of change (positive = risk-off flow)
  sc_flow <- c(NA, diff(log(stablecoin_supply + 1)))

  # Rolling change in dominance
  rolling_dom_change <- c(rep(NA, window),
                           sc_dominance[(window+1):n] - sc_dominance[1:(n-window)])

  # Signal: contrarian -- high stablecoin inflows often precede recoveries
  # as dry powder accumulates
  fwd_ret <- c(crypto_returns[-1], NA)
  sc_high_inflow <- sc_flow > quantile(sc_flow, 0.8, na.rm=TRUE)
  sc_outflow <- sc_flow < quantile(sc_flow, 0.2, na.rm=TRUE)

  list(
    sc_dominance = sc_dominance,
    sc_flow_rate = sc_flow,
    rolling_dominance_change = rolling_dom_change,
    fwd_return_high_inflow  = mean(fwd_ret[sc_high_inflow & !is.na(fwd_ret)], na.rm=TRUE)*252,
    fwd_return_on_outflow   = mean(fwd_ret[sc_outflow & !is.na(fwd_ret)], na.rm=TRUE)*252,
    current_dominance = sc_dominance[n],
    current_flow = sc_flow[n],
    market_sentiment = ifelse(sc_flow[n] > 0, "risk_off", "risk_on")
  )
}

# Extended crypto analytics example:
# nupl_res <- nupl_analysis(price_btc, market_cap, realized_cap)
# puell    <- puell_multiple(price_btc, block_reward = 3.125)
# beta_mat <- altcoin_beta_matrix(returns_matrix, btc_col = 1)
# mom_fac  <- crypto_momentum_factor(returns_matrix, formation_window=21)
# fa_sharpe <- funding_adjusted_sharpe(spot_returns, funding_rates)
# dom_sig  <- btc_dominance_signal(btc_dom_series, alt_returns, btc_returns)
