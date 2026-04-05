## alternative_data.R
## Alternative data: options flow, dark pool, unusual volume, sentiment
## Pure base R -- no library() calls

put_call_ratio <- function(put_volume, call_volume, smooth = 5) {
  pc     <- put_volume / (call_volume + 1)
  pc_sma <- as.numeric(stats::filter(pc, rep(1/smooth, smooth), sides = 1))
  list(raw = pc, smoothed = pc_sma,
       signal = ifelse(pc > quantile(pc, 0.8, na.rm = TRUE), -1,
                ifelse(pc < quantile(pc, 0.2, na.rm = TRUE),  1, 0)))
}

options_skew_signal <- function(iv_otm_put, iv_atm, iv_otm_call, lookback = 20) {
  skew   <- (iv_otm_put - iv_atm) - (iv_otm_call - iv_atm)
  n      <- length(skew); z_skew <- rep(NA_real_, n)
  for (i in seq(lookback, n)) {
    idx       <- seq(i - lookback + 1, i)
    z_skew[i] <- (skew[i] - mean(skew[idx])) / (sd(skew[idx]) + 1e-8)
  }
  list(skew = skew, z_score = z_skew,
       signal = ifelse(z_skew > 1.5, -1, ifelse(z_skew < -1.5, 1, 0)))
}

unusual_options_activity <- function(volume, open_interest, price,
                                     vol_threshold = 3, oi_mult = 1.5) {
  vol_avg   <- as.numeric(stats::filter(volume, rep(1/20, 20), sides = 1))
  vol_ratio <- volume / (vol_avg + 1)
  unusual   <- (vol_ratio > vol_threshold) & (volume > oi_mult * open_interest)
  cost      <- volume * price
  list(vol_ratio = vol_ratio, unusual = unusual,
       dollar_premium = cost, unusual_cost = ifelse(unusual, cost, NA))
}

dealer_gamma_exposure <- function(strikes, call_oi, put_oi,
                                  spot, sigma, r = 0.02, T_exp = 30/252) {
  d1_fn    <- function(K) (log(spot/K) + (r + sigma^2/2)*T_exp) / (sigma*sqrt(T_exp))
  gamma_fn <- function(K) dnorm(d1_fn(K)) / (spot * sigma * sqrt(T_exp) + 1e-12)
  g <- sapply(strikes, gamma_fn)
  list(gamma = g, net_gamma = sum(g*call_oi*100) - sum(g*put_oi*100),
       call_gamma = sum(g*call_oi*100), put_gamma = sum(g*put_oi*100))
}

dark_pool_fraction <- function(total_volume, lit_volume) {
  dp_frac <- pmax(total_volume - lit_volume, 0) / (total_volume + 1)
  dp_sma  <- as.numeric(stats::filter(dp_frac, rep(1/10, 10), sides = 1))
  list(fraction = dp_frac, smoothed = dp_sma,
       anomaly = abs(dp_frac - dp_sma) / (sd(dp_frac, na.rm = TRUE) + 1e-8) > 2)
}

kyle_lambda <- function(returns, signed_volume, window = 60) {
  n <- length(returns); lambda <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i); r <- returns[idx]; sv <- signed_volume[idx]
    if (var(sv) > 1e-10) lambda[i] <- cov(r, sv) / var(sv)
  }
  lambda
}

amihud_illiq <- function(returns, volume, window = 20) {
  illiq_d <- abs(returns) / (volume * mean(abs(returns)) / mean(volume) + 1e-8)
  roll    <- rep(NA_real_, length(illiq_d))
  for (i in seq(window, length(illiq_d)))
    roll[i] <- mean(illiq_d[seq(i - window + 1, i)])
  list(daily = illiq_d, rolling = roll)
}

volume_zscore <- function(volume, window = 20) {
  n <- length(volume); z <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    v   <- volume[seq(i - window + 1, i)]
    med <- median(v); mad <- median(abs(v - med)) * 1.4826
    z[i] <- (volume[i] - med) / (mad + 1e-8)
  }
  list(zscore = z, signal = ifelse(z > 2, 1, ifelse(z < -2, -1, 0)))
}

volume_price_divergence <- function(prices, volume, window = 10) {
  ret    <- c(NA, diff(log(prices)))
  vp     <- volume * sign(ret)
  vp_ma  <- as.numeric(stats::filter(vp,  rep(1/window, window), sides = 1))
  ret_ma <- as.numeric(stats::filter(ret, rep(1/window, window), sides = 1))
  list(signed_volume = vp, vp_ma = vp_ma, ret_ma = ret_ma,
       divergence = sign(ret_ma) != sign(vp_ma) & !is.na(ret_ma))
}

obv_indicator <- function(prices, volume) {
  n <- length(prices); obv <- numeric(n)
  for (i in 2:n)
    obv[i] <- obv[i-1] + volume[i] * sign(prices[i] - prices[i-1])
  obv
}

vwap_deviation_signal <- function(prices, volume, window = 20) {
  n <- length(prices); vwap <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx    <- seq(i - window + 1, i)
    vwap[i] <- sum(prices[idx] * volume[idx]) / (sum(volume[idx]) + 1e-12)
  }
  dev <- (prices - vwap) / (vwap + 1e-12)
  list(vwap = vwap, deviation = dev,
       signal = ifelse(dev > 0.02, -1, ifelse(dev < -0.02, 1, 0)))
}

sentiment_momentum <- function(scores, fast = 5, slow = 20) {
  fast_ma <- as.numeric(stats::filter(scores, rep(1/fast, fast), sides = 1))
  slow_ma <- as.numeric(stats::filter(scores, rep(1/slow, slow), sides = 1))
  macd    <- fast_ma - slow_ma
  list(fast = fast_ma, slow = slow_ma, diff = macd, signal = sign(macd))
}

fear_greed_index <- function(vol_index, put_call, breadth, momentum, junk_spread) {
  norm <- function(x) (x - min(x, na.rm=TRUE)) /
                       (max(x, na.rm=TRUE) - min(x, na.rm=TRUE) + 1e-8) * 100
  fg    <- (100-norm(vol_index) + 100-norm(put_call) +
              norm(breadth) + norm(momentum) + 100-norm(junk_spread)) / 5
  label <- cut(fg, c(0,25,45,55,75,100),
               labels = c("Extreme Fear","Fear","Neutral","Greed","Extreme Greed"))
  list(index = fg, label = label)
}

order_flow_imbalance <- function(bid_vol, ask_vol) {
  ofi    <- (bid_vol - ask_vol) / (bid_vol + ask_vol + 1)
  ofi_ma <- as.numeric(stats::filter(ofi, rep(1/10, 10), sides = 1))
  list(ofi = ofi, smoothed = ofi_ma, signal = sign(ofi_ma))
}

trade_imbalance_signal <- function(buy_trades, sell_trades, window = 20) {
  n <- length(buy_trades)
  tib <- (buy_trades - sell_trades) / (buy_trades + sell_trades + 1)
  z   <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx  <- seq(i - window + 1, i)
    z[i] <- (tib[i] - mean(tib[idx])) / (sd(tib[idx]) + 1e-8)
  }
  list(imbalance = tib, zscore = z,
       signal = ifelse(z > 1, 1, ifelse(z < -1, -1, 0)))
}

exchange_net_flow_signal <- function(inflow, outflow, window = 7) {
  net    <- inflow - outflow
  net_ma <- as.numeric(stats::filter(net, rep(1/window, window), sides = 1))
  list(net = net, cumulative = cumsum(net), smoothed = net_ma,
       signal = ifelse(net_ma > quantile(net_ma, 0.7, na.rm=TRUE), -1,
                ifelse(net_ma < quantile(net_ma, 0.3, na.rm=TRUE),  1, 0)))
}

funding_rate_signal <- function(funding_rate, thr_long = 0.001,
                                thr_short = -5e-4, window = 8) {
  sma <- as.numeric(stats::filter(funding_rate, rep(1/window, window), sides = 1))
  list(rate = funding_rate, smoothed = sma,
       signal = ifelse(sma > thr_long, -1, ifelse(sma < thr_short, 1, 0)),
       cum_cost = cumsum(funding_rate))
}

liquidation_signal <- function(long_liq, short_liq, window = 24) {
  total    <- long_liq + short_liq
  n        <- length(total); roll_max <- rep(NA_real_, n)
  for (i in seq(window, n)) roll_max[i] <- max(total[seq(i-window+1, i)])
  list(total = total, net = short_liq - long_liq,
       spike = total > 2 * roll_max, signal = sign(short_liq - long_liq))
}

combine_alt_signals <- function(signal_list, method = "equal_weight", weights = NULL) {
  sig_mat <- do.call(cbind, lapply(signal_list, as.numeric))
  n_sig   <- ncol(sig_mat)
  w <- if (method == "sharpe_weight") {
    raw <- apply(sig_mat, 2, function(s) {
      mn <- mean(s, na.rm=TRUE); sd_ <- sd(s, na.rm=TRUE)
      if (sd_ < 1e-8) 0 else mn/sd_
    }); raw / (sum(abs(raw)) + 1e-12)
  } else if (!is.null(weights)) weights/sum(abs(weights))
  else rep(1/n_sig, n_sig)
  list(combined = as.vector(sig_mat %*% w), weights = w)
}

ic_analysis <- function(factor_values, fwd_returns, groups = 5) {
  rank_ic <- cor(rank(factor_values, na.last="keep"),
                 rank(fwd_returns,   na.last="keep"),
                 use="pairwise.complete.obs", method="spearman")
  qs  <- quantile(factor_values, seq(0,1,1/groups), na.rm=TRUE)
  grp <- cut(factor_values, qs, labels=FALSE, include.lowest=TRUE)
  qr  <- tapply(fwd_returns, grp, mean, na.rm=TRUE)
  list(rank_ic = rank_ic, quintile_returns = qr,
       long_short = as.numeric(qr[groups]) - as.numeric(qr[1]))
}

signal_decay <- function(factor_values, returns_vec, lags = 1:20) {
  n   <- length(factor_values)
  ics <- sapply(lags, function(lag) {
    if (lag >= n) return(NA)
    cor(rank(factor_values[1:(n-lag)]), rank(returns_vec[(lag+1):n]),
        use="pairwise.complete.obs", method="spearman")
  })
  valid <- which(!is.na(ics))
  hl    <- tryCatch(
    suppressWarnings(approx(ics[valid], lags[valid], xout=ics[valid[1]]/2)$y),
    error=function(e) NA)
  list(lags=lags, ic=ics, half_life=hl)
}

bid_ask_bounce <- function(prices, bid, ask) {
  mid    <- (bid + ask) / 2
  dp     <- diff(prices)
  rc     <- -cov(dp[-length(dp)], dp[-1])
  list(quoted_spread    = (ask - bid) / (mid + 1e-8),
       roll_spread      = 2 * sqrt(max(rc, 0)),
       effective_spread = 2 * abs(prices - mid) / (mid + 1e-8),
       price_impact     = prices - mid)
}

intraday_volume_profile <- function(volume_by_hour) {
  vp <- volume_by_hour / (mean(volume_by_hour) + 1e-8)
  n  <- length(vp); n3 <- floor(n/3)
  oa <- mean(vp[1:n3]); ma2 <- mean(vp[(n3+1):(2*n3)]); ca <- mean(vp[(2*n3+1):n])
  list(profile=vp, open_intensity=oa, mid_intensity=ma2, close_intensity=ca,
       is_u_shape = oa > ma2 && ca > ma2)
}

miner_outflow_signal <- function(miner_transfers, price, window = 14) {
  norm_flow <- miner_transfers / (price + 1e-8)
  ma        <- as.numeric(stats::filter(norm_flow, rep(1/window,window), sides=1))
  z         <- (norm_flow - ma) / (sd(norm_flow, na.rm=TRUE) + 1e-8)
  list(normalized_flow=norm_flow, smoothed=ma, zscore=z,
       signal=ifelse(z > 2, -1, 0))
}


# ============================================================
# ADDITIONAL: MACRO ALTERNATIVE DATA
# ============================================================

yield_curve_signal <- function(short_rate, long_rate, window = 20) {
  spread <- long_rate - short_rate
  spread_ma <- as.numeric(stats::filter(spread, rep(1/window, window), sides = 1))
  z <- (spread - spread_ma) / (sd(spread, na.rm = TRUE) + 1e-8)
  list(spread = spread, smoothed = spread_ma, zscore = z,
       inverted = spread < 0,
       signal = ifelse(spread < -0.001, -1, ifelse(spread > 0.01, 1, 0)))
}

credit_spread_signal <- function(hy_spread, ig_spread, rf_rate, window = 30) {
  excess_hy <- hy_spread - rf_rate
  spread_ratio <- hy_spread / (ig_spread + 1e-8)
  z <- (spread_ratio - mean(spread_ratio, na.rm = TRUE)) /
       (sd(spread_ratio, na.rm = TRUE) + 1e-8)
  list(excess_hy = excess_hy, ratio = spread_ratio, zscore = z,
       risk_off = z > 1.5, signal = ifelse(z > 1.5, -1, ifelse(z < -1.5, 1, 0)))
}

commodity_momentum <- function(commodity_prices, window = 20) {
  n <- length(commodity_prices)
  ret <- c(NA, diff(log(commodity_prices)))
  ma  <- as.numeric(stats::filter(commodity_prices, rep(1/window, window), sides = 1))
  roc <- (commodity_prices - ma) / (ma + 1e-8)
  list(ret = ret, ma = ma, roc = roc,
       signal = ifelse(roc > 0.02, 1, ifelse(roc < -0.02, -1, 0)))
}

fx_carry_signal <- function(high_yield_fx, low_yield_fx,
                             rates_high, rates_low) {
  carry     <- rates_high - rates_low
  spot_ret  <- c(NA, diff(log(high_yield_fx / low_yield_fx)))
  total_ret <- carry / 252 + spot_ret
  list(carry = carry, spot_return = spot_ret,
       total_return = total_ret,
       signal = sign(carry))
}

# ============================================================
# ADDITIONAL: EVENT-DRIVEN SIGNALS
# ============================================================

earnings_surprise_signal <- function(actual_eps, consensus_eps,
                                      prices_before, prices_after) {
  surprise_pct <- (actual_eps - consensus_eps) / (abs(consensus_eps) + 1e-8) * 100
  price_reaction <- (prices_after - prices_before) / prices_before * 100
  list(surprise_pct = surprise_pct, price_reaction = price_reaction,
       surprise_signal = sign(surprise_pct),
       drift = ifelse(surprise_pct > 5, 1, ifelse(surprise_pct < -5, -1, 0)))
}

insider_activity_signal <- function(insider_buys, insider_sells,
                                     total_shares, window = 20) {
  net_insider <- insider_buys - insider_sells
  pct_float   <- net_insider / (total_shares + 1e-8)
  ma          <- as.numeric(stats::filter(pct_float, rep(1/window, window), sides = 1))
  list(net = net_insider, pct_float = pct_float, smoothed = ma,
       cluster_buy  = insider_buys > 3 * mean(insider_buys, na.rm = TRUE),
       cluster_sell = insider_sells > 3 * mean(insider_sells, na.rm = TRUE),
       signal = sign(ma))
}

short_interest_signal <- function(short_interest, float_shares,
                                   days_to_cover, window = 10) {
  si_ratio <- short_interest / (float_shares + 1e-8)
  dtc_ma   <- as.numeric(stats::filter(days_to_cover, rep(1/window, window), sides = 1))
  z_si     <- (si_ratio - mean(si_ratio, na.rm = TRUE)) /
              (sd(si_ratio, na.rm = TRUE) + 1e-8)
  list(si_ratio = si_ratio, dtc = days_to_cover, dtc_smoothed = dtc_ma,
       zscore = z_si,
       squeeze_risk = si_ratio > 0.2 & days_to_cover > 5,
       signal = ifelse(z_si > 2, 1, ifelse(z_si < -2, -1, 0)))  # contrarian: high SI -> potential squeeze
}

# ============================================================
# ADDITIONAL: FLOW-BASED SIGNALS
# ============================================================

etf_flow_signal <- function(etf_aum, etf_nav, window = 5) {
  n       <- length(etf_aum)
  flows   <- c(NA, diff(etf_aum)) - c(NA, diff(etf_nav) / etf_nav[-1] * etf_aum[-1])
  flow_ma <- as.numeric(stats::filter(flows, rep(1/window, window), sides = 1))
  cum_flow <- cumsum(replace(flows, is.na(flows), 0))
  list(flows = flows, smoothed = flow_ma, cumulative = cum_flow,
       signal = sign(flow_ma))
}

fund_positioning_signal <- function(long_positions, short_positions,
                                     total_aum, window = 4) {
  net_long <- long_positions - short_positions
  net_pct  <- net_long / (total_aum + 1e-8)
  net_ma   <- as.numeric(stats::filter(net_pct, rep(1/window, window), sides = 1))
  z        <- (net_pct - net_ma) / (sd(net_pct, na.rm = TRUE) + 1e-8)
  list(net_long = net_long, net_pct = net_pct, smoothed = net_ma,
       zscore = z, overcrowded = z > 2,
       signal = ifelse(z > 2, -1, ifelse(z < -2, 1, 0)))  # contrarian
}

cot_report_signal <- function(comm_long, comm_short, noncomm_long, noncomm_short,
                               window = 4) {
  comm_net    <- comm_long    - comm_short
  noncomm_net <- noncomm_long - noncomm_short
  total_oi    <- comm_long + comm_short + noncomm_long + noncomm_short
  comm_pct    <- comm_net    / (total_oi + 1e-8)
  noncomm_pct <- noncomm_net / (total_oi + 1e-8)
  comm_z <- (comm_pct - mean(comm_pct, na.rm = TRUE)) /
            (sd(comm_pct, na.rm = TRUE) + 1e-8)
  list(commercial_net = comm_net, noncommercial_net = noncomm_net,
       comm_pct = comm_pct, noncomm_pct = noncomm_pct, comm_zscore = comm_z,
       signal = sign(comm_z))  # follow commercials (smart money)
}

# ============================================================
# ADDITIONAL: BLOCKCHAIN ANALYTICS
# ============================================================

nvt_ratio <- function(network_value, daily_tx_volume, window = 28) {
  nvt    <- network_value / (daily_tx_volume + 1e-8)
  nvt_ma <- as.numeric(stats::filter(nvt, rep(1/window, window), sides = 1))
  z      <- (nvt - nvt_ma) / (sd(nvt, na.rm = TRUE) + 1e-8)
  list(nvt = nvt, smoothed = nvt_ma, zscore = z,
       overvalued = z > 2, undervalued = z < -2,
       signal = ifelse(z > 2, -1, ifelse(z < -2, 1, 0)))
}

sopr_signal <- function(spent_output_profit_ratio, window = 7) {
  sopr_ma <- as.numeric(stats::filter(spent_output_profit_ratio,
                                      rep(1/window, window), sides = 1))
  list(sopr = spent_output_profit_ratio, smoothed = sopr_ma,
       above_one = spent_output_profit_ratio > 1,
       capitulation = spent_output_profit_ratio < 0.95,
       signal = ifelse(sopr_ma > 1.05, -1, ifelse(sopr_ma < 0.98, 1, 0)))
}

active_addresses_signal <- function(active_addr, price, window = 30) {
  norm_addr <- log(active_addr + 1)
  addr_ma   <- as.numeric(stats::filter(norm_addr, rep(1/window, window), sides = 1))
  nvm_ratio <- log(price + 1) / (norm_addr + 1e-8)  # NVM ratio
  z         <- (nvm_ratio - mean(nvm_ratio, na.rm = TRUE)) /
               (sd(nvm_ratio, na.rm = TRUE) + 1e-8)
  list(active_addr = active_addr, nvm = nvm_ratio, zscore = z,
       addr_growth = c(NA, diff(log(active_addr + 1))),
       signal = ifelse(z > 2, -1, ifelse(z < -2, 1, 0)))
}

hash_rate_signal <- function(hash_rate, price, difficulty, window = 14) {
  miner_rev_ratio <- price / (difficulty + 1e-8)
  hr_ma  <- as.numeric(stats::filter(hash_rate, rep(1/window, window), sides = 1))
  hr_z   <- (hash_rate - hr_ma) / (sd(hash_rate, na.rm = TRUE) + 1e-8)
  list(hash_rate = hash_rate, smoothed = hr_ma, zscore = hr_z,
       miner_revenue_ratio = miner_rev_ratio,
       capitulation = hash_rate < hr_ma * 0.9,
       signal = ifelse(hr_z > 1, 1, 0))  # high hashrate = miner confidence
}
