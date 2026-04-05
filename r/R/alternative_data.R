## alternative_data.R
## Options flow, dark pool, satellite proxies, social sentiment
## Pure base R -- no library() calls

# Options signals
put_call_ratio <- function(put_vol, call_vol, smooth=5) {
  pc    <- put_vol/(call_vol+1)
  pc_ma <- as.numeric(stats::filter(pc, rep(1/smooth,smooth), sides=1))
  list(raw=pc, smoothed=pc_ma,
       signal=ifelse(pc>quantile(pc,.8,na.rm=TRUE),-1,
               ifelse(pc<quantile(pc,.2,na.rm=TRUE),1,0)))
}

skew_signal <- function(iv_put25, iv_atm, iv_call25, lookback=20) {
  sk <- (iv_put25-iv_atm)-(iv_call25-iv_atm)
  n  <- length(sk); z <- rep(NA_real_,n)
  for (i in seq(lookback,n)) {
    idx <- seq(i-lookback+1,i)
    z[i] <- (sk[i]-mean(sk[idx]))/(sd(sk[idx])+1e-8)
  }
  list(skew=sk, z=z, signal=ifelse(z>1.5,-1,ifelse(z< -1.5,1,0)))
}

gamma_exposure <- function(strikes, call_oi, put_oi,
                            spot, sigma, r=0.02, T_=30/252) {
  d1_fn    <- function(K) (log(spot/K)+(r+sigma^2/2)*T_)/(sigma*sqrt(T_)+1e-12)
  gamma_fn <- function(K) dnorm(d1_fn(K))/(spot*sigma*sqrt(T_)+1e-12)
  g <- sapply(strikes, gamma_fn)
  list(net=sum(g*call_oi*100)-sum(g*put_oi*100), per_strike=g)
}

# Dark pool
dark_pool_fraction <- function(total_vol, lit_vol) {
  dp   <- pmax(total_vol-lit_vol,0)/(total_vol+1)
  dpma <- as.numeric(stats::filter(dp, rep(1/10,10), sides=1))
  list(fraction=dp, smoothed=dpma,
       anomaly=abs(dp-dpma)/(sd(dp,na.rm=TRUE)+1e-8)>2)
}

amihud_illiq <- function(ret, vol, window=20) {
  ill <- abs(ret)/(vol*mean(abs(ret))/mean(vol)+1e-8)
  rol <- rep(NA_real_,length(ill))
  for (i in seq(window,length(ill)))
    rol[i] <- mean(ill[seq(i-window+1,i)])
  list(daily=ill, rolling=rol)
}

kyle_lambda <- function(ret, sv, window=60) {
  n<-length(ret); lam<-rep(NA_real_,n)
  for (i in seq(window,n)) {
    idx<-seq(i-window+1,i); r<-ret[idx]; s<-sv[idx]
    if(var(s)>1e-10) lam[i]<-cov(r,s)/var(s)
  }
  lam
}

# Volume signals
volume_z <- function(vol, window=20) {
  n<-length(vol); z<-rep(NA_real_,n)
  for (i in seq(window,n)) {
    v<-vol[seq(i-window+1,i)]; med<-median(v); mad<-median(abs(v-med))*1.4826
    z[i]<-(vol[i]-med)/(mad+1e-8)
  }
  list(z=z, signal=ifelse(z>2,1,ifelse(z< -2,-1,0)))
}

obv <- function(prices, volumes) {
  n<-length(prices); o<-numeric(n)
  for (i in 2:n) o[i]<-o[i-1]+volumes[i]*sign(prices[i]-prices[i-1])
  o
}

# Social sentiment
sentiment_momentum <- function(scores, fast=5, slow=20) {
  f <- as.numeric(stats::filter(scores, rep(1/fast,fast), sides=1))
  s <- as.numeric(stats::filter(scores, rep(1/slow,slow), sides=1))
  list(fast=f, slow=s, macd=f-s, signal=sign(f-s))
}

# Satellite proxy
shipping_signal <- function(ship_count) {
  z <- (ship_count-mean(ship_count,na.rm=TRUE))/sd(ship_count,na.rm=TRUE)
  list(zscore=z, signal=ifelse(z>1,1,ifelse(z< -1,-1,0)))
}

# Order flow
ofi <- function(bid_vol, ask_vol) {
  o    <- (bid_vol-ask_vol)/(bid_vol+ask_vol+1)
  o_ma <- as.numeric(stats::filter(o, rep(1/10,10), sides=1))
  list(ofi=o, smoothed=o_ma, signal=sign(o_ma))
}

# Crypto on-chain
exchange_flow <- function(inflow, outflow, window=7) {
  net    <- inflow-outflow
  net_ma <- as.numeric(stats::filter(net, rep(1/window,window), sides=1))
  list(net=net, smoothed=net_ma,
       signal=ifelse(net_ma>quantile(net_ma,.7,na.rm=TRUE),-1,
               ifelse(net_ma<quantile(net_ma,.3,na.rm=TRUE),1,0)))
}

funding_signal <- function(rate, thr_long=0.001, thr_short=-5e-4, window=8) {
  sma <- as.numeric(stats::filter(rate, rep(1/window,window), sides=1))
  list(smoothed=sma, signal=ifelse(sma>thr_long,-1,ifelse(sma<thr_short,1,0)))
}

# Combine
combine_signals <- function(sig_list, wts=NULL) {
  mat <- do.call(cbind, lapply(sig_list, as.numeric))
  w   <- if (!is.null(wts)) wts/sum(abs(wts)) else rep(1/ncol(mat), ncol(mat))
  list(combined=as.vector(mat %*% w), weights=w)
}

ic_analysis <- function(fac, fwd, groups=5) {
  ic  <- cor(rank(fac,na.last="keep"), rank(fwd,na.last="keep"),
              use="pairwise.complete.obs", method="spearman")
  qs  <- quantile(fac, seq(0,1,1/groups), na.rm=TRUE)
  grp <- cut(fac, qs, labels=FALSE, include.lowest=TRUE)
  qr  <- tapply(fwd, grp, mean, na.rm=TRUE)
  list(ic=ic, quintile_rets=qr,
       long_short=as.numeric(qr[groups])-as.numeric(qr[1]))
}


# ============================================================
# ADDITIONAL: MACRO SIGNALS
# ============================================================

yield_curve_factor <- function(rates_2y, rates_10y, rates_30y = NULL) {
  level <- (rates_2y + rates_10y) / 2
  slope <- rates_10y - rates_2y
  curv  <- if (!is.null(rates_30y)) rates_30y - 2*rates_10y + rates_2y else rep(NA, length(rates_2y))
  list(level=level, slope=slope, curvature=curv,
       inverted=slope<0,
       signal=ifelse(slope<0,-1,ifelse(slope>0.01,1,0)))
}

credit_cycle_indicator <- function(hy_spread, ig_spread, loans_growth,
                                    window=52) {
  spread_ratio <- hy_spread / (ig_spread+1e-8)
  z <- (spread_ratio - mean(spread_ratio,na.rm=TRUE)) /
       (sd(spread_ratio,na.rm=TRUE)+1e-8)
  loans_ma <- as.numeric(stats::filter(loans_growth, rep(1/window,window), sides=1))
  cycle_phase <- ifelse(z>1 & loans_ma<0, "tightening",
                  ifelse(z<-1 & loans_ma>0, "expansion","neutral"))
  list(spread_ratio=spread_ratio, zscore=z, loans_ma=loans_ma,
       phase=cycle_phase, risk_off=z>1.5)
}

# ============================================================
# ADDITIONAL: SENTIMENT MODELS
# ============================================================

news_sentiment_factor <- function(sentiment_scores, volume_weights=NULL,
                                   decay=0.94, window=10) {
  n <- length(sentiment_scores)
  if (is.null(volume_weights)) volume_weights <- rep(1, n)
  ewma_sent <- rep(NA, n); ewma_sent[1] <- sentiment_scores[1]
  for (i in 2:n)
    ewma_sent[i] <- decay*ewma_sent[i-1] + (1-decay)*sentiment_scores[i]
  vol_wt_sent <- sentiment_scores * volume_weights /
                 (as.numeric(stats::filter(volume_weights, rep(1/window,window), sides=1))+1e-8)
  list(ewma=ewma_sent, vol_weighted=vol_wt_sent,
       signal=sign(ewma_sent))
}

social_volume_signal <- function(tweet_count, reddit_posts, telegram_msgs=NULL,
                                  window=24) {
  total_social <- tweet_count + reddit_posts
  if (!is.null(telegram_msgs)) total_social <- total_social + telegram_msgs
  ma  <- as.numeric(stats::filter(total_social, rep(1/window,window), sides=1))
  z   <- (total_social-ma)/(sd(total_social,na.rm=TRUE)+1e-8)
  list(total=total_social, ma=ma, z=z,
       spike=z>2, signal=ifelse(z>3,1,ifelse(z< -1,-1,0)))
}

# ============================================================
# ADDITIONAL: ON-CHAIN ANALYTICS
# ============================================================

realized_cap_momentum <- function(realized_cap, window=30) {
  growth <- c(NA, diff(log(realized_cap)))
  ma     <- as.numeric(stats::filter(growth, rep(1/window,window), sides=1))
  list(growth=growth, ma=ma,
       acceleration=c(NA, diff(growth)),
       signal=ifelse(ma>0.001, 1, ifelse(ma< -0.001,-1,0)))
}

mvrv_ratio <- function(market_value, realized_value, window=90) {
  mvrv   <- market_value / (realized_value+1e-8)
  mvrv_ma <- as.numeric(stats::filter(mvrv, rep(1/window,window), sides=1))
  z <- (mvrv-mvrv_ma) / (sd(mvrv,na.rm=TRUE)+1e-8)
  list(mvrv=mvrv, smoothed=mvrv_ma, zscore=z,
       overvalued=mvrv>3.5, undervalued=mvrv<1,
       signal=ifelse(mvrv>3.5,-1,ifelse(mvrv<1,1,0)))
}

puell_multiple <- function(miner_daily_revenue, ma_365_revenue) {
  pm <- miner_daily_revenue / (ma_365_revenue+1e-8)
  list(puell=pm, overheated=pm>4, undervalued=pm<0.3,
       signal=ifelse(pm>4,-1,ifelse(pm<0.3,1,0)))
}

coin_days_destroyed <- function(coin_amounts, holding_days) {
  cdd    <- coin_amounts * holding_days
  cdd_ma <- as.numeric(stats::filter(cdd, rep(1/30,30), sides=1))
  z      <- (cdd-cdd_ma)/(sd(cdd,na.rm=TRUE)+1e-8)
  list(cdd=cdd, smoothed=cdd_ma, zscore=z,
       spike=z>2, signal=ifelse(z>2,-1,0))
}

# ============================================================
# ADDITIONAL: DERIVATIVES FLOW
# ============================================================

options_flow_imbalance <- function(call_premium, put_premium,
                                    call_delta, put_delta, window=5) {
  buy_flow  <- call_premium * call_delta
  sell_flow <- put_premium  * abs(put_delta)
  net_delta <- buy_flow - sell_flow
  nd_ma     <- as.numeric(stats::filter(net_delta, rep(1/window,window), sides=1))
  list(net_delta_flow=net_delta, smoothed=nd_ma,
       directional_skew=nd_ma>0,
       signal=sign(nd_ma))
}

iv_surface_signal <- function(atm_iv, skew_25d, fly_25d) {
  # Regime: normal (RR>0, fly normal), crisis (RR negative, fly high)
  crisis   <- skew_25d < quantile(skew_25d,.1,na.rm=TRUE) &
              fly_25d  > quantile(fly_25d, .9,na.rm=TRUE)
  complacent <- atm_iv < quantile(atm_iv,.2,na.rm=TRUE)
  z_atm <- (atm_iv - mean(atm_iv,na.rm=TRUE)) /
            (sd(atm_iv,na.rm=TRUE)+1e-8)
  list(crisis=crisis, complacent=complacent, z_atm=z_atm,
       regime=ifelse(crisis,"fear",ifelse(complacent,"greed","neutral")),
       signal=ifelse(crisis,-1,ifelse(complacent,1,0)))
}

gamma_weighted_flow <- function(strikes, call_oi, put_oi,
                                 spot, sigma, r=0.02, T_=30/252,
                                 direction=1) {
  d1_fn    <- function(K) (log(spot/K)+(r+sigma^2/2)*T_)/(sigma*sqrt(T_)+1e-12)
  gamma_fn <- function(K) dnorm(d1_fn(K))/(spot*sigma*sqrt(T_)+1e-12)
  g <- sapply(strikes, gamma_fn)
  net_gex <- sum(g*(call_oi-put_oi)*100*spot)
  list(net_gex=net_gex, by_strike=g*(call_oi-put_oi)*100*spot,
       dealer_long=net_gex>0, signal=ifelse(net_gex>0,-1,1))
}

# ============================================================
# ADDITIONAL: PORTFOLIO SIGNAL INTEGRATION
# ============================================================

signal_portfolio <- function(signal_matrix, fwd_returns, method="ic_weight") {
  n_sig <- ncol(signal_matrix)
  if (method == "ic_weight") {
    ics <- sapply(seq_len(n_sig), function(j) {
      s <- signal_matrix[,j]; f <- fwd_returns
      valid <- !is.na(s) & !is.na(f)
      if (sum(valid)<10) return(0)
      cor(rank(s[valid]), rank(f[valid]), method="spearman")
    })
    w <- ics / (sum(abs(ics))+1e-12)
  } else w <- rep(1/n_sig, n_sig)
  list(combined=as.vector(signal_matrix %*% w), weights=w,
       ic=cor(as.vector(signal_matrix %*% w),
              fwd_returns, use="complete.obs", method="spearman"))
}

alpha_decay_half_life <- function(factor, ret_1, ret_5, ret_20) {
  ic1  <- cor(factor, ret_1,  use="pairwise.complete.obs", method="spearman")
  ic5  <- cor(factor, ret_5,  use="pairwise.complete.obs", method="spearman")
  ic20 <- cor(factor, ret_20, use="pairwise.complete.obs", method="spearman")
  ics <- c(ic1, ic5, ic20); lags <- c(1,5,20)
  valid <- !is.na(ics) & ics > 0
  hl <- if (sum(valid)>=2) {
    tryCatch(approx(ics[valid], lags[valid], xout=ic1/2)$y, error=function(e) NA)
  } else NA
  list(ic_1d=ic1, ic_5d=ic5, ic_20d=ic20, half_life=hl)
}

# ============================================================
# ADDITIONAL: R/R ADVANCED SIGNALS
# ============================================================
token_velocity_signal <- function(daily_volume, circulating_supply,
                                   window=30) {
  velocity    <- daily_volume / (circulating_supply+1e-8)
  velocity_ma <- as.numeric(stats::filter(velocity, rep(1/window,window), sides=1))
  z <- (velocity-velocity_ma)/(sd(velocity,na.rm=TRUE)+1e-8)
  list(velocity=velocity, smoothed=velocity_ma, z=z,
       signal=ifelse(z>1.5,1,ifelse(z< -1.5,-1,0)))
}

whale_accumulation_signal <- function(large_transfers, price,
                                       threshold_usd=1e6, window=7) {
  large_buys  <- large_transfers[large_transfers>0]
  net_whale   <- sum(large_transfers, na.rm=TRUE)
  net_ma      <- as.numeric(stats::filter(large_transfers, rep(1/window,window), sides=1))
  normalized  <- net_whale / (mean(abs(price))+1e-8)
  list(net_flow=net_whale, smoothed=net_ma,
       normalized=normalized, accumulating=net_ma>0,
       signal=ifelse(net_ma>0,1,ifelse(net_ma<0,-1,0)))
}

volatility_surface_signal <- function(atm_iv, skew_25d, fly_25d,
                                       lookback=20) {
  n <- length(atm_iv)
  z_atm  <- (atm_iv-mean(atm_iv,na.rm=TRUE))/(sd(atm_iv,na.rm=TRUE)+1e-8)
  z_skew <- (skew_25d-mean(skew_25d,na.rm=TRUE))/(sd(skew_25d,na.rm=TRUE)+1e-8)
  z_fly  <- (fly_25d-mean(fly_25d,na.rm=TRUE))/(sd(fly_25d,na.rm=TRUE)+1e-8)
  composite <- -(z_atm + z_skew) / 2  # negative: high fear -> buy opportunity
  list(z_atm=z_atm, z_skew=z_skew, z_fly=z_fly,
       composite=composite, signal=sign(composite))
}

multi_timeframe_signal <- function(signal_daily, signal_weekly,
                                    signal_monthly, weights=c(.5,.3,.2)) {
  n <- length(signal_daily)
  if (length(signal_weekly) < n) signal_weekly <- rep(signal_weekly, length.out=n)
  if (length(signal_monthly) < n) signal_monthly <- rep(signal_monthly, length.out=n)
  combined <- weights[1]*signal_daily + weights[2]*signal_weekly +
              weights[3]*signal_monthly
  list(combined=combined, daily=signal_daily,
       signal=sign(combined))
}

cross_asset_z_score <- function(asset_returns_list, lookback=60) {
  mat <- do.call(cbind, asset_returns_list)
  n   <- nrow(mat)
  z   <- matrix(NA, n, ncol(mat))
  for (i in seq(lookback, n)) {
    idx   <- seq(i-lookback+1, i)
    mu    <- colMeans(mat[idx,],na.rm=TRUE)
    sg    <- apply(mat[idx,],2,sd,na.rm=TRUE)
    z[i,] <- (mat[i,]-mu)/(sg+1e-8)
  }
  list(z=z, cross_asset_score=rowMeans(z,na.rm=TRUE))
}


# ============================================================
# ADDITIONAL ALTERNATIVE DATA SIGNALS
# ============================================================

short_interest_signal <- function(short_interest, float_shares, window = 20) {
  sir   <- short_interest / (float_shares + 1)
  n     <- length(sir); z <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx  <- seq(i - window + 1, i)
    z[i] <- (sir[i] - mean(sir[idx])) / (sd(sir[idx]) + 1e-8)
  }
  list(sir = sir, z_score = z,
       crowded = z > 2,
       signal = ifelse(z > 1.5, -1, ifelse(z < -1, 1, 0)))
}

etf_flow_signal <- function(etf_aum, nav, window = 10) {
  flow_est  <- c(NA, diff(etf_aum) - diff(nav) * (etf_aum[-length(etf_aum)] / (nav[-length(nav)] + 1e-8)))
  flow_ma   <- as.numeric(stats::filter(flow_est, rep(1/window, window), sides=1))
  list(flow = flow_est, smoothed = flow_ma,
       signal = ifelse(flow_ma > 0, 1, -1),
       large_inflow = flow_est > quantile(flow_est, 0.9, na.rm=TRUE))
}

cot_report_signal <- function(commercial_long, commercial_short,
                               noncomm_long, noncomm_short, window = 52) {
  comm_net    <- commercial_long - commercial_short
  noncomm_net <- noncomm_long    - noncomm_short
  n  <- length(comm_net); z_comm <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx      <- seq(i - window + 1, i)
    z_comm[i] <- (comm_net[i] - mean(comm_net[idx])) / (sd(comm_net[idx]) + 1e-8)
  }
  list(commercial_net = comm_net, noncommercial_net = noncomm_net,
       z_commercial = z_comm,
       signal = ifelse(z_comm > 1, 1, ifelse(z_comm < -1, -1, 0)))
}

fund_positioning_signal <- function(gross_exposure, net_exposure,
                                     leverage, window = 13) {
  n  <- length(net_exposure); z_net <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx      <- seq(i - window + 1, i)
    z_net[i] <- (net_exposure[i] - mean(net_exposure[idx])) /
                  (sd(net_exposure[idx]) + 1e-8)
  }
  list(net_exposure = net_exposure, gross_exposure = gross_exposure,
       leverage = leverage, z_net = z_net,
       extreme_long = z_net > 2, extreme_short = z_net < -2,
       signal = ifelse(z_net > 2, -1, ifelse(z_net < -2, 1, 0)))
}

web_traffic_signal <- function(visits, window = 4) {
  visits_ma <- as.numeric(stats::filter(visits, rep(1/window, window), sides=1))
  growth    <- c(NA, diff(log(visits)))
  z         <- (visits - visits_ma) / (sd(visits, na.rm=TRUE) + 1e-8)
  list(visits = visits, smoothed = visits_ma, growth = growth, z_score = z,
       surge = z > 2,
       signal = ifelse(z > 1.5, 1, ifelse(z < -1, -1, 0)))
}

app_download_signal <- function(downloads, rank, window = 4) {
  dl_ma  <- as.numeric(stats::filter(downloads, rep(1/window, window), sides=1))
  rk_ma  <- as.numeric(stats::filter(rank, rep(1/window, window), sides=1))
  dl_z   <- (downloads - dl_ma) / (sd(downloads, na.rm=TRUE) + 1e-8)
  list(downloads = downloads, rank = rank, smoothed_downloads = dl_ma,
       smoothed_rank = rk_ma, z_score = dl_z,
       improving = dl_z > 0 & c(NA, diff(rank)) < 0,
       signal = ifelse(dl_z > 1, 1, 0))
}

satellite_data_signal <- function(parking_lot_occupancy,
                                    cargo_ship_count, window = 4) {
  occ_ma  <- as.numeric(stats::filter(parking_lot_occupancy, rep(1/window,window), sides=1))
  ship_ma <- as.numeric(stats::filter(cargo_ship_count, rep(1/window,window), sides=1))
  combined <- (parking_lot_occupancy / (occ_ma + 1e-8) +
                 cargo_ship_count / (ship_ma + 1e-8)) / 2 - 1
  list(occupancy = parking_lot_occupancy, ships = cargo_ship_count,
       composite = combined,
       signal = ifelse(combined > 0.05, 1, ifelse(combined < -0.05, -1, 0)))
}


# ─── ADDITIONAL: MACRO / SENTIMENT ───────────────────────────────────────────

news_sentiment_signal <- function(sentiment_scores, source_weights = NULL,
                                   window = 5) {
  if (!is.null(source_weights)) {
    weighted_scores <- rowSums(sentiment_scores * source_weights, na.rm=TRUE)
  } else {
    weighted_scores <- rowMeans(sentiment_scores, na.rm=TRUE)
  }
  sma <- as.numeric(stats::filter(weighted_scores, rep(1/window, window), sides=1))
  z   <- (weighted_scores - sma) / (sd(weighted_scores, na.rm=TRUE) + 1e-8)
  list(scores = weighted_scores, smoothed = sma, z_score = z,
       signal = ifelse(z > 1, 1, ifelse(z < -1, -1, 0)),
       extreme = abs(z) > 2)
}

search_trend_signal <- function(search_index, baseline_period = 52) {
  n   <- length(search_index)
  baseline <- mean(search_index[1:min(baseline_period, n)], na.rm=TRUE)
  scaled   <- search_index / (baseline + 1e-8) * 100
  trend    <- c(rep(NA, 4), diff(scaled, lag=4))
  list(scaled_index = scaled, trend = trend,
       breakout = scaled > 200,
       signal = ifelse(trend > 20, 1, ifelse(trend < -20, -1, 0)))
}

credit_card_spending_signal <- function(spending_by_category,
                                         category_weights,
                                         seasonal_adj = TRUE,
                                         window = 4) {
  composite <- as.vector(spending_by_category %*% category_weights)
  if (seasonal_adj) {
    n   <- length(composite)
    ma4 <- as.numeric(stats::filter(composite, rep(1/4, 4), sides=2))
    composite <- composite / (ma4 + 1e-8) * mean(ma4, na.rm=TRUE)
  }
  growth  <- c(NA, diff(log(composite + 1)))
  ma      <- as.numeric(stats::filter(composite, rep(1/window, window), sides=1))
  list(composite = composite, growth = growth, smoothed = ma,
       signal = ifelse(growth > 0.02, 1, ifelse(growth < -0.02, -1, 0)))
}

supply_chain_pressure_index <- function(lead_times, inventory_ratios,
                                         freight_rates, window = 3) {
  norm <- function(x) (x - mean(x, na.rm=TRUE)) / (sd(x, na.rm=TRUE) + 1e-8)
  scpi  <- (norm(lead_times) + norm(1/inventory_ratios) + norm(freight_rates)) / 3
  scpi_ma <- as.numeric(stats::filter(scpi, rep(1/window, window), sides=1))
  list(scpi = scpi, smoothed = scpi_ma,
       high_pressure = scpi > 1,
       signal = ifelse(scpi_ma > 1, -1, ifelse(scpi_ma < -1, 1, 0)))
}

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

z_score_normalize <- function(x, window = NULL) {
  if (is.null(window)) {
    (x - mean(x, na.rm=TRUE)) / (sd(x, na.rm=TRUE) + 1e-8)
  } else {
    n <- length(x); z <- rep(NA_real_, n)
    for (i in seq(window, n)) {
      idx <- seq(i - window + 1, i)
      z[i] <- (x[i] - mean(x[idx])) / (sd(x[idx]) + 1e-8)
    }
    z
  }
}

rank_normalize <- function(x) {
  r <- rank(x, na.last = "keep")
  (r - 0.5) / (sum(!is.na(r)) + 1e-8)
}

winsorize_signal <- function(x, lower_pct = 0.01, upper_pct = 0.99) {
  lo <- quantile(x, lower_pct, na.rm=TRUE)
  hi <- quantile(x, upper_pct, na.rm=TRUE)
  pmax(pmin(x, hi), lo)
}
