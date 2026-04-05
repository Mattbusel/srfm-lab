## execution_research.R
## Execution quality research and TCA methodology
## Pure base R -- no library() calls

# ============================================================
# 1. IMPLEMENTATION SHORTFALL RESEARCH
# ============================================================

is_decomposition_study <- function(exec_prices, exec_volumes,
                                    arrival_prices, vwap_prices,
                                    close_prices, directions,
                                    trade_metadata=NULL) {
  notional  <- exec_prices * exec_volumes
  exec_vwap <- sum(exec_prices*exec_volumes)/sum(exec_volumes)
  is_total  <- directions*(exec_vwap - arrival_prices)/arrival_prices*1e4
  delay     <- directions*(arrival_prices-close_prices)/close_prices*1e4
  impact    <- directions*(exec_vwap-vwap_prices)/vwap_prices*1e4
  timing    <- directions*(vwap_prices-close_prices)/close_prices*1e4
  list(IS=is_total, delay=delay, impact=impact, timing=timing,
       weighted_IS=weighted.mean(is_total, notional),
       weighted_impact=weighted.mean(impact, notional),
       total_cost_usd=sum(is_total/1e4*notional))
}

market_impact_regression <- function(is_bps, participation_rate,
                                      volatility, spread) {
  log_is   <- log(pmax(abs(is_bps),1e-8))
  log_pov  <- log(pmax(participation_rate,1e-8))
  log_vol  <- log(pmax(volatility,1e-8))
  log_spd  <- log(pmax(spread,1e-8))
  X   <- cbind(1, log_pov, log_vol, log_spd)
  b   <- tryCatch(solve(t(X)%*%X+diag(4)*1e-8)%*%t(X)%*%log_is,
                  error=function(e) rep(0,4))
  fitted <- exp(as.vector(X%*%b))
  list(coefs=b, fitted=fitted,
       r2=1-sum((log_is-as.vector(X%*%b))^2)/(sum((log_is-mean(log_is))^2)+1e-12))
}

# ============================================================
# 2. OPTIMAL EXECUTION RESEARCH
# ============================================================

almgren_chriss_research <- function(sigma_grid, eta_grid, risk_aversion,
                                     total_qty, T_hrs=8) {
  results <- matrix(NA, length(sigma_grid), length(eta_grid))
  for (i in seq_along(sigma_grid)) {
    for (j in seq_along(eta_grid)) {
      kappa <- sqrt(risk_aversion*sigma_grid[i]^2/(eta_grid[j]+1e-12))
      cost  <- eta_grid[j]*total_qty^2*kappa/(2*tanh(kappa*T_hrs/2)+1e-12)
      results[i,j] <- cost
    }
  }
  list(cost_surface=results, sigma=sigma_grid, eta=eta_grid)
}

urgency_vs_cost_tradeoff <- function(total_qty, sigma, eta, gamma,
                                      risk_aversion_grid=c(1e-8,1e-6,1e-4)) {
  sapply(risk_aversion_grid, function(ra) {
    kappa <- sqrt(ra*sigma^2/(eta+1e-12))
    list(kappa=kappa, urgency=kappa, cost=eta*total_qty^2*kappa/2)
  })
}

# ============================================================
# 3. MARKET MICROSTRUCTURE RESEARCH
# ============================================================

bid_ask_spread_dynamics <- function(spreads, midquotes, volumes, window=60) {
  n <- length(spreads)
  roll_spread <- rep(NA_real_,n)
  for (i in seq(window,n))
    roll_spread[i] <- mean(spreads[seq(i-window+1,i)], na.rm=TRUE)
  # Decompose spread: order processing + inventory + adverse selection
  rv   <- c(NA,diff(log(midquotes)))
  dp   <- diff(midquotes)
  # Roll measure
  rcov <- -cov(dp[-length(dp)], dp[-1])
  roll_est <- 2*sqrt(max(rcov,0))
  list(raw=spreads, rolling=roll_spread,
       roll_estimator=roll_est,
       spread_vol_corr=cor(spreads, volumes, use="complete.obs"))
}

price_impact_study <- function(returns, signed_volume, windows=c(30,60,120)) {
  lapply(windows, function(w) {
    n <- length(returns); lam <- rep(NA_real_,n)
    for (i in seq(w,n)) {
      idx <- seq(i-w+1,i); r <- returns[idx]; sv <- signed_volume[idx]
      if (var(sv)>1e-10) lam[i] <- cov(r,sv)/var(sv)
    }
    list(window=w, lambda=lam, mean_lambda=mean(lam,na.rm=TRUE))
  })
}

# ============================================================
# 4. SLIPPAGE BENCHMARKING
# ============================================================

cross_asset_slippage <- function(trade_list, asset_classes) {
  classes <- unique(asset_classes)
  by_class <- lapply(classes, function(ac) {
    idx   <- which(asset_classes==ac)
    tds   <- trade_list[idx]
    is_v  <- sapply(tds, function(t) t$is_bps)
    list(asset_class=ac, mean_IS=mean(is_v), sd_IS=sd(is_v),
         n_trades=length(is_v))
  })
  list(by_class=by_class,
       overall=mean(sapply(trade_list, function(t) t$is_bps)))
}

venue_quality_analysis <- function(exec_prices, venue_ids,
                                    benchmark_prices, directions) {
  venues  <- unique(venue_ids)
  results <- lapply(venues, function(v) {
    idx <- which(venue_ids==v)
    is_ <- directions[idx]*(exec_prices[idx]-benchmark_prices[idx])/
           benchmark_prices[idx]*1e4
    list(venue=v, mean_IS=mean(is_), sd_IS=sd(is_), n=length(is_))
  })
  best  <- results[[which.min(sapply(results, function(r) r$mean_IS))]]
  list(by_venue=results, best_venue=best$venue)
}

# ============================================================
# 5. INTRADAY PATTERNS
# ============================================================

intraday_spread_pattern <- function(spread_by_interval, volume_by_interval) {
  n    <- length(spread_by_interval)
  vp   <- volume_by_interval/sum(volume_by_interval)
  sp   <- spread_by_interval/mean(spread_by_interval)
  n3   <- floor(n/3)
  list(spread_profile=sp, volume_profile=vp,
       open_spread  =mean(sp[1:n3]),
       mid_spread   =mean(sp[(n3+1):(2*n3)]),
       close_spread =mean(sp[(2*n3+1):n]),
       spread_vol_corr=cor(sp,vp))
}

volatility_intraday_pattern <- function(returns_by_interval) {
  vol_by_hr <- sapply(returns_by_interval, sd)*sqrt(length(returns_by_interval)*252)
  list(vol_profile=vol_by_hr/mean(vol_by_hr),
       max_vol_period=which.max(vol_by_hr),
       min_vol_period=which.min(vol_by_hr))
}

# ============================================================
# 6. TCA METHODOLOGY RESEARCH
# ============================================================

tca_factor_model <- function(is_bps, features) {
  X  <- cbind(1, features)
  b  <- tryCatch(solve(t(X)%*%X+diag(ncol(X))*1e-8)%*%t(X)%*%is_bps,
                 error=function(e) rep(0,ncol(X)))
  res <- is_bps - as.vector(X%*%b)
  r2  <- 1-sum(res^2)/(sum((is_bps-mean(is_bps))^2)+1e-12)
  list(coefs=b, residuals=res, r2=r2,
       systematic=as.vector(X%*%b)-b[1],
       idiosyncratic=res)
}

benchmark_sensitivity <- function(exec_prices, exec_volumes,
                                   benchmarks_matrix, directions) {
  n_bench <- ncol(benchmarks_matrix)
  ev  <- sum(exec_prices*exec_volumes)/sum(exec_volumes)
  bps <- apply(benchmarks_matrix, 2, function(bv) {
    bvwap <- mean(bv)
    directions*(ev-bvwap)/bvwap*1e4
  })
  list(bps_vs_benchmarks=bps, best=colnames(benchmarks_matrix)[which.min(abs(bps))],
       range=max(bps)-min(bps))
}

participation_rate_impact <- function(qty_series, adv_series,
                                       slippage_series) {
  pov <- qty_series/(adv_series+1e-8)
  fit <- tryCatch({
    log_pov <- log(pmax(pov,1e-8)); log_sl <- log(pmax(abs(slippage_series),1e-8))
    b <- coef(lm(log_sl ~ log_pov))
    list(intercept=b[1], slope=b[2],
         r2=summary(lm(log_sl ~ log_pov))$r.squared)
  }, error=function(e) list(intercept=NA, slope=NA, r2=NA))
  list(pov=pov, slippage=slippage_series, model=fit,
       predicted_impact=exp(fit$intercept)*pov^fit$slope)
}

# ============================================================
# ADDITIONAL: EMPIRICAL MARKET IMPACT RESEARCH
# ============================================================
price_impact_persistence <- function(returns, signed_volume, lags=1:10) {
  sapply(lags, function(lag) {
    n <- length(returns)
    if (lag >= n) return(NA)
    cor(signed_volume[1:(n-lag)], returns[(lag+1):n], use="complete.obs")
  })
}

intraday_seasonality_impact <- function(impact_by_interval, volume_by_interval) {
  n   <- length(impact_by_interval)
  vp  <- volume_by_interval/sum(volume_by_interval)
  ip  <- impact_by_interval/mean(impact_by_interval,na.rm=TRUE)
  list(vol_profile=vp, impact_profile=ip,
       correlation=cor(vp, ip, use="complete.obs"),
       optimal_interval=which.min(ip))
}

execution_alpha_study <- function(is_bps_series, market_returns,
                                    strategy_type="momentum") {
  # Does execution quality vary with market conditions?
  n   <- length(is_bps_series)
  ret_z <- (market_returns-mean(market_returns,na.rm=TRUE))/
            (sd(market_returns,na.rm=TRUE)+1e-8)
  corr_is_ret <- cor(is_bps_series, market_returns, use="complete.obs")
  trending_days <- abs(ret_z) > 1
  list(is_trend_days=mean(is_bps_series[trending_days],na.rm=TRUE),
       is_calm_days=mean(is_bps_series[!trending_days],na.rm=TRUE),
       corr_with_market=corr_is_ret)
}

order_size_impact_study <- function(order_sizes, is_bps, adv) {
  pov <- order_sizes / (adv+1e-8)
  small  <- pov < 0.05; medium <- pov>=0.05&pov<0.15; large <- pov>=0.15
  list(
    small_mean_IS  = mean(is_bps[small],  na.rm=TRUE),
    medium_mean_IS = mean(is_bps[medium], na.rm=TRUE),
    large_mean_IS  = mean(is_bps[large],  na.rm=TRUE),
    impact_elasticity = if(var(log(pov+1e-8))>0)
      cov(is_bps, log(pov+1e-8))/var(log(pov+1e-8)) else NA
  )
}

venue_selection_study <- function(is_by_venue, volume_by_venue, venue_names) {
  weighted_IS <- sum(is_by_venue*volume_by_venue)/sum(volume_by_venue)
  best  <- venue_names[which.min(is_by_venue)]
  worst <- venue_names[which.max(is_by_venue)]
  list(by_venue=data.frame(venue=venue_names,is=is_by_venue,vol=volume_by_venue),
       weighted_IS=weighted_IS, best_venue=best, worst_venue=worst,
       dispersion=max(is_by_venue)-min(is_by_venue))
}


# ============================================================
# ADDITIONAL EXECUTION RESEARCH
# ============================================================

optimal_execution_regime_study <- function(price_series, volume_series,
                                            order_size, n_simulations = 100) {
  n       <- length(price_series)
  vwap    <- sum(price_series * volume_series) / (sum(volume_series) + 1e-8)
  vol_est <- sd(diff(log(price_series)), na.rm=TRUE)
  avg_vol <- mean(volume_series, na.rm=TRUE)
  pov     <- order_size / (avg_vol + 1e-12)
  results <- list()
  for (horizon in c(30, 60, 120, 240)) {
    n_slices    <- floor(horizon / 5)
    slice_size  <- order_size / n_slices
    impact_cost <- slice_size * 0.01 / sqrt(avg_vol + 1e-12) * n_slices
    timing_risk <- vol_est * sqrt(horizon / 252 / 78) * order_size
    total_cost  <- impact_cost + 0.5 * timing_risk
    results[[as.character(horizon)]] <- list(
      horizon = horizon, impact_cost = impact_cost,
      timing_risk = timing_risk, total_cost = total_cost,
      optimal = FALSE)
  }
  costs       <- sapply(results, function(r) r$total_cost)
  best        <- names(which.min(costs))
  results[[best]]$optimal <- TRUE
  list(scenarios = results, optimal_horizon = best, vwap = vwap, pov = pov)
}

market_impact_model_comparison <- function(order_sizes, avg_volume,
                                            volatility, price) {
  pov <- order_sizes / (avg_volume + 1e-12)
  sqrt_impact    <- 0.1 * volatility * sqrt(pov) * price
  linear_impact  <- 0.5 * volatility * pov * price
  power_impact   <- 0.1 * volatility * pov^0.6 * price
  data.frame(
    order_size   = order_sizes,
    pov          = pov,
    sqrt_model   = sqrt_impact,
    linear_model = linear_impact,
    power_model  = power_impact,
    avg_model    = (sqrt_impact + linear_impact + power_impact) / 3
  )
}

spread_decomposition_study <- function(bid, ask, trade_price,
                                        trade_direction, n_lags = 10) {
  mid         <- (bid + ask) / 2
  quoted_half <- (ask - bid) / 2
  effective_half <- (trade_price - mid) * trade_direction
  n      <- length(trade_price)
  future_mid <- c(mid[-1], NA)
  realized_half <- (future_mid - mid) * trade_direction
  adverse_sel <- realized_half
  order_proc  <- effective_half - adverse_sel
  autocov     <- sapply(1:n_lags, function(lag) {
    dp <- diff(trade_price)
    if (lag >= length(dp)) return(NA)
    cov(dp[1:(length(dp)-lag)], dp[(lag+1):length(dp)], use="pairwise.complete.obs")
  })
  roll_est <- -2 * sqrt(max(-autocov[1], 0))
  list(quoted_spread = 2 * quoted_half,
       effective_spread = 2 * effective_half,
       adverse_selection = adverse_sel,
       order_processing = order_proc,
       roll_estimate = roll_est,
       autocovariances = autocov)
}

execution_alpha_decay <- function(signals, execution_returns,
                                   holding_periods = c(1, 5, 10, 21)) {
  ics <- sapply(holding_periods, function(h) {
    n    <- length(signals)
    if (h >= n) return(NA)
    fwd  <- sapply(1:(n - h), function(i) sum(execution_returns[i:(i+h-1)]))
    fwd  <- c(fwd, rep(NA, h))
    cor(rank(signals, na.last="keep"), rank(fwd, na.last="keep"),
        use="pairwise.complete.obs", method="spearman")
  })
  list(holding_periods = holding_periods, ics = ics,
       optimal_holding = holding_periods[which.max(ics)],
       ic_decay_rate = if (sum(!is.na(ics)) > 1)
         coef(lm(ics[!is.na(ics)] ~ holding_periods[!is.na(ics)]))[2] else NA)
}

venue_quality_benchmark <- function(venues, fill_rates, avg_spreads,
                                     market_impact, latency_ms) {
  score <- 0.3 * fill_rates +
           0.3 * (1 - avg_spreads / (max(avg_spreads) + 1e-8)) +
           0.2 * (1 - market_impact / (max(market_impact) + 1e-8)) +
           0.2 * (1 - latency_ms / (max(latency_ms) + 1e-8))
  df <- data.frame(venue = venues, fill_rate = fill_rates,
                   avg_spread = avg_spreads, market_impact = market_impact,
                   latency_ms = latency_ms, composite_score = score)
  df[order(-df$composite_score), ]
}


# ─── ADDITIONAL EXECUTION RESEARCH ────────────────────────────────────────────

transaction_cost_model_validation <- function(predicted_cost_bps,
                                               actual_cost_bps,
                                               order_metadata) {
  errors     <- actual_cost_bps - predicted_cost_bps
  mae        <- mean(abs(errors), na.rm=TRUE)
  rmse       <- sqrt(mean(errors^2, na.rm=TRUE))
  bias       <- mean(errors, na.rm=TRUE)
  r_squared  <- 1 - var(errors, na.rm=TRUE) /
                      (var(actual_cost_bps, na.rm=TRUE) + 1e-8)
  overestimate_pct <- mean(errors < 0, na.rm=TRUE)
  list(mae=mae, rmse=rmse, bias=bias, r_squared=r_squared,
       overestimate_pct=overestimate_pct,
       well_calibrated = abs(bias) < 0.5 * mae)
}

liquidity_timing_study <- function(volume_profile, bid_ask_series,
                                    optimal_participation = 0.1) {
  n          <- length(volume_profile)
  total_vol  <- sum(volume_profile, na.rm=TRUE)
  participation_schedule <- volume_profile / (total_vol + 1e-12) * optimal_participation
  cost_by_time <- bid_ask_series / 2 + participation_schedule * 0.01
  optimal_time <- which.min(cost_by_time)
  list(participation = participation_schedule,
       cost_by_time = cost_by_time,
       optimal_execution_time = optimal_time,
       avg_spread = mean(bid_ask_series, na.rm=TRUE),
       min_cost_time = optimal_time)
}

slippage_factor_attribution <- function(slippage_bps, pov, volatility,
                                          bid_ask, urgency) {
  n <- length(slippage_bps)
  X <- cbind(1, pov, volatility, bid_ask, urgency)
  valid <- complete.cases(X, slippage_bps)
  if (sum(valid) < 10) return(list(error="insufficient data"))
  beta <- tryCatch(solve(t(X[valid,]) %*% X[valid,]) %*%
                     t(X[valid,]) %*% slippage_bps[valid],
                   error=function(e) rep(NA, ncol(X)))
  fitted    <- as.vector(X %*% beta)
  residuals <- slippage_bps - fitted
  list(coefficients = setNames(as.vector(beta),
                               c("intercept","pov","vol","spread","urgency")),
       fitted = fitted, residuals = residuals,
       r_squared = 1 - var(residuals, na.rm=TRUE) /
                       (var(slippage_bps, na.rm=TRUE) + 1e-8),
       pov_impact = beta[2], vol_impact = beta[3])
}

execution_quality_scorecard <- function(is_bps, timing_bps, spread_bps,
                                          market_impact_bps, benchmark = "vwap") {
  total_cost   <- is_bps + timing_bps + spread_bps + market_impact_bps
  peer_median  <- median(total_cost, na.rm=TRUE)
  pctile       <- ecdf(total_cost)(total_cost) * 100
  grade <- cut(100 - pctile,
               breaks=c(0,20,40,60,80,100),
               labels=c("D","C","B","A","A+"),
               include.lowest=TRUE)
  list(total_cost_bps = total_cost,
       is_bps = is_bps, timing_bps = timing_bps,
       spread_bps = spread_bps, market_impact_bps = market_impact_bps,
       peer_percentile = pctile, grade = grade,
       vs_peer_median_bps = total_cost - peer_median)
}

intraday_momentum_effect_study <- function(returns_5min, time_of_day,
                                            window = 10) {
  open_period  <- time_of_day <= 60
  close_period <- time_of_day >= 330
  mid_period   <- !open_period & !close_period
  auto_corr_fn <- function(x) {
    n <- length(x); if (n < 3) return(NA)
    cor(x[-length(x)], x[-1], use="pairwise.complete.obs")
  }
  list(open_autocorr  = auto_corr_fn(returns_5min[open_period]),
       mid_autocorr   = auto_corr_fn(returns_5min[mid_period]),
       close_autocorr = auto_corr_fn(returns_5min[close_period]),
       open_vol   = sd(returns_5min[open_period],  na.rm=TRUE),
       close_vol  = sd(returns_5min[close_period], na.rm=TRUE),
       u_shape_vol = sd(returns_5min[open_period], na.rm=TRUE) >
                     sd(returns_5min[mid_period],   na.rm=TRUE) &
                     sd(returns_5min[close_period], na.rm=TRUE) >
                     sd(returns_5min[mid_period],   na.rm=TRUE))
}

# ─── UTILITY / HELPER FUNCTIONS ───────────────────────────────────────────────

annualized_tc_drag <- function(daily_turnover, tc_bps) {
  daily_cost <- daily_turnover * tc_bps / 1e4
  list(daily_cost = daily_cost,
       annual_cost = mean(daily_cost, na.rm=TRUE) * 252,
       cumulative  = cumsum(daily_cost))
}

optimal_order_size <- function(daily_volume, max_pov = 0.1,
                                min_size = 1000, max_size = 1e6) {
  raw_size <- daily_volume * max_pov
  pmax(pmin(raw_size, max_size), min_size)
}

execution_pnl_attribution <- function(alpha_entry, alpha_exit,
                                       tc_entry_bps, tc_exit_bps,
                                       holding_return) {
  gross_pnl   <- alpha_entry + holding_return - alpha_exit
  net_pnl     <- gross_pnl - (tc_entry_bps + tc_exit_bps) / 1e4
  list(gross = gross_pnl, net = net_pnl,
       cost_drag = (tc_entry_bps + tc_exit_bps) / 1e4,
       timing_value = alpha_entry + (-alpha_exit),
       selection_value = holding_return)
}

reversion_to_vwap_study <- function(price_series, vwap_series,
                                      horizon = 5) {
  deviation     <- (price_series - vwap_series) / (vwap_series + 1e-8)
  n             <- length(price_series)
  fwd_reversion <- c(rep(NA, horizon),
                     (vwap_series[(horizon+1):n] - price_series[1:(n-horizon)]) /
                       (price_series[1:(n-horizon)] + 1e-8))
  ic <- cor(deviation, fwd_reversion, use="pairwise.complete.obs", method="spearman")
  list(deviation = deviation, fwd_reversion = fwd_reversion,
       ic = ic, reversion_detected = ic < -0.1)
}

# research module metadata
research_version <- function() "1.0.0"
research_info    <- function() list(version="1.0.0", base_r_only=TRUE)
# utility: safe correlation
safe_cor <- function(x, y, method="pearson") {
  tryCatch(cor(x, y, use="pairwise.complete.obs", method=method), error=function(e) NA)
}
# utility: rolling mean
roll_mean <- function(x, w) as.numeric(stats::filter(x, rep(1/w, w), sides=1))
# utility: annualize return
annualize_ret <- function(r, periods_per_year=252) mean(r, na.rm=TRUE) * periods_per_year
# utility: annualize vol
annualize_vol <- function(r, periods_per_year=252) sd(r, na.rm=TRUE) * sqrt(periods_per_year)
# end of file

# ─── ADDITIONAL UTILITY ───────────────────────────────────────────────────────
market_hours_filter <- function(timestamps, market_open = 9.5, market_close = 16) {
  hour_frac <- as.numeric(format(timestamps, "%H")) +
                 as.numeric(format(timestamps, "%M")) / 60
  hour_frac >= market_open & hour_frac <= market_close
}

order_size_bins <- function(order_sizes, breaks = c(0, 1e3, 1e4, 1e5, 1e6, Inf)) {
  cut(order_sizes, breaks=breaks,
      labels=c("nano","small","medium","large","block"),
      include.lowest=TRUE)
}

participation_rate_to_horizon <- function(order_size, avg_volume,
                                           pov_target = 0.05) {
  needed_volume  <- order_size / pov_target
  horizon_days   <- needed_volume / (avg_volume + 1e-12)
  list(horizon_days = horizon_days, horizon_minutes = horizon_days * 390,
       feasible = horizon_days < 5)
}
# execution research module loaded
.execution_research_loaded <- TRUE
