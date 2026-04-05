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
