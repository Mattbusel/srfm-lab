## execution_quality.R
## VWAP/TWAP benchmark, slippage attribution, market impact
## Pure base R -- no library() calls

vwap <- function(prices, volumes) {
  sum(prices * volumes) / (sum(volumes) + 1e-12)
}

rolling_vwap <- function(prices, volumes, window) {
  n <- length(prices); out <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx    <- seq(i-window+1, i)
    out[i] <- sum(prices[idx]*volumes[idx]) / (sum(volumes[idx])+1e-12)
  }
  out
}

vwap_slippage_bps <- function(exec_prices, exec_volumes,
                               benchmark_vwap, direction = 1) {
  ev <- sum(exec_prices*exec_volumes) / (sum(exec_volumes)+1e-12)
  direction * (ev - benchmark_vwap) / benchmark_vwap * 1e4
}

twap <- function(prices) mean(prices)

twap_slippage_bps <- function(exec_prices, interval_prices, direction = 1) {
  direction * (mean(exec_prices) - mean(interval_prices)) / mean(interval_prices) * 1e4
}

implementation_shortfall <- function(exec_prices, exec_volumes,
                                      arrival_price, direction = 1) {
  exec_vwap <- sum(exec_prices*exec_volumes) / (sum(exec_volumes)+1e-12)
  direction * (exec_vwap - arrival_price) / arrival_price * 1e4
}

# Slippage decomposition
slippage_decomp <- function(exec_price, arrival, vwap, close, direction=1) {
  is_bps      <- direction*(exec_price - arrival)/arrival*1e4
  delay_bps   <- direction*(arrival - close)/close*1e4
  impact_bps  <- direction*(exec_price - vwap)/vwap*1e4
  timing_bps  <- direction*(vwap - close)/close*1e4
  list(IS=is_bps, delay=delay_bps, impact=impact_bps, timing=timing_bps,
       total=is_bps+delay_bps)
}

# Market impact models
sqrt_impact <- function(qty, adv, sigma, coeff=0.1) {
  pov <- qty / (adv + 1e-12)
  coeff * sigma * sqrt(pov) * 1e4
}

linear_impact <- function(qty, adv, sigma, coeff=0.05) {
  coeff * sigma * qty / adv * 1e4
}

almgren_chriss_cost <- function(total_qty, sigma, eta, gamma,
                                 T_hrs, risk_aversion=1e-6) {
  kappa <- sqrt(risk_aversion * sigma^2 / (eta + 1e-12))
  eta * total_qty^2 * kappa / (2 * tanh(kappa * T_hrs / 2) + 1e-12)
}

# Participation rate schedule
pov_schedule <- function(total_qty, volume_forecast, pov_rate=0.10) {
  target <- volume_forecast * pov_rate
  sched  <- pmin(cumsum(target), total_qty)
  list(cumulative = sched, period = c(sched[1], diff(sched)))
}

vwap_schedule <- function(total_qty, volume_profile) {
  w <- volume_profile / sum(volume_profile)
  list(schedule = round(total_qty * w), weights = w)
}

# TCA metrics
tca_metrics <- function(exec_prices, exec_sizes, arrival_prices,
                         vwap_bench, directions) {
  notional <- exec_prices * exec_sizes
  is_bps   <- directions*(exec_prices-arrival_prices)/arrival_prices*1e4
  vw_bps   <- directions*(exec_prices-vwap_bench)/vwap_bench*1e4
  list(
    mean_IS          = weighted.mean(is_bps, notional),
    mean_vwap_slip   = weighted.mean(vw_bps, notional),
    vol_IS           = sd(is_bps),
    pct_positive_IS  = mean(is_bps > 0),
    total_cost_usd   = sum(is_bps/1e4 * notional),
    total_notional   = sum(notional)
  )
}

# Effective spread and adverse selection
effective_spread <- function(trade_price, midquote) {
  2 * abs(trade_price - midquote)
}

realized_spread <- function(trade_price, midquote, future_mid, direction) {
  2 * direction * (trade_price - future_mid)
}

adverse_selection <- function(trade_price, midquote, future_mid) {
  es <- effective_spread(trade_price, midquote)
  rs <- realized_spread(trade_price, midquote, future_mid,
                         sign(trade_price - midquote))
  list(effective=es, realized=rs, adverse=es-rs)
}

# Timing score
timing_score <- function(exec_price, day_low, day_high, direction) {
  r <- day_high - day_low + 1e-8
  if (direction == 1) 1 - (exec_price - day_low)/r
  else (exec_price - day_low)/r
}

# Roll's bid-ask spread estimator
roll_spread <- function(prices) {
  dp   <- diff(prices)
  rcov <- -cov(dp[-length(dp)], dp[-1])
  2 * sqrt(max(rcov, 0))
}

# Kyle's lambda (price impact coefficient)
kyle_lambda <- function(returns, signed_volume, window=60) {
  n <- length(returns); lam <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx <- seq(i-window+1, i); r <- returns[idx]; sv <- signed_volume[idx]
    if (var(sv) > 1e-10) lam[i] <- cov(r, sv) / var(sv)
  }
  lam
}

# Amihud illiquidity
amihud <- function(returns, volume, window=20) {
  ill <- abs(returns) / (volume*mean(abs(returns))/mean(volume)+1e-8)
  rol <- rep(NA_real_, length(ill))
  for (i in seq(window, length(ill)))
    rol[i] <- mean(ill[seq(i-window+1, i)])
  list(daily=ill, rolling=rol)
}

# Execution benchmark comparison
benchmark_comparison <- function(exec_prices, exec_sizes,
                                  arrival, vwap, twap_, close, direction) {
  n   <- sum(exec_sizes)
  ewa <- sum(exec_prices*exec_sizes)/n
  data.frame(
    benchmark   = c("Arrival","VWAP","TWAP","Close"),
    bps         = c(direction*(ewa-arrival)/arrival*1e4,
                    direction*(ewa-vwap)/vwap*1e4,
                    direction*(ewa-twap_)/twap_*1e4,
                    direction*(ewa-close)/close*1e4)
  )
}

# Intraday volume profile fitting
fit_volume_profile <- function(volume_by_interval) {
  n    <- length(volume_by_interval)
  vp   <- volume_by_interval / sum(volume_by_interval)
  t_   <- seq(0, 1, length.out = n)
  # U-shape test
  n3   <- floor(n/3)
  list(profile=vp, t=t_,
       open_wt  = mean(vp[1:n3]),
       mid_wt   = mean(vp[(n3+1):(2*n3)]),
       close_wt = mean(vp[(2*n3+1):n]),
       u_shape  = mean(vp[1:n3])>mean(vp[(n3+1):(2*n3)]) &&
                  mean(vp[(2*n3+1):n])>mean(vp[(n3+1):(2*n3)]))
}


# ============================================================
# ADDITIONAL: CRYPTO EXECUTION SPECIFICS
# ============================================================

crypto_market_impact <- function(qty_usd, adv_usd, sigma, venue="cex") {
  pov   <- qty_usd / (adv_usd + 1e-8)
  coeff <- if (venue == "cex") 0.15 else 0.25
  list(impact_bps = coeff * sigma * sqrt(pov) * 1e4,
       pov = pov, annualized_impact = coeff * sigma * sqrt(pov) * 365)
}

perp_execution_cost <- function(exec_price, mark_price, index_price,
                                 funding_rate, hold_hours, qty) {
  basis_cost   <- (mark_price - index_price) / index_price * 1e4
  funding_cost <- funding_rate * hold_hours / 8 * 1e4  # 8h per period
  slippage     <- (exec_price - mark_price) / mark_price * 1e4
  list(slippage_bps = slippage, basis_bps = basis_cost,
       funding_bps = funding_cost,
       total_bps = slippage + basis_cost + funding_cost)
}

dex_execution_analysis <- function(pool_price_before, pool_price_after,
                                    trade_size, pool_liquidity, fee_rate=0.003) {
  price_impact <- (pool_price_after - pool_price_before) / pool_price_before * 1e4
  fee_bps      <- fee_rate * 1e4
  slippage_bps <- price_impact - fee_bps
  lp_depth     <- pool_liquidity / trade_size
  list(price_impact_bps = price_impact, fee_bps = fee_bps,
       slippage_bps = slippage_bps, lp_depth_ratio = lp_depth,
       total_cost_bps = price_impact + fee_bps)
}

# ============================================================
# ADDITIONAL: OPTIMAL EXECUTION
# ============================================================

almgren_chriss_optimal <- function(qty, sigma, eta, gamma,
                                    risk_aversion, T_, n=20) {
  kappa <- sqrt(risk_aversion * sigma^2 / (eta + 1e-12))
  t_    <- seq(0, T_, length.out=n+1)
  xt    <- qty * sinh(kappa*(T_-t_)) / (sinh(kappa*T_)+1e-12)
  nt    <- -diff(xt)
  cost  <- eta*sum(nt^2/diff(t_)) + 0.5*gamma*sigma^2*sum(xt[-length(xt)]^2*diff(t_))
  list(inventory=xt, trades=nt, t=t_, cost=cost, kappa=kappa)
}

risk_adjusted_twap <- function(total_qty, sigma, risk_aversion, T_, n=20) {
  # TWAP modified by risk aversion
  kappa <- sqrt(risk_aversion) * sigma
  t_    <- seq(0, T_, length.out=n+1)
  urgency <- kappa * T_
  if (urgency < 0.1) {
    # Near-linear schedule
    trades <- rep(total_qty/n, n)
  } else {
    xt     <- total_qty * sinh(kappa*(T_-t_)) / (sinh(kappa*T_)+1e-12)
    trades <- -diff(xt)
  }
  list(trades=pmax(trades,0), cumulative=cumsum(pmax(trades,0)),
       urgency=urgency)
}

# ============================================================
# ADDITIONAL: BENCHMARK CONSTRUCTION
# ============================================================

synthetic_vwap_benchmark <- function(price_series, volume_series,
                                      start_time, end_time, n_intervals=24) {
  n     <- length(price_series)
  idx   <- seq(start_time, min(end_time, n))
  pvol  <- price_series[idx] * volume_series[idx]
  tvol  <- sum(volume_series[idx])
  list(vwap = sum(pvol) / (tvol + 1e-12),
       volume_weighted_interval = tapply(pvol, cut(seq_along(idx), n_intervals),
                                          sum) / (tvol / n_intervals))
}

arrival_price_benchmark <- function(prices, order_times, fill_times) {
  sapply(seq_along(order_times), function(i)
    prices[order_times[i]])
}

# ============================================================
# ADDITIONAL: SPREAD DECOMPOSITION
# ============================================================

hasbrouck_spread_decomposition <- function(prices, trade_indicator) {
  # Hasbrouck (1993) information share
  dp    <- diff(prices); n <- length(dp)
  r_buy  <- dp[trade_indicator[-1] == 1]
  r_sell <- dp[trade_indicator[-1] == -1]
  E_buy  <- mean(r_buy,  na.rm=TRUE)
  E_sell <- mean(r_sell, na.rm=TRUE)
  # Adverse selection: asymmetric price response
  adv_sel <- (E_buy - E_sell) / 2
  spread  <- mean(abs(prices[-1] - prices[-length(prices)]))
  list(adverse_selection = adv_sel,
       spread = spread,
       adv_sel_pct = adv_sel / (spread + 1e-8))
}

glosten_milgrom_model <- function(alpha, sigma_u, sigma_i) {
  # alpha = fraction of informed traders
  lambda  <- alpha * sigma_i^2 / (alpha * sigma_i^2 + (1-alpha) * sigma_u^2 + 1e-8)
  spread  <- 2 * lambda * sigma_i
  list(lambda=lambda, spread=spread, adverse_selection=lambda,
       order_processing=spread*(1-lambda))
}

# ============================================================
# ADDITIONAL: HIGH-FREQUENCY EXECUTION
# ============================================================

latency_cost <- function(price_volatility_per_ms, latency_ms,
                          order_size, n_orders_per_day=100) {
  # Expected cost from latency in high-frequency context
  cost_per_order <- price_volatility_per_ms * sqrt(latency_ms) * order_size
  daily_cost     <- cost_per_order * n_orders_per_day
  list(cost_per_order=cost_per_order, daily_cost=daily_cost,
       annualized=daily_cost*252)
}

fill_rate_analysis <- function(submitted_orders, filled_orders,
                                order_sizes, market_conditions) {
  fill_rate   <- filled_orders / (submitted_orders + 1e-8)
  partial_fill <- fill_rate < 1 & fill_rate > 0
  mkt_impact_unfill <- (1 - fill_rate) * order_sizes * market_conditions
  list(fill_rate=fill_rate, partial_fill=partial_fill,
       opportunity_cost=mkt_impact_unfill,
       mean_fill_rate=mean(fill_rate,na.rm=TRUE))
}

queue_position_model <- function(queue_depth, order_size,
                                  arrival_rate, cancel_rate) {
  # Expected time to fill given queue position
  fill_rate_est <- arrival_rate / (queue_depth + 1e-8)
  expected_wait <- queue_depth / (arrival_rate - cancel_rate * queue_depth + 1e-8)
  list(fill_rate=fill_rate_est, expected_wait=expected_wait,
       prob_fill_1min=1-exp(-fill_rate_est*60),
       queue_priority=order_size/queue_depth)
}

# ============================================================
# ADDITIONAL: ADVANCED TCA
# ============================================================
conditional_tca <- function(is_bps, features_df, conditioning_var) {
  levels_cv <- unique(conditioning_var)
  res <- lapply(levels_cv, function(lv) {
    idx <- conditioning_var == lv
    list(level=lv, mean_IS=mean(is_bps[idx],na.rm=TRUE),
         sd_IS=sd(is_bps[idx],na.rm=TRUE), n=sum(idx))
  })
  list(by_level=res, overall=mean(is_bps,na.rm=TRUE))
}

tca_peer_benchmark <- function(own_is_bps, peer_is_bps_matrix,
                                trade_characteristics) {
  n_peers <- ncol(peer_is_bps_matrix)
  peer_means <- colMeans(peer_is_bps_matrix, na.rm=TRUE)
  peer_50  <- median(peer_means)
  peer_25  <- quantile(peer_means, .25)
  percentile <- mean(peer_means < own_is_bps)
  list(own=own_is_bps, peer_median=peer_50, peer_25th=peer_25,
       percentile=percentile*100, better_than_median=own_is_bps < peer_50)
}

factor_model_tca <- function(is_bps, pov, sigma, spread, size) {
  log_is  <- log(pmax(abs(is_bps),1e-6))
  log_pov <- log(pmax(pov,1e-6))
  log_sig <- log(pmax(sigma,1e-6))
  log_spd <- log(pmax(spread,1e-6))
  log_sz  <- log(pmax(size,1e-6))
  X  <- cbind(1, log_pov, log_sig, log_spd, log_sz)
  b  <- tryCatch(solve(t(X)%*%X+diag(5)*1e-8)%*%t(X)%*%log_is,
                 error=function(e) rep(0,5))
  names(b) <- c("intercept","pov_coef","vol_coef","spread_coef","size_coef")
  resid <- log_is - as.vector(X%*%b)
  list(coefs=b, r2=1-var(resid)/(var(log_is)+1e-8),
       systematic=exp(as.vector(X%*%b)), residual=resid)
}

# ============================================================
# ADDITIONAL: LIQUIDITY METRICS
# ============================================================
market_depth_metric <- function(bid_qty, ask_qty, bid_px, ask_px, n_levels=5) {
  l  <- min(n_levels, length(bid_qty), length(ask_qty))
  bd <- sum(bid_qty[1:l]*bid_px[1:l]); ad <- sum(ask_qty[1:l]*ask_px[1:l])
  list(bid_depth=bd, ask_depth=ad, total=(bd+ad),
       imbalance=(bd-ad)/(bd+ad+1e-8))
}

intraday_liquidity_risk <- function(volume_series, window=24) {
  n   <- length(volume_series); liq_risk <- rep(NA,n)
  for (i in seq(window,n)) {
    v   <- volume_series[seq(i-window+1,i)]
    liq_risk[i] <- sd(v)/mean(v)
  }
  list(liquidity_risk=liq_risk,
       high_risk=liq_risk>quantile(liq_risk,.8,na.rm=TRUE))
}

# ============================================================
# ADDITIONAL: ALGO PERFORMANCE
# ============================================================
algo_performance_attribution <- function(algo_results_list, benchmark) {
  n  <- length(algo_results_list)
  is_vs_bench <- sapply(algo_results_list, function(r)
    r$is_bps - benchmark$is_bps)
  list(outperformance=is_vs_bench, mean_out=mean(is_vs_bench,na.rm=TRUE),
       best_algo=which.min(is_vs_bench), worst_algo=which.max(is_vs_bench))
}

slippage_prediction_model <- function(future_is, pov, vol, spread, size) {
  X  <- cbind(1, pov, vol, spread, size, pov*vol)
  b  <- tryCatch(solve(t(X)%*%X+diag(6)*1e-8)%*%t(X)%*%future_is,
                 error=function(e) rep(0,6))
  fitted <- as.vector(X%*%b); resid <- future_is-fitted
  list(coefs=b, fitted=fitted,
       r2=1-sum(resid^2)/(sum((future_is-mean(future_is))^2)+1e-12))
}


# ============================================================
# ADDITIONAL EXECUTION QUALITY MEASURES
# ============================================================

intraday_seasonality_adjustment <- function(volume_by_bar, bar_labels,
                                             n_bars_day = 78) {
  n      <- length(volume_by_bar)
  bar_id <- ((seq_len(n) - 1) %% n_bars_day) + 1
  avg_by_bar <- tapply(volume_by_bar, bar_id, mean, na.rm = TRUE)
  expected   <- avg_by_bar[bar_id]
  adj_vol    <- volume_by_bar / (expected + 1e-8)
  list(expected = expected, adjusted = adj_vol,
       participation_score = adj_vol / mean(adj_vol, na.rm=TRUE))
}

crossing_network_analysis <- function(midpoint, trade_price, trade_size,
                                       venue_type) {
  crossing   <- venue_type == "dark"
  price_imp  <- abs(trade_price - midpoint) / (midpoint + 1e-8)
  dark_ii    <- price_imp[crossing]
  lit_ii     <- price_imp[!crossing]
  list(dark_avg_impact = mean(dark_ii, na.rm=TRUE),
       lit_avg_impact  = mean(lit_ii,  na.rm=TRUE),
       dark_fraction   = mean(crossing),
       dark_saving_bps = (mean(lit_ii, na.rm=TRUE) - mean(dark_ii, na.rm=TRUE)) * 1e4)
}

market_open_close_analysis <- function(open_ret, close_ret, intraday_ret) {
  gap_fill     <- sign(open_ret) != sign(close_ret)
  open_premium <- abs(open_ret) > abs(close_ret)
  momentum     <- sign(open_ret) == sign(intraday_ret)
  list(gap_fill_rate = mean(gap_fill, na.rm=TRUE),
       open_premium_rate = mean(open_premium, na.rm=TRUE),
       open_momentum_rate = mean(momentum, na.rm=TRUE),
       avg_open_ret = mean(abs(open_ret), na.rm=TRUE),
       avg_close_ret = mean(abs(close_ret), na.rm=TRUE))
}

fill_quality_metrics <- function(limit_fills, market_fills,
                                   midpoint_at_order, midpoint_at_fill) {
  limit_slippage  <- (midpoint_at_fill[limit_fills] -
                        midpoint_at_order[limit_fills]) / midpoint_at_order[limit_fills]
  market_slippage <- (midpoint_at_fill[market_fills] -
                        midpoint_at_order[market_fills]) / midpoint_at_order[market_fills]
  list(limit_avg_slippage_bps = mean(limit_slippage, na.rm=TRUE) * 1e4,
       market_avg_slippage_bps = mean(market_slippage, na.rm=TRUE) * 1e4,
       limit_fill_rate = mean(limit_fills),
       avg_queue_time = NA)
}

momentum_impact_correction <- function(returns, trade_signs, window = 20) {
  n     <- length(returns)
  mom   <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx  <- seq(i - window + 1, i)
    mom[i] <- mean(returns[idx], na.rm = TRUE)
  }
  momentum_component <- mom * trade_signs
  idiosyncratic      <- returns - momentum_component
  list(momentum_component = momentum_component,
       idiosyncratic = idiosyncratic,
       momentum_cost_bps = mean(momentum_component, na.rm=TRUE) * 1e4)
}


# ─── ADDITIONAL: COST ATTRIBUTION ────────────────────────────────────────────

execution_cost_budget <- function(alpha_bps, tc_bps, market_impact_bps,
                                   spread_cost_bps, opportunity_cost_bps) {
  total_cost   <- tc_bps + market_impact_bps + spread_cost_bps + opportunity_cost_bps
  net_alpha    <- alpha_bps - total_cost
  cost_breakdown <- c(tc=tc_bps, impact=market_impact_bps,
                      spread=spread_cost_bps, opportunity=opportunity_cost_bps)
  list(total_cost_bps = total_cost, net_alpha_bps = net_alpha,
       cost_breakdown = cost_breakdown,
       cost_fractions = cost_breakdown / (total_cost + 1e-8),
       is_viable = net_alpha > 0)
}

order_book_imbalance_execution <- function(bid_qty, ask_qty, side,
                                            window = 10) {
  obi <- (bid_qty - ask_qty) / (bid_qty + ask_qty + 1)
  obi_ma <- as.numeric(stats::filter(obi, rep(1/window, window), sides=1))
  exec_advantage <- ifelse(side == "buy",  obi_ma,
                    ifelse(side == "sell", -obi_ma, 0))
  list(obi = obi, smoothed = obi_ma,
       execution_advantage_bps = exec_advantage * 10,
       favorable = exec_advantage > 0)
}

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

bps_to_pct <- function(bps) bps / 1e4
pct_to_bps <- function(pct) pct * 1e4

execution_summary_report <- function(is_bps, spread_bps, impact_bps,
                                      timing_bps) {
  total <- is_bps + spread_bps + impact_bps + timing_bps
  data.frame(
    metric = c("implementation_shortfall","spread","market_impact","timing","total"),
    bps    = c(is_bps, spread_bps, impact_bps, timing_bps, total),
    pct    = c(is_bps, spread_bps, impact_bps, timing_bps, total) / 1e4
  )
}

# version
module_version <- function() "1.0.0"
module_info <- function() list(version="1.0.0", base_r_only=TRUE, pure=TRUE)
# end of file

# placeholder
.module_loaded <- TRUE
