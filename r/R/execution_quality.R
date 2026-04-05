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
