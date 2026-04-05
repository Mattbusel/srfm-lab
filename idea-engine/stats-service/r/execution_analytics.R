## execution_analytics.R
## VWAP/TWAP analysis, slippage, market impact, TCA
## Pure base R -- no library() calls

# ============================================================
# 1. VWAP CALCULATION AND BENCHMARKING
# ============================================================

compute_vwap <- function(prices, volumes, period = NULL) {
  if (is.null(period)) {
    vwap <- sum(prices * volumes) / (sum(volumes) + 1e-12)
    return(list(vwap = vwap))
  }
  n   <- length(prices)
  out <- rep(NA_real_, n)
  for (i in seq(period, n)) {
    idx    <- seq(i - period + 1, i)
    out[i] <- sum(prices[idx] * volumes[idx]) / (sum(volumes[idx]) + 1e-12)
  }
  out
}

vwap_slippage <- function(exec_prices, exec_volumes, vwap_bench,
                           direction = 1) {
  # direction: +1 buy, -1 sell
  exec_vwap  <- sum(exec_prices * exec_volumes) / (sum(exec_volumes) + 1e-12)
  slippage   <- direction * (exec_vwap - vwap_bench) / vwap_bench * 1e4  # bps
  list(exec_vwap = exec_vwap, benchmark_vwap = vwap_bench,
       slippage_bps = slippage)
}

vwap_schedule <- function(total_qty, volume_profile, n_periods) {
  profile_norm <- volume_profile / sum(volume_profile)
  schedule     <- round(total_qty * profile_norm[seq_len(n_periods)])
  # Adjust for rounding
  diff_qty <- total_qty - sum(schedule)
  schedule[which.max(profile_norm)] <- schedule[which.max(profile_norm)] + diff_qty
  list(schedule = schedule, pct = schedule / total_qty)
}

intraday_vwap_trajectory <- function(prices, volumes, timestamps) {
  n      <- length(prices)
  cum_vp <- cumsum(prices * volumes)
  cum_v  <- cumsum(volumes)
  list(vwap_trajectory = cum_vp / (cum_v + 1e-12),
       timestamps = timestamps,
       final_vwap = sum(prices * volumes) / (sum(volumes) + 1e-12))
}

# ============================================================
# 2. TWAP
# ============================================================

twap_schedule <- function(total_qty, n_periods, randomize = 0.1, seed = 42) {
  set.seed(seed)
  base <- rep(total_qty / n_periods, n_periods)
  if (randomize > 0) {
    noise <- base * runif(n_periods, -randomize, randomize)
    base  <- base + noise
    base  <- pmax(base, 0)
    base  <- base / sum(base) * total_qty
  }
  list(schedule = round(base), pct = base / total_qty)
}

twap_execution_benchmark <- function(exec_prices, exec_times,
                                      interval_prices) {
  twap_bench <- mean(interval_prices)
  exec_twap  <- mean(exec_prices)
  slippage   <- (exec_twap - twap_bench) / twap_bench * 1e4
  list(exec_twap = exec_twap, benchmark_twap = twap_bench,
       slippage_bps = slippage)
}

# ============================================================
# 3. MARKET IMPACT MODELS
# ============================================================

# Almgren-Chriss linear temporary impact
almgren_chriss_impact <- function(q, sigma, eta, gamma, T_,
                                  risk_aversion = 1e-6) {
  # Optimal execution: minimize E[cost] + lambda * Var[cost]
  # kappa^2 = risk_aversion * sigma^2 / eta
  kappa2 <- risk_aversion * sigma^2 / eta
  kappa  <- sqrt(kappa2)
  # Optimal trajectory
  t_grid <- seq(0, T_, length.out = 100)
  n_j    <- q * sinh(kappa * (T_ - t_grid)) / sinh(kappa * T_)
  trade_rate <- q * kappa * cosh(kappa * (T_ - t_grid)) / sinh(kappa * T_)
  list(trajectory = n_j, trade_rate = trade_rate, t = t_grid,
       expected_cost = eta * q^2 * kappa / (2 * tanh(kappa * T_ / 2) + 1e-12))
}

# Square-root market impact (empirical)
sqrt_impact_model <- function(qty, adv, sigma_daily,
                               participation_rate = NULL,
                               coeff = 0.1) {
  if (is.null(participation_rate))
    participation_rate <- qty / (adv + 1e-12)
  impact_bps <- coeff * sigma_daily * sqrt(participation_rate) * 1e4
  list(impact_bps = impact_bps,
       impact_pct = impact_bps / 1e4,
       participation_rate = participation_rate)
}

# Power-law impact (Barra)
power_law_impact <- function(qty, adv, sigma, alpha = 0.6, beta = 1.0,
                              coeff = 0.314) {
  x   <- qty / adv
  imp <- coeff * sigma * sign(qty) * abs(x)^alpha * beta
  list(impact = imp, impact_bps = imp * 1e4)
}

# ============================================================
# 4. SLIPPAGE ATTRIBUTION
# ============================================================

slippage_attribution <- function(exec_price, arrival_price,
                                  close_price, vwap,
                                  direction = 1, notional) {
  # Implementation Shortfall decomposition
  is_total       <- direction * (exec_price - arrival_price) / arrival_price * 1e4
  delay_cost     <- direction * (arrival_price - close_price) / close_price * 1e4
  market_impact  <- direction * (exec_price - vwap) / vwap * 1e4
  timing_cost    <- direction * (vwap - close_price) / close_price * 1e4
  spread_cost    <- abs(exec_price - arrival_price) / arrival_price * 1e4 / 2

  list(
    implementation_shortfall = is_total,
    delay_cost      = delay_cost,
    market_impact   = market_impact,
    timing_cost     = timing_cost,
    spread_cost     = spread_cost,
    total_cost_bps  = is_total + delay_cost,
    total_cost_usd  = (is_total + delay_cost) / 1e4 * notional
  )
}

# ============================================================
# 5. TRANSACTION COST ANALYSIS (TCA)
# ============================================================

tca_summary <- function(trades_df) {
  # trades_df: data.frame with cols exec_price, arrival_price, vwap,
  #            close_price, direction, qty, price
  notional     <- trades_df$qty * trades_df$price
  is_bps       <- trades_df$direction *
                  (trades_df$exec_price - trades_df$arrival_price) /
                   trades_df$arrival_price * 1e4
  vwap_slip    <- trades_df$direction *
                  (trades_df$exec_price - trades_df$vwap) /
                   trades_df$vwap * 1e4
  total_cost   <- sum(is_bps * notional / 1e4, na.rm = TRUE)
  list(
    mean_IS_bps       = weighted.mean(is_bps, notional, na.rm = TRUE),
    mean_vwap_slip    = weighted.mean(vwap_slip, notional, na.rm = TRUE),
    total_cost_usd    = total_cost,
    total_notional    = sum(notional),
    cost_as_pct       = total_cost / sum(notional) * 100
  )
}

trade_timing_score <- function(exec_price, day_low, day_high, direction) {
  # 0 = worst, 1 = best
  range_ <- day_high - day_low + 1e-8
  if (direction == 1)  # buy: closer to low is better
    1 - (exec_price - day_low) / range_
  else                  # sell: closer to high is better
    (exec_price - day_low) / range_
}

# ============================================================
# 6. OPTIMAL EXECUTION STRATEGIES
# ============================================================

is_optimal_schedule <- function(total_qty, sigma, eta, gamma,
                                 risk_aversion, T_, n_steps = 20) {
  kappa  <- sqrt(risk_aversion * sigma^2 / (eta + 1e-12))
  t_grid <- seq(0, T_, length.out = n_steps + 1)
  n_j    <- total_qty * sinh(kappa*(T_ - t_grid)) / (sinh(kappa*T_) + 1e-12)
  trades <- -diff(n_j)
  list(inventory = n_j, trades = trades, t = t_grid,
       urgency = kappa * T_)
}

pov_schedule <- function(total_qty, volume_forecast, pov_rate = 0.10) {
  # Percentage of Volume
  target_per_period <- volume_forecast * pov_rate
  cumtarget <- cumsum(target_per_period)
  schedule  <- pmin(diff(c(0, pmin(cumtarget, total_qty))), 0)
  schedule  <- pmax(schedule, 0)
  list(schedule = schedule, cumulative = cumsum(schedule))
}

# ============================================================
# 7. MICROSTRUCTURE METRICS
# ============================================================

effective_spread <- function(trade_prices, midquotes) {
  2 * abs(trade_prices - midquotes)
}

realized_spread <- function(trade_prices, midquotes, future_midquotes) {
  direction <- sign(trade_prices - midquotes)
  2 * direction * (trade_prices - future_midquotes)
}

price_impact_permanent <- function(trade_prices, midquotes, future_midquotes) {
  direction <- sign(trade_prices - midquotes)
  direction * (future_midquotes - midquotes)
}

adverse_selection_cost <- function(trade_prices, midquotes, future_midquotes) {
  es <- effective_spread(trade_prices, midquotes)
  rs <- realized_spread(trade_prices, midquotes, future_midquotes)
  list(effective = es, realized = rs, adverse_selection = es - rs)
}

# ============================================================
# 8. EXECUTION QUALITY REPORTING
# ============================================================

execution_quality_report <- function(exec_prices, exec_volumes,
                                      arrival_prices, vwap_bench,
                                      twap_bench, close_prices,
                                      directions) {
  n          <- length(exec_prices)
  notional   <- exec_prices * exec_volumes
  is_bps     <- directions * (exec_prices - arrival_prices) /
                arrival_prices * 1e4
  vwap_bps   <- directions * (exec_prices - vwap_bench) /
                vwap_bench * 1e4
  twap_bps   <- directions * (exec_prices - twap_bench) /
                twap_bench * 1e4
  mktcl_bps  <- directions * (exec_prices - close_prices) /
                close_prices * 1e4

  list(
    n_trades             = n,
    total_notional       = sum(notional),
    mean_IS_bps          = weighted.mean(is_bps, notional),
    mean_vwap_bps        = weighted.mean(vwap_bps, notional),
    mean_twap_bps        = weighted.mean(twap_bps, notional),
    mean_vs_close_bps    = weighted.mean(mktcl_bps, notional),
    vol_IS_bps           = sd(is_bps),
    pct_positive_IS      = mean(is_bps > 0),
    total_cost_usd       = sum(is_bps / 1e4 * notional)
  )
}


# ============================================================
# ADDITIONAL: ADVANCED EXECUTION ALGORITHMS
# ============================================================

adaptive_pov_schedule <- function(total_qty, volume_forecast,
                                   realized_volume, base_pov = 0.10,
                                   max_pov = 0.30) {
  n          <- length(volume_forecast)
  remaining  <- total_qty
  schedule   <- numeric(n)
  for (i in seq_len(n)) {
    actual_vol <- if (i <= length(realized_volume)) realized_volume[i] else volume_forecast[i]
    pov_adj    <- pmin(base_pov * total_qty / (remaining + 1e-8), max_pov)
    qty_i      <- pmin(pov_adj * actual_vol, remaining)
    schedule[i] <- qty_i
    remaining   <- remaining - qty_i
    if (remaining <= 0) break
  }
  list(schedule = schedule, cumulative = cumsum(schedule),
       completion_pct = cumsum(schedule) / total_qty)
}

bayesian_vwap <- function(total_qty, prior_volume_profile,
                           observed_volumes, time_remaining) {
  n_total  <- length(prior_volume_profile)
  n_obs    <- length(observed_volumes)
  n_remain <- n_total - n_obs
  if (n_remain <= 0) return(list(schedule = numeric(0)))

  # Posterior update on volume profile
  alpha_prior  <- prior_volume_profile * 10  # Dirichlet prior
  alpha_post   <- alpha_prior
  alpha_post[1:n_obs] <- alpha_post[1:n_obs] + observed_volumes
  post_mean    <- alpha_post / sum(alpha_post)

  # Remaining schedule proportional to posterior remaining volume
  remain_vol_pct <- post_mean[(n_obs+1):n_total]
  remain_vol_pct <- remain_vol_pct / sum(remain_vol_pct)
  sched <- round(total_qty * remain_vol_pct)
  list(schedule = sched, posterior_profile = post_mean,
       expected_completion = cumsum(sched))
}

# ============================================================
# ADDITIONAL: PRE/POST TRADE ANALYTICS
# ============================================================

pre_trade_cost_estimate <- function(qty, adv, sigma, spread_bps,
                                     model = "sqrt") {
  pov <- qty / (adv + 1e-8)
  impact_est <- switch(model,
    sqrt   = 0.1 * sigma * sqrt(pov) * 1e4,
    linear = 0.05 * sigma * pov * 1e4,
    power  = 0.314 * sigma * pov^0.6 * 1e4
  )
  timing_est  <- sigma * sqrt(qty / adv) * 0.5 * 1e4
  spread_cost <- spread_bps / 2
  list(market_impact = impact_est, timing = timing_est,
       spread = spread_cost, total = impact_est + timing_est + spread_cost,
       model = model)
}

post_trade_reconciliation <- function(exec_data, pre_trade_estimate) {
  actual_cost <- exec_data$is_bps
  prediction_error <- actual_cost - pre_trade_estimate$total
  list(actual = actual_cost, estimated = pre_trade_estimate$total,
       error = prediction_error, error_pct = prediction_error / pre_trade_estimate$total,
       model_r2 = 1 - var(prediction_error) / (var(actual_cost) + 1e-8))
}

# ============================================================
# ADDITIONAL: LIQUIDITY ANALYSIS
# ============================================================

bid_ask_decomposition <- function(prices, midquotes, trade_directions) {
  # Decompose spread into components (Glosten-Harris)
  half_spread  <- prices - midquotes
  spread_obs   <- abs(half_spread) * 2
  # Adverse selection component
  dp_mid  <- c(NA, diff(midquotes))
  adv_sel <- trade_directions * dp_mid
  order_proc <- spread_obs - 2 * abs(adv_sel)
  list(total_spread = spread_obs,
       adverse_selection = 2 * abs(adv_sel),
       order_processing = pmax(order_proc, 0),
       inventory = pmax(spread_obs - order_proc - 2*abs(adv_sel), 0))
}

liquidity_adjusted_var <- function(returns, bid_ask_spreads,
                                    position_size, alpha = 0.05) {
  var_market  <- quantile(returns, alpha)
  liquidity_cost <- mean(bid_ask_spreads) / 2  # half spread
  lvar        <- var_market - liquidity_cost
  exogenous   <- sd(bid_ask_spreads) * qnorm(alpha) / 2
  list(market_var = var_market, liquidity_cost = liquidity_cost,
       lvar = lvar, exogenous_risk = exogenous,
       total_lvar = lvar + exogenous)
}

# ============================================================
# ADDITIONAL: CRYPTO EXECUTION SPECIFICS
# ============================================================

crypto_slippage_model <- function(qty_usd, adv_usd, sigma_1h,
                                   venue = "cex") {
  # Crypto markets: higher impact due to 24/7 and thinner books
  pov     <- qty_usd / (adv_usd + 1e-8)
  base_coeff <- if (venue == "cex") 0.15 else 0.25  # DEX has higher impact
  impact  <- base_coeff * sigma_1h * sqrt(pov) * 1e4
  spread_est <- if (venue == "cex") 2 else 20  # bps
  list(impact_bps = impact, spread_bps = spread_est,
       total_bps = impact + spread_est / 2,
       pov = pov)
}

funding_adjusted_execution <- function(exec_schedule, funding_rates,
                                        direction = -1) {
  # For short perp strategy: timing around funding payments
  # Funding paid every 8 hours; schedule execution to minimize funding paid
  n        <- length(exec_schedule)
  fund_adj <- numeric(n)
  for (i in seq_len(n)) {
    fund_adj[i] <- direction * funding_rates[i] * exec_schedule[i]
  }
  list(schedule = exec_schedule,
       funding_cost = fund_adj,
       cum_funding  = cumsum(fund_adj),
       optimal_time = which.min(funding_rates[1:n]))
}

multi_venue_routing <- function(qty, venues, ask_prices, bid_prices,
                                 available_qty, direction = 1) {
  n <- length(venues)
  if (direction == 1) prices <- ask_prices else prices <- bid_prices
  order_venues <- order(prices, decreasing = (direction == -1))
  filled <- 0; route <- numeric(n); cost <- 0
  for (i in order_venues) {
    if (filled >= qty) break
    fill_i      <- min(available_qty[i], qty - filled)
    route[i]    <- fill_i
    cost        <- cost + fill_i * prices[i]
    filled      <- filled + fill_i
  }
  avg_price <- cost / (filled + 1e-8)
  list(route = route, filled = filled, avg_price = avg_price,
       best_single_price = prices[order_venues[1]],
       routing_benefit   = (prices[order_venues[1]] - avg_price) * direction * 1e4)
}

# ============================================================
# ADDITIONAL: ORDER MANAGEMENT
# ============================================================
order_urgency_score <- function(target_qty, filled_qty, time_elapsed,
                                 time_budget, market_vol) {
  pct_filled  <- filled_qty / (target_qty + 1e-8)
  pct_time    <- time_elapsed / (time_budget + 1e-8)
  urgency     <- pct_time - pct_filled
  vol_adj     <- urgency * (1 + market_vol)
  list(base_urgency=urgency, vol_adjusted=vol_adj,
       must_rush=urgency > 0.3,
       pct_filled=pct_filled, pct_time=pct_time)
}

participation_rate_optimizer <- function(total_qty, adv, sigma,
                                          risk_aversion=1e-4,
                                          pov_min=0.02, pov_max=0.25) {
  obj <- function(pov) {
    T_hrs  <- (1/pov) * (total_qty/adv) * 8  # hours
    impact <- 0.1 * sigma * sqrt(pov) * 1e4
    timing_risk <- risk_aversion * sigma^2 * T_hrs^2
    impact + timing_risk
  }
  pov_grid <- seq(pov_min, pov_max, by=0.01)
  costs    <- sapply(pov_grid, obj)
  opt_pov  <- pov_grid[which.min(costs)]
  list(optimal_pov=opt_pov, cost_curve=data.frame(pov=pov_grid, cost=costs),
       min_cost=min(costs))
}

execution_shortfall_budget <- function(total_notional, is_budget_bps,
                                        component_budgets) {
  # Allocate budget to IS components
  total_budget <- sum(component_budgets)
  alloc <- component_budgets / total_budget * is_budget_bps
  budget_usd <- alloc / 1e4 * total_notional
  list(components=names(component_budgets),
       budget_bps=alloc, budget_usd=budget_usd,
       total_budget_usd=is_budget_bps/1e4*total_notional)
}

# ============================================================
# ADDITIONAL: ALGO SELECTION
# ============================================================
algo_selection_model <- function(qty, adv, sigma, urgency,
                                  spread_bps, market_cap_tier=1) {
  pov <- qty / (adv + 1e-8)
  # Score each algo
  vwap_score  <- 1 - urgency + (1-pov)
  twap_score  <- 0.5 * (1-urgency) + 0.5 * (1-pov)
  is_score    <- urgency + pov * sigma
  pov_score   <- 0.7 + 0.3 * (1-pov)

  scores <- c(VWAP=vwap_score, TWAP=twap_score, IS=is_score, POV=pov_score)
  best   <- names(which.max(scores))
  list(scores=scores, recommended=best,
       rationale=paste("POV:", round(pov,3), "Urgency:", round(urgency,2)))
}

dark_pool_routing_decision <- function(qty, spread_bps, dark_fill_prob,
                                        lit_impact_bps, dark_fee_bps=0.5) {
  # Expected cost: dark pool
  ec_dark <- (1-dark_fill_prob)*lit_impact_bps + dark_fill_prob*dark_fee_bps
  # Expected cost: lit market
  ec_lit  <- spread_bps/2 + lit_impact_bps
  list(ec_dark=ec_dark, ec_lit=ec_lit,
       prefer_dark=ec_dark < ec_lit,
       dark_benefit_bps=ec_lit-ec_dark)
}

# ============================================================
# ADDITIONAL: VENUE ANALYTICS
# ============================================================
exchange_quality_score <- function(avg_spread, fill_rate, latency_ms,
                                    uptime_pct, fee_bps) {
  spread_s  <- 1 - avg_spread/max(avg_spread)
  fill_s    <- fill_rate
  lat_s     <- 1 - latency_ms/max(latency_ms)
  uptime_s  <- uptime_pct/100
  fee_s     <- 1 - fee_bps/max(fee_bps)
  composite <- (spread_s + fill_s + lat_s + uptime_s + fee_s) / 5
  list(score=composite, spread=spread_s, fill=fill_s,
       latency=lat_s, uptime=uptime_s, fee=fee_s)
}

smart_order_routing <- function(asks, bid_asks, available_qty, order_qty,
                                 fees, direction=1) {
  n     <- length(asks)
  prices <- if (direction==1) asks else bid_asks
  order_ <- order(prices, decreasing=(direction==-1))
  filled <- 0; routes <- numeric(n); cost <- 0
  for (i in order_) {
    if (filled>=order_qty) break
    fill_i   <- min(available_qty[i], order_qty-filled)
    routes[i] <- fill_i
    cost      <- cost + fill_i * (prices[i] + direction*fees[i]/1e4)
    filled    <- filled + fill_i
  }
  list(routes=routes, filled=filled, avg_price=cost/(filled+1e-8),
       best_price=prices[order_[1]])
}
