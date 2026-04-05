## defi_research.R
## DeFi protocol research: AMM efficiency, liquidity dynamics, protocol analytics
## Pure base R -- no library() calls

# ============================================================
# 1. AMM EFFICIENCY RESEARCH
# ============================================================

amm_price_discovery <- function(pool_prices, cex_prices, timestamps) {
  spread      <- pool_prices - cex_prices
  spread_pct  <- spread / cex_prices * 100
  # Gonzalo-Granger price discovery
  n   <- length(pool_prices)
  y1  <- diff(log(pool_prices)); y2 <- diff(log(cex_prices))
  # Error correction measure
  ecm <- spread[-1] / sd(spread, na.rm=TRUE)
  list(spread=spread, spread_pct=spread_pct,
       mean_spread_bps=mean(abs(spread_pct))*100,
       lead_lag=cor(y1, c(NA,y2[-length(y2)]), use="complete.obs"),
       price_discovery_share=0.5)  # simplified
}

amm_lp_efficiency <- function(fee_revenue, impermanent_loss,
                               capital_deployed, horizon_days) {
  gross_yield  <- fee_revenue / capital_deployed
  net_yield    <- gross_yield + impermanent_loss / capital_deployed
  apr_gross    <- gross_yield * 365 / horizon_days
  apr_net      <- net_yield  * 365 / horizon_days
  list(gross_apr=apr_gross, net_apr=apr_net,
       il_drag=impermanent_loss/capital_deployed,
       capital_efficiency=gross_yield/(abs(impermanent_loss)/capital_deployed+1e-8))
}

pool_depth_analysis <- function(reserve_x, reserve_y, trade_sizes) {
  mid_price <- reserve_y / reserve_x
  impacts <- sapply(trade_sizes, function(sz) {
    out  <- reserve_y * sz / (reserve_x + sz) * 0.997
    exec <- out / sz
    (mid_price - exec) / mid_price * 100
  })
  list(trade_sizes=trade_sizes, price_impacts=impacts,
       depth_score=1/mean(impacts))
}

concentrated_lp_analysis <- function(price_series, price_low, price_high,
                                      capital, fee_rate=0.003) {
  n        <- length(price_series)
  in_range <- price_series >= price_low & price_series <= price_high
  # Capital efficiency vs full range
  sq_hi <- sqrt(price_high); sq_lo <- sqrt(price_low)
  full_range_factor <- 1
  conc_factor <- 1 / (1 - sq_lo/sq_hi + 1e-12)
  pct_in_range <- mean(in_range)
  list(pct_in_range=pct_in_range, capital_efficiency=conc_factor,
       effective_capital=capital*conc_factor*pct_in_range,
       out_of_range_days=sum(!in_range))
}

# ============================================================
# 2. LIQUIDITY DYNAMICS
# ============================================================

tvl_momentum <- function(tvl_series, fast=7, slow=30) {
  n    <- length(tvl_series)
  fast_ma <- as.numeric(stats::filter(tvl_series, rep(1/fast,fast), sides=1))
  slow_ma <- as.numeric(stats::filter(tvl_series, rep(1/slow,slow), sides=1))
  list(fast=fast_ma, slow=slow_ma, signal=sign(fast_ma-slow_ma),
       growth_rate=c(NA,diff(log(tvl_series))))
}

liquidity_concentration <- function(pool_tvls) {
  tot <- sum(pool_tvls)
  sh  <- pool_tvls/tot
  list(shares=sh, hhi=sum(sh^2),
       top3=sum(sort(sh,decreasing=TRUE)[1:min(3,length(sh))])*100,
       gini=1-2*sum((rank(sh)/length(sh)-sh/sum(sh+1e-12))))
}

yield_farming_dynamics <- function(reward_per_day, token_prices,
                                    total_staked, holding_days=30) {
  n        <- length(token_prices)
  daily_yr <- reward_per_day * token_prices / (total_staked+1e-8)
  apy      <- (1+daily_yr)^365-1
  # APY erosion as more capital enters
  list(daily_yield=daily_yr, apy=apy,
       sell_pressure=reward_per_day*token_prices,
       yield_dilution=c(NA,diff(apy)))
}

# ============================================================
# 3. PROTOCOL RISK RESEARCH
# ============================================================

protocol_risk_score <- function(tvl, audit_count, age_months,
                                 incident_count=0, oracle_type="chainlink") {
  tvl_risk    <- pmin(log(tvl/1e6+1)*0.1, 1)
  audit_risk  <- pmax(1-audit_count*0.3, 0)
  age_risk    <- exp(-age_months/12)*0.5
  inc_risk    <- incident_count*0.2
  oracle_risk <- if(oracle_type=="chainlink") 0.1 else 0.3
  total <- tvl_risk+audit_risk+age_risk+inc_risk+oracle_risk
  list(total=pmin(total,1), tvl=tvl_risk, audit=audit_risk,
       age=age_risk, incident=inc_risk, oracle=oracle_risk)
}

depeg_risk <- function(peg_price, actual_price, peg_value=1,
                        reserve_ratio=1, window=30) {
  deviation   <- (actual_price-peg_value)/peg_value
  n           <- length(deviation)
  roll_sd     <- rep(NA_real_,n)
  for (i in seq(window,n))
    roll_sd[i] <- sd(deviation[seq(i-window+1,i)])
  list(deviation=deviation, vol=roll_sd,
       at_risk=abs(deviation)>0.02,
       severity=abs(deviation)*reserve_ratio)
}

# ============================================================
# 4. CROSS-PROTOCOL ANALYSIS
# ============================================================

cross_protocol_correlation <- function(tvl_matrix) {
  cor_mat <- cor(tvl_matrix, use="pairwise.complete.obs")
  list(correlation=cor_mat,
       avg_corr=mean(cor_mat[upper.tri(cor_mat)]),
       max_corr=max(cor_mat[upper.tri(cor_mat)]))
}

defi_contagion_test <- function(tvl_series_list, shock_period_start) {
  n    <- length(tvl_series_list)
  results <- lapply(seq_len(n), function(i) {
    s1 <- seq_len(shock_period_start-1)
    s2 <- seq(shock_period_start, length(tvl_series_list[[i]]))
    list(pre_vol  = sd(diff(log(tvl_series_list[[i]][s1]))),
         post_vol = sd(diff(log(tvl_series_list[[i]][s2]))),
         vol_ratio= sd(diff(log(tvl_series_list[[i]][s2])))/
                    (sd(diff(log(tvl_series_list[[i]][s1])))+1e-8))
  })
  list(by_protocol=results,
       avg_vol_ratio=mean(sapply(results, function(r) r$vol_ratio)))
}

# ============================================================
# 5. AMM THEORETICAL BENCHMARKS
# ============================================================

cpmm_optimal_arbitrage <- function(pool_x, pool_y, external_price,
                                    fee=0.003) {
  # Optimal arb trade size for x->y swap
  k    <- pool_x * pool_y
  P_ext <- external_price
  dx   <- sqrt(k * P_ext * (1-fee)) - pool_x
  if (dx <= 0) dx <- 0
  dy   <- pool_y - k / (pool_x + dx * (1-fee))
  list(optimal_dx=dx, expected_dy=dy,
       profit=dy - dx*P_ext)
}

lp_value_function <- function(initial_x, initial_y,
                               price_series, fee_rate=0.003) {
  n        <- length(price_series)
  k        <- initial_x * initial_y
  lp_vals  <- numeric(n)
  hold_vals <- numeric(n)
  initial_price <- initial_y / initial_x
  for (i in seq_len(n)) {
    p          <- price_series[i]
    x_new      <- sqrt(k / p)
    y_new      <- sqrt(k * p)
    lp_vals[i] <- x_new * p + y_new
    hold_vals[i] <- initial_x * p + initial_y
  }
  lp_vals_norm   <- lp_vals / lp_vals[1]
  hold_vals_norm <- hold_vals / hold_vals[1]
  list(lp_value=lp_vals, hold_value=hold_vals,
       il=lp_vals_norm - hold_vals_norm,
       cumulative_il=(lp_vals-hold_vals)/hold_vals*100)
}

# ============================================================
# 6. YIELD OPTIMIZATION RESEARCH
# ============================================================

optimal_rebalance_threshold <- function(il_series, fee_per_rebalance,
                                         daily_fee_rate) {
  n_candidates <- seq(0.01, 0.20, by=0.01)
  returns <- sapply(n_candidates, function(thr) {
    rebal_events <- sum(abs(il_series) > thr, na.rm=TRUE)
    rebal_cost   <- rebal_events * fee_per_rebalance
    fee_income   <- length(il_series) * daily_fee_rate
    fee_income - rebal_cost - sum(pmax(abs(il_series)-thr, 0))
  })
  list(thresholds=n_candidates, net_returns=returns,
       optimal_threshold=n_candidates[which.max(returns)])
}

range_selection_analysis <- function(price_series, price_lo_candidates,
                                      price_hi_candidates, fee_rate=0.003) {
  results <- matrix(NA, length(price_lo_candidates), length(price_hi_candidates))
  for (i in seq_along(price_lo_candidates)) {
    for (j in seq_along(price_hi_candidates)) {
      lo <- price_lo_candidates[i]; hi <- price_hi_candidates[j]
      if (lo >= hi) next
      in_rng  <- mean(price_series >= lo & price_series <= hi)
      sq_range <- sqrt(hi) - sqrt(lo)
      efficiency <- 1 / (1 - sqrt(lo)/sqrt(hi) + 1e-12)
      results[i,j] <- in_rng * efficiency * fee_rate * 365
    }
  }
  best_idx <- which(results == max(results, na.rm=TRUE), arr.ind=TRUE)
  list(apr_matrix=results,
       best_lo=price_lo_candidates[best_idx[1]],
       best_hi=price_hi_candidates[best_idx[2]])
}
