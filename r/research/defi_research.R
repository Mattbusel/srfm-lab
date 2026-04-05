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

# ============================================================
# ADDITIONAL: EMPIRICAL DEFI STUDIES
# ============================================================
amm_price_efficiency_test <- function(pool_prices, cex_prices, window=100) {
  n   <- length(pool_prices)
  err <- pool_prices - cex_prices
  acf_vals <- acf(err[!is.na(err)], lag.max=10, plot=FALSE)$acf[,,1]
  var_ratio_5 <- var(diff(err,5))/(5*var(diff(err))+1e-8)
  list(mean_error=mean(err,na.rm=TRUE), sd_error=sd(err,na.rm=TRUE),
       acf_1=acf_vals[2], var_ratio_5=var_ratio_5,
       efficient=abs(acf_vals[2])<0.1 && abs(var_ratio_5-1)<0.1)
}

liquidity_provider_pnl_study <- function(initial_capital, price_series,
                                          fee_rate=0.003, rebalance=FALSE) {
  n    <- length(price_series)
  k    <- initial_capital^2/4  # constant product
  il_  <- numeric(n); fee_cum <- numeric(n)
  for (i in seq_len(n)) {
    r    <- price_series[i]/price_series[1]
    il_[i]  <- 2*sqrt(r)/(1+r)-1
    fee_cum[i] <- fee_rate * i * 0.001 * initial_capital  # rough estimate
  }
  net_pnl <- initial_capital*(1+il_) - initial_capital + fee_cum
  list(il=il_, fee_income=fee_cum, net_pnl=net_pnl,
       better_than_hold=net_pnl>0,
       breakeven_days=which(net_pnl>0)[1])
}

gas_cost_impact_study <- function(strategy_returns, gas_costs_usd,
                                   position_sizes_usd) {
  gas_drag <- gas_costs_usd / (position_sizes_usd + 1e-8)
  net_ret  <- strategy_returns - gas_drag
  list(gross=strategy_returns, gas_drag=gas_drag, net=net_ret,
       gas_adjusted_sr=mean(net_ret)/(sd(net_ret)+1e-8)*sqrt(365),
       min_position_for_viability=gas_costs_usd/(strategy_returns+1e-8))
}

mev_impact_on_lp <- function(lp_returns, mev_extracted, pool_tvl) {
  mev_rate     <- mev_extracted / (pool_tvl+1e-8)
  lp_ret_adj   <- lp_returns - mev_rate
  list(gross_lp=lp_returns, mev_rate=mev_rate, net_lp=lp_ret_adj,
       mev_impact_pct=mev_rate/lp_returns*100)
}

concentration_vs_returns <- function(position_ranges, lp_returns,
                                      price_vol) {
  width <- sapply(position_ranges, function(r) r[2]-r[1])
  efficiency <- 1/width
  df <- data.frame(width=width, efficiency=efficiency, returns=lp_returns, vol=price_vol)
  corr_eff_ret <- cor(df$efficiency, df$returns, use="complete.obs")
  list(df=df, efficiency_return_corr=corr_eff_ret,
       optimal_width=width[which.max(lp_returns)])
}


# ============================================================
# EMPIRICAL DEFI RESEARCH FUNCTIONS
# ============================================================

lp_survivorship_study <- function(pool_birth_dates, pool_death_dates,
                                   tvl_at_birth, current_date) {
  alive       <- is.na(pool_death_dates)
  age         <- ifelse(alive,
                        as.numeric(current_date - pool_birth_dates),
                        as.numeric(pool_death_dates - pool_birth_dates))
  survival_rate <- mean(alive)
  median_life   <- median(age[!alive], na.rm=TRUE)
  correlation   <- cor(log(tvl_at_birth + 1), age, use="pairwise.complete.obs")
  list(survival_rate = survival_rate, median_life_days = median_life,
       age = age, alive = alive,
       tvl_survival_corr = correlation)
}

amm_fee_vs_il_study <- function(fee_revenue, il_cost, tvl_series,
                                  time_periods) {
  net_lp_return <- fee_revenue - il_cost
  fee_apy       <- fee_revenue / (tvl_series + 1e-8) * 365 / time_periods
  il_apy        <- il_cost    / (tvl_series + 1e-8) * 365 / time_periods
  breakeven_tvl <- il_cost / (fee_revenue / (tvl_series + 1e-8) + 1e-12)
  list(net_lp_return = net_lp_return,
       fee_apy = fee_apy, il_apy = il_apy,
       net_apy = fee_apy - il_apy,
       is_profitable = net_lp_return > 0,
       breakeven_tvl = breakeven_tvl)
}

protocol_revenue_attribution <- function(total_protocol_rev,
                                          trading_fees, liquidation_fees,
                                          borrow_fees) {
  fractions <- c(trading = trading_fees, liquidation = liquidation_fees,
                 borrow = borrow_fees) / (total_protocol_rev + 1e-8)
  diversification_hhi <- sum(fractions^2)
  list(fractions = fractions, hhi = diversification_hhi,
       dominant_source = names(which.max(fractions)),
       concentrated = diversification_hhi > 0.5)
}

yield_aggregator_efficiency <- function(gross_yields, gas_costs,
                                         management_fees, compounding_freq) {
  net_apy <- ((1 + (gross_yields - management_fees) / compounding_freq)^compounding_freq - 1) -
               gas_costs
  efficiency_ratio <- net_apy / (gross_yields + 1e-8)
  list(gross_yields = gross_yields, net_apy = net_apy,
       efficiency_ratio = efficiency_ratio,
       gas_drag = gas_costs / (gross_yields + 1e-8))
}

governance_token_value_model <- function(protocol_revenue, token_supply,
                                          buyback_pct, discount_rate = 0.15,
                                          growth_rate = 0.2, horizon = 5) {
  fair_values <- numeric(horizon)
  for (y in seq_len(horizon)) {
    projected_rev    <- protocol_revenue * (1 + growth_rate)^y
    token_cash_flow  <- projected_rev * buyback_pct
    fair_values[y]   <- token_cash_flow / (discount_rate - growth_rate + 1e-8) /
                          (1 + discount_rate)^y
  }
  per_token <- fair_values / (token_supply + 1e-12)
  list(fair_value_scenarios = fair_values,
       per_token_fair_value = per_token,
       npv = sum(fair_values / (1 + discount_rate)^seq_len(horizon)))
}

defi_market_share_dynamics <- function(tvl_matrix, protocol_names) {
  # tvl_matrix: rows = time, cols = protocols
  total_tvl   <- rowSums(tvl_matrix, na.rm=TRUE)
  shares      <- sweep(tvl_matrix, 1, total_tvl + 1e-8, "/")
  hhi         <- apply(shares, 1, function(r) sum(r^2))
  dominance_idx <- apply(shares, 1, function(r) max(r, na.rm=TRUE))
  share_changes <- apply(shares, 2, diff)
  list(market_shares = shares, hhi = hhi,
       dominance_index = dominance_idx,
       share_velocity = share_changes,
       leader = apply(shares, 1, function(r) protocol_names[which.max(r)]))
}

bridge_flow_analysis <- function(inflows, outflows, bridge_names,
                                  window = 7) {
  net_flow    <- inflows - outflows
  net_ma      <- apply(net_flow, 2, function(x)
                   as.numeric(stats::filter(x, rep(1/window, window), sides=1)))
  total_bridged <- colSums(inflows + outflows, na.rm=TRUE)
  list(net_flow = net_flow, smoothed_net = net_ma,
       total_volume = total_bridged,
       dominant_bridge = bridge_names[which.max(total_bridged)],
       flow_concentration = total_bridged / sum(total_bridged))
}

liquidation_cascade_study <- function(collateral_prices, debt_levels,
                                       liq_thresholds, cascade_rounds = 5) {
  health <- collateral_prices / (debt_levels / liq_thresholds + 1e-8)
  liquidated <- health < 1.0
  total_liquidated_debt <- sum(debt_levels[liquidated], na.rm=TRUE)
  for (r in seq_len(cascade_rounds)) {
    price_impact  <- total_liquidated_debt * 0.0001
    collateral_prices <- collateral_prices * (1 - price_impact)
    health     <- collateral_prices / (debt_levels / liq_thresholds + 1e-8)
    new_liq    <- health < 1.0 & !liquidated
    liquidated <- liquidated | new_liq
    total_liquidated_debt <- sum(debt_levels[liquidated], na.rm=TRUE)
    if (!any(new_liq)) break
  }
  list(liquidated = liquidated,
       total_liquidated_debt = total_liquidated_debt,
       cascade_fraction = mean(liquidated),
       final_health = health)
}


# ─── ADDITIONAL DEFI RESEARCH ─────────────────────────────────────────────────

stablecoin_peg_stability_study <- function(price_series, target = 1.0,
                                            window = 30) {
  deviation      <- price_series - target
  abs_dev        <- abs(deviation)
  peg_breaks     <- abs_dev > 0.01
  recovery_time  <- diff(which(peg_breaks))
  vol_dev        <- sd(deviation, na.rm=TRUE)
  roll_vol       <- rep(NA, length(price_series))
  for (i in seq(window, length(price_series))) {
    idx <- seq(i - window + 1, i)
    roll_vol[i] <- sd(deviation[idx], na.rm=TRUE)
  }
  list(deviation = deviation, abs_deviation = abs_dev,
       peg_break_pct = mean(peg_breaks, na.rm=TRUE),
       avg_recovery_time = if (length(recovery_time)) mean(recovery_time) else NA,
       vol_of_deviation = vol_dev, rolling_vol = roll_vol,
       max_deviation = max(abs_dev, na.rm=TRUE))
}

amm_volume_tvl_ratio_study <- function(volume_series, tvl_series,
                                        window = 7) {
  ratio      <- volume_series / (tvl_series + 1e-8)
  ratio_ma   <- as.numeric(stats::filter(ratio, rep(1/window, window), sides=1))
  fee_yield  <- ratio * 0.003 * 365
  list(volume_to_tvl = ratio, smoothed = ratio_ma,
       annualized_fee_yield = fee_yield,
       high_utilization = ratio > quantile(ratio, 0.75, na.rm=TRUE))
}

lending_market_utilization_study <- function(supplied, borrowed,
                                              base_rate = 0.02,
                                              kink = 0.8, jump_rate = 0.5) {
  utilization <- borrowed / (supplied + 1e-8)
  borrow_rate <- ifelse(utilization < kink,
                         base_rate + utilization / kink * 0.1,
                         base_rate + 0.1 + (utilization - kink) / (1 - kink + 1e-8) * jump_rate)
  supply_rate <- borrow_rate * utilization * 0.9
  list(utilization = utilization, borrow_rate = borrow_rate,
       supply_rate = supply_rate,
       at_kink = abs(utilization - kink) < 0.05,
       crisis_zone = utilization > 0.95)
}

protocol_comparison_framework <- function(metrics_mat, weights,
                                           higher_is_better) {
  norm_mat <- matrix(NA, nrow(metrics_mat), ncol(metrics_mat))
  for (j in seq_len(ncol(metrics_mat))) {
    col <- metrics_mat[, j]
    mn  <- min(col, na.rm=TRUE); mx <- max(col, na.rm=TRUE)
    norm_col <- (col - mn) / (mx - mn + 1e-8)
    norm_mat[, j] <- if (higher_is_better[j]) norm_col else 1 - norm_col
  }
  scores <- as.vector(norm_mat %*% (weights / sum(weights)))
  list(normalized = norm_mat, scores = scores,
       ranking = order(-scores),
       best_protocol = which.max(scores))
}

# ─── UTILITY / HELPER FUNCTIONS ───────────────────────────────────────────────

defi_sharpe_equivalent <- function(apy, vol_daily, rf_daily = 0) {
  daily_ret <- apy / 365
  (daily_ret - rf_daily) / (vol_daily + 1e-8) * sqrt(365)
}

pool_concentration_risk <- function(top10_tvl, total_tvl) {
  top10_share <- top10_tvl / (total_tvl + 1e-8)
  list(top10_share = top10_share,
       tail_risk_premium = pmax(top10_share - 0.5, 0) * 0.2,
       is_concentrated = top10_share > 0.7)
}

defi_risk_adjusted_yield <- function(gross_yield, smart_contract_risk_pct,
                                      oracle_risk_pct, liquidity_risk_pct) {
  total_risk <- smart_contract_risk_pct + oracle_risk_pct + liquidity_risk_pct
  net_yield  <- gross_yield - total_risk
  list(gross_yield = gross_yield, total_risk = total_risk,
       net_yield = net_yield, is_positive = net_yield > 0,
       risk_components = c(sc=smart_contract_risk_pct, oracle=oracle_risk_pct,
                           liquidity=liquidity_risk_pct))
}

rolling_protocol_dominance <- function(tvl_matrix, window = 30) {
  n  <- nrow(tvl_matrix)
  hh <- rep(NA, n)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i)
    avg_tvls <- colMeans(tvl_matrix[idx, , drop=FALSE], na.rm=TRUE)
    shares   <- avg_tvls / (sum(avg_tvls) + 1e-12)
    hh[i]    <- sum(shares^2)
  }
  list(rolling_hhi = hh, high_concentration = hh > 0.25)
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

# defi research module loaded
.defi_research_loaded <- TRUE
# placeholder utility
defi_util <- function() invisible(NULL)
# end
