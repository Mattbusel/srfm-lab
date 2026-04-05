## defi_analytics.R
## AMM pricing, impermanent loss, liquidity pool math, yield farming APY
## Pure base R -- no library() calls

# ============================================================
# 1. AMM CONSTANT PRODUCT (x*y = k)
# ============================================================

amm_price <- function(reserve_x, reserve_y) reserve_y / reserve_x

amm_swap_out <- function(reserve_x, reserve_y, amount_in, fee = 0.003) {
  amount_in_adj  <- amount_in * (1 - fee)
  amount_out     <- reserve_y * amount_in_adj / (reserve_x + amount_in_adj)
  new_reserve_x  <- reserve_x + amount_in
  new_reserve_y  <- reserve_y - amount_out
  price_impact   <- amount_out / reserve_y
  effective_price <- amount_out / amount_in
  list(amount_out = amount_out, price_impact = price_impact,
       effective_price = effective_price, fee_paid = amount_in * fee,
       new_reserve_x = new_reserve_x, new_reserve_y = new_reserve_y)
}

amm_price_impact <- function(reserve_x, reserve_y, trade_size) {
  mid_price  <- reserve_y / reserve_x
  swap_res   <- amm_swap_out(reserve_x, reserve_y, trade_size)
  exec_price <- swap_res$effective_price
  (mid_price - exec_price) / mid_price * 100
}

amm_liquidity <- function(reserve_x, reserve_y) sqrt(reserve_x * reserve_y)

# ============================================================
# 2. CONCENTRATED LIQUIDITY (Uniswap V3)
# ============================================================

univ3_liquidity <- function(amount_x, amount_y, current_price,
                             price_low, price_high) {
  sq_p   <- sqrt(current_price); sq_lo <- sqrt(price_low); sq_hi <- sqrt(price_high)
  L_x <- amount_x / (1/sq_p - 1/sq_hi)
  L_y <- amount_y / (sq_p - sq_lo)
  min(L_x, L_y)
}

univ3_amounts <- function(L, current_price, price_low, price_high) {
  sq_p  <- sqrt(current_price); sq_lo <- sqrt(price_low); sq_hi <- sqrt(price_high)
  if (current_price < price_low) {
    amount_x <- L * (1/sq_lo - 1/sq_hi); amount_y <- 0
  } else if (current_price > price_high) {
    amount_x <- 0; amount_y <- L * (sq_hi - sq_lo)
  } else {
    amount_x <- L * (1/sq_p - 1/sq_hi)
    amount_y <- L * (sq_p - sq_lo)
  }
  list(amount_x = amount_x, amount_y = amount_y)
}

univ3_fees_earned <- function(L, fee_rate, volume_in_range, price_range_width) {
  # Approximate fee share proportional to L and price range
  fee_per_unit <- fee_rate * volume_in_range * price_range_width
  fee_per_unit * L
}

# ============================================================
# 3. IMPERMANENT LOSS
# ============================================================

impermanent_loss <- function(price_ratio) {
  # price_ratio = new_price / initial_price
  r   <- price_ratio
  il  <- 2 * sqrt(r) / (1 + r) - 1
  list(il = il, il_pct = il * 100,
       hodl_value_ratio = (1 + r) / 2,
       lp_value_ratio   = sqrt(r))
}

impermanent_loss_series <- function(price_0, price_series) {
  r  <- price_series / price_0
  il <- 2*sqrt(r)/(1+r) - 1
  list(price_ratio = r, il = il, il_pct = il * 100)
}

il_breakeven_fee <- function(price_ratio, holding_period_days) {
  il  <- abs(impermanent_loss(price_ratio)$il)
  daily_fee_needed <- -log(1 - il) / holding_period_days
  list(il = il, daily_fee_apy = daily_fee_needed * 365,
       required_daily_fee_pct = daily_fee_needed * 100)
}

# ============================================================
# 4. YIELD FARMING APY
# ============================================================

yield_farm_apy <- function(reward_rate_per_day, token_price,
                            total_staked_usd) {
  daily_yield <- reward_rate_per_day * token_price / (total_staked_usd + 1e-8)
  apy         <- (1 + daily_yield)^365 - 1
  apr         <- daily_yield * 365
  list(daily_yield = daily_yield, apr = apr, apy = apy)
}

compound_apy <- function(apr, compounds_per_year = 365) {
  (1 + apr / compounds_per_year)^compounds_per_year - 1
}

lp_total_return <- function(initial_value, il_pct, fee_apy, hold_days) {
  fee_return <- (1 + fee_apy)^(hold_days/365) - 1
  lp_return  <- (1 + fee_return) * (1 + il_pct/100) - 1
  hodl_return <- 0
  list(fee_return = fee_return, il_loss = il_pct/100,
       net_lp_return = lp_return,
       vs_hodl = lp_return - hodl_return)
}

# ============================================================
# 5. POOL ANALYTICS
# ============================================================

pool_tvl_series <- function(reserve_x, reserve_y, price_y_in_x) {
  reserve_x + reserve_y * price_y_in_x
}

pool_volume_to_tvl <- function(daily_volume, tvl) {
  daily_volume / (tvl + 1e-8)
}

pool_fee_apr <- function(daily_volume, tvl, fee_rate = 0.003) {
  daily_fee_yield <- daily_volume * fee_rate / (tvl + 1e-8)
  daily_fee_yield * 365
}

arb_opportunity <- function(pool_price, external_price, fee = 0.003) {
  threshold <- external_price * (1 + fee)
  arb_profit_pct <- (external_price - pool_price) / pool_price - fee
  list(arb_exists = abs(pool_price - external_price)/external_price > fee,
       arb_profit_pct = arb_profit_pct,
       direction = if (external_price > pool_price) "buy_pool" else "sell_pool")
}

# ============================================================
# 6. CURVE (STABLESWAP)
# ============================================================

curve_invariant <- function(x_vec, A) {
  n  <- length(x_vec)
  S  <- sum(x_vec)
  P  <- prod(x_vec)
  D  <- S  # initial guess
  for (iter in seq_len(255)) {
    D_P <- D
    for (xi in x_vec) D_P <- D_P * D / (n * xi + 1e-12)
    D_prev <- D
    D <- (A*n^n*S + n*D_P) * D / ((A*n^n - 1)*D + (n+1)*D_P + 1e-12)
    if (abs(D - D_prev) < 1e-9) break
  }
  D
}

curve_swap <- function(x_from, x_vec, A, i, j, dx) {
  x_new_i <- x_vec[i] + dx
  D       <- curve_invariant(x_vec, A)
  n       <- length(x_vec)
  # Solve for new x_j
  Ann  <- A * n^n
  c    <- D^(n+1) / (Ann * prod(x_vec[-c(i,j)]) * n^n + 1e-12)
  b_   <- sum(x_vec[-c(i,j)]) + x_new_i - D + D/Ann
  disc <- sqrt(b_^2 + 4*c + 1e-12)
  x_new_j <- (-b_ + disc) / 2
  dy <- x_vec[j] - x_new_j
  list(dy = dy, price_impact = dy/x_vec[j])
}

# ============================================================
# 7. LENDING PROTOCOL ANALYTICS
# ============================================================

utilization_rate <- function(total_borrows, total_supply) {
  total_borrows / (total_supply + 1e-8)
}

borrow_rate_kinked <- function(utilization, base_rate = 0.02,
                                kink = 0.80, slope1 = 0.1, slope2 = 1.0) {
  if (utilization <= kink)
    base_rate + utilization * slope1 / kink
  else
    base_rate + slope1 + (utilization - kink) * slope2 / (1 - kink)
}

supply_rate <- function(borrow_rate, utilization, reserve_factor = 0.1) {
  borrow_rate * utilization * (1 - reserve_factor)
}

health_factor <- function(collateral_value, collateral_factor,
                           debt_value) {
  (collateral_value * collateral_factor) / (debt_value + 1e-12)
}

liquidation_threshold <- function(health_factor_val, debt_value,
                                   collateral_factor = 0.75) {
  required_collateral <- debt_value / collateral_factor
  collateral_at_risk  <- required_collateral * (1 - health_factor_val)
  list(health_factor = health_factor_val,
       shortfall = pmax(1 - health_factor_val, 0) * debt_value,
       at_risk   = collateral_at_risk)
}

# ============================================================
# 8. DEFI RISK METRICS
# ============================================================

protocol_tvl_concentration <- function(pool_tvls) {
  total <- sum(pool_tvls)
  shares <- pool_tvls / total
  hhi  <- sum(shares^2)
  list(shares = shares, hhi = hhi,
       top3_pct = sum(sort(shares, decreasing = TRUE)[1:min(3, length(shares))]) * 100)
}

smart_contract_risk_premium <- function(tvl, audit_score,
                                         age_months, incident_history = 0) {
  # Higher TVL = more target, higher audit = less risk
  base   <- 0.02
  tvl_f  <- log(tvl / 1e6 + 1) * 0.005
  audit_f <- (1 - audit_score) * 0.05
  age_f   <- exp(-age_months / 24) * 0.03
  inc_f   <- incident_history * 0.02
  pmax(base + tvl_f + audit_f + age_f + inc_f, 0)
}

lp_position_value <- function(L, current_price, price_low, price_high,
                               fee_growth) {
  ams <- univ3_amounts(L, current_price, price_low, price_high)
  val <- ams$amount_x * current_price + ams$amount_y + fee_growth
  list(value = val, amount_x = ams$amount_x, amount_y = ams$amount_y,
       fee_component = fee_growth)
}

# ============================================================
# ADDITIONAL: PROTOCOL COMPARISON
# ============================================================
compare_amm_protocols <- function(tvl_list, volume_list, fee_rates,
                                   protocol_names) {
  n <- length(protocol_names)
  capital_eff <- mapply(function(v, tvl) v/(tvl+1e-8), volume_list, tvl_list)
  fee_apy     <- mapply(function(v, tvl, f) v*f*365/(tvl+1e-8), volume_list, tvl_list, fee_rates)
  data.frame(protocol=protocol_names, tvl=unlist(tvl_list),
             volume=unlist(volume_list), capital_efficiency=capital_eff,
             fee_apy=fee_apy)
}

liquidity_depth_score <- function(pool_reserves, trade_size_usd, fee_rate=0.003) {
  k     <- prod(pool_reserves)
  mid_p <- pool_reserves[2]/pool_reserves[1]
  impact_usd <- trade_size_usd
  impact_pct <- impact_usd/(sum(pool_reserves*c(mid_p,1))+1e-8)
  list(depth_score=1/impact_pct, mid_price=mid_p,
       impact_bps=impact_pct*1e4, fee_bps=fee_rate*1e4)
}

yield_aggregator_analysis <- function(strategies, apys, tvls, risks) {
  weighted_apy <- sum(apys*tvls)/sum(tvls)
  risk_adj_apy <- apys / (risks+1e-8)
  best_risk_adj <- which.max(risk_adj_apy)
  list(weighted_avg_apy=weighted_apy,
       risk_adj_apy=risk_adj_apy,
       best_strategy=strategies[best_risk_adj],
       allocation=tvls/sum(tvls))
}

# ============================================================
# ADDITIONAL: GOVERNANCE ANALYTICS
# ============================================================
governance_participation <- function(token_holders, votes_cast,
                                      proposal_outcomes) {
  turnout <- votes_cast / (token_holders+1e-8)
  pass_rate <- mean(proposal_outcomes == "passed")
  list(turnout=turnout, pass_rate=pass_rate,
       mean_turnout=mean(turnout,na.rm=TRUE),
       governance_score=mean(turnout)*pass_rate)
}

whale_voting_power <- function(holder_balances, total_supply) {
  pct   <- holder_balances / total_supply
  top10 <- sum(sort(pct,decreasing=TRUE)[1:min(10,length(pct))])
  hhi   <- sum(pct^2)
  list(pct=pct, top10_pct=top10*100, hhi=hhi,
       decentralized=top10<0.5)
}

# ============================================================
# ADDITIONAL: CROSS-CHAIN ANALYTICS
# ============================================================
bridge_volume_signal <- function(inflow_by_chain, outflow_by_chain,
                                  chain_names) {
  net_flow  <- inflow_by_chain - outflow_by_chain
  total_flow <- inflow_by_chain + outflow_by_chain
  preference <- net_flow / (total_flow+1e-8)
  list(net=net_flow, preference=preference,
       fastest_growing=chain_names[which.max(net_flow)],
       df=data.frame(chain=chain_names, net=net_flow, pref=preference))
}

cross_chain_arb <- function(token_prices, chain_names, bridge_costs) {
  n     <- length(chain_names)
  spread_mat <- outer(token_prices, token_prices, "-")
  arb_mat    <- abs(spread_mat) - matrix(bridge_costs, n, n, byrow=TRUE)
  arb_opp    <- which(arb_mat > 0, arr.ind=TRUE)
  list(spread_matrix=spread_mat, net_arb=arb_mat,
       opportunities=arb_opp, n_opportunities=nrow(arb_opp))
}

# ============================================================
# ADDITIONAL: DEFI RISK SCORING
# ============================================================
protocol_health_score <- function(tvl, revenue_30d, token_price,
                                   token_fdv, audit_score=0.8) {
  ps_ratio <- token_fdv / (tvl+1e-8)
  pe_ratio <- token_fdv / (revenue_30d*12+1e-8)
  rev_yield <- revenue_30d*12 / (token_fdv+1e-8)
  health <- audit_score * (1/pmax(ps_ratio,1)) * pmin(rev_yield*10, 1)
  list(ps=ps_ratio, pe=pe_ratio, rev_yield=rev_yield,
       health_score=pmin(health,1), grade=ifelse(health>.7,"A",
                                           ifelse(health>.4,"B","C")))
}


# ============================================================
# ADDITIONAL DEFI ANALYTICS
# ============================================================

amm_fee_revenue <- function(volume_series, fee_tier = 0.003) {
  daily_rev <- volume_series * fee_tier
  cumrev    <- cumsum(daily_rev)
  list(daily_revenue = daily_rev, cumulative = cumrev,
       annualized = mean(daily_rev, na.rm=TRUE) * 365)
}

concentrated_lp_range_pnl <- function(price, lower, upper,
                                       liquidity, fee_tier = 0.003,
                                       volume_series = NULL) {
  in_range <- price >= lower & price <= upper
  price_ratio <- pmin(price, upper) / pmax(price, lower)
  il_factor   <- 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
  if (!is.null(volume_series)) {
    fee_income <- ifelse(in_range, volume_series * fee_tier * liquidity, 0)
  } else {
    fee_income <- rep(0, length(price))
  }
  list(in_range = in_range, il_factor = il_factor,
       fee_income = fee_income,
       net_pnl = fee_income + il_factor * liquidity,
       range_utilization = mean(in_range, na.rm=TRUE))
}

rebase_token_mechanics <- function(supply, target_price, current_price,
                                    rebase_fraction = 0.1) {
  deviation   <- (current_price - target_price) / (target_price + 1e-8)
  rebase_amt  <- supply * deviation * rebase_fraction
  new_supply  <- supply + rebase_amt
  list(rebase_amount = rebase_amt, new_supply = new_supply,
       deviation_pct = deviation * 100,
       expansionary = rebase_amt > 0)
}

governance_quorum_analysis <- function(votes_for, votes_against,
                                        total_supply, quorum_pct = 0.04) {
  total_votes     <- votes_for + votes_against
  participation   <- total_votes / (total_supply + 1e-12)
  approval_rate   <- votes_for / (total_votes + 1e-12)
  quorum_reached  <- participation >= quorum_pct
  list(participation = participation, approval_rate = approval_rate,
       quorum_reached = quorum_reached,
       passed = quorum_reached & approval_rate > 0.5,
       effective_turnout = participation * approval_rate)
}

protocol_fee_switch <- function(total_fees, lp_share = 0.8, protocol_share = 0.2,
                                 token_buyback_fraction = 0.5) {
  lp_fees       <- total_fees * lp_share
  protocol_fees <- total_fees * protocol_share
  buyback       <- protocol_fees * token_buyback_fraction
  treasury      <- protocol_fees * (1 - token_buyback_fraction)
  list(lp_fees = lp_fees, protocol_fees = protocol_fees,
       buyback = buyback, treasury = treasury,
       revenue_multiple = cumsum(total_fees))
}

liquidity_migration_model <- function(apy_a, apy_b, tvl_a, total_liquidity,
                                       elasticity = 2.0) {
  apy_diff       <- apy_b - apy_a
  migration_pct  <- plogis(elasticity * apy_diff) - 0.5
  tvl_shift      <- migration_pct * tvl_a
  new_tvl_a      <- tvl_a - pmax(tvl_shift, 0)
  new_tvl_b      <- total_liquidity - new_tvl_a
  list(migration_pct = migration_pct, tvl_shift = tvl_shift,
       new_tvl_a = new_tvl_a, new_tvl_b = new_tvl_b)
}

defi_yield_decomposition <- function(base_apy, reward_apy, il_estimate,
                                      gas_cost_annual, risk_premium) {
  gross_yield  <- base_apy + reward_apy
  net_yield    <- gross_yield + il_estimate - gas_cost_annual
  risk_adj     <- net_yield - risk_premium
  list(gross_yield = gross_yield, net_yield = net_yield,
       risk_adjusted = risk_adj,
       il_drag = il_estimate,
       is_attractive = risk_adj > 0.05)
}

collateral_efficiency <- function(debt, collateral_value, liq_threshold = 0.825) {
  collateral_ratio <- collateral_value / (debt + 1e-8)
  utilization      <- debt / (collateral_value * liq_threshold + 1e-8)
  buffer           <- collateral_ratio - 1 / liq_threshold
  list(collateral_ratio = collateral_ratio,
       utilization = utilization,
       safety_buffer = buffer,
       at_risk = collateral_ratio < 1 / liq_threshold * 1.1)
}

amm_slippage_curve <- function(reserve_x, reserve_y, trade_sizes) {
  k       <- reserve_x * reserve_y
  out_amt <- sapply(trade_sizes, function(dx) {
    new_x <- reserve_x + dx
    new_y <- k / new_x
    reserve_y - new_y
  })
  spot_price <- reserve_y / reserve_x
  exec_price <- out_amt / trade_sizes
  slippage   <- (spot_price - exec_price) / spot_price
  list(trade_sizes = trade_sizes, output = out_amt,
       execution_price = exec_price, slippage_pct = slippage * 100)
}

token_emission_schedule <- function(initial_supply, emission_rate,
                                     vesting_schedule, n_periods = 48) {
  circulating <- numeric(n_periods)
  circulating[1] <- initial_supply
  for (t in 2:n_periods) {
    vest_t <- if (t <= length(vesting_schedule)) vesting_schedule[t] else 0
    circulating[t] <- circulating[t-1] + emission_rate + vest_t
  }
  inflation_rate <- c(NA, diff(circulating) / (circulating[-length(circulating)] + 1e-8))
  list(circulating = circulating, inflation_rate = inflation_rate,
       fully_diluted_pct = circulating / max(circulating))
}


# ─── ADDITIONAL: RISK METRICS ────────────────────────────────────────────────

defi_portfolio_var <- function(pool_returns_mat, weights, alpha = 0.05) {
  port_ret  <- as.vector(pool_returns_mat %*% weights)
  var_est   <- quantile(port_ret, alpha, na.rm=TRUE)
  es_est    <- mean(port_ret[port_ret <= var_est], na.rm=TRUE)
  list(var = var_est, es = es_est,
       max_dd = min(cumprod(1 + port_ret) / cummax(cumprod(1 + port_ret)) - 1))
}

defi_stress_test <- function(tvl, yield_apy, price_shock = -0.5,
                               liquidity_shock = 0.3, yield_shock = -0.5) {
  shocked_tvl   <- tvl * (1 + price_shock)
  shocked_liq   <- shocked_tvl * (1 - liquidity_shock)
  shocked_yield <- yield_apy * (1 + yield_shock)
  pnl_tvl       <- shocked_tvl - tvl
  list(base_tvl = tvl, shocked_tvl = shocked_tvl,
       available_liquidity = shocked_liq,
       shocked_yield_apy = shocked_yield,
       tvl_loss = pnl_tvl,
       pct_loss = pnl_tvl / (tvl + 1e-8))
}

stable_pool_curve_analysis <- function(A_param, balances, fee = 0.0004) {
  n <- length(balances)
  D_approx <- sum(balances)
  for (iter in 1:100) {
    Dprod <- D_approx^(n+1) / (n^n * prod(balances))
    D_approx <- (A_param * n^n * sum(balances) + Dprod) /
                  ((A_param * n^n - 1) + (n+1) * Dprod / D_approx + 1e-12) *
                  D_approx
    if (is.nan(D_approx) || !is.finite(D_approx)) break
  }
  price_impact_1pct <- fee + 1 / (A_param * n + 1e-8) * 0.01
  list(D = D_approx, A = A_param,
       fee = fee, estimated_price_impact_1pct = price_impact_1pct)
}

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

defi_apr_to_apy <- function(apr, compounding_periods = 365) {
  (1 + apr / compounding_periods)^compounding_periods - 1
}

defi_apy_to_apr <- function(apy, compounding_periods = 365) {
  compounding_periods * ((1 + apy)^(1/compounding_periods) - 1)
}

optimal_rebalance_band <- function(target_weight, vol, tc_bps, horizon = 21) {
  sigma_drift <- vol * sqrt(horizon / 252)
  band_half   <- sqrt(2 * tc_bps/1e4 * sigma_drift)
  list(lower = target_weight - band_half, upper = target_weight + band_half,
       band_width = 2 * band_half)
}
