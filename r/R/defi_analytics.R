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
