# =============================================================================
# defi_analytics.R
# DeFi Analytics: AMM mechanics, impermanent loss, concentrated liquidity,
# yield farming, liquidation cascades, MEV, and protocol revenue modeling.
# Pure base R. All formulas derived from first principles.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. AMM CONSTANT PRODUCT (x * y = k)
# ---------------------------------------------------------------------------

#' Compute AMM spot price given reserves
#' Price of token X in terms of token Y: P = y / x
amm_spot_price <- function(reserve_x, reserve_y) {
  stopifnot(reserve_x > 0, reserve_y > 0)
  reserve_y / reserve_x
}

#' Compute output amount for a given input (Uniswap v2 formula with fee)
#' dx input -> dy output, fee = 0.003 (0.3%)
#' dy = (dx * (1 - fee) * y) / (x + dx * (1 - fee))
amm_output <- function(dx, reserve_x, reserve_y, fee = 0.003) {
  dx_net <- dx * (1 - fee)
  dy <- (dx_net * reserve_y) / (reserve_x + dx_net)
  dy
}

#' Price impact of a trade (fraction of spot price moved)
#' Positive = buying X pushes price up
amm_price_impact <- function(dx, reserve_x, reserve_y, fee = 0.003) {
  price_before <- reserve_y / reserve_x
  dy <- amm_output(dx, reserve_x, reserve_y, fee)
  reserve_x_new <- reserve_x + dx
  reserve_y_new <- reserve_y - dy
  price_after <- reserve_y_new / reserve_x_new
  # Price impact as percentage move
  (price_after - price_before) / price_before
}

#' Slippage: difference between expected price and execution price
#' Expected price = spot; Execution price = dy / dx (effective)
amm_slippage <- function(dx, reserve_x, reserve_y, fee = 0.003) {
  spot <- reserve_y / reserve_x
  dy <- amm_output(dx, reserve_x, reserve_y, fee)
  exec_price <- dy / dx
  # Slippage = (spot - exec_price) / spot (negative = worse fill)
  (exec_price - spot) / spot
}

#' Compute how much X to buy to move price to target_price
#' Solve: (y - dy) / (x + dx) = target_price  with x*y = k and fee
amm_trade_to_price <- function(target_price, reserve_x, reserve_y, fee = 0.003) {
  # Without fee: new_x = sqrt(k / target_price), dx = new_x - x
  k <- reserve_x * reserve_y
  new_x <- sqrt(k / target_price)
  dx_gross <- new_x - reserve_x
  if (dx_gross <= 0) return(list(dx = 0, dy = 0, direction = "sell"))
  # Gross dx needed accounting for fee
  dx <- dx_gross / (1 - fee)
  dy <- amm_output(dx, reserve_x, reserve_y, fee)
  list(dx = dx, dy = dy, direction = "buy_x")
}

# Demo: 1M/1M liquidity pool, buy 10K worth
demo_amm <- function() {
  rx <- 1e6; ry <- 1e6
  cat("=== AMM Demo: 1M/1M pool ===\n")
  cat(sprintf("Spot price: %.6f\n", amm_spot_price(rx, ry)))
  for (dx in c(1000, 10000, 50000, 100000)) {
    imp <- amm_price_impact(dx, rx, ry)
    slp <- amm_slippage(dx, rx, ry)
    cat(sprintf("  dx=%7.0f  impact=%6.3f%%  slippage=%6.3f%%\n",
                dx, imp * 100, slp * 100))
  }
}

# ---------------------------------------------------------------------------
# 2. IMPERMANENT LOSS
# ---------------------------------------------------------------------------

#' Impermanent loss formula (exact)
#' IL = 2*sqrt(price_ratio) / (1 + price_ratio) - 1
#' where price_ratio = P_t / P_0 (price of X in terms of Y)
#' IL is always <= 0 (loss relative to holding)
impermanent_loss <- function(price_ratio) {
  # price_ratio = P_t / P_0
  2 * sqrt(price_ratio) / (1 + price_ratio) - 1
}

#' IL for a vector of price changes (returns percentage IL)
il_table <- function(pct_changes = c(-0.9, -0.75, -0.5, -0.25, 0, 0.25, 0.5,
                                     1, 2, 4, 9)) {
  ratios <- 1 + pct_changes
  il <- impermanent_loss(ratios)
  data.frame(
    price_change_pct = pct_changes * 100,
    price_ratio      = ratios,
    IL_pct           = il * 100
  )
}

#' Simulate IL over a price path (GBM)
#' Returns time series of LP value vs HODL value
simulate_il_path <- function(n = 252, mu = 0, sigma = 0.04, seed = 42,
                              initial_x = 1, initial_y = 1000,
                              fee_rate = 0.003, volume_as_fraction_of_tvl = 0.1) {
  set.seed(seed)
  # GBM daily log returns
  log_rets <- rnorm(n, mu - 0.5 * sigma^2, sigma)
  price_path <- initial_y / initial_x * exp(cumsum(c(0, log_rets)))  # P_t = y/x

  # Initial LP value (equal weights) = 2 * initial_y (y = P*x)
  initial_lp_value <- 2 * initial_y

  # HODL: keep initial_x tokens X and initial_y tokens Y
  hodl_value <- initial_x * price_path + initial_y

  # LP value at each t: V_LP = 2 * sqrt(k * P_t) where k = x0 * y0
  k <- initial_x * initial_y
  lp_value_no_fees <- 2 * sqrt(k * price_path)

  # Accumulated fees: daily volume * fee_rate applied to LP share
  # Assume LP captures fee_rate * volume, volume = fraction of TVL
  daily_fees <- fee_rate * volume_as_fraction_of_tvl * lp_value_no_fees
  cumulative_fees <- cumsum(c(0, daily_fees[-length(daily_fees)]))
  lp_value_with_fees <- lp_value_no_fees + cumulative_fees

  # IL relative to HODL
  il <- lp_value_no_fees / hodl_value - 1

  data.frame(
    t = 0:n,
    price = price_path,
    hodl_value = hodl_value,
    lp_value_no_fees = lp_value_no_fees,
    lp_value_with_fees = lp_value_with_fees,
    il_pct = il * 100,
    net_pnl_vs_hodl_pct = (lp_value_with_fees / hodl_value - 1) * 100
  )
}

# ---------------------------------------------------------------------------
# 3. FEE INCOME VS IL BREAKEVEN
# ---------------------------------------------------------------------------

#' At what daily volume (as fraction of TVL) does LP break even vs HODL?
#' IL is fixed by price move; fees accumulate proportionally to volume.
#' IL_pct = fee_rate * volume_fraction * days  => vol_fraction = |IL| / (fee_rate * days)
il_breakeven_volume <- function(price_ratio, fee_rate = 0.003, days = 30) {
  il <- abs(impermanent_loss(price_ratio))
  # Required daily volume as fraction of pool TVL
  required_vol_fraction <- il / (fee_rate * days)
  list(
    il_pct = il * 100,
    required_daily_volume_pct_of_tvl = required_vol_fraction * 100,
    realistic = required_vol_fraction < 1  # volume < 100% of TVL per day = realistic
  )
}

#' Breakeven volatility: at what sigma does LP exactly break even?
#' For lognormal price: E[IL] ≈ -sigma^2 * T / 8  (first-order approx)
#' Fee income ≈ fee_rate * vol_frac * T
#' Breakeven: sigma^2 ≈ 8 * fee_rate * vol_frac
il_breakeven_volatility <- function(fee_rate = 0.003,
                                    daily_vol_frac = 0.1) {
  # Annualized volatility at which IL expected = fee income
  sigma_annual <- sqrt(8 * fee_rate * daily_vol_frac * 365)
  sigma_daily  <- sqrt(8 * fee_rate * daily_vol_frac)
  list(
    annual_vol_breakeven = sigma_annual,
    daily_vol_breakeven  = sigma_daily,
    annual_vol_pct       = sigma_annual * 100
  )
}

#' Full LP profitability surface: sigma x volume grid
lp_profitability_surface <- function(sigmas = seq(0.01, 0.10, 0.01),
                                     vol_fracs = seq(0.02, 0.30, 0.02),
                                     fee_rate = 0.003, days = 30) {
  results <- expand.grid(sigma = sigmas, vol_frac = vol_fracs)
  # Expected IL over T days (lognormal approx)
  results$expected_il <- -(results$sigma^2 * days) / 8
  # Fee income
  results$fee_income <- fee_rate * results$vol_frac * days
  results$net_pnl_pct <- (results$fee_income + results$expected_il) * 100
  results$profitable <- results$net_pnl_pct > 0
  results
}

# ---------------------------------------------------------------------------
# 4. CONCENTRATED LIQUIDITY (UNISWAP V3)
# ---------------------------------------------------------------------------

#' Optimal tick range width for concentrated liquidity
#' Given current price P and annualized vol sigma, find [P_low, P_high]
#' that maximizes expected fee income after accounting for out-of-range risk.
#'
#' Capital efficiency multiplier for range [P_a, P_b] at current P:
#' CE = sqrt(P) / (sqrt(P) - sqrt(P_a))   (for P in [P_a, P_b])
#'
#' But if price leaves range, LP earns 0 fees.
#' Optimal range balances CE gain vs probability of staying in range.

#' Probability price stays within [P_a, P_b] over T days under GBM
prob_in_range <- function(P, P_low, P_high, sigma, T_days) {
  log_low  <- log(P_low / P)
  log_high <- log(P_high / P)
  sigma_T  <- sigma * sqrt(T_days / 365)
  # P(log_low <= log(P_T/P) <= log_high) under N(0, sigma_T^2)
  pnorm(log_high / sigma_T) - pnorm(log_low / sigma_T)
}

#' Capital efficiency for UniV3 position
#' Ratio of v3 liquidity to v2 equivalent capital (how much more concentrated)
v3_capital_efficiency <- function(P, P_low, P_high) {
  # From UniV3 whitepaper: CE = 1 / (1 - sqrt(P_low/P_high))
  # At P = geometric mean of range
  sqrt_P      <- sqrt(P)
  sqrt_P_low  <- sqrt(P_low)
  sqrt_P_high <- sqrt(P_high)
  # Virtual reserves scaling
  ce <- (sqrt_P - sqrt_P_low)
  ce_inv <- sqrt_P / ce
  ce_inv
}

#' Find optimal range width [P/k, P*k] that maximizes expected fees
#' Expected fees proportional to CE * prob_in_range
optimal_v3_range <- function(P = 1, sigma_annual = 0.80, T_days = 1,
                              k_grid = seq(1.01, 3.0, 0.01),
                              fee_rate = 0.003) {
  results <- data.frame(k = k_grid)
  results$P_low  <- P / results$k
  results$P_high <- P * results$k
  results$prob_in_range <- mapply(
    prob_in_range, P_low = results$P_low, P_high = results$P_high,
    MoreArgs = list(P = P, sigma = sigma_annual, T_days = T_days)
  )
  results$ce <- mapply(
    v3_capital_efficiency, P_low = results$P_low, P_high = results$P_high,
    MoreArgs = list(P = P)
  )
  # Expected fee multiple = CE * prob_in_range (relative to v2 baseline)
  results$expected_fee_mult <- results$ce * results$prob_in_range
  results$optimal <- results$expected_fee_mult == max(results$expected_fee_mult)
  results
}

# ---------------------------------------------------------------------------
# 5. YIELD FARMING APY WITH AUTO-COMPOUNDING AND TVL DECAY
# ---------------------------------------------------------------------------

#' Continuous compounding APY from APR
apr_to_apy <- function(apr, compounds_per_year = 365) {
  (1 + apr / compounds_per_year)^compounds_per_year - 1
}

#' Auto-compounding yield over time with TVL dilution
#' As more capital enters, reward APR decreases proportionally
#' TVL(t) = TVL_0 * (1 + inflow_rate)^t (logistic or exponential)
simulate_yield_farming <- function(
  initial_tvl    = 1e8,      # $100M initial TVL
  daily_emission = 1e5,      # $100K daily token emissions
  initial_stake  = 1e6,      # User stakes $1M
  days           = 180,
  tvl_growth_rate = 0.005,   # 0.5% daily TVL growth from new entrants
  token_price_drift = -0.002, # Token price declines 0.2%/day (emission sell pressure)
  compound_freq  = 1          # Compound daily
) {
  tvl <- numeric(days + 1)
  user_value <- numeric(days + 1)
  apy <- numeric(days + 1)
  token_price <- numeric(days + 1)

  tvl[1] <- initial_tvl
  user_value[1] <- initial_stake
  token_price[1] <- 1.0

  for (i in 2:(days + 1)) {
    # Token price evolves with drift
    token_price[i] <- token_price[i-1] * exp(token_price_drift)

    # Daily reward rate = daily emissions (in USD) / TVL
    daily_reward_rate <- (daily_emission * token_price[i]) / tvl[i-1]
    apy[i] <- apr_to_apy(daily_reward_rate * 365)

    # User's stake grows by reward (auto-compounded)
    user_value[i] <- user_value[i-1] * (1 + daily_reward_rate)

    # TVL grows (new entrants chasing yield)
    tvl[i] <- tvl[i-1] * (1 + tvl_growth_rate)
  }

  data.frame(
    day        = 0:days,
    tvl        = tvl,
    user_value = user_value,
    token_price = token_price,
    daily_apy  = apy,
    user_pnl_pct = (user_value / initial_stake - 1) * 100
  )
}

# ---------------------------------------------------------------------------
# 6. LIQUIDATION MECHANICS
# ---------------------------------------------------------------------------

#' Health factor for a lending position
#' HF = (collateral * liquidation_threshold) / debt
#' HF < 1 => liquidatable
health_factor <- function(collateral_value, debt_value,
                           liquidation_threshold = 0.825) {
  (collateral_value * liquidation_threshold) / debt_value
}

#' Loan-to-value ratio
ltv <- function(debt_value, collateral_value) {
  debt_value / collateral_value
}

#' Maximum borrowable given collateral
max_borrow <- function(collateral_value, max_ltv = 0.75) {
  collateral_value * max_ltv
}

#' Price at which position gets liquidated
#' collateral_value(P) = n_tokens * P
#' Liquidation when HF = 1: P_liq = debt / (n_tokens * liq_threshold)
liquidation_price <- function(n_collateral_tokens, debt_value,
                               liquidation_threshold = 0.825) {
  debt_value / (n_collateral_tokens * liquidation_threshold)
}

#' Liquidation bonus: liquidator acquires collateral at discount
#' Liquidator repays debt, receives collateral * (1 + bonus)
liquidation_pnl <- function(debt_repaid, collateral_price,
                             collateral_received, liquidation_bonus = 0.05) {
  collateral_value <- collateral_received * collateral_price
  profit <- collateral_value * (1 + liquidation_bonus) - debt_repaid
  # More precisely: liquidator pays debt_repaid, gets collateral at discount
  collateral_at_discount <- debt_repaid * (1 + liquidation_bonus) / collateral_price
  list(
    collateral_received = collateral_at_discount,
    profit_usd = collateral_at_discount * collateral_price - debt_repaid
  )
}

#' Cascade liquidation simulation
#' When price drops, positions become liquidatable -> forced sells -> more price drop
simulate_liquidation_cascade <- function(
  initial_price  = 50000,    # BTC price
  price_shock    = -0.20,    # Initial -20% shock
  n_positions    = 500,      # Number of leveraged positions
  seed           = 123,
  ltv_mean       = 0.65,
  ltv_sd         = 0.10,
  position_size_mean = 1e5,
  liq_threshold  = 0.825,
  market_impact_per_bn = 0.005,  # 0.5% price impact per $1B liquidated
  max_rounds     = 20
) {
  set.seed(seed)

  # Generate position portfolio
  ltvs <- pmin(pmax(rnorm(n_positions, ltv_mean, ltv_sd), 0.3), 0.90)
  sizes <- rlnorm(n_positions, log(position_size_mean), 0.8)
  # Debt = LTV * collateral_value (in USD at initial price)
  collateral_usd <- sizes
  debts <- ltvs * collateral_usd
  # Collateral in BTC units
  collateral_btc <- collateral_usd / initial_price

  price <- initial_price * (1 + price_shock)
  results <- data.frame(round = 0, price = initial_price,
                        liquidations = 0, volume_liquidated = 0,
                        cumulative_liquidated = 0)

  cum_liq <- 0
  for (r in 1:max_rounds) {
    # Check which positions are liquidatable at current price
    hf <- health_factor(collateral_btc * price, debts, liq_threshold)
    liq_mask <- hf < 1 & sizes > 0  # Not yet liquidated

    if (!any(liq_mask)) break

    # Liquidate all underwater positions this round
    liq_volume <- sum(collateral_btc[liq_mask] * price)
    cum_liq <- cum_liq + liq_volume

    # Mark as liquidated
    sizes[liq_mask] <- 0

    # Price impact from forced selling
    price_impact <- -market_impact_per_bn * liq_volume / 1e9
    price_new <- price * (1 + price_impact)

    results <- rbind(results, data.frame(
      round = r, price = price_new,
      liquidations = sum(liq_mask),
      volume_liquidated = liq_volume,
      cumulative_liquidated = cum_liq
    ))

    price <- price_new
  }

  attr(results, "total_price_drop_pct") <- (price / initial_price - 1) * 100
  attr(results, "cascade_amplification") <- abs(price / initial_price - 1) / abs(price_shock)
  results
}

# ---------------------------------------------------------------------------
# 7. MEV SANDWICH ATTACK COST ESTIMATION
# ---------------------------------------------------------------------------

#' Estimate cost of MEV sandwich attack on a victim trade
#' Attacker front-runs (buys X before victim), victim executes at worse price,
#' attacker back-runs (sells X after victim).
#'
#' Profit to attacker = (price after back-run - price paid in front-run) * amount
#' Cost to victim = extra slippage due to sandwiching
mev_sandwich_cost <- function(dx_victim, reserve_x, reserve_y,
                               dx_frontrun = NULL, fee = 0.003) {
  # If frontrun size not specified, optimize for attacker profit
  if (is.null(dx_frontrun)) {
    # Simple heuristic: front-run with ~10% of victim's trade
    dx_frontrun <- dx_victim * 0.1
  }

  # Step 1: Attacker front-runs (buys X, pushing price up)
  dy_frontrun <- amm_output(dx_frontrun, reserve_x, reserve_y, fee)
  rx1 <- reserve_x + dx_frontrun
  ry1 <- reserve_y - dy_frontrun
  price_after_frontrun <- ry1 / rx1

  # Step 2: Victim executes (worse price)
  dy_victim_sandwiched <- amm_output(dx_victim, rx1, ry1, fee)
  rx2 <- rx1 + dx_victim
  ry2 <- ry1 - dy_victim_sandwiched

  # Compare victim execution without sandwich
  dy_victim_clean <- amm_output(dx_victim, reserve_x, reserve_y, fee)

  # Step 3: Attacker back-runs (sells X received from front-run)
  # Attacker received dy_frontrun Y tokens for dx_frontrun X
  # Now sells dy_frontrun Y to get X back (reversed trade)
  # Actually: attacker sells X (dx_frontrun) back to pool
  dy_backrun <- amm_output(dx_frontrun, ry2, rx2, fee)  # Reversed reserves
  # Profit: attacker paid dy_frontrun Y, sold dx_frontrun X, received dy_backrun Y
  attacker_profit_y <- dy_backrun - dy_frontrun

  # Victim's extra cost
  victim_shortfall_y <- dy_victim_clean - dy_victim_sandwiched

  list(
    victim_dx           = dx_victim,
    frontrun_size       = dx_frontrun,
    dy_victim_clean     = dy_victim_clean,
    dy_victim_sandwiched = dy_victim_sandwiched,
    victim_shortfall_y  = victim_shortfall_y,
    victim_cost_pct     = victim_shortfall_y / dy_victim_clean * 100,
    attacker_profit_y   = attacker_profit_y,
    attacker_roi_pct    = attacker_profit_y / dy_frontrun * 100
  )
}

#' MEV cost as fraction of trade size across different pool depths
mev_sensitivity_analysis <- function(trade_sizes = c(1e3, 1e4, 5e4, 1e5, 5e5),
                                     pool_size = 1e7, fee = 0.003) {
  results <- lapply(trade_sizes, function(dx) {
    r <- mev_sandwich_cost(dx, pool_size, pool_size, fee = fee)
    data.frame(
      trade_size    = dx,
      victim_cost_pct = r$victim_cost_pct,
      attacker_roi_pct = r$attacker_roi_pct
    )
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 8. PROTOCOL REVENUE MODEL: TVL -> VOLUME -> FEES -> FDV
# ---------------------------------------------------------------------------

#' Estimate protocol revenue from TVL via volume and fees
#' P/S ratio (Price-to-Sales) is key DeFi valuation metric
protocol_revenue_model <- function(
  tvl            = 1e9,       # $1B TVL
  volume_to_tvl  = 0.15,      # 15% daily volume/TVL (Uniswap-like)
  fee_rate       = 0.003,     # 0.3% per trade
  protocol_take  = 0.20,      # 20% of fees to protocol treasury
  fdv_ps_ratio   = 15,        # FDV / annual revenue multiple
  token_supply   = 1e9        # Total token supply
) {
  daily_volume   <- tvl * volume_to_tvl
  daily_fees     <- daily_volume * fee_rate
  protocol_daily <- daily_fees * protocol_take
  annual_revenue <- protocol_daily * 365

  # FDV estimate
  fdv <- annual_revenue * fdv_ps_ratio
  token_price <- fdv / token_supply

  list(
    daily_volume   = daily_volume,
    daily_fees_total = daily_fees,
    daily_protocol_revenue = protocol_daily,
    annual_revenue = annual_revenue,
    fdv_estimate   = fdv,
    token_price    = token_price,
    ps_ratio       = fdv_ps_ratio
  )
}

#' Revenue sensitivity across TVL scenarios
revenue_sensitivity <- function(tvl_range = 10^seq(7, 11, 0.5),
                                 volume_to_tvl = 0.10, fee_rate = 0.003,
                                 protocol_take = 0.17) {
  results <- lapply(tvl_range, function(tvl) {
    r <- protocol_revenue_model(tvl, volume_to_tvl, fee_rate, protocol_take)
    data.frame(
      tvl = tvl,
      annual_revenue = r$annual_revenue,
      fdv_at_10x_ps = r$annual_revenue * 10,
      fdv_at_20x_ps = r$annual_revenue * 20
    )
  })
  do.call(rbind, results)
}

#' Token emission inflation impact on price
#' If emissions exceed fee buybacks, price declines structurally
token_emission_pressure <- function(
  fdv            = 1e9,
  circulating_supply_frac = 0.30,  # 30% of tokens in circulation
  annual_emission_pct = 0.15,      # 15% annual inflation
  annual_buyback_usd = 5e6,        # $5M annual buyback
  token_supply   = 1e9
) {
  circ_supply  <- token_supply * circulating_supply_frac
  token_price  <- fdv / token_supply
  market_cap   <- circ_supply * token_price

  annual_emission_tokens <- token_supply * annual_emission_pct
  emission_sell_pressure <- annual_emission_tokens * token_price

  # Net flow (negative = sell pressure dominates)
  net_pressure <- annual_buyback_usd - emission_sell_pressure
  net_pressure_pct_mcap <- net_pressure / market_cap * 100

  list(
    market_cap = market_cap,
    emission_sell_pressure_usd = emission_sell_pressure,
    buyback_usd = annual_buyback_usd,
    net_pressure_usd = net_pressure,
    net_pressure_pct_market_cap = net_pressure_pct_mcap,
    structurally_bullish = net_pressure > 0
  )
}

# ---------------------------------------------------------------------------
# 9. TVL GROWTH MODEL WITH NETWORK EFFECTS
# ---------------------------------------------------------------------------

#' Logistic TVL growth model
#' dTVL/dt = r * TVL * (1 - TVL / K)
#' K = carrying capacity (maximum achievable TVL)
logistic_tvl_growth <- function(tvl0 = 1e8, K = 1e10, r = 0.005, days = 365) {
  # Analytical solution: TVL(t) = K / (1 + ((K - tvl0)/tvl0) * exp(-r * t))
  t <- 0:days
  tvl <- K / (1 + ((K - tvl0) / tvl0) * exp(-r * t))
  data.frame(day = t, tvl = tvl,
             growth_rate = c(NA, diff(tvl) / tvl[-length(tvl)]))
}

#' Network effect multiplier: TVL attracts more TVL (Metcalfe's law)
#' Value proportional to n^2 where n = number of users
metcalfe_tvl_boost <- function(tvl_base, user_base, elasticity = 0.5) {
  # Each doubling of users multiplies TVL by 2^elasticity
  tvl_base * (user_base / 1000)^elasticity
}

# ---------------------------------------------------------------------------
# 10. DEX ARBITRAGE PROFITABILITY
# ---------------------------------------------------------------------------

#' Arbitrage profit between two AMM pools with same pair
#' Pool A has price P_A, Pool B has price P_B > P_A
#' Arb: buy X from A (cheaper), sell X to B (more expensive)
dex_arb_profit <- function(P_A, P_B, reserve_xA, reserve_yA,
                            reserve_xB, reserve_yB,
                            fee_A = 0.003, fee_B = 0.003,
                            gas_cost_usd = 10) {
  stopifnot(P_B > P_A)

  # Optimal arb: equalize prices via AMM formula
  # Buy dx in pool A to move P_A up; sell dx in pool B to move P_B down
  # Simplified: compute max profit with grid search
  dx_grid <- seq(0, min(reserve_xA, reserve_xB) * 0.1, length.out = 1000)[-1]

  profits <- sapply(dx_grid, function(dx) {
    dy_buy  <- amm_output(dx, reserve_xA, reserve_yA, fee_A)  # Spend dy_buy Y to get dx X from A
    # Actually: to buy X from A, we input Y. Let's think in terms of inputting Y:
    # Input dy_in Y to get dx_out X from pool A
    dy_in <- dx  # Treat as Y input for simplicity
    dx_out <- amm_output(dy_in, reserve_yA, reserve_xA, fee_A)  # Buy X with Y
    dy_out <- amm_output(dx_out, reserve_xB, reserve_yB, fee_B)  # Sell X for Y
    dy_out - dy_in  # Raw profit in Y
  })

  best_idx <- which.max(profits)
  list(
    optimal_dx    = dx_grid[best_idx],
    gross_profit_y = max(profits),
    gas_cost      = gas_cost_usd,
    net_profit    = max(profits) - gas_cost_usd,
    profitable    = max(profits) > gas_cost_usd
  )
}

# ---------------------------------------------------------------------------
# 11. SUMMARY ANALYTICS AND REPORTING
# ---------------------------------------------------------------------------

#' Full DeFi analytics report for a given pool
defi_pool_report <- function(reserve_x = 1e6, reserve_y = 1e9,
                              sigma_annual = 0.80, fee_rate = 0.003,
                              daily_vol_frac = 0.10) {
  cat("╔══════════════════════════════════════════════════════╗\n")
  cat("║          DeFi Pool Analytics Report                  ║\n")
  cat("╚══════════════════════════════════════════════════════╝\n\n")

  cat("--- Pool Parameters ---\n")
  cat(sprintf("  Reserve X:    %s\n", format(reserve_x, big.mark=",")))
  cat(sprintf("  Reserve Y:    %s\n", format(reserve_y, big.mark=",")))
  cat(sprintf("  Spot Price:   %.4f Y/X\n", amm_spot_price(reserve_x, reserve_y)))
  cat(sprintf("  TVL:          $%s\n", format(2*reserve_y, big.mark=",")))
  cat(sprintf("  Fee Rate:     %.2f%%\n", fee_rate * 100))
  cat("\n")

  cat("--- Price Impact (buying X) ---\n")
  for (pct in c(0.001, 0.005, 0.01, 0.05)) {
    dx <- reserve_x * pct
    imp <- amm_price_impact(dx, reserve_x, reserve_y, fee_rate)
    cat(sprintf("  Trade = %.1f%% of pool:  impact = %.4f%%\n",
                pct*100, imp*100))
  }
  cat("\n")

  cat("--- Impermanent Loss Table ---\n")
  cat("  Price Change | IL\n")
  for (r in c(0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 5.0)) {
    il <- impermanent_loss(r)
    cat(sprintf("  x%.2f (%+.0f%%)  | %.3f%%\n",
                r, (r-1)*100, il*100))
  }
  cat("\n")

  cat("--- Breakeven Analysis ---\n")
  be_vol <- il_breakeven_volatility(fee_rate, daily_vol_frac)
  cat(sprintf("  Daily vol fraction:     %.1f%%\n", daily_vol_frac*100))
  cat(sprintf("  Breakeven daily vol:    %.2f%%\n", be_vol$daily_vol_breakeven*100))
  cat(sprintf("  Breakeven annual vol:   %.1f%%\n", be_vol$annual_vol_pct))
  cat(sprintf("  BTC/ETH annual vol ~80%%: LP %s at this vol\n",
              if (0.80 > be_vol$annual_vol_breakeven) "LOSES" else "BREAKS EVEN"))
  cat("\n")

  cat("--- Optimal V3 Range (daily rebalance) ---\n")
  opt <- optimal_v3_range(P = 1, sigma_annual = sigma_annual,
                           T_days = 1, fee_rate = fee_rate)
  best_row <- opt[opt$optimal, ]
  cat(sprintf("  Best k (range = [P/k, P*k]): %.2f\n", best_row$k))
  cat(sprintf("  Range: [%.4f, %.4f]\n", best_row$P_low, best_row$P_high))
  cat(sprintf("  Prob in range (1 day): %.1f%%\n", best_row$prob_in_range*100))
  cat(sprintf("  Capital efficiency vs v2: %.1fx\n", best_row$ce))
  cat(sprintf("  Expected fee multiple:  %.2fx\n", best_row$expected_fee_mult))
  cat("\n")
}

# ---------------------------------------------------------------------------
# 12. MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  # AMM mechanics demo
  demo_amm()

  # IL table
  print(il_table())

  # Simulate LP path over 1 year
  path <- simulate_il_path(n = 365, sigma = 0.04, seed = 1)
  cat(sprintf("Final LP vs HODL: %.2f%%\n",
              tail(path$net_pnl_vs_hodl_pct, 1)))

  # Liquidation cascade
  casc <- simulate_liquidation_cascade(initial_price = 65000,
                                        price_shock = -0.15,
                                        n_positions = 1000)
  cat(sprintf("Cascade amplification: %.2fx\n",
              attr(casc, "cascade_amplification")))
  cat(sprintf("Total additional price drop: %.2f%%\n",
              attr(casc, "total_price_drop_pct")))

  # MEV sandwich
  mev <- mev_sensitivity_analysis()
  print(mev)

  # Protocol revenue
  rev <- protocol_revenue_model(tvl = 5e9, volume_to_tvl = 0.12)
  cat(sprintf("Annual protocol revenue: $%s\n",
              format(round(rev$annual_revenue), big.mark=",")))
  cat(sprintf("FDV estimate (15x P/S): $%s\n",
              format(round(rev$fdv_estimate), big.mark=",")))

  # Full pool report
  defi_pool_report()
}
