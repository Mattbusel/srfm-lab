# =============================================================================
# defi_research.R
# DeFi Research Toolkit: LP profitability, rebalancing, arbitrage estimation,
# TVL growth models, token emission analysis, governance premium, cross-protocol
# correlation, and DeFi vs CeFi return comparison.
# Pure base R.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. LP RETURN CALCULATOR WITH FEE TIERS AND IMPERMANENT LOSS
# ---------------------------------------------------------------------------

#' Compute total LP return over holding period
#' Accounts for: fee income, impermanent loss, and base asset performance
#'
#' UniV2: wide range, capital inefficient but always earns fees
#' UniV3: concentrated range, capital efficient but can leave range
lp_return_calculator <- function(
  initial_price    = 1000,    # Initial price of risky asset (in stable)
  final_price      = 1200,    # Price at exit
  initial_tvl      = 1e7,     # Total pool TVL at entry ($)
  lp_stake_usd     = 1e5,     # LP's initial stake ($)
  fee_tier         = 0.003,   # 0.3%
  daily_volume_frac = 0.12,   # Daily volume as fraction of TVL
  holding_days     = 30,
  is_concentrated  = FALSE,   # UniV3 vs UniV2
  range_width_k    = 1.2      # UniV3 range: [P/k, P*k]
) {
  # Price ratio
  price_ratio <- final_price / initial_price

  # Impermanent loss (always negative or zero)
  il <- 2 * sqrt(price_ratio) / (1 + price_ratio) - 1

  # LP share of pool
  lp_share <- lp_stake_usd / initial_tvl

  # Capital efficiency multiplier for UniV3
  ce_mult <- 1.0
  if (is_concentrated) {
    # CE = 1 / (1 - sqrt(P_low/P_high)) at current price
    p_low  <- initial_price / range_width_k
    p_high <- initial_price * range_width_k
    sqrt_ratio <- sqrt(p_low / p_high)
    ce_mult <- 1 / (1 - sqrt_ratio)

    # But: probability of staying in range over holding period
    # Use lognormal: P(P_low <= P_T <= P_high)
    sigma_daily <- 0.04  # Assume 4% daily vol
    sigma_T <- sigma_daily * sqrt(holding_days)
    prob_in_range <- pnorm(log(p_high/initial_price) / sigma_T) -
                     pnorm(log(p_low/initial_price)  / sigma_T)

    # Effective CE = ce_mult * prob_in_range
    ce_mult <- ce_mult * prob_in_range
  }

  # Fee income (annualized)
  daily_fee_rate  <- fee_tier * daily_volume_frac
  total_fee_rate  <- daily_fee_rate * holding_days * ce_mult  # With CE boost

  # Total LP return relative to initial stake
  hodl_return    <- (price_ratio + 1) / 2 - 1  # 50/50 portfolio return
  lp_return_nofee <- hodl_return + il
  lp_return_total <- lp_return_nofee + total_fee_rate

  list(
    price_ratio     = price_ratio,
    il_pct          = il * 100,
    fee_income_pct  = total_fee_rate * 100,
    hodl_return_pct = hodl_return * 100,
    lp_return_nofee_pct = lp_return_nofee * 100,
    lp_return_total_pct = lp_return_total * 100,
    outperforms_hodl    = lp_return_total > hodl_return,
    ce_multiplier       = ce_mult
  )
}

#' LP profitability across scenarios: price change grid x volatility
lp_profitability_grid <- function(
  price_changes = seq(-0.8, 2.0, by = 0.1),
  fee_tiers = c(0.0005, 0.003, 0.010),
  daily_vol_fracs = c(0.05, 0.10, 0.20),
  holding_days = 30
) {
  results <- expand.grid(
    price_change = price_changes,
    fee_tier     = fee_tiers,
    vol_frac     = daily_vol_fracs
  )

  results$lp_return <- mapply(function(pc, ft, vf) {
    final_p <- 1000 * (1 + pc)
    r <- lp_return_calculator(1000, final_p, 1e7, 1e5, ft, vf, holding_days)
    r$lp_return_total_pct
  }, results$price_change, results$fee_tier, results$vol_frac)

  results$hodl_return <- ((1 + results$price_change) + 1) / 2 - 1
  results$vs_hodl     <- results$lp_return - results$hodl_return * 100
  results
}

# ---------------------------------------------------------------------------
# 2. OPTIMAL REBALANCING FOR LP POSITIONS
# ---------------------------------------------------------------------------

#' Determine optimal rebalancing frequency for UniV3 LP
#' Trade-off: rebalancing costs (fees + gas) vs staying in-range
optimal_lp_rebalancing <- function(
  initial_price    = 1000,
  sigma_daily      = 0.04,
  range_width_k    = 1.2,     # Range = [P/k, P*k]
  gas_cost_usd     = 20,      # Gas per rebalance
  position_size    = 1e5,
  fee_rate         = 0.003,
  daily_vol_frac   = 0.10,
  max_days         = 60
) {
  results <- lapply(seq_len(max_days), function(d) {
    p_low  <- initial_price / range_width_k
    p_high <- initial_price * range_width_k
    sigma_T <- sigma_daily * sqrt(d)

    # Probability price stays in range over d days
    prob_in <- pnorm(log(p_high/initial_price) / sigma_T) -
               pnorm(log(p_low/initial_price)  / sigma_T)

    # Expected fee income if in range
    ce <- 1 / (1 - sqrt(p_low/p_high))
    expected_fee_pct <- fee_rate * daily_vol_frac * d * ce * prob_in

    # IL cost (expected)
    # E[IL] ≈ -sigma^2 * d / 8 (lognormal approximation)
    expected_il_pct <- -(sigma_daily^2 * d) / 8 * 100

    # Rebalancing cost (once per d days)
    rebal_cost_pct <- gas_cost_usd / position_size * 100

    # Net expected return vs HODL
    net_return <- expected_fee_pct * 100 + expected_il_pct - rebal_cost_pct

    data.frame(
      holding_days = d,
      prob_in_range = prob_in,
      expected_fee_pct = expected_fee_pct * 100,
      expected_il_pct  = expected_il_pct,
      rebal_cost_pct   = rebal_cost_pct,
      net_vs_hodl_pct  = net_return
    )
  })

  df <- do.call(rbind, results)
  df$optimal <- df$net_vs_hodl_pct == max(df$net_vs_hodl_pct)

  list(
    grid         = df,
    optimal_days = df$holding_days[df$optimal],
    max_return   = max(df$net_vs_hodl_pct)
  )
}

# ---------------------------------------------------------------------------
# 3. ARB PROFIT ESTIMATION: DEX vs CEX
# ---------------------------------------------------------------------------

#' Estimate arbitrage profit between DEX and CEX
#' When DEX price deviates from CEX price, arb bots profit
#' But LP loses: each arb trade takes value from the LP
dex_cex_arb_profit <- function(
  dex_price, cex_price,
  dex_reserve_x, dex_reserve_y,  # AMM reserves
  arb_size_usd   = NULL,          # If NULL: compute optimal arb size
  dex_fee        = 0.003,
  cex_fee        = 0.001,
  gas_cost_usd   = 15
) {
  stopifnot(dex_price != cex_price)

  if (dex_price < cex_price) {
    # Buy X cheap on DEX, sell X expensive on CEX
    direction <- "buy_dex_sell_cex"
    price_diff_pct <- (cex_price - dex_price) / dex_price
  } else {
    direction <- "buy_cex_sell_dex"
    price_diff_pct <- (dex_price - cex_price) / cex_price
  }

  # Estimate optimal arb size (maximizes profit accounting for price impact)
  k <- dex_reserve_x * dex_reserve_y

  # Gross price spread available
  gross_spread_pct <- price_diff_pct - dex_fee - cex_fee

  if (gross_spread_pct <= 0) {
    return(list(profitable = FALSE, direction = direction,
                price_diff_pct = price_diff_pct * 100))
  }

  # Optimal trade size = sqrt(k * spread) - reserve (rough approximation)
  if (is.null(arb_size_usd)) {
    # Maximize: profit = spread * Q - impact_cost * Q^2
    # dProfit/dQ = 0 => Q* = spread / (2 * impact_coef)
    impact_coef <- 1 / dex_reserve_y  # Lambda for AMM
    arb_size_usd <- gross_spread_pct / (2 * impact_coef) * dex_price
    arb_size_usd <- min(arb_size_usd, dex_reserve_y * 0.05)  # Cap at 5% of pool
  }

  # Price impact on DEX
  dx <- arb_size_usd / dex_price  # Tokens to buy
  # AMM output: dy = (dx*(1-fee)*ry) / (rx + dx*(1-fee))
  dx_net  <- dx * (1 - dex_fee)
  dy_out  <- (dx_net * dex_reserve_y) / (dex_reserve_x + dx_net)
  dex_exec_price <- dy_out / dx

  # Profit
  cex_sell_value <- dx * cex_price * (1 - cex_fee)
  dex_buy_cost   <- dy_out  # Paid in Y (stable)

  gross_profit <- cex_sell_value - dex_buy_cost
  net_profit   <- gross_profit - gas_cost_usd

  list(
    profitable       = net_profit > 0,
    direction        = direction,
    price_diff_pct   = price_diff_pct * 100,
    gross_spread_pct = gross_spread_pct * 100,
    arb_size_usd     = arb_size_usd,
    gross_profit     = gross_profit,
    gas_cost         = gas_cost_usd,
    net_profit       = net_profit,
    roi_pct          = net_profit / arb_size_usd * 100,
    lp_loss          = gross_profit  # LP loses what arb gains
  )
}

# ---------------------------------------------------------------------------
# 4. TVL GROWTH MODEL
# ---------------------------------------------------------------------------

#' Logistic + network effect TVL model
#' dTVL/dt = r * TVL * (1 - TVL/K) + alpha * TVL^2 / K  (network effect)
tvl_growth_model <- function(tvl0 = 1e8, K = 5e9, r = 0.004,
                              alpha = 0.001, days = 365, dt = 1) {
  t_seq <- seq(0, days, by = dt)
  tvl   <- numeric(length(t_seq))
  tvl[1] <- tvl0

  for (i in 2:length(t_seq)) {
    dTVL <- r * tvl[i-1] * (1 - tvl[i-1]/K) +
            alpha * tvl[i-1]^2 / K
    tvl[i] <- max(0, tvl[i-1] + dt * dTVL)
  }

  data.frame(
    day = t_seq,
    tvl = tvl,
    daily_inflow = c(NA, diff(tvl)),
    growth_rate  = c(NA, diff(tvl)/tvl[-length(tvl)])
  )
}

#' Protocol TVL vs market conditions
#' TVL tends to rise with BTC price (risk-on) and fall in bear markets
tvl_market_correlation <- function(n = 500, seed = 7) {
  set.seed(seed)
  btc_ret  <- rnorm(n, 0.001, 0.04)
  btc_price <- 50000 * exp(cumsum(btc_ret))

  # TVL has high beta to BTC but also independent growth
  tvl_beta  <- 1.5
  tvl_alpha <- 0.0005  # Daily organic growth

  log_tvl <- numeric(n)
  log_tvl[1] <- log(5e9)
  for (i in 2:n) {
    log_tvl[i] <- log_tvl[i-1] + tvl_alpha + tvl_beta * btc_ret[i] +
                  rnorm(1, 0, 0.015)
  }

  data.frame(
    t = seq_len(n), btc_price = btc_price,
    tvl = exp(log_tvl),
    btc_return = btc_ret,
    tvl_return = c(NA, diff(log_tvl))
  )
}

# ---------------------------------------------------------------------------
# 5. TOKEN EMISSION AND INFLATION IMPACT
# ---------------------------------------------------------------------------

#' Token emission schedule (typical: 4-year cliff + linear vesting)
token_emission_schedule <- function(
  total_supply    = 1e9,
  team_pct        = 0.20,    # 20% to team, 4-year vesting, 1-year cliff
  investor_pct    = 0.25,    # 25% to investors
  community_pct   = 0.55,    # 55% to community (staking rewards, treasury)
  vesting_years   = 4,
  cliff_years     = 1,
  days            = 365 * 5
) {
  d_seq <- seq_len(days)

  # Team tokens: 0 before cliff, linear after
  team_total      <- total_supply * team_pct
  team_daily_vest <- team_total / (vesting_years * 365 - cliff_years * 365)
  team_emission   <- ifelse(d_seq <= cliff_years * 365, 0,
                     ifelse(d_seq <= vesting_years * 365, team_daily_vest, 0))

  # Investor tokens: similar schedule but 3-year
  inv_total <- total_supply * investor_pct
  inv_daily <- inv_total / (3 * 365 - cliff_years * 365)
  inv_emission <- ifelse(d_seq <= cliff_years * 365, 0,
                  ifelse(d_seq <= 3 * 365, inv_daily, 0))

  # Community rewards: decreasing emission (halving model)
  comm_total <- total_supply * community_pct
  # Exponential decay with 2-year half-life
  halflife   <- 2 * 365
  lambda     <- log(2) / halflife
  comm_daily <- comm_total * lambda * exp(-lambda * d_seq)
  # Normalize so total = comm_total
  comm_daily <- comm_daily / sum(comm_daily) * comm_total

  total_daily <- team_emission + inv_emission + comm_daily
  circulating  <- cumsum(total_daily)

  data.frame(
    day = d_seq,
    team_emission  = team_emission,
    inv_emission   = inv_emission,
    comm_emission  = comm_daily,
    total_daily    = total_daily,
    circulating    = circulating,
    circ_pct_supply = circulating / total_supply * 100,
    annual_inflation_rate = c(NA, (total_daily[-1] * 365) / (circulating[-days] + 1))
  )
}

#' Inflation impact on token price via sell pressure model
inflation_price_impact <- function(emission_df, initial_price = 1.0,
                                    sell_fraction = 0.70,
                                    daily_buy_usd = 1e5) {
  n <- nrow(emission_df)
  price <- numeric(n)
  price[1] <- initial_price

  for (i in 2:n) {
    # Sell pressure: emission * sell_fraction * price
    sell_pressure_tokens <- emission_df$total_daily[i] * sell_fraction
    sell_pressure_usd    <- sell_pressure_tokens * price[i-1]

    # Net flow
    net_flow_usd <- daily_buy_usd - sell_pressure_usd

    # Price elasticity (simplified): 1% price change per 0.1% of market cap flow
    market_cap <- emission_df$circulating[i] * price[i-1]
    if (market_cap > 0) {
      price_change <- net_flow_usd / market_cap
      price[i] <- price[i-1] * (1 + price_change)
    } else {
      price[i] <- price[i-1]
    }
    price[i] <- max(price[i], 1e-6)
  }

  cbind(emission_df, price = price, market_cap = emission_df$circulating * price)
}

# ---------------------------------------------------------------------------
# 6. GOVERNANCE PARTICIPATION AND PRICE PREMIUM
# ---------------------------------------------------------------------------

#' Test whether high governance participation predicts returns
#' Hypothesis: engaged community = healthier protocol = higher price
governance_premium_analysis <- function(
  gov_participation_rate,  # Fraction of supply voting
  token_returns,
  governance_threshold = 0.10,  # 10% quorum = "active"
  window = 30
) {
  n <- length(token_returns)

  # Rolling average governance participation
  gov_ma <- numeric(n)
  for (i in window:n) {
    gov_ma[i] <- mean(gov_participation_rate[(i-window+1):i], na.rm=TRUE)
  }

  # High vs low governance periods
  high_gov  <- gov_ma > quantile(gov_ma[gov_ma > 0], 0.75, na.rm=TRUE)
  low_gov   <- gov_ma < quantile(gov_ma[gov_ma > 0], 0.25, na.rm=TRUE)

  # Returns in each regime
  ret_high  <- token_returns[high_gov & !is.na(high_gov)]
  ret_low   <- token_returns[low_gov  & !is.na(low_gov)]

  # Premium
  premium_daily <- mean(ret_high, na.rm=TRUE) - mean(ret_low, na.rm=TRUE)

  cat("=== Governance Participation Premium ===\n")
  cat(sprintf("High governance mean daily return: %.4f%%\n", mean(ret_high, na.rm=TRUE)*100))
  cat(sprintf("Low  governance mean daily return: %.4f%%\n", mean(ret_low, na.rm=TRUE)*100))
  cat(sprintf("Governance premium (daily):        %.4f%%\n", premium_daily*100))
  cat(sprintf("Annualized premium:                %.2f%%\n", premium_daily*252*100))

  # Simple t-test
  if (length(ret_high) >= 5 && length(ret_low) >= 5) {
    t_stat <- (mean(ret_high) - mean(ret_low)) /
              sqrt(var(ret_high)/length(ret_high) + var(ret_low)/length(ret_low))
    p_val  <- 2 * pt(-abs(t_stat), df = min(length(ret_high), length(ret_low)) - 1)
    cat(sprintf("T-stat: %.3f | P-value: %.4f | Significant: %s\n",
                t_stat, p_val, if(p_val < 0.05) "Yes" else "No"))
  }

  list(premium_daily = premium_daily, ret_high = ret_high, ret_low = ret_low)
}

# ---------------------------------------------------------------------------
# 7. CROSS-PROTOCOL CORRELATION ANALYSIS
# ---------------------------------------------------------------------------

#' Correlation network among DeFi protocols
#' High correlation = shared risk (e.g., both dependent on ETH price)
defi_correlation_analysis <- function(returns_matrix, protocol_names = NULL) {
  n <- ncol(returns_matrix)
  if (is.null(protocol_names)) protocol_names <- paste0("Protocol", seq_len(n))

  S <- cor(returns_matrix, use = "complete.obs")
  rownames(S) <- colnames(S) <- protocol_names

  # Average correlation (proxy for systemic risk)
  avg_corr <- mean(S[upper.tri(S)])

  # Rolling correlation for each pair
  T_ <- nrow(returns_matrix)
  window <- 30

  rolling_avg_corr <- numeric(T_)
  for (t in window:T_) {
    sub <- returns_matrix[(t-window+1):t, ]
    S_t <- cor(sub, use="complete.obs")
    rolling_avg_corr[t] <- mean(S_t[upper.tri(S_t)], na.rm=TRUE)
  }

  # Identify most/least correlated pairs
  corr_pairs <- data.frame(
    asset1 = character(), asset2 = character(), correlation = numeric(),
    stringsAsFactors = FALSE
  )
  for (i in seq_len(n-1)) {
    for (j in (i+1):n) {
      corr_pairs <- rbind(corr_pairs, data.frame(
        asset1 = protocol_names[i], asset2 = protocol_names[j],
        correlation = S[i, j]
      ))
    }
  }

  cat("=== DeFi Cross-Protocol Correlation ===\n")
  cat(sprintf("Average pairwise correlation: %.3f\n", avg_corr))
  cat("Highest correlated pairs:\n")
  print(head(corr_pairs[order(-corr_pairs$correlation), ], 5))
  cat("Lowest correlated pairs:\n")
  print(head(corr_pairs[order(corr_pairs$correlation), ], 5))

  invisible(list(corr_matrix = S, avg_corr = avg_corr,
                 rolling_avg_corr = rolling_avg_corr,
                 corr_pairs = corr_pairs))
}

# ---------------------------------------------------------------------------
# 8. DEFI vs CEFI RETURN COMPARISON
# ---------------------------------------------------------------------------

#' Compare DeFi strategy returns vs CeFi equivalent
#' DeFi: LP returns (fee income - IL) vs CeFi: centralized exchange maker rebates
defi_vs_cefi_comparison <- function(
  prices_matrix,      # [T x n_assets] price matrix
  defi_fee_rate = 0.003,
  defi_daily_vol_frac = 0.10,
  cefi_maker_rebate_bps = 2,   # 0.2bps rebate for market making
  cefi_fill_rate = 0.60,       # Fraction of quotes that get filled
  n_days = 252,
  seed = 42
) {
  set.seed(seed)
  T_ <- nrow(prices_matrix); n <- ncol(prices_matrix)

  # DeFi LP returns for each asset pair (against stablecoin)
  defi_returns <- sapply(seq_len(n), function(i) {
    prices <- prices_matrix[, i]
    daily_rets <- c(NA, diff(log(prices)))

    lp_rets <- numeric(T_)
    for (t in 2:T_) {
      price_ratio <- prices[t] / prices[t-1]
      il_daily <- 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
      fee_income <- defi_fee_rate * defi_daily_vol_frac
      lp_rets[t] <- il_daily + fee_income
    }
    lp_rets
  })

  # CeFi market making returns
  cefi_daily_ret <- cefi_maker_rebate_bps / 10000 * cefi_fill_rate

  # Performance comparison
  defi_sharpe <- apply(defi_returns[2:T_, ], 2, function(r) {
    mean(r, na.rm=TRUE) / (sd(r, na.rm=TRUE) + 1e-8) * sqrt(252)
  })

  cefi_sharpe <- cefi_daily_ret / (0.001) * sqrt(252)  # Assume small vol for MM

  cat("=== DeFi LP vs CeFi Market Making ===\n")
  cat(sprintf("CeFi MM Sharpe (estimated): %.2f\n", cefi_sharpe))
  cat(sprintf("CeFi MM Daily Return: %.3f bps\n", cefi_daily_ret * 10000))
  cat("\nDeFi LP by Asset:\n")
  for (i in seq_len(n)) {
    cum_ret <- prod(1 + defi_returns[-1, i]) - 1
    cat(sprintf("  Asset %d: Sharpe=%.2f, CumRet=%.2f%%\n",
                i, defi_sharpe[i], cum_ret*100))
  }

  invisible(list(defi_returns = defi_returns, defi_sharpe = defi_sharpe,
                 cefi_sharpe = cefi_sharpe))
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  set.seed(42)

  # LP return analysis
  cat("=== LP Return Calculator ===\n")
  for (final_p in c(800, 900, 1000, 1100, 1200, 1500, 2000)) {
    r <- lp_return_calculator(1000, final_p, holding_days=30)
    cat(sprintf("P=%5d: LP=%.2f%% | HODL=%.2f%% | IL=%.2f%% | Fee=%.2f%%\n",
                final_p, r$lp_return_total_pct, r$hodl_return_pct,
                r$il_pct, r$fee_income_pct))
  }

  # Optimal rebalancing
  rebal <- optimal_lp_rebalancing(1000, sigma_daily=0.04, range_width_k=1.2)
  cat(sprintf("\nOptimal rebalancing: every %d days\n", rebal$optimal_days))

  # Token emission
  em <- token_emission_schedule()
  cat(sprintf("\nYear 1 total inflation: %.1f%%\n",
              sum(em$total_daily[1:365]) / em$circulating[365] * 100))

  # TVL growth
  tvl_df <- tvl_growth_model(tvl0=1e8, K=5e9, r=0.004, days=365)
  cat(sprintf("\n1-year TVL growth: $%s -> $%s (%.0fx)\n",
              format(round(tvl_df$tvl[1]/1e6), nsmall=0),
              format(round(tail(tvl_df$tvl,1)/1e6), nsmall=0),
              tail(tvl_df$tvl,1)/tvl_df$tvl[1]))

  # Cross-protocol correlation
  n_protocols <- 8; T_ <- 300
  ret_mat <- matrix(rnorm(T_*n_protocols, 0.001, 0.04), T_, n_protocols)
  colnames(ret_mat) <- c("Uniswap","Aave","Compound","Curve","MakerDAO",
                          "Lido","dYdX","GMX")
  defi_correlation_analysis(ret_mat)
}
