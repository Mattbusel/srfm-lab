# =============================================================================
# defi_study.R
# DeFi Research Study: LP profitability across volatility regimes, IL severity
# by asset pair, MEV impact on retail LPs, yield farming sustainability,
# protocol revenue vs token price decoupling, cross-protocol contagion.
# Pure base R.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. LP PROFITABILITY STUDY ACROSS VOLATILITY REGIMES
# ---------------------------------------------------------------------------

#' Classify volatility regime from rolling realized volatility
classify_vol_regime <- function(realized_vol, n_regimes = 3) {
  quantiles <- quantile(realized_vol[!is.na(realized_vol)],
                        probs = seq(0, 1, length.out = n_regimes + 1))
  regime <- cut(realized_vol, breaks = quantiles, include.lowest = TRUE,
                labels = paste0("Regime_", seq_len(n_regimes)))
  as.integer(regime)
}

#' Simulate LP returns over multiple volatility regimes
lp_profitability_by_regime <- function(n = 1000, seed = 42,
                                        fee_rate = 0.003,
                                        daily_vol_base = 0.05,
                                        vol_regime_multipliers = c(0.5, 1.0, 2.0),
                                        vol_window = 30) {
  set.seed(seed)

  # Generate regime-switching volatility
  regime <- rep(NA, n)
  current_regime <- 2L
  for (t in seq_len(n)) {
    # Regime transition (Markov chain)
    u <- runif(1)
    if (current_regime == 1) {
      current_regime <- if (u < 0.90) 1L else if (u < 0.97) 2L else 3L
    } else if (current_regime == 2) {
      current_regime <- if (u < 0.05) 1L else if (u < 0.90) 2L else 3L
    } else {
      current_regime <- if (u < 0.15) 2L else if (u < 0.95) 3L else 1L
    }
    regime[t] <- current_regime
  }

  # Returns in each regime
  daily_returns <- rnorm(n, 0, daily_vol_base * vol_regime_multipliers[regime])

  # Rolling realized vol
  rv <- rep(NA, n)
  for (i in vol_window:n) {
    rv[i] <- sd(daily_returns[(i - vol_window + 1):i])
  }

  # LP metrics at each time step
  prices <- 1000 * exp(cumsum(daily_returns))
  lp_il  <- numeric(n)
  lp_fee <- numeric(n)
  for (t in 2:n) {
    pr <- prices[t] / prices[t-1]
    lp_il[t]  <- 2 * sqrt(pr) / (1 + pr) - 1  # Daily IL approximation
    lp_fee[t] <- fee_rate * daily_vol_base * vol_regime_multipliers[regime[t]] * 0.1
  }
  lp_net <- lp_il + lp_fee

  # Results by regime
  regime_stats <- lapply(seq_along(vol_regime_multipliers), function(r) {
    mask <- regime == r & !is.na(lp_net)
    data.frame(
      regime = r,
      vol_level = c("Low", "Medium", "High")[r],
      n_days = sum(mask),
      mean_daily_il_bps = mean(lp_il[mask], na.rm=TRUE) * 10000,
      mean_daily_fee_bps = mean(lp_fee[mask], na.rm=TRUE) * 10000,
      mean_net_pnl_bps = mean(lp_net[mask], na.rm=TRUE) * 10000,
      pct_profitable_days = mean(lp_net[mask] > 0, na.rm=TRUE) * 100
    )
  })

  regime_df <- do.call(rbind, regime_stats)

  cat("=== LP Profitability by Volatility Regime ===\n")
  print(regime_df)

  invisible(list(
    returns = daily_returns, prices = prices,
    regime = regime, rv = rv,
    lp_il = lp_il, lp_fee = lp_fee, lp_net = lp_net,
    regime_stats = regime_df
  ))
}

# ---------------------------------------------------------------------------
# 2. IMPERMANENT LOSS SEVERITY BY ASSET PAIR
# ---------------------------------------------------------------------------

#' Compare IL across different asset pair types
#' Stable-stable: very low IL (BTC-ETH correlated)
#' Volatile-stable: high IL risk (ETH-USDC)
il_by_asset_pair <- function(n = 500, seed = 7) {
  set.seed(seed)

  pair_specs <- list(
    stablecoin_pair = list(
      name = "USDC/USDT (stable-stable)",
      sigma1 = 0.001, sigma2 = 0.001, rho = 0.95
    ),
    btc_eth = list(
      name = "BTC/ETH (correlated volatile)",
      sigma1 = 0.04, sigma2 = 0.045, rho = 0.75
    ),
    eth_stable = list(
      name = "ETH/USDC (volatile-stable)",
      sigma1 = 0.045, sigma2 = 0.0005, rho = 0.0
    ),
    altcoin_stable = list(
      name = "SOL/USDC (high vol-stable)",
      sigma1 = 0.06, sigma2 = 0.0005, rho = 0.0
    ),
    btc_altcoin = list(
      name = "BTC/SOL (diff betas)",
      sigma1 = 0.04, sigma2 = 0.06, rho = 0.55
    )
  )

  results <- lapply(pair_specs, function(spec) {
    # Simulate correlated returns
    Z1 <- rnorm(n); Z2 <- rnorm(n)
    r1 <- spec$sigma1 * Z1
    r2 <- spec$sigma2 * (spec$rho * Z1 + sqrt(1 - spec$rho^2) * Z2)

    # Price ratio dynamics: X/Y price = exp(r1 - r2)
    # IL = 2*sqrt(P_t/P_0) / (1 + P_t/P_0) - 1
    price_ratio <- exp(cumsum(r1 - r2))

    il_path <- 2 * sqrt(price_ratio) / (1 + price_ratio) - 1

    # Distribution of IL at multiple horizons
    horizons <- c(1, 7, 30, 90, 180, 365)
    il_at_horizons <- sapply(horizons, function(h) {
      if (h <= n) il_path[h] else NA
    })

    # Rolling max IL drawdown
    max_il <- min(il_path)

    data.frame(
      pair = spec$name,
      annualized_vol_asset1 = spec$sigma1 * sqrt(252) * 100,
      annualized_vol_asset2 = spec$sigma2 * sqrt(252) * 100,
      correlation = spec$rho,
      mean_1d_il_bps = mean(il_path, na.rm=TRUE) * 10000,
      max_il_pct     = max_il * 100,
      il_7d_pct      = il_at_horizons[2] * 100,
      il_30d_pct     = il_at_horizons[3] * 100,
      stringsAsFactors = FALSE
    )
  })

  df <- do.call(rbind, results)
  cat("=== Impermanent Loss Severity by Asset Pair ===\n")
  print(df[order(df$max_il_pct), ])
  invisible(df)
}

# ---------------------------------------------------------------------------
# 3. MEV IMPACT ON RETAIL LIQUIDITY PROVIDERS
# ---------------------------------------------------------------------------

#' Quantify how much MEV extracts value from LP positions
#' MEV sources: arbitrage, sandwich attacks, JIT liquidity
mev_impact_study <- function(pool_size = 1e7, fee_rate = 0.003,
                              n_days = 252, daily_volume_frac = 0.10,
                              mev_fraction = 0.05,  # 5% of fees taken by MEV
                              retail_lp_share = 0.30, seed = 42) {
  set.seed(seed)

  # Simulate pool dynamics
  daily_volume <- pool_size * daily_volume_frac
  daily_fees   <- daily_volume * fee_rate

  # MEV components:
  # 1. Arbitrage: LP sells cheap, arb bot profits = LP's loss
  # 2. Sandwich attacks: victim pays more, LP gets sandwich tx fees but at cost to ecosystem
  # 3. JIT liquidity: bots provide liquidity just-in-time for big trades, stealing retail LP fees

  arb_mev_per_day     <- daily_fees * mev_fraction * 0.40  # 40% from arb
  sandwich_mev_per_day <- daily_fees * mev_fraction * 0.35  # 35% from sandwich
  jit_mev_per_day     <- daily_fees * mev_fraction * 0.25  # 25% from JIT

  total_mev_per_day   <- arb_mev_per_day + sandwich_mev_per_day + jit_mev_per_day

  # Retail LP impact: MEV reduces effective fees
  retail_fee_without_mev <- daily_fees * retail_lp_share
  retail_fee_with_mev    <- retail_fee_without_mev - total_mev_per_day * retail_lp_share

  # Annual cumulative
  annual_stats <- data.frame(
    metric = c("Total pool fees/year", "Total MEV extracted/year",
               "Retail LP fees (no MEV)", "Retail LP fees (with MEV)",
               "MEV impact on retail (%)"),
    value_usd = c(daily_fees * n_days, total_mev_per_day * n_days,
                   retail_fee_without_mev * n_days,
                   retail_fee_with_mev * n_days,
                   -(total_mev_per_day / daily_fees) * 100)
  )

  cat("=== MEV Impact on Retail LPs ===\n")
  cat(sprintf("Pool size: $%s | Fee rate: %.2f%%\n",
              format(pool_size, big.mark=","), fee_rate*100))
  cat(sprintf("MEV fraction of fees: %.1f%%\n", mev_fraction*100))
  for (i in seq_len(nrow(annual_stats))) {
    if (i < nrow(annual_stats)) {
      cat(sprintf("  %-35s $%s\n", annual_stats$metric[i],
                  format(round(annual_stats$value_usd[i]), big.mark=",")))
    } else {
      cat(sprintf("  %-35s %.2f%%\n", annual_stats$metric[i],
                  annual_stats$value_usd[i]))
    }
  }

  # Monte Carlo simulation of actual LP returns
  lp_returns_daily <- rnorm(n_days,
                             mean = retail_fee_with_mev / pool_size,
                             sd   = 0.001)
  # Add IL
  price_changes <- rnorm(n_days, 0, 0.04)
  price_ratio_cumulative <- exp(cumsum(price_changes))
  il_daily <- 2 * sqrt(price_ratio_cumulative) / (1 + price_ratio_cumulative) - 1

  lp_net <- lp_returns_daily + c(0, diff(il_daily))
  cum_lp_return <- (prod(1 + lp_net) - 1) * 100

  cat(sprintf("\n  Simulated annual LP return: %.2f%%\n", cum_lp_return))

  invisible(list(annual_stats = annual_stats,
                 mev_per_day = total_mev_per_day,
                 retail_fee_loss_pct = total_mev_per_day / daily_fees * 100))
}

# ---------------------------------------------------------------------------
# 4. YIELD FARMING SUSTAINABILITY ANALYSIS
# ---------------------------------------------------------------------------

#' Study when yield farming becomes unsustainable
#' Unsustainable when emission value > protocol fee revenue
yield_sustainability_analysis <- function(
  initial_tvl    = 1e9,
  initial_token_price = 1.0,
  token_supply   = 1e9,
  daily_emission_rate = 0.0002,  # 0.02% of supply per day
  fee_rate       = 0.003,
  daily_vol_frac = 0.10,
  tvl_elasticity = 0.5,   # TVL responds to yield: 1% higher yield -> 0.5% more TVL
  days = 365,
  seed = 42
) {
  set.seed(seed)
  T_ <- days

  tvl         <- numeric(T_); tvl[1] <- initial_tvl
  token_price <- numeric(T_); token_price[1] <- initial_token_price
  daily_emission_usd <- numeric(T_)
  daily_fee_rev      <- numeric(T_)
  yield_apy          <- numeric(T_)
  sustainable        <- logical(T_)

  for (t in 2:T_) {
    # Fee revenue
    daily_fee_rev[t] <- tvl[t-1] * daily_vol_frac * fee_rate

    # Emission in USD
    emit_tokens <- token_supply * daily_emission_rate
    daily_emission_usd[t] <- emit_tokens * token_price[t-1]

    # Current APY (emission + fees)
    yield_apy[t] <- (daily_emission_usd[t] + daily_fee_rev[t]) / tvl[t-1] * 365

    # Sustainability: fees cover emissions
    sustainable[t] <- daily_fee_rev[t] >= daily_emission_usd[t]

    # Token price: sell pressure from emissions
    emission_sell <- daily_emission_usd[t] * 0.70  # 70% sell immediately
    buyback_from_fees <- daily_fee_rev[t] * 0.20   # 20% of fees buy back tokens
    net_flow <- buyback_from_fees - emission_sell
    price_change <- net_flow / (tvl[t-1] * 0.10 + 1)  # Market impact
    token_price[t] <- max(1e-6, token_price[t-1] * exp(price_change + rnorm(1, 0, 0.02)))

    # TVL responds to yield
    tvl_change <- tvl[t-1] * (yield_apy[t] * tvl_elasticity / 365 - 0.001)
    tvl[t] <- max(1e5, tvl[t-1] + tvl_change + rnorm(1, 0, tvl[t-1] * 0.01))
  }

  days_sustainable <- sum(sustainable, na.rm=TRUE)
  first_unsustainable <- min(which(!sustainable & !is.na(sustainable)))

  cat("=== Yield Farming Sustainability Study ===\n")
  cat(sprintf("Initial TVL: $%s | Initial token price: $%.2f\n",
              format(initial_tvl, big.mark=","), initial_token_price))
  cat(sprintf("Days sustainable: %d/%d (%.1f%%)\n",
              days_sustainable, T_, days_sustainable/T_*100))
  cat(sprintf("First unsustainable day: %d\n", first_unsustainable))
  cat(sprintf("Final token price: $%.4f (%.1f%% of initial)\n",
              tail(token_price, 1), tail(token_price, 1)/initial_token_price*100))
  cat(sprintf("Final TVL: $%s\n", format(round(tail(tvl, 1)), big.mark=",")))

  data.frame(
    day = seq_len(T_),
    tvl = tvl, token_price = token_price,
    daily_fee_rev = daily_fee_rev,
    daily_emission_usd = daily_emission_usd,
    yield_apy = yield_apy,
    sustainable = sustainable
  )
}

# ---------------------------------------------------------------------------
# 5. PROTOCOL REVENUE VS TOKEN PRICE DECOUPLING
# ---------------------------------------------------------------------------

#' Empirical study of revenue-price disconnect in DeFi protocols
protocol_revenue_price_study <- function(n = 500, seed = 99) {
  set.seed(seed)

  # Simulate protocol with growing revenue but declining token price
  # (common in 2022-2023: protocols earn fees but tokens dump)

  # Revenue: grows steadily (usage grows)
  revenue_growth_daily <- 0.002 + rnorm(n, 0, 0.01)
  revenue <- 1e6 * exp(cumsum(revenue_growth_daily))

  # Token price: influenced by both revenue AND sentiment/speculation
  sentiment <- cumsum(rnorm(n, -0.001, 0.03))  # Declining sentiment
  revenue_component <- 0.3 * revenue_growth_daily  # Revenue drives price 30%
  spec_component    <- 0.7 * rnorm(n, -0.001, 0.03)  # Speculation 70%

  token_returns <- revenue_component + spec_component
  token_price   <- 10 * exp(cumsum(token_returns))

  # P/S ratio (Price-to-Sales = FDV / Annual Revenue)
  token_supply <- 1e9
  fdv          <- token_price * token_supply
  ann_revenue  <- revenue * 365
  ps_ratio     <- fdv / ann_revenue

  # Rolling correlation: revenue growth vs token price change
  window <- 30
  rolling_corr <- rep(NA, n)
  for (i in window:n) {
    rc <- revenue_growth_daily[(i-window+1):i]
    tp <- token_returns[(i-window+1):i]
    rolling_corr[i] <- cor(rc, tp)
  }

  # Periods of decoupling (correlation breaks down)
  decoupled <- rolling_corr < 0.2 & !is.na(rolling_corr)

  cat("=== Protocol Revenue vs Token Price Study ===\n")
  cat(sprintf("Overall revenue-price corr:  %.3f\n",
              cor(revenue_growth_daily, token_returns, use="complete.obs")))
  cat(sprintf("Periods of decoupling (<0.2 corr): %d days (%.1f%%)\n",
              sum(decoupled), mean(decoupled)*100))
  cat(sprintf("Final P/S ratio: %.1fx\n", tail(ps_ratio, 1)))
  cat(sprintf("P/S at peak: %.1fx | at trough: %.1fx\n",
              max(ps_ratio), min(ps_ratio)))

  data.frame(
    t = seq_len(n),
    revenue = revenue, token_price = token_price,
    fdv = fdv, ps_ratio = ps_ratio,
    rolling_corr = rolling_corr,
    decoupled = decoupled
  )
}

# ---------------------------------------------------------------------------
# 6. CROSS-PROTOCOL CONTAGION CASE STUDIES
# ---------------------------------------------------------------------------

#' Simulate LUNA/UST-style collapse with contagion
simulate_luna_contagion <- function(n_protocols = 10, n_assets = 5,
                                     seed = 42) {
  set.seed(seed)

  # Asset holdings matrix (protocols' exposure to each token)
  holdings <- matrix(runif(n_protocols * n_assets), n_protocols, n_assets)
  holdings <- holdings / rowSums(holdings)  # Normalize to portfolio weights
  rownames(holdings) <- paste0("Protocol_", seq_len(n_protocols))
  colnames(holdings) <- c("BTC", "ETH", "LUNA", "UST", "Other")

  # LUNA is 15-20% of most DeFi protocols
  holdings[, "LUNA"] <- runif(n_protocols, 0.10, 0.25)
  holdings[, "UST"]  <- runif(n_protocols, 0.05, 0.15)
  # Renormalize
  holdings <- holdings / rowSums(holdings)

  # LUNA/UST collapse scenario
  shock <- c(BTC = -0.30, ETH = -0.35, LUNA = -0.999, UST = -0.90, Other = -0.20)

  # First-round losses
  first_round <- as.vector(holdings %*% shock)

  # Second-round: protocols with >30% loss must liquidate, causing contagion
  distress_threshold <- -0.30
  distressed <- first_round < distress_threshold

  # Forced selling creates additional price pressure
  forced_sell_btc <- sum(holdings[distressed, "BTC"]) * abs(first_round[distressed])
  forced_sell_eth <- sum(holdings[distressed, "ETH"]) * abs(first_round[distressed])

  # Additional price impact
  btc_impact <- -forced_sell_btc * 0.10  # Market impact
  eth_impact <- -forced_sell_eth * 0.12

  second_shock <- c(BTC = btc_impact, ETH = eth_impact, LUNA = 0, UST = 0, Other = 0)
  second_round <- as.vector(holdings %*% second_shock)

  total_loss <- first_round + second_round

  cat("=== LUNA/UST Contagion Simulation ===\n")
  cat(sprintf("Protocols directly distressed: %d/%d\n",
              sum(distressed), n_protocols))
  cat(sprintf("Average first-round loss: %.1f%%\n", mean(first_round)*100))
  cat(sprintf("Average second-round loss: %.1f%%\n", mean(second_round)*100))
  cat(sprintf("Average total loss: %.1f%%\n", mean(total_loss)*100))

  # Protocols that survive
  survivors <- total_loss > -0.50
  cat(sprintf("Protocols surviving (>-50%%): %d/%d\n",
              sum(survivors), n_protocols))

  data.frame(
    protocol = rownames(holdings),
    first_round_pct = first_round * 100,
    second_round_pct = second_round * 100,
    total_loss_pct = total_loss * 100,
    distressed = distressed,
    survived = survivors,
    luna_exposure = holdings[, "LUNA"] * 100,
    ust_exposure = holdings[, "UST"] * 100
  )
}

#' FTX collapse contagion: exchange counterparty risk
simulate_ftx_contagion <- function(n_protocols = 15, seed = 77) {
  set.seed(seed)

  # Each protocol has some % of assets on FTX (exchange counterparty risk)
  ftx_exposure <- runif(n_protocols, 0.01, 0.40)

  # FTX goes to zero
  ftx_loss <- ftx_exposure * (-1.0)  # 100% loss on FTX exposure

  # Correlated assets (FTT, SOL, SRM) crash
  correlated_asset_exposure <- runif(n_protocols, 0.00, 0.15)
  correlated_shock <- -0.60  # FTT, SOL crashed ~60%
  correlated_loss  <- correlated_asset_exposure * correlated_shock

  # Contagion: confidence crisis causes withdrawals
  withdrawal_pressure <- 0.10 * (ftx_loss + correlated_loss)

  total_protocol_loss <- ftx_loss + correlated_loss + withdrawal_pressure

  cat("=== FTX Collapse Contagion ===\n")
  cat(sprintf("Average FTX exposure: %.1f%%\n", mean(ftx_exposure)*100))
  cat(sprintf("Average total protocol loss: %.1f%%\n",
              mean(total_protocol_loss)*100))
  cat(sprintf("Protocols >30%% impacted: %d/%d\n",
              sum(total_protocol_loss < -0.30), n_protocols))

  data.frame(
    protocol = paste0("P", seq_len(n_protocols)),
    ftx_exposure_pct = ftx_exposure * 100,
    ftx_loss_pct = ftx_loss * 100,
    correlated_loss_pct = correlated_loss * 100,
    total_loss_pct = total_protocol_loss * 100
  )
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  cat("\n### DeFi Study: Full Research Pipeline ###\n\n")

  # 1. LP profitability by regime
  lp_study <- lp_profitability_by_regime(n=1000, fee_rate=0.003)
  cat("\n")

  # 2. IL by asset pair
  il_study <- il_by_asset_pair(n=500)
  cat("\n")

  # 3. MEV impact
  mev_study <- mev_impact_study(pool_size=5e7, fee_rate=0.003, mev_fraction=0.08)
  cat("\n")

  # 4. Yield farming sustainability
  yf_study <- yield_sustainability_analysis(
    initial_tvl = 5e8,
    initial_token_price = 2.0,
    daily_emission_rate = 0.0003,
    days = 365
  )
  cat("\n")

  # 5. Revenue vs token price
  rtp_study <- protocol_revenue_price_study(n=500)
  cat("\n")

  # 6. Contagion case studies
  luna_study <- simulate_luna_contagion(n_protocols=12)
  cat("\n")
  ftx_study  <- simulate_ftx_contagion(n_protocols=15)
}
