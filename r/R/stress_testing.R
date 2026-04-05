# =============================================================================
# stress_testing.R
# Stress Testing Framework: historical scenario replay, hypothetical shocks,
# correlation stress, liquidity stress, model stress, reverse stress testing,
# combined multi-factor shocks, stressed VaR and ES.
# Pure base R.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. HISTORICAL SCENARIO REPLAY
# ---------------------------------------------------------------------------

#' Historical stress event library (crypto-specific)
historical_stress_scenarios <- function() {
  list(
    covid_march_2020 = list(
      name = "COVID Crash (Mar 2020)",
      btc_return = -0.50,
      eth_return = -0.60,
      altcoin_return = -0.65,
      stablecoin_depeg = 0.00,
      duration_days = 3,
      recovery_days = 60,
      volatility_spike = 3.5  # Multiplier on normal vol
    ),
    luna_collapse_2022 = list(
      name = "LUNA/UST Collapse (May 2022)",
      btc_return = -0.30,
      eth_return = -0.35,
      altcoin_return = -0.60,
      stablecoin_depeg = -0.30,  # UST went to zero
      duration_days = 5,
      recovery_days = 180,
      volatility_spike = 4.0
    ),
    ftx_collapse_2022 = list(
      name = "FTX Collapse (Nov 2022)",
      btc_return = -0.25,
      eth_return = -0.28,
      altcoin_return = -0.45,
      stablecoin_depeg = -0.05,  # USDT briefly depegged
      duration_days = 3,
      recovery_days = 90,
      volatility_spike = 2.8
    ),
    bear_2022_june = list(
      name = "Bear Market Jun 2022",
      btc_return = -0.40,
      eth_return = -0.45,
      altcoin_return = -0.65,
      stablecoin_depeg = 0.00,
      duration_days = 7,
      recovery_days = 365,
      volatility_spike = 2.5
    ),
    apr_2026_crash = list(
      name = "Apr 2026 Macro Shock",
      btc_return = -0.22,
      eth_return = -0.28,
      altcoin_return = -0.35,
      stablecoin_depeg = -0.01,
      duration_days = 2,
      recovery_days = 30,
      volatility_spike = 2.2
    )
  )
}

#' Replay historical scenario on current portfolio
replay_historical_scenario <- function(portfolio_weights, asset_names,
                                        scenario, correlation_matrix = NULL) {
  n <- length(portfolio_weights)

  # Map scenario returns to assets
  returns <- numeric(n)
  for (i in seq_len(n)) {
    asset <- tolower(asset_names[i])
    if (grepl("btc|bitcoin", asset)) {
      returns[i] <- scenario$btc_return
    } else if (grepl("eth|ethereum", asset)) {
      returns[i] <- scenario$eth_return
    } else if (grepl("usdc|usdt|dai|stable", asset)) {
      returns[i] <- scenario$stablecoin_depeg
    } else {
      # Default: altcoin return
      returns[i] <- scenario$altcoin_return
    }
  }

  # Portfolio return
  port_return <- sum(portfolio_weights * returns)

  # If correlation matrix provided, simulate correlated shock
  if (!is.null(correlation_matrix)) {
    vols <- abs(returns) / scenario$volatility_spike
    cov_stress <- diag(vols) %*% correlation_matrix %*% diag(vols)
    # Portfolio vol during stress
    port_vol_stress <- sqrt(as.numeric(t(portfolio_weights) %*% cov_stress %*% portfolio_weights))
    port_var_stress <- qnorm(0.01) * port_vol_stress
  } else {
    port_vol_stress <- NA; port_var_stress <- NA
  }

  list(
    scenario_name   = scenario$name,
    asset_returns   = returns,
    portfolio_return_pct = port_return * 100,
    vol_spike_factor = scenario$volatility_spike,
    duration_days   = scenario$duration_days,
    recovery_days   = scenario$recovery_days,
    stressed_var_pct = if (!is.na(port_var_stress)) port_var_stress * 100 else NA
  )
}

#' Run all historical scenarios
run_historical_stress_tests <- function(portfolio_weights, asset_names,
                                         correlation_matrix = NULL) {
  scenarios <- historical_stress_scenarios()

  results <- lapply(scenarios, function(sc) {
    replay_historical_scenario(portfolio_weights, asset_names, sc, correlation_matrix)
  })

  cat("╔══════════════════════════════════════════════════════╗\n")
  cat("║          Historical Stress Test Results               ║\n")
  cat("╚══════════════════════════════════════════════════════╝\n\n")
  cat(sprintf("%-35s | %10s | %8s\n", "Scenario", "Port PnL%", "Recovery"))
  cat(paste(rep("-", 60), collapse=""), "\n")

  for (r in results) {
    cat(sprintf("%-35s | %9.2f%% | %7d d\n",
                r$scenario_name, r$portfolio_return_pct,
                r$recovery_days))
  }

  invisible(results)
}

# ---------------------------------------------------------------------------
# 2. HYPOTHETICAL SCENARIOS
# ---------------------------------------------------------------------------

#' BTC -50% shock: what happens to all positions
btc_crash_scenario <- function(portfolio_weights, asset_names,
                                 btc_shock = -0.50,
                                 beta_matrix = NULL) {
  n <- length(portfolio_weights)

  # Default betas to BTC
  if (is.null(beta_matrix)) {
    btc_betas <- sapply(tolower(asset_names), function(a) {
      if (grepl("btc|bitcoin", a)) 1.0
      else if (grepl("eth|ethereum", a)) 0.85
      else if (grepl("stable|usd", a)) 0.02
      else runif(1, 0.6, 1.3)  # Generic altcoin
    })
  } else {
    btc_betas <- beta_matrix
  }

  # Asset returns = beta * btc_shock + idiosyncratic (assume 0 for pure scenario)
  asset_returns <- btc_betas * btc_shock

  port_return <- sum(portfolio_weights * asset_returns)

  list(
    btc_shock_pct   = btc_shock * 100,
    btc_betas       = btc_betas,
    asset_returns_pct = asset_returns * 100,
    portfolio_return_pct = port_return * 100,
    worst_asset     = asset_names[which.min(asset_returns)],
    best_asset      = asset_names[which.max(asset_returns)]
  )
}

#' Multi-asset hypothetical scenario specification
hypothetical_scenario <- function(portfolio_weights, asset_returns_vector,
                                   scenario_name = "Custom Scenario") {
  port_return <- sum(portfolio_weights * asset_returns_vector)

  # P&L decomposition by asset
  pnl_by_asset <- portfolio_weights * asset_returns_vector

  cat(sprintf("=== %s ===\n", scenario_name))
  cat(sprintf("Portfolio Return: %+.2f%%\n", port_return * 100))
  cat("Asset Contributions:\n")
  for (i in seq_along(pnl_by_asset)) {
    cat(sprintf("  Asset %d: weight=%.2f, return=%+.2f%%, contrib=%+.4f%%\n",
                i, portfolio_weights[i], asset_returns_vector[i]*100,
                pnl_by_asset[i]*100))
  }

  list(
    scenario = scenario_name,
    portfolio_return = port_return,
    pnl_by_asset = pnl_by_asset
  )
}

# ---------------------------------------------------------------------------
# 3. CORRELATION STRESS
# ---------------------------------------------------------------------------

#' Stress correlation matrix: push all pairwise correlations toward target_corr
#' Simulates "all correlations go to 1 in a crisis"
stress_correlations <- function(base_corr_matrix, target_corr = 0.95,
                                  stress_intensity = 0.80) {
  n <- nrow(base_corr_matrix)

  # Linear interpolation: R_stressed = (1-alpha)*R_base + alpha*R_target
  R_target <- matrix(target_corr, n, n)
  diag(R_target) <- 1.0

  R_stressed <- (1 - stress_intensity) * base_corr_matrix +
                 stress_intensity * R_target

  # Ensure valid correlation matrix (positive semi-definite)
  eig <- eigen(R_stressed)
  eig$values <- pmax(eig$values, 1e-6)
  R_stressed <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  D <- diag(1 / sqrt(diag(R_stressed)))
  R_stressed <- D %*% R_stressed %*% D
  diag(R_stressed) <- 1.0

  list(
    stressed_corr = R_stressed,
    avg_corr_before = mean(base_corr_matrix[upper.tri(base_corr_matrix)]),
    avg_corr_after  = mean(R_stressed[upper.tri(R_stressed)])
  )
}

#' Portfolio VaR under stressed correlation
stressed_portfolio_var <- function(weights, base_corr, individual_vols,
                                    target_corr = 0.95, stress_intensity = 0.80,
                                    q = 0.01) {
  sc <- stress_correlations(base_corr, target_corr, stress_intensity)

  # Base portfolio vol
  cov_base    <- diag(individual_vols) %*% base_corr %*% diag(individual_vols)
  port_vol_base <- sqrt(as.numeric(t(weights) %*% cov_base %*% weights))

  # Stressed portfolio vol
  cov_stressed  <- diag(individual_vols) %*% sc$stressed_corr %*% diag(individual_vols)
  port_vol_stress <- sqrt(as.numeric(t(weights) %*% cov_stressed %*% weights))

  # VaR at q
  var_base   <- qnorm(q) * port_vol_base
  var_stress <- qnorm(q) * port_vol_stress

  cat("=== Correlation Stress Test ===\n")
  cat(sprintf("Avg correlation before stress: %.3f\n", sc$avg_corr_before))
  cat(sprintf("Avg correlation after stress:  %.3f\n", sc$avg_corr_after))
  cat(sprintf("Portfolio vol: %.4f%% -> %.4f%% (+%.1f%%)\n",
              port_vol_base*100, port_vol_stress*100,
              (port_vol_stress/port_vol_base - 1)*100))
  cat(sprintf("VaR(1%%): %.4f%% -> %.4f%%\n", var_base*100, var_stress*100))

  list(
    var_base     = var_base,
    var_stressed = var_stress,
    var_increase_pct = (var_stress / var_base - 1) * 100,
    stressed_corr = sc$stressed_corr
  )
}

# ---------------------------------------------------------------------------
# 4. LIQUIDITY STRESS
# ---------------------------------------------------------------------------

#' Liquidity stress: widen all bid-ask spreads and increase market impact
liquidity_stressed_cost <- function(trade_sizes_usd, normal_spreads_bps,
                                     spread_multiplier = 5.0,
                                     impact_multiplier = 3.0,
                                     daily_adv_usd = 1e8,
                                     daily_vol = 0.04) {
  n <- length(trade_sizes_usd)

  # Normal costs
  normal_spread_cost <- trade_sizes_usd * normal_spreads_bps / 10000
  normal_impact <- 0.1 * daily_vol * sqrt(trade_sizes_usd / daily_adv_usd) *
                   trade_sizes_usd

  # Stressed costs
  stressed_spread_cost <- normal_spread_cost * spread_multiplier
  stressed_impact      <- normal_impact * impact_multiplier

  # Total
  normal_total  <- normal_spread_cost + normal_impact
  stressed_total <- stressed_spread_cost + stressed_impact

  data.frame(
    trade_size   = trade_sizes_usd,
    normal_spread_bps   = normal_spreads_bps,
    stressed_spread_bps = normal_spreads_bps * spread_multiplier,
    normal_cost_pct     = normal_total / trade_sizes_usd * 100,
    stressed_cost_pct   = stressed_total / trade_sizes_usd * 100,
    cost_increase_pct   = (stressed_total / normal_total - 1) * 100
  )
}

#' Time to liquidate position under stressed market conditions
liquidation_time_estimate <- function(position_size_usd,
                                       daily_adv_usd,
                                       max_participation_rate = 0.05,
                                       stress_adv_fraction = 0.30) {
  # Under stress: ADV drops to 30% of normal
  stressed_adv <- daily_adv_usd * stress_adv_fraction
  daily_liquidation <- stressed_adv * max_participation_rate

  days_to_liquidate <- position_size_usd / daily_liquidation
  hours_to_liquidate <- days_to_liquidate * 24

  list(
    normal_days  = position_size_usd / (daily_adv_usd * max_participation_rate),
    stressed_days = days_to_liquidate,
    stressed_hours = hours_to_liquidate,
    participates_pct_adv = max_participation_rate * 100
  )
}

# ---------------------------------------------------------------------------
# 5. MODEL STRESS: GARCH VOL UNDERESTIMATION
# ---------------------------------------------------------------------------

#' Scenario: GARCH model underestimates realized volatility
#' How much does risk look better than reality?
garch_underestimation_stress <- function(portfolio_returns,
                                          garch_forecast_vol,
                                          realized_vol = NULL,
                                          underest_factor = 2.0,
                                          q = 0.01) {
  T_ <- length(portfolio_returns)

  # Historical VaR
  hist_var <- quantile(portfolio_returns, q)

  # GARCH VaR (parametric)
  garch_var <- qnorm(q) * garch_forecast_vol

  # Stressed GARCH VaR: model underestimates by factor
  stressed_var <- garch_var * underest_factor

  # If realized vol provided, compare
  if (!is.null(realized_vol)) {
    realized_var <- qnorm(q) * realized_vol
    model_error  <- (garch_forecast_vol / realized_vol - 1) * 100
  } else {
    realized_var <- NA; model_error <- NA
  }

  cat("=== GARCH Model Stress ===\n")
  cat(sprintf("Historical VaR(1%%):          %.4f%%\n", hist_var*100))
  cat(sprintf("GARCH VaR(1%%):               %.4f%%\n", garch_var*100))
  cat(sprintf("Stressed VaR (%.0fx underest): %.4f%%\n",
              underest_factor, stressed_var*100))
  if (!is.na(realized_var)) {
    cat(sprintf("Realized VaR:                %.4f%%\n", realized_var*100))
    cat(sprintf("Model error:                 %+.1f%%\n", model_error))
  }

  list(
    hist_var    = hist_var,
    garch_var   = garch_var,
    stressed_var = stressed_var,
    var_underest_pct = (stressed_var / garch_var - 1) * 100
  )
}

# ---------------------------------------------------------------------------
# 6. REVERSE STRESS TEST
# ---------------------------------------------------------------------------
# Find the scenario that causes exactly a target loss (e.g., 30% drawdown)
# Work backwards: given loss target, find what combination of shocks achieves it

#' Find portfolio-killing scenario via grid search
reverse_stress_test <- function(portfolio_weights,
                                  cov_matrix,
                                  target_loss_pct = -0.30,
                                  n_scenarios = 10000,
                                  seed = 42) {
  set.seed(seed)
  n <- length(portfolio_weights)
  vols <- sqrt(diag(cov_matrix))

  # Generate random scenarios (multivariate normal shocks)
  L <- tryCatch(t(chol(cov_matrix + diag(1e-8, n))),
                error = function(e) diag(vols))

  # Scale shocks to cover range of plausible events
  Z <- matrix(rnorm(n_scenarios * n), n_scenarios, n)
  # Scale to look for extreme events
  scale_factors <- seq(0.5, 10, length.out = 20)

  best_scenario <- NULL
  best_dist <- Inf

  for (sf in scale_factors) {
    shocks <- (Z * sf) %*% t(L)  # Correlated shocks
    port_losses <- shocks %*% portfolio_weights

    # Find scenarios closest to target_loss_pct
    dists <- abs(port_losses - target_loss_pct)
    best_idx <- which.min(dists)

    if (dists[best_idx] < best_dist) {
      best_dist <- dists[best_idx]
      best_scenario <- list(
        shocks = shocks[best_idx, ],
        portfolio_loss = port_losses[best_idx],
        scale_factor = sf
      )
    }
  }

  cat("=== Reverse Stress Test ===\n")
  cat(sprintf("Target loss: %.1f%%\n", target_loss_pct * 100))
  cat(sprintf("Found scenario with loss: %.2f%%\n",
              best_scenario$portfolio_loss * 100))
  cat("Asset shocks:\n")
  for (i in seq_len(n)) {
    cat(sprintf("  Asset %d: %+.3f%%\n", i, best_scenario$shocks[i] * 100))
  }

  list(
    target_loss = target_loss_pct,
    achieved_loss = best_scenario$portfolio_loss,
    asset_shocks = best_scenario$shocks,
    probability = pnorm(target_loss_pct / sqrt(as.numeric(t(portfolio_weights) %*% cov_matrix %*% portfolio_weights)))
  )
}

# ---------------------------------------------------------------------------
# 7. COMBINED STRESS: MULTI-FACTOR SIMULTANEOUS SHOCK
# ---------------------------------------------------------------------------

#' Multi-factor stress: simultaneous shocks to price, vol, correlation, liquidity
combined_stress_test <- function(portfolio_weights, asset_names,
                                  cov_matrix,
                                  price_shocks,         # Vector of returns
                                  vol_multiplier = 2.0,
                                  corr_target = 0.90,
                                  spread_multiplier = 3.0,
                                  trade_sizes_usd = NULL) {
  n <- length(portfolio_weights)
  vols <- sqrt(diag(cov_matrix))
  base_corr <- cov2cor(cov_matrix)

  # 1. Price shock P&L
  price_pnl <- sum(portfolio_weights * price_shocks)

  # 2. Vol stress: recalculate VaR with stressed vol
  stressed_vols <- vols * vol_multiplier
  sc_result <- stress_correlations(base_corr, corr_target)
  stressed_cov <- diag(stressed_vols) %*% sc_result$stressed_corr %*% diag(stressed_vols)
  port_vol_stress <- sqrt(as.numeric(t(portfolio_weights) %*% stressed_cov %*% portfolio_weights))
  stressed_var99 <- qnorm(0.01) * port_vol_stress

  # 3. Liquidity cost
  if (!is.null(trade_sizes_usd)) {
    liq_cost <- sum(trade_sizes_usd * 0.003 * spread_multiplier)  # Simplified
    liq_cost_pct <- liq_cost / sum(trade_sizes_usd) * 100
  } else {
    liq_cost_pct <- 0.5 * spread_multiplier  # Default assumption
  }

  total_impact_pct <- price_pnl * 100 - liq_cost_pct

  cat("╔══════════════════════════════════════════════════╗\n")
  cat("║           Combined Multi-Factor Stress            ║\n")
  cat("╚══════════════════════════════════════════════════╝\n\n")
  cat(sprintf("  Price shock P&L:        %+.2f%%\n", price_pnl * 100))
  cat(sprintf("  Liquidity cost:         -%.2f%%\n", liq_cost_pct))
  cat(sprintf("  Total P&L impact:       %+.2f%%\n", total_impact_pct))
  cat(sprintf("  Vol multiplier:         %.1fx\n", vol_multiplier))
  cat(sprintf("  Corr stress target:     %.2f\n", corr_target))
  cat(sprintf("  Stressed portfolio vol: %.4f%%/day\n", port_vol_stress*100))
  cat(sprintf("  Stressed VaR (99%%):     %.4f%%\n", stressed_var99*100))

  list(
    price_pnl_pct = price_pnl * 100,
    liq_cost_pct  = liq_cost_pct,
    total_pnl_pct = total_impact_pct,
    stressed_var99 = stressed_var99,
    stressed_vol   = port_vol_stress
  )
}

# ---------------------------------------------------------------------------
# 8. STRESSED VaR AND STRESSED ES
# ---------------------------------------------------------------------------

#' Basel III Stressed VaR: compute VaR using 12-month stress window
#' Requires identifying the worst historical period of length window_days
stressed_var_basel <- function(portfolio_returns, window_days = 252,
                                 q_var = 0.01, q_es = 0.025) {
  n <- length(portfolio_returns)

  if (n < window_days) {
    cat("Warning: not enough data for full stress window\n")
    window_days <- max(60, n %/% 2)
  }

  # Find worst window (by total loss)
  window_pnl <- sapply(seq_len(n - window_days + 1), function(start) {
    sum(portfolio_returns[start:(start + window_days - 1)])
  })
  worst_start <- which.min(window_pnl)
  stress_period <- portfolio_returns[worst_start:(worst_start + window_days - 1)]

  # Stressed VaR at quantile q
  stressed_var <- quantile(stress_period, q_var)

  # Stressed ES (expected shortfall in stress period)
  stressed_es  <- mean(stress_period[stress_period <= quantile(stress_period, q_es)])

  # Normal VaR for comparison
  normal_var   <- quantile(portfolio_returns, q_var)
  normal_es    <- mean(portfolio_returns[portfolio_returns <= quantile(portfolio_returns, q_es)])

  cat("=== Basel III Stressed VaR ===\n")
  cat(sprintf("Stress period: obs %d to %d\n", worst_start,
              worst_start + window_days - 1))
  cat(sprintf("Stress period total loss: %.2f%%\n", sum(stress_period)*100))
  cat(sprintf("Normal VaR (1%%):    %.4f%%\n", normal_var*100))
  cat(sprintf("Stressed VaR (1%%):  %.4f%%\n", stressed_var*100))
  cat(sprintf("VaR multiplier:     %.2fx\n", abs(stressed_var/normal_var)))
  cat(sprintf("Normal ES (2.5%%):   %.4f%%\n", normal_es*100))
  cat(sprintf("Stressed ES (2.5%%): %.4f%%\n", stressed_es*100))

  list(
    stress_period      = stress_period,
    worst_window_start = worst_start,
    normal_var  = normal_var,
    stressed_var = stressed_var,
    normal_es   = normal_es,
    stressed_es  = stressed_es,
    var_multiplier = abs(stressed_var / normal_var)
  )
}

#' Comprehensive stress test dashboard
stress_test_dashboard <- function(portfolio_weights, asset_names,
                                   portfolio_returns_history,
                                   cov_matrix = NULL) {
  n <- length(portfolio_weights)
  if (is.null(cov_matrix)) {
    cov_matrix <- diag(apply(portfolio_returns_history, 2, var))
  }

  cat("\n══════════════════════════════════════════════════════════\n")
  cat("                STRESS TEST DASHBOARD                      \n")
  cat("══════════════════════════════════════════════════════════\n\n")

  # Historical scenarios
  run_historical_stress_tests(portfolio_weights, asset_names, cov2cor(cov_matrix))
  cat("\n")

  # BTC crash
  cat("--- BTC -50% Shock ---\n")
  btc_sc <- btc_crash_scenario(portfolio_weights, asset_names)
  cat(sprintf("Portfolio loss: %.2f%%\n\n", btc_sc$portfolio_return_pct))

  # Correlation stress
  vols <- sqrt(diag(cov_matrix))
  corr <- cov2cor(cov_matrix)
  stressed_portfolio_var(portfolio_weights, corr, vols)
  cat("\n")

  # Liquidity stress
  cat("--- Liquidity Stress (5x spread widening) ---\n")
  position_sizes <- portfolio_weights * 1e6  # Assume $1M total
  liq_df <- liquidity_stressed_cost(position_sizes[position_sizes > 0] * 1000,
                                     rep(5, sum(portfolio_weights > 0.01)),
                                     spread_multiplier = 5)
  cat(sprintf("Normal total cost: %.2fbps | Stressed: %.2fbps\n",
              mean(liq_df$normal_cost_pct * 100, na.rm=TRUE),
              mean(liq_df$stressed_cost_pct * 100, na.rm=TRUE)))
  cat("\n")

  # Reverse stress
  reverse_stress_test(portfolio_weights, cov_matrix, target_loss_pct = -0.30)
  cat("\n")

  # Stressed VaR
  port_rets_hist <- as.vector(portfolio_returns_history %*% portfolio_weights)
  stressed_var_basel(port_rets_hist)

  invisible(NULL)
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  set.seed(42)
  n <- 6; T_ <- 500

  # Portfolio
  names_vec <- c("BTC","ETH","SOL","BNB","USDC","AVAX")
  weights   <- c(0.35, 0.25, 0.15, 0.10, 0.10, 0.05)

  # Simulate returns
  btc  <- rnorm(T_, 0.001, 0.04)
  rets <- matrix(NA, T_, n)
  betas <- c(1.0, 0.85, 1.1, 0.8, 0.02, 1.05)
  for (i in seq_len(n)) rets[, i] <- betas[i]*btc + rnorm(T_, 0, 0.015)
  colnames(rets) <- names_vec

  cov_m <- cov(rets)

  # Run full stress test dashboard
  stress_test_dashboard(weights, names_vec, rets, cov_m)

  # Individual tests
  btc_sc <- btc_crash_scenario(weights, names_vec, btc_shock = -0.40)
  cat(sprintf("BTC -40%% portfolio loss: %.2f%%\n", btc_sc$portfolio_return_pct))

  # Combined stress
  price_shocks <- c(-0.30, -0.35, -0.40, -0.25, -0.02, -0.45)
  combined_stress_test(weights, names_vec, cov_m, price_shocks, vol_multiplier=2.5)
}
