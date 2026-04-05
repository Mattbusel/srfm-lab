# =============================================================================
# cross_asset_study.R
# Cross-Asset Research: crypto vs equities vs commodities return/vol/corr,
# crypto as inflation hedge, DXY impact, VIX regime analysis, Gold vs BTC,
# risk-on/risk-off crypto behavior.
# Pure base R.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. RETURN / VOLATILITY / CORRELATION TABLE
# ---------------------------------------------------------------------------

#' Compute comprehensive asset class statistics
asset_class_statistics <- function(returns_list, asset_names = NULL,
                                    annualize = 252) {
  n_assets <- length(returns_list)
  if (is.null(asset_names)) asset_names <- paste0("Asset", seq_len(n_assets))

  stats <- lapply(seq_len(n_assets), function(i) {
    r <- returns_list[[i]]
    r <- r[!is.na(r)]
    T_ <- length(r)

    # Basic statistics
    ann_ret <- mean(r) * annualize
    ann_vol <- sd(r) * sqrt(annualize)
    sharpe  <- ann_ret / (ann_vol + 1e-8)

    # Higher moments
    skew    <- mean(((r - mean(r)) / (sd(r) + 1e-8))^3)
    kurt    <- mean(((r - mean(r)) / (sd(r) + 1e-8))^4) - 3

    # Drawdown
    equity  <- cumprod(1 + r)
    max_dd  <- min(equity / cummax(equity) - 1)
    calmar  <- ann_ret / (abs(max_dd) + 1e-8)

    # VaR / ES
    var95   <- quantile(r, 0.05)
    es95    <- mean(r[r <= var95])
    var99   <- quantile(r, 0.01)
    es99    <- mean(r[r <= var99])

    # Best and worst periods
    best_month  <- max(sapply(seq(1, T_-21, by=21), function(s) prod(1+r[s:min(s+20,T_)])-1))
    worst_month <- min(sapply(seq(1, T_-21, by=21), function(s) prod(1+r[s:min(s+20,T_)])-1))

    data.frame(
      asset = asset_names[i],
      n_obs = T_,
      ann_return_pct = ann_ret * 100,
      ann_vol_pct    = ann_vol * 100,
      sharpe         = sharpe,
      max_dd_pct     = max_dd * 100,
      calmar         = calmar,
      skewness       = skew,
      excess_kurtosis = kurt,
      var95_pct      = var95 * 100,
      es95_pct       = es95 * 100,
      best_month_pct  = best_month * 100,
      worst_month_pct = worst_month * 100,
      stringsAsFactors = FALSE
    )
  })

  df <- do.call(rbind, stats)

  cat("╔═══════════════════════════════════════════════════════════════════╗\n")
  cat("║                 Cross-Asset Statistics Summary                     ║\n")
  cat("╚═══════════════════════════════════════════════════════════════════╝\n\n")
  cat(sprintf("%-12s | %9s | %8s | %7s | %8s | %7s | %6s\n",
              "Asset", "AnnRet%", "AnnVol%", "Sharpe", "MaxDD%", "Calmar", "Skew"))
  cat(paste(rep("-", 70), collapse=""), "\n")
  for (i in seq_len(nrow(df))) {
    cat(sprintf("%-12s | %9.2f | %8.2f | %7.3f | %8.2f | %7.3f | %6.2f\n",
                df$asset[i], df$ann_return_pct[i], df$ann_vol_pct[i],
                df$sharpe[i], df$max_dd_pct[i], df$calmar[i], df$skewness[i]))
  }

  invisible(df)
}

#' Pairwise correlation matrix with significance tests
pairwise_correlation_table <- function(returns_list, asset_names = NULL) {
  n <- length(returns_list)
  if (is.null(asset_names)) asset_names <- paste0("Asset", seq_len(n))

  # Align returns by length (use minimum)
  min_T <- min(sapply(returns_list, function(r) sum(!is.na(r))))
  ret_mat <- matrix(NA, min_T, n)
  for (i in seq_len(n)) {
    r <- returns_list[[i]]
    r <- r[!is.na(r)]
    ret_mat[, i] <- tail(r, min_T)
  }
  colnames(ret_mat) <- asset_names

  # Correlation matrix
  S <- cor(ret_mat, use = "complete.obs")

  # T-statistics for correlations
  T_ <- min_T
  t_mat <- S * sqrt((T_ - 2) / (1 - S^2))
  p_mat <- 2 * pt(-abs(t_mat), df = T_ - 2)

  cat("\n=== Pairwise Correlations ===\n")
  cat(sprintf("%-10s", ""))
  for (nm in asset_names) cat(sprintf(" %8s", nm))
  cat("\n")
  for (i in seq_len(n)) {
    cat(sprintf("%-10s", asset_names[i]))
    for (j in seq_len(n)) {
      sig <- if (!is.na(p_mat[i,j]) && p_mat[i,j] < 0.01) "**"
             else if (!is.na(p_mat[i,j]) && p_mat[i,j] < 0.05) "*" else ""
      cat(sprintf(" %6.3f%s", S[i,j], sig))
    }
    cat("\n")
  }
  cat("(** p<1%, * p<5%)\n")

  invisible(list(corr = S, t_stat = t_mat, p_value = p_mat))
}

# ---------------------------------------------------------------------------
# 2. CRYPTO AS INFLATION HEDGE
# ---------------------------------------------------------------------------

#' Empirical test: does crypto outperform during high inflation periods?
#' Compare BTC returns vs CPI change
crypto_inflation_hedge_test <- function(crypto_returns, cpi_changes,
                                         high_inflation_threshold = 0.004,
                                         window = 30) {
  n <- length(crypto_returns)
  stopifnot(length(cpi_changes) == n)

  # Rolling CPI growth
  cpi_ma <- rep(NA, n)
  for (i in window:n) {
    cpi_ma[i] <- mean(cpi_changes[(i-window+1):i])
  }

  high_inflation <- cpi_ma > high_inflation_threshold & !is.na(cpi_ma)
  low_inflation  <- cpi_ma <= high_inflation_threshold & !is.na(cpi_ma)

  # Returns in each regime
  ret_high <- crypto_returns[high_inflation]
  ret_low  <- crypto_returns[low_inflation]

  # Rolling correlation between crypto and CPI
  rolling_corr <- rep(NA, n)
  for (i in window:n) {
    cr <- crypto_returns[(i-window+1):i]
    ci <- cpi_changes[(i-window+1):i]
    valid <- !is.na(cr) & !is.na(ci)
    if (sum(valid) > 5) rolling_corr[i] <- cor(cr[valid], ci[valid])
  }

  # Hedge effectiveness: positive corr with CPI = hedge, negative = not
  avg_corr_with_cpi <- mean(rolling_corr, na.rm=TRUE)
  pct_positive_corr <- mean(rolling_corr > 0, na.rm=TRUE) * 100

  cat("=== Crypto as Inflation Hedge Analysis ===\n")
  cat(sprintf("High inflation threshold: %.1f%% monthly\n", high_inflation_threshold*100))
  cat(sprintf("High inflation periods:   %d days (%.1f%%)\n",
              sum(high_inflation), mean(high_inflation)*100))
  cat(sprintf("Return in high inflation: %+.4f%% daily (%.1f%% ann)\n",
              mean(ret_high, na.rm=TRUE)*100, mean(ret_high, na.rm=TRUE)*252*100))
  cat(sprintf("Return in low inflation:  %+.4f%% daily (%.1f%% ann)\n",
              mean(ret_low, na.rm=TRUE)*100, mean(ret_low, na.rm=TRUE)*252*100))
  cat(sprintf("Avg correlation with CPI: %.3f\n", avg_corr_with_cpi))
  cat(sprintf("Pct time positive corr:   %.1f%%\n", pct_positive_corr))

  is_hedge <- avg_corr_with_cpi > 0 && mean(ret_high, na.rm=TRUE) > mean(ret_low, na.rm=TRUE)
  cat(sprintf("Inflation hedge verdict:  %s\n",
              if (is_hedge) "WEAK POSITIVE HEDGE" else "NOT AN INFLATION HEDGE"))

  list(
    ret_high_inflation = mean(ret_high, na.rm=TRUE),
    ret_low_inflation  = mean(ret_low, na.rm=TRUE),
    avg_corr_with_cpi  = avg_corr_with_cpi,
    rolling_corr       = rolling_corr,
    is_inflation_hedge = is_hedge
  )
}

# ---------------------------------------------------------------------------
# 3. DXY IMPACT ON CRYPTO
# ---------------------------------------------------------------------------

#' Rolling beta of crypto to DXY (US Dollar Index)
#' Strong dollar = crypto headwind (risk-off + liquidity drain)
dxy_crypto_analysis <- function(crypto_returns, dxy_returns, window = 60) {
  n <- length(crypto_returns)
  stopifnot(length(dxy_returns) == n)

  # Rolling beta
  rolling_beta <- rep(NA, n)
  rolling_corr <- rep(NA, n)

  for (i in window:n) {
    idx <- (i-window+1):i
    cr  <- crypto_returns[idx]
    dx  <- dxy_returns[idx]
    valid <- !is.na(cr) & !is.na(dx)
    if (sum(valid) < 10) next

    rolling_beta[i] <- cov(cr[valid], dx[valid]) / (var(dx[valid]) + 1e-10)
    rolling_corr[i] <- cor(cr[valid], dx[valid])
  }

  # Period analysis: strong vs weak dollar
  dxy_strong <- dxy_returns > quantile(dxy_returns, 0.75, na.rm=TRUE)
  dxy_weak   <- dxy_returns < quantile(dxy_returns, 0.25, na.rm=TRUE)

  ret_strong <- mean(crypto_returns[dxy_strong & !is.na(dxy_strong)], na.rm=TRUE)
  ret_weak   <- mean(crypto_returns[dxy_weak   & !is.na(dxy_weak)],   na.rm=TRUE)

  # Overall OLS beta
  valid <- !is.na(crypto_returns) & !is.na(dxy_returns)
  X     <- cbind(1, dxy_returns[valid])
  beta_ols <- tryCatch(
    solve(crossprod(X), crossprod(X, crypto_returns[valid])),
    error = function(e) c(NA, NA)
  )

  cat("=== DXY Impact on Crypto ===\n")
  cat(sprintf("Overall OLS beta to DXY:    %.3f\n", beta_ols[2]))
  cat(sprintf("Overall correlation:        %.3f\n",
              cor(crypto_returns[valid], dxy_returns[valid])))
  cat(sprintf("Return in strong USD:       %+.4f%%/day\n", ret_strong*100))
  cat(sprintf("Return in weak USD:         %+.4f%%/day\n", ret_weak*100))
  cat(sprintf("Strong-minus-weak premium:  %+.4f%%/day\n", (ret_weak-ret_strong)*100))
  cat(sprintf("Avg rolling beta (60d):     %.3f\n", mean(rolling_beta, na.rm=TRUE)))

  list(
    ols_beta = beta_ols[2],
    rolling_beta = rolling_beta,
    rolling_corr = rolling_corr,
    ret_strong_usd = ret_strong,
    ret_weak_usd   = ret_weak
  )
}

# ---------------------------------------------------------------------------
# 4. VIX REGIME AND CRYPTO RETURNS
# ---------------------------------------------------------------------------

#' Classify VIX regime and analyze crypto returns
vix_regime_analysis <- function(crypto_returns, vix_levels,
                                  low_vix = 20, high_vix = 30) {
  n <- length(crypto_returns)
  stopifnot(length(vix_levels) == n)

  # Regimes: Low VIX (calm), Medium, High VIX (fear)
  regime <- ifelse(vix_levels < low_vix,  "Low (<20)",
            ifelse(vix_levels < high_vix, "Medium (20-30)", "High (>30)"))

  # Statistics by VIX regime
  regimes_unique <- c("Low (<20)", "Medium (20-30)", "High (>30)")
  regime_stats <- lapply(regimes_unique, function(r) {
    mask <- regime == r & !is.na(regime)
    rets <- crypto_returns[mask]
    rets <- rets[!is.na(rets)]
    if (length(rets) < 5) return(NULL)

    data.frame(
      vix_regime   = r,
      n_days       = length(rets),
      mean_ret_pct = mean(rets) * 100,
      ann_ret_pct  = mean(rets) * 252 * 100,
      vol_pct      = sd(rets) * sqrt(252) * 100,
      sharpe       = mean(rets) / (sd(rets) + 1e-8) * sqrt(252),
      pct_positive = mean(rets > 0) * 100
    )
  })
  regime_df <- do.call(rbind, Filter(Negate(is.null), regime_stats))

  cat("=== VIX Regime and Crypto Returns ===\n")
  print(regime_df)

  # VIX spike analysis: after VIX jumps >30%, what happens next?
  vix_changes <- c(NA, diff(log(vix_levels)))
  vix_spikes  <- !is.na(vix_changes) & vix_changes > 0.30

  if (sum(vix_spikes) > 3) {
    spike_idx   <- which(vix_spikes)
    post_spike_rets <- sapply(c(1, 3, 5, 10), function(h) {
      rets_h <- sapply(spike_idx, function(i) {
        if (i + h <= n) sum(crypto_returns[(i+1):(i+h)]) else NA
      })
      mean(rets_h, na.rm=TRUE) * 100
    })

    cat("\nPost-VIX-spike returns (>30% VIX jump):\n")
    for (j in seq_along(c(1,3,5,10))) {
      cat(sprintf("  %2d-day avg: %+.2f%%\n", c(1,3,5,10)[j], post_spike_rets[j]))
    }
  }

  invisible(list(regime_df = regime_df, regime = regime))
}

# ---------------------------------------------------------------------------
# 5. GOLD vs BTC: MACRO HEDGE COMPARISON
# ---------------------------------------------------------------------------

#' Compare gold and BTC as macro hedges
#' Hedge properties: negative or low correlation with equities,
#' positive returns during crises, positive beta to inflation
gold_vs_btc_analysis <- function(btc_returns, gold_returns, equity_returns,
                                   cpi_returns = NULL, window = 60) {
  n <- min(length(btc_returns), length(gold_returns), length(equity_returns))
  btc_r  <- tail(btc_returns,  n)
  gold_r <- tail(gold_returns, n)
  eq_r   <- tail(equity_returns, n)

  # Hedge ratio (beta) to equity
  valid <- !is.na(btc_r) & !is.na(gold_r) & !is.na(eq_r)
  X_eq  <- cbind(1, eq_r[valid])
  beta_btc  <- tryCatch(solve(crossprod(X_eq), crossprod(X_eq, btc_r[valid]))[2],
                          error=function(e) NA)
  beta_gold <- tryCatch(solve(crossprod(X_eq), crossprod(X_eq, gold_r[valid]))[2],
                          error=function(e) NA)

  corr_btc_eq  <- cor(btc_r[valid],  eq_r[valid])
  corr_gold_eq <- cor(gold_r[valid], eq_r[valid])

  # Performance during equity drawdowns
  eq_equity <- cumprod(1 + eq_r)
  eq_dd     <- eq_equity / cummax(eq_equity) - 1
  worst_eq  <- eq_dd < quantile(eq_dd, 0.10, na.rm=TRUE)  # Bottom 10% for equities

  btc_in_equity_crisis  <- mean(btc_r[worst_eq],  na.rm=TRUE)
  gold_in_equity_crisis <- mean(gold_r[worst_eq], na.rm=TRUE)

  # Rolling correlation
  roll_corr_btc  <- rep(NA, n)
  roll_corr_gold <- rep(NA, n)
  for (i in window:n) {
    idx <- (i-window+1):i
    v <- !is.na(btc_r[idx]) & !is.na(gold_r[idx]) & !is.na(eq_r[idx])
    if (sum(v) > 10) {
      roll_corr_btc[i]  <- cor(btc_r[idx][v], eq_r[idx][v])
      roll_corr_gold[i] <- cor(gold_r[idx][v], eq_r[idx][v])
    }
  }

  cat("╔══════════════════════════════════════════════════════╗\n")
  cat("║           Gold vs BTC: Macro Hedge Comparison        ║\n")
  cat("╚══════════════════════════════════════════════════════╝\n\n")

  metrics <- data.frame(
    metric = c("Ann Return", "Ann Volatility", "Sharpe", "Equity Beta",
               "Corr with Equities", "Return in Equity Crisis"),
    BTC  = c(mean(btc_r, na.rm=TRUE)*252*100, sd(btc_r, na.rm=TRUE)*sqrt(252)*100,
              mean(btc_r,na.rm=TRUE)/sd(btc_r,na.rm=TRUE)*sqrt(252),
              beta_btc*100, corr_btc_eq, btc_in_equity_crisis*100),
    Gold = c(mean(gold_r,na.rm=TRUE)*252*100, sd(gold_r,na.rm=TRUE)*sqrt(252)*100,
              mean(gold_r,na.rm=TRUE)/sd(gold_r,na.rm=TRUE)*sqrt(252),
              beta_gold*100, corr_gold_eq, gold_in_equity_crisis*100)
  )
  print(metrics)

  # Verdict
  btc_hedge_score  <- -corr_btc_eq  + (btc_in_equity_crisis > 0) * 0.5
  gold_hedge_score <- -corr_gold_eq + (gold_in_equity_crisis > 0) * 0.5
  cat(sprintf("\nHedge Score (higher = better hedge):\n"))
  cat(sprintf("  BTC:  %.3f\n  Gold: %.3f\n", btc_hedge_score, gold_hedge_score))
  cat(sprintf("  Better macro hedge: %s\n",
              if (gold_hedge_score > btc_hedge_score) "Gold" else "BTC"))

  invisible(list(
    metrics = metrics,
    roll_corr_btc = roll_corr_btc,
    roll_corr_gold = roll_corr_gold,
    btc_in_crisis = btc_in_equity_crisis,
    gold_in_crisis = gold_in_equity_crisis
  ))
}

# ---------------------------------------------------------------------------
# 6. RISK-ON / RISK-OFF: CRYPTO BEHAVIOR
# ---------------------------------------------------------------------------

#' Classify risk-on vs risk-off regimes using multiple indicators
classify_risk_regime <- function(equity_returns, vix_levels = NULL,
                                  credit_spreads = NULL, window = 20) {
  n <- length(equity_returns)

  # Score components
  eq_trend <- rep(NA, n)
  for (i in window:n) {
    eq_trend[i] <- sum(equity_returns[(i-window+1):i])
  }

  # VIX contribution
  if (!is.null(vix_levels)) {
    vix_ma   <- rep(NA, n)
    for (i in window:n) vix_ma[i] <- mean(vix_levels[(i-window+1):i])
    vix_signal <- -(vix_levels - vix_ma) / (vix_ma + 1)  # Negative VIX change = risk-on
  } else {
    vix_signal <- eq_trend / (abs(eq_trend) + 1e-8)
  }

  # Combined risk score
  risk_score <- eq_trend / (abs(eq_trend) + 1e-8) + vix_signal
  risk_score[is.na(risk_score)] <- 0

  risk_on  <- risk_score > 0
  risk_off <- risk_score <= 0

  list(
    risk_score = risk_score,
    risk_on    = risk_on,
    risk_off   = risk_off,
    pct_risk_on = mean(risk_on, na.rm=TRUE) * 100
  )
}

#' Analyze crypto behavior in risk-on vs risk-off
risk_on_off_crypto_analysis <- function(crypto_returns, equity_returns,
                                         vix_levels = NULL, window = 20) {
  regime <- classify_risk_regime(equity_returns, vix_levels, window=window)

  n <- length(crypto_returns)
  risk_on  <- regime$risk_on[seq_len(n)]
  risk_off <- regime$risk_off[seq_len(n)]

  ret_risk_on  <- crypto_returns[risk_on  & !is.na(risk_on)]
  ret_risk_off <- crypto_returns[risk_off & !is.na(risk_off)]

  # Beta in each regime
  valid_on  <- !is.na(risk_on) & risk_on & !is.na(equity_returns) & !is.na(crypto_returns)
  valid_off <- !is.na(risk_off) & risk_off & !is.na(equity_returns) & !is.na(crypto_returns)

  beta_on  <- if (sum(valid_on)  > 5) cov(crypto_returns[valid_on],  equity_returns[valid_on])  / (var(equity_returns[valid_on])  + 1e-10) else NA
  beta_off <- if (sum(valid_off) > 5) cov(crypto_returns[valid_off], equity_returns[valid_off]) / (var(equity_returns[valid_off]) + 1e-10) else NA

  cat("=== Risk-On / Risk-Off Crypto Analysis ===\n")
  cat(sprintf("Risk-On periods:  %d days (%.1f%%)\n",
              sum(risk_on, na.rm=TRUE), regime$pct_risk_on))
  cat(sprintf("Risk-Off periods: %d days (%.1f%%)\n",
              sum(risk_off, na.rm=TRUE), 100 - regime$pct_risk_on))
  cat(sprintf("Crypto in Risk-On:  %+.4f%%/day (Sharpe %.2f)\n",
              mean(ret_risk_on,  na.rm=TRUE)*100,
              mean(ret_risk_on,  na.rm=TRUE) / (sd(ret_risk_on, na.rm=TRUE) + 1e-8) * sqrt(252)))
  cat(sprintf("Crypto in Risk-Off: %+.4f%%/day (Sharpe %.2f)\n",
              mean(ret_risk_off, na.rm=TRUE)*100,
              mean(ret_risk_off, na.rm=TRUE) / (sd(ret_risk_off, na.rm=TRUE) + 1e-8) * sqrt(252)))
  cat(sprintf("Beta in Risk-On:  %.3f\n", beta_on))
  cat(sprintf("Beta in Risk-Off: %.3f\n", beta_off))
  cat(sprintf("Risk-On premium:  %+.4f%%/day\n",
              (mean(ret_risk_on, na.rm=TRUE) - mean(ret_risk_off, na.rm=TRUE))*100))

  list(
    ret_risk_on   = ret_risk_on,
    ret_risk_off  = ret_risk_off,
    beta_on = beta_on, beta_off = beta_off,
    risk_score = regime$risk_score
  )
}

# ---------------------------------------------------------------------------
# 7. GENERATE SYNTHETIC MULTI-ASSET DATA
# ---------------------------------------------------------------------------

#' Simulate realistic multi-asset returns with realistic correlations
generate_multi_asset_data <- function(T_ = 1000, seed = 42) {
  set.seed(seed)

  # Factor model: global risk-on factor, inflation factor, USD factor
  f_risk_on  <- rnorm(T_, 0.0003, 0.01)   # Risk-on factor
  f_inflation <- rnorm(T_, 0.0001, 0.003) # Inflation factor
  f_usd <- -f_risk_on * 0.3 + rnorm(T_, 0, 0.005)  # USD moves contra risk-on

  # BTC: high beta to risk-on, modest inflation hedge
  btc_ret <- 2.5 * f_risk_on + 0.5 * f_inflation + rnorm(T_, 0.0005, 0.03)

  # ETH: high beta, less inflation
  eth_ret <- 2.8 * f_risk_on + 0.3 * f_inflation + rnorm(T_, 0.0003, 0.035)

  # Gold: negative beta to risk-on (safe haven), positive inflation
  gold_ret <- -0.3 * f_risk_on + 1.5 * f_inflation + rnorm(T_, 0.0002, 0.008)

  # Oil: risk-on + inflation
  oil_ret  <- 1.2 * f_risk_on + 2.0 * f_inflation + rnorm(T_, 0.0001, 0.02)

  # Equities (S&P): risk-on beta
  sp500_ret <- 1.0 * f_risk_on + 0.2 * f_inflation + rnorm(T_, 0.0004, 0.012)

  # DXY: contra risk-on
  dxy_ret  <- -0.8 * f_risk_on - 0.5 * f_inflation + rnorm(T_, 0, 0.005)

  # CPI proxy: inflation factor
  cpi_change <- f_inflation + rnorm(T_, 0.0002, 0.001)

  # VIX proxy: negative risk-on
  vix_level <- 20 * exp(cumsum(-2 * f_risk_on + rnorm(T_, 0, 0.05)))
  vix_level <- pmax(10, pmin(80, vix_level))

  list(
    btc   = btc_ret,
    eth   = eth_ret,
    gold  = gold_ret,
    oil   = oil_ret,
    sp500 = sp500_ret,
    dxy   = dxy_ret,
    cpi   = cpi_change,
    vix   = vix_level,
    factors = data.frame(f_risk_on = f_risk_on,
                          f_inflation = f_inflation,
                          f_usd = f_usd)
  )
}

# ---------------------------------------------------------------------------
# 8. FULL CROSS-ASSET STUDY PIPELINE
# ---------------------------------------------------------------------------

#' Run complete cross-asset study
run_cross_asset_study <- function(T_ = 1000, seed = 42) {
  cat("\n══════════════════════════════════════════════════════════\n")
  cat("              CROSS-ASSET EMPIRICAL STUDY                  \n")
  cat("══════════════════════════════════════════════════════════\n\n")

  data <- generate_multi_asset_data(T_, seed)

  # 1. Summary statistics table
  cat("### 1. Asset Class Statistics ###\n")
  stats <- asset_class_statistics(
    list(data$btc, data$eth, data$gold, data$oil, data$sp500, data$dxy),
    asset_names = c("BTC", "ETH", "Gold", "Oil", "S&P500", "DXY")
  )
  cat("\n")

  # 2. Pairwise correlations
  cat("### 2. Pairwise Correlations ###\n")
  corr_result <- pairwise_correlation_table(
    list(data$btc, data$eth, data$gold, data$oil, data$sp500, data$dxy),
    asset_names = c("BTC", "ETH", "Gold", "Oil", "SP500", "DXY")
  )
  cat("\n")

  # 3. Inflation hedge
  cat("### 3. BTC as Inflation Hedge ###\n")
  inf_hedge <- crypto_inflation_hedge_test(data$btc, data$cpi)
  cat("\n")

  # 4. DXY impact
  cat("### 4. DXY Impact on BTC ###\n")
  dxy_impact <- dxy_crypto_analysis(data$btc, data$dxy, window=60)
  cat("\n")

  # 5. VIX regime
  cat("### 5. VIX Regime Analysis ###\n")
  vix_analysis <- vix_regime_analysis(data$btc, data$vix, low_vix=20, high_vix=30)
  cat("\n")

  # 6. Gold vs BTC
  cat("### 6. Gold vs BTC Hedge Comparison ###\n")
  hedge_comparison <- gold_vs_btc_analysis(data$btc, data$gold, data$sp500)
  cat("\n")

  # 7. Risk-on/off
  cat("### 7. Risk-On/Off Crypto Behavior ###\n")
  roro <- risk_on_off_crypto_analysis(data$btc, data$sp500, data$vix)
  cat("\n")

  # 8. Key findings summary
  cat("══════════════════════════════════════════════════════════\n")
  cat("                    KEY FINDINGS                           \n")
  cat("══════════════════════════════════════════════════════════\n\n")

  btc_sharpe <- stats$sharpe[stats$asset == "BTC"]
  sp_sharpe  <- stats$sharpe[stats$asset == "S&P500"]
  gold_sharpe <- stats$sharpe[stats$asset == "Gold"]

  cat(sprintf("1. BTC Sharpe (%.2f) vs S&P500 (%.2f) vs Gold (%.2f)\n",
              btc_sharpe, sp_sharpe, gold_sharpe))
  cat(sprintf("2. BTC-S&P500 correlation: %.3f\n",
              corr_result$corr["BTC", "SP500"]))
  cat(sprintf("3. BTC inflation hedge? %s\n",
              if (inf_hedge$is_inflation_hedge) "Yes" else "No"))
  cat(sprintf("4. DXY beta of BTC: %.3f\n", dxy_impact$ols_beta))
  cat(sprintf("5. BTC in risk-on: %+.2f%%/day | Risk-off: %+.2f%%/day\n",
              mean(roro$ret_risk_on, na.rm=TRUE)*100,
              mean(roro$ret_risk_off, na.rm=TRUE)*100))
  cat(sprintf("6. BTC in equity crisis: %+.2f%%/day (Gold: %+.2f%%/day)\n",
              hedge_comparison$btc_in_crisis*100,
              hedge_comparison$gold_in_crisis*100))

  invisible(list(data=data, stats=stats, corr=corr_result,
                 inf_hedge=inf_hedge, dxy=dxy_impact,
                 vix=vix_analysis, hedge=hedge_comparison, roro=roro))
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  # Run full cross-asset study
  study <- run_cross_asset_study(T_ = 1500, seed = 2025)

  # Additional: rolling correlations over time (BTC vs equities)
  set.seed(42)
  data <- generate_multi_asset_data(1000)
  window <- 60
  roll_corr_btc_sp500 <- rep(NA, 1000)
  for (i in window:1000) {
    idx <- (i-window+1):i
    roll_corr_btc_sp500[i] <- cor(data$btc[idx], data$sp500[idx])
  }
  cat(sprintf("\nRolling 60d BTC-SP500 corr: mean=%.3f, range=[%.3f, %.3f]\n",
              mean(roll_corr_btc_sp500, na.rm=TRUE),
              min(roll_corr_btc_sp500, na.rm=TRUE),
              max(roll_corr_btc_sp500, na.rm=TRUE)))
}
