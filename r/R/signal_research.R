# =============================================================================
# signal_research.R
# Signal Research Toolkit: IC/ICIR computation, Fama-MacBeth regression,
# IC decay analysis, signal orthogonalization (Gram-Schmidt), signal recycling,
# rolling ICIR stability, signal capacity analysis, and optimal combination
# via IC-covariance weighting.
# Pure base R.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. IC AND ICIR COMPUTATION
# ---------------------------------------------------------------------------

#' Information Coefficient: rank correlation of signal with forward return
#' Spearman rank correlation is standard (robust to outliers)
compute_ic <- function(signal, forward_return, method = "spearman") {
  valid <- !is.na(signal) & !is.na(forward_return)
  if (sum(valid) < 10) return(NA_real_)
  cor(signal[valid], forward_return[valid], method = method)
}

#' Rolling IC over time
rolling_ic <- function(signal, forward_return, window = 60) {
  n <- length(signal)
  ic <- rep(NA_real_, n)
  for (i in window:n) {
    idx <- (i - window + 1):i
    ic[i] <- compute_ic(signal[idx], forward_return[idx])
  }
  ic
}

#' IC Information Ratio: mean(IC) / sd(IC) -- risk-adjusted signal strength
#' High ICIR = consistent signal (not just occasionally lucky)
compute_icir <- function(signal, forward_return, window = 60, step = 1) {
  n   <- length(signal)
  ic_series <- numeric(0)

  for (end in seq(window, n, by = step)) {
    start <- end - window + 1
    ic <- compute_ic(signal[start:end], forward_return[start:end])
    ic_series <- c(ic_series, ic)
  }

  ic_series <- ic_series[!is.na(ic_series)]
  if (length(ic_series) < 3) {
    return(list(icir=NA, ic_mean=NA, ic_sd=NA, n_periods=0))
  }

  list(
    icir     = mean(ic_series) / (sd(ic_series) + 1e-8),
    ic_mean  = mean(ic_series),
    ic_sd    = sd(ic_series),
    ic_t_stat = mean(ic_series) / (sd(ic_series) / sqrt(length(ic_series)) + 1e-8),
    ic_pct_positive = mean(ic_series > 0) * 100,
    n_periods = length(ic_series),
    ic_series = ic_series
  )
}

#' Compute IC for multiple signals simultaneously
multi_signal_ic_table <- function(signals_matrix, forward_return,
                                   signal_names = NULL) {
  n_signals <- ncol(signals_matrix)
  if (is.null(signal_names)) signal_names <- paste0("Signal_", seq_len(n_signals))

  results <- lapply(seq_len(n_signals), function(j) {
    icir_result <- compute_icir(signals_matrix[, j], forward_return)
    data.frame(
      signal     = signal_names[j],
      ic_mean    = icir_result$ic_mean,
      ic_sd      = icir_result$ic_sd,
      icir       = icir_result$icir,
      ic_t_stat  = icir_result$ic_t_stat,
      pct_positive_ic = icir_result$ic_pct_positive,
      stringsAsFactors = FALSE
    )
  })

  df <- do.call(rbind, results)
  df[order(-abs(df$icir)), ]
}

# ---------------------------------------------------------------------------
# 2. FAMA-MACBETH CROSS-SECTIONAL REGRESSION
# ---------------------------------------------------------------------------
# Step 1: For each time period t, regress cross-sectional returns on factors
# Step 2: Average the time-series of coefficients
# Step 3: T-test on average coefficient (Newey-West SE adjustment)

#' Single period cross-sectional regression
cross_sectional_reg <- function(returns, factors_matrix) {
  # returns: N-vector of returns for period t
  # factors: N x K matrix of factor exposures at time t
  valid <- complete.cases(cbind(returns, factors_matrix))
  if (sum(valid) < ncol(factors_matrix) + 3) {
    return(rep(NA, ncol(factors_matrix) + 1))
  }
  X <- cbind(1, factors_matrix[valid, , drop=FALSE])
  y <- returns[valid]
  XtX_inv <- tryCatch(solve(crossprod(X) + diag(1e-8, ncol(X))),
                       error = function(e) NULL)
  if (is.null(XtX_inv)) return(rep(NA, ncol(X)))
  as.vector(XtX_inv %*% crossprod(X, y))
}

#' Full Fama-MacBeth regression
#' returns_matrix: [T x N]; factors_array: [T x N x K]
fama_macbeth <- function(returns_matrix, factors_array,
                          factor_names = NULL, lag = 1) {
  T_ <- nrow(returns_matrix); N <- ncol(returns_matrix)

  if (is.matrix(factors_array)) {
    # Single factor: replicate across assets (e.g., market beta)
    K <- 1
    factors_array <- array(factors_array, dim = c(T_, N, 1))
  } else {
    K <- dim(factors_array)[3]
  }

  if (is.null(factor_names)) factor_names <- paste0("F", seq_len(K))

  # For each time period, run cross-sectional regression
  gammas <- matrix(NA, T_, K + 1)  # K + intercept

  for (t in (lag + 1):T_ ) {
    ret_t    <- returns_matrix[t, ]
    factor_t <- matrix(factors_array[t - lag, , ], N, K)
    gammas[t, ] <- cross_sectional_reg(ret_t, factor_t)
  }

  # Average lambdas (FM step 2)
  valid_rows <- which(complete.cases(gammas))
  if (length(valid_rows) < 5) {
    return(list(lambda=rep(NA, K+1), t_stat=rep(NA, K+1)))
  }

  gamma_valid <- gammas[valid_rows, , drop=FALSE]
  lambda_mean <- colMeans(gamma_valid)
  lambda_sd   <- apply(gamma_valid, 2, sd)
  t_stat      <- lambda_mean / (lambda_sd / sqrt(length(valid_rows)) + 1e-8)

  # Newey-West standard errors (lag = 4)
  nw_se <- function(x, max_lag = 4) {
    n <- length(x); x_dm <- x - mean(x)
    # Variance + covariance terms
    nw_var <- sum(x_dm^2) / n
    for (l in seq_len(max_lag)) {
      w  <- 1 - l / (max_lag + 1)
      nw_var <- nw_var + 2 * w * sum(x_dm[(l+1):n] * x_dm[1:(n-l)]) / n
    }
    sqrt(pmax(nw_var / n, 0))
  }

  nw_ses <- apply(gamma_valid, 2, nw_se)
  nw_t   <- lambda_mean / (nw_ses + 1e-8)

  cat("=== Fama-MacBeth Regression ===\n")
  cat(sprintf("Periods: %d | Assets: %d\n", length(valid_rows), N))
  cat(sprintf("%-12s | %8s | %8s | %8s | %8s\n",
              "Factor", "Lambda", "T-stat", "NW T-stat", "Sig"))
  factor_labels <- c("Intercept", factor_names)
  for (k in seq_along(lambda_mean)) {
    sig <- if (abs(nw_t[k]) > 2.58) "***" else if (abs(nw_t[k]) > 1.96) "**" else
           if (abs(nw_t[k]) > 1.64) "*" else ""
    cat(sprintf("%-12s | %8.4f | %8.3f | %8.3f | %s\n",
                factor_labels[k], lambda_mean[k], t_stat[k], nw_t[k], sig))
  }

  list(
    lambda      = lambda_mean[-1],  # Exclude intercept
    intercept   = lambda_mean[1],
    t_stats     = t_stat[-1],
    nw_t_stats  = nw_t[-1],
    nw_ses      = nw_ses[-1],
    gamma_series = gamma_valid,
    significant = abs(nw_t[-1]) > 1.96
  )
}

# ---------------------------------------------------------------------------
# 3. SIGNAL DECAY ANALYSIS
# ---------------------------------------------------------------------------

#' IC at multiple forward horizons (1 to max_h bars)
ic_decay_curve <- function(signal, returns, max_h = 48,
                            window = NULL) {
  n <- length(signal)

  ic_by_h <- sapply(seq_len(max_h), function(h) {
    # Compound return over h periods
    fwd_ret <- numeric(n)
    for (i in seq_len(n - h)) {
      fwd_ret[i] <- sum(returns[(i + 1):(i + h)])  # Arithmetic sum (approx)
    }
    fwd_ret[(n - h + 1):n] <- NA

    if (!is.null(window)) {
      # Rolling IC, take average
      ics <- rolling_ic(signal, fwd_ret, window)
      mean(ics, na.rm = TRUE)
    } else {
      compute_ic(signal, fwd_ret)
    }
  })

  df <- data.frame(
    horizon = seq_len(max_h),
    ic = ic_by_h,
    ic_abs = abs(ic_by_h)
  )

  # Fit exponential decay: IC(h) = IC_0 * exp(-lambda * h)
  valid <- !is.na(df$ic) & df$ic_abs > 0
  if (sum(valid) >= 3) {
    log_ic <- log(pmax(df$ic_abs[valid], 1e-10))
    h_valid <- df$horizon[valid]
    fit_coef <- tryCatch({
      lm_r <- lm(log_ic ~ h_valid)
      coef(lm_r)
    }, error = function(e) c(NA, NA))
    decay_lambda <- -fit_coef[2]
    half_life <- if (!is.na(decay_lambda) && decay_lambda > 0) log(2) / decay_lambda else NA
  } else {
    decay_lambda <- NA; half_life <- NA
  }

  list(
    ic_by_horizon = df,
    peak_ic       = max(df$ic_abs, na.rm=TRUE),
    peak_horizon  = df$horizon[which.max(df$ic_abs)],
    half_life     = half_life,
    decay_lambda  = decay_lambda
  )
}

# ---------------------------------------------------------------------------
# 4. SIGNAL ORTHOGONALIZATION (GRAM-SCHMIDT)
# ---------------------------------------------------------------------------

#' Gram-Schmidt orthogonalization of signals
#' Removes redundancy: new_signal_k = signal_k - projection onto previous k-1
#' Useful for combining correlated signals without double-counting

gram_schmidt_signals <- function(signals_matrix) {
  n <- nrow(signals_matrix)
  k <- ncol(signals_matrix)
  signal_names <- colnames(signals_matrix)

  Q <- matrix(0, n, k)  # Orthogonalized signals

  for (j in seq_len(k)) {
    v <- signals_matrix[, j]
    v[is.na(v)] <- 0

    # Subtract projections onto previous orthogonal signals
    for (i in seq_len(j - 1)) {
      q_prev <- Q[, i]
      denom  <- sum(q_prev^2)
      if (denom > 1e-10) {
        v <- v - (sum(v * q_prev) / denom) * q_prev
      }
    }

    # Normalize
    norm_v <- sqrt(sum(v^2))
    if (norm_v > 1e-10) {
      Q[, j] <- v / norm_v * sd(signals_matrix[, j], na.rm=TRUE)  # Preserve scale
    }
  }

  colnames(Q) <- if (!is.null(signal_names)) paste0(signal_names, "_orth") else
                 paste0("S", seq_len(k), "_orth")

  # Correlation reduction
  corr_before <- cor(signals_matrix, use="complete.obs")
  corr_after  <- cor(Q)

  list(
    orthogonalized = Q,
    avg_corr_before = mean(abs(corr_before[upper.tri(corr_before)])),
    avg_corr_after  = mean(abs(corr_after[upper.tri(corr_after)])),
    reduction_pct   = (1 - mean(abs(corr_after[upper.tri(corr_after)])) /
                       mean(abs(corr_before[upper.tri(corr_before)]))) * 100
  )
}

# ---------------------------------------------------------------------------
# 5. SIGNAL RECYCLING: REVIVE DECAYED SIGNALS VIA COMBINATION
# ---------------------------------------------------------------------------

#' Signal recycling: when a signal's IC has decayed, try:
#' 1. Cross-timeframe combination (current + lagged)
#' 2. Residual extraction (remove market factor)
#' 3. Regime-conditional reactivation

#' Cross-timeframe signal combination
#' Combine signal at multiple lookback windows, weighted by IC
multi_timeframe_signal <- function(returns, windows = c(5, 10, 20, 60),
                                    forward_horizon = 1, estimation_window = 120) {
  n <- length(returns)
  n_windows <- length(windows)

  # For each window, compute momentum signal
  signals <- matrix(NA, n, n_windows)
  for (j in seq_len(n_windows)) {
    w <- windows[j]
    for (t in (w + 1):n) {
      signals[t, j] <- sum(returns[(t - w):(t - 1)])
    }
  }

  # Forward return (1 bar)
  fwd_ret <- c(returns[-1], NA)

  # Rolling IC-weighted combination
  combined <- rep(NA, n)

  for (t in (estimation_window + 1):n) {
    train_idx <- (t - estimation_window):(t - 1)
    fwd_train <- fwd_ret[train_idx]

    ics <- sapply(seq_len(n_windows), function(j) {
      compute_ic(signals[train_idx, j], fwd_train)
    })
    ics[is.na(ics)] <- 0

    # Positive IC weighting
    pos_ics <- pmax(ics, 0)
    if (sum(pos_ics) > 0) {
      w <- pos_ics / sum(pos_ics)
      combined[t] <- sum(w * signals[t, ], na.rm=TRUE)
    }
  }

  list(
    combined = combined,
    individual_signals = signals,
    windows = windows
  )
}

#' Factor-residualized signal
#' Remove market factor exposure to get idiosyncratic signal component
residualize_signal <- function(signal, market_factor, window = 60) {
  n <- length(signal)
  residual <- rep(NA_real_, n)

  for (t in window:n) {
    idx <- (t - window + 1):t
    s_t <- signal[idx]; m_t <- market_factor[idx]
    valid <- !is.na(s_t) & !is.na(m_t)
    if (sum(valid) < 10) next

    # Regress signal on market factor
    X <- cbind(1, m_t[valid])
    y <- s_t[valid]
    beta <- tryCatch(solve(crossprod(X) + diag(1e-8, 2), crossprod(X, y)),
                     error = function(e) c(mean(y), 0))
    # Residual for current period
    if (!is.na(signal[t]) && !is.na(market_factor[t])) {
      residual[t] <- signal[t] - beta[1] - beta[2] * market_factor[t]
    }
  }
  residual
}

# ---------------------------------------------------------------------------
# 6. ROLLING ICIR STABILITY
# ---------------------------------------------------------------------------

#' Assess ICIR stability over time: rolling ICIR
rolling_icir_stability <- function(signal, forward_return,
                                    ic_window = 60,
                                    icir_window = 30) {
  n <- length(signal)
  rolling_ics <- rolling_ic(signal, forward_return, ic_window)

  # Rolling ICIR: rolling mean/sd of ICs
  rolling_icir_vals <- rep(NA_real_, n)
  for (t in (ic_window + icir_window):n) {
    ic_slice <- rolling_ics[(t - icir_window + 1):t]
    valid    <- !is.na(ic_slice)
    if (sum(valid) < 5) next
    rolling_icir_vals[t] <- mean(ic_slice[valid]) / (sd(ic_slice[valid]) + 1e-8)
  }

  # Stability metrics
  icir_valid <- rolling_icir_vals[!is.na(rolling_icir_vals)]

  list(
    rolling_icir   = rolling_icir_vals,
    icir_mean      = mean(icir_valid),
    icir_sd        = sd(icir_valid),
    icir_stability = mean(icir_valid) / (sd(icir_valid) + 1e-8),  # ICIR of ICIR
    pct_positive_icir = mean(icir_valid > 0) * 100,
    min_icir       = min(icir_valid),
    max_icir       = max(icir_valid)
  )
}

# ---------------------------------------------------------------------------
# 7. SIGNAL CAPACITY: WHEN DOES ALPHA DECAY WITH AUM?
# ---------------------------------------------------------------------------
# A signal's IC decreases as AUM grows because:
# 1. Price impact from larger trades
# 2. Competition: other funds notice and trade against the signal

#' Model signal IC as function of AUM
#' IC(AUM) = IC_0 * (1 - AUM / AUM_max)^alpha
#' AUM_max = market liquidity * capacity_fraction
ic_vs_aum <- function(ic_0 = 0.05, aum_range = seq(0, 1e9, by=1e7),
                       market_adv_usd = 5e8,   # Daily ADV for the universe
                       capacity_fraction = 0.10, # Can trade 10% of ADV
                       decay_alpha = 0.5) {
  # Maximum tradeable AUM before completely degrading signal
  aum_max <- market_adv_usd * 252 * capacity_fraction  # Annualized capacity

  # IC as function of AUM
  ic_at_aum <- ic_0 * pmax(0, (1 - aum_range / aum_max))^decay_alpha

  # Dollars of alpha = IC * vol * sqrt(N) * AUM  (simplified)
  # Actually: expected alpha_bps = IC * vol_bps * information_ratio_multiple
  vol_bps <- 200  # 2% daily vol in bps
  alpha_bps <- ic_at_aum * vol_bps
  alpha_usd <- alpha_bps / 10000 * aum_range

  # Maximum alpha capacity: find AUM that maximizes total alpha $
  max_alpha_idx <- which.max(alpha_usd)

  cat("=== Signal Capacity Analysis ===\n")
  cat(sprintf("Initial IC (low AUM):       %.4f\n", ic_0))
  cat(sprintf("AUM capacity (max):         $%s\n", format(round(aum_max), big.mark=",")))
  cat(sprintf("AUM at peak alpha dollars:  $%s\n",
              format(round(aum_range[max_alpha_idx]), big.mark=",")))
  cat(sprintf("Peak alpha per year:        $%s\n",
              format(round(max(alpha_usd)), big.mark=",")))
  cat(sprintf("IC at peak AUM:             %.4f\n", ic_at_aum[max_alpha_idx]))

  df <- data.frame(
    aum = aum_range,
    ic  = ic_at_aum,
    alpha_bps = alpha_bps,
    alpha_usd = alpha_usd
  )
  list(capacity_curve = df,
       optimal_aum = aum_range[max_alpha_idx],
       peak_alpha_usd = max(alpha_usd),
       aum_max = aum_max)
}

# ---------------------------------------------------------------------------
# 8. SIGNAL COMBINATION: OPTIMAL WEIGHTS VIA IC-COVARIANCE
# ---------------------------------------------------------------------------
# Optimal combination weights minimize variance of combined IC
# Analogous to minimum-variance portfolio in IC space

#' IC-covariance matrix (covariance of signal ICs)
ic_covariance_matrix <- function(signals_matrix, forward_return,
                                  window = 60, step = 5) {
  n <- nrow(signals_matrix); k <- ncol(signals_matrix)

  ic_series_matrix <- matrix(NA, 0, k)
  for (end in seq(window, n, by = step)) {
    start <- end - window + 1
    ics <- sapply(seq_len(k), function(j) {
      compute_ic(signals_matrix[start:end, j], forward_return[start:end])
    })
    ic_series_matrix <- rbind(ic_series_matrix, ics)
  }

  valid_rows <- complete.cases(ic_series_matrix)
  ic_series_matrix <- ic_series_matrix[valid_rows, , drop=FALSE]

  if (nrow(ic_series_matrix) < k + 1) {
    return(list(cov_matrix = diag(k), mean_ic = rep(0, k)))
  }

  list(
    cov_matrix = cov(ic_series_matrix),
    mean_ic    = colMeans(ic_series_matrix),
    ic_series  = ic_series_matrix
  )
}

#' Optimal signal combination weights (mean-variance in IC space)
#' Maximize: E[IC_combined] / Var[IC_combined]
#' Analogous to max Sharpe in IC space
optimal_signal_weights <- function(signals_matrix, forward_return,
                                    window = 60, lambda_reg = 0.01,
                                    min_weight = 0, max_weight = 1) {
  k <- ncol(signals_matrix)

  ic_params <- ic_covariance_matrix(signals_matrix, forward_return, window)
  mu_ic  <- ic_params$mean_ic
  Sigma_ic <- ic_params$cov_matrix + diag(lambda_reg, k)

  # Only use signals with positive IC
  pos_mask <- mu_ic > 0
  if (sum(pos_mask) == 0) {
    return(list(weights = rep(1/k, k), expected_icir = 0))
  }

  mu_pos <- mu_ic[pos_mask]
  S_pos  <- Sigma_ic[pos_mask, pos_mask, drop=FALSE]

  # Max Sharpe in IC space: w* = Sigma^-1 * mu / (1' * Sigma^-1 * mu)
  S_inv <- tryCatch(solve(S_pos), error = function(e) solve(S_pos + diag(1e-6, sum(pos_mask))))
  w_raw <- as.vector(S_inv %*% mu_pos)
  w_raw <- pmax(w_raw, 0)  # Long-only IC combination
  if (sum(w_raw) < 1e-10) w_raw <- rep(1/sum(pos_mask), sum(pos_mask))
  w_pos <- w_raw / sum(w_raw)

  # Expand to full weight vector
  w_full <- rep(0, k)
  w_full[pos_mask] <- w_pos

  # Expected combined IC
  combined_ic_mean <- sum(w_full * mu_ic)
  combined_ic_var  <- as.numeric(t(w_full) %*% Sigma_ic %*% w_full)
  combined_icir    <- combined_ic_mean / sqrt(pmax(combined_ic_var, 1e-10))

  signal_names <- colnames(signals_matrix)
  if (is.null(signal_names)) signal_names <- paste0("S", seq_len(k))

  cat("=== Optimal Signal Combination Weights ===\n")
  cat(sprintf("%-15s | %8s | %8s | %8s\n", "Signal", "IC Mean", "IC Std", "Weight"))
  for (j in seq_len(k)) {
    cat(sprintf("%-15s | %8.4f | %8.4f | %8.4f\n",
                signal_names[j], mu_ic[j],
                sqrt(Sigma_ic[j,j]), w_full[j]))
  }
  cat(sprintf("\nCombined ICIR: %.3f\n", combined_icir))

  list(
    weights = w_full,
    expected_ic     = combined_ic_mean,
    expected_icir   = combined_icir,
    mu_ic = mu_ic,
    sigma_ic = Sigma_ic
  )
}

#' Apply combined signal and backtest
backtest_combined_signal <- function(signals_matrix, weights, returns,
                                      tc_bps = 5, signal_threshold = 0) {
  n <- nrow(signals_matrix)
  signal_names <- colnames(signals_matrix)

  # Combined signal
  combined <- as.vector(signals_matrix %*% weights)

  # Cross-sectional ranking: normalize to z-score
  combined_z <- (combined - mean(combined, na.rm=TRUE)) /
                (sd(combined, na.rm=TRUE) + 1e-8)

  # Position: long if above threshold, short if below -threshold
  position <- ifelse(combined_z > signal_threshold, 1,
              ifelse(combined_z < -signal_threshold, -1, 0))
  position[is.na(position)] <- 0

  # Daily strategy return
  position_lag <- c(0, position[-n])
  trade_indicator <- abs(position - position_lag)
  tc_cost <- trade_indicator * tc_bps / 10000

  strat_ret <- position_lag * returns - tc_cost

  # Metrics
  sharpe   <- mean(strat_ret, na.rm=TRUE) / (sd(strat_ret, na.rm=TRUE) + 1e-8) * sqrt(252)
  cum_ret  <- prod(1 + strat_ret, na.rm=TRUE) - 1
  equity   <- cumprod(1 + strat_ret)
  max_dd   <- min(equity / cummax(equity) - 1, na.rm=TRUE)
  n_trades <- sum(trade_indicator > 0, na.rm=TRUE)

  list(
    returns  = strat_ret,
    equity   = equity,
    sharpe   = sharpe,
    cum_ret  = cum_ret,
    max_dd   = max_dd,
    n_trades = n_trades,
    position = position
  )
}

# ---------------------------------------------------------------------------
# 9. SIGNAL STABILITY: CROSS-VALIDATION AND IS/OOS DEGRADATION
# ---------------------------------------------------------------------------

#' Walk-forward IC stability test
#' Split into in-sample estimation and out-of-sample validation
wf_ic_stability <- function(signal, forward_return, n_splits = 5) {
  n   <- length(signal)
  fold_size <- n %/% (n_splits + 1)

  is_ics <- numeric(n_splits)
  oos_ics <- numeric(n_splits)

  for (k in seq_len(n_splits)) {
    train_end  <- k * fold_size
    test_start <- train_end + 1
    test_end   <- min(test_start + fold_size - 1, n)

    is_ics[k]  <- compute_ic(signal[1:train_end], forward_return[1:train_end])
    oos_ics[k] <- compute_ic(signal[test_start:test_end],
                              forward_return[test_start:test_end])
  }

  degradation <- (oos_ics - is_ics) / (abs(is_ics) + 1e-8) * 100

  data.frame(
    fold = seq_len(n_splits),
    is_ic    = is_ics,
    oos_ic   = oos_ics,
    degradation_pct = degradation
  )
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  set.seed(42)
  n <- 600

  # Simulate returns
  returns <- rnorm(n, 0.001, 0.025)
  fwd_ret <- c(returns[-1], NA)

  # Generate 5 signals with different IC levels
  signals <- matrix(NA, n, 5)
  true_ics <- c(0.08, 0.05, -0.03, 0.06, 0.01)
  noise_sds <- sqrt(1/true_ics^2 - 1) * sd(fwd_ret)

  for (j in seq_len(5)) {
    pure  <- c(fwd_ret[-1], NA)
    noise <- rnorm(n, 0, abs(noise_sds[j]))
    signals[, j] <- pure + noise
  }
  colnames(signals) <- c("Momentum","Carry","Reversal","Quality","Macro")

  # IC table
  cat("=== Signal IC Table ===\n")
  ic_tbl <- multi_signal_ic_table(signals, fwd_ret)
  print(ic_tbl)
  cat("\n")

  # Fama-MacBeth (simplified: single asset, signal as factor)
  # Build multi-asset data
  N_assets <- 30; T_ <- 252
  asset_rets <- matrix(rnorm(T_*N_assets, 0.001, 0.03), T_, N_assets)
  btc_betas  <- matrix(runif(T_*N_assets, 0.3, 1.2), T_, N_assets)
  cat("=== Fama-MacBeth ===\n")
  fm <- fama_macbeth(asset_rets, btc_betas, factor_names = "BTC_beta")
  cat("\n")

  # IC decay
  cat("=== IC Decay Analysis: Momentum Signal ===\n")
  decay <- ic_decay_curve(signals[, 1], returns, max_h = 24)
  cat(sprintf("Peak IC=%.4f at horizon %d | Half-life=%.1f bars\n",
              decay$peak_ic, decay$peak_horizon, decay$half_life))
  cat("\n")

  # Orthogonalization
  orth <- gram_schmidt_signals(signals)
  cat(sprintf("=== Signal Orthogonalization ===\n"))
  cat(sprintf("Avg abs corr before: %.3f | after: %.3f (%.1f%% reduction)\n",
              orth$avg_corr_before, orth$avg_corr_after, orth$reduction_pct))
  cat("\n")

  # Optimal combination
  cat("=== Optimal Signal Combination ===\n")
  opt <- optimal_signal_weights(signals, fwd_ret, window=60)
  bt  <- backtest_combined_signal(signals, opt$weights, returns)
  cat(sprintf("Combined strategy Sharpe: %.3f | Cum return: %.2f%% | MaxDD: %.2f%%\n",
              bt$sharpe, bt$cum_ret*100, bt$max_dd*100))

  # Capacity analysis
  cat("\n=== Signal Capacity ===\n")
  cap <- ic_vs_aum(ic_0 = 0.08, market_adv_usd = 2e9)
}
