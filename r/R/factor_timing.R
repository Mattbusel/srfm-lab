# =============================================================================
# factor_timing.R
# Factor Timing Models for Crypto
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Not all factors work all the time. Value underperforms
# in momentum-driven bull markets; momentum crashes in sudden reversals.
# Factor timing dynamically adjusts factor exposure based on forecasts of
# which factor will do well in the next period, using momentum, valuation,
# volatility, and regime signals to over/underweight factors.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

roll_mean <- function(x, w) {
  n <- length(x); out <- rep(NA_real_, n)
  for (i in w:n) out[i] <- mean(x[(i-w+1):i], na.rm=TRUE)
  out
}

roll_sd <- function(x, w) {
  n <- length(x); out <- rep(NA_real_, n)
  for (i in w:n) out[i] <- sd(x[(i-w+1):i], na.rm=TRUE)
  out
}

roll_sum <- function(x, w) {
  n <- length(x); out <- rep(NA_real_, n)
  for (i in w:n) out[i] <- sum(x[(i-w+1):i], na.rm=TRUE)
  out
}

#' Information coefficient (IC): rank correlation of signal vs forward return
ic <- function(signal, forward_return) {
  valid <- !is.na(signal) & !is.na(forward_return)
  if (sum(valid) < 3) return(NA_real_)
  cor(rank(signal[valid]), rank(forward_return[valid]))
}

sharpe_ratio <- function(rets, ann = 252) {
  mu <- mean(rets, na.rm=TRUE); sg <- sd(rets, na.rm=TRUE)
  if (is.na(sg) || sg < 1e-12) return(NA_real_)
  mu / sg * sqrt(ann)
}

max_drawdown <- function(eq) {
  pk <- cummax(eq)
  min((eq - pk) / pk, na.rm = TRUE)
}

# ---------------------------------------------------------------------------
# 2. SYNTHETIC FACTOR RETURNS
# ---------------------------------------------------------------------------
# Simulate K factor return series with regime-dependent means.

simulate_factors <- function(T_ = 1000L, K = 5L, seed = 42L) {
  set.seed(seed)
  factor_names <- c("Momentum", "Value", "Low_Vol", "Quality", "OnChain")
  K <- min(K, length(factor_names))

  # Regime series: 1=risk-on, 2=risk-off
  regime <- integer(T_)
  regime[1] <- 1L
  for (t in 2:T_) {
    if (regime[t-1] == 1L) regime[t] <- sample(1:2, 1, prob=c(0.97,0.03))
    else                   regime[t] <- sample(1:2, 1, prob=c(0.15,0.85))
  }

  # Regime-dependent factor means (annualised -> daily)
  means_on  <- c(0.12, -0.04, 0.06, 0.08, 0.10) / 252
  means_off <- c(-0.15, 0.08, 0.10, 0.04, -0.05) / 252

  vols <- c(0.08, 0.06, 0.05, 0.05, 0.10) / sqrt(252)
  R    <- matrix(NA, T_, K)
  for (t in seq_len(T_)) {
    mu_t <- if (regime[t] == 1L) means_on[1:K] else means_off[1:K]
    R[t, ] <- mu_t + vols[1:K] * rnorm(K)
  }
  colnames(R) <- factor_names[1:K]
  list(returns = R, regime = regime, factor_names = factor_names[1:K])
}

# ---------------------------------------------------------------------------
# 3. FACTOR MOMENTUM
# ---------------------------------------------------------------------------
# Factor momentum: recent (e.g., 12-month) factor return predicts next month.
# Rank factors by past return; overweight top, underweight bottom.

compute_factor_momentum <- function(factor_returns,
                                     lookback = 252L,
                                     skip = 21L) {
  T_  <- nrow(factor_returns); K <- ncol(factor_returns)
  mom <- matrix(NA, T_, K)
  for (t in (lookback + skip + 1):T_) {
    end_   <- t - skip - 1L
    start_ <- end_ - lookback + 1L
    if (start_ < 1) next
    mom[t, ] <- colSums(factor_returns[start_:end_, , drop=FALSE], na.rm=TRUE)
  }
  mom
}

#' Factor momentum IC: does past factor return predict future?
factor_momentum_ic <- function(factor_returns, lookback = 252L, skip = 21L,
                                forward_window = 21L) {
  T_  <- nrow(factor_returns); K <- ncol(factor_returns)
  mom <- compute_factor_momentum(factor_returns, lookback, skip)
  ics <- numeric(T_)
  for (t in seq_len(T_ - forward_window)) {
    fwd <- colSums(factor_returns[(t+1):(t+forward_window), , drop=FALSE])
    sig <- mom[t, ]
    valid <- !is.na(sig) & !is.na(fwd)
    if (sum(valid) < 3) next
    ics[t] <- ic(sig[valid], fwd[valid])
  }
  ics
}

#' Long-short factor momentum portfolio weights
factor_momentum_weights <- function(factor_returns, lookback = 252L, skip = 21L,
                                     top_k = 2L) {
  T_  <- nrow(factor_returns); K <- ncol(factor_returns)
  mom <- compute_factor_momentum(factor_returns, lookback, skip)
  W   <- matrix(0, T_, K)
  for (t in seq_len(T_)) {
    sig <- mom[t, ]
    if (any(is.na(sig))) next
    ranks <- rank(sig)
    top   <- ranks >= (K - top_k + 1)
    bot   <- ranks <= top_k
    W[t, top] <-  1 / sum(top)
    W[t, bot] <- -1 / sum(bot)
  }
  W
}

# ---------------------------------------------------------------------------
# 4. FACTOR VALUATION (CHEAP VS EXPENSIVE)
# ---------------------------------------------------------------------------
# Compute "value spread" for each factor: IC of past return-reversal signal.
# Cheap factor = long stretch of underperformance -> mean reversion expected.

factor_value_signal <- function(factor_returns, val_window = 252L) {
  T_  <- nrow(factor_returns); K <- ncol(factor_returns)
  val <- matrix(NA, T_, K)
  for (t in (val_window + 1):T_) {
    # Cumulative return over val_window: low = "cheap"
    cum <- colSums(factor_returns[(t-val_window+1):t, , drop=FALSE])
    val[t, ] <- -cum   # cheap if underperformed (negative cum = positive val score)
  }
  val
}

# ---------------------------------------------------------------------------
# 5. FACTOR VOLATILITY TARGETING
# ---------------------------------------------------------------------------
# Scale each factor's weight by target_vol / factor_vol.
# Financial intuition: reduce size when factor is volatile (noisy signal).

vol_target_weights <- function(weights, factor_returns,
                                target_vol_ann = 0.10,
                                vol_window = 60L) {
  T_  <- nrow(weights); K <- ncol(weights)
  target_vol <- target_vol_ann / sqrt(252)
  W_scaled   <- matrix(0, T_, K)
  for (k in seq_len(K)) {
    sig_k <- roll_sd(factor_returns[, k], vol_window)
    for (t in seq_len(T_)) {
      if (is.na(sig_k[t]) || sig_k[t] < 1e-8) {
        W_scaled[t, k] <- weights[t, k]
      } else {
        scaler <- target_vol / sig_k[t]
        scaler <- clip(scaler, 0.2, 5.0)   # cap leverage
        W_scaled[t, k] <- weights[t, k] * scaler
      }
    }
  }
  W_scaled
}

# ---------------------------------------------------------------------------
# 6. FACTOR CORRELATION TIMING
# ---------------------------------------------------------------------------
# When factors are highly correlated, reduce total exposure (risk is concentrated).
# Diversification ratio = sqrt(w'Sigma_w) / (w' sigma_vec)

factor_correlation_timer <- function(weights, factor_returns,
                                      max_factor_corr = 0.7,
                                      cor_window = 60L) {
  T_  <- nrow(weights); K <- ncol(weights)
  W_adj <- weights

  for (t in (cor_window + 1):T_) {
    R_w   <- factor_returns[(t-cor_window+1):t, , drop=FALSE]
    C     <- tryCatch(cor(R_w), error = function(e) diag(K))
    C[is.na(C)] <- 0; diag(C) <- 1
    # Average off-diagonal correlation
    avg_cor <- (sum(C) - K) / (K * (K - 1))
    if (avg_cor > max_factor_corr) {
      scale_down <- max_factor_corr / avg_cor
      W_adj[t, ] <- weights[t, ] * scale_down
    }
  }
  W_adj
}

# ---------------------------------------------------------------------------
# 7. MACRO-CONDITIONAL FACTOR RETURNS
# ---------------------------------------------------------------------------
# Analyse each factor's performance conditional on macro regime.

macro_conditional_factor <- function(factor_returns, macro_regime) {
  # macro_regime: integer vector, 1=risk-on, 2=risk-off
  T_  <- nrow(factor_returns); K <- ncol(factor_returns)
  regimes <- sort(unique(macro_regime))
  result  <- list()
  for (r in regimes) {
    mask <- macro_regime == r
    sub  <- factor_returns[mask, , drop=FALSE]
    result[[r]] <- data.frame(
      regime = r,
      factor = colnames(factor_returns),
      mean_ann  = colMeans(sub, na.rm=TRUE) * 252,
      vol_ann   = apply(sub, 2, sd, na.rm=TRUE) * sqrt(252),
      sharpe    = sapply(seq_len(K), function(k) sharpe_ratio(sub[,k])),
      hit_rate  = colMeans(sub > 0, na.rm=TRUE)
    )
  }
  do.call(rbind, result)
}

# ---------------------------------------------------------------------------
# 8. FACTOR CROWDING DETECTION
# ---------------------------------------------------------------------------
# Crowding proxy: rolling correlation of factor with its own recent performance.
# High correlation = many funds on same side = crowded.

factor_crowding <- function(factor_returns, window = 63L) {
  T_  <- nrow(factor_returns); K <- ncol(factor_returns)
  crowding <- matrix(NA, T_, K)
  for (t in (2*window+1):T_) {
    for (k in seq_len(K)) {
      r1 <- factor_returns[(t-2*window+1):(t-window), k]
      r2 <- factor_returns[(t-window+1):t, k]
      if (sd(r1)<1e-8 || sd(r2)<1e-8) next
      crowding[t, k] <- cor(r1, r2)
    }
  }
  colnames(crowding) <- colnames(factor_returns)
  crowding
}

# ---------------------------------------------------------------------------
# 9. DYNAMIC FACTOR WEIGHTS VIA REGIME CLASSIFIER
# ---------------------------------------------------------------------------
# Simple regime classifier: if macro index > threshold -> risk-on weights.

regime_conditional_weights <- function(factor_returns, macro_signal,
                                         risk_on_bias  = c(1.0, -0.5, 0.5, 0.5, 1.0),
                                         risk_off_bias = c(-0.5, 1.0, 1.5, 1.0, -0.5),
                                         threshold = 0) {
  T_  <- nrow(factor_returns); K <- ncol(factor_returns)
  n_b <- min(length(risk_on_bias), K)
  W   <- matrix(0, T_, K)
  for (t in seq_len(T_)) {
    ms <- macro_signal[t]
    if (is.na(ms)) next
    if (ms > threshold) {
      raw <- risk_on_bias[1:K]
    } else {
      raw <- risk_off_bias[1:K]
    }
    # Normalise to unit sum of abs weights
    W[t, ] <- raw / (sum(abs(raw)) + 1e-8)
  }
  W
}

# ---------------------------------------------------------------------------
# 10. COMBINED FACTOR TIMING SIGNAL
# ---------------------------------------------------------------------------

blend_factor_signals <- function(mom_weights, val_weights, regime_weights,
                                  w_mom = 1/3, w_val = 1/3, w_reg = 1/3) {
  (w_mom * mom_weights + w_val * val_weights + w_reg * regime_weights)
}

# ---------------------------------------------------------------------------
# 11. FACTOR PORTFOLIO BACKTEST
# ---------------------------------------------------------------------------

factor_portfolio_return <- function(weights, factor_returns, tc = 0.002) {
  T_  <- nrow(weights); K <- ncol(weights)
  rets <- numeric(T_)
  for (t in 2:T_) {
    r    <- factor_returns[t, ]
    w_tm1 <- weights[t-1, ]
    ret  <- sum(w_tm1 * r, na.rm=TRUE)
    cost <- sum(abs(weights[t,] - w_tm1), na.rm=TRUE) * tc / 2
    rets[t] <- ret - cost
  }
  rets
}

# ---------------------------------------------------------------------------
# 12. INFORMATION COEFFICIENT DECAY
# ---------------------------------------------------------------------------
# How quickly does a factor signal lose predictive power?
# Compute IC at horizons 1, 5, 10, 21, 63 days.

ic_decay <- function(signal, future_returns, horizons = c(1,5,10,21,63)) {
  T_  <- nrow(future_returns); K <- ncol(future_returns)
  ic_mat <- matrix(NA, length(horizons), K)
  for (hi in seq_along(horizons)) {
    h <- horizons[hi]
    for (k in seq_len(K)) {
      fwd <- roll_sum(future_returns[, k], h)
      valid <- !is.na(signal[, k]) & !is.na(fwd)
      if (sum(valid) < 3) next
      ic_mat[hi, k] <- ic(signal[valid, k], fwd[valid])
    }
  }
  rownames(ic_mat) <- horizons
  colnames(ic_mat) <- colnames(future_returns)
  ic_mat
}

# ---------------------------------------------------------------------------
# 13. IS/OOS FACTOR TIMING BACKTEST
# ---------------------------------------------------------------------------

factor_timing_backtest <- function(factor_sim,
                                    is_frac  = 0.6,
                                    method   = "momentum") {
  R    <- factor_sim$returns
  reg  <- factor_sim$regime
  T_   <- nrow(R); K <- ncol(R)
  is_end <- as.integer(T_ * is_frac)

  # In-sample: find best factor (oracle)
  is_sharpes <- apply(R[1:is_end, ], 2, sharpe_ratio)
  best_factor_is <- which.max(is_sharpes)

  # OOS: apply timing signal
  oos_start <- is_end + 1L

  if (method == "momentum") {
    mom <- compute_factor_momentum(R, lookback=252L, skip=21L)
    W   <- factor_momentum_weights(R)
    W_vt <- vol_target_weights(W, R, target_vol_ann=0.10)
  } else {
    W_vt <- matrix(1/K, T_, K)
  }

  # Equal weight benchmark
  W_ew <- matrix(1/K, T_, K)

  rets_timing <- factor_portfolio_return(W_vt, R)
  rets_ew     <- factor_portfolio_return(W_ew, R)
  # BH best-factor-IS (oracle)
  rets_oracle <- R[, best_factor_is]

  oos_idx <- oos_start:T_
  list(
    IS = list(
      timing = sharpe_ratio(rets_timing[1:is_end]),
      ew     = sharpe_ratio(rets_ew[1:is_end]),
      oracle = sharpe_ratio(rets_oracle[1:is_end])
    ),
    OOS = list(
      timing = sharpe_ratio(rets_timing[oos_idx]),
      ew     = sharpe_ratio(rets_ew[oos_idx]),
      oracle = sharpe_ratio(rets_oracle[oos_idx])
    ),
    rets_timing = rets_timing,
    rets_ew     = rets_ew,
    W           = W_vt
  )
}

# ---------------------------------------------------------------------------
# 14. FACTOR TIMING TEARSHEET
# ---------------------------------------------------------------------------

factor_timing_tearsheet <- function(backtest_res) {
  rets_t  <- backtest_res$rets_timing
  rets_ew <- backtest_res$rets_ew
  eq_t    <- cumprod(1 + rets_t[rets_t != 0])
  eq_ew   <- cumprod(1 + rets_ew[rets_ew != 0])
  data.frame(
    strategy = c("Timing", "EW"),
    sharpe   = c(backtest_res$OOS$timing, backtest_res$OOS$ew),
    max_dd   = c(max_drawdown(eq_t), max_drawdown(eq_ew)),
    total_ret = c(tail(eq_t,1)-1, tail(eq_ew,1)-1)
  )
}

# ---------------------------------------------------------------------------
# 15. MAIN DEMO
# ---------------------------------------------------------------------------

run_factor_timing_demo <- function() {
  cat("=== Factor Timing Demo ===\n\n")

  # Simulate factor returns
  fac <- simulate_factors(T_ = 1000L, K = 5L)
  R   <- fac$returns
  reg <- fac$regime
  K   <- ncol(R)
  cat(sprintf("Factor returns: %d bars x %d factors\n", nrow(R), K))
  cat("Factors:", colnames(R), "\n")

  cat("\n--- 1. Factor Return Stats by Regime ---\n")
  cond_stats <- macro_conditional_factor(R, reg)
  print(cond_stats[, c("regime","factor","mean_ann","sharpe")])

  cat("\n--- 2. Factor Momentum Signal ---\n")
  mom  <- compute_factor_momentum(R, lookback=252L, skip=21L)
  mom_ic <- factor_momentum_ic(R, lookback=252L, skip=21L)
  valid <- !is.na(mom_ic) & mom_ic != 0
  cat(sprintf("  Mean IC of factor momentum: %.4f\n", mean(mom_ic[valid])))

  cat("\n--- 3. Factor Valuation Signal ---\n")
  val  <- factor_value_signal(R, val_window=252L)
  W_val <- val; W_val[is.na(W_val)] <- 0
  # Normalise
  rs <- rowSums(abs(W_val)) + 1e-8
  W_val <- W_val / rs

  cat("\n--- 4. Momentum Portfolio Weights (sample) ---\n")
  W_mom <- factor_momentum_weights(R)
  t_sample <- 500L
  cat("  Weights at t=500:", round(W_mom[t_sample,], 3), "\n")

  cat("\n--- 5. Vol-Targeted Weights ---\n")
  W_vt <- vol_target_weights(W_mom, R)
  cat("  Vol-targeted weights at t=500:", round(W_vt[t_sample,], 3), "\n")

  cat("\n--- 6. Correlation Timing ---\n")
  W_ct <- factor_correlation_timer(W_mom, R, max_factor_corr=0.5)
  cat("  Corr-timed weights at t=500:", round(W_ct[t_sample,], 3), "\n")

  cat("\n--- 7. Regime-Conditional Weights ---\n")
  macro_signal <- ifelse(reg == 1L, 1, -1)
  W_reg <- regime_conditional_weights(R, macro_signal)
  cat("  Regime weights (risk-on bar):", round(W_reg[which(macro_signal>0)[1],], 3), "\n")
  cat("  Regime weights (risk-off bar):", round(W_reg[which(macro_signal<0)[1],], 3), "\n")

  cat("\n--- 8. Combined Factor Timing Signal ---\n")
  W_blend <- blend_factor_signals(W_vt, W_val, W_reg)

  cat("\n--- 9. Factor Crowding ---\n")
  crowding <- factor_crowding(R, window=63L)
  valid_cr  <- !is.na(crowding)
  for (k in seq_len(K)) {
    cv <- crowding[, k][!is.na(crowding[,k])]
    cat(sprintf("  %-12s crowding mean=%.3f  max=%.3f\n",
                colnames(R)[k], mean(cv), max(cv)))
  }

  cat("\n--- 10. IC Decay Analysis ---\n")
  mom_clean <- mom; mom_clean[is.na(mom_clean)] <- 0
  icd <- ic_decay(mom_clean, R)
  cat("  IC at horizons 1,5,10,21,63 days:\n")
  print(round(icd, 4))

  cat("\n--- 11. IS/OOS Factor Timing Backtest ---\n")
  bt <- factor_timing_backtest(fac, is_frac=0.6, method="momentum")
  cat(sprintf("  IS:  Timing Sharpe=%.3f  EW=%.3f  Oracle=%.3f\n",
              bt$IS$timing, bt$IS$ew, bt$IS$oracle))
  cat(sprintf("  OOS: Timing Sharpe=%.3f  EW=%.3f  Oracle=%.3f\n",
              bt$OOS$timing, bt$OOS$ew, bt$OOS$oracle))

  cat("\n--- 12. Factor Timing Tearsheet ---\n")
  ts <- factor_timing_tearsheet(bt)
  print(ts)

  cat("\nDone.\n")
  invisible(list(fac=fac, bt=bt, crowding=crowding, icd=icd))
}

if (interactive()) {
  ft_results <- run_factor_timing_demo()
}

# ---------------------------------------------------------------------------
# 16. FACTOR RETURN SEASONALITY
# ---------------------------------------------------------------------------
# Test: do certain factors have consistent day-of-week effects?

factor_seasonality <- function(factor_returns, period_fn = function(t) t %% 5L) {
  T_ <- nrow(factor_returns); K <- ncol(factor_returns)
  period <- sapply(seq_len(T_), period_fn)
  results <- list()
  for (k in seq_len(K)) {
    r   <- factor_returns[,k]
    sea <- tapply(r, period, function(x) c(mean=mean(x,na.rm=TRUE),
                                            sd=sd(x,na.rm=TRUE),
                                            n=length(x)))
    means <- sapply(sea, `[`, 1)
    sds   <- sapply(sea, `[`, 2)
    ns    <- sapply(sea, `[`, 3)
    t_stats <- means / (sds / sqrt(ns))
    results[[k]] <- data.frame(factor=colnames(factor_returns)[k],
                                period=names(means), mean=means,
                                t_stat=t_stats)
  }
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 17. FACTOR PORTFOLIO TURNOVER MANAGEMENT
# ---------------------------------------------------------------------------

smooth_factor_weights <- function(W, decay = 0.7) {
  T_  <- nrow(W)
  W_s <- matrix(0, T_, ncol(W))
  W_s[1,] <- W[1,]
  for (t in 2:T_) {
    W_s[t,] <- decay * W_s[t-1,] + (1-decay) * W[t,]
  }
  W_s
}

#' Turnover statistics for factor portfolio
factor_turnover_stats <- function(W) {
  T_  <- nrow(W)
  to  <- apply(W, 2, function(w) mean(abs(diff(w)), na.rm=TRUE))
  list(by_factor=to, mean=mean(to), max=max(to))
}

# ---------------------------------------------------------------------------
# 18. FACTOR ALPHA DECAY
# ---------------------------------------------------------------------------
# Measure: IC at horizon h. Does momentum alpha decay quickly?

factor_alpha_decay <- function(factor_rets, horizons=1:20) {
  T_ <- nrow(factor_rets); K <- ncol(factor_rets)
  ic_mat <- matrix(NA, length(horizons), K)
  for (hi in seq_along(horizons)) {
    h <- horizons[hi]
    for (k in seq_len(K)) {
      r1 <- factor_rets[1:(T_-h), k]
      r2 <- sapply(seq_len(T_-h), function(t) sum(factor_rets[t:(t+h-1),k]))
      ic_mat[hi,k] <- cor(r1, r2, use="complete.obs")
    }
  }
  rownames(ic_mat) <- horizons
  colnames(ic_mat) <- colnames(factor_rets)
  ic_mat
}

# ---------------------------------------------------------------------------
# 19. FACTOR TIMING SHARPE ATTRIBUTION
# ---------------------------------------------------------------------------

sharpe_attribution <- function(timing_rets, ew_rets) {
  timing_mean <- mean(timing_rets, na.rm=TRUE)
  ew_mean     <- mean(ew_rets, na.rm=TRUE)
  timing_vol  <- sd(timing_rets, na.rm=TRUE)
  ew_vol      <- sd(ew_rets, na.rm=TRUE)

  # Sharpe improvement decomposed into return and vol contributions
  delta_sharpe <- sharpe_ratio(timing_rets) - sharpe_ratio(ew_rets)
  ret_contrib  <- (timing_mean - ew_mean) / max(ew_vol, 1e-8) * sqrt(252)
  vol_contrib  <- delta_sharpe - ret_contrib

  data.frame(delta_sharpe=delta_sharpe,
             return_contribution=ret_contrib,
             vol_contribution=vol_contrib,
             timing_sharpe=sharpe_ratio(timing_rets),
             ew_sharpe=sharpe_ratio(ew_rets))
}

# ---------------------------------------------------------------------------
# 20. FACTOR TIMING EXTENDED DEMO
# ---------------------------------------------------------------------------

run_factor_timing_extended_demo <- function() {
  cat("=== Factor Timing Extended Demo ===\n\n")
  fac <- simulate_factors(T_=800L, K=5L, seed=55L)
  R   <- fac$returns; K <- ncol(R)

  cat("--- Factor Seasonality (Day-of-Week) ---\n")
  sea <- factor_seasonality(R)
  best_rows <- sea[order(-abs(sea$t_stat)),][1:5,]
  print(best_rows)

  cat("\n--- Smooth Factor Weights (Reduce Turnover) ---\n")
  W_mom  <- factor_momentum_weights(R)
  W_sm   <- smooth_factor_weights(W_mom, decay=0.8)
  to_raw <- factor_turnover_stats(W_mom)
  to_sm  <- factor_turnover_stats(W_sm)
  cat(sprintf("  Mean turnover: raw=%.4f  smoothed=%.4f\n",
              to_raw$mean, to_sm$mean))

  rets_raw <- factor_portfolio_return(W_mom, R)
  rets_sm  <- factor_portfolio_return(W_sm, R)
  cat(sprintf("  Sharpe: raw=%.3f  smoothed=%.3f\n",
              sharpe_ratio(rets_raw), sharpe_ratio(rets_sm)))

  cat("\n--- Factor Alpha Decay ---\n")
  ad <- factor_alpha_decay(R, horizons=c(1,5,10,21))
  cat("  IC at horizons 1,5,10,21:\n")
  print(round(ad, 4))

  cat("\n--- Sharpe Attribution ---\n")
  W_ew   <- matrix(1/K, nrow(R), K)
  rets_ew <- factor_portfolio_return(W_ew, R)
  W_vt    <- vol_target_weights(W_mom, R)
  rets_vt <- factor_portfolio_return(W_vt, R)
  sa <- sharpe_attribution(rets_vt, rets_ew)
  print(sa)

  invisible(list(sea=sea, to_sm=to_sm, ad=ad, sa=sa))
}

if (interactive()) {
  ft_ext <- run_factor_timing_extended_demo()
}

# =============================================================================
# SECTION: FACTOR RISK PARITY ALLOCATION
# =============================================================================
# Equal-risk-contribution across factors: each factor contributes the same
# fraction of total portfolio variance.  Solved via IRLS / gradient descent.

factor_risk_parity <- function(Sigma, max_iter = 200, tol = 1e-8) {
  n <- ncol(Sigma)
  w <- rep(1/n, n)
  for (iter in seq_len(max_iter)) {
    Sw   <- Sigma %*% w
    rc   <- w * Sw / as.numeric(t(w) %*% Sw)   # risk contributions
    # Gradient: move weight from high-rc to low-rc factors
    grad <- rc - 1/n
    step <- 0.1 / (iter^0.5)
    w_new <- pmax(w - step * grad, 1e-6)
    w_new <- w_new / sum(w_new)
    if (max(abs(w_new - w)) < tol) break
    w <- w_new
  }
  w / sum(w)
}

# =============================================================================
# SECTION: FACTOR CONDITIONAL VALUE-AT-RISK TIMING
# =============================================================================
# Reduce exposure to factors that have elevated conditional tail risk.
# CVaR estimated via historical simulation on a rolling window.

factor_cvar_signal <- function(factor_rets, window = 60, alpha = 0.05) {
  # Returns scalar CVaR estimate (negative = left tail loss)
  if (length(factor_rets) < window) return(0)
  r   <- tail(factor_rets, window)
  q   <- quantile(r, alpha)
  mean(r[r <= q])
}

factor_cvar_timing <- function(F_mat, window = 60, alpha = 0.05,
                                cvar_thresh = -0.03) {
  # Reduce weight to factor when its CVaR exceeds threshold
  T <- nrow(F_mat); k <- ncol(F_mat)
  weights <- matrix(1/k, T, k)
  for (t in seq(window+1, T)) {
    w <- rep(1/k, k)
    for (j in seq_len(k)) {
      cv <- factor_cvar_signal(F_mat[1:(t-1), j], window, alpha)
      if (cv < cvar_thresh) w[j] <- 0.5 / k   # halve weight if tail-risky
    }
    weights[t,] <- w / sum(w)
  }
  weights
}

# =============================================================================
# SECTION: FACTOR INFORMATION COEFFICIENT DECAY
# =============================================================================
# IC decay measures how quickly a factor signal loses predictive power.
# A fast-decaying IC -> short holding period; slow decay -> hold longer.

ic_decay_curve <- function(factor_scores, fwd_rets, max_lag = 20) {
  # factor_scores: T x k matrix, fwd_rets: T x k or T x 1
  # Returns vector of IC at lag 1..max_lag
  T  <- nrow(factor_scores)
  ic <- numeric(max_lag)
  for (lag in seq_len(max_lag)) {
    if (T - lag < 5) break
    f  <- as.vector(factor_scores[1:(T-lag), ])
    r  <- as.vector(fwd_rets[(1+lag):T, ])
    if (sd(f) < 1e-9 || sd(r) < 1e-9) next
    ic[lag] <- cor(f, r, method = "spearman")
  }
  ic
}

ic_half_life <- function(ic_curve) {
  # Find lag at which |IC| drops to half its initial value
  if (length(ic_curve) == 0) return(NA_integer_)
  init <- abs(ic_curve[1])
  if (init < 1e-9) return(NA_integer_)
  which(abs(ic_curve) < 0.5 * init)[1]
}

# =============================================================================
# SECTION: FACTOR SIGNAL COMBINATION — EQUAL / IC-WEIGHTED
# =============================================================================

combine_factor_signals <- function(signals, method = c("equal", "ic_weighted"),
                                   ics = NULL) {
  method <- match.arg(method)
  if (method == "equal" || is.null(ics)) {
    rowMeans(signals, na.rm = TRUE)
  } else {
    pos_ics <- pmax(ics, 0)
    if (sum(pos_ics) < 1e-9) return(rowMeans(signals, na.rm = TRUE))
    w <- pos_ics / sum(pos_ics)
    as.vector(signals %*% w)
  }
}

# =============================================================================
# SECTION: FACTOR PORTFOLIO PERFORMANCE ATTRIBUTION
# =============================================================================
# Decompose total return into: factor return * beta + alpha + residual.

factor_perf_attribution <- function(port_rets, factor_rets) {
  # OLS regression of portfolio returns on factor returns
  df   <- data.frame(y = port_rets, as.data.frame(factor_rets))
  fit  <- lm(y ~ ., data = df)
  betas <- coef(fit)[-1]
  alpha <- coef(fit)[1]
  fac_contrib <- colMeans(factor_rets) * betas
  list(
    alpha       = alpha,
    betas       = betas,
    fac_contrib = fac_contrib,
    total_explained = sum(fac_contrib) + alpha,
    r_squared   = summary(fit)$r.squared
  )
}

# =============================================================================
# SECTION: FACTOR EXPOSURE NEUTRALISATION
# =============================================================================
# Build a factor-neutral portfolio by orthogonalising weights with respect
# to unwanted factor exposures.

neutralise_factor <- function(weights, factor_loading) {
  # Project out factor_loading direction from weights vector
  fl_norm <- factor_loading / (sum(factor_loading^2) + 1e-10)
  weights - sum(weights * factor_loading) * fl_norm
}

neutralise_multiple_factors <- function(weights, F_loadings) {
  # F_loadings: k x n matrix (k factors, n assets)
  w <- weights
  for (j in seq_len(nrow(F_loadings)))
    w <- neutralise_factor(w, F_loadings[j,])
  w
}

# =============================================================================
# SECTION: FACTOR TIMING WITH MACRO OVERLAY
# =============================================================================
# Adjust factor timing weights based on a macro regime signal.
# High-inflation regime: tilt to value/quality. Risk-on: tilt to momentum.

macro_factor_overlay <- function(base_weights, macro_score,
                                  thresholds = c(-0.5, 0.5)) {
  # macro_score: scalar, negative = risk-off, positive = risk-on
  # base_weights: named vector of factor weights
  if (macro_score < thresholds[1]) {
    # Risk-off: downscale momentum, upscale low-vol
    base_weights["momentum"] <- base_weights["momentum"] * 0.5
    base_weights["low_vol"]  <- base_weights["low_vol"]  * 1.5
  } else if (macro_score > thresholds[2]) {
    # Risk-on: upscale momentum
    base_weights["momentum"] <- base_weights["momentum"] * 1.5
    base_weights["low_vol"]  <- base_weights["low_vol"]  * 0.7
  }
  pmax(base_weights, 0) / sum(pmax(base_weights, 0))
}

# =============================================================================
# SECTION: FINAL DEMO
# =============================================================================

run_factor_timing_final_demo <- function() {
  set.seed(99)
  k <- 4; T <- 300
  fnames <- c("momentum", "value", "quality", "low_vol")
  F_mat  <- matrix(rnorm(T * k, 0, 0.01), T, k,
                   dimnames = list(NULL, fnames))

  cat("--- Factor Risk Parity Weights ---\n")
  Sigma <- cov(F_mat)
  w_rp  <- factor_risk_parity(Sigma)
  names(w_rp) <- fnames
  print(round(w_rp, 4))

  cat("\n--- Factor CVaR Timing (last 5 rows) ---\n")
  cvar_w <- factor_cvar_timing(F_mat)
  print(round(tail(cvar_w, 5), 4))

  cat("\n--- IC Decay Curve ---\n")
  fwd_r <- matrix(rnorm(T * k, 0, 0.01), T, k)
  ic_c  <- ic_decay_curve(F_mat, fwd_r, max_lag = 10)
  cat("IC at lags 1-10:", round(ic_c, 3), "\n")
  cat("IC half-life:", ic_half_life(ic_c), "\n")

  cat("\n--- Factor Performance Attribution ---\n")
  port_r <- F_mat %*% c(0.3, 0.2, 0.3, 0.2) + rnorm(T, 0, 0.002)
  attr_r <- factor_perf_attribution(port_r, F_mat)
  cat("Alpha:", round(attr_r$alpha * 252, 4),
      "  R2:", round(attr_r$r_squared, 4), "\n")
  cat("Factor betas:", round(attr_r$betas, 3), "\n")

  cat("\n--- Macro Overlay ---\n")
  bw <- setNames(rep(0.25, 4), fnames)
  ov <- macro_factor_overlay(bw, macro_score = 0.8)
  print(round(ov, 4))

  invisible(list(rp_weights = w_rp, ic_curve = ic_c))
}

if (interactive()) {
  ft_final <- run_factor_timing_final_demo()
}

# =============================================================================
# SECTION: FACTOR PORTFOLIO CONSTRUCTION — LONG-SHORT
# =============================================================================
# Sort assets into quintiles by factor score; go long top quintile,
# short bottom quintile.  Standard academic factor construction.

factor_quintile_portfolio <- function(scores, n_quintiles = 5) {
  n      <- length(scores)
  breaks <- quantile(scores, seq(0, 1, length.out = n_quintiles + 1))
  quintile <- cut(scores, breaks, labels = FALSE, include.lowest = TRUE)
  w <- numeric(n)
  long_n  <- sum(quintile == n_quintiles, na.rm=TRUE)
  short_n <- sum(quintile == 1,           na.rm=TRUE)
  if (long_n  > 0) w[quintile == n_quintiles] <-  1 / long_n
  if (short_n > 0) w[quintile == 1]           <- -1 / short_n
  w
}

# =============================================================================
# SECTION: FACTOR TIMING — CROSS-SECTIONAL MOMENTUM
# =============================================================================
# Rank factors by recent performance; tilt toward top-ranked factors.

cross_sectional_factor_momentum <- function(F_mat, lookback = 12) {
  T <- nrow(F_mat); k <- ncol(F_mat)
  weights <- matrix(1/k, T, k)
  for (t in seq(lookback+1, T)) {
    perf <- colSums(F_mat[(t-lookback):(t-1), , drop=FALSE])
    ranks <- rank(perf)
    w     <- ranks / sum(ranks)
    weights[t,] <- w
  }
  weights
}

# =============================================================================
# SECTION: FACTOR CORRELATION MATRIX STABILITY TEST
# =============================================================================
# Monitor factor correlation matrix for structural breaks using Frobenius norm.

factor_corr_stability <- function(F_mat, window = 60) {
  T <- nrow(F_mat)
  frobenius_change <- rep(NA_real_, T)
  C_prev <- NULL
  for (t in seq(window, T)) {
    C <- cor(F_mat[(t-window+1):t, , drop=FALSE])
    if (!is.null(C_prev)) {
      frobenius_change[t] <- sqrt(sum((C - C_prev)^2))
    }
    C_prev <- C
  }
  frobenius_change
}

# =============================================================================
# SECTION: FACTOR MODEL RESIDUAL ANALYSIS
# =============================================================================
# Examine residuals from factor model for autocorrelation (missed patterns)
# or heteroskedasticity (time-varying risk).

factor_residual_diagnostics <- function(port_rets, factor_rets) {
  df  <- data.frame(y = port_rets, as.data.frame(factor_rets))
  fit <- lm(y ~ ., data = df)
  res <- resid(fit)
  # Ljung-Box-like autocorrelation test (simplified)
  n   <- length(res)
  rho <- cor(res[-1], res[-n])
  lb_stat <- n * (n + 2) * rho^2 / (n - 1)
  list(
    autocorr   = rho,
    lb_stat    = lb_stat,
    lb_pvalue  = pchisq(lb_stat, df=1, lower.tail=FALSE),
    kurtosis   = mean((res - mean(res))^4) / (var(res)^2 + 1e-10) - 3
  )
}

if (interactive()) {
  set.seed(123)
  k <- 4; T <- 250
  F_mat <- matrix(rnorm(T*k, 0, 0.01), T, k,
                  dimnames=list(NULL, c("mom","val","qual","lvol")))
  scores <- rnorm(10)
  w_ls   <- factor_quintile_portfolio(scores)
  cat("Long-short weights sum:", round(sum(w_ls), 4), "\n")

  cs_mom <- cross_sectional_factor_momentum(F_mat, lookback=12)
  cat("CS momentum weights (last):", round(tail(cs_mom, 1), 4), "\n")

  stab <- factor_corr_stability(F_mat, window=40)
  cat("Max Frobenius change:", round(max(stab, na.rm=TRUE), 4), "\n")

  port_r <- F_mat %*% c(0.3,0.2,0.3,0.2) + rnorm(T,0,0.002)
  diag_r <- factor_residual_diagnostics(port_r, F_mat)
  cat("Residual autocorr:", round(diag_r$autocorr, 4),
      "  LB p-val:", round(diag_r$lb_pvalue, 4), "\n")
}
