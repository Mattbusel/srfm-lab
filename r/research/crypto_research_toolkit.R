# =============================================================================
# crypto_research_toolkit.R
# Research Toolkit for Crypto Quant Analysis
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Crypto alpha research follows a disciplined pipeline:
# (1) clean data, (2) find stylised facts, (3) build hypotheses, (4) test
# out-of-sample. Without this structure it is trivially easy to p-hack -- there
# are thousands of possible signal configurations and 5% of them will look
# significant by chance. This toolkit enforces discipline.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

sharpe_ratio <- function(rets, ann = 252) {
  mu <- mean(rets, na.rm=TRUE); sg <- sd(rets, na.rm=TRUE)
  if (is.na(sg) || sg < 1e-12) return(NA_real_)
  mu / sg * sqrt(ann)
}

max_drawdown <- function(eq) {
  pk <- cummax(eq); min((eq - pk) / pk, na.rm = TRUE)
}

mad_scale <- function(x) 1.4826 * median(abs(x - median(x, na.rm=TRUE)), na.rm=TRUE)

roll_mean <- function(x, w) {
  n <- length(x); out <- rep(NA,n)
  for (i in w:n) out[i] <- mean(x[(i-w+1):i], na.rm=TRUE)
  out
}

roll_sd <- function(x, w) {
  n <- length(x); out <- rep(NA,n)
  for (i in w:n) out[i] <- sd(x[(i-w+1):i], na.rm=TRUE)
  out
}

# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------

#' Remove extreme return outliers (> N*MAD from median)
clean_returns <- function(rets, n_mad = 6, replace_with = "winsorise") {
  mu  <- median(rets, na.rm=TRUE)
  sc  <- mad_scale(rets)
  lo  <- mu - n_mad * sc; hi <- mu + n_mad * sc
  if (replace_with == "winsorise") {
    rets[rets < lo] <- lo
    rets[rets > hi] <- hi
  } else if (replace_with == "na") {
    rets[rets < lo | rets > hi] <- NA
  }
  rets
}

#' Detect and remove zero-volume (halted) bars
remove_zero_vol <- function(prices, volumes) {
  mask <- volumes > 0 & !is.na(prices) & !is.na(volumes)
  list(prices = prices[mask], volumes = volumes[mask], mask = mask)
}

#' Forward-fill NA prices
ffill <- function(x) {
  for (i in which(is.na(x))) x[i] <- x[max(1, i-1)]
  x
}

#' Identify consecutive duplicate prices (likely data errors)
flag_stale_prices <- function(prices, min_streak = 3L) {
  n    <- length(prices)
  flag <- logical(n)
  streak <- 1L
  for (i in 2:n) {
    if (prices[i] == prices[i-1]) streak <- streak + 1L
    else                          streak <- 1L
    if (streak >= min_streak) flag[(i-streak+1):i] <- TRUE
  }
  flag
}

#' Full cleaning pipeline
clean_ohlcv <- function(open, high, low, close, volume) {
  n <- length(close)
  # Fix impossible OHLC
  for (t in seq_len(n)) {
    if (!is.na(high[t]) && !is.na(low[t])) {
      high[t] <- pmax(high[t], close[t])
      low[t]  <- pmin(low[t], close[t])
    }
  }
  close  <- ffill(close)
  stale  <- flag_stale_prices(close)
  rets   <- c(NA, diff(log(close)))
  rets   <- clean_returns(rets, n_mad=6)
  list(open=open, high=high, low=low, close=close,
       volume=volume, returns=rets, stale=stale,
       n_stale=sum(stale), n_cleaned=sum(is.na(rets)))
}

# ---------------------------------------------------------------------------
# 3. RETURN SEASONALITY ANALYSIS
# ---------------------------------------------------------------------------
# Test: are returns on certain hours/days/months significantly different from zero?
# Use t-test with Newey-West SE for autocorrelation.

seasonality_test <- function(returns, period_index,
                              alpha = 0.05) {
  periods <- sort(unique(period_index))
  results <- lapply(periods, function(p) {
    r_p  <- returns[period_index == p & !is.na(returns) & !is.na(period_index)]
    if (length(r_p) < 5) return(NULL)
    n    <- length(r_p)
    mu   <- mean(r_p)
    se   <- sd(r_p) / sqrt(n)
    t_   <- mu / max(se, 1e-12)
    pval <- 2 * pt(-abs(t_), df=n-1)
    data.frame(period=p, n=n, mean_ret=mu,
               se=se, t_stat=t_, p_value=pval,
               significant = pval < alpha)
  })
  do.call(rbind, Filter(Negate(is.null), results))
}

#' Hour-of-day seasonality
hourly_seasonality <- function(returns, hours) {
  seasonality_test(returns, hours)
}

#' Day-of-week seasonality (0=Sunday, 6=Saturday for POSIXlt)
dow_seasonality <- function(returns, dow) {
  seasonality_test(returns, dow)
}

#' Multiple testing correction (Benjamini-Hochberg FDR)
bh_correction <- function(p_values, alpha = 0.05) {
  n    <- length(p_values)
  rank <- rank(p_values)
  adj  <- p_values * n / rank
  adj  <- pmin(1, adj)
  list(adjusted = adj, reject = adj < alpha,
       n_reject  = sum(adj < alpha))
}

# ---------------------------------------------------------------------------
# 4. PAIRS ANALYSIS: COINTEGRATION
# ---------------------------------------------------------------------------
# Engle-Granger two-step: (1) OLS regression, (2) ADF on residuals.

#' Augmented Dickey-Fuller test (simplified, no trend)
adf_test <- function(x, lags = 1L) {
  n  <- length(x)
  dx <- diff(x)
  y  <- dx[-(1:lags)]
  X  <- cbind(x[(lags+1):(n-1)])   # lagged level
  if (lags > 0) {
    for (l in seq_len(lags)) X <- cbind(X, dx[(lags-l+1):(n-l-1)])
  }
  X  <- cbind(1, X)
  b  <- tryCatch(solve(t(X)%*%X + diag(1e-8, ncol(X)), t(X)%*%y),
                  error=function(e) rep(0,ncol(X)))
  res <- y - X %*% b
  se  <- sqrt(sum(res^2) / (nrow(X) - ncol(X)) * solve(t(X)%*%X + diag(1e-8,ncol(X)))[2,2])
  t_adf <- b[2] / max(se, 1e-12)
  # Critical values (MacKinnon approx, no constant): -3.43 @ 1%, -2.86 @ 5%, -2.57 @ 10%
  pval_approx <- ifelse(t_adf < -3.43, 0.01, ifelse(t_adf < -2.86, 0.05,
                  ifelse(t_adf < -2.57, 0.10, 0.20)))
  list(t_stat = t_adf, p_value = pval_approx, reject_null = t_adf < -2.86)
}

#' Cointegration test (Engle-Granger)
cointegration_test <- function(y1, y2) {
  X   <- cbind(1, y2)
  b   <- solve(t(X)%*%X + diag(1e-8,2), t(X)%*%y1)
  spread <- y1 - b[1] - b[2] * y2
  adf <- adf_test(spread, lags=1L)
  half_life <- -log(2) / log(max(1 + b[2], 1e-6))   # AR(1) mean reversion
  list(beta = b[2], intercept = b[1],
       spread = spread, adf = adf,
       cointegrated = adf$reject_null,
       half_life = abs(half_life))
}

#' Z-score of spread for entry signals
spread_zscore <- function(spread, window = 60L) {
  mu <- roll_mean(spread, window)
  sg <- roll_sd(spread, window)
  (spread - mu) / pmax(sg, 1e-8)
}

#' Simple pairs trading P&L
pairs_pnl <- function(z_score, threshold = 2.0, exit = 0.5, tc = 0.001) {
  n  <- length(z_score)
  pos <- 0L; rets <- numeric(n)
  for (t in seq_len(n-1)) {
    if (pos == 0L) {
      if (z_score[t] > threshold)  pos <- -1L   # spread too wide: short spread
      if (z_score[t] < -threshold) pos <-  1L   # spread too narrow: long spread
      if (pos != 0L) rets[t] <- -tc
    } else {
      dz <- z_score[t+1] - z_score[t]
      rets[t] <- pos * dz * 0.01   # scaled
      if ((pos == 1L && z_score[t] > -exit) ||
          (pos == -1L && z_score[t] < exit)) {
        rets[t] <- rets[t] - tc; pos <- 0L
      }
    }
  }
  list(rets=rets, equity=cumprod(1+rets),
       sharpe=sharpe_ratio(rets[rets!=0]))
}

# ---------------------------------------------------------------------------
# 5. VOLATILITY FORECASTING ACCURACY
# ---------------------------------------------------------------------------

#' QLIKE loss: penalises over-estimating more than under-estimating
qlike_loss <- function(realised, forecast) {
  mean(realised / pmax(forecast, 1e-8) - log(realised / pmax(forecast, 1e-8)) - 1, na.rm=TRUE)
}

#' MSE / RMSE of vol forecast
vol_forecast_rmse <- function(realised, forecast) {
  sqrt(mean((realised - forecast)^2, na.rm=TRUE))
}

#' Diebold-Mariano for vol forecasts
dm_vol_test <- function(realised, forecast_a, forecast_b) {
  loss_a <- (realised - forecast_a)^2
  loss_b <- (realised - forecast_b)^2
  d  <- loss_a - loss_b
  mu <- mean(d, na.rm=TRUE)
  se <- sd(d, na.rm=TRUE) / sqrt(sum(!is.na(d)))
  t_ <- mu / max(se, 1e-12)
  pval <- 2 * pt(-abs(t_), df=sum(!is.na(d))-1)
  data.frame(mean_diff=mu, t_stat=t_, p_value=pval)
}

# ---------------------------------------------------------------------------
# 6. SIGNAL COMBINATION
# ---------------------------------------------------------------------------
# Combine multiple signals (BH + on-chain + sentiment) using:
# (a) Equal weight, (b) IC-weighted, (c) Risk-parity over signal returns

combine_signals_ew <- function(signals_matrix) {
  rowMeans(signals_matrix, na.rm=TRUE)
}

combine_signals_ic_weighted <- function(signals_matrix, ic_vec) {
  # Weight signals by their historical IC
  ic_pos  <- pmax(ic_vec, 0)
  if (sum(ic_pos) < 1e-8) return(combine_signals_ew(signals_matrix))
  w <- ic_pos / sum(ic_pos)
  signals_matrix %*% w
}

combine_signals_risk_parity <- function(signal_rets_matrix) {
  vols <- apply(signal_rets_matrix, 2, sd, na.rm=TRUE)
  vols[vols < 1e-8] <- 1
  w    <- 1 / vols; w <- w / sum(w)
  signal_rets_matrix %*% w
}

# ---------------------------------------------------------------------------
# 7. RESEARCH HYPOTHESIS PIPELINE
# ---------------------------------------------------------------------------
# Step 1: Idea (verbal description)
# Step 2: IC test (signal vs 1-bar forward return)
# Step 3: Paper portfolio (fractional position = signal)
# Step 4: Full backtest

ic_test <- function(signal, forward_return) {
  valid <- !is.na(signal) & !is.na(forward_return)
  if (sum(valid) < 10) return(list(IC=NA, t_stat=NA, p_val=NA))
  ic_val <- cor(rank(signal[valid]), rank(forward_return[valid]))
  n      <- sum(valid)
  t_     <- ic_val * sqrt(n - 2) / sqrt(1 - ic_val^2)
  pval   <- 2 * pt(-abs(t_), df=n-2)
  list(IC=ic_val, t_stat=t_, p_val=pval, n=n,
       ICIR = ic_val / (sd(c(1,-1)) + 1e-8))   # approximation for demo
}

paper_portfolio <- function(signal, forward_returns, tc = 0) {
  pos   <- signal / (max(abs(signal), na.rm=TRUE) + 1e-8)   # normalise to [-1,1]
  rets  <- c(0, pos[-length(pos)] * forward_returns[-1]) - tc * abs(c(0, diff(pos)))
  list(returns=rets, equity=cumprod(1+rets),
       sharpe=sharpe_ratio(rets[rets!=0]))
}

# ---------------------------------------------------------------------------
# 8. P-HACKING DETECTION
# ---------------------------------------------------------------------------
# Simulate the null distribution of Sharpe ratios from random signals.
# Compare observed SR to null: if not > 95th pctile, likely noise.

null_sharpe_distribution <- function(n_obs = 252L, n_trials = 1000L,
                                      ann = 252L, seed = 42L) {
  set.seed(seed)
  sharpes <- numeric(n_trials)
  for (i in seq_len(n_trials)) {
    signal <- runif(n_obs, -1, 1)   # pure noise signal
    rets   <- signal * rnorm(n_obs, 0, 0.01)
    sharpes[i] <- sharpe_ratio(rets, ann)
  }
  sharpes
}

inflation_adjusted_pvalue <- function(observed_sharpe, null_dist) {
  mean(null_dist >= observed_sharpe)
}

# ---------------------------------------------------------------------------
# 9. REGIME-CONDITIONAL SIGNAL EVALUATION
# ---------------------------------------------------------------------------

signal_by_regime <- function(signal, forward_returns, regime) {
  regimes <- sort(unique(regime[!is.na(regime)]))
  results <- lapply(regimes, function(r) {
    mask  <- regime == r & !is.na(signal) & !is.na(forward_returns)
    s_r   <- signal[mask]
    fr_r  <- forward_returns[mask]
    ic_r  <- ic_test(s_r, fr_r)
    pos_r <- paper_portfolio(s_r, fr_r)
    data.frame(regime=r, n=sum(mask), IC=ic_r$IC,
               IC_t=ic_r$t_stat, IC_pval=ic_r$p_val,
               sharpe=pos_r$sharpe)
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 10. HALF-LIFE ANALYSIS
# ---------------------------------------------------------------------------

#' AR(1) half-life of mean reversion
ar1_halflife <- function(x) {
  n   <- length(x); x_lag <- x[-n]; x_cur <- x[-1]
  b   <- cov(x_lag, x_cur) / var(x_lag)
  -log(2) / log(abs(b))
}

#' Hurst exponent (R/S method)
hurst_exponent <- function(x, max_lag = NULL) {
  n <- length(x)
  if (is.null(max_lag)) max_lag <- floor(n / 4)
  lags <- unique(floor(exp(seq(log(10), log(max_lag), length.out=20))))
  rs_vals <- numeric(length(lags))
  for (i in seq_along(lags)) {
    k  <- lags[i]
    rs_k <- sapply(seq(1, n - k, by=k), function(s) {
      sub  <- x[s:(s+k-1)]
      mean_sub <- mean(sub); cum_dev <- cumsum(sub - mean_sub)
      r_s  <- max(cum_dev) - min(cum_dev)
      sg   <- sd(sub)
      if (sg < 1e-10) return(NA) else r_s / sg
    })
    rs_vals[i] <- mean(rs_k, na.rm=TRUE)
  }
  valid <- !is.na(rs_vals) & rs_vals > 0 & lags > 0
  if (sum(valid) < 3) return(0.5)
  X   <- cbind(1, log(lags[valid]))
  b   <- tryCatch(solve(t(X)%*%X, t(X)%*%log(rs_vals[valid])),
                   error=function(e) c(0,0.5))
  b[2]   # Hurst exponent
}

# ---------------------------------------------------------------------------
# 11. VOLATILITY CLUSTERING ANALYSIS
# ---------------------------------------------------------------------------

vol_clustering_test <- function(returns, lags = 1:10) {
  abs_rets <- abs(returns)
  sq_rets  <- returns^2
  n        <- length(returns)
  results  <- lapply(lags, function(lag) {
    valid <- seq_len(n - lag)
    ac_abs <- cor(abs_rets[valid], abs_rets[valid + lag], use="complete.obs")
    ac_sq  <- cor(sq_rets[valid], sq_rets[valid + lag], use="complete.obs")
    data.frame(lag=lag, ac_abs=ac_abs, ac_sq=ac_sq)
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 12. MAIN DEMO
# ---------------------------------------------------------------------------

run_research_toolkit_demo <- function() {
  cat("=== Crypto Research Toolkit Demo ===\n\n")
  set.seed(42)
  T_ <- 800L

  # Simulate dirty crypto data
  raw_close <- cumprod(1 + rnorm(T_, 0.0005, 0.03)) * 10000
  raw_vol   <- rpois(T_, 1000)
  raw_close[300:305] <- raw_close[299]   # stale prices
  raw_close[500] <- raw_close[499] * 5   # spike error

  cat("--- 1. Data Cleaning ---\n")
  cleaned <- clean_ohlcv(raw_close, raw_close*1.001, raw_close*0.999,
                          raw_close, raw_vol)
  cat(sprintf("  Stale bars detected: %d  |  Cleaned returns: %d\n",
              cleaned$n_stale, cleaned$n_cleaned))

  rets <- cleaned$returns[!is.na(cleaned$returns)]

  cat("\n--- 2. Return Seasonality ---\n")
  # Simulate hour-of-day index
  hours  <- rep(0:23, ceiling(T_/24))[1:length(rets)]
  h_seas <- hourly_seasonality(rets, hours)
  n_sig  <- sum(h_seas$significant, na.rm=TRUE)
  bh_adj <- bh_correction(h_seas$p_value)
  cat(sprintf("  Significant hours (raw): %d / 24  |  After BH correction: %d\n",
              n_sig, bh_adj$n_reject))

  cat("\n--- 3. Cointegration / Pairs Analysis ---\n")
  y1 <- cumsum(rnorm(T_, 0.001, 0.02))
  y2 <- y1 + cumsum(rnorm(T_, 0, 0.005)) * 0.3
  coint <- cointegration_test(y1, y2)
  cat(sprintf("  Cointegrated: %s  |  ADF t-stat: %.3f  |  Half-life: %.1f bars\n",
              coint$cointegrated, coint$adf$t_stat, coint$half_life))
  if (coint$cointegrated) {
    z  <- spread_zscore(coint$spread, window=60L)
    pp <- pairs_pnl(z[!is.na(z)])
    cat(sprintf("  Pairs Sharpe: %.3f\n", pp$sharpe))
  }

  cat("\n--- 4. Volatility Forecast Accuracy ---\n")
  # Simple EWMA forecasts
  ewma_fast <- rep(NA, length(rets))
  ewma_slow <- rep(NA, length(rets))
  ewma_fast[1] <- rets[1]^2; ewma_slow[1] <- rets[1]^2
  for (i in 2:length(rets)) {
    ewma_fast[i] <- 0.06 * rets[i]^2 + 0.94 * ewma_fast[i-1]
    ewma_slow[i] <- 0.02 * rets[i]^2 + 0.98 * ewma_slow[i-1]
  }
  realised <- rets^2
  qlike_f <- qlike_loss(realised[20:end<-length(rets)], ewma_fast[20:end])
  qlike_s <- qlike_loss(realised[20:end], ewma_slow[20:end])
  cat(sprintf("  QLIKE: EWMA_fast=%.4f  EWMA_slow=%.4f\n", qlike_f, qlike_s))
  dm_v <- dm_vol_test(realised[20:length(rets)], ewma_fast[20:length(rets)], ewma_slow[20:length(rets)])
  cat(sprintf("  DM test p-value: %.4f\n", dm_v$p_value))

  cat("\n--- 5. Signal Combination ---\n")
  sig1 <- tanh(rets / mad_scale(rets))   # momentum-like
  sig2 <- -tanh(rets / mad_scale(rets))  # mean-reversion-like
  sig3 <- rnorm(length(rets), 0, 0.5)   # noise
  sigs <- cbind(sig1, sig2, sig3)
  ic_vec <- c(ic_test(sig1[-1], rets[-length(rets)])$IC,
               ic_test(sig2[-1], rets[-length(rets)])$IC,
               ic_test(sig3[-1], rets[-length(rets)])$IC)
  ic_vec[is.na(ic_vec)] <- 0
  cat("  Signal ICs:", round(ic_vec, 4), "\n")
  combo_ic  <- combine_signals_ic_weighted(sigs, ic_vec)
  combo_ew  <- combine_signals_ew(sigs)
  cat(sprintf("  IC-weighted combo IC: %.4f  |  EW combo IC: %.4f\n",
              ic_test(combo_ic[-1], rets[-length(rets)])$IC,
              ic_test(combo_ew[-1], rets[-length(rets)])$IC))

  cat("\n--- 6. IC Test Pipeline ---\n")
  fwd_ret <- c(rets[-1], NA)
  ic_res  <- ic_test(sig1, fwd_ret)
  cat(sprintf("  Momentum signal IC=%.4f  t=%.3f  p=%.4f\n",
              ic_res$IC, ic_res$t_stat, ic_res$p_val))
  pp <- paper_portfolio(sig1, rets)
  cat(sprintf("  Paper portfolio Sharpe: %.3f\n", pp$sharpe))

  cat("\n--- 7. P-Hacking Detection ---\n")
  null_dist <- null_sharpe_distribution(n_obs=length(rets), n_trials=500L)
  obs_sr    <- sharpe_ratio(rets[rets!=0])
  adj_pval  <- inflation_adjusted_pvalue(obs_sr, null_dist)
  cat(sprintf("  Observed SR: %.3f  |  Null 95th pctile: %.3f  |  Adj p-val: %.3f\n",
              obs_sr, quantile(null_dist, 0.95), adj_pval))

  cat("\n--- 8. Regime-Conditional Signal Evaluation ---\n")
  vol_roll <- roll_sd(rets, 30L)
  regime   <- ifelse(is.na(vol_roll), 1L,
                      ifelse(vol_roll > median(vol_roll, na.rm=TRUE), 2L, 1L))
  reg_eval <- signal_by_regime(sig1, fwd_ret, regime)
  print(reg_eval[, c("regime","n","IC","IC_pval","sharpe")])

  cat("\n--- 9. Half-Life & Hurst Exponent ---\n")
  hl <- ar1_halflife(raw_close[!is.na(raw_close)])
  h  <- hurst_exponent(raw_close[!is.na(raw_close)])
  cat(sprintf("  Half-life (AR1): %.1f bars  |  Hurst exponent: %.3f\n", hl, h))
  cat(sprintf("  Hurst interpretation: %s\n",
              ifelse(h < 0.45, "Mean-reverting", ifelse(h > 0.55, "Trending", "Random walk"))))

  cat("\n--- 10. Volatility Clustering ---\n")
  vc <- vol_clustering_test(rets[1:200], lags=1:5)
  cat("  Abs-return autocorrelation at lags 1-5:\n")
  cat("  ", round(vc$ac_abs, 4), "\n")

  cat("\nDone.\n")
  invisible(list(cleaned=cleaned, coint=coint, ic_res=ic_res,
                 null_dist=null_dist, reg_eval=reg_eval))
}

if (interactive()) {
  rtk_results <- run_research_toolkit_demo()
}

# ---------------------------------------------------------------------------
# 13. CROSS-VALIDATION FOR FINANCIAL TIME SERIES
# ---------------------------------------------------------------------------
# Purged/embargo k-fold: prevents data leakage by purging observations
# near the training/test boundary and embedding an embargo gap.

purged_kfold_cv <- function(n, k=5L, embargo_pct=0.01) {
  fold_size <- floor(n / k)
  embargo   <- max(1L, as.integer(n * embargo_pct))
  folds     <- vector("list", k)
  for (f in seq_len(k)) {
    test_start <- (f-1)*fold_size + 1L
    test_end   <- min(f*fold_size, n)
    train_idx  <- setdiff(seq_len(n),
                           (test_start - embargo):(test_end + embargo))
    train_idx  <- train_idx[train_idx >= 1 & train_idx <= n]
    folds[[f]] <- list(train=train_idx, test=test_start:test_end)
  }
  folds
}

#' Run signal IC using purged k-fold
purged_cv_ic <- function(signal, forward_return, k=5L, embargo_pct=0.01) {
  n     <- length(signal)
  folds <- purged_kfold_cv(n, k, embargo_pct)
  ic_oos <- sapply(folds, function(f) {
    s_test  <- signal[f$test]; r_test <- forward_return[f$test]
    valid   <- !is.na(s_test) & !is.na(r_test)
    if (sum(valid) < 5) return(NA)
    cor(rank(s_test[valid]), rank(r_test[valid]))
  })
  list(ic_by_fold=ic_oos, mean_ic=mean(ic_oos, na.rm=TRUE),
       sd_ic=sd(ic_oos, na.rm=TRUE), ICIR=mean(ic_oos,na.rm=TRUE)/max(sd(ic_oos,na.rm=TRUE),1e-8))
}

# ---------------------------------------------------------------------------
# 14. STRUCTURAL BREAK DETECTION (CHOW TEST)
# ---------------------------------------------------------------------------

chow_test <- function(y, x, break_point) {
  n  <- length(y)
  if (break_point < 3 || break_point > n-3) return(list(F_stat=NA, p_value=NA))
  X   <- cbind(1, x)
  fit_full  <- lm(y ~ x)
  rss_full  <- sum(residuals(fit_full)^2)
  fit1 <- lm(y[1:break_point] ~ x[1:break_point])
  fit2 <- lm(y[(break_point+1):n] ~ x[(break_point+1):n])
  rss1 <- sum(residuals(fit1)^2)
  rss2 <- sum(residuals(fit2)^2)
  rss_r   <- rss1 + rss2
  k       <- 2   # number of parameters
  F_stat  <- ((rss_full - rss_r)/k) / (rss_r/(n-2*k))
  p_val   <- pf(F_stat, k, n-2*k, lower.tail=FALSE)
  list(F_stat=F_stat, p_value=p_val, reject=p_val < 0.05)
}

#' Scan for structural breaks
scan_breaks <- function(y, x, min_frac=0.15) {
  n   <- length(y)
  lo  <- as.integer(n * min_frac)
  hi  <- n - lo
  results <- lapply(lo:hi, function(bp) {
    ct <- chow_test(y, x, bp)
    data.frame(break_point=bp, F_stat=ct$F_stat, p_value=ct$p_value)
  })
  df <- do.call(rbind, results)
  df[which.max(df$F_stat),]
}

# ---------------------------------------------------------------------------
# 15. SIGNAL DECAY ANALYSIS (PREDICTIVE HORIZON)
# ---------------------------------------------------------------------------

signal_decay <- function(signal, returns, max_horizon=21L) {
  n   <- length(signal)
  ics <- sapply(seq_len(max_horizon), function(h) {
    fwd <- c(returns[(h+1):n], rep(NA,h))
    valid <- !is.na(signal) & !is.na(fwd)
    if (sum(valid) < 5) return(NA)
    cor(rank(signal[valid]), rank(fwd[valid]))
  })
  half_life <- which(abs(ics) < abs(ics[1]) / 2)[1]
  list(ics=ics, half_life=half_life,
       decay_rate=if(max_horizon>1) -log(abs(ics[max_horizon])/max(abs(ics[1]),1e-8))/max_horizon else NA)
}

# ---------------------------------------------------------------------------
# 16. EXTENDED RESEARCH TOOLKIT DEMO
# ---------------------------------------------------------------------------

run_research_toolkit_extended_demo <- function() {
  cat("=== Research Toolkit Extended Demo ===\n\n")
  set.seed(42); n <- 500L
  returns <- c(rnorm(250,0.001,0.02), rnorm(250,-0.001,0.03))
  signal  <- tanh(roll_mean(returns,10L)/pmax(roll_sd(returns,10L),1e-6))
  signal[is.na(signal)] <- 0
  fwd_ret <- c(returns[-1], NA)

  cat("--- Purged K-Fold CV (IC) ---\n")
  pcv <- purged_cv_ic(signal, fwd_ret, k=5L)
  cat(sprintf("  Mean OOS IC: %.4f  SD: %.4f  ICIR: %.4f\n",
              pcv$mean_ic, pcv$sd_ic, pcv$ICIR))

  cat("\n--- Chow Structural Break Test ---\n")
  x <- rnorm(n); y <- c(0.5*x[1:250] + rnorm(250,0,0.5), 2*x[251:500] + rnorm(250,0,0.5))
  sb <- scan_breaks(y, x)
  cat(sprintf("  Strongest break at bar %d  F=%.3f  p=%.4f\n",
              sb$break_point, sb$F_stat, sb$p_value))

  cat("\n--- Signal Decay ---\n")
  sd_res <- signal_decay(signal[!is.na(signal)], returns[!is.na(signal)], max_horizon=10L)
  cat("  IC at horizons 1-10:", round(sd_res$ics, 4), "\n")
  cat(sprintf("  IC half-life: %s bars\n",
              ifelse(is.na(sd_res$half_life), "NA", sd_res$half_life)))

  cat("\n--- Hurst Exponent for Returns ---\n")
  h_rets <- hurst_exponent(returns)
  cat(sprintf("  Hurst exponent of returns: %.4f\n", h_rets))
  h_abs  <- hurst_exponent(abs(returns))
  cat(sprintf("  Hurst of |returns| (vol clustering): %.4f\n", h_abs))

  invisible(list(pcv=pcv, sb=sb, sd_res=sd_res))
}

if (interactive()) {
  rtk_ext <- run_research_toolkit_extended_demo()
}

# =============================================================================
# SECTION: VARIANCE RATIO TEST (Lo-MacKinlay)
# =============================================================================
# Tests the random walk hypothesis: if returns are RW, variance of k-period
# returns should equal k times the variance of 1-period returns.

variance_ratio_test <- function(prices, k = 5) {
  n    <- length(prices)
  ret1 <- diff(log(prices))
  retk <- diff(log(prices[seq(1, n, by = k)]))
  # Unbiased variance estimates
  mu   <- mean(ret1)
  s1   <- sum((ret1 - mu)^2) / (length(ret1) - 1)
  sk   <- sum((retk - mu*k)^2) / (length(retk) - 1)
  vr   <- sk / (k * s1)
  # Asymptotic z-statistic (homoskedastic version)
  delta <- 2*(2*k - 1)*(k - 1) / (3*k * length(ret1))
  z_stat <- (vr - 1) / sqrt(delta)
  p_val  <- 2 * pnorm(-abs(z_stat))
  list(VR = vr, z = z_stat, p_value = p_val, k = k)
}

# =============================================================================
# SECTION: MULTIPLE VARIANCE RATIO (MVR) TEST
# =============================================================================
# Joint test across multiple holding periods using max |z| statistic.

multiple_vr_test <- function(prices, k_vec = c(2, 4, 8, 16)) {
  tests <- lapply(k_vec, function(k) variance_ratio_test(prices, k))
  z_max <- max(abs(sapply(tests, `[[`, "z")))
  # Approximate p-value via simulation (crude)
  list(
    individual = tests,
    max_z      = z_max,
    # Bonferroni corrected
    p_bonf     = min(1, min(sapply(tests, `[[`, "p_value")) * length(k_vec))
  )
}

# =============================================================================
# SECTION: CROSS-ASSET LEAD-LAG DETECTION
# =============================================================================
# Compute CCF at multiple lags to identify which asset leads price discovery.

lead_lag_analysis <- function(r1, r2, max_lag = 10) {
  n    <- min(length(r1), length(r2))
  r1   <- r1[seq_len(n)]; r2 <- r2[seq_len(n)]
  lags <- seq(-max_lag, max_lag)
  ccf_vals <- numeric(length(lags))
  for (i in seq_along(lags)) {
    lag <- lags[i]
    if (lag >= 0) ccf_vals[i] <- cor(r1[1:(n-lag)], r2[(lag+1):n])
    else          ccf_vals[i] <- cor(r1[(-lag+1):n], r2[1:(n+lag)])
  }
  best_lag <- lags[which.max(abs(ccf_vals))]
  list(lags = lags, ccf = ccf_vals, best_lag = best_lag,
       max_ccf = max(abs(ccf_vals)))
}

# =============================================================================
# SECTION: ROLLING CORRELATION REGIME DETECTION
# =============================================================================
# Flag periods where cross-asset correlations spike (risk-off contagion).

rolling_corr_regime <- function(r_mat, window = 30, high_corr_thresh = 0.6) {
  T  <- nrow(r_mat); n <- ncol(r_mat)
  rc <- rep(NA_real_, T)
  for (t in seq(window, T)) {
    sub <- r_mat[(t-window+1):t, , drop=FALSE]
    C   <- cor(sub)
    # Average off-diagonal absolute correlation
    rc[t] <- (sum(abs(C)) - n) / (n*(n-1))
  }
  regime <- ifelse(rc > high_corr_thresh, "risk_off", "normal")
  list(rolling_corr = rc, regime = regime)
}

# =============================================================================
# SECTION: SIGNAL COMBINATION WITH RIDGE REGRESSION
# =============================================================================
# Combine multiple alpha signals via ridge regression to avoid overfitting.
# Ridge shrinks noisy signals toward zero.

ridge_signal_combination <- function(signals, fwd_ret, lambda = 0.01) {
  # signals: T x k matrix, fwd_ret: T-vector
  n  <- nrow(signals); k <- ncol(signals)
  X  <- cbind(1, signals)
  # Ridge: beta = (X'X + lambda*I)^{-1} X'y
  XtX <- t(X) %*% X
  diag(XtX) <- diag(XtX) + lambda * c(0, rep(1, k))  # don't penalise intercept
  Xty <- t(X) %*% fwd_ret
  beta <- solve(XtX, Xty)
  list(beta = beta[-1], intercept = beta[1],
       fitted = as.vector(X %*% beta),
       r2    = cor(as.vector(X %*% beta), fwd_ret)^2)
}

# =============================================================================
# SECTION: INFORMATION COEFFICIENT STABILITY
# =============================================================================
# Rolling IC to detect signal decay or structural breaks.

rolling_ic <- function(signal, fwd_ret, window = 60) {
  n   <- min(length(signal), length(fwd_ret))
  ic  <- rep(NA_real_, n)
  for (t in seq(window, n)) {
    s <- signal[(t-window+1):t]
    r <- fwd_ret[(t-window+1):t]
    if (sd(s) < 1e-9 || sd(r) < 1e-9) next
    ic[t] <- cor(s, r, method = "spearman")
  }
  ic
}

ic_t_stat <- function(ic_series) {
  ic_clean <- ic_series[!is.na(ic_series)]
  n <- length(ic_clean)
  if (n < 2) return(NA_real_)
  mean(ic_clean) / (sd(ic_clean) / sqrt(n))
}

# =============================================================================
# SECTION: EXCHANGE FLOW ANOMALY DETECTION
# =============================================================================
# On-chain exchange inflows often precede sell-offs.
# Detect anomalous inflow spikes using z-score threshold.

detect_flow_anomalies <- function(flow_series, window = 30, z_thresh = 2.0) {
  n   <- length(flow_series)
  out <- rep(FALSE, n)
  for (t in seq(window+1, n)) {
    sub  <- flow_series[(t-window):(t-1)]
    mu   <- mean(sub); sig <- sd(sub)
    if (sig < 1e-9) next
    out[t] <- abs(flow_series[t] - mu) / sig > z_thresh
  }
  out
}

# =============================================================================
# SECTION: FINAL DEMO
# =============================================================================

run_research_toolkit_final_demo <- function() {
  set.seed(42)
  T <- 300
  prices <- cumprod(c(30000, exp(rnorm(T-1, 0, 0.02))))

  cat("--- Variance Ratio Test (k=5) ---\n")
  vrt <- variance_ratio_test(prices, k = 5)
  cat(sprintf("VR=%.4f  z=%.3f  p=%.4f\n", vrt$VR, vrt$z, vrt$p_value))

  cat("\n--- Multiple VR Test ---\n")
  mvr <- multiple_vr_test(prices)
  cat("Max |z|:", round(mvr$max_z, 3),
      "  Bonferroni p:", round(mvr$p_bonf, 4), "\n")

  cat("\n--- Lead-Lag Analysis ---\n")
  r1 <- diff(log(prices))
  r2 <- c(r1[-1], rnorm(1, 0, 0.02))  # r2 lags r1 by 1
  ll <- lead_lag_analysis(r1, r2, max_lag = 5)
  cat("Best lag:", ll$best_lag, "  Max CCF:", round(ll$max_ccf, 4), "\n")

  cat("\n--- Rolling Corr Regime ---\n")
  r_mat <- cbind(r1, r2, rnorm(length(r1), 0, 0.02))
  rcr   <- rolling_corr_regime(r_mat, window = 30)
  cat("% risk-off periods:", round(mean(rcr$regime == "risk_off", na.rm=TRUE)*100, 1), "%\n")

  cat("\n--- Ridge Signal Combination ---\n")
  sigs  <- matrix(rnorm(T*3), T, 3)
  fwd_r <- rnorm(T, 0, 0.01)
  rc    <- ridge_signal_combination(sigs, fwd_r)
  cat("Betas:", round(rc$beta, 4), "  R2:", round(rc$r2, 4), "\n")

  cat("\n--- IC T-Stat ---\n")
  ic  <- rolling_ic(sigs[,1], fwd_r, window = 40)
  cat("IC T-stat:", round(ic_t_stat(ic), 3), "\n")

  invisible(list(vrt=vrt, lead_lag=ll, ridge=rc))
}

if (interactive()) {
  rtk_final <- run_research_toolkit_final_demo()
}

# =============================================================================
# SECTION: MOMENTUM SIGNAL CONSTRUCTION (STANDARD)
# =============================================================================
# Cross-sectional momentum: rank assets by trailing 12-1 month return.
# Skip the most recent month to avoid short-term reversal contamination.

momentum_12_1 <- function(price_mat, skip_month = 21, lookback = 252) {
  # price_mat: T x n, returns momentum score for each asset
  T <- nrow(price_mat); n <- ncol(price_mat)
  scores <- matrix(NA_real_, T, n)
  for (t in seq(lookback + skip_month + 1, T)) {
    ret_lb  <- price_mat[t - skip_month, ] / price_mat[t - lookback, ] - 1
    scores[t, ] <- rank(ret_lb) / n  # cross-sectional rank
  }
  scores
}

# =============================================================================
# SECTION: EVENT STUDY — ABNORMAL RETURNS
# =============================================================================
# Measure abnormal returns around binary events (protocol upgrades, halvings).
# Estimate normal returns from pre-event window, compute AR in event window.

event_study_ar <- function(returns, event_dates, pre_window = 60,
                            post_window = 10) {
  n_events <- length(event_dates)
  ar_mat   <- matrix(NA_real_, n_events, post_window)
  for (i in seq_len(n_events)) {
    t0  <- event_dates[i]
    pre <- returns[(t0 - pre_window):(t0 - 1)]
    if (any(is.na(pre))) next
    mu_normal <- mean(pre)
    for (h in seq_len(post_window)) {
      if (t0 + h > length(returns)) break
      ar_mat[i, h] <- returns[t0 + h] - mu_normal
    }
  }
  # Car: cumulative abnormal return
  car <- colMeans(ar_mat, na.rm=TRUE)
  t_stats <- car / (apply(ar_mat, 2, sd, na.rm=TRUE) /
                    sqrt(colSums(!is.na(ar_mat))))
  list(ar=ar_mat, car=car, t_stats=t_stats)
}

# =============================================================================
# SECTION: SEASONALITY — DAY-OF-WEEK AND MONTH EFFECTS
# =============================================================================

dow_effect_test <- function(returns, timestamps) {
  # timestamps: integer day-of-week (1=Mon, ..., 7=Sun)
  mu_by_dow  <- tapply(returns, timestamps, mean)
  sd_by_dow  <- tapply(returns, timestamps, sd)
  n_by_dow   <- tapply(returns, timestamps, length)
  t_stats    <- mu_by_dow / (sd_by_dow / sqrt(n_by_dow))
  list(mean=mu_by_dow, t_stat=t_stats)
}

# =============================================================================
# SECTION: CORRELATION BETWEEN SIGNALS (AVOID OVERCROWDING)
# =============================================================================
# When alpha signals are highly correlated they add little diversification.
# Measure pairwise signal correlations to detect crowding.

signal_correlation_matrix <- function(signal_mat) {
  # signal_mat: T x k
  C <- cor(signal_mat, use="pairwise.complete.obs")
  # Average off-diagonal absolute correlation
  n  <- ncol(C)
  avg_cor <- (sum(abs(C)) - n) / (n*(n-1))
  list(corr_matrix=C, avg_abs_cor=avg_cor)
}

if (interactive()) {
  set.seed(55)
  T <- 300; n <- 5
  prices <- matrix(cumprod(c(1, exp(rnorm(T*n-1,0,0.02)))), T, n)
  mom    <- momentum_12_1(prices)
  cat("Momentum scores (last row):", round(tail(mom, 1), 3), "\n")

  r <- diff(log(prices[,1]))
  ev_dates <- c(100, 150, 200)
  es  <- event_study_ar(r, ev_dates, pre_window=30, post_window=5)
  cat("CAR over 5 days:", round(es$car * 100, 3), "%\n")

  dow <- dow_effect_test(r, sample(1:7, length(r), replace=TRUE))
  cat("Highest DOW return:", round(max(dow$mean)*100, 4), "%\n")

  sigs <- matrix(rnorm(T*3), T, 3)
  sc   <- signal_correlation_matrix(sigs)
  cat("Avg signal abs-cor:", round(sc$avg_abs_cor, 4), "\n")
}
