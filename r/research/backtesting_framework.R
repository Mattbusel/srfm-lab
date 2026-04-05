# =============================================================================
# backtesting_framework.R
# Comprehensive Backtest Framework for Crypto Strategies
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: A good backtest is not just an equity curve -- it must
# account for market impact, financing costs, realistic fills, and statistical
# significance. This framework provides event-driven bar-by-bar simulation,
# multiple sizing methods, walk-forward validation, and rigorous statistical
# tests to guard against overfitting.
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
  pk <- cummax(eq); min((eq - pk) / pk, na.rm=TRUE)
}

calmar_ratio <- function(rets, ann = 252) {
  eq <- cumprod(1 + rets)
  ann_ret <- mean(rets, na.rm=TRUE) * ann
  mdd <- abs(max_drawdown(eq))
  if (mdd < 1e-8) return(NA) else ann_ret / mdd
}

sortino_ratio <- function(rets, mar = 0, ann = 252) {
  mu <- mean(rets, na.rm=TRUE)
  downside <- rets[rets < mar] - mar
  dsr <- sqrt(mean(downside^2, na.rm=TRUE))
  if (dsr < 1e-8) return(NA) else (mu - mar/ann) / dsr * sqrt(ann)
}

# ---------------------------------------------------------------------------
# 2. BAR DATA STRUCTURE
# ---------------------------------------------------------------------------

#' Create a bar data list (OHLCV + returns)
make_bars <- function(open, high, low, close, volume) {
  n   <- length(close)
  ret <- c(NA, diff(log(close)))
  list(n=n, open=open, high=high, low=low, close=close,
       volume=volume, ret=ret)
}

#' Simulate synthetic OHLCV bars
simulate_bars <- function(n = 1000L, mu_ann = 0.15, sigma_ann = 0.70,
                           seed = 42L) {
  set.seed(seed)
  dt    <- 1/252
  mu_d  <- mu_ann * dt
  sig_d <- sigma_ann * sqrt(dt)
  close <- cumprod(1 + rnorm(n, mu_d, sig_d)) * 10000
  # OHLC
  intrabar_noise <- abs(rnorm(n, 0, sig_d * 0.5))
  high  <- close * (1 + intrabar_noise)
  low   <- close * (1 - intrabar_noise)
  open  <- c(close[1], close[-n])
  vol   <- rpois(n, 1000) + 100
  make_bars(open, high, low, close, vol)
}

# ---------------------------------------------------------------------------
# 3. SIGNAL INTERFACE
# ---------------------------------------------------------------------------
# Signals are functions: (bars, t) -> numeric in [-1, 1]

#' Momentum signal: sign of rolling mean return
momentum_signal <- function(bars, t, window = 20L) {
  if (t < window + 1) return(0)
  mu <- mean(bars$ret[(t-window+1):t], na.rm=TRUE)
  tanh(mu / (sd(bars$ret[(t-window+1):t], na.rm=TRUE) + 1e-8) * 10)
}

#' Mean reversion signal: z-score of close vs MA
mean_reversion_signal <- function(bars, t, window = 20L) {
  if (t < window + 1) return(0)
  ma  <- mean(bars$close[(t-window+1):t])
  sg  <- sd(bars$close[(t-window+1):t])
  if (sg < 1e-8) return(0)
  z   <- (bars$close[t] - ma) / sg
  tanh(-z)   # long when below MA, short above
}

#' Breakout signal: new N-bar high/low
breakout_signal <- function(bars, t, window = 20L) {
  if (t < window + 1) return(0)
  hi  <- max(bars$high[(t-window):(t-1)])
  lo  <- min(bars$low[(t-window):(t-1)])
  if (bars$close[t] > hi) return(1)
  if (bars$close[t] < lo) return(-1)
  return(0)
}

# ---------------------------------------------------------------------------
# 4. POSITION SIZING
# ---------------------------------------------------------------------------

#' Fixed fractional: bet fixed fraction of equity
fixed_fraction <- function(signal, equity, fraction = 0.1) {
  signal * fraction

}

#' Kelly fraction
kelly_sizing <- function(signal, recent_rets, fraction = 0.5) {
  if (length(recent_rets) < 5) return(0)
  mu  <- mean(recent_rets, na.rm=TRUE)
  s2  <- var(recent_rets, na.rm=TRUE)
  if (s2 < 1e-12) return(0)
  f   <- mu / s2
  signal * clip(fraction * f, -1, 1)
}

#' Volatility-targeted sizing: scale by target_vol / realised_vol
vol_target_sizing <- function(signal, recent_rets,
                               target_vol_ann = 0.15, ann = 252) {
  if (length(recent_rets) < 5) return(0)
  rv <- sd(recent_rets, na.rm=TRUE) * sqrt(ann)
  if (rv < 1e-8) return(0)
  scale <- clip(target_vol_ann / rv, 0.1, 5.0)
  signal * scale
}

#' Risk parity sizing: equal-risk contribution
risk_parity_sizing <- function(signals, recent_rets_matrix,
                                target_vol_ann = 0.15) {
  K   <- ncol(recent_rets_matrix)
  vols <- apply(recent_rets_matrix, 2, sd, na.rm=TRUE) * sqrt(252)
  vols[vols < 1e-8] <- 0.20   # floor
  raw_w <- 1 / vols
  raw_w <- raw_w / sum(abs(raw_w))
  signals * raw_w
}

# ---------------------------------------------------------------------------
# 5. TRANSACTION COST MODEL
# ---------------------------------------------------------------------------

#' Total transaction cost: spread + market impact + overnight financing
compute_tc <- function(trade_size, price, spread_pct = 0.001,
                        impact_param = 0.0001, financing_rate_ann = 0.0) {
  spread_cost  <- abs(trade_size) * price * spread_pct / 2
  impact_cost  <- abs(trade_size) * price * impact_param * sqrt(abs(trade_size))
  financing    <- abs(trade_size) * price * financing_rate_ann / 252
  spread_cost + impact_cost + financing
}

# ---------------------------------------------------------------------------
# 6. EVENT-DRIVEN BAR-BY-BAR BACKTEST ENGINE
# ---------------------------------------------------------------------------

run_backtest <- function(bars,
                          signal_fn,
                          sizing_fn        = fixed_fraction,
                          spread_pct       = 0.001,
                          impact_param     = 0.00005,
                          financing_ann    = 0.02,
                          max_position     = 1.0,
                          start_bar        = 30L,
                          signal_args      = list()) {
  n      <- bars$n
  equity <- rep(1.0, n)
  pos    <- 0.0
  positions <- numeric(n)
  rets_strat <- numeric(n)

  for (t in start_bar:n) {
    # 1. Compute signal
    sig  <- do.call(signal_fn, c(list(bars, t), signal_args))
    # 2. Size
    if (t > start_bar + 5) {
      recent <- bars$ret[(t-20):t]; recent <- recent[!is.na(recent)]
      new_pos <- sizing_fn(sig, recent)
    } else {
      new_pos <- sig * 0.1
    }
    new_pos <- clip(new_pos, -max_position, max_position)

    # 3. Execute: P&L from old position + cost of rebalancing
    ret_bar   <- if (!is.na(bars$ret[t])) bars$ret[t] else 0
    gross_ret <- pos * ret_bar
    trade_size <- new_pos - pos
    cost   <- compute_tc(trade_size, bars$close[t], spread_pct,
                          impact_param, financing_ann)
    net_ret <- gross_ret - cost

    equity[t]    <- equity[max(1,t-1)] * (1 + net_ret)
    rets_strat[t] <- net_ret
    positions[t]  <- new_pos
    pos           <- new_pos
  }

  list(equity = equity, returns = rets_strat,
       positions = positions, start = start_bar)
}

# ---------------------------------------------------------------------------
# 7. WALK-FORWARD ENGINE
# ---------------------------------------------------------------------------

walk_forward <- function(bars,
                          signal_fn,
                          sizing_fn   = fixed_fraction,
                          is_len      = 252L,
                          oos_len     = 63L,
                          start_bar   = 30L,
                          spread_pct  = 0.001,
                          signal_args = list()) {
  n       <- bars$n
  windows <- list()
  idx     <- 1L
  t       <- start_bar + is_len

  while (t + oos_len <= n) {
    is_start  <- t - is_len
    is_end    <- t - 1L
    oos_start <- t
    oos_end   <- min(t + oos_len - 1L, n)

    # Run backtest on IS window (for calibration / reporting)
    bars_is   <- lapply(bars[c("open","high","low","close","volume","ret","n")],
                         function(x) if (is.numeric(x)) x[is_start:is_end] else x)
    bars_is$n <- is_end - is_start + 1L

    bt_is <- run_backtest(bars_is, signal_fn, sizing_fn,
                           spread_pct=spread_pct, start_bar=start_bar,
                           signal_args=signal_args)

    # Run on OOS window using full history (look-ahead safe)
    bars_oos <- lapply(bars[c("open","high","low","close","volume","ret","n")],
                        function(x) if (is.numeric(x)) x[is_start:oos_end] else x)
    bars_oos$n <- oos_end - is_start + 1L
    bt_oos_full <- run_backtest(bars_oos, signal_fn, sizing_fn,
                                 spread_pct=spread_pct, start_bar=start_bar,
                                 signal_args=signal_args)
    oos_idx <- (is_end - is_start + 2):(oos_end - is_start + 1)

    oos_rets <- bt_oos_full$returns[oos_idx]

    windows[[idx]] <- list(
      window    = idx,
      is_sharpe = sharpe_ratio(bt_is$returns[bt_is$start:bt_is$n]),
      oos_sharpe = sharpe_ratio(oos_rets),
      is_mdd    = max_drawdown(cumprod(1 + bt_is$returns[bt_is$start:bt_is$n])),
      oos_mdd   = max_drawdown(cumprod(1 + oos_rets)),
      oos_rets  = oos_rets,
      t_start   = oos_start,
      t_end     = oos_end
    )
    idx <- idx + 1L
    t   <- t + oos_len
  }

  # Stitch OOS returns
  oos_rets_all <- unlist(lapply(windows, `[[`, "oos_rets"))
  list(windows = windows, oos_returns = oos_rets_all)
}

# ---------------------------------------------------------------------------
# 8. MULTIPLE STRATEGY COMPARISON
# ---------------------------------------------------------------------------

compare_strategies <- function(bars, signal_fns, signal_names = NULL,
                                sizing_fn = fixed_fraction) {
  if (is.null(signal_names)) signal_names <- paste0("Strat", seq_along(signal_fns))
  results <- lapply(seq_along(signal_fns), function(i) {
    bt <- run_backtest(bars, signal_fns[[i]], sizing_fn)
    r  <- bt$returns[bt$start:bars$n]
    eq <- cumprod(1 + r)
    data.frame(
      strategy  = signal_names[i],
      sharpe    = sharpe_ratio(r),
      calmar    = calmar_ratio(r),
      sortino   = sortino_ratio(r),
      max_dd    = max_drawdown(eq),
      total_ret = tail(eq,1) - 1,
      hit_rate  = mean(r > 0, na.rm=TRUE),
      ann_vol   = sd(r) * sqrt(252)
    )
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 9. BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------------------------

bootstrap_sharpe <- function(rets, n_boot = 1000L, conf = 0.95, seed = 42L) {
  set.seed(seed)
  n   <- length(rets)
  boot_sharpes <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    idx <- sample.int(n, n, replace=TRUE)
    boot_sharpes[b] <- sharpe_ratio(rets[idx])
  }
  alpha <- (1 - conf) / 2
  list(estimate = sharpe_ratio(rets),
       lower    = quantile(boot_sharpes, alpha, na.rm=TRUE),
       upper    = quantile(boot_sharpes, 1-alpha, na.rm=TRUE),
       boot_dist = boot_sharpes)
}

#' Block bootstrap (preserves serial dependence)
block_bootstrap_sharpe <- function(rets, block_size = 20L,
                                    n_boot = 1000L, conf = 0.95, seed = 42L) {
  set.seed(seed)
  n   <- length(rets); n_blocks <- ceiling(n / block_size)
  boot_sharpes <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    starts <- sample.int(max(n - block_size + 1, 1), n_blocks, replace=TRUE)
    idx    <- unlist(lapply(starts, function(s) s:min(s+block_size-1, n)))
    boot_sharpes[b] <- sharpe_ratio(rets[idx[1:n]])
  }
  alpha <- (1 - conf) / 2
  list(estimate = sharpe_ratio(rets),
       lower    = quantile(boot_sharpes, alpha, na.rm=TRUE),
       upper    = quantile(boot_sharpes, 1-alpha, na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 10. DEFLATED SHARPE RATIO
# ---------------------------------------------------------------------------
# Accounts for: (a) multiple trials, (b) fat tails, (c) short history.
# DSR < 0.5 means the SR is probably luck.

deflated_sharpe <- function(sharpe_obs, n_obs, n_trials,
                              skewness = 0, kurtosis = 3,
                              sharpe_benchmark = 0) {
  # Adjustment for non-normality (Lo 2002 correction)
  gamma <- kurtosis - 3   # excess kurtosis
  sr_adj <- sharpe_obs * sqrt((n_obs - 1) / n_obs) *
    sqrt(1 - skewness * sharpe_obs / 6 + (gamma / 24) * sharpe_obs^2)

  # Expected max Sharpe under null with n_trials
  ev_max_sr <- (1 - 0.5772) * qnorm(1 - 1/n_trials) +
    0.5772 * qnorm(1 - 1/(n_trials * exp(1)))
  ev_max_sr <- ev_max_sr / sqrt(n_obs)

  # DSR = P(SR_obs > SR_max under null)
  z   <- (sr_adj - sharpe_benchmark - ev_max_sr) /
    sqrt(1/n_obs + ev_max_sr^2 * (gamma/4 + 1/2))
  dsr <- pnorm(z)

  list(DSR = dsr, z = z, ev_max_sr = ev_max_sr,
       adjusted_sr = sr_adj)
}

# ---------------------------------------------------------------------------
# 11. CSCV (COMBINATORIALLY SYMMETRIC CROSS VALIDATION)
# ---------------------------------------------------------------------------
# Bailey et al. (2015): estimate probability of backtest overfitting.
# Split time series into S folds; for each subset-complement combination
# select best IS strategy and check if it's best OOS.

cscv_overfitting <- function(strategy_rets_matrix, S = 8L, seed = 42L) {
  set.seed(seed)
  T_  <- nrow(strategy_rets_matrix)
  K   <- ncol(strategy_rets_matrix)
  fold_size <- T_ %/% S

  pbo_samples <- numeric()   # probability of backtest overfitting

  for (trial in 1:min(50L, choose(S, S%/%2))) {
    # Random IS/OOS fold split
    folds <- rep(1:S, each=fold_size)[1:T_]
    is_folds  <- sample.int(S, S%/%2)
    oos_folds <- setdiff(1:S, is_folds)

    is_idx  <- which(folds %in% is_folds)
    oos_idx <- which(folds %in% oos_folds)

    is_sharpes  <- apply(strategy_rets_matrix[is_idx,  , drop=FALSE], 2, sharpe_ratio)
    oos_sharpes <- apply(strategy_rets_matrix[oos_idx, , drop=FALSE], 2, sharpe_ratio)

    best_is  <- which.max(is_sharpes)
    rank_oos <- rank(oos_sharpes)[best_is]   # rank of IS-best in OOS (1=worst)
    # Relative rank [0,1]: 0=worst, 1=best
    rel_rank <- (rank_oos - 1) / (K - 1)
    pbo_samples <- c(pbo_samples, rel_rank)
  }

  list(
    PBO = mean(pbo_samples < 0.5),   # frac where IS best is below median OOS
    mean_rank = mean(pbo_samples),
    samples   = pbo_samples
  )
}

# ---------------------------------------------------------------------------
# 12. DIEBOLD-MARIANO TEST (strategy A vs B)
# ---------------------------------------------------------------------------
# Test if two forecast error series have equal predictive accuracy.

diebold_mariano <- function(rets_a, rets_b, horizon = 1L) {
  n   <- min(length(rets_a), length(rets_b))
  d   <- rets_a[1:n] - rets_b[1:n]   # differential loss (neg returns = loss)
  mu_d <- mean(d, na.rm=TRUE)
  # Newey-West HAC variance
  gamma0 <- var(d, na.rm=TRUE)
  hac_var <- gamma0
  for (h in 1:(horizon - 1 + 3)) {  # a few lags
    gam_h  <- cov(d[1:(n-h)], d[(h+1):n])
    w_h    <- 1 - h / (horizon + 1)
    hac_var <- hac_var + 2 * w_h * gam_h
  }
  se  <- sqrt(pmax(hac_var, 0) / n)
  t_  <- mu_d / max(se, 1e-12)
  pval <- 2 * pt(-abs(t_), df = n - 1)
  data.frame(mean_diff = mu_d, t_stat = t_,
             p_value = pval, se = se,
             reject_null = pval < 0.05)
}

# ---------------------------------------------------------------------------
# 13. PERFORMANCE REPORT
# ---------------------------------------------------------------------------

full_performance_report <- function(rets, label = "Strategy", ann = 252) {
  rets <- rets[!is.na(rets)]
  eq   <- cumprod(1 + rets)
  n    <- length(rets)
  data.frame(
    label     = label,
    n_bars    = n,
    sharpe    = sharpe_ratio(rets, ann),
    calmar    = calmar_ratio(rets, ann),
    sortino   = sortino_ratio(rets, ann=ann),
    max_dd    = max_drawdown(eq),
    total_ret = tail(eq, 1) - 1,
    ann_ret   = mean(rets) * ann,
    ann_vol   = sd(rets) * sqrt(ann),
    hit_rate  = mean(rets > 0),
    avg_win   = mean(rets[rets > 0]),
    avg_loss  = mean(rets[rets < 0]),
    win_loss  = abs(mean(rets[rets>0]) / mean(rets[rets<0]))
  )
}

# ---------------------------------------------------------------------------
# 14. MAIN DEMO
# ---------------------------------------------------------------------------

run_backtest_demo <- function() {
  cat("=== Comprehensive Backtest Framework Demo ===\n\n")

  # Generate market data
  bars <- simulate_bars(n=1000L, seed=42L)
  cat(sprintf("Simulated bars: %d  |  Price: %.0f -> %.0f\n",
              bars$n, bars$close[1], tail(bars$close,1)))

  # Run three signals
  cat("\n--- 1. Strategy Comparison ---\n")
  signals <- list(momentum_signal, mean_reversion_signal, breakout_signal)
  names_s <- c("Momentum","MeanRev","Breakout")
  cmp <- compare_strategies(bars, signals, names_s)
  print(cmp[, c("strategy","sharpe","calmar","max_dd","total_ret")])

  # Walk-forward for momentum
  cat("\n--- 2. Walk-Forward Validation (Momentum) ---\n")
  wf <- walk_forward(bars, momentum_signal, is_len=252L, oos_len=63L)
  n_windows <- length(wf$windows)
  is_sh  <- sapply(wf$windows, `[[`, "is_sharpe")
  oos_sh <- sapply(wf$windows, `[[`, "oos_sharpe")
  cat(sprintf("  Windows: %d  |  IS Sharpe: %.3f  OOS Sharpe: %.3f\n",
              n_windows, mean(is_sh, na.rm=TRUE), mean(oos_sh, na.rm=TRUE)))
  oos_full <- wf$oos_returns
  cat(sprintf("  Combined OOS Sharpe: %.3f\n", sharpe_ratio(oos_full)))

  # Bootstrap CI
  cat("\n--- 3. Bootstrap Confidence Intervals on Sharpe ---\n")
  bt_run <- run_backtest(bars, momentum_signal)
  rets_mom <- bt_run$returns[bt_run$start:bars$n]
  bci <- bootstrap_sharpe(rets_mom, n_boot=500L)
  cat(sprintf("  Sharpe=%.3f  95%% CI=[%.3f, %.3f]\n",
              bci$estimate, bci$lower, bci$upper))
  bci_block <- block_bootstrap_sharpe(rets_mom, block_size=20L, n_boot=500L)
  cat(sprintf("  Block bootstrap CI=[%.3f, %.3f]\n",
              bci_block$lower, bci_block$upper))

  # Deflated Sharpe
  cat("\n--- 4. Deflated Sharpe Ratio ---\n")
  dsr <- deflated_sharpe(bci$estimate, n_obs=length(rets_mom),
                          n_trials=10L, skewness=-0.3, kurtosis=5)
  cat(sprintf("  DSR=%.4f  (>0.5 suggests genuine edge)\n", dsr$DSR))
  cat(sprintf("  Expected max SR under null (10 trials): %.3f\n", dsr$ev_max_sr))

  # CSCV overfitting test
  cat("\n--- 5. CSCV Overfitting Detection ---\n")
  # Build matrix of strategy returns for CSCV
  strat_mat <- matrix(NA, bars$n, 3)
  for (i in seq_along(signals)) {
    bt <- run_backtest(bars, signals[[i]])
    strat_mat[, i] <- bt$returns
  }
  strat_mat[is.na(strat_mat)] <- 0
  cscv_res <- cscv_overfitting(strat_mat, S=6L)
  cat(sprintf("  PBO=%.3f  (>0.5 = likely overfit)\n", cscv_res$PBO))

  # Diebold-Mariano test
  cat("\n--- 6. Diebold-Mariano Test (Momentum vs MeanRev) ---\n")
  bt1 <- run_backtest(bars, momentum_signal)
  bt2 <- run_backtest(bars, mean_reversion_signal)
  n_   <- min(length(bt1$returns), length(bt2$returns))
  dm   <- diebold_mariano(bt1$returns[1:n_], bt2$returns[1:n_])
  cat(sprintf("  DM stat=%.3f  p-value=%.4f  Reject H0: %s\n",
              dm$t_stat, dm$p_value, dm$reject_null))

  # Full report for best strategy
  cat("\n--- 7. Full Performance Report ---\n")
  best_idx <- which.max(cmp$sharpe)
  bt_best  <- run_backtest(bars, signals[[best_idx]])
  rpt <- full_performance_report(bt_best$returns[bt_best$start:bars$n],
                                  label = names_s[best_idx])
  print(t(rpt))

  cat("\nDone.\n")
  invisible(list(cmp=cmp, wf=wf, bci=bci, dsr=dsr, cscv=cscv_res))
}

if (interactive()) {
  bt_results <- run_backtest_demo()
}

# ---------------------------------------------------------------------------
# 15. REGIME-CONDITIONAL BACKTEST
# ---------------------------------------------------------------------------
# Run backtest separately per macro regime; compare Sharpe contributions.

regime_backtest <- function(bars, signal_fn, regime_vec,
                             sizing_fn=fixed_fraction, spread_pct=0.001) {
  bt_full <- run_backtest(bars, signal_fn, sizing_fn, spread_pct=spread_pct)
  results <- list()
  regimes <- sort(unique(regime_vec[!is.na(regime_vec)]))
  for (r in regimes) {
    idx   <- which(regime_vec == r)
    idx   <- idx[idx >= bt_full$start & idx <= bars$n]
    rets  <- bt_full$returns[idx]
    results[[r]] <- data.frame(regime=r, n_bars=length(rets),
                                sharpe=sharpe_ratio(rets),
                                mean_ret=mean(rets,na.rm=TRUE)*252,
                                max_dd=max_drawdown(cumprod(1+rets)))
  }
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 16. LEVERAGE IMPACT ANALYSIS
# ---------------------------------------------------------------------------

leverage_backtest <- function(bars, signal_fn, leverages=c(0.5,1,2,3),
                               spread_pct=0.001) {
  results <- lapply(leverages, function(lev) {
    bt  <- run_backtest(bars, signal_fn,
                         sizing_fn=function(sig, rets) fixed_fraction(sig)*lev,
                         spread_pct=spread_pct)
    r   <- bt$returns[bt$start:bars$n]
    eq  <- cumprod(1+r)
    data.frame(leverage=lev, sharpe=sharpe_ratio(r), calmar=calmar_ratio(r),
               max_dd=max_drawdown(eq), total_ret=tail(eq,1)-1)
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 17. SENSITIVITY ANALYSIS (PARAMETER GRID)
# ---------------------------------------------------------------------------

parameter_sensitivity <- function(bars, signal_fn_factory, param_grid,
                                   sizing_fn=fixed_fraction, spread_pct=0.001) {
  results <- lapply(seq_len(nrow(param_grid)), function(i) {
    params <- param_grid[i,]
    sig_fn <- do.call(signal_fn_factory, as.list(params))
    bt <- run_backtest(bars, sig_fn, sizing_fn, spread_pct=spread_pct)
    r  <- bt$returns[bt$start:bars$n]
    cbind(params, sharpe=sharpe_ratio(r),
          max_dd=max_drawdown(cumprod(1+r)))
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 18. EQUITY CURVE STATISTICS
# ---------------------------------------------------------------------------

equity_curve_stats <- function(equity, freq = "daily", ann = 252) {
  rets <- c(NA, diff(log(pmax(equity, 1e-12))))[-1]
  eq   <- equity / equity[1]
  # Omega ratio: P(gain) / P(loss) weighted by magnitude
  omega <- sum(pmax(rets, 0)) / max(sum(abs(pmin(rets, 0))), 1e-8)
  # Recovery factor: total return / max drawdown
  mdd  <- abs(max_drawdown(eq))
  rf   <- (tail(eq,1)-1) / max(mdd, 1e-4)
  data.frame(
    n_bars      = length(rets),
    total_ret   = tail(eq,1) - 1,
    ann_ret     = mean(rets,na.rm=TRUE)*ann,
    ann_vol     = sd(rets,na.rm=TRUE)*sqrt(ann),
    sharpe      = sharpe_ratio(rets,ann),
    max_dd      = mdd,
    calmar      = calmar_ratio(rets,ann),
    omega_ratio = omega,
    recovery_factor = rf,
    skewness    = mean((rets-mean(rets,na.rm=TRUE))^3,na.rm=TRUE)/sd(rets,na.rm=TRUE)^3
  )
}

# ---------------------------------------------------------------------------
# 19. MONTE CARLO FORWARD SIMULATION
# ---------------------------------------------------------------------------
# Simulate forward equity paths from the strategy's return distribution.

mc_forward <- function(strategy_rets, n_periods=252L, n_paths=5000L, seed=42L) {
  set.seed(seed)
  mu  <- mean(strategy_rets, na.rm=TRUE)
  sg  <- sd(strategy_rets, na.rm=TRUE)
  paths <- matrix(NA, n_paths, n_periods)
  for (p in seq_len(n_paths)) {
    r         <- rnorm(n_periods, mu, sg)
    paths[p,] <- cumprod(1+r)
  }
  list(
    median_path = apply(paths, 2, median),
    p05_path    = apply(paths, 2, quantile, 0.05),
    p95_path    = apply(paths, 2, quantile, 0.95),
    p_positive  = mean(paths[,n_periods] > 1),
    expected_end = mean(paths[,n_periods])
  )
}

# ---------------------------------------------------------------------------
# 20. EXTENDED BACKTEST DEMO
# ---------------------------------------------------------------------------

run_backtest_extended_demo <- function() {
  cat("=== Backtest Framework Extended Demo ===\n\n")
  bars   <- simulate_bars(n=800L, seed=99L)
  regime <- rep(c(1L,2L), each=50L)[1:bars$n]

  cat("--- Regime-Conditional Backtest ---\n")
  rb <- regime_backtest(bars, momentum_signal, regime)
  print(rb)

  cat("\n--- Leverage Impact ---\n")
  li <- leverage_backtest(bars, momentum_signal, leverages=c(0.5,1,2,3))
  print(li)

  cat("\n--- Equity Curve Statistics ---\n")
  bt  <- run_backtest(bars, momentum_signal)
  eq  <- bt$equity[bt$start:bars$n]
  ecs <- equity_curve_stats(eq)
  print(t(ecs))

  cat("\n--- Monte Carlo Forward Simulation ---\n")
  rets <- bt$returns[bt$start:bars$n]
  mc   <- mc_forward(rets, n_periods=126L, n_paths=2000L)
  cat(sprintf("  P(positive 126-bar): %.3f  Expected end equity: %.4f\n",
              mc$p_positive, mc$expected_end))
  cat(sprintf("  5th pctile: %.4f  Median: %.4f  95th: %.4f\n",
              tail(mc$p05_path,1), tail(mc$median_path,1), tail(mc$p95_path,1)))

  invisible(list(rb=rb, li=li, ecs=ecs, mc=mc))
}

if (interactive()) {
  bt_ext <- run_backtest_extended_demo()
}

# =============================================================================
# SECTION: PERMUTATION TEST FOR STRATEGY SIGNIFICANCE
# =============================================================================
# Randomly permute returns and recompute Sharpe to build null distribution.
# P-value = fraction of permuted Sharpes exceeding observed Sharpe.

permutation_sharpe_test <- function(rets, n_perm = 1000) {
  obs_sharpe <- mean(rets) / (sd(rets) + 1e-10) * sqrt(252)
  perm_sharpes <- numeric(n_perm)
  for (i in seq_len(n_perm))
    perm_sharpes[i] <- mean(sample(rets)) / (sd(rets) + 1e-10) * sqrt(252)
  p_val <- mean(perm_sharpes >= obs_sharpe)
  list(obs_sharpe = obs_sharpe, p_value = p_val,
       perm_sharpes = perm_sharpes)
}

# =============================================================================
# SECTION: CALMAR, MAR AND STERLING RATIOS
# =============================================================================

calmar_ratio <- function(rets, ann_factor = 252) {
  ann_ret <- mean(rets) * ann_factor
  cum     <- cumprod(1 + rets)
  dd      <- (cummax(cum) - cum) / cummax(cum)
  max_dd  <- max(dd, na.rm = TRUE)
  if (max_dd < 1e-9) return(NA_real_)
  ann_ret / max_dd
}

sterling_ratio <- function(rets, ann_factor = 252, top_n = 3) {
  # Average of top_n largest drawdowns
  ann_ret <- mean(rets) * ann_factor
  cum     <- cumprod(1 + rets)
  dd      <- (cummax(cum) - cum) / cummax(cum)
  avg_dd  <- mean(sort(dd, decreasing = TRUE)[seq_len(min(top_n, length(dd)))])
  if (avg_dd < 1e-9) return(NA_real_)
  ann_ret / avg_dd
}

# =============================================================================
# SECTION: OMEGA RATIO
# =============================================================================
# Omega = integral of (1-F(x)) above threshold / integral of F(x) below threshold.
# More informative than Sharpe because it uses the full return distribution.

omega_ratio <- function(rets, threshold = 0) {
  gains  <- sum(pmax(rets - threshold, 0))
  losses <- sum(pmax(threshold - rets, 0))
  if (losses < 1e-10) return(Inf)
  gains / losses
}

# =============================================================================
# SECTION: BACKTEST OVERFITTING VIA COMBINATORIAL PURGED CV
# =============================================================================
# Simplified version: split in-sample period into k folds, train on k-1,
# evaluate on held-out fold. Average IS Sharpe vs OOS Sharpe.

simple_purged_cv <- function(rets, signal_fn, k = 5, gap = 5) {
  n       <- length(rets)
  fold_sz <- floor(n / k)
  is_sharpes  <- numeric(k)
  oos_sharpes <- numeric(k)
  for (i in seq_len(k)) {
    oos_start <- (i-1)*fold_sz + 1
    oos_end   <- min(i*fold_sz, n)
    is_idx    <- setdiff(seq_len(n),
                         seq(max(1, oos_start - gap), min(n, oos_end + gap)))
    oos_idx   <- seq(oos_start, oos_end)
    if (length(is_idx) < 20 || length(oos_idx) < 5) next
    sig_is  <- signal_fn(rets[is_idx])
    sig_oos <- signal_fn(rets[oos_idx])
    pos_is  <- sign(sig_is) * rets[is_idx]
    pos_oos <- sign(sig_oos) * rets[oos_idx]
    is_sharpes[i]  <- mean(pos_is)  / (sd(pos_is)  + 1e-9) * sqrt(252)
    oos_sharpes[i] <- mean(pos_oos) / (sd(pos_oos) + 1e-9) * sqrt(252)
  }
  list(is_sharpe  = mean(is_sharpes),
       oos_sharpe = mean(oos_sharpes),
       overfitting = mean(is_sharpes) - mean(oos_sharpes))
}

# =============================================================================
# SECTION: STRATEGY COMBINATION — RANK WEIGHTING
# =============================================================================
# Combine multiple strategy signals by ranking their recent performance
# and allocating more weight to top-ranked strategies.

rank_weight_strategies <- function(strategy_rets_mat, lookback = 60) {
  # strategy_rets_mat: T x n_strategies
  T <- nrow(strategy_rets_mat); n <- ncol(strategy_rets_mat)
  combined <- numeric(T)
  for (t in seq(lookback+1, T)) {
    window_rets <- strategy_rets_mat[(t-lookback):(t-1), , drop=FALSE]
    sharpes <- apply(window_rets, 2, function(r)
      mean(r) / (sd(r) + 1e-9) * sqrt(252))
    ranks   <- rank(sharpes)
    w       <- ranks / sum(ranks)
    combined[t] <- sum(w * strategy_rets_mat[t,])
  }
  combined
}

# =============================================================================
# SECTION: MONTE CARLO STRESS TEST
# =============================================================================
# Resample blocks of returns to simulate bad-luck scenarios.

mc_stress_test <- function(rets, n_sim = 500, block_size = 20,
                            worst_pct = 0.05) {
  T       <- length(rets)
  n_blocks <- floor(T / block_size)
  sharpes <- numeric(n_sim)
  for (i in seq_len(n_sim)) {
    idx    <- sample(seq_len(n_blocks - 1), n_blocks, replace = TRUE)
    sim_r  <- unlist(lapply(idx, function(b)
      rets[(b*block_size+1):min((b+1)*block_size, T)]))
    sharpes[i] <- mean(sim_r) / (sd(sim_r) + 1e-9) * sqrt(252)
  }
  worst_sharpe <- quantile(sharpes, worst_pct, na.rm=TRUE)
  list(sharpe_dist = sharpes,
       worst_5pct  = worst_sharpe,
       prob_neg_sharpe = mean(sharpes < 0))
}

# =============================================================================
# SECTION: FINAL DEMO
# =============================================================================

run_backtest_final_demo <- function() {
  set.seed(111)
  T <- 500
  rets <- rnorm(T, 0.0003, 0.015)

  cat("--- Permutation Sharpe Test ---\n")
  pt <- permutation_sharpe_test(rets, n_perm = 500)
  cat("Observed Sharpe:", round(pt$obs_sharpe, 3),
      "  p-value:", round(pt$p_value, 3), "\n")

  cat("\n--- Calmar Ratio:", round(calmar_ratio(rets), 3), "\n")
  cat("Sterling Ratio:", round(sterling_ratio(rets), 3), "\n")
  cat("Omega Ratio:", round(omega_ratio(rets, 0), 3), "\n")

  cat("\n--- Purged CV ---\n")
  signal_fn <- function(r) cumsum(r)   # trivial momentum signal
  cv_res    <- simple_purged_cv(rets, signal_fn)
  cat("IS Sharpe:", round(cv_res$is_sharpe, 3),
      "  OOS Sharpe:", round(cv_res$oos_sharpe, 3),
      "  Overfitting:", round(cv_res$overfitting, 3), "\n")

  cat("\n--- MC Stress Test ---\n")
  st <- mc_stress_test(rets, n_sim = 300)
  cat("Worst 5% Sharpe:", round(st$worst_5pct, 3),
      "  P(Sharpe<0):", round(st$prob_neg_sharpe, 3), "\n")

  invisible(list(perm_test=pt, calmar=calmar_ratio(rets), stress=st))
}

if (interactive()) {
  bt_final <- run_backtest_final_demo()
}

# =============================================================================
# SECTION: EXPOSURE-ADJUSTED PERFORMANCE METRICS
# =============================================================================
# Standard Sharpe ignores leverage.  Exposure-adjusted Sharpe penalises
# for periods of high gross exposure (leveraged drawdowns are larger).

exposure_adjusted_sharpe <- function(rets, exposure, ann_factor = 252) {
  adj_rets <- rets / (exposure + 1e-9)
  mean(adj_rets) / (sd(adj_rets) + 1e-9) * sqrt(ann_factor)
}

# Hit rate and profit factor (common in systematic trading)
hit_rate <- function(rets) mean(rets > 0, na.rm=TRUE)
profit_factor <- function(rets) {
  gains  <- sum(pmax(rets, 0))
  losses <- sum(pmax(-rets, 0))
  if (losses < 1e-10) return(Inf)
  gains / losses
}

# =============================================================================
# SECTION: EQUITY CURVE ANALYSIS — PEAKS AND RECOVERIES
# =============================================================================

drawdown_series <- function(rets) {
  cum <- cumprod(1 + rets)
  (cummax(cum) - cum) / cummax(cum)
}

recovery_times <- function(rets) {
  dd     <- drawdown_series(rets)
  in_dd  <- dd > 0
  rle_dd <- rle(in_dd)
  # Duration of each drawdown episode
  rle_dd$lengths[rle_dd$values]
}

avg_recovery_time <- function(rets) {
  rt <- recovery_times(rets)
  if (length(rt) == 0) return(0)
  mean(rt)
}

# =============================================================================
# SECTION: TRANSACTION COST MODEL — SLIPPAGE + COMMISSION
# =============================================================================

tc_model <- function(rets, turnover, spread_bps = 5, commission_bps = 2) {
  # total cost per unit turnover
  tc_per_trade <- (spread_bps + commission_bps) / 10000
  tc_drag      <- turnover * tc_per_trade
  net_rets     <- rets - tc_drag
  net_rets
}

# =============================================================================
# SECTION: BACKTEST SUMMARY REPORT
# =============================================================================

full_backtest_report <- function(rets, label = "Strategy") {
  ann  <- mean(rets) * 252
  vol  <- sd(rets) * sqrt(252)
  sr   <- ann / (vol + 1e-9)
  dd   <- drawdown_series(rets)
  mdd  <- max(dd)
  hr   <- hit_rate(rets)
  pf   <- profit_factor(rets)
  rt   <- avg_recovery_time(rets)
  calmar <- ann / (mdd + 1e-9)
  cat(sprintf("=== %s ===\n", label))
  cat(sprintf("  Ann. Return:  %.2f%%\n", ann*100))
  cat(sprintf("  Ann. Vol:     %.2f%%\n", vol*100))
  cat(sprintf("  Sharpe:       %.3f\n",   sr))
  cat(sprintf("  Max DD:       %.2f%%\n", mdd*100))
  cat(sprintf("  Calmar:       %.3f\n",   calmar))
  cat(sprintf("  Hit Rate:     %.2f%%\n", hr*100))
  cat(sprintf("  Profit Factor:%.3f\n",   pf))
  cat(sprintf("  Avg Recovery: %.1f days\n", rt))
  invisible(list(annual_return=ann, vol=vol, sharpe=sr, max_dd=mdd,
                 calmar=calmar, hit_rate=hr, profit_factor=pf))
}

if (interactive()) {
  set.seed(222)
  r <- rnorm(500, 0.0004, 0.015)
  full_backtest_report(r, "Demo Strategy")
  cat("Omega ratio:", round(omega_ratio(r), 3), "\n")
}
