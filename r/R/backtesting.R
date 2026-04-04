# backtesting.R
# Custom backtesting engine for SRFM research.
# Implements: PerformanceAnalytics integration, custom engine, transaction costs,
#             bootstrap confidence intervals for Sharpe, block bootstrap.
# Dependencies: xts, zoo, ggplot2, PerformanceAnalytics (optional)

library(xts)
library(zoo)
library(ggplot2)

.has_pa <- requireNamespace("PerformanceAnalytics", quietly = TRUE)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Core Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────

#' run_backtest
#'
#' Run a strategy backtest from signals and returns.
#'
#' @param signals xts or matrix (n_obs × n_assets): position sizes (can be fractional)
#' @param returns xts or matrix (n_obs × n_assets): asset returns
#' @param rebalance_freq character: "daily", "weekly", "monthly"
#' @param tc_model list: transaction cost model parameters
#'   - type: "bps" (fixed bps), "linear", "sqrt"
#'   - cost_bps: cost per unit turnover in basis points
#'   - spread_bps: half-spread in bps
#'   - market_impact: alpha * (trade_size / ADV)^beta
#' @param initial_capital numeric
#' @param max_leverage numeric, maximum allowed leverage
#' @return BacktestResult list
run_backtest <- function(
  signals,
  returns,
  rebalance_freq = "daily",
  tc_model       = list(type = "bps", cost_bps = 10.0),
  initial_capital = 1e6,
  max_leverage    = 1.0,
  slippage_pct    = 0.0,
  rf              = 0.0
) {
  # Align signals and returns
  if (is.xts(signals) && is.xts(returns)) {
    common_idx <- intersect(index(signals), index(returns))
    signals    <- signals[common_idx, , drop = FALSE]
    returns    <- returns[common_idx, , drop = FALSE]
  }

  sig_mat <- as.matrix(signals)
  ret_mat <- as.matrix(returns)
  n_obs   <- nrow(sig_mat)
  n_assets <- ncol(ret_mat)
  dates   <- if (is.xts(returns)) index(returns) else seq_len(n_obs)

  # Normalize signal to portfolio weights
  weights <- .normalize_weights(sig_mat, max_leverage)

  # Initialize state
  port_values  <- numeric(n_obs + 1)
  port_returns <- numeric(n_obs)
  tc_costs     <- numeric(n_obs)
  turnover     <- numeric(n_obs)

  port_values[1] <- initial_capital
  prev_weights   <- rep(0.0, n_assets)

  for (t in seq_len(n_obs)) {
    w_t     <- weights[t, ]
    to_t    <- sum(abs(w_t - prev_weights)) / 2.0
    turnover[t] <- to_t

    # Transaction cost
    tc_t <- .compute_tc(to_t, port_values[t], tc_model)
    tc_costs[t] <- tc_t

    # Gross portfolio return
    r_t <- as.numeric(ret_mat[t, ])
    gross_ret <- sum(w_t * r_t)

    # Net return
    net_ret <- gross_ret - tc_t / port_values[t] - slippage_pct * to_t
    port_returns[t] <- net_ret
    port_values[t+1] <- port_values[t] * (1 + net_ret)

    prev_weights <- w_t * (1 + r_t) / (1 + gross_ret + 1e-10)
    prev_weights[!is.finite(prev_weights)] <- w_t[!is.finite(prev_weights)]
  }

  port_xts    <- xts(port_returns, order.by = dates)
  wealth_xts  <- xts(port_values[-1], order.by = dates)
  tc_xts      <- xts(tc_costs, order.by = dates)
  turnover_xts <- xts(turnover, order.by = dates)

  # Performance metrics
  perf <- compute_performance_metrics(port_xts, rf = rf)

  structure(
    list(
      returns       = port_xts,
      wealth        = wealth_xts,
      tc_costs      = tc_xts,
      turnover      = turnover_xts,
      weights       = xts(weights, order.by = dates),
      performance   = perf,
      initial_capital = initial_capital
    ),
    class = "BacktestResult"
  )
}


#' .normalize_weights
.normalize_weights <- function(sig_mat, max_leverage) {
  apply(sig_mat, 1, function(w) {
    if (all(w == 0)) return(w)
    w_long  <- sum(pmax(w, 0))
    w_short <- sum(pmin(w, 0))
    gross   <- w_long - w_short

    if (gross < 1e-10) return(w / (sum(abs(w)) + 1e-10))

    scale <- min(max_leverage, 1.0) / max(gross, 1.0)
    w * scale
  }) |> t()
}


#' .compute_tc
.compute_tc <- function(turnover, portfolio_value, tc_model) {
  type <- tc_model$type %||% "bps"
  cost_bps <- (tc_model$cost_bps %||% 10.0) / 10000.0

  if (type == "bps") {
    return(turnover * portfolio_value * cost_bps)
  }
  if (type == "sqrt") {
    alpha <- tc_model$market_impact %||% 0.05
    beta  <- tc_model$beta %||% 0.5
    adv   <- tc_model$ADV %||% 1e7
    trade_size <- turnover * portfolio_value
    return(alpha * (trade_size / adv)^beta * trade_size + cost_bps * portfolio_value * turnover)
  }
  # linear
  turnover * portfolio_value * cost_bps
}


`%||%` <- function(a, b) if (!is.null(a)) a else b


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

#' compute_performance_metrics
#'
#' Compute comprehensive performance metrics for a return series.
#'
#' @param returns xts or numeric vector of returns
#' @param rf numeric risk-free rate per period
#' @param annual_periods integer (252=daily, 52=weekly, 12=monthly)
#' @return list of performance metrics
compute_performance_metrics <- function(returns, rf = 0.0, annual_periods = 252L) {
  if (is.xts(returns)) {
    r <- as.numeric(returns)
  } else {
    r <- as.numeric(returns)
  }
  r <- r[!is.na(r)]
  n <- length(r)
  if (n < 2) return(list())

  exc_ret   <- r - rf
  ann_ret   <- mean(r) * annual_periods
  ann_vol   <- sd(r) * sqrt(annual_periods)
  sharpe    <- if (ann_vol > 0) (ann_ret - rf * annual_periods) / ann_vol else NA_real_

  # Sortino ratio (downside deviation)
  dd        <- r[r < rf] - rf
  down_dev  <- sqrt(mean(dd^2)) * sqrt(annual_periods)
  sortino   <- if (down_dev > 0) (ann_ret - rf * annual_periods) / down_dev else NA_real_

  # Max drawdown
  cum_ret   <- cumprod(1 + r)
  roll_max  <- cummax(cum_ret)
  drawdown  <- (cum_ret - roll_max) / roll_max
  max_dd    <- min(drawdown, na.rm = TRUE)

  # Drawdown duration
  in_dd  <- drawdown < -0.001
  if (any(in_dd)) {
    rle_dd  <- rle(in_dd)
    max_dur <- max(rle_dd$lengths[rle_dd$values], 0)
  } else {
    max_dur <- 0L
  }

  # Calmar ratio
  calmar <- if (abs(max_dd) > 0) ann_ret / abs(max_dd) else NA_real_

  # Information ratio (vs 0 benchmark)
  ir <- if (ann_vol > 0) ann_ret / ann_vol else NA_real_

  # Omega ratio
  threshold   <- rf
  gains       <- sum(pmax(r - threshold, 0))
  losses      <- sum(pmax(threshold - r, 0))
  omega       <- if (losses > 0) gains / losses else NA_real_

  # Hit rate
  hit_rate    <- mean(r > rf)

  # Skewness and kurtosis
  skew  <- if (n >= 3) {
    mu3 <- mean((r - mean(r))^3)
    mu2 <- mean((r - mean(r))^2)
    mu3 / mu2^1.5
  } else NA_real_
  kurt  <- if (n >= 4) {
    mu4 <- mean((r - mean(r))^4)
    mu2 <- mean((r - mean(r))^2)
    mu4 / mu2^2 - 3
  } else NA_real_

  # VaR and CVaR
  var_95  <- quantile(r, 0.05)
  cvar_95 <- mean(r[r <= var_95])

  # Turnover stats (if provided in returns attributes)

  list(
    n_periods         = n,
    ann_return        = ann_ret,
    ann_vol           = ann_vol,
    sharpe_ratio      = sharpe,
    sortino_ratio     = sortino,
    calmar_ratio      = calmar,
    omega_ratio       = omega,
    max_drawdown      = max_dd,
    max_dd_duration   = max_dur,
    hit_rate          = hit_rate,
    skewness          = skew,
    excess_kurtosis   = kurt,
    var_95            = var_95,
    cvar_95           = cvar_95,
    total_return      = prod(1 + r) - 1,
    information_ratio = ir
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: PerformanceAnalytics Integration
# ─────────────────────────────────────────────────────────────────────────────

#' pa_performance_table
#'
#' Generate PerformanceAnalytics-compatible performance table.
#' Falls back to custom metrics if PA not available.
#'
#' @param returns_list named list of xts return series
#' @param rf numeric risk-free rate
#' @return data.frame of performance metrics
pa_performance_table <- function(returns_list, rf = 0.0) {
  if (.has_pa) {
    library(PerformanceAnalytics)
    returns_merged <- do.call(merge.xts, returns_list)
    # Annualised stats
    pa_table <- table.AnnualizedReturns(returns_merged, Rf = rf, scale = 252)
    pa_dd    <- table.DrawdownsRatio(returns_merged)
    pa_stats <- rbind(pa_table, pa_dd)
    return(as.data.frame(t(pa_stats)))
  }

  # Custom implementation
  results <- lapply(names(returns_list), function(nm) {
    r <- returns_list[[nm]]
    perf <- compute_performance_metrics(r, rf = rf)
    data.frame(
      strategy         = nm,
      ann_return       = perf$ann_return,
      ann_vol          = perf$ann_vol,
      sharpe           = perf$sharpe_ratio,
      sortino          = perf$sortino_ratio,
      calmar           = perf$calmar_ratio,
      max_drawdown     = perf$max_drawdown,
      hit_rate         = perf$hit_rate,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, results)
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Transaction Cost Models
# ─────────────────────────────────────────────────────────────────────────────

#' linear_tc_model
#'
#' Linear transaction cost: TC = spread/2 + market_impact_coefficient * trade_size
#'
#' @param trade_sizes numeric vector (fraction of portfolio)
#' @param spread_bps numeric, bid-ask half-spread in basis points
#' @param impact_coef numeric, linear impact coefficient
#' @return numeric vector of costs (fraction of portfolio)
linear_tc_model <- function(trade_sizes, spread_bps = 5.0, impact_coef = 0.1) {
  abs_trades <- abs(trade_sizes)
  spread_cost  <- abs_trades * spread_bps / 10000
  impact_cost  <- impact_coef * abs_trades^2
  spread_cost + impact_cost
}


#' sqrt_tc_model
#'
#' Square-root market impact model: MI = alpha * sigma * sqrt(|Q| / ADV)
#'
#' @param trade_values numeric vector of trade sizes ($ notional)
#' @param ADV numeric, average daily volume ($)
#' @param sigma_bps numeric, daily volatility in bps
#' @param alpha numeric, impact coefficient (default 0.1)
#' @param spread_bps numeric
#' @return numeric vector of total costs ($ per trade)
sqrt_tc_model <- function(trade_values, ADV = 1e7, sigma_bps = 150, alpha = 0.1,
                           spread_bps = 5.0) {
  abs_trade  <- abs(trade_values)
  sigma_frac <- sigma_bps / 10000
  impact     <- alpha * sigma_frac * sqrt(abs_trade / (ADV + 1e-10)) * abs_trade
  spread     <- abs_trade * spread_bps / 10000
  impact + spread
}


#' rolling_tc_attribution
#'
#' Decompose realized transaction costs into spread vs impact components.
#'
#' @param bt BacktestResult from run_backtest
#' @param window integer, rolling window for attribution
#' @return data.frame with rolling cost decomposition
rolling_tc_attribution <- function(bt, window = 60L) {
  n       <- length(bt$returns)
  dates   <- index(bt$returns)
  costs   <- as.numeric(bt$tc_costs)
  to      <- as.numeric(bt$turnover)

  # Rolling stats
  roll_cost  <- zoo::rollmean(costs,    k = window, fill = NA, align = "right")
  roll_to    <- zoo::rollmean(to,       k = window, fill = NA, align = "right")
  cost_per_to <- costs / (to + 1e-10)
  roll_cpt   <- zoo::rollmean(cost_per_to, k = window, fill = NA, align = "right")

  data.frame(
    date          = as.Date(dates),
    rolling_cost  = as.numeric(roll_cost),
    rolling_turnover = as.numeric(roll_to),
    cost_per_turnover = as.numeric(roll_cpt),
    stringsAsFactors = FALSE
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

#' bootstrap_sharpe_ci
#'
#' Compute bootstrap confidence intervals for the Sharpe ratio.
#' Uses stationary (block) bootstrap to account for serial correlation.
#'
#' @param returns xts or numeric return series
#' @param n_bootstrap integer, number of bootstrap replications
#' @param block_size integer, block size for block bootstrap
#' @param rf numeric, risk-free rate
#' @param conf_level numeric, confidence level (default 0.95)
#' @param annual_periods integer
#' @return list with sharpe, ci_lower, ci_upper, bootstrap_dist
bootstrap_sharpe_ci <- function(
  returns,
  n_bootstrap    = 2000L,
  block_size     = 20L,
  rf             = 0.0,
  conf_level     = 0.95,
  annual_periods = 252L
) {
  if (is.xts(returns)) r <- as.numeric(returns) else r <- as.numeric(returns)
  r <- r[!is.na(r)]
  n <- length(r)

  # Observed Sharpe
  mu_obs  <- mean(r) * annual_periods
  sig_obs <- sd(r) * sqrt(annual_periods)
  sharpe_obs <- (mu_obs - rf * annual_periods) / sig_obs

  # Block bootstrap
  sharpe_boot <- numeric(n_bootstrap)
  for (b in seq_len(n_bootstrap)) {
    r_boot <- block_bootstrap_sample(r, block_size)
    mu_b   <- mean(r_boot) * annual_periods
    sig_b  <- sd(r_boot) * sqrt(annual_periods)
    sharpe_boot[b] <- if (sig_b > 0) (mu_b - rf * annual_periods) / sig_b else 0.0
  }

  alpha  <- 1 - conf_level
  ci_lo  <- quantile(sharpe_boot, alpha / 2)
  ci_hi  <- quantile(sharpe_boot, 1 - alpha / 2)

  # Bias-corrected
  bias   <- mean(sharpe_boot) - sharpe_obs
  sharpe_bc <- sharpe_obs - bias

  list(
    sharpe           = sharpe_obs,
    sharpe_bc        = sharpe_bc,
    ci_lower         = ci_lo,
    ci_upper         = ci_hi,
    se               = sd(sharpe_boot),
    bootstrap_dist   = sharpe_boot,
    n_bootstrap      = n_bootstrap,
    block_size       = block_size,
    conf_level       = conf_level
  )
}


#' block_bootstrap_sample
#'
#' Generate a block bootstrap sample of size n from vector x.
#' Uses stationary block bootstrap (Politis & Romano 1994).
#'
#' @param x numeric vector
#' @param block_size integer, mean block size
#' @return numeric vector of same length as x
block_bootstrap_sample <- function(x, block_size = 20L) {
  n    <- length(x)
  boot <- numeric(n)
  i    <- 1L
  while (i <= n) {
    start <- sample.int(n, 1)
    len   <- min(
      rgeom(1, prob = 1.0 / block_size),  # geometric block length
      n - i + 1L
    )
    len <- max(len, 1L)
    end_src <- min(start + len - 1L, n)
    actual_len <- end_src - start + 1L
    boot[i:min(i + actual_len - 1L, n)] <- x[start:end_src]
    i <- i + actual_len
  }
  boot[seq_len(n)]
}


#' bootstrap_metric_ci
#'
#' Generalized bootstrap CI for any performance metric.
#'
#' @param returns xts or numeric
#' @param metric_fn function(returns) -> scalar
#' @param n_bootstrap integer
#' @param block_size integer
#' @param conf_level numeric
#' @return list with observed, ci_lower, ci_upper, se, bootstrap_dist
bootstrap_metric_ci <- function(
  returns,
  metric_fn,
  n_bootstrap = 1000L,
  block_size  = 20L,
  conf_level  = 0.95
) {
  if (is.xts(returns)) r <- as.numeric(returns) else r <- as.numeric(returns)
  r <- r[!is.na(r)]

  obs_val <- metric_fn(r)

  boot_vals <- vapply(seq_len(n_bootstrap), function(b) {
    r_b <- block_bootstrap_sample(r, block_size)
    tryCatch(metric_fn(r_b), error = function(e) NA_real_)
  }, numeric(1))

  boot_vals <- boot_vals[!is.na(boot_vals)]
  alpha <- 1 - conf_level

  list(
    observed      = obs_val,
    ci_lower      = quantile(boot_vals, alpha/2),
    ci_upper      = quantile(boot_vals, 1 - alpha/2),
    se            = sd(boot_vals),
    bootstrap_dist = boot_vals
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Rolling Backtest and Walk-Forward
# ─────────────────────────────────────────────────────────────────────────────

#' walk_forward_backtest
#'
#' Run walk-forward backtest: re-estimate model on rolling training window,
#' apply to out-of-sample test window.
#'
#' @param returns xts or matrix of asset returns
#' @param signal_fn function(train_returns) -> signal_vector (for next period)
#' @param train_window integer, training window length
#' @param test_window integer, out-of-sample test length per fold
#' @param refit_freq integer, refit every N periods
#' @param tc_model list, transaction cost model
#' @return BacktestResult with rolling performance
walk_forward_backtest <- function(
  returns,
  signal_fn,
  train_window = 252L,
  test_window  = 21L,
  refit_freq   = 21L,
  tc_model     = list(type = "bps", cost_bps = 10.0)
) {
  R <- as.matrix(returns)
  n <- nrow(R)
  n_assets <- ncol(R)
  dates <- if (is.xts(returns)) index(returns) else seq_len(n)

  all_signals <- matrix(0.0, n, n_assets)
  colnames(all_signals) <- colnames(R)

  t <- train_window + 1L
  last_signal <- rep(1/n_assets, n_assets)

  while (t <= n) {
    if ((t - train_window) %% refit_freq == 0 || t == train_window + 1L) {
      train_idx  <- (t - train_window):(t - 1L)
      train_ret  <- R[train_idx, , drop = FALSE]
      new_signal <- tryCatch(
        signal_fn(train_ret),
        error = function(e) last_signal
      )
      new_signal <- as.numeric(new_signal)
      if (length(new_signal) == n_assets && all(is.finite(new_signal))) {
        last_signal <- new_signal
      }
    }
    all_signals[t, ] <- last_signal
    t <- t + 1L
  }

  # Run full backtest on out-of-sample period
  oos_idx <- (train_window + 1L):n
  sig_xts  <- xts(all_signals[oos_idx, , drop = FALSE], order.by = dates[oos_idx])
  ret_xts  <- xts(R[oos_idx, , drop = FALSE], order.by = dates[oos_idx])

  run_backtest(sig_xts, ret_xts, tc_model = tc_model)
}


#' print.BacktestResult
#'
#' Print summary of backtest result.
print.BacktestResult <- function(x, ...) {
  p <- x$performance
  cat("=== Backtest Results ===\n")
  cat(sprintf("Periods:         %d\n", p$n_periods))
  cat(sprintf("Total Return:    %.2f%%\n", p$total_return * 100))
  cat(sprintf("Ann. Return:     %.2f%%\n", p$ann_return * 100))
  cat(sprintf("Ann. Vol:        %.2f%%\n", p$ann_vol * 100))
  cat(sprintf("Sharpe Ratio:    %.3f\n", p$sharpe_ratio %||% NA))
  cat(sprintf("Sortino Ratio:   %.3f\n", p$sortino_ratio %||% NA))
  cat(sprintf("Calmar Ratio:    %.3f\n", p$calmar_ratio %||% NA))
  cat(sprintf("Max Drawdown:    %.2f%%\n", p$max_drawdown * 100))
  cat(sprintf("Hit Rate:        %.1f%%\n", p$hit_rate * 100))
  cat(sprintf("VaR (95%%):       %.2f%%\n", p$var_95 * 100))
  cat(sprintf("CVaR (95%%):      %.2f%%\n", p$cvar_95 * 100))
  cat(sprintf("Skewness:        %.3f\n", p$skewness %||% NA))
  cat(sprintf("Excess Kurtosis: %.3f\n", p$excess_kurtosis %||% NA))
  invisible(x)
}
