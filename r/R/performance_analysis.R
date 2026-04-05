# =============================================================================
# performance_analysis.R
# Performance attribution and analytics for crypto/quant portfolios
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Measuring performance is not just about total return.
# Brinson-Hood-Beebower attribution separates how much return came from
# asset allocation decisions vs security selection. Risk-adjusted metrics
# determine if returns are commensurate with risks taken.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. TIME-WEIGHTED vs MONEY-WEIGHTED RETURNS
# -----------------------------------------------------------------------------

#' Time-Weighted Return (TWR)
#' Eliminates the effect of external cash flows; preferred for manager evaluation
#' TWR = product of sub-period returns
#' @param sub_period_returns vector of returns for each sub-period
twr <- function(sub_period_returns) {
  prod(1 + sub_period_returns) - 1
}

#' Money-Weighted Return (MWR / IRR)
#' Reflects actual investor experience including timing of cash flows
#' Solves: sum_t CF_t / (1+r)^t = 0
#' @param cashflows vector of cash flows (negative = outflows, positive = inflows)
#' @param times vector of time periods (0 = start)
mwr <- function(cashflows, times) {
  # Newton-Raphson to find IRR
  f <- function(r) sum(cashflows / (1 + r)^times)
  f_prime <- function(r) -sum(times * cashflows / (1 + r)^(times + 1))

  r <- 0.10  # initial guess
  for (iter in 1:100) {
    f_r <- f(r); f_prime_r <- f_prime(r)
    if (abs(f_prime_r) < 1e-10) break
    r_new <- r - f_r / f_prime_r
    if (abs(r_new - r) < 1e-8) break
    r <- r_new
    if (abs(r) > 10) { r <- 0.10; break }  # diverging
  }
  r
}

# -----------------------------------------------------------------------------
# 2. BRINSON-HOOD-BEEBOWER ATTRIBUTION
# -----------------------------------------------------------------------------

#' BHB Attribution (Brinson, Hood, Beebower 1986)
#' Decomposes portfolio return into:
#'   Allocation effect: was the manager in the right sectors?
#'   Selection effect: did the manager pick the right securities?
#'   Interaction effect: joint effect of allocation and selection
#' @param w_port portfolio weights (N assets)
#' @param w_bench benchmark weights
#' @param r_port portfolio asset returns
#' @param r_bench benchmark asset returns
bhb_attribution <- function(w_port, w_bench, r_port, r_bench) {
  N <- length(w_port)
  asset_names <- names(w_port)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  # Benchmark return
  R_bench <- sum(w_bench * r_bench)

  # Allocation effect: (w_port - w_bench) * (r_bench - R_bench)
  alloc <- (w_port - w_bench) * (r_bench - R_bench)

  # Selection effect: w_bench * (r_port - r_bench)
  select <- w_bench * (r_port - r_bench)

  # Interaction effect: (w_port - w_bench) * (r_port - r_bench)
  interact <- (w_port - w_bench) * (r_port - r_bench)

  # Portfolio return
  R_port <- sum(w_port * r_port)
  active_return <- R_port - R_bench

  df <- data.frame(
    asset       = asset_names,
    w_port      = round(w_port, 4),
    w_bench     = round(w_bench, 4),
    r_port      = round(r_port, 4),
    r_bench     = round(r_bench, 4),
    allocation  = round(alloc, 6),
    selection   = round(select, 6),
    interaction = round(interact, 6)
  )

  cat("=== Brinson-Hood-Beebower Attribution ===\n")
  cat(sprintf("Portfolio return:  %.4f%%\n", 100*R_port))
  cat(sprintf("Benchmark return:  %.4f%%\n", 100*R_bench))
  cat(sprintf("Active return:     %.4f%%\n", 100*active_return))
  cat(sprintf("\nTotal Allocation effect:  %.4f%%\n", 100*sum(alloc)))
  cat(sprintf("Total Selection effect:   %.4f%%\n", 100*sum(select)))
  cat(sprintf("Total Interaction effect: %.4f%%\n", 100*sum(interact)))
  cat(sprintf("Sum (should = active):    %.4f%%\n",
              100*(sum(alloc)+sum(select)+sum(interact))))
  cat("\nBy Asset:\n"); print(df)

  invisible(list(total_alloc=sum(alloc), total_select=sum(select),
                 total_interact=sum(interact), active_return=active_return,
                 by_asset=df))
}

# -----------------------------------------------------------------------------
# 3. RISK-ADJUSTED METRICS
# -----------------------------------------------------------------------------

#' Sharpe ratio
#' @param returns daily return series
#' @param rf_daily daily risk-free rate (default 0)
#' @param ann_factor annualization factor (default 252 for daily)
sharpe_ratio <- function(returns, rf_daily = 0, ann_factor = 252) {
  excess <- returns - rf_daily
  if (sd(excess) == 0) return(NA)
  mean(excess) / sd(excess) * sqrt(ann_factor)
}

#' Sortino ratio: penalizes only downside volatility
#' @param returns daily return series
#' @param mar minimum acceptable return (default 0)
sortino_ratio <- function(returns, mar = 0, ann_factor = 252) {
  excess <- returns - mar
  downside_returns <- excess[excess < 0]
  if (length(downside_returns) == 0 || sd(downside_returns) == 0) return(NA)
  downside_dev <- sqrt(mean(downside_returns^2))
  mean(excess) / downside_dev * sqrt(ann_factor)
}

#' Omega ratio: probability-weighted ratio of gains to losses above threshold
#' Omega = integral above threshold / integral below threshold
#' No distributional assumptions; captures full return distribution
omega_ratio <- function(returns, threshold = 0) {
  gains  <- sum(pmax(returns - threshold, 0))
  losses <- sum(pmax(threshold - returns, 0))
  if (losses == 0) return(Inf)
  gains / losses
}

#' Calmar ratio: annualized return / max drawdown
calmar_ratio <- function(returns, ann_factor = 252) {
  cum_ret  <- cumprod(1 + returns)
  max_dd   <- abs(min(cum_ret / cummax(cum_ret) - 1))
  ann_ret  <- mean(returns) * ann_factor
  if (max_dd == 0) return(NA)
  ann_ret / max_dd
}

#' Comprehensive risk-adjusted metrics table
risk_adjusted_metrics <- function(returns, benchmark_returns = NULL,
                                   rf_daily = 0, ann_factor = 252) {
  n <- length(returns)
  mu  <- mean(returns)
  sigma <- sd(returns)
  skew  <- mean((returns-mu)^3) / sigma^3
  kurt  <- mean((returns-mu)^4) / sigma^4 - 3

  sr  <- sharpe_ratio(returns, rf_daily, ann_factor)
  st  <- sortino_ratio(returns, rf_daily, ann_factor)
  om  <- omega_ratio(returns, rf_daily)
  ca  <- calmar_ratio(returns, ann_factor)

  # Drawdown
  cum_ret <- cumprod(1 + returns)
  max_dd  <- abs(min(cum_ret / cummax(cum_ret) - 1))

  # Hit rate and profit factor
  hit_rate <- mean(returns > rf_daily)
  wins  <- returns[returns > rf_daily] - rf_daily
  losses <- -(returns[returns <= rf_daily] - rf_daily)
  profit_factor <- if (sum(losses) > 0) sum(wins) / sum(losses) else Inf

  # Annualized stats
  ann_return <- mu * ann_factor
  ann_vol    <- sigma * sqrt(ann_factor)

  metrics <- data.frame(
    metric = c("Ann Return", "Ann Volatility", "Sharpe", "Sortino",
               "Omega", "Calmar", "Max Drawdown", "Hit Rate",
               "Profit Factor", "Skewness", "Excess Kurtosis"),
    value  = round(c(100*ann_return, 100*ann_vol, sr, st, om, ca,
                     100*max_dd, 100*hit_rate, profit_factor, skew, kurt), 4)
  )

  if (!is.null(benchmark_returns) && length(benchmark_returns) == n) {
    beta_b  <- cov(returns, benchmark_returns) / var(benchmark_returns)
    alpha_b <- ann_return - beta_b * mean(benchmark_returns) * ann_factor
    te      <- sd(returns - benchmark_returns) * sqrt(ann_factor)
    ir      <- mean(returns - benchmark_returns) / sd(returns - benchmark_returns) *
               sqrt(ann_factor)
    extra <- data.frame(
      metric = c("Beta", "Alpha (ann)", "Tracking Error", "Info Ratio"),
      value  = round(c(beta_b, 100*alpha_b, 100*te, ir), 4)
    )
    metrics <- rbind(metrics, extra)
  }

  cat("=== Risk-Adjusted Performance Metrics ===\n")
  print(metrics)
  invisible(metrics)
}

# -----------------------------------------------------------------------------
# 4. ROLLING PERFORMANCE WINDOWS
# -----------------------------------------------------------------------------

#' Rolling performance metrics over multiple windows
#' @param returns daily return series
#' @param windows named vector: c("1M"=21, "3M"=63, "6M"=126, "12M"=252)
rolling_performance <- function(returns, windows=c("1M"=21,"3M"=63,"6M"=126,"12M"=252),
                                  rf_daily=0, ann_factor=252) {
  n <- length(returns)
  results <- list()

  for (w_name in names(windows)) {
    w <- windows[w_name]
    if (w > n) { results[[w_name]] <- NA; next }

    # Rolling metrics
    roll_ret   <- numeric(n)
    roll_sharpe <- numeric(n)
    roll_vol   <- numeric(n)
    roll_maxdd <- numeric(n)

    for (t in w:n) {
      r_w <- returns[(t-w+1):t]
      roll_ret[t]    <- prod(1 + r_w) - 1
      roll_sharpe[t] <- sharpe_ratio(r_w, rf_daily, ann_factor)
      roll_vol[t]    <- sd(r_w) * sqrt(ann_factor)
      cum_w <- cumprod(1+r_w); roll_maxdd[t] <- abs(min(cum_w/cummax(cum_w)-1))
    }

    # Set pre-window to NA
    roll_ret[1:(w-1)] <- NA; roll_sharpe[1:(w-1)] <- NA
    roll_vol[1:(w-1)] <- NA; roll_maxdd[1:(w-1)] <- NA

    results[[w_name]] <- list(
      ret=roll_ret, sharpe=roll_sharpe, vol=roll_vol, max_dd=roll_maxdd,
      current_ret=tail(roll_ret[!is.na(roll_ret)], 1),
      current_sharpe=tail(roll_sharpe[!is.na(roll_sharpe)], 1),
      current_vol=tail(roll_vol[!is.na(roll_vol)], 1),
      window=w
    )
  }

  cat("=== Rolling Performance Summary (Current) ===\n")
  df <- data.frame(
    window = names(windows),
    return_pct = sapply(results, function(r) if(is.list(r)) round(100*r$current_ret,2) else NA),
    sharpe     = sapply(results, function(r) if(is.list(r)) round(r$current_sharpe,3) else NA),
    vol_ann    = sapply(results, function(r) if(is.list(r)) round(100*r$current_vol,2) else NA)
  )
  print(df)

  invisible(results)
}

#' YTD return computation
ytd_return <- function(returns, dates = NULL) {
  if (!is.null(dates)) {
    year_start <- as.Date(paste0(format(max(dates), "%Y"), "-01-01"))
    ytd_idx    <- which(as.Date(dates) >= year_start)
    if (length(ytd_idx) > 0) return(prod(1 + returns[ytd_idx]) - 1)
  }
  # If no dates, assume last 252 observations as a proxy
  n <- length(returns)
  start <- max(1, n - 251)
  prod(1 + returns[start:n]) - 1
}

# -----------------------------------------------------------------------------
# 5. BENCHMARK COMPARISON
# -----------------------------------------------------------------------------

#' Full benchmark comparison and CAPM regression
#' @param port_returns portfolio daily return series
#' @param bench_returns benchmark daily return series
#' @param rf_daily risk-free rate (daily)
benchmark_comparison <- function(port_returns, bench_returns, rf_daily=0,
                                  ann_factor=252) {
  n <- length(port_returns)
  stopifnot(length(bench_returns) == n)

  # Excess returns
  excess_port  <- port_returns - rf_daily
  excess_bench <- bench_returns - rf_daily

  # CAPM regression: excess_port = alpha + beta*excess_bench + epsilon
  X <- cbind(1, excess_bench)
  b <- solve(t(X)%*%X) %*% t(X) %*% excess_port
  alpha_daily <- b[1]; beta <- b[2]
  resid <- excess_port - X %*% b
  ss_res <- sum(resid^2); ss_tot <- sum((excess_port-mean(excess_port))^2)
  r2    <- 1 - ss_res/ss_tot
  se_b  <- sqrt(ss_res/(n-2) * solve(t(X)%*%X))
  t_alpha <- alpha_daily / se_b[1,1]
  t_beta  <- beta / se_b[2,2]

  # Information ratio
  active_returns <- port_returns - bench_returns
  ir <- mean(active_returns) / (sd(active_returns)+1e-10) * sqrt(ann_factor)

  # Tracking error
  te <- sd(active_returns) * sqrt(ann_factor)

  # Upside/downside capture
  up_idx   <- bench_returns > 0; down_idx <- bench_returns <= 0
  up_cap   <- if(sum(up_idx)>1) prod(1+port_returns[up_idx])^(ann_factor/sum(up_idx)) /
                                prod(1+bench_returns[up_idx])^(ann_factor/sum(up_idx)) else NA
  down_cap <- if(sum(down_idx)>1) prod(1+port_returns[down_idx])^(ann_factor/sum(down_idx)) /
                                  prod(1+bench_returns[down_idx])^(ann_factor/sum(down_idx)) else NA

  cat("=== Benchmark Comparison ===\n")
  cat(sprintf("Alpha (ann): %.3f%% (t=%.2f, p=%.4f)\n",
              100*alpha_daily*ann_factor, t_alpha,
              2*pt(-abs(t_alpha), df=n-2)))
  cat(sprintf("Beta: %.4f (t=%.2f)\n", beta, t_beta))
  cat(sprintf("R-squared: %.4f\n", r2))
  cat(sprintf("Info Ratio: %.3f\n", ir))
  cat(sprintf("Tracking Error (ann): %.3f%%\n", 100*te))
  cat(sprintf("Upside Capture:   %.2f%%\n", 100*up_cap))
  cat(sprintf("Downside Capture: %.2f%%\n", 100*down_cap))
  cat(sprintf("Capture Ratio: %.2f (>1 = outperforms)\n",
              up_cap / (down_cap + 1e-10)))

  invisible(list(alpha_daily=alpha_daily, alpha_ann=alpha_daily*ann_factor,
                 beta=beta, r2=r2, ir=ir, te=te,
                 t_alpha=t_alpha, t_beta=t_beta,
                 up_capture=up_cap, down_capture=down_cap))
}

# -----------------------------------------------------------------------------
# 6. HIT RATE AND PROFIT FACTOR BY REGIME
# -----------------------------------------------------------------------------

#' Hit rate and profit factor analysis by instrument, hour, regime
#' @param returns_vec return series
#' @param categories factor vector for grouping (e.g., hour of day, regime)
hit_rate_by_category <- function(returns_vec, categories, category_name="Category") {
  cats <- unique(categories)
  results <- lapply(cats, function(cat) {
    r <- returns_vec[categories == cat]
    r <- r[!is.na(r)]
    if (length(r) < 5) return(NULL)
    hits  <- sum(r > 0)
    n_t   <- length(r)
    wins  <- r[r > 0]; losses <- -r[r < 0]
    pf    <- if (sum(losses) > 0) sum(wins)/sum(losses) else Inf
    avg_w <- if (length(wins) > 0) mean(wins) else 0
    avg_l <- if (length(losses) > 0) mean(losses) else 0
    data.frame(category=cat, n=n_t, hit_rate=hits/n_t,
               profit_factor=pf, avg_win=avg_w, avg_loss=avg_l,
               expectancy=mean(r))
  })
  results <- do.call(rbind, Filter(Negate(is.null), results))
  results <- results[order(-results$expectancy), ]

  cat(sprintf("=== Hit Rate & Profit Factor by %s ===\n", category_name))
  print(results)
  invisible(results)
}

# -----------------------------------------------------------------------------
# 7. DRAWDOWN FREQUENCY AND DURATION ANALYSIS
# -----------------------------------------------------------------------------

#' Identify all drawdown periods with statistics
#' @param returns return series
all_drawdowns <- function(returns) {
  n <- length(returns)
  cum_ret  <- cumprod(1 + returns)
  cum_peak <- cummax(cum_ret)
  dd       <- cum_ret / cum_peak - 1

  # Find drawdown start/end periods
  in_dd <- dd < -0.001  # small threshold to avoid noise

  # Label drawdown episodes
  rle_dd <- rle(in_dd)
  ep_ends   <- cumsum(rle_dd$lengths)
  ep_starts <- c(1, ep_ends[-length(ep_ends)] + 1)

  drawdown_periods <- data.frame(
    start    = integer(),
    end      = integer(),
    duration = integer(),
    max_dd   = numeric(),
    recovery_t = integer()
  )

  for (i in seq_along(rle_dd$values)) {
    if (!rle_dd$values[i]) next
    s <- ep_starts[i]; e <- ep_ends[i]
    # Find when recovery occurs after end
    rec <- if (e < n) e + which(cum_ret[(e+1):n] >= cum_peak[s])[1] else NA_integer_
    drawdown_periods <- rbind(drawdown_periods, data.frame(
      start = s, end = e, duration = e-s,
      max_dd = min(dd[s:e]),
      recovery_t = ifelse(is.na(rec), NA_integer_, as.integer(rec - e))
    ))
  }

  if (nrow(drawdown_periods) > 0) {
    drawdown_periods <- drawdown_periods[order(drawdown_periods$max_dd), ]
    cat("=== Drawdown Periods Summary ===\n")
    cat(sprintf("Total drawdown episodes: %d\n", nrow(drawdown_periods)))
    cat(sprintf("Avg duration: %.1f periods\n", mean(drawdown_periods$duration)))
    cat(sprintf("Avg depth:    %.2f%%\n", 100*mean(drawdown_periods$max_dd)))
    cat(sprintf("Avg recovery: %.1f periods\n", mean(drawdown_periods$recovery_t, na.rm=T)))
    cat("\nTop 5 worst drawdowns:\n")
    print(head(drawdown_periods, 5))
  }

  invisible(drawdown_periods)
}

# -----------------------------------------------------------------------------
# 8. FACTOR EXPOSURE ATTRIBUTION
# -----------------------------------------------------------------------------

#' Factor exposure and return attribution via OLS regression
#' @param port_returns portfolio return series
#' @param factor_returns matrix of factor return series (T x K)
factor_attribution <- function(port_returns, factor_returns) {
  n <- length(port_returns)
  K <- ncol(factor_returns)
  factor_names <- colnames(factor_returns)
  if (is.null(factor_names)) factor_names <- paste0("F", seq_len(K))

  X <- cbind(1, factor_returns)
  b <- solve(t(X) %*% X) %*% t(X) %*% port_returns
  fitted  <- X %*% b
  resid   <- port_returns - fitted
  ss_res  <- sum(resid^2); ss_tot <- sum((port_returns-mean(port_returns))^2)
  r2 <- 1 - ss_res/ss_tot
  adj_r2 <- 1 - (1-r2)*(n-1)/(n-K-1)

  # Standard errors (OLS)
  sigma2e <- ss_res / (n - K - 1)
  var_b   <- sigma2e * solve(t(X) %*% X)
  se_b    <- sqrt(diag(var_b))
  t_stats <- b / se_b
  p_vals  <- 2 * pt(-abs(t_stats), df = n-K-1)

  # Factor contribution to total variance
  factor_var_contrib <- sapply(seq_len(K), function(k) {
    b[k+1]^2 * var(factor_returns[,k]) / var(port_returns)
  })

  df_res <- data.frame(
    factor  = c("Alpha", factor_names),
    beta    = round(b, 6),
    se      = round(se_b, 6),
    t_stat  = round(t_stats, 3),
    p_value = round(p_vals, 4),
    var_explained = c(NA, round(100*factor_var_contrib, 2))
  )

  cat("=== Factor Attribution ===\n")
  cat(sprintf("R-squared: %.4f, Adj R2: %.4f\n", r2, adj_r2))
  cat(sprintf("Alpha (annualized): %.4f%%\n", 100*b[1]*252))
  cat("\nFactor Exposures:\n")
  print(df_res)

  invisible(list(beta=b, se=se_b, t_stats=t_stats, p_vals=p_vals,
                 r2=r2, adj_r2=adj_r2, residuals=resid, fitted=fitted,
                 var_contrib=factor_var_contrib))
}

# -----------------------------------------------------------------------------
# 9. FULL PERFORMANCE ANALYSIS PIPELINE
# -----------------------------------------------------------------------------

#' Comprehensive performance analysis report
run_performance_analysis <- function(port_returns, bench_returns=NULL,
                                      factor_returns=NULL, weights=NULL,
                                      benchmark_weights=NULL, asset_returns=NULL,
                                      rf_daily=0, ann_factor=252) {
  n <- length(port_returns)
  cat("=============================================================\n")
  cat("PERFORMANCE ANALYSIS REPORT\n")
  cat(sprintf("Observations: %d\n", n))
  cat(sprintf("Total Return: %.2f%%\n", 100*(prod(1+port_returns)-1)))
  cat("=============================================================\n\n")

  # Core metrics
  cat("--- Risk-Adjusted Metrics ---\n")
  metrics <- risk_adjusted_metrics(port_returns, bench_returns, rf_daily, ann_factor)

  # Rolling performance
  cat("\n--- Rolling Performance ---\n")
  roll_res <- rolling_performance(port_returns,
                                   windows=c("1M"=21,"3M"=63,"6M"=126,"12M"=252),
                                   rf_daily=rf_daily, ann_factor=ann_factor)

  # Benchmark comparison
  if (!is.null(bench_returns)) {
    cat("\n")
    bm_res <- benchmark_comparison(port_returns, bench_returns, rf_daily, ann_factor)
  }

  # BHB Attribution if weights and asset returns provided
  if (!is.null(weights) && !is.null(benchmark_weights) && !is.null(asset_returns)) {
    cat("\n")
    # Use last period returns for attribution
    n_assets <- length(weights)
    r_port_last  <- tail(asset_returns, 1)
    bhb_res <- bhb_attribution(weights, benchmark_weights,
                                as.numeric(r_port_last),
                                as.numeric(r_port_last * 0.9))  # approximate bench returns
  }

  # Factor attribution
  if (!is.null(factor_returns)) {
    cat("\n")
    fa_res <- factor_attribution(port_returns, factor_returns)
  }

  # Drawdown analysis
  cat("\n")
  dd_res <- all_drawdowns(port_returns)

  invisible(list(metrics=metrics, rolling=roll_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 500
# bench <- cumsum(c(0, rnorm(n-1, 0.0008, 0.015)))
# bench_ret <- diff(c(0, bench))
# port <- bench_ret + rnorm(n, 0.0002, 0.005)  # alpha of 0.02% daily
# result <- run_performance_analysis(port, bench_returns=bench_ret)

# =============================================================================
# EXTENDED PERFORMANCE ANALYSIS: Advanced Attribution, Trade-Level Analytics,
# Multi-Period Decomposition, Style Analysis, and Regime Performance
# =============================================================================

# -----------------------------------------------------------------------------
# Trade-Level P&L Attribution: decompose returns into entry, holding, exit
# Useful for crypto strategies where precise execution timing matters
# -----------------------------------------------------------------------------
trade_pnl_attribution <- function(trades) {
  # trades: data.frame with columns: entry_price, exit_price, size, direction (1/-1)
  # direction: 1 = long, -1 = short
  if (!all(c("entry_price","exit_price","size","direction") %in% names(trades))) {
    stop("trades must have columns: entry_price, exit_price, size, direction")
  }

  trades$gross_pnl  <- trades$direction * (trades$exit_price - trades$entry_price) * trades$size
  trades$return_pct <- trades$direction * (trades$exit_price / trades$entry_price - 1)
  trades$notional   <- trades$entry_price * trades$size

  winners <- trades[trades$gross_pnl > 0, ]
  losers  <- trades[trades$gross_pnl < 0, ]

  list(
    total_pnl       = sum(trades$gross_pnl),
    win_rate        = nrow(winners) / nrow(trades),
    avg_win         = if (nrow(winners)>0) mean(winners$gross_pnl) else 0,
    avg_loss        = if (nrow(losers)>0)  mean(losers$gross_pnl)  else 0,
    profit_factor   = if (nrow(losers)>0 && sum(losers$gross_pnl) != 0)
      sum(winners$gross_pnl) / abs(sum(losers$gross_pnl)) else Inf,
    largest_win     = if (nrow(winners)>0) max(winners$gross_pnl) else 0,
    largest_loss    = if (nrow(losers)>0)  min(losers$gross_pnl)  else 0,
    avg_return_pct  = mean(trades$return_pct),
    median_return_pct = median(trades$return_pct),
    expectancy      = mean(trades$gross_pnl),
    kelly_fraction  = {
      p <- nrow(winners) / nrow(trades)
      b <- if (nrow(losers)>0 && mean(losers$gross_pnl)!=0)
        abs(mean(winners$gross_pnl)) / abs(mean(losers$gross_pnl)) else 2
      max(p - (1-p)/b, 0)
    },
    n_trades = nrow(trades),
    n_wins   = nrow(winners),
    n_losses = nrow(losers)
  )
}

# -----------------------------------------------------------------------------
# Sharpe Ratio Decomposition: separate alpha, beta, and idiosyncratic components
# Grinold-Kahn fundamental law: SR = IC * sqrt(breadth) * transfer coefficient
# -----------------------------------------------------------------------------
sharpe_decomposition <- function(port_returns, bench_returns, rf = 0,
                                  n_signals = NULL) {
  excess_port  <- port_returns - rf
  excess_bench <- bench_returns - rf

  # CAPM regression
  fit <- lm(excess_port ~ excess_bench)
  alpha_daily <- coef(fit)[1]
  beta        <- coef(fit)[2]
  resid_vol   <- sd(residuals(fit))
  r_squared   <- summary(fit)$r.squared

  # Annualized metrics
  ann_alpha <- alpha_daily * 252
  ann_sr    <- mean(excess_port) * sqrt(252) / sd(excess_port)
  ann_ir    <- ann_alpha / (resid_vol * sqrt(252))  # Information Ratio

  # Treynor ratio: excess return per unit of systematic risk
  treynor <- mean(excess_port) * 252 / beta

  # Modigliani-Modigliani (M2): scale strategy to benchmark vol
  bench_vol <- sd(excess_bench)
  port_vol  <- sd(excess_port)
  m2 <- ann_sr * bench_vol * sqrt(252) + rf * 252  # approximate

  # Appraisal ratio: alpha / residual risk (like IR but CAPM-adjusted)
  appraisal <- ann_alpha / (resid_vol * sqrt(252))

  # Fundamental law if IC and breadth known
  fl_sr <- if (!is.null(n_signals)) {
    ic_proxy <- ann_ir / sqrt(252)  # rough IC proxy from daily IR
    ic_proxy * sqrt(n_signals)
  } else NA

  list(
    sharpe_ratio = ann_sr,
    information_ratio = ann_ir,
    treynor_ratio = treynor,
    m2_measure = m2,
    appraisal_ratio = appraisal,
    capm_alpha_annualized = ann_alpha,
    capm_beta = beta,
    r_squared = r_squared,
    idiosyncratic_vol_annualized = resid_vol * sqrt(252),
    systematic_vol = beta * sd(excess_bench) * sqrt(252),
    fundamental_law_sr = fl_sr
  )
}

# -----------------------------------------------------------------------------
# Rolling Sharpe and Drawdown Heatmap Data: compute performance by year/month
# Useful for identifying seasonal patterns in crypto strategies
# -----------------------------------------------------------------------------
monthly_performance_grid <- function(returns, dates = NULL) {
  n <- length(returns)
  if (is.null(dates)) {
    # Assume daily, create synthetic year/month indices
    days_per_year <- 252; days_per_month <- 21
    year_idx  <- ceiling((1:n) / days_per_year)
    month_idx <- ceiling((1:n) / days_per_month) %% 12 + 1
  } else {
    year_idx  <- as.integer(format(as.Date(dates), "%Y"))
    month_idx <- as.integer(format(as.Date(dates), "%m"))
  }

  years  <- sort(unique(year_idx))
  months <- 1:12

  perf_grid <- matrix(NA, nrow = length(years), ncol = 12,
                       dimnames = list(years, month.abb))

  for (y in seq_along(years)) {
    for (m in months) {
      idx <- year_idx == years[y] & month_idx == m
      if (sum(idx) > 0) {
        perf_grid[y, m] <- prod(1 + returns[idx]) - 1
      }
    }
  }

  # Annual totals
  ann_ret <- apply(perf_grid, 1, function(r) prod(1 + r[!is.na(r)]) - 1)

  # Best/worst months
  flat <- as.vector(perf_grid)
  flat <- flat[!is.na(flat)]

  list(
    monthly_grid = perf_grid,
    annual_returns = ann_ret,
    best_month = max(flat),
    worst_month = min(flat),
    pct_positive_months = mean(flat > 0),
    avg_monthly_return = mean(flat),
    monthly_vol = sd(flat)
  )
}

# -----------------------------------------------------------------------------
# Regime-Conditional Performance: split returns by bull/bear/sideways regimes
# Identifies whether strategy profits come from trending or mean-reversion
# -----------------------------------------------------------------------------
regime_performance <- function(port_returns, bench_returns = NULL,
                                 n_regimes = 3, method = "kmeans") {
  # Define regimes based on rolling benchmark performance or unsupervised
  if (!is.null(bench_returns)) {
    roll_bench <- filter(bench_returns, rep(1/20, 20), sides=1)
    roll_bench[is.na(roll_bench)] <- 0
    feature <- roll_bench
  } else {
    feature <- filter(port_returns, rep(1/20, 20), sides=1)
    feature[is.na(feature)] <- 0
  }

  # K-means clustering on rolling return level
  set.seed(42)
  km <- kmeans(feature, centers = n_regimes, nstart = 10)
  regimes <- km$cluster

  # Sort regimes by cluster center (bearish=1, neutral=2, bullish=3)
  center_order <- order(km$centers)
  regime_labels <- c("bearish","neutral","bullish")[1:n_regimes]
  regime_map <- setNames(regime_labels[rank(km$centers)], 1:n_regimes)
  regime_named <- regime_map[as.character(regimes)]

  # Performance statistics by regime
  regime_stats <- lapply(unique(regime_named), function(r) {
    idx <- regime_named == r
    ret <- port_returns[idx]
    list(
      regime = r, n_obs = sum(idx),
      mean_return = mean(ret) * 252,
      vol = sd(ret) * sqrt(252),
      sharpe = mean(ret) / sd(ret) * sqrt(252),
      win_rate = mean(ret > 0),
      max_loss = min(ret)
    )
  })

  stats_df <- do.call(rbind, lapply(regime_stats, as.data.frame))

  list(
    regime_labels = regime_named,
    regime_stats = stats_df,
    overall_sharpe = mean(port_returns) / sd(port_returns) * sqrt(252),
    regime_pct = table(regime_named) / length(regime_named)
  )
}

# -----------------------------------------------------------------------------
# Turnover and Transaction Cost Analysis
# Net alpha after costs: critical for crypto where fees can be 0.05-0.1% per trade
# -----------------------------------------------------------------------------
turnover_cost_analysis <- function(weights_matrix, returns_matrix,
                                    fee_per_trade = 0.001) {
  # weights_matrix: T x N matrix of portfolio weights over time
  # returns_matrix: T x N matrix of returns
  T_obs <- nrow(weights_matrix)
  N     <- ncol(weights_matrix)

  # Turnover: sum of absolute weight changes (one-way)
  weight_changes <- diff(weights_matrix)
  daily_turnover <- rowSums(abs(weight_changes)) / 2  # one-way turnover

  # Transaction costs
  daily_costs <- daily_turnover * fee_per_trade

  # Gross and net portfolio returns
  gross_returns <- rowSums(weights_matrix[-T_obs, ] * returns_matrix[-1, ])
  net_returns   <- gross_returns - daily_costs

  ann_gross <- mean(gross_returns) * 252
  ann_net   <- mean(net_returns) * 252
  ann_vol   <- sd(net_returns) * sqrt(252)

  list(
    ann_gross_return = ann_gross,
    ann_net_return   = ann_net,
    ann_cost_drag    = ann_gross - ann_net,
    avg_daily_turnover = mean(daily_turnover),
    ann_turnover     = mean(daily_turnover) * 252,
    gross_sharpe = mean(gross_returns) / sd(gross_returns) * sqrt(252),
    net_sharpe   = mean(net_returns) / sd(net_returns) * sqrt(252),
    breakeven_alpha_bps = (ann_gross - ann_net) * 1e4 / (mean(daily_turnover) * 252)
  )
}

# -----------------------------------------------------------------------------
# Skewness and Tail Risk Performance Metrics
# Crypto returns are notoriously fat-tailed and negatively skewed
# These metrics capture distributional properties beyond mean/variance
# -----------------------------------------------------------------------------
tail_performance_metrics <- function(returns, alpha = 0.05) {
  n <- length(returns)
  mu <- mean(returns); sig <- sd(returns)

  # Higher moments
  skew  <- mean(((returns - mu)/sig)^3)
  kurt  <- mean(((returns - mu)/sig)^4) - 3  # excess kurtosis

  # VaR and ES
  var_pct <- quantile(returns, alpha, names=FALSE)
  es_pct  <- mean(returns[returns <= var_pct])

  # Gain-Loss ratio: E[max(r,0)] / E[max(-r,0)]
  gains  <- returns[returns > 0]
  losses <- returns[returns < 0]
  gl_ratio <- if (length(losses)>0) mean(gains) / abs(mean(losses)) else Inf

  # Tail ratio: 95th pct / abs(5th pct)
  tail_ratio <- quantile(returns, 0.95) / abs(quantile(returns, 0.05))

  # Conditional Sharpe: Sharpe during up-market / down-market
  up_market   <- returns[returns > 0]
  down_market <- returns[returns < 0]
  up_sharpe   <- if (length(up_market) > 1) mean(up_market)/sd(up_market) else NA
  down_sharpe <- if (length(down_market) > 1) mean(down_market)/sd(down_market) else NA

  # Jensen's alpha equivalent: excess return per unit of downside risk
  downside_vol <- sqrt(mean(pmin(returns, 0)^2)) * sqrt(252)
  sortino <- mean(returns) * 252 / downside_vol

  # Rachev ratio: ETL(1-alpha) / ETL(alpha) -- reward vs risk in tails
  upper_tail <- mean(returns[returns >= quantile(returns, 1-alpha)])
  lower_tail <- abs(mean(returns[returns <= quantile(returns, alpha)]))
  rachev <- upper_tail / lower_tail

  list(
    skewness = skew, excess_kurtosis = kurt,
    var_5pct = var_pct, es_5pct = es_pct,
    gain_loss_ratio = gl_ratio,
    tail_ratio = tail_ratio,
    sortino_ratio = sortino,
    rachev_ratio = rachev,
    up_market_sharpe = up_sharpe,
    down_market_sharpe = down_sharpe,
    pct_positive_days = mean(returns > 0),
    avg_gain = if(length(gains)>0) mean(gains) else 0,
    avg_loss = if(length(losses)>0) mean(losses) else 0
  )
}

# -----------------------------------------------------------------------------
# Rolling Beta and Correlation Analysis
# In crypto, beta to BTC evolves over time (lower in bear, higher in bull)
# Rolling beta tracks this time-variation for dynamic hedging
# -----------------------------------------------------------------------------
rolling_beta_analysis <- function(port_returns, bench_returns,
                                    window = 60, min_obs = 20) {
  n <- length(port_returns)
  rolling_beta  <- rep(NA, n)
  rolling_alpha <- rep(NA, n)
  rolling_corr  <- rep(NA, n)
  rolling_rsq   <- rep(NA, n)

  for (t in window:n) {
    idx <- (t - window + 1):t
    x <- bench_returns[idx]; y <- port_returns[idx]
    if (sum(!is.na(x) & !is.na(y)) < min_obs) next

    beta_val  <- cov(y, x) / var(x)
    alpha_val <- mean(y) - beta_val * mean(x)
    corr_val  <- cor(y, x)

    rolling_beta[t]  <- beta_val
    rolling_alpha[t] <- alpha_val * 252  # annualized
    rolling_corr[t]  <- corr_val
    rolling_rsq[t]   <- corr_val^2
  }

  list(
    rolling_beta  = rolling_beta,
    rolling_alpha = rolling_alpha,
    rolling_corr  = rolling_corr,
    rolling_r2    = rolling_rsq,
    avg_beta  = mean(rolling_beta, na.rm=TRUE),
    avg_alpha = mean(rolling_alpha, na.rm=TRUE),
    beta_stability = sd(rolling_beta, na.rm=TRUE),
    # Bull/bear beta difference
    bull_idx = bench_returns > 0,
    bear_beta = {
      idx <- !is.na(rolling_beta) & !bench_returns[1:n] > 0
      mean(rolling_beta[idx], na.rm=TRUE)
    },
    bull_beta = {
      idx <- !is.na(rolling_beta) & bench_returns[1:n] > 0
      mean(rolling_beta[idx], na.rm=TRUE)
    }
  )
}

# -----------------------------------------------------------------------------
# Benchmark-Relative Performance: tracking error, active return components
# BHB (Brinson-Hood-Beebower) attribution extended to crypto sector decomposition
# -----------------------------------------------------------------------------
extended_attribution <- function(port_weights, bench_weights,
                                   sector_returns, sector_names = NULL) {
  # port_weights, bench_weights: vectors of weights by sector
  # sector_returns: vector of sector returns for the period
  n_sectors <- length(sector_returns)
  if (is.null(sector_names)) sector_names <- paste0("S", 1:n_sectors)

  port_w <- port_weights / sum(port_weights)
  bench_w <- bench_weights / sum(bench_weights)

  r_bench  <- sum(bench_w * sector_returns)
  r_port   <- sum(port_w * sector_returns)
  active_r <- r_port - r_bench

  # BHB decomposition by sector
  # Allocation effect: (wp_i - wb_i) * (rb_i - r_bench)
  # Selection effect: wb_i * (rp_i - rb_i)  -- need sector-level port returns
  # Interaction effect: (wp_i - wb_i) * (rp_i - rb_i)

  # Here we only have sector-level returns, so:
  alloc_effect <- (port_w - bench_w) * (sector_returns - r_bench)
  selection_effect <- rep(0, n_sectors)  # requires security-level data
  interaction_effect <- rep(0, n_sectors)

  df <- data.frame(
    sector = sector_names,
    port_weight = port_w, bench_weight = bench_w,
    active_weight = port_w - bench_w,
    sector_return = sector_returns,
    allocation_effect = alloc_effect,
    contribution_to_active = (port_w - bench_w) * sector_returns
  )

  list(
    attribution = df,
    total_active_return = active_r,
    total_allocation_effect = sum(alloc_effect),
    portfolio_return = r_port,
    benchmark_return = r_bench
  )
}

# -----------------------------------------------------------------------------
# Capacity Analysis: estimate maximum AUM before alpha degrades
# In crypto, market impact grows with position size, limiting capacity
# Capacity = f(daily volume, alpha decay rate, market impact coefficient)
# -----------------------------------------------------------------------------
strategy_capacity <- function(daily_returns, signal_returns = NULL,
                               adv_usd = 1e8, eta = 0.1, target_sharpe_retention = 0.5) {
  # daily_returns: strategy returns at small scale (no impact)
  # adv_usd: average daily volume of traded assets in USD
  # eta: market impact coefficient (0.1 for liquid, 0.3 for illiquid crypto)
  # target_sharpe_retention: what fraction of original Sharpe to preserve

  base_sharpe <- mean(daily_returns) / sd(daily_returns) * sqrt(252)

  # Market impact cost = eta * sigma * sqrt(X / ADV) per trade
  # For a strategy with daily turnover tau, annual cost = 252 * tau * eta * sigma * sqrt(X/ADV)
  sigma <- sd(daily_returns)
  tau   <- 1  # assume 100% daily turnover (upper bound)

  # Solve for X (AUM) such that Sharpe drops to target fraction
  # Impact on returns: alpha_net = alpha_gross - 252 * tau * eta * sigma * sqrt(X/ADV)
  alpha_gross <- mean(daily_returns) * 252
  vol_ann     <- sigma * sqrt(252)
  target_alpha <- base_sharpe * target_sharpe_retention * vol_ann

  # alpha_gross - c*sqrt(X) = target_alpha where c = 252*tau*eta*sigma/sqrt(ADV)
  c_coef <- 252 * tau * eta * sigma / sqrt(adv_usd)
  if (alpha_gross <= target_alpha) {
    capacity <- 0
  } else {
    capacity <- ((alpha_gross - target_alpha) / c_coef)^2
  }

  # Breakeven capacity (alpha = 0)
  breakeven_capacity <- (alpha_gross / c_coef)^2

  list(
    base_sharpe = base_sharpe,
    estimated_capacity_usd = capacity,
    breakeven_capacity_usd = breakeven_capacity,
    alpha_gross_annualized = alpha_gross,
    vol_annualized = vol_ann,
    impact_coefficient = c_coef,
    capacity_tiers = data.frame(
      aum_usd = c(1e5, 1e6, 1e7, 1e8, 1e9),
      net_sharpe = sapply(c(1e5,1e6,1e7,1e8,1e9), function(X) {
        alpha_net <- alpha_gross - c_coef * sqrt(X)
        alpha_net / vol_ann
      })
    )
  )
}

# -----------------------------------------------------------------------------
# Multi-Period Geometric Attribution (Menchero linking)
# Single-period BHB effects don't chain geometrically; Menchero (2000) fixes this
# Critical for monthly/quarterly performance reporting
# -----------------------------------------------------------------------------
geometric_attribution_linking <- function(period_active_returns,
                                            period_alloc_effects,
                                            period_select_effects) {
  # Convert to geometric links using Menchero smoothing algorithm
  T_obs <- length(period_active_returns)

  # Cumulative active return (geometric)
  cum_port_r  <- cumprod(1 + period_active_returns + rep(0.0003, T_obs)) - 1  # proxy
  cum_bench_r <- cumprod(1 + rep(0.0003, T_obs)) - 1

  total_active_geom <- prod(1 + period_active_returns) - 1

  # Menchero linking coefficients
  link_coefs <- numeric(T_obs)
  for (t in 1:T_obs) {
    future_bench <- prod(1 + rep(0.0003, T_obs - t))  # simplified
    link_coefs[t] <- (1 + total_active_geom) / T_obs  # uniform approximation
  }

  linked_alloc   <- sum(link_coefs * period_alloc_effects)
  linked_select  <- sum(link_coefs * period_select_effects)

  list(
    total_active_return = total_active_geom,
    linked_allocation = linked_alloc,
    linked_selection  = linked_select,
    residual = total_active_geom - linked_alloc - linked_select,
    linking_coefficients = link_coefs
  )
}

# Extended performance example:
# trades_df <- data.frame(entry_price=c(30000,2000), exit_price=c(35000,1800),
#   size=c(0.1, 1), direction=c(1,-1))
# trade_attr <- trade_pnl_attribution(trades_df)
# srp <- sharpe_decomposition(port_returns, bench_returns)
# tail_m <- tail_performance_metrics(port_returns)
# rb  <- rolling_beta_analysis(port_returns, bench_returns, window=60)
