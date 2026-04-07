# factor_analysis.R
# Cross-sectional factor analysis: Fama-MacBeth regression, IC/ICIR,
# quintile portfolio analysis, alpha t-statistics with Newey-West SEs,
# ggplot2 factor return charts.
#
# Dependencies: tidyverse, ggplot2, scales, sandwich (or manual Newey-West)
# Author: srfm-lab

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(scales)
})

# ===========================================================================
# 1. Newey-West HAC standard errors
# ===========================================================================

#' Newey-West heteroskedasticity and autocorrelation consistent covariance matrix
#' @param X    design matrix (T x p)
#' @param e    residual vector (T x 1)
#' @param lags number of lags in the Bartlett kernel (default: floor(4*(T/100)^(2/9)))
#' @return p x p HAC covariance matrix
newey_west_vcov <- function(X, e, lags = NULL) {
  T <- nrow(X)
  p <- ncol(X)
  if (is.null(lags)) lags <- floor(4 * (T / 100)^(2 / 9))

  # Meat: sum of weighted outer products of score vector s_t = X_t * e_t
  S <- matrix(0, p, p)
  for (t in seq_len(T)) {
    st <- X[t, , drop = FALSE] * e[t]
    S  <- S + t(st) %*% st
  }

  for (l in seq_len(lags)) {
    w  <- 1 - l / (lags + 1)  # Bartlett weight
    Sl <- matrix(0, p, p)
    for (t in (l + 1):T) {
      s_t  <- X[t,     , drop = FALSE] * e[t]
      s_tl <- X[t - l, , drop = FALSE] * e[t - l]
      Sl   <- Sl + t(s_t) %*% s_tl
    }
    S <- S + w * (Sl + t(Sl))
  }

  bread <- solve(t(X) %*% X)
  bread %*% S %*% bread
}

#' OLS with Newey-West standard errors
#' @param y  response vector
#' @param X  design matrix (no need to add intercept -- include it in X)
#' @param lags  Newey-West lags
#' @return list with coefficients, se_nw, t_nw, p_nw, r_squared
ols_nw <- function(y, X, lags = NULL) {
  if (!is.matrix(X)) X <- as.matrix(X)
  beta    <- solve(t(X) %*% X, t(X) %*% y)
  yhat    <- X %*% beta
  e       <- y - yhat
  n       <- length(y)
  p       <- ncol(X)
  r_sq    <- 1 - sum(e^2) / sum((y - mean(y))^2)
  vcov_nw <- newey_west_vcov(X, as.vector(e), lags)
  se_nw   <- sqrt(diag(vcov_nw))
  t_nw    <- as.vector(beta) / se_nw
  p_nw    <- 2 * pt(-abs(t_nw), df = n - p)

  list(
    coefficients = as.vector(beta),
    se_nw        = se_nw,
    t_nw         = t_nw,
    p_nw         = p_nw,
    r_squared    = r_sq,
    residuals    = as.vector(e),
    vcov         = vcov_nw
  )
}

# ===========================================================================
# 2. Fama-MacBeth cross-sectional regression
# ===========================================================================

#' Run Fama-MacBeth regression
#' Procedure:
#'   1. For each period t, run cross-sectional OLS: r_{i,t} = X_{i,t} * gamma_t + eps
#'   2. Compute time-series mean of gamma_t as the factor risk premium
#'   3. Compute Newey-West t-statistic on the time series of gamma_t
#'
#' @param panel  data.frame with columns: date, id, return, and factor columns
#' @param factor_cols  character vector of factor column names
#' @param add_intercept  whether to add intercept in cross-sectional regressions
#' @param nw_lags  Newey-West lags for second-stage (default: auto)
#' @return list with:
#'   gamma_ts:   tibble of date, factor, gamma_t (all cross-sectional coefs)
#'   summary:    tibble of factor, mean_gamma, se_nw, t_nw, p_nw, r_sq_mean
fama_macbeth <- function(panel, factor_cols, add_intercept = TRUE, nw_lags = NULL) {
  stopifnot(all(c("date", "id", "return") %in% names(panel)))
  stopifnot(all(factor_cols %in% names(panel)))

  dates <- sort(unique(panel$date))

  # First pass: cross-sectional regressions
  cs_coefs <- map_dfr(dates, function(d) {
    sub <- panel %>% filter(date == d) %>%
      select(all_of(c("return", factor_cols))) %>%
      drop_na()

    if (nrow(sub) < length(factor_cols) + 2) return(NULL)

    X <- as.matrix(sub[, factor_cols, drop = FALSE])
    if (add_intercept) X <- cbind(1, X)
    y <- sub$return

    beta <- tryCatch(
      as.vector(solve(t(X) %*% X, t(X) %*% y)),
      error = function(e) rep(NA_real_, ncol(X))
    )

    col_names <- if (add_intercept) c("intercept", factor_cols) else factor_cols
    tibble(
      date     = d,
      factor   = col_names,
      gamma    = beta,
      r_sq     = if (!any(is.na(beta))) {
        e <- y - X %*% beta
        1 - sum(e^2) / sum((y - mean(y))^2)
      } else NA_real_
    )
  })

  # Second pass: time-series mean and Newey-West t-stats
  all_factors <- unique(cs_coefs$factor)

  summary_df <- map_dfr(all_factors, function(f) {
    g_ts <- cs_coefs %>% filter(factor == f) %>%
      arrange(date) %>% pull(gamma)
    g_ts <- g_ts[is.finite(g_ts)]

    if (length(g_ts) < 5) {
      return(tibble(factor = f, mean_gamma = NA, se_nw = NA,
                    t_nw = NA, p_nw = NA, r_sq_mean = NA))
    }

    T <- length(g_ts)
    # NW on time series: regress on constant
    X_const <- matrix(1, T, 1)
    nw      <- ols_nw(g_ts, X_const, lags = nw_lags)

    tibble(
      factor     = f,
      mean_gamma = nw$coefficients,
      se_nw      = nw$se_nw,
      t_nw       = nw$t_nw,
      p_nw       = nw$p_nw
    )
  })

  r_sq_mean <- cs_coefs %>%
    group_by(factor) %>%
    summarise(r_sq_mean = mean(r_sq, na.rm = TRUE), .groups = "drop")

  summary_df <- left_join(summary_df, r_sq_mean, by = "factor")

  list(gamma_ts = cs_coefs, summary = summary_df)
}

# ===========================================================================
# 3. Information Coefficient (IC) and ICIR
# ===========================================================================

#' Compute rank IC between factor scores and forward returns
#' @param scores   data.frame with columns: date, id, score (factor values)
#' @param returns  data.frame with columns: date, id, return (forward period returns)
#' @param method   "spearman" (rank IC) or "pearson" (linear IC)
#' @return tibble with date, ic, and summary statistics
compute_ic_series <- function(scores, returns, method = "spearman") {
  panel <- inner_join(scores, returns, by = c("date", "id"))

  ic_ts <- panel %>%
    group_by(date) %>%
    summarise(
      ic    = if (n() >= 5) cor(score, return, method = method, use = "complete.obs")
              else NA_real_,
      n_obs = n(),
      .groups = "drop"
    )

  ic_ts
}

#' Compute IC summary statistics and ICIR
#' @param ic_ts  tibble from compute_ic_series()
#' @return list with mean_ic, icir, ic_positive_pct, ic_series
ic_summary <- function(ic_ts) {
  ic_vals <- ic_ts$ic[is.finite(ic_ts$ic)]

  mean_ic  <- mean(ic_vals)
  sd_ic    <- sd(ic_vals)
  icir     <- if (sd_ic > 0) mean_ic / sd_ic * sqrt(252 / mean(diff(as.numeric(ic_ts$date)), na.rm = TRUE))
              else NA_real_

  list(
    mean_ic         = mean_ic,
    sd_ic           = sd_ic,
    icir            = icir,
    ic_positive_pct = mean(ic_vals > 0),
    t_stat          = mean_ic / (sd_ic / sqrt(length(ic_vals))),
    ic_series       = ic_ts
  )
}

# ===========================================================================
# 4. Quintile portfolio analysis
# ===========================================================================

#' Sort cross-section into quintiles and compute portfolio returns
#' @param panel  data.frame with columns: date, id, score, return, (optional) weight
#' @param n_quantiles  number of portfolios (default 5 = quintiles)
#' @param weight_col   column for value-weighting; NULL = equal weight
#' @return list with:
#'   portfolio_returns: tibble of date, quantile, portfolio_return
#'   spread: Q5 - Q1 long-short return
quintile_portfolios <- function(panel, n_quantiles = 5L, weight_col = NULL) {
  stopifnot(all(c("date", "id", "score", "return") %in% names(panel)))

  port_rets <- panel %>%
    group_by(date) %>%
    mutate(
      quintile = ntile(score, n_quantiles)
    ) %>%
    ungroup() %>%
    group_by(date, quintile) %>%
    summarise(
      portfolio_return = if (!is.null(weight_col) && weight_col %in% names(panel)) {
        w <- get(weight_col)
        w <- pmax(w, 0)
        if (sum(w) > 0) weighted.mean(return, w) else mean(return)
      } else mean(return),
      n_stocks        = n(),
      .groups         = "drop"
    )

  # Long-short spread
  spread <- port_rets %>%
    filter(quintile %in% c(1, n_quantiles)) %>%
    pivot_wider(names_from = quintile, values_from = portfolio_return,
                names_prefix = "q") %>%
    mutate(spread = .data[[paste0("q", n_quantiles)]] - q1) %>%
    select(date, spread)

  list(portfolio_returns = port_rets, spread = spread)
}

#' Cumulative return from a vector of period returns
cum_return <- function(r) cumprod(1 + r) - 1

#' Annualize a vector of period returns given the annualization factor
annualize <- function(r, ann_factor = 252) {
  (prod(1 + r))^(ann_factor / length(r)) - 1
}

#' Sharpe ratio
sharpe <- function(r, rf = 0, ann_factor = 252) {
  r_excess <- r - rf
  if (sd(r_excess) == 0) return(NA)
  mean(r_excess) / sd(r_excess) * sqrt(ann_factor)
}

#' Summarize quintile portfolio performance
#' @param portfolio_returns  tibble from quintile_portfolios()$portfolio_returns
#' @param n_quantiles        number of quantiles
quintile_performance <- function(portfolio_returns, n_quantiles = 5L) {
  portfolio_returns %>%
    group_by(quintile) %>%
    summarise(
      ann_return = annualize(portfolio_return),
      ann_vol    = sd(portfolio_return) * sqrt(252),
      sharpe     = sharpe(portfolio_return),
      max_dd     = max_drawdown(portfolio_return),
      n_periods  = n(),
      .groups    = "drop"
    )
}

#' Maximum drawdown from a vector of period returns
max_drawdown <- function(r) {
  cum  <- cumprod(1 + r)
  peak <- cummax(cum)
  dd   <- (cum - peak) / peak
  min(dd)
}

# ===========================================================================
# 5. Alpha t-statistics with Newey-West standard errors
# ===========================================================================

#' Run time-series regression of portfolio return on factor returns with NW SEs
#' Model: r_p = alpha + beta_1 * F1 + ... + beta_k * Fk + eps
#' @param r_portfolio  numeric vector of portfolio returns
#' @param factor_returns  matrix T x k of factor returns (without intercept column)
#' @param nw_lags  Newey-West lags
#' @return list with alpha, t_alpha, p_alpha, betas, r_squared
alpha_regression <- function(r_portfolio, factor_returns, nw_lags = NULL) {
  if (!is.matrix(factor_returns)) factor_returns <- as.matrix(factor_returns)
  stopifnot(length(r_portfolio) == nrow(factor_returns))

  X    <- cbind(1, factor_returns)
  fit  <- ols_nw(r_portfolio, X, lags = nw_lags)

  list(
    alpha     = fit$coefficients[1],
    t_alpha   = fit$t_nw[1],
    p_alpha   = fit$p_nw[1],
    se_alpha  = fit$se_nw[1],
    betas     = fit$coefficients[-1],
    t_betas   = fit$t_nw[-1],
    r_squared = fit$r_squared
  )
}

# ===========================================================================
# 6. ggplot2 factor return charts
# ===========================================================================

#' Plot rolling IC over time
plot_ic_series <- function(ic_ts, window = 63L) {
  ic_ts <- ic_ts %>%
    arrange(date) %>%
    mutate(
      rolling_ic = slider_mean(ic, window)
    )

  ggplot(ic_ts, aes(x = date)) +
    geom_col(aes(y = ic), fill = "steelblue", alpha = 0.4, width = 1) +
    geom_line(aes(y = rolling_ic), color = "#E91E63", linewidth = 0.8) +
    geom_hline(yintercept = 0, color = "grey30", linewidth = 0.4) +
    scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
    labs(
      title    = "Factor IC Over Time",
      subtitle = sprintf("Pink line: %d-period rolling average IC", window),
      x        = "Date",
      y        = "IC (Rank Correlation)"
    ) +
    theme_minimal(base_size = 12)
}

#' Simple rolling mean without requiring slider package
slider_mean <- function(x, window) {
  n   <- length(x)
  out <- rep(NA_real_, n)
  for (i in window:n) {
    out[i] <- mean(x[(i - window + 1):i], na.rm = TRUE)
  }
  out
}

#' Plot cumulative returns for each quintile portfolio
plot_quintile_cumrets <- function(portfolio_returns, n_quantiles = 5L) {
  port_cum <- portfolio_returns %>%
    arrange(quintile, date) %>%
    group_by(quintile) %>%
    mutate(cum_ret = cum_return(portfolio_return)) %>%
    ungroup()

  pal <- colorRampPalette(c("#F44336", "#FFC107", "#4CAF50"))(n_quantiles)

  ggplot(port_cum, aes(x = date, y = cum_ret,
                        color = factor(quintile), group = factor(quintile))) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = pal,
                       labels = c(paste("Q1 (low)"),
                                  if (n_quantiles > 2) paste0("Q", 2:(n_quantiles-1)) else NULL,
                                  "Q5 (high)")[seq_len(n_quantiles)]) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
    labs(
      title  = "Quintile Portfolio Cumulative Returns",
      x      = "Date",
      y      = "Cumulative Return",
      color  = "Quintile"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "right")
}

#' Plot long-short spread cumulative return
plot_ls_spread <- function(spread_df) {
  spread_df %>%
    arrange(date) %>%
    mutate(cum_spread = cum_return(spread)) %>%
    ggplot(aes(x = date, y = cum_spread)) +
    geom_line(color = "#2196F3", linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
    labs(
      title = "Long-Short Factor Portfolio: Cumulative Return (Q5 - Q1)",
      x     = "Date",
      y     = "Cumulative Return"
    ) +
    theme_minimal(base_size = 12)
}

#' Plot Fama-MacBeth factor risk premia with confidence intervals
plot_fama_macbeth_premia <- function(fm_summary, conf = 0.95) {
  z <- qnorm((1 + conf) / 2)

  fm_summary %>%
    filter(factor != "intercept") %>%
    mutate(
      lo = mean_gamma - z * se_nw,
      hi = mean_gamma + z * se_nw,
      significant = abs(t_nw) > 1.96
    ) %>%
    ggplot(aes(x = reorder(factor, mean_gamma), y = mean_gamma,
               color = significant)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2, linewidth = 0.8) +
    geom_point(size = 3) +
    coord_flip() +
    scale_color_manual(values = c(`TRUE` = "#4CAF50", `FALSE` = "#9E9E9E"),
                       labels = c("Not significant", "Significant at 5%")) +
    scale_y_continuous(labels = percent_format(accuracy = 0.01)) +
    labs(
      title    = "Fama-MacBeth Factor Risk Premia",
      subtitle = sprintf("%d%% Newey-West confidence intervals", round(conf * 100)),
      x        = "Factor",
      y        = "Mean Risk Premium (per period)",
      color    = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
}

#' Full factor analysis report: IC, quintile portfolios, Fama-MacBeth
#' @param panel  data.frame with date, id, score, return
#' @param factor_name  label for chart titles
factor_analysis_report <- function(panel, factor_name = "Factor") {
  # IC
  ic_ts   <- compute_ic_series(
    panel %>% select(date, id, score),
    panel %>% select(date, id, return)
  )
  ic_stat <- ic_summary(ic_ts)

  # Quintile portfolios
  quint    <- quintile_portfolios(panel)
  quint_perf <- quintile_performance(quint$portfolio_returns)

  # Fama-MacBeth -- single factor version
  fm <- fama_macbeth(panel, factor_cols = "score")

  message(sprintf("[%s] Mean IC: %.4f  ICIR: %.2f  FM t-stat: %.2f",
                  factor_name, ic_stat$mean_ic, ic_stat$icir,
                  fm$summary %>% filter(factor == "score") %>% pull(t_nw)))

  plots <- list(
    ic_series    = plot_ic_series(ic_ts),
    cum_rets     = plot_quintile_cumrets(quint$portfolio_returns),
    ls_spread    = plot_ls_spread(quint$spread),
    fm_premia    = plot_fama_macbeth_premia(fm$summary)
  )

  list(
    ic          = ic_stat,
    quintiles   = quint_perf,
    fama_macbeth = fm$summary,
    plots       = plots
  )
}
