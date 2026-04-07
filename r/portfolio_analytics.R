# portfolio_analytics.R
# Portfolio-level analytics for the SRFM quantitative trading system.
# Dependencies: zoo, xts
# All functions follow base R + zoo/xts conventions.

suppressPackageStartupMessages({
  library(zoo)
  library(xts)
})


# ---------------------------------------------------------------------------
# compute_portfolio_returns
# ---------------------------------------------------------------------------

#' Compute rebalanced portfolio return series
#'
#' @param weights_df  matrix or data.frame (symbols x dates); columns are
#'                    rebalancing dates, rows are asset symbols.  Weights
#'                    should sum to 1 (or near 1) for each date.
#' @param returns_df  matrix or data.frame (symbols x dates); daily returns
#'                    for each asset.  Must share the same row names as
#'                    weights_df.
#' @return Named numeric vector of daily portfolio returns aligned to the
#'         dates in returns_df.
#' @details
#'   Between two consecutive rebalancing dates the weights drift with the
#'   asset returns (buy-and-hold).  At each rebalancing date the weights are
#'   reset to the target.  The function handles the case where rebalancing
#'   dates are a subset of the returns dates.
#'
#' @examples
#' \dontrun{
#'   w <- matrix(c(0.5, 0.5), nrow = 2, ncol = 3,
#'               dimnames = list(c("A","B"), c("2020-01-01","2020-02-01","2020-03-01")))
#'   r <- matrix(rnorm(200, 0, 0.01), nrow = 2,
#'               dimnames = list(c("A","B"), as.character(seq.Date(
#'                 as.Date("2020-01-01"), by = "day", length.out = 100))))
#'   pret <- compute_portfolio_returns(w, r)
#' }
compute_portfolio_returns <- function(weights_df, returns_df) {
  w_mat  <- as.matrix(weights_df)
  r_mat  <- as.matrix(returns_df)

  assets <- rownames(w_mat)
  if (is.null(assets)) stop("weights_df must have row names (asset symbols)")

  ret_assets <- rownames(r_mat)
  if (!all(assets %in% ret_assets)) {
    stop("Some assets in weights_df are missing from returns_df")
  }

  r_mat <- r_mat[assets, , drop = FALSE]

  ret_dates <- colnames(r_mat)
  reb_dates <- colnames(w_mat)

  if (is.null(ret_dates)) stop("returns_df must have column names (dates)")
  if (is.null(reb_dates)) stop("weights_df must have column names (dates)")

  port_returns <- numeric(ncol(r_mat))
  names(port_returns) <- ret_dates

  current_weights <- w_mat[, 1]
  current_weights <- current_weights / sum(current_weights)

  reb_idx <- match(reb_dates, ret_dates)
  reb_idx <- reb_idx[!is.na(reb_idx)]

  for (t in seq_along(ret_dates)) {
    if (t %in% reb_idx) {
      reb_col <- which(reb_idx == t)
      current_weights <- w_mat[, reb_col]
      if (any(is.na(current_weights))) {
        current_weights[is.na(current_weights)] <- 0
      }
      s <- sum(current_weights)
      if (s > 0) current_weights <- current_weights / s
    }

    day_returns    <- r_mat[, t]
    port_returns[t] <- sum(current_weights * day_returns, na.rm = TRUE)

    gross_new      <- current_weights * (1 + day_returns)
    s2             <- sum(gross_new, na.rm = TRUE)
    if (s2 > 0) {
      current_weights <- gross_new / s2
    }
  }

  port_returns
}


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------

#' Compute mean daily portfolio turnover
#'
#' @param weights_df  matrix or data.frame (symbols x dates).  Each column
#'                    is the target weight vector at a rebalancing date.
#' @return Scalar: mean one-way turnover per rebalancing period (sum of
#'         absolute weight changes, divided by 2 to avoid double counting,
#'         averaged over all periods).
#'
#' @examples
#' \dontrun{
#'   to <- compute_turnover(w)
#' }
compute_turnover <- function(weights_df) {
  w_mat <- as.matrix(weights_df)
  if (ncol(w_mat) < 2) {
    warning("weights_df has fewer than 2 columns -- turnover is undefined")
    return(NA_real_)
  }

  turnovers <- numeric(ncol(w_mat) - 1)
  for (t in 2:ncol(w_mat)) {
    turnovers[t - 1] <- sum(abs(w_mat[, t] - w_mat[, t - 1]), na.rm = TRUE) / 2
  }
  mean(turnovers)
}


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

#' Compute annualised Sharpe ratio
#'
#' @param returns          Numeric vector of period returns (e.g. daily).
#' @param rf               Risk-free rate per period (default 0).
#' @param annualize        Logical; if TRUE annualise using periods_per_year.
#' @param periods_per_year Number of periods per year (default 252 for daily).
#' @return Scalar Sharpe ratio.
#'
#' @examples
#' \dontrun{
#'   sharpe_ratio(rnorm(252, 0.0005, 0.01))
#' }
sharpe_ratio <- function(returns, rf = 0, annualize = TRUE,
                         periods_per_year = 252) {
  excess <- returns - rf
  mu     <- mean(excess, na.rm = TRUE)
  sigma  <- sd(excess, na.rm = TRUE)

  if (sigma == 0 || is.na(sigma)) return(NA_real_)

  sr <- mu / sigma
  if (annualize) sr <- sr * sqrt(periods_per_year)
  sr
}


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

#' Compute annualised Sortino ratio
#'
#' @param returns   Numeric vector of period returns.
#' @param mar       Minimum acceptable return per period (default 0).
#' @param annualize Logical; if TRUE annualise (assumes daily, 252 days/year).
#' @return Scalar Sortino ratio.
#'
#' @details
#'   Downside deviation is computed using only those observations where
#'   returns fall below the MAR.
#'
#' @examples
#' \dontrun{
#'   sortino_ratio(rnorm(252, 0.0005, 0.01))
#' }
sortino_ratio <- function(returns, mar = 0, annualize = TRUE) {
  excess    <- returns - mar
  mu        <- mean(excess, na.rm = TRUE)
  downside  <- pmin(excess, 0)
  dd_vol    <- sqrt(mean(downside^2, na.rm = TRUE))

  if (dd_vol == 0 || is.na(dd_vol)) return(NA_real_)

  sr <- mu / dd_vol
  if (annualize) sr <- sr * sqrt(252)
  sr
}


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

#' Compute Calmar ratio (annualised return / max drawdown)
#'
#' @param returns Numeric vector of daily returns.
#' @return Scalar Calmar ratio; NA if max drawdown is zero.
#'
#' @examples
#' \dontrun{
#'   calmar_ratio(rnorm(252, 0.0005, 0.01))
#' }
calmar_ratio <- function(returns) {
  ann_ret <- prod(1 + returns, na.rm = TRUE)^(252 / length(returns)) - 1
  nav     <- cumprod(1 + returns)
  mdd     <- max_drawdown(nav)

  if (mdd == 0 || is.na(mdd)) return(NA_real_)
  ann_ret / abs(mdd)
}

# Internal helper -- scalar max drawdown from NAV vector
max_drawdown <- function(nav) {
  peak <- cummax(nav)
  dd   <- (nav - peak) / peak
  min(dd, na.rm = TRUE)
}


# ---------------------------------------------------------------------------
# max_drawdown_series
# ---------------------------------------------------------------------------

#' Compute drawdown at each point in time (for plotting)
#'
#' @param nav Numeric vector representing a NAV or cumulative return series.
#' @return Numeric vector of the same length as nav, containing the drawdown
#'         (non-positive values) at each observation.
#'
#' @examples
#' \dontrun{
#'   nav <- cumprod(1 + rnorm(252, 0.0005, 0.01))
#'   dd  <- max_drawdown_series(nav)
#'   plot(dd, type = "l")
#' }
max_drawdown_series <- function(nav) {
  peak <- cummax(nav)
  dd   <- (nav - peak) / peak
  dd
}


# ---------------------------------------------------------------------------
# underwater_curve
# ---------------------------------------------------------------------------

#' Compute underwater curve (fraction of time spent below prior peak)
#'
#' @param nav Numeric vector representing a NAV series.
#' @return List with components:
#'   \describe{
#'     \item{underwater}{Logical vector: TRUE when NAV is below prior peak.}
#'     \item{pct_underwater}{Scalar fraction of observations underwater.}
#'     \item{longest_drawdown}{Integer: length of longest underwater period.}
#'   }
#'
#' @examples
#' \dontrun{
#'   uw <- underwater_curve(nav)
#'   cat("Time underwater:", uw$pct_underwater, "\n")
#' }
underwater_curve <- function(nav) {
  peak       <- cummax(nav)
  underwater <- nav < peak

  rle_res <- rle(underwater)
  uw_lengths <- rle_res$lengths[rle_res$values]
  longest    <- if (length(uw_lengths) > 0) max(uw_lengths) else 0L

  list(
    underwater       = underwater,
    pct_underwater   = mean(underwater, na.rm = TRUE),
    longest_drawdown = longest
  )
}


# ---------------------------------------------------------------------------
# tail_ratio
# ---------------------------------------------------------------------------

#' Compute tail ratio (95th percentile return / |5th percentile return|)
#'
#' @param returns  Numeric vector of returns.
#' @param quantile Lower quantile to use (default 0.05).  The upper quantile
#'                 is 1 - quantile.
#' @return Scalar tail ratio.  Values > 1 indicate fat right tail.
#'
#' @examples
#' \dontrun{
#'   tail_ratio(rnorm(1000))
#' }
tail_ratio <- function(returns, quantile = 0.05) {
  upper <- stats::quantile(returns, 1 - quantile, na.rm = TRUE)
  lower <- stats::quantile(returns, quantile,     na.rm = TRUE)

  if (lower == 0 || is.na(lower)) return(NA_real_)
  abs(upper / lower)
}


# ---------------------------------------------------------------------------
# omega_ratio
# ---------------------------------------------------------------------------

#' Compute Omega ratio
#'
#' @param returns   Numeric vector of returns.
#' @param threshold Return threshold (default 0).
#' @return Scalar Omega ratio: probability-weighted gain / probability-weighted
#'         loss above and below threshold.  Values > 1 indicate net positive
#'         outcomes.
#'
#' @examples
#' \dontrun{
#'   omega_ratio(rnorm(252, 0.0005, 0.01))
#' }
omega_ratio <- function(returns, threshold = 0) {
  gains  <- sum(pmax(returns - threshold, 0), na.rm = TRUE)
  losses <- sum(pmax(threshold - returns, 0), na.rm = TRUE)

  if (losses == 0 || is.na(losses)) return(NA_real_)
  gains / losses
}


# ---------------------------------------------------------------------------
# rolling_sharpe
# ---------------------------------------------------------------------------

#' Compute rolling Sharpe ratio using a zoo rolling window
#'
#' @param returns          Numeric vector or zoo object of daily returns.
#' @param window           Integer rolling window size (default 252).
#' @param min_obs          Minimum observations to compute; otherwise NA
#'                         (default 63).
#' @return zoo object with rolling Sharpe ratios.
#'
#' @examples
#' \dontrun{
#'   rs <- rolling_sharpe(rnorm(500, 0.0005, 0.01))
#'   plot(rs)
#' }
rolling_sharpe <- function(returns, window = 252, min_obs = 63) {
  if (!inherits(returns, "zoo")) {
    returns <- zoo::zoo(returns)
  }

  roller <- function(x) {
    if (sum(!is.na(x)) < min_obs) return(NA_real_)
    mu    <- mean(x, na.rm = TRUE)
    sigma <- sd(x, na.rm = TRUE)
    if (sigma == 0 || is.na(sigma)) return(NA_real_)
    (mu / sigma) * sqrt(252)
  }

  zoo::rollapply(returns, width = window, FUN = roller,
                 fill = NA, align = "right")
}


# ---------------------------------------------------------------------------
# monthly_returns_matrix
# ---------------------------------------------------------------------------

#' Convert daily returns to a months x years matrix
#'
#' @param daily_returns Named numeric vector with names parseable as dates
#'                      (e.g., "2020-01-15"), or a zoo/xts object with date
#'                      index.
#' @return Numeric matrix with 12 rows (months Jan-Dec) and one column per
#'         calendar year.  Values are compounded monthly returns.
#'
#' @examples
#' \dontrun{
#'   dates   <- seq.Date(as.Date("2018-01-01"), as.Date("2020-12-31"), by = "day")
#'   r       <- setNames(rnorm(length(dates), 0.0005, 0.01), as.character(dates))
#'   mat     <- monthly_returns_matrix(r)
#' }
monthly_returns_matrix <- function(daily_returns) {
  if (inherits(daily_returns, c("zoo", "xts"))) {
    dates   <- zoo::index(daily_returns)
    returns <- as.numeric(daily_returns)
  } else {
    dates   <- as.Date(names(daily_returns))
    returns <- as.numeric(daily_returns)
    if (any(is.na(dates))) {
      stop("names(daily_returns) must be parseable as dates")
    }
  }

  yr  <- as.integer(format(dates, "%Y"))
  mo  <- as.integer(format(dates, "%m"))
  all_years  <- sort(unique(yr))
  month_abbr <- month.abb

  result <- matrix(NA_real_, nrow = 12, ncol = length(all_years),
                   dimnames = list(month_abbr, as.character(all_years)))

  for (y in all_years) {
    for (m in 1:12) {
      idx <- which(yr == y & mo == m)
      if (length(idx) == 0) next
      result[m, as.character(y)] <-
        prod(1 + returns[idx], na.rm = TRUE) - 1
    }
  }

  result
}


# ---------------------------------------------------------------------------
# performance_summary_table
# ---------------------------------------------------------------------------

#' Print a formatted performance summary table
#'
#' @param returns Numeric vector of daily returns.
#' @return Invisibly returns a named list of all computed metrics.  As a side
#'         effect, prints a formatted table to the console.
#'
#' @examples
#' \dontrun{
#'   performance_summary_table(rnorm(252, 0.0005, 0.01))
#' }
performance_summary_table <- function(returns) {
  returns <- returns[!is.na(returns)]
  n       <- length(returns)

  if (n < 2) {
    warning("Fewer than 2 observations -- most metrics are undefined")
    return(invisible(NULL))
  }

  nav         <- cumprod(1 + returns)
  ann_ret     <- nav[n]^(252 / n) - 1
  ann_vol     <- sd(returns, na.rm = TRUE) * sqrt(252)
  sr          <- sharpe_ratio(returns)
  srt         <- sortino_ratio(returns)
  cr          <- calmar_ratio(returns)
  mdd         <- max_drawdown(nav)
  tr          <- tail_ratio(returns)
  om          <- omega_ratio(returns)
  uw          <- underwater_curve(nav)
  skew_val    <- if (n >= 3) {
    m3 <- mean((returns - mean(returns))^3)
    m2 <- mean((returns - mean(returns))^2)
    m3 / m2^1.5
  } else NA_real_
  kurt_val    <- if (n >= 4) {
    m4 <- mean((returns - mean(returns))^4)
    m2 <- mean((returns - mean(returns))^2)
    m4 / m2^2 - 3
  } else NA_real_

  metrics <- list(
    n_obs              = n,
    ann_return         = ann_ret,
    ann_volatility     = ann_vol,
    sharpe_ratio       = sr,
    sortino_ratio      = srt,
    calmar_ratio       = cr,
    max_drawdown       = mdd,
    tail_ratio         = tr,
    omega_ratio        = om,
    pct_underwater     = uw$pct_underwater,
    longest_drawdown   = uw$longest_drawdown,
    skewness           = skew_val,
    excess_kurtosis    = kurt_val,
    best_day           = max(returns),
    worst_day          = min(returns),
    positive_days_pct  = mean(returns > 0)
  )

  fmt <- function(x, digits = 4) {
    if (is.na(x)) return("NA")
    formatC(x, digits = digits, format = "f")
  }
  pct <- function(x, digits = 2) {
    if (is.na(x)) return("NA")
    paste0(formatC(x * 100, digits = digits, format = "f"), "%")
  }

  cat("=================================================================\n")
  cat("  PERFORMANCE SUMMARY\n")
  cat("=================================================================\n")
  cat(sprintf("  Observations            : %d\n",          metrics$n_obs))
  cat(sprintf("  Annualised Return       : %s\n",          pct(metrics$ann_return)))
  cat(sprintf("  Annualised Volatility   : %s\n",          pct(metrics$ann_volatility)))
  cat(sprintf("  Sharpe Ratio            : %s\n",          fmt(metrics$sharpe_ratio)))
  cat(sprintf("  Sortino Ratio           : %s\n",          fmt(metrics$sortino_ratio)))
  cat(sprintf("  Calmar Ratio            : %s\n",          fmt(metrics$calmar_ratio)))
  cat(sprintf("  Max Drawdown            : %s\n",          pct(metrics$max_drawdown)))
  cat(sprintf("  Tail Ratio (95/5)       : %s\n",          fmt(metrics$tail_ratio)))
  cat(sprintf("  Omega Ratio             : %s\n",          fmt(metrics$omega_ratio)))
  cat(sprintf("  Time Underwater         : %s\n",          pct(metrics$pct_underwater)))
  cat(sprintf("  Longest Drawdown (days) : %d\n",          metrics$longest_drawdown))
  cat(sprintf("  Skewness                : %s\n",          fmt(metrics$skewness)))
  cat(sprintf("  Excess Kurtosis         : %s\n",          fmt(metrics$excess_kurtosis)))
  cat(sprintf("  Best Day                : %s\n",          pct(metrics$best_day)))
  cat(sprintf("  Worst Day               : %s\n",          pct(metrics$worst_day)))
  cat(sprintf("  Positive Days           : %s\n",          pct(metrics$positive_days_pct)))
  cat("=================================================================\n")

  invisible(metrics)
}
