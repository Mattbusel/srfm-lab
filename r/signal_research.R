# signal_research.R
# Systematic signal research framework for the SRFM quantitative trading system.
# Dependencies: zoo, xts, stats
# All functions follow base R + zoo/xts conventions.

suppressPackageStartupMessages({
  library(zoo)
  library(xts)
})


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------

#' Compute cross-sectional Information Coefficient (IC) per period
#'
#' @param signal_matrix  Matrix (assets x periods).  Each column is the
#'                       cross-sectional signal for that period.
#' @param returns_matrix Matrix (assets x periods) of forward returns
#'                       corresponding to each signal period.
#' @param method         Correlation method: "spearman" (default) or "pearson".
#' @return Named numeric vector of IC values, one per period.
#'
#' @details
#'   For each period the function computes the correlation between the
#'   cross-sectional signal ranks and the cross-sectional forward return
#'   ranks.  Periods with fewer than 3 non-NA paired observations return NA.
#'
#' @examples
#' \dontrun{
#'   ic <- compute_ic(signal_matrix, returns_matrix)
#' }
compute_ic <- function(signal_matrix, returns_matrix,
                       method = c("spearman", "pearson")) {
  method <- match.arg(method)
  if (!identical(dim(signal_matrix), dim(returns_matrix))) {
    stop("signal_matrix and returns_matrix must have identical dimensions")
  }

  n_periods <- ncol(signal_matrix)
  ic_series <- numeric(n_periods)
  period_names <- colnames(signal_matrix)
  if (is.null(period_names)) period_names <- seq_len(n_periods)

  for (t in seq_len(n_periods)) {
    sig <- signal_matrix[, t]
    ret <- returns_matrix[, t]
    ok  <- !is.na(sig) & !is.na(ret)
    if (sum(ok) < 3) {
      ic_series[t] <- NA_real_
      next
    }
    ic_series[t] <- stats::cor(sig[ok], ret[ok], method = method)
  }

  names(ic_series) <- as.character(period_names)
  ic_series
}


# ---------------------------------------------------------------------------
# compute_icir
# ---------------------------------------------------------------------------

#' Compute Information Coefficient Information Ratio (ICIR)
#'
#' @param ic_series Numeric vector of IC values from compute_ic().
#' @param window    Optional integer rolling window.  If NULL the full-sample
#'                  ICIR is returned as a scalar.  If an integer, a zoo object
#'                  of rolling ICIRs is returned.
#' @return Scalar or zoo object: mean(IC) / sd(IC) * sqrt(periods_per_year).
#'
#' @details
#'   The annualisation factor uses 252 (daily) if window is NULL.  For rolling
#'   windows the annualisation is omitted and the raw ratio is returned so that
#'   it is comparable across windows.
#'
#' @examples
#' \dontrun{
#'   icir <- compute_icir(ic_series)
#'   rolling_icir <- compute_icir(ic_series, window = 60)
#' }
compute_icir <- function(ic_series, window = NULL) {
  safe_icir <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) < 2) return(NA_real_)
    s <- sd(x)
    if (s == 0 || is.na(s)) return(NA_real_)
    mean(x) / s
  }

  if (is.null(window)) {
    icir <- safe_icir(ic_series)
    return(icir * sqrt(252))
  }

  z <- zoo::zoo(ic_series)
  zoo::rollapply(z, width = window, FUN = safe_icir,
                 fill = NA, align = "right")
}


# ---------------------------------------------------------------------------
# ic_decay_curve
# ---------------------------------------------------------------------------

#' Compute IC at each forward horizon (IC decay curve)
#'
#' @param signal_matrix  Matrix (assets x periods) of signals.
#' @param returns_matrix Matrix (assets x periods) of 1-period forward returns.
#' @param max_lag        Maximum forward horizon in periods (default 20).
#' @return Named numeric vector of length max_lag with IC at each horizon.
#'
#' @details
#'   For horizon h the forward return at period t is the compounded return
#'   of periods t+1 ... t+h (or the sum if use_log is FALSE).  The function
#'   aligns signals with lagged returns columns.
#'
#' @examples
#' \dontrun{
#'   decay <- ic_decay_curve(signal_matrix, returns_matrix, max_lag = 20)
#'   plot(1:20, decay, type = "b", xlab = "Lag", ylab = "IC")
#' }
ic_decay_curve <- function(signal_matrix, returns_matrix, max_lag = 20) {
  if (!identical(dim(signal_matrix), dim(returns_matrix))) {
    stop("signal_matrix and returns_matrix must have identical dimensions")
  }

  n_assets  <- nrow(signal_matrix)
  n_periods <- ncol(signal_matrix)

  ic_by_lag <- numeric(max_lag)
  names(ic_by_lag) <- paste0("h", seq_len(max_lag))

  for (h in seq_len(max_lag)) {
    if (h >= n_periods) {
      ic_by_lag[h] <- NA_real_
      next
    }

    n_pairs  <- n_periods - h
    ic_vals  <- numeric(n_pairs)

    for (t in seq_len(n_pairs)) {
      sig <- signal_matrix[, t]
      # Compound return from t+1 to t+h
      fwd_ret <- apply(returns_matrix[, (t + 1):(t + h), drop = FALSE],
                       1, function(x) prod(1 + x, na.rm = TRUE) - 1)
      ok  <- !is.na(sig) & !is.na(fwd_ret)
      if (sum(ok) < 3) {
        ic_vals[t] <- NA_real_
        next
      }
      ic_vals[t] <- stats::cor(sig[ok], fwd_ret[ok], method = "spearman")
    }

    ic_by_lag[h] <- mean(ic_vals, na.rm = TRUE)
  }

  ic_by_lag
}


# ---------------------------------------------------------------------------
# fit_ic_decay
# ---------------------------------------------------------------------------

#' Fit exponential decay model to IC decay curve
#'
#' @param ic_decay Named numeric vector from ic_decay_curve().
#' @return List with components:
#'   \describe{
#'     \item{ic0}{Estimated IC at horizon 0 (intercept).}
#'     \item{lambda}{Decay rate.  IC(t) = IC0 * exp(-lambda * t).}
#'     \item{half_life}{Half-life in periods: log(2) / lambda.}
#'     \item{r_squared}{R-squared of the fit.}
#'     \item{fitted}{Numeric vector of fitted IC values.}
#'   }
#'
#' @examples
#' \dontrun{
#'   fit <- fit_ic_decay(decay)
#'   cat("Half-life:", fit$half_life, "periods\n")
#' }
fit_ic_decay <- function(ic_decay) {
  lags     <- seq_along(ic_decay)
  ok       <- !is.na(ic_decay) & ic_decay > 0
  if (sum(ok) < 3) {
    warning("Fewer than 3 positive IC values -- cannot fit exponential decay")
    return(list(ic0 = NA, lambda = NA, half_life = NA,
                r_squared = NA, fitted = rep(NA, length(ic_decay))))
  }

  log_ic   <- log(ic_decay[ok])
  lags_ok  <- lags[ok]

  fit      <- stats::lm(log_ic ~ lags_ok)
  coefs    <- stats::coef(fit)
  ic0      <- exp(coefs[1])
  lambda   <- -coefs[2]

  half_life   <- if (lambda > 0) log(2) / lambda else NA_real_
  fitted_vals <- ic0 * exp(-lambda * lags)
  ss_res      <- sum((ic_decay[ok] - ic0 * exp(-lambda * lags_ok))^2)
  ss_tot      <- sum((ic_decay[ok] - mean(ic_decay[ok]))^2)
  r2          <- if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_

  list(
    ic0       = ic0,
    lambda    = lambda,
    half_life = half_life,
    r_squared = r2,
    fitted    = fitted_vals
  )
}


# ---------------------------------------------------------------------------
# quintile_sort_returns
# ---------------------------------------------------------------------------

#' Compute quintile (or N-tile) sorted return series
#'
#' @param signal    Matrix (assets x periods) of signal values.
#' @param returns   Matrix (assets x periods) of 1-period forward returns.
#' @param n_quintiles Integer number of buckets (default 5).
#' @return List of n_quintiles numeric vectors (one per bucket), each
#'         containing the equal-weighted mean return of assets in that bucket
#'         at each period.  Bucket 1 = lowest signal, bucket N = highest.
#'
#' @examples
#' \dontrun{
#'   qs <- quintile_sort_returns(signal_matrix, returns_matrix)
#' }
quintile_sort_returns <- function(signal, returns, n_quintiles = 5) {
  if (!identical(dim(signal), dim(returns))) {
    stop("signal and returns must have identical dimensions")
  }

  n_periods <- ncol(signal)
  bucket_returns <- vector("list", n_quintiles)
  for (q in seq_len(n_quintiles)) {
    bucket_returns[[q]] <- numeric(n_periods)
    names(bucket_returns[[q]]) <- colnames(signal)
  }

  for (t in seq_len(n_periods)) {
    s   <- signal[, t]
    r   <- returns[, t]
    ok  <- !is.na(s) & !is.na(r)
    if (sum(ok) < n_quintiles) {
      for (q in seq_len(n_quintiles)) bucket_returns[[q]][t] <- NA_real_
      next
    }
    breaks <- stats::quantile(s[ok],
                              probs = seq(0, 1, length.out = n_quintiles + 1),
                              na.rm = TRUE)
    breaks[1]   <- -Inf
    breaks[n_quintiles + 1] <- Inf
    buckets <- cut(s[ok], breaks = breaks, labels = FALSE, include.lowest = TRUE)
    for (q in seq_len(n_quintiles)) {
      idx <- which(buckets == q)
      bucket_returns[[q]][t] <- if (length(idx) > 0) mean(r[ok][idx]) else NA_real_
    }
  }

  names(bucket_returns) <- paste0("Q", seq_len(n_quintiles))
  bucket_returns
}


# ---------------------------------------------------------------------------
# long_short_returns
# ---------------------------------------------------------------------------

#' Compute long/short portfolio returns (top N minus bottom N)
#'
#' @param signal   Matrix (assets x periods) of signal values.
#' @param returns  Matrix (assets x periods) of 1-period forward returns.
#' @param n_long   Integer number of assets in the long leg (default 20).
#' @param n_short  Integer number of assets in the short leg (default 20).
#' @return Named numeric vector of long-short portfolio returns per period.
#'
#' @details
#'   At each period the top n_long assets by signal are long (equal-weighted)
#'   and the bottom n_short assets are short.  The combined return is
#'   mean(long returns) - mean(short returns).
#'
#' @examples
#' \dontrun{
#'   ls_ret <- long_short_returns(signal_matrix, returns_matrix, 20, 20)
#' }
long_short_returns <- function(signal, returns, n_long = 20, n_short = 20) {
  if (!identical(dim(signal), dim(returns))) {
    stop("signal and returns must have identical dimensions")
  }

  n_assets  <- nrow(signal)
  n_periods <- ncol(signal)
  ls_ret    <- numeric(n_periods)
  names(ls_ret) <- colnames(signal)

  for (t in seq_len(n_periods)) {
    s  <- signal[, t]
    r  <- returns[, t]
    ok <- !is.na(s) & !is.na(r)
    n_ok <- sum(ok)
    if (n_ok < n_long + n_short) {
      ls_ret[t] <- NA_real_
      next
    }

    ord       <- order(s[ok], decreasing = TRUE)
    long_idx  <- ord[seq_len(n_long)]
    short_idx <- ord[(n_ok - n_short + 1):n_ok]

    r_ok      <- r[ok]
    ls_ret[t] <- mean(r_ok[long_idx]) - mean(r_ok[short_idx])
  }

  ls_ret
}


# ---------------------------------------------------------------------------
# signal_correlation_matrix
# ---------------------------------------------------------------------------

#' Compute pairwise Spearman correlation matrix of signals
#'
#' @param signals_list Named list of signal matrices (each assets x periods).
#'                     All matrices must share identical dimensions.
#' @return Symmetric correlation matrix of dimension length(signals_list).
#'
#' @details
#'   Correlations are computed by vectorising each signal matrix (stacking all
#'   columns) and computing Spearman correlation on the joint non-NA subset.
#'
#' @examples
#' \dontrun{
#'   cm <- signal_correlation_matrix(list(mom = mom_signal, val = val_signal))
#' }
signal_correlation_matrix <- function(signals_list) {
  if (length(signals_list) < 2) {
    stop("Need at least 2 signals to compute correlation matrix")
  }
  sig_names <- names(signals_list)
  if (is.null(sig_names)) sig_names <- paste0("S", seq_along(signals_list))

  n <- length(signals_list)
  cm <- matrix(1, nrow = n, ncol = n, dimnames = list(sig_names, sig_names))

  for (i in seq_len(n - 1)) {
    for (j in (i + 1):n) {
      x <- as.vector(signals_list[[i]])
      y <- as.vector(signals_list[[j]])
      ok <- !is.na(x) & !is.na(y)
      if (sum(ok) < 3) {
        cm[i, j] <- cm[j, i] <- NA_real_
      } else {
        r <- stats::cor(x[ok], y[ok], method = "spearman")
        cm[i, j] <- cm[j, i] <- r
      }
    }
  }

  cm
}


# ---------------------------------------------------------------------------
# novelty_filter
# ---------------------------------------------------------------------------

#' Test whether a new signal is novel relative to existing signals
#'
#' @param new_signal       Matrix (assets x periods) of the candidate signal.
#' @param existing_signals Named list of signal matrices (assets x periods).
#' @param threshold        Maximum absolute correlation allowed for the signal
#'                         to be considered novel (default 0.7).
#' @return List with components:
#'   \describe{
#'     \item{is_novel}{Logical: TRUE if max correlation < threshold.}
#'     \item{max_correlation}{Scalar maximum absolute pairwise correlation.}
#'     \item{most_similar}{Name of the most correlated existing signal.}
#'     \item{correlations}{Named numeric vector of correlations with each
#'           existing signal.}
#'   }
#'
#' @examples
#' \dontrun{
#'   nf <- novelty_filter(new_sig, existing_sigs)
#'   if (nf$is_novel) cat("Signal is novel\n")
#' }
novelty_filter <- function(new_signal, existing_signals, threshold = 0.7) {
  x     <- as.vector(new_signal)
  corrs <- numeric(length(existing_signals))
  names(corrs) <- names(existing_signals)
  if (is.null(names(existing_signals))) {
    names(corrs) <- paste0("S", seq_along(existing_signals))
  }

  for (i in seq_along(existing_signals)) {
    y  <- as.vector(existing_signals[[i]])
    ok <- !is.na(x) & !is.na(y)
    if (sum(ok) < 3) {
      corrs[i] <- NA_real_
      next
    }
    corrs[i] <- stats::cor(x[ok], y[ok], method = "spearman")
  }

  abs_corrs    <- abs(corrs)
  max_corr     <- max(abs_corrs, na.rm = TRUE)
  most_similar <- names(which.max(abs_corrs))

  list(
    is_novel       = max_corr < threshold,
    max_correlation = max_corr,
    most_similar   = most_similar,
    correlations   = corrs
  )
}


# ---------------------------------------------------------------------------
# batch_signal_test
# ---------------------------------------------------------------------------

#' Test a list of signal-generating functions in batch
#'
#' @param signal_fns_list Named list of functions; each must accept
#'                        (returns, universe, start, end) and return a
#'                        matrix (assets x periods).
#' @param returns         Matrix (assets x periods) of asset returns.
#' @param universe        Character vector of asset names (row subset).
#' @param start           Start date/index (column subset, inclusive).
#' @param end             End date/index (column subset, inclusive).
#' @return data.frame with columns: signal_name, mean_ic, icir, half_life,
#'         ls_sharpe.  One row per signal function.
#'
#' @examples
#' \dontrun{
#'   results <- batch_signal_test(
#'     list(mom = generate_momentum_factor, ar1 = function(r, u, s, e) {
#'       generate_ar1_factor(ncol(r[u, s:e]), length(u))
#'     }),
#'     returns, universe, 1, 252
#'   )
#' }
batch_signal_test <- function(signal_fns_list, returns, universe, start, end) {
  if (is.null(names(signal_fns_list))) {
    names(signal_fns_list) <- paste0("signal_", seq_along(signal_fns_list))
  }

  ret_sub <- returns[universe, start:end, drop = FALSE]
  results <- vector("list", length(signal_fns_list))

  for (i in seq_along(signal_fns_list)) {
    nm <- names(signal_fns_list)[i]
    fn <- signal_fns_list[[i]]

    sig <- tryCatch(
      fn(returns, universe, start, end),
      error = function(e) {
        warning(sprintf("Signal '%s' failed: %s", nm, conditionMessage(e)))
        NULL
      }
    )

    if (is.null(sig)) {
      results[[i]] <- data.frame(
        signal_name = nm, mean_ic = NA, icir = NA,
        half_life = NA, ls_sharpe = NA,
        stringsAsFactors = FALSE
      )
      next
    }

    # Align dimensions
    common_cols <- min(ncol(sig), ncol(ret_sub))
    sig_a <- sig[, seq_len(common_cols), drop = FALSE]
    ret_a <- ret_sub[, seq_len(common_cols), drop = FALSE]

    ic       <- compute_ic(sig_a, ret_a)
    mean_ic  <- mean(ic, na.rm = TRUE)
    icir_val <- compute_icir(ic)
    decay    <- ic_decay_curve(sig_a, ret_a, max_lag = min(20, common_cols - 1))
    fit      <- fit_ic_decay(decay)
    ls_ret   <- long_short_returns(sig_a, ret_a)
    ls_sr    <- if (sum(!is.na(ls_ret)) > 1) {
      mu <- mean(ls_ret, na.rm = TRUE)
      sg <- sd(ls_ret, na.rm = TRUE)
      if (sg > 0) (mu / sg) * sqrt(252) else NA_real_
    } else NA_real_

    results[[i]] <- data.frame(
      signal_name = nm,
      mean_ic     = mean_ic,
      icir        = icir_val,
      half_life   = fit$half_life,
      ls_sharpe   = ls_sr,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, results)
}


# ---------------------------------------------------------------------------
# signal_report
# ---------------------------------------------------------------------------

#' Print a formatted knitr-style signal evaluation report
#'
#' @param name     Character: signal name.
#' @param ic       Numeric vector of IC values.
#' @param icir     Scalar ICIR.
#' @param decay    Numeric vector from ic_decay_curve().
#' @param quintiles List from quintile_sort_returns().
#' @return Invisibly returns a list of all inputs and derived metrics.
#'
#' @examples
#' \dontrun{
#'   signal_report("Momentum_63d", ic, icir, decay, quintiles)
#' }
signal_report <- function(name, ic, icir, decay, quintiles) {
  mean_ic    <- mean(ic, na.rm = TRUE)
  ic_t_stat  <- if (sum(!is.na(ic)) > 1) {
    n <- sum(!is.na(ic))
    mean_ic / (sd(ic, na.rm = TRUE) / sqrt(n))
  } else NA_real_
  ic_pct_pos <- mean(ic > 0, na.rm = TRUE)
  fit        <- fit_ic_decay(decay)

  q_means <- sapply(quintiles, function(x) mean(x, na.rm = TRUE) * 252)
  spread   <- if (length(q_means) >= 2) {
    q_means[length(q_means)] - q_means[1]
  } else NA_real_

  cat("=================================================================\n")
  cat(sprintf("  SIGNAL REPORT: %s\n", name))
  cat("=================================================================\n")
  cat(sprintf("  Mean IC                : %.4f\n", mean_ic))
  cat(sprintf("  IC t-stat              : %.3f\n", ic_t_stat))
  cat(sprintf("  IC %% Positive          : %.1f%%\n", ic_pct_pos * 100))
  cat(sprintf("  ICIR (annualised)      : %.4f\n", icir))
  cat(sprintf("  Decay half-life        : %.1f periods\n",
              ifelse(is.na(fit$half_life), -1, fit$half_life)))
  cat(sprintf("  Decay fit R^2          : %.4f\n",
              ifelse(is.na(fit$r_squared), 0, fit$r_squared)))
  cat("  Quintile annualised returns (Q1=low signal):\n")
  for (i in seq_along(q_means)) {
    cat(sprintf("    Q%d: %+.2f%%\n", i, q_means[i] * 100))
  }
  cat(sprintf("  Q-spread (QN - Q1)     : %+.2f%% p.a.\n", spread * 100))
  cat("=================================================================\n")

  invisible(list(
    name      = name,
    mean_ic   = mean_ic,
    ic_t_stat = ic_t_stat,
    icir      = icir,
    decay_fit = fit,
    quintile_ann_returns = q_means,
    q_spread  = spread
  ))
}


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

#' Generate an AR(1) cross-sectional factor
#'
#' @param n_obs    Integer number of time periods.
#' @param n_assets Integer number of assets.
#' @param rho      AR(1) autocorrelation coefficient (default 0.5).
#' @param noise_sd Standard deviation of idiosyncratic noise (default 0.1).
#' @return Matrix (n_assets x n_obs) of AR(1) factor values.
#'
#' @details
#'   A single common AR(1) factor is simulated, then each asset's signal is
#'   the common factor plus independent Gaussian noise.
#'
#' @examples
#' \dontrun{
#'   sig <- generate_ar1_factor(252, 100, rho = 0.7)
#' }
generate_ar1_factor <- function(n_obs, n_assets, rho = 0.5, noise_sd = 0.1) {
  common <- numeric(n_obs)
  common[1] <- stats::rnorm(1)
  for (t in 2:n_obs) {
    common[t] <- rho * common[t - 1] + stats::rnorm(1, 0, sqrt(1 - rho^2))
  }

  noise  <- matrix(stats::rnorm(n_assets * n_obs, 0, noise_sd),
                   nrow = n_assets, ncol = n_obs)
  signal <- matrix(rep(common, each = n_assets), nrow = n_assets) + noise

  rownames(signal) <- paste0("A", seq_len(n_assets))
  colnames(signal) <- paste0("T", seq_len(n_obs))
  signal
}


#' Generate a standard momentum signal from returns
#'
#' @param returns  Matrix (assets x periods) of returns.
#' @param lookback Integer lookback window in periods (default 63).
#' @return Matrix (assets x periods) of momentum signals.  The first
#'         \code{lookback} columns will be NA.
#'
#' @details
#'   Momentum signal is defined as the cumulative compounded return over the
#'   prior \code{lookback} periods (skip-1 convention is NOT applied here;
#'   callers may shift if needed).
#'
#' @examples
#' \dontrun{
#'   mom <- generate_momentum_factor(returns, lookback = 63)
#' }
generate_momentum_factor <- function(returns, lookback = 63) {
  n_assets  <- nrow(returns)
  n_periods <- ncol(returns)

  signal <- matrix(NA_real_, nrow = n_assets, ncol = n_periods,
                   dimnames = dimnames(returns))

  if (lookback >= n_periods) {
    warning("lookback >= n_periods -- all signals are NA")
    return(signal)
  }

  for (t in (lookback + 1):n_periods) {
    window_ret <- returns[, (t - lookback):(t - 1), drop = FALSE]
    signal[, t] <- apply(window_ret, 1,
                         function(x) prod(1 + x, na.rm = TRUE) - 1)
  }

  signal
}
