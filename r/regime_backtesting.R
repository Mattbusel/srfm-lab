# regime_backtesting.R
# Conditional performance analysis by market regime for the SRFM system.
# Dependencies: zoo, xts, stats
# All functions follow base R + zoo/xts conventions.

suppressPackageStartupMessages({
  library(zoo)
  library(xts)
})


# ---------------------------------------------------------------------------
# Constants for regime labels
# ---------------------------------------------------------------------------

REGIME_TRENDING  <- "TRENDING"
REGIME_RANGING   <- "RANGING"
REGIME_HIGH_VOL  <- "HIGH_VOL"
REGIME_CRISIS    <- "CRISIS"
REGIME_LEVELS    <- c(REGIME_TRENDING, REGIME_RANGING, REGIME_HIGH_VOL, REGIME_CRISIS)


# ---------------------------------------------------------------------------
# classify_market_regime
# ---------------------------------------------------------------------------

#' Classify market regime at each time step
#'
#' @param returns     Numeric vector of daily market returns (e.g. index).
#' @param hurst_series Numeric vector of rolling Hurst exponents (same length
#'                    as returns).  H > 0.6 indicates trending; H < 0.4
#'                    indicates mean-reversion.  May contain NAs.
#' @param vol_series  Numeric vector of rolling annualised volatility (same
#'                    length as returns).
#' @return Factor vector of length equal to returns with levels:
#'         TRENDING, RANGING, HIGH_VOL, CRISIS.
#'
#' @details
#'   Classification priority (highest to lowest):
#'   1. CRISIS: vol > 2.5 * median(vol) OR single-day return < -4 * daily_sd
#'   2. HIGH_VOL: vol > 1.5 * median(vol)
#'   3. TRENDING: Hurst > 0.6 (when available)
#'   4. RANGING: Hurst < 0.4 (when available) or default
#'   When Hurst is NA the method falls back to autocorrelation of recent
#'   returns: positive autocorrelation -> TRENDING, negative -> RANGING.
#'
#' @examples
#' \dontrun{
#'   regimes <- classify_market_regime(returns, hurst, vol)
#' }
classify_market_regime <- function(returns, hurst_series, vol_series) {
  n <- length(returns)
  if (length(hurst_series) != n || length(vol_series) != n) {
    stop("returns, hurst_series, and vol_series must all have the same length")
  }

  regimes  <- character(n)
  vol_ok   <- vol_series[!is.na(vol_series)]
  med_vol  <- if (length(vol_ok) > 0) stats::median(vol_ok) else 1.0
  daily_sd <- sd(returns, na.rm = TRUE)

  for (t in seq_len(n)) {
    vol <- vol_series[t]
    h   <- hurst_series[t]
    r   <- returns[t]

    if (!is.na(vol) && !is.na(r) &&
        (vol > 2.5 * med_vol || (!is.na(daily_sd) && r < -4 * daily_sd))) {
      regimes[t] <- REGIME_CRISIS
    } else if (!is.na(vol) && vol > 1.5 * med_vol) {
      regimes[t] <- REGIME_HIGH_VOL
    } else if (!is.na(h) && h > 0.6) {
      regimes[t] <- REGIME_TRENDING
    } else if (!is.na(h) && h < 0.4) {
      regimes[t] <- REGIME_RANGING
    } else {
      # Fallback: autocorrelation of last 20 returns
      if (t >= 21) {
        recent <- returns[(t - 20):(t - 1)]
        ac1    <- tryCatch(
          stats::cor(recent[-length(recent)], recent[-1]),
          error = function(e) NA_real_
        )
        if (!is.na(ac1) && ac1 > 0.1) {
          regimes[t] <- REGIME_TRENDING
        } else {
          regimes[t] <- REGIME_RANGING
        }
      } else {
        regimes[t] <- REGIME_RANGING
      }
    }
  }

  factor(regimes, levels = REGIME_LEVELS)
}


# ---------------------------------------------------------------------------
# conditional_sharpe
# ---------------------------------------------------------------------------

#' Compute Sharpe ratio conditional on each market regime
#'
#' @param returns Numeric vector of strategy daily returns.
#' @param regimes Factor vector (from classify_market_regime()) of the same
#'                length as returns.
#' @return Named numeric vector: annualised Sharpe ratio for each regime.
#'         Regimes with fewer than 5 observations return NA.
#'
#' @examples
#' \dontrun{
#'   cs <- conditional_sharpe(strategy_returns, regimes)
#' }
conditional_sharpe <- function(returns, regimes) {
  if (length(returns) != length(regimes)) {
    stop("returns and regimes must have the same length")
  }

  lvls <- levels(regimes)
  if (is.null(lvls)) lvls <- unique(as.character(regimes))

  out <- vapply(lvls, function(lv) {
    idx <- which(as.character(regimes) == lv)
    if (length(idx) < 5) return(NA_real_)
    r   <- returns[idx]
    mu  <- mean(r, na.rm = TRUE)
    sg  <- sd(r, na.rm = TRUE)
    if (is.na(sg) || sg == 0) return(NA_real_)
    (mu / sg) * sqrt(252)
  }, numeric(1))

  out
}


# ---------------------------------------------------------------------------
# regime_transition_matrix
# ---------------------------------------------------------------------------

#' Compute the Markov transition probability matrix between regimes
#'
#' @param regimes Factor vector of regime classifications.
#' @return Square numeric matrix (regimes x regimes) of transition
#'         probabilities.  Row i, column j contains the probability of
#'         transitioning from regime i to regime j.
#'
#' @examples
#' \dontrun{
#'   tm <- regime_transition_matrix(regimes)
#' }
regime_transition_matrix <- function(regimes) {
  lvls <- levels(regimes)
  if (is.null(lvls)) lvls <- sort(unique(as.character(regimes)))
  k    <- length(lvls)

  counts <- matrix(0L, nrow = k, ncol = k,
                   dimnames = list(lvls, lvls))

  r_char <- as.character(regimes)
  for (t in seq_len(length(regimes) - 1)) {
    from <- r_char[t]
    to   <- r_char[t + 1]
    if (!is.na(from) && !is.na(to)) {
      counts[from, to] <- counts[from, to] + 1L
    }
  }

  row_sums <- rowSums(counts)
  probs    <- counts
  for (i in seq_len(k)) {
    if (row_sums[i] > 0) {
      probs[i, ] <- counts[i, ] / row_sums[i]
    }
  }

  probs
}


# ---------------------------------------------------------------------------
# conditional_drawdown
# ---------------------------------------------------------------------------

#' Compute maximum drawdown within each regime
#'
#' @param nav     Numeric vector: NAV or cumulative return series.
#' @param regimes Factor vector of regime classifications (same length as nav).
#' @return Named numeric vector: maximum drawdown (negative) for each regime.
#'
#' @examples
#' \dontrun{
#'   cd <- conditional_drawdown(nav, regimes)
#' }
conditional_drawdown <- function(nav, regimes) {
  if (length(nav) != length(regimes)) {
    stop("nav and regimes must have the same length")
  }

  lvls <- levels(regimes)
  if (is.null(lvls)) lvls <- sort(unique(as.character(regimes)))

  vapply(lvls, function(lv) {
    idx <- which(as.character(regimes) == lv)
    if (length(idx) < 2) return(NA_real_)
    sub_nav <- nav[idx]
    peak    <- cummax(sub_nav)
    min((sub_nav - peak) / peak, na.rm = TRUE)
  }, numeric(1))
}


# ---------------------------------------------------------------------------
# regime_stability_test
# ---------------------------------------------------------------------------

#' Test whether strategy beats benchmark in all regimes
#'
#' @param strategy_returns  Numeric vector of strategy daily returns.
#' @param benchmark_returns Numeric vector of benchmark daily returns
#'                          (same length).
#' @param regimes           Factor vector of regime classifications.
#' @return List with components:
#'   \describe{
#'     \item{regime_alpha}{Named vector: strategy minus benchmark Sharpe in
#'           each regime.}
#'     \item{regime_outperformance}{Named logical vector: TRUE if strategy
#'           Sharpe > benchmark Sharpe in each regime.}
#'     \item{all_regimes_positive}{Logical: TRUE if strategy beats benchmark
#'           in every regime.}
#'     \item{worst_regime}{Name of the regime with the lowest alpha.}
#'     \item{summary_df}{data.frame: regime, strat_sharpe, bench_sharpe,
#'           alpha, outperforms.}
#'   }
#'
#' @examples
#' \dontrun{
#'   rst <- regime_stability_test(strat_ret, bench_ret, regimes)
#' }
regime_stability_test <- function(strategy_returns, benchmark_returns, regimes) {
  if (length(strategy_returns) != length(regimes) ||
      length(benchmark_returns) != length(regimes)) {
    stop("All inputs must have the same length")
  }

  strat_sr  <- conditional_sharpe(strategy_returns, regimes)
  bench_sr  <- conditional_sharpe(benchmark_returns, regimes)

  alpha       <- strat_sr - bench_sr
  outperforms <- !is.na(alpha) & alpha > 0

  all_pos     <- all(outperforms, na.rm = TRUE)
  worst_reg   <- names(which.min(alpha))

  summary_df  <- data.frame(
    regime        = names(strat_sr),
    strat_sharpe  = strat_sr,
    bench_sharpe  = bench_sr,
    alpha         = alpha,
    outperforms   = outperforms,
    stringsAsFactors = FALSE
  )

  list(
    regime_alpha          = alpha,
    regime_outperformance = outperforms,
    all_regimes_positive  = all_pos,
    worst_regime          = worst_reg,
    summary_df            = summary_df
  )
}


# ---------------------------------------------------------------------------
# regime_aware_backtest
# ---------------------------------------------------------------------------

#' Run a full backtest with regime-conditional position sizing
#'
#' @param signal            Numeric vector: raw signal (not yet scaled to
#'                          position).  Positive = long, negative = short.
#' @param returns           Numeric vector of asset returns (same length as
#'                          signal).
#' @param regime_classifier Factor vector of regimes (from
#'                          classify_market_regime()) of the same length.
#' @param ...               Additional arguments passed to internal sizing:
#'   \describe{
#'     \item{trending_scale}{Multiplier for TRENDING regime (default 1.0).}
#'     \item{ranging_scale}{Multiplier for RANGING regime (default 1.0).}
#'     \item{high_vol_scale}{Multiplier for HIGH_VOL regime (default 0.5).}
#'     \item{crisis_scale}{Multiplier for CRISIS regime (default 0.0).}
#'     \item{vol_target}{Optional annualised volatility target.  If provided,
#'           positions are scaled to target this volatility before regime
#'           adjustments.}
#'   }
#' @return List with components:
#'   \describe{
#'     \item{returns}{Numeric vector: period returns of the strategy.}
#'     \item{nav}{Numeric vector: NAV series starting at 1.}
#'     \item{positions}{Numeric vector: position held each period.}
#'     \item{regime_sharpes}{Named vector: Sharpe by regime.}
#'     \item{overall_sharpe}{Scalar: overall annualised Sharpe.}
#'     \item{max_drawdown}{Scalar: maximum drawdown.}
#'   }
#'
#' @examples
#' \dontrun{
#'   bt <- regime_aware_backtest(signal, returns, regimes,
#'                               high_vol_scale = 0.5, crisis_scale = 0.0)
#' }
regime_aware_backtest <- function(signal, returns, regime_classifier, ...) {
  dots <- list(...)
  trending_scale <- if (!is.null(dots$trending_scale)) dots$trending_scale else 1.0
  ranging_scale  <- if (!is.null(dots$ranging_scale))  dots$ranging_scale  else 1.0
  high_vol_scale <- if (!is.null(dots$high_vol_scale)) dots$high_vol_scale else 0.5
  crisis_scale   <- if (!is.null(dots$crisis_scale))   dots$crisis_scale   else 0.0
  vol_target     <- dots$vol_target  # may be NULL

  n <- length(signal)
  if (length(returns) != n || length(regime_classifier) != n) {
    stop("signal, returns, and regime_classifier must have the same length")
  }

  regime_scale_map <- c(
    TRENDING = trending_scale,
    RANGING  = ranging_scale,
    HIGH_VOL = high_vol_scale,
    CRISIS   = crisis_scale
  )

  # Normalise signal
  sig_sd <- sd(signal, na.rm = TRUE)
  if (is.na(sig_sd) || sig_sd == 0) sig_sd <- 1
  norm_signal <- signal / sig_sd

  # Optional: vol targeting
  if (!is.null(vol_target) && vol_target > 0) {
    roll_vol <- zoo::rollapply(
      zoo::zoo(returns), width = 63,
      FUN = function(x) sd(x, na.rm = TRUE) * sqrt(252),
      fill = NA, align = "right"
    )
    roll_vol <- as.numeric(roll_vol)
    roll_vol[is.na(roll_vol) | roll_vol == 0] <- vol_target
    vol_scale <- vol_target / roll_vol
    norm_signal <- norm_signal * vol_scale
  }

  # Apply regime scaling
  reg_char <- as.character(regime_classifier)
  positions <- numeric(n)
  for (t in seq_len(n)) {
    rg  <- reg_char[t]
    scl <- if (!is.na(rg) && rg %in% names(regime_scale_map)) {
      regime_scale_map[rg]
    } else 1.0
    positions[t] <- norm_signal[t] * scl
  }

  # Compute returns: use lagged position (signal at t-1 -> return at t)
  lagged_pos <- c(0, positions[-n])
  strat_ret  <- lagged_pos * returns
  nav        <- cumprod(1 + strat_ret)

  mu  <- mean(strat_ret, na.rm = TRUE)
  sg  <- sd(strat_ret, na.rm = TRUE)
  osr <- if (!is.na(sg) && sg > 0) (mu / sg) * sqrt(252) else NA_real_

  peak <- cummax(nav)
  mdd  <- min((nav - peak) / peak, na.rm = TRUE)

  reg_sharpes <- conditional_sharpe(strat_ret, regime_classifier)

  list(
    returns        = strat_ret,
    nav            = nav,
    positions      = positions,
    regime_sharpes = reg_sharpes,
    overall_sharpe = osr,
    max_drawdown   = mdd
  )
}


# ---------------------------------------------------------------------------
# plot_regime_performance
# ---------------------------------------------------------------------------

#' Plot equity curve coloured by market regime (base R graphics)
#'
#' @param nav     Numeric vector: NAV series starting at 1.
#' @param regimes Factor vector of regime classifications (same length as nav).
#' @return Invisibly returns NULL.  Produces a base R plot as a side effect.
#'
#' @details
#'   Each segment of the equity curve is coloured according to the prevailing
#'   regime.  A legend is drawn in the top-left corner.
#'
#' @examples
#' \dontrun{
#'   plot_regime_performance(bt$nav, regimes)
#' }
plot_regime_performance <- function(nav, regimes) {
  if (length(nav) != length(regimes)) {
    stop("nav and regimes must have the same length")
  }

  regime_colors <- c(
    TRENDING = "#2166ac",  # blue
    RANGING  = "#4dac26",  # green
    HIGH_VOL = "#f4a582",  # orange
    CRISIS   = "#d6604d"   # red
  )

  lvls      <- REGIME_LEVELS
  reg_char  <- as.character(regimes)
  n         <- length(nav)

  x_vals <- seq_len(n)

  plot(x_vals, nav, type = "n",
       main = "Equity Curve by Market Regime",
       xlab = "Period", ylab = "NAV",
       panel.first = grid(col = "grey90", lty = 1))

  for (t in seq_len(n - 1)) {
    rg  <- reg_char[t]
    col <- if (!is.na(rg) && rg %in% names(regime_colors)) {
      regime_colors[rg]
    } else "grey50"
    graphics::segments(x_vals[t], nav[t], x_vals[t + 1], nav[t + 1],
                       col = col, lwd = 1.5)
  }

  # Drawdown shading
  peak <- cummax(nav)
  dd   <- (nav - peak) / peak
  dd_y_min <- min(nav) * 0.98
  dd_scaled <- dd * (max(nav) - min(nav)) / abs(min(dd) + 1e-10) * 0.1 + min(nav)

  # Legend
  graphics::legend("topleft",
                   legend = lvls,
                   col    = regime_colors[lvls],
                   lwd    = 2,
                   bty    = "n",
                   cex    = 0.85)

  invisible(NULL)
}
