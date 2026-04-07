# cointegration_pairs.R
# Pairs trading and cointegration analysis for the SRFM quantitative trading system.
# Dependencies: zoo, xts, stats, tseries (for ADF)
# All functions follow base R + zoo/xts conventions.

suppressPackageStartupMessages({
  library(zoo)
  library(xts)
})

# Helper: load tseries lazily to keep the package optional at source time
.require_tseries <- function() {
  if (!requireNamespace("tseries", quietly = TRUE)) {
    stop("Package 'tseries' is required for ADF tests. Install with install.packages('tseries').")
  }
}


# ---------------------------------------------------------------------------
# engle_granger_test
# ---------------------------------------------------------------------------

#' Perform Engle-Granger cointegration test on two price series
#'
#' @param y1 Numeric vector: price series of asset 1.
#' @param y2 Numeric vector: price series of asset 2.
#' @return List with components:
#'   \describe{
#'     \item{hedge_ratio}{OLS hedge ratio (beta): y1 ~ beta * y2 + alpha.}
#'     \item{alpha}{Intercept of the OLS regression.}
#'     \item{residuals}{Numeric vector: y1 - beta * y2 - alpha.}
#'     \item{adf_stat}{ADF test statistic on the residuals.}
#'     \item{p_value}{Approximate p-value from tseries::adf.test().}
#'   }
#'
#' @details
#'   The hedge ratio is estimated via OLS of y1 on y2.  An ADF test is then
#'   applied to the residuals.  A small p-value rejects the null of a unit
#'   root (i.e., indicates cointegration).
#'
#' @examples
#' \dontrun{
#'   res <- engle_granger_test(log(prices$SPY), log(prices$IVV))
#' }
engle_granger_test <- function(y1, y2) {
  .require_tseries()

  if (length(y1) != length(y2)) {
    stop("y1 and y2 must have the same length")
  }

  ok  <- !is.na(y1) & !is.na(y2)
  if (sum(ok) < 10) {
    stop("Fewer than 10 non-NA observations -- cannot perform ADF test")
  }

  lm_fit    <- stats::lm(y1[ok] ~ y2[ok])
  coefs     <- stats::coef(lm_fit)
  alpha_hat <- coefs[1]
  beta_hat  <- coefs[2]
  resids    <- y1[ok] - beta_hat * y2[ok] - alpha_hat

  adf_result <- tseries::adf.test(resids, alternative = "stationary")

  list(
    hedge_ratio = beta_hat,
    alpha       = alpha_hat,
    residuals   = resids,
    adf_stat    = unname(adf_result$statistic),
    p_value     = adf_result$p.value
  )
}


# ---------------------------------------------------------------------------
# find_cointegrated_pairs
# ---------------------------------------------------------------------------

#' Find all cointegrated pairs from a price data.frame
#'
#' @param prices_df       data.frame or matrix (dates x symbols) of prices.
#'                        Column names are symbol names.
#' @param pvalue_threshold Numeric: maximum p-value to consider a pair
#'                         cointegrated (default 0.05).
#' @return data.frame with columns: symbol1, symbol2, hedge_ratio, alpha,
#'         adf_stat, p_value.  Rows are sorted by p_value ascending.
#'
#' @details
#'   All unique pairs are tested using engle_granger_test().  The function
#'   logs prices internally (expects raw price levels as input).
#'
#' @examples
#' \dontrun{
#'   pairs <- find_cointegrated_pairs(prices_df, pvalue_threshold = 0.05)
#' }
find_cointegrated_pairs <- function(prices_df, pvalue_threshold = 0.05) {
  .require_tseries()

  p_mat    <- as.matrix(prices_df)
  symbols  <- colnames(p_mat)
  n_syms   <- length(symbols)

  if (n_syms < 2) stop("Need at least 2 symbols")

  results <- list()
  idx     <- 1L

  for (i in seq_len(n_syms - 1)) {
    for (j in (i + 1):n_syms) {
      y1 <- log(p_mat[, i])
      y2 <- log(p_mat[, j])

      res <- tryCatch(
        engle_granger_test(y1, y2),
        error = function(e) NULL
      )

      if (!is.null(res) && !is.na(res$p_value)) {
        results[[idx]] <- data.frame(
          symbol1     = symbols[i],
          symbol2     = symbols[j],
          hedge_ratio = res$hedge_ratio,
          alpha       = res$alpha,
          adf_stat    = res$adf_stat,
          p_value     = res$p_value,
          stringsAsFactors = FALSE
        )
        idx <- idx + 1L
      }
    }
  }

  if (length(results) == 0) {
    return(data.frame(
      symbol1 = character(0), symbol2 = character(0),
      hedge_ratio = numeric(0), alpha = numeric(0),
      adf_stat = numeric(0), p_value = numeric(0),
      stringsAsFactors = FALSE
    ))
  }

  out <- do.call(rbind, results)
  out <- out[out$p_value <= pvalue_threshold, , drop = FALSE]
  out <- out[order(out$p_value), , drop = FALSE]
  rownames(out) <- NULL
  out
}


# ---------------------------------------------------------------------------
# kalman_hedge_ratio
# ---------------------------------------------------------------------------

#' Estimate a time-varying hedge ratio via a Kalman filter
#'
#' @param y1    Numeric vector: price series of asset 1 (observation).
#' @param y2    Numeric vector: price series of asset 2 (regressor).
#' @param delta Numeric: process noise variance (state transition variance).
#'              Smaller values produce smoother estimates (default 0.0001).
#' @return List with components:
#'   \describe{
#'     \item{beta}{Numeric vector: filtered hedge ratio at each time step.}
#'     \item{P}{Numeric vector: filtered state variance (uncertainty).}
#'     \item{innovations}{Numeric vector: one-step prediction errors.}
#'     \item{innovation_var}{Numeric vector: innovation variances.}
#'   }
#'
#' @details
#'   State equation:  beta_t = beta_{t-1} + w_t,  w_t ~ N(0, delta)
#'   Observation:     y1_t   = beta_t * y2_t + v_t, v_t ~ N(0, R_t)
#'   The observation noise R_t is estimated adaptively from recent
#'   innovation variance.
#'
#' @examples
#' \dontrun{
#'   kf  <- kalman_hedge_ratio(log(prices$SPY), log(prices$IVV))
#'   plot(kf$beta, type = "l", main = "Time-varying hedge ratio")
#' }
kalman_hedge_ratio <- function(y1, y2, delta = 0.0001) {
  n <- length(y1)
  if (length(y2) != n) stop("y1 and y2 must have the same length")

  beta   <- numeric(n)
  P      <- numeric(n)
  innov  <- numeric(n)
  S_vec  <- numeric(n)  # innovation variance

  # Initialise
  ok_init <- which(!is.na(y1) & !is.na(y2))
  if (length(ok_init) < 2) stop("Insufficient non-NA observations for initialisation")

  t0 <- ok_init[1]
  # Initial OLS estimate over first min(20, available) obs
  n_init   <- min(20L, length(ok_init))
  init_idx <- ok_init[seq_len(n_init)]
  lm0      <- stats::lm(y1[init_idx] ~ y2[init_idx] - 1)
  beta[t0] <- stats::coef(lm0)[1]
  P[t0]    <- 1.0
  R        <- stats::var(stats::resid(lm0))
  if (is.na(R) || R <= 0) R <- 1e-4

  for (t in (t0 + 1):n) {
    if (is.na(y1[t]) || is.na(y2[t])) {
      beta[t] <- beta[t - 1]
      P[t]    <- P[t - 1] + delta
      innov[t]  <- NA_real_
      S_vec[t]  <- NA_real_
      next
    }

    # Predict
    beta_pred <- beta[t - 1]
    P_pred    <- P[t - 1] + delta

    # Observation: y1 = beta * y2
    H   <- y2[t]
    y_hat <- H * beta_pred
    S     <- H^2 * P_pred + R
    K     <- P_pred * H / S
    e     <- y1[t] - y_hat

    # Update
    beta[t]   <- beta_pred + K * e
    P[t]      <- (1 - K * H) * P_pred
    innov[t]  <- e
    S_vec[t]  <- S

    # Adaptive R update (exponential moving average of squared innovations)
    R <- 0.95 * R + 0.05 * e^2
  }

  list(
    beta           = beta,
    P              = P,
    innovations    = innov,
    innovation_var = S_vec
  )
}


# ---------------------------------------------------------------------------
# compute_spread
# ---------------------------------------------------------------------------

#' Compute the spread between two assets given a hedge ratio
#'
#' @param y1          Numeric vector: price or log-price of asset 1.
#' @param y2          Numeric vector: price or log-price of asset 2.
#' @param hedge_ratio Numeric scalar or vector.  If a vector it must match
#'                    the length of y1/y2 (used for time-varying hedge ratios).
#' @return Numeric vector: spread = y1 - hedge_ratio * y2.
#'
#' @examples
#' \dontrun{
#'   spread <- compute_spread(log(prices$SPY), log(prices$IVV), 0.98)
#' }
compute_spread <- function(y1, y2, hedge_ratio) {
  if (length(y1) != length(y2)) stop("y1 and y2 must have the same length")
  if (length(hedge_ratio) != 1 && length(hedge_ratio) != length(y1)) {
    stop("hedge_ratio must be a scalar or have the same length as y1")
  }
  y1 - hedge_ratio * y2
}


# ---------------------------------------------------------------------------
# spread_zscore
# ---------------------------------------------------------------------------

#' Compute rolling z-score of the spread
#'
#' @param spread Numeric vector: the spread series.
#' @param window Integer rolling window in periods (default 60).
#' @return Numeric vector (same length as spread) of z-scores.  The first
#'         \code{window - 1} values are NA.
#'
#' @examples
#' \dontrun{
#'   z <- spread_zscore(spread, window = 60)
#' }
spread_zscore <- function(spread, window = 60) {
  n    <- length(spread)
  z    <- rep(NA_real_, n)
  sp_z <- zoo::zoo(spread)

  roll_mean <- zoo::rollapply(sp_z, width = window, FUN = mean,
                              fill = NA, align = "right", na.rm = TRUE)
  roll_sd   <- zoo::rollapply(sp_z, width = window, FUN = sd,
                              fill = NA, align = "right", na.rm = TRUE)

  mask <- !is.na(roll_mean) & !is.na(roll_sd) & as.numeric(roll_sd) > 0
  z[mask] <- (spread[mask] - as.numeric(roll_mean)[mask]) /
               as.numeric(roll_sd)[mask]
  z
}


# ---------------------------------------------------------------------------
# half_life_ou
# ---------------------------------------------------------------------------

#' Estimate mean-reversion half-life of an OU process from AR(1) fit
#'
#' @param spread Numeric vector: the spread series.
#' @return List with components:
#'   \describe{
#'     \item{half_life}{Estimated half-life in periods.}
#'     \item{ar1_coef}{AR(1) coefficient.}
#'     \item{theta}{Mean-reversion speed: -log(ar1_coef).}
#'   }
#'
#' @details
#'   For an Ornstein-Uhlenbeck process the mean-reversion speed theta
#'   satisfies: spread_t = theta * (mu - spread_{t-1}) dt + sigma dW.
#'   In discrete time: spread_t = rho * spread_{t-1} + e_t, and
#'   theta = -log(rho), half_life = log(2) / theta.
#'
#' @examples
#' \dontrun{
#'   hl <- half_life_ou(spread)
#'   cat("Half-life:", hl$half_life, "days\n")
#' }
half_life_ou <- function(spread) {
  ok <- !is.na(spread)
  if (sum(ok) < 5) stop("Fewer than 5 non-NA observations")

  s   <- spread[ok]
  lag_s <- c(NA, s[-length(s)])
  fit   <- stats::lm(s ~ lag_s)
  rho   <- stats::coef(fit)["lag_s"]

  if (is.na(rho) || rho <= 0 || rho >= 1) {
    warning("AR(1) coefficient outside (0,1) -- half-life undefined")
    return(list(half_life = NA_real_, ar1_coef = rho, theta = NA_real_))
  }

  theta     <- -log(rho)
  half_life <- log(2) / theta

  list(
    half_life = half_life,
    ar1_coef  = rho,
    theta     = theta
  )
}


# ---------------------------------------------------------------------------
# pairs_backtest
# ---------------------------------------------------------------------------

#' Run a simple pairs trading backtest
#'
#' @param y1      Numeric vector: price series of asset 1.
#' @param y2      Numeric vector: price series of asset 2.
#' @param entry_z Numeric: z-score threshold to open a position (default 2).
#' @param exit_z  Numeric: z-score threshold to close a position (default 0.5).
#' @param stop_z  Numeric: z-score stop-loss threshold (default 3).
#' @return List with components:
#'   \describe{
#'     \item{trades_df}{data.frame of trades with entry/exit indices, PnL.}
#'     \item{equity_curve}{Numeric vector: cumulative PnL of the strategy.}
#'     \item{returns}{Numeric vector: period returns (scaled by initial spread SD).}
#'     \item{sharpe}{Annualised Sharpe ratio of the strategy.}
#'     \item{max_dd}{Maximum drawdown of the equity curve.}
#'     \item{n_trades}{Integer: total number of completed trades.}
#'   }
#'
#' @details
#'   The spread is computed with a fixed OLS hedge ratio estimated on the full
#'   sample.  A 60-period rolling z-score is used for entry/exit signals.
#'   Position: +1 (long spread) when z < -entry_z; -1 (short spread) when
#'   z > +entry_z.  Exit at +/- exit_z.  Stop-out at +/- stop_z.
#'
#' @examples
#' \dontrun{
#'   bt <- pairs_backtest(log(prices$SPY), log(prices$IVV))
#'   cat("Sharpe:", bt$sharpe, " Max DD:", bt$max_dd, "\n")
#' }
pairs_backtest <- function(y1, y2, entry_z = 2, exit_z = 0.5, stop_z = 3) {
  eg      <- engle_granger_test(y1, y2)
  beta    <- eg$hedge_ratio
  spread  <- compute_spread(y1, y2, beta)
  z       <- spread_zscore(spread, window = 60)

  n       <- length(z)
  position <- 0       # 0, +1, -1
  equity   <- numeric(n)
  period_r <- numeric(n)
  trades   <- list()
  trade_idx <- 0L
  entry_spread <- NA_real_

  for (t in 2:n) {
    if (is.na(z[t - 1])) {
      equity[t] <- equity[t - 1]
      next
    }

    # Check for exit/stop before entry
    if (position != 0) {
      exit_now <- (position ==  1 && z[t - 1] >= -exit_z) ||
                  (position == -1 && z[t - 1] <=  exit_z) ||
                  (position ==  1 && z[t - 1] <= -stop_z) ||
                  (position == -1 && z[t - 1] >=  stop_z)

      if (exit_now) {
        pnl <- position * (spread[t] - entry_spread)
        trade_idx <- trade_idx + 1L
        trades[[trade_idx]] <- data.frame(
          entry = t - 1L, exit = t,
          direction = position, pnl = pnl,
          stringsAsFactors = FALSE
        )
        position <- 0
        entry_spread <- NA_real_
      }
    }

    # Check for entry
    if (position == 0) {
      if (!is.na(z[t]) && z[t] < -entry_z) {
        position     <-  1
        entry_spread <- spread[t]
      } else if (!is.na(z[t]) && z[t] > entry_z) {
        position     <- -1
        entry_spread <- spread[t]
      }
    }

    # Mark-to-market
    mtm <- if (position != 0 && !is.na(entry_spread)) {
      position * (spread[t] - entry_spread)
    } else 0

    equity[t]   <- if (t > 1) equity[t - 1] + mtm - (if (t > 2) equity[t - 2] + mtm - equity[t - 1] else 0) else mtm
    period_r[t] <- spread[t] - spread[t - 1]
  }

  # Recompute equity as cumsum of position * spread changes
  pos_vec <- numeric(n)
  cur_pos  <- 0
  for (t in 2:n) {
    if (is.na(z[t - 1])) { pos_vec[t] <- cur_pos; next }
    if (cur_pos != 0) {
      ex <- (cur_pos ==  1 && z[t - 1] >= -exit_z) ||
            (cur_pos == -1 && z[t - 1] <=  exit_z) ||
            (cur_pos ==  1 && z[t - 1] <= -stop_z) ||
            (cur_pos == -1 && z[t - 1] >=  stop_z)
      if (ex) cur_pos <- 0
    }
    if (cur_pos == 0 && !is.na(z[t])) {
      if (z[t] < -entry_z) cur_pos <-  1
      else if (z[t] > entry_z) cur_pos <- -1
    }
    pos_vec[t] <- cur_pos
  }

  spread_changes <- c(0, diff(spread))
  spread_sd      <- sd(spread, na.rm = TRUE)
  if (is.na(spread_sd) || spread_sd == 0) spread_sd <- 1

  pnl_series  <- pos_vec * spread_changes
  scaled_ret  <- pnl_series / spread_sd
  equity_curve <- cumsum(pnl_series)

  mu <- mean(scaled_ret, na.rm = TRUE)
  sg <- sd(scaled_ret, na.rm = TRUE)
  sr <- if (!is.na(sg) && sg > 0) (mu / sg) * sqrt(252) else NA_real_

  nav <- 1 + cumsum(scaled_ret - min(cumsum(scaled_ret)))
  peak <- cummax(nav)
  mdd  <- min((nav - peak) / peak, na.rm = TRUE)

  trades_df <- if (length(trades) > 0) {
    do.call(rbind, trades)
  } else {
    data.frame(entry = integer(0), exit = integer(0),
               direction = integer(0), pnl = numeric(0),
               stringsAsFactors = FALSE)
  }

  list(
    trades_df    = trades_df,
    equity_curve = equity_curve,
    returns      = scaled_ret,
    sharpe       = sr,
    max_dd       = mdd,
    n_trades     = nrow(trades_df)
  )
}


# ---------------------------------------------------------------------------
# pairs_risk_metrics
# ---------------------------------------------------------------------------

#' Compute risk metrics for a pairs position
#'
#' @param spread       Numeric vector: the spread time series.
#' @param hedge_ratio  Scalar: hedge ratio (number of units of asset 2 per
#'                     unit of asset 1).
#' @param notional     Numeric: notional value of the position (in currency).
#' @return List with components:
#'   \describe{
#'     \item{spread_vol}{Annualised spread volatility.}
#'     \item{var_95}{95\% 1-day Value at Risk (parametric normal).}
#'     \item{var_99}{99\% 1-day Value at Risk.}
#'     \item{cvar_95}{95\% Conditional Value at Risk (Expected Shortfall).}
#'     \item{dv01}{DV01: change in spread value per 1bp move.}
#'     \item{half_life}{OU half-life in periods.}
#'   }
#'
#' @examples
#' \dontrun{
#'   rm <- pairs_risk_metrics(spread, 0.98, 1e6)
#' }
pairs_risk_metrics <- function(spread, hedge_ratio, notional) {
  spread_ret <- diff(spread)
  spread_ret <- spread_ret[!is.na(spread_ret)]

  vol_daily   <- sd(spread_ret, na.rm = TRUE)
  vol_annual  <- vol_daily * sqrt(252)
  var_95      <- notional * vol_daily * stats::qnorm(0.95)
  var_99      <- notional * vol_daily * stats::qnorm(0.99)

  # CVaR (Expected Shortfall) under normality
  alpha_95    <- 0.05
  cvar_95     <- notional * vol_daily *
    stats::dnorm(stats::qnorm(1 - alpha_95)) / alpha_95

  # DV01: sensitivity to 1 basis point move in spread
  dv01        <- notional * 0.0001

  hl <- tryCatch(half_life_ou(spread), error = function(e) list(half_life = NA))

  list(
    spread_vol  = vol_annual,
    var_95      = var_95,
    var_99      = var_99,
    cvar_95     = cvar_95,
    dv01        = dv01,
    half_life   = hl$half_life
  )
}
