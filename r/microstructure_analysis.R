# microstructure_analysis.R
# Market microstructure analytics: VPIN, Kyle's lambda, Amihud illiquidity,
# Hasbrouck information share, bid-ask decomposition, intraday seasonality.
#
# Dependencies: tidyverse, ggplot2, stats (base), MASS (base)
# Author: srfm-lab

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(scales)
  library(MASS)      # ginv() for Hasbrouck
})

# ===========================================================================
# 1. VPIN -- Volume-Synchronized Probability of Informed Trading
# ===========================================================================

#' Classify individual trades by tick rule:
#' +1 if price >= previous price, -1 if price < previous price.
#' First observation defaults to +1.
tick_rule <- function(prices) {
  n <- length(prices)
  direction <- integer(n)
  direction[1] <- 1L
  for (i in 2:n) {
    if (prices[i] > prices[i - 1])      direction[i] <-  1L
    else if (prices[i] < prices[i - 1]) direction[i] <- -1L
    else                                 direction[i] <- direction[i - 1]
  }
  direction
}

#' Bulk volume classification using the tick rule
#' @param price  vector of trade prices
#' @param volume vector of trade volumes (same length)
#' @return data.frame with buy_vol and sell_vol columns
classify_volume_tick <- function(price, volume) {
  direction <- tick_rule(price)
  buy_vol  <- volume * (direction == 1L)
  sell_vol <- volume * (direction == -1L)
  tibble(price, volume, direction, buy_vol, sell_vol)
}

#' Compute VPIN (Easley, Lopez de Prado, O'Hara 2012)
#' @param price   numeric vector of trade prices (chronological)
#' @param volume  numeric vector of trade volumes
#' @param bucket_size  volume per bucket (default = total_vol / 50)
#' @param n_buckets_window  rolling window for VPIN (default 50)
#' @return list with vpin_series (tibble) and scalar vpin (most recent)
compute_vpin <- function(price, volume, bucket_size = NULL, n_buckets_window = 50L) {
  stopifnot(length(price) == length(volume))

  total_vol <- sum(volume)
  if (is.null(bucket_size)) bucket_size <- total_vol / 50

  classified <- classify_volume_tick(price, volume)

  # Fill volume buckets
  buckets <- list()
  remaining_buy  <- 0
  remaining_sell <- 0

  for (i in seq_len(nrow(classified))) {
    remaining_buy  <- remaining_buy  + classified$buy_vol[i]
    remaining_sell <- remaining_sell + classified$sell_vol[i]

    total_remaining <- remaining_buy + remaining_sell
    n_full_buckets  <- floor(total_remaining / bucket_size)

    if (n_full_buckets >= 1) {
      frac_buy  <- remaining_buy  / total_remaining
      frac_sell <- remaining_sell / total_remaining

      for (b in seq_len(n_full_buckets)) {
        buckets[[length(buckets) + 1]] <- list(
          end_price  = classified$price[i],
          bucket_buy  = bucket_size * frac_buy,
          bucket_sell = bucket_size * frac_sell
        )
      }

      allocated <- n_full_buckets * bucket_size
      remaining_buy  <- remaining_buy  - allocated * frac_buy
      remaining_sell <- remaining_sell - allocated * frac_sell
    }
  }

  if (length(buckets) < n_buckets_window) {
    warning("Not enough buckets to compute VPIN; returning NA")
    return(list(vpin = NA_real_, vpin_series = tibble()))
  }

  bucket_df <- bind_rows(map(buckets, as_tibble))
  bucket_df <- bucket_df %>%
    mutate(imbalance = abs(bucket_buy - bucket_sell))

  # Rolling window VPIN
  n <- nrow(bucket_df)
  vpin_vals <- rep(NA_real_, n)
  for (i in n_buckets_window:n) {
    window_imbal <- bucket_df$imbalance[(i - n_buckets_window + 1):i]
    vpin_vals[i] <- mean(window_imbal) / bucket_size
  }

  vpin_series <- bucket_df %>%
    mutate(bucket_index = row_number(), vpin = vpin_vals)

  list(
    vpin        = tail(na.omit(vpin_vals), 1),
    vpin_series = vpin_series
  )
}

# ===========================================================================
# 2. Kyle's Lambda
# ===========================================================================

#' Estimate Kyle's lambda via OLS: delta_p = lambda * signed_volume + epsilon
#' @param price   vector of prices (mid or transaction)
#' @param volume  vector of trade volumes (signed: + buy, - sell)
#' @return list with lambda (slope), r_squared, tstat
kyles_lambda <- function(price, volume) {
  stopifnot(length(price) == length(volume), length(price) >= 10)

  delta_p <- diff(price)
  sv      <- volume[-length(volume)]  # lagged signed volume

  df <- tibble(delta_p, sv) %>% filter(is.finite(delta_p), is.finite(sv))
  fit <- lm(delta_p ~ sv - 1, data = df)   # force through origin

  lambda  <- coef(fit)[["sv"]]
  r_sq    <- summary(fit)$r.squared
  se      <- summary(fit)$coefficients["sv", "Std. Error"]
  tstat   <- lambda / se

  list(
    lambda    = lambda,
    r_squared = r_sq,
    t_stat    = tstat,
    se        = se,
    n_obs     = nrow(df)
  )
}

# ===========================================================================
# 3. Amihud Illiquidity Ratio
# ===========================================================================

#' Compute daily Amihud illiquidity: |return| / dollar volume
#' @param close  vector of closing prices
#' @param volume vector of daily share volumes
#' @param window rolling window (default 22 trading days)
#' @return tibble with date index, amihud, rolling_amihud
amihud_illiquidity <- function(close, volume, price = close, window = 22L) {
  stopifnot(length(close) == length(volume))
  n <- length(close)

  ret        <- c(NA, diff(log(close)))
  dollar_vol <- volume * price

  illiq <- abs(ret) / dollar_vol
  illiq[!is.finite(illiq)] <- NA

  # Rolling mean
  rolling_illiq <- rep(NA_real_, n)
  for (i in window:n) {
    vals <- illiq[(i - window + 1):i]
    rolling_illiq[i] <- mean(vals, na.rm = TRUE)
  }

  tibble(
    idx             = seq_len(n),
    log_return      = ret,
    dollar_volume   = dollar_vol,
    amihud_daily    = illiq,
    amihud_rolling  = rolling_illiq
  )
}

# ===========================================================================
# 4. Hasbrouck Information Share (state-space / Kalman)
# ===========================================================================

# The information share is estimated via a VECM-style approach:
# For two cointegrated price series p1, p2, we estimate the error-correction
# model and decompose variance of the common factor innovation.

#' Estimate Hasbrouck information share for two price series
#' using the Gonzalo-Granger (1995) common factor representation.
#' @param p1  price series 1 (vector)
#' @param p2  price series 2 (vector, same length)
#' @param lag number of VAR lags in the VECM (default 5)
#' @return list with IS1, IS2 (information shares summing to 1)
hasbrouck_info_share <- function(p1, p2, lag = 5L) {
  stopifnot(length(p1) == length(p2), length(p1) > lag * 2 + 10)

  dp1 <- diff(p1)
  dp2 <- diff(p2)
  n   <- length(dp1)

  # Error-correction term (lagged spread)
  ec  <- (p1 - p2)[-(length(p1))]  # length n

  # Build lagged difference matrix for VAR lags
  build_lag_matrix <- function(x, lags) {
    n <- length(x)
    do.call(cbind, lapply(seq_len(lags), function(l) {
      c(rep(NA, l), x[seq_len(n - l)])
    }))
  }

  lag_dp1 <- build_lag_matrix(dp1, lag)
  lag_dp2 <- build_lag_matrix(dp2, lag)

  # Trim to complete observations
  valid_rows <- (lag + 1):n
  Y1 <- dp1[valid_rows]
  Y2 <- dp2[valid_rows]
  ec_trim <- ec[valid_rows]
  X_lags  <- cbind(lag_dp1[valid_rows, , drop = FALSE],
                   lag_dp2[valid_rows, , drop = FALSE])

  X <- cbind(1, ec_trim, X_lags)

  # OLS for each equation
  fit1 <- tryCatch(lm.fit(X, Y1), error = function(e) NULL)
  fit2 <- tryCatch(lm.fit(X, Y2), error = function(e) NULL)

  if (is.null(fit1) || is.null(fit2)) {
    warning("Hasbrouck OLS failed; returning equal shares")
    return(list(IS1 = 0.5, IS2 = 0.5))
  }

  resid1 <- Y1 - X %*% fit1$coefficients
  resid2 <- Y2 - X %*% fit2$coefficients

  # Sigma matrix of residuals
  sigma_mat <- cov(cbind(resid1, resid2))

  # Adjustment vector: alpha (EC loading coefficients)
  alpha1 <- fit1$coefficients["ec_trim"]
  alpha2 <- fit2$coefficients["ec_trim"]

  # Gonzalo-Granger common factor: orthogonal complement of alpha
  # G_perp proportional to [-alpha2, alpha1] (orthogonal to alpha)
  g_perp <- c(-alpha2, alpha1)
  g_perp_norm <- g_perp / sqrt(sum(g_perp^2))

  # Information share of venue i: (g_perp[i])^2 * sigma_i / (G_perp' Sigma G_perp)
  # Upper and lower bounds (Hasbrouck 1995 use Cholesky factorization)
  # Here we compute the midpoint share using the diagonal approximation.
  chol_sigma <- tryCatch(chol(sigma_mat), error = function(e) {
    sigma_mat_reg <- sigma_mat + diag(1e-10, 2)
    chol(sigma_mat_reg)
  })

  # F = G_perp' * chol_sigma  (row vector)
  F_vec <- g_perp_norm %*% t(chol_sigma)
  IS_raw <- F_vec^2

  total_IS <- sum(IS_raw)
  if (total_IS == 0) {
    return(list(IS1 = 0.5, IS2 = 0.5, alpha1 = alpha1, alpha2 = alpha2))
  }

  list(
    IS1    = IS_raw[1] / total_IS,
    IS2    = IS_raw[2] / total_IS,
    alpha1 = alpha1,
    alpha2 = alpha2,
    sigma  = sigma_mat
  )
}

# ===========================================================================
# 5. Bid-Ask Spread Decomposition
# ===========================================================================

#' Roll model: estimate effective spread from return autocovariance
#' Roll (1984): c = sqrt(-cov(delta_p_t, delta_p_{t-1})) when > 0
#' @param mid_prices  vector of transaction/mid prices
#' @return list with effective_spread, c (half-spread), gamma_1 (autocovariance)
roll_spread <- function(mid_prices) {
  dp      <- diff(mid_prices)
  gamma_1 <- cov(dp[-length(dp)], dp[-1])

  if (gamma_1 >= 0) {
    warning("Roll model: autocovariance >= 0; spread estimate is 0")
    return(list(effective_spread = 0, c = 0, gamma_1 = gamma_1))
  }

  c_hat <- sqrt(-gamma_1)
  list(
    effective_spread = 2 * c_hat,
    c                = c_hat,
    gamma_1          = gamma_1
  )
}

#' Glosten-Harris (1988) spread decomposition
#' Regresses price change on signed order flow to decompose into:
#' (1) transitory component z (inventory / order-processing)
#' (2) permanent component lambda (adverse selection / information)
#'
#' Model: delta_p_t = (z + lambda) * q_t - z * q_{t-1} + epsilon_t
#' where q_t in {+1, -1} is trade direction.
#'
#' @param price      vector of transaction prices
#' @param direction  vector of trade directions (+1 buy, -1 sell)
#' @return list with z (transitory half-spread), lambda (adverse selection half-spread),
#'         total_half_spread, r_squared
glosten_harris <- function(price, direction) {
  stopifnot(length(price) == length(direction))

  dp      <- diff(price)
  q_t     <- direction[-1]
  q_lag   <- direction[-length(direction)]

  df <- tibble(dp, q_t, q_lag) %>%
    filter(is.finite(dp))

  if (nrow(df) < 20) stop("glosten_harris: insufficient observations")

  # OLS: dp = a1 * q_t + a2 * q_lag + e
  # a1 = z + lambda, a2 = -z  =>  z = -a2, lambda = a1 + a2
  fit  <- lm(dp ~ q_t + q_lag - 1, data = df)
  coef <- coef(fit)
  a1   <- coef[["q_t"]]
  a2   <- coef[["q_lag"]]

  z_hat      <- -a2
  lambda_hat <- a1 + a2

  list(
    z                  = z_hat,
    lambda             = lambda_hat,
    total_half_spread  = a1,
    r_squared          = summary(fit)$r.squared,
    adverse_selection_pct = lambda_hat / a1
  )
}

# ===========================================================================
# 6. Intraday Seasonality
# ===========================================================================

#' Compute intraday seasonality from 15-minute bar data
#' @param bars  data.frame with columns: datetime (POSIXct), volume, dollar_vol, spread
#' @return tibble with interval (hh:mm string), avg_volume, avg_dollar_vol,
#'         avg_spread, activity_index (normalized)
intraday_seasonality <- function(bars) {
  stopifnot(
    all(c("datetime", "volume") %in% names(bars)),
    inherits(bars$datetime, c("POSIXct", "POSIXt"))
  )

  bars %>%
    mutate(
      hour     = as.integer(format(datetime, "%H")),
      minute   = as.integer(format(datetime, "%M")),
      interval = sprintf("%02d:%02d", hour, (minute %/% 15) * 15)
    ) %>%
    group_by(interval) %>%
    summarise(
      avg_volume     = mean(volume,     na.rm = TRUE),
      avg_dollar_vol = if ("dollar_vol" %in% names(cur_data())) mean(dollar_vol, na.rm = TRUE) else NA_real_,
      avg_spread     = if ("spread"     %in% names(cur_data())) mean(spread,     na.rm = TRUE) else NA_real_,
      n_obs          = n(),
      .groups        = "drop"
    ) %>%
    mutate(
      activity_index = avg_volume / mean(avg_volume, na.rm = TRUE)
    ) %>%
    arrange(interval)
}

#' Plot intraday volume seasonality
plot_intraday_volume <- function(seasonality_df) {
  ggplot(seasonality_df, aes(x = interval, y = activity_index, group = 1)) +
    geom_line(color = "#2196F3", linewidth = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
    scale_y_continuous(labels = function(x) sprintf("%.1fx", x)) +
    labs(
      title    = "Intraday Volume Seasonality (15-min Bars)",
      subtitle = "Activity index: bar volume / average bar volume",
      x        = "Time of Day",
      y        = "Activity Index"
    ) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))
}

#' Plot VPIN time series
plot_vpin <- function(vpin_series) {
  vpin_series %>%
    filter(!is.na(vpin)) %>%
    ggplot(aes(x = bucket_index, y = vpin)) +
    geom_line(color = "#E91E63", linewidth = 0.8) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50",
               linewidth = 0.5) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(
      title    = "VPIN -- Volume-Synchronized Probability of Informed Trading",
      subtitle = "Values > 0.5 indicate elevated informed trading probability",
      x        = "Volume Bucket Index",
      y        = "VPIN"
    ) +
    theme_minimal(base_size = 12)
}

#' Summarize all microstructure metrics for one instrument
#' @param trades  data.frame with columns: price, volume, direction (signed)
#' @param daily   data.frame with columns: close, volume for Amihud
#' @return named list of metric estimates
microstructure_summary <- function(trades, daily = NULL) {
  roll   <- roll_spread(trades$price)
  lambda <- kyles_lambda(trades$price, trades$volume * trades$direction)
  vpin_r <- compute_vpin(trades$price, abs(trades$volume))

  out <- list(
    roll_effective_spread  = roll$effective_spread,
    roll_c                 = roll$c,
    kyles_lambda           = lambda$lambda,
    kyles_lambda_tstat     = lambda$t_stat,
    kyles_r_squared        = lambda$r_squared,
    vpin                   = vpin_r$vpin
  )

  if (!is.null(daily)) {
    amihud_df  <- amihud_illiquidity(daily$close, daily$volume)
    out$amihud <- tail(na.omit(amihud_df$amihud_rolling), 1)
  }

  if ("direction" %in% names(trades) && length(unique(trades$direction)) == 2) {
    gh <- tryCatch(
      glosten_harris(trades$price, trades$direction),
      error = function(e) NULL
    )
    if (!is.null(gh)) {
      out$gh_z           <- gh$z
      out$gh_lambda      <- gh$lambda
      out$gh_adv_sel_pct <- gh$adverse_selection_pct
    }
  }

  out
}
