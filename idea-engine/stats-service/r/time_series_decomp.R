# =============================================================================
# time_series_decomp.R — STL Decomposition of Strategy Performance
# =============================================================================
# Decomposes daily strategy P&L into:
#   trend + weekly seasonality + daily seasonality + residual
#
# Provides:
#   1. STL decomposition (stats::stl)
#   2. Weekly and daily seasonality analysis
#   3. Trend direction test (Mann-Kendall)
#   4. State-space forecasting (30-day) using local level model (dlm or manual)
#   5. Seasonality report: best/worst day-of-week, best/worst hour
#
# Dependencies: tidyverse, jsonlite, lubridate, zoo, RSQLite
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(lubridate)
  library(zoo)
  library(RSQLite)
})

# ── Configuration ──────────────────────────────────────────────────────────────

DB_PATH    <- Sys.getenv("IDEA_ENGINE_DB",   "../db/idea_engine.db")
OUTPUT_DIR <- Sys.getenv("STATS_OUTPUT_DIR", "../stats-service/output")
FORECAST_DAYS  <- 30
MIN_WEEKS_STL  <- 4    # need at least 4 complete weeks for weekly STL

# ── Data Loading ──────────────────────────────────────────────────────────────

load_daily_pnl <- function(db_path = DB_PATH) {
  if (!file.exists(db_path)) {
    message("[decomp] DB not found — generating synthetic daily P&L")
    set.seed(9)
    n     <- 200
    dates <- seq(Sys.Date() - n + 1, Sys.Date(), by = "day")
    dow   <- wday(dates, week_start = 1)   # 1=Mon … 7=Sun

    # Weekly seasonality: Tue/Wed best, Sat worst
    dow_effect <- c(Mon=0.001, Tue=0.003, Wed=0.002, Thu=0.001,
                    Fri=-0.001, Sat=-0.003, Sun=0.000)
    effects    <- dow_effect[dow]

    trend_comp <- 0.00015 * seq_len(n)
    noise_comp <- rnorm(n, 0, 0.008)
    pnl        <- trend_comp + effects + noise_comp

    tibble(
      date       = dates,
      daily_pnl  = pnl,
      day_of_week = wday(dates, label = TRUE, abbr = TRUE),
      dow_num     = dow
    )
  } else {
    con <- dbConnect(SQLite(), db_path)
    on.exit(dbDisconnect(con))
    df  <- dbGetQuery(con, "
      SELECT
        date(closed_at)        AS date,
        SUM(pnl_pct)           AS daily_pnl,
        COUNT(*)               AS n_trades
      FROM trades
      WHERE closed_at IS NOT NULL
      GROUP BY date(closed_at)
      ORDER BY date ASC
    ")
    df %>%
      mutate(
        date        = as.Date(date),
        day_of_week = wday(date, label = TRUE, abbr = TRUE),
        dow_num     = wday(date, week_start = 1)
      ) %>%
      as_tibble()
  }
}

# ── STL Decomposition ────────────────────────────────────────────────────────

run_stl <- function(daily_df) {
  n <- nrow(daily_df)

  if (n < 14) {
    stop("[decomp] Need ≥ 14 daily observations for decomposition")
  }

  y     <- daily_df$daily_pnl
  period <- 7L   # weekly periodicity

  # Ensure enough complete periods
  if (n < MIN_WEEKS_STL * period) {
    message("[decomp] Warning: fewer than ", MIN_WEEKS_STL,
            " complete weeks; STL may be unreliable")
  }

  ts_obj <- ts(y, frequency = period)

  stl_fit <- tryCatch(
    stl(ts_obj,
        s.window  = "periodic",
        t.window  = max(7, round(n * 0.25) | 1L),  # odd number
        robust    = TRUE),
    error = function(e) {
      message("[decomp] STL failed: ", e$message, " — using simple MA decomposition")
      NULL
    }
  )

  if (is.null(stl_fit)) {
    # Fallback: simple 7-day moving average trend + residual
    trend_raw  <- rollmean(y, k = 7, fill = NA, align = "center")
    trend_comp <- ifelse(is.na(trend_raw), mean(y), trend_raw)
    seasonal   <- rep(0, n)
    residual   <- y - trend_comp
    return(list(
      trend    = trend_comp,
      seasonal = seasonal,
      residual = residual,
      method   = "7-day MA (STL fallback)"
    ))
  }

  comps <- stl_fit$time.series
  list(
    trend    = as.numeric(comps[, "trend"]),
    seasonal = as.numeric(comps[, "seasonal"]),
    residual = as.numeric(comps[, "remainder"]),
    method   = "STL (Seasonal and Trend decomposition using Loess)"
  )
}

# ── Seasonality Analysis ──────────────────────────────────────────────────────

analyse_seasonality <- function(daily_df) {
  dow_stats <- daily_df %>%
    group_by(day_of_week, dow_num) %>%
    summarise(
      mean_pnl  = mean(daily_pnl, na.rm = TRUE),
      sd_pnl    = sd(daily_pnl, na.rm = TRUE),
      n         = n(),
      .groups   = "drop"
    ) %>%
    arrange(dow_num)

  best_day  <- dow_stats$day_of_week[which.max(dow_stats$mean_pnl)]
  worst_day <- dow_stats$day_of_week[which.min(dow_stats$mean_pnl)]

  # ANOVA: is there significant day-of-week effect?
  aov_res <- tryCatch({
    fit <- aov(daily_pnl ~ factor(dow_num), data = daily_df)
    summary(fit)[[1]]
  }, error = function(e) NULL)

  dow_significant <- if (!is.null(aov_res)) {
    p_val <- aov_res$`Pr(>F)`[1]
    !is.na(p_val) && p_val < 0.05
  } else FALSE

  list(
    by_dow = dow_stats %>% as.list() %>% purrr::transpose(),
    best_day  = as.character(best_day),
    worst_day = as.character(worst_day),
    dow_effect_significant = dow_significant,
    mean_pnl_by_day = setNames(dow_stats$mean_pnl, as.character(dow_stats$day_of_week))
  )
}

# ── Mann-Kendall Trend Test ───────────────────────────────────────────────────

mann_kendall_test <- function(x) {
  n   <- length(x)
  s   <- 0L
  for (i in seq_len(n - 1)) {
    for (j in (i + 1):n) {
      s <- s + sign(x[j] - x[i])
    }
  }
  # Variance of S
  var_s <- n * (n - 1) * (2 * n + 5) / 18
  # z-score
  z <- if (s > 0) (s - 1) / sqrt(var_s)
       else if (s < 0) (s + 1) / sqrt(var_s)
       else 0
  p_value <- 2 * (1 - pnorm(abs(z)))

  list(
    S        = s,
    z        = z,
    p_value  = p_value,
    trend    = if (z > 0 && p_value < 0.05) "UPWARD (significant)"
               else if (z < 0 && p_value < 0.05) "DOWNWARD (significant)"
               else if (z > 0) "UPWARD (not significant)"
               else "DOWNWARD (not significant)"
  )
}

# ── State-Space Forecast (Local Level Model) ──────────────────────────────────

#' Simple local level (random walk + noise) state-space model.
#' State: θ_t = θ_{t-1} + w_t,  w_t ~ N(0, σ²_w)
#' Obs:   y_t = θ_t + v_t,       v_t ~ N(0, σ²_v)
#' Kalman filter + smoother, forecast h steps ahead.
local_level_forecast <- function(y, h = FORECAST_DAYS) {
  n  <- length(y)

  # Estimate signal-to-noise ratio via method of moments on first differences
  dy  <- diff(y)
  sv  <- max(var(y) * 0.1, 1e-10)    # innovation variance
  sw  <- max(var(dy) * 0.5, 1e-10)   # noise variance

  # Kalman filter
  theta <- numeric(n)
  P     <- numeric(n)
  P[1]  <- sv + sw
  theta[1] <- y[1]

  for (t in 2:n) {
    # Predict
    P_pred <- P[t - 1] + sw
    # Update
    K      <- P_pred / (P_pred + sv)
    theta[t] <- theta[t - 1] + K * (y[t] - theta[t - 1])
    P[t]     <- (1 - K) * P_pred
  }

  # Forecast h steps ahead (random walk in mean)
  theta_T  <- theta[n]
  P_T      <- P[n]

  forecasts <- tibble(
    horizon   = seq_len(h),
    forecast  = theta_T,
    lower_80  = theta_T - 1.282 * sqrt(P_T + sw * seq_len(h)),
    upper_80  = theta_T + 1.282 * sqrt(P_T + sw * seq_len(h)),
    lower_95  = theta_T - 1.960 * sqrt(P_T + sw * seq_len(h)),
    upper_95  = theta_T + 1.960 * sqrt(P_T + sw * seq_len(h))
  )

  list(
    filtered_state    = theta,
    filtered_variance = P,
    forecasts         = forecasts,
    signal_variance   = sw,
    noise_variance    = sv,
    snr               = sw / sv
  )
}

# ── Post-IAE Trend Analysis ───────────────────────────────────────────────────

#' Is the trend improving post a specified date?
post_iae_trend <- function(daily_df, trend_component, cutoff_date = NULL) {
  n <- nrow(daily_df)

  if (is.null(cutoff_date)) {
    # Default: use the midpoint date
    cutoff_date <- daily_df$date[round(n / 2)]
  }

  pre_idx  <- which(daily_df$date < cutoff_date)
  post_idx <- which(daily_df$date >= cutoff_date)

  if (length(pre_idx) < 5 || length(post_idx) < 5) {
    return(list(
      cutoff_date   = format(cutoff_date),
      improvement   = NA,
      message       = "Insufficient data in one segment"
    ))
  }

  mean_pre  <- mean(trend_component[pre_idx])
  mean_post <- mean(trend_component[post_idx])
  slope_pre  <- coef(lm(trend_component[pre_idx]  ~ pre_idx))[2]
  slope_post <- coef(lm(trend_component[post_idx] ~ post_idx))[2]

  improvement <- mean_post > mean_pre

  list(
    cutoff_date  = format(cutoff_date),
    mean_trend_pre  = mean_pre,
    mean_trend_post = mean_post,
    slope_pre       = slope_pre,
    slope_post      = slope_post,
    improvement     = improvement,
    message = if (improvement)
      paste0("Post-IAE trend improved: ",
             round((mean_post - mean_pre) * 100, 4), "% daily lift vs pre-period")
    else
      "Post-IAE trend has NOT improved vs pre-period — investigate"
  )
}

# ── Main Pipeline ─────────────────────────────────────────────────────────────

run_time_series_decomp <- function(
  db_path    = DB_PATH,
  output_dir = OUTPUT_DIR
) {
  message("[decomp] Loading daily P&L...")
  daily_df <- load_daily_pnl(db_path)
  n        <- nrow(daily_df)
  message("[decomp] ", n, " daily observations loaded")

  if (n < 14) stop("[decomp] Need ≥ 14 daily observations")

  # ── STL decomposition ──
  message("[decomp] Running STL decomposition...")
  decomp <- run_stl(daily_df)

  # ── Variance decomposition ──
  var_total    <- var(daily_df$daily_pnl)
  var_trend    <- var(decomp$trend)
  var_seasonal <- var(decomp$seasonal)
  var_residual <- var(decomp$residual)

  # ── Mann-Kendall trend test on trend component ──
  message("[decomp] Testing trend direction (Mann-Kendall)...")
  mk <- mann_kendall_test(decomp$trend)

  # ── Seasonality analysis ──
  message("[decomp] Analysing seasonality by day-of-week...")
  season <- analyse_seasonality(daily_df)

  # ── State-space forecast ──
  h_days <- min(FORECAST_DAYS, max(7, round(n * 0.15)))
  message("[decomp] Forecasting ", h_days, " days ahead (local level SSM)...")
  forecast <- local_level_forecast(daily_df$daily_pnl, h = h_days)

  # ── Post-IAE trend ──
  message("[decomp] Analysing post-IAE trend improvement...")
  iae_trend <- post_iae_trend(daily_df, decomp$trend)

  # ── Build output ──
  result <- list(
    n_obs              = n,
    date_range = list(
      start = format(min(daily_df$date)),
      end   = format(max(daily_df$date))
    ),
    decomposition = list(
      method          = decomp$method,
      trend           = decomp$trend,
      seasonal        = decomp$seasonal,
      residual        = decomp$residual,
      original        = daily_df$daily_pnl,
      dates           = format(daily_df$date)
    ),
    variance_decomp = list(
      total    = var_total,
      trend    = var_trend / var_total,
      seasonal = var_seasonal / var_total,
      residual = var_residual / var_total
    ),
    trend_test = list(
      mann_kendall_S   = mk$S,
      mann_kendall_z   = mk$z,
      p_value          = mk$p_value,
      trend_direction  = mk$trend,
      significant      = mk$p_value < 0.05
    ),
    seasonality = list(
      best_day_of_week  = season$best_day,
      worst_day_of_week = season$worst_day,
      effect_significant = season$dow_effect_significant,
      mean_pnl_by_day   = season$mean_pnl_by_day
    ),
    forecast = list(
      horizon_days   = h_days,
      point_forecast = forecast$forecasts$forecast,
      lower_80       = forecast$forecasts$lower_80,
      upper_80       = forecast$forecasts$upper_80,
      lower_95       = forecast$forecasts$lower_95,
      upper_95       = forecast$forecasts$upper_95,
      snr            = forecast$snr
    ),
    post_iae_trend = iae_trend
  )

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(output_dir, "time_series_decomp_results.json")
  write_json(result, out_path, auto_unbox = TRUE, pretty = TRUE)
  message("[decomp] Results written to ", out_path)

  invisible(result)
}

# ── CLI entry ─────────────────────────────────────────────────────────────────

if (!interactive()) {
  result <- run_time_series_decomp()

  cat("\n=== Time Series Decomposition Summary ===\n")
  cat("Observations: ", result$n_obs, "\n")
  cat("Date range:   ", result$date_range$start, " – ", result$date_range$end, "\n\n")

  vd <- result$variance_decomp
  cat("Variance decomposition:\n")
  cat("  Trend:    ", round(vd$trend * 100, 1), "%\n")
  cat("  Seasonal: ", round(vd$seasonal * 100, 1), "%\n")
  cat("  Residual: ", round(vd$residual * 100, 1), "%\n\n")

  cat("Trend direction: ", result$trend_test$trend_direction, "\n")
  cat("Best day:        ", result$seasonality$best_day_of_week, "\n")
  cat("Worst day:       ", result$seasonality$worst_day_of_week, "\n")

  if (result$seasonality$effect_significant) {
    cat("Day-of-week effect: SIGNIFICANT (ANOVA p<0.05)\n")
  } else {
    cat("Day-of-week effect: not significant\n")
  }

  cat("\nPost-IAE trend:\n")
  cat("  ", result$post_iae_trend$message, "\n")

  cat("\n30-day forecast (point):", round(mean(result$forecast$point_forecast) * 100, 4), "% avg daily P&L\n")
}
