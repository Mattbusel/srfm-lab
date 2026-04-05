# =============================================================================
# factor_model.R — Multi-Factor Regression Model for Strategy Returns
# =============================================================================
# Factors: BTC_return, ETH_return, log(VIX), DXY_return,
#          hour_sin, hour_cos, day_of_week, hold_duration_log
#
# Methods:
#   1. OLS (lm)
#   2. Robust regression (MASS::rlm, Huber M-estimation)
#   3. Quantile regression (quantreg::rq) at τ = 0.10, 0.50, 0.90
#   4. Rolling 90-day coefficient stability
#   5. Chow test for structural breaks
#
# Dependencies: tidyverse, MASS, quantreg, car, jsonlite, lubridate, zoo, RSQLite
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(MASS)
  library(quantreg)
  library(car)
  library(jsonlite)
  library(lubridate)
  library(zoo)
  library(RSQLite)
})

# ── Configuration ──────────────────────────────────────────────────────────────

DB_PATH    <- Sys.getenv("IDEA_ENGINE_DB",   "../db/idea_engine.db")
OUTPUT_DIR <- Sys.getenv("STATS_OUTPUT_DIR", "../stats-service/output")

FACTORS <- c("btc_return", "eth_return", "log_vix",
             "dxy_return", "hour_sin", "hour_cos",
             "day_of_week", "hold_duration_log")

# ── Data Loading ──────────────────────────────────────────────────────────────

load_factor_data <- function(db_path = DB_PATH) {
  if (!file.exists(db_path)) {
    message("[factor_model] DB not found — generating synthetic factor data")
    set.seed(123)
    n <- 500

    hours <- sample(0:23, n, replace = TRUE)
    dow   <- sample(0:6, n, replace = TRUE)
    hold  <- rexp(n, rate = 1/4) + 0.1    # hold duration in hours

    btc_r <- rnorm(n, 0.0002, 0.025)
    eth_r <- btc_r * 0.88 + rnorm(n, 0, 0.015)
    vix   <- pmax(rnorm(n, 20, 8), 5)
    dxy_r <- rnorm(n, 0, 0.003)

    # Strategy return: driven by BTC, penalised in high vol, hour effects
    hour_effect <- ifelse(hours %in% c(14:17), 0.002, 0)
    strat_r     <- 0.001 +
                   0.35 * btc_r +
                   -0.05 * log(vix) +
                   hour_effect +
                   0.0005 * (dow %in% c(1, 3)) +
                   rnorm(n, 0, 0.008)

    tibble(
      strategy_return    = strat_r,
      btc_return         = btc_r,
      eth_return         = eth_r,
      log_vix            = log(vix),
      dxy_return         = dxy_r,
      hour_sin           = sin(2 * pi * hours / 24),
      hour_cos           = cos(2 * pi * hours / 24),
      day_of_week        = as.numeric(dow),
      hold_duration_log  = log(hold + 1),
      timestamp          = Sys.time() - rev(seq_len(n)) * 3600
    )
  } else {
    con <- dbConnect(SQLite(), db_path)
    on.exit(dbDisconnect(con))
    df  <- dbGetQuery(con, "
      SELECT
        pnl_pct                       AS strategy_return,
        btc_return_at_entry           AS btc_return,
        eth_return_at_entry           AS eth_return,
        log_vix                       AS log_vix,
        dxy_return                    AS dxy_return,
        CAST(strftime('%H', opened_at) AS REAL) AS hour,
        CAST(strftime('%w', opened_at) AS REAL) AS day_of_week,
        (julianday(closed_at) - julianday(opened_at)) * 24 AS hold_hours
      FROM trades
      WHERE closed_at IS NOT NULL
        AND btc_return_at_entry IS NOT NULL
      ORDER BY opened_at ASC
    ")
    df %>%
      mutate(
        hour_sin          = sin(2 * pi * hour / 24),
        hour_cos          = cos(2 * pi * hour / 24),
        hold_duration_log = log(pmax(hold_hours, 0.01) + 1),
        eth_return        = replace_na(eth_return, 0),
        log_vix           = replace_na(log_vix, log(20)),
        dxy_return        = replace_na(dxy_return, 0)
      ) %>%
      select(strategy_return, btc_return, eth_return, log_vix,
             dxy_return, hour_sin, hour_cos, day_of_week,
             hold_duration_log) %>%
      as_tibble()
  }
}

# ── Factor Importance ─────────────────────────────────────────────────────────

compute_factor_importance <- function(model_lm) {
  # Standardised coefficients as importance metric
  coefs    <- coef(model_lm)[-1]   # drop intercept
  vcov_mat <- vcov(model_lm)
  se       <- sqrt(diag(vcov_mat))[-1]
  t_stats  <- coefs / se

  tibble(
    factor     = names(coefs),
    coef       = coefs,
    std_error  = se,
    t_stat     = t_stats,
    p_value    = 2 * pt(abs(t_stats), df = df.residual(model_lm), lower.tail = FALSE),
    significant = abs(t_stats) > 2.0
  ) %>%
    arrange(desc(abs(t_stat)))
}

# ── Rolling Coefficient Estimation ────────────────────────────────────────────

rolling_coefficients <- function(df, window = 90, step = 5) {
  n       <- nrow(df)
  starts  <- seq(1, n - window + 1, by = step)

  formula_str <- paste("strategy_return ~", paste(FACTORS, collapse = " + "))
  fml         <- as.formula(formula_str)

  rolling_res <- map_dfr(starts, function(t_start) {
    t_end   <- t_start + window - 1
    sub_df  <- df[t_start:t_end, ]

    tryCatch({
      m       <- lm(fml, data = sub_df)
      coefs   <- coef(m)
      r2      <- summary(m)$r.squared
      alpha   <- coefs["(Intercept)"]

      tibble(
        window_start = t_start,
        window_end   = t_end,
        alpha        = alpha,
        r_squared    = r2,
        !!!setNames(as.list(coefs[FACTORS]),
                    paste0("coef_", FACTORS))
      )
    }, error = function(e) {
      tibble(window_start = t_start, window_end = t_end,
             alpha = NA_real_, r_squared = NA_real_)
    })
  })

  rolling_res
}

# ── Chow Structural Stability Test ────────────────────────────────────────────

chow_test <- function(df, break_point = NULL) {
  n <- nrow(df)
  if (is.null(break_point)) {
    break_point <- n %/% 2
  }

  formula_str <- paste("strategy_return ~", paste(FACTORS, collapse = " + "))
  fml         <- as.formula(formula_str)

  df1 <- df[1:break_point, ]
  df2 <- df[(break_point + 1):n, ]

  m_full <- lm(fml, data = df)
  m1     <- lm(fml, data = df1)
  m2     <- lm(fml, data = df2)

  k     <- length(coef(m_full))
  RSS_full <- sum(resid(m_full)^2)
  RSS_r    <- sum(resid(m1)^2) + sum(resid(m2)^2)

  F_stat  <- ((RSS_full - RSS_r) / k) / (RSS_r / (n - 2 * k))
  p_value <- pf(F_stat, df1 = k, df2 = n - 2 * k, lower.tail = FALSE)

  list(
    F_statistic   = F_stat,
    p_value       = p_value,
    break_point   = break_point,
    k             = k,
    significant   = p_value < 0.05,
    interpretation = if (p_value < 0.05)
      paste0("Structural break detected at t=", break_point, " (p=", round(p_value, 4), ")")
    else
      paste0("No significant structural break at t=", break_point, " (p=", round(p_value, 4), ")")
  )
}

# ── Main Pipeline ─────────────────────────────────────────────────────────────

run_factor_model <- function(
  db_path    = DB_PATH,
  output_dir = OUTPUT_DIR
) {
  message("[factor_model] Loading data...")
  df <- load_factor_data(db_path)
  n  <- nrow(df)
  message("[factor_model] ", n, " trade observations loaded")

  if (n < 20) stop("[factor_model] Insufficient data (need ≥ 20)")

  formula_str <- paste("strategy_return ~", paste(FACTORS, collapse = " + "))
  fml         <- as.formula(formula_str)

  # ── OLS ──
  message("[factor_model] Fitting OLS regression...")
  m_ols  <- lm(fml, data = df)
  s_ols  <- summary(m_ols)

  ols_coefs <- broom::tidy(m_ols) %>%
    rename(std_error = std.error, t_statistic = statistic)

  # ── Robust regression (rlm) ──
  message("[factor_model] Fitting robust regression (IRLS / Huber)...")
  m_rob  <- rlm(fml, data = df, method = "MM", maxit = 100)
  rob_coefs <- coef(m_rob)

  # ── Quantile regression ──
  message("[factor_model] Fitting quantile regression (τ = 0.10, 0.50, 0.90)...")
  m_q10 <- rq(fml, data = df, tau = 0.10)
  m_q50 <- rq(fml, data = df, tau = 0.50)
  m_q90 <- rq(fml, data = df, tau = 0.90)

  qr_coefs <- tibble(
    factor      = names(coef(m_q50)),
    coef_q10    = coef(m_q10),
    coef_q50    = coef(m_q50),
    coef_q90    = coef(m_q90),
    spread      = coef(m_q90) - coef(m_q10)   # asymmetry in factor effect
  )

  # ── Factor importance ──
  message("[factor_model] Computing factor importance...")
  importance <- compute_factor_importance(m_ols)

  # ── Rolling coefficients ──
  win_size <- min(90, n %/% 3)
  message("[factor_model] Computing rolling ", win_size, "-day coefficients...")
  rolling  <- rolling_coefficients(df, window = win_size, step = max(1, win_size %/% 10))

  alpha_stability <- list(
    mean_alpha    = mean(rolling$alpha, na.rm = TRUE),
    sd_alpha      = sd(rolling$alpha, na.rm = TRUE),
    pct_positive  = mean(rolling$alpha > 0, na.rm = TRUE)
  )

  # ── Chow test ──
  message("[factor_model] Running Chow structural stability test...")
  chow <- chow_test(df, break_point = n %/% 2)

  # ── Build output ──
  result <- list(
    n_obs            = n,
    n_factors        = length(FACTORS),
    factors_used     = FACTORS,
    ols = list(
      r_squared      = s_ols$r.squared,
      adj_r_squared  = s_ols$adj.r.squared,
      f_statistic    = s_ols$fstatistic[["value"]],
      coefficients   = ols_coefs %>% select(-any_of("term")) %>%
        mutate(factor = ols_coefs$term) %>%
        as.list() %>% purrr::transpose()
    ),
    robust = list(
      coefficients  = as.list(rob_coefs),
      method        = "MM-estimation (Tukey bisquare)"
    ),
    quantile = list(
      tau_10 = as.list(coef(m_q10)),
      tau_50 = as.list(coef(m_q50)),
      tau_90 = as.list(coef(m_q90)),
      spread = as.list(qr_coefs$spread) %>% setNames(qr_coefs$factor)
    ),
    factor_importance = importance %>%
      select(factor, coef, t_stat, p_value, significant) %>%
      as.list() %>% purrr::transpose(),
    rolling_stability = list(
      window_size         = win_size,
      n_windows           = nrow(rolling),
      mean_alpha_daily    = alpha_stability$mean_alpha,
      sd_alpha_daily      = alpha_stability$sd_alpha,
      pct_windows_pos_alpha = alpha_stability$pct_positive,
      rolling_alpha       = rolling$alpha,
      rolling_r2          = rolling$r_squared
    ),
    chow_test = chow,
    top_factor = importance$factor[1],
    most_asymmetric_factor = qr_coefs$factor[which.max(abs(qr_coefs$spread))]
  )

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(output_dir, "factor_model_results.json")
  write_json(result, out_path, auto_unbox = TRUE, pretty = TRUE)
  message("[factor_model] Results written to ", out_path)

  invisible(result)
}

# ── CLI entry ─────────────────────────────────────────────────────────────────

if (!interactive()) {
  result <- run_factor_model()

  cat("\n=== Factor Model Summary ===\n")
  cat("Observations: ", result$n_obs, "\n")
  cat("OLS R²:       ", round(result$ols$r_squared, 4), "\n")
  cat("OLS adj-R²:   ", round(result$ols$adj_r_squared, 4), "\n")
  cat("Top factor:   ", result$top_factor, "\n")
  cat("Most asymmetric factor (high vs low quantile): ", result$most_asymmetric_factor, "\n")
  cat("\nAlpha stability:\n")
  cat("  Mean daily alpha:   ", round(result$rolling_stability$mean_alpha_daily * 100, 4), "%\n")
  cat("  % windows +alpha:  ", round(result$rolling_stability$pct_windows_pos_alpha * 100, 1), "%\n")
  cat("\nChow test: ", result$chow_test$interpretation, "\n")
}
