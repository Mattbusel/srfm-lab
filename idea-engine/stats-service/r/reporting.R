# =============================================================================
# reporting.R — Tearsheet and Stats Report Generation
# =============================================================================
# Generates quantstrat-style performance tearsheets and saves JSON reports
# to the stats_reports table in idea_engine.db.
#
# Dependencies: RSQLite, PerformanceAnalytics, tidyverse, jsonlite,
#               lubridate, xts
# =============================================================================

suppressPackageStartupMessages({
  library(RSQLite)
  library(PerformanceAnalytics)
  library(tidyverse)
  library(jsonlite)
  library(lubridate)
  library(xts)
})

DB_PATH    <- Sys.getenv("IDEA_ENGINE_DB",   "../db/idea_engine.db")
OUTPUT_DIR <- Sys.getenv("STATS_OUTPUT_DIR", "../stats-service/output")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

open_db <- function() {
  if (!file.exists(DB_PATH)) stop(sprintf("DB not found: %s", DB_PATH))
  dbConnect(SQLite(), DB_PATH)
}

write_output <- function(data, name) {
  path <- file.path(OUTPUT_DIR, paste0(name, ".json"))
  write_json(data, path, auto_unbox = TRUE, pretty = TRUE)
  message(sprintf("[report] Wrote %s", path))
  invisible(path)
}

#' Persist a stats report to the stats_reports table
save_report_to_db <- function(run_id, report_type, content) {
  con <- tryCatch(open_db(), error = function(e) NULL)
  if (is.null(con)) {
    message("[report] DB unavailable; skipping persistence")
    return(invisible(NULL))
  }
  on.exit(dbDisconnect(con))

  content_json <- toJSON(content, auto_unbox = TRUE)

  dbExecute(
    con,
    "INSERT INTO stats_reports (run_id, report_type, content_json) VALUES (?, ?, ?)",
    list(run_id, report_type, content_json)
  )
  message(sprintf("[report] Saved %s report for run %s to DB", report_type, run_id))
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Core metrics helpers
# ---------------------------------------------------------------------------

.ann_return <- function(r, periods = 252L) {
  (prod(1 + r, na.rm = TRUE))^(periods / length(r)) - 1
}

.ann_vol <- function(r, periods = 252L) {
  sd(r, na.rm = TRUE) * sqrt(periods)
}

.sharpe <- function(r, rf = 0, periods = 252L) {
  excess <- r - rf / periods
  mu     <- mean(excess, na.rm = TRUE)
  sig    <- sd(excess,  na.rm = TRUE)
  if (is.na(sig) || sig == 0) return(NA_real_)
  (mu / sig) * sqrt(periods)
}

.sortino <- function(r, mar = 0, periods = 252L) {
  excess      <- r - mar / periods
  downside_r  <- excess[excess < 0]
  if (length(downside_r) == 0) return(NA_real_)
  dd_vol      <- sqrt(mean(downside_r^2, na.rm = TRUE)) * sqrt(periods)
  if (dd_vol == 0) return(NA_real_)
  mean(excess, na.rm = TRUE) * periods / dd_vol
}

.max_drawdown <- function(equity) {
  pk <- cummax(equity)
  dd <- (equity - pk) / pk
  min(dd, na.rm = TRUE)
}

.calmar <- function(r, periods = 252L) {
  ann_ret <- .ann_return(r, periods)
  eq      <- cumprod(1 + r)
  max_dd  <- abs(.max_drawdown(eq))
  if (max_dd == 0) return(NA_real_)
  ann_ret / max_dd
}

.omega_ratio <- function(r, threshold = 0) {
  gains  <- r[r > threshold] - threshold
  losses <- threshold - r[r < threshold]
  if (sum(losses) == 0) return(NA_real_)
  sum(gains) / sum(losses)
}

.var <- function(r, level = 0.05) {
  quantile(r, level, na.rm = TRUE)
}

.cvar <- function(r, level = 0.05) {
  q <- .var(r, level)
  mean(r[r <= q], na.rm = TRUE)
}

.skewness <- function(r) {
  n  <- length(r)
  mu <- mean(r, na.rm = TRUE)
  s  <- sd(r, na.rm = TRUE)
  if (s == 0) return(NA_real_)
  (n / ((n - 1) * (n - 2))) * sum(((r - mu) / s)^3, na.rm = TRUE)
}

.kurtosis <- function(r) {
  n  <- length(r)
  mu <- mean(r, na.rm = TRUE)
  s  <- sd(r, na.rm = TRUE)
  if (s == 0) return(NA_real_)
  m4 <- mean((r - mu)^4, na.rm = TRUE)
  m4 / s^4 - 3   # excess kurtosis
}

.win_rate <- function(r) mean(r > 0, na.rm = TRUE)

.profit_factor <- function(r) {
  gross_profit <- sum(r[r > 0], na.rm = TRUE)
  gross_loss   <- abs(sum(r[r < 0], na.rm = TRUE))
  if (gross_loss == 0) return(NA_real_)
  gross_profit / gross_loss
}

# ---------------------------------------------------------------------------
# Tearsheet
# ---------------------------------------------------------------------------

#' Quantstrat-style performance tearsheet
#'
#' @param equity_curve  Numeric vector or xts of portfolio equity values
#' @param returns       Numeric vector of period returns
#' @param benchmark     Optional numeric vector of benchmark returns
#' @param periods       Trading periods per year (default 252)
#' @return  List: summary_stats, monthly_returns, annual_returns,
#'          drawdown_table, rolling_sharpe
tearsheet <- function(equity_curve, returns, benchmark = NULL, periods = 252L) {
  eq_vec  <- as.numeric(equity_curve)
  ret_vec <- as.numeric(returns)

  stopifnot(
    length(eq_vec)  >= 20,
    length(ret_vec) >= 20
  )

  # ---- Summary statistics ------------------------------------------------
  ann_ret   <- .ann_return(ret_vec, periods)
  ann_vol   <- .ann_vol(ret_vec, periods)
  sharpe    <- .sharpe(ret_vec)
  sortino   <- .sortino(ret_vec)
  calmar    <- .calmar(ret_vec, periods)
  max_dd    <- .max_drawdown(eq_vec)
  omega     <- .omega_ratio(ret_vec)
  var95     <- .var(ret_vec, 0.05)
  cvar95    <- .cvar(ret_vec, 0.05)
  skew      <- .skewness(ret_vec)
  kurt      <- .kurtosis(ret_vec)
  win_rate  <- .win_rate(ret_vec)
  pf        <- .profit_factor(ret_vec)

  # ---- Monthly returns ---------------------------------------------------
  # Assume ret_vec is daily; aggregate to monthly
  n_days     <- length(ret_vec)
  month_idx  <- ceiling(seq_len(n_days) / 21)
  monthly_r  <- tapply(ret_vec, month_idx, function(r) prod(1 + r) - 1)
  monthly_df <- tibble(
    period       = seq_along(monthly_r),
    monthly_ret  = as.numeric(monthly_r)
  ) |>
    mutate(
      pct_return  = monthly_ret * 100,
      positive    = monthly_ret > 0
    )

  # ---- Annual returns ----------------------------------------------------
  year_idx  <- ceiling(seq_len(n_days) / 252)
  annual_r  <- tapply(ret_vec, year_idx, function(r) prod(1 + r) - 1)
  annual_df <- tibble(
    year        = seq_along(annual_r),
    annual_ret  = as.numeric(annual_r)
  )

  # ---- Drawdown table (top 5) --------------------------------------------
  pk        <- cummax(eq_vec)
  dd_series <- (eq_vec - pk) / pk
  in_dd     <- dd_series < -1e-8
  ep_id     <- cumsum(c(FALSE, diff(in_dd) == 1))
  ep_id[!in_dd] <- 0L

  dd_episodes <- tibble(
    t      = seq_len(n_days),
    dd_pct = dd_series,
    ep     = ep_id
  ) |>
    filter(ep > 0L) |>
    group_by(ep) |>
    summarise(
      start    = min(t),
      end      = max(t),
      max_dd   = min(dd_pct),
      duration = n(),
      .groups  = "drop"
    ) |>
    arrange(max_dd) |>
    slice_head(n = 5)

  # ---- Rolling Sharpe (63-day window) ------------------------------------
  window_r       <- 63L
  roll_sharpe    <- numeric(n_days)
  roll_sharpe[]  <- NA_real_
  for (t in seq(window_r, n_days)) {
    roll_sharpe[t] <- .sharpe(ret_vec[(t - window_r + 1):t])
  }

  # ---- Benchmark comparison ----------------------------------------------
  bench_stats <- NULL
  if (!is.null(benchmark)) {
    bench_vec   <- as.numeric(benchmark)
    if (length(bench_vec) == length(ret_vec)) {
      excess_ret   <- ret_vec - bench_vec
      information_ratio <- mean(excess_ret, na.rm = TRUE) /
                           sd(excess_ret, na.rm = TRUE) * sqrt(periods)
      beta         <- cov(ret_vec, bench_vec, use = "complete.obs") /
                      var(bench_vec, na.rm = TRUE)
      alpha        <- mean(ret_vec - beta * bench_vec, na.rm = TRUE) * periods
      bench_stats  <- list(
        information_ratio = information_ratio,
        beta              = beta,
        alpha_annualised  = alpha,
        benchmark_sharpe  = .sharpe(bench_vec)
      )
    }
  }

  result <- list(
    summary = list(
      annualised_return  = ann_ret,
      annualised_vol     = ann_vol,
      sharpe_ratio       = sharpe,
      sortino_ratio      = sortino,
      calmar_ratio       = calmar,
      max_drawdown       = max_dd,
      omega_ratio        = omega,
      var_95             = var95,
      cvar_95            = cvar95,
      skewness           = skew,
      excess_kurtosis    = kurt,
      win_rate           = win_rate,
      profit_factor      = pf,
      n_periods          = n_days
    ),
    monthly_returns  = monthly_df,
    annual_returns   = annual_df,
    drawdown_table   = dd_episodes,
    rolling_sharpe   = roll_sharpe,
    benchmark        = bench_stats
  )

  write_output(result, "tearsheet")
  result
}

# ---------------------------------------------------------------------------
# Full stats report
# ---------------------------------------------------------------------------

#' Generate and persist a complete statistical report for a backtest run
#'
#' Loads equity curve + returns from the database, builds a tearsheet,
#' and saves the report to the stats_reports table.
#'
#' @param run_id      Backtest run identifier
#' @param report_type Label for the report (default "full_tearsheet")
#' @return Invisibly the report list
generate_stats_report <- function(run_id, report_type = "full_tearsheet") {
  stopifnot(is.character(run_id), nchar(run_id) > 0)

  con <- tryCatch(open_db(), error = function(e) {
    message("[report] Cannot open DB: ", e$message)
    NULL
  })

  if (is.null(con)) {
    message("[report] DB unavailable; generating synthetic demo report")
    set.seed(42L)
    demo_rets  <- rnorm(504, 0.0004, 0.012)
    demo_eq    <- cumprod(1 + demo_rets) * 10000
    report     <- tearsheet(demo_eq, demo_rets)
    report$run_id     <- run_id
    report$report_type <- report_type
    report$generated_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ")
    write_output(report, sprintf("report_%s", run_id))
    return(invisible(report))
  }

  on.exit(dbDisconnect(con))

  # Load equity curve
  eq_raw <- dbGetQuery(
    con,
    sprintf(
      "SELECT ts, equity FROM equity_curves WHERE run_id = '%s' ORDER BY ts",
      run_id
    )
  )

  if (nrow(eq_raw) == 0) {
    message(sprintf("[report] No equity data for run_id=%s", run_id))
    dbDisconnect(con)
    on.exit(NULL)
    return(invisible(NULL))
  }

  eq_vec  <- eq_raw$equity
  ret_vec <- diff(log(eq_vec))
  ret_vec <- ret_vec[!is.na(ret_vec)]

  # Load benchmark if available
  bench_raw <- tryCatch(
    dbGetQuery(con, "SELECT ts, close FROM benchmark_prices ORDER BY ts"),
    error = function(e) NULL
  )
  benchmark <- if (!is.null(bench_raw) && nrow(bench_raw) == nrow(eq_raw)) {
    diff(log(bench_raw$close))
  } else {
    NULL
  }

  # Build tearsheet
  report <- tearsheet(eq_vec, ret_vec, benchmark = benchmark)

  report$run_id      <- run_id
  report$report_type <- report_type
  report$generated_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ")

  # Persist to DB
  save_report_to_db(run_id, report_type, report)

  # Also write JSON file
  write_output(report, sprintf("report_%s", run_id))

  message(sprintf("[report] Generated %s report for run %s", report_type, run_id))
  invisible(report)
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main <- function() {
  args    <- commandArgs(trailingOnly = TRUE)
  run_id  <- if (length(args) >= 1) args[1] else "DEMO"

  message(sprintf("[report] Generating stats report for run_id=%s", run_id))
  generate_stats_report(run_id)
  message("[report] Done")
}

if (!interactive()) {
  main()
}
