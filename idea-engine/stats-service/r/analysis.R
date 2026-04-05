# =============================================================================
# analysis.R — Core Statistical Analysis for the Idea Automation Engine
# =============================================================================
# Reads simulation results from idea_engine.db (RSQLite), runs a battery of
# quantitative tests, and writes JSON output to stats-service/output/.
#
# Dependencies: RSQLite, depmixS4, tidyverse, PerformanceAnalytics, boot, car,
#               jsonlite, lubridate, zoo, xts
# =============================================================================

suppressPackageStartupMessages({
  library(RSQLite)
  library(depmixS4)
  library(tidyverse)
  library(PerformanceAnalytics)
  library(boot)
  library(car)
  library(jsonlite)
  library(lubridate)
  library(zoo)
  library(xts)
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH     <- Sys.getenv("IDEA_ENGINE_DB", "../db/idea_engine.db")
OUTPUT_DIR  <- Sys.getenv("STATS_OUTPUT_DIR", "../stats-service/output")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

#' Open a read-only connection to idea_engine.db
#'
#' @return DBI connection object
open_db <- function() {
  if (!file.exists(DB_PATH)) {
    stop(sprintf("Database not found at: %s", DB_PATH))
  }
  dbConnect(SQLite(), DB_PATH)
}

#' Write JSON output file to the output directory
#'
#' @param data  R object to serialise
#' @param name  file base name (without .json)
write_output <- function(data, name) {
  path <- file.path(OUTPUT_DIR, paste0(name, ".json"))
  write_json(data, path, auto_unbox = TRUE, pretty = TRUE)
  message(sprintf("[stats] Wrote %s", path))
  invisible(path)
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

#' Load all backtest simulation results from the database
#'
#' @param run_id  Optional run_id filter (NULL = all runs)
#' @return tibble with columns: run_id, params_json, sharpe, calmar, max_dd,
#'         total_return, n_trades, created_at
load_backtest_results <- function(run_id = NULL) {
  con <- open_db()
  on.exit(dbDisconnect(con))

  query <- "
    SELECT
        run_id,
        params_json,
        sharpe,
        calmar,
        max_drawdown   AS max_dd,
        total_return,
        n_trades,
        created_at
    FROM backtest_results
  "
  if (!is.null(run_id)) {
    query <- paste0(query, sprintf(" WHERE run_id = '%s'", run_id))
  }

  tbl <- dbGetQuery(con, query) |>
    as_tibble() |>
    mutate(
      created_at = ymd_hms(created_at),
      params     = map(params_json, ~ fromJSON(.x))
    )

  message(sprintf("[stats] Loaded %d backtest results", nrow(tbl)))
  tbl
}

#' Load equity time-series for one or more runs
#'
#' @param run_ids  Character vector of run IDs.  NULL = all.
#' @return Named list of xts objects (one per run_id)
load_equity_curves <- function(run_ids = NULL) {
  con <- open_db()
  on.exit(dbDisconnect(con))

  query <- "
    SELECT run_id, ts, equity
    FROM equity_curves
    ORDER BY run_id, ts
  "
  if (!is.null(run_ids)) {
    ids_sql <- paste(sprintf("'%s'", run_ids), collapse = ", ")
    query   <- paste0(query, sprintf(" WHERE run_id IN (%s)", ids_sql))
  }

  raw <- dbGetQuery(con, query) |>
    as_tibble() |>
    mutate(ts = ymd_hms(ts))

  curves <- raw |>
    group_by(run_id) |>
    group_map(function(df, key) {
      x <- xts(df$equity, order.by = df$ts)
      colnames(x) <- "equity"
      x
    }) |>
    set_names(unique(raw$run_id))

  message(sprintf("[stats] Loaded equity curves for %d runs", length(curves)))
  curves
}

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

#' Fit a 3-state Hidden Markov Model on the return series of an equity curve
#'
#' Uses depmixS4 to fit a Gaussian-emission HMM with states labelled by mean
#' return: BEAR (most negative), NEUTRAL (middle), BULL (most positive).
#'
#' @param equity_curves  Named list of xts equity objects (from load_equity_curves)
#' @param n_regimes      Number of HMM states (default 3)
#' @return  List with elements:
#'   - regime_probs: tibble(ts, run_id, bull, bear, neutral, state)
#'   - transition_matrix: n_regimes x n_regimes matrix
#'   - state_stats: tibble of per-state mean/sd return
regime_clustering <- function(equity_curves, n_regimes = 3L) {
  stopifnot(length(equity_curves) >= 1L)

  all_probs <- vector("list", length(equity_curves))

  for (i in seq_along(equity_curves)) {
    rid   <- names(equity_curves)[[i]]
    eq    <- equity_curves[[i]]

    # Daily log returns; drop leading NA
    rets  <- diff(log(eq$equity))
    rets  <- rets[!is.na(rets)]

    if (length(rets) < n_regimes * 10) {
      warning(sprintf("[regime] run %s has too few observations (%d); skipping",
                      rid, length(rets)))
      next
    }

    ret_df <- data.frame(returns = as.numeric(rets))

    # Fit HMM with Gaussian emissions
    mod <- depmixS4::depmix(
      response  = returns ~ 1,
      data      = ret_df,
      nstates   = n_regimes,
      family    = gaussian()
    )

    fit <- tryCatch(
      depmixS4::fit(mod, verbose = FALSE),
      error = function(e) {
        warning(sprintf("[regime] HMM fit failed for run %s: %s", rid, e$message))
        NULL
      }
    )
    if (is.null(fit)) next

    post      <- depmixS4::posterior(fit)
    state_seq <- post$state

    # Extract emission parameters to label states
    params <- depmixS4::getpars(fit)
    # Emission means are stored inside the response models
    means  <- sapply(fit@response, function(r) r[[1]]@parameters$coefficients)

    # Sort states: BEAR = lowest mean, BULL = highest
    rank_order <- order(means)               # rank_order[1] = BEAR state index
    state_labels <- character(n_regimes)
    if (n_regimes == 3L) {
      state_labels[rank_order[1]] <- "BEAR"
      state_labels[rank_order[2]] <- "NEUTRAL"
      state_labels[rank_order[3]] <- "BULL"
    } else {
      state_labels <- paste0("S", seq_len(n_regimes))
    }

    # Posterior state probabilities
    prob_cols <- post[, paste0("S", seq_len(n_regimes)), drop = FALSE]
    colnames(prob_cols) <- state_labels[seq_len(n_regimes)]

    prob_tbl <- as_tibble(prob_cols) |>
      mutate(
        ts     = index(rets),
        run_id = rid,
        state  = state_labels[state_seq]
      )

    all_probs[[i]] <- prob_tbl
  }

  regime_probs <- bind_rows(compact(all_probs))

  # Transition matrix (averaged over all runs, majority-vote)
  trans_list <- lapply(seq_along(equity_curves), function(i) {
    rid  <- names(equity_curves)[[i]]
    sub  <- filter(regime_probs, run_id == rid)
    if (nrow(sub) == 0) return(NULL)
    states <- sub$state
    mat    <- table(head(states, -1), tail(states, -1))
    mat / rowSums(mat)
  })
  trans_list <- compact(trans_list)

  state_stats_list <- lapply(seq_along(equity_curves), function(i) {
    rid <- names(equity_curves)[[i]]
    eq  <- equity_curves[[i]]
    rets <- as.numeric(diff(log(eq$equity)))
    rets <- rets[!is.na(rets)]
    sub  <- filter(regime_probs, run_id == rid)
    if (nrow(sub) == 0 || length(rets) != nrow(sub)) return(NULL)
    tibble(
      run_id = rid,
      state  = sub$state,
      ret    = rets
    ) |>
      group_by(state) |>
      summarise(
        mean_ret = mean(ret),
        sd_ret   = sd(ret),
        n_bars   = n(),
        .groups  = "drop"
      )
  })

  result <- list(
    regime_probs      = regime_probs,
    state_stats       = bind_rows(compact(state_stats_list)),
    n_regimes         = n_regimes
  )

  write_output(result, "regime_clustering")
  result
}

# ---------------------------------------------------------------------------
# Bootstrap Sharpe
# ---------------------------------------------------------------------------

#' Bootstrap confidence interval for the annualised Sharpe ratio
#'
#' @param returns  Numeric vector of period returns (NOT log returns)
#' @param n_boot   Number of bootstrap replicates
#' @param conf     Confidence level (default 0.95)
#' @param periods_per_year  252 for daily, 52 for weekly, 12 for monthly
#' @return List: estimate, ci_lower, ci_upper, se
bootstrap_sharpe <- function(returns, n_boot = 1000L, conf = 0.95,
                              periods_per_year = 252) {
  stopifnot(is.numeric(returns), length(returns) >= 30)

  sharpe_stat <- function(x, idx) {
    r    <- x[idx]
    mu   <- mean(r, na.rm = TRUE)
    sig  <- sd(r, na.rm = TRUE)
    if (sig == 0) return(0)
    (mu / sig) * sqrt(periods_per_year)
  }

  boot_obj <- boot::boot(
    data      = returns,
    statistic = sharpe_stat,
    R         = n_boot,
    sim       = "ordinary"
  )

  ci <- boot::boot.ci(boot_obj, conf = conf, type = "perc")

  result <- list(
    estimate  = boot_obj$t0,
    ci_lower  = ci$percent[4],
    ci_upper  = ci$percent[5],
    se        = sd(boot_obj$t),
    n_boot    = n_boot,
    conf      = conf
  )

  write_output(result, "bootstrap_sharpe")
  result
}

# ---------------------------------------------------------------------------
# Factor attribution
# ---------------------------------------------------------------------------

#' OLS factor model (Fama-French style) for strategy returns
#'
#' @param returns     Numeric vector or xts of strategy returns
#' @param factors_df  data.frame with columns: Mkt.RF, SMB, HML (and optionally
#'                    MOM, QMJ, etc.)  — must be same length as returns
#' @return List: coefficients tibble, r_squared, residuals, alpha_annualised
factor_attribution <- function(returns, factors_df) {
  stopifnot(
    is.numeric(returns) || is.xts(returns),
    is.data.frame(factors_df),
    length(returns) == nrow(factors_df)
  )

  ret_vec <- as.numeric(returns)
  df      <- bind_cols(tibble(strategy = ret_vec), factors_df)

  factor_names <- setdiff(colnames(factors_df), "RF")

  # If RF (risk-free) column present, convert to excess returns
  if ("RF" %in% colnames(factors_df)) {
    df <- df |> mutate(strategy = strategy - RF)
  }

  formula_str <- paste("strategy ~", paste(factor_names, collapse = " + "))
  model <- lm(as.formula(formula_str), data = df)
  sm    <- summary(model)

  coef_tbl <- as_tibble(coef(sm), rownames = "term") |>
    rename(
      estimate  = Estimate,
      std_error = `Std. Error`,
      t_stat    = `t value`,
      p_value   = `Pr(>|t|)`
    )

  alpha_daily      <- coef(model)[["(Intercept)"]]
  alpha_annualised <- (1 + alpha_daily)^252 - 1

  result <- list(
    coefficients      = coef_tbl,
    r_squared         = sm$r.squared,
    adj_r_squared     = sm$adj.r.squared,
    alpha_daily       = alpha_daily,
    alpha_annualised  = alpha_annualised,
    residuals         = as.numeric(residuals(model)),
    f_statistic       = sm$fstatistic[["value"]],
    f_p_value         = pf(sm$fstatistic[1], sm$fstatistic[2],
                            sm$fstatistic[3], lower.tail = FALSE)
  )

  write_output(result, "factor_attribution")
  result
}

# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------

#' Rolling 30-day pairwise correlation analysis across multiple return series
#'
#' @param returns_df  data.frame or tibble where each column is a strategy
#'                    return series (same dates, rows = time)
#' @param window      Rolling window in bars (default 30)
#' @return List: mean_correlation ts, correlation_matrix (last window),
#'         correlation_regime (high/medium/low)
rolling_correlation_analysis <- function(returns_df, window = 30L) {
  stopifnot(is.data.frame(returns_df), nrow(returns_df) > window)

  n_strats <- ncol(returns_df)
  n_bars   <- nrow(returns_df)

  if (n_strats < 2) {
    stop("Need at least 2 strategies for correlation analysis")
  }

  mat <- as.matrix(returns_df)

  # Rolling mean pairwise correlation
  mean_corr <- numeric(n_bars)
  mean_corr[seq_len(window - 1L)] <- NA_real_

  for (t in seq(window, n_bars)) {
    sub    <- mat[(t - window + 1L):t, , drop = FALSE]
    cr     <- cor(sub, use = "pairwise.complete.obs")
    # Upper triangle only (excluding diagonal)
    upper  <- cr[upper.tri(cr)]
    mean_corr[t] <- mean(upper, na.rm = TRUE)
  }

  # Last-window full correlation matrix
  last_window   <- mat[(n_bars - window + 1L):n_bars, , drop = FALSE]
  last_cor_mat  <- cor(last_window, use = "pairwise.complete.obs")

  # Regime: high > 0.7, low < 0.3, medium otherwise
  current_mean  <- tail(na.omit(mean_corr), 1)
  regime        <- dplyr::case_when(
    current_mean > 0.7  ~ "HIGH",
    current_mean < 0.3  ~ "LOW",
    TRUE                ~ "MEDIUM"
  )

  result <- list(
    mean_correlation  = mean_corr,
    last_cor_matrix   = last_cor_mat,
    current_mean      = current_mean,
    regime            = regime,
    window            = window
  )

  write_output(result, "rolling_correlation")
  result
}

# ---------------------------------------------------------------------------
# Drawdown decomposition
# ---------------------------------------------------------------------------

#' Peak-to-trough drawdown decomposition with duration distribution
#'
#' @param equity  Numeric vector or xts of portfolio equity values
#' @return List: drawdowns tibble, summary stats, duration_distribution
drawdown_decomposition <- function(equity) {
  eq_vec  <- as.numeric(equity)
  n       <- length(eq_vec)

  stopifnot(n >= 10)

  # Compute running peak
  peak    <- cummax(eq_vec)
  dd_pct  <- (eq_vec - peak) / peak   # <= 0

  # Identify individual drawdown episodes
  in_dd      <- dd_pct < 0
  # Label contiguous drawdown periods
  dd_id      <- cumsum(c(FALSE, diff(in_dd) == 1))
  dd_id[!in_dd] <- 0L

  episodes <- tibble(
    t         = seq_len(n),
    equity    = eq_vec,
    peak      = peak,
    dd_pct    = dd_pct,
    in_dd     = in_dd,
    episode   = dd_id
  ) |>
    filter(in_dd, episode > 0) |>
    group_by(episode) |>
    summarise(
      start       = min(t),
      end         = max(t),
      duration    = n(),
      max_dd      = min(dd_pct),
      recovery_t  = NA_integer_,   # filled below
      .groups     = "drop"
    )

  # For each episode find when equity next exceeds the episode peak
  for (idx in seq_len(nrow(episodes))) {
    ep_end      <- episodes$end[idx]
    ep_peak_val <- peak[episodes$start[idx]]
    recovery    <- which(eq_vec[(ep_end + 1):n] >= ep_peak_val)
    if (length(recovery) > 0) {
      episodes$recovery_t[idx] <- ep_end + recovery[1]
    }
  }

  episodes <- episodes |>
    mutate(
      recovery_duration = recovery_t - end,
      severity_class    = dplyr::case_when(
        max_dd > -0.05  ~ "MINOR",
        max_dd > -0.15  ~ "MODERATE",
        max_dd > -0.30  ~ "SEVERE",
        TRUE            ~ "CATASTROPHIC"
      )
    )

  summary_stats <- list(
    max_drawdown        = min(dd_pct),
    mean_drawdown       = mean(dd_pct[in_dd], na.rm = TRUE),
    n_episodes          = nrow(episodes),
    mean_duration       = mean(episodes$duration),
    median_duration     = median(episodes$duration),
    mean_recovery       = mean(episodes$recovery_duration, na.rm = TRUE),
    ulcer_index         = sqrt(mean(dd_pct^2))
  )

  duration_dist <- hist(episodes$duration, plot = FALSE, breaks = "Sturges")

  result <- list(
    episodes          = episodes,
    summary           = summary_stats,
    duration_breaks   = duration_dist$breaks,
    duration_counts   = duration_dist$counts,
    drawdown_series   = dd_pct
  )

  write_output(result, "drawdown_decomposition")
  result
}

# ---------------------------------------------------------------------------
# Parameter sensitivity ANOVA
# ---------------------------------------------------------------------------

#' One-way (and interaction) ANOVA on how strategy parameters affect Sharpe
#'
#' @param results_df  tibble from load_backtest_results() with params column
#'                    (list-col) and sharpe column
#' @return List: anova_table tibble, significant_params char vector,
#'         param_effect_sizes tibble
parameter_sensitivity_anova <- function(results_df) {
  stopifnot("sharpe" %in% colnames(results_df))
  stopifnot("params" %in% colnames(results_df))

  # Unnest params list-column into separate columns
  params_wide <- results_df |>
    mutate(params_flat = map(params, ~ as_tibble(as.list(.x)))) |>
    select(sharpe, params_flat) |>
    unnest(params_flat)

  param_cols <- setdiff(colnames(params_wide), "sharpe")

  if (length(param_cols) == 0) {
    stop("No parameter columns found after unnesting")
  }

  anova_results <- map_dfr(param_cols, function(p_col) {
    sub <- params_wide |>
      select(sharpe, all_of(p_col)) |>
      filter(!is.na(.data[[p_col]])) |>
      mutate(param_fct = as.factor(.data[[p_col]]))

    if (n_distinct(sub$param_fct) < 2) return(NULL)

    form  <- as.formula("sharpe ~ param_fct")
    aov_m <- aov(form, data = sub)
    sm    <- summary(aov_m)[[1]]

    # Eta-squared effect size
    ss_effect <- sm["param_fct", "Sum Sq"]
    ss_total  <- sum(sm[, "Sum Sq"], na.rm = TRUE)
    eta_sq    <- ss_effect / ss_total

    tibble(
      param     = p_col,
      f_stat    = sm["param_fct", "F value"],
      p_value   = sm["param_fct", "Pr(>F)"],
      eta_sq    = eta_sq
    )
  })

  sig_params <- anova_results |>
    filter(p_value < 0.05) |>
    arrange(desc(eta_sq)) |>
    pull(param)

  result <- list(
    anova_table        = anova_results,
    significant_params = sig_params,
    n_runs             = nrow(results_df)
  )

  write_output(result, "parameter_sensitivity_anova")
  result
}

# ---------------------------------------------------------------------------
# White's Reality Check
# ---------------------------------------------------------------------------

#' White's Reality Check for data-snooping bias
#'
#' Tests the null that no strategy in a collection beats the benchmark,
#' accounting for the fact that many strategies were tried.
#'
#' Reference: White (2000) "A Reality Check for Data Snooping"
#'
#' @param strategy_returns  Matrix or data.frame; each column is one strategy
#' @param benchmark_returns Numeric vector of benchmark returns (same length)
#' @param n_boot            Bootstrap replications
#' @return List: p_value, max_mean_excess, bootstrap_distribution
white_reality_check <- function(strategy_returns, benchmark_returns,
                                 n_boot = 1000L) {
  strat_mat <- as.matrix(strategy_returns)
  bench_vec <- as.numeric(benchmark_returns)

  stopifnot(nrow(strat_mat) == length(bench_vec))

  n     <- nrow(strat_mat)
  n_s   <- ncol(strat_mat)

  # Excess returns over benchmark
  excess <- sweep(strat_mat, 1, bench_vec, "-")

  # Observed performance measure: mean excess return per strategy
  f_bar <- colMeans(excess, na.rm = TRUE)

  # Test statistic: max of mean excess returns
  V_n   <- sqrt(n) * max(f_bar)

  # Stationary bootstrap
  # Use geometric block lengths with mean block length ~sqrt(n)
  block_len <- max(1L, round(sqrt(n)))

  boot_max <- numeric(n_boot)
  set.seed(42L)

  for (b in seq_len(n_boot)) {
    # Resample blocks
    idx       <- integer(n)
    pos       <- 1L
    while (pos <= n) {
      start   <- sample.int(n, 1L)
      len     <- rgeom(1, 1 / block_len) + 1L
      take    <- ((start - 1L + seq_len(len) - 1L) %% n) + 1L
      take    <- take[seq_len(min(len, n - pos + 1L))]
      idx[pos:(pos + length(take) - 1L)] <- take
      pos     <- pos + length(take)
    }

    boot_excess <- excess[idx, , drop = FALSE]
    # Demeaned to get null distribution
    f_boot      <- colMeans(boot_excess - rep(f_bar, each = nrow(boot_excess)))
    boot_max[b] <- sqrt(n) * max(f_boot)
  }

  p_value <- mean(boot_max >= V_n)

  result <- list(
    p_value              = p_value,
    max_mean_excess      = max(f_bar),
    best_strategy_idx    = which.max(f_bar),
    V_n                  = V_n,
    bootstrap_quantile_95 = quantile(boot_max, 0.95),
    n_strategies         = n_s,
    n_boot               = n_boot
  )

  write_output(result, "white_reality_check")
  result
}

# ---------------------------------------------------------------------------
# Multiple hypothesis correction
# ---------------------------------------------------------------------------

#' Benjamini-Hochberg FDR correction for multiple hypothesis tests
#'
#' @param p_values  Named numeric vector of raw p-values
#' @param alpha     FDR level (default 0.05)
#' @return tibble: hypothesis, raw_p, adjusted_p, reject
multiple_hypothesis_correction <- function(p_values, alpha = 0.05) {
  stopifnot(is.numeric(p_values), all(p_values >= 0 & p_values <= 1))

  nms       <- names(p_values)
  if (is.null(nms)) nms <- paste0("H", seq_along(p_values))

  adj       <- p.adjust(p_values, method = "BH")
  reject    <- adj < alpha

  result_tbl <- tibble(
    hypothesis = nms,
    raw_p      = p_values,
    adjusted_p = adj,
    reject     = reject
  ) |>
    arrange(raw_p)

  result <- list(
    table          = result_tbl,
    n_rejected     = sum(reject),
    n_total        = length(p_values),
    fdr_level      = alpha,
    method         = "Benjamini-Hochberg"
  )

  write_output(result, "multiple_hypothesis_correction")
  result
}

# ---------------------------------------------------------------------------
# Main entry point (called from run_r_analysis.py via Rscript)
# ---------------------------------------------------------------------------

main <- function() {
  args   <- commandArgs(trailingOnly = TRUE)
  run_id <- if (length(args) >= 1) args[1] else NULL

  message(sprintf("[stats] Starting analysis for run_id=%s", run_id %||% "ALL"))

  # Load data
  results <- load_backtest_results(run_id)
  curves  <- load_equity_curves(run_ids = run_id)

  if (nrow(results) == 0) {
    message("[stats] No results found; exiting")
    quit(status = 0)
  }

  # Run all analyses
  if (length(curves) > 0) {
    regime_clustering(curves)

    # Drawdown for first curve
    drawdown_decomposition(curves[[1]]$equity)
  }

  if (nrow(results) >= 30) {
    # Use Sharpe series as a stand-in "returns" for bootstrap
    bootstrap_sharpe(results$sharpe)
  }

  if (nrow(results) >= 10) {
    parameter_sensitivity_anova(results)
  }

  # Rolling correlation requires multiple return series — skip if only 1
  if (length(curves) >= 2) {
    ret_df <- map_dfc(curves, function(x) {
      as.numeric(diff(log(x$equity)))
    })
    rolling_correlation_analysis(ret_df)
  }

  message("[stats] All analyses complete")
}

# Allow both sourcing and direct execution
if (!interactive()) {
  main()
}
