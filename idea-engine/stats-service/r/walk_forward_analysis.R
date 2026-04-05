# =============================================================================
# walk_forward_analysis.R — Walk-Forward Analysis & Kelly Criterion
# =============================================================================
# Proper walk-forward testing: splits historical data into rolling in-sample /
# out-of-sample windows, optimises in-sample, evaluates out-of-sample.
#
# Dependencies: RSQLite, tidyverse, jsonlite, lubridate, zoo
# =============================================================================

suppressPackageStartupMessages({
  library(RSQLite)
  library(tidyverse)
  library(jsonlite)
  library(lubridate)
  library(zoo)
})

DB_PATH    <- Sys.getenv("IDEA_ENGINE_DB",   "../db/idea_engine.db")
OUTPUT_DIR <- Sys.getenv("STATS_OUTPUT_DIR", "../stats-service/output")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

write_output <- function(data, name) {
  path <- file.path(OUTPUT_DIR, paste0(name, ".json"))
  write_json(data, path, auto_unbox = TRUE, pretty = TRUE)
  message(sprintf("[wfa] Wrote %s", path))
  invisible(path)
}

#' Annualised Sharpe ratio from a return vector
.sharpe <- function(r, periods = 252L) {
  mu  <- mean(r, na.rm = TRUE)
  sig <- sd(r,   na.rm = TRUE)
  if (is.na(sig) || sig == 0) return(NA_real_)
  (mu / sig) * sqrt(periods)
}

#' Calmar ratio: annualised return / max drawdown (absolute)
.calmar <- function(r, periods = 252L) {
  ann_ret <- mean(r, na.rm = TRUE) * periods
  eq      <- cumprod(1 + r)
  pk      <- cummax(eq)
  max_dd  <- abs(min((eq - pk) / pk, na.rm = TRUE))
  if (max_dd == 0) return(NA_real_)
  ann_ret / max_dd
}

# ---------------------------------------------------------------------------
# Walk-forward test
# ---------------------------------------------------------------------------

#' Proper anchored walk-forward test
#'
#' Splits a return series into rolling folds.  For each fold the function calls
#' an optional optimiser on the in-sample window (or simply evaluates a fixed
#' parameter set) and evaluates it on the subsequent out-of-sample window.
#'
#' @param returns            Numeric vector of daily returns (chronological)
#' @param params             Named list of parameter grid or fixed params
#' @param in_sample_months   Months of in-sample data per fold (default 12)
#' @param out_sample_months  Months of out-of-sample data per fold (default 3)
#' @param anchored           If TRUE use expanding in-sample window; else rolling
#' @param periods_per_year   Trading periods per year (default 252)
#' @return List: folds tibble, oos_sharpe vector, oos_calmar vector,
#'         combined_oos_sharpe, wfa_efficiency
walk_forward_test <- function(returns,
                               params,
                               in_sample_months  = 12L,
                               out_sample_months = 3L,
                               anchored          = FALSE,
                               periods_per_year  = 252L) {
  stopifnot(
    is.numeric(returns),
    length(returns) >= (in_sample_months + out_sample_months) * 21
  )

  n          <- length(returns)
  is_bars    <- round(in_sample_months  * 21)   # ~21 trading days / month
  oos_bars   <- round(out_sample_months * 21)
  step       <- oos_bars                         # non-overlapping OOS windows

  folds      <- list()
  fold_idx   <- 0L
  oos_start  <- is_bars + 1L

  while ((oos_start + oos_bars - 1L) <= n) {
    fold_idx <- fold_idx + 1L

    is_end   <- oos_start - 1L
    is_start <- if (anchored) 1L else max(1L, is_end - is_bars + 1L)
    oos_end  <- oos_start + oos_bars - 1L

    is_rets  <- returns[is_start:is_end]
    oos_rets <- returns[oos_start:oos_end]

    # In-sample: find best param (Sharpe) from params grid
    best_param  <- NULL
    best_is_sh  <- -Inf

    if (is.list(params) && all(sapply(params, is.numeric))) {
      # Treat as grid: each element is a vector of candidate values
      grid_df   <- expand.grid(params, stringsAsFactors = FALSE)
      for (row_i in seq_len(nrow(grid_df))) {
        # Placeholder: score = Sharpe of IS returns weighted by row params
        # In a real deployment this would call the backtest engine
        sh <- .sharpe(is_rets)
        if (!is.na(sh) && sh > best_is_sh) {
          best_is_sh  <- sh
          best_param  <- as.list(grid_df[row_i, , drop = FALSE])
        }
      }
    } else {
      # Fixed params: just evaluate
      best_param <- params
      best_is_sh <- .sharpe(is_rets)
    }

    is_calmar  <- .calmar(is_rets)
    oos_sharpe <- .sharpe(oos_rets)
    oos_calmar <- .calmar(oos_rets)

    folds[[fold_idx]] <- list(
      fold          = fold_idx,
      is_start      = is_start,
      is_end        = is_end,
      oos_start     = oos_start,
      oos_end       = oos_end,
      is_sharpe     = best_is_sh,
      is_calmar     = is_calmar,
      oos_sharpe    = oos_sharpe,
      oos_calmar    = oos_calmar,
      best_params   = best_param
    )

    oos_start <- oos_start + step
  }

  folds_df   <- bind_rows(folds)
  oos_rets_all <- returns[folds_df$oos_start[1]:tail(folds_df$oos_end, 1)]

  combined_oos_sharpe <- .sharpe(oos_rets_all)
  mean_is_sharpe      <- mean(folds_df$is_sharpe, na.rm = TRUE)

  wfa_efficiency <- if (!is.na(mean_is_sharpe) && mean_is_sharpe != 0) {
    combined_oos_sharpe / mean_is_sharpe
  } else {
    NA_real_
  }

  result <- list(
    folds               = folds_df,
    n_folds             = nrow(folds_df),
    combined_oos_sharpe = combined_oos_sharpe,
    mean_is_sharpe      = mean_is_sharpe,
    wfa_efficiency      = wfa_efficiency,
    in_sample_months    = in_sample_months,
    out_sample_months   = out_sample_months,
    anchored            = anchored
  )

  write_output(result, "walk_forward_test")
  result
}

# ---------------------------------------------------------------------------
# Parameter stability
# ---------------------------------------------------------------------------

#' Measure whether optimal parameters are stable across WFA folds
#'
#' @param results_list  Output of walk_forward_test (list with $folds)
#' @return List: stability_score (0-1), param_cv tibble (CV per param),
#'         is_stable logical
parameter_stability <- function(results_list) {
  folds <- results_list$folds

  if (is.null(folds) || nrow(folds) == 0) {
    stop("results_list must contain a non-empty $folds data frame")
  }

  # Extract best_params from each fold into a data.frame
  param_df <- map_dfr(
    seq_len(nrow(folds)),
    function(i) {
      bp <- folds$best_params[[i]]
      if (is.null(bp)) return(NULL)
      as_tibble(bp)
    }
  )

  if (ncol(param_df) == 0 || nrow(param_df) == 0) {
    return(list(
      stability_score = NA_real_,
      param_cv        = tibble(),
      is_stable       = NA
    ))
  }

  # Coefficient of variation per numeric parameter
  param_cv <- param_df |>
    summarise(across(
      where(is.numeric),
      list(
        mean = ~ mean(.x, na.rm = TRUE),
        sd   = ~   sd(.x, na.rm = TRUE),
        cv   = ~ {
          mu <- mean(.x, na.rm = TRUE)
          if (mu == 0) NA_real_ else sd(.x, na.rm = TRUE) / abs(mu)
        }
      ),
      .names = "{.col}__{.fn}"
    )) |>
    pivot_longer(everything(), names_to = "metric", values_to = "value") |>
    separate(metric, into = c("param", "stat"), sep = "__") |>
    pivot_wider(names_from = stat, values_from = value)

  cv_vals         <- param_cv$cv
  mean_cv         <- mean(cv_vals, na.rm = TRUE)
  stability_score <- max(0, 1 - mean_cv)    # 1 = perfectly stable, 0 = chaotic

  result <- list(
    stability_score = stability_score,
    param_cv        = param_cv,
    is_stable       = stability_score > 0.6,
    mean_cv         = mean_cv
  )

  write_output(result, "parameter_stability")
  result
}

# ---------------------------------------------------------------------------
# Efficiency ratio
# ---------------------------------------------------------------------------

#' How much of the in-sample edge survives out-of-sample?
#'
#' @param in_sample_sharpe   Numeric scalar or vector of IS Sharpe ratios
#' @param out_sample_sharpe  Numeric scalar or vector of OOS Sharpe ratios
#' @return List: efficiency (0-1), interpretation string
efficiency_ratio <- function(in_sample_sharpe, out_sample_sharpe) {
  stopifnot(
    is.numeric(in_sample_sharpe),
    is.numeric(out_sample_sharpe),
    length(in_sample_sharpe) == length(out_sample_sharpe)
  )

  valid   <- !is.na(in_sample_sharpe) & !is.na(out_sample_sharpe) &
             in_sample_sharpe != 0

  ratios  <- out_sample_sharpe[valid] / in_sample_sharpe[valid]
  eff     <- mean(ratios, na.rm = TRUE)

  interpretation <- dplyr::case_when(
    eff >= 0.75 ~ "EXCELLENT: most IS edge transfers OOS",
    eff >= 0.50 ~ "GOOD: substantial IS edge transfers OOS",
    eff >= 0.25 ~ "MARGINAL: some IS edge transfers OOS",
    eff >= 0.00 ~ "POOR: little IS edge transfers OOS",
    TRUE        ~ "NEGATIVE: strategy performs worse OOS"
  )

  result <- list(
    efficiency       = eff,
    fold_ratios      = ratios,
    interpretation   = interpretation,
    n_folds          = sum(valid)
  )

  write_output(result, "efficiency_ratio")
  result
}

# ---------------------------------------------------------------------------
# Optimal f / Kelly criterion
# ---------------------------------------------------------------------------

#' Kelly criterion and Optimal f for position sizing
#'
#' Computes:
#'  - Classical Kelly fraction (win-rate based)
#'  - Generalised Kelly (expected log growth maximisation)
#'  - Half-Kelly (conservative practical recommendation)
#'  - Optimal f (Vince method: maximises TWR)
#'
#' @param returns  Numeric vector of trade returns (as fraction of capital)
#' @return List: kelly_fraction, half_kelly, optimal_f, max_twr_at_f
optimal_f <- function(returns) {
  stopifnot(is.numeric(returns), length(returns) >= 10)

  # Remove zero returns
  rets     <- returns[returns != 0]

  wins     <- rets[rets > 0]
  losses   <- rets[rets < 0]

  p_win    <- length(wins)  / length(rets)
  p_loss   <- length(losses) / length(rets)

  avg_win  <- if (length(wins)   > 0) mean(wins)         else 0
  avg_loss <- if (length(losses) > 0) abs(mean(losses))  else 1e-8

  # Classical Kelly: K = p - q/b  where b = avg_win/avg_loss
  b             <- avg_win / avg_loss
  kelly         <- p_win - p_loss / b

  # Generalised Kelly via log-growth optimisation
  obj_log_growth <- function(f) {
    g <- mean(log(1 + f * rets), na.rm = TRUE)
    -g  # minimise negative
  }

  opt <- tryCatch(
    optimise(obj_log_growth, c(0, 1), maximum = FALSE),
    error = function(e) list(minimum = NA_real_, objective = NA_real_)
  )

  gen_kelly <- opt$minimum

  # Optimal f (Vince): TWR = prod(1 + f * (-R / largest_loss))
  largest_loss <- abs(min(rets))

  twr_fn <- function(f) {
    scaled_rets <- f * (-rets / largest_loss)
    twr         <- prod(1 + scaled_rets, na.rm = TRUE)
    -twr  # minimise negative
  }

  opt_f <- tryCatch(
    optimise(twr_fn, c(0, 1), maximum = FALSE),
    error = function(e) list(minimum = NA_real_, objective = NA_real_)
  )

  result <- list(
    kelly_fraction    = max(0, kelly),
    half_kelly        = max(0, kelly / 2),
    generalised_kelly = gen_kelly,
    optimal_f         = opt_f$minimum,
    max_twr           = if (!is.na(opt_f$minimum)) -opt_f$objective else NA_real_,
    p_win             = p_win,
    avg_win           = avg_win,
    avg_loss          = avg_loss,
    payoff_ratio      = b,
    n_trades          = length(rets)
  )

  write_output(result, "optimal_f")
  result
}

# ---------------------------------------------------------------------------
# Monte Carlo dominance
# ---------------------------------------------------------------------------

#' Test whether strategy A stochastically dominates strategy B
#'
#' Uses Monte Carlo resampling: repeatedly draw synthetic equity curves and
#' check whether A's Sharpe consistently beats B's.  Reports the fraction of
#' sims where A wins (first-order dominance proxy).
#'
#' @param strategy_a  Numeric vector of daily returns for strategy A
#' @param strategy_b  Numeric vector of daily returns for strategy B
#' @param n_sims      Number of Monte Carlo simulations
#' @return List: p_a_dominates (fraction A wins), sharpe_diff_distribution,
#'         verdict
monte_carlo_dominance <- function(strategy_a, strategy_b, n_sims = 10000L) {
  stopifnot(
    is.numeric(strategy_a), length(strategy_a) >= 20,
    is.numeric(strategy_b), length(strategy_b) >= 20
  )

  n_a   <- length(strategy_a)
  n_b   <- length(strategy_b)
  n_sim <- min(n_a, n_b)   # draw equal-length samples

  sharpe_diffs <- numeric(n_sims)
  set.seed(123L)

  for (i in seq_len(n_sims)) {
    sa_boot   <- sample(strategy_a, n_sim, replace = TRUE)
    sb_boot   <- sample(strategy_b, n_sim, replace = TRUE)
    sharpe_diffs[i] <- .sharpe(sa_boot) - .sharpe(sb_boot)
  }

  p_a_wins  <- mean(sharpe_diffs > 0, na.rm = TRUE)
  mean_diff <- mean(sharpe_diffs, na.rm = TRUE)

  verdict <- dplyr::case_when(
    p_a_wins >= 0.95 ~ "A STRONGLY DOMINATES B",
    p_a_wins >= 0.75 ~ "A WEAKLY DOMINATES B",
    p_a_wins >= 0.50 ~ "A MARGINALLY BETTER THAN B",
    p_a_wins >= 0.25 ~ "B MARGINALLY BETTER THAN A",
    p_a_wins >= 0.05 ~ "B WEAKLY DOMINATES A",
    TRUE             ~ "B STRONGLY DOMINATES A"
  )

  result <- list(
    p_a_dominates          = p_a_wins,
    p_b_dominates          = 1 - p_a_wins,
    mean_sharpe_diff       = mean_diff,
    sharpe_diff_q05        = quantile(sharpe_diffs, 0.05),
    sharpe_diff_q95        = quantile(sharpe_diffs, 0.95),
    n_sims                 = n_sims,
    verdict                = verdict
  )

  write_output(result, "monte_carlo_dominance")
  result
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main <- function() {
  args    <- commandArgs(trailingOnly = TRUE)
  run_id  <- if (length(args) >= 1) args[1] else NULL

  message(sprintf("[wfa] Starting walk-forward analysis for run_id=%s",
                  run_id %||% "ALL"))

  con <- tryCatch(
    dbConnect(SQLite(), DB_PATH),
    error = function(e) { message("[wfa] DB unavailable: ", e$message); NULL }
  )
  if (is.null(con)) quit(status = 0)

  query <- "SELECT run_id, ts, equity FROM equity_curves ORDER BY run_id, ts"
  if (!is.null(run_id)) {
    query <- paste0(query, sprintf(" WHERE run_id = '%s'", run_id))
  }

  raw  <- dbGetQuery(con, query)
  dbDisconnect(con)

  if (nrow(raw) == 0) {
    message("[wfa] No equity data; exiting")
    quit(status = 0)
  }

  # Use first run
  first_run  <- raw |>
    filter(run_id == raw$run_id[1]) |>
    arrange(ts)

  eq_vals    <- first_run$equity
  rets       <- diff(log(eq_vals))
  rets       <- rets[!is.na(rets)]

  if (length(rets) < 300) {
    message("[wfa] Insufficient data for WFA (need ≥300 bars)")
    quit(status = 0)
  }

  wfa_result   <- walk_forward_test(rets, params = list())
  param_stab   <- parameter_stability(wfa_result)
  eff          <- efficiency_ratio(
    wfa_result$folds$is_sharpe,
    wfa_result$folds$oos_sharpe
  )
  kel          <- optimal_f(rets)

  message("[wfa] All walk-forward analyses complete")
}

if (!interactive()) {
  main()
}
