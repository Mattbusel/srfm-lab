# =============================================================================
# bayesian_changepoint.R — Bayesian Changepoint Detection in Strategy P&L
# =============================================================================
# Detects structural breaks in strategy P&L series using:
#   1. PELT algorithm (via changepoint package) for fast detection
#   2. Bayesian posterior probability over changepoint locations
#   3. Regime characterisation: pre/post win rate and P&L statistics
#   4. Narrative output: "Strategy regime changed on DATE: WR X% → Y%"
#
# Dependencies: tidyverse, changepoint, jsonlite, lubridate, zoo, RSQLite
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(changepoint)
  library(jsonlite)
  library(lubridate)
  library(zoo)
  library(RSQLite)
})

# ── Configuration ──────────────────────────────────────────────────────────────

DB_PATH    <- Sys.getenv("IDEA_ENGINE_DB",    "../db/idea_engine.db")
OUTPUT_DIR <- Sys.getenv("STATS_OUTPUT_DIR",  "../stats-service/output")
MIN_POST_PROB <- 0.80   # minimum posterior probability to report a changepoint

# ── Data Loading ──────────────────────────────────────────────────────────────

load_pnl_series <- function(db_path = DB_PATH) {
  if (!file.exists(db_path)) {
    message("[changepoint] DB not found — generating synthetic P&L series for dev")
    set.seed(42)
    n <- 300
    dates <- seq(Sys.Date() - n + 1, Sys.Date(), by = "day")
    # Regime 1 (days 1-150): moderate WR=52%, small positive edge
    # Regime 2 (days 151-300): improved WR=60%, stronger edge (IAE improvement)
    pnl_r1 <- rnorm(150, mean = 0.002, sd = 0.012)
    pnl_r2 <- rnorm(150, mean = 0.008, sd = 0.010)
    win_r1  <- as.integer(pnl_r1 > 0)
    win_r2  <- as.integer(pnl_r2 > 0)
    tibble(
      date       = dates,
      daily_pnl  = c(pnl_r1, pnl_r2),
      win        = c(win_r1, win_r2),
      n_trades   = sample(3:12, n, replace = TRUE)
    )
  } else {
    con <- dbConnect(SQLite(), db_path)
    on.exit(dbDisconnect(con))
    df  <- dbGetQuery(con, "
      SELECT
        date(closed_at) AS date,
        SUM(pnl_pct)    AS daily_pnl,
        SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
        COUNT(*)        AS n_trades
      FROM trades
      WHERE closed_at IS NOT NULL
      GROUP BY date(closed_at)
      ORDER BY date ASC
    ")
    df %>%
      mutate(
        date = as.Date(date),
        win  = wins / n_trades
      ) %>%
      as_tibble()
  }
}

# ── PELT Changepoint Detection ────────────────────────────────────────────────

detect_changepoints_pelt <- function(pnl_series, penalty_value = "MBIC") {
  x <- pnl_series$daily_pnl

  # Mean + variance change detection (PELT)
  cpt_mv <- tryCatch(
    cpt.meanvar(x, method = "PELT", penalty = penalty_value,
                pen.value = ifelse(penalty_value == "Manual", 10, NULL)),
    error = function(e) {
      message("[changepoint] PELT failed: ", e$message)
      NULL
    }
  )

  # Mean-only change detection
  cpt_m <- tryCatch(
    cpt.mean(x, method = "PELT", penalty = penalty_value),
    error = function(e) NULL
  )

  list(
    cpt_meanvar = cpt_mv,
    cpt_mean    = cpt_m,
    n_obs       = length(x)
  )
}

# ── Bayesian Posterior over Changepoint Locations ─────────────────────────────

#' Compute posterior probability of a changepoint at each position.
#'
#' Bayesian approach: for each candidate position t, compute the log-likelihood
#' ratio of (one regime before t, one after) vs (no change).
#' Prior: uniform over candidate positions (1 expected changepoint).
#'
#' @param x         numeric vector (daily P&L)
#' @param min_seg   minimum segment length
#' @return tibble with position, log_lr, posterior_prob
bayesian_changepoint_posterior <- function(x, min_seg = 10) {
  n        <- length(x)
  log_lrs  <- numeric(n)

  # Log-likelihood under single-segment normal model
  ll_full <- function(v) {
    if (length(v) < 2) return(-Inf)
    mu  <- mean(v)
    s2  <- var(v)
    if (s2 <= 0) s2 <- 1e-10
    -length(v) / 2 * log(2 * pi * s2) - sum((v - mu)^2) / (2 * s2)
  }

  ll0 <- ll_full(x)

  for (t in (min_seg + 1):(n - min_seg)) {
    seg1    <- x[1:t]
    seg2    <- x[(t + 1):n]
    ll1     <- ll_full(seg1) + ll_full(seg2)
    log_lrs[t] <- ll1 - ll0
  }

  # Posterior ∝ exp(log_LR) (with uniform prior)
  valid <- (min_seg + 1):(n - min_seg)
  lrs   <- log_lrs[valid]
  # Numerically stable softmax
  lrs_s  <- lrs - max(lrs)
  probs  <- exp(lrs_s) / sum(exp(lrs_s))

  post_df <- tibble(
    position     = valid,
    log_lr       = log_lrs[valid],
    posterior_prob = probs
  )

  post_df
}

# ── Regime Characterisation ───────────────────────────────────────────────────

characterise_regime <- function(segment_df, label) {
  list(
    label        = label,
    n_days       = nrow(segment_df),
    mean_daily_pnl = mean(segment_df$daily_pnl, na.rm = TRUE),
    sd_daily_pnl   = sd(segment_df$daily_pnl, na.rm = TRUE),
    win_rate       = if ("win" %in% names(segment_df))
                       mean(segment_df$win, na.rm = TRUE)
                     else NA_real_,
    total_pnl      = sum(segment_df$daily_pnl, na.rm = TRUE),
    sharpe_daily   = {
      m <- mean(segment_df$daily_pnl, na.rm = TRUE)
      s <- sd(segment_df$daily_pnl, na.rm = TRUE)
      if (!is.na(s) && s > 1e-12) m / s else NA_real_
    },
    start_date     = min(segment_df$date),
    end_date       = max(segment_df$date)
  )
}

# ── Narrative Generation ──────────────────────────────────────────────────────

generate_narrative <- function(regimes, changepoints) {
  narratives <- character(0)

  for (i in seq_along(changepoints)) {
    cp_date <- changepoints[[i]]$date
    pre     <- regimes[[i]]
    post    <- if (i < length(regimes)) regimes[[i + 1]] else NULL

    if (is.null(post)) next

    wr_pre  <- round(pre$win_rate * 100,  1)
    wr_post <- round(post$win_rate * 100, 1)
    pnl_pre  <- round(pre$mean_daily_pnl * 100,  3)
    pnl_post <- round(post$mean_daily_pnl * 100, 3)

    direction <- if (post$mean_daily_pnl > pre$mean_daily_pnl) "IMPROVEMENT" else "DETERIORATION"

    narratives <- c(narratives, paste0(
      direction, " detected at ", cp_date, ": ",
      "pre-change WR=", wr_pre, "%, post-change WR=", wr_post, "% | ",
      "daily PnL: ", pnl_pre, "% → ", pnl_post, "% | ",
      "Sharpe: ", round(pre$sharpe_daily, 3), " → ", round(post$sharpe_daily, 3)
    ))
  }

  if (length(narratives) == 0) {
    narratives <- "No significant regime changes detected at the current threshold."
  }

  narratives
}

# ── Main Pipeline ─────────────────────────────────────────────────────────────

run_bayesian_changepoint <- function(
  db_path    = DB_PATH,
  output_dir = OUTPUT_DIR
) {
  message("[changepoint] Loading P&L series...")
  pnl_df <- load_pnl_series(db_path)
  n      <- nrow(pnl_df)
  message("[changepoint] ", n, " daily observations loaded")

  if (n < 30) {
    stop("[changepoint] Insufficient data: need ≥ 30 daily observations")
  }

  # ── PELT detection ──
  message("[changepoint] Running PELT algorithm...")
  pelt_out  <- detect_changepoints_pelt(pnl_df)

  pelt_cpts <- if (!is.null(pelt_out$cpt_meanvar)) {
    cpts_idx  <- cpts(pelt_out$cpt_meanvar)
    cpts_idx  <- cpts_idx[cpts_idx < n]   # exclude final index
    cpts_idx
  } else {
    integer(0)
  }

  message("[changepoint] PELT found ", length(pelt_cpts), " changepoint(s)")

  # ── Bayesian posterior ──
  message("[changepoint] Computing Bayesian posterior over changepoint locations...")
  post_df <- bayesian_changepoint_posterior(pnl_df$daily_pnl)

  # High-probability changepoints (posterior > MIN_POST_PROB threshold)
  # We identify local maxima of the posterior with prob > threshold
  bayesian_cpts <- post_df %>%
    filter(posterior_prob >= MIN_POST_PROB) %>%
    arrange(desc(posterior_prob))

  # ── Combine: PELT + top Bayesian candidates ──
  all_cpt_positions <- sort(unique(c(
    pelt_cpts,
    if (nrow(bayesian_cpts) > 0) head(bayesian_cpts$position, 5) else integer(0)
  )))

  # Filter: minimum segment length of 10 days between changepoints
  filtered_positions <- integer(0)
  prev <- 0L
  for (pos in all_cpt_positions) {
    if (pos - prev >= 10L && n - pos >= 10L) {
      filtered_positions <- c(filtered_positions, pos)
      prev <- pos
    }
  }

  message("[changepoint] Final changepoints at positions: ",
          paste(filtered_positions, collapse = ", "))

  # ── Segment characterisation ──
  all_positions <- c(0L, filtered_positions, n)
  regimes   <- list()
  changepoints_out <- list()

  for (i in seq_len(length(all_positions) - 1)) {
    t_start <- all_positions[i] + 1L
    t_end   <- all_positions[i + 1]
    seg     <- pnl_df[t_start:t_end, ]
    label   <- paste0("Regime_", i)
    regimes[[i]] <- characterise_regime(seg, label)

    if (i > 1) {
      cp_pos  <- all_positions[i]
      cp_date <- pnl_df$date[cp_pos]
      post_prob <- post_df %>%
        filter(position == cp_pos) %>%
        pull(posterior_prob)
      post_prob <- if (length(post_prob) == 0) NA_real_ else post_prob[1]

      changepoints_out[[length(changepoints_out) + 1]] <- list(
        position     = cp_pos,
        date         = format(cp_date),
        posterior_prob = round(post_prob, 4)
      )
    }
  }

  # ── Narratives ──
  narratives <- generate_narrative(regimes, changepoints_out)

  # ── Build output ──
  result <- list(
    n_obs           = n,
    date_range      = list(
      start = format(min(pnl_df$date)),
      end   = format(max(pnl_df$date))
    ),
    pelt_changepoints = pelt_cpts,
    n_changepoints   = length(filtered_positions),
    changepoints     = changepoints_out,
    regimes          = regimes,
    narratives       = narratives,
    posterior_profile = list(
      positions      = post_df$position,
      posterior_prob = round(post_df$posterior_prob, 6)
    ),
    threshold_used = MIN_POST_PROB
  )

  # ── Write output ──
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(output_dir, "bayesian_changepoint_results.json")
  write_json(result, out_path, auto_unbox = TRUE, pretty = TRUE)
  message("[changepoint] Results written to ", out_path)

  invisible(result)
}

# ── CLI entry ─────────────────────────────────────────────────────────────────

if (!interactive()) {
  result <- run_bayesian_changepoint()

  cat("\n=== Bayesian Changepoint Summary ===\n")
  cat("Observations: ", result$n_obs, "\n")
  cat("Date range:   ", result$date_range$start, " – ", result$date_range$end, "\n")
  cat("Changepoints: ", result$n_changepoints, "\n\n")

  for (cp in result$changepoints) {
    cat(sprintf("  [t=%d, date=%s, posterior_prob=%.3f]\n",
                cp$position, cp$date, cp$posterior_prob %||% NA))
  }

  cat("\nNarratives:\n")
  for (narr in result$narratives) {
    cat("  →", narr, "\n")
  }
}
