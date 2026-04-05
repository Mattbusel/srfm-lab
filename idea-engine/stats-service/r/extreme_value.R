# =============================================================================
# extreme_value.R — Extreme Value Theory for Strategy Tail Risk
# =============================================================================
# Fits Generalized Pareto Distribution (GPD) to the loss tail of trade P&L.
#
# Provides:
#   1. GPD fitting via MLE to losses exceeding the 95th percentile threshold
#   2. VaR at 99% and 99.9% confidence
#   3. Expected Shortfall (CVaR) — average loss in worst 1% of cases
#   4. Return level: expected maximum loss once per year / decade
#   5. Return level plot data (for dashboard)
#   6. Comparison: strategy tail vs naive buy-and-hold BTC
#
# Dependencies: tidyverse, evd (or manual GPD), jsonlite, lubridate, RSQLite
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(lubridate)
  library(RSQLite)
})

# ── Configuration ──────────────────────────────────────────────────────────────

DB_PATH       <- Sys.getenv("IDEA_ENGINE_DB",   "../db/idea_engine.db")
OUTPUT_DIR    <- Sys.getenv("STATS_OUTPUT_DIR", "../stats-service/output")
TAIL_QUANTILE <- 0.95    # threshold percentile for EVT (fit GPD to worst 5%)
TRADES_PER_YEAR <- 1500  # approximate trades per year for return level calc

# ── GPD Log-Likelihood and MLE ────────────────────────────────────────────────

#' GPD log-likelihood.
#' GPD(ξ, β): f(x) = (1/β)(1 + ξ x/β)^{-(1/ξ + 1)}  for x > 0
#' Special case ξ = 0: exponential distribution.
#'
#' @param exceedances positive numeric vector of threshold exceedances
#' @param xi  shape parameter (ξ; negative → bounded tail; 0 = exponential)
#' @param beta scale parameter (β > 0)
gpd_loglik <- function(exceedances, xi, beta) {
  if (beta <= 0) return(-Inf)
  n <- length(exceedances)

  if (abs(xi) < 1e-6) {
    # Exponential limit
    -n * log(beta) - sum(exceedances) / beta
  } else {
    z <- 1 + xi * exceedances / beta
    if (any(z <= 0)) return(-Inf)
    -n * log(beta) - (1 / xi + 1) * sum(log(z))
  }
}

#' MLE for GPD parameters via Nelder-Mead.
#' Returns (xi, beta, loglik, convergence_msg).
fit_gpd <- function(exceedances) {
  n  <- length(exceedances)
  x  <- sort(exceedances)

  # Method-of-moments starting values (Hosking & Wallis)
  m1 <- mean(x)
  m2 <- mean(x^2)
  xi_init   <- 0.5 * (1 - m1^2 / (m2 - m1^2))
  beta_init <- 0.5 * m1 * (m1^2 / (m2 - m1^2) + 1)

  if (!is.finite(xi_init) || !is.finite(beta_init)) {
    xi_init   <- 0.1
    beta_init <- m1
  }

  # Nelder-Mead minimisation of negative log-likelihood
  neg_ll <- function(params) {
    -gpd_loglik(x, params[1], exp(params[2]))
  }

  init_par  <- c(xi_init, log(max(beta_init, 1e-8)))
  opt_res   <- optim(init_par, neg_ll,
                     method  = "Nelder-Mead",
                     control = list(maxit = 2000, reltol = 1e-10))

  xi_hat   <- opt_res$par[1]
  beta_hat <- exp(opt_res$par[2])
  ll_hat   <- -opt_res$value

  # Profile confidence interval for xi (±1.96 SE from Hessian)
  hess_res  <- tryCatch(
    optimHess(opt_res$par, neg_ll),
    error = function(e) matrix(c(1e4, 0, 0, 1e4), 2, 2)
  )
  cov_mat <- tryCatch(solve(hess_res), error = function(e) matrix(c(1, 0, 0, 1), 2, 2))
  se_xi   <- sqrt(abs(cov_mat[1, 1]))
  se_logbeta <- sqrt(abs(cov_mat[2, 2]))

  list(
    xi            = xi_hat,
    beta          = beta_hat,
    loglik        = ll_hat,
    se_xi         = se_xi,
    xi_ci_lower   = xi_hat - 1.96 * se_xi,
    xi_ci_upper   = xi_hat + 1.96 * se_xi,
    converged     = opt_res$convergence == 0,
    n_exceedances = n,
    tail_label    = if (xi_hat > 0.3) "Heavy tail (Pareto-like)"
                    else if (xi_hat > -0.1) "Moderate tail (near-exponential)"
                    else "Bounded tail (Weibull)"
  )
}

# ── VaR and CVaR from GPD ─────────────────────────────────────────────────────

#' GPD-based VaR at probability level p (exceedance probability 1-p).
#' @param p    confidence level (e.g. 0.99)
#' @param u    threshold (95th percentile of losses)
#' @param n_total total number of loss observations
#' @param n_exceed number of observations exceeding threshold
gpd_var <- function(p, xi, beta, u, n_total, n_exceed) {
  # Exceedance probability over threshold: P(L > u) = n_exceed / n_total
  p_u <- n_exceed / n_total
  # P(L > VaR) = (1-p)  →  P(L > VaR | L > u) = (1-p) / p_u
  p_exceed_given_u <- (1 - p) / p_u

  if (p_exceed_given_u >= 1) return(u)   # VaR at or below threshold

  if (abs(xi) < 1e-6) {
    u - beta * log(p_exceed_given_u)
  } else {
    u + beta / xi * (p_exceed_given_u^(-xi) - 1)
  }
}

#' GPD-based CVaR (Expected Shortfall) at level p.
gpd_cvar <- function(p, xi, beta, u, n_total, n_exceed) {
  VaR <- gpd_var(p, xi, beta, u, n_total, n_exceed)
  if (xi >= 1) return(Inf)   # undefined for xi >= 1
  # ES = VaR + (beta + xi*(VaR - u)) / (1 - xi)
  beta_star <- beta + xi * (VaR - u)
  VaR + beta_star / (1 - xi)
}

# ── Return Levels ─────────────────────────────────────────────────────────────

#' Compute return levels: expected maximum loss for return periods T.
#' @param return_periods_trades vector of return periods in number of trades
compute_return_levels <- function(xi, beta, u, n_total, n_exceed,
                                  return_periods_trades = c(100, 500, 1000, 5000, 15000)) {
  p_levels <- 1 - 1 / return_periods_trades
  map_dfr(seq_along(return_periods_trades), function(i) {
    T_trades <- return_periods_trades[i]
    p        <- p_levels[i]
    rl       <- gpd_var(p, xi, beta, u, n_total, n_exceed)
    tibble(
      return_period_trades = T_trades,
      return_period_years  = T_trades / TRADES_PER_YEAR,
      return_level_loss    = rl
    )
  })
}

# ── Data Loading ──────────────────────────────────────────────────────────────

load_trade_losses <- function(db_path = DB_PATH) {
  if (!file.exists(db_path)) {
    message("[extreme_value] DB not found — generating synthetic losses")
    set.seed(77)
    n <- 2000
    # Strategy: mostly small losses, occasional heavy-tail loss events
    losses_strat <- pmax(
      -c(rnorm(1800, mean = 0.005, sd = 0.015),
         rt(200, df = 3) * 0.02),
      0
    )
    # BTC buy-and-hold: convert daily returns to loss (negative returns)
    btc_daily_rets <- rnorm(n, mean = 0.001, sd = 0.040)
    losses_btc     <- pmax(-btc_daily_rets, 0)

    list(
      strategy = losses_strat[losses_strat > 0],
      btc_bnh  = losses_btc[losses_btc > 0]
    )
  } else {
    con <- dbConnect(SQLite(), db_path)
    on.exit(dbDisconnect(con))

    strat_df <- dbGetQuery(con, "
      SELECT ABS(pnl_pct) AS loss
      FROM trades
      WHERE pnl_pct < 0 AND closed_at IS NOT NULL
    ")

    btc_df <- dbGetQuery(con, "
      SELECT ABS(btc_return_at_entry) AS loss
      FROM trades
      WHERE btc_return_at_entry < 0 AND closed_at IS NOT NULL
    ")

    list(
      strategy = strat_df$loss,
      btc_bnh  = btc_df$loss
    )
  }
}

# ── Main Pipeline ─────────────────────────────────────────────────────────────

run_extreme_value <- function(
  db_path    = DB_PATH,
  output_dir = OUTPUT_DIR
) {
  message("[extreme_value] Loading trade loss data...")
  data <- load_trade_losses(db_path)

  fit_tail <- function(losses, label) {
    n_total <- length(losses)
    if (n_total < 20) {
      warning(label, ": insufficient data")
      return(NULL)
    }

    # Threshold = 95th percentile of losses
    u <- quantile(losses, TAIL_QUANTILE)
    exceedances <- losses[losses > u] - u
    n_exceed    <- length(exceedances)

    message(sprintf("[extreme_value] %s: n=%d, threshold=%.5f, n_exceed=%d",
                    label, n_total, u, n_exceed))

    if (n_exceed < 10) {
      warning(label, ": too few exceedances (", n_exceed, ")")
      return(NULL)
    }

    # Fit GPD
    gpd <- fit_gpd(exceedances)

    # VaR and CVaR
    var_99   <- gpd_var(0.99,  gpd$xi, gpd$beta, u, n_total, n_exceed)
    var_999  <- gpd_var(0.999, gpd$xi, gpd$beta, u, n_total, n_exceed)
    cvar_99  <- gpd_cvar(0.99, gpd$xi, gpd$beta, u, n_total, n_exceed)

    # Empirical VaR for comparison
    var_99_emp  <- quantile(losses, 0.99)
    var_999_emp <- quantile(losses, 0.999)

    # Return levels
    rl_df <- compute_return_levels(gpd$xi, gpd$beta, u, n_total, n_exceed)

    # Return level plot: smooth VaR curve over range of p
    p_seq   <- seq(0.95, 0.9999, length.out = 80)
    rl_plot <- map_dfr(p_seq, function(p) {
      tibble(
        confidence   = p,
        var_gpd      = gpd_var(p, gpd$xi, gpd$beta, u, n_total, n_exceed),
        var_empirical = quantile(losses, p)
      )
    })

    list(
      label         = label,
      n_total       = n_total,
      n_exceedances = n_exceed,
      threshold_u   = u,
      tail_quantile = TAIL_QUANTILE,
      gpd = list(
        xi         = gpd$xi,
        beta       = gpd$beta,
        loglik     = gpd$loglik,
        converged  = gpd$converged,
        tail_label = gpd$tail_label,
        xi_ci      = c(gpd$xi_ci_lower, gpd$xi_ci_upper)
      ),
      var_99_gpd     = var_99,
      var_999_gpd    = var_999,
      cvar_99_gpd    = cvar_99,
      var_99_emp     = var_99_emp,
      var_999_emp    = var_999_emp,
      return_levels  = rl_df %>% as.list() %>% purrr::transpose(),
      rl_plot        = list(
        confidence   = rl_plot$confidence,
        var_gpd      = rl_plot$var_gpd,
        var_empirical = rl_plot$var_empirical
      )
    )
  }

  message("[extreme_value] Fitting GPD to strategy losses...")
  strat_fit <- fit_tail(data$strategy, "BH_strategy")

  message("[extreme_value] Fitting GPD to BTC buy-and-hold losses...")
  btc_fit   <- fit_tail(data$btc_bnh, "BTC_buyhold")

  # ── Comparison ──
  comparison <- if (!is.null(strat_fit) && !is.null(btc_fit)) {
    list(
      var_99_ratio       = strat_fit$var_99_gpd  / btc_fit$var_99_gpd,
      var_999_ratio      = strat_fit$var_999_gpd / btc_fit$var_999_gpd,
      cvar_99_ratio      = strat_fit$cvar_99_gpd / btc_fit$cvar_99_gpd,
      xi_comparison      = list(strategy = strat_fit$gpd$xi, btc = btc_fit$gpd$xi),
      interpretation     = {
        r <- strat_fit$var_99_gpd / btc_fit$var_99_gpd
        if (r < 0.5) "Strategy tail risk MUCH lower than BTC buy-and-hold (VaR<50%)"
        else if (r < 0.8) "Strategy tail risk lower than BTC buy-and-hold"
        else if (r < 1.2) "Strategy tail risk comparable to BTC buy-and-hold"
        else "Strategy tail risk HIGHER than BTC buy-and-hold — review risk controls"
      }
    )
  } else NULL

  result <- list(
    strategy  = strat_fit,
    btc_bnh   = btc_fit,
    comparison = comparison,
    config     = list(
      tail_quantile    = TAIL_QUANTILE,
      trades_per_year  = TRADES_PER_YEAR
    )
  )

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(output_dir, "extreme_value_results.json")
  write_json(result, out_path, auto_unbox = TRUE, pretty = TRUE)
  message("[extreme_value] Results written to ", out_path)

  invisible(result)
}

# ── CLI entry ─────────────────────────────────────────────────────────────────

if (!interactive()) {
  result <- run_extreme_value()

  cat("\n=== Extreme Value Theory Summary ===\n")

  if (!is.null(result$strategy)) {
    s <- result$strategy
    cat("\nStrategy (BH):\n")
    cat("  GPD shape ξ:     ", round(s$gpd$xi, 4), " [", s$gpd$tail_label, "]\n")
    cat("  VaR 99%  (GPD):  ", round(s$var_99_gpd  * 100, 3), "%\n")
    cat("  VaR 99.9%(GPD):  ", round(s$var_999_gpd * 100, 3), "%\n")
    cat("  CVaR 99% (GPD):  ", round(s$cvar_99_gpd * 100, 3), "%\n")

    rl <- result$strategy$return_levels
    cat("  1-year max loss: ~", round(rl[[which.min(abs(sapply(rl, `[[`, "return_period_years") - 1))]]$return_level_loss * 100, 2), "%\n")
  }

  if (!is.null(result$btc_bnh)) {
    b <- result$btc_bnh
    cat("\nBTC buy-and-hold:\n")
    cat("  GPD shape ξ:     ", round(b$gpd$xi, 4), "\n")
    cat("  VaR 99%  (GPD):  ", round(b$var_99_gpd * 100, 3), "%\n")
    cat("  CVaR 99% (GPD):  ", round(b$cvar_99_gpd * 100, 3), "%\n")
  }

  if (!is.null(result$comparison)) {
    cat("\nComparison: ", result$comparison$interpretation, "\n")
  }
}
