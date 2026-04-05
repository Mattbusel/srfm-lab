# =============================================================================
# alpha_testing.R
# Alpha signal testing framework for quant/crypto trading
# Base R only -- no external packages
# =============================================================================
# Financial intuition: A signal (e.g., momentum score, on-chain metric) has
# alpha if it predicts future returns. IC (Information Coefficient) measures
# the rank correlation between signal and subsequent return -- positive IC
# means the signal is informative. Rigorous testing requires correcting for
# multiple comparisons and measuring alpha decay.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. INFORMATION COEFFICIENT (IC)
# -----------------------------------------------------------------------------

#' Compute Information Coefficient: rank correlation between signal and return
#' IC = Spearman correlation(signal_t, return_{t+h})
#' IC > 0.05 is considered useful; IC > 0.10 is strong for equity factors
#' @param signal vector of signal values at time t
#' @param forward_return vector of returns at time t+h
#' @return IC value
compute_ic <- function(signal, forward_return) {
  valid <- !is.na(signal) & !is.na(forward_return)
  if (sum(valid) < 10) return(NA)
  cor(rank(signal[valid]), rank(forward_return[valid]))
}

#' Compute IC across multiple forecast horizons
#' Alpha decay: IC typically decays as horizon h increases
#' @param signal_mat matrix of signals: rows=time, cols=assets
#' @param return_mat matrix of returns: rows=time, cols=assets
#' @param horizons vector of forecast horizons (in periods)
ic_by_horizon <- function(signal_mat, return_mat, horizons = 1:20) {
  T_obs <- nrow(signal_mat)
  ic_by_h <- sapply(horizons, function(h) {
    if (h >= T_obs) return(NA)
    # Compute IC at each time step for horizon h
    ic_ts <- sapply(seq_len(T_obs - h), function(t) {
      sig_t <- signal_mat[t, ]
      ret_t <- return_mat[t + h, ]
      compute_ic(sig_t, ret_t)
    })
    mean(ic_ts, na.rm = TRUE)
  })
  data.frame(horizon=horizons, IC=ic_by_h)
}

#' IC information ratio: IC_mean / IC_std (analogous to Sharpe ratio for signals)
ic_information_ratio <- function(signal_mat, return_mat, horizon = 1) {
  T_obs <- nrow(signal_mat)
  ic_ts <- sapply(seq_len(T_obs - horizon), function(t) {
    compute_ic(signal_mat[t, ], return_mat[t + horizon, ])
  })
  ic_ts <- ic_ts[!is.na(ic_ts)]
  n <- length(ic_ts)
  ic_mean <- mean(ic_ts)
  ic_std  <- sd(ic_ts)
  ir      <- ic_mean / (ic_std + 1e-10) * sqrt(n)  # annualized-style
  t_stat  <- ic_mean / (ic_std / sqrt(n))

  cat("=== IC Summary ===\n")
  cat(sprintf("IC mean:   %.4f\n", ic_mean))
  cat(sprintf("IC std:    %.4f\n", ic_std))
  cat(sprintf("IC IR:     %.3f\n", ir))
  cat(sprintf("t-stat:    %.3f\n", t_stat))
  cat(sprintf("p-value:   %.4f\n", 2*pt(-abs(t_stat), df=n-1)))
  cat(sprintf("% positive: %.1f%%\n", 100*mean(ic_ts > 0)))

  list(ic_mean=ic_mean, ic_std=ic_std, ic_ts=ic_ts, IR=ir, t_stat=t_stat,
       pct_positive=mean(ic_ts>0))
}

# -----------------------------------------------------------------------------
# 2. FAMA-MACBETH CROSS-SECTIONAL REGRESSION
# -----------------------------------------------------------------------------

#' Fama-MacBeth (1973) cross-sectional regression
#' Step 1: At each time t, regress cross-section of returns on signals
#' Step 2: Time-series average of cross-sectional lambda (signal premium)
#' lambda_t = OLS coefficient from r_{i,t+1} = alpha_t + lambda_t * signal_{i,t} + e_{i,t}
#' @param signal_mat T x N matrix of signals
#' @param return_mat T x N matrix of returns
#' @param horizon forecast horizon
fama_macbeth <- function(signal_mat, return_mat, horizon = 1,
                          newey_west_lags = 3) {
  T_obs  <- nrow(signal_mat)
  N      <- ncol(signal_mat)
  lambdas <- numeric(T_obs - horizon)
  alphas  <- numeric(T_obs - horizon)
  r2_cs   <- numeric(T_obs - horizon)

  for (t in seq_len(T_obs - horizon)) {
    y <- return_mat[t + horizon, ]
    x <- signal_mat[t, ]
    valid <- !is.na(y) & !is.na(x)
    if (sum(valid) < 5) next
    y_v <- y[valid]; x_v <- x[valid]
    # Standardize signal
    x_std <- (x_v - mean(x_v)) / (sd(x_v) + 1e-10)
    X <- cbind(1, x_std)
    b <- solve(t(X) %*% X) %*% t(X) %*% y_v
    alphas[t]  <- b[1]
    lambdas[t] <- b[2]
    r_hat <- X %*% b
    r2_cs[t] <- 1 - sum((y_v - r_hat)^2) / sum((y_v - mean(y_v))^2)
  }

  # Time-series inference on lambda
  valid_l <- lambdas != 0
  lam_vec <- lambdas[valid_l]
  n_T     <- length(lam_vec)
  lam_mean <- mean(lam_vec)
  # Newey-West HAC standard errors
  nw_se <- newey_west_se(lam_vec, lags = newey_west_lags)
  t_stat <- lam_mean / nw_se

  cat("=== Fama-MacBeth Regression ===\n")
  cat(sprintf("Lambda (signal premium): %.6f\n", lam_mean))
  cat(sprintf("NW Std Error:            %.6f\n", nw_se))
  cat(sprintf("t-statistic:             %.3f\n", t_stat))
  cat(sprintf("p-value:                 %.4f\n", 2*pt(-abs(t_stat), df=n_T-1)))
  cat(sprintf("Avg cross-sectional R2:  %.4f\n", mean(r2_cs[valid_l])))

  list(lambda_mean=lam_mean, lambda_ts=lam_vec, se_nw=nw_se,
       t_stat=t_stat, pval=2*pt(-abs(t_stat), df=n_T-1),
       avg_r2=mean(r2_cs[valid_l]))
}

#' Newey-West HAC variance estimator
newey_west_se <- function(x, lags = 3) {
  n   <- length(x)
  x_c <- x - mean(x)
  s0  <- sum(x_c^2) / n
  s   <- s0
  for (l in seq_len(lags)) {
    w  <- 1 - l / (lags + 1)  # Bartlett weight
    s  <- s + 2 * w * sum(x_c[(l+1):n] * x_c[1:(n-l)]) / n
  }
  sqrt(s / n)
}

# -----------------------------------------------------------------------------
# 3. FACTOR PORTFOLIO CONSTRUCTION (QUINTILE SORTS)
# -----------------------------------------------------------------------------

#' Sort assets into quintile portfolios by signal value
#' Long top quintile, short bottom quintile = "L/S factor"
#' @param signal_mat T x N matrix of signals
#' @param return_mat T x N matrix of forward returns
#' @param n_groups number of groups (default 5 = quintiles)
quintile_sort_factor <- function(signal_mat, return_mat, n_groups = 5,
                                  horizon = 1, weight_scheme = "equal") {
  T_obs <- nrow(signal_mat)
  # Returns for each group at each time step
  group_returns <- matrix(NA, T_obs - horizon, n_groups)

  for (t in seq_len(T_obs - horizon)) {
    sig_t <- signal_mat[t, ]
    ret_t <- return_mat[t + horizon, ]
    valid <- !is.na(sig_t) & !is.na(ret_t)
    if (sum(valid) < n_groups * 3) next

    sig_v <- sig_t[valid]; ret_v <- ret_t[valid]
    breaks <- quantile(sig_v, probs = seq(0, 1, by = 1/n_groups))
    breaks[1] <- -Inf; breaks[n_groups+1] <- Inf
    grp    <- cut(sig_v, breaks = breaks, labels = seq_len(n_groups))

    for (g in seq_len(n_groups)) {
      idx_g <- which(grp == g)
      if (length(idx_g) == 0) next
      if (weight_scheme == "equal") {
        group_returns[t, g] <- mean(ret_v[idx_g])
      } else if (weight_scheme == "value") {
        # For value weighting we'd need market caps; fall back to equal
        group_returns[t, g] <- mean(ret_v[idx_g])
      }
    }
  }

  # L/S factor: long top group, short bottom group
  ls_factor <- group_returns[, n_groups] - group_returns[, 1]
  # Monotonic spread: do returns increase from group 1 to n?
  group_means <- colMeans(group_returns, na.rm = TRUE)
  spread_monotonic <- all(diff(group_means) > 0)

  # t-statistics for each group vs zero
  t_stats <- sapply(seq_len(n_groups), function(g) {
    x <- group_returns[, g]; x <- x[!is.na(x)]
    n <- length(x); if (n < 2) return(NA)
    mean(x) / (sd(x) / sqrt(n))
  })
  ls_t <- mean(ls_factor, na.rm=TRUE) / (sd(ls_factor, na.rm=TRUE) / sqrt(sum(!is.na(ls_factor))))

  cat("=== Quintile Sort Factor Returns ===\n")
  cat("Group means:\n")
  for (g in seq_len(n_groups)) {
    cat(sprintf("  Group %d: %.4f (t=%.2f)\n", g, group_means[g], t_stats[g]))
  }
  cat(sprintf("L/S Factor:  %.4f (t=%.2f)\n", mean(ls_factor, na.rm=TRUE), ls_t))
  cat(sprintf("Monotonic:   %s\n", spread_monotonic))

  list(group_returns=group_returns, ls_factor=ls_factor,
       group_means=group_means, t_stats=t_stats, ls_t=ls_t,
       monotonic=spread_monotonic)
}

# -----------------------------------------------------------------------------
# 4. MULTIPLE TESTING CORRECTION
# -----------------------------------------------------------------------------

#' Benjamini-Hochberg (BH) FDR correction
#' Controls False Discovery Rate -- more powerful than Bonferroni
#' for many simultaneous tests (e.g., testing 50 signals)
#' @param p_values vector of p-values
#' @param alpha FDR level (e.g., 0.05)
bh_correction <- function(p_values, alpha = 0.05) {
  m <- length(p_values)
  ord <- order(p_values)
  p_sorted <- p_values[ord]
  # BH threshold: reject H_k if p_(k) <= k/m * alpha
  k_vals <- seq_len(m)
  rejected <- p_sorted <= k_vals / m * alpha
  # Find largest k where rejection holds (all below are also rejected)
  if (any(rejected)) {
    k_max <- max(which(rejected))
    rejected_bh <- seq_len(k_max)
  } else {
    rejected_bh <- integer(0)
  }
  result <- logical(m)
  result[ord[rejected_bh]] <- TRUE
  adj_pval <- pmin(p_sorted * m / k_vals, 1)
  adj_pval <- rev(cummin(rev(adj_pval)))  # enforce monotonicity

  list(rejected = result, adj_p_values = adj_pval[order(ord)],
       n_rejected = sum(result), fdr_threshold = alpha)
}

#' Romano-Wolf bootstrap multiple testing correction
#' Strongly controls FWER (familywise error rate) via bootstrap
#' More powerful than Bonferroni, accounts for correlations between tests
#' @param test_stats observed test statistics (e.g., t-stats for each signal)
#' @param signals_mat T x K matrix of signals
#' @param returns_vec T-vector of returns
#' @param n_boot number of bootstrap replications
romano_wolf <- function(test_stats, signals_mat, returns_vec,
                         n_boot = 1000, alpha = 0.05) {
  K <- length(test_stats)
  n <- length(returns_vec)

  # Bootstrap distribution of max test statistic
  # Under null: shuffle returns (permutation test)
  max_stats_boot <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    idx_boot <- sample(seq_len(n), n, replace = TRUE)
    ret_boot  <- returns_vec[idx_boot]
    t_boot    <- sapply(seq_len(K), function(k) {
      sig_k  <- signals_mat[, k]
      valid  <- !is.na(sig_k) & !is.na(ret_boot)
      if (sum(valid) < 5) return(0)
      x <- sig_k[valid]; y <- ret_boot[valid]
      n_v <- sum(valid)
      r <- cor(x, y)
      r * sqrt(n_v - 2) / sqrt(1 - r^2 + 1e-10)
    })
    max_stats_boot[b] <- max(abs(t_boot))
  }

  # p-values: fraction of bootstrap max stats exceeding observed
  p_rw <- sapply(seq_len(K), function(k) {
    mean(max_stats_boot >= abs(test_stats[k]))
  })

  rejected <- p_rw < alpha

  cat("=== Romano-Wolf Multiple Testing ===\n")
  cat(sprintf("Signals tested: %d\n", K))
  cat(sprintf("Rejected at alpha=%.2f: %d\n", alpha, sum(rejected)))
  if (any(rejected)) {
    cat(sprintf("Rejected signal indices: %s\n",
                paste(which(rejected), collapse=",")))
  }

  list(p_rw=p_rw, rejected=rejected, max_stats_boot=max_stats_boot,
       n_rejected=sum(rejected))
}

# -----------------------------------------------------------------------------
# 5. ALPHA DECAY CURVE FITTING
# -----------------------------------------------------------------------------

#' Fit alpha decay: IC as function of horizon h
#' Exponential: IC(h) = IC_0 * exp(-lambda * h), halflife = log(2)/lambda
#' Power law:   IC(h) = IC_0 * h^{-alpha}
#' @param ic_by_horizon data.frame with columns horizon, IC
fit_alpha_decay <- function(ic_df) {
  h   <- ic_df$horizon
  ic  <- ic_df$IC
  valid <- !is.na(ic) & ic > 0
  if (sum(valid) < 4) {
    cat("Insufficient data for decay fitting\n")
    return(NULL)
  }
  h_v  <- h[valid]; ic_v <- ic[valid]

  # Exponential fit: log(IC) = log(IC_0) - lambda*h
  log_ic <- log(ic_v)
  X_exp  <- cbind(1, h_v)
  b_exp  <- solve(t(X_exp) %*% X_exp) %*% t(X_exp) %*% log_ic
  ic0_exp  <- exp(b_exp[1])
  lambda   <- -b_exp[2]
  halflife_exp <- log(2) / max(lambda, 1e-5)

  # Power law fit: log(IC) = log(IC_0) - alpha*log(h)
  log_h  <- log(h_v)
  X_pow  <- cbind(1, log_h)
  b_pow  <- solve(t(X_pow) %*% X_pow) %*% t(X_pow) %*% log_ic
  ic0_pow  <- exp(b_pow[1])
  decay_pw <- -b_pow[2]

  # Compute fitted values and R^2 for both
  fit_exp <- ic0_exp * exp(-lambda * h_v)
  fit_pow <- ic0_pow * h_v^(-decay_pw)
  r2_exp  <- 1 - sum((ic_v - fit_exp)^2) / sum((ic_v - mean(ic_v))^2)
  r2_pow  <- 1 - sum((ic_v - fit_pow)^2) / sum((ic_v - mean(ic_v))^2)

  cat("=== Alpha Decay Analysis ===\n")
  cat(sprintf("Exponential decay: IC_0=%.4f, lambda=%.3f, half-life=%.1f periods, R2=%.3f\n",
              ic0_exp, lambda, halflife_exp, r2_exp))
  cat(sprintf("Power law decay:   IC_0=%.4f, alpha=%.3f, R2=%.3f\n",
              ic0_pow, decay_pw, r2_pow))
  cat(sprintf("Preferred model: %s\n", ifelse(r2_exp > r2_pow, "Exponential", "Power Law")))

  list(exp = list(IC0=ic0_exp, lambda=lambda, halflife=halflife_exp, r2=r2_exp),
       power = list(IC0=ic0_pow, alpha=decay_pw, r2=r2_pow),
       best = ifelse(r2_exp > r2_pow, "exponential", "power"))
}

# -----------------------------------------------------------------------------
# 6. SIGNAL COMBINATION VIA IC-WEIGHTED ENSEMBLE
# -----------------------------------------------------------------------------

#' Combine multiple signals using IC-weighted average
#' Signals with higher IC get more weight in the composite
#' @param signal_list list of T x N signal matrices
#' @param ic_weights vector of IC values for each signal (from in-sample)
ic_weighted_ensemble <- function(signal_list, ic_weights = NULL) {
  K <- length(signal_list)
  T_obs <- nrow(signal_list[[1]])
  N     <- ncol(signal_list[[1]])

  if (is.null(ic_weights)) {
    ic_weights <- rep(1/K, K)
  }

  # Normalize IC weights (keep only positive ICs)
  w <- pmax(ic_weights, 0)
  if (sum(w) == 0) w <- rep(1/K, K)
  w <- w / sum(w)

  # Weighted average signal
  composite <- matrix(0, T_obs, N)
  for (k in seq_len(K)) {
    sig_k <- signal_list[[k]]
    # Cross-sectionally standardize each signal first
    for (t in seq_len(T_obs)) {
      s_t <- sig_k[t, ]
      valid <- !is.na(s_t)
      if (sum(valid) > 1) {
        sig_k[t, valid] <- (s_t[valid] - mean(s_t[valid])) / (sd(s_t[valid]) + 1e-10)
      }
    }
    composite <- composite + w[k] * sig_k
  }

  cat("=== IC-Weighted Signal Ensemble ===\n")
  cat("Signal weights:\n")
  for (k in seq_len(K)) cat(sprintf("  Signal %d: weight=%.3f\n", k, w[k]))

  list(composite=composite, weights=w)
}

# -----------------------------------------------------------------------------
# 7. WALK-FORWARD IS/OOS ALPHA STABILITY
# -----------------------------------------------------------------------------

#' Walk-forward alpha stability test
#' Split sample into multiple IS/OOS windows; measure IC in each
#' Stable alpha: OOS IC is consistently positive and similar to IS IC
#' @param signal_mat T x N signal matrix
#' @param return_mat T x N return matrix
#' @param n_windows number of walk-forward windows
#' @param oos_fraction fraction of each window used for OOS
#' @param horizon forecast horizon
walk_forward_alpha <- function(signal_mat, return_mat, n_windows = 10,
                                oos_fraction = 0.3, horizon = 1) {
  T_obs    <- nrow(signal_mat)
  win_size <- floor(T_obs / n_windows)
  is_size  <- round(win_size * (1 - oos_fraction))
  oos_size <- win_size - is_size

  results <- vector("list", n_windows - 1)

  for (w in seq_len(n_windows - 1)) {
    is_start  <- (w - 1) * win_size + 1
    is_end    <- is_start + is_size - 1
    oos_start <- is_end + 1
    oos_end   <- min(oos_start + oos_size - 1, T_obs - horizon)
    if (oos_start > T_obs - horizon) break

    # IS IC
    is_ic_ts <- sapply(seq(is_start, is_end - horizon), function(t) {
      compute_ic(signal_mat[t,], return_mat[t+horizon,])
    })
    # OOS IC
    oos_ic_ts <- sapply(seq(oos_start, oos_end), function(t) {
      compute_ic(signal_mat[t,], return_mat[t+horizon,])
    })

    results[[w]] <- list(
      window = w,
      is_ic  = mean(is_ic_ts, na.rm=TRUE),
      oos_ic = mean(oos_ic_ts, na.rm=TRUE),
      is_period  = c(is_start, is_end),
      oos_period = c(oos_start, oos_end)
    )
  }

  results <- Filter(Negate(is.null), results)
  is_ics  <- sapply(results, `[[`, "is_ic")
  oos_ics <- sapply(results, `[[`, "oos_ic")

  # IS/OOS IC correlation (high = stable alpha)
  ic_corr <- cor(is_ics, oos_ics)
  # Sharpe-like ratio of OOS IC
  oos_ir   <- mean(oos_ics) / (sd(oos_ics) + 1e-10) * sqrt(length(oos_ics))
  decay_ratio <- mean(oos_ics) / (mean(is_ics) + 1e-10)  # OOS/IS decay

  cat("=== Walk-Forward Alpha Stability ===\n")
  cat(sprintf("Windows: %d\n", length(results)))
  cat(sprintf("IS IC:   mean=%.4f, sd=%.4f\n", mean(is_ics), sd(is_ics)))
  cat(sprintf("OOS IC:  mean=%.4f, sd=%.4f\n", mean(oos_ics), sd(oos_ics)))
  cat(sprintf("IS/OOS IC correlation: %.3f\n", ic_corr))
  cat(sprintf("OOS/IS decay ratio:    %.3f\n", decay_ratio))
  cat(sprintf("OOS IR:                %.3f\n", oos_ir))
  cat(sprintf("% OOS windows positive IC: %.1f%%\n", 100*mean(oos_ics > 0)))

  list(windows=results, is_ics=is_ics, oos_ics=oos_ics,
       ic_corr=ic_corr, decay_ratio=decay_ratio, oos_ir=oos_ir)
}

# -----------------------------------------------------------------------------
# 8. COMPREHENSIVE ALPHA TESTING PIPELINE
# -----------------------------------------------------------------------------

#' Full alpha testing suite for a signal
#' @param signal_mat T x N matrix of signal values
#' @param return_mat T x N matrix of forward-realized returns
#' @param signal_name name of the signal
run_alpha_testing <- function(signal_mat, return_mat,
                               signal_name = "Signal",
                               horizons = c(1, 2, 3, 5, 10, 15, 20)) {
  cat("=============================================================\n")
  cat(sprintf("ALPHA TESTING: %s\n", signal_name))
  cat(sprintf("Dimensions: %d periods x %d assets\n\n",
              nrow(signal_mat), ncol(signal_mat)))

  # 1. IC by horizon (alpha decay)
  cat("--- IC by Horizon ---\n")
  ic_df <- ic_by_horizon(signal_mat, return_mat, horizons)
  print(ic_df)

  # 2. IC IR at horizon 1
  cat("\n--- IC Information Ratio (h=1) ---\n")
  ic_res <- ic_information_ratio(signal_mat, return_mat, horizon=1)

  # 3. Fama-MacBeth
  cat("\n--- Fama-MacBeth ---\n")
  fm_res <- fama_macbeth(signal_mat, return_mat, horizon=1)

  # 4. Quintile sort
  cat("\n--- Quintile Sort ---\n")
  qs_res <- quintile_sort_factor(signal_mat, return_mat, n_groups=5, horizon=1)

  # 5. Alpha decay fitting
  cat("\n--- Alpha Decay Fit ---\n")
  decay_res <- fit_alpha_decay(ic_df)

  # 6. Walk-forward
  cat("\n--- Walk-Forward Alpha Stability ---\n")
  if (nrow(signal_mat) >= 50) {
    wf_res <- walk_forward_alpha(signal_mat, return_mat, n_windows=5, horizon=1)
  } else {
    cat("Insufficient data for walk-forward test\n")
    wf_res <- NULL
  }

  # 7. Summary verdict
  cat("\n=== ALPHA VERDICT ===\n")
  ic_mean <- ic_res$ic_mean
  t_stat  <- ic_res$t_stat
  has_alpha <- ic_mean > 0.02 && t_stat > 2
  cat(sprintf("Signal: %s\n", signal_name))
  cat(sprintf("IC mean=%.4f, t=%.2f => %s\n",
              ic_mean, t_stat,
              ifelse(has_alpha, "ALPHA DETECTED", "NO SIGNIFICANT ALPHA")))
  if (!is.null(decay_res)) {
    cat(sprintf("Alpha half-life: %.1f periods\n", decay_res$exp$halflife))
  }
  if (!is.null(wf_res)) {
    cat(sprintf("OOS IC mean: %.4f (%s)\n", mean(wf_res$oos_ics),
                ifelse(mean(wf_res$oos_ics) > 0, "STABLE", "OVERFITTING WARNING")))
  }

  invisible(list(ic_df=ic_df, ic_res=ic_res, fm=fm_res,
                 qs=qs_res, decay=decay_res, walk_forward=wf_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(99)
# T_obs <- 300; N_assets <- 30
# # Simulate a signal with some true alpha at short horizons
# true_signal <- matrix(rnorm(T_obs * N_assets), T_obs, N_assets)
# noise <- matrix(rnorm(T_obs * N_assets, 0, 3), T_obs, N_assets)
# signal_mat <- true_signal + noise
# return_mat <- matrix(0, T_obs, N_assets)
# for (t in 1:(T_obs-1)) {
#   # Small positive IC built in
#   return_mat[t+1, ] <- 0.05 * true_signal[t, ] + rnorm(N_assets, 0, 0.02)
# }
# result <- run_alpha_testing(signal_mat, return_mat, "Momentum_Score")

# =============================================================================
# EXTENDED ALPHA TESTING: Signal Neutralization, Regime-Conditional IC,
# Turnover Analysis, Multi-Signal Combination, and Decay Profiling
# =============================================================================

# -----------------------------------------------------------------------------
# Signal Neutralization: remove market, sector, or factor exposures from signal
# Raw signals often correlate with beta, size, or momentum -- neutralizing
# isolates the unique alpha and prevents factor contamination
# -----------------------------------------------------------------------------
neutralize_signal <- function(signal_mat, factor_mat, method = "cross_sectional") {
  # signal_mat: T x N matrix of raw signals
  # factor_mat: T x N x F array or T x N matrix of single factor (e.g., market cap)
  T_obs <- nrow(signal_mat); N <- ncol(signal_mat)

  neutral_mat <- signal_mat

  for (t in 1:T_obs) {
    s <- signal_mat[t, ]
    if (all(is.na(s))) next

    if (is.matrix(factor_mat)) {
      # Single factor neutralization
      f <- factor_mat[t, ]
      valid <- !is.na(s) & !is.na(f)
      if (sum(valid) < 3) next
      # Regress signal on factor, use residuals
      fit <- lm(s[valid] ~ f[valid])
      neutral_mat[t, valid] <- residuals(fit)
    }
  }

  # Re-standardize after neutralization
  for (t in 1:T_obs) {
    s <- neutral_mat[t, ]
    valid <- !is.na(s)
    if (sum(valid) < 2) next
    mu_s <- mean(s[valid]); sd_s <- sd(s[valid])
    if (sd_s > 0) neutral_mat[t, valid] <- (s[valid] - mu_s) / sd_s
  }

  neutral_mat
}

# -----------------------------------------------------------------------------
# Regime-Conditional IC Analysis: test if signal works better in certain regimes
# Many alpha signals are regime-dependent (e.g., momentum works in trending markets)
# -----------------------------------------------------------------------------
regime_conditional_ic <- function(signal_mat, return_mat, regime_vec,
                                    horizon = 1) {
  # regime_vec: T-length vector of regime labels
  regimes <- sort(unique(regime_vec[!is.na(regime_vec)]))

  results <- lapply(regimes, function(r) {
    idx <- which(regime_vec == r)
    # Ensure we have forward returns
    valid_t <- idx[idx <= (nrow(signal_mat) - horizon)]
    if (length(valid_t) < 10) return(NULL)

    ics <- numeric(length(valid_t))
    for (i in seq_along(valid_t)) {
      t <- valid_t[i]
      s <- signal_mat[t, ]; ret <- return_mat[t + horizon, ]
      valid_n <- !is.na(s) & !is.na(ret)
      if (sum(valid_n) < 3) { ics[i] <- NA; next }
      ics[i] <- cor(s[valid_n], ret[valid_n], method = "spearman")
    }
    ics <- ics[!is.na(ics)]

    list(
      regime = r, n_obs = length(ics),
      mean_ic = mean(ics), sd_ic = sd(ics),
      ic_ir = mean(ics) / sd(ics) * sqrt(length(ics)),
      t_stat = mean(ics) / sd(ics) * sqrt(length(ics)),
      p_value = 2 * pt(-abs(mean(ics)/sd(ics)*sqrt(length(ics))), df=length(ics)-1),
      pct_positive = mean(ics > 0)
    )
  })

  results <- results[!sapply(results, is.null)]
  stats_df <- do.call(rbind, lapply(results, as.data.frame))

  list(
    regime_ic_stats = stats_df,
    best_regime = stats_df$regime[which.max(stats_df$mean_ic)],
    worst_regime = stats_df$regime[which.min(stats_df$mean_ic)],
    ic_regime_spread = diff(range(stats_df$mean_ic))
  )
}

# -----------------------------------------------------------------------------
# Signal Turnover Analysis: high turnover erodes alpha via transaction costs
# Turnover = fraction of portfolio that changes each period
# Autocorrelation of signal predicts turnover
# -----------------------------------------------------------------------------
signal_turnover_analysis <- function(signal_mat, top_pct = 0.20) {
  T_obs <- nrow(signal_mat); N <- ncol(signal_mat)
  n_top <- max(1, floor(N * top_pct))

  # Long/short assignment based on signal rank
  daily_longs <- apply(signal_mat, 1, function(s) {
    if (all(is.na(s))) return(rep(FALSE, N))
    rank_s <- rank(-s, na.last = "keep")
    !is.na(rank_s) & rank_s <= n_top
  })  # N x T matrix

  daily_shorts <- apply(signal_mat, 1, function(s) {
    if (all(is.na(s))) return(rep(FALSE, N))
    rank_s <- rank(s, na.last = "keep")
    !is.na(rank_s) & rank_s <= n_top
  })

  # Turnover: fraction of long/short portfolio that changes
  long_turnover <- numeric(T_obs - 1)
  for (t in 1:(T_obs-1)) {
    prev_long <- daily_longs[, t]; curr_long <- daily_longs[, t+1]
    # Turnover = (entries + exits) / 2 as fraction of portfolio
    entries <- sum(curr_long & !prev_long); exits <- sum(prev_long & !curr_long)
    long_turnover[t] <- (entries + exits) / (2 * n_top)
  }

  # Signal autocorrelation
  signal_ac <- apply(signal_mat, 2, function(s) {
    s <- s[!is.na(s)]
    if (length(s) < 10) return(NA)
    acf(s, lag.max = 1, plot = FALSE)$acf[2]
  })

  list(
    daily_turnover = long_turnover,
    avg_turnover = mean(long_turnover, na.rm=TRUE),
    signal_autocorrelation = mean(signal_ac, na.rm=TRUE),
    implied_half_life_days = log(0.5) / log(mean(signal_ac, na.rm=TRUE)),
    # Breakeven cost: alpha / (2 * turnover * fee) must > 1
    breakeven_fee_bps = function(annual_alpha_bps) {
      annual_alpha_bps / (2 * mean(long_turnover) * 252)
    }
  )
}

# -----------------------------------------------------------------------------
# Long/Short Backtesting with Transaction Costs
# Simple dollar-neutral L/S: long top decile, short bottom decile
# Net return after proportional transaction costs
# -----------------------------------------------------------------------------
ls_backtest_with_costs <- function(signal_mat, return_mat, top_pct = 0.10,
                                    fee_bps = 10, rebalance_freq = 1) {
  T_obs <- nrow(signal_mat); N <- ncol(signal_mat)
  n_side <- max(1, floor(N * top_pct))
  fee <- fee_bps / 1e4

  net_returns <- numeric(T_obs - 1)
  prev_longs  <- rep(FALSE, N)
  prev_shorts <- rep(FALSE, N)

  for (t in 1:(T_obs - 1)) {
    s <- signal_mat[t, ]
    if (all(is.na(s))) { net_returns[t] <- 0; next }

    if ((t - 1) %% rebalance_freq == 0) {
      # Rebalance
      r_rank <- rank(-s, na.last = "keep")
      curr_longs  <- !is.na(r_rank) & r_rank <= n_side
      curr_shorts <- !is.na(r_rank) & r_rank >= (N - n_side + 1)

      # Transaction costs
      long_turnover  <- mean(curr_longs != prev_longs)
      short_turnover <- mean(curr_shorts != prev_shorts)
      tc <- (long_turnover + short_turnover) * fee

      prev_longs <- curr_longs; prev_shorts <- curr_shorts
    } else {
      curr_longs <- prev_longs; curr_shorts <- prev_shorts; tc <- 0
    }

    ret <- return_mat[t + 1, ]
    long_ret  <- if (sum(curr_longs) > 0)  mean(ret[curr_longs], na.rm=TRUE)  else 0
    short_ret <- if (sum(curr_shorts) > 0) mean(ret[curr_shorts], na.rm=TRUE) else 0

    net_returns[t] <- (long_ret - short_ret) / 2 - tc
  }

  # Performance metrics
  valid <- !is.na(net_returns)
  r <- net_returns[valid]
  ann_ret <- mean(r) * 252
  ann_vol <- sd(r) * sqrt(252)

  list(
    net_returns = net_returns,
    ann_return = ann_ret,
    ann_vol = ann_vol,
    sharpe = if (ann_vol > 0) ann_ret / ann_vol else NA,
    max_drawdown = {
      w <- cumprod(1 + r); min((w - cummax(w))/cummax(w))
    },
    win_rate = mean(r > 0),
    pct_rebalance_days = 1 / rebalance_freq,
    fee_bps = fee_bps
  )
}

# -----------------------------------------------------------------------------
# Multi-Signal IC-Weighted Ensemble with Shrinkage
# Combine K signals using their historical IC as weights, with shrinkage
# toward equal weighting to prevent overfitting to IC estimation error
# -----------------------------------------------------------------------------
ic_shrinkage_ensemble <- function(signal_list, return_mat, horizon = 1,
                                    lookback = 60, shrinkage = 0.5) {
  # signal_list: list of T x N signal matrices
  K <- length(signal_list)
  T_obs <- nrow(return_mat)

  ensemble_signal <- matrix(0, T_obs, ncol(return_mat))

  for (t in (lookback+1):T_obs) {
    if (t + horizon > T_obs) break

    # Compute IC for each signal over lookback window
    ic_k <- numeric(K)
    for (k in 1:K) {
      ics <- numeric(lookback)
      for (tau in (t-lookback):(t-1)) {
        s <- signal_list[[k]][tau, ]
        r <- return_mat[tau + horizon, ]
        valid <- !is.na(s) & !is.na(r)
        if (sum(valid) < 3) { ics[tau-(t-lookback)+1] <- NA; next }
        ics[tau-(t-lookback)+1] <- cor(s[valid], r[valid], method="spearman")
      }
      ic_k[k] <- mean(ics, na.rm=TRUE)
    }

    # Shrinkage: blend IC weights with equal weights
    w_ic    <- pmax(ic_k, 0)
    if (sum(w_ic) > 0) w_ic <- w_ic / sum(w_ic)
    w_equal <- rep(1/K, K)
    w_blend <- (1 - shrinkage) * w_ic + shrinkage * w_equal

    # Construct ensemble signal
    sig_combined <- matrix(0, 1, ncol(return_mat))
    for (k in 1:K) {
      s_k <- signal_list[[k]][t, ]
      s_k[is.na(s_k)] <- 0
      # Rank-normalize each signal
      s_norm <- rank(s_k) / (length(s_k) + 1)
      sig_combined <- sig_combined + w_blend[k] * s_norm
    }
    ensemble_signal[t, ] <- as.vector(sig_combined)
  }

  ensemble_signal
}

# -----------------------------------------------------------------------------
# Alpha Stability Analysis: IS vs OOS Sharpe comparison across walk-forward windows
# Detects whether strategy performance is persistent or lucky in-sample
# -----------------------------------------------------------------------------
alpha_stability_test <- function(returns, n_windows = 5, is_fraction = 0.6) {
  n <- length(returns)
  window_size <- floor(n / n_windows)

  is_sharpes  <- numeric(n_windows)
  oos_sharpes <- numeric(n_windows)

  for (w in 1:n_windows) {
    is_start  <- (w-1) * window_size + 1
    is_end    <- is_start + floor(window_size * is_fraction) - 1
    oos_start <- is_end + 1
    oos_end   <- min(w * window_size, n)

    if (oos_start > n) break

    r_is  <- returns[is_start:is_end]
    r_oos <- returns[oos_start:oos_end]

    is_sharpes[w]  <- mean(r_is) / sd(r_is) * sqrt(252)
    oos_sharpes[w] <- mean(r_oos) / sd(r_oos) * sqrt(252)
  }

  # Sharpe decay: OOS / IS ratio (< 1 = degradation, > 1 = improvement)
  decay_ratios <- oos_sharpes / (is_sharpes + 1e-10)

  list(
    is_sharpes = is_sharpes,
    oos_sharpes = oos_sharpes,
    decay_ratios = decay_ratios,
    avg_is_sharpe  = mean(is_sharpes),
    avg_oos_sharpe = mean(oos_sharpes),
    avg_decay_ratio = mean(decay_ratios),
    stability_interpretation = ifelse(
      mean(decay_ratios) > 0.7, "stable alpha",
      ifelse(mean(decay_ratios) > 0.4, "moderate decay", "severe overfitting")
    ),
    corr_is_oos = cor(is_sharpes, oos_sharpes)
  )
}

# Extended alpha testing example:
# neutral_sig <- neutralize_signal(raw_signal, mktcap_mat)
# rc_ic <- regime_conditional_ic(neutral_sig, return_mat, regime_labels)
# ls_bt <- ls_backtest_with_costs(neutral_sig, return_mat, fee_bps = 10)
# stab  <- alpha_stability_test(ls_bt$net_returns)
# ensemble <- ic_shrinkage_ensemble(list(sig1, sig2, sig3), return_mat)
