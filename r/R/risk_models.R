# =============================================================================
# risk_models.R
# Comprehensive risk modeling for crypto/quant portfolios
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Risk management is the foundation of systematic trading.
# VaR (Value at Risk) answers "how much can I lose at X% confidence?".
# Expected Shortfall (ES/CVaR) answers "given I lose more than VaR, how much?".
# ES is more conservative and increasingly required by regulation (Basel III).
# =============================================================================

# -----------------------------------------------------------------------------
# 1. HISTORICAL VaR AND ES (NON-PARAMETRIC)
# -----------------------------------------------------------------------------

#' Historical simulation VaR
#' Pure non-parametric: uses empirical distribution of historical returns
#' @param returns numeric vector of portfolio returns
#' @param alpha confidence level (e.g., 0.05 for 95% VaR)
historical_var <- function(returns, alpha = 0.05) {
  var_est <- quantile(returns, probs = alpha)
  -var_est  # return as positive loss
}

#' Historical Expected Shortfall (ES / CVaR)
#' ES = mean of returns below VaR threshold
historical_es <- function(returns, alpha = 0.05) {
  threshold <- quantile(returns, probs = alpha)
  es_est <- mean(returns[returns <= threshold])
  -es_est
}

#' Full historical VaR/ES table at multiple confidence levels
historical_var_table <- function(returns, alphas = c(0.10, 0.05, 0.025, 0.01)) {
  df <- data.frame(
    alpha      = alphas,
    confidence = paste0(100*(1-alphas), "%"),
    VaR        = sapply(alphas, function(a) historical_var(returns, a)),
    ES         = sapply(alphas, function(a) historical_es(returns, a))
  )
  cat("=== Historical VaR / ES Table ===\n")
  print(df)
  invisible(df)
}

# -----------------------------------------------------------------------------
# 2. PARAMETRIC VAR
# -----------------------------------------------------------------------------

#' Normal parametric VaR
#' Assumes returns are normally distributed (often too optimistic for crypto)
parametric_var_normal <- function(returns, alpha = 0.05) {
  mu    <- mean(returns)
  sigma <- sd(returns)
  var_est <- -(mu + sigma * qnorm(alpha))
  list(VaR = var_est, mu = mu, sigma = sigma)
}

#' Student-t parametric VaR
#' More realistic for fat-tailed crypto returns
#' @param nu degrees of freedom (estimated if NULL)
parametric_var_t <- function(returns, alpha = 0.05, nu = NULL) {
  mu    <- mean(returns)
  sigma <- sd(returns)

  if (is.null(nu)) {
    # Estimate nu via kurtosis matching: kurt = 6/(nu-4) => nu = 6/kurt + 4
    k4 <- mean((returns-mu)^4) / sigma^4  # excess kurtosis (sample)
    nu <- if (k4 > 0) max(4.1, 6/k4 + 4) else 30
  }

  # For t-distribution: VaR = mu + sigma * t_{alpha,nu} * sqrt((nu-2)/nu)
  # Scale to match variance: need to adjust sigma
  scale <- sigma * sqrt((nu-2)/nu)  # scale parameter for the t distribution
  var_est <- -(mu + scale * qt(alpha, df=nu))
  es_factor <- dt(qt(alpha, df=nu), df=nu) / alpha * (nu + qt(alpha,df=nu)^2) / (nu-1)
  es_est <- -(mu - scale * es_factor)

  list(VaR=var_est, ES=es_est, mu=mu, sigma=sigma, nu=nu)
}

#' Skewed-t VaR (Fernandez-Steel skewed-t distribution)
#' Captures both fat tails AND skewness -- crypto often has negative skew (crash risk)
parametric_var_skewed_t <- function(returns, alpha = 0.05) {
  mu    <- mean(returns)
  sigma <- sd(returns)

  # Estimate skewness
  skew <- mean((returns-mu)^3) / sigma^3

  # Simple skew-adjustment to VaR using Cornish-Fisher (see below)
  # For a full skewed-t, we use Cornish-Fisher as approximation here
  k4   <- mean((returns-mu)^4) / sigma^4 - 3  # excess kurtosis
  z    <- qnorm(alpha)
  z_cf <- z + (z^2-1)*skew/6 + (z^3-3*z)*k4/24 - (2*z^3-5*z)*skew^2/36
  var_est <- -(mu + sigma * z_cf)

  list(VaR=var_est, mu=mu, sigma=sigma, skew=skew, kurt=k4)
}

# -----------------------------------------------------------------------------
# 3. CORNISH-FISHER VAR (MOMENT-ADJUSTED)
# -----------------------------------------------------------------------------

#' Cornish-Fisher VaR expansion
#' Modified quantile that accounts for skewness and excess kurtosis
#' Very useful for crypto which has both features
#' @param returns return vector
#' @param alpha confidence level
cornish_fisher_var <- function(returns, alpha = 0.05) {
  mu    <- mean(returns)
  sigma <- sd(returns)
  n     <- length(returns)

  # Sample moments
  skew_hat <- mean((returns-mu)^3) / sigma^3
  kurt_hat <- mean((returns-mu)^4) / sigma^4 - 3  # excess kurtosis

  # Cornish-Fisher expansion
  z_alpha <- qnorm(alpha)
  z_cf <- z_alpha +
          (z_alpha^2 - 1) * skew_hat / 6 +
          (z_alpha^3 - 3*z_alpha) * kurt_hat / 24 -
          (2*z_alpha^3 - 5*z_alpha) * skew_hat^2 / 36

  var_cf  <- -(mu + sigma * z_cf)
  var_n   <- -(mu + sigma * z_alpha)  # naive normal VaR for comparison
  uplift  <- var_cf / var_n - 1

  cat("=== Cornish-Fisher VaR ===\n")
  cat(sprintf("Skewness: %.4f, Excess kurtosis: %.4f\n", skew_hat, kurt_hat))
  cat(sprintf("Normal VaR (%.0f%%): %.5f\n", (1-alpha)*100, var_n))
  cat(sprintf("CF VaR    (%.0f%%): %.5f\n", (1-alpha)*100, var_cf))
  cat(sprintf("CF uplift: +%.1f%%\n", 100*uplift))

  list(VaR=var_cf, var_normal=var_n, z_cf=z_cf, skew=skew_hat, kurt=kurt_hat)
}

# -----------------------------------------------------------------------------
# 4. EXPECTED SHORTFALL AND DECOMPOSITION
# -----------------------------------------------------------------------------

#' ES decomposition by component asset
#' ES(portfolio) = sum_i w_i * ES(asset_i | portfolio in tail)
#' This gives the marginal contribution of each asset to portfolio tail loss
#' @param returns_mat T x N return matrix
#' @param weights portfolio weights
#' @param alpha confidence level
es_decomposition <- function(returns_mat, weights, alpha = 0.05) {
  n     <- nrow(returns_mat)
  N     <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  port_ret  <- returns_mat %*% weights
  threshold <- quantile(port_ret, alpha)
  tail_idx  <- which(port_ret <= threshold)

  # Component ES: E[r_i | portfolio in tail]
  comp_es <- sapply(seq_len(N), function(i) {
    -mean(returns_mat[tail_idx, i])
  })

  # Marginal contribution to ES
  marginal_es <- weights * comp_es
  pct_contrib <- marginal_es / sum(marginal_es) * 100

  total_es <- -mean(port_ret[tail_idx])

  df <- data.frame(
    asset        = asset_names,
    weight       = round(weights, 4),
    component_ES = round(comp_es, 5),
    marginal_ES  = round(marginal_es, 5),
    pct_contrib  = round(pct_contrib, 2)
  )
  df <- df[order(-df$pct_contrib), ]

  cat("=== Expected Shortfall Decomposition ===\n")
  cat(sprintf("Portfolio ES (%.0f%%): %.5f\n", (1-alpha)*100, total_es))
  cat(sprintf("Tail observations: %d (%.1f%%)\n", length(tail_idx), 100*length(tail_idx)/n))
  cat("\nContributions:\n")
  print(df)

  invisible(list(total_es=total_es, components=df, tail_idx=tail_idx))
}

# -----------------------------------------------------------------------------
# 5. STRESS TESTING
# -----------------------------------------------------------------------------

#' Define historical stress scenarios and compute portfolio impact
#' @param returns_mat T x N matrix of historical returns
#' @param weights portfolio weights
#' @param scenarios list of named scenario specifications
stress_test <- function(returns_mat, weights,
                         scenarios = NULL) {
  N <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  # Default crypto stress scenarios
  if (is.null(scenarios)) {
    scenarios <- list(
      "BTC_crash_30pct" = setNames(c(-0.30, -0.25, -0.28, -0.35, -0.20), asset_names[1:5]),
      "BTC_crash_50pct" = setNames(c(-0.50, -0.45, -0.48, -0.55, -0.40), asset_names[1:5]),
      "BTC_rally_40pct" = setNames(c(+0.40, +0.50, +0.45, +0.60, +0.35), asset_names[1:5]),
      "Correlation_spike"= setNames(rep(-0.15, 5), asset_names[1:5])
    )
    # Trim to available assets
    scenarios <- lapply(scenarios, function(s) s[names(s) %in% asset_names])
  }

  results <- lapply(names(scenarios), function(sc_name) {
    sc <- scenarios[[sc_name]]
    # Align with weight vector
    r_vec <- numeric(N)
    names(r_vec) <- asset_names
    for (a in names(sc)) {
      if (a %in% asset_names) r_vec[a] <- sc[a]
    }
    port_pnl <- sum(weights * r_vec)
    list(scenario=sc_name, r_vec=r_vec, portfolio_pnl=port_pnl)
  })

  df <- data.frame(
    scenario = sapply(results, `[[`, "scenario"),
    portfolio_pnl = sapply(results, `[[`, "portfolio_pnl")
  )
  df$loss = -df$portfolio_pnl
  df <- df[order(df$portfolio_pnl), ]

  cat("=== Stress Test Results ===\n")
  print(df[, c("scenario", "portfolio_pnl", "loss")])

  invisible(list(results=results, summary=df))
}

#' Historical worst periods analysis
#' Identifies the worst N historical windows for the portfolio
worst_periods <- function(returns_mat, weights, window=1, n_worst=10) {
  port_ret  <- (returns_mat %*% weights)
  n <- length(port_ret)

  if (window > 1) {
    roll_ret <- as.numeric(stats::filter(port_ret, rep(1, window), sides=1))
    roll_ret[1:window] <- NA
  } else {
    roll_ret <- as.numeric(port_ret)
  }

  valid_idx <- which(!is.na(roll_ret))
  worst_idx <- valid_idx[order(roll_ret[valid_idx])][1:n_worst]

  df <- data.frame(
    rank = seq_len(n_worst),
    t    = worst_idx,
    ret  = roll_ret[worst_idx],
    loss = -roll_ret[worst_idx]
  )

  cat(sprintf("=== Top %d Worst %d-Period Returns ===\n", n_worst, window))
  print(df)

  invisible(df)
}

# -----------------------------------------------------------------------------
# 6. DRAWDOWN RISK METRICS
# -----------------------------------------------------------------------------

#' Comprehensive drawdown analysis
#' @param returns portfolio return series
#' @return list of drawdown statistics
drawdown_analysis <- function(returns) {
  n        <- length(returns)
  # Cumulative return series
  cum_ret  <- cumprod(1 + returns)
  cum_peak <- cummax(cum_ret)
  drawdown <- cum_ret / cum_peak - 1

  # Max drawdown
  max_dd <- min(drawdown)
  max_dd_end   <- which.min(drawdown)
  max_dd_start <- which.max(cum_ret[1:max_dd_end])
  max_dd_dur   <- max_dd_end - max_dd_start

  # Recovery: how long to recover from max drawdown
  if (max_dd_end < n) {
    recovery_end <- max_dd_end + which(cum_ret[(max_dd_end+1):n] >= cum_peak[max_dd_end])[1]
    recovery_dur <- if (!is.na(recovery_end)) recovery_end - max_dd_end else NA
  } else {
    recovery_dur <- NA
  }

  # Calmar ratio: annualized return / max drawdown
  ann_return <- mean(returns) * 252
  calmar     <- ann_return / abs(max_dd)

  # Pain index: average drawdown depth
  pain_index <- mean(abs(drawdown))

  # Ulcer index: RMS of drawdown
  ulcer_index <- sqrt(mean(drawdown^2))

  # Sterling ratio: ann_return / avg of 3 worst drawdowns
  dd_depths  <- sort(drawdown)[1:min(3, length(drawdown))]
  sterling   <- ann_return / abs(mean(dd_depths))

  cat("=== Drawdown Analysis ===\n")
  cat(sprintf("Max Drawdown:    %.2f%%\n", 100*max_dd))
  cat(sprintf("DD Duration:     %d periods\n", max_dd_dur))
  cat(sprintf("Recovery Time:   %s\n", ifelse(is.na(recovery_dur), "not recovered",
                                               paste(recovery_dur, "periods"))))
  cat(sprintf("Calmar Ratio:    %.3f\n", calmar))
  cat(sprintf("Pain Index:      %.4f\n", pain_index))
  cat(sprintf("Ulcer Index:     %.4f\n", ulcer_index))
  cat(sprintf("Sterling Ratio:  %.3f\n", sterling))

  invisible(list(drawdown=drawdown, max_dd=max_dd,
                 max_dd_start=max_dd_start, max_dd_end=max_dd_end,
                 max_dd_dur=max_dd_dur, recovery_dur=recovery_dur,
                 calmar=calmar, pain_index=pain_index,
                 ulcer_index=ulcer_index, sterling=sterling))
}

# -----------------------------------------------------------------------------
# 7. CONCENTRATION RISK (HERFINDAHL INDEX)
# -----------------------------------------------------------------------------

#' Herfindahl-Hirschman Index (HHI) for portfolio concentration
#' HHI = sum(w_i^2), range [1/N, 1]
#' HHI = 1/N: perfectly diversified (equal weights)
#' HHI = 1:   fully concentrated (single asset)
#' Effective N = 1/HHI
herfindahl_concentration <- function(weights) {
  w_pos  <- pmax(weights, 0)
  w_pos  <- w_pos / sum(w_pos)  # normalize
  N      <- length(w_pos)
  hhi    <- sum(w_pos^2)
  eff_N  <- 1 / hhi
  # Normalized HHI: 0 = equal, 1 = fully concentrated
  hhi_norm <- (hhi - 1/N) / (1 - 1/N)

  cat("=== Concentration Risk ===\n")
  cat(sprintf("N assets: %d\n", N))
  cat(sprintf("HHI: %.4f  (min=%.4f for equal weights)\n", hhi, 1/N))
  cat(sprintf("Effective N: %.1f\n", eff_N))
  cat(sprintf("Normalized HHI: %.4f (0=diversified, 1=concentrated)\n", hhi_norm))

  invisible(list(hhi=hhi, eff_N=eff_N, hhi_norm=hhi_norm, weights=w_pos))
}

# -----------------------------------------------------------------------------
# 8. MARGINAL VaR AND COMPONENT VaR
# -----------------------------------------------------------------------------

#' Marginal VaR: change in portfolio VaR per unit increase in asset allocation
#' MVaR_i = rho_{i,p} * VaR_p / w_i  (delta-normal approximation)
#' Component VaR: w_i * MVaR_i = fraction of total VaR from asset i
#' @param returns_mat T x N return matrix
#' @param weights portfolio weights
#' @param alpha VaR confidence level
marginal_var <- function(returns_mat, weights, alpha = 0.05) {
  N <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  Sigma <- cov(returns_mat)
  mu    <- colMeans(returns_mat)

  # Portfolio variance and VaR
  port_var <- as.numeric(t(weights) %*% Sigma %*% weights)
  port_std  <- sqrt(port_var)
  port_mu   <- sum(weights * mu)
  z_alpha   <- qnorm(alpha)
  var_port  <- -(port_mu + port_std * z_alpha)

  # Marginal contributions: sigma_{i,p} / sigma_p
  cov_iP <- as.numeric(Sigma %*% weights)  # covariance of each asset with portfolio
  beta_i  <- cov_iP / (port_var + 1e-10)   # relative to portfolio variance

  # Marginal VaR
  mvar_i <- -z_alpha * cov_iP / port_std
  # Or: MVaR_i = -z_alpha * beta_i * port_std

  # Component VaR: w_i * MVaR_i
  comp_var_i <- weights * mvar_i
  pct_contrib <- comp_var_i / sum(comp_var_i) * 100

  df <- data.frame(
    asset = asset_names,
    weight = round(weights, 4),
    cov_with_port = round(cov_iP, 6),
    beta = round(beta_i, 4),
    marginal_var = round(mvар_i, 5),
    component_var = round(comp_var_i, 5),
    pct_contrib = round(pct_contrib, 2)
  )
  df <- df[order(-df$pct_contrib), ]

  cat("=== Marginal and Component VaR ===\n")
  cat(sprintf("Portfolio VaR (%.0f%%): %.5f\n", (1-alpha)*100, var_port))
  cat("\nRisk Attribution:\n")
  print(df)
  cat(sprintf("Sum of component VaRs: %.5f (check = port VaR: %.5f)\n",
              sum(comp_var_i), var_port))

  invisible(list(var_port=var_port, marginal=df, Sigma=Sigma))
}

# -----------------------------------------------------------------------------
# 9. COMPREHENSIVE RISK REPORT
# -----------------------------------------------------------------------------

#' Full risk model report for a portfolio
#' @param returns_mat T x N return matrix
#' @param weights portfolio weights
#' @param alpha VaR confidence level
run_risk_models <- function(returns_mat, weights=NULL, alpha=0.05) {
  N   <- ncol(returns_mat)
  T_obs <- nrow(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))
  if (is.null(weights)) weights <- rep(1/N, N)

  cat("=============================================================\n")
  cat("RISK MODEL REPORT\n")
  cat(sprintf("Assets: %s\n", paste(asset_names, collapse=", ")))
  cat(sprintf("Observations: %d, Weights: %s\n",
              T_obs, paste(round(weights, 3), collapse=", ")))
  cat("=============================================================\n\n")

  port_ret <- as.numeric(returns_mat %*% weights)

  # 1. Historical VaR table
  hist_table <- historical_var_table(port_ret)

  # 2. Normal VaR
  cat("\n--- Parametric VaR ---\n")
  var_n <- parametric_var_normal(port_ret, alpha)
  var_t <- parametric_var_t(port_ret, alpha)
  cat(sprintf("Normal VaR (%.0f%%): %.5f\n", (1-alpha)*100, var_n$VaR))
  cat(sprintf("t-dist VaR (%.0f%%, nu=%.1f): %.5f\n", (1-alpha)*100, var_t$nu, var_t$VaR))
  cat(sprintf("t-dist ES  (%.0f%%): %.5f\n", (1-alpha)*100, var_t$ES))

  # 3. Cornish-Fisher VaR
  cat("\n")
  cf_var <- cornish_fisher_var(port_ret, alpha)

  # 4. ES decomposition
  cat("\n")
  es_decomp <- es_decomposition(returns_mat, weights, alpha)

  # 5. Marginal VaR
  cat("\n")
  mvar_res <- marginal_var(returns_mat, weights, alpha)

  # 6. Drawdown analysis
  cat("\n")
  dd_res <- drawdown_analysis(port_ret)

  # 7. Concentration
  cat("\n")
  conc_res <- herfindahl_concentration(weights)

  # 8. Stress tests
  if (N >= 2) {
    cat("\n")
    # Auto-build scenarios from asset names
    sc_list <- list()
    sc_list[["mild_crash"]] <- setNames(rep(-0.10, N), asset_names)
    sc_list[["severe_crash"]] <- setNames(rep(-0.30, N), asset_names)
    # BTC-heavy shock if "BTC" in names
    if ("BTC" %in% asset_names) {
      sc <- setNames(rep(-0.05, N), asset_names)
      sc["BTC"] <- -0.40; sc_list[["BTC_crash"]] <- sc
    }
    stress_res <- stress_test(returns_mat, weights, sc_list)
  }

  cat("\n=== RISK SUMMARY ===\n")
  cat(sprintf("Historical 95%% VaR: %.5f\n", historical_var(port_ret, 0.05)))
  cat(sprintf("Historical 95%% ES:  %.5f\n", historical_es(port_ret, 0.05)))
  cat(sprintf("Cornish-Fisher VaR: %.5f\n", cf_var$VaR))
  cat(sprintf("Max Drawdown:       %.2f%%\n", 100*dd_res$max_dd))
  cat(sprintf("Effective N:        %.1f / %d\n", conc_res$eff_N, N))

  invisible(list(hist_table=hist_table, cf_var=cf_var,
                 es_decomp=es_decomp, mvar=mvar_res,
                 drawdown=dd_res, concentration=conc_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 500; N <- 5
# asset_names <- c("BTC","ETH","BNB","SOL","ADA")
# Sigma <- matrix(c(
#   0.0016, 0.0010, 0.0007, 0.0009, 0.0006,
#   0.0010, 0.0012, 0.0006, 0.0008, 0.0005,
#   0.0007, 0.0006, 0.0009, 0.0005, 0.0004,
#   0.0009, 0.0008, 0.0005, 0.0011, 0.0006,
#   0.0006, 0.0005, 0.0004, 0.0006, 0.0008), 5, 5)
# mu_v <- c(0.001, 0.0008, 0.0006, 0.0009, 0.0005)
# L <- chol(Sigma)
# returns_mat <- matrix(rnorm(n*N), n, N) %*% L + matrix(rep(mu_v, n), n, N, byrow=TRUE)
# colnames(returns_mat) <- asset_names
# weights <- c(0.40, 0.25, 0.15, 0.12, 0.08)
# result <- run_risk_models(returns_mat, weights, alpha=0.05)

# =============================================================================
# EXTENDED RISK MODELS: VaR Term Structure, GPD Tail Fitting, Spectral Risk,
# Robust Covariance, Liquidity-Adjusted VaR, Scenario Analysis
# =============================================================================

# -----------------------------------------------------------------------------
# VaR Term Structure: scale VaR estimates across horizons using square-root-of-time
# and empirically corrected scaling for autocorrelated returns
# -----------------------------------------------------------------------------
var_term_structure <- function(returns, alpha = 0.05, horizons = c(1,5,10,21,63,252)) {
  # Square-root-of-time rule: VaR(h) ~ VaR(1) * sqrt(h)
  # But with autocorrelation, the scaling differs:
  # Var(sum of h returns) = h*sigma^2 + 2*sum_{k=1}^{h-1}(h-k)*gamma(k)
  # where gamma(k) is the autocovariance at lag k
  n <- length(returns)
  var1 <- quantile(returns, alpha, names = FALSE)

  # Estimate autocovariance structure
  max_lag <- min(20, floor(n/5))
  acov <- acf(returns, lag.max = max_lag, type = "covariance", plot = FALSE)$acf[,,1]

  result <- data.frame(horizon = horizons, var_sqrt = NA, var_adjusted = NA,
                       es_sqrt = NA, annualized_vol = NA)

  for (i in seq_along(horizons)) {
    h <- horizons[i]
    # Square-root scaling
    result$var_sqrt[i] <- var1 * sqrt(h)

    # Autocorrelation-adjusted scaling
    scale_factor <- h * acov[1]
    for (k in 1:min(h-1, max_lag)) {
      scale_factor <- scale_factor + 2 * (h - k) * acov[k + 1]
    }
    scale_factor <- max(scale_factor, 0)
    ratio <- sqrt(scale_factor / acov[1])
    result$var_adjusted[i] <- var1 * ratio

    # ES scaling (same logic applied to tail mean)
    tail_obs <- returns[returns <= var1]
    if (length(tail_obs) > 0) {
      es1 <- mean(tail_obs)
      result$es_sqrt[i] <- es1 * sqrt(h)
    }

    # Annualized volatility for reference
    result$annualized_vol[i] <- sd(returns) * sqrt(252 / h)
  }

  result$var_sqrt_pct     <- result$var_sqrt * 100
  result$var_adjusted_pct <- result$var_adjusted * 100
  result
}

# -----------------------------------------------------------------------------
# GPD Tail Fitting: Peaks-Over-Threshold (POT) method for extreme tail modeling
# Estimates shape (xi) and scale (beta) of Generalized Pareto Distribution
# Xi > 0 = heavy tail (Frechet), xi = 0 = light tail (Gumbel), xi < 0 = bounded
# -----------------------------------------------------------------------------
fit_gpd_tail <- function(returns, threshold_quantile = 0.05) {
  # Extract exceedances below threshold (for loss tail)
  threshold <- quantile(returns, threshold_quantile, names = FALSE)
  losses <- -returns  # Work with losses (positive = bad)
  u <- -threshold
  exceedances <- losses[losses > u] - u  # Excess losses over threshold

  if (length(exceedances) < 10) {
    warning("Too few exceedances for GPD fitting")
    return(NULL)
  }

  n_total <- length(returns)
  n_exceed <- length(exceedances)
  p_exceed  <- n_exceed / n_total

  # GPD log-likelihood
  gpd_loglik <- function(params) {
    xi <- params[1]; beta <- params[2]
    if (beta <= 0) return(-1e10)
    if (xi != 0) {
      arg <- 1 + xi * exceedances / beta
      if (any(arg <= 0)) return(-1e10)
      ll <- -n_exceed * log(beta) - (1 + 1/xi) * sum(log(arg))
    } else {
      ll <- -n_exceed * log(beta) - sum(exceedances) / beta
    }
    ll
  }

  # Method of moments starting values
  m1 <- mean(exceedances); m2 <- var(exceedances)
  xi0 <- 0.5 * (1 - m1^2 / m2)
  beta0 <- 0.5 * m1 * (1 + m1^2 / m2)

  opt <- tryCatch(
    optim(c(xi0, beta0), gpd_loglik, control = list(fnscale = -1),
          method = "Nelder-Mead"),
    error = function(e) list(par = c(xi0, beta0), convergence = 1)
  )

  xi_hat   <- opt$par[1]
  beta_hat <- opt$par[2]

  # GPD VaR and ES at probability p
  gpd_var <- function(p) {
    # P(Loss > VaR) = p
    # VaR = u + beta/xi * ((p/p_exceed)^(-xi) - 1) for xi != 0
    if (p >= p_exceed) return(u)
    ratio <- p / p_exceed
    if (abs(xi_hat) < 1e-6) {
      var_val <- u + beta_hat * log(1 / ratio)
    } else {
      var_val <- u + beta_hat / xi_hat * (ratio^(-xi_hat) - 1)
    }
    -var_val  # Return as negative return (loss)
  }

  gpd_es <- function(p) {
    v <- gpd_var(p)
    # ES = VaR + (beta + xi*(VaR - u)) / (1 - xi) for xi < 1
    excess_at_var <- (-v) - u
    if (abs(xi_hat) < 1e-6) {
      es_val <- (-v) + beta_hat
    } else if (xi_hat < 1) {
      es_val <- (-v) + (beta_hat + xi_hat * excess_at_var) / (1 - xi_hat)
    } else {
      es_val <- Inf
    }
    -es_val
  }

  list(
    xi = xi_hat, beta = beta_hat,
    threshold = threshold, n_exceedances = n_exceed,
    p_exceed = p_exceed,
    convergence = opt$convergence,
    var_1pct = gpd_var(0.01),
    var_5pct = gpd_var(0.05),
    es_1pct  = gpd_es(0.01),
    es_5pct  = gpd_es(0.05),
    tail_index = if(xi_hat > 0) 1/xi_hat else Inf,
    gpd_var_fn = gpd_var,
    gpd_es_fn  = gpd_es
  )
}

# -----------------------------------------------------------------------------
# Spectral Risk Measure (SRM): weighted integral of quantile function
# Weights represent risk aversion spectrum; distortion risk measure family
# Includes Wang transform, power distortion, and exponential distortion
# -----------------------------------------------------------------------------
spectral_risk_measure <- function(returns, distortion = "exponential",
                                   gamma = 4, n_quantiles = 1000) {
  # Quantile function at evenly spaced probabilities
  probs <- seq(1/n_quantiles, 1 - 1/n_quantiles, length.out = n_quantiles)
  q_vals <- quantile(returns, probs, names = FALSE)

  # Distortion functions (phi maps [0,1] -> [0,1], risk weights)
  if (distortion == "exponential") {
    # Exponential distortion: more weight to extreme losses
    # phi(u) = (1 - exp(-gamma*u)) / (1 - exp(-gamma))
    phi <- (1 - exp(-gamma * probs)) / (1 - exp(-gamma))
  } else if (distortion == "power") {
    # Power distortion: phi(u) = u^(1/gamma), gamma > 1 = risk averse
    phi <- probs^(1 / gamma)
  } else if (distortion == "wang") {
    # Wang transform: phi(u) = Phi(Phi^{-1}(u) + lambda), lambda > 0 = risk loading
    lambda <- gamma * 0.5
    phi <- pnorm(qnorm(probs) + lambda)
  } else if (distortion == "cvar") {
    # CVaR is SRM with phi(u) = 0 for u < alpha, 1/(1-alpha) for u >= alpha
    alpha <- 1 / gamma  # gamma encodes confidence level
    phi <- ifelse(probs < alpha, 0, 1/(1 - alpha))
  } else {
    stop("Unknown distortion: choose exponential, power, wang, or cvar")
  }

  # SRM = integral of q(u) * d(phi(u)) = sum of q(u) * delta_phi
  # Discretized: SRM = sum_i q(p_i) * (phi(p_{i+1}) - phi(p_{i-1})) / 2
  dphi <- diff(phi) / diff(probs)
  # Trapezoidal integration
  mid_q <- (q_vals[-1] + q_vals[-n_quantiles]) / 2
  mid_dp <- diff(probs)
  srm <- sum(mid_q * diff(phi))

  # Also compute the implied risk weights (spectral function)
  spectral_weights <- diff(phi) / sum(diff(phi))

  list(
    srm = srm,
    distortion = distortion,
    gamma = gamma,
    es_5pct = mean(returns[returns <= quantile(returns, 0.05)]),
    srm_premium = srm - mean(returns),  # extra charge vs expected loss
    spectral_weights_summary = summary(spectral_weights)
  )
}

# -----------------------------------------------------------------------------
# Robust Covariance Estimation: Minimum Covariance Determinant (MCD)
# and Ledoit-Wolf style shrinkage to handle outliers in crypto returns
# Heavy-tailed distributions invalidate sample covariance; MCD uses subset
# -----------------------------------------------------------------------------
robust_covariance <- function(returns_mat, method = "mcd", h_frac = 0.75) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)

  if (method == "mcd") {
    # Approximate MCD via C-step algorithm
    # Start with random subset of size h, iterate to reduce determinant
    h <- floor(h_frac * n)
    best_det <- Inf; best_cov <- NULL; best_mu <- NULL

    set.seed(42)
    for (trial in 1:10) {
      # Random initial subset
      idx <- sample(n, h)
      for (step in 1:5) {
        sub <- returns_mat[idx, , drop = FALSE]
        mu_sub  <- colMeans(sub)
        cov_sub <- cov(sub)

        # Mahalanobis distances from current center
        cov_inv <- tryCatch(solve(cov_sub), error = function(e) {
          solve(cov_sub + diag(1e-6, p))
        })
        diffs <- sweep(returns_mat, 2, mu_sub)
        mah <- rowSums((diffs %*% cov_inv) * diffs)

        # New subset: h observations with smallest Mahalanobis distance
        idx <- order(mah)[1:h]
      }
      det_val <- det(cov_sub)
      if (det_val < best_det) {
        best_det <- det_val; best_cov <- cov_sub; best_mu <- mu_sub
      }
    }

    # Consistency factor for asymptotic unbiasedness
    c_factor <- 1 / qchisq(h_frac, p)
    robust_cov <- best_cov * c_factor
    robust_mu  <- best_mu

  } else if (method == "biweight") {
    # Biweight (Tukey) location and scatter
    # Iteratively reweighted using psi function
    mu0 <- apply(returns_mat, 2, median)
    cov0 <- cov(returns_mat)

    for (iter in 1:20) {
      cov_inv <- tryCatch(solve(cov0), error = function(e) solve(cov0 + diag(1e-6,p)))
      diffs <- sweep(returns_mat, 2, mu0)
      mah <- sqrt(rowSums((diffs %*% cov_inv) * diffs))
      c_val <- sqrt(qchisq(0.975, p))
      u <- mah / c_val
      # Biweight weights: w(u) = (1-u^2)^2 * I(|u| < 1)
      w <- ifelse(abs(u) < 1, (1 - u^2)^2, 0)
      if (sum(w) == 0) break
      mu_new <- colSums(sweep(returns_mat, 1, w, "*")) / sum(w)
      diffs_new <- sweep(returns_mat, 2, mu_new)
      cov_new <- (t(diffs_new) %*% diag(w) %*% diffs_new) / sum(w)
      if (max(abs(mu_new - mu0)) < 1e-8) break
      mu0 <- mu_new; cov0 <- cov_new
    }
    robust_cov <- cov0; robust_mu <- mu0

  } else {
    stop("method must be mcd or biweight")
  }

  # Compare with sample covariance via condition number
  sample_cov <- cov(returns_mat)
  cond_sample  <- kappa(sample_cov)
  cond_robust  <- kappa(robust_cov)

  list(
    robust_cov = robust_cov, robust_mu = robust_mu,
    sample_cov = sample_cov, sample_mu = colMeans(returns_mat),
    cond_sample = cond_sample, cond_robust = cond_robust,
    method = method,
    eigenvalues_robust = eigen(robust_cov, only.values = TRUE)$values
  )
}

# -----------------------------------------------------------------------------
# Liquidity-Adjusted VaR (LVaR): accounts for bid-ask spread and market impact
# when estimating the true cost of liquidating a position under stress
# Uses Almgren-Chriss framework: LVaR = VaR + Liquidity Cost
# -----------------------------------------------------------------------------
lvar <- function(returns, weights, market_caps = NULL, adv = NULL,
                 position_value = 1e6, alpha = 0.05, horizon_days = 1) {
  n <- length(weights)
  weights <- weights / sum(weights)

  # Base VaR
  port_returns <- if (is.matrix(returns)) {
    as.vector(returns %*% weights)
  } else {
    returns
  }
  base_var <- -quantile(port_returns, alpha, names = FALSE)

  # Bid-ask spread component: half-spread * position size
  # Typical crypto spreads: 0.05%-0.2% for liquid, 0.5%-2% for illiquid
  if (is.null(adv)) {
    # Use proxy: assume ADV proportional to market cap
    if (is.null(market_caps)) {
      adv <- rep(position_value * 10, n)  # default: position is 10% of daily volume
    } else {
      adv <- market_caps * 0.005  # 0.5% of mktcap trades daily (rough proxy)
    }
  }

  # Participation rate: fraction of ADV we trade
  asset_values <- weights * position_value
  participation <- asset_values / adv

  # Almgren-Chriss market impact: temporary impact = eta * sigma * sqrt(v/ADV)
  # eta ≈ 0.1 for typical US markets; higher for crypto
  eta <- 0.2  # crypto market impact coefficient
  sigmas <- if (is.matrix(returns)) apply(returns, 2, sd) else rep(sd(returns), n)

  # Temporary impact cost per asset
  temp_impact <- eta * sigmas * sqrt(participation) * asset_values

  # Permanent impact: gamma * sigma^2 * v / ADV (price slippage that stays)
  gamma_perm <- 0.1
  perm_impact <- gamma_perm * sigmas^2 * asset_values / adv * asset_values

  # Bid-ask spread cost (assume spread = k * sigma^0.5 for crypto)
  k_spread <- 0.002  # empirical crypto spread coefficient
  spread_cost <- k_spread * sqrt(sigmas) * asset_values

  total_liquidity_cost <- sum(temp_impact + perm_impact + spread_cost)

  # Liquidation horizon: time to liquidate without excessive impact
  # Optimal: T* = sqrt(gamma * X^2 / (2 * eta * sigma^2 * ADV))
  opt_horizon <- sqrt(gamma_perm * sum(asset_values^2) /
                        (2 * eta * sum(sigmas^2 * adv)))

  lvar_val <- base_var * position_value + total_liquidity_cost

  list(
    base_var = base_var,
    base_var_dollar = base_var * position_value,
    liquidity_cost = total_liquidity_cost,
    lvar_dollar = lvar_val,
    lvar_pct = lvar_val / position_value,
    spread_cost = sum(spread_cost),
    temp_impact_cost = sum(temp_impact),
    perm_impact_cost = sum(perm_impact),
    optimal_liquidation_horizon_days = opt_horizon,
    participation_rates = participation
  )
}

# -----------------------------------------------------------------------------
# Expected Shortfall Contribution: decompose ES by asset for risk budgeting
# Uses Euler decomposition: ES(portfolio) = sum_i w_i * partial_ES_i
# Critical for understanding which crypto positions drive tail risk
# -----------------------------------------------------------------------------
es_contribution <- function(returns_mat, weights, alpha = 0.05) {
  weights <- weights / sum(weights)
  port_returns <- as.vector(returns_mat %*% weights)

  # Tail threshold
  var_val <- quantile(port_returns, alpha, names = FALSE)
  tail_idx <- port_returns <= var_val
  n_tail   <- sum(tail_idx)

  if (n_tail == 0) stop("No tail observations found")

  # Euler decomposition: ES contribution of asset i
  # partial_ES_i = E[r_i | r_portfolio <= VaR] * w_i
  es_contribs <- numeric(ncol(returns_mat))
  for (i in 1:ncol(returns_mat)) {
    es_contribs[i] <- weights[i] * mean(returns_mat[tail_idx, i])
  }

  port_es <- mean(port_returns[tail_idx])

  # Marginal ES (sensitivity of ES to weight changes)
  delta <- 1e-4
  marginal_es <- numeric(ncol(returns_mat))
  for (i in 1:ncol(returns_mat)) {
    w_up <- weights; w_up[i] <- w_up[i] + delta; w_up <- w_up / sum(w_up)
    r_up <- as.vector(returns_mat %*% w_up)
    es_up <- mean(r_up[r_up <= quantile(r_up, alpha)])
    marginal_es[i] <- (es_up - port_es) / delta
  }

  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("Asset", 1:ncol(returns_mat))

  data.frame(
    asset = asset_names,
    weight = weights,
    es_contribution = es_contribs,
    pct_of_es = es_contribs / port_es * 100,
    marginal_es = marginal_es,
    es_beta = marginal_es / port_es  # normalized sensitivity
  )
}

# -----------------------------------------------------------------------------
# Scenario-Based Stress Testing with Historical and Hypothetical Scenarios
# Maps named market events to return shocks across assets
# Useful for crypto where we have Black Thursday, FTX collapse, etc.
# -----------------------------------------------------------------------------
scenario_stress_test <- function(returns_mat, weights,
                                  custom_scenarios = NULL) {
  weights <- weights / sum(weights)
  n_assets <- ncol(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:n_assets)

  # Historical worst-period scenarios from actual data
  port_returns <- as.vector(returns_mat %*% weights)
  n <- length(port_returns)

  # Find worst 1-day, 5-day, and 21-day windows
  worst_1d  <- which.min(port_returns)
  roll5  <- filter(port_returns, rep(1/5, 5), sides=1)
  roll21 <- filter(port_returns, rep(1/21, 21), sides=1)
  worst_5d  <- which.min(roll5)
  worst_21d <- which.min(roll21)

  scenarios <- list()

  # Historical scenarios
  scenarios[["worst_1day"]] <- list(
    shocks = returns_mat[worst_1d, ],
    port_loss = port_returns[worst_1d],
    label = paste("Worst 1-day (t=", worst_1d, ")")
  )

  if (!is.na(worst_5d) && worst_5d >= 5) {
    idx5 <- (worst_5d-4):worst_5d
    scenarios[["worst_5day"]] <- list(
      shocks = colSums(returns_mat[idx5, , drop=FALSE]),
      port_loss = sum(port_returns[idx5]),
      label = "Worst 5-day window"
    )
  }

  # Hypothetical crypto scenarios
  hyp_scenarios <- list(
    btc_crash_40pct = setNames(
      c(-0.40, -0.45, -0.35, -0.50, -0.30)[1:n_assets], asset_names),
    defi_contagion  = setNames(
      c(-0.20, -0.35, -0.40, -0.30, -0.25)[1:n_assets], asset_names),
    stablecoin_depeg = setNames(
      c(-0.15, -0.20, -0.25, -0.18, -0.12)[1:n_assets], asset_names),
    regulatory_shock = setNames(
      c(-0.30, -0.25, -0.20, -0.28, -0.22)[1:n_assets], asset_names)
  )

  if (!is.null(custom_scenarios)) {
    hyp_scenarios <- c(hyp_scenarios, custom_scenarios)
  }

  for (nm in names(hyp_scenarios)) {
    shk <- hyp_scenarios[[nm]][1:n_assets]
    scenarios[[nm]] <- list(
      shocks = shk,
      port_loss = sum(weights * shk),
      label = nm
    )
  }

  # Summary table
  summary_df <- do.call(rbind, lapply(names(scenarios), function(nm) {
    s <- scenarios[[nm]]
    data.frame(
      scenario = nm,
      portfolio_loss = s$port_loss,
      portfolio_loss_pct = s$port_loss * 100,
      stringsAsFactors = FALSE
    )
  }))
  summary_df <- summary_df[order(summary_df$portfolio_loss), ]

  list(scenarios = scenarios, summary = summary_df)
}

# -----------------------------------------------------------------------------
# Risk Contribution Decomposition via Euler's theorem
# Total risk (vol or VaR) is decomposed into per-asset contributions
# Risk contribution_i = w_i * (partial Risk / partial w_i)
# -----------------------------------------------------------------------------
risk_contribution <- function(returns_mat, weights, risk_measure = "volatility") {
  weights <- weights / sum(weights)
  n <- ncol(returns_mat)
  port_returns <- as.vector(returns_mat %*% weights)

  if (risk_measure == "volatility") {
    cov_mat <- cov(returns_mat)
    port_var <- as.numeric(t(weights) %*% cov_mat %*% weights)
    port_vol <- sqrt(port_var)

    # Marginal contribution to risk: (Sigma * w)_i / port_vol
    mcr <- as.vector(cov_mat %*% weights) / port_vol

    # Risk contribution = w_i * MCR_i
    rc <- weights * mcr
    pct_rc <- rc / port_vol * 100

  } else if (risk_measure == "var") {
    alpha <- 0.05
    var_val <- -quantile(port_returns, alpha, names = FALSE)
    delta <- 1e-4
    mcr <- numeric(n)
    for (i in 1:n) {
      w2 <- weights; w2[i] <- w2[i] + delta; w2 <- w2/sum(w2)
      r2 <- as.vector(returns_mat %*% w2)
      var2 <- -quantile(r2, alpha)
      mcr[i] <- (var2 - var_val) / delta
    }
    rc  <- weights * mcr
    pct_rc <- rc / var_val * 100
  }

  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("Asset", 1:n)

  df <- data.frame(
    asset = asset_names, weight = weights,
    marginal_contribution = mcr,
    risk_contribution = rc,
    pct_contribution = pct_rc
  )

  # Check: risk contributions sum to total risk
  total_rc <- sum(rc)

  list(contributions = df, total_risk = total_rc, risk_measure = risk_measure,
       diversification_ratio = sum(weights * apply(returns_mat,2,sd)) /
         sqrt(as.numeric(t(weights) %*% cov(returns_mat) %*% weights)))
}

# -----------------------------------------------------------------------------
# Maximum Drawdown Path Analysis: full drawdown path with duration, recovery
# Implements running maximum and identifies all drawdown episodes
# -----------------------------------------------------------------------------
drawdown_path <- function(returns, min_dd = -0.05) {
  # Cumulative wealth index
  wealth <- cumprod(1 + returns)
  n <- length(wealth)

  # Running maximum
  peak <- cummax(wealth)

  # Drawdown series
  dd <- (wealth - peak) / peak

  # Find all drawdown episodes
  in_dd <- FALSE; episodes <- list(); ep_start <- NA; ep_peak <- NA
  for (t in 1:n) {
    if (!in_dd && dd[t] < 0) {
      in_dd <- TRUE
      ep_start <- t
      ep_peak <- wealth[t-1]  # Peak just before drawdown
    }
    if (in_dd && dd[t] == 0) {
      # Recovery
      ep_end <- t
      ep_trough <- which.min(dd[ep_start:ep_end]) + ep_start - 1
      ep_depth <- dd[ep_trough]
      if (ep_depth <= min_dd) {
        episodes[[length(episodes)+1]] <- list(
          start = ep_start, trough = ep_trough, recovery = ep_end,
          depth = ep_depth,
          duration_to_trough = ep_trough - ep_start,
          duration_recovery = ep_end - ep_trough,
          total_duration = ep_end - ep_start
        )
      }
      in_dd <- FALSE
    }
  }

  # Still in drawdown at end
  if (in_dd) {
    ep_trough <- which.min(dd[ep_start:n]) + ep_start - 1
    episodes[[length(episodes)+1]] <- list(
      start = ep_start, trough = ep_trough, recovery = NA,
      depth = dd[ep_trough],
      duration_to_trough = ep_trough - ep_start,
      duration_recovery = NA,
      total_duration = n - ep_start
    )
  }

  ep_df <- if (length(episodes) > 0) {
    do.call(rbind, lapply(episodes, as.data.frame))
  } else {
    data.frame()
  }

  list(
    drawdown_series = dd,
    max_drawdown = min(dd),
    current_drawdown = dd[n],
    episodes = ep_df,
    avg_depth = if (nrow(ep_df) > 0) mean(ep_df$depth) else 0,
    avg_duration = if (nrow(ep_df) > 0) mean(ep_df$total_duration) else 0,
    calmar = mean(returns) * 252 / abs(min(dd))
  )
}

# -----------------------------------------------------------------------------
# Risk Parity Portfolio: allocate weights so each asset contributes equally
# to portfolio volatility. Solved via iterative algorithm.
# Risk parity is popular in crypto as it avoids concentration in BTC
# -----------------------------------------------------------------------------
risk_parity_weights <- function(returns_mat, tol = 1e-8, max_iter = 1000) {
  p <- ncol(returns_mat)
  cov_mat <- cov(returns_mat)

  # Target: equal risk contribution = 1/p of total portfolio vol
  # Use cyclical coordinate descent
  w <- rep(1/p, p)  # start equal weight

  for (iter in 1:max_iter) {
    w_old <- w
    port_var <- as.numeric(t(w) %*% cov_mat %*% w)
    port_vol <- sqrt(port_var)

    # Risk contribution of each asset
    rc <- w * as.vector(cov_mat %*% w) / port_vol

    # Gradient of risk contribution divergence from target (1/p * port_vol)
    target_rc <- port_vol / p
    grad <- rc - target_rc

    # Newton step for each weight
    for (i in 1:p) {
      # d(RC_i)/d(w_i) = (Sigma w)_i / vol + w_i * Sigma_ii / vol - RC_i * Sigma_ii / port_var
      dmcr <- cov_mat[i,i] / port_vol - as.vector(cov_mat %*% w)[i]^2 / (port_vol * port_var)
      drc  <- as.vector(cov_mat %*% w)[i] / port_vol + w[i] * dmcr
      w[i] <- max(w[i] - (rc[i] - target_rc) / (drc + 1e-10), 1e-6)
    }
    w <- w / sum(w)

    if (max(abs(w - w_old)) < tol) break
  }

  port_var_final <- as.numeric(t(w) %*% cov_mat %*% w)
  port_vol_final <- sqrt(port_var_final)
  rc_final <- w * as.vector(cov_mat %*% w) / port_vol_final

  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:p)

  list(
    weights = setNames(w, asset_names),
    risk_contributions = setNames(rc_final, asset_names),
    pct_risk_contributions = setNames(rc_final / port_vol_final * 100, asset_names),
    portfolio_vol = port_vol_final,
    converged = iter < max_iter,
    iterations = iter
  )
}

# -----------------------------------------------------------------------------
# Conditional Value-at-Risk (CVaR) Optimization: minimize ES subject to
# return target using Rockafellar-Uryasev linear programming reformulation
# Avoids the nonlinearity of direct ES optimization
# -----------------------------------------------------------------------------
cvar_optimize <- function(returns_mat, target_return = NULL,
                           alpha = 0.05, max_iter = 500) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)

  # Rockafellar-Uryasev: CVaR = VaR + 1/(n*alpha) * sum(max(-r - VaR, 0))
  # Minimize over (w, z_t, nu) where nu = VaR, z_t = excess losses
  # This is a convex QP-like problem; we solve via gradient descent on dual

  objective <- function(w_nu) {
    w   <- w_nu[1:p]
    nu  <- w_nu[p+1]
    if (any(w < 0) || sum(w) < 0.99 || sum(w) > 1.01) return(1e10)
    r_port <- as.vector(returns_mat %*% w)
    cvar <- nu + mean(pmax(-r_port - nu, 0)) / alpha
    cvar
  }

  # Constraint: weights sum to 1, non-negative
  w0 <- rep(1/p, p)
  nu0 <- -quantile(as.vector(returns_mat %*% w0), alpha)
  init <- c(w0, nu0)

  opt <- optim(init, objective, method = "SANN",
               control = list(maxit = max_iter, temp = 1, trace = FALSE))

  w_opt <- opt$par[1:p]
  w_opt <- pmax(w_opt, 0)
  w_opt <- w_opt / sum(w_opt)
  nu_opt <- opt$par[p+1]

  r_opt <- as.vector(returns_mat %*% w_opt)
  cvar_opt <- nu_opt + mean(pmax(-r_opt - nu_opt, 0)) / alpha

  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:p)

  list(
    weights = setNames(w_opt, asset_names),
    cvar = cvar_opt,
    var = -quantile(r_opt, alpha),
    expected_return = mean(r_opt) * 252,
    sharpe = mean(r_opt) / sd(r_opt) * sqrt(252),
    convergence = opt$convergence
  )
}

# Extended usage example:
# gpd <- fit_gpd_tail(returns_vec, threshold_quantile = 0.05)
# srm  <- spectral_risk_measure(returns_vec, distortion = "wang", gamma = 2)
# rc   <- risk_contribution(returns_mat, weights, risk_measure = "volatility")
# lvar_result <- lvar(returns_mat, weights, position_value = 5e6)
# rp   <- risk_parity_weights(returns_mat)
