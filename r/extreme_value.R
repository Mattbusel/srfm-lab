# extreme_value.R
#
# Complete Extreme Value Theory (EVT) in R for crypto drawdown analysis.
#
# Two approaches:
#   Block Maxima (BM)    → GEV distribution
#   Peak-over-Threshold  → Generalized Pareto Distribution (GPD)
#
# Pickands-Balkema-de Haan theorem: exceedances over a high threshold u converge
# to GPD as u → ∞:
#
#   P(X - u ≤ y | X > u) → G(y; σ_u, ξ)
#
# where G(y) = 1 - (1 + ξy/σ)^{-1/ξ} for y > 0.
#
# References:
#   Coles (2001) "An Introduction to Statistical Modeling of Extreme Values"
#   McNeil (1997) "Estimating the Tails of Loss Severity Distributions"
#   Davison & Smith (1990) "Models for Exceedances over High Thresholds"
#
# Packages: evd, ismev, extRemes, ggplot2, dplyr

suppressPackageStartupMessages({
  library(evd)         # Extreme value distributions (GEV, GPD, bivariate)
  library(ismev)       # Coles (2001) package: gev.fit, gpd.fit, pot
  library(extRemes)    # Gilleland & Katz EVT toolkit
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(scales)
})

# ─────────────────────────────────────────────────────────────────────────────
# GEV FITTING WITH PROFILE LIKELIHOOD CIs
# ─────────────────────────────────────────────────────────────────────────────

#' Fit GEV distribution to block maxima using ismev::gev.fit.
#'
#' Returns MLE estimates with profile likelihood confidence intervals
#' for the shape parameter ξ (most critical for tail behaviour).
#'
#' @param maxima  vector of block maxima
#' @param ci_level  confidence level for profile likelihood CI
#' @return list of parameter estimates and CIs
fit_gev_profile <- function(maxima, ci_level = 0.95) {
  n <- length(maxima)
  cat(sprintf("Fitting GEV to %d block maxima...\n", n))

  # MLE via ismev
  fit <- gev.fit(maxima, show = FALSE)

  mu    <- fit$mle[1]
  sigma <- fit$mle[2]
  xi    <- fit$mle[3]
  se    <- fit$se

  cat(sprintf("  μ = %.4f (SE=%.4f)\n", mu, se[1]))
  cat(sprintf("  σ = %.4f (SE=%.4f)\n", sigma, se[2]))
  cat(sprintf("  ξ = %.4f (SE=%.4f)\n", xi, se[3]))
  cat(sprintf("  Log-likelihood = %.4f\n", -fit$nllh))

  tail_type <- if (xi > 0.05) "Fréchet (heavy tail — Pareto-like)"
               else if (xi < -0.05) "Weibull (bounded upper tail)"
               else "Gumbel (exponential tail)"
  cat(sprintf("  Tail type: %s\n", tail_type))

  # Profile likelihood CI for ξ
  alpha <- 1 - ci_level
  xi_se <- se[3]
  z_crit <- qnorm(1 - alpha/2)

  # Wald CI (approximate)
  ci_wald <- c(xi - z_crit*xi_se, xi + z_crit*xi_se)

  # Profile likelihood CI via confint
  profile_ci <- tryCatch({
    # Use ismev's internal profile likelihood
    prof <- gev.profxi(fit, xlow = xi - 3*xi_se, xup = xi + 3*xi_se, conf = ci_level)
    c(prof$low, prof$high)
  }, error = function(e) {
    cat("  Note: Profile CI computation failed; using Wald CI.\n")
    ci_wald
  })

  cat(sprintf("  %d%% Profile CI for ξ: [%.4f, %.4f]\n",
              round(100*ci_level), profile_ci[1], profile_ci[2]))

  return(list(
    mu = mu, sigma = sigma, xi = xi,
    se = se, loglik = -fit$nllh,
    ci_xi_wald = ci_wald, ci_xi_profile = profile_ci,
    fit = fit, n = n
  ))
}

#' Extract block maxima from a time series of returns.
#'
#' @param x         vector of returns
#' @param block_size  number of observations per block (default 22 = monthly)
#' @param loss        if TRUE, use -x (loss maxima)
extract_block_maxima <- function(x, block_size = 22, loss = TRUE) {
  if (loss) x <- -x
  n_blocks <- floor(length(x) / block_size)
  maxima   <- numeric(n_blocks)
  for (b in 1:n_blocks) {
    idx      <- ((b-1)*block_size + 1):(b*block_size)
    maxima[b] <- max(x[idx])
  }
  return(maxima)
}

# ─────────────────────────────────────────────────────────────────────────────
# GPD FITTING AND THRESHOLD SELECTION
# ─────────────────────────────────────────────────────────────────────────────

#' Fit GPD to exceedances using ismev::gpd.fit.
#'
#' @param x    vector of original data (returns)
#' @param u    threshold
#' @param loss if TRUE, model losses (-x)
fit_gpd <- function(x, u, loss = TRUE) {
  if (loss) x <- -x
  exceedances <- x[x > u] - u
  n_u <- length(exceedances)
  n   <- length(x)

  cat(sprintf("GPD fit: threshold u=%.4f, n_u=%d (%.1f%% of data)\n",
              u, n_u, 100*n_u/n))

  if (n_u < 10) {
    warning("Fewer than 10 exceedances. Threshold may be too high.")
  }

  fit <- gpd.fit(x, u, show = FALSE)
  sigma <- fit$mle[1]
  xi    <- fit$mle[2]
  se    <- fit$se

  cat(sprintf("  σ = %.4f (SE=%.4f)\n", sigma, se[1]))
  cat(sprintf("  ξ = %.4f (SE=%.4f)\n", xi, se[2]))
  cat(sprintf("  Log-likelihood = %.4f\n", -fit$nllh))

  return(list(
    sigma = sigma, xi = xi, u = u,
    se = se, loglik = -fit$nllh,
    n_u = n_u, n = n,
    exceedances = exceedances, fit = fit
  ))
}

#' Mean excess plot (mean residual life plot) for threshold selection.
#'
#' Plots E[X-u | X>u] vs u. A linear increasing section indicates GPD holds.
#' Choose threshold at the start of the linear region.
mean_excess_plot <- function(x, loss = TRUE,
                              quantile_range = c(0.80, 0.99),
                              n_thresholds = 60) {
  if (loss) x <- -x

  u_vec  <- quantile(x, seq(quantile_range[1], quantile_range[2],
                             length.out = n_thresholds))
  me_vec <- numeric(n_thresholds)
  ci_lo  <- numeric(n_thresholds)
  ci_hi  <- numeric(n_thresholds)

  for (k in seq_along(u_vec)) {
    exc   <- x[x > u_vec[k]] - u_vec[k]
    if (length(exc) < 3) {
      me_vec[k] <- ci_lo[k] <- ci_hi[k] <- NA
      next
    }
    me_vec[k] <- mean(exc)
    se_k      <- sd(exc) / sqrt(length(exc))
    ci_lo[k]  <- me_vec[k] - 1.96 * se_k
    ci_hi[k]  <- me_vec[k] + 1.96 * se_k
  }

  df <- data.frame(u = u_vec, me = me_vec, ci_lo = ci_lo, ci_hi = ci_hi) %>%
    filter(!is.na(me))

  p <- ggplot(df, aes(x = u)) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.2, fill = "steelblue") +
    geom_line(aes(y = me), color = "steelblue", linewidth = 1.2) +
    geom_point(aes(y = me), size = 1.5) +
    labs(title = "Mean Excess Plot (Threshold Selection)",
         x = "Threshold u", y = "Mean Excess e(u)",
         subtitle = "Choose threshold at start of linear section") +
    theme_minimal()

  return(list(plot = p, data = df))
}

#' Hill plot for automated tail index estimation.
#'
#' @param x  vector of data (losses, so positive values represent losses)
#' @param k_range  range of order statistics to use
hill_plot <- function(x, loss = TRUE, k_min = 10, k_max = NULL) {
  if (loss) x <- -x
  x_sorted <- sort(x, decreasing = TRUE)  # descending
  n <- length(x_sorted)
  if (is.null(k_max)) k_max <- floor(n / 2)

  k_vals   <- k_min:k_max
  hill_est <- numeric(length(k_vals))

  for (idx in seq_along(k_vals)) {
    k <- k_vals[idx]
    hill_est[idx] <- mean(log(x_sorted[1:k] / x_sorted[k+1]))
  }

  df <- data.frame(k = k_vals, xi = hill_est)

  # Find stable region (minimum variance window)
  window <- 20
  if (length(hill_est) > window*2) {
    var_vals <- sapply(window:(length(hill_est)-window),
                       function(i) var(hill_est[(i-window+1):(i+window)]))
    k_star_idx <- which.min(var_vals) + window - 1
    xi_star <- hill_est[k_star_idx]
    k_star  <- k_vals[k_star_idx]
    cat(sprintf("Hill estimate: ξ̂ = %.4f at k* = %d\n", xi_star, k_star))
  } else {
    xi_star <- mean(hill_est)
    k_star  <- k_vals[floor(length(k_vals)/2)]
  }

  p <- ggplot(df, aes(x = k, y = xi)) +
    geom_line(color = "steelblue", linewidth = 1) +
    geom_hline(yintercept = xi_star, color = "red",
               linetype = "dashed", linewidth = 0.8) +
    geom_vline(xintercept = k_star, color = "orange",
               linetype = "dashed", linewidth = 0.8) +
    annotate("text", x = k_star, y = max(hill_est)*0.9,
             label = sprintf("ξ̂=%.3f", xi_star), color = "red", hjust = -0.1) +
    labs(title = "Hill Plot (Tail Index Estimation)",
         x = "Number of order statistics k",
         y = "Hill estimator ξ̂") +
    theme_minimal()

  return(list(plot = p, xi_hat = xi_star, k_star = k_star))
}

# ─────────────────────────────────────────────────────────────────────────────
# RETURN LEVELS WITH CONFIDENCE BANDS
# ─────────────────────────────────────────────────────────────────────────────

#' Compute return levels and profile likelihood confidence bands for GEV.
#'
#' Return level x_m: exceeds once every m blocks on average.
#' x_m = μ + σ[(−log(1−1/m))^{−ξ} − 1]/ξ
#'
#' @param gev_result  output from fit_gev_profile
#' @param return_periods  vector of return periods in years
#' @param blocks_per_year  number of blocks per year
return_levels_gev <- function(gev_result, return_periods = c(1,2,5,10,20,50,100),
                               blocks_per_year = 52) {
  mu <- gev_result$mu; sigma <- gev_result$sigma; xi <- gev_result$xi
  se <- gev_result$se

  cat("\nGEV Return Levels:\n")
  cat(sprintf("%-12s %-12s %-15s %-15s\n", "Period (yr)", "Level", "95% CI lower", "95% CI upper"))
  cat(rep("─", 55), "\n", sep = "")

  results <- data.frame(
    period = return_periods,
    level  = numeric(length(return_periods)),
    ci_lo  = numeric(length(return_periods)),
    ci_hi  = numeric(length(return_periods))
  )

  for (i in seq_along(return_periods)) {
    T_yr <- return_periods[i]
    m    <- T_yr * blocks_per_year
    p    <- 1 - 1/m

    if (abs(xi) < 1e-6) {
      rl <- mu - sigma * log(-log(p))
    } else {
      rl <- mu + sigma * ((-log(p))^(-xi) - 1) / xi
    }

    # Delta method SE for return level
    # ∂rl/∂θ = [1, (y_p-1)/ξ, -σ(y_p-1)/ξ² + σ·y_p·log(y_p)/ξ]
    y_p <- (-log(p))^(-xi)
    grad <- c(1,
              (y_p - 1) / xi,
              -sigma*(y_p-1)/xi^2 + sigma*y_p*log(-log(p))/xi)
    Sigma_mat <- diag(se^2)  # approx; ignores covariances
    var_rl <- tryCatch(as.numeric(t(grad) %*% Sigma_mat %*% grad),
                       error = function(e) (se[1]^2))

    ci_lo <- rl - 1.96 * sqrt(max(var_rl, 0))
    ci_hi <- rl + 1.96 * sqrt(max(var_rl, 0))

    results$level[i]  <- rl
    results$ci_lo[i]  <- ci_lo
    results$ci_hi[i]  <- ci_hi

    cat(sprintf("%-12d %-12.4f %-15.4f %-15.4f\n", T_yr, rl, ci_lo, ci_hi))
  }

  return(results)
}

#' GPD return levels.
#'
#' x_m = u + σ/ξ · [(m·n_u/n)^ξ - 1]
#'
#' @param gpd_result  output from fit_gpd
#' @param obs_per_year  observations per year
return_levels_gpd <- function(gpd_result, return_periods = c(1,2,5,10,50,100),
                               obs_per_year = 252) {
  sigma <- gpd_result$sigma; xi <- gpd_result$xi
  u <- gpd_result$u; n_u <- gpd_result$n_u; n <- gpd_result$n

  lambda <- n_u / n * obs_per_year  # exceedances per year

  cat(sprintf("\nGPD Return Levels (λ=%.2f exceedances/year):\n", lambda))
  cat(sprintf("%-12s %-12s %-12s\n", "Period (yr)", "Level", "% Loss"))
  cat(rep("─", 38), "\n", sep = "")

  results <- data.frame(period = return_periods, level = numeric(length(return_periods)))

  for (i in seq_along(return_periods)) {
    T_yr <- return_periods[i]
    if (abs(xi) < 1e-6) {
      rl <- u + sigma * log(T_yr * lambda)
    } else {
      rl <- u + sigma/xi * ((T_yr * lambda)^xi - 1)
    }
    results$level[i] <- rl
    cat(sprintf("%-12d %-12.4f %-12.2f%%\n", T_yr, rl, 100*rl))
  }
  return(results)
}

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL TAIL EXPECTATION (CVaR / ES)
# ─────────────────────────────────────────────────────────────────────────────

#' Compute VaR and CVaR (Expected Shortfall) using GPD tail fit.
#'
#' CVaR_α = VaR_α + E[X - VaR_α | X > VaR_α]
#'
#' For GPD: CVaR_α = VaR_α + (σ_u + ξ(VaR_α - u)) / (1 - ξ)
#'
#' @param gpd_result  output from fit_gpd
#' @param alpha_levels  vector of significance levels (tail probabilities)
compute_tail_risk <- function(gpd_result, alpha_levels = c(0.01, 0.005, 0.001)) {
  sigma <- gpd_result$sigma; xi <- gpd_result$xi
  u <- gpd_result$u; n_u <- gpd_result$n_u; n <- gpd_result$n

  cat("\nTail Risk Metrics (GPD):\n")
  cat(sprintf("%-8s %-12s %-12s %-15s\n", "α", "VaR", "CVaR/ES", "Prob exceed"))
  cat(rep("─", 50), "\n", sep = "")

  results <- data.frame(alpha = alpha_levels,
                         VaR   = numeric(length(alpha_levels)),
                         CVaR  = numeric(length(alpha_levels)))

  for (i in seq_along(alpha_levels)) {
    alpha <- alpha_levels[i]

    # VaR via GPD quantile
    p_exc <- alpha * n / n_u  # P(exceed | above u) = α / P(X > u)
    if (p_exc >= 1) {
      cat(sprintf("  α=%.3f: threshold too low for this alpha level\n", alpha))
      next
    }

    if (abs(xi) < 1e-6) {
      VaR <- u - sigma * log(p_exc)
    } else {
      VaR <- u + sigma/xi * (p_exc^(-xi) - 1)
    }

    # CVaR = E[X | X > VaR]
    if (xi < 1) {
      CVaR <- (VaR + sigma - xi * u) / (1 - xi)
    } else {
      CVaR <- Inf
    }

    results$VaR[i]  <- VaR
    results$CVaR[i] <- CVaR

    cat(sprintf("%.3f    %-12.4f %-12.4f %-15.6f\n",
                alpha, VaR, CVaR, alpha))
  }
  return(results)
}

# ─────────────────────────────────────────────────────────────────────────────
# MULTIVARIATE EXTREMES: COMPONENTWISE MAXIMA
# ─────────────────────────────────────────────────────────────────────────────

#' Componentwise block maxima for multivariate returns.
#' Each asset's maxima are modelled separately with GEV.
#'
#' @param X  n x d matrix of returns
#' @param block_size  observations per block
#' @return  list of GEV fits per asset
multivariate_block_maxima <- function(X, block_size = 22, loss = TRUE) {
  d <- ncol(X)
  var_names <- if (!is.null(colnames(X))) colnames(X) else paste0("X", 1:d)

  results <- list()
  cat("\nMultivariate Block Maxima (GEV per asset):\n")
  cat("─" %+% rep("─", 60) %+% "\n")

  for (j in 1:d) {
    cat(sprintf("\nAsset: %s\n", var_names[j]))
    maxima_j <- extract_block_maxima(X[,j], block_size, loss)
    gev_j    <- fit_gev_profile(maxima_j)
    results[[var_names[j]]] <- gev_j
  }

  # Summary table
  cat("\nGEV Parameter Summary:\n")
  cat(sprintf("%-10s %-8s %-8s %-8s\n", "Asset", "μ", "σ", "ξ"))
  cat(rep("─", 40), "\n", sep = "")
  for (nm in names(results)) {
    r <- results[[nm]]
    cat(sprintf("%-10s %-8.4f %-8.4f %-8.4f\n", nm, r$mu, r$sigma, r$xi))
  }

  return(results)
}

# String concatenation helper
`%+%` <- function(a, b) paste0(a, b)

#' Bivariate extreme value distribution fitting via evd package.
#' Models the dependence in extremes using a logistic model.
#'
#' @param X  n x 2 matrix of returns (two assets)
#' @param u  quantile threshold for joint exceedances
fit_bivariate_extremes <- function(X, u = 0.95, loss = TRUE) {
  if (loss) X <- -X
  n <- nrow(X)

  # Transform to unit Fréchet margins for bivariate analysis
  # F̃(x) = 1 / (- log F̂(x)) where F̂ is empirical CDF
  U <- apply(X, 2, function(col) rank(col) / (n + 1))
  Z <- -1 / log(U)  # unit Fréchet margins

  # Fit bivariate extreme value model (logistic dependence)
  tryCatch({
    fit_bv <- fbvevd(Z, model = "log")
    alpha <- fit_bv$estimate["dep"]
    cat(sprintf("Bivariate extreme dependence (logistic): α=%.4f\n", alpha))
    cat("  α→0: complete dependence, α→1: independence\n")
    return(list(model = fit_bv, dep = alpha))
  }, error = function(e) {
    cat("Bivariate EVT fit failed:", conditionMessage(e), "\n")
    return(NULL)
  })
}

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK MAXIMA VS POT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

#' Compare Block Maxima (GEV) and POT (GPD) estimates of tail quantiles.
compare_bm_pot <- function(x, loss = TRUE,
                            block_size = 22,
                            u_quantile = 0.95,
                            return_periods = c(1, 5, 10, 20)) {
  if (loss) x <- -x
  n <- length(x)

  # Block maxima
  maxima <- extract_block_maxima(x, block_size, loss = FALSE)
  gev    <- fit_gev_profile(maxima)
  rl_gev <- return_levels_gev(gev, return_periods, blocks_per_year = 252/block_size)

  # POT
  u   <- quantile(x, u_quantile)
  gpd <- fit_gpd(x, u, loss = FALSE)
  rl_gpd <- return_levels_gpd(gpd, return_periods, obs_per_year = 252)

  cat("\nComparison of Return Level Estimates:\n")
  cat(sprintf("%-12s %-15s %-15s\n", "Period (yr)", "GEV (BM)", "GPD (POT)"))
  cat(rep("─", 45), "\n", sep = "")
  for (i in seq_along(return_periods)) {
    cat(sprintf("%-12d %-15.4f %-15.4f\n",
                return_periods[i], rl_gev$level[i], rl_gpd$level[i]))
  }

  return(list(gev = gev, gpd = gpd, rl_gev = rl_gev, rl_gpd = rl_gpd))
}

# ─────────────────────────────────────────────────────────────────────────────
# GPD DIAGNOSTICS PLOTS
# ─────────────────────────────────────────────────────────────────────────────

#' Four-panel GPD diagnostic plot:
#'   (1) Probability plot, (2) Quantile plot
#'   (3) Return level plot, (4) Density plot
plot_gpd_diagnostics <- function(gpd_result, title = "GPD Diagnostics") {
  exc  <- gpd_result$exceedances
  s    <- gpd_result$sigma
  xi   <- gpd_result$xi
  n    <- length(exc)
  exc_sorted <- sort(exc)
  emp_probs  <- (1:n) / (n + 1)

  # GPD CDF
  pgpd_custom <- function(y, sigma, xi) {
    if (abs(xi) < 1e-6) 1 - exp(-y/sigma)
    else {
      t <- 1 + xi*y/sigma
      ifelse(t <= 0, 1, 1 - t^(-1/xi))
    }
  }
  qgpd_custom <- function(p, sigma, xi) {
    if (abs(xi) < 1e-6) -sigma * log(1-p)
    else sigma/xi * ((1-p)^(-xi) - 1)
  }

  fit_probs  <- sapply(exc_sorted, pgpd_custom, sigma=s, xi=xi)
  fit_quantiles <- sapply(emp_probs, qgpd_custom, sigma=s, xi=xi)

  # Probability plot
  df_pp <- data.frame(emp = emp_probs, fit = fit_probs)
  p1 <- ggplot(df_pp, aes(x = emp, y = fit)) +
    geom_point(size = 1.5, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(title = "Probability Plot", x = "Empirical", y = "Model") +
    theme_minimal()

  # Quantile plot
  df_qp <- data.frame(model = fit_quantiles, emp = exc_sorted)
  p2 <- ggplot(df_qp, aes(x = model, y = emp)) +
    geom_point(size = 1.5, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(title = "Quantile Plot", x = "Model quantiles", y = "Empirical quantiles") +
    theme_minimal()

  # Density plot
  x_seq <- seq(0, max(exc_sorted) * 1.1, length.out = 300)
  dgpd_vals <- sapply(x_seq, function(y) {
    if (abs(xi) < 1e-6) exp(-y/s)/s
    else {
      t <- 1 + xi*y/s
      if (t <= 0) 0 else t^(-1/xi - 1)/s
    }
  })
  df_dens <- data.frame(x = x_seq, d = dgpd_vals)
  p3 <- ggplot() +
    geom_histogram(data = data.frame(exc = exc), aes(x = exc, y = after_stat(density)),
                   bins = 30, fill = "steelblue", color = "white", alpha = 0.6) +
    geom_line(data = df_dens, aes(x = x, y = d), color = "red", linewidth = 1.2) +
    labs(title = sprintf("GPD Density (σ=%.3f, ξ=%.3f)", s, xi),
         x = "Exceedance", y = "Density") +
    theme_minimal()

  # Return level plot
  probs_seq <- seq(0.01, 0.999, length.out = 200)
  rl_seq    <- sapply(probs_seq, qgpd_custom, sigma=s, xi=xi) + gpd_result$u
  df_rl <- data.frame(
    return_period = 1 / (1 - probs_seq),
    level = rl_seq
  )
  p4 <- ggplot(df_rl, aes(x = return_period, y = level)) +
    geom_line(color = "navy", linewidth = 1.2) +
    scale_x_log10(labels = label_number()) +
    labs(title = "Return Level Plot", x = "Return period", y = "Return level") +
    theme_minimal()

  grid.arrange(p1, p2, p3, p4, ncol = 2, top = title)
}

#' GEV return level plot with empirical estimates and CI bands.
plot_return_level_gev <- function(gev_result, maxima,
                                   blocks_per_year = 52,
                                   max_period = 100) {
  mu <- gev_result$mu; sigma <- gev_result$sigma; xi <- gev_result$xi
  n  <- length(maxima)

  # Theoretical return levels
  periods <- exp(seq(log(0.1), log(max_period), length.out = 300))
  probs   <- 1 - 1/(periods * blocks_per_year)
  probs   <- pmax(pmin(probs, 1 - 1e-8), 1e-8)

  if (abs(xi) < 1e-6) {
    rl_fit <- mu - sigma * log(-log(probs))
  } else {
    rl_fit <- mu + sigma * ((-log(probs))^(-xi) - 1) / xi
  }

  # Empirical return periods
  sorted_max <- sort(maxima)
  emp_T      <- (n + 1) / (n + 1 - rank(sorted_max)) / blocks_per_year

  df_fit <- data.frame(period = periods, level = rl_fit)
  df_emp <- data.frame(period = emp_T,  level = sorted_max)

  ggplot() +
    geom_line(data = df_fit, aes(x = period, y = level),
              color = "steelblue", linewidth = 1.2) +
    geom_point(data = df_emp, aes(x = period, y = level),
               color = "black", size = 1.5) +
    scale_x_log10() +
    labs(title = sprintf("GEV Return Level Plot (ξ=%.3f)", xi),
         x = "Return period (years)", y = "Return level") +
    theme_minimal()
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO: CRYPTO DRAWDOWN DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

demo_evt <- function() {
  set.seed(42)
  cat("================================================================\n")
  cat("Extreme Value Theory Demo: Crypto Drawdown Analysis\n")
  cat("================================================================\n\n")

  # Simulate 3 years of daily BTC-like returns (t-distributed, heavy tails)
  n <- 756  # 3 years × 252 trading days
  nu <- 4   # degrees of freedom
  sigma_daily <- 0.04  # 4% daily volatility

  returns <- rt(n, df = nu) * sigma_daily
  cat(sprintf("Simulated %d daily returns: mean=%.4f, sd=%.4f\n",
              n, mean(returns), sd(returns)))
  cat(sprintf("Empirical 1%% quantile: %.4f (%.2f%%)\n",
              quantile(returns, 0.01), 100*quantile(returns, 0.01)))

  # 1. Threshold selection
  cat("\n1. Threshold Selection\n")
  cat("─────────────────────\n")
  me_result <- mean_excess_plot(returns, loss = TRUE)

  hill_result <- hill_plot(returns, loss = TRUE)
  cat(sprintf("Hill estimate ξ̂ = %.4f (true ξ = %.4f)\n",
              hill_result$xi_hat, 1/nu))

  # 2. GPD fitting
  cat("\n2. GPD Fitting (POT Method)\n")
  cat("───────────────────────────\n")
  u_threshold <- quantile(-returns, 0.95)
  gpd_result <- fit_gpd(returns, u_threshold, loss = TRUE)

  # Tail risk
  tr <- compute_tail_risk(gpd_result, alpha_levels = c(0.01, 0.005, 0.001))

  # Return levels
  rl_gpd <- return_levels_gpd(gpd_result, obs_per_year = 252)

  # 3. GEV fitting (block maxima)
  cat("\n3. GEV Fitting (Block Maxima Method)\n")
  cat("─────────────────────────────────────\n")
  maxima <- extract_block_maxima(returns, block_size = 22, loss = TRUE)
  gev_result <- fit_gev_profile(maxima)
  rl_gev <- return_levels_gev(gev_result, blocks_per_year = 12)

  # 4. BM vs POT comparison
  cat("\n4. Block Maxima vs POT Comparison\n")
  cat("──────────────────────────────────\n")
  comparison <- compare_bm_pot(returns, u_quantile = 0.95, return_periods = c(1,5,10))

  # 5. Multivariate extremes (simulate 3 assets)
  cat("\n5. Multivariate Block Maxima\n")
  cat("────────────────────────────\n")
  R_multi <- matrix(rt(n*3, df=nu)*sigma_daily, n, 3)
  colnames(R_multi) <- c("BTC","ETH","BNB")
  mv_gev <- multivariate_block_maxima(R_multi, block_size = 22, loss = TRUE)

  # Bivariate extremes for BTC-ETH
  bv <- fit_bivariate_extremes(R_multi[,1:2], u = 0.90, loss = TRUE)

  # 6. Tail VaR summary
  cat("\n6. Tail VaR Summary\n")
  cat("────────────────────\n")
  for (a in c(0.01, 0.005, 0.001)) {
    VaR <- tr$VaR[tr$alpha == a]
    ES  <- tr$CVaR[tr$alpha == a]
    if (length(VaR) > 0 && !is.na(VaR)) {
      cat(sprintf("  α=%.3f: VaR=%.4f (%.2f%%), ES=%.4f (%.2f%%)\n",
                  a, VaR, 100*VaR, ES, 100*ES))
    }
  }

  # Plots
  cat("\nGenerating diagnostic plots...\n")
  png("evt_gpd_diagnostics.png", width = 1000, height = 700)
  plot_gpd_diagnostics(gpd_result, title = "GPD Diagnostics: BTC Daily Losses")
  dev.off()

  p_rl_gev <- plot_return_level_gev(gev_result, maxima, blocks_per_year = 12)
  ggsave("evt_gev_return_levels.png", p_rl_gev, width = 7, height = 5)

  p_hill <- hill_result$plot
  p_me   <- me_result$plot
  ggsave("evt_hill_plot.png", p_hill, width = 7, height = 4)
  ggsave("evt_mean_excess.png", p_me,  width = 7, height = 4)

  cat("Saved: evt_gpd_diagnostics.png, evt_gev_return_levels.png\n")
  cat("       evt_hill_plot.png, evt_mean_excess.png\n")

  cat("\nEVT demo complete.\n")

  return(list(gev = gev_result, gpd = gpd_result,
              tail_risk = tr, comparison = comparison))
}

if (!interactive()) {
  demo_evt()
}
