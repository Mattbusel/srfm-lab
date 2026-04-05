# =============================================================================
# hypothesis_testing.R
# Statistical hypothesis testing for trading strategy validation
# Base R only -- no external packages
# =============================================================================
# Financial intuition: A strategy that looks profitable in-sample may be
# pure luck. Rigorous hypothesis testing guards against data mining bias.
# The probability of finding a "significant" result by chance across 100
# strategies tested = 1-(0.95^100) = 99.4%. Multiple testing corrections
# and bootstrap permutation tests are essential.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. T-TESTS AND NON-PARAMETRIC TESTS
# -----------------------------------------------------------------------------

#' One-sample t-test for strategy returns
#' H0: mean return = mu_0 (usually 0)
one_sample_ttest <- function(x, mu_0=0, alpha=0.05) {
  n    <- length(x)
  xbar <- mean(x); s <- sd(x)
  se   <- s / sqrt(n)
  t    <- (xbar - mu_0) / se
  df   <- n - 1
  p    <- 2 * pt(-abs(t), df=df)
  ci   <- xbar + c(-1,1) * qt(1-alpha/2, df=df) * se

  cat("=== One-Sample t-test ===\n")
  cat(sprintf("H0: mu = %.4f\n", mu_0))
  cat(sprintf("x_bar=%.6f, s=%.6f, n=%d\n", xbar, s, n))
  cat(sprintf("t = %.3f, df = %d, p = %.4f\n", t, df, p))
  cat(sprintf("95%% CI: [%.6f, %.6f]\n", ci[1], ci[2]))
  cat(sprintf("Result: %s at alpha=%.2f\n",
              ifelse(p < alpha, "REJECT H0 (significant mean)", "FAIL TO REJECT H0"), alpha))

  invisible(list(statistic=t, df=df, p=p, ci=ci, xbar=xbar, se=se))
}

#' Two-sample t-test: compare returns across two regimes/strategies
two_sample_ttest <- function(x, y, alpha=0.05, equal_var=FALSE) {
  nx <- length(x); ny <- length(y)
  xbar <- mean(x); ybar <- mean(y)
  sx <- sd(x); sy <- sd(y)

  if (equal_var) {
    sp2 <- ((nx-1)*sx^2 + (ny-1)*sy^2) / (nx+ny-2)
    se  <- sqrt(sp2*(1/nx + 1/ny))
    df  <- nx + ny - 2
  } else {
    # Welch's t-test
    se_x <- sx^2/nx; se_y <- sy^2/ny
    se   <- sqrt(se_x + se_y)
    df   <- (se_x + se_y)^2 / (se_x^2/(nx-1) + se_y^2/(ny-1))
  }

  t  <- (xbar - ybar) / se
  p  <- 2 * pt(-abs(t), df=df)
  ci <- (xbar-ybar) + c(-1,1) * qt(1-alpha/2, df=df) * se

  cat("=== Two-Sample t-test ===\n")
  cat(sprintf("x: n=%d, mean=%.5f, sd=%.5f\n", nx, xbar, sx))
  cat(sprintf("y: n=%d, mean=%.5f, sd=%.5f\n", ny, ybar, sy))
  cat(sprintf("t = %.3f, df = %.1f, p = %.4f\n", t, df, p))
  cat(sprintf("Difference CI: [%.5f, %.5f]\n", ci[1], ci[2]))
  cat(sprintf("Result: %s\n", ifelse(p<alpha, "SIGNIFICANT DIFFERENCE", "NO SIGNIFICANT DIFFERENCE")))
  invisible(list(t=t, df=df, p=p, ci=ci, diff=xbar-ybar))
}

#' Wilcoxon signed-rank test (non-parametric, one-sample)
#' Does not assume normality -- better for fat-tailed crypto returns
wilcoxon_signed_rank <- function(x, mu_0=0, alpha=0.05) {
  x_c <- x - mu_0
  n_nonzero <- sum(x_c != 0)
  x_c <- x_c[x_c != 0]
  n   <- length(x_c)

  ranks <- rank(abs(x_c))
  W_pos <- sum(ranks[x_c > 0])
  W_neg <- sum(ranks[x_c < 0])
  W     <- min(W_pos, W_neg)

  # Normal approximation for large n
  mu_W  <- n*(n+1)/4
  sig_W <- sqrt(n*(n+1)*(2*n+1)/24)
  z     <- (W - mu_W) / sig_W
  p     <- 2 * pnorm(-abs(z))

  cat("=== Wilcoxon Signed-Rank Test ===\n")
  cat(sprintf("n (non-zero): %d, W+=%d, W-=%d, W_min=%d\n", n, W_pos, W_neg, W))
  cat(sprintf("z = %.3f, p = %.4f\n", z, p))
  cat(sprintf("Result: %s\n", ifelse(p<alpha, "SIGNIFICANT (reject H0)", "NOT significant")))
  invisible(list(W=W, z=z, p=p, W_pos=W_pos, W_neg=W_neg))
}

#' Mann-Whitney U test (non-parametric two-sample)
mann_whitney <- function(x, y, alpha=0.05) {
  nx <- length(x); ny <- length(y)
  # Compute U statistic
  U_x <- sum(sapply(x, function(xi) sum(xi > y) + 0.5*sum(xi==y)))
  U_y <- nx*ny - U_x

  # Normal approximation
  mu_U  <- nx*ny/2
  sig_U <- sqrt(nx*ny*(nx+ny+1)/12)
  z     <- (min(U_x, U_y) - mu_U) / sig_U
  p     <- 2 * pnorm(-abs(z))

  cat("=== Mann-Whitney U Test ===\n")
  cat(sprintf("U_x=%d, U_y=%d, z=%.3f, p=%.4f\n", U_x, U_y, z, p))
  cat(sprintf("Result: %s\n", ifelse(p<alpha, "SIGNIFICANT", "NOT significant")))
  invisible(list(U=min(U_x,U_y), z=z, p=p))
}

# -----------------------------------------------------------------------------
# 2. MULTIPLE COMPARISONS CORRECTIONS
# -----------------------------------------------------------------------------

#' Bonferroni correction (most conservative)
bonferroni <- function(p_values, alpha=0.05) {
  m <- length(p_values)
  adj_p <- pmin(p_values * m, 1)
  list(adj_p=adj_p, rejected=adj_p<alpha, n_rejected=sum(adj_p<alpha))
}

#' Benjamini-Hochberg FDR (already in alpha_testing.R, included for completeness)
bh_fdr <- function(p_values, alpha=0.05) {
  m   <- length(p_values)
  ord <- order(p_values)
  p_s <- p_values[ord]
  k   <- seq_len(m)
  rej <- which(p_s <= k/m * alpha)
  k_max <- if (length(rej)>0) max(rej) else 0
  rejected <- logical(m)
  if (k_max > 0) rejected[ord[1:k_max]] <- TRUE
  adj_p <- pmin(p_s * m/k, 1)
  adj_p <- rev(cummin(rev(adj_p)))
  list(rejected=rejected, adj_p=adj_p[order(ord)], n_rejected=sum(rejected))
}

# -----------------------------------------------------------------------------
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# -----------------------------------------------------------------------------

#' Percentile bootstrap CI
#' @param x data vector
#' @param stat_fn function to compute statistic on data
#' @param n_boot number of bootstrap replications
#' @param alpha significance level
bootstrap_ci_percentile <- function(x, stat_fn=mean, n_boot=2000, alpha=0.05) {
  n         <- length(x)
  boot_vals <- replicate(n_boot, stat_fn(sample(x, n, replace=TRUE)))
  ci        <- quantile(boot_vals, c(alpha/2, 1-alpha/2))
  theta_hat <- stat_fn(x)

  cat(sprintf("Bootstrap Percentile CI (%.0f%%): [%.5f, %.5f]\n",
              100*(1-alpha), ci[1], ci[2]))
  cat(sprintf("Point estimate: %.5f\n", theta_hat))
  invisible(list(ci=ci, boot_dist=boot_vals, theta=theta_hat))
}

#' Bias-Corrected and Accelerated (BCa) bootstrap CI
#' More accurate than percentile bootstrap, especially when distribution is skewed
bootstrap_ci_bca <- function(x, stat_fn=mean, n_boot=2000, alpha=0.05) {
  n         <- length(x)
  theta_hat <- stat_fn(x)
  boot_vals <- replicate(n_boot, stat_fn(sample(x, n, replace=TRUE)))

  # Bias correction z0
  z0 <- qnorm(mean(boot_vals < theta_hat))

  # Acceleration a: jacknife
  jack_vals <- sapply(seq_len(n), function(i) stat_fn(x[-i]))
  jack_mean <- mean(jack_vals)
  a <- sum((jack_mean - jack_vals)^3) / (6 * sum((jack_mean - jack_vals)^2)^(3/2) + 1e-10)

  # Adjusted quantiles
  za <- qnorm(alpha/2); zb <- qnorm(1-alpha/2)
  a1 <- pnorm(z0 + (z0+za)/(1-a*(z0+za)))
  a2 <- pnorm(z0 + (z0+zb)/(1-a*(z0+zb)))
  ci  <- quantile(boot_vals, c(a1, a2))

  cat(sprintf("BCa Bootstrap CI (%.0f%%): [%.5f, %.5f]\n",
              100*(1-alpha), ci[1], ci[2]))
  cat(sprintf("z0=%.3f, a=%.4f\n", z0, a))
  invisible(list(ci=ci, boot_dist=boot_vals, theta=theta_hat, z0=z0, a=a))
}

# -----------------------------------------------------------------------------
# 4. PERMUTATION TESTS FOR TRADING STRATEGY SIGNIFICANCE
# -----------------------------------------------------------------------------

#' Permutation test for strategy Sharpe ratio
#' H0: strategy has no predictive power (returns are exchangeable)
#' Shuffle returns and recompute Sharpe to build null distribution
#' @param signals signal series
#' @param returns return series
#' @param stat_fn test statistic function (default: Sharpe ratio)
#' @param n_perm number of permutations
permutation_test_strategy <- function(signals, returns, n_perm=2000,
                                       stat_fn=NULL, alpha=0.05) {
  n <- length(signals)
  stopifnot(length(returns) == n)

  if (is.null(stat_fn)) {
    stat_fn <- function(s, r) {
      pnl <- sign(s) * r
      mean(pnl) / (sd(pnl)+1e-10) * sqrt(252)
    }
  }

  # Observed statistic
  T_obs <- stat_fn(signals, returns)

  # Null distribution via permutation
  T_perm <- replicate(n_perm, {
    r_perm <- sample(returns, n, replace=FALSE)
    stat_fn(signals, r_perm)
  })

  p_val <- mean(T_perm >= T_obs)

  cat("=== Permutation Test for Strategy ===\n")
  cat(sprintf("Observed Sharpe: %.3f\n", T_obs))
  cat(sprintf("Null distribution: mean=%.3f, sd=%.3f\n", mean(T_perm), sd(T_perm)))
  cat(sprintf("p-value: %.4f (one-sided)\n", p_val))
  cat(sprintf("Result: %s at alpha=%.2f\n",
              ifelse(p_val<alpha, "STRATEGY IS SIGNIFICANT", "NOT SIGNIFICANT (may be luck)"), alpha))

  invisible(list(T_obs=T_obs, T_perm=T_perm, p_val=p_val))
}

# -----------------------------------------------------------------------------
# 5. WHITE'S REALITY CHECK AND HANSEN'S SPA
# -----------------------------------------------------------------------------

#' White's (2000) Reality Check
#' Tests H0: no strategy among m candidates has positive expected return
#' Uses stationary bootstrap to simulate null distribution
#' @param returns_mat T x m matrix of strategy returns
whites_reality_check <- function(returns_mat, n_boot=1000, alpha=0.05) {
  T_obs <- nrow(returns_mat); m <- ncol(returns_mat)
  mu_hat <- colMeans(returns_mat)
  T_stat  <- max(mu_hat)  # performance of best strategy

  # Stationary bootstrap (simplified: circular block bootstrap)
  block_size <- max(1, round(sqrt(T_obs)))
  boot_max <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    # Generate bootstrap indices (block)
    boot_ret <- matrix(0, T_obs, m)
    t_idx <- 1
    while (t_idx <= T_obs) {
      block_start <- sample(seq_len(T_obs), 1)
      blk_len <- min(block_size, T_obs - t_idx + 1)
      boot_idx <- (block_start + seq_len(blk_len) - 2) %% T_obs + 1
      boot_ret[t_idx:(t_idx+blk_len-1), ] <- returns_mat[boot_idx, ]
      t_idx <- t_idx + blk_len
    }
    boot_mu <- colMeans(boot_ret)
    boot_max[b] <- max(boot_mu - mu_hat)  # centered at zero under H0
  }

  p_val <- mean(boot_max >= T_stat)
  crit  <- quantile(boot_max, 1-alpha)

  cat("=== White's Reality Check ===\n")
  cat(sprintf("Strategies tested: %d\n", m))
  cat(sprintf("Best strategy mean return: %.5f\n", T_stat))
  cat(sprintf("Bootstrap critical value: %.5f\n", crit))
  cat(sprintf("p-value: %.4f\n", p_val))
  cat(sprintf("Result: %s\n",
              ifelse(p_val<alpha, "Significant alpha (not just data mining)",
                     "Cannot reject null (may be data mining)")))

  invisible(list(T_stat=T_stat, p_val=p_val, boot_max=boot_max, crit=crit))
}

#' Hansen's SPA (Superior Predictive Ability) test
#' Improves on White's by removing poor strategies from null distribution
hansens_spa <- function(returns_mat, n_boot=1000, alpha=0.05) {
  T_obs <- nrow(returns_mat); m <- ncol(returns_mat)
  mu_hat <- colMeans(returns_mat)

  # Consistent estimate of omega (standard errors)
  omega_sq <- apply(returns_mat, 2, var) / T_obs

  # SPA statistic (studentized version)
  # Use only strategies with non-negative performance for max
  t_stat_all <- mu_hat / (sqrt(omega_sq) + 1e-10)
  T_spa <- max(pmax(mu_hat, 0) / (sqrt(omega_sq)+1e-10))

  # Bootstrap null distribution (uses MA block bootstrap)
  block_size <- max(1, round(T_obs^(1/3)))
  boot_spa <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    boot_ret <- matrix(0, T_obs, m)
    t_idx <- 1
    while (t_idx <= T_obs) {
      block_start <- sample(seq_len(T_obs), 1)
      blk_len <- min(block_size, T_obs - t_idx + 1)
      boot_idx <- (block_start + seq_len(blk_len) - 2) %% T_obs + 1
      boot_ret[t_idx:(t_idx+blk_len-1), ] <- returns_mat[boot_idx, ]
      t_idx <- t_idx + blk_len
    }
    boot_mu <- colMeans(boot_ret)
    # Use consistent subset (only strategies not too poor in sample)
    lb <- -sqrt(omega_sq) * sqrt(2 * log(log(T_obs)))
    boot_subset <- pmax(boot_mu - pmax(mu_hat, lb), 0) / (sqrt(omega_sq)+1e-10)
    boot_spa[b] <- max(boot_subset)
  }

  p_spa <- mean(boot_spa >= T_spa)

  cat("=== Hansen's SPA Test ===\n")
  cat(sprintf("SPA statistic: %.3f, p-value: %.4f\n", T_spa, p_spa))
  cat(sprintf("Result: %s\n", ifelse(p_spa<alpha, "SIGNIFICANT alpha", "No evidence of alpha")))

  invisible(list(T_spa=T_spa, p_spa=p_spa, boot_spa=boot_spa))
}

# -----------------------------------------------------------------------------
# 6. VARIANCE RATIO TESTS
# -----------------------------------------------------------------------------

#' Lo-MacKinlay Variance Ratio Test
#' Tests whether price changes are uncorrelated (random walk H0)
#' VR(q) = Var(q-period return) / (q * Var(1-period return)) = 1 under RW
#' VR > 1: positive autocorrelation (momentum); VR < 1: mean-reversion
#' @param prices price series
#' @param q aggregation period
variance_ratio_test <- function(prices, q=5, alpha=0.05) {
  n  <- length(prices)
  T_r <- n - 1
  r  <- diff(log(prices))
  mu_hat <- mean(r)

  # Variance of 1-period returns
  sigma2_1 <- sum((r - mu_hat)^2) / (T_r - 1)

  # Variance of q-period returns
  r_q <- sapply(1:(T_r - q + 1), function(t) sum(r[t:(t+q-1)]))
  mu_q <- mean(r_q)
  sigma2_q <- sum((r_q - mu_q)^2) / (length(r_q) - 1)

  # Variance ratio
  VR <- sigma2_q / (q * sigma2_1)

  # Heteroskedasticity-consistent Z-statistic (Lo-MacKinlay)
  delta_j <- function(j) {
    numer <- sum((r[(j+1):T_r] - mu_hat)^2 * (r[1:(T_r-j)] - mu_hat)^2)
    denom <- sum((r - mu_hat)^2)^2 / T_r
    T_r * numer / denom
  }
  theta_q <- sum(sapply(1:(q-1), function(j) (2*(q-j)/q)^2 * delta_j(j)))
  Z_star  <- (VR - 1) * sqrt(T_r) / sqrt(theta_q + 1e-10)
  p_val   <- 2 * pnorm(-abs(Z_star))

  cat(sprintf("=== Variance Ratio Test (q=%d) ===\n", q))
  cat(sprintf("VR(q) = %.4f\n", VR))
  cat(sprintf("Interpretation: %s\n",
              ifelse(VR>1.05, "Positive autocorrelation (momentum)",
              ifelse(VR<0.95, "Negative autocorrelation (mean-reversion)",
                     "Approximately random walk"))))
  cat(sprintf("Z* = %.3f, p = %.4f\n", Z_star, p_val))
  cat(sprintf("Result: %s at alpha=%.2f\n",
              ifelse(p_val<alpha, "REJECT random walk", "FAIL TO REJECT random walk"), alpha))

  invisible(list(VR=VR, Z=Z_star, p=p_val, q=q))
}

#' Multiple q variance ratio test
variance_ratio_multiple <- function(prices, q_vals=c(2,4,8,16)) {
  results <- lapply(q_vals, function(q) variance_ratio_test(prices, q))
  df <- data.frame(
    q   = q_vals,
    VR  = sapply(results, `[[`, "VR"),
    Z   = sapply(results, `[[`, "Z"),
    p   = sapply(results, `[[`, "p")
  )
  cat("\n=== Multiple VR Test Summary ===\n")
  print(df)
  invisible(df)
}

# -----------------------------------------------------------------------------
# 7. COINTEGRATION TESTS
# -----------------------------------------------------------------------------

#' Engle-Granger two-step cointegration test
#' Step 1: Regress y on x, get residuals
#' Step 2: Test if residuals are stationary (ADF test)
#' If residuals are I(0), series are cointegrated -- fundamental for pairs trading
#' @param y,x price series for the pair
engle_granger_test <- function(y, x, alpha=0.05) {
  n <- length(y)
  # Step 1: cointegrating regression
  X_mat <- cbind(1, x)
  b     <- solve(t(X_mat)%*%X_mat) %*% t(X_mat) %*% y
  resid <- y - X_mat %*% b
  beta  <- b[2]

  # Step 2: ADF test on residuals
  adf <- adf_test(resid, lags=1)

  cat("=== Engle-Granger Cointegration Test ===\n")
  cat(sprintf("Cointegrating coefficient: %.4f\n", beta))
  cat(sprintf("Residual ADF statistic: %.3f\n", adf$stat))
  # Critical values for EG test (different from standard ADF; MacKinnon 1990)
  # Approximate 5% critical value for n=50-500: -3.37
  crit_val <- -3.37
  cat(sprintf("Critical value (5%%): %.3f\n", crit_val))
  cointegrated <- adf$stat < crit_val
  cat(sprintf("Conclusion: %s\n",
              ifelse(cointegrated, "COINTEGRATED (pairs trade possible)",
                     "NOT cointegrated")))
  invisible(list(beta=beta, resid=resid, adf=adf, cointegrated=cointegrated))
}

#' Augmented Dickey-Fuller (ADF) test for unit root
#' H0: unit root (non-stationary)
#' @param x time series
#' @param lags number of lagged difference terms
adf_test <- function(x, lags=1, trend=FALSE) {
  n  <- length(x)
  dx <- diff(x)
  T_eff <- n - lags - 1

  # Build regression: dx_t = alpha + beta*x_{t-1} + sum gamma_l * dx_{t-l} + eps
  y_reg <- dx[(lags+1):length(dx)]
  X_reg <- matrix(0, length(y_reg), 1 + lags + (if(trend) 1 else 0))
  X_reg[, 1] <- x[(lags+1):(n-1)]  # lagged level
  for (l in seq_len(lags)) {
    X_reg[, l+1] <- dx[(lags+1-l):(length(dx)-l)]
  }
  if (trend) X_reg[, ncol(X_reg)] <- seq_len(nrow(X_reg))

  X_aug <- cbind(1, X_reg)
  b  <- solve(t(X_aug)%*%X_aug) %*% t(X_aug) %*% y_reg
  res <- y_reg - X_aug %*% b
  se2 <- sum(res^2) / (length(y_reg) - ncol(X_aug))
  var_b <- se2 * solve(t(X_aug)%*%X_aug)
  se_beta <- sqrt(var_b[2,2])  # se for x_{t-1} coefficient

  # ADF statistic: t-stat for beta = 0 (unit root)
  adf_stat <- b[2] / se_beta

  # Critical values (MacKinnon 1994, constant only, n~250)
  cv <- c("1%"=-3.43, "5%"=-2.86, "10%"=-2.57)

  cat(sprintf("ADF test: statistic=%.3f  (cv: 1%%=%.2f, 5%%=%.2f)\n",
              adf_stat, cv["1%"], cv["5%"]))
  invisible(list(stat=adf_stat, cv=cv, coef=b, lags=lags))
}

# -----------------------------------------------------------------------------
# 8. STRUCTURAL BREAK TESTS
# -----------------------------------------------------------------------------

#' CUSUM test for structural breaks (Brown, Durbin, Evans 1975)
#' Tests if OLS recursive residuals are stable over time
#' Significant CUSUM = structural break in relationship
#' @param y dependent variable
#' @param X design matrix (should include intercept)
cusum_test <- function(y, X, alpha=0.05) {
  n  <- nrow(X); k <- ncol(X)
  # Recursive OLS residuals
  w <- numeric(n)
  for (t in (k+1):n) {
    X_t <- X[1:t, , drop=FALSE]; y_t <- y[1:t]
    b_t <- tryCatch(solve(t(X_t)%*%X_t)%*%t(X_t)%*%y_t, error=function(e) NULL)
    if (is.null(b_t)) { w[t] <- NA; next }
    # One-step ahead prediction error
    X_new <- X[t, , drop=FALSE]
    sigma_t <- sqrt(sum((y_t - X_t%*%b_t)^2) / (t-k))
    leverage <- as.numeric(1 + X_new %*% solve(t(X_t[-t,,drop=F])%*%X_t[-t,,drop=F]+diag(k)*1e-8) %*% t(X_new))
    w[t] <- if (!is.na(sigma_t) && sigma_t > 0) (y[t] - sum(X_new * b_t)) / (sigma_t * sqrt(leverage)) else NA
  }

  w_valid <- w[(k+1):n]
  T_eff   <- sum(!is.na(w_valid))
  cusum   <- cumsum(ifelse(is.na(w_valid), 0, w_valid)) / sqrt(T_eff)

  # Critical value: 0.948*sqrt(T) for 5% level
  crit <- 0.948
  max_cusum <- max(abs(cusum), na.rm=TRUE)
  break_idx  <- which.max(abs(cusum)) + k

  cat("=== CUSUM Structural Break Test ===\n")
  cat(sprintf("Max |CUSUM|: %.3f (critical=%.3f at 5%%)\n", max_cusum, crit))
  cat(sprintf("Break location (approx): t=%d of n=%d\n", break_idx, n))
  cat(sprintf("Result: %s\n",
              ifelse(max_cusum>crit, "BREAK DETECTED", "No structural break")))

  invisible(list(cusum=cusum, max_cusum=max_cusum, break_idx=break_idx, crit=crit))
}

#' Chow test for a known breakpoint
chow_test <- function(y, X, break_point) {
  n <- length(y); k <- ncol(X)
  y1 <- y[1:break_point]; X1 <- X[1:break_point, ]
  y2 <- y[(break_point+1):n]; X2 <- X[(break_point+1):n, ]

  rss_all  <- sum(lm.fit(X, y)$residuals^2)
  rss1     <- sum(lm.fit(X1, y1)$residuals^2)
  rss2     <- sum(lm.fit(X2, y2)$residuals^2)

  F_stat <- ((rss_all - rss1 - rss2) / k) / ((rss1 + rss2) / (n - 2*k))
  p_val  <- pf(F_stat, df1=k, df2=n-2*k, lower.tail=FALSE)

  cat(sprintf("=== Chow Test (break at t=%d) ===\n", break_point))
  cat(sprintf("F(%d, %d) = %.3f, p = %.4f\n", k, n-2*k, F_stat, p_val))
  cat(sprintf("Result: %s\n", ifelse(p_val<0.05, "STRUCTURAL BREAK confirmed", "No break")))
  invisible(list(F=F_stat, p=p_val, rss_all=rss_all, rss1=rss1, rss2=rss2))
}

#' Andrews (1993) sup-Wald test: unknown breakpoint
andrews_supwald <- function(y, X, trim=0.15, alpha=0.05) {
  n <- length(y); k <- ncol(X)
  t_range <- seq(round(n*trim), round(n*(1-trim)))
  wald_stats <- sapply(t_range, function(bp) {
    res <- chow_test(y, X, bp)
    res$F
  })

  sup_W  <- max(wald_stats)
  break_est <- t_range[which.max(wald_stats)]

  # Approximate 5% critical value (Andrews 1993, k params)
  # Simplified: use chi2(k)/k * 2 approximation
  crit <- qchisq(0.95, df=k) / k * 1.5

  cat("=== Andrews Sup-Wald Test ===\n")
  cat(sprintf("Sup-Wald: %.3f, critical (approx 5%%): %.3f\n", sup_W, crit))
  cat(sprintf("Estimated break: t=%d\n", break_est))
  invisible(list(sup_W=sup_W, break_est=break_est, wald_series=wald_stats, t_range=t_range))
}

# -----------------------------------------------------------------------------
# 9. COMPREHENSIVE HYPOTHESIS TESTING PIPELINE
# -----------------------------------------------------------------------------

#' Run all relevant tests for a strategy
run_hypothesis_tests <- function(strategy_returns, signal=NULL, prices=NULL,
                                   benchmark_returns=NULL) {
  cat("=============================================================\n")
  cat("HYPOTHESIS TESTING SUITE\n")
  cat(sprintf("n = %d observations\n\n", length(strategy_returns)))

  # 1. Basic return tests
  cat("--- Return Distribution Tests ---\n")
  tt <- one_sample_ttest(strategy_returns)
  wr <- wilcoxon_signed_rank(strategy_returns)

  # 2. Variance ratio (random walk test)
  if (!is.null(prices)) {
    cat("\n--- Variance Ratio Tests ---\n")
    vr_res <- variance_ratio_multiple(prices)
  }

  # 3. Cointegration (if benchmark provided)
  if (!is.null(benchmark_returns) && length(benchmark_returns)==length(strategy_returns)) {
    cat("\n--- EG Cointegration (strategy vs benchmark) ---\n")
    # Work on cumulative return levels
    y_cum <- cumsum(strategy_returns); x_cum <- cumsum(benchmark_returns)
    eg_res <- engle_granger_test(y_cum, x_cum)
  }

  # 4. Structural break in returns
  cat("\n--- Structural Break Test ---\n")
  n <- length(strategy_returns)
  X_const <- matrix(1, n, 1)
  cusum_res <- cusum_test(strategy_returns, X_const)

  # 5. Permutation test
  if (!is.null(signal)) {
    cat("\n--- Permutation Test ---\n")
    perm_res <- permutation_test_strategy(signal, strategy_returns)
  }

  cat("\n=== SUMMARY ===\n")
  cat(sprintf("t-test p-value: %.4f (%s)\n", tt$p,
              ifelse(tt$p<0.05, "SIGNIFICANT", "not significant")))
  cat(sprintf("Wilcoxon p-value: %.4f\n", wr$p))
  cat(sprintf("Structural break: %s\n",
              ifelse(cusum_res$max_cusum > 0.948, "DETECTED", "not detected")))

  invisible(list(ttest=tt, wilcoxon=wr, cusum=cusum_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 500
# # Strategy returns with slight positive mean
# strategy_ret <- rnorm(n, mean=0.0005, sd=0.015)
# prices_btc   <- cumsum(c(30000, rnorm(n-1, 0, 500)))
# signal_vals  <- rnorm(n)
# run_hypothesis_tests(strategy_ret, signal=signal_vals, prices=abs(prices_btc))

# =============================================================================
# EXTENDED HYPOTHESIS TESTING: Sequential Testing, Bayesian Hypothesis Tests,
# Equivalence Testing, Distributional Tests, and False Discovery Control
# =============================================================================

# -----------------------------------------------------------------------------
# Sequential Probability Ratio Test (SPRT): Wald's test for strategy monitoring
# Allows stopping early when evidence is strong -- no need to wait for N=1000
# Used for live strategy monitoring: shut down or confirm strategy sooner
# -----------------------------------------------------------------------------
sprt_test <- function(returns, mu0 = 0, mu1 = NULL, sigma = NULL,
                       alpha = 0.05, beta = 0.10) {
  # H0: mu = mu0 (no alpha), H1: mu = mu1 (positive alpha)
  n <- length(returns)
  if (is.null(sigma)) sigma <- sd(returns)
  if (is.null(mu1))   mu1   <- mu0 + 0.5 * sigma  # half-sigma effect

  # Boundaries: A = (1-beta)/alpha (reject H0), B = beta/(1-alpha) (accept H0)
  A <- log((1 - beta) / alpha)
  B <- log(beta / (1 - alpha))

  # Log-likelihood ratio: sum of log [f1(x_t) / f0(x_t)]
  llr <- cumsum((mu1 - mu0) / sigma^2 * returns - (mu1^2 - mu0^2) / (2*sigma^2))

  # Decision at each step
  decision <- rep("continue", n)
  decision[llr >= A] <- "reject H0 (alpha detected)"
  decision[llr <= B] <- "accept H0 (no alpha)"

  # First stopping time
  stop_time_reject <- which(llr >= A)[1]
  stop_time_accept <- which(llr <= B)[1]
  stop_time <- min(stop_time_reject, stop_time_accept, na.rm=TRUE)
  if (length(stop_time) == 0) stop_time <- NA

  list(
    llr_path = llr, decision_path = decision,
    upper_boundary = A, lower_boundary = B,
    stopping_time = stop_time,
    final_decision = if (!is.na(stop_time)) decision[stop_time] else "inconclusive",
    pct_early_stop = if (!is.na(stop_time)) stop_time/n else 1.0,
    mu0 = mu0, mu1 = mu1, alpha = alpha, beta = beta
  )
}

# -----------------------------------------------------------------------------
# Equivalence Testing (TOST): Two One-Sided Tests
# Tests whether an effect is practically zero (within epsilon bounds)
# Useful for confirming a strategy is market-neutral or has no momentum
# TOST rejects the null of practical equivalence if effect is large
# -----------------------------------------------------------------------------
tost_test <- function(x, mu0 = 0, epsilon = NULL, alpha = 0.05) {
  # Null: |mean(x) - mu0| >= epsilon (not equivalent)
  # Reject null (conclude equivalent) if both one-sided tests significant
  n <- length(x)
  mean_x <- mean(x); se_x <- sd(x) / sqrt(n)

  if (is.null(epsilon)) epsilon <- 0.5 * se_x * qt(0.975, df=n-1)

  # Test 1: mean(x) - mu0 > -epsilon (lower equivalence bound)
  t1 <- (mean_x - mu0 + epsilon) / se_x
  p1 <- pt(t1, df = n-1, lower.tail = TRUE)  # one-sided p-value

  # Test 2: mean(x) - mu0 < +epsilon (upper equivalence bound)
  t2 <- (mean_x - mu0 - epsilon) / se_x
  p2 <- pt(t2, df = n-1, lower.tail = FALSE)

  # Conclude equivalence if max(p1, p2) < alpha
  p_tost <- max(p1, p2)

  list(
    mean_x = mean_x, se = se_x,
    epsilon = epsilon, n = n,
    t_stat_lower = t1, t_stat_upper = t2,
    p_value_lower = p1, p_value_upper = p2,
    p_value_tost = p_tost,
    equivalent = p_tost < alpha,
    ci_90 = c(mean_x - qt(0.95, n-1)*se_x, mean_x + qt(0.95, n-1)*se_x)
  )
}

# -----------------------------------------------------------------------------
# Kolmogorov-Smirnov Two-Sample Test: compare two return distributions
# Detects any difference in distribution shape, scale, or location
# Non-parametric: no distributional assumptions
# Useful for comparing strategy returns in different market regimes
# -----------------------------------------------------------------------------
ks_two_sample <- function(x, y, alternative = "two.sided") {
  n_x <- length(x); n_y <- length(y)
  combined <- sort(c(x, y))

  # Empirical CDFs
  ecdf_x <- stepfun(sort(x), c(0, (1:n_x)/n_x))
  ecdf_y <- stepfun(sort(y), c(0, (1:n_y)/n_y))

  # D statistic: max |F_x(t) - F_y(t)|
  diffs <- abs(ecdf_x(combined) - ecdf_y(combined))
  D <- max(diffs)

  # Asymptotic p-value (Kolmogorov distribution)
  m_eff <- n_x * n_y / (n_x + n_y)
  lambda <- (sqrt(m_eff) + 0.12 + 0.11/sqrt(m_eff)) * D
  p_value <- 2 * sum((-1)^(1:100 - 1) * exp(-2 * (1:100)^2 * lambda^2))
  p_value <- max(0, min(1, p_value))

  list(
    D_statistic = D, p_value = p_value,
    n_x = n_x, n_y = n_y,
    significant = p_value < 0.05,
    mean_diff = mean(x) - mean(y),
    vol_diff = sd(x) - sd(y),
    median_diff = median(x) - median(y)
  )
}

# -----------------------------------------------------------------------------
# Anderson-Darling Test: more powerful than KS for detecting tail differences
# Weights deviations by 1/[F(1-F)], emphasizing the tails
# Particularly relevant for crypto where tail behavior is the key concern
# -----------------------------------------------------------------------------
anderson_darling_test <- function(x, distribution = "normal") {
  n <- length(x)
  x_sorted <- sort(x)

  if (distribution == "normal") {
    mu_hat  <- mean(x); sig_hat <- sd(x)
    F_vals  <- pnorm(x_sorted, mu_hat, sig_hat)
  } else if (distribution == "t") {
    # Fit t-distribution via MOM
    nu_hat <- max(2 * var(x) / (var(x) - 1), 4)
    F_vals <- pt(x_sorted / sd(x), df=nu_hat)
  } else {
    stop("distribution must be normal or t")
  }

  # Clamp to avoid log(0)
  F_vals <- pmax(pmin(F_vals, 1 - 1e-10), 1e-10)

  # AD statistic
  i <- 1:n
  A2 <- -n - mean((2*i - 1) * (log(F_vals) + log(1 - rev(F_vals))))

  # Correction factor for estimated parameters
  A2_adj <- A2 * (1 + 4/n - 25/n^2)

  # Approximate p-value (normal distribution case)
  p_value <- if (A2_adj <= 0.200) {
    1 - exp(-13.436 + 101.14*A2_adj - 223.73*A2_adj^2)
  } else if (A2_adj <= 0.340) {
    1 - exp(-8.318 + 42.796*A2_adj - 59.938*A2_adj^2)
  } else if (A2_adj <= 0.600) {
    exp(0.9177 - 4.279*A2_adj - 1.38*A2_adj^2)
  } else {
    exp(1.2937 - 5.709*A2_adj + 0.0186*A2_adj^2)
  }
  p_value <- max(0, min(1, p_value))

  list(
    A2 = A2, A2_adjusted = A2_adj, p_value = p_value,
    distribution = distribution,
    reject_normality = p_value < 0.05,
    sample_skewness = mean(((x-mean(x))/sd(x))^3),
    sample_kurtosis = mean(((x-mean(x))/sd(x))^4) - 3
  )
}

# -----------------------------------------------------------------------------
# Jarque-Bera Test for Normality: tests skewness and kurtosis jointly
# JB = n/6 * (S^2 + K^2/4) ~ chi^2(2) under normality
# -----------------------------------------------------------------------------
jarque_bera_test <- function(x) {
  n <- length(x)
  mu3 <- mean(((x - mean(x))/sd(x))^3)  # skewness
  mu4 <- mean(((x - mean(x))/sd(x))^4)  # kurtosis
  excess_k <- mu4 - 3

  JB <- n / 6 * (mu3^2 + excess_k^2 / 4)
  p_value <- pchisq(JB, df=2, lower.tail=FALSE)

  list(
    jb_statistic = JB, p_value = p_value,
    skewness = mu3, excess_kurtosis = excess_k,
    reject_normality = p_value < 0.05,
    interpretation = ifelse(p_value < 0.05,
      "Return distribution is non-normal (fat tails or skew detected)",
      "Cannot reject normality at 5%")
  )
}

# -----------------------------------------------------------------------------
# Bayesian t-test: compare means using Bayesian framework
# Computes Bayes Factor (BF10) comparing H1 (mu != 0) vs H0 (mu = 0)
# BF > 3: moderate evidence for H1; BF > 10: strong evidence
# -----------------------------------------------------------------------------
bayesian_ttest <- function(x, mu0 = 0, r_scale = 0.707) {
  # Jeffreys-Zellner-Siow (JZS) prior: beta ~ Cauchy(0, r_scale)
  # Analytic approximation via Wetzels & Wagenmakers (2012)
  n <- length(x); t_stat <- (mean(x) - mu0) / (sd(x)/sqrt(n))

  # Numerical integration for Bayes Factor
  log_BF_fn <- function(g) {
    # g: prior variance parameter
    (1 + n*g)^(-0.5) * exp(-t_stat^2*n*g / (2*(1+n*g))) *
    (1 + g/r_scale^2)^(-1.5)  # Cauchy prior on delta
  }

  # Integrate out g from 0 to Inf
  result <- tryCatch(
    integrate(log_BF_fn, 0, Inf, rel.tol=1e-4),
    error = function(e) list(value=1)
  )

  # BF10 = marginal likelihood under H1 / marginal likelihood under H0
  BF10 <- result$value

  list(
    t_statistic = t_stat,
    n = n, df = n-1,
    p_value = 2*pt(-abs(t_stat), df=n-1),
    BF10 = BF10,
    BF01 = 1/BF10,
    evidence = ifelse(BF10 > 100, "extreme for H1",
               ifelse(BF10 > 10, "strong for H1",
               ifelse(BF10 > 3, "moderate for H1",
               ifelse(BF10 > 1, "anecdotal for H1",
               ifelse(BF10 > 1/3, "anecdotal for H0", "moderate for H0")))))
  )
}

# Extended hypothesis testing examples:
# sprt_res <- sprt_test(strategy_returns, mu0=0, mu1=0.0005)
# tost_res <- tost_test(residuals_vec, epsilon=0.001)
# ks_res   <- ks_two_sample(returns_bull, returns_bear)
# ad_res   <- anderson_darling_test(returns_vec, distribution="normal")
# jb_res   <- jarque_bera_test(returns_vec)
# bt_res   <- bayesian_ttest(strategy_returns, mu0=0)
