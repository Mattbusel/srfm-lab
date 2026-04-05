# =============================================================================
# econometrics.R
# Advanced econometrics for trading research
# Base R only -- no external packages
# =============================================================================
# Financial intuition: VAR models estimate how macro/on-chain variables
# interact dynamically. IRFs show how a shock (e.g., BTC price spike) ripples
# through other variables over time. FEVD tells us how much of ETH's
# volatility is explained by BTC shocks vs its own dynamics.
# Quantile regression reveals if BTC momentum predicts returns differentially
# in the tail (a key input for tail risk management).
# =============================================================================

# -----------------------------------------------------------------------------
# 1. VAR MODEL ESTIMATION AND GRANGER CAUSALITY
# -----------------------------------------------------------------------------

#' Estimate Vector Autoregression (VAR) model of order p
#' Y_t = A_1*Y_{t-1} + ... + A_p*Y_{t-p} + c + e_t
#' Estimated equation by equation via OLS
#' @param Y T x K matrix of variables
#' @param p lag order
estimate_var <- function(Y, p=1) {
  T_obs <- nrow(Y); K <- ncol(Y)
  var_names <- colnames(Y)
  if (is.null(var_names)) var_names <- paste0("Y", seq_len(K))
  T_eff <- T_obs - p

  # Build regressor matrix: constant + p lags of each variable
  X <- matrix(1, T_eff, 1 + K*p)  # intercept + lags
  col_idx <- 2
  for (lag in seq_len(p)) {
    X[, col_idx:(col_idx+K-1)] <- Y[(p-lag+1):(T_obs-lag), ]
    col_idx <- col_idx + K
  }
  Y_dep <- Y[(p+1):T_obs, ]

  # OLS for each equation
  B <- solve(t(X)%*%X) %*% t(X) %*% Y_dep  # (1+K*p) x K coefficient matrix
  resid <- Y_dep - X %*% B
  Sigma_u <- t(resid) %*% resid / (T_eff - 1 - K*p)

  # Coefficient arrays: A_l[i,j] = effect of Y_j,t-l on Y_i,t
  A_list <- vector("list", p)
  for (lag in seq_len(p)) {
    row_start <- 1 + (lag-1)*K + 1  # skip intercept (row 1)
    A_list[[lag]] <- t(B[row_start:(row_start+K-1), ])
  }
  intercept <- B[1, ]  # intercept terms

  cat("=== VAR(%d) Estimation ===\n", p)
  cat(sprintf("Variables: %s, T=%d, T_eff=%d\n",
              paste(var_names, collapse=", "), T_obs, T_eff))
  cat(sprintf("Sigma_u diag: %s\n",
              paste(round(diag(Sigma_u), 6), collapse=", ")))

  # Information criteria
  log_det_Sigma <- log(det(Sigma_u))
  n_params <- K^2 * p + K  # coefficients (excluding intercept for simplicity)
  AIC <- log_det_Sigma + 2 * n_params / T_eff
  BIC <- log_det_Sigma + log(T_eff) * n_params / T_eff

  cat(sprintf("AIC=%.4f, BIC=%.4f\n", AIC, BIC))

  invisible(list(A=A_list, intercept=intercept, Sigma_u=Sigma_u,
                 B=B, X=X, Y=Y_dep, resid=resid,
                 var_names=var_names, p=p, K=K, T_eff=T_eff,
                 AIC=AIC, BIC=BIC))
}

#' VAR lag order selection via AIC/BIC
var_lag_selection <- function(Y, p_max=8) {
  criteria <- lapply(seq_len(p_max), function(p) {
    tryCatch({
      fit <- estimate_var(Y, p)
      list(p=p, AIC=fit$AIC, BIC=fit$BIC)
    }, error=function(e) list(p=p, AIC=Inf, BIC=Inf))
  })
  df <- data.frame(
    p   = sapply(criteria, `[[`, "p"),
    AIC = sapply(criteria, `[[`, "AIC"),
    BIC = sapply(criteria, `[[`, "BIC")
  )
  cat("=== VAR Lag Selection ===\n")
  print(df)
  cat(sprintf("Optimal p (AIC): %d\n", df$p[which.min(df$AIC)]))
  cat(sprintf("Optimal p (BIC): %d\n", df$p[which.min(df$BIC)]))
  invisible(df)
}

#' Granger causality test: does variable i Granger-cause variable j?
#' F-test comparing restricted (no lags of i) vs unrestricted VAR equation for j
granger_causality <- function(var_fit, cause_var, effect_var) {
  K <- var_fit$K; p <- var_fit$p
  var_names <- var_fit$var_names
  T_eff <- var_fit$T_eff

  i_idx <- which(var_names == cause_var)
  j_idx <- which(var_names == effect_var)
  if (length(i_idx)==0 || length(j_idx)==0) stop("Variable not found")

  # Unrestricted: all lags of all variables
  X_u <- var_fit$X
  y_j <- var_fit$Y[, j_idx]
  b_u <- solve(t(X_u)%*%X_u) %*% t(X_u) %*% y_j
  rss_u <- sum((y_j - X_u%*%b_u)^2)
  df_u  <- T_eff - ncol(X_u)

  # Restricted: remove lags of cause_var (columns 1+(i_idx-1)*... offsets)
  lag_cols <- numeric(p)
  for (lag in seq_len(p)) {
    lag_cols[lag] <- 1 + (lag-1)*K + i_idx  # column index for Y_i,t-lag
  }
  X_r <- X_u[, -lag_cols, drop=FALSE]
  b_r <- solve(t(X_r)%*%X_r) %*% t(X_r) %*% y_j
  rss_r <- sum((y_j - X_r%*%b_r)^2)
  df_r  <- T_eff - ncol(X_r)

  F_stat <- ((rss_r - rss_u)/p) / (rss_u/df_u)
  p_val  <- pf(F_stat, df1=p, df2=df_u, lower.tail=FALSE)

  cat(sprintf("Granger causality: %s -> %s\n", cause_var, effect_var))
  cat(sprintf("F(%d, %d) = %.3f, p = %.4f\n", p, df_u, F_stat, p_val))
  cat(sprintf("Conclusion: %s Granger-causes %s at 5%%: %s\n",
              cause_var, effect_var, ifelse(p_val<0.05, "YES", "NO")))

  invisible(list(F=F_stat, p=p_val, cause=cause_var, effect=effect_var))
}

# -----------------------------------------------------------------------------
# 2. STRUCTURAL VAR (SVAR) IDENTIFICATION
# -----------------------------------------------------------------------------

#' SVAR via Cholesky identification (recursive ordering)
#' Lower triangular A_0 (Cholesky): first variable affects all, last affects none contemporaneously
#' @param var_fit fitted VAR model
svar_cholesky <- function(var_fit) {
  Sigma_u <- var_fit$Sigma_u
  # Cholesky decomposition: Sigma_u = P * P'
  P <- t(chol(Sigma_u))  # lower triangular
  # Structural shocks: epsilon = P^{-1} * u
  P_inv <- solve(P)

  cat("=== SVAR (Cholesky Identification) ===\n")
  cat("Impact matrix P (lower triangular):\n")
  print(round(P, 6))
  cat("P_inv (structural shock loadings):\n")
  print(round(P_inv, 4))

  invisible(list(P=P, P_inv=P_inv, Sigma_u=Sigma_u))
}

# -----------------------------------------------------------------------------
# 3. IMPULSE RESPONSE FUNCTIONS WITH BOOTSTRAP BANDS
# -----------------------------------------------------------------------------

#' Compute impulse response functions (IRFs)
#' IRF_{ij}(h) = response of variable i to a unit shock in variable j at horizon h
#' @param var_fit fitted VAR model
#' @param svar_P structural impact matrix (from Cholesky)
#' @param n_periods number of periods ahead
irf_var <- function(var_fit, svar_P=NULL, n_periods=20) {
  K <- var_fit$K; p <- var_fit$p
  A_list <- var_fit$A

  if (is.null(svar_P)) svar_P <- diag(K)  # no structural identification

  # Compute VMA coefficients: Phi_h where Y_t = sum_h Phi_h * e_{t-h}
  Phi <- array(0, dim=c(n_periods+1, K, K))
  Phi[1,,] <- diag(K)  # Phi_0 = I

  for (h in seq_len(n_periods)) {
    for (lag in seq_len(min(h, p))) {
      Phi[h+1,,] <- Phi[h+1,,] + Phi[h-lag+1,,] %*% A_list[[lag]]
    }
  }

  # Structural IRFs: Theta_h = Phi_h * P
  Theta <- array(0, dim=c(n_periods+1, K, K))
  for (h in 0:n_periods) {
    Theta[h+1,,] <- Phi[h+1,,] %*% svar_P
  }

  invisible(list(Phi=Phi, Theta=Theta, n_periods=n_periods,
                 var_names=var_fit$var_names))
}

#' Bootstrap confidence bands for IRFs
irf_bootstrap_bands <- function(var_fit, n_boot=500, n_periods=20,
                                  alpha=0.10) {
  K <- var_fit$K; p <- var_fit$p
  T_eff <- var_fit$T_eff
  resid <- var_fit$resid

  # Center residuals
  resid_c <- scale(resid, center=TRUE, scale=FALSE)

  # Point estimate
  svar_P  <- svar_cholesky(var_fit)$P
  irf_obs <- irf_var(var_fit, svar_P, n_periods)$Theta

  # Bootstrap
  boot_irfs <- array(0, dim=c(n_boot, n_periods+1, K, K))
  for (b in seq_len(n_boot)) {
    # Resample residuals
    idx_boot <- sample(seq_len(T_eff), T_eff, replace=TRUE)
    u_boot   <- resid_c[idx_boot, ]

    # Generate bootstrap data from VAR
    Y_boot <- matrix(0, T_eff + p, K)
    Y_boot[1:p, ] <- var_fit$Y[1:p, ]  # initial values
    for (t in seq_len(T_eff)) {
      Y_t <- var_fit$intercept
      for (lag in seq_len(p)) {
        Y_t <- Y_t + var_fit$A[[lag]] %*% Y_boot[p+t-lag, ]
      }
      Y_boot[p+t, ] <- Y_t + u_boot[t, ]
    }

    # Re-estimate VAR on bootstrap data
    fit_b <- tryCatch(estimate_var(Y_boot, p), error=function(e) NULL)
    if (is.null(fit_b)) next
    svar_P_b <- tryCatch(svar_cholesky(fit_b)$P, error=function(e) NULL)
    if (is.null(svar_P_b)) next
    boot_irfs[b,,,] <- irf_var(fit_b, svar_P_b, n_periods)$Theta
  }

  # Confidence bands
  irf_lower <- array(apply(boot_irfs, c(2,3,4), quantile, probs=alpha/2), dim(irf_obs))
  irf_upper <- array(apply(boot_irfs, c(2,3,4), quantile, probs=1-alpha/2), dim(irf_obs))

  cat("=== IRF Bootstrap Bands ===\n")
  cat(sprintf("Confidence level: %.0f%%, Bootstrap paths: %d\n", 100*(1-alpha), n_boot))

  invisible(list(irf_median=irf_obs, irf_lower=irf_lower, irf_upper=irf_upper,
                 var_names=var_fit$var_names))
}

# -----------------------------------------------------------------------------
# 4. VARIANCE DECOMPOSITION (FEVD)
# -----------------------------------------------------------------------------

#' Forecast Error Variance Decomposition (FEVD)
#' FEVD(i,j,h): fraction of variance in h-step forecast error of variable i
#'              that is due to shock j
#' @param irf_result output of irf_var
fevd <- function(irf_result, n_periods=20) {
  Theta <- irf_result$Theta  # (n_periods+1) x K x K
  K     <- dim(Theta)[2]
  var_names <- irf_result$var_names

  fevd_arr <- array(0, dim=c(n_periods, K, K))

  for (h in seq_len(n_periods)) {
    # Total forecast error variance matrix at horizon h
    total_var <- matrix(0, K, K)
    for (s in 0:(h-1)) {
      total_var <- total_var + Theta[s+1,,] %*% t(Theta[s+1,,])
    }
    # Contribution of shock j to variable i
    for (j in seq_len(K)) {
      theta_j <- Theta[1:h, , j]  # h x K
      if (h == 1) theta_j <- matrix(theta_j, 1, K)
      contrib_j <- colSums(theta_j^2)  # K-vector
      for (i in seq_len(K)) {
        fevd_arr[h, i, j] <- contrib_j[i] / (diag(total_var)[i] + 1e-20)
      }
    }
  }

  cat("=== Forecast Error Variance Decomposition ===\n")
  cat("FEVD at horizon 10 (rows=variable, cols=shock source):\n")
  fevd_h10 <- fevd_arr[min(10, n_periods), , ]
  rownames(fevd_h10) <- var_names; colnames(fevd_h10) <- var_names
  print(round(fevd_h10, 4))

  invisible(list(fevd=fevd_arr, var_names=var_names))
}

# -----------------------------------------------------------------------------
# 5. PANEL DATA MODELS
# -----------------------------------------------------------------------------

#' Fixed effects panel regression (within estimator)
#' y_it = alpha_i + X_it*beta + e_it
#' Removes individual fixed effects by demeaning within each unit
#' @param Y T*N-vector of outcomes (pooled)
#' @param X T*N x K matrix of regressors
#' @param panel_id T*N-vector of unit identifiers
panel_fixed_effects <- function(Y, X, panel_id) {
  N_units <- length(unique(panel_id))
  K <- ncol(X)

  # Within transformation: demean by unit
  Y_dm <- numeric(length(Y))
  X_dm <- matrix(0, length(Y), K)

  for (unit in unique(panel_id)) {
    idx <- which(panel_id == unit)
    Y_dm[idx] <- Y[idx] - mean(Y[idx])
    for (k in seq_len(K)) X_dm[idx, k] <- X[idx, k] - mean(X[idx, k])
  }

  # OLS on demeaned data
  b_fe <- solve(t(X_dm)%*%X_dm) %*% t(X_dm) %*% Y_dm
  resid_fe <- Y_dm - X_dm %*% b_fe
  n_obs  <- length(Y); df <- n_obs - N_units - K
  sigma2 <- sum(resid_fe^2) / df
  var_b  <- sigma2 * solve(t(X_dm)%*%X_dm)
  se_b   <- sqrt(diag(var_b))
  t_stats <- b_fe / se_b

  cat("=== Fixed Effects Panel Regression ===\n")
  cat(sprintf("N units=%d, T_avg=%.1f, K=%d\n",
              N_units, length(Y)/N_units, K))
  for (k in seq_len(K)) {
    cat(sprintf("beta_%d: %.5f (se=%.5f, t=%.2f)\n",
                k, b_fe[k], se_b[k], t_stats[k]))
  }

  invisible(list(beta=b_fe, se=se_b, t_stats=t_stats, sigma2=sigma2))
}

#' Random effects panel regression (GLS)
panel_random_effects <- function(Y, X, panel_id) {
  N_units <- length(unique(panel_id))
  n_obs   <- length(Y)
  T_bar   <- n_obs / N_units
  K <- ncol(X)

  # OLS residuals (for variance component estimation)
  X_aug <- cbind(1, X)
  b_ols <- solve(t(X_aug)%*%X_aug) %*% t(X_aug) %*% Y
  e_ols <- Y - X_aug %*% b_ols
  sigma2_e <- sum(e_ols^2) / (n_obs - K - 1)

  # Between-group residuals
  Y_bar <- tapply(Y, panel_id, mean)[panel_id]
  X_bar <- apply(X, 2, function(x) tapply(x, panel_id, mean)[panel_id])
  e_between <- Y_bar - cbind(1, X_bar) %*% b_ols
  sigma2_u  <- max(mean(tapply(e_between, panel_id, function(e) e[1]^2)) - sigma2_e/T_bar, 0)

  # GLS transformation
  theta <- 1 - sqrt(sigma2_e / (T_bar * sigma2_u + sigma2_e))
  Y_re  <- Y - theta * Y_bar
  X_re  <- X - theta * X_bar
  X_re_aug <- cbind(1 - theta, X_re)

  b_re  <- solve(t(X_re_aug)%*%X_re_aug) %*% t(X_re_aug) %*% Y_re
  resid_re <- Y_re - X_re_aug %*% b_re
  sigma2_re <- sum(resid_re^2) / (n_obs - K - 1)
  se_re <- sqrt(diag(sigma2_re * solve(t(X_re_aug)%*%X_re_aug)))

  cat("=== Random Effects Panel Regression ===\n")
  cat(sprintf("theta (GLS weight): %.4f\n", theta))
  cat(sprintf("sigma2_e=%.6f, sigma2_u=%.6f\n", sigma2_e, sigma2_u))
  for (k in seq_len(K+1)) {
    cat(sprintf("beta_%d: %.5f (se=%.5f)\n", k-1, b_re[k], se_re[k]))
  }

  invisible(list(beta=b_re, se=se_re, theta=theta,
                 sigma2_e=sigma2_e, sigma2_u=sigma2_u))
}

# -----------------------------------------------------------------------------
# 6. INSTRUMENTAL VARIABLES (2SLS)
# -----------------------------------------------------------------------------

#' Two-Stage Least Squares (2SLS)
#' Stage 1: regress endogenous X on instruments Z
#' Stage 2: regress Y on fitted X_hat
#' Use case: endogeneity in volume-return regressions
#' @param Y outcome variable
#' @param X endogenous regressors (n x k_endog)
#' @param Z instruments (n x k_inst)
#' @param W exogenous controls (n x k_exog, optional)
tsls <- function(Y, X, Z, W=NULL) {
  n <- length(Y)
  if (!is.matrix(X)) X <- matrix(X, n, 1)
  if (!is.matrix(Z)) Z <- matrix(Z, n, 1)
  k_endog <- ncol(X); k_inst <- ncol(Z)

  # Check order condition: k_inst >= k_endog
  if (k_inst < k_endog) stop("Order condition: need at least as many instruments as endogenous vars")

  # Build full instrument matrix (Z and exogenous controls W)
  if (!is.null(W)) {
    Z_full <- cbind(1, W, Z)
    X_full <- cbind(1, W, X)
  } else {
    Z_full <- cbind(1, Z)
    X_full <- cbind(1, X)
  }

  # Stage 1: X = Z*pi + v
  X_hat <- Z_full %*% solve(t(Z_full)%*%Z_full) %*% t(Z_full) %*% X_full
  X_hat <- X_hat[, ncol(X_hat) - k_endog + seq_len(k_endog), drop=FALSE]

  # Stage 2: Y = X_hat*beta + epsilon
  if (!is.null(W)) {
    X2 <- cbind(1, W, X_hat)
  } else {
    X2 <- cbind(1, X_hat)
  }
  b_2sls <- solve(t(X2)%*%X2) %*% t(X2) %*% Y

  # Standard errors (using original X, not X_hat, in residuals)
  if (!is.null(W)) {
    X_orig <- cbind(1, W, X)
  } else {
    X_orig <- cbind(1, X)
  }
  resid <- Y - X_orig %*% b_2sls
  sigma2 <- sum(resid^2) / (n - ncol(X2))
  P_Z    <- Z_full %*% solve(t(Z_full)%*%Z_full) %*% t(Z_full)
  var_2sls <- sigma2 * solve(t(X_orig) %*% P_Z %*% X_orig)
  se_2sls  <- sqrt(diag(var_2sls))
  t_stats  <- b_2sls / se_2sls

  cat("=== 2SLS Instrumental Variables ===\n")
  cat(sprintf("n=%d, endogenous vars=%d, instruments=%d\n", n, k_endog, k_inst))
  for (k in seq_along(b_2sls)) {
    cat(sprintf("beta_%d: %.5f (se=%.5f, t=%.2f)\n",
                k-1, b_2sls[k], se_2sls[k], t_stats[k]))
  }

  invisible(list(beta=b_2sls, se=se_2sls, t_stats=t_stats,
                 resid=resid, X_hat=X_hat))
}

# -----------------------------------------------------------------------------
# 7. QUANTILE REGRESSION (KOENKER-BASSETT)
# -----------------------------------------------------------------------------

#' Quantile regression via interior point method (simplex via IRLS)
#' @param Y response variable
#' @param X design matrix (include intercept)
#' @param tau quantile level (e.g., 0.05 for 5th percentile)
quantile_regression <- function(Y, X, tau=0.5, max_iter=200, tol=1e-6) {
  n <- nrow(X); k <- ncol(X)
  # Initialize with OLS
  b <- solve(t(X)%*%X) %*% t(X) %*% Y

  for (iter in seq_len(max_iter)) {
    resid <- Y - X %*% b
    # IRLS weights: tau for positive residuals, 1-tau for negative
    w <- ifelse(resid > 0, tau, 1-tau) / (abs(resid) + 1e-8)
    W <- diag(as.numeric(w))
    b_new <- solve(t(X)%*%W%*%X) %*% t(X)%*%W%*%Y
    if (max(abs(b_new - b)) < tol) { b <- b_new; break }
    b <- b_new
  }

  resid <- Y - X %*% b
  # Asymptotic standard errors (Powell's sandwich)
  h_bw <- 1.06 * sd(resid) * n^(-1/5)  # bandwidth
  f_hat <- mean(abs(resid) <= h_bw) / (2*h_bw + 1e-10)  # kernel density at 0
  var_b <- tau*(1-tau) / f_hat^2 * solve(t(X)%*%X) %*% t(X)%*%X %*% solve(t(X)%*%X)
  se_b  <- sqrt(diag(var_b))
  t_stats <- b / se_b

  cat(sprintf("=== Quantile Regression (tau=%.2f) ===\n", tau))
  for (j in seq_len(k)) {
    cat(sprintf("beta_%d: %.5f (se=%.5f, t=%.2f)\n", j-1, b[j], se_b[j], t_stats[j]))
  }

  invisible(list(beta=b, se=se_b, t_stats=t_stats, resid=resid, tau=tau))
}

#' Quantile regression across multiple tau levels
quantile_process <- function(Y, X, taus=seq(0.05, 0.95, by=0.10)) {
  results <- lapply(taus, function(tau) quantile_regression(Y, X, tau))
  beta_mat <- do.call(cbind, lapply(results, `[[`, "beta"))
  colnames(beta_mat) <- paste0("tau=", taus)
  cat("=== Quantile Process ===\n")
  cat("Coefficient matrix across quantiles:\n")
  print(round(beta_mat, 4))
  invisible(list(betas=beta_mat, taus=taus, results=results))
}

# -----------------------------------------------------------------------------
# 8. ARFIMA (FRACTIONALLY INTEGRATED ARMA)
# -----------------------------------------------------------------------------

#' ARFIMA(0,d,0): fractional differencing
#' Long memory: d in (0,0.5) means slow decay of ACF (autocorrelation)
#' Crypto: realized volatility often shows long memory (d ~ 0.3-0.4)
#' @param x time series
#' @param d fractional differencing parameter (0 < d < 0.5)
fracdiff <- function(x, d) {
  n <- length(x)
  # Binomial coefficients pi_k = (-1)^k * C(d,k) = Gamma(k-d)/(Gamma(k+1)*Gamma(-d))
  max_k <- n - 1
  pi_k  <- numeric(max_k + 1)
  pi_k[1] <- 1  # pi_0 = 1
  for (k in seq_len(max_k)) {
    pi_k[k+1] <- pi_k[k] * (k - 1 - d) / k
  }
  # Apply fractional filter: (1-L)^d x = sum_k pi_k * x_{t-k}
  result <- numeric(n)
  for (t in seq_len(n)) {
    s <- 0
    for (k in 0:(t-1)) {
      s <- s + pi_k[k+1] * x[t-k]
    }
    result[t] <- s
  }
  result
}

#' Estimate d via GPH (Geweke-Porter-Hudak) log-periodogram regression
estimate_d_gph <- function(x, bandwidth_frac=0.65) {
  n  <- length(x)
  # Periodogram
  T_half <- floor(n / 2)
  x_c  <- x - mean(x)
  dft  <- fft(x_c)[2:(T_half+1)]
  pgram <- Mod(dft)^2 / n
  freq  <- seq_len(T_half) / n
  # Use only low-frequency ordinates (bandwidth m = n^0.65)
  m    <- floor(n^bandwidth_frac)
  pgram_m <- pgram[1:m]
  freq_m  <- freq[1:m]
  omega_m <- 2 * pi * freq_m

  # GPH regression: log(pgram) = c - 2d*log(2*sin(omega/2)) + error
  y_gph <- log(pgram_m)
  x_gph <- log(2 * sin(omega_m/2))
  X_gph <- cbind(1, x_gph)
  b_gph <- solve(t(X_gph)%*%X_gph) %*% t(X_gph) %*% y_gph
  d_hat <- -b_gph[2] / 2
  se_d  <- sqrt(pi^2/6 / sum((x_gph - mean(x_gph))^2))
  t_d   <- d_hat / se_d

  cat("=== GPH Long Memory Estimator ===\n")
  cat(sprintf("d_hat = %.4f (se=%.4f, t=%.2f)\n", d_hat, se_d, t_d))
  cat(sprintf("Classification: %s\n",
              ifelse(d_hat < 0, "Anti-persistent (mean-reverting)",
              ifelse(d_hat < 0.5, "Long memory (persistent ACF)",
                     "Non-stationary"))))
  invisible(list(d=d_hat, se=se_d, t=t_d))
}

# -----------------------------------------------------------------------------
# 9. COMPREHENSIVE ECONOMETRICS PIPELINE
# -----------------------------------------------------------------------------

#' Run full econometric analysis on a multivariate time series
run_econometrics <- function(Y, var_names=NULL, target_var=1,
                              n_irf_periods=20, n_boot_irf=200) {
  K <- ncol(Y); T_obs <- nrow(Y)
  if (is.null(var_names)) var_names <- paste0("Y", seq_len(K))
  colnames(Y) <- var_names

  cat("=============================================================\n")
  cat("ECONOMETRICS PIPELINE\n")
  cat(sprintf("Variables: %s\n", paste(var_names, collapse=", ")))
  cat(sprintf("T = %d observations\n\n", T_obs))

  # 1. Lag selection
  cat("--- VAR Lag Selection ---\n")
  lag_sel <- var_lag_selection(Y, p_max=min(6, floor(T_obs/10)))
  p_opt   <- lag_sel$p[which.min(lag_sel$BIC)]

  # 2. VAR estimation
  cat(sprintf("\n--- VAR(%d) Estimation ---\n", p_opt))
  var_fit <- estimate_var(Y, p=p_opt)

  # 3. Granger causality
  cat("\n--- Granger Causality Tests ---\n")
  for (i in seq_len(K)) {
    for (j in seq_len(K)) {
      if (i != j) granger_causality(var_fit, var_names[i], var_names[j])
    }
  }

  # 4. SVAR identification
  cat("\n--- SVAR Identification ---\n")
  svar_res <- svar_cholesky(var_fit)

  # 5. IRFs
  cat("\n--- Impulse Response Functions ---\n")
  irf_res <- irf_var(var_fit, svar_res$P, n_irf_periods)
  cat(sprintf("IRF computed for %d horizons\n", n_irf_periods))

  # 6. FEVD
  cat("\n--- Forecast Error Variance Decomposition ---\n")
  fevd_res <- fevd(irf_res, n_irf_periods)

  # 7. Long memory test on target variable
  cat(sprintf("\n--- Long Memory Test: %s ---\n", var_names[target_var]))
  gph_res <- estimate_d_gph(Y[, target_var])

  # 8. Quantile regression
  if (K >= 2) {
    cat("\n--- Quantile Regression ---\n")
    cat(sprintf("Regressing %s on %s\n", var_names[target_var], var_names[2]))
    qr_res <- quantile_process(Y[, target_var], cbind(1, Y[, 2]),
                                taus=c(0.10, 0.25, 0.50, 0.75, 0.90))
  }

  invisible(list(var=var_fit, svar=svar_res, irf=irf_res, fevd=fevd_res,
                 gph=gph_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 300
# # Simulate VAR(1) system: BTC, ETH, Funding Rate
# A_true <- matrix(c(0.8, 0.1, 0.05,
#                    0.3, 0.7, 0.02,
#                    0.01, 0.02, 0.9), 3, 3, byrow=TRUE)
# Y <- matrix(0, n, 3)
# Y[1,] <- rnorm(3, 0, 0.01)
# for (t in 2:n) Y[t,] <- A_true %*% Y[t-1,] + rnorm(3, 0, c(0.02, 0.018, 0.001))
# colnames(Y) <- c("BTC", "ETH", "Funding")
# result <- run_econometrics(Y, target_var=1, n_irf_periods=15, n_boot_irf=100)

# =============================================================================
# EXTENDED ECONOMETRICS: Threshold VAR, Smooth Transition, Dynamic Conditional
# Correlation, Panel Unit Root, and Spatial Econometrics Approximations
# =============================================================================

# -----------------------------------------------------------------------------
# Threshold VAR (TVAR): VAR parameters switch based on a threshold variable
# In crypto: regime switches based on BTC volatility or funding rate level
# Estimated via concentrated least squares: grid search over threshold
# -----------------------------------------------------------------------------
threshold_var <- function(Y, threshold_var_idx = 1, p = 1, delay = 1,
                           trim_pct = 0.15, n_grid = 50) {
  n <- nrow(Y); k <- ncol(Y)
  asset_names <- colnames(Y)

  # Threshold variable: lagged value of the chosen series
  q_t <- Y[(delay):(n - 1), threshold_var_idx]  # delay periods lagged

  # Trim: exclude extreme quantiles for threshold search
  q_lo <- quantile(q_t, trim_pct); q_hi <- quantile(q_t, 1 - trim_pct)
  gamma_grid <- seq(q_lo, q_hi, length.out = n_grid)

  best_rss <- Inf; best_gamma <- NA
  best_model <- NULL

  for (gamma in gamma_grid) {
    # Indicator: regime 1 when threshold_var <= gamma
    I_t <- as.integer(q_t <= gamma)

    # Build design matrix for each regime
    # Lag Y by p periods
    Y_lag <- Y[(p):(n-1-delay+1), , drop=FALSE]
    Y_dep  <- Y[(p+delay):(n), , drop=FALSE]
    if (nrow(Y_dep) != length(I_t[p:length(I_t)])) next

    T_eff <- nrow(Y_dep)
    I_use <- I_t[p:min(length(I_t), p + T_eff - 1)]

    # Regime 1 (below threshold)
    idx1 <- which(I_use == 1)
    idx2 <- which(I_use == 0)

    if (length(idx1) < k * p + 1 || length(idx2) < k * p + 1) next

    # OLS for each regime
    rss <- 0
    models <- list()
    for (regime in list(idx1, idx2)) {
      X_r <- cbind(1, Y_lag[regime, , drop=FALSE])
      Y_r <- Y_dep[regime, , drop=FALSE]
      fit <- lm.fit(X_r, Y_r)
      rss <- rss + sum(residuals(fit)^2)
    }

    if (rss < best_rss) {
      best_rss <- rss; best_gamma <- gamma
    }
  }

  # Final model at best threshold
  I_use_final <- as.integer(q_t <= best_gamma)
  T_eff <- n - p - delay + 1
  Y_dep <- Y[(p+delay):n, , drop=FALSE]
  Y_lag <- Y[p:(n-delay), , drop=FALSE]
  I_f   <- I_use_final[p:min(length(I_use_final), p+nrow(Y_dep)-1)]
  n_use <- min(nrow(Y_dep), length(I_f))
  Y_dep <- Y_dep[1:n_use, ]; Y_lag <- Y_lag[1:n_use, ]; I_f <- I_f[1:n_use]

  coef_low <- coef_high <- NULL
  for (regime_val in c(1, 0)) {
    idx <- which(I_f == regime_val)
    if (length(idx) < 2) next
    X_r <- cbind(1, Y_lag[idx, , drop=FALSE])
    Y_r <- Y_dep[idx, , drop=FALSE]
    b <- lm.fit(X_r, Y_r)$coefficients
    if (regime_val == 1) coef_low <- b else coef_high <- b
  }

  list(
    threshold = best_gamma,
    rss = best_rss,
    pct_in_low_regime = mean(I_use_final),
    coef_low_regime = coef_low,
    coef_high_regime = coef_high,
    threshold_var = colnames(Y)[threshold_var_idx]
  )
}

# -----------------------------------------------------------------------------
# Smooth Transition Regression (STR/LSTAR): parameters transition smoothly
# between two regimes via logistic function of a transition variable
# Less abrupt than threshold models; more realistic for financial data
# -----------------------------------------------------------------------------
smooth_transition_regression <- function(y, x_mat, transition_var,
                                          gamma_init = 5, c_init = NULL,
                                          max_iter = 100) {
  n <- length(y)
  k <- ncol(x_mat)

  if (is.null(c_init)) c_init <- median(transition_var)

  # Logistic transition function
  G <- function(q, gamma, c) 1 / (1 + exp(-gamma * (q - c)))

  # STR log-likelihood (minimize SSE)
  str_sse <- function(params) {
    gamma_p <- exp(params[1])  # ensure positive
    c_p     <- params[2]
    beta1   <- params[3:(2+k)]
    beta2   <- params[(3+k):(2+2*k)]

    g <- G(transition_var, gamma_p, c_p)
    y_hat <- x_mat %*% beta1 + (x_mat * g) %*% beta2
    sum((y - y_hat)^2)
  }

  # Initial OLS for beta starting values
  ols <- lm(y ~ x_mat - 1)
  beta0 <- rep(coef(ols), 2) * c(1, 0.1)  # small regime 2 effect initially

  init_params <- c(log(gamma_init), c_init, beta0)
  opt <- tryCatch(
    optim(init_params, str_sse, method = "BFGS",
          control = list(maxit = max_iter)),
    error = function(e) list(par = init_params, value = str_sse(init_params), convergence = 1)
  )

  gamma_hat <- exp(opt$par[1]); c_hat <- opt$par[2]
  beta1_hat <- opt$par[3:(2+k)]; beta2_hat <- opt$par[(3+k):(2+2*k)]

  g_hat <- G(transition_var, gamma_hat, c_hat)
  y_hat <- x_mat %*% beta1_hat + (x_mat * g_hat) %*% beta2_hat
  resid <- y - y_hat
  r2    <- 1 - sum(resid^2) / sum((y - mean(y))^2)

  list(
    gamma = gamma_hat, c = c_hat,
    beta_low_regime = beta1_hat,
    beta_high_regime = beta1_hat + beta2_hat,
    transition_values = g_hat,
    r_squared = r2,
    sse = opt$value,
    convergence = opt$convergence,
    pct_in_high_regime = mean(g_hat > 0.5)
  )
}

# -----------------------------------------------------------------------------
# Dynamic Conditional Correlation (DCC) Model: Engle (2002)
# Extends GARCH to multivariate setting: correlations vary over time
# Two-step estimation: (1) univariate GARCH for each series,
# (2) DCC parameters for dynamic correlation evolution
# -----------------------------------------------------------------------------
estimate_dcc <- function(returns_mat, a_init = 0.05, b_init = 0.90,
                          max_iter = 100) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)

  # Step 1: Fit univariate GARCH(1,1) to each series
  garch_fit <- function(r, omega = NULL, alpha = 0.05, beta = 0.90) {
    n_r <- length(r)
    sigma2 <- rep(var(r), n_r)

    garch_ll <- function(params) {
      om <- exp(params[1]); al <- plogis(params[2]); be <- plogis(params[3])
      if (al + be >= 1) return(1e10)
      h <- rep(var(r), n_r)
      for (t in 2:n_r) h[t] <- om + al*r[t-1]^2 + be*h[t-1]
      -sum(dnorm(r, 0, sqrt(h), log = TRUE))
    }
    opt <- tryCatch(
      optim(c(log(var(r)*0.05), qlogis(0.05), qlogis(0.90)), garch_ll,
            method = "Nelder-Mead", control = list(maxit = max_iter)),
      error = function(e) list(par = c(log(var(r)*0.05), qlogis(0.05), qlogis(0.90)))
    )
    om <- exp(opt$par[1]); al <- plogis(opt$par[2]); be <- plogis(opt$par[3])
    h <- rep(var(r), n_r)
    for (t in 2:n_r) h[t] <- om + al*r[t-1]^2 + be*h[t-1]
    list(h = h, omega = om, alpha = al, beta = be)
  }

  # Standardized residuals
  h_list <- lapply(1:p, function(j) garch_fit(returns_mat[, j]))
  std_resid <- matrix(0, n, p)
  for (j in 1:p) {
    std_resid[, j] <- returns_mat[, j] / sqrt(h_list[[j]]$h)
  }

  # Step 2: DCC Q evolution
  Q_bar <- cov(std_resid)  # unconditional correlation
  Q_t   <- array(0, dim = c(p, p, n))
  R_t   <- array(0, dim = c(p, p, n))
  Q_t[,,1] <- Q_bar

  dcc_ll <- function(params) {
    a <- plogis(params[1]) * 0.5; b <- plogis(params[2]) * 0.95
    if (a + b >= 1) return(1e10)
    Q_cur <- Q_bar; ll <- 0
    for (t in 2:n) {
      e_prev <- matrix(std_resid[t-1, ], p, 1)
      Q_cur  <- (1 - a - b) * Q_bar + a * (e_prev %*% t(e_prev)) + b * Q_cur
      D_inv  <- 1 / sqrt(diag(Q_cur))
      R_cur  <- Q_cur * outer(D_inv, D_inv)
      R_inv  <- tryCatch(solve(R_cur), error = function(e) diag(p))
      e_t    <- matrix(std_resid[t, ], p, 1)
      ll <- ll - 0.5 * (log(det(R_cur)) + t(e_t) %*% R_inv %*% e_t)
    }
    -ll
  }

  opt <- tryCatch(
    optim(c(qlogis(a_init/0.5), qlogis(b_init/0.95)), dcc_ll,
          method = "Nelder-Mead", control = list(maxit = max_iter)),
    error = function(e) list(par = c(qlogis(0.05/0.5), qlogis(0.90/0.95)))
  )

  a_hat <- plogis(opt$par[1]) * 0.5; b_hat <- plogis(opt$par[2]) * 0.95

  # Compute R_t at estimated parameters
  Q_cur <- Q_bar
  corr_series <- matrix(0, n, p*(p-1)/2)
  pair_idx <- which(lower.tri(matrix(0,p,p)), arr.ind=TRUE)

  for (t in 2:n) {
    e_prev <- matrix(std_resid[t-1, ], p, 1)
    Q_cur  <- (1-a_hat-b_hat)*Q_bar + a_hat*(e_prev %*% t(e_prev)) + b_hat*Q_cur
    D_inv  <- 1/sqrt(diag(Q_cur))
    R_cur  <- Q_cur * outer(D_inv, D_inv)
    corr_series[t, ] <- R_cur[lower.tri(R_cur)]
  }

  list(
    a = a_hat, b = b_hat,
    dynamic_correlations = corr_series,
    unconditional_corr = Q_bar * outer(1/sqrt(diag(Q_bar)), 1/sqrt(diag(Q_bar))),
    garch_params = lapply(h_list, function(h) c(omega=h$omega, alpha=h$alpha, beta=h$beta)),
    corr_persistence = a_hat + b_hat,
    std_residuals = std_resid
  )
}

# Extended econometrics example:
# tvar_res <- threshold_var(Y, threshold_var_idx=1, p=1, delay=1)
# str_res  <- smooth_transition_regression(y, X, transition_var=q_vec)
# dcc_res  <- estimate_dcc(returns_mat, a_init=0.05, b_init=0.90)
