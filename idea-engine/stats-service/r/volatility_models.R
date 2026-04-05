# =============================================================================
# volatility_models.R
# Comprehensive volatility modeling suite for crypto trading
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Crypto volatility is not constant -- it clusters, has
# leverage effects (bad news increases vol more than good news), and follows
# regime shifts. Accurate vol forecasting is essential for position sizing,
# options pricing, and risk management.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GARCH(1,1) FROM SCRATCH
# -----------------------------------------------------------------------------

#' GARCH(1,1) conditional variance recursion
#' sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
#' Stationarity: alpha + beta < 1
#' Unconditional variance: omega / (1 - alpha - beta)
#' @param returns numeric vector of returns
#' @param params named numeric: omega, alpha, beta
garch11_variance <- function(returns, params) {
  omega <- params["omega"]; alpha <- params["alpha"]; beta  <- params["beta"]
  n <- length(returns)
  sigma2 <- numeric(n)
  sigma2[1] <- var(returns)  # initialize at unconditional variance
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * returns[t-1]^2 + beta * sigma2[t-1]
  }
  sigma2
}

#' GARCH(1,1) log-likelihood (Gaussian innovations)
garch11_loglik <- function(params, returns) {
  if (any(params <= 0)) return(1e10)
  if (params["alpha"] + params["beta"] >= 1) return(1e10)
  sigma2 <- garch11_variance(returns, params)
  if (any(sigma2 <= 0)) return(1e10)
  n <- length(returns)
  ll <- -0.5 * sum(log(2 * pi) + log(sigma2) + returns^2 / sigma2)
  -ll  # return negative for minimization
}

#' Fit GARCH(1,1) via numerical optimization (Nelder-Mead)
#' @param returns numeric vector of log-returns
#' @return list with params, sigma2, residuals, loglik, AIC, BIC
fit_garch11 <- function(returns) {
  n <- length(returns)
  # Initial parameters: small omega, typical alpha=0.1, beta=0.8
  sigma2_unc <- var(returns)
  init_params <- c(omega = sigma2_unc * 0.05, alpha = 0.1, beta = 0.85)

  fit <- optim(init_params, garch11_loglik, returns = returns,
               method = "Nelder-Mead",
               control = list(maxit = 5000, reltol = 1e-8))

  params <- fit$par
  sigma2 <- garch11_variance(returns, params)
  std_resid <- returns / sqrt(sigma2)

  loglik <- -fit$value
  k <- 3
  aic <- -2 * loglik + 2 * k
  bic <- -2 * loglik + log(n) * k

  cat("=== GARCH(1,1) Estimation ===\n")
  cat(sprintf("omega = %.6f\n", params["omega"]))
  cat(sprintf("alpha = %.4f\n", params["alpha"]))
  cat(sprintf("beta  = %.4f\n", params["beta"]))
  cat(sprintf("alpha+beta = %.4f (persistence)\n", params["alpha"]+params["beta"]))
  cat(sprintf("Unconditional vol = %.4f (annualized: %.2f%%)\n",
              sqrt(params["omega"]/(1-params["alpha"]-params["beta"])),
              sqrt(params["omega"]/(1-params["alpha"]-params["beta"])) * sqrt(365) * 100))
  cat(sprintf("Log-likelihood: %.2f  AIC: %.2f  BIC: %.2f\n", loglik, aic, bic))

  list(params = params, sigma2 = sigma2, std_resid = std_resid,
       loglik = loglik, AIC = aic, BIC = bic, convergence = fit$convergence)
}

# -----------------------------------------------------------------------------
# 2. EGARCH (Exponential GARCH) -- asymmetric volatility
# -----------------------------------------------------------------------------

#' EGARCH(1,1): log(sigma^2_t) = omega + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1} + beta*log(sigma^2_{t-1})
#' gamma < 0 captures leverage effect: negative shocks increase vol more
#' Log-specification ensures sigma^2 > 0 without constraints on parameters
egarch11_variance <- function(returns, params) {
  omega <- params["omega"]; alpha <- params["alpha"]
  beta  <- params["beta"];  gamma <- params["gamma"]
  n <- length(returns)
  log_sigma2 <- numeric(n)
  log_sigma2[1] <- log(var(returns))
  E_abs_z <- sqrt(2 / pi)  # E[|z|] for standard normal
  z <- numeric(n)

  for (t in 2:n) {
    z[t-1] <- returns[t-1] / exp(0.5 * log_sigma2[t-1])
    log_sigma2[t] <- omega +
                     alpha * (abs(z[t-1]) - E_abs_z) +
                     gamma * z[t-1] +
                     beta  * log_sigma2[t-1]
  }
  exp(log_sigma2)
}

egarch11_loglik <- function(params, returns) {
  sigma2 <- tryCatch(egarch11_variance(returns, params),
                     error = function(e) NULL)
  if (is.null(sigma2) || any(!is.finite(sigma2)) || any(sigma2 <= 0)) return(1e10)
  n <- length(returns)
  -(-0.5 * sum(log(2*pi) + log(sigma2) + returns^2/sigma2))
}

fit_egarch11 <- function(returns) {
  n <- length(returns)
  init_params <- c(omega = 0, alpha = 0.1, beta = 0.9, gamma = -0.05)
  fit <- optim(init_params, egarch11_loglik, returns = returns,
               method = "Nelder-Mead",
               control = list(maxit = 5000, reltol = 1e-8))
  params  <- fit$par
  sigma2  <- egarch11_variance(returns, params)
  loglik  <- -fit$value
  std_res <- returns / sqrt(sigma2)
  k <- 4
  cat("=== EGARCH(1,1) Estimation ===\n")
  cat(sprintf("omega = %.6f, alpha = %.4f, beta = %.4f, gamma = %.4f\n",
              params["omega"], params["alpha"], params["beta"], params["gamma"]))
  cat(sprintf("Leverage effect (gamma<0): %s\n", params["gamma"] < 0))
  cat(sprintf("Log-likelihood: %.2f  AIC: %.2f\n", loglik, -2*loglik+2*k))
  list(params=params, sigma2=sigma2, std_resid=std_res,
       loglik=loglik, AIC=-2*loglik+2*k, BIC=-2*loglik+log(n)*k)
}

# -----------------------------------------------------------------------------
# 3. GJR-GARCH (Glosten-Jagannathan-Runkle)
# -----------------------------------------------------------------------------

#' GJR-GARCH(1,1): sigma^2_t = omega + (alpha + gamma*I(r_{t-1}<0)) * r^2_{t-1} + beta*sigma^2_{t-1}
#' gamma > 0 means negative returns increase vol more (leverage effect)
gjr_garch_variance <- function(returns, params) {
  omega <- params["omega"]; alpha <- params["alpha"]
  beta  <- params["beta"];  gamma <- params["gamma"]
  n <- length(returns)
  sigma2 <- numeric(n)
  sigma2[1] <- var(returns)
  for (t in 2:n) {
    I_neg <- as.numeric(returns[t-1] < 0)
    sigma2[t] <- omega + (alpha + gamma * I_neg) * returns[t-1]^2 + beta * sigma2[t-1]
  }
  sigma2
}

gjr_garch_loglik <- function(params, returns) {
  if (params["omega"] <= 0 || params["beta"] < 0) return(1e10)
  if (params["alpha"] < 0 || params["alpha"] + params["gamma"]/2 + params["beta"] >= 1) return(1e10)
  sigma2 <- tryCatch(gjr_garch_variance(returns, params), error=function(e) NULL)
  if (is.null(sigma2) || any(sigma2 <= 0)) return(1e10)
  -(-0.5 * sum(log(2*pi) + log(sigma2) + returns^2/sigma2))
}

fit_gjr_garch <- function(returns) {
  n <- length(returns)
  init_params <- c(omega = var(returns)*0.05, alpha=0.05, beta=0.85, gamma=0.1)
  fit <- optim(init_params, gjr_garch_loglik, returns=returns,
               method="Nelder-Mead", control=list(maxit=5000, reltol=1e-8))
  params <- fit$par
  sigma2 <- gjr_garch_variance(returns, params)
  loglik <- -fit$value
  std_res <- returns / sqrt(sigma2)
  k <- 4
  cat("=== GJR-GARCH(1,1) Estimation ===\n")
  cat(sprintf("omega=%.6f, alpha=%.4f, beta=%.4f, gamma(asym)=%.4f\n",
              params["omega"], params["alpha"], params["beta"], params["gamma"]))
  cat(sprintf("News impact asymmetry (gamma>0 = leverage): %s\n", params["gamma"]>0))
  cat(sprintf("Log-likelihood: %.2f  AIC: %.2f\n", loglik, -2*loglik+2*k))
  list(params=params, sigma2=sigma2, std_resid=std_res,
       loglik=loglik, AIC=-2*loglik+2*k, BIC=-2*loglik+log(n)*k)
}

# -----------------------------------------------------------------------------
# 4. HAR-RV (Heterogeneous AutoRegressive Realized Volatility)
# -----------------------------------------------------------------------------

#' Compute daily, weekly, monthly realized variance averages
#' HAR-RV models the observation that volatility is determined by traders
#' with different horizons: HFT (daily), institutions (weekly), long-term (monthly)
#' @param rv vector of daily realized variances
#' @return list of daily, weekly (5-day avg), monthly (22-day avg) RV
compute_har_components <- function(rv) {
  n <- length(rv)
  rv_d <- rv  # daily
  rv_w <- stats::filter(rv, rep(1/5,  5), sides=1)  # 5-day average
  rv_m <- stats::filter(rv, rep(1/22, 22), sides=1) # 22-day average
  list(rv_d=rv_d, rv_w=as.numeric(rv_w), rv_m=as.numeric(rv_m))
}

#' Fit HAR-RV model: RV_t = c + beta_d*RV_{t-1} + beta_w*RV_w_{t-1} + beta_m*RV_m_{t-1} + epsilon
#' @param rv daily realized variance series
fit_har_rv <- function(rv) {
  n <- length(rv)
  comp <- compute_har_components(rv)

  # Build design matrix (starting from period 23 to have all components)
  start <- 23
  y  <- comp$rv_d[(start+1):n]
  Xd <- comp$rv_d[start:(n-1)]
  Xw <- comp$rv_w[start:(n-1)]
  Xm <- comp$rv_m[start:(n-1)]

  # Remove NAs
  valid <- !is.na(y) & !is.na(Xd) & !is.na(Xw) & !is.na(Xm)
  y  <- y[valid]; Xd <- Xd[valid]; Xw <- Xw[valid]; Xm <- Xm[valid]
  n_eff <- length(y)

  X <- cbind(1, Xd, Xw, Xm)
  # OLS
  beta <- solve(t(X) %*% X) %*% t(X) %*% y
  fitted <- X %*% beta
  resid  <- y - fitted
  ss_res <- sum(resid^2)
  ss_tot <- sum((y - mean(y))^2)
  r2 <- 1 - ss_res / ss_tot

  # Standard errors
  sigma2_e <- ss_res / (n_eff - 4)
  var_beta  <- sigma2_e * solve(t(X) %*% X)
  se_beta   <- sqrt(diag(var_beta))
  t_stats   <- beta / se_beta

  cat("=== HAR-RV Model ===\n")
  cat(sprintf("c      = %.6f (t=%.2f)\n", beta[1], t_stats[1]))
  cat(sprintf("beta_d = %.4f (t=%.2f)  [daily horizon]\n",  beta[2], t_stats[2]))
  cat(sprintf("beta_w = %.4f (t=%.2f)  [weekly horizon]\n", beta[3], t_stats[3]))
  cat(sprintf("beta_m = %.4f (t=%.2f)  [monthly horizon]\n",beta[4], t_stats[4]))
  cat(sprintf("R^2 = %.4f\n", r2))

  # Forecast next period RV
  last_d <- tail(comp$rv_d, 1)
  last_w <- tail(na.omit(comp$rv_w), 1)
  last_m <- tail(na.omit(comp$rv_m), 1)
  forecast_rv <- beta[1] + beta[2]*last_d + beta[3]*last_w + beta[4]*last_m
  cat(sprintf("1-day RV forecast: %.6f  (vol=%.4f)\n", forecast_rv, sqrt(forecast_rv)))

  list(beta=beta, se=se_beta, t_stats=t_stats, r2=r2,
       fitted=fitted, residuals=resid, forecast_rv=forecast_rv)
}

# -----------------------------------------------------------------------------
# 5. MIDAS REGRESSION FOR MIXED-FREQUENCY VOLATILITY
# -----------------------------------------------------------------------------

#' MIDAS (Mixed Data Sampling) regression for volatility forecasting
#' Allows high-frequency data (intraday) to predict lower-frequency outcomes
#' Uses Beta lag polynomial for parsimonious weighting
#' @param y_low low-frequency dependent variable (e.g., monthly vol)
#' @param x_high high-frequency predictor (e.g., daily rv, stacked as matrix)
#'        each row of x_high is one low-freq period, cols are HF observations
#' @param theta_init initial Beta polynomial parameters [theta1, theta2]
midas_beta_weights <- function(k, theta1, theta2) {
  # Beta weight function on [0,1]
  j <- seq_len(k) / k
  w <- j^(theta1-1) * (1-j)^(theta2-1)
  w / sum(w)
}

fit_midas <- function(y_low, x_high, theta_init = c(1, 5)) {
  n_low <- length(y_low)
  k_high <- ncol(x_high)  # number of HF obs per LF period

  objective <- function(theta) {
    w <- midas_beta_weights(k_high, max(theta[1], 0.1), max(theta[2], 0.1))
    X_midas <- x_high %*% w
    # OLS for beta_0, beta_1
    X_ols <- cbind(1, X_midas)
    beta <- solve(t(X_ols) %*% X_ols) %*% t(X_ols) %*% y_low
    resid <- y_low - X_ols %*% beta
    sum(resid^2)  # minimize SSR
  }

  fit <- optim(theta_init, objective, method="Nelder-Mead",
               control=list(maxit=2000))
  theta_opt <- pmax(fit$par, 0.1)
  w_opt <- midas_beta_weights(k_high, theta_opt[1], theta_opt[2])
  X_midas <- x_high %*% w_opt
  X_ols   <- cbind(1, X_midas)
  beta    <- solve(t(X_ols) %*% X_ols) %*% t(X_ols) %*% y_low
  fitted  <- X_ols %*% beta
  resid   <- y_low - fitted
  r2 <- 1 - sum(resid^2) / sum((y_low - mean(y_low))^2)

  cat("=== MIDAS Regression ===\n")
  cat(sprintf("Beta polynomial: theta1=%.2f, theta2=%.2f\n", theta_opt[1], theta_opt[2]))
  cat(sprintf("beta_0=%.4f, beta_1=%.4f, R^2=%.4f\n", beta[1], beta[2], r2))

  list(theta=theta_opt, weights=w_opt, beta=beta, fitted=fitted,
       residuals=resid, r2=r2)
}

# -----------------------------------------------------------------------------
# 6. STOCHASTIC VOLATILITY MODEL (Method of Moments)
# -----------------------------------------------------------------------------

#' Simple stochastic volatility model via moment matching
#' log(sigma^2_t) = mu + phi*(log(sigma^2_{t-1}) - mu) + eta_t
#' Returns follow: r_t = sigma_t * epsilon_t
#' Moments: E[r^2] = exp(mu + tau^2/2), E[r^4] = 3*exp(2*mu + 2*tau^2)
#' where tau^2 = Var(log(sigma^2))
fit_sv_moments <- function(returns) {
  n <- length(returns)
  r2 <- returns^2
  log_r2 <- log(r2 + 1e-10)  # proxy for log-vol

  # AR(1) on log(r^2) to get phi and mu
  log_r2_lag <- log_r2[-n]
  log_r2_cur <- log_r2[-1]
  X <- cbind(1, log_r2_lag)
  b <- solve(t(X) %*% X) %*% t(X) %*% log_r2_cur
  phi_hat <- b[2]
  mu_hat  <- b[1] / (1 - b[2])  # unconditional mean

  # Estimate tau^2 from the variance of log(r^2)
  # Var(log r^2) = tau^2/(1-phi^2) + pi^2/2 (where pi^2/2 is from log-chi2 noise)
  var_log_r2 <- var(log_r2)
  tau2_hat <- max((var_log_r2 - pi^2/2) * (1 - phi_hat^2), 0.001)

  cat("=== Stochastic Volatility (Moment Matching) ===\n")
  cat(sprintf("mu  = %.4f  (unconditional log-variance)\n", mu_hat))
  cat(sprintf("phi = %.4f  (vol persistence)\n", phi_hat))
  cat(sprintf("tau = %.4f  (vol-of-vol)\n", sqrt(tau2_hat)))
  cat(sprintf("Unconditional vol = %.4f\n", exp(mu_hat/2 + tau2_hat/4)))

  # Filter latent volatility (Kalman-like smoother on log-vol)
  h <- numeric(n)
  h[1] <- mu_hat
  for (t in 2:n) {
    h[t] <- mu_hat + phi_hat * (h[t-1] - mu_hat)
    # Simple update: pull toward observation
    innov <- log_r2[t] - h[t] - 1.2704  # E[log chi2(1)] = -1.2704
    h[t] <- h[t] + 0.5 * innov  # Kalman gain ≈ 0.5 (simplified)
  }
  sigma2_filtered <- exp(h)

  list(mu=mu_hat, phi=phi_hat, tau2=tau2_hat,
       sigma2=sigma2_filtered, log_vol=h)
}

# -----------------------------------------------------------------------------
# 7. VOLATILITY OF VOLATILITY (GARCH on squared returns)
# -----------------------------------------------------------------------------

#' Estimate vol-of-vol: GARCH(1,1) applied to squared standardized returns
#' Intuition: in crypto, the intensity of vol clustering itself is time-varying
#' @param garch_fit output of fit_garch11
vol_of_vol <- function(garch_fit) {
  std_resid <- garch_fit$std_resid
  # Squared standardized residuals are proxies for normalized "vol shocks"
  z2 <- std_resid^2
  # Fit GARCH(1,1) to z^2 - 1 (centered around 0)
  cat("=== Vol-of-Vol: GARCH on Squared Std Residuals ===\n")
  vov_fit <- fit_garch11(z2 - 1)
  cat(sprintf("Vol-of-vol persistence: %.4f\n",
              vov_fit$params["alpha"] + vov_fit$params["beta"]))
  vov_fit
}

# -----------------------------------------------------------------------------
# 8. VOLATILITY TERM STRUCTURE
# -----------------------------------------------------------------------------

#' Compute volatility term structure across multiple horizons
#' Uses GARCH(1,1) multi-step variance forecasts
#' sigma^2_{t+h|t} = omega*(sum_{k=0}^{h-1}(alpha+beta)^k) + (alpha+beta)^h * sigma^2_t
#' @param garch_fit fitted GARCH model
#' @param horizons forecast horizons (in days)
garch_term_structure <- function(garch_fit, horizons = c(1, 5, 10, 22, 63, 126, 252)) {
  params  <- garch_fit$params
  omega   <- params["omega"]; alpha <- params["alpha"]; beta <- params["beta"]
  pers    <- alpha + beta
  sigma2_unc <- omega / (1 - pers)
  sigma2_cur <- tail(garch_fit$sigma2, 1)

  term_struct <- sapply(horizons, function(h) {
    # h-step ahead forecast
    if (pers < 1) {
      s2 <- sigma2_unc + pers^h * (sigma2_cur - sigma2_unc)
    } else {
      s2 <- sigma2_cur + h * omega
    }
    sqrt(s2 * h)  # annualize by sqrt(h) and return as vol level
  })

  df <- data.frame(
    horizon = horizons,
    vol_forecast = term_struct / sqrt(horizons),  # per-period vol
    vol_annualized = term_struct / sqrt(horizons) * sqrt(365)
  )
  cat("=== Volatility Term Structure ===\n")
  print(df)
  invisible(df)
}

# -----------------------------------------------------------------------------
# 9. VOLATILITY REGIME SWITCHING (2-state Markov, manual EM)
# -----------------------------------------------------------------------------

#' 2-state Markov regime-switching volatility model
#' State 1 = low vol, State 2 = high vol
#' EM algorithm: E-step computes state probabilities, M-step updates params
#' @param returns numeric vector
#' @param max_iter maximum EM iterations
#' @param tol convergence tolerance
fit_markov_switching_vol <- function(returns, max_iter = 200, tol = 1e-6) {
  n <- length(returns)

  # Initialize: classify into high/low vol regimes by return magnitude
  high_vol_idx <- abs(returns) > quantile(abs(returns), 0.6)
  mu1 <- 0; mu2 <- 0
  sigma1 <- sd(returns[!high_vol_idx])
  sigma2 <- sd(returns[high_vol_idx])
  if (is.na(sigma1) || sigma1 == 0) sigma1 <- sd(returns) * 0.7
  if (is.na(sigma2) || sigma2 == 0) sigma2 <- sd(returns) * 1.3

  p11 <- 0.95; p22 <- 0.95  # transition probabilities (stay in same state)
  p12 <- 1 - p11; p21 <- 1 - p22

  ll_prev <- -Inf

  for (iter in seq_len(max_iter)) {
    # ===== E-STEP: Forward-Backward algorithm =====
    # Emission densities
    f1 <- dnorm(returns, mu1, sigma1)
    f2 <- dnorm(returns, mu2, sigma2)
    f1 <- pmax(f1, 1e-300); f2 <- pmax(f2, 1e-300)

    # Stationary initial distribution
    pi1_init <- p21 / (p12 + p21)
    pi2_init <- 1 - pi1_init

    # Forward probabilities (scaled)
    alpha_f  <- matrix(0, n, 2)
    scale_f  <- numeric(n)
    alpha_f[1, ] <- c(pi1_init * f1[1], pi2_init * f2[1])
    scale_f[1]   <- sum(alpha_f[1, ])
    alpha_f[1, ] <- alpha_f[1, ] / scale_f[1]

    for (t in 2:n) {
      alpha_f[t, 1] <- (alpha_f[t-1, 1] * p11 + alpha_f[t-1, 2] * p21) * f1[t]
      alpha_f[t, 2] <- (alpha_f[t-1, 1] * p12 + alpha_f[t-1, 2] * p22) * f2[t]
      scale_f[t] <- sum(alpha_f[t, ])
      if (scale_f[t] > 0) alpha_f[t, ] <- alpha_f[t, ] / scale_f[t]
    }

    # Backward probabilities
    beta_b <- matrix(1, n, 2)
    for (t in (n-1):1) {
      beta_b[t, 1] <- p11 * f1[t+1] * beta_b[t+1, 1] +
                      p12 * f2[t+1] * beta_b[t+1, 2]
      beta_b[t, 2] <- p21 * f1[t+1] * beta_b[t+1, 1] +
                      p22 * f2[t+1] * beta_b[t+1, 2]
      sc <- max(beta_b[t, ]); if (sc > 0) beta_b[t, ] <- beta_b[t, ] / sc
    }

    # Smoothed state probabilities
    gamma_probs <- alpha_f * beta_b
    gamma_probs <- gamma_probs / rowSums(gamma_probs)

    # Transition counts (xi)
    xi <- array(0, dim = c(n-1, 2, 2))
    for (t in seq_len(n-1)) {
      xi[t, 1, 1] <- alpha_f[t,1] * p11 * f1[t+1] * beta_b[t+1, 1]
      xi[t, 1, 2] <- alpha_f[t,1] * p12 * f2[t+1] * beta_b[t+1, 2]
      xi[t, 2, 1] <- alpha_f[t,2] * p21 * f1[t+1] * beta_b[t+1, 1]
      xi[t, 2, 2] <- alpha_f[t,2] * p22 * f2[t+1] * beta_b[t+1, 2]
      xi[t, , ]   <- xi[t, , ] / sum(xi[t, , ])
    }

    # ===== M-STEP: Update parameters =====
    # Transition probabilities
    p11 <- sum(xi[, 1, 1]) / sum(xi[, 1, ])
    p22 <- sum(xi[, 2, 2]) / sum(xi[, 2, ])
    p12 <- 1 - p11; p21 <- 1 - p22

    # Means and variances per state
    g1 <- gamma_probs[, 1]; g2 <- gamma_probs[, 2]
    mu1 <- sum(g1 * returns) / sum(g1)
    mu2 <- sum(g2 * returns) / sum(g2)
    sigma1 <- sqrt(sum(g1 * (returns - mu1)^2) / sum(g1))
    sigma2 <- sqrt(sum(g2 * (returns - mu2)^2) / sum(g2))
    # Ensure state 1 is low-vol state
    if (sigma1 > sigma2) {
      mu1 <- mu1 + mu2; mu2 <- mu1 - mu2; mu1 <- mu1 - mu2
      sigma1 <- sigma1 + sigma2; sigma2 <- sigma1 - sigma2; sigma1 <- sigma1 - sigma2
      gamma_probs <- gamma_probs[, c(2,1)]
      p11_old <- p11; p11 <- p22; p22 <- p11_old
    }

    # Log-likelihood
    ll <- sum(log(scale_f))
    if (abs(ll - ll_prev) < tol) break
    ll_prev <- ll
  }

  cat("=== 2-State Markov Switching Volatility ===\n")
  cat(sprintf("State 1 (low vol):  mu=%.4f, sigma=%.4f, p11=%.3f\n", mu1, sigma1, p11))
  cat(sprintf("State 2 (high vol): mu=%.4f, sigma=%.4f, p22=%.3f\n", mu2, sigma2, p22))
  cat(sprintf("E[duration state 1]: %.1f periods\n", 1/p12))
  cat(sprintf("E[duration state 2]: %.1f periods\n", 1/p21))
  cat(sprintf("Log-likelihood: %.2f (converged in %d iters)\n", ll, iter))

  regime <- apply(gamma_probs, 1, which.max)
  list(mu=c(mu1,mu2), sigma=c(sigma1,sigma2),
       P=matrix(c(p11,p12,p21,p22),2,2,byrow=TRUE),
       smoothed_probs=gamma_probs, regime=regime,
       loglik=ll, iter=iter)
}

# -----------------------------------------------------------------------------
# 10. CRYPTO-SPECIFIC: VOL CLUSTERING ANALYSIS
# -----------------------------------------------------------------------------

#' Test for volatility clustering: autocorrelation of squared returns
#' If alpha + beta ~ 1 in GARCH, ACF of r^2 decays slowly = clustering
#' @param returns return series
#' @param max_lag maximum lag for ACF
vol_clustering_test <- function(returns, max_lag = 20) {
  r2 <- returns^2
  n  <- length(returns)

  # Ljung-Box test on squared returns
  acf_vals <- numeric(max_lag)
  for (k in seq_len(max_lag)) {
    acf_vals[k] <- cor(r2[(k+1):n], r2[1:(n-k)])
  }

  # Ljung-Box statistic
  lb_stat <- n * (n + 2) * sum(acf_vals^2 / (n - seq_len(max_lag)))
  lb_pval <- 1 - pchisq(lb_stat, df = max_lag)

  # ARCH LM test (regression of r^2 on lags)
  lags <- 5
  Y_lm <- r2[(lags+1):n]
  X_lm <- matrix(0, length(Y_lm), lags+1)
  X_lm[, 1] <- 1
  for (k in seq_len(lags)) X_lm[, k+1] <- r2[(lags+1-k):(n-k)]
  b_lm <- solve(t(X_lm) %*% X_lm) %*% t(X_lm) %*% Y_lm
  fit_lm <- X_lm %*% b_lm
  res_lm <- Y_lm - fit_lm
  ss_res <- sum(res_lm^2); ss_tot <- sum((Y_lm - mean(Y_lm))^2)
  arch_r2 <- 1 - ss_res/ss_tot
  arch_stat <- length(Y_lm) * arch_r2
  arch_pval <- 1 - pchisq(arch_stat, df=lags)

  cat("=== Volatility Clustering Tests ===\n")
  cat(sprintf("Ljung-Box Q(%d) on r^2: stat=%.2f, p=%.4f\n", max_lag, lb_stat, lb_pval))
  cat(sprintf("ARCH LM test (lags=5): stat=%.2f, p=%.4f\n", arch_stat, arch_pval))
  cat(sprintf("Conclusion: %s\n",
              ifelse(lb_pval < 0.05, "Significant volatility clustering detected",
                     "No significant volatility clustering")))

  list(acf_r2=acf_vals, lb_stat=lb_stat, lb_pval=lb_pval,
       arch_stat=arch_stat, arch_pval=arch_pval)
}

#' Leverage effect test: correlation between past returns and future volatility
#' Negative correlation = leverage (characteristic of equity)
#' Crypto often shows positive (momentum-in-vol) or weak leverage effect
leverage_effect_test <- function(returns, max_lag = 10) {
  n <- length(returns)
  log_r2 <- log(returns^2 + 1e-10)

  corrs <- sapply(seq_len(max_lag), function(k) {
    cor(returns[1:(n-k)], log_r2[(k+1):n])
  })

  # Test significance: t = r * sqrt(n-k-2) / sqrt(1-r^2)
  t_stats <- corrs * sqrt(n - seq_len(max_lag) - 2) /
             sqrt(1 - corrs^2 + 1e-10)
  p_vals  <- 2 * pt(-abs(t_stats), df = n - seq_len(max_lag) - 2)

  cat("=== Leverage Effect Test ===\n")
  cat("Lag | Corr(r_t, log_r^2_{t+k}) | t-stat | p-value\n")
  for (k in seq_len(max_lag)) {
    cat(sprintf("%3d | %25.4f | %6.2f | %.4f\n",
                k, corrs[k], t_stats[k], p_vals[k]))
  }
  cat(sprintf("Sign: %s (negative = leverage, positive = momentum-vol)\n",
              ifelse(corrs[1] < 0, "NEGATIVE", "POSITIVE")))

  list(corrs=corrs, t_stats=t_stats, p_vals=p_vals)
}

# -----------------------------------------------------------------------------
# 11. GARCH MODEL COMPARISON
# -----------------------------------------------------------------------------

#' Compare GARCH, EGARCH, GJR-GARCH on same data
compare_garch_models <- function(returns) {
  cat("=============================================================\n")
  cat("GARCH MODEL COMPARISON\n")
  cat("=============================================================\n\n")

  g11   <- fit_garch11(returns)
  cat("\n")
  eg    <- fit_egarch11(returns)
  cat("\n")
  gjr   <- fit_gjr_garch(returns)
  cat("\n")

  results <- data.frame(
    Model   = c("GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH(1,1)"),
    LogLik  = c(g11$loglik, eg$loglik, gjr$loglik),
    AIC     = c(g11$AIC,    eg$AIC,    gjr$AIC),
    BIC     = c(g11$BIC,    eg$BIC,    gjr$BIC)
  )
  results <- results[order(results$AIC), ]
  cat("\n=== Model Comparison (sorted by AIC) ===\n")
  print(results)
  cat(sprintf("Best model by AIC: %s\n", results$Model[1]))

  invisible(list(garch11=g11, egarch=eg, gjr=gjr, comparison=results))
}

# -----------------------------------------------------------------------------
# 12. MAIN VOLATILITY ANALYSIS PIPELINE
# -----------------------------------------------------------------------------

#' Full volatility analysis on a return series
#' @param returns daily log-returns
#' @param asset_name for labeling output
run_volatility_analysis <- function(returns, asset_name = "Asset",
                                     rv_series = NULL) {
  cat("=============================================================\n")
  cat(sprintf("VOLATILITY ANALYSIS: %s\n", asset_name))
  cat(sprintf("n = %d, period vol = %.4f, annualized = %.2f%%\n",
              length(returns), sd(returns), sd(returns)*sqrt(365)*100))
  cat("=============================================================\n\n")

  # GARCH model comparison
  garch_comp <- compare_garch_models(returns)

  # HAR-RV if realized variance provided
  if (!is.null(rv_series)) {
    cat("\n")
    har_fit <- fit_har_rv(rv_series)
  }

  # Stochastic volatility
  cat("\n")
  sv_fit <- fit_sv_moments(returns)

  # Regime switching
  cat("\n")
  ms_fit <- fit_markov_switching_vol(returns, max_iter=100)

  # Clustering and leverage tests
  cat("\n")
  clust <- vol_clustering_test(returns)
  lev   <- leverage_effect_test(returns, max_lag=5)

  # Term structure from best GARCH
  cat("\n")
  ts <- garch_term_structure(garch_comp$garch11)

  invisible(list(garch=garch_comp, sv=sv_fit, ms=ms_fit,
                 clustering=clust, leverage=lev, term_structure=ts))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 1000
# # Simulate GARCH(1,1) returns
# sigma2 <- numeric(n); r <- numeric(n)
# sigma2[1] <- 0.0004
# omega_t <- 0.00001; alpha_t <- 0.1; beta_t <- 0.88
# for (t in 2:n) {
#   sigma2[t] <- omega_t + alpha_t*r[t-1]^2 + beta_t*sigma2[t-1]
#   r[t] <- rnorm(1, 0, sqrt(sigma2[t]))
# }
# result <- run_volatility_analysis(r, "BTC_simulated")

# =============================================================================
# EXTENDED VOLATILITY MODELS: APARCH, Component GARCH, Volatility Forecasting
# Evaluation, Intraday Volatility Patterns, and Vol Surface Construction
# =============================================================================

# -----------------------------------------------------------------------------
# APARCH (Asymmetric Power ARCH): Ding, Granger & Engle (1993)
# sigma_t^delta = omega + alpha*(|eps_{t-1}| - gamma*eps_{t-1})^delta + beta*sigma_{t-1}^delta
# Nests GARCH (delta=2, gamma=0), GJR (delta=2), TARCH (delta=1), etc.
# Power transformation delta is estimated from data
# -----------------------------------------------------------------------------
aparch_variance <- function(params, returns) {
  omega <- exp(params[1]); alpha <- plogis(params[2]) * 0.3
  gamma <- tanh(params[3]) * 0.9  # asymmetry: in (-1, 1)
  beta  <- plogis(params[4]) * 0.97; delta <- exp(params[5]) + 0.5
  if (alpha + beta >= 1) return(rep(1e6, length(returns)))

  n <- length(returns); sigma_d <- rep(var(returns)^(delta/2), n)
  for (t in 2:n) {
    e <- returns[t-1]
    sigma_d[t] <- omega + alpha*(abs(e) - gamma*e)^delta + beta*sigma_d[t-1]
  }
  sigma2 <- sigma_d^(2/delta); sigma2
}

fit_aparch <- function(returns, ...) {
  obj <- function(p) {
    sigma2 <- aparch_variance(p, returns)
    -sum(dnorm(returns, 0, sqrt(pmax(sigma2, 1e-10)), log=TRUE))
  }
  init <- c(log(var(returns)*0.05), qlogis(0.1/0.3), 0, qlogis(0.85/0.97), log(2-0.5))
  opt  <- optim(init, obj, method="Nelder-Mead", control=list(maxit=2000))
  p    <- opt$par
  list(
    omega = exp(p[1]), alpha = plogis(p[2])*0.3,
    gamma = tanh(p[3])*0.9, beta = plogis(p[4])*0.97,
    delta = exp(p[5])+0.5,
    loglik = -opt$value, convergence = opt$convergence,
    variance = aparch_variance(p, returns)
  )
}

# -----------------------------------------------------------------------------
# Component GARCH (Engle & Lee 1999): decomposes vol into permanent and transitory
# sigma_t^2 = q_t + alpha*(eps_{t-1}^2 - q_{t-1}) + beta*(sigma_{t-1}^2 - q_{t-1})
# q_t = omega + rho*(q_{t-1} - omega) + phi*(eps_{t-1}^2 - sigma_{t-1}^2)
# Captures both short-run and long-run vol components
# -----------------------------------------------------------------------------
component_garch <- function(returns, max_iter = 2000) {
  n <- length(returns)
  unc_var <- var(returns)

  obj <- function(params) {
    omega <- exp(params[1]); rho <- plogis(params[2]) * 0.99
    phi   <- exp(params[3]) * 0.1; alpha <- plogis(params[4]) * 0.3
    beta  <- plogis(params[5]) * 0.95

    q <- rep(unc_var, n); h <- rep(unc_var, n)
    for (t in 2:n) {
      q[t] <- omega + rho*(q[t-1] - omega) + phi*(returns[t-1]^2 - h[t-1])
      q[t] <- max(q[t], 1e-10)
      h[t] <- q[t] + alpha*(returns[t-1]^2 - q[t-1]) + beta*(h[t-1] - q[t-1])
      h[t] <- max(h[t], 1e-10)
    }
    -sum(dnorm(returns, 0, sqrt(h), log=TRUE))
  }

  init <- c(log(unc_var*0.01), qlogis(0.95), log(0.05/0.1), qlogis(0.1/0.3), qlogis(0.85/0.95))
  opt  <- optim(init, obj, method="Nelder-Mead", control=list(maxit=max_iter))
  p    <- opt$par

  omega <- exp(p[1]); rho <- plogis(p[2])*0.99
  phi   <- exp(p[3])*0.1; alpha <- plogis(p[4])*0.3; beta <- plogis(p[5])*0.95

  q <- rep(unc_var, n); h <- rep(unc_var, n)
  for (t in 2:n) {
    q[t] <- omega + rho*(q[t-1]-omega) + phi*(returns[t-1]^2 - h[t-1])
    q[t] <- max(q[t], 1e-10)
    h[t] <- q[t] + alpha*(returns[t-1]^2 - q[t-1]) + beta*(h[t-1] - q[t-1])
    h[t] <- max(h[t], 1e-10)
  }

  list(
    omega=omega, rho=rho, phi=phi, alpha=alpha, beta=beta,
    long_run_variance = q, total_variance = h,
    pct_long_run = mean(q/h),
    loglik = -opt$value, convergence = opt$convergence
  )
}

# -----------------------------------------------------------------------------
# Intraday Volatility Pattern (U-Shape): crypto markets show elevated vol
# at UTC 00:00, 08:00, and 16:00 corresponding to Asian, European, US open
# FFF (Flexible Fourier Form) seasonality model from Andersen & Bollerslev
# -----------------------------------------------------------------------------
intraday_vol_pattern <- function(intraday_returns, n_periods_per_day = 24,
                                  n_harmonics = 3) {
  n_total <- length(intraday_returns)
  n_days  <- floor(n_total / n_periods_per_day)

  # Standardize returns to isolate intraday pattern
  abs_ret_mat <- matrix(abs(intraday_returns[1:(n_days*n_periods_per_day)]),
                         n_days, n_periods_per_day, byrow=TRUE)

  # Average absolute return by intraday period (raw U-shape)
  avg_pattern <- colMeans(abs_ret_mat, na.rm=TRUE)

  # Normalize: divide by overall mean
  pattern_norm <- avg_pattern / mean(avg_pattern)

  # FFF: fit Fourier series to the pattern
  # f(s) = c0 + sum_{j=1}^K [a_j * cos(2*pi*j*s/S) + b_j * sin(2*pi*j*s/S)]
  s <- 0:(n_periods_per_day - 1)
  S <- n_periods_per_day
  X_fff <- cbind(1)
  for (j in 1:n_harmonics) {
    X_fff <- cbind(X_fff, cos(2*pi*j*s/S), sin(2*pi*j*s/S))
  }
  fit_fff <- lm(pattern_norm ~ X_fff - 1)
  fitted_pattern <- as.vector(predict(fit_fff))

  # High-vol periods (> 1 std above mean)
  high_vol_periods <- which(fitted_pattern > 1 + sd(fitted_pattern))

  list(
    raw_pattern = avg_pattern,
    normalized_pattern = pattern_norm,
    fff_fitted = fitted_pattern,
    r2_fff = summary(fit_fff)$r.squared,
    high_vol_periods = high_vol_periods,
    vol_ratio_peak_trough = max(fitted_pattern) / min(fitted_pattern)
  )
}

# -----------------------------------------------------------------------------
# Volatility Risk Premium (VRP) Decomposition
# VRP = Implied Vol^2 - Realized Vol^2 (IV overstates future RV on average)
# In crypto, VRP tends to be large due to demand for option protection
# Components: (1) pure risk premium, (2) jump premium, (3) microstructure
# -----------------------------------------------------------------------------
vrp_decomposition <- function(implied_vol_series, realized_vol_series,
                               jump_component = NULL, window = 21) {
  n <- min(length(implied_vol_series), length(realized_vol_series))
  iv <- implied_vol_series[1:n]; rv <- realized_vol_series[1:n]

  vrp <- iv^2 - rv^2  # variance risk premium (annualized)
  vrp_vol <- iv - rv  # vol risk premium (levels)

  # Rolling VRP
  roll_vrp <- filter(vrp, rep(1/window, window), sides=1)

  # Jump premium: if realized jump component known
  if (!is.null(jump_component)) {
    jc <- jump_component[1:n]
    continuous_rv <- rv^2 - jc  # continuous component of realized variance
    jump_premium <- jc  # premium paid for jump protection
    pure_diffusive_vrp <- iv^2 - continuous_rv^2 - jump_premium
  } else {
    jump_premium <- NA; pure_diffusive_vrp <- NA
  }

  # Predictability: does VRP predict future returns?
  # High VRP (options expensive) often coincides with subsequent rallies
  fwd_rv <- c(rv[-1], NA)
  vrp_rv_corr <- cor(vrp[-n], fwd_rv[-n], use="complete.obs")

  list(
    vrp_variance = vrp,
    vrp_vol_levels = vrp_vol,
    rolling_vrp = roll_vrp,
    avg_vrp = mean(vrp, na.rm=TRUE),
    pct_positive_vrp = mean(vrp > 0, na.rm=TRUE),
    vrp_rv_correlation = vrp_rv_corr,
    jump_premium = jump_premium
  )
}

# Extended volatility example:
# aparch_fit <- fit_aparch(returns_vec)
# cgarch_fit <- component_garch(returns_vec)
# intra_pat  <- intraday_vol_pattern(tick_returns, n_periods_per_day=24)
# vrp_dec    <- vrp_decomposition(iv_series, rv_series)
