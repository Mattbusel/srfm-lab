# =============================================================================
# time_series_advanced.R
# Advanced Time Series: TBATS, Dynamic Factor Model (DFM via EM), DCC-GARCH,
# BEKK-GARCH, PSY explosive root test, LSTAR, Periodic ARMA, Bates-Granger
# optimal forecast combination.
# Pure base R.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. TBATS: Trigonometric + Box-Cox + ARMA + Trend + Seasonal
# ---------------------------------------------------------------------------
# de Livera, Hyndman & Snyder (2011). Handles multiple seasonalities.
# State: level (l), trend (b), seasonal components (s), ARMA errors

#' Box-Cox transformation
box_cox <- function(y, lambda) {
  if (abs(lambda) < 1e-6) log(y)
  else (y^lambda - 1) / lambda
}

box_cox_inv <- function(z, lambda) {
  if (abs(lambda) < 1e-6) exp(z)
  else (lambda * z + 1)^(1/lambda)
}

#' Fit simplified TBATS(lambda, p, q, m) model
#' Single seasonality version; full multi-seasonal is an extension
tbats_fit <- function(y, m = 7,           # Seasonal period (m=7 for weekly)
                       lambda = NULL,      # Box-Cox param (NULL = auto)
                       K = 3,             # Number of trigonometric pairs
                       max_iter = 50,
                       alpha0 = 0.1, beta0 = 0.01, gamma0 = 0.001) {
  n <- length(y)

  # Auto Box-Cox: find lambda via Guerrero method
  if (is.null(lambda)) {
    lambdas <- seq(-1, 1, by = 0.1)
    cv_vals <- sapply(lambdas, function(l) {
      y_tr <- if (abs(l) < 1e-6) log(y) else (y^l - 1) / l
      # CV of seasonal ranges
      groups <- split(y_tr, rep(1:ceiling(n/m), each=m, length.out=n))
      sds    <- sapply(groups, function(g) sd(g, na.rm=TRUE))
      means  <- sapply(groups, function(g) abs(mean(g, na.rm=TRUE)))
      cv <- sd(sds / (means + 1e-8), na.rm=TRUE)
      cv
    })
    lambda <- lambdas[which.min(cv_vals)]
  }

  # Transform
  y_bc <- box_cox(y, lambda)

  # Initialize state
  l <- y_bc[1]  # Level
  b <- 0         # Trend
  # Seasonal: K trigonometric pairs for period m
  # s_j = s_{j-1}*cos(lambda_j) + s*_{j-1}*sin(lambda_j) + gamma*e
  # Initialize seasonals via simple averages
  season_avg <- numeric(m)
  for (j in seq_len(m)) {
    idx <- seq(j, n, by = m)
    season_avg[j] <- mean(y_bc[idx]) - mean(y_bc)
  }

  # Trigonometric seasonal state vectors [s_j, s*_j] for j=1..K
  s  <- numeric(K); ss <- numeric(K)
  for (j in seq_len(K)) {
    omega_j <- 2 * pi * j / m
    s[j]  <- mean(season_avg * cos(omega_j * seq_len(m)))
    ss[j] <- mean(season_avg * sin(omega_j * seq_len(m)))
  }

  # Smoothing parameters
  alpha <- alpha0; beta <- beta0; gamma_vec <- rep(gamma0, K)

  fitted_vals <- numeric(n)
  residuals   <- numeric(n)

  for (t in seq_len(n)) {
    # Compute current seasonal effect
    season_effect <- sum(sapply(seq_len(K), function(j) s[j]))

    # Forecast for time t
    y_hat <- l + b + season_effect
    fitted_vals[t] <- y_hat
    e <- y_bc[t] - y_hat
    residuals[t] <- e

    # Update state
    l_new <- l + b + alpha * e
    b_new <- b + beta * e

    # Update seasonal trig components
    s_new  <- numeric(K); ss_new <- numeric(K)
    for (j in seq_len(K)) {
      omega_j <- 2 * pi * j / m
      s_new[j]  <- s[j]  * cos(omega_j) + ss[j] * sin(omega_j) + gamma_vec[j] * e
      ss_new[j] <- -s[j] * sin(omega_j) + ss[j] * cos(omega_j) + gamma_vec[j] * e
    }

    l <- l_new; b <- b_new; s <- s_new; ss <- ss_new
  }

  # Forecast h steps ahead
  forecast_fn <- function(h = 10) {
    l_f <- l; b_f <- b; s_f <- s; ss_f <- ss
    fcast <- numeric(h)
    for (i in seq_len(h)) {
      se <- sum(s_f)
      fcast[i] <- l_f + i * b_f + se
      # Evolve seasonal without noise
      for (j in seq_len(K)) {
        omega_j <- 2*pi*j/m
        s_tmp  <- s_f[j]*cos(omega_j) + ss_f[j]*sin(omega_j)
        ss_tmp <- -s_f[j]*sin(omega_j) + ss_f[j]*cos(omega_j)
        s_f[j] <- s_tmp; ss_f[j] <- ss_tmp
      }
      l_f <- l_f + b_f
    }
    # Back-transform
    box_cox_inv(fcast, lambda)
  }

  list(
    fitted    = box_cox_inv(fitted_vals, lambda),
    residuals = residuals,
    lambda    = lambda,
    alpha     = alpha, beta = beta,
    forecast  = forecast_fn,
    aic       = n * log(mean(residuals^2)) + 2 * (3 + 2*K)  # Rough AIC
  )
}

# ---------------------------------------------------------------------------
# 2. DYNAMIC FACTOR MODEL (DFM) VIA EM ALGORITHM
# ---------------------------------------------------------------------------
# X_t = Lambda * F_t + e_t,  e_t ~ N(0, Sigma)
# F_t = A * F_{t-1} + u_t,   u_t ~ N(0, Q)
# Stock-Watson (1989), Bai-Ng (2002)

#' Initialize DFM parameters
dfm_init <- function(X, k = 1) {
  n <- ncol(X); T_ <- nrow(X)

  # PCA initialization
  X_std <- scale(X)
  X_std[is.na(X_std)] <- 0
  svd_r <- svd(X_std, nu = k, nv = k)

  F_init  <- X_std %*% svd_r$v[, seq_len(k), drop=FALSE]
  Lambda_init <- t(svd_r$v[, seq_len(k), drop=FALSE]) * svd_r$d[seq_len(k)]
  Lambda_init <- t(Lambda_init)

  # Residuals
  resid   <- X_std - F_init %*% t(Lambda_init)
  Sigma   <- diag(apply(resid, 2, var))

  # Factor AR(1): F_t = A*F_{t-1} + u
  A_init  <- diag(0.9, k)
  Q_init  <- diag(1, k)

  list(Lambda = Lambda_init, Sigma = Sigma, A = A_init, Q = Q_init,
       F_init = F_init, X_std = X_std, k = k)
}

#' Kalman filter for DFM
kalman_filter <- function(X, Lambda, Sigma, A, Q, k) {
  T_ <- nrow(X); n <- ncol(X)

  # Initial state
  F_filtered  <- matrix(0, T_, k)
  P_filtered  <- array(diag(1, k), dim = c(k, k, T_))
  F_predicted <- matrix(0, T_, k)
  P_predicted <- array(diag(1, k), dim = c(k, k, T_))
  loglik <- 0

  F_curr <- rep(0, k)
  P_curr <- diag(10, k)

  for (t in seq_len(T_)) {
    # Predict
    F_pred <- A %*% F_curr
    P_pred <- A %*% P_curr %*% t(A) + Q

    F_predicted[t, ] <- F_pred
    P_predicted[, , t] <- P_pred

    # Update using observation X_t
    x_t <- as.vector(X[t, ])
    valid <- !is.na(x_t)

    if (any(valid)) {
      Lambda_v <- Lambda[valid, , drop=FALSE]
      Sigma_v  <- Sigma[valid, valid, drop=FALSE]

      # Innovation
      innov <- x_t[valid] - Lambda_v %*% F_pred

      # Innovation covariance
      S <- Lambda_v %*% P_pred %*% t(Lambda_v) + Sigma_v
      S_inv <- tryCatch(solve(S), error = function(e) solve(S + diag(1e-6, nrow(S))))

      # Kalman gain
      K_gain <- P_pred %*% t(Lambda_v) %*% S_inv

      # Update
      F_curr <- F_pred + as.vector(K_gain %*% innov)
      P_curr <- (diag(1, k) - K_gain %*% Lambda_v) %*% P_pred

      # Log-likelihood contribution
      sign_det <- tryCatch(determinant(S, logarithm=TRUE), error=function(e) list(modulus=-Inf))
      loglik   <- loglik - 0.5 * (sum(valid) * log(2*pi) + sign_det$modulus +
                                   as.numeric(t(innov) %*% S_inv %*% innov))
    } else {
      F_curr <- F_pred; P_curr <- P_pred
    }

    F_filtered[t, ]  <- F_curr
    P_filtered[, , t] <- P_curr
  }

  list(F_filtered = F_filtered, P_filtered = P_filtered,
       F_predicted = F_predicted, P_predicted = P_predicted,
       loglik = loglik)
}

#' EM algorithm for DFM
dfm_em <- function(X, k = 1, max_iter = 50, tol = 1e-4) {
  X_std <- scale(X); X_std[is.na(X_std)] <- 0
  n <- ncol(X_std); T_ <- nrow(X_std)

  # Initialize
  params <- dfm_init(X_std, k)
  Lambda <- params$Lambda; Sigma <- params$Sigma
  A      <- params$A;      Q     <- params$Q

  logliks <- numeric(max_iter)

  for (iter in seq_len(max_iter)) {
    # E-step: Kalman filter + smoother
    kf <- kalman_filter(X_std, Lambda, Sigma, A, Q, k)
    logliks[iter] <- kf$loglik

    # Simplified M-step (ignore smoother for brevity)
    F_ <- kf$F_filtered

    # Update Lambda: OLS of X on F
    Lambda_new <- t(X_std) %*% F_ %*% solve(t(F_) %*% F_ + diag(1e-8, k))

    # Update Sigma
    resid    <- X_std - F_ %*% t(Lambda_new)
    Sigma_new <- diag(apply(resid, 2, var))

    # Update A: AR(1) on factors
    if (T_ > k + 1) {
      F_lag <- F_[-T_, ]; F_curr <- F_[-1, ]
      A_new <- t(F_curr) %*% F_lag %*% solve(t(F_lag) %*% F_lag + diag(1e-8, k))
    } else A_new <- A

    # Update Q
    u_ <- F_curr - F_lag %*% t(A_new)
    Q_new <- cov(u_)

    # Check convergence
    if (iter > 1 && abs(logliks[iter] - logliks[iter-1]) < tol) break

    Lambda <- Lambda_new; Sigma <- Sigma_new; A <- A_new; Q <- Q_new
  }

  list(
    factors = F_,
    Lambda  = Lambda,
    Sigma   = Sigma,
    A       = A, Q = Q,
    loglik  = logliks[seq_len(iter)],
    k       = k
  )
}

# ---------------------------------------------------------------------------
# 3. DCC-GARCH: DYNAMIC CONDITIONAL CORRELATION
# ---------------------------------------------------------------------------
# Engle (2002): model time-varying correlation between assets
# Step 1: Fit univariate GARCH(1,1) to each asset
# Step 2: Fit DCC to standardized residuals

#' Univariate GARCH(1,1) via QML estimation
garch11_fit <- function(r, omega0 = 1e-5, alpha0 = 0.1, beta0 = 0.85,
                         max_iter = 200) {
  n <- length(r)

  # Negative log-likelihood for GARCH(1,1)
  neg_loglik <- function(params) {
    omega <- exp(params[1]); alpha <- sigmoid_clip(params[2]); beta <- sigmoid_clip(params[3])
    if (alpha + beta >= 0.9999) return(1e8)

    h <- numeric(n)
    h[1] <- var(r)
    for (t in 2:n) {
      h[t] <- omega + alpha * r[t-1]^2 + beta * h[t-1]
    }
    if (any(h <= 0)) return(1e8)
    0.5 * sum(log(h) + r^2 / h)
  }

  sigmoid_clip <- function(x) 1 / (1 + exp(-x)) * 0.99

  # Initial parameters (log-scale for omega, logit-scale for alpha/beta)
  p0 <- c(log(omega0), log(alpha0/(1-alpha0)), log(beta0/(1-beta0)))

  # Nelder-Mead optimization (simple implementation)
  result <- tryCatch({
    optim_nm(neg_loglik, p0, max_iter = max_iter)
  }, error = function(e) list(par = p0))

  omega <- exp(result$par[1])
  alpha <- sigmoid_clip(result$par[2])
  beta  <- sigmoid_clip(result$par[3])

  h <- numeric(n); h[1] <- var(r)
  for (t in 2:n) h[t] <- omega + alpha * r[t-1]^2 + beta * h[t-1]

  list(omega=omega, alpha=alpha, beta=beta, variance=h,
       std_resid = r / sqrt(pmax(h, 1e-10)),
       loglik = -neg_loglik(result$par))
}

# Simple Nelder-Mead optimizer
optim_nm <- function(f, x0, max_iter = 500, tol = 1e-8) {
  n <- length(x0)
  sim <- rbind(x0, sweep(diag(n)*0.1, 2, x0, "+"))
  fv  <- apply(sim, 1, f)
  for (iter in seq_len(max_iter)) {
    ord <- order(fv); sim <- sim[ord,]; fv <- fv[ord]
    if (diff(range(fv)) < tol) break
    xo  <- colMeans(sim[-nrow(sim), , drop=FALSE])
    xr  <- xo + (xo - sim[nrow(sim),])
    fr  <- f(xr)
    if (fr < fv[1]) {
      xe <- xo + 2*(xr - xo); fe <- f(xe)
      if (fe < fr) { sim[nrow(sim),] <- xe; fv[nrow(sim)] <- fe }
      else         { sim[nrow(sim),] <- xr; fv[nrow(sim)] <- fr }
    } else if (fr < fv[nrow(sim)-1]) {
      sim[nrow(sim),] <- xr; fv[nrow(sim)] <- fr
    } else {
      xc <- xo + 0.5*(sim[nrow(sim),]-xo); fc <- f(xc)
      if (fc < fv[nrow(sim)]) { sim[nrow(sim),] <- xc; fv[nrow(sim)] <- fc }
      else {
        for (i in 2:nrow(sim)) sim[i,] <- sim[1,] + 0.5*(sim[i,]-sim[1,])
        fv <- apply(sim, 1, f)
      }
    }
  }
  list(par = sim[1,], value = fv[1])
}

#' DCC-GARCH(1,1) estimation
#' Engle (2002) two-step estimator
dcc_garch <- function(returns_matrix, max_iter = 100) {
  n_assets <- ncol(returns_matrix)
  T_       <- nrow(returns_matrix)

  # Step 1: Univariate GARCH for each asset
  garch_fits <- lapply(seq_len(n_assets), function(i) {
    garch11_fit(returns_matrix[, i])
  })

  # Standardized residuals
  Z <- matrix(NA, T_, n_assets)
  for (i in seq_len(n_assets)) {
    Z[, i] <- garch_fits[[i]]$std_resid
  }

  # Unconditional correlation
  R_bar <- cor(Z)

  # Step 2: DCC parameters (a, b) via concentrated log-likelihood
  dcc_loglik <- function(params) {
    a <- 1/(1+exp(-params[1])) * 0.1
    b <- 1/(1+exp(-params[2])) * 0.9
    if (a + b >= 0.9999) return(1e8)

    Q_t <- R_bar
    loglik <- 0

    for (t in seq_len(T_)) {
      z_t <- as.vector(Z[t, ])

      if (t > 1) {
        z_prev <- as.vector(Z[t-1, ])
        Q_t <- (1 - a - b) * R_bar + a * outer(z_prev, z_prev) + b * Q_t
      }

      # Normalize Q_t to get R_t
      D_inv <- diag(1 / sqrt(pmax(diag(Q_t), 1e-10)))
      R_t <- D_inv %*% Q_t %*% D_inv

      # Log-likelihood
      det_R <- det(R_t)
      if (det_R <= 0) return(1e8)
      loglik <- loglik - 0.5 * (log(det_R) +
                                  as.numeric(t(z_t) %*% solve(R_t) %*% z_t) -
                                  sum(z_t^2))
    }
    -loglik
  }

  result <- tryCatch(optim_nm(dcc_loglik, c(0, 0)), error = function(e) list(par=c(0,0)))
  a_dcc <- 1/(1+exp(-result$par[1])) * 0.1
  b_dcc <- 1/(1+exp(-result$par[2])) * 0.9

  # Compute dynamic correlations
  Q_path <- array(NA, dim = c(n_assets, n_assets, T_))
  R_path <- array(NA, dim = c(n_assets, n_assets, T_))
  Q_t    <- R_bar

  for (t in seq_len(T_)) {
    if (t > 1) {
      z_prev <- as.vector(Z[t-1, ])
      Q_t    <- (1 - a_dcc - b_dcc) * R_bar + a_dcc * outer(z_prev, z_prev) + b_dcc * Q_t
    }
    Q_path[, , t] <- Q_t
    D_inv  <- diag(1 / sqrt(pmax(diag(Q_t), 1e-10)))
    R_path[, , t] <- D_inv %*% Q_t %*% D_inv
  }

  list(
    garch_fits = garch_fits,
    a = a_dcc, b = b_dcc,
    R_bar = R_bar,
    dynamic_correlations = R_path,
    std_residuals = Z,
    # Extract pairwise correlation series
    corr_series = function(i, j) sapply(seq_len(T_), function(t) R_path[i, j, t])
  )
}

# ---------------------------------------------------------------------------
# 4. BEKK-GARCH (MULTIVARIATE GARCH WITH FULL INTERACTION)
# ---------------------------------------------------------------------------
# H_t = C'C + A'eps_{t-1}eps_{t-1}'A + B'H_{t-1}B
# Baba-Engle-Kraft-Kroner (1990/1995)

#' Simplified BEKK-GARCH(1,1) for 2 assets (diagonal BEKK)
bekk_garch_diagonal <- function(returns_matrix, max_iter = 50) {
  n <- ncol(returns_matrix); T_ <- nrow(returns_matrix)

  # Diagonal BEKK: A and B are diagonal matrices
  # H_t = C'C + diag(a)^2 * eps*eps' * diag(a)^2 + diag(b)^2 * H_{t-1} * diag(b)^2

  neg_loglik <- function(params) {
    c_vec <- params[1:n]  # Lower cholesky of C'C
    a_vec <- abs(params[(n+1):(2*n)])
    b_vec <- abs(params[(2*n+1):(3*n)])

    if (any(a_vec^2 + b_vec^2 >= 1)) return(1e8)

    # C'C
    C_sq <- diag(c_vec^2) + diag(1e-6, n)

    H_t <- cov(returns_matrix)  # Initial H
    loglik <- 0

    for (t in 2:T_) {
      eps <- as.vector(returns_matrix[t-1, ])
      eps_outer <- outer(eps, eps)

      H_new <- C_sq +
               diag(a_vec) %*% eps_outer %*% diag(a_vec) +
               diag(b_vec) %*% H_t %*% diag(b_vec)

      H_t <- H_new
      det_H <- det(H_t)
      if (det_H <= 0 || !is.finite(det_H)) return(1e8)

      eps_t <- as.vector(returns_matrix[t, ])
      loglik <- loglik - 0.5 * (log(det_H) +
                                  as.numeric(t(eps_t) %*% solve(H_t) %*% eps_t))
    }
    -loglik
  }

  init_par <- c(rep(0.01, n), rep(0.3, n), rep(0.85, n))
  result <- tryCatch(optim_nm(neg_loglik, init_par, max_iter=max_iter),
                     error = function(e) list(par = init_par))

  # Compute fitted variances/covariances
  a_v <- abs(result$par[(n+1):(2*n)])
  b_v <- abs(result$par[(2*n+1):(3*n)])
  c_v <- result$par[1:n]

  list(a = a_v, b = b_v, c = c_v, params = result$par, loglik = -result$value)
}

# ---------------------------------------------------------------------------
# 5. PSY EXPLOSIVE ROOT TEST (RIGHT-TAILED ADF FOR BUBBLES)
# ---------------------------------------------------------------------------
# Phillips, Shi, Yu (2015): right-tailed unit root test to detect bubbles
# GSADF (Generalized Sup ADF) statistic

#' ADF test statistic (right-tailed, for explosiveness)
adf_right_tail <- function(y, lag = 1) {
  n <- length(y)
  dy <- diff(y)
  T_ <- length(dy)

  if (T_ < lag + 5) return(NA)

  # Build regression matrix
  y_lag1 <- y[lag:(T_-1+lag)]  # y_{t-1}
  if (lag > 0) {
    dy_lags <- matrix(NA, T_ - lag, lag)
    for (l in seq_len(lag)) {
      dy_lags[, l] <- dy[(lag - l + 1):(T_ - l)]
    }
    X <- cbind(1, y_lag1[(lag+1):T_], dy_lags)
    Y <- dy[(lag+1):T_]
  } else {
    X <- cbind(1, y_lag1)
    Y <- dy
  }

  n_obs <- length(Y)
  if (n_obs < 5) return(NA)

  XtX_inv <- tryCatch(solve(crossprod(X) + diag(1e-8, ncol(X))),
                       error = function(e) NULL)
  if (is.null(XtX_inv)) return(NA)

  beta  <- XtX_inv %*% crossprod(X, Y)
  resid <- Y - X %*% beta
  s2    <- sum(resid^2) / (n_obs - ncol(X))
  se_rho <- sqrt(s2 * XtX_inv[2, 2])

  t_stat <- beta[2] / se_rho
  t_stat
}

#' SADF (Sup ADF): supremum of ADF statistics over expanding windows
sadf_test <- function(y, r0 = 0.10, max_window = NULL) {
  n <- length(y)
  min_obs <- max(10, round(r0 * n))
  if (is.null(max_window)) max_window <- n

  adf_stats <- sapply(min_obs:max_window, function(t_end) {
    adf_right_tail(y[1:t_end], lag = 1)
  })

  sadf_stat <- max(adf_stats, na.rm = TRUE)

  # Critical values (simulated, approximate for n=100)
  cv_90 <- 1.15; cv_95 <- 1.42; cv_99 <- 1.98

  list(
    sadf_stat = sadf_stat,
    cv_90 = cv_90, cv_95 = cv_95, cv_99 = cv_99,
    bubble_detected_95 = sadf_stat > cv_95,
    adf_series = adf_stats
  )
}

#' BSADF (Backward SADF): rolling detection of bubble periods
bsadf_test <- function(y, r0 = 0.10) {
  n   <- length(y)
  r   <- round(r0 * n)
  bsadf <- rep(NA, n)

  for (t in (r + r):n) {
    # Backward: start = [t-r_w, t], varying start
    stats_t <- sapply(1:(t-r), function(s) {
      if (t - s < r) return(NA)
      adf_right_tail(y[s:t], lag=1)
    })
    bsadf[t] <- max(stats_t, na.rm=TRUE)
  }

  # Date-stamping: bubble when bsadf > critical value
  cv_95 <- 1.42
  bubble_periods <- bsadf > cv_95 & !is.na(bsadf)

  list(bsadf = bsadf, bubble_periods = bubble_periods,
       n_bubble_periods = sum(bubble_periods, na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 6. LSTAR: LOGISTIC SMOOTH TRANSITION AR
# ---------------------------------------------------------------------------
# y_t = (phi1_0 + phi1_1*y_{t-1} + ... + phi1_p*y_{t-p}) * (1-F) +
#        (phi2_0 + phi2_1*y_{t-1} + ... + phi2_p*y_{t-p}) * F + e_t
# where F = 1/(1+exp(-gamma*(s_t - c)))  [logistic transition function]

#' LSTAR transition function
lstar_transition <- function(s, gamma, c) {
  1 / (1 + exp(-gamma * (s - c)))
}

#' LSTAR(p) estimation by nonlinear least squares (grid search for gamma,c)
lstar_fit <- function(y, p = 1, d = 1,  # d: lag for transition variable
                       gamma_grid = c(1, 5, 10, 20),
                       c_quantiles = seq(0.15, 0.85, by = 0.05)) {
  n <- length(y)
  max_lag <- max(p, d)

  # Transition variable: y_{t-d}
  s <- y[(max_lag - d + 1):(n - d)]

  # Response and predictors
  Y  <- y[(max_lag + 1):n]
  T_ <- length(Y)

  X_lin <- matrix(NA, T_, p + 1)
  X_lin[, 1] <- 1
  for (j in seq_len(p)) {
    X_lin[, j + 1] <- y[(max_lag + 1 - j):(n - j)]
  }

  best_ssr <- Inf; best_params <- NULL

  for (gamma in gamma_grid) {
    for (c_q in c_quantiles) {
      c_val <- quantile(s, c_q)
      F_t   <- lstar_transition(s, gamma, c_val)

      # Linear regression: [X_lin, X_lin * F_t] on Y
      X_full <- cbind(X_lin, X_lin * F_t)
      XtX_inv <- tryCatch(solve(crossprod(X_full) + diag(1e-8, ncol(X_full))),
                           error = function(e) NULL)
      if (is.null(XtX_inv)) next

      beta <- XtX_inv %*% crossprod(X_full, Y)
      resid <- Y - X_full %*% beta
      ssr   <- sum(resid^2)

      if (ssr < best_ssr) {
        best_ssr <- ssr; best_params <- list(gamma=gamma, c=c_val, beta=beta, F_t=F_t)
      }
    }
  }

  if (is.null(best_params)) stop("LSTAR fitting failed")

  # Fitted values
  F_t   <- best_params$F_t
  X_lin_full <- cbind(X_lin, X_lin * F_t)
  fitted <- as.vector(X_lin_full %*% best_params$beta)
  resid  <- Y - fitted

  list(
    gamma = best_params$gamma,
    c     = best_params$c,
    beta  = best_params$beta,
    fitted = fitted,
    residuals = resid,
    ssr    = best_ssr,
    sigma2 = best_ssr / T_,
    transition = best_params$F_t
  )
}

# ---------------------------------------------------------------------------
# 7. PERIODIC ARMA
# ---------------------------------------------------------------------------
# AR coefficients vary by season/period
# phi_s(B) * y_t = theta_s(B) * e_t where s = season(t)

#' Periodic AR(p) estimation
#' Separate AR for each period within the seasonal cycle
periodic_ar_fit <- function(y, period = 7, p = 2) {
  n <- length(y)
  seasons <- ((seq_len(n) - 1) %% period) + 1

  # Fit separate AR(p) for each season
  fits <- lapply(seq_len(period), function(s) {
    # Observations at season s
    idx_s <- which(seasons == s)
    idx_s <- idx_s[idx_s > p]  # Need p lags

    if (length(idx_s) < p + 2) return(NULL)

    # Response
    Y_s <- y[idx_s]

    # Lagged predictors (standard lags, not seasonal lags)
    X_s <- matrix(NA, length(idx_s), p + 1)
    X_s[, 1] <- 1
    for (j in seq_len(p)) X_s[, j + 1] <- y[idx_s - j]

    valid <- complete.cases(X_s) & !is.na(Y_s)
    if (sum(valid) < p + 2) return(NULL)

    XtX_inv <- solve(crossprod(X_s[valid,]) + diag(1e-8, p+1))
    beta <- XtX_inv %*% crossprod(X_s[valid,], Y_s[valid])
    resid <- Y_s[valid] - X_s[valid,] %*% beta

    list(beta = beta, sigma2 = mean(resid^2), season = s, n_obs = sum(valid))
  })

  # Compute fitted values
  fitted <- rep(NA, n)
  for (t in (p+1):n) {
    s <- seasons[t]
    fit_s <- fits[[s]]
    if (!is.null(fit_s)) {
      x_t <- c(1, y[t - seq_len(p)])
      fitted[t] <- sum(fit_s$beta * x_t)
    }
  }

  list(fits = fits, fitted = fitted, period = period, p = p,
       residuals = y - fitted)
}

# ---------------------------------------------------------------------------
# 8. FORECAST COMBINATION: BATES-GRANGER OPTIMAL WEIGHTS
# ---------------------------------------------------------------------------
# Bates & Granger (1969): optimal linear combination of forecasts
# minimizes MSE when individual forecast errors have known variance-covariance

#' Bates-Granger optimal combination weights
#' W = Sigma^{-1} * 1 / (1' * Sigma^{-1} * 1)
bates_granger_weights <- function(forecast_errors_matrix) {
  # forecast_errors_matrix: [T x k] where k = number of forecasters
  k <- ncol(forecast_errors_matrix)

  Sigma <- cov(forecast_errors_matrix, use = "complete.obs")
  Sigma_reg <- Sigma + diag(1e-8, k)  # Regularize

  ones  <- rep(1, k)
  Sinv  <- solve(Sigma_reg)
  w_num <- Sinv %*% ones
  w_den <- as.numeric(t(ones) %*% Sinv %*% ones)

  w <- as.vector(w_num / w_den)
  w <- w / sum(w)  # Ensure sum to 1

  list(
    weights = w,
    covariance = Sigma,
    effective_k = 1 / sum(w^2)  # Effective number of forecasters
  )
}

#' Rolling forecast combination with Bates-Granger weights
rolling_forecast_combination <- function(forecasts_matrix, realized,
                                          window = 60) {
  T_ <- nrow(forecasts_matrix); k <- ncol(forecasts_matrix)

  combined <- rep(NA, T_)
  weights_history <- matrix(NA, T_, k)

  for (t in (window + 1):T_) {
    train_idx <- (t - window):(t - 1)
    errors    <- realized[train_idx] - forecasts_matrix[train_idx, , drop=FALSE]
    errors    <- errors[complete.cases(errors), ]

    if (nrow(errors) < 10) next

    bg <- bates_granger_weights(errors)
    w  <- bg$weights

    weights_history[t, ] <- w
    combined[t] <- sum(w * forecasts_matrix[t, ])
  }

  list(
    combined = combined,
    weights  = weights_history,
    equal_weight_forecast = rowMeans(forecasts_matrix, na.rm=TRUE)
  )
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  set.seed(42)
  T_ <- 500

  # Generate data with weekly seasonality
  t <- seq_len(T_)
  y_seasonal <- 100 + 0.05*t + 10*sin(2*pi*t/7) + 5*sin(2*pi*t/30) +
                cumsum(rnorm(T_, 0, 0.5))

  # TBATS
  cat("=== TBATS ===\n")
  tb <- tbats_fit(y_seasonal, m = 7, K = 3)
  rmse_tbats <- sqrt(mean(tb$residuals^2, na.rm=TRUE))
  cat(sprintf("TBATS RMSE: %.3f | Lambda: %.3f\n", rmse_tbats, tb$lambda))
  fcast <- tb$forecast(10)
  cat(sprintf("10-step forecast: %.2f to %.2f\n", min(fcast), max(fcast)))

  # DFM
  cat("\n=== Dynamic Factor Model ===\n")
  X_multi <- matrix(rnorm(T_*5, 0, 1), T_, 5)
  common  <- cumsum(rnorm(T_, 0, 0.5))
  for (i in seq_len(5)) X_multi[, i] <- X_multi[, i] + 0.8*common
  dfm_result <- dfm_em(X_multi, k=1, max_iter=20)
  cat(sprintf("DFM log-lik final: %.2f\n", tail(dfm_result$loglik, 1)))

  # DCC-GARCH
  cat("\n=== DCC-GARCH ===\n")
  btc_rets <- rnorm(300, 0, 0.04)
  eth_rets <- 0.8 * btc_rets + rnorm(300, 0, 0.025)
  ret_mat2 <- cbind(btc_rets, eth_rets)
  dcc <- dcc_garch(ret_mat2, max_iter=30)
  avg_corr <- mean(dcc$corr_series(1, 2))
  cat(sprintf("DCC: a=%.4f, b=%.4f, avg corr=%.3f\n", dcc$a, dcc$b, avg_corr))

  # PSY Bubble Test
  cat("\n=== PSY Bubble Test ===\n")
  # Simulate bubble
  y_bubble <- c(rep(1, 100), cumprod(c(1, exp(rnorm(100, 0.02, 0.03)))),
                 rep(3, 50), cumprod(c(3, exp(rnorm(100, -0.01, 0.03)))))
  sadf <- sadf_test(y_bubble)
  cat(sprintf("SADF: %.3f | Bubble detected: %s\n",
              sadf$sadf_stat, sadf$bubble_detected_95))

  # Bates-Granger forecast combination
  cat("\n=== Bates-Granger Combination ===\n")
  true_vals <- cumsum(rnorm(200))
  forecasts <- cbind(
    true_vals + rnorm(200, 0, 0.5),
    true_vals + rnorm(200, 0, 0.8),
    true_vals + rnorm(200, 0.1, 0.4)
  )
  combo <- rolling_forecast_combination(forecasts[-200,], true_vals[-1])
  cat(sprintf("BG combined MSE:  %.4f\n", mean((combo$combined - true_vals[-200])^2, na.rm=TRUE)))
  cat(sprintf("EW combined MSE:  %.4f\n", mean((combo$equal_weight_forecast - true_vals[-200])^2, na.rm=TRUE)))
}
