###############################################################################
# time_series_models.R
# Complete Time Series Modeling Library in R
# ARIMA, GARCH, Exponential Smoothing, Structural, VAR, VECM,
# Regime Switching, Threshold, Spectral, Wavelet, Long Memory, Tests
###############################################################################

# =============================================================================
# SECTION 1: ARIMA
# =============================================================================

#' Difference a time series
#'
#' @param x numeric vector
#' @param d order of differencing
#' @param D seasonal differencing order
#' @param s seasonal period
#' @return differenced series
#' @export
ts_diff <- function(x, d = 1, D = 0, s = 1) {
  result <- x
  # Regular differencing
  for (i in seq_len(d)) {
    result <- diff(result)
  }
  # Seasonal differencing
  for (i in seq_len(D)) {
    n <- length(result)
    if (n <= s) break
    result <- result[(s + 1):n] - result[1:(n - s)]
  }
  result
}

#' Inverse difference to recover original scale
#' @keywords internal
ts_undiff <- function(x_diff, x_orig, d = 1, D = 0, s = 1) {
  result <- x_diff
  # Undo seasonal differencing
  for (i in seq_len(D)) {
    n <- length(result)
    prefix <- tail(x_orig, s)
    extended <- numeric(n + s)
    extended[1:s] <- prefix
    for (j in seq_len(n)) {
      extended[s + j] <- result[j] + extended[j]
    }
    result <- extended[(s + 1):(s + n)]
  }
  # Undo regular differencing
  for (i in seq_len(d)) {
    last_val <- tail(x_orig, 1)
    result <- cumsum(c(last_val, result))[-1]
  }
  result
}

#' Compute ACF
#'
#' @param x numeric vector
#' @param max_lag maximum lag
#' @return autocorrelation values
#' @export
acf_compute <- function(x, max_lag = NULL) {
  n <- length(x)
  if (is.null(max_lag)) max_lag <- min(n - 1, floor(10 * log10(n)))
  x_centered <- x - mean(x)
  var_x <- sum(x_centered^2) / n

  acf_vals <- numeric(max_lag + 1)
  for (h in 0:max_lag) {
    acf_vals[h + 1] <- sum(x_centered[1:(n - h)] * x_centered[(1 + h):n]) /
      (n * var_x)
  }
  acf_vals
}

#' Compute PACF (Levinson-Durbin)
#'
#' @param x numeric vector
#' @param max_lag maximum lag
#' @return partial autocorrelation values
#' @export
pacf_compute <- function(x, max_lag = NULL) {
  n <- length(x)
  if (is.null(max_lag)) max_lag <- min(n - 1, floor(10 * log10(n)))

  acf_vals <- acf_compute(x, max_lag)[-1]  # Remove lag 0
  pacf_vals <- numeric(max_lag)

  # Levinson-Durbin recursion
  phi <- matrix(0, nrow = max_lag, ncol = max_lag)

  phi[1, 1] <- acf_vals[1]
  pacf_vals[1] <- acf_vals[1]

  for (k in 2:max_lag) {
    num <- acf_vals[k] - sum(phi[k - 1, 1:(k - 1)] *
                               acf_vals[(k - 1):1])
    den <- 1 - sum(phi[k - 1, 1:(k - 1)] * acf_vals[1:(k - 1)])

    phi[k, k] <- num / den
    pacf_vals[k] <- phi[k, k]

    for (j in 1:(k - 1)) {
      phi[k, j] <- phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
    }
  }

  pacf_vals
}

#' Fit AR model via Yule-Walker
#'
#' @param x time series
#' @param p AR order
#' @return list with coefficients, sigma2, aic
#' @export
ar_yule_walker <- function(x, p) {
  n <- length(x)
  x_centered <- x - mean(x)

  if (p == 0) {
    sigma2 <- var(x)
    aic <- n * log(sigma2) + 2
    return(list(coefficients = numeric(0), sigma2 = sigma2,
                aic = aic, mean = mean(x)))
  }

  acf_vals <- acf_compute(x_centered, max_lag = p)

  # Toeplitz system
  R <- toeplitz(acf_vals[1:p])
  r <- acf_vals[2:(p + 1)]

  phi <- solve(R, r)

  # Residual variance
  sigma2 <- acf_vals[1] * (1 - sum(phi * acf_vals[2:(p + 1)])) * var(x)

  aic <- n * log(sigma2) + 2 * (p + 1)

  list(
    coefficients = phi,
    sigma2 = sigma2,
    aic = aic,
    mean = mean(x),
    p = p
  )
}

#' Innovations algorithm for MA estimation
#' @keywords internal
innovations_algorithm <- function(acvf, n, q) {
  theta <- matrix(0, nrow = n, ncol = n)
  v <- numeric(n + 1)
  v[1] <- acvf[1]

  for (i in 1:n) {
    for (k in 0:(i - 1)) {
      if (k == 0) {
        theta[i, i - k] <- acvf[i + 1] / v[1]
      } else {
        s <- sum(theta[k, k - (1:min(k - 1, i - 1)) + 1] *
                   theta[i, i - (1:min(k - 1, i - 1)) + 1] *
                   v[1:min(k - 1, i - 1) + 1])
        theta[i, i - k] <- (acvf[i - k + 1] - s) / v[k + 1]
      }
    }
    v[i + 1] <- acvf[1] - sum(theta[i, 1:i]^2 * v[1:i])
  }

  # Extract MA coefficients
  if (q > 0) {
    ma_coef <- theta[n, (n - q + 1):n]
  } else {
    ma_coef <- numeric(0)
  }

  list(theta = ma_coef, v = v)
}

#' ARIMA estimation via CSS-MLE
#'
#' Conditional Sum of Squares initialization followed by
#' approximate Maximum Likelihood refinement.
#'
#' @param x time series
#' @param order c(p, d, q) ARIMA order
#' @param seasonal list(order = c(P, D, Q), period = s) optional
#' @param include.mean logical
#' @param method "CSS-ML" or "CSS" or "ML"
#' @return ARIMA model
#' @export
arima_fit <- function(x, order = c(1, 0, 0), seasonal = NULL,
                       include.mean = TRUE, method = "CSS-ML") {
  p <- order[1]
  d <- order[2]
  q <- order[3]

  n_orig <- length(x)

  # Handle seasonal
  P <- 0; D <- 0; Q <- 0; s <- 1
  if (!is.null(seasonal)) {
    P <- seasonal$order[1]
    D <- seasonal$order[2]
    Q <- seasonal$order[3]
    s <- seasonal$period
  }

  # Difference
  x_diff <- ts_diff(x, d = d, D = D, s = s)
  n <- length(x_diff)

  if (include.mean && d == 0 && D == 0) {
    x_mean <- mean(x_diff)
    x_work <- x_diff - x_mean
  } else {
    x_mean <- 0
    x_work <- x_diff
  }

  # Total AR and MA orders (including seasonal)
  p_total <- p + P * s
  q_total <- q + Q * s
  n_params <- p + q + P + Q + as.integer(include.mean && d == 0 && D == 0)

  # CSS objective function
  css_objective <- function(params) {
    ar_coefs <- if (p > 0) params[1:p] else numeric(0)
    ma_coefs <- if (q > 0) params[(p + 1):(p + q)] else numeric(0)
    sar_coefs <- if (P > 0) params[(p + q + 1):(p + q + P)] else numeric(0)
    sma_coefs <- if (Q > 0) params[(p + q + P + 1):(p + q + P + Q)] else numeric(0)

    # Expand seasonal into full polynomial
    ar_full <- expand_seasonal_poly(ar_coefs, sar_coefs, s, p, P)
    ma_full <- expand_seasonal_poly(ma_coefs, sma_coefs, s, q, Q)

    residuals <- compute_arma_residuals(x_work, ar_full, ma_full)
    sum(residuals^2, na.rm = TRUE)
  }

  # Initial parameter guess
  init_params <- numeric(p + q + P + Q)
  if (p > 0) {
    ar_init <- ar_yule_walker(x_work, p)
    init_params[1:p] <- ar_init$coefficients * 0.5
  }

  # Optimize
  if (length(init_params) > 0) {
    opt <- tryCatch(
      optim(init_params, css_objective, method = "BFGS",
            control = list(maxit = 500)),
      error = function(e) {
        optim(init_params, css_objective, method = "Nelder-Mead",
              control = list(maxit = 1000))
      }
    )
    params <- opt$par
  } else {
    params <- numeric(0)
    opt <- list(value = sum(x_work^2))
  }

  # Extract coefficients
  ar_coefs <- if (p > 0) params[1:p] else numeric(0)
  ma_coefs <- if (q > 0) params[(p + 1):(p + q)] else numeric(0)
  sar_coefs <- if (P > 0) params[(p + q + 1):(p + q + P)] else numeric(0)
  sma_coefs <- if (Q > 0) params[(p + q + P + 1):(p + q + P + Q)] else numeric(0)

  # Full coefficient expansion
  ar_full <- expand_seasonal_poly(ar_coefs, sar_coefs, s, p, P)
  ma_full <- expand_seasonal_poly(ma_coefs, sma_coefs, s, q, Q)

  # Compute residuals
  residuals <- compute_arma_residuals(x_work, ar_full, ma_full)

  # Sigma squared
  sigma2 <- sum(residuals^2, na.rm = TRUE) / (n - n_params)

  # Information criteria
  log_lik <- -n / 2 * (log(2 * pi) + log(sigma2) + 1)
  aic <- -2 * log_lik + 2 * (n_params + 1)
  bic <- -2 * log_lik + log(n) * (n_params + 1)
  aicc <- aic + 2 * (n_params + 1) * (n_params + 2) / (n - n_params - 2)

  structure(
    list(
      ar = ar_coefs,
      ma = ma_coefs,
      sar = sar_coefs,
      sma = sma_coefs,
      ar_full = ar_full,
      ma_full = ma_full,
      intercept = x_mean,
      sigma2 = sigma2,
      residuals = residuals,
      fitted = x_work - residuals,
      log_likelihood = log_lik,
      aic = aic,
      bic = bic,
      aicc = aicc,
      order = order,
      seasonal = seasonal,
      n = n,
      n_params = n_params,
      x = x,
      x_diff = x_diff,
      x_work = x_work,
      include.mean = include.mean
    ),
    class = "arima_model"
  )
}

#' Expand seasonal ARMA polynomial
#' @keywords internal
expand_seasonal_poly <- function(nonseasonal, seasonal, s, p, P) {
  # (1 - phi_1 B - ... - phi_p B^p)(1 - Phi_1 B^s - ... - Phi_P B^{Ps})
  max_order <- p + P * s

  if (max_order == 0) return(numeric(0))

  # Non-seasonal polynomial
  ns_poly <- c(1, if (p > 0) -nonseasonal else numeric(0))
  ns_poly <- c(ns_poly, rep(0, max_order + 1 - length(ns_poly)))

  # Seasonal polynomial
  s_poly <- rep(0, max_order + 1)
  s_poly[1] <- 1
  for (j in seq_len(P)) {
    s_poly[j * s + 1] <- -seasonal[j]
  }

  # Multiply polynomials
  result_poly <- rep(0, max_order + 1)
  for (i in seq_along(ns_poly)) {
    for (j in seq_along(s_poly)) {
      if (i + j - 2 <= max_order) {
        result_poly[i + j - 1] <- result_poly[i + j - 1] +
          ns_poly[i] * s_poly[j]
      }
    }
  }

  # Return coefficients (negative, excluding lag 0)
  -result_poly[2:(max_order + 1)]
}

#' Compute ARMA residuals given coefficients
#' @keywords internal
compute_arma_residuals <- function(x, ar_coefs, ma_coefs) {
  n <- length(x)
  p <- length(ar_coefs)
  q <- length(ma_coefs)

  residuals <- numeric(n)

  for (t in seq_len(n)) {
    ar_part <- 0
    if (p > 0) {
      for (j in seq_len(min(p, t - 1))) {
        ar_part <- ar_part + ar_coefs[j] * x[t - j]
      }
    }

    ma_part <- 0
    if (q > 0) {
      for (j in seq_len(min(q, t - 1))) {
        ma_part <- ma_part + ma_coefs[j] * residuals[t - j]
      }
    }

    residuals[t] <- x[t] - ar_part - ma_part
  }

  residuals
}

#' Auto ARIMA model selection
#'
#' Searches over ARIMA(p,d,q) models using AIC/BIC.
#'
#' @param x time series
#' @param max_p maximum AR order
#' @param max_d maximum differencing order
#' @param max_q maximum MA order
#' @param ic information criterion ("aic", "bic", "aicc")
#' @param stepwise logical, use stepwise search
#' @return best ARIMA model
#' @export
auto_arima <- function(x, max_p = 5, max_d = 2, max_q = 5,
                        ic = "aicc", stepwise = TRUE) {
  n <- length(x)

  # Determine d via unit root tests
  d <- 0
  x_test <- x
  for (dd in 0:max_d) {
    if (dd > 0) x_test <- diff(x_test)
    adf_result <- adf_test(x_test)
    if (adf_result$p_value < 0.05) {
      d <- dd
      break
    }
    d <- dd
  }

  x_diff <- ts_diff(x, d = d)

  best_ic <- Inf
  best_model <- NULL
  best_order <- NULL

  if (stepwise) {
    # Stepwise search starting from (0,d,0), (1,d,0), (0,d,1), (1,d,1)
    candidates <- list(c(0, d, 0), c(1, d, 0), c(0, d, 1), c(1, d, 1),
                        c(2, d, 0), c(0, d, 2))

    evaluated <- list()

    while (length(candidates) > 0) {
      current <- candidates[[1]]
      candidates <- candidates[-1]

      key <- paste(current, collapse = ",")
      if (key %in% evaluated) next
      evaluated <- c(evaluated, key)

      p_try <- current[1]; q_try <- current[3]
      if (p_try > max_p || q_try > max_q) next
      if (p_try < 0 || q_try < 0) next

      model <- tryCatch(
        arima_fit(x, order = current),
        error = function(e) NULL
      )

      if (is.null(model)) next

      ic_val <- switch(ic,
                        "aic" = model$aic,
                        "bic" = model$bic,
                        "aicc" = model$aicc)

      if (ic_val < best_ic) {
        best_ic <- ic_val
        best_model <- model
        best_order <- current

        # Add neighbors
        new_candidates <- list(
          c(p_try + 1, d, q_try),
          c(p_try - 1, d, q_try),
          c(p_try, d, q_try + 1),
          c(p_try, d, q_try - 1),
          c(p_try + 1, d, q_try + 1),
          c(p_try - 1, d, q_try - 1)
        )
        candidates <- c(candidates, new_candidates)
      }
    }
  } else {
    # Exhaustive search
    for (p_try in 0:max_p) {
      for (q_try in 0:max_q) {
        model <- tryCatch(
          arima_fit(x, order = c(p_try, d, q_try)),
          error = function(e) NULL
        )

        if (is.null(model)) next

        ic_val <- switch(ic,
                          "aic" = model$aic,
                          "bic" = model$bic,
                          "aicc" = model$aicc)

        if (ic_val < best_ic) {
          best_ic <- ic_val
          best_model <- model
          best_order <- c(p_try, d, q_try)
        }
      }
    }
  }

  best_model$best_ic <- best_ic
  best_model
}

#' ARIMA Forecasting
#'
#' @param model ARIMA model
#' @param h forecast horizon
#' @param level confidence level(s)
#' @return forecasts with confidence intervals
#' @export
arima_forecast <- function(model, h = 10, level = c(0.80, 0.95)) {
  n <- length(model$x_work)
  ar <- model$ar_full
  ma <- model$ma_full
  p <- length(ar)
  q <- length(ma)

  # Extend series with zeros for forecasts
  x_ext <- c(model$x_work, rep(0, h))
  resid_ext <- c(model$residuals, rep(0, h))

  forecasts <- numeric(h)

  for (t in seq_len(h)) {
    idx <- n + t
    ar_part <- 0
    if (p > 0) {
      for (j in seq_len(p)) {
        if (idx - j >= 1) {
          ar_part <- ar_part + ar[j] * x_ext[idx - j]
        }
      }
    }

    ma_part <- 0
    if (q > 0) {
      for (j in seq_len(q)) {
        if (idx - j >= 1 && idx - j <= n) {
          ma_part <- ma_part + ma[j] * resid_ext[idx - j]
        }
      }
    }

    forecasts[t] <- ar_part + ma_part + model$intercept
    x_ext[idx] <- forecasts[t]
  }

  # Compute MA(infinity) representation for prediction intervals
  psi <- compute_psi_weights(ar, ma, h)
  cumulative_var <- cumsum(c(1, psi^2)) * model$sigma2

  # Confidence intervals
  intervals <- lapply(level, function(lev) {
    z <- qnorm((1 + lev) / 2)
    se <- sqrt(cumulative_var[1:h])
    data.frame(
      lower = forecasts - z * se,
      upper = forecasts + z * se,
      level = lev
    )
  })

  structure(
    list(
      mean = forecasts,
      se = sqrt(cumulative_var[1:h]),
      intervals = intervals,
      level = level,
      h = h,
      model = model
    ),
    class = "arima_forecast"
  )
}

#' Compute psi weights (MA infinity representation)
#' @keywords internal
compute_psi_weights <- function(ar, ma, h) {
  p <- length(ar)
  q <- length(ma)
  psi <- numeric(h)

  for (j in seq_len(h)) {
    psi[j] <- if (j <= q) ma[j] else 0
    if (p > 0) {
      for (i in seq_len(min(p, j - 1))) {
        psi[j] <- psi[j] + ar[i] * if (j - i >= 1) {
          if (j - i == 0) 1 else psi[j - i]
        } else {
          0
        }
      }
    }
  }

  psi
}

#' ARIMA Residual Diagnostics
#'
#' @param model ARIMA model
#' @return diagnostic test results
#' @export
arima_diagnostics <- function(model) {
  resid <- model$residuals

  # Ljung-Box test
  lb <- ljung_box_test(resid, lag = 20)

  # Normality test (Jarque-Bera)
  jb <- jarque_bera_test(resid)

  # ARCH-LM test
  arch <- arch_lm_test(resid, lags = 5)

  # ACF of residuals
  resid_acf <- acf_compute(resid, max_lag = 20)

  list(
    ljung_box = lb,
    jarque_bera = jb,
    arch_lm = arch,
    residual_acf = resid_acf,
    mean_resid = mean(resid),
    sd_resid = sd(resid),
    skewness = moment_skewness(resid),
    kurtosis = moment_kurtosis(resid)
  )
}

#' Print method for ARIMA
#' @export
print.arima_model <- function(x, ...) {
  cat("ARIMA(", x$order[1], ",", x$order[2], ",", x$order[3], ")\n",
      sep = "")
  if (!is.null(x$seasonal)) {
    cat("Seasonal(", x$seasonal$order[1], ",", x$seasonal$order[2], ",",
        x$seasonal$order[3], ")[", x$seasonal$period, "]\n", sep = "")
  }
  cat("\nCoefficients:\n")
  if (length(x$ar) > 0) {
    cat("  AR:", round(x$ar, 4), "\n")
  }
  if (length(x$ma) > 0) {
    cat("  MA:", round(x$ma, 4), "\n")
  }
  if (x$include.mean) {
    cat("  Mean:", round(x$intercept, 4), "\n")
  }
  cat("\nSigma^2:", round(x$sigma2, 6), "\n")
  cat("Log-likelihood:", round(x$log_likelihood, 2), "\n")
  cat("AIC:", round(x$aic, 2), "  BIC:", round(x$bic, 2),
      "  AICc:", round(x$aicc, 2), "\n")
  invisible(x)
}

# =============================================================================
# SECTION 2: GARCH FAMILY
# =============================================================================

#' GARCH(1,1) Estimation via MLE
#'
#' @param x returns series
#' @param dist error distribution ("normal" or "t")
#' @return GARCH model
#' @export
garch_fit <- function(x, dist = "normal") {
  n <- length(x)
  x <- x - mean(x)

  # Log-likelihood function
  garch_loglik <- function(params) {
    omega <- params[1]
    alpha <- params[2]
    beta <- params[3]

    if (omega <= 0 || alpha < 0 || beta < 0 || alpha + beta >= 1) {
      return(1e10)
    }

    sigma2 <- numeric(n)
    sigma2[1] <- omega / (1 - alpha - beta)

    for (t in 2:n) {
      sigma2[t] <- omega + alpha * x[t - 1]^2 + beta * sigma2[t - 1]
      if (sigma2[t] <= 0) return(1e10)
    }

    if (dist == "normal") {
      ll <- -0.5 * sum(log(2 * pi) + log(sigma2) + x^2 / sigma2)
    } else {
      # Student-t: additional df parameter
      nu <- params[4]
      if (nu <= 2) return(1e10)
      ll <- sum(lgamma((nu + 1) / 2) - lgamma(nu / 2) -
                  0.5 * log((nu - 2) * pi * sigma2) -
                  (nu + 1) / 2 * log(1 + x^2 / ((nu - 2) * sigma2)))
    }

    -ll
  }

  # Initial parameters
  init_omega <- var(x) * 0.1
  init_alpha <- 0.1
  init_beta <- 0.8

  if (dist == "normal") {
    init <- c(init_omega, init_alpha, init_beta)
    lower <- c(1e-8, 1e-8, 1e-8)
    upper <- c(var(x), 0.5, 0.999)
  } else {
    init <- c(init_omega, init_alpha, init_beta, 5)
    lower <- c(1e-8, 1e-8, 1e-8, 2.1)
    upper <- c(var(x), 0.5, 0.999, 100)
  }

  opt <- tryCatch(
    optim(init, garch_loglik, method = "L-BFGS-B",
          lower = lower, upper = upper,
          control = list(maxit = 1000)),
    error = function(e) {
      optim(init, garch_loglik, method = "Nelder-Mead",
            control = list(maxit = 2000))
    }
  )

  omega <- opt$par[1]
  alpha <- opt$par[2]
  beta <- opt$par[3]
  nu <- if (dist == "t") opt$par[4] else NULL

  # Compute conditional variances
  sigma2 <- numeric(n)
  sigma2[1] <- omega / (1 - alpha - beta)
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * x[t - 1]^2 + beta * sigma2[t - 1]
  }

  # Standardized residuals
  std_resid <- x / sqrt(sigma2)

  # Unconditional variance
  uncond_var <- omega / (1 - alpha - beta)

  # Persistence
  persistence <- alpha + beta

  # Half-life of volatility shock
  half_life <- log(0.5) / log(persistence)

  # Information criteria
  n_params <- if (dist == "normal") 3 else 4
  log_lik <- -opt$value
  aic <- -2 * log_lik + 2 * n_params
  bic <- -2 * log_lik + log(n) * n_params

  structure(
    list(
      omega = omega,
      alpha = alpha,
      beta = beta,
      nu = nu,
      sigma2 = sigma2,
      sigma = sqrt(sigma2),
      std_residuals = std_resid,
      log_likelihood = log_lik,
      aic = aic,
      bic = bic,
      persistence = persistence,
      half_life = half_life,
      uncond_var = uncond_var,
      dist = dist,
      n = n,
      x = x
    ),
    class = "garch_model"
  )
}

#' EGARCH(1,1) Estimation
#'
#' log(sigma^2_t) = omega + alpha*g(z_{t-1}) + beta*log(sigma^2_{t-1})
#' where g(z) = theta*z + gamma*(|z| - E|z|)
#'
#' @param x returns series
#' @return EGARCH model
#' @export
egarch_fit <- function(x) {
  n <- length(x)
  x <- x - mean(x)

  egarch_loglik <- function(params) {
    omega <- params[1]
    alpha <- params[2]
    beta <- params[3]
    gamma <- params[4]

    if (abs(beta) >= 1) return(1e10)

    log_sigma2 <- numeric(n)
    log_sigma2[1] <- omega / (1 - beta)

    E_abs_z <- sqrt(2 / pi)

    for (t in 2:n) {
      z <- x[t - 1] / exp(log_sigma2[t - 1] / 2)
      log_sigma2[t] <- omega + alpha * z + gamma * (abs(z) - E_abs_z) +
        beta * log_sigma2[t - 1]
    }

    sigma2 <- exp(log_sigma2)
    ll <- -0.5 * sum(log(2 * pi) + log_sigma2 + x^2 / sigma2)
    -ll
  }

  init <- c(-0.1, -0.1, 0.95, 0.1)
  opt <- tryCatch(
    optim(init, egarch_loglik, method = "BFGS",
          control = list(maxit = 1000)),
    error = function(e) {
      optim(init, egarch_loglik, method = "Nelder-Mead",
            control = list(maxit = 2000))
    }
  )

  omega <- opt$par[1]
  alpha <- opt$par[2]
  beta <- opt$par[3]
  gamma <- opt$par[4]

  # Compute conditional variances
  log_sigma2 <- numeric(n)
  log_sigma2[1] <- omega / (1 - beta)
  E_abs_z <- sqrt(2 / pi)
  for (t in 2:n) {
    z <- x[t - 1] / exp(log_sigma2[t - 1] / 2)
    log_sigma2[t] <- omega + alpha * z + gamma * (abs(z) - E_abs_z) +
      beta * log_sigma2[t - 1]
  }
  sigma2 <- exp(log_sigma2)

  log_lik <- -opt$value
  aic <- -2 * log_lik + 2 * 4
  bic <- -2 * log_lik + log(n) * 4

  structure(
    list(
      omega = omega,
      alpha = alpha,
      beta = beta,
      gamma = gamma,
      sigma2 = sigma2,
      sigma = sqrt(sigma2),
      std_residuals = x / sqrt(sigma2),
      log_likelihood = log_lik,
      aic = aic,
      bic = bic,
      n = n,
      x = x
    ),
    class = "egarch_model"
  )
}

#' GJR-GARCH(1,1) Estimation
#'
#' sigma^2_t = omega + (alpha + gamma*I(x<0))*x^2_{t-1} + beta*sigma^2_{t-1}
#'
#' @param x returns series
#' @return GJR-GARCH model
#' @export
gjr_garch_fit <- function(x) {
  n <- length(x)
  x <- x - mean(x)

  gjr_loglik <- function(params) {
    omega <- params[1]
    alpha <- params[2]
    beta <- params[3]
    gamma <- params[4]

    if (omega <= 0 || alpha < 0 || beta < 0 || gamma < 0) return(1e10)
    if (alpha + beta + gamma / 2 >= 1) return(1e10)

    sigma2 <- numeric(n)
    sigma2[1] <- omega / (1 - alpha - beta - gamma / 2)

    for (t in 2:n) {
      indicator <- as.numeric(x[t - 1] < 0)
      sigma2[t] <- omega + (alpha + gamma * indicator) * x[t - 1]^2 +
        beta * sigma2[t - 1]
      if (sigma2[t] <= 0) return(1e10)
    }

    ll <- -0.5 * sum(log(2 * pi) + log(sigma2) + x^2 / sigma2)
    -ll
  }

  init <- c(var(x) * 0.05, 0.05, 0.85, 0.1)
  lower <- c(1e-8, 1e-8, 1e-8, 1e-8)
  upper <- c(var(x), 0.5, 0.999, 0.5)

  opt <- optim(init, gjr_loglik, method = "L-BFGS-B",
               lower = lower, upper = upper)

  omega <- opt$par[1]; alpha <- opt$par[2]
  beta <- opt$par[3]; gamma <- opt$par[4]

  sigma2 <- numeric(n)
  sigma2[1] <- omega / (1 - alpha - beta - gamma / 2)
  for (t in 2:n) {
    indicator <- as.numeric(x[t - 1] < 0)
    sigma2[t] <- omega + (alpha + gamma * indicator) * x[t - 1]^2 +
      beta * sigma2[t - 1]
  }

  log_lik <- -opt$value
  news_impact_pos <- function(z) omega + alpha * z^2 + beta * sigma2[n]
  news_impact_neg <- function(z) omega + (alpha + gamma) * z^2 + beta * sigma2[n]

  structure(
    list(
      omega = omega, alpha = alpha, beta = beta, gamma = gamma,
      sigma2 = sigma2, sigma = sqrt(sigma2),
      std_residuals = x / sqrt(sigma2),
      log_likelihood = log_lik,
      aic = -2 * log_lik + 8,
      bic = -2 * log_lik + 4 * log(n),
      leverage_effect = gamma,
      persistence = alpha + beta + gamma / 2,
      n = n, x = x
    ),
    class = "gjr_garch_model"
  )
}

#' APARCH(1,1) Estimation
#'
#' sigma^delta_t = omega + alpha*(|x_{t-1}| - gamma*x_{t-1})^delta + beta*sigma^delta_{t-1}
#'
#' @param x returns series
#' @return APARCH model
#' @export
aparch_fit <- function(x) {
  n <- length(x)
  x <- x - mean(x)

  aparch_loglik <- function(params) {
    omega <- params[1]
    alpha <- params[2]
    beta <- params[3]
    gamma <- params[4]
    delta <- params[5]

    if (omega <= 0 || alpha < 0 || beta < 0 || delta <= 0) return(1e10)
    if (abs(gamma) >= 1) return(1e10)

    sigma_d <- numeric(n)
    sigma_d[1] <- (omega / (1 - alpha - beta))
    if (sigma_d[1] <= 0) sigma_d[1] <- var(x)^(delta / 2)

    for (t in 2:n) {
      sigma_d[t] <- omega +
        alpha * (abs(x[t - 1]) - gamma * x[t - 1])^delta +
        beta * sigma_d[t - 1]
      if (sigma_d[t] <= 0) return(1e10)
    }

    sigma2 <- sigma_d^(2 / delta)
    ll <- -0.5 * sum(log(2 * pi) + log(sigma2) + x^2 / sigma2)
    -ll
  }

  init <- c(var(x) * 0.05, 0.1, 0.85, 0.1, 2.0)
  opt <- tryCatch(
    optim(init, aparch_loglik, method = "Nelder-Mead",
          control = list(maxit = 3000)),
    error = function(e) list(par = init, value = 1e10)
  )

  params <- opt$par
  log_lik <- -opt$value

  structure(
    list(
      omega = params[1], alpha = params[2], beta = params[3],
      gamma = params[4], delta = params[5],
      log_likelihood = log_lik,
      aic = -2 * log_lik + 10,
      bic = -2 * log_lik + 5 * log(n),
      n = n, x = x
    ),
    class = "aparch_model"
  )
}

#' GARCH Forecasting
#'
#' @param model GARCH-family model
#' @param h forecast horizon
#' @return variance forecasts
#' @export
garch_forecast <- function(model, h = 10) {
  n <- model$n
  sigma2_forecast <- numeric(h)

  if (inherits(model, "garch_model")) {
    omega <- model$omega
    alpha <- model$alpha
    beta <- model$beta
    last_sigma2 <- model$sigma2[n]
    last_x2 <- model$x[n]^2

    sigma2_forecast[1] <- omega + alpha * last_x2 + beta * last_sigma2

    uncond_var <- model$uncond_var
    for (t in 2:h) {
      sigma2_forecast[t] <- uncond_var +
        (alpha + beta)^(t - 1) * (sigma2_forecast[1] - uncond_var)
    }
  }

  list(
    sigma2 = sigma2_forecast,
    sigma = sqrt(sigma2_forecast),
    h = h
  )
}

# =============================================================================
# SECTION 3: EXPONENTIAL SMOOTHING
# =============================================================================

#' Simple Exponential Smoothing
#'
#' @param x time series
#' @param alpha smoothing parameter (NULL for optimized)
#' @param h forecast horizon
#' @return SES model and forecasts
#' @export
ses_model <- function(x, alpha = NULL, h = 10) {
  n <- length(x)

  ses_sse <- function(a) {
    level <- numeric(n + 1)
    level[1] <- x[1]
    for (t in seq_len(n)) {
      level[t + 1] <- a * x[t] + (1 - a) * level[t]
    }
    sum((x - level[1:n])^2)
  }

  if (is.null(alpha)) {
    opt <- optimize(ses_sse, interval = c(0.001, 0.999))
    alpha <- opt$minimum
  }

  level <- numeric(n + 1)
  level[1] <- x[1]
  for (t in seq_len(n)) {
    level[t + 1] <- alpha * x[t] + (1 - alpha) * level[t]
  }

  fitted <- level[1:n]
  residuals <- x - fitted

  # Forecast
  forecasts <- rep(level[n + 1], h)

  # Prediction intervals
  sigma2 <- sum(residuals^2) / (n - 1)
  se <- sqrt(sigma2 * (1 + (seq_len(h) - 1) * alpha^2))

  structure(
    list(
      alpha = alpha,
      level = level,
      fitted = fitted,
      residuals = residuals,
      forecasts = forecasts,
      se = se,
      sigma2 = sigma2,
      n = n, x = x
    ),
    class = "ses_model"
  )
}

#' Holt's Linear Trend Method
#'
#' @param x time series
#' @param alpha level smoothing
#' @param beta trend smoothing
#' @param damped logical, use damped trend
#' @param phi damping parameter
#' @param h forecast horizon
#' @return Holt model
#' @export
holt_model <- function(x, alpha = NULL, beta = NULL,
                        damped = FALSE, phi = 0.98, h = 10) {
  n <- length(x)

  holt_sse <- function(params) {
    a <- params[1]
    b <- params[2]
    p <- if (damped) params[3] else 1

    level <- numeric(n + 1)
    trend <- numeric(n + 1)
    level[1] <- x[1]
    trend[1] <- x[2] - x[1]

    sse <- 0
    for (t in seq_len(n)) {
      forecast_t <- level[t] + p * trend[t]
      sse <- sse + (x[t] - forecast_t)^2
      level[t + 1] <- a * x[t] + (1 - a) * (level[t] + p * trend[t])
      trend[t + 1] <- b * (level[t + 1] - level[t]) + (1 - b) * p * trend[t]
    }
    sse
  }

  if (is.null(alpha) || is.null(beta)) {
    if (damped) {
      init <- c(0.3, 0.1, 0.98)
      lower <- c(0.001, 0.001, 0.8)
      upper <- c(0.999, 0.999, 0.999)
    } else {
      init <- c(0.3, 0.1)
      lower <- c(0.001, 0.001)
      upper <- c(0.999, 0.999)
    }

    opt <- optim(init, holt_sse, method = "L-BFGS-B",
                 lower = lower, upper = upper)
    alpha <- opt$par[1]
    beta <- opt$par[2]
    if (damped) phi <- opt$par[3]
  }

  p <- if (damped) phi else 1

  level <- numeric(n + 1)
  trend <- numeric(n + 1)
  level[1] <- x[1]
  trend[1] <- x[2] - x[1]

  for (t in seq_len(n)) {
    level[t + 1] <- alpha * x[t] + (1 - alpha) * (level[t] + p * trend[t])
    trend[t + 1] <- beta * (level[t + 1] - level[t]) + (1 - beta) * p * trend[t]
  }

  fitted <- level[1:n] + p * trend[1:n]
  residuals <- x - fitted

  # Forecasts
  forecasts <- numeric(h)
  for (j in seq_len(h)) {
    if (damped) {
      phi_sum <- sum(phi^(1:j))
      forecasts[j] <- level[n + 1] + phi_sum * trend[n + 1]
    } else {
      forecasts[j] <- level[n + 1] + j * trend[n + 1]
    }
  }

  structure(
    list(
      alpha = alpha, beta = beta, phi = phi, damped = damped,
      level = level, trend = trend,
      fitted = fitted, residuals = residuals,
      forecasts = forecasts,
      n = n, x = x
    ),
    class = "holt_model"
  )
}

#' Holt-Winters Exponential Smoothing
#'
#' @param x time series
#' @param s seasonal period
#' @param type "additive" or "multiplicative"
#' @param alpha level parameter
#' @param beta trend parameter
#' @param gamma seasonal parameter
#' @param h forecast horizon
#' @return Holt-Winters model
#' @export
holt_winters <- function(x, s, type = "additive",
                          alpha = NULL, beta = NULL, gamma = NULL,
                          h = 2 * s) {
  n <- length(x)
  stopifnot(n >= 2 * s)

  # Initialize: decompose first two periods
  seasonal_init <- numeric(s)
  level_init <- mean(x[1:s])
  trend_init <- (mean(x[(s + 1):(2 * s)]) - mean(x[1:s])) / s

  if (type == "additive") {
    for (j in seq_len(s)) {
      seasonal_init[j] <- mean(x[seq(j, min(n, j + s), by = s)]) - level_init
    }
  } else {
    for (j in seq_len(s)) {
      seasonal_init[j] <- mean(x[seq(j, min(n, j + s), by = s)]) / level_init
    }
  }

  hw_sse <- function(params) {
    a <- params[1]; b <- params[2]; g <- params[3]

    level <- numeric(n + 1)
    trend <- numeric(n + 1)
    seasonal <- numeric(n + s)

    level[1] <- level_init
    trend[1] <- trend_init
    seasonal[1:s] <- seasonal_init

    sse <- 0
    for (t in seq_len(n)) {
      s_idx <- ((t - 1) %% s) + 1

      if (type == "additive") {
        forecast_t <- level[t] + trend[t] + seasonal[s_idx]
        level[t + 1] <- a * (x[t] - seasonal[s_idx]) +
          (1 - a) * (level[t] + trend[t])
        trend[t + 1] <- b * (level[t + 1] - level[t]) + (1 - b) * trend[t]
        seasonal[s_idx + s] <- g * (x[t] - level[t + 1]) +
          (1 - g) * seasonal[s_idx]
      } else {
        forecast_t <- (level[t] + trend[t]) * seasonal[s_idx]
        level[t + 1] <- a * (x[t] / seasonal[s_idx]) +
          (1 - a) * (level[t] + trend[t])
        trend[t + 1] <- b * (level[t + 1] - level[t]) + (1 - b) * trend[t]
        seasonal[s_idx + s] <- g * (x[t] / level[t + 1]) +
          (1 - g) * seasonal[s_idx]
      }

      sse <- sse + (x[t] - forecast_t)^2
    }
    sse
  }

  # Optimize if not provided
  if (is.null(alpha) || is.null(beta) || is.null(gamma)) {
    opt <- optim(c(0.3, 0.1, 0.1), hw_sse, method = "L-BFGS-B",
                 lower = c(0.001, 0.001, 0.001),
                 upper = c(0.999, 0.999, 0.999))
    alpha <- opt$par[1]; beta <- opt$par[2]; gamma <- opt$par[3]
  }

  # Compute states
  level <- numeric(n + 1)
  trend <- numeric(n + 1)
  seasonal <- numeric(n + s + h)

  level[1] <- level_init
  trend[1] <- trend_init
  seasonal[1:s] <- seasonal_init

  fitted_vals <- numeric(n)

  for (t in seq_len(n)) {
    s_idx <- ((t - 1) %% s) + 1

    if (type == "additive") {
      fitted_vals[t] <- level[t] + trend[t] + seasonal[s_idx]
      level[t + 1] <- alpha * (x[t] - seasonal[s_idx]) +
        (1 - alpha) * (level[t] + trend[t])
      trend[t + 1] <- beta * (level[t + 1] - level[t]) + (1 - beta) * trend[t]
      seasonal[s_idx + s] <- gamma * (x[t] - level[t + 1]) +
        (1 - gamma) * seasonal[s_idx]
    } else {
      fitted_vals[t] <- (level[t] + trend[t]) * seasonal[s_idx]
      level[t + 1] <- alpha * (x[t] / seasonal[s_idx]) +
        (1 - alpha) * (level[t] + trend[t])
      trend[t + 1] <- beta * (level[t + 1] - level[t]) + (1 - beta) * trend[t]
      seasonal[s_idx + s] <- gamma * (x[t] / level[t + 1]) +
        (1 - gamma) * seasonal[s_idx]
    }
  }

  residuals <- x - fitted_vals

  # Forecasts
  forecasts <- numeric(h)
  for (j in seq_len(h)) {
    s_idx <- ((n + j - 1) %% s) + 1
    seas_val <- seasonal[s_idx + s * floor((n + j - 1) / s)]
    if (is.na(seas_val)) seas_val <- seasonal[s_idx]

    if (type == "additive") {
      forecasts[j] <- level[n + 1] + j * trend[n + 1] + seas_val
    } else {
      forecasts[j] <- (level[n + 1] + j * trend[n + 1]) * seas_val
    }
  }

  structure(
    list(
      alpha = alpha, beta = beta, gamma = gamma,
      type = type, s = s,
      level = level, trend = trend, seasonal = seasonal,
      fitted = fitted_vals, residuals = residuals,
      forecasts = forecasts,
      n = n, x = x
    ),
    class = "hw_model"
  )
}

# =============================================================================
# SECTION 4: STRUCTURAL TIME SERIES (STATE SPACE / KALMAN FILTER)
# =============================================================================

#' Kalman Filter
#'
#' State space model: y_t = Z*alpha_t + epsilon_t (epsilon ~ N(0,H))
#'                    alpha_{t+1} = T*alpha_t + R*eta_t (eta ~ N(0,Q))
#'
#' @param y observations
#' @param Z observation matrix
#' @param H observation variance
#' @param T_mat transition matrix
#' @param R_mat state noise selection
#' @param Q state noise covariance
#' @param a0 initial state mean
#' @param P0 initial state covariance
#' @return filtered states and log-likelihood
#' @export
kalman_filter <- function(y, Z, H, T_mat, R_mat, Q, a0, P0) {
  n <- length(y)
  m <- length(a0)

  # Storage
  a_pred <- matrix(0, nrow = n + 1, ncol = m)
  P_pred <- array(0, dim = c(m, m, n + 1))
  a_filt <- matrix(0, nrow = n, ncol = m)
  P_filt <- array(0, dim = c(m, m, n))
  v <- numeric(n)  # Innovation
  F_var <- numeric(n)  # Innovation variance
  K <- matrix(0, nrow = n, ncol = m)  # Kalman gain

  a_pred[1, ] <- a0
  P_pred[, , 1] <- P0

  log_lik <- 0

  for (t in seq_len(n)) {
    # Prediction error
    v[t] <- y[t] - sum(Z * a_pred[t, ])

    # Innovation variance
    F_var[t] <- sum(Z * (P_pred[, , t] %*% Z)) + H

    if (F_var[t] <= 0) F_var[t] <- 1e-10

    # Kalman gain
    K[t, ] <- (P_pred[, , t] %*% Z) / F_var[t]

    # Update
    a_filt[t, ] <- a_pred[t, ] + K[t, ] * v[t]
    P_filt[, , t] <- P_pred[, , t] - F_var[t] * outer(K[t, ], K[t, ])

    # Prediction
    a_pred[t + 1, ] <- T_mat %*% a_filt[t, ]
    P_pred[, , t + 1] <- T_mat %*% P_filt[, , t] %*% t(T_mat) +
      R_mat %*% Q %*% t(R_mat)

    # Log-likelihood contribution
    log_lik <- log_lik - 0.5 * (log(2 * pi) + log(F_var[t]) +
                                    v[t]^2 / F_var[t])
  }

  list(
    a_pred = a_pred,
    P_pred = P_pred,
    a_filt = a_filt,
    P_filt = P_filt,
    v = v,
    F_var = F_var,
    K = K,
    log_likelihood = log_lik
  )
}

#' Kalman Smoother (Rauch-Tung-Striebel)
#'
#' @param kf_result output from kalman_filter
#' @param T_mat transition matrix
#' @return smoothed states
#' @export
kalman_smoother <- function(kf_result, T_mat) {
  n <- nrow(kf_result$a_filt)
  m <- ncol(kf_result$a_filt)

  a_smooth <- matrix(0, nrow = n, ncol = m)
  P_smooth <- array(0, dim = c(m, m, n))

  a_smooth[n, ] <- kf_result$a_filt[n, ]
  P_smooth[, , n] <- kf_result$P_filt[, , n]

  for (t in (n - 1):1) {
    L <- kf_result$P_filt[, , t] %*% t(T_mat) %*%
      solve(kf_result$P_pred[, , t + 1])

    a_smooth[t, ] <- kf_result$a_filt[t, ] +
      L %*% (a_smooth[t + 1, ] - kf_result$a_pred[t + 1, ])
    P_smooth[, , t] <- kf_result$P_filt[, , t] +
      L %*% (P_smooth[, , t + 1] - kf_result$P_pred[, , t + 1]) %*% t(L)
  }

  list(a_smooth = a_smooth, P_smooth = P_smooth)
}

#' Local Level Model (Random Walk + Noise)
#'
#' y_t = mu_t + epsilon_t, epsilon ~ N(0, sigma_e^2)
#' mu_{t+1} = mu_t + eta_t, eta ~ N(0, sigma_eta^2)
#'
#' @param y observations
#' @return structural model
#' @export
local_level_model <- function(y) {
  n <- length(y)

  ll_fn <- function(params) {
    sigma_e <- exp(params[1])
    sigma_eta <- exp(params[2])

    Z <- matrix(1, 1, 1)
    H <- sigma_e^2
    T_mat <- matrix(1, 1, 1)
    R_mat <- matrix(1, 1, 1)
    Q <- matrix(sigma_eta^2, 1, 1)
    a0 <- y[1]
    P0 <- matrix(10 * var(y), 1, 1)

    kf <- kalman_filter(y, Z, H, T_mat, R_mat, Q, a0, P0)
    -kf$log_likelihood
  }

  init <- log(c(sd(y) * 0.5, sd(y) * 0.5))
  opt <- optim(init, ll_fn, method = "BFGS")

  sigma_e <- exp(opt$par[1])
  sigma_eta <- exp(opt$par[2])

  Z <- matrix(1, 1, 1)
  T_mat <- matrix(1, 1, 1)
  R_mat <- matrix(1, 1, 1)
  Q <- matrix(sigma_eta^2, 1, 1)

  kf <- kalman_filter(y, Z, sigma_e^2, T_mat, R_mat, Q, y[1],
                       matrix(10 * var(y), 1, 1))
  ks <- kalman_smoother(kf, T_mat)

  structure(
    list(
      sigma_e = sigma_e, sigma_eta = sigma_eta,
      filtered = kf$a_filt[, 1],
      smoothed = ks$a_smooth[, 1],
      log_likelihood = -opt$value,
      signal_to_noise = sigma_eta^2 / sigma_e^2,
      kf = kf, ks = ks,
      n = n, y = y
    ),
    class = "local_level"
  )
}

#' Local Linear Trend Model
#'
#' @param y observations
#' @return structural model with level and trend
#' @export
local_trend_model <- function(y) {
  n <- length(y)

  ll_fn <- function(params) {
    sigma_e <- exp(params[1])
    sigma_level <- exp(params[2])
    sigma_trend <- exp(params[3])

    Z <- c(1, 0)
    H <- sigma_e^2
    T_mat <- matrix(c(1, 0, 1, 1), 2, 2)
    R_mat <- diag(2)
    Q <- diag(c(sigma_level^2, sigma_trend^2))
    a0 <- c(y[1], 0)
    P0 <- diag(c(10 * var(y), var(y)))

    kf <- kalman_filter(y, Z, H, T_mat, R_mat, Q, a0, P0)
    -kf$log_likelihood
  }

  init <- log(c(sd(y), sd(y) * 0.5, sd(y) * 0.1))
  opt <- optim(init, ll_fn, method = "BFGS")

  sigma_e <- exp(opt$par[1])
  sigma_level <- exp(opt$par[2])
  sigma_trend <- exp(opt$par[3])

  Z <- c(1, 0)
  T_mat <- matrix(c(1, 0, 1, 1), 2, 2)
  R_mat <- diag(2)
  Q <- diag(c(sigma_level^2, sigma_trend^2))

  kf <- kalman_filter(y, Z, sigma_e^2, T_mat, R_mat, Q,
                       c(y[1], 0), diag(c(10 * var(y), var(y))))
  ks <- kalman_smoother(kf, T_mat)

  structure(
    list(
      sigma_e = sigma_e, sigma_level = sigma_level,
      sigma_trend = sigma_trend,
      level = ks$a_smooth[, 1],
      trend = ks$a_smooth[, 2],
      log_likelihood = -opt$value,
      n = n, y = y
    ),
    class = "local_trend"
  )
}

# =============================================================================
# SECTION 5: VAR (Vector Autoregression)
# =============================================================================

#' VAR Model Estimation
#'
#' @param Y multivariate time series matrix (n x k)
#' @param p lag order
#' @param include.const logical, include intercept
#' @return VAR model
#' @export
var_estimate <- function(Y, p = 1, include.const = TRUE) {
  if (!is.matrix(Y)) Y <- as.matrix(Y)
  n <- nrow(Y)
  k <- ncol(Y)

  # Construct lagged matrix
  n_eff <- n - p
  Y_dep <- Y[(p + 1):n, , drop = FALSE]

  Z <- matrix(0, nrow = n_eff, ncol = k * p + as.integer(include.const))
  for (lag in seq_len(p)) {
    cols <- ((lag - 1) * k + 1):(lag * k)
    Z[, cols] <- Y[(p + 1 - lag):(n - lag), ]
  }
  if (include.const) {
    Z[, ncol(Z)] <- 1
  }

  # OLS estimation equation by equation
  B <- solve(crossprod(Z), crossprod(Z, Y_dep))

  # Residuals
  residuals <- Y_dep - Z %*% B

  # Covariance matrix
  Sigma <- crossprod(residuals) / (n_eff - ncol(Z))

  # Information criteria
  log_det <- determinant(Sigma, logarithm = TRUE)$modulus[1]
  n_params <- k * (k * p + as.integer(include.const))
  aic <- log_det + 2 * n_params / n_eff
  bic <- log_det + log(n_eff) * n_params / n_eff

  structure(
    list(
      B = B,
      Sigma = Sigma,
      residuals = residuals,
      fitted = Y_dep - residuals,
      Y = Y,
      p = p,
      k = k,
      n = n,
      n_eff = n_eff,
      include.const = include.const,
      aic = aic,
      bic = bic
    ),
    class = "var_model"
  )
}

#' VAR Lag Order Selection
#'
#' @param Y multivariate time series
#' @param max_p maximum lag order
#' @param ic information criterion
#' @return optimal lag order
#' @export
var_select <- function(Y, max_p = 10, ic = "aic") {
  ic_values <- numeric(max_p)
  for (p in seq_len(max_p)) {
    model <- var_estimate(Y, p = p)
    ic_values[p] <- switch(ic, "aic" = model$aic, "bic" = model$bic)
  }
  list(optimal_p = which.min(ic_values), ic_values = ic_values)
}

#' Granger Causality Test
#'
#' Tests if variable j Granger-causes variable i.
#'
#' @param Y multivariate time series
#' @param cause column index of cause variable
#' @param effect column index of effect variable
#' @param p lag order
#' @return test result
#' @export
granger_causality <- function(Y, cause, effect, p = 1) {
  k <- ncol(Y)

  # Unrestricted model
  var_full <- var_estimate(Y, p = p)
  rss_full <- sum(var_full$residuals[, effect]^2)

  # Restricted model (exclude cause variable lags)
  Y_restricted <- Y[, -cause, drop = FALSE]
  var_restricted <- var_estimate(Y_restricted, p = p)

  # Need to match the effect variable in restricted model
  effect_restricted <- if (cause < effect) effect - 1 else effect
  rss_restricted <- sum(var_restricted$residuals[, effect_restricted]^2)

  # F-test
  n_eff <- var_full$n_eff
  q <- p  # Number of restrictions
  df1 <- q
  df2 <- n_eff - k * p - 1

  f_stat <- ((rss_restricted - rss_full) / df1) / (rss_full / df2)
  p_value <- pf(f_stat, df1, df2, lower.tail = FALSE)

  list(
    f_stat = f_stat,
    df1 = df1,
    df2 = df2,
    p_value = p_value,
    cause = cause,
    effect = effect,
    reject = p_value < 0.05
  )
}

#' VAR Impulse Response Function
#'
#' @param model VAR model
#' @param n_ahead number of periods ahead
#' @param ortho logical, orthogonalized IRF (Cholesky)
#' @return impulse response matrices
#' @export
var_irf <- function(model, n_ahead = 20, ortho = TRUE) {
  k <- model$k
  p <- model$p
  B <- model$B

  # Extract coefficient matrices A_1, ..., A_p
  A <- vector("list", p)
  for (lag in seq_len(p)) {
    rows <- ((lag - 1) * k + 1):(lag * k)
    A[[lag]] <- t(B[rows, ])
  }

  # Companion form
  Phi <- array(0, dim = c(k, k, n_ahead + 1))
  Phi[, , 1] <- diag(k)

  for (h in seq_len(n_ahead)) {
    for (j in seq_len(min(h, p))) {
      Phi[, , h + 1] <- Phi[, , h + 1] + A[[j]] %*% Phi[, , h - j + 1]
    }
  }

  # Orthogonalize using Cholesky
  if (ortho) {
    P <- t(chol(model$Sigma))
    for (h in seq_len(n_ahead + 1)) {
      Phi[, , h] <- Phi[, , h] %*% P
    }
  }

  structure(
    list(
      irf = Phi,
      n_ahead = n_ahead,
      ortho = ortho,
      k = k
    ),
    class = "var_irf"
  )
}

#' Forecast Error Variance Decomposition
#'
#' @param model VAR model
#' @param n_ahead horizon
#' @return FEVD matrices
#' @export
var_fevd <- function(model, n_ahead = 20) {
  irf_result <- var_irf(model, n_ahead = n_ahead, ortho = TRUE)
  Phi <- irf_result$irf
  k <- model$k

  # Cumulative squared IRF
  fevd <- array(0, dim = c(k, k, n_ahead + 1))

  cumsum_sq <- array(0, dim = c(k, k))

  for (h in seq_len(n_ahead + 1)) {
    cumsum_sq <- cumsum_sq + Phi[, , h]^2

    total_var <- rowSums(cumsum_sq)
    for (i in seq_len(k)) {
      if (total_var[i] > 0) {
        fevd[i, , h] <- cumsum_sq[i, ] / total_var[i]
      }
    }
  }

  structure(
    list(fevd = fevd, n_ahead = n_ahead, k = k),
    class = "var_fevd"
  )
}

#' VAR Forecasting
#'
#' @param model VAR model
#' @param h forecast horizon
#' @return forecasts
#' @export
var_forecast <- function(model, h = 10) {
  k <- model$k
  p <- model$p
  n <- model$n
  B <- model$B

  forecasts <- matrix(0, nrow = h, ncol = k)
  Y_ext <- rbind(model$Y, matrix(0, nrow = h, ncol = k))

  for (t in seq_len(h)) {
    idx <- n + t
    z <- numeric(k * p + as.integer(model$include.const))

    for (lag in seq_len(p)) {
      cols <- ((lag - 1) * k + 1):(lag * k)
      z[cols] <- Y_ext[idx - lag, ]
    }
    if (model$include.const) z[length(z)] <- 1

    Y_ext[idx, ] <- z %*% B
    forecasts[t, ] <- Y_ext[idx, ]
  }

  list(forecasts = forecasts, h = h)
}

# =============================================================================
# SECTION 6: VECM / COINTEGRATION
# =============================================================================

#' Johansen Cointegration Test
#'
#' @param Y multivariate time series
#' @param p lag order (in levels)
#' @param type "trace" or "max_eigen"
#' @return test results
#' @export
johansen_test <- function(Y, p = 1, type = "trace") {
  if (!is.matrix(Y)) Y <- as.matrix(Y)
  n <- nrow(Y)
  k <- ncol(Y)

  # First differences
  dY <- diff(Y)
  Y_lag <- Y[1:(n - 1), ]

  # Lagged differences
  n_eff <- nrow(dY) - p + 1
  if (n_eff < 1) stop("Not enough observations")

  dY_dep <- dY[p:nrow(dY), ]
  Y_lag_dep <- Y_lag[p:nrow(Y_lag), ]

  # Lagged differences
  Z <- NULL
  if (p > 1) {
    for (lag in 1:(p - 1)) {
      Z <- cbind(Z, dY[(p - lag):(nrow(dY) - lag), ])
    }
  }

  # Add constant
  Z <- cbind(Z, 1)

  # Regress dY and Y_{t-1} on Z to get residuals
  if (!is.null(Z)) {
    QZ <- qr(Z)
    R0 <- qr.resid(QZ, dY_dep)
    R1 <- qr.resid(QZ, Y_lag_dep)
  } else {
    R0 <- dY_dep
    R1 <- Y_lag_dep
  }

  # Moment matrices
  S00 <- crossprod(R0) / n_eff
  S01 <- crossprod(R0, R1) / n_eff
  S10 <- crossprod(R1, R0) / n_eff
  S11 <- crossprod(R1) / n_eff

  # Solve generalized eigenvalue problem
  S11_inv <- solve(S11)
  M <- S11_inv %*% S10 %*% solve(S00) %*% S01

  eig <- eigen(M)
  eigenvalues <- Re(eig$values)
  eigenvectors <- Re(eig$vectors)

  # Order by eigenvalue magnitude
  ord <- order(eigenvalues, decreasing = TRUE)
  eigenvalues <- eigenvalues[ord]
  eigenvectors <- eigenvectors[, ord]

  # Test statistics
  trace_stats <- numeric(k)
  max_eigen_stats <- numeric(k)

  for (r in 0:(k - 1)) {
    # Trace statistic
    trace_stats[r + 1] <- -n_eff * sum(log(1 - eigenvalues[(r + 1):k]))
    # Max eigenvalue statistic
    max_eigen_stats[r + 1] <- -n_eff * log(1 - eigenvalues[r + 1])
  }

  # Critical values (approximate, for k=2,3,4)
  # These are rough approximations
  trace_cv_5 <- c(15.41, 29.68, 47.21, 68.52)[1:k]
  max_cv_5 <- c(14.07, 20.97, 27.07, 33.46)[1:k]

  list(
    eigenvalues = eigenvalues,
    eigenvectors = eigenvectors,
    trace_stats = trace_stats,
    max_eigen_stats = max_eigen_stats,
    trace_cv = trace_cv_5,
    max_cv = max_cv_5,
    n_eff = n_eff,
    k = k,
    cointegrating_rank = sum(trace_stats > trace_cv_5)
  )
}

#' VECM Estimation
#'
#' @param Y multivariate time series
#' @param r cointegrating rank
#' @param p lag order (in levels)
#' @return VECM model
#' @export
vecm_estimate <- function(Y, r = 1, p = 2) {
  if (!is.matrix(Y)) Y <- as.matrix(Y)
  n <- nrow(Y)
  k <- ncol(Y)

  # Johansen to get cointegrating vectors
  joh <- johansen_test(Y, p = p)
  beta <- joh$eigenvectors[, 1:r, drop = FALSE]

  # Error correction terms
  ect <- Y %*% beta

  # Construct VECM regression
  dY <- diff(Y)
  n_eff <- nrow(dY) - p + 1

  dY_dep <- dY[p:nrow(dY), ]
  ect_lag <- ect[p:(n - 1), , drop = FALSE]

  Z <- ect_lag
  if (p > 1) {
    for (lag in 1:(p - 1)) {
      Z <- cbind(Z, dY[(p - lag):(nrow(dY) - lag), ])
    }
  }
  Z <- cbind(Z, 1)

  B <- solve(crossprod(Z), crossprod(Z, dY_dep))
  residuals <- dY_dep - Z %*% B

  alpha <- B[1:r, , drop = FALSE]

  Sigma <- crossprod(residuals) / n_eff

  structure(
    list(
      alpha = t(alpha),
      beta = beta,
      Pi = t(alpha) %*% t(beta),
      B = B,
      Sigma = Sigma,
      residuals = residuals,
      r = r,
      p = p,
      k = k,
      n = n,
      Y = Y
    ),
    class = "vecm_model"
  )
}

# =============================================================================
# SECTION 7: REGIME SWITCHING
# =============================================================================

#' Markov-Switching AR Model (2-state)
#'
#' Hamilton filter for regime identification.
#'
#' @param y time series
#' @param p AR order within each regime
#' @return MS-AR model
#' @export
ms_ar_fit <- function(y, p = 1) {
  n <- length(y)
  n_eff <- n - p

  # Hamilton filter
  ms_loglik <- function(params) {
    mu1 <- params[1]; mu2 <- params[2]
    sigma1 <- exp(params[3]); sigma2 <- exp(params[4])
    phi <- if (p > 0) params[5:(4 + p)] else numeric(0)
    p11 <- 1 / (1 + exp(-params[5 + p]))  # logistic transform
    p22 <- 1 / (1 + exp(-params[6 + p]))

    P <- matrix(c(p11, 1 - p22, 1 - p11, p22), 2, 2)

    # Filtered probabilities
    xi_filt <- matrix(0, nrow = n_eff, ncol = 2)
    xi_pred <- matrix(0, nrow = n_eff, ncol = 2)

    # Ergodic probabilities for initialization
    ergodic <- c((1 - p22) / (2 - p11 - p22),
                  (1 - p11) / (2 - p11 - p22))
    xi_pred[1, ] <- ergodic

    log_lik <- 0

    for (t in seq_len(n_eff)) {
      t_orig <- t + p

      # AR component
      ar_part <- 0
      for (j in seq_len(p)) {
        ar_part <- ar_part + phi[j] * y[t_orig - j]
      }

      # Likelihood in each regime
      dens1 <- dnorm(y[t_orig] - ar_part, mean = mu1, sd = sigma1)
      dens2 <- dnorm(y[t_orig] - ar_part, mean = mu2, sd = sigma2)

      # Joint density
      joint <- xi_pred[t, ] * c(dens1, dens2)
      marginal <- sum(joint)
      if (marginal < 1e-300) marginal <- 1e-300

      log_lik <- log_lik + log(marginal)

      # Update
      xi_filt[t, ] <- joint / marginal

      # Predict
      if (t < n_eff) {
        xi_pred[t + 1, ] <- as.numeric(P %*% xi_filt[t, ])
      }
    }

    -log_lik
  }

  # Initialize
  n_params <- 4 + p + 2  # mu1, mu2, log_s1, log_s2, phi, logit_p11, logit_p22
  init <- c(quantile(y, 0.25), quantile(y, 0.75),
            log(sd(y)), log(sd(y)),
            rep(0.1, p), 2, 2)

  opt <- tryCatch(
    optim(init, ms_loglik, method = "BFGS",
          control = list(maxit = 1000)),
    error = function(e) {
      optim(init, ms_loglik, method = "Nelder-Mead",
            control = list(maxit = 3000))
    }
  )

  params <- opt$par
  mu1 <- params[1]; mu2 <- params[2]
  sigma1 <- exp(params[3]); sigma2 <- exp(params[4])
  phi <- if (p > 0) params[5:(4 + p)] else numeric(0)
  p11 <- 1 / (1 + exp(-params[5 + p]))
  p22 <- 1 / (1 + exp(-params[6 + p]))

  # Rerun filter to get probabilities
  P_mat <- matrix(c(p11, 1 - p22, 1 - p11, p22), 2, 2)
  ergodic <- c((1 - p22) / (2 - p11 - p22), (1 - p11) / (2 - p11 - p22))

  xi_filt <- matrix(0, nrow = n_eff, ncol = 2)
  xi_pred <- matrix(0, nrow = n_eff, ncol = 2)
  xi_pred[1, ] <- ergodic

  for (t in seq_len(n_eff)) {
    t_orig <- t + p
    ar_part <- sum(phi * y[(t_orig - 1):(t_orig - p)])
    dens <- c(dnorm(y[t_orig] - ar_part, mu1, sigma1),
              dnorm(y[t_orig] - ar_part, mu2, sigma2))
    joint <- xi_pred[t, ] * dens
    xi_filt[t, ] <- joint / sum(joint)
    if (t < n_eff) xi_pred[t + 1, ] <- as.numeric(P_mat %*% xi_filt[t, ])
  }

  # Kim smoother
  xi_smooth <- matrix(0, nrow = n_eff, ncol = 2)
  xi_smooth[n_eff, ] <- xi_filt[n_eff, ]
  for (t in (n_eff - 1):1) {
    for (j in 1:2) {
      xi_smooth[t, j] <- xi_filt[t, j] *
        sum(P_mat[j, ] * xi_smooth[t + 1, ] / xi_pred[t + 1, ])
    }
  }

  structure(
    list(
      mu = c(mu1, mu2),
      sigma = c(sigma1, sigma2),
      phi = phi,
      P = P_mat,
      p11 = p11, p22 = p22,
      filtered_probs = xi_filt,
      smoothed_probs = xi_smooth,
      predicted_probs = xi_pred,
      regime = apply(xi_smooth, 1, which.max),
      log_likelihood = -opt$value,
      aic = 2 * opt$value + 2 * n_params,
      bic = 2 * opt$value + log(n_eff) * n_params,
      n = n, p = p
    ),
    class = "ms_ar_model"
  )
}

# =============================================================================
# SECTION 8: THRESHOLD MODELS
# =============================================================================

#' Self-Exciting Threshold AR (SETAR)
#'
#' @param y time series
#' @param p AR order
#' @param d delay parameter
#' @param threshold threshold value (NULL for grid search)
#' @return SETAR model
#' @export
setar_fit <- function(y, p = 1, d = 1, threshold = NULL) {
  n <- length(y)
  n_eff <- n - max(p, d)

  # Construct design matrix
  start <- max(p, d) + 1
  y_dep <- y[start:n]
  X <- matrix(0, nrow = n_eff, ncol = p)
  for (j in seq_len(p)) {
    X[, j] <- y[(start - j):(n - j)]
  }
  threshold_var <- y[(start - d):(n - d)]

  if (is.null(threshold)) {
    # Grid search over percentiles
    candidates <- quantile(threshold_var, probs = seq(0.15, 0.85, by = 0.01))
    best_sse <- Inf
    best_thresh <- NULL

    for (thresh in candidates) {
      low_idx <- which(threshold_var <= thresh)
      high_idx <- which(threshold_var > thresh)

      if (length(low_idx) < p + 2 || length(high_idx) < p + 2) next

      # Fit separate models
      X_low <- cbind(1, X[low_idx, ])
      X_high <- cbind(1, X[high_idx, ])
      beta_low <- qr.coef(qr(X_low), y_dep[low_idx])
      beta_high <- qr.coef(qr(X_high), y_dep[high_idx])

      sse <- sum((y_dep[low_idx] - X_low %*% beta_low)^2) +
        sum((y_dep[high_idx] - X_high %*% beta_high)^2)

      if (sse < best_sse) {
        best_sse <- sse
        best_thresh <- thresh
      }
    }
    threshold <- best_thresh
  }

  # Final estimation
  low_idx <- which(threshold_var <= threshold)
  high_idx <- which(threshold_var > threshold)

  X_low <- cbind(1, X[low_idx, ])
  X_high <- cbind(1, X[high_idx, ])

  beta_low <- qr.coef(qr(X_low), y_dep[low_idx])
  beta_high <- qr.coef(qr(X_high), y_dep[high_idx])

  resid_low <- y_dep[low_idx] - X_low %*% beta_low
  resid_high <- y_dep[high_idx] - X_high %*% beta_high

  sigma_low <- sqrt(sum(resid_low^2) / length(low_idx))
  sigma_high <- sqrt(sum(resid_high^2) / length(high_idx))

  residuals <- numeric(n_eff)
  residuals[low_idx] <- resid_low
  residuals[high_idx] <- resid_high

  fitted <- y_dep - residuals

  aic <- n_eff * log(sum(residuals^2) / n_eff) + 2 * (2 * (p + 1) + 1)

  structure(
    list(
      beta_low = beta_low,
      beta_high = beta_high,
      threshold = threshold,
      sigma_low = sigma_low,
      sigma_high = sigma_high,
      residuals = residuals,
      fitted = fitted,
      regime = ifelse(threshold_var <= threshold, 1, 2),
      n_low = length(low_idx),
      n_high = length(high_idx),
      aic = aic,
      p = p, d = d, n = n
    ),
    class = "setar_model"
  )
}

# =============================================================================
# SECTION 9: SPECTRAL ANALYSIS
# =============================================================================

#' Periodogram
#'
#' @param x time series
#' @param detrend logical, remove mean
#' @return periodogram ordinates and frequencies
#' @export
periodogram <- function(x, detrend = TRUE) {
  n <- length(x)
  if (detrend) x <- x - mean(x)

  # FFT
  fft_vals <- fft(x)
  n_freq <- floor(n / 2)
  freqs <- (1:n_freq) / n

  # Periodogram = |FFT|^2 / n
  I <- Mod(fft_vals[2:(n_freq + 1)])^2 / n

  list(
    frequency = freqs,
    spectrum = I,
    n = n
  )
}

#' Smoothed Periodogram (Daniell kernel)
#'
#' @param x time series
#' @param spans vector of span widths for modified Daniell kernel
#' @return smoothed spectrum
#' @export
smoothed_periodogram <- function(x, spans = c(3, 3)) {
  peri <- periodogram(x)
  spec <- peri$spectrum

  # Apply modified Daniell kernel smoothing
  for (span in spans) {
    half <- floor(span / 2)
    kernel <- rep(1, span)
    kernel[1] <- 0.5
    kernel[span] <- 0.5
    kernel <- kernel / sum(kernel)

    n_spec <- length(spec)
    smoothed <- numeric(n_spec)
    for (i in seq_len(n_spec)) {
      indices <- (i - half):(i + half)
      indices <- pmax(1, pmin(n_spec, indices))
      smoothed[i] <- sum(kernel * spec[indices])
    }
    spec <- smoothed
  }

  list(
    frequency = peri$frequency,
    spectrum = spec,
    spans = spans,
    n = peri$n
  )
}

#' AR Spectral Density
#'
#' @param x time series
#' @param p AR order (NULL for AIC selection)
#' @param n_freq number of frequency points
#' @return AR spectrum
#' @export
ar_spectrum <- function(x, p = NULL, n_freq = 500) {
  n <- length(x)

  if (is.null(p)) {
    # AIC selection
    aic_vals <- sapply(1:min(20, n - 1), function(pp) {
      ar_yule_walker(x, pp)$aic
    })
    p <- which.min(aic_vals)
  }

  ar_model <- ar_yule_walker(x, p)
  phi <- ar_model$coefficients
  sigma2 <- ar_model$sigma2

  freqs <- seq(0, 0.5, length.out = n_freq)
  spec <- numeric(n_freq)

  for (i in seq_along(freqs)) {
    f <- freqs[i]
    denom <- 1
    for (j in seq_len(p)) {
      denom <- denom - phi[j] * exp(-2i * pi * f * j)
    }
    spec[i] <- sigma2 / (2 * pi * Mod(denom)^2)
  }

  list(
    frequency = freqs,
    spectrum = spec,
    p = p,
    ar_coef = phi,
    sigma2 = sigma2
  )
}

# =============================================================================
# SECTION 10: WAVELET ANALYSIS
# =============================================================================

#' Haar Wavelet DWT
#'
#' @param x time series (length must be power of 2)
#' @param level decomposition level (NULL for max)
#' @return wavelet coefficients
#' @export
dwt_haar <- function(x, level = NULL) {
  n <- length(x)

  # Pad to power of 2 if needed
  n_padded <- 2^ceiling(log2(n))
  if (n_padded > n) {
    x <- c(x, rep(0, n_padded - n))
  }

  if (is.null(level)) level <- log2(n_padded)

  details <- vector("list", level)
  approx <- x

  for (j in seq_len(level)) {
    m <- length(approx)
    if (m < 2) break

    half <- m / 2
    new_approx <- numeric(half)
    detail <- numeric(half)

    for (i in seq_len(half)) {
      new_approx[i] <- (approx[2 * i - 1] + approx[2 * i]) / sqrt(2)
      detail[i] <- (approx[2 * i - 1] - approx[2 * i]) / sqrt(2)
    }

    details[[j]] <- detail
    approx <- new_approx
  }

  list(
    details = details,
    approx = approx,
    level = level,
    n = n
  )
}

#' MODWT (Maximal Overlap DWT) with Haar filter
#'
#' @param x time series
#' @param level decomposition level
#' @return MODWT coefficients
#' @export
modwt_haar <- function(x, level = NULL) {
  n <- length(x)
  if (is.null(level)) level <- floor(log2(n))

  details <- vector("list", level)
  approx_coeffs <- vector("list", level)

  current <- x

  for (j in seq_len(level)) {
    scale <- 2^(j - 1)
    detail <- numeric(n)
    smooth <- numeric(n)

    h <- 1 / (2^(j / 2) * sqrt(2))
    g <- h

    for (t in seq_len(n)) {
      idx1 <- ((t - 1) %% n) + 1
      idx2 <- ((t - 1 - scale) %% n) + 1
      smooth[t] <- h * (current[idx1] + current[idx2])
      detail[t] <- g * (current[idx1] - current[idx2])
    }

    details[[j]] <- detail
    approx_coeffs[[j]] <- smooth
    current <- smooth
  }

  list(
    details = details,
    approx = approx_coeffs,
    level = level,
    n = n
  )
}

#' Wavelet Variance
#'
#' @param x time series
#' @param level decomposition level
#' @return wavelet variance by scale
#' @export
wavelet_variance <- function(x, level = NULL) {
  modwt <- modwt_haar(x, level)

  wvar <- sapply(modwt$details, function(d) mean(d^2, na.rm = TRUE))
  scales <- 2^(seq_along(wvar))

  list(
    variance = wvar,
    scales = scales,
    level = modwt$level
  )
}

# =============================================================================
# SECTION 11: LONG MEMORY
# =============================================================================

#' Hurst Exponent via R/S Analysis
#'
#' @param x time series
#' @return Hurst exponent estimate
#' @export
hurst_rs <- function(x) {
  n <- length(x)

  # R/S for different block sizes
  block_sizes <- unique(floor(exp(seq(log(10), log(n / 2),
                                       length.out = 30))))
  block_sizes <- block_sizes[block_sizes >= 10]

  log_rs <- numeric(length(block_sizes))
  log_n <- numeric(length(block_sizes))

  for (i in seq_along(block_sizes)) {
    bs <- block_sizes[i]
    n_blocks <- floor(n / bs)
    rs_vals <- numeric(n_blocks)

    for (b in seq_len(n_blocks)) {
      block <- x[((b - 1) * bs + 1):(b * bs)]
      block_mean <- mean(block)
      cumdev <- cumsum(block - block_mean)
      R <- max(cumdev) - min(cumdev)
      S <- sd(block)
      if (S > 0) {
        rs_vals[b] <- R / S
      } else {
        rs_vals[b] <- 0
      }
    }

    log_rs[i] <- log(mean(rs_vals[rs_vals > 0]))
    log_n[i] <- log(bs)
  }

  valid <- is.finite(log_rs)
  fit <- lm(log_rs[valid] ~ log_n[valid])
  H <- coef(fit)[2]

  list(
    H = H,
    log_rs = log_rs,
    log_n = log_n,
    block_sizes = block_sizes
  )
}

#' GPH Estimator for Fractional Differencing Parameter
#'
#' Geweke and Porter-Hudak semiparametric estimator.
#'
#' @param x time series
#' @param bandwidth_fraction fraction of frequencies to use
#' @return fractional differencing parameter d
#' @export
gph_estimator <- function(x, bandwidth_fraction = 0.5) {
  n <- length(x)
  peri <- periodogram(x)

  m <- floor(n^bandwidth_fraction)
  freqs <- peri$frequency[1:m]
  I <- peri$spectrum[1:m]

  # Regression: log(I) ~ a + d * log(4*sin^2(pi*f))
  X_gph <- -log(4 * sin(pi * freqs)^2)
  y_gph <- log(I)

  valid <- is.finite(y_gph) & is.finite(X_gph)
  fit <- lm(y_gph[valid] ~ X_gph[valid])

  d <- coef(fit)[2]
  se_d <- summary(fit)$coefficients[2, 2]

  list(
    d = d,
    se = se_d,
    H = d + 0.5,
    bandwidth = m
  )
}

#' ARFIMA(p,d,q) Estimation
#'
#' @param x time series
#' @param p AR order
#' @param q MA order
#' @return ARFIMA model
#' @export
arfima_fit <- function(x, p = 0, q = 0) {
  n <- length(x)
  x <- x - mean(x)

  # Estimate d via GPH
  gph <- gph_estimator(x)
  d_init <- gph$d

  # Fractional differencing via binomial expansion
  frac_diff <- function(x, d, max_terms = 100) {
    n <- length(x)
    result <- numeric(n)

    # Compute weights
    weights <- numeric(max_terms + 1)
    weights[1] <- 1
    for (k in seq_len(max_terms)) {
      weights[k + 1] <- weights[k] * (d - k + 1) / k
    }

    for (t in seq_len(n)) {
      max_j <- min(t - 1, max_terms)
      result[t] <- sum(weights[1:(max_j + 1)] * x[t:(t - max_j)])
    }
    result
  }

  # Optimize d and ARMA parameters jointly
  arfima_loglik <- function(params) {
    d <- params[1]
    ar_coefs <- if (p > 0) params[2:(1 + p)] else numeric(0)
    ma_coefs <- if (q > 0) params[(2 + p):(1 + p + q)] else numeric(0)

    if (abs(d) >= 0.5) return(1e10)

    x_fd <- frac_diff(x, d)
    resid <- compute_arma_residuals(x_fd, ar_coefs, ma_coefs)
    sigma2 <- mean(resid^2)

    if (sigma2 <= 0) return(1e10)
    n * log(sigma2)
  }

  init <- c(d_init, rep(0, p + q))
  opt <- optim(init, arfima_loglik, method = "Nelder-Mead",
               control = list(maxit = 2000))

  d <- opt$par[1]
  ar_coefs <- if (p > 0) opt$par[2:(1 + p)] else numeric(0)
  ma_coefs <- if (q > 0) opt$par[(2 + p):(1 + p + q)] else numeric(0)

  x_fd <- frac_diff(x, d)
  residuals <- compute_arma_residuals(x_fd, ar_coefs, ma_coefs)
  sigma2 <- mean(residuals^2)

  n_params <- 1 + p + q + 1
  log_lik <- -n / 2 * (log(2 * pi * sigma2) + 1)
  aic <- -2 * log_lik + 2 * n_params
  bic <- -2 * log_lik + log(n) * n_params

  structure(
    list(
      d = d,
      ar = ar_coefs,
      ma = ma_coefs,
      sigma2 = sigma2,
      residuals = residuals,
      log_likelihood = log_lik,
      aic = aic,
      bic = bic,
      H = d + 0.5,
      n = n
    ),
    class = "arfima_model"
  )
}

# =============================================================================
# SECTION 12: STATISTICAL TESTS
# =============================================================================

#' Augmented Dickey-Fuller Test
#'
#' @param x time series
#' @param max_lag maximum lag order
#' @param type "none", "drift", "trend"
#' @return ADF test result
#' @export
adf_test <- function(x, max_lag = NULL, type = "drift") {
  n <- length(x)
  if (is.null(max_lag)) max_lag <- floor((n - 1)^(1 / 3))

  dx <- diff(x)
  x_lag <- x[-n]

  # Construct regression
  n_eff <- length(dx) - max_lag

  y <- dx[(max_lag + 1):length(dx)]
  X <- x_lag[(max_lag + 1):length(x_lag)]

  # Add lagged differences
  if (max_lag > 0) {
    for (lag in seq_len(max_lag)) {
      X <- cbind(X, dx[(max_lag + 1 - lag):(length(dx) - lag)])
    }
  }

  if (type == "drift") {
    X <- cbind(X, 1)
  } else if (type == "trend") {
    X <- cbind(X, 1, seq_len(n_eff))
  }

  # OLS
  qr_X <- qr(X)
  beta <- qr.coef(qr_X, y)
  resid <- y - X %*% beta
  se <- sqrt(sum(resid^2) / (n_eff - ncol(X))) /
    sqrt(sum(qr.resid(qr(X[, -1, drop = FALSE]), X[, 1])^2))

  t_stat <- beta[1] / se

  # Approximate p-value (MacKinnon)
  # Using rough interpolation for type="drift"
  if (type == "drift") {
    if (t_stat < -3.96) {
      p_value <- 0.001
    } else if (t_stat < -3.41) {
      p_value <- 0.01
    } else if (t_stat < -2.86) {
      p_value <- 0.05
    } else if (t_stat < -2.57) {
      p_value <- 0.10
    } else {
      p_value <- min(1, 0.10 + (t_stat + 2.57) * 0.3)
    }
  } else {
    p_value <- 2 * pt(abs(t_stat), df = n_eff - ncol(X), lower.tail = FALSE)
  }

  list(
    statistic = t_stat,
    p_value = p_value,
    type = type,
    lags = max_lag,
    n = n_eff
  )
}

#' KPSS Test
#'
#' @param x time series
#' @param type "level" or "trend"
#' @param lags number of lags for HAC bandwidth
#' @return KPSS test result
#' @export
kpss_test <- function(x, type = "level", lags = NULL) {
  n <- length(x)
  if (is.null(lags)) lags <- floor(4 * (n / 100)^0.25)

  # Detrend
  if (type == "level") {
    resid <- x - mean(x)
  } else {
    t_idx <- seq_len(n)
    fit <- lm(x ~ t_idx)
    resid <- residuals(fit)
  }

  # Partial sums
  S <- cumsum(resid)

  # Long-run variance (Bartlett kernel)
  gamma0 <- sum(resid^2) / n
  lrv <- gamma0
  for (j in seq_len(lags)) {
    w <- 1 - j / (lags + 1)
    gamma_j <- sum(resid[1:(n - j)] * resid[(j + 1):n]) / n
    lrv <- lrv + 2 * w * gamma_j
  }

  # Test statistic
  stat <- sum(S^2) / (n^2 * lrv)

  # Critical values
  if (type == "level") {
    cv <- c("10%" = 0.347, "5%" = 0.463, "1%" = 0.739)
  } else {
    cv <- c("10%" = 0.119, "5%" = 0.146, "1%" = 0.216)
  }

  p_value <- if (stat > cv["1%"]) 0.01 else if (stat > cv["5%"]) 0.05 else
    if (stat > cv["10%"]) 0.10 else 0.20

  list(
    statistic = stat,
    p_value = p_value,
    critical_values = cv,
    type = type,
    lags = lags
  )
}

#' Phillips-Perron Test
#'
#' @param x time series
#' @param type "drift" or "trend"
#' @param lags truncation lag
#' @return PP test result
#' @export
pp_test <- function(x, type = "drift", lags = NULL) {
  n <- length(x)
  if (is.null(lags)) lags <- floor(4 * (n / 100)^0.25)

  dx <- diff(x)
  x_lag <- x[-n]
  n_eff <- length(dx)

  if (type == "drift") {
    X <- cbind(x_lag, 1)
  } else {
    X <- cbind(x_lag, 1, seq_len(n_eff))
  }

  # OLS
  beta <- qr.coef(qr(X), dx)
  resid <- dx - X %*% beta
  sigma2 <- sum(resid^2) / n_eff

  # Long-run variance
  lrv <- sigma2
  for (j in seq_len(lags)) {
    w <- 1 - j / (lags + 1)
    gamma_j <- sum(resid[1:(n_eff - j)] * resid[(j + 1):n_eff]) / n_eff
    lrv <- lrv + 2 * w * gamma_j
  }

  # PP adjustment
  se_rho <- sqrt(sigma2 / sum((x_lag - mean(x_lag))^2))
  t_stat_ols <- beta[1] / se_rho

  correction <- (lrv - sigma2) / (2 * sqrt(lrv))
  pp_stat <- t_stat_ols * sqrt(sigma2 / lrv) -
    correction * n_eff * se_rho / sqrt(lrv)

  # Approximate p-value
  p_value <- if (pp_stat < -3.96) 0.001 else if (pp_stat < -3.41) 0.01 else
    if (pp_stat < -2.86) 0.05 else if (pp_stat < -2.57) 0.10 else 0.50

  list(
    statistic = pp_stat,
    p_value = p_value,
    type = type,
    lags = lags
  )
}

#' Ljung-Box Test
#'
#' @param x time series or residuals
#' @param lag number of lags
#' @param fitdf degrees of freedom used in fitting
#' @return Ljung-Box test result
#' @export
ljung_box_test <- function(x, lag = 20, fitdf = 0) {
  n <- length(x)
  acf_vals <- acf_compute(x, max_lag = lag)[-1]

  Q <- n * (n + 2) * sum(acf_vals^2 / (n - seq_len(lag)))
  df <- lag - fitdf
  p_value <- pchisq(Q, df = df, lower.tail = FALSE)

  list(
    statistic = Q,
    df = df,
    p_value = p_value,
    lag = lag
  )
}

#' ARCH-LM Test
#'
#' @param x residuals
#' @param lags number of ARCH lags
#' @return ARCH-LM test result
#' @export
arch_lm_test <- function(x, lags = 5) {
  n <- length(x)
  x2 <- x^2

  y <- x2[(lags + 1):n]
  X <- matrix(0, nrow = length(y), ncol = lags + 1)
  X[, 1] <- 1
  for (j in seq_len(lags)) {
    X[, j + 1] <- x2[(lags + 1 - j):(n - j)]
  }

  fit <- qr.coef(qr(X), y)
  resid <- y - X %*% fit
  tss <- sum((y - mean(y))^2)
  rss <- sum(resid^2)
  r2 <- 1 - rss / tss

  stat <- length(y) * r2
  p_value <- pchisq(stat, df = lags, lower.tail = FALSE)

  list(
    statistic = stat,
    df = lags,
    p_value = p_value,
    r_squared = r2
  )
}

#' Jarque-Bera Normality Test
#'
#' @param x numeric vector
#' @return JB test result
#' @export
jarque_bera_test <- function(x) {
  n <- length(x)
  s <- moment_skewness(x)
  k <- moment_kurtosis(x)

  jb <- n / 6 * (s^2 + (k - 3)^2 / 4)
  p_value <- pchisq(jb, df = 2, lower.tail = FALSE)

  list(
    statistic = jb,
    p_value = p_value,
    skewness = s,
    kurtosis = k
  )
}

#' BDS Independence Test
#'
#' @param x time series
#' @param m embedding dimension
#' @param eps epsilon (distance threshold)
#' @return BDS test result
#' @export
bds_test <- function(x, m = 3, eps = NULL) {
  n <- length(x)
  if (is.null(eps)) eps <- sd(x) * 0.75

  # Correlation integral
  C_m <- function(x, m, eps) {
    n <- length(x) - m + 1
    count <- 0
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        close <- TRUE
        for (k in 0:(m - 1)) {
          if (abs(x[i + k] - x[j + k]) > eps) {
            close <- FALSE
            break
          }
        }
        if (close) count <- count + 1
      }
    }
    2 * count / (n * (n - 1))
  }

  C1 <- C_m(x, 1, eps)
  Cm <- C_m(x, m, eps)

  # BDS statistic
  n_eff <- n - m + 1
  sigma_bds <- sqrt(4 * (C1^(2 * m - 2) * (1 - C1^2)^2 +
                            2 * (m - 1) * C1^(2 * m) * (1 - C1)^2))
  # Simplified standard error

  bds_stat <- sqrt(n_eff) * (Cm - C1^m) / (sigma_bds + 1e-10)
  p_value <- 2 * pnorm(abs(bds_stat), lower.tail = FALSE)

  list(
    statistic = bds_stat,
    p_value = p_value,
    C1 = C1,
    Cm = Cm,
    m = m,
    eps = eps
  )
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Skewness
#' @keywords internal
moment_skewness <- function(x) {
  n <- length(x)
  m <- mean(x)
  s <- sd(x)
  sum((x - m)^3) / (n * s^3)
}

#' Kurtosis
#' @keywords internal
moment_kurtosis <- function(x) {
  n <- length(x)
  m <- mean(x)
  s <- sd(x)
  sum((x - m)^4) / (n * s^4)
}

# =============================================================================
# SECTION 13: FORECASTING UTILITIES
# =============================================================================

#' Multi-step Forecast with Confidence Intervals
#'
#' @param model any time series model
#' @param h forecast horizon
#' @param level confidence levels
#' @param method forecast method
#' @return forecast object
#' @export
ts_forecast <- function(model, h = 10, level = c(0.80, 0.95),
                         method = NULL) {
  if (inherits(model, "arima_model")) {
    return(arima_forecast(model, h = h, level = level))
  }

  if (inherits(model, "garch_model")) {
    return(garch_forecast(model, h = h))
  }

  if (inherits(model, "ses_model")) {
    se <- model$se[1:h]
    intervals <- lapply(level, function(lev) {
      z <- qnorm((1 + lev) / 2)
      data.frame(
        lower = model$forecasts[1:h] - z * se,
        upper = model$forecasts[1:h] + z * se,
        level = lev
      )
    })
    return(list(mean = model$forecasts[1:h], se = se,
                intervals = intervals))
  }

  if (inherits(model, "holt_model") || inherits(model, "hw_model")) {
    sigma <- sd(model$residuals)
    se <- sigma * sqrt(seq_len(h))
    intervals <- lapply(level, function(lev) {
      z <- qnorm((1 + lev) / 2)
      data.frame(
        lower = model$forecasts[1:h] - z * se,
        upper = model$forecasts[1:h] + z * se,
        level = lev
      )
    })
    return(list(mean = model$forecasts[1:h], se = se,
                intervals = intervals))
  }

  if (inherits(model, "var_model")) {
    return(var_forecast(model, h = h))
  }

  stop("Unsupported model class")
}

#' Forecast Combination
#'
#' Combine forecasts from multiple models.
#'
#' @param forecasts list of forecast vectors
#' @param weights combination weights (NULL for equal, "inverse_mse" for
#'   inverse MSE weighting)
#' @param actuals actual values for weight computation
#' @return combined forecast
#' @export
forecast_combine <- function(forecasts, weights = NULL, actuals = NULL) {
  n_models <- length(forecasts)
  h <- length(forecasts[[1]])

  if (is.null(weights)) {
    if (!is.null(actuals) && length(actuals) > 0) {
      # Inverse MSE weighting
      mse_vals <- sapply(forecasts, function(f) {
        n <- min(length(f), length(actuals))
        mean((f[1:n] - actuals[1:n])^2)
      })
      weights <- (1 / mse_vals) / sum(1 / mse_vals)
    } else {
      weights <- rep(1 / n_models, n_models)
    }
  }

  combined <- rep(0, h)
  for (m in seq_len(n_models)) {
    combined <- combined + weights[m] * forecasts[[m]]
  }

  list(
    combined = combined,
    weights = weights,
    individual = forecasts
  )
}

#' Forecast Accuracy Metrics
#'
#' @param actual actual values
#' @param forecast forecast values
#' @return named vector of accuracy metrics
#' @export
forecast_accuracy <- function(actual, forecast) {
  n <- min(length(actual), length(forecast))
  actual <- actual[1:n]
  forecast <- forecast[1:n]
  errors <- actual - forecast

  c(
    ME = mean(errors),
    RMSE = sqrt(mean(errors^2)),
    MAE = mean(abs(errors)),
    MPE = mean(errors / actual) * 100,
    MAPE = mean(abs(errors / actual)) * 100,
    MASE = mean(abs(errors)) / mean(abs(diff(actual))),
    SMAPE = mean(2 * abs(errors) / (abs(actual) + abs(forecast))) * 100
  )
}

#' Diebold-Mariano Test
#'
#' @param e1 forecast errors from model 1
#' @param e2 forecast errors from model 2
#' @param h forecast horizon
#' @param power loss function power (1=MAE, 2=MSE)
#' @return DM test result
#' @export
dm_test <- function(e1, e2, h = 1, power = 2) {
  d <- abs(e1)^power - abs(e2)^power
  n <- length(d)
  d_bar <- mean(d)

  # HAC variance
  gamma0 <- var(d)
  lrv <- gamma0
  if (h > 1) {
    for (j in 1:(h - 1)) {
      gamma_j <- cov(d[1:(n - j)], d[(j + 1):n])
      lrv <- lrv + 2 * gamma_j
    }
  }

  se <- sqrt(lrv / n)
  dm_stat <- d_bar / se
  p_value <- 2 * pnorm(abs(dm_stat), lower.tail = FALSE)

  list(
    statistic = dm_stat,
    p_value = p_value,
    d_bar = d_bar,
    conclusion = if (p_value < 0.05) {
      if (d_bar < 0) "Model 1 significantly better" else
        "Model 2 significantly better"
    } else {
      "No significant difference"
    }
  )
}

#' Fan Chart Data Generation
#'
#' @param forecast_obj forecast object from ts_forecast
#' @param levels confidence levels
#' @return data for fan chart plotting
#' @export
fan_chart_data <- function(forecast_obj, levels = seq(0.1, 0.99, by = 0.1)) {
  h <- length(forecast_obj$mean)
  se <- forecast_obj$se

  result <- data.frame(
    horizon = rep(seq_len(h), length(levels) * 2),
    level = rep(rep(levels, each = h), 2),
    bound = c(rep("lower", h * length(levels)),
              rep("upper", h * length(levels))),
    value = NA
  )

  idx <- 1
  for (lev in levels) {
    z <- qnorm((1 + lev) / 2)
    for (t in seq_len(h)) {
      result$value[idx] <- forecast_obj$mean[t] - z * se[t]
      result$value[idx + h * length(levels)] <- forecast_obj$mean[t] + z * se[t]
      idx <- idx + 1
    }
  }

  result
}

###############################################################################
# END OF FILE: time_series_models.R
###############################################################################
