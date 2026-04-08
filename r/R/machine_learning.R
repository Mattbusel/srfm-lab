###############################################################################
# machine_learning.R
# Comprehensive Machine Learning Library in R
# Implements core ML algorithms from scratch with minimal dependencies
###############################################################################

# =============================================================================
# SECTION 1: LINEAR REGRESSION
# =============================================================================

#' Ordinary Least Squares Linear Regression
#'
#' Fits a linear model using the normal equations with QR decomposition
#' for numerical stability.
#'
#' @param X numeric matrix of predictors (n x p)
#' @param y numeric response vector (length n)
#' @param intercept logical, whether to add intercept column
#' @return list with coefficients, fitted values, residuals, diagnostics
#' @export
ols_regression <- function(X, y, intercept = TRUE) {
  stopifnot(is.numeric(X), is.numeric(y))
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)
  stopifnot(length(y) == n)

  if (intercept) {
    X <- cbind(1, X)
    p <- p + 1
  }

  # QR decomposition for numerical stability

  qr_decomp <- qr(X)
  if (qr_decomp$rank < p) {
    warning("Design matrix is rank deficient. Rank = ", qr_decomp$rank,
            ", expected = ", p)
  }

  beta <- qr.coef(qr_decomp, y)
  fitted <- as.numeric(X %*% beta)
  residuals <- y - fitted

  # Degrees of freedom

df_model <- p - as.integer(intercept)
  df_residual <- n - p

  # Residual standard error
  rss <- sum(residuals^2)
  tss <- sum((y - mean(y))^2)
  sigma2 <- rss / df_residual
  sigma <- sqrt(sigma2)

  # R-squared and adjusted R-squared
  r_squared <- 1 - rss / tss
  adj_r_squared <- 1 - (1 - r_squared) * (n - 1) / df_residual

  # Standard errors of coefficients
  XtX_inv <- chol2inv(qr.R(qr_decomp))
  se_beta <- sqrt(diag(XtX_inv) * sigma2)

  # t-statistics and p-values
  t_stats <- beta / se_beta
  p_values <- 2 * pt(abs(t_stats), df = df_residual, lower.tail = FALSE)

  # F-statistic
  if (df_model > 0) {
    f_stat <- ((tss - rss) / df_model) / (rss / df_residual)
    f_pvalue <- pf(f_stat, df_model, df_residual, lower.tail = FALSE)
  } else {
    f_stat <- NA
    f_pvalue <- NA
  }

  # Hat matrix diagonal for leverage
  hat_diag <- rowSums(qr.Q(qr_decomp)^2)

  # Cook's distance
  standardized_resid <- residuals / (sigma * sqrt(1 - hat_diag))
  cooks_d <- (standardized_resid^2 * hat_diag) / (p * (1 - hat_diag))

  # VIF (Variance Inflation Factor)
  vif <- NULL
  if (p > 1 + as.integer(intercept)) {
    start_col <- 1 + as.integer(intercept)
    vif <- numeric(ncol(X) - as.integer(intercept))
    Xpred <- X[, (start_col + 1):ncol(X), drop = FALSE]
    for (j in seq_len(ncol(Xpred))) {
      others <- Xpred[, -j, drop = FALSE]
      if (intercept) others <- cbind(1, others)
      r2_j <- 1 - sum(qr.resid(qr(others), Xpred[, j])^2) /
        sum((Xpred[, j] - mean(Xpred[, j]))^2)
      vif[j] <- 1 / (1 - r2_j)
    }
  }

  # Durbin-Watson statistic
  dw <- sum(diff(residuals)^2) / rss

  # AIC and BIC
  log_lik <- -n / 2 * (log(2 * pi) + log(rss / n) + 1)
  aic <- -2 * log_lik + 2 * p
  bic <- -2 * log_lik + log(n) * p

  structure(
    list(
      coefficients = beta,
      fitted.values = fitted,
      residuals = residuals,
      se = se_beta,
      t_stats = t_stats,
      p_values = p_values,
      sigma = sigma,
      r_squared = r_squared,
      adj_r_squared = adj_r_squared,
      f_stat = f_stat,
      f_pvalue = f_pvalue,
      hat_diag = hat_diag,
      cooks_d = cooks_d,
      vif = vif,
      dw_stat = dw,
      aic = aic,
      bic = bic,
      df_model = df_model,
      df_residual = df_residual,
      n = n,
      p = p,
      intercept = intercept,
      X = X,
      y = y
    ),
    class = "ols_model"
  )
}

#' Print method for OLS model
#' @export
print.ols_model <- function(x, ...) {
  cat("OLS Linear Regression\n")
  cat("=====================\n")
  cat("n =", x$n, ", p =", x$p, "\n")
  cat("\nCoefficients:\n")
  coef_table <- data.frame(
    Estimate = x$coefficients,
    Std.Error = x$se,
    t.value = x$t_stats,
    p.value = x$p_values
  )
  print(round(coef_table, 6))
  cat("\nResidual standard error:", round(x$sigma, 4),
      "on", x$df_residual, "degrees of freedom\n")
  cat("R-squared:", round(x$r_squared, 4),
      "  Adjusted R-squared:", round(x$adj_r_squared, 4), "\n")
  cat("F-statistic:", round(x$f_stat, 4), "on", x$df_model, "and",
      x$df_residual, "DF, p-value:", format.pval(x$f_pvalue), "\n")
  cat("AIC:", round(x$aic, 2), "  BIC:", round(x$bic, 2), "\n")
  cat("Durbin-Watson:", round(x$dw_stat, 4), "\n")
  invisible(x)
}

#' Predict method for OLS model
#' @export
predict.ols_model <- function(object, newdata = NULL, interval = "none",
                               level = 0.95, ...) {
  if (is.null(newdata)) {
    return(object$fitted.values)
  }
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  if (object$intercept) newdata <- cbind(1, newdata)
  pred <- as.numeric(newdata %*% object$coefficients)

  if (interval == "confidence" || interval == "prediction") {
    XtX_inv <- chol2inv(qr.R(qr(object$X)))
    alpha <- 1 - level
    t_crit <- qt(1 - alpha / 2, df = object$df_residual)

    se_fit <- sqrt(rowSums((newdata %*% XtX_inv) * newdata)) * object$sigma

    if (interval == "confidence") {
      margin <- t_crit * se_fit
    } else {
      se_pred <- sqrt(se_fit^2 + object$sigma^2)
      margin <- t_crit * se_pred
    }

    return(data.frame(
      fit = pred,
      lwr = pred - margin,
      upr = pred + margin
    ))
  }

  pred
}

# =============================================================================
# Ridge Regression with GCV
# =============================================================================

#' Ridge Regression with Generalized Cross-Validation
#'
#' Solves (X'X + lambda*I)beta = X'y using SVD.
#' Optionally selects lambda via GCV.
#'
#' @param X numeric matrix of predictors
#' @param y numeric response vector
#' @param lambda regularization parameter (NULL for GCV selection)
#' @param lambda_seq sequence of lambda values to search over
#' @param intercept logical, whether to center/add intercept
#' @return list with coefficients, fitted values, lambda, GCV scores
#' @export
ridge_regression <- function(X, y, lambda = NULL, lambda_seq = NULL,
                              intercept = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)
  stopifnot(length(y) == n)

  # Center data if intercept
  if (intercept) {
    y_mean <- mean(y)
    X_means <- colMeans(X)
    X_centered <- scale(X, center = TRUE, scale = FALSE)
    y_centered <- y - y_mean
  } else {
    X_centered <- X
    y_centered <- y
    y_mean <- 0
    X_means <- rep(0, p)
  }

  # SVD of centered X
  svd_X <- svd(X_centered)
  U <- svd_X$u
  d <- svd_X$d
  V <- svd_X$v

  # GCV function
  gcv_score <- function(lam) {
    d2 <- d^2
    df_lam <- sum(d2 / (d2 + lam))
    # Coefficients via SVD
    beta_lam <- V %*% (d / (d^2 + lam) * crossprod(U, y_centered))
    resid_lam <- y_centered - X_centered %*% beta_lam
    rss_lam <- sum(resid_lam^2)
    n * rss_lam / (n - df_lam)^2
  }

  # Select lambda via GCV if not provided
  if (is.null(lambda)) {
    if (is.null(lambda_seq)) {
      lambda_seq <- 10^seq(-6, 6, length.out = 200)
    }
    gcv_scores <- sapply(lambda_seq, gcv_score)
    lambda <- lambda_seq[which.min(gcv_scores)]
  } else {
    gcv_scores <- NULL
  }

  # Compute ridge coefficients
  d2 <- d^2
  beta <- V %*% (d / (d2 + lambda) * crossprod(U, y_centered))
  beta <- as.numeric(beta)

  # Intercept
  if (intercept) {
    beta0 <- y_mean - sum(X_means * beta)
  } else {
    beta0 <- 0
  }

  fitted <- as.numeric(X_centered %*% beta) + y_mean
  residuals <- y - fitted

  # Effective degrees of freedom
  df_eff <- sum(d2 / (d2 + lambda))

  # GCV at selected lambda
  rss <- sum(residuals^2)
  gcv_opt <- n * rss / (n - df_eff)^2

  structure(
    list(
      coefficients = beta,
      intercept = beta0,
      fitted.values = fitted,
      residuals = residuals,
      lambda = lambda,
      df_effective = df_eff,
      gcv = gcv_opt,
      gcv_scores = gcv_scores,
      lambda_seq = lambda_seq,
      rss = rss,
      n = n,
      p = p
    ),
    class = "ridge_model"
  )
}

#' Print method for Ridge model
#' @export
print.ridge_model <- function(x, ...) {
  cat("Ridge Regression\n")
  cat("================\n")
  cat("Lambda:", x$lambda, "\n")
  cat("Effective df:", round(x$df_effective, 2), "\n")
  cat("GCV:", round(x$gcv, 4), "\n")
  cat("RSS:", round(x$rss, 4), "\n")
  cat("\nCoefficients (first 20):\n")
  print(head(x$coefficients, 20))
  invisible(x)
}

# =============================================================================
# LASSO via Coordinate Descent
# =============================================================================

#' Soft-thresholding operator
#' @keywords internal
soft_threshold <- function(z, gamma) {
  sign(z) * pmax(abs(z) - gamma, 0)
}

#' LASSO Regression via Coordinate Descent
#'
#' Implements the LASSO (L1 regularization) using cyclic coordinate descent.
#' Supports warm starts and computes the full regularization path.
#'
#' @param X numeric matrix of predictors
#' @param y numeric response vector
#' @param lambda regularization parameter (or vector for path)
#' @param max_iter maximum iterations
#' @param tol convergence tolerance
#' @param intercept logical
#' @param standardize logical, whether to standardize predictors
#' @return list with coefficients, fitted values, lambda path
#' @export
lasso_regression <- function(X, y, lambda = NULL, max_iter = 1000,
                              tol = 1e-7, intercept = TRUE,
                              standardize = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)
  stopifnot(length(y) == n)

  # Standardize
  if (intercept) {
    y_mean <- mean(y)
    X_means <- colMeans(X)
    y_c <- y - y_mean
    X_c <- sweep(X, 2, X_means)
  } else {
    y_c <- y
    X_c <- X
    y_mean <- 0
    X_means <- rep(0, p)
  }

  if (standardize) {
    X_sds <- apply(X_c, 2, sd)
    X_sds[X_sds == 0] <- 1
    X_c <- sweep(X_c, 2, X_sds, "/")
  } else {
    X_sds <- rep(1, p)
  }

  # Compute lambda_max (smallest lambda with all zeros)
  lambda_max <- max(abs(crossprod(X_c, y_c))) / n

  if (is.null(lambda)) {
    n_lambda <- 100
    lambda_ratio <- 1e-4
    lambda <- lambda_max * exp(seq(log(1), log(lambda_ratio),
                                    length.out = n_lambda))
  }

  n_lambda <- length(lambda)

  # Storage for coefficient path
  beta_path <- matrix(0, nrow = p, ncol = n_lambda)
  intercept_path <- numeric(n_lambda)
  n_nonzero <- integer(n_lambda)
  iterations <- integer(n_lambda)

  # Precompute X'X diagonal
  X_sq_sum <- colSums(X_c^2) / n

  # Warm start: initialize beta
  beta <- rep(0, p)
  residuals <- y_c

  for (l in seq_len(n_lambda)) {
    lam <- lambda[l]

    for (iter in seq_len(max_iter)) {
      beta_old <- beta
      max_change <- 0

      for (j in seq_len(p)) {
        # Partial residual
        r_j <- residuals + X_c[, j] * beta[j]

        # Coordinate update
        z_j <- sum(X_c[, j] * r_j) / n
        beta_new <- soft_threshold(z_j, lam) / X_sq_sum[j]

        if (beta_new != beta[j]) {
          residuals <- residuals - X_c[, j] * (beta_new - beta[j])
          change <- abs(beta_new - beta[j])
          if (change > max_change) max_change <- change
          beta[j] <- beta_new
        }
      }

      if (max_change < tol) break
    }

    iterations[l] <- iter

    # Store (unstandardize)
    beta_path[, l] <- beta / X_sds
    n_nonzero[l] <- sum(abs(beta) > 0)

    if (intercept) {
      intercept_path[l] <- y_mean - sum(X_means * beta_path[, l])
    }
  }

  # Compute fitted values for last lambda
  if (intercept) {
    fitted <- as.numeric(X %*% beta_path[, n_lambda]) + intercept_path[n_lambda]
  } else {
    fitted <- as.numeric(X %*% beta_path[, n_lambda])
  }

  # Cross-validation for lambda selection
  # (Simplified: use last lambda; full CV below)

  structure(
    list(
      coefficients = beta_path[, n_lambda],
      beta_path = beta_path,
      intercept_path = intercept_path,
      lambda = lambda,
      lambda_min = lambda[n_lambda],
      n_nonzero = n_nonzero,
      iterations = iterations,
      fitted.values = fitted,
      residuals = y - fitted,
      n = n,
      p = p
    ),
    class = "lasso_model"
  )
}

#' LASSO with K-fold Cross-Validation
#'
#' @param X predictor matrix
#' @param y response vector
#' @param n_folds number of folds
#' @param lambda_seq optional lambda sequence
#' @return list with cv results and optimal model
#' @export
lasso_cv <- function(X, y, n_folds = 10, lambda_seq = NULL, ...) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  # Get lambda sequence from full model
  full_fit <- lasso_regression(X, y, lambda = lambda_seq, ...)
  lambda_seq <- full_fit$lambda
  n_lambda <- length(lambda_seq)

  # Create folds
  fold_ids <- sample(rep(seq_len(n_folds), length.out = n))

  # CV error matrix
  cv_errors <- matrix(0, nrow = n_folds, ncol = n_lambda)

  for (k in seq_len(n_folds)) {
    test_idx <- which(fold_ids == k)
    train_idx <- which(fold_ids != k)

    fit_k <- lasso_regression(X[train_idx, , drop = FALSE],
                               y[train_idx],
                               lambda = lambda_seq, ...)

    for (l in seq_len(n_lambda)) {
      pred <- as.numeric(X[test_idx, , drop = FALSE] %*%
                           fit_k$beta_path[, l]) + fit_k$intercept_path[l]
      cv_errors[k, l] <- mean((y[test_idx] - pred)^2)
    }
  }

  cv_mean <- colMeans(cv_errors)
  cv_se <- apply(cv_errors, 2, sd) / sqrt(n_folds)

  idx_min <- which.min(cv_mean)
  lambda_min <- lambda_seq[idx_min]

  # 1-SE rule
  threshold <- cv_mean[idx_min] + cv_se[idx_min]
  idx_1se <- min(which(cv_mean <= threshold))
  lambda_1se <- lambda_seq[idx_1se]

  # Refit on full data with lambda_min
  final_fit <- lasso_regression(X, y, lambda = lambda_min, ...)

  structure(
    list(
      lambda_min = lambda_min,
      lambda_1se = lambda_1se,
      cv_mean = cv_mean,
      cv_se = cv_se,
      lambda_seq = lambda_seq,
      idx_min = idx_min,
      idx_1se = idx_1se,
      model = final_fit
    ),
    class = "lasso_cv"
  )
}

# =============================================================================
# Elastic Net
# =============================================================================

#' Elastic Net Regression via Coordinate Descent
#'
#' Combines L1 and L2 penalties: alpha * ||beta||_1 + (1-alpha)/2 * ||beta||_2^2
#'
#' @param X predictor matrix
#' @param y response vector
#' @param alpha mixing parameter (0=ridge, 1=lasso)
#' @param lambda regularization strength
#' @param max_iter maximum iterations
#' @param tol convergence tolerance
#' @param intercept logical
#' @param standardize logical
#' @return elastic net model
#' @export
elastic_net <- function(X, y, alpha = 0.5, lambda = NULL,
                         max_iter = 1000, tol = 1e-7,
                         intercept = TRUE, standardize = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)

  # Center and standardize
  if (intercept) {
    y_mean <- mean(y)
    X_means <- colMeans(X)
    y_c <- y - y_mean
    X_c <- sweep(X, 2, X_means)
  } else {
    y_c <- y
    X_c <- X
    y_mean <- 0
    X_means <- rep(0, p)
  }

  if (standardize) {
    X_sds <- apply(X_c, 2, sd)
    X_sds[X_sds == 0] <- 1
    X_c <- sweep(X_c, 2, X_sds, "/")
  } else {
    X_sds <- rep(1, p)
  }

  # Lambda max
  lambda_max <- max(abs(crossprod(X_c, y_c))) / (n * max(alpha, 1e-3))

  if (is.null(lambda)) {
    n_lambda <- 100
    lambda <- lambda_max * exp(seq(log(1), log(1e-4), length.out = n_lambda))
  }

  n_lambda <- length(lambda)
  beta_path <- matrix(0, nrow = p, ncol = n_lambda)
  intercept_path <- numeric(n_lambda)

  X_sq_sum <- colSums(X_c^2) / n
  beta <- rep(0, p)
  residuals <- y_c

  for (l in seq_len(n_lambda)) {
    lam <- lambda[l]

    for (iter in seq_len(max_iter)) {
      beta_old <- beta
      max_change <- 0

      for (j in seq_len(p)) {
        r_j <- residuals + X_c[, j] * beta[j]
        z_j <- sum(X_c[, j] * r_j) / n

        # Elastic net update
        beta_new <- soft_threshold(z_j, lam * alpha) /
          (X_sq_sum[j] + lam * (1 - alpha))

        if (beta_new != beta[j]) {
          residuals <- residuals - X_c[, j] * (beta_new - beta[j])
          change <- abs(beta_new - beta[j])
          if (change > max_change) max_change <- change
          beta[j] <- beta_new
        }
      }

      if (max_change < tol) break
    }

    beta_path[, l] <- beta / X_sds
    if (intercept) {
      intercept_path[l] <- y_mean - sum(X_means * beta_path[, l])
    }
  }

  fitted <- as.numeric(X %*% beta_path[, n_lambda]) + intercept_path[n_lambda]

  structure(
    list(
      coefficients = beta_path[, n_lambda],
      beta_path = beta_path,
      intercept_path = intercept_path,
      lambda = lambda,
      alpha = alpha,
      fitted.values = fitted,
      residuals = y - fitted,
      n = n,
      p = p
    ),
    class = "elastic_net_model"
  )
}

# =============================================================================
# SECTION 2: LOGISTIC REGRESSION
# =============================================================================

#' Sigmoid / logistic function
#' @keywords internal
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

#' Logistic Regression via IRLS
#'
#' Iteratively Reweighted Least Squares for maximum likelihood estimation.
#'
#' @param X predictor matrix
#' @param y binary response (0/1)
#' @param max_iter maximum IRLS iterations
#' @param tol convergence tolerance
#' @param intercept logical
#' @return logistic regression model
#' @export
logistic_regression <- function(X, y, max_iter = 100, tol = 1e-8,
                                 intercept = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)

  if (intercept) {
    X <- cbind(1, X)
    p <- p + 1
  }

  stopifnot(all(y %in% c(0, 1)))

  # Initialize coefficients
  beta <- rep(0, p)

  log_lik_prev <- -Inf
  converged <- FALSE

  for (iter in seq_len(max_iter)) {
    # Linear predictor
    eta <- as.numeric(X %*% beta)

    # Probabilities (with numerical safeguard)
    mu <- sigmoid(eta)
    mu <- pmin(pmax(mu, 1e-10), 1 - 1e-10)

    # Working weights
    W <- mu * (1 - mu)

    # Working response
    z <- eta + (y - mu) / W

    # Weighted least squares
    W_sqrt <- sqrt(W)
    XW <- X * W_sqrt
    zW <- z * W_sqrt

    qr_XW <- qr(XW)
    beta_new <- qr.coef(qr_XW, zW)

    # Check for NA (singularity)
    if (any(is.na(beta_new))) {
      warning("Singular matrix in IRLS iteration ", iter)
      break
    }

    # Log-likelihood
    eta_new <- as.numeric(X %*% beta_new)
    log_lik <- sum(y * eta_new - log(1 + exp(eta_new)))

    # Check convergence
    if (abs(log_lik - log_lik_prev) < tol * (abs(log_lik_prev) + 1)) {
      converged <- TRUE
      beta <- beta_new
      break
    }

    beta <- beta_new
    log_lik_prev <- log_lik
  }

  # Final quantities
  eta <- as.numeric(X %*% beta)
  mu <- sigmoid(eta)
  mu_safe <- pmin(pmax(mu, 1e-10), 1 - 1e-10)
  W <- mu_safe * (1 - mu_safe)

  # Standard errors via Fisher information
  XWX <- crossprod(X * W, X)
  XWX_inv <- tryCatch(solve(XWX), error = function(e) {
    warning("Fisher information matrix is singular")
    matrix(NA, p, p)
  })
  se_beta <- sqrt(diag(XWX_inv))

  # z-statistics and p-values
  z_stats <- beta / se_beta
  p_values <- 2 * pnorm(abs(z_stats), lower.tail = FALSE)

  # Deviance
  log_lik_sat <- sum(ifelse(y == 1, log(pmax(y, 1e-10)),
                             log(pmax(1 - y, 1e-10))))
  deviance <- -2 * (log_lik - log_lik_sat)

  # Null deviance
  p_null <- mean(y)
  log_lik_null <- sum(y * log(p_null) + (1 - y) * log(1 - p_null))
  null_deviance <- -2 * (log_lik_null - log_lik_sat)

  # AIC
  aic <- -2 * log_lik + 2 * p

  # McFadden's pseudo R-squared
  pseudo_r2 <- 1 - log_lik / log_lik_null

  # Confusion matrix at threshold 0.5
  predicted <- as.integer(mu >= 0.5)
  conf_matrix <- table(Actual = y, Predicted = predicted)

  accuracy <- mean(predicted == y)

  structure(
    list(
      coefficients = beta,
      se = se_beta,
      z_stats = z_stats,
      p_values = p_values,
      fitted.values = mu,
      linear.predictor = eta,
      deviance = deviance,
      null_deviance = null_deviance,
      log_likelihood = log_lik,
      aic = aic,
      pseudo_r2 = pseudo_r2,
      confusion_matrix = conf_matrix,
      accuracy = accuracy,
      iterations = iter,
      converged = converged,
      n = n,
      p = p,
      intercept = intercept,
      X = X,
      y = y
    ),
    class = "logistic_model"
  )
}

#' Print method for logistic model
#' @export
print.logistic_model <- function(x, ...) {
  cat("Logistic Regression (IRLS)\n")
  cat("==========================\n")
  cat("Converged:", x$converged, "in", x$iterations, "iterations\n\n")
  coef_table <- data.frame(
    Estimate = x$coefficients,
    Std.Error = x$se,
    z.value = x$z_stats,
    p.value = x$p_values
  )
  print(round(coef_table, 6))
  cat("\nNull deviance:", round(x$null_deviance, 2), "\n")
  cat("Residual deviance:", round(x$deviance, 2), "\n")
  cat("AIC:", round(x$aic, 2), "\n")
  cat("Pseudo R-squared:", round(x$pseudo_r2, 4), "\n")
  cat("Accuracy:", round(x$accuracy, 4), "\n")
  invisible(x)
}

#' Predict method for logistic model
#' @export
predict.logistic_model <- function(object, newdata = NULL,
                                    type = "response", ...) {
  if (is.null(newdata)) {
    if (type == "response") return(object$fitted.values)
    if (type == "link") return(object$linear.predictor)
    if (type == "class") return(as.integer(object$fitted.values >= 0.5))
  }

  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  if (object$intercept) newdata <- cbind(1, newdata)

  eta <- as.numeric(newdata %*% object$coefficients)
  if (type == "link") return(eta)
  prob <- sigmoid(eta)
  if (type == "class") return(as.integer(prob >= 0.5))
  prob
}

#' L1-Regularized Logistic Regression (Lasso Logistic)
#'
#' Coordinate descent with proximal gradient for L1 penalty
#'
#' @param X predictor matrix
#' @param y binary response
#' @param lambda regularization parameter
#' @param max_iter maximum iterations
#' @param tol convergence tolerance
#' @return regularized logistic model
#' @export
logistic_l1 <- function(X, y, lambda = 0.1, max_iter = 500,
                          tol = 1e-6, intercept = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)

  if (intercept) {
    X_aug <- cbind(1, X)
    p_aug <- p + 1
  } else {
    X_aug <- X
    p_aug <- p
  }

  beta <- rep(0, p_aug)
  penalty_factor <- c(if (intercept) 0 else NULL, rep(1, p))

  for (iter in seq_len(max_iter)) {
    beta_old <- beta
    eta <- as.numeric(X_aug %*% beta)
    mu <- sigmoid(eta)
    mu <- pmin(pmax(mu, 1e-10), 1 - 1e-10)
    W <- mu * (1 - mu)

    for (j in seq_len(p_aug)) {
      r_j <- y - mu + W * X_aug[, j] * beta[j]
      z_j <- sum(X_aug[, j] * r_j) / n
      denom <- sum(W * X_aug[, j]^2) / n

      if (penalty_factor[j] > 0) {
        beta[j] <- soft_threshold(z_j, lambda * penalty_factor[j]) / denom
      } else {
        beta[j] <- z_j / denom
      }

      # Update mu and W
      eta <- as.numeric(X_aug %*% beta)
      mu <- sigmoid(eta)
      mu <- pmin(pmax(mu, 1e-10), 1 - 1e-10)
      W <- mu * (1 - mu)
    }

    if (max(abs(beta - beta_old)) < tol) break
  }

  fitted <- sigmoid(as.numeric(X_aug %*% beta))

  structure(
    list(
      coefficients = beta,
      fitted.values = fitted,
      lambda = lambda,
      n_nonzero = sum(abs(beta[-1]) > 1e-10),
      iterations = iter,
      n = n,
      p = p
    ),
    class = "logistic_l1_model"
  )
}

#' L2-Regularized Logistic Regression (Ridge Logistic)
#'
#' @param X predictor matrix
#' @param y binary response
#' @param lambda regularization parameter
#' @param max_iter maximum iterations
#' @param tol convergence tolerance
#' @return regularized logistic model
#' @export
logistic_l2 <- function(X, y, lambda = 0.1, max_iter = 100,
                          tol = 1e-8, intercept = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)

  if (intercept) {
    X_aug <- cbind(1, X)
    p_aug <- p + 1
    pen_diag <- c(0, rep(lambda, p))
  } else {
    X_aug <- X
    p_aug <- p
    pen_diag <- rep(lambda, p)
  }

  beta <- rep(0, p_aug)

  for (iter in seq_len(max_iter)) {
    eta <- as.numeric(X_aug %*% beta)
    mu <- sigmoid(eta)
    mu <- pmin(pmax(mu, 1e-10), 1 - 1e-10)
    W <- mu * (1 - mu)

    # Score
    score <- crossprod(X_aug, y - mu) / n - pen_diag * beta

    # Hessian
    H <- crossprod(X_aug * W, X_aug) / n + diag(pen_diag)

    # Newton step
    delta <- solve(H, score)
    beta <- beta + delta

    if (max(abs(delta)) < tol) break
  }

  fitted <- sigmoid(as.numeric(X_aug %*% beta))

  structure(
    list(
      coefficients = beta,
      fitted.values = fitted,
      lambda = lambda,
      iterations = iter,
      n = n,
      p = p
    ),
    class = "logistic_l2_model"
  )
}

# =============================================================================
# SECTION 3: DECISION TREE (CART)
# =============================================================================

#' Gini impurity
#' @keywords internal
gini_impurity <- function(y) {
  if (length(y) == 0) return(0)
  p <- table(y) / length(y)
  1 - sum(p^2)
}

#' Entropy (information gain criterion)
#' @keywords internal
entropy <- function(y) {
  if (length(y) == 0) return(0)
  p <- table(y) / length(y)
  p <- p[p > 0]
  -sum(p * log2(p))
}

#' Variance (for regression trees)
#' @keywords internal
node_variance <- function(y) {
  if (length(y) <= 1) return(0)
  var(y) * (length(y) - 1) / length(y)
}

#' Find the best split for a node
#' @keywords internal
find_best_split <- function(X, y, criterion = "gini", min_samples_leaf = 1,
                             feature_subset = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  best_gain <- -Inf
  best_feature <- NULL
  best_threshold <- NULL

  if (is.null(feature_subset)) {
    features <- seq_len(p)
  } else {
    features <- feature_subset
  }

  is_classification <- is.factor(y) || is.character(y)

  if (is_classification) {
    impurity_fn <- if (criterion == "gini") gini_impurity else entropy
    parent_impurity <- impurity_fn(y)
  } else {
    parent_impurity <- node_variance(y)
  }

  for (j in features) {
    x_j <- X[, j]
    unique_vals <- sort(unique(x_j))

    if (length(unique_vals) <= 1) next

    # Candidate thresholds: midpoints
    if (length(unique_vals) > 50) {
      # Subsample thresholds for efficiency
      quantile_vals <- quantile(x_j, probs = seq(0.02, 0.98, length.out = 50))
      thresholds <- unique(quantile_vals)
    } else {
      thresholds <- (unique_vals[-length(unique_vals)] +
                       unique_vals[-1]) / 2
    }

    for (thresh in thresholds) {
      left_idx <- which(x_j <= thresh)
      right_idx <- which(x_j > thresh)

      if (length(left_idx) < min_samples_leaf ||
          length(right_idx) < min_samples_leaf) {
        next
      }

      n_left <- length(left_idx)
      n_right <- length(right_idx)

      if (is_classification) {
        left_impurity <- impurity_fn(y[left_idx])
        right_impurity <- impurity_fn(y[right_idx])
      } else {
        left_impurity <- node_variance(y[left_idx])
        right_impurity <- node_variance(y[right_idx])
      }

      # Weighted impurity decrease
      gain <- parent_impurity -
        (n_left / n) * left_impurity -
        (n_right / n) * right_impurity

      if (gain > best_gain) {
        best_gain <- gain
        best_feature <- j
        best_threshold <- thresh
      }
    }
  }

  list(
    feature = best_feature,
    threshold = best_threshold,
    gain = best_gain
  )
}

#' Build a decision tree recursively
#' @keywords internal
build_tree <- function(X, y, depth = 0, max_depth = 30,
                        min_samples_split = 2, min_samples_leaf = 1,
                        criterion = "gini", feature_subset_fn = NULL,
                        node_id = 1) {
  n <- nrow(X)
  is_classification <- is.factor(y) || is.character(y)

  # Create leaf node
  make_leaf <- function() {
    if (is_classification) {
      prediction <- names(which.max(table(y)))
      probs <- table(y) / length(y)
    } else {
      prediction <- mean(y)
      probs <- NULL
    }
    list(
      is_leaf = TRUE,
      prediction = prediction,
      probs = probs,
      n = n,
      node_id = node_id,
      depth = depth
    )
  }

  # Check stopping conditions
  if (n < min_samples_split || depth >= max_depth) {
    return(make_leaf())
  }

  if (is_classification && length(unique(y)) == 1) {
    return(make_leaf())
  }

  if (!is_classification && var(y) < 1e-10) {
    return(make_leaf())
  }

  # Feature subset (for random forests)
  if (!is.null(feature_subset_fn)) {
    feature_subset <- feature_subset_fn(ncol(X))
  } else {
    feature_subset <- NULL
  }

  # Find best split
  split <- find_best_split(X, y, criterion = criterion,
                            min_samples_leaf = min_samples_leaf,
                            feature_subset = feature_subset)

  if (is.null(split$feature) || split$gain <= 0) {
    return(make_leaf())
  }

  # Split data
  left_idx <- which(X[, split$feature] <= split$threshold)
  right_idx <- which(X[, split$feature] > split$threshold)

  # Recursive build
  left_child <- build_tree(X[left_idx, , drop = FALSE], y[left_idx],
                            depth = depth + 1, max_depth = max_depth,
                            min_samples_split = min_samples_split,
                            min_samples_leaf = min_samples_leaf,
                            criterion = criterion,
                            feature_subset_fn = feature_subset_fn,
                            node_id = 2 * node_id)

  right_child <- build_tree(X[right_idx, , drop = FALSE], y[right_idx],
                              depth = depth + 1, max_depth = max_depth,
                              min_samples_split = min_samples_split,
                              min_samples_leaf = min_samples_leaf,
                              criterion = criterion,
                              feature_subset_fn = feature_subset_fn,
                              node_id = 2 * node_id + 1)

  list(
    is_leaf = FALSE,
    feature = split$feature,
    threshold = split$threshold,
    gain = split$gain,
    left = left_child,
    right = right_child,
    n = n,
    node_id = node_id,
    depth = depth
  )
}

#' Predict with a decision tree node
#' @keywords internal
predict_tree_single <- function(node, x) {
  if (node$is_leaf) {
    return(node$prediction)
  }

  if (x[node$feature] <= node$threshold) {
    predict_tree_single(node$left, x)
  } else {
    predict_tree_single(node$right, x)
  }
}

#' Predict probabilities with a decision tree
#' @keywords internal
predict_tree_probs_single <- function(node, x) {
  if (node$is_leaf) {
    return(node$probs)
  }

  if (x[node$feature] <= node$threshold) {
    predict_tree_probs_single(node$left, x)
  } else {
    predict_tree_probs_single(node$right, x)
  }
}

#' CART Decision Tree
#'
#' Classification and Regression Tree with Gini/entropy splitting
#' and cost-complexity pruning.
#'
#' @param X predictor matrix
#' @param y response (factor for classification, numeric for regression)
#' @param max_depth maximum tree depth
#' @param min_samples_split minimum samples to attempt split
#' @param min_samples_leaf minimum samples in leaf
#' @param criterion splitting criterion ("gini" or "entropy")
#' @param cp complexity parameter for pruning (NULL = no pruning)
#' @return decision tree model
#' @export
decision_tree <- function(X, y, max_depth = 30, min_samples_split = 2,
                           min_samples_leaf = 1, criterion = "gini",
                           cp = NULL) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)

  tree <- build_tree(X, y, max_depth = max_depth,
                      min_samples_split = min_samples_split,
                      min_samples_leaf = min_samples_leaf,
                      criterion = criterion)

  is_classification <- is.factor(y) || is.character(y)

  model <- structure(
    list(
      tree = tree,
      is_classification = is_classification,
      n_features = ncol(X),
      classes = if (is_classification) levels(factor(y)) else NULL,
      n = nrow(X),
      criterion = criterion,
      max_depth = max_depth
    ),
    class = "cart_model"
  )

  # Apply cost-complexity pruning if requested
  if (!is.null(cp) && cp > 0) {
    model <- prune_tree(model, cp = cp)
  }

  model
}

#' Count leaves in a subtree
#' @keywords internal
count_leaves <- function(node) {
  if (node$is_leaf) return(1)
  count_leaves(node$left) + count_leaves(node$right)
}

#' Compute subtree error
#' @keywords internal
subtree_error <- function(node, X, y, is_classification) {
  if (node$is_leaf) {
    if (is_classification) {
      pred <- node$prediction
      return(sum(y != pred))
    } else {
      return(sum((y - node$prediction)^2))
    }
  }

  left_idx <- which(X[, node$feature] <= node$threshold)
  right_idx <- which(X[, node$feature] > node$threshold)

  err_left <- if (length(left_idx) > 0) {
    subtree_error(node$left, X[left_idx, , drop = FALSE],
                   y[left_idx], is_classification)
  } else {
    0
  }

  err_right <- if (length(right_idx) > 0) {
    subtree_error(node$right, X[right_idx, , drop = FALSE],
                   y[right_idx], is_classification)
  } else {
    0
  }

  err_left + err_right
}

#' Cost-Complexity Pruning
#'
#' @param model decision tree model
#' @param cp complexity parameter
#' @return pruned model
#' @export
prune_tree <- function(model, cp = 0.01) {
  prune_node <- function(node) {
    if (node$is_leaf) return(node)

    # Recursively prune children
    node$left <- prune_node(node$left)
    node$right <- prune_node(node$right)

    # Check if this internal node should become a leaf
    n_leaves <- count_leaves(node)
    if (node$gain <= cp * n_leaves) {
      # Replace with leaf
      if (model$is_classification) {
        # Majority vote (approximate - uses left/right predictions)
        list(
          is_leaf = TRUE,
          prediction = node$left$prediction,
          probs = node$left$probs,
          n = node$n,
          node_id = node$node_id,
          depth = node$depth
        )
      } else {
        list(
          is_leaf = TRUE,
          prediction = (node$left$prediction * node$left$n +
                          node$right$prediction * node$right$n) /
            (node$left$n + node$right$n),
          probs = NULL,
          n = node$n,
          node_id = node$node_id,
          depth = node$depth
        )
      }
    } else {
      node
    }
  }

  model$tree <- prune_node(model$tree)
  model
}

#' Predict method for CART
#' @export
predict.cart_model <- function(object, newdata = NULL, type = "class", ...) {
  if (is.null(newdata)) stop("newdata required")
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)

  n <- nrow(newdata)

  if (type == "class" || !object$is_classification) {
    predictions <- sapply(seq_len(n), function(i) {
      predict_tree_single(object$tree, newdata[i, ])
    })
    if (!object$is_classification) {
      predictions <- as.numeric(predictions)
    }
    return(predictions)
  }

  if (type == "prob") {
    probs_list <- lapply(seq_len(n), function(i) {
      predict_tree_probs_single(object$tree, newdata[i, ])
    })
    probs_mat <- do.call(rbind, lapply(probs_list, function(p) {
      as.numeric(p[object$classes])
    }))
    colnames(probs_mat) <- object$classes
    return(probs_mat)
  }
}

# =============================================================================
# SECTION 4: RANDOM FOREST
# =============================================================================

#' Random Forest
#'
#' Bagging with random feature subsets at each split.
#'
#' @param X predictor matrix
#' @param y response
#' @param n_trees number of trees
#' @param max_features number of features per split (NULL = sqrt(p) for
#'   classification, p/3 for regression)
#' @param max_depth max tree depth
#' @param min_samples_split minimum samples to split
#' @param min_samples_leaf minimum samples in leaf
#' @param criterion splitting criterion
#' @param oob logical, compute OOB error
#' @return random forest model
#' @export
random_forest <- function(X, y, n_trees = 100, max_features = NULL,
                           max_depth = 30, min_samples_split = 2,
                           min_samples_leaf = 1, criterion = "gini",
                           oob = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)
  is_classification <- is.factor(y) || is.character(y)

  if (is.null(max_features)) {
    max_features <- if (is_classification) {
      max(1, floor(sqrt(p)))
    } else {
      max(1, floor(p / 3))
    }
  }

  feature_subset_fn <- function(p_total) {
    sample.int(p_total, size = min(max_features, p_total))
  }

  trees <- vector("list", n_trees)
  oob_indices <- vector("list", n_trees)

  for (b in seq_len(n_trees)) {
    # Bootstrap sample
    boot_idx <- sample.int(n, replace = TRUE)
    oob_idx <- setdiff(seq_len(n), unique(boot_idx))
    oob_indices[[b]] <- oob_idx

    X_boot <- X[boot_idx, , drop = FALSE]
    y_boot <- y[boot_idx]

    trees[[b]] <- build_tree(X_boot, y_boot, max_depth = max_depth,
                              min_samples_split = min_samples_split,
                              min_samples_leaf = min_samples_leaf,
                              criterion = criterion,
                              feature_subset_fn = feature_subset_fn)
  }

  # OOB error estimation
  oob_error <- NULL
  oob_predictions <- NULL
  if (oob) {
    if (is_classification) {
      oob_votes <- matrix(0, nrow = n, ncol = length(unique(y)))
      class_labels <- levels(factor(y))
      colnames(oob_votes) <- class_labels

      for (b in seq_len(n_trees)) {
        if (length(oob_indices[[b]]) == 0) next
        for (i in oob_indices[[b]]) {
          pred <- predict_tree_single(trees[[b]], X[i, ])
          idx <- match(pred, class_labels)
          if (!is.na(idx)) oob_votes[i, idx] <- oob_votes[i, idx] + 1
        }
      }

      oob_predicted <- apply(oob_votes, 1, function(row) {
        if (sum(row) == 0) return(NA)
        class_labels[which.max(row)]
      })

      valid <- !is.na(oob_predicted)
      oob_error <- mean(oob_predicted[valid] != as.character(y[valid]))
      oob_predictions <- oob_predicted
    } else {
      oob_sum <- numeric(n)
      oob_count <- integer(n)

      for (b in seq_len(n_trees)) {
        for (i in oob_indices[[b]]) {
          pred <- as.numeric(predict_tree_single(trees[[b]], X[i, ]))
          oob_sum[i] <- oob_sum[i] + pred
          oob_count[i] <- oob_count[i] + 1
        }
      }

      valid <- oob_count > 0
      oob_predicted <- rep(NA, n)
      oob_predicted[valid] <- oob_sum[valid] / oob_count[valid]
      oob_error <- mean((y[valid] - oob_predicted[valid])^2)
      oob_predictions <- oob_predicted
    }
  }

  # Variable importance (permutation-based)
  var_importance <- compute_variable_importance(trees, oob_indices, X, y,
                                                 is_classification)

  structure(
    list(
      trees = trees,
      n_trees = n_trees,
      is_classification = is_classification,
      classes = if (is_classification) levels(factor(y)) else NULL,
      oob_error = oob_error,
      oob_predictions = oob_predictions,
      var_importance = var_importance,
      n_features = p,
      n = n
    ),
    class = "random_forest_model"
  )
}

#' Compute permutation-based variable importance
#' @keywords internal
compute_variable_importance <- function(trees, oob_indices, X, y,
                                         is_classification) {
  n <- nrow(X)
  p <- ncol(X)
  n_trees <- length(trees)
  importance <- numeric(p)

  for (j in seq_len(p)) {
    imp_j <- 0
    n_valid <- 0

    for (b in seq_len(n_trees)) {
      oob_idx <- oob_indices[[b]]
      if (length(oob_idx) < 2) next

      # Original OOB predictions
      orig_preds <- sapply(oob_idx, function(i) {
        predict_tree_single(trees[[b]], X[i, ])
      })

      # Permuted OOB predictions
      X_perm <- X[oob_idx, , drop = FALSE]
      X_perm[, j] <- sample(X_perm[, j])
      perm_preds <- sapply(seq_len(nrow(X_perm)), function(i) {
        predict_tree_single(trees[[b]], X_perm[i, ])
      })

      if (is_classification) {
        y_oob <- as.character(y[oob_idx])
        orig_acc <- mean(orig_preds == y_oob)
        perm_acc <- mean(perm_preds == y_oob)
        imp_j <- imp_j + (orig_acc - perm_acc)
      } else {
        y_oob <- y[oob_idx]
        orig_mse <- mean((as.numeric(orig_preds) - y_oob)^2)
        perm_mse <- mean((as.numeric(perm_preds) - y_oob)^2)
        imp_j <- imp_j + (perm_mse - orig_mse)
      }
      n_valid <- n_valid + 1
    }

    importance[j] <- if (n_valid > 0) imp_j / n_valid else 0
  }

  importance
}

#' Predict method for Random Forest
#' @export
predict.random_forest_model <- function(object, newdata, type = "class", ...) {
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  n <- nrow(newdata)
  n_trees <- object$n_trees

  if (object$is_classification) {
    # Collect votes
    all_preds <- matrix(NA_character_, nrow = n, ncol = n_trees)
    for (b in seq_len(n_trees)) {
      for (i in seq_len(n)) {
        all_preds[i, b] <- predict_tree_single(object$trees[[b]],
                                                 newdata[i, ])
      }
    }

    if (type == "class") {
      predictions <- apply(all_preds, 1, function(row) {
        tab <- table(row)
        names(tab)[which.max(tab)]
      })
      return(predictions)
    }

    if (type == "prob") {
      probs <- t(apply(all_preds, 1, function(row) {
        tab <- table(factor(row, levels = object$classes))
        as.numeric(tab) / sum(tab)
      }))
      colnames(probs) <- object$classes
      return(probs)
    }
  } else {
    all_preds <- matrix(NA_real_, nrow = n, ncol = n_trees)
    for (b in seq_len(n_trees)) {
      for (i in seq_len(n)) {
        all_preds[i, b] <- as.numeric(
          predict_tree_single(object$trees[[b]], newdata[i, ]))
      }
    }
    return(rowMeans(all_preds))
  }
}

#' Print method for Random Forest
#' @export
print.random_forest_model <- function(x, ...) {
  cat("Random Forest\n")
  cat("=============\n")
  cat("Number of trees:", x$n_trees, "\n")
  cat("Type:", if (x$is_classification) "Classification" else "Regression", "\n")
  cat("Number of features:", x$n_features, "\n")
  if (!is.null(x$oob_error)) {
    if (x$is_classification) {
      cat("OOB error rate:", round(x$oob_error, 4), "\n")
    } else {
      cat("OOB MSE:", round(x$oob_error, 4), "\n")
    }
  }
  if (!is.null(x$var_importance)) {
    cat("\nVariable Importance:\n")
    imp_sorted <- sort(x$var_importance, decreasing = TRUE)
    print(round(head(imp_sorted, 20), 6))
  }
  invisible(x)
}

# =============================================================================
# SECTION 5: GRADIENT BOOSTING
# =============================================================================

#' Gradient Boosting Machine
#'
#' Implements gradient boosting with squared loss (regression) or
#' deviance loss (classification).
#'
#' @param X predictor matrix
#' @param y response
#' @param n_trees number of boosting rounds
#' @param learning_rate shrinkage factor
#' @param max_depth max depth of base learners
#' @param min_samples_leaf minimum samples in leaf
#' @param subsample fraction of data to sample per tree
#' @param loss loss function ("squared" or "deviance")
#' @return gradient boosting model
#' @export
gradient_boosting <- function(X, y, n_trees = 100, learning_rate = 0.1,
                               max_depth = 3, min_samples_leaf = 5,
                               subsample = 1.0, loss = "squared") {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  is_classification <- (loss == "deviance")

  if (is_classification) {
    classes <- sort(unique(y))
    stopifnot(length(classes) == 2)
    y_binary <- as.numeric(y == classes[2])

    # Initialize with log-odds
    p_hat <- mean(y_binary)
    F_current <- rep(log(p_hat / (1 - p_hat)), n)
  } else {
    # Initialize with mean
    F_current <- rep(mean(y), n)
  }

  trees <- vector("list", n_trees)
  train_loss <- numeric(n_trees)

  for (m in seq_len(n_trees)) {
    # Compute negative gradient (pseudo-residuals)
    if (is_classification) {
      prob <- sigmoid(F_current)
      pseudo_resid <- y_binary - prob
    } else {
      pseudo_resid <- y - F_current
    }

    # Subsample
    if (subsample < 1.0) {
      n_sub <- max(1, floor(n * subsample))
      sub_idx <- sample.int(n, n_sub)
    } else {
      sub_idx <- seq_len(n)
    }

    # Fit regression tree to pseudo-residuals
    tree_m <- build_tree(X[sub_idx, , drop = FALSE],
                          pseudo_resid[sub_idx],
                          max_depth = max_depth,
                          min_samples_leaf = min_samples_leaf,
                          criterion = "gini")  # variance-based for regression

    trees[[m]] <- tree_m

    # Update predictions
    for (i in seq_len(n)) {
      leaf_pred <- as.numeric(predict_tree_single(tree_m, X[i, ]))
      F_current[i] <- F_current[i] + learning_rate * leaf_pred
    }

    # Compute training loss
    if (is_classification) {
      prob <- sigmoid(F_current)
      train_loss[m] <- -mean(y_binary * log(pmax(prob, 1e-10)) +
                                (1 - y_binary) * log(pmax(1 - prob, 1e-10)))
    } else {
      train_loss[m] <- mean((y - F_current)^2)
    }
  }

  structure(
    list(
      trees = trees,
      n_trees = n_trees,
      learning_rate = learning_rate,
      is_classification = is_classification,
      classes = if (is_classification) classes else NULL,
      init_value = if (is_classification) log(p_hat / (1 - p_hat)) else mean(y),
      train_loss = train_loss,
      loss = loss,
      n_features = ncol(X)
    ),
    class = "gbm_model"
  )
}

#' Predict method for GBM
#' @export
predict.gbm_model <- function(object, newdata, n_trees = NULL,
                                type = "response", ...) {
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  n <- nrow(newdata)
  if (is.null(n_trees)) n_trees <- object$n_trees

  F_pred <- rep(object$init_value, n)

  for (m in seq_len(n_trees)) {
    for (i in seq_len(n)) {
      leaf_pred <- as.numeric(
        predict_tree_single(object$trees[[m]], newdata[i, ]))
      F_pred[i] <- F_pred[i] + object$learning_rate * leaf_pred
    }
  }

  if (object$is_classification) {
    if (type == "response") {
      return(sigmoid(F_pred))
    }
    if (type == "class") {
      return(ifelse(sigmoid(F_pred) >= 0.5,
                    object$classes[2], object$classes[1]))
    }
  }

  F_pred
}

#' Print method for GBM
#' @export
print.gbm_model <- function(x, ...) {
  cat("Gradient Boosting Machine\n")
  cat("=========================\n")
  cat("Trees:", x$n_trees, "\n")
  cat("Learning rate:", x$learning_rate, "\n")
  cat("Loss:", x$loss, "\n")
  cat("Final training loss:", round(tail(x$train_loss, 1), 6), "\n")
  invisible(x)
}

# =============================================================================
# SECTION 6: SUPPORT VECTOR MACHINE (Simplified SMO)
# =============================================================================

#' Kernel functions
#' @keywords internal
kernel_linear <- function(x, y) {
  sum(x * y)
}

#' @keywords internal
kernel_rbf <- function(x, y, gamma = 1) {
  exp(-gamma * sum((x - y)^2))
}

#' @keywords internal
kernel_polynomial <- function(x, y, degree = 3, coef0 = 1) {
  (sum(x * y) + coef0)^degree
}

#' Compute kernel matrix
#' @keywords internal
compute_kernel_matrix <- function(X, kernel = "linear", gamma = NULL,
                                   degree = 3, coef0 = 1) {
  n <- nrow(X)
  K <- matrix(0, n, n)

  if (is.null(gamma)) gamma <- 1 / ncol(X)

  for (i in seq_len(n)) {
    for (j in i:n) {
      val <- switch(kernel,
                     "linear" = kernel_linear(X[i, ], X[j, ]),
                     "rbf" = kernel_rbf(X[i, ], X[j, ], gamma),
                     "polynomial" = kernel_polynomial(X[i, ], X[j, ],
                                                       degree, coef0))
      K[i, j] <- val
      K[j, i] <- val
    }
  }
  K
}

#' Simplified SMO Algorithm for SVM
#'
#' @param X predictor matrix
#' @param y response (+1/-1)
#' @param C regularization parameter
#' @param kernel kernel type ("linear", "rbf", "polynomial")
#' @param gamma RBF parameter
#' @param degree polynomial degree
#' @param max_iter maximum passes over data
#' @param tol tolerance
#' @return SVM model
#' @export
svm_train <- function(X, y, C = 1.0, kernel = "linear",
                       gamma = NULL, degree = 3, coef0 = 1,
                       max_iter = 100, tol = 1e-3) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)

  # Convert labels to +1/-1
  classes <- sort(unique(y))
  stopifnot(length(classes) == 2)
  y_svm <- ifelse(y == classes[2], 1, -1)

  if (is.null(gamma)) gamma <- 1 / p

  # Compute kernel matrix
  K <- compute_kernel_matrix(X, kernel, gamma, degree, coef0)

  # Initialize
  alpha <- rep(0, n)
  b <- 0
  passes <- 0

  while (passes < max_iter) {
    num_changed <- 0

    for (i in seq_len(n)) {
      # Compute E_i
      f_i <- sum(alpha * y_svm * K[i, ]) + b
      E_i <- f_i - y_svm[i]

      # Check KKT conditions
      if ((y_svm[i] * E_i < -tol && alpha[i] < C) ||
          (y_svm[i] * E_i > tol && alpha[i] > 0)) {

        # Select j randomly (simplified SMO)
        j <- sample(setdiff(seq_len(n), i), 1)

        f_j <- sum(alpha * y_svm * K[j, ]) + b
        E_j <- f_j - y_svm[j]

        alpha_i_old <- alpha[i]
        alpha_j_old <- alpha[j]

        # Compute bounds
        if (y_svm[i] != y_svm[j]) {
          L <- max(0, alpha[j] - alpha[i])
          H <- min(C, C + alpha[j] - alpha[i])
        } else {
          L <- max(0, alpha[i] + alpha[j] - C)
          H <- min(C, alpha[i] + alpha[j])
        }

        if (abs(L - H) < 1e-10) next

        # Compute eta
        eta <- 2 * K[i, j] - K[i, i] - K[j, j]
        if (eta >= 0) next

        # Update alpha_j
        alpha[j] <- alpha[j] - y_svm[j] * (E_i - E_j) / eta
        alpha[j] <- min(H, max(L, alpha[j]))

        if (abs(alpha[j] - alpha_j_old) < 1e-5) next

        # Update alpha_i
        alpha[i] <- alpha[i] + y_svm[i] * y_svm[j] *
          (alpha_j_old - alpha[j])

        # Update bias
        b1 <- b - E_i - y_svm[i] * (alpha[i] - alpha_i_old) * K[i, i] -
          y_svm[j] * (alpha[j] - alpha_j_old) * K[i, j]
        b2 <- b - E_j - y_svm[i] * (alpha[i] - alpha_i_old) * K[i, j] -
          y_svm[j] * (alpha[j] - alpha_j_old) * K[j, j]

        if (alpha[i] > 0 && alpha[i] < C) {
          b <- b1
        } else if (alpha[j] > 0 && alpha[j] < C) {
          b <- b2
        } else {
          b <- (b1 + b2) / 2
        }

        num_changed <- num_changed + 1
      }
    }

    if (num_changed == 0) {
      passes <- passes + 1
    } else {
      passes <- 0
    }
  }

  # Support vectors
  sv_idx <- which(alpha > 1e-7)

  # For linear kernel, compute weight vector
  w <- NULL
  if (kernel == "linear") {
    w <- colSums(alpha[sv_idx] * y_svm[sv_idx] * X[sv_idx, , drop = FALSE])
  }

  structure(
    list(
      alpha = alpha,
      b = b,
      sv_idx = sv_idx,
      X_sv = X[sv_idx, , drop = FALSE],
      y_sv = y_svm[sv_idx],
      alpha_sv = alpha[sv_idx],
      w = w,
      kernel = kernel,
      gamma = gamma,
      degree = degree,
      coef0 = coef0,
      C = C,
      classes = classes,
      n_sv = length(sv_idx),
      X = X,
      y_svm = y_svm
    ),
    class = "svm_model"
  )
}

#' Predict method for SVM
#' @export
predict.svm_model <- function(object, newdata, type = "class", ...) {
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  n <- nrow(newdata)

  # Compute decision function
  decision <- numeric(n)

  for (i in seq_len(n)) {
    val <- 0
    for (s in seq_len(object$n_sv)) {
      k_val <- switch(object$kernel,
                       "linear" = kernel_linear(newdata[i, ],
                                                 object$X_sv[s, ]),
                       "rbf" = kernel_rbf(newdata[i, ],
                                           object$X_sv[s, ],
                                           object$gamma),
                       "polynomial" = kernel_polynomial(newdata[i, ],
                                                         object$X_sv[s, ],
                                                         object$degree,
                                                         object$coef0))
      val <- val + object$alpha_sv[s] * object$y_sv[s] * k_val
    }
    decision[i] <- val + object$b
  }

  if (type == "decision") return(decision)

  # Classify
  predicted <- ifelse(decision >= 0, object$classes[2], object$classes[1])
  predicted
}

# =============================================================================
# SECTION 7: K-NEAREST NEIGHBORS
# =============================================================================

#' Euclidean distance between two vectors
#' @keywords internal
euclidean_dist <- function(a, b) {
  sqrt(sum((a - b)^2))
}

#' Manhattan distance
#' @keywords internal
manhattan_dist <- function(a, b) {
  sum(abs(a - b))
}

#' KD-Tree node structure
#' @keywords internal
kdtree_build <- function(X, indices = NULL, depth = 0) {
  if (is.null(indices)) indices <- seq_len(nrow(X))
  n <- length(indices)

  if (n == 0) return(NULL)
  if (n == 1) {
    return(list(
      point = X[indices[1], ],
      index = indices[1],
      left = NULL,
      right = NULL,
      split_dim = NA,
      split_val = NA,
      is_leaf = TRUE
    ))
  }

  p <- ncol(X)
  split_dim <- (depth %% p) + 1

  # Sort by split dimension
  vals <- X[indices, split_dim]
  ord <- order(vals)
  sorted_indices <- indices[ord]

  median_idx <- ceiling(n / 2)
  median_point_idx <- sorted_indices[median_idx]

  left_indices <- sorted_indices[seq_len(median_idx - 1)]
  right_indices <- if (median_idx < n) {
    sorted_indices[(median_idx + 1):n]
  } else {
    integer(0)
  }

  list(
    point = X[median_point_idx, ],
    index = median_point_idx,
    left = kdtree_build(X, left_indices, depth + 1),
    right = kdtree_build(X, right_indices, depth + 1),
    split_dim = split_dim,
    split_val = X[median_point_idx, split_dim],
    is_leaf = FALSE
  )
}

#' KD-Tree nearest neighbor search
#' @keywords internal
kdtree_search <- function(node, query, k, best = NULL, dist_fn = euclidean_dist) {
  if (is.null(node)) return(best)

  if (is.null(best)) {
    best <- list(
      indices = integer(0),
      distances = numeric(0)
    )
  }

  d <- dist_fn(query, node$point)

  # Update best list
  if (length(best$distances) < k) {
    best$indices <- c(best$indices, node$index)
    best$distances <- c(best$distances, d)
  } else if (d < max(best$distances)) {
    worst <- which.max(best$distances)
    best$indices[worst] <- node$index
    best$distances[worst] <- d
  }

  if (is.null(node$left) && is.null(node$right)) return(best)

  # Determine which side to search first
  diff <- query[node$split_dim] - node$split_val
  if (diff <= 0) {
    first <- node$left
    second <- node$right
  } else {
    first <- node$right
    second <- node$left
  }

  best <- kdtree_search(first, query, k, best, dist_fn)

  # Check if we need to search the other side
  max_dist <- if (length(best$distances) < k) Inf else max(best$distances)
  if (abs(diff) < max_dist) {
    best <- kdtree_search(second, query, k, best, dist_fn)
  }

  best
}

#' K-Nearest Neighbors
#'
#' Classification and regression with brute force or KD-tree.
#'
#' @param X_train training predictors
#' @param y_train training response
#' @param X_test test predictors
#' @param k number of neighbors
#' @param method "brute" or "kdtree"
#' @param weights "uniform" or "distance"
#' @param distance "euclidean" or "manhattan"
#' @return predictions
#' @export
knn_predict <- function(X_train, y_train, X_test, k = 5,
                         method = "brute", weights = "uniform",
                         distance = "euclidean") {
  if (is.vector(X_train)) X_train <- matrix(X_train, ncol = 1)
  if (is.vector(X_test)) X_test <- matrix(X_test, ncol = 1)

  n_train <- nrow(X_train)
  n_test <- nrow(X_test)
  is_classification <- is.factor(y_train) || is.character(y_train)

  dist_fn <- if (distance == "euclidean") euclidean_dist else manhattan_dist

  if (method == "kdtree") {
    tree <- kdtree_build(X_train)
  }

  predictions <- if (is_classification) character(n_test) else numeric(n_test)

  for (i in seq_len(n_test)) {
    if (method == "brute") {
      # Compute all distances
      dists <- apply(X_train, 1, function(row) dist_fn(X_test[i, ], row))
      nn_idx <- order(dists)[seq_len(k)]
      nn_dists <- dists[nn_idx]
    } else {
      result <- kdtree_search(tree, X_test[i, ], k, dist_fn = dist_fn)
      ord <- order(result$distances)
      nn_idx <- result$indices[ord]
      nn_dists <- result$distances[ord]
    }

    nn_labels <- y_train[nn_idx]

    if (weights == "uniform") {
      if (is_classification) {
        predictions[i] <- names(which.max(table(nn_labels)))
      } else {
        predictions[i] <- mean(as.numeric(nn_labels))
      }
    } else {
      # Distance weighting (inverse distance)
      w <- 1 / (nn_dists + 1e-10)
      w <- w / sum(w)

      if (is_classification) {
        class_weights <- tapply(w, nn_labels, sum)
        predictions[i] <- names(which.max(class_weights))
      } else {
        predictions[i] <- sum(w * as.numeric(nn_labels))
      }
    }
  }

  if (!is_classification) predictions <- as.numeric(predictions)
  predictions
}

# =============================================================================
# SECTION 8: NAIVE BAYES
# =============================================================================

#' Gaussian Naive Bayes Classifier
#'
#' @param X predictor matrix
#' @param y class labels
#' @param laplace Laplace smoothing parameter
#' @return naive bayes model
#' @export
naive_bayes_gaussian <- function(X, y, laplace = 0) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  classes <- sort(unique(y))
  n <- nrow(X)
  p <- ncol(X)
  n_classes <- length(classes)

  # Prior probabilities
  priors <- table(y) / n

  # Class-conditional means and variances
  means <- matrix(0, nrow = n_classes, ncol = p)
  vars <- matrix(0, nrow = n_classes, ncol = p)

  for (c in seq_len(n_classes)) {
    idx <- which(y == classes[c])
    means[c, ] <- colMeans(X[idx, , drop = FALSE])
    vars[c, ] <- apply(X[idx, , drop = FALSE], 2, var) + laplace
  }

  structure(
    list(
      classes = classes,
      priors = as.numeric(priors),
      means = means,
      vars = vars,
      n_classes = n_classes,
      n_features = p
    ),
    class = "gaussian_nb"
  )
}

#' Predict method for Gaussian NB
#' @export
predict.gaussian_nb <- function(object, newdata, type = "class", ...) {
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  n <- nrow(newdata)

  log_probs <- matrix(0, nrow = n, ncol = object$n_classes)

  for (c in seq_len(object$n_classes)) {
    log_prior <- log(object$priors[c])
    for (j in seq_len(object$n_features)) {
      log_probs[, c] <- log_probs[, c] +
        dnorm(newdata[, j], mean = object$means[c, j],
              sd = sqrt(object$vars[c, j]), log = TRUE)
    }
    log_probs[, c] <- log_probs[, c] + log_prior
  }

  if (type == "class") {
    return(object$classes[apply(log_probs, 1, which.max)])
  }

  # Normalize to probabilities
  max_log <- apply(log_probs, 1, max)
  log_probs_shifted <- log_probs - max_log
  probs <- exp(log_probs_shifted)
  probs <- probs / rowSums(probs)
  colnames(probs) <- object$classes
  probs
}

#' Categorical (Multinomial) Naive Bayes
#'
#' @param X predictor matrix (integer features)
#' @param y class labels
#' @param laplace Laplace smoothing
#' @return categorical NB model
#' @export
naive_bayes_categorical <- function(X, y, laplace = 1) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  classes <- sort(unique(y))
  n <- nrow(X)
  p <- ncol(X)

  priors <- (table(y) + laplace) / (n + laplace * length(classes))

  # Feature-class conditional probabilities
  cond_probs <- vector("list", p)

  for (j in seq_len(p)) {
    levels_j <- sort(unique(X[, j]))
    cond_probs[[j]] <- list()

    for (c in seq_along(classes)) {
      idx <- which(y == classes[c])
      counts <- table(factor(X[idx, j], levels = levels_j))
      cond_probs[[j]][[c]] <- (as.numeric(counts) + laplace) /
        (length(idx) + laplace * length(levels_j))
      names(cond_probs[[j]][[c]]) <- levels_j
    }
  }

  structure(
    list(
      classes = classes,
      priors = as.numeric(priors),
      cond_probs = cond_probs,
      n_classes = length(classes),
      n_features = p
    ),
    class = "categorical_nb"
  )
}

#' Predict method for Categorical NB
#' @export
predict.categorical_nb <- function(object, newdata, type = "class", ...) {
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  n <- nrow(newdata)

  log_probs <- matrix(0, nrow = n, ncol = object$n_classes)

  for (c in seq_len(object$n_classes)) {
    log_probs[, c] <- log(object$priors[c])
    for (j in seq_len(object$n_features)) {
      for (i in seq_len(n)) {
        val <- as.character(newdata[i, j])
        prob <- object$cond_probs[[j]][[c]][val]
        if (is.na(prob)) prob <- 1e-10
        log_probs[i, c] <- log_probs[i, c] + log(prob)
      }
    }
  }

  if (type == "class") {
    return(object$classes[apply(log_probs, 1, which.max)])
  }

  max_log <- apply(log_probs, 1, max)
  probs <- exp(log_probs - max_log)
  probs <- probs / rowSums(probs)
  colnames(probs) <- object$classes
  probs
}

# =============================================================================
# SECTION 9: PCA
# =============================================================================

#' Principal Component Analysis (SVD-based)
#'
#' @param X data matrix (n x p)
#' @param n_components number of components to retain (NULL = all)
#' @param scale logical, whether to scale to unit variance
#' @return PCA model with scores, loadings, variance explained
#' @export
pca_analysis <- function(X, n_components = NULL, scale = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)

  # Center and scale
  X_means <- colMeans(X)
  X_centered <- sweep(X, 2, X_means)

  if (scale) {
    X_sds <- apply(X_centered, 2, sd)
    X_sds[X_sds == 0] <- 1
    X_centered <- sweep(X_centered, 2, X_sds, "/")
  } else {
    X_sds <- rep(1, p)
  }

  # SVD
  svd_result <- svd(X_centered)

  # Eigenvalues (proportional to variance)
  eigenvalues <- svd_result$d^2 / (n - 1)
  var_explained <- eigenvalues / sum(eigenvalues)
  cumulative_var <- cumsum(var_explained)

  # Components
  if (is.null(n_components)) {
    n_components <- min(n, p)
  }
  n_components <- min(n_components, length(eigenvalues))

  # Scores (projections)
  scores <- svd_result$u[, seq_len(n_components), drop = FALSE] %*%
    diag(svd_result$d[seq_len(n_components)], nrow = n_components)

  # Loadings
  loadings <- svd_result$v[, seq_len(n_components), drop = FALSE]

  # Kaiser criterion (eigenvalue > 1)
  kaiser_n <- sum(eigenvalues > 1)

  # Broken stick model
  broken_stick <- numeric(length(eigenvalues))
  for (j in seq_along(eigenvalues)) {
    broken_stick[j] <- sum(1 / (j:length(eigenvalues))) / length(eigenvalues)
  }

  structure(
    list(
      scores = scores,
      loadings = loadings,
      eigenvalues = eigenvalues,
      var_explained = var_explained,
      cumulative_var = cumulative_var,
      n_components = n_components,
      kaiser_n = kaiser_n,
      broken_stick = broken_stick,
      center = X_means,
      scale = X_sds,
      svd = svd_result,
      n = n,
      p = p
    ),
    class = "pca_model"
  )
}

#' Print method for PCA
#' @export
print.pca_model <- function(x, ...) {
  cat("Principal Component Analysis\n")
  cat("============================\n")
  cat("Components retained:", x$n_components, "\n")
  cat("Kaiser criterion suggests:", x$kaiser_n, "components\n\n")
  cat("Variance Explained:\n")
  df <- data.frame(
    PC = seq_len(min(10, length(x$eigenvalues))),
    Eigenvalue = x$eigenvalues[seq_len(min(10, length(x$eigenvalues)))],
    VarExplained = x$var_explained[seq_len(min(10, length(x$eigenvalues)))],
    Cumulative = x$cumulative_var[seq_len(min(10, length(x$eigenvalues)))]
  )
  print(round(df, 4))
  invisible(x)
}

#' Transform new data using PCA model
#' @export
predict.pca_model <- function(object, newdata, ...) {
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  newdata <- sweep(newdata, 2, object$center)
  newdata <- sweep(newdata, 2, object$scale, "/")
  newdata %*% object$loadings
}

#' Inverse transform from PC scores to original space
#' @export
pca_inverse_transform <- function(model, scores) {
  if (is.vector(scores)) scores <- matrix(scores, nrow = 1)
  reconstructed <- scores %*% t(model$loadings)
  reconstructed <- sweep(reconstructed, 2, model$scale, "*")
  sweep(reconstructed, 2, model$center, "+")
}

# =============================================================================
# SECTION 10: CLUSTERING
# =============================================================================

#' K-Means Clustering (Lloyd's Algorithm)
#'
#' @param X data matrix
#' @param k number of clusters
#' @param max_iter maximum iterations
#' @param n_init number of random initializations
#' @param init "random" or "kmeans++"
#' @return kmeans result
#' @export
kmeans_cluster <- function(X, k, max_iter = 300, n_init = 10,
                            init = "kmeans++") {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)

  best_inertia <- Inf
  best_result <- NULL

  for (run in seq_len(n_init)) {
    # Initialize centroids
    if (init == "kmeans++") {
      centroids <- kmeans_pp_init(X, k)
    } else {
      idx <- sample.int(n, k)
      centroids <- X[idx, , drop = FALSE]
    }

    for (iter in seq_len(max_iter)) {
      # Assignment step
      assignments <- assign_clusters(X, centroids)

      # Update step
      new_centroids <- matrix(0, nrow = k, ncol = p)
      for (c in seq_len(k)) {
        members <- which(assignments == c)
        if (length(members) == 0) {
          # Reinitialize empty cluster
          new_centroids[c, ] <- X[sample.int(n, 1), ]
        } else {
          new_centroids[c, ] <- colMeans(X[members, , drop = FALSE])
        }
      }

      # Check convergence
      if (max(abs(new_centroids - centroids)) < 1e-8) break
      centroids <- new_centroids
    }

    # Compute inertia (within-cluster sum of squares)
    inertia <- 0
    for (c in seq_len(k)) {
      members <- which(assignments == c)
      if (length(members) > 0) {
        diffs <- sweep(X[members, , drop = FALSE], 2, centroids[c, ])
        inertia <- inertia + sum(diffs^2)
      }
    }

    if (inertia < best_inertia) {
      best_inertia <- inertia
      best_result <- list(
        centroids = centroids,
        assignments = assignments,
        inertia = inertia,
        iterations = iter
      )
    }
  }

  # Compute cluster sizes
  sizes <- table(factor(best_result$assignments, levels = seq_len(k)))

  # Silhouette scores
  sil <- silhouette_scores(X, best_result$assignments, k)

  structure(
    c(best_result, list(
      k = k,
      sizes = as.integer(sizes),
      silhouette = sil$mean_silhouette,
      silhouette_per_point = sil$per_point,
      n = n,
      p = p
    )),
    class = "kmeans_result"
  )
}

#' K-Means++ initialization
#' @keywords internal
kmeans_pp_init <- function(X, k) {
  n <- nrow(X)
  p <- ncol(X)
  centroids <- matrix(0, nrow = k, ncol = p)

  # First centroid random
  centroids[1, ] <- X[sample.int(n, 1), ]

  for (c in 2:k) {
    # Compute distances to nearest centroid
    dists <- apply(X, 1, function(x) {
      min(apply(centroids[seq_len(c - 1), , drop = FALSE], 1,
                function(ctr) sum((x - ctr)^2)))
    })

    # Sample proportional to distance squared
    probs <- dists / sum(dists)
    centroids[c, ] <- X[sample.int(n, 1, prob = probs), ]
  }

  centroids
}

#' Assign points to nearest centroid
#' @keywords internal
assign_clusters <- function(X, centroids) {
  n <- nrow(X)
  k <- nrow(centroids)
  assignments <- integer(n)

  for (i in seq_len(n)) {
    dists <- apply(centroids, 1, function(c) sum((X[i, ] - c)^2))
    assignments[i] <- which.min(dists)
  }

  assignments
}

#' Compute silhouette scores
#' @keywords internal
silhouette_scores <- function(X, assignments, k) {
  n <- nrow(X)
  sil <- numeric(n)

  for (i in seq_len(n)) {
    cluster_i <- assignments[i]

    # a(i): mean distance to points in same cluster
    same_idx <- which(assignments == cluster_i)
    same_idx <- setdiff(same_idx, i)
    if (length(same_idx) == 0) {
      sil[i] <- 0
      next
    }
    a_i <- mean(apply(X[same_idx, , drop = FALSE], 1,
                       function(x) sqrt(sum((X[i, ] - x)^2))))

    # b(i): minimum mean distance to points in other clusters
    b_i <- Inf
    for (c in seq_len(k)) {
      if (c == cluster_i) next
      other_idx <- which(assignments == c)
      if (length(other_idx) == 0) next
      mean_dist <- mean(apply(X[other_idx, , drop = FALSE], 1,
                               function(x) sqrt(sum((X[i, ] - x)^2))))
      b_i <- min(b_i, mean_dist)
    }

    sil[i] <- (b_i - a_i) / max(a_i, b_i)
  }

  list(per_point = sil, mean_silhouette = mean(sil))
}

#' Hierarchical Clustering
#'
#' Agglomerative clustering with various linkage methods.
#'
#' @param X data matrix
#' @param k number of clusters to cut
#' @param method linkage method ("ward", "complete", "single", "average")
#' @return hierarchical clustering result
#' @export
hierarchical_cluster <- function(X, k = 2, method = "ward") {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  # Compute distance matrix
  dist_mat <- as.matrix(dist(X))

  # Initialize each point as its own cluster
  cluster_ids <- seq_len(n)
  cluster_members <- lapply(seq_len(n), function(i) i)
  merge_matrix <- matrix(0, nrow = n - 1, ncol = 2)
  heights <- numeric(n - 1)

  active_clusters <- seq_len(n)

  for (step in seq_len(n - 1)) {
    n_active <- length(active_clusters)
    if (n_active < 2) break

    # Find closest pair
    best_dist <- Inf
    best_i <- 0
    best_j <- 0

    for (ii in seq_len(n_active - 1)) {
      for (jj in (ii + 1):n_active) {
        ci <- active_clusters[ii]
        cj <- active_clusters[jj]
        members_i <- cluster_members[[ci]]
        members_j <- cluster_members[[cj]]

        d <- switch(method,
                     "single" = min(dist_mat[members_i, members_j]),
                     "complete" = max(dist_mat[members_i, members_j]),
                     "average" = mean(dist_mat[members_i, members_j]),
                     "ward" = {
                       ni <- length(members_i)
                       nj <- length(members_j)
                       center_i <- colMeans(X[members_i, , drop = FALSE])
                       center_j <- colMeans(X[members_j, , drop = FALSE])
                       (ni * nj) / (ni + nj) * sum((center_i - center_j)^2)
                     })

        if (d < best_dist) {
          best_dist <- d
          best_i <- ci
          best_j <- cj
        }
      }
    }

    # Merge clusters
    new_id <- n + step
    cluster_members[[new_id]] <- c(cluster_members[[best_i]],
                                    cluster_members[[best_j]])

    merge_matrix[step, ] <- c(
      ifelse(best_i <= n, -best_i, best_i - n),
      ifelse(best_j <= n, -best_j, best_j - n)
    )
    heights[step] <- best_dist

    active_clusters <- setdiff(active_clusters, c(best_i, best_j))
    active_clusters <- c(active_clusters, new_id)
  }

  # Cut tree at k clusters
  assignments <- cut_dendrogram(merge_matrix, n, k)

  structure(
    list(
      assignments = assignments,
      merge = merge_matrix,
      heights = heights,
      method = method,
      k = k,
      n = n
    ),
    class = "hclust_result"
  )
}

#' Cut dendrogram at k clusters
#' @keywords internal
cut_dendrogram <- function(merge_matrix, n, k) {
  n_merges <- nrow(merge_matrix)
  assignments <- seq_len(n)

  # Apply merges from last to first, stopping when we have k clusters
  # Actually, apply merges from first to last until we have k clusters
  current_labels <- seq_len(n)

  for (step in seq_len(n_merges)) {
    if (length(unique(current_labels)) <= k) break

    m1 <- merge_matrix[step, 1]
    m2 <- merge_matrix[step, 2]

    # Convert merge notation
    idx1 <- if (m1 < 0) which(current_labels == -m1) else {
      # Find all members of cluster formed at step m1
      which(current_labels %in% current_labels[current_labels == (n + m1)])
    }
    idx2 <- if (m2 < 0) which(current_labels == -m2) else {
      which(current_labels %in% current_labels[current_labels == (n + m2)])
    }

    new_label <- n + step
    current_labels[c(idx1, idx2)] <- new_label
  }

  # Remap to 1:k
  unique_labels <- unique(current_labels)
  mapping <- setNames(seq_along(unique_labels), unique_labels)
  as.integer(mapping[as.character(current_labels)])
}

#' DBSCAN Clustering
#'
#' Density-Based Spatial Clustering of Applications with Noise.
#'
#' @param X data matrix
#' @param eps neighborhood radius
#' @param min_pts minimum points for core point
#' @return DBSCAN result
#' @export
dbscan_cluster <- function(X, eps, min_pts = 5) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  # Compute distance matrix
  dist_mat <- as.matrix(dist(X))

  # Find neighbors for each point
  neighbors <- lapply(seq_len(n), function(i) {
    which(dist_mat[i, ] <= eps)
  })

  # Identify core points
  is_core <- sapply(neighbors, length) >= min_pts

  # Cluster assignment (-1 = noise)
  assignments <- rep(-1L, n)
  cluster_id <- 0L
  visited <- logical(n)

  for (i in seq_len(n)) {
    if (visited[i]) next
    if (!is_core[i]) next

    visited[i] <- TRUE
    cluster_id <- cluster_id + 1L
    assignments[i] <- cluster_id

    # BFS to expand cluster
    queue <- neighbors[[i]]
    queue <- setdiff(queue, i)

    while (length(queue) > 0) {
      j <- queue[1]
      queue <- queue[-1]

      if (!visited[j]) {
        visited[j] <- TRUE
        if (is_core[j]) {
          new_neighbors <- setdiff(neighbors[[j]], which(visited))
          queue <- c(queue, new_neighbors)
        }
      }

      if (assignments[j] == -1L) {
        assignments[j] <- cluster_id
      }
    }
  }

  structure(
    list(
      assignments = assignments,
      n_clusters = cluster_id,
      n_noise = sum(assignments == -1),
      core_points = which(is_core),
      eps = eps,
      min_pts = min_pts,
      n = n
    ),
    class = "dbscan_result"
  )
}

#' Gaussian Mixture Model via EM
#'
#' @param X data matrix
#' @param k number of components
#' @param max_iter maximum EM iterations
#' @param tol convergence tolerance
#' @param n_init number of initializations
#' @return GMM model
#' @export
gmm_cluster <- function(X, k, max_iter = 200, tol = 1e-6, n_init = 5) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  p <- ncol(X)

  best_ll <- -Inf
  best_result <- NULL

  for (run in seq_len(n_init)) {
    # Initialize with K-means
    km <- kmeans_cluster(X, k, n_init = 1)

    # Initialize parameters
    weights <- km$sizes / n
    means <- km$centroids
    covs <- vector("list", k)
    for (c in seq_len(k)) {
      members <- which(km$assignments == c)
      if (length(members) > p) {
        covs[[c]] <- cov(X[members, , drop = FALSE]) + diag(1e-6, p)
      } else {
        covs[[c]] <- diag(1, p)
      }
    }

    log_lik_prev <- -Inf

    for (iter in seq_len(max_iter)) {
      # E-step: compute responsibilities
      resp <- matrix(0, nrow = n, ncol = k)

      for (c in seq_len(k)) {
        diff <- sweep(X, 2, means[c, ])
        cov_inv <- tryCatch(solve(covs[[c]]),
                             error = function(e) solve(covs[[c]] + diag(1e-4, p)))
        log_det <- determinant(covs[[c]], logarithm = TRUE)$modulus[1]

        mahal <- rowSums((diff %*% cov_inv) * diff)
        resp[, c] <- log(weights[c]) - 0.5 * (p * log(2 * pi) +
                                                   log_det + mahal)
      }

      # Log-sum-exp trick
      max_resp <- apply(resp, 1, max)
      resp <- resp - max_resp
      resp <- exp(resp)
      row_sums <- rowSums(resp)
      log_lik <- sum(max_resp + log(row_sums))
      resp <- resp / row_sums

      # Check convergence
      if (abs(log_lik - log_lik_prev) < tol * abs(log_lik_prev)) break
      log_lik_prev <- log_lik

      # M-step
      Nk <- colSums(resp)

      for (c in seq_len(k)) {
        weights[c] <- Nk[c] / n
        means[c, ] <- colSums(resp[, c] * X) / Nk[c]

        diff <- sweep(X, 2, means[c, ])
        covs[[c]] <- crossprod(diff * sqrt(resp[, c]), diff *
                                  sqrt(resp[, c])) / Nk[c] + diag(1e-6, p)
      }
    }

    if (log_lik > best_ll) {
      best_ll <- log_lik
      best_result <- list(
        weights = weights,
        means = means,
        covs = covs,
        responsibilities = resp,
        log_likelihood = log_lik,
        iterations = iter
      )
    }
  }

  # BIC and AIC
  n_params <- k - 1 + k * p + k * p * (p + 1) / 2
  aic <- -2 * best_result$log_likelihood + 2 * n_params
  bic <- -2 * best_result$log_likelihood + log(n) * n_params

  assignments <- apply(best_result$responsibilities, 1, which.max)

  structure(
    c(best_result, list(
      assignments = assignments,
      k = k,
      aic = aic,
      bic = bic,
      n = n,
      p = p
    )),
    class = "gmm_result"
  )
}

# =============================================================================
# SECTION 11: CROSS-VALIDATION
# =============================================================================

#' K-Fold Cross-Validation
#'
#' @param X predictor matrix
#' @param y response
#' @param model_fn function(X_train, y_train, ...) returning model
#' @param predict_fn function(model, X_test) returning predictions
#' @param metric_fn function(y_true, y_pred) returning score
#' @param k number of folds
#' @param ... additional arguments to model_fn
#' @return CV results
#' @export
kfold_cv <- function(X, y, model_fn, predict_fn, metric_fn,
                      k = 10, ...) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  fold_ids <- sample(rep(seq_len(k), length.out = n))
  scores <- numeric(k)
  predictions <- vector("list", k)

  for (fold in seq_len(k)) {
    test_idx <- which(fold_ids == fold)
    train_idx <- which(fold_ids != fold)

    model <- model_fn(X[train_idx, , drop = FALSE], y[train_idx], ...)
    preds <- predict_fn(model, X[test_idx, , drop = FALSE])

    scores[fold] <- metric_fn(y[test_idx], preds)
    predictions[[fold]] <- list(idx = test_idx, preds = preds)
  }

  list(
    scores = scores,
    mean_score = mean(scores),
    se_score = sd(scores) / sqrt(k),
    predictions = predictions
  )
}

#' Stratified K-Fold Cross-Validation
#'
#' Maintains class proportions in each fold.
#'
#' @param X predictor matrix
#' @param y class labels
#' @param model_fn model training function
#' @param predict_fn prediction function
#' @param metric_fn evaluation metric function
#' @param k number of folds
#' @param ... additional arguments
#' @return CV results
#' @export
stratified_kfold_cv <- function(X, y, model_fn, predict_fn, metric_fn,
                                  k = 10, ...) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  # Stratified fold assignment
  fold_ids <- integer(n)
  classes <- unique(y)

  for (cls in classes) {
    idx <- which(y == cls)
    fold_ids[idx] <- sample(rep(seq_len(k), length.out = length(idx)))
  }

  scores <- numeric(k)

  for (fold in seq_len(k)) {
    test_idx <- which(fold_ids == fold)
    train_idx <- which(fold_ids != fold)

    model <- model_fn(X[train_idx, , drop = FALSE], y[train_idx], ...)
    preds <- predict_fn(model, X[test_idx, , drop = FALSE])
    scores[fold] <- metric_fn(y[test_idx], preds)
  }

  list(
    scores = scores,
    mean_score = mean(scores),
    se_score = sd(scores) / sqrt(k)
  )
}

#' Time Series Cross-Validation (Rolling Window)
#'
#' @param X predictor matrix (ordered by time)
#' @param y response
#' @param model_fn model training function
#' @param predict_fn prediction function
#' @param metric_fn evaluation metric
#' @param initial_window initial training window size
#' @param horizon forecast horizon
#' @param step step size between windows
#' @param expanding logical, expanding or sliding window
#' @return CV results
#' @export
ts_cv <- function(X, y, model_fn, predict_fn, metric_fn,
                   initial_window, horizon = 1, step = 1,
                   expanding = TRUE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  starts <- seq(initial_window + 1, n - horizon + 1, by = step)
  scores <- numeric(length(starts))

  for (i in seq_along(starts)) {
    test_start <- starts[i]
    test_end <- min(test_start + horizon - 1, n)
    test_idx <- test_start:test_end

    if (expanding) {
      train_idx <- seq_len(test_start - 1)
    } else {
      train_start <- max(1, test_start - initial_window)
      train_idx <- train_start:(test_start - 1)
    }

    model <- model_fn(X[train_idx, , drop = FALSE], y[train_idx])
    preds <- predict_fn(model, X[test_idx, , drop = FALSE])
    scores[i] <- metric_fn(y[test_idx], preds)
  }

  list(
    scores = scores,
    mean_score = mean(scores),
    se_score = sd(scores) / sqrt(length(scores))
  )
}

#' Nested Cross-Validation
#'
#' Outer CV for model evaluation, inner CV for hyperparameter selection.
#'
#' @param X predictor matrix
#' @param y response
#' @param model_fn function(X, y, params) -> model
#' @param predict_fn prediction function
#' @param metric_fn evaluation metric
#' @param param_grid list of parameter vectors to search
#' @param outer_k outer CV folds
#' @param inner_k inner CV folds
#' @return nested CV results
#' @export
nested_cv <- function(X, y, model_fn, predict_fn, metric_fn,
                       param_grid, outer_k = 5, inner_k = 5) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  outer_folds <- sample(rep(seq_len(outer_k), length.out = n))
  outer_scores <- numeric(outer_k)
  best_params_list <- vector("list", outer_k)

  for (fold in seq_len(outer_k)) {
    test_idx <- which(outer_folds == fold)
    train_idx <- which(outer_folds != fold)

    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test <- X[test_idx, , drop = FALSE]
    y_test <- y[test_idx]

    # Inner CV to select best parameters
    n_params <- length(param_grid)
    inner_scores <- numeric(n_params)

    for (p_idx in seq_len(n_params)) {
      params <- param_grid[[p_idx]]

      inner_cv_result <- kfold_cv(
        X_train, y_train,
        model_fn = function(X, y, ...) model_fn(X, y, params),
        predict_fn = predict_fn,
        metric_fn = metric_fn,
        k = inner_k
      )
      inner_scores[p_idx] <- inner_cv_result$mean_score
    }

    best_param_idx <- which.max(inner_scores)
    best_params <- param_grid[[best_param_idx]]
    best_params_list[[fold]] <- best_params

    # Train with best params on full outer training set
    model <- model_fn(X_train, y_train, best_params)
    preds <- predict_fn(model, X_test)
    outer_scores[fold] <- metric_fn(y_test, preds)
  }

  list(
    scores = outer_scores,
    mean_score = mean(outer_scores),
    se_score = sd(outer_scores) / sqrt(outer_k),
    best_params = best_params_list
  )
}

# =============================================================================
# SECTION 12: EVALUATION METRICS
# =============================================================================

#' Accuracy
#' @export
metric_accuracy <- function(y_true, y_pred) {
  mean(y_true == y_pred)
}

#' Precision (per-class or macro-averaged)
#' @export
metric_precision <- function(y_true, y_pred, average = "macro") {
  classes <- unique(c(y_true, y_pred))
  precisions <- sapply(classes, function(cls) {
    tp <- sum(y_pred == cls & y_true == cls)
    fp <- sum(y_pred == cls & y_true != cls)
    if (tp + fp == 0) return(0)
    tp / (tp + fp)
  })

  if (average == "macro") return(mean(precisions))
  if (average == "weighted") {
    weights <- table(y_true)[as.character(classes)] / length(y_true)
    return(sum(precisions * weights))
  }
  setNames(precisions, classes)
}

#' Recall (per-class or macro-averaged)
#' @export
metric_recall <- function(y_true, y_pred, average = "macro") {
  classes <- unique(c(y_true, y_pred))
  recalls <- sapply(classes, function(cls) {
    tp <- sum(y_pred == cls & y_true == cls)
    fn <- sum(y_pred != cls & y_true == cls)
    if (tp + fn == 0) return(0)
    tp / (tp + fn)
  })

  if (average == "macro") return(mean(recalls))
  if (average == "weighted") {
    weights <- table(y_true)[as.character(classes)] / length(y_true)
    return(sum(recalls * weights))
  }
  setNames(recalls, classes)
}

#' F1 Score
#' @export
metric_f1 <- function(y_true, y_pred, average = "macro") {
  p <- metric_precision(y_true, y_pred, average = "none")
  r <- metric_recall(y_true, y_pred, average = "none")

  f1 <- 2 * p * r / (p + r)
  f1[is.nan(f1)] <- 0

  if (average == "macro") return(mean(f1))
  if (average == "weighted") {
    classes <- names(f1)
    weights <- table(y_true)[classes] / length(y_true)
    return(sum(f1 * weights))
  }
  f1
}

#' AUC-ROC (binary classification)
#'
#' @param y_true binary labels (0/1)
#' @param y_scores predicted probabilities
#' @return AUC value
#' @export
metric_auc_roc <- function(y_true, y_scores) {
  n <- length(y_true)
  ord <- order(y_scores, decreasing = TRUE)
  y_sorted <- y_true[ord]

  n_pos <- sum(y_true == 1)
  n_neg <- n - n_pos

  if (n_pos == 0 || n_neg == 0) return(NA)

  tpr <- cumsum(y_sorted == 1) / n_pos
  fpr <- cumsum(y_sorted == 0) / n_neg

  # Prepend origin
  tpr <- c(0, tpr)
  fpr <- c(0, fpr)

  # Trapezoidal rule
  auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  auc
}

#' ROC curve points
#' @export
roc_curve <- function(y_true, y_scores, n_thresholds = 200) {
  thresholds <- seq(0, 1, length.out = n_thresholds)
  tpr <- numeric(n_thresholds)
  fpr <- numeric(n_thresholds)

  n_pos <- sum(y_true == 1)
  n_neg <- sum(y_true == 0)

  for (i in seq_along(thresholds)) {
    predicted <- as.integer(y_scores >= thresholds[i])
    tp <- sum(predicted == 1 & y_true == 1)
    fp <- sum(predicted == 1 & y_true == 0)
    tpr[i] <- tp / n_pos
    fpr[i] <- fp / n_neg
  }

  data.frame(threshold = thresholds, fpr = fpr, tpr = tpr)
}

#' Log Loss (binary cross-entropy)
#' @export
metric_log_loss <- function(y_true, y_prob) {
  y_prob <- pmin(pmax(y_prob, 1e-15), 1 - 1e-15)
  -mean(y_true * log(y_prob) + (1 - y_true) * log(1 - y_prob))
}

#' Mean Squared Error
#' @export
metric_mse <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}

#' Root Mean Squared Error
#' @export
metric_rmse <- function(y_true, y_pred) {
  sqrt(mean((y_true - y_pred)^2))
}

#' Mean Absolute Error
#' @export
metric_mae <- function(y_true, y_pred) {
  mean(abs(y_true - y_pred))
}

#' R-squared (coefficient of determination)
#' @export
metric_r_squared <- function(y_true, y_pred) {
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  1 - ss_res / ss_tot
}

#' Mean Absolute Percentage Error
#' @export
metric_mape <- function(y_true, y_pred) {
  valid <- y_true != 0
  mean(abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100
}

#' Confusion matrix with detailed metrics
#' @export
confusion_matrix <- function(y_true, y_pred) {
  classes <- sort(unique(c(y_true, y_pred)))
  cm <- table(Actual = factor(y_true, levels = classes),
              Predicted = factor(y_pred, levels = classes))

  n <- length(y_true)
  accuracy <- sum(diag(cm)) / n

  # Per-class metrics
  per_class <- data.frame(
    Class = classes,
    Precision = sapply(classes, function(c) {
      tp <- cm[c, c]
      fp <- sum(cm[, c]) - tp
      if (tp + fp == 0) 0 else tp / (tp + fp)
    }),
    Recall = sapply(classes, function(c) {
      tp <- cm[c, c]
      fn <- sum(cm[c, ]) - tp
      if (tp + fn == 0) 0 else tp / (tp + fn)
    }),
    F1 = NA,
    Support = as.integer(rowSums(cm))
  )
  per_class$F1 <- with(per_class, 2 * Precision * Recall /
                          (Precision + Recall))
  per_class$F1[is.nan(per_class$F1)] <- 0

  list(
    matrix = cm,
    accuracy = accuracy,
    per_class = per_class,
    macro_precision = mean(per_class$Precision),
    macro_recall = mean(per_class$Recall),
    macro_f1 = mean(per_class$F1)
  )
}

# =============================================================================
# SECTION 13: FEATURE SELECTION
# =============================================================================

#' Forward Stepwise Selection
#'
#' @param X predictor matrix
#' @param y response
#' @param model_fn function(X, y) -> model
#' @param metric_fn function(model) -> score (higher is better)
#' @param max_features maximum features to select
#' @return selected feature indices and scores
#' @export
forward_stepwise <- function(X, y, model_fn = NULL, metric_fn = NULL,
                              max_features = NULL) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  p <- ncol(X)

  if (is.null(max_features)) max_features <- p

  if (is.null(model_fn)) {
    model_fn <- function(X, y) ols_regression(X, y)
  }
  if (is.null(metric_fn)) {
    metric_fn <- function(m) -m$aic
  }

  selected <- integer(0)
  remaining <- seq_len(p)
  scores <- numeric(0)

  for (step in seq_len(max_features)) {
    best_score <- -Inf
    best_feature <- NULL

    for (j in remaining) {
      candidate <- c(selected, j)
      model <- model_fn(X[, candidate, drop = FALSE], y)
      score <- metric_fn(model)

      if (score > best_score) {
        best_score <- score
        best_feature <- j
      }
    }

    if (is.null(best_feature)) break

    selected <- c(selected, best_feature)
    remaining <- setdiff(remaining, best_feature)
    scores <- c(scores, best_score)
  }

  list(
    selected = selected,
    scores = scores,
    n_selected = length(selected)
  )
}

#' Backward Elimination
#'
#' @param X predictor matrix
#' @param y response
#' @param model_fn model training function
#' @param metric_fn scoring function (higher = better)
#' @param min_features minimum features to retain
#' @return elimination sequence and scores
#' @export
backward_elimination <- function(X, y, model_fn = NULL, metric_fn = NULL,
                                   min_features = 1) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  p <- ncol(X)

  if (is.null(model_fn)) {
    model_fn <- function(X, y) ols_regression(X, y)
  }
  if (is.null(metric_fn)) {
    metric_fn <- function(m) -m$aic
  }

  current_features <- seq_len(p)
  eliminated <- integer(0)
  scores <- numeric(0)

  # Score with all features
  full_model <- model_fn(X[, current_features, drop = FALSE], y)
  current_score <- metric_fn(full_model)
  scores <- c(scores, current_score)

  while (length(current_features) > min_features) {
    best_score <- -Inf
    worst_feature <- NULL

    for (j in current_features) {
      candidate <- setdiff(current_features, j)
      if (length(candidate) == 0) next

      model <- model_fn(X[, candidate, drop = FALSE], y)
      score <- metric_fn(model)

      if (score > best_score) {
        best_score <- score
        worst_feature <- j
      }
    }

    if (best_score <= current_score && length(eliminated) > 0) break

    current_features <- setdiff(current_features, worst_feature)
    eliminated <- c(eliminated, worst_feature)
    scores <- c(scores, best_score)
    current_score <- best_score
  }

  list(
    selected = current_features,
    eliminated = eliminated,
    scores = scores
  )
}

#' Recursive Feature Elimination
#'
#' @param X predictor matrix
#' @param y response
#' @param model_fn model function that returns object with var_importance
#' @param n_features_to_select target number of features
#' @param step number of features to remove per step
#' @return RFE results
#' @export
rfe_selection <- function(X, y, model_fn = NULL,
                           n_features_to_select = NULL, step = 1) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  p <- ncol(X)

  if (is.null(n_features_to_select)) {
    n_features_to_select <- max(1, floor(p / 2))
  }

  if (is.null(model_fn)) {
    model_fn <- function(X, y) {
      rf <- random_forest(X, y, n_trees = 50, oob = TRUE)
      list(importance = rf$var_importance, oob_error = rf$oob_error)
    }
  }

  current_features <- seq_len(p)
  ranking <- integer(0)
  history <- list()

  while (length(current_features) > n_features_to_select) {
    result <- model_fn(X[, current_features, drop = FALSE], y)
    imp <- result$importance

    history <- c(history, list(list(
      n_features = length(current_features),
      features = current_features,
      score = result$oob_error
    )))

    # Remove least important features
    n_remove <- min(step, length(current_features) - n_features_to_select)
    remove_idx <- order(imp)[seq_len(n_remove)]
    removed <- current_features[remove_idx]
    ranking <- c(ranking, removed)
    current_features <- current_features[-remove_idx]
  }

  list(
    selected = current_features,
    ranking = c(rev(current_features), ranking),
    history = history,
    n_selected = length(current_features)
  )
}

# =============================================================================
# SECTION 14: HYPERPARAMETER TUNING
# =============================================================================

#' Grid Search with Cross-Validation
#'
#' @param X predictor matrix
#' @param y response
#' @param model_fn function(X, y, params) -> model
#' @param predict_fn prediction function
#' @param metric_fn evaluation metric
#' @param param_grid named list of parameter vectors
#' @param cv_folds number of CV folds
#' @return grid search results
#' @export
grid_search <- function(X, y, model_fn, predict_fn, metric_fn,
                         param_grid, cv_folds = 5) {
  # Generate all combinations
  param_names <- names(param_grid)
  combinations <- expand.grid(param_grid, stringsAsFactors = FALSE)
  n_combos <- nrow(combinations)

  results <- data.frame(combinations)
  results$mean_score <- NA
  results$se_score <- NA

  best_score <- -Inf
  best_params <- NULL
  best_idx <- NULL

  for (i in seq_len(n_combos)) {
    params <- as.list(combinations[i, ])

    cv_result <- kfold_cv(
      X, y,
      model_fn = function(X, y, ...) model_fn(X, y, params),
      predict_fn = predict_fn,
      metric_fn = metric_fn,
      k = cv_folds
    )

    results$mean_score[i] <- cv_result$mean_score
    results$se_score[i] <- cv_result$se_score

    if (cv_result$mean_score > best_score) {
      best_score <- cv_result$mean_score
      best_params <- params
      best_idx <- i
    }
  }

  list(
    results = results,
    best_params = best_params,
    best_score = best_score,
    best_idx = best_idx
  )
}

#' Random Search with Cross-Validation
#'
#' @param X predictor matrix
#' @param y response
#' @param model_fn model function
#' @param predict_fn prediction function
#' @param metric_fn evaluation metric
#' @param param_distributions named list of sampling functions
#' @param n_iter number of random samples
#' @param cv_folds number of CV folds
#' @return random search results
#' @export
random_search <- function(X, y, model_fn, predict_fn, metric_fn,
                           param_distributions, n_iter = 20,
                           cv_folds = 5) {
  param_names <- names(param_distributions)
  results <- vector("list", n_iter)

  best_score <- -Inf
  best_params <- NULL

  for (i in seq_len(n_iter)) {
    # Sample parameters
    params <- lapply(param_distributions, function(fn) fn())

    cv_result <- kfold_cv(
      X, y,
      model_fn = function(X, y, ...) model_fn(X, y, params),
      predict_fn = predict_fn,
      metric_fn = metric_fn,
      k = cv_folds
    )

    results[[i]] <- list(
      params = params,
      mean_score = cv_result$mean_score,
      se_score = cv_result$se_score
    )

    if (cv_result$mean_score > best_score) {
      best_score <- cv_result$mean_score
      best_params <- params
    }
  }

  list(
    results = results,
    best_params = best_params,
    best_score = best_score,
    n_iter = n_iter
  )
}

# =============================================================================
# SECTION 15: ENSEMBLE METHODS
# =============================================================================

#' Stacking Ensemble
#'
#' Train base models and a meta-learner on their predictions.
#'
#' @param X predictor matrix
#' @param y response
#' @param base_models list of model training functions
#' @param base_predict list of prediction functions
#' @param meta_model meta-learner training function
#' @param meta_predict meta-learner prediction function
#' @param cv_folds folds for generating meta-features
#' @return stacking ensemble model
#' @export
stacking_ensemble <- function(X, y, base_models, base_predict,
                               meta_model = NULL, meta_predict = NULL,
                               cv_folds = 5) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  n_base <- length(base_models)

  if (is.null(meta_model)) {
    meta_model <- function(X, y) ols_regression(X, y)
  }
  if (is.null(meta_predict)) {
    meta_predict <- function(m, X) predict.ols_model(m, X)
  }

  # Generate meta-features via CV
  meta_features <- matrix(0, nrow = n, ncol = n_base)
  fold_ids <- sample(rep(seq_len(cv_folds), length.out = n))

  for (fold in seq_len(cv_folds)) {
    test_idx <- which(fold_ids == fold)
    train_idx <- which(fold_ids != fold)

    for (b in seq_len(n_base)) {
      model_b <- base_models[[b]](X[train_idx, , drop = FALSE],
                                   y[train_idx])
      meta_features[test_idx, b] <- base_predict[[b]](
        model_b, X[test_idx, , drop = FALSE])
    }
  }

  # Train meta-learner
  meta_trained <- meta_model(meta_features, y)

  # Train final base models on full data
  final_base_models <- lapply(seq_len(n_base), function(b) {
    base_models[[b]](X, y)
  })

  structure(
    list(
      base_models = final_base_models,
      base_predict = base_predict,
      meta_model = meta_trained,
      meta_predict = meta_predict,
      n_base = n_base
    ),
    class = "stacking_model"
  )
}

#' Predict with stacking ensemble
#' @export
predict.stacking_model <- function(object, newdata, ...) {
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  n <- nrow(newdata)

  meta_features <- matrix(0, nrow = n, ncol = object$n_base)
  for (b in seq_len(object$n_base)) {
    meta_features[, b] <- object$base_predict[[b]](
      object$base_models[[b]], newdata)
  }

  object$meta_predict(object$meta_model, meta_features)
}

#' Blending Ensemble
#'
#' Simpler than stacking: uses a holdout set for meta-features.
#'
#' @param X predictor matrix
#' @param y response
#' @param base_models list of model functions
#' @param base_predict list of predict functions
#' @param meta_model meta-learner
#' @param meta_predict meta prediction function
#' @param holdout_frac fraction of data for meta-feature generation
#' @return blending model
#' @export
blending_ensemble <- function(X, y, base_models, base_predict,
                               meta_model = NULL, meta_predict = NULL,
                               holdout_frac = 0.2) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)
  n_base <- length(base_models)

  if (is.null(meta_model)) {
    meta_model <- function(X, y) ols_regression(X, y)
  }
  if (is.null(meta_predict)) {
    meta_predict <- function(m, X) predict.ols_model(m, X)
  }

  # Split into train and holdout
  holdout_idx <- sample.int(n, floor(n * holdout_frac))
  train_idx <- setdiff(seq_len(n), holdout_idx)

  # Train base models on training set
  trained_bases <- lapply(seq_len(n_base), function(b) {
    base_models[[b]](X[train_idx, , drop = FALSE], y[train_idx])
  })

  # Generate meta-features on holdout set
  meta_features <- matrix(0, nrow = length(holdout_idx), ncol = n_base)
  for (b in seq_len(n_base)) {
    meta_features[, b] <- base_predict[[b]](
      trained_bases[[b]], X[holdout_idx, , drop = FALSE])
  }

  # Train meta-learner
  meta_trained <- meta_model(meta_features, y[holdout_idx])

  # Retrain base models on full data
  final_bases <- lapply(seq_len(n_base), function(b) {
    base_models[[b]](X, y)
  })

  structure(
    list(
      base_models = final_bases,
      base_predict = base_predict,
      meta_model = meta_trained,
      meta_predict = meta_predict,
      n_base = n_base
    ),
    class = "blending_model"
  )
}

#' Model Averaging Ensemble
#'
#' @param models list of trained models
#' @param predict_fns list of prediction functions
#' @param weights optional weights (default uniform)
#' @return averaging ensemble
#' @export
model_averaging <- function(models, predict_fns, weights = NULL) {
  n_models <- length(models)
  if (is.null(weights)) weights <- rep(1 / n_models, n_models)

  structure(
    list(
      models = models,
      predict_fns = predict_fns,
      weights = weights,
      n_models = n_models
    ),
    class = "averaging_model"
  )
}

#' Predict with model averaging
#' @export
predict.averaging_model <- function(object, newdata, ...) {
  if (is.vector(newdata)) newdata <- matrix(newdata, ncol = 1)
  n <- nrow(newdata)

  preds <- matrix(0, nrow = n, ncol = object$n_models)
  for (m in seq_len(object$n_models)) {
    preds[, m] <- object$predict_fns[[m]](object$models[[m]], newdata)
  }

  as.numeric(preds %*% object$weights)
}

# =============================================================================
# SECTION 16: APPLICATION WRAPPERS
# =============================================================================

#' Train a Classifier
#'
#' Unified interface for training classification models.
#'
#' @param X predictor matrix
#' @param y class labels
#' @param method algorithm name
#' @param ... additional arguments passed to specific method
#' @return trained model
#' @export
train_classifier <- function(X, y, method = "random_forest", ...) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)

  y <- factor(y)

  model <- switch(method,
    "logistic" = logistic_regression(X, y = as.integer(y) - 1, ...),
    "decision_tree" = decision_tree(X, y, ...),
    "random_forest" = random_forest(X, y, ...),
    "gradient_boosting" = gradient_boosting(X, y, loss = "deviance", ...),
    "svm" = svm_train(X, y, ...),
    "knn" = list(X_train = X, y_train = y,
                  method = "knn", class = "knn_wrapper"),
    "naive_bayes" = naive_bayes_gaussian(X, y, ...),
    stop("Unknown method: ", method)
  )

  model$method <- method
  model
}

#' Train a Regressor
#'
#' Unified interface for training regression models.
#'
#' @param X predictor matrix
#' @param y numeric response
#' @param method algorithm name
#' @param ... additional arguments
#' @return trained model
#' @export
train_regressor <- function(X, y, method = "random_forest", ...) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)

  model <- switch(method,
    "ols" = ols_regression(X, y, ...),
    "ridge" = ridge_regression(X, y, ...),
    "lasso" = lasso_regression(X, y, ...),
    "elastic_net" = elastic_net(X, y, ...),
    "decision_tree" = decision_tree(X, y, ...),
    "random_forest" = random_forest(X, y, ...),
    "gradient_boosting" = gradient_boosting(X, y, loss = "squared", ...),
    "svm" = svm_train(X, y, ...),
    stop("Unknown method: ", method)
  )

  model$method <- method
  model
}

#' Evaluate a Model
#'
#' Comprehensive model evaluation with multiple metrics.
#'
#' @param model trained model
#' @param X_test test predictors
#' @param y_test test response
#' @param type "classification" or "regression"
#' @return list of metrics
#' @export
evaluate_model <- function(model, X_test, y_test,
                            type = "classification") {
  if (is.vector(X_test)) X_test <- matrix(X_test, ncol = 1)

  # Get predictions based on model class
  model_class <- class(model)[1]

  predictions <- switch(model_class,
    "ols_model" = predict.ols_model(model, X_test),
    "logistic_model" = predict.logistic_model(model, X_test, type = "class"),
    "cart_model" = predict.cart_model(model, X_test),
    "random_forest_model" = {
      if (model$is_classification) {
        predict.random_forest_model(model, X_test, type = "class")
      } else {
        predict.random_forest_model(model, X_test)
      }
    },
    "gbm_model" = {
      if (model$is_classification) {
        predict.gbm_model(model, X_test, type = "class")
      } else {
        predict.gbm_model(model, X_test)
      }
    },
    "svm_model" = predict.svm_model(model, X_test),
    "gaussian_nb" = predict.gaussian_nb(model, X_test),
    stop("Unknown model class: ", model_class)
  )

  if (type == "classification") {
    cm <- confusion_matrix(y_test, predictions)
    auc <- tryCatch({
      if (model_class == "logistic_model") {
        probs <- predict.logistic_model(model, X_test, type = "response")
        metric_auc_roc(as.integer(y_test) - 1, probs)
      } else if (model_class == "random_forest_model") {
        probs <- predict.random_forest_model(model, X_test, type = "prob")
        if (ncol(probs) == 2) {
          metric_auc_roc(as.integer(factor(y_test)) - 1, probs[, 2])
        } else {
          NA
        }
      } else {
        NA
      }
    }, error = function(e) NA)

    return(list(
      accuracy = cm$accuracy,
      precision = cm$macro_precision,
      recall = cm$macro_recall,
      f1 = cm$macro_f1,
      auc = auc,
      confusion_matrix = cm$matrix,
      per_class = cm$per_class
    ))
  } else {
    return(list(
      mse = metric_mse(y_test, predictions),
      rmse = metric_rmse(y_test, predictions),
      mae = metric_mae(y_test, predictions),
      r_squared = metric_r_squared(y_test, predictions),
      mape = metric_mape(y_test, predictions)
    ))
  }
}

#' Train-Test Split
#'
#' @param X predictor matrix
#' @param y response
#' @param test_size fraction for test set
#' @param stratify logical, stratified split for classification
#' @return list with train/test splits
#' @export
train_test_split <- function(X, y, test_size = 0.2, stratify = FALSE) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)
  n <- nrow(X)

  if (stratify) {
    classes <- unique(y)
    test_idx <- integer(0)
    for (cls in classes) {
      cls_idx <- which(y == cls)
      n_test <- max(1, round(length(cls_idx) * test_size))
      test_idx <- c(test_idx, sample(cls_idx, n_test))
    }
  } else {
    n_test <- round(n * test_size)
    test_idx <- sample.int(n, n_test)
  }

  train_idx <- setdiff(seq_len(n), test_idx)

  list(
    X_train = X[train_idx, , drop = FALSE],
    y_train = y[train_idx],
    X_test = X[test_idx, , drop = FALSE],
    y_test = y[test_idx],
    train_idx = train_idx,
    test_idx = test_idx
  )
}

#' Complete ML Pipeline
#'
#' End-to-end pipeline: split, train, evaluate, with optional CV.
#'
#' @param X predictor matrix
#' @param y response
#' @param method ML algorithm
#' @param type "classification" or "regression"
#' @param test_size test fraction
#' @param cv_folds if > 0, perform CV
#' @param ... additional model arguments
#' @return pipeline results
#' @export
ml_pipeline <- function(X, y, method = "random_forest",
                         type = "classification", test_size = 0.2,
                         cv_folds = 0, ...) {
  if (is.vector(X)) X <- matrix(X, ncol = 1)

  # Split data
  split <- train_test_split(X, y, test_size = test_size,
                             stratify = (type == "classification"))

  # Train model
  if (type == "classification") {
    model <- train_classifier(split$X_train, split$y_train,
                               method = method, ...)
  } else {
    model <- train_regressor(split$X_train, split$y_train,
                              method = method, ...)
  }

  # Evaluate
  eval_result <- evaluate_model(model, split$X_test, split$y_test,
                                 type = type)

  # Optional cross-validation
  cv_result <- NULL
  if (cv_folds > 0) {
    metric_fn <- if (type == "classification") metric_accuracy else {
      function(y_true, y_pred) -metric_mse(y_true, y_pred)
    }

    cv_result <- kfold_cv(
      split$X_train, split$y_train,
      model_fn = function(X, y, ...) {
        if (type == "classification") {
          train_classifier(X, y, method = method)
        } else {
          train_regressor(X, y, method = method)
        }
      },
      predict_fn = function(model, X) {
        model_class <- class(model)[1]
        if (type == "classification") {
          switch(model_class,
            "random_forest_model" = predict.random_forest_model(model, X,
                                                                 type = "class"),
            "cart_model" = predict.cart_model(model, X),
            predict.random_forest_model(model, X, type = "class")
          )
        } else {
          switch(model_class,
            "ols_model" = predict.ols_model(model, X),
            "random_forest_model" = predict.random_forest_model(model, X),
            predict.random_forest_model(model, X)
          )
        }
      },
      metric_fn = metric_fn,
      k = cv_folds
    )
  }

  list(
    model = model,
    evaluation = eval_result,
    cv = cv_result,
    split = split,
    method = method,
    type = type
  )
}

###############################################################################
# END OF FILE: machine_learning.R
###############################################################################
