# =============================================================================
# robust_methods.R
# Robust Statistics for Crypto/Quant Trading
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Crypto returns are rife with outliers -- flash crashes,
# exchange outages, wash-trading spikes. Classical mean/variance estimates are
# swamped by these events. Robust estimators down-weight or clip extremes,
# giving stable estimates that reflect the "typical" market, not one-off chaos.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

#' Median absolute deviation (consistent scale estimate)
mad_scale <- function(x, constant = 1.4826) {
  constant * median(abs(x - median(x, na.rm = TRUE)), na.rm = TRUE)
}

#' Winsorise x at quantile level p (two-sided)
winsorise <- function(x, p = 0.01) {
  lo <- quantile(x, p,     na.rm = TRUE)
  hi <- quantile(x, 1 - p, na.rm = TRUE)
  pmax(lo, pmin(hi, x))
}

#' Safe matrix inverse via Cholesky; falls back to pseudoinverse
safe_inv <- function(M, tol = 1e-10) {
  tryCatch({
    chol2inv(chol(M))
  }, error = function(e) {
    # Moore-Penrose pseudoinverse via SVD
    sv <- svd(M)
    d  <- sv$d
    d[d < tol] <- 0
    di <- ifelse(d > 0, 1/d, 0)
    sv$v %*% diag(di, length(d)) %*% t(sv$u)
  })
}

# ---------------------------------------------------------------------------
# 2. M-ESTIMATORS
# ---------------------------------------------------------------------------
# M-estimators generalise the MLE: they minimise sum rho(r_i/s) where rho is
# a loss function less steep than the square -- clipping influence of outliers.

# -- 2a. Huber's rho and psi --
# rho(u) = u^2/2 if |u|<=k, k*|u|-k^2/2 otherwise
# psi(u) = u     if |u|<=k, k*sign(u)    otherwise  (derivative of rho)

huber_psi <- function(u, k = 1.345) {
  ifelse(abs(u) <= k, u, k * sign(u))
}

huber_rho <- function(u, k = 1.345) {
  ifelse(abs(u) <= k, u^2 / 2, k * abs(u) - k^2 / 2)
}

# -- 2b. Tukey biweight (bisquare) --
# Fully rejects extreme outliers (psi -> 0 beyond cutoff c)

biweight_psi <- function(u, c = 4.685) {
  ifelse(abs(u) <= c, u * (1 - (u/c)^2)^2, 0)
}

biweight_rho <- function(u, c = 4.685) {
  ifelse(abs(u) <= c,
         c^2 / 6 * (1 - (1 - (u/c)^2)^3),
         c^2 / 6)
}

# -- 2c. Andrews wave --
# Sine-based; smooth transition to zero influence

andrews_psi <- function(u, a = 1.339 * pi) {
  ifelse(abs(u) <= a, sin(u / a) * a, 0)
}

# ---------------------------------------------------------------------------
# 3. IRLS (ITERATIVELY REWEIGHTED LEAST SQUARES) ROBUST REGRESSION
# ---------------------------------------------------------------------------
# At each iteration: weights w_i = psi(r_i/s) / r_i; solve WLS; recompute r.
# Converges to M-estimate of regression coefficients.

irls_robust <- function(y, X,
                         psi_fn    = huber_psi,
                         k         = 1.345,
                         max_iter  = 100L,
                         tol       = 1e-8) {
  n <- length(y)
  p <- ncol(X)
  # OLS start
  beta <- tryCatch(solve(t(X) %*% X, t(X) %*% y),
                   error = function(e) rep(0, p))

  for (iter in seq_len(max_iter)) {
    resid <- as.numeric(y - X %*% beta)
    s     <- mad_scale(resid)
    if (s < 1e-12) break
    u     <- resid / s
    w     <- psi_fn(u, k) / ifelse(abs(u) < 1e-10, 1e-10, u)
    w     <- pmax(0, w)
    W     <- diag(w)
    XtWX  <- t(X) %*% W %*% X
    XtWy  <- t(X) %*% W %*% y
    beta_new <- tryCatch(solve(XtWX, XtWy),
                          error = function(e) beta)
    if (max(abs(beta_new - beta)) < tol) {
      beta <- beta_new
      break
    }
    beta <- beta_new
  }

  resid <- as.numeric(y - X %*% beta)
  s     <- mad_scale(resid)
  list(coefficients = as.numeric(beta),
       residuals    = resid,
       scale        = s,
       iterations   = iter)
}

#' Convenience wrapper: robust simple linear regression y ~ x
robust_lm <- function(y, x, intercept = TRUE, ...) {
  X <- if (intercept) cbind(1, x) else matrix(x, ncol = 1)
  irls_robust(y, X, ...)
}

# ---------------------------------------------------------------------------
# 4. MINIMUM COVARIANCE DETERMINANT (fast-MCD)
# ---------------------------------------------------------------------------
# MCD finds the subset of h observations whose covariance matrix has the
# smallest determinant.  Robust to up to (n-h)/n fraction of outliers.
# We implement the C-step algorithm (Rousseeuw & Van Driessen 1999).

fast_mcd <- function(X, alpha = 0.5, n_starts = 10L, max_iter = 50L, seed = 1L) {
  set.seed(seed)
  n <- nrow(X)
  p <- ncol(X)
  h <- max(p + 1L, as.integer(n * (1 - alpha)))   # subset size

  best_det  <- Inf
  best_loc  <- rep(0, p)
  best_cov  <- diag(p)

  for (start in seq_len(n_starts)) {
    # Random initial subset
    idx <- sample.int(n, h)
    for (iter in seq_len(max_iter)) {
      sub  <- X[idx, , drop = FALSE]
      loc  <- colMeans(sub)
      cov0 <- cov(sub)
      cov0 <- cov0 + diag(1e-8, p)  # regularise
      cov_inv <- safe_inv(cov0)
      # Mahalanobis distances for all n points
      cent  <- sweep(X, 2, loc)
      mah2  <- rowSums((cent %*% cov_inv) * cent)
      # New subset: h smallest distances
      idx_new <- order(mah2)[1:h]
      if (setequal(idx, idx_new)) break
      idx <- idx_new
    }
    det_val <- det(cov0)
    if (is.finite(det_val) && det_val < best_det) {
      best_det <- det_val
      best_loc <- loc
      best_cov <- cov0
    }
  }

  # Consistency factor correction (approx)
  q_alpha <- qchisq(1 - alpha, df = p)
  c_alpha <- (1 - alpha) / pchisq(q_alpha, df = p + 2)
  best_cov_scaled <- c_alpha * best_cov

  list(location   = best_loc,
       covariance = best_cov_scaled,
       raw_cov    = best_cov,
       det        = best_det)
}

#' Robust Mahalanobis distance using MCD estimates
robust_mahalanobis <- function(X, mcd = NULL, alpha = 0.5) {
  if (is.null(mcd)) mcd <- fast_mcd(X, alpha = alpha)
  cent    <- sweep(X, 2, mcd$location)
  cov_inv <- safe_inv(mcd$covariance)
  mah2    <- rowSums((cent %*% cov_inv) * cent)
  sqrt(pmax(0, mah2))
}

# ---------------------------------------------------------------------------
# 5. ROBUST PCA VIA ADMM
# ---------------------------------------------------------------------------
# Decompose M = L + S where L is low-rank (robust signal) and S is sparse
# (outlier component).  ADMM alternates between updating L, S, and dual vars.
# Financial intuition: crypto factor structure is L; idiosyncratic spikes = S.

robust_pca_admm <- function(M,
                              lambda   = NULL,
                              max_iter = 200L,
                              mu       = 1.0,
                              tol      = 1e-6) {
  m <- nrow(M); n <- ncol(M)
  if (is.null(lambda)) lambda <- 1 / sqrt(max(m, n))

  # Singular value thresholding
  svt <- function(A, tau) {
    sv  <- svd(A, nu = min(m, n), nv = min(m, n))
    d   <- pmax(sv$d - tau, 0)
    k   <- sum(d > 0)
    if (k == 0) return(matrix(0, m, n))
    sv$u[, 1:k, drop=FALSE] %*% diag(d[1:k], k) %*% t(sv$v[, 1:k, drop=FALSE])
  }
  # Soft threshold (element-wise)
  soft <- function(A, tau) sign(A) * pmax(abs(A) - tau, 0)

  L <- matrix(0, m, n)
  S <- matrix(0, m, n)
  Y <- matrix(0, m, n)   # dual variable

  for (iter in seq_len(max_iter)) {
    L_new <- svt(M - S + Y / mu, 1 / mu)
    S_new <- soft(M - L_new + Y / mu, lambda / mu)
    Y     <- Y + mu * (M - L_new - S_new)
    err   <- norm(M - L_new - S_new, "F") / (norm(M, "F") + 1e-12)
    L <- L_new; S <- S_new
    if (err < tol) break
  }

  list(L = L, S = S, iterations = iter,
       rank_L = sum(svd(L, nu=0, nv=0)$d > 1e-8))
}

# ---------------------------------------------------------------------------
# 6. ROBUST CORRELATION MEASURES
# ---------------------------------------------------------------------------

#' Spearman rank correlation matrix
spearman_cor <- function(X) {
  n <- ncol(X)
  R <- apply(X, 2, rank, ties.method = "average")
  cor(R)
}

#' Kendall tau correlation (O(n^2), exact)
kendall_tau <- function(x, y) {
  n <- length(x)
  con <- dis <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      dx <- x[i] - x[j]; dy <- y[i] - y[j]
      if (dx * dy > 0) con <- con + 1
      else if (dx * dy < 0) dis <- dis + 1
    }
  }
  2 * (con - dis) / (n * (n - 1))
}

#' Biweight midcorrelation (robust correlation)
biweight_midcor <- function(x, y) {
  mx <- median(x, na.rm = TRUE); my <- median(y, na.rm = TRUE)
  ux <- x - mx; uy <- y - my
  sx <- mad_scale(x); sy <- mad_scale(y)
  if (sx == 0 || sy == 0) return(NA_real_)
  wx <- (1 - (ux / (9 * sx))^2)^2 * (abs(ux) < 9 * sx)
  wy <- (1 - (uy / (9 * sy))^2)^2 * (abs(uy) < 9 * sy)
  w  <- wx * wy
  num <- sum(w * ux * uy)
  den <- sqrt(sum(w * ux^2) * sum(w * uy^2))
  if (den == 0) return(NA_real_)
  num / den
}

#' Robust correlation matrix using biweight midcorrelation
biweight_cor_matrix <- function(X) {
  p   <- ncol(X)
  R   <- diag(1.0, p)
  nms <- colnames(X)
  for (i in 1:(p - 1)) {
    for (j in (i + 1):p) {
      r       <- biweight_midcor(X[, i], X[, j])
      R[i, j] <- r
      R[j, i] <- r
    }
  }
  if (!is.null(nms)) { rownames(R) <- nms; colnames(R) <- nms }
  R
}

# ---------------------------------------------------------------------------
# 7. ROBUST SHARPE RATIO
# ---------------------------------------------------------------------------
# Replace mean with median, std with MAD for outlier resistance.
# Financial intuition: a single day of 30% gain inflates classical Sharpe;
# robust Sharpe reflects whether the *typical* day is positive.

robust_sharpe <- function(rets, ann = 252) {
  med <- median(rets, na.rm = TRUE)
  s   <- mad_scale(rets)
  if (s < 1e-12) return(NA_real_)
  (med / s) * sqrt(ann)
}

#' Compare classical vs robust Sharpe
sharpe_comparison <- function(rets, ann = 252) {
  classical <- mean(rets, na.rm=TRUE) / sd(rets, na.rm=TRUE) * sqrt(ann)
  robust    <- robust_sharpe(rets, ann)
  data.frame(classical = classical, robust = robust,
             ratio = robust / classical)
}

# ---------------------------------------------------------------------------
# 8. OUTLIER DETECTION
# ---------------------------------------------------------------------------

#' Classical Mahalanobis distance (non-robust)
classical_mahalanobis <- function(X) {
  mu  <- colMeans(X, na.rm = TRUE)
  S   <- cov(X)
  S   <- S + diag(1e-8, ncol(X))
  Si  <- safe_inv(S)
  cent <- sweep(X, 2, mu)
  sqrt(pmax(0, rowSums((cent %*% Si) * cent)))
}

#' Flag outliers: robust Mahalanobis > chi2 cutoff
flag_outliers_mcd <- function(X, alpha_mcd = 0.5, p_cut = 0.975) {
  mcd  <- fast_mcd(X, alpha = alpha_mcd)
  dist <- robust_mahalanobis(X, mcd)
  cutoff <- sqrt(qchisq(p_cut, df = ncol(X)))
  data.frame(
    obs      = seq_len(nrow(X)),
    rob_mah  = dist,
    outlier  = dist > cutoff,
    cutoff   = cutoff
  )
}

#' Isolation Forest concept (random partitions depth scoring)
#' Each point's anomaly score = avg depth to isolation across trees
isolation_forest <- function(X, n_trees = 100L, sample_size = 256L, seed = 42L) {
  set.seed(seed)
  n <- nrow(X); p <- ncol(X)
  sample_size <- min(sample_size, n)

  # Expected depth of isolation under null (harmonic number approx)
  H <- function(i) log(i) + 0.5772156649
  c_n <- 2 * H(sample_size - 1) - 2 * (sample_size - 1) / sample_size

  depths <- matrix(0, n, n_trees)

  for (tree in seq_len(n_trees)) {
    samp <- sample.int(n, sample_size)
    Xs   <- X[samp, , drop = FALSE]

    # Recursively build isolation tree, return depth for each point in X
    isolate <- function(Xt, indices, depth) {
      if (length(indices) <= 1 || depth > ceiling(log2(sample_size))) {
        return(rep(depth + c_n, length(indices)))
      }
      j  <- sample.int(p, 1)
      xj <- Xt[, j]
      lo <- min(xj); hi <- max(xj)
      if (lo == hi) return(rep(depth + c_n, length(indices)))
      sp <- runif(1, lo, hi)
      left  <- which(xj <= sp)
      right <- which(xj > sp)
      res <- numeric(length(indices))
      if (length(left)  > 0) res[left]  <- isolate(Xt[left,  , drop=FALSE], left,  depth + 1)
      if (length(right) > 0) res[right] <- isolate(Xt[right, , drop=FALSE], right, depth + 1)
      res
    }

    # Compute depth for each full-data point using tree splits learned on sample
    # Simplified: apply tree to all n points
    all_depths <- isolate(X, seq_len(n), 0L)
    depths[, tree] <- all_depths
  }

  avg_depth <- rowMeans(depths)
  score     <- 2^(-avg_depth / c_n)   # anomaly score [0,1]; higher = more anomalous
  data.frame(obs = seq_len(n), avg_depth = avg_depth, anomaly_score = score)
}

# ---------------------------------------------------------------------------
# 9. ROBUST GARCH / FILTERED HISTORICAL SIMULATION
# ---------------------------------------------------------------------------
# Standard GARCH with t-distributed residuals; robust variance via EWMA+MAD

#' EWMA variance (lambda-weighted)
ewma_var <- function(rets, lambda = 0.94) {
  n   <- length(rets)
  var <- numeric(n)
  var[1] <- rets[1]^2
  for (i in 2:n) {
    var[i] <- lambda * var[i-1] + (1 - lambda) * rets[i]^2
  }
  var
}

#' Robust EWMA variance using Huber-weighted squares
robust_ewma_var <- function(rets, lambda = 0.94, k = 2.0) {
  n   <- length(rets)
  var <- numeric(n)
  var[1] <- rets[1]^2
  for (i in 2:n) {
    sig <- sqrt(var[i-1])
    u   <- rets[i] / max(sig, 1e-12)
    w   <- if (abs(u) <= k) 1 else k / abs(u)
    var[i] <- lambda * var[i-1] + (1 - lambda) * w * rets[i]^2
  }
  var
}

#' Filtered Historical Simulation with robust EWMA vol
fhs_var <- function(rets,
                     conf  = 0.99,
                     lbk   = 252L,
                     lambda = 0.94) {
  n     <- length(rets)
  rv    <- robust_ewma_var(rets, lambda)
  sigma <- sqrt(rv)
  # Standardised residuals
  std_r <- rets / pmax(sigma, 1e-12)
  # 1-day VaR using historical quantile of std residuals x current sigma
  var_est <- numeric(n)
  for (i in (lbk + 1):n) {
    hist_std <- std_r[(i - lbk):(i - 1)]
    q_std    <- quantile(hist_std, 1 - conf, na.rm = TRUE)
    var_est[i] <- sigma[i] * q_std   # negative = loss
  }
  list(VaR = var_est, sigma = sigma, std_resid = std_r)
}

#' Exceedance ratio (actual losses beyond VaR)
fhs_backtest <- function(rets, fhs_out, conf = 0.99) {
  n    <- length(rets)
  start <- which(fhs_out$VaR != 0)[1]
  if (is.na(start)) return(NA_real_)
  hits <- sum(rets[start:n] < fhs_out$VaR[start:n], na.rm = TRUE)
  expected <- (n - start + 1) * (1 - conf)
  list(hits = hits, expected = round(expected, 1),
       ratio = hits / max(expected, 1))
}

# ---------------------------------------------------------------------------
# 10. ROBUST LINEAR FACTOR MODEL
# ---------------------------------------------------------------------------

#' Fit factor model via IRLS; return factor loadings + robust residuals
robust_factor_model <- function(returns_matrix, factors_matrix) {
  # returns_matrix: T x N  (T periods, N assets)
  # factors_matrix: T x K  (T periods, K factors)
  T_  <- nrow(returns_matrix)
  N   <- ncol(returns_matrix)
  K   <- ncol(factors_matrix)
  X   <- cbind(1, factors_matrix)   # add intercept column

  loadings  <- matrix(NA, N, K + 1)
  residuals <- matrix(NA, T_, N)

  for (i in seq_len(N)) {
    fit <- irls_robust(returns_matrix[, i], X)
    loadings[i, ]  <- fit$coefficients
    residuals[, i] <- fit$residuals
  }
  colnames(loadings) <- c("alpha", paste0("beta_", seq_len(K)))
  list(loadings  = loadings,
       residuals = residuals,
       robust_cov_resid = biweight_cor_matrix(residuals))
}

# ---------------------------------------------------------------------------
# 11. ROBUST PORTFOLIO OPTIMISATION (min variance via robust cov)
# ---------------------------------------------------------------------------

#' Global minimum variance using MCD covariance
robust_gmv <- function(returns_matrix) {
  mcd   <- fast_mcd(returns_matrix)
  Sigma <- mcd$covariance
  n     <- ncol(Sigma)
  ones  <- rep(1, n)
  Si    <- safe_inv(Sigma)
  raw   <- Si %*% ones
  w     <- raw / sum(raw)
  w[w < 0] <- 0; w <- w / sum(w)
  list(weights = as.numeric(w),
       cov     = Sigma,
       mcd_loc = mcd$location)
}

# ---------------------------------------------------------------------------
# 12. ROLLING ROBUST STATISTICS
# ---------------------------------------------------------------------------

#' Rolling robust mean (median) and scale (MAD) over window w
rolling_robust_stats <- function(x, w = 60L) {
  n    <- length(x)
  out  <- data.frame(t  = seq_len(n),
                     mu = rep(NA, n),
                     s  = rep(NA, n))
  for (i in w:n) {
    xw <- x[(i - w + 1):i]
    out$mu[i] <- median(xw, na.rm = TRUE)
    out$s[i]  <- mad_scale(xw)
  }
  out
}

#' Rolling biweight correlation
rolling_biweight_cor <- function(x, y, w = 60L) {
  n   <- length(x)
  out <- rep(NA, n)
  for (i in w:n) {
    out[i] <- biweight_midcor(x[(i-w+1):i], y[(i-w+1):i])
  }
  out
}

# ---------------------------------------------------------------------------
# 13. SIMULATION STUDY: BREAKDOWN UNDER CONTAMINATION
# ---------------------------------------------------------------------------

contamination_study <- function(n = 500L, contamination_pcts = seq(0, 0.30, 0.05),
                                 seed = 77L) {
  set.seed(seed)
  # True parameters
  true_beta <- c(0.5, 1.5)
  x_clean   <- rnorm(n)
  eps_clean <- rnorm(n, sd = 0.5)
  y_clean   <- true_beta[1] + true_beta[2] * x_clean + eps_clean

  results <- lapply(contamination_pcts, function(pct) {
    n_cont <- as.integer(n * pct)
    y <- y_clean
    if (n_cont > 0) {
      cont_idx <- sample.int(n, n_cont)
      y[cont_idx] <- y[cont_idx] + 10 * rnorm(n_cont)  # large outliers
    }
    X   <- cbind(1, x_clean)
    # OLS
    ols  <- solve(t(X) %*% X, t(X) %*% y)
    # Robust IRLS (Huber)
    hub  <- irls_robust(y, X, psi_fn = huber_psi)
    # Robust IRLS (Tukey)
    tuk  <- irls_robust(y, X, psi_fn = biweight_psi)

    data.frame(
      pct          = pct,
      ols_beta1    = ols[2],
      huber_beta1  = hub$coefficients[2],
      tukey_beta1  = tuk$coefficients[2],
      ols_err      = abs(ols[2] - true_beta[2]),
      huber_err    = abs(hub$coefficients[2] - true_beta[2]),
      tukey_err    = abs(tuk$coefficients[2] - true_beta[2])
    )
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 14. MAIN DEMO
# ---------------------------------------------------------------------------

run_robust_demo <- function() {
  cat("=== Robust Methods for Crypto/Quant Trading ===\n\n")
  set.seed(123)

  # Synthetic crypto return series with outliers
  n <- 1000L
  rets <- c(rnorm(900, 0, 0.02),
            rnorm(100, 0, 0.20))   # 10% outlier contamination
  rets <- sample(rets)             # shuffle

  cat("--- 1. Robust vs Classical Location/Scale ---\n")
  cat(sprintf("  Classical mean: %.5f  Classical sd: %.5f\n",
              mean(rets), sd(rets)))
  cat(sprintf("  Robust median:  %.5f  MAD scale:    %.5f\n",
              median(rets), mad_scale(rets)))

  cat("\n--- 2. Sharpe Comparison ---\n")
  sc <- sharpe_comparison(rets)
  print(sc)

  cat("\n--- 3. IRLS Robust Regression ---\n")
  x  <- rnorm(n)
  y  <- 0.5 + 1.5 * x + rnorm(n, 0, 0.5)
  y[1:50] <- y[1:50] + runif(50, 5, 15)   # add leverage outliers
  X  <- cbind(1, x)
  ols_b <- solve(t(X) %*% X, t(X) %*% y)
  hub   <- irls_robust(y, X, psi_fn = huber_psi)
  tuk   <- irls_robust(y, X, psi_fn = biweight_psi)
  cat(sprintf("  OLS beta1=%.3f  Huber beta1=%.3f  Tukey beta1=%.3f  (true=1.5)\n",
              ols_b[2], hub$coefficients[2], tuk$coefficients[2]))

  cat("\n--- 4. Robust PCA (ADMM) ---\n")
  T_ <- 200L; N <- 5L
  F1 <- rnorm(T_)
  R  <- matrix(NA, T_, N)
  for (i in seq_len(N)) R[, i] <- 0.6 * F1 + 0.4 * rnorm(T_)
  # Add sparse outliers
  R[sample.int(T_, 20), sample.int(N, 2)] <- rnorm(20, 0, 5)
  rpca <- robust_pca_admm(R)
  cat(sprintf("  Rank of L: %d  |S|_0 (nonzero sparse): %d\n",
              rpca$rank_L, sum(abs(rpca$S) > 0.01)))

  cat("\n--- 5. MCD Robust Covariance ---\n")
  X2 <- matrix(rnorm(300 * 4), 300, 4)
  X2[1:20, ] <- X2[1:20, ] + 5   # outlier block
  mcd <- fast_mcd(X2, alpha = 0.5)
  cat("  MCD Location:", round(mcd$location, 3), "\n")
  cat("  Classical Location:", round(colMeans(X2), 3), "\n")

  cat("\n--- 6. Outlier Detection (MCD Mahalanobis) ---\n")
  flags <- flag_outliers_mcd(X2)
  n_out <- sum(flags$outlier)
  cat(sprintf("  Detected %d outliers (expected ~20)\n", n_out))

  cat("\n--- 7. Contamination Study (Breakdown Point) ---\n")
  study <- contamination_study(n = 300L,
                                contamination_pcts = seq(0, 0.30, 0.10))
  cat("  Pct contamination -> Beta1 estimation error:\n")
  print(study[, c("pct","ols_err","huber_err","tukey_err")])

  cat("\n--- 8. Filtered Historical Simulation VaR ---\n")
  set.seed(42)
  mkt_rets <- c(rnorm(900, 0, 0.02), rnorm(100, 0, 0.08))
  mkt_rets <- sample(mkt_rets)
  fhs <- fhs_var(mkt_rets, conf = 0.99)
  bt  <- fhs_backtest(mkt_rets, fhs, conf = 0.99)
  cat(sprintf("  VaR exceedances: %d (expected ~%.1f) | ratio: %.2f\n",
              bt$hits, bt$expected, bt$ratio))

  cat("\n--- 9. Biweight Correlation ---\n")
  x1 <- rnorm(200); x2 <- 0.7 * x1 + 0.3 * rnorm(200)
  x1[1:10] <- x1[1:10] + 10  # outliers
  cat(sprintf("  Pearson: %.3f  Biweight midcor: %.3f\n",
              cor(x1, x2), biweight_midcor(x1, x2)))

  cat("\nDone.\n")
  invisible(list(hub = hub, tuk = tuk, rpca = rpca, mcd = mcd,
                 fhs = fhs, study = study))
}

if (interactive()) {
  robust_results <- run_robust_demo()
}

# ---------------------------------------------------------------------------
# 15. ROBUST ROLLING BETA (THEIL-SEN ESTIMATOR)
# ---------------------------------------------------------------------------
# Theil-Sen: median of all pairwise slopes. More robust than OLS beta.
# Financial intuition: a single extreme return day can inflate OLS beta by 30%;
# Theil-Sen is resistant to a fraction of contaminated observations.

theil_sen_slope <- function(x, y) {
  n     <- length(x)
  slopes <- numeric(n * (n - 1) / 2)
  k <- 0L
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      dx <- x[j] - x[i]
      if (abs(dx) > 1e-12) { k <- k + 1L; slopes[k] <- (y[j] - y[i]) / dx }
    }
  }
  median(slopes[1:k])
}

rolling_theil_sen_beta <- function(asset_rets, market_rets, window = 60L) {
  n   <- length(asset_rets)
  beta_ts  <- rep(NA, n)
  ols_beta <- rep(NA, n)
  for (t in window:n) {
    x <- market_rets[(t-window+1):t]
    y <- asset_rets[(t-window+1):t]
    valid <- !is.na(x) & !is.na(y)
    if (sum(valid) < 10) next
    beta_ts[t]  <- theil_sen_slope(x[valid], y[valid])
    ols_beta[t] <- cov(x[valid], y[valid]) / var(x[valid])
  }
  data.frame(t=seq_len(n), theil_sen=beta_ts, ols=ols_beta,
             divergence=abs(beta_ts - ols_beta))
}

# ---------------------------------------------------------------------------
# 16. ROBUST EXPONENTIALLY WEIGHTED COVARIANCE
# ---------------------------------------------------------------------------
# Applies Huber weights to each observation before exponential accumulation.

robust_ewcov <- function(X, lambda = 0.94, k = 2.5) {
  n <- nrow(X); p <- ncol(X)
  mu  <- colMeans(X[1:min(20L,n), ], na.rm=TRUE)
  cov_ <- diag(p) * 0.01
  for (t in seq_len(n)) {
    x   <- X[t, ] - mu
    mah <- as.numeric(sqrt(pmax(t(x) %*% solve(cov_ + diag(1e-8,p)) %*% x, 0)))
    w   <- if (mah <= k) 1 else k / mah
    mu  <- lambda * mu  + (1 - lambda) * w * X[t,]
    cov_ <- lambda * cov_ + (1 - lambda) * w * outer(x, x)
  }
  list(mean = mu, cov = cov_)
}

# ---------------------------------------------------------------------------
# 17. ROBUST QUANTILE REGRESSION
# ---------------------------------------------------------------------------
# Minimise sum rho_tau(y_i - x_i'b) where rho_tau is the check function.
# tau=0.5 is median regression (LAD); other taus give conditional quantiles.

quantile_regression <- function(y, X, tau = 0.5,
                                  max_iter = 500L, tol = 1e-8) {
  n <- length(y); p <- ncol(X)
  b <- tryCatch(solve(t(X)%*%X + diag(1e-8,p), t(X)%*%y),
                 error=function(e) rep(0,p))
  for (iter in seq_len(max_iter)) {
    resid <- y - X %*% b
    # IRLS weights for quantile regression
    w  <- ifelse(abs(resid) < 1e-6, 1e6,
                  ifelse(resid > 0, tau / abs(resid),
                          (1-tau) / abs(resid)))
    W  <- diag(as.numeric(w))
    b_new <- tryCatch(solve(t(X)%*%W%*%X + diag(1e-8,p), t(X)%*%W%*%y),
                       error=function(e) b)
    if (max(abs(b_new - b)) < tol) { b <- b_new; break }
    b <- b_new
  }
  list(coef=as.numeric(b), tau=tau,
       residuals=as.numeric(y - X%*%b))
}

#' Quantile regression portfolio: condition on factor quantiles
quantile_factor_model <- function(returns, factors,
                                   taus = c(0.10, 0.25, 0.50, 0.75, 0.90)) {
  X   <- cbind(1, factors)
  results <- lapply(taus, function(tau) {
    qr  <- quantile_regression(returns, X, tau)
    data.frame(tau=tau,
               alpha=qr$coef[1],
               beta=qr$coef[-1])
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 18. ROBUST MEAN-VARIANCE PORTFOLIO (WORST-CASE)
# ---------------------------------------------------------------------------
# Minimax robust portfolio: optimise for the worst-case mean in an
# uncertainty set around the sample mean (ellipsoidal ambiguity set).

robust_minimax_portfolio <- function(returns_matrix,
                                      kappa = 0.5,    # uncertainty radius
                                      risk_aversion = 3) {
  n    <- nrow(returns_matrix); p <- ncol(returns_matrix)
  mu   <- colMeans(returns_matrix)
  Sig  <- cov(returns_matrix) + diag(1e-6, p)
  Si   <- solve(Sig)
  # Worst-case mean: mu_wc = mu - kappa * sqrt(diag(Sig)) (element-wise)
  mu_wc <- mu - kappa * sqrt(diag(Sig))
  # Standard MV with worst-case mean
  raw_w <- Si %*% mu_wc / risk_aversion
  w     <- pmax(raw_w, 0); w <- w / sum(w)
  list(weights = as.numeric(w), mu_wc = mu_wc,
       port_ret = sum(w * mu), port_vol = sqrt(as.numeric(t(w)%*%Sig%*%w)))
}

# ---------------------------------------------------------------------------
# 19. ROBUST FACTOR SELECTION (HARD THRESHOLDING)
# ---------------------------------------------------------------------------
# Select factors whose robust t-statistics exceed a threshold.

robust_factor_select <- function(returns, factors_matrix, threshold = 2.0) {
  K   <- ncol(factors_matrix)
  X   <- cbind(1, factors_matrix)
  fit <- irls_robust(returns, X)
  # Robust t-statistics using MAD of residuals
  s   <- fit$scale
  n   <- length(returns)
  XtX_inv <- tryCatch(solve(t(X)%*%X + diag(1e-8,K+1)),
                       error=function(e) diag(K+1))
  se  <- s * sqrt(diag(XtX_inv))
  t_stats <- fit$coefficients / pmax(se, 1e-8)
  selected <- which(abs(t_stats[-1]) > threshold)
  list(selected = selected, t_stats = t_stats[-1],
       coefficients = fit$coefficients[-1])
}

# ---------------------------------------------------------------------------
# 20. ROBUST MAXIMUM SHARPE PORTFOLIO
# ---------------------------------------------------------------------------

robust_max_sharpe <- function(returns_matrix, rf = 0) {
  mcd  <- fast_mcd(returns_matrix, alpha = 0.5)
  mu   <- mcd$location - rf
  Sig  <- mcd$covariance + diag(1e-6, ncol(returns_matrix))
  Si   <- solve(Sig)
  raw  <- Si %*% mu
  w    <- raw / sum(abs(raw))
  w[w < 0] <- 0; w <- w / sum(w)
  port_ret <- sum(w * mcd$location)
  port_vol <- sqrt(as.numeric(t(w) %*% Sig %*% w))
  list(weights=as.numeric(w), sharpe=(port_ret-rf)/max(port_vol,1e-8),
       port_ret=port_ret, port_vol=port_vol)
}

# ---------------------------------------------------------------------------
# 21. EXTENDED ROBUST DEMO
# ---------------------------------------------------------------------------

run_robust_extended_demo <- function() {
  cat("=== Robust Methods Extended Demo ===\n\n")
  set.seed(42); n <- 300L

  market_r <- rnorm(n, 0, 0.01)
  asset_r  <- 1.2 * market_r + rnorm(n, 0, 0.008)
  # Add outliers
  asset_r[c(50,100,150)] <- asset_r[c(50,100,150)] + c(0.15,-0.12,0.18)

  cat("--- Theil-Sen Rolling Beta ---\n")
  tsb <- rolling_theil_sen_beta(asset_r, market_r, window=60L)
  valid <- !is.na(tsb$theil_sen)
  cat(sprintf("  Mean Theil-Sen beta: %.4f  OLS: %.4f\n",
              mean(tsb$theil_sen[valid]), mean(tsb$ols[valid])))
  cat(sprintf("  Mean divergence: %.4f\n", mean(tsb$divergence[valid])))

  cat("\n--- Robust EW Covariance ---\n")
  X2 <- matrix(rnorm(200*3), 200, 3)
  X2[1:10,] <- X2[1:10,] + 5
  rewcov <- robust_ewcov(X2)
  cat("  Robust EWCOV diagonal:", round(diag(rewcov$cov), 4), "\n")

  cat("\n--- Quantile Regression ---\n")
  x_q <- rnorm(200); y_q <- 0.5 + 1.5*x_q + rnorm(200,0,0.5)
  y_q[1:20] <- y_q[1:20] + 5
  X_q <- cbind(1, x_q)
  qf <- quantile_factor_model(y_q, matrix(x_q), taus=c(0.25,0.5,0.75))
  cat("  Quantile factor betas (tau=0.25,0.5,0.75):", round(qf$beta, 3), "\n")

  cat("\n--- Robust Minimax Portfolio ---\n")
  R_port <- matrix(rnorm(300*5, 0.001, 0.02), 300, 5)
  rmp <- robust_minimax_portfolio(R_port, kappa=0.3)
  cat("  Weights:", round(rmp$weights, 3), "\n")
  cat(sprintf("  Port ret=%.4f  Port vol=%.4f\n", rmp$port_ret, rmp$port_vol))

  cat("\n--- Robust Max Sharpe Portfolio ---\n")
  rms <- robust_max_sharpe(R_port)
  cat("  Weights:", round(rms$weights, 3), "\n")
  cat(sprintf("  Robust Sharpe: %.4f\n", rms$sharpe))

  invisible(list(tsb=tsb, qf=qf, rmp=rmp, rms=rms))
}

if (interactive()) {
  robust_ext <- run_robust_extended_demo()
}

# ---------------------------------------------------------------------------
# 22. ROBUST AUTOCORRELATION (SPECTRAL TRIMMING)
# ---------------------------------------------------------------------------
# Classical ACF is sensitive to outliers. Robust ACF uses Huber-transformed
# residuals to estimate autocorrelation, avoiding inflated estimates from spikes.

robust_acf <- function(x, max_lag = 20L, k = 1.5) {
  x_clean <- x - median(x, na.rm=TRUE)
  scale_  <- mad_scale(x_clean)
  z       <- x_clean / max(scale_, 1e-8)
  # Huber-transform
  x_hub   <- ifelse(abs(z) <= k, x_clean, k * sign(z) * scale_)
  n       <- length(x_hub)
  acf_val <- numeric(max_lag + 1)
  acf_val[1] <- 1
  v0 <- mean(x_hub^2, na.rm=TRUE)
  for (lag in seq_len(max_lag)) {
    acf_val[lag + 1] <- mean(x_hub[1:(n-lag)] * x_hub[(lag+1):n], na.rm=TRUE) /
      max(v0, 1e-12)
  }
  data.frame(lag = 0:max_lag, acf = acf_val)
}

# ---------------------------------------------------------------------------
# 23. S-ESTIMATOR (HIGH BREAKDOWN POINT REGRESSION)
# ---------------------------------------------------------------------------
# S-estimators minimise the M-estimate of scale of residuals.
# They have 50% breakdown point -- the highest possible for regression.
# We implement via a simple iterative reweighting with biweight.

s_estimator <- function(y, X, n_start = 20L, max_iter = 100L, seed = 1L) {
  set.seed(seed)
  n <- length(y); p <- ncol(X)
  best_scale <- Inf; best_b <- rep(0, p)

  for (start in seq_len(n_start)) {
    # Random p+1 observation start
    idx <- sample.int(n, p + 1)
    b   <- tryCatch(solve(X[idx,,drop=FALSE], y[idx]),
                     error=function(e) rep(0,p))
    for (iter in seq_len(max_iter)) {
      r   <- y - X %*% b
      sc  <- mad_scale(r)
      if (sc < 1e-12) break
      u   <- r / sc
      w   <- biweight_psi(u) / ifelse(abs(u) < 1e-10, 1e-10, u)
      w   <- pmax(0, w)
      W   <- diag(w)
      b_n <- tryCatch(solve(t(X)%*%W%*%X + diag(1e-8,p), t(X)%*%W%*%y),
                       error=function(e) b)
      if (max(abs(b_n - b)) < 1e-8) { b <- b_n; break }
      b <- b_n
    }
    r_f <- y - X %*% b; sc_f <- mad_scale(r_f)
    if (sc_f < best_scale) { best_scale <- sc_f; best_b <- b }
  }
  list(coefficients = as.numeric(best_b), scale = best_scale,
       residuals = as.numeric(y - X %*% best_b))
}

# ---------------------------------------------------------------------------
# 24. ROBUST EXPONENTIAL SMOOTHING
# ---------------------------------------------------------------------------
# Huber-weight the update in Holt's exponential smoothing.

robust_ets <- function(x, alpha = 0.2, k = 2.5) {
  n    <- length(x)
  l    <- numeric(n); l[1] <- x[1]
  for (t in 2:n) {
    r   <- x[t] - l[t-1]
    sc  <- mad_scale(x[1:t])
    u   <- r / max(sc, 1e-8)
    w   <- if (abs(u) <= k) 1 else k / abs(u)
    l[t] <- l[t-1] + alpha * w * r
  }
  list(level=l, fitted=c(NA, l[-n]),
       residuals=x - c(NA, l[-n]),
       forecast_1=tail(l,1))
}

# ---------------------------------------------------------------------------
# 25. ROBUST PORTFOLIO BACKTEST
# ---------------------------------------------------------------------------

robust_portfolio_backtest <- function(returns_matrix,
                                       window = 60L,
                                       tc = 0.001,
                                       method = "mcd") {
  T_  <- nrow(returns_matrix); N <- ncol(returns_matrix)
  equity <- numeric(T_); equity[1] <- 1.0
  w      <- rep(1/N, N); prev_w <- w

  for (t in (window + 1):T_) {
    R_w <- returns_matrix[(t-window+1):(t-1), , drop=FALSE]
    w_new <- if (method == "mcd") {
      tryCatch(robust_gmv(R_w)$weights, error=function(e) rep(1/N,N))
    } else {
      rep(1/N, N)
    }
    ret   <- sum(w * returns_matrix[t, ])
    cost  <- sum(abs(w_new - w)) * tc / 2
    equity[t] <- equity[t-1] * (1 + ret - cost)
    prev_w <- w; w <- w_new
  }
  rets <- c(rep(NA, window), diff(log(equity[(window):T_])))
  eq_clean <- equity[!is.na(rets)]
  list(equity=equity, returns=rets,
       sharpe=sharpe_ratio(rets[!is.na(rets)]),
       max_dd=max_drawdown(pmax(eq_clean,1e-8)))
}

# ---------------------------------------------------------------------------
# 26. FINAL ROBUST METHODS DEMO
# ---------------------------------------------------------------------------

run_robust_final_demo <- function() {
  cat("=== Robust Methods Final Demo ===\n\n")
  set.seed(42); n <- 400L
  rets <- c(rnorm(350,0,0.02), rnorm(50,0,0.15))
  rets <- sample(rets)

  cat("--- Robust ACF ---\n")
  racf <- robust_acf(rets, max_lag=10L)
  cat("  Robust ACF lags 1-5:", round(racf$acf[2:6],4), "\n")
  classical_acf <- acf(rets, lag.max=5, plot=FALSE)$acf[-1]
  cat("  Classical ACF lags 1-5:", round(classical_acf,4), "\n")

  cat("\n--- S-Estimator vs OLS ---\n")
  x_s <- rnorm(200); y_s <- 0.5 + 1.5*x_s + rnorm(200,0,0.5)
  y_s[1:20] <- y_s[1:20] + 10
  X_s <- cbind(1, x_s)
  ols_s <- solve(t(X_s)%*%X_s, t(X_s)%*%y_s)
  s_est <- s_estimator(y_s, X_s, n_start=10L)
  cat(sprintf("  OLS beta1=%.3f  S-Est beta1=%.3f  (true=1.5)\n",
              ols_s[2], s_est$coefficients[2]))

  cat("\n--- Robust ETS ---\n")
  price <- cumprod(1 + rets) * 100
  price[c(100,200,300)] <- price[c(100,200,300)] * c(1.5,0.6,1.4)
  rets_robust <- robust_ets(price, alpha=0.15)
  cat(sprintf("  1-step ahead forecast: %.2f  Level at T: %.2f\n",
              rets_robust$forecast_1, tail(rets_robust$level,1)))

  cat("\n--- Robust Portfolio Backtest (MCD) ---\n")
  R_mat <- matrix(c(rnorm(400*4,0,0.02)), 400, 4)
  R_mat[sample.int(400,40),] <- R_mat[sample.int(400,40),] + 0.15
  rpbt_mcd <- robust_portfolio_backtest(R_mat, method="mcd")
  rpbt_ew  <- robust_portfolio_backtest(R_mat, method="ew")
  cat(sprintf("  MCD portfolio Sharpe=%.3f  EW Sharpe=%.3f\n",
              rpbt_mcd$sharpe, rpbt_ew$sharpe))

  invisible(list(racf=racf, s_est=s_est, ets=rets_robust, rpbt=rpbt_mcd))
}

if (interactive()) {
  robust_final <- run_robust_final_demo()
}
