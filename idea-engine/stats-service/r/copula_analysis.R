# =============================================================================
# copula_analysis.R
# Copula dependence modeling for crypto/quant trading
# Uses base R only -- no external packages
# =============================================================================
# Financial intuition: linear correlation misses tail co-movement. Copulas
# separate the marginal distributions from the joint dependence structure,
# allowing us to model "when BTC crashes, ETH also crashes" more accurately
# than Pearson correlation alone.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. UNIFORM MARGINALS VIA EMPIRICAL CDF
# -----------------------------------------------------------------------------

#' Transform data to pseudo-uniform marginals using empirical CDF
#' This is the first step in copula fitting: "probability integral transform"
#' @param x numeric vector of returns
#' @return numeric vector in (0,1)
emp_cdf_transform <- function(x) {
  n <- length(x)
  ranks <- rank(x, ties.method = "average")
  u <- ranks / (n + 1)  # rescale to open unit interval
  u
}

#' Transform a matrix of returns to uniform marginals
#' @param X matrix, each column is an asset's return series
#' @return matrix of pseudo-uniform observations (copula data)
returns_to_copula_data <- function(X) {
  apply(X, 2, emp_cdf_transform)
}

# -----------------------------------------------------------------------------
# 2. GAUSSIAN COPULA
# -----------------------------------------------------------------------------

#' Fit Gaussian copula: estimate linear correlation on normal scores
#' The Gaussian copula captures symmetric tail dependence (actually zero
#' tail dependence -- an important limitation for crypto crash modeling)
#' @param U matrix of uniform marginals (n x d)
#' @return list with correlation matrix R
fit_gaussian_copula <- function(U) {
  # Transform uniforms to standard normal scores
  Z <- qnorm(U)
  # MLE of correlation matrix = sample correlation of normal scores
  R <- cor(Z)
  list(type = "gaussian", R = R, dim = ncol(U))
}

#' Log-likelihood of Gaussian copula
#' @param U matrix of uniform marginals
#' @param R correlation matrix
gaussian_copula_loglik <- function(U, R) {
  Z <- qnorm(U)
  n <- nrow(Z)
  d <- ncol(Z)
  # log|R|
  log_det_R <- log(det(R))
  # Ri = R^{-1}
  Ri <- solve(R)
  # log-likelihood: sum_i [-0.5 * log|R| - 0.5 * z_i'(R^{-1}-I)z_i]
  ll <- 0
  I_d <- diag(d)
  diff_mat <- Ri - I_d
  for (i in seq_len(n)) {
    z <- Z[i, ]
    ll <- ll - 0.5 * log_det_R - 0.5 * (t(z) %*% diff_mat %*% z)
  }
  as.numeric(ll)
}

#' Simulate from a Gaussian copula
#' @param n number of simulations
#' @param R correlation matrix
#' @return matrix (n x d) of uniform marginals from copula
simulate_gaussian_copula <- function(n, R) {
  d <- nrow(R)
  # Cholesky decomposition for correlated normals
  L <- chol(R)
  Z_ind <- matrix(rnorm(n * d), nrow = n, ncol = d)
  Z_cor <- Z_ind %*% L
  # Transform to uniform marginals
  pnorm(Z_cor)
}

# -----------------------------------------------------------------------------
# 3. STUDENT-T COPULA
# -----------------------------------------------------------------------------

#' Fit Student-t copula via profile likelihood
#' The t-copula captures symmetric TAIL DEPENDENCE -- critical for crypto
#' where joint crashes are more common than a Gaussian copula would predict
#' @param U matrix of uniform marginals
#' @param nu_grid degrees of freedom to search over
#' @return list with R and nu
fit_t_copula <- function(U, nu_grid = seq(3, 30, by = 1)) {
  d <- ncol(U)
  best_ll <- -Inf
  best_nu <- nu_grid[1]
  best_R  <- diag(d)

  for (nu in nu_grid) {
    # Transform uniforms to t-scores with nu df
    Z <- qt(U, df = nu)
    R_candidate <- cor(Z)
    # Ensure positive definiteness
    ev <- eigen(R_candidate, symmetric = TRUE)
    if (any(ev$values <= 0)) next
    ll <- t_copula_loglik(U, R_candidate, nu)
    if (ll > best_ll) {
      best_ll <- ll
      best_nu <- nu
      best_R  <- R_candidate
    }
  }
  list(type = "t", R = best_R, nu = best_nu, loglik = best_ll, dim = d)
}

#' Log-likelihood of t-copula
t_copula_loglik <- function(U, R, nu) {
  Z <- qt(U, df = nu)
  n <- nrow(Z)
  d <- ncol(Z)
  Ri <- tryCatch(solve(R), error = function(e) NULL)
  if (is.null(Ri)) return(-Inf)
  log_det_R <- log(det(R))
  ll <- 0
  for (i in seq_len(n)) {
    z <- Z[i, ]
    quad <- as.numeric(t(z) %*% Ri %*% z)
    # multivariate t density minus sum of marginal t densities
    ll_mv <- lgamma((nu + d) / 2) - lgamma(nu / 2) -
              (d / 2) * log(nu * pi) - 0.5 * log_det_R -
              ((nu + d) / 2) * log(1 + quad / nu)
    ll_margins <- sum(dt(z, df = nu, log = TRUE))
    ll <- ll + ll_mv - ll_margins
  }
  ll
}

#' Simulate from Student-t copula
simulate_t_copula <- function(n, R, nu) {
  d <- nrow(R)
  L <- chol(R)
  Z_ind <- matrix(rnorm(n * d), nrow = n, ncol = d)
  Z_cor <- Z_ind %*% L
  # Scale by chi-squared to get multivariate t
  chi_sq <- rchisq(n, df = nu)
  T_mat <- Z_cor / sqrt(chi_sq / nu)
  # Transform to uniform marginals
  pt(T_mat, df = nu)
}

#' Tail dependence coefficient for t-copula
#' lambda_U = lambda_L = 2 * t_{nu+1}(-sqrt((nu+1)*(1-rho)/(1+rho)))
t_copula_tail_dependence <- function(R, nu) {
  d <- nrow(R)
  lambda <- matrix(0, d, d)
  for (i in 1:(d-1)) {
    for (j in (i+1):d) {
      rho <- R[i, j]
      val <- sqrt((nu + 1) * (1 - rho) / (1 + rho + 1e-10))
      td <- 2 * pt(-val, df = nu + 1)
      lambda[i, j] <- td
      lambda[j, i] <- td
    }
  }
  lambda
}

# -----------------------------------------------------------------------------
# 4. ARCHIMEDEAN COPULAS (bivariate, manual implementations)
# -----------------------------------------------------------------------------

#' Clayton copula CDF: C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}
#' Captures lower tail dependence (joint crashes) -- very relevant for crypto
#' @param u,v uniform marginals in (0,1)
#' @param theta Clayton parameter (theta > 0; higher = more lower tail dep)
clayton_cdf <- function(u, v, theta) {
  if (theta <= 0) stop("Clayton theta must be positive")
  pmax((u^(-theta) + v^(-theta) - 1)^(-1/theta), 0)
}

#' Clayton copula density (for log-likelihood)
clayton_density <- function(u, v, theta) {
  C <- clayton_cdf(u, v, theta)
  (1 + theta) * (u * v)^(-theta - 1) * C^(-2 - 1/theta)
}

#' Gumbel copula CDF: C(u,v) = exp(-[(-log u)^theta + (-log v)^theta]^{1/theta})
#' Captures upper tail dependence (joint booms) -- good for bull market co-movement
#' @param theta >= 1; theta=1 means independence, larger = more upper tail dep
gumbel_cdf <- function(u, v, theta) {
  if (theta < 1) stop("Gumbel theta must be >= 1")
  A <- (-log(u))^theta + (-log(v))^theta
  exp(-A^(1/theta))
}

gumbel_density <- function(u, v, theta) {
  lu <- -log(u); lv <- -log(v)
  A <- lu^theta + lv^theta
  C <- exp(-A^(1/theta))
  C / (u * v) * A^(2/theta - 2) * (lu * lv)^(theta - 1) *
    (A^(1/theta) + theta - 1)
}

#' Frank copula: C(u,v) = -1/theta * log(1 + (e^{-theta*u}-1)(e^{-theta*v}-1)/(e^{-theta}-1))
#' Symmetric tail dependence (actually NO tail dependence), but flexible middle
frank_cdf <- function(u, v, theta) {
  if (abs(theta) < 1e-10) return(u * v)  # independence limit
  num <- (exp(-theta * u) - 1) * (exp(-theta * v) - 1)
  denom <- exp(-theta) - 1
  -1/theta * log(1 + num / denom)
}

frank_density <- function(u, v, theta) {
  if (abs(theta) < 1e-10) return(rep(1, length(u)))
  e_theta <- exp(-theta)
  ea <- exp(-theta * u)
  eb <- exp(-theta * v)
  denom <- exp(-theta) - 1
  num <- (ea - 1) * (eb - 1) / denom
  numer <- -theta * e_theta * (1 + num)^(-1) * ... # simplified form:
  # Use the analytic form directly
  theta * (1 - exp(-theta)) * exp(-theta * (u + v)) /
    (1 - exp(-theta) - (1 - exp(-theta*u)) * (1 - exp(-theta*v)))^2
}

# Cleaner Frank density implementation
frank_density <- function(u, v, theta) {
  if (abs(theta) < 1e-10) return(rep(1, length(u)))
  a <- exp(-theta * u)
  b <- exp(-theta * v)
  c <- exp(-theta)
  theta * (1 - c) * a * b / ((1 - c) - (1 - a) * (1 - b))^2
}

# -----------------------------------------------------------------------------
# 5. KENDALL'S TAU AND COPULA PARAMETER INVERSION
# -----------------------------------------------------------------------------

#' Compute Kendall's tau between two series
#' Kendall tau is copula-invariant: tau = 4*E[C(U,V)] - 1
#' @return Kendall's tau (numeric)
kendall_tau <- function(x, y) {
  n <- length(x)
  concordant <- 0
  discordant <- 0
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      dx <- x[j] - x[i]
      dy <- y[j] - y[i]
      if (dx * dy > 0) concordant <- concordant + 1
      else if (dx * dy < 0) discordant <- discordant + 1
    }
  }
  (concordant - discordant) / (n * (n-1) / 2)
}

#' Fast Kendall tau using sorting (O(n log n))
kendall_tau_fast <- function(x, y) {
  n <- length(x)
  ord <- order(x)
  y_ord <- y[ord]
  # Count inversions in y_ord using merge sort
  count_inversions <- function(v) {
    n <- length(v)
    if (n <= 1) return(list(sorted = v, inv = 0))
    mid <- n %/% 2
    left  <- count_inversions(v[1:mid])
    right <- count_inversions(v[(mid+1):n])
    inv <- left$inv + right$inv
    merged <- numeric(n)
    i <- 1; j <- 1; k <- 1
    while (i <= length(left$sorted) && j <= length(right$sorted)) {
      if (left$sorted[i] <= right$sorted[j]) {
        merged[k] <- left$sorted[i]; i <- i + 1
      } else {
        merged[k] <- right$sorted[j]
        inv <- inv + length(left$sorted) - i + 1
        j <- j + 1
      }
      k <- k + 1
    }
    while (i <= length(left$sorted))  { merged[k] <- left$sorted[i];  i <- i+1; k <- k+1 }
    while (j <= length(right$sorted)) { merged[k] <- right$sorted[j]; j <- j+1; k <- k+1 }
    list(sorted = merged, inv = inv)
  }
  res <- count_inversions(y_ord)
  n_pairs <- n * (n - 1) / 2
  concordant <- n_pairs - res$inv
  discordant <- res$inv
  (concordant - discordant) / n_pairs
}

#' Invert tau -> copula parameter
#' Clayton: tau = theta / (theta + 2)  =>  theta = 2*tau / (1 - tau)
#' Gumbel:  tau = 1 - 1/theta           =>  theta = 1 / (1 - tau)
#' Frank:   tau solved numerically
tau_to_clayton <- function(tau) {
  if (tau <= 0) stop("Clayton requires positive tau")
  2 * tau / (1 - tau)
}

tau_to_gumbel <- function(tau) {
  if (tau <= 0) stop("Gumbel requires positive tau")
  1 / (1 - tau)
}

tau_to_frank <- function(tau, tol = 1e-8) {
  # Debye function: D1(theta) = (1/theta) * integral_0^theta t/(exp(t)-1) dt
  # tau = 1 - 4/theta * (1 - D1(theta)) -- solved numerically via bisection
  debye1 <- function(theta, n_terms = 100) {
    s <- 0
    for (k in 1:n_terms) s <- s + theta^k / (k * factorial(k))  # series expansion (unstable for large theta)
    # Numerical integration instead:
    integrate(function(t) t / (exp(t) - 1), 0, theta)$value / theta
  }
  frank_tau <- function(theta) 1 - 4 / theta * (1 - debye1(theta))
  # Bisect
  lo <- 1e-6; hi <- 100
  for (i in 1:100) {
    mid <- (lo + hi) / 2
    if (frank_tau(mid) < tau) lo <- mid else hi <- mid
    if (hi - lo < tol) break
  }
  (lo + hi) / 2
}

# -----------------------------------------------------------------------------
# 6. COPULA SIMULATION VIA CONDITIONAL METHOD
# -----------------------------------------------------------------------------

#' Simulate bivariate Clayton copula via conditional distribution
#' C(v|u) = u^{-(theta+1)} * (u^{-theta} + v^{-theta} - 1)^{-(1 + 1/theta)}
simulate_clayton <- function(n, theta) {
  u <- runif(n)
  t_vals <- runif(n)
  # Conditional quantile function C^{-1}(t|u)
  # v = (u^{-theta} * (t^{-theta/(theta+1)} - 1) + 1)^{-1/theta}
  v <- (u^(-theta) * (t_vals^(-theta/(theta+1)) - 1) + 1)^(-1/theta)
  cbind(u = u, v = pmax(pmin(v, 1-1e-10), 1e-10))
}

#' Simulate bivariate Gumbel copula via Marshall-Olkin method
simulate_gumbel <- function(n, theta) {
  # Use the stable distribution approach
  # Generate stable(1/theta, 1) random variable via Chambers-Mallows-Stuck
  alpha <- 1 / theta
  V <- runif(n) * pi
  E <- rexp(n)
  S <- sin(alpha * V) / sin(V)^(1/alpha) *
       (sin((1 - alpha) * V) / E)^((1 - alpha)/alpha)
  # Generate two independent exponentials
  E1 <- rexp(n); E2 <- rexp(n)
  u <- exp(-(E1/S)^(1/theta))
  v <- exp(-(E2/S)^(1/theta))
  cbind(u = pmax(pmin(u, 1-1e-10), 1e-10),
        v = pmax(pmin(v, 1-1e-10), 1e-10))
}

#' Simulate bivariate Frank copula via conditional method
simulate_frank <- function(n, theta) {
  u <- runif(n)
  t_vals <- runif(n)
  # Conditional quantile: v = -1/theta * log(1 + t*(e^{-theta}-1)/(t*(e^{-theta*u}-1) - e^{-theta*u} + ... ))
  # Simpler explicit form:
  e_t <- exp(-theta)
  eu  <- exp(-theta * u)
  v <- -1/theta * log(1 + t_vals * (e_t - 1) / (t_vals * (eu - 1) - eu + 1e-15))
  cbind(u = u, v = pmax(pmin(v, 1-1e-10), 1e-10))
}

# -----------------------------------------------------------------------------
# 7. TAIL DEPENDENCE COEFFICIENTS
# -----------------------------------------------------------------------------

#' Empirical lower tail dependence coefficient
#' lambda_L = lim_{q->0} P(V < q | U < q)
#' @param U matrix (n x 2) of uniform marginals
#' @param q_grid quantile thresholds to use
empirical_lower_tail_dep <- function(U, q_grid = seq(0.01, 0.15, by = 0.01)) {
  u <- U[, 1]; v <- U[, 2]
  lambdas <- sapply(q_grid, function(q) {
    mean(v < q & u < q) / q
  })
  # Extrapolate to q=0 by linear regression on log scale
  fit <- lm(lambdas ~ q_grid)
  list(lambdas = lambdas, q_grid = q_grid,
       extrapolated = max(coef(fit)[1], 0))
}

#' Empirical upper tail dependence coefficient
empirical_upper_tail_dep <- function(U, q_grid = seq(0.85, 0.99, by = 0.01)) {
  u <- U[, 1]; v <- U[, 2]
  lambdas <- sapply(q_grid, function(q) {
    mean(v > q & u > q) / (1 - q)
  })
  list(lambdas = lambdas, q_grid = q_grid,
       extrapolated = max(lambdas[length(lambdas)], 0))
}

#' Summary of tail dependence for all copula types
tail_dependence_summary <- function(U) {
  lower <- empirical_lower_tail_dep(U)
  upper <- empirical_upper_tail_dep(U)
  tau   <- kendall_tau_fast(U[,1], U[,2])
  theta_clayton <- tau_to_clayton(max(tau, 0.01))
  theta_gumbel  <- tau_to_gumbel(max(tau, 0.01))
  # Clayton: lambda_L = 2^{-1/theta}, lambda_U = 0
  clayton_lower <- 2^(-1/theta_clayton)
  # Gumbel: lambda_L = 0, lambda_U = 2 - 2^{1/theta}
  gumbel_upper  <- 2 - 2^(1/theta_gumbel)

  cat("=== Tail Dependence Summary ===\n")
  cat(sprintf("Kendall tau:              %.4f\n", tau))
  cat(sprintf("Empirical lower tail dep: %.4f\n", lower$extrapolated))
  cat(sprintf("Empirical upper tail dep: %.4f\n", upper$extrapolated))
  cat(sprintf("Clayton implied lambda_L: %.4f\n", clayton_lower))
  cat(sprintf("Gumbel implied lambda_U:  %.4f\n", gumbel_upper))
  invisible(list(tau = tau, lower = lower, upper = upper,
                 clayton_lower = clayton_lower, gumbel_upper = gumbel_upper))
}

# -----------------------------------------------------------------------------
# 8. PORTFOLIO VaR UNDER COPULA DEPENDENCE (Monte Carlo, 10K sims)
# -----------------------------------------------------------------------------

#' Fit parametric marginals: returns a list of (mu, sigma, nu) per asset
#' Uses t-distribution for heavy tails -- important for crypto
fit_t_marginals <- function(X) {
  lapply(seq_len(ncol(X)), function(j) {
    x <- X[, j]
    mu    <- mean(x)
    sigma <- sd(x)
    # MOM estimation of degrees of freedom: nu = 2*kurtosis/(kurtosis-3) + 4
    k4 <- mean((x - mu)^4) / sigma^4  # excess kurtosis
    nu <- if (k4 > 3) max(4, 2 * k4 / (k4 - 3)) else 30
    list(mu = mu, sigma = sigma, nu = nu)
  })
}

#' Portfolio VaR via copula Monte Carlo simulation
#' @param U_hist historical uniform marginals (used to fit copula)
#' @param marginals list of fitted marginal parameters
#' @param weights portfolio weights (sum to 1)
#' @param n_sim number of Monte Carlo simulations
#' @param alpha VaR confidence level (e.g., 0.95)
#' @param copula_type "gaussian", "t", "clayton", "gumbel"
copula_portfolio_var <- function(U_hist, marginals, weights,
                                  n_sim = 10000, alpha = 0.95,
                                  copula_type = "t") {
  d <- ncol(U_hist)
  stopifnot(length(weights) == d, length(marginals) == d)

  # Fit copula
  cop <- switch(copula_type,
    gaussian = fit_gaussian_copula(U_hist),
    t        = fit_t_copula(U_hist),
    stop("Use gaussian or t for multivariate")
  )

  # Simulate from copula
  U_sim <- switch(copula_type,
    gaussian = simulate_gaussian_copula(n_sim, cop$R),
    t        = simulate_t_copula(n_sim, cop$R, cop$nu)
  )

  # Invert marginals: uniform -> returns
  R_sim <- matrix(0, nrow = n_sim, ncol = d)
  for (j in seq_len(d)) {
    m <- marginals[[j]]
    R_sim[, j] <- m$mu + m$sigma * qt(U_sim[, j], df = m$nu)
  }

  # Portfolio returns
  port_ret <- R_sim %*% weights

  # VaR and ES
  var_alpha <- quantile(port_ret, probs = 1 - alpha)
  es_alpha  <- mean(port_ret[port_ret <= var_alpha])

  list(VaR = -var_alpha, ES = -es_alpha,
       port_ret_sim = port_ret, copula = cop,
       alpha = alpha, n_sim = n_sim)
}

# -----------------------------------------------------------------------------
# 9. TIME-VARYING COPULA (rolling windows)
# -----------------------------------------------------------------------------

#' Estimate rolling copula parameters to detect changing dependence regimes
#' @param U matrix of uniform marginals
#' @param window rolling window size
#' @param copula_type "gaussian" or "t"
rolling_copula <- function(U, window = 60, copula_type = "gaussian") {
  n <- nrow(U)
  d <- ncol(U)
  results <- vector("list", n - window + 1)

  for (i in seq_len(n - window + 1)) {
    U_w <- U[i:(i + window - 1), ]
    cop <- switch(copula_type,
      gaussian = fit_gaussian_copula(U_w),
      t        = fit_t_copula(U_w, nu_grid = c(4, 6, 8, 10, 15, 20))
    )
    results[[i]] <- cop
  }

  # Extract correlation time series (first pair as example)
  if (d == 2) {
    rho_ts <- sapply(results, function(cop) cop$R[1, 2])
    nu_ts  <- if (copula_type == "t") sapply(results, function(cop) cop$nu) else NULL
    return(list(rho_ts = rho_ts, nu_ts = nu_ts, copulas = results))
  }

  list(copulas = results)
}

#' Detect structural breaks in rolling copula correlations
#' Uses CUSUM statistic on rolling rho series
copula_structural_break <- function(rho_ts) {
  n <- length(rho_ts)
  mu <- mean(rho_ts)
  sigma <- sd(rho_ts)
  cusum <- cumsum(rho_ts - mu) / (sigma * sqrt(n))
  break_point <- which.max(abs(cusum))
  critical_val <- 1.358  # 5% level for CUSUM test
  list(cusum = cusum,
       break_point = break_point,
       max_cusum = max(abs(cusum)),
       significant = max(abs(cusum)) > critical_val)
}

# -----------------------------------------------------------------------------
# 10. REGIME-CONDITIONAL COPULAS
# -----------------------------------------------------------------------------

#' Classify market regimes based on return quantiles
#' @param market_ret market-wide return series (e.g., BTC)
#' @param bull_threshold quantile above which = "bull"
#' @param bear_threshold quantile below which = "bear"
classify_regimes <- function(market_ret,
                              bull_threshold = 0.7,
                              bear_threshold = 0.3) {
  q <- quantile(market_ret, probs = c(bear_threshold, bull_threshold))
  regime <- rep("neutral", length(market_ret))
  regime[market_ret > q[2]] <- "bull"
  regime[market_ret < q[1]] <- "bear"
  # Stress: bottom 10%
  q_stress <- quantile(market_ret, 0.10)
  regime[market_ret < q_stress] <- "stress"
  regime
}

#' Fit separate copulas for each regime
#' @param U matrix of uniform marginals for the pair of interest
#' @param regime character vector of regime labels
regime_copulas <- function(U, regime) {
  unique_regimes <- unique(regime)
  result <- list()

  for (r in unique_regimes) {
    idx <- which(regime == r)
    if (length(idx) < 20) {
      result[[r]] <- list(n_obs = length(idx), note = "insufficient data")
      next
    }
    U_r <- U[idx, , drop = FALSE]
    # Re-rank within regime
    U_r_ranked <- returns_to_copula_data(U_r)
    cop_g <- fit_gaussian_copula(U_r_ranked)
    cop_t <- fit_t_copula(U_r_ranked, nu_grid = c(4, 6, 8, 10, 20))
    tau_r <- kendall_tau_fast(U_r_ranked[,1], U_r_ranked[,2])
    result[[r]] <- list(
      n_obs = length(idx),
      gaussian = cop_g,
      t_cop    = cop_t,
      kendall_tau = tau_r,
      regime = r
    )
  }
  result
}

# -----------------------------------------------------------------------------
# 11. GOODNESS-OF-FIT TESTS
# -----------------------------------------------------------------------------

#' Rosenblatt transform-based GOF test for copulas
#' Transform (U,V) through the copula's conditional distributions;
#' if the copula is correct, the transformed variables should be uniform
rosenblatt_gof <- function(U, copula_type = "gaussian", cop_params = NULL) {
  n <- nrow(U)
  u <- U[, 1]; v <- U[, 2]

  if (copula_type == "gaussian") {
    rho <- if (!is.null(cop_params)) cop_params$R[1,2] else cor(qnorm(u), qnorm(v))
    # Conditional CDF of V given U under Gaussian copula:
    # C(v|u) = Phi((Phi^{-1}(v) - rho*Phi^{-1}(u)) / sqrt(1 - rho^2))
    z_u <- qnorm(u)
    z_v <- qnorm(v)
    e1 <- u  # first transformed variable is just U
    e2 <- pnorm((z_v - rho * z_u) / sqrt(1 - rho^2))
  } else if (copula_type == "t") {
    nu  <- cop_params$nu
    rho <- cop_params$R[1, 2]
    t_u <- qt(u, df = nu)
    t_v <- qt(v, df = nu)
    e1  <- u
    # Conditional t quantile
    scale <- sqrt((nu + t_u^2) * (1 - rho^2) / (nu + 1))
    e2 <- pt((t_v - rho * t_u) / scale, df = nu + 1)
  } else {
    stop("GOF currently supports gaussian and t")
  }

  # Test if e1, e2 are Uniform(0,1) using KS test
  ks_e1 <- ks.test(e1, "punif")
  ks_e2 <- ks.test(e2, "punif")

  # Cramer-von Mises statistic on joint uniforms
  cvm_stat <- sum((sort(e1) - (seq_len(n) - 0.5) / n)^2) +
              sum((sort(e2) - (seq_len(n) - 0.5) / n)^2)

  cat("=== Copula GOF Test (Rosenblatt) ===\n")
  cat(sprintf("KS test e1: stat=%.4f, p=%.4f\n", ks_e1$statistic, ks_e1$p.value))
  cat(sprintf("KS test e2: stat=%.4f, p=%.4f\n", ks_e2$statistic, ks_e2$p.value))
  cat(sprintf("CvM statistic: %.4f\n", cvm_stat))

  list(ks_e1 = ks_e1, ks_e2 = ks_e2, cvm = cvm_stat,
       e1 = e1, e2 = e2)
}

#' AIC/BIC model selection across copula types
copula_model_selection <- function(U) {
  d <- ncol(U)
  if (d != 2) stop("Model selection currently for bivariate only")
  u <- U[,1]; v <- U[,2]
  n <- nrow(U)

  results <- list()

  # Gaussian copula
  cop_g <- fit_gaussian_copula(U)
  ll_g  <- gaussian_copula_loglik(U, cop_g$R)
  results[["Gaussian"]] <- list(loglik = ll_g, k = 1,
                                 AIC = -2*ll_g + 2*1,
                                 BIC = -2*ll_g + log(n)*1,
                                 params = cop_g)

  # t copula
  cop_t <- fit_t_copula(U)
  ll_t  <- cop_t$loglik
  results[["t"]] <- list(loglik = ll_t, k = 2,
                          AIC = -2*ll_t + 2*2,
                          BIC = -2*ll_t + log(n)*2,
                          params = cop_t)

  # Clayton copula
  tau <- kendall_tau_fast(u, v)
  if (tau > 0.01) {
    theta_c <- tau_to_clayton(tau)
    ll_c <- sum(log(pmax(clayton_density(u, v, theta_c), 1e-300)))
    results[["Clayton"]] <- list(loglik = ll_c, k = 1,
                                  AIC = -2*ll_c + 2,
                                  BIC = -2*ll_c + log(n),
                                  params = list(theta = theta_c))
  }

  # Gumbel copula
  if (tau > 0.01) {
    theta_gu <- tau_to_gumbel(tau)
    ll_gu <- sum(log(pmax(gumbel_density(u, v, theta_gu), 1e-300)))
    results[["Gumbel"]] <- list(loglik = ll_gu, k = 1,
                                 AIC = -2*ll_gu + 2,
                                 BIC = -2*ll_gu + log(n),
                                 params = list(theta = theta_gu))
  }

  # Frank copula
  theta_f <- tryCatch(tau_to_frank(tau), error = function(e) NA)
  if (!is.na(theta_f)) {
    ll_f <- sum(log(pmax(frank_density(u, v, theta_f), 1e-300)))
    results[["Frank"]] <- list(loglik = ll_f, k = 1,
                                AIC = -2*ll_f + 2,
                                BIC = -2*ll_f + log(n),
                                params = list(theta = theta_f))
  }

  # Sort by AIC
  aics <- sapply(results, function(r) r$AIC)
  sorted_idx <- order(aics)
  cat("=== Copula Model Selection ===\n")
  for (nm in names(results)[sorted_idx]) {
    cat(sprintf("%-10s  loglik=%8.2f  AIC=%8.2f  BIC=%8.2f\n",
                nm, results[[nm]]$loglik,
                results[[nm]]$AIC, results[[nm]]$BIC))
  }
  invisible(results[sorted_idx])
}

# -----------------------------------------------------------------------------
# 12. FULL ANALYSIS PIPELINE
# -----------------------------------------------------------------------------

#' Run complete copula analysis on a returns matrix
#' @param returns_mat matrix of log-returns, each column = one asset
#' @param market_col index of the market (e.g., BTC) column for regime
#' @param weights portfolio weights
#' @return comprehensive list of results
run_copula_analysis <- function(returns_mat, market_col = 1,
                                 weights = NULL, n_sim = 10000) {
  d <- ncol(returns_mat)
  n <- nrow(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("Asset", seq_len(d))
  if (is.null(weights)) weights <- rep(1/d, d)

  cat("=============================================================\n")
  cat("COPULA ANALYSIS PIPELINE\n")
  cat(sprintf("Assets: %s\n", paste(asset_names, collapse=", ")))
  cat(sprintf("Observations: %d\n\n", n))

  # Step 1: Transform to pseudo-uniforms
  U <- returns_to_copula_data(returns_mat)

  # Step 2: Kendall tau matrix
  tau_mat <- matrix(0, d, d)
  for (i in 1:d) for (j in 1:d) {
    if (i == j) tau_mat[i,j] <- 1
    else if (i < j) {
      t_val <- kendall_tau_fast(U[,i], U[,j])
      tau_mat[i,j] <- t_val; tau_mat[j,i] <- t_val
    }
  }
  cat("Kendall Tau Matrix:\n")
  print(round(tau_mat, 3))

  # Step 3: Fit copulas (bivariate for first pair as example)
  if (d >= 2) {
    cat("\n--- Bivariate analysis: first pair ---\n")
    U2 <- U[, 1:2]
    model_sel <- copula_model_selection(U2)
    tail_dep  <- tail_dependence_summary(U2)
  }

  # Step 4: Multivariate t copula
  cat("\n--- Multivariate t-Copula ---\n")
  cop_t <- fit_t_copula(U)
  cat(sprintf("Fitted nu: %.1f\n", cop_t$nu))
  cat("Correlation matrix:\n")
  print(round(cop_t$R, 3))

  # Step 5: Portfolio VaR
  cat("\n--- Portfolio VaR (Monte Carlo) ---\n")
  marginals <- fit_t_marginals(returns_mat)
  var_result <- copula_portfolio_var(U, marginals, weights,
                                      n_sim = n_sim, alpha = 0.95,
                                      copula_type = "t")
  cat(sprintf("95%% VaR (t-copula): %.4f\n", var_result$VaR))
  cat(sprintf("95%% ES  (t-copula): %.4f\n", var_result$ES))

  # Compare with Gaussian copula VaR
  var_gauss <- copula_portfolio_var(U, marginals, weights,
                                    n_sim = n_sim, alpha = 0.95,
                                    copula_type = "gaussian")
  cat(sprintf("95%% VaR (Gaussian copula): %.4f\n", var_gauss$VaR))
  cat(sprintf("Tail risk uplift (t vs Gaussian): %.2f%%\n",
              100 * (var_result$VaR / var_gauss$VaR - 1)))

  # Step 6: Rolling copula
  cat("\n--- Rolling Copula (window=60) ---\n")
  if (d == 2 && n >= 80) {
    roll <- rolling_copula(U, window = 60)
    cat(sprintf("Rho range: [%.3f, %.3f]\n",
                min(roll$rho_ts), max(roll$rho_ts)))
    brk <- copula_structural_break(roll$rho_ts)
    cat(sprintf("Structural break: obs %d, significant=%s\n",
                brk$break_point, brk$significant))
  }

  # Step 7: Regime copulas
  cat("\n--- Regime-Conditional Copulas ---\n")
  regime <- classify_regimes(returns_mat[, market_col])
  table_regime <- table(regime)
  print(table_regime)
  if (d >= 2) {
    reg_cops <- regime_copulas(U[, 1:2, drop=FALSE], regime)
    for (r in names(reg_cops)) {
      rc <- reg_cops[[r]]
      if (!is.null(rc$kendall_tau)) {
        cat(sprintf("Regime %-8s: n=%3d, tau=%.3f\n",
                    r, rc$n_obs, rc$kendall_tau))
      }
    }
  }

  # Step 8: GOF test on best copula
  cat("\n--- Copula GOF Test ---\n")
  if (d >= 2) {
    gof <- rosenblatt_gof(U[,1:2], "t", cop_t)
  }

  invisible(list(
    U = U, tau_mat = tau_mat,
    t_copula = cop_t,
    var_t = var_result, var_gauss = var_gauss,
    regime_copulas = if (d >= 2) reg_cops else NULL
  ))
}

# -----------------------------------------------------------------------------
# EXAMPLE USAGE (commented out)
# -----------------------------------------------------------------------------
# set.seed(42)
# n <- 500
# # Simulate correlated crypto-like returns (fat tails, positive correlation)
# R_true <- matrix(c(1, 0.7, 0.5,
#                    0.7, 1, 0.6,
#                    0.5, 0.6, 1), 3, 3)
# L <- chol(R_true)
# Z <- matrix(rnorm(n * 3), n, 3) %*% L
# # Add heavy tails by mixing with high-variance normal
# mix <- rbinom(n, 1, 0.05)
# shocks <- matrix(rnorm(n*3, 0, 5), n, 3)
# returns_sim <- Z * 0.03 + mix * shocks * 0.03
# colnames(returns_sim) <- c("BTC", "ETH", "SOL")
# result <- run_copula_analysis(returns_sim, weights = c(0.5, 0.3, 0.2))
