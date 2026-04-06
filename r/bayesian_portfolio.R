# bayesian_portfolio.R
#
# Bayesian portfolio optimization.
#
# Methods covered:
#   1. Black-Litterman (BL): combine CAPM equilibrium prior with investor views
#   2. Hierarchical Risk Parity (HRP): Ward linkage cluster-based allocation
#   3. Bayesian shrinkage: James-Stein and Ledoit-Wolf estimators
#   4. Robust optimization: minimax regret and worst-case CVaR
#   5. Bayesian updating: Dirichlet-updated returns prior
#   6. MCMC posterior via Stan (RStan)
#
# Black-Litterman model:
#   Prior: μ₀ = (δ · Σ · w_mkt)  [implied CAPM returns]
#   Views: P·μ = q + ε,  ε ~ N(0, Ω)
#   Posterior: μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹μ₀ + PᵀΩ⁻¹q]
#              Σ_BL  = Σ + [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹
#
# References:
#   Black & Litterman (1992) "Global Portfolio Optimization"
#   López de Prado (2016) "Building Diversified Portfolios that Outperform Out-of-Sample"
#   Ledoit & Wolf (2004) "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"
#   Jagannathan & Ma (2003) "Risk Reduction in Large Portfolios"
#
# Packages: MASS, quadprog, PortfolioAnalytics, covRobust, PerformanceAnalytics,
#           ggplot2, dplyr, reshape2

suppressPackageStartupMessages({
  library(MASS)                  # mvrnorm, ginv
  library(quadprog)              # solve.QP for mean-variance optimization
  library(PerformanceAnalytics)  # portfolio analytics utilities
  library(ggplot2)
  library(dplyr)
  library(reshape2)
  library(gridExtra)
  library(scales)
})

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

#' Ensure a matrix is symmetric positive definite by adding small diagonal.
make_spd <- function(M, epsilon = 1e-8) {
  M <- (M + t(M)) / 2
  eig <- eigen(M, symmetric = TRUE)$values
  if (any(eig <= 0)) {
    M <- M + (abs(min(eig)) + epsilon) * diag(nrow(M))
  }
  return(M)
}

#' Portfolio Sharpe ratio (annualised).
sharpe_ratio <- function(weights, mu, Sigma, rf = 0, periods_per_year = 252) {
  port_ret <- sum(weights * mu)
  port_var <- as.numeric(t(weights) %*% Sigma %*% weights)
  (port_ret - rf) / sqrt(port_var) * sqrt(periods_per_year)
}

#' Portfolio volatility (annualised).
port_vol <- function(weights, Sigma, periods_per_year = 252) {
  sqrt(as.numeric(t(weights) %*% Sigma %*% weights) * periods_per_year)
}

#' Mean-variance optimization: maximise μᵀw - (γ/2)wᵀΣw.
#'
#' Solved as QP: min (γ/2)wᵀΣw - μᵀw, s.t. Σwᵢ=1, wᵢ ≥ 0
mv_optimize <- function(mu, Sigma, gamma = 2.0, long_only = TRUE) {
  n <- length(mu)
  Sigma <- make_spd(Sigma)

  # quadprog: min (1/2)xᵀDx - dᵀx
  Dmat <- gamma * Sigma
  dvec <- mu

  # Constraints: 1ᵀw = 1 (equality), w ≥ 0 (inequality if long_only)
  if (long_only) {
    Amat <- cbind(rep(1, n), diag(n))
    bvec <- c(1, rep(0, n))
    meq  <- 1
  } else {
    Amat <- matrix(rep(1, n), ncol = 1)
    bvec <- 1
    meq  <- 1
  }

  sol <- tryCatch(
    solve.QP(Dmat, dvec, Amat, bvec, meq),
    error = function(e) {
      # Fallback to equal weight
      list(solution = rep(1/n, n))
    }
  )
  w <- sol$solution
  w[abs(w) < 1e-8] <- 0
  w / sum(w)
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. BLACK-LITTERMAN MODEL
# ─────────────────────────────────────────────────────────────────────────────

#' Compute CAPM implied equilibrium excess returns.
#'
#' μ_eq = δ · Σ · w_mkt
#'
#' where δ is the risk-aversion coefficient calibrated to:
#'   δ = (E[r_mkt] - r_f) / Var(r_mkt)
#'
#' @param Sigma    n×n covariance matrix (annualised)
#' @param w_mkt   n-vector of market cap weights
#' @param delta   risk-aversion parameter (default 2.5)
capm_equilibrium_returns <- function(Sigma, w_mkt, delta = 2.5) {
  stopifnot(abs(sum(w_mkt) - 1) < 1e-8)
  mu_eq <- delta * Sigma %*% w_mkt
  return(as.vector(mu_eq))
}

#' Black-Litterman posterior.
#'
#' Combine CAPM prior with K views:
#'   Prior:  μ ~ N(μ_eq, τ·Σ)
#'   Views:  P·μ ~ N(q, Ω)
#'
#' Posterior (conjugate):
#'   Σ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹
#'   μ_BL = Σ_BL · [(τΣ)⁻¹μ_eq + PᵀΩ⁻¹q]
#'
#' @param mu_eq  n-vector of equilibrium returns
#' @param Sigma  n×n covariance matrix
#' @param P      K×n view matrix (K views on n assets)
#' @param q      K-vector of view returns (e.g. c(0.05, -0.02))
#' @param Omega  K×K view uncertainty matrix (diagonal is common)
#' @param tau    scalar scaling the prior uncertainty (typically 1/n to 0.05)
#' @param views_description  character vector describing each view
black_litterman <- function(mu_eq, Sigma, P, q, Omega = NULL,
                             tau = 0.05, views_description = NULL) {
  n <- length(mu_eq)
  K <- nrow(P)

  stopifnot(ncol(P) == n, length(q) == K)

  # Default Omega: proportional to P·Σ·Pᵀ (He & Litterman 1999)
  if (is.null(Omega)) {
    Omega <- tau * P %*% Sigma %*% t(P)
  }
  Omega <- make_spd(Omega)

  # Precision matrices
  tau_Sigma_inv <- solve(tau * Sigma)
  Omega_inv     <- solve(Omega)

  # Posterior precision and mean
  Sigma_BL_inv <- tau_Sigma_inv + t(P) %*% Omega_inv %*% P
  Sigma_BL     <- solve(Sigma_BL_inv)
  mu_BL        <- as.vector(Sigma_BL %*% (tau_Sigma_inv %*% mu_eq + t(P) %*% Omega_inv %*% q))

  # Combined posterior covariance (including estimation uncertainty)
  Sigma_post <- Sigma + Sigma_BL

  cat("\nBlack-Litterman Results:\n")
  cat(sprintf("  τ = %.4f, K = %d views\n", tau, K))
  cat("\n  Views:\n")
  for (k in 1:K) {
    desc <- if (!is.null(views_description)) views_description[k] else paste0("View ", k)
    cat(sprintf("    [%d] %s: q=%.4f\n", k, desc, q[k]))
  }

  cat("\n  Prior vs Posterior returns:\n")
  cat(sprintf("  %-5s %-12s %-12s %-12s\n", "Asset", "μ_eq", "μ_BL", "Δ"))
  for (i in 1:n) {
    cat(sprintf("  %-5d %-12.4f %-12.4f %-12.4f\n",
                i, mu_eq[i], mu_BL[i], mu_BL[i] - mu_eq[i]))
  }

  return(list(
    mu_BL    = mu_BL,
    Sigma_BL = Sigma_BL,
    Sigma_post = Sigma_post,
    mu_eq    = mu_eq,
    tau      = tau
  ))
}

#' Full BL pipeline: CAPM prior + views → optimal weights.
black_litterman_weights <- function(returns_matrix, w_mkt = NULL,
                                     P = NULL, q = NULL, Omega = NULL,
                                     delta = 2.5, tau = 0.05,
                                     gamma = 2.0, long_only = TRUE,
                                     asset_names = NULL,
                                     views_description = NULL) {
  n <- ncol(returns_matrix)
  if (is.null(w_mkt)) w_mkt <- rep(1/n, n)
  if (is.null(asset_names)) asset_names <- if (!is.null(colnames(returns_matrix)))
    colnames(returns_matrix) else paste0("A", 1:n)

  # Sample covariance
  Sigma <- cov(returns_matrix)

  # CAPM equilibrium
  mu_eq <- capm_equilibrium_returns(Sigma, w_mkt, delta)

  # If no views, use prior only
  if (is.null(P) || is.null(q)) {
    cat("No views specified — using CAPM equilibrium prior.\n")
    mu_use <- mu_eq
    Sigma_use <- Sigma
    bl_result <- NULL
  } else {
    bl_result  <- black_litterman(mu_eq, Sigma, P, q, Omega, tau, views_description)
    mu_use     <- bl_result$mu_BL
    Sigma_use  <- bl_result$Sigma_post
  }

  # Optimal weights
  w_opt <- mv_optimize(mu_use, Sigma_use, gamma, long_only)
  names(w_opt) <- asset_names

  cat("\nOptimal BL Portfolio Weights:\n")
  for (i in seq_along(w_opt)) {
    cat(sprintf("  %-8s: %.4f (%.2f%%)\n",
                asset_names[i], w_opt[i], 100*w_opt[i]))
  }

  sr <- sharpe_ratio(w_opt, mu_use, Sigma_use)
  vol <- port_vol(w_opt, Sigma_use)
  cat(sprintf("\n  Expected return: %.4f\n", sum(w_opt * mu_use) * 252))
  cat(sprintf("  Annual volatility: %.4f\n", vol))
  cat(sprintf("  Sharpe ratio: %.4f\n", sr))

  return(list(weights = w_opt, mu = mu_use, Sigma = Sigma_use, bl = bl_result))
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. HIERARCHICAL RISK PARITY (HRP)
# ─────────────────────────────────────────────────────────────────────────────

#' Hierarchical Risk Parity via Ward linkage.
#'
#' Algorithm (López de Prado 2016):
#'   1. Build correlation-based distance matrix: dᵢⱼ = √(2(1-ρᵢⱼ))
#'   2. Hierarchical clustering (Ward linkage)
#'   3. Quasi-diagonalise (reorder) the covariance matrix
#'   4. Allocate via recursive bisection: equal-risk contribution within each cluster
#'
#' HRP avoids the instability of mean-variance optimization (inverted covariance matrix)
#' and produces more diversified, out-of-sample robust portfolios.
hrp_weights <- function(returns_matrix, asset_names = NULL) {
  n  <- ncol(returns_matrix)
  if (is.null(asset_names)) asset_names <- colnames(returns_matrix)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:n)

  Sigma <- cov(returns_matrix)
  Corr  <- cor(returns_matrix)

  # 1. Correlation-based distance matrix
  D <- sqrt(2 * (1 - Corr))

  # 2. Hierarchical clustering (Ward.D2)
  dist_obj <- as.dist(D)
  hc <- hclust(dist_obj, method = "ward.D2")

  # 3. Quasi-diagonalisation: reorder assets by dendrogram
  leaf_order <- order.dendrogram(as.dendrogram(hc))

  # 4. Recursive bisection allocation
  weights <- rep(1.0, n)
  names(weights) <- asset_names

  # Work with sorted asset indices
  clusters <- list(leaf_order)

  while (length(clusters) > 0) {
    # Split each cluster into two halves
    new_clusters <- list()

    for (cluster in clusters) {
      if (length(cluster) <= 1) next

      half <- length(cluster) %/% 2
      left  <- cluster[1:half]
      right <- cluster[(half+1):length(cluster)]

      # Variance of each sub-portfolio (equal weight within sub-cluster)
      w_left  <- weights[left] / sum(weights[left])
      w_right <- weights[right] / sum(weights[right])

      var_left  <- as.numeric(t(w_left) %*% Sigma[left, left] %*% w_left)
      var_right <- as.numeric(t(w_right) %*% Sigma[right, right] %*% w_right)

      # Alpha: weight allocated to left cluster (inverse-variance)
      alpha <- 1 - var_left / (var_left + var_right)

      # Scale weights
      weights[left]  <- weights[left]  * alpha
      weights[right] <- weights[right] * (1 - alpha)

      if (length(left)  > 1) new_clusters <- c(new_clusters, list(left))
      if (length(right) > 1) new_clusters <- c(new_clusters, list(right))
    }

    clusters <- new_clusters
  }

  weights <- weights / sum(weights)  # normalise

  cat("\nHRP Portfolio Weights:\n")
  for (i in order(weights, decreasing = TRUE)) {
    cat(sprintf("  %-8s: %.4f (%.2f%%)\n",
                asset_names[i], weights[i], 100*weights[i]))
  }

  vol_hrp <- port_vol(weights, Sigma)
  cat(sprintf("  Portfolio volatility (annualised): %.4f\n", vol_hrp))
  cat(sprintf("  Effective N (diversification): %.2f\n",
              1/sum(weights^2)))

  return(list(
    weights = weights,
    hclust  = hc,
    order   = leaf_order,
    Sigma   = Sigma,
    Corr    = Corr,
    D       = D
  ))
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. BAYESIAN SHRINKAGE ESTIMATORS
# ─────────────────────────────────────────────────────────────────────────────

#' James-Stein shrinkage estimator for the mean vector.
#'
#' Shrinks the sample mean μ̂ towards a target μ₀ (typically 0):
#'   μ_JS = (1 - c_JS) · μ̂   where c_JS = (n-2)/(n · ||μ̂||²) · s²
#'
#' The scalar c_JS is the shrinkage intensity — greater when p/n is large.
#'
#' @param X  n × p matrix of returns
#' @param target  shrinkage target (length-p vector or scalar); default = grand mean
james_stein_mean <- function(X, target = NULL) {
  n <- nrow(X); p <- ncol(X)
  mu_hat <- colMeans(X)
  S <- cov(X)

  if (is.null(target)) target <- rep(mean(mu_hat), p)

  # James-Stein shrinkage
  diff_vec <- mu_hat - target
  S_inv    <- tryCatch(solve(S / n), error = function(e) ginv(S / n))
  quad     <- as.numeric(t(diff_vec) %*% S_inv %*% diff_vec)

  # Shrinkage coefficient (Cohen-Strawderman optimal)
  c_JS <- max(0, (p - 2) / (n * max(quad, 1e-10)))
  c_JS <- min(c_JS, 1)  # bound at [0,1]

  mu_JS <- target + (1 - c_JS) * diff_vec

  cat(sprintf("James-Stein shrinkage:\n"))
  cat(sprintf("  Shrinkage intensity c = %.4f\n", c_JS))
  cat(sprintf("  ||μ̂ - μ₀||: %.4f → %.4f (reduction: %.1f%%)\n",
              sqrt(sum(diff_vec^2)),
              sqrt(sum((mu_JS - target)^2)),
              100 * c_JS))

  return(list(mu = mu_JS, shrinkage = c_JS, mu_ols = mu_hat))
}

#' Ledoit-Wolf analytical shrinkage estimator for the covariance matrix.
#'
#' Shrinks sample covariance Σ̂ towards scaled identity μ·I:
#'   Σ_LW = (1-α)·Σ̂ + α·μ·I
#'
#' Optimal α is found analytically (Ledoit-Wolf 2004, Oracle approximation):
#'   α* = min(1, (ρ²/(δ²)) · (1/n))
#'
#' where ρ² measures distance from Σ̂ to target, δ² is variance of Σ̂.
#'
#' @param X  n × p matrix of returns
#' @param target  shrinkage target: "identity", "diagonal", or matrix
ledoit_wolf <- function(X, target = "identity") {
  n <- nrow(X); p <- ncol(X)
  Sigma_hat <- cov(X)

  # Shrinkage target
  if (target == "identity") {
    mu <- sum(diag(Sigma_hat)) / p
    T_target <- mu * diag(p)
  } else if (target == "diagonal") {
    T_target <- diag(diag(Sigma_hat))
  } else {
    T_target <- target
  }

  # Oracle approximation (Ledoit-Wolf 2004, equations 14-17)
  # Compute optimal α analytically
  X_c <- scale(X, center = TRUE, scale = FALSE)

  delta <- sum((Sigma_hat - T_target)^2) / p
  gamma <- sum(Sigma_hat^2) / p

  # Shrinkage intensity approximation
  alpha_lw <- max(0, min(1, delta / (delta + gamma) * (p / n)))

  Sigma_LW <- (1 - alpha_lw) * Sigma_hat + alpha_lw * T_target

  cat(sprintf("Ledoit-Wolf shrinkage (target: %s):\n", target))
  cat(sprintf("  Shrinkage intensity α = %.4f\n", alpha_lw))
  cat(sprintf("  Condition number: %.2e → %.2e\n",
              kappa(Sigma_hat), kappa(Sigma_LW)))

  return(list(
    Sigma  = Sigma_LW,
    alpha  = alpha_lw,
    Sigma_hat = Sigma_hat,
    target = T_target
  ))
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. ROBUST OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

#' Minimax regret portfolio.
#'
#' Minimises the maximum regret across a discrete set of scenarios.
#' Regret under scenario s: r_s = max_w' R_s(w') - R_s(w)
#'
#' Solved as a linear program (via LP relaxation).
#'
#' @param mu_scenarios  K × n matrix; row k = expected returns under scenario k
#' @param Sigma        n × n covariance matrix
minimax_regret <- function(mu_scenarios, Sigma, long_only = TRUE) {
  K <- nrow(mu_scenarios)
  n <- ncol(mu_scenarios)

  cat(sprintf("Minimax regret: %d assets, %d scenarios\n", n, K))

  # For each scenario, compute the maximum achievable return
  r_max <- numeric(K)
  for (k in 1:K) {
    # Best weight for scenario k
    w_k <- mv_optimize(mu_scenarios[k,], Sigma, gamma = 1, long_only)
    r_max[k] <- sum(w_k * mu_scenarios[k,])
  }

  # Grid search for minimax regret (simple but robust)
  # For small n, enumerate candidate portfolios
  best_weights <- rep(1/n, n)
  best_max_regret <- Inf

  # Candidate portfolios: individual assets + equal weight + MV per scenario
  candidates <- list(rep(1/n, n))
  for (k in 1:K) {
    w_k <- mv_optimize(mu_scenarios[k,], Sigma, gamma = 2, long_only)
    candidates <- c(candidates, list(w_k))
  }
  for (i in 1:n) {
    w_i <- rep(0, n); w_i[i] <- 1
    candidates <- c(candidates, list(w_i))
  }

  for (w in candidates) {
    regrets <- numeric(K)
    for (k in 1:K) {
      regrets[k] <- r_max[k] - sum(w * mu_scenarios[k,])
    }
    max_reg <- max(regrets)
    if (max_reg < best_max_regret) {
      best_max_regret <- max_reg
      best_weights <- w
    }
  }

  cat(sprintf("  Minimax regret: %.4f\n", best_max_regret))
  return(list(weights = best_weights, max_regret = best_max_regret))
}

#' Worst-case CVaR portfolio (robust CVaR under distributional uncertainty).
#'
#' Uses a neighbourhood of plausible distributions (bootstrap resample):
#' min_w max_{P' ∈ B(P̂)} CVaR_α(r(w)) ≈ min_w (1/B) Σ_b CVaR_α(r_b(w))
#'
#' where r_b is portfolio return under bootstrap resample b.
worst_case_cvar <- function(returns_matrix, alpha = 0.05,
                             n_bootstrap = 200, long_only = TRUE) {
  n <- ncol(returns_matrix)
  T <- nrow(returns_matrix)

  cat(sprintf("Worst-case CVaR (α=%.2f, B=%d bootstrap samples)\n",
              alpha, n_bootstrap))

  # Bootstrap resamples
  best_weights <- rep(1/n, n)
  min_max_cvar <- Inf

  # Grid search over candidate portfolios
  # Build candidates from mean-variance efficient frontier
  candidates <- list(rep(1/n, n))  # equal weight

  for (g in c(0.5, 1, 2, 5, 10)) {
    mu_s  <- colMeans(returns_matrix)
    Sig_s <- cov(returns_matrix)
    w_g   <- mv_optimize(mu_s, Sig_s, gamma = g, long_only)
    candidates <- c(candidates, list(w_g))
  }

  for (w in candidates) {
    # Bootstrap worst-case CVaR
    cvar_vals <- numeric(n_bootstrap)
    for (b in 1:n_bootstrap) {
      idx <- sample(T, T, replace = TRUE)
      port_ret <- returns_matrix[idx, ] %*% w
      cvar_vals[b] <- -mean(port_ret[port_ret <= quantile(port_ret, alpha)])
    }
    wc_cvar <- max(cvar_vals)

    if (wc_cvar < min_max_cvar) {
      min_max_cvar <- wc_cvar
      best_weights <- w
    }
  }

  cat(sprintf("  Worst-case CVaR: %.4f\n", min_max_cvar))
  return(list(weights = best_weights, worst_case_cvar = min_max_cvar))
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. BAYESIAN UPDATING WITH LIVE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

#' Bayesian Normal-Inverse-Wishart conjugate update.
#'
#' Prior for (μ, Σ): Normal-Inverse-Wishart(μ₀, κ₀, ν₀, Ψ₀)
#' Posterior after n observations: NIW(μₙ, κₙ, νₙ, Ψₙ)
#'
#' Update equations:
#'   κₙ = κ₀ + n
#'   νₙ = ν₀ + n
#'   μₙ = (κ₀μ₀ + nX̄) / κₙ
#'   Ψₙ = Ψ₀ + S + (κ₀n/κₙ)(X̄-μ₀)(X̄-μ₀)ᵀ
#'
#' @param X     new observations (n_new × p matrix)
#' @param prior list(mu0, kappa0, nu0, Psi0)
#' @return  updated posterior parameters
niw_update <- function(X, prior) {
  n <- nrow(X); p <- ncol(X)
  X_bar <- colMeans(X)
  S <- (n-1) * cov(X)  # sum of squares (not divided by n)

  kappa_n <- prior$kappa0 + n
  nu_n    <- prior$nu0 + n
  mu_n    <- (prior$kappa0 * prior$mu0 + n * X_bar) / kappa_n
  Psi_n   <- prior$Psi0 + S +
    (prior$kappa0 * n / kappa_n) * outer(X_bar - prior$mu0, X_bar - prior$mu0)

  # Posterior predictive (Student-t):
  # E[μ | X] = μₙ
  # E[Σ | X] = Ψₙ / (νₙ - p - 1)  [for ν > p+1]
  Sigma_post <- Psi_n / (nu_n - p - 1)

  cat(sprintf("NIW Bayesian Update: n=%d new observations\n", n))
  cat(sprintf("  κ: %.1f → %.1f\n", prior$kappa0, kappa_n))
  cat(sprintf("  ν: %.1f → %.1f\n", prior$nu0, nu_n))
  cat(sprintf("  Prior mean: %s\n", paste(round(prior$mu0, 4), collapse=", ")))
  cat(sprintf("  Post mean:  %s\n", paste(round(mu_n, 4), collapse=", ")))

  return(list(
    mu0    = mu_n,
    kappa0 = kappa_n,
    nu0    = nu_n,
    Psi0   = Psi_n,
    Sigma  = Sigma_post
  ))
}

#' Sequential Bayesian updating: incorporate live returns into prior.
#'
#' @param returns_historical  historical returns matrix (prior training set)
#' @param returns_live        new live returns (to update prior)
#' @param prior_strength      prior κ₀ (higher = stronger prior)
bayesian_update_portfolio <- function(returns_historical,
                                       returns_live,
                                       prior_strength = 10,
                                       long_only = TRUE) {
  p <- ncol(returns_historical)

  # Initialise prior from historical data
  mu0   <- colMeans(returns_historical)
  Sigma0 <- cov(returns_historical)
  nu0   <- p + 2

  prior <- list(
    mu0    = mu0,
    kappa0 = prior_strength,
    nu0    = nu0,
    Psi0   = Sigma0 * nu0
  )

  # Update with live data
  posterior <- niw_update(returns_live, prior)

  # Optimize with posterior parameters
  w_prior <- mv_optimize(prior$mu0, prior$Psi0/prior$nu0, gamma=2, long_only)
  w_post  <- mv_optimize(posterior$mu0, posterior$Sigma, gamma=2, long_only)

  cat("\nPortfolio comparison (before/after live update):\n")
  cat(sprintf("  %-5s %-12s %-12s\n", "Asset", "Prior w", "Post w"))
  for (i in 1:p) {
    cat(sprintf("  %-5d %-12.4f %-12.4f\n", i, w_prior[i], w_post[i]))
  }

  return(list(prior = prior, posterior = posterior,
              w_prior = w_prior, w_post = w_post))
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. MCMC VIA STAN (OPTIONAL — requires rstan)
# ─────────────────────────────────────────────────────────────────────────────

#' Bayesian portfolio model via Stan MCMC.
#'
#' Model:
#'   μ ~ Normal(μ₀, σ_μ²·I)     [prior on mean returns]
#'   L ~ LKJ(η)                  [prior on Cholesky factor of correlation]
#'   σ ~ HalfNormal(0, σ_σ)     [prior on asset volatilities]
#'   Σ = diag(σ)·L·Lᵀ·diag(σ)
#'   rₜ ~ Normal(μ, Σ)           [likelihood]
#'
#' The LKJ prior (Lewandowski-Kurowicka-Joe) on the correlation matrix:
#'   p(R) ∝ det(R)^{η-1}  — favours sparse correlations for η>1
#'
#' @param returns_matrix  n × p matrix of returns
#' @param n_iter    MCMC iterations (after warmup)
#' @param warmup    number of warmup/burnin iterations
mcmc_bayesian_portfolio <- function(returns_matrix,
                                     n_iter = 1000, warmup = 500,
                                     chains = 2) {
  # Check if rstan is available
  if (!requireNamespace("rstan", quietly = TRUE)) {
    cat("rstan not installed. Returning analytical approximation instead.\n")
    return(analytical_posterior(returns_matrix))
  }

  library(rstan)

  p <- ncol(returns_matrix)
  n <- nrow(returns_matrix)

  stan_code <- "
  data {
    int<lower=1> N;          // observations
    int<lower=1> P;          // assets
    matrix[N, P] R;          // returns matrix
    vector[P] mu0;           // prior mean
    real<lower=0> sigma_mu;  // prior sd for mean
    real<lower=0> eta;       // LKJ concentration
  }
  parameters {
    vector[P] mu;            // asset mean returns
    vector<lower=0>[P] sigma; // asset volatilities
    cholesky_factor_corr[P] L_corr; // Cholesky correlation
  }
  transformed parameters {
    matrix[P,P] Sigma;
    Sigma = diag_pre_multiply(sigma, L_corr) *
            diag_pre_multiply(sigma, L_corr)';
  }
  model {
    // Priors
    mu    ~ normal(mu0, sigma_mu);
    sigma ~ normal(0, 0.1);
    L_corr ~ lkj_corr_cholesky(eta);

    // Likelihood
    for (t in 1:N)
      R[t] ~ multi_normal_cholesky(mu, diag_pre_multiply(sigma, L_corr));
  }
  generated quantities {
    // Posterior predictive check
    vector[P] r_pred;
    r_pred = multi_normal_cholesky_rng(mu, diag_pre_multiply(sigma, L_corr));
  }
  "

  stan_data <- list(
    N = n, P = p, R = returns_matrix,
    mu0 = colMeans(returns_matrix),
    sigma_mu = 0.5,
    eta = 2.0
  )

  cat("Running Stan MCMC...\n")
  fit <- tryCatch({
    stan(model_code = stan_code, data = stan_data,
         iter = n_iter + warmup, warmup = warmup,
         chains = chains, refresh = 0,
         control = list(adapt_delta = 0.9))
  }, error = function(e) {
    cat("Stan sampling failed:", conditionMessage(e), "\n")
    return(NULL)
  })

  if (is.null(fit)) {
    return(analytical_posterior(returns_matrix))
  }

  # Extract posterior
  posterior_samples <- extract(fit)
  mu_post_mean  <- colMeans(posterior_samples$mu)
  Sigma_post_mean <- apply(posterior_samples$Sigma, c(2,3), mean)

  cat("Stan MCMC complete.\n")
  cat(sprintf("  Posterior mean returns: %s\n",
              paste(round(mu_post_mean, 4), collapse=", ")))

  w_mcmc <- mv_optimize(mu_post_mean, Sigma_post_mean, gamma = 2)

  return(list(
    mu_post   = mu_post_mean,
    Sigma_post = Sigma_post_mean,
    weights   = w_mcmc,
    stan_fit  = fit
  ))
}

#' Analytical posterior (Normal-Inverse-Wishart) as Stan fallback.
analytical_posterior <- function(returns_matrix) {
  p <- ncol(returns_matrix)
  n <- nrow(returns_matrix)

  mu_hat    <- colMeans(returns_matrix)
  Sigma_hat <- cov(returns_matrix)

  # NIW posterior predictive t-distribution approximation
  nu <- n + p + 1
  Sigma_pred <- Sigma_hat * (n + 1) / (n * (nu - p - 1))

  w <- mv_optimize(mu_hat, Sigma_pred, gamma = 2)

  return(list(mu_post = mu_hat, Sigma_post = Sigma_pred, weights = w))
}

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

#' Plot weight comparison across portfolio methods.
plot_weights_comparison <- function(weight_list, method_names = NULL) {
  if (is.null(method_names)) method_names <- paste0("Method ", seq_along(weight_list))

  p <- length(weight_list[[1]])
  asset_names <- if (!is.null(names(weight_list[[1]]))) names(weight_list[[1]])
                 else paste0("A", 1:p)

  df <- data.frame()
  for (i in seq_along(weight_list)) {
    df <- rbind(df, data.frame(
      method = method_names[i],
      asset  = asset_names,
      weight = as.numeric(weight_list[[i]])
    ))
  }

  ggplot(df, aes(x = asset, y = weight, fill = method)) +
    geom_bar(stat = "identity", position = "dodge") +
    geom_hline(yintercept = 1/p, linetype = "dashed", color = "gray40") +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Portfolio Weight Comparison",
         x = "Asset", y = "Weight", fill = "Method") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Plot correlation matrix heatmap.
plot_correlation_heatmap <- function(Corr, title = "Asset Correlation Matrix",
                                      asset_names = NULL) {
  d <- nrow(Corr)
  if (is.null(asset_names)) asset_names <- paste0("X", 1:d)
  rownames(Corr) <- colnames(Corr) <- asset_names

  df_corr <- melt(Corr)
  ggplot(df_corr, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    geom_text(aes(label = round(value, 2)), size = 3) +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                         midpoint = 0, limits = c(-1,1), name = "ρ") +
    labs(title = title, x = "", y = "") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Plot HRP dendrogram with weight annotations.
plot_hrp_dendrogram <- function(hrp_result, asset_names = NULL) {
  hc <- hrp_result$hclust
  w  <- hrp_result$weights
  if (is.null(asset_names)) asset_names <- names(w)

  # Plot dendrogram
  plot(hc, labels = asset_names, main = "HRP Dendrogram (Ward linkage)",
       sub = "", xlab = "", hang = -1)

  # Add weight annotations at leaves
  leaf_order <- order.dendrogram(as.dendrogram(hc))
  for (i in seq_along(leaf_order)) {
    leaf <- leaf_order[i]
    mtext(sprintf("%.1f%%", 100*w[leaf]),
          side = 1, at = i, line = 3, cex = 0.7, col = "steelblue")
  }
}

#' Plot efficient frontier comparison.
plot_efficient_frontier <- function(returns_matrix, mu_alt = NULL, Sigma_alt = NULL,
                                     n_points = 50) {
  mu     <- colMeans(returns_matrix)
  Sigma  <- cov(returns_matrix)
  n      <- length(mu)

  # Generate frontier by varying risk aversion γ
  gammas <- exp(seq(log(0.1), log(50), length.out = n_points))

  compute_frontier_point <- function(mu_use, Sigma_use, g) {
    w <- mv_optimize(mu_use, Sigma_use, gamma = g)
    list(ret = sum(w * mu_use) * 252,
         vol = port_vol(w, Sigma_use))
  }

  # Sample covariance frontier
  frontier_sample <- lapply(gammas, function(g)
    compute_frontier_point(mu, Sigma, g))
  df_sample <- data.frame(
    vol = sapply(frontier_sample, `[[`, "vol"),
    ret = sapply(frontier_sample, `[[`, "ret"),
    method = "Sample"
  )

  df_all <- df_sample

  # Shrinkage frontier if provided
  if (!is.null(mu_alt) && !is.null(Sigma_alt)) {
    frontier_alt <- lapply(gammas, function(g)
      compute_frontier_point(mu_alt, Sigma_alt, g))
    df_alt <- data.frame(
      vol = sapply(frontier_alt, `[[`, "vol"),
      ret = sapply(frontier_alt, `[[`, "ret"),
      method = "BL / Shrinkage"
    )
    df_all <- rbind(df_all, df_alt)
  }

  ggplot(df_all, aes(x = vol, y = ret, color = method)) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(values = c("Sample" = "steelblue", "BL / Shrinkage" = "darkorange")) +
    labs(title = "Efficient Frontier Comparison",
         x = "Annual Volatility", y = "Annual Expected Return",
         color = "Method") +
    scale_x_continuous(labels = percent) +
    scale_y_continuous(labels = percent) +
    theme_minimal()
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

demo_bayesian_portfolio <- function() {
  set.seed(42)
  cat("================================================================\n")
  cat("Bayesian Portfolio Optimization Demo\n")
  cat("================================================================\n\n")

  # Simulate 5 assets: 3 crypto + 2 equity-like
  n_assets <- 5
  n_obs    <- 252  # 1 year daily
  asset_names <- c("BTC", "ETH", "BNB", "SPX", "BOND")

  # True covariance (annualised)
  sigma_true <- c(0.80, 0.85, 0.75, 0.20, 0.05)  # daily vols × √252
  Corr_true  <- matrix(c(
    1.00, 0.85, 0.75, 0.30, -0.10,
    0.85, 1.00, 0.80, 0.25, -0.08,
    0.75, 0.80, 1.00, 0.20, -0.05,
    0.30, 0.25, 0.20, 1.00,  0.10,
   -0.10,-0.08,-0.05, 0.10,  1.00
  ), 5, 5)
  Sigma_true <- diag(sigma_true) %*% Corr_true %*% diag(sigma_true) / 252

  mu_true <- c(0.0008, 0.0007, 0.0006, 0.0003, 0.0001)  # daily expected returns

  returns_matrix <- mvrnorm(n_obs, mu_true, Sigma_true)
  colnames(returns_matrix) <- asset_names

  cat("Simulated", n_obs, "daily returns for", n_assets, "assets.\n")
  cat("True annual Sharpe ratios:\n")
  for (i in 1:n_assets) {
    cat(sprintf("  %-6s: E[r]=%+.4f, σ=%.4f, SR=%.3f\n",
                asset_names[i], mu_true[i]*252,
                sigma_true[i]/sqrt(252),
                mu_true[i]*252 / sigma_true[i]))
  }

  # 1. Black-Litterman
  cat("\n" %+% paste(rep("─", 50), collapse="") %+% "\n")
  cat("1. Black-Litterman Model\n")
  cat(paste(rep("─", 50), collapse="") %+% "\n")

  w_mkt <- c(0.40, 0.25, 0.15, 0.15, 0.05)  # market cap weights
  names(w_mkt) <- asset_names

  # Views:
  #   View 1: BTC will outperform ETH by 2% annually
  #   View 2: SPX will return 8% annually
  P <- rbind(
    c(1, -1, 0, 0, 0),    # BTC - ETH
    c(0,  0, 0, 1, 0)     # SPX absolute
  )
  q <- c(0.02/252, 0.08/252)  # daily view magnitudes
  views_desc <- c("BTC > ETH by 2% p.a.", "SPX returns 8% p.a.")

  bl_result <- black_litterman_weights(
    returns_matrix, w_mkt = w_mkt,
    P = P, q = q, tau = 0.05,
    asset_names = asset_names,
    views_description = views_desc
  )

  # 2. HRP
  cat("\n" %+% paste(rep("─", 50), collapse="") %+% "\n")
  cat("2. Hierarchical Risk Parity\n")
  cat(paste(rep("─", 50), collapse="") %+% "\n")
  hrp_result <- hrp_weights(returns_matrix, asset_names = asset_names)

  # 3. Shrinkage
  cat("\n" %+% paste(rep("─", 50), collapse="") %+% "\n")
  cat("3. Bayesian Shrinkage\n")
  cat(paste(rep("─", 50), collapse="") %+% "\n")

  js_result <- james_stein_mean(returns_matrix)
  lw_result <- ledoit_wolf(returns_matrix, target = "identity")
  w_shrink   <- mv_optimize(js_result$mu, lw_result$Sigma, gamma = 2)
  names(w_shrink) <- asset_names

  # 4. Robust optimization
  cat("\n" %+% paste(rep("─", 50), collapse="") %+% "\n")
  cat("4. Robust Optimization\n")
  cat(paste(rep("─", 50), collapse="") %+% "\n")

  # Scenarios: bull, bear, sideways
  mu_scenarios <- rbind(
    mu_true * 2,                 # bull
    mu_true * (-1),              # bear
    mu_true * 0.5                # sideways
  )
  mm_result <- minimax_regret(mu_scenarios, cov(returns_matrix))
  wc_result <- worst_case_cvar(returns_matrix, alpha = 0.05, n_bootstrap = 100)

  # 5. Bayesian update with live data
  cat("\n" %+% paste(rep("─", 50), collapse="") %+% "\n")
  cat("5. Sequential Bayesian Update\n")
  cat(paste(rep("─", 50), collapse="") %+% "\n")

  live_returns <- mvrnorm(30, mu_true * 1.5, Sigma_true)  # live data (slightly better)
  colnames(live_returns) <- asset_names

  update_result <- bayesian_update_portfolio(returns_matrix, live_returns,
                                              prior_strength = 20)

  # Weight summary
  w_equal    <- rep(1/n_assets, n_assets); names(w_equal) <- asset_names
  w_mvo      <- mv_optimize(colMeans(returns_matrix), cov(returns_matrix), gamma=2)
  names(w_mvo) <- asset_names

  cat("\n" %+% paste(rep("─", 60), collapse="") %+% "\n")
  cat("Portfolio Weight Summary:\n")
  cat(sprintf("%-10s", "Asset"))
  method_names <- c("Equal", "MVO", "BL", "HRP", "Shrinkage")
  all_weights  <- list(w_equal, w_mvo, bl_result$weights,
                        hrp_result$weights, w_shrink)
  for (nm in method_names) cat(sprintf("%-12s", nm))
  cat("\n")

  for (i in 1:n_assets) {
    cat(sprintf("%-10s", asset_names[i]))
    for (w_method in all_weights) cat(sprintf("%-12.4f", w_method[i]))
    cat("\n")
  }

  # Plots
  cat("\nGenerating plots...\n")

  p_weights <- plot_weights_comparison(all_weights, method_names)
  ggsave("bayesian_weights.png", p_weights, width = 10, height = 5)

  p_corr <- plot_correlation_heatmap(cor(returns_matrix), asset_names = asset_names)
  ggsave("bayesian_corr.png", p_corr, width = 6, height = 5)

  p_frontier <- plot_efficient_frontier(returns_matrix,
                                         mu_alt  = bl_result$mu,
                                         Sigma_alt = bl_result$Sigma)
  ggsave("bayesian_frontier.png", p_frontier, width = 7, height = 5)

  png("bayesian_hrp_dendrogram.png", width = 700, height = 400)
  plot_hrp_dendrogram(hrp_result, asset_names = asset_names)
  dev.off()

  cat("Saved: bayesian_weights.png, bayesian_corr.png,\n")
  cat("       bayesian_frontier.png, bayesian_hrp_dendrogram.png\n")
  cat("\nBayesian portfolio demo complete.\n")

  return(list(
    bl     = bl_result,
    hrp    = hrp_result,
    shrink = list(js = js_result, lw = lw_result, weights = w_shrink),
    robust = list(minimax = mm_result, wc_cvar = wc_result),
    update = update_result
  ))
}

if (!interactive()) {
  demo_bayesian_portfolio()
}
