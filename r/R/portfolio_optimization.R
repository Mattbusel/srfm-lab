# portfolio_optimization.R
# Portfolio optimization methods for SRFM research.
# Implements: MVO (quadprog), risk parity (PortfolioAnalytics),
#             Black-Litterman (full), HRP (dendextend), robust optimization (MASS)
# Dependencies: quadprog, MASS, ggplot2, zoo, xts

library(zoo)
library(xts)
library(ggplot2)
library(MASS)

.has_quadprog <- requireNamespace("quadprog", quietly = TRUE)
.has_pa       <- requireNamespace("PortfolioAnalytics", quietly = TRUE)
.has_dendext  <- requireNamespace("dendextend", quietly = TRUE)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Mean-Variance Optimization (quadprog)
# ─────────────────────────────────────────────────────────────────────────────

#' mvo_optimize
#'
#' Mean-variance optimization using quadratic programming.
#' Minimizes: w' * Sigma * w - lambda * w' * mu
#' subject to: sum(w) = 1, w >= lb
#'
#' @param mu numeric vector of expected returns
#' @param Sigma covariance matrix (n × n)
#' @param lambda numeric, risk aversion (Inf = minimum variance)
#' @param lb numeric or vector, lower bound on weights (default 0 = long-only)
#' @param ub numeric or vector, upper bound on weights (default 1)
#' @param target_ret numeric, target return (for efficient frontier point)
#' @return list with weights, exp_return, exp_vol, sharpe
mvo_optimize <- function(mu, Sigma, lambda = 2.0, lb = 0.0, ub = 1.0,
                          target_ret = NULL) {
  n <- length(mu)
  lb <- rep(lb, length.out = n)
  ub <- rep(ub, length.out = n)

  if (.has_quadprog) {
    return(.mvo_quadprog(mu, Sigma, lambda, lb, ub, target_ret))
  }

  # Fallback: projected gradient
  .mvo_projected_gradient(mu, Sigma, lambda, lb, ub)
}


#' .mvo_quadprog
#'
#' Internal: MVO using quadprog::solve.QP
.mvo_quadprog <- function(mu, Sigma, lambda, lb, ub, target_ret) {
  library(quadprog)
  n <- length(mu)

  # Regularize Sigma
  Sigma_reg <- Sigma + 1e-8 * diag(n)

  # QP: min w' D w - d' w  =>  D = Sigma, d = (lambda/2) * mu
  Dmat <- 2 * Sigma_reg
  dvec <- lambda * mu

  # Constraints:
  # 1. sum(w) = 1         (equality)
  # 2. w_i >= lb_i         (inequality)
  # 3. w_i <= ub_i = -(-w_i) >= -ub_i (inequality via negation)
  # Optional: 4. w' mu = target_ret (equality)

  # Amat: each column is a constraint vector (Amat' w >= bvec)
  Amat <- cbind(
    matrix(1, n, 1),      # sum = 1 (equality: counted twice via meq)
    diag(n),              # w >= lb
    -diag(n)              # w <= ub  => -w >= -ub
  )
  bvec <- c(1, lb, -ub)
  meq  <- 1L

  if (!is.null(target_ret)) {
    Amat <- cbind(Amat, mu)
    bvec <- c(bvec, target_ret)
    meq  <- 2L
  }

  sol <- tryCatch(
    quadprog::solve.QP(Dmat, dvec, Amat, bvec, meq = meq),
    error = function(e) NULL
  )

  if (is.null(sol)) {
    # Fallback: equal weights
    w <- rep(1/n, n)
  } else {
    w <- sol$solution
    w <- pmax(pmin(w, ub), lb)
    w <- w / sum(w)
  }

  .portfolio_stats(w, mu, Sigma)
}


#' .mvo_projected_gradient
#'
#' Fallback MVO via projected gradient descent.
.mvo_projected_gradient <- function(mu, Sigma, lambda, lb, ub, max_iter = 2000L) {
  n <- length(mu)
  w <- rep(1/n, n)

  step <- 1.0 / (2 * max(eigen(Sigma, only.values = TRUE)$values))

  for (iter in seq_len(max_iter)) {
    # Gradient: 2*Sigma*w - lambda*mu
    grad <- 2 * Sigma %*% w - lambda * mu
    w_new <- w - step * grad
    # Project onto simplex with box constraints
    w_new <- project_simplex(w_new, lb, ub)
    if (max(abs(w_new - w)) < 1e-8) break
    w <- w_new
  }

  .portfolio_stats(w, mu, Sigma)
}


#' project_simplex
#'
#' Project vector onto probability simplex with box constraints [lb, ub].
project_simplex <- function(v, lb, ub) {
  n   <- length(v)
  v_c <- pmin(pmax(v, lb), ub)
  # Now project onto sum = 1 via water-filling
  excess <- sum(v_c) - 1.0
  if (abs(excess) < 1e-10) return(v_c)

  # Simple iterative projection
  for (iter in seq_len(100)) {
    adj  <- excess / n
    v_c  <- pmin(pmax(v_c - adj, lb), ub)
    excess <- sum(v_c) - 1.0
    if (abs(excess) < 1e-10) break
  }
  v_c
}


#' efficient_frontier
#'
#' Compute the efficient frontier by varying lambda (or target return).
#'
#' @param mu numeric vector of expected returns
#' @param Sigma covariance matrix
#' @param n_points integer, number of frontier points
#' @param lb numeric, lower weight bound
#' @param ub numeric, upper weight bound
#' @return data.frame with: exp_return, exp_vol, sharpe, weights_matrix
efficient_frontier <- function(mu, Sigma, n_points = 50L, lb = 0.0, ub = 1.0,
                                rf = 0.0) {
  lambdas <- exp(seq(log(0.01), log(200), length.out = n_points))

  results <- lapply(lambdas, function(lam) {
    sol <- mvo_optimize(mu, Sigma, lambda = lam, lb = lb, ub = ub)
    c(exp_return = sol$exp_return, exp_vol = sol$exp_vol, sharpe = sol$sharpe,
      lambda = lam)
  })

  df <- as.data.frame(do.call(rbind, results))
  df <- df[order(df$exp_vol), ]
  # Remove dominated points
  max_ret <- -Inf
  keep    <- logical(nrow(df))
  for (i in seq_len(nrow(df))) {
    if (df$exp_return[i] >= max_ret) {
      keep[i]  <- TRUE
      max_ret  <- df$exp_return[i]
    }
  }
  df <- df[keep, ]
  df
}


.portfolio_stats <- function(w, mu, Sigma) {
  exp_ret <- as.numeric(t(w) %*% mu)
  exp_vol <- as.numeric(sqrt(t(w) %*% Sigma %*% w))
  sharpe  <- if (exp_vol > 1e-10) exp_ret / exp_vol else 0.0
  list(weights = w, exp_return = exp_ret, exp_vol = exp_vol, sharpe = sharpe)
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Risk Parity
# ─────────────────────────────────────────────────────────────────────────────

#' risk_parity_weights
#'
#' Compute risk parity (equal risk contribution) weights.
#' Each asset contributes equally to portfolio volatility.
#' Uses iterative Newton-Raphson method.
#'
#' @param Sigma covariance matrix
#' @param budget numeric vector of risk budget (default: equal 1/n)
#' @param max_iter integer
#' @param tol numeric convergence tolerance
#' @return list with weights, risk_contributions, portfolio_vol
risk_parity_weights <- function(Sigma, budget = NULL, max_iter = 500L, tol = 1e-10) {
  n <- nrow(Sigma)
  if (is.null(budget)) budget <- rep(1/n, n)
  budget <- budget / sum(budget)

  # Initialize with 1/vol weights
  vols <- sqrt(diag(Sigma))
  vols[vols < 1e-10] <- 1e-10
  w <- (1 / vols) / sum(1 / vols)

  for (iter in seq_len(max_iter)) {
    # Portfolio variance
    port_var <- as.numeric(t(w) %*% Sigma %*% w)
    if (port_var < 1e-14) break
    port_vol <- sqrt(port_var)

    # Marginal risk contributions
    mrc <- as.numeric(Sigma %*% w) / port_vol

    # Risk contributions
    rc  <- w * mrc

    # Target: rc_i = budget_i * port_vol
    target_rc <- budget * port_vol

    # Gradient of ||rc - target||^2
    # d/dw_i = 2 * (rc - target) * d(rc_i)/d(w_j)
    # Approximate update:
    w_new <- w * sqrt(target_rc / (rc + 1e-14))
    w_new <- pmax(w_new, 1e-8)
    w_new <- w_new / sum(w_new)

    if (max(abs(w_new - w)) < tol) {
      w <- w_new
      break
    }
    w <- w_new
  }

  port_var <- as.numeric(t(w) %*% Sigma %*% w)
  port_vol <- sqrt(port_var)
  mrc <- as.numeric(Sigma %*% w) / port_vol
  rc  <- w * mrc

  list(
    weights             = w,
    risk_contributions  = rc,
    pct_risk_contrib    = rc / port_vol,
    portfolio_vol       = port_vol
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Black-Litterman Model
# ─────────────────────────────────────────────────────────────────────────────

#' black_litterman
#'
#' Full Black-Litterman portfolio optimization.
#'
#' The BL model:
#'   Posterior mu_BL = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1} *
#'                     [(tau*Sigma)^{-1}*Pi + P'*Omega^{-1}*Q]
#'   Posterior Sigma_BL = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1} + Sigma
#'
#' @param Sigma n×n covariance matrix of asset returns
#' @param w_mkt n-vector of market-cap weights
#' @param P k×n views matrix (k = number of views)
#' @param Q k-vector of views expected returns
#' @param Omega k×k views uncertainty matrix (diagonal or full)
#' @param tau numeric, scaling of prior uncertainty (default 1/T)
#' @param delta numeric, risk aversion implied by market (default 2.5)
#' @param Omega_method character: "proportional", "he_litterman", "idzorek"
#' @return list with mu_BL, Sigma_BL, w_BL, implied_returns, prior_mu
black_litterman <- function(
  Sigma,
  w_mkt,
  P         = NULL,
  Q         = NULL,
  Omega     = NULL,
  tau       = 0.05,
  delta     = 2.5,
  Omega_method = "proportional",
  confidences  = NULL   # Idzorek-style confidence levels per view (0-1)
) {
  n <- nrow(Sigma)
  w_mkt <- w_mkt / sum(w_mkt)

  # Implied equilibrium excess returns: Pi = delta * Sigma * w_mkt
  Pi <- delta * Sigma %*% w_mkt

  if (is.null(P) || is.null(Q)) {
    # No views: return market equilibrium
    w_BL <- mvo_optimize(as.numeric(Pi), Sigma, lambda = delta)
    return(list(
      mu_BL         = as.numeric(Pi),
      Sigma_BL      = Sigma,
      w_BL          = w_BL$weights,
      implied_returns = as.numeric(Pi),
      prior_mu      = as.numeric(Pi)
    ))
  }

  k <- nrow(P)
  stopifnot(length(Q) == k, ncol(P) == n)

  # Build Omega (views uncertainty)
  if (is.null(Omega)) {
    Omega <- .build_omega(P, Sigma, tau, Omega_method, confidences, delta)
  }

  tau_Sigma     <- tau * Sigma
  tau_Sigma_inv <- tryCatch(solve(tau_Sigma), error = function(e) ginv(tau_Sigma))
  Omega_inv     <- tryCatch(solve(Omega),      error = function(e) ginv(Omega))

  # Posterior precision
  M_inv_inv <- tau_Sigma_inv + t(P) %*% Omega_inv %*% P
  M_inv     <- tryCatch(solve(M_inv_inv), error = function(e) ginv(M_inv_inv))

  # Posterior mean
  mu_BL_vec <- M_inv %*% (tau_Sigma_inv %*% Pi + t(P) %*% Omega_inv %*% Q)
  mu_BL     <- as.numeric(mu_BL_vec)

  # Posterior covariance (posterior + estimation uncertainty)
  Sigma_BL <- M_inv + Sigma

  # Optimal weights: w* = (delta * Sigma_BL)^{-1} * mu_BL
  Sigma_BL_reg <- Sigma_BL + 1e-8 * diag(n)
  w_BL_raw <- tryCatch(
    solve(delta * Sigma_BL_reg, mu_BL),
    error = function(e) rep(1/n, n)
  )
  w_BL <- pmax(w_BL_raw, 0)
  if (sum(w_BL) < 1e-10) w_BL <- rep(1/n, n)
  w_BL <- w_BL / sum(w_BL)

  # Also compute unconstrained via MVO
  mvo_sol <- mvo_optimize(mu_BL, Sigma_BL, lambda = delta)

  list(
    mu_BL           = mu_BL,
    Sigma_BL        = Sigma_BL,
    w_BL            = w_BL,
    w_BL_mvo        = mvo_sol$weights,
    implied_returns  = as.numeric(Pi),
    prior_mu        = as.numeric(Pi),
    posterior_precision = M_inv_inv,
    views_P         = P,
    views_Q         = Q,
    Omega           = Omega
  )
}


#' .build_omega
#'
#' Internal: construct Omega (views uncertainty) matrix.
.build_omega <- function(P, Sigma, tau, method, confidences, delta) {
  k <- nrow(P)

  if (method == "proportional") {
    # Omega = diag(tau * P * Sigma * P')
    Omega_diag <- diag(tau * P %*% Sigma %*% t(P))
    return(diag(Omega_diag))
  }

  if (method == "he_litterman") {
    # He-Litterman: Omega = diag(P * (tau*Sigma) * P')
    Omega_diag <- diag(P %*% (tau * Sigma) %*% t(P))
    return(diag(Omega_diag))
  }

  if (method == "idzorek" && !is.null(confidences)) {
    # Idzorek (2005): scale Omega by 1/confidence - 1 factor
    # alpha_i = (1 - c_i) / c_i
    Omega_base <- diag(diag(P %*% (tau * Sigma) %*% t(P)))
    alpha <- (1 - pmin(pmax(confidences, 0.01), 0.99)) /
             pmin(pmax(confidences, 0.01), 0.99)
    return(diag(alpha) %*% Omega_base)
  }

  # Default: proportional
  diag(diag(tau * P %*% Sigma %*% t(P)))
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Hierarchical Risk Parity (HRP)
# ─────────────────────────────────────────────────────────────────────────────

#' hrp_weights
#'
#' Hierarchical Risk Parity (López de Prado 2016).
#' Steps:
#'   1. Compute correlation-based distance and cluster (dendrogram)
#'   2. Quasi-diagonalize covariance matrix
#'   3. Recursive bisection allocation
#'
#' @param Sigma n×n covariance matrix
#' @param labels character vector of asset names
#' @return list with weights, cluster_order, dendrogram
hrp_weights <- function(Sigma, labels = NULL) {
  n <- nrow(Sigma)
  if (is.null(labels)) labels <- paste0("Asset", seq_len(n))

  # Correlation matrix
  D <- diag(sqrt(diag(Sigma)))
  D_inv <- diag(1 / sqrt(diag(Sigma)))
  corr <- D_inv %*% Sigma %*% D_inv
  corr <- (corr + t(corr)) / 2  # ensure symmetry

  # Distance matrix: d_{ij} = sqrt(0.5 * (1 - rho_{ij}))
  dist_mat <- sqrt(pmax(0.5 * (1 - corr), 0))
  diag(dist_mat) <- 0
  dist_obj <- as.dist(dist_mat)

  # Hierarchical clustering (single linkage for HRP)
  hclust_obj <- hclust(dist_obj, method = "single")

  # Get quasi-diagonal order (leaf order of dendrogram)
  leaf_order <- hclust_obj$order

  # Recursive bisection
  weights <- rep(1.0, n)
  names(weights) <- labels
  cluster_list <- list(leaf_order)

  while (length(cluster_list) > 0) {
    new_clusters <- list()
    for (cluster in cluster_list) {
      if (length(cluster) == 1) next
      mid <- length(cluster) %/% 2L
      left  <- cluster[1:mid]
      right <- cluster[(mid+1):length(cluster)]

      # Cluster volatilities
      var_left  <- cluster_variance(Sigma, left)
      var_right <- cluster_variance(Sigma, right)

      # Allocation factor
      alpha_left <- 1 - var_left / (var_left + var_right)

      # Scale weights
      weights[left]  <- weights[left]  * alpha_left
      weights[right] <- weights[right] * (1 - alpha_left)

      if (length(left)  > 1) new_clusters <- c(new_clusters, list(left))
      if (length(right) > 1) new_clusters <- c(new_clusters, list(right))
    }
    cluster_list <- new_clusters
  }

  weights <- weights / sum(weights)

  # Portfolio stats
  exp_vol <- sqrt(as.numeric(t(weights) %*% Sigma %*% weights))

  list(
    weights       = weights,
    cluster_order = leaf_order,
    labels        = labels,
    dendrogram    = as.dendrogram(hclust_obj),
    portfolio_vol = exp_vol
  )
}


#' cluster_variance
#'
#' Compute variance of a cluster using inverse-variance weights within cluster.
cluster_variance <- function(Sigma, indices) {
  Sigma_c <- Sigma[indices, indices, drop = FALSE]
  n_c <- length(indices)
  # Inverse variance weights for the cluster
  ivols <- 1 / sqrt(diag(Sigma_c))
  ivols[!is.finite(ivols)] <- 0
  w_c <- ivols / sum(ivols)
  as.numeric(t(w_c) %*% Sigma_c %*% w_c)
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Robust Optimization
# ─────────────────────────────────────────────────────────────────────────────

#' robust_mvo
#'
#' Robust MVO using MASS::cov.rob for robust covariance estimation
#' and ellipsoidal uncertainty set.
#'
#' @param returns matrix n_obs × n_assets of historical returns
#' @param lambda numeric, risk aversion
#' @param kappa numeric, robustness parameter (uncertainty set size)
#' @param method character: "mcd" (Minimum Covariance Determinant) or "mve"
#' @param lb numeric, lower weight bound
#' @return list with weights, exp_return, exp_vol, robust_mu, robust_Sigma
robust_mvo <- function(
  returns,
  lambda   = 2.0,
  kappa    = 0.5,
  method   = "mcd",
  lb       = 0.0
) {
  R <- as.matrix(returns)
  n <- ncol(R)

  # Robust covariance estimation
  rob_cov <- tryCatch(
    MASS::cov.rob(R, method = method),
    error = function(e) {
      list(center = colMeans(R, na.rm = TRUE), cov = cov(R))
    }
  )

  mu_rob    <- rob_cov$center
  Sigma_rob <- rob_cov$cov
  Sigma_rob <- (Sigma_rob + t(Sigma_rob)) / 2  # ensure symmetry

  # Robust optimization: worst-case over ellipsoidal uncertainty set
  # min_w max_{mu in U} [ lambda * w'Sigma*w - w'mu ]
  # where U = {mu : ||Sigma^{-0.5}(mu - mu_hat)||^2 <= kappa^2}
  # Optimal weights use: mu_robust = mu_hat - kappa * Sigma_robust * w / ||Sigma^{0.5}*w||
  # Solved iteratively

  w <- rep(1/n, n)
  ub <- rep(1.0, n)
  lb_vec <- rep(lb, n)

  for (iter in seq_len(100L)) {
    port_vol_w <- as.numeric(sqrt(t(w) %*% Sigma_rob %*% w))
    if (port_vol_w < 1e-10) port_vol_w <- 1e-10

    # Worst-case mu
    mu_wc <- mu_rob - kappa * as.numeric(Sigma_rob %*% w) / port_vol_w

    # Optimize given mu_wc
    sol <- mvo_optimize(mu_wc, Sigma_rob, lambda = lambda, lb = lb_vec, ub = ub)
    w_new <- sol$weights

    if (max(abs(w_new - w)) < 1e-8) {
      w <- w_new
      break
    }
    w <- w_new
  }

  # Report stats with robust estimates
  stats <- .portfolio_stats(w, mu_rob, Sigma_rob)
  stats$robust_mu    <- mu_rob
  stats$robust_Sigma <- Sigma_rob
  stats$kappa        <- kappa
  stats
}


#' ledoit_wolf_shrinkage
#'
#' Compute Ledoit-Wolf shrinkage covariance estimate.
#' Shrinks sample covariance toward scaled identity.
#'
#' @param returns matrix n_obs × n_assets
#' @param target character: "identity" or "constant_corr"
#' @return list with Sigma_lw (shrinkage cov), optimal_alpha (shrinkage intensity)
ledoit_wolf_shrinkage <- function(returns, target = "identity") {
  R <- as.matrix(returns)
  T <- nrow(R)
  n <- ncol(R)

  S <- cov(R)

  if (target == "identity") {
    # Shrink toward (tr(S)/n) * I
    mu_hat <- sum(diag(S)) / n
    F_target <- mu_hat * diag(n)
  } else {
    # Constant correlation target
    var_diag <- diag(S)
    sd_diag  <- sqrt(var_diag)
    corr_mat <- S / outer(sd_diag, sd_diag)
    avg_rho   <- (sum(corr_mat) - n) / (n * (n - 1))
    F_target  <- avg_rho * outer(sd_diag, sd_diag)
    diag(F_target) <- var_diag
  }

  # Ledoit-Wolf shrinkage intensity via analytical formula
  # Oracle shrinkage: alpha* = argmin E[||alpha*F + (1-alpha)*S - Sigma||^2_F]
  # Closed-form via Ledoit & Wolf (2004)
  delta_sq <- sum((S - F_target)^2)
  # Estimation of rho (Frobenius norm of population cov)
  pi_hat <- 0.0
  R_c <- scale(R, center = TRUE, scale = FALSE)
  for (k in seq_len(T)) {
    r_k <- R_c[k, ]
    pi_hat <- pi_hat + sum((outer(r_k, r_k) - S)^2)
  }
  pi_hat <- pi_hat / T^2

  rho_hat  <- sum(pi_hat)
  alpha_opt <- max(0, min(1, rho_hat / (delta_sq / T)))

  Sigma_lw <- alpha_opt * F_target + (1 - alpha_opt) * S

  list(
    Sigma_lw      = Sigma_lw,
    optimal_alpha = alpha_opt,
    sample_cov    = S,
    target        = F_target
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Portfolio Analytics & Visualization
# ─────────────────────────────────────────────────────────────────────────────

#' compare_portfolios
#'
#' Compare multiple portfolio strategies on a return matrix.
#'
#' @param returns matrix n_obs × n_assets
#' @param ... named weight vectors or functions returning weight vectors
#' @param rf numeric, risk-free rate per period
#' @return data.frame with stats per portfolio
compare_portfolios <- function(returns, ..., rf = 0.0) {
  R <- as.matrix(returns)
  n <- ncol(R)
  T <- nrow(R)

  portfolio_list <- list(...)
  if (length(portfolio_list) == 0) {
    # Default comparison
    mu    <- colMeans(R)
    Sigma <- cov(R)
    portfolio_list <- list(
      EqualWeight  = rep(1/n, n),
      MinVar       = mvo_optimize(mu, Sigma, lambda = 100)$weights,
      MaxSharpe    = mvo_optimize(mu, Sigma, lambda = 2)$weights,
      RiskParity   = risk_parity_weights(Sigma)$weights,
      HRP          = hrp_weights(Sigma)$weights
    )
  }

  results <- lapply(names(portfolio_list), function(nm) {
    w_or_fn <- portfolio_list[[nm]]
    if (is.function(w_or_fn)) {
      w <- w_or_fn(R)
    } else {
      w <- as.numeric(w_or_fn)
    }
    w <- w / sum(w)

    port_ret <- as.numeric(R %*% w) - rf
    mu_p     <- mean(port_ret) * 252
    sig_p    <- sd(port_ret) * sqrt(252)
    sharpe   <- if (sig_p > 0) mu_p / sig_p else NA_real_

    # Max drawdown
    cum_ret <- cumprod(1 + port_ret)
    roll_max <- cummax(cum_ret)
    drawdown <- (cum_ret - roll_max) / roll_max
    max_dd   <- min(drawdown, na.rm = TRUE)

    data.frame(
      portfolio    = nm,
      ann_return   = mu_p,
      ann_vol      = sig_p,
      sharpe_ratio = sharpe,
      max_drawdown = max_dd,
      turnover     = NA_real_,
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, results)
}


#' plot_efficient_frontier_custom
#'
#' Plot efficient frontier with specific portfolio points highlighted.
#'
#' @param mu numeric expected returns
#' @param Sigma covariance matrix
#' @param labels character asset labels
#' @param n_points integer, frontier resolution
#' @return ggplot2 object
plot_efficient_frontier_custom <- function(mu, Sigma, labels = NULL, n_points = 100L,
                                            rf = 0.0, lb = 0.0) {
  frontier <- efficient_frontier(mu, Sigma, n_points = n_points, lb = lb, rf = rf)

  # Individual assets
  asset_vol <- sqrt(diag(Sigma))
  n         <- length(mu)
  asset_df  <- data.frame(
    vol    = asset_vol,
    ret    = mu,
    label  = if (is.null(labels)) paste0("A", seq_len(n)) else labels
  )

  # Tangency portfolio
  excess_ret   <- mu - rf
  Sigma_inv_mu <- tryCatch(solve(Sigma, excess_ret), error = function(e) ginv(Sigma) %*% excess_ret)
  w_tan        <- pmax(Sigma_inv_mu, 0)
  if (sum(w_tan) < 1e-10) w_tan <- rep(1/n, n)
  w_tan <- w_tan / sum(w_tan)
  tan_ret <- as.numeric(t(w_tan) %*% mu)
  tan_vol <- as.numeric(sqrt(t(w_tan) %*% Sigma %*% w_tan))

  ggplot(frontier, aes(x = exp_vol, y = exp_return)) +
    geom_line(color = "#2196F3", linewidth = 1.2) +
    geom_point(data = asset_df, aes(x = vol, y = ret, label = label),
               color = "#e63946", size = 2.5) +
    ggrepel::geom_text_repel(data = asset_df,
                              aes(x = vol, y = ret, label = label),
                              size = 3, na.rm = TRUE) +
    geom_point(aes(x = tan_vol, y = tan_ret), color = "#ff9800", size = 4, shape = 18) +
    annotate("text", x = tan_vol, y = tan_ret * 1.05,
             label = "Tangency", size = 3, color = "#ff9800") +
    labs(
      title = "Efficient Frontier",
      x     = "Annualised Volatility",
      y     = "Annualised Expected Return"
    ) +
    theme_minimal(base_size = 10)
}
