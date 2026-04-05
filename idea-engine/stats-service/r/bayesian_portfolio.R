# =============================================================================
# bayesian_portfolio.R
# Bayesian portfolio methods for crypto/quant trading
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Classical portfolio optimization is notoriously
# sensitive to input estimates (expected returns and covariance). Bayesian
# approaches shrink extreme estimates toward priors, dramatically improving
# out-of-sample performance. Black-Litterman combines market equilibrium
# (prior) with analyst views (likelihood) using Bayes' theorem.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. BLACK-LITTERMAN MODEL
# -----------------------------------------------------------------------------

#' Black-Litterman model implementation
#' Prior: equilibrium excess returns pi = lambda * Sigma * w_mkt
#' Posterior: mu_BL = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1} * [(tau*Sigma)^{-1}*pi + P'*Omega^{-1}*q]
#'
#' @param Sigma covariance matrix of asset returns (N x N)
#' @param w_mkt market cap weights (prior weights)
#' @param P views matrix (K x N): each row is one view
#' @param q view returns vector (K x 1)
#' @param Omega view uncertainty matrix (K x K, diagonal)
#' @param lambda risk aversion coefficient (default 2.5)
#' @param tau scalar controlling weight on prior (default 0.025)
#' @return list with BL expected returns and covariance
black_litterman <- function(Sigma, w_mkt, P = NULL, q = NULL,
                             Omega = NULL, lambda = 2.5, tau = 0.025) {
  N <- nrow(Sigma)
  asset_names <- rownames(Sigma)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  # Implied equilibrium returns (reverse optimization)
  # pi = lambda * Sigma * w_mkt
  pi_eq <- lambda * Sigma %*% w_mkt
  cat("=== Black-Litterman Model ===\n")
  cat("Equilibrium returns (pi):\n")
  for (i in seq_len(N)) {
    cat(sprintf("  %s: %.4f (%.1f%% ann)\n",
                asset_names[i], pi_eq[i], pi_eq[i]*sqrt(252)*100))
  }

  if (is.null(P) || is.null(q)) {
    # No views: return prior (equilibrium)
    cat("\nNo views provided. Returning equilibrium allocation.\n")
    # Optimal weights: w = (lambda*Sigma)^{-1} * pi
    w_opt <- solve(lambda * Sigma) %*% pi_eq
    w_opt <- pmax(w_opt, 0)  # long-only constraint
    w_opt <- w_opt / sum(w_opt)
    return(list(mu_bl=as.vector(pi_eq), Sigma_bl=Sigma*(1+tau),
                w_opt=as.vector(w_opt), pi=as.vector(pi_eq)))
  }

  K <- nrow(P)
  # Default Omega: proportional to P*Sigma*P' (standard BL choice)
  if (is.null(Omega)) {
    Omega <- diag(diag(tau * P %*% Sigma %*% t(P)))
  }

  cat(sprintf("\nViews (%d):\n", K))
  for (k in seq_len(K)) {
    view_desc <- paste0(round(P[k,],2), "*", asset_names, collapse=" + ")
    cat(sprintf("  View %d: %s = %.4f (uncertainty=%.4f)\n",
                k, view_desc, q[k], sqrt(Omega[k,k])))
  }

  # BL posterior mean
  # tau_Sigma_inv = (tau*Sigma)^{-1}
  tau_Sigma <- tau * Sigma
  tau_Sigma_inv <- solve(tau_Sigma)
  Omega_inv <- solve(Omega)

  M_mat <- tau_Sigma_inv + t(P) %*% Omega_inv %*% P
  rhs   <- tau_Sigma_inv %*% pi_eq + t(P) %*% Omega_inv %*% q
  mu_bl <- solve(M_mat, rhs)

  # BL posterior covariance
  Sigma_bl <- Sigma + solve(M_mat)

  # Optimal BL weights (unconstrained mean-variance)
  w_bl <- solve(lambda * Sigma_bl) %*% mu_bl
  # Scale to sum to 1
  w_bl_norm <- w_bl / sum(abs(w_bl))

  cat("\nBL Posterior Returns:\n")
  for (i in seq_len(N)) {
    cat(sprintf("  %s: prior=%.4f, posterior=%.4f\n",
                asset_names[i], pi_eq[i], mu_bl[i]))
  }

  cat("\nBL Optimal Weights:\n")
  for (i in seq_len(N)) {
    cat(sprintf("  %s: mkt=%.3f, BL=%.3f\n",
                asset_names[i], w_mkt[i], w_bl_norm[i]))
  }

  invisible(list(mu_bl=as.vector(mu_bl), Sigma_bl=Sigma_bl,
                 w_opt=as.vector(w_bl_norm), pi=as.vector(pi_eq),
                 tau_Sigma_inv=tau_Sigma_inv))
}

# -----------------------------------------------------------------------------
# 2. BAYESIAN COVARIANCE SHRINKAGE (LEDOIT-WOLF VIA EM)
# -----------------------------------------------------------------------------

#' Ledoit-Wolf analytical shrinkage estimator
#' Sigma_lw = (1-rho)*S + rho*mu_target*I
#' where rho is the optimal shrinkage intensity and mu_target = trace(S)/N
#' This estimator has the best Frobenius-norm MSE among linear combinations
#' @param X data matrix (T x N)
ledoit_wolf <- function(X) {
  n <- nrow(X)
  p <- ncol(X)

  # Sample covariance
  S <- cov(X)

  # Target: scaled identity
  mu_target <- sum(diag(S)) / p
  F_target <- mu_target * diag(p)

  # Ledoit-Wolf (2004) oracle-approximating shrinkage coefficient
  # delta^2 = ||S - F||^2 = sum_{i,j} (s_ij - f_ij)^2
  delta2 <- sum((S - F_target)^2)

  # Estimate beta^2 (sum of asymptotic variances of s_ij)
  # Using the analytical formula from Ledoit-Wolf (2004)
  X_c <- scale(X, center=TRUE, scale=FALSE)
  sum_sq <- 0
  for (k in seq_len(n)) {
    xi <- X_c[k, , drop=FALSE]
    sum_sq <- sum_sq + sum((t(xi) %*% xi - S)^2)
  }
  beta2_bar <- sum_sq / n^2
  beta2 <- min(beta2_bar, delta2)

  # Shrinkage intensity
  rho_hat <- beta2 / delta2
  rho_hat <- max(0, min(1, rho_hat))

  # Shrunk estimator
  Sigma_lw <- (1 - rho_hat) * S + rho_hat * F_target

  cat(sprintf("=== Ledoit-Wolf Shrinkage ===\n"))
  cat(sprintf("Shrinkage intensity rho = %.4f\n", rho_hat))
  cat(sprintf("Target (scaled identity): mu = %.6f\n", mu_target))

  invisible(list(Sigma=Sigma_lw, rho=rho_hat, S=S,
                 target=F_target, mu_target=mu_target))
}

#' Oracle approximating shrinkage toward constant correlation target
#' More informative target than scaled identity for financial data
ledoit_wolf_constant_corr <- function(X) {
  n <- nrow(X); p <- ncol(X)
  S <- cov(X)
  std_devs <- sqrt(diag(S))
  R <- cov2cor(S)  # correlation matrix

  # Constant correlation target
  r_bar <- (sum(R) - p) / (p * (p-1))  # mean of off-diagonal correlations
  F_target <- diag(std_devs) %*%
              (r_bar * matrix(1,p,p) + (1-r_bar) * diag(p)) %*%
              diag(std_devs)

  delta2 <- sum((S - F_target)^2)
  X_c <- scale(X, center=TRUE, scale=FALSE)
  sum_sq <- 0
  for (k in seq_len(n)) {
    xi <- X_c[k, , drop=FALSE]
    sum_sq <- sum_sq + sum((t(xi) %*% xi - S)^2)
  }
  beta2_bar <- sum_sq / n^2
  rho_hat   <- max(0, min(1, beta2_bar / (delta2 + 1e-20)))

  Sigma_lw  <- (1 - rho_hat) * S + rho_hat * F_target
  cat(sprintf("LW Constant Correlation Shrinkage: rho=%.4f, r_bar=%.4f\n",
              rho_hat, r_bar))
  invisible(list(Sigma=Sigma_lw, rho=rho_hat, r_bar=r_bar))
}

# -----------------------------------------------------------------------------
# 3. PRIOR ELICITATION FROM ANALYST VIEWS
# -----------------------------------------------------------------------------

#' Convert analyst view statements to BL (P, q, Omega) format
#' View types:
#'   "absolute": asset i will return X%
#'   "relative": asset i will outperform asset j by X%
#'   "momentum": past winner (sorted rank) will outperform past loser
#' @param views list of view specifications
#' @param asset_names character vector of asset names
#' @param confidence_levels numeric vector of confidences (0-1) per view
elicit_views <- function(views, asset_names, confidence_levels = NULL) {
  N <- length(asset_names)
  K <- length(views)
  P <- matrix(0, K, N)
  colnames(P) <- asset_names
  q <- numeric(K)
  Omega_diag <- numeric(K)  # view uncertainties

  default_conf <- 0.5

  for (k in seq_len(K)) {
    v <- views[[k]]
    conf <- if (!is.null(confidence_levels)) confidence_levels[k] else default_conf

    if (v$type == "absolute") {
      # Asset i will return q_k
      i_idx <- which(asset_names == v$asset)
      P[k, i_idx] <- 1
      q[k] <- v$return
      Omega_diag[k] <- (v$return * (1-conf) / qnorm(0.975))^2

    } else if (v$type == "relative") {
      # Asset i outperforms asset j by q_k
      i_idx <- which(asset_names == v$long)
      j_idx <- which(asset_names == v$short)
      P[k, i_idx] <- 1; P[k, j_idx] <- -1
      q[k] <- v$spread
      Omega_diag[k] <- (v$spread * (1-conf) / qnorm(0.975))^2

    } else if (v$type == "group") {
      # Group of assets will outperform another group
      for (a in v$long_assets)  P[k, which(asset_names==a)] <-  1/length(v$long_assets)
      for (a in v$short_assets) P[k, which(asset_names==a)] <- -1/length(v$short_assets)
      q[k] <- v$spread
      Omega_diag[k] <- (v$spread * (1-conf) / qnorm(0.975))^2
    }
  }

  Omega <- diag(pmax(Omega_diag, 1e-8))
  cat("=== Elicited Views ===\n")
  cat(sprintf("Views: %d\n", K))
  for (k in seq_len(K)) {
    cat(sprintf("  View %d: q=%.4f, uncertainty=%.4f\n",
                k, q[k], sqrt(Omega[k,k])))
  }

  list(P=P, q=q, Omega=Omega)
}

# -----------------------------------------------------------------------------
# 4. MCMC PORTFOLIO OPTIMIZATION (METROPOLIS-HASTINGS)
# -----------------------------------------------------------------------------

#' Bayesian portfolio optimization via Metropolis-Hastings MCMC
#' Posterior distribution over portfolio weights given:
#'   prior: Dirichlet(alpha_dir) on weights
#'   likelihood: portfolio Sharpe ratio is the "reward"
#' Returns samples from the posterior-like distribution over efficient portfolios
#' @param returns_mat T x N return matrix
#' @param n_iter MCMC iterations
#' @param burn_in burn-in period
#' @param proposal_sd proposal distribution std dev
mcmc_portfolio <- function(returns_mat, n_iter = 5000, burn_in = 1000,
                            proposal_sd = 0.05) {
  N <- ncol(returns_mat)
  mu_hat <- colMeans(returns_mat)
  Sigma_hat <- cov(returns_mat)

  # Portfolio Sharpe ratio as log-posterior (up to constant)
  # Prior: uniform on simplex (Dirichlet(1,...,1))
  log_posterior <- function(w) {
    if (any(w < 0) || abs(sum(w) - 1) > 1e-6) return(-Inf)
    port_ret <- sum(w * mu_hat)
    port_var <- as.numeric(t(w) %*% Sigma_hat %*% w)
    if (port_var <= 0) return(-Inf)
    # Penalized Sharpe: maximize risk-adjusted return
    sharpe <- port_ret / sqrt(port_var)
    # Add Dirichlet prior (log): sum(alpha_i - 1)*log(w_i)
    prior_val <- sum(log(pmax(w, 1e-10)))  # alpha = 2 (mild prior)
    sharpe * sqrt(nrow(returns_mat)) + prior_val
  }

  # Initialize at equal weights
  w_curr <- rep(1/N, N)
  log_p_curr <- log_posterior(w_curr)

  samples <- matrix(0, n_iter, N)
  accepted <- 0

  for (iter in seq_len(n_iter)) {
    # Proposal: perturb weights with simplex projection
    w_prop <- w_curr + rnorm(N, 0, proposal_sd)
    # Project to simplex
    w_prop <- pmax(w_prop, 0)
    if (sum(w_prop) > 0) w_prop <- w_prop / sum(w_prop) else w_prop <- w_curr

    log_p_prop <- log_posterior(w_prop)
    log_ratio  <- log_p_prop - log_p_curr

    if (log(runif(1)) < log_ratio) {
      w_curr <- w_prop
      log_p_curr <- log_p_prop
      accepted <- accepted + 1
    }
    samples[iter, ] <- w_curr
  }

  # Post burn-in
  post_samples <- samples[(burn_in+1):n_iter, ]
  w_mean  <- colMeans(post_samples)
  w_lower <- apply(post_samples, 2, quantile, probs=0.025)
  w_upper <- apply(post_samples, 2, quantile, probs=0.975)

  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))

  cat("=== MCMC Portfolio Optimization ===\n")
  cat(sprintf("Acceptance rate: %.2f%%\n", 100*accepted/n_iter))
  cat("Posterior mean weights:\n")
  for (i in seq_len(N)) {
    cat(sprintf("  %s: %.3f [%.3f, %.3f]\n",
                asset_names[i], w_mean[i], w_lower[i], w_upper[i]))
  }

  invisible(list(samples=post_samples, w_mean=w_mean,
                 w_lower=w_lower, w_upper=w_upper,
                 acceptance_rate=accepted/n_iter))
}

# -----------------------------------------------------------------------------
# 5. BAYESIAN VAR ESTIMATION
# -----------------------------------------------------------------------------

#' Bayesian VaR with posterior uncertainty bands
#' Uses conjugate normal-inverse-gamma prior on return distribution
#' @param returns numeric vector of returns
#' @param alpha VaR confidence level
#' @param prior_mu prior mean on mu
#' @param prior_kappa prior precision multiplier on mu
#' @param prior_nu prior degrees of freedom for sigma^2
#' @param prior_s2 prior scale for sigma^2
bayesian_var <- function(returns, alpha = 0.05,
                          prior_mu = 0, prior_kappa = 1,
                          prior_nu = 3, prior_s2 = NULL) {
  n <- length(returns)
  x_bar <- mean(returns)
  s2    <- var(returns)

  if (is.null(prior_s2)) prior_s2 <- s2  # empirical Bayes prior

  # Normal-InverseGamma conjugate posterior
  kappa_n  <- prior_kappa + n
  mu_n     <- (prior_kappa * prior_mu + n * x_bar) / kappa_n
  nu_n     <- prior_nu + n
  s2_n     <- (prior_nu * prior_s2 + (n-1)*s2 +
               (prior_kappa * n / kappa_n) * (x_bar - prior_mu)^2) / nu_n

  # Predictive distribution: Student-t with nu_n degrees of freedom
  # mean = mu_n, scale = s2_n * (kappa_n + 1) / kappa_n
  pred_scale <- sqrt(s2_n * (kappa_n + 1) / kappa_n)
  pred_nu    <- nu_n

  # Point estimate of VaR
  var_bayes <- mu_n + pred_scale * qt(alpha, df=pred_nu)

  # Posterior predictive interval for VaR via Monte Carlo
  n_mc <- 10000
  sigma2_draws <- nu_n * s2_n / rchisq(n_mc, df=nu_n)  # inverse-chi2
  mu_draws     <- rnorm(n_mc, mu_n, sqrt(sigma2_draws / kappa_n))
  # For each draw, compute VaR
  var_draws <- mu_draws + sqrt(sigma2_draws) * qt(alpha, df=nu_n)
  var_lower <- quantile(var_draws, 0.025)
  var_upper <- quantile(var_draws, 0.975)

  cat("=== Bayesian VaR ===\n")
  cat(sprintf("VaR (%.0f%%): %.5f\n", (1-alpha)*100, var_bayes))
  cat(sprintf("95%% credible interval: [%.5f, %.5f]\n", var_lower, var_upper))
  cat(sprintf("Uncertainty (width): %.5f\n", var_upper - var_lower))
  cat(sprintf("Posterior mu: %.5f, posterior sigma: %.5f\n",
              mu_n, sqrt(s2_n)))

  list(VaR=var_bayes, lower=var_lower, upper=var_upper,
       mu_n=mu_n, s2_n=s2_n, nu_n=nu_n, kappa_n=kappa_n)
}

# -----------------------------------------------------------------------------
# 6. PREDICTIVE DISTRIBUTIONS FOR PORTFOLIO RETURNS
# -----------------------------------------------------------------------------

#' Compute predictive distribution for portfolio returns
#' Accounts for parameter uncertainty by integrating over posterior
#' @param returns_mat T x N return matrix
#' @param weights portfolio weights
#' @param n_pred number of predictive samples
predictive_distribution <- function(returns_mat, weights, n_pred = 10000) {
  n <- nrow(returns_mat)
  N <- ncol(returns_mat)

  # Bayesian update on multivariate normal: Normal-Inverse-Wishart
  # Posterior parameters
  x_bar <- colMeans(returns_mat)
  S_n   <- (n-1) * cov(returns_mat)
  nu_n  <- n - N - 1 + N + 1  # degrees of freedom
  kappa_n <- n

  # Predictive: multivariate t with nu_n - N + 1 degrees of freedom
  # (marginalized over mu and Sigma)
  pred_nu <- max(nu_n - N + 1, 2)
  pred_scale <- S_n / (kappa_n * (pred_nu))  # scale matrix

  # Cholesky of scale matrix
  L <- tryCatch(chol(pred_scale), error=function(e) {
    chol(pred_scale + diag(N)*1e-8)
  })

  # Sample from multivariate t
  pred_returns <- matrix(0, n_pred, N)
  for (s in seq_len(n_pred)) {
    z <- rnorm(N)
    chi2 <- rchisq(1, df=pred_nu)
    pred_returns[s, ] <- x_bar + sqrt(pred_nu / chi2) * (L %*% z)
  }

  port_returns <- pred_returns %*% weights
  var_95 <- quantile(port_returns, 0.05)
  es_95  <- mean(port_returns[port_returns < var_95])

  cat("=== Predictive Portfolio Distribution ===\n")
  cat(sprintf("E[r_port]: %.4f\n", mean(port_returns)))
  cat(sprintf("Std[r_port]: %.4f\n", sd(port_returns)))
  cat(sprintf("Skew: %.3f, Kurt (excess): %.3f\n",
              mean((port_returns-mean(port_returns))^3)/sd(port_returns)^3,
              mean((port_returns-mean(port_returns))^4)/sd(port_returns)^4 - 3))
  cat(sprintf("5%% VaR: %.5f\n", var_95))
  cat(sprintf("5%% ES:  %.5f\n", es_95))

  list(pred_returns=port_returns, VaR=var_95, ES=es_95,
       mean=mean(port_returns), sd=sd(port_returns))
}

# -----------------------------------------------------------------------------
# 7. BAYESIAN MODEL AVERAGING (BMA) ACROSS GARCH SPECIFICATIONS
# -----------------------------------------------------------------------------

#' Bayesian Model Averaging over volatility models
#' Weights models by their marginal likelihood (approximated by BIC)
#' @param model_list list of fitted models (each with loglik, k, and sigma2)
#' @param returns return series used to fit all models
bma_volatility <- function(model_list, returns) {
  n <- length(returns)
  model_names <- names(model_list)
  K <- length(model_list)

  # Compute BIC for each model
  bics <- sapply(model_list, function(m) -2*m$loglik + log(n)*m$k)
  # BIC approximation to marginal likelihood: exp(-BIC/2)
  # Normalize to get posterior model probabilities
  min_bic  <- min(bics)
  log_probs <- -0.5 * (bics - min_bic)
  probs     <- exp(log_probs) / sum(exp(log_probs))

  cat("=== Bayesian Model Averaging ===\n")
  for (k in seq_len(K)) {
    cat(sprintf("  %-20s BIC=%.2f  P(M|data)=%.4f\n",
                model_names[k], bics[k], probs[k]))
  }

  # BMA forecast: weighted average of model-specific variance forecasts
  # Each model contributes its last-period sigma^2 weighted by posterior prob
  sigma2_models <- sapply(model_list, function(m) tail(m$sigma2, 1))
  sigma2_bma <- sum(probs * sigma2_models)

  # Prediction uncertainty: mixture variance
  sigma2_mix <- sum(probs * sigma2_models^2) - sigma2_bma^2  # model uncertainty

  cat(sprintf("\nBMA variance forecast: %.8f (vol=%.5f)\n",
              sigma2_bma, sqrt(sigma2_bma)))
  cat(sprintf("Model uncertainty contribution: %.8f\n", sigma2_mix))
  cat(sprintf("Best model: %s (P=%.4f)\n",
              model_names[which.max(probs)], max(probs)))

  invisible(list(probs=probs, sigma2_bma=sigma2_bma,
                 bics=bics, best_model=model_names[which.max(probs)]))
}

# -----------------------------------------------------------------------------
# 8. SEQUENTIAL BAYESIAN UPDATING
# -----------------------------------------------------------------------------

#' Online Bayesian updating of return distribution parameters
#' As new data arrives, update the posterior without full re-estimation
#' @param prior list with mu_0, kappa_0, nu_0, s2_0 (initial prior)
#' @param new_obs new return observations (can be vector)
sequential_bayes_update <- function(prior, new_obs) {
  mu_0 <- prior$mu; kappa_0 <- prior$kappa
  nu_0 <- prior$nu; s2_0 <- prior$s2

  for (x in new_obs) {
    n_new <- 1; x_bar_new <- x
    # Normal-InverseGamma update (single observation)
    kappa_n <- kappa_0 + 1
    mu_n    <- (kappa_0 * mu_0 + x_bar_new) / kappa_n
    nu_n    <- nu_0 + 1
    s2_n    <- (nu_0 * s2_0 + (kappa_0 / kappa_n) * (x_bar_new - mu_0)^2) / nu_n

    mu_0 <- mu_n; kappa_0 <- kappa_n
    nu_0 <- nu_n; s2_0 <- s2_n
  }

  # Predictive VaR
  pred_scale <- sqrt(s2_n * (kappa_n + 1) / kappa_n)
  var_95 <- mu_n + pred_scale * qt(0.05, df=nu_n)

  list(mu=mu_n, kappa=kappa_n, nu=nu_n, s2=s2_n, VaR_95=var_95)
}

#' Run sequential Bayesian updating over a full return series
#' @param returns time series of returns
#' @param prior_params initial prior
sequential_bayes_analysis <- function(returns, prior_params = NULL) {
  n <- length(returns)
  if (is.null(prior_params)) {
    # Initialize with first 30 obs as empirical prior
    init <- head(returns, 30)
    prior_params <- list(mu=mean(init), kappa=1, nu=length(init)-1, s2=var(init))
  }

  state <- prior_params
  mu_ts  <- numeric(n)
  var_ts <- numeric(n)
  var95_ts <- numeric(n)

  for (t in seq_len(n)) {
    state <- sequential_bayes_update(state, returns[t])
    mu_ts[t]  <- state$mu
    var_ts[t] <- state$s2
    var95_ts[t] <- state$VaR_95
  }

  cat("=== Sequential Bayesian Updating ===\n")
  cat(sprintf("Final posterior: mu=%.5f, sigma=%.5f\n",
              tail(mu_ts,1), sqrt(tail(var_ts,1))))
  cat(sprintf("Final 95%% VaR: %.5f\n", tail(var95_ts,1)))

  invisible(list(mu_ts=mu_ts, var_ts=var_ts, var95_ts=var95_ts))
}

# -----------------------------------------------------------------------------
# 9. FULL BAYESIAN PORTFOLIO ANALYSIS PIPELINE
# -----------------------------------------------------------------------------

#' Complete Bayesian portfolio analysis
#' @param returns_mat T x N return matrix
#' @param w_mkt market cap weights (for BL prior)
#' @param views list of analyst views (optional)
run_bayesian_portfolio <- function(returns_mat, w_mkt = NULL,
                                    views = NULL, lambda = 2.5) {
  N <- ncol(returns_mat)
  T_obs <- nrow(returns_mat)
  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(N))
  if (is.null(w_mkt)) w_mkt <- rep(1/N, N)

  cat("=============================================================\n")
  cat("BAYESIAN PORTFOLIO ANALYSIS\n")
  cat(sprintf("Assets: %s\n", paste(asset_names, collapse=", ")))
  cat(sprintf("Observations: %d\n\n", T_obs))

  # 1. Covariance shrinkage
  cat("--- Covariance Shrinkage ---\n")
  lw_res <- ledoit_wolf(returns_mat)
  Sigma_shrunk <- lw_res$Sigma

  # 2. Black-Litterman
  cat("\n--- Black-Litterman ---\n")
  P_views <- NULL; q_views <- NULL; Omega_views <- NULL
  if (!is.null(views)) {
    view_res <- elicit_views(views, asset_names)
    P_views  <- view_res$P; q_views <- view_res$q; Omega_views <- view_res$Omega
  }
  bl_res <- black_litterman(Sigma_shrunk, w_mkt, P=P_views, q=q_views,
                             Omega=Omega_views, lambda=lambda)

  # 3. MCMC portfolio optimization
  cat("\n--- MCMC Portfolio Optimization ---\n")
  mcmc_res <- mcmc_portfolio(returns_mat, n_iter=2000, burn_in=500)

  # 4. Bayesian VaR for BL portfolio
  cat("\n--- Bayesian VaR (BL Portfolio) ---\n")
  bl_weights <- bl_res$w_opt
  port_rets  <- returns_mat %*% bl_weights
  bayes_var  <- bayesian_var(port_rets)

  # 5. Predictive distribution
  cat("\n--- Predictive Return Distribution ---\n")
  pred_dist <- predictive_distribution(returns_mat, bl_weights)

  # 6. Sequential updating
  cat("\n--- Sequential Bayesian Updating ---\n")
  seq_res <- sequential_bayes_analysis(port_rets)

  cat("\n=== SUMMARY ===\n")
  cat(sprintf("BL Weights: %s\n",
              paste0(asset_names, "=", round(bl_weights, 3), collapse=", ")))
  cat(sprintf("MCMC Weights: %s\n",
              paste0(asset_names, "=", round(mcmc_res$w_mean, 3), collapse=", ")))
  cat(sprintf("Bayesian VaR (95%%): %.5f\n", bayes_var$VaR))

  invisible(list(lw=lw_res, bl=bl_res, mcmc=mcmc_res,
                 bayes_var=bayes_var, pred_dist=pred_dist, seq=seq_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 500; N <- 5
# asset_names <- c("BTC","ETH","BNB","SOL","ADA")
# Sigma_true <- matrix(c(
#   0.0016, 0.0010, 0.0007, 0.0009, 0.0006,
#   0.0010, 0.0012, 0.0006, 0.0008, 0.0005,
#   0.0007, 0.0006, 0.0009, 0.0005, 0.0004,
#   0.0009, 0.0008, 0.0005, 0.0011, 0.0006,
#   0.0006, 0.0005, 0.0004, 0.0006, 0.0008), 5, 5)
# mu_true <- c(0.001, 0.0008, 0.0006, 0.0009, 0.0005)
# L <- chol(Sigma_true)
# returns_mat <- matrix(rnorm(n*N), n, N) %*% L +
#                matrix(rep(mu_true, n), n, N, byrow=TRUE)
# colnames(returns_mat) <- asset_names
# w_mkt <- c(0.45, 0.25, 0.12, 0.10, 0.08)
# # Example views: BTC will return 0.15% daily, ETH outperforms ADA by 0.05%
# views <- list(
#   list(type="absolute", asset="BTC", return=0.0015),
#   list(type="relative", long="ETH", short="ADA", spread=0.0005)
# )
# result <- run_bayesian_portfolio(returns_mat, w_mkt, views)

# =============================================================================
# EXTENDED BAYESIAN PORTFOLIO: Hierarchical Priors, Bayesian Factor Models,
# Posterior Predictive Checks, Dynamic Allocation, Entropy Pooling
# =============================================================================

# -----------------------------------------------------------------------------
# Entropy Pooling: Meucci (2008) method to twist a prior distribution
# to match arbitrary views expressed as moments or quantile constraints
# More general than Black-Litterman (handles non-Gaussian views)
# -----------------------------------------------------------------------------
entropy_pooling <- function(scenarios, prior_probs = NULL,
                              view_fns = NULL, view_targets = NULL,
                              max_iter = 200, tol = 1e-6) {
  # scenarios: S x N matrix of scenario returns (S scenarios, N assets)
  # view_fns: list of functions mapping scenario matrix to view statistic
  # view_targets: target values for each view function
  S <- nrow(scenarios)
  if (is.null(prior_probs)) prior_probs <- rep(1/S, S)

  if (is.null(view_fns)) {
    # No views: return prior
    return(list(posterior_probs = prior_probs, divergence = 0))
  }

  K <- length(view_fns)
  stopifnot(length(view_targets) == K)

  # Iterative proportional fitting / gradient descent on KL divergence
  # Minimize KL(q || p) subject to E_q[f_k] = target_k for all k
  # Lagrangian: L = sum q_s log(q_s/p_s) - sum lambda_k (E_q[f_k] - t_k)

  lambdas <- rep(0, K)

  for (iter in 1:max_iter) {
    # Current posterior: q_s proportional to p_s * exp(sum lambda_k * f_k(s))
    f_vals <- matrix(0, S, K)
    for (k in 1:K) {
      f_vals[, k] <- apply(scenarios, 1, function(row) view_fns[[k]](row))
    }

    log_q <- log(prior_probs) + f_vals %*% lambdas
    log_q <- log_q - max(log_q)  # numerical stability
    q <- exp(log_q); q <- q / sum(q)

    # Gradient: d L / d lambda_k = E_q[f_k] - target_k
    E_f <- t(f_vals) %*% q  # K-vector of expected values under q
    grad <- as.vector(E_f) - view_targets

    # Newton-Raphson step using Hessian approximation
    # H_kl = Cov_q(f_k, f_l)
    f_centered <- sweep(f_vals, 2, E_f)
    H <- t(f_centered) %*% diag(as.vector(q)) %*% f_centered

    H_reg <- H + diag(1e-6, K)
    step <- tryCatch(solve(H_reg, grad), error = function(e) grad * 0.01)
    lambdas <- lambdas - step

    if (max(abs(grad)) < tol) break
  }

  # KL divergence from prior to posterior
  kl_div <- sum(q * (log(q + 1e-10) - log(prior_probs + 1e-10)))

  list(
    posterior_probs = q,
    lambdas = lambdas,
    kl_divergence = kl_div,
    converged = iter < max_iter,
    view_satisfaction = as.vector(t(f_vals) %*% q) - view_targets
  )
}

# -----------------------------------------------------------------------------
# Hierarchical Bayes Prior for Mean Returns
# Shrinks asset-specific means toward a common prior mean
# mu_i ~ N(mu_0, tau^2);  r_i | mu_i ~ N(mu_i, sigma_i^2)
# Posterior: mu_i* = (mu_i_hat/sigma_i^2 + mu_0/tau^2) / (1/sigma_i^2 + 1/tau^2)
# -----------------------------------------------------------------------------
hierarchical_bayes_means <- function(returns_mat, tau_prior = NULL) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)
  mu_hat  <- colMeans(returns_mat)
  sig2    <- apply(returns_mat, 2, var) / n  # standard error^2

  # Common prior mean: grand mean (equal-weight average)
  mu_0 <- mean(mu_hat)

  # Empirical Bayes estimate of tau^2 (between-asset variance)
  # Method of moments: E[mu_hat_i] = mu_0, Var[mu_hat_i] = tau^2 + sig2_i
  if (is.null(tau_prior)) {
    # MOM estimate
    between_var <- var(mu_hat) - mean(sig2)
    tau2 <- max(between_var, 0)
  } else {
    tau2 <- tau_prior^2
  }

  # Posterior means (shrinkage toward mu_0)
  shrinkage_factor <- tau2 / (tau2 + sig2)
  mu_posterior <- shrinkage_factor * mu_hat + (1 - shrinkage_factor) * mu_0

  # Posterior variances
  var_posterior <- 1 / (1/sig2 + 1/(tau2 + 1e-10))

  asset_names <- colnames(returns_mat)
  if (is.null(asset_names)) asset_names <- paste0("A", 1:p)

  data.frame(
    asset = asset_names,
    mle_mean = mu_hat,
    posterior_mean = mu_posterior,
    shrinkage = 1 - shrinkage_factor,
    posterior_se = sqrt(var_posterior),
    tau2 = tau2, mu_0 = mu_0
  )
}

# -----------------------------------------------------------------------------
# Bayesian Factor Model: estimate factor loadings with prior on sparsity
# Regression of returns on factors with Gaussian priors on betas
# Posterior is Gaussian (conjugate), allowing analytical updates
# -----------------------------------------------------------------------------
bayesian_factor_model <- function(returns_mat, factor_mat,
                                    prior_beta_var = 1, prior_alpha_var = 0.01) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)
  k <- ncol(factor_mat)

  if (nrow(factor_mat) != n) stop("factor_mat must have same rows as returns_mat")

  asset_names  <- colnames(returns_mat)
  factor_names <- colnames(factor_mat)
  if (is.null(asset_names))  asset_names  <- paste0("A", 1:p)
  if (is.null(factor_names)) factor_names <- paste0("F", 1:k)

  results <- lapply(1:p, function(i) {
    y <- returns_mat[, i]

    # OLS with prior: beta | sigma2 ~ N(0, prior_var * I)
    # Posterior: beta* = (X'X/sigma2 + I/prior_var)^{-1} X'y/sigma2
    X <- cbind(1, factor_mat)
    ols_fit  <- lm(y ~ factor_mat)
    sigma2   <- var(residuals(ols_fit))

    # Prior precision matrix
    prior_prec <- diag(c(1/prior_alpha_var, rep(1/prior_beta_var, k)))

    # Posterior precision and mean
    XtX <- t(X) %*% X
    post_prec <- XtX / sigma2 + prior_prec
    post_var  <- tryCatch(solve(post_prec), error = function(e) diag(k+1))
    post_mean <- post_var %*% (t(X) %*% y / sigma2)

    # Posterior predictive R^2 adjustment
    r2_ols  <- summary(ols_fit)$r.squared
    y_pred  <- X %*% post_mean
    r2_bayes <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)

    list(
      asset = asset_names[i],
      ols_betas = coef(ols_fit),
      posterior_betas = as.vector(post_mean),
      posterior_se = sqrt(diag(post_var)),
      r2_ols = r2_ols, r2_bayes = r2_bayes,
      residual_vol = sqrt(sigma2) * sqrt(252)
    )
  })

  # Aggregate into tidy form
  beta_mat <- do.call(rbind, lapply(results, function(r) {
    data.frame(asset = r$asset,
               alpha = r$posterior_betas[1],
               t(r$posterior_betas[-1]))
  }))
  colnames(beta_mat)[-1] <- c("alpha", factor_names)

  list(
    asset_results = results,
    posterior_betas = beta_mat,
    avg_r2 = mean(sapply(results, function(r) r$r2_bayes))
  )
}

# -----------------------------------------------------------------------------
# Posterior Predictive Check: simulate from posterior predictive distribution
# Compare simulated return distributions with observed to assess model fit
# If simulated quantiles match observed, model is well-specified
# -----------------------------------------------------------------------------
posterior_predictive_check <- function(returns_mat, n_sim = 1000, n_quantiles = 20) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)

  # Fit Normal-InverseWishart conjugate model
  mu_n   <- colMeans(returns_mat)
  S_mat  <- cov(returns_mat) * (n - 1)
  nu_n   <- n + p + 1  # degrees of freedom
  kappa_n <- n

  # Simulate from posterior predictive (Normal-InverseWishart)
  port_returns <- rowMeans(returns_mat)  # equal-weight portfolio

  q_probs <- seq(0.05, 0.95, length.out = n_quantiles)
  obs_quantiles  <- quantile(port_returns, q_probs)

  # Bootstrap as proxy for posterior predictive
  sim_quantiles <- matrix(NA, n_sim, n_quantiles)
  for (s in 1:n_sim) {
    idx <- sample(n, n, replace = TRUE)
    sim_quantiles[s, ] <- quantile(port_returns[idx], q_probs)
  }

  # 90% credible intervals for each quantile
  ci_lower <- apply(sim_quantiles, 2, quantile, 0.05)
  ci_upper <- apply(sim_quantiles, 2, quantile, 0.95)

  pct_covered <- mean(obs_quantiles >= ci_lower & obs_quantiles <= ci_upper)

  list(
    observed_quantiles = obs_quantiles,
    ci_lower = ci_lower, ci_upper = ci_upper,
    quantile_probs = q_probs,
    pct_in_ci = pct_covered,
    model_fit = ifelse(pct_covered > 0.8, "adequate", "poor"),
    # Kolmogorov-Smirnov style: max deviation
    max_deviation = max(abs(obs_quantiles - colMeans(sim_quantiles)))
  )
}

# -----------------------------------------------------------------------------
# Dynamic Bayesian Allocation: update portfolio weights using rolling Bayesian
# inference. At each step, posterior from t becomes prior for t+1.
# Uses exponential forgetting to weight recent data more heavily
# -----------------------------------------------------------------------------
dynamic_bayes_allocation <- function(returns_mat, forgetting = 0.97,
                                      risk_aversion = 3) {
  n <- nrow(returns_mat); p <- ncol(returns_mat)

  # Initialize with diffuse prior
  mu_post   <- colMeans(returns_mat[1:min(20,n), ])
  cov_post  <- cov(returns_mat[1:min(20,n), ])
  kappa_post <- 20

  weights_history <- matrix(NA, n, p)
  portfolio_returns <- rep(NA, n)

  for (t in 21:n) {
    # Current observation
    r_t <- returns_mat[t, ]

    # Update posterior (Bayesian sequential update with forgetting)
    kappa_new <- forgetting * kappa_post + 1
    mu_new    <- (forgetting * kappa_post * mu_post + r_t) / kappa_new
    cov_new   <- forgetting * cov_post +
                 forgetting * kappa_post / kappa_new *
                 outer(r_t - mu_post, r_t - mu_post)
    mu_post   <- mu_new; cov_post <- cov_new; kappa_post <- kappa_new

    # Mean-variance optimal weights using posterior estimates
    cov_inv <- tryCatch(solve(cov_post + diag(1e-6, p)),
                        error = function(e) diag(1/diag(cov_post)))
    w_raw <- cov_inv %*% mu_post / risk_aversion
    w_raw <- pmax(w_raw, 0)  # long-only
    if (sum(w_raw) > 0) w_raw <- w_raw / sum(w_raw)

    weights_history[t, ] <- w_raw
    portfolio_returns[t] <- if (t < n) sum(w_raw * returns_mat[t+1, ]) else NA
  }

  valid <- !is.na(portfolio_returns)
  r_valid <- portfolio_returns[valid]

  list(
    weights_history = weights_history,
    portfolio_returns = portfolio_returns,
    ann_return = mean(r_valid, na.rm=TRUE) * 252,
    ann_vol    = sd(r_valid, na.rm=TRUE) * sqrt(252),
    sharpe     = mean(r_valid, na.rm=TRUE) / sd(r_valid, na.rm=TRUE) * sqrt(252),
    forgetting_factor = forgetting
  )
}

# Extended Bayesian portfolio examples:
# hier <- hierarchical_bayes_means(returns_mat)
# bf   <- bayesian_factor_model(returns_mat, macro_factors)
# ppc  <- posterior_predictive_check(returns_mat)
# dba  <- dynamic_bayes_allocation(returns_mat, forgetting = 0.97)
