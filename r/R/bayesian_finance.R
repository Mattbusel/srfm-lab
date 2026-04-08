##############################################################################
# bayesian_finance.R -- Bayesian Methods for Finance
# Conjugate priors, MCMC (Gibbs, MH), Bayesian portfolio, Black-Litterman,
# regime switching, VaR, changepoint detection, stochastic volatility,
# model averaging, hierarchical models, diagnostics
##############################################################################

# ---------------------------------------------------------------------------
# Conjugate Priors: Normal-Normal
# ---------------------------------------------------------------------------
normal_normal_update <- function(y, mu_0, sigma2_0, sigma2) {
  n <- length(y)
  y_bar <- mean(y)
  precision_0 <- 1 / sigma2_0
  precision_data <- n / sigma2
  precision_post <- precision_0 + precision_data
  mu_post <- (precision_0 * mu_0 + precision_data * y_bar) / precision_post
  sigma2_post <- 1 / precision_post
  list(mu_post = mu_post, sigma2_post = sigma2_post,
       ci_95 = c(mu_post - 1.96 * sqrt(sigma2_post),
                 mu_post + 1.96 * sqrt(sigma2_post)),
       shrinkage = precision_data / precision_post)
}

# ---------------------------------------------------------------------------
# Conjugate Priors: Normal-Inverse-Gamma
# ---------------------------------------------------------------------------
normal_inverse_gamma_update <- function(y, mu_0, kappa_0, alpha_0, beta_0) {
  n <- length(y)
  y_bar <- mean(y)
  kappa_n <- kappa_0 + n
  mu_n <- (kappa_0 * mu_0 + n * y_bar) / kappa_n
  alpha_n <- alpha_0 + n / 2
  beta_n <- beta_0 + 0.5 * sum((y - y_bar)^2) +
    kappa_0 * n * (y_bar - mu_0)^2 / (2 * kappa_n)
  sigma2_mean <- beta_n / (alpha_n - 1)
  mu_mean <- mu_n
  mu_var <- beta_n / (alpha_n * kappa_n)
  list(mu_n = mu_n, kappa_n = kappa_n, alpha_n = alpha_n, beta_n = beta_n,
       sigma2_mean = sigma2_mean, mu_mean = mu_mean, mu_var = mu_var)
}

# ---------------------------------------------------------------------------
# Conjugate Priors: Beta-Binomial
# ---------------------------------------------------------------------------
beta_binomial_update <- function(successes, trials, alpha_0, beta_0) {
  alpha_n <- alpha_0 + successes
  beta_n <- beta_0 + trials - successes
  mean_post <- alpha_n / (alpha_n + beta_n)
  var_post <- alpha_n * beta_n / ((alpha_n + beta_n)^2 * (alpha_n + beta_n + 1))
  ci_95 <- qbeta(c(0.025, 0.975), alpha_n, beta_n)
  list(alpha_n = alpha_n, beta_n = beta_n, mean = mean_post,
       var = var_post, ci_95 = ci_95)
}

# ---------------------------------------------------------------------------
# Gibbs Sampler: Bayesian Linear Regression
# ---------------------------------------------------------------------------
gibbs_linear_regression <- function(y, X, n_iter = 5000, burn = 1000,
                                     mu_0 = NULL, V_0 = NULL,
                                     alpha_0 = 0.01, beta_0 = 0.01,
                                     seed = 42) {
  set.seed(seed)
  n <- length(y); p <- ncol(X)
  X1 <- cbind(1, X); k <- ncol(X1)
  if (is.null(mu_0)) mu_0 <- rep(0, k)
  if (is.null(V_0)) V_0 <- diag(100, k)
  V_0_inv <- solve(V_0)
  XtX <- crossprod(X1)
  Xty <- crossprod(X1, y)
  beta_samples <- matrix(0, n_iter, k)
  sigma2_samples <- numeric(n_iter)
  beta_cur <- as.vector(solve(XtX) %*% Xty)
  sigma2_cur <- sum((y - X1 %*% beta_cur)^2) / (n - k)
  for (iter in 1:n_iter) {
    V_n_inv <- V_0_inv + XtX / sigma2_cur
    V_n <- solve(V_n_inv)
    mu_n <- V_n %*% (V_0_inv %*% mu_0 + Xty / sigma2_cur)
    L <- t(chol(V_n))
    beta_cur <- as.vector(mu_n + L %*% rnorm(k))
    resid <- y - X1 %*% beta_cur
    alpha_n <- alpha_0 + n / 2
    beta_n <- beta_0 + sum(resid^2) / 2
    sigma2_cur <- 1 / rgamma(1, alpha_n, beta_n)
    beta_samples[iter, ] <- beta_cur
    sigma2_samples[iter] <- sigma2_cur
  }
  post_idx <- (burn + 1):n_iter
  beta_post <- beta_samples[post_idx, , drop = FALSE]
  sigma2_post <- sigma2_samples[post_idx]
  beta_mean <- colMeans(beta_post)
  beta_sd <- apply(beta_post, 2, sd)
  beta_ci <- apply(beta_post, 2, quantile, c(0.025, 0.975))
  sigma2_mean <- mean(sigma2_post)
  names(beta_mean) <- c("(Intercept)", colnames(X))
  list(beta_mean = beta_mean, beta_sd = beta_sd, beta_ci = beta_ci,
       sigma2_mean = sigma2_mean, sigma2_sd = sd(sigma2_post),
       beta_samples = beta_post, sigma2_samples = sigma2_post,
       n_iter = n_iter, burn = burn, method = "Gibbs-LinReg")
}

# ---------------------------------------------------------------------------
# Metropolis-Hastings MCMC (generic)
# ---------------------------------------------------------------------------
metropolis_hastings <- function(log_posterior, init, proposal_sd,
                                 n_iter = 10000, burn = 2000, seed = 42) {
  set.seed(seed)
  p <- length(init)
  samples <- matrix(0, n_iter, p)
  current <- init
  current_lp <- log_posterior(current)
  accept <- 0
  for (iter in 1:n_iter) {
    proposal <- current + rnorm(p) * proposal_sd
    proposal_lp <- log_posterior(proposal)
    log_alpha <- proposal_lp - current_lp
    if (log(runif(1)) < log_alpha) {
      current <- proposal
      current_lp <- proposal_lp
      accept <- accept + 1
    }
    samples[iter, ] <- current
  }
  post_idx <- (burn + 1):n_iter
  post_samples <- samples[post_idx, , drop = FALSE]
  list(samples = post_samples, all_samples = samples,
       acceptance_rate = accept / n_iter,
       mean = colMeans(post_samples),
       sd = apply(post_samples, 2, sd),
       ci_95 = apply(post_samples, 2, quantile, c(0.025, 0.975)),
       method = "MH")
}

# ---------------------------------------------------------------------------
# Adaptive Metropolis-Hastings
# ---------------------------------------------------------------------------
adaptive_mh <- function(log_posterior, init, n_iter = 10000, burn = 2000,
                         adapt_start = 500, seed = 42) {
  set.seed(seed)
  p <- length(init)
  samples <- matrix(0, n_iter, p)
  current <- init
  current_lp <- log_posterior(current)
  accept <- 0
  sd_scale <- 2.38^2 / p
  Sigma <- diag(0.01, p)
  for (iter in 1:n_iter) {
    if (iter > adapt_start && iter > p + 1) {
      past <- samples[1:(iter - 1), , drop = FALSE]
      Sigma <- sd_scale * (cov(past) + 1e-6 * diag(p))
    }
    L <- tryCatch(t(chol(Sigma)), error = function(e) diag(sqrt(diag(Sigma))))
    proposal <- current + as.vector(L %*% rnorm(p))
    proposal_lp <- log_posterior(proposal)
    log_alpha <- proposal_lp - current_lp
    if (log(runif(1)) < log_alpha) {
      current <- proposal; current_lp <- proposal_lp; accept <- accept + 1
    }
    samples[iter, ] <- current
  }
  post_idx <- (burn + 1):n_iter
  list(samples = samples[post_idx, , drop = FALSE],
       acceptance_rate = accept / n_iter,
       mean = colMeans(samples[post_idx, , drop = FALSE]),
       method = "Adaptive-MH")
}

# ---------------------------------------------------------------------------
# Bayesian Portfolio Optimization
# ---------------------------------------------------------------------------
bayesian_portfolio <- function(returns, n_iter = 5000, burn = 1000,
                                risk_aversion = 2, seed = 42) {
  set.seed(seed)
  n <- nrow(returns); p <- ncol(returns)
  mu_0 <- rep(0, p)
  kappa_0 <- 1
  nu_0 <- p + 2
  S_0 <- diag(p) * var(as.vector(returns))
  mu_hat <- colMeans(returns)
  S_hat <- cov(returns)
  kappa_n <- kappa_0 + n
  mu_n <- (kappa_0 * mu_0 + n * mu_hat) / kappa_n
  nu_n <- nu_0 + n
  S_n <- S_0 + (n - 1) * S_hat + kappa_0 * n / kappa_n * tcrossprod(mu_hat - mu_0)
  weight_samples <- matrix(0, n_iter, p)
  mu_samples <- matrix(0, n_iter, p)
  sigma_samples <- array(0, dim = c(p, p, n_iter))
  for (iter in 1:n_iter) {
    Sigma_inv <- rWishart(1, nu_n, solve(S_n))[, , 1]
    Sigma <- solve(Sigma_inv)
    L <- t(chol(Sigma / kappa_n))
    mu <- mu_n + as.vector(L %*% rnorm(p))
    mu_samples[iter, ] <- mu
    sigma_samples[, , iter] <- Sigma
    Sigma_inv_cur <- Sigma_inv
    w <- as.vector(solve(risk_aversion * Sigma) %*% mu)
    w <- w / sum(abs(w))
    weight_samples[iter, ] <- w
  }
  post_idx <- (burn + 1):n_iter
  w_post <- weight_samples[post_idx, , drop = FALSE]
  mu_post <- mu_samples[post_idx, , drop = FALSE]
  w_mean <- colMeans(w_post)
  w_sd <- apply(w_post, 2, sd)
  w_ci <- apply(w_post, 2, quantile, c(0.025, 0.975))
  expected_ret <- mean(w_post %*% t(mu_post[1:nrow(w_post), ]))
  port_vol <- numeric(nrow(w_post))
  for (s in 1:nrow(w_post)) {
    Sig_s <- sigma_samples[, , post_idx[s]]
    port_vol[s] <- sqrt(as.numeric(t(w_post[s, ]) %*% Sig_s %*% w_post[s, ]))
  }
  names(w_mean) <- colnames(returns)
  list(weights_mean = w_mean, weights_sd = w_sd, weights_ci = w_ci,
       expected_return = mean(mu_post %*% w_mean),
       expected_vol = mean(port_vol),
       sharpe = mean(mu_post %*% w_mean) / mean(port_vol) * sqrt(252),
       weight_samples = w_post, mu_posterior = colMeans(mu_post),
       method = "Bayesian-Portfolio")
}

# ---------------------------------------------------------------------------
# Bayesian Black-Litterman
# ---------------------------------------------------------------------------
bayesian_black_litterman <- function(returns, market_weights, views_P,
                                      views_q, views_omega = NULL,
                                      tau = 0.05, risk_aversion = 2.5) {
  Sigma <- cov(returns)
  p <- ncol(returns)
  pi_eq <- risk_aversion * Sigma %*% market_weights
  k <- nrow(views_P)
  if (is.null(views_omega)) {
    views_omega <- diag(diag(views_P %*% (tau * Sigma) %*% t(views_P)))
  }
  tau_Sigma <- tau * Sigma
  tau_Sigma_inv <- solve(tau_Sigma)
  omega_inv <- solve(views_omega)
  post_precision <- tau_Sigma_inv + t(views_P) %*% omega_inv %*% views_P
  post_cov <- solve(post_precision)
  mu_bl <- post_cov %*% (tau_Sigma_inv %*% pi_eq + t(views_P) %*% omega_inv %*% views_q)
  Sigma_bl <- Sigma + post_cov
  w_bl <- as.vector(solve(risk_aversion * Sigma_bl) %*% mu_bl)
  w_bl <- w_bl / sum(abs(w_bl))
  names(w_bl) <- colnames(returns)
  tilt <- w_bl - market_weights
  list(mu_bl = as.vector(mu_bl), sigma_bl = Sigma_bl, weights = w_bl,
       equilibrium_returns = as.vector(pi_eq), tilt = tilt,
       posterior_cov = post_cov, method = "Bayesian-BL")
}

# ---------------------------------------------------------------------------
# Bayesian Regime Switching (Hamilton Filter with Prior)
# ---------------------------------------------------------------------------
bayesian_regime_switching <- function(y, n_regimes = 2, n_iter = 3000,
                                       burn = 500, seed = 42) {
  set.seed(seed)
  n <- length(y)
  K <- n_regimes
  mu <- quantile(y, seq(0.2, 0.8, length.out = K))
  sigma2 <- rep(var(y), K)
  P <- matrix(1 / K, K, K)
  diag(P) <- 0.9
  P <- P / rowSums(P)
  xi <- matrix(1 / K, n, K)
  mu_samples <- matrix(0, n_iter, K)
  sigma2_samples <- matrix(0, n_iter, K)
  P_samples <- array(0, dim = c(K, K, n_iter))
  state_samples <- matrix(0, n_iter, n)
  for (iter in 1:n_iter) {
    # Forward filter
    alpha <- matrix(0, n, K)
    alpha[1, ] <- dnorm(y[1], mu, sqrt(sigma2)) / K
    alpha[1, ] <- alpha[1, ] / sum(alpha[1, ])
    for (t in 2:n) {
      for (j in 1:K) {
        alpha[t, j] <- dnorm(y[t], mu[j], sqrt(sigma2[j])) *
          sum(alpha[t - 1, ] * P[, j])
      }
      alpha[t, ] <- alpha[t, ] / (sum(alpha[t, ]) + 1e-300)
    }
    # Backward sampling
    states <- integer(n)
    states[n] <- sample(1:K, 1, prob = alpha[n, ])
    for (t in (n - 1):1) {
      probs <- alpha[t, ] * P[, states[t + 1]]
      probs <- probs / (sum(probs) + 1e-300)
      states[t] <- sample(1:K, 1, prob = probs)
    }
    state_samples[iter, ] <- states
    # Sample parameters
    for (k in 1:K) {
      idx_k <- which(states == k)
      nk <- length(idx_k)
      if (nk < 2) next
      yk <- y[idx_k]
      # Normal-InverseGamma posterior
      kappa_0 <- 0.01; mu_0 <- mean(y); alpha_0 <- 2; beta_0 <- var(y)
      kappa_n <- kappa_0 + nk
      mu_n <- (kappa_0 * mu_0 + nk * mean(yk)) / kappa_n
      alpha_n <- alpha_0 + nk / 2
      beta_n <- beta_0 + 0.5 * sum((yk - mean(yk))^2) +
        kappa_0 * nk * (mean(yk) - mu_0)^2 / (2 * kappa_n)
      sigma2[k] <- 1 / rgamma(1, alpha_n, beta_n)
      mu[k] <- rnorm(1, mu_n, sqrt(sigma2[k] / kappa_n))
    }
    # Sample transition matrix (Dirichlet)
    for (i in 1:K) {
      counts <- numeric(K)
      for (t in 1:(n - 1)) {
        if (states[t] == i) counts[states[t + 1]] <- counts[states[t + 1]] + 1
      }
      P[i, ] <- rgamma(K, counts + 1)
      P[i, ] <- P[i, ] / sum(P[i, ])
    }
    mu_samples[iter, ] <- mu
    sigma2_samples[iter, ] <- sigma2
    P_samples[, , iter] <- P
  }
  post_idx <- (burn + 1):n_iter
  mu_post <- colMeans(mu_samples[post_idx, , drop = FALSE])
  sigma2_post <- colMeans(sigma2_samples[post_idx, , drop = FALSE])
  P_post <- apply(P_samples[, , post_idx], c(1, 2), mean)
  state_probs <- matrix(0, n, K)
  for (iter in post_idx) {
    for (t in 1:n) state_probs[t, state_samples[iter, t]] <-
      state_probs[t, state_samples[iter, t]] + 1
  }
  state_probs <- state_probs / length(post_idx)
  state_map <- apply(state_probs, 1, which.max)
  list(mu = mu_post, sigma2 = sigma2_post, P = P_post,
       state_probs = state_probs, state_map = state_map,
       mu_samples = mu_samples[post_idx, ], sigma2_samples = sigma2_samples[post_idx, ],
       method = "Bayesian-Regime-Switching")
}

# ---------------------------------------------------------------------------
# Bayesian VaR: Posterior Predictive
# ---------------------------------------------------------------------------
bayesian_var <- function(returns, alpha_levels = c(0.01, 0.05),
                          n_iter = 5000, burn = 1000, seed = 42) {
  set.seed(seed)
  n <- length(returns)
  mu_0 <- 0; kappa_0 <- 0.01; alpha_0 <- 2; beta_0 <- 0.01
  y_bar <- mean(returns); s2 <- var(returns)
  kappa_n <- kappa_0 + n
  mu_n <- (kappa_0 * mu_0 + n * y_bar) / kappa_n
  alpha_n <- alpha_0 + n / 2
  beta_n <- beta_0 + 0.5 * (n - 1) * s2 + kappa_0 * n * (y_bar - mu_0)^2 / (2 * kappa_n)
  predictive_samples <- numeric(n_iter - burn)
  for (iter in 1:(n_iter - burn)) {
    sigma2 <- 1 / rgamma(1, alpha_n, beta_n)
    mu <- rnorm(1, mu_n, sqrt(sigma2 / kappa_n))
    predictive_samples[iter] <- rnorm(1, mu, sqrt(sigma2))
  }
  var_posterior <- list()
  for (a in alpha_levels) {
    q <- quantile(predictive_samples, a)
    es <- mean(predictive_samples[predictive_samples <= q])
    var_posterior[[paste0("alpha_", a)]] <- list(VaR = -q, ES = -es, alpha = a)
  }
  list(var_estimates = var_posterior, predictive_samples = predictive_samples,
       mu_post = mu_n, sigma2_post = beta_n / alpha_n,
       method = "Bayesian-VaR")
}

# ---------------------------------------------------------------------------
# Bayesian Hypothesis Testing: Bayes Factor (Normal means)
# ---------------------------------------------------------------------------
bayes_factor_normal <- function(y, mu_null = 0, sigma2 = NULL,
                                 prior_sd = 1) {
  n <- length(y)
  if (is.null(sigma2)) sigma2 <- var(y)
  y_bar <- mean(y)
  se2 <- sigma2 / n
  bf_01 <- sqrt((se2 + prior_sd^2) / se2) *
    exp(-0.5 * y_bar^2 * prior_sd^2 / (se2 * (se2 + prior_sd^2)))
  bf_10 <- 1 / bf_01
  interpretation <- if (bf_10 > 100) "Decisive for H1"
    else if (bf_10 > 30) "Very strong for H1"
    else if (bf_10 > 10) "Strong for H1"
    else if (bf_10 > 3) "Moderate for H1"
    else if (bf_10 > 1) "Weak for H1"
    else if (1 / bf_10 > 3) "Moderate for H0"
    else "Inconclusive"
  list(bf_10 = bf_10, bf_01 = bf_01, log_bf_10 = log(bf_10),
       interpretation = interpretation, method = "Bayes-Factor")
}

# ---------------------------------------------------------------------------
# Bayes Factor: BIC Approximation
# ---------------------------------------------------------------------------
bic_bayes_factor <- function(loglik_1, loglik_0, k_1, k_0, n) {
  bic_1 <- -2 * loglik_1 + k_1 * log(n)
  bic_0 <- -2 * loglik_0 + k_0 * log(n)
  log_bf_10 <- -0.5 * (bic_1 - bic_0)
  list(log_bf_10 = log_bf_10, bf_10 = exp(log_bf_10),
       bic_1 = bic_1, bic_0 = bic_0, method = "BIC-BF")
}

# ---------------------------------------------------------------------------
# Hierarchical Bayesian Model for Multi-Asset Returns
# ---------------------------------------------------------------------------
hierarchical_returns <- function(returns_list, n_iter = 5000, burn = 1000,
                                  seed = 42) {
  set.seed(seed)
  K <- length(returns_list)
  ni <- sapply(returns_list, length)
  y_bar <- sapply(returns_list, mean)
  s2 <- sapply(returns_list, var)
  mu_hyper <- mean(y_bar); tau2_hyper <- var(y_bar)
  sigma2_pool <- mean(s2)
  mu_k <- y_bar
  mu_samples <- matrix(0, n_iter, K)
  mu_hyper_samples <- numeric(n_iter)
  tau2_samples <- numeric(n_iter)
  sigma2_samples <- matrix(0, n_iter, K)
  for (iter in 1:n_iter) {
    for (k in 1:K) {
      prec_prior <- 1 / tau2_hyper
      prec_data <- ni[k] / sigma2_pool
      prec_post <- prec_prior + prec_data
      mu_post <- (prec_prior * mu_hyper + prec_data * y_bar[k]) / prec_post
      mu_k[k] <- rnorm(1, mu_post, sqrt(1 / prec_post))
    }
    mu_hyper <- rnorm(1, mean(mu_k), sqrt(tau2_hyper / K))
    alpha_tau <- 1 + (K - 1) / 2
    beta_tau <- 0.01 + 0.5 * sum((mu_k - mu_hyper)^2)
    tau2_hyper <- 1 / rgamma(1, alpha_tau, beta_tau)
    for (k in 1:K) {
      yk <- returns_list[[k]]
      alpha_s <- 2 + ni[k] / 2
      beta_s <- 0.01 + 0.5 * sum((yk - mu_k[k])^2)
      s2_k <- 1 / rgamma(1, alpha_s, beta_s)
      sigma2_samples[iter, k] <- s2_k
    }
    sigma2_pool <- mean(sigma2_samples[iter, ])
    mu_samples[iter, ] <- mu_k
    mu_hyper_samples[iter] <- mu_hyper
    tau2_samples[iter] <- tau2_hyper
  }
  post_idx <- (burn + 1):n_iter
  shrinkage <- numeric(K)
  for (k in 1:K) {
    pooled_mean <- mean(mu_hyper_samples[post_idx])
    shrinkage[k] <- 1 - var(mu_samples[post_idx, k]) /
      max(var(y_bar), 1e-10)
  }
  list(mu_posterior = colMeans(mu_samples[post_idx, ]),
       mu_sd = apply(mu_samples[post_idx, ], 2, sd),
       mu_hyper_posterior = mean(mu_hyper_samples[post_idx]),
       tau2_posterior = mean(tau2_samples[post_idx]),
       shrinkage = shrinkage,
       mu_samples = mu_samples[post_idx, ],
       method = "Hierarchical-Returns")
}

# ---------------------------------------------------------------------------
# Bayesian Changepoint Detection: Product Partition Model
# ---------------------------------------------------------------------------
bayesian_changepoint <- function(y, lambda = 100, mu_0 = 0, kappa_0 = 0.01,
                                  alpha_0 = 2, beta_0 = 0.01) {
  n <- length(y)
  log_prior_seg <- function(len) {
    dpois(len - 1, lambda, log = TRUE)
  }
  log_marginal_seg <- function(ys) {
    ns <- length(ys)
    if (ns == 0) return(0)
    kappa_n <- kappa_0 + ns
    mu_n <- (kappa_0 * mu_0 + ns * mean(ys)) / kappa_n
    alpha_n <- alpha_0 + ns / 2
    beta_n <- beta_0 + 0.5 * sum((ys - mean(ys))^2) +
      kappa_0 * ns * (mean(ys) - mu_0)^2 / (2 * kappa_n)
    lgamma(alpha_n) - lgamma(alpha_0) +
      alpha_0 * log(beta_0) - alpha_n * log(beta_n) +
      0.5 * log(kappa_0 / kappa_n) - (ns / 2) * log(2 * pi)
  }
  # Forward recursion
  log_P <- rep(-Inf, n + 1)
  log_P[1] <- 0
  best_cp <- integer(n + 1)
  for (t in 1:n) {
    best_val <- -Inf
    best_s <- 0
    for (s in 0:(t - 1)) {
      val <- log_P[s + 1] + log_marginal_seg(y[(s + 1):t]) +
        log_prior_seg(t - s)
      if (val > best_val) { best_val <- val; best_s <- s }
    }
    log_P[t + 1] <- best_val
    best_cp[t + 1] <- best_s
  }
  # Backtrack
  cps <- c()
  pos <- n
  while (pos > 0) {
    cp <- best_cp[pos + 1]
    if (cp > 0) cps <- c(cp, cps)
    pos <- cp
  }
  # Segment parameters
  segments <- list()
  boundaries <- c(0, cps, n)
  for (i in 1:(length(boundaries) - 1)) {
    idx <- (boundaries[i] + 1):boundaries[i + 1]
    ys <- y[idx]
    post <- normal_inverse_gamma_update(ys, mu_0, kappa_0, alpha_0, beta_0)
    segments[[i]] <- list(start = boundaries[i] + 1, end = boundaries[i + 1],
                          mu = post$mu_mean, sigma2 = post$sigma2_mean,
                          n = length(ys))
  }
  list(changepoints = cps, n_segments = length(segments),
       segments = segments, log_evidence = log_P[n + 1],
       method = "PPM-Changepoint")
}

# ---------------------------------------------------------------------------
# Bayesian Online Changepoint Detection (BOCPD)
# ---------------------------------------------------------------------------
bocpd <- function(y, hazard_rate = 1 / 200, mu_0 = 0, kappa_0 = 0.01,
                   alpha_0 = 0.01, beta_0 = 0.01) {
  n <- length(y)
  R <- matrix(0, n + 1, n + 1)
  R[1, 1] <- 1
  mu_params <- matrix(mu_0, n + 1, n + 1)
  kappa_params <- matrix(kappa_0, n + 1, n + 1)
  alpha_params <- matrix(alpha_0, n + 1, n + 1)
  beta_params <- matrix(beta_0, n + 1, n + 1)
  run_length_probs <- matrix(0, n, n + 1)
  for (t in 1:n) {
    pred_probs <- numeric(t)
    for (rl in 0:(t - 1)) {
      mu_p <- mu_params[rl + 1, t]
      kappa_p <- kappa_params[rl + 1, t]
      alpha_p <- alpha_params[rl + 1, t]
      beta_p <- beta_params[rl + 1, t]
      var_pred <- beta_p * (kappa_p + 1) / (alpha_p * kappa_p)
      pred_probs[rl + 1] <- dt((y[t] - mu_p) / sqrt(var_pred),
                                2 * alpha_p) / sqrt(var_pred)
    }
    growth_probs <- R[1:t, t] * pred_probs * (1 - hazard_rate)
    cp_prob <- sum(R[1:t, t] * pred_probs * hazard_rate)
    R[2:(t + 1), t + 1] <- growth_probs
    R[1, t + 1] <- cp_prob
    evidence <- sum(R[1:(t + 1), t + 1])
    R[1:(t + 1), t + 1] <- R[1:(t + 1), t + 1] / (evidence + 1e-300)
    for (rl in 0:(t - 1)) {
      mu_old <- mu_params[rl + 1, t]
      kappa_old <- kappa_params[rl + 1, t]
      alpha_old <- alpha_params[rl + 1, t]
      beta_old <- beta_params[rl + 1, t]
      kappa_new <- kappa_old + 1
      mu_new <- (kappa_old * mu_old + y[t]) / kappa_new
      alpha_new <- alpha_old + 0.5
      beta_new <- beta_old + kappa_old * (y[t] - mu_old)^2 / (2 * kappa_new)
      mu_params[rl + 2, t + 1] <- mu_new
      kappa_params[rl + 2, t + 1] <- kappa_new
      alpha_params[rl + 2, t + 1] <- alpha_new
      beta_params[rl + 2, t + 1] <- beta_new
    }
    mu_params[1, t + 1] <- mu_0
    kappa_params[1, t + 1] <- kappa_0
    alpha_params[1, t + 1] <- alpha_0
    beta_params[1, t + 1] <- beta_0
    run_length_probs[t, 1:(t + 1)] <- R[1:(t + 1), t + 1]
  }
  map_rl <- apply(run_length_probs, 1, which.max) - 1
  cp_probs <- run_length_probs[, 1]
  detected_cps <- which(cp_probs > 0.5)
  list(run_length_probs = run_length_probs, map_run_length = map_rl,
       cp_probs = cp_probs, detected_changepoints = detected_cps,
       method = "BOCPD")
}

# ---------------------------------------------------------------------------
# MCMC Diagnostics
# ---------------------------------------------------------------------------
mcmc_diagnostics <- function(samples) {
  if (is.null(dim(samples))) samples <- matrix(samples, ncol = 1)
  n <- nrow(samples); p <- ncol(samples)
  ess <- numeric(p)
  for (j in 1:p) {
    x <- samples[, j]
    x_dm <- x - mean(x)
    max_lag <- min(n - 1, 500)
    acf_vals <- numeric(max_lag)
    var_x <- sum(x_dm^2) / n
    for (lag in 1:max_lag) {
      acf_vals[lag] <- sum(x_dm[1:(n - lag)] * x_dm[(lag + 1):n]) / (n * var_x)
      if (lag > 1 && acf_vals[lag] + acf_vals[lag - 1] < 0) break
    }
    tau <- 1 + 2 * sum(acf_vals[1:lag])
    ess[j] <- n / max(tau, 1)
  }
  list(ess = ess, mean = colMeans(samples), sd = apply(samples, 2, sd),
       method = "MCMC-Diagnostics")
}

gelman_rubin <- function(chains) {
  m <- length(chains)
  n <- nrow(chains[[1]])
  p <- ncol(chains[[1]])
  Rhat <- numeric(p)
  for (j in 1:p) {
    chain_means <- sapply(chains, function(ch) mean(ch[, j]))
    chain_vars <- sapply(chains, function(ch) var(ch[, j]))
    B <- n * var(chain_means)
    W <- mean(chain_vars)
    var_hat <- (1 - 1 / n) * W + (1 / n) * B
    Rhat[j] <- sqrt(var_hat / W)
  }
  converged <- all(Rhat < 1.1)
  list(Rhat = Rhat, converged = converged, method = "Gelman-Rubin")
}

# ---------------------------------------------------------------------------
# Bayesian Model Averaging
# ---------------------------------------------------------------------------
bayesian_model_averaging <- function(y, X, models = NULL, n_iter = 3000,
                                      burn = 500, seed = 42) {
  set.seed(seed)
  p <- ncol(X)
  if (is.null(models)) {
    n_models <- min(2^p, 1000)
    if (2^p <= 1000) {
      models <- lapply(0:(2^p - 1), function(m) {
        which(as.logical(intToBits(m)[1:p]))
      })
      models <- models[sapply(models, length) > 0]
    } else {
      models <- lapply(1:n_models, function(i) {
        sort(sample(1:p, sample(1:p, 1)))
      })
      models <- unique(models)
    }
  }
  n_models <- length(models)
  log_marginals <- numeric(n_models)
  n <- length(y)
  for (m in seq_along(models)) {
    vars <- models[[m]]
    k_m <- length(vars)
    Xm <- cbind(1, X[, vars, drop = FALSE])
    XtX <- crossprod(Xm)
    Xty <- crossprod(Xm, y)
    beta_hat <- solve(XtX) %*% Xty
    resid <- y - Xm %*% beta_hat
    ssr <- sum(resid^2)
    log_marginals[m] <- -0.5 * n * log(ssr / n) - 0.5 * k_m * log(n)
  }
  max_lm <- max(log_marginals)
  weights <- exp(log_marginals - max_lm)
  weights <- weights / sum(weights)
  inclusion_prob <- numeric(p)
  for (m in seq_along(models)) {
    for (v in models[[m]]) {
      inclusion_prob[v] <- inclusion_prob[v] + weights[m]
    }
  }
  beta_bma <- rep(0, p)
  for (m in seq_along(models)) {
    vars <- models[[m]]
    Xm <- cbind(1, X[, vars, drop = FALSE])
    beta_hat <- solve(crossprod(Xm)) %*% crossprod(Xm, y)
    beta_bma[vars] <- beta_bma[vars] + weights[m] * beta_hat[-1]
  }
  best_model <- models[[which.max(weights)]]
  names(beta_bma) <- colnames(X)
  names(inclusion_prob) <- colnames(X)
  list(weights = weights, inclusion_prob = inclusion_prob,
       beta_bma = beta_bma, best_model = best_model,
       best_weight = max(weights), n_models = n_models,
       method = "BMA")
}

# ---------------------------------------------------------------------------
# Bayesian Stochastic Volatility Model
# ---------------------------------------------------------------------------
bayesian_sv <- function(y, n_iter = 5000, burn = 1000, seed = 42) {
  set.seed(seed)
  n <- length(y)
  mu <- mean(log(y^2 + 1e-8))
  phi <- 0.95
  sigma_eta <- 0.2
  h <- rep(mu, n)
  mu_samples <- numeric(n_iter)
  phi_samples <- numeric(n_iter)
  sigma_eta_samples <- numeric(n_iter)
  h_samples <- matrix(0, n_iter, n)
  for (iter in 1:n_iter) {
    # Sample h (single-site Gibbs via MH)
    for (t in 1:n) {
      h_prop <- h[t] + rnorm(1, 0, 0.3)
      log_lik_cur <- dnorm(y[t], 0, exp(h[t] / 2), log = TRUE)
      log_lik_prop <- dnorm(y[t], 0, exp(h_prop / 2), log = TRUE)
      if (t == 1) {
        log_prior_cur <- dnorm(h[t], mu + phi * (mu - mu), sigma_eta / sqrt(1 - phi^2), log = TRUE)
        log_prior_prop <- dnorm(h_prop, mu + phi * (mu - mu), sigma_eta / sqrt(1 - phi^2), log = TRUE)
      } else if (t == n) {
        log_prior_cur <- dnorm(h[t], mu + phi * (h[t - 1] - mu), sigma_eta, log = TRUE)
        log_prior_prop <- dnorm(h_prop, mu + phi * (h[t - 1] - mu), sigma_eta, log = TRUE)
      } else {
        mean_t <- (mu * (1 - phi) + phi * (h[t - 1] + h[t + 1]) - phi^2 * mu) / (1 + phi^2)
        var_t <- sigma_eta^2 / (1 + phi^2)
        log_prior_cur <- dnorm(h[t], mean_t, sqrt(var_t), log = TRUE)
        log_prior_prop <- dnorm(h_prop, mean_t, sqrt(var_t), log = TRUE)
      }
      log_alpha <- (log_lik_prop + log_prior_prop) - (log_lik_cur + log_prior_cur)
      if (log(runif(1)) < log_alpha) h[t] <- h_prop
    }
    # Sample mu
    h_dm <- h - phi * c(mu, h[-n])
    mu_var <- 1 / (n * (1 - phi)^2 / sigma_eta^2 + 0.01)
    mu_mean <- mu_var * (sum(h_dm) * (1 - phi) / sigma_eta^2)
    mu <- rnorm(1, mu_mean, sqrt(mu_var))
    # Sample phi (MH)
    phi_prop <- phi + rnorm(1, 0, 0.05)
    if (abs(phi_prop) < 0.999) {
      log_lik_phi_cur <- sum(dnorm(h[-1], mu + phi * (h[-n] - mu), sigma_eta, log = TRUE))
      log_lik_phi_prop <- sum(dnorm(h[-1], mu + phi_prop * (h[-n] - mu), sigma_eta, log = TRUE))
      log_prior_phi <- dbeta((phi_prop + 1) / 2, 20, 1.5, log = TRUE) -
        dbeta((phi + 1) / 2, 20, 1.5, log = TRUE)
      if (log(runif(1)) < log_lik_phi_prop - log_lik_phi_cur + log_prior_phi) {
        phi <- phi_prop
      }
    }
    # Sample sigma_eta
    resid_h <- h[-1] - mu - phi * (h[-n] - mu)
    alpha_s <- 2 + (n - 1) / 2
    beta_s <- 0.01 + 0.5 * sum(resid_h^2)
    sigma_eta <- sqrt(1 / rgamma(1, alpha_s, beta_s))
    mu_samples[iter] <- mu
    phi_samples[iter] <- phi
    sigma_eta_samples[iter] <- sigma_eta
    h_samples[iter, ] <- h
  }
  post_idx <- (burn + 1):n_iter
  h_mean <- colMeans(h_samples[post_idx, ])
  vol_mean <- exp(h_mean / 2)
  list(mu = mean(mu_samples[post_idx]), phi = mean(phi_samples[post_idx]),
       sigma_eta = mean(sigma_eta_samples[post_idx]),
       h_mean = h_mean, volatility = vol_mean,
       h_ci = apply(h_samples[post_idx, ], 2, quantile, c(0.025, 0.975)),
       mu_samples = mu_samples[post_idx],
       phi_samples = phi_samples[post_idx],
       sigma_eta_samples = sigma_eta_samples[post_idx],
       method = "Bayesian-SV")
}

# ---------------------------------------------------------------------------
# Posterior Predictive Distribution
# ---------------------------------------------------------------------------
posterior_predictive <- function(y, X = NULL, n_iter = 3000, burn = 500,
                                  n_ahead = 1, seed = 42) {
  set.seed(seed)
  n <- length(y)
  if (is.null(X)) {
    mu_0 <- 0; kappa_0 <- 0.01; alpha_0 <- 2; beta_0 <- 0.01
    y_bar <- mean(y); s2 <- var(y)
    kappa_n <- kappa_0 + n
    mu_n <- (kappa_0 * mu_0 + n * y_bar) / kappa_n
    alpha_n <- alpha_0 + n / 2
    beta_n <- beta_0 + 0.5 * (n - 1) * s2 + kappa_0 * n * (y_bar - mu_0)^2 / (2 * kappa_n)
    pred <- numeric(n_iter - burn)
    for (i in 1:(n_iter - burn)) {
      sigma2 <- 1 / rgamma(1, alpha_n, beta_n)
      mu <- rnorm(1, mu_n, sqrt(sigma2 / kappa_n))
      pred[i] <- rnorm(1, mu, sqrt(sigma2))
    }
    return(list(predictive = pred, mean = mean(pred), sd = sd(pred),
                ci_95 = quantile(pred, c(0.025, 0.975)),
                method = "Posterior-Predictive"))
  }
  gibbs <- gibbs_linear_regression(y, X, n_iter = n_iter, burn = burn, seed = seed)
  beta_post <- gibbs$beta_samples
  sigma2_post <- gibbs$sigma2_samples
  n_post <- nrow(beta_post)
  X_new <- cbind(1, X[n, , drop = FALSE])
  pred <- numeric(n_post)
  for (i in 1:n_post) {
    pred[i] <- as.vector(X_new %*% beta_post[i, ]) + rnorm(1, 0, sqrt(sigma2_post[i]))
  }
  list(predictive = pred, mean = mean(pred), sd = sd(pred),
       ci_95 = quantile(pred, c(0.025, 0.975)),
       method = "Posterior-Predictive-Reg")
}

# ---------------------------------------------------------------------------
# Bayesian Information Criterion weights
# ---------------------------------------------------------------------------
bic_weights <- function(loglik_vec, k_vec, n) {
  bic_vec <- -2 * loglik_vec + k_vec * log(n)
  delta_bic <- bic_vec - min(bic_vec)
  weights <- exp(-0.5 * delta_bic)
  weights <- weights / sum(weights)
  list(bic = bic_vec, weights = weights, best = which.min(bic_vec))
}

# ---------------------------------------------------------------------------
# Predictive density evaluation (log predictive score)
# ---------------------------------------------------------------------------
log_predictive_score <- function(y_test, predictive_samples_list) {
  n_test <- length(y_test)
  lps <- numeric(n_test)
  for (t in 1:n_test) {
    samp <- predictive_samples_list[[t]]
    bw <- 1.06 * sd(samp) * length(samp)^(-0.2)
    dens <- mean(dnorm(y_test[t], samp, bw))
    lps[t] <- log(max(dens, 1e-300))
  }
  list(lps = lps, mean_lps = mean(lps), sum_lps = sum(lps),
       method = "Log-Predictive-Score")
}

# ---------------------------------------------------------------------------
# Bayesian Sharpe Ratio
# ---------------------------------------------------------------------------
bayesian_sharpe <- function(returns, rf = 0, n_iter = 5000, burn = 1000,
                             seed = 42) {
  set.seed(seed)
  excess <- returns - rf
  n <- length(excess)
  mu_0 <- 0; kappa_0 <- 0.01; alpha_0 <- 2; beta_0 <- 0.01
  y_bar <- mean(excess); s2 <- var(excess)
  kappa_n <- kappa_0 + n
  mu_n <- (kappa_0 * mu_0 + n * y_bar) / kappa_n
  alpha_n <- alpha_0 + n / 2
  beta_n <- beta_0 + 0.5 * (n - 1) * s2 + kappa_0 * n * (y_bar - mu_0)^2 / (2 * kappa_n)
  sharpe_samples <- numeric(n_iter - burn)
  for (i in 1:(n_iter - burn)) {
    sigma2 <- 1 / rgamma(1, alpha_n, beta_n)
    mu <- rnorm(1, mu_n, sqrt(sigma2 / kappa_n))
    sharpe_samples[i] <- mu / sqrt(sigma2) * sqrt(252)
  }
  list(mean = mean(sharpe_samples), sd = sd(sharpe_samples),
       ci_95 = quantile(sharpe_samples, c(0.025, 0.975)),
       prob_positive = mean(sharpe_samples > 0),
       samples = sharpe_samples, method = "Bayesian-Sharpe")
}
