# =============================================================================
# simulation_studies.R
# Monte Carlo simulation framework for trading research
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Simulation lets us test strategies on thousands of
# synthetic paths with known properties. This reveals whether observed
# strategy performance is due to true alpha or lucky path realization.
# Variance reduction techniques cut the number of paths needed by 10-50x.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. GEOMETRIC BROWNIAN MOTION SIMULATION (MULTI-ASSET, CORRELATED)
# -----------------------------------------------------------------------------

#' Simulate multi-asset GBM paths
#' dS_i = mu_i * S_i * dt + sigma_i * S_i * dW_i
#' where E[dW_i * dW_j] = rho_ij * dt
#' @param S0 initial prices (length N)
#' @param mu drift vector (annual)
#' @param sigma vol vector (annual)
#' @param Sigma_corr correlation matrix
#' @param T time horizon (years)
#' @param n_steps number of time steps
#' @param n_paths number of simulated paths
simulate_gbm <- function(S0, mu, sigma, Sigma_corr=NULL, T_horizon=1,
                          n_steps=252, n_paths=1000) {
  N  <- length(S0)
  dt <- T_horizon / n_steps

  if (is.null(Sigma_corr)) Sigma_corr <- diag(N)
  # Covariance matrix of returns
  Sigma_cov <- diag(sigma) %*% Sigma_corr %*% diag(sigma)
  L <- chol(Sigma_cov)  # Cholesky for correlated normals

  # Log-return drift: (mu - 0.5*sigma^2)*dt
  drift <- (mu - 0.5 * sigma^2) * dt

  # Output: array [n_steps+1, N, n_paths]
  S <- array(0, dim=c(n_steps+1, N, n_paths))
  S[1, , ] <- S0

  for (p in seq_len(n_paths)) {
    Z <- matrix(rnorm(n_steps * N), n_steps, N)
    dW <- Z %*% L * sqrt(dt)  # correlated innovations
    for (t in seq_len(n_steps)) {
      S[t+1, , p] <- S[t, , p] * exp(drift + dW[t, ])
    }
  }

  cat(sprintf("=== GBM Simulation: %d paths, %d steps, %d assets ===\n",
              n_paths, n_steps, N))
  for (i in seq_len(N)) {
    final_prices <- S[n_steps+1, i, ]
    cat(sprintf("Asset %d: E[S_T]=%.2f, Std[S_T]=%.2f, 5th pct=%.2f\n",
                i, mean(final_prices), sd(final_prices), quantile(final_prices,0.05)))
  }

  invisible(S)
}

#' Extract terminal return statistics from simulation
gbm_terminal_stats <- function(S_sim) {
  dims <- dim(S_sim)
  n_paths <- dims[3]; N <- dims[2]
  S0  <- S_sim[1, , 1]
  S_T <- S_sim[dims[1], , ]  # N x n_paths

  list(
    mean_return = rowMeans(S_T / S0 - 1),
    std_return  = apply(S_T / S0 - 1, 1, sd),
    var_5pct    = apply(S_T / S0 - 1, 1, quantile, probs=0.05),
    es_5pct     = apply(S_T / S0 - 1, 1, function(r) mean(r[r <= quantile(r,0.05)]))
  )
}

# -----------------------------------------------------------------------------
# 2. HESTON STOCHASTIC VOLATILITY SIMULATION
# -----------------------------------------------------------------------------

#' Simulate Heston (1993) model
#' dS = mu * S * dt + sqrt(V) * S * dW_S
#' dV = kappa*(theta-V)*dt + sigma_v*sqrt(V)*dW_V
#' E[dW_S * dW_V] = rho * dt
#' @param S0 initial price, V0 initial variance
#' @param mu,kappa,theta,sigma_v,rho Heston parameters
simulate_heston <- function(S0=100, V0=0.04, mu=0.05, kappa=2.0, theta=0.04,
                              sigma_v=0.3, rho=-0.7, T_horizon=1, n_steps=252,
                              n_paths=1000, scheme="milstein") {
  dt    <- T_horizon / n_steps
  S_mat <- matrix(S0, n_steps+1, n_paths)
  V_mat <- matrix(V0, n_steps+1, n_paths)

  # Cholesky for correlated Brownians
  rho_mat <- matrix(c(1, rho, rho, 1), 2, 2)
  L       <- chol(rho_mat)

  for (t in seq_len(n_steps)) {
    Z  <- matrix(rnorm(2 * n_paths), 2, n_paths)
    dW <- L %*% Z * sqrt(dt)  # 2 x n_paths

    V_curr <- pmax(V_mat[t, ], 0)  # Ensure non-negative

    if (scheme == "milstein") {
      # Milstein scheme for variance (better for CIR-type processes)
      dV <- kappa*(theta - V_curr)*dt + sigma_v*sqrt(V_curr)*dW[2,] +
            0.25*sigma_v^2*(dW[2,]^2 - dt)
    } else {
      # Euler-Maruyama
      dV <- kappa*(theta - V_curr)*dt + sigma_v*sqrt(V_curr)*dW[2,]
    }

    V_mat[t+1, ] <- pmax(V_curr + dV, 0)  # Full truncation for stability
    S_mat[t+1, ] <- S_mat[t, ] * exp((mu - 0.5*V_curr)*dt + sqrt(V_curr)*dW[1,])
  }

  cat(sprintf("=== Heston Simulation: %d paths ===\n", n_paths))
  S_T <- S_mat[n_steps+1, ]
  cat(sprintf("Terminal price: mean=%.2f, std=%.2f\n", mean(S_T), sd(S_T)))
  cat(sprintf("Terminal vol:   mean=%.4f, std=%.4f\n",
              mean(sqrt(V_mat[n_steps+1,])), sd(sqrt(V_mat[n_steps+1,]))))

  invisible(list(S=S_mat, V=V_mat, dt=dt, params=list(
    S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho)))
}

# -----------------------------------------------------------------------------
# 3. JUMP-DIFFUSION SIMULATION (MERTON)
# -----------------------------------------------------------------------------

#' Simulate Merton jump-diffusion
#' dS = (mu - lambda*k_bar)*S*dt + sigma*S*dW + S*dJ
#' where J is a compound Poisson process with log-normal jumps
simulate_jump_diffusion <- function(S0=100, mu=0.05, sigma=0.2, r=0.0,
                                     lambda=5, mu_j=-0.05, sigma_j=0.10,
                                     T_horizon=1, n_steps=252, n_paths=1000) {
  dt     <- T_horizon / n_steps
  k_bar  <- exp(mu_j + sigma_j^2/2) - 1
  drift_adj <- mu - lambda*k_bar  # risk-neutral drift adjustment

  S_mat <- matrix(S0, n_steps+1, n_paths)

  for (t in seq_len(n_steps)) {
    dW     <- rnorm(n_paths) * sqrt(dt)
    # Number of jumps in [t, t+dt]
    n_jumps <- rpois(n_paths, lambda * dt)
    # Jump magnitude
    log_jump <- sapply(n_jumps, function(nj) {
      if (nj == 0) return(0)
      sum(rnorm(nj, mu_j, sigma_j))
    })
    S_mat[t+1, ] <- S_mat[t, ] * exp(
      (drift_adj - 0.5*sigma^2)*dt + sigma*dW + log_jump
    )
  }

  cat(sprintf("=== Jump-Diffusion: lambda=%.1f, mu_j=%.2f, sigma_j=%.2f ===\n",
              lambda, mu_j, sigma_j))
  S_T <- S_mat[n_steps+1, ]
  cat(sprintf("Terminal price: mean=%.2f, 5th pct=%.2f\n",
              mean(S_T), quantile(S_T, 0.05)))
  invisible(S_mat)
}

# -----------------------------------------------------------------------------
# 4. REGIME-SWITCHING SIMULATION
# -----------------------------------------------------------------------------

#' Simulate 2-state regime-switching model
#' State 1: bull market (high return, low vol)
#' State 2: bear market (low/neg return, high vol)
simulate_regime_switching <- function(mu=c(0.0005, -0.001),
                                       sigma=c(0.01, 0.03),
                                       P=matrix(c(0.98, 0.02, 0.05, 0.95), 2, 2),
                                       T_steps=1000, n_paths=500) {
  # Stationary distribution
  pi_stat <- c(P[2,1], P[1,2]) / sum(c(P[2,1], P[1,2]))

  all_returns <- matrix(0, T_steps, n_paths)
  all_regimes <- matrix(0, T_steps, n_paths)

  for (p in seq_len(n_paths)) {
    state <- sample(1:2, 1, prob=pi_stat)
    for (t in seq_len(T_steps)) {
      all_regimes[t, p] <- state
      all_returns[t, p] <- rnorm(1, mu[state], sigma[state])
      # Transition
      state <- sample(1:2, 1, prob=P[state, ])
    }
  }

  # Statistics
  cat(sprintf("=== Regime-Switching Simulation ===\n"))
  cat(sprintf("% time in state 1 (bull): %.1f%%\n",
              100*mean(all_regimes==1)))
  cat(sprintf("% time in state 2 (bear): %.1f%%\n",
              100*mean(all_regimes==2)))
  cat(sprintf("Overall mean return: %.5f (ann: %.3f%%)\n",
              mean(all_returns), 100*mean(all_returns)*252))
  cat(sprintf("Overall vol: %.5f (ann: %.2f%%)\n",
              sd(all_returns[,1]), 100*sd(all_returns[,1])*sqrt(252)))

  invisible(list(returns=all_returns, regimes=all_regimes,
                 params=list(mu=mu, sigma=sigma, P=P)))
}

# -----------------------------------------------------------------------------
# 5. BOOTSTRAP SIMULATION OF TRADING STRATEGY
# -----------------------------------------------------------------------------

#' Block bootstrap simulation of strategy returns
#' Preserves autocorrelation structure in returns (important for risk models)
#' @param returns observed strategy return series
#' @param block_size block length for bootstrap
#' @param n_sim number of simulated paths
#' @param n_steps length of each simulated path
block_bootstrap_strategy <- function(returns, block_size=10, n_sim=1000,
                                      n_steps=NULL) {
  n <- length(returns)
  if (is.null(n_steps)) n_steps <- n

  sim_paths <- matrix(0, n_steps, n_sim)
  for (s in seq_len(n_sim)) {
    path <- numeric(n_steps)
    t <- 1
    while (t <= n_steps) {
      # Random starting block
      block_start <- sample(seq_len(n - block_size + 1), 1)
      block       <- returns[block_start:(block_start + block_size - 1)]
      len         <- min(block_size, n_steps - t + 1)
      path[t:(t+len-1)] <- block[1:len]
      t <- t + len
    }
    sim_paths[, s] <- path
  }

  # Statistics of simulated strategy
  sim_sharpes <- apply(sim_paths, 2, function(r) mean(r)/(sd(r)+1e-10)*sqrt(252))
  sim_maxdd   <- apply(sim_paths, 2, function(r) {
    cum <- cumprod(1+r); abs(min(cum/cummax(cum)-1))
  })
  sim_ret     <- apply(sim_paths, 2, function(r) prod(1+r)-1)

  obs_sharpe <- mean(returns) / (sd(returns)+1e-10) * sqrt(252)

  cat("=== Block Bootstrap Strategy Simulation ===\n")
  cat(sprintf("Observed Sharpe: %.3f\n", obs_sharpe))
  cat(sprintf("Bootstrap Sharpe: mean=%.3f, 5th pct=%.3f, 95th pct=%.3f\n",
              mean(sim_sharpes), quantile(sim_sharpes,0.05), quantile(sim_sharpes,0.95)))
  cat(sprintf("Bootstrap Max DD: mean=%.2f%%, 95th pct=%.2f%%\n",
              100*mean(sim_maxdd), 100*quantile(sim_maxdd,0.95)))

  invisible(list(paths=sim_paths, sharpes=sim_sharpes,
                 max_dd=sim_maxdd, returns=sim_ret))
}

# -----------------------------------------------------------------------------
# 6. VARIANCE REDUCTION TECHNIQUES
# -----------------------------------------------------------------------------

#' Antithetic variates: pair each path with its mirror (U, 1-U)
#' Reduces variance by ~50% for monotone payoffs (e.g., calls)
mc_antithetic <- function(f_payoff, n_sim=10000) {
  n_half <- n_sim %/% 2
  Z <- rnorm(n_half)
  # Evaluate payoff for Z and -Z (antithetic pair)
  payoff_pos <- f_payoff(Z)
  payoff_neg <- f_payoff(-Z)
  # Average of paired estimates
  payoff_paired <- (payoff_pos + payoff_neg) / 2
  mean_est <- mean(payoff_paired)
  se_est   <- sd(payoff_paired) / sqrt(n_half)

  # Compare to crude MC
  Z_crude  <- rnorm(n_sim)
  crude_est <- mean(f_payoff(Z_crude))
  crude_se  <- sd(f_payoff(Z_crude)) / sqrt(n_sim)

  cat("=== Antithetic Variates ===\n")
  cat(sprintf("Antithetic: est=%.5f, se=%.6f\n", mean_est, se_est))
  cat(sprintf("Crude MC:   est=%.5f, se=%.6f\n", crude_est, crude_se))
  cat(sprintf("Variance reduction: %.1f%%\n", 100*(1 - se_est^2/crude_se^2)))

  list(estimate=mean_est, se=se_est, crude=crude_est, crude_se=crude_se)
}

#' Control variate: use a known-mean quantity to reduce variance
#' If Y has known mean mu_Y, use X* = X - b*(Y - mu_Y) as improved estimator
#' where b = Cov(X,Y)/Var(Y) is the optimal control variate coefficient
mc_control_variate <- function(f_payoff, f_control, mu_control, n_sim=10000) {
  Z <- rnorm(n_sim)
  X <- f_payoff(Z)
  Y <- f_control(Z)

  # Optimal control variate coefficient
  b_opt <- cov(X, Y) / var(Y)
  X_cv  <- X - b_opt * (Y - mu_control)

  mean_cv <- mean(X_cv)
  se_cv   <- sd(X_cv) / sqrt(n_sim)
  mean_crude <- mean(X)
  se_crude   <- sd(X) / sqrt(n_sim)

  var_reduction <- 1 - var(X_cv) / var(X)

  cat("=== Control Variate ===\n")
  cat(sprintf("Control variate coeff b: %.4f\n", b_opt))
  cat(sprintf("Crude MC:   est=%.5f, se=%.6f\n", mean_crude, se_crude))
  cat(sprintf("Control CV: est=%.5f, se=%.6f\n", mean_cv, se_cv))
  cat(sprintf("Variance reduction: %.1f%%\n", 100*var_reduction))

  list(estimate=mean_cv, se=se_cv, crude=mean_crude, crude_se=se_crude,
       b_opt=b_opt, var_reduction=var_reduction)
}

#' Importance sampling for rare events (tail probabilities)
#' Shift the sampling distribution to put more weight on the tail region
#' @param f_event indicator function for the event of interest
#' @param mu_shift how much to shift the mean (in standard deviations)
mc_importance_sampling <- function(f_event, mu_shift=-2, n_sim=10000) {
  Z <- rnorm(n_sim, mean=mu_shift, sd=1)
  indicators <- f_event(Z)
  # Likelihood ratio weights
  lr <- dnorm(Z, 0, 1) / dnorm(Z, mu_shift, 1)
  IS_estimate <- mean(indicators * lr)
  se_IS <- sd(indicators * lr) / sqrt(n_sim)

  # Crude estimate
  Z_crude <- rnorm(n_sim)
  crude_est <- mean(f_event(Z_crude))
  se_crude  <- sd(f_event(Z_crude)) / sqrt(n_sim)

  cat("=== Importance Sampling ===\n")
  cat(sprintf("Importance: est=%.6f, se=%.8f\n", IS_estimate, se_IS))
  cat(sprintf("Crude MC:   est=%.6f, se=%.8f\n", crude_est, se_crude))
  cat(sprintf("Variance reduction: %.1f%%\n", 100*(1-se_IS^2/se_crude^2)))

  list(IS=IS_estimate, se_IS=se_IS, crude=crude_est, se_crude=se_crude)
}

# -----------------------------------------------------------------------------
# 7. PARALLEL MONTE CARLO (USING LAPPLY AS ANALOGUE FOR MCLAPPLY)
# -----------------------------------------------------------------------------

#' Parallel-style Monte Carlo using chunked lapply
#' On Windows mclapply uses single core; we split into chunks for demonstration
#' @param f_mc function that generates one chunk of n_per_chunk simulations
#' @param n_total total simulations
#' @param n_chunks number of parallel chunks
parallel_mc <- function(f_mc, n_total=100000, n_chunks=4) {
  n_per <- ceiling(n_total / n_chunks)
  seeds  <- sample.int(1e6, n_chunks)

  results <- lapply(seq_len(n_chunks), function(chunk) {
    set.seed(seeds[chunk])
    f_mc(n_per)
  })

  # Combine results
  all_results <- unlist(results)
  list(
    mean   = mean(all_results),
    se     = sd(all_results) / sqrt(length(all_results)),
    n_total = length(all_results)
  )
}

# -----------------------------------------------------------------------------
# 8. POWER ANALYSIS FOR STRATEGY SIGNIFICANCE TESTS
# -----------------------------------------------------------------------------

#' Power analysis: how many observations needed to detect an alpha of mu_alpha?
#' Power = P(reject H0 | alpha > 0) = P(t > t_{alpha/2} | non-central t)
#' @param mu_alpha true mean return (daily)
#' @param sigma_ret return volatility (daily)
#' @param power_target desired statistical power (default 0.80)
#' @param alpha_level significance level (default 0.05)
power_analysis_returns <- function(mu_alpha, sigma_ret, power_target=0.80,
                                    alpha_level=0.05) {
  # Required sample size for one-sample t-test
  # n >= ((z_{1-alpha/2} + z_{1-beta}) / delta)^2
  # where delta = mu / sigma (standardized effect)
  z_alpha <- qnorm(1 - alpha_level/2)
  z_power <- qnorm(power_target)
  delta   <- abs(mu_alpha) / sigma_ret
  n_required <- ceiling(((z_alpha + z_power) / delta)^2)

  # Simulation-based power check
  n_sim <- 1000
  rejections <- numeric(n_sim)
  for (s in seq_len(n_sim)) {
    r_sim <- rnorm(n_required, mu_alpha, sigma_ret)
    t_stat <- mean(r_sim) / (sd(r_sim)/sqrt(n_required))
    rejections[s] <- abs(t_stat) > z_alpha
  }
  empirical_power <- mean(rejections)

  cat("=== Power Analysis ===\n")
  cat(sprintf("True alpha (daily): %.5f (%.1f%% ann)\n",
              mu_alpha, 100*mu_alpha*252))
  cat(sprintf("Return vol (daily): %.4f (%.1f%% ann)\n",
              sigma_ret, 100*sigma_ret*sqrt(252)))
  cat(sprintf("Standardized effect: %.4f\n", delta))
  cat(sprintf("Required sample size: %d days (%.1f years)\n",
              n_required, n_required/252))
  cat(sprintf("Empirical power (simulation): %.3f\n", empirical_power))

  # Power as function of n
  n_grid  <- seq(50, n_required*3, length.out=20)
  power_n <- sapply(n_grid, function(n) {
    ncp <- delta * sqrt(n)
    pt(qt(alpha_level/2, df=n-1), df=n-1, ncp=ncp) +
    pt(qt(1-alpha_level/2, df=n-1), df=n-1, ncp=ncp, lower.tail=FALSE)
  })

  invisible(list(n_required=n_required, delta=delta,
                 empirical_power=empirical_power,
                 power_curve=data.frame(n=n_grid, power=power_n)))
}

# -----------------------------------------------------------------------------
# 9. FULL SIMULATION STUDY PIPELINE
# -----------------------------------------------------------------------------

#' Run comprehensive simulation study for a given strategy
#' @param returns observed strategy returns
#' @param n_bootstrap number of bootstrap paths
run_simulation_study <- function(returns, asset_name="Strategy",
                                  n_bootstrap=1000) {
  n <- length(returns)
  cat("=============================================================\n")
  cat(sprintf("SIMULATION STUDY: %s\n", asset_name))
  cat(sprintf("Observed: n=%d, mean=%.5f, vol=%.5f\n",
              n, mean(returns), sd(returns)))
  cat("=============================================================\n\n")

  # 1. GBM simulation comparison
  cat("--- GBM Simulation (single asset) ---\n")
  sigma_ann <- sd(returns) * sqrt(252)
  mu_ann    <- mean(returns) * 252
  S_gbm <- simulate_gbm(S0=100, mu=mu_ann, sigma=sigma_ann,
                          T_horizon=1, n_steps=n, n_paths=500)
  S_T   <- S_gbm[dim(S_gbm)[1], 1, ]
  gbm_var5 <- quantile(S_T/100 - 1, 0.05)
  cat(sprintf("GBM 5%% VaR (1yr): %.2f%%\n", 100*abs(gbm_var5)))

  # 2. Jump-diffusion comparison
  cat("\n--- Jump-Diffusion vs GBM ---\n")
  # Estimate jump parameters from return distribution
  kurt <- mean((returns-mean(returns))^4)/sd(returns)^4 - 3
  lambda_est <- max(1, kurt * 0.5)  # rough estimate
  S_jd <- simulate_jump_diffusion(S0=100, mu=mu_ann, sigma=sigma_ann*0.8,
                                   lambda=lambda_est, mu_j=-0.03, sigma_j=0.05,
                                   T_horizon=1, n_steps=n, n_paths=500)
  S_T_jd <- S_jd[dim(S_jd)[1], ]
  jd_var5 <- quantile(S_T_jd/100 - 1, 0.05)
  cat(sprintf("JD 5%% VaR (1yr):  %.2f%%\n", 100*abs(jd_var5)))
  cat(sprintf("VaR uplift (JD vs GBM): %.1f%%\n",
              100*(abs(jd_var5)/abs(gbm_var5)-1)))

  # 3. Block bootstrap of strategy
  cat("\n--- Block Bootstrap ---\n")
  boot_res <- block_bootstrap_strategy(returns, block_size=10, n_sim=n_bootstrap)

  # 4. Power analysis
  cat("\n--- Power Analysis ---\n")
  pow_res <- power_analysis_returns(mean(returns), sd(returns))

  # 5. Variance reduction demo (for option pricing context)
  cat("\n--- Variance Reduction (ATM call pricing demo) ---\n")
  S0=100; K=100; T_e=30/365; sigma_v=sigma_ann; r=0
  f_call  <- function(Z) pmax(S0*exp((r-0.5*sigma_v^2)*T_e + sigma_v*sqrt(T_e)*Z) - K, 0) *
                           exp(-r*T_e)
  mc_anti <- mc_antithetic(f_call, n_sim=10000)

  cat("\n=== SIMULATION SUMMARY ===\n")
  cat(sprintf("Observed Sharpe: %.3f\n", mean(returns)/sd(returns)*sqrt(252)))
  cat(sprintf("Bootstrap Sharpe (median): %.3f\n", median(boot_res$sharpes)))
  cat(sprintf("Bootstrap 95%% CI: [%.3f, %.3f]\n",
              quantile(boot_res$sharpes,0.025), quantile(boot_res$sharpes,0.975)))
  cat(sprintf("Obs needed for 80%% power: %d days\n", pow_res$n_required))

  invisible(list(gbm=S_gbm, jd=S_jd, bootstrap=boot_res, power=pow_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 500
# # Simulate a strategy with positive Sharpe
# strategy_ret <- rnorm(n, mean=0.0008, sd=0.015)
# result <- run_simulation_study(strategy_ret, "BTC_momentum")

# =============================================================================
# EXTENDED SIMULATION STUDIES: Advanced Monte Carlo Methods, Quasi-Random,
# Multi-Asset Simulation, Variance Reduction, Backtesting Simulation
# =============================================================================

# -----------------------------------------------------------------------------
# Quasi-Monte Carlo using Halton sequence (low-discrepancy)
# Halton sequences fill space more uniformly than pseudo-random numbers
# Critical for high-dimensional option pricing (Heston model, basket options)
# -----------------------------------------------------------------------------
halton_sequence <- function(n, base) {
  # Generate n elements of Halton sequence in given base
  result <- numeric(n)
  for (i in 1:n) {
    f <- 1; r <- 0; k <- i
    while (k > 0) {
      f <- f / base
      r <- r + f * (k %% base)
      k <- floor(k / base)
    }
    result[i] <- r
  }
  result
}

halton_matrix <- function(n, d) {
  # Generate n x d matrix of Halton quasi-random numbers
  # Uses first d prime numbers as bases
  primes <- c(2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71)
  if (d > length(primes)) stop("d exceeds available prime bases")
  mat <- matrix(0, n, d)
  for (j in 1:d) mat[, j] <- halton_sequence(n, primes[j])
  mat
}

qmc_option_price <- function(S0, K, r, sigma, T, n_sim = 10000,
                               option_type = "call") {
  # Quasi-Monte Carlo GBM option pricing using Halton sequence
  # Compare with standard MC to quantify variance reduction from QMC
  u <- halton_sequence(n_sim, base = 2)
  z <- qnorm(pmax(pmin(u, 1-1e-10), 1e-10))  # inverse normal CDF

  ST <- S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*z)

  payoffs <- if (option_type == "call") pmax(ST - K, 0) else pmax(K - ST, 0)
  price <- exp(-r*T) * mean(payoffs)
  se    <- exp(-r*T) * sd(payoffs) / sqrt(n_sim)

  # Compare with standard MC
  z_mc <- rnorm(n_sim)
  ST_mc <- S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*z_mc)
  payoffs_mc <- if (option_type == "call") pmax(ST_mc - K, 0) else pmax(K - ST_mc, 0)
  price_mc <- exp(-r*T) * mean(payoffs_mc)
  se_mc    <- exp(-r*T) * sd(payoffs_mc) / sqrt(n_sim)

  list(qmc_price = price, qmc_se = se, mc_price = price_mc, mc_se = se_mc,
       variance_reduction_ratio = se_mc^2 / se^2)
}

# -----------------------------------------------------------------------------
# Correlated Multi-Asset Monte Carlo Simulation
# Simulate N assets with given correlation structure using Cholesky decomposition
# Used for portfolio VaR, basket option pricing, macro scenario generation
# -----------------------------------------------------------------------------
simulate_correlated_assets <- function(mu_vec, sigma_vec, corr_mat, n_steps,
                                        dt = 1/252, n_paths = 1000,
                                        S0 = NULL, antithetic = TRUE) {
  N <- length(mu_vec)
  if (is.null(S0)) S0 <- rep(100, N)

  # Cholesky decomposition of correlation matrix
  L <- tryCatch(t(chol(corr_mat)), error = function(e) {
    # Make PD via eigenvalue adjustment
    ev <- eigen(corr_mat); ev$values <- pmax(ev$values, 1e-6)
    R <- ev$vectors %*% diag(sqrt(ev$values))
    R
  })

  if (antithetic) n_paths_half <- n_paths / 2 else n_paths_half <- n_paths

  # Drift and diffusion
  drift <- (mu_vec - 0.5 * sigma_vec^2) * dt
  diff_coef <- sigma_vec * sqrt(dt)

  # Price paths: list of n_paths matrices (n_steps x N)
  all_paths <- array(NA, dim = c(n_steps + 1, N, n_paths))
  all_paths[1, , ] <- S0

  for (path in 1:n_paths_half) {
    Z <- matrix(rnorm(n_steps * N), n_steps, N)
    Z_corr <- Z %*% t(L)
    shocks  <- sweep(Z_corr, 2, diff_coef, "*")
    log_path <- matrix(0, n_steps + 1, N)
    log_path[1, ] <- log(S0)
    for (t in 1:n_steps) {
      log_path[t+1, ] <- log_path[t, ] + drift + shocks[t, ]
    }
    all_paths[, , path] <- exp(log_path)

    if (antithetic && path <= n_paths_half) {
      Z_anti <- -Z
      Z_anti_corr <- Z_anti %*% t(L)
      shocks_anti  <- sweep(Z_anti_corr, 2, diff_coef, "*")
      log_path_anti <- matrix(0, n_steps + 1, N)
      log_path_anti[1, ] <- log(S0)
      for (t in 1:n_steps) {
        log_path_anti[t+1, ] <- log_path_anti[t, ] + drift + shocks_anti[t, ]
      }
      all_paths[, , n_paths_half + path] <- exp(log_path_anti)
    }
  }

  # Terminal prices
  terminal <- all_paths[n_steps + 1, , ]
  terminal_returns <- log(terminal / matrix(S0, N, n_paths))

  list(
    paths = all_paths,
    terminal_prices = terminal,
    terminal_log_returns = terminal_returns,
    asset_names = if(!is.null(names(mu_vec))) names(mu_vec) else paste0("A",1:N)
  )
}

# -----------------------------------------------------------------------------
# Stratified Sampling Monte Carlo: divide sample space into strata
# Ensures each region of the distribution is sampled, reducing variance
# Particularly useful for far out-of-the-money option pricing
# -----------------------------------------------------------------------------
mc_stratified <- function(n_strata = 100, n_per_stratum = 10,
                           payoff_fn, dist_fn = qnorm) {
  # Stratify [0,1] into n_strata equal-probability cells
  n_total <- n_strata * n_per_stratum
  strata_bounds <- seq(0, 1, length.out = n_strata + 1)

  samples <- numeric(n_total)
  k <- 1
  for (s in 1:n_strata) {
    # Uniform draws within stratum s
    u <- runif(n_per_stratum, strata_bounds[s], strata_bounds[s+1])
    samples[k:(k + n_per_stratum - 1)] <- dist_fn(u)
    k <- k + n_per_stratum
  }

  payoffs  <- payoff_fn(samples)
  estimate <- mean(payoffs)
  se_strat <- sd(payoffs) / sqrt(n_total)

  # Compare with crude MC
  z_mc <- dist_fn(runif(n_total))
  payoffs_mc <- payoff_fn(z_mc)
  se_mc <- sd(payoffs_mc) / sqrt(n_total)

  list(
    estimate = estimate, se = se_strat,
    mc_estimate = mean(payoffs_mc), mc_se = se_mc,
    variance_reduction = se_mc^2 / se_strat^2
  )
}

# -----------------------------------------------------------------------------
# Variance Reduction via Common Random Numbers (CRN)
# Use same random numbers to compare two strategies/models
# Reduces noise in A/B comparisons of trading strategies
# -----------------------------------------------------------------------------
mc_common_random <- function(model_fn_a, model_fn_b, n_sim = 10000, seed = 42) {
  set.seed(seed)
  z <- rnorm(n_sim)

  result_a <- model_fn_a(z)
  result_b <- model_fn_b(z)

  diff <- result_a - result_b

  list(
    mean_a = mean(result_a), se_a = sd(result_a)/sqrt(n_sim),
    mean_b = mean(result_b), se_b = sd(result_b)/sqrt(n_sim),
    mean_diff = mean(diff),
    se_diff_crn = sd(diff) / sqrt(n_sim),  # CRN standard error
    se_diff_iid = sqrt(var(result_a) + var(result_b)) / sqrt(n_sim),
    variance_reduction = (var(result_a) + var(result_b)) / var(diff),
    t_stat = mean(diff) / (sd(diff)/sqrt(n_sim)),
    p_value = 2 * pt(-abs(mean(diff) / (sd(diff)/sqrt(n_sim))), df = n_sim - 1)
  )
}

# -----------------------------------------------------------------------------
# Bootstrap-Based Overfitting Detection (Bailey et al. CSCV method)
# Combinatorially Symmetric Cross-Validation for strategy selection
# Measures probability of overfitting when selecting best from N strategies
# -----------------------------------------------------------------------------
cscv_backtest <- function(strategy_returns_matrix, n_splits = 16) {
  # strategy_returns_matrix: T x S matrix (T time periods, S strategies)
  T_obs <- nrow(strategy_returns_matrix)
  S     <- ncol(strategy_returns_matrix)

  # Split time series into n_splits equal subsets
  split_size <- floor(T_obs / n_splits)
  splits <- lapply(1:n_splits, function(i) {
    ((i-1)*split_size + 1):(i*split_size)
  })

  # All combinations of n_splits/2 for in-sample
  n_is <- n_splits / 2
  combs <- combn(n_splits, n_is)
  n_combs <- ncol(combs)

  pbo_list <- numeric(n_combs)

  for (c_idx in 1:n_combs) {
    is_splits  <- combs[, c_idx]
    oos_splits <- setdiff(1:n_splits, is_splits)

    is_idx  <- unlist(splits[is_splits])
    oos_idx <- unlist(splits[oos_splits])

    # Select best strategy in IS
    is_sharpes  <- apply(strategy_returns_matrix[is_idx, ], 2,
                          function(r) mean(r)/sd(r)*sqrt(252))
    best_s <- which.max(is_sharpes)

    # Performance rank of selected strategy in OOS
    oos_sharpes <- apply(strategy_returns_matrix[oos_idx, ], 2,
                          function(r) mean(r)/sd(r)*sqrt(252))
    rank_best_oos <- rank(oos_sharpes)[best_s]
    pbo_list[c_idx] <- rank_best_oos <= (S / 2)  # below median = overfit
  }

  pbo <- mean(pbo_list)  # Probability of Backtest Overfitting

  list(
    pbo = pbo,
    n_strategies = S, n_combinations = n_combs,
    interpretation = if(pbo > 0.5) "likely overfit" else "acceptable",
    pbo_logit = log(pbo / (1 - pbo + 1e-10))
  )
}

# -----------------------------------------------------------------------------
# Fat-Tail Asset Price Simulation: Student-t GBM
# Standard GBM uses normal innovations; Student-t captures crypto fat tails
# -----------------------------------------------------------------------------
simulate_t_gbm <- function(S0, mu, sigma, nu = 5, n_steps, dt = 1/252,
                             n_paths = 1000) {
  # Standardized Student-t innovations: scale so variance = sigma^2
  # t(nu) has variance nu/(nu-2) for nu > 2
  t_scale <- sqrt((nu - 2) / nu) * sigma * sqrt(dt)
  drift   <- (mu - 0.5 * sigma^2) * dt  # drift unchanged

  paths <- matrix(NA, n_steps + 1, n_paths)
  paths[1, ] <- S0

  for (t in 1:n_steps) {
    z <- rt(n_paths, df = nu)
    paths[t+1, ] <- paths[t, ] * exp(drift + t_scale * z)
  }

  terminal <- paths[n_steps + 1, ]
  list(
    paths = paths,
    terminal = terminal,
    mean_terminal = mean(terminal),
    sd_terminal = sd(terminal),
    skewness = mean(((terminal - mean(terminal))/sd(terminal))^3),
    kurtosis = mean(((terminal - mean(terminal))/sd(terminal))^4) - 3
  )
}

# -----------------------------------------------------------------------------
# Scenario Generator: Markov Chain for Macro Regime Simulation
# Simulate sequences of macro states (expansion, slowdown, recession, recovery)
# Each state has different return/vol characteristics for stress testing
# -----------------------------------------------------------------------------
simulate_macro_scenarios <- function(n_periods, regime_params,
                                      transition_matrix, initial_regime = 1) {
  # regime_params: list of lists with mu, sigma per regime
  # transition_matrix: S x S matrix of transition probabilities
  S <- length(regime_params)

  # Simulate regime sequence
  regime_seq <- integer(n_periods)
  regime_seq[1] <- initial_regime
  for (t in 2:n_periods) {
    probs <- transition_matrix[regime_seq[t-1], ]
    regime_seq[t] <- sample(1:S, 1, prob = probs)
  }

  # Simulate returns conditional on regime
  returns <- numeric(n_periods)
  for (t in 1:n_periods) {
    r  <- regime_seq[t]
    mu <- regime_params[[r]]$mu
    sg <- regime_params[[r]]$sigma
    returns[t] <- rnorm(1, mu, sg)
  }

  # Summary statistics by regime
  stats_by_regime <- lapply(1:S, function(r) {
    idx <- regime_seq == r
    list(regime = r, freq = mean(idx),
         mean_return = mean(returns[idx]),
         vol = sd(returns[idx]))
  })

  list(
    returns = returns,
    regime_sequence = regime_seq,
    regime_stats = do.call(rbind, lapply(stats_by_regime, as.data.frame)),
    unconditional_mean = mean(returns),
    unconditional_vol = sd(returns),
    stationary_distribution = {
      # Stationary dist: solve pi * P = pi
      # Approximate via eigen decomposition
      ev <- eigen(t(transition_matrix))
      pi_approx <- Re(ev$vectors[, 1])
      pi_approx / sum(pi_approx)
    }
  )
}

# Default crypto regime parameters for demonstration:
# Bull regime: mu=0.003, sigma=0.03
# Sideways: mu=0.0005, sigma=0.025
# Bear: mu=-0.003, sigma=0.045
crypto_regime_params <- list(
  list(mu = 0.003,  sigma = 0.030),   # Bull
  list(mu = 0.0005, sigma = 0.025),   # Sideways
  list(mu = -0.003, sigma = 0.045)    # Bear
)
crypto_transition_matrix <- matrix(c(
  0.92, 0.06, 0.02,
  0.05, 0.90, 0.05,
  0.03, 0.12, 0.85
), 3, 3, byrow = TRUE)

# Extended example:
# sim <- simulate_correlated_assets(c(0.001,0.0008), c(0.03,0.025),
#   matrix(c(1,0.7,0.7,1),2,2), n_steps=252, n_paths=5000)
# qmc <- qmc_option_price(30000, 30000, 0.04, 0.7, 1, n_sim=10000)
# macro_sim <- simulate_macro_scenarios(500, crypto_regime_params,
#   crypto_transition_matrix)
