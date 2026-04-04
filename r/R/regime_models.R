# regime_models.R
# Regime detection and regime-conditional modeling for SRFM research.
# Implements: HMM (depmix), Markov switching regression (MSwM),
#             structural break tests, rolling correlation regimes, GARCH-DCC regimes.
# Dependencies: depmixS4, MSwM, strucchange, zoo, xts, ggplot2, rugarch, rmgarch

library(zoo)
library(xts)
library(ggplot2)

.has_depmix    <- requireNamespace("depmixS4",   quietly = TRUE)
.has_mswm      <- requireNamespace("MSwM",       quietly = TRUE)
.has_strucchange <- requireNamespace("strucchange", quietly = TRUE)
.has_rugarch   <- requireNamespace("rugarch",    quietly = TRUE)
.has_rmgarch   <- requireNamespace("rmgarch",    quietly = TRUE)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Hidden Markov Model (depmixS4)
# ─────────────────────────────────────────────────────────────────────────────

#' fit_hmm_gaussian
#'
#' Fit a Gaussian HMM to return series using depmixS4.
#' Each state has Gaussian emission N(mu_k, sigma_k).
#'
#' @param returns numeric vector or xts of returns
#' @param n_states integer, number of hidden states
#' @param n_iter integer, EM iterations
#' @param n_trials integer, random restarts
#' @return list with model, viterbi_states, state_probs, BIC
fit_hmm_gaussian <- function(returns, n_states = 3L, n_iter = 500L, n_trials = 5L) {
  if (!.has_depmix) {
    message("depmixS4 not available; using custom Baum-Welch HMM.")
    return(fit_hmm_custom(returns, n_states = n_states, n_iter = n_iter))
  }

  if (is.xts(returns)) {
    dates <- index(returns)
    r_vec <- as.numeric(returns)
  } else {
    dates <- seq_along(returns)
    r_vec <- as.numeric(returns)
  }

  df <- data.frame(ret = r_vec)

  library(depmixS4)
  best_model <- NULL
  best_loglik <- -Inf

  for (trial in seq_len(n_trials)) {
    set.seed(trial * 42L)
    tryCatch({
      mod <- depmix(ret ~ 1, data = df, nstates = n_states,
                    family = gaussian())
      fit <- fit(mod, verbose = FALSE, emcontrol = em.control(maxit = n_iter))
      ll <- logLik(fit)
      if (ll > best_loglik) {
        best_loglik <- ll
        best_model  <- fit
      }
    }, error = function(e) NULL)
  }

  if (is.null(best_model)) {
    return(fit_hmm_custom(returns, n_states = n_states, n_iter = n_iter))
  }

  # Extract results
  vit      <- viterbi(best_model)
  state_probs <- posterior(best_model)

  # State parameters
  state_params <- lapply(seq_len(n_states), function(k) {
    coef_k <- getpars(best_model$response[[k]][[1]])
    list(mu = coef_k[1], sigma = coef_k[2])
  })

  list(
    model         = best_model,
    viterbi_states = vit$state,
    state_probs   = as.matrix(state_probs[, -1, drop = FALSE]),
    state_params  = state_params,
    log_likelihood = as.numeric(best_loglik),
    bic           = BIC(best_model),
    aic           = AIC(best_model),
    n_states      = n_states,
    dates         = dates
  )
}


#' fit_hmm_custom
#'
#' Custom Baum-Welch HMM implementation (fallback when depmixS4 unavailable).
#' Gaussian emissions.
#'
#' @param returns numeric vector or xts
#' @param n_states integer
#' @param n_iter integer, EM iterations
#' @return list with viterbi_states, state_probs, state_params
fit_hmm_custom <- function(returns, n_states = 3L, n_iter = 200L) {
  if (is.xts(returns)) {
    dates <- index(returns)
    r     <- as.numeric(returns)
  } else {
    dates <- seq_along(returns)
    r     <- as.numeric(returns)
  }
  n <- length(r)
  K <- n_states

  # Initialize parameters
  # Sort returns and assign initial states by quantile
  q_breaks <- quantile(r, seq(0, 1, length.out = K + 1), na.rm = TRUE)
  mu    <- numeric(K)
  sigma <- numeric(K)
  for (k in seq_len(K)) {
    idx_k  <- r >= q_breaks[k] & r < q_breaks[k+1]
    mu[k]    <- mean(r[idx_k], na.rm = TRUE)
    sigma[k] <- sd(r[idx_k], na.rm = TRUE)
    if (is.na(sigma[k]) || sigma[k] < 1e-8) sigma[k] <- sd(r, na.rm = TRUE)
  }
  # Sort by mu (ascending: bearish → bullish)
  ord    <- order(mu)
  mu     <- mu[ord]
  sigma  <- sigma[ord]

  # Transition matrix: slight persistence
  A <- matrix(0.1 / (K - 1), K, K)
  diag(A) <- 0.85
  A <- A / rowSums(A)

  # Initial distribution
  pi0 <- rep(1/K, K)

  # Gaussian density
  gauss_dens <- function(x, m, s) dnorm(x, mean = m, sd = max(s, 1e-8))

  # EM algorithm (Baum-Welch)
  log_lik <- -Inf

  for (iter in seq_len(n_iter)) {
    # E-step: forward-backward
    # Emission probabilities
    B <- matrix(NA_real_, n, K)
    for (k in seq_len(K)) {
      B[, k] <- gauss_dens(r, mu[k], sigma[k])
    }
    B[B < 1e-300] <- 1e-300

    # Forward algorithm (scaled)
    alpha_sc  <- matrix(0.0, n, K)
    c_scale   <- numeric(n)
    alpha_sc[1, ] <- pi0 * B[1, ]
    c_scale[1]    <- sum(alpha_sc[1, ])
    alpha_sc[1, ] <- alpha_sc[1, ] / c_scale[1]

    for (t in 2:n) {
      for (k in seq_len(K)) {
        alpha_sc[t, k] <- sum(alpha_sc[t-1, ] * A[, k]) * B[t, k]
      }
      c_scale[t] <- sum(alpha_sc[t, ])
      if (c_scale[t] > 0) {
        alpha_sc[t, ] <- alpha_sc[t, ] / c_scale[t]
      }
    }

    # Backward algorithm (scaled)
    beta_sc <- matrix(0.0, n, K)
    beta_sc[n, ] <- 1.0
    for (t in (n-1):1) {
      for (i in seq_len(K)) {
        beta_sc[t, i] <- sum(A[i, ] * B[t+1, ] * beta_sc[t+1, ])
      }
      bsum <- sum(beta_sc[t, ])
      if (bsum > 0) beta_sc[t, ] <- beta_sc[t, ] / bsum
    }

    # Gamma: P(state_t = k | obs)
    gamma <- alpha_sc * beta_sc
    rs    <- rowSums(gamma)
    rs[rs < 1e-300] <- 1e-300
    gamma <- gamma / rs

    # Xi: P(state_t=i, state_{t+1}=j | obs)
    xi <- array(0.0, dim = c(n-1, K, K))
    for (t in seq_len(n-1)) {
      for (i in seq_len(K)) {
        for (j in seq_len(K)) {
          xi[t, i, j] <- alpha_sc[t, i] * A[i, j] * B[t+1, j] * beta_sc[t+1, j]
        }
      }
      xi_sum <- sum(xi[t, , ])
      if (xi_sum > 0) xi[t, , ] <- xi[t, , ] / xi_sum
    }

    # M-step: update parameters
    pi0_new <- gamma[1, ]
    A_new   <- matrix(0.0, K, K)
    for (i in seq_len(K)) {
      for (j in seq_len(K)) {
        A_new[i, j] <- sum(xi[, i, j])
      }
      rs_a <- sum(A_new[i, ])
      if (rs_a > 0) A_new[i, ] <- A_new[i, ] / rs_a
    }

    mu_new    <- colSums(gamma * r) / (colSums(gamma) + 1e-10)
    sigma_new <- sqrt(colSums(gamma * outer(r, mu_new, "-")^2) / (colSums(gamma) + 1e-10))
    sigma_new <- pmax(sigma_new, 1e-8)

    # Convergence
    new_ll <- sum(log(c_scale + 1e-300))
    if (abs(new_ll - log_lik) < 1e-6) {
      log_lik <- new_ll
      break
    }
    log_lik <- new_ll

    # Update
    pi0   <- pi0_new
    A     <- A_new
    mu    <- mu_new
    sigma <- sigma_new
  }

  # Viterbi decoding
  vit <- hmm_viterbi(r, A, pi0, mu, sigma)

  state_params <- lapply(seq_len(K), function(k) list(mu = mu[k], sigma = sigma[k]))

  list(
    viterbi_states = vit,
    state_probs    = gamma,
    state_params   = state_params,
    transition_matrix = A,
    initial_dist   = pi0,
    log_likelihood = log_lik,
    n_states       = K,
    dates          = dates
  )
}


#' hmm_viterbi
#'
#' Viterbi decoding for Gaussian HMM.
#'
#' @param r numeric observations
#' @param A K × K transition matrix
#' @param pi0 K-length initial distribution
#' @param mu K-length means
#' @param sigma K-length standard deviations
#' @return integer vector of most likely states
hmm_viterbi <- function(r, A, pi0, mu, sigma) {
  n <- length(r)
  K <- length(mu)

  delta <- matrix(-Inf, n, K)
  psi   <- matrix(0L, n, K)

  # Log emission
  log_B <- matrix(NA_real_, n, K)
  for (k in seq_len(K)) {
    log_B[, k] <- dnorm(r, mean = mu[k], sd = max(sigma[k], 1e-8), log = TRUE)
  }

  delta[1, ] <- log(pi0 + 1e-300) + log_B[1, ]

  for (t in 2:n) {
    for (j in seq_len(K)) {
      score    <- delta[t-1, ] + log(A[, j] + 1e-300)
      best_i   <- which.max(score)
      psi[t, j]   <- best_i
      delta[t, j] <- score[best_i] + log_B[t, j]
    }
  }

  # Backtrack
  states <- integer(n)
  states[n] <- which.max(delta[n, ])
  for (t in (n-1):1) {
    states[t] <- psi[t+1, states[t+1]]
  }

  states
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Markov Switching Regression (MSwM)
# ─────────────────────────────────────────────────────────────────────────────

#' fit_markov_switching
#'
#' Fit Markov switching regression model.
#' Model: y_t = alpha_k + beta_k * X_t + epsilon_{k,t} where k = regime at t
#'
#' @param y numeric vector (dependent variable)
#' @param X matrix of regressors (n × p)
#' @param n_regimes integer (usually 2 or 3)
#' @return list with regime assignments, parameters, transition matrix
fit_markov_switching <- function(y, X, n_regimes = 2L) {
  if (!.has_mswm) {
    message("MSwM not available; using custom MS-regression.")
    return(fit_ms_regression_custom(y, X, n_regimes))
  }

  library(MSwM)
  df       <- data.frame(y = y, X)
  fmla_str <- paste("y ~", paste(colnames(X), collapse = " + "))
  base_lm  <- lm(as.formula(fmla_str), data = df)

  fit <- tryCatch({
    msmFit(base_lm, k = n_regimes, sw = rep(TRUE, ncol(X) + 2L),
           control = list(parallelization = FALSE, maxiter = 500L))
  }, error = function(e) {
    message("MSwM failed: ", e$message)
    NULL
  })

  if (is.null(fit)) {
    return(fit_ms_regression_custom(y, X, n_regimes))
  }

  states <- apply(fit@Fit@smoProb, 1, which.max)

  list(
    model         = fit,
    states        = states,
    smooth_probs  = fit@Fit@smoProb,
    n_regimes     = n_regimes
  )
}


#' fit_ms_regression_custom
#'
#' Custom Markov switching regression via EM.
#'
#' @param y numeric vector
#' @param X matrix n × p
#' @param n_regimes integer
#' @return list with states, regime_params, transition_matrix
fit_ms_regression_custom <- function(y, X, n_regimes = 2L) {
  n <- length(y)
  K <- n_regimes
  p <- ncol(X)
  X_aug <- cbind(1, X)
  q     <- ncol(X_aug)

  # Initialize via k-means on y
  km   <- kmeans(y, centers = K, nstart = 10)
  gamma <- matrix(0.0, n, K)
  for (i in seq_len(n)) gamma[i, km$cluster[i]] <- 1.0

  # EM
  beta  <- matrix(0.0, q, K)
  sigma <- rep(sd(y), K)
  A     <- matrix(0.1/(K-1), K, K)
  diag(A) <- 0.85
  A <- A / rowSums(A)

  for (iter in seq_len(300L)) {
    # M-step: update beta, sigma for each regime
    for (k in seq_len(K)) {
      w_k <- gamma[, k]
      if (sum(w_k) < 2) next
      W    <- diag(w_k)
      XtWX <- t(X_aug) %*% W %*% X_aug
      XtWy <- t(X_aug) %*% W %*% y
      beta[, k] <- tryCatch(
        solve(XtWX + 1e-8 * diag(q), XtWy),
        error = function(e) beta[, k]
      )
      resid_k  <- y - X_aug %*% beta[, k]
      sigma[k] <- sqrt(sum(w_k * resid_k^2) / (sum(w_k) + 1e-10))
      sigma[k] <- max(sigma[k], 1e-8)
    }

    # Update transition matrix from smoothed probabilities
    for (i in seq_len(K)) {
      for (j in seq_len(K)) {
        A[i, j] <- sum(gamma[1:(n-1), i] * gamma[2:n, j])
      }
      rs <- sum(A[i, ])
      if (rs > 0) A[i, ] <- A[i, ] / rs
    }

    # E-step: compute emission probabilities
    B <- matrix(NA_real_, n, K)
    for (k in seq_len(K)) {
      mu_k <- X_aug %*% beta[, k]
      B[, k] <- dnorm(y, mean = mu_k, sd = sigma[k])
    }
    B[B < 1e-300] <- 1e-300

    # Forward pass
    pi0 <- colMeans(gamma)
    alpha_sc <- matrix(0.0, n, K)
    c_s      <- numeric(n)
    alpha_sc[1, ] <- pi0 * B[1, ]
    c_s[1] <- sum(alpha_sc[1, ])
    alpha_sc[1, ] <- alpha_sc[1, ] / max(c_s[1], 1e-300)
    for (t in 2:n) {
      for (k in seq_len(K)) {
        alpha_sc[t, k] <- sum(alpha_sc[t-1, ] * A[, k]) * B[t, k]
      }
      c_s[t] <- sum(alpha_sc[t, ])
      alpha_sc[t, ] <- alpha_sc[t, ] / max(c_s[t], 1e-300)
    }
    # Backward pass
    beta_sc <- matrix(1.0, n, K)
    for (t in (n-1):1) {
      for (i in seq_len(K)) {
        beta_sc[t, i] <- sum(A[i, ] * B[t+1, ] * beta_sc[t+1, ])
      }
      bsum <- sum(beta_sc[t, ])
      if (bsum > 0) beta_sc[t, ] <- beta_sc[t, ] / bsum
    }
    new_gamma <- alpha_sc * beta_sc
    rs <- rowSums(new_gamma)
    rs[rs < 1e-300] <- 1e-300
    gamma <- new_gamma / rs
  }

  states <- apply(gamma, 1, which.max)

  list(
    states            = states,
    smooth_probs      = gamma,
    regime_params     = lapply(seq_len(K), function(k) list(beta = beta[, k], sigma = sigma[k])),
    transition_matrix = A,
    n_regimes         = K
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Structural Break Tests
# ─────────────────────────────────────────────────────────────────────────────

#' detect_structural_breaks
#'
#' Detect structural breaks using the Bai-Perron method (strucchange package)
#' and supplementary CUSUM test.
#'
#' @param y numeric vector or xts of dependent variable
#' @param X matrix of regressors (optional; uses intercept only if NULL)
#' @param max_breaks integer, maximum breaks to consider
#' @return list with break_dates, break_indices, BIC_table, cusum_test
detect_structural_breaks <- function(y, X = NULL, max_breaks = 5L) {
  if (is.xts(y)) {
    dates <- index(y)
    y_vec <- as.numeric(y)
  } else {
    dates <- seq_along(y)
    y_vec <- as.numeric(y)
  }
  n <- length(y_vec)

  # CUSUM structural change (base R version)
  cusum_stat <- cusum_test(y_vec)

  if (.has_strucchange) {
    library(strucchange)
    df_y <- data.frame(y = y_vec)
    if (!is.null(X)) {
      df_y <- cbind(df_y, X)
      fmla <- as.formula(paste("y ~", paste(colnames(X), collapse = " + ")))
    } else {
      fmla <- y ~ 1
    }

    bp <- tryCatch(
      breakpoints(fmla, data = df_y, breaks = max_breaks),
      error = function(e) NULL
    )
    if (!is.null(bp)) {
      bp_summary <- summary(bp)
      best_m <- which.min(bp_summary$RSS["BIC", ])
      if (length(best_m) > 0 && best_m > 0) {
        bp_best <- breakpoints(bp, breaks = best_m - 1L)
        break_indices <- bp_best$breakpoints
      } else {
        break_indices <- integer(0)
      }
      break_dates <- if (is.numeric(dates)) dates[break_indices] else dates[break_indices]

      return(list(
        break_indices  = break_indices,
        break_dates    = break_dates,
        bp_object      = bp,
        cusum          = cusum_stat,
        n_breaks       = length(break_indices)
      ))
    }
  }

  # Fallback: simple Chow-test grid
  break_indices <- simple_break_detection(y_vec, max_breaks = max_breaks)
  break_dates   <- if (is.numeric(dates)) dates[break_indices] else dates[break_indices]

  list(
    break_indices = break_indices,
    break_dates   = break_dates,
    cusum         = cusum_stat,
    n_breaks      = length(break_indices)
  )
}


#' cusum_test
#'
#' Compute empirical CUSUM statistic for structural stability.
#'
#' @param y numeric vector
#' @return list with statistic, critical_value, reject_H0
cusum_test <- function(y) {
  n  <- length(y)
  mu <- mean(y, na.rm = TRUE)
  sg <- sd(y, na.rm = TRUE)
  if (sg < 1e-10) sg <- 1e-10

  std_resid <- (y - mu) / sg
  cusum_vec <- cumsum(std_resid) / sqrt(n)
  stat      <- max(abs(cusum_vec))

  # Asymptotic critical value at 5%: ~1.36
  crit_val  <- 1.36
  list(
    statistic     = stat,
    critical_value = crit_val,
    reject_H0     = stat > crit_val,
    cusum_path    = cusum_vec
  )
}


#' simple_break_detection
#'
#' Detect breaks via exhaustive grid Chow-test (for small n).
#'
#' @param y numeric vector
#' @param max_breaks integer
#' @return integer vector of break point indices
simple_break_detection <- function(y, max_breaks = 3L) {
  n   <- length(y)
  min_seg <- max(5L, round(n * 0.05))
  breaks  <- integer(0)
  remaining <- list(1:n)

  for (b in seq_len(max_breaks)) {
    best_idx   <- NA_integer_
    best_fstat <- 0.0

    for (seg in remaining) {
      if (length(seg) < 2 * min_seg) next
      cands <- seg[(min_seg):(length(seg) - min_seg)]
      for (idx_local in seq_along(cands)) {
        bp <- cands[idx_local]
        y1 <- y[seg[1:idx_local]]
        y2 <- y[seg[(idx_local+1):length(seg)]]
        if (length(y1) < 2 || length(y2) < 2) next
        # F-statistic for mean shift
        grand_ss <- sum((y[seg] - mean(y[seg]))^2)
        split_ss <- sum((y1 - mean(y1))^2) + sum((y2 - mean(y2))^2)
        f_stat   <- (grand_ss - split_ss) / (split_ss / (length(seg) - 2))
        if (!is.finite(f_stat)) next
        if (f_stat > best_fstat) {
          best_fstat <- f_stat
          best_idx   <- bp
        }
      }
    }

    if (!is.na(best_idx) && best_fstat > qf(0.95, 1, n - 2)) {
      breaks <- c(breaks, best_idx)
      # Update remaining segments (simplified: don't recurse)
    } else {
      break
    }
  }

  sort(breaks)
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Rolling Correlation Regimes
# ─────────────────────────────────────────────────────────────────────────────

#' rolling_correlation_regimes
#'
#' Compute rolling pairwise correlations and classify into low/medium/high regimes.
#'
#' @param returns matrix n_obs × n_assets
#' @param window integer, rolling window
#' @param n_regimes integer, number of correlation regimes (2 or 3)
#' @return list with rolling_corr (xts), regime (integer vector), regime_stats
rolling_correlation_regimes <- function(returns, window = 60L, n_regimes = 3L) {
  n     <- nrow(returns)
  dates <- if (is.xts(returns)) index(returns) else seq_len(n)
  R     <- as.matrix(returns)
  p     <- ncol(R)

  # Average pairwise correlation per window
  avg_corr <- rep(NA_real_, n)
  for (t in window:n) {
    R_win  <- R[(t - window + 1):t, ]
    C      <- cor(R_win, use = "pairwise.complete.obs")
    # Upper triangle mean
    ut     <- C[upper.tri(C)]
    avg_corr[t] <- mean(ut, na.rm = TRUE)
  }

  # Classify regime using quantile breaks
  valid_idx <- which(!is.na(avg_corr))
  regimes   <- rep(NA_integer_, n)

  q_breaks <- quantile(avg_corr[valid_idx], probs = seq(0, 1, length.out = n_regimes + 1), na.rm = TRUE)
  for (i in valid_idx) {
    for (k in seq_len(n_regimes)) {
      if (avg_corr[i] >= q_breaks[k] && avg_corr[i] <= q_breaks[k+1]) {
        regimes[i] <- k
        break
      }
    }
  }

  regime_labels <- if (n_regimes == 2) c("Low Corr", "High Corr") else
                   if (n_regimes == 3) c("Low Corr", "Med Corr", "High Corr") else
                   paste("Regime", seq_len(n_regimes))

  regime_stats <- data.frame(
    regime     = seq_len(n_regimes),
    label      = regime_labels,
    lower_bound = q_breaks[seq_len(n_regimes)],
    upper_bound = q_breaks[seq_len(n_regimes) + 1L],
    n_obs      = tabulate(regimes, nbins = n_regimes)
  )

  list(
    rolling_avg_corr = xts(avg_corr, order.by = dates),
    regimes          = regimes,
    regime_labels    = regime_labels,
    regime_stats     = regime_stats
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: GARCH-DCC Regime
# ─────────────────────────────────────────────────────────────────────────────

#' fit_dcc_garch
#'
#' Fit DCC-GARCH model and extract regime from dynamic correlations.
#' Uses rugarch + rmgarch if available; otherwise uses EWMA fallback.
#'
#' @param returns matrix n_obs × n_assets
#' @param garch_order c(p,q) for each univariate GARCH
#' @param dcc_order c(a,b) for DCC
#' @return list with conditional_correlations (array), volatilities (matrix), regimes
fit_dcc_garch <- function(returns, garch_order = c(1,1), dcc_order = c(1,1)) {
  if (!.has_rugarch || !.has_rmgarch) {
    message("rugarch/rmgarch not available; using EWMA DCC fallback.")
    return(ewma_dcc(returns))
  }

  library(rugarch)
  library(rmgarch)

  p <- ncol(returns)
  garch_spec <- ugarchspec(
    variance.model    = list(model = "sGARCH", garchOrder = garch_order),
    mean.model        = list(armaOrder = c(0,0), include.mean = TRUE),
    distribution.model = "norm"
  )

  multispec <- multispec(replicate(p, garch_spec))
  dcc_spec  <- dccspec(multispec,
                        dccOrder = dcc_order,
                        distribution = "mvnorm")

  fit <- tryCatch(
    dccfit(dcc_spec, data = as.data.frame(returns), fit.control = list(eval.se = FALSE)),
    error = function(e) {
      message("DCC fit failed: ", e$message)
      NULL
    }
  )

  if (is.null(fit)) return(ewma_dcc(returns))

  # Extract conditional correlations
  R_t <- rcor(fit)  # p × p × n array
  # Average pairwise conditional correlation per time step
  n <- dim(R_t)[3]
  avg_cond_corr <- sapply(seq_len(n), function(t) {
    mean(R_t[upper.tri(R_t[, , t]), t], na.rm = TRUE)
  })

  # Conditional volatilities
  cond_vols <- sigma(fit)

  # Regime from correlation dynamics
  corr_regimes <- rolling_correlation_regimes(
    xts(returns, order.by = seq.Date(Sys.Date() - n + 1, Sys.Date(), by = "day")),
    window = 20L, n_regimes = 2L
  )

  list(
    model               = fit,
    conditional_corr    = R_t,
    avg_conditional_corr = avg_cond_corr,
    conditional_vols    = cond_vols,
    correlation_regimes = corr_regimes$regimes
  )
}


#' ewma_dcc
#'
#' EWMA-based dynamic conditional correlation (fallback).
#'
#' @param returns matrix n × p
#' @param lambda numeric, EWMA decay factor
#' @return list with conditional_corr (array), avg_conditional_corr, regimes
ewma_dcc <- function(returns, lambda = 0.94) {
  R <- as.matrix(returns)
  n <- nrow(R)
  p <- ncol(R)

  # EWMA covariance
  H <- cov(R) * (1 - lambda)  # initial
  H_array <- array(0.0, dim = c(p, p, n))
  H_array[, , 1] <- H

  for (t in 2:n) {
    r_t <- R[t-1, ]
    H <- lambda * H + (1 - lambda) * outer(r_t, r_t)
    H_array[, , t] <- H
  }

  # Conditional correlations
  R_array <- H_array
  avg_corr <- numeric(n)
  for (t in seq_len(n)) {
    D     <- diag(sqrt(diag(H_array[, , t])))
    D_inv <- diag(1 / diag(D))
    Rt    <- D_inv %*% H_array[, , t] %*% D_inv
    R_array[, , t] <- Rt
    avg_corr[t] <- mean(Rt[upper.tri(Rt)])
  }

  # Regimes
  q_mid <- median(avg_corr, na.rm = TRUE)
  regimes <- ifelse(avg_corr > q_mid, 2L, 1L)

  list(
    conditional_corr    = R_array,
    avg_conditional_corr = avg_corr,
    correlation_regimes = regimes
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Regime Visualization
# ─────────────────────────────────────────────────────────────────────────────

#' plot_regime_states
#'
#' Plot smoothed regime probabilities from HMM.
#'
#' @param hmm_result list from fit_hmm_gaussian or fit_hmm_custom
#' @param returns xts or numeric returns for top panel
#' @return ggplot2 object (list of panels)
plot_regime_states <- function(hmm_result, returns = NULL) {
  state_probs <- as.data.frame(hmm_result$state_probs)
  n <- nrow(state_probs)
  K <- ncol(state_probs)
  dates <- if (!is.null(hmm_result$dates)) as.Date(hmm_result$dates) else seq_len(n)
  colnames(state_probs) <- paste0("State ", seq_len(K))
  state_probs$date <- dates

  # Melt for ggplot
  df_long <- tidyr::gather(state_probs, key = "state", value = "probability", -date)

  state_colors <- c("#2196F3", "#e63946", "#4caf50", "#ff9800", "#9c27b0")[seq_len(K)]
  names(state_colors) <- paste0("State ", seq_len(K))

  p_probs <- ggplot(df_long, aes(x = date, y = probability, fill = state)) +
    geom_area(position = "stack") +
    scale_fill_manual(values = state_colors) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "HMM Regime Probabilities", x = "Date", y = "P(State)", fill = "State") +
    theme_minimal(base_size = 10)

  p_probs
}


#' plot_ms_regression
#'
#' Plot Markov switching regression states over time.
#'
#' @param ms_result list from fit_markov_switching
#' @param dates Date vector
#' @return ggplot2 object
plot_ms_regression <- function(ms_result, dates = NULL) {
  probs  <- as.data.frame(ms_result$smooth_probs)
  n      <- nrow(probs)
  K      <- ncol(probs)
  if (is.null(dates)) dates <- seq_len(n)
  colnames(probs) <- paste0("Regime ", seq_len(K))
  probs$date <- as.Date(dates)

  df_long <- tidyr::gather(probs, key = "regime", value = "probability", -date)

  colors <- c("#2196F3", "#e63946", "#4caf50")[seq_len(K)]
  names(colors) <- paste0("Regime ", seq_len(K))

  ggplot(df_long, aes(x = date, y = probability, color = regime)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = colors) +
    labs(title = "Markov Switching Regime Probabilities",
         x = "Date", y = "Probability", color = "Regime") +
    theme_minimal(base_size = 10)
}
