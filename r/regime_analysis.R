# regime_analysis.R
# Regime detection and analysis: Markov Switching regression (manual EM),
# Hidden Markov Model with 3 states, regime-conditional Sharpe ratios,
# time series diagnostics, regime persistence statistics.
#
# Dependencies: tidyverse, ggplot2, stats (base)
# Author: srfm-lab

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(scales)
})

# ===========================================================================
# 1. Markov Switching Regression -- manual EM algorithm
# ===========================================================================

#' Initialize Markov Switching parameters from k-means clustering on y
#' @param y       numeric vector (dependent variable)
#' @param x       numeric matrix of regressors (with or without intercept)
#' @param n_states number of regimes
ms_init_params <- function(y, x, n_states) {
  km <- kmeans(y, centers = n_states, nstart = 10)
  params <- lapply(seq_len(n_states), function(s) {
    idx <- which(km$cluster == s)
    ys  <- y[idx]
    xs  <- x[idx, , drop = FALSE]
    if (length(idx) < ncol(x) + 1) {
      list(beta = rep(0, ncol(x)), sigma = sd(ys) + 1e-4)
    } else {
      beta <- tryCatch(coef(lm.fit(xs, ys)), error = function(e) rep(0, ncol(x)))
      resid <- ys - xs %*% beta
      list(beta = beta, sigma = max(sd(resid), 1e-4))
    }
  })

  # Equal transition matrix and ergodic probabilities
  trans <- matrix(1 / n_states, n_states, n_states)
  pi0   <- rep(1 / n_states, n_states)

  list(state_params = params, trans = trans, pi0 = pi0)
}

#' Compute state-conditional densities N(y | x*beta_s, sigma_s)
#' Returns T x K matrix
ms_emission_densities <- function(y, x, state_params) {
  T <- length(y)
  K <- length(state_params)
  dens <- matrix(0, T, K)
  for (s in seq_len(K)) {
    mu_s <- x %*% state_params[[s]]$beta
    sig  <- state_params[[s]]$sigma
    dens[, s] <- dnorm(y, mean = mu_s, sd = sig)
  }
  dens[dens < 1e-300] <- 1e-300
  dens
}

#' Hamilton filter (forward pass) for Markov Switching model
#' @param dens  T x K emission density matrix
#' @param trans K x K transition matrix (row = from, col = to)
#' @param pi0   K-vector of initial state probabilities
#' @return list with xi (T x K filtered probs) and log-likelihood
hamilton_filter <- function(dens, trans, pi0) {
  T <- nrow(dens)
  K <- ncol(dens)

  xi      <- matrix(0, T, K)
  xi_pred <- matrix(0, T, K)
  log_lik <- 0

  xi_pred[1, ] <- pi0
  eta     <- xi_pred[1, ] * dens[1, ]
  c1      <- sum(eta)
  log_lik <- log_lik + log(c1)
  xi[1, ] <- eta / c1

  for (t in 2:T) {
    xi_pred[t, ] <- as.vector(t(trans) %*% xi[t - 1, ])
    eta     <- xi_pred[t, ] * dens[t, ]
    ct      <- sum(eta)
    log_lik <- log_lik + log(ct)
    xi[t, ] <- eta / ct
  }

  list(xi = xi, xi_pred = xi_pred, log_lik = log_lik)
}

#' Kim smoother (backward pass) for Markov Switching model
#' @param xi      T x K filtered probabilities from hamilton_filter
#' @param xi_pred T x K predicted probabilities from hamilton_filter
#' @param trans   K x K transition matrix
#' @return T x K smoothed state probabilities
kim_smoother <- function(xi, xi_pred, trans) {
  T <- nrow(xi)
  K <- ncol(xi)

  xi_smooth <- matrix(0, T, K)
  xi_smooth[T, ] <- xi[T, ]

  for (t in (T - 1):1) {
    for (j in seq_len(K)) {
      denom <- xi_pred[t + 1, j]
      if (denom < 1e-300) denom <- 1e-300
      xi_smooth[t, j] <- sum(
        xi_smooth[t + 1, ] * trans[j, ] * xi[t, j] / denom
      )
    }
    s <- sum(xi_smooth[t, ])
    if (s > 0) xi_smooth[t, ] <- xi_smooth[t, ] / s
  }
  xi_smooth
}

#' EM M-step: update parameters given smoothed state probabilities
ms_m_step <- function(y, x, xi_smooth, xi_smooth_joint,
                      state_params, trans, n_states) {
  T <- length(y)
  K <- n_states

  # Update regression coefficients and sigma per state
  new_params <- lapply(seq_len(K), function(s) {
    w    <- xi_smooth[, s]
    w_sq <- sqrt(w)
    yw   <- y * w_sq
    xw   <- x * w_sq
    beta <- tryCatch(
      coef(lm.wfit(x, y, w = w)),
      error = function(e) state_params[[s]]$beta
    )
    resid <- y - x %*% beta
    sigma <- sqrt(sum(w * resid^2) / sum(w))
    sigma <- max(sigma, 1e-4)
    list(beta = beta, sigma = sigma)
  })

  # Update transition matrix -- joint smoothed probs xi_smooth_joint is T x K x K
  # P(i,j) = sum_t xi_smooth_joint[t,i,j] / sum_t xi_smooth[t-1,i]
  new_trans <- matrix(0, K, K)
  for (i in seq_len(K)) {
    for (j in seq_len(K)) {
      new_trans[i, j] <- sum(xi_smooth_joint[, i, j])
    }
    row_sum <- sum(new_trans[i, ])
    if (row_sum > 0) new_trans[i, ] <- new_trans[i, ] / row_sum
    else             new_trans[i, ] <- rep(1 / K, K)
  }

  new_pi0 <- xi_smooth[1, ]

  list(state_params = new_params, trans = new_trans, pi0 = new_pi0)
}

#' Compute joint smoothed probabilities P(S_t=i, S_{t+1}=j | data)
#' Returns array of dim (T-1) x K x K
compute_xi_joint <- function(xi, xi_pred, xi_smooth, dens, trans) {
  T <- nrow(xi)
  K <- ncol(xi)

  joint <- array(0, dim = c(T - 1, K, K))
  for (t in seq_len(T - 1)) {
    denom <- xi_pred[t + 1, ]
    for (i in seq_len(K)) {
      for (j in seq_len(K)) {
        d <- denom[j]
        if (d < 1e-300) d <- 1e-300
        joint[t, i, j] <- xi_smooth[t + 1, j] * trans[i, j] *
          xi[t, i] * dens[t + 1, j] / d
      }
    }
    s <- sum(joint[t, , ])
    if (s > 0) joint[t, , ] <- joint[t, , ] / s
  }
  joint
}

#' Fit a Markov Switching regression via EM algorithm
#' @param y        dependent variable (numeric vector, length T)
#' @param x        regressors (matrix T x p; include column of 1s for intercept)
#' @param n_states number of regimes K (default 2)
#' @param max_iter maximum EM iterations
#' @param tol      log-likelihood convergence tolerance
#' @return list with smoothed_probs, state_params, trans, log_lik_history
fit_ms_regression <- function(y, x, n_states = 2L, max_iter = 200L, tol = 1e-6) {
  if (!is.matrix(x)) x <- as.matrix(x)
  stopifnot(nrow(x) == length(y))

  params   <- ms_init_params(y, x, n_states)
  ll_prev  <- -Inf
  ll_hist  <- numeric(max_iter)

  for (iter in seq_len(max_iter)) {
    dens   <- ms_emission_densities(y, x, params$state_params)
    filt   <- hamilton_filter(dens, params$trans, params$pi0)
    smooth <- kim_smoother(filt$xi, filt$xi_pred, params$trans)
    joint  <- compute_xi_joint(filt$xi, filt$xi_pred, smooth, dens, params$trans)

    ll_curr     <- filt$log_lik
    ll_hist[iter] <- ll_curr

    params <- ms_m_step(y, x, smooth, joint, params$state_params,
                        params$trans, n_states)

    if (abs(ll_curr - ll_prev) < tol) {
      ll_hist <- ll_hist[seq_len(iter)]
      break
    }
    ll_prev <- ll_curr
  }

  # Label states by mean return (ascending)
  state_means <- sapply(params$state_params, function(p) mean(p$beta))
  state_order <- order(state_means)
  state_labels <- c("bear", "neutral", "bull")[seq_len(n_states)]

  list(
    smoothed_probs  = smooth,
    state_params    = params$state_params[state_order],
    state_labels    = state_labels,
    trans           = params$trans[state_order, state_order],
    pi0             = params$pi0,
    log_lik         = ll_prev,
    ll_history      = ll_hist,
    n_states        = n_states
  )
}

# ===========================================================================
# 2. Hidden Markov Model with 3 states (Gaussian emissions, Baum-Welch)
# ===========================================================================

#' Fit a Gaussian HMM with n_states via Baum-Welch
#' @param y        observation sequence (numeric vector)
#' @param n_states number of hidden states (default 3)
#' @param max_iter maximum EM iterations
#' @param tol      log-likelihood convergence tolerance
fit_hmm <- function(y, n_states = 3L, max_iter = 300L, tol = 1e-6) {
  T <- length(y)

  # Initialize via k-means
  km      <- kmeans(y, centers = n_states, nstart = 10)
  mu_init <- as.vector(km$centers)[order(km$centers)]
  sd_init <- tapply(y, km$cluster, sd)
  sd_init[is.na(sd_init) | sd_init < 1e-4] <- sd(y) * 0.5

  mu_vec  <- mu_init
  sd_vec  <- as.vector(sd_init[order(tapply(y, km$cluster, mean))])
  trans   <- matrix(0.9 / (n_states - 1), n_states, n_states)
  diag(trans) <- 0.9
  pi0     <- rep(1 / n_states, n_states)

  ll_prev <- -Inf
  ll_hist <- numeric(max_iter)

  for (iter in seq_len(max_iter)) {
    # E-step: forward-backward
    B <- outer(y, seq_len(n_states), function(obs, s) dnorm(obs, mu_vec[s], sd_vec[s]))
    B[B < 1e-300] <- 1e-300

    # Forward
    alpha <- matrix(0, T, n_states)
    scale <- numeric(T)
    alpha[1, ] <- pi0 * B[1, ]
    scale[1]   <- sum(alpha[1, ])
    alpha[1, ] <- alpha[1, ] / scale[1]
    for (t in 2:T) {
      alpha[t, ] <- (alpha[t-1, ] %*% trans) * B[t, ]
      scale[t]   <- sum(alpha[t, ])
      if (scale[t] < 1e-300) scale[t] <- 1e-300
      alpha[t, ] <- alpha[t, ] / scale[t]
    }
    ll_curr     <- sum(log(scale))
    ll_hist[iter] <- ll_curr

    # Backward
    beta <- matrix(1, T, n_states)
    for (t in (T-1):1) {
      beta[t, ] <- trans %*% (B[t+1, ] * beta[t+1, ])
      s         <- max(beta[t, ])
      if (s > 0) beta[t, ] <- beta[t, ] / s
    }

    # Gamma (posterior state probabilities)
    gamma <- alpha * beta
    gamma <- gamma / rowSums(gamma)

    # Xi (joint transition probabilities)
    xi <- array(0, c(T-1, n_states, n_states))
    for (t in seq_len(T-1)) {
      num <- outer(alpha[t, ], B[t+1, ] * beta[t+1, ]) * trans
      s   <- sum(num)
      if (s > 0) xi[t, , ] <- num / s
    }

    # M-step
    pi0    <- gamma[1, ]
    for (i in seq_len(n_states)) {
      for (j in seq_len(n_states)) {
        trans[i, j] <- sum(xi[, i, j]) / sum(gamma[seq_len(T-1), i])
      }
    }
    mu_vec <- colSums(gamma * y) / colSums(gamma)
    sd_vec <- sqrt(colSums(gamma * outer(y, mu_vec, function(a, b) (a - b)^2)) /
                     colSums(gamma))
    sd_vec <- pmax(sd_vec, 1e-4)

    if (abs(ll_curr - ll_prev) < tol) {
      ll_hist <- ll_hist[seq_len(iter)]
      break
    }
    ll_prev <- ll_curr
  }

  state_order <- order(mu_vec)
  list(
    smoothed_probs = gamma[, state_order],
    mu             = mu_vec[state_order],
    sigma          = sd_vec[state_order],
    trans          = trans[state_order, state_order],
    pi0            = pi0[state_order],
    log_lik        = ll_prev,
    ll_history     = ll_hist,
    state_labels   = c("bear", "neutral", "bull")[seq_len(n_states)]
  )
}

#' Assign hard state labels from smoothed probability matrix
#' @return integer vector (1-indexed state assignment)
viterbi_decode <- function(smoothed_probs) {
  apply(smoothed_probs, 1, which.max)
}

# ===========================================================================
# 3. Regime-conditional performance statistics
# ===========================================================================

#' Compute Sharpe and other statistics conditional on regime
#' @param returns        numeric vector of period returns
#' @param state_probs    T x K smoothed state probability matrix
#' @param rf             risk-free rate per period (default 0)
#' @param ann_factor     annualization factor (252 for daily)
#' @return tibble with regime, mean_ret, vol, sharpe, skew, n_obs
regime_performance <- function(returns, state_probs, rf = 0, ann_factor = 252) {
  K      <- ncol(state_probs)
  states <- viterbi_decode(state_probs)

  map_dfr(seq_len(K), function(s) {
    idx <- which(states == s)
    if (length(idx) < 5) {
      return(tibble(regime = s, n_obs = length(idx),
                    mean_ret = NA, vol = NA, sharpe = NA, skew = NA, kurt = NA))
    }
    r    <- returns[idx]
    mean_r <- mean(r, na.rm = TRUE)
    vol_r  <- sd(r, na.rm = TRUE)
    sharpe <- if (vol_r > 0) (mean_r - rf) / vol_r * sqrt(ann_factor) else NA
    skew   <- (mean(r^3, na.rm = TRUE)) / vol_r^3
    kurt   <- (mean(r^4, na.rm = TRUE)) / vol_r^4 - 3

    tibble(regime = s, n_obs = length(idx),
           mean_ret = mean_r * ann_factor,
           vol      = vol_r * sqrt(ann_factor),
           sharpe   = sharpe,
           skew     = skew,
           kurt     = kurt)
  })
}

# ===========================================================================
# 4. Time series analysis: autocorrelation, spectral density
# ===========================================================================

#' Compute Ljung-Box test statistics up to max_lag
#' @param x       numeric time series
#' @param max_lag maximum lag to test
#' @return tibble with lag, Q_stat, p_value
ljung_box_series <- function(x, max_lag = 20L) {
  n  <- length(x)
  ac <- acf(x, lag.max = max_lag, plot = FALSE)$acf[-1]

  map_dfr(seq_len(max_lag), function(h) {
    Q <- n * (n + 2) * sum((ac[seq_len(h)]^2) / (n - seq_len(h)))
    p <- pchisq(Q, df = h, lower.tail = FALSE)
    tibble(lag = h, Q_stat = Q, p_value = p)
  })
}

#' Estimate spectral density using Welch's method (squared periodogram smoothing)
#' @param x       numeric time series
#' @param n_segments  number of Welch segments (default 8)
#' @return tibble with freq, power (one-sided)
spectral_density_welch <- function(x, n_segments = 8L) {
  n        <- length(x)
  seg_len  <- floor(n / n_segments)
  overlap  <- floor(seg_len / 2)

  starts <- seq(1, n - seg_len + 1, by = seg_len - overlap)
  starts <- starts[starts + seg_len - 1 <= n]

  # Hann window
  hann <- function(m) 0.5 * (1 - cos(2 * pi * seq(0, m - 1) / (m - 1)))
  w    <- hann(seg_len)
  w_sq_sum <- sum(w^2)

  psds <- lapply(starts, function(s) {
    seg      <- x[s:(s + seg_len - 1)] * w
    ft       <- fft(seg)
    m        <- floor(seg_len / 2) + 1
    psd      <- (Mod(ft[seq_len(m)])^2) / w_sq_sum
    psd[2:(m-1)] <- 2 * psd[2:(m-1)]  # one-sided doubling
    psd
  })

  m   <- floor(seg_len / 2) + 1
  avg_psd <- Reduce(`+`, psds) / length(psds)
  freq    <- seq(0, 0.5, length.out = m)

  tibble(freq = freq, power = avg_psd)
}

# ===========================================================================
# 5. Regime persistence statistics
# ===========================================================================

#' Compute regime persistence and transition statistics
#' @param states  integer vector of hard state assignments
#' @param n_states number of states K
#' @return list with duration_stats, transition_matrix, expected_durations
regime_persistence <- function(states, n_states = NULL) {
  if (is.null(n_states)) n_states <- max(states)

  # Compute run lengths in each state
  rle_states <- rle(states)
  durations  <- tibble(state = rle_states$values, duration = rle_states$lengths)

  duration_stats <- durations %>%
    group_by(state) %>%
    summarise(
      n_episodes  = n(),
      mean_dur    = mean(duration),
      median_dur  = median(duration),
      max_dur     = max(duration),
      min_dur     = min(duration),
      .groups     = "drop"
    )

  # Empirical transition counts
  trans_count <- matrix(0L, n_states, n_states)
  for (t in seq_len(length(states) - 1)) {
    i <- states[t]
    j <- states[t + 1]
    if (i >= 1 && i <= n_states && j >= 1 && j <= n_states)
      trans_count[i, j] <- trans_count[i, j] + 1L
  }

  trans_prob <- trans_count / pmax(rowSums(trans_count), 1)

  # Expected duration = 1 / (1 - p_ii)
  expected_dur <- sapply(seq_len(n_states), function(s) {
    p_ii <- trans_prob[s, s]
    if (p_ii < 1) 1 / (1 - p_ii) else Inf
  })

  list(
    duration_stats    = duration_stats,
    transition_counts = trans_count,
    transition_probs  = trans_prob,
    expected_durations = expected_dur
  )
}

# ===========================================================================
# 6. Visualization
# ===========================================================================

#' Plot smoothed regime probabilities over time
#' @param dates       vector of dates (Date or POSIXct)
#' @param state_probs T x K smoothed probability matrix
#' @param returns     optional returns series to overlay
plot_regime_probs <- function(dates, state_probs, returns = NULL,
                               state_labels = NULL) {
  K <- ncol(state_probs)
  if (is.null(state_labels)) state_labels <- paste("State", seq_len(K))

  df <- as_tibble(state_probs, .name_repair = "minimal")
  colnames(df) <- state_labels
  df$date <- dates

  long_df <- pivot_longer(df, -date, names_to = "state", values_to = "prob")
  colors  <- c("#F44336", "#FFC107", "#4CAF50")[seq_len(K)]

  p <- ggplot(long_df, aes(x = date, y = prob, fill = state)) +
    geom_area(alpha = 0.7) +
    scale_fill_manual(values = setNames(colors, state_labels)) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(
      title = "Regime Probabilities Over Time",
      x     = "Date",
      y     = "Smoothed Probability",
      fill  = "Regime"
    ) +
    theme_minimal(base_size = 12)

  if (!is.null(returns)) {
    ret_df <- tibble(date = dates, return = returns)
    p <- p +
      geom_line(data = ret_df, aes(x = date, y = (return - min(return)) /
                                     (max(return) - min(return))),
                inherit.aes = FALSE, color = "black", linewidth = 0.4, alpha = 0.5)
  }
  p
}
