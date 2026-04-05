# =============================================================================
# online_learning.R
# Online / Adaptive Learning Algorithms for Trading
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Markets are non-stationary. A model trained once and
# frozen quickly degrades. Online algorithms update parameters bar-by-bar,
# adapting to concept drift without expensive re-training windows. This file
# implements the core online learning primitives used in signal selection,
# portfolio construction, and drift-adaptive position sizing.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

softmax <- function(w) {
  w <- w - max(w)   # numerical stability
  e <- exp(w)
  e / sum(e)
}

#' Project w onto simplex {w >= 0, sum(w) = 1}
project_simplex <- function(v) {
  n <- length(v)
  u <- sort(v, decreasing = TRUE)
  cssv <- cumsum(u)
  rho  <- max(which(u > (cssv - 1) / seq_len(n)))
  lam  <- (cssv[rho] - 1) / rho
  pmax(v - lam, 0)
}

#' Sharpe ratio (annualised)
sharpe_ratio <- function(rets, ann = 252) {
  mu <- mean(rets, na.rm = TRUE)
  sg <- sd(rets, na.rm = TRUE)
  if (is.na(sg) || sg < 1e-12) return(NA_real_)
  mu / sg * sqrt(ann)
}

#' Maximum drawdown
max_drawdown <- function(equity) {
  peak <- cummax(equity)
  dd   <- (equity - peak) / peak
  min(dd, na.rm = TRUE)
}

# ---------------------------------------------------------------------------
# 2. SGD FAMILY: AdaGrad, RMSprop, Adam
# ---------------------------------------------------------------------------
# These are online gradient methods for minimising a streaming loss.
# In trading we minimise a prediction error (e.g., next-bar return).

#' AdaGrad optimizer state
adagrad_init <- function(d) {
  list(G = rep(0, d),   # accumulated squared gradients
       t = 0L)
}

#' AdaGrad update: returns new theta
adagrad_step <- function(state, theta, grad, eta = 0.01, eps = 1e-8) {
  state$G  <- state$G + grad^2
  state$t  <- state$t + 1L
  theta    <- theta - eta / (sqrt(state$G) + eps) * grad
  list(theta = theta, state = state)
}

#' RMSprop optimizer state
rmsprop_init <- function(d) {
  list(v = rep(0, d), t = 0L)
}

#' RMSprop update
rmsprop_step <- function(state, theta, grad, eta = 0.01, rho = 0.9, eps = 1e-8) {
  state$v <- rho * state$v + (1 - rho) * grad^2
  state$t <- state$t + 1L
  theta   <- theta - eta / (sqrt(state$v) + eps) * grad
  list(theta = theta, state = state)
}

#' Adam optimizer state
adam_init <- function(d) {
  list(m = rep(0, d), v = rep(0, d), t = 0L)
}

#' Adam update
adam_step <- function(state, theta, grad, eta = 0.001,
                       beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  state$t <- state$t + 1L
  state$m <- beta1 * state$m + (1 - beta1) * grad
  state$v <- beta2 * state$v + (1 - beta2) * grad^2
  m_hat <- state$m / (1 - beta1^state$t)
  v_hat <- state$v / (1 - beta2^state$t)
  theta <- theta - eta * m_hat / (sqrt(v_hat) + eps)
  list(theta = theta, state = state)
}

#' Online linear regression with Adam (MSE loss)
online_linear_adam <- function(X, y,
                                eta    = 0.001,
                                lambda = 0.0001,   # L2 regularisation
                                seed   = 1L) {
  set.seed(seed)
  T_    <- nrow(X)
  d     <- ncol(X)
  theta <- rnorm(d, 0, 0.01)
  opt   <- adam_init(d)

  preds  <- numeric(T_)
  losses <- numeric(T_)

  for (t in seq_len(T_)) {
    xt  <- X[t, ]
    yt  <- y[t]
    hat <- sum(theta * xt)
    preds[t]  <- hat
    losses[t] <- (hat - yt)^2
    # Gradient of MSE + L2
    grad <- 2 * (hat - yt) * xt + 2 * lambda * theta
    upd  <- adam_step(opt, theta, grad, eta = eta)
    theta <- upd$theta
    opt   <- upd$state
  }
  list(theta = theta, preds = preds, losses = losses,
       mse = mean(losses, na.rm = TRUE))
}

# ---------------------------------------------------------------------------
# 3. FTRL-PROXIMAL (Follow-the-Regularised-Leader)
# ---------------------------------------------------------------------------
# FTRL is the algorithm behind Google's ad click prediction.
# For portfolio: it efficiently handles L1+L2 sparsity on signal weights.
# w_t = argmin_{w} eta * <g_sum, w> + lambda1|w|_1 + lambda2/2 ||w||^2 + sum sigma_i w_i^2
# Closed form via soft-threshold.

ftrl_proximal_init <- function(d, alpha = 1.0, beta = 1.0) {
  list(z    = rep(0, d),   # accumulated gradient (modified)
       n    = rep(0, d),   # accumulated squared gradients
       alpha = alpha,
       beta  = beta,
       t    = 0L)
}

#' FTRL-Proximal weight computation
ftrl_get_weights <- function(state, lambda1, lambda2) {
  alpha <- state$alpha; beta <- state$beta
  z     <- state$z; n <- state$n
  sigma_t <- (sqrt(n) + beta) / alpha
  w <- -(1 / (sigma_t + lambda2)) * (z - sign(z) * lambda1)
  w[abs(z) <= lambda1] <- 0   # sparsity via L1
  w
}

#' FTRL-Proximal update step
ftrl_step <- function(state, grad, lambda1, lambda2) {
  state$t <- state$t + 1L
  n_new   <- state$n + grad^2
  sigma   <- (sqrt(n_new) - sqrt(state$n)) / state$alpha
  state$z <- state$z + grad - sigma * ftrl_get_weights(state, lambda1, lambda2)
  state$n <- n_new
  state
}

#' Online signal selection via FTRL (sparse)
ftrl_signal_selection <- function(signals_matrix, y,
                                   alpha   = 1.0,
                                   lambda1 = 0.001,
                                   lambda2 = 0.0001) {
  T_   <- nrow(signals_matrix)
  d    <- ncol(signals_matrix)
  opt  <- ftrl_proximal_init(d, alpha)
  preds  <- numeric(T_)
  w_path <- matrix(0, T_, d)

  for (t in seq_len(T_)) {
    w <- ftrl_get_weights(opt, lambda1, lambda2)
    w_path[t, ] <- w
    xt <- signals_matrix[t, ]
    hat <- sum(w * xt)
    preds[t] <- hat
    # Gradient of squared loss
    grad <- 2 * (hat - y[t]) * xt
    opt  <- ftrl_step(opt, grad, lambda1, lambda2)
  }
  list(preds = preds, w_path = w_path, final_weights = w_path[T_, ])
}

# ---------------------------------------------------------------------------
# 4. HEDGE ALGORITHM (Exponential Weights)
# ---------------------------------------------------------------------------
# Hedge maintains a probability distribution over N expert predictions.
# Expert that performs better gains weight; this is the foundation of
# ensemble learning and online portfolio theory.

hedge_init <- function(n_experts) {
  list(weights = rep(1 / n_experts, n_experts),
       log_eta = log(sqrt(2 * log(n_experts))),   # Freund-Schapire eta
       cum_loss = rep(0, n_experts))
}

#' Hedge update: losses is a vector of length n_experts for this round
hedge_step <- function(state, losses, eta = NULL) {
  if (is.null(eta)) eta <- exp(state$log_eta)
  state$cum_loss <- state$cum_loss + losses
  w <- state$weights * exp(-eta * losses)
  state$weights  <- w / sum(w)
  state
}

#' Run Hedge algorithm on a matrix of expert returns
run_hedge <- function(expert_returns, eta = 0.05) {
  # expert_returns: T x K matrix (row = time, col = expert)
  T_  <- nrow(expert_returns)
  K   <- ncol(expert_returns)
  opt <- hedge_init(K)
  opt$log_eta <- log(eta)

  portfolio_ret <- numeric(T_)
  weight_path   <- matrix(0, T_, K)

  for (t in seq_len(T_)) {
    w  <- opt$weights
    weight_path[t, ] <- w
    # Portfolio return = weighted avg expert return
    portfolio_ret[t] <- sum(w * expert_returns[t, ])
    # Loss = negative return (minimise negative return = maximise return)
    losses <- -expert_returns[t, ]
    opt    <- hedge_step(opt, losses, eta = eta)
  }
  list(portfolio_ret = portfolio_ret,
       weight_path   = weight_path,
       final_weights = opt$weights)
}

# ---------------------------------------------------------------------------
# 5. UNIVERSAL PORTFOLIO (Cover 1991)
# ---------------------------------------------------------------------------
# The Universal Portfolio achieves the same asymptotic growth rate as the
# best CONSTANT-rebalanced portfolio in hindsight, without knowing which
# one it is.  Weights are updated as the average under the wealth-weighted
# distribution over the simplex.
# Approximation via discretisation of the simplex (efficient for N <= 4).

universal_portfolio_init <- function(n_assets, grid_resolution = 10L) {
  # Generate simplex grid points
  if (n_assets == 2L) {
    steps <- seq(0, 1, length.out = grid_resolution + 1L)
    pts   <- cbind(steps, 1 - steps)
  } else if (n_assets == 3L) {
    pts <- list()
    for (a in 0:grid_resolution) {
      for (b in 0:(grid_resolution - a)) {
        c_ <- grid_resolution - a - b
        pts[[length(pts) + 1]] <- c(a, b, c_) / grid_resolution
      }
    }
    pts <- do.call(rbind, pts)
  } else {
    # Random simplex sampling for larger N
    raw <- matrix(rexp(1000 * n_assets), 1000, n_assets)
    pts <- raw / rowSums(raw)
  }
  # Initial wealth for each portfolio = 1
  list(portfolios  = pts,
       wealth      = rep(1.0, nrow(pts)),
       n_portfolios = nrow(pts))
}

#' Universal Portfolio step: update wealth and return current weights
universal_portfolio_step <- function(state, price_relatives) {
  # price_relatives: vector of length n_assets, p_t / p_{t-1}
  # Returns of each portfolio = dot(w, price_relatives)
  rets <- state$portfolios %*% price_relatives
  state$wealth <- state$wealth * pmax(rets, 1e-12)
  # New weights = wealth-weighted average of portfolios
  total_w <- sum(state$wealth)
  w_new   <- colSums(state$portfolios * state$wealth) / total_w
  list(state   = state,
       weights = w_new / sum(w_new))
}

run_universal_portfolio <- function(price_matrix) {
  # price_matrix: T x N  (each row = prices at time t)
  T_  <- nrow(price_matrix)
  N   <- ncol(price_matrix)
  opt <- universal_portfolio_init(N, grid_resolution = 8L)

  weights     <- matrix(1/N, T_, N)
  port_rets   <- numeric(T_)

  for (t in 2:T_) {
    pr  <- price_matrix[t, ] / price_matrix[t - 1, ]
    res <- universal_portfolio_step(opt, pr)
    opt <- res$state
    weights[t, ] <- res$weights
    port_rets[t] <- sum(weights[t - 1, ] * (pr - 1))
  }
  list(weights = weights, returns = port_rets,
       equity  = cumprod(1 + port_rets))
}

# ---------------------------------------------------------------------------
# 6. PAMR (Passive Aggressive Mean Reversion)
# ---------------------------------------------------------------------------
# PAMR exploits short-term mean reversion in relative prices.
# Financial intuition: if an asset moved far from equal-weight, revert.
# The "aggressive" parameter C controls how far we deviate from prev weights.

pamr_step <- function(w, price_relatives, C = 500, eps = 0.5) {
  n     <- length(w)
  x_hat <- price_relatives / mean(price_relatives)   # normalise
  port_ret <- sum(w * x_hat)
  loss  <- max(0, port_ret - eps)
  tau   <- loss / (sum((x_hat - mean(x_hat))^2) + 1 / (2 * C))
  # PAMR-1 update
  w_new <- w - tau * (x_hat - mean(x_hat))
  w_new <- project_simplex(w_new)
  list(w = w_new, loss = loss, tau = tau)
}

run_pamr <- function(price_matrix, C = 500, eps = 0.5) {
  T_  <- nrow(price_matrix); N <- ncol(price_matrix)
  w   <- rep(1/N, N)
  weights   <- matrix(0, T_, N); weights[1, ] <- w
  port_rets <- numeric(T_)

  for (t in 2:T_) {
    pr  <- price_matrix[t, ] / price_matrix[t - 1, ]
    port_rets[t] <- sum(w * (pr - 1))
    res <- pamr_step(w, pr, C, eps)
    w   <- res$w
    weights[t, ] <- w
  }
  list(weights = weights, returns = port_rets,
       equity  = cumprod(1 + port_rets))
}

# ---------------------------------------------------------------------------
# 7. OLMAR (Online Moving Average Reversion)
# ---------------------------------------------------------------------------
# Predict next price relative as moving average reversion; update via PAMR.

olmar_predict <- function(prices_window) {
  # Predicted price relative = MA / current
  n   <- length(prices_window)
  ma  <- mean(prices_window)
  pr  <- ma / prices_window[n]
  pr / mean(pr)   # normalise to simplex-compatible input
}

run_olmar <- function(price_matrix, window = 5L, C = 500) {
  T_  <- nrow(price_matrix); N <- ncol(price_matrix)
  w   <- rep(1/N, N)
  weights   <- matrix(0, T_, N); weights[1, ] <- w
  port_rets <- numeric(T_)

  for (t in (window + 1L):T_) {
    hist_pr   <- price_matrix[(t - window):t, , drop = FALSE]
    pred_rel  <- apply(hist_pr, 2, function(p) mean(p) / p[window])
    pred_rel  <- pmax(pred_rel, 1e-6)
    norm_pr   <- pred_rel / mean(pred_rel) * N   # rescale for PAMR
    port_rets[t] <- sum(w * (price_matrix[t, ] / price_matrix[t-1, ] - 1))
    res  <- pamr_step(w, norm_pr, C, eps = 1)
    w    <- res$w
    weights[t, ] <- w
  }
  list(weights = weights, returns = port_rets,
       equity  = cumprod(1 + port_rets))
}

# ---------------------------------------------------------------------------
# 8. LinUCB CONTEXTUAL BANDIT (Signal Selection)
# ---------------------------------------------------------------------------
# LinUCB selects the best trading signal (arm) at each bar given context
# (market features). The UCB term encourages exploration of uncertain signals.
# Financial intuition: different signals dominate in different regimes;
# LinUCB learns which one to trust given current market context.

linucb_init <- function(n_arms, d_context, alpha = 1.0) {
  list(
    A     = replicate(n_arms, diag(d_context), simplify = FALSE),
    b     = replicate(n_arms, rep(0, d_context), simplify = FALSE),
    alpha = alpha,
    n_arms = n_arms,
    d     = d_context
  )
}

linucb_select <- function(state, context) {
  ucb_vals <- numeric(state$n_arms)
  for (k in seq_len(state$n_arms)) {
    A_inv    <- solve(state$A[[k]] + diag(1e-8, state$d))
    theta_k  <- A_inv %*% state$b[[k]]
    mu       <- as.numeric(t(theta_k) %*% context)
    var_     <- as.numeric(t(context) %*% A_inv %*% context)
    ucb_vals[k] <- mu + state$alpha * sqrt(var_)
  }
  which.max(ucb_vals)
}

linucb_update <- function(state, arm, context, reward) {
  state$A[[arm]] <- state$A[[arm]] + outer(context, context)
  state$b[[arm]] <- state$b[[arm]] + reward * context
  state
}

run_linucb <- function(signals_matrix, context_matrix, rewards,
                        alpha = 1.0) {
  # signals_matrix: T x K (K arms, each col is signal at each bar)
  # context_matrix: T x D (D context features)
  # rewards:        T x K matrix (actual reward for each arm at each bar)
  T_      <- nrow(signals_matrix)
  K       <- ncol(signals_matrix)
  D       <- ncol(context_matrix)
  state   <- linucb_init(K, D, alpha)

  chosen  <- integer(T_)
  got_rew <- numeric(T_)

  for (t in seq_len(T_)) {
    ctx <- context_matrix[t, ]
    k   <- linucb_select(state, ctx)
    r   <- rewards[t, k]
    state    <- linucb_update(state, k, ctx, r)
    chosen[t]  <- k
    got_rew[t] <- r
  }
  list(chosen = chosen, rewards_earned = got_rew,
       cumulative_reward = cumsum(got_rew),
       state = state)
}

# ---------------------------------------------------------------------------
# 9. ADWIN (ADaptive WINdowing) Drift Detection
# ---------------------------------------------------------------------------
# ADWIN maintains a window of recent observations and detects when the
# mean has drifted by comparing sub-windows. When drift detected, old data
# is discarded. Used to trigger model resets in live trading systems.

adwin_init <- function(delta = 0.002) {
  list(window = numeric(0), delta = delta, t = 0L, n_drifts = 0L)
}

#' Add one observation; return whether drift was detected
adwin_step <- function(state, x) {
  state$window <- c(state$window, x)
  state$t <- state$t + 1L
  n <- length(state$window)
  drift_detected <- FALSE

  # Check all possible splits of window
  if (n >= 2L) {
    for (split in seq_len(n - 1L)) {
      w0 <- state$window[1:split]
      w1 <- state$window[(split + 1):n]
      n0 <- length(w0); n1 <- length(w1)
      mu0 <- mean(w0); mu1 <- mean(w1)
      # Hoeffding-like bound (assuming bounded [0,1] range)
      m_inv <- 1/n0 + 1/n1
      eps_cut <- sqrt(0.5 * m_inv * log(4 * n / state$delta))
      if (abs(mu0 - mu1) > eps_cut) {
        # Drift at 'split'; drop older portion
        state$window <- state$window[(split + 1):n]
        state$n_drifts <- state$n_drifts + 1L
        drift_detected <- TRUE
        break
      }
    }
  }
  list(state = state, drift = drift_detected,
       current_mean = mean(state$window),
       window_size  = length(state$window))
}

run_adwin <- function(series, delta = 0.002) {
  n     <- length(series)
  state <- adwin_init(delta)
  drifts <- logical(n)
  means  <- numeric(n)
  wins   <- integer(n)

  for (t in seq_len(n)) {
    res       <- adwin_step(state, series[t])
    state     <- res$state
    drifts[t] <- res$drift
    means[t]  <- res$current_mean
    wins[t]   <- res$window_size
  }
  list(drifts = drifts, means = means, window_sizes = wins,
       total_drifts = sum(drifts),
       drift_times  = which(drifts))
}

# ---------------------------------------------------------------------------
# 10. ONLINE SHARPE RATIO
# ---------------------------------------------------------------------------
# Update mean and variance incrementally using Welford's algorithm.
# Compute Sharpe without storing history.

welford_init <- function() {
  list(n = 0L, mean = 0.0, M2 = 0.0)
}

welford_step <- function(state, x) {
  state$n    <- state$n + 1L
  delta      <- x - state$mean
  state$mean <- state$mean + delta / state$n
  delta2     <- x - state$mean
  state$M2   <- state$M2 + delta * delta2
  state
}

welford_variance <- function(state) {
  if (state$n < 2L) return(NA_real_)
  state$M2 / (state$n - 1L)
}

online_sharpe <- function(state, ann = 252) {
  var_ <- welford_variance(state)
  if (is.na(var_) || var_ <= 0) return(NA_real_)
  (state$mean / sqrt(var_)) * sqrt(ann)
}

run_online_sharpe <- function(rets, ann = 252) {
  n     <- length(rets)
  state <- welford_init()
  sharpes <- numeric(n)
  for (t in seq_len(n)) {
    state      <- welford_step(state, rets[t])
    sharpes[t] <- online_sharpe(state, ann)
  }
  sharpes
}

# ---------------------------------------------------------------------------
# 11. EXPONENTIAL GRADIENT DESCENT (portfolio weights)
# ---------------------------------------------------------------------------
# EGD maintains weights on the simplex; gradient step then re-normalise.
# Equivalent to natural gradient on the simplex.

egp_step <- function(w, price_relatives, eta = 0.05) {
  n   <- length(w)
  ret <- sum(w * price_relatives)
  if (ret < 1e-12) return(w)
  # Gradient of log portfolio return
  grad <- price_relatives / ret
  w_new <- w * exp(eta * grad)
  w_new / sum(w_new)
}

run_egp <- function(price_matrix, eta = 0.05) {
  T_  <- nrow(price_matrix); N <- ncol(price_matrix)
  w   <- rep(1/N, N)
  weights   <- matrix(0, T_, N); weights[1, ] <- w
  port_rets <- numeric(T_)

  for (t in 2:T_) {
    pr <- price_matrix[t, ] / price_matrix[t-1, ]
    port_rets[t] <- sum(w * (pr - 1))
    w  <- egp_step(w, pr, eta)
    weights[t, ] <- w
  }
  list(weights = weights, returns = port_rets,
       equity  = cumprod(1 + port_rets))
}

# ---------------------------------------------------------------------------
# 12. CONSTANT REBALANCING PORTFOLIO BENCHMARK
# ---------------------------------------------------------------------------

run_crp <- function(price_matrix, weights = NULL) {
  T_  <- nrow(price_matrix); N <- ncol(price_matrix)
  if (is.null(weights)) weights <- rep(1/N, N)
  port_rets <- numeric(T_)
  for (t in 2:T_) {
    pr <- price_matrix[t, ] / price_matrix[t-1, ]
    port_rets[t] <- sum(weights * (pr - 1))
  }
  list(returns = port_rets, equity = cumprod(1 + port_rets))
}

# ---------------------------------------------------------------------------
# 13. BEST CONSTANT REBALANCED PORTFOLIO (BCRP) -- in-hindsight optimal
# ---------------------------------------------------------------------------
# BCRP is the oracle competitor: the CRP that maximises terminal wealth.
# We find it via grid search over the simplex.

bcrp_find <- function(price_matrix, n_grid = 50L) {
  N <- ncol(price_matrix)
  if (N == 2) {
    best_eq <- -Inf; best_w <- c(0.5, 0.5)
    for (a in seq(0, 1, length.out = n_grid)) {
      w  <- c(a, 1 - a)
      eq <- run_crp(price_matrix, w)$equity
      if (tail(eq, 1) > best_eq) {
        best_eq <- tail(eq, 1); best_w <- w
      }
    }
  } else {
    # Random simplex search
    set.seed(42)
    candidates <- matrix(rexp(n_grid * N), n_grid, N)
    candidates <- candidates / rowSums(candidates)
    best_eq <- -Inf; best_w <- candidates[1, ]
    for (i in seq_len(n_grid)) {
      w  <- candidates[i, ]
      eq <- run_crp(price_matrix, w)$equity
      if (tail(eq, 1) > best_eq) { best_eq <- tail(eq, 1); best_w <- w }
    }
  }
  list(weights = best_w, terminal_wealth = best_eq)
}

# ---------------------------------------------------------------------------
# 14. TRACKING REGRET
# ---------------------------------------------------------------------------
# Regret = wealth of best strategy - wealth of online algo (log scale)

compute_regret <- function(online_equity, benchmark_equity) {
  n   <- length(online_equity)
  nb  <- length(benchmark_equity)
  n   <- min(n, nb)
  log_online    <- log(pmax(online_equity[1:n], 1e-12))
  log_benchmark <- log(pmax(benchmark_equity[1:n], 1e-12))
  cumulative_regret <- log_benchmark - log_online
  list(
    cumulative = cumulative_regret,
    final      = tail(cumulative_regret, 1),
    per_period = diff(c(0, cumulative_regret))
  )
}

# ---------------------------------------------------------------------------
# 15. ONLINE PERFORMANCE REPORT
# ---------------------------------------------------------------------------

online_performance <- function(rets, label = "Strategy", ann = 252) {
  rets  <- rets[!is.na(rets)]
  equity <- cumprod(1 + rets)
  data.frame(
    label     = label,
    sharpe    = sharpe_ratio(rets, ann),
    total_ret = tail(equity, 1) - 1,
    max_dd    = max_drawdown(equity),
    ann_vol   = sd(rets) * sqrt(ann)
  )
}

# ---------------------------------------------------------------------------
# 16. MAIN DEMO
# ---------------------------------------------------------------------------

run_online_demo <- function() {
  cat("=== Online/Adaptive Learning Demo ===\n\n")
  set.seed(42)
  T_ <- 1000L; N <- 4L

  # Simulate correlated asset prices
  F1  <- cumsum(rnorm(T_, 0, 0.01))
  prices <- matrix(NA, T_, N)
  for (i in seq_len(N)) {
    prices[, i] <- 100 * exp(0.5 * F1 + cumsum(rnorm(T_, 0, 0.015)))
  }
  rets <- apply(prices, 2, function(p) c(NA, diff(log(p))))

  cat("--- 1. Online Linear Regression (Adam) ---\n")
  X_feat <- matrix(rnorm(T_ * 5), T_, 5)
  y_sig  <- 0.3 * X_feat[, 1] - 0.2 * X_feat[, 2] + rnorm(T_, 0, 0.5)
  ola <- online_linear_adam(X_feat, y_sig, eta = 0.001)
  cat(sprintf("  Final MSE: %.4f  |  Final weights: %s\n",
              ola$mse, paste(round(ola$theta, 3), collapse = ", ")))

  cat("\n--- 2. FTRL Signal Selection ---\n")
  K <- 3L
  sigs <- matrix(rnorm(T_ * K), T_, K)
  y2   <- rowMeans(sigs) + rnorm(T_, 0, 0.3)
  ftrl_res <- ftrl_signal_selection(sigs, y2, lambda1 = 0.01)
  n0 <- sum(abs(ftrl_res$final_weights) < 1e-6)
  cat(sprintf("  Sparse weights: %d / %d zero\n", n0, K))

  cat("\n--- 3. Hedge Algorithm ---\n")
  expert_rets <- rets[2:T_, ]
  expert_rets[is.na(expert_rets)] <- 0
  hedge_res <- run_hedge(expert_rets, eta = 0.1)
  cat(sprintf("  Final weights: %s\n",
              paste(round(tail(hedge_res$weight_path, 1), 3), collapse = ", ")))
  p1 <- online_performance(hedge_res$portfolio_ret, "Hedge")
  cat(sprintf("  Sharpe=%.3f  TotRet=%.1f%%\n", p1$sharpe, p1$total_ret * 100))

  cat("\n--- 4. Universal Portfolio ---\n")
  up_res <- run_universal_portfolio(prices)
  p2 <- online_performance(up_res$returns[-1], "Universal")
  cat(sprintf("  Sharpe=%.3f  TotRet=%.1f%%\n", p2$sharpe, p2$total_ret * 100))

  cat("\n--- 5. PAMR ---\n")
  pamr_res <- run_pamr(prices, C = 500, eps = 0.5)
  p3 <- online_performance(pamr_res$returns[-1], "PAMR")
  cat(sprintf("  Sharpe=%.3f  TotRet=%.1f%%\n", p3$sharpe, p3$total_ret * 100))

  cat("\n--- 6. OLMAR ---\n")
  olmar_res <- run_olmar(prices, window = 5L)
  p4 <- online_performance(olmar_res$returns[6:T_], "OLMAR")
  cat(sprintf("  Sharpe=%.3f  TotRet=%.1f%%\n", p4$sharpe, p4$total_ret * 100))

  cat("\n--- 7. EGP (Exponential Gradient) ---\n")
  egp_res <- run_egp(prices, eta = 0.1)
  p5 <- online_performance(egp_res$returns[-1], "EGP")
  cat(sprintf("  Sharpe=%.3f  TotRet=%.1f%%\n", p5$sharpe, p5$total_ret * 100))

  cat("\n--- 8. BCRP (Best Constant Rebalanced, oracle) ---\n")
  bcrp  <- bcrp_find(prices, n_grid = 30L)
  bcrp_crp <- run_crp(prices, bcrp$weights)
  p6 <- online_performance(bcrp_crp$returns[-1], "BCRP")
  cat(sprintf("  Weights: %s\n", paste(round(bcrp$weights, 3), collapse = ", ")))
  cat(sprintf("  Sharpe=%.3f  TotRet=%.1f%%\n", p6$sharpe, p6$total_ret * 100))

  cat("\n--- 9. LinUCB Signal Selection ---\n")
  D    <- 3L; K2 <- 4L
  ctx  <- matrix(rnorm(T_ * D), T_, D)
  rews <- matrix(rnorm(T_ * K2, 0, 0.02), T_, K2)
  # True arm 1 is best in high context[,1]
  rews[ctx[,1] > 0, 1] <- rews[ctx[,1] > 0, 1] + 0.02
  linucb_res <- run_linucb(matrix(0, T_, K2), ctx, rews, alpha = 0.3)
  cat(sprintf("  Arm selections: %s\n",
              paste(tabulate(linucb_res$chosen, K2), collapse = ", ")))
  cat(sprintf("  Cumulative reward: %.3f\n",
              tail(linucb_res$cumulative_reward, 1)))

  cat("\n--- 10. ADWIN Drift Detection ---\n")
  # Series with a mean shift at t=500
  drift_series <- c(rnorm(500, 0, 1), rnorm(500, 1, 1))
  adwin_res <- run_adwin(drift_series, delta = 0.002)
  cat(sprintf("  Drifts detected: %d\n", adwin_res$total_drifts))
  if (adwin_res$total_drifts > 0)
    cat(sprintf("  First drift at bar: %d (true shift at 500)\n",
                adwin_res$drift_times[1]))

  cat("\n--- 11. Online Sharpe Convergence ---\n")
  os <- run_online_sharpe(rnorm(500, 0.001, 0.02))
  cat(sprintf("  Online Sharpe at t=100: %.3f  t=500: %.3f\n",
              os[100], os[500]))

  cat("\n--- 12. Regret vs BCRP ---\n")
  reg_up   <- compute_regret(up_res$equity,  bcrp_crp$equity)
  reg_pamr <- compute_regret(pamr_res$equity, bcrp_crp$equity)
  cat(sprintf("  Universal Portfolio final regret: %.4f\n", reg_up$final))
  cat(sprintf("  PAMR final regret vs BCRP:        %.4f\n", reg_pamr$final))

  cat("\nDone.\n")
  invisible(list(hedge = hedge_res, up = up_res, pamr = pamr_res,
                 olmar = olmar_res, egp = egp_res, bcrp = bcrp,
                 linucb = linucb_res, adwin = adwin_res))
}

if (interactive()) {
  online_results <- run_online_demo()
}

# ---------------------------------------------------------------------------
# 17. ONLINE LOGISTIC REGRESSION (for directional signal classification)
# ---------------------------------------------------------------------------
# Classify next-bar direction {up=1, down=0} using a logistic model updated
# bar-by-bar via stochastic gradient descent.

sigmoid <- function(z) 1 / (1 + exp(-clip(z, -30, 30)))

online_logistic_sgd <- function(X, y_binary, eta = 0.01,
                                  lambda = 0.001, seed = 5L) {
  set.seed(seed)
  T_  <- nrow(X); d <- ncol(X)
  theta <- rnorm(d, 0, 0.01)
  preds <- numeric(T_); losses <- numeric(T_)

  for (t in seq_len(T_)) {
    p          <- sigmoid(sum(theta * X[t,]))
    preds[t]   <- p
    losses[t]  <- -(y_binary[t]*log(max(p,1e-12)) + (1-y_binary[t])*log(max(1-p,1e-12)))
    grad       <- (p - y_binary[t]) * X[t,] + 2*lambda*theta
    theta      <- theta - eta * grad
  }
  list(theta=theta, predictions=preds, losses=losses,
       accuracy=mean((preds>0.5)==y_binary, na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 18. MULTI-ARMED BANDIT: THOMPSON SAMPLING
# ---------------------------------------------------------------------------
# Thompson sampling maintains Beta(alpha,beta) posterior for each arm.
# Sample from each arm's posterior; pull arm with highest sample.
# Financial intuition: treat K signals as arms; reward = next-bar sign match.

thompson_sampling_init <- function(K) {
  list(alpha = rep(1, K), beta = rep(1, K), K = K)
}

thompson_sampling_step <- function(state, reward, chosen) {
  if (reward > 0) state$alpha[chosen] <- state$alpha[chosen] + 1
  else            state$beta[chosen]  <- state$beta[chosen]  + 1
  state
}

thompson_sampling_select <- function(state) {
  samples <- rbeta(state$K, state$alpha, state$beta)
  which.max(samples)
}

run_thompson_bandit <- function(arm_rewards, seed = 42L) {
  set.seed(seed)
  T_ <- nrow(arm_rewards); K <- ncol(arm_rewards)
  state  <- thompson_sampling_init(K)
  chosen <- integer(T_); rewards_earned <- numeric(T_)
  regrets <- numeric(T_)

  for (t in seq_len(T_)) {
    k <- thompson_sampling_select(state)
    r <- arm_rewards[t, k]
    state         <- thompson_sampling_step(state, r, k)
    chosen[t]     <- k
    rewards_earned[t] <- r
    regrets[t]    <- max(arm_rewards[t,]) - r
  }
  list(chosen=chosen, rewards=rewards_earned,
       cumulative=cumsum(rewards_earned),
       cum_regret=cumsum(regrets),
       final_alpha=state$alpha, final_beta=state$beta)
}

# ---------------------------------------------------------------------------
# 19. PASSIVE AGGRESSIVE REGRESSION (PA-I / PA-II)
# ---------------------------------------------------------------------------
# Online regression with passive-aggressive update. Like PAMR for scalars.

pa_regression_step <- function(theta, x, y, C=1.0, variant="PA-II") {
  hat  <- sum(theta * x)
  loss <- max(0, abs(hat - y) - 0.01)   # epsilon-insensitive
  if (loss < 1e-12) return(theta)
  norm_x2 <- sum(x^2)
  if (variant == "PA-I") {
    tau <- min(C, loss / (norm_x2 + 1e-12))
  } else {
    tau <- loss / (norm_x2 + 1/(2*C))
  }
  theta + tau * sign(y - hat) * x
}

run_pa_regression <- function(X, y, C=1.0, variant="PA-II") {
  T_  <- nrow(X); d <- ncol(X)
  theta <- rep(0, d); preds <- numeric(T_)
  for (t in seq_len(T_)) {
    preds[t] <- sum(theta * X[t,])
    theta    <- pa_regression_step(theta, X[t,], y[t], C, variant)
  }
  list(theta=theta, predictions=preds,
       mse=mean((preds-y)^2, na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 20. ONLINE PORTFOLIO WITH TRANSACTION COSTS (OLU)
# ---------------------------------------------------------------------------
# Online Learning with Updates (OLU): similar to EGP but incorporates
# transaction cost explicitly via a proximal step.

olu_step <- function(w, price_relatives, eta=0.05, tc=0.001) {
  n    <- length(w)
  ret  <- sum(w * price_relatives)
  if (ret < 1e-12) return(w)
  g    <- price_relatives / ret
  # Gradient step
  w_hat <- w * exp(eta * g)
  w_hat <- w_hat / sum(w_hat)
  # Proximal step for TC: shrink changes
  w_new <- w_hat - tc * sign(w_hat - w) * pmin(abs(w_hat - w), abs(w_hat - w))
  w_new <- project_simplex(w_new)
  w_new
}

run_olu <- function(prices, eta=0.05, tc=0.001) {
  T_  <- nrow(prices); N <- ncol(prices)
  w   <- rep(1/N, N); equity <- numeric(T_); equity[1] <- 1.0
  w_mat <- matrix(1/N, T_, N)
  for (t in 2:T_) {
    pr  <- prices[t,] / prices[t-1,]
    ret <- sum(w * (pr - 1))
    cost <- sum(abs(olu_step(w, pr, eta, tc) - w)) * tc / 2
    equity[t] <- equity[t-1] * (1 + ret - cost)
    w <- olu_step(w, pr, eta, tc)
    w_mat[t,] <- w
  }
  rets <- c(NA, diff(log(equity)))
  list(equity=equity, returns=rets[-1], weights=w_mat)
}

# ---------------------------------------------------------------------------
# 21. CONCEPT DRIFT SUMMARY STATISTICS
# ---------------------------------------------------------------------------

drift_summary <- function(series, window=60L) {
  n    <- length(series)
  means <- rep(NA,n); vars <- rep(NA,n)
  for (i in window:n) {
    means[i] <- mean(series[(i-window+1):i], na.rm=TRUE)
    vars[i]  <- var(series[(i-window+1):i],  na.rm=TRUE)
  }
  # Drift indicator: change in rolling mean exceeds 2 SD
  dmean  <- abs(c(NA, diff(means[!is.na(means)])))
  sd_dm  <- sd(dmean, na.rm=TRUE)
  drift_flag <- dmean > 2 * sd_dm
  list(means=means, vars=vars, drift_flags=drift_flag,
       n_drifts=sum(drift_flag, na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 22. ONLINE LEARNING EXTENDED DEMO
# ---------------------------------------------------------------------------

run_online_extended_demo <- function() {
  cat("=== Online Learning Extended Demo ===\n\n")
  set.seed(42); T_ <- 500L; N <- 4L

  prices <- matrix(100, T_, N)
  for (i in seq_len(N))
    prices[,i] <- cumprod(1 + rnorm(T_, 0.0005, 0.02)) * 100

  cat("--- 1. Online Logistic Regression ---\n")
  X_feat <- matrix(rnorm(T_*4), T_, 4)
  y_bin  <- as.integer(c(NA, diff(prices[,1])) > 0)
  ols_r  <- online_logistic_sgd(X_feat, y_bin, eta=0.01)
  cat(sprintf("  Accuracy: %.3f\n", ols_r$accuracy))

  cat("\n--- 2. Thompson Sampling Bandit ---\n")
  arm_r  <- matrix(rnorm(T_*N, 0, 0.01), T_, N)
  arm_r[, 1] <- arm_r[, 1] + 0.005   # arm 1 is best
  ts_res <- run_thompson_bandit(arm_r)
  cat(sprintf("  Arm selections: %s\n", paste(tabulate(ts_res$chosen,N), collapse=",")))
  cat(sprintf("  Cumulative regret: %.4f\n", tail(ts_res$cum_regret,1)))

  cat("\n--- 3. PA Regression ---\n")
  X_pa <- matrix(rnorm(T_*3), T_, 3)
  y_pa <- 0.5*X_pa[,1] - 0.3*X_pa[,2] + rnorm(T_,0,0.5)
  pa1  <- run_pa_regression(X_pa, y_pa, C=1.0, variant="PA-I")
  pa2  <- run_pa_regression(X_pa, y_pa, C=1.0, variant="PA-II")
  cat(sprintf("  PA-I MSE=%.4f  PA-II MSE=%.4f\n", pa1$mse, pa2$mse))

  cat("\n--- 4. OLU Portfolio ---\n")
  olu_res <- run_olu(prices, eta=0.1, tc=0.001)
  cat(sprintf("  OLU Sharpe=%.3f  Terminal equity=%.4f\n",
              sharpe_ratio(olu_res$returns), tail(olu_res$equity,1)))

  cat("\n--- 5. Concept Drift Summary ---\n")
  drift_series <- c(rnorm(250,0,1), rnorm(250,1.5,1))
  ds <- drift_summary(drift_series, window=30L)
  cat(sprintf("  Drift events detected: %d\n", ds$n_drifts))

  invisible(list(ts=ts_res, pa=pa1, olu=olu_res, ds=ds))
}

if (interactive()) {
  online_ext <- run_online_extended_demo()
}
