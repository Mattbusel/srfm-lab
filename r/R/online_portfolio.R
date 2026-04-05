# =============================================================================
# online_portfolio.R
# Online Portfolio Algorithms for Crypto
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Unlike one-shot Markowitz, online portfolio algorithms
# update weights bar-by-bar, exploiting price relatives without assuming
# stationarity. Cover's Universal Portfolio achieves growth-optimal performance
# in an adversarial (worst-case) sense. OLMAR and CORN exploit mean reversion
# and pattern similarity respectively.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

#' Project onto probability simplex
project_simplex <- function(v) {
  n  <- length(v)
  u  <- sort(v, decreasing = TRUE)
  cs <- cumsum(u)
  rho <- max(which(u > (cs - 1) / seq_len(n)))
  lam <- (cs[rho] - 1) / rho
  pmax(v - lam, 0)
}

#' Sharpe ratio (annualised)
sharpe_ratio <- function(rets, ann = 252) {
  mu <- mean(rets, na.rm = TRUE)
  sg <- sd(rets, na.rm = TRUE)
  if (is.na(sg) || sg < 1e-12) return(NA_real_)
  mu / sg * sqrt(ann)
}

#' Maximum drawdown from equity curve
max_drawdown <- function(eq) {
  peak <- cummax(eq)
  min((eq - peak) / peak, na.rm = TRUE)
}

#' Performance summary
perf_summary <- function(rets, label = "Strategy") {
  eq <- cumprod(1 + rets)
  data.frame(
    label     = label,
    sharpe    = sharpe_ratio(rets),
    total_ret = tail(eq, 1) - 1,
    max_dd    = max_drawdown(eq),
    ann_vol   = sd(rets) * sqrt(252),
    hit_rate  = mean(rets > 0, na.rm = TRUE)
  )
}

# ---------------------------------------------------------------------------
# 2. DATA SIMULATION
# ---------------------------------------------------------------------------

simulate_prices <- function(T_ = 1000L, N = 5L,
                              mu_ann = 0.10, sigma_ann = 0.60,
                              rho = 0.3, seed = 42L) {
  set.seed(seed)
  mu_d    <- mu_ann / 252
  sig_d   <- sigma_ann / sqrt(252)
  # Correlated returns via Cholesky
  Sigma   <- matrix(rho * sig_d^2, N, N)
  diag(Sigma) <- sig_d^2
  L       <- t(chol(Sigma))
  prices  <- matrix(100, T_, N)
  for (t in 2:T_) {
    z       <- L %*% rnorm(N)
    prices[t, ] <- prices[t-1, ] * exp(mu_d + z)
  }
  prices
}

# ---------------------------------------------------------------------------
# 3. BUY AND HOLD BENCHMARKS
# ---------------------------------------------------------------------------

run_bah <- function(prices, weights = NULL) {
  T_  <- nrow(prices); N <- ncol(prices)
  if (is.null(weights)) weights <- rep(1/N, N)
  # Initial shares, never rebalance
  initial_value <- sum(weights)
  shares <- weights / prices[1, ]
  equity <- numeric(T_)
  for (t in seq_len(T_)) equity[t] <- sum(shares * prices[t, ])
  rets   <- c(NA, diff(log(equity)))
  list(equity = equity, returns = rets[-1],
       weights_final = shares * prices[T_, ] / equity[T_])
}

run_equal_weight <- function(prices) run_bah(prices, rep(1/ncol(prices), ncol(prices)))

# ---------------------------------------------------------------------------
# 4. CONSTANT REBALANCING PORTFOLIO (CRP)
# ---------------------------------------------------------------------------

run_crp <- function(prices, weights = NULL, tc = 0.001) {
  T_  <- nrow(prices); N <- ncol(prices)
  if (is.null(weights)) weights <- rep(1/N, N)
  equity <- numeric(T_); equity[1] <- 1.0
  w_cur  <- weights
  for (t in 2:T_) {
    pr   <- prices[t, ] / prices[t-1, ]
    ret  <- sum(w_cur * (pr - 1))
    # Transaction cost for rebalancing back to target
    w_drifted <- w_cur * pr / sum(w_cur * pr)
    turnover  <- sum(abs(weights - w_drifted)) / 2
    equity[t] <- equity[t-1] * (1 + ret - tc * turnover)
    w_cur     <- weights
  }
  rets <- c(NA, diff(log(equity)))
  list(equity = equity, returns = rets[-1], weights = weights)
}

# ---------------------------------------------------------------------------
# 5. BEST CONSTANT REBALANCED PORTFOLIO (BCRP)
# ---------------------------------------------------------------------------
# Grid-search on the simplex for the CRP maximising terminal wealth.

find_bcrp <- function(prices, n_grid = 100L, tc = 0.001) {
  N   <- ncol(prices)
  best_tw  <- -Inf
  best_w   <- rep(1/N, N)

  # Latin hypercube-style simplex sampling
  set.seed(42)
  raw <- matrix(rexp(n_grid * N), n_grid, N)
  candidates <- raw / rowSums(raw)

  for (i in seq_len(n_grid)) {
    w  <- candidates[i, ]
    eq <- run_crp(prices, w, tc)$equity
    tw <- tail(eq, 1)
    if (tw > best_tw) { best_tw <- tw; best_w <- w }
  }
  list(weights = best_w, terminal_wealth = best_tw,
       result  = run_crp(prices, best_w, tc))
}

# ---------------------------------------------------------------------------
# 6. UNIVERSAL PORTFOLIO (Cover 1991)
# ---------------------------------------------------------------------------
# Wealth-weighted mixture of all CRPs on the simplex.
# Approximated via a discrete set of portfolio atoms.

up_init <- function(N, n_atoms = 200L, seed = 1L) {
  set.seed(seed)
  raw   <- matrix(rexp(n_atoms * N), n_atoms, N)
  atoms <- raw / rowSums(raw)
  list(atoms  = atoms,
       wealth = rep(1.0, n_atoms),
       N      = N,
       n_atoms = n_atoms)
}

up_step <- function(state, price_relatives) {
  # Growth of each atom = dot(atom, pr)
  growths        <- state$atoms %*% price_relatives
  state$wealth   <- state$wealth * pmax(growths, 1e-12)
  # Portfolio weights = wealth-weighted average of atoms
  total_w <- sum(state$wealth)
  w_new   <- colSums(state$atoms * state$wealth) / total_w
  list(state = state, weights = w_new / sum(w_new))
}

run_universal_portfolio <- function(prices, n_atoms = 200L, tc = 0.001) {
  T_   <- nrow(prices); N <- ncol(prices)
  state <- up_init(N, n_atoms)
  equity  <- numeric(T_); equity[1] <- 1.0
  w_mat   <- matrix(1/N, T_, N)
  w_prev  <- rep(1/N, N)

  for (t in 2:T_) {
    pr  <- prices[t, ] / prices[t-1, ]
    ret <- sum(w_prev * (pr - 1))
    turnover <- sum(abs(w_mat[t-1,] * pr / sum(w_mat[t-1,]*pr) - w_mat[t-1,]))
    equity[t] <- equity[t-1] * (1 + ret - tc * turnover / 2)
    res <- up_step(state, pr)
    state <- res$state
    w_mat[t, ] <- res$weights
    w_prev <- res$weights
  }
  rets <- c(NA, diff(log(equity)))
  list(equity = equity, returns = rets[-1], weights = w_mat)
}

# ---------------------------------------------------------------------------
# 7. EXPONENTIAL GRADIENT (EG)
# ---------------------------------------------------------------------------

eg_step <- function(w, price_relatives, eta = 0.05) {
  ret  <- sum(w * price_relatives)
  if (ret < 1e-12) return(w)
  g    <- price_relatives / ret
  w_new <- w * exp(eta * g)
  w_new / sum(w_new)
}

run_eg <- function(prices, eta = 0.05, tc = 0.001) {
  T_  <- nrow(prices); N <- ncol(prices)
  w   <- rep(1/N, N)
  equity <- numeric(T_); equity[1] <- 1.0
  w_mat  <- matrix(1/N, T_, N)

  for (t in 2:T_) {
    pr   <- prices[t, ] / prices[t-1, ]
    ret  <- sum(w * (pr - 1))
    drift <- w * pr / sum(w * pr)
    cost  <- sum(abs(eg_step(w, pr, eta) - drift)) * tc / 2
    equity[t] <- equity[t-1] * (1 + ret - cost)
    w         <- eg_step(w, pr, eta)
    w_mat[t,] <- w
  }
  rets <- c(NA, diff(log(equity)))
  list(equity = equity, returns = rets[-1], weights = w_mat)
}

# ---------------------------------------------------------------------------
# 8. OLMAR (Online Moving Average Reversion)
# ---------------------------------------------------------------------------
# Predict price relative via moving average; update positions via PAMR.

pamr_update <- function(w, pred_rel, C = 500, eps = 0.5) {
  n <- length(w)
  pred_norm <- pred_rel / mean(pred_rel)
  port_pred <- sum(w * pred_norm)
  loss <- max(0, port_pred - eps)
  tau  <- loss / (sum((pred_norm - mean(pred_norm))^2) + 1 / (2*C))
  w_new <- w - tau * (pred_norm - mean(pred_norm))
  project_simplex(w_new)
}

run_olmar <- function(prices, window = 5L, C = 500, tc = 0.001) {
  T_  <- nrow(prices); N <- ncol(prices)
  w   <- rep(1/N, N)
  equity <- numeric(T_); equity[1] <- 1.0
  w_mat  <- matrix(1/N, T_, N)

  for (t in (window + 1L):T_) {
    pr    <- prices[t, ] / prices[t-1, ]
    ret   <- sum(w * (pr - 1))
    # Predict via MA
    ma_pred <- apply(prices[(t-window):t, , drop=FALSE], 2, mean) / prices[t, ]
    w_new   <- pamr_update(w, ma_pred, C, eps = 1.0)
    cost    <- sum(abs(w_new - w)) * tc / 2
    equity[t] <- equity[t-1] * (1 + ret - cost)
    w <- w_new; w_mat[t, ] <- w
  }
  rets <- c(NA, diff(log(equity)))
  list(equity = equity, returns = rets[-1], weights = w_mat)
}

# ---------------------------------------------------------------------------
# 9. PAMR (Passive Aggressive Mean Reversion)
# ---------------------------------------------------------------------------

run_pamr <- function(prices, C = 500, eps = 0.5, tc = 0.001) {
  T_  <- nrow(prices); N <- ncol(prices)
  w   <- rep(1/N, N)
  equity <- numeric(T_); equity[1] <- 1.0
  w_mat  <- matrix(1/N, T_, N)

  for (t in 2:T_) {
    pr  <- prices[t, ] / prices[t-1, ]
    ret <- sum(w * (pr - 1))
    w_new <- pamr_update(w, pr, C, eps)
    cost  <- sum(abs(w_new - w)) * tc / 2
    equity[t] <- equity[t-1] * (1 + ret - cost)
    w <- w_new; w_mat[t, ] <- w
  }
  rets <- c(NA, diff(log(equity)))
  list(equity = equity, returns = rets[-1], weights = w_mat)
}

# ---------------------------------------------------------------------------
# 10. CORN (Correlation-driven Nonparametric Learning)
# ---------------------------------------------------------------------------
# CORN selects similar historical windows and averages their successor returns.
# Financial intuition: if today's price pattern matches historical patterns,
# weight assets to exploit what those patterns were followed by.

corn_similarity <- function(x1, x2) {
  # Correlation-based similarity
  x1c <- x1 - mean(x1); x2c <- x2 - mean(x2)
  n1  <- sqrt(sum(x1c^2)); n2 <- sqrt(sum(x2c^2))
  if (n1 < 1e-12 || n2 < 1e-12) return(0)
  sum(x1c * x2c) / (n1 * n2)
}

run_corn <- function(prices, window = 5L, rho_threshold = 0.5,
                      tc = 0.001) {
  T_  <- nrow(prices); N <- ncol(prices)
  # Relative prices within window
  rel_prices <- function(t) {
    prices[(t - window + 1):t, ] / prices[t - window, ]
  }
  w   <- rep(1/N, N)
  equity <- numeric(T_); equity[1] <- 1.0
  w_mat  <- matrix(1/N, T_, N)
  start  <- 2 * window + 1L

  for (t in start:T_) {
    pr  <- prices[t, ] / prices[t-1, ]
    ret <- sum(w * (pr - 1))

    # Current window pattern
    cur_pat  <- as.vector(rel_prices(t))
    # Search historical windows
    expert_rets <- list()
    for (s in (window + 1):(t - 1)) {
      hist_pat <- as.vector(rel_prices(s))
      sim      <- corn_similarity(cur_pat, hist_pat)
      if (sim > rho_threshold) {
        next_pr <- prices[s + 1, ] / prices[s, ]
        expert_rets[[length(expert_rets) + 1]] <- next_pr
      }
    }
    if (length(expert_rets) > 0) {
      # Average the successor price relatives
      avg_pr <- Reduce("+", expert_rets) / length(expert_rets)
      # Best response to predicted relatives
      w_new  <- project_simplex(avg_pr / sum(avg_pr) - 1/N + w)
    } else {
      w_new <- w
    }
    cost <- sum(abs(w_new - w)) * tc / 2
    equity[t] <- equity[t-1] * (1 + ret - cost)
    w <- w_new; w_mat[t, ] <- w
  }
  rets <- c(NA, diff(log(equity)))
  list(equity = equity, returns = rets[-1], weights = w_mat)
}

# ---------------------------------------------------------------------------
# 11. TRACKING REGRET
# ---------------------------------------------------------------------------

tracking_regret <- function(online_equity, bcrp_equity) {
  n  <- min(length(online_equity), length(bcrp_equity))
  log_reg <- log(pmax(bcrp_equity[1:n], 1e-12)) -
    log(pmax(online_equity[1:n], 1e-12))
  list(cumulative = log_reg,
       final      = tail(log_reg, 1),
       per_period = c(0, diff(log_reg)))
}

# ---------------------------------------------------------------------------
# 12. TURNOVER ANALYSIS
# ---------------------------------------------------------------------------

compute_turnover <- function(weights_matrix) {
  T_  <- nrow(weights_matrix)
  to  <- numeric(T_)
  for (t in 2:T_) {
    to[t] <- sum(abs(weights_matrix[t, ] - weights_matrix[t-1, ])) / 2
  }
  list(turnover = to,
       mean_to  = mean(to[-1]),
       ann_to   = mean(to[-1]) * 252)
}

# ---------------------------------------------------------------------------
# 13. PERFORMANCE COMPARISON TABLE
# ---------------------------------------------------------------------------

compare_strategies <- function(strategy_list) {
  do.call(rbind, lapply(names(strategy_list), function(nm) {
    s   <- strategy_list[[nm]]
    ret <- s$returns[!is.na(s$returns)]
    perf_summary(ret, nm)
  }))
}

# ---------------------------------------------------------------------------
# 14. ROLLING PERFORMANCE WINDOWS
# ---------------------------------------------------------------------------

rolling_perf <- function(rets, window = 126L) {
  n   <- length(rets)
  out <- data.frame(t = seq_len(n),
                    roll_sharpe = NA_real_,
                    roll_ret    = NA_real_,
                    roll_vol    = NA_real_)
  for (i in window:n) {
    rw <- rets[(i - window + 1):i]
    rw <- rw[!is.na(rw)]
    out$roll_sharpe[i] <- sharpe_ratio(rw)
    out$roll_ret[i]    <- prod(1 + rw) - 1
    out$roll_vol[i]    <- sd(rw) * sqrt(252)
  }
  out
}

# ---------------------------------------------------------------------------
# 15. MAIN DEMO
# ---------------------------------------------------------------------------

run_online_portfolio_demo <- function() {
  cat("=== Online Portfolio Algorithms Demo ===\n\n")

  # Simulate prices
  prices <- simulate_prices(T_ = 500L, N = 4L, seed = 42L)
  cat(sprintf("Price matrix: %d bars x %d assets\n", nrow(prices), ncol(prices)))
  for (i in seq_len(ncol(prices)))
    cat(sprintf("  Asset %d: %.0f -> %.0f (%.1f%% total)\n",
                i, prices[1,i], tail(prices,1)[i],
                (tail(prices,1)[i]/prices[1,i]-1)*100))

  # Run all strategies
  cat("\nRunning strategies...\n")
  bah_res  <- run_bah(prices)
  crp_res  <- run_crp(prices, tc = 0.001)
  eg_res   <- run_eg(prices, eta = 0.1)
  up_res   <- run_universal_portfolio(prices, n_atoms = 100L)
  pamr_res <- run_pamr(prices, C = 500)
  olmar_res <- run_olmar(prices, window = 5L)
  corn_res <- run_corn(prices, window = 5L, rho_threshold = 0.4)

  cat("Finding BCRP (oracle)...\n")
  bcrp_res <- find_bcrp(prices, n_grid = 50L)

  strategies <- list(
    "Buy-and-Hold"   = bah_res,
    "CRP (EW)"       = crp_res,
    "EG"             = eg_res,
    "Universal"      = up_res,
    "PAMR"           = pamr_res,
    "OLMAR"          = olmar_res,
    "CORN"           = corn_res,
    "BCRP (oracle)"  = bcrp_res$result
  )

  cat("\n--- Strategy Comparison ---\n")
  cmp <- compare_strategies(strategies)
  print(cmp)

  cat("\n--- Regret vs BCRP ---\n")
  for (nm in c("Universal","EG","PAMR","OLMAR")) {
    reg <- tracking_regret(strategies[[nm]]$equity, bcrp_res$result$equity)
    cat(sprintf("  %-12s final log-regret: %.4f\n", nm, reg$final))
  }

  cat("\n--- Turnover Analysis ---\n")
  for (nm in c("EG","PAMR","OLMAR","CORN")) {
    to <- compute_turnover(strategies[[nm]]$weights)
    cat(sprintf("  %-8s mean turnover: %.3f  (ann: %.1f)\n",
                nm, to$mean_to, to$ann_to))
  }

  cat("\n--- BCRP Optimal Weights ---\n")
  cat("  Weights:", round(bcrp_res$weights, 3), "\n")
  cat("  Terminal wealth:", round(bcrp_res$terminal_wealth, 4), "\n")

  cat("\n--- Rolling Sharpe (Universal Portfolio) ---\n")
  rp <- rolling_perf(up_res$returns[!is.na(up_res$returns)], window = 60L)
  valid <- !is.na(rp$roll_sharpe)
  cat(sprintf("  Mean rolling Sharpe: %.3f  |  Min: %.3f  Max: %.3f\n",
              mean(rp$roll_sharpe[valid]),
              min(rp$roll_sharpe[valid]),
              max(rp$roll_sharpe[valid])))

  cat("\nDone.\n")
  invisible(strategies)
}

if (interactive()) {
  op_results <- run_online_portfolio_demo()
}

# ---------------------------------------------------------------------------
# 16. ANTI-CORRELATION PORTFOLIO (AGAINST MARKET PORTFOLIO)
# ---------------------------------------------------------------------------
# Find the portfolio maximally anti-correlated with BTC as hedge overlay.

anti_correlation_portfolio <- function(prices, market_col = 1L) {
  T_  <- nrow(prices); N <- ncol(prices)
  rets <- apply(prices, 2, function(p) c(NA, diff(log(p))))
  rets <- rets[-1, ]
  mkt  <- rets[, market_col]
  cors  <- apply(rets, 2, function(r) cor(r, mkt, use="complete.obs"))
  w    <- pmax(-cors, 0); if (sum(w) < 1e-8) w <- rep(1/N,N)
  w    <- w / sum(w)
  port_rets <- rets %*% w
  list(weights = w, returns = port_rets,
       equity  = cumprod(1 + port_rets),
       corr_with_market = cor(port_rets, mkt, use="complete.obs"))
}

# ---------------------------------------------------------------------------
# 17. ONLINE PORTFOLIO WITH STOP-LOSS
# ---------------------------------------------------------------------------
# EGP with a trailing stop-loss: if equity drops > max_dd_stop from peak,
# flatten to cash until equity recovers.

run_egp_stoploss <- function(prices, eta = 0.05, max_dd_stop = 0.15,
                              recovery_threshold = 0.05, tc = 0.001) {
  T_  <- nrow(prices); N <- ncol(prices)
  w   <- rep(1/N, N); equity <- numeric(T_); equity[1] <- 1.0
  w_mat <- matrix(1/N, T_, N)
  peak_eq <- 1.0; in_cash <- FALSE

  for (t in 2:T_) {
    pr  <- prices[t,] / prices[t-1,]
    if (in_cash) {
      equity[t] <- equity[t-1]   # cash earns nothing
      # Re-enter when equity recovers
      if (equity[t] >= peak_eq * (1 - max_dd_stop + recovery_threshold)) {
        in_cash <- FALSE; w <- rep(1/N, N)
      }
    } else {
      ret <- sum(w * (pr - 1))
      equity[t] <- equity[t-1] * (1 + ret - tc*sum(abs(eg_step(w,pr,eta)-w))/2)
      w   <- eg_step(w, pr, eta)
      peak_eq <- max(peak_eq, equity[t])
      if ((equity[t] - peak_eq) / peak_eq < -max_dd_stop) {
        in_cash <- TRUE
        w <- rep(0, N)   # go to cash
      }
    }
    w_mat[t,] <- w
  }
  rets <- c(NA, diff(log(pmax(equity, 1e-12))))
  list(equity=equity, returns=rets[-1], weights=w_mat)
}

# ---------------------------------------------------------------------------
# 18. PORTFOLIO PERFORMANCE DECOMPOSITION
# ---------------------------------------------------------------------------

decompose_portfolio_perf <- function(rets, benchmark_rets) {
  n   <- min(length(rets), length(benchmark_rets))
  r   <- rets[1:n]; rb <- benchmark_rets[1:n]
  active_ret <- r - rb
  beta <- cov(r,rb,use="complete.obs") / var(rb, na.rm=TRUE)
  alpha <- mean(r,na.rm=TRUE) - beta * mean(rb,na.rm=TRUE)
  te   <- sd(active_ret, na.rm=TRUE) * sqrt(252)
  ir   <- mean(active_ret,na.rm=TRUE)*252 / max(te, 1e-8)
  data.frame(alpha_ann=alpha*252, beta=beta, tracking_error=te,
             info_ratio=ir,
             up_capture=mean(r[rb>0],na.rm=TRUE)/mean(rb[rb>0],na.rm=TRUE),
             down_capture=mean(r[rb<0],na.rm=TRUE)/mean(rb[rb<0],na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 19. REBALANCING CALENDAR EFFECT
# ---------------------------------------------------------------------------
# Does the day-of-week matter for CRP rebalancing? Test each weekday.

rebalancing_dow_effect <- function(prices, target_weights = NULL, tc = 0.001) {
  T_  <- nrow(prices); N <- ncol(prices)
  if (is.null(target_weights)) target_weights <- rep(1/N, N)
  dow <- (seq_len(T_) - 1L) %% 5L   # 0=Mon,...,4=Fri
  results <- lapply(0:4, function(d) {
    W_eff <- matrix(target_weights, T_, N, byrow=TRUE)
    port_rets <- numeric(T_)
    w <- target_weights
    for (t in 2:T_) {
      pr  <- 1 + apply(prices[(t-1):t,, drop=FALSE], 2, function(p) p[2]/p[1]-1)
      port_rets[t] <- sum(w*(pr-1))
      if (dow[t] == d) { cost <- sum(abs(target_weights - w * pr/sum(w*pr)))*tc/2; port_rets[t] <- port_rets[t] - cost; w <- target_weights }
      else w <- w*pr/sum(w*pr)
    }
    data.frame(dow=d, sharpe=sharpe_ratio(port_rets[-1]),
               total_ret=prod(1+port_rets)-1)
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 20. EXTENDED ONLINE PORTFOLIO DEMO
# ---------------------------------------------------------------------------

run_online_portfolio_extended_demo <- function() {
  cat("=== Online Portfolio Extended Demo ===\n\n")
  prices <- simulate_prices(T_=500L, N=4L, seed=77L)

  cat("--- Anti-Correlation Portfolio ---\n")
  acp <- anti_correlation_portfolio(prices)
  cat(sprintf("  Corr with market: %.4f  Sharpe: %.3f\n",
              acp$corr_with_market, sharpe_ratio(acp$returns)))

  cat("\n--- EGP with Stop-Loss ---\n")
  esl <- run_egp_stoploss(prices, eta=0.1, max_dd_stop=0.12)
  cat(sprintf("  Sharpe=%.3f  MaxDD=%.1f%%  Terminal=%.3f\n",
              sharpe_ratio(esl$returns), max_drawdown(esl$equity)*100,
              tail(esl$equity,1)))

  cat("\n--- Decomposition vs Benchmark ---\n")
  bh  <- run_bah(prices)
  crp <- run_crp(prices)
  dec <- decompose_portfolio_perf(crp$returns, bh$returns)
  print(dec)

  cat("\n--- Rebalancing DOW Effect ---\n")
  dow_res <- rebalancing_dow_effect(prices)
  print(dow_res)

  invisible(list(acp=acp, esl=esl, dec=dec, dow=dow_res))
}

if (interactive()) {
  op_ext <- run_online_portfolio_extended_demo()
}

# =============================================================================
# SECTION: FIXED-SHARE SWITCHING PORTFOLIO
# =============================================================================
# Fixed-share mixes the best-so-far expert with uniform weights at rate alpha.
# This creates an adaptive portfolio that can track non-stationary markets.

fixed_share_step <- function(w, r, alpha = 0.01) {
  # Update: multiply by returns then mix back toward uniform
  w_new <- w * r
  if (sum(w_new) < .Machine$double.eps) w_new <- rep(1/length(w), length(w))
  w_new <- w_new / sum(w_new)
  (1 - alpha) * w_new + alpha / length(w_new)
}

run_fixed_share <- function(R_mat, alpha = 0.01) {
  T <- nrow(R_mat); n <- ncol(R_mat)
  w <- rep(1/n, n); cum <- 1; wealth <- numeric(T)
  for (t in seq_len(T)) {
    port_ret <- sum(w * R_mat[t,])
    cum <- cum * port_ret; wealth[t] <- cum
    w <- fixed_share_step(w, R_mat[t,], alpha)
  }
  list(wealth = wealth, final = cum)
}

# =============================================================================
# SECTION: VARIABLE SHARE — decay the mixing rate over time
# =============================================================================

run_variable_share <- function(R_mat, alpha_fn = function(t) 0.1 / sqrt(t)) {
  T <- nrow(R_mat); n <- ncol(R_mat)
  w <- rep(1/n, n); cum <- 1; wealth <- numeric(T)
  for (t in seq_len(T)) {
    port_ret <- sum(w * R_mat[t,])
    cum <- cum * port_ret; wealth[t] <- cum
    w_new <- w * R_mat[t,]
    if (sum(w_new) < .Machine$double.eps) w_new <- rep(1/n, n)
    w_new <- w_new / sum(w_new)
    a <- alpha_fn(t)
    w <- (1 - a) * w_new + a / n
  }
  list(wealth = wealth, final = cum)
}

# =============================================================================
# SECTION: FTRL PORTFOLIO — Follow the Regularised Leader
# =============================================================================
# FTRL: at each step play weights proportional to softmax(eta * cumulative log-returns).
# This is equivalent to the Exponentiated Gradient algorithm in continuous limit.

run_ftrl_portfolio <- function(R_mat, eta = 1.0) {
  T <- nrow(R_mat); n <- ncol(R_mat)
  cum_log <- rep(0, n); cum <- 1; wealth <- numeric(T)
  for (t in seq_len(T)) {
    x <- eta * cum_log; x <- x - max(x)
    w <- exp(x) / sum(exp(x))
    port_ret <- sum(w * R_mat[t,])
    cum <- cum * port_ret; wealth[t] <- cum
    cum_log <- cum_log + log(pmax(R_mat[t,], 1e-10))
  }
  list(wealth = wealth, final = cum)
}

# =============================================================================
# SECTION: ANTICORRELATION MEAN-REVERSION PORTFOLIO
# =============================================================================
# Overweight assets that recently underperformed (relative to their vol).
# This is a contrarian strategy that profits from short-term mean reversion.

run_anticorr_portfolio <- function(R_mat, window = 10) {
  T <- nrow(R_mat); n <- ncol(R_mat)
  w <- rep(1/n, n); cum <- 1; wealth <- numeric(T)
  for (t in seq_len(T)) {
    port_ret <- sum(w * R_mat[t,])
    cum <- cum * port_ret; wealth[t] <- cum
    if (t >= window) {
      sub  <- R_mat[(t - window + 1):t, , drop = FALSE]
      mu   <- colMeans(sub - 1)
      sig  <- pmax(apply(sub, 2, sd), 1e-6)
      score <- -mu / sig
      score <- score - min(score) + 0.01
      w <- score / sum(score)
    }
  }
  list(wealth = wealth, final = cum)
}

# =============================================================================
# SECTION: ONLINE SHARPE TRACKING via Welford's algorithm
# =============================================================================

welford_init_s <- function() list(n = 0L, mean = 0, M2 = 0)

welford_update_s <- function(s, x) {
  s$n <- s$n + 1L; d <- x - s$mean
  s$mean <- s$mean + d / s$n
  s$M2   <- s$M2 + d * (x - s$mean); s
}

welford_sharpe <- function(s, ann = 252) {
  if (s$n < 2) return(NA_real_)
  s$mean / sqrt(s$M2 / (s$n - 1)) * sqrt(ann)
}

track_online_sharpes <- function(strat_wealth) {
  T <- length(strat_wealth[[1]])
  nms <- names(strat_wealth)
  rets <- lapply(strat_wealth, function(w) {
    r <- diff(w) / pmax(head(w,-1), 1e-10); c(0, r)
  })
  trackers <- setNames(lapply(nms, function(x) welford_init_s()), nms)
  out <- matrix(NA_real_, T, length(nms), dimnames = list(NULL, nms))
  for (t in seq_len(T)) {
    for (s in nms) {
      trackers[[s]] <- welford_update_s(trackers[[s]], rets[[s]][t])
      out[t, s]     <- welford_sharpe(trackers[[s]])
    }
  }
  out
}

# =============================================================================
# SECTION: CORN — CORrelation-driven Nonparametric learning
# =============================================================================
# CORN finds historical windows most similar to the current window and
# uses their subsequent returns to set portfolio weights.

corn_similarity <- function(v1, v2) {
  # Pearson correlation as similarity measure
  if (sd(v1) < 1e-9 || sd(v2) < 1e-9) return(0)
  cor(v1, v2)
}

run_corn <- function(R_mat, window = 5, rho = 0.1) {
  T <- nrow(R_mat); n <- ncol(R_mat)
  w <- rep(1/n, n); cum <- 1; wealth <- numeric(T)
  for (t in seq_len(T)) {
    if (t > window + 1) {
      cur <- as.vector(R_mat[(t-window):(t-1), ])
      weights_sum <- rep(0, n)
      count <- 0
      for (s in seq(window+1, t-1)) {
        hist_w <- as.vector(R_mat[(s-window):(s-1), ])
        sim <- corn_similarity(cur, hist_w)
        if (sim >= rho) {
          weights_sum <- weights_sum + R_mat[s, ]
          count <- count + 1
        }
      }
      if (count > 0) {
        ws <- weights_sum / count
        ws <- pmax(ws, 0); if (sum(ws) > 0) w <- ws / sum(ws)
      }
    }
    port_ret <- sum(w * R_mat[t,])
    cum <- cum * port_ret; wealth[t] <- cum
  }
  list(wealth = wealth, final = cum)
}

# =============================================================================
# SECTION: FULL STRATEGY BENCHMARK
# =============================================================================

benchmark_all_online <- function(R_mat) {
  strats <- list(
    BAH    = run_bah(R_mat)$wealth,
    CRP    = run_crp(R_mat)$wealth,
    EG     = run_eg(R_mat)$wealth,
    OLMAR  = run_olmar(R_mat)$wealth,
    PAMR   = run_pamr(R_mat)$wealth,
    FS     = run_fixed_share(R_mat)$wealth,
    FTRL   = run_ftrl_portfolio(R_mat)$wealth,
    ANTI   = run_anticorr_portfolio(R_mat)$wealth,
    CORN   = run_corn(R_mat)$wealth
  )
  finals <- sapply(strats, tail, 1)
  cat("=== Online Portfolio Benchmark ===\n")
  for (s in names(finals))
    cat(sprintf("  %-8s  final wealth: %.4f\n", s, finals[s]))

  cat("\n--- Online Sharpe (annualised) ---\n")
  sh <- track_online_sharpes(strats)
  print(round(tail(sh, 1), 3))
  invisible(strats)
}

run_online_portfolio_final_demo <- function() {
  set.seed(77)
  n <- 5; T <- 500
  R_mat <- matrix(exp(matrix(rnorm(T*n, 3e-4, 0.02), T, n)), T, n)
  colnames(R_mat) <- paste0("A", seq_len(n))

  cat("Fixed-Share final wealth:",
      round(run_fixed_share(R_mat)$final, 4), "\n")
  cat("Variable-Share final wealth:",
      round(run_variable_share(R_mat)$final, 4), "\n")
  cat("FTRL final wealth:",
      round(run_ftrl_portfolio(R_mat)$final, 4), "\n")
  cat("Anticorr final wealth:",
      round(run_anticorr_portfolio(R_mat)$final, 4), "\n")
  cat("CORN final wealth:",
      round(run_corn(R_mat)$final, 4), "\n")

  invisible(benchmark_all_online(R_mat))
}

if (interactive()) {
  op_final <- run_online_portfolio_final_demo()
}

# =============================================================================
# SECTION: PORTFOLIO REGRET ANALYSIS
# =============================================================================
# Regret = best-in-hindsight wealth - achieved wealth (log scale).
# Sublinear regret guarantees algorithmic competitiveness.

compute_regret <- function(strategy_wealth, best_asset_wealth) {
  log(best_asset_wealth) - log(strategy_wealth)
}

summarise_regret <- function(R_mat, strategy_wealth) {
  # best single asset in hindsight
  asset_wealths <- apply(R_mat, 2, function(r) cumprod(r))
  best <- apply(asset_wealths, 1, max)
  regret <- compute_regret(strategy_wealth, best)
  list(
    terminal_regret    = tail(regret, 1),
    cumulative_regret  = sum(pmax(regret, 0)),
    max_regret         = max(regret)
  )
}

# =============================================================================
# SECTION: ADAPTIVE LEARNING RATE SCHEDULE
# =============================================================================
# Choosing the learning rate eta adaptively improves convergence.
# Doubling trick: double eta each time the iteration count doubles.

doubling_trick_eta <- function(t, eta0 = 0.5) {
  # eta0 / sqrt(2^floor(log2(t)))
  if (t <= 0) return(eta0)
  eta0 / sqrt(2^floor(log2(max(t, 1))))
}

run_eg_adaptive <- function(R_mat, eta0 = 0.5) {
  T <- nrow(R_mat); n <- ncol(R_mat)
  w <- rep(1/n, n); cum <- 1; wealth <- numeric(T)
  for (t in seq_len(T)) {
    eta <- doubling_trick_eta(t, eta0)
    port_ret <- sum(w * R_mat[t,])
    cum <- cum * port_ret; wealth[t] <- cum
    # EG update with adaptive eta
    log_w <- log(pmax(w, 1e-300)) + eta * (R_mat[t,] / (port_ret + 1e-10))
    log_w <- log_w - max(log_w)
    w <- exp(log_w) / sum(exp(log_w))
  }
  list(wealth = wealth, final = cum)
}

# =============================================================================
# SECTION: STRATEGY ENSEMBLE VIA PREDICTION WITH EXPERT ADVICE
# =============================================================================
# Aggregate multiple portfolio strategies using exponential weights.
# This is the meta-algorithm that wraps any set of base strategies.

expert_aggregation <- function(strategy_wealth_mat, eta = 0.1) {
  # strategy_wealth_mat: T x K (K strategies)
  T <- nrow(strategy_wealth_mat); K <- ncol(strategy_wealth_mat)
  w <- rep(1/K, K)
  ensemble_wealth <- numeric(T)
  cum <- 1
  for (t in seq_len(T)) {
    if (t == 1) {
      strat_rets <- rep(1, K)
    } else {
      strat_rets <- strategy_wealth_mat[t,] / (strategy_wealth_mat[t-1,] + 1e-10)
    }
    port_ret <- sum(w * strat_rets)
    cum <- cum * port_ret; ensemble_wealth[t] <- cum
    # Exponential update
    w <- w * exp(eta * strat_rets)
    w <- w / sum(w)
  }
  list(wealth = ensemble_wealth, final = cum)
}

run_ensemble_demo <- function(R_mat) {
  strats <- list(
    BAH  = run_bah(R_mat)$wealth,
    EG   = run_eg(R_mat)$wealth,
    FTRL = run_ftrl_portfolio(R_mat)$wealth,
    ANTI = run_anticorr_portfolio(R_mat)$wealth
  )
  wmat <- do.call(cbind, strats)
  ens  <- expert_aggregation(wmat)
  cat("Ensemble final wealth:", round(ens$final, 4), "\n")
  ens
}

if (interactive()) {
  set.seed(88)
  R_demo <- matrix(exp(matrix(rnorm(300*5, 3e-4, 0.02), 300, 5)), 300, 5)
  run_ensemble_demo(R_demo)
}
