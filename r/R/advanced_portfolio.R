# =============================================================================
# advanced_portfolio.R
# Advanced Portfolio Construction: Maximum Diversification Portfolio (MDP),
# Minimum Correlation Portfolio, Equal Risk Contribution with CVaR, Tail Risk
# Parity, Momentum/Carry/Quality overlays, and combined overlays.
# Pure base R.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. MAXIMUM DIVERSIFICATION PORTFOLIO (MDP)
# ---------------------------------------------------------------------------
# MDP maximizes the Diversification Ratio (DR):
# DR = (w' * sigma_i) / sqrt(w' * Sigma * w)
# where sigma_i = individual volatilities, Sigma = covariance matrix
# Choueifaty & Coignard (2008)

#' Diversification ratio of a portfolio
diversification_ratio <- function(weights, cov_matrix) {
  vols        <- sqrt(diag(cov_matrix))
  weighted_vols <- sum(weights * vols)
  port_vol    <- sqrt(as.numeric(t(weights) %*% cov_matrix %*% weights))
  if (port_vol < 1e-12) return(1)
  weighted_vols / port_vol
}

#' Maximum Diversification Portfolio via gradient ascent
#' Maximizes DR subject to: sum(w) = 1, w >= lb
mdp_optimize <- function(cov_matrix, lb = 0, n_starts = 10,
                          n_iter = 500, lr = 0.01, seed = 42) {
  n <- nrow(cov_matrix)
  vols <- sqrt(diag(cov_matrix))

  best_dr <- -Inf; best_w <- rep(1/n, n)

  set.seed(seed)
  for (start in seq_len(n_starts)) {
    # Random feasible initialization
    w <- runif(n) + 0.01
    w <- pmax(w, lb)
    w <- w / sum(w)

    for (iter in seq_len(n_iter)) {
      port_vol <- sqrt(as.numeric(t(w) %*% cov_matrix %*% w))
      weighted_vol <- sum(w * vols)

      # Gradient of DR w.r.t. w
      # dDR/dw_i = (sigma_i * port_vol - weighted_vol * d(port_vol)/dw_i) / port_vol^2
      marginal_port_vol <- as.vector(cov_matrix %*% w) / port_vol
      grad_dr <- (vols * port_vol - weighted_vol * marginal_port_vol) / port_vol^2

      # Gradient ascent step
      w_new <- w + lr * grad_dr
      w_new <- pmax(w_new, lb)
      w_new <- w_new / sum(w_new)

      if (max(abs(w_new - w)) < 1e-8) break
      w <- w_new
    }

    dr <- diversification_ratio(w, cov_matrix)
    if (dr > best_dr) {
      best_dr <- dr; best_w <- w
    }
  }

  list(
    weights = best_w,
    diversification_ratio = best_dr,
    portfolio_vol = sqrt(as.numeric(t(best_w) %*% cov_matrix %*% best_w)),
    individual_vols = vols
  )
}

# ---------------------------------------------------------------------------
# 2. MINIMUM CORRELATION PORTFOLIO
# ---------------------------------------------------------------------------
# Minimizes average pairwise correlation, weighted by portfolio weights
# Equivalent to maximizing diversification in correlation space
# de Miguel et al. approach

#' Average portfolio pairwise correlation
avg_portfolio_correlation <- function(weights, cor_matrix) {
  n <- length(weights)
  corr_sum <- 0; n_pairs <- 0
  for (i in seq_len(n-1)) {
    for (j in (i+1):n) {
      corr_sum <- corr_sum + weights[i] * weights[j] * cor_matrix[i, j]
      n_pairs  <- n_pairs + weights[i] * weights[j]
    }
  }
  if (n_pairs < 1e-12) return(1)
  corr_sum / n_pairs
}

#' Minimum Correlation Portfolio via gradient descent
mincorr_optimize <- function(cor_matrix, lb = 0, n_iter = 500,
                              lr = 0.005, seed = 42) {
  n <- nrow(cor_matrix)
  set.seed(seed)
  w <- runif(n); w <- pmax(w, lb); w <- w / sum(w)

  for (iter in seq_len(n_iter)) {
    # Gradient of average pairwise correlation w.r.t. w
    grad <- numeric(n)
    for (i in seq_len(n)) {
      for (j in seq_len(n)) {
        if (i != j) grad[i] <- grad[i] + w[j] * cor_matrix[i, j]
      }
    }
    grad <- grad / n

    # Gradient descent (minimize correlation)
    w_new <- w - lr * grad
    w_new <- pmax(w_new, lb)
    w_new <- w_new / sum(w_new)

    if (max(abs(w_new - w)) < 1e-8) break
    w <- w_new
  }

  list(
    weights = w,
    avg_correlation = avg_portfolio_correlation(w, cor_matrix),
    equal_weight_corr = avg_portfolio_correlation(rep(1/n, n), cor_matrix)
  )
}

# ---------------------------------------------------------------------------
# 3. MOST DIVERSIFIED PORTFOLIO (CHOUEIFATY-COIGNARD FULL)
# ---------------------------------------------------------------------------
# Same as MDP but with analytical insight: MDP holds assets in proportion to
# their Sharpe ratio in the uncorrelated (decorrelated) space.

#' Decorrelation-based MDP (analytical solution via correlation matrix)
mdp_analytical <- function(cov_matrix) {
  n <- nrow(cov_matrix)
  vols <- sqrt(diag(cov_matrix))

  # Correlation matrix
  D <- diag(1 / vols)
  cor_matrix <- D %*% cov_matrix %*% D

  # Solve: Sigma_corr * w = 1_n (optimal in correlation space)
  cor_inv <- tryCatch(solve(cor_matrix), error = function(e) {
    solve(cor_matrix + diag(1e-6, n))
  })

  w_raw <- cor_inv %*% vols  # Weight inversely to correlation structure
  w_raw <- pmax(w_raw, 0)
  w     <- w_raw / sum(w_raw)

  dr <- diversification_ratio(as.vector(w), cov_matrix)
  list(weights = as.vector(w), diversification_ratio = dr)
}

# ---------------------------------------------------------------------------
# 4. EQUAL RISK CONTRIBUTION WITH CVaR
# ---------------------------------------------------------------------------
# ERC-CVaR: choose weights such that each asset contributes equally to
# portfolio CVaR (tail risk), not variance.
# CVaR_i contribution = w_i * (partial CVaR / partial w_i)

#' Portfolio CVaR (Expected Shortfall)
portfolio_cvar <- function(weights, returns_matrix, q = 0.05) {
  port_rets <- returns_matrix %*% weights
  threshold <- quantile(port_rets, q)
  mean(port_rets[port_rets <= threshold])
}

#' Marginal CVaR contribution for asset i
marginal_cvar <- function(weights, returns_matrix, q = 0.05, eps = 1e-4) {
  n   <- length(weights)
  cvr <- portfolio_cvar(weights, returns_matrix, q)
  sapply(seq_len(n), function(i) {
    w_bump <- weights; w_bump[i] <- w_bump[i] + eps
    w_bump <- w_bump / sum(w_bump)
    (portfolio_cvar(w_bump, returns_matrix, q) - cvr) / eps
  })
}

#' Component CVaR (each asset's contribution)
component_cvar_contribution <- function(weights, returns_matrix, q = 0.05) {
  mc  <- marginal_cvar(weights, returns_matrix, q)
  weights * mc
}

#' ERC-CVaR optimization via Newton's method on budget equations
#' Budget equation: w_i * MC_i = c for all i (equal contributions)
erc_cvar_optimize <- function(returns_matrix, q = 0.05, lb = 0.01,
                               n_iter = 200, step = 0.001, seed = 42) {
  n <- ncol(returns_matrix)
  set.seed(seed)
  w <- rep(1/n, n)

  for (iter in seq_len(n_iter)) {
    cc    <- component_cvar_contribution(w, returns_matrix, q)
    total_cvar <- sum(cc)
    if (abs(total_cvar) < 1e-10) break

    target_contrib <- total_cvar / n

    # Gradient: increase weight of undercontributing assets
    diff <- target_contrib - cc
    w_new <- w + step * diff / (abs(total_cvar) + 1e-10)
    w_new <- pmax(w_new, lb)
    w_new <- w_new / sum(w_new)

    if (max(abs(w_new - w)) < 1e-7) break
    w <- w_new
  }

  cc_final <- component_cvar_contribution(w, returns_matrix, q)

  list(
    weights    = w,
    component_cvar = cc_final,
    cvar_shares = cc_final / sum(cc_final),
    portfolio_cvar = portfolio_cvar(w, returns_matrix, q),
    max_contrib_diff = max(cc_final) - min(cc_final)
  )
}

# ---------------------------------------------------------------------------
# 5. TAIL RISK PARITY: EQUALIZE CVaR CONTRIBUTIONS
# ---------------------------------------------------------------------------
# More extreme than ERC-CVaR: specifically focuses on tail scenarios
# Uses empirical CVaR at very low quantile (q = 0.01)

#' Tail Risk Parity portfolio
tail_risk_parity <- function(returns_matrix, q = 0.01, lb = 0.005,
                              n_iter = 300, seed = 42) {
  # ERC-CVaR at extreme quantile = Tail Risk Parity
  erc_cvar_optimize(returns_matrix, q = q, lb = lb, n_iter = n_iter, seed = seed)
}

#' Compare risk parity approaches
compare_risk_parity <- function(returns_matrix, q_erc = 0.05, q_tail = 0.01) {
  n <- ncol(returns_matrix)
  cov_m <- cov(returns_matrix)

  # 1. Equal Weight
  ew <- rep(1/n, n)

  # 2. Standard ERC (variance-based)
  vols <- sqrt(diag(cov_m))
  iv_weights <- (1/vols) / sum(1/vols)  # Inverse vol weighting approximates ERC

  # 3. ERC-CVaR
  erc <- erc_cvar_optimize(returns_matrix, q = q_erc)

  # 4. Tail Risk Parity
  trp <- tail_risk_parity(returns_matrix, q = q_tail)

  # 5. MDP
  mdp <- mdp_optimize(cov_m)

  portfolios <- list(
    "Equal Weight" = ew,
    "Inv Vol (ERC approx)" = iv_weights,
    "ERC-CVaR (5%)" = erc$weights,
    "Tail Risk Parity (1%)" = trp$weights,
    "MDP" = mdp$weights
  )

  # Performance comparison
  cat("=== Risk Parity Approaches Comparison ===\n")
  results <- lapply(names(portfolios), function(nm) {
    w <- portfolios[[nm]]
    port_rets <- returns_matrix %*% w
    sharpe <- mean(port_rets) / (sd(port_rets) + 1e-8) * sqrt(252)
    var_99 <- quantile(port_rets, 0.01)
    cvar_5 <- portfolio_cvar(w, returns_matrix, 0.05)
    dr     <- diversification_ratio(w, cov_m)

    cat(sprintf("%-25s | Sharpe=%5.2f | VaR(1%%)=%6.3f%% | CVaR(5%%)=%6.3f%% | DR=%.3f\n",
                nm, sharpe, var_99*100, cvar_5*100, dr))

    data.frame(portfolio=nm, sharpe=sharpe, var99=var_99, cvar5=cvar_5, dr=dr)
  })

  invisible(do.call(rbind, results))
}

# ---------------------------------------------------------------------------
# 6. MOMENTUM OVERLAY
# ---------------------------------------------------------------------------
# Tilt portfolio weights toward recent winners
# Momentum signal: risk-adjusted trailing return
# New weight = base_weight * (1 + momentum_tilt * momentum_score_normalized)

#' Compute momentum scores (cross-sectional, normalized)
compute_momentum_scores <- function(returns_matrix, lookback = 20,
                                     skip_last = 1) {
  T_ <- nrow(returns_matrix)
  n  <- ncol(returns_matrix)

  if (T_ <= lookback + skip_last) {
    return(rep(0, n))
  }

  # Use returns from [T-lookback-skip_last+1 : T-skip_last]
  start_idx <- T_ - lookback - skip_last + 1
  end_idx   <- T_ - skip_last

  past_rets <- returns_matrix[start_idx:end_idx, ]
  cum_rets  <- apply(past_rets, 2, function(r) prod(1 + r) - 1)
  past_vols <- apply(past_rets, 2, sd)

  # Risk-adjusted momentum
  mom_score <- cum_rets / (past_vols * sqrt(lookback) + 1e-8)

  # Normalize to mean=0, sd=1 (cross-sectional z-score)
  mom_score <- (mom_score - mean(mom_score)) / (sd(mom_score) + 1e-8)
  mom_score
}

#' Apply momentum overlay to base portfolio weights
momentum_overlay <- function(base_weights, momentum_scores,
                              tilt_strength = 0.20,  # 20% tilt
                              min_weight = 0.02,
                              max_tilt_per_asset = 0.50) {
  n <- length(base_weights)

  # Limit tilt magnitude per asset
  tilt_scores <- pmax(-max_tilt_per_asset,
                      pmin(max_tilt_per_asset, momentum_scores))

  # New weights
  w_new <- base_weights * (1 + tilt_strength * tilt_scores)
  w_new <- pmax(w_new, min_weight)
  w_new <- w_new / sum(w_new)  # Renormalize

  list(
    weights_overlaid = w_new,
    weights_change   = w_new - base_weights,
    tilt_scores      = tilt_scores,
    max_change       = max(abs(w_new - base_weights))
  )
}

# ---------------------------------------------------------------------------
# 7. CARRY OVERLAY
# ---------------------------------------------------------------------------
# Tilt toward assets with positive funding rates (longs getting paid)
# and away from assets with expensive funding (negative carry)

#' Compute carry scores from funding rates
compute_carry_scores <- function(funding_rates_matrix, window = 8) {
  # funding_rates_matrix: [T x n_assets], each entry is 8h funding rate
  T_ <- nrow(funding_rates_matrix); n <- ncol(funding_rates_matrix)

  if (T_ < window) {
    return(rep(0, n))
  }

  # Recent average funding rate per asset
  recent_funding <- colMeans(funding_rates_matrix[(T_-window+1):T_, , drop=FALSE])

  # Annualize: 3 settlements per day * 365
  ann_funding <- recent_funding * 3 * 365

  # Cross-sectional z-score
  carry_z <- (ann_funding - mean(ann_funding)) / (sd(ann_funding) + 1e-8)

  # Sign flip: positive funding for longs is negative carry (shorts collect)
  # If we're neutral: positive funding = bearish (market is overly long)
  # Carry signal: prefer assets where shorts pay us (negative funding)
  -carry_z  # Or positive carry: pick direction based on strategy
}

#' Apply carry overlay
carry_overlay <- function(base_weights, carry_scores,
                           tilt_strength = 0.15, min_weight = 0.02) {
  w_new <- base_weights * (1 + tilt_strength * carry_scores)
  w_new <- pmax(w_new, min_weight)
  w_new <- w_new / sum(w_new)

  list(
    weights_overlaid = w_new,
    weights_change   = w_new - base_weights,
    carry_scores     = carry_scores
  )
}

# ---------------------------------------------------------------------------
# 8. COMBINED OVERLAY: MOMENTUM + CARRY + QUALITY
# ---------------------------------------------------------------------------
# Quality proxy: Sharpe ratio over medium-term window
# Combined signal: weighted average of momentum, carry, quality scores

#' Quality scores (medium-term Sharpe)
compute_quality_scores <- function(returns_matrix, window = 60) {
  T_ <- nrow(returns_matrix); n <- ncol(returns_matrix)

  if (T_ < window) return(rep(0, n))

  recent <- returns_matrix[(T_-window+1):T_, ]
  sharpes <- apply(recent, 2, function(r) {
    mean(r) / (sd(r) + 1e-8) * sqrt(252)
  })

  # Z-score
  (sharpes - mean(sharpes)) / (sd(sharpes) + 1e-8)
}

#' Combined overlay with user-defined weights for each signal
combined_overlay <- function(base_weights, returns_matrix,
                              funding_rates_matrix = NULL,
                              mom_weight    = 0.40,  # Weight of momentum signal
                              carry_weight  = 0.30,  # Weight of carry signal
                              quality_weight = 0.30, # Weight of quality signal
                              tilt_strength = 0.25,
                              min_weight    = 0.02,
                              lookback_mom  = 20,
                              lookback_qual = 60,
                              lookback_carry = 8) {
  n <- length(base_weights)

  # Compute individual signals
  mom_scores  <- compute_momentum_scores(returns_matrix, lookback_mom)
  qual_scores <- compute_quality_scores(returns_matrix, lookback_qual)

  if (!is.null(funding_rates_matrix)) {
    carry_scores <- compute_carry_scores(funding_rates_matrix, lookback_carry)
  } else {
    carry_scores <- rep(0, n)
    carry_weight <- 0
    # Redistribute carry weight
    total_w <- mom_weight + quality_weight
    mom_weight    <- mom_weight    / total_w
    quality_weight <- quality_weight / total_w
  }

  # Composite signal (weighted average, each already z-scored)
  composite <- mom_weight * mom_scores +
               carry_weight * carry_scores +
               quality_weight * qual_scores

  # Normalize composite to unit std
  composite <- composite / (sd(composite) + 1e-8)

  # Apply overlay
  w_new <- base_weights * (1 + tilt_strength * composite)
  w_new <- pmax(w_new, min_weight)
  w_new <- w_new / sum(w_new)

  list(
    weights_overlaid = w_new,
    weights_change   = w_new - base_weights,
    composite_score  = composite,
    mom_scores  = mom_scores,
    qual_scores = qual_scores,
    carry_scores = carry_scores
  )
}

# ---------------------------------------------------------------------------
# 9. PERFORMANCE METRICS FOR ALL PORTFOLIOS
# ---------------------------------------------------------------------------

#' Comprehensive portfolio performance metrics
portfolio_metrics <- function(weights, returns_matrix, rf = 0,
                               portfolio_name = "Portfolio") {
  port_rets <- as.vector(returns_matrix %*% weights)
  T_ <- length(port_rets)

  cum_ret    <- prod(1 + port_rets) - 1
  ann_ret    <- (1 + cum_ret)^(252/T_) - 1
  ann_vol    <- sd(port_rets) * sqrt(252)
  sharpe     <- (ann_ret - rf) / (ann_vol + 1e-8)

  # Drawdown
  equity     <- cumprod(1 + port_rets)
  drawdown   <- equity / cummax(equity) - 1
  max_dd     <- min(drawdown)

  # CVaR
  cvar5  <- mean(port_rets[port_rets <= quantile(port_rets, 0.05)])
  cvar1  <- mean(port_rets[port_rets <= quantile(port_rets, 0.01)])

  # Calmar ratio
  calmar <- ann_ret / (abs(max_dd) + 1e-8)

  # Skewness and kurtosis
  skew  <- mean(((port_rets - mean(port_rets)) / (sd(port_rets) + 1e-8))^3)
  kurt  <- mean(((port_rets - mean(port_rets)) / (sd(port_rets) + 1e-8))^4)

  list(
    name       = portfolio_name,
    cum_return = cum_ret * 100,
    ann_return = ann_ret * 100,
    ann_vol    = ann_vol * 100,
    sharpe     = sharpe,
    max_drawdown = max_dd * 100,
    calmar     = calmar,
    cvar5_pct  = cvar5 * 100,
    cvar1_pct  = cvar1 * 100,
    skewness   = skew,
    excess_kurtosis = kurt - 3
  )
}

#' Run all portfolio construction methods and compare
full_portfolio_comparison <- function(returns_matrix, funding_rates = NULL) {
  n    <- ncol(returns_matrix)
  cov_m <- cov(returns_matrix)

  # Base portfolios
  ew   <- rep(1/n, n)
  mdp  <- mdp_optimize(cov_m, seed = 1)
  mcp  <- mincorr_optimize(cor(returns_matrix), seed = 1)
  erccvar <- erc_cvar_optimize(returns_matrix, q = 0.05)
  trp  <- tail_risk_parity(returns_matrix, q = 0.01)

  # Overlays on top of MDP base
  mom_ov <- momentum_overlay(mdp$weights,
                              compute_momentum_scores(returns_matrix))$weights_overlaid
  combined <- combined_overlay(mdp$weights, returns_matrix, funding_rates)$weights_overlaid

  portfolios <- list(
    "Equal Weight"         = ew,
    "MDP"                  = mdp$weights,
    "Min Correlation"      = mcp$weights,
    "ERC-CVaR (5%)"        = erccvar$weights,
    "Tail Risk Parity (1%)" = trp$weights,
    "MDP + Momentum"       = mom_ov,
    "MDP + Combined"       = combined
  )

  cat("╔══════════════════════════════════════════════════════════════════════╗\n")
  cat("║                    Portfolio Method Comparison                       ║\n")
  cat("╚══════════════════════════════════════════════════════════════════════╝\n\n")
  cat(sprintf("%-25s | %7s | %7s | %7s | %8s | %7s\n",
              "Portfolio", "AnnRet%", "AnnVol%", "Sharpe", "MaxDD%", "CVaR5%"))
  cat(paste(rep("-", 75), collapse=""), "\n")

  results <- lapply(names(portfolios), function(nm) {
    m <- portfolio_metrics(portfolios[[nm]], returns_matrix, portfolio_name = nm)
    cat(sprintf("%-25s | %7.2f | %7.2f | %7.3f | %8.2f | %7.3f\n",
                nm, m$ann_return, m$ann_vol, m$sharpe, m$max_drawdown, m$cvar5_pct))
    m
  })

  invisible(list(portfolios = portfolios, metrics = results))
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  set.seed(42)
  n <- 10; T_ <- 500

  # Simulate returns with correlation structure
  btc <- rnorm(T_, 0.001, 0.04)
  ret_mat <- matrix(NA, T_, n)
  betas <- runif(n, 0.3, 1.2)
  for (i in seq_len(n)) {
    ret_mat[, i] <- betas[i] * btc + rnorm(T_, 0, 0.02)
  }
  colnames(ret_mat) <- c("BTC","ETH","BNB","SOL","AVAX",
                          "DOT","MATIC","LINK","UNI","AAVE")

  # Run full comparison
  comparison <- full_portfolio_comparison(ret_mat)

  # MDP details
  cov_m <- cov(ret_mat)
  mdp <- mdp_optimize(cov_m)
  cat(sprintf("\nMDP Diversification Ratio: %.3f\n", mdp$diversification_ratio))
  cat("MDP Weights:\n")
  for (i in seq_len(n)) {
    cat(sprintf("  %-8s: %.3f\n", colnames(ret_mat)[i], mdp$weights[i]))
  }

  # ERC-CVaR
  erc <- erc_cvar_optimize(ret_mat, q=0.05)
  cat(sprintf("\nERC-CVaR: max spread between contributions = %.4f\n",
              erc$max_contrib_diff))
}
