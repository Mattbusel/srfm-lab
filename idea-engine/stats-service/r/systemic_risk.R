# =============================================================================
# systemic_risk.R
# Systemic Risk Measures: CoVaR, MES, SRISK, network contagion, Granger
# causality networks, exchange interconnectedness, crypto token exposure.
# Pure base R. Based on Adrian-Brunnermeier 2016, Acharya et al 2012.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. CoVaR: CONDITIONAL VALUE AT RISK
# ---------------------------------------------------------------------------
# CoVaR_q(system | institution i) = VaR_q of system conditional on
# institution i being at its VaR_q.
# Estimated via quantile regression: r_sys = alpha + beta * r_i + epsilon
# CoVaR = alpha_q + beta_q * VaR_q(r_i)

#' Quantile regression via simplex / linear programming approximation
#' Minimizes sum of quantile loss: rho_tau(e) = e*(tau - I(e<0))
#' Uses IRLS (iteratively reweighted least squares) approximation
quantile_regression <- function(y, X, tau = 0.05, max_iter = 200, tol = 1e-8) {
  n <- length(y)
  if (is.vector(X)) X <- cbind(1, X)
  p <- ncol(X)

  # Initialize with OLS
  beta <- as.vector(solve(crossprod(X) + diag(1e-8, p), crossprod(X, y)))

  for (iter in 1:max_iter) {
    resid <- y - X %*% beta
    # Huber-like weights for quantile regression
    weights <- ifelse(resid >= 0, tau, 1 - tau)
    weights <- pmax(weights, 1e-6)
    W <- diag(as.vector(weights))
    beta_new <- as.vector(
      solve(crossprod(X, W) %*% X + diag(1e-8, p), crossprod(X, W) %*% y)
    )
    if (max(abs(beta_new - beta)) < tol) break
    beta <- beta_new
  }
  beta
}

#' Compute CoVaR for a given institution vs system
#' Returns: VaR_i, CoVaR (system | i at VaR), Delta-CoVaR
compute_covar <- function(r_system, r_institution, q = 0.05) {
  # Quantile regression: r_sys = a + b * r_i
  X <- cbind(1, r_institution)

  # Two quantile regressions at q (tail) and 0.5 (median state)
  beta_q   <- quantile_regression(r_system, X, tau = q)
  beta_med <- quantile_regression(r_system, X, tau = 0.5)

  # VaR of institution at quantile q
  var_i <- quantile(r_institution, q)

  # CoVaR: system VaR conditional on institution being at its VaR
  covar <- beta_q[1] + beta_q[2] * var_i

  # Delta-CoVaR: difference from median state
  covar_median <- beta_med[1] + beta_med[2] * var_i

  list(
    VaR_institution  = var_i,
    CoVaR_system     = covar,
    CoVaR_median     = covar_median,
    DeltaCoVaR       = covar - covar_median,  # Contribution to systemic risk
    beta_q           = beta_q[2],
    beta_median      = beta_med[2]
  )
}

#' Pairwise CoVaR matrix for multiple assets
covar_matrix <- function(returns_matrix, q = 0.05) {
  n_assets <- ncol(returns_matrix)
  asset_names <- colnames(returns_matrix)
  if (is.null(asset_names)) asset_names <- paste0("A", seq_len(n_assets))

  # System return = equal-weight portfolio
  r_system <- rowMeans(returns_matrix)

  results <- lapply(seq_len(n_assets), function(i) {
    r <- compute_covar(r_system, returns_matrix[, i], q)
    data.frame(
      asset = asset_names[i],
      VaR = r$VaR_institution,
      CoVaR = r$CoVaR_system,
      DeltaCoVaR = r$DeltaCoVaR,
      beta_q = r$beta_q
    )
  })

  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 2. MES: MARGINAL EXPECTED SHORTFALL
# ---------------------------------------------------------------------------
# MES_i = E[r_i | r_market <= VaR_q(market)]
# Measures how much institution i loses when market is in its tail

#' Marginal Expected Shortfall for one asset
compute_mes <- function(r_asset, r_market, q = 0.05) {
  threshold <- quantile(r_market, q)
  tail_mask <- r_market <= threshold
  if (sum(tail_mask) < 5) {
    warning("Too few tail observations")
    return(NA)
  }
  mes <- mean(r_asset[tail_mask])
  list(
    MES     = mes,
    VaR_mkt = threshold,
    n_tail  = sum(tail_mask),
    beta    = cov(r_asset[tail_mask], r_market[tail_mask]) /
              var(r_market[tail_mask])
  )
}

#' MES for all assets in portfolio
portfolio_mes <- function(returns_matrix, q = 0.05, market_weights = NULL) {
  n <- ncol(returns_matrix)
  if (is.null(market_weights)) market_weights <- rep(1/n, n)
  r_market <- returns_matrix %*% market_weights

  sapply(seq_len(n), function(i) {
    mes <- compute_mes(returns_matrix[, i], r_market, q)
    if (is.list(mes)) mes$MES else NA
  })
}

# ---------------------------------------------------------------------------
# 3. SRISK: CAPITAL SHORTFALL MEASURE
# ---------------------------------------------------------------------------
# SRISK_i = max(0, k*(D_i + W_i) - (1-LRMES_i)*W_i)
# where k = prudential capital ratio (8%), D_i = debt, W_i = market equity
# LRMES = Long-Run MES: expected equity loss in a market crash of -40%

#' Long-Run MES (LRMES) estimate
#' Approximation: LRMES ≈ 1 - exp(18 * MES_daily)
#' Based on Brownlees-Engle 2012
lrmes <- function(mes_daily) {
  1 - exp(18 * mes_daily)  # 6-month horizon, ~18 * daily MES
}

#' SRISK for a single institution
compute_srisk <- function(equity_value, debt_value, mes_daily,
                           k = 0.08) {
  lrmes_val <- lrmes(mes_daily)
  srisk <- k * (debt_value + equity_value) - (1 - lrmes_val) * equity_value
  list(
    SRISK         = max(0, srisk),
    LRMES         = lrmes_val,
    capital_need  = srisk,
    leverage      = (debt_value + equity_value) / equity_value
  )
}

#' System-level SRISK from multiple institutions
system_srisk <- function(equity_values, debt_values, mes_daily_vec, k = 0.08) {
  n <- length(equity_values)
  results <- lapply(seq_len(n), function(i) {
    compute_srisk(equity_values[i], debt_values[i], mes_daily_vec[i], k)
  })

  total_srisk <- sum(sapply(results, function(r) r$SRISK))
  individual_srisk <- sapply(results, function(r) r$SRISK)

  list(
    total_system_srisk = total_srisk,
    individual_srisk   = individual_srisk,
    srisk_share        = individual_srisk / total_srisk,
    lrmes_vec          = sapply(results, function(r) r$LRMES)
  )
}

# ---------------------------------------------------------------------------
# 4. NETWORK CONTAGION: DEBTRANK ALGORITHM
# ---------------------------------------------------------------------------
# DebtRank (Battiston et al 2012): measures indirect contagion via
# exposure network. Each node has equity buffer; shocks propagate.

#' Build exposure matrix from pairwise lending/holdings
#' W[i,j] = exposure of i to j as fraction of i's equity
build_exposure_matrix <- function(n = 10, seed = 42, sparsity = 0.3) {
  set.seed(seed)
  W <- matrix(0, n, n)
  # Random sparse exposures
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i != j && runif(1) < sparsity) {
        W[i, j] <- runif(1, 0, 0.15)
      }
    }
    # Normalize rows to sum <= 0.5 (max 50% exposure to any counterparty)
    row_sum <- sum(W[i, ])
    if (row_sum > 0.5) W[i, ] <- W[i, ] * 0.5 / row_sum
  }
  W
}

#' DebtRank contagion simulation
#' Initial shock: one or more nodes default (stress level = 1)
#' Contagion propagates via exposure matrix
debtrank <- function(W, initial_stress, equity = NULL, max_iter = 100) {
  n <- nrow(W)
  if (is.null(equity)) equity <- rep(1, n)

  # h[i] = stress level of node i in [0, 1]
  h <- initial_stress  # vector of initial stress
  h_prev <- rep(-1, n)

  # Distress status: 0 = inactive, 1 = active (being stressed), 2 = done
  status <- ifelse(h > 0, 1L, 0L)

  for (iter in seq_len(max_iter)) {
    if (max(abs(h - h_prev)) < 1e-10) break
    h_prev <- h
    h_new <- h

    for (i in seq_len(n)) {
      if (status[i] != 1) next
      # Propagate distress from i to j
      for (j in seq_len(n)) {
        if (i == j) next
        impact <- W[j, i] * h[i]  # j's exposure to i * i's stress
        h_new[j] <- min(1, h_new[j] + impact)
        if (h_new[j] > 0 && status[j] == 0) status[j] <- 1L
      }
      status[i] <- 2L
    }
    h <- h_new
  }

  # DebtRank = sum of h[i] * equity[i] - initial_shock_value
  equity_loss <- sum(h * equity)
  initial_loss <- sum(initial_stress * equity)
  debt_rank <- (equity_loss - initial_loss) / sum(equity)

  list(
    stress_levels = h,
    debt_rank     = debt_rank,
    equity_loss   = equity_loss,
    contagion_ratio = equity_loss / initial_loss,
    n_distressed  = sum(h > 0.5)
  )
}

#' Systemic importance ranking via DebtRank (shock each node individually)
node_systemic_importance <- function(W, equity = NULL) {
  n <- nrow(W)
  if (is.null(equity)) equity <- rep(1, n)

  importance <- sapply(seq_len(n), function(i) {
    h0 <- rep(0, n)
    h0[i] <- 1.0  # Full default of node i
    r <- debtrank(W, h0, equity)
    r$debt_rank
  })

  data.frame(
    node = seq_len(n),
    debtrank_importance = importance,
    rank = rank(-importance)
  )
}

# ---------------------------------------------------------------------------
# 5. GRANGER CAUSALITY NETWORK FOR SYSTEMIC LINKAGES
# ---------------------------------------------------------------------------

#' Granger causality test: does X Granger-cause Y?
#' Compare restricted (AR of Y) vs unrestricted (AR of Y + lags of X)
granger_test <- function(y, x, lag = 5) {
  n <- length(y)
  if (n < lag * 3) return(list(F_stat = NA, p_value = NA, granger_causes = FALSE))

  # Build lag matrices
  max_lag <- lag
  y_dep <- y[(max_lag + 1):n]

  # Restricted: only y lags
  Xr <- matrix(NA, length(y_dep), max_lag)
  for (l in seq_len(max_lag)) Xr[, l] <- y[(max_lag + 1 - l):(n - l)]
  Xr <- cbind(1, Xr)

  # Unrestricted: y lags + x lags
  Xu <- Xr
  for (l in seq_len(max_lag)) {
    Xu <- cbind(Xu, x[(max_lag + 1 - l):(n - l)])
  }

  # OLS residuals
  solve_ols <- function(X, y) {
    XtX_inv <- tryCatch(solve(crossprod(X)), error = function(e) {
      solve(crossprod(X) + diag(1e-8, ncol(X)))
    })
    beta <- XtX_inv %*% crossprod(X, y)
    resid <- y - X %*% beta
    sum(resid^2)
  }

  RSS_r  <- solve_ols(Xr, y_dep)
  RSS_ur <- solve_ols(Xu, y_dep)

  n_obs <- length(y_dep)
  q <- max_lag   # number of restrictions
  k <- ncol(Xu)  # parameters in unrestricted

  F_stat  <- ((RSS_r - RSS_ur) / q) / (RSS_ur / (n_obs - k))
  p_value <- pf(F_stat, q, n_obs - k, lower.tail = FALSE)

  list(
    F_stat = F_stat,
    p_value = p_value,
    granger_causes = p_value < 0.05
  )
}

#' Build Granger causality adjacency matrix for a system of assets
#' A[i,j] = 1 if asset j Granger-causes asset i
granger_network <- function(returns_matrix, lag = 5, alpha = 0.05) {
  n <- ncol(returns_matrix)
  names_ <- colnames(returns_matrix)
  if (is.null(names_)) names_ <- paste0("A", seq_len(n))

  adj <- matrix(0, n, n)
  pvals <- matrix(NA, n, n)

  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i == j) next
      g <- granger_test(returns_matrix[, i], returns_matrix[, j], lag)
      adj[i, j] <- as.integer(g$granger_causes)
      pvals[i, j] <- g$p_value
    }
  }

  rownames(adj) <- colnames(adj) <- names_
  list(
    adjacency = adj,
    p_values  = pvals,
    in_degree  = colSums(adj),  # How many others cause this node
    out_degree = rowSums(adj),  # How many others this node causes
    hubs       = names_[which(rowSums(adj) == max(rowSums(adj)))]
  )
}

# ---------------------------------------------------------------------------
# 6. CONTAGION INDEX FROM CORRELATION NETWORK
# ---------------------------------------------------------------------------

#' Absorb eigenvector centrality from correlation matrix
#' High centrality = node whose stress propagates widely
correlation_contagion_index <- function(returns_matrix, threshold = 0.5) {
  S <- cor(returns_matrix)
  n <- nrow(S)

  # Build adjacency from thresholded correlations
  A <- (abs(S) > threshold) * 1
  diag(A) <- 0

  # Eigenvector centrality via power iteration
  x <- rep(1/n, n)
  for (iter in seq_len(100)) {
    x_new <- A %*% x
    norm_val <- sqrt(sum(x_new^2))
    if (norm_val < 1e-12) break
    x_new <- x_new / norm_val
    if (max(abs(x_new - x)) < 1e-10) break
    x <- x_new
  }

  names_ <- colnames(returns_matrix)
  if (is.null(names_)) names_ <- paste0("A", seq_len(n))

  data.frame(
    asset = names_,
    eigenvector_centrality = as.vector(x),
    degree = rowSums(A),
    contagion_rank = rank(-as.vector(x))
  )
}

#' Contagion-adjusted VaR (higher centrality = higher systemic VaR)
contagion_adjusted_var <- function(individual_vars, centrality_scores,
                                    lambda = 0.5) {
  # Adjusted VaR = individual VaR * (1 + lambda * normalized_centrality)
  norm_centrality <- centrality_scores / max(centrality_scores)
  individual_vars * (1 + lambda * norm_centrality)
}

# ---------------------------------------------------------------------------
# 7. EXCHANGE INTERCONNECTEDNESS SCORING
# ---------------------------------------------------------------------------

#' Score exchange interconnectedness from shared liquidity / user overlap
#' Proxy: cross-listed tokens, volume correlation, shared stablecoin flows

exchange_interconnectedness <- function(exchange_returns, volume_matrix = NULL) {
  n_exchanges <- ncol(exchange_returns)
  names_ <- colnames(exchange_returns)

  # Correlation matrix of exchange returns (proxy for interconnectedness)
  corr <- cor(exchange_returns)

  # Contagion index from correlation network
  ci <- correlation_contagion_index(exchange_returns, threshold = 0.4)

  # Volume-weighted interconnectedness (if volume data provided)
  vol_score <- rep(1, n_exchanges)
  if (!is.null(volume_matrix)) {
    # Normalize by total volume
    vol_score <- rowSums(volume_matrix) / sum(volume_matrix)
  }

  # Composite score
  composite <- ci$eigenvector_centrality * vol_score
  composite <- composite / max(composite)

  data.frame(
    exchange = names_,
    correlation_centrality = ci$eigenvector_centrality,
    volume_weight = vol_score,
    composite_interconnectedness = composite,
    avg_correlation_to_others = (rowSums(corr) - 1) / (n_exchanges - 1)
  )
}

# ---------------------------------------------------------------------------
# 8. CRYPTO SHARED TOKEN EXPOSURE MATRIX
# ---------------------------------------------------------------------------

#' Build shared token exposure matrix
#' exposure[i,j] = fraction of exchange/protocol i's assets also held by j
shared_token_exposure <- function(portfolio_matrix) {
  # portfolio_matrix[i, k] = fraction of protocol i in token k
  n <- nrow(portfolio_matrix)
  names_ <- rownames(portfolio_matrix)

  exposure <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i == j) next
      # Cosine similarity of token holdings
      a <- portfolio_matrix[i, ]
      b <- portfolio_matrix[j, ]
      denom <- sqrt(sum(a^2)) * sqrt(sum(b^2))
      if (denom > 0) exposure[i, j] <- sum(a * b) / denom
    }
  }

  rownames(exposure) <- colnames(exposure) <- names_
  exposure
}

#' Simulate crypto ecosystem contagion via token exposure
#' If BTC crashes: all protocols holding BTC lose proportionally
crypto_ecosystem_contagion <- function(token_holdings, token_shock,
                                        n_protocols = 10) {
  # token_holdings[i,k] = fraction of protocol i in token k
  # token_shock[k] = return of token k in stress scenario
  if (is.null(dim(token_holdings))) {
    stop("token_holdings must be a matrix [protocols x tokens]")
  }

  # Protocol loss = weighted sum of token shocks
  protocol_returns <- token_holdings %*% token_shock

  # Second-round: distressed protocols withdraw liquidity, hurting others
  exposure <- shared_token_exposure(token_holdings)
  debtrank_stress <- pmax(0, -protocol_returns)  # Initial stress from token shock

  contagion <- debtrank(exposure, debtrank_stress)

  list(
    first_round_returns = protocol_returns,
    debtrank_total = contagion$debt_rank,
    final_stress = contagion$stress_levels,
    most_affected = which.max(contagion$stress_levels)
  )
}

#' Generate synthetic crypto ecosystem for testing
generate_crypto_ecosystem <- function(n_protocols = 10, n_tokens = 20,
                                       seed = 42) {
  set.seed(seed)
  # Each protocol has exposure to different token mix
  holdings <- matrix(runif(n_protocols * n_tokens), n_protocols, n_tokens)
  # Normalize rows to sum to 1 (portfolio weights)
  holdings <- holdings / rowSums(holdings)
  rownames(holdings) <- paste0("Protocol_", seq_len(n_protocols))
  colnames(holdings) <- paste0("Token_", seq_len(n_tokens))

  # Simulate correlated token returns (common factor = BTC)
  btc_beta <- runif(n_tokens, 0.3, 1.5)
  eth_beta <- runif(n_tokens, 0.2, 1.2)
  btc_ret  <- rnorm(252, 0, 0.04)
  eth_ret  <- rnorm(252, 0, 0.035)
  idio     <- matrix(rnorm(252 * n_tokens, 0, 0.02), 252, n_tokens)

  token_returns <- outer(btc_ret, btc_beta) + outer(eth_ret, eth_beta) + idio

  list(holdings = holdings, token_returns = token_returns)
}

# ---------------------------------------------------------------------------
# 9. SYSTEMIC RISK DASHBOARD
# ---------------------------------------------------------------------------

#' Compute all systemic risk measures for a set of returns
systemic_risk_dashboard <- function(returns_matrix, q = 0.05) {
  n <- ncol(returns_matrix)
  T_ <- nrow(returns_matrix)
  names_ <- colnames(returns_matrix)
  if (is.null(names_)) names_ <- paste0("Asset", seq_len(n))

  cat("╔══════════════════════════════════════════════════════╗\n")
  cat("║           Systemic Risk Dashboard                     ║\n")
  cat("╚══════════════════════════════════════════════════════╝\n\n")
  cat(sprintf("Assets: %d | Observations: %d | Tail quantile: %.0f%%\n\n",
              n, T_, q * 100))

  # CoVaR
  cat("--- CoVaR Analysis (system = equal-weight portfolio) ---\n")
  cv <- covar_matrix(returns_matrix, q)
  cv <- cv[order(cv$DeltaCoVaR), ]
  print(cv)
  cat("\n")

  # MES
  cat("--- Marginal Expected Shortfall ---\n")
  r_mkt <- rowMeans(returns_matrix)
  mes_vals <- sapply(seq_len(n), function(i) {
    compute_mes(returns_matrix[, i], r_mkt, q)$MES
  })
  mes_df <- data.frame(asset = names_, MES = mes_vals,
                        MES_rank = rank(mes_vals))
  print(mes_df[order(mes_df$MES), ])
  cat("\n")

  # Correlation contagion
  cat("--- Contagion Centrality (Eigenvector) ---\n")
  cc <- correlation_contagion_index(returns_matrix, threshold = 0.4)
  print(cc[order(-cc$eigenvector_centrality), ])
  cat("\n")

  # Granger network (only if reasonable size)
  if (n <= 15 && T_ >= 100) {
    cat("--- Granger Causality Network Summary ---\n")
    gn <- granger_network(returns_matrix, lag = 3)
    g_df <- data.frame(
      asset = names_,
      causes_others = gn$out_degree,
      caused_by_others = gn$in_degree
    )
    print(g_df[order(-g_df$causes_others), ])
    cat("\n")
  }

  invisible(list(covar = cv, mes = mes_df, centrality = cc))
}

# ---------------------------------------------------------------------------
# 10. STRESS SCENARIO SYSTEMICS
# ---------------------------------------------------------------------------

#' Estimate systemic impact of a single large exchange failure
#' Modeled as forced selling of all assets held on that exchange
exchange_failure_impact <- function(exchange_holdings_usd,
                                     market_depths_usd,
                                     price_impact_elasticity = 0.5) {
  # Price impact: forcing large sell into shallow market
  # Price_impact = -(holdings / depth)^elasticity
  price_impacts <- -(exchange_holdings_usd / market_depths_usd)^price_impact_elasticity
  price_impacts <- pmax(price_impacts, -0.95)  # Floor at -95%

  cascade_tvl_loss <- sum(exchange_holdings_usd * abs(price_impacts))

  list(
    price_impacts = price_impacts,
    worst_affected = which.min(price_impacts),
    cascade_tvl_loss = cascade_tvl_loss,
    market_wide_loss_pct = cascade_tvl_loss / sum(market_depths_usd) * 100
  )
}

#' Rolling systemic risk indicators over time
rolling_systemic_risk <- function(returns_matrix, window = 60, step = 5,
                                   q = 0.05) {
  T_ <- nrow(returns_matrix)
  n  <- ncol(returns_matrix)
  times <- seq(window, T_, by = step)

  results <- lapply(times, function(t) {
    idx  <- (t - window + 1):t
    sub  <- returns_matrix[idx, , drop = FALSE]
    r_mkt <- rowMeans(sub)

    mes_vals <- sapply(seq_len(n), function(i) {
      m <- compute_mes(sub[, i], r_mkt, q)
      if (is.list(m)) m$MES else NA
    })

    avg_corr <- mean(cor(sub)[upper.tri(cor(sub))])

    data.frame(
      t = t,
      avg_MES = mean(mes_vals, na.rm = TRUE),
      max_MES = min(mes_vals, na.rm = TRUE),  # Most negative = highest risk
      avg_pairwise_corr = avg_corr
    )
  })

  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  set.seed(99)
  n <- 8; T_ <- 500
  # Simulate correlated returns with common factor
  btc <- rnorm(T_, 0, 0.04)
  ret <- matrix(NA, T_, n)
  betas <- c(1.0, 0.8, 0.9, 0.6, 0.4, 1.2, 0.7, 0.5)
  for (i in seq_len(n)) {
    ret[, i] <- betas[i] * btc + rnorm(T_, 0, 0.02)
  }
  colnames(ret) <- c("BTC","ETH","BNB","SOL","AVAX","MATIC","DOT","ADA")

  # Full dashboard
  systemic_risk_dashboard(ret, q = 0.05)

  # DebtRank example
  W <- build_exposure_matrix(n = 10, seed = 1)
  h0 <- rep(0, 10); h0[1] <- 1.0  # Node 1 defaults
  dr <- debtrank(W, h0)
  cat(sprintf("DebtRank from node 1 default: %.3f\n", dr$debt_rank))
  cat(sprintf("Contagion ratio: %.2fx\n", dr$contagion_ratio))

  # Crypto ecosystem contagion
  eco <- generate_crypto_ecosystem(n_protocols = 10, n_tokens = 20)
  # BTC crash scenario: tokens 1-3 (BTC/ETH/BNB proxies) crash 50%
  shock <- rep(-0.01, 20); shock[1:3] <- -0.50
  contagion_result <- crypto_ecosystem_contagion(eco$holdings, shock)
  cat(sprintf("First-round avg loss: %.2f%%\n",
              mean(contagion_result$first_round_returns) * 100))
  cat(sprintf("DebtRank amplification: %.3f\n",
              contagion_result$debtrank_total))
}
