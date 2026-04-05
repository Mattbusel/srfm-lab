## portfolio_attribution.R
## Brinson attribution, factor attribution, risk decomposition, performance analysis
## Pure base R -- no library() calls

# ============================================================
# 1. BRINSON-HOOD-BEEBOWER ATTRIBUTION
# ============================================================

brinson_attribution <- function(port_weights, bench_weights,
                                port_returns, bench_returns,
                                sector_labels = NULL) {
  # Allocation effect: (wp - wb) * (rb - rb_total)
  # Selection effect:  wb * (rp - rb)
  # Interaction:       (wp - wb) * (rp - rb)
  rb_total <- sum(bench_weights * bench_returns, na.rm = TRUE)
  rp_total <- sum(port_weights  * port_returns,  na.rm = TRUE)

  alloc    <- (port_weights - bench_weights) * (bench_returns - rb_total)
  select   <- bench_weights * (port_returns - bench_returns)
  interact <- (port_weights - bench_weights) * (port_returns - bench_returns)
  total    <- alloc + select + interact

  active_ret <- rp_total - rb_total

  res <- data.frame(
    allocation   = alloc,
    selection    = select,
    interaction  = interact,
    total        = total
  )
  if (!is.null(sector_labels)) rownames(res) <- sector_labels

  list(
    attribution     = res,
    total_alloc     = sum(alloc, na.rm = TRUE),
    total_select    = sum(select, na.rm = TRUE),
    total_interact  = sum(interact, na.rm = TRUE),
    total_active    = active_ret,
    port_return     = rp_total,
    bench_return    = rb_total
  )
}

brinson_fachler <- function(port_weights, bench_weights,
                             port_returns, bench_returns) {
  # BF variant: allocation uses (rb_sector - rb_total) not rb
  rb_total <- sum(bench_weights * bench_returns, na.rm = TRUE)
  alloc    <- (port_weights - bench_weights) * (bench_returns - rb_total)
  select   <- bench_weights * (port_returns - bench_returns)
  interact <- (port_weights - bench_weights) * (port_returns - bench_returns)
  list(allocation  = alloc, selection = select, interaction = interact,
       total = alloc + select + interact)
}

multi_period_attribution <- function(port_w_list, bench_w_list,
                                     port_r_list, bench_r_list) {
  n_periods <- length(port_r_list)
  period_results <- vector("list", n_periods)
  for (t in seq_len(n_periods)) {
    period_results[[t]] <- brinson_attribution(
      port_w_list[[t]], bench_w_list[[t]],
      port_r_list[[t]], bench_r_list[[t]])
  }
  # Geometric linking (Carino method approximation)
  port_cumret  <- cumprod(sapply(period_results, function(r) 1 + r$port_return))
  bench_cumret <- cumprod(sapply(period_results, function(r) 1 + r$bench_return))
  linked_active <- tail(port_cumret, 1) - tail(bench_cumret, 1)
  list(periods = period_results, linked_active = linked_active,
       cum_port = tail(port_cumret,1)-1, cum_bench = tail(bench_cumret,1)-1)
}

# ============================================================
# 2. FACTOR ATTRIBUTION
# ============================================================

factor_attribution <- function(port_returns, factor_returns,
                                factor_weights = NULL) {
  # OLS regression of portfolio returns on factors
  T_ <- length(port_returns)
  X  <- cbind(1, factor_returns)
  b  <- tryCatch(
    as.vector(solve(t(X) %*% X + diag(ncol(X)) * 1e-8) %*% t(X) %*% port_returns),
    error = function(e) rep(0, ncol(X)))
  alpha  <- b[1]; betas <- b[-1]
  fitted <- as.vector(X %*% b)
  resid  <- port_returns - fitted

  # Factor contributions
  factor_contrib <- betas * colMeans(factor_returns)
  r2    <- 1 - sum(resid^2) / sum((port_returns - mean(port_returns))^2)

  list(alpha = alpha, betas = betas,
       factor_contrib = factor_contrib,
       alpha_contrib  = alpha,
       specific_return = mean(resid),
       r_squared      = r2,
       total_explained = sum(factor_contrib))
}

rolling_factor_attribution <- function(port_returns, factor_returns,
                                        window = 60) {
  T_ <- length(port_returns); n_f <- ncol(factor_returns)
  betas_mat <- matrix(NA, T_, n_f)
  alpha_vec <- rep(NA, T_)
  for (i in seq(window, T_)) {
    idx <- seq(i - window + 1, i)
    res <- factor_attribution(port_returns[idx], factor_returns[idx,])
    betas_mat[i,] <- res$betas
    alpha_vec[i]  <- res$alpha
  }
  list(betas = betas_mat, alpha = alpha_vec)
}

# ============================================================
# 3. RISK DECOMPOSITION
# ============================================================

risk_decomposition <- function(weights, cov_matrix) {
  port_var   <- as.numeric(t(weights) %*% cov_matrix %*% weights)
  port_vol   <- sqrt(port_var)
  marginal   <- as.vector(cov_matrix %*% weights) / port_vol
  component  <- weights * marginal
  pct_contrib <- component / port_vol

  list(
    portfolio_volatility = port_vol,
    portfolio_variance   = port_var,
    marginal_risk        = marginal,
    component_risk       = component,
    pct_contribution     = pct_contrib
  )
}

tail_risk_decomp <- function(returns_matrix, weights, alpha = 0.05) {
  port_ret <- as.vector(returns_matrix %*% weights)
  threshold <- quantile(port_ret, alpha)
  crisis_idx <- which(port_ret <= threshold)
  comp_es <- apply(returns_matrix[crisis_idx, ], 2, mean) * weights
  list(portfolio_es = mean(port_ret[crisis_idx]),
       component_es  = comp_es,
       pct_contribution = comp_es / sum(comp_es))
}

risk_factor_decomp <- function(weights, factor_loadings, factor_cov,
                                idio_var) {
  # Barra-style decomposition
  B   <- factor_loadings   # N x K
  F   <- factor_cov        # K x K
  D   <- diag(idio_var)    # N x N
  Sigma <- B %*% F %*% t(B) + D
  total_var <- as.numeric(t(weights) %*% Sigma %*% weights)
  factor_var <- as.numeric(t(weights) %*% B %*% F %*% t(B) %*% weights)
  idio_var_p <- as.numeric(t(weights) %*% D %*% weights)
  list(total_vol     = sqrt(total_var),
       factor_vol    = sqrt(factor_var),
       specific_vol  = sqrt(idio_var_p),
       factor_pct    = factor_var / total_var,
       specific_pct  = idio_var_p / total_var)
}

# ============================================================
# 4. PERFORMANCE METRICS
# ============================================================

performance_metrics <- function(returns, rf = 0, annualize = 252) {
  excess <- returns - rf / annualize
  mu_ann <- mean(returns) * annualize
  sd_ann <- sd(returns) * sqrt(annualize)
  sharpe <- mean(excess) / sd(excess) * sqrt(annualize)
  down_sd <- sd(pmin(excess, 0)) * sqrt(annualize)
  sortino  <- mean(excess) / (sd(pmin(excess, 0)) + 1e-12) * sqrt(annualize)

  cum   <- cumprod(1 + returns)
  peak  <- cummax(cum)
  dd    <- (cum - peak) / peak
  max_dd <- min(dd)
  calmar <- mu_ann / abs(max_dd + 1e-12)

  # Information ratio vs zero active
  ir    <- mean(excess) / (sd(excess) + 1e-12) * sqrt(annualize)

  list(
    ann_return   = mu_ann, ann_vol  = sd_ann, sharpe = sharpe,
    sortino      = sortino, calmar  = calmar, max_dd = max_dd,
    ir           = ir, down_vol = down_sd,
    skew         = mean((returns - mean(returns))^3) / sd(returns)^3,
    kurt         = mean((returns - mean(returns))^4) / sd(returns)^4 - 3
  )
}

benchmark_relative_metrics <- function(port_ret, bench_ret, rf = 0, annualize = 252) {
  active    <- port_ret - bench_ret
  te        <- sd(active) * sqrt(annualize)
  ir        <- mean(active) / (sd(active) + 1e-12) * sqrt(annualize)
  alpha_ann <- mean(active) * annualize
  pm_p      <- performance_metrics(port_ret, rf, annualize)
  pm_b      <- performance_metrics(bench_ret, rf, annualize)
  beta      <- cov(port_ret, bench_ret) / (var(bench_ret) + 1e-12)
  list(tracking_error = te, information_ratio = ir,
       active_return = alpha_ann, beta = beta,
       port_sharpe = pm_p$sharpe, bench_sharpe = pm_b$sharpe,
       up_capture   = {
         u <- bench_ret > 0
         if (sum(u) > 0) mean(port_ret[u]) / mean(bench_ret[u]) else NA
       },
       down_capture = {
         d <- bench_ret < 0
         if (sum(d) > 0) mean(port_ret[d]) / mean(bench_ret[d]) else NA
       })
}

# ============================================================
# 5. TRANSACTION COST ATTRIBUTION
# ============================================================

tca_attribution <- function(trades, prices_vwap, prices_arrival,
                            prices_close, direction) {
  # direction: +1 buy, -1 sell
  implementation_shortfall <- direction * (prices_vwap - prices_arrival) /
                               prices_arrival
  market_impact     <- direction * (prices_arrival - prices_close) /
                       prices_close
  timing_cost       <- direction * (prices_close - prices_vwap) / prices_vwap
  list(
    implementation_shortfall = implementation_shortfall,
    market_impact     = market_impact,
    timing_cost       = timing_cost,
    total_cost        = implementation_shortfall + timing_cost,
    mean_IS           = mean(implementation_shortfall, na.rm = TRUE),
    mean_impact       = mean(market_impact, na.rm = TRUE)
  )
}


# ============================================================
# ADDITIONAL: RISK-ADJUSTED ATTRIBUTION
# ============================================================

appraisal_ratio <- function(alpha, tracking_error) {
  alpha / (tracking_error + 1e-8)
}

m_squared <- function(port_ret, bench_ret, rf = 0, ann = 252) {
  sharpe_p <- (mean(port_ret) - rf/ann) / sd(port_ret) * sqrt(ann)
  sharpe_b <- (mean(bench_ret) - rf/ann) / sd(bench_ret) * sqrt(ann)
  vol_b    <- sd(bench_ret) * sqrt(ann)
  list(m2 = sharpe_p * vol_b, sharpe_port = sharpe_p, sharpe_bench = sharpe_b,
       m2_minus_bench = sharpe_p * vol_b - mean(bench_ret) * ann)
}

omega_ratio <- function(returns, threshold = 0) {
  gains  <- sum(pmax(returns - threshold, 0))
  losses <- sum(pmax(threshold - returns, 0))
  gains / (losses + 1e-8)
}

pain_ratio <- function(returns) {
  cum   <- cumprod(1 + returns)
  dd    <- (cum - cummax(cum)) / cummax(cum)
  mean(abs(dd)) # Pain Index
}

# ============================================================
# ADDITIONAL: SECTOR ROTATION ATTRIBUTION
# ============================================================

sector_rotation_score <- function(port_weights_ts, bench_weights_ts,
                                   sector_returns) {
  T_   <- nrow(port_weights_ts)
  N    <- ncol(port_weights_ts)
  active_ts  <- port_weights_ts - bench_weights_ts
  rotation_score <- numeric(T_)
  for (t in seq_len(T_)) {
    rotation_score[t] <- sum(active_ts[t, ] * sector_returns[t, ])
  }
  cum_rot <- cumsum(rotation_score)
  list(rotation_return = rotation_score, cumulative = cum_rot,
       mean = mean(rotation_score), vol = sd(rotation_score),
       hit_rate = mean(rotation_score > 0))
}

sector_timing_model <- function(sector_returns, macro_indicators) {
  K  <- ncol(sector_returns); M <- ncol(macro_indicators)
  T_ <- nrow(sector_returns)
  betas <- matrix(NA, K, M)
  for (k in seq_len(K)) {
    X  <- cbind(1, macro_indicators)
    y  <- sector_returns[, k]
    b  <- tryCatch(solve(t(X) %*% X + diag(M+1) * 1e-8) %*% t(X) %*% y,
                   error = function(e) rep(0, M+1))
    betas[k, ] <- b[-1]
  }
  predicted_ranks <- apply(macro_indicators %*% t(betas), 1,
                           function(r) rank(-r))
  list(betas = betas, predicted_ranks = predicted_ranks)
}

# ============================================================
# ADDITIONAL: CRYPTO PORTFOLIO ATTRIBUTION
# ============================================================

crypto_factor_attribution <- function(returns, btc_ret, eth_ret,
                                       defi_ret = NULL, l1_ret = NULL) {
  factors <- cbind(btc_ret, eth_ret)
  fname   <- c("BTC","ETH")
  if (!is.null(defi_ret)) { factors <- cbind(factors, defi_ret); fname <- c(fname,"DeFi") }
  if (!is.null(l1_ret))   { factors <- cbind(factors, l1_ret);   fname <- c(fname,"L1") }

  X  <- cbind(1, factors)
  b  <- tryCatch(solve(t(X) %*% X + diag(ncol(X)) * 1e-8) %*% t(X) %*% returns,
                 error = function(e) rep(0, ncol(X)))
  alpha  <- b[1]; betas <- b[-1]
  fitted <- as.vector(X %*% b); resid <- returns - fitted
  r2     <- 1 - sum(resid^2) / (sum((returns - mean(returns))^2) + 1e-12)
  contrib <- betas * colMeans(factors)
  list(alpha = alpha, betas = setNames(betas, fname),
       contributions = setNames(contrib, fname),
       r2 = r2, idiosyncratic = mean(resid))
}

rolling_crypto_attribution <- function(returns, btc_ret, eth_ret,
                                        window = 60) {
  T_  <- length(returns)
  bbtc <- rep(NA, T_); beth <- rep(NA, T_); alpha <- rep(NA, T_)
  for (i in seq(window, T_)) {
    idx <- seq(i - window + 1, i)
    res <- crypto_factor_attribution(returns[idx], btc_ret[idx], eth_ret[idx])
    bbtc[i] <- res$betas["BTC"]; beth[i] <- res$betas["ETH"]
    alpha[i] <- res$alpha
  }
  list(btc_beta = bbtc, eth_beta = beth, alpha = alpha)
}

# ============================================================
# ADDITIONAL: BENCHMARK CONSTRUCTION
# ============================================================

equal_weight_benchmark <- function(returns_matrix) {
  rowMeans(returns_matrix)
}

cap_weight_benchmark <- function(returns_matrix, market_caps) {
  T_ <- nrow(returns_matrix); N <- ncol(returns_matrix)
  bret <- numeric(T_)
  for (t in seq_len(T_)) {
    w    <- market_caps[t, ] / (sum(market_caps[t, ]) + 1e-12)
    bret[t] <- sum(w * returns_matrix[t, ])
  }
  bret
}

risk_parity_benchmark <- function(returns_matrix, window = 60) {
  T_ <- nrow(returns_matrix); N <- ncol(returns_matrix)
  bret <- rep(NA, T_)
  for (t in seq(window, T_)) {
    idx <- seq(t - window + 1, t)
    vols <- apply(returns_matrix[idx, ], 2, sd) + 1e-8
    w    <- (1/vols) / sum(1/vols)
    bret[t] <- sum(w * returns_matrix[t, ])
  }
  bret
}

benchmark_attribution <- function(port_weights, bench_weights,
                                   asset_returns, bench_type = "custom") {
  rb <- sum(bench_weights * asset_returns, na.rm = TRUE)
  rp <- sum(port_weights  * asset_returns, na.rm = TRUE)
  active_w <- port_weights - bench_weights
  active_r <- rp - rb
  list(port_return = rp, bench_return = rb, active_return = active_r,
       active_weights = active_w,
       active_bet = sum(abs(active_w)) / 2)  # active share
}
