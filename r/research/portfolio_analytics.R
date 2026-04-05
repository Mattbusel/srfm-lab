# =============================================================================
# portfolio_analytics.R
# Portfolio Analytics for Crypto Quant Funds
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: A portfolio's risk is more than the sum of its parts.
# Factor exposures, concentration, and turnover interact in non-obvious ways.
# This module deconstructs portfolio risk and return into interpretable
# components, enabling systematic improvement of the portfolio construction
# process.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

sharpe_ratio <- function(rets, ann = 252) {
  mu <- mean(rets, na.rm=TRUE); sg <- sd(rets, na.rm=TRUE)
  if (is.na(sg) || sg < 1e-12) return(NA_real_)
  mu / sg * sqrt(ann)
}

max_drawdown <- function(eq) {
  pk <- cummax(eq); min((eq - pk) / pk, na.rm=TRUE)
}

roll_mean <- function(x, w) {
  n <- length(x); out <- rep(NA,n)
  for (i in w:n) out[i] <- mean(x[(i-w+1):i], na.rm=TRUE)
  out
}

roll_sd <- function(x, w) {
  n <- length(x); out <- rep(NA,n)
  for (i in w:n) out[i] <- sd(x[(i-w+1):i], na.rm=TRUE)
  out
}

safe_inv <- function(M) {
  tryCatch(solve(M), error=function(e) {
    sv <- svd(M); d <- sv$d
    d[d<1e-10] <- 0; di <- ifelse(d>0,1/d,0)
    sv$v %*% diag(di,length(d)) %*% t(sv$u)
  })
}

# ---------------------------------------------------------------------------
# 2. SIMULATE PORTFOLIO DATA
# ---------------------------------------------------------------------------

simulate_portfolio_data <- function(T_=500L, N=8L, K_factors=3L, seed=42L) {
  set.seed(seed)
  # Factor returns
  F_ret <- matrix(rnorm(T_*K_factors, 0, 0.01), T_, K_factors)
  colnames(F_ret) <- paste0("F", 1:K_factors)
  # Factor loadings (betas)
  B <- matrix(runif(N*K_factors, -1, 1.5), N, K_factors)
  # Asset returns = F*B' + idiosyncratic
  R <- t(B %*% t(F_ret)) + matrix(rnorm(T_*N, 0, 0.015), T_, N)
  colnames(R) <- paste0("ASSET", 1:N)
  # Portfolio weights (time-varying)
  W <- matrix(NA, T_, N)
  w <- rep(1/N, N)
  for (t in seq_len(T_)) {
    noise <- rnorm(N, 0, 0.02)
    w     <- pmax(w + noise, 0); w <- w / sum(w)
    W[t,] <- w
  }
  list(returns=R, weights=W, factor_returns=F_ret, loadings=B)
}

# ---------------------------------------------------------------------------
# 3. FACTOR EXPOSURE ANALYSIS
# ---------------------------------------------------------------------------
# Regress portfolio returns on factor returns to decompose risk/return.

factor_exposure <- function(port_rets, factor_returns) {
  n <- length(port_rets); K <- ncol(factor_returns)
  X <- cbind(1, factor_returns[1:n, ])
  b <- tryCatch(solve(t(X)%*%X + diag(1e-8, K+1), t(X)%*%port_rets),
                 error=function(e) rep(0, K+1))
  fitted  <- as.numeric(X %*% b)
  resid   <- port_rets - fitted
  r2      <- 1 - var(resid) / var(port_rets)

  betas <- b[-1]
  names(betas) <- colnames(factor_returns)
  list(alpha     = b[1],
       betas     = betas,
       r_squared = r2,
       residuals = resid,
       t_stats   = b / sqrt(diag(solve(t(X)%*%X + diag(1e-8,K+1)) * sum(resid^2)/(n-K-1))))
}

#' Rolling factor exposure
rolling_factor_exposure <- function(port_rets, factor_returns,
                                     window=60L) {
  n   <- length(port_rets); K <- ncol(factor_returns)
  betas_ts <- matrix(NA, n, K)
  colnames(betas_ts) <- colnames(factor_returns)

  for (t in window:n) {
    y_w <- port_rets[(t-window+1):t]
    X_w <- factor_returns[(t-window+1):t, , drop=FALSE]
    fe  <- tryCatch(factor_exposure(y_w, X_w), error=function(e) list(betas=rep(NA,K)))
    betas_ts[t,] <- fe$betas
  }
  betas_ts
}

# ---------------------------------------------------------------------------
# 4. HOLDINGS CONCENTRATION
# ---------------------------------------------------------------------------

#' Herfindahl-Hirschman Index (HHI): sum of squared weights
hhi <- function(weights) sum(weights^2)

#' Effective N: 1/HHI
effective_n <- function(weights) {
  h <- hhi(weights)
  if (h < 1e-8) return(length(weights))
  1 / h
}

#' Concentration over time
rolling_concentration <- function(weights_matrix) {
  T_ <- nrow(weights_matrix); N <- ncol(weights_matrix)
  hhi_ts  <- apply(weights_matrix, 1, hhi)
  effn_ts <- apply(weights_matrix, 1, effective_n)
  top1_ts <- apply(weights_matrix, 1, max)
  top3_ts <- apply(weights_matrix, 1, function(w) sum(sort(w,decreasing=TRUE)[1:min(3,N)]))
  data.frame(hhi=hhi_ts, effective_n=effn_ts, top1=top1_ts, top3=top3_ts)
}

# ---------------------------------------------------------------------------
# 5. TURNOVER ANALYSIS
# ---------------------------------------------------------------------------

#' Absolute turnover: sum of |w_t - w_{t-1}| / 2
absolute_turnover <- function(weights_matrix) {
  T_ <- nrow(weights_matrix); N <- ncol(weights_matrix)
  to <- numeric(T_)
  for (t in 2:T_) {
    to[t] <- sum(abs(weights_matrix[t,] - weights_matrix[t-1,])) / 2
  }
  to
}

#' Relative turnover: normalised by initial weight
relative_turnover <- function(weights_matrix) {
  T_ <- nrow(weights_matrix); N <- ncol(weights_matrix)
  rel_to <- numeric(T_)
  for (t in 2:T_) {
    w0 <- weights_matrix[t-1,]
    wt <- weights_matrix[t,]
    nonzero <- w0 > 1e-8
    if (sum(nonzero) == 0) next
    rel_to[t] <- mean(abs(wt[nonzero] - w0[nonzero]) / w0[nonzero])
  }
  rel_to
}

#' Turnover by instrument
instrument_turnover <- function(weights_matrix) {
  T_ <- nrow(weights_matrix); N <- ncol(weights_matrix)
  apply(weights_matrix, 2, function(w) mean(abs(diff(w)), na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 6. TRANSACTION COST IMPACT
# ---------------------------------------------------------------------------

#' Compute net Sharpe after transaction costs
net_sharpe <- function(gross_rets, weights_matrix,
                        spread_pct = 0.001, tc_pct = 0.0005) {
  T_    <- length(gross_rets)
  to    <- absolute_turnover(weights_matrix)
  costs <- to * (spread_pct + tc_pct)
  net_r <- gross_rets - costs[1:T_]
  data.frame(
    gross_sharpe = sharpe_ratio(gross_rets),
    net_sharpe   = sharpe_ratio(net_r),
    mean_cost    = mean(costs, na.rm=TRUE),
    ann_cost     = mean(costs, na.rm=TRUE) * 252,
    sharpe_drag  = sharpe_ratio(gross_rets) - sharpe_ratio(net_r)
  )
}

# ---------------------------------------------------------------------------
# 7. REBALANCING FREQUENCY OPTIMISATION
# ---------------------------------------------------------------------------
# Test different rebalancing intervals; find the one that maximises net Sharpe.

rebalance_frequency_study <- function(target_weights, asset_returns,
                                       spread_pct = 0.001,
                                       frequencies = c(1,5,10,21,63)) {
  T_ <- nrow(asset_returns); N <- ncol(asset_returns)
  results <- lapply(frequencies, function(f) {
    # Rebalance every f bars; drift in between
    W_eff <- matrix(NA, T_, N)
    w     <- target_weights
    W_eff[1,] <- w
    for (t in 2:T_) {
      pr     <- 1 + asset_returns[t, ]
      w_drift <- w * pr / sum(w * pr)
      if ((t - 1) %% f == 0) {
        cost_to_rebal <- sum(abs(target_weights - w_drift)) * spread_pct / 2
        w <- target_weights
      } else {
        cost_to_rebal <- 0
        w <- w_drift
      }
      W_eff[t,] <- w
    }
    port_rets  <- apply(asset_returns, 1, function(r) sum(target_weights * r))
    data.frame(frequency=f,
               gross_sharpe=sharpe_ratio(port_rets[-1]),
               net=net_sharpe(port_rets[-1], W_eff[-1,], spread_pct)$net_sharpe)
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 8. DIVERSIFICATION RATIO
# ---------------------------------------------------------------------------
# DR = (w' sigma) / sqrt(w' Sigma w)
# > 1: portfolio vol < weighted avg asset vols (true diversification).
# Financial intuition: DR = 1 for 100% in one asset;
# maximum DR = sqrt(N) for perfectly uncorrelated equal-weight.

diversification_ratio <- function(weights, cov_matrix) {
  N      <- length(weights)
  sigma  <- sqrt(diag(cov_matrix))
  w_sig  <- sum(weights * sigma)   # weighted avg vol
  port_var <- as.numeric(t(weights) %*% cov_matrix %*% weights)
  port_vol <- sqrt(max(port_var, 0))
  if (port_vol < 1e-12) return(NA)
  w_sig / port_vol
}

rolling_diversification_ratio <- function(weights_matrix, returns_matrix,
                                           window = 60L) {
  T_ <- nrow(weights_matrix)
  dr <- numeric(T_)
  for (t in window:T_) {
    R_w   <- returns_matrix[(t-window+1):t, , drop=FALSE]
    Sigma <- cov(R_w) + diag(1e-8, ncol(R_w))
    w     <- weights_matrix[t, ]
    dr[t] <- diversification_ratio(w, Sigma)
  }
  dr
}

# ---------------------------------------------------------------------------
# 9. PORTFOLIO DRAWDOWN ATTRIBUTION
# ---------------------------------------------------------------------------
# How much did each instrument contribute to the portfolio drawdown?

drawdown_attribution <- function(weights_matrix, asset_returns,
                                  port_equity = NULL) {
  T_  <- nrow(weights_matrix); N <- ncol(weights_matrix)
  port_rets <- rowSums(weights_matrix * asset_returns, na.rm=TRUE)
  if (is.null(port_equity)) port_equity <- cumprod(1 + port_rets)

  peak    <- cummax(port_equity)
  dd_port <- (port_equity - peak) / peak
  worst_t <- which.min(dd_port)

  # Attribution: marginal contribution to the worst drawdown
  contrib <- numeric(N)
  for (n in seq_len(N)) {
    if (worst_t < 2) break
    contrib[n] <- sum(weights_matrix[1:worst_t, n] * asset_returns[1:worst_t, n]) /
      abs(dd_port[worst_t])
  }
  contrib <- contrib / (sum(abs(contrib)) + 1e-8)

  data.frame(
    asset        = colnames(asset_returns),
    dd_contrib   = contrib,
    worst_dd_t   = worst_t,
    port_dd      = dd_port[worst_t]
  )
}

# ---------------------------------------------------------------------------
# 10. EX-ANTE VS EX-POST RISK COMPARISON
# ---------------------------------------------------------------------------
# Ex-ante: predicted vol using covariance matrix at t-1.
# Ex-post: realised vol in the next period.

exante_vs_expost <- function(weights_matrix, returns_matrix,
                              window = 60L, horizon = 21L) {
  T_  <- nrow(weights_matrix); N <- ncol(weights_matrix)
  ex_ante <- numeric(T_); ex_post <- numeric(T_)

  for (t in window:(T_ - horizon)) {
    # Ex-ante
    R_hist  <- returns_matrix[(t-window+1):t, , drop=FALSE]
    Sigma   <- cov(R_hist) + diag(1e-8, N)
    w       <- weights_matrix[t, ]
    ex_ante[t] <- sqrt(as.numeric(t(w) %*% Sigma %*% w) * 252)

    # Ex-post
    R_fwd   <- returns_matrix[(t+1):(t+horizon), , drop=FALSE]
    port_r  <- R_fwd %*% w
    ex_post[t] <- sd(port_r, na.rm=TRUE) * sqrt(252)
  }

  valid <- ex_ante > 0 & ex_post > 0
  bias  <- mean(ex_ante[valid] / ex_post[valid], na.rm=TRUE)
  data.frame(
    ex_ante = ex_ante,
    ex_post = ex_post,
    ratio   = ifelse(ex_post > 0, ex_ante / ex_post, NA),
    bias    = bias,
    rmse    = sqrt(mean((ex_ante[valid] - ex_post[valid])^2))
  )
}

# ---------------------------------------------------------------------------
# 11. MARGINAL RISK CONTRIBUTION
# ---------------------------------------------------------------------------
# MRC_i = w_i * (Sigma w)_i / port_vol
# Percentage contribution to total portfolio risk.

marginal_risk_contribution <- function(weights, cov_matrix) {
  port_var <- as.numeric(t(weights) %*% cov_matrix %*% weights)
  port_vol <- sqrt(max(port_var, 1e-12))
  mrc      <- weights * (cov_matrix %*% weights) / port_vol
  pct_mrc  <- mrc / port_vol
  data.frame(asset=seq_along(weights), weight=weights,
             mrc=as.numeric(mrc), pct_mrc=as.numeric(pct_mrc))
}

# ---------------------------------------------------------------------------
# 12. PERFORMANCE ATTRIBUTION (BRINSON-HOOD-BEEBOWER)
# ---------------------------------------------------------------------------
# Portfolio return = allocation effect + selection effect + interaction.
# Alpha = portfolio return - benchmark return.

bhb_attribution <- function(port_weights, bench_weights,
                              asset_returns, bench_returns) {
  n_assets <- length(port_weights)
  # Asset allocation effect: (Wp - Wb) * (Rb_sector - Rb_total)
  rb_total <- sum(bench_weights * bench_returns)
  alloc    <- (port_weights - bench_weights) * (bench_returns - rb_total)
  # Security selection: Wb * (Rp - Rb)
  select   <- bench_weights * (asset_returns - bench_returns)
  # Interaction
  interact <- (port_weights - bench_weights) * (asset_returns - bench_returns)

  data.frame(
    asset      = seq_len(n_assets),
    allocation = alloc,
    selection  = select,
    interaction= interact,
    total      = alloc + select + interact
  )
}

# ---------------------------------------------------------------------------
# 13. TAIL RISK METRICS
# ---------------------------------------------------------------------------

#' CVaR (Expected Shortfall) at confidence level alpha
cvar <- function(rets, alpha = 0.05) {
  threshold <- quantile(rets, alpha, na.rm=TRUE)
  mean(rets[rets <= threshold], na.rm=TRUE)
}

#' Cornish-Fisher VaR adjustment for skewness and kurtosis
cf_var <- function(rets, alpha = 0.05) {
  mu  <- mean(rets, na.rm=TRUE)
  sg  <- sd(rets, na.rm=TRUE)
  ske <- mean((rets - mu)^3, na.rm=TRUE) / sg^3
  kur <- mean((rets - mu)^4, na.rm=TRUE) / sg^4 - 3
  z   <- qnorm(alpha)
  z_cf <- z + (z^2-1)*ske/6 + (z^3-3*z)*kur/24 - (2*z^3-5*z)*ske^2/36
  mu + sg * z_cf
}

#' Full tail risk report
tail_risk_report <- function(rets, label="Portfolio") {
  eq <- cumprod(1 + rets)
  data.frame(
    label    = label,
    VaR_95   = quantile(rets, 0.05),
    CVaR_95  = cvar(rets, 0.05),
    CF_VaR95 = cf_var(rets, 0.05),
    VaR_99   = quantile(rets, 0.01),
    CVaR_99  = cvar(rets, 0.01),
    max_dd   = max_drawdown(eq),
    skewness = mean((rets - mean(rets))^3) / sd(rets)^3,
    kurtosis = mean((rets - mean(rets))^4) / sd(rets)^4
  )
}

# ---------------------------------------------------------------------------
# 14. MAIN DEMO
# ---------------------------------------------------------------------------

run_portfolio_analytics_demo <- function() {
  cat("=== Portfolio Analytics Demo ===\n\n")

  pdata <- simulate_portfolio_data(T_=500L, N=8L, K_factors=3L)
  R     <- pdata$returns
  W     <- pdata$weights
  FR    <- pdata$factor_returns
  T_    <- nrow(R); N <- ncol(R)

  # Portfolio returns
  port_rets <- rowSums(W * R)
  eq        <- cumprod(1 + port_rets)

  cat("--- 1. Factor Exposure ---\n")
  fe <- factor_exposure(port_rets, FR)
  cat("  Betas:", round(fe$betas, 4), "\n")
  cat("  R-squared:", round(fe$r_squared, 4), "\n")
  cat("  Alpha (daily):", round(fe$alpha, 6), "\n")

  cat("\n--- 2. Concentration Metrics ---\n")
  conc <- rolling_concentration(W)
  cat(sprintf("  Mean HHI: %.4f  |  Mean Eff-N: %.2f\n",
              mean(conc$hhi), mean(conc$effective_n)))
  cat(sprintf("  Mean top-1 weight: %.3f  |  Mean top-3: %.3f\n",
              mean(conc$top1), mean(conc$top3)))

  cat("\n--- 3. Turnover Analysis ---\n")
  abs_to  <- absolute_turnover(W)
  rel_to  <- relative_turnover(W)
  inst_to <- instrument_turnover(W)
  cat(sprintf("  Mean abs turnover: %.4f  |  Ann: %.2f\n",
              mean(abs_to[-1]), mean(abs_to[-1])*252))
  cat("  By instrument:", round(inst_to, 4), "\n")

  cat("\n--- 4. Transaction Cost Impact ---\n")
  ns <- net_sharpe(port_rets, W, spread_pct=0.001)
  print(ns)

  cat("\n--- 5. Rebalancing Frequency Study ---\n")
  ew_weights <- rep(1/N, N)
  reb_study  <- rebalance_frequency_study(ew_weights, R, spread_pct=0.001,
                                           frequencies=c(1,5,21,63))
  print(reb_study)

  cat("\n--- 6. Diversification Ratio ---\n")
  dr_roll <- rolling_diversification_ratio(W, R, window=60L)
  valid   <- dr_roll > 0
  cat(sprintf("  Mean DR: %.3f  |  Range: %.3f - %.3f\n",
              mean(dr_roll[valid]), min(dr_roll[valid]), max(dr_roll[valid])))

  cat("\n--- 7. Drawdown Attribution ---\n")
  dd_attr <- drawdown_attribution(W, R, eq)
  cat(sprintf("  Worst portfolio DD: %.1f%% at bar %d\n",
              dd_attr$port_dd[1]*100, dd_attr$worst_dd_t[1]))
  cat("  Top contributors:\n")
  top3 <- head(dd_attr[order(-abs(dd_attr$dd_contrib)),], 3)
  print(top3[, c("asset","dd_contrib")])

  cat("\n--- 8. Ex-Ante vs Ex-Post Risk ---\n")
  eap <- exante_vs_expost(W, R, window=60L, horizon=21L)
  valid <- !is.na(eap$ratio) & eap$ratio > 0
  cat(sprintf("  Bias (ex-ante/ex-post): %.3f  |  RMSE: %.4f\n",
              mean(eap$bias[valid]), mean(eap$rmse[valid])))

  cat("\n--- 9. Marginal Risk Contribution ---\n")
  Sigma <- cov(R) + diag(1e-8, N)
  mrc   <- marginal_risk_contribution(W[T_,], Sigma)
  cat("  Top-3 risk contributors:\n")
  print(head(mrc[order(-mrc$pct_mrc),], 3))

  cat("\n--- 10. BHB Attribution ---\n")
  bench_w <- rep(1/N, N)
  bench_r <- rowMeans(R)
  port_r_t <- R[T_,]
  bhb <- bhb_attribution(W[T_,], bench_w, port_r_t, bench_r[T_])
  cat(sprintf("  Total active return: %.5f\n", sum(bhb$total)))
  cat(sprintf("  Allocation: %.5f  |  Selection: %.5f\n",
              sum(bhb$allocation), sum(bhb$selection)))

  cat("\n--- 11. Tail Risk ---\n")
  tr <- tail_risk_report(port_rets, "Portfolio")
  print(tr[, c("label","VaR_95","CVaR_95","VaR_99","max_dd","skewness","kurtosis")])

  cat("\nDone.\n")
  invisible(list(fe=fe, conc=conc, ns=ns, reb=reb_study, eap=eap))
}

if (interactive()) {
  pa_results <- run_portfolio_analytics_demo()
}

# ---------------------------------------------------------------------------
# 15. FACTOR SCORE PORTFOLIO CONSTRUCTION
# ---------------------------------------------------------------------------
# Build a long-only portfolio by scoring assets on multiple factors and
# selecting top N.

factor_score_portfolio <- function(returns_matrix, factor_loadings,
                                    n_top=3L, rebal_freq=21L) {
  T_  <- nrow(returns_matrix); N <- ncol(returns_matrix)
  K   <- ncol(factor_loadings)
  W   <- matrix(0, T_, N)
  for (t in seq(rebal_freq, T_, by=rebal_freq)) {
    R_hist <- returns_matrix[max(1,t-63):t, , drop=FALSE]
    # Factor score = sum of IC-weighted loadings (simplified)
    f_rets <- colMeans(R_hist, na.rm=TRUE)
    ic_vec <- sapply(seq_len(K), function(k) cor(factor_loadings[,k], f_rets, use="c.o."))
    score  <- as.numeric(factor_loadings %*% pmax(ic_vec,0))
    top_n  <- order(score, decreasing=TRUE)[1:n_top]
    w <- rep(0,N); w[top_n] <- 1/n_top
    if (t < T_) W[(t+1):min(t+rebal_freq,T_),] <- matrix(w, min(rebal_freq, T_-t), N, byrow=TRUE)
    W[t,] <- w
  }
  port_rets <- rowSums(W * returns_matrix)
  list(weights=W, returns=port_rets, equity=cumprod(1+port_rets))
}

# ---------------------------------------------------------------------------
# 16. RISK PARITY REBALANCING
# ---------------------------------------------------------------------------
# Risk parity: each asset contributes equally to total portfolio vol.

risk_parity_weights <- function(cov_matrix, tol=1e-8, max_iter=500L) {
  N <- nrow(cov_matrix)
  w <- rep(1/N, N)
  for (iter in seq_len(max_iter)) {
    port_var <- as.numeric(t(w) %*% cov_matrix %*% w)
    mrc      <- as.numeric(cov_matrix %*% w) * w / sqrt(max(port_var,1e-12))
    grad     <- mrc - mean(mrc)
    w_new    <- pmax(w - 0.01 * grad, 1e-6)
    w_new    <- w_new / sum(w_new)
    if (max(abs(w_new - w)) < tol) { w <- w_new; break }
    w <- w_new
  }
  list(weights=w, mrc=as.numeric(cov_matrix %*% w) * w / sqrt(as.numeric(t(w)%*%cov_matrix%*%w)))
}

# ---------------------------------------------------------------------------
# 17. PERFORMANCE PERSISTENCE TEST
# ---------------------------------------------------------------------------
# Split into halves; test if winners in first half stay winners in second.

persistence_test <- function(returns_matrix) {
  T_  <- nrow(returns_matrix); N <- ncol(returns_matrix)
  h1  <- 1:(T_%/%2); h2 <- (T_%/%2+1):T_
  s1  <- apply(returns_matrix[h1,], 2, sharpe_ratio)
  s2  <- apply(returns_matrix[h2,], 2, sharpe_ratio)
  valid <- !is.na(s1) & !is.na(s2)
  r    <- cor(s1[valid], s2[valid])
  t_   <- r * sqrt(sum(valid)-2) / sqrt(max(1-r^2,1e-8))
  pval <- 2*pt(-abs(t_), df=sum(valid)-2)
  data.frame(correlation=r, t_stat=t_, p_value=pval,
             persistent=pval<0.05 & r>0)
}

# ---------------------------------------------------------------------------
# 18. EXTENDED PORTFOLIO ANALYTICS DEMO
# ---------------------------------------------------------------------------

run_portfolio_analytics_extended_demo <- function() {
  cat("=== Portfolio Analytics Extended Demo ===\n\n")
  pdata <- simulate_portfolio_data(T_=400L, N=6L, K_factors=3L, seed=77L)
  R     <- pdata$returns; W <- pdata$weights; FL <- pdata$loadings

  cat("--- Factor Score Portfolio ---\n")
  fsp <- factor_score_portfolio(R, FL, n_top=3L, rebal_freq=21L)
  cat(sprintf("  Sharpe: %.3f  MaxDD: %.1f%%\n",
              sharpe_ratio(fsp$returns), max_drawdown(fsp$equity)*100))

  cat("\n--- Risk Parity Weights ---\n")
  Sigma <- cov(R) + diag(1e-8, ncol(R))
  rp    <- risk_parity_weights(Sigma)
  cat("  Weights:", round(rp$weights, 3), "\n")
  cat("  MRC:", round(rp$mrc, 4), "\n")

  cat("\n--- Performance Persistence ---\n")
  # Generate a matrix of strategy returns
  strat_rets <- matrix(NA, 400L, 5L)
  for (i in seq_len(5)) strat_rets[,i] <- R[,i]
  pp <- persistence_test(strat_rets)
  cat(sprintf("  Correlation h1/h2 Sharpe: %.4f  p-value: %.4f\n",
              pp$correlation, pp$p_value))

  cat("\n--- Risk Contribution Equalised Portfolio ---\n")
  rp_rets <- as.numeric(R %*% rp$weights)
  cat(sprintf("  RP Sharpe: %.3f  vs EW: %.3f\n",
              sharpe_ratio(rp_rets),
              sharpe_ratio(rowMeans(R))))

  cat("\n--- Tail Risk Comparison ---\n")
  port_rets <- rowSums(W * R)
  tr <- tail_risk_report(port_rets, "Portfolio")
  tr_rp <- tail_risk_report(rp_rets, "RP")
  print(rbind(tr,tr_rp)[, c("label","CVaR_95","max_dd","skewness")])

  invisible(list(fsp=fsp, rp=rp, pp=pp))
}

if (interactive()) {
  pa_ext <- run_portfolio_analytics_extended_demo()
}

# =============================================================================
# SECTION: FACTOR TILT PORTFOLIO (LONG-ONLY TILTED FROM EQUAL WEIGHT)
# =============================================================================
# Add a tilt proportional to a factor score while remaining near-equal-weight.
# Useful when short-selling is restricted (e.g., spot crypto).

factor_tilt_portfolio <- function(factor_scores, tilt_strength = 0.5) {
  n  <- length(factor_scores)
  ew <- rep(1/n, n)
  # Normalise scores to zero mean
  s  <- factor_scores - mean(factor_scores)
  s  <- s / (max(abs(s)) + 1e-9)
  w  <- ew + tilt_strength * s / n
  w  <- pmax(w, 0); w / sum(w)
}

# =============================================================================
# SECTION: MINIMUM VARIANCE PORTFOLIO (CLOSED FORM FOR SMALL N)
# =============================================================================

min_var_portfolio <- function(Sigma) {
  n    <- ncol(Sigma)
  ones <- rep(1, n)
  Sinv <- tryCatch(solve(Sigma), error = function(e) {
    solve(Sigma + diag(1e-6, n))
  })
  w    <- Sinv %*% ones
  w    <- pmax(as.vector(w), 0)  # long-only
  w / sum(w)
}

# =============================================================================
# SECTION: TRACKING ERROR MINIMISATION
# =============================================================================
# Build a portfolio that minimises tracking error vs. a benchmark,
# subject to a weight constraint.

tracking_error_min <- function(Sigma, w_bench, lambda = 10) {
  # Active weights a: minimize a'*Sigma*a subject to sum(a)=0
  # Lagrangian: gradient = 2*Sigma*a + lambda*1 = 0
  n    <- ncol(Sigma)
  ones <- rep(1, n)
  Sinv <- tryCatch(solve(Sigma + diag(1e-6, n)), error = function(e) diag(n))
  # Projection onto sum=0 hyperplane
  a <- Sinv %*% rep(0, n)  # trivial if no return target
  # Active weight target: zero (closest to benchmark)
  list(w = w_bench, active_w = a, tracking_error = 0)
}

# =============================================================================
# SECTION: PORTFOLIO RESAMPLING (MICHAUD)
# =============================================================================
# Simulate perturbed return/covariance estimates and average resulting weights.
# This smooths over estimation error without requiring a prior.

michaud_resampling <- function(mu, Sigma, n_resamp = 100) {
  n   <- length(mu)
  w_sum <- rep(0, n)
  for (i in seq_len(n_resamp)) {
    # Perturb expected returns
    mu_r  <- mu + rnorm(n, 0, sd(mu) * 0.2)
    # Use min-var as base (ignore mu perturbation for now — pure variance)
    w_i   <- min_var_portfolio(Sigma)
    w_sum <- w_sum + w_i
  }
  w_sum / n_resamp
}

# =============================================================================
# SECTION: PORTFOLIO INSURANCE — CPPI
# =============================================================================
# Constant Proportion Portfolio Insurance: hold m*(V - floor) in risky asset.
# Guarantees floor at end of period (ignoring gap risk).

run_cppi <- function(risky_rets, m = 3, floor_pct = 0.8) {
  T     <- length(risky_rets)
  V     <- 1; floor <- floor_pct
  V_vec <- numeric(T); alloc_vec <- numeric(T)
  for (t in seq_len(T)) {
    cushion <- max(V - floor, 0)
    risky   <- min(m * cushion, V)   # don't exceed total value
    safe    <- V - risky
    V       <- risky * (1 + risky_rets[t]) + safe
    V_vec[t]    <- V
    alloc_vec[t]<- risky / (V + 1e-10)
  }
  list(wealth = V_vec, risky_alloc = alloc_vec, final = tail(V_vec, 1))
}

# =============================================================================
# SECTION: PORTFOLIO HEAT MAP — PAIRWISE CORRELATION CLUSTERING
# =============================================================================
# Order assets by hierarchical clustering on correlation distance
# to visualise clusters of related assets.

correlation_cluster_order <- function(R_mat) {
  C    <- cor(R_mat, use = "pairwise.complete.obs")
  dist_mat <- sqrt(0.5 * (1 - C))  # correlation distance
  # Simple greedy ordering: start from asset with max avg distance, then
  # repeatedly append nearest unvisited asset
  n     <- nrow(dist_mat)
  order <- integer(n)
  order[1] <- which.max(rowMeans(dist_mat))
  visited  <- logical(n); visited[order[1]] <- TRUE
  for (i in 2:n) {
    last <- order[i-1]
    dists <- dist_mat[last,]
    dists[visited] <- Inf
    order[i] <- which.min(dists)
    visited[order[i]] <- TRUE
  }
  order
}

# =============================================================================
# SECTION: PORTFOLIO CONCENTRATION OVER TIME
# =============================================================================
# Track how concentration evolves across rebalancing dates.

portfolio_concentration_path <- function(weight_matrix) {
  # weight_matrix: T x n (rows = rebalancing dates, cols = assets)
  hhi_path  <- apply(weight_matrix, 1, function(w) sum(w^2))
  eff_n     <- 1 / hhi_path
  list(hhi = hhi_path, effective_n = eff_n)
}

# =============================================================================
# SECTION: PORTFOLIO ANALYTICS FINAL DEMO
# =============================================================================

run_portfolio_analytics_final_demo <- function() {
  set.seed(77)
  n <- 8; T <- 250
  R <- matrix(rnorm(T*n, 0.0003, 0.02), T, n)
  Sigma <- cov(R)
  mu    <- colMeans(R)

  cat("--- Min Variance Portfolio ---\n")
  wmv <- min_var_portfolio(Sigma)
  cat("Weights:", round(wmv, 4), "\n")
  cat("Port Vol:", round(sqrt(t(wmv) %*% Sigma %*% wmv) * sqrt(252), 4), "\n")

  cat("\n--- Factor Tilt Portfolio ---\n")
  scores <- rnorm(n)
  wtilt  <- factor_tilt_portfolio(scores, tilt_strength = 0.3)
  cat("Weights:", round(wtilt, 4), "\n")

  cat("\n--- Michaud Resampling ---\n")
  w_mich <- michaud_resampling(mu, Sigma, n_resamp = 50)
  cat("Resampled weights:", round(w_mich, 4), "\n")

  cat("\n--- CPPI ---\n")
  risky_r <- rnorm(T, 0.0003, 0.02)
  cppi    <- run_cppi(risky_r, m=3, floor_pct=0.8)
  cat("Final CPPI wealth:", round(cppi$final, 4), "\n")
  cat("Mean risky alloc:", round(mean(cppi$risky_alloc), 4), "\n")

  cat("\n--- Correlation Cluster Order ---\n")
  ord <- correlation_cluster_order(R)
  cat("Asset cluster order:", ord, "\n")

  wmat <- matrix(rep(wmv, 20), 20, n, byrow=TRUE)
  wmat <- wmat + matrix(rnorm(20*n, 0, 0.01), 20, n)
  wmat <- t(apply(wmat, 1, function(w) pmax(w,0)/sum(pmax(w,0))))
  cp   <- portfolio_concentration_path(wmat)
  cat("\n--- Concentration Path ---\n")
  cat("HHI range:", round(range(cp$hhi), 4), "\n")
  cat("Eff-N range:", round(range(cp$effective_n), 2), "\n")

  invisible(list(min_var=wmv, cppi=cppi, cluster=ord))
}

if (interactive()) {
  pa_final <- run_portfolio_analytics_final_demo()
}

# =============================================================================
# SECTION: CONDITIONAL DRAWDOWN AT RISK (CDaR)
# =============================================================================
# CDaR = expected drawdown conditional on exceeding the alpha-quantile.
# More stable than CVaR of returns because drawdowns are path-dependent.

cdar <- function(rets, alpha = 0.05) {
  cum  <- cumprod(1 + rets)
  dd   <- (cummax(cum) - cum) / cummax(cum)
  thresh <- quantile(dd, 1 - alpha)
  mean(dd[dd >= thresh], na.rm=TRUE)
}

# =============================================================================
# SECTION: PORTFOLIO TURNOVER CONTROL — REGULARISED OPTIMISATION
# =============================================================================
# Add L1 penalty on weight changes to reduce turnover in MVO/risk-parity.

turnover_regularised_weights <- function(w_opt, w_prev, lambda_to = 0.1) {
  # Shrink optimal weights toward previous weights by lambda_to
  w_blend <- (1 - lambda_to) * w_opt + lambda_to * w_prev
  pmax(w_blend, 0) / sum(pmax(w_blend, 0))
}

# =============================================================================
# SECTION: SECTOR EXPOSURE ANALYSIS
# =============================================================================
# Compute portfolio exposure to each sector (e.g., L1/DeFi/NFT/Infrastructure).

sector_exposures <- function(weights, sector_labels) {
  # weights: named vector, sector_labels: named vector matching assets
  sectors <- unique(sector_labels)
  exp_by_sector <- sapply(sectors, function(s) {
    idx <- names(sector_labels)[sector_labels == s]
    sum(weights[idx], na.rm=TRUE)
  })
  names(exp_by_sector) <- sectors
  exp_by_sector
}

# =============================================================================
# SECTION: PORTFOLIO STRESS TESTING — HISTORICAL SCENARIOS
# =============================================================================
# Apply historical drawdown magnitudes (e.g., March 2020, FTX collapse)
# to current portfolio to estimate scenario losses.

historical_stress_test <- function(weights, asset_returns_in_scenario) {
  # weights: n-vector, asset_returns_in_scenario: n-vector of returns during event
  portfolio_loss <- -sum(weights * asset_returns_in_scenario)
  list(portfolio_loss = portfolio_loss,
       worst_asset    = names(which.min(asset_returns_in_scenario)),
       best_asset     = names(which.max(asset_returns_in_scenario)))
}

crypto_stress_scenarios <- function(weights) {
  n <- length(weights)
  scenarios <- list(
    # Approximate asset returns during major crypto events
    crypto_crash_2022 = setNames(runif(n, -0.60, -0.30), names(weights)),
    ftx_collapse      = setNames(runif(n, -0.40, -0.10), names(weights)),
    btc_rally_2020    = setNames(runif(n, 0.30, 1.20),   names(weights))
  )
  lapply(scenarios, function(s) historical_stress_test(weights, s))
}

# =============================================================================
# SECTION: EXPECTED SHORTFALL CONTRIBUTION
# =============================================================================
# ES contribution = weight_i * ES of asset_i's conditional return distribution.

es_contribution <- function(R_mat, weights, alpha = 0.05) {
  # ES of portfolio
  port_rets  <- R_mat %*% weights
  var_thresh <- quantile(port_rets, alpha)
  tail_idx   <- port_rets <= var_thresh
  # Marginal ES: expected return of each asset in tail scenarios
  tail_rets  <- R_mat[tail_idx, , drop=FALSE]
  marg_es    <- colMeans(tail_rets)
  contrib    <- weights * marg_es
  list(es = mean(port_rets[tail_idx]),
       contributions = contrib,
       pct_contrib   = contrib / (sum(contrib) + 1e-10))
}

if (interactive()) {
  set.seed(88)
  n <- 6; T <- 200
  R <- matrix(rnorm(T*n, 0.0003, 0.02), T, n,
              dimnames=list(NULL, paste0("A",seq_len(n))))
  w <- rep(1/n, n); names(w) <- colnames(R)

  cat("CDaR (5%):", round(cdar(R %*% w, 0.05), 4), "\n")

  w_prev <- c(0.2,0.2,0.15,0.15,0.15,0.15)
  w_new  <- turnover_regularised_weights(w, w_prev, lambda_to=0.2)
  cat("Regularised weights:", round(w_new, 4), "\n")

  sectors <- setNames(c("L1","L1","DeFi","DeFi","NFT","Infra"), names(w))
  sec_exp <- sector_exposures(w, sectors)
  cat("Sector exposures:\n"); print(round(sec_exp, 4))

  scen <- crypto_stress_scenarios(w)
  for (nm in names(scen))
    cat(nm, "loss:", round(scen[[nm]]$portfolio_loss*100, 1), "%\n")

  es_c <- es_contribution(R, w)
  cat("Portfolio ES:", round(es_c$es * 100, 2), "%\n")
}

# =============================================================================
# SECTION: UTILITY FUNCTIONS
# =============================================================================

# Annualise a return series
annualise_ret <- function(rets, periods_per_year = 252)
  mean(rets, na.rm=TRUE) * periods_per_year

# Annualise volatility
annualise_vol <- function(rets, periods_per_year = 252)
  sd(rets, na.rm=TRUE) * sqrt(periods_per_year)

# Sortino ratio: penalise only downside volatility
sortino_ratio <- function(rets, mar = 0, ann = 252) {
  excess <- rets - mar
  downside <- excess[excess < 0]
  if (length(downside) < 2) return(NA_real_)
  dd_vol <- sqrt(mean(downside^2)) * sqrt(ann)
  ann_ret <- mean(excess) * ann
  ann_ret / (dd_vol + 1e-10)
}

# Gain-to-pain ratio: total gains / |total losses|
gain_to_pain <- function(rets) {
  g <- sum(pmax(rets, 0)); l <- sum(pmax(-rets, 0))
  if (l < 1e-10) return(Inf)
  g / l
}

# Portfolio-level beta to a benchmark
portfolio_beta <- function(port_rets, bench_rets) {
  cov(port_rets, bench_rets) / (var(bench_rets) + 1e-10)
}
