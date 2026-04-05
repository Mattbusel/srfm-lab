## systemic_risk.R
## Contagion, CoVaR, SRISK, network centrality -- crypto market focus
## Pure base R -- no library() calls

.qreg <- function(y, X, tau, max_iter = 300, tol = 1e-8) {
  beta <- rep(0, ncol(X))
  for (iter in seq_len(max_iter)) {
    r <- y - X %*% beta
    w <- ifelse(r > 0, tau, 1 - tau) + 1e-8
    b <- tryCatch(
      solve(t(X) %*% diag(as.vector(w)) %*% X) %*%
        (t(X) %*% diag(as.vector(w)) %*% y),
      error = function(e) beta)
    if (sum(abs(b - beta)) < tol) { beta <- b; break }
    beta <- b
  }
  as.vector(beta)
}

compute_covar <- function(r_sys, r_inst, state_vars, tau = 0.05) {
  X   <- cbind(1, r_inst, state_vars)
  Xi  <- cbind(1, state_vars)
  bq  <- .qreg(r_sys,  X,  tau)
  bm  <- .qreg(r_sys,  X,  0.5)
  biq <- .qreg(r_inst, Xi, tau)
  bim <- .qreg(r_inst, Xi, 0.5)
  var_q   <- as.vector(Xi %*% biq)
  var_med <- as.vector(Xi %*% bim)
  cov_q   <- as.vector(X[,-2] %*% bq[-2]) + bq[2] * var_q
  cov_med <- as.vector(X[,-2] %*% bm[-2]) + bm[2] * var_med
  list(CoVaR = cov_q, DeltaCoVaR = cov_q - cov_med,
       mean_DCoVaR = mean(cov_q - cov_med))
}

compute_mes <- function(r_inst, r_sys, alpha = 0.05) {
  thr <- quantile(r_sys, alpha)
  ci  <- which(r_sys <= thr)
  rho <- cor(r_inst, r_sys); si <- sd(r_inst); ss <- sd(r_sys)
  list(MES_emp = mean(r_inst[ci]),
       MES_anal = rho * si / ss * (-ss * dnorm(qnorm(alpha)) / alpha),
       n_crisis = length(ci))
}

compute_srisk <- function(mes, equity, debt, k = 0.08) {
  lrmes <- pmin(pmax(1 - exp(18 * mes), 0), 1)
  srisk <- pmax(k * debt - (1-k) * equity * (1-lrmes), 0)
  list(SRISK = srisk, LRMES = lrmes,
       pct = srisk / (sum(srisk)+1e-12), total = sum(srisk))
}

var_ols <- function(Y, p) {
  T_ <- nrow(Y); N <- ncol(Y)
  Ydep <- Y[(p+1):T_,]
  X    <- cbind(1, do.call(cbind, lapply(1:p, function(l) Y[(p+1-l):(T_-l),])))
  B    <- tryCatch(solve(t(X)%*%X + diag(ncol(X))*1e-6) %*% t(X) %*% Ydep,
                   error = function(e) matrix(0, ncol(X), N))
  E    <- Ydep - X %*% B
  list(B=B, Sigma=t(E)%*%E/(nrow(E)-ncol(X)),
       A=lapply(1:p, function(l) t(B[2:(N+1)+(l-1)*N,])))
}

spillover_index <- function(R, p = 2, H = 10) {
  N <- ncol(R); vr <- var_ols(R, p)
  P <- tryCatch(t(chol(vr$Sigma + diag(N)*1e-8)), error=function(e) diag(N))
  comp <- matrix(0, N*p, N*p)
  for (l in 1:p) comp[1:N, (l-1)*N+1:N] <- vr$A[[l]]
  if (p>1) comp[(N+1):(N*p), 1:(N*(p-1))] <- diag(N*(p-1))
  FEVD <- matrix(0,N,N); denom <- rep(0,N); cp <- diag(N*p)
  for (h in 1:H) {
    cp <- cp %*% comp; Ph <- cp[1:N,1:N]
    for (i in 1:N) {
      ei <- rep(0,N); ei[i] <- 1
      for (j in 1:N) {
        ej <- rep(0,N); ej[j] <- 1
        FEVD[i,j] <- FEVD[i,j] + (t(ei) %*% Ph %*% P %*% ej)^2
      }
      denom[i] <- denom[i] + t(ei) %*% Ph %*% vr$Sigma %*% Ph %*% ei
    }
  }
  for (i in 1:N) FEVD[i,] <- FEVD[i,]/(denom[i]+1e-12)
  list(FEVD=FEVD,
       total=(sum(FEVD)-sum(diag(FEVD)))/N*100,
       from=colSums(FEVD)-diag(FEVD),
       to=rowSums(FEVD)-diag(FEVD),
       net=rowSums(FEVD)-colSums(FEVD))
}

rolling_spillover <- function(R, window=200, p=2, H=10) {
  T_<-nrow(R); out<-rep(NA_real_,T_)
  for (i in seq(window,T_)) {
    res <- tryCatch(spillover_index(R[seq(i-window+1,i),],p,H), error=function(e)NULL)
    if (!is.null(res)) out[i] <- res$total
  }
  out
}

absorb_ratio <- function(R, nc=NULL) {
  N <- ncol(R); if(is.null(nc)) nc <- max(1L,floor(N/5))
  ev <- eigen(cov(R), only.values=TRUE)$values
  sum(ev[1:nc])/sum(ev)
}

network_centrality <- function(corr_matrix) {
  adj <- abs(corr_matrix); diag(adj) <- 0
  ev  <- Re(eigen(adj)$vectors[,1])
  list(degree=rowSums(adj),
       eigenvector=ev/(max(abs(ev))+1e-12),
       strength=rowSums(adj * abs(corr_matrix)))
}

contagion_test_dcc <- function(r1, r2, pre_end) {
  pre <- seq_len(pre_end); cri <- seq(pre_end+1, length(r1))
  rho_pre <- cor(r1[pre], r2[pre])
  rho_cri <- cor(r1[cri], r2[cri])
  delta   <- var(r1[cri]) / var(r1[pre])
  rho_adj <- rho_cri / sqrt(1 + delta*(1-rho_cri^2)/(1+rho_cri^2)*(delta-1)+1e-12)
  list(rho_pre=rho_pre, rho_crisis=rho_cri, rho_adjusted=rho_adj,
       contagion=rho_adj > rho_pre)
}

systemic_risk_index <- function(R, window=252) {
  T_<-nrow(R); sri<-rep(NA_real_,T_)
  for (i in seq(window,T_)) {
    idx <- seq(i-window+1,i); r <- R[idx,]
    ar  <- absorb_ratio(r)
    pr  <- rowMeans(r); es <- -mean(pr[pr<=quantile(pr,0.05)])
    mc  <- mean(cor(r)[upper.tri(cor(r))])
    sri[i] <- ar*mc*es
  }
  sri
}

tail_dependence <- function(r1, r2, alpha=0.05) {
  n   <- length(r1)
  u1  <- rank(r1)/n; u2 <- rank(r2)/n
  thr <- alpha
  lower_td <- mean(u1 <= thr & u2 <= thr) / thr
  upper_td <- mean(u1 >= 1-thr & u2 >= 1-thr) / thr
  list(lower=lower_td, upper=upper_td)
}

joint_failure_prob <- function(R, alpha=0.05, n_sim=10000, seed=42) {
  set.seed(seed); N<-ncol(R); mu<-colMeans(R)
  Sig<-cov(R); thr<-apply(R,2,quantile,alpha)
  L<-t(chol(Sig+diag(N)*1e-8))
  sim<-t(L%*%matrix(rnorm(N*n_sim),N,n_sim))+matrix(mu,n_sim,N,byrow=TRUE)
  fail<-sim < matrix(thr,n_sim,N,byrow=TRUE)
  kf<-rowSums(fail)
  list(prob_any=mean(kf>0), prob_all=mean(kf==N),
       expected=mean(kf), dist=table(kf)/n_sim)
}


# ============================================================
# ADDITIONAL: CRYPTO-SPECIFIC SYSTEMIC RISK
# ============================================================

crypto_systemic_index <- function(returns_matrix, btc_dominance,
                                   stablecoin_supply, window = 30) {
  T_   <- nrow(returns_matrix)
  port_ret <- rowMeans(returns_matrix)
  corr_stress <- rep(NA, T_)
  vol_stress  <- rep(NA, T_)
  for (i in seq(window, T_)) {
    idx <- seq(i-window+1, i)
    corr_stress[i] <- mean(cor(returns_matrix[idx,])[upper.tri(
      cor(returns_matrix[idx,]))])
    vol_stress[i]  <- sd(port_ret[idx]) * sqrt(365)
  }
  # Composite: high correlation + high vol + high BTC dominance
  sri <- (corr_stress / max(corr_stress, na.rm=TRUE) +
          vol_stress  / max(vol_stress, na.rm=TRUE) +
          btc_dominance / max(btc_dominance, na.rm=TRUE)) / 3
  list(sri=sri, corr_stress=corr_stress, vol_stress=vol_stress,
       regime=ifelse(sri>quantile(sri,.8,na.rm=TRUE),"high",
               ifelse(sri<quantile(sri,.2,na.rm=TRUE),"low","normal")))
}

defi_contagion_risk <- function(protocol_tvls, protocol_returns,
                                 window = 30) {
  T_ <- nrow(protocol_returns); N <- ncol(protocol_returns)
  contagion <- rep(NA, T_)
  for (i in seq(window, T_)) {
    idx <- seq(i-window+1, i)
    C   <- cor(protocol_returns[idx,], use="pairwise.complete.obs")
    wts <- protocol_tvls[i,] / (sum(protocol_tvls[i,])+1e-8)
    contagion[i] <- sum(wts %o% wts * C) - sum(wts^2)  # weighted avg off-diag corr
  }
  list(contagion_index=contagion,
       high_contagion=contagion>quantile(contagion,.8,na.rm=TRUE))
}

# ============================================================
# ADDITIONAL: MACRO-CRYPTO LINKAGE
# ============================================================

crypto_macro_beta <- function(crypto_returns, macro_factor, window = 60) {
  n <- length(crypto_returns); beta <- rep(NA, n)
  for (i in seq(window, n)) {
    idx    <- seq(i-window+1, i)
    r      <- crypto_returns[idx]; m <- macro_factor[idx]
    if (var(m) > 1e-10) beta[i] <- cov(r, m) / var(m)
  }
  list(beta=beta, mean_beta=mean(beta,na.rm=TRUE),
       rising_beta=mean(beta[!is.na(beta)]>0))
}

liquidity_spiral_risk <- function(returns_matrix, volumes_matrix, window=21) {
  T_ <- nrow(returns_matrix); N <- ncol(returns_matrix)
  lsr <- rep(NA, T_)
  for (i in seq(window, T_)) {
    idx <- seq(i-window+1, i)
    corr_rv <- sapply(seq_len(N), function(j)
      cor(abs(returns_matrix[idx,j]), 1/(volumes_matrix[idx,j]+1e-8),
          use="complete.obs"))
    lsr[i] <- mean(corr_rv, na.rm=TRUE)
  }
  list(spiral_index=lsr,
       alert=lsr>quantile(lsr,.8,na.rm=TRUE))
}

# ============================================================
# ADDITIONAL: MEASURES
# ============================================================

marginal_contribution_to_systemic_risk <- function(returns_matrix, weights,
                                                     alpha = 0.05) {
  N     <- ncol(returns_matrix)
  sys_r <- as.vector(returns_matrix %*% weights)
  thr   <- quantile(sys_r, alpha)
  ci    <- which(sys_r <= thr)
  mcsr  <- apply(returns_matrix[ci,], 2, mean) * weights
  list(MCSR = mcsr, total = sum(mcsr),
       pct = mcsr / (sum(mcsr)+1e-12))
}

systemic_event_probability <- function(returns_matrix, alpha = 0.05,
                                        k_min = 3, n_sim = 10000, seed = 42) {
  set.seed(seed); N <- ncol(returns_matrix); mu <- colMeans(returns_matrix)
  Sig <- cov(returns_matrix); thr <- apply(returns_matrix,2,quantile,alpha)
  L   <- t(chol(Sig+diag(N)*1e-8))
  sim <- t(L %*% matrix(rnorm(N*n_sim),N,n_sim)) + matrix(mu,n_sim,N,byrow=TRUE)
  kf  <- rowSums(sim < matrix(thr,n_sim,N,byrow=TRUE))
  list(prob_k_fail = table(kf)/n_sim,
       prob_systemic = mean(kf >= k_min),
       expected_failures = mean(kf))
}

cross_market_var <- function(returns_list, weights_list, alpha = 0.05) {
  K <- length(returns_list)
  combined <- Reduce("+", mapply(function(r, w) r * w, returns_list, weights_list,
                                  SIMPLIFY = FALSE))
  var_indiv <- sapply(returns_list, function(r) quantile(r, alpha))
  var_port  <- quantile(combined, alpha)
  list(var_portfolio = var_port,
       var_individual = var_indiv,
       diversification_benefit = sum(var_indiv) - var_port)
}

# ============================================================
# ADDITIONAL: STRESS INDICATORS
# ============================================================

financial_stress_index <- function(vol_index, credit_spread, ted_spread,
                                    equity_corr, window = 20) {
  norm <- function(x) {
    mn <- mean(x, na.rm=TRUE); sd_ <- sd(x, na.rm=TRUE)
    (x - mn) / (sd_ + 1e-8)
  }
  fsi <- (norm(vol_index) + norm(credit_spread) + norm(ted_spread) + norm(equity_corr)) / 4
  n   <- length(fsi)
  ma  <- as.numeric(stats::filter(fsi, rep(1/window, window), sides=1))
  list(fsi=fsi, smoothed=ma,
       crisis=fsi>2, elevated=fsi>1,
       percentile=rank(fsi,na.last="keep")/sum(!is.na(fsi)))
}

regime_transition_probability <- function(fsi, n_regimes=3, window=60) {
  n  <- length(fsi)
  breaks <- quantile(fsi, seq(0,1,1/n_regimes), na.rm=TRUE)
  regime <- as.numeric(cut(fsi, breaks, labels=FALSE, include.lowest=TRUE))
  trans  <- matrix(0, n_regimes, n_regimes)
  for (i in seq_len(n-1)) {
    if (!is.na(regime[i]) && !is.na(regime[i+1]))
      trans[regime[i], regime[i+1]] <- trans[regime[i], regime[i+1]] + 1
  }
  trans_prob <- trans / (rowSums(trans) + 1e-8)
  list(regime=regime, transition_matrix=trans_prob,
       stationary=tryCatch({
         ev <- eigen(t(trans_prob)); Re(ev$vectors[,1])/sum(Re(ev$vectors[,1]))
       }, error=function(e) rep(1/n_regimes, n_regimes)))
}

# ============================================================
# ADDITIONAL: CRYPTO CONTAGION CHANNELS
# ============================================================
stable_depeg_contagion <- function(stablecoin_prices, asset_returns,
                                    crisis_threshold = 0.99) {
  depeg_events <- stablecoin_prices < crisis_threshold
  crisis_rets  <- asset_returns[depeg_events,]
  normal_rets  <- asset_returns[!depeg_events,]
  list(
    crisis_mean  = if(nrow(crisis_rets)>0) colMeans(crisis_rets) else rep(NA,ncol(asset_returns)),
    normal_mean  = colMeans(normal_rets),
    vol_ratio    = apply(crisis_rets,2,sd)/(apply(normal_rets,2,sd)+1e-8),
    n_crisis     = sum(depeg_events)
  )
}

leverage_cascade_risk <- function(positions, collateral, prices,
                                   liquidation_threshold = 0.8) {
  ltvs      <- positions / (collateral * prices + 1e-8)
  at_risk   <- ltvs > liquidation_threshold
  cascade_vol <- sum(positions[at_risk])
  list(ltvs=ltvs, at_risk=at_risk,
       cascade_volume=cascade_vol,
       systemic_pct=cascade_vol/sum(positions))
}

exchange_failure_impact <- function(exchange_market_share,
                                     returns_matrix, failed_exchange) {
  # Estimate market impact if an exchange fails
  vol_loss <- exchange_market_share[failed_exchange]
  liquidity_impact <- -vol_loss * 0.5  # rough impact
  n <- ncol(returns_matrix)
  contagion_rets <- returns_matrix + liquidity_impact
  list(vol_loss_pct=vol_loss*100, liquidity_impact=liquidity_impact,
       expected_market_drawdown=liquidity_impact*0.5)
}

# ============================================================
# ADDITIONAL: REGULATORY RISK
# ============================================================
regulatory_risk_index <- function(jurisdictions, enforcement_actions,
                                   legislation_risk, adoption_metrics) {
  norm <- function(x) (x-min(x,na.rm=TRUE))/(max(x,na.rm=TRUE)-min(x,na.rm=TRUE)+1e-8)
  composite <- (norm(enforcement_actions) + norm(legislation_risk)) / 2
  adoption_factor <- 1 - norm(adoption_metrics) * 0.5
  list(risk_index=composite*adoption_factor,
       jurisdictions=jurisdictions,
       high_risk=jurisdictions[composite*adoption_factor>.7])
}

# ============================================================
# ADDITIONAL: MARKET DEPTH SYSTEMIC METRICS
# ============================================================
market_resilience_index <- function(bid_depths, ask_depths, spread_series,
                                     volume_series, window=20) {
  n    <- length(spread_series)
  depth_imb <- (bid_depths-ask_depths)/(bid_depths+ask_depths+1e-8)
  norm_sprd <- spread_series/mean(spread_series,na.rm=TRUE)
  norm_vol  <- volume_series/mean(volume_series,na.rm=TRUE)
  mri <- (1/norm_sprd) * norm_vol * (1-abs(depth_imb))
  mri_ma <- as.numeric(stats::filter(mri, rep(1/window,window), sides=1))
  list(mri=mri, smoothed=mri_ma,
       fragile=mri<quantile(mri,.2,na.rm=TRUE))
}

order_book_depth_systemic <- function(depth_matrix, threshold_pct=0.02) {
  # depth_matrix: T x N (time x asset)
  T_ <- nrow(depth_matrix); N <- ncol(depth_matrix)
  thin_count <- rowSums(depth_matrix < threshold_pct * colMeans(depth_matrix))
  list(thin_market_count=thin_count,
       systemic_thin=thin_count/N > 0.5,
       mean_depth=rowMeans(depth_matrix))
}


# ============================================================
# ADDITIONAL SYSTEMIC RISK MEASURES
# ============================================================

entropy_based_risk <- function(returns_mat) {
  cov_mat  <- cov(returns_mat, use = "pairwise.complete.obs")
  eig_vals <- eigen(cov_mat, only.values = TRUE)$values
  eig_vals <- pmax(eig_vals, 0)
  props    <- eig_vals / sum(eig_vals)
  entropy  <- -sum(props * log(props + 1e-12))
  max_ent  <- log(length(props))
  list(eigenvalues = eig_vals, proportions = props,
       entropy = entropy, normalized_entropy = entropy / max_ent,
       diversification_ratio = entropy / max_ent)
}

conditional_correlation_risk <- function(returns_mat, quantile_thr = 0.1) {
  n_asset <- ncol(returns_mat)
  tail_corr <- matrix(NA, n_asset, n_asset)
  for (i in seq_len(n_asset)) {
    for (j in seq_len(n_asset)) {
      if (i != j) {
        thr_i <- quantile(returns_mat[, i], quantile_thr, na.rm = TRUE)
        thr_j <- quantile(returns_mat[, j], quantile_thr, na.rm = TRUE)
        both_tail <- returns_mat[, i] < thr_i & returns_mat[, j] < thr_j
        if (sum(both_tail, na.rm=TRUE) > 5)
          tail_corr[i,j] <- cor(returns_mat[both_tail, i],
                                 returns_mat[both_tail, j],
                                 use = "pairwise.complete.obs")
      }
    }
  }
  diag(tail_corr) <- 1
  list(tail_correlation = tail_corr,
       mean_tail_corr = mean(tail_corr[lower.tri(tail_corr)], na.rm=TRUE))
}

network_contagion_model <- function(adjacency_mat, default_prob,
                                     recovery_rate = 0.4, rounds = 10) {
  n      <- nrow(adjacency_mat)
  status <- rbinom(n, 1, default_prob)
  for (r in seq_len(rounds)) {
    exposure <- as.vector(adjacency_mat %*% status)
    new_def  <- (1 - status) * (exposure > (1 - recovery_rate))
    status   <- pmin(status + new_def, 1)
  }
  list(final_defaults = status,
       total_defaults = sum(status),
       cascade_ratio = sum(status) / n)
}

var_components_decomp <- function(returns, weights, alpha = 0.05) {
  port_ret  <- as.vector(returns %*% weights)
  port_var  <- quantile(port_ret, alpha, na.rm = TRUE)
  cov_mat   <- cov(returns, use = "pairwise.complete.obs")
  sigma_p   <- sqrt(as.numeric(t(weights) %*% cov_mat %*% weights))
  marg_var  <- as.vector(cov_mat %*% weights) / (sigma_p + 1e-12)
  comp_var  <- weights * marg_var
  pct_var   <- comp_var / sum(comp_var)
  list(portfolio_var = port_var, marginal_var = marg_var,
       component_var = comp_var, pct_contribution = pct_var)
}

liquidity_coverage_ratio <- function(liquid_assets, stressed_outflows,
                                      stressed_inflows, stress_factor = 0.75) {
  net_outflows <- stressed_outflows - stress_factor * stressed_inflows
  lcr          <- liquid_assets / (net_outflows + 1e-8)
  list(lcr = lcr, adequate = lcr >= 1.0, stressed_net_outflows = net_outflows,
       shortfall = pmax(net_outflows - liquid_assets, 0))
}


# ─── ADDITIONAL: MACRO LINKAGE ───────────────────────────────────────────────

macro_financial_linkage <- function(financial_stress, gdp_growth,
                                     credit_growth, window = 12) {
  n   <- length(financial_stress)
  corr_fs_gdp  <- cor(financial_stress, gdp_growth, use="pairwise.complete.obs")
  corr_fs_cred <- cor(financial_stress, credit_growth, use="pairwise.complete.obs")
  lead_corrs <- sapply(1:6, function(lag) {
    if (lag >= n) return(NA)
    cor(financial_stress[1:(n-lag)], gdp_growth[(lag+1):n],
        use="pairwise.complete.obs")
  })
  list(fs_gdp_corr = corr_fs_gdp, fs_credit_corr = corr_fs_cred,
       leading_corrs = lead_corrs,
       best_lead = which.min(lead_corrs))
}

defi_systemic_exposure <- function(protocol_tvls, shared_collateral_pct,
                                    liquidation_thresholds) {
  total_tvl       <- sum(protocol_tvls, na.rm=TRUE)
  concentration   <- protocol_tvls / (total_tvl + 1e-8)
  systemic_risk   <- concentration * shared_collateral_pct *
                       (1 - liquidation_thresholds)
  list(concentration = concentration, systemic_risk = systemic_risk,
       total_systemic = sum(systemic_risk),
       most_exposed = which.max(systemic_risk))
}

exchange_stress_scenario <- function(assets, liabilities, haircuts,
                                      run_fraction = 0.3) {
  stressed_assets <- assets * (1 - haircuts)
  withdrawals     <- liabilities * run_fraction
  shortfall       <- pmax(withdrawals - stressed_assets, 0)
  coverage_ratio  <- stressed_assets / (withdrawals + 1e-8)
  list(stressed_assets = stressed_assets, withdrawals = withdrawals,
       shortfall = shortfall, coverage_ratio = coverage_ratio,
       solvent = shortfall == 0)
}

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

systemic_risk_score <- function(covar, mes, srisk, weights = c(0.33, 0.33, 0.34)) {
  norm <- function(x) (x - min(x, na.rm=TRUE)) / (max(x, na.rm=TRUE) - min(x, na.rm=TRUE) + 1e-8)
  score <- weights[1]*norm(-covar) + weights[2]*norm(-mes) + weights[3]*norm(srisk)
  list(score = score, ranking = order(-score),
       high_risk = score > quantile(score, 0.8, na.rm=TRUE))
}

rolling_systemic_summary <- function(score_series, window = 21) {
  n   <- length(score_series)
  ma  <- as.numeric(stats::filter(score_series, rep(1/window, window), sides=1))
  list(smoothed = ma, trend = c(rep(NA, window), diff(ma, lag=window)),
       elevated = ma > quantile(ma, 0.75, na.rm=TRUE))
}

# version
module_version <- function() "1.0.0"
module_info <- function() list(version="1.0.0", base_r_only=TRUE, pure=TRUE)
# end of file

# placeholder
.module_loaded <- TRUE
