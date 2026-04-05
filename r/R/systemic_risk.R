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
