## systemic_risk_study.R
## Systemic risk measurement across crypto markets -- research toolkit
## Pure base R -- no library() calls

# ============================================================
# 1. CRYPTO SYSTEMIC RISK MEASUREMENT
# ============================================================

crypto_srisk_study <- function(returns_matrix, market_caps, debt_estimates,
                                window=90, alpha=0.05) {
  T_<-nrow(returns_matrix); N<-ncol(returns_matrix)
  sys_ret <- rowMeans(returns_matrix)
  srisk_ts <- matrix(NA, T_, N)
  mes_ts   <- matrix(NA, T_, N)
  for (t in seq(window, T_)) {
    idx <- seq(t-window+1, t)
    for (i in seq_len(N)) {
      thr <- quantile(sys_ret[idx], alpha)
      ci  <- which(sys_ret[idx] <= thr)
      mes_ts[t,i] <- mean(returns_matrix[idx[ci], i])
      lrmes  <- 1 - exp(18 * mes_ts[t,i])
      lrmes  <- pmin(pmax(lrmes, 0), 1)
      srisk_ts[t,i] <- pmax(0.08*market_caps[i] - 0.92*market_caps[i]*(1-lrmes), 0)
    }
  }
  list(srisk=srisk_ts, mes=mes_ts,
       total_srisk=rowSums(srisk_ts, na.rm=TRUE),
       systemic_index=rowSums(srisk_ts, na.rm=TRUE)/sum(market_caps))
}

# ============================================================
# 2. CONTAGION CHANNELS RESEARCH
# ============================================================

identify_contagion_channel <- function(returns_matrix, crisis_dates,
                                        window_pre=60, window_post=30) {
  N       <- ncol(returns_matrix)
  results <- list()
  for (crisis in crisis_dates) {
    pre  <- seq(max(1,crisis-window_pre), crisis-1)
    post <- seq(crisis, min(nrow(returns_matrix), crisis+window_post))
    rp   <- returns_matrix[pre,  ]
    rc   <- returns_matrix[post, ]
    results[[length(results)+1]] <- list(
      crisis_date  = crisis,
      pre_corr     = cor(rp,  use="pairwise.complete.obs"),
      post_corr    = cor(rc,  use="pairwise.complete.obs"),
      pre_vol      = apply(rp, 2, sd) * sqrt(252),
      post_vol     = apply(rc, 2, sd) * sqrt(252),
      corr_change  = mean(cor(rc)[upper.tri(cor(rc))]) -
                     mean(cor(rp)[upper.tri(cor(rp))])
    )
  }
  results
}

# ============================================================
# 3. NETWORK CENTRALITY RESEARCH
# ============================================================

build_return_network <- function(returns_matrix, min_corr=0.3) {
  C   <- cor(returns_matrix, use="pairwise.complete.obs")
  adj <- ifelse(abs(C) > min_corr, abs(C), 0)
  diag(adj) <- 0
  deg <- rowSums(adj)
  str <- rowSums(adj)
  ev  <- Re(eigen(adj)$vectors[,1])
  ev  <- ev / (max(abs(ev))+1e-12)
  list(adj=adj, corr=C, degree=deg, strength=str, eigenvector=ev)
}

network_resilience <- function(adj_matrix, n_remove=5) {
  N   <- nrow(adj_matrix)
  deg <- rowSums(adj_matrix)
  remove_order <- order(deg, decreasing=TRUE)
  connectivity <- numeric(n_remove+1)
  cur_adj <- adj_matrix
  connectivity[1] <- sum(cur_adj > 0) / (N*(N-1))
  for (k in seq_len(n_remove)) {
    rm_node <- remove_order[k]
    cur_adj[rm_node,] <- 0; cur_adj[,rm_node] <- 0
    connectivity[k+1] <- sum(cur_adj > 0) / (N*(N-1))
  }
  list(connectivity=connectivity,
       fragility=1-connectivity[n_remove+1]/connectivity[1])
}

# ============================================================
# 4. VOLATILITY TRANSMISSION
# ============================================================

garch_dcc_simplified <- function(returns_matrix, window=60) {
  T_<-nrow(returns_matrix); N<-ncol(returns_matrix)
  cond_corr <- array(NA, c(N,N,T_))
  for (t in seq(window,T_)) {
    idx <- seq(t-window+1,t)
    cond_corr[,,t] <- cor(returns_matrix[idx,], use="pairwise.complete.obs")
  }
  avg_corr <- sapply(seq(window,T_), function(t)
    mean(cond_corr[,,t][upper.tri(cond_corr[,,t])], na.rm=TRUE))
  list(dcc=cond_corr, avg_corr=c(rep(NA,window-1), avg_corr))
}

volatility_spillover_study <- function(returns_matrix, p=2, H=10,
                                        rolling_window=200) {
  T_ <- nrow(returns_matrix); out <- rep(NA_real_,T_)
  var_ols_fn <- function(Y, lag) {
    Tn <- nrow(Y); N <- ncol(Y)
    Ydep <- Y[(lag+1):Tn,]
    X    <- cbind(1, do.call(cbind,lapply(1:lag,function(l) Y[(lag+1-l):(Tn-l),])))
    B    <- tryCatch(solve(t(X)%*%X+diag(ncol(X))*1e-6)%*%t(X)%*%Ydep,
                     error=function(e)matrix(0,ncol(X),N))
    E    <- Ydep-X%*%B
    list(Sigma=t(E)%*%E/(nrow(E)-ncol(X)),
         A=lapply(1:lag,function(l)t(B[2:(N+1)+(l-1)*N,])))
  }
  for (i in seq(rolling_window,T_)) {
    idx <- seq(i-rolling_window+1,i)
    R2  <- returns_matrix[idx,]
    vr  <- tryCatch(var_ols_fn(R2, p), error=function(e) NULL)
    if (is.null(vr)) next
    N <- ncol(R2)
    P  <- tryCatch(t(chol(vr$Sigma+diag(N)*1e-8)), error=function(e)diag(N))
    comp <- matrix(0,N*p,N*p)
    for (l in 1:p) comp[1:N,(l-1)*N+1:N] <- vr$A[[l]]
    if (p>1) comp[(N+1):(N*p),1:(N*(p-1))] <- diag(N*(p-1))
    FEVD <- matrix(0,N,N); denom <- rep(0,N); cp <- diag(N*p)
    for (h in 1:H) {
      cp <- cp%*%comp; Ph <- cp[1:N,1:N]
      for (ii in 1:N) {
        ei <- rep(0,N); ei[ii] <- 1
        for (j in 1:N) {
          ej <- rep(0,N); ej[j] <- 1
          FEVD[ii,j] <- FEVD[ii,j]+(t(ei)%*%Ph%*%P%*%ej)^2
        }
        denom[ii] <- denom[ii]+t(ei)%*%Ph%*%vr$Sigma%*%Ph%*%ei
      }
    }
    for (ii in 1:N) FEVD[ii,] <- FEVD[ii,]/(denom[ii]+1e-12)
    out[i] <- (sum(FEVD)-sum(diag(FEVD)))/N*100
  }
  out
}

# ============================================================
# 5. STRESS TESTING METHODOLOGY
# ============================================================

historical_stress_scenarios <- function(returns_matrix,
                                         scenario_names=c("covid","ftx","luna")) {
  T_ <- nrow(returns_matrix)
  # Compute worst rolling windows
  port_ret <- rowMeans(returns_matrix)
  n_scenarios <- length(scenario_names)
  n_days <- 30
  roll_losses <- sapply(seq_len(T_-n_days), function(i) {
    sum(port_ret[seq(i, i+n_days-1)])
  })
  worst_starts <- order(roll_losses)[1:n_scenarios]
  scenarios <- lapply(seq_along(worst_starts), function(k) {
    s   <- worst_starts[k]
    idx <- seq(s, s+n_days-1)
    list(name=scenario_names[k], start=s, loss=roll_losses[s],
         returns=returns_matrix[idx,])
  })
  list(scenarios=scenarios, worst_loss=min(roll_losses))
}

monte_carlo_stress <- function(returns_matrix, n_sim=1000,
                                stress_multiplier=2, seed=42) {
  set.seed(seed)
  N   <- ncol(returns_matrix); mu <- colMeans(returns_matrix)
  Sig <- cov(returns_matrix)
  # Stressed covariance
  Sig_stress <- Sig * stress_multiplier
  L <- t(chol(Sig_stress + diag(N)*1e-8))
  sim <- t(L %*% matrix(rnorm(N*n_sim), N, n_sim)) +
         matrix(mu, n_sim, N, byrow=TRUE)
  port_losses <- -rowMeans(sim)
  list(var_95=quantile(port_losses,.95),
       es_95 =mean(port_losses[port_losses>=quantile(port_losses,.95)]),
       var_99=quantile(port_losses,.99),
       sim_losses=port_losses)
}

# ============================================================
# 6. REGIME-DEPENDENT RISK
# ============================================================

regime_risk_analysis <- function(returns_matrix, n_regimes=2) {
  port_ret <- rowMeans(returns_matrix)
  T_       <- nrow(returns_matrix)
  # Simple regime classification by volatility
  vol_30   <- rep(NA_real_, T_)
  for (i in seq(30,T_)) vol_30[i] <- sd(port_ret[seq(i-29,i)])*sqrt(252)
  regime   <- ifelse(vol_30 > median(vol_30, na.rm=TRUE), 2, 1)
  r1_idx   <- which(regime==1); r2_idx <- which(regime==2)
  res <- list()
  for (reg in 1:n_regimes) {
    idx <- if(reg==1) r1_idx else r2_idx
    R   <- returns_matrix[idx,]
    res[[reg]] <- list(
      n_obs=length(idx), mean_ret=colMeans(R),
      vol=apply(R,2,sd)*sqrt(252),
      corr=cor(R,use="pairwise.complete.obs"),
      tail_dep=mean(apply(R,2,function(x) mean(x<quantile(x,.05))))
    )
  }
  list(regimes=res, regime_series=regime)
}
