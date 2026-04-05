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

# ============================================================
# ADDITIONAL: EMPIRICAL CRYPTO SYSTEMIC RISK
# ============================================================
btc_dominance_systemic_role <- function(btc_returns, altcoin_returns,
                                         btc_dom, window=60) {
  T_ <- length(btc_returns); N <- ncol(altcoin_returns)
  betas <- matrix(NA, T_, N)
  for (t in seq(window,T_)) {
    idx <- seq(t-window+1,t)
    for (i in seq_len(N))
      if(var(btc_returns[idx])>1e-10)
        betas[t,i] <- cov(altcoin_returns[idx,i],btc_returns[idx])/var(btc_returns[idx])
  }
  # When BTC dominance rises, altcoin betas should too
  btc_dom_lag <- c(NA, btc_dom[-length(btc_dom)])
  corr_dom_beta <- cor(btc_dom_lag, rowMeans(betas,na.rm=TRUE), use="complete.obs")
  list(betas=betas, mean_beta=rowMeans(betas,na.rm=TRUE),
       dominance_beta_corr=corr_dom_beta)
}

defi_protocol_network <- function(tvl_matrix, correlation_threshold=0.5) {
  C   <- cor(tvl_matrix, use="pairwise.complete.obs")
  adj <- ifelse(abs(C)>correlation_threshold, abs(C), 0); diag(adj) <- 0
  deg <- rowSums(adj)
  ev  <- Re(eigen(adj)$vectors[,1])
  list(adjacency=adj, degree=deg,
       eigenvector=ev/(max(abs(ev))+1e-12),
       most_central=which.max(deg))
}

tail_risk_contribution_study <- function(returns_matrix, window=90, alpha=0.05) {
  T_ <- nrow(returns_matrix); N <- ncol(returns_matrix)
  trc_ts <- matrix(NA,T_,N)
  for (t in seq(window,T_)) {
    idx  <- seq(t-window+1,t); R <- returns_matrix[idx,]
    pr   <- rowMeans(R); thr <- quantile(pr,alpha); ci <- which(pr<=thr)
    if(length(ci)>1)
      trc_ts[t,] <- colMeans(R[ci,])/colMeans(R)
  }
  list(trc=trc_ts, mean_trc=colMeans(trc_ts,na.rm=TRUE),
       most_systemic=which.max(colMeans(trc_ts,na.rm=TRUE)))
}

crypto_market_stress_events <- function(returns_matrix, threshold_sd=2) {
  port_ret <- rowMeans(returns_matrix)
  vol_30   <- as.numeric(stats::filter(abs(port_ret), rep(1/30,30), sides=1))
  stress_z <- (abs(port_ret)-vol_30)/(sd(abs(port_ret),na.rm=TRUE)+1e-8)
  events   <- which(stress_z > threshold_sd)
  list(stress_events=events, stress_z=stress_z,
       n_events=length(events),
       pct_stress=length(events)/length(port_ret)*100)
}


# ============================================================
# EMPIRICAL SYSTEMIC RISK RESEARCH
# ============================================================

rolling_covar_study <- function(returns_mat, system_ret, quantile = 0.05,
                                  window = 126) {
  n        <- nrow(returns_mat)
  n_assets <- ncol(returns_mat)
  covar_mat <- matrix(NA, n, n_assets)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i)
    sys <- system_ret[idx]
    thr <- quantile(sys, quantile, na.rm=TRUE)
    stress_idx <- which(sys <= thr)
    for (j in seq_len(n_assets)) {
      asset_stress <- returns_mat[idx, j][stress_idx]
      covar_mat[i, j] <- quantile(asset_stress, quantile, na.rm=TRUE)
    }
  }
  list(covar = covar_mat,
       mean_covar = colMeans(covar_mat, na.rm=TRUE),
       covar_trend = apply(covar_mat, 2, function(x) {
         valid <- which(!is.na(x)); if (length(valid) < 10) return(NA)
         coef(lm(x[valid] ~ valid))[2]
       }))
}

market_stress_episode_detection <- function(returns, vol_threshold = 2.0,
                                              correlation_threshold = 0.7,
                                              min_duration = 5) {
  vol   <- sd(returns, na.rm=TRUE)
  z     <- (returns - mean(returns, na.rm=TRUE)) / (vol + 1e-8)
  is_stressed <- abs(z) > vol_threshold
  runs  <- rle(is_stressed)
  episode_start <- cumsum(c(1, runs$lengths[-length(runs$lengths)]))
  stress_episodes <- episode_start[runs$values & runs$lengths >= min_duration]
  list(z_scores = z, is_stressed = is_stressed,
       n_episodes = length(stress_episodes),
       episode_starts = stress_episodes,
       pct_stressed = mean(is_stressed, na.rm=TRUE))
}

cross_market_contagion_test <- function(returns_a, returns_b,
                                         calm_period, crisis_period) {
  rho_calm   <- cor(returns_a[calm_period], returns_b[calm_period],
                    use="pairwise.complete.obs")
  rho_crisis <- cor(returns_a[crisis_period], returns_b[crisis_period],
                    use="pairwise.complete.obs")
  n_calm   <- sum(!is.na(returns_a[calm_period]) & !is.na(returns_b[calm_period]))
  n_crisis <- sum(!is.na(returns_a[crisis_period]) & !is.na(returns_b[crisis_period]))
  fisher_z_diff <- atanh(rho_crisis) - atanh(rho_calm)
  se_diff  <- sqrt(1/(n_crisis-3) + 1/(n_calm-3))
  z_stat   <- fisher_z_diff / (se_diff + 1e-8)
  p_value  <- 2 * (1 - pnorm(abs(z_stat)))
  list(rho_calm = rho_calm, rho_crisis = rho_crisis,
       correlation_increase = rho_crisis - rho_calm,
       z_statistic = z_stat, p_value = p_value,
       contagion_detected = p_value < 0.05 & rho_crisis > rho_calm)
}

systemic_risk_factor_model <- function(returns_mat, n_factors = 3) {
  cov_mat  <- cov(returns_mat, use="pairwise.complete.obs")
  eig      <- eigen(cov_mat)
  loadings <- eig$vectors[, 1:n_factors]
  factors  <- as.matrix(returns_mat) %*% loadings
  resid_var <- diag(cov_mat) - rowSums(loadings^2 * rep(eig$values[1:n_factors],
                                                         each=nrow(loadings)))
  systemic_var <- rowSums(loadings^2 * rep(eig$values[1:n_factors],
                                            each=nrow(loadings)))
  systemic_pct <- systemic_var / (diag(cov_mat) + 1e-12)
  list(factor_loadings = loadings, factors = factors,
       systemic_pct = systemic_pct,
       idiosyncratic_var = resid_var,
       most_systemic = which.max(systemic_pct))
}

network_topology_study <- function(adjacency_mat, threshold = 0.5) {
  bin_adj  <- (abs(adjacency_mat) > threshold) * 1
  diag(bin_adj) <- 0
  degree       <- rowSums(bin_adj)
  strength     <- rowSums(adjacency_mat * bin_adj)
  clustering   <- sapply(seq_len(nrow(bin_adj)), function(i) {
    nbrs <- which(bin_adj[i, ] == 1)
    k    <- length(nbrs)
    if (k < 2) return(0)
    triangles <- sum(bin_adj[nbrs, nbrs]) / 2
    triangles / (k * (k-1) / 2)
  })
  list(degree = degree, strength = strength, clustering = clustering,
       density = sum(bin_adj) / (nrow(bin_adj) * (nrow(bin_adj)-1)),
       most_connected = which.max(degree))
}


# ─── ADDITIONAL SYSTEMIC RISK RESEARCH ────────────────────────────────────────

tail_risk_correlation_dynamics <- function(returns_mat, quantile_thr = 0.1,
                                            window = 126) {
  n <- nrow(returns_mat); n_a <- ncol(returns_mat)
  roll_tail_corr <- rep(NA, n)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i)
    sub <- returns_mat[idx, , drop=FALSE]
    pair_corrs <- c()
    for (j in 1:(n_a-1)) {
      for (k in (j+1):n_a) {
        thr_j <- quantile(sub[,j], quantile_thr, na.rm=TRUE)
        thr_k <- quantile(sub[,k], quantile_thr, na.rm=TRUE)
        both  <- sub[,j] < thr_j & sub[,k] < thr_k
        if (sum(both) > 5) {
          pair_corrs <- c(pair_corrs,
            cor(sub[both,j], sub[both,k], use="pairwise.complete.obs"))
        }
      }
    }
    roll_tail_corr[i] <- mean(pair_corrs, na.rm=TRUE)
  }
  list(rolling_tail_corr = roll_tail_corr,
       mean_tail_corr = mean(roll_tail_corr, na.rm=TRUE),
       high_regime = roll_tail_corr > quantile(roll_tail_corr, 0.75, na.rm=TRUE))
}

financial_cycle_analysis <- function(credit_growth, asset_prices,
                                      gdp, window = 20) {
  credit_gap  <- credit_growth - as.numeric(stats::filter(credit_growth,
                                rep(1/window, window), sides=2))
  price_gap   <- log(asset_prices) - as.numeric(stats::filter(log(asset_prices),
                                rep(1/window, window), sides=2))
  fc_index    <- (credit_gap + price_gap) / 2
  fc_index[is.na(fc_index)] <- 0
  boom        <- fc_index > quantile(fc_index, 0.75, na.rm=TRUE)
  bust        <- fc_index < quantile(fc_index, 0.25, na.rm=TRUE)
  list(credit_gap = credit_gap, price_gap = price_gap,
       financial_cycle = fc_index, boom = boom, bust = bust,
       cycle_amplitude = diff(range(fc_index, na.rm=TRUE)))
}

systemic_event_backtest <- function(risk_index, actual_crises,
                                     thresholds = seq(0.5, 2.5, 0.25)) {
  results <- lapply(thresholds, function(thr) {
    predicted <- risk_index > thr
    tp <- sum(predicted & actual_crises, na.rm=TRUE)
    fp <- sum(predicted & !actual_crises, na.rm=TRUE)
    fn <- sum(!predicted & actual_crises, na.rm=TRUE)
    precision <- tp / (tp + fp + 1e-12)
    recall    <- tp / (tp + fn + 1e-12)
    f1        <- 2 * precision * recall / (precision + recall + 1e-12)
    list(threshold=thr, tp=tp, fp=fp, fn=fn,
         precision=precision, recall=recall, f1=f1)
  })
  f1s     <- sapply(results, function(r) r$f1)
  best    <- results[[which.max(f1s)]]
  list(all_results = results, best_threshold = best$threshold,
       best_f1 = best$f1, best_precision = best$precision,
       best_recall = best$recall)
}

crypto_stress_contagion_model <- function(btc_crash, altcoin_betas,
                                           defi_tvls, leverage_ratios,
                                           rounds = 5) {
  btc_impact  <- btc_crash
  alt_impacts <- btc_impact * altcoin_betas
  tvl_losses  <- defi_tvls * pmax(-alt_impacts, 0)
  liq_cascade <- tvl_losses * leverage_ratios * 0.1
  total_impact <- alt_impacts
  for (r in seq_len(rounds)) {
    secondary <- -liq_cascade / (sum(defi_tvls) + 1e-8) * altcoin_betas
    total_impact <- total_impact + secondary
    liq_cascade <- liq_cascade * 0.5
    if (max(abs(secondary)) < 0.001) break
  }
  list(initial_impact = alt_impacts, total_impact = total_impact,
       tvl_losses = tvl_losses,
       cascade_multiplier = total_impact / (alt_impacts + 1e-12))
}

# ─── UTILITY / HELPER FUNCTIONS ───────────────────────────────────────────────

annualize_systemic_prob <- function(daily_prob) {
  1 - (1 - daily_prob)^252
}

systemic_risk_index_composite <- function(vol_idx, credit_spread,
                                           interbank_spread, equity_corr,
                                           weights = c(0.25,0.25,0.25,0.25)) {
  norm <- function(x) (x - min(x,na.rm=TRUE)) / (max(x,na.rm=TRUE) - min(x,na.rm=TRUE) + 1e-8)
  mat  <- cbind(norm(vol_idx), norm(credit_spread),
                norm(interbank_spread), norm(equity_corr))
  score <- as.vector(mat %*% weights)
  list(score = score, components = mat,
       crisis = score > quantile(score, 0.9, na.rm=TRUE))
}

crisis_dating_algorithm <- function(systemic_score, threshold = NULL,
                                     min_duration = 5, cooldown = 10) {
  if (is.null(threshold)) threshold <- quantile(systemic_score, 0.9, na.rm=TRUE)
  above  <- systemic_score > threshold
  n      <- length(above)
  crises <- logical(n)
  in_crisis <- FALSE; crisis_end <- -Inf
  start <- NA
  for (i in seq_len(n)) {
    if (!in_crisis && above[i] && (i - crisis_end) > cooldown) {
      in_crisis <- TRUE; start <- i
    } else if (in_crisis && !above[i]) {
      duration <- i - start
      if (duration >= min_duration) crises[start:(i-1)] <- TRUE
      in_crisis <- FALSE; crisis_end <- i
    }
  }
  list(crisis_periods = crises, n_crises = sum(rle(crises)$values),
       pct_in_crisis = mean(crises))
}

# research module metadata
research_version <- function() "1.0.0"
research_info    <- function() list(version="1.0.0", base_r_only=TRUE)
# utility: safe correlation
safe_cor <- function(x, y, method="pearson") {
  tryCatch(cor(x, y, use="pairwise.complete.obs", method=method), error=function(e) NA)
}
# utility: rolling mean
roll_mean <- function(x, w) as.numeric(stats::filter(x, rep(1/w, w), sides=1))
# utility: annualize return
annualize_ret <- function(r, periods_per_year=252) mean(r, na.rm=TRUE) * periods_per_year
# utility: annualize vol
annualize_vol <- function(r, periods_per_year=252) sd(r, na.rm=TRUE) * sqrt(periods_per_year)
# end of file

# systemic risk study module loaded
.srisk_study_loaded <- TRUE
# placeholder
srisk_util <- function() invisible(NULL)
# end
# placeholder2
srisk_util2 <- function(x) x
