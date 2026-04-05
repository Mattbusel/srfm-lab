## portfolio_attribution.R
## Brinson-Hood-Beebower, factor attribution, risk decomposition
## Pure base R -- no library() calls

brinson_attribution <- function(wp, wb, rp, rb) {
  rb_total <- sum(wb*rb, na.rm=TRUE)
  rp_total <- sum(wp*rp, na.rm=TRUE)
  alloc    <- (wp-wb)*(rb-rb_total)
  select   <- wb*(rp-rb)
  interact <- (wp-wb)*(rp-rb)
  list(allocation=alloc, selection=select, interaction=interact,
       total=alloc+select+interact,
       total_alloc=sum(alloc), total_select=sum(select),
       total_interact=sum(interact),
       active_return=rp_total-rb_total,
       port_return=rp_total, bench_return=rb_total)
}

brinson_fachler <- function(wp, wb, rp, rb) {
  rb_total <- sum(wb*rb, na.rm=TRUE)
  list(allocation  = (wp-wb)*(rb-rb_total),
       selection   = wb*(rp-rb),
       interaction = (wp-wb)*(rp-rb))
}

multi_period_attribution <- function(wp_list, wb_list, rp_list, rb_list) {
  n <- length(rp_list)
  prs <- lapply(seq_len(n), function(t)
    brinson_attribution(wp_list[[t]], wb_list[[t]], rp_list[[t]], rb_list[[t]]))
  cp <- cumprod(sapply(prs, function(r) 1+r$port_return))
  cb <- cumprod(sapply(prs, function(r) 1+r$bench_return))
  list(periods=prs, linked_active=tail(cp,1)-tail(cb,1),
       cum_port=tail(cp,1)-1, cum_bench=tail(cb,1)-1)
}

factor_attribution <- function(port_ret, factor_ret) {
  X  <- cbind(1, factor_ret)
  b  <- tryCatch(
    solve(t(X)%*%X+diag(ncol(X))*1e-8)%*%t(X)%*%port_ret,
    error=function(e) rep(0,ncol(X)))
  alpha <- b[1]; betas <- b[-1]
  fitted <- as.vector(X%*%b); resid <- port_ret-fitted
  list(alpha=alpha, betas=betas,
       factor_contrib=betas*colMeans(factor_ret),
       r2=1-sum(resid^2)/(sum((port_ret-mean(port_ret))^2)+1e-12))
}

rolling_factor_attr <- function(port_ret, factor_ret, window=60) {
  T_<-length(port_ret); nf<-ncol(factor_ret)
  bmat<-matrix(NA,T_,nf); avec<-rep(NA,T_)
  for (i in seq(window,T_)) {
    idx<-seq(i-window+1,i)
    res<-factor_attribution(port_ret[idx],factor_ret[idx,])
    bmat[i,]<-res$betas; avec[i]<-res$alpha
  }
  list(betas=bmat, alpha=avec)
}

risk_decomp <- function(weights, cov_matrix) {
  pv  <- as.numeric(t(weights)%*%cov_matrix%*%weights)
  psd <- sqrt(pv)
  mc  <- as.vector(cov_matrix%*%weights)/psd
  cc  <- weights*mc
  list(vol=psd, var=pv, marginal=mc, component=cc, pct=cc/psd)
}

tail_risk_decomp <- function(ret_matrix, weights, alpha=0.05) {
  pr  <- as.vector(ret_matrix%*%weights)
  thr <- quantile(pr,alpha); ci <- which(pr<=thr)
  cc  <- apply(ret_matrix[ci,],2,mean)*weights
  list(es=mean(pr[ci]), component_es=cc, pct=cc/(sum(cc)+1e-12))
}

factor_risk_decomp <- function(weights, B, F, idio_var) {
  D     <- diag(idio_var)
  Sigma <- B%*%F%*%t(B)+D
  tv    <- as.numeric(t(weights)%*%Sigma%*%weights)
  fv    <- as.numeric(t(weights)%*%B%*%F%*%t(B)%*%weights)
  iv    <- as.numeric(t(weights)%*%D%*%weights)
  list(total_vol=sqrt(tv), factor_vol=sqrt(fv), specific_vol=sqrt(iv),
       factor_pct=fv/tv, specific_pct=iv/tv)
}

perf_metrics <- function(ret, rf=0, ann=252) {
  ex  <- ret-rf/ann; mu<-mean(ret)*ann; sg<-sd(ret)*sqrt(ann)
  cum <- cumprod(1+ret); dd<-(cum-cummax(cum))/cummax(cum)
  list(ann_ret=mu, ann_vol=sg, sharpe=mu/(sg+1e-8),
       sortino=mean(ex)/(sd(pmin(ex,0))+1e-8)*sqrt(ann),
       max_dd=min(dd), calmar=mu/(abs(min(dd))+1e-8),
       skew=mean((ret-mean(ret))^3)/sd(ret)^3,
       kurt=mean((ret-mean(ret))^4)/sd(ret)^4-3)
}

relative_metrics <- function(pr, br, rf=0, ann=252) {
  active <- pr-br; te <- sd(active)*sqrt(ann)
  ir     <- mean(active)/(sd(active)+1e-8)*sqrt(ann)
  beta   <- cov(pr,br)/(var(br)+1e-12)
  u <- br>0; d <- br<0
  list(te=te, ir=ir, active_return=mean(active)*ann, beta=beta,
       up_capture  = if(sum(u)>0) mean(pr[u])/mean(br[u]) else NA,
       down_capture= if(sum(d)>0) mean(pr[d])/mean(br[d]) else NA)
}


# ============================================================
# ADDITIONAL: CRYPTO ATTRIBUTION
# ============================================================

crypto_sector_attribution <- function(wp, wb, rp, rb,
                                       sector_labels = NULL) {
  rb_total <- sum(wb * rb, na.rm = TRUE)
  rp_total <- sum(wp * rp, na.rm = TRUE)
  alloc    <- (wp - wb) * (rb - rb_total)
  select   <- wb * (rp - rb)
  interact <- (wp - wb) * (rp - rb)
  df <- data.frame(alloc=alloc, select=select, interact=interact,
                   total=alloc+select+interact)
  if (!is.null(sector_labels)) rownames(df) <- sector_labels
  list(attr=df, active_ret=rp_total-rb_total,
       total_alloc=sum(alloc), total_select=sum(select))
}

crypto_beta_attribution <- function(port_ret, btc_ret, eth_ret,
                                     altcoin_ret = NULL) {
  factors <- cbind(btc_ret, eth_ret)
  if (!is.null(altcoin_ret)) factors <- cbind(factors, altcoin_ret)
  X  <- cbind(1, factors)
  b  <- tryCatch(solve(t(X)%*%X+diag(ncol(X))*1e-8)%*%t(X)%*%port_ret,
                 error=function(e) rep(0,ncol(X)))
  fitted <- as.vector(X%*%b); resid <- port_ret-fitted
  list(alpha=b[1], betas=b[-1],
       factor_contrib=b[-1]*colMeans(factors),
       r2=1-sum(resid^2)/(sum((port_ret-mean(port_ret))^2)+1e-12))
}

rolling_attribution <- function(wp_ts, wb_ts, rp_ts, rb_ts, window=12) {
  T_ <- nrow(rp_ts); results <- vector("list", T_)
  for (t in seq(window, T_)) {
    idx <- seq(t-window+1, t)
    res <- multi_period_attribution(
      as.list(as.data.frame(t(wp_ts[idx,]))),
      as.list(as.data.frame(t(wb_ts[idx,]))),
      as.list(as.data.frame(t(rp_ts[idx,]))),
      as.list(as.data.frame(t(rb_ts[idx,]))))
    results[[t]] <- res
  }
  results
}

# ============================================================
# ADDITIONAL: PERFORMANCE PERSISTENCE
# ============================================================

performance_persistence <- function(returns_list, n_periods = 2) {
  K <- length(returns_list); n <- floor(K / n_periods)
  sharpes <- sapply(returns_list, function(r)
    mean(r) / (sd(r)+1e-8) * sqrt(252))
  contingency <- table(
    cut(sharpes[1:(n*n_periods-n)], breaks=2, labels=c("below","above")),
    cut(sharpes[(n+1):(n*n_periods)], breaks=2, labels=c("below","above"))
  )
  list(sharpes=sharpes, contingency=contingency,
       repeat_winner_pct = contingency["above","above"] /
         (contingency["above","above"] + contingency["above","below"] + 1e-8) * 100)
}

information_ratio_stability <- function(active_returns, window=12) {
  n <- length(active_returns); ir <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx   <- seq(i-window+1, i)
    ir[i] <- mean(active_returns[idx]) /
              (sd(active_returns[idx])+1e-8) * sqrt(252)
  }
  list(ir=ir, stability=1-sd(ir,na.rm=TRUE)/(mean(abs(ir),na.rm=TRUE)+1e-8))
}

# ============================================================
# ADDITIONAL: RISK BUDGETING
# ============================================================

risk_budget_weights <- function(cov_matrix, risk_budget) {
  N   <- nrow(cov_matrix)
  w0  <- rep(1/N, N)
  obj <- function(w) {
    pv <- as.numeric(t(w) %*% cov_matrix %*% w)
    pv_sqrt <- sqrt(pv)
    rc <- w * as.vector(cov_matrix %*% w) / pv_sqrt
    sum((rc - risk_budget * pv_sqrt)^2)
  }
  res <- tryCatch(
    optim(w0, obj, method="L-BFGS-B",
          lower=rep(0,N), upper=rep(1,N),
          control=list(maxit=1000)),
    error=function(e) list(par=w0))
  w_opt <- res$par / sum(res$par)
  pv    <- as.numeric(t(w_opt) %*% cov_matrix %*% w_opt)
  rc    <- w_opt * as.vector(cov_matrix %*% w_opt) / sqrt(pv)
  list(weights=w_opt, risk_contributions=rc,
       risk_pct=rc/sqrt(pv))
}

equal_risk_contribution <- function(cov_matrix) {
  N <- nrow(cov_matrix)
  risk_budget_weights(cov_matrix, rep(1/N, N))
}

# ============================================================
# ADDITIONAL: DRAWDOWN ANALYSIS
# ============================================================

drawdown_analysis <- function(returns) {
  cum   <- cumprod(1 + returns)
  peak  <- cummax(cum)
  dd    <- (cum - peak) / peak
  max_dd <- min(dd)
  dd_idx <- which(dd == max_dd)[1]
  peak_before <- which(cum[1:dd_idx] == max(cum[1:dd_idx]))
  recovery    <- which(cum[dd_idx:length(cum)] >= cum[peak_before])
  list(drawdown_series = dd, max_dd = max_dd,
       max_dd_start = peak_before[length(peak_before)],
       max_dd_end   = dd_idx,
       recovery_time = if (length(recovery)>0) recovery[1] else NA,
       underwater_days = sum(dd < -0.01))
}

conditional_drawdown_at_risk <- function(returns, alpha = 0.05, n_boot = 500,
                                          seed = 42) {
  set.seed(seed)
  n <- length(returns)
  compute_maxdd <- function(r) {
    cum <- cumprod(1+r); min((cum-cummax(cum))/cummax(cum))
  }
  boot_mdd <- replicate(n_boot, {
    idx <- sample(n, n, replace=TRUE)
    compute_maxdd(returns[idx])
  })
  cdar <- mean(sort(boot_mdd)[1:floor(n_boot * alpha)])
  list(CDaR = cdar, VaR_dd = quantile(boot_mdd, alpha),
       max_dd_dist = boot_mdd)
}
