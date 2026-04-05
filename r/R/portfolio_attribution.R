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

# ============================================================
# ADDITIONAL: ADVANCED PORTFOLIO ANALYTICS
# ============================================================
style_attribution <- function(port_ret, value_idx, growth_idx,
                               small_cap_idx, large_cap_idx) {
  factors <- cbind(value_idx-growth_idx, small_cap_idx-large_cap_idx)
  X  <- cbind(1, factors)
  b  <- tryCatch(solve(t(X)%*%X+diag(3)*1e-8)%*%t(X)%*%port_ret,
                 error=function(e) rep(0,3))
  fitted <- as.vector(X%*%b); resid <- port_ret-fitted
  list(value_tilt=b[2], size_tilt=b[3], alpha=b[1],
       r2=1-var(resid)/(var(port_ret)+1e-8))
}

geometric_attribution <- function(port_period_rets, bench_period_rets) {
  n <- length(port_period_rets)
  cum_p <- cumprod(1+port_period_rets); cum_b <- cumprod(1+bench_period_rets)
  total_active <- tail(cum_p,1)/tail(cum_b,1) - 1
  # Geometric linking factor
  link_factors <- cumprod(1+bench_period_rets) / cumprod(1+port_period_rets)
  list(total_active_geometric=total_active,
       link_factors=link_factors, n_periods=n)
}

holdings_based_attribution <- function(holdings_ts, prices_ts,
                                        bench_holdings_ts) {
  T_ <- nrow(prices_ts); N <- ncol(prices_ts)
  port_ret <- bench_ret <- numeric(T_-1)
  for (t in seq_len(T_-1)) {
    ret_t     <- prices_ts[t+1,]/prices_ts[t,] - 1
    pw        <- holdings_ts[t,]/(sum(holdings_ts[t,])+1e-8)
    bw        <- bench_holdings_ts[t,]/(sum(bench_holdings_ts[t,])+1e-8)
    port_ret[t] <- sum(pw*ret_t,na.rm=TRUE)
    bench_ret[t] <- sum(bw*ret_t,na.rm=TRUE)
  }
  list(port=port_ret, bench=bench_ret, active=port_ret-bench_ret,
       cum_active=prod(1+port_ret)/prod(1+bench_ret)-1)
}

currency_attribution <- function(local_returns, fx_returns, weights) {
  currency_effect <- weights * fx_returns
  local_effect    <- weights * local_returns
  interaction     <- weights * local_returns * fx_returns
  list(local=sum(local_effect), currency=sum(currency_effect),
       interaction=sum(interaction),
       total=sum(local_effect+currency_effect+interaction))
}


# ============================================================
# ADDITIONAL PORTFOLIO ATTRIBUTION
# ============================================================

transaction_cost_analysis <- function(port_ret, bench_ret, turnover,
                                       tc_bps = 10, window = 12) {
  cost_drag  <- turnover * tc_bps / 1e4
  net_ret    <- port_ret - cost_drag
  n          <- length(port_ret); roll_alpha <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i)
    roll_alpha[i] <- mean(net_ret[idx] - bench_ret[idx], na.rm=TRUE) * 252
  }
  list(gross_return = port_ret, net_return = net_ret,
       cost_drag = cost_drag, rolling_net_alpha = roll_alpha,
       annualized_cost = mean(cost_drag, na.rm=TRUE) * 252)
}

factor_exposure_drift <- function(exposures_mat, target_exposures,
                                   window = 60) {
  n   <- nrow(exposures_mat)
  drift <- exposures_mat - matrix(rep(target_exposures, n), n, byrow=TRUE)
  roll_drift_norm <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i)
    roll_drift_norm[i] <- mean(apply(drift[idx, , drop=FALSE], 1, function(r) sqrt(sum(r^2))))
  }
  list(drift = drift, rolling_drift_norm = roll_drift_norm,
       requires_rebalance = roll_drift_norm > quantile(roll_drift_norm, 0.9, na.rm=TRUE))
}

expected_shortfall_attribution <- function(port_ret, asset_rets,
                                            weights, alpha = 0.05) {
  threshold   <- quantile(port_ret, alpha, na.rm=TRUE)
  tail_idx    <- which(port_ret <= threshold)
  if (length(tail_idx) == 0) return(list(es=NA, contributions=rep(NA, ncol(asset_rets))))
  tail_asset  <- colMeans(asset_rets[tail_idx, , drop=FALSE], na.rm=TRUE)
  contribution <- weights * tail_asset
  list(es = mean(port_ret[tail_idx], na.rm=TRUE),
       asset_tail_returns = tail_asset,
       contributions = contribution,
       pct_contribution = contribution / (sum(contribution) + 1e-12))
}

sector_rotation_attribution <- function(sector_weights_t, sector_returns_t,
                                          bench_sector_weights) {
  n_t  <- nrow(sector_weights_t)
  timing_ret <- selection_ret <- numeric(n_t)
  for (t in seq_len(n_t)) {
    bw <- bench_sector_weights[t, ]
    pw <- sector_weights_t[t, ]
    sr <- sector_returns_t[t, ]
    bench_sector_ret <- sum(bw * sr, na.rm=TRUE)
    timing_ret[t]    <- sum((pw - bw) * sr, na.rm=TRUE)
    selection_ret[t] <- sum(pw * (sr - bench_sector_ret), na.rm=TRUE)
  }
  list(timing = timing_ret, selection = selection_ret,
       total = timing_ret + selection_ret,
       cumulative_timing = cumprod(1 + timing_ret),
       cumulative_selection = cumprod(1 + selection_ret))
}

appraisal_analysis <- function(alpha, specific_risk, ir_target = 0.5) {
  appraisal_ratio <- alpha / (specific_risk + 1e-8)
  required_alpha  <- ir_target * specific_risk
  list(appraisal_ratio = appraisal_ratio,
       required_alpha_for_target_ir = required_alpha,
       value_added = alpha - required_alpha,
       is_skilful = appraisal_ratio > ir_target)
}

holdings_drift_analysis <- function(weights_t0, weights_t1, returns_t0_t1) {
  drift_due_to_returns <- weights_t0 * (1 + returns_t0_t1) /
                            (sum(weights_t0 * (1 + returns_t0_t1)) + 1e-12)
  rebalance_effect <- weights_t1 - drift_due_to_returns
  list(passive_drift = drift_due_to_returns,
       rebalance_effect = rebalance_effect,
       net_trade = rebalance_effect,
       turnover = sum(abs(rebalance_effect)) / 2)
}

risk_adjusted_attribution <- function(brinson_alloc, brinson_select,
                                       tracking_error, alpha) {
  total_active  <- brinson_alloc + brinson_select
  ir            <- alpha / (tracking_error + 1e-8)
  alloc_frac    <- brinson_alloc / (total_active + 1e-12)
  select_frac   <- brinson_select / (total_active + 1e-12)
  list(information_ratio = ir,
       allocation_fraction = alloc_frac,
       selection_fraction = select_frac,
       total_active_return = sum(total_active, na.rm=TRUE),
       risk_efficiency = ir / (tracking_error + 1e-8))
}


# ─── ADDITIONAL ATTRIBUTION ───────────────────────────────────────────────────

factor_timing_attribution <- function(factor_rets_mat, bench_exposures,
                                       port_exposures_mat) {
  n_t   <- nrow(port_exposures_mat)
  timing <- port_exposures_mat - matrix(rep(bench_exposures, n_t), n_t, byrow=TRUE)
  timing_rets <- rowSums(timing * factor_rets_mat, na.rm=TRUE)
  list(timing = timing, timing_return = timing_rets,
       cumulative = cumprod(1 + timing_rets) - 1,
       total = prod(1 + timing_rets) - 1,
       positive_timing_pct = mean(timing_rets > 0, na.rm=TRUE))
}

benchmark_replication_quality <- function(port_ret, bench_ret,
                                           windows = c(21, 63, 126)) {
  correlations <- sapply(windows, function(w) {
    n <- length(port_ret)
    if (w >= n) return(NA)
    cor(port_ret[(n-w+1):n], bench_ret[(n-w+1):n], use="pairwise.complete.obs")
  })
  tracking_err <- sapply(windows, function(w) {
    n <- length(port_ret)
    if (w >= n) return(NA)
    sd(port_ret[(n-w+1):n] - bench_ret[(n-w+1):n], na.rm=TRUE) * sqrt(252)
  })
  names(correlations) <- paste0("corr_", windows, "d")
  names(tracking_err) <- paste0("te_", windows, "d")
  c(as.list(correlations), as.list(tracking_err))
}

tail_risk_budget <- function(weights, returns_mat, alpha = 0.05,
                              budget_pct = NULL) {
  port_ret  <- as.vector(returns_mat %*% weights)
  var_port  <- quantile(port_ret, alpha, na.rm=TRUE)
  tail_idx  <- which(port_ret <= var_port)
  cvar_assets <- colMeans(returns_mat[tail_idx, , drop=FALSE], na.rm=TRUE)
  contrib   <- weights * cvar_assets
  pct_contrib <- contrib / (sum(contrib) + 1e-12)
  overage <- if (!is.null(budget_pct)) pct_contrib - budget_pct else NULL
  list(cvar = mean(port_ret[tail_idx], na.rm=TRUE),
       contributions = contrib,
       pct_contribution = pct_contrib,
       budget_overage = overage)
}

alpha_stability_test <- function(alpha_series, window = 12) {
  n   <- length(alpha_series)
  t_stats <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i)
    a   <- alpha_series[idx]
    t_stats[i] <- mean(a, na.rm=TRUE) / (sd(a, na.rm=TRUE) / sqrt(sum(!is.na(a))) + 1e-8)
  }
  list(t_stats = t_stats, significant = abs(t_stats) > 2,
       consistent_positive = t_stats > 2,
       persistent_pct = mean(abs(t_stats) > 1, na.rm=TRUE))
}

crypto_portfolio_construction <- function(signals, volatilities,
                                           max_weight = 0.3, vol_target = 0.2) {
  raw_weights <- signals / (volatilities + 1e-8)
  raw_weights <- raw_weights / (sum(abs(raw_weights)) + 1e-12)
  raw_weights <- pmax(pmin(raw_weights, max_weight), -max_weight)
  port_vol    <- sqrt(sum(raw_weights^2 * volatilities^2))
  scaled_w    <- raw_weights * vol_target / (port_vol + 1e-8)
  scaled_w    <- pmax(pmin(scaled_w, max_weight), -max_weight)
  list(weights = scaled_w, expected_vol = port_vol,
       scaled_vol = vol_target,
       long_pct = sum(scaled_w[scaled_w > 0]),
       short_pct = sum(scaled_w[scaled_w < 0]))
}

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

portfolio_stats_summary <- function(returns, rf = 0) {
  excess   <- returns - rf/252
  ann_ret  <- mean(returns, na.rm=TRUE) * 252
  ann_vol  <- sd(returns, na.rm=TRUE) * sqrt(252)
  sharpe   <- mean(excess, na.rm=TRUE) / (sd(excess, na.rm=TRUE) + 1e-8) * sqrt(252)
  cum_ret  <- cumprod(1 + returns)
  max_dd   <- min(cum_ret / cummax(cum_ret) - 1, na.rm=TRUE)
  calmar   <- ann_ret / (abs(max_dd) + 1e-8)
  data.frame(ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
             max_drawdown=max_dd, calmar=calmar,
             skewness=mean((returns - mean(returns,na.rm=TRUE))^3,na.rm=TRUE) /
                      (sd(returns,na.rm=TRUE)^3 + 1e-8),
             kurtosis=mean((returns - mean(returns,na.rm=TRUE))^4,na.rm=TRUE) /
                      (sd(returns,na.rm=TRUE)^4 + 1e-8) - 3)
}

information_ratio_series <- function(active_ret, window = 63) {
  n <- length(active_ret); ir <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i)
    ir[i] <- mean(active_ret[idx], na.rm=TRUE) /
               (sd(active_ret[idx], na.rm=TRUE) + 1e-8) * sqrt(252)
  }
  ir
}

rolling_beta <- function(port_ret, bench_ret, window = 63) {
  n <- length(port_ret); beta <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx <- seq(i - window + 1, i)
    beta[i] <- cov(port_ret[idx], bench_ret[idx], use="pairwise.complete.obs") /
                 (var(bench_ret[idx], na.rm=TRUE) + 1e-8)
  }
  beta
}

# version
module_version <- function() "1.0.0"
module_info <- function() list(version="1.0.0", base_r_only=TRUE, pure=TRUE)
# end of file

# placeholder
.module_loaded <- TRUE
