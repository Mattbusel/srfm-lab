## volatility_research.R
## Vol surface dynamics, term structure research
## Pure base R -- no library() calls

# ============================================================
# 1. VOLATILITY SURFACE DYNAMICS RESEARCH
# ============================================================

vol_surface_pca <- function(iv_matrix) {
  # iv_matrix: T x K (T observations, K moneyness/expiry combinations)
  X    <- scale(iv_matrix, center=TRUE, scale=TRUE)
  X[is.nan(X)] <- 0
  Sig  <- cov(X, use="pairwise.complete.obs")
  ev   <- eigen(Sig)
  list(loadings    = ev$vectors,
       variances   = ev$values,
       pct_var     = ev$values/sum(ev$values),
       cum_var     = cumsum(ev$values)/sum(ev$values),
       scores      = X %*% ev$vectors,
       n_factors_90 = which(cumsum(ev$values)/sum(ev$values)>=0.9)[1])
}

vol_factor_dynamics <- function(iv_matrix, n_factors=3) {
  pca  <- vol_surface_pca(iv_matrix)
  T_   <- nrow(iv_matrix)
  facs <- pca$scores[,1:n_factors]
  # Level, slope, curvature interpretation
  fac_names <- c("Level","Slope","Curvature")[1:n_factors]
  list(factors=facs, loadings=pca$loadings[,1:n_factors],
       factor_names=fac_names,
       factor_corr=cor(facs, use="pairwise.complete.obs"),
       autocorr=apply(facs,2,function(f) {
         n <- length(f); c(NA,f[-1]) ; acf(f,plot=FALSE,lag.max=5)$acf[,,1]
       }))
}

# ============================================================
# 2. ATM VOL FORECASTING
# ============================================================

garch11_forecast <- function(returns, omega, alpha, beta,
                              h_ahead=10) {
  n   <- length(returns); h2 <- numeric(n)
  h2[1] <- var(returns)
  for (t in 2:n)
    h2[t] <- omega + alpha*returns[t-1]^2 + beta*h2[t-1]
  h2_fwd <- numeric(h_ahead)
  h2_fwd[1] <- omega + alpha*returns[n]^2 + beta*h2[n]
  for (t in 2:h_ahead)
    h2_fwd[t] <- omega + (alpha+beta)*h2_fwd[t-1]
  list(cond_var=h2, forecast=h2_fwd, forecast_vol=sqrt(h2_fwd)*sqrt(252))
}

realized_vol_forecast <- function(returns, window=22, lambda=0.94) {
  n <- length(returns)
  rv <- rep(NA_real_,n)
  for (i in seq(window,n)) rv[i] <- sd(returns[seq(i-window+1,i)])*sqrt(252)
  # EWMA RV forecast
  ewma <- rep(NA_real_,n); ewma[window] <- rv[window]
  for (i in seq(window+1,n))
    ewma[i] <- lambda*ewma[i-1] + (1-lambda)*returns[i]^2*252
  list(rolling_rv=rv, ewma_forecast=sqrt(pmax(ewma,0)),
       har_inputs=list(daily=rv, weekly=as.numeric(stats::filter(rv,rep(1/5,5),sides=1)),
                       monthly=as.numeric(stats::filter(rv,rep(1/22,22),sides=1))))
}

har_vol_model <- function(rv_daily) {
  n <- length(rv_daily); w5 <- 5; w22 <- 22
  rv_w <- as.numeric(stats::filter(rv_daily,rep(1/w5, w5), sides=1))
  rv_m <- as.numeric(stats::filter(rv_daily,rep(1/w22,w22),sides=1))
  # HAR(1,5,22): RV_{t+1} = b0 + b1*RV_d + b2*RV_w + b3*RV_m + e
  valid <- complete.cases(cbind(rv_daily,rv_w,rv_m))
  y  <- rv_daily[valid]; X <- cbind(1,rv_daily[valid],rv_w[valid],rv_m[valid])
  b  <- tryCatch(solve(t(X)%*%X+diag(4)*1e-8)%*%t(X)%*%y,
                 error=function(e) rep(0,4))
  fitted <- as.vector(X%*%b)
  list(coefs=b, fitted=fitted,
       r2=1-sum((y-fitted)^2)/(sum((y-mean(y))^2)+1e-12),
       forecast_1d=sum(b*c(1,tail(rv_daily,1),tail(rv_w,1),tail(rv_m,1))))
}

# ============================================================
# 3. VOL RISK PREMIUM RESEARCH
# ============================================================

vol_risk_premium <- function(implied_vol, realized_vol, window=22) {
  n   <- length(implied_vol)
  vrp <- implied_vol - realized_vol
  vrp_ma <- as.numeric(stats::filter(vrp,rep(1/window,window),sides=1))
  list(vrp=vrp, smoothed=vrp_ma,
       mean=mean(vrp,na.rm=TRUE), sd=sd(vrp,na.rm=TRUE),
       pct_pos=mean(vrp>0,na.rm=TRUE),
       signal=ifelse(vrp>quantile(vrp,.8,na.rm=TRUE),-1,
               ifelse(vrp<quantile(vrp,.2,na.rm=TRUE),1,0)))
}

term_structure_slope <- function(short_iv, long_iv, dt=21) {
  slope <- (long_iv - short_iv) / dt
  list(slope=slope, inverted=short_iv>long_iv,
       z_slope=(slope-mean(slope,na.rm=TRUE))/(sd(slope,na.rm=TRUE)+1e-8))
}

skew_premium_study <- function(atm_iv, otm_put_iv, otm_call_iv, fwd_returns) {
  rr  <- otm_put_iv - otm_call_iv
  fly <- (otm_put_iv+otm_call_iv)/2 - atm_iv
  # Is skew predictive of future returns?
  ic_rr  <- cor(rr[-length(rr)], fwd_returns[-1], use="complete.obs")
  ic_fly <- cor(fly[-length(fly)], fwd_returns[-1], use="complete.obs")
  list(risk_reversal=rr, butterfly=fly,
       rr_ic=ic_rr, fly_ic=ic_fly,
       mean_rr=mean(rr,na.rm=TRUE), mean_fly=mean(fly,na.rm=TRUE))
}

# ============================================================
# 4. SVI RESEARCH
# ============================================================

svi_w <- function(k,a,b,rho,m,sigma) a+b*(rho*(k-m)+sqrt((k-m)^2+sigma^2))

svi_calibrate <- function(k_vec, iv_vec, T_, n_starts=10, seed=42) {
  set.seed(seed); w_mkt <- iv_vec^2*T_
  obj <- function(p) {
    a<-p[1];b<-p[2];rho<-p[3];m<-p[4];sg<-p[5]
    if(b<0||sg<1e-4||abs(rho)>=1||a+b*sg*sqrt(1-rho^2)<0) return(1e10)
    wf <- svi_w(k_vec,a,b,rho,m,sg); if(any(wf<0)) return(1e10)
    sum((sqrt(wf)-sqrt(w_mkt))^2)
  }
  bobj<-Inf; bpar<-c(0.04,0.1,-0.3,0,0.2)
  for (s in seq_len(n_starts)) {
    p0 <- c(runif(1,.001,.1),runif(1,.01,.5),runif(1,-.9,.9),
            runif(1,-.5,.5),runif(1,.01,.5))
    res <- tryCatch(optim(p0,obj,method="Nelder-Mead",
                          control=list(maxit=5000)),
                    error=function(e)list(value=Inf,par=p0))
    if(res$value<bobj){bobj<-res$value;bpar<-res$par}
  }
  names(bpar) <- c("a","b","rho","m","sigma")
  list(params=bpar, obj=bobj,
       fitted=sqrt(pmax(svi_w(k_vec,bpar[1],bpar[2],bpar[3],bpar[4],bpar[5]),0)/T_))
}

svi_parameter_dynamics <- function(k_vec, iv_matrix, T_vec) {
  n <- nrow(iv_matrix)
  params_ts <- matrix(NA, n, 5)
  colnames(params_ts) <- c("a","b","rho","m","sigma")
  for (i in seq_len(n)) {
    if (any(!is.finite(iv_matrix[i,]))) next
    res <- tryCatch(svi_calibrate(k_vec, iv_matrix[i,], T_vec[i]),
                    error=function(e) NULL)
    if (!is.null(res)) params_ts[i,] <- res$params
  }
  list(params=params_ts,
       atm_vol=sqrt(pmax(params_ts[,"a"],0)),
       skew=params_ts[,"b"]*params_ts[,"rho"],
       curvature=params_ts[,"b"]*sqrt(1-params_ts[,"rho"]^2+1e-12))
}

# ============================================================
# 5. REALIZED VS IMPLIED RESEARCH
# ============================================================

rv_iv_spread_strategy <- function(realized_vol, implied_vol,
                                   fwd_returns, threshold=0.02) {
  vrp    <- implied_vol - realized_vol
  signal <- ifelse(vrp > threshold, -1, ifelse(vrp < -threshold, 1, 0))
  strategy_ret <- signal[-length(signal)] * fwd_returns[-1]
  list(vrp=vrp, signal=signal,
       strategy_ret=strategy_ret,
       sharpe=mean(strategy_ret,na.rm=TRUE)/(sd(strategy_ret,na.rm=TRUE)+1e-8)*sqrt(252),
       hit_rate=mean(sign(signal[-length(signal)])==sign(fwd_returns[-1]),na.rm=TRUE))
}

vol_regime_classification <- function(iv_series, rv_series, window=30) {
  n   <- length(iv_series)
  vrp <- iv_series - rv_series
  iv_z <- (iv_series - mean(iv_series,na.rm=TRUE)) /
           (sd(iv_series,na.rm=TRUE)+1e-8)
  regime <- ifelse(iv_z > 1, "high_vol",
             ifelse(iv_z < -1, "low_vol","normal"))
  list(regime=regime, iv_z=iv_z, vrp=vrp)
}

# ============================================================
# 6. CRYPTO VOL SPECIFICS
# ============================================================

crypto_vol_term_structure <- function(iv_1w, iv_1m, iv_3m, iv_6m) {
  carry <- iv_1w - iv_1m  # positive = backwardation
  slope_short <- (iv_1m - iv_1w) / (1/12 - 1/52)
  slope_long  <- (iv_6m - iv_3m) / (0.5 - 0.25)
  list(carry_1w=carry, slope_short=slope_short, slope_long=slope_long,
       is_inverted=iv_1w>iv_1m)
}

event_vol_study <- function(returns, event_dates, pre_window=5,
                             post_window=10) {
  n   <- length(returns)
  results <- lapply(event_dates, function(ev) {
    pre  <- seq(max(1,ev-pre_window),  ev-1)
    post <- seq(ev, min(n,ev+post_window))
    list(pre_vol  = sd(returns[pre])*sqrt(252),
         post_vol = sd(returns[post])*sqrt(252),
         event_ret = returns[ev],
         vol_jump  = sd(returns[post])/sd(returns[pre]))
  })
  mean_vol_jump <- mean(sapply(results,function(r) r$vol_jump), na.rm=TRUE)
  list(by_event=results, mean_vol_jump=mean_vol_jump)
}

# ============================================================
# ADDITIONAL: EMPIRICAL VOL STUDIES
# ============================================================
vol_clustering_study <- function(returns, window=22) {
  n    <- length(returns); rv <- rep(NA,n)
  for (i in seq(window,n)) rv[i] <- sd(returns[seq(i-window+1,i)])*sqrt(252)
  rv_clean <- rv[!is.na(rv)]
  # ARCH test
  sq_ret  <- returns^2
  n_sq    <- length(sq_ret)
  acf_sq  <- acf(sq_ret, lag.max=5, plot=FALSE)$acf[,,1]
  # Persistence
  ar1_fit <- cor(rv_clean[-1], rv_clean[-length(rv_clean)])
  list(rv=rv, acf_squared=acf_sq, ar1_persistence=ar1_fit,
       clustering=ar1_fit>0.7, arch_evidence=acf_sq[2]>0.1)
}

realized_bipower_variation <- function(returns) {
  n   <- length(returns)
  mu1 <- sqrt(2/pi)
  bv  <- sum(abs(returns[-1]) * abs(returns[-n])) * pi/2
  rv  <- sum(returns^2)
  jump_ratio <- pmax(rv-bv, 0)/rv
  list(rv=rv, bv=bv, jump_component=rv-bv,
       jump_ratio=jump_ratio, has_jumps=jump_ratio>0.1)
}

vol_forecast_comparison <- function(actual_rv, forecasts_list,
                                     forecast_names) {
  errors <- lapply(forecasts_list, function(f) actual_rv-f)
  mses   <- sapply(errors, function(e) mean(e^2,na.rm=TRUE))
  maes   <- sapply(errors, function(e) mean(abs(e),na.rm=TRUE))
  list(mse=setNames(mses,forecast_names),
       mae=setNames(maes,forecast_names),
       best_mse=forecast_names[which.min(mses)],
       best_mae=forecast_names[which.min(maes)])
}

skew_premium_dynamics <- function(rr_25d, fwd_ret_1w, fwd_ret_1m, window=20) {
  n    <- length(rr_25d)
  ic_w <- cor(rr_25d[-c((n-6):n)], fwd_ret_1w[-(1:6)], use="complete.obs")
  ic_m <- cor(rr_25d[-c((n-19):n)], fwd_ret_1m[-(1:19)], use="complete.obs")
  roll_ic <- rep(NA,n)
  for (i in seq(window,n-5))
    roll_ic[i] <- cor(rr_25d[seq(i-window+1,i)],
                      fwd_ret_1w[seq(i-window+2,i+1)], use="complete.obs")
  list(ic_1w=ic_w, ic_1m=ic_m, rolling_ic=roll_ic,
       skew_predicts=abs(ic_w)>0.1)
}

implied_vs_realized_correlation <- function(implied_corr, realized_corr,
                                             window=30) {
  correlation_premium <- implied_corr - realized_corr
  list(premium=correlation_premium,
       mean_prem=mean(correlation_premium,na.rm=TRUE),
       pct_pos=mean(correlation_premium>0,na.rm=TRUE),
       ic=cor(implied_corr[-length(implied_corr)],
              realized_corr[-1], use="complete.obs"))
}


# ============================================================
# ADDITIONAL VOLATILITY RESEARCH
# ============================================================

realized_vol_estimators_comparison <- function(open, high, low, close,
                                                 volume = NULL) {
  n   <- length(close)
  ret <- c(NA, diff(log(close)))
  # Close-to-close
  cc  <- sd(ret, na.rm=TRUE) * sqrt(252)
  # Parkinson
  u   <- log(high / low)
  park <- sqrt(mean(u^2, na.rm=TRUE) / (4 * log(2))) * sqrt(252)
  # Garman-Klass
  gk <- sqrt(mean(0.5 * u^2 - (2*log(2)-1) * ret^2, na.rm=TRUE)) * sqrt(252)
  # Rogers-Satchell
  co  <- log(close / open); ho <- log(high / open); lo <- log(low / open)
  rs  <- sqrt(mean(ho*(ho-co) + lo*(lo-co), na.rm=TRUE)) * sqrt(252)
  # Yang-Zhang
  oc  <- c(NA, log(open[-1] / close[-length(close)]))
  k   <- 0.34 / (1.34 + (n+1)/(n-1))
  yz  <- sqrt((var(oc, na.rm=TRUE) + k * var(co, na.rm=TRUE) +
               (1-k) * var(ret, na.rm=TRUE)) * 252)
  list(cc=cc, parkinson=park, garman_klass=gk, rogers_satchell=rs, yang_zhang=yz,
       efficiency = c(park=park/cc, gk=gk/cc, rs=rs/cc, yz=yz/cc))
}

variance_risk_premium_study <- function(implied_vol, realized_vol,
                                         window = 21) {
  n   <- length(implied_vol)
  vrp <- implied_vol^2 - realized_vol^2
  vrp_ma <- as.numeric(stats::filter(vrp, rep(1/window, window), sides=1))
  # Is VRP a predictor of future returns?
  lead_rv <- c(realized_vol[(window+1):n], rep(NA, window))
  ic_vrp_rv <- cor(vrp, lead_rv, use="pairwise.complete.obs", method="spearman")
  list(vrp = vrp, smoothed = vrp_ma,
       mean_vrp = mean(vrp, na.rm=TRUE),
       pct_positive = mean(vrp > 0, na.rm=TRUE),
       ic_with_future_rv = ic_vrp_rv,
       signal = ifelse(vrp_ma > 0, -1, 1))
}

vol_regime_classification <- function(vol_series, n_regimes = 3) {
  v <- vol_series[!is.na(vol_series)]
  breaks <- quantile(v, seq(0, 1, length.out = n_regimes + 1), na.rm=TRUE)
  regime <- cut(vol_series, breaks, labels=FALSE, include.lowest=TRUE)
  regime_names <- c("low", "medium", "high")[1:n_regimes]
  list(regime = regime,
       regime_label = regime_names[pmin(pmax(regime, 1L), n_regimes)],
       breaks = breaks,
       transition_matrix = {
         tmat <- table(regime[-length(regime)], regime[-1])
         sweep(tmat, 1, rowSums(tmat) + 1e-12, "/")
       })
}

skew_term_structure_research <- function(near_skew, mid_skew, far_skew,
                                          near_mat, mid_mat, far_mat) {
  slope_near_far <- (far_skew - near_skew) / (far_mat - near_mat + 1e-8)
  butterfly_ts   <- mid_skew - (near_skew + far_skew) / 2
  list(slope = slope_near_far, butterfly = butterfly_ts,
       is_inverted = near_skew > far_skew,
       steepening = c(NA, diff(slope_near_far)) > 0)
}

jump_detection_study <- function(returns, vol_estimate, threshold = 3.0) {
  z_scores  <- returns / (vol_estimate + 1e-8)
  is_jump   <- abs(z_scores) > threshold
  jump_ret  <- returns[is_jump]
  n_jumps   <- sum(is_jump, na.rm=TRUE)
  up_jumps  <- sum(is_jump & returns > 0, na.rm=TRUE)
  dn_jumps  <- sum(is_jump & returns < 0, na.rm=TRUE)
  jump_var  <- var(jump_ret, na.rm=TRUE)
  cont_var  <- var(returns[!is_jump], na.rm=TRUE)
  list(is_jump = is_jump, z_scores = z_scores,
       n_jumps = n_jumps, up_jumps = up_jumps, down_jumps = dn_jumps,
       jump_variance_share = jump_var * n_jumps / (var(returns, na.rm=TRUE) * length(returns) + 1e-8),
       avg_jump_size = mean(abs(jump_ret), na.rm=TRUE))
}

vol_surface_pca_research <- function(iv_panel, n_components = 3) {
  # iv_panel: rows = time, cols = (strike, maturity) grid
  complete_rows <- complete.cases(iv_panel)
  iv_clean      <- iv_panel[complete_rows, ]
  pca           <- prcomp(iv_clean, center=TRUE, scale.=FALSE)
  explained_var <- pca$sdev^2 / sum(pca$sdev^2)
  loadings      <- pca$rotation[, 1:n_components]
  list(scores = pca$x[, 1:n_components],
       loadings = loadings,
       explained_var = explained_var[1:n_components],
       cumulative_var = cumsum(explained_var)[1:n_components],
       level_factor = loadings[, 1],
       slope_factor = loadings[, 2],
       curvature_factor = if (n_components >= 3) loadings[, 3] else NULL)
}


# ─── ADDITIONAL VOLATILITY RESEARCH ───────────────────────────────────────────

vol_surface_dynamics_pca <- function(iv_panel_changes, n_components = 3) {
  complete_rows <- complete.cases(iv_panel_changes)
  clean         <- iv_panel_changes[complete_rows, , drop=FALSE]
  if (nrow(clean) < 10) return(list(error="insufficient data"))
  pca <- prcomp(clean, center=TRUE, scale.=FALSE)
  ev  <- pca$sdev^2 / sum(pca$sdev^2)
  list(scores = pca$x[, 1:min(n_components, ncol(pca$x))],
       loadings = pca$rotation[, 1:min(n_components, ncol(pca$rotation))],
       explained_var = ev[1:min(n_components, length(ev))],
       total_explained = sum(ev[1:min(n_components, length(ev))]),
       n_factors_90pct = min(which(cumsum(ev) >= 0.9)))
}

gamma_exposure_pnl_study <- function(gamma, spot, realized_vol,
                                      implied_vol, dt = 1/252) {
  daily_pnl     <- 0.5 * gamma * spot^2 *
                     (realized_vol^2 - implied_vol^2) * dt
  cum_pnl       <- cumsum(daily_pnl)
  vol_carry     <- realized_vol - implied_vol
  sharpe_carry  <- mean(vol_carry, na.rm=TRUE) /
                   (sd(vol_carry, na.rm=TRUE) + 1e-8) * sqrt(252)
  list(daily_pnl = daily_pnl, cumulative_pnl = cum_pnl,
       vol_carry = vol_carry, sharpe_of_carry = sharpe_carry,
       avg_daily_pnl = mean(daily_pnl, na.rm=TRUE),
       pct_positive = mean(daily_pnl > 0, na.rm=TRUE))
}

straddle_pnl_decomposition <- function(straddle_price, realized_move,
                                        delta, gamma, vega,
                                        iv_change, theta, dt = 1/252) {
  delta_pnl <- delta * realized_move
  gamma_pnl <- 0.5 * gamma * realized_move^2
  vega_pnl  <- vega * iv_change
  theta_pnl <- theta * dt
  residual  <- straddle_price - delta_pnl - gamma_pnl - vega_pnl - theta_pnl
  list(total = straddle_price,
       delta = delta_pnl, gamma = gamma_pnl,
       vega = vega_pnl, theta = theta_pnl,
       residual = residual,
       fractions = c(delta=delta_pnl, gamma=gamma_pnl, vega=vega_pnl,
                     theta=theta_pnl, residual=residual) /
                   (abs(straddle_price) + 1e-8))
}

vol_forecasting_model_comparison <- function(realized_vol_series,
                                              horizon = 21) {
  n      <- length(realized_vol_series)
  fwd_rv <- c(realized_vol_series[(horizon+1):n], rep(NA, horizon))
  # EWMA
  lambda <- 0.94
  ewma   <- rep(NA, n); ewma[1] <- realized_vol_series[1]^2
  for (i in 2:n) ewma[i] <- lambda*ewma[i-1] + (1-lambda)*realized_vol_series[i-1]^2
  ewma_vol <- sqrt(ewma)
  # Simple moving average
  ma_vol <- as.numeric(stats::filter(realized_vol_series,
                                     rep(1/horizon, horizon), sides=1))
  # Evaluate: correlation with forward RV
  valid <- !is.na(fwd_rv) & !is.na(ewma_vol) & !is.na(ma_vol)
  ic_ewma <- cor(ewma_vol[valid], fwd_rv[valid], use="pairwise.complete.obs")
  ic_ma   <- cor(ma_vol[valid],   fwd_rv[valid], use="pairwise.complete.obs")
  list(ewma_vol = ewma_vol, ma_vol = ma_vol, forward_rv = fwd_rv,
       ic_ewma = ic_ewma, ic_ma = ic_ma,
       better_model = if (ic_ewma > ic_ma) "EWMA" else "MA")
}

# ─── UTILITY / HELPER FUNCTIONS ───────────────────────────────────────────────

vol_carry_backtest <- function(implied_vol, realized_vol_lead,
                                holding_period = 21, cost_vol_pts = 0.01) {
  n       <- length(implied_vol)
  carry   <- realized_vol_lead - implied_vol
  valid   <- seq_len(n - holding_period)
  returns <- carry[valid]
  net_ret <- returns - cost_vol_pts
  list(gross_returns = returns, net_returns = net_ret,
       sharpe = mean(net_ret, na.rm=TRUE) / (sd(net_ret, na.rm=TRUE) + 1e-8) * sqrt(252/holding_period),
       pct_positive = mean(net_ret > 0, na.rm=TRUE))
}

normalized_vix_signal <- function(vix, window = 252) {
  n  <- length(vix); z <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    idx  <- seq(i - window + 1, i)
    z[i] <- (vix[i] - mean(vix[idx])) / (sd(vix[idx]) + 1e-8)
  }
  list(z_score = z, elevated = z > 1.5, depressed = z < -1,
       signal = ifelse(z > 2, -1, ifelse(z < -1.5, 1, 0)))
}

delta_adjusted_vega_neutral <- function(options_portfolio,
                                         spot_move = 0.01,
                                         vol_move = 0.01) {
  delta_pnl  <- sum(options_portfolio$delta * spot_move, na.rm=TRUE)
  vega_pnl   <- sum(options_portfolio$vega  * vol_move,  na.rm=TRUE)
  gamma_pnl  <- 0.5 * sum(options_portfolio$gamma * spot_move^2, na.rm=TRUE)
  list(delta_pnl = delta_pnl, vega_pnl = vega_pnl, gamma_pnl = gamma_pnl,
       total_pnl = delta_pnl + vega_pnl + gamma_pnl)
}

historical_vol_cone_percentile <- function(current_vol, vol_cone) {
  # vol_cone: list with names = horizon labels, values = vol distributions
  sapply(names(vol_cone), function(h) {
    ecdf(vol_cone[[h]])(current_vol) * 100
  })
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

# ─── ADDITIONAL UTILITY ───────────────────────────────────────────────────────
vol_percentile_rank <- function(current_vol, historical_vols) {
  ecdf(historical_vols)(current_vol) * 100
}

vol_regime_expected_return <- function(regime, regime_returns) {
  tapply(regime_returns, regime, function(r)
    list(mean=mean(r,na.rm=TRUE), sd=sd(r,na.rm=TRUE),
         sharpe=mean(r,na.rm=TRUE)/(sd(r,na.rm=TRUE)+1e-8)*sqrt(252)))
}
# volatility research module loaded
.vol_research_loaded <- TRUE
