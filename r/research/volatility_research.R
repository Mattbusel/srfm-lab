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
