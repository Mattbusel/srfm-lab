## volatility_surface.R
## Options vol surface, SVI, SABR calibration
## Pure base R -- no library() calls

bs_price <- function(S, K, r, q=0, sigma, T_, type="call") {
  d1 <- (log(S/K)+(r-q+sigma^2/2)*T_)/(sigma*sqrt(T_)+1e-12)
  d2 <- d1 - sigma*sqrt(T_)
  if (type=="call") S*exp(-q*T_)*pnorm(d1)-K*exp(-r*T_)*pnorm(d2)
  else K*exp(-r*T_)*pnorm(-d2)-S*exp(-q*T_)*pnorm(-d1)
}

bs_vega <- function(S, K, r, q=0, sigma, T_) {
  d1 <- (log(S/K)+(r-q+sigma^2/2)*T_)/(sigma*sqrt(T_)+1e-12)
  S*exp(-q*T_)*dnorm(d1)*sqrt(T_)
}

implied_vol <- function(price, S, K, r, q=0, T_, type="call",
                         tol=1e-8, max_iter=200) {
  sg <- 0.25
  for (i in seq_len(max_iter)) {
    p  <- bs_price(S, K, r, q, sg, T_, type)
    v  <- bs_vega(S, K, r, q, sg, T_)
    if (abs(v) < 1e-12) break
    sg <- sg - (p-price)/v
    sg <- max(1e-6, min(sg, 10))
    if (abs(p-price) < tol) break
  }
  sg
}

svi_w <- function(k, a, b, rho, m, sigma) {
  a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
}

svi_iv <- function(k, a, b, rho, m, sigma, T_) {
  sqrt(pmax(svi_w(k,a,b,rho,m,sigma),0)/T_)
}

svi_calibrate <- function(k_vec, iv_vec, T_, n_starts=15, seed=42) {
  set.seed(seed)
  w_mkt <- iv_vec^2 * T_
  obj <- function(p) {
    a<-p[1]; b<-p[2]; rho<-p[3]; m<-p[4]; sg<-p[5]
    if (b<0||sg<1e-4||abs(rho)>=1||a+b*sg*sqrt(1-rho^2)<0) return(1e10)
    wf <- svi_w(k_vec,a,b,rho,m,sg)
    if (any(wf<0)) return(1e10)
    sum((sqrt(wf)-sqrt(w_mkt))^2)
  }
  bobj <- Inf; bpar <- c(0.04,0.1,-0.3,0,0.2)
  for (s in seq_len(n_starts)) {
    p0 <- c(runif(1,.001,.1), runif(1,.01,.5), runif(1,-.9,.9),
            runif(1,-.5,.5), runif(1,.01,.5))
    res <- tryCatch(optim(p0, obj, method="Nelder-Mead",
                          control=list(maxit=5000)),
                    error=function(e) list(value=Inf,par=p0))
    if (res$value < bobj) { bobj<-res$value; bpar<-res$par }
  }
  names(bpar) <- c("a","b","rho","m","sigma")
  list(params=bpar, obj=bobj,
       fitted=svi_iv(k_vec,bpar[1],bpar[2],bpar[3],bpar[4],bpar[5],T_))
}

sabr_vol <- function(F, K, T_, alpha, beta=0.5, rho, nu) {
  if (F<=0||K<=0||T_<=0) return(NA)
  lFK <- log(F/K); FK <- F*K
  z   <- nu/alpha*(FK)^((1-beta)/2)*lFK
  x_  <- log((sqrt(1-2*rho*z+z^2)+z-rho)/(1-rho+1e-8))
  A   <- alpha/((FK)^((1-beta)/2)*(1+(1-beta)^2/24*lFK^2+(1-beta)^4/1920*lFK^4))
  B1  <- 1+((1-beta)^2/24*alpha^2/(FK)^(1-beta)+
             rho*beta*nu*alpha/(4*(FK)^((1-beta)/2))+
             (2-3*rho^2)/24*nu^2)*T_
  if (abs(z)<1e-8) A*B1 else A*(z/(x_+1e-8))*B1
}

sabr_calibrate <- function(strikes, ivols, F, T_, beta=0.5, seed=42) {
  set.seed(seed)
  obj <- function(p) {
    al<-p[1]; rho<-p[2]; nu<-p[3]
    if (al<=0||abs(rho)>=1||nu<=0) return(1e10)
    fit <- sapply(strikes, function(K)
      tryCatch(sabr_vol(F,K,T_,al,beta,rho,nu), error=function(e) NA))
    if (any(is.na(fit))) return(1e10)
    sum((fit-ivols)^2)
  }
  bobj<-Inf; bpar<-c(0.3,-0.3,0.3)
  for (s in 1:20) {
    p0 <- c(runif(1,.05,1), runif(1,-.9,.9), runif(1,.05,1))
    res <- tryCatch(optim(p0,obj,method="Nelder-Mead",
                          control=list(maxit=3000)),
                    error=function(e) list(value=Inf,par=p0))
    if (res$value<bobj) { bobj<-res$value; bpar<-res$par }
  }
  names(bpar) <- c("alpha","rho","nu")
  fitted <- sapply(strikes, function(K)
    sabr_vol(F,K,T_,bpar[1],beta,bpar[2],bpar[3]))
  list(params=c(bpar,beta=beta), fitted=fitted, obj=bobj)
}

vol_surface_interp <- function(expiries, strike_lists, iv_lists,
                                T_target, K_target) {
  idx_lo <- max(1, findInterval(T_target, expiries))
  idx_hi <- min(length(expiries), idx_lo+1)
  iv_at  <- function(i) approx(strike_lists[[i]], iv_lists[[i]],
                                xout=K_target, rule=2)$y
  if (idx_lo == idx_hi) return(iv_at(idx_lo))
  T_lo <- expiries[idx_lo]; T_hi <- expiries[idx_hi]
  w_lo <- iv_at(idx_lo)^2*T_lo; w_hi <- iv_at(idx_hi)^2*T_hi
  w_t  <- w_lo + (T_target-T_lo)/(T_hi-T_lo)*(w_hi-w_lo)
  sqrt(pmax(w_t/T_target, 0))
}

skew_metrics <- function(K_vec, iv_vec, F, T_) {
  k_vec <- log(K_vec/F)
  atm_iv <- approx(k_vec, iv_vec, xout=0, rule=2)$y
  d25    <- qnorm(0.25)*atm_iv*sqrt(T_)
  c25    <- approx(k_vec, iv_vec, xout= d25, rule=2)$y
  p25    <- approx(k_vec, iv_vec, xout=-d25, rule=2)$y
  if (length(k_vec)>2) {
    cf <- coef(lm(iv_vec ~ k_vec + I(k_vec^2)))
    sl <- cf[2]; curv <- 2*cf[3]
  } else { sl <- NA; curv <- NA }
  list(atm_iv=atm_iv, rr_25d=c25-p25, fly_25d=(c25+p25)/2-atm_iv,
       slope=sl, curvature=curv)
}

term_structure <- function(expiries, atm_ivs) {
  tot_var <- atm_ivs^2 * expiries
  fwd_var <- c(tot_var[1], diff(tot_var)) / c(expiries[1], diff(expiries))
  list(atm_ivs=atm_ivs, expiries=expiries,
       forward_vol=sqrt(pmax(fwd_var,0)),
       slope=(atm_ivs[length(atm_ivs)]-atm_ivs[1])/(expiries[length(expiries)]-expiries[1]),
       inverted=atm_ivs[1]>atm_ivs[length(atm_ivs)])
}

var_swap_strike <- function(K_vec, iv_vec, F, r, T_) {
  trapz <- function(x,y) sum(diff(x)*(y[-length(y)]+y[-1])/2)
  calls <- K_vec[K_vec>=F]; puts <- K_vec[K_vec<F]
  c_iv  <- iv_vec[K_vec>=F]; p_iv <- iv_vec[K_vec<F]
  i_c <- if(length(calls)>1) trapz(calls, 2*sapply(seq_along(calls), function(j)
    bs_price(F,calls[j],0,0,c_iv[j],T_,"call")/calls[j]^2)) else 0
  i_p <- if(length(puts)>1) trapz(puts, 2*sapply(seq_along(puts), function(j)
    bs_price(F,puts[j],0,0,p_iv[j],T_,"put")/puts[j]^2)) else 0
  kv <- exp(r*T_)*(i_c+i_p)/T_
  list(var_strike=kv, vol_strike=sqrt(kv))
}

butterfly_arb_check <- function(a, b, rho, m, sigma,
                                 k_grid=seq(-2,2,.01)) {
  w   <- svi_w(k_grid,a,b,rho,m,sigma)
  dw  <- diff(w)/diff(k_grid)
  dw2 <- diff(dw)/diff(k_grid[-1])
  k2  <- k_grid[-c(1,2)]; w2 <- w[-c(1,2)]; dw_m <- dw[-1]
  g   <- (1-k2*dw_m/(2*w2+1e-12))^2 - dw_m^2/4*(1/(w2+1e-12)+0.25) + dw2/2
  list(g=g, k=k2, butterfly_free=all(g>=-1e-8))
}

calendar_arb_check <- function(expiries, svi_params_list) {
  n <- length(expiries); ok <- TRUE
  for (i in seq_len(n-1)) {
    p1 <- svi_params_list[[i]]; p2 <- svi_params_list[[i+1]]
    k  <- seq(-1,1,.05)
    w1 <- svi_w(k,p1[1],p1[2],p1[3],p1[4],p1[5])*expiries[i]
    w2 <- svi_w(k,p2[1],p2[2],p2[3],p2[4],p2[5])*expiries[i+1]
    if (any(w2 < w1)) ok <- FALSE
  }
  ok
}


# ============================================================
# ADDITIONAL: VOL SURFACE ANALYTICS
# ============================================================

vol_surface_pca <- function(iv_matrix) {
  X   <- scale(iv_matrix, center=TRUE, scale=TRUE); X[is.nan(X)] <- 0
  S   <- cov(X, use="pairwise.complete.obs"); ev <- eigen(S)
  list(loadings=ev$vectors, variances=ev$values,
       pct_var=ev$values/sum(ev$values),
       cum_var=cumsum(ev$values)/sum(ev$values),
       scores=X %*% ev$vectors)
}

vol_risk_premium_surface <- function(implied_iv_matrix, realized_iv_matrix) {
  vrp  <- implied_iv_matrix - realized_iv_matrix
  list(vrp=vrp, mean_vrp=colMeans(vrp,na.rm=TRUE),
       vol_vrp=apply(vrp,2,sd,na.rm=TRUE),
       pct_positive=apply(vrp>0,2,mean,na.rm=TRUE))
}

# ============================================================
# ADDITIONAL: HESTON MODEL
# ============================================================

heston_cf <- function(phi, S, K, r, q, T_, kappa, theta, sv, rho, V0) {
  xi <- kappa - 1i*rho*sv*phi
  d  <- sqrt(xi^2 + sv^2*(phi^2+1i*phi))
  g  <- (xi-d)/(xi+d)
  C  <- (r-q)*1i*phi*T_ + kappa*theta/sv^2*((xi-d)*T_-2*log((1-g*exp(-d*T_))/(1-g)))
  D  <- (xi-d)/sv^2*(1-exp(-d*T_))/(1-g*exp(-d*T_))
  exp(C + D*V0 + 1i*phi*log(S))
}

heston_price <- function(S, K, r, q, T_, kappa, theta, sv, rho, V0, N=64) {
  phi_ <- seq(.01, 100, length.out=N); dphi <- phi_[2]-phi_[1]
  lSK  <- log(S/K)
  P1 <- .5 + 1/pi * sum(Re(exp(-1i*phi_*lSK) *
    heston_cf(phi_-1i,S,K,r,q,T_,kappa,theta,sv,rho,V0) /
    (1i*phi_*heston_cf(-1i,S,K,r,q,T_,kappa,theta,sv,rho,V0))) * dphi)
  P2 <- .5 + 1/pi * sum(Re(exp(-1i*phi_*lSK) *
    heston_cf(phi_,S,K,r,q,T_,kappa,theta,sv,rho,V0) /
    (1i*phi_)) * dphi)
  S*exp(-q*T_)*P1 - K*exp(-r*T_)*P2
}

# ============================================================
# ADDITIONAL: LOCAL VOL
# ============================================================

dupire_local_vol <- function(vol_fn, K, T_, dK=.01, dT=1/365) {
  C0  <- vol_fn(K,T_); CdT <- vol_fn(K,T_+dT)
  Cup <- vol_fn(K+dK,T_); Cdn <- vol_fn(K-dK,T_)
  lv2 <- 2*(CdT-C0)/dT / (K^2*(Cup-2*C0+Cdn)/dK^2 + 1e-12)
  list(lv=sqrt(pmax(lv2,0)), lv2=lv2)
}

# ============================================================
# ADDITIONAL: VARIANCE TERM STRUCTURE
# ============================================================

forward_variance_curve <- function(expiries, atm_ivs) {
  tot_var <- atm_ivs^2 * expiries
  fwd_var <- c(tot_var[1], diff(tot_var)) / c(expiries[1], diff(expiries))
  list(total_var=tot_var, fwd_var=fwd_var, fwd_vol=sqrt(pmax(fwd_var,0)),
       slope=(atm_ivs[length(atm_ivs)]-atm_ivs[1]) /
              (expiries[length(expiries)]-expiries[1]+1e-8))
}

var_swap_replication <- function(K_vec, iv_vec, F, r, T_) {
  trapz <- function(x,y) sum(diff(x)*(y[-length(y)]+y[-1])/2)
  c_K   <- K_vec[K_vec>=F]; p_K <- K_vec[K_vec<F]
  c_iv  <- iv_vec[K_vec>=F]; p_iv <- iv_vec[K_vec<F]
  c_int <- if(length(c_K)>1) trapz(c_K, 2*sapply(seq_along(c_K), function(j)
    bs_price(F,c_K[j],0,0,c_iv[j],T_,"call")/c_K[j]^2)) else 0
  p_int <- if(length(p_K)>1) trapz(p_K, 2*sapply(seq_along(p_K), function(j)
    bs_price(F,p_K[j],0,0,p_iv[j],T_,"put")/p_K[j]^2)) else 0
  kv <- exp(r*T_)*(c_int+p_int)/T_
  list(var_strike=kv, vol_strike=sqrt(kv))
}

# ============================================================
# ADDITIONAL: CRYPTO VOL SPECIFICS
# ============================================================

btc_vol_regime <- function(rv_series, iv_series, window=30) {
  n   <- length(rv_series)
  vrp <- iv_series - rv_series
  vrp_z <- (vrp-mean(vrp,na.rm=TRUE))/(sd(vrp,na.rm=TRUE)+1e-8)
  iv_z  <- (iv_series-mean(iv_series,na.rm=TRUE))/(sd(iv_series,na.rm=TRUE)+1e-8)
  regime <- ifelse(iv_z>1.5 & vrp_z>1, "fear_with_premium",
             ifelse(iv_z>1.5 & vrp_z<0, "fear_no_premium",
             ifelse(iv_z< -1, "complacency", "normal")))
  list(vrp=vrp, iv_z=iv_z, vrp_z=vrp_z, regime=regime)
}

crypto_term_structure_model <- function(iv_1w, iv_1m, iv_3m) {
  carry   <- iv_1w - iv_1m
  rolldown <- (iv_1m - iv_1w) / (1/12 - 1/52 + 1e-8)
  list(carry=carry, rolldown_rate=rolldown,
       inverted=iv_1w>iv_1m,
       term_prem=iv_3m-iv_1m)
}

perpetual_vol_surface <- function(iv_atm, skew, curvature,
                                   moneyness_grid=seq(.7,1.3,by=.05)) {
  k <- log(moneyness_grid)
  iv_surface <- iv_atm + skew*k + curvature*k^2/2
  list(moneyness=moneyness_grid, iv=pmax(iv_surface,0.01))
}

vol_cone_crypto <- function(returns, windows=c(7,14,30,60,90)) {
  n <- length(returns)
  res <- lapply(windows, function(w) {
    rv <- rep(NA,n)
    for (i in seq(w,n)) rv[i] <- sd(returns[seq(i-w+1,i)])*sqrt(365)
    qs <- quantile(rv, c(.05,.25,.5,.75,.95), na.rm=TRUE)
    c(window=w, q5=qs[1], q25=qs[2], median=qs[3], q75=qs[4], q95=qs[5])
  })
  do.call(rbind, lapply(res, as.data.frame))
}
