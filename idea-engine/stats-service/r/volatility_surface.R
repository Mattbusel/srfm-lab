## volatility_surface.R
## SVI parametrization, SABR calibration, vol surface interpolation, skew analysis
## Pure base R -- no library() calls

# ============================================================
# 1. BLACK-SCHOLES UTILITIES
# ============================================================

bs_price <- function(S, K, r, q = 0, sigma, T_, type = "call") {
  d1  <- (log(S/K) + (r - q + sigma^2/2)*T_) / (sigma*sqrt(T_) + 1e-12)
  d2  <- d1 - sigma*sqrt(T_)
  if (type == "call")
    S*exp(-q*T_)*pnorm(d1) - K*exp(-r*T_)*pnorm(d2)
  else
    K*exp(-r*T_)*pnorm(-d2) - S*exp(-q*T_)*pnorm(-d1)
}

bs_vega <- function(S, K, r, q = 0, sigma, T_) {
  d1 <- (log(S/K) + (r - q + sigma^2/2)*T_) / (sigma*sqrt(T_) + 1e-12)
  S * exp(-q*T_) * dnorm(d1) * sqrt(T_)
}

implied_vol_newton <- function(price, S, K, r, q = 0, T_, type = "call",
                                tol = 1e-8, max_iter = 100) {
  sigma <- 0.25
  for (i in seq_len(max_iter)) {
    p    <- bs_price(S, K, r, q, sigma, T_, type)
    v    <- bs_vega(S, K, r, q, sigma, T_)
    if (abs(v) < 1e-12) break
    diff_ <- p - price
    sigma <- sigma - diff_ / v
    sigma <- max(1e-6, min(sigma, 10))
    if (abs(diff_) < tol) break
  }
  sigma
}

# ============================================================
# 2. SVI PARAMETRIZATION
# ============================================================

svi_total_var <- function(k, a, b, rho, m, sigma) {
  # k = log-moneyness = log(K/F)
  # w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
  a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))
}

svi_iv <- function(k, a, b, rho, m, sigma, T_) {
  w <- svi_total_var(k, a, b, rho, m, sigma)
  sqrt(pmax(w, 0) / T_)
}

svi_calibrate <- function(k_vec, iv_vec, T_,
                           init = c(0.04, 0.1, -0.3, 0, 0.2),
                           n_starts = 10, seed = 42) {
  set.seed(seed)
  w_mkt <- iv_vec^2 * T_
  obj   <- function(params) {
    a <- params[1]; b <- params[2]; rho <- params[3]
    m <- params[4]; sig <- params[5]
    if (b < 0 || sig < 1e-4 || abs(rho) >= 1) return(1e10)
    if (a + b*sig*sqrt(1-rho^2) < 0) return(1e10)
    w_fit <- svi_total_var(k_vec, a, b, rho, m, sig)
    if (any(w_fit < 0)) return(1e10)
    sum((sqrt(w_fit) - sqrt(w_mkt))^2)
  }

  best_obj <- Inf; best_params <- init
  for (s in seq_len(n_starts)) {
    p0 <- c(
      runif(1, 0.001, 0.1),
      runif(1, 0.01,  0.5),
      runif(1, -0.9,  0.9),
      runif(1, -0.5,  0.5),
      runif(1, 0.01,  0.5)
    )
    res <- tryCatch(
      optim(p0, obj, method = "Nelder-Mead",
            control = list(maxit = 5000, reltol = 1e-10)),
      error = function(e) list(value = Inf, par = p0))
    if (res$value < best_obj) {
      best_obj <- res$value; best_params <- res$par
    }
  }
  names(best_params) <- c("a","b","rho","m","sigma")
  list(params = best_params, obj = best_obj,
       fitted_iv = svi_iv(k_vec, best_params[1], best_params[2],
                          best_params[3], best_params[4], best_params[5], T_))
}

svi_butterfly_arbitrage <- function(a, b, rho, m, sigma, k_grid = seq(-2,2,0.01)) {
  w  <- svi_total_var(k_grid, a, b, rho, m, sigma)
  dw <- diff(w) / diff(k_grid)
  # g(k) = (1 - k*dw/(2w))^2 - dw^2/4*(1/w + 1/4) + d2w/2
  # Butterfly-free iff g(k) >= 0 for all k
  dw2 <- diff(dw) / diff(k_grid[-1])
  k2  <- k_grid[-c(1,2)]
  w2  <- w[-c(1,2)]; dw_mid <- dw[-1]
  g   <- (1 - k2*dw_mid/(2*w2+1e-12))^2 -
          dw_mid^2/4*(1/(w2+1e-12) + 0.25) + dw2/2
  list(g = g, k = k2, butterfly_free = all(g >= -1e-8))
}

# ============================================================
# 3. SABR MODEL
# ============================================================

sabr_implied_vol <- function(F, K, T_, alpha, beta, rho, nu) {
  if (F <= 0 || K <= 0 || T_ <= 0) return(NA)
  eps <- 1e-8
  FK  <- F * K; logFK <- log(F/K)
  z   <- nu/alpha * (FK)^((1-beta)/2) * logFK
  x_  <- log((sqrt(1-2*rho*z+z^2) + z - rho) / (1 - rho + eps))

  A <- alpha / ((FK)^((1-beta)/2) *
       (1 + (1-beta)^2/24*logFK^2 + (1-beta)^4/1920*logFK^4))
  B1 <- 1 + ((1-beta)^2/24*alpha^2/(FK)^(1-beta) +
              rho*beta*nu*alpha/(4*(FK)^((1-beta)/2)) +
              (2-3*rho^2)/24*nu^2) * T_

  if (abs(z) < eps) A * B1
  else A * (z / (x_ + eps)) * B1
}

sabr_calibrate <- function(strikes, ivols, F, T_,
                            beta = 0.5, seed = 42) {
  set.seed(seed)
  obj <- function(params) {
    alpha <- params[1]; rho <- params[2]; nu <- params[3]
    if (alpha <= 0 || abs(rho) >= 1 || nu <= 0) return(1e10)
    fit <- sapply(strikes, function(K)
      tryCatch(sabr_implied_vol(F, K, T_, alpha, beta, rho, nu),
               error = function(e) NA))
    if (any(is.na(fit))) return(1e10)
    sum((fit - ivols)^2)
  }
  best_obj <- Inf; best_par <- c(0.3, -0.3, 0.3)
  for (s in 1:20) {
    p0 <- c(runif(1, 0.05, 1.0), runif(1, -0.9, 0.9), runif(1, 0.05, 1.0))
    res <- tryCatch(optim(p0, obj, method = "Nelder-Mead",
                          control = list(maxit = 3000)),
                    error = function(e) list(value=Inf, par=p0))
    if (res$value < best_obj) { best_obj <- res$value; best_par <- res$par }
  }
  names(best_par) <- c("alpha","rho","nu")
  fitted <- sapply(strikes, function(K)
    sabr_implied_vol(F, K, T_, best_par[1], beta, best_par[2], best_par[3]))
  list(params = c(best_par, beta=beta), fitted = fitted, obj = best_obj)
}

# ============================================================
# 4. VOLATILITY SURFACE CONSTRUCTION
# ============================================================

build_vol_surface <- function(expiries, strikes_list, iv_matrix) {
  # expiries: vector of expiries (years)
  # strikes_list: list of strike vectors per expiry
  # iv_matrix: list of IV vectors per expiry
  surface <- lapply(seq_along(expiries), function(i) {
    list(T = expiries[i], K = strikes_list[[i]], iv = iv_matrix[[i]])
  })
  structure(list(expiries = expiries, surface = surface,
                 n_expiries = length(expiries)), class = "vol_surface")
}

interpolate_vol_surface <- function(surface, T_target, K_target,
                                     method = "bilinear") {
  expiries <- surface$expiries
  # Find bracketing expiries
  idx_lo <- max(1, findInterval(T_target, expiries))
  idx_hi <- min(length(expiries), idx_lo + 1)

  if (idx_lo == idx_hi) {
    sl <- surface$surface[[idx_lo]]
    return(approx(sl$K, sl$iv, xout = K_target, rule = 2)$y)
  }

  T_lo <- expiries[idx_lo]; T_hi <- expiries[idx_hi]
  sl_lo <- surface$surface[[idx_lo]]; sl_hi <- surface$surface[[idx_hi]]
  iv_lo <- approx(sl_lo$K, sl_lo$iv, xout = K_target, rule = 2)$y
  iv_hi <- approx(sl_hi$K, sl_hi$iv, xout = K_target, rule = 2)$y

  # Variance interpolation (flat-forward)
  w_lo <- iv_lo^2 * T_lo; w_hi <- iv_hi^2 * T_hi
  w_t  <- w_lo + (T_target - T_lo)/(T_hi - T_lo) * (w_hi - w_lo)
  sqrt(pmax(w_t / T_target, 0))
}

# ============================================================
# 5. SKEW METRICS
# ============================================================

vol_skew_metrics <- function(K_vec, iv_vec, F, T_) {
  k_vec  <- log(K_vec / F)
  # Fit SVI
  cal    <- tryCatch(
    svi_calibrate(k_vec, iv_vec, T_),
    error = function(e) NULL)

  atm_idx  <- which.min(abs(k_vec))
  atm_iv   <- iv_vec[atm_idx]

  # 25-delta risk reversal and butterfly
  d25      <- qnorm(0.25) * atm_iv * sqrt(T_)  # approx delta=0.25 log-moneyness
  iv_call25 <- approx(k_vec, iv_vec, xout =  d25, rule = 2)$y
  iv_put25  <- approx(k_vec, iv_vec, xout = -d25, rule = 2)$y
  rr25      <- iv_call25 - iv_put25
  fly25     <- (iv_call25 + iv_put25) / 2 - atm_iv

  # Slope and curvature
  if (length(k_vec) > 2) {
    fit_coef <- coef(lm(iv_vec ~ k_vec + I(k_vec^2)))
    slope_atm    <- fit_coef[2]
    curvature    <- 2 * fit_coef[3]
  } else { slope_atm <- NA; curvature <- NA }

  list(atm_iv = atm_iv, rr_25d = rr25, fly_25d = fly25,
       slope = slope_atm, curvature = curvature,
       svi_params = if (!is.null(cal)) cal$params else NULL)
}

term_structure_metrics <- function(expiries, atm_ivs) {
  n      <- length(expiries)
  slope  <- if (n > 1)
    (atm_ivs[n] - atm_ivs[1]) / (expiries[n] - expiries[1]) else NA
  # Forward variances
  tot_var <- atm_ivs^2 * expiries
  fwd_var <- c(tot_var[1], diff(tot_var)) / c(expiries[1], diff(expiries))
  list(atm_ivs = atm_ivs, expiries = expiries,
       term_slope = slope,
       forward_var = fwd_var,
       forward_vol = sqrt(pmax(fwd_var, 0)),
       inverted = atm_ivs[1] > atm_ivs[n])
}

# ============================================================
# 6. VARIANCE SWAP REPLICATION
# ============================================================

var_swap_strike <- function(strikes, call_ivs, put_ivs, F, r, T_) {
  # Carr-Madan: K_var = 2/T * integral
  n_calls <- length(strikes[strikes >= F])
  n_puts  <- length(strikes[strikes <  F])
  call_K  <- strikes[strikes >= F]; put_K <- strikes[strikes < F]
  call_iv <- call_ivs[strikes >= F]; put_iv <- put_ivs[strikes < F]

  integrand_call <- sapply(seq_along(call_K), function(i) {
    p <- bs_price(F, call_K[i], 0, 0, call_iv[i], T_, "call")
    2 * p / call_K[i]^2
  })
  integrand_put <- sapply(seq_along(put_K), function(i) {
    p <- bs_price(F, put_K[i], 0, 0, put_iv[i], T_, "put")
    2 * p / put_K[i]^2
  })

  int_call <- if (length(call_K) > 1) trapz(call_K, integrand_call) else 0
  int_put  <- if (length(put_K)  > 1) trapz(put_K, integrand_put)   else 0
  k_var    <- exp(r*T_) * (int_call + int_put) / T_

  list(var_swap_strike = k_var, vol_swap_strike = sqrt(k_var),
       int_call = int_call, int_put = int_put)
}

trapz <- function(x, y) sum(diff(x) * (y[-length(y)] + y[-1]) / 2)

# ============================================================
# 7. LOCAL VOLATILITY
# ============================================================

dupire_local_vol <- function(vol_surface_fn, K, T_, dK = 0.01, dT = 1/365) {
  # Dupire formula: sigma_loc^2 = dC/dT / (0.5*K^2*d2C/dK2)
  C_0  <- vol_surface_fn(K, T_)
  C_dT <- vol_surface_fn(K, T_ + dT)
  C_up <- vol_surface_fn(K + dK, T_)
  C_dn <- vol_surface_fn(K - dK, T_)
  dC_dT  <- (C_dT - C_0) / dT
  d2C_dK2 <- (C_up - 2*C_0 + C_dn) / dK^2
  lv2 <- 2 * dC_dT / (K^2 * d2C_dK2 + 1e-12)
  list(local_var = lv2, local_vol = sqrt(pmax(lv2, 0)))
}


# ============================================================
# ADDITIONAL: VOL SURFACE DYNAMICS
# ============================================================

vol_surface_pca <- function(iv_matrix) {
  X    <- scale(iv_matrix, center = TRUE, scale = TRUE)
  X[is.nan(X)] <- 0
  Sig  <- cov(X, use = "pairwise.complete.obs")
  ev   <- eigen(Sig)
  list(loadings = ev$vectors, variances = ev$values,
       pct_var  = ev$values / sum(ev$values),
       cum_var  = cumsum(ev$values) / sum(ev$values),
       scores   = X %*% ev$vectors,
       n_for_90 = which(cumsum(ev$values)/sum(ev$values) >= 0.9)[1])
}

vol_risk_premium <- function(implied_vol, realized_vol, window = 22) {
  vrp    <- implied_vol - realized_vol
  vrp_ma <- as.numeric(stats::filter(vrp, rep(1/window, window), sides = 1))
  list(vrp = vrp, smoothed = vrp_ma,
       mean = mean(vrp, na.rm = TRUE), sd = sd(vrp, na.rm = TRUE),
       pct_positive = mean(vrp > 0, na.rm = TRUE),
       signal = ifelse(vrp > quantile(vrp, .8, na.rm=TRUE), -1,
                ifelse(vrp < quantile(vrp, .2, na.rm=TRUE),  1, 0)))
}

# ============================================================
# ADDITIONAL: HESTON MODEL CALIBRATION
# ============================================================

heston_char_fn <- function(phi, S, K, r, q, T_,
                            kappa, theta, sigma_v, rho, V0) {
  xi  <- kappa - 1i * rho * sigma_v * phi
  d   <- sqrt(xi^2 + sigma_v^2 * (phi^2 + 1i * phi))
  g   <- (xi - d) / (xi + d)
  C   <- (r - q) * 1i * phi * T_ +
         kappa * theta / sigma_v^2 *
           ((xi - d) * T_ - 2 * log((1 - g * exp(-d * T_)) / (1 - g)))
  D   <- (xi - d) / sigma_v^2 * (1 - exp(-d * T_)) /
         (1 - g * exp(-d * T_))
  exp(C + D * V0 + 1i * phi * log(S))
}

heston_call_price <- function(S, K, r, q, T_,
                               kappa, theta, sigma_v, rho, V0,
                               N_int = 100) {
  phi_grid <- seq(0.01, 50, length.out = N_int)
  dphi     <- phi_grid[2] - phi_grid[1]
  log_SK   <- log(S / K)

  integrand1 <- function(phi) {
    cf  <- heston_char_fn(phi - 1i, S, K, r, q, T_, kappa, theta, sigma_v, rho, V0)
    Re(exp(-1i * phi * log_SK) * cf /
       (1i * phi * heston_char_fn(-1i, S, K, r, q, T_, kappa, theta, sigma_v, rho, V0)))
  }
  integrand2 <- function(phi) {
    cf <- heston_char_fn(phi, S, K, r, q, T_, kappa, theta, sigma_v, rho, V0)
    Re(exp(-1i * phi * log_SK) * cf / (1i * phi))
  }

  P1 <- 0.5 + 1/pi * sum(sapply(phi_grid, integrand1)) * dphi
  P2 <- 0.5 + 1/pi * sum(sapply(phi_grid, integrand2)) * dphi

  S * exp(-q*T_) * P1 - K * exp(-r*T_) * P2
}

heston_calibrate <- function(strikes, iv_market, S, r, q, T_,
                              seed = 42, n_starts = 5) {
  set.seed(seed)
  obj <- function(p) {
    kappa <- p[1]; theta <- p[2]; sigma_v <- p[3]; rho <- p[4]; V0 <- p[5]
    if (kappa<=0||theta<=0||sigma_v<=0||abs(rho)>=1||V0<=0) return(1e10)
    if (2*kappa*theta <= sigma_v^2) return(1e10)  # Feller condition
    fit <- sapply(strikes, function(K) {
      price <- tryCatch(heston_call_price(S,K,r,q,T_,kappa,theta,sigma_v,rho,V0),
                        error=function(e) NA)
      if (is.na(price) || price <= 0) return(NA)
      tryCatch({
        bs_implied <- implied_vol_newton(price, S, K, r, q, T_, "call")
        bs_implied
      }, error=function(e) NA)
    })
    if (any(is.na(fit))) return(1e10)
    sum((fit - iv_market)^2)
  }
  bobj <- Inf; bpar <- c(2, 0.04, 0.3, -0.5, 0.04)
  for (s in seq_len(n_starts)) {
    p0 <- c(runif(1,.5,5), runif(1,.01,.15), runif(1,.1,.8),
            runif(1,-.9,-.1), runif(1,.01,.15))
    res <- tryCatch(optim(p0, obj, method="Nelder-Mead",
                          control=list(maxit=2000)),
                    error=function(e) list(value=Inf,par=p0))
    if (res$value < bobj) { bobj <- res$value; bpar <- res$par }
  }
  names(bpar) <- c("kappa","theta","sigma_v","rho","V0")
  list(params=bpar, obj=bobj)
}

# ============================================================
# ADDITIONAL: CRYPTO VOL FEATURES
# ============================================================

crypto_vol_surface_features <- function(iv_matrix_by_expiry,
                                         expiries, strikes_pct) {
  n_exp <- length(expiries)
  atm_iv     <- numeric(n_exp)
  skew_25d   <- numeric(n_exp)
  fly_25d    <- numeric(n_exp)
  for (i in seq_len(n_exp)) {
    iv <- iv_matrix_by_expiry[, i]
    T_ <- expiries[i]
    atm_idx  <- which.min(abs(strikes_pct - 1))
    atm_iv[i] <- iv[atm_idx]
    # 25-delta approx as +/- 10% moneyness
    call25_idx <- which.min(abs(strikes_pct - 1.10))
    put25_idx  <- which.min(abs(strikes_pct - 0.90))
    skew_25d[i] <- iv[call25_idx] - iv[put25_idx]
    fly_25d[i]  <- (iv[call25_idx] + iv[put25_idx])/2 - atm_iv[i]
  }
  list(atm_iv = atm_iv, skew_25d = skew_25d, fly_25d = fly_25d,
       term_slope = (atm_iv[n_exp] - atm_iv[1]) /
                    (expiries[n_exp] - expiries[1] + 1e-8))
}

vol_cone <- function(returns, windows = c(5, 10, 21, 63, 126, 252),
                     quantiles = c(0.05, 0.25, 0.50, 0.75, 0.95)) {
  n  <- length(returns)
  res <- lapply(windows, function(w) {
    rvs <- rep(NA, n)
    for (i in seq(w, n))
      rvs[i] <- sd(returns[seq(i-w+1,i)]) * sqrt(252)
    qs <- quantile(rvs, quantiles, na.rm=TRUE)
    c(window = w, setNames(as.vector(qs), paste0("q", quantiles*100)))
  })
  do.call(rbind, lapply(res, as.data.frame))
}
