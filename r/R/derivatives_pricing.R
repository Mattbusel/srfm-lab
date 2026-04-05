# =============================================================================
# derivatives_pricing.R
# Options and derivatives pricing for crypto markets
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Crypto options (on BTC, ETH) trade on Deribit and CME.
# Black-Scholes is a baseline but underprices tails; Heston adds stochastic
# vol. Perpetual futures have no expiry -- they use funding rates instead of
# basis to keep prices near spot. Understanding derivatives pricing is essential
# for vol arbitrage, hedging, and delta-neutral strategies.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. BLACK-SCHOLES PRICING AND GREEKS
# -----------------------------------------------------------------------------

#' Black-Scholes call and put pricing
#' @param S spot price
#' @param K strike price
#' @param T time to expiry (in years)
#' @param r risk-free rate (annual, continuous)
#' @param sigma implied/historical vol (annual)
#' @param type "call" or "put"
bs_price <- function(S, K, T_exp, r=0, sigma) {
  if (T_exp <= 0 || sigma <= 0 || S <= 0 || K <= 0) {
    call_px <- max(S - K, 0); put_px <- max(K - S, 0)
    return(list(call=call_px, put=put_px))
  }
  d1 <- (log(S/K) + (r + 0.5*sigma^2)*T_exp) / (sigma*sqrt(T_exp))
  d2 <- d1 - sigma*sqrt(T_exp)

  call_px <- S * pnorm(d1) - K * exp(-r*T_exp) * pnorm(d2)
  put_px  <- K * exp(-r*T_exp) * pnorm(-d2) - S * pnorm(-d1)

  list(call=call_px, put=put_px, d1=d1, d2=d2)
}

#' Black-Scholes Greeks
#' @return list of greeks for both call and put
bs_greeks <- function(S, K, T_exp, r=0, sigma) {
  if (T_exp <= 0 || sigma <= 0) {
    return(list(delta_c=as.numeric(S>K), delta_p=as.numeric(S<K)-1,
                gamma=0, vega=0, theta_c=0, theta_p=0, rho_c=0, rho_p=0))
  }
  d1 <- (log(S/K) + (r + 0.5*sigma^2)*T_exp) / (sigma*sqrt(T_exp))
  d2 <- d1 - sigma*sqrt(T_exp)
  phi_d1 <- dnorm(d1)
  sqrt_T <- sqrt(T_exp)

  # Delta
  delta_c <- pnorm(d1)
  delta_p <- delta_c - 1

  # Gamma (same for call and put)
  gamma <- phi_d1 / (S * sigma * sqrt_T)

  # Vega (per 1% change in sigma, so divide by 100)
  vega  <- S * phi_d1 * sqrt_T / 100  # per 1 vol point

  # Theta (per calendar day)
  theta_c <- (-S * phi_d1 * sigma / (2 * sqrt_T) -
               r * K * exp(-r*T_exp) * pnorm(d2)) / 365
  theta_p <- (-S * phi_d1 * sigma / (2 * sqrt_T) +
               r * K * exp(-r*T_exp) * pnorm(-d2)) / 365

  # Rho (per 1% change in r)
  rho_c <- K * T_exp * exp(-r*T_exp) * pnorm(d2) / 100
  rho_p <- -K * T_exp * exp(-r*T_exp) * pnorm(-d2) / 100

  # Charm (delta decay per day)
  charm_c <- -phi_d1 * (2*r*T_exp - d2*sigma*sqrt_T) /
              (2 * T_exp * sigma * sqrt_T) / 365

  list(delta_c=delta_c, delta_p=delta_p, gamma=gamma,
       vega=vega, theta_c=theta_c, theta_p=theta_p,
       rho_c=rho_c, rho_p=rho_p, charm_c=charm_c,
       d1=d1, d2=d2)
}

#' Greeks for a specific option type
option_greeks <- function(S, K, T_exp, r=0, sigma, type="call") {
  g <- bs_greeks(S, K, T_exp, r, sigma)
  p <- bs_price(S, K, T_exp, r, sigma)
  cat(sprintf("=== %s Option Greeks (K=%g, T=%.2f, sigma=%.2f) ===\n",
              toupper(type), K, T_exp, sigma))
  cat(sprintf("Price:  %.4f\n", if(type=="call") p$call else p$put))
  cat(sprintf("Delta:  %.4f\n", if(type=="call") g$delta_c else g$delta_p))
  cat(sprintf("Gamma:  %.6f\n", g$gamma))
  cat(sprintf("Vega:   %.4f (per 1 vol pt)\n", g$vega))
  cat(sprintf("Theta:  %.4f (per day)\n", if(type=="call") g$theta_c else g$theta_p))
  cat(sprintf("Rho:    %.4f (per 1%% rate change)\n",
              if(type=="call") g$rho_c else g$rho_p))
  invisible(g)
}

# -----------------------------------------------------------------------------
# 2. IMPLIED VOLATILITY VIA BRENT'S METHOD
# -----------------------------------------------------------------------------

#' Brent's method bisection for implied volatility
#' Given an observed option price, find sigma that makes BS price match
#' @param market_price observed option price
#' @param S spot, K strike, T_exp expiry, r rate
#' @param type "call" or "put"
#' @param lower lower bound for sigma search
#' @param upper upper bound
implied_vol_brent <- function(market_price, S, K, T_exp, r=0,
                               type="call", lower=0.001, upper=5, tol=1e-6) {
  bs_func <- function(sigma) {
    p <- bs_price(S, K, T_exp, r, sigma)
    if (type=="call") p$call - market_price else p$put - market_price
  }

  fa <- bs_func(lower); fb <- bs_func(upper)
  if (fa * fb > 0) {
    # Try to bracket
    for (try_upper in c(5, 10, 20)) {
      fb <- bs_func(try_upper)
      if (fa * fb <= 0) { upper <- try_upper; break }
    }
    if (fa * fb > 0) return(NA)
  }

  # Brent's method
  a <- lower; b <- upper
  fa <- bs_func(a); fb <- bs_func(b)
  s <- a
  if (abs(fa) < abs(fb)) { tmp<-a; a<-b; b<-tmp; tmp<-fa; fa<-fb; fb<-tmp }
  c_val <- a; fc <- fa
  mflag <- TRUE; d <- 0

  for (iter in 1:100) {
    if (abs(b - a) < tol) break
    if (fa != fc && fb != fc) {
      # Inverse quadratic interpolation
      s <- (a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) +
            c_val*fa*fb/((fc-fa)*(fc-fb)))
    } else {
      # Secant method
      s <- b - fb * (b-a)/(fb-fa)
    }
    cond1 <- !((3*a+b)/4 < s & s < b || b < s & s < (3*a+b)/4)
    cond2 <- mflag  && abs(s-b) >= abs(b-c_val)/2
    cond3 <- !mflag && abs(s-b) >= abs(c_val-d)/2
    if (cond1 || cond2 || cond3) { s <- (a+b)/2; mflag<-TRUE } else mflag<-FALSE
    fs <- bs_func(s)
    d <- c_val; c_val <- b; fc <- fb
    if (fa*fs < 0) { b<-s; fb<-fs } else { a<-s; fa<-fs }
    if (abs(fa) < abs(fb)) { tmp<-a; a<-b; b<-tmp; tmp<-fa; fa<-fb; fb<-tmp }
  }
  b
}

# -----------------------------------------------------------------------------
# 3. VOLATILITY SURFACE FITTING (SVI PARAMETRIZATION)
# -----------------------------------------------------------------------------

#' Stochastic Volatility Inspired (SVI) parametrization
#' Developed by Jim Gatheral; widely used in FX and crypto vol surfaces
#' w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
#' where k = log(K/F), w = total implied variance, (a,b,rho,m,sigma) = params
#' @param k log-moneyness vector
#' @param params named: a, b, rho, m, sigma
svi_variance <- function(k, params) {
  a <- params["a"]; b <- params["b"]; rho <- params["rho"]
  m <- params["m"]; sigma_svi <- params["sigma"]
  a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma_svi^2))
}

#' Fit SVI to market implied vols
#' @param k_obs observed log-moneyness values
#' @param iv_obs observed implied vols
#' @param T_exp time to expiry
svi_fit <- function(k_obs, iv_obs, T_exp) {
  # Convert to total implied variance
  w_obs <- iv_obs^2 * T_exp

  # Initial parameters
  init_params <- c(a=min(w_obs)*0.9, b=0.1, rho=-0.3, m=0, sigma=0.2)

  # Objective: least squares in total variance space
  objective <- function(params) {
    if (params["b"] < 0 || abs(params["rho"]) >= 1 || params["sigma"] <= 0) return(1e6)
    w_fit <- tryCatch(svi_variance(k_obs, params), error=function(e) rep(Inf, length(k_obs)))
    if (any(w_fit < 0)) return(1e6)
    sum((w_obs - w_fit)^2)
  }

  fit <- optim(init_params, objective, method="Nelder-Mead",
               control=list(maxit=3000))
  params_opt <- fit$par

  # Compute fitted vols
  w_fit  <- svi_variance(k_obs, params_opt)
  iv_fit <- sqrt(pmax(w_fit, 0) / T_exp)

  rmse   <- sqrt(mean((iv_obs - iv_fit)^2))

  cat("=== SVI Vol Surface Fit ===\n")
  cat(sprintf("a=%.4f, b=%.4f, rho=%.3f, m=%.3f, sigma=%.3f\n",
              params_opt["a"], params_opt["b"], params_opt["rho"],
              params_opt["m"], params_opt["sigma"]))
  cat(sprintf("RMSE: %.4f vol points\n", rmse))

  invisible(list(params=params_opt, iv_fit=iv_fit, rmse=rmse, T_exp=T_exp))
}

#' Compute ATM vol, skew, and convexity from SVI surface
svi_analytics <- function(svi_result) {
  p <- svi_result$params
  T_exp <- svi_result$T_exp
  # ATM (k=0)
  w_atm <- svi_variance(0, p)
  iv_atm <- sqrt(max(w_atm, 0) / T_exp)

  # Skew: d(IV)/dk at k=0 (risk-reversal proxy)
  dk <- 0.001
  w_up <- svi_variance(dk, p); w_dn <- svi_variance(-dk, p)
  skew <- (sqrt(w_up/T_exp) - sqrt(w_dn/T_exp)) / (2*dk)

  # Convexity: d^2(IV)/dk^2 at k=0 (butterfly proxy)
  convex <- (sqrt(w_up/T_exp) - 2*iv_atm + sqrt(w_dn/T_exp)) / dk^2

  cat(sprintf("ATM vol: %.4f, Skew: %.4f, Convexity: %.4f\n",
              iv_atm, skew, convex))
  list(iv_atm=iv_atm, skew=skew, convexity=convex)
}

# -----------------------------------------------------------------------------
# 4. HESTON MODEL PRICING VIA CHARACTERISTIC FUNCTION FFT
# -----------------------------------------------------------------------------

#' Heston characteristic function
#' Captures stochastic volatility and the vol-of-vol smile
#' @param u Fourier variable
#' @param S0 initial spot
#' @param v0 initial variance
#' @param kappa mean reversion speed
#' @param theta long-term variance
#' @param sigma vol of vol
#' @param rho correlation between spot and vol
#' @param r risk-free rate
#' @param T_exp time to expiry
heston_char_fn <- function(u, S0, v0, kappa, theta, sigma, rho, r, T_exp) {
  i <- 1i
  d <- sqrt((rho*sigma*i*u - kappa)^2 - sigma^2*(-(u^2+i*u)))
  g <- (kappa - rho*sigma*i*u - d) / (kappa - rho*sigma*i*u + d)

  exp_dT <- exp(-d * T_exp)
  C <- r*i*u*T_exp + kappa*theta/sigma^2 *
       ((kappa - rho*sigma*i*u - d)*T_exp - 2*log((1 - g*exp_dT)/(1-g)))
  D <- (kappa - rho*sigma*i*u - d) / sigma^2 * (1 - exp_dT)/(1 - g*exp_dT)

  exp(C + D*v0 + i*u*log(S0))
}

#' Heston call price via Carr-Madan FFT
#' @param S0 spot, K strike, v0 initial var
#' @param kappa,theta,sigma,rho Heston parameters
#' @param r rate, T_exp expiry
#' @param N FFT grid size
heston_call_fft <- function(S0, K, v0, kappa, theta, sigma, rho, r, T_exp,
                              N=4096, alpha=1.5) {
  eta    <- 0.25        # step in log-strike space
  lambda <- 2*pi / (N*eta)
  b      <- N*lambda/2  # upper bound for log-strike

  # Fourier transform of modified call price
  # phi(v) = exp(-r*T) * char_fn(v - (alpha+1)*i) / (alpha^2 + alpha - v^2 + i*(2*alpha+1)*v)
  v_j <- (seq_len(N) - 1) * eta

  psi <- sapply(v_j, function(v) {
    numer <- heston_char_fn(v - (alpha+1)*1i, S0, v0, kappa, theta, sigma, rho, r, T_exp)
    denom <- alpha^2 + alpha - v^2 + 1i*(2*alpha+1)*v
    exp(-r*T_exp) * numer / (denom + 1e-30)
  })

  # Apply Simpson's rule weights
  w <- rep(eta, N); w[1] <- eta/3; w[N] <- eta/3
  for (j in seq(2, N-1)) w[j] <- if (j %% 2 == 0) 4*eta/3 else 2*eta/3

  # FFT
  x  <- exp(1i * b * v_j) * psi * w
  y  <- Re(fft(x)) * exp(-alpha * ((-b + lambda*(seq_len(N)-1)))) / pi

  # Log-strikes
  k_grid <- -b + lambda * (seq_len(N)-1)
  K_grid <- exp(k_grid)

  # Interpolate to find call price at desired strike
  k_target <- log(K)
  if (k_target < min(k_grid) || k_target > max(k_grid)) {
    # Fall back to BS
    sigma_approx <- sqrt(v0)
    return(bs_price(S0, K, T_exp, r, sigma_approx)$call)
  }
  call_price <- approx(k_grid, y, xout=k_target)$y
  max(call_price, max(S0 - K*exp(-r*T_exp), 0))
}

# -----------------------------------------------------------------------------
# 5. MONTE CARLO OPTION PRICING (GBM + JUMPS)
# -----------------------------------------------------------------------------

#' Monte Carlo option pricing: geometric Brownian motion
#' @param S0 spot, K strike, r rate, sigma vol, T_exp expiry
#' @param n_sim number of simulations, type "call" or "put"
mc_option_gbm <- function(S0, K, r=0, sigma, T_exp, n_sim=50000, type="call") {
  Z  <- rnorm(n_sim)
  ST <- S0 * exp((r - 0.5*sigma^2)*T_exp + sigma*sqrt(T_exp)*Z)
  payoff <- if (type=="call") pmax(ST-K, 0) else pmax(K-ST, 0)
  price  <- exp(-r*T_exp) * mean(payoff)
  se     <- exp(-r*T_exp) * sd(payoff) / sqrt(n_sim)
  list(price=price, se=se, ci=c(price-2*se, price+2*se))
}

#' Monte Carlo option pricing: Merton jump-diffusion
#' Jumps modeled as Poisson process with log-normal jump sizes
#' @param lambda jump intensity (jumps per year)
#' @param mu_j mean log-jump size
#' @param sigma_j vol of log-jump size
mc_option_jump_diffusion <- function(S0, K, r=0, sigma, T_exp, n_sim=50000,
                                      lambda=1.0, mu_j=-0.10, sigma_j=0.15,
                                      type="call") {
  # Expected jump impact: k_bar = exp(mu_j + sigma_j^2/2) - 1
  k_bar <- exp(mu_j + sigma_j^2/2) - 1
  # Drift adjustment: r_adj = r - lambda*k_bar
  r_adj <- r - lambda * k_bar

  # Simulate paths
  ST <- numeric(n_sim)
  for (i in seq_len(n_sim)) {
    # Diffusion component
    Z <- rnorm(1)
    S_diff <- S0 * exp((r_adj - 0.5*sigma^2)*T_exp + sigma*sqrt(T_exp)*Z)
    # Jump component: N(T_exp) ~ Poisson(lambda*T_exp)
    n_jumps <- rpois(1, lambda*T_exp)
    if (n_jumps > 0) {
      log_jumps <- sum(rnorm(n_jumps, mu_j, sigma_j))
      S_diff <- S_diff * exp(log_jumps)
    }
    ST[i] <- S_diff
  }

  payoff <- if (type=="call") pmax(ST-K, 0) else pmax(K-ST, 0)
  price  <- exp(-r*T_exp) * mean(payoff)
  se     <- exp(-r*T_exp) * sd(payoff) / sqrt(n_sim)

  # Implied vol of jump-diffusion price
  iv_jd <- tryCatch(
    implied_vol_brent(price, S0, K, T_exp, r, type),
    error=function(e) NA
  )

  cat(sprintf("MC Jump-Diffusion %s: price=%.4f (se=%.5f), impl_vol=%.4f\n",
              type, price, se, iv_jd))
  list(price=price, se=se, impl_vol=iv_jd)
}

# -----------------------------------------------------------------------------
# 6. CRYPTO PERPETUAL FUNDING RATE MODELING
# -----------------------------------------------------------------------------

#' Perpetual futures fair value and funding rate
#' Perpetuals maintain price near spot via funding mechanism:
#' Funding rate = (perp_price - spot_price) / spot_price * funding_rate_multiplier
#' Positive funding: longs pay shorts (perp > spot = long premium)
#' @param spot_prices spot price series
#' @param perp_prices perpetual futures price series
#' @param periods_per_day number of funding periods per day (default 3)
perp_funding_model <- function(spot_prices, perp_prices, periods_per_day=3) {
  n <- length(spot_prices)

  # Basis: perp - spot
  basis    <- perp_prices - spot_prices
  basis_pct <- basis / spot_prices

  # Funding rate (typically 8-hour): premium / premium_interval
  # Actual funding = max(0.05%, premium) or min(-0.05%, premium) depending on exchange
  funding_rate <- basis_pct / 3  # divide by 3 for 8-hour funding

  # Apply funding caps typical of exchanges (0.05% per 8h = 0.375% per day annualized ~137%)
  funding_cap <- 0.0005
  funding_capped <- pmin(pmax(funding_rate, -funding_cap), funding_cap)

  # Annualized carry from funding
  ann_carry <- mean(funding_capped, na.rm=TRUE) * periods_per_day * 365 * 100

  # Basis trade: long spot + short perp, collect funding
  # P&L = funding_rate + (spot_return - perp_return) / spot
  spot_ret <- c(NA, diff(log(spot_prices)))
  perp_ret <- c(NA, diff(log(perp_prices)))
  basis_trade_pnl <- funding_capped - (perp_ret - spot_ret)

  valid_pnl <- basis_trade_pnl[!is.na(basis_trade_pnl)]
  basis_sharpe <- mean(valid_pnl) / (sd(valid_pnl)+1e-10) * sqrt(periods_per_day*365)

  cat("=== Perpetual Funding Rate Analysis ===\n")
  cat(sprintf("Average funding rate: %.5f per 8h\n", mean(funding_capped, na.rm=TRUE)))
  cat(sprintf("Annualized carry: %.2f%%\n", ann_carry))
  cat(sprintf("Basis trade Sharpe: %.3f\n", basis_sharpe))
  cat(sprintf("% of time funding positive (longs pay): %.1f%%\n",
              100*mean(funding_capped > 0, na.rm=TRUE)))

  invisible(list(basis=basis, funding=funding_capped,
                 ann_carry=ann_carry, basis_sharpe=basis_sharpe,
                 basis_trade_pnl=basis_trade_pnl))
}

# -----------------------------------------------------------------------------
# 7. REALIZED VS IMPLIED VOL SPREAD
# -----------------------------------------------------------------------------

#' Crypto options: realized vs implied vol spread (vol risk premium)
#' VRP = IV - RV (implied > realized = vol sellers profit)
#' VRP is persistently positive across asset classes (vol is overpriced)
#' @param iv_series daily ATM implied vol series
#' @param returns daily log-returns
#' @param horizon forward realized vol window (default 21 days)
vrp_analysis <- function(iv_series, returns, horizon=21) {
  n <- length(iv_series)

  # Compute realized vol over forward window
  rv_forward <- numeric(n)
  for (t in seq_len(n - horizon)) {
    rv_forward[t] <- sd(returns[(t+1):(t+horizon)]) * sqrt(252)
  }
  rv_forward[(n-horizon+1):n] <- NA

  # VRP
  vrp <- iv_series - rv_forward
  vrp_valid <- vrp[!is.na(vrp)]

  # Strategy: sell straddle when IV high, buy when IV low
  iv_zscore <- (iv_series - mean(iv_series, na.rm=T)) / (sd(iv_series, na.rm=T)+1e-10)
  # Short vol when z > 1 (IV high), long vol when z < -1 (IV low)
  position <- ifelse(iv_zscore > 1, -1, ifelse(iv_zscore < -1, 1, 0))
  vol_strategy_pnl <- position * vrp

  sharpe_vrp <- mean(vrp_valid) / (sd(vrp_valid)+1e-10) * sqrt(252/horizon)

  cat("=== Realized vs Implied Vol (VRP) ===\n")
  cat(sprintf("Average VRP: %.4f (IV - RV)\n", mean(vrp_valid)))
  cat(sprintf("VRP Std: %.4f\n", sd(vrp_valid)))
  cat(sprintf("% of time VRP > 0 (IV overpriced): %.1f%%\n",
              100*mean(vrp_valid > 0)))
  cat(sprintf("VRP Sharpe: %.3f\n", sharpe_vrp))

  invisible(list(vrp=vrp, iv=iv_series, rv_forward=rv_forward,
                 sharpe=sharpe_vrp, position=position))
}

# -----------------------------------------------------------------------------
# 8. VOLATILITY TERM STRUCTURE
# -----------------------------------------------------------------------------

#' Fit and analyze the implied vol term structure
#' @param expiries vector of time to expiry (in years)
#' @param atm_ivs vector of ATM implied vols for each expiry
vol_term_structure <- function(expiries, atm_ivs) {
  n <- length(expiries)
  if (n < 2) stop("Need at least 2 expiry points")

  # Fit Nelson-Siegel-like term structure
  # IV(T) = a + b*exp(-c*T) + d*(T*exp(-c*T))
  objective <- function(params) {
    a<-params[1]; b<-params[2]; c<-max(params[3],0.1); d<-params[4]
    fitted <- a + b*exp(-c*expiries) + d*(expiries*exp(-c*expiries))
    sum((atm_ivs - fitted)^2)
  }
  init <- c(mean(atm_ivs), atm_ivs[1]-mean(atm_ivs), 2, 0)
  fit  <- optim(init, objective, method="Nelder-Mead", control=list(maxit=2000))
  a<-fit$par[1]; b<-fit$par[2]; c<-max(fit$par[3],0.1); d<-fit$par[4]

  T_grid  <- seq(min(expiries)*0.5, max(expiries)*1.5, length.out=50)
  iv_fit  <- a + b*exp(-c*T_grid) + d*(T_grid*exp(-c*T_grid))

  # Term structure slope: short-dated minus long-dated
  ts_slope <- atm_ivs[1] - atm_ivs[n]  # positive = backwardation (stress)

  # Implied forward vol between expiry i and i+1
  fwd_var <- diff(atm_ivs^2 * expiries) / diff(expiries)
  fwd_vol <- sqrt(pmax(fwd_var, 0))

  cat("=== Implied Vol Term Structure ===\n")
  cat("Expiry  | ATM IV  | Fitted\n")
  for (i in seq_len(n)) {
    iv_f <- a + b*exp(-c*expiries[i]) + d*(expiries[i]*exp(-c*expiries[i]))
    cat(sprintf("%.2f yr | %.3f   | %.3f\n", expiries[i], atm_ivs[i], iv_f))
  }
  cat(sprintf("Term structure slope (front-back): %+.3f\n", ts_slope))
  cat(sprintf("Shape: %s\n",
              ifelse(ts_slope > 0.02, "Inverted (stress/backwardation)",
              ifelse(ts_slope < -0.02, "Normal (contango)", "Flat"))))

  invisible(list(params=fit$par, T_grid=T_grid, iv_grid=iv_fit,
                 ts_slope=ts_slope, fwd_vol=fwd_vol))
}

# -----------------------------------------------------------------------------
# 9. FULL DERIVATIVES PRICING PIPELINE
# -----------------------------------------------------------------------------

#' Comprehensive options analysis
#' @param S spot price
#' @param K_grid vector of strikes
#' @param T_exp time to expiry
#' @param r risk-free rate
#' @param sigma_atm ATM implied vol
run_derivatives_analysis <- function(S=50000, K_grid=NULL, T_exp=30/365,
                                      r=0, sigma_atm=0.70) {
  if (is.null(K_grid)) {
    # Create a grid around ATM
    K_grid <- S * seq(0.70, 1.30, by=0.05)
  }

  cat("=============================================================\n")
  cat("DERIVATIVES PRICING ANALYSIS\n")
  cat(sprintf("Spot: %.2f, T: %.3f yr, ATM vol: %.2f%%\n",
              S, T_exp, 100*sigma_atm))
  cat("=============================================================\n\n")

  # 1. BS Pricing and Greeks at ATM
  cat("--- BS Greeks at ATM ---\n")
  g_atm <- option_greeks(S, S, T_exp, r, sigma_atm, type="call")

  # 2. Full vol smile table
  cat("\n--- Option Prices and Greeks (Call) ---\n")
  opt_table <- do.call(rbind, lapply(K_grid, function(K) {
    moneyness <- S/K
    p <- bs_price(S, K, T_exp, r, sigma_atm)
    g <- bs_greeks(S, K, T_exp, r, sigma_atm)
    data.frame(K=K, moneyness=round(moneyness,3),
               call=round(p$call,2), put=round(p$put,2),
               delta=round(g$delta_c,4), gamma=round(g$gamma,6),
               vega=round(g$vega,4), theta=round(g$theta_c,5))
  }))
  print(opt_table)

  # 3. IV from market prices (reverse engineer)
  cat("\n--- Implied Vol from Market Prices ---\n")
  # Generate "market" prices with a smile (using SVI)
  k_obs <- log(K_grid / S)
  svi_params <- c(a=sigma_atm^2*T_exp*0.9, b=0.15, rho=-0.3, m=-0.05, sigma=0.10)
  iv_svi <- sqrt(pmax(svi_variance(k_obs, svi_params) / T_exp, 0.01))
  mkt_prices <- sapply(seq_along(K_grid), function(i) {
    bs_price(S, K_grid[i], T_exp, r, iv_svi[i])$call
  })

  # Recover IV
  iv_recovered <- sapply(seq_along(K_grid), function(i) {
    tryCatch(implied_vol_brent(mkt_prices[i], S, K_grid[i], T_exp, r, "call"),
             error=function(e) NA)
  })

  # Fit SVI to recovered IVs
  cat("\n--- SVI Surface Fit ---\n")
  valid_iv <- !is.na(iv_recovered)
  svi_fit_res <- svi_fit(k_obs[valid_iv], iv_recovered[valid_iv], T_exp)

  # 4. MC pricing comparison
  cat("\n--- Monte Carlo vs BS Comparison (ATM Call) ---\n")
  bs_atm  <- bs_price(S, S, T_exp, r, sigma_atm)$call
  mc_gbm  <- mc_option_gbm(S, S, r, sigma_atm, T_exp, n_sim=20000)
  mc_jump <- mc_option_jump_diffusion(S, S, r, sigma_atm*0.8, T_exp,
                                       n_sim=20000, lambda=5, mu_j=-0.05, sigma_j=0.08)
  cat(sprintf("BS price:          %.4f\n", bs_atm))
  cat(sprintf("MC GBM price:      %.4f (se=%.4f)\n", mc_gbm$price, mc_gbm$se))
  cat(sprintf("Jump-Diffusion:    %.4f\n", mc_jump$price))

  invisible(list(opt_table=opt_table, svi=svi_fit_res,
                 iv_smile=data.frame(K=K_grid, k=k_obs, iv=iv_recovered)))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# result <- run_derivatives_analysis(S=50000, T_exp=30/365, sigma_atm=0.70)

# =============================================================================
# EXTENDED DERIVATIVES PRICING: Barrier Options, Asian Options, Calendar Spreads,
# Delta Hedging Simulation, Greeks Sensitivity Analysis, Crypto Derivatives
# =============================================================================

# -----------------------------------------------------------------------------
# Barrier Option Pricing via Monte Carlo
# Down-and-out call: pays BS call payoff if S never hits barrier H < S0
# Common in crypto structured products (e.g., accumulator notes)
# -----------------------------------------------------------------------------
mc_barrier_option <- function(S0, K, H, r, sigma, T, n_steps = 252,
                               n_sim = 10000, type = "down_out_call") {
  dt <- T / n_steps

  # Simulate paths
  Z <- matrix(rnorm(n_sim * n_steps), n_sim, n_steps)
  log_S <- log(S0) + cumsum(matrix((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*t(Z),
                                    n_steps, n_sim), dims=1)
  S_paths <- exp(t(log_S))  # n_sim x n_steps

  # Check barrier
  if (type == "down_out_call") {
    # Knocked out if S ever hits H from above
    knocked_out <- apply(S_paths, 1, function(path) any(path <= H))
    ST <- S_paths[, n_steps]
    payoffs <- ifelse(knocked_out, 0, pmax(ST - K, 0))

  } else if (type == "up_out_call") {
    knocked_out <- apply(S_paths, 1, function(path) any(path >= H))
    ST <- S_paths[, n_steps]
    payoffs <- ifelse(knocked_out, 0, pmax(ST - K, 0))

  } else if (type == "down_in_call") {
    knocked_in <- apply(S_paths, 1, function(path) any(path <= H))
    ST <- S_paths[, n_steps]
    payoffs <- ifelse(knocked_in, pmax(ST - K, 0), 0)
  } else {
    stop("Unknown barrier type")
  }

  price <- exp(-r*T) * mean(payoffs)
  se    <- exp(-r*T) * sd(payoffs) / sqrt(n_sim)
  pct_knocked <- mean(if(grepl("out",type)) apply(S_paths,1,function(p)any(p<=H))
                      else apply(S_paths,1,function(p)any(p>=H)))

  list(price=price, se=se, type=type, barrier=H,
       pct_barrier_events = pct_knocked,
       vanilla_price = bs_price(S0, K, r, 0, sigma, T))
}

# -----------------------------------------------------------------------------
# Asian Option Pricing (Arithmetic Average): MC with control variate
# Asian options are cheaper than vanilla; popular for reducing manipulation risk
# Geometric Asian has closed-form solution used as control variate
# -----------------------------------------------------------------------------
asian_option_mc <- function(S0, K, r, sigma, T, n_steps = 252, n_sim = 10000,
                             option_type = "call") {
  dt <- T / n_steps

  Z <- matrix(rnorm(n_sim * n_steps), n_sim, n_steps)
  log_increments <- (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
  log_S <- t(apply(log_increments, 1, function(row) log(S0) + cumsum(row)))
  S_paths <- exp(log_S)  # n_sim x n_steps

  # Arithmetic average
  S_arith <- rowMeans(S_paths)

  # Geometric average (for control variate)
  S_geom <- exp(rowMeans(log_S))

  # Payoffs
  if (option_type == "call") {
    payoffs_arith <- pmax(S_arith - K, 0)
    payoffs_geom  <- pmax(S_geom - K, 0)
  } else {
    payoffs_arith <- pmax(K - S_arith, 0)
    payoffs_geom  <- pmax(K - S_geom, 0)
  }

  # Geometric Asian closed-form (Kemna & Vorst)
  sigma_g <- sigma * sqrt((2*n_steps+1) / (6*(n_steps+1)))
  r_g     <- 0.5 * (r - 0.5*sigma^2 + sigma_g^2)
  geom_price <- bs_price(S0, K, r_g, 0, sigma_g, T)

  # Control variate correction
  b_cv <- cov(payoffs_arith, payoffs_geom) / var(payoffs_geom)
  price_cv <- exp(-r*T) * (mean(payoffs_arith) - b_cv * (mean(payoffs_geom) - geom_price * exp(r*T)))

  list(
    price_mc = exp(-r*T) * mean(payoffs_arith),
    price_cv = price_cv,
    price_geom_cf = geom_price,
    se_mc = exp(-r*T) * sd(payoffs_arith) / sqrt(n_sim),
    variance_reduction = var(payoffs_arith) / var(payoffs_arith - b_cv*payoffs_geom),
    b_cv = b_cv
  )
}

# -----------------------------------------------------------------------------
# Delta Hedging Simulation: profit/loss from hedging an option
# Perfect hedging in BS world = 0 P&L; in practice: hedging error from discrete
# rebalancing, vol mismatch, and transaction costs
# Critical for crypto market makers running delta-neutral books
# -----------------------------------------------------------------------------
delta_hedge_simulation <- function(S0, K, r, sigma_true, sigma_hedge,
                                    T, n_steps = 21, n_sim = 1000,
                                    fee_per_trade = 0.0005) {
  dt <- T / n_steps

  pnl <- numeric(n_sim)

  for (sim in 1:n_sim) {
    # Simulate price path with true vol
    S <- S0; t_remaining <- T; delta_prev <- 0; hedge_value <- 0

    option_value_t0 <- bs_price(S0, K, r, 0, sigma_hedge, T)
    portfolio_value <- option_value_t0  # start: sell option, receive premium

    for (step in 1:n_steps) {
      t_remaining <- T - (step - 1) * dt
      if (t_remaining <= 0) break

      # Current delta (using hedge vol, not true vol)
      delta_curr <- bs_greeks(S, K, r, 0, sigma_hedge, t_remaining)$delta

      # Rebalance cost
      shares_change <- delta_curr - delta_prev
      portfolio_value <- portfolio_value - shares_change * S - fee_per_trade * abs(shares_change) * S

      # Price path step
      z <- rnorm(1)
      S <- S * exp((r - 0.5*sigma_true^2)*dt + sigma_true*sqrt(dt)*z)
      delta_prev <- delta_curr

      # Accrue interest on portfolio
      portfolio_value <- portfolio_value * exp(r * dt)
      portfolio_value <- portfolio_value + delta_curr * S * (exp(r*dt) - 1)
    }

    # Unwind delta at expiry, pay off option
    option_payoff <- max(S - K, 0)
    pnl[sim] <- portfolio_value + delta_prev * S - option_payoff
  }

  list(
    mean_pnl = mean(pnl),
    sd_pnl   = sd(pnl),
    pnl_sharpe = mean(pnl) / sd(pnl) * sqrt(252),
    pct_profitable = mean(pnl > 0),
    q5_pnl = quantile(pnl, 0.05),
    sigma_true = sigma_true, sigma_hedge = sigma_hedge,
    vol_mismatch = sigma_true - sigma_hedge,
    pnl_distribution = pnl
  )
}

# -----------------------------------------------------------------------------
# Term Structure of Implied Volatility: fit Nelson-Siegel model to vol surface
# Captures level, slope, and curvature of the vol term structure
# Useful for interpolating IV at arbitrary expiries
# -----------------------------------------------------------------------------
vol_term_structure_ns <- function(expiries, atm_vols) {
  # Nelson-Siegel: IV(T) = b0 + b1*(1-exp(-T/tau))/T/tau + b2*((1-exp(-T/tau))/T/tau - exp(-T/tau))
  stopifnot(length(expiries) == length(atm_vols))

  ns_fit <- function(params) {
    b0 <- params[1]; b1 <- params[2]; b2 <- params[3]; tau <- exp(params[4])
    x <- expiries / tau
    fitted <- b0 + b1*(1-exp(-x))/x + b2*((1-exp(-x))/x - exp(-x))
    sum((fitted - atm_vols)^2)
  }

  init <- c(mean(atm_vols), 0, 0, log(0.5))
  opt  <- optim(init, ns_fit, method="BFGS")
  b0 <- opt$par[1]; b1 <- opt$par[2]; b2 <- opt$par[3]; tau <- exp(opt$par[4])

  # Fitted values and term structure prediction
  predict_ns <- function(T_new) {
    x <- T_new / tau
    b0 + b1*(1-exp(-x))/x + b2*((1-exp(-x))/x - exp(-x))
  }

  list(
    b0=b0, b1=b1, b2=b2, tau=tau,
    fitted_vols = predict_ns(expiries),
    r2 = 1 - sum((predict_ns(expiries)-atm_vols)^2)/sum((atm_vols-mean(atm_vols))^2),
    predict_fn = predict_ns,
    # Typical structure: b0 = long-run level, b1 = slope, b2 = hump
    structure_type = ifelse(b1 < 0, "backwardation",
                     ifelse(b1 > 0, "contango", "flat"))
  )
}

# Extended derivatives example:
# barrier <- mc_barrier_option(S0=50000, K=55000, H=40000, r=0.04,
#               sigma=0.70, T=30/365, type="down_out_call")
# asian   <- asian_option_mc(S0=50000, K=50000, r=0.04, sigma=0.70, T=30/365)
# hedge   <- delta_hedge_simulation(S0=50000, K=50000, r=0.04,
#               sigma_true=0.75, sigma_hedge=0.70, T=30/365)
# ts_ns   <- vol_term_structure_ns(expiries=c(7,14,30,60,90)/365,
#               atm_vols=c(0.90,0.80,0.72,0.68,0.65))
