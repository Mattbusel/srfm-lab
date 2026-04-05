# =============================================================================
# crypto_derivatives.R
# Crypto-Specific Derivatives Analytics
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Crypto derivatives are structurally different from
# traditional markets. Perpetual futures (invented by BitMEX) dominate volume;
# their funding mechanism creates predictable flows. Options markets imply
# vol surfaces that reveal market fear and skew. Liquidation cascades are
# unique to leveraged perp markets and can amplify moves dramatically.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

pnorm_ <- pnorm; qnorm_ <- qnorm; dnorm_ <- dnorm

#' Black-Scholes d1
bs_d1 <- function(S, K, r, q, sigma, T_) {
  (log(S/K) + (r - q + 0.5*sigma^2)*T_) / (sigma*sqrt(T_))
}

#' Black-Scholes call (with continuous dividend q)
bs_call <- function(S, K, r, q, sigma, T_) {
  if (T_ <= 0 || sigma <= 0) return(max(S*exp(-q*T_) - K*exp(-r*T_), 0))
  d1 <- bs_d1(S, K, r, q, sigma, T_)
  d2 <- d1 - sigma*sqrt(T_)
  S*exp(-q*T_)*pnorm_(d1) - K*exp(-r*T_)*pnorm_(d2)
}

#' Black-Scholes put
bs_put <- function(S, K, r, q, sigma, T_) {
  if (T_ <= 0) return(max(K*exp(-r*T_) - S*exp(-q*T_), 0))
  d1 <- bs_d1(S, K, r, q, sigma, T_)
  d2 <- d1 - sigma*sqrt(T_)
  K*exp(-r*T_)*pnorm_(-d2) - S*exp(-q*T_)*pnorm_(-d1)
}

#' BS Greeks -- delta, gamma, vega, theta
bs_greeks <- function(S, K, r, q, sigma, T_, type = "call") {
  if (T_ <= 0) return(list(delta=as.numeric(S>K),gamma=0,vega=0,theta=0,rho=0))
  d1 <- bs_d1(S, K, r, q, sigma, T_)
  d2 <- d1 - sigma*sqrt(T_)
  delta <- if (type=="call") exp(-q*T_)*pnorm_(d1) else -exp(-q*T_)*pnorm_(-d1)
  gamma <- exp(-q*T_)*dnorm_(d1) / (S*sigma*sqrt(T_))
  vega  <- S*exp(-q*T_)*dnorm_(d1)*sqrt(T_)
  theta_call <- (-S*exp(-q*T_)*dnorm_(d1)*sigma/(2*sqrt(T_))
                 - r*K*exp(-r*T_)*pnorm_(d2) + q*S*exp(-q*T_)*pnorm_(d1))
  theta <- if (type=="call") theta_call else theta_call + r*K*exp(-r*T_) - q*S*exp(-q*T_)
  list(delta=delta, gamma=gamma, vega=vega, theta=theta/365, rho=NA)
}

# ---------------------------------------------------------------------------
# 2. PERPETUAL FUTURES: FUNDING RATE DYNAMICS
# ---------------------------------------------------------------------------
# Perpetual futures have no expiry. The funding rate keeps mark price ≈ spot.
# Funding rate = clamp(premium_index + interest_rate_differential, -cap, +cap)
# If mark > spot: longs pay shorts (funding > 0); else shorts pay longs.

#' Compute 8-hour funding rate
funding_rate <- function(mark_price, index_price,
                          rate_cap = 0.0075,     # 0.75% cap (Binance)
                          interest_diff = 0.0001) {
  premium <- (mark_price - index_price) / index_price
  rate    <- premium + interest_diff
  clip(rate, -rate_cap, rate_cap)
}

#' Funding P&L for a position over a time series of funding rates
funding_pnl <- function(position_sizes, funding_rates) {
  # Long position: pays when rate > 0, receives when rate < 0
  -position_sizes * funding_rates
}

#' Simulate perpetual mark price (mean-reverting to spot)
simulate_perp <- function(spot_prices, mean_reversion = 0.1,
                           funding_cap = 0.0075, seed = 42L) {
  set.seed(seed)
  n    <- length(spot_prices)
  mark <- numeric(n); mark[1] <- spot_prices[1]
  funding <- numeric(n)
  basis   <- numeric(n)

  for (t in 2:n) {
    # Mark reverts to spot + noise
    noise   <- rnorm(1, 0, 0.002 * spot_prices[t-1])
    mark[t] <- mark[t-1] + mean_reversion * (spot_prices[t] - mark[t-1]) + noise
    # 8h funding rate (assuming 3 per day; use every 8h bar conceptually)
    fr        <- funding_rate(mark[t], spot_prices[t], funding_cap)
    funding[t] <- fr
    basis[t]  <- mark[t] - spot_prices[t]
  }

  data.frame(
    spot    = spot_prices,
    mark    = mark,
    basis   = basis,
    basis_pct = basis / spot_prices,
    funding   = funding,
    cum_funding = cumsum(funding)
  )
}

# ---------------------------------------------------------------------------
# 3. FUNDING RATE ARBITRAGE (CASH-AND-CARRY)
# ---------------------------------------------------------------------------
# Strategy: Long spot + short perpetual. Collect funding when basis > 0.
# P&L = accumulated funding received - borrowing cost - execution costs.

cash_carry_pnl <- function(perp_data, funding_per_period = 3L,
                             borrow_rate_ann = 0.05,
                             tc_open = 0.001, tc_close = 0.001) {
  n  <- nrow(perp_data)
  dt <- 1 / (365 * funding_per_period)   # fraction of year per funding period

  # Funding income (collect when rate > 0, pay when < 0)
  funding_income <- cumsum(perp_data$funding)

  # Borrow cost (USDT loan to buy spot)
  borrow_cost <- cumsum(rep(borrow_rate_ann * dt, n))

  # Net P&L (on notional = 1 unit)
  net_pnl <- funding_income - borrow_cost - (tc_open + tc_close)

  data.frame(
    t              = seq_len(n),
    funding_income = funding_income,
    borrow_cost    = borrow_cost,
    net_pnl        = net_pnl,
    ann_yield      = cumsum(perp_data$funding) / (seq_len(n) / (365 * funding_per_period))
  )
}

# ---------------------------------------------------------------------------
# 4. IMPLIED VOLATILITY (Newton-Raphson)
# ---------------------------------------------------------------------------
# Invert BS formula to find sigma from observed option price.

implied_vol <- function(price_obs, S, K, r, q, T_, type = "call",
                         tol = 1e-8, max_iter = 100L) {
  # Initial guess via Brenner-Subrahmanyam
  sigma <- price_obs / (S * sqrt(T_)) * sqrt(2 * pi)
  sigma <- clip(sigma, 0.01, 5.0)

  for (iter in seq_len(max_iter)) {
    bs_price <- if (type == "call") bs_call(S, K, r, q, sigma, T_)
                else                bs_put(S, K, r, q, sigma, T_)
    vega <- S * exp(-q*T_) * dnorm_(bs_d1(S,K,r,q,sigma,T_)) * sqrt(T_)
    diff_ <- bs_price - price_obs
    if (abs(diff_) < tol) break
    if (abs(vega) < 1e-12) { sigma <- sigma + 0.01; next }
    sigma <- sigma - diff_ / vega
    sigma <- clip(sigma, 0.001, 10.0)
  }
  sigma
}

# ---------------------------------------------------------------------------
# 5. IMPLIED VOL SURFACE
# ---------------------------------------------------------------------------
# Build a term structure + skew surface from market option quotes.

build_vol_surface <- function(strikes, maturities, market_prices,
                               S, r = 0.05, q = 0.0, option_type = "call") {
  n_K <- length(strikes); n_T <- length(maturities)
  iv_surface <- matrix(NA, n_K, n_T)
  for (i in seq_len(n_K)) {
    for (j in seq_len(n_T)) {
      iv_surface[i, j] <- tryCatch(
        implied_vol(market_prices[i, j], S, strikes[i], r, q, maturities[j], option_type),
        error = function(e) NA_real_)
    }
  }
  rownames(iv_surface) <- paste0("K=", round(strikes))
  colnames(iv_surface) <- paste0("T=", round(maturities * 365), "d")
  iv_surface
}

#' Vol surface summary statistics
vol_surface_stats <- function(iv_surface) {
  ATM_idx <- round(nrow(iv_surface) / 2)
  term_structure <- iv_surface[ATM_idx, ]   # ATM across maturities
  skew           <- iv_surface[1, ] - iv_surface[nrow(iv_surface), ]   # 25d put - 25d call IV
  list(atm_term_structure = term_structure,
       skew_by_maturity   = skew,
       surface_mean       = mean(iv_surface, na.rm=TRUE),
       surface_min        = min(iv_surface, na.rm=TRUE),
       surface_max        = max(iv_surface, na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 6. VOL RISK PREMIUM (VRP)
# ---------------------------------------------------------------------------
# VRP = Implied Vol - Realised Vol.
# Selling options (short vol) earns the premium on average.
# In crypto, VRP is large due to demand for crash protection.

vol_risk_premium <- function(implied_vols, realized_vols, window = 30L) {
  vrp  <- implied_vols - realized_vols
  data.frame(
    implied   = implied_vols,
    realized  = realized_vols,
    vrp       = vrp,
    roll_vrp  = {
      n <- length(vrp); out <- rep(NA,n)
      for(i in window:n) out[i] <- mean(vrp[(i-window+1):i], na.rm=TRUE)
      out
    },
    signal    = sign(vrp)   # > 0 -> sell vol; < 0 -> buy vol
  )
}

#' Simulate realised vol for comparison
realised_vol_ann <- function(returns, window = 30L) {
  n  <- length(returns)
  rv <- rep(NA, n)
  for (i in window:n) rv[i] <- sd(returns[(i-window+1):i], na.rm=TRUE) * sqrt(252)
  rv
}

# ---------------------------------------------------------------------------
# 7. LIQUIDATION CASCADE MODELLING
# ---------------------------------------------------------------------------
# Model leveraged long positions on a perpetual as a ladder of margin levels.
# When price drops below margin threshold, position is force-liquidated,
# adding sell pressure -> further price drop -> more liquidations.

simulate_liquidation_cascade <- function(initial_price,
                                          n_traders = 1000L,
                                          mean_lev = 10, sd_lev = 5,
                                          mean_entry = NULL,
                                          price_shock = -0.10,
                                          n_sims = 500L, seed = 42L) {
  set.seed(seed)
  if (is.null(mean_entry)) mean_entry <- initial_price

  total_liq_pct <- numeric(n_sims)
  final_price_drop <- numeric(n_sims)

  for (sim in seq_len(n_sims)) {
    # Each trader: entry price, leverage, notional
    leverages <- pmax(rnorm(n_traders, mean_lev, sd_lev), 1.5)
    entries   <- mean_entry * exp(rnorm(n_traders, 0, 0.05))
    notionals <- rexp(n_traders, 1/10000) + 100   # USDT notional

    # Liquidation price: entry * (1 - 1/leverage)  (maintenance margin ≈ 0)
    liq_prices <- entries * (1 - 1/leverages)

    # Simulate price path
    price <- initial_price * (1 + price_shock)
    n_liq <- 0; cum_sell_vol <- 0

    # Cascade: each liquidation adds sell pressure -> further price drop
    impact_per_unit <- 0.0001 / mean(notionals)   # price impact
    liq_mask <- logical(n_traders)

    for (round_ in 1:20) {
      new_liqs <- !liq_mask & (liq_prices > price)
      if (sum(new_liqs) == 0) break
      sell_vol  <- sum(notionals[new_liqs]) / price
      price     <- price * (1 - impact_per_unit * sell_vol)
      liq_mask  <- liq_mask | new_liqs
      n_liq     <- n_liq + sum(new_liqs)
      cum_sell_vol <- cum_sell_vol + sell_vol
    }

    total_liq_pct[sim]    <- n_liq / n_traders
    final_price_drop[sim] <- (price - initial_price) / initial_price
  }

  list(
    mean_liq_pct   = mean(total_liq_pct),
    p95_liq_pct    = quantile(total_liq_pct, 0.95),
    mean_price_drop = mean(final_price_drop),
    p95_price_drop  = quantile(final_price_drop, 0.05),   # 5th pctile = worst drop
    dist_liq       = quantile(total_liq_pct, c(0.25,0.5,0.75,0.95))
  )
}

# ---------------------------------------------------------------------------
# 8. BASIS TRADING: SPOT VS PERPETUAL CONVERGENCE
# ---------------------------------------------------------------------------
# When basis (mark - spot) is positive, basis trader:
#   - Buys spot, sells perp -> earns convergence + funding
# When basis is negative:
#   - Buys perp, sells spot

run_basis_trade <- function(perp_data, entry_threshold = 0.003,
                              exit_threshold = 0.001,
                              max_hold = 100L, tc = 0.001) {
  n      <- nrow(perp_data)
  basis  <- perp_data$basis_pct
  pnl    <- numeric(n)
  in_trade <- FALSE
  entry_basis <- 0
  entry_t     <- NA

  for (t in seq_len(n)) {
    if (!in_trade) {
      if (abs(basis[t]) > entry_threshold) {
        in_trade    <- TRUE
        entry_basis <- basis[t]
        entry_t     <- t
      }
    } else {
      hold_days <- t - entry_t
      # Exit when basis reverts or max hold exceeded
      if (abs(basis[t]) < exit_threshold || hold_days >= max_hold) {
        pnl[t]   <- (entry_basis - basis[t]) * sign(entry_basis) - tc
        in_trade <- FALSE
      }
    }
  }

  trades  <- which(pnl != 0)
  eq      <- cumprod(1 + pnl)
  data.frame(
    t        = seq_len(n),
    pnl      = pnl,
    equity   = eq,
    sharpe   = {rets<-pnl[pnl!=0]; if(length(rets)>2) mean(rets)/sd(rets)*sqrt(252) else NA},
    n_trades = length(trades)
  )
}

# ---------------------------------------------------------------------------
# 9. CROSS-EXCHANGE BASIS ARBITRAGE
# ---------------------------------------------------------------------------
# BTC price on Exchange A vs Exchange B. When spread > cost, arb.

simulate_cross_exchange_basis <- function(T_ = 500L, seed = 42L) {
  set.seed(seed)
  # Common price process
  common <- 30000 * cumprod(1 + rnorm(T_, 0, 0.01))
  # Exchange-specific noise
  priceA <- common * exp(rnorm(T_, 0, 0.001))
  priceB <- common * exp(rnorm(T_, 0, 0.001))
  data.frame(t=seq_len(T_), priceA=priceA, priceB=priceB,
             spread=priceA-priceB,
             spread_pct=(priceA-priceB)/((priceA+priceB)/2))
}

run_xex_arb <- function(basis_data, threshold_bps = 5, tc_bps = 3) {
  n     <- nrow(basis_data)
  spread_bps <- basis_data$spread_pct * 1e4
  pnl   <- numeric(n)
  for (t in seq_len(n)) {
    sp <- spread_bps[t]
    if (abs(sp) > threshold_bps) {
      pnl[t] <- abs(sp) / 1e4 - tc_bps / 1e4
    }
  }
  list(pnl = pnl, equity = cumprod(1 + pnl),
       sharpe = {r<-pnl[pnl!=0]; if(length(r)>2) mean(r)/sd(r)*sqrt(252) else NA},
       n_arbs = sum(pnl != 0))
}

# ---------------------------------------------------------------------------
# 10. CRYPTO OPTIONS SKEW ANALYSIS
# ---------------------------------------------------------------------------
# In crypto, put skew is typically large (crash protection expensive).
# Skew = IV_25d_put - IV_25d_call; steep skew -> fear.

skew_signal <- function(iv_surface) {
  # Assumes iv_surface rows = strikes (low to high), cols = maturities
  n_K <- nrow(iv_surface)
  if (n_K < 2) return(rep(NA, ncol(iv_surface)))
  # 25-delta put proxy: bottom 25% strike; call: top 25%
  put_row  <- max(1L, as.integer(n_K * 0.25))
  call_row <- min(n_K, as.integer(n_K * 0.75))
  iv_surface[put_row,] - iv_surface[call_row,]   # positive = put skew
}

#' Term structure slope (contango vs backwardation in vol)
vol_term_slope <- function(iv_atm) {
  n <- length(iv_atm)
  if (n < 2) return(NA)
  # Simple slope: short - long term
  short_iv <- mean(iv_atm[1:max(1, n%/%3)], na.rm=TRUE)
  long_iv  <- mean(iv_atm[(2*n%/%3+1):n], na.rm=TRUE)
  list(slope = long_iv - short_iv,
       contango = long_iv > short_iv)   # contango: long vol > short vol
}

# ---------------------------------------------------------------------------
# 11. DELTA-HEDGED OPTIONS P&L (GAMMA SCALPING)
# ---------------------------------------------------------------------------
# Delta-hedge a long gamma position bar-by-bar; P&L = gamma * (dS)^2 / 2 - theta

gamma_scalp_pnl <- function(S_path, K, r, q, sigma_imp, T_total,
                              hedge_freq = 1L) {
  n    <- length(S_path)
  T_rem <- seq(T_total, 0, length.out = n)
  pnl   <- numeric(n)
  delta_prev <- 0
  # Assume we own 1 call option at t=0
  for (t in seq(1, n-1, by = hedge_freq)) {
    T_ <- T_rem[t]
    if (T_ <= 0) break
    g   <- bs_greeks(S_path[t], K, r, q, sigma_imp, T_)
    theta_dt <- g$theta * hedge_freq / 252   # theta cost
    dS   <- S_path[t+1] - S_path[t]
    # Gamma P&L (long gamma = long (dS)^2)
    gamma_pnl <- 0.5 * g$gamma * dS^2
    pnl[t]    <- gamma_pnl + theta_dt
    delta_prev <- g$delta
  }
  list(pnl = pnl, equity = cumprod(1 + pnl / (S_path[1] * 0.01)),
       total_pnl = sum(pnl))
}

# ---------------------------------------------------------------------------
# 12. FUNDING RATE SIGNALS
# ---------------------------------------------------------------------------
# High positive funding -> overcrowded long -> contrarian short signal.
# Funding mean reversion: extreme funding reverts to zero.

funding_signal <- function(funding_rates, lookback = 24L, z_threshold = 2.0) {
  n   <- length(funding_rates)
  mu  <- rep(NA, n); sg <- rep(NA, n)
  for (i in lookback:n) {
    fw <- funding_rates[(i-lookback+1):i]
    mu[i] <- mean(fw, na.rm=TRUE); sg[i] <- sd(fw, na.rm=TRUE)
  }
  z      <- (funding_rates - mu) / pmax(sg, 1e-8)
  signal <- ifelse(z > z_threshold, -1L,
                    ifelse(z < -z_threshold, 1L, 0L))
  data.frame(funding = funding_rates, z = z, signal = signal,
             cum_signal = cumsum(signal))
}

# ---------------------------------------------------------------------------
# 13. MAIN DEMO
# ---------------------------------------------------------------------------

run_crypto_derivatives_demo <- function() {
  cat("=== Crypto Derivatives Analytics Demo ===\n\n")
  set.seed(42)
  T_  <- 500L

  # Spot price path
  spot <- cumprod(1 + rnorm(T_, 0.0005, 0.03)) * 30000

  cat("--- 1. Perpetual Futures Simulation ---\n")
  perp <- simulate_perp(spot, mean_reversion=0.15)
  cat(sprintf("  Mean basis: %.2f (%.3f%%)  |  Mean funding rate: %.4f%%\n",
              mean(perp$basis), mean(perp$basis_pct)*100,
              mean(perp$funding)*100))
  cat(sprintf("  Cumulative funding (500 bars): %.2f%%\n",
              tail(perp$cum_funding,1)*100))

  cat("\n--- 2. Cash-and-Carry P&L ---\n")
  cc <- cash_carry_pnl(perp, funding_per_period=3L, borrow_rate_ann=0.05)
  cat(sprintf("  Net P&L: %.3f  |  Funding income: %.3f  |  Borrow cost: %.3f\n",
              tail(cc$net_pnl,1), tail(cc$funding_income,1), tail(cc$borrow_cost,1)))
  cat(sprintf("  Annualised yield: %.2f%%\n", tail(cc$ann_yield,1)*100))

  cat("\n--- 3. Options Pricing ---\n")
  S0 <- 30000; K_atm <- 30000; T1 <- 30/365; sigma0 <- 0.80; r0 <- 0.05
  c_price <- bs_call(S0, K_atm, r0, 0, sigma0, T1)
  p_price <- bs_put(S0, K_atm, r0, 0, sigma0, T1)
  g_call  <- bs_greeks(S0, K_atm, r0, 0, sigma0, T1)
  cat(sprintf("  ATM Call: $%.2f  Put: $%.2f  |  Delta: %.4f  Vega: %.2f\n",
              c_price, p_price, g_call$delta, g_call$vega))

  cat("\n--- 4. Implied Volatility Smile ---\n")
  strikes    <- S0 * c(0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20)
  maturities <- c(7, 30, 90) / 365
  # Simulate market prices with skew
  mkt_prices <- matrix(NA, length(strikes), length(maturities))
  for (i in seq_along(strikes)) {
    for (j in seq_along(maturities)) {
      # Skewed vol: OTM puts have higher IV
      moneyness <- log(strikes[i] / S0)
      skew_adj  <- -0.30 * moneyness + 0.10 * moneyness^2
      sigma_ij  <- clip(sigma0 + skew_adj, 0.30, 2.0)
      mkt_prices[i,j] <- bs_call(S0, strikes[i], r0, 0, sigma_ij, maturities[j])
    }
  }
  iv_surf <- build_vol_surface(strikes, maturities, mkt_prices, S0, r0)
  cat("  IV Surface (rows=strike, cols=maturity):\n")
  print(round(iv_surf, 3))

  cat("\n--- 5. Vol Risk Premium ---\n")
  imp_vol_series <- 0.80 + rnorm(T_, 0, 0.05)
  ret_series     <- diff(log(spot))
  rvol           <- realised_vol_ann(ret_series)
  vrp_data       <- vol_risk_premium(imp_vol_series[-1], rvol[!is.na(rvol)])
  cat(sprintf("  Mean VRP: %.3f  |  Positive fraction: %.1f%%\n",
              mean(vrp_data$vrp, na.rm=TRUE),
              mean(vrp_data$vrp > 0, na.rm=TRUE)*100))

  cat("\n--- 6. Liquidation Cascade ---\n")
  liq_sim <- simulate_liquidation_cascade(
    initial_price=30000, n_traders=500L, mean_lev=10, price_shock=-0.10,
    n_sims=200L)
  cat(sprintf("  Mean pct liquidated: %.1f%%  |  P95: %.1f%%\n",
              liq_sim$mean_liq_pct*100, liq_sim$p95_liq_pct*100))
  cat(sprintf("  Mean price drop (incl. cascade): %.2f%%  |  P5: %.2f%%\n",
              liq_sim$mean_price_drop*100, liq_sim$p95_price_drop*100))

  cat("\n--- 7. Basis Trading ---\n")
  bt <- run_basis_trade(perp, entry_threshold=0.005, exit_threshold=0.001)
  cat(sprintf("  Basis trades: %d  |  Sharpe: %.3f\n",
              bt$n_trades[1], bt$sharpe[1]))

  cat("\n--- 8. Cross-Exchange Arbitrage ---\n")
  xex_data <- simulate_cross_exchange_basis(T_=T_)
  xex_arb  <- run_xex_arb(xex_data, threshold_bps=3, tc_bps=2)
  cat(sprintf("  Arb opportunities: %d  |  Sharpe: %.3f  |  Equity: %.3fx\n",
              xex_arb$n_arbs, xex_arb$sharpe, tail(xex_arb$equity,1)))

  cat("\n--- 9. Funding Rate Signal ---\n")
  fr_signal <- funding_signal(perp$funding, lookback=24L, z_threshold=1.5)
  n_long  <- sum(fr_signal$signal ==  1, na.rm=TRUE)
  n_short <- sum(fr_signal$signal == -1, na.rm=TRUE)
  cat(sprintf("  Long signals: %d  |  Short signals: %d\n", n_long, n_short))

  cat("\n--- 10. Vol Surface Skew ---\n")
  skew <- skew_signal(iv_surf)
  cat("  Skew by maturity (put IV - call IV):", round(skew, 4), "\n")
  vs   <- vol_surface_stats(iv_surf)
  ts   <- vol_term_slope(vs$atm_term_structure)
  cat(sprintf("  Vol term slope: %.4f  |  Contango: %s\n",
              ts$slope, ts$contango))

  cat("\n--- 11. Gamma Scalping P&L ---\n")
  S_path <- 30000 * cumprod(1 + rnorm(21, 0, 0.03))  # 3 week path
  gs <- gamma_scalp_pnl(S_path, K=30000, r=0.05, q=0, sigma_imp=0.80,
                          T_total=21/365)
  cat(sprintf("  Gamma scalp 21-day P&L: $%.2f\n", gs$total_pnl))

  cat("\nDone.\n")
  invisible(list(perp=perp, iv_surf=iv_surf, liq=liq_sim, cc=cc))
}

if (interactive()) {
  deriv_results <- run_crypto_derivatives_demo()
}

# ---------------------------------------------------------------------------
# 14. OPTIONS PORTFOLIO GREEKS AGGREGATION
# ---------------------------------------------------------------------------
# For a book of options, aggregate delta, gamma, vega across all positions.

options_book_greeks <- function(positions) {
  # positions: list of lists with fields S,K,r,q,sigma,T_,type,qty
  agg <- list(delta=0, gamma=0, vega=0, theta=0)
  for (pos in positions) {
    g <- bs_greeks(pos$S, pos$K, pos$r, pos$q, pos$sigma, pos$T_, pos$type)
    agg$delta <- agg$delta + pos$qty * g$delta
    agg$gamma <- agg$gamma + pos$qty * g$gamma
    agg$vega  <- agg$vega  + pos$qty * g$vega
    agg$theta <- agg$theta + pos$qty * g$theta
  }
  agg
}

# ---------------------------------------------------------------------------
# 15. PERPETUAL FUNDING RATE MEAN REVERSION STRATEGY
# ---------------------------------------------------------------------------

funding_mean_reversion <- function(funding_rates, threshold_sd = 2.0,
                                    holding_periods = 8L, tc = 0.001) {
  n   <- length(funding_rates)
  mu  <- mean(funding_rates, na.rm=TRUE)
  sg  <- sd(funding_rates, na.rm=TRUE)
  z   <- (funding_rates - mu) / max(sg, 1e-8)

  pnl   <- numeric(n); in_trade <- FALSE; hold <- 0L; trade_dir <- 0L
  for (t in seq_len(n)) {
    if (!in_trade) {
      if (z[t] > threshold_sd)  { in_trade <- TRUE; trade_dir <- -1L; hold <- 0L; pnl[t] <- -tc }
      if (z[t] < -threshold_sd) { in_trade <- TRUE; trade_dir <-  1L; hold <- 0L; pnl[t] <- -tc }
    } else {
      hold <- hold + 1L
      pnl[t] <- trade_dir * funding_rates[t]
      if (hold >= holding_periods || abs(z[t]) < 0.5) {
        pnl[t] <- pnl[t] - tc; in_trade <- FALSE
      }
    }
  }
  list(pnl=pnl, equity=cumprod(1+pnl),
       sharpe=sharpe_ratio(pnl[pnl!=0]),
       n_trades=sum(diff(c(FALSE,in_trade))==1))
}

# ---------------------------------------------------------------------------
# 16. CRYPTO OPTIONS: BINARY / DIGITAL OPTIONS
# ---------------------------------------------------------------------------
# Digital call pays $1 if S_T > K, else 0.
# Price = e^{-rT} * N(d2)

digital_call_price <- function(S, K, r, q, sigma, T_) {
  if (T_ <= 0) return(as.numeric(S > K))
  d2 <- bs_d1(S,K,r,q,sigma,T_) - sigma*sqrt(T_)
  exp(-r*T_) * pnorm(d2)
}

digital_put_price <- function(S, K, r, q, sigma, T_) {
  1 - digital_call_price(S,K,r,q,sigma,T_) / exp(-r*T_) * exp(-r*T_)
}

# ---------------------------------------------------------------------------
# 17. VARIANCE SWAP PRICING
# ---------------------------------------------------------------------------
# Fair variance swap strike K_var ≈ (2/T) * sum_K [C(K)/K^2 + P(K)/K^2] * dK
# Financial intuition: variance swaps let you trade realised vol vs implied vol
# without delta hedging; widely used in crypto vol risk premium strategies.

variance_swap_fair_strike <- function(S, r, q, sigma_surface,
                                       strikes, T_) {
  n_K  <- length(strikes)
  dK   <- c(diff(strikes), strikes[n_K] - strikes[n_K-1])
  contrib <- numeric(n_K)
  for (i in seq_len(n_K)) {
    K <- strikes[i]; sig <- sigma_surface[i]
    if (K < S) p <- bs_put(S,  K, r, q, sig, T_)
    else       p <- bs_call(S, K, r, q, sig, T_)
    contrib[i] <- p / (K^2) * dK[i]
  }
  K_var <- 2 / T_ * sum(contrib) * exp(r * T_)
  list(K_var = K_var, fair_vol = sqrt(K_var))
}

# ---------------------------------------------------------------------------
# 18. EXTENDED DERIVATIVES DEMO
# ---------------------------------------------------------------------------

run_derivatives_extended_demo <- function() {
  cat("=== Crypto Derivatives Extended Demo ===\n\n")
  set.seed(42); S0 <- 30000; T_ <- 300L

  spot  <- cumprod(1 + rnorm(T_, 0.0005, 0.025)) * S0
  perp  <- simulate_perp(spot)

  cat("--- Options Book Greeks ---\n")
  book <- list(
    list(S=S0,K=30000,r=0.05,q=0,sigma=0.8,T_=30/365,type="call",qty=10),
    list(S=S0,K=28000,r=0.05,q=0,sigma=0.9,T_=30/365,type="put",qty=-5),
    list(S=S0,K=32000,r=0.05,q=0,sigma=0.75,T_=30/365,type="call",qty=-8)
  )
  greeks <- options_book_greeks(book)
  cat(sprintf("  Book delta=%.4f  gamma=%.6f  vega=%.2f  theta=%.2f\n",
              greeks$delta, greeks$gamma, greeks$vega, greeks$theta))

  cat("\n--- Funding Mean Reversion ---\n")
  fmr <- funding_mean_reversion(perp$funding, threshold_sd=1.5, holding_periods=8L)
  cat(sprintf("  Trades: %d  Sharpe: %.3f\n", fmr$n_trades, fmr$sharpe))

  cat("\n--- Digital Options ---\n")
  dc <- digital_call_price(S0, 32000, 0.05, 0, 0.8, 30/365)
  dp <- digital_put_price(S0,  28000, 0.05, 0, 0.9, 30/365)
  cat(sprintf("  Digital call (K=32000): $%.4f  Digital put (K=28000): $%.4f\n",
              dc, dp))

  cat("\n--- Variance Swap ---\n")
  strikes   <- S0 * seq(0.70, 1.30, by=0.05)
  sigma_vec <- 0.80 + seq(0.15,-0.05,length.out=length(strikes))   # skew
  vs <- variance_swap_fair_strike(S0, 0.05, 0, sigma_vec, strikes, 30/365)
  cat(sprintf("  Fair variance strike: %.4f  Fair vol: %.1f%%\n",
              vs$K_var, vs$fair_vol*100))

  cat("\n--- Implied Vol from Digital Price ---\n")
  # Back-solve for sigma from digital call price
  target_dc <- 0.35
  sigma_imp <- tryCatch(
    uniroot(function(s) digital_call_price(S0,30500,0.05,0,s,30/365)-target_dc,
            c(0.01,5))$root,
    error=function(e) NA)
  cat(sprintf("  Implied vol from digital price %.2f: %.1f%%\n",
              target_dc, sigma_imp*100))

  invisible(list(greeks=greeks, fmr=fmr, vs=vs))
}

if (interactive()) {
  deriv_ext <- run_derivatives_extended_demo()
}

# =============================================================================
# SECTION: PERPETUAL FUNDING ARBITRAGE WITH POSITION LIMITS
# =============================================================================
# In basis trading, the funding arbitrage profit depends on funding rate,
# holding cost, and position limits imposed by margin requirements.

funding_arb_pnl <- function(spot_prices, funding_rates, entry_day,
                             size = 1, margin_rate = 0.1) {
  # Short perp + hold spot (cash-and-carry for perps)
  T     <- length(spot_prices)
  pnl   <- 0
  for (t in seq(entry_day, T)) {
    # Funding collected by short perp position
    pnl <- pnl + funding_rates[t] * size * spot_prices[t]
    # Margin call check
    margin_req <- margin_rate * size * spot_prices[t]
    if (margin_req > size * spot_prices[entry_day]) {
      cat("Margin call at day", t, "\n")
      break
    }
  }
  pnl
}

# =============================================================================
# SECTION: IMPLIED VOLATILITY SURFACE INTERPOLATION
# =============================================================================
# Simple bilinear interpolation on the vol surface grid.

vol_surface_interp <- function(surface, strikes, expiries, K_query, T_query) {
  # surface: matrix [length(strikes) x length(expiries)]
  # Returns interpolated vol at (K_query, T_query)
  k1 <- max(which(strikes <= K_query)); k2 <- min(which(strikes >= K_query))
  t1 <- max(which(expiries <= T_query)); t2 <- min(which(expiries >= T_query))
  k1 <- max(1, min(k1, length(strikes))); k2 <- max(1, min(k2, length(strikes)))
  t1 <- max(1, min(t1, length(expiries))); t2 <- max(1, min(t2, length(expiries)))
  if (k1 == k2 && t1 == t2) return(surface[k1, t1])
  wk <- if (k1 == k2) 0.5 else (K_query - strikes[k1]) / (strikes[k2] - strikes[k1] + 1e-10)
  wt <- if (t1 == t2) 0.5 else (T_query - expiries[t1]) / (expiries[t2] - expiries[t1] + 1e-10)
  v11 <- surface[k1, t1]; v12 <- surface[k1, t2]
  v21 <- surface[k2, t1]; v22 <- surface[k2, t2]
  (1-wk)*(1-wt)*v11 + (1-wk)*wt*v12 + wk*(1-wt)*v21 + wk*wt*v22
}

# =============================================================================
# SECTION: REALIZED VARIANCE AND VOLATILITY RISK PREMIUM
# =============================================================================
# VRP = implied variance - realized variance. Positive VRP suggests
# options are overpriced; selling vol strategies should be profitable.

realized_variance_log <- function(prices, ann_factor = 252) {
  lret <- diff(log(prices))
  mean(lret^2) * ann_factor
}

vol_risk_premium_series <- function(implied_vols, prices, window = 21,
                                     ann_factor = 252) {
  # implied_vols: daily IV, prices: daily close
  T   <- length(prices)
  rv  <- rep(NA_real_, T)
  vrp <- rep(NA_real_, T)
  for (t in seq(window+1, T)) {
    rv[t]  <- realized_variance_log(prices[(t-window):t], ann_factor)
    vrp[t] <- implied_vols[t]^2 - rv[t]
  }
  list(rv = rv, vrp = vrp)
}

# =============================================================================
# SECTION: BARRIER OPTIONS — DOWN-AND-OUT CALL (Monte Carlo)
# =============================================================================
# Crypto options often have barrier features tied to liquidation levels.
# Price a down-and-out call by simulation.

barrier_call_mc <- function(S0, K, H, r, sigma, T_years,
                             n_paths = 5000, n_steps = 100) {
  dt    <- T_years / n_steps
  disc  <- exp(-r * T_years)
  payoffs <- numeric(n_paths)
  for (i in seq_len(n_paths)) {
    S <- S0
    knocked <- FALSE
    for (j in seq_len(n_steps)) {
      S <- S * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*rnorm(1))
      if (S <= H) { knocked <- TRUE; break }
    }
    payoffs[i] <- if (!knocked) max(S - K, 0) else 0
  }
  disc * mean(payoffs)
}

# =============================================================================
# SECTION: OPTION GREEKS VIA FINITE DIFFERENCE
# =============================================================================

bs_price_fd <- function(S, K, r, sigma, T_years, is_call = TRUE) {
  # Black-Scholes price (base R, no packages)
  d1 <- (log(S/K) + (r + 0.5*sigma^2)*T_years) / (sigma*sqrt(T_years))
  d2 <- d1 - sigma*sqrt(T_years)
  pnorm_d1 <- pnorm(d1); pnorm_d2 <- pnorm(d2)
  if (is_call) S*pnorm_d1 - K*exp(-r*T_years)*pnorm_d2
  else          K*exp(-r*T_years)*pnorm(-d2) - S*pnorm(-d1)
}

compute_greeks_fd <- function(S, K, r, sigma, T_years, is_call = TRUE,
                               dS = 0.01, dSigma = 0.001, dT = 1/365) {
  price  <- bs_price_fd(S, K, r, sigma, T_years, is_call)
  delta  <- (bs_price_fd(S+dS, K, r, sigma, T_years, is_call) -
             bs_price_fd(S-dS, K, r, sigma, T_years, is_call)) / (2*dS)
  gamma  <- (bs_price_fd(S+dS, K, r, sigma, T_years, is_call) -
             2*price +
             bs_price_fd(S-dS, K, r, sigma, T_years, is_call)) / (dS^2)
  vega   <- (bs_price_fd(S, K, r, sigma+dSigma, T_years, is_call) -
             bs_price_fd(S, K, r, sigma-dSigma, T_years, is_call)) / (2*dSigma)
  theta  <- (bs_price_fd(S, K, r, sigma, T_years-dT, is_call) - price) / dT
  list(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)
}

# =============================================================================
# SECTION: PERPETUAL OPTION PRICING (Kirk's approximation)
# =============================================================================
# Crypto perpetual options (everlasting options) have no expiry; price via
# the risk-neutral steady-state under GBM.

perpetual_american_put <- function(S, K, r, sigma) {
  # Analytical formula for perpetual American put
  if (r <= 0) return(max(K - S, 0))
  beta <- 0.5 - r/sigma^2 - sqrt((r/sigma^2 - 0.5)^2 + 2*r/sigma^2)
  S_star <- K * beta / (beta - 1)
  if (S >= S_star) {
    # Not optimal to exercise: price as function of S
    (K - S_star) * (S / S_star)^beta
  } else {
    K - S  # exercise immediately
  }
}

# =============================================================================
# SECTION: FINAL DEMO
# =============================================================================

run_derivatives_final_demo <- function() {
  set.seed(42)
  T <- 100
  spot <- cumprod(c(50000, exp(rnorm(T-1, 0, 0.03))))
  fr   <- rnorm(T, 0.0001, 0.0005)

  cat("--- Funding Arb PnL (short perp + hold spot) ---\n")
  pnl <- funding_arb_pnl(spot, fr, entry_day = 1, size = 0.1)
  cat("Total funding collected:", round(pnl, 2), "USD\n")

  cat("\n--- VRP Series (last 5) ---\n")
  iv  <- rep(0.7, T) + rnorm(T, 0, 0.05)
  vrp <- vol_risk_premium_series(iv, spot)
  cat("Mean VRP:", round(mean(vrp$vrp, na.rm=TRUE), 4), "\n")

  cat("\n--- Barrier Option (down-and-out call) ---\n")
  bo <- barrier_call_mc(S0=50000, K=52000, H=45000, r=0.05, sigma=0.7,
                        T_years=0.1, n_paths=2000, n_steps=50)
  cat("Barrier call price:", round(bo, 2), "USD\n")

  cat("\n--- Greeks via Finite Difference ---\n")
  g <- compute_greeks_fd(S=50000, K=50000, r=0.05, sigma=0.7,
                          T_years=0.1, is_call=TRUE)
  cat(sprintf("Price=%.1f  Delta=%.4f  Gamma=%.6f  Vega=%.2f  Theta=%.2f\n",
              g$price, g$delta, g$gamma, g$vega, g$theta))

  cat("\n--- Perpetual American Put ---\n")
  pp <- perpetual_american_put(S=48000, K=50000, r=0.05, sigma=0.7)
  cat("Perpetual put price:", round(pp, 2), "USD\n")

  invisible(list(pnl=pnl, vrp=vrp, barrier=bo, greeks=g))
}

if (interactive()) {
  deriv_final <- run_derivatives_final_demo()
}

# =============================================================================
# SECTION: TERM STRUCTURE OF FUNDING RATES
# =============================================================================
# Model the funding rate term structure as mean-reverting (OU process).
# Estimate half-life and long-run mean from historical funding data.

ou_fit_funding <- function(funding_rates, dt = 1/365) {
  # Estimate OU parameters via OLS: f_{t+1} = (1-k*dt)*f_t + k*theta*dt + e
  n   <- length(funding_rates)
  y   <- funding_rates[-1]
  x   <- funding_rates[-n]
  fit <- lm(y ~ x)
  b   <- coef(fit)
  phi <- b["x"]  # AR(1) coefficient
  k   <- -log(phi) / dt  # mean-reversion speed
  theta <- b["(Intercept)"] / (1 - phi)  # long-run mean
  sigma_e <- sd(resid(fit))
  sigma_ou <- sigma_e / sqrt(dt)
  half_life <- log(2) / max(k, 1e-9)
  list(k=k, theta=theta, sigma=sigma_ou, half_life_days=half_life*365)
}

# Simulate OU funding rate path
simulate_ou_funding <- function(f0, k, theta, sigma, n_steps, dt=1/365) {
  f <- numeric(n_steps); f[1] <- f0
  for (t in 2:n_steps)
    f[t] <- f[t-1] + k*(theta - f[t-1])*dt + sigma*sqrt(dt)*rnorm(1)
  f
}

# =============================================================================
# SECTION: OPTION STRATEGIES — STRADDLE / STRANGLE
# =============================================================================
# Straddle: buy ATM call + ATM put. Profits from large moves in either direction.
# Useful as a volatility bet ahead of major crypto events.

straddle_pnl <- function(S_expiry, K, premium_paid) {
  # Payoff of long straddle at expiry
  payoff <- abs(S_expiry - K)
  payoff - premium_paid
}

strangle_pnl <- function(S_expiry, K_put, K_call, premium_paid) {
  payoff <- pmax(K_put - S_expiry, 0) + pmax(S_expiry - K_call, 0)
  payoff - premium_paid
}

# Expected PnL of straddle under log-normal
straddle_expected_pnl <- function(S0, K, sigma, T_years, r=0) {
  # E[|S_T - K|] under log-normal - premium
  d1 <- (log(S0/K) + (r + 0.5*sigma^2)*T_years) / (sigma*sqrt(T_years))
  d2 <- d1 - sigma*sqrt(T_years)
  call <- S0*pnorm(d1) - K*exp(-r*T_years)*pnorm(d2)
  put  <- K*exp(-r*T_years)*pnorm(-d2) - S0*pnorm(-d1)
  list(straddle_price = call + put,
       breakeven_up   = K + call + put,
       breakeven_dn   = K - call - put)
}

# =============================================================================
# SECTION: RISK REVERSAL AND BUTTERFLY
# =============================================================================

risk_reversal_price <- function(S, K_call, K_put, sigma_call, sigma_put,
                                 r, T_years) {
  # 25-delta risk reversal: call25 - put25 price difference
  # Measures skew premium in the market
  call_p <- bs_price_fd(S, K_call, r, sigma_call, T_years, is_call=TRUE)
  put_p  <- bs_price_fd(S, K_put,  r, sigma_put,  T_years, is_call=FALSE)
  call_p - put_p   # positive = calls more expensive (upside skew)
}

butterfly_price <- function(S, K_lo, K_atm, K_hi, sigma_lo, sigma_atm,
                              sigma_hi, r, T_years) {
  # Butterfly: long K_lo call + short 2x K_atm call + long K_hi call
  c_lo  <- bs_price_fd(S, K_lo,  r, sigma_lo,  T_years, TRUE)
  c_atm <- bs_price_fd(S, K_atm, r, sigma_atm, T_years, TRUE)
  c_hi  <- bs_price_fd(S, K_hi,  r, sigma_hi,  T_years, TRUE)
  c_lo - 2*c_atm + c_hi
}

if (interactive()) {
  S <- 50000; r <- 0.05; sigma <- 0.70; T <- 0.1
  cat("--- Straddle Expected PnL ---\n")
  st <- straddle_expected_pnl(S, S, sigma, T, r)
  cat("Straddle price:", round(st$straddle_price, 2),
      "  Breakeven up:", round(st$breakeven_up, 0), "\n")

  cat("\n--- OU Funding Rate Fit ---\n")
  set.seed(7)
  fr <- simulate_ou_funding(f0=0.0001, k=50, theta=0.0001, sigma=0.002,
                             n_steps=200)
  ou_p <- ou_fit_funding(fr)
  cat("k:", round(ou_p$k, 1), "  theta:", round(ou_p$theta, 6),
      "  half-life (days):", round(ou_p$half_life_days, 1), "\n")

  cat("\n--- Risk Reversal ---\n")
  rr <- risk_reversal_price(S, K_call=55000, K_put=45000,
                             sigma_call=0.75, sigma_put=0.65, r=r, T_years=T)
  cat("25-delta RR:", round(rr, 2), "USD\n")
}
