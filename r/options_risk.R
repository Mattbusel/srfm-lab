# options_risk.R
# Production options risk analytics: BS pricing, Greeks, implied vol,
# SVI vol surface, no-arbitrage checks, scenario P&L, ggplot2 reports.
#
# Dependencies: tidyverse, ggplot2, stats (base)
# Author: srfm-lab

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(scales)
})

# ===========================================================================
# 1. Black-Scholes pricing and Greeks (vectorized)
# ===========================================================================

#' Standard normal PDF
dnorm_std <- function(x) dnorm(x, 0, 1)

#' Standard normal CDF
pnorm_std <- function(x) pnorm(x, 0, 1)

#' Compute BS d1 and d2
#' @param S  underlying price (vector)
#' @param K  strike price (vector)
#' @param r  risk-free rate (continuous, annual)
#' @param q  continuous dividend yield
#' @param sigma  implied vol (annual)
#' @param tau  time to expiry in years
bs_d1d2 <- function(S, K, r, q, sigma, tau) {
  d1 <- (log(S / K) + (r - q + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau))
  d2 <- d1 - sigma * sqrt(tau)
  list(d1 = d1, d2 = d2)
}

#' Black-Scholes call price (vectorized)
bs_call <- function(S, K, r, q, sigma, tau) {
  stopifnot(all(tau > 0), all(sigma > 0), all(S > 0), all(K > 0))
  dd  <- bs_d1d2(S, K, r, q, sigma, tau)
  Fwd <- S * exp(-q * tau)
  Disc <- exp(-r * tau)
  Disc * (Fwd * pnorm_std(dd$d1) - K * pnorm_std(dd$d2))
}

#' Black-Scholes put price (vectorized)
bs_put <- function(S, K, r, q, sigma, tau) {
  stopifnot(all(tau > 0), all(sigma > 0), all(S > 0), all(K > 0))
  dd  <- bs_d1d2(S, K, r, q, sigma, tau)
  Fwd <- S * exp(-q * tau)
  Disc <- exp(-r * tau)
  Disc * (K * pnorm_std(-dd$d2) - Fwd * pnorm_std(-dd$d1))
}

#' BS price dispatcher
#' @param type "call" or "put"
bs_price <- function(S, K, r, q, sigma, tau, type = "call") {
  if (type == "call") bs_call(S, K, r, q, sigma, tau)
  else if (type == "put") bs_put(S, K, r, q, sigma, tau)
  else stop("type must be 'call' or 'put'")
}

#' BS Delta
#' @return vectorized delta
bs_delta <- function(S, K, r, q, sigma, tau, type = "call") {
  dd <- bs_d1d2(S, K, r, q, sigma, tau)
  e_mq <- exp(-q * tau)
  if (type == "call") e_mq * pnorm_std(dd$d1)
  else e_mq * (pnorm_std(dd$d1) - 1)
}

#' BS Gamma (same for calls and puts)
bs_gamma <- function(S, K, r, q, sigma, tau) {
  dd <- bs_d1d2(S, K, r, q, sigma, tau)
  exp(-q * tau) * dnorm_std(dd$d1) / (S * sigma * sqrt(tau))
}

#' BS Vega -- dV/d(sigma), in price units per unit vol change
bs_vega <- function(S, K, r, q, sigma, tau) {
  dd <- bs_d1d2(S, K, r, q, sigma, tau)
  S * exp(-q * tau) * dnorm_std(dd$d1) * sqrt(tau)
}

#' BS Theta -- dV/dt per calendar day
bs_theta <- function(S, K, r, q, sigma, tau, type = "call") {
  dd <- bs_d1d2(S, K, r, q, sigma, tau)
  term1 <- -S * exp(-q * tau) * dnorm_std(dd$d1) * sigma / (2 * sqrt(tau))
  if (type == "call") {
    term2 <- -r * K * exp(-r * tau) * pnorm_std(dd$d2)
    term3 <-  q * S * exp(-q * tau) * pnorm_std(dd$d1)
  } else {
    term2 <-  r * K * exp(-r * tau) * pnorm_std(-dd$d2)
    term3 <- -q * S * exp(-q * tau) * pnorm_std(-dd$d1)
  }
  (term1 + term2 + term3) / 365  # per calendar day
}

#' BS Rho -- dV/dr per unit rate change
bs_rho <- function(S, K, r, q, sigma, tau, type = "call") {
  dd <- bs_d1d2(S, K, r, q, sigma, tau)
  if (type == "call") K * tau * exp(-r * tau) * pnorm_std(dd$d2)
  else -K * tau * exp(-r * tau) * pnorm_std(-dd$d2)
}

#' BS Vanna -- d2V / (dS d_sigma)
bs_vanna <- function(S, K, r, q, sigma, tau) {
  dd  <- bs_d1d2(S, K, r, q, sigma, tau)
  vega <- bs_vega(S, K, r, q, sigma, tau)
  -vega * dd$d2 / (sigma * S)
}

#' BS Volga -- d2V / d_sigma^2
bs_volga <- function(S, K, r, q, sigma, tau) {
  dd   <- bs_d1d2(S, K, r, q, sigma, tau)
  vega <- bs_vega(S, K, r, q, sigma, tau)
  vega * dd$d1 * dd$d2 / sigma
}

#' Compute all Greeks for a data frame of options
#' @param df data.frame with columns: S, K, r, q, sigma, tau, type
#' @return df augmented with price, delta, gamma, vega, theta, rho, vanna, volga
compute_greeks <- function(df) {
  df %>%
    mutate(
      price  = bs_price(S, K, r, q, sigma, tau, type),
      delta  = bs_delta(S, K, r, q, sigma, tau, type),
      gamma  = bs_gamma(S, K, r, q, sigma, tau),
      vega   = bs_vega(S, K, r, q, sigma, tau),
      theta  = bs_theta(S, K, r, q, sigma, tau, type),
      rho    = bs_rho(S, K, r, q, sigma, tau, type),
      vanna  = bs_vanna(S, K, r, q, sigma, tau),
      volga  = bs_volga(S, K, r, q, sigma, tau)
    )
}

# ===========================================================================
# 2. Implied volatility solver -- Newton-Raphson
# ===========================================================================

#' Solve implied volatility via Newton-Raphson iteration
#' @param market_price  observed market price of the option
#' @param S, K, r, q, tau, type  option parameters (scalars)
#' @param tol  convergence tolerance on price difference
#' @param max_iter  maximum Newton iterations
#' @return implied vol (scalar), NA if did not converge
implied_vol_nr <- function(market_price, S, K, r, q, tau, type = "call",
                            tol = 1e-8, max_iter = 200) {
  # Intrinsic value check
  intrinsic <- if (type == "call") max(S * exp(-q * tau) - K * exp(-r * tau), 0)
               else max(K * exp(-r * tau) - S * exp(-q * tau), 0)
  if (market_price < intrinsic - tol) return(NA_real_)

  # Initial guess: Brenner-Subrahmanyam approximation
  sigma <- sqrt(2 * pi / tau) * market_price / S

  # Fallback to 0.2 if initial guess is nonsensical
  if (!is.finite(sigma) || sigma <= 0) sigma <- 0.20

  for (i in seq_len(max_iter)) {
    price <- bs_price(S, K, r, q, sigma, tau, type)
    v     <- bs_vega(S, K, r, q, sigma, tau)

    diff <- price - market_price
    if (abs(diff) < tol) return(sigma)

    # Safeguard against near-zero vega
    if (abs(v) < 1e-12) break

    sigma_new <- sigma - diff / v

    # Keep vol in a reasonable range to avoid divergence
    sigma_new <- max(1e-6, min(sigma_new, 10.0))

    if (abs(sigma_new - sigma) < tol * 1e-3) return(sigma_new)
    sigma <- sigma_new
  }

  warning(sprintf("implied_vol_nr did not converge for price=%.6f", market_price))
  NA_real_
}

#' Vectorized wrapper for implied_vol_nr
#' @param market_prices  numeric vector
#' @param S, K, r, q, tau, type  can be vectors of the same length
implied_vol_vec <- function(market_prices, S, K, r, q, tau, type = "call",
                             tol = 1e-8, max_iter = 200) {
  n <- length(market_prices)
  type <- rep_len(type, n)
  mapply(implied_vol_nr, market_prices, S, K, r, q, tau, type,
         MoreArgs = list(tol = tol, max_iter = max_iter))
}

# ===========================================================================
# 3. SVI vol surface parameterization
# ===========================================================================
# Raw SVI: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
# where k = log(K/F), w = sigma_bs^2 * tau, F = forward price

#' SVI total variance function
#' @param k     log-moneyness vector: log(K/F)
#' @param a,b,rho_svi,m,nu  SVI parameters
svi_w <- function(k, a, b, rho_svi, m, nu) {
  a + b * (rho_svi * (k - m) + sqrt((k - m)^2 + nu^2))
}

#' SVI implied vol as a function of log-moneyness
svi_vol <- function(k, tau, a, b, rho_svi, m, nu) {
  w <- svi_w(k, a, b, rho_svi, m, nu)
  w <- pmax(w, 1e-12)  # clamp to avoid sqrt of negative
  sqrt(w / tau)
}

#' Fit SVI parameters to observed (k, sigma_market) pairs via optim()
#' @param k     vector of log-moneyness values
#' @param sigma vector of observed implied vols
#' @param tau   scalar time to expiry (years)
#' @return named list with a, b, rho_svi, m, nu and fit diagnostics
fit_svi <- function(k, sigma, tau) {
  stopifnot(length(k) == length(sigma), tau > 0)

  w_obs <- sigma^2 * tau  # total variance

  # Objective: sum of squared differences in total variance
  objective <- function(params) {
    a <- params[1]; b <- params[2]; rho_svi <- params[3]
    m <- params[4]; nu <- params[5]

    # SVI parameter constraints (necessary for no-arbitrage):
    # b >= 0, |rho_svi| < 1, nu > 0, a + b*nu*(1 - |rho_svi|) >= 0
    if (b < 0 || abs(rho_svi) >= 1 || nu <= 0) return(1e12)
    if (a + b * nu * (1 - abs(rho_svi)) < 0) return(1e12)

    w_hat <- svi_w(k, a, b, rho_svi, m, nu)
    if (any(w_hat <= 0)) return(1e12)
    sum((w_hat - w_obs)^2)
  }

  # Initial parameter guess from data moments
  a0    <- mean(w_obs) * 0.5
  b0    <- 0.1
  rho0  <- -0.3
  m0    <- 0.0
  nu0   <- 0.1

  fit <- optim(
    par     = c(a0, b0, rho0, m0, nu0),
    fn      = objective,
    method  = "Nelder-Mead",
    control = list(maxit = 5000, reltol = 1e-10)
  )

  params <- fit$par
  list(
    a       = params[1],
    b       = params[2],
    rho_svi = params[3],
    m       = params[4],
    nu      = params[5],
    converged = fit$convergence == 0,
    value   = fit$value,
    message = fit$message
  )
}

#' Evaluate SVI surface on a grid of strikes and expiries
#' @param strikes  numeric vector of strike prices
#' @param F        scalar forward price
#' @param tau      scalar time to expiry
#' @param svi_params  list from fit_svi()
svi_surface_grid <- function(strikes, F, tau, svi_params) {
  k <- log(strikes / F)
  vol <- svi_vol(k, tau, svi_params$a, svi_params$b,
                 svi_params$rho_svi, svi_params$m, svi_params$nu)
  tibble(strike = strikes, log_moneyness = k, tau = tau, impl_vol = vol)
}

# ===========================================================================
# 4. No-arbitrage checks
# ===========================================================================

#' Butterfly spread check: for consecutive strikes K1 < K2 < K3,
#' butterfly price = C(K1) - 2*C(K2) + C(K3) >= 0.
#' Returns TRUE if all butterflies are non-negative (no arbitrage).
#' @param strikes  sorted numeric vector of strikes
#' @param prices   corresponding call prices
check_butterfly <- function(strikes, prices) {
  n <- length(strikes)
  if (n < 3) return(list(ok = TRUE, violations = integer(0)))

  butterfly <- prices[1:(n-2)] - 2 * prices[2:(n-1)] + prices[3:n]
  violations <- which(butterfly < -1e-8) + 1  # index of middle strike

  list(ok = length(violations) == 0, violations = violations,
       butterfly_spreads = butterfly)
}

#' Calendar spread check: for the same strike across two expiries T1 < T2,
#' C(T2) >= C(T1). Returns TRUE if no violations.
#' @param prices_t1  call prices for shorter expiry
#' @param prices_t2  call prices for longer expiry (same strikes)
check_calendar <- function(prices_t1, prices_t2) {
  stopifnot(length(prices_t1) == length(prices_t2))
  diff_vec <- prices_t2 - prices_t1
  violations <- which(diff_vec < -1e-8)
  list(ok = length(violations) == 0, violations = violations,
       spread = diff_vec)
}

#' Run a full no-arbitrage audit on an option chain
#' @param chain  data.frame with columns: strike, tau, call_price (sorted by strike within tau)
no_arb_audit <- function(chain) {
  # Butterfly check per expiry
  expiries <- unique(chain$tau)
  bf_results <- map(expiries, function(t) {
    sub <- chain %>% filter(tau == t) %>% arrange(strike)
    r   <- check_butterfly(sub$strike, sub$call_price)
    list(tau = t, ok = r$ok, violations = r$violations)
  })

  # Calendar check per strike (where multiple expiries exist)
  strikes <- unique(chain$strike)
  cal_results <- map(strikes, function(k) {
    sub <- chain %>% filter(strike == k) %>% arrange(tau)
    if (nrow(sub) < 2) return(list(strike = k, ok = TRUE, violations = integer(0)))
    r   <- check_calendar(sub$call_price[-nrow(sub)], sub$call_price[-1])
    list(strike = k, ok = r$ok, violations = r$violations)
  })

  bf_ok  <- all(map_lgl(bf_results, ~ .x$ok))
  cal_ok <- all(map_lgl(cal_results, ~ .x$ok))

  list(
    butterfly_ok       = bf_ok,
    calendar_ok        = cal_ok,
    no_arbitrage       = bf_ok && cal_ok,
    butterfly_results  = bf_results,
    calendar_results   = cal_results
  )
}

# ===========================================================================
# 5. Scenario P&L under vol surface perturbations
# ===========================================================================

#' Compute portfolio P&L under parallel vol shift, twist, and smile change
#' @param portfolio  data.frame: instrument, S, K, r, q, sigma, tau, type, quantity, multiplier
#' @param shift      parallel shift in vol (e.g. +0.01 = +1 vol point)
#' @param twist      slope change (multiplied by log-moneyness): d_sigma = twist * log(K/F)
#' @param smile_chg  curvature perturbation: d_sigma = smile_chg * log(K/F)^2
scenario_pnl <- function(portfolio, shift = 0, twist = 0, smile_chg = 0) {
  pf <- portfolio %>%
    mutate(
      F          = S * exp((r - q) * tau),
      k          = log(K / F),
      sigma_base = sigma,
      sigma_scen = sigma + shift + twist * k + smile_chg * k^2,
      sigma_scen = pmax(sigma_scen, 1e-4),
      price_base = bs_price(S, K, r, q, sigma_base, tau, type),
      price_scen = bs_price(S, K, r, q, sigma_scen, tau, type),
      pnl        = quantity * multiplier * (price_scen - price_base)
    )

  list(
    positions   = pf,
    total_pnl   = sum(pf$pnl),
    shift       = shift,
    twist       = twist,
    smile_chg   = smile_chg
  )
}

#' Run a suite of vol scenarios and return a summary data frame
#' @param portfolio  data.frame as per scenario_pnl()
#' @param scenarios  list of named lists with shift, twist, smile_chg elements
run_scenario_suite <- function(portfolio, scenarios) {
  map_dfr(scenarios, function(sc) {
    res <- scenario_pnl(portfolio, sc$shift, sc$twist, sc$smile_chg)
    tibble(
      scenario   = sc$name,
      shift      = sc$shift,
      twist      = sc$twist,
      smile_chg  = sc$smile_chg,
      total_pnl  = res$total_pnl
    )
  })
}

# ===========================================================================
# 6. Risk report generation with ggplot2
# ===========================================================================

#' Plot the delta profile across strikes for a given expiry
plot_delta_profile <- function(portfolio, expiry_tau = NULL) {
  pf <- if (!is.null(expiry_tau)) filter(portfolio, abs(tau - expiry_tau) < 1e-4)
        else portfolio

  pf_greeks <- compute_greeks(pf)

  ggplot(pf_greeks, aes(x = K, y = delta * quantity * multiplier, fill = type)) +
    geom_col(position = "stack") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = comma) +
    scale_fill_manual(values = c(call = "#2196F3", put = "#F44336")) +
    labs(
      title    = "Portfolio Delta Profile by Strike",
      subtitle = if (!is.null(expiry_tau)) sprintf("Expiry: %.4f years", expiry_tau) else "All Expiries",
      x        = "Strike",
      y        = "Dollar Delta (quantity * multiplier * delta)",
      fill     = "Type"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
}

#' Plot the vol surface as a heatmap of SVI-fitted implied vols
plot_vol_surface <- function(surface_df) {
  # surface_df: tibble with strike, tau, impl_vol
  ggplot(surface_df, aes(x = strike, y = tau, fill = impl_vol)) +
    geom_tile() +
    scale_fill_viridis_c(name = "Impl. Vol", labels = percent_format(accuracy = 0.1)) +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = function(x) sprintf("%.2f yr", x)) +
    labs(
      title = "Implied Volatility Surface (SVI Fit)",
      x     = "Strike",
      y     = "Time to Expiry (years)"
    ) +
    theme_minimal(base_size = 12)
}

#' Plot scenario P&L bar chart
plot_scenario_pnl <- function(scenario_summary) {
  # scenario_summary: data.frame with columns scenario, total_pnl
  scenario_summary %>%
    mutate(color = if_else(total_pnl >= 0, "gain", "loss")) %>%
    ggplot(aes(x = reorder(scenario, total_pnl), y = total_pnl, fill = color)) +
    geom_col() +
    geom_hline(yintercept = 0, linewidth = 0.5, color = "grey30") +
    coord_flip() +
    scale_y_continuous(labels = dollar_format(accuracy = 1)) +
    scale_fill_manual(values = c(gain = "#4CAF50", loss = "#F44336"), guide = "none") +
    labs(
      title = "Scenario P&L Summary",
      x     = "Scenario",
      y     = "P&L ($)"
    ) +
    theme_minimal(base_size = 12)
}

#' Plot gamma exposure by strike
plot_gamma_profile <- function(portfolio) {
  pf_greeks <- compute_greeks(portfolio)
  pf_greeks %>%
    mutate(dollar_gamma = gamma * quantity * multiplier * S^2 * 0.01^2) %>%
    ggplot(aes(x = K, y = dollar_gamma, fill = type)) +
    geom_col(position = "stack") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = dollar_format(accuracy = 1)) +
    scale_fill_manual(values = c(call = "#2196F3", put = "#F44336")) +
    labs(
      title    = "Dollar Gamma Profile (1% move in S)",
      x        = "Strike",
      y        = "Dollar Gamma ($)",
      fill     = "Type"
    ) +
    theme_minimal(base_size = 12)
}

#' Generate a full risk report PDF (or list of plots if no output path given)
#' @param portfolio  data.frame of positions
#' @param surface_df vol surface tibble
#' @param scenario_results  output from run_scenario_suite()
#' @param output_path  path for PDF output; if NULL returns a list of ggplots
generate_risk_report <- function(portfolio, surface_df = NULL,
                                  scenario_results = NULL,
                                  output_path = NULL) {
  plots <- list()
  plots$delta  <- plot_delta_profile(portfolio)
  plots$gamma  <- plot_gamma_profile(portfolio)
  if (!is.null(surface_df)) plots$vol_surface <- plot_vol_surface(surface_df)
  if (!is.null(scenario_results)) plots$scenarios <- plot_scenario_pnl(scenario_results)

  if (!is.null(output_path)) {
    pdf(output_path, width = 11, height = 8.5)
    walk(plots, print)
    dev.off()
    message("Risk report written to: ", output_path)
  }

  invisible(plots)
}
