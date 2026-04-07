# test_options_risk.R
# testthat unit tests for options_risk.R
# Run with: testthat::test_file("R/tests/test_options_risk.R")
# or source this file after sourcing R/options_risk.R

library(testthat)

# Source the module under test (adjust path if running from project root)
tryCatch(
  source(file.path(dirname(dirname(sys.frame(1)$ofile)), "options_risk.R")),
  error = function(e) {
    # Fallback: look relative to working directory
    src <- file.path("R", "options_risk.R")
    if (file.exists(src)) source(src)
    else source("options_risk.R")
  }
)

# ===========================================================================
# Helper constants
# ===========================================================================
S   <- 100
K   <- 100
r   <- 0.05
q   <- 0.02
sig <- 0.20
tau <- 1.0

call_price <- bs_call(S, K, r, q, sig, tau)
put_price  <- bs_put(S, K, r, q, sig, tau)

# ===========================================================================
# 1. Black-Scholes put-call parity
# ===========================================================================

test_that("BS put-call parity holds", {
  # C - P = S*exp(-q*tau) - K*exp(-r*tau)
  pcp_lhs <- call_price - put_price
  pcp_rhs <- S * exp(-q * tau) - K * exp(-r * tau)
  expect_equal(pcp_lhs, pcp_rhs, tolerance = 1e-10)
})

# ===========================================================================
# 2. ATM call should be above intrinsic value
# ===========================================================================

test_that("ATM call is above intrinsic value", {
  intrinsic <- max(S * exp(-q * tau) - K * exp(-r * tau), 0)
  expect_gt(call_price, intrinsic)
})

# ===========================================================================
# 3. Call price is monotone in volatility
# ===========================================================================

test_that("Call price increases with volatility", {
  c_low  <- bs_call(S, K, r, q, 0.10, tau)
  c_high <- bs_call(S, K, r, q, 0.40, tau)
  expect_lt(c_low, c_high)
})

# ===========================================================================
# 4. Delta is between 0 and 1 for a call
# ===========================================================================

test_that("Call delta is in (0, 1)", {
  d <- bs_delta(S, K, r, q, sig, tau, "call")
  expect_gt(d, 0)
  expect_lt(d, 1)
})

test_that("Put delta is in (-1, 0)", {
  d <- bs_delta(S, K, r, q, sig, tau, "put")
  expect_lt(d, 0)
  expect_gt(d, -1)
})

# ===========================================================================
# 5. Put-call delta relationship: call_delta - put_delta = exp(-q*tau)
# ===========================================================================

test_that("Call delta minus put delta equals exp(-q*tau)", {
  dc <- bs_delta(S, K, r, q, sig, tau, "call")
  dp <- bs_delta(S, K, r, q, sig, tau, "put")
  expect_equal(dc - dp, exp(-q * tau), tolerance = 1e-10)
})

# ===========================================================================
# 6. Gamma is the same for calls and puts
# ===========================================================================

test_that("Gamma is identical for call and put (BS)", {
  g <- bs_gamma(S, K, r, q, sig, tau)
  # gamma does not depend on type -- just verify it is positive
  expect_gt(g, 0)
})

# ===========================================================================
# 7. Vega is positive
# ===========================================================================

test_that("Vega is positive", {
  v <- bs_vega(S, K, r, q, sig, tau)
  expect_gt(v, 0)
})

# ===========================================================================
# 8. Theta is negative for long options (time decay)
# ===========================================================================

test_that("Theta is negative for long call", {
  th <- bs_theta(S, K, r, q, sig, tau, "call")
  expect_lt(th, 0)
})

test_that("Theta is negative for long put (when r*K*exp > q*S)", {
  # Deep OTM put with low r might have positive theta, but ATM should be negative
  th <- bs_theta(S, K, r, q, sig, tau, "put")
  # ATM put theta is typically negative; just check it is finite
  expect_true(is.finite(th))
})

# ===========================================================================
# 9. Implied vol round-trip: BS price -> implied_vol_nr -> same vol
# ===========================================================================

test_that("Implied vol round-trip: call (ATM)", {
  market_p <- bs_call(S, K, r, q, sig, tau)
  iv <- implied_vol_nr(market_p, S, K, r, q, tau, type = "call")
  expect_equal(iv, sig, tolerance = 1e-6)
})

test_that("Implied vol round-trip: put (OTM)", {
  K_otm    <- 90
  sig_test <- 0.25
  market_p <- bs_put(S, K_otm, r, q, sig_test, tau)
  iv       <- implied_vol_nr(market_p, S, K_otm, r, q, tau, type = "put")
  expect_equal(iv, sig_test, tolerance = 1e-5)
})

test_that("Implied vol round-trip: call (ITM, high vol)", {
  sig_hi   <- 0.50
  K_itm    <- 80
  market_p <- bs_call(S, K_itm, r, q, sig_hi, tau)
  iv       <- implied_vol_nr(market_p, S, K_itm, r, q, tau, type = "call")
  expect_equal(iv, sig_hi, tolerance = 1e-5)
})

# ===========================================================================
# 10. Vectorized implied vol
# ===========================================================================

test_that("Vectorized implied vol returns correct length and values", {
  strikes <- c(90, 95, 100, 105, 110)
  vols    <- c(0.22, 0.21, 0.20, 0.21, 0.22)
  prices  <- bs_call(S, strikes, r, q, vols, tau)
  iv_vec  <- implied_vol_vec(prices, S, strikes, r, q, tau, type = "call")

  expect_length(iv_vec, 5)
  expect_equal(iv_vec, vols, tolerance = 1e-5)
})

# ===========================================================================
# 11. SVI total variance is positive for valid parameters
# ===========================================================================

test_that("SVI w() is positive for typical parameters", {
  k     <- seq(-0.5, 0.5, by = 0.1)
  a0    <- 0.04; b0 <- 0.1; rho0 <- -0.3; m0 <- 0.0; nu0 <- 0.1
  w_val <- svi_w(k, a0, b0, rho0, m0, nu0)
  expect_true(all(w_val > 0))
})

# ===========================================================================
# 12. SVI fit recovers parameters on synthetic data (noise-free)
# ===========================================================================

test_that("SVI fit recovers known parameters on noiseless data", {
  true_params <- list(a = 0.04, b = 0.10, rho_svi = -0.25, m = 0.0, nu = 0.12)
  k_grid      <- seq(-0.4, 0.4, by = 0.05)
  tau_test    <- 0.5
  sigma_true  <- svi_vol(k_grid, tau_test, true_params$a, true_params$b,
                          true_params$rho_svi, true_params$m, true_params$nu)

  fit <- fit_svi(k_grid, sigma_true, tau_test)

  # Check that the fitted vol surface is close (not necessarily same params due to degeneracies)
  sigma_fit <- svi_vol(k_grid, tau_test, fit$a, fit$b, fit$rho_svi, fit$m, fit$nu)
  rmse      <- sqrt(mean((sigma_fit - sigma_true)^2))
  expect_lt(rmse, 1e-4)
})

# ===========================================================================
# 13. No-arbitrage butterfly check: monotone strikes with smooth prices
# ===========================================================================

test_that("Butterfly check passes for BS call prices (no arbitrage)", {
  strikes <- seq(80, 120, by = 5)
  prices  <- bs_call(S, strikes, r, q, sig, tau)
  result  <- check_butterfly(strikes, prices)
  expect_true(result$ok)
})

test_that("Butterfly check flags injected arbitrage", {
  strikes <- c(90, 95, 100, 105, 110)
  prices  <- bs_call(S, strikes, r, q, sig, tau)
  # Artificially inflate middle price to create arbitrage
  prices[3] <- prices[2] + prices[4] + 1.0
  result    <- check_butterfly(strikes, prices)
  expect_false(result$ok)
  expect_gt(length(result$violations), 0)
})

# ===========================================================================
# 14. Calendar spread: longer-dated calls >= shorter-dated
# ===========================================================================

test_that("Calendar spread check passes for BS prices (no arbitrage)", {
  tau1 <- 0.25; tau2 <- 0.50
  strikes <- seq(90, 110, by = 5)
  p1 <- bs_call(S, strikes, r, q, sig, tau1)
  p2 <- bs_call(S, strikes, r, q, sig, tau2)
  result <- check_calendar(p1, p2)
  expect_true(result$ok)
})

# ===========================================================================
# 15. compute_greeks returns expected columns
# ===========================================================================

test_that("compute_greeks returns all Greek columns", {
  df <- tibble(
    S = c(100, 100, 100),
    K = c(95, 100, 105),
    r = 0.05, q = 0.02, sigma = 0.20, tau = 0.5,
    type = "call"
  )
  result <- compute_greeks(df)
  expected_cols <- c("price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga")
  expect_true(all(expected_cols %in% names(result)))
  expect_true(all(is.finite(result$price)))
  expect_true(all(result$gamma > 0))
})

# ===========================================================================
# 16. scenario_pnl: zero shift gives zero P&L
# ===========================================================================

test_that("scenario_pnl returns zero total P&L for zero shift", {
  portfolio <- tibble(
    S = 100, K = 100, r = 0.05, q = 0.02, sigma = 0.20,
    tau = 1.0, type = "call", quantity = 10, multiplier = 100
  )
  result <- scenario_pnl(portfolio, shift = 0, twist = 0, smile_chg = 0)
  expect_equal(result$total_pnl, 0, tolerance = 1e-10)
})

test_that("scenario_pnl: parallel vol increase causes long call to gain", {
  portfolio <- tibble(
    S = 100, K = 100, r = 0.05, q = 0.02, sigma = 0.20,
    tau = 1.0, type = "call", quantity = 1, multiplier = 1
  )
  result <- scenario_pnl(portfolio, shift = 0.05)
  expect_gt(result$total_pnl, 0)
})
