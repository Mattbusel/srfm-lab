# test_signal_research.R
# testthat tests for signal_research.R
# Run with: testthat::test_file("r/tests/test_signal_research.R")

library(testthat)

source(file.path(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)),
                 "signal_research.R"), local = TRUE)

tryCatch(
  invisible(NULL),
  error = function(e) source("r/signal_research.R", local = TRUE)
)


# Helper: generate a simple signal/returns pair with known IC sign
make_signal_returns <- function(n_assets = 30, n_periods = 100, seed = 42) {
  set.seed(seed)
  # Signal predicts returns with positive rank correlation
  factor_scores <- matrix(rnorm(n_periods), nrow = 1)
  signal  <- matrix(NA_real_, nrow = n_assets, ncol = n_periods)
  returns <- matrix(NA_real_, nrow = n_assets, ncol = n_periods)
  asset_loadings <- rnorm(n_assets, 0.5, 0.2)
  for (t in seq_len(n_periods)) {
    signal[, t]  <- asset_loadings * factor_scores[1, t] + rnorm(n_assets, 0, 0.1)
    returns[, t] <- asset_loadings * factor_scores[1, t] + rnorm(n_assets, 0, 1.0)
  }
  rownames(signal)  <- paste0("A", seq_len(n_assets))
  rownames(returns) <- paste0("A", seq_len(n_assets))
  colnames(signal)  <- paste0("T", seq_len(n_periods))
  colnames(returns) <- paste0("T", seq_len(n_periods))
  list(signal = signal, returns = returns)
}


# ---------------------------------------------------------------------------
# compute_ic
# ---------------------------------------------------------------------------

test_that("compute_ic returns vector of correct length", {
  tc <- make_signal_returns()
  ic <- compute_ic(tc$signal, tc$returns)
  expect_equal(length(ic), ncol(tc$signal))
})

test_that("compute_ic values are in [-1, 1]", {
  tc <- make_signal_returns()
  ic <- compute_ic(tc$signal, tc$returns)
  expect_true(all(ic >= -1 - 1e-10 | is.na(ic)))
  expect_true(all(ic <=  1 + 1e-10 | is.na(ic)))
})

test_that("compute_ic mean IC is positive for predictive signal", {
  tc <- make_signal_returns(n_assets = 50, n_periods = 200)
  ic <- compute_ic(tc$signal, tc$returns)
  expect_gt(mean(ic, na.rm = TRUE), 0)
})

test_that("compute_ic errors on dimension mismatch", {
  tc <- make_signal_returns()
  bad_returns <- tc$returns[, -1]
  expect_error(compute_ic(tc$signal, bad_returns), "identical dimensions")
})

test_that("compute_ic pearson method runs without error", {
  tc <- make_signal_returns()
  expect_no_error(compute_ic(tc$signal, tc$returns, method = "pearson"))
})


# ---------------------------------------------------------------------------
# compute_icir
# ---------------------------------------------------------------------------

test_that("compute_icir returns scalar for full-sample mode", {
  tc <- make_signal_returns()
  ic <- compute_ic(tc$signal, tc$returns)
  icir <- compute_icir(ic)
  expect_length(icir, 1)
  expect_true(is.finite(icir))
})

test_that("compute_icir rolling mode returns zoo object", {
  tc <- make_signal_returns(n_periods = 150)
  ic <- compute_ic(tc$signal, tc$returns)
  rolling_icir <- compute_icir(ic, window = 40)
  expect_s3_class(rolling_icir, "zoo")
})

test_that("compute_icir is positive when mean IC > 0", {
  ic <- c(0.05, 0.04, 0.06, 0.03, 0.07, 0.05)
  expect_gt(compute_icir(ic), 0)
})


# ---------------------------------------------------------------------------
# ic_decay_curve
# ---------------------------------------------------------------------------

test_that("ic_decay_curve returns named vector of length max_lag", {
  tc    <- make_signal_returns(n_periods = 80)
  decay <- ic_decay_curve(tc$signal, tc$returns, max_lag = 10)
  expect_length(decay, 10)
  expect_equal(names(decay), paste0("h", 1:10))
})

test_that("ic_decay_curve values decay (h1 > h10 for predictive signal)", {
  tc    <- make_signal_returns(n_assets = 50, n_periods = 200)
  decay <- ic_decay_curve(tc$signal, tc$returns, max_lag = 10)
  # IC should generally be larger at short horizons
  expect_gte(abs(decay[1]), abs(decay[10]) - 0.05)
})


# ---------------------------------------------------------------------------
# fit_ic_decay
# ---------------------------------------------------------------------------

test_that("fit_ic_decay returns list with required components", {
  tc    <- make_signal_returns(n_assets = 50, n_periods = 200)
  decay <- ic_decay_curve(tc$signal, tc$returns, max_lag = 10)
  fit   <- fit_ic_decay(pmax(decay, 1e-6))
  expect_named(fit, c("ic0","lambda","half_life","r_squared","fitted"),
               ignore.order = TRUE)
})

test_that("fit_ic_decay half_life is positive when lambda > 0", {
  decay_vals <- 0.1 * exp(-0.2 * 1:15) + rnorm(15, 0, 0.002)
  decay_vals <- pmax(decay_vals, 1e-6)
  fit <- fit_ic_decay(decay_vals)
  if (!is.na(fit$half_life)) expect_gt(fit$half_life, 0)
})


# ---------------------------------------------------------------------------
# quintile_sort_returns
# ---------------------------------------------------------------------------

test_that("quintile_sort_returns returns list of n_quintiles vectors", {
  tc <- make_signal_returns()
  qs <- quintile_sort_returns(tc$signal, tc$returns, n_quintiles = 5)
  expect_length(qs, 5)
  expect_named(qs, paste0("Q", 1:5))
})

test_that("quintile_sort_returns each bucket has correct length", {
  tc <- make_signal_returns()
  qs <- quintile_sort_returns(tc$signal, tc$returns)
  expect_equal(length(qs$Q1), ncol(tc$signal))
})


# ---------------------------------------------------------------------------
# long_short_returns
# ---------------------------------------------------------------------------

test_that("long_short_returns returns vector of correct length", {
  tc    <- make_signal_returns(n_assets = 60)
  ls    <- long_short_returns(tc$signal, tc$returns, n_long = 10, n_short = 10)
  expect_equal(length(ls), ncol(tc$signal))
})

test_that("long_short_returns mean is positive for predictive signal", {
  tc <- make_signal_returns(n_assets = 60, n_periods = 200)
  ls <- long_short_returns(tc$signal, tc$returns, n_long = 10, n_short = 10)
  expect_gt(mean(ls, na.rm = TRUE), 0)
})


# ---------------------------------------------------------------------------
# signal_correlation_matrix
# ---------------------------------------------------------------------------

test_that("signal_correlation_matrix returns square symmetric matrix", {
  tc  <- make_signal_returns()
  cm  <- signal_correlation_matrix(list(s1 = tc$signal, s2 = tc$signal + rnorm(length(tc$signal), 0, 0.5)))
  expect_equal(nrow(cm), 2)
  expect_equal(ncol(cm), 2)
  expect_equal(cm[1, 2], cm[2, 1], tolerance = 1e-10)
})

test_that("signal_correlation_matrix diagonal is 1", {
  tc  <- make_signal_returns()
  cm  <- signal_correlation_matrix(list(s1 = tc$signal, s2 = tc$signal * 2))
  expect_equal(diag(cm), c(1, 1))
})


# ---------------------------------------------------------------------------
# novelty_filter
# ---------------------------------------------------------------------------

test_that("novelty_filter identifies duplicate signal as non-novel", {
  tc  <- make_signal_returns()
  nf  <- novelty_filter(tc$signal, list(existing = tc$signal), threshold = 0.7)
  expect_false(nf$is_novel)
})

test_that("novelty_filter identifies independent signal as novel", {
  set.seed(99)
  tc  <- make_signal_returns()
  independent_signal <- matrix(rnorm(length(tc$signal)), nrow = nrow(tc$signal))
  rownames(independent_signal) <- rownames(tc$signal)
  nf  <- novelty_filter(independent_signal, list(existing = tc$signal), threshold = 0.7)
  expect_true(nf$is_novel)
})


# ---------------------------------------------------------------------------
# generate_ar1_factor
# ---------------------------------------------------------------------------

test_that("generate_ar1_factor returns matrix of correct dimensions", {
  sig <- generate_ar1_factor(100, 30, rho = 0.6)
  expect_equal(dim(sig), c(30, 100))
})

test_that("generate_ar1_factor columns have AR(1) autocorrelation", {
  set.seed(1)
  sig <- generate_ar1_factor(500, 5, rho = 0.7, noise_sd = 0.01)
  # The common factor drives all rows; check that column 1 (across time) shows autocorrelation
  # by checking any single asset row
  x   <- as.numeric(sig[1, ])
  ac1 <- stats::cor(x[-length(x)], x[-1])
  expect_gt(ac1, 0.3)
})


# ---------------------------------------------------------------------------
# generate_momentum_factor
# ---------------------------------------------------------------------------

test_that("generate_momentum_factor returns matrix with correct dimensions", {
  set.seed(2)
  n_assets  <- 20
  n_periods <- 150
  returns   <- matrix(rnorm(n_assets * n_periods, 0, 0.01),
                      nrow = n_assets, ncol = n_periods,
                      dimnames = list(paste0("A", seq_len(n_assets)),
                                      paste0("T", seq_len(n_periods))))
  mom <- generate_momentum_factor(returns, lookback = 63)
  expect_equal(dim(mom), c(n_assets, n_periods))
})

test_that("generate_momentum_factor first lookback columns are NA", {
  set.seed(3)
  n_assets <- 10
  n_periods <- 100
  returns <- matrix(rnorm(n_assets * n_periods, 0, 0.01),
                    nrow = n_assets, ncol = n_periods,
                    dimnames = list(paste0("A", seq_len(n_assets)),
                                    paste0("T", seq_len(n_periods))))
  mom <- generate_momentum_factor(returns, lookback = 63)
  expect_true(all(is.na(mom[, 1:63])))
})
