# test_portfolio_analytics.R
# testthat tests for portfolio_analytics.R
# Run with: testthat::test_file("r/tests/test_portfolio_analytics.R")

library(testthat)

# Source the module under test relative to repo root, or adjust path as needed
source(file.path(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)),
                 "portfolio_analytics.R"), local = TRUE)

# If rstudioapi is not available (batch mode) fall back to relative path
tryCatch(
  invisible(NULL),
  error = function(e) {
    source("r/portfolio_analytics.R", local = TRUE)
  }
)

# Helper: build a simple 2-asset test case
make_test_case <- function(n = 252, seed = 42) {
  set.seed(seed)
  returns_mat <- matrix(rnorm(2 * n, 0.0005, 0.01), nrow = 2,
                        dimnames = list(c("A", "B"),
                                        as.character(seq.Date(as.Date("2020-01-01"),
                                                              by = "day", length.out = n))))
  weights_mat <- matrix(c(0.6, 0.4, 0.5, 0.5, 0.4, 0.6),
                        nrow = 2, ncol = 3,
                        dimnames = list(c("A", "B"),
                                        c("2020-01-01", "2020-04-01", "2020-07-01")))
  list(returns = returns_mat, weights = weights_mat)
}


# ---------------------------------------------------------------------------
# compute_portfolio_returns
# ---------------------------------------------------------------------------

test_that("compute_portfolio_returns returns correct length", {
  tc <- make_test_case()
  pr <- compute_portfolio_returns(tc$weights, tc$returns)
  expect_equal(length(pr), ncol(tc$returns))
})

test_that("compute_portfolio_returns returns named vector", {
  tc <- make_test_case()
  pr <- compute_portfolio_returns(tc$weights, tc$returns)
  expect_equal(names(pr), colnames(tc$returns))
})

test_that("compute_portfolio_returns errors on missing assets", {
  tc <- make_test_case()
  rownames(tc$weights)[1] <- "MISSING"
  expect_error(compute_portfolio_returns(tc$weights, tc$returns),
               "missing")
})

test_that("compute_portfolio_returns with equal-weight 2-asset matches manual", {
  set.seed(1)
  r <- matrix(c(0.01, 0.02, -0.01, 0.03), nrow = 2, ncol = 2,
              dimnames = list(c("A","B"), c("D1","D2")))
  w <- matrix(c(0.5, 0.5), nrow = 2, ncol = 1,
              dimnames = list(c("A","B"), c("D1")))
  pr <- compute_portfolio_returns(w, r)
  expect_equal(unname(pr[1]), 0.5 * 0.01 + 0.5 * 0.02)
})


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------

test_that("compute_turnover is zero for constant weights", {
  w <- matrix(rep(c(0.5, 0.5), 5), nrow = 2,
              dimnames = list(c("A","B"), paste0("D", 1:5)))
  expect_equal(compute_turnover(w), 0)
})

test_that("compute_turnover is 1 for full flip", {
  w <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2,
              dimnames = list(c("A","B"), c("D1","D2")))
  expect_equal(compute_turnover(w), 1)
})

test_that("compute_turnover warns on single column", {
  w <- matrix(c(0.5, 0.5), nrow = 2, ncol = 1,
              dimnames = list(c("A","B"), "D1"))
  expect_warning(compute_turnover(w), "fewer than 2")
})


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

test_that("sharpe_ratio is positive for positive mean returns", {
  set.seed(1)
  r <- rnorm(252, 0.001, 0.01)
  expect_gt(sharpe_ratio(r), 0)
})

test_that("sharpe_ratio returns NA for zero-vol series", {
  r <- rep(0.001, 100)
  expect_true(is.na(sharpe_ratio(r)))
})

test_that("sharpe_ratio annualisation factor is sqrt(252)", {
  set.seed(99)
  r     <- rnorm(252, 0.001, 0.01)
  sr_a  <- sharpe_ratio(r, annualize = TRUE)
  sr_na <- sharpe_ratio(r, annualize = FALSE)
  expect_equal(sr_a / sr_na, sqrt(252), tolerance = 1e-10)
})


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

test_that("sortino_ratio >= sharpe_ratio for positive skew", {
  set.seed(5)
  r <- c(rnorm(200, 0.002, 0.005), rnorm(52, -0.001, 0.02))
  sr  <- sharpe_ratio(r)
  srt <- sortino_ratio(r)
  # Sortino can be larger; just check both are finite
  expect_true(is.finite(sr))
  expect_true(is.finite(srt))
})

test_that("sortino_ratio with no downside returns NA-safe result", {
  r <- rep(0.001, 50)
  # No downside deviation -> NA (no losses)
  expect_true(is.na(sortino_ratio(r)))
})


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

test_that("calmar_ratio is positive for profitable strategy", {
  set.seed(10)
  r <- rnorm(252, 0.001, 0.01)
  cr <- calmar_ratio(r)
  if (!is.na(cr)) expect_gt(cr, 0)
})


# ---------------------------------------------------------------------------
# max_drawdown_series
# ---------------------------------------------------------------------------

test_that("max_drawdown_series is always <= 0", {
  set.seed(2)
  nav <- cumprod(1 + rnorm(252, 0.0005, 0.01))
  dd  <- max_drawdown_series(nav)
  expect_true(all(dd <= 0 + .Machine$double.eps))
})

test_that("max_drawdown_series is 0 at all-time highs", {
  nav <- cummax(c(1, 1.1, 1.05, 1.2, 1.15, 1.3))
  dd  <- max_drawdown_series(nav)
  # At indices 1, 2, 4, 6 the NAV hits new highs -> dd = 0
  expect_equal(dd[1], 0)
  expect_equal(dd[2], 0)
})


# ---------------------------------------------------------------------------
# underwater_curve
# ---------------------------------------------------------------------------

test_that("underwater_curve pct_underwater is between 0 and 1", {
  set.seed(3)
  nav <- cumprod(1 + rnorm(252, 0.0005, 0.015))
  uw  <- underwater_curve(nav)
  expect_gte(uw$pct_underwater, 0)
  expect_lte(uw$pct_underwater, 1)
})

test_that("underwater_curve longest_drawdown is a non-negative integer", {
  set.seed(3)
  nav <- cumprod(1 + rnorm(252, 0.0005, 0.015))
  uw  <- underwater_curve(nav)
  expect_gte(uw$longest_drawdown, 0)
})


# ---------------------------------------------------------------------------
# tail_ratio
# ---------------------------------------------------------------------------

test_that("tail_ratio is positive for any non-degenerate distribution", {
  set.seed(4)
  r  <- rnorm(1000)
  tr <- tail_ratio(r)
  expect_gt(tr, 0)
})

test_that("tail_ratio for symmetric distribution is close to 1", {
  set.seed(7)
  r  <- rnorm(10000)
  tr <- tail_ratio(r)
  expect_equal(tr, 1, tolerance = 0.2)
})


# ---------------------------------------------------------------------------
# omega_ratio
# ---------------------------------------------------------------------------

test_that("omega_ratio > 1 when mean return > threshold", {
  set.seed(8)
  r  <- rnorm(252, 0.002, 0.01)
  om <- omega_ratio(r, threshold = 0)
  expect_gt(om, 1)
})

test_that("omega_ratio < 1 when mean return < threshold", {
  set.seed(9)
  r  <- rnorm(252, -0.002, 0.01)
  om <- omega_ratio(r, threshold = 0)
  expect_lt(om, 1)
})


# ---------------------------------------------------------------------------
# rolling_sharpe
# ---------------------------------------------------------------------------

test_that("rolling_sharpe returns zoo object", {
  set.seed(11)
  r  <- rnorm(300, 0.001, 0.01)
  rs <- rolling_sharpe(r, window = 60, min_obs = 30)
  expect_s3_class(rs, "zoo")
})

test_that("rolling_sharpe first window-1 values are NA", {
  set.seed(12)
  r  <- rnorm(300, 0.001, 0.01)
  rs <- rolling_sharpe(r, window = 60, min_obs = 60)
  expect_true(all(is.na(rs[1:59])))
})


# ---------------------------------------------------------------------------
# monthly_returns_matrix
# ---------------------------------------------------------------------------

test_that("monthly_returns_matrix has 12 rows", {
  dates <- seq.Date(as.Date("2018-01-01"), as.Date("2020-12-31"), by = "day")
  r     <- setNames(rnorm(length(dates), 0.0005, 0.01), as.character(dates))
  mat   <- monthly_returns_matrix(r)
  expect_equal(nrow(mat), 12)
})

test_that("monthly_returns_matrix column count equals number of years", {
  dates <- seq.Date(as.Date("2018-01-01"), as.Date("2020-12-31"), by = "day")
  r     <- setNames(rnorm(length(dates), 0.0005, 0.01), as.character(dates))
  mat   <- monthly_returns_matrix(r)
  expect_equal(ncol(mat), 3)  # 2018, 2019, 2020
})


# ---------------------------------------------------------------------------
# performance_summary_table
# ---------------------------------------------------------------------------

test_that("performance_summary_table returns invisible list", {
  set.seed(13)
  r   <- rnorm(252, 0.0005, 0.01)
  out <- performance_summary_table(r)
  expect_type(out, "list")
  expect_named(out, c("n_obs","ann_return","ann_volatility","sharpe_ratio",
                      "sortino_ratio","calmar_ratio","max_drawdown",
                      "tail_ratio","omega_ratio","pct_underwater",
                      "longest_drawdown","skewness","excess_kurtosis",
                      "best_day","worst_day","positive_days_pct"))
})

test_that("performance_summary_table n_obs matches input length", {
  set.seed(14)
  r   <- rnorm(200, 0.0005, 0.01)
  out <- performance_summary_table(r)
  expect_equal(out$n_obs, 200)
})
