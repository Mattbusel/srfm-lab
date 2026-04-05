# =============================================================================
# high_frequency_stats.R
# High-frequency statistics for crypto trading
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Intraday crypto data trades 24/7 at high frequency.
# Realized variance estimators convert intraday price data into daily vol
# measures that are model-free and far more accurate than using daily returns
# alone. Detecting jumps separates "diffusive" moves from "jump" risk.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. REALIZED VARIANCE (RV)
# -----------------------------------------------------------------------------

#' Compute realized variance from intraday log-returns
#' RV_t = sum_{j=1}^{M} r_{t,j}^2 where r_{t,j} = log(P_{t,j}/P_{t,j-1})
#' @param log_prices intraday log-price vector for a single day
#' @return scalar realized variance
realized_variance <- function(log_prices) {
  r <- diff(log_prices)
  sum(r^2)
}

#' Compute RV for multiple days
#' @param prices_list list of intraday log-price vectors (one per day)
#' @return numeric vector of daily RVs
daily_realized_variance <- function(prices_list) {
  sapply(prices_list, realized_variance)
}

#' Annualized realized volatility
realized_vol_annualized <- function(log_prices, periods_per_year = 365) {
  rv <- realized_variance(log_prices)
  sqrt(rv * periods_per_year)  # assuming rv is per-day
}

# -----------------------------------------------------------------------------
# 2. BIPOWER VARIATION (BV) -- robust to jumps
# -----------------------------------------------------------------------------

#' Barndorff-Nielsen & Shephard (2004) bipower variation
#' BV_t = mu_1^{-2} * sum_{j=2}^{M} |r_{t,j}| * |r_{t,j-1}|
#' Under no jumps, BV_t -> IV_t (integrated variance) in probability
#' Key: product of adjacent absolute returns; jumps don't appear in adjacent pairs
#' @param log_prices intraday log-price vector for one day
bipower_variation <- function(log_prices) {
  r <- diff(log_prices)
  mu1 <- sqrt(2 / pi)  # E[|Z|] for Z ~ N(0,1)
  if (length(r) < 2) return(NA)
  bv <- mu1^(-2) * sum(abs(r[2:length(r)]) * abs(r[1:(length(r)-1)]))
  bv
}

#' Tripower variation (used in jump tests)
#' @param log_prices intraday log-price vector
tripower_quarticity <- function(log_prices) {
  r  <- diff(log_prices)
  M  <- length(r)
  if (M < 3) return(NA)
  mu_4_3 <- 2^(2/3) * gamma(7/6) / gamma(1/2)  # E[|Z|^{4/3}]
  tq <- M * mu_4_3^(-3) * sum(
    abs(r[3:M])^(4/3) * abs(r[2:(M-1)])^(4/3) * abs(r[1:(M-2)])^(4/3)
  )
  tq
}

# -----------------------------------------------------------------------------
# 3. REALIZED KERNEL -- noise-robust estimator
# -----------------------------------------------------------------------------

#' Realized kernel of Barndorff-Nielsen et al. (2008)
#' More robust to market microstructure noise than RV
#' K(x) = Parzen kernel; bandwidth H chosen by bandwidth selection rule
#' @param log_prices intraday log-price vector
#' @param H bandwidth (number of autocovariance lags)
realized_kernel <- function(log_prices, H = NULL) {
  r <- diff(log_prices)
  M <- length(r)

  # Default bandwidth: H ~ c_star * n^(3/5) (from optimal MSE)
  if (is.null(H)) H <- max(1, round(4 * (M/100)^(3/5)))

  # Parzen kernel function
  parzen_k <- function(x) {
    ax <- abs(x)
    ifelse(ax <= 0.5,
           1 - 6*ax^2 + 6*ax^3,
           ifelse(ax <= 1, 2*(1-ax)^3, 0))
  }

  # Autocovariance of returns
  gamma_hat <- function(h) {
    if (h >= M) return(0)
    sum(r[(h+1):M] * r[1:(M-h)])
  }

  rk <- gamma_hat(0)
  for (h in seq_len(H)) {
    k_h <- parzen_k(h / (H + 1))
    rk  <- rk + 2 * k_h * gamma_hat(h)
  }
  rk
}

# -----------------------------------------------------------------------------
# 4. JUMP TESTS
# -----------------------------------------------------------------------------

#' Lee-Mykland (2008) jump test -- identifies individual jumps
#' Test statistic: L(t) = r_{t,j} / sqrt(BV_local)
#' Large |L(t)| indicates a jump at time t
#' @param log_prices intraday log-price vector
#' @param window local window for variance estimation (default 252 obs)
#' @param alpha significance level
lee_mykland_jump_test <- function(log_prices, window = 252, alpha = 0.01) {
  r <- diff(log_prices)
  M <- length(r)
  if (M < window + 1) stop("Insufficient data for Lee-Mykland test")

  # Local bipower variation for each observation
  local_bv <- numeric(M)
  for (j in (window+1):M) {
    r_w <- r[(j - window):(j-1)]
    mu1 <- sqrt(2/pi)
    local_bv[j] <- mu1^(-2) *
      sum(abs(r_w[2:length(r_w)]) * abs(r_w[1:(length(r_w)-1)])) / window
  }

  # Standardize returns
  L_stat <- ifelse(local_bv > 0, r / sqrt(local_bv), 0)

  # Critical value from Gumbel distribution of max|N(0,1)|
  # P(max|L_j| <= c_alpha) ≈ exp(-2*exp(-beta_n*(c_alpha - a_n)))
  a_n <- (2 * log(M))^(1/2) - (log(pi) + log(log(M))) / (2 * (2*log(M))^(1/2))
  beta_n <- 1 / (2*log(M))^(1/2)
  c_alpha <- a_n - beta_n * log(-log(1 - alpha))

  jumps <- which(abs(L_stat) > c_alpha)
  jump_sizes <- r[jumps]

  cat("=== Lee-Mykland Jump Test ===\n")
  cat(sprintf("Critical value (alpha=%.3f): %.4f\n", alpha, c_alpha))
  cat(sprintf("Number of jumps detected: %d (%.2f%% of obs)\n",
              length(jumps), 100*length(jumps)/M))
  if (length(jumps) > 0) {
    cat(sprintf("Largest jump: %.5f at index %d\n",
                jump_sizes[which.max(abs(jump_sizes))],
                jumps[which.max(abs(jump_sizes))]))
  }

  list(L_stat = L_stat, jumps = jumps, jump_sizes = jump_sizes,
       critical_val = c_alpha, n_jumps = length(jumps))
}

#' Barndorff-Nielsen-Shephard jump test (daily)
#' Null: no jumps, so RV/BV -> 1 as M -> infinity
#' Test statistic: (RV - BV) / sqrt(variance) ~ N(0,1) under H0
#' @param log_prices intraday log-price vector for one day
bns_jump_test <- function(log_prices) {
  M  <- length(log_prices) - 1  # number of returns
  rv <- realized_variance(log_prices)
  bv <- bipower_variation(log_prices)
  tq <- tripower_quarticity(log_prices)

  if (is.na(bv) || is.na(tq) || rv == 0) {
    return(list(stat = NA, p_value = NA, jump_component = 0))
  }

  # Ratio statistic
  mu_1 <- sqrt(2/pi)
  mu_4_3 <- 2^(2/3) * gamma(7/6) / gamma(1/2)

  # Variance of (RV - BV) under H0
  phi <- pi^2/4 + pi - 5
  variance_ratio <- (phi / M) * (mu_1^(-4)) * tq / bv^2

  if (variance_ratio <= 0) return(list(stat=NA, p_value=NA, jump_component=0))
  z_stat <- (rv - bv) / (sqrt(variance_ratio * rv^2))

  p_val  <- 1 - pnorm(z_stat)  # one-sided (jumps increase RV above BV)
  jump_component <- pmax(rv - bv, 0)

  cat(sprintf("BNS Jump Test: Z=%.3f, p=%.4f, jump_var=%.6f\n",
              z_stat, p_val, jump_component))

  list(stat = z_stat, p_value = p_val, RV = rv, BV = bv,
       jump_component = jump_component,
       jump_fraction = jump_component / rv)
}

# -----------------------------------------------------------------------------
# 5. MICROSTRUCTURE NOISE ESTIMATION
# -----------------------------------------------------------------------------

#' Estimate microstructure noise variance
#' @param log_prices vector of high-frequency log-prices
#' @return noise variance estimate (2*a^2 where a = noise std)
estimate_noise_variance <- function(log_prices) {
  r <- diff(log_prices)
  n <- length(r)
  # Zhang, Mykland & Ait-Sahalia (2005): noise = -Cov(r_t, r_{t-1}) / 2
  lag1_cov <- sum(r[2:n] * r[1:(n-1)]) / n
  noise_var <- max(-lag1_cov, 0)
  cat(sprintf("Microstructure noise variance: %.8f (std: %.6f)\n",
              noise_var, sqrt(noise_var)))
  list(noise_variance = noise_var, noise_std = sqrt(noise_var),
       first_order_autocov = lag1_cov)
}

#' Noise-robust estimator: Two-Scale Realized Variance (TSRV)
#' TSRV = RV_{slow} - (M_slow/M_fast)*RV_{fast}
#' The fast scale captures noise; slow scale captures signal + noise;
#' taking the difference cancels out the noise
tsrv <- function(log_prices, K = 5) {
  n <- length(log_prices)
  r_all <- diff(log_prices)
  M_all <- length(r_all)

  # All-returns RV (fast scale)
  rv_fast <- sum(r_all^2)

  # Subsampled RV at scale K (slow scale)
  rv_subsample <- 0
  n_sub <- floor(n / K)
  for (k_start in seq_len(K)) {
    idx <- seq(k_start, n, by = K)
    r_sub <- diff(log_prices[idx])
    rv_subsample <- rv_subsample + sum(r_sub^2)
  }
  rv_slow <- rv_subsample / K

  # M for each
  M_slow <- n_sub - 1

  # TSRV
  tsrv_est <- rv_slow - (M_slow / M_all) * rv_fast
  cat(sprintf("TSRV estimate: %.8f, RV all-ticks: %.8f\n", tsrv_est, rv_fast))

  list(tsrv = tsrv_est, rv_alltick = rv_fast, rv_subsample = rv_slow, K = K)
}

# -----------------------------------------------------------------------------
# 6. OPTIMAL SAMPLING FREQUENCY (SIGNATURE PLOT)
# -----------------------------------------------------------------------------

#' Signature plot: plot RV as a function of sampling frequency
#' Under pure Brownian motion, RV should be flat across frequencies
#' Under microstructure noise, RV increases as frequency increases
#' Optimal sampling = the frequency where the "flat" region begins
#' @param log_prices high-frequency log-price vector
#' @param freqs_to_test vector of subsampling frequencies to test
signature_plot <- function(log_prices, freqs_to_test = c(1, 2, 5, 10, 15, 30, 60)) {
  rv_by_freq <- sapply(freqs_to_test, function(K) {
    idx <- seq(1, length(log_prices), by = K)
    if (length(idx) < 3) return(NA)
    r <- diff(log_prices[idx])
    # Annualize appropriately
    sum(r^2) * K  # scale back to per-tick equivalent
  })

  df <- data.frame(
    freq = freqs_to_test,
    RV   = rv_by_freq
  )
  cat("=== Signature Plot (RV vs Sampling Frequency) ===\n")
  print(df)

  # Optimal sampling: pick frequency where slope of RV vs freq first flattens
  diffs <- diff(rv_by_freq)
  opt_idx <- which(abs(diffs) == min(abs(diffs), na.rm=TRUE))[1]
  opt_freq <- freqs_to_test[opt_idx]
  cat(sprintf("Suggested optimal sampling frequency: every %d ticks\n", opt_freq))

  invisible(list(signature = df, optimal_freq = opt_freq))
}

# -----------------------------------------------------------------------------
# 7. INTRADAY SEASONALITY ADJUSTMENT (FFF REGRESSION)
# -----------------------------------------------------------------------------

#' Flexible Fourier Form (FFF) intraday seasonality estimation
#' Returns contain strong intraday periodicity (e.g., high vol at open/close)
#' FFF decomposes the pattern into Fourier terms
#' @param r2_intraday matrix of squared returns: rows=days, cols=intraday periods
#' @param n_harmonics number of Fourier harmonics to use
fff_seasonality <- function(r2_intraday, n_harmonics = 3) {
  n_days <- nrow(r2_intraday)
  M      <- ncol(r2_intraday)
  tau    <- seq(0, 1, length.out = M + 1)[1:M]  # normalized time of day

  # Build Fourier basis
  X_fourier <- matrix(1, M, 1 + 2 * n_harmonics)
  for (k in seq_len(n_harmonics)) {
    X_fourier[, 2*k]   <- cos(2 * pi * k * tau)
    X_fourier[, 2*k+1] <- sin(2 * pi * k * tau)
  }

  # Pool across all days: regress r^2_{i,j} on time-of-day dummies
  y_pool <- as.vector(t(r2_intraday))  # length n_days * M
  X_pool <- do.call(rbind, replicate(n_days, X_fourier, simplify=FALSE))

  # OLS
  beta <- solve(t(X_pool) %*% X_pool) %*% t(X_pool) %*% y_pool
  seasonal_component <- X_fourier %*% beta

  # Scale: seasonality should integrate to 1 over the day
  seasonal_factor <- seasonal_component / mean(seasonal_component)

  # Seasonality-adjusted returns: r_adj = r / sqrt(seasonal_factor)
  adj_factors <- matrix(rep(sqrt(pmax(seasonal_factor, 0.01)), n_days),
                        nrow = n_days, byrow = TRUE)

  cat("=== FFF Intraday Seasonality ===\n")
  cat(sprintf("Peak seasonality factor: %.2f (at period %d of %d)\n",
              max(seasonal_factor), which.max(seasonal_factor), M))
  cat(sprintf("Min  seasonality factor: %.2f (at period %d of %d)\n",
              min(seasonal_factor), which.min(seasonal_factor), M))

  list(seasonal_factor = seasonal_factor, beta = beta,
       adj_factors = adj_factors,
       r2_adj = r2_intraday / adj_factors^2)
}

# -----------------------------------------------------------------------------
# 8. REALIZED COVARIANCE AND CORRELATION MATRICES
# -----------------------------------------------------------------------------

#' Compute realized covariance matrix from synchronous intraday returns
#' @param returns_matrix matrix of synchronous intraday returns: rows=time, cols=assets
realized_covariance <- function(returns_matrix) {
  # Simple sum of outer products: RCov = sum_t r_t r_t'
  M <- nrow(returns_matrix)
  d <- ncol(returns_matrix)
  rcov <- matrix(0, d, d)
  for (t in seq_len(M)) {
    r <- returns_matrix[t, ]
    rcov <- rcov + outer(r, r)
  }
  rcov
}

#' Realized correlation matrix
realized_correlation <- function(returns_matrix) {
  rcov <- realized_covariance(returns_matrix)
  diag_sqrt <- sqrt(diag(rcov))
  rcov / outer(diag_sqrt, diag_sqrt)
}

#' Daily realized covariance series across multiple days
#' @param returns_list list of matrices (one per day), each (M x d)
daily_realized_covariance <- function(returns_list) {
  lapply(returns_list, realized_covariance)
}

# -----------------------------------------------------------------------------
# 9. HAYASHI-YOSHIDA ESTIMATOR (non-synchronous trading)
# -----------------------------------------------------------------------------

#' Hayashi-Yoshida (2005) cross-covariance estimator
#' Handles the case where two assets do NOT trade at the same times
#' Standard RV requires synchronization; HY does not
#' HY = sum over all pairs (i,j) where [s_i, t_i] and [s_j, t_j] overlap:
#'      r_X(i) * r_Y(j)
#' @param times_x,times_y transaction time vectors (POSIX or numeric)
#' @param prices_x,prices_y corresponding price vectors
hayashi_yoshida <- function(times_x, prices_x, times_y, prices_y) {
  # Compute log-returns and associated time intervals
  n_x <- length(times_x); n_y <- length(times_y)
  if (n_x < 2 || n_y < 2) return(NA)

  r_x <- diff(log(prices_x))
  r_y <- diff(log(prices_y))
  # Intervals: [times_x[i], times_x[i+1]], [times_y[j], times_y[j+1]]
  s_x <- times_x[1:(n_x-1)]; t_x <- times_x[2:n_x]
  s_y <- times_y[1:(n_y-1)]; t_y <- times_y[2:n_y]

  hy_cov <- 0
  for (i in seq_along(r_x)) {
    for (j in seq_along(r_y)) {
      # Check if intervals overlap: overlap iff s_i < t_j and s_j < t_i
      if (s_x[i] < t_y[j] && s_y[j] < t_x[i]) {
        hy_cov <- hy_cov + r_x[i] * r_y[j]
      }
    }
  }
  hy_cov
}

#' Vectorized (faster) Hayashi-Yoshida
hayashi_yoshida_fast <- function(times_x, prices_x, times_y, prices_y) {
  r_x <- diff(log(prices_x)); r_y <- diff(log(prices_y))
  s_x <- times_x[-length(times_x)]; t_x <- times_x[-1]
  s_y <- times_y[-length(times_y)]; t_y <- times_y[-1]
  nx  <- length(r_x); ny <- length(r_y)

  # Vectorized overlap check
  s_x_mat <- matrix(s_x, nx, ny)
  t_x_mat <- matrix(t_x, nx, ny)
  s_y_mat <- matrix(s_y, nx, ny, byrow=TRUE)
  t_y_mat <- matrix(t_y, nx, ny, byrow=TRUE)

  overlap <- (s_x_mat < t_y_mat) & (s_y_mat < t_x_mat)
  hy_cov  <- sum(outer(r_x, r_y) * overlap)
  hy_cov
}

# -----------------------------------------------------------------------------
# 10. INTRADAY SUMMARY STATISTICS
# -----------------------------------------------------------------------------

#' Summarize intraday trading statistics for a single day
#' @param log_prices intraday log-price vector
#' @param times numeric time index or POSIX times
intraday_summary <- function(log_prices, times = NULL) {
  r <- diff(log_prices)
  M <- length(r)
  rv <- sum(r^2)
  bv <- bipower_variation(log_prices)

  # Separate continuous and jump components
  jump_comp <- max(rv - bv, 0)
  cont_comp <- bv

  # Max absolute return (proxy for largest move)
  max_move <- max(abs(r))
  max_move_idx <- which.max(abs(r))

  # Number of sign changes (proxy for noise)
  sign_changes <- sum(diff(sign(r)) != 0) / M

  # Realized skewness: (M^{1/2} / RV^{3/2}) * sum(r^3)
  r_skew <- (sqrt(M) / rv^(3/2)) * sum(r^3)

  # Realized kurtosis: (M / RV^2) * sum(r^4)
  r_kurt <- (M / rv^2) * sum(r^4)

  cat("=== Intraday Summary ===\n")
  cat(sprintf("Observations (M): %d\n", M))
  cat(sprintf("Realized Variance: %.8f  (vol=%.4f)\n", rv, sqrt(rv)))
  cat(sprintf("Bipower Variation: %.8f\n", bv))
  cat(sprintf("Jump component:    %.8f (%.1f%% of RV)\n",
              jump_comp, 100*jump_comp/rv))
  cat(sprintf("Max |return|:      %.5f at obs %d\n", max_move, max_move_idx))
  cat(sprintf("Realized skewness: %.3f\n", r_skew))
  cat(sprintf("Realized kurtosis: %.3f (excess: %.3f)\n", r_kurt, r_kurt-3))
  cat(sprintf("Sign change rate:  %.2f%%\n", 100*sign_changes))

  list(M=M, rv=rv, bv=bv, jump_component=jump_comp,
       cont_component=cont_comp, max_abs_return=max_move,
       r_skew=r_skew, r_kurt=r_kurt, sign_change_rate=sign_changes)
}

# -----------------------------------------------------------------------------
# 11. FULL HF STATISTICS PIPELINE
# -----------------------------------------------------------------------------

#' Full high-frequency analysis for a day's intraday data
#' @param log_prices vector of intraday log-prices for one day
#' @param asset_name asset label
run_hf_analysis <- function(log_prices, asset_name = "Asset") {
  cat("=============================================================\n")
  cat(sprintf("HIGH-FREQUENCY ANALYSIS: %s\n", asset_name))
  cat(sprintf("Observations: %d\n\n", length(log_prices)))

  # Basic measures
  summary_stats <- intraday_summary(log_prices)

  # Realized kernel
  rk <- realized_kernel(log_prices)
  cat(sprintf("\nRealized Kernel: %.8f\n", rk))

  # TSRV
  cat("\n")
  tsrv_res <- tsrv(log_prices, K=5)

  # Noise variance
  cat("\n")
  noise_est <- estimate_noise_variance(log_prices)

  # BNS Jump test
  cat("\n")
  bns_res <- bns_jump_test(log_prices)

  # Lee-Mykland jump test (requires more data)
  lm_res <- NULL
  if (length(log_prices) > 300) {
    cat("\n")
    lm_res <- lee_mykland_jump_test(log_prices, window = 200, alpha = 0.05)
  }

  # Signature plot
  cat("\n")
  sig_plot <- signature_plot(log_prices)

  cat("\n=== Estimation Quality Summary ===\n")
  cat(sprintf("RV (all ticks):     %.8f\n", summary_stats$rv))
  cat(sprintf("BV (jump-robust):   %.8f\n", summary_stats$bv))
  cat(sprintf("Realized Kernel:    %.8f\n", rk))
  cat(sprintf("TSRV (noise-adj):   %.8f\n", tsrv_res$tsrv))
  cat(sprintf("Jump fraction:      %.2f%%\n",
              100*summary_stats$jump_component/summary_stats$rv))

  invisible(list(summary=summary_stats, rk=rk, tsrv=tsrv_res,
                 noise=noise_est, bns=bns_res, lm=lm_res,
                 signature=sig_plot))
}

# -----------------------------------------------------------------------------
# 12. INTRADAY RETURN AUTOCORRELATION STRUCTURE
# -----------------------------------------------------------------------------

#' Compute intraday return autocorrelations at multiple lags
#' Negative first-order autocorrelation = bid-ask bounce
#' @param r intraday log-returns
intraday_acf <- function(r, max_lag = 20) {
  n <- length(r)
  acf_vals <- sapply(seq_len(max_lag), function(k) {
    cor(r[1:(n-k)], r[(k+1):n])
  })
  # Bartlett standard errors for WN
  se <- 1 / sqrt(n)
  sig_idx <- which(abs(acf_vals) > 2 * se)

  cat("=== Intraday Return ACF ===\n")
  cat(sprintf("Lag 1 autocorrelation: %.4f (%s)\n",
              acf_vals[1],
              ifelse(acf_vals[1] < -0.05, "bid-ask bounce signature",
                     ifelse(acf_vals[1] > 0.05, "momentum", "no pattern"))))
  cat(sprintf("Significant lags: %s\n",
              if (length(sig_idx)==0) "none" else paste(sig_idx, collapse=",")))

  list(acf=acf_vals, se=se, sig_lags=sig_idx)
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# # Simulate 1-minute returns for one trading day (24*60 = 1440 minutes for crypto)
# M <- 1440
# sigma_intra <- 0.0002  # per-minute vol
# sigma_noise <- 0.00005  # microstructure noise
# log_p_true <- cumsum(c(0, rnorm(M, 0, sigma_intra)))
# # Add noise
# log_p_obs <- log_p_true + rnorm(M+1, 0, sigma_noise)
# result <- run_hf_analysis(log_p_obs, "BTC_1min")

# =============================================================================
# EXTENDED HIGH-FREQUENCY STATISTICS: Market Microstructure Models,
# Liquidity Measures, Optimal Execution, and Intraday Patterns
# =============================================================================

# -----------------------------------------------------------------------------
# Roll Spread Estimator: estimate effective bid-ask spread from price changes
# Roll (1984): spread = 2 * sqrt(-cov(delta_p_t, delta_p_{t-1})) if cov < 0
# The bid-ask bounce induces negative serial covariance in transaction prices
# -----------------------------------------------------------------------------
roll_spread <- function(log_prices, min_obs = 30) {
  delta_p <- diff(log_prices)
  n <- length(delta_p)
  if (n < min_obs) return(list(spread = NA, gamma = NA))

  # Serial covariance of price changes
  gamma <- mean(delta_p[-n] * delta_p[-1])  # cov(delta_p_t, delta_p_{t-1})

  spread <- if (gamma < 0) 2 * sqrt(-gamma) else 0

  list(spread = spread, gamma = gamma,
       spread_bps = spread * 1e4,
       n_obs = n)
}

# -----------------------------------------------------------------------------
# Amihud Illiquidity Measure: |return| / dollar_volume
# Higher values = more price impact per dollar traded = less liquid
# Widely used in crypto to proxy for market depth
# -----------------------------------------------------------------------------
amihud_illiquidity <- function(returns, dollar_volumes, window = 30) {
  n <- length(returns)
  illiq <- abs(returns) / (dollar_volumes + 1)  # +1 to avoid div by zero

  # Rolling average
  rolling_illiq <- filter(illiq, rep(1/window, window), sides=1)

  list(
    daily_illiquidity = illiq,
    rolling_illiquidity = rolling_illiq,
    avg_illiquidity = mean(illiq, na.rm=TRUE),
    # Annualized price impact per $1M traded (rough)
    price_impact_per_1M = mean(illiq) * 1e6,
    illiquidity_ratio = mean(illiq) / sd(illiq)
  )
}

# -----------------------------------------------------------------------------
# Kyle's Lambda (Price Impact): slope of price change on order flow imbalance
# Lambda = cov(delta_p, order_flow) / var(order_flow)
# In crypto, can proxy order flow by signed volume (buy - sell)
# -----------------------------------------------------------------------------
kyle_lambda <- function(price_changes, order_flow, window = 60) {
  n <- length(price_changes)
  stopifnot(length(order_flow) == n)

  # Full-sample lambda
  lambda_full <- cov(price_changes, order_flow) / var(order_flow)

  # Rolling lambda
  rolling_lambda <- rep(NA, n)
  for (t in window:n) {
    idx <- (t - window + 1):t
    v_of <- var(order_flow[idx])
    if (v_of > 0) {
      rolling_lambda[t] <- cov(price_changes[idx], order_flow[idx]) / v_of
    }
  }

  list(
    lambda = lambda_full,
    rolling_lambda = rolling_lambda,
    avg_lambda = mean(rolling_lambda, na.rm=TRUE),
    # Interpretation: price moves lambda * order_flow per unit of flow
    interpretation = paste0("$1 of order imbalance moves price by ",
                             round(lambda_full * 1e4, 4), " bps")
  )
}

# -----------------------------------------------------------------------------
# PIN Model (Probability of Informed Trading): Easley et al. (1996)
# Separates informed trades from noise traders via buy/sell order counts
# High PIN = more information asymmetry = wider spreads expected
# -----------------------------------------------------------------------------
pin_model <- function(buys, sells, max_iter = 200) {
  # buys, sells: vectors of daily buy and sell counts
  n_days <- length(buys)
  stopifnot(length(sells) == n_days)

  # Log-likelihood for PIN model
  # Parameters: alpha (P(info event)), delta (P(bad news|event)),
  #             mu (informed arrival rate), epsilon_b, epsilon_s (noise rates)
  pin_loglik <- function(params) {
    alpha <- plogis(params[1])   # constrain to (0,1)
    delta <- plogis(params[2])
    mu    <- exp(params[3])      # positive
    eps_b <- exp(params[4])
    eps_s <- exp(params[5])

    # Conditional density for each day
    ll <- 0
    for (i in 1:n_days) {
      B <- buys[i]; S <- sells[i]
      # Three scenarios: no event, good news, bad news
      lam_b_no  <- eps_b;       lam_s_no  <- eps_s
      lam_b_good <- mu + eps_b; lam_s_good <- eps_s
      lam_b_bad  <- eps_b;      lam_s_bad  <- mu + eps_s

      # Poisson log-densities (log to avoid underflow)
      log_no   <- B*log(lam_b_no)   - lam_b_no   + S*log(lam_s_no)   - lam_s_no
      log_good <- B*log(lam_b_good) - lam_b_good + S*log(lam_s_good) - lam_s_good
      log_bad  <- B*log(lam_b_bad)  - lam_b_bad  + S*log(lam_s_bad)  - lam_s_bad

      # Log mixture probability (using log-sum-exp for stability)
      max_log <- max(log_no, log_good, log_bad)
      log_mix <- max_log + log(
        (1 - alpha) * exp(log_no - max_log) +
        alpha * (1 - delta) * exp(log_good - max_log) +
        alpha * delta * exp(log_bad - max_log)
      )
      ll <- ll + log_mix - lgamma(B+1) - lgamma(S+1)
    }
    ll
  }

  # Optimize
  init <- c(0, 0, log(mean(buys + sells)/2), log(mean(buys)), log(mean(sells)))
  opt <- tryCatch(
    optim(init, pin_loglik, control = list(fnscale=-1, maxit=max_iter)),
    error = function(e) list(par = init, convergence = 1)
  )

  alpha <- plogis(opt$par[1]); delta <- plogis(opt$par[2])
  mu    <- exp(opt$par[3]); eps_b <- exp(opt$par[4]); eps_s <- exp(opt$par[5])

  # PIN = alpha*mu / (alpha*mu + eps_b + eps_s)
  pin <- alpha * mu / (alpha * mu + eps_b + eps_s)

  list(
    pin = pin,
    alpha = alpha, delta = delta,
    mu = mu, epsilon_buy = eps_b, epsilon_sell = eps_s,
    loglik = opt$value,
    convergence = opt$convergence,
    interpretation = paste0("PIN = ", round(pin, 4),
                             " (", round(pin*100, 1), "% of trades are informed)")
  )
}

# -----------------------------------------------------------------------------
# Corwin-Schultz Spread Estimator: uses daily high-low ranges
# Spread derived from H/L ratio over 1-day and 2-day windows
# Useful when tick-level data unavailable but OHLC data exists
# -----------------------------------------------------------------------------
corwin_schultz_spread <- function(highs, lows) {
  n <- length(highs); stopifnot(length(lows) == n)

  beta <- function(h, l) {
    sum((log(h/l))^2)  # sum of squared log(H/L) ratios
  }

  spreads <- rep(NA, n-1)
  for (i in 2:n) {
    # Beta_t: sum of squared 1-period log-range
    beta_t <- (log(highs[i] / lows[i]))^2 + (log(highs[i-1] / lows[i-1]))^2
    # Gamma_t: squared 2-period log-range
    gamma_t <- (log(max(highs[i], highs[i-1]) / min(lows[i], lows[i-1])))^2

    # Spread estimate
    alpha_t <- (sqrt(2 * beta_t) - sqrt(beta_t)) / (3 - 2*sqrt(2)) -
               sqrt(gamma_t / (3 - 2*sqrt(2)))
    spreads[i] <- 2 * (exp(alpha_t) - 1) / (1 + exp(alpha_t))
    spreads[i] <- max(spreads[i], 0)  # negative estimates set to 0
  }

  list(
    spreads = spreads,
    avg_spread = mean(spreads, na.rm=TRUE),
    avg_spread_bps = mean(spreads, na.rm=TRUE) * 1e4,
    pct_negative = mean(spreads < 0, na.rm=TRUE)
  )
}

# -----------------------------------------------------------------------------
# VWAP and TWAP: volume/time weighted average prices for execution benchmarking
# Strategies are evaluated against these benchmarks to measure execution quality
# -----------------------------------------------------------------------------
compute_vwap <- function(prices, volumes) {
  stopifnot(length(prices) == length(volumes))
  vwap <- cumsum(prices * volumes) / cumsum(volumes)
  twap <- cumsum(prices) / seq_along(prices)

  # VWAP deviation: how far execution price deviates from VWAP
  list(
    vwap = vwap, twap = twap,
    final_vwap = vwap[length(vwap)],
    final_twap = twap[length(twap)],
    price_range = range(prices),
    vol_weighted_std = sqrt(sum(volumes * (prices - vwap[length(vwap)])^2) / sum(volumes))
  )
}

# -----------------------------------------------------------------------------
# Almgren-Chriss Optimal Execution: liquidate X shares over T periods
# Minimizes E[cost] + lambda * Var[cost] where lambda = risk aversion
# Gives optimal trading schedule as function of urgency vs market impact
# -----------------------------------------------------------------------------
almgren_chriss_schedule <- function(X, T_periods, sigma, eta, gamma_perm,
                                     lambda = 1e-6) {
  # X: total shares (or USD notional) to liquidate
  # T: number of time periods (e.g., minutes or 5-min bars)
  # sigma: per-period volatility
  # eta: temporary impact coefficient
  # gamma: permanent impact coefficient
  # lambda: risk aversion coefficient

  # Optimal trajectory: x_t = X * sinh(kappa*(T-t)/T) / sinh(kappa)
  # where kappa = arccosh(1 + lambda * sigma^2 * T^2 / (2*eta))
  kappa_arg <- 1 + lambda * sigma^2 * T_periods^2 / (2 * eta)
  if (kappa_arg < 1) kappa_arg <- 1
  kappa <- acosh(kappa_arg)

  t_grid <- 0:T_periods
  # Remaining inventory at each time step
  x_t <- X * sinh(kappa * (T_periods - t_grid) / T_periods) /
           (sinh(kappa) + 1e-10)
  x_t <- pmax(x_t, 0)

  # Trading rates (shares per period)
  n_t <- -diff(x_t)

  # Expected cost and variance
  E_cost <- 0.5 * gamma_perm * X^2 +
            eta * sum(n_t^2) +
            0.5 * gamma_perm * sum(x_t[-1] * n_t)
  Var_cost <- sigma^2 * sum(x_t[-1]^2)

  list(
    inventory_path = x_t,
    trade_schedule = n_t,
    expected_cost = E_cost,
    variance_cost = Var_cost,
    risk_adjusted_cost = E_cost + lambda * Var_cost,
    kappa = kappa,
    # Urgency: kappa = 0 is TWAP, high kappa = front-load trades
    urgency = ifelse(kappa < 0.1, "TWAP-like", ifelse(kappa < 1, "moderate", "urgent"))
  )
}

# -----------------------------------------------------------------------------
# Intraday Return Autocorrelation by Lag and Time-of-Day
# Identify microstructure effects: bid-ask bounce (lag-1 negative AC),
# momentum at short horizons, and mean-reversion at longer horizons
# -----------------------------------------------------------------------------
intraday_microstructure_ac <- function(log_prices, n_lags = 20) {
  returns <- diff(log_prices)
  n <- length(returns)

  # Standard ACF
  ac_vals <- acf(returns, lag.max = n_lags, plot = FALSE)$acf[-1]

  # Ljung-Box test for all lags
  lb_stat <- n * (n+2) * sum(ac_vals^2 / (n - 1:n_lags))
  lb_pval <- pchisq(lb_stat, df = n_lags, lower.tail = FALSE)

  # Variance ratio to detect random walk deviation
  # VR(k) = Var(k-period return) / (k * Var(1-period return))
  var1 <- var(returns)
  vr <- sapply(2:min(n_lags, floor(n/10)), function(k) {
    returns_k <- diff(log_prices[seq(1, n+1, by=k)])
    var(returns_k) / (k * var1)
  })

  # Bid-ask bounce signature: first lag AC should be negative for noisy data
  bounce_detected <- ac_vals[1] < -0.1

  list(
    autocorrelations = ac_vals,
    lags = 1:n_lags,
    ljung_box_stat = lb_stat, ljung_box_pval = lb_pval,
    variance_ratios = vr,
    vr_lags = 2:min(n_lags, floor(n/10)),
    bid_ask_bounce = bounce_detected,
    first_lag_ac = ac_vals[1],
    # Effective spread proxy from bid-ask bounce
    roll_spread_proxy = if (ac_vals[1] < 0) 2*sqrt(-ac_vals[1]*var1) else 0
  )
}

# Extended HF example:
# spread <- roll_spread(log_prices)
# illiq  <- amihud_illiquidity(returns, dollar_vol)
# kyle   <- kyle_lambda(delta_prices, order_flow)
# cs_spr <- corwin_schultz_spread(high_prices, low_prices)
# exec   <- almgren_chriss_schedule(X=1e6, T_periods=20, sigma=0.001,
#             eta=0.01, gamma_perm=0.001, lambda=1e-5)
# mic_ac <- intraday_microstructure_ac(log_prices)
