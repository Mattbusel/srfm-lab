# =============================================================================
# market_microstructure.R
# Market Microstructure Analytics for Crypto
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Crypto markets operate 24/7 with highly variable
# liquidity. Understanding the bid-ask spread, price impact, and order flow
# toxicity is essential for realistic backtesting and execution optimisation.
# These metrics also predict short-term returns: high illiquidity -> reversal;
# high informed trading (PIN) -> momentum.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

#' Rolling mean
roll_mean <- function(x, w) {
  n <- length(x); out <- rep(NA, n)
  for (i in w:n) out[i] <- mean(x[(i-w+1):i], na.rm=TRUE)
  out
}

#' Rolling sd
roll_sd <- function(x, w) {
  n <- length(x); out <- rep(NA, n)
  for (i in w:n) out[i] <- sd(x[(i-w+1):i], na.rm=TRUE)
  out
}

#' Sharpe ratio
sharpe_ratio <- function(rets, ann = 252) {
  mu <- mean(rets, na.rm=TRUE); sg <- sd(rets, na.rm=TRUE)
  if (is.na(sg) || sg < 1e-12) return(NA_real_)
  mu / sg * sqrt(ann)
}

# ---------------------------------------------------------------------------
# 2. DATA SIMULATION (Intraday Bars)
# ---------------------------------------------------------------------------

simulate_intraday <- function(n_days = 30L, bars_per_day = 24L,
                               seed = 42L) {
  set.seed(seed)
  n_total <- n_days * bars_per_day
  hour    <- rep(0:(bars_per_day-1), n_days)
  day     <- rep(seq_len(n_days), each = bars_per_day)

  # Intraday volatility seasonality (U-shaped for crypto)
  hour_vol <- 0.8 + 0.4 * cos(2 * pi * (hour - 4) / 24)^2

  price    <- numeric(n_total); price[1] <- 10000
  volume   <- numeric(n_total)
  spread   <- numeric(n_total)

  for (t in 2:n_total) {
    sig        <- 0.005 * hour_vol[t]
    price[t]   <- price[t-1] * exp(rnorm(1, 0, sig))
    volume[t]  <- rpois(1, 1000 * hour_vol[t]) + 1
    # Spread: positively correlated with vol
    spread[t]  <- pmax(0.0002 + 0.002 * hour_vol[t] + rnorm(1, 0, 0.0001), 0.0001)
  }

  high  <- price * (1 + abs(rnorm(n_total, 0, 0.001)))
  low   <- price * (1 - abs(rnorm(n_total, 0, 0.001)))
  # Ensure OHLC relationship
  open  <- c(price[1], price[-n_total])
  close <- price

  data.frame(t = seq_len(n_total), hour = hour, day = day,
             open = open, high = high, low = low, close = close,
             volume = volume, spread_pct = spread)
}

# ---------------------------------------------------------------------------
# 3. ROLL SPREAD ESTIMATOR
# ---------------------------------------------------------------------------
# Roll (1984): implied spread = 2 * sqrt(-Cov(delta_p_t, delta_p_{t-1}))
# Works on price changes; uses serial covariance of returns.
# Financial intuition: bid-ask bounce creates negative serial correlation.

roll_spread <- function(prices, window = 50L) {
  n   <- length(prices)
  dp  <- diff(prices)
  spr <- rep(NA, n)
  for (i in (window + 1):n) {
    dp_w <- dp[(i - window):(i - 1)]
    cov_ <- cov(dp_w[-length(dp_w)], dp_w[-1])
    spr[i] <- if (cov_ < 0) 2 * sqrt(-cov_) else 0
  }
  # As fraction of mid-price
  spr / prices
}

# ---------------------------------------------------------------------------
# 4. KYLE LAMBDA (PRICE IMPACT)
# ---------------------------------------------------------------------------
# Lambda = dP / dOrderFlow
# Higher lambda -> more price impact per unit of signed volume.
# Estimated by regressing price change on signed order flow.

#' Estimate Kyle lambda via rolling regression
kyle_lambda <- function(price_changes, signed_volume, window = 50L) {
  n   <- length(price_changes)
  lam <- rep(NA, n)
  for (i in (window + 1):n) {
    y <- price_changes[(i - window):(i - 1)]
    x <- signed_volume[(i - window):(i - 1)]
    if (var(x) < 1e-12) next
    b <- cov(x, y) / var(x)   # OLS slope
    lam[i] <- b
  }
  lam
}

#' Simulate signed order flow (buy/sell imbalance)
simulate_order_flow <- function(returns, noise_scale = 0.5) {
  # Order flow correlated with contemporaneous returns
  sign_r <- sign(returns)
  flow   <- sign_r * abs(rnorm(length(returns), 1, noise_scale))
  flow
}

# ---------------------------------------------------------------------------
# 5. AMIHUD ILLIQUIDITY RATIO
# ---------------------------------------------------------------------------
# Amihud (2002): ILLIQ = |r_t| / Volume_t
# High ILLIQ = big price movement per unit of volume = illiquid.
# Daily rolling average gives the illiquidity regime.

amihud_illiq <- function(returns, volume, window = 20L) {
  n    <- length(returns)
  ilq  <- abs(returns) / pmax(volume, 1)
  roll_mean(ilq, window)
}

# ---------------------------------------------------------------------------
# 6. PIN MODEL (Probability of Informed Trading)
# ---------------------------------------------------------------------------
# Easley et al.: market has informed and noise traders.
# Observed buys B and sells S on a day come from a mixture:
#   - With prob alpha: informed day (extra mu buys if good news, mu sells if bad)
#   - With prob 1-alpha: no info event
# Parameters: alpha (prob info event), delta (bad news conditional prob),
#             mu (informed arrival rate), epsilon (noise arrival rates)
# MLE via EM-like numerical optimisation.

pin_loglike <- function(params, B, S) {
  alpha   <- clip(params[1], 1e-4, 1-1e-4)
  delta   <- clip(params[2], 1e-4, 1-1e-4)
  mu      <- pmax(params[3], 1e-4)
  eps_b   <- pmax(params[4], 1e-4)
  eps_s   <- pmax(params[5], 1e-4)

  n <- length(B)
  ll <- 0
  for (i in seq_len(n)) {
    b <- B[i]; s <- S[i]
    # No-info: B~Pois(eps_b), S~Pois(eps_s)
    L0 <- (1 - alpha) * dpois(b, eps_b) * dpois(s, eps_s)
    # Good news: B~Pois(mu+eps_b), S~Pois(eps_s)
    L1 <- alpha * (1 - delta) * dpois(b, mu + eps_b) * dpois(s, eps_s)
    # Bad news: B~Pois(eps_b), S~Pois(mu+eps_s)
    L2 <- alpha * delta * dpois(b, eps_b) * dpois(s, mu + eps_s)
    L  <- L0 + L1 + L2
    ll <- ll + log(pmax(L, 1e-300))
  }
  -ll   # negative for minimisation
}

#' Fit PIN model via grid start + Nelder-Mead simplex
fit_pin <- function(B, S, n_starts = 10L, seed = 1L) {
  set.seed(seed)
  best_ll  <- Inf
  best_par <- c(0.3, 0.5, mean(c(B,S))/2, mean(B), mean(S))

  for (s in seq_len(n_starts)) {
    init <- c(runif(1, 0.1, 0.6),
              runif(1, 0.2, 0.8),
              runif(1, 0.5, 2) * mean(c(B, S)),
              max(mean(B) * runif(1, 0.5, 1.5), 0.01),
              max(mean(S) * runif(1, 0.5, 1.5), 0.01))
    res <- tryCatch(
      optim(init, pin_loglike, B = B, S = S, method = "Nelder-Mead",
            control = list(maxit = 1000, reltol = 1e-6)),
      error = function(e) list(value = Inf, par = init))
    if (res$value < best_ll) { best_ll <- res$value; best_par <- res$par }
  }

  alpha <- clip(best_par[1], 0, 1)
  delta <- clip(best_par[2], 0, 1)
  mu    <- pmax(best_par[3], 0)
  eps_b <- pmax(best_par[4], 0)
  eps_s <- pmax(best_par[5], 0)
  PIN   <- alpha * mu / (alpha * mu + eps_b + eps_s)

  list(PIN = PIN, alpha = alpha, delta = delta,
       mu = mu, eps_b = eps_b, eps_s = eps_s,
       loglik = -best_ll)
}

# ---------------------------------------------------------------------------
# 7. CORWIN-SCHULTZ HIGH-LOW SPREAD ESTIMATOR
# ---------------------------------------------------------------------------
# CS (2012): spread from H/L over 1-day and 2-day windows.
# s = 2*(exp(alpha) - 1) / (1 + exp(alpha))

cs_spread <- function(high, low) {
  n <- length(high)
  beta <- numeric(n - 1)
  for (t in 2:n) {
    ln_h2 <- (log(pmax(high[t],  high[t-1])  / pmin(low[t],  low[t-1])))^2
    ln_h1 <- ((log(high[t]   / low[t]))^2 + (log(high[t-1] / low[t-1]))^2)
    beta[t-1] <- ln_h2 - ln_h1 / 2
  }
  gamma <- (log(pmax(high[-1], high[-n]) / pmin(low[-1], low[-n])))^2
  alpha <- (sqrt(2 * pmax(beta, 0)) - sqrt(pmax(beta, 0))) / (3 - 2*sqrt(2)) -
    sqrt(gamma / (3 - 2*sqrt(2)))
  spread <- 2 * (exp(alpha) - 1) / (1 + exp(alpha))
  c(NA, pmax(spread, 0))
}

# ---------------------------------------------------------------------------
# 8. INTRADAY SEASONALITY: FFF REGRESSION
# ---------------------------------------------------------------------------
# Flexible Fourier Form for intraday vol seasonality.
# sigma(tau) = c0 + sum_{k=1}^{K} [a_k*cos(2pi*k*tau/M) + b_k*sin(2pi*k*tau/M)]
# where tau = bar index within day, M = bars per day.

#' Build FFF design matrix
fff_matrix <- function(hour_index, M = 24L, K = 3L) {
  tau <- hour_index
  X   <- matrix(1, length(tau), 1 + 2*K)
  for (k in seq_len(K)) {
    X[, 2*k]     <- cos(2 * pi * k * tau / M)
    X[, 2*k + 1] <- sin(2 * pi * k * tau / M)
  }
  X
}

#' Fit FFF model by OLS
fit_fff <- function(squared_returns, hour_index, M = 24L, K = 3L) {
  X    <- fff_matrix(hour_index, M, K)
  y    <- pmax(squared_returns, 0)
  coef <- tryCatch(solve(t(X) %*% X + diag(1e-8, ncol(X)),
                          t(X) %*% y),
                    error = function(e) rep(0, ncol(X)))
  fitted <- as.numeric(X %*% coef)
  list(coef = coef, fitted = fitted, residuals = y - fitted)
}

#' Diurnal-adjusted returns (standardised by FFF vol estimate)
diurnal_adjust <- function(returns, hour_index, M = 24L) {
  sq_ret <- returns^2
  fff    <- fit_fff(sq_ret, hour_index, M)
  sigma_hat <- sqrt(pmax(fff$fitted, 1e-10))
  list(adjusted = returns / sigma_hat,
       sigma_hat = sigma_hat,
       fff = fff)
}

# ---------------------------------------------------------------------------
# 9. QUOTE STUFFING / LAYERING DETECTION
# ---------------------------------------------------------------------------
# Heuristic: unusually high quote-to-trade (QTR) ratio in a short window
# signals potential layering / spoofing.

quote_stuffing_score <- function(quote_count, trade_count,
                                  window = 10L, threshold = 10.0) {
  n    <- length(quote_count)
  qtr  <- quote_count / pmax(trade_count, 1)
  roll_qtr  <- roll_mean(qtr, window)
  baseline  <- roll_mean(qtr, window * 5L)

  score <- ifelse(is.na(baseline), 0,
                   roll_qtr / pmax(baseline, 0.01))
  flag  <- score > threshold
  data.frame(qtr = qtr, roll_qtr = roll_qtr,
             score = score, flag = flag)
}

# ---------------------------------------------------------------------------
# 10. VWAP EXECUTION BENCHMARK
# ---------------------------------------------------------------------------
# VWAP = sum(price * volume) / sum(volume) over trading window.
# Execution quality: compare fill price to VWAP (implementation shortfall).

compute_vwap <- function(prices, volumes, window = NULL) {
  n <- length(prices)
  if (is.null(window)) {
    # Full-day VWAP
    vwap <- sum(prices * volumes) / sum(volumes)
    return(vwap)
  }
  # Rolling VWAP
  vwap <- rep(NA, n)
  for (i in window:n) {
    p_w <- prices[(i-window+1):i]
    v_w <- volumes[(i-window+1):i]
    vwap[i] <- sum(p_w * v_w) / sum(v_w)
  }
  vwap
}

#' VWAP execution slippage
vwap_slippage <- function(fill_prices, fill_volumes, bar_prices, bar_volumes,
                            start_bar, end_bar) {
  # VWAP of market over execution window
  mkt_vwap <- compute_vwap(bar_prices[start_bar:end_bar],
                            bar_volumes[start_bar:end_bar])
  # Weighted avg fill
  avg_fill <- sum(fill_prices * fill_volumes) / sum(fill_volumes)
  data.frame(
    vwap      = mkt_vwap,
    avg_fill  = avg_fill,
    slippage  = avg_fill - mkt_vwap,
    slippage_bps = (avg_fill - mkt_vwap) / mkt_vwap * 1e4
  )
}

#' TWAP benchmark
compute_twap <- function(prices, start, end) mean(prices[start:end])

# ---------------------------------------------------------------------------
# 11. EFFECTIVE SPREAD
# ---------------------------------------------------------------------------
# Effective spread = 2 * |trade_price - midpoint|
# Realised spread (short-horizon): compares fill to midpoint 5 bars later

effective_spread <- function(trade_prices, mid_prices) {
  2 * abs(trade_prices - mid_prices)
}

realised_spread <- function(trade_prices, mid_prices, delay = 5L) {
  n  <- length(trade_prices)
  rs <- rep(NA, n)
  for (t in seq_len(n - delay)) {
    rs[t] <- 2 * (trade_prices[t] - mid_prices[t + delay])
  }
  rs
}

price_impact <- function(trade_prices, mid_prices, delay = 5L) {
  n  <- length(trade_prices)
  pi_  <- rep(NA, n)
  for (t in seq_len(n - delay)) {
    pi_[t] <- mid_prices[t + delay] - mid_prices[t]
  }
  pi_
}

# ---------------------------------------------------------------------------
# 12. LIQUIDITY-ADJUSTED RETURNS
# ---------------------------------------------------------------------------
# Adjust raw returns for estimated bid-ask cost.
# Financial intuition: a strategy's true edge must exceed transaction costs.

liquidity_adjusted_ret <- function(raw_return, spread, position_change) {
  # Cost = (spread/2) * |position change|
  raw_return - spread / 2 * abs(position_change)
}

#' Impact of illiquidity on strategy returns
illiquidity_drag <- function(returns, amihud, position_size, vol_window = 20L) {
  n      <- length(returns)
  ilq_ma <- roll_mean(amihud, vol_window)
  drag   <- position_size^2 * ilq_ma   # Kyle-Amihud impact
  adj_returns <- returns - drag
  data.frame(gross_ret = returns, drag = drag, net_ret = adj_returns,
             cumulative_drag = cumsum(drag))
}

# ---------------------------------------------------------------------------
# 13. MICROSTRUCTURE NOISE DECOMPOSITION
# ---------------------------------------------------------------------------
# Separate variance into "fundamental" variance and noise variance.
# Method: use autocorrelation of high-frequency returns.

noise_variance <- function(returns) {
  n   <- length(returns)
  # Under noise model: Var(r_t) = sig_f^2 + 2*sig_noise^2
  # Cov(r_t, r_{t-1}) = -sig_noise^2
  var_r <- var(returns, na.rm=TRUE)
  acov1 <- cov(returns[-n], returns[-1])
  sig2_noise <- pmax(-acov1, 0)
  sig2_fund  <- pmax(var_r - 2 * sig2_noise, 0)
  list(
    total_var       = var_r,
    fundamental_var = sig2_fund,
    noise_var       = sig2_noise,
    noise_ratio     = sig2_noise / max(var_r, 1e-12)
  )
}

# ---------------------------------------------------------------------------
# 14. ORDER BOOK IMBALANCE
# ---------------------------------------------------------------------------
# OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume) in [-1,1]
# Positive OBI -> buying pressure -> price likely to rise.

order_book_imbalance <- function(bid_volumes, ask_volumes) {
  tot <- bid_volumes + ask_volumes
  (bid_volumes - ask_volumes) / pmax(tot, 1)
}

#' OBI signal: rolling z-score
obi_signal <- function(bid_vol, ask_vol, window = 20L) {
  obi    <- order_book_imbalance(bid_vol, ask_vol)
  mu_obi <- roll_mean(obi, window)
  sd_obi <- roll_sd(obi, window)
  zscore <- (obi - mu_obi) / pmax(sd_obi, 1e-8)
  data.frame(obi = obi, z = zscore)
}

# ---------------------------------------------------------------------------
# 15. MAIN DEMO
# ---------------------------------------------------------------------------

run_microstructure_demo <- function() {
  cat("=== Market Microstructure Analytics Demo ===\n\n")

  # Simulate intraday data
  cat("Simulating 30 days x 24 hourly bars...\n")
  bars <- simulate_intraday(n_days = 30L, bars_per_day = 24L, seed = 42L)
  cat(sprintf("  Bars: %d  |  Price range: %.0f - %.0f\n",
              nrow(bars), min(bars$close), max(bars$close)))

  cat("\n--- 1. Roll Spread Estimator ---\n")
  rs    <- roll_spread(bars$close, window = 40L)
  valid <- !is.na(rs)
  cat(sprintf("  Estimated spread (bps): mean=%.2f  max=%.2f\n",
              mean(rs[valid]) * 1e4, max(rs[valid]) * 1e4))

  cat("\n--- 2. Kyle Lambda ---\n")
  rets_bar <- c(NA, diff(log(bars$close)))
  of       <- simulate_order_flow(rets_bar)
  kl       <- kyle_lambda(rets_bar, of, window = 40L)
  valid    <- !is.na(kl)
  cat(sprintf("  Kyle lambda: mean=%.6f  (price move per $1 flow)\n",
              mean(kl[valid])))

  cat("\n--- 3. Amihud Illiquidity ---\n")
  amihud <- amihud_illiq(rets_bar, bars$volume, window = 20L)
  valid  <- !is.na(amihud)
  cat(sprintf("  Amihud (x1e4): mean=%.4f\n", mean(amihud[valid]) * 1e4))

  cat("\n--- 4. Corwin-Schultz Spread ---\n")
  cs_spr <- cs_spread(bars$high, bars$low)
  valid  <- !is.na(cs_spr) & cs_spr > 0
  if (sum(valid) > 0)
    cat(sprintf("  CS spread (bps): mean=%.2f\n", mean(cs_spr[valid]) * 1e4))
  else
    cat("  CS spread: insufficient valid estimates\n")

  cat("\n--- 5. PIN Model ---\n")
  # Aggregate to daily Buy/Sell counts
  set.seed(11)
  n_days <- 60L
  B_day  <- rpois(n_days, 500)
  S_day  <- rpois(n_days, 480)
  # Add some informed trading days
  informed_days <- sample.int(n_days, 10)
  B_day[informed_days] <- B_day[informed_days] + rpois(10, 300)
  pin_fit <- fit_pin(B_day, S_day, n_starts = 5L)
  cat(sprintf("  PIN=%.4f  alpha=%.3f  delta=%.3f  mu=%.1f\n",
              pin_fit$PIN, pin_fit$alpha, pin_fit$delta, pin_fit$mu))
  # Financial intuition: PIN > 0.15 suggests significant informed trading

  cat("\n--- 6. FFF Intraday Seasonality ---\n")
  da <- diurnal_adjust(rets_bar[-1], bars$hour[-1], M = 24L)
  cat(sprintf("  Sigma_hat range: %.4f - %.4f\n",
              min(da$sigma_hat), max(da$sigma_hat)))
  # Check seasonality: compare high-vol hours to low-vol hours
  cat(sprintf("  Ratio high/low hour vol: %.2fx\n",
              max(da$sigma_hat) / min(da$sigma_hat)))

  cat("\n--- 7. Noise Variance Decomposition ---\n")
  nv <- noise_variance(rets_bar[!is.na(rets_bar)])
  cat(sprintf("  Total var: %.6f  |  Fundamental: %.6f  |  Noise ratio: %.3f\n",
              nv$total_var, nv$fundamental_var, nv$noise_ratio))

  cat("\n--- 8. VWAP Analysis ---\n")
  day1_prices <- bars$close[1:24]
  day1_vols   <- bars$volume[1:24]
  vwap_full <- compute_vwap(day1_prices, day1_vols)
  cat(sprintf("  Day 1 VWAP: %.2f  |  Close: %.2f\n",
              vwap_full, tail(day1_prices, 1)))
  # Simulated execution fills
  fill_p <- day1_prices[8:12] + rnorm(5, 0, 1)
  fill_v <- rep(200, 5)
  slippage <- vwap_slippage(fill_p, fill_v, day1_prices, day1_vols, 8, 12)
  cat(sprintf("  Execution slippage vs VWAP: %.2f bps\n", slippage$slippage_bps))

  cat("\n--- 9. Quote Stuffing Detection ---\n")
  set.seed(22)
  quotes <- rpois(nrow(bars), 50)
  trades <- rpois(nrow(bars), 10)
  # Inject stuffing episode
  stuffing_bars <- 200:210
  quotes[stuffing_bars] <- 2000
  qs <- quote_stuffing_score(quotes, trades, window = 10L)
  n_flags <- sum(qs$flag, na.rm=TRUE)
  cat(sprintf("  Flagged bars: %d  |  Max score: %.1f  (injected at bars 200-210)\n",
              n_flags, max(qs$score, na.rm=TRUE)))

  cat("\n--- 10. Order Book Imbalance Signal ---\n")
  set.seed(33)
  bid_v <- rpois(nrow(bars), 500) + 100 * (rets_bar > 0)
  ask_v <- rpois(nrow(bars), 500) + 100 * (rets_bar < 0)
  obi   <- obi_signal(bid_v[!is.na(bid_v)], ask_v[!is.na(ask_v)], window=20L)
  cor_obi_ret <- cor(obi$z[21:nrow(obi)],
                      rets_bar[22:length(rets_bar)], use="complete.obs")
  cat(sprintf("  OBI z-score vs next-bar return corr: %.4f\n", cor_obi_ret))

  cat("\n--- 11. Liquidity-Adjusted Returns ---\n")
  pos_changes <- diff(c(0, sign(rets_bar[!is.na(rets_bar)])))
  net_rets <- liquidity_adjusted_ret(
    rets_bar[!is.na(rets_bar)],
    rs[!is.na(rs)][1:sum(!is.na(rets_bar))],
    pos_changes)
  cat(sprintf("  Gross Sharpe: %.3f  |  Net (after spread): %.3f\n",
              sharpe_ratio(rets_bar, ann=252*24),
              sharpe_ratio(net_rets, ann=252*24)))

  cat("\nDone.\n")
  invisible(list(bars=bars, pin=pin_fit, da=da, nv=nv))
}

if (interactive()) {
  ms_results <- run_microstructure_demo()
}

# ---------------------------------------------------------------------------
# 15. TICK RULE AND LEE-READY TRADE CLASSIFICATION
# ---------------------------------------------------------------------------
# Assign trade direction: buy if price up-tick, sell if down-tick.
# Lee-Ready uses midpoint rule for quote-stamped data.

tick_rule <- function(prices) {
  n     <- length(prices)
  signs <- integer(n)
  last  <- 1L
  for (t in seq_len(n)) {
    if (t == 1) { signs[t] <- 1L; next }
    if (prices[t] > prices[t-1])      { signs[t] <- 1L;    last <- 1L }
    else if (prices[t] < prices[t-1]) { signs[t] <- -1L;   last <- -1L }
    else                               signs[t] <- last
  }
  signs
}

lee_ready <- function(trade_prices, bid, ask) {
  mid   <- (bid + ask) / 2
  signs <- integer(length(trade_prices))
  for (t in seq_along(trade_prices)) {
    if (trade_prices[t] > mid[t])       signs[t] <-  1L
    else if (trade_prices[t] < mid[t])  signs[t] <- -1L
    else signs[t] <- tick_rule(trade_prices[1:t])[t]
  }
  signs
}

# ---------------------------------------------------------------------------
# 16. TRADE IMBALANCE SIGNAL
# ---------------------------------------------------------------------------

trade_imbalance <- function(signed_volumes, window=20L) {
  n   <- length(signed_volumes)
  tib <- rep(NA, n)
  for (t in window:n) {
    sv  <- signed_volumes[(t-window+1):t]
    pos <- sum(sv[sv>0]); neg <- sum(abs(sv[sv<0]))
    tib[t] <- (pos - neg) / max(pos + neg, 1)
  }
  tib
}

# ---------------------------------------------------------------------------
# 17. REALIZED SPREAD DECOMPOSITION
# ---------------------------------------------------------------------------
# Effective spread = Adverse selection (price impact) + Realized spread
# (dealer's profit). Measures who wins in a trade.

spread_decomposition <- function(trade_prices, mid_prices, volumes,
                                  delay=5L) {
  n  <- length(trade_prices); results <- numeric(n)
  for (t in seq_len(n - delay)) {
    eff_spread <- 2 * abs(trade_prices[t] - mid_prices[t])
    real_spread <- 2 * (trade_prices[t] - mid_prices[t+delay]) *
      sign(trade_prices[t] - mid_prices[t])
    price_impact <- 2 * (mid_prices[t+delay] - mid_prices[t]) *
      sign(trade_prices[t] - mid_prices[t])
    results[t] <- price_impact / max(eff_spread, 1e-8)
  }
  list(adverse_selection_frac=results,
       mean_as=mean(results[results!=0],na.rm=TRUE))
}

# ---------------------------------------------------------------------------
# 18. MARKET IMPACT MODEL (ALMGREN-CHRISS SIMPLIFIED)
# ---------------------------------------------------------------------------
# Optimal execution: balance market impact vs timing risk.
# Cost = eta * (X/T)^2 * T + sigma^2 * T  (simplified)

ac_optimal_trajectory <- function(X_total, T_periods, eta, sigma) {
  # Optimal: trade equal amounts each period (simplified, risk-neutral)
  rate  <- X_total / T_periods
  times <- seq_len(T_periods)
  inventory <- X_total - rate * times
  impact_cost <- eta * rate^2 * T_periods
  risk_cost   <- 0.5 * sigma^2 * sum(inventory^2)
  list(rate=rate, inventory=c(X_total, inventory),
       impact_cost=impact_cost, risk_cost=risk_cost,
       total_cost=impact_cost+risk_cost)
}

# ---------------------------------------------------------------------------
# 19. INTRADAY LIQUIDITY PROFILE
# ---------------------------------------------------------------------------

intraday_liquidity_profile <- function(bars, vol_window=5L) {
  T_  <- nrow(bars)
  liq <- data.frame(
    hour        = bars$hour,
    spread      = bars$spread_pct,
    volume      = bars$volume,
    roll_spread = roll_mean(bars$spread_pct, vol_window),
    vol_score   = bars$volume / (roll_mean(bars$volume, 20L) + 1)
  )
  by_hour <- tapply(liq$spread, liq$hour, mean, na.rm=TRUE)
  list(by_bar=liq, by_hour=by_hour,
       best_hour=which.min(by_hour),
       worst_hour=which.max(by_hour))
}

# ---------------------------------------------------------------------------
# 20. EXTENDED MICROSTRUCTURE DEMO
# ---------------------------------------------------------------------------

run_microstructure_extended_demo <- function() {
  cat("=== Microstructure Extended Demo ===\n\n")
  bars <- simulate_intraday(n_days=20L, bars_per_day=24L, seed=77L)

  cat("--- Tick Rule ---\n")
  tr <- tick_rule(bars$close)
  cat(sprintf("  Buy ticks: %d  Sell ticks: %d\n",
              sum(tr==1), sum(tr==-1)))

  cat("\n--- Trade Imbalance ---\n")
  vol_signed <- bars$volume * tr
  ti <- trade_imbalance(vol_signed, window=20L)
  valid <- !is.na(ti)
  cat(sprintf("  Mean trade imbalance: %.4f  SD: %.4f\n",
              mean(ti[valid]), sd(ti[valid])))

  cat("\n--- Spread Decomposition ---\n")
  n_t <- nrow(bars) - 20L
  sd_res <- spread_decomposition(bars$close[1:n_t],
                                   (bars$high[1:n_t]+bars$low[1:n_t])/2,
                                   bars$volume[1:n_t])
  cat(sprintf("  Mean adverse selection fraction: %.4f\n", sd_res$mean_as))

  cat("\n--- Almgren-Chriss Execution ---\n")
  ac <- ac_optimal_trajectory(X_total=10000, T_periods=10L,
                               eta=0.001, sigma=0.005)
  cat(sprintf("  Rate: %.0f/bar  Impact cost: %.4f  Risk cost: %.4f\n",
              ac$rate, ac$impact_cost, ac$risk_cost))

  cat("\n--- Intraday Liquidity Profile ---\n")
  ilp <- intraday_liquidity_profile(bars)
  cat(sprintf("  Best liquidity hour: %d  Worst: %d\n",
              ilp$best_hour-1, ilp$worst_hour-1))
  cat("  Avg spread by hour (0-5):", round(ilp$by_hour[1:6]*1e4,2), "(bps)\n")

  invisible(list(tr=tr, ti=ti, ac=ac, ilp=ilp))
}

if (interactive()) {
  ms_ext <- run_microstructure_extended_demo()
}

# =============================================================================
# SECTION: HASBROUCK'S INFORMATION SHARE
# =============================================================================
# When an asset trades on multiple venues, Hasbrouck's information share
# attributes price discovery to each venue via VECM variance decomposition.
# Simplified scalar version for two venues.

hasbrouck_info_share <- function(p1, p2) {
  # p1, p2: log price series from two venues
  # Estimate VECM(1) by OLS and compute info shares
  n   <- length(p1)
  dp1 <- diff(p1); dp2 <- diff(p2)
  ec  <- p1[-1] - p2[-1]                 # error-correction term (lagged)
  # Two equations: dp1 ~ alpha1*ec + e1, dp2 ~ alpha2*ec + e2
  alpha1 <- coef(lm(dp1 ~ ec - 1))[[1]]
  alpha2 <- coef(lm(dp2 ~ ec - 1))[[1]]
  # Residuals
  e1 <- dp1 - alpha1 * ec
  e2 <- dp2 - alpha2 * ec
  Sigma <- cov(cbind(e1, e2))
  # Hasbrouck IS (upper/lower bounds simplified to midpoint)
  s11 <- Sigma[1,1]; s22 <- Sigma[2,2]; s12 <- Sigma[1,2]
  # Orthogonalise: midpoint info shares
  psi <- c(-alpha2, alpha1)
  if (sum(psi^2) < 1e-12) return(c(IS1 = 0.5, IS2 = 0.5))
  var_common <- as.numeric(t(psi) %*% Sigma %*% psi)
  G1 <- (psi[1]^2 * s11) / var_common
  G2 <- (psi[2]^2 * s22) / var_common
  tot <- G1 + G2
  c(IS1 = G1 / tot, IS2 = G2 / tot)
}

# =============================================================================
# SECTION: ROLL'S IMPLIED SPREAD COVARIANCE MODEL
# =============================================================================
# Roll (1984): bid-ask bounce induces negative first-order autocovariance.
# Spread estimate: s = 2 * sqrt(max(0, -cov(dp_t, dp_{t-1})))

roll_implied_spread <- function(prices) {
  dp  <- diff(log(prices))
  cov1 <- cov(dp[-1], dp[-length(dp)])
  2 * sqrt(max(0, -cov1))
}

rolling_roll_spread <- function(prices, window = 60) {
  dp  <- diff(log(prices))
  n   <- length(dp)
  out <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    sub <- dp[(i - window + 1):i]
    out[i] <- 2 * sqrt(max(0, -cov(sub[-1], sub[-length(sub)])))
  }
  out
}

# =============================================================================
# SECTION: MARKET IMPACT — SQUARE-ROOT LAW
# =============================================================================
# Empirical finding: price impact ~ sigma * sqrt(Q/V) where Q is order size
# and V is average daily volume. Used to estimate execution cost.

sqrt_law_impact <- function(sigma_daily, order_size, avg_daily_volume,
                            eta = 0.1) {
  # Returns expected price impact as fraction of price
  eta * sigma_daily * sqrt(order_size / avg_daily_volume)
}

# Almgren-Chriss optimal execution: minimize variance + lambda*market_impact
almgren_chriss_schedule <- function(X, T_periods, sigma, eta, lambda = 1e-5) {
  # X: total shares to sell, T_periods: number of periods
  # Returns optimal trading schedule x_t (shares sold each period)
  kappa <- sqrt(lambda * sigma^2 / eta)
  t_seq <- seq(0, T_periods)
  # Hyperbolic sine schedule
  sinh_kT <- sinh(kappa * T_periods)
  if (abs(sinh_kT) < 1e-10) return(rep(X / T_periods, T_periods))
  x_t <- X * sinh(kappa * (T_periods - t_seq)) / sinh(kappa * T_periods)
  # Trading rate: shares sold in each interval
  diff(-x_t)
}

# =============================================================================
# SECTION: TRADE CLASSIFICATION — BULK CLASSIFICATION
# =============================================================================
# Ellis, Michaely, O'Hara (2000) bulk classification:
# Uses quote midpoint and trade price to sign trades en masse.

bulk_classify <- function(trade_prices, bid, ask) {
  mid <- (bid + ask) / 2
  sign_trade <- ifelse(trade_prices > mid, 1L,
                ifelse(trade_prices < mid, -1L, 0L))
  sign_trade
}

# Compute order flow imbalance from signed trades
order_flow_imbalance <- function(signed_trades, window = 20) {
  n <- length(signed_trades)
  ofi <- rep(NA_real_, n)
  for (i in seq(window, n)) {
    sub <- signed_trades[(i - window + 1):i]
    buys  <- sum(sub == 1L)
    sells <- sum(sub == -1L)
    ofi[i] <- (buys - sells) / (buys + sells + 1e-9)
  }
  ofi
}

# =============================================================================
# SECTION: VPIN — Volume-Synchronised PIN
# =============================================================================
# Easley et al. (2012): measure order-flow toxicity via VPIN.
# Partition volume into equal-sized buckets; classify buy/sell volume.

compute_vpin <- function(trade_prices, trade_volumes, bucket_size = NULL) {
  n <- length(trade_prices)
  if (is.null(bucket_size)) bucket_size <- sum(trade_volumes) / 50
  # Assign each trade a direction via tick rule
  direction <- numeric(n)
  direction[1] <- 1
  for (i in 2:n)
    direction[i] <- if (trade_prices[i] > trade_prices[i-1]) 1 else
                    if (trade_prices[i] < trade_prices[i-1]) -1 else direction[i-1]

  buy_vol  <- pmax(direction, 0) * trade_volumes
  sell_vol <- pmax(-direction, 0) * trade_volumes

  # Fill buckets
  cum_vol <- cumsum(trade_volumes)
  n_buckets <- floor(sum(trade_volumes) / bucket_size)
  if (n_buckets < 2) return(NA_real_)
  vpin_vals <- numeric(n_buckets - 1)
  for (b in seq_len(n_buckets - 1)) {
    lo <- cum_vol >= (b-1) * bucket_size
    hi <- cum_vol <  b     * bucket_size
    idx <- which(lo & hi)
    if (length(idx) == 0) next
    vpin_vals[b] <- abs(sum(buy_vol[idx]) - sum(sell_vol[idx])) /
                    (sum(trade_volumes[idx]) + 1e-10)
  }
  mean(vpin_vals, na.rm = TRUE)
}

# =============================================================================
# SECTION: INTRADAY SEASONALITY REMOVAL (Z-SCORE BY BUCKET)
# =============================================================================
# Crypto trades 24/7 but volume and volatility are still periodic.
# Normalise metrics by their intraday distribution for each time bucket.

remove_intraday_seasonality <- function(x, bucket_ids) {
  # x: numeric vector, bucket_ids: integer time-of-day labels
  out <- numeric(length(x))
  for (b in unique(bucket_ids)) {
    idx  <- bucket_ids == b
    mu   <- mean(x[idx], na.rm = TRUE)
    sig  <- sd(x[idx], na.rm = TRUE)
    out[idx] <- if (sig > 1e-9) (x[idx] - mu) / sig else 0
  }
  out
}

# =============================================================================
# SECTION: FINAL DEMO
# =============================================================================

run_microstructure_final_demo <- function() {
  set.seed(55)
  n <- 500
  p <- cumsum(c(100, rnorm(n-1, 0, 0.5)))
  p <- pmax(p, 1)

  cat("--- Roll Implied Spread ---\n")
  cat("Estimated spread:", round(roll_implied_spread(p), 5), "\n")

  rrs <- rolling_roll_spread(p, 60)
  cat("Rolling spread (mean):", round(mean(rrs, na.rm=TRUE), 5), "\n")

  cat("\n--- Hasbrouck Info Share (two noisy venues) ---\n")
  p2 <- p + rnorm(n, 0, 0.1)
  is <- hasbrouck_info_share(log(p), log(p2))
  cat("IS venue 1:", round(is["IS1"], 3),
      "  IS venue 2:", round(is["IS2"], 3), "\n")

  cat("\n--- Square-Root Impact ---\n")
  imp <- sqrt_law_impact(sigma_daily = 0.02, order_size = 1000,
                         avg_daily_volume = 50000)
  cat("Expected impact:", round(imp * 100, 4), "%\n")

  cat("\n--- Almgren-Chriss Schedule ---\n")
  sched <- almgren_chriss_schedule(X = 10000, T_periods = 10,
                                   sigma = 0.02, eta = 0.1)
  cat("Trade schedule:", round(sched, 1), "\n")

  cat("\n--- VPIN ---\n")
  vols <- abs(rnorm(n, 100, 20))
  vpin <- compute_vpin(p, vols)
  cat("VPIN:", round(vpin, 4), "\n")

  invisible(list(roll_spread = rrs, info_share = is, vpin = vpin))
}

if (interactive()) {
  ms_final <- run_microstructure_final_demo()
}

# =============================================================================
# SECTION: PRICE IMPACT REGRESSION
# =============================================================================
# Regress signed order flow on subsequent price changes to estimate
# lambda (Kyle's lambda) and alpha (temporary vs. permanent impact).

price_impact_regression <- function(trade_sizes, price_changes) {
  # trade_sizes: signed (positive=buy, negative=sell)
  # price_changes: dp_t = lambda * x_t + noise
  n  <- min(length(trade_sizes), length(price_changes))
  df <- data.frame(dp = price_changes[seq_len(n)],
                   x  = trade_sizes[seq_len(n)])
  fit <- lm(dp ~ x - 1, data = df)
  list(lambda   = coef(fit)["x"],
       r_squared = summary(fit)$r.squared,
       residuals = resid(fit))
}

# =============================================================================
# SECTION: MICROSTRUCTURE NOISE DECOMPOSITION
# =============================================================================
# Bandi-Russell (2006): decompose total return variance into
# noise variance and true price variance.

bandi_russell_noise <- function(high_freq_rets, low_freq_rets) {
  # Noise variance = 0.5 * (HF variance - LF variance)
  var_hf <- var(high_freq_rets)
  var_lf <- var(low_freq_rets)
  noise_var <- 0.5 * max(var_hf - var_lf, 0)
  true_var  <- pmax(var_lf, 0)
  list(noise_var = noise_var, true_var = true_var,
       noise_ratio = noise_var / (noise_var + true_var + 1e-10))
}

# =============================================================================
# SECTION: LATENCY ARBITRAGE DETECTION
# =============================================================================
# Fast traders exploit stale quotes; detect by measuring quote update speed
# relative to trade arrival at stale prices.

latency_arb_score <- function(trade_prices, quote_midpoints, timestamps) {
  n <- min(length(trade_prices), length(quote_midpoints), length(timestamps))
  # Trades at prices far from current mid suggest latency exploitation
  deviations <- abs(trade_prices[seq_len(n)] - quote_midpoints[seq_len(n)])
  # Normalize by volatility proxy
  sig <- sd(diff(quote_midpoints[seq_len(n)]), na.rm=TRUE)
  if (sig < 1e-9) return(rep(0, n))
  deviations / sig
}

# =============================================================================
# SECTION: EXTENDED MICROSTRUCTURE UTILITY FUNCTIONS
# =============================================================================

# Realized spread: profit to the market maker per trade
realized_spread <- function(trade_prices, trade_signs, mid_prices, delay = 5) {
  n <- length(trade_prices)
  if (n <= delay) return(NA_real_)
  rs <- trade_signs[1:(n-delay)] * (trade_prices[1:(n-delay)] -
                                      mid_prices[(1+delay):n])
  mean(2 * rs, na.rm = TRUE)
}

# Price reversal after large trades: measure of temporary impact
price_reversal <- function(prices, trade_signs, trade_sizes, horizon = 10,
                            size_thresh = NULL) {
  n <- length(prices)
  if (is.null(size_thresh)) size_thresh <- median(abs(trade_sizes))
  large <- abs(trade_sizes) > size_thresh
  if (sum(large) == 0) return(NA_real_)
  idx  <- which(large)
  idx  <- idx[idx + horizon <= n]
  sign_t <- trade_signs[idx]
  dp     <- prices[idx + horizon] - prices[idx]
  # Reversal: price moves back against initial direction
  -mean(sign_t * dp, na.rm = TRUE)
}

if (interactive()) {
  set.seed(66)
  n  <- 300
  p  <- cumsum(c(50000, rnorm(n-1, 0, 50)))
  ts <- sign(rnorm(n))
  sz <- abs(rnorm(n, 100, 30))
  cat("Realized spread:", round(realized_spread(p, ts, p), 4), "\n")
  cat("Price reversal:", round(price_reversal(p, ts, sz), 4), "\n")
  pir <- price_impact_regression(ts * sz, diff(c(p[1], p)))
  cat("Kyle lambda:", round(pir$lambda, 6), "\n")
}
