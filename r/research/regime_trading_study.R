# =============================================================================
# regime_trading_study.R
# Regime-Based Trading Research for Crypto
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: The same strategy can have Sharpe > 2 in one macro
# regime and Sharpe < 0 in another. Risk-on regimes (falling VIX, rising DXY)
# are historically favourable for crypto momentum; risk-off regimes reward
# defensive sizing or cash. Identifying the current regime and conditioning
# signal size on it is a high-value alpha source that is hard to arbitrage.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

roll_mean <- function(x, w) {
  n <- length(x); out <- rep(NA,n)
  for (i in w:n) out[i] <- mean(x[(i-w+1):i], na.rm=TRUE)
  out
}

roll_sd <- function(x, w) {
  n <- length(x); out <- rep(NA,n)
  for (i in w:n) out[i] <- sd(x[(i-w+1):i], na.rm=TRUE)
  out
}

sharpe_ratio <- function(rets, ann=252) {
  mu <- mean(rets, na.rm=TRUE); sg <- sd(rets, na.rm=TRUE)
  if (is.na(sg)||sg<1e-12) return(NA_real_)
  mu/sg*sqrt(ann)
}

max_drawdown <- function(eq) {
  pk <- cummax(eq); min((eq-pk)/pk, na.rm=TRUE)
}

# ---------------------------------------------------------------------------
# 2. MACRO REGIME IDENTIFICATION
# ---------------------------------------------------------------------------
# Construct regime from proxies:
#   - VIX proxy: rolling vol of crypto
#   - DXY proxy: rolling correlation of crypto with USD index
#   - Yield curve proxy: simulated spread (10Y - 2Y)
#
# Regime 1: Risk-on  (low vol, USD weak, flat/steepening curve)
# Regime 2: Risk-off (high vol, USD strong, inverting curve)

simulate_macro_proxies <- function(T_=1000L, seed=42L) {
  set.seed(seed)
  # VIX proxy: stochastic vol (GARCH-like)
  vix <- numeric(T_); vix[1] <- 0.20
  for (t in 2:T_) vix[t] <- 0.94*vix[t-1] + 0.06*(0.20 + 0.10*abs(rnorm(1)))
  # DXY proxy: random walk with persistence
  dxy <- cumsum(rnorm(T_, 0, 0.005))
  # Yield spread: slow mean-reverting
  ys  <- numeric(T_); ys[1] <- 0.01
  for (t in 2:T_) ys[t] <- 0.98*ys[t-1] + 0.02*0.01 + rnorm(1,0,0.001)

  list(vix=vix, dxy=dxy, yield_spread=ys)
}

classify_macro_regime <- function(vix, dxy, yield_spread,
                                   vix_window=21L, dxy_window=21L) {
  T_  <- length(vix)
  # Standardise each proxy
  std_vix <- (vix - roll_mean(vix, vix_window)) / pmax(roll_sd(vix, vix_window), 1e-6)
  std_dxy <- (dxy - roll_mean(dxy, dxy_window)) / pmax(roll_sd(dxy, dxy_window), 1e-6)
  std_ys  <- (yield_spread - roll_mean(yield_spread, 21L)) /
    pmax(roll_sd(yield_spread, 21L), 1e-6)

  # Composite risk score: high = risk-off
  risk_score <- 0.5*std_vix + 0.3*std_dxy - 0.2*std_ys
  risk_score[is.na(risk_score)] <- 0

  # 3 regimes: risk-on, neutral, risk-off
  q33 <- quantile(risk_score, 0.33, na.rm=TRUE)
  q67 <- quantile(risk_score, 0.67, na.rm=TRUE)
  regime <- ifelse(risk_score < q33, 1L,   # risk-on
                    ifelse(risk_score < q67, 2L, 3L))  # neutral, risk-off
  list(regime=regime, risk_score=risk_score,
       std_vix=std_vix, std_dxy=std_dxy, std_ys=std_ys)
}

# ---------------------------------------------------------------------------
# 3. CRYPTO PERFORMANCE BY MACRO REGIME
# ---------------------------------------------------------------------------

crypto_by_regime <- function(crypto_returns, regime, regime_labels=NULL) {
  if (is.null(regime_labels)) regime_labels <- c("Risk-On","Neutral","Risk-Off")
  regimes <- sort(unique(regime[!is.na(regime)]))
  results <- lapply(regimes, function(r) {
    mask <- regime == r & !is.na(crypto_returns)
    r_r  <- crypto_returns[mask]
    if (length(r_r) < 5) return(NULL)
    eq   <- cumprod(1 + r_r)
    lbl  <- if (r <= length(regime_labels)) regime_labels[r] else paste("Regime",r)
    data.frame(
      regime    = lbl,
      n_bars    = length(r_r),
      pct_bars  = length(r_r) / length(regime[!is.na(regime)]),
      mean_ann  = mean(r_r, na.rm=TRUE) * 252,
      vol_ann   = sd(r_r, na.rm=TRUE) * sqrt(252),
      sharpe    = sharpe_ratio(r_r),
      max_dd    = max_drawdown(eq),
      hit_rate  = mean(r_r > 0, na.rm=TRUE)
    )
  })
  do.call(rbind, Filter(Negate(is.null), results))
}

# ---------------------------------------------------------------------------
# 4. BH SIGNAL EFFECTIVENESS BY REGIME
# ---------------------------------------------------------------------------
# Does the Breakout / Momentum / BH signal work equally in all regimes?

signal_regime_effectiveness <- function(signal, forward_returns, regime) {
  regimes <- sort(unique(regime[!is.na(regime)]))
  results <- lapply(regimes, function(r) {
    mask  <- regime == r & !is.na(signal) & !is.na(forward_returns)
    s_r   <- signal[mask]; fr_r <- forward_returns[mask]
    if (length(s_r) < 10) return(NULL)
    ic_val <- cor(rank(s_r), rank(fr_r))
    pos    <- sign(s_r)
    rets_r <- pos * fr_r
    data.frame(regime=r, n=length(s_r), IC=ic_val,
               signal_sharpe=sharpe_ratio(rets_r),
               hit_rate=mean(rets_r > 0, na.rm=TRUE))
  })
  do.call(rbind, Filter(Negate(is.null), results))
}

# ---------------------------------------------------------------------------
# 5. OPTIMAL SIZING MULTIPLIER PER REGIME
# ---------------------------------------------------------------------------
# Given a base signal, find the Kelly-optimal scaling per regime.

regime_optimal_multiplier <- function(base_returns_by_regime) {
  results <- lapply(names(base_returns_by_regime), function(nm) {
    r <- base_returns_by_regime[[nm]]
    mu <- mean(r, na.rm=TRUE); s2 <- var(r, na.rm=TRUE)
    kelly <- if (s2 > 1e-12) mu/s2 else 0
    data.frame(regime=nm, mu_ann=mu*252, vol_ann=sqrt(s2)*sqrt(252),
               kelly_mult=clip(kelly*252, 0, 3))
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 6. REGIME TRANSITION PROBABILITIES (MARKOV)
# ---------------------------------------------------------------------------

estimate_transition_matrix <- function(regime_series) {
  regimes <- sort(unique(regime_series[!is.na(regime_series)]))
  K       <- length(regimes)
  trans   <- matrix(0, K, K)
  n       <- length(regime_series)
  for (t in 2:n) {
    if (is.na(regime_series[t]) || is.na(regime_series[t-1])) next
    from <- match(regime_series[t-1], regimes)
    to   <- match(regime_series[t],   regimes)
    trans[from, to] <- trans[from, to] + 1
  }
  # Normalise rows
  P <- (trans + 0.5) / (rowSums(trans + 0.5))
  rownames(P) <- colnames(P) <- paste0("R",regimes)
  P
}

#' Expected duration in each regime (geometric distribution)
regime_durations <- function(P) {
  diag_p <- diag(P)
  1 / (1 - diag_p)   # E[duration] = 1/(1-p_ii)
}

#' Stationary distribution of the Markov chain
stationary_dist <- function(P) {
  K    <- nrow(P)
  # Solve pi * P = pi, sum(pi) = 1
  A    <- t(P) - diag(K)
  A[K,] <- 1
  b    <- c(rep(0, K-1), 1)
  tryCatch(solve(A, b), error=function(e) rep(1/K,K))
}

# ---------------------------------------------------------------------------
# 7. IS VS OOS REGIME TIMING
# ---------------------------------------------------------------------------
# Is knowing the regime at all useful OOS? Compare:
#   - Oracle: uses true future regime (infeasible)
#   - Regime-conditioned: uses current regime + Markov forecast
#   - Equal-weight: ignores regime

run_regime_strategy <- function(crypto_returns, regime,
                                  sizing_table,   # named: regime -> multiplier
                                  base_signal = NULL) {
  n <- length(crypto_returns)
  if (is.null(base_signal)) base_signal <- rep(1, n)  # long-only

  pos   <- numeric(n)
  rets  <- numeric(n)
  for (t in seq_len(n)) {
    r <- regime[t]
    mult <- sizing_table[[as.character(r)]]
    if (is.null(mult)) mult <- 1.0
    pos[t]  <- base_signal[t] * mult
    rets[t] <- pos[t] * crypto_returns[t]
  }
  equity <- cumprod(1 + rets)
  list(rets=rets, equity=equity,
       sharpe=sharpe_ratio(rets), max_dd=max_drawdown(equity))
}

regime_timing_backtest <- function(crypto_returns, regime,
                                    is_frac = 0.6) {
  T_    <- length(crypto_returns)
  is_n  <- as.integer(T_ * is_frac)
  oos_n <- T_ - is_n

  # IS: calibrate optimal multipliers
  cr_is <- crypto_by_regime(crypto_returns[1:is_n], regime[1:is_n])
  sizing_is <- setNames(as.list(clip(cr_is$sharpe / max(abs(cr_is$sharpe),0.1), 0, 2)),
                         as.character(1:nrow(cr_is)))

  # OOS: apply
  oos_reg <- run_regime_strategy(crypto_returns[(is_n+1):T_],
                                   regime[(is_n+1):T_], sizing_is)
  oos_ew  <- run_regime_strategy(crypto_returns[(is_n+1):T_],
                                   regime[(is_n+1):T_],
                                   setNames(as.list(rep(1,nrow(cr_is))),
                                             as.character(1:nrow(cr_is))))

  list(
    is_stats  = cr_is,
    sizing    = sizing_is,
    oos_timed = oos_reg,
    oos_ew    = oos_ew,
    IS_sharpe_timed = run_regime_strategy(crypto_returns[1:is_n],
                                           regime[1:is_n], sizing_is)$sharpe,
    IS_sharpe_ew    = sharpe_ratio(crypto_returns[1:is_n])
  )
}

# ---------------------------------------------------------------------------
# 8. COMBINING REGIME WITH ON-CHAIN SIGNALS
# ---------------------------------------------------------------------------
# On-chain signals: NVT (network value to transactions), MVRV (market value
# to realised value), exchange netflows.
# Here we simulate proxies.

simulate_onchain <- function(T_=1000L, seed=42L) {
  set.seed(seed)
  # NVT: high = overvalued, low = undervalued
  nvt  <- exp(cumsum(rnorm(T_, 0, 0.01))) * 50
  # MVRV z-score: > 2 = overvalued, < 0 = undervalued
  mvrv <- cumsum(rnorm(T_, 0, 0.05))
  # Exchange netflows: negative = BTC leaving exchanges (bullish)
  netflow <- rnorm(T_, -0.001, 0.02)
  list(nvt=nvt, mvrv=mvrv, netflow=netflow)
}

#' Composite on-chain signal: normalised [-1,1]
onchain_signal <- function(onchain, window=30L) {
  # NVT z-score (high NVT = bearish)
  nvt_z  <- (onchain$nvt - roll_mean(onchain$nvt,window)) /
    pmax(roll_sd(onchain$nvt, window), 1e-6)
  # MVRV (high MVRV = overvalued = bearish)
  mvrv_z <- (onchain$mvrv - roll_mean(onchain$mvrv, window)) /
    pmax(roll_sd(onchain$mvrv, window), 1e-6)
  # Netflow (negative netflow = bullish)
  nf_z   <- -(onchain$netflow - roll_mean(onchain$netflow, window)) /
    pmax(roll_sd(onchain$netflow, window), 1e-6)
  # Composite: simple average, invert NVT and MVRV
  tanh((-nvt_z * 0.3 - mvrv_z * 0.3 + nf_z * 0.4))
}

#' Combine regime signal with on-chain signal
regime_onchain_signal <- function(regime, oc_signal,
                                   regime_weights = c(0.5, 0.0, -0.5),
                                   oc_weight = 0.5) {
  n <- length(regime)
  reg_part <- numeric(n)
  for (t in seq_len(n)) {
    r <- regime[t]; if (is.na(r)) next
    reg_part[t] <- if (r <= length(regime_weights)) regime_weights[r] else 0
  }
  combo <- (1 - oc_weight) * reg_part + oc_weight * oc_signal
  clip(combo, -1, 1)
}

# ---------------------------------------------------------------------------
# 9. LIVE REGIME DASHBOARD
# ---------------------------------------------------------------------------
# Report current regime state and forward probabilities.

regime_dashboard <- function(macro_proxies, P, current_t = NULL) {
  regime_info <- classify_macro_regime(macro_proxies$vix,
                                        macro_proxies$dxy,
                                        macro_proxies$yield_spread)
  if (is.null(current_t)) current_t <- length(regime_info$regime)
  current_regime <- regime_info$regime[current_t]
  current_score  <- regime_info$risk_score[current_t]

  # 1-step ahead forecast from Markov
  if (!is.na(current_regime) && current_regime <= nrow(P)) {
    fwd_probs <- P[current_regime, ]
  } else {
    fwd_probs <- rep(1/nrow(P), nrow(P))
  }

  list(
    current_regime  = current_regime,
    risk_score      = current_score,
    regime_label    = c("Risk-On","Neutral","Risk-Off")[current_regime],
    fwd_probs       = fwd_probs,
    std_vix         = regime_info$std_vix[current_t],
    std_dxy         = regime_info$std_dxy[current_t]
  )
}

# ---------------------------------------------------------------------------
# 10. REGIME FORECAST ACCURACY
# ---------------------------------------------------------------------------

regime_forecast_accuracy <- function(true_regime, predicted_regime) {
  n     <- min(length(true_regime), length(predicted_regime))
  same  <- true_regime[1:n] == predicted_regime[1:n]
  same  <- same[!is.na(same)]
  data.frame(
    accuracy  = mean(same),
    n_correct = sum(same),
    n_total   = length(same)
  )
}

# ---------------------------------------------------------------------------
# 11. REGIME-ADJUSTED POSITION LIMITS
# ---------------------------------------------------------------------------
# Risk-off: reduce max position; risk-on: allow full Kelly.

regime_position_limit <- function(base_position, regime,
                                   limits = c(1.0, 0.6, 0.3)) {
  n <- length(base_position)
  out <- numeric(n)
  for (t in seq_len(n)) {
    r   <- regime[t]; if (is.na(r)) r <- 2L
    lim <- if (r <= length(limits)) limits[r] else 0.5
    out[t] <- clip(base_position[t], -lim, lim)
  }
  out
}

# ---------------------------------------------------------------------------
# 12. MAIN DEMO
# ---------------------------------------------------------------------------

run_regime_study_demo <- function() {
  cat("=== Regime-Based Trading Study Demo ===\n\n")
  set.seed(42)
  T_ <- 1000L

  # Simulate macro proxies
  macro <- simulate_macro_proxies(T_)
  reg_info <- classify_macro_regime(macro$vix, macro$dxy, macro$yield_spread)
  regime <- reg_info$regime
  cat(sprintf("Regime distribution: %s\n",
              paste(names(table(regime)), table(regime), sep="=", collapse="  ")))

  # Simulate crypto returns (correlated with regime)
  crypto_ret <- rnorm(T_, 0, 0.02) +
    ifelse(regime==1L, 0.002, ifelse(regime==3L, -0.003, 0))

  cat("\n--- 1. Crypto Performance by Macro Regime ---\n")
  perf <- crypto_by_regime(crypto_ret, regime)
  print(perf[, c("regime","n_bars","mean_ann","vol_ann","sharpe","max_dd")])

  cat("\n--- 2. Regime Transition Matrix ---\n")
  P <- estimate_transition_matrix(regime)
  print(round(P, 3))

  cat("\n--- 3. Expected Regime Durations ---\n")
  durations <- regime_durations(P)
  cat("  Durations (bars):", round(durations, 1), "\n")
  stat_dist <- stationary_dist(P)
  cat("  Stationary distribution:", round(stat_dist, 3), "\n")

  cat("\n--- 4. BH Signal by Regime ---\n")
  # Momentum signal
  momentum <- tanh(roll_mean(crypto_ret, 10L) / pmax(roll_sd(crypto_ret, 10L), 1e-6))
  fwd_ret   <- c(crypto_ret[-1], NA)
  sig_eff   <- signal_regime_effectiveness(momentum, fwd_ret, regime)
  print(sig_eff)

  cat("\n--- 5. Optimal Sizing Multipliers per Regime ---\n")
  by_regime_rets <- split(momentum * fwd_ret, regime)
  # Remove NA
  by_regime_rets <- lapply(by_regime_rets, function(r) r[!is.na(r)])
  opt_mult <- regime_optimal_multiplier(by_regime_rets)
  print(opt_mult)

  cat("\n--- 6. IS/OOS Regime Timing Backtest ---\n")
  bt <- regime_timing_backtest(crypto_ret, regime, is_frac=0.6)
  cat(sprintf("  IS  | Timed Sharpe=%.3f  EW Sharpe=%.3f\n",
              bt$IS_sharpe_timed, bt$IS_sharpe_ew))
  cat(sprintf("  OOS | Timed Sharpe=%.3f  EW Sharpe=%.3f\n",
              bt$oos_timed$sharpe, bt$oos_ew$sharpe))
  cat("  Is regime timing worth it?",
      ifelse(bt$oos_timed$sharpe > bt$oos_ew$sharpe, "YES (OOS improvement)", "NO (OOS degradation)"), "\n")

  cat("\n--- 7. On-Chain Signal Integration ---\n")
  oc    <- simulate_onchain(T_)
  oc_sig <- onchain_signal(oc, window=30L)
  combo  <- regime_onchain_signal(regime, oc_sig)
  combo_rets <- combo * fwd_ret
  combo_rets[is.na(combo_rets)] <- 0
  cat(sprintf("  Combined signal Sharpe: %.3f\n", sharpe_ratio(combo_rets)))

  # Vs pure regime or pure on-chain
  pure_oc_rets  <- oc_sig * fwd_ret; pure_oc_rets[is.na(pure_oc_rets)] <- 0
  pure_reg_rets <- (ifelse(regime==1,1,ifelse(regime==3,-1,0))) * fwd_ret
  pure_reg_rets[is.na(pure_reg_rets)] <- 0
  cat(sprintf("  Pure regime Sharpe: %.3f  |  Pure on-chain Sharpe: %.3f\n",
              sharpe_ratio(pure_reg_rets), sharpe_ratio(pure_oc_rets)))

  cat("\n--- 8. Regime Position Limits ---\n")
  base_pos  <- momentum
  lim_pos   <- regime_position_limit(base_pos, regime, limits=c(1.0,0.6,0.3))
  cat(sprintf("  Base mean abs pos: %.3f  |  Limited: %.3f\n",
              mean(abs(base_pos), na.rm=TRUE), mean(abs(lim_pos), na.rm=TRUE)))
  lim_rets  <- lim_pos * fwd_ret; lim_rets[is.na(lim_rets)] <- 0
  cat(sprintf("  Limited strategy Sharpe: %.3f  |  Base Sharpe: %.3f\n",
              sharpe_ratio(lim_rets), sharpe_ratio(momentum*fwd_ret)))

  cat("\n--- 9. Live Regime Dashboard ---\n")
  dash <- regime_dashboard(macro, P)
  cat(sprintf("  Current regime: %s (score=%.3f)\n",
              dash$regime_label, dash$risk_score))
  cat(sprintf("  Forward probabilities: %s\n",
              paste(names(dash$fwd_probs), round(dash$fwd_probs, 3), sep="=", collapse="  ")))
  cat(sprintf("  Std VIX: %.3f  |  Std DXY: %.3f\n",
              dash$std_vix, dash$std_dxy))

  cat("\nDone.\n")
  invisible(list(perf=perf, P=P, bt=bt, combo=combo, dash=dash))
}

if (interactive()) {
  regime_results <- run_regime_study_demo()
}

# ---------------------------------------------------------------------------
# 13. HMRC: HIDDEN MARKOV REGIME CLASSIFIER
# ---------------------------------------------------------------------------
# 2-state HMM on a macro composite index; states align with risk-on/off.
# Use Viterbi for state decoding; Baum-Welch for parameter estimation.

hmm2_init <- function(obs) {
  list(pi=c(0.6,0.4),
       A=matrix(c(0.95,0.05,0.10,0.90),2,2,byrow=TRUE),
       mu=c(quantile(obs,0.3,na.rm=TRUE), quantile(obs,0.7,na.rm=TRUE)),
       sigma=rep(sd(obs,na.rm=TRUE)/2,2), K=2L)
}

hmm2_forward <- function(obs, model) {
  T_ <- length(obs); K <- 2L
  B  <- matrix(0,T_,K)
  for(k in 1:K) B[,k] <- dnorm(obs, model$mu[k], pmax(model$sigma[k],1e-6))
  B[B<1e-300] <- 1e-300
  alpha <- matrix(0,T_,K); sc <- numeric(T_)
  alpha[1,] <- model$pi*B[1,]; sc[1] <- sum(alpha[1,])
  alpha[1,] <- alpha[1,]/sc[1]
  for(t in 2:T_) {
    for(k in 1:K) alpha[t,k] <- sum(alpha[t-1,]*model$A[,k])*B[t,k]
    sc[t] <- sum(alpha[t,]); if(sc[t]<1e-300) sc[t] <- 1e-300
    alpha[t,] <- alpha[t,]/sc[t]
  }
  list(alpha=alpha,scale=sc,loglik=sum(log(sc)))
}

hmm2_viterbi <- function(obs, model) {
  T_ <- length(obs); K <- 2L
  B  <- matrix(0,T_,K)
  for(k in 1:K) B[,k] <- log(pmax(dnorm(obs,model$mu[k],pmax(model$sigma[k],1e-6)),1e-300))
  delta <- matrix(-Inf,T_,K); psi <- matrix(0L,T_,K)
  delta[1,] <- log(pmax(model$pi,1e-300)) + B[1,]
  for(t in 2:T_) {
    for(j in 1:K) {
      vals <- delta[t-1,] + log(pmax(model$A[,j],1e-300))
      psi[t,j] <- which.max(vals); delta[t,j] <- max(vals)+B[t,j]
    }
  }
  path <- integer(T_); path[T_] <- which.max(delta[T_,])
  for(t in (T_-1):1) path[t] <- psi[t+1,path[t+1]]
  path
}

hmm2_fit <- function(obs, max_iter=30L) {
  model <- hmm2_init(obs); T_ <- length(obs); K <- 2L
  for(iter in seq_len(max_iter)) {
    fwd <- hmm2_forward(obs, model)
    # Simple M-step via gamma (no full E-step for brevity)
    path <- hmm2_viterbi(obs, model)
    for(k in 1:K) {
      idx <- path==k; if(sum(idx)<1) next
      model$mu[k]    <- mean(obs[idx])
      model$sigma[k] <- max(sd(obs[idx]),1e-4)
    }
    # Update transition matrix
    for(i in 1:K) for(j in 1:K) {
      model$A[i,j] <- sum(path[-T_]==i & path[-1]==j) + 0.5
    }
    model$A <- model$A / rowSums(model$A)
  }
  list(model=model, states=hmm2_viterbi(obs, model))
}

# ---------------------------------------------------------------------------
# 14. REGIME PERFORMANCE ATTRIBUTION
# ---------------------------------------------------------------------------

regime_performance_attribution <- function(returns, regime, regime_names=c("R-On","Neutral","R-Off")) {
  regimes <- sort(unique(regime[!is.na(regime)]))
  equity  <- cumprod(1+returns)
  results <- lapply(regimes, function(r) {
    mask <- regime==r & !is.na(returns)
    r_r  <- returns[mask]
    lbl  <- if(r<=length(regime_names)) regime_names[r] else paste("R",r)
    data.frame(regime=lbl, contribution=sum(r_r,na.rm=TRUE),
               n=length(r_r), pct_time=length(r_r)/length(returns),
               avg_return=mean(r_r,na.rm=TRUE)*252,
               vol=sd(r_r,na.rm=TRUE)*sqrt(252))
  })
  result <- do.call(rbind,results)
  result$pct_contribution <- result$contribution / max(sum(returns,na.rm=TRUE),1e-8)
  result
}

# ---------------------------------------------------------------------------
# 15. EXTENDED REGIME DEMO
# ---------------------------------------------------------------------------

run_regime_extended_demo <- function() {
  cat("=== Regime Study Extended Demo ===\n\n")
  set.seed(77); T_ <- 600L

  macro <- simulate_macro_proxies(T_)
  ri    <- classify_macro_regime(macro$vix, macro$dxy, macro$yield_spread)
  regime <- ri$regime
  crypto_ret <- rnorm(T_,0,0.02) + ifelse(regime==1L,0.003,ifelse(regime==3L,-0.003,0))

  cat("--- HMM2 Regime Classifier ---\n")
  hmm_res <- hmm2_fit(ri$risk_score[!is.na(ri$risk_score)])
  cat(sprintf("  HMM2 mu: %.4f, %.4f  |  States: %s\n",
              hmm_res$model$mu[1], hmm_res$model$mu[2],
              paste(table(hmm_res$states), collapse=",")))

  cat("\n--- Regime Performance Attribution ---\n")
  rpa <- regime_performance_attribution(crypto_ret, regime)
  print(rpa[,c("regime","avg_return","vol","pct_time","pct_contribution")])

  cat("\n--- HMM vs Rule-Based Regime Accuracy ---\n")
  # Map HMM states to rule-based regimes (1=low risk score = risk-on)
  valid_idx <- which(!is.na(ri$risk_score))
  hmm_mapped <- ifelse(hmm_res$states == which.min(hmm_res$model$mu), 1L, 3L)
  rule_states <- regime[valid_idx]
  agree <- mean(hmm_mapped == rule_states | hmm_mapped == 2L, na.rm=TRUE)
  cat(sprintf("  HMM-rule agreement: %.1f%%\n", agree*100))

  cat("\n--- Regime Forecast 3-step ahead ---\n")
  P <- estimate_transition_matrix(regime)
  curr <- tail(regime[!is.na(regime)],1)
  p3   <- P %^% 3  # 3-step matrix power (approx via repeated mult)
  p3   <- P %*% P %*% P
  cat(sprintf("  From regime %d, 3-step probs: %s\n",
              curr, paste(round(p3[curr,],3),collapse=",")))

  invisible(list(hmm=hmm_res, rpa=rpa))
}

if (interactive()) {
  regime_ext <- run_regime_extended_demo()
}

# =============================================================================
# SECTION: THRESHOLD REGIME MODEL (TAR / SETAR)
# =============================================================================
# Self-Exciting Threshold Autoregression: dynamics switch when a threshold
# variable crosses a boundary.  Common in crypto: VIX proxy > 30 = high vol regime.

tar_fit <- function(y, threshold_var, thresh) {
  # Two-regime TAR: y_t = a1 + b1*y_{t-1} + e  if threshold_{t-1} <= thresh
  #                 y_t = a2 + b2*y_{t-1} + e  if threshold_{t-1} >  thresh
  n    <- length(y)
  reg1 <- threshold_var[-n] <= thresh
  reg2 <- !reg1
  # Regime 1
  y1   <- y[-1][reg1]; x1 <- y[-n][reg1]
  fit1 <- if (length(y1) > 2) lm(y1 ~ x1) else NULL
  # Regime 2
  y2   <- y[-1][reg2]; x2 <- y[-n][reg2]
  fit2 <- if (length(y2) > 2) lm(y2 ~ x2) else NULL
  list(
    regime1_coef = if (!is.null(fit1)) coef(fit1) else c(NA,NA),
    regime2_coef = if (!is.null(fit2)) coef(fit2) else c(NA,NA),
    n_reg1 = sum(reg1), n_reg2 = sum(reg2),
    threshold = thresh
  )
}

scan_threshold <- function(y, threshold_var, n_thresh = 20) {
  thresh_grid <- quantile(threshold_var, seq(0.1, 0.9, length.out = n_thresh))
  rss_vec <- sapply(thresh_grid, function(th) {
    n <- length(y)
    reg1 <- threshold_var[-n] <= th
    reg2 <- !reg1
    rss  <- 0
    for (idx in list(which(reg1), which(reg2))) {
      if (length(idx) < 3) next
      yi <- y[-1][idx]; xi <- y[-n][idx]
      fit <- lm(yi ~ xi)
      rss <- rss + sum(resid(fit)^2)
    }
    rss
  })
  best_thresh <- thresh_grid[which.min(rss_vec)]
  list(best_thresh = best_thresh, rss = rss_vec, thresh_grid = thresh_grid)
}

# =============================================================================
# SECTION: REGIME-CONDITIONAL KELLY CRITERION
# =============================================================================
# Apply full Kelly in bull regimes, fractional Kelly in bear regimes.

regime_kelly <- function(mu_by_regime, sigma_by_regime, regime_labels,
                          kelly_fraction_bear = 0.25) {
  # mu_by_regime, sigma_by_regime: named vectors by regime label
  # Returns Kelly fraction for each regime
  kelly_full <- mu_by_regime / (sigma_by_regime^2 + 1e-10)
  kelly_adj  <- kelly_full
  bear_regs  <- grep("bear|risk_off|low|3", names(kelly_full), value=TRUE)
  for (br in bear_regs)
    if (br %in% names(kelly_adj))
      kelly_adj[br] <- kelly_full[br] * kelly_fraction_bear
  pmax(kelly_adj, 0)
}

# =============================================================================
# SECTION: REGIME PERSISTENCE AND TRANSITION TIMING
# =============================================================================
# Measure how long the system stays in each regime and model the transition
# waiting time with a geometric (memoryless) distribution.

regime_persistence <- function(regime_sequence) {
  runs <- rle(regime_sequence)
  durations_by_regime <- tapply(runs$lengths, runs$values, identity)
  list(
    mean_duration = sapply(durations_by_regime, mean),
    median_duration = sapply(durations_by_regime, median),
    max_duration  = sapply(durations_by_regime, max),
    geometric_p   = sapply(durations_by_regime, function(d) 1 / mean(d))
  )
}

# =============================================================================
# SECTION: REGIME-AWARE POSITION SIZING
# =============================================================================
# Scale position size based on both signal strength and current regime.

regime_position_size <- function(signal_strength, regime_label,
                                  regime_multipliers = c(bull=1.0, neutral=0.5, bear=0.25),
                                  max_pos = 1.0) {
  mult <- regime_multipliers[regime_label]
  if (is.na(mult)) mult <- 0.5
  pmin(abs(signal_strength) * mult, max_pos) * sign(signal_strength)
}

# =============================================================================
# SECTION: HMM 3-STATE VITERBI (EXTENSION)
# =============================================================================
# Extend the 2-state HMM to 3 states (bull / neutral / bear) via Viterbi.

viterbi_3state <- function(obs, A, mu_vec, sigma_vec) {
  # A: 3x3 transition matrix, mu_vec/sigma_vec: vectors length 3
  T  <- length(obs)
  K  <- 3
  delta <- matrix(-Inf, K, T)
  psi   <- matrix(0L, K, T)
  # Initialise with uniform start
  for (k in seq_len(K))
    delta[k, 1] <- log(1/K) + dnorm(obs[1], mu_vec[k], sigma_vec[k], log=TRUE)
  # Recursion
  for (t in 2:T) {
    for (k in seq_len(K)) {
      val <- delta[, t-1] + log(A[, k] + 1e-300)
      psi[k, t]   <- which.max(val)
      delta[k, t] <- max(val) + dnorm(obs[t], mu_vec[k], sigma_vec[k], log=TRUE)
    }
  }
  # Backtrack
  path <- integer(T)
  path[T] <- which.max(delta[, T])
  for (t in seq(T-1, 1)) path[t] <- psi[path[t+1], t+1]
  path
}

# =============================================================================
# SECTION: REGIME SIGNAL PORTFOLIO BACKTEST
# =============================================================================
# Run a full backtest that scales strategy allocation based on detected regime.

regime_strategy_backtest <- function(asset_rets, regime_vec,
                                      signal_vec,
                                      regime_alloc = c("1"=1.0, "2"=0.5, "3"=0.0)) {
  # asset_rets: returns, regime_vec: 1/2/3, signal_vec: +/-1
  T  <- length(asset_rets)
  w  <- numeric(T)
  for (t in seq_len(T)) {
    r_label <- as.character(regime_vec[t])
    alloc   <- regime_alloc[r_label]
    if (is.na(alloc)) alloc <- 0.5
    w[t] <- alloc * signal_vec[t]
  }
  port_rets <- w * asset_rets
  cum       <- cumprod(1 + port_rets)
  sharpe    <- mean(port_rets) / (sd(port_rets) + 1e-9) * sqrt(252)
  list(weights = w, port_rets = port_rets, wealth = cum, sharpe = sharpe)
}

# =============================================================================
# SECTION: FINAL REGIME DEMO
# =============================================================================

run_regime_final_demo <- function() {
  set.seed(31)
  T <- 400
  # Simulate macro proxy and regime
  macro <- cumsum(rnorm(T, 0, 0.5))
  regime_true <- ifelse(macro > 5, 3L, ifelse(macro > -5, 2L, 1L))
  asset_rets  <- rnorm(T, 0.001 * (2 - regime_true), 0.02)
  signal_vec  <- sign(c(0, diff(macro)))

  cat("--- TAR Threshold Scan ---\n")
  scan <- scan_threshold(macro, macro)
  cat("Best threshold:", round(scan$best_thresh, 3), "\n")

  cat("\n--- TAR Model ---\n")
  tar <- tar_fit(macro, macro, scan$best_thresh)
  cat("Regime 1 coef:", round(tar$regime1_coef, 4),
      "  n=", tar$n_reg1, "\n")
  cat("Regime 2 coef:", round(tar$regime2_coef, 4),
      "  n=", tar$n_reg2, "\n")

  cat("\n--- Regime Persistence ---\n")
  rp <- regime_persistence(regime_true)
  cat("Mean durations:\n"); print(round(rp$mean_duration, 1))

  cat("\n--- Viterbi 3-State HMM ---\n")
  A <- matrix(c(0.8,0.1,0.1, 0.1,0.8,0.1, 0.1,0.1,0.8), 3, 3, byrow=TRUE)
  vpath <- viterbi_3state(asset_rets, A,
                          mu_vec    = c(0.002, 0.0, -0.002),
                          sigma_vec = c(0.01, 0.02, 0.03))
  cat("Decoded regime counts:", table(vpath), "\n")

  cat("\n--- Regime-Aware Backtest ---\n")
  bt <- regime_strategy_backtest(asset_rets, regime_true, signal_vec)
  cat("Sharpe:", round(bt$sharpe, 3),
      "  Final wealth:", round(tail(bt$wealth, 1), 4), "\n")

  invisible(list(tar=tar, persistence=rp, viterbi=vpath, backtest=bt))
}

if (interactive()) {
  regime_final <- run_regime_final_demo()
}

# =============================================================================
# SECTION: REGIME RISK BUDGETING
# =============================================================================
# Adjust risk budget (position size) based on current regime.
# In uncertain regimes, reduce risk budget to preserve capital.

regime_risk_budget <- function(regime_label, base_budget = 1.0,
                                multipliers = list("1"=1.2, "2"=1.0, "3"=0.4)) {
  mult <- multipliers[[as.character(regime_label)]]
  if (is.null(mult)) mult <- 1.0
  base_budget * mult
}

# =============================================================================
# SECTION: REGIME DETECTION VIA ROLLING STATISTICS
# =============================================================================
# Simple 3-regime detection based on rolling mean and vol of returns:
# Bull: high mean, low vol. Bear: low mean, high vol. Neutral: in between.

rolling_regime_classify <- function(rets, window = 30) {
  n    <- length(rets)
  regime <- rep(2L, n)  # default neutral
  for (t in seq(window, n)) {
    sub  <- rets[(t - window + 1):t]
    mu   <- mean(sub)
    sig  <- sd(sub)
    if      (mu >  0.001 && sig < 0.015) regime[t] <- 1L  # bull
    else if (mu < -0.001 || sig > 0.030) regime[t] <- 3L  # bear
    else                                 regime[t] <- 2L  # neutral
  }
  regime
}

# =============================================================================
# SECTION: REGIME-CONDITIONED VOLATILITY FORECAST
# =============================================================================
# Use regime classification to blend EWMA volatility estimates.
# Each regime has its own EWMA vol estimate.

regime_vol_forecast <- function(rets, regimes, lambda = 0.94) {
  n      <- length(rets)
  vol    <- rep(NA_real_, n)
  vol_by_regime <- c("1"=0.01, "2"=0.02, "3"=0.04)  # initial guess
  for (t in 2:n) {
    r_prev <- rets[t-1]
    reg    <- as.character(regimes[t-1])
    if (!is.na(reg)) {
      v_prev <- vol_by_regime[reg]
      if (is.na(v_prev)) v_prev <- 0.02
      vol_by_regime[reg] <- sqrt(lambda * v_prev^2 + (1-lambda) * r_prev^2)
    }
    vol[t] <- vol_by_regime[as.character(regimes[t])]
  }
  vol
}

# =============================================================================
# SECTION: REGIME SIGNAL COMBINATION (ENSEMBLE)
# =============================================================================
# Combine multiple regime detectors by majority vote.

regime_ensemble <- function(regime_mat) {
  # regime_mat: T x K matrix of regime labels (1/2/3)
  apply(regime_mat, 1, function(row) {
    tt <- table(row)
    as.integer(names(which.max(tt)))
  })
}

# =============================================================================
# SECTION: UTILITY FUNCTIONS AND EXTENDED DEMO
# =============================================================================

regime_sharpe_table <- function(rets, regimes) {
  regs <- sort(unique(regimes))
  out  <- sapply(regs, function(r) {
    ri <- rets[regimes == r]
    if (length(ri) < 5) return(c(n=length(ri), sharpe=NA))
    c(n = length(ri),
      ann_ret = mean(ri) * 252,
      sharpe  = mean(ri) / (sd(ri) + 1e-9) * sqrt(252))
  })
  t(out)
}

run_regime_extension_demo <- function() {
  set.seed(41)
  T <- 350
  rets <- rnorm(T, 0.0002, 0.02)

  cat("--- Rolling Regime Classification ---\n")
  reg <- rolling_regime_classify(rets, window=30)
  cat("Regime distribution:\n"); print(table(reg))

  cat("\n--- Regime Vol Forecast ---\n")
  vf <- regime_vol_forecast(rets, reg)
  cat("Mean vol by regime:\n")
  for (r in 1:3)
    cat(sprintf("  Regime %d: %.4f\n", r, mean(vf[reg==r], na.rm=TRUE)))

  cat("\n--- Regime Sharpe Table ---\n")
  rst <- regime_sharpe_table(rets, reg)
  print(round(rst, 4))

  cat("\n--- Regime Risk Budget ---\n")
  for (r in 1:3)
    cat(sprintf("  Regime %d budget: %.2f\n", r, regime_risk_budget(r)))

  cat("\n--- Regime Ensemble (3 detectors) ---\n")
  reg2 <- rolling_regime_classify(rets + rnorm(T,0,0.005), window=20)
  reg3 <- rolling_regime_classify(rets + rnorm(T,0,0.005), window=40)
  reg_mat <- cbind(reg, reg2, reg3)
  reg_ens <- regime_ensemble(reg_mat)
  cat("Ensemble regime distribution:\n"); print(table(reg_ens))

  invisible(list(regimes=reg, vol_forecast=vf, sharpe_tbl=rst))
}

if (interactive()) {
  regime_ext2 <- run_regime_extension_demo()
}

# =============================================================================
# SECTION: REGIME UTILITY HELPERS
# =============================================================================

# Fraction of time spent in each regime
regime_time_allocation <- function(regimes) {
  tab <- table(regimes)
  tab / sum(tab)
}

# Compute regime transition counts
count_transitions <- function(regimes) {
  pairs <- paste0(head(regimes,-1), "->", tail(regimes,-1))
  sort(table(pairs), decreasing=TRUE)
}

# Average return in each regime
regime_mean_return <- function(rets, regimes)
  tapply(rets, regimes, mean, na.rm=TRUE)

# Regime-conditional Sharpe
regime_conditional_sharpe <- function(rets, regimes, ann=252) {
  regs <- sort(unique(regimes))
  setNames(sapply(regs, function(r) {
    ri <- rets[regimes == r]
    if (length(ri) < 2) return(NA_real_)
    mean(ri) / (sd(ri)+1e-9) * sqrt(ann)
  }), regs)
}

# Expected regime given current streak
streak_regime_prior <- function(regimes, current_regime, min_streak=3) {
  n <- length(regimes)
  # Find all past occurrences of min_streak same-regime run
  matches <- 0; next_reg <- integer(0)
  for (t in seq(min_streak, n-1)) {
    if (all(regimes[(t-min_streak+1):t] == current_regime)) {
      matches <- matches + 1
      next_reg <- c(next_reg, regimes[t+1])
    }
  }
  if (matches == 0) return(NULL)
  list(n_matches=matches, next_regime_dist=table(next_reg)/matches)
}
