# =============================================================================
# reinforcement_learning.R
# RL for Trading: Q-Learning, SARSA, Policy Evaluation
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: RL frames trading as a sequential decision problem.
# The agent observes market "state" (vol regime, trend, time-of-day), picks
# a position fraction, receives a P&L reward, and updates its value estimates.
# Over thousands of episodes the agent learns which states warrant aggression
# and which demand caution -- without hand-crafting rules.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

#' Clip value to [lo, hi]
clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

#' Rolling standard deviation (population, width w)
roll_sd <- function(x, w) {
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in w:n) {
    out[i] <- sd(x[(i - w + 1):i])
  }
  out
}

#' Rolling mean (width w)
roll_mean <- function(x, w) {
  n <- length(x)
  out <- rep(NA_real_, n)
  for (i in w:n) {
    out[i] <- mean(x[(i - w + 1):i])
  }
  out
}

#' Sharpe ratio (annualised, daily returns)
sharpe_ratio <- function(rets, ann = 252) {
  mu <- mean(rets, na.rm = TRUE)
  sg <- sd(rets, na.rm = TRUE)
  if (is.na(sg) || sg == 0) return(NA_real_)
  (mu / sg) * sqrt(ann)
}

#' Maximum drawdown from an equity curve
max_drawdown <- function(equity) {
  peak <- cummax(equity)
  dd   <- (equity - peak) / peak
  min(dd, na.rm = TRUE)
}

#' Hit rate: fraction of positive returns
hit_rate <- function(rets) mean(rets > 0, na.rm = TRUE)

# ---------------------------------------------------------------------------
# 2. SYNTHETIC MARKET DATA GENERATOR
# ---------------------------------------------------------------------------
# We simulate a stylised crypto price series with:
#   - Stochastic vol (regime switching between low/high vol)
#   - Mild autocorrelation (trend)
#   - Fat tails via t-distributed shocks
#
generate_market <- function(n = 5000, seed = 42) {
  set.seed(seed)
  price <- numeric(n)
  vol   <- numeric(n)
  price[1] <- 10000
  vol[1]   <- 0.02

  # Regime: 0 = low vol, 1 = high vol
  regime <- integer(n)
  regime[1] <- 0L

  for (i in 2:n) {
    # Regime transition (Markov)
    if (regime[i - 1] == 0L) {
      regime[i] <- sample(c(0L, 1L), 1, prob = c(0.97, 0.03))
    } else {
      regime[i] <- sample(c(0L, 1L), 1, prob = c(0.10, 0.90))
    }
    base_vol <- if (regime[i] == 0L) 0.015 else 0.04
    # Mean-reverting vol
    vol[i] <- 0.8 * vol[i - 1] + 0.2 * base_vol + 0.003 * rnorm(1)
    vol[i] <- max(0.005, vol[i])
    # t-distributed return for fat tails
    ret <- vol[i] * rt(1, df = 5) / sqrt(5 / 3)
    # Slight autocorrelation (trend)
    if (i > 2) {
      prev_ret <- (price[i - 1] - price[i - 2]) / price[i - 2]
      ret <- ret + 0.05 * prev_ret
    }
    price[i] <- price[i - 1] * exp(ret)
  }

  list(
    price  = price,
    vol    = vol,
    regime = regime,
    ret    = c(NA, diff(log(price)))
  )
}

# ---------------------------------------------------------------------------
# 3. STATE DISCRETIZATION
# ---------------------------------------------------------------------------
# State = (vol_bin, trend_bin, hour_bin)
# vol_bin    : 1=low, 2=med, 3=high  (quantile-based on recent 60-bar vol)
# trend_bin  : 1=down, 2=flat, 3=up  (sign of 20-bar return)
# hour_bin   : 1..8 (simulated hour-of-day, 3-hour buckets)
#
# Total states = 3 x 3 x 8 = 72

N_VOL_BINS   <- 3L
N_TREND_BINS <- 3L
N_HOUR_BINS  <- 8L
N_STATES     <- N_VOL_BINS * N_TREND_BINS * N_HOUR_BINS  # 72

#' Compute state index (1-based) from features
#' @param vol_q   rolling vol quantile rank [0,1]
#' @param ret20   20-bar log return
#' @param hour    integer hour 0..23
state_index <- function(vol_q, ret20, hour) {
  # vol bin
  vb <- if (vol_q < 1/3) 1L else if (vol_q < 2/3) 2L else 3L
  # trend bin
  tb <- if (ret20 < -0.01) 1L else if (ret20 > 0.01) 3L else 2L
  # hour bin (3-hour buckets)
  hb <- as.integer(hour %/% 3) + 1L
  hb <- clip(hb, 1L, N_HOUR_BINS)
  # encode
  (vb - 1L) * (N_TREND_BINS * N_HOUR_BINS) +
    (tb - 1L) * N_HOUR_BINS +
    hb
}

#' Build full state sequence from market data
build_states <- function(mkt, lookback_vol = 60L, lookback_trend = 20L) {
  n     <- length(mkt$price)
  ret   <- mkt$ret
  price <- mkt$price
  vol   <- mkt$vol

  # Rolling vol rank (quantile within trailing window)
  vol_q <- numeric(n)
  for (i in (lookback_vol + 1):n) {
    w <- vol[(i - lookback_vol):(i - 1)]
    vol_q[i] <- mean(w < vol[i])   # empirical CDF rank
  }

  # 20-bar trend
  ret20 <- numeric(n)
  for (i in (lookback_trend + 1):n) {
    ret20[i] <- sum(ret[(i - lookback_trend + 1):i], na.rm = TRUE)
  }

  # Simulated hour (cycle)
  hour <- (seq_len(n) - 1L) %% 24L

  states <- integer(n)
  for (i in (lookback_vol + 1):n) {
    states[i] <- state_index(vol_q[i], ret20[i], hour[i])
  }
  list(states = states, vol_q = vol_q, ret20 = ret20, hour = hour,
       start  = lookback_vol + 2L)
}

# ---------------------------------------------------------------------------
# 4. ACTION SPACE
# ---------------------------------------------------------------------------
ACTIONS        <- c(0, 0.25, 0.50, 0.75, 1.00)   # position fractions
N_ACTIONS      <- length(ACTIONS)
TC_PER_TRADE   <- 0.001   # 10 bps transaction cost per unit of position change

# ---------------------------------------------------------------------------
# 5. Q-TABLE INITIALISATION
# ---------------------------------------------------------------------------

init_Q <- function(n_states = N_STATES, n_actions = N_ACTIONS) {
  matrix(0.0, nrow = n_states, ncol = n_actions)
}

# ---------------------------------------------------------------------------
# 6. EPSILON-GREEDY POLICY
# ---------------------------------------------------------------------------

#' Select action index (1-based) using epsilon-greedy
epsilon_greedy <- function(Q, state, epsilon) {
  if (runif(1) < epsilon) {
    sample.int(N_ACTIONS, 1L)
  } else {
    which.max(Q[state, ])
  }
}

# ---------------------------------------------------------------------------
# 7. REWARD FUNCTION
# ---------------------------------------------------------------------------

#' Compute single-step reward
#' position  : current position fraction (before action)
#' new_pos   : new position fraction (after action)
#' next_ret  : market log return over the next bar
compute_reward <- function(position, new_pos, next_ret) {
  pnl  <- new_pos * next_ret
  cost <- abs(new_pos - position) * TC_PER_TRADE
  pnl - cost
}

# ---------------------------------------------------------------------------
# 8. Q-LEARNING TRAINING LOOP
# ---------------------------------------------------------------------------
# Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

train_q_learning <- function(mkt, states_info,
                              n_episodes  = 1000L,
                              alpha       = 0.05,
                              gamma       = 0.95,
                              eps_start   = 1.0,
                              eps_end     = 0.05,
                              eps_decay   = 0.995,
                              seed        = 7L) {
  set.seed(seed)
  Q        <- init_Q()
  ret      <- mkt$ret
  states   <- states_info$states
  start    <- states_info$start
  n_bars   <- length(ret)
  episode_len <- 252L   # one "trading year" per episode

  # Storage for learning curve
  episode_rewards <- numeric(n_episodes)
  epsilon         <- eps_start

  for (ep in seq_len(n_episodes)) {
    # Sample a random start within valid range
    max_start <- n_bars - episode_len - 1L
    if (max_start < start) max_start <- start
    t0 <- sample(start:max(start, max_start), 1L)

    pos     <- 0.0   # initial position
    ep_rew  <- 0.0

    for (t in t0:(t0 + episode_len - 1L)) {
      if (t + 1L > n_bars) break
      s  <- states[t]
      if (s == 0L) next   # uninitialised state
      ai <- epsilon_greedy(Q, s, epsilon)
      new_pos <- ACTIONS[ai]
      r  <- compute_reward(pos, new_pos, ret[t + 1L])
      s2 <- states[t + 1L]
      if (s2 == 0L) s2 <- s
      # TD update
      best_next <- max(Q[s2, ])
      Q[s, ai]  <- Q[s, ai] + alpha * (r + gamma * best_next - Q[s, ai])
      pos    <- new_pos
      ep_rew <- ep_rew + r
    }
    episode_rewards[ep] <- ep_rew
    epsilon <- max(eps_end, epsilon * eps_decay)
  }

  list(Q = Q, episode_rewards = episode_rewards, epsilon_final = epsilon)
}

# ---------------------------------------------------------------------------
# 9. SARSA TRAINING LOOP
# ---------------------------------------------------------------------------
# On-policy variant: uses the action actually taken for the next step
# Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

train_sarsa <- function(mkt, states_info,
                         n_episodes = 1000L,
                         alpha      = 0.05,
                         gamma      = 0.95,
                         eps_start  = 1.0,
                         eps_end    = 0.05,
                         eps_decay  = 0.995,
                         seed       = 8L) {
  set.seed(seed)
  Q        <- init_Q()
  ret      <- mkt$ret
  states   <- states_info$states
  start    <- states_info$start
  n_bars   <- length(ret)
  episode_len <- 252L

  episode_rewards <- numeric(n_episodes)
  epsilon         <- eps_start

  for (ep in seq_len(n_episodes)) {
    max_start <- n_bars - episode_len - 1L
    if (max_start < start) max_start <- start
    t0  <- sample(start:max(start, max_start), 1L)
    pos <- 0.0
    ep_rew <- 0.0

    s  <- states[t0]
    if (s == 0L) s <- 1L
    ai <- epsilon_greedy(Q, s, epsilon)

    for (t in t0:(t0 + episode_len - 1L)) {
      if (t + 1L > n_bars) break
      new_pos <- ACTIONS[ai]
      r <- compute_reward(pos, new_pos, ret[t + 1L])
      s2 <- states[t + 1L]
      if (s2 == 0L) s2 <- s
      # Next action (on-policy)
      ai2 <- epsilon_greedy(Q, s2, epsilon)
      # SARSA update (uses ai2, not max)
      Q[s, ai] <- Q[s, ai] + alpha * (r + gamma * Q[s2, ai2] - Q[s, ai])
      pos <- new_pos
      ep_rew <- ep_rew + r
      s  <- s2
      ai <- ai2
    }
    episode_rewards[ep] <- ep_rew
    epsilon <- max(eps_end, epsilon * eps_decay)
  }

  list(Q = Q, episode_rewards = episode_rewards, epsilon_final = epsilon)
}

# ---------------------------------------------------------------------------
# 10. GREEDY POLICY EXECUTION (EVALUATION)
# ---------------------------------------------------------------------------
# Run the learned Q-table on a held-out period, greedy (epsilon=0)

execute_policy <- function(Q, mkt, states_info,
                            eval_start = NULL, eval_end = NULL) {
  ret    <- mkt$ret
  states <- states_info$states
  n_bars <- length(ret)
  start  <- states_info$start
  if (is.null(eval_start)) eval_start <- as.integer(0.8 * n_bars)
  if (is.null(eval_end))   eval_end   <- n_bars - 1L

  positions <- numeric(n_bars)
  pos <- 0.0
  for (t in eval_start:eval_end) {
    s  <- states[t]
    if (s == 0L) { positions[t] <- pos; next }
    ai <- which.max(Q[s, ])
    positions[t] <- ACTIONS[ai]
    pos <- ACTIONS[ai]
  }

  # Step returns
  rets_strat <- numeric(n_bars)
  for (t in eval_start:(eval_end - 1L)) {
    if (t + 1L > n_bars) break
    rets_strat[t] <- positions[t] * ret[t + 1L] -
      abs(positions[t + 1L] - positions[t]) * TC_PER_TRADE
  }

  active <- rets_strat[eval_start:eval_end]
  equity <- cumprod(1 + active)

  list(
    positions  = positions,
    rets_strat = rets_strat,
    active     = active,
    equity     = equity,
    sharpe     = sharpe_ratio(active),
    hit        = hit_rate(active),
    mdd        = max_drawdown(equity),
    total_ret  = tail(equity, 1) - 1
  )
}

# ---------------------------------------------------------------------------
# 11. BUY-AND-HOLD BENCHMARK
# ---------------------------------------------------------------------------

bh_benchmark <- function(mkt, eval_start, eval_end) {
  ret    <- mkt$ret
  active <- ret[eval_start:eval_end]
  active[is.na(active)] <- 0
  equity <- cumprod(1 + active)
  list(
    active    = active,
    equity    = equity,
    sharpe    = sharpe_ratio(active),
    hit       = hit_rate(active),
    mdd       = max_drawdown(equity),
    total_ret = tail(equity, 1) - 1
  )
}

# ---------------------------------------------------------------------------
# 12. LEARNING CURVE SMOOTHING
# ---------------------------------------------------------------------------

smooth_learning_curve <- function(rewards, window = 50L) {
  n   <- length(rewards)
  out <- numeric(n)
  for (i in seq_len(n)) {
    lo <- max(1L, i - window + 1L)
    out[i] <- mean(rewards[lo:i])
  }
  out
}

# ---------------------------------------------------------------------------
# 13. POLICY INSPECTION: OPTIMAL ACTION PER STATE
# ---------------------------------------------------------------------------

#' Extract greedy action for every state in the Q-table
policy_table <- function(Q) {
  n_s <- nrow(Q)
  df  <- data.frame(
    state       = seq_len(n_s),
    best_action = ACTIONS[apply(Q, 1, which.max)],
    Q_value     = apply(Q, 1, max),
    vol_bin     = ((seq_len(n_s) - 1L) %/% (N_TREND_BINS * N_HOUR_BINS)) + 1L,
    trend_bin   = (((seq_len(n_s) - 1L) %/% N_HOUR_BINS) %% N_TREND_BINS) + 1L,
    hour_bin    = ((seq_len(n_s) - 1L) %% N_HOUR_BINS) + 1L
  )
  df
}

#' Summarise average position by vol_bin
summarise_policy_by_vol <- function(ptable) {
  tapply(ptable$best_action, ptable$vol_bin, mean)
}

#' Summarise average position by trend_bin
summarise_policy_by_trend <- function(ptable) {
  tapply(ptable$best_action, ptable$trend_bin, mean)
}

# ---------------------------------------------------------------------------
# 14. PERFORMANCE COMPARISON TABLE
# ---------------------------------------------------------------------------

compare_policies <- function(ql_eval, sarsa_eval, bh_eval) {
  data.frame(
    strategy   = c("Q-Learning", "SARSA", "Buy-and-Hold"),
    sharpe     = c(ql_eval$sharpe,    sarsa_eval$sharpe,    bh_eval$sharpe),
    hit_rate   = c(ql_eval$hit,       sarsa_eval$hit,       bh_eval$hit),
    max_dd     = c(ql_eval$mdd,       sarsa_eval$mdd,       bh_eval$mdd),
    total_ret  = c(ql_eval$total_ret, sarsa_eval$total_ret, bh_eval$total_ret)
  )
}

# ---------------------------------------------------------------------------
# 15. HYPERPARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
# Vary alpha and gamma, report OOS Sharpe

sensitivity_analysis <- function(mkt, states_info,
                                  alphas = c(0.01, 0.05, 0.10, 0.20),
                                  gammas = c(0.80, 0.90, 0.95, 0.99),
                                  n_ep   = 300L) {
  eval_start <- as.integer(0.8 * length(mkt$ret))
  eval_end   <- length(mkt$ret) - 1L
  results    <- list()
  idx        <- 1L
  for (a in alphas) {
    for (g in gammas) {
      res <- train_q_learning(mkt, states_info,
                               n_episodes = n_ep,
                               alpha = a, gamma = g, seed = 99L)
      ev  <- execute_policy(res$Q, mkt, states_info, eval_start, eval_end)
      results[[idx]] <- data.frame(alpha = a, gamma = g,
                                    sharpe = ev$sharpe, mdd = ev$mdd)
      idx <- idx + 1L
    }
  }
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 16. REWARD SHAPING VARIANT
# ---------------------------------------------------------------------------
# Add a shaped reward component: penalise excessive drawdown in the episode

train_q_learning_shaped <- function(mkt, states_info,
                                     n_episodes   = 500L,
                                     alpha        = 0.05,
                                     gamma        = 0.95,
                                     eps_start    = 1.0,
                                     eps_end      = 0.05,
                                     eps_decay    = 0.995,
                                     dd_penalty   = 2.0,
                                     seed         = 11L) {
  set.seed(seed)
  Q        <- init_Q()
  ret      <- mkt$ret
  states   <- states_info$states
  start    <- states_info$start
  n_bars   <- length(ret)
  episode_len <- 252L
  epsilon  <- eps_start
  episode_rewards <- numeric(n_episodes)

  for (ep in seq_len(n_episodes)) {
    max_start <- n_bars - episode_len - 1L
    if (max_start < start) max_start <- start
    t0 <- sample(start:max(start, max_start), 1L)

    pos     <- 0.0
    ep_rew  <- 0.0
    peak_eq <- 1.0
    eq      <- 1.0

    for (t in t0:(t0 + episode_len - 1L)) {
      if (t + 1L > n_bars) break
      s  <- states[t]
      if (s == 0L) next
      ai  <- epsilon_greedy(Q, s, epsilon)
      new_pos <- ACTIONS[ai]
      r_raw   <- compute_reward(pos, new_pos, ret[t + 1L])
      eq      <- eq * (1 + r_raw)
      peak_eq <- max(peak_eq, eq)
      # Drawdown penalty
      dd      <- (eq - peak_eq) / peak_eq
      r       <- r_raw + dd_penalty * min(0, dd + 0.05)  # penalise if DD > 5%
      s2 <- states[t + 1L]
      if (s2 == 0L) s2 <- s
      best_next <- max(Q[s2, ])
      Q[s, ai]  <- Q[s, ai] + alpha * (r + gamma * best_next - Q[s, ai])
      pos    <- new_pos
      ep_rew <- ep_rew + r_raw
    }
    episode_rewards[ep] <- ep_rew
    epsilon <- max(eps_end, epsilon * eps_decay)
  }

  list(Q = Q, episode_rewards = episode_rewards)
}

# ---------------------------------------------------------------------------
# 17. DOUBLE Q-LEARNING (reduces maximisation bias)
# ---------------------------------------------------------------------------

train_double_q <- function(mkt, states_info,
                            n_episodes = 500L,
                            alpha      = 0.05,
                            gamma      = 0.95,
                            eps_start  = 1.0,
                            eps_end    = 0.05,
                            eps_decay  = 0.995,
                            seed       = 13L) {
  set.seed(seed)
  Q1 <- init_Q()
  Q2 <- init_Q()
  ret    <- mkt$ret
  states <- states_info$states
  start  <- states_info$start
  n_bars <- length(ret)
  episode_len <- 252L
  epsilon <- eps_start
  episode_rewards <- numeric(n_episodes)

  for (ep in seq_len(n_episodes)) {
    max_start <- n_bars - episode_len - 1L
    if (max_start < start) max_start <- start
    t0  <- sample(start:max(start, max_start), 1L)
    pos <- 0.0
    ep_rew <- 0.0

    for (t in t0:(t0 + episode_len - 1L)) {
      if (t + 1L > n_bars) break
      s  <- states[t]
      if (s == 0L) next
      # Action from combined Q
      combined <- Q1[s, ] + Q2[s, ]
      ai <- if (runif(1) < epsilon) sample.int(N_ACTIONS, 1L) else which.max(combined)
      new_pos <- ACTIONS[ai]
      r  <- compute_reward(pos, new_pos, ret[t + 1L])
      s2 <- states[t + 1L]; if (s2 == 0L) s2 <- s
      # Alternate which Q gets updated
      if (runif(1) < 0.5) {
        best_ai <- which.max(Q1[s2, ])
        Q1[s, ai] <- Q1[s, ai] + alpha * (r + gamma * Q2[s2, best_ai] - Q1[s, ai])
      } else {
        best_ai <- which.max(Q2[s2, ])
        Q2[s, ai] <- Q2[s, ai] + alpha * (r + gamma * Q1[s2, best_ai] - Q2[s, ai])
      }
      pos    <- new_pos
      ep_rew <- ep_rew + r
    }
    episode_rewards[ep] <- ep_rew
    epsilon <- max(eps_end, epsilon * eps_decay)
  }

  Q_avg <- (Q1 + Q2) / 2
  list(Q = Q_avg, Q1 = Q1, Q2 = Q2, episode_rewards = episode_rewards)
}

# ---------------------------------------------------------------------------
# 18. ROLLING OOS EVALUATION (walk-forward)
# ---------------------------------------------------------------------------

rolling_rl_eval <- function(mkt, states_info,
                             train_len = 1000L,
                             eval_len  = 126L,
                             n_ep      = 200L) {
  n_bars <- length(mkt$ret)
  start  <- states_info$start
  results <- list()
  idx     <- 1L

  t <- start + train_len
  while (t + eval_len <= n_bars) {
    train_start <- t - train_len
    train_end   <- t - 1L
    eval_start  <- t
    eval_end    <- min(t + eval_len - 1L, n_bars - 1L)

    # Train on window
    res <- train_q_learning(mkt, states_info,
                             n_episodes = n_ep, seed = idx * 7L)
    ev  <- execute_policy(res$Q, mkt, states_info, eval_start, eval_end)

    results[[idx]] <- data.frame(
      window    = idx,
      eval_start = eval_start,
      eval_end  = eval_end,
      sharpe    = ev$sharpe,
      mdd       = ev$mdd,
      total_ret = ev$total_ret
    )
    idx <- idx + 1L
    t   <- t + eval_len
  }
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 19. STATE VISIT FREQUENCY
# ---------------------------------------------------------------------------

state_visit_count <- function(states_info, eval_start, eval_end) {
  s <- states_info$states[eval_start:eval_end]
  s <- s[s > 0]
  counts <- tabulate(s, nbins = N_STATES)
  data.frame(state = seq_len(N_STATES), visits = counts,
             freq  = counts / sum(counts))
}

# ---------------------------------------------------------------------------
# 20. MAIN DEMO RUNNER
# ---------------------------------------------------------------------------

run_rl_demo <- function() {
  cat("=== RL for Trading Demo ===\n\n")

  # Generate synthetic market
  cat("Generating synthetic crypto market (5000 bars)...\n")
  mkt <- generate_market(n = 5000, seed = 42)
  cat(sprintf("  Price range: %.0f - %.0f\n", min(mkt$price), max(mkt$price)))
  cat(sprintf("  Annualised vol (full): %.1f%%\n",
              sd(mkt$ret, na.rm=TRUE) * sqrt(252) * 100))

  # Build states
  cat("\nBuilding state features...\n")
  si <- build_states(mkt)
  cat(sprintf("  Valid states from bar %d\n", si$start))
  cat(sprintf("  Unique states visited: %d / %d\n",
              length(unique(si$states[si$states > 0])), N_STATES))

  # Split
  eval_start <- as.integer(0.8 * length(mkt$ret))
  eval_end   <- length(mkt$ret) - 1L
  cat(sprintf("\nTrain: bars 1-%d | OOS eval: bars %d-%d\n",
              eval_start - 1L, eval_start, eval_end))

  # Train Q-Learning
  cat("\nTraining Q-Learning (1000 episodes)...\n")
  t0 <- proc.time()
  ql_res <- train_q_learning(mkt, si, n_episodes = 1000L)
  cat(sprintf("  Done in %.1f sec\n", (proc.time() - t0)["elapsed"]))

  # Train SARSA
  cat("Training SARSA (1000 episodes)...\n")
  t0 <- proc.time()
  sa_res <- train_sarsa(mkt, si, n_episodes = 1000L)
  cat(sprintf("  Done in %.1f sec\n", (proc.time() - t0)["elapsed"]))

  # Train Double Q
  cat("Training Double Q-Learning (500 episodes)...\n")
  dq_res <- train_double_q(mkt, si, n_episodes = 500L)

  # Evaluate
  ql_ev  <- execute_policy(ql_res$Q,    mkt, si, eval_start, eval_end)
  sa_ev  <- execute_policy(sa_res$Q,    mkt, si, eval_start, eval_end)
  dq_ev  <- execute_policy(dq_res$Q,    mkt, si, eval_start, eval_end)
  bh_ev  <- bh_benchmark(mkt, eval_start, eval_end)

  cat("\n--- OOS Performance ---\n")
  cmp <- compare_policies(ql_ev, sa_ev, bh_ev)
  print(cmp)
  cat(sprintf("\nDouble Q-Learning  Sharpe=%.3f  MaxDD=%.1f%%  Ret=%.1f%%\n",
              dq_ev$sharpe, dq_ev$mdd * 100, dq_ev$total_ret * 100))

  # Policy inspection
  cat("\n--- Q-Learning Policy: avg position by vol regime ---\n")
  pt   <- policy_table(ql_res$Q)
  vsum <- summarise_policy_by_vol(pt)
  cat("  Vol bin 1 (low):  avg pos =", round(vsum[1], 3), "\n")
  cat("  Vol bin 2 (med):  avg pos =", round(vsum[2], 3), "\n")
  cat("  Vol bin 3 (high): avg pos =", round(vsum[3], 3), "\n")
  # Financial intuition: well-trained agent should reduce position in high vol

  # Learning curve: convergence check
  smooth_ql <- smooth_learning_curve(ql_res$episode_rewards, 50L)
  smooth_sa <- smooth_learning_curve(sa_res$episode_rewards, 50L)
  cat("\n--- Learning Curve (last 100 ep avg reward) ---\n")
  cat("  Q-Learning: ", round(mean(tail(ql_res$episode_rewards, 100)), 4), "\n")
  cat("  SARSA:      ", round(mean(tail(sa_res$episode_rewards, 100)), 4), "\n")

  # Reward shaping
  cat("\nTraining reward-shaped Q (500 ep)...\n")
  shaped <- train_q_learning_shaped(mkt, si, n_episodes = 500L)
  sh_ev  <- execute_policy(shaped$Q, mkt, si, eval_start, eval_end)
  cat(sprintf("  Shaped Q  Sharpe=%.3f  MaxDD=%.1f%%\n",
              sh_ev$sharpe, sh_ev$mdd * 100))

  # Sensitivity
  cat("\nHyperparameter sensitivity (alpha x gamma grid, 100 ep each)...\n")
  sens <- sensitivity_analysis(mkt, si, n_ep = 100L)
  best <- sens[which.max(sens$sharpe), ]
  cat(sprintf("  Best: alpha=%.2f  gamma=%.2f  OOS Sharpe=%.3f\n",
              best$alpha, best$gamma, best$sharpe))

  invisible(list(ql = ql_res, sarsa = sa_res, dq = dq_res,
                 evals = list(ql = ql_ev, sarsa = sa_ev,
                              dq = dq_ev, bh = bh_ev),
                 sens = sens))
}

# Run demo when sourced interactively
if (interactive()) {
  rl_results <- run_rl_demo()
}

# ---------------------------------------------------------------------------
# 21. EXPERIENCE REPLAY BUFFER
# ---------------------------------------------------------------------------
# Store (s,a,r,s') transitions; sample mini-batches for Q updates.
# Breaks temporal correlation, stabilises learning -- borrowed from DQN.

replay_buffer_init <- function(capacity = 5000L) {
  list(capacity = capacity, size = 0L, idx = 0L,
       s  = integer(capacity), a = integer(capacity),
       r  = numeric(capacity), s2 = integer(capacity))
}

replay_buffer_push <- function(buf, s, a, r, s2) {
  idx <- (buf$idx %% buf$capacity) + 1L
  buf$s[idx] <- s; buf$a[idx] <- a
  buf$r[idx] <- r; buf$s2[idx] <- s2
  buf$idx  <- buf$idx + 1L
  buf$size <- min(buf$size + 1L, buf$capacity)
  buf
}

replay_buffer_sample <- function(buf, batch_size = 32L) {
  n   <- buf$size
  idx <- sample.int(n, min(batch_size, n))
  list(s = buf$s[idx], a = buf$a[idx],
       r = buf$r[idx], s2 = buf$s2[idx])
}

#' Q-Learning with experience replay
train_q_replay <- function(mkt, states_info,
                            n_episodes   = 500L,
                            alpha        = 0.05,
                            gamma        = 0.95,
                            eps_start    = 1.0,
                            eps_end      = 0.05,
                            eps_decay    = 0.995,
                            batch_size   = 32L,
                            buf_capacity = 5000L,
                            seed         = 21L) {
  set.seed(seed)
  Q   <- init_Q()
  buf <- replay_buffer_init(buf_capacity)
  ret <- mkt$ret; states <- states_info$states
  start <- states_info$start; n_bars <- length(ret)
  episode_len <- 252L; epsilon <- eps_start
  episode_rewards <- numeric(n_episodes)

  for (ep in seq_len(n_episodes)) {
    max_start <- max(start, n_bars - episode_len - 1L)
    t0 <- sample(start:max_start, 1L)
    pos <- 0.0; ep_rew <- 0.0

    for (t in t0:(t0 + episode_len - 1L)) {
      if (t + 1L > n_bars) break
      s <- states[t]; if (s == 0L) next
      ai <- epsilon_greedy(Q, s, epsilon)
      new_pos <- ACTIONS[ai]
      r  <- compute_reward(pos, new_pos, ret[t + 1L])
      s2 <- states[t + 1L]; if (s2 == 0L) s2 <- s
      buf <- replay_buffer_push(buf, s, ai, r, s2)

      if (buf$size >= batch_size) {
        batch <- replay_buffer_sample(buf, batch_size)
        for (b in seq_len(length(batch$s))) {
          best_next <- max(Q[batch$s2[b], ])
          Q[batch$s[b], batch$a[b]] <- Q[batch$s[b], batch$a[b]] +
            alpha * (batch$r[b] + gamma * best_next - Q[batch$s[b], batch$a[b]])
        }
      }
      pos <- new_pos; ep_rew <- ep_rew + r
    }
    episode_rewards[ep] <- ep_rew
    epsilon <- max(eps_end, epsilon * eps_decay)
  }
  list(Q = Q, episode_rewards = episode_rewards, buffer_size = buf$size)
}

# ---------------------------------------------------------------------------
# 22. Q-VALUE HEATMAP BY HOUR
# ---------------------------------------------------------------------------

q_heatmap_hour <- function(Q, hour_bin = 1L) {
  mat <- matrix(NA, N_VOL_BINS, N_TREND_BINS)
  for (vb in 1:N_VOL_BINS) {
    for (tb in 1:N_TREND_BINS) {
      s_idx <- (vb - 1L) * (N_TREND_BINS * N_HOUR_BINS) +
                (tb - 1L) * N_HOUR_BINS + hour_bin
      mat[vb, tb] <- max(Q[s_idx, ])
    }
  }
  rownames(mat) <- paste0("Vol", 1:N_VOL_BINS)
  colnames(mat) <- paste0("Trend", 1:N_TREND_BINS)
  mat
}

# ---------------------------------------------------------------------------
# 23. CONVERGENCE DIAGNOSTICS
# ---------------------------------------------------------------------------

q_convergence <- function(mkt, states_info, n_episodes = 200L,
                           alpha = 0.05, gamma = 0.95) {
  Q   <- init_Q(); ret <- mkt$ret; states <- states_info$states
  start <- states_info$start; n_bars <- length(ret)
  episode_len <- 252L; norms <- numeric(n_episodes); epsilon <- 1.0

  for (ep in seq_len(n_episodes)) {
    Q_old     <- Q
    max_start <- max(start, n_bars - episode_len - 1L)
    t0  <- sample(start:max_start, 1L)
    pos <- 0.0
    for (t in t0:(t0 + episode_len - 1L)) {
      if (t + 1L > n_bars) break
      s <- states[t]; if (s == 0L) next
      ai <- epsilon_greedy(Q, s, epsilon)
      new_pos <- ACTIONS[ai]; r <- compute_reward(pos, new_pos, ret[t+1L])
      s2 <- states[t+1L]; if (s2 == 0L) s2 <- s
      Q[s, ai] <- Q[s, ai] + alpha*(r + gamma*max(Q[s2,]) - Q[s, ai])
      pos <- new_pos
    }
    norms[ep] <- sum((Q - Q_old)^2)
    epsilon   <- max(0.05, epsilon * 0.995)
  }
  list(norms = norms, Q_final = Q)
}

# ---------------------------------------------------------------------------
# 24. TRADE STATISTICS
# ---------------------------------------------------------------------------

trade_stats <- function(rets) {
  wins   <- rets[rets > 0]
  losses <- rets[rets < 0]
  pf     <- if (length(losses) > 0 && sum(abs(losses)) > 0)
              sum(wins) / sum(abs(losses)) else NA
  rle_obj <- rle(rets <= 0)
  max_cl  <- if (any(rle_obj$values)) max(rle_obj$lengths[rle_obj$values]) else 0L
  data.frame(
    n_trades     = length(rets),
    n_wins       = length(wins),
    n_losses     = length(losses),
    profit_factor = pf,
    avg_win      = if (length(wins)  > 0) mean(wins)   else NA,
    avg_loss     = if (length(losses)> 0) mean(losses)  else NA,
    max_consec_loss = max_cl
  )
}

# ---------------------------------------------------------------------------
# 25. MULTI-ASSET INDEPENDENT Q-TABLES
# ---------------------------------------------------------------------------
# Each asset gets its own Q-table; position fraction chosen independently.

train_multi_asset_rl <- function(mkt_list, n_episodes = 200L,
                                  alpha = 0.05, gamma = 0.95, seed = 99L) {
  set.seed(seed)
  N_assets   <- length(mkt_list)
  Q_tables   <- lapply(seq_len(N_assets), function(i) init_Q())
  si_list    <- lapply(mkt_list, build_states)
  ep_rewards <- matrix(0, n_episodes, N_assets)
  epsilon    <- 1.0

  for (ep in seq_len(n_episodes)) {
    for (i in seq_len(N_assets)) {
      mkt <- mkt_list[[i]]; si <- si_list[[i]]
      n_bars <- length(mkt$ret); episode_len <- 252L
      max_start <- max(si$start, n_bars - episode_len - 1L)
      t0 <- sample(si$start:max_start, 1L)
      pos <- 0.0; ep_rew <- 0.0
      for (t in t0:(t0 + episode_len - 1L)) {
        if (t + 1L > n_bars) break
        s <- si$states[t]; if (s == 0L) next
        ai <- epsilon_greedy(Q_tables[[i]], s, epsilon)
        new_pos <- ACTIONS[ai]
        r  <- compute_reward(pos, new_pos, mkt$ret[t + 1L])
        s2 <- si$states[t + 1L]; if (s2 == 0L) s2 <- s
        Q_tables[[i]][s, ai] <- Q_tables[[i]][s, ai] +
          alpha * (r + gamma * max(Q_tables[[i]][s2, ]) - Q_tables[[i]][s, ai])
        pos <- new_pos; ep_rew <- ep_rew + r
      }
      ep_rewards[ep, i] <- ep_rew
    }
    epsilon <- max(0.05, epsilon * 0.995)
  }
  list(Q_tables = Q_tables, episode_rewards = ep_rewards)
}

# ---------------------------------------------------------------------------
# 26. EPISODE REWARD SMOOTHING AND STATISTICS
# ---------------------------------------------------------------------------

episode_reward_stats <- function(rewards, window = 50L) {
  n <- length(rewards)
  smoothed <- smooth_learning_curve(rewards, window)
  list(
    smoothed    = smoothed,
    final_mean  = mean(tail(rewards, window)),
    improvement = mean(tail(rewards, window)) - mean(head(rewards, window)),
    volatility  = sd(rewards),
    monotone    = all(diff(smoothed[!is.na(smoothed)]) >= 0)
  )
}

# ---------------------------------------------------------------------------
# 27. EXTENDED DEMO
# ---------------------------------------------------------------------------

run_rl_extended_demo <- function() {
  cat("=== RL Extended Features Demo ===\n\n")
  mkt <- generate_market(n = 3000L, seed = 55L)
  si  <- build_states(mkt)

  cat("Training Q with experience replay (300 episodes)...\n")
  ql_rep <- train_q_replay(mkt, si, n_episodes = 300L, batch_size = 32L)
  eval_start <- as.integer(0.8 * length(mkt$ret))
  ev <- execute_policy(ql_rep$Q, mkt, si, eval_start, length(mkt$ret) - 1L)
  cat(sprintf("  Replay Q  Sharpe=%.3f  MaxDD=%.1f%%\n",
              ev$sharpe, ev$mdd * 100))

  cat("Q-value heatmap (hour bin 1):\n")
  print(round(q_heatmap_hour(ql_rep$Q, 1L), 3))

  cat("Convergence diagnostics (100 episodes)...\n")
  conv <- q_convergence(mkt, si, n_episodes = 100L)
  cat(sprintf("  Norm ep10=%.6f  ep100=%.6f\n",
              conv$norms[10], conv$norms[100]))

  cat("Episode reward stats:\n")
  es <- episode_reward_stats(ql_rep$episode_rewards)
  cat(sprintf("  Improvement: %.4f  Monotone: %s\n",
              es$improvement, es$monotone))

  cat("Trade stats:\n")
  active_rets <- ev$active[ev$active != 0]
  print(trade_stats(active_rets))

  cat("Multi-asset RL (2 markets, 200 episodes)...\n")
  mkt2 <- generate_market(n = 3000L, seed = 77L)
  ma_res <- train_multi_asset_rl(list(mkt, mkt2), n_episodes = 200L)
  ev1 <- execute_policy(ma_res$Q_tables[[1]], mkt,  si,  eval_start, length(mkt$ret)-1L)
  ev2 <- execute_policy(ma_res$Q_tables[[2]], mkt2, build_states(mkt2),
                         eval_start, length(mkt2$ret)-1L)
  cat(sprintf("  Asset1 Sharpe=%.3f  Asset2 Sharpe=%.3f\n",
              ev1$sharpe, ev2$sharpe))

  invisible(list(ql_rep=ql_rep, conv=conv, ma=ma_res))
}

if (interactive()) {
  rl_ext <- run_rl_extended_demo()
}
