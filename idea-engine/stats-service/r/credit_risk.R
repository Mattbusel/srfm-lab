# =============================================================================
# credit_risk.R
# Credit Risk Analytics for Crypto Counterparties
# Pure base R -- no external packages
# =============================================================================
# Financial intuition: Crypto lending, perpetuals, and DeFi protocols carry
# substantial counterparty credit risk. The 2022 FTX/Celsius/3AC cascade
# demonstrated that crypto credit risk is highly correlated and can freeze
# markets within days. This module adapts classical credit risk models
# (Merton, KMV, migration matrices, CVA) to the crypto context.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ---------------------------------------------------------------------------

#' Standard normal CDF
pnorm_approx <- function(x) pnorm(x)

#' Black-Scholes d1, d2
bs_d1 <- function(S, K, r, sigma, T_) {
  (log(S/K) + (r + 0.5 * sigma^2) * T_) / (sigma * sqrt(T_))
}
bs_d2 <- function(S, K, r, sigma, T_) bs_d1(S, K, r, sigma, T_) - sigma * sqrt(T_)

#' Black-Scholes call price
bs_call <- function(S, K, r, sigma, T_) {
  if (T_ <= 0 || sigma <= 0) return(max(S - K, 0))
  d1 <- bs_d1(S, K, r, sigma, T_)
  d2 <- d1 - sigma * sqrt(T_)
  S * pnorm(d1) - K * exp(-r * T_) * pnorm(d2)
}

#' Clip
clip <- function(x, lo, hi) pmax(lo, pmin(hi, x))

# ---------------------------------------------------------------------------
# 2. MERTON STRUCTURAL MODEL
# ---------------------------------------------------------------------------
# Equity = Call option on firm value: E = V*N(d1) - D*e^{-rT}*N(d2)
# Firm value V and asset volatility sigma_V are unobservable.
# We solve a 2-equation system: (1) E = bs_call(V,D,r,sigma_V,T)
#                                (2) sigma_E * E = N(d1) * sigma_V * V

#' Solve Merton model via iterative procedure
#' @param E       equity market cap
#' @param D       face value of debt
#' @param r       risk-free rate
#' @param T_mat   debt maturity (years)
#' @param sigma_E equity return volatility (annualised)
merton_solve <- function(E, D, r, T_mat, sigma_E,
                          max_iter = 200L, tol = 1e-8) {
  # Initial guess: V = E + D
  V     <- E + D
  s_V   <- sigma_E * E / (E + D)

  for (iter in seq_len(max_iter)) {
    d1   <- bs_d1(V, D, r, s_V, T_mat)
    # Update sigma_V from hedge ratio equation
    n_d1 <- pnorm(d1)
    s_V_new <- sigma_E * E / (n_d1 * V)
    s_V_new <- clip(s_V_new, 0.001, 5.0)
    # Update V from call equation
    E_hat <- bs_call(V, D, r, s_V_new, T_mat)
    V_new <- V + (E - E_hat)   # Newton-like step
    V_new <- max(V_new, E * 0.1)

    if (abs(V_new - V) < tol && abs(s_V_new - s_V) < tol) {
      V <- V_new; s_V <- s_V_new; break
    }
    V   <- 0.7 * V + 0.3 * V_new
    s_V <- 0.7 * s_V + 0.3 * s_V_new
  }

  d1 <- bs_d1(V, D, r, s_V, T_mat)
  d2 <- d1 - s_V * sqrt(T_mat)
  list(V        = V,
       sigma_V  = s_V,
       d1       = d1,
       d2       = d2,
       PD       = pnorm(-d2),       # risk-neutral default probability
       DD       = d2,               # distance-to-default
       equity_hat = bs_call(V, D, r, s_V, T_mat))
}

#' Batch Merton for a panel of crypto entities
batch_merton <- function(panel) {
  # panel: data.frame with cols E, D, r, T_mat, sigma_E, name
  results <- lapply(seq_len(nrow(panel)), function(i) {
    res <- tryCatch(
      merton_solve(panel$E[i], panel$D[i], panel$r[i],
                   panel$T_mat[i], panel$sigma_E[i]),
      error = function(e) list(V=NA,sigma_V=NA,d1=NA,d2=NA,PD=NA,DD=NA)
    )
    data.frame(name = panel$name[i], E=panel$E[i], D=panel$D[i],
               V=res$V, sigma_V=res$sigma_V, DD=res$DD, PD=res$PD)
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 3. KMV DISTANCE-TO-DEFAULT
# ---------------------------------------------------------------------------
# KMV uses empirical (physical) default probabilities by mapping DD to
# historical default frequencies (Expected Default Frequency = EDF).
# Here we approximate with a lookup table calibrated to crypto volatility.

#' KMV DD -> EDF mapping (log-logistic approximation for crypto)
dd_to_edf <- function(DD, scale = 0.5, shape = 1.2) {
  # High-vol crypto: EDF is much higher at given DD than IG corporates
  # scale: base hazard for crypto universe
  PD <- 1 / (1 + (pmax(DD, 0) / scale)^shape)
  clip(PD, 0, 1)
}

#' Rolling distance-to-default from equity time series
rolling_dd <- function(equity_prices, debt_face, r = 0.05,
                        T_mat = 1.0, window = 252L) {
  n  <- length(equity_prices)
  DD <- rep(NA, n)
  PD <- rep(NA, n)
  for (i in (window + 1):n) {
    E_t     <- equity_prices[i]
    ret_w   <- diff(log(equity_prices[(i-window):i]))
    sigma_E <- sd(ret_w, na.rm=TRUE) * sqrt(252)
    if (is.na(sigma_E) || sigma_E <= 0) next
    res     <- tryCatch(
      merton_solve(E_t, debt_face, r, T_mat, sigma_E),
      error = function(e) list(DD=NA, PD=NA))
    DD[i] <- res$DD
    PD[i] <- res$PD
  }
  data.frame(t = seq_len(n), equity = equity_prices, DD = DD, PD = PD)
}

# ---------------------------------------------------------------------------
# 4. EXCHANGE SOLVENCY SCORING (Proof-of-Reserves Proxy)
# ---------------------------------------------------------------------------
# In absence of real PoR data, we construct a scoring model using:
#   - Leverage proxy: liabilities / assets (lower = safer)
#   - Liquidity buffer: cash-equivalent fraction
#   - Withdrawal velocity: relative to buffer (stress scenario)
#   - On-chain concentration: large custodial wallets as fraction

exchange_solvency_score <- function(assets, liabilities,
                                     liquid_ratio,
                                     withdrawal_stress = 0.20,
                                     concentration = 0.5) {
  leverage  <- liabilities / max(assets, 1)
  buffer    <- liquid_ratio * assets - withdrawal_stress * liabilities
  # Score components [0,100]
  lev_score  <- 100 * (1 - clip(leverage, 0, 1))
  buf_score  <- 100 * clip(buffer / max(assets, 1) + 0.5, 0, 1)
  conc_score <- 100 * (1 - clip(concentration, 0, 1))
  # Weighted composite
  score <- 0.4 * lev_score + 0.4 * buf_score + 0.2 * conc_score
  data.frame(score = score, leverage = leverage,
             buffer_ratio = buffer / max(assets,1),
             lev_score = lev_score, buf_score = buf_score,
             conc_score = conc_score,
             label = ifelse(score > 70, "Solvent",
                            ifelse(score > 40, "Stressed", "Insolvent")))
}

# ---------------------------------------------------------------------------
# 5. CONTAGION DEFAULT CASCADE
# ---------------------------------------------------------------------------
# Model: N entities, bilateral exposures matrix X.
# If entity i defaults, it transmits loss_fraction * X[i,j] to entity j.
# Entity j defaults if total losses exceed its equity buffer.

simulate_default_cascade <- function(equity, exposures, loss_fraction = 0.6,
                                      initial_default = NULL,
                                      n_sims = 1000L, seed = 42L) {
  set.seed(seed)
  n <- length(equity)
  if (is.null(initial_default)) initial_default <- 1L

  total_defaults_per_sim <- integer(n_sims)

  for (sim in seq_len(n_sims)) {
    E     <- equity + rnorm(n, 0, 0.05 * equity)   # perturb equity
    E     <- pmax(E, 0)
    defaulted <- logical(n)
    # Initial shock
    defaulted[initial_default] <- TRUE

    changed <- TRUE
    while (changed) {
      changed <- FALSE
      for (j in seq_len(n)) {
        if (defaulted[j]) next
        # Loss from defaulted counterparties
        total_loss <- sum(loss_fraction * exposures[defaulted, j])
        if (total_loss > E[j]) {
          defaulted[j] <- TRUE
          E[j] <- 0
          changed <- TRUE
        }
      }
    }
    total_defaults_per_sim[sim] <- sum(defaulted) - 1L  # excluding initial
  }

  list(
    mean_cascade   = mean(total_defaults_per_sim),
    median_cascade = median(total_defaults_per_sim),
    p95_cascade    = quantile(total_defaults_per_sim, 0.95),
    dist           = table(total_defaults_per_sim),
    systemic_risk  = mean(total_defaults_per_sim >= n / 2)
  )
}

# ---------------------------------------------------------------------------
# 6. CVA (CREDIT VALUATION ADJUSTMENT)
# ---------------------------------------------------------------------------
# CVA = Expected Loss on OTC derivatives due to counterparty default.
# CVA = sum_t  EE(t) * PD(t, t+dt) * LGD * df(t)
# where EE = expected exposure, PD = marginal default probability, df = discount.

#' Compute CVA for a stream of exposures
compute_cva <- function(expected_exposure,   # vector of expected exposure at each t
                         pd_marginal,         # marginal default probability at each t
                         lgd      = 0.6,
                         discount = 0.95) {
  T_  <- length(expected_exposure)
  df  <- discount^(seq_len(T_) / T_)   # simple discount curve
  cva <- sum(expected_exposure * pd_marginal * lgd * df, na.rm = TRUE)
  cva
}

#' Simulate expected exposure profile for a perpetual futures position
exposure_profile_perp <- function(notional, sigma_daily, T_steps = 252L,
                                   n_paths = 1000L, seed = 77L) {
  set.seed(seed)
  # MtM exposure under random walks (simplified)
  paths <- matrix(0, n_paths, T_steps)
  for (p in seq_len(n_paths)) {
    shock    <- cumsum(rnorm(T_steps, 0, sigma_daily))
    paths[p,] <- notional * abs(shock / max(abs(shock), 1e-6) * sigma_daily * sqrt(1:T_steps))
  }
  ee <- colMeans(paths)  # expected exposure profile
  ee
}

#' Marginal default probability from survival function
marginal_pd <- function(hazard_rate, T_steps = 252L) {
  dt      <- 1 / T_steps
  surv    <- exp(-hazard_rate * dt * seq_len(T_steps))
  pd_marg <- c(1 - surv[1], -diff(surv))
  pd_marg
}

# ---------------------------------------------------------------------------
# 7. WRONG-WAY RISK (WWR) IN CRYPTO LENDING
# ---------------------------------------------------------------------------
# WWR: exposure and default probability move together (adverse correlation).
# In crypto lending against volatile collateral:
#   - Collateral value falls -> more exposure (undercollateralised)
#   - Low collateral value -> higher PD (borrower margin called)
# We model this correlation using a Gaussian copula.

simulate_wwr <- function(notional, sigma_collateral, rho_wwr = 0.7,
                           hazard_rate = 0.05, T_ = 1.0,
                           n_sims = 10000L, seed = 88L) {
  set.seed(seed)
  # Correlated normal draws for collateral return and default trigger
  Z1 <- rnorm(n_sims); Z2 <- rnorm(n_sims)
  Zc <- rho_wwr * Z1 + sqrt(1 - rho_wwr^2) * Z2   # collateral factor
  Zd <- Z1                                           # default factor (correlated)

  # Collateral at T_
  collateral_return <- exp((-0.5 * sigma_collateral^2) * T_ +
                             sigma_collateral * sqrt(T_) * Zc)
  # Default if Zd < Phi^{-1}(PD)
  pd_total <- 1 - exp(-hazard_rate * T_)
  default  <- Zd < qnorm(pd_total)

  # Exposure at default: max(notional - collateral, 0)
  collateral_value <- notional * collateral_return
  ead              <- pmax(notional - collateral_value, 0)
  lgd              <- 0.6   # assume 60% loss given default

  # CVA without WWR (independence)
  cva_no_wwr <- mean(ead) * pd_total * lgd

  # CVA with WWR (correlated exposure and default)
  cva_wwr    <- mean(ead[default]) * mean(default) * lgd

  data.frame(
    cva_no_wwr       = cva_no_wwr,
    cva_wwr          = cva_wwr,
    wwr_multiplier   = cva_wwr / max(cva_no_wwr, 1e-6),
    pd_total         = pd_total,
    avg_ead          = mean(ead),
    avg_ead_default  = mean(ead[default])
  )
}

# ---------------------------------------------------------------------------
# 8. MIGRATION MATRIX ESTIMATION
# ---------------------------------------------------------------------------
# States: 1=AAA-equivalent, 2=BBB-equivalent, 3=B-equivalent,
#         4=CCC-equivalent, 5=Default
# For crypto: states map to exchange/protocol health scores.

#' Estimate migration matrix from panel of credit scores
estimate_migration_matrix <- function(scores_panel, thresholds = c(80, 60, 40, 20)) {
  # scores_panel: T x N matrix of scores (0-100)
  # thresholds define state boundaries
  T_  <- nrow(scores_panel)
  N   <- ncol(scores_panel)
  K   <- length(thresholds) + 1L   # number of states (last = default)

  classify <- function(s) {
    if (is.na(s)) return(NA_integer_)
    state <- K   # default
    for (i in seq_along(thresholds)) {
      if (s > thresholds[i]) { state <- i; break }
    }
    state
  }

  # Count transitions
  trans <- matrix(0L, K, K)
  for (i in seq_len(N)) {
    for (t in 1:(T_-1)) {
      from <- classify(scores_panel[t, i])
      to   <- classify(scores_panel[t+1, i])
      if (!is.na(from) && !is.na(to)) trans[from, to] <- trans[from, to] + 1L
    }
  }

  # Normalise to probabilities (with Laplace smoothing)
  mig_mat <- (trans + 0.5) / (rowSums(trans + 0.5))
  rownames(mig_mat) <- colnames(mig_mat) <- c("AAA","BBB","B","CCC","D")
  list(migration = mig_mat, counts = trans)
}

#' Simulate credit score evolution using migration matrix
simulate_migration <- function(migration_matrix, initial_state = 2L,
                                T_ = 24L, n_sims = 1000L, seed = 55L) {
  set.seed(seed)
  K   <- nrow(migration_matrix)
  paths <- matrix(0L, n_sims, T_)

  for (sim in seq_len(n_sims)) {
    state    <- initial_state
    paths[sim, 1] <- state
    for (t in 2:T_) {
      if (state == K) { paths[sim, t] <- K; next }  # absorbing default
      probs <- migration_matrix[state, ]
      state <- sample.int(K, 1, prob = probs)
      paths[sim, t] <- state
    }
  }

  # Default probability by time horizon
  pd_horizon <- colMeans(paths == K)
  list(paths = paths, pd_horizon = pd_horizon,
       expected_state = colMeans(paths))
}

# ---------------------------------------------------------------------------
# 9. CREDIT-ADJUSTED POSITION SIZING
# ---------------------------------------------------------------------------
# Kelly fraction adjusted for counterparty default risk:
# Effective edge = raw edge - PD * LGD * position
# Adjusted Kelly = f * (1 - PD * LGD * leverage)

credit_adjusted_kelly <- function(mu, sigma, PD, LGD = 0.6, leverage = 1.0) {
  # Raw Kelly fraction
  kelly_raw <- mu / sigma^2
  # Credit cost
  credit_cost <- PD * LGD * leverage
  # Adjusted
  kelly_adj <- kelly_raw * (1 - credit_cost)
  list(kelly_raw  = kelly_raw,
       kelly_adj  = kelly_adj,
       credit_cost = credit_cost,
       reduction  = 1 - kelly_adj / max(kelly_raw, 1e-12))
}

#' Position size across multiple counterparties with different credit quality
portfolio_credit_sizing <- function(positions, PD_vec, LGD = 0.6,
                                     max_leverage = 3.0) {
  n <- length(positions)
  adjusted <- numeric(n)
  for (i in seq_len(n)) {
    credit_factor <- 1 - PD_vec[i] * LGD
    adjusted[i]   <- positions[i] * credit_factor
  }
  # Scale so total leverage <= max_leverage
  total_lev <- sum(abs(adjusted))
  if (total_lev > max_leverage) adjusted <- adjusted * max_leverage / total_lev
  data.frame(
    raw_position  = positions,
    PD            = PD_vec,
    credit_factor = 1 - PD_vec * LGD,
    adjusted_pos  = adjusted
  )
}

# ---------------------------------------------------------------------------
# 10. CREDIT SPREAD IMPLIED PD
# ---------------------------------------------------------------------------
# For traded credit instruments: strip PD from observed spread.
# Spread ≈ PD * LGD / T  (simplified flat hazard)

spread_to_pd <- function(spread, T_ = 1.0, LGD = 0.6) {
  spread / (LGD / T_)
}

pd_to_spread <- function(pd, T_ = 1.0, LGD = 0.6) {
  pd * LGD / T_
}

#' Term structure of implied PDs from observed credit spreads
term_structure_pd <- function(spreads, maturities, LGD = 0.6) {
  pds <- spreads / LGD * maturities
  data.frame(maturity = maturities, spread = spreads,
             implied_pd = clip(pds, 0, 1),
             hazard_rate = -log(1 - clip(pds, 0, 0.999)) / maturities)
}

# ---------------------------------------------------------------------------
# 11. PORTFOLIO CREDIT VaR
# ---------------------------------------------------------------------------
# One-factor credit model (Vasicek): default correlation via shared factor.
# P(default_i | factor Z) = Phi((Phi^{-1}(PD_i) - sqrt(rho)*Z) / sqrt(1-rho))

vasicek_portfolio_loss <- function(PDs, LGDs, notionals,
                                    rho = 0.15,
                                    n_sims = 50000L,
                                    conf = 0.99,
                                    seed = 12L) {
  set.seed(seed)
  N <- length(PDs)
  losses <- numeric(n_sims)

  for (sim in seq_len(n_sims)) {
    Z <- rnorm(1)   # systematic factor
    for (i in seq_len(N)) {
      cond_pd <- pnorm((qnorm(PDs[i]) - sqrt(rho) * Z) / sqrt(1 - rho))
      if (runif(1) < cond_pd) losses[sim] <- losses[sim] + LGDs[i] * notionals[i]
    }
  }
  list(
    EL       = mean(losses),
    CVaR99   = quantile(losses, conf),
    VaR99    = quantile(losses, conf),
    CVaR999  = quantile(losses, 0.999),
    max_loss = max(losses)
  )
}

# ---------------------------------------------------------------------------
# 12. MAIN DEMO
# ---------------------------------------------------------------------------

run_credit_risk_demo <- function() {
  cat("=== Credit Risk for Crypto Demo ===\n\n")

  # --- 1. Merton Model ---
  cat("--- 1. Merton Structural Model ---\n")
  panel <- data.frame(
    name    = c("ExchangeA","ExchangeB","ExchangeC"),
    E       = c(1e9, 5e8, 2e8),          # equity market cap
    D       = c(8e8, 6e8, 4e8),          # debt face value
    r       = c(0.05, 0.05, 0.05),
    T_mat   = c(1.0, 1.0, 1.0),
    sigma_E = c(0.60, 0.80, 1.20)        # high vol for crypto
  )
  res <- batch_merton(panel)
  print(res[, c("name","DD","PD","sigma_V")])

  # --- 2. KMV Rolling DD ---
  cat("\n--- 2. KMV Rolling Distance-to-Default ---\n")
  set.seed(42)
  eq_prices <- cumprod(1 + rnorm(400, 0.0005, 0.04)) * 1e9
  rdd <- rolling_dd(eq_prices, debt_face = 8e8, r = 0.05, window = 120L)
  valid <- !is.na(rdd$DD)
  cat(sprintf("  Mean DD (valid): %.3f  |  Mean PD: %.4f\n",
              mean(rdd$DD[valid]), mean(rdd$PD[valid])))

  # --- 3. Exchange Solvency Score ---
  cat("\n--- 3. Exchange Solvency Scoring ---\n")
  exchanges <- list(
    list(assets=10e9, liabilities=8e9, liquid=0.30, concentration=0.40),
    list(assets=5e9,  liabilities=4.8e9, liquid=0.10, concentration=0.60),
    list(assets=2e9,  liabilities=0.5e9, liquid=0.60, concentration=0.20)
  )
  for (i in seq_along(exchanges)) {
    ex <- exchanges[[i]]
    sc <- exchange_solvency_score(ex$assets, ex$liabilities, ex$liquid,
                                   concentration = ex$concentration)
    cat(sprintf("  Exchange %d: Score=%.1f  Label=%s\n", i, sc$score, sc$label))
  }

  # --- 4. Contagion Cascade ---
  cat("\n--- 4. Default Cascade Simulation ---\n")
  set.seed(77)
  N   <- 8L
  eq_vec <- runif(N, 1e8, 5e8)
  exp_mat <- matrix(runif(N*N, 0, 1e8), N, N)
  diag(exp_mat) <- 0
  casc <- simulate_default_cascade(eq_vec, exp_mat, loss_fraction=0.6,
                                    initial_default=1L, n_sims=500L)
  cat(sprintf("  Mean cascade: %.2f  |  P(systemic): %.3f\n",
              casc$mean_cascade, casc$systemic_risk))

  # --- 5. CVA ---
  cat("\n--- 5. Credit Valuation Adjustment ---\n")
  ee    <- exposure_profile_perp(notional=1e6, sigma_daily=0.03, T_steps=252L, n_paths=500L)
  pd_m  <- marginal_pd(hazard_rate=0.10, T_steps=252L)
  cva   <- compute_cva(ee, pd_m, lgd=0.60)
  cat(sprintf("  CVA on $1M notional perp (hazard=10%%): $%.0f  (%.2f bps)\n",
              cva, cva / 1e6 * 1e4))

  # --- 6. Wrong-Way Risk ---
  cat("\n--- 6. Wrong-Way Risk in Crypto Lending ---\n")
  wwr <- simulate_wwr(notional=1e6, sigma_collateral=0.80, rho_wwr=0.7,
                       hazard_rate=0.10, n_sims=5000L)
  cat(sprintf("  CVA no-WWR: $%.0f  CVA with WWR: $%.0f  Multiplier: %.2fx\n",
              wwr$cva_no_wwr, wwr$cva_wwr, wwr$wwr_multiplier))

  # --- 7. Migration Matrix ---
  cat("\n--- 7. Migration Matrix ---\n")
  set.seed(9)
  T_  <- 24L; Nc <- 20L
  scores_panel <- matrix(clip(50 + cumsum(rnorm(T_*Nc, 0, 5)), 0, 100),
                          T_, Nc)
  mig <- estimate_migration_matrix(scores_panel)
  cat("  Migration matrix (row=from, col=to):\n")
  print(round(mig$migration, 3))

  # --- 8. Credit-Adjusted Sizing ---
  cat("\n--- 8. Credit-Adjusted Position Sizing ---\n")
  pos_raw <- c(0.30, 0.25, 0.20)
  pds_vec <- c(0.02, 0.08, 0.25)
  cred <- portfolio_credit_sizing(pos_raw, pds_vec, max_leverage=1.0)
  print(cred)

  # --- 9. Portfolio Credit VaR ---
  cat("\n--- 9. Portfolio Credit VaR (Vasicek one-factor) ---\n")
  N_ent <- 10L
  pds_v <- runif(N_ent, 0.02, 0.30)
  lgd_v <- rep(0.60, N_ent)
  not_v <- runif(N_ent, 1e7, 1e8)
  pvd <- vasicek_portfolio_loss(pds_v, lgd_v, not_v, rho=0.20, n_sims=20000L)
  cat(sprintf("  EL=$%.0fK  VaR99=$%.0fK  CVaR999=$%.0fK\n",
              pvd$EL/1e3, pvd$CVaR99/1e3, pvd$CVaR999/1e3))

  cat("\nDone.\n")
  invisible(list(merton=res, mig=mig, casc=casc, wwr=wwr, cva=cva))
}

if (interactive()) {
  credit_results <- run_credit_risk_demo()
}

# ---------------------------------------------------------------------------
# 13. REDUCED-FORM CREDIT MODEL (INTENSITY / COX PROCESS)
# ---------------------------------------------------------------------------
# Default as the first jump of a Cox process with stochastic hazard rate.
# lambda(t) = lambda_0 + beta * X(t), where X(t) is a risk factor.
# Financial intuition: exchange solvency is influenced by market-wide stress;
# the Cox process captures this time-varying default intensity.

simulate_cox_default <- function(T_ = 252L,
                                  lambda0 = 0.05 / 252,
                                  beta    = 0.5,
                                  X_path  = NULL,
                                  n_sims  = 5000L, seed = 42L) {
  set.seed(seed)
  if (is.null(X_path)) X_path <- pmax(cumsum(rnorm(T_, 0, 0.02)) + 0.5, 0)
  dt <- 1 / 252
  default_times <- integer(n_sims)

  for (sim in seq_len(n_sims)) {
    survived <- TRUE
    for (t in seq_len(T_)) {
      lambda_t <- lambda0 + beta * pmax(X_path[t], 0)
      if (runif(1) < lambda_t * dt) {
        default_times[sim] <- t; survived <- FALSE; break
      }
    }
    if (survived) default_times[sim] <- T_ + 1L
  }

  pd_horizon <- sapply(seq_len(T_), function(t) mean(default_times <= t))
  list(default_times = default_times,
       pd_horizon    = pd_horizon,
       pd_1year      = mean(default_times <= 252L),
       median_survival = median(default_times))
}

# ---------------------------------------------------------------------------
# 14. COUNTERPARTY RISK NETTING
# ---------------------------------------------------------------------------
# In a bilateral netting arrangement, gross exposures are replaced by the
# net exposure: max(sum(MtM), 0). Reduces CVA significantly.

gross_vs_net_exposure <- function(mtm_values) {
  gross_pos <- sum(pmax(mtm_values, 0))
  gross_neg <- sum(pmin(mtm_values, 0))
  net_exp   <- max(sum(mtm_values), 0)
  data.frame(
    gross_positive = gross_pos,
    gross_negative = abs(gross_neg),
    gross_total    = gross_pos + abs(gross_neg),
    net_exposure   = net_exp,
    netting_benefit = gross_pos - net_exp,
    netting_ratio   = 1 - net_exp / max(gross_pos, 1e-8)
  )
}

# ---------------------------------------------------------------------------
# 15. CREDIT STRESS TEST
# ---------------------------------------------------------------------------
# Scenario: simultaneous shock to (1) collateral values, (2) funding spreads,
# (3) default correlation spike. How does portfolio loss change?

stress_test_credit <- function(PDs_base, LGDs, notionals,
                                scenarios = list(
                                  "Base"    = list(pd_mult=1.0, lgd_mult=1.0, rho=0.10),
                                  "Stress"  = list(pd_mult=3.0, lgd_mult=1.2, rho=0.40),
                                  "Severe"  = list(pd_mult=6.0, lgd_mult=1.5, rho=0.70)
                                ),
                                n_sims = 20000L, seed = 42L) {
  results <- lapply(names(scenarios), function(nm) {
    sc   <- scenarios[[nm]]
    pds  <- clip(PDs_base * sc$pd_mult, 0, 0.999)
    lgds <- clip(LGDs * sc$lgd_mult, 0, 1)
    set.seed(seed)
    N    <- length(pds); losses <- numeric(n_sims)
    for (sim in seq_len(n_sims)) {
      Z <- rnorm(1)
      for (i in seq_len(N)) {
        cpd <- pnorm((qnorm(pds[i]) - sqrt(sc$rho)*Z) / sqrt(1-sc$rho))
        if (runif(1) < cpd) losses[sim] <- losses[sim] + lgds[i]*notionals[i]
      }
    }
    data.frame(scenario=nm, EL=mean(losses), VaR99=quantile(losses,0.99),
               CVaR99=mean(losses[losses>quantile(losses,0.99)]),
               max_loss=max(losses))
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 16. CREDIT RISK EXTENDED DEMO
# ---------------------------------------------------------------------------

run_credit_extended_demo <- function() {
  cat("=== Credit Risk Extended Demo ===\n\n")

  cat("--- 1. Cox Process Default Simulation ---\n")
  set.seed(42)
  X_risk <- pmax(cumsum(rnorm(252, 0, 0.015)) + 0.3, 0)
  cox_res <- simulate_cox_default(T_=252L, lambda0=0.02/252, beta=0.5,
                                    X_path=X_risk, n_sims=2000L)
  cat(sprintf("  1-year PD: %.4f  Median survival: %d bars\n",
              cox_res$pd_1year, cox_res$median_survival))

  cat("\n--- 2. Netting Agreement ---\n")
  mtm_vals <- rnorm(20, 0, 1e6)
  net <- gross_vs_net_exposure(mtm_vals)
  cat(sprintf("  Gross exposure: $%.0fK  Net: $%.0fK  Netting benefit: %.1f%%\n",
              net$gross_positive/1e3, net$net_exposure/1e3,
              net$netting_ratio*100))

  cat("\n--- 3. Credit Stress Test ---\n")
  pds  <- runif(5, 0.02, 0.15)
  lgds <- rep(0.60, 5)
  nots <- runif(5, 1e7, 5e7)
  stress <- stress_test_credit(pds, lgds, nots, n_sims=10000L)
  print(stress[, c("scenario","EL","VaR99","CVaR99")])

  cat("\n--- 4. Migration Matrix Simulation (24-month) ---\n")
  mig_mat <- matrix(c(0.90,0.07,0.02,0.01, 0.04,0.85,0.08,0.03,
                       0.01,0.05,0.80,0.14, 0.00,0.00,0.00,1.00),
                     4, 4, byrow=TRUE)
  rownames(mig_mat) <- colnames(mig_mat) <- c("BBB","BB","B","D")
  sim_mig <- simulate_migration(mig_mat, initial_state=2L, T_=24L, n_sims=2000L)
  cat("  2-year default probabilities (by month):\n")
  cat("  ", round(sim_mig$pd_horizon[c(3,6,12,18,24)], 4), "\n")
  cat("  (months 3,6,12,18,24)\n")

  invisible(list(cox=cox_res, net=net, stress=stress))
}

if (interactive()) {
  credit_ext <- run_credit_extended_demo()
}

# ---------------------------------------------------------------------------
# 17. CREDIT DEFAULT SWAP (CDS) PRICING
# ---------------------------------------------------------------------------
# CDS spread = sum of protection leg discounted PD / sum of premium leg.
# Financial intuition: buying a CDS on an exchange is like buying insurance
# against that exchange's insolvency; the spread reflects market-implied PD.

cds_price <- function(hazard_rates, recovery=0.4, r=0.05, T_=1.0, freq=4L) {
  # Quarterly payment dates
  times    <- seq(1/freq, T_, by=1/freq)
  dt       <- 1/freq
  surv_prob <- exp(-cumsum(hazard_rates[1:length(times)]) * dt)
  df        <- exp(-r * times)

  # Premium leg: pay spread * dt each period if survived
  prem_leg  <- sum(surv_prob * df * dt)
  # Protection leg: pay (1-recovery) * PD
  pd_marg   <- c(1-surv_prob[1], -diff(surv_prob))
  prot_leg  <- sum(pd_marg * (1-recovery) * df)

  # Fair CDS spread
  spread <- prot_leg / max(prem_leg, 1e-12)
  list(spread=spread, prem_leg=prem_leg, prot_leg=prot_leg,
       survival_probs=surv_prob)
}

#' CDS spread term structure from constant hazard rate
cds_term_structure <- function(hazard_rate, maturities=c(0.5,1,2,3,5),
                                recovery=0.4, r=0.05) {
  results <- lapply(maturities, function(T_) {
    hz <- rep(hazard_rate, as.integer(T_*4))
    cs <- cds_price(hz, recovery, r, T_)
    data.frame(maturity=T_, spread_bps=cs$spread*1e4, pd=1-tail(cs$survival_probs,1))
  })
  do.call(rbind, results)
}

# ---------------------------------------------------------------------------
# 18. SURVIVAL ANALYSIS FOR CRYPTO ENTITIES
# ---------------------------------------------------------------------------
# Kaplan-Meier estimator for entity survival.

kaplan_meier <- function(times, events) {
  # times: observed survival times; events: 1=default, 0=censored
  unique_times <- sort(unique(times[events==1]))
  surv <- 1.0; km <- numeric(length(unique_times))
  for (i in seq_along(unique_times)) {
    t_i <- unique_times[i]
    n_i <- sum(times >= t_i)
    d_i <- sum(times == t_i & events == 1)
    surv <- surv * (1 - d_i / max(n_i, 1))
    km[i] <- surv
  }
  data.frame(time=unique_times, survival=km, hazard=1-km)
}

# ---------------------------------------------------------------------------
# 19. PORTFOLIO CREDIT LOSS DISTRIBUTION MOMENTS
# ---------------------------------------------------------------------------

credit_loss_moments <- function(PDs, LGDs, notionals, rho=0.10) {
  n   <- length(PDs)
  EL  <- sum(PDs * LGDs * notionals)
  # Variance from one-factor model
  var_idio <- sum((PDs * (1-PDs)) * (LGDs * notionals)^2)
  cov_sys  <- 0
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i==j) next
      pd_cov <- rho * sqrt(PDs[i]*(1-PDs[i])) * sqrt(PDs[j]*(1-PDs[j]))
      cov_sys <- cov_sys + pd_cov * LGDs[i]*notionals[i] * LGDs[j]*notionals[j]
    }
  }
  var_total <- var_idio + cov_sys
  data.frame(EL=EL, variance=var_total, sd=sqrt(max(var_total,0)),
             UL=sqrt(max(var_total,0))*2.33,   # 99% unexpected loss
             EL_pct=EL/sum(notionals))
}

# ---------------------------------------------------------------------------
# 20. CREDIT RISK FINAL DEMO
# ---------------------------------------------------------------------------

run_credit_final_demo <- function() {
  cat("=== Credit Risk Final Demo ===\n\n")

  cat("--- CDS Pricing ---\n")
  hz <- rep(0.05/4, 20)   # flat hazard, quarterly
  cds <- cds_price(hz, recovery=0.4, r=0.05, T_=5.0)
  cat(sprintf("  5-year CDS spread: %.0f bps\n", cds$spread*1e4))
  cat(sprintf("  5-year survival probability: %.4f\n", tail(cds$survival_probs,1)))

  cat("\n--- CDS Term Structure ---\n")
  ts_cds <- cds_term_structure(hazard_rate=0.05, maturities=c(0.5,1,2,5))
  print(ts_cds)

  cat("\n--- Kaplan-Meier Survival ---\n")
  set.seed(42)
  n_entities <- 50L
  times  <- rexp(n_entities, rate=0.10)
  events <- rbinom(n_entities, 1, 0.7)
  km     <- kaplan_meier(times, events)
  cat("  KM survival at time 5:", round(km$survival[km$time<=5][length(km$survival[km$time<=5])],4), "\n")
  cat("  Number of default times observed:", sum(events), "\n")

  cat("\n--- Credit Loss Moments ---\n")
  pds  <- runif(8, 0.02, 0.20)
  lgds <- rep(0.60, 8)
  nots <- runif(8, 5e6, 50e6)
  clm  <- credit_loss_moments(pds, lgds, nots, rho=0.15)
  cat(sprintf("  EL=$%.0fK  UL(99%%)=$%.0fK  EL%%=%.2f%%\n",
              clm$EL/1e3, clm$UL/1e3, clm$EL_pct*100))

  cat("\n--- Wrong-Way Risk Sensitivity to Correlation ---\n")
  rho_grid <- c(0.0, 0.3, 0.5, 0.7, 0.9)
  for (rho_ in rho_grid) {
    wwr <- simulate_wwr(notional=1e6, sigma_collateral=0.80,
                         rho_wwr=rho_, hazard_rate=0.10, n_sims=2000L)
    cat(sprintf("  rho=%.1f  WWR multiplier=%.3f\n", rho_, wwr$wwr_multiplier))
  }

  invisible(list(cds=cds, ts_cds=ts_cds, km=km, clm=clm))
}

if (interactive()) {
  credit_final <- run_credit_final_demo()
}

# =============================================================================
# SECTION: COUNTERPARTY CREDIT RISK — POTENTIAL FUTURE EXPOSURE
# =============================================================================
# PFE is the high quantile of exposure distribution at a future time horizon.
# For a crypto derivative book, simulate paths and take the 95th percentile.

simulate_pfe <- function(S0, sigma, r, T_years, n_paths = 2000, n_steps = 50,
                          alpha = 0.95) {
  dt   <- T_years / n_steps
  S    <- matrix(NA_real_, n_paths, n_steps + 1)
  S[, 1] <- S0
  for (j in seq_len(n_steps))
    S[, j+1] <- S[, j] * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*rnorm(n_paths))
  # Mark-to-market value at each step (long call, K=S0)
  bs_call <- function(St, tau) {
    if (tau < 1e-6) return(pmax(St - S0, 0))
    d1 <- (log(St/S0) + (r + 0.5*sigma^2)*tau) / (sigma*sqrt(tau))
    d2 <- d1 - sigma*sqrt(tau)
    St * pnorm(d1) - S0 * exp(-r*tau) * pnorm(d2)
  }
  exposure <- matrix(NA_real_, n_paths, n_steps + 1)
  for (j in seq(0, n_steps)) {
    tau <- (n_steps - j) * dt
    exposure[, j+1] <- pmax(bs_call(S[, j+1], tau), 0)
  }
  pfe_profile <- apply(exposure, 2, quantile, alpha)
  epfe        <- apply(exposure, 2, mean)
  list(pfe = pfe_profile, epfe = epfe, t_grid = seq(0, T_years, length.out = n_steps+1))
}

# =============================================================================
# SECTION: SIMPLIFIED BASEL III CAPITAL FOR CRYPTO EXPOSURES
# =============================================================================
# Under Basel III standardised approach, crypto assets may attract 100-1250%
# risk weight. Compute RWA and required capital.

basel_rwa_crypto <- function(exposure, risk_weight = 1.0, capital_ratio = 0.08) {
  # exposure: notional in USD
  # risk_weight: 1.0 = 100%, 12.5 = 1250% (Group 2b)
  rwa     <- exposure * risk_weight
  capital <- rwa * capital_ratio
  list(rwa = rwa, capital_required = capital,
       leverage_ratio = capital / (exposure + 1e-10))
}

# =============================================================================
# SECTION: WRONG-WAY RISK METRICS
# =============================================================================
# Compute the correlation between counterparty default probability and
# exposure to quantify wrong-way risk at portfolio level.

wwr_correlation <- function(default_indicators, exposures) {
  # Both vectors length T (simulated paths)
  if (sd(default_indicators) < 1e-9 || sd(exposures) < 1e-9) return(0)
  cor(default_indicators, exposures)
}

# =============================================================================
# SECTION: CREDIT VaR DECOMPOSITION
# =============================================================================
# Decompose portfolio credit VaR into systematic and idiosyncratic components
# using the Vasicek model's closed-form analytic results.

vasicek_credit_var_decomp <- function(pd_vec, rho_vec, LGD = 0.45,
                                       confidence = 0.999) {
  # pd_vec: vector of individual PDs
  # rho_vec: asset correlations with systematic factor
  n   <- length(pd_vec)
  # Systematic VaR (one-factor): conditional loss at confidence level
  Z   <- qnorm(confidence)
  conditional_loss <- mapply(function(pd, rho) {
    q_cond <- (qnorm(pd) - sqrt(rho) * Z) / sqrt(1 - rho)
    LGD * pnorm(q_cond)
  }, pd_vec, rho_vec)
  expected_loss <- pd_vec * LGD
  credit_var    <- sum(conditional_loss) - sum(expected_loss)
  list(
    conditional_loss = conditional_loss,
    expected_loss    = expected_loss,
    credit_var       = credit_var,
    unexpected_loss  = credit_var
  )
}

# =============================================================================
# SECTION: CREDIT RISK FINAL EXTENSION DEMO
# =============================================================================

run_credit_extension_demo <- function() {
  set.seed(99)

  cat("--- Potential Future Exposure (PFE) ---\n")
  pfe_res <- simulate_pfe(S0 = 50000, sigma = 0.6, r = 0.05,
                           T_years = 0.5, n_paths = 500)
  cat("Peak PFE:", round(max(pfe_res$pfe), 2), "USD\n")
  cat("Peak EPFE:", round(max(pfe_res$epfe), 2), "USD\n")

  cat("\n--- Basel III RWA ---\n")
  b3 <- basel_rwa_crypto(exposure = 1e6, risk_weight = 1.0)
  cat("RWA:", round(b3$rwa, 0), "  Capital:", round(b3$capital_required, 0), "\n")
  b3b <- basel_rwa_crypto(exposure = 1e6, risk_weight = 12.5)
  cat("Group 2b RWA:", round(b3b$rwa, 0), "  Capital:", round(b3b$capital_required, 0), "\n")

  cat("\n--- WWR Correlation ---\n")
  set.seed(5)
  def_ind <- rbinom(1000, 1, 0.05)
  exp_val <- pmax(rnorm(1000, 100, 30) + 50 * def_ind, 0)
  cat("WWR correlation:", round(wwr_correlation(def_ind, exp_val), 4), "\n")

  cat("\n--- Credit VaR Decomposition ---\n")
  cvd <- vasicek_credit_var_decomp(
    pd_vec  = c(0.01, 0.02, 0.05, 0.10),
    rho_vec = c(0.12, 0.15, 0.20, 0.30)
  )
  cat("Expected loss:", round(sum(cvd$expected_loss), 4), "\n")
  cat("Credit VaR:",    round(cvd$credit_var, 4), "\n")

  invisible(list(pfe=pfe_res, rwa=b3, cvd=cvd))
}

if (interactive()) {
  credit_ext2 <- run_credit_extension_demo()
}
