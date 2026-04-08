###############################################################################
# portfolio_risk.R
# Portfolio Risk Analytics Library in R
# VaR, CVaR, Component Risk, Stress Testing, Factor Risk, Drawdown,
# Tail Risk, Copula, Backtesting, Liquidity, Credit, Systemic, Risk Budgeting
###############################################################################

# =============================================================================
# SECTION 1: VALUE AT RISK (VaR)
# =============================================================================

#' Historical VaR
#'
#' @param returns numeric vector of returns
#' @param alpha confidence level (e.g. 0.95 or 0.99)
#' @param method "empirical" or "weighted"
#' @param decay exponential decay for weighted (lambda)
#' @return VaR estimate (positive number representing loss)
#' @export
var_historical <- function(returns, alpha = 0.95, method = "empirical",
                            decay = 0.94) {
  n <- length(returns)

  if (method == "empirical") {
    q <- quantile(returns, probs = 1 - alpha, type = 7)
    var_val <- -as.numeric(q)
  } else {
    # Exponentially weighted
    weights <- decay^((n - 1):0)
    weights <- weights / sum(weights)
    ord <- order(returns)
    sorted_returns <- returns[ord]
    sorted_weights <- weights[ord]
    cum_weights <- cumsum(sorted_weights)
    idx <- which(cum_weights >= (1 - alpha))[1]
    var_val <- -sorted_returns[idx]
  }

  structure(
    list(
      var = var_val,
      alpha = alpha,
      method = paste("historical", method),
      n = n
    ),
    class = "var_result"
  )
}

#' Parametric VaR (Normal)
#'
#' @param returns numeric vector of returns
#' @param alpha confidence level
#' @return VaR estimate
#' @export
var_parametric_normal <- function(returns, alpha = 0.95) {
  mu <- mean(returns)
  sigma <- sd(returns)
  z <- qnorm(alpha)
  var_val <- -(mu - z * sigma)

  structure(
    list(
      var = var_val,
      alpha = alpha,
      mu = mu,
      sigma = sigma,
      z = z,
      method = "parametric_normal"
    ),
    class = "var_result"
  )
}

#' Parametric VaR (Student-t)
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @param nu degrees of freedom (NULL for MLE)
#' @return VaR estimate
#' @export
var_parametric_t <- function(returns, alpha = 0.95, nu = NULL) {
  mu <- mean(returns)
  sigma <- sd(returns)

  if (is.null(nu)) {
    # MLE for t distribution
    t_loglik <- function(par) {
      nu_try <- par[1]
      if (nu_try <= 2) return(1e10)
      s <- sigma * sqrt((nu_try - 2) / nu_try)
      -sum(dt((returns - mu) / s, df = nu_try, log = TRUE) - log(s))
    }
    opt <- optimize(t_loglik, interval = c(2.1, 100))
    nu <- opt$minimum
  }

  t_quantile <- qt(1 - alpha, df = nu)
  s <- sigma * sqrt((nu - 2) / nu)
  var_val <- -(mu + s * t_quantile)

  structure(
    list(
      var = var_val,
      alpha = alpha,
      mu = mu,
      sigma = sigma,
      nu = nu,
      method = "parametric_t"
    ),
    class = "var_result"
  )
}

#' Cornish-Fisher VaR
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @return VaR with skewness/kurtosis adjustment
#' @export
var_cornish_fisher <- function(returns, alpha = 0.95) {
  mu <- mean(returns)
  sigma <- sd(returns)
  n <- length(returns)

  # Skewness and excess kurtosis
  s <- sum((returns - mu)^3) / (n * sigma^3)
  k <- sum((returns - mu)^4) / (n * sigma^4) - 3

  z <- qnorm(alpha)

  # Cornish-Fisher expansion
  z_cf <- z + (z^2 - 1) * s / 6 +
    (z^3 - 3 * z) * k / 24 -
    (2 * z^3 - 5 * z) * s^2 / 36

  var_val <- -(mu - z_cf * sigma)

  structure(
    list(
      var = var_val,
      alpha = alpha,
      mu = mu,
      sigma = sigma,
      skewness = s,
      excess_kurtosis = k,
      z_cf = z_cf,
      method = "cornish_fisher"
    ),
    class = "var_result"
  )
}

#' Monte Carlo VaR
#'
#' @param returns numeric vector or matrix of returns
#' @param alpha confidence level
#' @param n_sim number of simulations
#' @param weights portfolio weights (for multi-asset)
#' @param method "normal" or "bootstrap"
#' @return VaR estimate
#' @export
var_monte_carlo <- function(returns, alpha = 0.95, n_sim = 10000,
                             weights = NULL, method = "normal") {
  if (is.vector(returns)) {
    # Single asset
    if (method == "normal") {
      mu <- mean(returns)
      sigma <- sd(returns)
      sim_returns <- rnorm(n_sim, mean = mu, sd = sigma)
    } else {
      sim_returns <- sample(returns, n_sim, replace = TRUE)
    }
    var_val <- -quantile(sim_returns, probs = 1 - alpha)
  } else {
    # Multi-asset
    if (is.null(weights)) weights <- rep(1 / ncol(returns), ncol(returns))

    mu <- colMeans(returns)
    Sigma <- cov(returns)

    if (method == "normal") {
      # Cholesky decomposition
      L <- t(chol(Sigma))
      Z <- matrix(rnorm(n_sim * ncol(returns)), nrow = n_sim)
      sim_returns <- sweep(Z %*% t(L), 2, mu, "+")
    } else {
      idx <- sample(nrow(returns), n_sim, replace = TRUE)
      sim_returns <- returns[idx, ]
    }

    portfolio_returns <- sim_returns %*% weights
    var_val <- -quantile(portfolio_returns, probs = 1 - alpha)
  }

  structure(
    list(
      var = as.numeric(var_val),
      alpha = alpha,
      n_sim = n_sim,
      method = paste("monte_carlo", method)
    ),
    class = "var_result"
  )
}

#' EVT VaR using Generalized Pareto Distribution
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @param threshold_quantile quantile for threshold selection
#' @return EVT VaR
#' @export
var_evt <- function(returns, alpha = 0.99,
                     threshold_quantile = 0.10) {
  losses <- -returns
  n <- length(losses)

  # Set threshold
  u <- quantile(losses, probs = 1 - threshold_quantile)

  # Exceedances
  exceedances <- losses[losses > u] - u
  n_exceed <- length(exceedances)

  if (n_exceed < 10) {
    warning("Too few exceedances for EVT, using historical VaR")
    return(var_historical(returns, alpha))
  }

  # GPD MLE
  gpd_loglik <- function(params) {
    xi <- params[1]
    beta <- exp(params[2])

    if (beta <= 0) return(1e10)

    if (abs(xi) < 1e-7) {
      ll <- -n_exceed * log(beta) - sum(exceedances) / beta
    } else {
      z <- 1 + xi * exceedances / beta
      if (any(z <= 0)) return(1e10)
      ll <- -n_exceed * log(beta) - (1 + 1 / xi) * sum(log(z))
    }
    -ll
  }

  init <- c(0.1, log(mean(exceedances)))
  opt <- optim(init, gpd_loglik, method = "Nelder-Mead")

  xi <- opt$par[1]
  beta <- exp(opt$par[2])

  # VaR from GPD
  p_exceed <- n_exceed / n
  p_tail <- 1 - alpha

  if (abs(xi) < 1e-7) {
    var_val <- u + beta * log(p_exceed / p_tail)
  } else {
    var_val <- u + (beta / xi) * ((p_exceed / p_tail)^xi - 1)
  }

  structure(
    list(
      var = var_val,
      alpha = alpha,
      xi = xi,
      beta = beta,
      threshold = u,
      n_exceedances = n_exceed,
      method = "evt_gpd"
    ),
    class = "var_result"
  )
}

#' Print VaR result
#' @export
print.var_result <- function(x, ...) {
  cat("Value at Risk\n")
  cat("=============\n")
  cat("Method:", x$method, "\n")
  cat("Confidence level:", x$alpha * 100, "%\n")
  cat("VaR:", round(x$var, 6), "\n")
  invisible(x)
}

# =============================================================================
# SECTION 2: CVaR / EXPECTED SHORTFALL
# =============================================================================

#' Historical CVaR (Expected Shortfall)
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @return CVaR estimate
#' @export
cvar_historical <- function(returns, alpha = 0.95) {
  threshold <- quantile(returns, probs = 1 - alpha)
  tail_returns <- returns[returns <= threshold]

  if (length(tail_returns) == 0) {
    cvar_val <- -threshold
  } else {
    cvar_val <- -mean(tail_returns)
  }

  list(
    cvar = cvar_val,
    var = -as.numeric(threshold),
    alpha = alpha,
    n_tail = length(tail_returns),
    method = "historical"
  )
}

#' Parametric CVaR (Normal)
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @return CVaR
#' @export
cvar_parametric_normal <- function(returns, alpha = 0.95) {
  mu <- mean(returns)
  sigma <- sd(returns)
  z <- qnorm(1 - alpha)
  phi_z <- dnorm(z)

  cvar_val <- -(mu + sigma * phi_z / (1 - alpha))

  list(
    cvar = cvar_val,
    var = -(mu + sigma * qnorm(1 - alpha)),
    alpha = alpha,
    method = "parametric_normal"
  )
}

#' Parametric CVaR (Student-t)
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @param nu degrees of freedom
#' @return CVaR
#' @export
cvar_parametric_t <- function(returns, alpha = 0.95, nu = 5) {
  mu <- mean(returns)
  sigma <- sd(returns)
  s <- sigma * sqrt((nu - 2) / nu)

  t_q <- qt(1 - alpha, df = nu)
  dt_q <- dt(t_q, df = nu)

  cvar_val <- -(mu + s * (dt_q / (1 - alpha)) * (nu + t_q^2) / (nu - 1))

  list(
    cvar = cvar_val,
    alpha = alpha,
    nu = nu,
    method = "parametric_t"
  )
}

#' Monte Carlo CVaR
#'
#' @param returns numeric vector or matrix
#' @param alpha confidence level
#' @param n_sim number of simulations
#' @param weights portfolio weights
#' @return CVaR
#' @export
cvar_monte_carlo <- function(returns, alpha = 0.95, n_sim = 10000,
                              weights = NULL) {
  if (is.vector(returns)) {
    mu <- mean(returns)
    sigma <- sd(returns)
    sim <- rnorm(n_sim, mu, sigma)
  } else {
    if (is.null(weights)) weights <- rep(1 / ncol(returns), ncol(returns))
    mu <- colMeans(returns)
    Sigma <- cov(returns)
    L <- t(chol(Sigma))
    Z <- matrix(rnorm(n_sim * ncol(returns)), nrow = n_sim)
    sim_all <- sweep(Z %*% t(L), 2, mu, "+")
    sim <- as.numeric(sim_all %*% weights)
  }

  threshold <- quantile(sim, probs = 1 - alpha)
  cvar_val <- -mean(sim[sim <= threshold])

  list(
    cvar = cvar_val,
    var = -as.numeric(threshold),
    alpha = alpha,
    n_sim = n_sim,
    method = "monte_carlo"
  )
}

#' EVT CVaR
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @param threshold_quantile quantile for GPD threshold
#' @return EVT CVaR
#' @export
cvar_evt <- function(returns, alpha = 0.99, threshold_quantile = 0.10) {
  var_result <- var_evt(returns, alpha, threshold_quantile)
  xi <- var_result$xi
  beta <- var_result$beta

  # CVaR from GPD
  var_val <- var_result$var
  cvar_val <- var_val / (1 - xi) + (beta - xi * var_result$threshold) /
    (1 - xi)

  list(
    cvar = cvar_val,
    var = var_val,
    xi = xi,
    alpha = alpha,
    method = "evt_gpd"
  )
}

# =============================================================================
# SECTION 3: COMPONENT VaR
# =============================================================================

#' Component VaR (Euler Decomposition)
#'
#' Decomposes portfolio VaR into contributions from each position.
#'
#' @param returns matrix of asset returns (n x k)
#' @param weights portfolio weights
#' @param alpha confidence level
#' @return component VaR decomposition
#' @export
component_var <- function(returns, weights, alpha = 0.95) {
  if (is.vector(returns)) stop("returns must be a matrix")
  k <- ncol(returns)
  n <- nrow(returns)

  mu <- colMeans(returns)
  Sigma <- cov(returns)

  port_mu <- sum(weights * mu)
  port_sigma <- sqrt(as.numeric(t(weights) %*% Sigma %*% weights))
  z <- qnorm(alpha)

  # Total VaR
  total_var <- -(port_mu - z * port_sigma)

  # Marginal VaR: d(VaR)/d(w_i) = -mu_i + z * (Sigma %*% w)_i / sigma_p
  marginal_var <- -mu + z * as.numeric(Sigma %*% weights) / port_sigma

  # Component VaR = w_i * marginal_var_i
  comp_var <- weights * marginal_var

  # Percentage contributions
  pct_contribution <- comp_var / total_var * 100

  list(
    total_var = total_var,
    marginal_var = marginal_var,
    component_var = comp_var,
    pct_contribution = pct_contribution,
    alpha = alpha,
    port_mu = port_mu,
    port_sigma = port_sigma,
    weights = weights
  )
}

#' Incremental VaR
#'
#' VaR impact of adding a position to the portfolio.
#'
#' @param returns matrix of returns
#' @param weights current weights
#' @param new_weight weight of new position
#' @param new_returns returns of new asset
#' @param alpha confidence level
#' @return incremental VaR
#' @export
incremental_var <- function(returns, weights, new_weight, new_returns,
                             alpha = 0.95) {
  # Current portfolio VaR
  port_returns <- as.numeric(returns %*% weights)
  var_before <- -quantile(port_returns, probs = 1 - alpha)

  # Portfolio with new position
  # Scale down existing weights
  scale_factor <- 1 - new_weight
  new_weights_full <- c(weights * scale_factor, new_weight)
  returns_full <- cbind(returns, new_returns)
  port_returns_new <- as.numeric(returns_full %*% new_weights_full)
  var_after <- -quantile(port_returns_new, probs = 1 - alpha)

  list(
    var_before = as.numeric(var_before),
    var_after = as.numeric(var_after),
    incremental_var = as.numeric(var_after - var_before),
    alpha = alpha
  )
}

# =============================================================================
# SECTION 4: STRESS TESTING
# =============================================================================

#' Historical Stress Scenarios
#'
#' @param returns matrix of asset returns
#' @param weights portfolio weights
#' @param scenario named scenario
#' @return stress test results
#' @export
stress_historical <- function(returns, weights,
                               scenario = "2008_crisis") {
  # Pre-defined shock vectors (approximate)
  scenarios <- list(
    "2008_crisis" = list(
      name = "2008 Global Financial Crisis",
      equity = -0.40,
      bonds = 0.05,
      credit = -0.15,
      commodities = -0.35,
      fx_usd = 0.10,
      vol = 0.80,
      duration_months = 18
    ),
    "2020_covid" = list(
      name = "2020 COVID-19 Crash",
      equity = -0.34,
      bonds = 0.08,
      credit = -0.10,
      commodities = -0.25,
      fx_usd = 0.05,
      vol = 0.65,
      duration_months = 2
    ),
    "2022_rate_hike" = list(
      name = "2022 Rate Hiking Cycle",
      equity = -0.25,
      bonds = -0.15,
      credit = -0.08,
      commodities = 0.10,
      fx_usd = 0.15,
      vol = 0.20,
      duration_months = 12
    ),
    "1998_ltcm" = list(
      name = "1998 LTCM / Russian Crisis",
      equity = -0.20,
      bonds = 0.10,
      credit = -0.20,
      commodities = -0.15,
      fx_usd = 0.05,
      vol = 0.60,
      duration_months = 4
    ),
    "2011_euro_crisis" = list(
      name = "2011 European Sovereign Debt Crisis",
      equity = -0.22,
      bonds = 0.03,
      credit = -0.12,
      commodities = -0.10,
      fx_usd = 0.08,
      vol = 0.40,
      duration_months = 6
    )
  )

  if (is.character(scenario)) {
    if (!scenario %in% names(scenarios)) {
      stop("Unknown scenario. Available: ",
           paste(names(scenarios), collapse = ", "))
    }
    sc <- scenarios[[scenario]]
  } else {
    sc <- scenario
  }

  # Apply shocks to portfolio
  k <- ncol(returns)
  if (k <= length(sc) - 2) {
    shocks <- unlist(sc[3:(2 + k)])
  } else {
    shocks <- rep(sc$equity, k)
  }

  portfolio_loss <- sum(weights * shocks)

  # Simulate stressed returns
  Sigma <- cov(returns)
  stressed_mu <- colMeans(returns) + shocks / 252
  stressed_Sigma <- Sigma * (1 + sc$vol)

  list(
    scenario = sc$name,
    portfolio_loss = portfolio_loss,
    asset_shocks = shocks,
    stressed_port_return = portfolio_loss,
    duration_months = sc$duration_months,
    weights = weights
  )
}

#' Hypothetical Stress Scenarios
#'
#' @param returns matrix of returns
#' @param weights portfolio weights
#' @param rate_shock interest rate shock (bps)
#' @param equity_shock equity shock (fraction)
#' @param vol_shock volatility shock (fraction)
#' @param spread_shock credit spread shock (bps)
#' @param fx_shock FX shock (fraction)
#' @return stress test results
#' @export
stress_hypothetical <- function(returns, weights,
                                 rate_shock = 0,
                                 equity_shock = 0,
                                 vol_shock = 0,
                                 spread_shock = 0,
                                 fx_shock = 0) {
  k <- ncol(returns)

  # Estimate sensitivities from data
  mu <- colMeans(returns)
  sigma <- apply(returns, 2, sd)

  # Simple linear mapping of shocks to returns
  # In practice, would use factor model sensitivities
  shocks <- numeric(k)

  # Apply equity shock uniformly (simplified)
  if (equity_shock != 0) {
    shocks <- shocks + equity_shock * sigma / max(sigma)
  }

  # Rate shock: affects all assets via duration
  if (rate_shock != 0) {
    rate_sensitivity <- -sigma * 0.5  # Simplified
    shocks <- shocks + rate_shock / 10000 * rate_sensitivity
  }

  # Vol shock: affects via vega-like sensitivity
  if (vol_shock != 0) {
    shocks <- shocks - vol_shock * sigma * 0.1
  }

  portfolio_loss <- sum(weights * shocks)

  list(
    portfolio_loss = portfolio_loss,
    asset_shocks = shocks,
    rate_shock_bps = rate_shock,
    equity_shock = equity_shock,
    vol_shock = vol_shock,
    spread_shock_bps = spread_shock,
    fx_shock = fx_shock,
    weights = weights
  )
}

#' Comprehensive Stress Test Suite
#'
#' Runs multiple scenarios and reports results.
#'
#' @param returns matrix of returns
#' @param weights portfolio weights
#' @return stress test report
#' @export
stress_test_suite <- function(returns, weights) {
  historical_scenarios <- c("2008_crisis", "2020_covid",
                             "2022_rate_hike", "1998_ltcm",
                             "2011_euro_crisis")

  historical_results <- lapply(historical_scenarios, function(sc) {
    stress_historical(returns, weights, scenario = sc)
  })

  hypothetical_scenarios <- list(
    list(name = "Rate +200bps", rate_shock = 200),
    list(name = "Rate -100bps", rate_shock = -100),
    list(name = "Equity -20%", equity_shock = -0.20),
    list(name = "Equity -40%", equity_shock = -0.40),
    list(name = "Vol +50%", vol_shock = 0.50),
    list(name = "Combined: Rate+200, Equity-20%",
         rate_shock = 200, equity_shock = -0.20),
    list(name = "Tail event: 5-sigma",
         equity_shock = -5 * sd(returns %*% weights))
  )

  hypothetical_results <- lapply(hypothetical_scenarios, function(sc) {
    result <- stress_hypothetical(returns, weights,
                                   rate_shock = sc$rate_shock %||% 0,
                                   equity_shock = sc$equity_shock %||% 0,
                                   vol_shock = sc$vol_shock %||% 0)
    result$name <- sc$name
    result
  })

  # Summary
  hist_losses <- sapply(historical_results, function(r) r$portfolio_loss)
  hyp_losses <- sapply(hypothetical_results, function(r) r$portfolio_loss)

  list(
    historical = historical_results,
    hypothetical = hypothetical_results,
    worst_historical = min(hist_losses),
    worst_hypothetical = min(hyp_losses),
    summary = data.frame(
      Scenario = c(sapply(historical_results, function(r) r$scenario),
                    sapply(hypothetical_results, function(r) r$name)),
      Loss = c(hist_losses, hyp_losses)
    )
  )
}

#' Null coalesce operator
#' @keywords internal
`%||%` <- function(x, y) if (is.null(x)) y else x

# =============================================================================
# SECTION 5: FACTOR RISK
# =============================================================================

#' PCA-Based Factor Risk Decomposition
#'
#' @param returns matrix of asset returns
#' @param weights portfolio weights
#' @param n_factors number of factors
#' @return factor risk decomposition
#' @export
factor_risk_pca <- function(returns, weights, n_factors = 3) {
  n <- nrow(returns)
  k <- ncol(returns)

  # PCA on returns
  pca <- prcomp(returns, center = TRUE, scale. = FALSE)
  loadings <- pca$rotation[, 1:n_factors]
  factors <- pca$x[, 1:n_factors]
  eigenvalues <- pca$sdev^2

  # Factor variance
  factor_var <- eigenvalues[1:n_factors]
  total_var <- sum(eigenvalues)
  var_explained <- factor_var / total_var

  # Portfolio factor exposures
  port_exposures <- as.numeric(t(weights) %*% loadings)

  # Factor contribution to portfolio variance
  port_var <- as.numeric(t(weights) %*% cov(returns) %*% weights)
  factor_contrib <- port_exposures^2 * factor_var
  specific_risk <- port_var - sum(factor_contrib)

  list(
    factor_exposures = port_exposures,
    factor_variance = factor_var,
    factor_contribution = factor_contrib,
    specific_risk = max(0, specific_risk),
    total_risk = port_var,
    pct_systematic = sum(factor_contrib) / port_var * 100,
    pct_specific = max(0, specific_risk) / port_var * 100,
    var_explained = var_explained,
    loadings = loadings,
    n_factors = n_factors
  )
}

#' Specified Factor Model Risk
#'
#' @param returns matrix of asset returns
#' @param factors matrix of factor returns
#' @param weights portfolio weights
#' @return factor model risk decomposition
#' @export
factor_risk_specified <- function(returns, factors, weights) {
  if (!is.matrix(factors)) factors <- as.matrix(factors)
  k <- ncol(returns)
  n_factors <- ncol(factors)

  # Regress each asset on factors
  betas <- matrix(0, nrow = k, ncol = n_factors)
  specific_var <- numeric(k)

  for (i in seq_len(k)) {
    fit <- lm(returns[, i] ~ factors)
    betas[i, ] <- coef(fit)[-1]
    specific_var[i] <- var(residuals(fit))
  }

  # Factor covariance
  factor_cov <- cov(factors)

  # Portfolio factor exposure
  port_beta <- as.numeric(t(weights) %*% betas)

  # Systematic risk
  systematic_var <- as.numeric(t(port_beta) %*% factor_cov %*% port_beta)

  # Specific (idiosyncratic) risk
  spec_var <- sum((weights^2) * specific_var)

  # Total risk
  total_var <- systematic_var + spec_var

  # Factor contributions
  factor_marginal <- as.numeric(factor_cov %*% port_beta) * port_beta
  factor_pct <- factor_marginal / total_var * 100

  list(
    betas = betas,
    port_beta = port_beta,
    factor_cov = factor_cov,
    systematic_var = systematic_var,
    specific_var = spec_var,
    total_var = total_var,
    systematic_pct = systematic_var / total_var * 100,
    specific_pct = spec_var / total_var * 100,
    factor_contribution = factor_marginal,
    factor_pct = factor_pct
  )
}

# =============================================================================
# SECTION 6: DRAWDOWN ANALYSIS
# =============================================================================

#' Maximum Drawdown
#'
#' @param returns numeric vector of returns
#' @return drawdown analysis
#' @export
max_drawdown <- function(returns) {
  n <- length(returns)
  cum_returns <- cumprod(1 + returns)
  hwm <- cummax(cum_returns)
  drawdowns <- cum_returns / hwm - 1

  max_dd <- min(drawdowns)
  max_dd_end <- which.min(drawdowns)
  max_dd_start <- which.max(cum_returns[1:max_dd_end])

  # Recovery
  if (max_dd_end < n) {
    recovery <- which(cum_returns[(max_dd_end + 1):n] >= hwm[max_dd_end])
    if (length(recovery) > 0) {
      recovery_idx <- max_dd_end + recovery[1]
    } else {
      recovery_idx <- NA
    }
  } else {
    recovery_idx <- NA
  }

  # Drawdown duration
  duration <- max_dd_end - max_dd_start
  recovery_duration <- if (!is.na(recovery_idx)) recovery_idx - max_dd_end else NA

  # All drawdowns
  in_drawdown <- drawdowns < 0
  dd_periods <- rle(in_drawdown)

  list(
    max_drawdown = max_dd,
    start = max_dd_start,
    trough = max_dd_end,
    recovery = recovery_idx,
    duration = duration,
    recovery_duration = recovery_duration,
    drawdown_series = drawdowns,
    cumulative_returns = cum_returns,
    hwm = hwm
  )
}

#' Average Drawdown
#'
#' @param returns numeric vector
#' @param top_n average of top N drawdowns
#' @return average drawdown
#' @export
average_drawdown <- function(returns, top_n = 5) {
  cum_returns <- cumprod(1 + returns)
  hwm <- cummax(cum_returns)
  drawdowns <- cum_returns / hwm - 1

  # Find distinct drawdown episodes
  in_dd <- drawdowns < 0
  episodes <- list()
  current_dd <- NULL

  for (i in seq_along(drawdowns)) {
    if (in_dd[i]) {
      if (is.null(current_dd)) {
        current_dd <- list(start = i, values = drawdowns[i])
      } else {
        current_dd$values <- c(current_dd$values, drawdowns[i])
      }
    } else {
      if (!is.null(current_dd)) {
        current_dd$min_dd <- min(current_dd$values)
        current_dd$duration <- length(current_dd$values)
        episodes <- c(episodes, list(current_dd))
        current_dd <- NULL
      }
    }
  }
  if (!is.null(current_dd)) {
    current_dd$min_dd <- min(current_dd$values)
    current_dd$duration <- length(current_dd$values)
    episodes <- c(episodes, list(current_dd))
  }

  dd_magnitudes <- sapply(episodes, function(e) e$min_dd)
  dd_durations <- sapply(episodes, function(e) e$duration)

  ord <- order(dd_magnitudes)
  top_dds <- dd_magnitudes[ord[seq_len(min(top_n, length(ord)))]]

  list(
    average_dd = mean(top_dds),
    top_drawdowns = top_dds,
    n_episodes = length(episodes),
    avg_duration = mean(dd_durations),
    max_duration = max(dd_durations)
  )
}

#' Conditional Drawdown at Risk (CDaR)
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @return CDaR
#' @export
cdar <- function(returns, alpha = 0.95) {
  cum_returns <- cumprod(1 + returns)
  hwm <- cummax(cum_returns)
  drawdowns <- -(cum_returns / hwm - 1)  # Positive drawdowns

  var_dd <- quantile(drawdowns, probs = alpha)
  cdar_val <- mean(drawdowns[drawdowns >= var_dd])

  list(
    cdar = cdar_val,
    var_dd = as.numeric(var_dd),
    alpha = alpha,
    mean_dd = mean(drawdowns),
    max_dd = max(drawdowns)
  )
}

#' Underwater Equity Curve
#'
#' @param returns numeric vector
#' @return underwater curve data
#' @export
underwater_curve <- function(returns) {
  cum_returns <- cumprod(1 + returns)
  hwm <- cummax(cum_returns)
  underwater <- cum_returns / hwm - 1

  list(
    underwater = underwater,
    cum_returns = cum_returns,
    hwm = hwm,
    pct_time_underwater = mean(underwater < 0) * 100,
    current_drawdown = tail(underwater, 1)
  )
}

# =============================================================================
# SECTION 7: TAIL RISK
# =============================================================================

#' Hill Estimator for Tail Index
#'
#' @param x numeric vector (losses, positive)
#' @param k number of upper order statistics
#' @return Hill estimate of tail index
#' @export
hill_estimator <- function(x, k = NULL) {
  x_sorted <- sort(x, decreasing = TRUE)
  n <- length(x)

  if (is.null(k)) {
    k <- floor(sqrt(n))
  }

  # Hill estimator for different k
  hill_estimates <- numeric(k)
  for (j in 1:k) {
    hill_estimates[j] <- mean(log(x_sorted[1:j])) - log(x_sorted[j + 1])
  }

  # Use selected k
  alpha_hat <- 1 / hill_estimates[k]

  list(
    alpha = alpha_hat,
    xi = 1 / alpha_hat,
    hill_plot = hill_estimates,
    k = k,
    threshold = x_sorted[k + 1]
  )
}

#' Mean Excess Plot
#'
#' @param x numeric vector
#' @param n_thresholds number of threshold values
#' @return mean excess values at various thresholds
#' @export
mean_excess_plot <- function(x, n_thresholds = 100) {
  x_sorted <- sort(x)
  thresholds <- quantile(x, probs = seq(0.5, 0.99, length.out = n_thresholds))

  mean_excess <- sapply(thresholds, function(u) {
    exceedances <- x[x > u] - u
    if (length(exceedances) > 2) mean(exceedances) else NA
  })

  n_exceed <- sapply(thresholds, function(u) sum(x > u))

  list(
    thresholds = thresholds,
    mean_excess = mean_excess,
    n_exceedances = n_exceed
  )
}

#' GPD Threshold Selection
#'
#' Uses parameter stability method.
#'
#' @param x numeric vector
#' @param n_thresholds number of thresholds to try
#' @return optimal threshold and GPD parameters
#' @export
gpd_threshold_select <- function(x, n_thresholds = 50) {
  thresholds <- quantile(x, probs = seq(0.7, 0.98, length.out = n_thresholds))

  results <- data.frame(
    threshold = thresholds,
    xi = NA,
    beta = NA,
    n_exceed = NA
  )

  for (i in seq_along(thresholds)) {
    u <- thresholds[i]
    exceedances <- x[x > u] - u
    n_exceed <- length(exceedances)

    if (n_exceed < 10) next

    # MLE for GPD
    gpd_ll <- function(params) {
      xi <- params[1]
      beta <- exp(params[2])
      if (abs(xi) < 1e-7) {
        ll <- -n_exceed * log(beta) - sum(exceedances) / beta
      } else {
        z <- 1 + xi * exceedances / beta
        if (any(z <= 0)) return(1e10)
        ll <- -n_exceed * log(beta) - (1 + 1 / xi) * sum(log(z))
      }
      -ll
    }

    opt <- tryCatch(
      optim(c(0.1, log(mean(exceedances))), gpd_ll, method = "Nelder-Mead"),
      error = function(e) NULL
    )

    if (!is.null(opt)) {
      results$xi[i] <- opt$par[1]
      results$beta[i] <- exp(opt$par[2])
      results$n_exceed[i] <- n_exceed
    }
  }

  # Find stable region (minimum variance of xi)
  valid <- !is.na(results$xi)
  if (sum(valid) < 5) {
    optimal_idx <- which(valid)[1]
  } else {
    xi_vals <- results$xi[valid]
    rolling_var <- sapply(3:(length(xi_vals) - 2), function(i) {
      var(xi_vals[(i - 2):(i + 2)])
    })
    optimal_idx <- which(valid)[which.min(rolling_var) + 2]
  }

  list(
    optimal_threshold = results$threshold[optimal_idx],
    xi = results$xi[optimal_idx],
    beta = results$beta[optimal_idx],
    results = results[valid, ]
  )
}

# =============================================================================
# SECTION 8: COPULA RISK
# =============================================================================

#' Gaussian Copula VaR
#'
#' @param returns matrix of asset returns
#' @param weights portfolio weights
#' @param alpha confidence level
#' @param n_sim number of simulations
#' @return copula VaR
#' @export
copula_var_gaussian <- function(returns, weights, alpha = 0.95,
                                 n_sim = 10000) {
  k <- ncol(returns)
  n <- nrow(returns)

  # Estimate rank correlation
  R <- cor(returns, method = "spearman")

  # Convert to Pearson for Gaussian copula
  R_pearson <- 2 * sin(pi * R / 6)

  # Ensure positive definite
  eig <- eigen(R_pearson)
  eig$values <- pmax(eig$values, 1e-6)
  R_pearson <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  diag(R_pearson) <- 1

  # Simulate from Gaussian copula
  L <- t(chol(R_pearson))
  Z <- matrix(rnorm(n_sim * k), nrow = n_sim)
  U <- pnorm(Z %*% t(L))

  # Transform to marginal distributions (empirical)
  sim_returns <- matrix(0, nrow = n_sim, ncol = k)
  for (j in seq_len(k)) {
    sim_returns[, j] <- quantile(returns[, j],
                                  probs = U[, j], type = 7)
  }

  port_returns <- sim_returns %*% weights
  var_val <- -quantile(port_returns, probs = 1 - alpha)

  list(
    var = as.numeric(var_val),
    cvar = -mean(port_returns[port_returns <= -var_val]),
    alpha = alpha,
    copula = "gaussian",
    correlation = R_pearson
  )
}

#' Student-t Copula VaR
#'
#' @param returns matrix of returns
#' @param weights portfolio weights
#' @param alpha confidence level
#' @param nu degrees of freedom
#' @param n_sim simulations
#' @return t-copula VaR
#' @export
copula_var_t <- function(returns, weights, alpha = 0.95,
                          nu = 5, n_sim = 10000) {
  k <- ncol(returns)
  R <- cor(returns, method = "spearman")
  R_pearson <- 2 * sin(pi * R / 6)

  eig <- eigen(R_pearson)
  eig$values <- pmax(eig$values, 1e-6)
  R_pearson <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  diag(R_pearson) <- 1

  # Simulate from t-copula
  L <- t(chol(R_pearson))
  Z <- matrix(rnorm(n_sim * k), nrow = n_sim)
  chi2 <- rchisq(n_sim, df = nu)
  T_sim <- Z / sqrt(chi2 / nu)
  T_corr <- T_sim %*% t(L)
  U <- pt(T_corr, df = nu)

  sim_returns <- matrix(0, nrow = n_sim, ncol = k)
  for (j in seq_len(k)) {
    sim_returns[, j] <- quantile(returns[, j], probs = U[, j], type = 7)
  }

  port_returns <- sim_returns %*% weights
  var_val <- -quantile(port_returns, probs = 1 - alpha)

  # Tail dependence
  lambda_lower <- 2 * pt(-sqrt((nu + 1) * (1 - R_pearson[1, 2]) /
                                  (1 + R_pearson[1, 2])), df = nu + 1)

  list(
    var = as.numeric(var_val),
    cvar = -mean(port_returns[port_returns <= -var_val]),
    alpha = alpha,
    copula = "t",
    nu = nu,
    tail_dependence_lower = lambda_lower
  )
}

#' Clayton Copula VaR
#'
#' @param returns matrix of bivariate returns
#' @param weights portfolio weights (length 2)
#' @param alpha confidence level
#' @param n_sim simulations
#' @return Clayton copula VaR
#' @export
copula_var_clayton <- function(returns, weights, alpha = 0.95,
                                n_sim = 10000) {
  stopifnot(ncol(returns) == 2)

  # Estimate Clayton parameter via Kendall's tau
  tau <- cor(returns[, 1], returns[, 2], method = "kendall")
  theta <- 2 * tau / (1 - tau)
  if (theta <= 0) theta <- 0.1

  # Simulate from Clayton copula
  V <- rgamma(n_sim, shape = 1 / theta, rate = 1)
  E1 <- rexp(n_sim)
  E2 <- rexp(n_sim)

  U1 <- (1 + E1 / V)^(-1 / theta)
  U2 <- (1 + E2 / V)^(-1 / theta)

  sim_returns <- matrix(0, nrow = n_sim, ncol = 2)
  for (j in 1:2) {
    sim_returns[, j] <- quantile(returns[, j],
                                  probs = c(U1, U2)[(j - 1) * n_sim + 1:n_sim],
                                  type = 7)
  }

  port_returns <- sim_returns %*% weights
  var_val <- -quantile(port_returns, probs = 1 - alpha)

  # Lower tail dependence for Clayton
  lambda_lower <- 2^(-1 / theta)

  list(
    var = as.numeric(var_val),
    alpha = alpha,
    copula = "clayton",
    theta = theta,
    tail_dependence_lower = lambda_lower
  )
}

# =============================================================================
# SECTION 9: VaR BACKTESTING
# =============================================================================

#' Kupiec Proportion of Failures (POF) Test
#'
#' @param returns actual returns
#' @param var_estimates VaR estimates (positive)
#' @param alpha VaR confidence level
#' @return Kupiec test result
#' @export
kupiec_pof_test <- function(returns, var_estimates, alpha = 0.95) {
  n <- length(returns)
  violations <- returns < -var_estimates
  n_violations <- sum(violations)
  expected_rate <- 1 - alpha
  observed_rate <- n_violations / n

  # Likelihood ratio test
  if (n_violations == 0 || n_violations == n) {
    lr_stat <- NA
    p_value <- NA
  } else {
    lr_stat <- -2 * (
      n_violations * log(expected_rate) +
        (n - n_violations) * log(1 - expected_rate) -
        n_violations * log(observed_rate) -
        (n - n_violations) * log(1 - observed_rate)
    )
    p_value <- pchisq(lr_stat, df = 1, lower.tail = FALSE)
  }

  list(
    lr_stat = lr_stat,
    p_value = p_value,
    n_violations = n_violations,
    expected_violations = round(n * expected_rate),
    observed_rate = observed_rate,
    expected_rate = expected_rate,
    reject = if (!is.na(p_value)) p_value < 0.05 else FALSE,
    n = n
  )
}

#' Christoffersen Independence Test
#'
#' @param returns actual returns
#' @param var_estimates VaR estimates
#' @param alpha VaR confidence level
#' @return independence test result
#' @export
christoffersen_test <- function(returns, var_estimates, alpha = 0.95) {
  violations <- as.integer(returns < -var_estimates)
  n <- length(violations)

  # Transition counts
  n00 <- sum(violations[-n] == 0 & violations[-1] == 0)
  n01 <- sum(violations[-n] == 0 & violations[-1] == 1)
  n10 <- sum(violations[-n] == 1 & violations[-1] == 0)
  n11 <- sum(violations[-n] == 1 & violations[-1] == 1)

  # Transition probabilities
  p01 <- if (n00 + n01 > 0) n01 / (n00 + n01) else 0
  p11 <- if (n10 + n11 > 0) n11 / (n10 + n11) else 0
  p_hat <- (n01 + n11) / (n - 1)

  # LR test for independence
  if (p01 == 0 || p01 == 1 || p11 == 0 || p11 == 1 || p_hat == 0 || p_hat == 1) {
    lr_ind <- 0
  } else {
    lr_ind <- -2 * (
      (n00 + n10) * log(1 - p_hat) + (n01 + n11) * log(p_hat) -
        n00 * log(1 - p01) - n01 * log(p01) -
        n10 * log(1 - p11) - n11 * log(p11)
    )
  }

  p_value <- pchisq(lr_ind, df = 1, lower.tail = FALSE)

  # Combined (conditional coverage) test
  kupiec <- kupiec_pof_test(returns, var_estimates, alpha)
  lr_cc <- kupiec$lr_stat + lr_ind
  p_cc <- pchisq(lr_cc, df = 2, lower.tail = FALSE)

  list(
    lr_independence = lr_ind,
    p_value_independence = p_value,
    lr_coverage = kupiec$lr_stat,
    p_value_coverage = kupiec$p_value,
    lr_conditional = lr_cc,
    p_value_conditional = p_cc,
    transition_matrix = matrix(c(n00, n01, n10, n11), 2, 2,
                                dimnames = list(c("0", "1"), c("0", "1"))),
    p01 = p01,
    p11 = p11
  )
}

#' Traffic Light VaR Backtest (Basel)
#'
#' @param n_violations number of violations
#' @param n_days number of trading days (typically 250)
#' @param alpha VaR confidence level
#' @return traffic light zone
#' @export
traffic_light_test <- function(n_violations, n_days = 250,
                                alpha = 0.99) {
  expected <- n_days * (1 - alpha)

  # Basel traffic light zones for 99% VaR over 250 days
  # Green: 0-4, Yellow: 5-9, Red: 10+
  if (alpha == 0.99 && n_days == 250) {
    if (n_violations <= 4) {
      zone <- "green"
      multiplier <- 3.0
    } else if (n_violations <= 9) {
      zone <- "yellow"
      multiplier <- 3.0 + 0.2 * (n_violations - 4)
    } else {
      zone <- "red"
      multiplier <- 4.0
    }
  } else {
    # General case using binomial
    p_val <- pbinom(n_violations - 1, size = n_days,
                     prob = 1 - alpha, lower.tail = FALSE)
    if (p_val > 0.05) {
      zone <- "green"
      multiplier <- 3.0
    } else if (p_val > 0.01) {
      zone <- "yellow"
      multiplier <- 3.5
    } else {
      zone <- "red"
      multiplier <- 4.0
    }
  }

  list(
    zone = zone,
    n_violations = n_violations,
    expected_violations = expected,
    capital_multiplier = multiplier,
    n_days = n_days,
    alpha = alpha
  )
}

# =============================================================================
# SECTION 10: LIQUIDITY RISK
# =============================================================================

#' Amihud Illiquidity Measure
#'
#' @param returns absolute returns
#' @param volume trading volume
#' @return Amihud illiquidity ratio
#' @export
amihud_illiquidity <- function(returns, volume) {
  n <- length(returns)
  stopifnot(length(volume) == n)

  daily_illiq <- abs(returns) / (volume + 1e-10)
  amihud <- mean(daily_illiq, na.rm = TRUE)

  list(
    amihud = amihud,
    daily_illiquidity = daily_illiq,
    n = n
  )
}

#' Bid-Ask Spread Estimator (Roll)
#'
#' @param prices price series
#' @return estimated bid-ask spread
#' @export
roll_spread <- function(prices) {
  dp <- diff(prices)
  n <- length(dp)
  autocovariance <- cov(dp[-1], dp[-n])

  if (autocovariance < 0) {
    spread <- 2 * sqrt(-autocovariance)
  } else {
    spread <- 0
  }

  list(
    spread = spread,
    spread_pct = spread / mean(prices) * 100,
    autocovariance = autocovariance
  )
}

#' Holding Period VaR Scaling
#'
#' @param var_1day 1-day VaR
#' @param holding_period target holding period (days)
#' @param method "sqrt" or "adjusted"
#' @param returns optional returns for adjusted method
#' @return scaled VaR
#' @export
var_holding_period <- function(var_1day, holding_period,
                                method = "sqrt", returns = NULL) {
  if (method == "sqrt") {
    var_scaled <- var_1day * sqrt(holding_period)
  } else {
    # Adjusted for autocorrelation
    if (is.null(returns)) stop("returns required for adjusted method")
    rho <- acf(returns, lag.max = 1, plot = FALSE)$acf[2]
    adjustment <- sqrt(holding_period + 2 * rho *
                         (holding_period - 1 - holding_period * rho +
                            rho^holding_period) / (1 - rho)^2)
    var_scaled <- var_1day * adjustment
  }

  list(
    var_scaled = var_scaled,
    var_1day = var_1day,
    holding_period = holding_period,
    scaling_factor = var_scaled / var_1day,
    method = method
  )
}

#' Days to Liquidate
#'
#' @param position_size dollar position size
#' @param avg_daily_volume average daily volume
#' @param participation_rate maximum fraction of ADV
#' @param price current price
#' @return liquidation time estimate
#' @export
days_to_liquidate <- function(position_size, avg_daily_volume,
                               participation_rate = 0.10, price = 1) {
  shares <- position_size / price
  daily_capacity <- avg_daily_volume * participation_rate
  days <- ceiling(shares / daily_capacity)

  list(
    days = days,
    daily_capacity_shares = daily_capacity,
    daily_capacity_dollars = daily_capacity * price,
    position_shares = shares,
    participation_rate = participation_rate
  )
}

# =============================================================================
# SECTION 11: CREDIT RISK
# =============================================================================

#' Portfolio Default Correlation (Asset Correlation Model)
#'
#' @param default_probs vector of default probabilities
#' @param asset_correlation asset correlation matrix
#' @param n_sim number of simulations
#' @param lgd loss given default (fraction)
#' @param exposure exposure at default
#' @return credit VaR
#' @export
credit_var <- function(default_probs, asset_correlation = 0.3,
                        n_sim = 10000, lgd = 0.45,
                        exposure = NULL) {
  k <- length(default_probs)
  if (is.null(exposure)) exposure <- rep(1, k)

  # Asset correlation matrix
  if (is.numeric(asset_correlation) && length(asset_correlation) == 1) {
    rho <- asset_correlation
    R <- matrix(rho, k, k)
    diag(R) <- 1
  } else {
    R <- asset_correlation
  }

  # Default thresholds (Merton model)
  thresholds <- qnorm(default_probs)

  # Simulate correlated asset values
  L <- t(chol(R))
  Z <- matrix(rnorm(n_sim * k), nrow = n_sim)
  A <- Z %*% t(L)

  # Determine defaults
  defaults <- sweep(A, 2, thresholds, "<")

  # Portfolio losses
  losses <- defaults %*% (lgd * exposure)

  # Credit VaR
  expected_loss <- mean(losses)
  var_99 <- quantile(losses, probs = 0.99)
  var_999 <- quantile(losses, probs = 0.999)
  cvar_99 <- mean(losses[losses >= var_99])

  # Economic capital
  economic_capital <- var_99 - expected_loss

  list(
    expected_loss = expected_loss,
    var_99 = as.numeric(var_99),
    var_999 = as.numeric(var_999),
    cvar_99 = cvar_99,
    economic_capital = as.numeric(economic_capital),
    default_rate = mean(defaults),
    loss_distribution = losses,
    n_obligors = k,
    total_exposure = sum(exposure)
  )
}

# =============================================================================
# SECTION 12: SYSTEMIC RISK
# =============================================================================

#' CoVaR (Adrian and Brunnermeier)
#'
#' @param returns_system system returns
#' @param returns_firm firm returns
#' @param alpha confidence level
#' @return CoVaR and Delta-CoVaR
#' @export
covar_measure <- function(returns_system, returns_firm, alpha = 0.95) {
  n <- length(returns_system)
  q <- 1 - alpha

  # VaR of firm
  var_firm <- quantile(returns_firm, probs = q)

  # CoVaR: VaR of system conditional on firm being at its VaR
  # Estimate via quantile regression (simplified: linear)
  # system_return = a + b * firm_return + epsilon

  fit <- lm(returns_system ~ returns_firm)
  a <- coef(fit)[1]
  b <- coef(fit)[2]
  resid <- residuals(fit)

  # Conditional VaR of residuals
  resid_var <- quantile(resid, probs = q)

  # CoVaR
  covar <- a + b * as.numeric(var_firm) + as.numeric(resid_var)

  # Unconditional VaR of system
  var_system <- quantile(returns_system, probs = q)

  # Delta-CoVaR: CoVaR - VaR_system
  delta_covar <- covar - as.numeric(var_system)

  list(
    covar = covar,
    delta_covar = delta_covar,
    var_system = as.numeric(var_system),
    var_firm = as.numeric(var_firm),
    alpha = alpha,
    beta = b
  )
}

#' Marginal Expected Shortfall (MES)
#'
#' @param returns_system system returns
#' @param returns_firm firm returns
#' @param alpha tail threshold
#' @return MES
#' @export
mes_measure <- function(returns_system, returns_firm, alpha = 0.05) {
  # MES = E[r_firm | r_system < VaR_system(alpha)]
  var_system <- quantile(returns_system, probs = alpha)
  tail_idx <- returns_system <= var_system

  if (sum(tail_idx) == 0) {
    mes <- NA
  } else {
    mes <- mean(returns_firm[tail_idx])
  }

  list(
    mes = mes,
    var_system = as.numeric(var_system),
    n_tail = sum(tail_idx),
    alpha = alpha
  )
}

#' SRISK (Systemic Risk Index)
#'
#' @param equity market equity
#' @param debt book debt
#' @param lrmes long-run MES
#' @param k prudential capital ratio (default 0.08)
#' @return SRISK
#' @export
srisk_measure <- function(equity, debt, lrmes, k = 0.08) {
  # SRISK = max(0, k * (Debt + Equity) - Equity * (1 - LRMES))
  srisk <- pmax(0, k * (debt + equity) - equity * (1 - lrmes))

  list(
    srisk = srisk,
    equity = equity,
    debt = debt,
    lrmes = lrmes,
    leverage = (debt + equity) / equity,
    capital_shortfall = srisk > 0
  )
}

# =============================================================================
# SECTION 13: RISK BUDGETING
# =============================================================================

#' Equal Risk Contribution (ERC) Portfolio
#'
#' Solves for weights such that each asset contributes equally to risk.
#'
#' @param Sigma covariance matrix
#' @param budget risk budget (NULL for equal)
#' @param max_iter maximum iterations
#' @param tol convergence tolerance
#' @return ERC weights
#' @export
erc_portfolio <- function(Sigma, budget = NULL, max_iter = 1000,
                           tol = 1e-8) {
  k <- ncol(Sigma)
  if (is.null(budget)) budget <- rep(1 / k, k)

  # Cyclical coordinate descent / Newton method
  w <- rep(1 / k, k)

  for (iter in seq_len(max_iter)) {
    w_old <- w

    # Portfolio risk
    port_var <- as.numeric(t(w) %*% Sigma %*% w)
    port_vol <- sqrt(port_var)

    # Marginal risk contribution
    mrc <- as.numeric(Sigma %*% w) / port_vol

    # Risk contribution
    rc <- w * mrc

    # Target risk contribution
    target_rc <- budget * port_vol

    # Update weights (Newton-like)
    for (i in seq_len(k)) {
      if (mrc[i] > 0) {
        w[i] <- w[i] * (target_rc[i] / rc[i])^0.5
      }
    }

    # Normalize
    w <- w / sum(w)

    if (max(abs(w - w_old)) < tol) break
  }

  # Final risk decomposition
  port_var <- as.numeric(t(w) %*% Sigma %*% w)
  port_vol <- sqrt(port_var)
  mrc <- as.numeric(Sigma %*% w) / port_vol
  rc <- w * mrc
  rc_pct <- rc / port_vol * 100

  list(
    weights = w,
    risk_contribution = rc,
    risk_contribution_pct = rc_pct,
    marginal_risk = mrc,
    portfolio_vol = port_vol,
    iterations = iter,
    budget = budget
  )
}

#' Risk Parity Optimization
#'
#' @param Sigma covariance matrix
#' @param target_vol target portfolio volatility (annual)
#' @return risk parity portfolio
#' @export
risk_parity <- function(Sigma, target_vol = NULL) {
  erc <- erc_portfolio(Sigma)

  if (!is.null(target_vol)) {
    # Scale to target volatility
    current_vol <- erc$portfolio_vol
    leverage <- target_vol / current_vol
    scaled_weights <- erc$weights * leverage

    list(
      weights = erc$weights,
      scaled_weights = scaled_weights,
      leverage = leverage,
      target_vol = target_vol,
      unlevered_vol = current_vol,
      risk_contribution = erc$risk_contribution,
      risk_contribution_pct = erc$risk_contribution_pct
    )
  } else {
    erc
  }
}

# =============================================================================
# SECTION 14: DYNAMIC RISK
# =============================================================================

#' Regime-Conditional VaR
#'
#' @param returns numeric vector
#' @param alpha confidence level
#' @param n_regimes number of regimes (2)
#' @return regime-conditional VaR
#' @export
regime_var <- function(returns, alpha = 0.95, n_regimes = 2) {
  n <- length(returns)

  # Simple regime identification via rolling volatility
  window <- min(63, n %/% 4)
  rolling_vol <- numeric(n)
  for (t in window:n) {
    rolling_vol[t] <- sd(returns[(t - window + 1):t])
  }
  rolling_vol[1:(window - 1)] <- rolling_vol[window]

  # Threshold for regime classification
  vol_median <- median(rolling_vol[rolling_vol > 0])
  regimes <- ifelse(rolling_vol > vol_median, 2, 1)

  # VaR per regime
  var_per_regime <- numeric(n_regimes)
  cvar_per_regime <- numeric(n_regimes)

  for (r in seq_len(n_regimes)) {
    regime_returns <- returns[regimes == r]
    if (length(regime_returns) > 10) {
      var_per_regime[r] <- -quantile(regime_returns, probs = 1 - alpha)
      threshold <- quantile(regime_returns, probs = 1 - alpha)
      cvar_per_regime[r] <- -mean(regime_returns[regime_returns <= threshold])
    }
  }

  # Current regime
  current_regime <- regimes[n]
  current_var <- var_per_regime[current_regime]

  list(
    current_var = current_var,
    current_regime = current_regime,
    var_by_regime = var_per_regime,
    cvar_by_regime = cvar_per_regime,
    regimes = regimes,
    n_per_regime = table(regimes),
    rolling_vol = rolling_vol
  )
}

#' Time-Varying Copula VaR
#'
#' DCC-based time-varying correlation for copula.
#'
#' @param returns matrix of returns
#' @param weights portfolio weights
#' @param alpha confidence level
#' @param window rolling window for correlation
#' @return time-varying VaR series
#' @export
tv_copula_var <- function(returns, weights, alpha = 0.95,
                           window = 126) {
  n <- nrow(returns)
  k <- ncol(returns)

  var_series <- numeric(n)
  correlation_series <- array(NA, dim = c(k, k, n))

  for (t in window:n) {
    # Rolling window data
    r_window <- returns[(t - window + 1):t, ]

    # Estimate correlation
    R_t <- cor(r_window)
    correlation_series[, , t] <- R_t

    # Parametric VaR with current correlation
    mu <- colMeans(r_window)
    sigma <- apply(r_window, 2, sd)
    D <- diag(sigma)
    Sigma_t <- D %*% R_t %*% D

    port_mu <- sum(weights * mu)
    port_sigma <- sqrt(as.numeric(t(weights) %*% Sigma_t %*% weights))
    z <- qnorm(alpha)

    var_series[t] <- -(port_mu - z * port_sigma)
  }

  # Fill initial values
  var_series[1:(window - 1)] <- var_series[window]

  list(
    var_series = var_series,
    correlation_series = correlation_series,
    current_var = var_series[n],
    mean_var = mean(var_series[window:n]),
    max_var = max(var_series[window:n]),
    window = window,
    alpha = alpha
  )
}

#' Comprehensive Risk Report
#'
#' @param returns matrix of asset returns
#' @param weights portfolio weights
#' @param alpha confidence level
#' @return comprehensive risk report
#' @export
risk_report <- function(returns, weights, alpha = 0.95) {
  if (is.vector(returns)) {
    port_returns <- returns
    is_multi <- FALSE
  } else {
    port_returns <- as.numeric(returns %*% weights)
    is_multi <- TRUE
  }

  # Basic statistics
  stats <- list(
    mean_return = mean(port_returns),
    volatility = sd(port_returns),
    annualized_return = mean(port_returns) * 252,
    annualized_vol = sd(port_returns) * sqrt(252),
    sharpe_ratio = mean(port_returns) / sd(port_returns) * sqrt(252),
    skewness = sum((port_returns - mean(port_returns))^3) /
      (length(port_returns) * sd(port_returns)^3),
    kurtosis = sum((port_returns - mean(port_returns))^4) /
      (length(port_returns) * sd(port_returns)^4)
  )

  # VaR measures
  var_hist <- var_historical(port_returns, alpha)
  var_param <- var_parametric_normal(port_returns, alpha)
  var_cf <- var_cornish_fisher(port_returns, alpha)

  # CVaR
  cvar <- cvar_historical(port_returns, alpha)

  # Drawdown
  dd <- max_drawdown(port_returns)

  # Component risk (if multi-asset)
  comp_risk <- NULL
  if (is_multi) {
    comp_risk <- component_var(returns, weights, alpha)
  }

  list(
    statistics = stats,
    var = list(
      historical = var_hist$var,
      parametric = var_param$var,
      cornish_fisher = var_cf$var
    ),
    cvar = cvar$cvar,
    max_drawdown = dd$max_drawdown,
    component_risk = comp_risk,
    alpha = alpha,
    n = length(port_returns)
  )
}

#' Print Risk Report
#' @export
print_risk_report <- function(report) {
  cat("Portfolio Risk Report\n")
  cat("=====================\n\n")

  cat("BASIC STATISTICS\n")
  cat("  Annualized Return:", round(report$statistics$annualized_return * 100, 2), "%\n")
  cat("  Annualized Volatility:", round(report$statistics$annualized_vol * 100, 2), "%\n")
  cat("  Sharpe Ratio:", round(report$statistics$sharpe_ratio, 3), "\n")
  cat("  Skewness:", round(report$statistics$skewness, 3), "\n")
  cat("  Kurtosis:", round(report$statistics$kurtosis, 3), "\n\n")

  cat("VALUE AT RISK (", report$alpha * 100, "%)\n", sep = "")
  cat("  Historical:", round(report$var$historical * 100, 3), "%\n")
  cat("  Parametric:", round(report$var$parametric * 100, 3), "%\n")
  cat("  Cornish-Fisher:", round(report$var$cornish_fisher * 100, 3), "%\n\n")

  cat("CVaR:", round(report$cvar * 100, 3), "%\n\n")
  cat("MAX DRAWDOWN:", round(report$max_drawdown * 100, 2), "%\n\n")

  if (!is.null(report$component_risk)) {
    cat("COMPONENT VaR:\n")
    for (i in seq_along(report$component_risk$component_var)) {
      cat("  Asset", i, ":", round(report$component_risk$pct_contribution[i], 1), "%\n")
    }
  }

  invisible(report)
}

###############################################################################
# END OF FILE: portfolio_risk.R
###############################################################################
