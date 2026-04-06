# vine_copula.R
#
# Vine copula models for multivariate dependence in crypto portfolios.
#
# A vine copula decomposes a multivariate distribution into a cascade of
# bivariate copulas (pair copulas) organized in a tree structure:
#
#   f(x₁,...,xₙ) = ∏ fₖ(xₖ) · ∏_{(i,j|D)∈V} c_{ij|D}(F(xᵢ|xD), F(xⱼ|xD))
#
# Two common structures:
#   C-vine: one variable is the "center" at each tree level
#   D-vine: variables are connected in a sequence (most common for time series)
#
# References:
#   Joe (2014) "Dependence Modeling with Copulas"
#   Aas et al. (2009) "Pair-copula constructions of multiple dependence"
#   Czado (2019) "Analyzing Dependent Data with Vine Copulas"
#
# Package dependencies: VineCopula, copula, ggplot2, reshape2, dplyr

.require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing missing package: ", pkg)
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}
invisible(lapply(c("VineCopula", "copula", "ggplot2", "reshape2", "dplyr",
                   "gridExtra", "RColorBrewer"), .require_pkg))

# ─────────────────────────────────────────────────────────────────────────────
# PSEUDO-OBSERVATIONS (UNIFORM MARGINALS)
# ─────────────────────────────────────────────────────────────────────────────

#' Transform returns to pseudo-uniform observations via empirical CDF.
#'
#' u_i = rank(x_i) / (n+1)  — avoids boundary issues at 0 and 1.
#'
#' @param X  n x d matrix of returns
#' @return   n x d matrix of pseudo-uniform observations in (0,1)
to_pseudo_uniform <- function(X) {
  n <- nrow(X)
  d <- ncol(X)
  U <- matrix(0, n, d)
  for (j in 1:d) {
    U[, j] <- rank(X[, j]) / (n + 1)
  }
  return(U)
}

# ─────────────────────────────────────────────────────────────────────────────
# COPULA FAMILY SELECTION: AIC/BIC
# ─────────────────────────────────────────────────────────────────────────────

#' Fit multiple bivariate copula families to a pair (u, v) and select
#' the best by AIC.
#'
#' Families tested:
#'   1  = Gaussian, 2 = Student-t, 3 = Clayton, 4 = Gumbel,
#'   5 = Frank, 6 = Joe, 7 = BB1, 13 = Survival Clayton,
#'   14 = Survival Gumbel, 16 = Survival Joe, 23/24 = rotated variants
#'
#' @param u  vector of pseudo-uniform observations for variable 1
#' @param v  vector of pseudo-uniform observations for variable 2
#' @return   data.frame with family, AIC, BIC, parameters for each family
select_bivariate_copula <- function(u, v) {
  # Candidate families (VineCopula integer codes)
  families <- c(
    1,   # Gaussian
    2,   # Student-t
    3,   # Clayton (lower tail)
    4,   # Gumbel (upper tail)
    5,   # Frank (symmetric)
    6,   # Joe (upper tail, stronger)
    7,   # BB1 (Clayton + Gumbel mixture)
    13,  # Survival Clayton (upper tail)
    14,  # Survival Gumbel (lower tail)
    16,  # Survival Joe (lower tail)
    23,  # Rotated Clayton 180° (upper tail)
    24   # Rotated Gumbel 180° (lower tail)
  )

  family_names <- c(
    "Gaussian", "Student-t", "Clayton", "Gumbel", "Frank", "Joe", "BB1",
    "Surv.Clayton", "Surv.Gumbel", "Surv.Joe",
    "Rot.Clayton", "Rot.Gumbel"
  )

  results <- data.frame(
    family      = integer(0),
    family_name = character(0),
    par1        = numeric(0),
    par2        = numeric(0),
    loglik      = numeric(0),
    AIC         = numeric(0),
    BIC         = numeric(0)
  )

  n <- length(u)

  for (i in seq_along(families)) {
    fam <- families[i]
    tryCatch({
      fit <- BiCopEst(u, v, family = fam, method = "mle")
      ll  <- fit$logLik
      k   <- ifelse(fam == 2 || fam == 7, 2, 1)  # df for t-cop; 2 params for BB1
      aic <- -2 * ll + 2 * k
      bic <- -2 * ll + log(n) * k

      results <- rbind(results, data.frame(
        family      = fam,
        family_name = family_names[i],
        par1        = fit$par,
        par2        = ifelse(is.null(fit$par2) || is.na(fit$par2), NA, fit$par2),
        loglik      = ll,
        AIC         = aic,
        BIC         = bic
      ))
    }, error = function(e) {
      # Skip families that fail to converge
    })
  }

  results <- results[order(results$AIC), ]
  return(results)
}

# ─────────────────────────────────────────────────────────────────────────────
# D-VINE CONSTRUCTION AND FITTING
# ─────────────────────────────────────────────────────────────────────────────

#' Construct and fit a D-vine copula.
#'
#' A D-vine has variables arranged in a sequence 1—2—3—...—d.
#' Tree 1: pairs (1,2), (2,3), ..., (d-1,d)
#' Tree k: conditional pairs ((1,k+1|2,...,k), ...) — each conditioned on k-1 variables.
#'
#' @param U     n x d matrix of pseudo-uniform observations
#' @param d     number of variables (inferred from U if NULL)
#' @param selcrit  selection criterion: "AIC" or "BIC"
#' @return      RVineMatrix object from VineCopula package
fit_dvine <- function(U, selcrit = "AIC") {
  d <- ncol(U)
  n <- nrow(U)

  cat(sprintf("Fitting D-vine copula: d=%d variables, n=%d observations\n", d, n))
  cat("Using", selcrit, "for pair copula family selection.\n\n")

  # Use VineCopula's automatic vine selection with D-vine structure
  # type = 2 restricts to D-vine (type = 1 = C-vine, 0 = R-vine general)
  RVM <- RVineStructureSelect(
    data    = U,
    familyset = c(1, 2, 3, 4, 5, 6, 7, 13, 14),
    type    = 2,  # D-vine
    selcrit = selcrit,
    method  = "mle",
    indeptest = TRUE,
    level   = 0.05
  )

  cat("D-vine fitting complete.\n")
  summary_vine(RVM)
  return(RVM)
}

#' Construct and fit a C-vine copula.
#'
#' A C-vine has one "center" variable at each tree level connected to all others.
#' Optimal root variable ordering can improve fit.
#'
#' @param U  n x d matrix of pseudo-uniform observations
#' @param root_order  integer vector giving variable ordering (NULL = auto)
fit_cvine <- function(U, root_order = NULL, selcrit = "AIC") {
  d <- ncol(U)
  cat(sprintf("Fitting C-vine copula: d=%d variables\n", d))

  # If root order not given, select variable with max sum of |Kendall's tau|
  if (is.null(root_order)) {
    tau_mat <- abs(TauMatrix(U))
    diag(tau_mat) <- 0
    tau_sums <- colSums(tau_mat)
    root_order <- order(tau_sums, decreasing = TRUE)
    cat("Auto root order:", root_order, "\n")
  }

  RVM <- RVineStructureSelect(
    data    = U[, root_order],
    familyset = c(1, 2, 3, 4, 5, 6, 13, 14),
    type    = 1,  # C-vine
    selcrit = selcrit,
    method  = "mle"
  )

  cat("C-vine fitting complete.\n")
  return(RVM)
}

#' Print a summary of vine structure.
summary_vine <- function(RVM) {
  d <- RVM$Matrix[1, 1]  # number of variables
  cat("Vine structure summary:\n")
  cat("  Variables:", ncol(RVM$Matrix), "\n")
  cat("  Pair copulas at each tree level:\n")

  for (tree in 1:(ncol(RVM$Matrix) - 1)) {
    n_pairs <- ncol(RVM$Matrix) - tree
    cat(sprintf("    Tree %d: %d pairs\n", tree, n_pairs))
  }

  # Most significant pair copulas
  cat("\n  Top pair copulas by |Kendall's tau|:\n")
  pairs_info <- RVinePar2Tau(RVM)
  tau_flat   <- as.vector(pairs_info)[!is.na(as.vector(pairs_info))]
  cat(sprintf("  Mean |τ|: %.4f, Max |τ|: %.4f\n",
              mean(abs(tau_flat)), max(abs(tau_flat))))
}

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL INDEPENDENCE TESTS
# ─────────────────────────────────────────────────────────────────────────────

#' Test conditional independence for pairs in the vine.
#' Uses Kendall's tau test on partial copula residuals.
#'
#' @param U  pseudo-uniform observations
#' @param alpha  significance level
test_conditional_independence <- function(U, alpha = 0.05) {
  d <- ncol(U)
  cat("\nConditional Independence Tests (Kendall's tau):\n")
  cat("─────────────────────────────────────────────\n")

  results <- list()

  # Test all pairs
  for (i in 1:(d-1)) {
    for (j in (i+1):d) {
      tau_test <- cor.test(U[,i], U[,j], method = "kendall")
      p_val <- tau_test$p.value
      tau   <- tau_test$estimate

      sig <- ifelse(p_val < alpha, "*", "")
      cat(sprintf("  Pair (%d,%d): τ=%.4f, p=%.4f %s\n",
                  i, j, tau, p_val, sig))

      results[[paste0(i, "_", j)]] <- list(
        tau = tau, p_value = p_val, significant = p_val < alpha
      )
    }
  }

  n_sig <- sum(sapply(results, `[[`, "significant"))
  cat(sprintf("\n  %d/%d pairs show significant dependence at α=%.2f\n",
              n_sig, length(results), alpha))
  return(results)
}

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION FROM FITTED VINE
# ─────────────────────────────────────────────────────────────────────────────

#' Simulate from a fitted vine copula model.
#'
#' @param RVM      fitted RVineMatrix object
#' @param n_sim    number of simulations
#' @param margins  list of marginal distributions (NULL = return uniform)
#' @return  matrix of simulated observations
simulate_vine <- function(RVM, n_sim, margins = NULL) {
  # Simulate pseudo-uniform samples from vine
  U_sim <- RVineSim(n_sim, RVM)

  if (is.null(margins)) {
    return(U_sim)
  }

  # Apply inverse marginal CDFs
  X_sim <- matrix(0, n_sim, length(margins))
  for (j in seq_along(margins)) {
    X_sim[, j] <- margins[[j]]$quantile(U_sim[, j])
  }
  return(X_sim)
}

# ─────────────────────────────────────────────────────────────────────────────
# TAIL DEPENDENCE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

#' Compute upper and lower tail dependence coefficients for all pairs.
#'
#' For copula C, the tail dependence coefficients are:
#'   λᵤ = lim_{u→1} C̄(u,u) / (1-u)   [upper]
#'   λₗ = lim_{u→0} C(u,u) / u         [lower]
#'
#' For key families:
#'   Clayton:    λₗ = 2^{-1/θ}, λᵤ = 0
#'   Gumbel:     λᵤ = 2 - 2^{1/θ}, λₗ = 0
#'   Student-t:  λᵤ = λₗ = 2·t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
#'   Gaussian:   λᵤ = λₗ = 0
#'
#' @param RVM  fitted vine
#' @return     list of (upper, lower) tail dependence matrices
extract_tail_dependence <- function(RVM) {
  d   <- ncol(RVM$Matrix)
  ltu <- matrix(NA, d, d)  # upper
  ltl <- matrix(NA, d, d)  # lower

  # Iterate over tree-1 pairs (direct bivariate relationships)
  for (j in 2:d) {
    for (i in 1:(j-1)) {
      fam  <- RVM$family[i, j]
      par1 <- RVM$par[i, j]
      par2 <- RVM$par2[i, j]

      td <- bivariate_tail_dependence(fam, par1, par2)
      ltu[i, j] <- ltu[j, i] <- td$upper
      ltl[i, j] <- ltl[j, i] <- td$lower
    }
  }

  return(list(upper = ltu, lower = ltl))
}

#' Compute tail dependence for a single bivariate copula.
bivariate_tail_dependence <- function(family, par, par2 = NA) {
  upper <- 0.0
  lower <- 0.0

  if (family == 1) {
    # Gaussian: no tail dependence
    upper <- 0; lower <- 0
  } else if (family == 2) {
    # Student-t
    rho <- par; nu <- par2
    if (!is.na(nu) && nu > 0) {
      td <- 2 * pt(-sqrt((nu + 1) * (1 - rho) / (1 + rho)), df = nu + 1)
      upper <- td; lower <- td
    }
  } else if (family == 3) {
    # Clayton: lower tail only
    lower <- 2^(-1/par)
    upper <- 0
  } else if (family == 4) {
    # Gumbel: upper tail only
    upper <- 2 - 2^(1/par)
    lower <- 0
  } else if (family == 5) {
    # Frank: no tail dependence
    upper <- 0; lower <- 0
  } else if (family == 6) {
    # Joe: upper tail only
    upper <- 2 - 2^(1/par)
    lower <- 0
  } else if (family == 13) {
    # Survival Clayton: upper tail
    upper <- 2^(-1/par)
    lower <- 0
  } else if (family == 14) {
    # Survival Gumbel: lower tail
    lower <- 2 - 2^(1/par)
    upper <- 0
  } else if (family == 7) {
    # BB1: both tails
    theta <- par; delta <- par2
    if (!is.na(delta)) {
      lower <- 2^(-1/(theta * delta))
      upper <- 2 - 2^(1/delta)
    }
  }

  return(list(upper = upper, lower = lower))
}

#' Empirical tail dependence estimation (non-parametric).
#'
#' @param u   vector of pseudo-uniform observations, variable 1
#' @param v   vector of pseudo-uniform observations, variable 2
#' @param q   tail quantile level (e.g. 0.05 for lower, 0.95 for upper)
empirical_tail_dependence <- function(u, v, q = 0.05) {
  n <- length(u)

  # Lower tail
  idx_l <- u <= q
  lambda_lower <- sum(v[idx_l] <= q) / sum(idx_l)

  # Upper tail
  idx_u <- u >= 1 - q
  lambda_upper <- sum(v[idx_u] >= 1 - q) / sum(idx_u)

  return(list(lower = lambda_lower, upper = lambda_upper))
}

# ─────────────────────────────────────────────────────────────────────────────
# CRYPTO PORTFOLIO JOINT TAIL RISK
# ─────────────────────────────────────────────────────────────────────────────

#' Compute joint tail risk metrics for a crypto portfolio.
#'
#' Uses the fitted vine copula to assess:
#'   - Probability of joint drawdown > threshold
#'   - Expected shortfall under joint adverse scenario
#'   - Kendall's tau dependence matrix
#'
#' @param returns_matrix  n x d matrix of daily log-returns
#' @param weights         d-vector of portfolio weights
#' @param RVM             fitted vine copula
#' @param n_sim           number of Monte Carlo simulations
portfolio_tail_risk_vine <- function(returns_matrix, weights, RVM,
                                     n_sim = 10000, alpha = 0.01) {
  d <- ncol(returns_matrix)
  n <- nrow(returns_matrix)

  cat("\nPortfolio Tail Risk (Vine Copula Monte Carlo)\n")
  cat("Simulating", n_sim, "scenarios...\n")

  # Simulate from vine
  U_sim <- RVineSim(n_sim, RVM)

  # Apply empirical marginal inverse CDFs
  X_sim <- matrix(0, n_sim, d)
  for (j in 1:d) {
    X_sim[, j] <- quantile(returns_matrix[, j],
                            probs = U_sim[, j], type = 7)
  }

  # Portfolio P&L
  portfolio_sim <- X_sim %*% weights

  # Risk metrics
  VaR_alpha  <- quantile(portfolio_sim, alpha)
  CVaR_alpha <- mean(portfolio_sim[portfolio_sim <= VaR_alpha])

  # Joint drawdown probability
  threshold <- -0.05  # 5% joint loss
  prob_joint <- mean(apply(X_sim, 1, function(r) all(r < threshold)))

  cat(sprintf("  VaR(%.1f%%)  = %.4f (%.2f%%)\n", 100*alpha, VaR_alpha, 100*VaR_alpha))
  cat(sprintf("  CVaR(%.1f%%) = %.4f (%.2f%%)\n", 100*alpha, CVaR_alpha, 100*CVaR_alpha))
  cat(sprintf("  P(all assets < %.0f%%) = %.4f\n", 100*threshold, prob_joint))

  return(list(
    VaR   = VaR_alpha,
    CVaR  = CVaR_alpha,
    prob_joint_drawdown = prob_joint,
    portfolio_sim = portfolio_sim
  ))
}

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

#' Plot copula heatmaps: scatter plot matrix coloured by copula density.
plot_copula_heatmaps <- function(U, var_names = NULL) {
  d <- ncol(U)
  if (is.null(var_names)) var_names <- paste0("X", 1:d)

  plots <- list()
  k <- 1
  for (i in 1:(d-1)) {
    for (j in (i+1):d) {
      df_pair <- data.frame(u = U[, i], v = U[, j])

      # Estimate copula density via kernel
      p <- ggplot(df_pair, aes(x = u, y = v)) +
        stat_density_2d(aes(fill = after_stat(density)), geom = "raster", contour = FALSE) +
        scale_fill_viridis_c(name = "Density") +
        geom_point(alpha = 0.1, size = 0.5) +
        labs(title = paste(var_names[i], "vs", var_names[j]),
             x = var_names[i], y = var_names[j]) +
        theme_minimal(base_size = 10) +
        theme(legend.position = "none")

      plots[[k]] <- p
      k <- k + 1
    }
  }

  n_plots <- length(plots)
  n_cols  <- ceiling(sqrt(n_plots))
  n_rows  <- ceiling(n_plots / n_cols)

  do.call(grid.arrange, c(plots, ncol = n_cols))
}

#' Plot vine tree structure as a graph diagram.
plot_vine_tree <- function(RVM, var_names = NULL) {
  d <- ncol(RVM$Matrix)
  if (is.null(var_names)) var_names <- paste0("X", 1:d)

  # Extract tree 1 pairs
  family_matrix <- RVM$family
  par_matrix     <- RVM$par
  tau_matrix     <- RVinePar2Tau(RVM)

  cat("\nVine Tree 1 Structure:\n")
  cat("─────────────────────\n")
  cat(sprintf("%-15s %-15s %-10s %-10s\n", "Node i", "Node j", "Family", "Kendall τ"))
  cat(sprintf("%-15s %-15s %-10s %-10s\n",
              "─────────────", "─────────────", "────────", "─────────"))

  # Family code → name lookup
  fam_names <- c(
    "0"="Indep", "1"="Gaussian", "2"="Student-t",
    "3"="Clayton", "4"="Gumbel", "5"="Frank", "6"="Joe", "7"="BB1",
    "13"="Surv.Clayton", "14"="Surv.Gumbel", "16"="Surv.Joe"
  )

  for (j in 2:d) {
    i   <- RVM$Matrix[d + 1 - j + 1, j]  # conditioning set edge
    fam <- RVM$family[d - j + 1, j]
    tau <- tryCatch(tau_matrix[d - j + 1, j], error = function(e) NA)
    fn  <- fam_names[as.character(fam)]
    fn  <- ifelse(is.na(fn), paste("F:", fam), fn)

    cat(sprintf("%-15s %-15s %-10s %-10.4f\n",
                var_names[i], var_names[j], fn, ifelse(is.na(tau), 0, tau)))
  }
}

#' Plot tail dependence heatmap.
plot_tail_dependence <- function(td_matrix, title = "Upper Tail Dependence", var_names = NULL) {
  d <- nrow(td_matrix)
  if (is.null(var_names)) var_names <- paste0("X", 1:d)

  df_td <- melt(td_matrix)
  df_td$Var1 <- factor(df_td$Var1, labels = var_names)
  df_td$Var2 <- factor(df_td$Var2, labels = var_names)

  ggplot(df_td, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = ifelse(is.na(value), "", sprintf("%.2f", value))),
              size = 3.5, color = "black") +
    scale_fill_gradient2(low = "white", high = "darkred", mid = "salmon",
                         midpoint = 0.3, na.value = "gray90",
                         name = "λ", limits = c(0, 1)) +
    labs(title = title, x = "", y = "") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Plot portfolio P&L distribution from vine copula simulation.
plot_portfolio_distribution <- function(portfolio_sim, VaR, CVaR, alpha) {
  df <- data.frame(pnl = portfolio_sim)

  ggplot(df, aes(x = pnl)) +
    geom_histogram(aes(y = after_stat(density)), bins = 60,
                   fill = "steelblue", color = "white", alpha = 0.7) +
    geom_density(color = "navy", linewidth = 1) +
    geom_vline(xintercept = VaR,  color = "red",    linestyle = "dashed", linewidth = 1) +
    geom_vline(xintercept = CVaR, color = "darkred", linestyle = "dashed", linewidth = 1) +
    annotate("text", x = VaR,  y = Inf, label = paste("VaR:", round(VaR, 4)),
             vjust = 2, color = "red") +
    annotate("text", x = CVaR, y = Inf, label = paste("CVaR:", round(CVaR, 4)),
             vjust = 4, color = "darkred") +
    labs(title = sprintf("Portfolio P&L Distribution (Vine Copula, %d simulations)",
                          length(portfolio_sim)),
         x = "Portfolio Return", y = "Density") +
    theme_minimal()
}

# ─────────────────────────────────────────────────────────────────────────────
# KENDALL'S TAU MATRIX
# ─────────────────────────────────────────────────────────────────────────────

#' Compute full Kendall's tau matrix.
kendall_tau_matrix <- function(X) {
  d <- ncol(X)
  tau_mat <- matrix(1, d, d)
  for (i in 1:(d-1)) {
    for (j in (i+1):d) {
      tau <- cor(X[,i], X[,j], method = "kendall")
      tau_mat[i,j] <- tau_mat[j,i] <- tau
    }
  }
  return(tau_mat)
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

demo_vine_copula <- function() {
  set.seed(42)
  cat("================================================================\n")
  cat("Vine Copula Demo: Crypto Portfolio Dependence\n")
  cat("================================================================\n\n")

  # Simulate 4 crypto assets with known dependence structure
  # BTC, ETH, BNB, SOL — using a Clayton copula for lower tail dependence
  n <- 500
  d <- 4
  asset_names <- c("BTC", "ETH", "BNB", "SOL")

  # True joint distribution: Student-t copula + t-marginals
  Sigma <- matrix(c(
    1.00, 0.75, 0.60, 0.55,
    0.75, 1.00, 0.65, 0.50,
    0.60, 0.65, 1.00, 0.45,
    0.55, 0.50, 0.45, 1.00
  ), 4, 4)

  # Simulate from t-copula
  t_cop <- tCopula(param = Sigma[lower.tri(Sigma)], dim = d,
                   dispstr = "un", df = 5)
  U_true <- rCopula(n, t_cop)

  # Apply t-distributed marginals (heavy tails)
  X_returns <- matrix(0, n, d)
  dfs <- c(5, 6, 5, 7)
  sigs <- c(0.04, 0.05, 0.045, 0.06)
  for (j in 1:d) {
    X_returns[, j] <- qt(U_true[, j], df = dfs[j]) * sigs[j]
  }
  colnames(X_returns) <- asset_names

  cat("Simulated", n, "daily returns for", d, "crypto assets.\n")
  cat("Summary statistics:\n")
  for (j in 1:d) {
    cat(sprintf("  %s: mean=%.4f, sd=%.4f, skew=%.3f, kurt=%.3f\n",
                asset_names[j],
                mean(X_returns[,j]), sd(X_returns[,j]),
                mean(((X_returns[,j] - mean(X_returns[,j]))/sd(X_returns[,j]))^3),
                mean(((X_returns[,j] - mean(X_returns[,j]))/sd(X_returns[,j]))^4) - 3))
  }

  # Convert to pseudo-uniform margins
  U <- to_pseudo_uniform(X_returns)

  # Conditional independence tests
  ci_tests <- test_conditional_independence(U)

  # Select best bivariate copula for BTC-ETH pair
  cat("\nBivariate copula selection (BTC-ETH):\n")
  biv_select <- select_bivariate_copula(U[,1], U[,2])
  cat(sprintf("  Best family: %s (AIC=%.2f)\n",
              biv_select$family_name[1], biv_select$AIC[1]))
  print(head(biv_select[, c("family_name", "AIC", "BIC", "par1")], 5))

  # Fit D-vine
  cat("\nFitting D-vine...\n")
  dvine <- fit_dvine(U, selcrit = "AIC")

  # Vine tree summary
  plot_vine_tree(dvine, var_names = asset_names)

  # Tail dependence
  td <- extract_tail_dependence(dvine)
  cat("\nUpper tail dependence coefficients:\n")
  print(round(td$upper, 4))
  cat("\nLower tail dependence coefficients:\n")
  print(round(td$lower, 4))

  # Empirical tail dependence comparison
  cat("\nEmpirical (non-parametric) tail dependence BTC-ETH:\n")
  emp_td <- empirical_tail_dependence(U[,1], U[,2], q = 0.05)
  cat(sprintf("  Lower: %.4f, Upper: %.4f\n", emp_td$lower, emp_td$upper))

  # Portfolio tail risk
  weights <- c(0.40, 0.30, 0.20, 0.10)
  risk <- portfolio_tail_risk_vine(X_returns, weights, dvine, n_sim = 5000)

  # Kendall tau matrix
  tau_mat <- kendall_tau_matrix(X_returns)
  cat("\nKendall's tau matrix:\n")
  print(round(tau_mat, 3))

  # Plots
  cat("\nGenerating plots...\n")

  # Copula heatmaps
  png("vine_copula_heatmaps.png", width = 1000, height = 400)
  plot_copula_heatmaps(U[, 1:3], var_names = asset_names[1:3])
  dev.off()

  # Tail dependence heatmap
  p_td <- plot_tail_dependence(td$upper, "Upper Tail Dependence (Vine)", asset_names)
  ggsave("vine_tail_dep.png", p_td, width = 6, height = 5)

  # Portfolio distribution
  p_port <- plot_portfolio_distribution(risk$portfolio_sim, risk$VaR, risk$CVaR, 0.01)
  ggsave("vine_portfolio_dist.png", p_port, width = 8, height = 5)

  cat("\nSaved: vine_copula_heatmaps.png, vine_tail_dep.png, vine_portfolio_dist.png\n")
  cat("\nVine copula demo complete.\n")

  return(list(vine = dvine, U = U, X = X_returns, tail_dep = td, risk = risk))
}

# Run demo
if (!interactive()) {
  demo_vine_copula()
}
