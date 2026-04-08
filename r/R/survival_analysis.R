##############################################################################
# survival_analysis.R -- Survival / Duration Analysis for Finance
# KM, Nelson-Aalen, Cox PH, parametric AFT, competing risks, frailty,
# credit default, trade duration, drawdown duration, diagnostics
##############################################################################

# ---------------------------------------------------------------------------
# Kaplan-Meier Estimator
# ---------------------------------------------------------------------------
kaplan_meier <- function(time, event, group = NULL) {
  if (is.null(group)) group <- rep(1, length(time))
  groups <- unique(group)
  results <- list()
  for (g in groups) {
    mask <- group == g
    t_g <- time[mask]; d_g <- event[mask]
    ord <- order(t_g)
    t_g <- t_g[ord]; d_g <- d_g[ord]
    utimes <- sort(unique(t_g[d_g == 1]))
    n_risk <- numeric(length(utimes))
    n_event <- numeric(length(utimes))
    for (j in seq_along(utimes)) {
      n_risk[j] <- sum(t_g >= utimes[j])
      n_event[j] <- sum(t_g == utimes[j] & d_g == 1)
    }
    surv <- cumprod(1 - n_event / n_risk)
    var_greenwood <- surv^2 * cumsum(n_event / (n_risk * (n_risk - n_event)))
    var_greenwood[is.nan(var_greenwood)] <- 0
    se <- sqrt(var_greenwood)
    ci_lower <- pmax(0, surv - 1.96 * se)
    ci_upper <- pmin(1, surv + 1.96 * se)
    log_log_ci_lower <- surv^exp(1.96 * se / (surv * log(surv)))
    log_log_ci_upper <- surv^exp(-1.96 * se / (surv * log(surv)))
    median_idx <- which(surv <= 0.5)
    median_surv <- if (length(median_idx) > 0) utimes[median_idx[1]] else NA
    rmst <- sum(diff(c(0, utimes)) * c(1, surv[-length(surv)]))
    results[[as.character(g)]] <- list(
      time = utimes, surv = surv, se = se, n_risk = n_risk, n_event = n_event,
      ci_lower = ci_lower, ci_upper = ci_upper,
      log_log_ci_lower = log_log_ci_lower, log_log_ci_upper = log_log_ci_upper,
      median = median_surv, rmst = rmst, n = sum(mask), events = sum(d_g),
      group = g
    )
  }
  structure(results, class = "km_fit")
}

print.km_fit <- function(x, ...) {
  for (nm in names(x)) {
    g <- x[[nm]]
    cat(sprintf("Group: %s | n=%d events=%d median=%.2f RMST=%.2f\n",
                nm, g$n, g$events,
                ifelse(is.na(g$median), Inf, g$median), g$rmst))
  }
}

# ---------------------------------------------------------------------------
# Nelson-Aalen Cumulative Hazard Estimator
# ---------------------------------------------------------------------------
nelson_aalen <- function(time, event) {
  ord <- order(time)
  time <- time[ord]; event <- event[ord]
  utimes <- sort(unique(time[event == 1]))
  n_risk <- numeric(length(utimes))
  n_event <- numeric(length(utimes))
  for (j in seq_along(utimes)) {
    n_risk[j] <- sum(time >= utimes[j])
    n_event[j] <- sum(time == utimes[j] & event == 1)
  }
  cum_haz <- cumsum(n_event / n_risk)
  var_na <- cumsum(n_event / n_risk^2)
  se <- sqrt(var_na)
  surv_na <- exp(-cum_haz)
  list(time = utimes, cum_hazard = cum_haz, se = se, surv = surv_na,
       n_risk = n_risk, n_event = n_event, method = "Nelson-Aalen")
}

# ---------------------------------------------------------------------------
# Log-Rank Test
# ---------------------------------------------------------------------------
log_rank_test <- function(time, event, group) {
  groups <- unique(group)
  K <- length(groups)
  utimes <- sort(unique(time[event == 1]))
  J <- length(utimes)
  O <- numeric(K)
  E <- numeric(K)
  V_mat <- matrix(0, K - 1, K - 1)
  for (j in 1:J) {
    tj <- utimes[j]
    d_j <- sum(time == tj & event == 1)
    n_j <- sum(time >= tj)
    if (n_j < 2) next
    for (k in 1:K) {
      mask_k <- group == groups[k]
      d_kj <- sum(time[mask_k] == tj & event[mask_k] == 1)
      n_kj <- sum(time[mask_k] >= tj)
      O[k] <- O[k] + d_kj
      E[k] <- E[k] + n_kj * d_j / n_j
    }
    for (a in 1:(K - 1)) {
      n_aj <- sum(time[group == groups[a]] >= tj)
      for (b in 1:(K - 1)) {
        n_bj <- sum(time[group == groups[b]] >= tj)
        if (n_j > 1) {
          cov_ab <- d_j * (n_j - d_j) / (n_j^2 * (n_j - 1)) *
            (ifelse(a == b, 1, 0) * n_aj * (n_j - n_aj) / n_j -
               (1 - ifelse(a == b, 1, 0)) * n_aj * n_bj / n_j)
          V_mat[a, b] <- V_mat[a, b] + d_j * n_aj / n_j *
            (ifelse(a == b, 1, 0) - n_bj / n_j) * (n_j - d_j) / (n_j - 1)
        }
      }
    }
  }
  if (K == 2) {
    chi2 <- (O[1] - E[1])^2 / V_mat[1, 1]
  } else {
    diff_OE <- (O - E)[1:(K - 1)]
    V_inv <- tryCatch(solve(V_mat), error = function(e) MASS::ginv(V_mat))
    chi2 <- as.numeric(t(diff_OE) %*% V_inv %*% diff_OE)
  }
  df <- K - 1
  pval <- 1 - pchisq(chi2, df)
  list(chi2 = chi2, df = df, pval = pval, O = O, E = E,
       groups = groups, method = "Log-Rank")
}

# ---------------------------------------------------------------------------
# Cox Proportional Hazards: Breslow Partial Likelihood
# ---------------------------------------------------------------------------
cox_ph <- function(time, event, X, max_iter = 100, tol = 1e-9,
                   ties = "breslow") {
  n <- nrow(X); p <- ncol(X)
  ord <- order(time, -event)
  time <- time[ord]; event <- event[ord]; X <- X[ord, , drop = FALSE]
  beta <- rep(0, p)
  for (iter in 1:max_iter) {
    eta <- as.vector(X %*% beta)
    exp_eta <- exp(eta)
    utimes <- unique(time[event == 1])
    loglik <- 0
    score <- rep(0, p)
    hessian <- matrix(0, p, p)
    risk_sum <- 0; risk_x_sum <- rep(0, p)
    risk_xx_sum <- matrix(0, p, p)
    ptr <- n
    for (j in length(utimes):1) {
      tj <- utimes[j]
      while (ptr >= 1 && time[ptr] >= tj) {
        risk_sum <- risk_sum + exp_eta[ptr]
        risk_x_sum <- risk_x_sum + exp_eta[ptr] * X[ptr, ]
        risk_xx_sum <- risk_xx_sum + exp_eta[ptr] * tcrossprod(X[ptr, ])
        ptr <- ptr - 1
      }
    }
    risk_sum <- 0; risk_x_sum <- rep(0, p); risk_xx_sum <- matrix(0, p, p)
    event_times <- time[event == 1]
    for (i in n:1) {
      risk_sum <- risk_sum + exp_eta[i]
      risk_x_sum <- risk_x_sum + exp_eta[i] * X[i, ]
      risk_xx_sum <- risk_xx_sum + exp_eta[i] * tcrossprod(X[i, ])
      if (event[i] == 1) {
        x_bar <- risk_x_sum / risk_sum
        loglik <- loglik + eta[i] - log(risk_sum)
        score <- score + X[i, ] - x_bar
        hessian <- hessian - (risk_xx_sum / risk_sum - tcrossprod(x_bar))
      }
    }
    H_inv <- tryCatch(solve(-hessian), error = function(e) {
      diag(1e-6, p)
    })
    delta <- H_inv %*% score
    beta_new <- beta + as.vector(delta)
    if (max(abs(delta)) < tol) {
      beta <- beta_new
      break
    }
    beta <- beta_new
  }
  vcov <- -H_inv
  se <- sqrt(diag(vcov))
  z <- beta / se
  pval <- 2 * pnorm(-abs(z))
  hr <- exp(beta)
  names(beta) <- colnames(X)
  names(hr) <- colnames(X)
  eta_final <- as.vector(X %*% beta)
  exp_eta_final <- exp(eta_final)
  baseline_haz <- numeric(0); baseline_times <- numeric(0)
  cum_haz <- numeric(n)
  utimes_all <- sort(unique(time[event == 1]))
  h0 <- numeric(length(utimes_all))
  for (j in seq_along(utimes_all)) {
    tj <- utimes_all[j]
    d_j <- sum(time == tj & event == 1)
    risk_j <- sum(exp_eta_final[time >= tj])
    h0[j] <- d_j / risk_j
  }
  H0 <- cumsum(h0)
  for (i in 1:n) {
    idx <- max(which(utimes_all <= time[i]), 0)
    cum_haz[i] <- if (idx > 0) H0[idx] * exp_eta_final[i] else 0
  }
  surv_baseline <- exp(-H0)
  concordance <- compute_concordance(time, event, eta_final)
  lrt <- 2 * (loglik - 0)
  lrt_pval <- 1 - pchisq(lrt, p)
  list(coefficients = beta, se = se, z = z, pval = pval, hr = hr,
       vcov = vcov, loglik = loglik, n = n, n_events = sum(event),
       baseline_hazard = data.frame(time = utimes_all, h0 = h0, H0 = H0,
                                     S0 = surv_baseline),
       concordance = concordance, lrt = lrt, lrt_pval = lrt_pval,
       cum_hazard = cum_haz, iterations = iter, converged = iter < max_iter,
       time = time, event = event, X = X,
       method = "Cox-PH")
}

# ---------------------------------------------------------------------------
# Concordance Index (C-statistic)
# ---------------------------------------------------------------------------
compute_concordance <- function(time, event, risk_score) {
  concordant <- 0; discordant <- 0; tied <- 0; total <- 0
  idx_event <- which(event == 1)
  n_ev <- length(idx_event)
  if (n_ev > 5000) {
    idx_event <- sample(idx_event, 5000)
    n_ev <- 5000
  }
  for (ii in 1:n_ev) {
    i <- idx_event[ii]
    candidates <- which(time > time[i])
    if (length(candidates) == 0) next
    if (length(candidates) > 200) candidates <- sample(candidates, 200)
    for (j in candidates) {
      total <- total + 1
      if (risk_score[i] > risk_score[j]) concordant <- concordant + 1
      else if (risk_score[i] < risk_score[j]) discordant <- discordant + 1
      else tied <- tied + 1
    }
  }
  if (total == 0) return(NA)
  (concordant + 0.5 * tied) / total
}

# ---------------------------------------------------------------------------
# Cox PH with Time-Varying Covariates
# ---------------------------------------------------------------------------
cox_ph_tv <- function(start, stop, event, X, max_iter = 100, tol = 1e-9) {
  n <- length(event); p <- ncol(X)
  beta <- rep(0, p)
  for (iter in 1:max_iter) {
    eta <- as.vector(X %*% beta)
    exp_eta <- exp(eta)
    utimes <- sort(unique(stop[event == 1]))
    loglik <- 0; score <- rep(0, p); hessian <- matrix(0, p, p)
    for (tj in utimes) {
      at_risk <- which(start < tj & stop >= tj)
      failed <- which(stop == tj & event == 1)
      if (length(at_risk) == 0) next
      r_sum <- sum(exp_eta[at_risk])
      r_x <- colSums(exp_eta[at_risk] * X[at_risk, , drop = FALSE])
      r_xx <- crossprod(X[at_risk, , drop = FALSE] * exp_eta[at_risk],
                        X[at_risk, , drop = FALSE])
      x_bar <- r_x / r_sum
      for (f in failed) {
        loglik <- loglik + eta[f] - log(r_sum)
        score <- score + X[f, ] - x_bar
        hessian <- hessian - (r_xx / r_sum - tcrossprod(x_bar))
      }
    }
    H_inv <- tryCatch(solve(-hessian), error = function(e) diag(1e-6, p))
    delta <- H_inv %*% score
    beta_new <- beta + as.vector(delta)
    if (max(abs(delta)) < tol) { beta <- beta_new; break }
    beta <- beta_new
  }
  vcov <- -H_inv
  se <- sqrt(diag(vcov))
  z <- beta / se
  pval <- 2 * pnorm(-abs(z))
  names(beta) <- colnames(X)
  list(coefficients = beta, se = se, z = z, pval = pval, hr = exp(beta),
       vcov = vcov, loglik = loglik, converged = iter < max_iter,
       method = "Cox-PH-TV")
}

# ---------------------------------------------------------------------------
# Stratified Cox Model
# ---------------------------------------------------------------------------
cox_ph_stratified <- function(time, event, X, strata, max_iter = 100,
                               tol = 1e-9) {
  n <- length(time); p <- ncol(X)
  strata_levels <- unique(strata)
  beta <- rep(0, p)
  for (iter in 1:max_iter) {
    eta <- as.vector(X %*% beta)
    exp_eta <- exp(eta)
    loglik <- 0; score <- rep(0, p); hessian <- matrix(0, p, p)
    for (s in strata_levels) {
      mask <- strata == s
      t_s <- time[mask]; e_s <- event[mask]
      X_s <- X[mask, , drop = FALSE]
      eta_s <- eta[mask]; exp_eta_s <- exp_eta[mask]
      ord_s <- order(t_s, -e_s)
      t_s <- t_s[ord_s]; e_s <- e_s[ord_s]
      X_s <- X_s[ord_s, , drop = FALSE]; exp_eta_s <- exp_eta_s[ord_s]
      eta_s <- eta_s[ord_s]
      n_s <- sum(mask)
      r_sum <- 0; r_x <- rep(0, p); r_xx <- matrix(0, p, p)
      for (i in n_s:1) {
        r_sum <- r_sum + exp_eta_s[i]
        r_x <- r_x + exp_eta_s[i] * X_s[i, ]
        r_xx <- r_xx + exp_eta_s[i] * tcrossprod(X_s[i, ])
        if (e_s[i] == 1) {
          x_bar <- r_x / r_sum
          loglik <- loglik + eta_s[i] - log(r_sum)
          score <- score + X_s[i, ] - x_bar
          hessian <- hessian - (r_xx / r_sum - tcrossprod(x_bar))
        }
      }
    }
    H_inv <- tryCatch(solve(-hessian), error = function(e) diag(1e-6, p))
    delta <- H_inv %*% score
    beta_new <- beta + as.vector(delta)
    if (max(abs(delta)) < tol) { beta <- beta_new; break }
    beta <- beta_new
  }
  vcov <- -H_inv
  se <- sqrt(diag(vcov))
  z <- beta / se
  pval <- 2 * pnorm(-abs(z))
  names(beta) <- colnames(X)
  list(coefficients = beta, se = se, z = z, pval = pval, hr = exp(beta),
       vcov = vcov, loglik = loglik, strata_levels = strata_levels,
       method = "Stratified-Cox")
}

# ---------------------------------------------------------------------------
# Parametric Survival: Exponential
# ---------------------------------------------------------------------------
surv_exponential <- function(time, event, X = NULL) {
  n <- length(time)
  if (is.null(X)) {
    lambda <- sum(event) / sum(time)
    loglik <- sum(event) * log(lambda) - lambda * sum(time)
    se_lambda <- lambda / sqrt(sum(event))
    return(list(lambda = lambda, se = se_lambda, loglik = loglik,
                median = log(2) / lambda, mean = 1 / lambda,
                method = "Exponential"))
  }
  p <- ncol(X)
  X1 <- cbind(1, X)
  beta <- rep(0, p + 1)
  for (iter in 1:100) {
    eta <- as.vector(X1 %*% beta)
    lambda <- exp(eta)
    loglik <- sum(event * eta - lambda * time)
    score <- colSums(X1 * (event - lambda * time))
    hessian <- -crossprod(X1 * (lambda * time), X1)
    H_inv <- tryCatch(solve(-hessian), error = function(e) diag(1e-6, p + 1))
    delta <- H_inv %*% score
    beta <- beta + as.vector(delta)
    if (max(abs(delta)) < 1e-9) break
  }
  vcov <- -H_inv
  se <- sqrt(diag(vcov))
  names(beta) <- c("(Intercept)", colnames(X))
  list(coefficients = beta, se = se, vcov = vcov, loglik = loglik,
       hr = exp(beta[-1]), method = "Exponential-Reg")
}

# ---------------------------------------------------------------------------
# Parametric Survival: Weibull
# ---------------------------------------------------------------------------
surv_weibull <- function(time, event, X = NULL, max_iter = 200, tol = 1e-9) {
  n <- length(time)
  log_t <- log(pmax(time, 1e-10))
  if (is.null(X)) {
    neg_loglik <- function(par) {
      k <- exp(par[1]); lam <- exp(par[2])
      -sum(event * (log(k) - log(lam) + (k - 1) * (log_t - log(lam))) -
             (time / lam)^k)
    }
    opt <- optim(c(0, log(mean(time))), neg_loglik, method = "Nelder-Mead",
                 hessian = TRUE)
    k <- exp(opt$par[1]); lam <- exp(opt$par[2])
    loglik <- -opt$value
    se_par <- sqrt(diag(solve(opt$hessian)))
    return(list(shape = k, scale = lam, loglik = loglik,
                median = lam * log(2)^(1 / k),
                mean = lam * gamma(1 + 1 / k),
                se = se_par, method = "Weibull"))
  }
  p <- ncol(X)
  neg_loglik <- function(par) {
    log_k <- par[1]; beta <- par[2:(p + 2)]
    k <- exp(log_k)
    eta <- cbind(1, X) %*% beta
    log_lam <- eta
    lam <- exp(log_lam)
    ll <- sum(event * (log(k) - log_lam + (k - 1) * (log_t - log_lam)) -
                (time / lam)^k)
    -ll
  }
  start <- c(0, log(mean(time)), rep(0, p))
  opt <- optim(start, neg_loglik, method = "BFGS", hessian = TRUE,
               control = list(maxit = max_iter))
  k <- exp(opt$par[1])
  beta <- opt$par[2:(p + 2)]
  loglik <- -opt$value
  vcov <- tryCatch(solve(opt$hessian), error = function(e) diag(1e-4, p + 2))
  se <- sqrt(diag(vcov))
  names(beta) <- c("(Intercept)", colnames(X))
  list(shape = k, coefficients = beta, se = se[-1], loglik = loglik,
       vcov = vcov, method = "Weibull-Reg")
}

# ---------------------------------------------------------------------------
# Parametric Survival: Log-Normal
# ---------------------------------------------------------------------------
surv_lognormal <- function(time, event, X = NULL) {
  n <- length(time)
  log_t <- log(pmax(time, 1e-10))
  if (is.null(X)) {
    mu <- mean(log_t[event == 1])
    sigma <- sd(log_t[event == 1])
    neg_loglik <- function(par) {
      mu <- par[1]; sigma <- exp(par[2])
      z <- (log_t - mu) / sigma
      ll <- sum(event * dnorm(z, log = TRUE) - event * log(sigma) - event * log_t +
                  (1 - event) * pnorm(-z, log.p = TRUE))
      -ll
    }
    opt <- optim(c(mu, log(sigma)), neg_loglik, method = "BFGS", hessian = TRUE)
    mu <- opt$par[1]; sigma <- exp(opt$par[2])
    return(list(mu = mu, sigma = sigma, loglik = -opt$value,
                median = exp(mu), mean = exp(mu + sigma^2 / 2),
                method = "Log-Normal"))
  }
  p <- ncol(X)
  neg_loglik <- function(par) {
    beta <- par[1:(p + 1)]; log_sigma <- par[p + 2]
    sigma <- exp(log_sigma)
    eta <- cbind(1, X) %*% beta
    z <- (log_t - eta) / sigma
    ll <- sum(event * (dnorm(z, log = TRUE) - log(sigma) - log_t) +
                (1 - event) * pnorm(-z, log.p = TRUE))
    -ll
  }
  start <- c(mean(log_t), rep(0, p), log(sd(log_t)))
  opt <- optim(start, neg_loglik, method = "BFGS", hessian = TRUE)
  beta <- opt$par[1:(p + 1)]; sigma <- exp(opt$par[p + 2])
  vcov <- tryCatch(solve(opt$hessian), error = function(e) diag(1e-4, p + 2))
  se <- sqrt(diag(vcov))
  names(beta) <- c("(Intercept)", colnames(X))
  list(coefficients = beta, sigma = sigma, se = se, loglik = -opt$value,
       vcov = vcov, method = "LogNormal-AFT")
}

# ---------------------------------------------------------------------------
# Parametric Survival: Log-Logistic
# ---------------------------------------------------------------------------
surv_loglogistic <- function(time, event, X = NULL) {
  n <- length(time)
  log_t <- log(pmax(time, 1e-10))
  if (is.null(X)) {
    neg_loglik <- function(par) {
      alpha <- par[1]; log_gamma <- par[2]
      gamma_p <- exp(log_gamma)
      z <- (log_t - alpha) / gamma_p
      ll <- sum(event * (z - log(gamma_p) - log_t - 2 * log(1 + exp(z))) +
                  (1 - event) * (-log(1 + exp(z))))
      -ll
    }
    start <- c(mean(log_t), log(sd(log_t)))
    opt <- optim(start, neg_loglik, method = "BFGS", hessian = TRUE)
    alpha <- opt$par[1]; gamma_p <- exp(opt$par[2])
    return(list(alpha = alpha, gamma = gamma_p, loglik = -opt$value,
                median = exp(alpha), method = "Log-Logistic"))
  }
  p <- ncol(X)
  neg_loglik <- function(par) {
    beta <- par[1:(p + 1)]; log_gamma <- par[p + 2]
    gamma_p <- exp(log_gamma)
    eta <- cbind(1, X) %*% beta
    z <- (log_t - eta) / gamma_p
    ll <- sum(event * (z - log(gamma_p) - log_t - 2 * log(1 + exp(z))) +
                (1 - event) * (-log(1 + exp(z))))
    -ll
  }
  start <- c(mean(log_t), rep(0, p), log(sd(log_t)))
  opt <- optim(start, neg_loglik, method = "BFGS", hessian = TRUE)
  beta <- opt$par[1:(p + 1)]; gamma_p <- exp(opt$par[p + 2])
  vcov <- tryCatch(solve(opt$hessian), error = function(e) diag(1e-4, p + 2))
  se <- sqrt(diag(vcov))
  names(beta) <- c("(Intercept)", colnames(X))
  list(coefficients = beta, gamma = gamma_p, se = se, loglik = -opt$value,
       vcov = vcov, method = "LogLogistic-AFT")
}

# ---------------------------------------------------------------------------
# Accelerated Failure Time (AFT) -- general wrapper
# ---------------------------------------------------------------------------
aft_model <- function(time, event, X, distribution = c("weibull", "lognormal",
                                                         "loglogistic")) {
  distribution <- match.arg(distribution)
  switch(distribution,
    weibull = surv_weibull(time, event, X),
    lognormal = surv_lognormal(time, event, X),
    loglogistic = surv_loglogistic(time, event, X)
  )
}

# ---------------------------------------------------------------------------
# Competing Risks: Cause-Specific Hazard
# ---------------------------------------------------------------------------
cause_specific_hazard <- function(time, event_type, X, cause) {
  # event_type: 0=censored, 1,2,...=different causes
  event_cs <- as.integer(event_type == cause)
  cox_ph(time, event_cs, X)
}

# ---------------------------------------------------------------------------
# Competing Risks: Cumulative Incidence Function (Aalen-Johansen)
# ---------------------------------------------------------------------------
cumulative_incidence <- function(time, event_type) {
  causes <- sort(unique(event_type[event_type > 0]))
  n <- length(time)
  utimes <- sort(unique(time[event_type > 0]))
  J <- length(utimes)
  cif <- matrix(0, J, length(causes))
  colnames(cif) <- paste0("cause_", causes)
  surv_prev <- 1
  for (j in 1:J) {
    tj <- utimes[j]
    n_risk <- sum(time >= tj)
    for (k in seq_along(causes)) {
      d_kj <- sum(time == tj & event_type == causes[k])
      h_kj <- d_kj / n_risk
      cif[j, k] <- if (j == 1) surv_prev * h_kj
                    else cif[j - 1, k] + surv_prev * h_kj
    }
    d_all <- sum(time == tj & event_type > 0)
    surv_prev <- surv_prev * (1 - d_all / n_risk)
  }
  list(time = utimes, cif = cif, causes = causes, method = "CIF-AJ")
}

# ---------------------------------------------------------------------------
# Frailty Model (Shared Gamma Frailty)
# ---------------------------------------------------------------------------
frailty_gamma <- function(time, event, X, cluster, max_iter = 100,
                           tol = 1e-8) {
  n <- length(time); p <- ncol(X)
  ucl <- unique(cluster)
  G <- length(ucl)
  beta <- rep(0, p)
  theta <- 1
  for (iter in 1:max_iter) {
    eta <- as.vector(X %*% beta)
    exp_eta <- exp(eta)
    ord <- order(time, -event)
    t_ord <- time[ord]; e_ord <- event[ord]; ee_ord <- exp_eta[ord]
    cl_ord <- cluster[ord]; X_ord <- X[ord, , drop = FALSE]
    utimes <- sort(unique(t_ord[e_ord == 1]))
    h0 <- numeric(length(utimes))
    for (j in seq_along(utimes)) {
      tj <- utimes[j]
      d_j <- sum(t_ord == tj & e_ord == 1)
      risk <- sum(ee_ord[t_ord >= tj])
      h0[j] <- d_j / risk
    }
    H0_i <- numeric(n)
    for (i in 1:n) {
      idx <- which(utimes <= time[i])
      H0_i[i] <- if (length(idx) > 0) sum(h0[idx]) else 0
    }
    w_g <- numeric(G)
    d_g <- numeric(G)
    H_g <- numeric(G)
    for (g in seq_along(ucl)) {
      mask <- cluster == ucl[g]
      d_g[g] <- sum(event[mask])
      H_g[g] <- sum(H0_i[mask] * exp_eta[mask])
      w_g[g] <- (d_g[g] + 1 / theta) / (H_g[g] + 1 / theta)
    }
    w_expand <- numeric(n)
    for (g in seq_along(ucl)) {
      mask <- cluster == ucl[g]
      w_expand[mask] <- w_g[g]
    }
    loglik <- 0; score <- rep(0, p); hessian <- matrix(0, p, p)
    for (j in seq_along(utimes)) {
      tj <- utimes[j]
      at_risk <- which(time >= tj)
      failed <- which(time == tj & event == 1)
      r_sum <- sum(w_expand[at_risk] * exp_eta[at_risk])
      r_x <- colSums(w_expand[at_risk] * exp_eta[at_risk] * X[at_risk, , drop = FALSE])
      r_xx <- crossprod(X[at_risk, , drop = FALSE] * sqrt(w_expand[at_risk] * exp_eta[at_risk]),
                        X[at_risk, , drop = FALSE] * sqrt(w_expand[at_risk] * exp_eta[at_risk]))
      x_bar <- r_x / r_sum
      for (f in failed) {
        loglik <- loglik + log(w_expand[f]) + eta[f] - log(r_sum)
        score <- score + X[f, ] - x_bar
        hessian <- hessian - (r_xx / r_sum - tcrossprod(x_bar))
      }
    }
    H_inv <- tryCatch(solve(-hessian), error = function(e) diag(1e-6, p))
    delta <- H_inv %*% score
    beta_new <- beta + as.vector(delta)
    profile_theta <- function(th) {
      ll <- 0
      for (g in seq_along(ucl)) {
        dg <- d_g[g]; Hg <- H_g[g]
        ll <- ll + lgamma(dg + 1 / th) - lgamma(1 / th) -
          (1 / th) * log(1 + th * Hg) + dg * log(th * Hg / (1 + th * Hg))
      }
      -ll
    }
    opt_theta <- optim(log(theta), function(lt) profile_theta(exp(lt)),
                       method = "Brent", lower = -10, upper = 10)
    theta_new <- exp(opt_theta$par)
    if (max(abs(beta_new - beta)) < tol && abs(theta_new - theta) < tol) {
      beta <- beta_new; theta <- theta_new; break
    }
    beta <- beta_new; theta <- theta_new
  }
  vcov <- -H_inv
  se <- sqrt(diag(vcov))
  names(beta) <- colnames(X)
  list(coefficients = beta, se = se, hr = exp(beta), theta = theta,
       frailty = w_g, vcov = vcov, loglik = loglik,
       converged = iter < max_iter, method = "Gamma-Frailty")
}

# ---------------------------------------------------------------------------
# Schoenfeld Residuals for PH Assumption Testing
# ---------------------------------------------------------------------------
schoenfeld_residuals <- function(cox_fit) {
  time <- cox_fit$time; event <- cox_fit$event; X <- cox_fit$X
  beta <- cox_fit$coefficients
  n <- length(time); p <- length(beta)
  eta <- as.vector(X %*% beta)
  exp_eta <- exp(eta)
  event_idx <- which(event == 1)
  n_events <- length(event_idx)
  schoenfeld <- matrix(0, n_events, p)
  for (ii in seq_along(event_idx)) {
    i <- event_idx[ii]
    at_risk <- which(time >= time[i])
    wts <- exp_eta[at_risk]
    wts <- wts / sum(wts)
    x_bar <- colSums(X[at_risk, , drop = FALSE] * wts)
    schoenfeld[ii, ] <- X[i, ] - x_bar
  }
  colnames(schoenfeld) <- colnames(X)
  event_times <- time[event_idx]
  scaled <- schoenfeld %*% solve(cox_fit$vcov) * n_events
  test_stat <- numeric(p)
  pval <- numeric(p)
  for (j in 1:p) {
    rho_j <- cor(event_times, scaled[, j])
    chi2 <- n_events * rho_j^2
    test_stat[j] <- chi2
    pval[j] <- 1 - pchisq(chi2, 1)
  }
  list(schoenfeld = schoenfeld, scaled = scaled, event_times = event_times,
       test_stat = test_stat, pval = pval,
       ph_rejected = any(pval < 0.05), method = "Schoenfeld-Test")
}

# ---------------------------------------------------------------------------
# Martingale Residuals
# ---------------------------------------------------------------------------
martingale_residuals <- function(cox_fit) {
  event <- cox_fit$event
  cum_haz <- cox_fit$cum_hazard
  event - cum_haz
}

# ---------------------------------------------------------------------------
# Deviance Residuals
# ---------------------------------------------------------------------------
deviance_residuals <- function(cox_fit) {
  mg <- martingale_residuals(cox_fit)
  sign(mg) * sqrt(-2 * (mg + cox_fit$event * log(cox_fit$event - mg)))
}

# ---------------------------------------------------------------------------
# Application: Time-to-Default Modeling
# ---------------------------------------------------------------------------
default_model <- function(firm_data, time_col = "survival_time",
                          event_col = "defaulted",
                          covariates = c("leverage", "profitability",
                                         "size", "volatility", "interest_coverage")) {
  time <- firm_data[[time_col]]
  event <- firm_data[[event_col]]
  X <- as.matrix(firm_data[, covariates, drop = FALSE])
  valid <- complete.cases(time, event, X) & time > 0
  time <- time[valid]; event <- event[valid]; X <- X[valid, , drop = FALSE]
  cox_fit <- cox_ph(time, event, X)
  weibull_fit <- surv_weibull(time, event, X)
  sch <- schoenfeld_residuals(cox_fit)
  km <- kaplan_meier(time, event)
  cutpoints <- quantile(X[, "leverage"], c(0.33, 0.66), na.rm = TRUE)
  risk_group <- ifelse(X[, "leverage"] < cutpoints[1], "low",
                       ifelse(X[, "leverage"] < cutpoints[2], "mid", "high"))
  km_by_risk <- kaplan_meier(time, event, risk_group)
  lr <- log_rank_test(time, event, risk_group)
  list(cox = cox_fit, weibull = weibull_fit, schoenfeld = sch,
       km = km, km_by_risk = km_by_risk, logrank = lr,
       n = sum(valid), n_defaults = sum(event),
       method = "Default-Model")
}

# ---------------------------------------------------------------------------
# Application: Trade Holding Period Analysis
# ---------------------------------------------------------------------------
trade_duration_model <- function(trade_data, covariates = NULL) {
  time <- trade_data$holding_period
  event <- trade_data$exited
  if (is.null(covariates)) {
    covariates <- intersect(names(trade_data),
                            c("entry_signal_strength", "volatility_at_entry",
                              "position_size_pct", "unrealized_pnl",
                              "market_regime", "spread_at_entry"))
  }
  X <- as.matrix(trade_data[, covariates, drop = FALSE])
  valid <- complete.cases(time, event, X) & time > 0
  time <- time[valid]; event <- event[valid]; X <- X[valid, , drop = FALSE]
  cox_fit <- cox_ph(time, event, X)
  aft_weibull <- surv_weibull(time, event, X)
  aft_lognormal <- surv_lognormal(time, event, X)
  aic_weibull <- -2 * aft_weibull$loglik + 2 * length(aft_weibull$coefficients)
  aic_lognormal <- -2 * aft_lognormal$loglik + 2 * (length(aft_lognormal$coefficients) + 1)
  km <- kaplan_meier(time, event)
  list(cox = cox_fit, aft_weibull = aft_weibull, aft_lognormal = aft_lognormal,
       aic_weibull = aic_weibull, aic_lognormal = aic_lognormal,
       km = km, best_aft = if (aic_weibull < aic_lognormal) "Weibull" else "LogNormal",
       method = "Trade-Duration")
}

# ---------------------------------------------------------------------------
# Application: Drawdown Duration Modeling
# ---------------------------------------------------------------------------
drawdown_duration_model <- function(equity_curve, threshold = 0) {
  n <- length(equity_curve)
  running_max <- cummax(equity_curve)
  drawdown <- (running_max - equity_curve) / running_max
  in_dd <- drawdown > threshold
  dd_starts <- c(); dd_durations <- c(); dd_depths <- c(); dd_recovered <- c()
  i <- 1
  while (i <= n) {
    if (in_dd[i]) {
      start <- i
      max_depth <- drawdown[i]
      while (i <= n && in_dd[i]) {
        max_depth <- max(max_depth, drawdown[i])
        i <- i + 1
      }
      duration <- i - start
      recovered <- (i <= n)
      dd_starts <- c(dd_starts, start)
      dd_durations <- c(dd_durations, duration)
      dd_depths <- c(dd_depths, max_depth)
      dd_recovered <- c(dd_recovered, as.integer(recovered))
    } else {
      i <- i + 1
    }
  }
  if (length(dd_durations) < 3) {
    return(list(n_drawdowns = length(dd_durations), durations = dd_durations,
                depths = dd_depths, method = "Drawdown-Duration"))
  }
  km <- kaplan_meier(dd_durations, dd_recovered)
  weibull_fit <- surv_weibull(dd_durations, dd_recovered)
  depth_tercile <- cut(dd_depths, breaks = quantile(dd_depths, c(0, 0.33, 0.66, 1)),
                       labels = c("shallow", "medium", "deep"), include.lowest = TRUE)
  km_by_depth <- kaplan_meier(dd_durations, dd_recovered, depth_tercile)
  if (length(unique(depth_tercile)) > 1) {
    lr <- log_rank_test(dd_durations, dd_recovered, depth_tercile)
  } else {
    lr <- NULL
  }
  list(n_drawdowns = length(dd_durations), durations = dd_durations,
       depths = dd_depths, recovered = dd_recovered,
       km = km, weibull = weibull_fit, km_by_depth = km_by_depth,
       logrank = lr, method = "Drawdown-Duration")
}

# ---------------------------------------------------------------------------
# Model Comparison (AIC/BIC for parametric models)
# ---------------------------------------------------------------------------
surv_model_compare <- function(time, event, X,
                                models = c("exponential", "weibull",
                                           "lognormal", "loglogistic")) {
  results <- list()
  for (m in models) {
    fit <- switch(m,
      exponential = surv_exponential(time, event, X),
      weibull = surv_weibull(time, event, X),
      lognormal = surv_lognormal(time, event, X),
      loglogistic = surv_loglogistic(time, event, X)
    )
    k <- length(fit$coefficients) + ifelse(m %in% c("weibull"), 1, 0) +
      ifelse(m %in% c("lognormal", "loglogistic"), 1, 0)
    n <- length(time)
    aic <- -2 * fit$loglik + 2 * k
    bic <- -2 * fit$loglik + log(n) * k
    results[[m]] <- list(loglik = fit$loglik, aic = aic, bic = bic, k = k)
  }
  aic_vals <- sapply(results, function(r) r$aic)
  bic_vals <- sapply(results, function(r) r$bic)
  list(results = results, best_aic = names(which.min(aic_vals)),
       best_bic = names(which.min(bic_vals)), method = "Model-Compare")
}

# ---------------------------------------------------------------------------
# Survival Prediction (from Cox model)
# ---------------------------------------------------------------------------
predict_survival <- function(cox_fit, newdata, times = NULL) {
  beta <- cox_fit$coefficients
  bh <- cox_fit$baseline_hazard
  if (is.null(times)) times <- bh$time
  eta_new <- as.vector(as.matrix(newdata) %*% beta)
  n_new <- length(eta_new)
  n_times <- length(times)
  surv_mat <- matrix(1, n_new, n_times)
  for (j in 1:n_times) {
    idx <- max(which(bh$time <= times[j]), 0)
    H0_t <- if (idx > 0) bh$H0[idx] else 0
    surv_mat[, j] <- exp(-H0_t * exp(eta_new))
  }
  colnames(surv_mat) <- paste0("t=", round(times, 2))
  list(surv = surv_mat, times = times, risk_scores = eta_new)
}

# ---------------------------------------------------------------------------
# Simulate survival data
# ---------------------------------------------------------------------------
simulate_survival <- function(n = 1000, p = 5, beta = NULL,
                               distribution = "weibull", shape = 1.5,
                               scale = 10, censor_rate = 0.3, seed = 42) {
  set.seed(seed)
  if (is.null(beta)) beta <- runif(p, -0.5, 0.5)
  X <- matrix(rnorm(n * p), n, p)
  colnames(X) <- paste0("x", 1:p)
  eta <- as.vector(X %*% beta)
  if (distribution == "weibull") {
    u <- runif(n)
    true_time <- scale * exp(-eta / shape) * (-log(u))^(1 / shape)
  } else if (distribution == "exponential") {
    lambda <- exp(eta) / scale
    true_time <- rexp(n, lambda)
  } else if (distribution == "lognormal") {
    mu <- log(scale) - eta
    true_time <- exp(rnorm(n, mu, 1))
  } else {
    stop("Unknown distribution")
  }
  censor_time <- rexp(n, censor_rate / mean(true_time))
  time <- pmin(true_time, censor_time)
  event <- as.integer(true_time <= censor_time)
  data.frame(time = time, event = event, X)
}
