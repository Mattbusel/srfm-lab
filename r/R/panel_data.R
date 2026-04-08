##############################################################################
# panel_data.R -- Panel Data Econometrics for Financial Applications
# Dense implementation: FE/RE/GMM estimators, panel tests, Fama-MacBeth,
# panel VAR, Dumitrescu-Hurlin, Driscoll-Kraay, clustered SEs, etc.
##############################################################################

# ---------------------------------------------------------------------------
# Utility: demean within groups
# ---------------------------------------------------------------------------
demean_panel <- function(x, id) {
  grp_means <- tapply(x, id, mean, na.rm = TRUE)
  x - grp_means[as.character(id)]
}

demean_matrix <- function(X, id) {
  apply(X, 2, function(col) demean_panel(col, id))
}

# ---------------------------------------------------------------------------
# Panel structure helper
# ---------------------------------------------------------------------------
make_panel <- function(data, id_col, time_col) {
  data <- data[order(data[[id_col]], data[[time_col]]), ]
  ids <- data[[id_col]]
  times <- data[[time_col]]
  uid <- unique(ids)
  utimes <- sort(unique(times))
  N <- length(uid)
  TT <- length(utimes)
  list(data = data, ids = ids, times = times, uid = uid, utimes = utimes,
       N = N, TT = TT, n = nrow(data))
}

balanced_check <- function(panel) {
  tab <- table(panel$ids)
  all(tab == panel$TT)
}

# ---------------------------------------------------------------------------
# Fixed Effects: Within Estimator
# ---------------------------------------------------------------------------
fe_within <- function(y, X, id, time = NULL, weights = NULL) {
  n <- length(y)
  k <- ncol(X)
  uid <- unique(id)
  N <- length(uid)
  y_dm <- demean_panel(y, id)
  X_dm <- demean_matrix(X, id)
  if (!is.null(weights)) {
    w_sqrt <- sqrt(weights)
    y_dm <- y_dm * w_sqrt
    X_dm <- X_dm * w_sqrt
  }
  XtX <- crossprod(X_dm)
  Xty <- crossprod(X_dm, y_dm)
  XtX_inv <- tryCatch(solve(XtX), error = function(e) {
    MASS_ginv(XtX)
  })
  beta <- as.vector(XtX_inv %*% Xty)
  names(beta) <- colnames(X)
  resid <- y_dm - X_dm %*% beta
  df <- n - N - k
  sigma2 <- as.numeric(crossprod(resid) / df)
  vcov <- sigma2 * XtX_inv
  se <- sqrt(diag(vcov))
  tstat <- beta / se
  pval <- 2 * pt(-abs(tstat), df)
  y_bar_i <- tapply(y, id, mean, na.rm = TRUE)
  X_bar_i <- aggregate(X, by = list(id), FUN = mean, na.rm = TRUE)[, -1]
  alpha_i <- y_bar_i - as.vector(as.matrix(X_bar_i) %*% beta)
  fitted <- as.vector(X %*% beta) + alpha_i[as.character(id)]
  r2_within <- 1 - sum(resid^2) / sum(y_dm^2)
  r2_overall <- cor(fitted, y)^2
  list(coefficients = beta, se = se, tstat = tstat, pval = pval,
       vcov = vcov, sigma2 = sigma2, residuals = as.vector(resid),
       fitted = fitted, alpha_i = alpha_i, df = df, N = N, k = k, n = n,
       r2_within = r2_within, r2_overall = r2_overall, method = "FE-within")
}

# ---------------------------------------------------------------------------
# Fixed Effects: LSDV (Least Squares Dummy Variable)
# ---------------------------------------------------------------------------
fe_lsdv <- function(y, X, id) {
  uid <- unique(id)
  N <- length(uid)
  n <- length(y)
  k <- ncol(X)
  D <- matrix(0, n, N - 1)
  for (j in 2:N) {
    D[id == uid[j], j - 1] <- 1
  }
  Z <- cbind(X, D)
  ZtZ <- crossprod(Z)
  Zty <- crossprod(Z, y)
  ZtZ_inv <- solve(ZtZ)
  gamma <- as.vector(ZtZ_inv %*% Zty)
  beta <- gamma[1:k]
  names(beta) <- colnames(X)
  alpha_diff <- gamma[(k + 1):(k + N - 1)]
  resid <- y - Z %*% gamma
  df <- n - k - (N - 1) - 1
  sigma2 <- as.numeric(crossprod(resid) / df)
  vcov_full <- sigma2 * ZtZ_inv
  vcov_beta <- vcov_full[1:k, 1:k, drop = FALSE]
  se <- sqrt(diag(vcov_beta))
  tstat <- beta / se
  pval <- 2 * pt(-abs(tstat), df)
  list(coefficients = beta, se = se, tstat = tstat, pval = pval,
       vcov = vcov_beta, sigma2 = sigma2, residuals = as.vector(resid),
       alpha_diff = alpha_diff, df = df, method = "FE-LSDV")
}

# ---------------------------------------------------------------------------
# Pooled OLS (baseline)
# ---------------------------------------------------------------------------
pooled_ols <- function(y, X) {
  X1 <- cbind(1, X)
  XtX <- crossprod(X1)
  Xty <- crossprod(X1, y)
  beta <- as.vector(solve(XtX) %*% Xty)
  resid <- y - X1 %*% beta
  n <- length(y)
  k <- ncol(X1)
  sigma2 <- as.numeric(crossprod(resid) / (n - k))
  vcov <- sigma2 * solve(XtX)
  se <- sqrt(diag(vcov))
  names(beta) <- c("(Intercept)", colnames(X))
  list(coefficients = beta, se = se, vcov = vcov, sigma2 = sigma2,
       residuals = as.vector(resid), df = n - k, method = "Pooled-OLS")
}

# ---------------------------------------------------------------------------
# Random Effects: GLS (Swamy-Arora)
# ---------------------------------------------------------------------------
re_gls <- function(y, X, id, time = NULL) {
  n <- length(y)
  k <- ncol(X)
  uid <- unique(id)
  N <- length(uid)
  Ti_vec <- as.vector(table(id))
  fe_fit <- fe_within(y, X, id)
  sigma2_e <- fe_fit$sigma2
  pool_fit <- pooled_ols(y, X)
  resid_pool <- pool_fit$residuals
  id_means_resid <- tapply(resid_pool, id, mean, na.rm = TRUE)
  sigma2_between <- sum(Ti_vec * id_means_resid^2) / (N - k - 1) - sigma2_e / mean(Ti_vec)
  sigma2_u <- max(sigma2_between, 0)
  theta <- numeric(n)
  for (i in seq_along(uid)) {
    mask <- id == uid[i]
    Ti <- sum(mask)
    theta[mask] <- 1 - sqrt(sigma2_e / (Ti * sigma2_u + sigma2_e))
  }
  y_star <- y - theta * tapply(y, id, mean, na.rm = TRUE)[as.character(id)]
  X_star <- X - theta * do.call(rbind, lapply(uid, function(u) {
    mask <- id == u
    Ti <- sum(mask)
    matrix(colMeans(X[mask, , drop = FALSE]), nrow = Ti, ncol = k, byrow = TRUE)
  }))
  X1_star <- cbind(1 - theta, X_star)
  XtX <- crossprod(X1_star)
  Xty <- crossprod(X1_star, y_star)
  gamma <- as.vector(solve(XtX) %*% Xty)
  beta <- gamma[-1]
  intercept <- gamma[1]
  names(beta) <- colnames(X)
  resid <- y - intercept - as.vector(X %*% beta)
  df <- n - k - 1
  sigma2 <- as.numeric(crossprod(resid) / df)
  vcov <- sigma2 * solve(XtX)
  vcov_beta <- vcov[-1, -1, drop = FALSE]
  se <- sqrt(diag(vcov_beta))
  tstat <- beta / se
  pval <- 2 * pt(-abs(tstat), df)
  list(coefficients = beta, intercept = intercept, se = se, tstat = tstat,
       pval = pval, vcov = vcov_beta, sigma2_e = sigma2_e, sigma2_u = sigma2_u,
       theta = theta, residuals = resid, df = df, N = N, n = n,
       method = "RE-GLS")
}

# ---------------------------------------------------------------------------
# Hausman Test: FE vs RE
# ---------------------------------------------------------------------------
hausman_test <- function(y, X, id) {
  fe <- fe_within(y, X, id)
  re <- re_gls(y, X, id)
  b_fe <- fe$coefficients
  b_re <- re$coefficients
  k <- length(b_fe)
  diff_b <- b_fe - b_re
  diff_V <- fe$vcov - re$vcov
  diff_V_inv <- tryCatch(solve(diff_V), error = function(e) {
    MASS::ginv(diff_V)
  })
  chi2 <- as.numeric(t(diff_b) %*% diff_V_inv %*% diff_b)
  df <- k
  pval <- 1 - pchisq(chi2, df)
  decision <- if (pval < 0.05) "FE preferred" else "RE preferred"
  list(chi2 = chi2, df = df, pval = pval, decision = decision,
       b_fe = b_fe, b_re = b_re, method = "Hausman")
}

# ---------------------------------------------------------------------------
# First Differences Estimator
# ---------------------------------------------------------------------------
first_diff <- function(y, X, id, time) {
  ord <- order(id, time)
  y <- y[ord]; X <- X[ord, , drop = FALSE]; id <- id[ord]; time <- time[ord]
  uid <- unique(id)
  dy_list <- list(); dX_list <- list()
  for (u in uid) {
    mask <- id == u
    yi <- y[mask]; Xi <- X[mask, , drop = FALSE]
    Ti <- sum(mask)
    if (Ti < 2) next
    dy_list[[length(dy_list) + 1]] <- diff(yi)
    dX_list[[length(dX_list) + 1]] <- diff(as.matrix(Xi))
  }
  dy <- unlist(dy_list)
  dX <- do.call(rbind, dX_list)
  n <- length(dy); k <- ncol(dX)
  dX1 <- cbind(1, dX)
  XtX <- crossprod(dX1)
  Xty <- crossprod(dX1, dy)
  gamma <- as.vector(solve(XtX) %*% Xty)
  resid <- dy - dX1 %*% gamma
  df <- n - ncol(dX1)
  sigma2 <- as.numeric(crossprod(resid) / df)
  vcov <- sigma2 * solve(XtX)
  se <- sqrt(diag(vcov))
  tstat <- gamma / se
  pval <- 2 * pt(-abs(tstat), df)
  names(gamma) <- c("(Intercept)", colnames(X))
  list(coefficients = gamma, se = se, tstat = tstat, pval = pval,
       vcov = vcov, sigma2 = sigma2, residuals = as.vector(resid),
       df = df, method = "FirstDiff")
}

# ---------------------------------------------------------------------------
# Clustered Standard Errors
# ---------------------------------------------------------------------------
cluster_se <- function(X, resid, cluster) {
  X1 <- if (is.null(dim(X))) cbind(1, X) else cbind(1, X)
  n <- nrow(X1); k <- ncol(X1)
  XtX_inv <- solve(crossprod(X1))
  ucl <- unique(cluster)
  G <- length(ucl)
  meat <- matrix(0, k, k)
  for (g in ucl) {
    mask <- cluster == g
    Xg <- X1[mask, , drop = FALSE]
    eg <- resid[mask]
    score_g <- crossprod(Xg, eg)
    meat <- meat + score_g %*% t(score_g)
  }
  scale <- (G / (G - 1)) * ((n - 1) / (n - k))
  vcov_cl <- scale * XtX_inv %*% meat %*% XtX_inv
  sqrt(diag(vcov_cl))
}

cluster_se_entity <- function(model, id) {
  cluster_se(model$X_used, model$residuals, id)
}

cluster_se_time <- function(model, time) {
  cluster_se(model$X_used, model$residuals, time)
}

# ---------------------------------------------------------------------------
# Two-way Clustered Standard Errors (Cameron-Gelbach-Miller)
# ---------------------------------------------------------------------------
twoway_cluster_se <- function(X, resid, cl1, cl2) {
  X1 <- cbind(1, X)
  n <- nrow(X1); k <- ncol(X1)
  XtX_inv <- solve(crossprod(X1))
  compute_meat <- function(cluster) {
    ucl <- unique(cluster)
    G <- length(ucl)
    meat <- matrix(0, k, k)
    for (g in ucl) {
      mask <- cluster == g
      Xg <- X1[mask, , drop = FALSE]
      eg <- resid[mask]
      sg <- crossprod(Xg, eg)
      meat <- meat + sg %*% t(sg)
    }
    sc <- (G / (G - 1)) * ((n - 1) / (n - k))
    sc * XtX_inv %*% meat %*% XtX_inv
  }
  V1 <- compute_meat(cl1)
  V2 <- compute_meat(cl2)
  cl_int <- paste(cl1, cl2, sep = "_")
  V12 <- compute_meat(cl_int)
  vcov_tw <- V1 + V2 - V12
  vcov_tw <- (vcov_tw + t(vcov_tw)) / 2
  ev <- eigen(vcov_tw, symmetric = TRUE)$values
  if (any(ev < 0)) {
    vcov_tw <- vcov_tw - min(ev) * diag(k)
  }
  list(vcov = vcov_tw, se = sqrt(diag(vcov_tw)))
}

# ---------------------------------------------------------------------------
# Instrumental Variables: 2SLS Within (Panel IV)
# ---------------------------------------------------------------------------
panel_iv_2sls <- function(y, X_endog, X_exog, Z_instruments, id) {
  y_dm <- demean_panel(y, id)
  Xe_dm <- demean_matrix(X_endog, id)
  Xx_dm <- if (!is.null(X_exog)) demean_matrix(X_exog, id) else NULL
  Z_dm <- demean_matrix(Z_instruments, id)
  if (!is.null(Xx_dm)) {
    Z_full <- cbind(Xx_dm, Z_dm)
    X_full <- cbind(Xe_dm, Xx_dm)
  } else {
    Z_full <- Z_dm
    X_full <- Xe_dm
  }
  n <- nrow(X_full); k <- ncol(X_full)
  uid <- unique(id); N <- length(uid)
  Pz <- Z_full %*% solve(crossprod(Z_full)) %*% t(Z_full)
  X_hat <- Pz %*% X_full
  XhX <- crossprod(X_hat, X_full)
  Xhy <- crossprod(X_hat, y_dm)
  beta <- as.vector(solve(XhX) %*% Xhy)
  resid <- y_dm - X_full %*% beta
  df <- n - N - k
  sigma2 <- as.numeric(crossprod(resid) / df)
  XhXh <- crossprod(X_hat)
  vcov <- sigma2 * solve(XhXh)
  se <- sqrt(diag(vcov))
  tstat <- beta / se
  pval <- 2 * pt(-abs(tstat), df)
  names(beta) <- colnames(X_full)
  sargan_stat <- NA; sargan_pval <- NA
  n_inst <- ncol(Z_full)
  n_endog <- ncol(X_endog)
  if (n_inst > n_endog) {
    u_hat <- resid
    ZtZ_inv <- solve(crossprod(Z_dm))
    Zu <- crossprod(Z_dm, u_hat)
    sargan_stat <- as.numeric(t(Zu) %*% ZtZ_inv %*% Zu) / sigma2
    sargan_df <- n_inst - n_endog
    sargan_pval <- 1 - pchisq(sargan_stat, sargan_df)
  }
  first_stage_F <- numeric(ncol(X_endog))
  for (j in 1:ncol(X_endog)) {
    dep <- Xe_dm[, j]
    fit_full <- lm.fit(Z_full, dep)
    fit_restr <- if (!is.null(Xx_dm)) lm.fit(Xx_dm, dep) else list(residuals = dep)
    ssr_full <- sum(fit_full$residuals^2)
    ssr_restr <- sum(fit_restr$residuals^2)
    q <- ncol(Z_dm)
    first_stage_F[j] <- ((ssr_restr - ssr_full) / q) / (ssr_full / (n - ncol(Z_full)))
  }
  list(coefficients = beta, se = se, tstat = tstat, pval = pval,
       vcov = vcov, sigma2 = sigma2, residuals = as.vector(resid),
       sargan = sargan_stat, sargan_pval = sargan_pval,
       first_stage_F = first_stage_F, df = df, method = "Panel-2SLS")
}

# ---------------------------------------------------------------------------
# Arellano-Bond GMM Estimator for Dynamic Panels
# ---------------------------------------------------------------------------
arellano_bond_gmm <- function(y, X, id, time, lags = 1, gmm_lags = 2:99,
                               twostep = TRUE) {
  ord <- order(id, time)
  y <- y[ord]; X <- X[ord, , drop = FALSE]; id <- id[ord]; time <- time[ord]
  uid <- unique(id)
  N <- length(uid)
  utimes <- sort(unique(time))
  TT <- length(utimes)
  time_idx <- match(time, utimes)
  build_entity_data <- function(u) {
    mask <- id == u
    yi <- y[mask]; Xi <- X[mask, , drop = FALSE]; ti <- time_idx[mask]
    Ti <- length(yi)
    if (Ti < lags + 2) return(NULL)
    dy <- diff(yi)
    dX <- diff(as.matrix(Xi))
    dti <- ti[-1]
    lagged_dy <- dy[-length(dy)]
    dy <- dy[-1]
    dX <- dX[-1, , drop = FALSE]
    dti <- dti[-1]
    Zi_rows <- list()
    valid <- rep(TRUE, length(dy))
    for (s in seq_along(dy)) {
      t_s <- dti[s]
      max_lag <- min(t_s - 1, max(gmm_lags))
      min_lag <- min(gmm_lags)
      if (max_lag < min_lag) { valid[s] <- FALSE; next }
      inst_idx <- which(ti <= (t_s - min_lag) & ti >= (t_s - max_lag))
      if (length(inst_idx) == 0) { valid[s] <- FALSE; next }
      Zi_rows[[s]] <- yi[mask][inst_idx]
    }
    dy <- dy[valid]; dX <- dX[valid, , drop = FALSE]
    lagged_dy <- lagged_dy[valid]
    Zi_rows <- Zi_rows[valid]
    list(dy = dy, dX = dX, lagged_dy = lagged_dy, instruments = Zi_rows)
  }
  entity_data <- lapply(uid, build_entity_data)
  entity_data <- entity_data[!sapply(entity_data, is.null)]
  if (length(entity_data) == 0) stop("Insufficient data for AB-GMM")
  max_inst <- max(sapply(entity_data, function(ed)
    max(sapply(ed$instruments, length))))
  build_block_diag_Z <- function(ed) {
    ns <- length(ed$dy)
    rows <- list()
    for (s in 1:ns) {
      row <- rep(0, max_inst)
      iv <- ed$instruments[[s]]
      row[1:length(iv)] <- iv
      rows[[s]] <- row
    }
    do.call(rbind, rows)
  }
  all_dy <- unlist(lapply(entity_data, function(ed) ed$dy))
  all_lagged_dy <- unlist(lapply(entity_data, function(ed) ed$lagged_dy))
  all_dX <- do.call(rbind, lapply(entity_data, function(ed) ed$dX))
  W_full <- cbind(all_lagged_dy, all_dX)
  Z_list <- lapply(entity_data, build_block_diag_Z)
  Z_all <- do.call(rbind, Z_list)
  n_obs <- length(all_dy)
  gmm_one_step <- function(W_mat) {
    ZtW <- crossprod(Z_all, W_full)
    Zty <- crossprod(Z_all, all_dy)
    ZtZ <- crossprod(Z_all)
    W_inv <- tryCatch(solve(W_mat), error = function(e) {
      diag(1, nrow(W_mat))
    })
    A <- t(ZtW) %*% W_inv %*% ZtW
    b <- t(ZtW) %*% W_inv %*% Zty
    beta <- as.vector(solve(A) %*% b)
    resid <- all_dy - W_full %*% beta
    list(beta = beta, resid = resid)
  }
  H <- diag(n_obs)
  for (i in 2:n_obs) H[i, i - 1] <- -1; H[1, 1] <- 2
  for (i in 1:(n_obs - 1)) H[i, i + 1] <- -1
  W1 <- crossprod(Z_all, H[1:n_obs, 1:n_obs, drop = FALSE]) %*% Z_all / N
  step1 <- gmm_one_step(W1)
  if (twostep) {
    e1 <- step1$resid
    meat2 <- matrix(0, ncol(Z_all), ncol(Z_all))
    offset <- 0
    for (ed in entity_data) {
      ni <- length(ed$dy)
      idx <- (offset + 1):(offset + ni)
      Zi <- Z_all[idx, , drop = FALSE]
      ei <- e1[idx]
      meat2 <- meat2 + crossprod(Zi, ei %*% t(ei)) %*% Zi
      offset <- offset + ni
    }
    W2 <- meat2 / N
    step2 <- gmm_one_step(W2)
    final <- step2
  } else {
    final <- step1
  }
  beta <- final$beta
  resid <- final$resid
  k <- length(beta)
  names(beta) <- c("y_lag1", colnames(X))
  sigma2 <- as.numeric(crossprod(resid) / (n_obs - k))
  ZtW <- crossprod(Z_all, W_full)
  if (twostep) {
    W_inv <- tryCatch(solve(W2), error = function(e) diag(1, nrow(W2)))
  } else {
    W_inv <- tryCatch(solve(W1), error = function(e) diag(1, nrow(W1)))
  }
  bread <- solve(t(ZtW) %*% W_inv %*% ZtW)
  vcov <- bread * n_obs
  se <- sqrt(diag(abs(vcov)))
  tstat <- beta / se
  pval <- 2 * pnorm(-abs(tstat))
  m1_stat <- NA; m2_stat <- NA
  list(coefficients = beta, se = se, tstat = tstat, pval = pval,
       vcov = vcov, residuals = resid, n_obs = n_obs, N = length(entity_data),
       ar1_test = m1_stat, ar2_test = m2_stat, method = "Arellano-Bond-GMM")
}

# ---------------------------------------------------------------------------
# Panel Unit Root: Levin-Lin-Chu
# ---------------------------------------------------------------------------
llc_test <- function(y, id, time, lags = 1, trend = FALSE) {
  ord <- order(id, time)
  y <- y[ord]; id <- id[ord]; time <- time[ord]
  uid <- unique(id)
  N <- length(uid)
  delta_y_all <- c(); y_lag_all <- c()
  for (u in uid) {
    mask <- id == u
    yi <- y[mask]
    Ti <- length(yi)
    if (Ti < lags + 3) next
    dy <- diff(yi)
    y_lag <- yi[-Ti]
    if (lags > 0) {
      dy_lags <- embed(dy, lags + 1)
      dy_dep <- dy_lags[, 1]
      dy_lagged <- dy_lags[, -1, drop = FALSE]
      y_lag <- y_lag[(lags + 1):length(y_lag)]
      fit_aux <- lm.fit(cbind(1, dy_lagged), dy_dep)
      e_tilde <- fit_aux$residuals
      fit_y <- lm.fit(cbind(1, dy_lagged), y_lag)
      v_tilde <- fit_y$residuals
    } else {
      e_tilde <- dy[-1]
      v_tilde <- y_lag[-length(y_lag)]
    }
    si <- sqrt(sum(e_tilde^2) / (length(e_tilde) - 1))
    delta_y_all <- c(delta_y_all, e_tilde / si)
    y_lag_all <- c(y_lag_all, v_tilde / si)
  }
  n_total <- length(delta_y_all)
  fit <- lm.fit(cbind(1, y_lag_all), delta_y_all)
  delta_hat <- fit$coefficients[2]
  se_delta <- sqrt(sum(fit$residuals^2) / (n_total - 2)) /
    sqrt(sum((y_lag_all - mean(y_lag_all))^2))
  t_star <- delta_hat / se_delta
  mu_star <- -0.5 * N
  sigma_star <- sqrt(N / 3)
  z_star <- (t_star - mu_star) / sigma_star
  pval <- pnorm(z_star)
  list(t_star = t_star, z_star = z_star, pval = pval,
       delta_hat = delta_hat, N = N, method = "LLC")
}

# ---------------------------------------------------------------------------
# Panel Unit Root: Im-Pesaran-Shin (IPS)
# ---------------------------------------------------------------------------
ips_test <- function(y, id, time, lags = 1) {
  ord <- order(id, time)
  y <- y[ord]; id <- id[ord]; time <- time[ord]
  uid <- unique(id)
  N <- length(uid)
  t_bars <- numeric(N)
  Ti_vec <- numeric(N)
  for (i in seq_along(uid)) {
    mask <- id == uid[i]
    yi <- y[mask]
    Ti <- length(yi)
    Ti_vec[i] <- Ti
    if (Ti < lags + 3) { t_bars[i] <- NA; next }
    dy <- diff(yi)
    y_lag <- yi[-Ti]
    if (lags > 0) {
      emb <- embed(dy, lags + 1)
      dep <- emb[, 1]
      regs <- cbind(1, y_lag[(lags + 1):length(y_lag)], emb[, -1, drop = FALSE])
    } else {
      dep <- dy
      regs <- cbind(1, y_lag)
    }
    fit <- lm.fit(regs, dep)
    coefs <- fit$coefficients
    resid <- fit$residuals
    se_coef <- sqrt(sum(resid^2) / (length(dep) - ncol(regs))) /
      sqrt(sum((regs[, 2] - mean(regs[, 2]))^2))
    t_bars[i] <- coefs[2] / se_coef
  }
  valid <- !is.na(t_bars)
  t_bars <- t_bars[valid]
  N_valid <- sum(valid)
  t_bar <- mean(t_bars)
  E_t <- -1.52
  V_t <- 0.74
  W_tbar <- sqrt(N_valid) * (t_bar - E_t) / sqrt(V_t)
  pval <- pnorm(W_tbar)
  list(t_bar = t_bar, W_tbar = W_tbar, pval = pval, N = N_valid,
       individual_t = t_bars, method = "IPS")
}

# ---------------------------------------------------------------------------
# Panel Unit Root: Fisher-ADF (Maddala-Wu)
# ---------------------------------------------------------------------------
fisher_adf_test <- function(y, id, time, lags = 1) {
  ord <- order(id, time)
  y <- y[ord]; id <- id[ord]; time <- time[ord]
  uid <- unique(id)
  N <- length(uid)
  pvals_i <- numeric(N)
  for (i in seq_along(uid)) {
    mask <- id == uid[i]
    yi <- y[mask]
    Ti <- length(yi)
    if (Ti < lags + 4) { pvals_i[i] <- 0.5; next }
    dy <- diff(yi)
    y_lag <- yi[-Ti]
    if (lags > 0) {
      emb <- embed(dy, lags + 1)
      dep <- emb[, 1]
      regs <- cbind(1, y_lag[(lags + 1):length(y_lag)], emb[, -1, drop = FALSE])
    } else {
      dep <- dy
      regs <- cbind(1, y_lag)
    }
    fit <- lm.fit(regs, dep)
    rho_hat <- fit$coefficients[2]
    resid <- fit$residuals
    se_rho <- sqrt(sum(resid^2) / (length(dep) - ncol(regs))) /
      sqrt(sum((regs[, 2] - mean(regs[, 2]))^2))
    t_stat <- rho_hat / se_rho
    df_eff <- length(dep) - ncol(regs)
    pvals_i[i] <- pt(t_stat, df_eff)
  }
  P_stat <- -2 * sum(log(pvals_i))
  df_chi <- 2 * N
  pval <- 1 - pchisq(P_stat, df_chi)
  Z_stat <- (P_stat - df_chi) / sqrt(2 * df_chi)
  list(P = P_stat, Z = Z_stat, pval = pval, df = df_chi,
       individual_pvals = pvals_i, N = N, method = "Fisher-ADF")
}

# ---------------------------------------------------------------------------
# Panel Cointegration: Pedroni Test
# ---------------------------------------------------------------------------
pedroni_test <- function(y, X, id, time, lags = 1) {
  ord <- order(id, time)
  y <- y[ord]; X <- X[ord, , drop = FALSE]; id <- id[ord]; time <- time[ord]
  uid <- unique(id)
  N <- length(uid)
  panel_rho <- 0; panel_t <- 0; panel_pp <- 0; panel_adf <- 0
  group_rho <- 0; group_t <- 0; group_adf <- 0
  n_total <- 0
  for (u in uid) {
    mask <- id == u
    yi <- y[mask]; Xi <- X[mask, , drop = FALSE]
    Ti <- length(yi)
    if (Ti < 5) next
    fit <- lm.fit(cbind(1, as.matrix(Xi)), yi)
    ei <- fit$residuals
    ei_lag <- ei[-Ti]
    dei <- diff(ei)
    n_i <- length(dei)
    if (n_i < 3) next
    rho_i <- sum(ei_lag * dei) / sum(ei_lag^2)
    resid_i <- dei - rho_i * ei_lag
    s2_i <- sum(resid_i^2) / (n_i - 1)
    se_rho_i <- sqrt(s2_i / sum(ei_lag^2))
    t_rho_i <- rho_i / se_rho_i
    panel_rho <- panel_rho + sum(ei_lag * dei)
    panel_t <- panel_t + t_rho_i
    group_rho <- group_rho + Ti * rho_i
    group_t <- group_t + t_rho_i
    n_total <- n_total + n_i
    if (lags > 0 && n_i > lags + 2) {
      emb <- embed(dei, lags + 1)
      dep_adf <- emb[, 1]
      regs_adf <- cbind(ei_lag[(lags + 1):length(ei_lag)], emb[, -1, drop = FALSE])
      fit_adf <- lm.fit(regs_adf, dep_adf)
      rho_adf <- fit_adf$coefficients[1]
      resid_adf <- fit_adf$residuals
      se_adf <- sqrt(sum(resid_adf^2) / (length(dep_adf) - ncol(regs_adf))) /
        sqrt(sum((regs_adf[, 1] - mean(regs_adf[, 1]))^2))
      group_adf <- group_adf + rho_adf / se_adf
    }
  }
  panel_rho_stat <- (panel_rho - N * 0.5) / sqrt(N / 3)
  group_rho_stat <- (group_rho - N * 0.5) / sqrt(N / 3)
  panel_t_stat <- panel_t / sqrt(N)
  group_t_stat <- group_t / sqrt(N)
  group_adf_stat <- group_adf / sqrt(N)
  list(panel_rho = panel_rho_stat, panel_t = panel_t_stat,
       group_rho = group_rho_stat, group_t = group_t_stat,
       group_adf = group_adf_stat, N = N, method = "Pedroni")
}

# ---------------------------------------------------------------------------
# Panel Cointegration: Kao Test
# ---------------------------------------------------------------------------
kao_test <- function(y, X, id, time) {
  ord <- order(id, time)
  y <- y[ord]; X <- X[ord, , drop = FALSE]; id <- id[ord]; time <- time[ord]
  fe <- fe_within(y, X, id)
  ei <- fe$residuals
  uid <- unique(id)
  N <- length(uid)
  ei_lag <- c(); dei <- c()
  for (u in uid) {
    mask <- id == u
    e_u <- ei[mask]
    Ti <- length(e_u)
    if (Ti < 2) next
    ei_lag <- c(ei_lag, e_u[-Ti])
    dei <- c(dei, diff(e_u))
  }
  rho <- sum(ei_lag * dei) / sum(ei_lag^2)
  resid <- dei - rho * ei_lag
  n <- length(dei)
  sigma2 <- sum(resid^2) / n
  se_rho <- sqrt(sigma2 / sum(ei_lag^2))
  t_rho <- rho / se_rho
  ADF_stat <- t_rho
  pval <- pnorm(ADF_stat)
  list(ADF = ADF_stat, pval = pval, rho = rho, N = N, method = "Kao")
}

# ---------------------------------------------------------------------------
# Pesaran CD Test for Cross-Sectional Dependence
# ---------------------------------------------------------------------------
pesaran_cd_test <- function(residuals_matrix) {
  N <- ncol(residuals_matrix)
  TT <- nrow(residuals_matrix)
  rho_sum <- 0
  count <- 0
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      valid <- complete.cases(residuals_matrix[, i], residuals_matrix[, j])
      if (sum(valid) < 3) next
      ri <- residuals_matrix[valid, i]
      rj <- residuals_matrix[valid, j]
      Tij <- sum(valid)
      rho_ij <- cor(ri, rj)
      rho_sum <- rho_sum + sqrt(Tij) * rho_ij
      count <- count + 1
    }
  }
  CD <- sqrt(2 / (N * (N - 1))) * rho_sum
  pval <- 2 * (1 - pnorm(abs(CD)))
  list(CD = CD, pval = pval, N = N, TT = TT, method = "Pesaran-CD")
}

# ---------------------------------------------------------------------------
# Frees Test for Cross-Sectional Dependence
# ---------------------------------------------------------------------------
frees_test <- function(residuals_matrix) {
  N <- ncol(residuals_matrix)
  TT <- nrow(residuals_matrix)
  ranks <- apply(residuals_matrix, 2, function(col) rank(col, na.last = "keep"))
  rho_sq_sum <- 0
  count <- 0
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      valid <- complete.cases(ranks[, i], ranks[, j])
      if (sum(valid) < 3) next
      ri <- ranks[valid, i]
      rj <- ranks[valid, j]
      rho_s <- cor(ri, rj, method = "spearman")
      rho_sq_sum <- rho_sq_sum + rho_s^2
      count <- count + 1
    }
  }
  avg_rho_sq <- rho_sq_sum / count
  FRE <- N * avg_rho_sq
  Q <- (TT - 1) / (TT + 2) * (1 / (TT - 1))
  test_stat <- (avg_rho_sq - Q) * sqrt(count)
  list(FRE = FRE, test_stat = test_stat, avg_rho_sq = avg_rho_sq,
       N = N, TT = TT, method = "Frees")
}

# ---------------------------------------------------------------------------
# Driscoll-Kraay Standard Errors
# ---------------------------------------------------------------------------
driscoll_kraay_se <- function(y, X, id, time, max_lag = NULL) {
  n <- length(y)
  k <- ncol(X) + 1
  X1 <- cbind(1, X)
  uid <- unique(id)
  utimes <- sort(unique(time))
  TT <- length(utimes)
  N <- length(uid)
  if (is.null(max_lag)) max_lag <- floor(TT^(1 / 4))
  XtX_inv <- solve(crossprod(X1))
  fit <- lm.fit(X1, y)
  beta <- fit$coefficients
  resid <- fit$residuals
  h_t <- matrix(0, TT, k)
  for (tt in seq_along(utimes)) {
    mask <- time == utimes[tt]
    Xt <- X1[mask, , drop = FALSE]
    et <- resid[mask]
    h_t[tt, ] <- colSums(Xt * et)
  }
  S <- crossprod(h_t) / TT
  if (max_lag > 0) {
    for (l in 1:max_lag) {
      w <- 1 - l / (max_lag + 1)
      Gamma_l <- matrix(0, k, k)
      for (tt in (l + 1):TT) {
        Gamma_l <- Gamma_l + h_t[tt, ] %o% h_t[tt - l, ]
      }
      Gamma_l <- Gamma_l / TT
      S <- S + w * (Gamma_l + t(Gamma_l))
    }
  }
  vcov <- XtX_inv %*% S %*% XtX_inv * TT
  se <- sqrt(diag(vcov))
  list(coefficients = beta, se = se, vcov = vcov, max_lag = max_lag,
       method = "Driscoll-Kraay")
}

# ---------------------------------------------------------------------------
# Newey-West Panel Long-Run Variance
# ---------------------------------------------------------------------------
newey_west_panel <- function(X, resid, id, time, max_lag = NULL) {
  n <- length(resid)
  X1 <- cbind(1, X)
  k <- ncol(X1)
  utimes <- sort(unique(time))
  TT <- length(utimes)
  if (is.null(max_lag)) max_lag <- floor(TT^(1 / 3))
  XtX_inv <- solve(crossprod(X1))
  S0 <- crossprod(X1 * resid) / n
  S <- S0
  for (l in 1:max_lag) {
    w <- 1 - l / (max_lag + 1)
    Gamma_l <- matrix(0, k, k)
    for (i in unique(id)) {
      mask <- id == i
      Xi <- X1[mask, , drop = FALSE]
      ei <- resid[mask]
      Ti <- sum(mask)
      if (Ti <= l) next
      for (tt in (l + 1):Ti) {
        Gamma_l <- Gamma_l + (Xi[tt, ] * ei[tt]) %o% (Xi[tt - l, ] * ei[tt - l])
      }
    }
    Gamma_l <- Gamma_l / n
    S <- S + w * (Gamma_l + t(Gamma_l))
  }
  vcov <- n * XtX_inv %*% S %*% XtX_inv
  list(vcov = vcov, se = sqrt(diag(vcov)), max_lag = max_lag)
}

# ---------------------------------------------------------------------------
# Fama-MacBeth Cross-Sectional Regressions
# ---------------------------------------------------------------------------
fama_macbeth <- function(y, X, id, time, nw_lags = 0) {
  utimes <- sort(unique(time))
  TT <- length(utimes)
  k <- ncol(X) + 1
  gamma_mat <- matrix(NA, TT, k)
  r2_vec <- numeric(TT)
  n_vec <- integer(TT)
  for (tt in seq_along(utimes)) {
    mask <- time == utimes[tt]
    yt <- y[mask]
    Xt <- cbind(1, X[mask, , drop = FALSE])
    nt <- sum(mask)
    if (nt < k + 1) next
    n_vec[tt] <- nt
    fit <- lm.fit(Xt, yt)
    gamma_mat[tt, ] <- fit$coefficients
    ss_res <- sum(fit$residuals^2)
    ss_tot <- sum((yt - mean(yt))^2)
    r2_vec[tt] <- 1 - ss_res / ss_tot
  }
  valid <- complete.cases(gamma_mat)
  gamma_valid <- gamma_mat[valid, , drop = FALSE]
  TT_valid <- nrow(gamma_valid)
  gamma_bar <- colMeans(gamma_valid)
  if (nw_lags == 0) {
    gamma_var <- apply(gamma_valid, 2, var) / TT_valid
  } else {
    gamma_var <- numeric(k)
    for (j in 1:k) {
      gj <- gamma_valid[, j]
      gj_dm <- gj - mean(gj)
      v <- sum(gj_dm^2)
      for (l in 1:nw_lags) {
        w <- 1 - l / (nw_lags + 1)
        cov_l <- sum(gj_dm[(l + 1):TT_valid] * gj_dm[1:(TT_valid - l)])
        v <- v + 2 * w * cov_l
      }
      gamma_var[j] <- v / TT_valid^2
    }
  }
  se <- sqrt(gamma_var)
  tstat <- gamma_bar / se
  pval <- 2 * pt(-abs(tstat), TT_valid - 1)
  names(gamma_bar) <- c("(Intercept)", colnames(X))
  list(coefficients = gamma_bar, se = se, tstat = tstat, pval = pval,
       gamma_ts = gamma_mat, r2_ts = r2_vec, n_ts = n_vec,
       TT = TT_valid, method = "Fama-MacBeth")
}

# ---------------------------------------------------------------------------
# Panel VAR: VAR with Fixed Effects
# ---------------------------------------------------------------------------
panel_var <- function(data_list, id, time, p = 1) {
  # data_list: list of variable vectors, each named
  K <- length(data_list)
  var_names <- names(data_list)
  uid <- unique(id)
  N <- length(uid)
  Y_list <- list(); X_list <- list()
  for (u in uid) {
    mask <- id == u
    ord_m <- order(time[mask])
    Y_u <- sapply(data_list, function(v) v[mask][ord_m])
    Ti <- nrow(Y_u)
    if (Ti < p + 2) next
    for (tt in (p + 1):Ti) {
      y_row <- Y_u[tt, ]
      x_row <- c()
      for (l in 1:p) {
        x_row <- c(x_row, Y_u[tt - l, ])
      }
      Y_list[[length(Y_list) + 1]] <- y_row
      X_list[[length(X_list) + 1]] <- x_row
    }
  }
  Y <- do.call(rbind, Y_list)
  X <- do.call(rbind, X_list)
  n <- nrow(Y)
  # Demean by entity for FE
  Y_dm <- Y  # simplified: assume demeaning done upstream
  X_dm <- X
  coef_names <- c()
  for (l in 1:p) {
    for (vn in var_names) coef_names <- c(coef_names, paste0(vn, "_L", l))
  }
  colnames(X_dm) <- coef_names
  B <- matrix(0, K, K * p)
  Sigma <- matrix(0, K, K)
  resid_mat <- matrix(0, n, K)
  for (eq in 1:K) {
    fit <- lm.fit(X_dm, Y_dm[, eq])
    B[eq, ] <- fit$coefficients
    resid_mat[, eq] <- fit$residuals
  }
  rownames(B) <- var_names
  colnames(B) <- coef_names
  Sigma <- crossprod(resid_mat) / (n - K * p)
  companion <- matrix(0, K * p, K * p)
  companion[1:K, ] <- B
  if (p > 1) {
    companion[(K + 1):(K * p), 1:(K * (p - 1))] <- diag(K * (p - 1))
  }
  eigenvalues <- eigen(companion)$values
  stable <- all(Mod(eigenvalues) < 1)
  irf <- compute_panel_irf(B, Sigma, K, p, horizon = 20)
  list(B = B, Sigma = Sigma, companion = companion, eigenvalues = eigenvalues,
       stable = stable, residuals = resid_mat, irf = irf,
       var_names = var_names, p = p, n = n, method = "Panel-VAR")
}

compute_panel_irf <- function(B, Sigma, K, p, horizon = 20) {
  chol_Sigma <- tryCatch(t(chol(Sigma)), error = function(e) diag(K))
  Phi <- array(0, dim = c(K, K, horizon + 1))
  Phi[, , 1] <- diag(K)
  A_list <- list()
  for (l in 1:p) {
    A_list[[l]] <- B[, ((l - 1) * K + 1):(l * K)]
  }
  for (h in 1:horizon) {
    for (l in 1:min(h, p)) {
      Phi[, , h + 1] <- Phi[, , h + 1] + Phi[, , h - l + 1] %*% A_list[[l]]
    }
  }
  irf_orth <- array(0, dim = c(K, K, horizon + 1))
  for (h in 0:horizon) {
    irf_orth[, , h + 1] <- Phi[, , h + 1] %*% chol_Sigma
  }
  list(irf = Phi, irf_orthogonal = irf_orth, horizon = horizon)
}

# ---------------------------------------------------------------------------
# Dumitrescu-Hurlin Panel Granger Causality Test
# ---------------------------------------------------------------------------
dumitrescu_hurlin_test <- function(y, x, id, time, lags = 1) {
  ord <- order(id, time)
  y <- y[ord]; x <- x[ord]; id <- id[ord]; time <- time[ord]
  uid <- unique(id)
  N <- length(uid)
  W_i <- numeric(N)
  Ti_vec <- integer(N)
  for (i in seq_along(uid)) {
    mask <- id == uid[i]
    yi <- y[mask]; xi <- x[mask]
    Ti <- length(yi)
    Ti_vec[i] <- Ti
    if (Ti < 2 * lags + 3) { W_i[i] <- NA; next }
    dep <- yi[(lags + 1):Ti]
    regs_r <- matrix(0, length(dep), lags)
    regs_u <- matrix(0, length(dep), 2 * lags)
    for (l in 1:lags) {
      regs_r[, l] <- yi[(lags + 1 - l):(Ti - l)]
      regs_u[, l] <- yi[(lags + 1 - l):(Ti - l)]
      regs_u[, lags + l] <- xi[(lags + 1 - l):(Ti - l)]
    }
    fit_r <- lm.fit(cbind(1, regs_r), dep)
    fit_u <- lm.fit(cbind(1, regs_u), dep)
    ssr_r <- sum(fit_r$residuals^2)
    ssr_u <- sum(fit_u$residuals^2)
    n_eff <- length(dep)
    k_u <- 2 * lags + 1
    W_i[i] <- ((ssr_r - ssr_u) / lags) / (ssr_u / (n_eff - k_u))
  }
  valid <- !is.na(W_i)
  W_valid <- W_i[valid]
  N_valid <- sum(valid)
  W_bar <- mean(W_valid)
  T_bar <- mean(Ti_vec[valid])
  Z_bar <- sqrt(N_valid / (2 * lags)) * (W_bar - lags)
  if (T_bar > 5 + 3 * lags) {
    E_Wi <- lags / (T_bar - 2 * lags - 1)
    V_Wi <- 2 * lags * (T_bar - lags - 1)^2 /
      ((T_bar - 2 * lags - 1)^2 * (T_bar - 2 * lags - 3))
    Z_tilde <- sqrt(N_valid) * (W_bar - E_Wi) / sqrt(V_Wi)
  } else {
    Z_tilde <- Z_bar
  }
  pval <- 2 * (1 - pnorm(abs(Z_tilde)))
  list(W_bar = W_bar, Z_bar = Z_bar, Z_tilde = Z_tilde, pval = pval,
       W_individual = W_i, N = N_valid, lags = lags,
       method = "Dumitrescu-Hurlin")
}

# ---------------------------------------------------------------------------
# F-test for Fixed Effects (poolability test)
# ---------------------------------------------------------------------------
f_test_fe <- function(y, X, id) {
  n <- length(y)
  k <- ncol(X)
  uid <- unique(id)
  N <- length(uid)
  pool <- pooled_ols(y, X)
  ssr_pool <- sum(pool$residuals^2)
  fe <- fe_within(y, X, id)
  ssr_fe <- sum(fe$residuals^2)
  df1 <- N - 1
  df2 <- n - N - k
  F_stat <- ((ssr_pool - ssr_fe) / df1) / (ssr_fe / df2)
  pval <- 1 - pf(F_stat, df1, df2)
  list(F_stat = F_stat, df1 = df1, df2 = df2, pval = pval, method = "F-test-FE")
}

# ---------------------------------------------------------------------------
# Breusch-Pagan LM test for RE
# ---------------------------------------------------------------------------
bp_lm_test <- function(y, X, id) {
  pool <- pooled_ols(y, X)
  e <- pool$residuals
  uid <- unique(id)
  N <- length(uid)
  n <- length(y)
  sum_sq_group <- 0
  for (u in uid) {
    mask <- id == u
    sum_sq_group <- sum_sq_group + (sum(e[mask]))^2
  }
  LM <- (n / (2 * (length(unique(table(id))) - 1))) *
    (sum_sq_group / sum(e^2) - 1)^2
  Ti_vec <- as.vector(table(id))
  A <- sum(Ti_vec^2)
  num <- (n^2 / A) * (sum_sq_group / sum(e^2) - 1)^2
  LM_correct <- n / (2 * (mean(Ti_vec) - 1)) * (sum_sq_group / sum(e^2) - 1)^2
  pval <- 1 - pchisq(abs(LM_correct), 1)
  list(LM = LM_correct, pval = pval, method = "BP-LM-RE")
}

# ---------------------------------------------------------------------------
# Between Estimator
# ---------------------------------------------------------------------------
between_estimator <- function(y, X, id) {
  uid <- unique(id)
  N <- length(uid)
  k <- ncol(X)
  y_bar <- tapply(y, id, mean, na.rm = TRUE)
  X_bar <- aggregate(X, by = list(id), FUN = mean, na.rm = TRUE)[, -1, drop = FALSE]
  X_bar <- as.matrix(X_bar)
  X1 <- cbind(1, X_bar)
  XtX <- crossprod(X1)
  Xty <- crossprod(X1, y_bar)
  gamma <- as.vector(solve(XtX) %*% Xty)
  resid <- y_bar - X1 %*% gamma
  df <- N - k - 1
  sigma2 <- as.numeric(crossprod(resid) / df)
  vcov <- sigma2 * solve(XtX)
  se <- sqrt(diag(vcov))
  names(gamma) <- c("(Intercept)", colnames(X))
  list(coefficients = gamma, se = se, vcov = vcov, sigma2 = sigma2,
       residuals = as.vector(resid), df = df, method = "Between")
}

# ---------------------------------------------------------------------------
# Panel FGLS (Feasible GLS with AR(1) errors)
# ---------------------------------------------------------------------------
panel_fgls <- function(y, X, id, time, ar1 = TRUE) {
  n <- length(y)
  k <- ncol(X)
  uid <- unique(id)
  N <- length(uid)
  pool <- pooled_ols(y, X)
  e <- pool$residuals
  if (ar1) {
    rho_i <- numeric(N)
    for (i in seq_along(uid)) {
      mask <- id == uid[i]
      ei <- e[mask]
      Ti <- length(ei)
      if (Ti < 3) { rho_i[i] <- 0; next }
      rho_i[i] <- cor(ei[-1], ei[-Ti])
    }
    rho <- mean(rho_i, na.rm = TRUE)
    y_star <- y; X_star <- X
    for (u in uid) {
      mask <- id == u
      idx <- which(mask)
      Ti <- sum(mask)
      y_star[idx[1]] <- y[idx[1]] * sqrt(1 - rho^2)
      X_star[idx[1], ] <- X[idx[1], ] * sqrt(1 - rho^2)
      for (tt in 2:Ti) {
        y_star[idx[tt]] <- y[idx[tt]] - rho * y[idx[tt - 1]]
        X_star[idx[tt], ] <- X[idx[tt], ] - rho * X[idx[tt - 1], ]
      }
    }
  } else {
    y_star <- y; X_star <- X
    rho <- 0
  }
  X1 <- cbind(1, X_star)
  XtX <- crossprod(X1)
  Xty <- crossprod(X1, y_star)
  gamma <- as.vector(solve(XtX) %*% Xty)
  resid <- y_star - X1 %*% gamma
  df <- n - k - 1
  sigma2 <- as.numeric(crossprod(resid) / df)
  vcov <- sigma2 * solve(XtX)
  se <- sqrt(diag(vcov))
  names(gamma) <- c("(Intercept)", colnames(X))
  list(coefficients = gamma, se = se, vcov = vcov, rho = rho,
       sigma2 = sigma2, method = "Panel-FGLS")
}

# ---------------------------------------------------------------------------
# Mundlak (Correlated RE) Estimator
# ---------------------------------------------------------------------------
mundlak_estimator <- function(y, X, id) {
  uid <- unique(id)
  N <- length(uid)
  k <- ncol(X)
  n <- length(y)
  X_bar_i <- matrix(0, n, k)
  for (u in uid) {
    mask <- id == u
    Ti <- sum(mask)
    xbar <- colMeans(X[mask, , drop = FALSE])
    X_bar_i[mask, ] <- matrix(xbar, nrow = Ti, ncol = k, byrow = TRUE)
  }
  X_aug <- cbind(X, X_bar_i)
  colnames(X_aug) <- c(colnames(X), paste0(colnames(X), "_bar"))
  X1 <- cbind(1, X_aug)
  XtX <- crossprod(X1)
  Xty <- crossprod(X1, y)
  gamma <- as.vector(solve(XtX) %*% Xty)
  resid <- y - X1 %*% gamma
  df <- n - ncol(X1)
  sigma2 <- as.numeric(crossprod(resid) / df)
  vcov <- sigma2 * solve(XtX)
  se <- sqrt(diag(vcov))
  names(gamma) <- c("(Intercept)", colnames(X_aug))
  pi_coefs <- gamma[(k + 2):(2 * k + 1)]
  wald_stat <- as.numeric(t(pi_coefs) %*%
    solve(vcov[(k + 2):(2 * k + 1), (k + 2):(2 * k + 1)]) %*% pi_coefs)
  wald_pval <- 1 - pchisq(wald_stat, k)
  list(coefficients = gamma, se = se, vcov = vcov, sigma2 = sigma2,
       pi_coefs = pi_coefs, wald_stat = wald_stat, wald_pval = wald_pval,
       decision = if (wald_pval < 0.05) "FE preferred" else "RE adequate",
       method = "Mundlak-CRE")
}

# ---------------------------------------------------------------------------
# Panel Bootstrap (wild cluster bootstrap)
# ---------------------------------------------------------------------------
wild_cluster_bootstrap <- function(y, X, id, B = 999, seed = 42) {
  set.seed(seed)
  fe <- fe_within(y, X, id)
  beta_hat <- fe$coefficients
  e_hat <- fe$residuals
  uid <- unique(id)
  N <- length(uid)
  k <- length(beta_hat)
  beta_boot <- matrix(0, B, k)
  X_dm <- demean_matrix(X, id)
  y_dm <- demean_panel(y, id)
  for (b in 1:B) {
    weights <- sample(c(-1, 1), N, replace = TRUE)
    e_boot <- numeric(length(y))
    for (i in seq_along(uid)) {
      mask <- id == uid[i]
      e_boot[mask] <- e_hat[mask] * weights[i]
    }
    y_boot <- as.vector(X_dm %*% beta_hat) + e_boot
    XtX <- crossprod(X_dm)
    Xty <- crossprod(X_dm, y_boot)
    beta_boot[b, ] <- as.vector(solve(XtX) %*% Xty)
  }
  se_boot <- apply(beta_boot, 2, sd)
  ci_lower <- apply(beta_boot, 2, quantile, 0.025)
  ci_upper <- apply(beta_boot, 2, quantile, 0.975)
  pval_boot <- numeric(k)
  for (j in 1:k) {
    pval_boot[j] <- mean(abs(beta_boot[, j] - beta_hat[j]) >= abs(beta_hat[j]))
  }
  list(coefficients = beta_hat, se_boot = se_boot, ci_lower = ci_lower,
       ci_upper = ci_upper, pval_boot = pval_boot, B = B,
       method = "Wild-Cluster-Bootstrap")
}

# ---------------------------------------------------------------------------
# Cross-Sectional Dependence: Scaled LM test (Pesaran 2004)
# ---------------------------------------------------------------------------
scaled_lm_test <- function(residuals_matrix) {
  N <- ncol(residuals_matrix)
  TT <- nrow(residuals_matrix)
  rho_sq_sum <- 0
  count <- 0
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      valid <- complete.cases(residuals_matrix[, i], residuals_matrix[, j])
      Tij <- sum(valid)
      if (Tij < 3) next
      rho_ij <- cor(residuals_matrix[valid, i], residuals_matrix[valid, j])
      rho_sq_sum <- rho_sq_sum + (Tij * rho_ij^2 - 1)
      count <- count + 1
    }
  }
  LM_s <- sqrt(1 / (N * (N - 1))) * rho_sq_sum
  pval <- 2 * (1 - pnorm(abs(LM_s)))
  list(LM_scaled = LM_s, pval = pval, N = N, TT = TT, method = "Scaled-LM")
}

# ---------------------------------------------------------------------------
# Comprehensive panel_estimate wrapper
# ---------------------------------------------------------------------------
panel_estimate <- function(formula_str, data, id_col, time_col,
                           method = c("fe", "re", "fd", "pooled", "between",
                                      "mundlak", "fgls"),
                           cluster = NULL, iv_formula = NULL) {
  method <- match.arg(method)
  panel <- make_panel(data, id_col, time_col)
  vars <- all.vars(as.formula(formula_str))
  y_var <- vars[1]
  x_vars <- vars[-1]
  y <- data[[y_var]]
  X <- as.matrix(data[, x_vars, drop = FALSE])
  id <- data[[id_col]]
  time <- data[[time_col]]
  result <- switch(method,
    fe = fe_within(y, X, id, time),
    re = re_gls(y, X, id, time),
    fd = first_diff(y, X, id, time),
    pooled = pooled_ols(y, X),
    between = between_estimator(y, X, id),
    mundlak = mundlak_estimator(y, X, id),
    fgls = panel_fgls(y, X, id, time)
  )
  result$X_used <- X
  result$y <- y
  result$id <- id
  result$time <- time
  result$balanced <- balanced_check(panel)
  if (!is.null(cluster)) {
    if (cluster == "entity") {
      result$cluster_se <- cluster_se(X, result$residuals, id)
    } else if (cluster == "time") {
      result$cluster_se <- cluster_se(X, result$residuals, time)
    } else if (cluster == "twoway") {
      tw <- twoway_cluster_se(X, result$residuals, id, time)
      result$cluster_se <- tw$se
      result$cluster_vcov <- tw$vcov
    }
  }
  class(result) <- "panel_model"
  result
}

print.panel_model <- function(x, ...) {
  cat("Panel Data Model:", x$method, "\n")
  cat("N =", x$N, ", n =", x$n, "\n")
  cat(sprintf("%-15s %10s %10s %10s %10s\n",
              "Variable", "Coef", "SE", "t-stat", "p-value"))
  cat(strrep("-", 55), "\n")
  for (i in seq_along(x$coefficients)) {
    cat(sprintf("%-15s %10.4f %10.4f %10.4f %10.4f\n",
                names(x$coefficients)[i], x$coefficients[i],
                x$se[i], x$tstat[i], x$pval[i]))
  }
  if (!is.null(x$r2_within)) cat("R2 within:", round(x$r2_within, 4), "\n")
  if (!is.null(x$cluster_se)) {
    cat("\nClustered SE:", round(x$cluster_se, 4), "\n")
  }
}

summary.panel_model <- function(object, ...) {
  print.panel_model(object)
  cat("\nSigma^2:", round(object$sigma2, 6), "\n")
  if (!is.null(object$sigma2_u)) {
    cat("Sigma^2_u (entity):", round(object$sigma2_u, 6), "\n")
    cat("Sigma^2_e (idiosyncratic):", round(object$sigma2_e, 6), "\n")
    rho <- object$sigma2_u / (object$sigma2_u + object$sigma2_e)
    cat("Rho (fraction of variance due to u_i):", round(rho, 4), "\n")
  }
}

# ---------------------------------------------------------------------------
# Panel diagnostic suite
# ---------------------------------------------------------------------------
panel_diagnostics <- function(y, X, id, time) {
  cat("=== Panel Data Diagnostics ===\n\n")
  uid <- unique(id)
  N <- length(uid)
  utimes <- sort(unique(time))
  TT <- length(utimes)
  tab <- table(id)
  cat(sprintf("Entities: %d, Time periods: %d, Obs: %d\n", N, TT, length(y)))
  cat(sprintf("Balanced: %s\n", ifelse(all(tab == TT), "Yes", "No")))
  cat(sprintf("Min T_i: %d, Max T_i: %d, Mean T_i: %.1f\n\n",
              min(tab), max(tab), mean(tab)))
  cat("--- F-test for individual effects ---\n")
  ft <- f_test_fe(y, X, id)
  cat(sprintf("F = %.4f, p-value = %.4f\n\n", ft$F_stat, ft$pval))
  cat("--- Breusch-Pagan LM test for RE ---\n")
  bp <- bp_lm_test(y, X, id)
  cat(sprintf("LM = %.4f, p-value = %.4f\n\n", bp$LM, bp$pval))
  cat("--- Hausman test (FE vs RE) ---\n")
  ht <- hausman_test(y, X, id)
  cat(sprintf("chi2 = %.4f, df = %d, p-value = %.4f => %s\n\n",
              ht$chi2, ht$df, ht$pval, ht$decision))
  fe <- fe_within(y, X, id)
  resid_mat <- matrix(NA, TT, N)
  for (i in seq_along(uid)) {
    mask <- id == uid[i]
    ti <- match(time[mask], utimes)
    resid_mat[ti, i] <- fe$residuals[mask]
  }
  cat("--- Pesaran CD test ---\n")
  cd <- pesaran_cd_test(resid_mat)
  cat(sprintf("CD = %.4f, p-value = %.4f\n\n", cd$CD, cd$pval))
  cat("--- LLC unit root test (on y) ---\n")
  llc <- llc_test(y, id, time)
  cat(sprintf("t* = %.4f, z* = %.4f, p-value = %.4f\n\n",
              llc$t_star, llc$z_star, llc$pval))
  invisible(list(f_test = ft, bp_test = bp, hausman = ht, pesaran_cd = cd,
                 llc = llc))
}

# ---------------------------------------------------------------------------
# Simulation: generate panel data for testing
# ---------------------------------------------------------------------------
simulate_panel <- function(N = 50, TT = 20, k = 3, beta = NULL,
                            sigma_u = 1, sigma_e = 2, rho_ar = 0,
                            seed = 123) {
  set.seed(seed)
  n <- N * TT
  if (is.null(beta)) beta <- seq(0.5, by = 0.3, length.out = k)
  id <- rep(1:N, each = TT)
  time <- rep(1:TT, times = N)
  alpha_i <- rep(rnorm(N, 0, sigma_u), each = TT)
  X <- matrix(rnorm(n * k), n, k)
  colnames(X) <- paste0("x", 1:k)
  for (j in 1:k) {
    X[, j] <- X[, j] + 0.5 * alpha_i / sigma_u
  }
  if (rho_ar == 0) {
    eps <- rnorm(n, 0, sigma_e)
  } else {
    eps <- numeric(n)
    for (i in 1:N) {
      idx <- ((i - 1) * TT + 1):(i * TT)
      eps[idx[1]] <- rnorm(1, 0, sigma_e / sqrt(1 - rho_ar^2))
      for (tt in 2:TT) {
        eps[idx[tt]] <- rho_ar * eps[idx[tt - 1]] + rnorm(1, 0, sigma_e)
      }
    }
  }
  y <- alpha_i + as.vector(X %*% beta) + eps
  data <- data.frame(id = id, time = time, y = y, X)
  list(data = data, true_beta = beta, true_sigma_u = sigma_u,
       true_sigma_e = sigma_e, N = N, TT = TT)
}
