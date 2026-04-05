## ml_alpha_research.R
## ML-based factor construction and evaluation
## Pure base R -- no library() calls

# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================

build_return_features <- function(prices, volumes, window_short=5,
                                   window_long=20, window_vol=20) {
  n <- length(prices)
  ret      <- c(NA, diff(log(prices)))
  ma_s     <- as.numeric(stats::filter(prices, rep(1/window_short,window_short),  sides=1))
  ma_l     <- as.numeric(stats::filter(prices, rep(1/window_long, window_long),   sides=1))
  vol_roll <- rep(NA_real_,n)
  for (i in seq(window_vol,n))
    vol_roll[i] <- sd(ret[seq(i-window_vol+1,i)], na.rm=TRUE)*sqrt(252)

  mom_1    <- c(rep(NA,1),  diff(log(prices), lag=1))
  mom_5    <- c(rep(NA,5),  diff(log(prices), lag=5))
  mom_20   <- c(rep(NA,20), diff(log(prices), lag=20))
  vol_ma   <- as.numeric(stats::filter(volumes, rep(1/10,10), sides=1))
  vol_z    <- (volumes - vol_ma) / (sd(volumes,na.rm=TRUE)+1e-8)
  rsi_val  <- .rsi(ret, 14)
  ma_cross <- ma_s / (ma_l + 1e-8) - 1

  X <- cbind(mom_1, mom_5, mom_20, vol_roll, vol_z, rsi_val, ma_cross)
  colnames(X) <- c("mom1","mom5","mom20","vol","vol_z","rsi","ma_cross")
  X
}

.rsi <- function(ret, period=14) {
  n   <- length(ret); rsi <- rep(NA_real_,n)
  for (i in seq(period+1,n)) {
    idx   <- seq(i-period+1, i)
    gains <- pmax(ret[idx],0); losses <- abs(pmin(ret[idx],0))
    ag <- mean(gains); al <- mean(losses)
    rsi[i] <- if (al < 1e-8) 100 else 100 - 100/(1+ag/al)
  }
  rsi
}

build_crypto_features <- function(prices, volumes, funding_rates=NULL,
                                   open_interest=NULL) {
  n <- length(prices)
  base_feat <- build_return_features(prices, volumes)
  extra <- list()
  if (!is.null(funding_rates)) {
    fr_ma  <- as.numeric(stats::filter(funding_rates, rep(1/8,8), sides=1))
    fr_cum <- cumsum(replace(funding_rates,is.na(funding_rates),0))
    extra[["fr_ma"]]  <- fr_ma
    extra[["fr_cum"]] <- fr_cum
  }
  if (!is.null(open_interest)) {
    oi_chg <- c(NA, diff(log(open_interest)))
    extra[["oi_chg"]] <- oi_chg
  }
  if (length(extra)>0) cbind(base_feat, do.call(cbind, extra))
  else base_feat
}

# ============================================================
# 2. DECISION TREE (Regression)
# ============================================================

.build_tree <- function(X, y, depth=0, max_depth=5, min_leaf=5) {
  node <- list(n=length(y), pred=mean(y))
  if (depth>=max_depth || length(y)<=min_leaf) return(node)
  best_j<-NULL; best_thr<-NULL; best_loss<-Inf
  for (j in seq_len(ncol(X))) {
    vals <- sort(unique(X[,j]))
    if (length(vals)<2) next
    thrs <- (vals[-length(vals)]+vals[-1])/2
    for (thr in thrs) {
      li <- X[,j]<=thr; ri <- !li
      if (sum(li)<min_leaf||sum(ri)<min_leaf) next
      loss <- sum((y[li]-mean(y[li]))^2)+sum((y[ri]-mean(y[ri]))^2)
      if (loss<best_loss){best_loss<-loss;best_j<-j;best_thr<-thr}
    }
  }
  if (is.null(best_j)) return(node)
  li <- X[,best_j]<=best_thr; ri <- !li
  node$j<-best_j; node$thr<-best_thr
  node$left  <- .build_tree(X[li,,drop=FALSE], y[li],  depth+1, max_depth, min_leaf)
  node$right <- .build_tree(X[ri,,drop=FALSE], y[ri],  depth+1, max_depth, min_leaf)
  node
}

.pred_tree <- function(node, x) {
  if (is.null(node$j)) return(node$pred)
  if (x[node$j]<=node$thr) .pred_tree(node$left, x)
  else .pred_tree(node$right, x)
}

.pred_tree_mat <- function(tree, X) apply(X,1,function(x) .pred_tree(tree,x))

# ============================================================
# 3. RANDOM FOREST ALPHA
# ============================================================

rf_alpha_model <- function(X, y, n_trees=100, max_depth=5,
                            mtry=NULL, seed=42) {
  set.seed(seed)
  n<-nrow(X); p<-ncol(X)
  if(is.null(mtry)) mtry<-max(1L,floor(sqrt(p)))
  trees <- vector("list",n_trees)
  oob   <- matrix(NA,n,n_trees)
  for (t in seq_len(n_trees)) {
    bi   <- sample(n,n,replace=TRUE)
    oob_ <- setdiff(seq_len(n),unique(bi))
    fi   <- sort(sample(p,mtry))
    tr   <- .build_tree(X[bi,fi,drop=FALSE], y[bi], max_depth=max_depth)
    trees[[t]] <- list(tree=tr, fi=fi)
    if (length(oob_)>0)
      oob[oob_,t] <- .pred_tree_mat(tr, X[oob_,fi,drop=FALSE])
  }
  oob_pred <- rowMeans(oob, na.rm=TRUE)
  list(trees=trees, oob_rmse=sqrt(mean((y-oob_pred)^2,na.rm=TRUE)),
       mtry=mtry, p=p)
}

rf_predict <- function(rf, X) {
  preds <- sapply(rf$trees, function(t)
    .pred_tree_mat(t$tree, X[,t$fi,drop=FALSE]))
  rowMeans(preds)
}

rf_importance <- function(rf, X, y) {
  base <- sqrt(mean((y-rf_predict(rf,X))^2))
  imp  <- sapply(seq_len(rf$p), function(j) {
    Xp <- X; Xp[,j] <- sample(Xp[,j])
    sqrt(mean((y-rf_predict(rf,Xp))^2))-base
  })
  imp/(sum(abs(imp))+1e-12)
}

# ============================================================
# 4. GRADIENT BOOSTING ALPHA
# ============================================================

gbm_alpha <- function(X, y, n_trees=200, max_depth=3,
                       lr=0.05, subsample=0.8, seed=42) {
  set.seed(seed); n <- nrow(X)
  f0 <- mean(y); Fp <- rep(f0,n); trees <- vector("list",n_trees)
  for (t in seq_len(n_trees)) {
    r   <- y-Fp
    idx <- sample(n, floor(n*subsample))
    tr  <- .build_tree(X[idx,,drop=FALSE], r[idx], max_depth=max_depth)
    g   <- .pred_tree_mat(tr, X)
    Fp  <- Fp+lr*g; trees[[t]] <- tr
  }
  list(f0=f0, trees=trees, lr=lr, n_trees=n_trees)
}

gbm_predict <- function(gbm, X, n_use=NULL) {
  if(is.null(n_use)) n_use<-gbm$n_trees
  Fp <- rep(gbm$f0, nrow(X))
  for (t in seq_len(n_use))
    Fp <- Fp + gbm$lr * .pred_tree_mat(gbm$trees[[t]], X)
  Fp
}

# ============================================================
# 5. ALPHA EVALUATION FRAMEWORK
# ============================================================

alpha_ic_timeseries <- function(factor_matrix, fwd_returns, window=60) {
  T_<-nrow(factor_matrix); p<-ncol(factor_matrix)
  ics <- matrix(NA, T_, p)
  for (i in seq(window,T_)) {
    idx <- seq(i-window+1,i)
    for (j in seq_len(p)) {
      fv <- factor_matrix[idx,j]; fr <- fwd_returns[idx]
      ics[i,j] <- cor(rank(fv,na.last="keep"), rank(fr,na.last="keep"),
                      use="pairwise.complete.obs", method="spearman")
    }
  }
  ics
}

alpha_quintile_backtest <- function(factor, fwd_returns, n_groups=5,
                                     fee_bps=5) {
  qs  <- quantile(factor, seq(0,1,1/n_groups), na.rm=TRUE)
  grp <- cut(factor, qs, labels=FALSE, include.lowest=TRUE)
  qr  <- tapply(fwd_returns, grp, mean, na.rm=TRUE)
  ls  <- as.numeric(qr[n_groups])-as.numeric(qr[1])-2*fee_bps/1e4
  list(quintile_returns=qr, long_short=ls, spread=as.numeric(qr[n_groups])-as.numeric(qr[1]))
}

walk_forward_ml <- function(X, y, model_fn, predict_fn,
                             train_pct=0.6, step=20, horizon=5) {
  n     <- nrow(X); init <- floor(n*train_pct)
  starts <- seq(init, n-horizon, by=step)
  res   <- lapply(starts, function(s) {
    tr   <- seq_len(s); val <- seq(s+1, min(s+horizon,n))
    m    <- model_fn(X[tr,,drop=FALSE], y[tr])
    phat <- predict_fn(m, X[val,,drop=FALSE])
    list(preds=phat, actual=y[val], rmse=sqrt(mean((y[val]-phat)^2)),
         ic=cor(phat,y[val],use="complete.obs"))
  })
  list(results=res,
       mean_ic  =mean(sapply(res,function(r) r$ic),   na.rm=TRUE),
       mean_rmse=mean(sapply(res,function(r) r$rmse), na.rm=TRUE))
}

# ============================================================
# 6. FACTOR COMBINATION RESEARCH
# ============================================================

factor_pca <- function(factor_matrix) {
  X_sc <- scale(factor_matrix)
  X_sc[is.nan(X_sc)] <- 0
  Sigma <- cov(X_sc, use="pairwise.complete.obs")
  ev    <- eigen(Sigma)
  list(loadings   = ev$vectors,
       variances  = ev$values,
       pct_var    = ev$values/sum(ev$values),
       cum_var    = cumsum(ev$values)/sum(ev$values),
       scores     = X_sc %*% ev$vectors)
}

sparse_factor_selection <- function(factor_matrix, fwd_returns, k=5) {
  p   <- ncol(factor_matrix)
  ics <- sapply(seq_len(p), function(j) {
    fv <- factor_matrix[,j]
    cor(rank(fv,na.last="keep"), rank(fwd_returns,na.last="keep"),
        use="pairwise.complete.obs", method="spearman")
  })
  top_k <- order(abs(ics), decreasing=TRUE)[1:k]
  list(ic=ics, top_factors=top_k,
       selected_matrix=factor_matrix[,top_k,drop=FALSE])
}

# ============================================================
# ADDITIONAL: FACTOR EVALUATION FRAMEWORK
# ============================================================
factor_information_ratio <- function(factor_values, fwd_returns,
                                      window=60, ann=252) {
  n   <- length(factor_values)
  ics <- rep(NA,n)
  for (i in seq(window,n)) {
    idx  <- seq(i-window+1,i)
    ics[i] <- cor(rank(factor_values[idx],na.last="keep"),
                  rank(fwd_returns[idx],na.last="keep"),
                  use="pairwise.complete.obs", method="spearman")
  }
  list(ic_series=ics, icir=mean(ics,na.rm=TRUE)/(sd(ics,na.rm=TRUE)+1e-8),
       mean_ic=mean(ics,na.rm=TRUE), t_stat=mean(ics,na.rm=TRUE)/
               (sd(ics,na.rm=TRUE)/sqrt(sum(!is.na(ics)))+1e-8))
}

factor_neutralization <- function(factor, controls) {
  X   <- cbind(1, controls)
  b   <- tryCatch(solve(t(X)%*%X+diag(ncol(X))*1e-8)%*%t(X)%*%factor,
                  error=function(e) rep(0,ncol(X)))
  residual <- factor - as.vector(X%*%b)
  list(neutralized=residual, r2=1-var(residual)/(var(factor)+1e-8),
       controls_removed=b[-1])
}

factor_turnover_analysis <- function(factor_values, n_groups=5) {
  n  <- nrow(factor_values); p <- ncol(factor_values)
  to <- numeric(n-1)
  for (t in seq_len(n-1)) {
    grp_t  <- cut(factor_values[t,], breaks=quantile(factor_values[t,],seq(0,1,1/n_groups),na.rm=TRUE),
                  labels=FALSE, include.lowest=TRUE)
    grp_t1 <- cut(factor_values[t+1,], breaks=quantile(factor_values[t+1,],seq(0,1,1/n_groups),na.rm=TRUE),
                  labels=FALSE, include.lowest=TRUE)
    to[t] <- mean(grp_t != grp_t1, na.rm=TRUE)
  }
  list(turnover=to, mean_turnover=mean(to,na.rm=TRUE),
       implied_holding_days=1/mean(to,na.rm=TRUE))
}

alpha_stability_test <- function(factor_values, fwd_returns,
                                  n_splits=5) {
  n      <- length(factor_values)
  split_sz <- floor(n/n_splits)
  ics    <- numeric(n_splits)
  for (s in seq_len(n_splits)) {
    idx  <- seq((s-1)*split_sz+1, min(s*split_sz, n))
    ics[s] <- cor(rank(factor_values[idx],na.last="keep"),
                  rank(fwd_returns[idx],na.last="keep"),
                  use="pairwise.complete.obs", method="spearman")
  }
  list(ic_by_split=ics, mean_ic=mean(ics,na.rm=TRUE),
       stability=1-sd(ics,na.rm=TRUE)/(abs(mean(ics,na.rm=TRUE))+1e-8),
       consistent_sign=all(sign(ics)==sign(ics[1]),na.rm=TRUE))
}

ensemble_alpha_model <- function(factor_list, fwd_returns, method="equal") {
  n_fac <- length(factor_list)
  ic_weights <- sapply(factor_list, function(f)
    cor(rank(f,na.last="keep"), rank(fwd_returns,na.last="keep"),
        use="pairwise.complete.obs", method="spearman"))
  w <- if(method=="ic") ic_weights/(sum(abs(ic_weights))+1e-12) else rep(1/n_fac,n_fac)
  mat <- do.call(cbind, factor_list)
  combined <- as.vector(mat %*% w)
  list(combined=combined, weights=w, ic_weights=ic_weights,
       ensemble_ic=cor(rank(combined,na.last="keep"),
                       rank(fwd_returns,na.last="keep"),
                       use="pairwise.complete.obs", method="spearman"))
}


# ============================================================
# ADDITIONAL ML ALPHA RESEARCH
# ============================================================

walk_forward_optimization <- function(features, targets, model_fn,
                                       train_window = 252, test_window = 21,
                                       step = 21) {
  n     <- nrow(features)
  starts <- seq(train_window + 1, n - test_window, by = step)
  results <- lapply(starts, function(s) {
    train_idx <- seq(s - train_window, s - 1)
    test_idx  <- seq(s, min(s + test_window - 1, n))
    model  <- model_fn(features[train_idx, , drop=FALSE], targets[train_idx])
    preds  <- predict(model, features[test_idx, , drop=FALSE])
    list(test_idx = test_idx, predictions = preds,
         actuals = targets[test_idx],
         ic = cor(rank(preds), rank(targets[test_idx]),
                  use="pairwise.complete.obs", method="spearman"))
  })
  ics   <- sapply(results, function(r) r$ic)
  preds_all <- unlist(lapply(results, function(r) r$predictions))
  test_idx_all <- unlist(lapply(results, function(r) r$test_idx))
  list(ics = ics, mean_ic = mean(ics, na.rm=TRUE),
       ic_ir = mean(ics, na.rm=TRUE) / (sd(ics, na.rm=TRUE) + 1e-8),
       all_predictions = preds_all[order(test_idx_all)])
}

feature_importance_stability <- function(features, targets,
                                          n_bootstrap = 50, sample_pct = 0.8) {
  n_feat <- ncol(features); n_obs <- nrow(features)
  imp_mat <- matrix(NA, n_bootstrap, n_feat)
  for (b in seq_len(n_bootstrap)) {
    idx <- sample(n_obs, floor(n_obs * sample_pct))
    X   <- features[idx, , drop=FALSE]
    y   <- targets[idx]
    cors <- apply(X, 2, function(x) {
      tryCatch(cor(x, y, use="pairwise.complete.obs"), error=function(e) NA)
    })
    imp_mat[b, ] <- abs(cors)
  }
  list(mean_importance = colMeans(imp_mat, na.rm=TRUE),
       sd_importance = apply(imp_mat, 2, sd, na.rm=TRUE),
       cv_importance = apply(imp_mat, 2, sd, na.rm=TRUE) /
                       (colMeans(imp_mat, na.rm=TRUE) + 1e-8),
       stable_features = which(apply(imp_mat, 2, sd, na.rm=TRUE) /
                                (colMeans(imp_mat, na.rm=TRUE) + 1e-8) < 0.5))
}

alpha_decay_analysis <- function(factor_values, returns_mat_lags) {
  # returns_mat_lags: columns are lag 1, 2, ... n returns
  n_lags <- ncol(returns_mat_lags)
  ics    <- numeric(n_lags)
  for (lag in seq_len(n_lags)) {
    valid <- !is.na(factor_values) & !is.na(returns_mat_lags[, lag])
    if (sum(valid) < 10) { ics[lag] <- NA; next }
    ics[lag] <- cor(rank(factor_values[valid]),
                    rank(returns_mat_lags[valid, lag]),
                    use="pairwise.complete.obs", method="spearman")
  }
  half_life <- tryCatch({
    valid_idx <- which(!is.na(ics))
    approx(x = ics[valid_idx], y = valid_idx,
           xout = ics[valid_idx[1]] / 2)$y
  }, error = function(e) NA)
  list(ics = ics, lags = seq_len(n_lags), half_life = half_life,
       persistence = sum(!is.na(ics) & ics > 0) / sum(!is.na(ics)))
}

multi_factor_combination_study <- function(factors_mat, returns,
                                            methods = c("equal", "ic_weighted", "pca")) {
  results <- list()
  n_f <- ncol(factors_mat)
  ics <- apply(factors_mat, 2, function(f) {
    cor(rank(f), rank(returns), use="pairwise.complete.obs", method="spearman")
  })
  for (meth in methods) {
    w <- switch(meth,
      "equal"       = rep(1/n_f, n_f),
      "ic_weighted" = { w_raw <- pmax(ics, 0); w_raw / (sum(w_raw) + 1e-12) },
      "pca"         = {
        pca <- prcomp(factors_mat, scale.=TRUE)
        abs(pca$rotation[, 1]) / sum(abs(pca$rotation[, 1]))
      }
    )
    combined <- as.vector(factors_mat %*% w)
    ic_combined <- cor(rank(combined), rank(returns),
                       use="pairwise.complete.obs", method="spearman")
    results[[meth]] <- list(weights = w, combined = combined,
                             ic = ic_combined)
  }
  results
}


# ─── ADDITIONAL ML ALPHA RESEARCH ─────────────────────────────────────────────

information_coefficient_analysis <- function(factors_mat, returns,
                                              rolling_window = 63) {
  n_obs  <- nrow(factors_mat)
  n_fct  <- ncol(factors_mat)
  roll_ic <- matrix(NA, n_obs, n_fct)
  for (i in seq(rolling_window, n_obs)) {
    idx <- seq(i - rolling_window + 1, i)
    for (j in seq_len(n_fct)) {
      roll_ic[i, j] <- cor(rank(factors_mat[idx, j], na.last="keep"),
                            rank(returns[idx], na.last="keep"),
                            use="pairwise.complete.obs", method="spearman")
    }
  }
  ic_mean <- colMeans(roll_ic, na.rm=TRUE)
  ic_sd   <- apply(roll_ic, 2, sd, na.rm=TRUE)
  icir    <- ic_mean / (ic_sd + 1e-8)
  list(rolling_ic = roll_ic, ic_mean = ic_mean, ic_sd = ic_sd, icir = icir,
       top_factors = order(-abs(ic_mean))[1:min(5, n_fct)])
}

regime_conditional_alpha <- function(alpha_series, regime_indicator) {
  regimes <- sort(unique(regime_indicator[!is.na(regime_indicator)]))
  results <- lapply(regimes, function(r) {
    idx  <- which(regime_indicator == r)
    a    <- alpha_series[idx]
    list(regime = r, mean_alpha = mean(a, na.rm=TRUE),
         sd_alpha = sd(a, na.rm=TRUE),
         sharpe = mean(a, na.rm=TRUE) / (sd(a, na.rm=TRUE) + 1e-8) * sqrt(252),
         pct_positive = mean(a > 0, na.rm=TRUE),
         n_obs = length(a))
  })
  names(results) <- paste0("regime_", regimes)
  results
}

portfolio_optimization_study <- function(expected_returns, cov_mat,
                                          risk_aversion_levels = c(1, 2, 5, 10)) {
  n <- length(expected_returns)
  results <- lapply(risk_aversion_levels, function(lambda) {
    # Mean-variance optimal (unconstrained closed form)
    inv_cov  <- tryCatch(solve(cov_mat), error=function(e) diag(n))
    raw_w    <- as.vector(inv_cov %*% expected_returns) / lambda
    # Long-only constraint: simple normalization of positive weights
    lo_w     <- pmax(raw_w, 0)
    lo_w     <- lo_w / (sum(lo_w) + 1e-12)
    port_ret <- sum(lo_w * expected_returns)
    port_var <- as.numeric(t(lo_w) %*% cov_mat %*% lo_w)
    list(lambda=lambda, weights=lo_w, expected_return=port_ret,
         volatility=sqrt(port_var * 252),
         sharpe=port_ret / (sqrt(port_var) + 1e-8) * sqrt(252))
  })
  names(results) <- paste0("lambda_", risk_aversion_levels)
  efficient_lambda <- risk_aversion_levels[which.max(sapply(results, function(r) r$sharpe))]
  list(results=results, efficient_lambda=efficient_lambda)
}

# ─── UTILITY / HELPER FUNCTIONS ───────────────────────────────────────────────

factor_zscore_crosssectional <- function(factor_mat) {
  # Cross-sectional z-score each row
  t(apply(factor_mat, 1, function(r) {
    (r - mean(r, na.rm=TRUE)) / (sd(r, na.rm=TRUE) + 1e-8)
  }))
}

neutralize_factor <- function(factor_values, market_cap) {
  log_mc <- log(market_cap + 1)
  fit    <- lm(factor_values ~ log_mc)
  residuals(fit)
}

long_short_quintile_backtest <- function(signal, returns, n_quantiles = 5,
                                          cost_bps = 5) {
  valid <- !is.na(signal) & !is.na(returns)
  qs    <- quantile(signal[valid], seq(0, 1, 1/n_quantiles), na.rm=TRUE)
  grp   <- cut(signal, qs, labels=FALSE, include.lowest=TRUE)
  q_ret <- tapply(returns, grp, mean, na.rm=TRUE)
  ls_ret <- q_ret[n_quantiles] - q_ret[1] - 2 * cost_bps/1e4
  list(quintile_returns = q_ret, long_short_return = ls_ret,
       spread = q_ret[n_quantiles] - q_ret[1],
       monotonic = all(diff(q_ret) > 0, na.rm=TRUE) |
                   all(diff(q_ret) < 0, na.rm=TRUE))
}

# research module metadata
research_version <- function() "1.0.0"
research_info    <- function() list(version="1.0.0", base_r_only=TRUE)
# utility: safe correlation
safe_cor <- function(x, y, method="pearson") {
  tryCatch(cor(x, y, use="pairwise.complete.obs", method=method), error=function(e) NA)
}
# utility: rolling mean
roll_mean <- function(x, w) as.numeric(stats::filter(x, rep(1/w, w), sides=1))
# utility: annualize return
annualize_ret <- function(r, periods_per_year=252) mean(r, na.rm=TRUE) * periods_per_year
# utility: annualize vol
annualize_vol <- function(r, periods_per_year=252) sd(r, na.rm=TRUE) * sqrt(periods_per_year)
# end of file

# ml alpha research module loaded
.ml_alpha_loaded <- TRUE
# placeholder
ml_util <- function() invisible(NULL)
