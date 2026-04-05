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
