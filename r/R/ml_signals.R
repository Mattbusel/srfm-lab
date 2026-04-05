# =============================================================================
# ml_signals.R
# Machine learning signal research in R -- all from scratch, base R only
# =============================================================================
# Financial intuition: ML models can capture non-linear relationships between
# features (signals) and returns that linear models miss. However, overfitting
# is a constant danger -- time-series cross-validation and walk-forward testing
# are essential. We implement models from scratch to ensure full understanding
# and control of the prediction logic.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. DECISION TREE (recursive partitioning, entropy criterion)
# -----------------------------------------------------------------------------

#' Entropy for a binary classification problem
entropy <- function(y) {
  n <- length(y)
  if (n == 0) return(0)
  p <- mean(y > 0)  # fraction of positive returns
  if (p == 0 || p == 1) return(0)
  -(p * log2(p) + (1-p) * log2(1-p))
}

#' Information gain from splitting on feature j at threshold t
info_gain <- function(y, x, t) {
  n <- length(y)
  left  <- y[x <= t]
  right <- y[x > t]
  nl <- length(left); nr <- length(right)
  if (nl == 0 || nr == 0) return(0)
  entropy(y) - (nl/n)*entropy(left) - (nr/n)*entropy(right)
}

#' Variance reduction criterion for regression trees
variance_reduction <- function(y, x, t) {
  n <- length(y)
  left  <- y[x <= t]
  right <- y[x > t]
  nl <- length(left); nr <- length(right)
  if (nl == 0 || nr == 0) return(0)
  var(y) - (nl/n)*var(left) - (nr/n)*var(right)
}

#' Find best split for a feature
best_split_feature <- function(y, x, n_thresholds=20, criterion="variance") {
  x_sorted <- sort(unique(x))
  if (length(x_sorted) < 2) return(list(gain=0, threshold=NA))
  # Sample thresholds
  idx <- round(seq(1, length(x_sorted)-1, length.out=min(n_thresholds, length(x_sorted)-1)))
  thresholds <- (x_sorted[idx] + x_sorted[idx+1]) / 2

  gains <- sapply(thresholds, function(t) {
    if (criterion == "entropy") info_gain(y, x, t)
    else variance_reduction(y, x, t)
  })
  best_idx <- which.max(gains)
  list(gain=gains[best_idx], threshold=thresholds[best_idx])
}

#' Recursive tree building
build_tree <- function(X, y, max_depth=5, min_samples=10,
                        depth=0, criterion="variance") {
  n <- nrow(X)
  p <- ncol(X)

  # Leaf conditions
  if (depth >= max_depth || n < min_samples) {
    return(list(leaf=TRUE, value=mean(y), n=n))
  }

  best_gain   <- 0
  best_feat   <- 1
  best_thresh <- NA

  for (j in seq_len(p)) {
    res <- best_split_feature(y, X[,j], criterion=criterion)
    if (res$gain > best_gain && !is.na(res$threshold)) {
      best_gain   <- res$gain
      best_feat   <- j
      best_thresh <- res$threshold
    }
  }

  if (best_gain == 0 || is.na(best_thresh)) {
    return(list(leaf=TRUE, value=mean(y), n=n))
  }

  left_idx  <- X[, best_feat] <= best_thresh
  right_idx <- !left_idx

  if (sum(left_idx) < min_samples || sum(right_idx) < min_samples) {
    return(list(leaf=TRUE, value=mean(y), n=n))
  }

  left_tree  <- build_tree(X[left_idx,, drop=FALSE], y[left_idx],
                            max_depth, min_samples, depth+1, criterion)
  right_tree <- build_tree(X[right_idx,, drop=FALSE], y[right_idx],
                            max_depth, min_samples, depth+1, criterion)

  list(leaf=FALSE, feature=best_feat, threshold=best_thresh,
       gain=best_gain, left=left_tree, right=right_tree, n=n)
}

#' Predict with a tree
predict_tree <- function(tree, X) {
  n <- nrow(X)
  pred <- numeric(n)
  for (i in seq_len(n)) {
    node <- tree
    while (!node$leaf) {
      if (X[i, node$feature] <= node$threshold) node <- node$left
      else node <- node$right
    }
    pred[i] <- node$value
  }
  pred
}

# -----------------------------------------------------------------------------
# 2. RANDOM FOREST ENSEMBLE
# -----------------------------------------------------------------------------

#' Random forest: bootstrap + feature subsampling
#' @param X feature matrix (n x p)
#' @param y response vector
#' @param n_trees number of trees
#' @param mtry number of features per split (default sqrt(p))
#' @param max_depth tree depth limit
#' @param min_samples minimum samples per leaf
random_forest <- function(X, y, n_trees=100, mtry=NULL, max_depth=5,
                            min_samples=10, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- nrow(X); p <- ncol(X)
  if (is.null(mtry)) mtry <- max(1, round(sqrt(p)))
  feature_names <- colnames(X)

  trees <- vector("list", n_trees)
  oob_predictions <- matrix(NA, n, n_trees)
  feature_importance <- numeric(p)

  for (b in seq_len(n_trees)) {
    # Bootstrap sample
    boot_idx <- sample(seq_len(n), n, replace=TRUE)
    oob_idx  <- setdiff(seq_len(n), unique(boot_idx))
    X_boot   <- X[boot_idx, , drop=FALSE]
    y_boot   <- y[boot_idx]

    # Random feature subsample for each tree (wrap standard tree to use mtry)
    feat_subset <- sample(seq_len(p), mtry)
    X_sub <- X_boot[, feat_subset, drop=FALSE]
    tree  <- build_tree(X_sub, y_boot, max_depth=max_depth,
                         min_samples=min_samples)
    trees[[b]] <- list(tree=tree, feat_subset=feat_subset)

    # OOB predictions
    if (length(oob_idx) > 0) {
      pred_oob <- predict_tree(tree, X[oob_idx, feat_subset, drop=FALSE])
      oob_predictions[oob_idx, b] <- pred_oob
    }

    # Feature importance: total variance reduction
    accumulate_importance <- function(node, fi) {
      if (node$leaf) return(fi)
      orig_feat <- feat_subset[node$feature]
      fi[orig_feat] <- fi[orig_feat] + node$gain * node$n / n
      fi <- accumulate_importance(node$left, fi)
      fi <- accumulate_importance(node$right, fi)
      fi
    }
    feature_importance <- accumulate_importance(tree, feature_importance)
  }

  # OOB error
  oob_pred_mean <- rowMeans(oob_predictions, na.rm=TRUE)
  oob_mask <- !is.na(oob_pred_mean)
  oob_r2 <- if (sum(oob_mask)>2) 1 - sum((y[oob_mask]-oob_pred_mean[oob_mask])^2)/
                                       sum((y[oob_mask]-mean(y[oob_mask]))^2) else NA

  # Normalize feature importance
  if (sum(feature_importance) > 0) feature_importance <- feature_importance / sum(feature_importance)
  names(feature_importance) <- feature_names

  cat(sprintf("=== Random Forest (n_trees=%d, mtry=%d) ===\n", n_trees, mtry))
  cat(sprintf("OOB R-squared: %.4f\n", oob_r2))
  cat("Feature Importance (top 5):\n")
  print(sort(feature_importance, decreasing=TRUE)[1:min(5, p)])

  invisible(list(trees=trees, oob_r2=oob_r2,
                 feature_importance=feature_importance,
                 oob_pred=oob_pred_mean, n_trees=n_trees, mtry=mtry))
}

#' Predict with random forest
predict_rf <- function(rf, X) {
  preds <- sapply(rf$trees, function(t_obj) {
    predict_tree(t_obj$tree, X[, t_obj$feat_subset, drop=FALSE])
  })
  rowMeans(preds)
}

# -----------------------------------------------------------------------------
# 3. GRADIENT BOOSTING (AdaBoost variant for regression)
# -----------------------------------------------------------------------------

#' Gradient Boosting: fits additive model F(x) = sum_m gamma_m * h_m(x)
#' where each h_m is a shallow tree fitted to the current residuals
#' @param X feature matrix
#' @param y response vector
#' @param n_iter number of boosting iterations
#' @param lr learning rate (shrinkage)
#' @param max_depth depth per tree
#' @param subsample fraction of data used per tree (stochastic GB)
gradient_boosting <- function(X, y, n_iter=100, lr=0.1,
                               max_depth=3, min_samples=10, subsample=0.8) {
  n <- nrow(X); p <- ncol(X)
  F_current <- rep(mean(y), n)  # initialize at mean
  trees  <- vector("list", n_iter)
  lrs    <- numeric(n_iter)

  for (m in seq_len(n_iter)) {
    # Compute negative gradient (residuals for MSE loss)
    resid <- y - F_current

    # Subsample
    idx_sub <- sample(seq_len(n), round(n*subsample), replace=FALSE)
    X_sub   <- X[idx_sub, , drop=FALSE]
    r_sub   <- resid[idx_sub]

    # Fit tree to residuals
    tree <- build_tree(X_sub, r_sub, max_depth=max_depth,
                        min_samples=min_samples)
    trees[[m]] <- tree

    # Line search: optimal learning rate (simplified: use fixed lr)
    h_m <- predict_tree(tree, X)
    F_current <- F_current + lr * h_m
    lrs[m] <- lr
  }

  # Train MSE
  train_mse <- mean((y - F_current)^2)
  train_r2  <- 1 - sum((y-F_current)^2)/sum((y-mean(y))^2)

  cat(sprintf("=== Gradient Boosting (n_iter=%d, lr=%.3f) ===\n", n_iter, lr))
  cat(sprintf("Train R-squared: %.4f\n", train_r2))

  invisible(list(trees=trees, lrs=lrs, lr=lr, F_init=mean(y),
                 train_r2=train_r2, train_pred=F_current))
}

#' Predict with gradient boosting model
predict_gb <- function(gb, X) {
  F_pred <- rep(gb$F_init, nrow(X))
  for (m in seq_along(gb$trees)) {
    if (is.null(gb$trees[[m]])) next
    F_pred <- F_pred + gb$lr * predict_tree(gb$trees[[m]], X)
  }
  F_pred
}

# -----------------------------------------------------------------------------
# 4. TIME-SERIES CROSS-VALIDATION
# -----------------------------------------------------------------------------

#' Time-series split cross-validation
#' Respects temporal ordering: train on past, validate on future
#' @param n total number of observations
#' @param n_splits number of splits
#' @param initial minimum training size
ts_cv_splits <- function(n, n_splits=5, initial=NULL) {
  if (is.null(initial)) initial <- round(n * 0.5)
  step <- round((n - initial) / n_splits)
  splits <- vector("list", n_splits)
  for (k in seq_len(n_splits)) {
    train_end <- initial + (k-1)*step
    val_start <- train_end + 1
    val_end   <- min(train_end + step, n)
    if (val_start > n) break
    splits[[k]] <- list(
      train = seq_len(train_end),
      val   = val_start:val_end
    )
  }
  Filter(Negate(is.null), splits)
}

#' Purged k-fold cross-validation for time series
#' Adds embargo (gap) between training and validation to prevent leakage
#' @param n total observations
#' @param k number of folds
#' @param embargo gap between train end and val start (in periods)
purged_kfold <- function(n, k=5, embargo=10) {
  fold_size <- floor(n / k)
  splits <- vector("list", k)
  for (i in seq_len(k)) {
    val_start <- (i-1)*fold_size + 1
    val_end   <- min(i*fold_size, n)
    train_idx <- c(if (val_start > 1 + embargo) seq_len(max(0, val_start-embargo-1)) else integer(0),
                   if (val_end + embargo < n) (val_end+embargo):n else integer(0))
    splits[[i]] <- list(train=train_idx, val=val_start:val_end)
  }
  splits
}

# -----------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (PERMUTATION-BASED)
# -----------------------------------------------------------------------------

#' Permutation-based feature importance
#' Measures how much model performance degrades when a feature is shuffled
#' More reliable than impurity-based importance for trees
#' @param model_fn function(X_train, y_train, X_test) -> predictions
#' @param X feature matrix
#' @param y response vector
#' @param n_repeats number of permutations per feature
permutation_importance <- function(model_fn, X, y, n_repeats=10, seed=42) {
  set.seed(seed)
  n <- nrow(X); p <- ncol(X)
  feat_names <- colnames(X)
  if (is.null(feat_names)) feat_names <- paste0("X", seq_len(p))

  # Baseline score (e.g., R2 on full features)
  splits <- ts_cv_splits(n, n_splits=3, initial=round(n*0.6))
  base_scores <- sapply(splits, function(s) {
    pred <- model_fn(X[s$train,, drop=FALSE], y[s$train], X[s$val,, drop=FALSE])
    r <- cor(pred, y[s$val]); r^2
  })
  baseline <- mean(base_scores, na.rm=TRUE)

  # Permutation scores per feature
  importances <- numeric(p)
  for (j in seq_len(p)) {
    perm_scores <- numeric(n_repeats)
    for (r in seq_len(n_repeats)) {
      X_perm <- X
      X_perm[, j] <- sample(X[, j])  # permute feature j
      r_scores <- sapply(splits, function(s) {
        pred <- model_fn(X_perm[s$train,, drop=FALSE], y[s$train],
                          X_perm[s$val,, drop=FALSE])
        r <- cor(pred, y[s$val]); r^2
      })
      perm_scores[r] <- mean(r_scores, na.rm=TRUE)
    }
    importances[j] <- baseline - mean(perm_scores)
  }

  names(importances) <- feat_names
  importances_norm <- pmax(importances, 0) / (sum(pmax(importances, 0)) + 1e-10)

  cat("=== Permutation Feature Importance ===\n")
  cat(sprintf("Baseline R2: %.4f\n", baseline))
  print(sort(importances_norm, decreasing=TRUE))

  invisible(list(importances=importances, importances_norm=importances_norm,
                 baseline=baseline))
}

# -----------------------------------------------------------------------------
# 6. LASSO AND RIDGE REGRESSION
# -----------------------------------------------------------------------------

#' Ridge regression via closed-form solution
#' beta = (X'X + lambda*I)^{-1} X'y
ridge_regression <- function(X, y, lambda=1.0, standardize=TRUE) {
  n <- nrow(X); p <- ncol(X)
  if (standardize) {
    X_mean <- colMeans(X); X_sd <- apply(X, 2, sd)
    X_std  <- scale(X, center=X_mean, scale=X_sd)
  } else {
    X_std <- X; X_mean <- rep(0,p); X_sd <- rep(1,p)
  }
  X_aug <- cbind(1, X_std)
  lambda_mat <- diag(c(0, rep(lambda, p)))  # don't penalize intercept
  beta <- solve(t(X_aug) %*% X_aug + lambda_mat) %*% t(X_aug) %*% y
  list(intercept=beta[1], coef=beta[-1]/X_sd, X_mean=X_mean, X_sd=X_sd,
       lambda=lambda)
}

#' LASSO via coordinate descent
#' Soft-thresholding operator: S(z, lambda) = sign(z)*max(|z|-lambda, 0)
soft_threshold <- function(z, lambda) sign(z) * pmax(abs(z) - lambda, 0)

lasso_coord_descent <- function(X, y, lambda=0.01, max_iter=1000, tol=1e-6,
                                  standardize=TRUE) {
  n <- nrow(X); p <- ncol(X)
  if (standardize) {
    X_mean <- colMeans(X); X_sd <- pmax(apply(X, 2, sd), 1e-10)
    X_std  <- scale(X, center=X_mean, scale=X_sd)
  } else {
    X_std <- X; X_mean <- rep(0,p); X_sd <- rep(1,p)
  }
  y_mean <- mean(y); y_c <- y - y_mean

  beta <- rep(0, p)
  for (iter in seq_len(max_iter)) {
    beta_old <- beta
    for (j in seq_len(p)) {
      # Partial residual
      r_j <- y_c - X_std[, -j, drop=FALSE] %*% beta[-j]
      # OLS estimate for beta_j (rho_j = X_j' * r_j / n)
      rho_j  <- sum(X_std[, j] * r_j) / n
      beta[j] <- soft_threshold(rho_j, lambda)
    }
    if (max(abs(beta - beta_old)) < tol) break
  }

  # Intercept
  intercept <- y_mean - sum(X_mean * beta / X_sd)
  coef_orig <- beta / X_sd

  n_nonzero <- sum(beta != 0)
  cat(sprintf("LASSO (lambda=%.4f): %d/%d non-zero coefficients, %d iterations\n",
              lambda, n_nonzero, p, iter))

  list(intercept=intercept, coef=coef_orig, coef_std=beta,
       n_nonzero=n_nonzero, lambda=lambda, iter=iter)
}

#' LASSO path: fit for multiple lambda values
lasso_path <- function(X, y, n_lambda=20, lambda_min_ratio=0.01) {
  # Determine lambda_max: smallest lambda that zeros all coefficients
  X_std <- scale(X)
  y_c   <- y - mean(y)
  lambda_max <- max(abs(t(X_std) %*% y_c)) / nrow(X)
  lambda_min <- lambda_max * lambda_min_ratio
  lambdas    <- exp(seq(log(lambda_max), log(lambda_min), length.out=n_lambda))

  coef_path <- matrix(0, n_lambda, ncol(X))
  for (k in seq_along(lambdas)) {
    fit_k <- lasso_coord_descent(X, y, lambda=lambdas[k], standardize=TRUE)
    coef_path[k, ] <- fit_k$coef
  }

  cat("=== LASSO Solution Path ===\n")
  cat(sprintf("Lambda range: [%.4f, %.4f]\n", lambda_min, lambda_max))
  cat(sprintf("Active variables at min lambda: %d\n", sum(coef_path[n_lambda,]!=0)))

  invisible(list(lambdas=lambdas, coef_path=coef_path))
}

# -----------------------------------------------------------------------------
# 7. SUPPORT VECTOR MACHINE (linear kernel, dual problem via gradient)
# -----------------------------------------------------------------------------

#' Linear SVM via sub-gradient descent on hinge loss
#' Minimize: 0.5*||w||^2 + C * sum_i max(0, 1 - y_i*(w'x_i + b))
#' @param X feature matrix
#' @param y binary labels (must be +1 or -1)
#' @param C regularization parameter
svm_linear <- function(X, y, C=1.0, max_iter=1000, lr=0.001) {
  n <- nrow(X); p <- ncol(X)
  # Ensure labels are +1/-1
  y_svm <- ifelse(y > 0, 1, -1)
  X_std <- scale(X)
  X_mean <- attr(X_std, "scaled:center")
  X_sd   <- attr(X_std, "scaled:scale")

  w <- rep(0, p); b <- 0

  for (iter in seq_len(max_iter)) {
    # Stochastic gradient update on random sample
    i <- sample(seq_len(n), 1)
    margin <- y_svm[i] * (sum(w * X_std[i,]) + b)

    if (margin >= 1) {
      # No margin violation
      w <- w - lr * w
    } else {
      # Margin violation: hinge loss gradient
      w <- w - lr * (w - C * y_svm[i] * X_std[i,])
      b <- b + lr * C * y_svm[i]
    }
  }

  # Scale weights back to original feature space
  w_orig <- w / (X_sd + 1e-10)
  b_orig <- b - sum(w_orig * X_mean)

  # Predictions on training set
  scores <- X_std %*% w + b
  pred_label <- ifelse(scores > 0, 1, -1)
  accuracy   <- mean(pred_label == y_svm)

  cat(sprintf("=== Linear SVM (C=%.2f) ===\n", C))
  cat(sprintf("Training accuracy: %.3f\n", accuracy))

  invisible(list(w=w_orig, b=b_orig, w_std=w, b_std=b,
                 X_mean=X_mean, X_sd=X_sd, accuracy=accuracy))
}

#' Predict with linear SVM
predict_svm <- function(svm_model, X) {
  X_std <- scale(X, center=svm_model$X_mean, scale=svm_model$X_sd)
  scores <- as.numeric(X_std %*% svm_model$w_std + svm_model$b_std)
  list(score=scores, label=ifelse(scores>0,1,-1))
}

# -----------------------------------------------------------------------------
# 8. MODEL EVALUATION: IC, HIT RATE, SHARPE BY MODEL
# -----------------------------------------------------------------------------

#' Evaluate a prediction model using financial metrics
#' @param predictions model predictions (continuous, proportional to expected return)
#' @param realized_returns actual realized returns
#' @param model_name label for output
evaluate_model <- function(predictions, realized_returns, model_name="Model") {
  valid <- !is.na(predictions) & !is.na(realized_returns)
  pred  <- predictions[valid]
  real  <- realized_returns[valid]
  n     <- length(pred)

  # IC (rank correlation)
  ic <- cor(rank(pred), rank(real))

  # Hit rate: correct sign prediction
  hit_rate <- mean(sign(pred) == sign(real))

  # Sharpe ratio of a simple signal strategy
  # Position: sign(pred), rebalance each period
  strategy_ret <- sign(pred) * real
  sharpe_ann <- mean(strategy_ret) / (sd(strategy_ret)+1e-10) * sqrt(252)

  # Profit factor
  wins   <- strategy_ret[strategy_ret > 0]
  losses <- -strategy_ret[strategy_ret < 0]
  pf <- if (sum(losses) > 0) sum(wins)/sum(losses) else Inf

  # R-squared (predictive)
  r2 <- tryCatch(cor(pred, real)^2, error=function(e) NA)

  cat(sprintf("=== %s Evaluation ===\n", model_name))
  cat(sprintf("IC (Spearman):    %.4f\n", ic))
  cat(sprintf("Hit Rate:         %.3f\n", hit_rate))
  cat(sprintf("Sharpe (ann):     %.3f\n", sharpe_ann))
  cat(sprintf("Profit Factor:    %.3f\n", pf))
  cat(sprintf("R-squared:        %.4f\n", r2))

  list(IC=ic, hit_rate=hit_rate, sharpe=sharpe_ann, pf=pf, r2=r2, n=n)
}

# -----------------------------------------------------------------------------
# 9. FULL ML PIPELINE
# -----------------------------------------------------------------------------

#' Complete ML signal research pipeline
#' @param X feature matrix (T x p)
#' @param y target returns (T)
#' @param test_frac fraction for out-of-sample test
run_ml_pipeline <- function(X, y, test_frac = 0.20, seed=42) {
  set.seed(seed)
  n <- nrow(X); p <- ncol(X)
  n_test  <- round(n * test_frac)
  n_train <- n - n_test

  cat("=============================================================\n")
  cat("ML SIGNAL RESEARCH PIPELINE\n")
  cat(sprintf("Total obs: %d, Features: %d\n", n, p))
  cat(sprintf("Train: %d, Test: %d\n\n", n_train, n_test))

  X_train <- X[1:n_train, , drop=FALSE]
  y_train <- y[1:n_train]
  X_test  <- X[(n_train+1):n, , drop=FALSE]
  y_test  <- y[(n_train+1):n]

  results <- list()

  # 1. Decision Tree
  cat("--- Decision Tree ---\n")
  tree <- build_tree(X_train, y_train, max_depth=5, min_samples=20)
  pred_tree <- predict_tree(tree, X_test)
  results$tree <- evaluate_model(pred_tree, y_test, "Decision Tree")

  # 2. Random Forest
  cat("\n--- Random Forest ---\n")
  rf <- random_forest(X_train, y_train, n_trees=50, max_depth=4, seed=seed)
  pred_rf <- predict_rf(rf, X_test)
  results$rf <- evaluate_model(pred_rf, y_test, "Random Forest")

  # 3. Gradient Boosting
  cat("\n--- Gradient Boosting ---\n")
  gb <- gradient_boosting(X_train, y_train, n_iter=50, lr=0.05, max_depth=3)
  pred_gb <- predict_gb(gb, X_test)
  results$gb <- evaluate_model(pred_gb, y_test, "Gradient Boosting")

  # 4. Ridge Regression
  cat("\n--- Ridge Regression ---\n")
  ridge <- ridge_regression(X_train, y_train, lambda=0.1)
  pred_ridge <- X_test %*% ridge$coef + ridge$intercept
  results$ridge <- evaluate_model(as.numeric(pred_ridge), y_test, "Ridge")

  # 5. LASSO
  cat("\n--- LASSO ---\n")
  lasso <- lasso_coord_descent(X_train, y_train, lambda=0.001)
  pred_lasso <- X_test %*% lasso$coef + lasso$intercept
  results$lasso <- evaluate_model(as.numeric(pred_lasso), y_test, "LASSO")

  # 6. SVM
  cat("\n--- Linear SVM ---\n")
  y_bin <- ifelse(y_train > median(y_train), 1, -1)
  svm_m <- svm_linear(X_train, y_bin, C=1.0, max_iter=2000, lr=0.0005)
  svm_p <- predict_svm(svm_m, X_test)
  results$svm <- evaluate_model(svm_p$score, y_test, "Linear SVM")

  # Summary comparison
  cat("\n=== MODEL COMPARISON ===\n")
  model_names <- names(results)
  comp_df <- data.frame(
    Model    = model_names,
    IC       = sapply(results, `[[`, "IC"),
    HitRate  = sapply(results, `[[`, "hit_rate"),
    Sharpe   = sapply(results, `[[`, "sharpe"),
    R2       = sapply(results, `[[`, "r2")
  )
  comp_df <- comp_df[order(-comp_df$Sharpe), ]
  print(comp_df)
  cat(sprintf("\nBest model by Sharpe: %s\n", comp_df$Model[1]))

  invisible(list(results=results, comparison=comp_df, rf=rf, gb=gb, lasso=lasso))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(42)
# n <- 400; p <- 10
# X <- matrix(rnorm(n*p), n, p)
# colnames(X) <- paste0("F", 1:p)
# # True signal: first 3 features predict y
# y <- 0.3*X[,1] + 0.2*X[,2] - 0.15*X[,3] + rnorm(n, 0, 0.8)
# result <- run_ml_pipeline(X, y)

# =============================================================================
# EXTENDED ML SIGNALS: Ensemble Methods, Online Learning, Feature Engineering,
# Conformal Prediction, and Neural Network from Scratch
# =============================================================================

# -----------------------------------------------------------------------------
# Feature Engineering for Financial Time Series
# Creates lag features, rolling statistics, and interaction terms
# Critical for ML models: raw returns are noisy; features must capture signal
# -----------------------------------------------------------------------------
engineer_features <- function(X_mat, y_vec = NULL, n_lags = 5,
                               rolling_windows = c(5, 21, 63),
                               add_interactions = FALSE) {
  n <- nrow(X_mat); p <- ncol(X_mat)
  col_names <- colnames(X_mat)
  if (is.null(col_names)) col_names <- paste0("F", 1:p)

  feature_list <- list(X_mat)

  # Lag features
  for (lag in 1:n_lags) {
    lag_mat <- rbind(matrix(NA, lag, p), X_mat[1:(n-lag), , drop=FALSE])
    colnames(lag_mat) <- paste0(col_names, "_lag", lag)
    feature_list[[length(feature_list)+1]] <- lag_mat
  }

  # Rolling mean and std features
  for (w in rolling_windows) {
    roll_mean <- apply(X_mat, 2, function(x) filter(x, rep(1/w, w), sides=1))
    roll_std  <- apply(X_mat, 2, function(x) {
      sapply(1:length(x), function(t) {
        if (t < w) NA else sd(x[(t-w+1):t])
      })
    })
    colnames(roll_mean) <- paste0(col_names, "_rmean", w)
    colnames(roll_std)  <- paste0(col_names, "_rstd", w)
    feature_list[[length(feature_list)+1]] <- roll_mean
    feature_list[[length(feature_list)+1]] <- roll_std
  }

  # Interaction terms (pairwise products, only for small p)
  if (add_interactions && p <= 10) {
    for (i in 1:(p-1)) {
      for (j in (i+1):p) {
        inter <- X_mat[, i] * X_mat[, j]
        inter_mat <- matrix(inter, n, 1)
        colnames(inter_mat) <- paste0(col_names[i], "x", col_names[j])
        feature_list[[length(feature_list)+1]] <- inter_mat
      }
    }
  }

  result <- do.call(cbind, feature_list)

  # Remove rows with any NA (from lags/rolling)
  complete_rows <- complete.cases(result)

  list(
    features = result[complete_rows, ],
    complete_idx = which(complete_rows),
    y_aligned = if (!is.null(y_vec)) y_vec[complete_rows] else NULL,
    feature_names = colnames(result),
    n_features = ncol(result)
  )
}

# -----------------------------------------------------------------------------
# Simple Neural Network (2-layer MLP) from Scratch
# Architecture: input -> hidden (ReLU) -> output (linear)
# Trained via backpropagation with mini-batch stochastic gradient descent
# -----------------------------------------------------------------------------
relu <- function(x) pmax(x, 0)
relu_grad <- function(x) as.numeric(x > 0)

nn_forward <- function(X, W1, b1, W2, b2) {
  # Layer 1: linear + ReLU
  Z1 <- X %*% W1 + matrix(rep(b1, nrow(X)), nrow(X), length(b1), byrow=TRUE)
  A1 <- relu(Z1)
  # Layer 2: linear
  Z2 <- A1 %*% W2 + b2
  list(Z1=Z1, A1=A1, Z2=Z2, output=Z2)
}

nn_backward <- function(X, y, Z1, A1, Z2, W1, W2, b1, b2) {
  n <- nrow(X)
  # Output layer gradient (MSE loss)
  dL_dZ2 <- (Z2 - matrix(y, n, 1)) * 2 / n
  dL_dW2 <- t(A1) %*% dL_dZ2
  dL_db2 <- colSums(dL_dZ2)
  # Hidden layer gradient
  dL_dA1 <- dL_dZ2 %*% t(W2)
  dL_dZ1 <- dL_dA1 * relu_grad(Z1)
  dL_dW1 <- t(X) %*% dL_dZ1
  dL_db1 <- colSums(dL_dZ1)
  list(dW1=dL_dW1, db1=dL_db1, dW2=dL_dW2, db2=dL_db2)
}

train_nn <- function(X, y, hidden_size = 32, lr = 0.01, epochs = 100,
                      batch_size = 32, l2 = 0.001) {
  n <- nrow(X); p <- ncol(X)
  # Xavier initialization
  W1 <- matrix(rnorm(p * hidden_size, 0, sqrt(2/p)), p, hidden_size)
  b1 <- rep(0, hidden_size)
  W2 <- matrix(rnorm(hidden_size, 0, sqrt(2/hidden_size)), hidden_size, 1)
  b2 <- 0

  loss_history <- numeric(epochs)

  for (epoch in 1:epochs) {
    # Mini-batch SGD
    idx <- sample(n)
    for (start in seq(1, n, by=batch_size)) {
      end <- min(start + batch_size - 1, n)
      batch_idx <- idx[start:end]
      X_b <- X[batch_idx, , drop=FALSE]
      y_b <- y[batch_idx]

      fwd <- nn_forward(X_b, W1, b1, W2, b2)
      grads <- nn_backward(X_b, y_b, fwd$Z1, fwd$A1, fwd$Z2, W1, W2, b1, b2)

      # L2 regularization
      W1 <- W1 - lr * (grads$dW1 + l2 * W1)
      b1 <- b1 - lr * grads$db1
      W2 <- W2 - lr * (grads$dW2 + l2 * W2)
      b2 <- b2 - lr * grads$db2
    }

    fwd_full <- nn_forward(X, W1, b1, W2, b2)
    loss_history[epoch] <- mean((fwd_full$output - y)^2)
  }

  list(W1=W1, b1=b1, W2=W2, b2=b2, loss_history=loss_history,
       final_loss = loss_history[epochs])
}

predict_nn <- function(model, X_new) {
  fwd <- nn_forward(X_new, model$W1, model$b1, model$W2, model$b2)
  as.vector(fwd$output)
}

# -----------------------------------------------------------------------------
# Online Learning: Passive-Aggressive Regression (Crammer et al. 2006)
# Updates model with each new observation; ideal for non-stationary financial data
# No need to retrain from scratch; adjusts weights to minimize hinge loss
# -----------------------------------------------------------------------------
pa_regression <- function(X, y, C = 1.0, variant = 2) {
  # Passive-Aggressive Regression
  # Variant 1: tau = min(C, loss/||x||^2)
  # Variant 2: tau = loss / (||x||^2 + 1/(2C))
  n <- nrow(X); p <- ncol(X)
  w <- rep(0, p); b <- 0

  predictions <- numeric(n)
  losses <- numeric(n)

  for (t in 1:n) {
    x_t <- X[t, ]; y_t <- y[t]

    # Predict
    y_hat <- sum(w * x_t) + b
    predictions[t] <- y_hat

    # Epsilon-insensitive loss
    eps <- 0.01
    loss_t <- max(0, abs(y_hat - y_t) - eps)
    losses[t] <- loss_t

    if (loss_t > 0) {
      # Update
      norm2 <- sum(x_t^2)
      sign_err <- sign(y_hat - y_t)

      if (variant == 1) {
        tau <- min(C, loss_t / (norm2 + 1e-8))
      } else {
        tau <- loss_t / (norm2 + 1/(2*C) + 1e-8)
      }

      w <- w - tau * sign_err * x_t
      b <- b - tau * sign_err
    }
  }

  list(
    weights = w, bias = b,
    predictions = predictions,
    mse = mean((predictions - y)^2),
    avg_loss = mean(losses),
    pct_zero_loss = mean(losses == 0)
  )
}

# -----------------------------------------------------------------------------
# Conformal Prediction Intervals: distribution-free coverage guarantees
# For any alpha, conformal intervals contain the true value 1-alpha fraction
# Split conformal: calibrate on held-out set, apply to test
# -----------------------------------------------------------------------------
conformal_predict <- function(X_train, y_train, X_cal, y_cal, X_test,
                               model_fn, predict_fn, alpha = 0.10) {
  # Fit model on training set
  model <- model_fn(X_train, y_train)

  # Calibration scores: |y_i - y_hat_i|
  y_hat_cal <- predict_fn(model, X_cal)
  cal_scores <- abs(y_cal - y_hat_cal)

  # Quantile of calibration scores (inflate by 1/(n+1) for finite-sample coverage)
  n_cal <- length(cal_scores)
  q_level <- ceiling((1 - alpha) * (n_cal + 1)) / n_cal
  q_level <- min(q_level, 1)
  q_hat <- quantile(cal_scores, q_level, names=FALSE)

  # Test set predictions and intervals
  y_hat_test <- predict_fn(model, X_test)
  intervals  <- cbind(lower = y_hat_test - q_hat,
                      point = y_hat_test,
                      upper = y_hat_test + q_hat)

  list(
    intervals = intervals,
    q_hat = q_hat,
    coverage_alpha = 1 - alpha,
    interval_width = 2 * q_hat,
    model = model
  )
}

# -----------------------------------------------------------------------------
# Elastic Net: combines LASSO (L1) and Ridge (L2) penalties
# Useful when features are correlated (LASSO picks one arbitrarily; EN spreads)
# Solved via coordinate descent with both penalties
# -----------------------------------------------------------------------------
elastic_net <- function(X, y, alpha_mix = 0.5, lambda = 0.01,
                         max_iter = 1000, tol = 1e-6) {
  # alpha_mix: mixing parameter (0=Ridge, 1=LASSO, 0.5=ElasticNet)
  n <- nrow(X); p <- ncol(X)

  # Standardize
  X_mean <- colMeans(X); X_sd <- apply(X, 2, sd)
  X_sd[X_sd == 0] <- 1
  Xs <- scale(X, center=X_mean, scale=X_sd)
  y_mean <- mean(y); ys <- y - y_mean

  beta <- rep(0, p); r <- ys  # residuals

  for (iter in 1:max_iter) {
    beta_old <- beta
    for (j in 1:p) {
      r_j <- r + Xs[, j] * beta[j]  # partial residual
      rho_j <- mean(Xs[, j] * r_j)  # partial correlation

      # Coordinate update: soft-threshold with elastic net
      l1_pen <- lambda * alpha_mix
      l2_pen <- lambda * (1 - alpha_mix)
      beta[j] <- sign(rho_j) * max(abs(rho_j) - l1_pen, 0) /
                 (mean(Xs[, j]^2) + l2_pen)
      r <- r_j - Xs[, j] * beta[j]
    }
    if (max(abs(beta - beta_old)) < tol) break
  }

  # Unstandardize
  beta_orig <- beta / X_sd
  intercept  <- y_mean - sum(beta_orig * X_mean)

  y_hat <- as.vector(X %*% beta_orig) + intercept
  list(
    beta = beta_orig, intercept = intercept,
    mse = mean((y - y_hat)^2),
    r2  = 1 - sum((y-y_hat)^2)/sum((y-y_mean)^2),
    n_nonzero = sum(beta != 0),
    alpha_mix = alpha_mix, lambda = lambda
  )
}

# Extended ML example:
# feats <- engineer_features(X_raw, y, n_lags=5)
# nn    <- train_nn(feats$features, feats$y_aligned, hidden_size=32)
# pa    <- pa_regression(X_stream, y_stream, C=1.0)
# en    <- elastic_net(X, y, alpha_mix=0.5, lambda=0.01)
# cp    <- conformal_predict(X_tr, y_tr, X_cal, y_cal, X_test,
#            model_fn=function(X,y) lm(y~X),
#            predict_fn=function(m,X) predict(m, data.frame(X=X)))
