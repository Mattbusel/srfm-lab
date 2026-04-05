## machine_learning_advanced.R
## Advanced machine learning from scratch: gradient boosting, random forest,
## neural networks, cross-validation, feature importance
## Pure base R — no library() calls

# ============================================================
# 1. DECISION TREE (regression / classification stub)
# ============================================================

.split_gini <- function(y_left, y_right) {
  n <- length(y_left) + length(y_right)
  gini <- function(y) {
    if (length(y) == 0) return(0)
    p <- table(y) / length(y)
    1 - sum(p^2)
  }
  (length(y_left) / n) * gini(y_left) +
    (length(y_right) / n) * gini(y_right)
}

.split_mse <- function(y_left, y_right) {
  mse <- function(y) if (length(y) < 2) 0 else var(y) * (length(y) - 1) / length(y)
  n <- length(y_left) + length(y_right)
  (length(y_left) / n) * mse(y_left) +
    (length(y_right) / n) * mse(y_right)
}

.best_split <- function(X, y, min_samples = 2, type = "regression") {
  best_feat <- NULL; best_thresh <- NULL; best_loss <- Inf
  n_feat <- ncol(X)
  for (j in seq_len(n_feat)) {
    vals <- sort(unique(X[, j]))
    if (length(vals) < 2) next
    thresholds <- (vals[-length(vals)] + vals[-1]) / 2
    for (thr in thresholds) {
      left  <- y[X[, j] <= thr]
      right <- y[X[, j] >  thr]
      if (length(left) < min_samples || length(right) < min_samples) next
      loss <- if (type == "regression") .split_mse(left, right) else .split_gini(left, right)
      if (loss < best_loss) {
        best_loss <- loss; best_feat <- j; best_thresh <- thr
      }
    }
  }
  list(feature = best_feat, threshold = best_thresh, loss = best_loss)
}

.build_tree <- function(X, y, depth = 0, max_depth = 5,
                        min_samples_leaf = 5, type = "regression") {
  node <- list(n = length(y), prediction = mean(y))
  if (depth >= max_depth || length(y) <= min_samples_leaf) return(node)
  sp <- .best_split(X, y, min_samples_leaf, type)
  if (is.null(sp$feature)) return(node)
  left_idx  <- X[, sp$feature] <= sp$threshold
  right_idx <- !left_idx
  node$feature   <- sp$feature
  node$threshold <- sp$threshold
  node$left  <- .build_tree(X[left_idx,  , drop = FALSE], y[left_idx],
                             depth + 1, max_depth, min_samples_leaf, type)
  node$right <- .build_tree(X[right_idx, , drop = FALSE], y[right_idx],
                             depth + 1, max_depth, min_samples_leaf, type)
  node
}

.predict_tree <- function(node, x) {
  if (is.null(node$feature)) return(node$prediction)
  if (x[node$feature] <= node$threshold)
    .predict_tree(node$left, x)
  else
    .predict_tree(node$right, x)
}

predict_tree_matrix <- function(tree, X) {
  apply(X, 1, function(x) .predict_tree(tree, x))
}

# ============================================================
# 2. RANDOM FOREST
# ============================================================

random_forest_train <- function(X, y, n_trees = 100, max_depth = 6,
                                min_samples_leaf = 5, mtry = NULL,
                                type = "regression", seed = 42) {
  set.seed(seed)
  n <- nrow(X); p <- ncol(X)
  if (is.null(mtry)) mtry <- max(1L, floor(sqrt(p)))
  trees <- vector("list", n_trees)
  oob_preds <- matrix(NA_real_, n, n_trees)

  for (t in seq_len(n_trees)) {
    bag_idx <- sample(n, n, replace = TRUE)
    oob_idx <- setdiff(seq_len(n), unique(bag_idx))
    feat_idx <- sort(sample(p, mtry))
    Xb <- X[bag_idx, feat_idx, drop = FALSE]
    yb <- y[bag_idx]
    tree <- .build_tree(Xb, yb, max_depth = max_depth,
                        min_samples_leaf = min_samples_leaf, type = type)
    trees[[t]] <- list(tree = tree, features = feat_idx)
    if (length(oob_idx) > 0) {
      Xoob <- X[oob_idx, feat_idx, drop = FALSE]
      oob_preds[oob_idx, t] <- predict_tree_matrix(tree, Xoob)
    }
  }

  oob_pred_mean <- rowMeans(oob_preds, na.rm = TRUE)
  oob_rmse <- sqrt(mean((y - oob_pred_mean)^2, na.rm = TRUE))

  list(trees = trees, oob_rmse = oob_rmse, n_trees = n_trees,
       n_features = p, mtry = mtry, type = type)
}

random_forest_predict <- function(rf, X) {
  preds <- sapply(rf$trees, function(t) {
    predict_tree_matrix(t$tree, X[, t$features, drop = FALSE])
  })
  if (rf$type == "regression") rowMeans(preds)
  else {
    apply(preds, 1, function(row) {
      tt <- table(round(row)); as.numeric(names(tt)[which.max(tt)])
    })
  }
}

rf_feature_importance <- function(rf, X, y, n_permute = 1) {
  base_pred <- random_forest_predict(rf, X)
  base_rmse <- sqrt(mean((y - base_pred)^2))
  p <- ncol(X)
  importance <- numeric(p)
  for (j in seq_len(p)) {
    perm_rmse <- numeric(n_permute)
    for (k in seq_len(n_permute)) {
      Xp <- X; Xp[, j] <- sample(Xp[, j])
      pp <- random_forest_predict(rf, Xp)
      perm_rmse[k] <- sqrt(mean((y - pp)^2))
    }
    importance[j] <- mean(perm_rmse) - base_rmse
  }
  importance / sum(abs(importance) + 1e-12)
}

# ============================================================
# 3. GRADIENT BOOSTING (regression)
# ============================================================

gbm_train <- function(X, y, n_trees = 200, max_depth = 3,
                      learning_rate = 0.05, subsample = 0.8,
                      min_samples_leaf = 10, seed = 42) {
  set.seed(seed)
  n <- nrow(X)
  f0 <- mean(y)
  F_pred <- rep(f0, n)
  trees <- vector("list", n_trees)
  train_loss <- numeric(n_trees)

  for (t in seq_len(n_trees)) {
    residuals <- y - F_pred
    idx <- sample(n, floor(n * subsample))
    Xs <- X[idx, , drop = FALSE]; rs <- residuals[idx]
    tree <- .build_tree(Xs, rs, max_depth = max_depth,
                        min_samples_leaf = min_samples_leaf,
                        type = "regression")
    gamma <- predict_tree_matrix(tree, X)
    F_pred <- F_pred + learning_rate * gamma
    trees[[t]] <- tree
    train_loss[t] <- sqrt(mean(residuals^2))
  }

  list(f0 = f0, trees = trees, learning_rate = learning_rate,
       n_trees = n_trees, train_loss = train_loss)
}

gbm_predict <- function(gbm, X, n_trees = NULL) {
  if (is.null(n_trees)) n_trees <- gbm$n_trees
  F_pred <- rep(gbm$f0, nrow(X))
  for (t in seq_len(n_trees)) {
    F_pred <- F_pred + gbm$learning_rate * predict_tree_matrix(gbm$trees[[t]], X)
  }
  F_pred
}

gbm_feature_importance <- function(gbm, X, y) {
  base_pred <- gbm_predict(gbm, X)
  base_mse  <- mean((y - base_pred)^2)
  p <- ncol(X)
  imp <- numeric(p)
  for (j in seq_len(p)) {
    Xp <- X; Xp[, j] <- sample(Xp[, j])
    pp <- gbm_predict(gbm, Xp)
    imp[j] <- mean((y - pp)^2) - base_mse
  }
  imp / (sum(abs(imp)) + 1e-12)
}

gbm_partial_dependence <- function(gbm, X, feature_idx, grid_size = 20) {
  x_vals <- seq(min(X[, feature_idx]), max(X[, feature_idx]), length.out = grid_size)
  pdp <- numeric(grid_size)
  for (i in seq_along(x_vals)) {
    Xm <- X; Xm[, feature_idx] <- x_vals[i]
    pdp[i] <- mean(gbm_predict(gbm, Xm))
  }
  data.frame(x = x_vals, y = pdp)
}

# ============================================================
# 4. NEURAL NETWORK (MLP with backprop)
# ============================================================

.sigmoid   <- function(x) 1 / (1 + exp(-x))
.sigmoid_d <- function(x) { s <- .sigmoid(x); s * (1 - s) }
.relu      <- function(x) pmax(x, 0)
.relu_d    <- function(x) ifelse(x > 0, 1, 0)
.tanh_d    <- function(x) 1 - tanh(x)^2

nn_init <- function(layer_sizes, activation = "relu", seed = 42) {
  set.seed(seed)
  n_layers <- length(layer_sizes)
  weights <- vector("list", n_layers - 1)
  biases  <- vector("list", n_layers - 1)
  for (l in seq_len(n_layers - 1)) {
    fan_in <- layer_sizes[l]; fan_out <- layer_sizes[l + 1]
    scale  <- sqrt(2 / fan_in)
    weights[[l]] <- matrix(rnorm(fan_in * fan_out, 0, scale), fan_in, fan_out)
    biases[[l]]  <- rep(0, fan_out)
  }
  list(weights = weights, biases = biases,
       activation = activation, layer_sizes = layer_sizes)
}

nn_forward <- function(net, X) {
  af <- switch(net$activation, relu = .relu, sigmoid = .sigmoid, tanh = tanh)
  n_layers <- length(net$weights)
  activations <- vector("list", n_layers + 1)
  pre_acts    <- vector("list", n_layers)
  activations[[1]] <- X
  for (l in seq_len(n_layers)) {
    z <- activations[[l]] %*% net$weights[[l]] +
      matrix(net$biases[[l]], nrow(X), length(net$biases[[l]]), byrow = TRUE)
    pre_acts[[l]] <- z
    activations[[l + 1]] <- if (l < n_layers) af(z) else z
  }
  list(output = activations[[n_layers + 1]],
       activations = activations, pre_acts = pre_acts)
}

nn_backward <- function(net, fwd, y, lr = 0.001, lambda = 1e-4) {
  n  <- nrow(y)
  af_d <- switch(net$activation,
                 relu    = .relu_d,
                 sigmoid = .sigmoid_d,
                 tanh    = .tanh_d)
  n_layers <- length(net$weights)
  dW <- vector("list", n_layers)
  db <- vector("list", n_layers)

  delta <- (fwd$output - y) / n
  for (l in rev(seq_len(n_layers))) {
    dW[[l]] <- t(fwd$activations[[l]]) %*% delta + lambda * net$weights[[l]]
    db[[l]] <- colMeans(delta)
    if (l > 1) delta <- (delta %*% t(net$weights[[l]])) * af_d(fwd$pre_acts[[l - 1]])
  }
  for (l in seq_len(n_layers)) {
    net$weights[[l]] <- net$weights[[l]] - lr * dW[[l]]
    net$biases[[l]]  <- net$biases[[l]]  - lr * db[[l]]
  }
  net
}

nn_train <- function(net, X, y, epochs = 500, lr = 0.001, batch_size = 64,
                     lambda = 1e-4, verbose = TRUE) {
  y <- matrix(y, ncol = 1)
  n <- nrow(X)
  loss_hist <- numeric(epochs)
  for (ep in seq_len(epochs)) {
    idx <- sample(n)
    for (start in seq(1, n, by = batch_size)) {
      end <- min(start + batch_size - 1, n)
      bi  <- idx[start:end]
      fwd <- nn_forward(net, X[bi, , drop = FALSE])
      net <- nn_backward(net, fwd, y[bi, , drop = FALSE], lr, lambda)
    }
    fwd_full <- nn_forward(net, X)
    loss_hist[ep] <- mean((fwd_full$output - y)^2)
    if (verbose && ep %% 100 == 0)
      cat(sprintf("Epoch %4d | MSE: %.6f\n", ep, loss_hist[ep]))
  }
  list(net = net, loss_hist = loss_hist)
}

nn_predict <- function(net, X) {
  fwd <- nn_forward(net, X)
  as.vector(fwd$output)
}

# ============================================================
# 5. CROSS-VALIDATION FRAMEWORK
# ============================================================

kfold_cv <- function(X, y, model_fn, predict_fn, k = 5,
                     metric = "rmse", seed = 42, ...) {
  set.seed(seed)
  n   <- nrow(X)
  idx <- sample(n)
  folds <- cut(seq_len(n), breaks = k, labels = FALSE)
  scores <- numeric(k)

  for (fold in seq_len(k)) {
    val_idx   <- idx[folds == fold]
    train_idx <- idx[folds != fold]
    model <- model_fn(X[train_idx, , drop = FALSE], y[train_idx], ...)
    preds <- predict_fn(model, X[val_idx, , drop = FALSE])
    scores[fold] <- switch(metric,
      rmse = sqrt(mean((y[val_idx] - preds)^2)),
      mae  = mean(abs(y[val_idx] - preds)),
      r2   = 1 - sum((y[val_idx] - preds)^2) / sum((y[val_idx] - mean(y[val_idx]))^2)
    )
  }
  list(scores = scores, mean = mean(scores), sd = sd(scores),
       metric = metric, k = k)
}

time_series_cv <- function(X, y, model_fn, predict_fn,
                           initial = 0.6, horizon = 20, step = 10,
                           metric = "rmse", ...) {
  n    <- nrow(X)
  init <- floor(n * initial)
  starts <- seq(init, n - horizon, by = step)
  scores <- numeric(length(starts))

  for (i in seq_along(starts)) {
    s   <- starts[i]
    tr  <- seq_len(s)
    val <- seq(s + 1, min(s + horizon, n))
    model <- model_fn(X[tr, , drop = FALSE], y[tr], ...)
    preds <- predict_fn(model, X[val, , drop = FALSE])
    scores[i] <- switch(metric,
      rmse = sqrt(mean((y[val] - preds)^2)),
      mae  = mean(abs(y[val] - preds)),
      r2   = 1 - sum((y[val] - preds)^2) / sum((y[val] - mean(y[val]))^2)
    )
  }
  list(scores = scores, mean = mean(scores), sd = sd(scores), metric = metric)
}

# ============================================================
# 6. HYPERPARAMETER TUNING
# ============================================================

grid_search_cv <- function(X, y, model_fn, predict_fn, param_grid,
                           k = 5, metric = "rmse", seed = 42) {
  grid <- expand.grid(param_grid, stringsAsFactors = FALSE)
  results <- vector("list", nrow(grid))
  for (i in seq_len(nrow(grid))) {
    params <- as.list(grid[i, ])
    cv_res <- tryCatch(
      do.call(kfold_cv, c(list(X = X, y = y,
                               model_fn = model_fn,
                               predict_fn = predict_fn,
                               k = k, metric = metric, seed = seed),
                          params)),
      error = function(e) list(mean = Inf, sd = NA)
    )
    results[[i]] <- c(params, list(cv_mean = cv_res$mean, cv_sd = cv_res$sd))
  }
  res_df <- do.call(rbind, lapply(results, as.data.frame))
  best   <- which.min(res_df$cv_mean)
  list(results = res_df, best_params = as.list(res_df[best, ]),
       best_score = res_df$cv_mean[best])
}

random_search_cv <- function(X, y, model_fn, predict_fn, param_ranges,
                             n_iter = 30, k = 5, metric = "rmse", seed = 42) {
  set.seed(seed)
  results <- vector("list", n_iter)
  for (i in seq_len(n_iter)) {
    params <- lapply(param_ranges, function(r) {
      if (is.numeric(r)) runif(1, r[1], r[2])
      else sample(r, 1)
    })
    cv_res <- tryCatch(
      do.call(kfold_cv, c(list(X = X, y = y,
                               model_fn = model_fn,
                               predict_fn = predict_fn,
                               k = k, metric = metric, seed = seed + i),
                          params)),
      error = function(e) list(mean = Inf, sd = NA)
    )
    results[[i]] <- c(params, list(cv_mean = cv_res$mean, cv_sd = cv_res$sd))
  }
  res_df <- do.call(rbind, lapply(results, as.data.frame))
  best   <- which.min(res_df$cv_mean)
  list(results = res_df, best_params = as.list(res_df[best, ]),
       best_score = res_df$cv_mean[best])
}

# ============================================================
# 7. FEATURE ENGINEERING & SELECTION
# ============================================================

feature_importance_shap_approx <- function(model_fn, predict_fn,
                                           X, y, n_samples = 100, seed = 42) {
  set.seed(seed)
  n <- nrow(X); p <- ncol(X)
  model <- model_fn(X, y)
  shap_vals <- matrix(0, n_samples, p)

  for (i in seq_len(n_samples)) {
    baseline <- X[sample(n, 1), ]
    point    <- X[sample(n, 1), ]
    perm     <- sample(p)
    current  <- baseline
    for (j in seq_len(p)) {
      feat <- perm[j]
      prev_pred <- predict_fn(model, matrix(current, nrow = 1))
      current[feat] <- point[feat]
      new_pred  <- predict_fn(model, matrix(current, nrow = 1))
      shap_vals[i, feat] <- shap_vals[i, feat] + (new_pred - prev_pred)
    }
  }
  colMeans(abs(shap_vals))
}

mutual_information <- function(x, y, n_bins = 10) {
  bx <- cut(x, breaks = n_bins, labels = FALSE)
  by <- cut(y, breaks = n_bins, labels = FALSE)
  joint <- table(bx, by) / length(x)
  px    <- rowSums(joint)
  py    <- colSums(joint)
  mi <- 0
  for (i in seq_len(nrow(joint)))
    for (j in seq_len(ncol(joint)))
      if (joint[i, j] > 0 && px[i] > 0 && py[j] > 0)
        mi <- mi + joint[i, j] * log(joint[i, j] / (px[i] * py[j]))
  mi
}

recursive_feature_elimination <- function(X, y, model_fn, predict_fn,
                                          importance_fn, k = 5,
                                          min_features = 2) {
  p    <- ncol(X)
  feat <- seq_len(p)
  path <- list()
  while (length(feat) > min_features) {
    cv  <- kfold_cv(X[, feat, drop = FALSE], y, model_fn, predict_fn, k = k)
    imp <- importance_fn(model_fn(X[, feat, drop = FALSE], y),
                         X[, feat, drop = FALSE], y)
    path <- c(path, list(list(features = feat, cv_score = cv$mean, importance = imp)))
    worst <- feat[which.min(imp)]
    feat  <- setdiff(feat, worst)
  }
  path
}

# ============================================================
# 8. STACKING / ENSEMBLE
# ============================================================

stacking_ensemble <- function(X, y, base_learners, meta_learner_fn,
                              meta_predict_fn, k = 5, seed = 42) {
  set.seed(seed)
  n  <- nrow(X)
  nb <- length(base_learners)
  oof_preds <- matrix(NA_real_, n, nb)
  idx   <- sample(n)
  folds <- cut(seq_len(n), breaks = k, labels = FALSE)

  for (b in seq_len(nb)) {
    bl <- base_learners[[b]]
    for (fold in seq_len(k)) {
      val_idx   <- idx[folds == fold]
      train_idx <- idx[folds != fold]
      m <- bl$train(X[train_idx, , drop = FALSE], y[train_idx])
      oof_preds[val_idx, b] <- bl$predict(m, X[val_idx, , drop = FALSE])
    }
  }

  base_models <- lapply(base_learners, function(bl) bl$train(X, y))
  meta_model  <- meta_learner_fn(oof_preds, y)

  list(base_models = base_models, meta_model = meta_model,
       oof_preds = oof_preds, meta_learner_fn = meta_learner_fn,
       meta_predict_fn = meta_predict_fn,
       base_learners = base_learners)
}

stacking_predict <- function(stack, X) {
  nb <- length(stack$base_models)
  base_preds <- matrix(NA_real_, nrow(X), nb)
  for (b in seq_len(nb))
    base_preds[, b] <- stack$base_learners[[b]]$predict(
      stack$base_models[[b]], X)
  stack$meta_predict_fn(stack$meta_model, base_preds)
}

# ============================================================
# 9. EVALUATION UTILITIES
# ============================================================

ml_metrics <- function(y_true, y_pred, type = "regression") {
  if (type == "regression") {
    resid <- y_true - y_pred
    ss_res <- sum(resid^2)
    ss_tot <- sum((y_true - mean(y_true))^2)
    list(
      rmse   = sqrt(mean(resid^2)),
      mae    = mean(abs(resid)),
      mape   = mean(abs(resid / (y_true + 1e-8))) * 100,
      r2     = 1 - ss_res / ss_tot,
      adj_r2 = NA,
      max_err = max(abs(resid))
    )
  } else {
    classes <- unique(y_true)
    acc <- mean(y_pred == y_true)
    cm  <- table(Predicted = y_pred, Actual = y_true)
    list(accuracy = acc, confusion_matrix = cm)
  }
}

learning_curve <- function(X, y, model_fn, predict_fn,
                           train_sizes = seq(0.1, 1, by = 0.1),
                           k = 3, seed = 42) {
  set.seed(seed)
  n    <- nrow(X)
  res  <- vector("list", length(train_sizes))
  for (i in seq_along(train_sizes)) {
    sz     <- floor(n * train_sizes[i])
    idx    <- sample(n, sz)
    Xs     <- X[idx, , drop = FALSE]; ys <- y[idx]
    cv_res <- kfold_cv(Xs, ys, model_fn, predict_fn, k = k, seed = seed)
    train_model <- model_fn(Xs, ys)
    train_pred  <- predict_fn(train_model, Xs)
    train_rmse  <- sqrt(mean((ys - train_pred)^2))
    res[[i]] <- list(size = sz, train_score = train_rmse,
                     cv_score = cv_res$mean, cv_sd = cv_res$sd)
  }
  do.call(rbind, lapply(res, as.data.frame))
}

calibration_analysis <- function(y_true, y_prob, n_bins = 10) {
  breaks <- seq(0, 1, length.out = n_bins + 1)
  bins   <- cut(y_prob, breaks, include.lowest = TRUE)
  cal    <- tapply(y_true, bins, mean)
  mid    <- (breaks[-1] + breaks[-length(breaks)]) / 2
  data.frame(predicted = mid, observed = as.numeric(cal),
             n = as.numeric(table(bins)))
}
