# =============================================================================
# advanced_ml.R
# Advanced Machine Learning from scratch in pure base R:
# SVM, Gaussian Processes, MLP neural net with Adam, attention mechanism,
# ensemble stacking, hyperparameter search, SHAP values, calibration.
# All implemented without external packages.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. SUPPORT VECTOR MACHINE (Linear Kernel, Simplified SMO)
# ---------------------------------------------------------------------------

#' Simplified SMO (Sequential Minimal Optimization) for linear SVM
#' Solves: max sum(alpha_i) - 0.5 * sum_ij(alpha_i*alpha_j*y_i*y_j*x_i'x_j)
#' subject to: 0 <= alpha_i <= C, sum(alpha_i * y_i) = 0
svm_smo <- function(X, y, C = 1.0, tol = 1e-3, max_passes = 50) {
  n <- nrow(X)
  p <- ncol(X)

  # Compute kernel (linear = dot product)
  K <- X %*% t(X)

  alpha <- rep(0, n)
  b <- 0
  passes <- 0

  # Cache of decision function
  f_val <- function() K %*% (alpha * y) + b

  while (passes < max_passes) {
    num_changed <- 0
    f <- f_val()
    E <- f - y  # Error

    for (i in seq_len(n)) {
      # KKT violation check
      ri <- y[i] * E[i]
      if ((ri < -tol && alpha[i] < C) || (ri > tol && alpha[i] > 0)) {
        # Select j randomly (not i)
        j <- sample(setdiff(seq_len(n), i), 1)

        # Compute bounds
        if (y[i] == y[j]) {
          L <- max(0, alpha[i] + alpha[j] - C)
          H <- min(C, alpha[i] + alpha[j])
        } else {
          L <- max(0, alpha[j] - alpha[i])
          H <- min(C, C + alpha[j] - alpha[i])
        }
        if (L >= H) next

        eta <- 2 * K[i, j] - K[i, i] - K[j, j]
        if (eta >= 0) next

        # Update alpha_j
        alpha_j_new <- alpha[j] - y[j] * (E[i] - E[j]) / eta
        alpha_j_new <- min(H, max(L, alpha_j_new))
        if (abs(alpha_j_new - alpha[j]) < 1e-5) next

        # Update alpha_i
        alpha_i_new <- alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new)

        # Update bias
        b1 <- b - E[i] - y[i]*(alpha_i_new-alpha[i])*K[i,i] -
              y[j]*(alpha_j_new-alpha[j])*K[i,j]
        b2 <- b - E[j] - y[i]*(alpha_i_new-alpha[i])*K[i,j] -
              y[j]*(alpha_j_new-alpha[j])*K[j,j]

        alpha[i] <- alpha_i_new
        alpha[j] <- alpha_j_new
        b <- if (0 < alpha[i] && alpha[i] < C) b1
             else if (0 < alpha[j] && alpha[j] < C) b2
             else (b1 + b2) / 2

        num_changed <- num_changed + 1
      }
    }

    if (num_changed == 0) passes <- passes + 1 else passes <- 0
  }

  # Weight vector w = sum(alpha_i * y_i * x_i)
  sv_mask <- alpha > 1e-5
  w <- as.vector(t(X[sv_mask, , drop=FALSE]) %*% (alpha[sv_mask] * y[sv_mask]))

  list(
    alpha = alpha,
    b = b,
    w = w,
    support_vectors = which(sv_mask),
    n_sv = sum(sv_mask),
    predict = function(Xnew) sign(as.vector(Xnew %*% w) + b)
  )
}

#' SVM with RBF kernel (store dual form, predict via support vectors)
svm_rbf <- function(X, y, C = 1.0, gamma = NULL, tol = 1e-3, max_passes = 30) {
  n <- nrow(X)
  if (is.null(gamma)) gamma <- 1 / ncol(X)

  # RBF kernel matrix
  K <- matrix(0, n, n)
  for (i in seq_len(n)) {
    for (j in i:n) {
      d2 <- sum((X[i,] - X[j,])^2)
      K[i,j] <- K[j,i] <- exp(-gamma * d2)
    }
  }

  alpha <- rep(0, n); b <- 0; passes <- 0

  f_val <- function() K %*% (alpha * y) + b

  while (passes < max_passes) {
    num_changed <- 0
    f <- f_val()
    E <- f - y

    for (i in seq_len(n)) {
      ri <- y[i] * E[i]
      if ((ri < -tol && alpha[i] < C) || (ri > tol && alpha[i] > 0)) {
        j <- sample(setdiff(seq_len(n), i), 1)
        if (y[i] == y[j]) {L <- max(0,alpha[i]+alpha[j]-C); H <- min(C,alpha[i]+alpha[j])}
        else               {L <- max(0,alpha[j]-alpha[i]);   H <- min(C,C+alpha[j]-alpha[i])}
        if (L >= H) next
        eta <- 2*K[i,j] - K[i,i] - K[j,j]
        if (eta >= 0) next
        aj_new <- min(H, max(L, alpha[j] - y[j]*(E[i]-E[j])/eta))
        if (abs(aj_new - alpha[j]) < 1e-5) next
        ai_new <- alpha[i] + y[i]*y[j]*(alpha[j] - aj_new)
        b1 <- b - E[i] - y[i]*(ai_new-alpha[i])*K[i,i] - y[j]*(aj_new-alpha[j])*K[i,j]
        b2 <- b - E[j] - y[i]*(ai_new-alpha[i])*K[i,j] - y[j]*(aj_new-alpha[j])*K[j,j]
        alpha[i] <- ai_new; alpha[j] <- aj_new
        b <- if (0<alpha[i]&&alpha[i]<C) b1 else if (0<alpha[j]&&alpha[j]<C) b2 else (b1+b2)/2
        num_changed <- num_changed + 1
      }
    }
    if (num_changed == 0) passes <- passes + 1 else passes <- 0
  }

  sv_mask <- alpha > 1e-5
  X_sv <- X[sv_mask, , drop=FALSE]
  alpha_sv <- alpha[sv_mask]
  y_sv <- y[sv_mask]

  predict_fn <- function(Xnew) {
    K_new <- matrix(0, nrow(Xnew), sum(sv_mask))
    for (i in seq_len(nrow(Xnew))) {
      for (j in seq_len(sum(sv_mask))) {
        d2 <- sum((Xnew[i,] - X_sv[j,])^2)
        K_new[i,j] <- exp(-gamma * d2)
      }
    }
    f_new <- K_new %*% (alpha_sv * y_sv) + b
    sign(f_new)
  }

  list(alpha=alpha, b=b, n_sv=sum(sv_mask), predict=predict_fn)
}

# ---------------------------------------------------------------------------
# 2. GAUSSIAN PROCESS REGRESSION
# ---------------------------------------------------------------------------

#' RBF (squared exponential) kernel
#' k(x,x') = sigma_f^2 * exp(-||x-x'||^2 / (2*l^2))
rbf_kernel <- function(X1, X2, l = 1.0, sigma_f = 1.0) {
  n1 <- nrow(X1); n2 <- nrow(X2)
  K <- matrix(0, n1, n2)
  for (i in seq_len(n1)) {
    for (j in seq_len(n2)) {
      d2 <- sum((X1[i,] - X2[j,])^2)
      K[i,j] <- sigma_f^2 * exp(-d2 / (2 * l^2))
    }
  }
  K
}

#' Gaussian Process regression: posterior predictive distribution
#' Given training (X, y), predict at X_test
#' Returns: mean prediction and variance (uncertainty)
gp_predict <- function(X_train, y_train, X_test,
                        l = 1.0, sigma_f = 1.0, sigma_n = 0.1) {
  n_train <- nrow(X_train)
  n_test  <- nrow(X_test)

  # Kernel matrices
  K_tt <- rbf_kernel(X_train, X_train, l, sigma_f)
  K_ss <- rbf_kernel(X_test,  X_test,  l, sigma_f)
  K_ts <- rbf_kernel(X_train, X_test,  l, sigma_f)

  # Add noise to diagonal: K_y = K_tt + sigma_n^2 * I
  K_y <- K_tt + diag(sigma_n^2, n_train)

  # Cholesky for stable inversion
  L <- tryCatch(t(chol(K_y + diag(1e-8, n_train))),
                error = function(e) chol(K_y + diag(1e-6, n_train)))

  # Posterior mean: mu* = K_ts' * K_y^-1 * y
  # Posterior cov:  C*  = K_ss - K_ts' * K_y^-1 * K_ts
  alpha_vec <- backsolve(t(L), forwardsolve(L, y_train))
  mu_pred <- as.vector(t(K_ts) %*% alpha_vec)

  v <- forwardsolve(L, K_ts)
  cov_pred <- K_ss - t(v) %*% v

  list(
    mean = mu_pred,
    variance = diag(cov_pred),
    covariance = cov_pred,
    lower95 = mu_pred - 1.96 * sqrt(pmax(0, diag(cov_pred))),
    upper95 = mu_pred + 1.96 * sqrt(pmax(0, diag(cov_pred)))
  )
}

#' Log marginal likelihood for GP hyperparameter optimization
gp_log_marginal_likelihood <- function(X, y, l, sigma_f, sigma_n) {
  n <- nrow(X)
  K <- rbf_kernel(X, X, l, sigma_f) + diag(sigma_n^2 + 1e-8, n)
  L <- tryCatch(t(chol(K)), error = function(e) return(NULL))
  if (is.null(L)) return(-Inf)

  alpha_v <- backsolve(t(L), forwardsolve(L, y))
  lml <- -0.5 * sum(y * alpha_v) - sum(log(diag(L))) - n/2 * log(2*pi)
  lml
}

#' Grid search for GP hyperparameters
gp_tune <- function(X, y, l_grid = c(0.1, 0.5, 1, 2, 5),
                    sigma_f_grid = c(0.5, 1, 2),
                    sigma_n_grid = c(0.01, 0.1, 0.5)) {
  best_lml <- -Inf; best_params <- NULL
  for (l in l_grid) {
    for (sf in sigma_f_grid) {
      for (sn in sigma_n_grid) {
        lml <- gp_log_marginal_likelihood(X, y, l, sf, sn)
        if (lml > best_lml) {
          best_lml <- lml
          best_params <- c(l=l, sigma_f=sf, sigma_n=sn)
        }
      }
    }
  }
  list(params=best_params, log_marginal_likelihood=best_lml)
}

# ---------------------------------------------------------------------------
# 3. 3-LAYER MLP WITH ADAM OPTIMIZER (FROM SCRATCH)
# ---------------------------------------------------------------------------

# Activation functions
relu    <- function(x) pmax(0, x)
relu_d  <- function(x) (x > 0) * 1.0
sigmoid <- function(x) 1 / (1 + exp(-pmax(-500, pmin(500, x))))
sigmoid_d <- function(x) { s <- sigmoid(x); s * (1 - s) }
tanh_d  <- function(x) 1 - tanh(x)^2

#' Initialize MLP weights (He initialization for ReLU)
mlp_init <- function(input_dim, hidden1, hidden2, output_dim, seed = 42) {
  set.seed(seed)
  list(
    W1 = matrix(rnorm(input_dim * hidden1, 0, sqrt(2/input_dim)),   input_dim, hidden1),
    b1 = rep(0, hidden1),
    W2 = matrix(rnorm(hidden1  * hidden2, 0, sqrt(2/hidden1)),    hidden1, hidden2),
    b2 = rep(0, hidden2),
    W3 = matrix(rnorm(hidden2  * output_dim, 0, sqrt(2/hidden2)), hidden2, output_dim),
    b3 = rep(0, output_dim)
  )
}

#' Forward pass through 3-layer MLP
mlp_forward <- function(X, params) {
  # Layer 1: ReLU
  Z1 <- sweep(X %*% params$W1, 2, params$b1, "+")
  A1 <- relu(Z1)

  # Layer 2: ReLU
  Z2 <- sweep(A1 %*% params$W2, 2, params$b2, "+")
  A2 <- relu(Z2)

  # Layer 3: Linear (regression) or sigmoid (classification)
  Z3 <- sweep(A2 %*% params$W3, 2, params$b3, "+")
  A3 <- Z3  # Linear output for regression

  list(Z1=Z1, A1=A1, Z2=Z2, A2=A2, Z3=Z3, A3=A3)
}

#' MSE loss and its gradient w.r.t. output
mse_loss <- function(y_pred, y_true) {
  n <- length(y_true)
  loss <- mean((y_pred - y_true)^2)
  grad <- 2 * (y_pred - y_true) / n
  list(loss=loss, grad=grad)
}

#' Backpropagation
mlp_backward <- function(X, y, params, cache) {
  n <- nrow(X)

  # Output layer gradient
  lg <- mse_loss(cache$A3, y)
  dZ3 <- matrix(lg$grad, ncol=ncol(cache$A3))

  dW3 <- t(cache$A2) %*% dZ3 / n
  db3 <- colMeans(dZ3)

  # Layer 2 backprop
  dA2 <- dZ3 %*% t(params$W3)
  dZ2 <- dA2 * relu_d(cache$Z2)
  dW2 <- t(cache$A1) %*% dZ2 / n
  db2 <- colMeans(dZ2)

  # Layer 1 backprop
  dA1 <- dZ2 %*% t(params$W2)
  dZ1 <- dA1 * relu_d(cache$Z1)
  dW1 <- t(X) %*% dZ1 / n
  db1 <- colMeans(dZ1)

  list(dW1=dW1, db1=db1, dW2=dW2, db2=db2, dW3=dW3, db3=db3, loss=lg$loss)
}

#' Adam optimizer update step
adam_update <- function(params, grads, m, v, t, lr=0.001, beta1=0.9,
                         beta2=0.999, eps=1e-8) {
  param_names <- c("W1","b1","W2","b2","W3","b3")
  grad_names  <- c("dW1","db1","dW2","db2","dW3","db3")

  for (k in seq_along(param_names)) {
    pn <- param_names[k]; gn <- grad_names[k]
    g <- grads[[gn]]
    m[[pn]] <- beta1 * m[[pn]] + (1 - beta1) * g
    v[[pn]] <- beta2 * v[[pn]] + (1 - beta2) * g^2
    m_hat <- m[[pn]] / (1 - beta1^t)
    v_hat <- v[[pn]] / (1 - beta2^t)
    params[[pn]] <- params[[pn]] - lr * m_hat / (sqrt(v_hat) + eps)
  }
  list(params=params, m=m, v=v)
}

#' Train MLP with Adam optimizer
mlp_train <- function(X, y, hidden1 = 64, hidden2 = 32,
                       lr = 0.001, epochs = 200, batch_size = 64,
                       seed = 42) {
  n <- nrow(X); p <- ncol(X)
  if (is.vector(y)) y <- matrix(y, ncol=1)

  params <- mlp_init(p, hidden1, hidden2, ncol(y), seed)

  # Initialize Adam moments (matching structure)
  init_like <- function(params) lapply(params, function(p) p * 0)
  m <- init_like(params); v <- init_like(params)

  loss_history <- numeric(epochs)
  t <- 0

  for (epoch in seq_len(epochs)) {
    # Shuffle mini-batches
    idx <- sample(n)
    epoch_loss <- 0; n_batches <- 0

    for (start in seq(1, n, by=batch_size)) {
      end <- min(start + batch_size - 1, n)
      batch_idx <- idx[start:end]
      X_batch <- X[batch_idx, , drop=FALSE]
      y_batch <- y[batch_idx, , drop=FALSE]

      t <- t + 1
      cache  <- mlp_forward(X_batch, params)
      grads  <- mlp_backward(X_batch, y_batch, params, cache)
      upd    <- adam_update(params, grads, m, v, t, lr)
      params <- upd$params; m <- upd$m; v <- upd$v

      epoch_loss <- epoch_loss + grads$loss
      n_batches  <- n_batches + 1
    }

    loss_history[epoch] <- epoch_loss / n_batches
  }

  predict_fn <- function(Xnew) {
    if (is.vector(Xnew)) Xnew <- matrix(Xnew, nrow=1)
    mlp_forward(Xnew, params)$A3
  }

  list(
    params = params, loss_history = loss_history,
    final_loss = tail(loss_history, 1),
    predict = predict_fn
  )
}

# ---------------------------------------------------------------------------
# 4. ATTENTION MECHANISM: SCALED DOT-PRODUCT
# ---------------------------------------------------------------------------

#' Scaled dot-product attention
#' Attention(Q, K, V) = softmax(Q K' / sqrt(d_k)) V
#' Q, K, V are matrices [seq_len x d_k]
scaled_dot_product_attention <- function(Q, K, V, mask = NULL) {
  d_k <- ncol(K)

  # Scores = Q * K^T / sqrt(d_k)
  scores <- Q %*% t(K) / sqrt(d_k)

  # Optional causal mask (for time series: can't attend to future)
  if (!is.null(mask)) {
    scores[mask] <- -1e9
  }

  # Softmax over keys dimension (per query row)
  softmax_rows <- function(M) {
    t(apply(M, 1, function(row) {
      row <- row - max(row)  # Numerical stability
      exp_row <- exp(row)
      exp_row / sum(exp_row)
    }))
  }

  attn_weights <- softmax_rows(scores)

  # Context = weighted sum of values
  context <- attn_weights %*% V

  list(context = context, attention_weights = attn_weights)
}

#' Multi-head attention: concatenate multiple attention heads
multi_head_attention <- function(X, n_heads = 4, d_model = NULL) {
  seq_len <- nrow(X)
  d_input <- ncol(X)
  if (is.null(d_model)) d_model <- d_input
  d_k <- d_model %/% n_heads

  # For simplicity, use learnable projections (initialized randomly)
  set.seed(1)
  heads <- lapply(seq_len(n_heads), function(h) {
    Wq <- matrix(rnorm(d_input * d_k, 0, 0.1), d_input, d_k)
    Wk <- matrix(rnorm(d_input * d_k, 0, 0.1), d_input, d_k)
    Wv <- matrix(rnorm(d_input * d_k, 0, 0.1), d_input, d_k)
    Q <- X %*% Wq; K <- X %*% Wk; V <- X %*% Wv
    scaled_dot_product_attention(Q, K, V)
  })

  # Concatenate head outputs
  context_all <- do.call(cbind, lapply(heads, function(h) h$context))
  attn_all    <- lapply(heads, function(h) h$attention_weights)

  list(context = context_all, attention_weights = attn_all)
}

#' Self-attention for time series feature extraction
#' Treats each time step as a token; learns which past steps matter most
time_series_attention <- function(ts_matrix, n_heads = 2) {
  # ts_matrix: [T x features]
  if (is.vector(ts_matrix)) ts_matrix <- matrix(ts_matrix, ncol=1)
  T_ <- nrow(ts_matrix); d <- ncol(ts_matrix)

  # Causal mask: position i can only attend to positions <= i
  mask <- outer(seq_len(T_), seq_len(T_), function(i, j) j > i)

  # Single-head attention with causal mask
  d_k <- max(1, d)
  Wq <- matrix(rnorm(d * d_k, 0, 0.1), d, d_k)
  Wk <- matrix(rnorm(d * d_k, 0, 0.1), d, d_k)
  Wv <- matrix(rnorm(d * d_k, 0, 0.1), d, d_k)

  Q <- ts_matrix %*% Wq
  K <- ts_matrix %*% Wk
  V <- ts_matrix %*% Wv

  result <- scaled_dot_product_attention(Q, K, V, mask = mask)

  # Return enriched representation + attention pattern
  list(
    enriched_features = result$context,
    attention_weights = result$attention_weights,
    most_attended_lags = apply(result$attention_weights, 1, which.max)
  )
}

# ---------------------------------------------------------------------------
# 5. ENSEMBLE STACKING WITH OUT-OF-FOLD PREDICTIONS
# ---------------------------------------------------------------------------

#' Generate out-of-fold (OOF) predictions for a base model
#' Uses time-series aware CV (expanding window)
oof_predictions_ts <- function(X, y, model_fn, n_folds = 5) {
  n <- nrow(X)
  # Time-series CV: train on first k/(n_folds+1)*n points, predict next block
  fold_size <- n %/% (n_folds + 1)
  oof_preds <- rep(NA, n)

  for (k in seq_len(n_folds)) {
    train_end  <- k * fold_size
    test_start <- train_end + 1
    test_end   <- min(test_start + fold_size - 1, n)

    if (test_start > n) break

    X_train <- X[seq_len(train_end), , drop=FALSE]
    y_train <- y[seq_len(train_end)]
    X_test  <- X[test_start:test_end, , drop=FALSE]

    model <- model_fn(X_train, y_train)
    preds <- model$predict(X_test)
    oof_preds[test_start:test_end] <- as.vector(preds)
  }
  oof_preds
}

#' Ridge regression meta-learner for stacking
ridge_meta_learner <- function(oof_matrix, y, lambda = 0.01) {
  # Remove rows where any OOF pred is NA
  valid <- complete.cases(oof_matrix) & !is.na(y)
  X_meta <- cbind(1, oof_matrix[valid, , drop=FALSE])
  y_meta <- y[valid]

  XtX <- crossprod(X_meta)
  Xty <- crossprod(X_meta, y_meta)
  beta <- solve(XtX + lambda * diag(ncol(X_meta)), Xty)

  predict_fn <- function(oof_new) {
    cbind(1, oof_new) %*% beta
  }

  list(
    beta = beta,
    predict = predict_fn,
    weights = beta[-1] / sum(abs(beta[-1]))  # Normalized model weights
  )
}

#' Full stacking ensemble
#' base_model_fns: named list of model-fitting functions
stacking_ensemble <- function(X, y, base_model_fns, lambda_meta = 0.01,
                               n_folds = 5) {
  n_models <- length(base_model_fns)
  model_names <- names(base_model_fns)

  # Generate OOF predictions for each base model
  oof_matrix <- matrix(NA, length(y), n_models)
  colnames(oof_matrix) <- model_names

  cat("Generating OOF predictions...\n")
  for (i in seq_len(n_models)) {
    cat(sprintf("  Model %d/%d: %s\n", i, n_models, model_names[i]))
    oof_matrix[, i] <- oof_predictions_ts(X, y, base_model_fns[[i]], n_folds)
  }

  # Train meta-learner on OOF predictions
  meta <- ridge_meta_learner(oof_matrix, y, lambda_meta)

  cat(sprintf("Meta-learner weights: %s\n",
              paste(sprintf("%s=%.3f", model_names, meta$weights), collapse=", ")))

  # Train final base models on full data
  final_models <- lapply(base_model_fns, function(fn) fn(X, y))

  predict_fn <- function(Xnew) {
    base_preds <- sapply(final_models, function(m) as.vector(m$predict(Xnew)))
    if (is.vector(base_preds)) base_preds <- matrix(base_preds, nrow=1)
    meta$predict(base_preds)
  }

  list(
    oof_matrix   = oof_matrix,
    meta_weights = meta$weights,
    final_models = final_models,
    predict      = predict_fn
  )
}

# ---------------------------------------------------------------------------
# 6. HYPERPARAMETER TUNING: RANDOM SEARCH WITH TIME-SERIES CV
# ---------------------------------------------------------------------------

#' Random search hyperparameter optimization
#' param_space: named list of vectors of values to sample from
#' model_fn: function(X_train, y_train, params) -> model with $predict
#' score_fn: function(y_pred, y_true) -> scalar (higher = better)
random_search_cv <- function(X, y, model_fn_factory, param_space,
                              n_iter = 20, n_folds = 3, score_fn = NULL,
                              seed = 42) {
  set.seed(seed)
  if (is.null(score_fn)) {
    score_fn <- function(pred, true) -mean((pred - true)^2, na.rm=TRUE)
  }

  n <- nrow(X)
  fold_size <- n %/% (n_folds + 1)

  results <- vector("list", n_iter)

  for (iter in seq_len(n_iter)) {
    # Sample random parameter combination
    params <- lapply(param_space, function(vals) sample(vals, 1))

    # Time-series CV
    scores <- numeric(n_folds)
    for (k in seq_len(n_folds)) {
      train_end  <- k * fold_size
      test_start <- train_end + 1
      test_end   <- min(test_start + fold_size - 1, n)
      if (test_start > n) { scores[k] <- NA; next }

      X_tr <- X[seq_len(train_end), , drop=FALSE]
      y_tr <- y[seq_len(train_end)]
      X_te <- X[test_start:test_end, , drop=FALSE]
      y_te <- y[test_start:test_end]

      model <- model_fn_factory(params)(X_tr, y_tr)
      preds <- as.vector(model$predict(X_te))
      scores[k] <- score_fn(preds, y_te)
    }

    results[[iter]] <- list(
      params = params,
      mean_score = mean(scores, na.rm=TRUE),
      sd_score   = sd(scores, na.rm=TRUE)
    )
  }

  scores_vec <- sapply(results, function(r) r$mean_score)
  best_idx <- which.max(scores_vec)

  list(
    best_params = results[[best_idx]]$params,
    best_score  = results[[best_idx]]$mean_score,
    all_results = results,
    scores      = scores_vec
  )
}

# ---------------------------------------------------------------------------
# 7. SHAP VALUES (TREESHAP CONCEPT FOR TREE ENSEMBLES)
# ---------------------------------------------------------------------------

#' Permutation-based SHAP approximation (model-agnostic)
#' Exact Shapley: average marginal contribution over all permutations
#' Approximation: sample n_perm random permutations
compute_shap <- function(predict_fn, X, x_instance, n_perm = 100,
                          baseline = NULL) {
  n_features <- ncol(X)
  if (is.null(baseline)) baseline <- colMeans(X)

  shap_values <- numeric(n_features)

  for (perm_i in seq_len(n_perm)) {
    feature_order <- sample(n_features)
    x_prev <- baseline

    for (pos in seq_len(n_features)) {
      f <- feature_order[pos]
      x_curr <- x_prev
      x_curr[f] <- x_instance[f]

      # Marginal contribution of feature f
      pred_with    <- predict_fn(matrix(x_curr, nrow=1))
      pred_without <- predict_fn(matrix(x_prev, nrow=1))
      shap_values[f] <- shap_values[f] + (pred_with - pred_without)

      x_prev <- x_curr
    }
  }

  shap_values <- shap_values / n_perm

  data.frame(
    feature    = paste0("X", seq_len(n_features)),
    shap_value = shap_values,
    abs_shap   = abs(shap_values)
  )[order(-abs(shap_values)), ]
}

# ---------------------------------------------------------------------------
# 8. CALIBRATION: PLATT SCALING AND ISOTONIC REGRESSION
# ---------------------------------------------------------------------------

#' Platt scaling: fit sigmoid to raw scores to get calibrated probabilities
#' P(y=1 | f) = 1 / (1 + exp(A*f + B))
platt_scaling <- function(scores, labels) {
  # Maximum likelihood optimization via Newton's method
  A <- 0; B <- log((sum(labels == -1) + 1) / (sum(labels == 1) + 1))

  for (iter in seq_len(100)) {
    fapb <- A * scores + B
    p    <- sigmoid(fapb)

    # Gradient
    gA <- sum(scores * (p - (labels == 1)))
    gB <- sum(p - (labels == 1))

    # Hessian (diagonal elements)
    hAA <- sum(scores^2 * p * (1-p))
    hBB <- sum(p * (1-p))
    hAB <- sum(scores * p * (1-p))

    det <- hAA * hBB - hAB^2
    if (abs(det) < 1e-12) break

    A_new <- A - (hBB * gA - hAB * gB) / det
    B_new <- B - (hAA * gB - hAB * gA) / det

    if (max(abs(c(A_new - A, B_new - B))) < 1e-8) break
    A <- A_new; B <- B_new
  }

  predict_fn <- function(new_scores) sigmoid(A * new_scores + B)

  list(A = A, B = B, predict = predict_fn)
}

#' Isotonic regression: non-decreasing calibration
#' Pool adjacent violators algorithm (PAVA)
isotonic_regression <- function(y, weights = NULL) {
  n <- length(y)
  if (is.null(weights)) weights <- rep(1, n)

  # Pool adjacent violators
  blocks <- as.list(seq_len(n))  # Each point is a block
  block_means <- y
  block_weights <- weights

  changed <- TRUE
  while (changed) {
    changed <- FALSE
    i <- 1
    while (i < length(block_means)) {
      if (block_means[i] > block_means[i + 1]) {
        # Merge blocks i and i+1
        w_new <- block_weights[i] + block_weights[i + 1]
        m_new <- (block_weights[i] * block_means[i] +
                  block_weights[i+1] * block_means[i+1]) / w_new
        block_means[i] <- m_new
        block_weights[i] <- w_new
        block_means <- block_means[-(i+1)]
        block_weights <- block_weights[-(i+1)]
        changed <- TRUE
      } else {
        i <- i + 1
      }
    }
  }

  # Expand back to original length
  result <- numeric(n)
  pos <- 1
  for (b in seq_along(block_means)) {
    block_len <- round(block_weights[b] / weights[1])
    if (pos + block_len - 1 <= n) {
      result[pos:(pos + block_len - 1)] <- block_means[b]
    }
    pos <- pos + block_len
  }
  # Fill any remaining with last value
  if (pos <= n) result[pos:n] <- tail(block_means, 1)

  result
}

#' Calibrate probability predictions using isotonic regression
calibrate_isotonic <- function(raw_probs, true_labels, n_bins = 10) {
  # Sort by raw probability
  ord <- order(raw_probs)
  sorted_probs  <- raw_probs[ord]
  sorted_labels <- true_labels[ord]

  # Fit isotonic regression: E[y | score] should be non-decreasing
  fitted <- isotonic_regression(sorted_labels)

  # Calibration curve (reliability diagram)
  bins <- cut(raw_probs, breaks = seq(0, 1, length.out = n_bins + 1),
              include.lowest = TRUE)
  cal_df <- data.frame(
    bin = levels(bins),
    mean_pred = tapply(raw_probs, bins, mean, na.rm=TRUE),
    mean_obs  = tapply(true_labels, bins, mean, na.rm=TRUE)
  )

  # Expected calibration error
  cal_df <- cal_df[complete.cases(cal_df), ]
  n_in_bins <- as.vector(table(bins)[!is.na(table(bins))])
  ece <- sum(abs(cal_df$mean_pred - cal_df$mean_obs) *
               n_in_bins / length(raw_probs), na.rm=TRUE)

  list(
    calibration_curve = cal_df,
    ece = ece,
    fitted_values = fitted[order(ord)],
    well_calibrated = ece < 0.05
  )
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  set.seed(1234)
  n <- 300; p <- 5

  # Generate classification data
  X <- matrix(rnorm(n*p), n, p)
  y_class <- sign(X[,1] - X[,2] + 0.5*X[,3] + rnorm(n, 0, 0.5))

  # SVM
  cat("=== SVM (linear kernel) ===\n")
  svm_fit <- svm_smo(X, y_class, C = 1.0, max_passes = 30)
  preds <- svm_fit$predict(X)
  cat(sprintf("Training accuracy: %.1f%%\n", mean(preds == y_class)*100))
  cat(sprintf("Support vectors: %d/%d\n", svm_fit$n_sv, n))

  # GP regression
  cat("\n=== Gaussian Process Regression ===\n")
  y_reg <- X[,1]^2 + sin(X[,2]) + rnorm(n, 0, 0.1)
  Xmat <- X[, 1:2, drop=FALSE]
  train_idx <- 1:200
  gp_fit <- gp_predict(Xmat[train_idx,], y_reg[train_idx],
                        Xmat[201:250,], l=1, sigma_f=2, sigma_n=0.1)
  rmse <- sqrt(mean((gp_fit$mean - y_reg[201:250])^2))
  cat(sprintf("OOS RMSE: %.4f\n", rmse))

  # MLP
  cat("\n=== 3-Layer MLP (Adam) ===\n")
  mlp_fit <- mlp_train(X[1:200,], y_reg[1:200], hidden1=32, hidden2=16,
                        lr=0.001, epochs=100)
  cat(sprintf("Final training loss: %.4f\n", mlp_fit$final_loss))

  # Attention
  cat("\n=== Self-Attention on Time Series ===\n")
  ts_data <- matrix(cumsum(rnorm(100*3)), 100, 3)
  attn_result <- time_series_attention(ts_data)
  cat(sprintf("Most attended lag (last 5 steps): %s\n",
              paste(tail(attn_result$most_attended_lags, 5), collapse=",")))
}
