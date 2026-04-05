# =============================================================================
# ml_backtesting.R
# ML Signal Backtesting: walk-forward backtest of GP, SVM, neural net, RF,
# GBM; IS/OOS Sharpe comparison; feature stability; model combination;
# overfitting detection via IS-OOS degradation; deflated Sharpe ratio test.
# Pure base R.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING FOR CRYPTO SIGNALS
# ---------------------------------------------------------------------------

#' Build feature matrix from raw price series
build_features <- function(prices, volumes = NULL, window_short = 5,
                            window_medium = 20, window_long = 60) {
  n <- length(prices)
  returns <- c(NA, diff(log(prices)))

  features <- data.frame(
    # Momentum features
    mom_5d   = c(rep(NA, window_short),
                 sapply((window_short+1):n, function(i) sum(returns[(i-window_short+1):i]))),
    mom_20d  = c(rep(NA, window_medium),
                 sapply((window_medium+1):n, function(i) sum(returns[(i-window_medium+1):i]))),
    mom_60d  = c(rep(NA, window_long),
                 sapply((window_long+1):n, function(i) sum(returns[(i-window_long+1):i]))),

    # Volatility features
    vol_5d   = c(rep(NA, window_short),
                 sapply((window_short+1):n, function(i) sd(returns[(i-window_short+1):i]))),
    vol_20d  = c(rep(NA, window_medium),
                 sapply((window_medium+1):n, function(i) sd(returns[(i-window_medium+1):i]))),

    # Mean reversion
    zscore_20d = NA_real_,
    zscore_60d = NA_real_,

    # Trend strength
    trend_ratio = NA_real_
  )

  # Z-scores
  for (i in (window_medium+1):n) {
    idx <- (i-window_medium+1):i
    r_window <- returns[idx]; r_window <- r_window[!is.na(r_window)]
    if (length(r_window) > 3) {
      features$zscore_20d[i] <- (returns[i] - mean(r_window)) / (sd(r_window) + 1e-8)
    }
  }
  for (i in (window_long+1):n) {
    idx <- (i-window_long+1):i
    r_window <- returns[idx]; r_window <- r_window[!is.na(r_window)]
    if (length(r_window) > 5) {
      features$zscore_60d[i] <- (returns[i] - mean(r_window)) / (sd(r_window) + 1e-8)
    }
  }

  # Trend ratio: short vol / long vol
  features$trend_ratio <- features$vol_5d / (features$vol_20d + 1e-8)

  # Add volume features if provided
  if (!is.null(volumes)) {
    vol_ma <- c(rep(NA, window_medium),
                sapply((window_medium+1):n, function(i) mean(volumes[(i-window_medium+1):i])))
    features$vol_ratio <- volumes / (vol_ma + 1)
    features$vol_ratio[is.infinite(features$vol_ratio)] <- 1
  }

  # Add forward return (target)
  features$fwd_ret_1d <- c(returns[-1], NA)

  features$t <- seq_len(n)
  features
}

# ---------------------------------------------------------------------------
# 2. SIMPLE ML MODELS FROM SCRATCH
# ---------------------------------------------------------------------------

#' Random Forest (regression trees with bagging)
#' Pure R implementation: simplified CART trees
build_tree <- function(X, y, max_depth = 5, min_samples = 10) {
  if (length(y) < min_samples * 2 || max_depth == 0) {
    return(list(leaf = TRUE, value = mean(y, na.rm=TRUE)))
  }

  n <- nrow(X); p <- ncol(X)
  best_split <- NULL; best_loss <- Inf

  # Try random subset of features (sqrt(p) features)
  n_try <- max(1, round(sqrt(p)))
  feature_idx <- sample(p, n_try)

  for (j in feature_idx) {
    x_j <- X[, j]
    valid <- !is.na(x_j) & !is.na(y)
    if (sum(valid) < min_samples * 2) next

    sorted_vals <- sort(unique(x_j[valid]))
    if (length(sorted_vals) < 2) next

    # Try quantile splits
    thresholds <- quantile(x_j[valid], probs = c(0.25, 0.5, 0.75), na.rm=TRUE)

    for (thresh in thresholds) {
      left  <- valid & x_j <= thresh
      right <- valid & x_j > thresh
      if (sum(left) < min_samples || sum(right) < min_samples) next

      loss <- sum(left) * var(y[left]) + sum(right) * var(y[right])
      if (loss < best_loss) {
        best_loss <- loss
        best_split <- list(feature = j, threshold = thresh)
      }
    }
  }

  if (is.null(best_split)) {
    return(list(leaf = TRUE, value = mean(y, na.rm=TRUE)))
  }

  left_mask  <- !is.na(X[, best_split$feature]) & X[, best_split$feature] <= best_split$threshold
  right_mask <- !left_mask

  list(
    leaf = FALSE,
    feature = best_split$feature,
    threshold = best_split$threshold,
    left  = build_tree(X[left_mask,  , drop=FALSE], y[left_mask],  max_depth-1, min_samples),
    right = build_tree(X[right_mask, , drop=FALSE], y[right_mask], max_depth-1, min_samples)
  )
}

predict_tree <- function(tree, X) {
  if (is.vector(X)) X <- matrix(X, nrow=1)
  apply(X, 1, function(x) {
    node <- tree
    while (!node$leaf) {
      val <- x[node$feature]
      if (is.na(val) || val <= node$threshold) node <- node$left
      else node <- node$right
    }
    node$value
  })
}

#' Random Forest
random_forest <- function(X, y, n_trees = 50, max_depth = 5,
                           min_samples = 10, seed = 42) {
  set.seed(seed)
  n <- nrow(X)
  trees <- vector("list", n_trees)

  for (t in seq_len(n_trees)) {
    # Bootstrap sample
    boot_idx <- sample(n, n, replace = TRUE)
    trees[[t]] <- build_tree(X[boot_idx, , drop=FALSE], y[boot_idx],
                              max_depth, min_samples)
  }

  predict_fn <- function(Xnew) {
    if (is.vector(Xnew)) Xnew <- matrix(Xnew, nrow=1)
    preds <- sapply(trees, function(tr) predict_tree(tr, Xnew))
    if (nrow(Xnew) == 1) mean(preds) else rowMeans(preds)
  }

  list(trees = trees, n_trees = n_trees, predict = predict_fn)
}

#' Gradient Boosting (regression, simple implementation)
#' Uses shallow trees as weak learners
gradient_boost <- function(X, y, n_trees = 100, learning_rate = 0.1,
                            max_depth = 3, seed = 42) {
  set.seed(seed)
  n <- nrow(X)

  # Initialize: predict mean
  F_pred <- rep(mean(y, na.rm=TRUE), n)
  trees  <- vector("list", n_trees)
  init_pred <- mean(y, na.rm=TRUE)

  for (t in seq_len(n_trees)) {
    # Negative gradient = residuals for MSE loss
    residuals <- y - F_pred
    valid <- !is.na(residuals)

    # Fit tree to residuals
    tree <- build_tree(X[valid, , drop=FALSE], residuals[valid], max_depth, 5)
    trees[[t]] <- tree

    # Update predictions
    update <- predict_tree(tree, X)
    F_pred <- F_pred + learning_rate * update
  }

  predict_fn <- function(Xnew) {
    if (is.vector(Xnew)) Xnew <- matrix(Xnew, nrow=1)
    pred <- rep(init_pred, nrow(Xnew))
    for (t in seq_len(n_trees)) {
      pred <- pred + learning_rate * predict_tree(trees[[t]], Xnew)
    }
    pred
  }

  list(trees = trees, n_trees = n_trees, predict = predict_fn,
       init_pred = init_pred)
}

#' Linear ridge regression (strong baseline)
ridge_model <- function(X, y, lambda = 0.01) {
  if (is.vector(X)) X <- matrix(X, ncol=1)
  Xm <- cbind(1, X)
  beta <- tryCatch(
    solve(crossprod(Xm) + lambda * diag(ncol(Xm)), crossprod(Xm, y)),
    error = function(e) rep(0, ncol(Xm))
  )
  predict_fn <- function(Xnew) {
    if (is.vector(Xnew)) Xnew <- matrix(Xnew, nrow=1)
    as.vector(cbind(1, Xnew) %*% beta)
  }
  list(beta = beta, predict = predict_fn)
}

# ---------------------------------------------------------------------------
# 3. WALK-FORWARD BACKTEST ENGINE
# ---------------------------------------------------------------------------

#' Walk-forward backtest with expanding or rolling window
walk_forward_backtest <- function(features_df, model_fn, model_name,
                                   target_col = "fwd_ret_1d",
                                   feature_cols = NULL,
                                   train_window = 200,
                                   test_window  = 50,
                                   expanding    = FALSE,
                                   tc_bps       = 5,
                                   signal_threshold = 0) {
  n <- nrow(features_df)

  if (is.null(feature_cols)) {
    feature_cols <- setdiff(names(features_df), c(target_col, "t"))
  }

  # Remove target from features
  feature_cols <- setdiff(feature_cols, target_col)

  strat_returns <- rep(NA, n)
  predictions   <- rep(NA, n)
  is_sharpes    <- numeric(0)
  oos_sharpes   <- numeric(0)

  start_test <- train_window + 1

  while (start_test + test_window - 1 <= n) {
    end_test   <- start_test + test_window - 1
    if (expanding) {
      start_train <- 1
    } else {
      start_train <- start_test - train_window
    }
    end_train <- start_test - 1

    # Prepare training data
    train_idx <- start_train:end_train
    test_idx  <- start_test:end_test

    train_df <- features_df[train_idx, c(feature_cols, target_col), drop=FALSE]
    test_df  <- features_df[test_idx,  c(feature_cols, target_col), drop=FALSE]

    # Remove NAs
    valid_train <- complete.cases(train_df)
    if (sum(valid_train) < 30) {
      start_test <- start_test + test_window
      next
    }

    X_train <- as.matrix(train_df[valid_train, feature_cols, drop=FALSE])
    y_train <- train_df[[target_col]][valid_train]

    # Fit model
    model <- tryCatch(model_fn(X_train, y_train),
                       error = function(e) NULL)
    if (is.null(model)) {
      start_test <- start_test + test_window; next
    }

    # In-sample predictions for Sharpe calc
    is_preds <- tryCatch(as.vector(model$predict(X_train)), error = function(e) NULL)
    if (!is.null(is_preds)) {
      is_pos <- sign(is_preds - signal_threshold)
      is_ret <- is_pos * y_train - abs(diff(c(0, is_pos))) * tc_bps/10000
      is_sharpes <- c(is_sharpes, mean(is_ret, na.rm=TRUE) /
                        (sd(is_ret, na.rm=TRUE) + 1e-8) * sqrt(252))
    }

    # Out-of-sample predictions
    valid_test <- complete.cases(test_df)
    if (sum(valid_test) == 0) {
      start_test <- start_test + test_window; next
    }

    X_test <- as.matrix(test_df[valid_test, feature_cols, drop=FALSE])
    y_test <- test_df[[target_col]][valid_test]

    oos_preds <- tryCatch(as.vector(model$predict(X_test)), error = function(e) NULL)
    if (!is.null(oos_preds)) {
      predictions[test_idx[valid_test]] <- oos_preds
      pos  <- sign(oos_preds - signal_threshold)
      cost <- abs(c(0, diff(pos))) * tc_bps/10000
      rets <- pos * y_test - cost[-length(cost)]
      strat_returns[test_idx[valid_test]] <- rets

      oos_sharpes <- c(oos_sharpes, mean(rets) / (sd(rets) + 1e-8) * sqrt(252))
    }

    start_test <- start_test + test_window
  }

  # Performance metrics
  oos_rets <- strat_returns[!is.na(strat_returns)]
  cum_ret  <- if (length(oos_rets) > 0) prod(1 + oos_rets) - 1 else NA
  sharpe   <- if (length(oos_rets) > 5) mean(oos_rets) / (sd(oos_rets) + 1e-8) * sqrt(252) else NA
  max_dd   <- if (length(oos_rets) > 0) {
    eq <- cumprod(1 + oos_rets); min(eq / cummax(eq) - 1)
  } else NA

  list(
    model_name   = model_name,
    returns      = strat_returns,
    predictions  = predictions,
    sharpe_oos   = sharpe,
    sharpe_is    = mean(is_sharpes, na.rm=TRUE),
    cum_ret_pct  = cum_ret * 100,
    max_dd_pct   = max_dd * 100,
    n_oos_obs    = sum(!is.na(strat_returns))
  )
}

# ---------------------------------------------------------------------------
# 4. IS/OOS SHARPE COMPARISON
# ---------------------------------------------------------------------------

#' Compare IS and OOS performance for all models
compare_is_oos_sharpe <- function(results_list) {
  df <- lapply(results_list, function(r) {
    data.frame(
      model      = r$model_name,
      sharpe_is  = r$sharpe_is,
      sharpe_oos = r$sharpe_oos,
      cum_ret    = r$cum_ret_pct,
      max_dd     = r$max_dd_pct,
      degradation = if (!is.na(r$sharpe_is) && r$sharpe_is != 0) {
        (r$sharpe_oos - r$sharpe_is) / abs(r$sharpe_is) * 100
      } else NA
    )
  })

  df <- do.call(rbind, df)

  cat("╔═══════════════════════════════════════════════════════════╗\n")
  cat("║              IS vs OOS Sharpe Comparison                   ║\n")
  cat("╚═══════════════════════════════════════════════════════════╝\n\n")
  cat(sprintf("%-20s | %8s | %8s | %8s | %9s | %8s\n",
              "Model", "IS Sharpe", "OOS Sharpe", "OOS CumRet", "MaxDD%", "Degrad%"))
  cat(paste(rep("-", 75), collapse=""), "\n")
  for (i in seq_len(nrow(df))) {
    cat(sprintf("%-20s | %8.3f | %10.3f | %10.2f | %9.2f | %8.1f\n",
                df$model[i], df$sharpe_is[i], df$sharpe_oos[i],
                df$cum_ret[i], df$max_dd[i],
                if (!is.na(df$degradation[i])) df$degradation[i] else 0))
  }

  invisible(df)
}

# ---------------------------------------------------------------------------
# 5. FEATURE STABILITY ANALYSIS
# ---------------------------------------------------------------------------

#' Assess feature importance stability across time periods
feature_stability_analysis <- function(features_df, target_col = "fwd_ret_1d",
                                        feature_cols = NULL, n_periods = 5) {
  n <- nrow(features_df)
  if (is.null(feature_cols)) {
    feature_cols <- setdiff(names(features_df), c(target_col, "t"))
  }

  period_size <- n %/% n_periods

  # For each period, compute IC of each feature
  ic_matrix <- matrix(NA, n_periods, length(feature_cols))
  colnames(ic_matrix) <- feature_cols

  for (k in seq_len(n_periods)) {
    start <- (k-1) * period_size + 1
    end   <- min(k * period_size, n)
    sub   <- features_df[start:end, ]

    for (j in seq_along(feature_cols)) {
      fc <- feature_cols[j]
      valid <- !is.na(sub[[fc]]) & !is.na(sub[[target_col]])
      if (sum(valid) < 10) next
      ic_matrix[k, j] <- cor(sub[[fc]][valid], sub[[target_col]][valid],
                               method = "spearman")
    }
  }

  # Stability: coefficient of variation of IC across periods
  stability <- apply(ic_matrix, 2, function(x) {
    valid <- !is.na(x)
    if (sum(valid) < 2) return(NA)
    abs(mean(x[valid])) / (sd(x[valid]) + 1e-8)
  })

  df <- data.frame(
    feature   = feature_cols,
    mean_ic   = colMeans(ic_matrix, na.rm=TRUE),
    ic_sd     = apply(ic_matrix, 2, sd, na.rm=TRUE),
    stability = stability,
    stringsAsFactors = FALSE
  )

  cat("=== Feature Stability Analysis ===\n")
  print(df[order(-df$stability), ])
  cat("\n")

  invisible(list(ic_matrix = ic_matrix, stability_df = df))
}

# ---------------------------------------------------------------------------
# 6. OVERFITTING DETECTION: IS-OOS DEGRADATION
# ---------------------------------------------------------------------------

#' Degradation test: significant IS-OOS deterioration = overfitting
#' Bailey et al. (2014) Deflated Sharpe Ratio concept
is_oos_degradation_test <- function(is_returns, oos_returns) {
  is_sharpe  <- mean(is_returns, na.rm=TRUE) / (sd(is_returns, na.rm=TRUE) + 1e-8) * sqrt(252)
  oos_sharpe <- mean(oos_returns, na.rm=TRUE) / (sd(oos_returns, na.rm=TRUE) + 1e-8) * sqrt(252)

  degradation_pct <- (oos_sharpe - is_sharpe) / (abs(is_sharpe) + 1e-8) * 100

  # Test: H0: OOS Sharpe >= IS Sharpe (no overfitting)
  # T-test on return difference
  if (length(is_returns) > 5 && length(oos_returns) > 5) {
    # Pool and test
    t_stat <- (mean(oos_returns, na.rm=TRUE) - mean(is_returns, na.rm=TRUE)) /
              sqrt(var(is_returns, na.rm=TRUE) / length(is_returns) +
                   var(oos_returns, na.rm=TRUE) / length(oos_returns) + 1e-10)
    p_value <- pt(t_stat, df = min(length(is_returns), length(oos_returns)) - 1)
  } else {
    t_stat <- NA; p_value <- NA
  }

  list(
    is_sharpe    = is_sharpe,
    oos_sharpe   = oos_sharpe,
    degradation  = degradation_pct,
    t_stat       = t_stat,
    p_value      = p_value,
    overfitting  = !is.na(degradation_pct) && degradation_pct < -50  # >50% degradation
  )
}

# ---------------------------------------------------------------------------
# 7. DEFLATED SHARPE RATIO TEST
# ---------------------------------------------------------------------------
# Lopez de Prado & Bailey (2014): adjust Sharpe for number of trials
# DSR = SR * sqrt(T/n) / sigma_SR_approx
# Accounts for data-snooping bias in backtest selection

#' Probabilistic Sharpe Ratio (PSR)
probabilistic_sharpe_ratio <- function(returns, sr_benchmark = 0,
                                        skewness = NULL, kurtosis = NULL) {
  T_  <- sum(!is.na(returns))
  sr  <- mean(returns, na.rm=TRUE) / (sd(returns, na.rm=TRUE) + 1e-8) * sqrt(252)

  if (is.null(skewness)) {
    skewness <- mean(((returns - mean(returns, na.rm=TRUE)) / (sd(returns, na.rm=TRUE) + 1e-8))^3, na.rm=TRUE)
  }
  if (is.null(kurtosis)) {
    kurtosis <- mean(((returns - mean(returns, na.rm=TRUE)) / (sd(returns, na.rm=TRUE) + 1e-8))^4, na.rm=TRUE)
  }

  # Variance of SR estimate (Lo, 2002)
  sr_ann <- sr / sqrt(252)  # Daily SR
  sr_var <- (1 + 0.5 * sr_ann^2 - skewness * sr_ann + (kurtosis-3)/4 * sr_ann^2) / T_
  sr_se  <- sqrt(pmax(sr_var, 1e-10))

  psr <- pnorm((sr_ann - sr_benchmark/sqrt(252)) / sr_se)

  list(sharpe = sr, psr = psr, se = sr_se * sqrt(252), T = T_)
}

#' Deflated Sharpe Ratio: adjust for multiple testing
deflated_sharpe_ratio <- function(best_sharpe, n_trials, T_obs,
                                   skewness_vec = NULL, kurtosis_vec = NULL) {
  # Expected maximum SR under iid Gaussian after n_trials
  # E[max SR] ≈ (1 - gamma) * Z^{-1}(1 - 1/n) + gamma * Z^{-1}(1 - 1/(n*e))
  # where gamma = Euler-Mascheroni constant ≈ 0.5772
  gamma_EM <- 0.5772156649

  if (n_trials == 1) {
    expected_max <- 0
  } else {
    z1 <- qnorm(1 - 1/n_trials)
    z2 <- qnorm(1 - 1/(n_trials * exp(1)))
    expected_max <- (1 - gamma_EM) * z1 + gamma_EM * z2
  }

  # Scale to same time horizon
  expected_max_scaled <- expected_max * sqrt(252 / T_obs)

  # Compute variance of SR
  if (!is.null(skewness_vec)) {
    sk <- mean(skewness_vec)
    ku <- if (!is.null(kurtosis_vec)) mean(kurtosis_vec) else 3
  } else {
    sk <- 0; ku <- 3
  }

  sr_daily <- best_sharpe / sqrt(252)
  sr_var  <- (1 + 0.5 * sr_daily^2 - sk * sr_daily + (ku-3)/4 * sr_daily^2) / T_obs
  sr_se   <- sqrt(pmax(sr_var, 1e-10)) * sqrt(252)

  # DSR: probability that best_sharpe is above expected maximum under chance
  dsr <- pnorm((best_sharpe - expected_max_scaled) / sr_se)

  cat("=== Deflated Sharpe Ratio Test ===\n")
  cat(sprintf("Best IS Sharpe:         %.3f\n", best_sharpe))
  cat(sprintf("Number of trials:       %d\n", n_trials))
  cat(sprintf("Expected max under H0:  %.3f\n", expected_max_scaled))
  cat(sprintf("SR Standard Error:      %.3f\n", sr_se))
  cat(sprintf("Deflated SR (prob):     %.4f\n", dsr))
  cat(sprintf("Genuine alpha?          %s\n",
              if (dsr > 0.95) "YES (95% confident)" else "UNCERTAIN"))

  list(
    best_sharpe      = best_sharpe,
    expected_max     = expected_max_scaled,
    dsr              = dsr,
    genuine_alpha    = dsr > 0.95,
    sr_se            = sr_se
  )
}

# ---------------------------------------------------------------------------
# 8. MODEL COMBINATION VS BEST SINGLE MODEL
# ---------------------------------------------------------------------------

#' Simple equal-weight ensemble of model predictions
ensemble_predictions <- function(results_list) {
  # Extract prediction series from each model
  preds_list <- lapply(results_list, function(r) r$predictions)
  n <- max(sapply(preds_list, length))

  preds_mat <- matrix(NA, n, length(results_list))
  for (i in seq_along(preds_list)) {
    pv <- preds_list[[i]]
    if (length(pv) == n) preds_mat[, i] <- pv
  }

  # Equal-weight ensemble
  ensemble_pred <- rowMeans(preds_mat, na.rm=TRUE)

  # Also try best-on-IS selection
  is_sharpes <- sapply(results_list, function(r) r$sharpe_is)
  best_model_idx <- which.max(is_sharpes)
  best_pred <- preds_list[[best_model_idx]]

  list(ensemble = ensemble_pred, best_pred = best_pred,
       best_model = results_list[[best_model_idx]]$model_name)
}

# ---------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

if (FALSE) {
  set.seed(42)
  T_ <- 600

  # Generate synthetic price series
  prices  <- 50000 * exp(cumsum(rnorm(T_, 0.001, 0.03)))
  volumes <- rlnorm(T_, 20, 0.5)

  # Build features
  features_df <- build_features(prices, volumes)

  # Remove rows with any NA
  feature_cols <- c("mom_5d","mom_20d","mom_60d","vol_5d","vol_20d",
                     "zscore_20d","zscore_60d","trend_ratio")

  # Feature stability
  cat("=== Feature Stability ===\n")
  fs <- feature_stability_analysis(features_df, feature_cols = feature_cols)

  # Define model factories
  rf_fn <- function(X, y) random_forest(X, y, n_trees=20, max_depth=4, seed=1)
  gbm_fn <- function(X, y) gradient_boost(X, y, n_trees=30, learning_rate=0.05, max_depth=3)
  ridge_fn <- function(X, y) ridge_model(X, y, lambda=0.1)

  model_list <- list(
    list(fn = ridge_fn, name = "Ridge Regression"),
    list(fn = rf_fn,    name = "Random Forest"),
    list(fn = gbm_fn,   name = "Gradient Boost")
  )

  cat("\n=== Walk-Forward Backtests ===\n")
  results <- lapply(model_list, function(ml) {
    cat(sprintf("Running %s...\n", ml$name))
    walk_forward_backtest(
      features_df, ml$fn, ml$name,
      feature_cols = feature_cols,
      train_window = 150, test_window = 30,
      tc_bps = 5
    )
  })

  # IS/OOS Comparison
  compare_is_oos_sharpe(results)

  # Deflated Sharpe Ratio
  best_is_sharpe <- max(sapply(results, function(r) r$sharpe_is), na.rm=TRUE)
  deflated_sharpe_ratio(best_is_sharpe, n_trials = length(results), T_obs = 150)

  # Model combination
  ens <- ensemble_predictions(results)
  cat(sprintf("\nBest single model: %s\n", ens$best_model))

  # PSR for best model
  best_idx <- which.max(sapply(results, function(r) r$sharpe_oos))
  best_oos_rets <- results[[best_idx]]$returns
  best_oos_rets <- best_oos_rets[!is.na(best_oos_rets)]
  if (length(best_oos_rets) > 10) {
    psr <- probabilistic_sharpe_ratio(best_oos_rets)
    cat(sprintf("Best OOS Sharpe: %.3f | PSR: %.4f\n", psr$sharpe, psr$psr))
  }
}
