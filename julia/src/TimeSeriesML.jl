"""
TimeSeriesML — Machine learning methods for financial time series.

Pure Julia implementation (Statistics, LinearAlgebra, Random stdlib only).

Covers:
  - Gradient boosting from scratch (decision stumps, gradient updates)
  - Random forests for regime classification
  - Cross-validation for time series: purged K-fold, combinatorial purged
  - Feature importance via permutation
  - Online learning: passive-aggressive algorithm for non-stationary signals
  - Conformal prediction intervals for return forecasts
  - Model selection via BIC/AIC for ARIMA/GARCH family
  - Ensemble methods: bagging, boosting, stacking for financial signals
"""
module TimeSeriesML

using LinearAlgebra
using Statistics
using Random

export DecisionStump, fit_stump, predict_stump
export GradientBoostingRegressor, fit_gbm, predict_gbm, feature_importance_gbm
export RandomForestClassifier, fit_rf, predict_rf, predict_rf_proba
export PurgedKFold, combinatorial_purged_cv, walk_forward_cv
export permutation_importance, permutation_importance_ts
export PassiveAggressiveRegressor, pa_update!, predict_pa
export ConformalPredictor, fit_conformal, predict_interval
export ARIMA, fit_arima, forecast_arima
export GARCHSpec, fit_garch_aic, garch_model_selection
export BaggingEnsemble, fit_bagging, predict_bagging
export StackedEnsemble, fit_stacking, predict_stacking
export aic, bic

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Decision Stumps (Weak Learners for Boosting)
# ─────────────────────────────────────────────────────────────────────────────

"""
    DecisionStump

A single-feature, single-split decision stump (depth-1 tree).
The weakest possible learner; used as the base estimator in gradient boosting.
split_feature: column index, split_val: threshold, left_val/right_val: leaf predictions.
"""
struct DecisionStump
    split_feature::Int
    split_val::Float64
    left_val::Float64   # prediction when feature ≤ split_val
    right_val::Float64  # prediction when feature > split_val
    train_mse::Float64  # MSE on training data
end

"""
    fit_stump(X, r; min_leaf) -> DecisionStump

Fit a decision stump to pseudo-residuals r using feature matrix X.
Exhaustively searches all features and their sorted split points.
min_leaf: minimum number of samples in a leaf.
"""
function fit_stump(X::Matrix{Float64}, r::Vector{Float64};
                    min_leaf::Int=5)::DecisionStump
    n, p = size(X)
    best_mse = Inf
    best_feat = 1
    best_split = 0.0
    best_lval  = mean(r)
    best_rval  = mean(r)

    for j in 1:p
        sorted_idx = sortperm(X[:, j])
        x_sorted   = X[sorted_idx, j]
        r_sorted   = r[sorted_idx]

        # Incremental variance computation
        n_left = 0
        sum_left = 0.0
        sum2_left = 0.0
        sum_total = sum(r)
        sum2_total = sum(r.^2)

        for i in 1:(n - min_leaf)
            n_left    += 1
            sum_left  += r_sorted[i]
            sum2_left += r_sorted[i]^2

            n_right    = n - n_left
            n_left < min_leaf && continue
            n_right < min_leaf && continue

            # Avoid redundant splits (same value as next)
            i < n && x_sorted[i] == x_sorted[i+1] && continue

            mean_l = sum_left / n_left
            var_l  = sum2_left/n_left - mean_l^2

            sum_right  = sum_total - sum_left
            sum2_right = sum2_total - sum2_left
            mean_r     = sum_right / n_right
            var_r      = sum2_right/n_right - mean_r^2

            mse = (n_left * var_l + n_right * var_r) / n

            if mse < best_mse
                best_mse   = mse
                best_feat  = j
                best_split = (x_sorted[i] + x_sorted[min(i+1, n)]) / 2
                best_lval  = mean_l
                best_rval  = mean_r
            end
        end
    end

    return DecisionStump(best_feat, best_split, best_lval, best_rval, best_mse)
end

"""
    predict_stump(stump, X) -> Vector{Float64}

Generate predictions from a decision stump.
"""
function predict_stump(stump::DecisionStump, X::Matrix{Float64})::Vector{Float64}
    n = size(X, 1)
    preds = zeros(n)
    for i in 1:n
        preds[i] = X[i, stump.split_feature] <= stump.split_val ?
                   stump.left_val : stump.right_val
    end
    return preds
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Gradient Boosting
# ─────────────────────────────────────────────────────────────────────────────

"""
    GradientBoostingRegressor

Gradient boosting for regression using decision stumps as base learners.
Uses L2 (squared error) loss: residual_t = y - F_{t-1}(x).
F_t(x) = F_{t-1}(x) + learning_rate * h_t(x)
"""
struct GradientBoostingRegressor
    stumps::Vector{DecisionStump}
    n_estimators::Int
    learning_rate::Float64
    init_pred::Float64    # initial prediction (mean of y)
    subsample::Float64    # fraction of data for each tree
    n_features_used::Int
end

"""
    fit_gbm(X, y; n_estimators, lr, subsample, min_leaf, rng) -> GradientBoostingRegressor

Fit gradient boosting regressor.
n_estimators: number of boosting rounds.
lr: learning rate (shrinkage).
subsample: row subsampling fraction per round.
"""
function fit_gbm(X::Matrix{Float64}, y::Vector{Float64};
                  n_estimators::Int=100, lr::Float64=0.1,
                  subsample::Float64=0.8, min_leaf::Int=5,
                  feature_subsample::Float64=1.0,
                  rng::AbstractRNG=Random.default_rng())::GradientBoostingRegressor
    n, p = size(X)
    init_pred = mean(y)
    n_feat    = max(1, round(Int, feature_subsample * p))

    F = fill(init_pred, n)  # current ensemble prediction
    stumps = DecisionStump[]

    for t in 1:n_estimators
        # Compute pseudo-residuals (gradient of L2 loss = y - F)
        r = y .- F

        # Row subsampling
        n_sub = max(min_leaf * 2, round(Int, subsample * n))
        sub_idx = randperm(rng, n)[1:n_sub]
        X_sub   = X[sub_idx, :]
        r_sub   = r[sub_idx]

        # Feature subsampling
        feat_idx = sort(randperm(rng, p)[1:n_feat])
        X_sub_f  = X_sub[:, feat_idx]

        # Fit stump to residuals
        stump_sub = fit_stump(X_sub_f, r_sub; min_leaf=min_leaf)

        # Re-map to original feature index
        stump = DecisionStump(feat_idx[stump_sub.split_feature],
                               stump_sub.split_val, stump_sub.left_val,
                               stump_sub.right_val, stump_sub.train_mse)
        push!(stumps, stump)

        # Update predictions
        F .+= lr .* predict_stump(stump, X)
    end

    return GradientBoostingRegressor(stumps, n_estimators, lr, init_pred, subsample, n_feat)
end

"""
    predict_gbm(model, X) -> Vector{Float64}

Predict using fitted gradient boosting model.
"""
function predict_gbm(model::GradientBoostingRegressor, X::Matrix{Float64})::Vector{Float64}
    n = size(X, 1)
    F = fill(model.init_pred, n)
    for stump in model.stumps
        F .+= model.learning_rate .* predict_stump(stump, X)
    end
    return F
end

"""
    feature_importance_gbm(model, n_features) -> Vector{Float64}

Compute feature importance from a GBM as split frequency per feature.
More splits at a feature → higher importance.
"""
function feature_importance_gbm(model::GradientBoostingRegressor,
                                  n_features::Int)::Vector{Float64}
    importance = zeros(n_features)
    for stump in model.stumps
        if 1 <= stump.split_feature <= n_features
            importance[stump.split_feature] += 1
        end
    end
    total = sum(importance)
    total > 0 && (importance ./= total)
    return importance
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Random Forests
# ─────────────────────────────────────────────────────────────────────────────

"""
    RandomForestClassifier

Random forest for binary classification using bagged decision stumps.
Each tree is a single stump fit on a bootstrap sample with random features.
Predictions are probability of class 1.
"""
struct RandomForestClassifier
    stumps::Vector{DecisionStump}
    thresholds::Vector{Float64}  # each stump's decision boundary for class 1
    n_estimators::Int
    max_features::Int
end

"""
    fit_rf(X, y; n_estimators, max_features, rng) -> RandomForestClassifier

Fit random forest classifier.
y: binary labels (0 or 1).
max_features: number of features to consider at each split (√p default).
"""
function fit_rf(X::Matrix{Float64}, y::Vector{Bool};
                 n_estimators::Int=100, max_features::Int=0,
                 rng::AbstractRNG=Random.default_rng())::RandomForestClassifier
    n, p = size(X)
    mf = max_features == 0 ? max(1, round(Int, sqrt(p))) : max_features

    stumps     = DecisionStump[]
    thresholds = Float64[]

    for t in 1:n_estimators
        # Bootstrap sample
        boot_idx = rand(rng, 1:n, n)
        X_boot   = X[boot_idx, :]
        y_boot   = Float64.(y[boot_idx])

        # Random feature subset
        feat_idx = sort(randperm(rng, p)[1:min(mf, p)])
        X_feat   = X_boot[:, feat_idx]

        # Fit stump (treating y as continuous 0/1)
        stump_local = fit_stump(X_feat, y_boot; min_leaf=5)

        # Re-index features
        stump = DecisionStump(feat_idx[stump_local.split_feature],
                               stump_local.split_val, stump_local.left_val,
                               stump_local.right_val, stump_local.train_mse)
        push!(stumps, stump)
        push!(thresholds, 0.5)
    end

    return RandomForestClassifier(stumps, thresholds, n_estimators, mf)
end

"""
    predict_rf_proba(model, X) -> Vector{Float64}

Predict class-1 probability by averaging stump predictions.
"""
function predict_rf_proba(model::RandomForestClassifier, X::Matrix{Float64})::Vector{Float64}
    n = size(X, 1)
    proba = zeros(n)
    for stump in model.stumps
        proba .+= predict_stump(stump, X)
    end
    return clamp.(proba ./ model.n_estimators, 0.0, 1.0)
end

"""
    predict_rf(model, X; threshold) -> Vector{Bool}

Predict class labels using threshold on probability.
"""
function predict_rf(model::RandomForestClassifier, X::Matrix{Float64};
                     threshold::Float64=0.5)::Vector{Bool}
    return predict_rf_proba(model, X) .>= threshold
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Time Series Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

"""
    PurgedKFold

Purged K-fold cross-validation for financial time series.
Purging: removes observations from the training set that are within
a `purge_gap` of any test observation (to prevent lookahead bias).
Embargoing: further removes a `embargo_pct` fraction from the end of training.
"""
struct PurgedKFold
    k::Int
    purge_gap::Int       # bars to exclude around test observations
    embargo_pct::Float64 # fraction of test set to embargo from training end
end

PurgedKFold(k=5; purge_gap=10, embargo_pct=0.01) =
    PurgedKFold(k, purge_gap, embargo_pct)

"""
    cv_splits(pkf, n) -> Vector{NamedTuple}

Generate (train, test) index pairs for purged k-fold CV.
"""
function cv_splits(pkf::PurgedKFold, n::Int)::Vector{NamedTuple}
    fold_size = div(n, pkf.k)
    splits    = NamedTuple[]

    for fold in 1:pkf.k
        test_start = (fold-1) * fold_size + 1
        test_end   = fold == pkf.k ? n : fold * fold_size
        test_idx   = test_start:test_end

        # Purge: exclude within purge_gap of test set
        purge_lo = max(1,   test_start - pkf.purge_gap)
        purge_hi = min(n,   test_end   + pkf.purge_gap)

        # Embargo: also exclude embargo_pct of data after test end
        embargo_end = min(n, test_end + round(Int, pkf.embargo_pct * length(test_idx)))

        # Training: everything not in purge zone
        train_idx = [i for i in 1:n if !(purge_lo <= i <= purge_hi) &&
                     !(test_end < i <= embargo_end)]

        isempty(train_idx) && continue
        push!(splits, (train=train_idx, test=collect(test_idx), fold=fold))
    end
    return splits
end

"""
    combinatorial_purged_cv(n, k, n_test_splits; purge_gap) -> Vector{NamedTuple}

Combinatorial Purged CV (Lopez de Prado 2018): select n_test_splits folds
as test set (C(k, n_test) combinations), train on non-purged remainder.
Returns all combinations as (train, test) pairs.
"""
function combinatorial_purged_cv(n::Int, k::Int=6, n_test_splits::Int=2;
                                   purge_gap::Int=10)::Vector{NamedTuple}
    fold_size = div(n, k)
    folds     = [((f-1)*fold_size + 1):(f == k ? n : f*fold_size) for f in 1:k]

    # Generate all C(k, n_test_splits) combinations of test folds
    combs = _combinations(1:k, n_test_splits)
    splits = NamedTuple[]

    for test_fold_ids in combs
        test_idx = sort(vcat([collect(folds[f]) for f in test_fold_ids]...))

        # Purge zone
        purge_set = Set{Int}()
        for t in test_idx
            for p in max(1,t-purge_gap):min(n,t+purge_gap)
                push!(purge_set, p)
            end
        end

        train_idx = [i for i in 1:n if !(i in purge_set) && !(i in Set(test_idx))]
        isempty(train_idx) && continue
        push!(splits, (train=train_idx, test=test_idx, test_folds=test_fold_ids))
    end
    return splits
end

function _combinations(items, r)
    result = Vector{Vector{Int}}()
    n = length(items)
    r > n && return result
    indices = collect(1:r)
    while true
        push!(result, [items[i] for i in indices])
        i = r
        while i > 0 && indices[i] == n - r + i
            i -= 1
        end
        i == 0 && break
        indices[i] += 1
        for j in (i+1):r
            indices[j] = indices[j-1] + 1
        end
    end
    return result
end

"""
    walk_forward_cv(n; n_train, n_test, step) -> Vector{NamedTuple}

Classic walk-forward (expanding window or rolling window) CV.
"""
function walk_forward_cv(n::Int; n_train::Int=252, n_test::Int=63,
                           step::Int=63, expanding::Bool=true)::Vector{NamedTuple}
    splits = NamedTuple[]
    t = n_train + 1
    while t + n_test - 1 <= n
        train_start = expanding ? 1 : max(1, t - n_train)
        train_idx   = collect(train_start:(t-1))
        test_idx    = collect(t:(t+n_test-1))
        push!(splits, (train=train_idx, test=test_idx))
        t += step
    end
    return splits
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Permutation Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

"""
    permutation_importance(X, y, predict_fn, loss_fn; n_perms, rng) -> Vector{Float64}

General permutation feature importance.
For each feature j: permute column j, compute increase in loss vs baseline.
predict_fn(X) -> predictions, loss_fn(preds, y) -> scalar loss.
"""
function permutation_importance(X::Matrix{Float64}, y::Vector,
                                  predict_fn::Function, loss_fn::Function;
                                  n_perms::Int=10,
                                  rng::AbstractRNG=Random.default_rng())::Vector{Float64}
    n, p = size(X)
    base_loss = loss_fn(predict_fn(X), y)
    importances = zeros(p)

    for j in 1:p
        loss_increases = zeros(n_perms)
        for perm in 1:n_perms
            X_perm = copy(X)
            X_perm[:, j] = X_perm[randperm(rng, n), j]
            loss_increases[perm] = loss_fn(predict_fn(X_perm), y) - base_loss
        end
        importances[j] = mean(loss_increases)
    end
    return importances
end

"""
    permutation_importance_ts(X, y, model_fit_fn, model_pred_fn, loss_fn;
                               n_perms, rng) -> Vector{Float64}

Permutation importance with refitting (more accurate for time series).
Permutes feature j, refits model, measures performance.
"""
function permutation_importance_ts(X::Matrix{Float64}, y::Vector,
                                     model_fit_fn::Function,
                                     model_pred_fn::Function,
                                     loss_fn::Function;
                                     n_perms::Int=5,
                                     rng::AbstractRNG=Random.default_rng())::Vector{Float64}
    n, p = size(X)
    importances = zeros(p)

    base_model = model_fit_fn(X, y)
    base_loss  = loss_fn(model_pred_fn(base_model, X), y)

    for j in 1:p
        perm_losses = zeros(n_perms)
        for perm in 1:n_perms
            X_perm = copy(X)
            X_perm[:, j] = X_perm[randperm(rng, n), j]
            m_perm = model_fit_fn(X_perm, y)
            perm_losses[perm] = loss_fn(model_pred_fn(m_perm, X_perm), y)
        end
        importances[j] = mean(perm_losses) - base_loss
    end
    return importances
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Online Learning - Passive-Aggressive Algorithm
# ─────────────────────────────────────────────────────────────────────────────

"""
    PassiveAggressiveRegressor

Online learning regressor implementing the Passive-Aggressive algorithm
(Crammer et al. 2006). Suitable for non-stationary financial signals.
PA-I variant with aggressiveness parameter C.
"""
mutable struct PassiveAggressiveRegressor
    weights::Vector{Float64}
    bias::Float64
    C::Float64     # aggressiveness (larger = more aggressive updates)
    epsilon::Float64  # insensitive band
    n_updates::Int
end

"""
    PassiveAggressiveRegressor(n_features; C, epsilon) -> PassiveAggressiveRegressor
"""
function PassiveAggressiveRegressor(n_features::Int; C::Float64=0.1,
                                     epsilon::Float64=0.01)::PassiveAggressiveRegressor
    return PassiveAggressiveRegressor(zeros(n_features), 0.0, C, epsilon, 0)
end

"""
    pa_update!(model, x, y_true) -> Float64

Update the PA regressor with a single observation (x, y_true).
Returns the prediction before update (for online evaluation).
"""
function pa_update!(model::PassiveAggressiveRegressor,
                     x::Vector{Float64}, y_true::Float64)::Float64
    y_hat = dot(model.weights, x) + model.bias
    loss  = max(0.0, abs(y_hat - y_true) - model.epsilon)

    # PA-I update: step size τ = loss / ||x||²
    x_norm_sq = dot(x, x) + 1.0  # +1 for bias term
    tau = min(model.C, loss / x_norm_sq)

    # Sign of update
    sign_err = y_hat < y_true ? 1.0 : -1.0
    model.weights .+= tau * sign_err .* x
    model.bias      += tau * sign_err
    model.n_updates += 1

    return y_hat
end

"""
    predict_pa(model, X) -> Vector{Float64}

Predict using fitted PA model.
"""
function predict_pa(model::PassiveAggressiveRegressor, X::Matrix{Float64})::Vector{Float64}
    return X * model.weights .+ model.bias
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Conformal Prediction Intervals
# ─────────────────────────────────────────────────────────────────────────────

"""
    ConformalPredictor

Split conformal predictor for return forecasts.
Calibrated on a holdout set; provides coverage guarantee.
"""
struct ConformalPredictor
    base_model_preds::Vector{Float64}  # predictions on calibration set
    calibration_residuals::Vector{Float64}  # sorted non-conformity scores
    alpha::Float64    # miscoverage rate (1-alpha = coverage)
end

"""
    fit_conformal(y_true, y_pred_calib; alpha) -> ConformalPredictor

Fit conformal predictor using calibration set predictions and true values.
alpha: desired miscoverage rate (e.g. 0.10 for 90% coverage).
"""
function fit_conformal(y_true::Vector{Float64}, y_pred::Vector{Float64};
                         alpha::Float64=0.10)::ConformalPredictor
    n = length(y_true)
    scores = abs.(y_true .- y_pred)  # non-conformity scores
    return ConformalPredictor(y_pred, sort(scores), alpha)
end

"""
    predict_interval(conformal, y_hat; method) -> Tuple{Float64, Float64}

Produce a conformal prediction interval for a new point with prediction y_hat.
Returns (lower, upper) interval with guaranteed (1-alpha) coverage.
"""
function predict_interval(conformal::ConformalPredictor, y_hat::Float64;
                            method::Symbol=:split)::Tuple{Float64,Float64}
    n = length(conformal.calibration_residuals)
    # Find (1-alpha) quantile of residuals
    q_idx = clamp(ceil(Int, (1 - conformal.alpha) * (n + 1)), 1, n)
    q_hat = conformal.calibration_residuals[q_idx]

    return (y_hat - q_hat, y_hat + q_hat)
end

"""
    coverage_rate(y_true, y_pred, conformal) -> Float64

Compute empirical coverage rate of conformal intervals.
"""
function coverage_rate(y_true::Vector{Float64}, y_pred::Vector{Float64},
                         conformal::ConformalPredictor)::Float64
    n = length(y_true)
    covered = 0
    for i in 1:n
        lo, hi = predict_interval(conformal, y_pred[i])
        covered += (lo <= y_true[i] <= hi) ? 1 : 0
    end
    return covered / n
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: AIC/BIC for ARIMA/GARCH Model Selection
# ─────────────────────────────────────────────────────────────────────────────

"""
    aic(ll, k) -> Float64

Akaike Information Criterion: AIC = -2*ll + 2*k.
Lower = better model (penalises complexity).
"""
aic(ll::Float64, k::Int)::Float64 = -2*ll + 2*k

"""
    bic(ll, k, n) -> Float64

Bayesian Information Criterion: BIC = -2*ll + k*log(n).
More strongly penalises additional parameters than AIC.
"""
bic(ll::Float64, k::Int, n::Int)::Float64 = -2*ll + k*log(n)

"""
    ARIMA

ARIMA(p,d,q) model specification and fitted parameters.
"""
struct ARIMA
    p::Int               # AR order
    d::Int               # differencing order
    q::Int               # MA order
    ar_coefs::Vector{Float64}
    ma_coefs::Vector{Float64}
    sigma2::Float64      # residual variance
    ll::Float64          # log-likelihood
    n_obs::Int
end

"""
    fit_arima(y, p, d, q) -> ARIMA

Fit ARIMA(p,d,q) model by conditional least squares.
"""
function fit_arima(y::Vector{Float64}, p::Int=1, d::Int=0, q::Int=0)::ARIMA
    n = length(y)

    # Differencing
    y_diff = copy(y)
    for _ in 1:d
        y_diff = diff(y_diff)
    end
    n_d = length(y_diff)

    # Simple AR(p) estimation via OLS (ignore MA for simplicity)
    # For full ARIMA, we'd use MLE with Kalman filter; use OLS AR here
    n_use = n_d - max(p, q)
    n_use < 5 && return ARIMA(p, d, q, zeros(p), zeros(q), var(y_diff), -Inf, n)

    # Build design matrix for AR(p)
    Y = y_diff[(max(p,q)+1):end]
    X = hcat([y_diff[(max(p,q)+1-l):(end-l)] for l in 1:p]...)

    ar_coefs = (X' * X) \ (X' * Y)
    residuals = Y .- X * ar_coefs
    sigma2 = var(residuals)

    # Log-likelihood (normal)
    n_eff = length(Y)
    ll = -n_eff/2 * log(2π*sigma2) - sum(residuals.^2) / (2*sigma2)

    ma_coefs = zeros(q)  # MA not estimated in this simplified version

    return ARIMA(p, d, q, ar_coefs, ma_coefs, sigma2, ll, n_eff)
end

"""
    forecast_arima(model, y, h) -> Vector{Float64}

Generate h-step-ahead forecasts from fitted ARIMA model.
"""
function forecast_arima(model::ARIMA, y::Vector{Float64}, h::Int=10)::Vector{Float64}
    y_diff = copy(y)
    for _ in 1:model.d
        y_diff = diff(y_diff)
    end

    forecasts = zeros(h)
    history   = copy(y_diff)

    for step in 1:h
        pred = 0.0
        for (l, c) in enumerate(model.ar_coefs)
            idx = length(history) - l + 1
            idx > 0 && (pred += c * history[idx])
        end
        push!(history, pred)
        forecasts[step] = pred
    end

    # Un-difference
    if model.d > 0
        last_val = y[end]
        for step in 1:h
            forecasts[step] = last_val + forecasts[step]
            last_val = forecasts[step]
        end
    end

    return forecasts
end

"""
    GARCHSpec

GARCH(p,q) specification for model selection.
"""
struct GARCHSpec
    p::Int    # GARCH lags
    q::Int    # ARCH lags
    distribution::Symbol  # :normal or :t
    omega::Float64
    alpha::Vector{Float64}  # ARCH coefficients (q of them)
    beta::Vector{Float64}   # GARCH coefficients (p of them)
    nu::Float64             # df if t-distribution
    ll::Float64
    n_params::Int
end

"""
    fit_garch_aic(returns, p, q; distribution) -> GARCHSpec

Fit GARCH(p,q) and return the fitted spec with AIC/BIC.
Uses recursive variance estimation with method-of-moments initialisation.
"""
function fit_garch_aic(returns::Vector{Float64}, p::Int=1, q::Int=1;
                         distribution::Symbol=:normal)::GARCHSpec
    n = length(returns)
    n < 2*(p+q) + 5 && return _null_garch(p, q, distribution)

    # Simplified: fit only GARCH(1,1) analytically
    # For p>1 or q>1, uses grid search over persistence
    mu = mean(returns)
    target_var = var(returns)

    # Method of moments: use ACF of r² to estimate alpha+beta
    r2 = (returns .- mu).^2
    acf_r2 = length(r2) > 2 ? cor(r2[1:end-1], r2[2:end]) : 0.3
    persistence = clamp(acf_r2, 0.0, 0.99)

    alpha_total = persistence * 0.3
    beta_total  = persistence * 0.7

    alpha_v = fill(alpha_total / q, q)
    beta_v  = fill(beta_total  / p, p)
    omega   = target_var * (1 - alpha_total - beta_total)
    omega   = max(omega, 1e-8)

    # GARCH(1,1) for log-likelihood
    h_t = omega / max(1 - alpha_total - beta_total, 0.001)
    ll  = 0.0
    for t in 2:n
        h_t = omega + alpha_v[1] * (returns[t-1] - mu)^2 + beta_v[1] * h_t
        h_t = max(h_t, 1e-12)
        if distribution == :normal
            ll -= 0.5 * (log(2π*h_t) + (returns[t]-mu)^2 / h_t)
        else
            nu = 6.0  # default df
            ll -= lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(π*(nu-2)*h_t) -
                  (nu+1)/2 * log(1 + (returns[t]-mu)^2/((nu-2)*h_t))
        end
    end

    n_params = 2 + p + q + (distribution == :t ? 1 : 0)

    return GARCHSpec(p, q, distribution, omega, alpha_v, beta_v,
                      distribution == :t ? 6.0 : Inf, ll, n_params)
end

function _null_garch(p, q, dist)
    return GARCHSpec(p, q, dist, 1e-5, zeros(q), zeros(p), Inf, -Inf, p+q+2)
end

"""
    garch_model_selection(returns; max_p, max_q, criterion) -> NamedTuple

Select best GARCH(p,q) by AIC or BIC.
"""
function garch_model_selection(returns::Vector{Float64};
                                 max_p::Int=2, max_q::Int=2,
                                 criterion::Symbol=:bic)::NamedTuple
    n = length(returns)
    best_score = Inf
    best_spec  = _null_garch(1, 1, :normal)
    results    = NamedTuple[]

    for p in 1:max_p, q in 1:max_q
        spec = fit_garch_aic(returns, p, q; distribution=:normal)
        spec.ll == -Inf && continue

        score = criterion == :aic ? aic(spec.ll, spec.n_params) :
                                    bic(spec.ll, spec.n_params, n)
        push!(results, (p=p, q=q, ll=spec.ll, aic=aic(spec.ll, spec.n_params),
                         bic=bic(spec.ll, spec.n_params, n)))

        if score < best_score
            best_score = score
            best_spec  = spec
        end
    end

    return (best_spec=best_spec, best_score=best_score, criterion=criterion, all_results=results)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Ensemble Methods
# ─────────────────────────────────────────────────────────────────────────────

"""
    BaggingEnsemble

Bagging: train n_estimators models on bootstrap samples, average predictions.
"""
struct BaggingEnsemble
    models::Vector{GradientBoostingRegressor}
    n_estimators::Int
end

"""
    fit_bagging(X, y; n_estimators, n_sub, lr, rng) -> BaggingEnsemble

Fit bagging ensemble of gradient boosting trees.
"""
function fit_bagging(X::Matrix{Float64}, y::Vector{Float64};
                      n_estimators::Int=10, n_sub::Int=50,
                      lr::Float64=0.1, rng::AbstractRNG=Random.default_rng())::BaggingEnsemble
    n, p = size(X)
    models = GradientBoostingRegressor[]

    for _ in 1:n_estimators
        boot_idx = rand(rng, 1:n, n)
        m = fit_gbm(X[boot_idx, :], y[boot_idx]; n_estimators=n_sub,
                     lr=lr, rng=rng)
        push!(models, m)
    end

    return BaggingEnsemble(models, n_estimators)
end

"""
    predict_bagging(ensemble, X) -> Vector{Float64}

Predict using bagging ensemble (simple average).
"""
function predict_bagging(ensemble::BaggingEnsemble, X::Matrix{Float64})::Vector{Float64}
    preds = zeros(size(X, 1))
    for m in ensemble.models
        preds .+= predict_gbm(m, X)
    end
    return preds ./ ensemble.n_estimators
end

"""
    StackedEnsemble

Stacking: train base models, then a meta-learner on their out-of-fold predictions.
"""
struct StackedEnsemble
    base_models::Vector{GradientBoostingRegressor}
    meta_weights::Vector{Float64}  # linear meta-learner weights
    meta_intercept::Float64
end

"""
    fit_stacking(X, y; n_base, k_fold, lr, rng) -> StackedEnsemble

Fit stacking ensemble with k-fold out-of-fold predictions.
Base models: GBM with different hyperparameters.
Meta-learner: OLS regression on base model predictions.
"""
function fit_stacking(X::Matrix{Float64}, y::Vector{Float64};
                       n_base::Int=3, k_fold::Int=5, lr::Float64=0.1,
                       rng::AbstractRNG=Random.default_rng())::StackedEnsemble
    n, p = size(X)
    fold_size = div(n, k_fold)
    oof_preds = zeros(n, n_base)  # out-of-fold predictions

    # Different configurations for base models
    configs = [(n_est=50, lr=0.05), (n_est=100, lr=0.1), (n_est=80, lr=0.05)]
    configs = configs[1:min(n_base, length(configs))]

    for (bi, cfg) in enumerate(configs)
        for fold in 1:k_fold
            test_lo = (fold-1) * fold_size + 1
            test_hi = fold == k_fold ? n : fold * fold_size
            test_idx  = test_lo:test_hi
            train_idx = [i for i in 1:n if !(test_lo <= i <= test_hi)]

            isempty(train_idx) && continue
            m = fit_gbm(X[train_idx, :], y[train_idx];
                         n_estimators=cfg.n_est, lr=cfg.lr, rng=rng)
            oof_preds[test_idx, bi] = predict_gbm(m, X[test_idx, :])
        end
    end

    # Train meta-learner: OLS on oof predictions
    X_meta = hcat(ones(n), oof_preds)
    coeffs = (X_meta' * X_meta) \ (X_meta' * y)
    meta_intercept = coeffs[1]
    meta_weights   = coeffs[2:end]

    # Refit base models on full training data
    final_base = [fit_gbm(X, y; n_estimators=cfg.n_est, lr=cfg.lr, rng=rng)
                  for cfg in configs]

    return StackedEnsemble(final_base, meta_weights, meta_intercept)
end

"""
    predict_stacking(ensemble, X) -> Vector{Float64}

Predict using stacking ensemble.
"""
function predict_stacking(ensemble::StackedEnsemble, X::Matrix{Float64})::Vector{Float64}
    n = size(X, 1)
    base_preds = hcat([predict_gbm(m, X) for m in ensemble.base_models]...)
    return base_preds * ensemble.meta_weights .+ ensemble.meta_intercept
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    mse_loss(preds, y) -> Float64

Mean squared error loss function.
"""
mse_loss(preds::Vector{Float64}, y::Vector{Float64})::Float64 = mean((preds .- y).^2)

"""
    mae_loss(preds, y) -> Float64

Mean absolute error loss function.
"""
mae_loss(preds::Vector{Float64}, y::Vector{Float64})::Float64 = mean(abs.(preds .- y))

"""
    standardise(X; dims) -> Tuple{Matrix, Vector, Vector}

Standardise feature matrix (zero mean, unit variance).
Returns (X_std, means, stds).
"""
function standardise(X::Matrix{Float64}; dims::Int=1)
    mu  = mean(X; dims=dims)[:]
    sig = std(X;  dims=dims)[:]
    sig[sig .< 1e-8] .= 1.0
    X_std = (X .- mu') ./ sig'
    return X_std, mu, sig
end

"""
    unstandardise(X_std, mu, sig) -> Matrix{Float64}

Reverse standardisation.
"""
function unstandardise(X_std::Matrix{Float64}, mu::Vector{Float64},
                         sig::Vector{Float64})::Matrix{Float64}
    return X_std .* sig' .+ mu'
end

end  # module TimeSeriesML
