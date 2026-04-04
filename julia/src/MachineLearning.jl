"""
MachineLearning — Machine learning for return prediction and alpha generation.

Implements:
  - LSTM for return prediction (via Flux.jl)
  - Gradient boosting / XGBoost-style trees (custom implementation)
  - Gaussian process regression with uncertainty quantification
  - Online learning: RLS, Kalman filter regression, exponential forgetting
  - Feature selection: LASSO, elastic net, forward stepwise, mutual information
  - Cross-validation: walk-forward, purged k-fold, combinatorial purged CV
"""
module MachineLearning

using LinearAlgebra
using Statistics
using Distributions
using Random
using DataFrames

# Flux for neural networks
using Flux
using Flux: LSTM, Dense, Chain, gradient, params, Adam, Optimise

export LSTMPredictor, train_lstm, predict_lstm
export GradientBoostingTree, GBTree, train_gbtree, predict_gbtree, feature_importance_gb
export GaussianProcess, gp_predict, gp_optimize_hyperparams
export RLSRegressor, rls_update!, kalman_filter_regression, ExpForgetLearner
export lasso_coordinate_descent, elastic_net, forward_stepwise, mutual_information_selection
export WalkForwardCV, purged_kfold_cv, combinatorial_purged_cv

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: LSTM for Return Prediction
# ─────────────────────────────────────────────────────────────────────────────

"""
    LSTMPredictor

LSTM-based return prediction model wrapping Flux.jl.
"""
struct LSTMPredictor
    model::Any           # Flux Chain
    n_features::Int
    n_hidden::Int
    seq_len::Int
    n_ahead::Int
end

"""
    build_lstm_model(n_features::Int, n_hidden::Int, n_layers::Int, n_ahead::Int) -> Chain

Build a multi-layer LSTM model for time series prediction.
"""
function build_lstm_model(n_features::Int, n_hidden::Int, n_layers::Int, n_ahead::Int)
    layers = []
    push!(layers, LSTM(n_features => n_hidden))
    for _ in 2:n_layers
        push!(layers, LSTM(n_hidden => n_hidden))
    end
    push!(layers, Dense(n_hidden => n_ahead))
    return Chain(layers...)
end

"""
    prepare_sequences(X::Matrix{Float64}, y::Vector{Float64}, seq_len::Int)
    -> Tuple{Array{Float64,3}, Matrix{Float64}}

Convert feature matrix to (seq_len, n_features, n_samples) 3D array
and target matrix.
"""
function prepare_sequences(X::Matrix{Float64}, y::Vector{Float64}, seq_len::Int)
    n_obs, n_features = size(X)
    n_samples = n_obs - seq_len + 1
    X_seq = zeros(Float32, seq_len, n_features, n_samples)
    y_out = zeros(Float32, 1, n_samples)

    for i in 1:n_samples
        X_seq[:, :, i] = Float32.(X[i:i+seq_len-1, :])
        y_out[1, i] = Float32(y[i + seq_len - 1])
    end

    return X_seq, y_out
end

"""
    train_lstm(
        X::Matrix{Float64},
        y::Vector{Float64};
        n_hidden=64,
        n_layers=2,
        seq_len=20,
        n_ahead=1,
        epochs=100,
        lr=0.001,
        batch_size=32,
        rng=Random.GLOBAL_RNG
    ) -> LSTMPredictor

Train an LSTM model on (X, y) data.
"""
function train_lstm(
    X::Matrix{Float64},
    y::Vector{Float64};
    n_hidden::Int=64,
    n_layers::Int=2,
    seq_len::Int=20,
    n_ahead::Int=1,
    epochs::Int=100,
    lr::Float64=0.001,
    batch_size::Int=32,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    verbose::Bool=false
)
    n_features = size(X, 2)
    model = build_lstm_model(n_features, n_hidden, n_layers, n_ahead)
    opt = Adam(lr)

    X_seq, y_targets = prepare_sequences(X, y, seq_len)
    n_samples = size(X_seq, 3)

    loss_history = Float64[]

    for epoch in 1:epochs
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle
        idx = randperm(rng, n_samples)

        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            batch_idx = idx[batch_start:batch_end]

            X_batch = X_seq[:, :, batch_idx]
            y_batch = y_targets[:, batch_idx]

            # Reset LSTM state
            Flux.reset!(model)

            # Process sequence step by step
            # Convert to format Flux expects: vector of matrices
            x_steps = [X_batch[t, :, :] for t in 1:seq_len]

            gs = gradient(params(model)) do
                # Run through LSTM
                for t in 1:(seq_len-1)
                    model[1:end-1](x_steps[t])
                end
                ŷ = model(x_steps[end])
                Flux.mse(ŷ, y_batch)
            end

            Optimise.update!(opt, params(model), gs)

            # Compute batch loss for reporting
            Flux.reset!(model)
            for t in 1:(seq_len-1)
                model[1:end-1](x_steps[t])
            end
            ŷ = model(x_steps[end])
            batch_loss = Flux.mse(ŷ, y_batch)
            epoch_loss += batch_loss
            n_batches += 1
        end

        avg_loss = epoch_loss / max(n_batches, 1)
        push!(loss_history, avg_loss)

        if verbose && epoch % 10 == 0
            @info "Epoch $epoch / $epochs, Loss = $(round(avg_loss, digits=6))"
        end
    end

    return LSTMPredictor(model, n_features, n_hidden, seq_len, n_ahead)
end

"""
    predict_lstm(predictor::LSTMPredictor, X::Matrix{Float64}) -> Vector{Float64}

Generate return predictions using trained LSTM.
"""
function predict_lstm(predictor::LSTMPredictor, X::Matrix{Float64})
    model = predictor.model
    seq_len = predictor.seq_len
    n_obs, n_features = size(X)

    if n_obs < seq_len
        return zeros(0)
    end

    n_preds = n_obs - seq_len + 1
    predictions = zeros(n_preds)

    Flux.reset!(model)

    for i in 1:n_preds
        x_window = X[i:i+seq_len-1, :]
        x_steps = [Float32.(x_window[t:t, :])' for t in 1:seq_len]

        Flux.reset!(model)
        for t in 1:(seq_len-1)
            model[1:end-1](x_steps[t])
        end
        ŷ = model(x_steps[end])
        predictions[i] = Float64(ŷ[1, 1])
    end

    return predictions
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Gradient Boosting Trees
# ─────────────────────────────────────────────────────────────────────────────

"""
    GBTree

A single regression tree for gradient boosting.
"""
struct GBTree
    split_feature::Int
    split_value::Float64
    left_val::Float64
    right_val::Float64
    left_tree::Union{Nothing, GBTree}
    right_tree::Union{Nothing, GBTree}
    depth::Int
end

"""
    fit_regression_tree(X, residuals, max_depth, min_samples_leaf, rng) -> GBTree

Fit a single regression tree to residuals for gradient boosting.
Uses MSE splitting criterion.
"""
function fit_regression_tree(
    X::Matrix{Float64},
    residuals::Vector{Float64},
    max_depth::Int,
    min_samples_leaf::Int,
    rng::AbstractRNG
)
    return _fit_tree(X, residuals, 1, max_depth, min_samples_leaf)
end

function _fit_tree(X, y, depth, max_depth, min_samples_leaf)
    n, p = size(X)

    leaf_val = mean(y)

    if depth >= max_depth || n < 2 * min_samples_leaf
        return GBTree(0, 0.0, leaf_val, leaf_val, nothing, nothing, depth)
    end

    best_feat = 0
    best_val = 0.0
    best_mse = Inf

    for feat in 1:p
        sorted_vals = sort(unique(X[:, feat]))
        for i in 1:(length(sorted_vals)-1)
            threshold = (sorted_vals[i] + sorted_vals[i+1]) / 2
            left_mask = X[:, feat] .<= threshold
            right_mask = .!left_mask

            n_left = sum(left_mask)
            n_right = sum(right_mask)

            if n_left < min_samples_leaf || n_right < min_samples_leaf
                continue
            end

            y_left = y[left_mask]
            y_right = y[right_mask]

            mse = sum((y_left .- mean(y_left)) .^ 2) + sum((y_right .- mean(y_right)) .^ 2)

            if mse < best_mse
                best_mse = mse
                best_feat = feat
                best_val = threshold
            end
        end
    end

    if best_feat == 0
        return GBTree(0, 0.0, leaf_val, leaf_val, nothing, nothing, depth)
    end

    left_mask = X[:, best_feat] .<= best_val
    right_mask = .!left_mask

    left_subtree = _fit_tree(X[left_mask, :], y[left_mask], depth+1, max_depth, min_samples_leaf)
    right_subtree = _fit_tree(X[right_mask, :], y[right_mask], depth+1, max_depth, min_samples_leaf)

    return GBTree(best_feat, best_val, mean(y[left_mask]), mean(y[right_mask]),
                   left_subtree, right_subtree, depth)
end

function _predict_tree(tree::GBTree, x::Vector{Float64})
    if tree.split_feature == 0
        return tree.left_val
    end

    if x[tree.split_feature] <= tree.split_value
        if isnothing(tree.left_tree)
            return tree.left_val
        end
        return _predict_tree(tree.left_tree, x)
    else
        if isnothing(tree.right_tree)
            return tree.right_val
        end
        return _predict_tree(tree.right_tree, x)
    end
end

"""
    GradientBoostingTree

Full gradient boosting ensemble.
"""
struct GradientBoostingTree
    trees::Vector{GBTree}
    learning_rate::Float64
    n_estimators::Int
    base_prediction::Float64
    feature_importances::Vector{Float64}
    n_features::Int
end

"""
    train_gbtree(
        X::Matrix{Float64},
        y::Vector{Float64};
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        subsample=0.8,
        rng=Random.GLOBAL_RNG
    ) -> GradientBoostingTree

Train gradient boosting tree ensemble for regression.
"""
function train_gbtree(
    X::Matrix{Float64},
    y::Vector{Float64};
    n_estimators::Int=100,
    learning_rate::Float64=0.1,
    max_depth::Int=3,
    min_samples_leaf::Int=5,
    subsample::Float64=0.8,
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    n, p = size(X)
    base_pred = mean(y)
    residuals = y .- base_pred
    trees = GBTree[]

    # Feature importance accumulator
    feat_importance = zeros(p)

    for iter in 1:n_estimators
        # Subsample
        n_sub = round(Int, n * subsample)
        sub_idx = randperm(rng, n)[1:n_sub]
        X_sub = X[sub_idx, :]
        r_sub = residuals[sub_idx]

        tree = fit_regression_tree(X_sub, r_sub, max_depth, min_samples_leaf, rng)
        push!(trees, tree)

        # Update residuals
        for i in 1:n
            pred_i = _predict_tree(tree, X[i, :])
            residuals[i] -= learning_rate * pred_i
        end

        # Accumulate feature importance (using split feature counts)
        _accumulate_importance!(feat_importance, tree)
    end

    feat_importance ./= sum(feat_importance) + 1e-10

    return GradientBoostingTree(trees, learning_rate, n_estimators, base_pred,
                                 feat_importance, p)
end

function _accumulate_importance!(fi::Vector{Float64}, tree::GBTree)
    if tree.split_feature > 0
        fi[tree.split_feature] += 1.0
        if !isnothing(tree.left_tree)
            _accumulate_importance!(fi, tree.left_tree)
        end
        if !isnothing(tree.right_tree)
            _accumulate_importance!(fi, tree.right_tree)
        end
    end
end

"""
    predict_gbtree(gb::GradientBoostingTree, X::Matrix{Float64}) -> Vector{Float64}

Generate predictions from trained gradient boosting model.
"""
function predict_gbtree(gb::GradientBoostingTree, X::Matrix{Float64})
    n = size(X, 1)
    preds = fill(gb.base_prediction, n)

    for tree in gb.trees
        for i in 1:n
            preds[i] += gb.learning_rate * _predict_tree(tree, X[i, :])
        end
    end

    return preds
end

"""
    feature_importance_gb(gb::GradientBoostingTree, feature_names=nothing) -> DataFrame

Return feature importances from gradient boosting model.
"""
function feature_importance_gb(gb::GradientBoostingTree; feature_names=nothing)
    p = gb.n_features
    names = isnothing(feature_names) ? ["feature_$i" for i in 1:p] : feature_names
    df = DataFrame(feature=names, importance=gb.feature_importances)
    sort!(df, :importance, rev=true)
    return df
end

"""
    shap_values(gb::GradientBoostingTree, X::Matrix{Float64};
                n_background=50, rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Compute SHAP values via sampling (approximate TreeSHAP).
Returns n_samples × n_features matrix of SHAP values.
"""
function shap_values(
    gb::GradientBoostingTree,
    X::Matrix{Float64};
    n_background::Int=50,
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    n, p = size(X)
    n_bg = min(n_background, n)
    bg_idx = randperm(rng, n)[1:n_bg]
    X_bg = X[bg_idx, :]

    shap = zeros(n, p)

    for i in 1:n
        x_i = X[i, :]
        for feat in 1:p
            # Marginal contribution of feature feat
            # Compare prediction with feat set to x_i[feat] vs background
            with_feat = 0.0
            without_feat = 0.0
            for b in 1:n_bg
                x_with = copy(X_bg[b, :])
                x_with[feat] = x_i[feat]
                x_without = X_bg[b, :]
                with_feat += predict_gbtree(gb, reshape(x_with, 1, p))[1]
                without_feat += predict_gbtree(gb, reshape(x_without, 1, p))[1]
            end
            shap[i, feat] = (with_feat - without_feat) / n_bg
        end
    end

    return shap
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Gaussian Process Regression
# ─────────────────────────────────────────────────────────────────────────────

"""
    GaussianProcess

Gaussian process regression model.
"""
struct GaussianProcess
    X_train::Matrix{Float64}
    y_train::Vector{Float64}
    K_inv::Matrix{Float64}     # inverse of kernel matrix + noise
    alpha::Vector{Float64}     # K_inv * y
    length_scale::Float64
    sigma_f::Float64           # signal variance
    sigma_n::Float64           # noise variance
    kernel::Symbol             # :rbf, :matern32, :matern52
end

"""
    rbf_kernel(X1, X2, l, sigma_f) -> Matrix

Radial basis function (squared exponential) kernel.
k(x1, x2) = sigma_f^2 * exp(-||x1-x2||^2 / (2*l^2))
"""
function rbf_kernel(X1::Matrix{Float64}, X2::Matrix{Float64},
                     l::Float64, sigma_f::Float64)
    n1, n2 = size(X1, 1), size(X2, 1)
    K = zeros(n1, n2)
    for i in 1:n1
        for j in 1:n2
            d2 = sum((X1[i, :] .- X2[j, :]) .^ 2)
            K[i, j] = sigma_f^2 * exp(-d2 / (2 * l^2))
        end
    end
    return K
end

"""
    matern32_kernel(X1, X2, l, sigma_f) -> Matrix

Matérn 3/2 kernel.
k(r) = sigma_f^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
"""
function matern32_kernel(X1::Matrix{Float64}, X2::Matrix{Float64},
                          l::Float64, sigma_f::Float64)
    n1, n2 = size(X1, 1), size(X2, 1)
    K = zeros(n1, n2)
    sqrt3 = sqrt(3.0)
    for i in 1:n1
        for j in 1:n2
            r = norm(X1[i, :] .- X2[j, :])
            K[i, j] = sigma_f^2 * (1 + sqrt3 * r / l) * exp(-sqrt3 * r / l)
        end
    end
    return K
end

"""
    matern52_kernel(X1, X2, l, sigma_f) -> Matrix

Matérn 5/2 kernel.
"""
function matern52_kernel(X1::Matrix{Float64}, X2::Matrix{Float64},
                          l::Float64, sigma_f::Float64)
    n1, n2 = size(X1, 1), size(X2, 1)
    K = zeros(n1, n2)
    sqrt5 = sqrt(5.0)
    for i in 1:n1
        for j in 1:n2
            r = norm(X1[i, :] .- X2[j, :])
            K[i, j] = sigma_f^2 * (1 + sqrt5*r/l + 5*r^2/(3*l^2)) * exp(-sqrt5*r/l)
        end
    end
    return K
end

function compute_kernel(X1, X2, l, sigma_f, kernel::Symbol)
    if kernel == :rbf
        return rbf_kernel(X1, X2, l, sigma_f)
    elseif kernel == :matern32
        return matern32_kernel(X1, X2, l, sigma_f)
    else
        return matern52_kernel(X1, X2, l, sigma_f)
    end
end

"""
    fit_gp(
        X_train::Matrix{Float64},
        y_train::Vector{Float64};
        length_scale=1.0,
        sigma_f=1.0,
        sigma_n=0.1,
        kernel=:rbf,
        optimize=true
    ) -> GaussianProcess

Fit a Gaussian process regression model.
"""
function fit_gp(
    X_train::Matrix{Float64},
    y_train::Vector{Float64};
    length_scale::Float64=1.0,
    sigma_f::Float64=1.0,
    sigma_n::Float64=0.1,
    kernel::Symbol=:rbf,
    optimize_hp::Bool=true
)
    n = size(X_train, 1)
    l = length_scale
    sf = sigma_f
    sn = sigma_n

    if optimize_hp
        # Optimize hyperparameters via marginal log likelihood
        function neg_log_marginal_likelihood(params)
            l_p = exp(params[1])
            sf_p = exp(params[2])
            sn_p = exp(params[3])
            try
                K = compute_kernel(X_train, X_train, l_p, sf_p, kernel)
                K_noise = K + (sn_p^2 + 1e-6) * I
                L = cholesky(Symmetric(K_noise)).L
                alpha_p = L' \ (L \ y_train)
                # -log p(y|X) = 0.5*(y'*alpha + log|K| + n*log(2pi))
                return 0.5 * dot(y_train, alpha_p) + sum(log.(diag(L))) + 0.5 * n * log(2π)
            catch
                return 1e10
            end
        end

        x0 = [log(l), log(sf), log(sn)]
        result = optimize(neg_log_marginal_likelihood, x0, NelderMead();
            options=Optim.Options(iterations=500))

        x_opt = Optim.minimizer(result)
        l = exp(x_opt[1])
        sf = exp(x_opt[2])
        sn = exp(x_opt[3])
    end

    K = compute_kernel(X_train, X_train, l, sf, kernel)
    K_noise = K + (sn^2 + 1e-6) * I

    K_inv = try
        inv(Symmetric(K_noise))
    catch
        inv(K_noise + 1e-8 * I)
    end

    alpha = K_inv * y_train

    return GaussianProcess(X_train, y_train, K_inv, alpha, l, sf, sn, kernel)
end

"""
    gp_predict(gp::GaussianProcess, X_test::Matrix{Float64}) -> NamedTuple

Generate GP predictions with uncertainty estimates.
Returns: (mean, variance, std, lower_95, upper_95)
"""
function gp_predict(gp::GaussianProcess, X_test::Matrix{Float64})
    K_star = compute_kernel(X_test, gp.X_train, gp.length_scale, gp.sigma_f, gp.kernel)
    K_starstar = compute_kernel(X_test, X_test, gp.length_scale, gp.sigma_f, gp.kernel)

    mu = K_star * gp.alpha
    cov_pred = K_starstar - K_star * gp.K_inv * K_star'
    var_pred = max.(diag(cov_pred), 0.0)
    std_pred = sqrt.(var_pred)

    z = 1.96
    return (
        mean=mu,
        variance=var_pred,
        std=std_pred,
        lower_95=mu .- z .* std_pred,
        upper_95=mu .+ z .* std_pred
    )
end

"""
    gp_optimize_hyperparams(
        X::Matrix{Float64}, y::Vector{Float64};
        kernels=[:rbf, :matern32, :matern52],
        cv_folds=5,
        rng=Random.GLOBAL_RNG
    ) -> NamedTuple

Select best GP kernel and hyperparameters via cross-validation.
"""
function gp_optimize_hyperparams(
    X::Matrix{Float64}, y::Vector{Float64};
    kernels::Vector{Symbol}=[:rbf, :matern32, :matern52],
    cv_folds::Int=5,
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    n = size(X, 1)
    fold_size = n ÷ cv_folds

    best_kernel = :rbf
    best_rmse = Inf
    results = []

    for k in kernels
        rmse_folds = Float64[]
        for fold in 1:cv_folds
            val_start = (fold - 1) * fold_size + 1
            val_end = min(fold * fold_size, n)
            val_idx = val_start:val_end
            train_idx = setdiff(1:n, val_idx)

            X_tr, y_tr = X[train_idx, :], y[train_idx]
            X_val, y_val = X[val_idx, :], y[val_idx]

            try
                gp = fit_gp(X_tr, y_tr; kernel=k, optimize_hp=false)
                pred = gp_predict(gp, X_val)
                rmse = sqrt(mean((pred.mean .- y_val) .^ 2))
                push!(rmse_folds, rmse)
            catch
                push!(rmse_folds, Inf)
            end
        end

        mean_rmse = mean(rmse_folds)
        push!(results, (kernel=k, rmse=mean_rmse))

        if mean_rmse < best_rmse
            best_rmse = mean_rmse
            best_kernel = k
        end
    end

    best_gp = fit_gp(X, y; kernel=best_kernel, optimize_hp=true)

    return (gp=best_gp, best_kernel=best_kernel, best_rmse=best_rmse, results=results)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Online Learning
# ─────────────────────────────────────────────────────────────────────────────

"""
    RLSRegressor

Recursive Least Squares regression with exponential forgetting.
"""
mutable struct RLSRegressor
    theta::Vector{Float64}    # parameter vector
    P::Matrix{Float64}        # covariance matrix
    lambda::Float64           # forgetting factor
    n_features::Int
end

"""
    RLSRegressor(n_features::Int; lambda=0.99, P_init=1000.0) -> RLSRegressor

Initialize an RLS regressor.
"""
function RLSRegressor(n_features::Int; lambda::Float64=0.99, P_init::Float64=1000.0)
    theta = zeros(n_features)
    P = P_init * I(n_features)
    return RLSRegressor(theta, Matrix(P), lambda, n_features)
end

"""
    rls_update!(rls::RLSRegressor, x::Vector{Float64}, y::Float64)

Update RLS estimate with new observation (x, y).
Uses the matrix inversion lemma for O(p^2) update.
"""
function rls_update!(rls::RLSRegressor, x::Vector{Float64}, y::Float64)
    # Prediction error
    y_hat = dot(rls.theta, x)
    error = y - y_hat

    # Gain vector: K = P*x / (lambda + x'*P*x)
    Px = rls.P * x
    denom = rls.lambda + dot(x, Px)
    K = Px ./ denom

    # Update parameters
    rls.theta .+= K .* error

    # Update covariance
    rls.P = (rls.P .- K * x' * rls.P) ./ rls.lambda

    return y_hat, error
end

"""
    rls_predict(rls::RLSRegressor, X::Matrix{Float64}) -> Vector{Float64}
"""
function rls_predict(rls::RLSRegressor, X::Matrix{Float64})
    return X * rls.theta
end

"""
    kalman_filter_regression(
        X::Matrix{Float64},
        y::Vector{Float64};
        sigma_w=0.01,  # process noise
        sigma_v=0.1,   # observation noise
        lambda_kf=0.99
    ) -> NamedTuple

Run Kalman filter regression where coefficients evolve as a random walk.
State: theta_t = theta_{t-1} + w_t (w ~ N(0, sigma_w^2 * I))
Observation: y_t = x_t' * theta_t + v_t (v ~ N(0, sigma_v^2))
"""
function kalman_filter_regression(
    X::Matrix{Float64},
    y::Vector{Float64};
    sigma_w::Float64=0.01,
    sigma_v::Float64=0.1,
    lambda_kf::Float64=0.99
)
    n, p = size(X)
    theta = zeros(p, n + 1)
    P = 100.0 * Matrix(I(p))
    Q = sigma_w^2 * Matrix(I(p))  # process noise
    R = sigma_v^2                  # observation noise

    innovations = zeros(n)
    predictions = zeros(n)
    theta_history = zeros(p, n)

    for t in 1:n
        x_t = X[t, :]

        # Predict
        theta_pred = theta[:, t]
        P_pred = P + Q

        # Innovation
        y_hat = dot(x_t, theta_pred)
        S = dot(x_t, P_pred * x_t) + R
        K = (P_pred * x_t) ./ S
        innov = y[t] - y_hat

        # Update
        theta[:, t+1] = theta_pred .+ K .* innov
        P = (I(p) - K * x_t') * P_pred

        innovations[t] = innov
        predictions[t] = y_hat
        theta_history[:, t] = theta[:, t+1]
    end

    residuals = y .- predictions
    return (
        theta_final=theta[:, end],
        theta_history=theta_history,
        predictions=predictions,
        residuals=residuals,
        innovations=innovations
    )
end

"""
    ExpForgetLearner

Exponential forgetting adaptive learner (stochastic gradient with decay).
"""
mutable struct ExpForgetLearner
    theta::Vector{Float64}
    lambda::Float64   # forgetting factor
    alpha::Float64    # learning rate
    n_obs::Int
end

function ExpForgetLearner(n_features::Int; lambda::Float64=0.99, alpha::Float64=0.01)
    return ExpForgetLearner(zeros(n_features), lambda, alpha, 0)
end

function update!(efl::ExpForgetLearner, x::Vector{Float64}, y::Float64)
    y_hat = dot(efl.theta, x)
    error = y - y_hat
    # Gradient update with forgetting
    efl.theta = efl.lambda .* efl.theta .+ efl.alpha .* error .* x
    efl.n_obs += 1
    return y_hat, error
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Feature Selection
# ─────────────────────────────────────────────────────────────────────────────

"""
    lasso_coordinate_descent(
        X::Matrix{Float64},
        y::Vector{Float64},
        lambda::Float64;
        max_iter=1000,
        tol=1e-6,
        intercept=true
    ) -> Vector{Float64}

LASSO regression via coordinate descent.
Minimizes: (1/2n) ||y - X*beta||^2 + lambda * ||beta||_1
"""
function lasso_coordinate_descent(
    X::Matrix{Float64},
    y::Vector{Float64},
    lambda::Float64;
    max_iter::Int=1000,
    tol::Float64=1e-6,
    intercept::Bool=true
)
    n, p = size(X)

    # Standardize features
    X_mean = vec(mean(X, dims=1))
    X_std = vec(std(X, dims=1))
    X_std[X_std .< 1e-10] .= 1.0
    X_sc = (X .- X_mean') ./ X_std'

    y_mean = mean(y)
    y_c = y .- y_mean

    beta = zeros(p)
    resid = copy(y_c)

    # Precompute column norms
    col_norms_sq = vec(sum(X_sc .^ 2, dims=1)) ./ n

    for iter in 1:max_iter
        beta_old = copy(beta)

        for j in 1:p
            # Partial residual (exclude feature j contribution)
            resid_j = resid .+ X_sc[:, j] .* beta[j]

            # OLS coefficient for feature j
            rho_j = dot(X_sc[:, j], resid_j) / (n * col_norms_sq[j] + 1e-10)

            # Soft-threshold
            beta[j] = soft_threshold(rho_j, lambda / col_norms_sq[j])

            # Update residual
            resid = resid_j .- X_sc[:, j] .* beta[j]
        end

        if norm(beta - beta_old) < tol
            break
        end
    end

    # Unscale
    beta_unscaled = beta ./ X_std
    intercept_val = intercept ? y_mean - dot(X_mean, beta_unscaled) : 0.0

    return vcat(intercept_val, beta_unscaled)
end

"""
    soft_threshold(x::Float64, lambda::Float64) -> Float64

Soft-thresholding operator.
"""
soft_threshold(x::Float64, lambda::Float64) = sign(x) * max(abs(x) - lambda, 0.0)

"""
    elastic_net(
        X::Matrix{Float64},
        y::Vector{Float64},
        alpha::Float64,   # mixing parameter (1=lasso, 0=ridge)
        lambda::Float64;  # overall regularization
        max_iter=1000,
        tol=1e-6
    ) -> Vector{Float64}

Elastic net via coordinate descent.
Minimizes: (1/2n)||y-Xb||^2 + lambda*(alpha*||b||_1 + (1-alpha)/2*||b||^2)
"""
function elastic_net(
    X::Matrix{Float64},
    y::Vector{Float64},
    alpha::Float64,
    lambda::Float64;
    max_iter::Int=1000,
    tol::Float64=1e-6
)
    n, p = size(X)
    X_mean = vec(mean(X, dims=1))
    X_std = vec(std(X, dims=1))
    X_std[X_std .< 1e-10] .= 1.0
    X_sc = (X .- X_mean') ./ X_std'

    y_mean = mean(y)
    y_c = y .- y_mean

    beta = zeros(p)
    resid = copy(y_c)
    col_norms_sq = vec(sum(X_sc .^ 2, dims=1)) ./ n

    for iter in 1:max_iter
        beta_old = copy(beta)

        for j in 1:p
            resid_j = resid .+ X_sc[:, j] .* beta[j]
            rho_j = dot(X_sc[:, j], resid_j) / n

            # Elastic net update
            denom = col_norms_sq[j] + lambda * (1 - alpha)
            beta[j] = soft_threshold(rho_j, lambda * alpha) / denom

            resid = resid_j .- X_sc[:, j] .* beta[j]
        end

        if norm(beta - beta_old) < tol
            break
        end
    end

    beta_unscaled = beta ./ X_std
    intercept = y_mean - dot(X_mean, beta_unscaled)
    return vcat(intercept, beta_unscaled)
end

"""
    forward_stepwise(
        X::Matrix{Float64},
        y::Vector{Float64};
        max_features=nothing,
        criterion=:aic
    ) -> NamedTuple

Forward stepwise feature selection using AIC, BIC, or cross-val criterion.
"""
function forward_stepwise(
    X::Matrix{Float64},
    y::Vector{Float64};
    max_features::Union{Nothing,Int}=nothing,
    criterion::Symbol=:aic
)
    n, p = size(X)
    max_k = isnothing(max_features) ? min(p, n - 1) : min(max_features, p)

    selected = Int[]
    remaining = collect(1:p)
    scores = Float64[]
    feature_order = Int[]

    current_X = ones(n, 1)  # intercept only

    for k in 1:max_k
        best_feat = nothing
        best_score = Inf

        for feat in remaining
            X_trial = hcat(current_X, X[:, feat])
            score = _compute_criterion(X_trial, y, criterion)
            if score < best_score
                best_score = score
                best_feat = feat
            end
        end

        if isnothing(best_feat)
            break
        end

        # Check if adding feature improves criterion
        if k > 1 && best_score >= scores[end] - 0.1
            break  # No improvement
        end

        push!(selected, best_feat)
        push!(scores, best_score)
        push!(feature_order, best_feat)
        filter!(x -> x != best_feat, remaining)
        current_X = hcat(current_X, X[:, best_feat])
    end

    return (selected_features=selected, scores=scores, feature_order=feature_order)
end

function _compute_criterion(X::Matrix{Float64}, y::Vector{Float64}, criterion::Symbol)
    n, p = size(X)
    if n <= p
        return Inf
    end
    beta = X \ y
    y_hat = X * beta
    resid = y .- y_hat
    rss = sum(resid .^ 2)
    sigma2 = rss / (n - p)
    log_lik = -n/2 * log(2π * sigma2) - rss / (2 * sigma2)

    if criterion == :aic
        return -2 * log_lik + 2 * p
    elseif criterion == :bic
        return -2 * log_lik + p * log(n)
    else
        return rss  # RSS
    end
end

"""
    mutual_information_selection(
        X::Matrix{Float64},
        y::Vector{Float64};
        n_bins=20,
        top_k=10
    ) -> Vector{Int}

Feature selection via mutual information I(X_j; y).
Discretizes continuous variables using equal-frequency binning.
"""
function mutual_information_selection(
    X::Matrix{Float64},
    y::Vector{Float64};
    n_bins::Int=20,
    top_k::Int=10
)
    n, p = size(X)

    # Discretize y
    y_disc = discretize_equal_freq(y, n_bins)

    mi_scores = zeros(p)
    for j in 1:p
        x_disc = discretize_equal_freq(X[:, j], n_bins)
        mi_scores[j] = mutual_information(x_disc, y_disc, n_bins)
    end

    # Rank features by MI
    ranked = sortperm(mi_scores, rev=true)
    return ranked[1:min(top_k, p)]
end

function discretize_equal_freq(x::Vector{Float64}, n_bins::Int)
    n = length(x)
    sorted_x = sort(x)
    bin_size = n ÷ n_bins
    bins = zeros(Int, n)
    for i in eachindex(x)
        # Find rank
        rank = searchsortedfirst(sorted_x, x[i])
        bins[i] = clamp(div(rank - 1, max(bin_size, 1)) + 1, 1, n_bins)
    end
    return bins
end

function mutual_information(x_bins::Vector{Int}, y_bins::Vector{Int}, n_bins::Int)
    n = length(x_bins)
    # Joint distribution
    joint = zeros(n_bins, n_bins)
    for i in 1:n
        xi = clamp(x_bins[i], 1, n_bins)
        yi = clamp(y_bins[i], 1, n_bins)
        joint[xi, yi] += 1
    end
    joint ./= n

    px = vec(sum(joint, dims=2))
    py = vec(sum(joint, dims=1))

    mi = 0.0
    for i in 1:n_bins
        for j in 1:n_bins
            if joint[i, j] > 1e-10 && px[i] > 1e-10 && py[j] > 1e-10
                mi += joint[i, j] * log(joint[i, j] / (px[i] * py[j]))
            end
        end
    end
    return max(mi, 0.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

"""
    WalkForwardCV

Walk-forward cross-validation configuration.
"""
struct WalkForwardCV
    n_train::Int        # minimum training window
    n_test::Int         # test window size
    step::Int           # step size between validation windows
    expanding::Bool     # expanding window (true) vs rolling (false)
end

"""
    walk_forward_splits(n_obs::Int, wf::WalkForwardCV) -> Vector{Tuple}

Generate train/test index splits for walk-forward CV.
Returns vector of (train_idx, test_idx) tuples.
"""
function walk_forward_splits(n_obs::Int, wf::WalkForwardCV)
    splits = Tuple{Vector{Int},Vector{Int}}[]
    t = wf.n_train

    while t + wf.n_test <= n_obs
        test_start = t + 1
        test_end = min(t + wf.n_test, n_obs)

        if wf.expanding
            train_start = 1
        else
            train_start = max(1, t - wf.n_train + 1)
        end

        train_idx = collect(train_start:t)
        test_idx = collect(test_start:test_end)
        push!(splits, (train_idx, test_idx))
        t += wf.step
    end

    return splits
end

"""
    purged_kfold_cv(
        n_obs::Int,
        n_folds::Int;
        embargo_pct=0.01
    ) -> Vector{Tuple}

Purged K-fold cross-validation (López de Prado 2018).
Removes observations from training that are within embargo_pct * n_obs
of test period to prevent data leakage.
"""
function purged_kfold_cv(n_obs::Int, n_folds::Int; embargo_pct::Float64=0.01)
    embargo = round(Int, embargo_pct * n_obs)
    fold_size = n_obs ÷ n_folds
    splits = Tuple{Vector{Int},Vector{Int}}[]

    for fold in 1:n_folds
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == n_folds ? n_obs : fold * fold_size

        test_idx = collect(test_start:test_end)

        # Purge: remove train obs near test period
        purge_start = max(1, test_start - embargo)
        purge_end = min(n_obs, test_end + embargo)

        train_idx = [i for i in 1:n_obs
                     if !(purge_start <= i <= purge_end)]

        push!(splits, (train_idx, test_idx))
    end

    return splits
end

"""
    combinatorial_purged_cv(
        n_obs::Int,
        n_splits::Int,    # number of test groups
        n_test_groups::Int;  # groups used for testing per fold
        embargo_pct=0.01
    ) -> Vector{Tuple}

Combinatorial purged cross-validation (CPCV).
Creates C(n_splits, n_test_groups) unique train/test splits with purging.
"""
function combinatorial_purged_cv(
    n_obs::Int,
    n_splits::Int,
    n_test_groups::Int;
    embargo_pct::Float64=0.01
)
    embargo = round(Int, embargo_pct * n_obs)
    group_size = n_obs ÷ n_splits
    groups = [collect((g-1)*group_size+1 : min(g*group_size, n_obs)) for g in 1:n_splits]

    # All combinations of n_test_groups from n_splits
    all_combos = collect(Iterators.filter(c -> length(c) == n_test_groups,
        combinations(1:n_splits, n_test_groups)))

    splits = Tuple{Vector{Int},Vector{Int}}[]

    for test_combo in all_combos
        test_idx = vcat([groups[i] for i in test_combo]...)
        test_min, test_max = minimum(test_idx), maximum(test_idx)

        train_idx = [i for i in 1:n_obs
                     if !(test_min - embargo <= i <= test_max + embargo) &&
                        i ∉ test_idx]
        push!(splits, (train_idx, test_idx))
    end

    return splits
end

"""
    combinations(arr, k)

Generate all k-combinations from array. Simple recursive implementation.
"""
function combinations(arr, k)
    n = length(arr)
    if k == 0
        return [[]]
    end
    if k > n
        return []
    end
    result = []
    for (i, x) in enumerate(arr)
        for rest in combinations(arr[i+1:end], k-1)
            push!(result, vcat([x], rest))
        end
    end
    return result
end

"""
    cross_val_score(
        model_fn::Function,  # (X_train, y_train) -> model
        predict_fn::Function, # (model, X_test) -> y_pred
        X::Matrix{Float64},
        y::Vector{Float64},
        splits::Vector{Tuple};
        metric=:rmse
    ) -> Vector{Float64}

Evaluate a model across CV splits and return per-fold scores.
"""
function cross_val_score(
    model_fn::Function,
    predict_fn::Function,
    X::Matrix{Float64},
    y::Vector{Float64},
    splits::Vector{Tuple};
    metric::Symbol=:rmse
)
    scores = Float64[]

    for (train_idx, test_idx) in splits
        if length(train_idx) < 10 || isempty(test_idx)
            push!(scores, NaN)
            continue
        end

        model = try
            model_fn(X[train_idx, :], y[train_idx])
        catch
            push!(scores, NaN)
            continue
        end

        y_pred = predict_fn(model, X[test_idx, :])
        y_true = y[test_idx]

        score = if metric == :rmse
            sqrt(mean((y_pred .- y_true) .^ 2))
        elseif metric == :mae
            mean(abs.(y_pred .- y_true))
        elseif metric == :ic
            cor(y_pred, y_true)
        elseif metric == :r2
            ss_res = sum((y_true .- y_pred) .^ 2)
            ss_tot = sum((y_true .- mean(y_true)) .^ 2)
            ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0
        else
            sqrt(mean((y_pred .- y_true) .^ 2))
        end

        push!(scores, score)
    end

    return scores
end

# Import Optim since it's used in GP section
import Optim

end # module MachineLearning
