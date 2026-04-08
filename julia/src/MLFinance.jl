###############################################################################
# MLFinance.jl
#
# Machine learning for finance: linear models, trees, forests, GBT, KNN,
# Gaussian processes, PCA, clustering, HMM, neural networks, online learning,
# feature engineering, walk-forward CV, feature importance.
#
# Dependencies: LinearAlgebra, Statistics, Random  (stdlib only)
###############################################################################

module MLFinance

using LinearAlgebra, Statistics, Random

export ridge_regression, lasso_regression, elastic_net, logistic_regression
export DecisionTree, RandomForest, GradientBoostedTrees
export fit!, predict, predict_proba
export KNNClassifier, KNNRegressor, KDTree
export GaussianProcess, gp_fit, gp_predict
export pca_transform, kernel_pca, sparse_pca
export kmeans, hierarchical_clustering, dbscan
export HMM, baum_welch, viterbi, forward_backward
export NeuralNetwork, DenseLayer, relu, sigmoid_act, adam_update
export FTRLProximal, PassiveAggressive, HedgeAlgorithm
export rolling_features, technical_indicators, cross_sectional_rank
export walk_forward_cv, purged_walk_forward
export permutation_importance, tree_shap

# ─────────────────────────────────────────────────────────────────────────────
# §1  Linear Models
# ─────────────────────────────────────────────────────────────────────────────

"""
    ridge_regression(X, y; lambda=1.0) -> beta, intercept

Ridge regression (L2 regularization).
"""
function ridge_regression(X::AbstractMatrix{T}, y::AbstractVector{T};
                          lambda::T=T(1.0), fit_intercept::Bool=true) where T<:Real
    n, p = size(X)
    if fit_intercept
        X_aug = hcat(ones(T, n), X)
        reg = lambda .* Matrix{T}(I, p + 1, p + 1)
        reg[1, 1] = zero(T)  # don't regularize intercept
        beta = (X_aug' * X_aug .+ reg) \ (X_aug' * y)
        return beta[2:end], beta[1]
    else
        beta = (X' * X .+ lambda .* I(p)) \ (X' * y)
        return beta, zero(T)
    end
end

"""Ridge with GCV (generalized cross-validation) for lambda selection."""
function ridge_gcv(X::AbstractMatrix{T}, y::AbstractVector{T};
                   lambdas::AbstractVector{T}=T.(10.0 .^ range(-4, 4, length=50))) where T<:Real
    n, p = size(X)
    best_lambda = lambdas[1]
    best_gcv = T(Inf)
    for lam in lambdas
        X_aug = hcat(ones(T, n), X)
        H = X_aug * inv(X_aug' * X_aug .+ lam .* I(p + 1)) * X_aug'
        y_hat = H * y
        residuals = y .- y_hat
        trace_H = tr(H)
        gcv = sum(residuals .^ 2) / (n * (one(T) - trace_H / n)^2)
        if gcv < best_gcv
            best_gcv = gcv
            best_lambda = lam
        end
    end
    beta, intercept = ridge_regression(X, y; lambda=best_lambda)
    return beta, intercept, best_lambda
end

"""
    lasso_regression(X, y; lambda=1.0, max_iter=1000) -> beta, intercept

LASSO via coordinate descent.
"""
function lasso_regression(X::AbstractMatrix{T}, y::AbstractVector{T};
                          lambda::T=T(1.0), max_iter::Int=1000,
                          tol::T=T(1e-6), fit_intercept::Bool=true) where T<:Real
    n, p = size(X)
    beta = zeros(T, p)
    intercept = fit_intercept ? mean(y) : zero(T)
    residuals = y .- intercept
    # Precompute X'X diagonal and X'y
    xx_diag = vec(sum(X .^ 2; dims=1))
    for iter in 1:max_iter
        max_change = zero(T)
        for j in 1:p
            # Partial residual
            r_j = residuals .+ X[:, j] .* beta[j]
            rho = dot(X[:, j], r_j)
            # Soft-thresholding
            if abs(rho) <= lambda
                new_beta = zero(T)
            elseif rho > lambda
                new_beta = (rho - lambda) / (xx_diag[j] + T(1e-10))
            else
                new_beta = (rho + lambda) / (xx_diag[j] + T(1e-10))
            end
            delta = new_beta - beta[j]
            max_change = max(max_change, abs(delta))
            residuals .-= X[:, j] .* delta
            beta[j] = new_beta
        end
        if fit_intercept
            intercept = mean(y .- X * beta)
            residuals = y .- X * beta .- intercept
        end
        if max_change < tol
            break
        end
    end
    return beta, intercept
end

"""LASSO path: solutions for sequence of lambdas."""
function lasso_path(X::AbstractMatrix{T}, y::AbstractVector{T};
                    n_lambdas::Int=50, lambda_ratio::T=T(1e-3)) where T<:Real
    n, p = size(X)
    lambda_max = maximum(abs.(X' * (y .- mean(y)))) / n
    lambdas = lambda_max .* T.(10.0 .^ range(0, log10(lambda_ratio), length=n_lambdas))
    coefs = Matrix{T}(undef, p, n_lambdas)
    intercepts = Vector{T}(undef, n_lambdas)
    for (k, lam) in enumerate(lambdas)
        beta, intercept = lasso_regression(X, y; lambda=lam)
        coefs[:, k] = beta
        intercepts[k] = intercept
    end
    return lambdas, coefs, intercepts
end

"""
    elastic_net(X, y; lambda=1.0, alpha=0.5, max_iter=1000) -> beta, intercept

Elastic Net: alpha*L1 + (1-alpha)*L2.
"""
function elastic_net(X::AbstractMatrix{T}, y::AbstractVector{T};
                     lambda::T=T(1.0), alpha::T=T(0.5),
                     max_iter::Int=1000, tol::T=T(1e-6)) where T<:Real
    n, p = size(X)
    beta = zeros(T, p)
    intercept = mean(y)
    residuals = y .- intercept
    xx_diag = vec(sum(X .^ 2; dims=1))
    l1_weight = alpha * lambda
    l2_weight = (one(T) - alpha) * lambda
    for iter in 1:max_iter
        max_change = zero(T)
        for j in 1:p
            r_j = residuals .+ X[:, j] .* beta[j]
            rho = dot(X[:, j], r_j)
            denom = xx_diag[j] + l2_weight + T(1e-10)
            if abs(rho) <= l1_weight
                new_beta = zero(T)
            elseif rho > l1_weight
                new_beta = (rho - l1_weight) / denom
            else
                new_beta = (rho + l1_weight) / denom
            end
            delta = new_beta - beta[j]
            max_change = max(max_change, abs(delta))
            residuals .-= X[:, j] .* delta
            beta[j] = new_beta
        end
        intercept = mean(y .- X * beta)
        residuals = y .- X * beta .- intercept
        if max_change < tol
            break
        end
    end
    return beta, intercept
end

"""
    logistic_regression(X, y; kwargs...) -> beta, intercept

Logistic regression via SGD. y ∈ {0, 1}.
"""
function logistic_regression(X::AbstractMatrix{T}, y::AbstractVector{T};
                              lambda::T=T(0.01), lr::T=T(0.01),
                              max_iter::Int=1000, batch_size::Int=32,
                              rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n, p = size(X)
    beta = zeros(T, p)
    intercept = zero(T)
    for iter in 1:max_iter
        idx = rand(rng, 1:n, min(batch_size, n))
        X_batch = X[idx, :]
        y_batch = y[idx]
        z = X_batch * beta .+ intercept
        prob = one(T) ./ (one(T) .+ exp.(-z))
        err = prob .- y_batch
        grad_beta = X_batch' * err ./ length(idx) .+ lambda .* beta
        grad_intercept = mean(err)
        beta .-= lr .* grad_beta
        intercept -= lr * grad_intercept
    end
    return beta, intercept
end

"""Predict probabilities for logistic regression."""
function logistic_predict_proba(X::AbstractMatrix{T}, beta::AbstractVector{T},
                                 intercept::T) where T<:Real
    z = X * beta .+ intercept
    one(T) ./ (one(T) .+ exp.(-z))
end

"""Binary cross-entropy loss."""
function bce_loss(y_true::AbstractVector{T}, y_pred::AbstractVector{T}) where T<:Real
    eps = T(1e-12)
    -mean(y_true .* log.(y_pred .+ eps) .+ (one(T) .- y_true) .* log.(one(T) .- y_pred .+ eps))
end

# ─────────────────────────────────────────────────────────────────────────────
# §2  Decision Trees
# ─────────────────────────────────────────────────────────────────────────────

mutable struct TreeNode{T<:Real}
    feature::Int
    threshold::T
    value::T            # prediction at leaf
    left::Union{Nothing, TreeNode{T}}
    right::Union{Nothing, TreeNode{T}}
    n_samples::Int
    impurity::T
    is_leaf::Bool
end

TreeNode(T::Type=Float64) = TreeNode{T}(0, zero(T), zero(T), nothing, nothing, 0, zero(T), true)

struct DecisionTree{T<:Real}
    root::TreeNode{T}
    max_depth::Int
    min_samples_split::Int
    min_samples_leaf::Int
    max_features::Int
    criterion::Symbol  # :mse, :gini, :entropy
end

"""Gini impurity."""
function gini_impurity(y::AbstractVector{T}) where T<:Real
    n = length(y)
    if n == 0 return zero(T) end
    classes = unique(y)
    imp = one(T)
    for c in classes
        p = count(x -> x == c, y) / n
        imp -= p^2
    end
    imp
end

"""Entropy."""
function entropy_impurity(y::AbstractVector{T}) where T<:Real
    n = length(y)
    if n == 0 return zero(T) end
    classes = unique(y)
    ent = zero(T)
    for c in classes
        p = count(x -> x == c, y) / n
        if p > T(1e-16)
            ent -= p * log(p)
        end
    end
    ent
end

"""MSE impurity for regression."""
mse_impurity(y::AbstractVector{T}) where T<:Real = length(y) > 0 ? var(y) * length(y) : zero(T)

"""Find best split for a node."""
function _best_split(X::AbstractMatrix{T}, y::AbstractVector{T},
                     features::AbstractVector{Int},
                     criterion::Symbol,
                     min_samples_leaf::Int) where T<:Real
    n = length(y)
    best_gain = T(-Inf)
    best_feature = 0
    best_threshold = zero(T)
    impurity_func = criterion == :gini ? gini_impurity :
                    criterion == :entropy ? entropy_impurity : mse_impurity
    parent_impurity = impurity_func(y)
    for feat in features
        vals = X[:, feat]
        sorted_idx = sortperm(vals)
        sorted_vals = vals[sorted_idx]
        sorted_y = y[sorted_idx]
        for i in min_samples_leaf:n-min_samples_leaf
            if sorted_vals[i] == sorted_vals[min(i+1, n)]
                continue
            end
            threshold = (sorted_vals[i] + sorted_vals[min(i+1, n)]) / 2
            left_y = sorted_y[1:i]
            right_y = sorted_y[i+1:n]
            imp_left = impurity_func(left_y)
            imp_right = impurity_func(right_y)
            gain = parent_impurity - (T(length(left_y)) * imp_left + T(length(right_y)) * imp_right) / T(n)
            if gain > best_gain
                best_gain = gain
                best_feature = feat
                best_threshold = threshold
            end
        end
    end
    return best_feature, best_threshold, best_gain
end

"""Build tree recursively."""
function _build_tree(X::AbstractMatrix{T}, y::AbstractVector{T},
                     depth::Int, max_depth::Int,
                     min_samples_split::Int, min_samples_leaf::Int,
                     max_features::Int, criterion::Symbol,
                     rng::AbstractRNG) where T<:Real
    n, p = size(X)
    node = TreeNode(T)
    node.n_samples = n
    node.value = criterion in (:gini, :entropy) ? _mode(y) : mean(y)
    if n < min_samples_split || depth >= max_depth
        return node
    end
    n_feat = min(max_features, p)
    features = sort(randperm(rng, p)[1:n_feat])
    feat, thresh, gain = _best_split(X, y, features, criterion, min_samples_leaf)
    if feat == 0 || gain <= T(1e-10)
        return node
    end
    node.is_leaf = false
    node.feature = feat
    node.threshold = thresh
    left_mask = X[:, feat] .<= thresh
    right_mask = .!left_mask
    node.left = _build_tree(X[left_mask, :], y[left_mask], depth + 1,
                            max_depth, min_samples_split, min_samples_leaf,
                            max_features, criterion, rng)
    node.right = _build_tree(X[right_mask, :], y[right_mask], depth + 1,
                             max_depth, min_samples_split, min_samples_leaf,
                             max_features, criterion, rng)
    return node
end

function _mode(y::AbstractVector{T}) where T<:Real
    counts = Dict{T, Int}()
    for v in y
        counts[v] = get(counts, v, 0) + 1
    end
    findmax(counts)[2] |> k -> first(k for (k,v) in counts if v == findmax(counts)[1])
end

"""
    fit!(tree_config, X, y) -> DecisionTree

Train a decision tree.
"""
function fit_tree(X::AbstractMatrix{T}, y::AbstractVector{T};
                  max_depth::Int=10, min_samples_split::Int=5,
                  min_samples_leaf::Int=2, max_features::Int=0,
                  criterion::Symbol=:mse,
                  rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    p = size(X, 2)
    if max_features <= 0
        max_features = criterion in (:gini, :entropy) ? round(Int, sqrt(p)) : p
    end
    root = _build_tree(X, y, 0, max_depth, min_samples_split, min_samples_leaf,
                       max_features, criterion, rng)
    DecisionTree{T}(root, max_depth, min_samples_split, min_samples_leaf,
                    max_features, criterion)
end

"""Predict with a single tree."""
function predict_tree(tree::DecisionTree{T}, X::AbstractMatrix{T}) where T<:Real
    n = size(X, 1)
    preds = Vector{T}(undef, n)
    for i in 1:n
        preds[i] = _traverse(tree.root, X[i, :])
    end
    preds
end

function _traverse(node::TreeNode{T}, x::AbstractVector{T}) where T<:Real
    if node.is_leaf
        return node.value
    end
    if x[node.feature] <= node.threshold
        return _traverse(node.left, x)
    else
        return _traverse(node.right, x)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# §3  Random Forest
# ─────────────────────────────────────────────────────────────────────────────

struct RandomForest{T<:Real}
    trees::Vector{DecisionTree{T}}
    n_trees::Int
    max_depth::Int
    max_features::Int
    criterion::Symbol
    oob_score::T
end

"""
    fit_random_forest(X, y; n_trees=100, kwargs...) -> RandomForest

Train random forest with bagging.
"""
function fit_random_forest(X::AbstractMatrix{T}, y::AbstractVector{T};
                           n_trees::Int=100, max_depth::Int=10,
                           max_features::Int=0, criterion::Symbol=:mse,
                           min_samples_split::Int=5, min_samples_leaf::Int=2,
                           rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n, p = size(X)
    if max_features <= 0
        max_features = criterion in (:gini, :entropy) ? max(1, round(Int, sqrt(p))) : max(1, div(p, 3))
    end
    trees = Vector{DecisionTree{T}}(undef, n_trees)
    oob_preds = zeros(T, n)
    oob_counts = zeros(Int, n)
    for t in 1:n_trees
        # Bootstrap sample
        bag_idx = rand(rng, 1:n, n)
        oob_idx = setdiff(1:n, unique(bag_idx))
        tree = fit_tree(X[bag_idx, :], y[bag_idx];
                       max_depth=max_depth, max_features=max_features,
                       criterion=criterion, min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf, rng=rng)
        trees[t] = tree
        # OOB predictions
        if !isempty(oob_idx)
            oob_pred = predict_tree(tree, X[oob_idx, :])
            for (k, idx) in enumerate(oob_idx)
                oob_preds[idx] += oob_pred[k]
                oob_counts[idx] += 1
            end
        end
    end
    # OOB score
    valid = oob_counts .> 0
    oob_mean = oob_preds[valid] ./ oob_counts[valid]
    oob_score = if criterion in (:gini, :entropy)
        mean(round.(oob_mean) .== y[valid])
    else
        one(T) - sum((oob_mean .- y[valid]).^2) / max(sum((y[valid] .- mean(y[valid])).^2), T(1e-16))
    end
    RandomForest{T}(trees, n_trees, max_depth, max_features, criterion, oob_score)
end

"""Predict with random forest."""
function predict_forest(forest::RandomForest{T}, X::AbstractMatrix{T}) where T<:Real
    n = size(X, 1)
    preds = zeros(T, n)
    for tree in forest.trees
        preds .+= predict_tree(tree, X)
    end
    preds ./= forest.n_trees
    if forest.criterion in (:gini, :entropy)
        preds = round.(preds)
    end
    preds
end

"""Feature importance from random forest (impurity-based)."""
function feature_importance_rf(forest::RandomForest{T}, p::Int) where T<:Real
    importance = zeros(T, p)
    for tree in forest.trees
        _accumulate_importance!(importance, tree.root)
    end
    importance ./= forest.n_trees
    s = sum(importance)
    if s > T(1e-16)
        importance ./= s
    end
    importance
end

function _accumulate_importance!(importance::Vector{T}, node::TreeNode{T}) where T<:Real
    if node.is_leaf
        return
    end
    importance[node.feature] += node.n_samples * node.impurity
    if node.left !== nothing
        _accumulate_importance!(importance, node.left)
    end
    if node.right !== nothing
        _accumulate_importance!(importance, node.right)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# §4  Gradient Boosted Trees
# ─────────────────────────────────────────────────────────────────────────────

struct GradientBoostedTrees{T<:Real}
    trees::Vector{DecisionTree{T}}
    learning_rate::T
    initial_prediction::T
    n_trees::Int
    loss::Symbol  # :mse, :logloss
end

"""
    fit_gbt(X, y; n_trees=100, lr=0.1, max_depth=3, loss=:mse) -> GBT

Gradient boosted trees.
"""
function fit_gbt(X::AbstractMatrix{T}, y::AbstractVector{T};
                 n_trees::Int=100, learning_rate::T=T(0.1),
                 max_depth::Int=3, loss::Symbol=:mse,
                 subsample::T=T(0.8), col_subsample::T=T(0.8),
                 min_samples_leaf::Int=5,
                 rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n, p = size(X)
    initial_pred = loss == :mse ? mean(y) : log(mean(y) / (one(T) - mean(y) + T(1e-10)))
    pred = fill(initial_pred, n)
    trees = Vector{DecisionTree{T}}()
    for t in 1:n_trees
        # Compute pseudo-residuals
        if loss == :mse
            residuals = y .- pred
        elseif loss == :logloss
            prob = one(T) ./ (one(T) .+ exp.(-pred))
            residuals = y .- prob
        else
            residuals = y .- pred
        end
        # Subsample
        n_sub = max(1, round(Int, subsample * n))
        sub_idx = sort(randperm(rng, n)[1:n_sub])
        n_col = max(1, round(Int, col_subsample * p))
        tree = fit_tree(X[sub_idx, :], residuals[sub_idx];
                       max_depth=max_depth, criterion=:mse,
                       min_samples_leaf=min_samples_leaf,
                       max_features=n_col, rng=rng)
        push!(trees, tree)
        # Update predictions
        tree_pred = predict_tree(tree, X)
        pred .+= learning_rate .* tree_pred
    end
    GradientBoostedTrees{T}(trees, learning_rate, initial_pred, n_trees, loss)
end

"""Predict with GBT."""
function predict_gbt(model::GradientBoostedTrees{T}, X::AbstractMatrix{T}) where T<:Real
    n = size(X, 1)
    pred = fill(model.initial_prediction, n)
    for tree in model.trees
        pred .+= model.learning_rate .* predict_tree(tree, X)
    end
    if model.loss == :logloss
        pred = one(T) ./ (one(T) .+ exp.(-pred))
    end
    pred
end

"""GBT feature importance."""
function feature_importance_gbt(model::GradientBoostedTrees{T}, p::Int) where T<:Real
    importance = zeros(T, p)
    for tree in model.trees
        _accumulate_importance!(importance, tree.root)
    end
    s = sum(importance)
    if s > T(1e-16)
        importance ./= s
    end
    importance
end

# ─────────────────────────────────────────────────────────────────────────────
# §5  K-Nearest Neighbors with KD-Tree
# ─────────────────────────────────────────────────────────────────────────────

mutable struct KDNode{T<:Real}
    point::Vector{T}
    label::T
    index::Int
    split_dim::Int
    left::Union{Nothing, KDNode{T}}
    right::Union{Nothing, KDNode{T}}
end

struct KDTree{T<:Real}
    root::Union{Nothing, KDNode{T}}
    n_points::Int
    n_dims::Int
end

"""Build KD-tree from data."""
function build_kdtree(X::AbstractMatrix{T}, y::AbstractVector{T};
                      depth::Int=0) where T<:Real
    n, p = size(X)
    if n == 0
        return KDTree{T}(nothing, 0, p)
    end
    root = _build_kd(X, y, collect(1:n), 0, p)
    KDTree{T}(root, n, p)
end

function _build_kd(X::AbstractMatrix{T}, y::AbstractVector{T},
                   indices::Vector{Int}, depth::Int, ndims::Int) where T<:Real
    if isempty(indices)
        return nothing
    end
    split_dim = mod(depth, ndims) + 1
    sorted = sort(indices; by=i -> X[i, split_dim])
    mid = div(length(sorted), 2) + 1
    idx = sorted[mid]
    node = KDNode{T}(Vector{T}(X[idx, :]), y[idx], idx, split_dim, nothing, nothing)
    if mid > 1
        node.left = _build_kd(X, y, sorted[1:mid-1], depth + 1, ndims)
    end
    if mid < length(sorted)
        node.right = _build_kd(X, y, sorted[mid+1:end], depth + 1, ndims)
    end
    return node
end

"""K-nearest neighbor search in KD-tree."""
function knn_search(tree::KDTree{T}, query::AbstractVector{T}, k::Int) where T<:Real
    neighbors = Vector{Tuple{T, Int, T}}()  # (distance, index, label)
    _knn_search!(tree.root, query, k, neighbors)
    sort!(neighbors; by=x -> x[1])
    neighbors[1:min(k, length(neighbors))]
end

function _knn_search!(node::Union{Nothing, KDNode{T}}, query::AbstractVector{T},
                      k::Int, neighbors::Vector{Tuple{T, Int, T}}) where T<:Real
    if node === nothing
        return
    end
    dist = sqrt(sum((node.point .- query) .^ 2))
    if length(neighbors) < k || dist < neighbors[end][1]
        push!(neighbors, (dist, node.index, node.label))
        sort!(neighbors; by=x -> x[1])
        if length(neighbors) > k
            pop!(neighbors)
        end
    end
    diff = query[node.split_dim] - node.point[node.split_dim]
    close = diff < zero(T) ? node.left : node.right
    far = diff < zero(T) ? node.right : node.left
    _knn_search!(close, query, k, neighbors)
    max_dist = length(neighbors) < k ? T(Inf) : neighbors[end][1]
    if abs(diff) < max_dist
        _knn_search!(far, query, k, neighbors)
    end
end

struct KNNClassifier{T<:Real}
    tree::KDTree{T}
    k::Int
    X_train::Matrix{T}
    y_train::Vector{T}
end

struct KNNRegressor{T<:Real}
    tree::KDTree{T}
    k::Int
    X_train::Matrix{T}
    y_train::Vector{T}
end

"""Fit KNN classifier."""
function fit_knn_classifier(X::AbstractMatrix{T}, y::AbstractVector{T};
                            k::Int=5) where T<:Real
    tree = build_kdtree(X, y)
    KNNClassifier{T}(tree, k, Matrix{T}(X), Vector{T}(y))
end

"""Fit KNN regressor."""
function fit_knn_regressor(X::AbstractMatrix{T}, y::AbstractVector{T};
                           k::Int=5) where T<:Real
    tree = build_kdtree(X, y)
    KNNRegressor{T}(tree, k, Matrix{T}(X), Vector{T}(y))
end

"""Predict with KNN classifier."""
function predict_knn(model::KNNClassifier{T}, X::AbstractMatrix{T}) where T<:Real
    n = size(X, 1)
    preds = Vector{T}(undef, n)
    for i in 1:n
        neighbors = knn_search(model.tree, X[i, :], model.k)
        labels = [nb[3] for nb in neighbors]
        # Majority vote
        counts = Dict{T, Int}()
        for l in labels
            counts[l] = get(counts, l, 0) + 1
        end
        preds[i] = first(sort(collect(counts); by=x -> -x[2]))[1]
    end
    preds
end

"""Predict with KNN regressor."""
function predict_knn(model::KNNRegressor{T}, X::AbstractMatrix{T}) where T<:Real
    n = size(X, 1)
    preds = Vector{T}(undef, n)
    for i in 1:n
        neighbors = knn_search(model.tree, X[i, :], model.k)
        # Distance-weighted average
        total_w = zero(T)
        total_v = zero(T)
        for (d, _, label) in neighbors
            w = one(T) / (d + T(1e-10))
            total_w += w
            total_v += w * label
        end
        preds[i] = total_v / total_w
    end
    preds
end

# ─────────────────────────────────────────────────────────────────────────────
# §6  Gaussian Process Regression
# ─────────────────────────────────────────────────────────────────────────────

struct GaussianProcess{T<:Real}
    X_train::Matrix{T}
    y_train::Vector{T}
    alpha::Vector{T}  # K_inv * y
    L::LowerTriangular{T, Matrix{T}}
    length_scale::T
    signal_var::T
    noise_var::T
end

"""RBF (squared exponential) kernel."""
function rbf_kernel(x1::AbstractVector{T}, x2::AbstractVector{T};
                    length_scale::T=T(1.0), signal_var::T=T(1.0)) where T<:Real
    sq_dist = sum((x1 .- x2) .^ 2)
    signal_var * exp(-sq_dist / (T(2) * length_scale^2))
end

"""Build kernel matrix."""
function kernel_matrix(X::AbstractMatrix{T};
                       length_scale::T=T(1.0), signal_var::T=T(1.0),
                       noise_var::T=T(0.01)) where T<:Real
    n = size(X, 1)
    K = Matrix{T}(undef, n, n)
    for j in 1:n, i in j:n
        K[i,j] = rbf_kernel(X[i,:], X[j,:]; length_scale=length_scale, signal_var=signal_var)
        K[j,i] = K[i,j]
    end
    K .+= noise_var .* I(n)
    K
end

"""Cross-kernel matrix."""
function cross_kernel_matrix(X1::AbstractMatrix{T}, X2::AbstractMatrix{T};
                             length_scale::T=T(1.0), signal_var::T=T(1.0)) where T<:Real
    n1 = size(X1, 1)
    n2 = size(X2, 1)
    K = Matrix{T}(undef, n1, n2)
    for j in 1:n2, i in 1:n1
        K[i,j] = rbf_kernel(X1[i,:], X2[j,:]; length_scale=length_scale, signal_var=signal_var)
    end
    K
end

"""
    gp_fit(X, y; kwargs...) -> GaussianProcess

Fit Gaussian Process regression.
"""
function gp_fit(X::AbstractMatrix{T}, y::AbstractVector{T};
                length_scale::T=T(1.0), signal_var::T=T(1.0),
                noise_var::T=T(0.01),
                optimize_hyperparams::Bool=true,
                max_iter::Int=100) where T<:Real
    n = size(X, 1)
    if optimize_hyperparams
        length_scale, signal_var, noise_var = _optimize_gp_hyperparams(
            X, y, length_scale, signal_var, noise_var; max_iter=max_iter)
    end
    K = kernel_matrix(X; length_scale=length_scale, signal_var=signal_var, noise_var=noise_var)
    L = cholesky(Symmetric(K)).L
    alpha = L' \ (L \ y)
    GaussianProcess{T}(Matrix{T}(X), Vector{T}(y), alpha, L,
                       length_scale, signal_var, noise_var)
end

"""Optimize GP hyperparameters via marginal likelihood."""
function _optimize_gp_hyperparams(X::AbstractMatrix{T}, y::AbstractVector{T},
                                   ls::T, sv::T, nv::T;
                                   max_iter::Int=100, lr::T=T(0.01)) where T<:Real
    n = size(X, 1)
    log_ls = log(ls)
    log_sv = log(sv)
    log_nv = log(nv)
    for iter in 1:max_iter
        ls_curr = exp(log_ls)
        sv_curr = exp(log_sv)
        nv_curr = exp(log_nv)
        K = kernel_matrix(X; length_scale=ls_curr, signal_var=sv_curr, noise_var=nv_curr)
        try
            L = cholesky(Symmetric(K)).L
            alpha = L' \ (L \ y)
            # Log marginal likelihood
            lml = -T(0.5) * dot(y, alpha) - sum(log.(diag(L))) - T(n/2) * log(T(2π))
            # Numerical gradients
            delta = T(1e-4)
            for (param_ref, param_val) in [(Ref(log_ls), log_ls),
                                            (Ref(log_sv), log_sv),
                                            (Ref(log_nv), log_nv)]
                old = param_ref[]
                param_ref[] = old + delta
                K2 = kernel_matrix(X; length_scale=exp(log_ls), signal_var=exp(log_sv), noise_var=exp(log_nv))
                L2 = cholesky(Symmetric(K2)).L
                alpha2 = L2' \ (L2 \ y)
                lml2 = -T(0.5) * dot(y, alpha2) - sum(log.(diag(L2))) - T(n/2) * log(T(2π))
                grad = (lml2 - lml) / delta
                param_ref[] = old + lr * grad
            end
        catch
            break
        end
    end
    return exp(log_ls), exp(log_sv), exp(log_nv)
end

"""
    gp_predict(gp, X_test) -> mean, variance

Predict with GP.
"""
function gp_predict(gp::GaussianProcess{T}, X_test::AbstractMatrix{T}) where T<:Real
    K_star = cross_kernel_matrix(X_test, gp.X_train;
                                length_scale=gp.length_scale, signal_var=gp.signal_var)
    mu = K_star * gp.alpha
    v = gp.L \ K_star'
    K_star_star = [rbf_kernel(X_test[i,:], X_test[i,:];
                              length_scale=gp.length_scale, signal_var=gp.signal_var)
                   for i in 1:size(X_test, 1)]
    var_pred = K_star_star .- vec(sum(v .^ 2; dims=1))
    var_pred = max.(var_pred, T(1e-10))
    return mu, var_pred
end

# ─────────────────────────────────────────────────────────────────────────────
# §7  PCA and Kernel PCA
# ─────────────────────────────────────────────────────────────────────────────

"""
    pca_transform(X; n_components=2) -> Z, components, explained_var

PCA dimensionality reduction.
"""
function pca_transform(X::AbstractMatrix{T}; n_components::Int=2) where T<:Real
    n, p = size(X)
    mu = vec(mean(X; dims=1))
    X_centered = X .- mu'
    Sigma = X_centered' * X_centered / (n - 1)
    F = eigen(Symmetric(Sigma); sortby=x -> -x)
    k = min(n_components, p)
    components = F.vectors[:, 1:k]
    explained = F.values[1:k] ./ sum(F.values)
    Z = X_centered * components
    return Z, components, explained, mu
end

"""Reconstruct from PCA."""
function pca_reconstruct(Z::AbstractMatrix{T}, components::AbstractMatrix{T},
                          mu::AbstractVector{T}) where T<:Real
    Z * components' .+ mu'
end

"""
    kernel_pca(X; n_components=2, kernel=:rbf, gamma=1.0) -> Z

Kernel PCA.
"""
function kernel_pca(X::AbstractMatrix{T}; n_components::Int=2,
                    kernel::Symbol=:rbf, gamma::T=T(1.0)) where T<:Real
    n = size(X, 1)
    K = Matrix{T}(undef, n, n)
    if kernel == :rbf
        for j in 1:n, i in j:n
            K[i,j] = exp(-gamma * sum((X[i,:] .- X[j,:]).^2))
            K[j,i] = K[i,j]
        end
    elseif kernel == :poly
        for j in 1:n, i in j:n
            K[i,j] = (dot(X[i,:], X[j,:]) + one(T))^3
            K[j,i] = K[i,j]
        end
    end
    # Center kernel matrix
    one_n = fill(one(T) / n, n, n)
    K_centered = K .- one_n * K .- K * one_n .+ one_n * K * one_n
    F = eigen(Symmetric(K_centered); sortby=x -> -x)
    k = min(n_components, n)
    alphas = F.vectors[:, 1:k]
    for j in 1:k
        alphas[:, j] ./= sqrt(max(F.values[j], T(1e-16)))
    end
    Z = K_centered * alphas
    return Z
end

"""
    sparse_pca(X; n_components=2, alpha=1.0) -> components

Sparse PCA via elastic net on components.
"""
function sparse_pca(X::AbstractMatrix{T}; n_components::Int=2,
                    sparsity::T=T(1.0), max_iter::Int=100) where T<:Real
    n, p = size(X)
    mu = vec(mean(X; dims=1))
    X_c = X .- mu'
    _, V_init, _ = pca_transform(X; n_components=n_components)
    V = V_init[:, 1:min(n_components, size(V_init, 2))]
    k = size(V, 2)
    for iter in 1:max_iter
        Z = X_c * V
        # Update V via elastic net
        for j in 1:k
            target = X_c' * Z[:, j]
            beta, _ = lasso_regression(X_c' * X_c, target; lambda=sparsity)
            nrm = norm(beta)
            if nrm > T(1e-16)
                V[:, j] = beta ./ nrm
            end
        end
    end
    return V, mu
end

# ─────────────────────────────────────────────────────────────────────────────
# §8  Clustering
# ─────────────────────────────────────────────────────────────────────────────

"""
    kmeans(X, k; max_iter=100) -> labels, centers, inertia

K-means clustering.
"""
function kmeans(X::AbstractMatrix{T}, k::Int;
                max_iter::Int=100, n_init::Int=10,
                rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n, p = size(X)
    best_inertia = T(Inf)
    best_labels = zeros(Int, n)
    best_centers = zeros(T, k, p)
    for init in 1:n_init
        # K-means++ initialization
        centers = Matrix{T}(undef, k, p)
        idx = rand(rng, 1:n)
        centers[1, :] = X[idx, :]
        for c in 2:k
            dists = [minimum(sum((X[i,:] .- centers[j,:]).^2) for j in 1:c-1) for i in 1:n]
            probs = dists ./ (sum(dists) + T(1e-16))
            cumprobs = cumsum(probs)
            r = rand(rng, T)
            idx = findfirst(x -> x >= r, cumprobs)
            idx = idx === nothing ? n : idx
            centers[c, :] = X[idx, :]
        end
        labels = zeros(Int, n)
        for iter in 1:max_iter
            # Assign
            new_labels = zeros(Int, n)
            for i in 1:n
                min_dist = T(Inf)
                for c in 1:k
                    d = sum((X[i,:] .- centers[c,:]).^2)
                    if d < min_dist
                        min_dist = d
                        new_labels[i] = c
                    end
                end
            end
            if new_labels == labels
                break
            end
            labels = new_labels
            # Update centers
            for c in 1:k
                members = findall(labels .== c)
                if !isempty(members)
                    centers[c, :] = vec(mean(X[members, :]; dims=1))
                end
            end
        end
        inertia = sum(sum((X[i,:] .- centers[labels[i],:]).^2) for i in 1:n)
        if inertia < best_inertia
            best_inertia = inertia
            best_labels = labels
            best_centers = centers
        end
    end
    return best_labels, best_centers, best_inertia
end

"""Silhouette score."""
function silhouette_score(X::AbstractMatrix{T}, labels::AbstractVector{Int}) where T<:Real
    n = size(X, 1)
    scores = Vector{T}(undef, n)
    for i in 1:n
        # a(i): mean distance to same cluster
        same_cluster = findall(labels .== labels[i])
        if length(same_cluster) <= 1
            scores[i] = zero(T)
            continue
        end
        a_i = mean(sqrt(sum((X[i,:] .- X[j,:]).^2)) for j in same_cluster if j != i)
        # b(i): min mean distance to other clusters
        b_i = T(Inf)
        for c in unique(labels)
            if c == labels[i] continue end
            other = findall(labels .== c)
            if !isempty(other)
                mean_dist = mean(sqrt(sum((X[i,:] .- X[j,:]).^2)) for j in other)
                b_i = min(b_i, mean_dist)
            end
        end
        scores[i] = (b_i - a_i) / max(a_i, b_i, T(1e-16))
    end
    mean(scores)
end

"""
    hierarchical_clustering(X; n_clusters=3, linkage=:ward) -> labels

Agglomerative hierarchical clustering.
"""
function hierarchical_clustering(X::AbstractMatrix{T};
                                  n_clusters::Int=3,
                                  linkage::Symbol=:ward) where T<:Real
    n = size(X, 1)
    labels = collect(1:n)
    clusters = Dict{Int, Vector{Int}}(i => [i] for i in 1:n)
    # Distance matrix
    D = Matrix{T}(undef, n, n)
    for j in 1:n, i in j:n
        D[i,j] = sqrt(sum((X[i,:] .- X[j,:]).^2))
        D[j,i] = D[i,j]
    end
    for i in 1:n D[i,i] = T(Inf) end
    active = Set(1:n)
    next_id = n + 1
    while length(active) > n_clusters
        # Find closest pair
        min_dist = T(Inf)
        mi, mj = 0, 0
        for i in active, j in active
            if i < j && D[i,j] < min_dist
                min_dist = D[i,j]
                mi, mj = i, j
            end
        end
        if mi == 0 break end
        # Merge
        new_cluster = vcat(clusters[mi], clusters[mj])
        delete!(active, mj)
        clusters[mi] = new_cluster
        delete!(clusters, mj)
        # Update distances
        for k in active
            if k == mi continue end
            if linkage == :single
                D[mi,k] = D[k,mi] = min(D[mi,k], D[mj,k])
            elseif linkage == :complete
                D[mi,k] = D[k,mi] = max(D[mi,k], D[mj,k])
            elseif linkage == :average
                n1 = length(clusters[mi]) - length(get(clusters, mj, Int[]))
                n2_orig = length(get(clusters, mj, Int[]))
                if n2_orig == 0 n2_orig = 1 end
                D[mi,k] = D[k,mi] = (D[mi,k] * n1 + D[mj,k] * n2_orig) / (n1 + n2_orig)
            elseif linkage == :ward
                nk = length(get(clusters, k, [k]))
                ni = length(clusters[mi])
                nj_orig = 0
                D[mi,k] = D[k,mi] = sqrt(((ni + nk) * D[mi,k]^2 +
                           (nj_orig + nk) * D[mj,k]^2 - nk * min_dist^2) /
                           max(ni + nj_orig + nk, 1))
            end
        end
        D[mj, :] .= T(Inf)
        D[:, mj] .= T(Inf)
    end
    # Assign labels
    result_labels = zeros(Int, n)
    for (label, (_, members)) in enumerate(clusters)
        for m in members
            result_labels[m] = label
        end
    end
    result_labels
end

"""
    dbscan(X; eps=0.5, min_pts=5) -> labels

DBSCAN density-based clustering.
"""
function dbscan(X::AbstractMatrix{T}; eps::T=T(0.5), min_pts::Int=5) where T<:Real
    n = size(X, 1)
    labels = zeros(Int, n)  # 0 = unvisited, -1 = noise
    cluster_id = 0
    for i in 1:n
        if labels[i] != 0
            continue
        end
        neighbors = _range_query(X, i, eps)
        if length(neighbors) < min_pts
            labels[i] = -1  # noise
            continue
        end
        cluster_id += 1
        labels[i] = cluster_id
        seed_set = copy(neighbors)
        k = 1
        while k <= length(seed_set)
            q = seed_set[k]
            if labels[q] == -1
                labels[q] = cluster_id
            end
            if labels[q] != 0
                k += 1
                continue
            end
            labels[q] = cluster_id
            q_neighbors = _range_query(X, q, eps)
            if length(q_neighbors) >= min_pts
                for nn in q_neighbors
                    if !(nn in seed_set)
                        push!(seed_set, nn)
                    end
                end
            end
            k += 1
        end
    end
    labels
end

function _range_query(X::AbstractMatrix{T}, idx::Int, eps::T) where T<:Real
    n = size(X, 1)
    neighbors = Int[]
    for j in 1:n
        if j == idx continue end
        d = sqrt(sum((X[idx,:] .- X[j,:]).^2))
        if d <= eps
            push!(neighbors, j)
        end
    end
    neighbors
end

# ─────────────────────────────────────────────────────────────────────────────
# §9  Hidden Markov Model
# ─────────────────────────────────────────────────────────────────────────────

struct HMM{T<:Real}
    n_states::Int
    A::Matrix{T}       # transition probabilities
    mu::Vector{T}      # emission means (Gaussian)
    sigma::Vector{T}   # emission stds
    pi0::Vector{T}     # initial state distribution
end

function HMM(n_states::Int, T::Type=Float64)
    A = fill(one(T)/n_states, n_states, n_states)
    mu = randn(T, n_states)
    sigma = ones(T, n_states)
    pi0 = fill(one(T)/n_states, n_states)
    HMM{T}(n_states, A, mu, sigma, pi0)
end

"""Gaussian emission probability."""
function _emission(x::T, mu::T, sigma::T) where T<:Real
    exp(-(x - mu)^2 / (T(2) * sigma^2)) / (sigma * sqrt(T(2π)))
end

"""
    forward_backward(hmm, observations) -> alpha, beta, gamma, xi, log_likelihood

Forward-backward algorithm.
"""
function forward_backward(hmm::HMM{T}, obs::AbstractVector{T}) where T<:Real
    N = hmm.n_states
    T_len = length(obs)
    # Forward
    alpha = Matrix{T}(undef, T_len, N)
    scaling = Vector{T}(undef, T_len)
    for j in 1:N
        alpha[1, j] = hmm.pi0[j] * _emission(obs[1], hmm.mu[j], hmm.sigma[j])
    end
    scaling[1] = sum(alpha[1, :])
    alpha[1, :] ./= max(scaling[1], T(1e-300))
    for t in 2:T_len
        for j in 1:N
            alpha[t, j] = sum(alpha[t-1, i] * hmm.A[i, j] for i in 1:N) *
                          _emission(obs[t], hmm.mu[j], hmm.sigma[j])
        end
        scaling[t] = sum(alpha[t, :])
        alpha[t, :] ./= max(scaling[t], T(1e-300))
    end
    # Backward
    beta = Matrix{T}(undef, T_len, N)
    beta[T_len, :] .= one(T)
    for t in T_len-1:-1:1
        for i in 1:N
            beta[t, i] = sum(hmm.A[i, j] * _emission(obs[t+1], hmm.mu[j], hmm.sigma[j]) *
                            beta[t+1, j] for j in 1:N)
        end
        beta[t, :] ./= max(scaling[t+1], T(1e-300))
    end
    # Gamma and Xi
    gamma = Matrix{T}(undef, T_len, N)
    for t in 1:T_len
        denom = sum(alpha[t, j] * beta[t, j] for j in 1:N)
        for j in 1:N
            gamma[t, j] = alpha[t, j] * beta[t, j] / max(denom, T(1e-300))
        end
    end
    xi = Array{T}(undef, T_len - 1, N, N)
    for t in 1:T_len-1
        denom = zero(T)
        for i in 1:N, j in 1:N
            denom += alpha[t, i] * hmm.A[i, j] * _emission(obs[t+1], hmm.mu[j], hmm.sigma[j]) * beta[t+1, j]
        end
        for i in 1:N, j in 1:N
            xi[t, i, j] = alpha[t, i] * hmm.A[i, j] *
                          _emission(obs[t+1], hmm.mu[j], hmm.sigma[j]) * beta[t+1, j] /
                          max(denom, T(1e-300))
        end
    end
    log_likelihood = sum(log.(max.(scaling, T(1e-300))))
    return alpha, beta, gamma, xi, log_likelihood
end

"""
    baum_welch(hmm, observations; max_iter=100) -> trained_hmm, log_likelihoods

Baum-Welch EM algorithm for HMM training.
"""
function baum_welch(hmm::HMM{T}, obs::AbstractVector{T};
                    max_iter::Int=100, tol::T=T(1e-6)) where T<:Real
    N = hmm.n_states
    T_len = length(obs)
    A = copy(hmm.A)
    mu = copy(hmm.mu)
    sigma = copy(hmm.sigma)
    pi0 = copy(hmm.pi0)
    log_liks = T[]
    prev_ll = T(-Inf)
    for iter in 1:max_iter
        current_hmm = HMM{T}(N, A, mu, sigma, pi0)
        _, _, gamma, xi, ll = forward_backward(current_hmm, obs)
        push!(log_liks, ll)
        if abs(ll - prev_ll) < tol
            break
        end
        prev_ll = ll
        # Update parameters
        pi0 = gamma[1, :]
        pi0 ./= max(sum(pi0), T(1e-300))
        for i in 1:N
            for j in 1:N
                A[i, j] = sum(xi[t, i, j] for t in 1:T_len-1) /
                          max(sum(gamma[t, i] for t in 1:T_len-1), T(1e-300))
            end
            row_sum = sum(A[i, :])
            A[i, :] ./= max(row_sum, T(1e-300))
        end
        for j in 1:N
            gamma_sum = sum(gamma[:, j])
            mu[j] = sum(gamma[t, j] * obs[t] for t in 1:T_len) / max(gamma_sum, T(1e-300))
            sigma[j] = sqrt(sum(gamma[t, j] * (obs[t] - mu[j])^2 for t in 1:T_len) /
                           max(gamma_sum, T(1e-300)))
            sigma[j] = max(sigma[j], T(1e-6))
        end
    end
    trained_hmm = HMM{T}(N, A, mu, sigma, pi0)
    return trained_hmm, log_liks
end

"""
    viterbi(hmm, observations) -> best_path, log_prob

Viterbi algorithm for most likely state sequence.
"""
function viterbi(hmm::HMM{T}, obs::AbstractVector{T}) where T<:Real
    N = hmm.n_states
    T_len = length(obs)
    delta = Matrix{T}(undef, T_len, N)
    psi = Matrix{Int}(undef, T_len, N)
    for j in 1:N
        delta[1, j] = log(max(hmm.pi0[j], T(1e-300))) +
                       log(max(_emission(obs[1], hmm.mu[j], hmm.sigma[j]), T(1e-300)))
        psi[1, j] = 0
    end
    for t in 2:T_len
        for j in 1:N
            vals = [delta[t-1, i] + log(max(hmm.A[i, j], T(1e-300))) for i in 1:N]
            best_i = argmax(vals)
            delta[t, j] = vals[best_i] + log(max(_emission(obs[t], hmm.mu[j], hmm.sigma[j]), T(1e-300)))
            psi[t, j] = best_i
        end
    end
    # Backtrack
    path = Vector{Int}(undef, T_len)
    path[T_len] = argmax(delta[T_len, :])
    log_prob = delta[T_len, path[T_len]]
    for t in T_len-1:-1:1
        path[t] = psi[t+1, path[t+1]]
    end
    return path, log_prob
end

# ─────────────────────────────────────────────────────────────────────────────
# §10  Neural Network
# ─────────────────────────────────────────────────────────────────────────────

"""Activation functions."""
relu(x::T) where T<:Real = max(x, zero(T))
relu_deriv(x::T) where T<:Real = x > zero(T) ? one(T) : zero(T)
sigmoid_act(x::T) where T<:Real = one(T) / (one(T) + exp(-x))
sigmoid_deriv(x::T) where T<:Real = sigmoid_act(x) * (one(T) - sigmoid_act(x))
tanh_act(x::T) where T<:Real = tanh(x)
tanh_deriv(x::T) where T<:Real = one(T) - tanh(x)^2

mutable struct DenseLayer{T<:Real}
    W::Matrix{T}
    b::Vector{T}
    activation::Symbol
    # Adam state
    m_W::Matrix{T}
    v_W::Matrix{T}
    m_b::Vector{T}
    v_b::Vector{T}
end

function DenseLayer(n_in::Int, n_out::Int; activation::Symbol=:relu,
                    T::Type=Float64, rng::AbstractRNG=Random.GLOBAL_RNG)
    # Xavier initialization
    scale = sqrt(T(2) / (n_in + n_out))
    W = randn(rng, T, n_out, n_in) .* scale
    b = zeros(T, n_out)
    DenseLayer{T}(W, b, activation,
                  zeros(T, n_out, n_in), zeros(T, n_out, n_in),
                  zeros(T, n_out), zeros(T, n_out))
end

struct NeuralNetwork{T<:Real}
    layers::Vector{DenseLayer{T}}
    loss::Symbol  # :mse, :bce
end

function NeuralNetwork(layer_sizes::Vector{Int};
                       activations::Vector{Symbol}=Symbol[],
                       loss::Symbol=:mse, T::Type=Float64,
                       rng::AbstractRNG=Random.GLOBAL_RNG)
    n_layers = length(layer_sizes) - 1
    if isempty(activations)
        activations = fill(:relu, n_layers)
        activations[end] = loss == :bce ? :sigmoid : :linear
    end
    layers = [DenseLayer(layer_sizes[i], layer_sizes[i+1];
                         activation=activations[i], T=T, rng=rng)
              for i in 1:n_layers]
    NeuralNetwork{T}(layers, loss)
end

"""Apply activation function."""
function _activate(x::AbstractVector{T}, act::Symbol) where T<:Real
    if act == :relu
        relu.(x)
    elseif act == :sigmoid
        sigmoid_act.(x)
    elseif act == :tanh
        tanh_act.(x)
    else
        x
    end
end

function _activate_deriv(x::AbstractVector{T}, act::Symbol) where T<:Real
    if act == :relu
        relu_deriv.(x)
    elseif act == :sigmoid
        sigmoid_deriv.(x)
    elseif act == :tanh
        tanh_deriv.(x)
    else
        ones(T, length(x))
    end
end

"""Forward pass."""
function forward(nn::NeuralNetwork{T}, x::AbstractVector{T}) where T<:Real
    activations = Vector{Vector{T}}()
    pre_activations = Vector{Vector{T}}()
    push!(activations, x)
    h = x
    for layer in nn.layers
        z = layer.W * h .+ layer.b
        push!(pre_activations, z)
        h = _activate(z, layer.activation)
        push!(activations, h)
    end
    return activations, pre_activations
end

"""Backpropagation."""
function backward(nn::NeuralNetwork{T}, x::AbstractVector{T},
                  y::AbstractVector{T}) where T<:Real
    activations, pre_activations = forward(nn, x)
    n_layers = length(nn.layers)
    # Output error
    output = activations[end]
    if nn.loss == :mse
        delta = (output .- y) .* _activate_deriv(pre_activations[end], nn.layers[end].activation)
    elseif nn.loss == :bce
        delta = output .- y  # sigmoid + BCE simplifies
    else
        delta = (output .- y)
    end
    grad_W = Vector{Matrix{T}}(undef, n_layers)
    grad_b = Vector{Vector{T}}(undef, n_layers)
    grad_W[end] = delta * activations[end-1]'
    grad_b[end] = delta
    for l in n_layers-1:-1:1
        delta = (nn.layers[l+1].W' * delta) .* _activate_deriv(pre_activations[l], nn.layers[l].activation)
        grad_W[l] = delta * activations[l]'
        grad_b[l] = delta
    end
    return grad_W, grad_b
end

"""
    adam_update(layer, grad_W, grad_b, t; lr=0.001, beta1=0.9, beta2=0.999) -> updated layer

Adam optimizer step.
"""
function adam_update!(layer::DenseLayer{T}, grad_W::AbstractMatrix{T},
                     grad_b::AbstractVector{T}, t::Int;
                     lr::T=T(0.001), beta1::T=T(0.9), beta2::T=T(0.999),
                     eps::T=T(1e-8), weight_decay::T=T(0.0)) where T<:Real
    layer.m_W = beta1 .* layer.m_W .+ (one(T) - beta1) .* grad_W
    layer.v_W = beta2 .* layer.v_W .+ (one(T) - beta2) .* grad_W .^ 2
    layer.m_b = beta1 .* layer.m_b .+ (one(T) - beta1) .* grad_b
    layer.v_b = beta2 .* layer.v_b .+ (one(T) - beta2) .* grad_b .^ 2
    m_hat_W = layer.m_W ./ (one(T) - beta1^t)
    v_hat_W = layer.v_W ./ (one(T) - beta2^t)
    m_hat_b = layer.m_b ./ (one(T) - beta1^t)
    v_hat_b = layer.v_b ./ (one(T) - beta2^t)
    layer.W .-= lr .* (m_hat_W ./ (sqrt.(v_hat_W) .+ eps) .+ weight_decay .* layer.W)
    layer.b .-= lr .* m_hat_b ./ (sqrt.(v_hat_b) .+ eps)
end

"""Train neural network."""
function train_nn!(nn::NeuralNetwork{T}, X::AbstractMatrix{T}, y::AbstractMatrix{T};
                   epochs::Int=100, lr::T=T(0.001), batch_size::Int=32,
                   weight_decay::T=T(0.0),
                   rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n = size(X, 1)
    losses = T[]
    for epoch in 1:epochs
        # Shuffle
        perm = randperm(rng, n)
        epoch_loss = zero(T)
        n_batches = 0
        for batch_start in 1:batch_size:n
            batch_end = min(batch_start + batch_size - 1, n)
            batch_idx = perm[batch_start:batch_end]
            bs = length(batch_idx)
            # Accumulate gradients
            n_layers = length(nn.layers)
            acc_grad_W = [zeros(T, size(l.W)) for l in nn.layers]
            acc_grad_b = [zeros(T, size(l.b)) for l in nn.layers]
            for i in batch_idx
                gW, gb = backward(nn, X[i, :], y[i, :])
                for l in 1:n_layers
                    acc_grad_W[l] .+= gW[l] ./ bs
                    acc_grad_b[l] .+= gb[l] ./ bs
                end
                # Loss
                out = forward(nn, X[i, :])[1][end]
                if nn.loss == :mse
                    epoch_loss += sum((out .- y[i, :]).^2) / 2
                elseif nn.loss == :bce
                    epoch_loss -= sum(y[i,:] .* log.(out .+ T(1e-12)) .+
                                    (one(T) .- y[i,:]) .* log.(one(T) .- out .+ T(1e-12)))
                end
            end
            t = (epoch - 1) * div(n, batch_size) + div(batch_start, batch_size) + 1
            for l in 1:n_layers
                adam_update!(nn.layers[l], acc_grad_W[l], acc_grad_b[l], t;
                            lr=lr, weight_decay=weight_decay)
            end
            n_batches += 1
        end
        push!(losses, epoch_loss / n)
    end
    losses
end

"""Predict with neural network."""
function predict_nn(nn::NeuralNetwork{T}, X::AbstractMatrix{T}) where T<:Real
    n = size(X, 1)
    out_size = size(nn.layers[end].W, 1)
    preds = Matrix{T}(undef, n, out_size)
    for i in 1:n
        activations, _ = forward(nn, X[i, :])
        preds[i, :] = activations[end]
    end
    preds
end

# ─────────────────────────────────────────────────────────────────────────────
# §11  Online Learning
# ─────────────────────────────────────────────────────────────────────────────

"""
    FTRLProximal: Follow The Regularized Leader with proximal term.
"""
mutable struct FTRLProximal{T<:Real}
    z::Vector{T}
    n_sq::Vector{T}  # sum of squared gradients
    alpha::T
    beta::T
    lambda1::T
    lambda2::T
    w::Vector{T}
end

function FTRLProximal(p::Int; alpha::Float64=0.1, beta::Float64=1.0,
                      lambda1::Float64=0.1, lambda2::Float64=0.1)
    FTRLProximal{Float64}(zeros(p), zeros(p), alpha, beta, lambda1, lambda2, zeros(p))
end

"""Update FTRL with new observation."""
function ftrl_update!(model::FTRLProximal{T}, x::AbstractVector{T}, y::T) where T<:Real
    p = length(model.z)
    # Compute prediction
    for i in 1:p
        if abs(model.z[i]) <= model.lambda1
            model.w[i] = zero(T)
        else
            sign_z = model.z[i] > zero(T) ? one(T) : -one(T)
            model.w[i] = -(model.z[i] - sign_z * model.lambda1) /
                          ((model.beta + sqrt(model.n_sq[i])) / model.alpha + model.lambda2)
        end
    end
    pred = dot(model.w, x)
    # Gradient
    grad = pred - y
    for i in 1:p
        g = grad * x[i]
        sigma = (sqrt(model.n_sq[i] + g^2) - sqrt(model.n_sq[i])) / model.alpha
        model.z[i] += g - sigma * model.w[i]
        model.n_sq[i] += g^2
    end
    return pred
end

"""Passive-Aggressive algorithm."""
mutable struct PassiveAggressive{T<:Real}
    w::Vector{T}
    C::T  # aggressiveness parameter
end

PassiveAggressive(p::Int; C::Float64=1.0) = PassiveAggressive{Float64}(zeros(p), C)

function pa_update!(model::PassiveAggressive{T}, x::AbstractVector{T}, y::T) where T<:Real
    pred = dot(model.w, x)
    loss = max(zero(T), one(T) - y * pred)  # hinge loss
    if loss > zero(T)
        tau = min(model.C, loss / (dot(x, x) + T(1e-10)))
        model.w .+= tau .* y .* x
    end
    return pred
end

"""Hedge algorithm for expert aggregation."""
mutable struct HedgeAlgorithm{T<:Real}
    weights::Vector{T}
    eta::T  # learning rate
    n_experts::Int
end

function HedgeAlgorithm(n_experts::Int; eta::Float64=0.1)
    HedgeAlgorithm{Float64}(fill(1.0 / n_experts, n_experts), eta, n_experts)
end

"""Update Hedge weights based on expert losses."""
function hedge_update!(model::HedgeAlgorithm{T},
                       losses::AbstractVector{T}) where T<:Real
    model.weights .*= exp.(-model.eta .* losses)
    total = sum(model.weights)
    model.weights ./= max(total, T(1e-300))
    return copy(model.weights)
end

"""Hedge prediction: weighted combination of expert predictions."""
function hedge_predict(model::HedgeAlgorithm{T},
                       expert_preds::AbstractVector{T}) where T<:Real
    dot(model.weights, expert_preds)
end

# ─────────────────────────────────────────────────────────────────────────────
# §12  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

"""
    rolling_features(prices; windows=[5,10,21,63]) -> feature_matrix

Compute rolling statistical features.
"""
function rolling_features(prices::AbstractVector{T};
                          windows::Vector{Int}=[5, 10, 21, 63]) where T<:Real
    n = length(prices)
    returns = zeros(T, n)
    for t in 2:n
        returns[t] = (prices[t] - prices[t-1]) / prices[t-1]
    end
    n_features = length(windows) * 6  # mean, std, skew, kurt, min, max per window
    features = zeros(T, n, n_features)
    for (wi, w) in enumerate(windows)
        offset = (wi - 1) * 6
        for t in w:n
            r = returns[t-w+1:t]
            features[t, offset + 1] = mean(r)
            features[t, offset + 2] = std(r)
            mu = mean(r)
            s = std(r)
            if s > T(1e-16)
                features[t, offset + 3] = sum((r .- mu).^3) / (length(r) * s^3)
                features[t, offset + 4] = sum((r .- mu).^4) / (length(r) * s^4) - T(3)
            end
            features[t, offset + 5] = minimum(r)
            features[t, offset + 6] = maximum(r)
        end
    end
    features
end

"""
    technical_indicators(prices, volumes) -> feature_matrix

Compute common technical indicators.
"""
function technical_indicators(prices::AbstractVector{T};
                               volumes::Union{Nothing, AbstractVector{T}}=nothing) where T<:Real
    n = length(prices)
    features = Dict{String, Vector{T}}()
    # Returns
    returns = zeros(T, n)
    for t in 2:n
        returns[t] = (prices[t] - prices[t-1]) / prices[t-1]
    end
    features["return"] = returns
    # SMA
    for w in [5, 10, 20, 50, 200]
        sma = zeros(T, n)
        for t in w:n
            sma[t] = mean(prices[t-w+1:t])
        end
        features["sma_$w"] = sma
        features["price_sma_ratio_$w"] = prices ./ max.(sma, T(1e-10))
    end
    # EMA
    for w in [12, 26]
        ema = zeros(T, n)
        alpha = T(2) / (w + 1)
        ema[1] = prices[1]
        for t in 2:n
            ema[t] = alpha * prices[t] + (one(T) - alpha) * ema[t-1]
        end
        features["ema_$w"] = ema
    end
    # MACD
    ema12 = features["ema_12"]
    ema26 = features["ema_26"]
    macd = ema12 .- ema26
    signal = zeros(T, n)
    alpha_s = T(2) / 10
    signal[1] = macd[1]
    for t in 2:n
        signal[t] = alpha_s * macd[t] + (one(T) - alpha_s) * signal[t-1]
    end
    features["macd"] = macd
    features["macd_signal"] = signal
    features["macd_hist"] = macd .- signal
    # RSI
    rsi = zeros(T, n)
    w_rsi = 14
    for t in w_rsi+1:n
        gains = sum(max(returns[i], zero(T)) for i in t-w_rsi+1:t)
        losses = sum(max(-returns[i], zero(T)) for i in t-w_rsi+1:t)
        rs = gains / max(losses, T(1e-10))
        rsi[t] = T(100) - T(100) / (one(T) + rs)
    end
    features["rsi"] = rsi
    # Bollinger Bands
    bb_w = 20
    bb_upper = zeros(T, n)
    bb_lower = zeros(T, n)
    bb_pctb = zeros(T, n)
    for t in bb_w:n
        mu = mean(prices[t-bb_w+1:t])
        s = std(prices[t-bb_w+1:t])
        bb_upper[t] = mu + T(2) * s
        bb_lower[t] = mu - T(2) * s
        bb_pctb[t] = (prices[t] - bb_lower[t]) / max(bb_upper[t] - bb_lower[t], T(1e-10))
    end
    features["bb_pctb"] = bb_pctb
    # ATR
    atr = zeros(T, n)
    for t in 2:n
        atr[t] = abs(prices[t] - prices[t-1])
    end
    atr_smooth = zeros(T, n)
    for t in 15:n
        atr_smooth[t] = mean(atr[t-13:t])
    end
    features["atr"] = atr_smooth
    # Volatility
    for w in [5, 21, 63]
        vol = zeros(T, n)
        for t in w:n
            vol[t] = std(returns[t-w+1:t]) * sqrt(T(252))
        end
        features["vol_$w"] = vol
    end
    # Volume features
    if volumes !== nothing
        features["volume_sma_ratio"] = volumes ./ max.(begin
            v = zeros(T, n)
            for t in 20:n
                v[t] = mean(volumes[t-19:t])
            end
            v
        end, T(1e-10))
    end
    # Convert to matrix
    keys_sorted = sort(collect(keys(features)))
    feat_matrix = Matrix{T}(undef, n, length(keys_sorted))
    for (j, k) in enumerate(keys_sorted)
        feat_matrix[:, j] = features[k]
    end
    feat_matrix, keys_sorted
end

"""
    cross_sectional_rank(X) -> ranks

Cross-sectional rank normalization at each time step.
"""
function cross_sectional_rank(X::AbstractMatrix{T}) where T<:Real
    n, p = size(X)
    ranks = Matrix{T}(undef, n, p)
    for t in 1:n
        row = X[t, :]
        sorted_idx = sortperm(row)
        for (rank, idx) in enumerate(sorted_idx)
            ranks[t, idx] = T(rank) / T(p)
        end
    end
    ranks
end

"""Interaction features (pairwise products)."""
function interaction_features(X::AbstractMatrix{T}; max_pairs::Int=100) where T<:Real
    n, p = size(X)
    n_pairs = min(div(p * (p - 1), 2), max_pairs)
    interactions = Matrix{T}(undef, n, n_pairs)
    k = 0
    for i in 1:p
        for j in i+1:p
            k += 1
            if k > n_pairs break end
            interactions[:, k] = X[:, i] .* X[:, j]
        end
        if k > n_pairs break end
    end
    interactions[:, 1:k]
end

# ─────────────────────────────────────────────────────────────────────────────
# §13  Walk-Forward Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

"""
    walk_forward_cv(X, y, model_func; kwargs...) -> oos_predictions, scores

Walk-forward cross-validation with purging.
model_func(X_train, y_train) -> model that has predict(model, X_test) method.
"""
function walk_forward_cv(X::AbstractMatrix{T}, y::AbstractVector{T},
                         train_predict_func::Function;
                         train_window::Int=252, test_window::Int=21,
                         gap::Int=5, metric::Symbol=:mse) where T<:Real
    n = size(X, 1)
    oos_preds = T[]
    oos_true = T[]
    t = train_window + gap + 1
    while t + test_window - 1 <= n
        train_X = X[t-train_window-gap:t-gap-1, :]
        train_y = y[t-train_window-gap:t-gap-1]
        test_X = X[t:t+test_window-1, :]
        test_y = y[t:t+test_window-1]
        preds = train_predict_func(train_X, train_y, test_X)
        append!(oos_preds, preds)
        append!(oos_true, test_y)
        t += test_window
    end
    # Compute metric
    score = if metric == :mse
        mean((oos_preds .- oos_true).^2)
    elseif metric == :mae
        mean(abs.(oos_preds .- oos_true))
    elseif metric == :correlation
        if length(oos_preds) > 1
            cor(oos_preds, oos_true)
        else
            zero(T)
        end
    elseif metric == :accuracy
        mean(sign.(oos_preds) .== sign.(oos_true))
    else
        zero(T)
    end
    return oos_preds, oos_true, score
end

"""Purged walk-forward with embargo."""
function purged_walk_forward(X::AbstractMatrix{T}, y::AbstractVector{T},
                             train_predict_func::Function;
                             n_folds::Int=5, purge::Int=5,
                             embargo::Int=5) where T<:Real
    n = size(X, 1)
    fold_size = div(n, n_folds)
    all_preds = T[]
    all_true = T[]
    for fold in 1:n_folds
        test_start = (fold - 1) * fold_size + 1
        test_end = min(fold * fold_size, n)
        # Training: all data except test + purge + embargo
        train_mask = trues(n)
        for t in max(1, test_start - purge):min(n, test_end + embargo)
            train_mask[t] = false
        end
        train_idx = findall(train_mask)
        if length(train_idx) < 10 continue end
        preds = train_predict_func(X[train_idx, :], y[train_idx],
                                   X[test_start:test_end, :])
        append!(all_preds, preds)
        append!(all_true, y[test_start:test_end])
    end
    return all_preds, all_true
end

# ─────────────────────────────────────────────────────────────────────────────
# §14  Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

"""
    permutation_importance(model_predict, X, y; n_repeats=10) -> importances

Permutation feature importance.
"""
function permutation_importance(predict_func::Function,
                                X::AbstractMatrix{T}, y::AbstractVector{T};
                                n_repeats::Int=10, metric::Symbol=:mse,
                                rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    n, p = size(X)
    base_preds = predict_func(X)
    base_score = if metric == :mse
        mean((base_preds .- y).^2)
    elseif metric == :accuracy
        mean(round.(base_preds) .== y)
    else
        mean((base_preds .- y).^2)
    end
    importances = zeros(T, p)
    for j in 1:p
        perm_scores = Vector{T}(undef, n_repeats)
        for r in 1:n_repeats
            X_perm = copy(X)
            X_perm[:, j] = X_perm[randperm(rng, n), j]
            perm_preds = predict_func(X_perm)
            perm_scores[r] = if metric == :mse
                mean((perm_preds .- y).^2)
            elseif metric == :accuracy
                mean(round.(perm_preds) .== y)
            else
                mean((perm_preds .- y).^2)
            end
        end
        if metric == :accuracy
            importances[j] = base_score - mean(perm_scores)
        else
            importances[j] = mean(perm_scores) - base_score
        end
    end
    importances
end

"""
    tree_shap(tree, x) -> shap_values

SHAP-like values via tree path decomposition (simplified TreeSHAP).
"""
function tree_shap(tree::DecisionTree{T}, x::AbstractVector{T}) where T<:Real
    p = length(x)
    shap_values = zeros(T, p)
    base_value = tree.root.value
    _tree_shap_recurse!(shap_values, tree.root, x, one(T), one(T), base_value)
    return shap_values
end

function _tree_shap_recurse!(shap_values::Vector{T}, node::TreeNode{T},
                              x::AbstractVector{T},
                              weight_left::T, weight_right::T,
                              expected_value::T) where T<:Real
    if node.is_leaf
        return
    end
    feat = node.feature
    if x[feat] <= node.threshold
        # Goes left
        if node.left !== nothing && node.right !== nothing
            left_val = node.left.value
            right_val = node.right.value
            n_left = max(node.left.n_samples, 1)
            n_right = max(node.right.n_samples, 1)
            total = n_left + n_right
            contribution = left_val - (n_left * left_val + n_right * right_val) / total
            shap_values[feat] += contribution * weight_left
            _tree_shap_recurse!(shap_values, node.left, x, weight_left, weight_right, expected_value)
        end
    else
        if node.left !== nothing && node.right !== nothing
            left_val = node.left.value
            right_val = node.right.value
            n_left = max(node.left.n_samples, 1)
            n_right = max(node.right.n_samples, 1)
            total = n_left + n_right
            contribution = right_val - (n_left * left_val + n_right * right_val) / total
            shap_values[feat] += contribution * weight_right
            _tree_shap_recurse!(shap_values, node.right, x, weight_left, weight_right, expected_value)
        end
    end
end

"""Aggregate SHAP values across forest."""
function forest_shap(forest::RandomForest{T}, x::AbstractVector{T}) where T<:Real
    p = length(x)
    total_shap = zeros(T, p)
    for tree in forest.trees
        total_shap .+= tree_shap(tree, x)
    end
    total_shap ./= forest.n_trees
    total_shap
end

# ─────────────────────────────────────────────────────────────────────────────
# §15  Model Evaluation Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""R-squared score."""
function r2_score(y_true::AbstractVector{T}, y_pred::AbstractVector{T}) where T<:Real
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    one(T) - ss_res / max(ss_tot, T(1e-16))
end

"""Mean absolute error."""
mae(y_true::AbstractVector{T}, y_pred::AbstractVector{T}) where T<:Real = mean(abs.(y_true .- y_pred))

"""Root mean squared error."""
rmse(y_true::AbstractVector{T}, y_pred::AbstractVector{T}) where T<:Real = sqrt(mean((y_true .- y_pred).^2))

"""Information coefficient (rank correlation)."""
function information_coefficient(y_true::AbstractVector{T},
                                  y_pred::AbstractVector{T}) where T<:Real
    n = length(y_true)
    rank_true = sortperm(sortperm(y_true))
    rank_pred = sortperm(sortperm(y_pred))
    d = rank_true .- rank_pred
    one(T) - T(6) * sum(d.^2) / (T(n) * (T(n)^2 - one(T)))
end

"""Confusion matrix for binary classification."""
function confusion_matrix(y_true::AbstractVector{T},
                           y_pred::AbstractVector{T};
                           threshold::T=T(0.5)) where T<:Real
    pred_class = y_pred .>= threshold
    true_class = y_true .>= threshold
    tp = count(pred_class .& true_class)
    fp = count(pred_class .& .!true_class)
    fn = count(.!pred_class .& true_class)
    tn = count(.!pred_class .& .!true_class)
    return (tp=tp, fp=fp, fn=fn, tn=tn,
            precision=tp / max(tp + fp, 1),
            recall=tp / max(tp + fn, 1),
            f1=2 * tp / max(2 * tp + fp + fn, 1),
            accuracy=(tp + tn) / max(tp + fp + fn + tn, 1))
end

"""AUC-ROC approximation."""
function auc_roc(y_true::AbstractVector{T}, y_scores::AbstractVector{T}) where T<:Real
    n = length(y_true)
    sorted_idx = sortperm(y_scores; rev=true)
    n_pos = count(y_true .> T(0.5))
    n_neg = n - n_pos
    if n_pos == 0 || n_neg == 0
        return T(0.5)
    end
    auc = zero(T)
    tp = 0
    fp = 0
    prev_fp = 0
    prev_tp = 0
    for i in sorted_idx
        if y_true[i] > T(0.5)
            tp += 1
        else
            fp += 1
            auc += T(tp)  # add rectangle area
        end
    end
    auc / (T(n_pos) * T(n_neg))
end

end # module MLFinance
