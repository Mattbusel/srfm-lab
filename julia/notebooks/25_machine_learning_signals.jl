# ============================================================
# Notebook 25: Machine Learning-Based Alpha Signals
# ============================================================
# Topics:
#   1. Feature engineering from price/volume data
#   2. Decision tree and random forest logic (from scratch)
#   3. Gradient boosting for return prediction
#   4. Cross-validation framework for financial ML
#   5. Feature importance and signal analysis
#   6. Ensemble combination of ML signals
#   7. Walk-forward model evaluation
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 25: Machine Learning Alpha Signals")
println("="^60)

# ── Section 1: Simulate Price/Volume Data ──────────────────

println("\n--- Section 1: Data Simulation ---")

n_obs = 2000
n_stocks = 10
rng_state = UInt64(42)
function lcg_randn()
    global rng_state
    rng_state = rng_state * 6364136223846793005 + 1442695040888963407
    u1 = max((rng_state >> 11) / Float64(2^53), 1e-15)
    rng_state = rng_state * 6364136223846793005 + 1442695040888963407
    u2 = (rng_state >> 11) / Float64(2^53)
    return sqrt(-2.0 * log(u1)) * cos(2π * u2)
end

# Simulate returns with some predictability
true_factor = [lcg_randn() for _ in 1:n_obs]
returns = zeros(n_obs, n_stocks)
for j in 1:n_stocks
    beta = 0.3 + 0.2 * lcg_randn()
    for t in 1:n_obs
        noise = 0.02 * lcg_randn()
        returns[t, j] = beta * true_factor[t] * 0.01 + noise
    end
end

println("Simulated $n_obs obs × $n_stocks stocks")
println("Return stats: mean=$(round(mean(returns), digits=5)), std=$(round(std(returns), digits=4))")

# ── Section 2: Feature Engineering ────────────────────────

println("\n--- Section 2: Feature Engineering ---")

function build_features(rets::Matrix{Float64}, volumes::Matrix{Float64}=Matrix{Float64}(undef,0,0))
    T, N = size(rets)
    windows = [5, 10, 21, 63]
    features_list = Dict{String, Matrix{Float64}}()

    # Momentum at multiple horizons
    for w in windows
        mom = zeros(T, N)
        for t in w+1:T
            mom[t, :] = sum(rets[t-w+1:t, :], dims=1)[:]
        end
        features_list["mom_$w"] = mom
    end

    # Reversal (short-term mean reversion)
    rev = zeros(T, N)
    for t in 2:T
        rev[t, :] = -rets[t-1, :]  # previous return reversal
    end
    features_list["reversal_1"] = rev

    # Volatility (rolling std)
    for w in [10, 21]
        vol = zeros(T, N)
        for t in w+1:T
            vol[t, :] = std(rets[t-w+1:t, :], dims=1)[:]
        end
        features_list["vol_$w"] = vol
    end

    # Skewness
    skew = zeros(T, N)
    for t in 22:T
        for j in 1:N
            r = rets[t-21:t, j]
            mu = mean(r); s = std(r)
            skew[t, j] = s > 1e-12 ? mean((r .- mu).^3) / s^3 : 0.0
        end
    end
    features_list["skew_21"] = skew

    return features_list
end

features = build_features(returns)
feature_names = sort(collect(keys(features)))
println("Generated $(length(feature_names)) feature types: $(join(feature_names, ", "))")

# Assemble design matrix (use single stock for simplicity)
stock_idx = 1
lookback = 63
n_valid = n_obs - lookback - 21
X_mat = zeros(n_valid, length(feature_names))
y_vec = zeros(n_valid)  # forward 5-day return

for (col, fname) in enumerate(feature_names)
    feat_mat = features[fname]
    for t in 1:n_valid
        X_mat[t, col] = feat_mat[lookback + t, stock_idx]
    end
end
for t in 1:n_valid
    y_vec[t] = sum(returns[lookback+t+1:lookback+t+5, stock_idx])
end

println("Design matrix: $(size(X_mat, 1)) obs × $(size(X_mat, 2)) features")
println("Target: 5-day forward return, mean=$(round(mean(y_vec), digits=5))")

# ── Section 3: Linear Baseline (Ridge Regression) ──────────

println("\n--- Section 3: Linear Baseline ---")

function ridge_regression(X::Matrix{Float64}, y::Vector{Float64}, lambda::Float64=1e-2)
    n, p = size(X)
    # Standardize
    mu_x = mean(X, dims=1)
    sig_x = std(X, dims=1) .+ 1e-12
    X_std = (X .- mu_x) ./ sig_x
    beta = (X_std'X_std + lambda*I) \ (X_std'y)
    intercept = mean(y) - dot(mean(X_std, dims=1)[:], beta)
    return beta, mu_x[:], sig_x[:], intercept
end

function predict_ridge(X::Matrix{Float64}, beta::Vector{Float64},
                        mu_x::Vector{Float64}, sig_x::Vector{Float64},
                        intercept::Float64)
    X_std = (X .- mu_x') ./ sig_x'
    return X_std * beta .+ intercept
end

# Train/test split
n_train = round(Int, n_valid * 0.7)
X_train, X_test = X_mat[1:n_train, :], X_mat[n_train+1:end, :]
y_train, y_test = y_vec[1:n_train], y_vec[n_train+1:end]

beta_ridge, mu_x, sig_x, intercept = ridge_regression(X_train, y_train, 0.01)
y_pred_ridge = predict_ridge(X_test, beta_ridge, mu_x, sig_x, intercept)

ridge_ic = cor(y_pred_ridge, y_test)
println("Ridge regression IC (test): $(round(ridge_ic, digits=4))")

# ── Section 4: Decision Tree (from scratch) ───────────────

println("\n--- Section 4: Decision Tree ---")

mutable struct DecisionNode
    feature_idx::Int
    threshold::Float64
    left::Union{DecisionNode, Nothing}
    right::Union{DecisionNode, Nothing}
    leaf_value::Float64
    is_leaf::Bool
end

function gini_impurity_regression(y::Vector{Float64})
    return var(y) * length(y)  # MSE * n
end

function best_split(X::Matrix{Float64}, y::Vector{Float64}, n_features_sample::Int=5)
    n, p = size(X)
    best_gain = 0.0
    best_feat = 1
    best_thresh = 0.0
    parent_var = var(y) * n

    feat_sample = min(n_features_sample, p)
    # Sample features
    feat_indices = sort([mod(i * 1664525 + 42, p) + 1 for i in 1:feat_sample])
    unique!(feat_indices)

    for f in feat_indices
        vals = sort(unique(X[:, f]))
        thresholds = length(vals) > 10 ?
            [vals[round(Int, i * length(vals) / 10)] for i in 1:9] :
            vals[1:end-1]
        for thresh in thresholds
            left_mask = X[:, f] .<= thresh
            right_mask = .!left_mask
            n_l, n_r = sum(left_mask), sum(right_mask)
            if n_l < 2 || n_r < 2; continue; end
            child_var = var(y[left_mask]) * n_l + var(y[right_mask]) * n_r
            gain = parent_var - child_var
            if gain > best_gain
                best_gain = gain
                best_feat = f
                best_thresh = thresh
            end
        end
    end
    return best_feat, best_thresh, best_gain
end

function build_tree(X::Matrix{Float64}, y::Vector{Float64},
                     depth::Int=0, max_depth::Int=4, min_leaf::Int=20)
    if depth >= max_depth || length(y) <= min_leaf
        return DecisionNode(1, 0.0, nothing, nothing, mean(y), true)
    end
    feat, thresh, gain = best_split(X, y)
    if gain <= 1e-10
        return DecisionNode(1, 0.0, nothing, nothing, mean(y), true)
    end
    left_mask = X[:, feat] .<= thresh
    right_mask = .!left_mask
    left = build_tree(X[left_mask, :], y[left_mask], depth+1, max_depth, min_leaf)
    right = build_tree(X[right_mask, :], y[right_mask], depth+1, max_depth, min_leaf)
    return DecisionNode(feat, thresh, left, right, mean(y), false)
end

function predict_tree(node::DecisionNode, x::Vector{Float64})
    if node.is_leaf; return node.leaf_value; end
    if x[node.feature_idx] <= node.threshold
        return predict_tree(node.left, x)
    else
        return predict_tree(node.right, x)
    end
end

# Train a shallow tree
tree = build_tree(X_train, y_train, 0, 3, 30)
y_pred_tree = [predict_tree(tree, X_test[i, :]) for i in 1:size(X_test, 1)]
tree_ic = cor(y_pred_tree, y_test)
println("Decision tree IC (test, max_depth=3): $(round(tree_ic, digits=4))")

# ── Section 5: Random Forest ──────────────────────────────

println("\n--- Section 5: Random Forest ---")

function bootstrap_sample(X::Matrix{Float64}, y::Vector{Float64}, state::UInt64)
    n = length(y)
    indices = [Int(state * 6364136223846793005 % n) + 1 for _ in 1:n]
    state = state * 6364136223846793005 + 1442695040888963407
    return X[indices, :], y[indices], state
end

function random_forest(X::Matrix{Float64}, y::Vector{Float64},
                        n_trees::Int=20, max_depth::Int=4)
    trees = Vector{DecisionNode}()
    state = UInt64(99)
    for b in 1:n_trees
        # Bootstrap sample
        n = length(y)
        idx = [Int((state * 6364136223846793005 + i * 1013904223) % n) + 1 for i in 1:n]
        state = state * 6364136223846793005 + 1442695040888963407
        X_boot, y_boot = X[idx, :], y[idx]
        tree = build_tree(X_boot, y_boot, 0, max_depth, 25)
        push!(trees, tree)
    end
    return trees
end

function predict_forest(trees::Vector{DecisionNode}, x::Vector{Float64})
    return mean(predict_tree(t, x) for t in trees)
end

rf = random_forest(X_train, y_train, 15, 4)
y_pred_rf = [predict_forest(rf, X_test[i, :]) for i in 1:size(X_test, 1)]
rf_ic = cor(y_pred_rf, y_test)
println("Random Forest IC (test, 15 trees): $(round(rf_ic, digits=4))")

# ── Section 6: Gradient Boosting ─────────────────────────

println("\n--- Section 6: Gradient Boosting ---")

function gradient_boost(X::Matrix{Float64}, y::Vector{Float64},
                          n_rounds::Int=30, lr::Float64=0.1, max_depth::Int=3)
    n = length(y)
    F = fill(mean(y), n)  # initial prediction
    trees = Vector{DecisionNode}()
    push!(trees, DecisionNode(1, 0.0, nothing, nothing, mean(y), true))

    for _ in 1:n_rounds
        residuals = y .- F
        tree = build_tree(X, residuals, 0, max_depth, 20)
        tree_pred = [predict_tree(tree, X[i, :]) for i in 1:n]
        F .+= lr .* tree_pred
        push!(trees, tree)
    end
    return trees, lr
end

function predict_gbm(trees::Vector{DecisionNode}, X::Matrix{Float64}, lr::Float64)
    n = size(X, 1)
    F = fill(trees[1].leaf_value, n)
    for tree in trees[2:end]
        tree_pred = [predict_tree(tree, X[i, :]) for i in 1:n]
        F .+= lr .* tree_pred
    end
    return F
end

gbm_trees, gbm_lr = gradient_boost(X_train, y_train, 25, 0.05, 3)
y_pred_gbm = predict_gbm(gbm_trees, X_test, gbm_lr)
gbm_ic = cor(y_pred_gbm, y_test)
println("Gradient Boosting IC (test, 25 rounds): $(round(gbm_ic, digits=4))")

# ── Section 7: Feature Importance ────────────────────────

println("\n--- Section 7: Feature Importance ---")

# Permutation importance for linear model
function permutation_importance(X::Matrix{Float64}, y::Vector{Float64},
                                  beta::Vector{Float64}, mu_x::Vector{Float64},
                                  sig_x::Vector{Float64}, intercept::Float64,
                                  feature_names::Vector{String})
    base_pred = predict_ridge(X, beta, mu_x, sig_x, intercept)
    base_ic = abs(cor(base_pred, y))
    importances = zeros(length(feature_names))
    for f in 1:length(feature_names)
        X_perm = copy(X)
        # Shuffle feature f
        n = size(X, 1)
        state = UInt64(f * 42)
        for i in n:-1:2
            state = state * 6364136223846793005 + 1442695040888963407
            j = Int(state % i) + 1
            X_perm[i, f], X_perm[j, f] = X_perm[j, f], X_perm[i, f]
        end
        perm_pred = predict_ridge(X_perm, beta, mu_x, sig_x, intercept)
        perm_ic = abs(cor(perm_pred, y))
        importances[f] = base_ic - perm_ic
    end
    return importances
end

importances = permutation_importance(X_test, y_test, beta_ridge, mu_x, sig_x, intercept, feature_names)
sorted_idx = sortperm(importances, rev=true)
println("Top 5 features by permutation importance:")
for k in 1:min(5, length(sorted_idx))
    i = sorted_idx[k]
    println("  $(feature_names[i]): $(round(importances[i], digits=5))")
end

# ── Section 8: Ensemble Combination ──────────────────────

println("\n--- Section 8: Ensemble Combination ---")

# Stack predictions
y_preds = hcat(y_pred_ridge, y_pred_tree, y_pred_rf, y_pred_gbm)
model_names = ["Ridge", "Decision Tree", "Random Forest", "GBM"]
model_ics = [cor(y_preds[:, i], y_test) for i in 1:4]

println("Individual model ICs:")
for (name, ic) in zip(model_names, model_ics)
    println("  $name: $(round(ic, digits=4))")
end

# Equal-weight ensemble
ensemble_pred = mean(y_preds, dims=2)[:]
ensemble_ic = cor(ensemble_pred, y_test)
println("Equal-weight ensemble IC: $(round(ensemble_ic, digits=4))")

# IC-weighted ensemble
ic_weights = max.(model_ics, 0.0)
ic_weights ./= sum(ic_weights) + 1e-12
ensemble_weighted = y_preds * ic_weights
weighted_ic = cor(ensemble_weighted, y_test)
println("IC-weighted ensemble IC: $(round(weighted_ic, digits=4))")

# ── Section 9: Walk-Forward Evaluation ───────────────────

println("\n--- Section 9: Walk-Forward Evaluation ---")

# Rolling walk-forward: train on W periods, test on next H
train_window = 300
test_window = 50
n_rounds = 0
wf_ics = Float64[]
wf_returns = Float64[]

t = train_window
while t + test_window <= n_valid
    Xtr = X_mat[t-train_window+1:t, :]
    ytr = y_vec[t-train_window+1:t]
    Xte = X_mat[t+1:t+test_window, :]
    yte = y_vec[t+1:t+test_window]

    beta, mu, sig, inter = ridge_regression(Xtr, ytr, 0.01)
    pred = predict_ridge(Xte, beta, mu, sig, inter)

    ic = cor(pred, yte)
    # Long-short return: go long top quintile, short bottom
    sorted_idx_wf = sortperm(pred)
    long_idx = sorted_idx_wf[end-9:end]
    short_idx = sorted_idx_wf[1:10]
    ls_ret = mean(yte[long_idx]) - mean(yte[short_idx])

    push!(wf_ics, ic)
    push!(wf_returns, ls_ret)
    t += test_window
    n_rounds += 1
end

println("Walk-forward: $n_rounds rounds, train=$train_window, test=$test_window")
println("  Mean IC: $(round(mean(wf_ics), digits=4))")
println("  IC t-stat: $(round(mean(wf_ics)/std(wf_ics)*sqrt(n_rounds), digits=3))")
println("  Avg LS return per period: $(round(mean(wf_returns)*10_000, digits=2)) bps")
println("  LS Sharpe (annualized): $(round(mean(wf_returns)/std(wf_returns)*sqrt(252.0/test_window), digits=3))")

# ── Section 10: Signal Decay Analysis ────────────────────

println("\n--- Section 10: Signal Decay Analysis ---")

# Compute IC at different forward horizons
horizons = [1, 3, 5, 10, 21]
println("IC decay by forward horizon (last 500 obs of test):")
for h in horizons
    fwd_rets = zeros(n_valid - h)
    for t in 1:n_valid-h
        fwd_rets[t] = sum(returns[lookback+t+1:lookback+t+h, stock_idx])
    end
    sig = X_mat[1:n_valid-h, 1]  # use momentum feature as example signal
    valid_mask = abs.(fwd_rets) .> 1e-12
    if sum(valid_mask) > 20
        ic_h = cor(sig[valid_mask], fwd_rets[valid_mask])
        println("  h=$h: IC=$(round(ic_h, digits=4))")
    end
end

println("\n✓ Notebook 25 complete")
