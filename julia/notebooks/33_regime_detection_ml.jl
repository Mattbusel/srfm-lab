# ============================================================
# Notebook 33: ML-Based Market Regime Detection
# ============================================================
# Topics:
#   1. Regime feature engineering
#   2. Gaussian Mixture Model (GMM) regime detection
#   3. Hidden Markov Model (HMM) for regime switching
#   4. K-means++ regime clustering
#   5. Volatility regime detection
#   6. Regime-conditional trading strategies
#   7. Regime transition matrices and duration analysis
#   8. Out-of-sample regime classification
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 33: ML-Based Market Regime Detection")
println("="^60)

# ── RNG ───────────────────────────────────────────────────
rng_s = UInt64(77777)
function rnd()
    global rng_s
    rng_s = rng_s * 6364136223846793005 + 1442695040888963407
    (rng_s >> 11) / Float64(2^53)
end
function rndn()
    u1 = max(rnd(), 1e-15); u2 = rnd()
    sqrt(-2.0*log(u1)) * cos(2π*u2)
end

# ── Section 1: Generate Regime Data ──────────────────────

println("\n--- Section 1: Multi-Regime Market Simulation ---")

n_obs = 1000

# Define 3 regimes
# Regime 1: Bull (trending up, low vol)
# Regime 2: Bear (trending down, high vol)
# Regime 3: Sideways (flat, medium vol)
regime_params = [
    (0.0008,  0.008, "Bull"),     # mean return/day, vol/day
    (-0.0015, 0.020, "Bear"),
    (0.0001,  0.012, "Sideways"),
]

# Transition matrix
P_trans = [0.98 0.01 0.01;
            0.02 0.95 0.03;
            0.03 0.02 0.95]

# Generate regime sequence via Markov chain
true_regimes = zeros(Int, n_obs)
true_regimes[1] = 1
for t in 2:n_obs
    r = rnd()
    prev = true_regimes[t-1]
    cumprob = cumsum(P_trans[prev, :])
    true_regimes[t] = findfirst(cumprob .>= r)
end

# Generate returns
market_returns = zeros(n_obs)
for t in 1:n_obs
    r = true_regimes[t]
    mu, sig, _ = regime_params[r]
    market_returns[t] = mu + sig * rndn()
end

println("Regime distribution:")
for (r, (mu, sig, name)) in enumerate(regime_params)
    count = sum(true_regimes .== r)
    pct = count / n_obs * 100
    println("  Regime $r ($name): $count days ($(round(pct,digits=1))%), " *
            "E[r]=$(round(mu*252*100,digits=1))%/yr, σ=$(round(sig*sqrt(252)*100,digits=1))%/yr")
end

# Realized statistics per regime
println("\nRealized statistics:")
for r in 1:3
    _, _, name = regime_params[r]
    mask = true_regimes .== r
    rets = market_returns[mask]
    println("  $(lpad(name,8)): E[r]=$(round(mean(rets)*252*100,digits=1))%/yr, " *
            "σ=$(round(std(rets)*sqrt(252)*100,digits=1))%/yr, Sharpe=$(round(mean(rets)/std(rets)*sqrt(252),digits=2))")
end

# ── Section 2: Feature Engineering ──────────────────────

println("\n--- Section 2: Regime Feature Engineering ---")

function build_regime_features(rets, windows=[5,10,21,63])
    n = length(rets)
    features = Dict{String, Vector{Float64}}()

    # Returns at multiple horizons
    for w in windows
        mom = zeros(n)
        for t in w+1:n
            mom[t] = sum(rets[t-w+1:t])
        end
        features["mom_$w"] = mom
    end

    # Realized volatility
    for w in windows
        vol = zeros(n)
        for t in w+1:n
            vol[t] = std(rets[t-w+1:t])
        end
        features["vol_$w"] = vol
    end

    # Volatility-of-volatility
    vol21 = features["vol_21"]
    volvol = zeros(n)
    for t in 22:n
        window = vol21[t-20:t]
        volvol[t] = std(window[window .> 0])
    end
    features["volvol"] = volvol

    # Drawdown
    dd = zeros(n)
    peak = rets[1]
    cumrets = cumsum(rets)
    peak_cum = cumrets[1]
    for t in 1:n
        peak_cum = max(peak_cum, cumrets[t])
        dd[t] = cumrets[t] - peak_cum
    end
    features["drawdown"] = dd

    # Skewness (21-day rolling)
    skew21 = zeros(n)
    for t in 22:n
        r = rets[t-20:t]
        mu = mean(r); s = std(r)
        skew21[t] = s > 1e-12 ? mean((r.-mu).^3)/s^3 : 0.0
    end
    features["skew21"] = skew21

    # Autocorrelation (5-day)
    ac5 = zeros(n)
    for t in 7:n
        r = rets[t-5:t]
        ac5[t] = length(r) > 2 ? cor(r[1:end-1], r[2:end]) : 0.0
    end
    features["ac5"] = ac5

    names_sorted = sort(collect(keys(features)))
    F = hcat([features[k] for k in names_sorted]...)
    return F, names_sorted
end

features, feat_names = build_regime_features(market_returns)
println("Feature set: $(length(feat_names)) features")
println("  $(join(feat_names, ", "))")

# Trim to valid observations (after warmup)
warmup = 65
F_valid = features[warmup+1:end, :]
labels_valid = true_regimes[warmup+1:end]
n_valid = size(F_valid, 1)

# Standardize features
feat_mu = mean(F_valid, dims=1)
feat_std = std(F_valid, dims=1) .+ 1e-12
F_std = (F_valid .- feat_mu) ./ feat_std

println("Valid observations: $n_valid (after $(warmup)-period warmup)")

# ── Section 3: K-Means Regime Clustering ─────────────────

println("\n--- Section 3: K-Means++ Regime Clustering ---")

function kmeans_pp(X, k, max_iter=100)
    n, d = size(X)
    # K-means++ initialization
    centers = zeros(k, d)
    # Choose first center randomly
    idx = round(Int, rnd() * (n-1)) + 1
    centers[1, :] = X[idx, :]

    for c in 2:k
        # Compute distances to nearest center
        dists = [minimum(sum((X[i,:] .- centers[j,:]).^2) for j in 1:c-1) for i in 1:n]
        probs = dists ./ sum(dists)
        cumprobs = cumsum(probs)
        r = rnd()
        new_idx = findfirst(cumprobs .>= r)
        centers[c, :] = X[isnothing(new_idx) ? n : new_idx, :]
    end

    assignments = zeros(Int, n)
    for _ in 1:max_iter
        # Assign
        for i in 1:n
            dists = [sum((X[i,:] .- centers[c,:]).^2) for c in 1:k]
            assignments[i] = argmin(dists)
        end
        # Update centers
        new_centers = zeros(k, d)
        counts = zeros(Int, k)
        for i in 1:n
            new_centers[assignments[i], :] .+= X[i, :]
            counts[assignments[i]] += 1
        end
        for c in 1:k
            if counts[c] > 0
                new_centers[c, :] ./= counts[c]
            else
                new_centers[c, :] = centers[c, :]
            end
        end
        if norm(new_centers .- centers) < 1e-8; break; end
        centers = new_centers
    end
    return assignments, centers
end

k_regimes = 3
# Use only key features for clustering
key_feat_idx = findall(in.(feat_names, [["mom_21", "vol_21", "drawdown", "skew21"]]))
F_key = F_std[:, key_feat_idx]
km_labels, km_centers = kmeans_pp(F_key, k_regimes)

println("K-means regime statistics:")
println("  Cluster | Size | Mean Ret/day | Vol/day | Drawdown | Interpretation")
println("  " * "-"^68)
for c in 1:k_regimes
    mask = km_labels .== c
    rets_c = market_returns[warmup+1:end][mask]
    feat_c = F_valid[mask, :]
    mu_r = mean(rets_c) * 100
    vol_r = std(rets_c) * 100
    dd_c = mean(feat_c[:, findfirst(feat_names .== "drawdown")])
    interp = mu_r > 0.05 ? "Bull" : mu_r < -0.05 ? "Bear" : "Sideways"
    println("  $(lpad(c,7)) | $(lpad(sum(mask),4)) | $(lpad(round(mu_r,digits=3),12))% | $(lpad(round(vol_r,digits=3),7))% | $(lpad(round(dd_c,digits=4),8)) | $interp")
end

# Confusion matrix vs true regimes
confusion = zeros(Int, k_regimes, 3)
for i in 1:n_valid
    confusion[km_labels[i], labels_valid[i]] += 1
end
println("\nConfusion matrix (K-means vs True):")
println("  KM\\True |  Bull  |  Bear  | Sideways")
println("  " * "-"^38)
for c in 1:k_regimes
    println("  Cluster $c | $(lpad(confusion[c,1],6)) | $(lpad(confusion[c,2],6)) | $(lpad(confusion[c,3],8))")
end

# ── Section 4: Gaussian Mixture Model ────────────────────

println("\n--- Section 4: Gaussian Mixture Model (EM) ---")

# GMM with k=3 components using EM algorithm
# Features: return and volatility only (2D for clarity)
vol5 = F_valid[:, findfirst(feat_names .== "vol_5")]
mom5 = F_valid[:, findfirst(feat_names .== "mom_5")]
X_gmm = hcat(mom5 ./ (std(mom5)+1e-12), vol5 ./ (std(vol5)+1e-12))

function gmm_em(X, k, max_iter=100)
    n, d = size(X)
    # Initialize with K-means labels
    km_init, _ = kmeans_pp(X, k, 50)
    pi_k = zeros(k)
    mu_k = zeros(k, d)
    sigma_k = [Matrix{Float64}(I, d, d) for _ in 1:k]

    for c in 1:k
        mask = km_init .== c
        pi_k[c] = max(sum(mask) / n, 1e-3)
        if sum(mask) > 1
            mu_k[c, :] = mean(X[mask, :], dims=1)[:]
            sigma_k[c] = cov(X[mask, :]) .+ 1e-6*I
        end
    end
    pi_k ./= sum(pi_k)

    function mvnorm_logpdf(x, mu, Sigma)
        d = length(x)
        diff = x .- mu
        L = chol_decomp_simple(Sigma)
        # solve L*y = diff
        y = zeros(d)
        for i in 1:d
            y[i] = (diff[i] - dot(L[i,1:i-1], y[1:i-1])) / max(L[i,i], 1e-15)
        end
        log_det = 2*sum(log(max(L[i,i], 1e-30)) for i in 1:d)
        return -0.5*(d*log(2π) + log_det + dot(y,y))
    end

    function chol_decomp_simple(A)
        d = size(A,1)
        L = zeros(d, d)
        for i in 1:d
            for j in 1:i
                s = A[i,j] - sum(L[i,l]*L[j,l] for l in 1:j-1; init=0.0)
                L[i,j] = i==j ? sqrt(max(s,1e-15)) : s/max(L[j,j],1e-15)
            end
        end
        return L
    end

    responsibilities = zeros(n, k)
    for iter in 1:max_iter
        # E-step
        log_r = zeros(n, k)
        for c in 1:k
            for i in 1:n
                log_r[i, c] = log(pi_k[c]) + mvnorm_logpdf(X[i,:], mu_k[c,:], sigma_k[c])
            end
        end
        # Normalize
        for i in 1:n
            max_lr = maximum(log_r[i,:])
            log_r[i,:] .-= max_lr
            r = exp.(log_r[i,:])
            responsibilities[i, :] = r ./ max(sum(r), 1e-15)
        end
        # M-step
        N_k = vec(sum(responsibilities, dims=1))
        for c in 1:k
            pi_k[c] = N_k[c] / n
            if N_k[c] > 1e-6
                mu_k[c, :] = (responsibilities[:, c]' * X)[:] ./ N_k[c]
                diff = X .- mu_k[c, :]'
                sigma_k[c] = (diff' * (responsibilities[:, c] .* diff)) ./ N_k[c] .+ 1e-6*I
            end
        end
    end
    gmm_labels = [argmax(responsibilities[i, :]) for i in 1:n]
    return gmm_labels, mu_k, sigma_k, pi_k
end

gmm_labels, gmm_mu, gmm_sig, gmm_pi = gmm_em(X_gmm, k_regimes)

println("GMM regime summary:")
for c in 1:k_regimes
    mask = gmm_labels .== c
    rets_c = market_returns[warmup+1:end][mask]
    println("  Component $c (π=$(round(gmm_pi[c],digits=3))): n=$(sum(mask)), " *
            "E[r]=$(round(mean(rets_c)*100*252,digits=1))%/yr, vol=$(round(std(rets_c)*sqrt(252)*100,digits=1))%/yr")
end

# ── Section 5: Hidden Markov Model ───────────────────────

println("\n--- Section 5: Hidden Markov Model ---")

# Simplified HMM with Gaussian emissions, k=3 regimes
# Use only the daily return as observation

function hmm_em(obs, k, max_iter=50)
    n = length(obs)
    # Initialize
    mu_hmm = [quantile(sort(obs), q) for q in range(0.2, 0.8, length=k)]
    sig_hmm = fill(std(obs), k)
    A_trans = Matrix{Float64}(I, k, k) .* 0.9 .+ 0.1/k
    for r in 1:k; A_trans[r,:] ./= sum(A_trans[r,:]); end
    pi_init = fill(1.0/k, k)

    function gauss_pdf(x, mu, sig)
        return exp(-0.5*((x-mu)/max(sig,1e-10))^2) / (max(sig,1e-10)*sqrt(2π))
    end

    alpha = zeros(n, k)
    beta_hmm = zeros(n, k)
    gamma_hmm = zeros(n, k)
    xi = zeros(n-1, k, k)

    for iter in 1:max_iter
        # E-step: Forward-Backward
        B = zeros(n, k)
        for t in 1:n
            for s in 1:k
                B[t, s] = gauss_pdf(obs[t], mu_hmm[s], sig_hmm[s])
            end
        end
        # Forward
        alpha[1, :] = pi_init .* B[1, :]
        c = sum(alpha[1,:]); alpha[1,:] ./= max(c, 1e-30)
        for t in 2:n
            for j in 1:k
                alpha[t,j] = sum(alpha[t-1,l]*A_trans[l,j] for l in 1:k) * B[t,j]
            end
            c = sum(alpha[t,:]); alpha[t,:] ./= max(c, 1e-30)
        end
        # Backward
        beta_hmm[n,:] .= 1.0
        for t in n-1:-1:1
            for i in 1:k
                beta_hmm[t,i] = sum(A_trans[i,j]*B[t+1,j]*beta_hmm[t+1,j] for j in 1:k)
            end
            c = sum(beta_hmm[t,:]); beta_hmm[t,:] ./= max(c, 1e-30)
        end
        # Gamma
        for t in 1:n
            ab = alpha[t,:] .* beta_hmm[t,:]
            gamma_hmm[t,:] = ab ./ max(sum(ab), 1e-30)
        end
        # Xi
        for t in 1:n-1
            for i in 1:k, j in 1:k
                xi[t,i,j] = alpha[t,i] * A_trans[i,j] * B[t+1,j] * beta_hmm[t+1,j]
            end
            xi[t,:,:] ./= max(sum(xi[t,:,:]), 1e-30)
        end
        # M-step
        pi_init = gamma_hmm[1, :]
        for i in 1:k, j in 1:k
            A_trans[i,j] = sum(xi[:,i,j]) / max(sum(gamma_hmm[1:end-1,i]), 1e-30)
        end
        for j in 1:k
            wts = gamma_hmm[:,j]
            mu_hmm[j] = dot(wts, obs) / max(sum(wts), 1e-30)
            sig_hmm[j] = sqrt(dot(wts, (obs .- mu_hmm[j]).^2) / max(sum(wts), 1e-30))
            sig_hmm[j] = max(sig_hmm[j], 1e-4)
        end
    end

    viterbi = [argmax(gamma_hmm[t,:]) for t in 1:n]
    return viterbi, mu_hmm, sig_hmm, A_trans
end

hmm_labels, hmm_mu, hmm_sig, hmm_A = hmm_em(market_returns, k_regimes)

println("HMM regime parameters:")
println("  State | Mean (daily%) | Vol (daily%) | Stationary prob")
println("  " * "-"^52)
# Compute stationary distribution via power iteration
pi_stat = ones(k_regimes) ./ k_regimes
for _ in 1:200
    pi_stat = hmm_A' * pi_stat
    pi_stat ./= sum(pi_stat)
end
for s in 1:k_regimes
    ann_mu = hmm_mu[s] * 252 * 100
    ann_sig = hmm_sig[s] * sqrt(252) * 100
    println("  $(lpad(s,5)) | $(lpad(round(hmm_mu[s]*100,digits=4),13)) | $(lpad(round(hmm_sig[s]*100,digits=4),12)) | $(round(pi_stat[s]*100,digits=1))%")
end

println("\nHMM transition matrix:")
for i in 1:k_regimes
    row = join([lpad(round(hmm_A[i,j]*100,digits=1)*"%", 7) for j in 1:k_regimes], "  ")
    println("  State $i → [$row]")
end

# ── Section 6: Regime-Conditional Strategies ─────────────

println("\n--- Section 6: Regime-Conditional Trading ---")

# Use HMM detected regimes to condition portfolio allocation
# Regime 1 (bull): 100% equity, Regime 2 (bear): 0% equity, Regime 3 (sideways): 50%

println("Regime-conditional strategy backtest:")
# Map HMM regimes to bull/bear/sideways by mean return
hmm_order = sortperm(hmm_mu)  # sort by mean return: lowest to highest
regime_map = Dict(hmm_order[1] => :bear, hmm_order[2] => :sideways, hmm_order[3] => :bull)

allocations = Dict(:bull=>1.0, :bear=>0.0, :sideways=>0.5)

strategy_returns = zeros(n_obs)
for t in 1:n_obs
    t_valid = t - warmup
    if t_valid >= 1 && t_valid <= length(hmm_labels)
        regime = regime_map[hmm_labels[t_valid]]
        alloc = allocations[regime]
        strategy_returns[t] = alloc * market_returns[t]
    else
        strategy_returns[t] = 0.5 * market_returns[t]  # 50% default
    end
end

for (label, rets) in [("Buy & Hold", market_returns), ("Regime Strategy", strategy_returns)]
    ann_r = mean(rets) * 252 * 100
    ann_v = std(rets) * sqrt(252) * 100
    sharpe = ann_v > 0 ? ann_r / ann_v : 0.0
    cum = (prod(1.0 .+ rets) - 1.0) * 100
    cum_rets = cumsum(rets)
    peak = -Inf; max_dd = 0.0
    for r in cum_rets; peak = max(peak, r); max_dd = max(max_dd, peak - r); end
    println("  $(lpad(label, 16)): Ret=$(round(ann_r,digits=1))%, Sharpe=$(round(sharpe,digits=2)), " *
            "Cum=$(round(cum,digits=1))%, MaxDD=$(round(max_dd*100,digits=1))%")
end

# ── Section 7: Regime Duration Analysis ──────────────────

println("\n--- Section 7: Regime Duration Analysis ---")

# Compute regime durations from true regimes
function compute_durations(regime_seq, n_regimes)
    durations = [Float64[] for _ in 1:n_regimes]
    current = regime_seq[1]
    current_dur = 1
    for t in 2:length(regime_seq)
        if regime_seq[t] == current
            current_dur += 1
        else
            push!(durations[current], current_dur)
            current = regime_seq[t]
            current_dur = 1
        end
    end
    push!(durations[current], current_dur)
    return durations
end

durations = compute_durations(true_regimes, 3)
println("Regime duration analysis (true regimes):")
println("  Regime    | # Episodes | Avg Days | Median | Max Days | Expected (from Markov)")
println("  " * "-"^68)
for (r, (mu, sig, name)) in enumerate(regime_params)
    durs = durations[r]
    expected_dur = !isempty(durs) ? 1.0 / (1.0 - P_trans[r, r]) : 0.0
    println("  $(lpad(name,9)) | $(lpad(length(durs),10)) | $(lpad(round(mean(durs),digits=1),8)) | " *
            "$(lpad(round(median(durs),digits=1),6)) | $(lpad(round(maximum(durs),digits=1),8)) | $(round(expected_dur,digits=1))")
end

# ── Section 8: Out-of-Sample Classification ──────────────

println("\n--- Section 8: Out-of-Sample Regime Classification ---")

# Train on first 700 obs, test on last 300
train_end = 700
test_start = train_end + warmup + 1
n_train = train_end - warmup
n_test = n_valid - n_train

println("Train: first $(n_train) valid obs, Test: last $(n_test) valid obs")

# Re-train K-means on training set
F_train = F_key[1:n_train, :]
km_labels_train, km_centers_train = kmeans_pp(F_train, k_regimes)

# Classify test set by nearest centroid
F_test = F_key[n_train+1:end, :]
km_labels_test = zeros(Int, n_test)
for i in 1:n_test
    dists = [sum((F_test[i,:] .- km_centers_train[c,:]).^2) for c in 1:k_regimes]
    km_labels_test[i] = argmin(dists)
end

# Compare test regime classification to true regimes
true_test = labels_valid[n_train+1:end]
# Compute accuracy (need to find best regime mapping)
best_acc = 0.0
best_perm = [1,2,3]
for perm in [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    mapped = [perm[l] for l in km_labels_test]
    acc = mean(mapped .== true_test)
    if acc > best_acc
        best_acc = acc
        best_perm = perm
    end
end

println("Out-of-sample regime classification accuracy: $(round(best_acc*100, digits=1))%")

# Regime-conditional returns (test period)
println("\nTest period regime-conditional returns:")
for c in 1:k_regimes
    mask = km_labels_test .== c
    if sum(mask) < 5; continue; end
    rets_c = market_returns[test_start:end][mask[1:min(end,n_test)]]
    if isempty(rets_c); continue; end
    ann_r = mean(rets_c) * 252 * 100
    println("  K-means Regime $c: n=$(sum(mask)), E[r]=$(round(ann_r,digits=1))%/yr")
end

println("\n✓ Notebook 33 complete")
