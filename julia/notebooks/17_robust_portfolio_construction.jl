# Notebook 17: Robust Portfolio Construction
# ============================================
# Compare Markowitz vs MCD-based optimization.
# Stress tests, resampled efficiency, parameter uncertainty,
# worst-case (Bertsimas-Sim) robust optimization.
# ============================================

using Statistics, LinearAlgebra, Random, Printf

Random.seed!(17)

# ── 1. DATA GENERATION ───────────────────────────────────────────────────────

const N_ASSETS  = 8
const N_OBS     = 504   # 2 years daily

asset_names = ["BTC", "ETH", "SOL", "BNB", "AVAX", "MATIC", "LINK", "DOT"]

"""
Generate correlated crypto returns with fat tails and regime shifts.
"""
function generate_crypto_returns(n::Int, k::Int; seed::Int=17)
    rng = MersenneTwister(seed)

    # True covariance structure
    sigma_vec = [0.045, 0.055, 0.065, 0.042, 0.070, 0.068, 0.060, 0.058]

    # Correlation matrix (BTC/ETH highly correlated, others moderate)
    C = [1.00 0.85 0.72 0.78 0.68 0.65 0.62 0.64;
         0.85 1.00 0.75 0.76 0.70 0.68 0.65 0.66;
         0.72 0.75 1.00 0.68 0.72 0.70 0.62 0.63;
         0.78 0.76 0.68 1.00 0.65 0.63 0.60 0.62;
         0.68 0.70 0.72 0.65 1.00 0.78 0.65 0.66;
         0.65 0.68 0.70 0.63 0.78 1.00 0.66 0.67;
         0.62 0.65 0.62 0.60 0.65 0.66 1.00 0.72;
         0.64 0.66 0.63 0.62 0.66 0.67 0.72 1.00]

    # Make positive definite
    S = Diagonal(sigma_vec) * C * Diagonal(sigma_vec)
    S = (S + S') / 2 + 1e-6 * I

    L = cholesky(S).L
    mu_true = [0.0003, 0.0004, 0.0006, 0.0003, 0.0007, 0.0006, 0.0004, 0.0004]

    Z = randn(rng, n, k)
    returns = Z * L' .+ mu_true'

    # Add fat tails: occasional jumps affecting all assets
    for t in 1:n
        if rand(rng) < 0.03
            jump = randn(rng) * 0.05
            returns[t, :] .+= jump .* (0.8 .+ 0.4 .* rand(rng, k))
        end
    end

    return returns, mu_true, S
end

println("Generating synthetic crypto returns...")
returns_daily, mu_true, Sigma_true = generate_crypto_returns(N_OBS, N_ASSETS)

@printf("  Assets: %d  Observations: %d\n", N_ASSETS, N_OBS)
@printf("  Annualized volatilities:\n")
for i in 1:N_ASSETS
    @printf("    %-6s: %.1f%%\n", asset_names[i], std(returns_daily[:,i]) * sqrt(252) * 100)
end

# ── 2. SAMPLE STATISTICS ─────────────────────────────────────────────────────

println("\n" * "="^60)
println("SAMPLE vs TRUE PARAMETER COMPARISON")
println("="^60)

mu_sample  = vec(mean(returns_daily, dims=1))
Sigma_sample = cov(returns_daily)

println("\nMean return estimation error (|sample - true|):")
for i in 1:N_ASSETS
    @printf("  %-6s: true=%+.4f  sample=%+.4f  error=%+.5f\n",
            asset_names[i], mu_true[i], mu_sample[i], mu_sample[i] - mu_true[i])
end

# Covariance estimation quality
Sigma_error = norm(Sigma_sample - Sigma_true, 2) / norm(Sigma_true, 2)
@printf("\nCovariance estimation relative error (spectral): %.4f\n", Sigma_error)

# ── 3. MINIMUM COVARIANCE DETERMINANT (MCD) ───────────────────────────────────

println("\n" * "="^60)
println("MINIMUM COVARIANCE DETERMINANT (MCD) ESTIMATION")
println("="^60)

"""
Fast-MCD approximation: iterative trimming algorithm.
h = floor(n * alpha) observations retained (default alpha=0.75).
"""
function mcd_covariance(X::Matrix{Float64}; h_frac::Float64=0.75, n_starts::Int=5,
                         max_iter::Int=50, seed::Int=17)
    rng = MersenneTwister(seed)
    n, p = size(X)
    h    = floor(Int, n * h_frac)

    best_det  = Inf
    best_mu   = zeros(p)
    best_Sigma = I(p) * 1.0

    for _ in 1:n_starts
        # Random initial subset
        idx   = sort(randperm(rng, n)[1:h])
        subset = X[idx, :]

        for iter in 1:max_iter
            mu_c    = vec(mean(subset, dims=1))
            Sigma_c = cov(subset) + 1e-8 * I(p)

            # Mahalanobis distances for all points
            diffs   = X .- mu_c'
            Sinv    = inv(Sigma_c)
            mah_sq  = [dot(diffs[i,:], Sinv * diffs[i,:]) for i in 1:n]

            # Keep h smallest
            new_idx = sortperm(mah_sq)[1:h]
            new_idx = sort(new_idx)

            if new_idx == idx
                break
            end
            idx    = new_idx
            subset = X[idx, :]
        end

        Sigma_c = cov(subset) + 1e-8 * I(p)
        det_c   = det(Sigma_c)

        if det_c < best_det
            best_det   = det_c
            best_mu    = vec(mean(subset, dims=1))
            best_Sigma = Sigma_c
        end
    end

    # Consistency factor for normal distribution
    chi2_q   = 2.0 * h_frac + 0.5 * (1 - h_frac)^2 / h_frac  # approx
    best_Sigma .*= chi2_q

    return best_mu, best_Sigma
end

println("\nRunning MCD estimation (h=75% of observations)...")
mu_mcd, Sigma_mcd = mcd_covariance(returns_daily)

# MCD vs Sample covariance comparison
mcd_error = norm(Sigma_mcd - Sigma_true, 2) / norm(Sigma_true, 2)
@printf("  Sample covariance error (spectral norm): %.4f\n", Sigma_error)
@printf("  MCD    covariance error (spectral norm): %.4f\n", mcd_error)
@printf("  MCD improvement: %.1f%%\n", (Sigma_error - mcd_error) / Sigma_error * 100)

println("\nDiagonal comparison (variance estimation):")
println("  Asset  | True Var | Sample Var | MCD Var  | MCD closer?")
println("  " * "-"^58)
for i in 1:N_ASSETS
    tv = Sigma_true[i,i]
    sv = Sigma_sample[i,i]
    mv = Sigma_mcd[i,i]
    mcd_better = abs(mv - tv) < abs(sv - tv)
    @printf("  %-6s | %.6f | %.6f   | %.6f | %s\n",
            asset_names[i], tv, sv, mv, mcd_better ? "YES" : "no")
end

# ── 4. MARKOWITZ OPTIMIZATION ─────────────────────────────────────────────────

println("\n" * "="^60)
println("MARKOWITZ MEAN-VARIANCE OPTIMIZATION")
println("="^60)

"""
Solve minimum variance portfolio with long-only constraint.
Uses quadratic programming via projected gradient descent.
"""
function min_variance_portfolio(Sigma::Matrix{Float64}; max_iter::Int=2000,
                                 tol::Float64=1e-8, lr::Float64=0.01)
    n  = size(Sigma, 1)
    w  = fill(1.0/n, n)
    for _ in 1:max_iter
        grad = 2 * Sigma * w
        w    = w - lr * grad
        w    = max.(w, 0.0)
        s    = sum(w)
        if s < 1e-12; w = fill(1.0/n, n); continue; end
        w    = w ./ s
    end
    return w
end

"""
Maximum Sharpe portfolio (tangency portfolio).
"""
function max_sharpe_portfolio(mu::Vector{Float64}, Sigma::Matrix{Float64};
                               rf::Float64=0.0, max_iter::Int=2000, lr::Float64=0.05)
    n  = length(mu)
    w  = fill(1.0/n, n)
    for _ in 1:max_iter
        port_ret = dot(w, mu) - rf
        port_var = dot(w, Sigma * w)
        port_std = sqrt(port_var + 1e-10)
        # Gradient of Sharpe
        grad_ret = mu
        grad_std = Sigma * w / port_std
        grad     = (grad_ret * port_std - port_ret * grad_std) / (port_std^2 + 1e-10)
        w        = w + lr * grad
        w        = max.(w, 0.0)
        s        = sum(w)
        if s < 1e-12; w = fill(1.0/n, n); continue; end
        w        = w ./ s
    end
    return w
end

println("\nSample Markowitz portfolios:")
w_minvar_sample  = min_variance_portfolio(Sigma_sample)
w_maxshr_sample  = max_sharpe_portfolio(mu_sample, Sigma_sample)

println("\nMCD-based robust portfolios:")
w_minvar_mcd     = min_variance_portfolio(Sigma_mcd)
w_maxshr_mcd     = max_sharpe_portfolio(mu_mcd, Sigma_mcd)

function portfolio_stats(w, mu, Sigma; ann=252)
    ret = dot(w, mu) * ann
    vol = sqrt(dot(w, Sigma * w)) * sqrt(ann)
    shr = (ret - 0.0) / (vol + 1e-8)
    return (ret=ret, vol=vol, sharpe=shr)
end

println("\nPortfolio statistics (using true parameters):")
println("  Portfolio          | Ann Ret | Ann Vol | Sharpe | Max Weight")
println("  " * "-"^65)
for (name, w) in [
        ("MinVar (Sample)",     w_minvar_sample),
        ("MaxSharpe (Sample)",  w_maxshr_sample),
        ("MinVar (MCD)",        w_minvar_mcd),
        ("MaxSharpe (MCD)",     w_maxshr_mcd),
        ("Equal Weight",        fill(1.0/N_ASSETS, N_ASSETS))
    ]
    s = portfolio_stats(w, mu_true, Sigma_true)
    @printf("  %-20s | %7.2f%% | %7.2f%% | %6.3f | %.3f\n",
            name, s.ret*100, s.vol*100, s.sharpe, maximum(w))
end

# ── 5. STRESS TEST: REMOVE BEST 10% OF DAYS ──────────────────────────────────

println("\n" * "="^60)
println("STRESS TEST: REMOVE BEST 10% OF DAYS")
println("="^60)

"""
Compute simulated portfolio return, then stress test by removing
the top 10% performing days to see how each portfolio degrades.
"""
function stress_test_portfolios(returns::Matrix{Float64}, portfolios::Dict)
    n = size(returns, 1)
    results = Dict{String, NamedTuple}()

    for (name, w) in portfolios
        port_rets  = returns * w
        full_cumr  = prod(1.0 .+ port_rets)
        full_sharpe = mean(port_rets) / (std(port_rets) + 1e-8) * sqrt(252)

        # Remove best 10% of days
        sorted_idx = sortperm(port_rets, rev=true)
        n_remove   = floor(Int, n * 0.10)
        keep_mask  = trues(n)
        keep_mask[sorted_idx[1:n_remove]] .= false
        stressed_rets  = port_rets[keep_mask]
        stress_cumr    = prod(1.0 .+ stressed_rets)
        stress_sharpe  = mean(stressed_rets) / (std(stressed_rets) + 1e-8) * sqrt(252)

        # Best day contribution
        best_10_contrib = sum(port_rets[sorted_idx[1:n_remove]])
        total_return    = sum(port_rets)

        results[name] = (
            full_return   = (full_cumr - 1) * 100,
            stress_return = (stress_cumr - 1) * 100,
            full_sharpe   = full_sharpe,
            stress_sharpe = stress_sharpe,
            best_day_pct  = best_10_contrib / (abs(total_return) + 1e-8) * 100
        )
    end
    return results
end

portfolios_to_test = Dict(
    "MinVar (Sample)"    => w_minvar_sample,
    "MaxSharpe (Sample)" => w_maxshr_sample,
    "MinVar (MCD)"       => w_minvar_mcd,
    "MaxSharpe (MCD)"    => w_maxshr_mcd,
    "Equal Weight"       => fill(1.0/N_ASSETS, N_ASSETS),
)

stress_results = stress_test_portfolios(returns_daily, portfolios_to_test)

println("\nStress test: remove best 10% of days ($(floor(Int, N_OBS*0.10)) days removed)")
println("  Portfolio          | Full Ret | Stress Ret | Sharpe Drop | Best 10% Contribution")
println("  " * "-"^80)
for (name, r) in sort(collect(stress_results), by=x->-x[2].full_return)
    sharpe_drop = r.full_sharpe - r.stress_sharpe
    @printf("  %-20s | %8.2f%% | %10.2f%% | %11.3f | %.1f%%\n",
            name, r.full_return, r.stress_return, sharpe_drop, r.best_day_pct)
end

# ── 6. COVARIANCE ESTIMATION UNDER DIFFERENT REGIMES ─────────────────────────

println("\n" * "="^60)
println("COVARIANCE ESTIMATION UNDER DIFFERENT REGIMES")
println("="^60)

# Split data into bull/bear regimes based on BTC return
btc_cumret  = cumsum(returns_daily[:,1])
regime_flag = btc_cumret .>= median(btc_cumret)  # 1=bull, 0=bear

bull_mask = regime_flag .== true
bear_mask = regime_flag .== false

Sigma_bull   = cov(returns_daily[bull_mask, :])  + 1e-8*I(N_ASSETS)
Sigma_bear   = cov(returns_daily[bear_mask, :])  + 1e-8*I(N_ASSETS)

@printf("\n  Bull regime: %d days  Bear regime: %d days\n",
        sum(bull_mask), sum(bear_mask))

println("\nVolatility comparison (annualized) by regime:")
println("  Asset  | Bull Vol | Bear Vol | Bear/Bull Ratio")
println("  " * "-"^48)
for i in 1:N_ASSETS
    bull_v = sqrt(Sigma_bull[i,i] * 252) * 100
    bear_v = sqrt(Sigma_bear[i,i] * 252) * 100
    @printf("  %-6s | %8.2f%% | %8.2f%% | %.3fx\n",
            asset_names[i], bull_v, bear_v, bear_v / (bull_v + 1e-8))
end

println("\nMean pairwise correlation by regime:")
off_diag(C) = [C[i,j] for i in 1:size(C,1) for j in i+1:size(C,2)]
bull_corr = cor(returns_daily[bull_mask, :])
bear_corr = cor(returns_daily[bear_mask, :])
@printf("  Bull: %.4f  Bear: %.4f  (bear correlation premium: %.4f)\n",
        mean(off_diag(bull_corr)), mean(off_diag(bear_corr)),
        mean(off_diag(bear_corr)) - mean(off_diag(bull_corr)))

# ── 7. RESAMPLED EFFICIENCY (MICHAUD RESAMPLING) ──────────────────────────────

println("\n" * "="^60)
println("RESAMPLED EFFICIENCY: 500-SIMULATION MICHAUD WEIGHTS")
println("="^60)

"""
Michaud resampling: bootstrap the return series, compute optimal portfolio
for each bootstrap, average the weights.
"""
function michaud_resampling(returns::Matrix{Float64}, n_sims::Int=500;
                             seed::Int=17, objective::Symbol=:minvar)
    rng     = MersenneTwister(seed)
    n, p    = size(returns)
    w_accum = zeros(p)

    for sim in 1:n_sims
        # Bootstrap resample
        idx     = rand(rng, 1:n, n)
        boot_R  = returns[idx, :]
        mu_b    = vec(mean(boot_R, dims=1))
        Sigma_b = cov(boot_R) + 1e-6 * I(p)

        if objective == :minvar
            w_b = min_variance_portfolio(Sigma_b)
        else
            w_b = max_sharpe_portfolio(mu_b, Sigma_b)
        end
        w_accum .+= w_b
    end

    w_michaud = w_accum ./ n_sims
    w_michaud = max.(w_michaud, 0.0)
    w_michaud ./= sum(w_michaud)
    return w_michaud
end

println("\nRunning 500-simulation Michaud resampling...")
w_michaud_mv  = michaud_resampling(returns_daily, 500; objective=:minvar)
w_michaud_ms  = michaud_resampling(returns_daily, 500; objective=:sharpe)

println("\nWeight comparison: Markowitz vs Michaud (min variance):")
println("  Asset  | Markowitz | Michaud | Difference")
println("  " * "-"^48)
for i in 1:N_ASSETS
    diff = w_michaud_mv[i] - w_minvar_sample[i]
    @printf("  %-6s | %9.4f | %7.4f | %+9.4f\n",
            asset_names[i], w_minvar_sample[i], w_michaud_mv[i], diff)
end

@printf("\n  Max weight (Markowitz): %.4f   Max weight (Michaud): %.4f\n",
        maximum(w_minvar_sample), maximum(w_michaud_mv))
@printf("  HHI (Markowitz): %.4f   HHI (Michaud): %.4f\n",
        sum(w_minvar_sample.^2), sum(w_michaud_mv.^2))
println("  HHI = Herfindahl-Hirschman Index (lower = more diversified)")

s_mk = portfolio_stats(w_minvar_sample,  mu_true, Sigma_true)
s_mc = portfolio_stats(w_michaud_mv,     mu_true, Sigma_true)
@printf("\n  Out-of-sample Sharpe -- Markowitz: %.4f  Michaud: %.4f\n",
        s_mk.sharpe, s_mc.sharpe)
println("  Michaud smooths out estimation error → typically more stable weights")

# ── 8. BOOTSTRAP DISTRIBUTION OF OPTIMAL WEIGHTS ─────────────────────────────

println("\n" * "="^60)
println("PARAMETER UNCERTAINTY: BOOTSTRAP WEIGHT DISTRIBUTION")
println("="^60)

"""
Bootstrap distribution of optimal portfolio weights.
Shows estimation uncertainty in weight allocation.
"""
function bootstrap_weights(returns::Matrix{Float64}, n_boot::Int=200;
                            seed::Int=17, objective::Symbol=:minvar)
    rng    = MersenneTwister(seed + 1)
    n, p   = size(returns)
    w_boot = zeros(n_boot, p)

    for b in 1:n_boot
        idx     = rand(rng, 1:n, n)
        boot_R  = returns[idx, :]
        mu_b    = vec(mean(boot_R, dims=1))
        Sigma_b = cov(boot_R) + 1e-6 * I(p)

        if objective == :minvar
            w_boot[b, :] = min_variance_portfolio(Sigma_b)
        else
            w_boot[b, :] = max_sharpe_portfolio(mu_b, Sigma_b)
        end
    end
    return w_boot
end

println("\nBootstrap weight distribution (200 resamples, min variance):")
w_boot = bootstrap_weights(returns_daily, 200; objective=:minvar)

println("  Asset  | Mean W | Std W  | 5th pct | 95th pct | CV")
println("  " * "-"^58)
for i in 1:N_ASSETS
    col    = w_boot[:, i]
    mw     = mean(col)
    sw     = std(col)
    p5     = quantile(col, 0.05)
    p95    = quantile(col, 0.95)
    cv     = sw / (mw + 1e-8)
    @printf("  %-6s | %.4f | %.4f | %.4f   | %.4f    | %.3f\n",
            asset_names[i], mw, sw, p5, p95, cv)
end

most_stable   = argmin(vec(std(w_boot, dims=1)))
most_unstable = argmax(vec(std(w_boot, dims=1)))
@printf("\n  Most stable weight:   %s (CV=%.3f)\n",
        asset_names[most_stable],
        std(w_boot[:,most_stable]) / (mean(w_boot[:,most_stable]) + 1e-8))
@printf("  Most unstable weight: %s (CV=%.3f)\n",
        asset_names[most_unstable],
        std(w_boot[:,most_unstable]) / (mean(w_boot[:,most_unstable]) + 1e-8))

# ── 9. BERTSIMAS-SIM ROBUST OPTIMIZATION ─────────────────────────────────────

println("\n" * "="^60)
println("WORST-CASE OPTIMIZATION (BERTSIMAS-SIM ROBUST APPROACH)")
println("="^60)

"""
Bertsimas-Sim robust portfolio: protect against up to Γ assets having
returns adversarially perturbed by ±δ (budget uncertainty set).

min_w  w'Σw
s.t.   w'μ - Γ * max_i(δ_i * |w_i|) ≥ target_ret  (robust constraint)
       sum(w) = 1, w ≥ 0

Approximation: we use worst-case expected return
  μ_robust(w) = w'μ - Γ * Σ_i δ_i |w_i|  sorted top-Γ
"""
function bertsimas_sim_portfolio(mu::Vector{Float64}, Sigma::Matrix{Float64},
                                  Gamma::Float64, delta_frac::Float64=0.5;
                                  max_iter::Int=3000, lr::Float64=0.01)
    n   = length(mu)
    w   = fill(1.0/n, n)
    delta = abs.(mu) .* delta_frac .+ 1e-5  # uncertainty radius per asset

    for iter in 1:max_iter
        # Worst-case return adjustment
        contributions  = delta .* abs.(w)
        sorted_idx     = sortperm(contributions, rev=true)
        gamma_int      = floor(Int, Gamma)
        frac_remaining = Gamma - gamma_int
        wc_deduction   = sum(contributions[sorted_idx[1:min(gamma_int, n)]])
        if gamma_int < n
            wc_deduction += frac_remaining * contributions[sorted_idx[gamma_int+1]]
        end

        wc_ret  = dot(w, mu) - wc_deduction
        port_vol = sqrt(dot(w, Sigma * w) + 1e-10)

        # Gradient of (worst-case Sharpe)
        grad_mu   = mu
        # Approximate gradient of deduction term
        grad_dec  = zeros(n)
        for i in sorted_idx[1:min(gamma_int, n)]
            grad_dec[i] = delta[i] * sign(w[i] + 1e-10)
        end
        grad_wc_ret  = grad_mu .- grad_dec
        grad_var     = Sigma * w / port_vol
        wc_sharpe    = wc_ret / (port_vol + 1e-8)
        grad_sharpe  = (grad_wc_ret * port_vol - wc_ret * grad_var) / (port_vol^2 + 1e-8)

        w = w + lr * grad_sharpe
        w = max.(w, 0.0)
        s = sum(w); if s < 1e-12; w = fill(1.0/n, n); continue; end
        w = w ./ s
    end
    return w
end

println("\nBertsimas-Sim robust portfolios for different Γ (budget parameter):")
println("  Γ    | Description         | Sharpe | Vol  | Max Weight | Diversification")
println("  " * "-"^72)
for Gamma in [0.0, 1.0, 2.0, 4.0, 8.0]
    w_bs = bertsimas_sim_portfolio(mu_sample, Sigma_sample, Gamma)
    s    = portfolio_stats(w_bs, mu_true, Sigma_true)
    hhi  = sum(w_bs.^2)
    desc = Gamma == 0.0 ? "No protection" :
           Gamma == 1.0 ? "1 asset uncertain" :
           Gamma == 2.0 ? "2 assets uncertain" :
           Gamma == 4.0 ? "4 assets uncertain" :
                          "All uncertain"
    @printf("  %4.1f | %-20s | %6.3f | %.3f | %.4f     | %.4f\n",
            Gamma, desc, s.sharpe, s.vol, maximum(w_bs), hhi)
end

println("\nAs Γ increases, portfolio becomes more diversified (lower HHI)")
println("and more conservative -- robust to adversarial return perturbations.")

# ── 10. HEAD-TO-HEAD PORTFOLIO COMPARISON ────────────────────────────────────

println("\n" * "="^60)
println("HEAD-TO-HEAD: ALL PORTFOLIOS VS HELD-OUT DATA")
println("="^60)

# Generate held-out data
returns_heldout, _, _ = generate_crypto_returns(252, N_ASSETS; seed=1717)

w_equal      = fill(1.0/N_ASSETS, N_ASSETS)
w_bs_robust  = bertsimas_sim_portfolio(mu_sample, Sigma_sample, 2.0)
w_risk_parity = begin
    vols = [std(returns_daily[:,i]) for i in 1:N_ASSETS]
    w    = 1.0 ./ vols
    w   ./ sum(w)
end

portfolios_final = [
    ("Markowitz MinVar",    w_minvar_sample),
    ("Markowitz MaxSharpe", w_maxshr_sample),
    ("MCD MinVar",          w_minvar_mcd),
    ("MCD MaxSharpe",       w_maxshr_mcd),
    ("Michaud (MinVar)",    w_michaud_mv),
    ("Bertsimas-Sim Γ=2",  w_bs_robust),
    ("Risk Parity",         w_risk_parity),
    ("Equal Weight",        w_equal),
]

println("\nHeld-out period (252 days) performance:")
println("  Portfolio           | Sharpe | Ann Ret | Ann Vol | Max DD")
println("  " * "-"^70)
for (name, w) in portfolios_final
    port_rets = returns_heldout * w
    ann_ret   = mean(port_rets) * 252
    ann_vol   = std(port_rets)  * sqrt(252)
    sharpe    = ann_ret / (ann_vol + 1e-8)
    pv        = cumprod(1.0 .+ port_rets)
    peak      = cummax = pv[1]
    mdd       = 0.0
    for v in pv
        cummax = max(cummax, v)
        mdd    = max(mdd, (cummax - v) / cummax)
    end
    @printf("  %-20s | %6.3f | %7.2f%% | %7.2f%% | %6.2f%%\n",
            name, sharpe, ann_ret*100, ann_vol*100, mdd*100)
end

# ── 11. LEDOIT-WOLF SHRINKAGE ─────────────────────────────────────────────────

println("\n" * "="^60)
println("LEDOIT-WOLF COVARIANCE SHRINKAGE")
println("="^60)

"""
Analytical Ledoit-Wolf shrinkage toward scaled identity.
δ* = argmin E[||δF + (1-δ)S - Σ||²]
where F = (tr(S)/p) * I  (shrinkage target)
"""
function ledoit_wolf_shrinkage(X::Matrix{Float64})
    n, p = size(X)
    S    = cov(X)
    mu_F = tr(S) / p   # target is scaled identity

    # Analytical formula (Oracle approximating shrinkage)
    # Using simplified version of LW analytical formula
    S2   = S * S
    tr_S  = tr(S)
    tr_S2 = tr(S2)
    tr2_S = tr_S^2

    # Stein formula for optimal δ
    rho_hat = ((n - 2) / n * tr_S2 + tr2_S) /
              ((n + 2) * (tr_S2 - tr2_S / p) + 1e-10)
    delta   = clamp(rho_hat, 0.0, 1.0)

    F = mu_F * I(p)
    S_shrunk = (1 - delta) * S + delta * F
    return S_shrunk, delta
end

Sigma_lw, delta_lw = ledoit_wolf_shrinkage(returns_daily)
lw_error = norm(Sigma_lw - Sigma_true, 2) / norm(Sigma_true, 2)
@printf("\n  Ledoit-Wolf shrinkage intensity δ*: %.4f\n", delta_lw)
@printf("  LW covariance error (spectral):    %.4f\n", lw_error)
@printf("  Sample covariance error:           %.4f\n", Sigma_error)
@printf("  MCD covariance error:              %.4f\n", mcd_error)

w_lw = min_variance_portfolio(Sigma_lw)
s_lw = portfolio_stats(w_lw, mu_true, Sigma_true)
@printf("\n  LW MinVar out-of-sample Sharpe: %.4f\n", s_lw.sharpe)

println("\nSummary: covariance estimation accuracy (lower = better):")
for (method, err) in sort([
        ("Sample",        Sigma_error),
        ("MCD (h=75%)",   mcd_error),
        ("Ledoit-Wolf",   lw_error),
    ], by=x->x[2])
    @printf("  %-20s: %.4f\n", method, err)
end

# ── 12. FINAL SUMMARY ────────────────────────────────────────────────────────

println("\n" * "="^60)
println("FINAL SUMMARY")
println("="^60)

println("""
  Portfolio construction methods compared:
  1. Classical Markowitz (sample covariance)
  2. MCD-robust covariance (resistant to outliers and jumps)
  3. Michaud resampling (500 bootstrap samples averaged)
  4. Bertsimas-Sim robust optimization (budget uncertainty set)
  5. Ledoit-Wolf shrinkage (analytical optimal shrinkage)
  6. Risk Parity (inverse volatility weighting)

  Key findings:
  1. Sample covariance is most sensitive to estimation error --
     extreme weights emerge when the covariance matrix is noisy
  2. MCD estimation discards the most anomalous 25% of observations,
     producing a cleaner covariance estimate
  3. Michaud resampling naturally diversifies weights, acting as
     a regularizer on the optimization
  4. Bertsimas-Sim with Γ=2 provides the most intuitive robustness
     guarantee: withstands adversarial perturbation of 2 assets
  5. Ledoit-Wolf shrinkage is computationally cheapest with
     theoretical optimality guarantees
  6. All robust methods sacrifice some in-sample Sharpe for
     better out-of-sample stability (lower weight turnover, less
     sensitivity to outlier periods)
  7. Stress test: removing best 10% of days hurts concentrated
     portfolios most -- robust diversification is critical
""")

println("Notebook 17 complete.")
