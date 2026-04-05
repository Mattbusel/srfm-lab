# ============================================================
# Notebook 32: Multi-Factor Attribution & Performance Decomposition
# ============================================================
# Topics:
#   1. Factor model construction (Fama-French style)
#   2. Brinson-Hood-Beebower attribution
#   3. Factor-based attribution
#   4. Active share and tracking error decomposition
#   5. Time-varying factor exposures
#   6. Risk attribution
#   7. Alpha decay and persistence analysis
#   8. Multi-period attribution compounding
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 32: Factor Attribution & Performance")
println("="^60)

# ── RNG ───────────────────────────────────────────────────
rng_s = UInt64(999)
function rnd()
    global rng_s
    rng_s = rng_s * 6364136223846793005 + 1442695040888963407
    (rng_s >> 11) / Float64(2^53)
end
function rndn()
    u1 = max(rnd(), 1e-15); u2 = rnd()
    sqrt(-2.0*log(u1)) * cos(2π*u2)
end

# ── Section 1: Factor Simulation ─────────────────────────

println("\n--- Section 1: Factor Model Setup ---")

n_periods = 60    # monthly, 5 years
n_stocks = 50
n_factors = 5

factor_names = ["Market", "Size", "Value", "Momentum", "Quality"]

# Simulate factor returns
factor_returns = zeros(n_periods, n_factors)
factor_vols = [0.05, 0.03, 0.02, 0.04, 0.02]  # monthly vols
factor_means = [0.007, 0.001, 0.002, 0.003, 0.001]  # monthly means
# Correlation between factors
factor_corr = [1.00  0.10 -0.20  0.00  0.10;
               0.10  1.00  0.15 -0.10  0.20;
              -0.20  0.15  1.00 -0.25  0.10;
               0.00 -0.10 -0.25  1.00 -0.15;
               0.10  0.20  0.10 -0.15  1.00]
L_chol = zeros(n_factors, n_factors)
for i in 1:n_factors
    for j in 1:i
        s = factor_corr[i,j] * factor_vols[i] * factor_vols[j]
        for k in 1:j-1
            s -= L_chol[i,k] * L_chol[j,k]
        end
        L_chol[i,j] = i == j ? sqrt(max(s, 1e-15)) : s / max(L_chol[j,j], 1e-15)
    end
end

for t in 1:n_periods
    z = [rndn() for _ in 1:n_factors]
    factor_returns[t, :] = factor_means .+ L_chol * z
end

# Stock factor exposures (loadings)
stock_betas = zeros(n_stocks, n_factors)
for j in 1:n_stocks
    stock_betas[j, 1] = 0.8 + 0.4 * rndn()   # market beta
    stock_betas[j, 2] = 0.5 * rndn()           # size
    stock_betas[j, 3] = 0.5 * rndn()           # value
    stock_betas[j, 4] = 0.5 * rndn()           # momentum
    stock_betas[j, 5] = 0.3 * rndn()           # quality
end

# Stock returns = factor model + idiosyncratic
stock_returns = zeros(n_periods, n_stocks)
stock_alphas = 0.001 .* [rndn() for _ in 1:n_stocks]  # monthly alpha
for t in 1:n_periods
    for j in 1:n_stocks
        systematic = dot(stock_betas[j,:], factor_returns[t,:])
        idio = 0.04 * rndn()  # idiosyncratic
        stock_returns[t, j] = stock_alphas[j] + systematic + idio
    end
end

println("Factor return summary ($(n_periods) months):")
println("  Factor    | Mean(%/mo) | Std(%/mo) | Sharpe | Cum Return")
println("  " * "-"^55)
for (k, name) in enumerate(factor_names)
    mu = mean(factor_returns[:,k]) * 100
    sig = std(factor_returns[:,k]) * 100
    sharpe = sig > 0 ? mu/sig*sqrt(12) : 0.0
    cum = (prod(1.0 .+ factor_returns[:,k]) - 1.0) * 100
    println("  $(lpad(name,9)) | $(lpad(round(mu,digits=3),10)) | $(lpad(round(sig,digits=3),9)) | $(lpad(round(sharpe,digits=2),6)) | $(round(cum,digits=1))%")
end

# ── Section 2: Portfolio Construction ────────────────────

println("\n--- Section 2: Portfolio Construction ---")

# Benchmark: market-cap weighted (proxy with equal weight)
bench_weights = fill(1.0/n_stocks, n_stocks)

# Active portfolio: tilted toward quality and momentum factors
# Stocks with high quality and momentum get overweight
quality_scores = stock_betas[:, 5] .+ stock_betas[:, 4]  # quality + momentum tilt
active_tilts = quality_scores .- mean(quality_scores)
portfolio_weights = bench_weights .+ 0.02 .* active_tilts
portfolio_weights = max.(portfolio_weights, 0.0)
portfolio_weights ./= sum(portfolio_weights)

println("Portfolio vs Benchmark:")
println("  Active positions: $(sum(abs.(portfolio_weights .- bench_weights) .> 0.001)) stocks")
println("  Active share: $(round(sum(abs.(portfolio_weights .- bench_weights))/2*100, digits=2))%")
println("  Max weight: $(round(maximum(portfolio_weights)*100, digits=2))%")
println("  Min weight: $(round(minimum(portfolio_weights)*100, digits=2))% ($(sum(portfolio_weights .< 0.005)) stocks < 0.5%)")

# Compute portfolio and benchmark returns
port_returns = vec(stock_returns * portfolio_weights)
bench_returns = vec(stock_returns * bench_weights)
active_returns = port_returns .- bench_returns

println("\nPerformance summary ($(n_periods) months):")
for (name, rets) in [("Portfolio", port_returns), ("Benchmark", bench_returns), ("Active", active_returns)]
    ann_ret = mean(rets) * 12 * 100
    ann_std = std(rets) * sqrt(12) * 100
    sharpe = ann_std > 0 ? ann_ret/ann_std : 0.0
    cum = (prod(1.0 .+ rets) - 1.0) * 100
    println("  $(lpad(name,10)): Ann=$(round(ann_ret,digits=2))%, Std=$(round(ann_std,digits=2))%, Sharpe=$(round(sharpe,digits=2)), Cum=$(round(cum,digits=1))%")
end

# ── Section 3: Factor Attribution ────────────────────────

println("\n--- Section 3: Factor-Based Attribution ---")

# Regress portfolio excess returns on factors
function ols_regression(y, X)
    n, p = size(X)
    XtX = X'X + 1e-10*I
    beta = XtX \ (X'y)
    fitted = X * beta
    resid = y .- fitted
    ss_res = sum(resid.^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return beta, resid, r2
end

# Portfolio factor exposures
X_factors = hcat(ones(n_periods), factor_returns)
beta_port, resid_port, r2_port = ols_regression(port_returns, X_factors)
beta_bench, resid_bench, r2_bench = ols_regression(bench_returns, X_factors)

println("Portfolio factor loadings vs Benchmark:")
println("  Factor    | Portfolio | Benchmark | Active Tilt")
println("  " * "-"^48)
println("  Alpha(mo) | $(lpad(round(beta_port[1]*100,digits=3),9)) | $(lpad(round(beta_bench[1]*100,digits=3),9)) | $(round((beta_port[1]-beta_bench[1])*100,digits=4))%")
for (k, name) in enumerate(factor_names)
    bp = beta_port[k+1]
    bb = beta_bench[k+1]
    diff = bp - bb
    println("  $(lpad(name,9)) | $(lpad(round(bp,digits=3),9)) | $(lpad(round(bb,digits=3),9)) | $(round(diff,digits=3))")
end
println("  R²        | $(lpad(round(r2_port,digits=4),9)) | $(lpad(round(r2_bench,digits=4),9)) |")

# Factor contribution to active returns
println("\nFactor attribution of active return:")
println("  Factor    | Factor Ret (ann%) | Active Tilt | Contribution (ann bps)")
println("  " * "-"^62)
total_attributed = 0.0
for (k, name) in enumerate(factor_names)
    factor_ann = mean(factor_returns[:,k]) * 12 * 100  # annualized %
    tilt = beta_port[k+1] - beta_bench[k+1]
    contrib_bps = tilt * mean(factor_returns[:,k]) * 12 * 10_000
    total_attributed += contrib_bps
    println("  $(lpad(name,9)) | $(lpad(round(factor_ann,digits=2),17)) | $(lpad(round(tilt,digits=3),11)) | $(round(contrib_bps, digits=1))")
end
alpha_bps = (beta_port[1] - beta_bench[1]) * 12 * 10_000
println("  $(lpad("Alpha",9)) |                   |             | $(round(alpha_bps, digits=1))")
println("  Total                                          | $(round(total_attributed + alpha_bps, digits=1))")
actual_active_bps = mean(active_returns) * 12 * 10_000
println("  Actual active return:                          | $(round(actual_active_bps, digits=1))")

# ── Section 4: Brinson Attribution ───────────────────────

println("\n--- Section 4: Brinson-Hood-Beebower Attribution ---")

# Sector-level attribution
# Assign stocks to 5 sectors
n_sectors = 5
sector_names = ["Tech", "Finance", "Health", "Energy", "Consumer"]
sector_assign = [((j-1) % n_sectors) + 1 for j in 1:n_stocks]

# Portfolio and benchmark weights by sector
w_sector_port = zeros(n_sectors)
w_sector_bench = zeros(n_sectors)
for j in 1:n_stocks
    s = sector_assign[j]
    w_sector_port[s] += portfolio_weights[j]
    w_sector_bench[s] += bench_weights[j]
end

# Average sector returns
r_sector_port = zeros(n_periods, n_sectors)
r_sector_bench = zeros(n_periods, n_sectors)
for t in 1:n_periods
    for s in 1:n_sectors
        idx = findall(sector_assign .== s)
        if !isempty(idx)
            w_in_s = portfolio_weights[idx] / max(sum(portfolio_weights[idx]), 1e-12)
            r_sector_port[t, s] = dot(w_in_s, stock_returns[t, idx])
            r_sector_bench[t, s] = mean(stock_returns[t, idx])  # equal weight within sector
        end
    end
end

ann_r_sector_port = vec(mean(r_sector_port, dims=1)) .* 12
ann_r_sector_bench = vec(mean(r_sector_bench, dims=1)) .* 12
bench_total_ann = mean(bench_returns) * 12

println("Brinson attribution by sector:")
println("  Sector   | w_port | w_bench | r_port | r_bench | Alloc  | Select | Total")
println("  " * "-"^77)
total_alloc = 0.0; total_select = 0.0
for s in 1:n_sectors
    wp = w_sector_port[s]; wb = w_sector_bench[s]
    rp = ann_r_sector_port[s] * 100; rb = ann_r_sector_bench[s] * 100
    rb_avg = mean(bench_returns) * 12 * 100
    # Allocation effect: (wp - wb) * (rb - rb_total)
    alloc = (wp - wb) * (rb - rb_avg) * 100  # in bps
    # Selection effect: wb * (rp - rb)
    select = wb * (rp - rb) * 100
    total_alloc += alloc; total_select += select
    println("  $(lpad(sector_names[s],8)) | $(lpad(round(wp*100,digits=1),6))% | $(lpad(round(wb*100,digits=1),7))% | " *
            "$(lpad(round(rp,digits=2),6))% | $(lpad(round(rb,digits=2),7))% | $(lpad(round(alloc,digits=1),6)) | $(lpad(round(select,digits=1),6)) | $(round(alloc+select,digits=1))")
end
println("  Total                                                   | $(round(total_alloc,digits=1)) | $(round(total_select,digits=1)) | $(round(total_alloc+total_select,digits=1)) bps/yr")

# ── Section 5: Tracking Error Analysis ───────────────────

println("\n--- Section 5: Tracking Error Decomposition ---")

te = std(active_returns) * sqrt(12) * 100
println("Total tracking error: $(round(te, digits=2))% ann.")

# Decompose TE into factor and idiosyncratic components
# Factor covariance contribution
active_betas = beta_port[2:end] .- beta_bench[2:end]
factor_cov = cov(factor_returns) * 12  # annualized
te_factor_var = dot(active_betas, factor_cov * active_betas)
te_idio_var = var(resid_port .- resid_bench) * 12

te_factor = sqrt(max(te_factor_var, 0.0)) * 100
te_idio = sqrt(max(te_idio_var, 0.0)) * 100
println("  Factor TE:       $(round(te_factor, digits=2))% ($(round(te_factor_var/(te_factor_var+te_idio_var)*100,digits=1))%)")
println("  Idiosyncratic TE: $(round(te_idio, digits=2))% ($(round(te_idio_var/(te_factor_var+te_idio_var)*100,digits=1))%)")
println("  Total (approx):  $(round(sqrt(te_factor_var+te_idio_var)*100,digits=2))%")

# Individual factor TE contributions
println("\nFactor TE contributions:")
for (k, name) in enumerate(factor_names)
    marginal_te_var = active_betas[k]^2 * factor_cov[k,k]
    frac = marginal_te_var / max(te_factor_var, 1e-12) * 100
    println("  $(lpad(name,9)): $(lpad(round(sqrt(marginal_te_var)*100,digits=3),8))% ($(round(frac,digits=1))% of factor TE)")
end

# ── Section 6: Alpha Decay Analysis ──────────────────────

println("\n--- Section 6: Alpha Decay & Persistence ---")

# Compute rolling alpha using 12-month windows
window = 12
rolling_alphas = zeros(n_periods - window)
for t in window+1:n_periods
    y = active_returns[t-window+1:t]
    X_w = hcat(ones(window), factor_returns[t-window+1:t, :])
    beta_w = (X_w'X_w + 1e-10*I) \ (X_w'y)
    rolling_alphas[t-window] = beta_w[1]
end

println("Rolling alpha statistics:")
println("  Mean:     $(round(mean(rolling_alphas)*12*100, digits=2))% ann.")
println("  Std:      $(round(std(rolling_alphas)*12*100, digits=2))% ann.")
println("  % positive periods: $(round(mean(rolling_alphas .> 0)*100, digits=1))%")
println("  Alpha autocorrelation(1): $(round(cor(rolling_alphas[1:end-1], rolling_alphas[2:end]), digits=4))")

# ── Section 7: Risk Attribution ──────────────────────────

println("\n--- Section 7: Risk Attribution ---")

# Portfolio variance decomposition
port_var = var(port_returns)
factor_var = var(factor_returns * stock_betas' * portfolio_weights)
idio_var = var(port_returns) - factor_var

println("Portfolio variance decomposition:")
println("  Total monthly variance: $(round(port_var*100, digits=4))%²")
println("  Factor systematic:      $(round(max(factor_var,0.0)*100, digits=4))%² ($(round(max(factor_var,0.0)/port_var*100,digits=1))%)")
println("  Idiosyncratic:          $(round(max(idio_var,0.0)*100, digits=4))%² ($(round(max(idio_var,0.0)/port_var*100,digits=1))%)")

# Component VaR
for (k, name) in enumerate(factor_names)
    factor_exp = dot(portfolio_weights, stock_betas[:, k])
    comp_var = factor_exp^2 * var(factor_returns[:,k]) / port_var * 100
    println("  $(lpad(name,9)) contribution: $(round(comp_var, digits=1))%")
end

# ── Section 8: Multi-Period Attribution ──────────────────

println("\n--- Section 8: Multi-Period Geometric Attribution ---")

# Compounding attribution over full period
cum_port = prod(1.0 .+ port_returns) - 1.0
cum_bench = prod(1.0 .+ bench_returns) - 1.0
cum_active = (1.0 + cum_port) / (1.0 + cum_bench) - 1.0

println("Cumulative performance ($(n_periods) months):")
println("  Portfolio return:  $(round(cum_port*100, digits=2))%")
println("  Benchmark return:  $(round(cum_bench*100, digits=2))%")
println("  Active return:     $(round(cum_active*100, digits=2))%")

# Menchero geometric linking
println("\nGeometric attribution linking (Menchero approximation):")
sector_cum_port = zeros(n_sectors)
sector_cum_bench = zeros(n_sectors)
for s in 1:n_sectors
    sector_cum_port[s]  = prod(1.0 .+ r_sector_port[:, s]) - 1.0
    sector_cum_bench[s] = prod(1.0 .+ r_sector_bench[:, s]) - 1.0
end

# Linking factors (simplified: use 1+R adjustment)
linking_factor = (1.0 + cum_port) / mean(1.0 .+ port_returns) / n_periods
for s in 1:n_sectors
    alloc_s = (w_sector_port[s] - w_sector_bench[s]) * sector_cum_bench[s]
    select_s = w_sector_bench[s] * (sector_cum_port[s] - sector_cum_bench[s])
    linked_alloc = alloc_s * linking_factor
    linked_select = select_s * linking_factor
    println("  $(lpad(sector_names[s],8)): Alloc=$(round(linked_alloc*100,digits=2))%, Select=$(round(linked_select*100,digits=2))%")
end

println("\nSummary statistics:")
ir = mean(active_returns) / std(active_returns) * sqrt(12)
println("  Information Ratio: $(round(ir, digits=3))")
println("  Hit rate (monthly): $(round(mean(active_returns .> 0)*100, digits=1))%")
println("  Up-market capture:  $(round(mean(port_returns[bench_returns .> 0]) / mean(bench_returns[bench_returns .> 0]) * 100, digits=1))%")
println("  Down-market capture: $(round(mean(port_returns[bench_returns .< 0]) / mean(bench_returns[bench_returns .< 0]) * 100, digits=1))%")

println("\n✓ Notebook 32 complete")
