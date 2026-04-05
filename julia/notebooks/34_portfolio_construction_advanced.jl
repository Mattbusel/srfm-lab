# ============================================================
# Notebook 34: Advanced Portfolio Construction Methods
# ============================================================
# Topics:
#   1. Black-Litterman model
#   2. Robust optimization (Ellipsoidal uncertainty)
#   3. Risk Parity and Equal Risk Contribution
#   4. Maximum Diversification portfolio
#   5. Factor-constrained optimization
#   6. Regime-dependent allocation
#   7. Dynamic risk budgeting
#   8. Transaction-cost-aware rebalancing
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 34: Advanced Portfolio Construction")
println("="^60)

# ── RNG ───────────────────────────────────────────────────
rng_s = UInt64(314159)
function rnd()
    global rng_s
    rng_s = rng_s * 6364136223846793005 + 1442695040888963407
    (rng_s >> 11) / Float64(2^53)
end
function rndn()
    u1 = max(rnd(), 1e-15); u2 = rnd()
    sqrt(-2.0*log(u1)) * cos(2π*u2)
end

# ── Section 0: Market Setup ───────────────────────────────

n_assets = 10
asset_names = ["SPY","QQQ","IWM","EFA","EEM","TLT","GLD","BTC","REIT","HY"]

# Covariance matrix and expected returns
sigma_vols = [0.17, 0.22, 0.24, 0.20, 0.25, 0.12, 0.16, 0.80, 0.20, 0.10]
corr_matrix = Matrix{Float64}(I, n_assets, n_assets)
base_corr = [1.00  0.90  0.85  0.80  0.70 -0.30  0.10  0.30  0.75  0.60;
              0.90  1.00  0.80  0.75  0.65 -0.35  0.05  0.35  0.70  0.55;
              0.85  0.80  1.00  0.75  0.70 -0.30  0.10  0.25  0.72  0.58;
              0.80  0.75  0.75  1.00  0.80 -0.20  0.15  0.20  0.65  0.50;
              0.70  0.65  0.70  0.80  1.00 -0.15  0.20  0.25  0.60  0.45;
             -0.30 -0.35 -0.30 -0.20 -0.15  1.00  0.30 -0.20  0.00  0.20;
              0.10  0.05  0.10  0.15  0.20  0.30  1.00  0.30  0.10  0.05;
              0.30  0.35  0.25  0.20  0.25 -0.20  0.30  1.00  0.20  0.15;
              0.75  0.70  0.72  0.65  0.60  0.00  0.10  0.20  1.00  0.55;
              0.60  0.55  0.58  0.50  0.45  0.20  0.05  0.15  0.55  1.00]

Sigma = base_corr .* (sigma_vols * sigma_vols')
# Ensure positive definite
Sigma = Sigma + 1e-4 * I

# Market cap weights (proxy)
mkt_weights = [0.30, 0.15, 0.08, 0.12, 0.08, 0.10, 0.04, 0.02, 0.06, 0.05]
mkt_weights ./= sum(mkt_weights)

# Risk-free rate
r_f = 0.05

println("Asset universe: $(join(asset_names, ", "))")
println("Market weights: $(join([round(w*100,digits=1) for w in mkt_weights], "%, ")*"%")")

# ── Section 1: Black-Litterman Model ─────────────────────

println("\n--- Section 1: Black-Litterman Model ---")

# BL: π = λ * Σ * w_mkt (equilibrium returns)
# Posterior: μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]

tau = 0.05   # scaling parameter
lambda_bl = 3.0  # risk aversion (calibrated to market SR)

# Equilibrium excess returns
pi_eq = lambda_bl .* (Sigma * mkt_weights)
println("Equilibrium excess returns (Black-Litterman):")
for (i, name) in enumerate(asset_names)
    println("  $(lpad(name, 4)): π=$(round(pi_eq[i]*100, digits=2))%")
end

# Views (investor views):
# View 1: BTC will outperform EEM by 20% (absolute)
# View 2: TLT will return 5% (absolute)
# View 3: SPY will outperform EFA by 3%
P = [0.0  0.0  0.0  0.0 -1.0  0.0  0.0  1.0  0.0  0.0;  # BTC > EEM by 20%
     0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0;   # TLT = 5%
     1.0  0.0  0.0 -1.0  0.0  0.0  0.0  0.0  0.0  0.0]   # SPY > EFA by 3%
Q = [0.20, 0.05, 0.03]

# Confidence in views
omega_diag = [0.05^2, 0.02^2, 0.01^2]  # uncertainty in each view
Omega = diagm(omega_diag)

# BL posterior
tau_Sigma = tau .* Sigma
tau_Sigma_inv = inv(tau_Sigma + 1e-8*I)
PtOmega_inv_P = P' * inv(Omega) * P
mu_bl = inv(tau_Sigma_inv + PtOmega_inv_P) * (tau_Sigma_inv * pi_eq + P' * inv(Omega) * Q)

println("\nBlack-Litterman posterior returns:")
for (i, name) in enumerate(asset_names)
    diff = (mu_bl[i] - pi_eq[i]) * 100
    println("  $(lpad(name, 4)): BL=$(round(mu_bl[i]*100,digits=2))%, π=$(round(pi_eq[i]*100,digits=2))%, Δ=$(round(diff,digits=2))%")
end

# BL optimal weights
Sigma_bl = Sigma .+ tau_Sigma  # posterior covariance
w_bl = (Sigma_bl + 1e-8*I) \ (mu_bl .- r_f) ./ lambda_bl
w_bl = max.(w_bl, 0.0)
w_bl ./= sum(w_bl)

println("\nBlack-Litterman portfolio weights:")
for (i, name) in enumerate(asset_names)
    println("  $(lpad(name, 4)): $(round(w_bl[i]*100, digits=1))%  (vs mkt $(round(mkt_weights[i]*100,digits=1))%)")
end

# ── Section 2: Risk Parity / ERC ─────────────────────────

println("\n--- Section 2: Risk Parity & Equal Risk Contribution ---")

function portfolio_variance(w, Sigma)
    return dot(w, Sigma * w)
end

function risk_contributions(w, Sigma)
    port_vol = sqrt(portfolio_variance(w, Sigma))
    marginal_risk = Sigma * w
    rc = w .* marginal_risk ./ max(port_vol, 1e-12)
    return rc
end

# ERC: find weights such that all risk contributions are equal
function erc_weights(Sigma; n_iter=1000, tol=1e-8)
    n = size(Sigma, 1)
    w = fill(1.0/n, n)  # initialize equal weight

    for _ in 1:n_iter
        rc = risk_contributions(w, Sigma)
        port_vol = sqrt(portfolio_variance(w, Sigma))
        # Update: Newton-like step
        target_rc = port_vol / n
        for i in 1:n
            marginal = Sigma[i,:] ⋅ w
            w[i] *= target_rc / max(rc[i], 1e-12)
        end
        w = max.(w, 1e-8)
        w ./= sum(w)
        # Check convergence
        rc_new = risk_contributions(w, Sigma)
        if std(rc_new) < tol * mean(rc_new)
            break
        end
    end
    return w
end

w_erc = erc_weights(Sigma)
w_ew = fill(1.0/n_assets, n_assets)
w_mvo = max.((Sigma + 1e-6*I) \ (pi_eq .- r_f) ./ lambda_bl, 0.0)
w_mvo ./= sum(w_mvo)

println("Risk contribution comparison:")
println("  Asset | ERC weight | RC(ERC) | EW weight | RC(EW)  | MVO weight | RC(MVO)")
println("  " * "-"^76)
rc_erc = risk_contributions(w_erc, Sigma)
rc_ew  = risk_contributions(w_ew, Sigma)
rc_mvo = risk_contributions(w_mvo, Sigma)
for (i, name) in enumerate(asset_names)
    println("  $(lpad(name,4))  | $(lpad(round(w_erc[i]*100,digits=1),10))% | $(lpad(round(rc_erc[i]*100,digits=2),7))% | " *
            "$(lpad(round(w_ew[i]*100,digits=1),9))% | $(lpad(round(rc_ew[i]*100,digits=2),7))% | " *
            "$(lpad(round(w_mvo[i]*100,digits=1),10))% | $(round(rc_mvo[i]*100,digits=2))%")
end

for (label, w) in [("ERC", w_erc), ("Equal Weight", w_ew), ("MVO", w_mvo)]
    vol = sqrt(portfolio_variance(w, Sigma)) * 100
    ret = dot(w, pi_eq + r_f) * 100
    sharpe = (ret - r_f*100) / vol
    rc = risk_contributions(w, Sigma)
    rc_gini = std(rc) / mean(rc)
    println("  $(lpad(label, 12)): Vol=$(round(vol,digits=2))%, Ret=$(round(ret,digits=2))%, Sharpe=$(round(sharpe,digits=2)), RC-Gini=$(round(rc_gini,digits=3))")
end

# ── Section 3: Maximum Diversification ───────────────────

println("\n--- Section 3: Maximum Diversification Portfolio ---")

# Diversification ratio: (sum(w_i * sigma_i)) / portfolio_vol
function diversification_ratio(w, Sigma, vols)
    port_vol = sqrt(max(dot(w, Sigma * w), 1e-12))
    weighted_vols = dot(w, vols)
    return weighted_vols / port_vol
end

# Maximize DR by gradient ascent on the Lagrangian
function max_diversification(Sigma, vols; n_iter=500)
    n = length(vols)
    w = vols ./ sum(vols)  # initialize proportional to vols

    for _ in 1:n_iter
        port_var = dot(w, Sigma * w)
        port_vol = sqrt(max(port_var, 1e-12))
        weighted_sum_vols = dot(w, vols)

        # Gradient of DR w.r.t. w
        grad_num = vols ./ port_vol
        grad_denom = -(weighted_sum_vols * (Sigma * w)) ./ (port_vol^2)
        grad = (grad_num .+ grad_denom) ./ port_vol

        # Projected gradient step
        w_new = w .+ 0.01 .* grad
        w_new = max.(w_new, 0.0)
        w_new ./= sum(w_new)
        if norm(w_new - w) < 1e-8; break; end
        w = w_new
    end
    return w
end

w_mxd = max_diversification(Sigma, sigma_vols)
dr_mxd = diversification_ratio(w_mxd, Sigma, sigma_vols)
dr_ew  = diversification_ratio(w_ew,  Sigma, sigma_vols)
dr_mvo = diversification_ratio(w_mvo, Sigma, sigma_vols)
dr_erc = diversification_ratio(w_erc, Sigma, sigma_vols)

println("Diversification ratios:")
println("  MXD portfolio: DR=$(round(dr_mxd, digits=4))")
println("  ERC portfolio: DR=$(round(dr_erc, digits=4))")
println("  Equal weight:  DR=$(round(dr_ew, digits=4))")
println("  MVO portfolio: DR=$(round(dr_mvo, digits=4))")
println("\nMax-Diversification weights:")
for (i, name) in enumerate(asset_names)
    if w_mxd[i] > 0.005
        println("  $(lpad(name,4)): $(round(w_mxd[i]*100, digits=1))%")
    end
end

# ── Section 4: Robust Optimization ───────────────────────

println("\n--- Section 4: Robust Optimization ---")

# Robust MVO: min_w w'Σw - λ(μ'w - kappa*||Σ_mu^{0.5}*w||)
# where kappa controls uncertainty in expected returns
# Approximation: shrink expected returns toward zero

function robust_mvo(mu, Sigma, lambda_ro, kappa; n_iter=200)
    n = length(mu)
    w = fill(1.0/n, n)
    for _ in 1:n_iter
        port_vol = sqrt(max(dot(w, Sigma*w), 1e-12))
        # Robust return adjustment
        mu_robust = mu .- kappa .* (Sigma * w) ./ max(port_vol, 1e-12)
        # MVO update
        w_new = max.((Sigma + 1e-6*I) \ (mu_robust .- r_f) ./ lambda_ro, 0.0)
        s = sum(w_new)
        w_new = s > 0 ? w_new ./ s : fill(1.0/n, n)
        if norm(w_new - w) < 1e-8; break; end
        w = w_new
    end
    return w
end

kappas = [0.0, 0.1, 0.3, 0.5, 1.0]
println("Robust optimization: Effect of uncertainty aversion (kappa):")
println("  Kappa | Exp Return | Vol     | Sharpe | HHI (concentration)")
println("  " * "-"^55)
for kappa in kappas
    w_rob = robust_mvo(pi_eq .+ r_f, Sigma, lambda_bl, kappa)
    exp_ret = dot(w_rob, pi_eq .+ r_f) * 100
    vol = sqrt(portfolio_variance(w_rob, Sigma)) * 100
    sharpe = vol > 0 ? (exp_ret - r_f*100) / vol : 0.0
    hhi = sum(w_rob.^2)
    println("  $(lpad(kappa, 5)) | $(lpad(round(exp_ret,digits=2),10))% | $(lpad(round(vol,digits=2),7))% | $(lpad(round(sharpe,digits=2),6)) | $(round(hhi, digits=4))")
end

# ── Section 5: Regime-Dependent Allocation ───────────────

println("\n--- Section 5: Regime-Dependent Allocation ---")

# Regime-specific expected returns
regime_adjustments = Dict(
    :bull      => [0.05, 0.07, 0.08, 0.04, 0.06, -0.02, 0.0, 0.15, 0.04, 0.02],
    :bear      => [-0.05,-0.07,-0.08,-0.04,-0.06, 0.08, 0.05,-0.10,-0.04,-0.01],
    :sideways  => [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.01, 0.01],
)

regime_probs = Dict(:bull=>0.55, :bear=>0.20, :sideways=>0.25)

# Compute regime-conditional optimal portfolios
portfolios = Dict{Symbol, Vector{Float64}}()
for (regime, adj) in regime_adjustments
    mu_regime = pi_eq .+ adj
    w = max.((Sigma + 1e-6*I) \ (mu_regime .- r_f) ./ lambda_bl, 0.0)
    w ./= sum(w) + 1e-12
    portfolios[regime] = w
end

# Blended portfolio using regime probabilities
w_blended = sum(regime_probs[r] .* portfolios[r] for r in keys(regime_probs))
w_blended ./= sum(w_blended)

println("Regime-conditional portfolios:")
for (regime, adj) in regime_adjustments
    w = portfolios[regime]
    exp_ret = dot(w, pi_eq .+ adj .+ r_f) * 100
    vol = sqrt(portfolio_variance(w, Sigma)) * 100
    println("  $(lpad(string(regime),9)) (p=$(regime_probs[regime])): " *
            "Ret=$(round(exp_ret,digits=1))%, Vol=$(round(vol,digits=1))%, " *
            "Top: $(asset_names[argmax(w)]) ($(round(maximum(w)*100,digits=1))%)")
end

println("\nBlended portfolio (probability-weighted):")
for (i, name) in enumerate(asset_names)
    if w_blended[i] > 0.01
        println("  $(lpad(name,4)): $(round(w_blended[i]*100, digits=1))%")
    end
end

# ── Section 6: Dynamic Risk Budgeting ────────────────────

println("\n--- Section 6: Dynamic Risk Budgeting ---")

# CPPI-inspired dynamic risk budgeting
total_budget = 1.0  # 100% of capital
floor = 0.80         # protect 80% of capital
multiplier = 3.0     # leverage multiplier
risk_asset_vol = sqrt(portfolio_variance(w_erc, Sigma))

n_sim_periods = 252
portfolio_values = zeros(n_sim_periods + 1)
risky_allocations = zeros(n_sim_periods)
portfolio_values[1] = total_budget

for t in 1:n_sim_periods
    cushion = portfolio_values[t] - floor
    risky_alloc = max(min(multiplier * cushion, portfolio_values[t]), 0.0)
    safe_alloc = portfolio_values[t] - risky_alloc
    risky_allocations[t] = risky_alloc / portfolio_values[t]

    # Simulate return
    risky_ret = dot(w_erc, pi_eq) / 252 + risk_asset_vol / sqrt(252) * rndn()
    safe_ret  = r_f / 252

    portfolio_values[t+1] = risky_alloc * (1 + risky_ret) + safe_alloc * (1 + safe_ret)
    portfolio_values[t+1] = max(portfolio_values[t+1], floor * 0.95)  # allow slight breach
end

println("CPPI Dynamic Risk Budgeting (1 year simulation):")
println("  Starting value:     $(round(portfolio_values[1]*100, digits=2))%")
println("  Ending value:       $(round(portfolio_values[end]*100, digits=2))%")
println("  Floor:              $(floor*100)%")
println("  Floor breach:       $(portfolio_values[end] < floor ? "YES" : "no")")
println("  Min value:          $(round(minimum(portfolio_values)*100, digits=2))%")
println("  Avg risky alloc:    $(round(mean(risky_allocations)*100, digits=1))%")
ann_ret = (portfolio_values[end] - 1.0) * 100
println("  Annual return:      $(round(ann_ret, digits=2))%")

# ── Section 7: Transaction-Cost-Aware Rebalancing ────────

println("\n--- Section 7: Transaction-Cost-Aware Rebalancing ---")

# Current portfolio vs target
w_current = mkt_weights  # start at market weights
w_target = w_erc

# Cost of immediate full rebalance
tc_bps = 10.0  # 10 bps per side
immediate_cost = sum(abs.(w_target .- w_current)) / 2.0 * tc_bps / 10_000.0

# Gradual rebalancing: trade fraction δ of gap each period
delta_vals = [0.10, 0.25, 0.50, 1.00]
n_rebal_periods = 20  # monthly rebalancing

println("Rebalancing cost analysis:")
println("  Delta | Periods to 90% | Total TC | Tracking Error Cost")
println("  " * "-"^52)
for delta in delta_vals
    w = copy(w_current)
    total_tc = 0.0
    periods_to_90 = n_rebal_periods
    te_cost_total = 0.0
    for t in 1:n_rebal_periods
        delta_w = w_target .- w
        w_new = w .+ delta .* delta_w
        trade = abs.(w_new .- w)
        tc = sum(trade) / 2.0 * tc_bps / 10_000.0
        total_tc += tc
        # Tracking error cost: how far from target * daily TE penalty
        te = sqrt(portfolio_variance(w_target .- w, Sigma)) * 10.0 / 10_000.0
        te_cost_total += te
        w = w_new
        if norm(w - w_target) < 0.10 * norm(w_current - w_target) && periods_to_90 == n_rebal_periods
            periods_to_90 = t
        end
    end
    println("  $(lpad(delta, 5)) | $(lpad(periods_to_90, 14)) | $(lpad(round(total_tc*10_000,digits=1),8)) bps | $(round(te_cost_total*10_000, digits=1)) bps")
end

# Optimal trading speed: minimize total_TC + TE_cost
println("\nOptimal rebalancing decision:")
# No-trade zone: don't trade if total cost < benefit
total_drift = sum(abs.(w_target .- w_current))
println("  Total drift: $(round(total_drift*100, digits=1))%")
println("  Immediate rebalance cost: $(round(immediate_cost*10_000, digits=1)) bps")
println("  No-trade zone threshold: $(round(tc_bps * 2, digits=0)) bps drift equivalent")

# ── Section 8: Performance Comparison ────────────────────

println("\n--- Section 8: Portfolio Comparison Summary ---")

portfolios_compare = [
    ("Market Cap", mkt_weights),
    ("Equal Weight", w_ew),
    ("MVO", w_mvo),
    ("ERC", w_erc),
    ("Max Div", w_mxd),
    ("Black-Litt", w_bl),
    ("Blended Reg", w_blended),
]

# Simulate returns over n_sim periods
n_periods = 252
println("Monte Carlo portfolio comparison ($n_periods periods, 1000 simulations):")
println("  Portfolio    | E[Return] | E[Vol]  | E[Sharpe] | E[MaxDD]  | E[Diversif]")
println("  " * "-"^72)

function sim_portfolio(w, mu_vec, Sigma, n_p, n_paths=500)
    all_rets = zeros(n_paths)
    all_vols = zeros(n_paths)
    all_dds  = zeros(n_paths)
    port_vol_ann = sqrt(max(portfolio_variance(w, Sigma), 0.0)) * sqrt(252)
    exp_ret_ann  = dot(w, mu_vec) * 252
    for p in 1:n_paths
        rets = zeros(n_p)
        for t in 1:n_p
            r = exp_ret_ann / n_p + port_vol_ann / sqrt(n_p) * rndn()
            rets[t] = r
        end
        all_rets[p] = sum(rets)
        all_vols[p] = std(rets) * sqrt(n_p)
        # Max drawdown
        cum = cumsum(rets); peak = -Inf; dd = 0.0
        for r in cum; peak = max(peak,r); dd = max(dd, peak-r); end
        all_dds[p] = dd
    end
    return mean(all_rets)*100, mean(all_vols)*100, mean(all_dds)*100
end

for (name, w) in portfolios_compare
    e_ret, e_vol, e_dd = sim_portfolio(w, pi_eq .+ r_f, Sigma, n_periods)
    e_sharpe = e_vol > 0 ? (e_ret - r_f*100) / e_vol : 0.0
    dr = diversification_ratio(w, Sigma, sigma_vols)
    println("  $(lpad(name, 12)) | $(lpad(round(e_ret,digits=1),9))% | $(lpad(round(e_vol,digits=1),7))% | " *
            "$(lpad(round(e_sharpe,digits=2),9)) | $(lpad(round(e_dd,digits=1),9))% | $(round(dr, digits=3))")
end

println("\n✓ Notebook 34 complete")
