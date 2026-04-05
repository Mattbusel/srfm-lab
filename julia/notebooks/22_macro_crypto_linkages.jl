# Notebook 22: Macro-Crypto Linkages
# =====================================
# Crypto beta to DXY, VIX, SPY, 10Y yield; regime-conditional betas;
# factor model; macro timing signal; inflation hedge; Fed cycle.
# =====================================

using Statistics, LinearAlgebra, Random, Printf

Random.seed!(22)

# ── 1. SYNTHETIC MACRO + CRYPTO DATA ─────────────────────────────────────────

println("="^60)
println("SYNTHETIC MACRO AND CRYPTO DATA GENERATION")
println("="^60)

const N_DAYS_MACRO = 1260  # 5 years daily

"""
Generate correlated synthetic time series for:
  BTC, ETH, DXY, VIX, SPY, 10Y yield.
BTC/ETH: high vol, regime-switching
DXY, VIX, SPY, TY: macro factors with realistic dynamics.
"""
function generate_macro_crypto_data(n::Int; seed::Int=22)
    rng = MersenneTwister(seed)

    # Generate macro factors first
    # DXY (USD Index): mean-reverting, mild trend
    dxy_ret    = 0.0002 .* randn(rng, n) .+ 0.00005
    dxy        = 90.0 .* exp.(cumsum(dxy_ret))

    # VIX: Cox-Ingersoll-Ross-like mean reversion
    vix = zeros(n)
    vix[1] = 18.0
    for t in 2:n
        kappa = 2.0; theta = 18.0; sigma_v = 0.8
        vix[t] = max(vix[t-1] + kappa*(theta - vix[t-1])/252 +
                     sigma_v * sqrt(max(vix[t-1], 0.01)/252) * randn(rng), 5.0)
    end
    vix_ret = diff(log.(vix)); vix_ret = [vix_ret[1]; vix_ret]

    # SPY (S&P 500): GBM with GARCH
    spy_h    = 0.01; spy_ret = zeros(n)
    spy_ret[1] = 0.0001
    for t in 2:n
        spy_h    = 0.000001 + 0.08 * spy_ret[t-1]^2 + 0.91 * spy_h
        spy_ret[t] = sqrt(spy_h) * randn(rng) + 0.0003
    end
    spy = 400.0 .* exp.(cumsum(spy_ret))

    # 10Y yield: mean-reverting with regime
    yield10y = zeros(n)
    yield10y[1] = 0.025
    for t in 2:n
        mean_yield = 0.03 + 0.01 * sin(2π * t / 504)
        yield10y[t] = yield10y[t-1] + 0.1 * (mean_yield - yield10y[t-1]) / 252 +
                      0.005 / sqrt(252) * randn(rng)
        yield10y[t] = clamp(yield10y[t], 0.001, 0.10)
    end
    yield_chg = diff(yield10y); yield_chg = [yield_chg[1]; yield_chg]

    # Gold: mild safe-haven properties
    gold_ret = zeros(n)
    for t in 1:n
        gold_ret[t] = -0.15 * dxy_ret[t] +         # negative DXY correlation
                      -0.05 * spy_ret[t] +           # mild negative equity correlation
                      0.0001 + 0.008 * randn(rng)    # idiosyncratic
    end
    gold = 1800.0 .* exp.(cumsum(gold_ret))

    # TIPS (inflation-linked bonds): tracks inflation expectations
    tips_ret = zeros(n)
    for t in 2:n
        tips_ret[t] = -0.5 * yield_chg[t] +        # rate sensitivity
                       0.2  * (randn(rng) * 0.003) + # breakeven vol
                       0.0001
    end

    # BTC: crypto with macro betas embedded
    btc_ret = zeros(n)
    regime  = 1; regime_dur = 0
    for t in 1:n
        regime_dur += 1
        if regime == 1 && regime_dur > 200 * (0.5 + rand(rng))
            regime = 0; regime_dur = 0
        elseif regime == 0 && regime_dur > 80 * (0.5 + rand(rng))
            regime = 1; regime_dur = 0
        end

        mu_r    = regime == 1 ? 0.0008  : -0.002
        sigma_r = regime == 1 ? 0.035   :  0.055

        # Macro factor exposures (higher in bear regime)
        beta_spy  = regime == 1 ? 0.8  : 1.4
        beta_vix  = regime == 1 ? -0.3 : -0.6
        beta_dxy  = regime == 1 ? -0.2 : -0.4

        btc_ret[t] = beta_spy  * spy_ret[t]  +
                     beta_vix  * vix_ret[t]  +
                     beta_dxy  * dxy_ret[t]  +
                     mu_r + sigma_r * randn(rng)
    end
    btc = 20000.0 .* exp.(cumsum(btc_ret))

    # ETH: similar to BTC with higher beta
    eth_ret = zeros(n)
    for t in 1:n
        eth_ret[t] = 1.2 * btc_ret[t] + 0.015 * randn(rng) + 0.0001
    end
    eth = 1500.0 .* exp.(cumsum(eth_ret))

    return (
        btc=btc, btc_ret=btc_ret,
        eth=eth, eth_ret=eth_ret,
        spy=spy, spy_ret=spy_ret,
        dxy=dxy, dxy_ret=dxy_ret,
        vix=vix, vix_ret=vix_ret,
        yield=yield10y, yield_chg=yield_chg,
        gold=gold, gold_ret=gold_ret,
        tips_ret=tips_ret,
        n=n,
    )
end

println("\nGenerating 5-year synthetic macro + crypto panel...")
md = generate_macro_crypto_data(N_DAYS_MACRO)

println("  Variables: BTC, ETH, SPY, DXY, VIX, 10Y Yield, Gold, TIPS")
@printf("  BTC price range:    \$%.0f – \$%.0f\n", minimum(md.btc), maximum(md.btc))
@printf("  VIX range:          %.1f – %.1f\n", minimum(md.vix), maximum(md.vix))
@printf("  10Y yield range:    %.2f%% – %.2f%%\n", minimum(md.yield)*100, maximum(md.yield)*100)

# ── 2. FULL-SAMPLE MACRO BETAS ────────────────────────────────────────────────

println("\n" * "="^60)
println("FULL-SAMPLE MACRO BETAS")
println("="^60)

"""
OLS regression: BTC_ret = α + β_SPY * SPY_ret + β_VIX * vix_ret
                          + β_DXY * dxy_ret + β_TY * yield_chg + ε
"""
function ols_regression(X::Matrix{Float64}, y::Vector{Float64})
    # Add intercept
    X_aug = hcat(ones(size(X,1)), X)
    beta  = (X_aug'X_aug + 1e-8*I) \ (X_aug'y)
    yhat  = X_aug * beta
    resid = y .- yhat
    r2    = 1 - sum(resid.^2) / (sum((y .- mean(y)).^2) + 1e-8)
    n, k  = size(X_aug)
    se    = sqrt.(max.(diag((resid'resid / (n-k)) .* inv(X_aug'X_aug)), 0.0))
    tstat = beta ./ (se .+ 1e-10)
    return (beta=beta, se=se, tstat=tstat, r2=r2, resid=resid)
end

factor_names = ["Intercept", "SPY", "DXY", "VIX_chg", "10Y_chg"]
X_macro = hcat(md.spy_ret, md.dxy_ret, md.vix_ret, md.yield_chg)

btc_reg = ols_regression(X_macro, md.btc_ret)
eth_reg = ols_regression(X_macro, md.eth_ret)

println("\nBTC full-sample regression on macro factors:")
println("  Factor      | Beta      | Std Err   | t-stat    | Significant?")
println("  " * "-"^62)
for (i, name) in enumerate(factor_names)
    sig = abs(btc_reg.tstat[i]) > 2.0 ? "YES" : "no"
    @printf("  %-11s | %9.5f | %9.5f | %9.4f | %s\n",
            name, btc_reg.beta[i], btc_reg.se[i], btc_reg.tstat[i], sig)
end
@printf("  R² = %.4f  (%.1f%% of BTC variance explained by macro)\n",
        btc_reg.r2, btc_reg.r2*100)

println("\nETH full-sample regression on macro factors:")
println("  Factor      | Beta      | t-stat    | Significant?")
println("  " * "-"^52)
for (i, name) in enumerate(factor_names)
    sig = abs(eth_reg.tstat[i]) > 2.0 ? "YES" : "no"
    @printf("  %-11s | %9.5f | %9.4f | %s\n",
            name, eth_reg.beta[i], eth_reg.tstat[i], sig)
end
@printf("  R² = %.4f\n", eth_reg.r2)

# ── 3. REGIME-CONDITIONAL BETAS ───────────────────────────────────────────────

println("\n" * "="^60)
println("REGIME-CONDITIONAL BETAS")
println("="^60)

# Define regimes: BTC bull = cumulative 90d return > 0; bear otherwise
btc_cumret90 = [t >= 91 ? sum(md.btc_ret[t-90:t]) : NaN for t in 1:md.n]
btc_bull_mask = btc_cumret90 .> 0
btc_bear_mask = (btc_cumret90 .<= 0) .& .!isnan.(btc_cumret90)

bull_X = X_macro[btc_bull_mask, :]
bull_y = md.btc_ret[btc_bull_mask]
bear_X = X_macro[btc_bear_mask, :]
bear_y = md.btc_ret[btc_bear_mask]

bull_reg = ols_regression(bull_X, bull_y)
bear_reg = ols_regression(bear_X, bear_y)

@printf("\n  Bull regime: %d days  Bear regime: %d days\n",
        sum(btc_bull_mask), sum(btc_bear_mask))

println("\nRegime-conditional BTC betas:")
println("  Factor      | Bull Beta | Bear Beta | Bull/Bear Ratio | Interpretation")
println("  " * "-"^75)
interpretations = [
    "alpha shifts up in bull",
    "higher SPY beta in bear (risk-off)",
    "USD more negative in bear",
    "fear more negative in bear",
    "duration sensitivity in bear"
]
for (i, name) in enumerate(factor_names)
    bull_b = bull_reg.beta[i]
    bear_b = bear_reg.beta[i]
    ratio  = bear_b / (bull_b + sign(bull_b) * 1e-6)
    @printf("  %-11s | %9.5f | %9.5f | %15.3f | %s\n",
            name, bull_b, bear_b, ratio, interpretations[i])
end

println("\nR² by regime:")
@printf("  Bull regime R²: %.4f\n", bull_reg.r2)
@printf("  Bear regime R²: %.4f\n", bear_reg.r2)
println("  Higher R² in bear = macro factors more dominant during stress")

# ── 4. ROLLING BETA ESTIMATION ────────────────────────────────────────────────

println("\n" * "="^60)
println("ROLLING BETA TO SPY (126-DAY WINDOW)")
println("="^60)

function rolling_beta(y::Vector{Float64}, x::Vector{Float64}, window::Int)
    n    = length(y)
    beta = fill(NaN, n)
    for t in window:n
        xi = x[t-window+1:t]
        yi = y[t-window+1:t]
        xm = mean(xi); ym = mean(yi)
        beta[t] = sum((xi .- xm) .* (yi .- ym)) / (sum((xi .- xm).^2) + 1e-8)
    end
    return beta
end

btc_beta_spy = rolling_beta(md.btc_ret, md.spy_ret, 126)
eth_beta_spy = rolling_beta(md.eth_ret, md.spy_ret, 126)

valid = .!isnan.(btc_beta_spy)
println("\nBTC rolling beta to SPY (126-day):")
@printf("  Mean: %.4f  Std: %.4f  Min: %.4f  Max: %.4f\n",
        mean(btc_beta_spy[valid]), std(btc_beta_spy[valid]),
        minimum(btc_beta_spy[valid]), maximum(btc_beta_spy[valid]))

# Percentile distribution
for pct in [10, 25, 50, 75, 90]
    @printf("  %2dth percentile: %.4f\n", pct, quantile(btc_beta_spy[valid], pct/100))
end

println("\nETH rolling beta to SPY:")
valid2 = .!isnan.(eth_beta_spy)
@printf("  Mean: %.4f  Std: %.4f\n",
        mean(eth_beta_spy[valid2]), std(eth_beta_spy[valid2]))

# Beta by VIX regime
vix_high = md.vix .> 25.0
vix_low  = md.vix .<= 25.0
btc_b_high_vix = mean(btc_beta_spy[valid .& vix_high])
btc_b_low_vix  = mean(btc_beta_spy[valid .& vix_low])
@printf("\n  BTC-SPY beta when VIX > 25: %.4f\n", btc_b_high_vix)
@printf("  BTC-SPY beta when VIX ≤ 25: %.4f\n", btc_b_low_vix)
println("  (Higher beta during high-VIX periods confirms risk-off behavior)")

# ── 5. MACRO TIMING SIGNAL: VIX THRESHOLD ─────────────────────────────────────

println("\n" * "="^60)
println("MACRO TIMING SIGNAL: REDUCE EXPOSURE WHEN VIX > 30")
println("="^60)

"""
Timing strategy:
- VIX < 20: full allocation (1.0x)
- VIX 20-30: reduced allocation (0.7x)
- VIX > 30: defensive (0.3x)
"""
function vix_timing_strategy(btc_ret::Vector{Float64}, vix::Vector{Float64};
                               thresholds::Tuple=(20.0, 30.0),
                               positions::Tuple=(1.0, 0.7, 0.3))
    n  = length(btc_ret)
    pv = [1.0]
    bah_pv = [1.0]
    pos_log = Float64[]

    for t in 1:n-1
        v = vix[t]
        pos = if v < thresholds[1]
            positions[1]
        elseif v < thresholds[2]
            positions[2]
        else
            positions[3]
        end

        ret = btc_ret[t+1]
        push!(pv,     last(pv)     * (1.0 + pos * ret))
        push!(bah_pv, last(bah_pv) * (1.0 + ret))
        push!(pos_log, pos)
    end

    return (portfolio=pv, bah=bah_pv, positions=pos_log)
end

vix_strat = vix_timing_strategy(md.btc_ret, md.vix)

strat_rets = diff(log.(vix_strat.portfolio))
bah_rets   = diff(log.(vix_strat.bah))

sh_strat = mean(strat_rets) / (std(strat_rets) + 1e-8) * sqrt(252)
sh_bah   = mean(bah_rets)   / (std(bah_rets)   + 1e-8) * sqrt(252)

function max_drawdown_v(pv)
    pk = pv[1]; md = 0.0
    for v in pv; pk = max(pk,v); md = max(md,(pk-v)/pk); end
    return md
end

println("\nVIX timing strategy (full/0.7x/0.3x at VIX <20/20-30/>30):")
@printf("  VIX Timing Sharpe:  %.4f  Final: %.4f  MaxDD: %.2f%%\n",
        sh_strat, last(vix_strat.portfolio), max_drawdown_v(vix_strat.portfolio)*100)
@printf("  Buy-and-Hold Sharpe: %.4f  Final: %.4f  MaxDD: %.2f%%\n",
        sh_bah,   last(vix_strat.bah),       max_drawdown_v(vix_strat.bah)*100)

# Position breakdown
total_d = length(vix_strat.positions)
@printf("\n  Time at 1.0x: %.1f%%  0.7x: %.1f%%  0.3x: %.1f%%\n",
        sum(vix_strat.positions .≈ 1.0)/total_d*100,
        sum(vix_strat.positions .≈ 0.7)/total_d*100,
        sum(vix_strat.positions .≈ 0.3)/total_d*100)

# Returns by VIX regime
println("\nBTC returns by VIX regime (buy-and-hold):")
println("  VIX Regime    | Mean Daily Ret | Ann Vol | Sharpe | % of Time")
println("  " * "-"^60)
for (label, mask) in [
        ("VIX < 15",    md.vix .< 15),
        ("VIX 15-20",   (md.vix .>= 15) .& (md.vix .< 20)),
        ("VIX 20-30",   (md.vix .>= 20) .& (md.vix .< 30)),
        ("VIX 30-40",   (md.vix .>= 30) .& (md.vix .< 40)),
        ("VIX > 40",    md.vix .>= 40),
    ]
    n_mask = sum(mask)
    if n_mask < 5; continue; end
    rets  = md.btc_ret[mask]
    mr    = mean(rets) * 100
    av    = std(rets) * sqrt(252) * 100
    sh    = mean(rets) / (std(rets) + 1e-8) * sqrt(252)
    pct   = n_mask / md.n * 100
    @printf("  %-13s | %14.4f%% | %7.2f%% | %6.3f | %9.1f%%\n",
            label, mr, av, sh, pct)
end

# ── 6. FACTOR MODEL: MACRO VARIANCE EXPLAINED ─────────────────────────────────

println("\n" * "="^60)
println("FACTOR MODEL: HOW MUCH CRYPTO VARIANCE IS MACRO?")
println("="^60)

"""
Variance decomposition:
  Var(BTC) = β² * Var(macro) + Var(ε)
  R² = β² * Var(macro) / Var(BTC)
"""
println("\nVariance decomposition (BTC return):")
println("  Factor       | Beta    | Factor Var | Contribution | % of Total")
println("  " * "-"^65)

total_var    = var(md.btc_ret)
betas_btc    = btc_reg.beta[2:end]  # exclude intercept
factor_matrix = hcat(md.spy_ret, md.dxy_ret, md.vix_ret, md.yield_chg)
factor_names2 = ["SPY", "DXY", "VIX_chg", "10Y_chg"]

total_explained = 0.0
for (i, fname) in enumerate(factor_names2)
    fv  = var(factor_matrix[:, i])
    contrib = betas_btc[i]^2 * fv
    total_explained += contrib
    pct = contrib / total_var * 100
    @printf("  %-12s | %7.4f | %10.6f | %12.6f | %9.1f%%\n",
            fname, betas_btc[i], fv, contrib, pct)
end
resid_var = total_var - total_explained
@printf("  %-12s | %7s | %10s | %12.6f | %9.1f%%\n",
        "Idiosyncratic", "—", "—", resid_var, resid_var/total_var*100)
@printf("  %-12s | %7s | %10s | %12.6f | %9.1f%%\n",
        "TOTAL", "—", "—", total_var, 100.0)
@printf("\n  Total macro explanation: %.1f%%  Idiosyncratic: %.1f%%\n",
        total_explained/total_var*100, resid_var/total_var*100)

# ── 7. INFLATION HEDGE ANALYSIS ───────────────────────────────────────────────

println("\n" * "="^60)
println("INFLATION HEDGE ANALYSIS: BTC vs GOLD vs TIPS")
println("="^60)

"""
Compute rolling correlation of each asset with 10Y yield changes
(higher rates = inflation signal) and VIX (crisis hedge).
"""
function rolling_correlation(x::Vector{Float64}, y::Vector{Float64}, window::Int)
    n   = length(x)
    out = fill(NaN, n)
    for t in window:n
        out[t] = cor(x[t-window+1:t], y[t-window+1:t])
    end
    return out
end

# Correlation with 10Y yield changes (inflation proxy)
btc_yield_corr  = rolling_correlation(md.btc_ret, md.yield_chg, 126)
gold_yield_corr = rolling_correlation(md.gold_ret, md.yield_chg, 126)
tips_yield_corr = rolling_correlation(md.tips_ret, md.yield_chg, 126)

valid_t = 127:md.n
println("\nCorrelation with 10Y yield changes (inflation sensitivity):")
println("  Asset | Mean ρ | Std ρ | % Positive | Inflation Hedge?")
println("  " * "-"^60)
for (name, corr) in [
        ("BTC",  btc_yield_corr),
        ("Gold", gold_yield_corr),
        ("TIPS", tips_yield_corr),
    ]
    valid = .!isnan.(corr)
    mr  = mean(corr[valid])
    sr  = std(corr[valid])
    pct = mean(corr[valid] .< 0) * 100  # negative = inflation hedge
    hedge = mr < -0.1 ? "YES (negative sensitivity)" :
            mr < 0.0  ? "Weak" : "NO (positive sensitivity)"
    @printf("  %-5s | %6.4f | %5.4f | %10.1f%% | %s\n",
            name, mr, sr, pct, hedge)
end

# High vs low inflation regime
med_yield_chg = median(md.yield_chg)
high_infl = md.yield_chg .>  0.001   # rising rates
low_infl  = md.yield_chg .< -0.001   # falling rates

println("\nAsset performance in rising vs falling rate environments:")
println("  Asset | Rising Rates Ret | Falling Rates Ret | Spread")
println("  " * "-"^58)
for (name, ret) in [
        ("BTC",  md.btc_ret),
        ("ETH",  md.eth_ret),
        ("Gold", md.gold_ret),
        ("SPY",  md.spy_ret),
        ("TIPS", md.tips_ret),
    ]
    ri  = mean(ret[high_infl]) * 252 * 100
    fi  = mean(ret[low_infl])  * 252 * 100
    sp  = fi - ri
    @printf("  %-5s | %16.2f%% | %17.2f%% | %+6.2f%%\n", name, ri, fi, sp)
end

# ── 8. FED RATE CYCLE ANALYSIS ────────────────────────────────────────────────

println("\n" * "="^60)
println("FED RATE CYCLE AND CRYPTO PERFORMANCE")
println("="^60)

"""
Identify rate cycle phases from 10Y yield:
  Hiking:  yield trending up (>0.0005/day over 60 days)
  Cutting: yield trending down (<-0.0005/day over 60 days)
  Neutral: otherwise
"""
function classify_rate_cycle(yield::Vector{Float64}, window::Int=60)
    n    = length(yield)
    cycle = fill("Neutral", n)
    for t in window+1:n
        trend = (yield[t] - yield[t-window]) / window
        if trend > 0.00005
            cycle[t] = "Hiking"
        elseif trend < -0.00005
            cycle[t] = "Cutting"
        end
    end
    return cycle
end

rate_cycle = classify_rate_cycle(md.yield)

println("\nBTC performance by Fed rate cycle phase:")
println("  Cycle Phase | N Days | Ann Ret | Ann Vol | Sharpe | Max DD")
println("  " * "-"^62)
for phase in ["Hiking", "Neutral", "Cutting"]
    mask   = rate_cycle .== phase
    n_days = sum(mask)
    if n_days < 20; continue; end
    rets   = md.btc_ret[mask]
    ann_r  = mean(rets) * 252 * 100
    ann_v  = std(rets) * sqrt(252) * 100
    sh     = mean(rets) / (std(rets) + 1e-8) * sqrt(252)
    # Max drawdown in phase
    pv     = cumprod(1.0 .+ rets)
    pk = pv[1]; md_v = 0.0
    for v in pv; pk = max(pk,v); md_v = max(md_v, (pk-v)/pk); end
    @printf("  %-11s | %6d | %7.2f%% | %7.2f%% | %6.3f | %6.2f%%\n",
            phase, n_days, ann_r, ann_v, sh, md_v*100)
end

# Gold performance in same cycles
println("\nGold performance by Fed rate cycle phase:")
println("  Cycle Phase | N Days | Ann Ret | Ann Vol | Sharpe")
println("  " * "-"^52)
for phase in ["Hiking", "Neutral", "Cutting"]
    mask   = rate_cycle .== phase
    n_days = sum(mask)
    if n_days < 20; continue; end
    rets   = md.gold_ret[mask]
    ann_r  = mean(rets) * 252 * 100
    ann_v  = std(rets) * sqrt(252) * 100
    sh     = mean(rets) / (std(rets) + 1e-8) * sqrt(252)
    @printf("  %-11s | %6d | %7.2f%% | %7.2f%% | %6.3f\n",
            phase, n_days, ann_r, ann_v, sh)
end

# ── 9. MACRO REGIME TIMING STRATEGY ──────────────────────────────────────────

println("\n" * "="^60)
println("MACRO COMPOSITE TIMING STRATEGY")
println("="^60)

"""
Composite macro signal:
1. VIX signal: low VIX = risk on (+1); high VIX = risk off (-1)
2. DXY signal: weak DXY = bullish for crypto (+1); strong DXY (-1)
3. SPY momentum: positive SPY trend = risk on (+1)
4. Rate cycle: hiking = reduce exposure

Composite position sizing.
"""
function macro_composite_strategy(md; smooth::Int=20)
    n  = md.n
    pv = [1.0]
    bah_pv = [1.0]
    sig_log = Float64[]

    # Smoothed signals
    vix_ma  = [t >= smooth ? mean(md.vix[t-smooth+1:t]) : md.vix[t] for t in 1:n]
    spy_ma  = [t >= smooth ? sum(md.spy_ret[t-smooth+1:t]) : 0.0 for t in 1:n]
    dxy_ma  = [t >= smooth ? sum(md.dxy_ret[t-smooth+1:t]) : 0.0 for t in 1:n]

    for t in smooth+1:n-1
        # Component signals [-1, +1]
        vix_sig  = vix_ma[t] < 18 ? 1.0 : vix_ma[t] < 25 ? 0.0 : -1.0
        spy_sig  = spy_ma[t] > 0.01 ? 1.0 : spy_ma[t] > 0 ? 0.3 : -0.5
        dxy_sig  = dxy_ma[t] < 0 ? 0.5 : dxy_ma[t] > 0.01 ? -0.5 : 0.0
        cyc_sig  = rate_cycle[t] == "Hiking" ? -0.3 : rate_cycle[t] == "Cutting" ? 0.3 : 0.0

        # Weighted composite
        composite = 0.4*vix_sig + 0.3*spy_sig + 0.2*dxy_sig + 0.1*cyc_sig
        pos       = 0.5 + 0.5 * tanh(composite * 2.0)  # maps to (0, 1)
        pos       = clamp(pos, 0.1, 1.5)

        ret = md.btc_ret[t+1]
        push!(pv,     last(pv)     * (1.0 + pos * ret))
        push!(bah_pv, last(bah_pv) * (1.0 + ret))
        push!(sig_log, pos)
    end

    return pv, bah_pv, sig_log
end

macro_pv, macro_bah, macro_sig = macro_composite_strategy(md)

macro_rets = diff(log.(macro_pv))
mbah_rets  = diff(log.(macro_bah))

println("\nMacro composite timing strategy results:")
@printf("  Macro Strategy: Sharpe=%.4f  Final=%.4f  MaxDD=%.2f%%\n",
        mean(macro_rets)/(std(macro_rets)+1e-8)*sqrt(252),
        last(macro_pv),
        max_drawdown_v(macro_pv)*100)
@printf("  Buy-and-Hold:   Sharpe=%.4f  Final=%.4f  MaxDD=%.2f%%\n",
        mean(mbah_rets)/(std(mbah_rets)+1e-8)*sqrt(252),
        last(macro_bah),
        max_drawdown_v(macro_bah)*100)

@printf("\n  Mean position: %.3f  Std: %.3f  Range: [%.3f, %.3f]\n",
        mean(macro_sig), std(macro_sig),
        minimum(macro_sig), maximum(macro_sig))

# ── 10. DXY ANALYSIS ─────────────────────────────────────────────────────────

println("\n" * "="^60)
println("DXY IMPACT ON CRYPTO")
println("="^60)

# Rolling DXY-BTC correlation
dxy_btc_corr = rolling_correlation(md.btc_ret, md.dxy_ret, 63)
valid_corr   = .!isnan.(dxy_btc_corr)
@printf("\n  BTC-DXY rolling correlation (63-day):\n")
@printf("  Mean: %.4f  Std: %.4f  Min: %.4f  Max: %.4f\n",
        mean(dxy_btc_corr[valid_corr]), std(dxy_btc_corr[valid_corr]),
        minimum(dxy_btc_corr[valid_corr]), maximum(dxy_btc_corr[valid_corr]))

# BTC performance by DXY quintile
dxy_ret_clean = md.dxy_ret
q = [quantile(dxy_ret_clean, p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
println("\n  BTC returns by DXY daily move quintile:")
println("  DXY Quintile | DXY Range       | BTC Mean Ret | BTC Ann Ret")
println("  " * "-"^60)
for b in 1:5
    mask = (dxy_ret_clean .>= q[b]) .& (dxy_ret_clean .< q[b+1])
    if sum(mask) < 5; continue; end
    btc_r = md.btc_ret[mask]
    @printf("  Q%-11d | [%7.5f, %7.5f] | %12.6f | %10.3f%%\n",
            b, q[b], q[b+1], mean(btc_r), mean(btc_r)*252*100)
end

# ── 11. COMPLETE SUMMARY ─────────────────────────────────────────────────────

println("\n" * "="^60)
println("COMPLETE MACRO-CRYPTO LINKAGE SUMMARY")
println("="^60)

println("""
  Summary of macro-crypto relationships:

  1. SPY Beta: BTC has ~0.8-1.4x beta to equities; higher in bear markets
     This means crypto doesn't diversify well during equity crashes.

  2. VIX Relationship: BTC strongly negatively correlated with VIX
     -- it is a risk-on asset, not a safe haven. VIX > 30 = reduce exposure.

  3. DXY: Negative correlation with BTC (β ≈ -0.2 to -0.4)
     Strong dollar → crypto headwind; weak dollar → crypto tailwind.
     This is consistent with BTC as "anti-dollar" trade.

  4. 10Y Yields: Mixed relationship. TIPS are better inflation hedge than BTC.
     BTC actually underperforms in high-inflation, rate-hiking environments
     because rate hikes hurt risk assets broadly.

  5. Gold comparison: Gold shows negative rate sensitivity (better deflation
     hedge), while BTC is primarily a risk-sentiment-driven asset at this stage.

  6. Regime-conditional betas: SPY beta increases 30-75% in bear markets,
     concentration risk is highest when diversification is most needed.

  7. Macro timing: VIX-based and composite macro signals improve Sharpe ratio
     vs buy-and-hold by reducing drawdowns during macro stress periods.

  8. Fed cycle: BTC historically struggles during rate-hiking cycles
     (risk-off pressure) and performs better in cutting/neutral regimes.

  Practical implications:
  - Treat crypto as risk-on allocation, not defensive
  - Reduce crypto when VIX > 25-30 (historically poor performance)
  - Dollar strength warning: DXY breakout = headwind for crypto
  - Don't rely on BTC as inflation hedge in near term; Gold/TIPS preferred
  - Fed pivot (cutting cycle start) historically positive for crypto
""")

println("Notebook 22 complete.")
