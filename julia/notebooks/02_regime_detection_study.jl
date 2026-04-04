## Notebook 02: Regime Detection Study
## Compare HMM (Bayesian Markov Switching) vs BH rule-based regime detection

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SRFMResearch
using SRFMResearch.BHPhysics
using SRFMResearch.Bayesian
using SRFMResearch.SRFMStats
using DataFrames, Statistics, Dates, Plots, Random

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate regime-switching synthetic data
# ─────────────────────────────────────────────────────────────────────────────

function generate_regime_prices(n::Int=2000; seed::Int=42)
    rng = MersenneTwister(seed)

    # True regime: bull / bear alternating
    regime_true = zeros(Int, n)
    # Bull: 200 bars, Bear: 100 bars, cycle
    state = 1  # 1=bull, 2=bear
    state_dur = 0
    bull_dur  = 200
    bear_dur  = 100

    prices  = zeros(n)
    returns = zeros(n)
    prices[1] = 100.0

    for i in 2:n
        state_dur += 1
        if state == 1 && state_dur > bull_dur
            state     = 2; state_dur = 1
        elseif state == 2 && state_dur > bear_dur
            state     = 1; state_dur = 1
        end

        regime_true[i] = state
        mu_r    = state == 1 ? 0.0003 : -0.0003
        sigma_r = state == 1 ? 0.010  : 0.018    # higher vol in bear
        returns[i] = mu_r + sigma_r * randn(rng)
        prices[i]  = prices[i-1] * exp(returns[i])
    end

    return prices, returns, regime_true
end

println("=== Regime Detection Study ===\n")

prices, returns, regime_true = generate_regime_prices(2000)

# ─────────────────────────────────────────────────────────────────────────────
# 2. BH-based regime detection
# ─────────────────────────────────────────────────────────────────────────────

println("--- BH Rule-Based Regime Detection ---")

config = BHConfig(cf=0.003, bh_form=0.15, bh_collapse=0.06, bh_decay=0.97, ctl_req=3)
ms     = mass_series(prices, config)
bh_reg = bh_regime(ms.masses, ms.active, ms.bh_dir)

# Map BH regime to 2-state
bh_state = [r in ("BH_BULL", "BH_NEUTRAL") ? 1 : 2 for r in bh_reg]

# Compute accuracy vs ground truth
n = length(prices)
accuracy_bh   = mean(bh_state .== regime_true)
accuracy_inv  = mean(bh_state .!= regime_true)   # sometimes inverted
accuracy_bh_f = max(accuracy_bh, accuracy_inv)

println("BH regime accuracy: $(round(accuracy_bh_f * 100, digits=1))%")
println("BH active fraction: $(round(mean(ms.active) * 100, digits=1))%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Bayesian Markov Switching
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Bayesian Markov Switching HMM ---")

# Sample a subset for speed
sub_returns = returns[1:500]

model  = markov_switching(sub_returns, 2)
chains = run_mcmc(model, nothing; n_samples=500, n_chains=2)

# State probabilities
probs = regime_posterior(chains, sub_returns, 2)

# Assign state = argmax of probability
hmm_states = [argmax(probs[i, :]) for i in 1:length(sub_returns)]
true_sub   = regime_true[1:500]

# Accuracy (try both assignments)
acc1 = mean(hmm_states .== true_sub)
acc2 = mean(hmm_states .!= true_sub)
accuracy_hmm = max(acc1, acc2)
println("HMM accuracy (first 500 bars): $(round(accuracy_hmm * 100, digits=1))%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Momentum filter (simple rolling mean crossover)
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Rolling Momentum Baseline ---")

window_fast = 20; window_slow = 60
ma_fast = [i >= window_fast ? mean(prices[i-window_fast+1:i]) : NaN for i in 1:n]
ma_slow = [i >= window_slow ? mean(prices[i-window_slow+1:i]) : NaN for i in 1:n]

mom_state = [!isnan(ma_fast[i]) && !isnan(ma_slow[i]) ?
             (ma_fast[i] > ma_slow[i] ? 1 : 2) : 1 for i in 1:n]

acc_mom = mean(mom_state .== regime_true)
println("Momentum accuracy: $(round(max(acc_mom,1-acc_mom)*100, digits=1))%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Strategy comparison by regime quality
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Strategy Performance by Regime Method ---")

function backtest_with_signal(prices, states; commission=0.0004)
    n = length(prices)
    eq = fill(1.0, n)
    pos = 0
    for i in 2:n
        sig = states[i] == 1 ? 1 : -1
        if pos != sig
            eq[i] = eq[i-1] * (1 - commission)
            pos = sig
        else
            bar_ret = log(prices[i] / prices[i-1]) * pos
            eq[i] = eq[i-1] * exp(bar_ret)
        end
    end
    return eq
end

eq_bh  = backtest_with_signal(prices, bh_state)
eq_mom = backtest_with_signal(prices, mom_state)
eq_true = backtest_with_signal(prices, regime_true)   # oracle

rets_bh   = diff(log.(eq_bh))
rets_mom  = diff(log.(eq_mom))
rets_true = diff(log.(eq_true))

println("Oracle   Sharpe: $(round(sharpe_ratio(rets_true), digits=3))")
println("BH       Sharpe: $(round(sharpe_ratio(rets_bh), digits=3))")
println("Momentum Sharpe: $(round(sharpe_ratio(rets_mom), digits=3))")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Transition matrix analysis
# ─────────────────────────────────────────────────────────────────────────────

println("\n--- Regime Transition Analysis ---")

function transition_matrix(states, K=2)
    T = zeros(K, K)
    for i in 2:length(states)
        T[states[i-1], states[i]] += 1
    end
    for k in 1:K
        s = sum(T[k, :])
        s > 0 && (T[k, :] ./= s)
    end
    return T
end

T_true = transition_matrix(regime_true)
T_bh   = transition_matrix(bh_state)
T_mom  = transition_matrix(mom_state)

println("True transition matrix:")
display(T_true)
println("BH transition matrix:")
display(T_bh)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────────────────────────────────────

# Equity curves comparison
p1 = plot(
    title="Regime Detection Strategy Comparison",
    xlabel="Bar", ylabel="Equity",
    size=(1200, 500), legend=:topleft
)
plot!(p1, eq_true; label="Oracle", linewidth=2, color=:gold)
plot!(p1, eq_bh;   label="BH Regime", linewidth=2, color=:steelblue)
plot!(p1, eq_mom;  label="Momentum", linewidth=1.5, color=:red, linestyle=:dash)

savefig(p1, joinpath(@__DIR__, "02_regime_comparison.png"))

# BH mass series with true regime overlay
p2 = plot_bh_mass_series(prices[1:500], ms.masses[1:500], ms.active[1:500],
                          ms.bh_dir[1:500];
                          cf=config.cf, bh_form=config.bh_form,
                          title="BH Mass vs True Regime")

savefig(p2, joinpath(@__DIR__, "02_bh_vs_regime.png"))

# Rolling regime fractions
p3 = plot_regime_transitions(bh_reg[1:1000])
savefig(p3, joinpath(@__DIR__, "02_regime_transitions.png"))

println("\n02_regime_detection_study.jl complete.")
