"""
AlphaResearch — Alpha research framework for quantitative strategy development.

Covers:
  - IC (Information Coefficient) computation and time series
  - ICIR (IC Information Ratio) and decay analysis
  - Factor turnover and alpha decay modelling
  - IC-weighted alpha combination (ensemble)
  - Backtest of combined alpha vs individual signals
  - Risk-adjusted alpha: Sharpe contribution per signal
  - Alpha recycling: detect correlated signals, combine optimally
  - Walk-forward alpha stability tests
"""
module AlphaResearch

using LinearAlgebra
using Statistics
using Random

export ic, icir, rolling_ic, rolling_icir
export alpha_decay_fit, alpha_halflife, optimal_horizon
export ic_weighted_ensemble, ensemble_backtest
export alpha_sharpe_contribution, risk_adjusted_alpha
export alpha_correlation_matrix, detect_redundant_alphas
export recycle_alpha_combination
export walk_forward_alpha_test, alpha_stability_report
export signal_turnover, turnover_cost_breakeven

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: IC Computation
# ─────────────────────────────────────────────────────────────────────────────

"""
    ic(signal, returns; method) -> Float64

Compute Information Coefficient between signal and forward returns.
method = :pearson (linear) or :spearman (rank, default).
IC is the correlation between the signal at time t and returns at t+1.
"""
function ic(signal::Vector{Float64}, returns::Vector{Float64};
             method::Symbol=:spearman)::Float64
    n = min(length(signal), length(returns))
    n < 5 && return 0.0

    s = signal[1:n]
    r = returns[1:n]

    if method == :spearman
        rs = rank_transform(s)
        rr = rank_transform(r)
        return cor(rs, rr)
    else
        return cor(s, r)
    end
end

"""
    rank_transform(x) -> Vector{Float64}

Convert x to ranks (1 = lowest). Handles ties by averaging.
"""
function rank_transform(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    ranks = zeros(n)
    sorted_idx = sortperm(x)
    # Assign ranks with tie averaging
    i = 1
    while i <= n
        # Find run of equal values
        j = i
        while j < n && x[sorted_idx[j+1]] == x[sorted_idx[j]]
            j += 1
        end
        avg_rank = (i + j) / 2
        for k in i:j
            ranks[sorted_idx[k]] = avg_rank
        end
        i = j + 1
    end
    return ranks
end

"""
    rolling_ic(signal, returns; window, horizon, method) -> Vector{Float64}

Rolling window IC computation. At each time t, computes IC over the
preceding `window` observations between signal[t-window:t-1] and
returns[t-window+horizon:t+horizon-1].
"""
function rolling_ic(signal::Vector{Float64}, returns::Vector{Float64};
                     window::Int=60, horizon::Int=1,
                     method::Symbol=:spearman)::Vector{Float64}
    n   = length(signal)
    res = fill(NaN, n)

    for t in (window + horizon):n
        s_w = signal[(t - window + 1):(t - horizon)]
        r_w = returns[(t - window + 2):(t - horizon + 1)]
        length(s_w) != length(r_w) && continue
        res[t] = ic(s_w, r_w; method=method)
    end
    return res
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: ICIR
# ─────────────────────────────────────────────────────────────────────────────

"""
    icir(ic_series; window) -> Float64

IC Information Ratio: mean(IC) / std(IC) over the series.
A rolling version uses a `window`-length moving average.
ICIR > 0.5 is considered a strong signal.
"""
function icir(ic_series::Vector{Float64}; window::Int=0)::Float64
    valid = filter(isfinite, ic_series)
    isempty(valid) && return 0.0
    m = mean(valid)
    s = std(valid)
    s < 1e-10 && return 0.0
    return m / s
end

"""
    rolling_icir(ic_series; window) -> Vector{Float64}

Rolling ICIR: ICIR computed over each `window`-length window of IC values.
"""
function rolling_icir(ic_series::Vector{Float64}; window::Int=60)::Vector{Float64}
    n = length(ic_series)
    res = fill(NaN, n)
    for t in window:n
        w = ic_series[(t-window+1):t]
        valid_w = filter(isfinite, w)
        length(valid_w) < 5 && continue
        res[t] = icir(valid_w)
    end
    return res
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Alpha Decay Modelling
# ─────────────────────────────────────────────────────────────────────────────

"""
    AlphaDecayModel

Fitted exponential decay model for IC vs forecast horizon.
IC(h) = IC0 * exp(-h / tau) where tau is the decay time constant.
"""
struct AlphaDecayModel
    IC0::Float64          # initial IC (at h=1)
    tau::Float64          # decay time constant (bars)
    half_life::Float64    # half-life = tau * log(2)
    r2::Float64           # model fit quality
    decay_rate::Float64   # 1/tau
end

"""
    alpha_decay_fit(horizons, ic_vals) -> AlphaDecayModel

Fit exponential decay IC(h) = IC0 * exp(-h/tau) to IC values at
multiple horizons. Uses log-linear regression on positive IC values.
"""
function alpha_decay_fit(horizons::Vector{Int}, ic_vals::Vector{Float64})::AlphaDecayModel
    valid = ic_vals .> 1e-5
    sum(valid) < 3 && return AlphaDecayModel(ic_vals[1], 1.0, log(2), 0.0, 1.0)

    hv = Float64.(horizons[valid])
    iv = ic_vals[valid]
    log_ic = log.(iv)

    n     = length(hv)
    h_bar = mean(hv)
    i_bar = mean(log_ic)
    b = sum((hv .- h_bar) .* (log_ic .- i_bar)) / max(sum((hv .- h_bar).^2), 1e-10)
    a = i_bar - b * h_bar

    IC0 = exp(a)
    tau = b < -1e-10 ? -1/b : Inf

    preds  = a .+ b .* hv
    ss_res = sum((log_ic .- preds).^2)
    ss_tot = sum((log_ic .- i_bar).^2)
    r2     = ss_tot > 1e-10 ? 1 - ss_res/ss_tot : 0.0

    return AlphaDecayModel(IC0, tau, tau * log(2), clamp(r2, 0.0, 1.0), max(-b, 0.0))
end

"""
    alpha_halflife(signal, returns; max_horizon, min_ic) -> Float64

Estimate the alpha half-life directly from signal-return IC series.
Finds the horizon h* where IC(h*) ≈ IC(1) / 2.
"""
function alpha_halflife(signal::Vector{Float64}, returns::Vector{Float64};
                          max_horizon::Int=50, min_ic::Float64=0.001)::Float64
    n    = length(signal)
    ic1  = ic(signal[1:end-1], returns[2:end])
    ic1 < min_ic && return 1.0

    target = ic1 / 2
    for h in 2:max_horizon
        n > h + 1 || break
        ic_h = ic(signal[1:(n-h)], [sum(returns[(t+1):(t+h)]) for t in 1:(n-h)])
        ic_h <= target && return Float64(h)
    end
    return Float64(max_horizon)  # decay slower than measured range
end

"""
    optimal_horizon(decay_model; n_annual) -> NamedTuple

Find the optimal forecast horizon that maximises the FLAM IR.
IR(h) ∝ IC(h) * sqrt(breadth) where breadth = n_annual / h.
"""
function optimal_horizon(decay_model::AlphaDecayModel; n_annual::Int=252)::NamedTuple
    best_h  = 1
    best_ir = 0.0
    ir_vals = Float64[]

    for h in 1:100
        ic_h    = decay_model.IC0 * exp(-h / max(decay_model.tau, 1e-10))
        breadth = max(n_annual / h, 1)
        ir_h    = ic_h * sqrt(breadth)
        push!(ir_vals, ir_h)
        if ir_h > best_ir
            best_ir = ir_h
            best_h  = h
        end
    end
    return (optimal_h=best_h, ir_at_opt=best_ir, ir_series=ir_vals)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: IC-Weighted Ensemble
# ─────────────────────────────────────────────────────────────────────────────

"""
    ic_weighted_ensemble(signals, ic_vals) -> Vector{Float64}

Combine multiple signals using IC-proportional weights.
weight_i = |IC_i| / sum_j |IC_j|
Returns combined signal of same length as inputs.
"""
function ic_weighted_ensemble(signals::Vector{Vector{Float64}},
                                ic_vals::Vector{Float64})::Vector{Float64}
    @assert length(signals) == length(ic_vals) "signals and ic_vals must match"
    total = sum(abs.(ic_vals))
    total < 1e-10 && return mean(hcat(signals...), dims=2)[:]

    weights = abs.(ic_vals) ./ total
    n       = length(signals[1])
    combined = zeros(n)

    for (i, s) in enumerate(signals)
        length(s) == n || continue
        combined .+= weights[i] .* s
    end
    return combined
end

"""
    ensemble_backtest(signals, returns, names; horizon, n_train_frac, transaction_cost) -> NamedTuple

Backtest an IC-weighted ensemble vs individual signals.
Trains IC weights on the first `n_train_frac` of data, tests on the rest.
"""
function ensemble_backtest(signals::Vector{Vector{Float64}},
                             returns::Vector{Float64},
                             names::Vector{String};
                             horizon::Int=1,
                             n_train_frac::Float64=0.60,
                             transaction_cost::Float64=0.001)::NamedTuple
    n_sig  = length(signals)
    n_obs  = length(returns)
    n_train = round(Int, n_train_frac * n_obs)

    # Estimate IC weights on training data
    ic_train = Float64[]
    for s in signals
        ic_t = ic(s[1:(n_train-horizon)],
                   returns[(1+horizon):n_train])
        push!(ic_train, ic_t)
    end

    # Combined signal
    combined = ic_weighted_ensemble(signals, ic_train)

    # Evaluate on test data
    test_signals = [s[(n_train+1):n_obs] for s in signals]
    test_combined = combined[(n_train+1):n_obs]
    test_returns  = returns[(n_train+1):n_obs]
    n_test = length(test_returns)

    # Compute test performance for each signal
    result = Dict{String,NamedTuple}()
    for (i, name) in enumerate(names)
        ts = test_signals[i]
        length(ts) < 5 && continue
        ic_test = ic(ts[1:end-horizon], test_returns[(1+horizon):end])
        sr      = sharpe_from_signal(ts, test_returns; horizon=horizon,
                                      cost=transaction_cost)
        result[name] = (ic=ic_test, sharpe=sr)
    end

    # Ensemble
    ic_ens = ic(test_combined[1:end-horizon], test_returns[(1+horizon):end])
    sr_ens = sharpe_from_signal(test_combined, test_returns;
                                  horizon=horizon, cost=transaction_cost)
    result["ENSEMBLE"] = (ic=ic_ens, sharpe=sr_ens)

    return (results=result, ic_weights=ic_train, n_train=n_train, n_test=n_test)
end

"""
    sharpe_from_signal(signal, returns; horizon, cost, annualise) -> Float64

Compute annualised Sharpe from a signal by going long when signal > 0.
Applies simple proportional transaction cost on signal changes.
"""
function sharpe_from_signal(signal::Vector{Float64}, returns::Vector{Float64};
                              horizon::Int=1, cost::Float64=0.001,
                              annualise::Int=252)::Float64
    n = min(length(signal), length(returns))
    n < 10 && return 0.0

    sig = signal[1:(n-horizon)]
    ret = returns[(1+horizon):n]

    # Signal direction: +1 if positive, 0 otherwise
    positions = sig .> 0

    # Strategy returns with transaction costs
    strat_rets = Float64[]
    prev_pos   = false
    for i in 1:length(positions)
        tc = (positions[i] != prev_pos) ? cost : 0.0
        push!(strat_rets, Float64(positions[i]) * ret[i] - tc)
        prev_pos = positions[i]
    end

    length(strat_rets) < 5 && return 0.0
    m = mean(strat_rets)
    s = std(strat_rets)
    s < 1e-10 && return 0.0
    return m / s * sqrt(annualise)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Risk-Adjusted Alpha Attribution
# ─────────────────────────────────────────────────────────────────────────────

"""
    alpha_sharpe_contribution(signals, returns, names; horizon) -> Dict{String,Float64}

Compute the Sharpe ratio contribution of each signal in an ensemble.
Uses the Fong-Lin (2007) marginal contribution approach:
  contribution_i = weight_i * cov(r_i, r_portfolio) / var(r_portfolio)
where r_i is the alpha of signal i and r_portfolio is the combined alpha.
"""
function alpha_sharpe_contribution(signals::Vector{Vector{Float64}},
                                    returns::Vector{Float64},
                                    names::Vector{String};
                                    horizon::Int=1)::Dict{String,Float64}
    n_obs = length(returns)
    ic_vals = [ic(s[1:end-horizon], returns[(1+horizon):end]) for s in signals]
    weights  = abs.(ic_vals) ./ max(sum(abs.(ic_vals)), 1e-10)

    # Compute signal returns
    n_sig = length(signals)
    sig_rets = zeros(n_obs - horizon, n_sig)
    for (i, s) in enumerate(signals)
        pos = s[1:(n_obs-horizon)] .> 0
        sig_rets[:, i] = Float64.(pos) .* returns[(1+horizon):end]
    end

    # Portfolio return (weighted)
    port_ret = sig_rets * weights
    port_var = var(port_ret)
    port_var < 1e-12 && return Dict(names .=> weights)

    contrib = Dict{String,Float64}()
    for (i, name) in enumerate(names)
        cov_i = cov(sig_rets[:, i], port_ret)
        contrib[name] = weights[i] * cov_i / port_var
    end
    return contrib
end

"""
    risk_adjusted_alpha(signal, returns, benchmark_returns; horizon) -> NamedTuple

Compute risk-adjusted alpha metrics for a single signal.
Returns: alpha (CAPM), beta, t-stat, information ratio.
"""
function risk_adjusted_alpha(signal::Vector{Float64}, returns::Vector{Float64},
                               benchmark_returns::Vector{Float64};
                               horizon::Int=1)::NamedTuple
    n = min(length(signal), length(returns), length(benchmark_returns))
    n < 20 && return (alpha=0.0, beta=0.0, ir=0.0, t_alpha=0.0)

    pos      = signal[1:(n-horizon)] .> 0
    strat_r  = Float64.(pos) .* returns[(1+horizon):n]
    bench_r  = benchmark_returns[(1+horizon):n]

    n_use = length(strat_r)

    # CAPM OLS: strat_r = alpha + beta * bench_r + eps
    X = hcat(ones(n_use), bench_r)
    coeffs = (X' * X) \ (X' * strat_r)
    alpha_val = coeffs[1]
    beta_val  = coeffs[2]

    resid = strat_r .- X * coeffs
    s2 = var(resid)
    # Standard error of alpha
    invXX = inv(X' * X)
    se_alpha = sqrt(s2 * invXX[1, 1])
    t_alpha  = se_alpha > 1e-10 ? alpha_val / se_alpha : 0.0

    # Information ratio = alpha / tracking error
    tracking_err = std(strat_r .- bench_r)
    ir = tracking_err > 1e-10 ? mean(strat_r .- bench_r) / tracking_err * sqrt(252) : 0.0

    return (alpha=alpha_val, beta=beta_val, ir=ir, t_alpha=t_alpha,
            se_alpha=se_alpha, r2=1 - s2/max(var(strat_r), 1e-10))
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Alpha Correlation and Redundancy Detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    alpha_correlation_matrix(signals, returns, names; horizon) -> Matrix{Float64}

Compute pairwise IC correlations between signals (via their return series).
IC correlation = correlation of signal return streams.
"""
function alpha_correlation_matrix(signals::Vector{Vector{Float64}},
                                   returns::Vector{Float64},
                                   names::Vector{String};
                                   horizon::Int=1)::Matrix{Float64}
    n_sig = length(signals)
    n_obs = length(returns)

    sig_rets = [Float64.(s[1:(n_obs-horizon)] .> 0) .* returns[(1+horizon):end]
                for s in signals]

    R = zeros(n_sig, n_sig)
    for i in 1:n_sig
        for j in 1:n_sig
            R[i, j] = cor(sig_rets[i], sig_rets[j])
        end
    end
    return R
end

"""
    detect_redundant_alphas(signals, returns, names; corr_threshold, horizon) -> Vector{String}

Identify signals that are highly correlated with others (redundant).
Returns names of redundant signals (candidates for removal or combination).
A signal is "redundant" if its max pairwise correlation > corr_threshold
AND its IC is lower than the signal it correlates with.
"""
function detect_redundant_alphas(signals::Vector{Vector{Float64}},
                                   returns::Vector{Float64},
                                   names::Vector{String};
                                   corr_threshold::Float64=0.70,
                                   horizon::Int=1)::Vector{String}
    n_sig = length(signals)
    R     = alpha_correlation_matrix(signals, returns, names; horizon=horizon)
    ics   = [ic(s[1:end-horizon], returns[(1+horizon):end]) for s in signals]

    redundant = String[]
    for i in 1:n_sig
        for j in (i+1):n_sig
            abs(R[i, j]) > corr_threshold || continue
            # Remove the one with lower IC
            if ics[i] < ics[j]
                names[i] ∉ redundant && push!(redundant, names[i])
            else
                names[j] ∉ redundant && push!(redundant, names[j])
            end
        end
    end
    return redundant
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Alpha Recycling
# ─────────────────────────────────────────────────────────────────────────────

"""
    recycle_alpha_combination(primary_signal, secondary_signal, returns;
                               horizon_primary, horizon_secondary, n_periods) -> Vector{Float64}

Combine a primary (momentum) and secondary (OU/MR) signal to recycle
decayed alpha. The combination weight adapts as a function of forecast horizon:
  - At short horizons: weight towards momentum (primary)
  - At long horizons: weight towards mean reversion (secondary)
Uses IC estimates at each horizon to determine optimal weights.
"""
function recycle_alpha_combination(primary::Vector{Float64},
                                    secondary::Vector{Float64},
                                    returns::Vector{Float64};
                                    max_horizon::Int=30,
                                    n_train::Int=200)::NamedTuple
    n = min(length(primary), length(secondary), length(returns))

    # Estimate IC of each signal at multiple horizons
    ic_primary   = zeros(max_horizon)
    ic_secondary = zeros(max_horizon)

    for h in 1:max_horizon
        n_use = min(n_train, n - h)
        n_use < 10 && break
        ic_primary[h]   = ic(primary[1:n_use],
                               [sum(returns[(t+1):(t+h)]) for t in 1:n_use])
        ic_secondary[h] = ic(secondary[1:n_use],
                               [sum(returns[(t+1):(t+h)]) for t in 1:n_use])
    end

    # Optimal weights at each horizon
    weights_primary   = zeros(max_horizon)
    weights_secondary = zeros(max_horizon)
    combined_signals  = Vector{Vector{Float64}}(undef, max_horizon)

    for h in 1:max_horizon
        ip = abs(ic_primary[h])
        is = abs(ic_secondary[h])
        total = ip + is
        if total < 1e-10
            weights_primary[h]   = 0.5
            weights_secondary[h] = 0.5
        else
            weights_primary[h]   = ip / total
            weights_secondary[h] = is / total
        end
        combined_signals[h] = weights_primary[h] .* primary .+
                               weights_secondary[h] .* secondary
    end

    return (
        ic_primary   = ic_primary,
        ic_secondary = ic_secondary,
        weights_primary   = weights_primary,
        weights_secondary = weights_secondary,
        combined_at_h = (h) -> combined_signals[clamp(h, 1, max_horizon)],
        optimal_h = argmax(ic_primary .+ ic_secondary),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Walk-Forward Alpha Stability Test
# ─────────────────────────────────────────────────────────────────────────────

"""
    walk_forward_alpha_test(signal, returns; n_windows, horizon) -> NamedTuple

Test alpha stability by computing IC in each of n_windows sequential windows.
Returns per-window ICs, mean, std, t-statistic, p-value for H0: mean_IC=0.
"""
function walk_forward_alpha_test(signal::Vector{Float64}, returns::Vector{Float64};
                                   n_windows::Int=10, horizon::Int=1)::NamedTuple
    n = length(signal)
    window_size = div(n, n_windows)
    window_size < horizon + 5 && return _empty_wf_result()

    ic_vals = Float64[]
    for w in 0:(n_windows-1)
        s_idx = w * window_size + 1
        e_idx = min((w+1) * window_size, n - horizon)
        e_idx <= s_idx + 5 && continue

        s_w = signal[s_idx:e_idx]
        r_w = returns[(s_idx+horizon):(e_idx+horizon)]
        length(r_w) != length(s_w) && continue

        push!(ic_vals, ic(s_w, r_w))
    end

    isempty(ic_vals) && return _empty_wf_result()

    n_w     = length(ic_vals)
    mean_ic = mean(ic_vals)
    std_ic  = std(ic_vals)
    t_stat  = std_ic > 1e-10 ? mean_ic / (std_ic / sqrt(n_w)) : 0.0
    icir_v  = icir(ic_vals)

    # Approximate p-value (normal approximation for t-test)
    p_val = 2 * (1 - Φ(abs(t_stat)))

    pct_positive = mean(ic_vals .> 0) * 100
    stable = p_val < 0.05 && mean_ic > 0

    return (ic_vals=ic_vals, n_windows=n_w, mean_ic=mean_ic, std_ic=std_ic,
            icir=icir_v, t_stat=t_stat, p_val=p_val,
            pct_positive=pct_positive, stable=stable)
end

function _empty_wf_result()
    return (ic_vals=Float64[], n_windows=0, mean_ic=0.0, std_ic=0.0,
            icir=0.0, t_stat=0.0, p_val=1.0, pct_positive=0.0, stable=false)
end

"""
    Φ(x) -> Float64

Standard normal CDF.
"""
function Φ(x::Float64)::Float64
    t = 1 / (1 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 +
                t * (-1.453152027 + t * 1.061405429))))
    cdf = 1 - poly * exp(-x^2)
    return x >= 0 ? cdf : 1 - cdf
end

"""
    alpha_stability_report(signal, returns, name; n_windows, horizons) -> Nothing

Print a comprehensive alpha stability report for a single signal.
"""
function alpha_stability_report(signal::Vector{Float64}, returns::Vector{Float64},
                                  name::String;
                                  n_windows::Int=10,
                                  horizons::Vector{Int}=[1,4,8,12,24])::Nothing
    println("=== Alpha Stability Report: $name ===")

    # Walk-forward test
    wf = walk_forward_alpha_test(signal, returns; n_windows=n_windows)
    println("\nWalk-Forward IC Analysis:")
    println("  Mean IC:  $(round(wf.mean_ic, digits=5))")
    println("  Std IC:   $(round(wf.std_ic, digits=5))")
    println("  ICIR:     $(round(wf.icir, digits=4))")
    println("  t-stat:   $(round(wf.t_stat, digits=3))")
    println("  p-value:  $(round(wf.p_val, digits=5))")
    println("  Positive windows: $(round(wf.pct_positive, digits=1))%")
    println("  STABLE:   $(wf.stable ? "YES" : "NO")")

    # IC decay
    println("\nIC Decay Profile:")
    ic_h_vals = Float64[]
    for h in horizons
        n_use = length(signal) - h
        n_use < 10 && (push!(ic_h_vals, 0.0); continue)
        ih = ic(signal[1:n_use], [sum(returns[(t+1):(t+h)]) for t in 1:n_use])
        push!(ic_h_vals, ih)
        println("  IC(h=$h):  $(round(ih, digits=5))")
    end

    # Fit decay
    if !isempty(ic_h_vals) && any(ic_h_vals .> 0)
        dm = alpha_decay_fit(horizons[1:length(ic_h_vals)], ic_h_vals)
        println("\nFitted Decay Model:")
        println("  IC0 = $(round(dm.IC0, digits=5))")
        println("  τ   = $(round(dm.tau, digits=2)) bars")
        println("  t½  = $(round(dm.half_life, digits=2)) bars")
        println("  R²  = $(round(dm.r2, digits=4))")
    end

    # Turnover
    to = signal_turnover(signal)
    println("\nSignal Turnover: $(round(to, digits=4)) per bar")
    println("=== End Report ===\n")

    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Signal Turnover Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    signal_turnover(signal) -> Float64

Normalised signal turnover: mean(|s_t - s_{t-1}|) / mean(|s_t|).
High turnover = signal changes fast = high TC requirement.
"""
function signal_turnover(signal::Vector{Float64})::Float64
    n = length(signal)
    n < 2 && return 0.0
    abs_change = mean(abs.(diff(signal)))
    avg_level  = mean(abs.(signal))
    avg_level < 1e-10 && return Inf
    return abs_change / avg_level
end

"""
    turnover_cost_breakeven(decay_model, sigma_per_bar; cost_bps) -> Float64

Compute the minimum signal IC needed to break even against transaction costs.
IC_min = TC / (signal_vol * sqrt(breadth)) ≈ TC / sigma (rough estimate).
More precisely: alpha = IC * sigma, TC = cost * turnover,
breakeven when IC * sigma >= cost * turnover.
"""
function turnover_cost_breakeven(decay_model::AlphaDecayModel, sigma_per_bar::Float64;
                                   cost_bps::Float64=10.0)::NamedTuple
    cost = cost_bps / 10000
    # Turnover ≈ sqrt(2) * std(ds) where ds is signal increment
    # For AR(1) with phi = exp(-1/tau): std(ds) ≈ sqrt(2*(1-phi)) * 1
    phi = exp(-1 / max(decay_model.tau, 0.01))
    sig_ds = sqrt(2 * (1 - phi))  # normalised signal std

    ic_min = cost * sig_ds / max(sigma_per_bar, 1e-10)
    h_max  = -decay_model.tau * log(max(ic_min / max(decay_model.IC0, 1e-10), 1e-10))

    return (ic_min=ic_min, max_viable_horizon=max(h_max, 1.0),
            cost_bps=cost_bps, sigma=sigma_per_bar)
end

end  # module AlphaResearch
