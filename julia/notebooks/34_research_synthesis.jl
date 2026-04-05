## Notebook 34: Research Synthesis
## Compile findings from notebooks 01-33, priority ranking, implementation roadmap,
## risk budget review, signal inventory, 12-month research agenda
## Run as a Julia script or in Pluto.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics, Dates, Random, LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# 1. Compile Findings from Notebooks 01-33
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 70)
println("NOTEBOOK 34: RESEARCH SYNTHESIS — SRFM LAB FINDINGS")
println("=" ^ 70)
println("Synthesizing findings from notebooks 01-33")
println("Date: April 2026")
println()

"""Structured representation of a research finding."""
struct ResearchFinding
    notebook::Int
    topic::String
    finding::String
    actionable::Bool
    sharpe_impact::Float64   # estimated Sharpe improvement if implemented
    effort_days::Int         # implementation effort in days
    confidence::Float64      # confidence in the finding (0-1)
end

findings = [
    # Notebook 01-05: BH Physics, Regimes, Portfolio, Stochastic, Bayesian
    ResearchFinding(1, "BH Physics Mass Dynamics",
        "Mass concentration predicts regime transitions; high mass = trend continuation",
        true, 0.08, 10, 0.65),
    ResearchFinding(2, "Regime Detection (HMM/GARCH)",
        "3-regime HMM with GARCH vol achieves 75-80% regime accuracy; bull/bear/choppy",
        true, 0.15, 14, 0.80),
    ResearchFinding(3, "Portfolio Optimization (MVO)",
        "Robust MVO (Γ=1.5) outperforms standard MVO OOS by 15-20% Sharpe",
        true, 0.12, 7, 0.85),
    ResearchFinding(4, "Stochastic Process Calibration",
        "Heston model fits BTC vol smile better than Black-Scholes; NIG distribution for returns",
        true, 0.06, 20, 0.75),
    ResearchFinding(5, "Bayesian CF Estimation",
        "Bayesian characteristic function estimation provides better tail risk estimates",
        false, 0.04, 30, 0.60),

    # Notebooks 06-10
    ResearchFinding(6, "Factor Model",
        "Market (BTC) + momentum + vol factors explain 70% of cross-asset variance",
        true, 0.10, 7, 0.85),
    ResearchFinding(7, "Copula Dependence",
        "Clayton copula in tails: left tail dependence significantly higher than Gaussian assumption",
        true, 0.08, 14, 0.80),
    ResearchFinding(8, "Market Microstructure",
        "Order flow imbalance (OFI) has IC of 0.05-0.10 for 5-min returns; significant alpha",
        true, 0.20, 21, 0.75),
    ResearchFinding(9, "Jump Risk",
        "BTC jumps cluster; post-jump drift reversal signal has IC ~0.08",
        true, 0.07, 14, 0.70),
    ResearchFinding(10, "Extreme Value Theory",
        "GPD tail fit: 99.9% VaR underestimated by 40% using normal distribution",
        true, 0.05, 7, 0.90),

    # Notebooks 11-15
    ResearchFinding(11, "Interest Rates / Macro",
        "Fed rate surprises predict BTC direction next 5 days with IC ~0.12",
        true, 0.10, 10, 0.70),
    ResearchFinding(12, "Machine Learning (Random Forest/GBM)",
        "GBM ensemble achieves IC of 0.06-0.12 for 1-day BTC returns; feature importance shows vol+momentum",
        true, 0.18, 14, 0.80),
    ResearchFinding(13, "Macro Factors",
        "DXY, gold, VIX are significant risk-off predictors for crypto; combine in macro overlay",
        true, 0.08, 10, 0.75),
    ResearchFinding(14, "Market Regimes (deep)",
        "Regime-conditional position sizing improves Sharpe by 0.2-0.3 vs static sizing",
        true, 0.25, 14, 0.85),
    ResearchFinding(15, "Network Analysis",
        "Graph centrality of crypto-exchange network predicts contagion risk",
        false, 0.05, 21, 0.60),

    # Notebooks 16-22
    ResearchFinding(16, "Online Learning (adaptive)",
        "AdaGrad/RMSProp adaptation improves signal stability by 20% in regime transitions",
        true, 0.10, 14, 0.75),
    ResearchFinding(17, "Optimal Execution",
        "VWAP-based execution reduces impact by 15-25% vs naive market orders",
        true, 0.15, 7, 0.90),
    ResearchFinding(18, "Risk Management (CVaR)",
        "CVaR-based position limits outperform VaR-based limits in stress testing",
        true, 0.10, 5, 0.90),
    ResearchFinding(19, "Robust Statistics",
        "Huber M-estimator for covariance improves OOS portfolio vol estimation by 15%",
        true, 0.08, 5, 0.85),
    ResearchFinding(20, "Sentiment Analysis",
        "Crypto Twitter sentiment signal IC = 0.03-0.05; highest in low-vol environments",
        true, 0.06, 21, 0.65),
    ResearchFinding(21, "Signal Processing (wavelets)",
        "Wavelet decomposition at 16-64 hour frequency captures key alpha horizon",
        true, 0.07, 14, 0.70),
    ResearchFinding(22, "RL Portfolio Agent",
        "SAC reinforcement learning agent achieves Sharpe 0.8-1.2 in simulation but unstable OOS",
        false, 0.05, 60, 0.45),

    # Notebooks 23-34 (current batch)
    ResearchFinding(23, "DeFi Analytics",
        "MEV exposure for our order sizes is negligible (<\$20/trade); basis trading Sharpe 0.8-1.5",
        true, 0.08, 7, 0.80),
    ResearchFinding(24, "Systemic Risk",
        "CoVaR/MES framework: AVAX/SOL most vulnerable to BTC stress; build systemic risk monitor",
        true, 0.07, 10, 0.80),
    ResearchFinding(25, "Advanced ML (GP/SVM/VAE)",
        "GP uncertainty + SVM regime + VAE latent state ensemble: +15% IC improvement",
        true, 0.12, 21, 0.70),
    ResearchFinding(26, "Alternative Data",
        "Whale on-chain flow: highest IC of alt-data sources (0.04-0.08); composite alt-data Sharpe ~0.6",
        true, 0.15, 14, 0.75),
    ResearchFinding(27, "Numerical Methods",
        "L-BFGS for portfolio optimization: 5x faster than BFGS; Sobol MC: 10x more accurate",
        true, 0.05, 5, 0.95),
    ResearchFinding(28, "Crypto Mechanics",
        "Maker vs taker fee: 3-8 bps/trade difference; funding carry Sharpe 0.5-1.0",
        true, 0.10, 3, 0.95),
    ResearchFinding(29, "Performance Attribution",
        "London/NY overlap hours generate 60% of daily alpha; alpha half-life is 3-7 days",
        true, 0.12, 5, 0.85),
    ResearchFinding(30, "Vol Surface Crypto",
        "Vol risk premium ~8% annualized; delta-hedged straddle Sharpe 0.5-1.2",
        true, 0.10, 14, 0.80),
    ResearchFinding(31, "Advanced Portfolio Opt.",
        "HRP most robust OOS; tail risk parity best drawdown control; BL with IAE views adds 10-15% Sharpe",
        true, 0.15, 10, 0.85),
    ResearchFinding(32, "Time Series Forecasting",
        "Returns not predictable by ARIMA; DCC-GARCH for corr forecasting; PSY bubble detection actionable",
        true, 0.08, 7, 0.80),
    ResearchFinding(33, "Stress Testing",
        "2x leverage survives all scenarios; 3x+ risks blowup in combined stress; cash buffer critical",
        true, 0.10, 3, 0.95),
]

println("Total findings compiled: $(length(findings)) from notebooks 01-33")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Priority Ranking: Expected Sharpe Lift
# ─────────────────────────────────────────────────────────────────────────────

"""
Priority score = Sharpe impact * confidence / sqrt(effort_days).
Higher = implement first.
"""
function priority_score(f::ResearchFinding)
    return f.sharpe_impact * f.confidence / sqrt(f.effort_days)
end

actionable = filter(f -> f.actionable, findings)
sorted_findings = sort(actionable, by=f -> priority_score(f), rev=true)

println("\n=== Priority Ranking: Expected Sharpe Improvement ===")
println("(Score = Sharpe impact × confidence / √effort)")
println()
println(lpad("Rank", 6), lpad("NB", 5), lpad("Topic", 35), lpad("ΔSharpe", 9),
        lpad("Effort(d)", 11), lpad("Confidence", 12), lpad("Score", 8))
println("-" ^ 90)

for (rank, f) in enumerate(sorted_findings[1:min(20, length(sorted_findings))])
    score = priority_score(f)
    println(lpad(string(rank), 6),
            lpad(string(f.notebook), 5),
            lpad(f.topic[1:min(34,length(f.topic))], 35),
            lpad(string(round(f.sharpe_impact,digits=3)), 9),
            lpad(string(f.effort_days), 11),
            lpad(string(round(f.confidence,digits=2)), 12),
            lpad(string(round(score,digits=4)), 8))
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Implementation Roadmap
# ─────────────────────────────────────────────────────────────────────────────

println("\n=== Implementation Roadmap ===")
println()

"""
Phase-based implementation plan.
Phase 1: Quick wins (high score, low effort) — Weeks 1-4
Phase 2: Core improvements — Weeks 5-12
Phase 3: Research initiatives — Months 4-12
"""

# Phase 1: Quick wins
phase1 = filter(f -> priority_score(f) > 0.04 && f.effort_days <= 7, sorted_findings)
# Phase 2: Core improvements
phase2 = filter(f -> priority_score(f) > 0.02 && f.effort_days <= 21 && !(f in phase1), sorted_findings)
# Phase 3: Research
phase3 = filter(f -> !(f in phase1) && !(f in phase2), sorted_findings)

println("PHASE 1 — Quick Wins (Weeks 1-4, <7 days each):")
for f in phase1
    println("  [NB $(f.notebook)] $(f.topic): +$(round(f.sharpe_impact,digits=3)) Sharpe, $(f.effort_days)d")
    println("    → $(f.finding[1:min(80,length(f.finding))])")
end

println("\nPHASE 2 — Core Improvements (Weeks 5-12):")
for f in phase2[1:min(8, length(phase2))]
    println("  [NB $(f.notebook)] $(f.topic): +$(round(f.sharpe_impact,digits=3)) Sharpe, $(f.effort_days)d")
end

println("\nPHASE 3 — Research Initiatives (Months 4-12):")
for f in phase3[1:min(6, length(phase3))]
    println("  [NB $(f.notebook)] $(f.topic): +$(round(f.sharpe_impact,digits=3)) Sharpe (est.)")
end

# Total expected Sharpe improvement
total_sharpe = sum(f.sharpe_impact * f.confidence for f in actionable)
println("\nTotal expected cumulative Sharpe improvement: +$(round(total_sharpe,digits=3))")
println("(Note: not additive due to overlap; realistic gain = 30-50% of total)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Risk Budget Review
# ─────────────────────────────────────────────────────────────────────────────

"""Current vs optimal risk allocation."""
struct RiskBudget
    source::String
    current_pct::Float64  # current % of risk budget
    optimal_pct::Float64  # recommended % based on findings
    current_sharpe::Float64
    optimal_sharpe::Float64
end

risk_budgets = [
    RiskBudget("BTC Directional Beta",     45.0, 30.0, 0.6,  0.7),
    RiskBudget("ETH/Alt Directional",      20.0, 15.0, 0.5,  0.6),
    RiskBudget("Momentum Signals",         15.0, 20.0, 0.8,  1.0),
    RiskBudget("Mean Reversion (basis)",    5.0, 10.0, 1.2,  1.3),
    RiskBudget("Alternative Data",          3.0,  8.0, 0.6,  0.9),
    RiskBudget("Volatility (option sell)",  2.0,  7.0, 0.9,  1.1),
    RiskBudget("Execution/Market Making",   5.0,  5.0, 1.5,  1.5),
    RiskBudget("Cash / Hedge",              5.0,  5.0, 0.0,  0.0),
]

println("\n=== Risk Budget Review ===")
println(lpad("Risk Source", 28), lpad("Current%", 10), lpad("Optimal%", 10),
        lpad("Curr Sharpe", 13), lpad("Opt Sharpe", 12), lpad("Action", 10))
println("-" ^ 86)

for rb in risk_budgets
    diff = rb.optimal_pct - rb.current_pct
    action = diff > 3.0 ? "INCREASE" : diff < -3.0 ? "DECREASE" : "MAINTAIN"
    println(lpad(rb.source, 28),
            lpad(string(round(rb.current_pct,digits=0))*"%", 10),
            lpad(string(round(rb.optimal_pct,digits=0))*"%", 10),
            lpad(string(round(rb.current_sharpe,digits=2)), 13),
            lpad(string(round(rb.opt_sharpe,digits=2)), 12),
            lpad(action, 10))
end

total_current = sum(rb.current_pct for rb in risk_budgets)
total_optimal = sum(rb.optimal_pct for rb in risk_budgets)
println("\nTotal risk budget: $(total_current)% (current), $(total_optimal)% (optimal)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Signal Inventory
# ─────────────────────────────────────────────────────────────────────────────

"""Comprehensive signal inventory."""
struct Signal
    name::String
    category::String
    ic::Float64
    ic_std::Float64
    half_life_days::Int
    current_weight::Float64  # current portfolio weight
    optimal_weight::Float64  # recommended weight based on IC²
    status::String           # "live", "research", "deprecated"
end

signals = [
    Signal("BTC Trend (20d MA)",         "Momentum",     0.04, 0.08, 15, 0.20, 0.12, "live"),
    Signal("ETH Trend (20d MA)",         "Momentum",     0.03, 0.09, 12, 0.10, 0.07, "live"),
    Signal("BTC Momentum (5d)",          "Momentum",     0.05, 0.07, 5,  0.15, 0.18, "live"),
    Signal("Vol Regime (GARCH)",         "Regime",       0.08, 0.06, 7,  0.10, 0.18, "live"),
    Signal("HMM Regime Signal",          "Regime",       0.10, 0.08, 10, 0.05, 0.20, "live"),
    Signal("Order Flow Imbalance",       "Microstructure",0.08, 0.12, 2, 0.05, 0.15, "live"),
    Signal("Funding Rate Carry",         "Carry",        0.04, 0.05, 20, 0.08, 0.09, "live"),
    Signal("Basis Z-Score",              "Carry",        0.06, 0.07, 8,  0.05, 0.11, "live"),
    Signal("PCR Contrarian",             "Options",      0.03, 0.10, 5,  0.02, 0.05, "research"),
    Signal("Whale On-Chain Flow",        "Alt Data",     0.07, 0.15, 4,  0.00, 0.12, "research"),
    Signal("Search Trend Signal",        "Alt Data",     0.03, 0.10, 8,  0.00, 0.04, "research"),
    Signal("Term Structure Slope",       "Options",      0.04, 0.08, 12, 0.02, 0.06, "live"),
    Signal("MacroFactor (DXY/VIX)",      "Macro",        0.05, 0.09, 20, 0.05, 0.08, "live"),
    Signal("Jump Reversal",              "Event",        0.08, 0.15, 3,  0.02, 0.08, "live"),
    Signal("GPT Sentiment",              "Alt Data",     0.03, 0.12, 3,  0.00, 0.03, "research"),
    Signal("Vol Risk Premium",           "Volatility",   0.06, 0.08, 30, 0.02, 0.07, "live"),
    Signal("BH Physics Mass Shift",      "Regime",       0.04, 0.10, 7,  0.05, 0.06, "live"),
    Signal("ML Ensemble (GBM)",          "ML",           0.09, 0.07, 5,  0.00, 0.18, "research"),
    Signal("PSY Bubble Detector",        "Regime",       0.05, 0.15, 14, 0.00, 0.05, "research"),
    Signal("Cross-Asset Correlation",    "Risk",         0.04, 0.09, 20, 0.00, 0.04, "research"),
]

println("\n=== Signal Inventory (All Signals) ===")
println(lpad("Signal Name", 30), lpad("Category", 16), lpad("IC", 7), lpad("IC/std", 8),
        lpad("HL(d)", 7), lpad("Curr%", 7), lpad("Opt%", 7), lpad("Status", 10))
println("-" ^ 98)

# Sort by IC
sorted_signals = sort(signals, by=s -> s.ic * s.current_weight == 0 ? s.ic * 0.5 : s.ic, rev=true)

for s in sorted_signals
    ic_ir = s.ic_std > 0 ? s.ic / s.ic_std : 0.0
    println(lpad(s.name[1:min(29,length(s.name))], 30),
            lpad(s.category, 16),
            lpad(string(round(s.ic,digits=4)), 7),
            lpad(string(round(ic_ir,digits=2)), 8),
            lpad(string(s.half_life_days), 7),
            lpad(string(round(s.current_weight*100,digits=0))*"%", 7),
            lpad(string(round(s.optimal_weight*100,digits=0))*"%", 7),
            lpad(s.status, 10))
end

# IC-squared optimal weights (normalized)
ic2_weights = [s.ic^2 for s in signals]
total_ic2 = sum(ic2_weights)
opt_weights_normalized = ic2_weights ./ total_ic2

println("\nTop 5 signals by IC²-weighted optimal allocation:")
sorted_by_ic2 = sort(collect(zip(signals, opt_weights_normalized)), by=x->x[2], rev=true)
for (s, w) in sorted_by_ic2[1:5]
    println("  $(s.name): $(round(w*100,digits=1))% (IC=$(round(s.ic,digits=4)))")
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. 12-Month Research Agenda
# ─────────────────────────────────────────────────────────────────────────────

println("\n=== 12-Month Research Agenda (Apr 2026 - Mar 2027) ===")

agenda = [
    # Q1 (Apr-Jun 2026): Implementation of quick wins + data infrastructure
    (quarter="Q1 Apr-Jun", month=1, priority="P1",
     hypothesis="H1: Whale on-chain flow signal generates IC > 0.06 on 1-day BTC returns when combined with funding rate context",
     test="Paper trade: whale signal + funding regime filter vs standalone",
     success_criteria="IC > 0.06 in live paper trading over 60 days"),
    (quarter="Q1 Apr-Jun", month=1, priority="P1",
     hypothesis="H2: Regime-conditional position sizing (HMM-based) improves realized Sharpe by ≥0.2 vs static sizing",
     test="Deploy HMM regime detector live; compare 90-day rolling Sharpe",
     success_criteria="Sharpe improvement ≥0.15 over 90-day window"),
    (quarter="Q1 Apr-Jun", month=2, priority="P1",
     hypothesis="H3: Maker fill rate >70% reduces transaction cost drag from ~3 bps to <1 bps per trade",
     test="Switch primary orders to limit orders; measure fill rate and effective spread",
     success_criteria="Maker fill rate >65%, effective cost <1.5 bps"),
    (quarter="Q1 Apr-Jun", month=3, priority="P2",
     hypothesis="H4: BTC-basis trading (spot-perp convergence) achieves Sharpe > 1.0 with 5% AUM allocation",
     test="Backtest 2-year data, then paper trade with 5% allocation for 60 days",
     success_criteria="Sharpe > 0.8 OOS, max drawdown < 8%"),

    # Q2 (Jul-Sep 2026): ML infrastructure + alt-data
    (quarter="Q2 Jul-Sep", month=4, priority="P1",
     hypothesis="H5: GBM ML ensemble achieves IC > 0.08 on 1-day returns using 30+ features",
     test="Train on 2022-2025 data; OOS test on 2026; walk-forward validation",
     success_criteria="IC > 0.06 OOS, Sharpe > 0.8 in paper trading"),
    (quarter="Q2 Jul-Sep", month=5, priority="P2",
     hypothesis="H6: Black-Litterman with 5 IAE hypotheses improves allocation Sharpe by 10-15% vs equal-weight IAE",
     test="Implement BL model; compare risk-adjusted performance vs control",
     success_criteria="Sharpe improvement ≥0.1 over 90-day window"),
    (quarter="Q2 Jul-Sep", month=6, priority="P2",
     hypothesis="H7: Alt-data composite (whale + options PCR + search trends) has IC > 0.05 with diversification benefit vs standalone",
     test="Compute IC and correlation between alt-data signals; build IC-weighted composite",
     success_criteria="Composite IC > any individual signal by ≥20%"),

    # Q3 (Oct-Dec 2026): Vol strategies + advanced execution
    (quarter="Q3 Oct-Dec", month=7, priority="P2",
     hypothesis="H8: Delta-hedged vol selling strategy achieves Sharpe > 0.8 with max drawdown < 20%",
     test="Deploy small-scale vol selling (5% AUM) on Deribit; track actual P&L vs model",
     success_criteria="Sharpe > 0.6 live, max drawdown < 20% in 90 days"),
    (quarter="Q3 Oct-Dec", month=8, priority="P2",
     hypothesis="H9: L-BFGS portfolio rebalancing with HRP reduces turnover by 20% vs MVO while maintaining Sharpe",
     test="Compare rebalancing schedules; measure implemented costs",
     success_criteria="Turnover reduction ≥15%, Sharpe maintained within 10%"),
    (quarter="Q3 Oct-Dec", month=9, priority="P3",
     hypothesis="H10: PSY bubble detector, when triggered, reduces BTC allocation by 50% and improves drawdown control",
     test="Backtest PSY signal overlay 2019-2026; paper trade forward",
     success_criteria="Max drawdown improvement ≥20% vs full-exposure strategy"),

    # Q4 (Jan-Mar 2027): Cross-asset + systemic risk management
    (quarter="Q4 Jan-Mar", month=10, priority="P2",
     hypothesis="H11: DCC-GARCH dynamic correlation improves portfolio VaR accuracy by 30% vs static correlation VaR",
     test="Compare DCC vs static VaR exceedances on rolling 90-day windows",
     success_criteria="VaR exceedance rate closer to 5% with DCC vs static"),
    (quarter="Q4 Jan-Mar", month=11, priority="P3",
     hypothesis="H12: Tail risk parity allocation reduces max drawdown by ≥25% vs equal-weight with <10% Sharpe sacrifice",
     test="Implement tail risk parity; 90-day live comparison",
     success_criteria="Max drawdown improvement ≥20%, Sharpe sacrifice <10%"),
    (quarter="Q4 Jan-Mar", month=12, priority="P1",
     hypothesis="H13: Combined system (regime + momentum + ML + alt-data + optimal execution) achieves Sharpe > 2.0 in simulation",
     test="Full system integration test; walk-forward 2024-2026 data",
     success_criteria="Sharpe > 1.5 OOS, Calmar > 1.0, max drawdown < 20%"),
]

println()
for item in agenda
    println("$(item.quarter) — $(item.priority) | $(item.hypothesis[1:min(75,length(item.hypothesis))])")
    println("  Test: $(item.test[1:min(90,length(item.test))])")
    println("  Success: $(item.success_criteria)")
    println()
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Overall Research Summary Statistics
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 70)
println("RESEARCH SUMMARY STATISTICS")
println("=" ^ 70)
println()

n_live = sum(s.status == "live" for s in signals)
n_research = sum(s.status == "research" for s in signals)
avg_ic_live = mean([s.ic for s in signals if s.status == "live"])
avg_ic_research = mean([s.ic for s in signals if s.status == "research"])

println("Signal inventory:")
println("  Live signals: $n_live (avg IC: $(round(avg_ic_live,digits=4)))")
println("  Research signals: $n_research (avg IC: $(round(avg_ic_research,digits=4)))")
println("  Total signals tracked: $(length(signals))")

n_phase1 = length(phase1)
n_phase2 = length(phase2)
n_phase3 = length(phase3)
total_effort = sum(f.effort_days for f in [phase1; phase2; phase3])

println("\nImplementation roadmap:")
println("  Phase 1 (quick wins): $n_phase1 items")
println("  Phase 2 (core): $n_phase2 items")
println("  Phase 3 (research): $n_phase3 items")
println("  Total estimated effort: $total_effort developer-days")

sharpe_p1 = sum(f.sharpe_impact * f.confidence for f in phase1)
sharpe_p2 = sum(f.sharpe_impact * f.confidence for f in phase2)
println("\nExpected Sharpe improvement (risk-adjusted by confidence):")
println("  Phase 1 alone: +$(round(sharpe_p1,digits=3))")
println("  Phase 1+2: +$(round(sharpe_p1+sharpe_p2,digits=3))")
println("  All phases: +$(round(total_sharpe,digits=3)) (nominal), +$(round(total_sharpe*0.4,digits=3)) (realistic)")

println("\n12-month research agenda: 13 testable hypotheses")
p1_count = sum(item.priority == "P1" for item in agenda)
p2_count = sum(item.priority == "P2" for item in agenda)
p3_count = sum(item.priority == "P3" for item in agenda)
println("  P1 (must-test): $p1_count hypotheses")
println("  P2 (important): $p2_count hypotheses")
println("  P3 (exploratory): $p3_count hypotheses")

println("\n" * "=" ^ 70)
println("KEY STRATEGIC RECOMMENDATIONS")
println("=" ^ 70)
println("""
IMMEDIATE ACTIONS (next 30 days):
  1. Switch to limit orders: reduce fee drag by 2-4 bps/trade (NB28)
  2. Deploy HMM regime detector live: regime-conditional sizing (NB14/NB2)
  3. Implement CVaR position limits replacing VaR limits (NB18)
  4. Hard stop at -15% AUM drawdown + 2x max leverage cap (NB33)

MEDIUM-TERM (1-3 months):
  5. Integrate whale on-chain flow signal (NB26): highest IC alt-data source
  6. Deploy HRP portfolio optimization replacing equal-weight (NB31)
  7. Add basis trading strategy with 5-10% AUM allocation (NB28)
  8. Build GBM ML ensemble with 30-feature set (NB25/NB12)

LONG-TERM RESEARCH (3-12 months):
  9. Develop vol selling overlay strategy (NB30)
  10. Integrate full BL + IAE views framework (NB31)
  11. Build real-time DCC correlation tracker for dynamic risk (NB32)
  12. Deploy PSY bubble detector as tactical allocation filter (NB32)

RISK PRIORITIES:
  - Reduce BTC directional beta from 45% to 30% of risk budget
  - Increase momentum and mean-reversion allocations
  - Maintain 10%+ cash buffer for liquidity in stress scenarios
  - Never exceed 2x leverage given combined stress test results (NB33)
""")
