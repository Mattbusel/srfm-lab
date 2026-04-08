"""
analogy_engine.py
=================
Creative cross-domain analogy engine for generating novel trading hypotheses.

The core insight: financial markets are complex adaptive systems, and many
phenomena that appear market-specific have deep structural parallels in
ecology, physics, biology, and game theory. By formalizing these analogies,
we can systematically generate testable hypotheses that would not arise from
purely quantitative first-principles analysis.

Architecture
------------
- AnalogyLibrary   : 22 curated analogies with formal concept mappings
- AnalogyEngine    : selects, combines, and instantiates analogies
- HypothesisCandidate : structured output with source/target/predictions
- CrossDomainSearch : keyword-based search over the library
"""

from __future__ import annotations

import random
import itertools
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


# ──────────────────────────────────────────────────────────────────────────────
# Domain taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class Domain(str, Enum):
    ECOLOGY            = "ecology"
    GAME_THEORY        = "game_theory"
    INFORMATION_THEORY = "information_theory"
    BIOLOGY            = "biology"
    COMPLEX_SYSTEMS    = "complex_systems"
    FLUID_DYNAMICS     = "fluid_dynamics"
    MATERIALS_SCIENCE  = "materials_science"
    FINANCE            = "finance"


class Confidence(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ──────────────────────────────────────────────────────────────────────────────
# Core data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalogyMapping:
    """A single conceptual mapping from source-domain concept to market concept."""
    source_concept: str
    target_concept: str
    mapping_rationale: str


@dataclass
class Analogy:
    """
    A formal analogy from a scientific source domain to financial markets.

    Fields
    ------
    name          : Short snake_case identifier
    source_domain : Originating scientific domain
    description   : Plain-English description of the source phenomenon
    mappings      : Ordered list of concept-level correspondences
    prediction    : The specific market behaviour the analogy predicts
    conditions    : Empirical conditions under which the analogy holds
    falsifiers    : Observations that would definitively falsify the analogy
    confidence    : Prior probability that the analogy generates actionable alpha
    tags          : Searchable keyword list
    """
    name: str
    source_domain: Domain
    description: str
    mappings: list[AnalogyMapping]
    prediction: str
    conditions: list[str]
    falsifiers: list[str]
    confidence: Confidence
    tags: list[str] = field(default_factory=list)


@dataclass
class HypothesisCandidate:
    """
    A structured, testable trading hypothesis generated from an analogy.
    Suitable for hand-off to the quantitative research pipeline.
    """
    hypothesis_id: str
    source_domain: Domain
    target_domain: Domain
    analogy_name: str
    title: str
    narrative: str
    prediction: str
    testable_conditions: list[str]
    falsifiers: list[str]
    suggested_signals: list[str]
    suggested_universe: str
    suggested_lookback: str
    confidence: Confidence
    novelty_score: float       # 0–1, how far the leap is from finance
    cross_domain_score: float  # 0–1, conceptual distance
    provenance: list[str] = field(default_factory=list)   # analogy chain

    def summary(self) -> str:
        lines = [
            f"╔══ {self.title} ══",
            f"  ID         : {self.hypothesis_id}",
            f"  Source     : {self.source_domain.value}",
            f"  Analogy    : {self.analogy_name}",
            f"  Confidence : {self.confidence.value}",
            f"  Novelty    : {self.novelty_score:.2f}",
            f"  Universe   : {self.suggested_universe}",
            f"  Lookback   : {self.suggested_lookback}",
            "",
            "  Narrative:",
        ]
        for line in self.narrative.split(". "):
            lines.append(f"    {line.strip()}.")
        lines += [
            "",
            "  Prediction:",
            f"    {self.prediction}",
            "",
            "  Testable conditions:",
        ]
        for c in self.testable_conditions:
            lines.append(f"    • {c}")
        lines += ["", "  Suggested signals:"]
        for s in self.suggested_signals:
            lines.append(f"    • {s}")
        lines += ["", "  Falsifiers:"]
        for f_ in self.falsifiers:
            lines.append(f"    ✗ {f_}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# The Analogy Library  —  22 structured analogies
# ──────────────────────────────────────────────────────────────────────────────

ANALOGY_LIBRARY: list[Analogy] = [

    # ── ECOLOGY (4) ──────────────────────────────────────────────────────────

    Analogy(
        name="predator_prey_cycles",
        source_domain=Domain.ECOLOGY,
        description=(
            "In Lotka–Volterra dynamics, predator and prey populations oscillate "
            "with a predictable phase lag: prey abundance peaks before predators "
            "do. Predator overshoot depletes prey, causing predator collapse, which "
            "then allows prey to recover. The cycle repeats endogenously."
        ),
        mappings=[
            AnalogyMapping("prey population",    "price momentum / alpha",
                           "Abundant trending prices attract momentum capital"),
            AnalogyMapping("predator population", "trend-following AUM",
                           "Capital chasing trends is the 'predator'"),
            AnalogyMapping("predation rate",      "crowding / capacity erosion",
                           "More trend-followers degrade the signal they exploit"),
            AnalogyMapping("prey reproduction",   "continuous mispricing generation",
                           "Markets continuously regenerate exploitable inefficiency"),
        ],
        prediction=(
            "Trend-following returns should exhibit multi-year cycles inversely "
            "correlated with estimated CTA / trend-following AUM. When trend AUM is "
            "elevated relative to a 3-year trailing average, forward 12-month "
            "trend-following returns should be significantly below average."
        ),
        conditions=[
            "Observable proxy for trend-following AUM (CTA indices, 13-F filings)",
            "Sufficient time series length (>15 years) to observe multiple cycles",
            "Liquid futures markets where CTAs are known to concentrate",
        ],
        falsifiers=[
            "No negative autocorrelation between CTA AUM growth and subsequent returns",
            "Alpha erosion fully explained by transaction costs alone",
        ],
        confidence=Confidence.HIGH,
        tags=["trend", "crowding", "AUM", "cycles", "capacity"],
    ),

    Analogy(
        name="ecological_niche",
        source_domain=Domain.ECOLOGY,
        description=(
            "Species occupy niches defined by resource utilisation. Gause's "
            "competitive exclusion principle prevents two species from occupying "
            "the exact same niche at equilibrium; niche differentiation allows "
            "stable coexistence of multiple strategies."
        ),
        mappings=[
            AnalogyMapping("ecological niche",     "alpha source / strategy type",
                           "Each strategy exploits a distinct market inefficiency"),
            AnalogyMapping("competitive exclusion", "strategy convergence / crowding",
                           "Identical strategies compete until one is driven out"),
            AnalogyMapping("niche differentiation", "strategy diversification",
                           "Strategies differing on time-scale or signal survive together"),
            AnalogyMapping("resource abundance",    "available mispricing pool",
                           "Finite pool of exploitable edge shrinks as AUM grows"),
        ],
        prediction=(
            "Pairs of hedge fund strategies with high pairwise signal correlation "
            "should show declining combined Sharpe over time as combined AUM grows; "
            "complementary-niche strategy pairs should show stable combined Sharpe."
        ),
        conditions=[
            "Ability to measure signal overlap between strategy indices",
            "Multi-strategy fund returns or index data spanning >10 years",
        ],
        falsifiers=[
            "No AUM-conditioned performance degradation for high-overlap strategies",
        ],
        confidence=Confidence.MEDIUM,
        tags=["strategy", "diversification", "crowding", "niche", "alpha decay"],
    ),

    Analogy(
        name="invasive_species",
        source_domain=Domain.ECOLOGY,
        description=(
            "Invasive species introduced to a new environment initially grow "
            "exponentially (no natural predators, abundant resources), then crash "
            "after depleting resources or after the native ecosystem adapts. "
            "The trajectory is sigmoid not linear."
        ),
        mappings=[
            AnalogyMapping("invasive species",        "novel financial instrument",
                           "New instruments enter with no established pricing framework"),
            AnalogyMapping("absence of predators",    "absence of informed arbitrageurs",
                           "New instruments lack short-sellers and hedgers initially"),
            AnalogyMapping("exponential growth phase","early-adoption bubble phase",
                           "Price disconnects from fundamentals while niche is uncontested"),
            AnalogyMapping("ecosystem adaptation",    "derivatives/shorting introduction",
                           "Hedging tools mature → price efficiency improves rapidly"),
        ],
        prediction=(
            "Newly listed instruments (first ETF on an asset class, first futures "
            "contract) should exhibit elevated volatility and positive momentum in "
            "the first 12–24 months, followed by mean-reversion and vol compression "
            "as arbitrage capacity matures. Magnitude scales with novelty of the asset."
        ),
        conditions=[
            "Instrument launch date with at least 36 months post-launch data",
            "Short-selling availability and options open interest timeline",
        ],
        falsifiers=[
            "New instruments price efficiently from day one with no excess momentum",
        ],
        confidence=Confidence.HIGH,
        tags=["new instruments", "bubble", "efficiency", "short interest", "ETF"],
    ),

    Analogy(
        name="keystone_species",
        source_domain=Domain.ECOLOGY,
        description=(
            "A keystone species has disproportionate ecosystem impact relative "
            "to its biomass. Its removal triggers cascade collapse because it "
            "maintains the web of interdependencies that others rely on."
        ),
        mappings=[
            AnalogyMapping("keystone species",      "systemically important dealer/asset",
                           "A single actor whose distress destabilises the whole network"),
            AnalogyMapping("ecosystem collapse",    "contagion / market crash cascade",
                           "Removal of a key liquidity provider triggers wave of failures"),
            AnalogyMapping("disproportionate role", "network centrality vs. balance sheet size",
                           "Keystone entities may be small by AUM but central by flow"),
        ],
        prediction=(
            "Network betweenness-centrality on the asset correlation graph identifies "
            "'keystone' assets. Exogenous shocks to high-centrality assets should produce "
            "statistically larger cross-asset drawdowns than equal-magnitude shocks to "
            "low-centrality assets, controlling for asset volatility."
        ),
        conditions=[
            "Intraday or daily returns for a broad cross-section of assets",
            "Ability to identify exogenous, asset-specific shocks",
        ],
        falsifiers=[
            "Betweenness centrality does not predict contagion magnitude",
        ],
        confidence=Confidence.HIGH,
        tags=["systemic risk", "contagion", "network", "centrality", "cascade"],
    ),

    # ── GAME THEORY (3) ──────────────────────────────────────────────────────

    Analogy(
        name="prisoners_dilemma_market_making",
        source_domain=Domain.GAME_THEORY,
        description=(
            "In a repeated Prisoner's Dilemma, individually rational defection "
            "leads to collectively suboptimal outcomes. Repeated interaction "
            "enables cooperation (tit-for-tat); defection triggers punishment phases "
            "that restore incentive compatibility."
        ),
        mappings=[
            AnalogyMapping("cooperate",       "patient two-sided market-making",
                           "Providing tight quotes sustains healthy price discovery"),
            AnalogyMapping("defect",          "aggressive toxic flow / front-running",
                           "Traders who exploit market-makers degrade liquidity"),
            AnalogyMapping("punishment phase","liquidity withdrawal / spread widening",
                           "Market-makers widen spreads after detecting toxic flow"),
            AnalogyMapping("grim trigger",    "algorithmic toxicity detection",
                           "HFT systems withdraw quotes upon detecting adverse selection"),
        ],
        prediction=(
            "Following detected order-flow toxicity episodes (elevated VPIN, high "
            "PIN estimates, large order imbalance), bid-ask spreads should widen "
            "disproportionately within the next trading session as market-makers "
            "impose a punishment phase."
        ),
        conditions=[
            "Intraday data with order imbalance or VPIN computation",
            "Tick-level spread data with sufficient cross-sectional breadth",
        ],
        falsifiers=[
            "Post-toxicity spread widening is fully explained by inventory effects",
        ],
        confidence=Confidence.MEDIUM,
        tags=["microstructure", "liquidity", "spreads", "game theory", "VPIN"],
    ),

    Analogy(
        name="coordination_game_focal_points",
        source_domain=Domain.GAME_THEORY,
        description=(
            "In coordination games with multiple equilibria, agents converge on "
            "Schelling focal points — psychologically salient solutions that everyone "
            "expects others to choose. Equilibrium selection is cultural and historical."
        ),
        mappings=[
            AnalogyMapping("Schelling focal point", "round-number price levels",
                           "Psychologically salient price levels coordinate expectations"),
            AnalogyMapping("coordination failure",  "flash crash / equilibrium gap",
                           "Loss of shared focal point causes sudden equilibrium shift"),
            AnalogyMapping("equilibrium selection", "self-fulfilling support/resistance",
                           "Commonly watched technical levels become self-fulfilling"),
        ],
        prediction=(
            "Price reactions at psychologically salient levels (multiples of 100 or "
            "1000, all-time highs/lows, index round numbers) should be statistically "
            "larger in magnitude than at non-salient nearby levels, even after "
            "controlling for mechanical limit-order clustering at round numbers."
        ),
        conditions=[
            "Tick or minute-level price data with sufficient resolution",
            "Large enough cross-section to distinguish mechanical from strategic effects",
        ],
        falsifiers=[
            "After removing mechanical order clustering, no excess price reaction",
        ],
        confidence=Confidence.HIGH,
        tags=["technical analysis", "round numbers", "focal points", "coordination"],
    ),

    Analogy(
        name="evolutionary_stable_strategy",
        source_domain=Domain.GAME_THEORY,
        description=(
            "An Evolutionarily Stable Strategy (ESS) cannot be invaded by a "
            "rare mutant strategy when already played by the population majority. "
            "Mixed ESS implies stable coexistence of multiple strategy phenotypes "
            "at fixed frequencies determined by payoff crossovers."
        ),
        mappings=[
            AnalogyMapping("ESS",              "dominant fund strategy mix",
                           "Market strategy mix that resists entry of new capital"),
            AnalogyMapping("mutant strategy",  "novel quant approach",
                           "New approach with positive expected value before crowding"),
            AnalogyMapping("invasion barrier", "replication difficulty / data moat",
                           "Proprietary data and infrastructure prevent easy copying"),
            AnalogyMapping("payoff matrix",    "alpha vs. Sharpe under competition",
                           "Strategies compete on risk-adjusted returns, not raw PnL"),
        ],
        prediction=(
            "Strategies with high barriers to replication (proprietary alternative "
            "data, high-frequency infrastructure) should exhibit more persistent "
            "alpha than easily replicated strategies, measured by 5-year IR decay."
        ),
        conditions=[
            "Cross-sectional hedge fund return data across strategy styles",
            "Proxy for replication difficulty (data cost, latency requirements)",
        ],
        falsifiers=[
            "Alpha persistence is uncorrelated with replication difficulty proxy",
        ],
        confidence=Confidence.MEDIUM,
        tags=["alpha persistence", "barriers", "strategy competition", "ESS"],
    ),

    # ── INFORMATION THEORY (3) ───────────────────────────────────────────────

    Analogy(
        name="shannon_channel_capacity",
        source_domain=Domain.INFORMATION_THEORY,
        description=(
            "Shannon's channel capacity theorem: a noisy channel has a maximum "
            "rate of error-free information transmission, determined by its "
            "signal-to-noise ratio. Exceeding capacity causes irrecoverable errors. "
            "This is a hard physical limit, not a practical one."
        ),
        mappings=[
            AnalogyMapping("channel capacity",   "alpha capacity of a trading signal",
                           "Maximum AUM a signal can support before noise dominates"),
            AnalogyMapping("signal-to-noise",    "information ratio at small scale",
                           "High IR at low AUM degrades continuously with scale"),
            AnalogyMapping("bandwidth",          "trading frequency / turnover",
                           "Higher-frequency strategies have higher dollar capacity"),
            AnalogyMapping("channel noise",      "market impact + transaction costs",
                           "The 'noise' that limits the effective signal"),
        ],
        prediction=(
            "The information ratio of a strategy should decline approximately as "
            "IR(AUM) = IR_0 / sqrt(1 + AUM / C) where C is a capacity constant "
            "that scales positively with turnover. Cross-strategy test: strategies "
            "with higher turnover should have larger capacity constants in dollar terms."
        ),
        conditions=[
            "Time series of strategy returns with associated AUM (TASS database or similar)",
            "Multiple strategies across frequency bands (daily, weekly, monthly)",
        ],
        falsifiers=[
            "IR shows no systematic relationship to AUM relative to turnover",
        ],
        confidence=Confidence.HIGH,
        tags=["capacity", "information ratio", "scaling", "AUM", "degradation"],
    ),

    Analogy(
        name="data_compression_entropy",
        source_domain=Domain.INFORMATION_THEORY,
        description=(
            "Lempel–Ziv complexity measures sequence compressibility as a proxy for "
            "entropy rate. Highly compressible sequences are redundant and carry less "
            "new information per symbol. Sample entropy (SampEn) quantifies regularity "
            "in time series without assuming stationarity."
        ),
        mappings=[
            AnalogyMapping("compressible sequence",  "predictable return pattern",
                           "Low-entropy returns indicate persistent exploitable structure"),
            AnalogyMapping("LZ complexity",           "signal entropy score",
                           "Low LZ complexity → more predictable return path"),
            AnalogyMapping("entropy rate",            "alpha decay rate",
                           "Higher entropy = faster decay of predictive content"),
        ],
        prediction=(
            "Assets ranked by low Sample Entropy (SampEn) on trailing 252-day returns "
            "should exhibit higher forward momentum or mean-reversion predictability "
            "than high-SampEn assets, in a cross-sectional test across liquid equities."
        ),
        conditions=[
            ">500 daily observations per asset for stable SampEn estimation",
            "Cross-section of ≥200 liquid equities with consistent data",
        ],
        falsifiers=[
            "SampEn quintile has no monotonic relationship to OOS momentum/MR t-stat",
        ],
        confidence=Confidence.MEDIUM,
        tags=["entropy", "predictability", "time series", "complexity", "SampEn"],
    ),

    Analogy(
        name="source_coding_model_mismatch",
        source_domain=Domain.INFORMATION_THEORY,
        description=(
            "The source coding theorem states that optimal lossless compression "
            "requires knowing the true data distribution. Using a mismatched model "
            "incurs an information-theoretic penalty equal to the KL divergence "
            "between the true and assumed distributions."
        ),
        mappings=[
            AnalogyMapping("assumed distribution", "risk model / covariance estimate",
                           "Factor model or Gaussian covariance used for construction"),
            AnalogyMapping("true distribution",    "realised joint return distribution",
                           "Actual fat-tailed, correlated distribution that generates returns"),
            AnalogyMapping("KL-divergence penalty","model risk / unexpected drawdown",
                           "Mismatch between model and reality costs portfolio performance"),
        ],
        prediction=(
            "Portfolios constructed with flexible distribution assumptions (t-copula, "
            "regime-switching covariance) should outperform Gaussian-assumption "
            "equivalents specifically during tail events (VIX > 30), as measured by "
            "max drawdown and conditional Sharpe ratio."
        ),
        conditions=[
            "Controlled experiment: identical alpha signals, varied risk model",
            "Long enough history to include multiple tail episodes",
        ],
        falsifiers=[
            "Distribution choice does not affect drawdown outcomes in tail environments",
        ],
        confidence=Confidence.HIGH,
        tags=["risk model", "distribution", "tail risk", "KL divergence", "model risk"],
    ),

    # ── BIOLOGY (4) ──────────────────────────────────────────────────────────

    Analogy(
        name="homeostasis_feedback",
        source_domain=Domain.BIOLOGY,
        description=(
            "Biological homeostasis maintains internal state within a viable range "
            "via negative feedback loops. Deviations trigger corrective responses "
            "with characteristic time lags and potential overshoots. The system "
            "trades speed of correction against oscillation risk."
        ),
        mappings=[
            AnalogyMapping("setpoint",           "fundamental fair value",
                           "Intrinsic value acts as the target the system returns to"),
            AnalogyMapping("negative feedback",  "mean-reversion / arbitrage capital",
                           "Deviations attract capital that corrects the price"),
            AnalogyMapping("feedback lag",       "price discovery lag",
                           "Arbitrageurs observe and react with a structural delay"),
            AnalogyMapping("overshoot",          "momentum continuation past fair value",
                           "Aggressive mean-reversion capital overshoots the target"),
        ],
        prediction=(
            "Assets with larger analyst coverage and higher institutional ownership "
            "(proxies for stronger feedback strength) should exhibit faster half-life "
            "of reversion to P/E z-score of zero, measured in trading days."
        ),
        conditions=[
            "Cross-sectional analyst coverage and institutional ownership data",
            "Consistent fundamental valuation metric across the universe",
        ],
        falsifiers=[
            "Coverage and ownership do not predict reversion half-life",
        ],
        confidence=Confidence.MEDIUM,
        tags=["mean reversion", "feedback", "fundamental", "efficiency", "arbitrage"],
    ),

    Analogy(
        name="circadian_rhythm_intraday",
        source_domain=Domain.BIOLOGY,
        description=(
            "Circadian rhythms are endogenous ~24-hour biological clocks synchronised "
            "to external zeitgebers (light/dark cycle, temperature). They persist even "
            "without external cues but can be phase-shifted by strong stimuli. "
            "Different organisms have different chronotypes."
        ),
        mappings=[
            AnalogyMapping("circadian oscillation", "U-shaped intraday volume pattern",
                           "Markets have robust intraday periodicity in activity"),
            AnalogyMapping("zeitgeber",             "scheduled economic releases",
                           "Economic announcements reset and synchronise intraday rhythms"),
            AnalogyMapping("chronotype variation",  "asset-class-specific peak times",
                           "Equities, bonds, and FX have different peak-activity windows"),
            AnalogyMapping("phase shift",           "daylight saving / market hour changes",
                           "Calendar disruptions shift intraday volume/volatility shapes"),
        ],
        prediction=(
            "Cross-asset strategies exploiting intraday timing mismatches "
            "(asset A peaks at open, asset B peaks at close) should show statistically "
            "significant directional drift between the two activity windows, "
            "after controlling for overnight gap effects."
        ),
        conditions=[
            "5-minute or finer intraday data across multiple asset classes",
            "≥250 trading days per asset for stable intraday shape estimation",
        ],
        falsifiers=[
            "Intraday timing differences are not predictive cross-sectionally",
        ],
        confidence=Confidence.HIGH,
        tags=["intraday", "seasonality", "volume", "timing", "cross-asset"],
    ),

    Analogy(
        name="immune_memory_crisis",
        source_domain=Domain.BIOLOGY,
        description=(
            "Adaptive immunity: first pathogen exposure triggers a slow, costly "
            "primary immune response. Re-exposure to the same pathogen triggers a "
            "fast, amplified secondary response via memory B-cells. Autoimmune "
            "disorders occur when immune response attacks self-tissue."
        ),
        mappings=[
            AnalogyMapping("pathogen signature",   "stress-event factor pattern",
                           "A specific cluster of factor exposures defines a stress type"),
            AnalogyMapping("memory B-cells",       "institutional risk limits / VaR models",
                           "Risk managers 'remember' previous stress signatures"),
            AnalogyMapping("secondary response",   "faster deleveraging on familiar shock",
                           "Recognised patterns trigger faster risk-off"),
            AnalogyMapping("autoimmune disorder",  "self-fulfilling risk-off cascade",
                           "Risk management rules trigger the very crisis they guard against"),
        ],
        prediction=(
            "The second occurrence of a stress event with the same factor signature "
            "as a prior major drawdown should produce a faster initial selling "
            "wave (higher intraday speed-of-decline) but ultimately a smaller "
            "peak-to-trough loss than the first occurrence."
        ),
        conditions=[
            "Factor-model decomposition of historical crisis returns",
            "Ability to classify crises by signature similarity (cosine distance)",
        ],
        falsifiers=[
            "Second-occurrence crises are not faster or shallower than first occurrences",
        ],
        confidence=Confidence.MEDIUM,
        tags=["crisis", "stress", "risk management", "memory", "drawdown"],
    ),

    Analogy(
        name="punctuated_equilibrium_vol",
        source_domain=Domain.BIOLOGY,
        description=(
            "Gould and Eldredge's punctuated equilibrium: evolution proceeds via "
            "long periods of stasis punctuated by rapid morphological change. "
            "Stasis = strong stabilising selection; punctuation = environmental "
            "disruption removes selection pressure and opens new niches."
        ),
        mappings=[
            AnalogyMapping("stasis",        "low-volatility macro regime",
                           "Stable policy environment keeps prices range-bound"),
            AnalogyMapping("punctuation",   "regime change / volatility spike",
                           "Macro shock resets the pricing equilibrium suddenly"),
            AnalogyMapping("speciation",    "sector rotation / factor regime flip",
                           "New equilibrium selects for different factor exposures"),
        ],
        prediction=(
            "Periods of unusually low realised volatility (bottom quintile of "
            "trailing 252-day realised vol) should predict elevated probability of "
            "a subsequent volatility spike (top quintile) within 6–18 months, "
            "significantly above base rate, consistent with non-linear dynamics."
        ),
        conditions=[
            "Long time series of realised volatility (≥20 years)",
            "Multiple regime episodes for statistical power",
        ],
        falsifiers=[
            "Low-vol periods have the same forward vol distribution as all periods",
        ],
        confidence=Confidence.HIGH,
        tags=["volatility", "regime change", "low-vol anomaly", "tail risk", "punctuation"],
    ),

    # ── COMPLEX SYSTEMS (4) ──────────────────────────────────────────────────

    Analogy(
        name="self_organised_criticality",
        source_domain=Domain.COMPLEX_SYSTEMS,
        description=(
            "Bak, Tang & Wiesenfeld: many dissipative systems self-organise to a "
            "critical state where perturbations trigger avalanches of all sizes "
            "following a power-law distribution. The sandpile model is archetypal: "
            "each grain addition can cause avalanches from single grains to full collapse."
        ),
        mappings=[
            AnalogyMapping("critical state",    "market near systemic tipping point",
                           "Leverage, correlation, and concentration near threshold"),
            AnalogyMapping("sand grain",        "individual trade or news item",
                           "Each event has unpredictable systemic impact magnitude"),
            AnalogyMapping("avalanche",         "cascade liquidation / crash",
                           "Power-law distributed loss events across all time scales"),
            AnalogyMapping("sandpile slope",    "aggregate system leverage",
                           "Steeper slope (more leverage) = larger potential avalanche"),
        ],
        prediction=(
            "The distribution of intraday drawdown sizes should fit a power law "
            "whose tail exponent decreases (heavier tails) as aggregate estimated "
            "leverage (margin debt / NYSE composite ratio) increases. "
            "Kolmogorov-Smirnov test of power-law fit should be significant."
        ),
        conditions=[
            "High-frequency intraday return data across long history (≥10 years)",
            "Quarterly proxy for aggregate leverage (margin balances, prime broker surveys)",
        ],
        falsifiers=[
            "Drawdown size distribution fails power-law fit",
            "Tail exponent does not correlate with leverage proxy across time",
        ],
        confidence=Confidence.HIGH,
        tags=["power law", "crash", "leverage", "tail risk", "SOC", "avalanche"],
    ),

    Analogy(
        name="tipping_points_early_warning",
        source_domain=Domain.COMPLEX_SYSTEMS,
        description=(
            "Systems near tipping points exhibit 'critical slowing down': "
            "autocorrelation increases, variance increases, and recovery from "
            "small perturbations takes longer. These statistical fingerprints are "
            "early warning signals appearing before the transition itself."
        ),
        mappings=[
            AnalogyMapping("critical slowing down",  "pre-crash autocorrelation rise",
                           "Returns become more autocorrelated before a major crash"),
            AnalogyMapping("rising variance",         "implied vol rising trend",
                           "Market anticipates larger future moves"),
            AnalogyMapping("slower recovery",         "lengthening drawdown durations",
                           "Each dip takes longer to fully recover as system weakens"),
        ],
        prediction=(
            "A composite early-warning index (30-day AR(1) coefficient of index returns "
            "+ 21-day realised variance trend + mean time to recover 0.5% intraday dips) "
            "should predict the probability of a ≥10% index drawdown in the next 60 days "
            "with positive in-sample and out-of-sample AUROC > 0.60."
        ),
        conditions=[
            "Daily index return data spanning ≥5 completed bear markets",
            "Intraday data for recovery-time computation",
        ],
        falsifiers=[
            "Composite early-warning index has AUROC ≤ 0.50 out-of-sample",
        ],
        confidence=Confidence.HIGH,
        tags=["crash prediction", "early warning", "variance", "autocorrelation", "AUROC"],
    ),

    Analogy(
        name="ising_model_herding",
        source_domain=Domain.COMPLEX_SYSTEMS,
        description=(
            "The Ising model of ferromagnetism maps to opinion/herding dynamics: "
            "each agent's 'spin' (opinion) is influenced by nearest neighbours. "
            "Near the Curie temperature, susceptibility diverges and small external "
            "fields cause macroscopic magnetisation shifts."
        ),
        mappings=[
            AnalogyMapping("spin alignment",   "correlated trader positions",
                           "Traders aligning direction = spins aligning magnetically"),
            AnalogyMapping("external field",   "central bank policy signal",
                           "Policy announcements 'magnetise' market in a direction"),
            AnalogyMapping("Curie temperature","critical correlation level",
                           "Near critical correlation, small news causes large moves"),
            AnalogyMapping("susceptibility",   "market impact amplification",
                           "Amplification of news impact near critical state"),
        ],
        prediction=(
            "During periods when cross-asset pairwise correlation is in the top "
            "quintile (system near 'Curie temperature'), the 5-minute return "
            "response to scheduled macro releases should be 2–3× larger in "
            "magnitude than during low-correlation (bottom quintile) periods."
        ),
        conditions=[
            "Daily cross-asset correlation estimates (rolling 21-day)",
            "High-frequency event-study data around scheduled macro releases",
        ],
        falsifiers=[
            "News impact magnitude is constant across correlation regimes",
        ],
        confidence=Confidence.MEDIUM,
        tags=["correlation", "herding", "macro", "impact", "Ising", "critical"],
    ),

    Analogy(
        name="emergence_sentiment_regime",
        source_domain=Domain.COMPLEX_SYSTEMS,
        description=(
            "Emergent properties arise from local interactions between simple agents "
            "and cannot be predicted from individual components alone. Whole-system "
            "behaviour is qualitatively different from any individual's behaviour. "
            "Market regimes emerge from aggregated micro-level heuristics."
        ),
        mappings=[
            AnalogyMapping("local interaction rules", "individual trader heuristics",
                           "Simple rules (momentum, stop-loss) at individual level"),
            AnalogyMapping("emergent phenomenon",     "market-wide bull/bear regime",
                           "Macroscopic regimes emerge from micro-level coordination"),
            AnalogyMapping("symmetry breaking",       "sentiment phase transition",
                           "Sudden consensus formation displaces previous equilibrium"),
        ],
        prediction=(
            "Aggregate sentiment indicators (put/call ratio, AAII survey, fund "
            "manager surveys) should lead changes in cross-sectional return dispersion "
            "by 1–4 weeks: rising dispersion preceding sentiment regime change as "
            "a new consensus emerges and old positions unwind heterogeneously."
        ),
        conditions=[
            "Weekly sentiment survey data and daily dispersion measures",
            "Minimum 10 years to capture multiple sentiment cycles",
        ],
        falsifiers=[
            "Sentiment does not Granger-cause changes in return dispersion",
        ],
        confidence=Confidence.LOW,
        tags=["sentiment", "regime", "emergence", "dispersion", "Granger"],
    ),

    # ── FLUID DYNAMICS (2) ───────────────────────────────────────────────────

    Analogy(
        name="turbulence_reynolds_number",
        source_domain=Domain.FLUID_DYNAMICS,
        description=(
            "Fluid transitions from laminar (smooth, predictable) to turbulent flow "
            "above a critical Reynolds number (Re = inertial / viscous forces). "
            "In turbulence, energy cascades from large scales to small via Kolmogorov's "
            "cascade, producing a characteristic -5/3 power spectrum."
        ),
        mappings=[
            AnalogyMapping("laminar flow",     "trending / directional market",
                           "Smooth, directional price movement with low noise"),
            AnalogyMapping("turbulence",       "choppy / mean-reverting market",
                           "Random, multi-directional price movement"),
            AnalogyMapping("Reynolds number",  "order flow momentum / liquidity ratio",
                           "High momentum relative to market-making capacity → turbulence"),
            AnalogyMapping("energy cascade",   "volatility across time scales",
                           "Large moves cascade into smaller-scale volatility bursts"),
        ],
        prediction=(
            "A 'market Reynolds number' proxy (ratio of absolute net order flow to "
            "two-sided quoted depth) should predict the transition from trending to "
            "mean-reverting intraday behaviour: high Re → mean-reversion, low Re → trend."
        ),
        conditions=[
            "Level-2 order book data for depth estimation",
            "High-frequency trade data for order flow computation",
        ],
        falsifiers=[
            "Order flow / depth ratio does not predict intraday momentum vs. MR regime",
        ],
        confidence=Confidence.MEDIUM,
        tags=["microstructure", "order flow", "liquidity", "regime", "turbulence"],
    ),

    Analogy(
        name="karman_vortex_index_rebalancing",
        source_domain=Domain.FLUID_DYNAMICS,
        description=(
            "Flow past a bluff body alternately sheds vortices (Kármán vortex street) "
            "at a frequency proportional to flow speed and inversely proportional to "
            "body size (Strouhal number). The oscillation is periodic and predictable "
            "given flow conditions."
        ),
        mappings=[
            AnalogyMapping("bluff body",         "large passive index fund",
                           "Mechanical rebalancing creates predictable periodic order flow"),
            AnalogyMapping("vortex shedding",    "index rebalancing front-running",
                           "Predictable buy/sell pressure creates recurrent oscillations"),
            AnalogyMapping("Strouhal frequency", "rebalancing frequency × recent drift",
                           "More drift + more frequent rebalance → more predictable trade"),
        ],
        prediction=(
            "Stocks with high weight in frequently rebalanced indices should exhibit "
            "predictable price pressure around rebalance dates; magnitude should scale "
            "with the product of index weight and recent weight drift from target."
        ),
        conditions=[
            "Index constituent weights and published rebalance schedules",
            "Daily price and volume data around rebalance implementation dates",
        ],
        falsifiers=[
            "Rebalancing pressure is fully anticipated and pre-absorbed 5+ days prior",
        ],
        confidence=Confidence.HIGH,
        tags=["index rebalancing", "front-running", "passive investing", "predictable flow"],
    ),

    # ── MATERIALS SCIENCE (3) ────────────────────────────────────────────────

    Analogy(
        name="fatigue_fracture_systemic_risk",
        source_domain=Domain.MATERIALS_SCIENCE,
        description=(
            "Materials fail under cyclic stress below ultimate tensile strength: "
            "fatigue damage accumulates subcritically until a crack initiates at a "
            "defect site and propagates catastrophically (Paris Law). The number of "
            "cycles to failure decreases non-linearly with stress amplitude."
        ),
        mappings=[
            AnalogyMapping("cyclic stress",       "recurring market stress episodes",
                           "Repeated non-fatal drawdowns weaken systemic resilience"),
            AnalogyMapping("fatigue accumulation","rising aggregate leverage and fragility",
                           "Each stress event increases latent vulnerability"),
            AnalogyMapping("pre-existing defect", "concentrated/fragile balance sheet",
                           "Weakness concentrates at the most leveraged participant"),
            AnalogyMapping("crack propagation",   "contagion through balance sheets",
                           "Failure propagates through interconnected institutions"),
        ],
        prediction=(
            "A 'market fatigue index' (count of ≥2σ drawdown days in prior 12 months + "
            "CDS spread percentile + VIX term structure slope inversion flag) should "
            "predict 6-month forward systemic crisis probability with positive AUROC."
        ),
        conditions=[
            "Long time series including multiple crisis episodes (≥3 major crises)",
            "Investment-grade CDS index and VIX term structure data",
        ],
        falsifiers=[
            "Fatigue index has AUROC ≤ 0.50 out-of-sample for crisis prediction",
        ],
        confidence=Confidence.MEDIUM,
        tags=["systemic risk", "crisis prediction", "leverage", "CDS", "fragility", "fatigue"],
    ),

    Analogy(
        name="crystallisation_nucleation_correction",
        source_domain=Domain.MATERIALS_SCIENCE,
        description=(
            "Supercooled liquids remain amorphous below equilibrium freezing point "
            "until nucleation sites catalyse rapid crystallisation. The waiting time "
            "is stochastic, but the density of nucleation sites and degree of "
            "undercooling determine the expected crystallisation speed."
        ),
        mappings=[
            AnalogyMapping("supercooled liquid",    "overvalued but stable market",
                           "Prices above fundamental value sustained by momentum"),
            AnalogyMapping("nucleation site",       "catalyst event",
                           "Earnings miss, geopolitical shock, or Fed surprise"),
            AnalogyMapping("rapid crystallisation", "rapid repricing cascade",
                           "Once one sector corrects, correlations spike and others follow"),
            AnalogyMapping("degree of undercooling","magnitude of overvaluation",
                           "More overvalued → faster and larger correction once triggered"),
        ],
        prediction=(
            "Post-catalyst correction speed (% decline per day in first 5 days) should "
            "be positively correlated with both (a) degree of prior overvaluation "
            "(forward P/E z-score) and (b) hedge fund crowding (factor exposure concentration). "
            "Their interaction term should be the strongest predictor."
        ),
        conditions=[
            "Cross-sectional valuation data at monthly frequency",
            "Hedge fund 13-F position data as crowding proxy",
            "Event-study framework with identified catalyst dates",
        ],
        falsifiers=[
            "Correction speed is independent of prior valuation and crowding level",
        ],
        confidence=Confidence.MEDIUM,
        tags=["correction", "crowding", "valuation", "catalyst", "cascade", "nucleation"],
    ),

    Analogy(
        name="latent_heat_consolidation",
        source_domain=Domain.MATERIALS_SCIENCE,
        description=(
            "First-order phase transitions involve latent heat: temperature stalls "
            "at the transition boundary while structural reorganisation absorbs energy. "
            "Apparent inactivity masks deep internal restructuring. The transition "
            "completes suddenly once reorganisation is complete."
        ),
        mappings=[
            AnalogyMapping("latent heat",         "price consolidation / coiling",
                           "Price stalls while investor positioning reorganises beneath"),
            AnalogyMapping("phase boundary",       "technical consolidation zone",
                           "Range where buying and selling are in temporary equilibrium"),
            AnalogyMapping("entropy increase",     "volatility-of-volatility spike",
                           "Structural disorder increases during the transition phase"),
            AnalogyMapping("completion of transition","breakout with follow-through",
                           "Sudden directional resolution after the coiling period"),
        ],
        prediction=(
            "Periods combining (a) low 21-day realised volatility AND (b) elevated "
            "VVIX (VIX of VIX) should predict a subsequent directional breakout with "
            "above-average magnitude and momentum follow-through over the next 10 days."
        ),
        conditions=[
            "Daily VIX, VVIX, and realised volatility data",
            "Definition of breakout: ≥1.5σ daily move within 10 trading days",
        ],
        falsifiers=[
            "Low-realised-vol / high-VVIX periods show no excess breakout probability",
        ],
        confidence=Confidence.HIGH,
        tags=["breakout", "VIX", "VVIX", "consolidation", "volatility regime", "latent heat"],
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# AnalogyEngine
# ──────────────────────────────────────────────────────────────────────────────

# Domain-specific novelty scores (how far from pure finance)
_DOMAIN_NOVELTY: dict[str, float] = {
    "ecology":            0.85,
    "game_theory":        0.50,
    "information_theory": 0.60,
    "biology":            0.80,
    "complex_systems":    0.70,
    "fluid_dynamics":     0.90,
    "materials_science":  0.88,
}


class AnalogyEngine:
    """
    Core engine for generating trading hypotheses via cross-domain analogies.

    Usage
    -----
        engine = AnalogyEngine(seed=42)
        hypotheses = engine.generate_hypothesis(n=5)
        for h in hypotheses:
            print(h.summary())

        # Targeted search
        results = engine.cross_domain_search("volatility crash prediction")
        for analogy, score in results:
            print(analogy.name, score)
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._library: list[Analogy] = list(ANALOGY_LIBRARY)
        self._hypothesis_counter = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def select_analogies(
        self,
        domains: list[Domain] | None = None,
        tags: list[str] | None = None,
        min_confidence: Confidence | None = None,
        n: int = 5,
        shuffle: bool = True,
    ) -> list[Analogy]:
        """
        Filter and return analogies matching given criteria.

        Parameters
        ----------
        domains         : Restrict to these source domains (None = all)
        tags            : Require all listed tags to be present
        min_confidence  : Minimum confidence level
        n               : Maximum number of analogies to return
        shuffle         : Randomise the order before selecting
        """
        conf_rank = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
        candidates = list(self._library)

        if domains:
            candidates = [a for a in candidates if a.source_domain in domains]
        if tags:
            candidates = [a for a in candidates
                          if all(t in a.tags for t in tags)]
        if min_confidence:
            min_rank = conf_rank[min_confidence]
            candidates = [a for a in candidates
                          if conf_rank[a.confidence] >= min_rank]

        if shuffle:
            self._rng.shuffle(candidates)
        return candidates[:n]

    def generate_hypothesis(
        self,
        n: int = 3,
        domains: list[Domain] | None = None,
        combine: bool = False,
        min_confidence: Confidence | None = None,
    ) -> list[HypothesisCandidate]:
        """
        Generate n HypothesisCandidate objects, optionally including one
        compound hypothesis formed by combining two analogies.

        Parameters
        ----------
        n               : Number of single-analogy hypotheses to generate
        domains         : Restrict source domains
        combine         : If True, append a cross-analogy compound hypothesis
        min_confidence  : Filter by minimum confidence
        """
        analogies = self.select_analogies(
            domains=domains, min_confidence=min_confidence, n=n + 2
        )
        hypotheses: list[HypothesisCandidate] = [
            self._instantiate_hypothesis(a) for a in analogies[:n]
        ]

        if combine and len(analogies) >= 2:
            hypotheses.append(self._combine_analogies(analogies[0], analogies[1]))

        return hypotheses

    def cross_domain_search(
        self,
        target_phenomenon: str,
        top_k: int = 3,
    ) -> list[tuple[Analogy, float]]:
        """
        Find analogies relevant to a described market phenomenon via TF-like scoring.

        Returns
        -------
        List of (Analogy, relevance_score) sorted descending.
        """
        query_tokens = set(target_phenomenon.lower().replace(",", " ").split())
        scored: list[tuple[Analogy, float]] = []

        for analogy in self._library:
            corpus: set[str] = set()
            corpus.update(analogy.name.lower().split("_"))
            corpus.update(analogy.description.lower().split())
            corpus.update(analogy.prediction.lower().split())
            for tag in analogy.tags:
                corpus.update(tag.lower().split())
            for m in analogy.mappings:
                corpus.update(m.target_concept.lower().split())
                corpus.update(m.source_concept.lower().split())

            hits = len(query_tokens & corpus)
            if hits:
                score = hits / max(len(query_tokens), 1)
                scored.append((analogy, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_all_domains(self) -> list[Domain]:
        """Return sorted list of all source domains in the library."""
        return sorted({a.source_domain for a in self._library}, key=lambda d: d.value)

    def get_all_tags(self) -> list[str]:
        """Return sorted list of all tags across the library."""
        tags: set[str] = set()
        for a in self._library:
            tags.update(a.tags)
        return sorted(tags)

    def library_summary(self) -> str:
        """Human-readable summary of the analogy library."""
        lines = [f"AnalogyEngine  ({len(self._library)} analogies)\n"]
        by_domain: dict[str, list[str]] = {}
        for a in self._library:
            by_domain.setdefault(a.source_domain.value, []).append(a.name)
        for domain, names in sorted(by_domain.items()):
            lines.append(f"  {domain}  [{len(names)}]")
            for nm in names:
                lines.append(f"    • {nm}")
        return "\n".join(lines)

    def list_high_confidence(self) -> list[Analogy]:
        """Return all HIGH confidence analogies."""
        return [a for a in self._library if a.confidence == Confidence.HIGH]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _next_id(self) -> str:
        self._hypothesis_counter += 1
        return f"HYP-{self._hypothesis_counter:04d}"

    def _instantiate_hypothesis(self, analogy: Analogy) -> HypothesisCandidate:
        novelty = _DOMAIN_NOVELTY.get(analogy.source_domain.value, 0.5)
        signals = [
            f"[{m.source_concept}] → [{m.target_concept}]: {m.mapping_rationale}"
            for m in analogy.mappings
        ]
        universe, lookback = self._infer_universe_lookback(analogy)

        narrative = (
            f"Inspired by '{analogy.name}' from {analogy.source_domain.value}. "
            f"{analogy.description} "
            f"The key market mapping: "
            + "; ".join(
                f"'{m.source_concept}' ≈ '{m.target_concept}'"
                for m in analogy.mappings[:2]
            )
            + "."
        )

        return HypothesisCandidate(
            hypothesis_id=self._next_id(),
            source_domain=analogy.source_domain,
            target_domain=Domain.FINANCE,
            analogy_name=analogy.name,
            title=f"{analogy.name.replace('_', ' ').title()} Hypothesis",
            narrative=narrative,
            prediction=analogy.prediction,
            testable_conditions=analogy.conditions,
            falsifiers=analogy.falsifiers,
            suggested_signals=signals,
            suggested_universe=universe,
            suggested_lookback=lookback,
            confidence=analogy.confidence,
            novelty_score=round(novelty, 3),
            cross_domain_score=round(novelty, 3),
            provenance=[analogy.name],
        )

    def _combine_analogies(self, a1: Analogy, a2: Analogy) -> HypothesisCandidate:
        """Generate a compound hypothesis by fusing two source analogies."""
        conditions = list(dict.fromkeys(a1.conditions + a2.conditions))[:5]
        falsifiers = list(dict.fromkeys(a1.falsifiers + a2.falsifiers))[:4]
        signals = (
            [f"[{a1.source_domain.value}] {m.target_concept}" for m in a1.mappings[:2]]
            + [f"[{a2.source_domain.value}] {m.target_concept}" for m in a2.mappings[:2]]
        )
        n1 = _DOMAIN_NOVELTY.get(a1.source_domain.value, 0.5)
        n2 = _DOMAIN_NOVELTY.get(a2.source_domain.value, 0.5)
        combined_novelty = min(1.0, (n1 + n2) / 2 + 0.10)

        return HypothesisCandidate(
            hypothesis_id=self._next_id(),
            source_domain=a1.source_domain,
            target_domain=Domain.FINANCE,
            analogy_name=f"{a1.name} × {a2.name}",
            title=(
                f"Compound: {a1.name.replace('_',' ').title()} "
                f"meets {a2.name.replace('_',' ').title()}"
            ),
            narrative=(
                f"This compound hypothesis fuses two analogies: "
                f"'{a1.name}' ({a1.source_domain.value}) and "
                f"'{a2.name}' ({a2.source_domain.value}). "
                f"The first predicts: {a1.prediction[:140]}. "
                f"The second predicts: {a2.prediction[:140]}. "
                f"Together they imply a market where both mechanisms operate simultaneously, "
                f"potentially amplifying each other's effects."
            ),
            prediction=(
                f"Markets exhibiting conditions for both '{a1.name}' and '{a2.name}' "
                f"simultaneously should show amplified effects: "
                f"{a1.prediction[:100]}; AND ALSO {a2.prediction[:100]}."
            ),
            testable_conditions=conditions,
            falsifiers=falsifiers,
            suggested_signals=signals,
            suggested_universe="Broad cross-asset universe",
            suggested_lookback="10–20 years",
            confidence=Confidence.LOW,
            novelty_score=round(combined_novelty, 3),
            cross_domain_score=0.95,
            provenance=[a1.name, a2.name],
        )

    @staticmethod
    def _infer_universe_lookback(analogy: Analogy) -> tuple[str, str]:
        tags = set(analogy.tags)
        if "intraday" in tags:
            return "Liquid equities/futures (top 500 by ADV)", "2–5 years intraday"
        if any(t in tags for t in ("systemic risk", "contagion", "cascade")):
            return "Global equities + credit + rates (broad cross-asset)", "15–25 years"
        if any(t in tags for t in ("trend", "momentum", "AUM")):
            return "Diversified futures: equity indices, bonds, commodities, FX", "20+ years"
        if any(t in tags for t in ("microstructure", "VPIN", "order flow")):
            return "US equities with TAQ data", "3–5 years tick data"
        if any(t in tags for t in ("volatility", "crash", "SOC", "tail risk")):
            return "Major equity indices (SPX, NDX, EuroStoxx, Nikkei, FTSE)", "25+ years"
        if "index rebalancing" in tags:
            return "S&P 500, Russell 2000, MSCI ACWI constituents", "10+ years"
        if "entropy" in tags or "complexity" in tags:
            return "Liquid global equities (≥200 names)", "10–15 years daily"
        return "Global equities or diversified futures", "10–15 years"


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    engine = AnalogyEngine(seed=7)
    print(engine.library_summary())

    print("\n" + "=" * 72)
    print("Generating 3 hypotheses (ecology + complex systems, with combine=True)")
    print("=" * 72)
    hyps = engine.generate_hypothesis(
        n=3,
        domains=[Domain.ECOLOGY, Domain.COMPLEX_SYSTEMS],
        combine=True,
    )
    for h in hyps:
        print("\n" + h.summary())
        print("-" * 72)

    print("\n" + "=" * 72)
    print("High-confidence analogies:")
    print("=" * 72)
    for a in engine.list_high_confidence():
        print(f"  • [{a.source_domain.value:20s}] {a.name}")

    print("\n" + "=" * 72)
    print("Cross-domain search: 'volatility crash leverage prediction'")
    print("=" * 72)
    for analogy, score in engine.cross_domain_search(
        "volatility crash leverage prediction", top_k=4
    ):
        print(f"  [{score:.3f}] {analogy.name}  ({analogy.source_domain.value})")


if __name__ == "__main__":
    _demo()
