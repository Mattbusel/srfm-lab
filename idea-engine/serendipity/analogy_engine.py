"""
analogy_engine.py — Cross-Domain Analogy Library
=================================================
Provides a rich library of structural analogies between scientific
domains (thermodynamics, ecology, fluid dynamics, information theory,
network theory, game theory) and trading strategy concepts.

Each analogy connects a **source concept** in its domain to a
**target concept** in the BH trading strategy, with a suggested
**experiment** — the concrete parameter change or signal addition
that the analogy implies.

Classes
-------
    Analogy            — Data container for one analogy.
    AnalogyEngine      — Library lookup and analogy-to-experiment mapper.

Usage
-----
    engine   = AnalogyEngine()
    analogies = engine.find_analogies("entropy")
    for a in analogies:
        print(a.source_concept, "→", a.target_concept)
        experiment = engine.analogize_to_experiment(a)
        print("  Experiment:", experiment)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Analogy:
    """
    Represents a structural analogy between a scientific domain and trading.

    Attributes
    ----------
    domain : str
        Scientific domain (e.g. "thermodynamics").
    source_concept : str
        The concept in the source domain.
    target_concept : str
        The corresponding trading concept.
    description : str
        Narrative description of the structural similarity.
    experiment_hint : str
        Brief hint for an experiment that tests this analogy.
    confidence : float
        Prior confidence that the analogy is actionable (0–1).
    tags : List[str]
        Strategy component tags (entry_signal, risk_management, etc.).
    """

    domain:          str
    source_concept:  str
    target_concept:  str
    description:     str
    experiment_hint: str
    confidence:      float         = 0.6
    tags:            List[str]     = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Analogy(domain={self.domain!r}, "
            f"src={self.source_concept!r} -> tgt={self.target_concept!r})"
        )


# ---------------------------------------------------------------------------
# Analogy library — 50+ entries across 6 domains
# ---------------------------------------------------------------------------

ANALOGY_LIBRARY: List[Analogy] = [

    # ===================================================================
    # THERMODYNAMICS
    # ===================================================================
    Analogy(
        domain="thermodynamics",
        source_concept="entropy",
        target_concept="market disorder / bid-ask spread",
        description=(
            "In thermodynamics, entropy measures the number of microstates "
            "available to a system. In markets, high entropy corresponds to "
            "high disorder — wide spreads, low liquidity, erratic price paths. "
            "Shannon entropy of return distributions can quantify this."
        ),
        experiment_hint="Add a Shannon-entropy filter: only trade when entropy < threshold.",
        confidence=0.72,
        tags=["regime_filter", "entry_signal"],
    ),
    Analogy(
        domain="thermodynamics",
        source_concept="temperature",
        target_concept="realised volatility",
        description=(
            "Temperature measures average kinetic energy of particles. "
            "Volatility measures average kinetic energy of price moves. "
            "High-temperature regimes suggest cautious position sizing, "
            "just as thermal expansion changes material properties."
        ),
        experiment_hint="Scale position size inversely with sqrt(realised_vol).",
        confidence=0.80,
        tags=["position_sizing", "risk_management"],
    ),
    Analogy(
        domain="thermodynamics",
        source_concept="phase transition",
        target_concept="market regime change",
        description=(
            "Phase transitions (solid/liquid/gas) occur at critical temperatures "
            "where the system's behaviour qualitatively changes. Markets similarly "
            "transition between trending, mean-reverting, and crisis regimes. "
            "The order parameter analogy is trend strength (e.g., ADX)."
        ),
        experiment_hint="Detect regime transitions with a 2-state HMM on vol and returns.",
        confidence=0.83,
        tags=["regime_filter"],
    ),
    Analogy(
        domain="thermodynamics",
        source_concept="heat diffusion",
        target_concept="momentum diffusion across assets",
        description=(
            "Heat diffuses from hot to cold regions. Momentum diffuses from "
            "high-momentum to correlated assets (cross-asset momentum spillover). "
            "The diffusion equation suggests lagged cross-asset signals."
        ),
        experiment_hint="Test lagged BTC momentum as predictor for ETH entry signal.",
        confidence=0.70,
        tags=["entry_signal"],
    ),
    Analogy(
        domain="thermodynamics",
        source_concept="Maxwell-Boltzmann distribution",
        target_concept="return distribution fat tails",
        description=(
            "Particle speed follows Maxwell-Boltzmann; returns follow a "
            "fat-tailed distribution. Modelling returns as Lévy-stable "
            "improves VaR estimates and stop-loss placement."
        ),
        experiment_hint="Fit Student-t distribution to returns; use t-VaR for stops.",
        confidence=0.75,
        tags=["risk_management"],
    ),
    Analogy(
        domain="thermodynamics",
        source_concept="Carnot efficiency",
        target_concept="maximum achievable alpha",
        description=(
            "Carnot efficiency sets an upper bound on work extractable from "
            "a heat engine. Similarly, market efficiency limits the alpha "
            "extractable from any signal before transaction costs erode it."
        ),
        experiment_hint="Estimate theoretical alpha decay rate; set holding period accordingly.",
        confidence=0.65,
        tags=["exit_rule"],
    ),
    Analogy(
        domain="thermodynamics",
        source_concept="free energy minimisation",
        target_concept="portfolio optimisation as energy minimisation",
        description=(
            "Systems evolve to minimise free energy (Helmholtz / Gibbs). "
            "Portfolio weights that minimise tracking error can be framed as "
            "free-energy minimisation under a Lagrangian constraint."
        ),
        experiment_hint="Use energy-based objective function for weight optimisation.",
        confidence=0.60,
        tags=["position_sizing"],
    ),
    Analogy(
        domain="thermodynamics",
        source_concept="Brownian motion",
        target_concept="random walk / Ornstein-Uhlenbeck price process",
        description=(
            "Brownian motion is the thermal jiggling of particles. The OU "
            "process is mean-reverting Brownian motion — the standard model "
            "for spread trading, pairs trading, and stat arb."
        ),
        experiment_hint="Fit OU parameters (theta, mu, sigma) daily; trade when z-score > 2.",
        confidence=0.88,
        tags=["entry_signal", "exit_rule"],
    ),

    # ===================================================================
    # ECOLOGY
    # ===================================================================
    Analogy(
        domain="ecology",
        source_concept="predator-prey dynamics (Lotka-Volterra)",
        target_concept="market maker vs. informed trader",
        description=(
            "Predator-prey cycles (rabbits and foxes) model the interaction "
            "between market makers (prey that provide liquidity) and informed "
            "traders (predators that consume it). Spread dynamics and order "
            "flow toxicity follow Lotka-Volterra-like oscillations."
        ),
        experiment_hint="Model spread oscillation; enter when market-maker activity peaks.",
        confidence=0.68,
        tags=["entry_signal", "regime_filter"],
    ),
    Analogy(
        domain="ecology",
        source_concept="niche / competitive exclusion",
        target_concept="strategy edge uniqueness",
        description=(
            "Two species cannot occupy the same ecological niche indefinitely. "
            "Two strategies with identical signals will compete for the same "
            "alpha until one 'goes extinct'. Strategy differentiation is key "
            "to long-term survival."
        ),
        experiment_hint="Add unique signal feature (e.g., on-chain data) to differentiate.",
        confidence=0.60,
        tags=["entry_signal"],
    ),
    Analogy(
        domain="ecology",
        source_concept="extinction / ruin probability",
        target_concept="account ruin / maximum drawdown",
        description=(
            "Species go extinct when population falls below a critical threshold. "
            "Trading accounts face ruin when drawdown exceeds capital reserves. "
            "Kelly criterion is the ecological 'carrying capacity' analogy."
        ),
        experiment_hint="Set hard stop at 25% drawdown, equivalent to extinction threshold.",
        confidence=0.85,
        tags=["risk_management"],
    ),
    Analogy(
        domain="ecology",
        source_concept="adaptive radiation",
        target_concept="strategy diversification across timeframes",
        description=(
            "After a mass extinction, surviving species rapidly diversify into "
            "empty niches. After a strategy drawdown, consider diversifying "
            "into uncorrelated timeframes to exploit unfilled niches."
        ),
        experiment_hint="Add a 4h timeframe variant to complement the primary 1h strategy.",
        confidence=0.65,
        tags=["regime_filter", "entry_signal"],
    ),
    Analogy(
        domain="ecology",
        source_concept="mutualism / symbiosis",
        target_concept="signal correlation and ensemble",
        description=(
            "Mutualistic species benefit each other (clownfish and anemone). "
            "Signals that are slightly negatively correlated but individually "
            "profitable form a symbiotic ensemble that reduces portfolio variance."
        ),
        experiment_hint="Build ensemble of momentum + mean-reversion signals with inverse weights.",
        confidence=0.74,
        tags=["position_sizing", "entry_signal"],
    ),
    Analogy(
        domain="ecology",
        source_concept="invasive species",
        target_concept="new market participant disrupting alpha",
        description=(
            "Invasive species disrupt established ecosystems. HFT entry, "
            "new institutional crypto products, or quant crowding can invade "
            "and destroy existing alpha sources rapidly."
        ),
        experiment_hint="Monitor signal autocorrelation decay; reduce exposure when alpha erodes.",
        confidence=0.70,
        tags=["risk_management", "regime_filter"],
    ),
    Analogy(
        domain="ecology",
        source_concept="carrying capacity",
        target_concept="market capacity / position size limit",
        description=(
            "A habitat's carrying capacity is the maximum sustainable population. "
            "Markets have a capacity for any given strategy — beyond which "
            "slippage and market impact consume all profit."
        ),
        experiment_hint="Estimate capacity via market impact model; cap position size at 1% ADV.",
        confidence=0.80,
        tags=["position_sizing"],
    ),
    Analogy(
        domain="ecology",
        source_concept="camouflage / mimicry",
        target_concept="order disguising / iceberg orders",
        description=(
            "Prey animals use camouflage to avoid detection. Large orders "
            "are disguised through iceberg orders and TWAP/VWAP execution "
            "to avoid adverse price impact."
        ),
        experiment_hint="Implement TWAP execution for positions > 0.5% of hourly volume.",
        confidence=0.72,
        tags=["exit_rule"],
    ),

    # ===================================================================
    # INFORMATION THEORY
    # ===================================================================
    Analogy(
        domain="information_theory",
        source_concept="Shannon entropy",
        target_concept="signal uncertainty / market predictability",
        description=(
            "Shannon entropy H = -Σ p log p measures information content. "
            "Low-entropy return distributions are more predictable; "
            "high-entropy distributions approach noise. H can be used as "
            "a regime filter or a signal quality metric."
        ),
        experiment_hint="Compute rolling 20-bar entropy; trade only when H < H_median.",
        confidence=0.78,
        tags=["regime_filter", "entry_signal"],
    ),
    Analogy(
        domain="information_theory",
        source_concept="mutual information",
        target_concept="signal correlation / causal discovery",
        description=(
            "Mutual information I(X;Y) captures non-linear dependencies. "
            "Replacing Pearson correlation with mutual information for signal "
            "selection detects non-linear alpha sources that correlation misses."
        ),
        experiment_hint="Rank signals by mutual information with forward returns; use top-3.",
        confidence=0.82,
        tags=["entry_signal"],
    ),
    Analogy(
        domain="information_theory",
        source_concept="channel capacity (Shannon-Hartley)",
        target_concept="maximum information ratio",
        description=(
            "Channel capacity is the max information transmissible per unit "
            "time. The 'alpha channel' from signal to trade has limited "
            "capacity determined by signal-to-noise ratio (IC² × N)."
        ),
        experiment_hint="Use IC (information coefficient) to size positions via signal strength.",
        confidence=0.75,
        tags=["position_sizing"],
    ),
    Analogy(
        domain="information_theory",
        source_concept="Kolmogorov complexity",
        target_concept="strategy complexity / overfitting risk",
        description=(
            "Minimum description length principle: prefer the simplest model "
            "that explains data. Strategies with fewer parameters generalise "
            "better out-of-sample (Occam's razor for alpha)."
        ),
        experiment_hint="Penalise strategy fitness with parameter count; limit to 5 free params.",
        confidence=0.77,
        tags=["risk_management"],
    ),
    Analogy(
        domain="information_theory",
        source_concept="data compression",
        target_concept="dimensionality reduction of feature space",
        description=(
            "Compression removes redundancy while preserving information. "
            "PCA / ICA on a feature matrix removes correlated signals, "
            "leaving orthogonal factors that improve signal quality."
        ),
        experiment_hint="Apply PCA to 20 features; use first 3 components as entry signal.",
        confidence=0.73,
        tags=["entry_signal"],
    ),
    Analogy(
        domain="information_theory",
        source_concept="error-correcting codes",
        target_concept="ensemble voting / signal redundancy",
        description=(
            "Error-correcting codes add redundancy to tolerate noisy channels. "
            "Requiring 2-of-3 signal agreement before entry trades reliability "
            "for frequency, analogously correcting for 'noisy' individual signals."
        ),
        experiment_hint="Require 2/3 of {momentum, OU, vol-breakout} to agree before entry.",
        confidence=0.79,
        tags=["entry_signal"],
    ),

    # ===================================================================
    # NETWORK THEORY
    # ===================================================================
    Analogy(
        domain="network_theory",
        source_concept="centrality (PageRank / eigenvector)",
        target_concept="asset influence / systemic importance",
        description=(
            "Central nodes in a network have outsized influence. BTC is the "
            "'PageRank' hub of the crypto asset network; its moves propagate "
            "to altcoins with lags proportional to edge weights (correlations)."
        ),
        experiment_hint="Use BTC's eigenvector centrality change as a leading indicator for alts.",
        confidence=0.77,
        tags=["entry_signal", "regime_filter"],
    ),
    Analogy(
        domain="network_theory",
        source_concept="clustering coefficient",
        target_concept="correlation regime / crowding",
        description=(
            "High clustering means a network's nodes are densely interconnected. "
            "High average pairwise correlation means markets are 'clustered' — "
            "individual diversification breaks down in crisis regimes."
        ),
        experiment_hint="Gate diversification based on clustering coefficient of return matrix.",
        confidence=0.75,
        tags=["regime_filter", "risk_management"],
    ),
    Analogy(
        domain="network_theory",
        source_concept="small-world network",
        target_concept="information diffusion speed",
        description=(
            "In small-world networks, information travels quickly across short "
            "path lengths. Crypto markets behave as small-world networks where "
            "news propagates in minutes — decay of signal edge is faster."
        ),
        experiment_hint="Reduce holding period to < 4h to capture fast information diffusion.",
        confidence=0.68,
        tags=["exit_rule"],
    ),
    Analogy(
        domain="network_theory",
        source_concept="percolation threshold",
        target_concept="correlation contagion / flash crash",
        description=(
            "Percolation theory identifies the threshold at which a network "
            "becomes connected (giant component). Correlation contagion "
            "during flash crashes follows percolation dynamics."
        ),
        experiment_hint="Monitor percolation metric; reduce exposure at 80% correlation threshold.",
        confidence=0.72,
        tags=["risk_management", "regime_filter"],
    ),
    Analogy(
        domain="network_theory",
        source_concept="preferential attachment (scale-free networks)",
        target_concept="momentum / rich-get-richer in trending assets",
        description=(
            "In scale-free networks, popular nodes attract more connections. "
            "Trending assets attract more capital flows — momentum is the "
            "financial analogue of preferential attachment."
        ),
        experiment_hint="Weight entry size by recent capital flow (volume trend) proxy.",
        confidence=0.73,
        tags=["position_sizing", "entry_signal"],
    ),

    # ===================================================================
    # FLUID DYNAMICS
    # ===================================================================
    Analogy(
        domain="fluid_dynamics",
        source_concept="turbulence",
        target_concept="volatility spike / market stress",
        description=(
            "Turbulent flow is chaotic with eddies at all scales. Market "
            "turbulence (vol spikes, gap openings) shares scale-free "
            "statistical properties with fluid turbulence (Kolmogorov scaling)."
        ),
        experiment_hint="Use turbulence indicator: pause trading when vol > 3 × EWMA(vol).",
        confidence=0.80,
        tags=["regime_filter", "risk_management"],
    ),
    Analogy(
        domain="fluid_dynamics",
        source_concept="laminar flow",
        target_concept="trending / low-volatility market",
        description=(
            "Laminar flow is smooth and predictable. Trending markets with "
            "stable low volatility (low VIX equivalent) exhibit laminar "
            "properties — momentum strategies thrive here."
        ),
        experiment_hint="Increase leverage to 1.5× when ADX > 30 AND vol < vol_median.",
        confidence=0.78,
        tags=["position_sizing", "regime_filter"],
    ),
    Analogy(
        domain="fluid_dynamics",
        source_concept="Reynolds number",
        target_concept="momentum threshold (Re = inertia / viscosity)",
        description=(
            "The Reynolds number Re = ρvL/μ predicts whether flow will be "
            "laminar or turbulent. A 'momentum Reynolds number' can be defined "
            "as (recent_return / realised_vol) — a normalised momentum signal."
        ),
        experiment_hint="Use (20-bar return) / (20-bar vol) as a normalised momentum entry signal.",
        confidence=0.85,
        tags=["entry_signal"],
    ),
    Analogy(
        domain="fluid_dynamics",
        source_concept="Bernoulli principle (pressure drops in fast flow)",
        target_concept="order book thinning during price spikes",
        description=(
            "Bernoulli: faster flow → lower pressure. Fast price moves thin "
            "the order book, widening spreads. Use order-book depth as a "
            "secondary signal; exit when book thins significantly."
        ),
        experiment_hint="Add order-book depth filter: exit if top-5-level depth < 50% of median.",
        confidence=0.71,
        tags=["exit_rule", "risk_management"],
    ),
    Analogy(
        domain="fluid_dynamics",
        source_concept="vortex / eddy",
        target_concept="mean-reversion after impulse",
        description=(
            "Eddies are circular flows that arise downstream of obstacles. "
            "After a sharp price impulse (high momentum), mean-reversion "
            "eddies form as price circles back to equilibrium."
        ),
        experiment_hint="After |return| > 2σ, fade the move on the next bar with OU signal.",
        confidence=0.74,
        tags=["entry_signal", "exit_rule"],
    ),
    Analogy(
        domain="fluid_dynamics",
        source_concept="hydraulic jump",
        target_concept="support/resistance level break",
        description=(
            "A hydraulic jump is a rapid transition from supercritical to "
            "subcritical flow. Price breaking through a key support/resistance "
            "level is the trading analogue — energy changes state abruptly."
        ),
        experiment_hint="Add S/R breakout confirmation: trade only on volume > 2× average.",
        confidence=0.70,
        tags=["entry_signal"],
    ),
    Analogy(
        domain="fluid_dynamics",
        source_concept="viscosity",
        target_concept="market impact / slippage",
        description=(
            "Viscosity resists flow. Market impact is the 'viscosity' of the "
            "trading environment — it resists rapid position changes and "
            "creates additional cost proportional to order size."
        ),
        experiment_hint="Model market impact as quadratic in order size; adjust sizing accordingly.",
        confidence=0.76,
        tags=["position_sizing"],
    ),

    # ===================================================================
    # GAME THEORY
    # ===================================================================
    Analogy(
        domain="game_theory",
        source_concept="Nash equilibrium",
        target_concept="crowded strategy equilibrium",
        description=(
            "A Nash equilibrium is a strategy profile where no player can "
            "benefit by unilaterally changing their strategy. When all quants "
            "adopt the same signals, returns converge to Nash equilibrium "
            "(crowding), reducing alpha to near zero."
        ),
        experiment_hint="Monitor signal correlation across competing strategies; rotate when crowded.",
        confidence=0.72,
        tags=["regime_filter", "entry_signal"],
    ),
    Analogy(
        domain="game_theory",
        source_concept="minimax strategy",
        target_concept="worst-case risk budgeting",
        description=(
            "Minimax minimises the worst possible outcome. Risk budgeting "
            "that minimises maximum possible drawdown is the trading equivalent "
            "— useful under model uncertainty."
        ),
        experiment_hint="Use minimax regret criterion for position sizing under vol uncertainty.",
        confidence=0.73,
        tags=["risk_management", "position_sizing"],
    ),
    Analogy(
        domain="game_theory",
        source_concept="prisoner's dilemma",
        target_concept="coordination failure in market exits",
        description=(
            "In a prisoner's dilemma, individually rational choices lead to "
            "collectively inferior outcomes. During market panics, all actors "
            "simultaneously reduce risk — creating cascade selling."
        ),
        experiment_hint="Fade liquidation cascades: buy when realised vol exceeds 3× EWMA.",
        confidence=0.69,
        tags=["entry_signal", "regime_filter"],
    ),
    Analogy(
        domain="game_theory",
        source_concept="repeated games / reputation",
        target_concept="strategy consistency and market impact",
        description=(
            "In repeated games, cooperation emerges through reputation. "
            "Predictable order patterns invite front-running. Randomising "
            "entry timing reduces adverse selection from HFTs."
        ),
        experiment_hint="Randomise order timing by ±30s; split orders into 3 tranches.",
        confidence=0.74,
        tags=["entry_signal", "exit_rule"],
    ),
    Analogy(
        domain="game_theory",
        source_concept="auction theory",
        target_concept="limit order book dynamics",
        description=(
            "Markets are continuous double auctions. Auction theory explains "
            "bid-ask spread determination, price discovery, and the value "
            "of information. Vickrey auction analogy → true value estimation."
        ),
        experiment_hint="Estimate fair value via order-book imbalance; post limit orders at fair value.",
        confidence=0.76,
        tags=["entry_signal"],
    ),
    Analogy(
        domain="game_theory",
        source_concept="Stackelberg leader-follower",
        target_concept="lead-lag relationships between assets",
        description=(
            "In Stackelberg competition, a leader commits first and followers "
            "react. BTC leads altcoin markets — use BTC as the Stackelberg "
            "leader signal with altcoins as followers."
        ),
        experiment_hint="Use BTC 15-min return as leading signal for ETH entry on next bar.",
        confidence=0.81,
        tags=["entry_signal"],
    ),
]

# ---------------------------------------------------------------------------
# Concept → synonym mapping for fuzzy search
# ---------------------------------------------------------------------------

CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "entropy":      ["disorder", "uncertainty", "randomness", "noise", "chaos"],
    "temperature":  ["volatility", "vol", "variance", "heat", "energy"],
    "phase":        ["regime", "state", "transition", "break", "change"],
    "momentum":     ["trend", "inertia", "drift", "velocity", "speed"],
    "diffusion":    ["spread", "propagation", "contagion", "diffuse"],
    "turbulence":   ["spike", "stress", "crisis", "shock", "jump"],
    "predator":     ["hft", "informed trader", "smart money", "whale"],
    "prey":         ["market maker", "liquidity provider", "passive"],
    "niche":        ["edge", "alpha", "advantage", "signal"],
    "extinction":   ["ruin", "drawdown", "bankruptcy", "blowup"],
    "centrality":   ["influence", "importance", "hub", "dominant"],
    "clustering":   ["correlation", "crowding", "comovement", "beta"],
    "entropy_it":   ["information", "mutual information", "ic", "predictability"],
    "capacity":     ["throughput", "maximum", "limit", "constraint"],
    "viscosity":    ["impact", "friction", "slippage", "cost"],
    "reynolds":     ["threshold", "normalised", "regime", "transition"],
    "nash":         ["equilibrium", "crowded", "efficient", "saturated"],
    "minimax":      ["worst case", "conservative", "robust", "safe"],
    "auction":      ["order book", "bid ask", "spread", "price discovery"],
    "leader":       ["lead lag", "btc", "dominant", "causal"],
}


# ---------------------------------------------------------------------------
# AnalogyEngine
# ---------------------------------------------------------------------------

class AnalogyEngine:
    """
    Lookup engine for the cross-domain analogy library.

    Methods
    -------
    find_analogies(concept) → List[Analogy]
        Search the library for analogies matching *concept*.
    find_by_domain(domain) → List[Analogy]
        Return all analogies for a given domain.
    find_by_component(component) → List[Analogy]
        Return analogies tagged with a specific strategy component.
    analogize_to_experiment(analogy) → dict
        Convert an analogy to a concrete experiment dict.
    random_analogy(domain=None) → Analogy
        Return a random analogy (optionally restricted to domain).
    """

    SUPPORTED_DOMAINS: List[str] = [
        "thermodynamics",
        "fluid_dynamics",
        "ecology",
        "game_theory",
        "information_theory",
        "network_theory",
    ]

    def __init__(self) -> None:
        self._library: List[Analogy] = list(ANALOGY_LIBRARY)

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def find_analogies(self, concept: str) -> List[Analogy]:
        """
        Search for analogies whose source_concept, target_concept, or
        description contains *concept* (or its synonyms).

        Parameters
        ----------
        concept : str
            Search term (case-insensitive).

        Returns
        -------
        List[Analogy]
            Sorted by confidence descending.
        """
        concept_l = concept.lower()
        # Expand with synonyms
        search_terms = {concept_l}
        for key, syns in CONCEPT_SYNONYMS.items():
            if concept_l in key or any(concept_l in s for s in syns):
                search_terms.update(syns)
                search_terms.add(key)

        results: List[Analogy] = []
        for analogy in self._library:
            haystack = (
                analogy.source_concept + " " +
                analogy.target_concept + " " +
                analogy.description + " " +
                analogy.domain
            ).lower()
            if any(term in haystack for term in search_terms):
                results.append(analogy)

        results.sort(key=lambda a: a.confidence, reverse=True)
        return results

    def find_by_domain(self, domain: str) -> List[Analogy]:
        """
        Return all analogies for *domain*.

        Parameters
        ----------
        domain : str
            One of SUPPORTED_DOMAINS.

        Returns
        -------
        List[Analogy]
        """
        domain_l = domain.lower().replace(" ", "_")
        return [a for a in self._library if a.domain.lower() == domain_l]

    def find_by_component(self, component: str) -> List[Analogy]:
        """
        Return analogies tagged with strategy component *component*.

        Parameters
        ----------
        component : str

        Returns
        -------
        List[Analogy]
        """
        return [a for a in self._library if component in a.tags]

    def all_domains(self) -> List[str]:
        """Return the sorted list of unique domain names in the library."""
        return sorted({a.domain for a in self._library})

    def stats(self) -> dict:
        """Return summary statistics about the analogy library."""
        from collections import Counter
        domain_counts = Counter(a.domain for a in self._library)
        component_counts: Counter = Counter()
        for a in self._library:
            for tag in a.tags:
                component_counts[tag] += 1
        return {
            "total":           len(self._library),
            "domains":         dict(domain_counts),
            "components":      dict(component_counts),
            "avg_confidence":  round(
                sum(a.confidence for a in self._library) / len(self._library), 3
            ),
        }

    # ------------------------------------------------------------------
    # Experiment generation
    # ------------------------------------------------------------------

    def analogize_to_experiment(self, analogy: Analogy) -> dict:
        """
        Convert an analogy to a concrete experiment specification.

        The experiment dict can be passed to the hypothesis generator
        as a ``param_delta`` or ``experiment_json`` payload.

        Parameters
        ----------
        analogy : Analogy

        Returns
        -------
        dict
            Keys: idea_text, domain, experiment_type, param_delta,
                  rationale, priority, components.
        """
        # Map domain to experiment types
        domain_experiment_map: Dict[str, str] = {
            "thermodynamics":    "parameter_sweep",
            "fluid_dynamics":    "regime_gate",
            "ecology":           "signal_addition",
            "game_theory":       "signal_addition",
            "information_theory": "signal_addition",
            "network_theory":    "regime_gate",
        }
        exp_type = domain_experiment_map.get(analogy.domain, "parameter_sweep")

        # Extract numeric hints from the experiment_hint string
        param_delta: dict = {}
        numbers = re.findall(r"(\d+(?:\.\d+)?)", analogy.experiment_hint)
        if numbers:
            param_delta["suggested_value"] = float(numbers[0])

        # Add structured hints based on tags
        for tag in analogy.tags:
            if tag == "regime_filter":
                param_delta["regime_method"] = "heuristic"
            elif tag == "position_sizing":
                param_delta["sizing_method"] = "vol_scaled"
            elif tag == "entry_signal":
                param_delta["signal_type"] = "derived"

        return {
            "idea_text":       (
                f"[{analogy.domain.upper()}] {analogy.source_concept} → "
                f"{analogy.target_concept}"
            ),
            "domain":          analogy.domain,
            "experiment_type": exp_type,
            "param_delta":     param_delta,
            "rationale":       analogy.description,
            "priority":        analogy.confidence,
            "components":      analogy.tags,
            "experiment_hint": analogy.experiment_hint,
        }

    # ------------------------------------------------------------------
    # Random selection
    # ------------------------------------------------------------------

    def random_analogy(self, domain: Optional[str] = None) -> Analogy:
        """
        Return a random analogy, optionally restricted to *domain*.

        Parameters
        ----------
        domain : str or None

        Returns
        -------
        Analogy
        """
        import random
        pool = self.find_by_domain(domain) if domain else self._library
        if not pool:
            raise ValueError(f"No analogies found for domain: {domain!r}")
        return random.choice(pool)

    def random_n(self, n: int, domain: Optional[str] = None) -> List[Analogy]:
        """
        Return *n* distinct random analogies.

        Parameters
        ----------
        n : int
        domain : str or None

        Returns
        -------
        List[Analogy]
        """
        import random
        pool = self.find_by_domain(domain) if domain else self._library
        return random.sample(pool, min(n, len(pool)))

    def __repr__(self) -> str:
        return f"AnalogyEngine(analogies={len(self._library)})"

    def __len__(self) -> int:
        return len(self._library)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = AnalogyEngine()
    print(f"Analogy library: {len(engine)} entries")
    print("Stats:", engine.stats())
    print()
    results = engine.find_analogies("entropy")
    print(f"Query 'entropy' → {len(results)} analogies:")
    for a in results:
        print(f"  [{a.domain:20s}] {a.source_concept:<35s} → {a.target_concept}")
    print()
    exp = engine.analogize_to_experiment(results[0])
    print("Experiment for first result:")
    for k, v in exp.items():
        print(f"  {k}: {v}")
