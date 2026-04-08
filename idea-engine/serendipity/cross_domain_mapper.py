"""
cross_domain_mapper.py
Maps trading/financial concepts to other scientific domains to generate
creative, non-obvious hypotheses via structural analogy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrossDomainMapping:
    source_domain: str          # e.g., "trading"
    target_domain: str          # e.g., "fluid_dynamics"
    concept: str                # trading concept being mapped
    analogy: str                # the analogous concept in the target domain
    trading_insight: str        # actionable trading insight derived from the analogy
    strength: float             # 0–1: how structurally sound is the analogy
    serendipity_score: float    # 0–1: how non-obvious / surprising is this mapping
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# All 30+ domain mappings
# ---------------------------------------------------------------------------
_MAPPINGS: list[CrossDomainMapping] = [

    # --- Fluid Dynamics ---
    CrossDomainMapping(
        source_domain="trading", target_domain="fluid_dynamics",
        concept="order flow",
        analogy="fluid flow through a pipe",
        trading_insight=(
            "Like fluid, order flow follows the path of least resistance (liquidity). "
            "Turbulence in flow (erratic, high-variance order arrival) signals "
            "transition to a chaotic regime — widen stops and reduce size."
        ),
        strength=0.82, serendipity_score=0.55,
        tags=["liquidity", "order_flow", "regime"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="fluid_dynamics",
        concept="market crash / selloff cascade",
        analogy="hydraulic jump (sudden drop in flow velocity)",
        trading_insight=(
            "A hydraulic jump occurs when fast shallow flow hits slow deep water. "
            "Analog: when fast-moving momentum hits deep value buying, violent "
            "deceleration and reversal can occur — look for value anchors as "
            "natural resistance levels in panics."
        ),
        strength=0.70, serendipity_score=0.78,
        tags=["crash", "momentum", "value", "reversal"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="fluid_dynamics",
        concept="liquidity",
        analogy="viscosity",
        trading_insight=(
            "High-viscosity fluids resist flow; low-viscosity fluids flow freely. "
            "Low market liquidity = high viscosity = orders move prices more. "
            "Price impact models should use a viscosity proxy (bid-ask spread) "
            "to adjust expected slippage dynamically."
        ),
        strength=0.88, serendipity_score=0.45,
        tags=["liquidity", "market_impact", "execution"],
    ),

    # --- Thermodynamics ---
    CrossDomainMapping(
        source_domain="trading", target_domain="thermodynamics",
        concept="market volatility",
        analogy="temperature",
        trading_insight=(
            "High temperature = high particle kinetic energy = high vol. "
            "Markets have a 'thermal equilibrium' they return to (VIX mean reversion). "
            "Phase transitions (regime changes) occur at critical temperatures — "
            "watch for vol at extremes as a phase transition warning."
        ),
        strength=0.85, serendipity_score=0.40,
        tags=["volatility", "regime", "mean_reversion"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="thermodynamics",
        concept="information in price discovery",
        analogy="entropy",
        trading_insight=(
            "Maximum entropy in a market = maximum uncertainty = no exploitable signal. "
            "Alpha decays as entropy increases (information diffuses). "
            "Measure market entropy via permutation entropy of price series — "
            "low entropy periods are windows of exploitable inefficiency."
        ),
        strength=0.78, serendipity_score=0.72,
        tags=["alpha_decay", "information", "efficiency"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="thermodynamics",
        concept="trend vs mean reversion",
        analogy="heat engines vs refrigerators",
        trading_insight=(
            "A heat engine extracts work from temperature differentials (trend following). "
            "A refrigerator pumps heat against gradient using work input (mean reversion). "
            "Both are viable strategies but require different conditions (entropy gradient). "
            "Market regimes determine which 'engine' is efficient at a given time."
        ),
        strength=0.62, serendipity_score=0.85,
        tags=["trend_following", "mean_reversion", "regime"],
    ),

    # --- Ecology ---
    CrossDomainMapping(
        source_domain="trading", target_domain="ecology",
        concept="crowded trades",
        analogy="competitive exclusion principle",
        trading_insight=(
            "Two species cannot occupy the same ecological niche indefinitely. "
            "Two strategies with identical risk/return cannot both survive in the "
            "same market. As a trade becomes crowded, its risk-adjusted return "
            "must decline. Monitor strategy capacity and AUM concentration as "
            "leading indicators of crowding-driven alpha decay."
        ),
        strength=0.80, serendipity_score=0.68,
        tags=["crowding", "alpha_decay", "capacity"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="ecology",
        concept="market maker profitability",
        analogy="predator-prey dynamics (Lotka-Volterra)",
        trading_insight=(
            "Market makers (predators) feed on informed traders (prey). "
            "Too many MMs → prey is depleted → MMs become unprofitable → exit. "
            "Cycles in MM profitability predict liquidity supply cycles. "
            "When MM P&L is negative (many MMs exiting), spreads will widen."
        ),
        strength=0.73, serendipity_score=0.76,
        tags=["market_making", "liquidity", "cycles"],
    ),

    # --- Evolutionary Biology ---
    CrossDomainMapping(
        source_domain="trading", target_domain="evolutionary_biology",
        concept="strategy survival in markets",
        analogy="natural selection / fitness landscape",
        trading_insight=(
            "Strategies that survive market evolution are locally optimal, not globally. "
            "A strategy that works in trending markets may be 'fit' only in that niche. "
            "Like species on an island, isolated (niche) strategies can thrive. "
            "Genetic drift analog: random market regime changes can wipe out 'fit' strategies."
        ),
        strength=0.75, serendipity_score=0.65,
        tags=["strategy_survival", "regime", "diversification"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="evolutionary_biology",
        concept="signal adaptation / overfitting",
        analogy="evolutionary arms race",
        trading_insight=(
            "As traders exploit a signal, the market adapts (like prey evolving defenses). "
            "The signal's half-life shortens as more capital pursues it. "
            "Counter-strategy: seek signals in market 'niches' with fewer competitors, "
            "or rotate to next-generation signals before the arms race commoditizes current ones."
        ),
        strength=0.82, serendipity_score=0.70,
        tags=["alpha_decay", "signal", "adaptation"],
    ),

    # --- Quantum Mechanics ---
    CrossDomainMapping(
        source_domain="trading", target_domain="quantum_mechanics",
        concept="price observation changing the market",
        analogy="observer effect / wave function collapse",
        trading_insight=(
            "Publishing a trading signal or portfolio holding 'collapses' the "
            "information uncertainty — other traders front-run it, eroding the edge. "
            "Signals should be treated as 'superposed' (private) until acted upon. "
            "Delay of any signal publication is equivalent to extending coherence time."
        ),
        strength=0.55, serendipity_score=0.90,
        tags=["signal", "information", "front_running"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="quantum_mechanics",
        concept="option pricing uncertainty",
        analogy="Heisenberg uncertainty principle",
        trading_insight=(
            "You cannot simultaneously know price AND volatility with arbitrary precision. "
            "Attempts to pin down expected price (via forecasts) increase vol uncertainty. "
            "Trade both legs: directional view + vol overlay is more robust than either alone."
        ),
        strength=0.50, serendipity_score=0.88,
        tags=["options", "uncertainty", "volatility"],
    ),

    # --- Information Theory ---
    CrossDomainMapping(
        source_domain="trading", target_domain="information_theory",
        concept="alpha from data",
        analogy="channel capacity (Shannon)",
        trading_insight=(
            "The maximum achievable information rate over a noisy channel is log2(1 + SNR). "
            "Analog: alpha from a signal is bounded by its signal-to-noise ratio. "
            "Combining N uncorrelated signals multiplies SNR — diversification has "
            "an information-theoretic bound: sqrt(N) improvement in Sharpe."
        ),
        strength=0.88, serendipity_score=0.62,
        tags=["alpha", "diversification", "signal_quality"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="information_theory",
        concept="portfolio rebalancing",
        analogy="data compression",
        trading_insight=(
            "Rebalancing a portfolio removes 'redundant' exposure (like compression). "
            "Optimal rebalancing frequency = optimal compression rate for the return process. "
            "Rebalancing too often (over-compression) wastes transaction costs; too rarely "
            "loses diversification benefit — there is an optimal rebalancing entropy."
        ),
        strength=0.72, serendipity_score=0.80,
        tags=["rebalancing", "portfolio", "transaction_costs"],
    ),

    # --- Game Theory ---
    CrossDomainMapping(
        source_domain="trading", target_domain="game_theory",
        concept="front-running / market impact",
        analogy="prisoner's dilemma",
        trading_insight=(
            "If all large funds hold back orders to avoid impact, everyone benefits. "
            "But each fund has an incentive to execute faster (defect). "
            "Result: Nash equilibrium is everyone executes aggressively — impact is high. "
            "Cooperative execution protocols (crossing networks, dark pools) attempt to "
            "escape the dilemma but require credible commitment mechanisms."
        ),
        strength=0.83, serendipity_score=0.58,
        tags=["execution", "market_impact", "dark_pool"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="game_theory",
        concept="market-maker spread setting",
        analogy="Bertrand competition",
        trading_insight=(
            "MMs compete on spread like firms compete on price. "
            "In Bertrand equilibrium, price → marginal cost (spread → adverse selection cost). "
            "When MM competition increases (more ECNs), spreads narrow to bare adverse selection. "
            "Signal: count of active liquidity providers as a predictor of spread tightness."
        ),
        strength=0.79, serendipity_score=0.60,
        tags=["market_making", "spread", "liquidity"],
    ),

    # --- Materials Science ---
    CrossDomainMapping(
        source_domain="trading", target_domain="materials_science",
        concept="market support/resistance levels",
        analogy="crystal lattice defects / dislocations",
        trading_insight=(
            "Just as crystal dislocations concentrate stress at impurities, "
            "price levels with high historical trading volume concentrate future "
            "buy/sell orders (support/resistance). Breakouts require enough energy "
            "(volume/momentum) to overcome the 'lattice binding energy' at these levels."
        ),
        strength=0.68, serendipity_score=0.82,
        tags=["support_resistance", "technical", "volume"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="materials_science",
        concept="portfolio diversification",
        analogy="composite materials",
        trading_insight=(
            "Composites combine materials with different stress/strain profiles to exceed "
            "the properties of either alone. A portfolio combining assets with different "
            "correlation structures in stress periods (not just normal periods) is more "
            "'damage-resistant' — seek assets that are uncorrelated in tail scenarios."
        ),
        strength=0.77, serendipity_score=0.71,
        tags=["diversification", "tail_risk", "correlation"],
    ),

    # --- Neuroscience ---
    CrossDomainMapping(
        source_domain="trading", target_domain="neuroscience",
        concept="trend-following signal persistence",
        analogy="neural habituation",
        trading_insight=(
            "Neurons reduce firing rate under sustained constant stimulus (habituation). "
            "Markets habituate to persistent trends — momentum slows as the trend is "
            "'priced in'. After habituation, a novel stimulus (trend break) causes "
            "outsized response. Trade the re-sensitization: buy/sell breakouts after "
            "prolonged low-vol consolidations."
        ),
        strength=0.65, serendipity_score=0.83,
        tags=["momentum", "breakout", "habituation"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="neuroscience",
        concept="market overreaction / underreaction",
        analogy="predictive coding framework",
        trading_insight=(
            "The brain generates predictions and only processes prediction errors. "
            "Markets similarly react more to surprises than to expected outcomes. "
            "An earnings beat that was 'expected' causes minimal price move. "
            "Quantify expected vs. actual analyst estimate dispersion to find "
            "signals in 'residual surprise' not captured by consensus."
        ),
        strength=0.80, serendipity_score=0.72,
        tags=["earnings", "surprise", "analyst_estimates"],
    ),

    # --- Epidemiology ---
    CrossDomainMapping(
        source_domain="trading", target_domain="epidemiology",
        concept="market contagion / correlation spikes",
        analogy="infectious disease spread (SIR model)",
        trading_insight=(
            "Panic selling spreads like an infection: S (susceptible) = leveraged longs; "
            "I (infected) = margin-called sellers; R (recovered) = deleveraged cash holders. "
            "Peak selling (peak I) occurs before bottom — monitor margin debt as S proxy, "
            "forced selling flow as I proxy. Trough is when R (cash) is maximized."
        ),
        strength=0.78, serendipity_score=0.74,
        tags=["contagion", "crash", "margin_debt", "capitulation"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="epidemiology",
        concept="narrative/sentiment spread",
        analogy="viral transmission R0",
        trading_insight=(
            "A bullish narrative with R0 > 1 (each believer converts >1 skeptic) grows "
            "exponentially — early detection of high-R0 narratives allows positioning "
            "ahead of the crowd. NLP on social/news data can estimate narrative R0 "
            "by tracking mention growth rates."
        ),
        strength=0.70, serendipity_score=0.80,
        tags=["narrative", "sentiment", "nlp", "momentum"],
    ),

    # --- Complex Networks ---
    CrossDomainMapping(
        source_domain="trading", target_domain="complex_networks",
        concept="systemic risk / too-big-to-fail",
        analogy="network hubs and cascade failure",
        trading_insight=(
            "In scale-free networks, highly connected hubs drive cascade failures. "
            "Financial system hubs (large banks, prime brokers) create systemic risk "
            "when they fail. Map counterparty networks to identify systemic hubs — "
            "long their CDS or put options as tail hedges before stress periods."
        ),
        strength=0.85, serendipity_score=0.55,
        tags=["systemic_risk", "tail_hedge", "counterparty"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="complex_networks",
        concept="cross-asset correlation dynamics",
        analogy="small-world network topology",
        trading_insight=(
            "Small-world networks have high clustering and short path lengths. "
            "In risk-off environments, financial asset correlations spike (path length shrinks). "
            "Monitor the 'clustering coefficient' of asset return correlations — "
            "as it rises toward 1.0, diversification is failing; reduce gross exposure."
        ),
        strength=0.76, serendipity_score=0.70,
        tags=["correlation", "diversification", "risk_off"],
    ),

    # --- Geology ---
    CrossDomainMapping(
        source_domain="trading", target_domain="geology",
        concept="volatility accumulation before a crash",
        analogy="tectonic stress accumulation (earthquake model)",
        trading_insight=(
            "Stress accumulates along fault lines until rupture. Markets accumulate "
            "imbalances (leverage, valuation extremes, positioning) until a shock triggers "
            "release. The Gutenberg-Richter law: small corrections are frequent, large ones rare. "
            "Power-law tail fitting of historical drawdowns can estimate VaR more accurately "
            "than Gaussian assumptions."
        ),
        strength=0.72, serendipity_score=0.76,
        tags=["tail_risk", "var", "leverage", "crash"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="geology",
        concept="slow-moving macro trends",
        analogy="continental drift",
        trading_insight=(
            "Tectonic plates move millimeters per year but create mountains over millennia. "
            "Demographic and debt-cycle trends are market tectonic forces — invisible in daily "
            "noise but dominant over decades. Separate 'tectonic' from 'weather' signals "
            "in macro analysis; tectonic signals rarely reverse and warrant highest conviction sizing."
        ),
        strength=0.75, serendipity_score=0.65,
        tags=["macro", "long_term", "debt_cycle", "demographics"],
    ),

    # --- Meteorology ---
    CrossDomainMapping(
        source_domain="trading", target_domain="meteorology",
        concept="market regime prediction",
        analogy="ensemble weather forecasting",
        trading_insight=(
            "Weather is forecast via ensemble of models — no single model is trusted. "
            "Market regime prediction should similarly use model ensembles. "
            "Spread across ensemble members = uncertainty; narrow spread = high confidence. "
            "When regime ensemble disagrees, reduce position size proportionally."
        ),
        strength=0.85, serendipity_score=0.50,
        tags=["regime", "ensemble", "uncertainty", "forecasting"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="meteorology",
        concept="momentum vs mean reversion cycle",
        analogy="El Nino / La Nina (ENSO cycle)",
        trading_insight=(
            "ENSO alternates between warm and cold phases on 3–7 year cycles. "
            "Equity markets alternate between trending (momentum) and range-bound "
            "(mean reversion) regimes. Identify the 'ENSO analog' in equity markets: "
            "credit cycle, earnings cycle. Position sizing should reflect cycle phase."
        ),
        strength=0.65, serendipity_score=0.78,
        tags=["cycles", "regime", "momentum", "mean_reversion"],
    ),

    # --- Optics ---
    CrossDomainMapping(
        source_domain="trading", target_domain="optics",
        concept="signal filtering / noise reduction",
        analogy="optical low-pass filter",
        trading_insight=(
            "Low-pass optical filters remove high-frequency light. Moving averages are "
            "financial low-pass filters. Choosing the right filter cutoff frequency "
            "(MA period) requires knowing the signal's dominant frequency. "
            "Use Fourier analysis on returns to identify dominant cycles before "
            "choosing smoothing parameters."
        ),
        strength=0.80, serendipity_score=0.60,
        tags=["signal_processing", "moving_average", "fourier"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="optics",
        concept="diversification across time horizons",
        analogy="chromatic dispersion (different wavelengths travel at different speeds)",
        trading_insight=(
            "Just as different wavelengths of light travel at different speeds in glass, "
            "different information horizons propagate through markets at different speeds. "
            "Short-term micro signals travel fast (intraday); macro signals travel slowly. "
            "A 'prism portfolio' that separates exposures by holding period avoids "
            "conflating fast and slow signals."
        ),
        strength=0.68, serendipity_score=0.87,
        tags=["multi_horizon", "signal_separation", "portfolio"],
    ),

    # --- Statistical Physics ---
    CrossDomainMapping(
        source_domain="trading", target_domain="statistical_physics",
        concept="herding / momentum",
        analogy="Ising model spin alignment",
        trading_insight=(
            "In the Ising model, spins align when neighbor coupling exceeds thermal noise. "
            "Market participants 'align' (herd) when social/momentum coupling exceeds "
            "private information noise. The phase transition (from disordered to ordered) "
            "corresponds to a momentum regime onset. Correlation across participants "
            "(fund flows all in one direction) is the order parameter."
        ),
        strength=0.80, serendipity_score=0.72,
        tags=["herding", "momentum", "phase_transition"],
    ),
    CrossDomainMapping(
        source_domain="trading", target_domain="statistical_physics",
        concept="market bubble",
        analogy="percolation threshold",
        trading_insight=(
            "Percolation: a connected cluster spanning the entire lattice forms abruptly "
            "at a critical density. Market bubble: optimism 'percolates' — at critical "
            "density of buyers, every seller can find a buyer (price insensitive buying). "
            "Measure market 'density of optimism' via sentiment surveys, P/E expansion, "
            "IPO volume — bubble bursts when density falls below threshold."
        ),
        strength=0.74, serendipity_score=0.82,
        tags=["bubble", "sentiment", "phase_transition"],
    ),
]


# ---------------------------------------------------------------------------
# DomainMapper
# ---------------------------------------------------------------------------
class DomainMapper:
    """Maps trading concepts to other scientific domains for creative insight generation."""

    def __init__(self) -> None:
        self._mappings: list[CrossDomainMapping] = list(_MAPPINGS)

    # -----------------------------------------------------------------------
    def map_concept(self, trading_concept: str) -> list[CrossDomainMapping]:
        """
        Return all mappings where the concept matches (case-insensitive substring).
        """
        concept_lower = trading_concept.lower()
        return [m for m in self._mappings if concept_lower in m.concept.lower()]

    # -----------------------------------------------------------------------
    def find_analogies_for_regime(self, regime: str) -> list[CrossDomainMapping]:
        """
        Return mappings relevant to a given market regime.
        regime: e.g. "trending", "mean_reverting", "risk_off", "low_vol", "crash"
        Matches against tags and concept/trading_insight text.
        """
        regime_lower = regime.lower()
        results = []
        for m in self._mappings:
            tag_match = any(regime_lower in t for t in m.tags)
            text_match = (
                regime_lower in m.trading_insight.lower()
                or regime_lower in m.concept.lower()
            )
            if tag_match or text_match:
                results.append(m)
        return sorted(results, key=lambda x: x.strength, reverse=True)

    # -----------------------------------------------------------------------
    def generate_novel_hypothesis(
        self,
        domain1: str,
        domain2: str,
        seed: Optional[int] = None,
    ) -> dict[str, str]:
        """
        Combine analogies from two different scientific domains to produce
        a 'hypothesis seed' — a novel trading idea at their intersection.
        Returns a dict with keys: domains, premise, hypothesis, test_suggestion.
        """
        rng = random.Random(seed)

        d1_maps = [m for m in self._mappings if m.target_domain == domain1]
        d2_maps = [m for m in self._mappings if m.target_domain == domain2]

        if not d1_maps:
            raise ValueError(f"No mappings found for domain: {domain1}")
        if not d2_maps:
            raise ValueError(f"No mappings found for domain: {domain2}")

        m1 = rng.choice(d1_maps)
        m2 = rng.choice(d2_maps)

        hypothesis = (
            f"Combining the '{m1.analogy}' from {domain1} with the "
            f"'{m2.analogy}' from {domain2}: "
            f"If {m1.trading_insight.split('.')[0].strip().lower()}, "
            f"then also consider that {m2.trading_insight.split('.')[0].strip().lower()}. "
            f"A unified model might detect regime transitions by simultaneously tracking "
            f"signals from both frameworks."
        )

        return {
            "domains": f"{domain1} x {domain2}",
            "domain1_analogy": m1.analogy,
            "domain1_insight": m1.trading_insight,
            "domain2_analogy": m2.analogy,
            "domain2_insight": m2.trading_insight,
            "premise": (
                f"Both {domain1} and {domain2} describe systems with "
                f"phase transitions, non-linear dynamics, and emergent order."
            ),
            "hypothesis": hypothesis,
            "test_suggestion": (
                f"Backtest a combined signal: use {m1.concept} as the primary entry filter "
                f"and {m2.concept} as the regime confirmation. Measure alpha vs. each signal alone."
            ),
            "combined_serendipity": round(
                (m1.serendipity_score + m2.serendipity_score) / 2
                + abs(m1.serendipity_score - m2.serendipity_score) * 0.3, 3
            ),
        }

    # -----------------------------------------------------------------------
    def analogy_strength_score(
        self,
        mapping: CrossDomainMapping,
        current_regime: str = "trending",
    ) -> float:
        """
        Compute a context-adjusted utility score for an analogy given
        the current market state. Penalizes analogies that are regime-mismatched.
        Score is in [0, 1].
        """
        base = mapping.strength
        regime_lower = current_regime.lower()

        # Boost if tags align with regime
        tag_bonus = 0.1 if any(regime_lower in t for t in mapping.tags) else 0.0

        # Penalize very high serendipity in high-conviction regimes (too speculative)
        spec_penalty = 0.05 if mapping.serendipity_score > 0.85 else 0.0

        score = min(1.0, base + tag_bonus - spec_penalty)
        return round(score, 4)

    # -----------------------------------------------------------------------
    def serendipity_score(self, mapping: CrossDomainMapping) -> float:
        """
        Return the raw serendipity score — how surprising and non-obvious
        the cross-domain connection is.
        Adjusted by: domain distance heuristic (more distant domains = more serendipitous).
        """
        domain_distance_map: dict[str, int] = {
            "fluid_dynamics": 2,
            "thermodynamics": 2,
            "ecology": 4,
            "evolutionary_biology": 4,
            "quantum_mechanics": 5,
            "information_theory": 2,
            "game_theory": 1,
            "materials_science": 4,
            "neuroscience": 4,
            "epidemiology": 4,
            "complex_networks": 2,
            "geology": 5,
            "meteorology": 3,
            "optics": 4,
            "statistical_physics": 2,
        }
        distance = domain_distance_map.get(mapping.target_domain, 3)
        # Normalize distance contribution [0, 1]
        distance_factor = distance / 5.0
        # Blend raw score with distance factor
        adjusted = 0.7 * mapping.serendipity_score + 0.3 * distance_factor
        return round(min(1.0, adjusted), 4)

    # -----------------------------------------------------------------------
    def top_insights(self, n: int = 5, by: str = "strength") -> list[CrossDomainMapping]:
        """Return top-n mappings sorted by 'strength' or 'serendipity_score'."""
        if by not in ("strength", "serendipity_score"):
            raise ValueError("by must be 'strength' or 'serendipity_score'")
        return sorted(self._mappings, key=lambda m: getattr(m, by), reverse=True)[:n]

    # -----------------------------------------------------------------------
    def all_domains(self) -> list[str]:
        """Return list of unique target domains in the mapping library."""
        return sorted(set(m.target_domain for m in self._mappings))

    # -----------------------------------------------------------------------
    def summary_table(self) -> list[dict[str, str]]:
        """Return a compact summary of all mappings for display or export."""
        return [
            {
                "concept": m.concept,
                "target_domain": m.target_domain,
                "analogy": m.analogy,
                "strength": str(m.strength),
                "serendipity": str(m.serendipity_score),
            }
            for m in self._mappings
        ]
