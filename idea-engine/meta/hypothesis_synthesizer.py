"""
Hypothesis Synthesizer: auto-generate new hypothesis templates from backtest patterns.

Mines successful backtest results, clusters them by structural features,
and mutates existing templates to discover new alpha sources.

Integration:
  - Reads from idea-engine/scoring/hypothesis_scorer.py for performance data
  - Uses idea-engine/genetic-hypothesis/mutation.py for template mutation
  - Outputs new hypothesis dicts for the debate chamber
"""

from __future__ import annotations
import math
import random
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SuccessfulPattern:
    """A pattern extracted from a confirmed-profitable hypothesis."""
    template_type: str
    regime: str
    direction: float
    conviction: float
    sharpe: float
    win_rate: float
    avg_hold_bars: float
    key_signals: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class SynthesizedTemplate:
    """A new hypothesis template created by the synthesizer."""
    template_id: str
    name: str
    parent_templates: List[str]
    template_type: str
    regime_affinity: List[str]
    entry_signals: List[str]
    exit_signals: List[str]
    expected_sharpe: float
    novelty_score: float
    mutation_type: str  # crossover / mutation / interpolation / inversion
    description: str


class PatternMiner:
    """Extract structural patterns from successful backtest records."""

    def mine(self, records: List[Dict]) -> List[SuccessfulPattern]:
        """
        Mine successful patterns from backtest results.
        records: list of dicts with keys like template_type, regime, sharpe, win_rate, etc.
        """
        patterns = []
        for r in records:
            if r.get("sharpe", 0) <= 0 or r.get("win_rate", 0) < 0.4:
                continue
            patterns.append(SuccessfulPattern(
                template_type=r.get("template_type", "unknown"),
                regime=r.get("regime", "unknown"),
                direction=r.get("direction", 0),
                conviction=r.get("conviction", 0.5),
                sharpe=r.get("sharpe", 0),
                win_rate=r.get("win_rate", 0.5),
                avg_hold_bars=r.get("avg_hold_bars", 10),
                key_signals=r.get("key_signals", []),
                metadata=r.get("metadata", {}),
            ))
        return patterns

    def cluster_patterns(self, patterns: List[SuccessfulPattern], n_clusters: int = 5) -> Dict[int, List[SuccessfulPattern]]:
        """Cluster patterns by structural similarity using simple k-means on features."""
        if len(patterns) < n_clusters:
            return {0: patterns}

        # Feature vector: [sharpe, win_rate, avg_hold, conviction, direction]
        features = np.array([
            [p.sharpe, p.win_rate, p.avg_hold_bars / 100, p.conviction, p.direction]
            for p in patterns
        ])

        # Normalize
        means = features.mean(axis=0)
        stds = features.std(axis=0) + 1e-10
        normed = (features - means) / stds

        # Simple k-means
        rng = np.random.default_rng(42)
        centroids = normed[rng.choice(len(normed), n_clusters, replace=False)]

        for _ in range(20):
            # Assign
            dists = np.array([[np.linalg.norm(x - c) for c in centroids] for x in normed])
            labels = dists.argmin(axis=1)
            # Update
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    centroids[k] = normed[mask].mean(axis=0)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(patterns[i])
        return dict(clusters)


class TemplateMutator:
    """Mutate and recombine existing templates to create new ones."""

    SIGNAL_POOL = [
        "momentum_12m", "momentum_1m", "mean_reversion_zscore", "rsi_14",
        "macd", "bollinger_pct", "atr_ratio", "volume_breakout",
        "funding_rate", "options_flow", "order_flow_imbalance", "vpin",
        "hurst_exponent", "entropy", "bh_mass", "geodesic_deviation",
        "regime_alignment", "correlation_regime", "volatility_surface_skew",
        "credit_spread", "yield_curve_slope", "macro_momentum",
    ]

    TEMPLATE_TYPES = [
        "momentum", "mean_reversion", "breakout", "volatility_arb",
        "stat_arb", "regime_adaptive", "physics_inspired", "microstructure",
        "macro_micro_fusion", "carry", "liquidation_cascade",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"synth_{self._counter:04d}"

    def mutate_template(self, pattern: SuccessfulPattern) -> SynthesizedTemplate:
        """Create a mutated version of a successful pattern."""
        # Randomly modify one aspect
        mutations = ["add_signal", "change_regime", "flip_direction", "change_timeframe"]
        mutation = self.rng.choice(mutations)

        new_signals = list(pattern.key_signals) if pattern.key_signals else [self.rng.choice(self.SIGNAL_POOL)]
        new_regime = pattern.regime
        new_direction = pattern.direction
        template_type = pattern.template_type

        if mutation == "add_signal":
            available = [s for s in self.SIGNAL_POOL if s not in new_signals]
            if available:
                new_signals.append(self.rng.choice(available))
            desc = f"Added signal {new_signals[-1]} to {template_type}"

        elif mutation == "change_regime":
            regimes = ["trending_bull", "trending_bear", "mean_reverting", "high_volatility", "crisis"]
            new_regime = self.rng.choice([r for r in regimes if r != pattern.regime])
            desc = f"Adapted {template_type} for {new_regime} regime (was {pattern.regime})"

        elif mutation == "flip_direction":
            new_direction = -pattern.direction
            desc = f"Inverted {template_type} from {'long' if pattern.direction > 0 else 'short'} to {'short' if pattern.direction > 0 else 'long'}"

        elif mutation == "change_timeframe":
            template_type = self.rng.choice(self.TEMPLATE_TYPES)
            desc = f"Reframed {pattern.template_type} as {template_type}"

        return SynthesizedTemplate(
            template_id=self._next_id(),
            name=f"Synth: {template_type} [{new_regime}]",
            parent_templates=[pattern.template_type],
            template_type=template_type,
            regime_affinity=[new_regime],
            entry_signals=new_signals,
            exit_signals=["atr_trailing_stop", "regime_change"],
            expected_sharpe=pattern.sharpe * 0.7,
            novelty_score=0.8,
            mutation_type="mutation",
            description=desc,
        )

    def crossover_templates(self, pattern_a: SuccessfulPattern,
                             pattern_b: SuccessfulPattern) -> SynthesizedTemplate:
        """Combine two successful patterns into a hybrid."""
        # Take signals from both parents
        signals_a = set(pattern_a.key_signals or [])
        signals_b = set(pattern_b.key_signals or [])
        combined = list(signals_a | signals_b)
        if not combined:
            combined = [self.rng.choice(self.SIGNAL_POOL)]

        # Mix properties
        template_type = self.rng.choice([pattern_a.template_type, pattern_b.template_type])
        regime = self.rng.choice([pattern_a.regime, pattern_b.regime])
        direction = pattern_a.direction if abs(pattern_a.sharpe) > abs(pattern_b.sharpe) else pattern_b.direction
        expected_sharpe = (pattern_a.sharpe + pattern_b.sharpe) / 2 * 0.8

        return SynthesizedTemplate(
            template_id=self._next_id(),
            name=f"Hybrid: {pattern_a.template_type} x {pattern_b.template_type}",
            parent_templates=[pattern_a.template_type, pattern_b.template_type],
            template_type=template_type,
            regime_affinity=[pattern_a.regime, pattern_b.regime],
            entry_signals=combined[:5],
            exit_signals=["atr_trailing_stop", "regime_change", "time_decay"],
            expected_sharpe=expected_sharpe,
            novelty_score=0.9,
            mutation_type="crossover",
            description=f"Hybrid of {pattern_a.template_type} (Sharpe {pattern_a.sharpe:.2f}) and {pattern_b.template_type} (Sharpe {pattern_b.sharpe:.2f})",
        )


class HypothesisSynthesizer:
    """
    End-to-end pipeline: mine patterns -> cluster -> mutate/crossover -> output new templates.
    """

    def __init__(self, seed: int = 42):
        self.miner = PatternMiner()
        self.mutator = TemplateMutator(seed)
        self.rng = random.Random(seed)

    def synthesize(
        self,
        backtest_records: List[Dict],
        n_templates: int = 10,
        n_clusters: int = 5,
    ) -> List[SynthesizedTemplate]:
        """
        Generate new hypothesis templates from backtest results.

        Steps:
          1. Mine successful patterns
          2. Cluster by structural similarity
          3. For each cluster: generate mutations and crossovers
          4. Rank by expected novelty and return top N
        """
        # 1. Mine
        patterns = self.miner.mine(backtest_records)
        if not patterns:
            return []

        # 2. Cluster
        clusters = self.miner.cluster_patterns(patterns, min(n_clusters, len(patterns)))

        # 3. Generate new templates
        candidates = []

        for cluster_id, cluster_patterns in clusters.items():
            # Mutation: take best pattern in cluster, mutate it
            best = max(cluster_patterns, key=lambda p: p.sharpe)
            for _ in range(2):
                candidates.append(self.mutator.mutate_template(best))

            # Crossover: combine two patterns from different clusters
            other_clusters = [p for cid, ps in clusters.items() if cid != cluster_id for p in ps]
            if other_clusters:
                partner = self.rng.choice(other_clusters)
                candidates.append(self.mutator.crossover_templates(best, partner))

        # 4. Rank by novelty * expected_sharpe and return top N
        candidates.sort(key=lambda t: t.novelty_score * t.expected_sharpe, reverse=True)
        return candidates[:n_templates]

    def format_for_debate(self, templates: List[SynthesizedTemplate]) -> List[Dict]:
        """Convert synthesized templates into dicts the debate chamber can consume."""
        return [
            {
                "id": t.template_id,
                "name": t.name,
                "template_type": t.template_type,
                "regime": t.regime_affinity[0] if t.regime_affinity else "unknown",
                "direction": 1.0,
                "conviction": 0.5,
                "entry_signals": t.entry_signals,
                "exit_signals": t.exit_signals,
                "expected_sharpe": t.expected_sharpe,
                "description": t.description,
                "source": "auto_synthesized",
                "parent_templates": t.parent_templates,
            }
            for t in templates
        ]
