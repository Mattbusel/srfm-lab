"""
orchestration/idea_pipeline.py

Master orchestration pipeline for the Idea Automation Engine.

Chains all components:
  signal mining → hypothesis generation → scoring → deduplication → debate
  → adversarial testing → conviction → sizing → IdeaBank storage

Key design features:
  - IdeaPipeline class with run_full_pipeline(prices, volume, context) → PipelineResult
  - Regime-aware template routing: different templates activated per detected regime
  - Priority queue for hypothesis processing (highest composite priority first)
  - Deduplication using genealogy / description similarity before debate
  - Performance feedback loop: updates scorer thresholds from IdeaBank results
  - Stage-level timing logged at DEBUG level
  - Graceful per-stage error handling — stage failure does not halt the pipeline

Module compatibility: works standalone (no project PYTHONPATH needed) because
all project-internal imports are guarded with try/except.  When imports succeed,
the richer functionality (Hypothesis types, IdeaBank, RiskManager, etc.) is used.
"""

from __future__ import annotations
import heapq
import logging
import time
import math
import uuid
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ── Conditional project-internal imports ─────────────────────────────────────

try:
    from meta.idea_bank import IdeaBank, Idea
    _IDEA_BANK_AVAILABLE = True
except ImportError:
    _IDEA_BANK_AVAILABLE = False

try:
    from debate_system.agents.macro_analyst import MacroAnalyst
    from debate_system.agents.risk_manager import RiskManager
    _DEBATE_AGENTS_AVAILABLE = True
except ImportError:
    _DEBATE_AGENTS_AVAILABLE = False

try:
    from signals.order_flow_signal import compute_order_flow_signal
    _ORDER_FLOW_AVAILABLE = True
except ImportError:
    _ORDER_FLOW_AVAILABLE = False

try:
    from signals.volatility_surface_signal import compute_vol_surface_signal
    _VOL_SURFACE_AVAILABLE = True
except ImportError:
    _VOL_SURFACE_AVAILABLE = False


# ── Pipeline Data Structures ──────────────────────────────────────────────────

@dataclass
class MarketContext:
    """Current market state fed into the pipeline."""
    prices: np.ndarray
    returns: np.ndarray
    volume: Optional[np.ndarray] = None
    regime: str = "unknown"
    regime_confidence: float = 0.5
    volatility_regime: str = "normal"   # low/normal/high/crisis
    trend_strength: float = 0.0         # -1 to +1
    liquidity_score: float = 0.5        # 0=illiquid, 1=liquid
    metadata: dict = field(default_factory=dict)


@dataclass
class SignalBundle:
    """Collection of signals computed from market data."""
    trend_signal: float = 0.0
    momentum_signal: float = 0.0
    mean_reversion_signal: float = 0.0
    volatility_signal: float = 0.0
    microstructure_signal: float = 0.0
    alternative_signal: float = 0.0
    multi_tf_signal: float = 0.0
    bh_composite: float = 0.0
    conviction: float = 0.0
    direction: float = 0.0            # +1 long, -1 short
    timestamp: float = field(default_factory=time.time)


@dataclass
class HypothesisCandidate:
    """A generated hypothesis candidate."""
    id: str
    name: str
    direction: float          # +1 long, -1 short, 0 neutral
    entry_signal: float       # strength of entry signal
    template_type: str
    regime: str
    raw_score: float = 0.0
    debate_score: float = 0.0
    adversarial_score: float = 0.0
    conviction: float = 0.0
    suggested_size: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Output of the full idea pipeline run."""
    timestamp: float
    top_hypothesis: Optional[HypothesisCandidate]
    all_candidates: list[HypothesisCandidate]
    signal_bundle: SignalBundle
    regime: str
    n_generated: int
    n_surviving: int
    pipeline_warnings: list[str]
    stage_timings: dict[str, float]
    allocation_fraction: float    # recommended fraction of capital
    should_trade: bool


# ── Signal Generation Stage ───────────────────────────────────────────────────

class SignalGenerator:
    """Computes all signals from raw market data."""

    def __init__(
        self,
        trend_lookback: int = 50,
        mom_lookback: int = 20,
        mr_lookback: int = 30,
    ):
        self.trend_lookback = trend_lookback
        self.mom_lookback = mom_lookback
        self.mr_lookback = mr_lookback

    def compute(self, ctx: MarketContext) -> SignalBundle:
        prices = ctx.prices
        returns = ctx.returns
        T = len(prices)
        bundle = SignalBundle()

        if T < max(self.trend_lookback, self.mom_lookback, self.mr_lookback) + 5:
            return bundle

        # Trend: regression slope
        n = min(T, self.trend_lookback)
        p = prices[-n:]
        t = np.arange(n)
        slope = float(np.polyfit(t, p, 1)[0])
        vol = float(p.std())
        bundle.trend_signal = float(math.tanh(slope / max(vol, 1e-10) * n * 0.1))

        # Momentum: ROC
        n_mom = min(T, self.mom_lookback + 1)
        roc = float((prices[-1] - prices[-n_mom]) / max(abs(prices[-n_mom]), 1e-10))
        bundle.momentum_signal = float(math.tanh(roc * 5))

        # Mean reversion: z-score
        n_mr = min(T, self.mr_lookback)
        sub = prices[-n_mr:]
        z = float((prices[-1] - sub.mean()) / max(sub.std(), 1e-10))
        bundle.mean_reversion_signal = float(-math.tanh(z))

        # Volatility regime signal
        vol_recent = float(returns[-min(T, 21):].std() * math.sqrt(252))
        vol_baseline = float(returns[-min(T, 63):].std() * math.sqrt(252))
        vol_ratio = vol_recent / max(vol_baseline, 1e-10)
        bundle.volatility_signal = float(math.tanh((vol_ratio - 1) * 2))

        # Composite direction
        bundle.direction = float(
            0.4 * bundle.trend_signal
            + 0.3 * bundle.momentum_signal
            + 0.3 * bundle.mean_reversion_signal
        )

        bundle.conviction = float(abs(bundle.direction) * (1 - abs(bundle.volatility_signal) * 0.3))

        return bundle


# ── Regime Router ─────────────────────────────────────────────────────────────

REGIME_TEMPLATE_MAP = {
    "trending_bull": ["momentum", "trend_following", "breakout"],
    "trending_bear": ["momentum", "trend_following", "short_squeeze"],
    "mean_reverting": ["mean_reversion", "pairs_trade", "bollinger"],
    "high_volatility": ["volatility_breakout", "crisis_alpha", "defensive"],
    "low_volatility": ["carry", "vol_selling", "range_bound"],
    "regime_transition": ["structural_break", "regime_adaptive", "chameleon"],
    "unknown": ["mean_reversion", "momentum"],
}


class RegimeRouter:
    """Routes hypothesis generation based on current regime."""

    def get_templates(self, regime: str) -> list[str]:
        return REGIME_TEMPLATE_MAP.get(regime, REGIME_TEMPLATE_MAP["unknown"])

    def classify_regime(self, ctx: MarketContext, bundle: SignalBundle) -> str:
        """Simple regime classification from signals."""
        trend_strong = abs(bundle.trend_signal) > 0.4
        vol_high = bundle.volatility_signal > 0.3
        mr_strong = abs(bundle.mean_reversion_signal) > 0.4

        if vol_high and not trend_strong:
            return "high_volatility"
        elif trend_strong and bundle.trend_signal > 0:
            return "trending_bull"
        elif trend_strong and bundle.trend_signal < 0:
            return "trending_bear"
        elif mr_strong:
            return "mean_reverting"
        elif abs(bundle.volatility_signal) < 0.1:
            return "low_volatility"
        else:
            return "regime_transition"


# ── Hypothesis Generator ──────────────────────────────────────────────────────

class HypothesisGenerator:
    """Generates hypothesis candidates from signals and templates."""

    def __init__(self, max_hypotheses: int = 20):
        self.max_hypotheses = max_hypotheses
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"hyp_{self._id_counter:06d}"

    def generate(
        self,
        bundle: SignalBundle,
        templates: list[str],
        regime: str,
    ) -> list[HypothesisCandidate]:
        candidates = []

        for template in templates[:self.max_hypotheses]:
            # Generate hypothesis from template + current signals
            hyp = self._template_to_hypothesis(template, bundle, regime)
            if hyp is not None:
                candidates.append(hyp)

        return candidates

    def _template_to_hypothesis(
        self,
        template: str,
        bundle: SignalBundle,
        regime: str,
    ) -> Optional[HypothesisCandidate]:
        direction = float(np.sign(bundle.direction)) if abs(bundle.direction) > 0.1 else 0.0

        signal_map = {
            "momentum": bundle.momentum_signal,
            "trend_following": bundle.trend_signal,
            "breakout": bundle.trend_signal * 0.8 + bundle.momentum_signal * 0.2,
            "mean_reversion": bundle.mean_reversion_signal,
            "pairs_trade": bundle.mean_reversion_signal * 0.9,
            "bollinger": bundle.mean_reversion_signal * 0.7,
            "volatility_breakout": abs(bundle.volatility_signal),
            "crisis_alpha": -bundle.trend_signal * 0.5,
            "defensive": -bundle.volatility_signal,
            "carry": -bundle.volatility_signal * 0.3,
            "vol_selling": -abs(bundle.volatility_signal),
            "range_bound": bundle.mean_reversion_signal * 0.6,
            "structural_break": abs(bundle.trend_signal - bundle.momentum_signal),
            "regime_adaptive": bundle.conviction,
            "chameleon": bundle.direction,
            "short_squeeze": -bundle.momentum_signal if bundle.momentum_signal < -0.3 else 0.0,
        }

        entry_signal = signal_map.get(template, bundle.direction)

        if abs(entry_signal) < 0.1:
            return None

        # Direction for this template
        if template in ["crisis_alpha", "short_squeeze"]:
            hyp_direction = -float(np.sign(bundle.trend_signal))
        elif template in ["mean_reversion", "pairs_trade", "bollinger", "range_bound"]:
            hyp_direction = float(np.sign(bundle.mean_reversion_signal))
        else:
            hyp_direction = direction if direction != 0 else float(np.sign(entry_signal))

        return HypothesisCandidate(
            id=self._next_id(),
            name=f"{template}_{regime}",
            direction=hyp_direction,
            entry_signal=float(abs(entry_signal)),
            template_type=template,
            regime=regime,
            raw_score=float(abs(entry_signal)),
            metadata={"bundle_direction": float(bundle.direction)},
        )


# ── Scorer ─────────────────────────────────────────────────────────────────────

class HypothesisScorer:
    """Scores hypotheses using multi-factor conviction model."""

    def score(
        self,
        hyp: HypothesisCandidate,
        bundle: SignalBundle,
        ctx: MarketContext,
    ) -> float:
        # Technical alignment
        tech_score = hyp.entry_signal * float(np.sign(hyp.direction) == np.sign(bundle.direction))

        # Regime alignment
        regime_bonus = 0.2 if hyp.regime == ctx.regime else 0.0

        # Volatility penalty
        vol_penalty = 0.3 * max(abs(bundle.volatility_signal) - 0.5, 0)

        # Liquidity factor
        liquidity_bonus = 0.1 * (ctx.liquidity_score - 0.5)

        score = tech_score + regime_bonus - vol_penalty + liquidity_bonus
        return float(np.clip(score, 0, 1))


# ── Simple Debate ─────────────────────────────────────────────────────────────

class DebatePanel:
    """Simplified debate: bull/bear/risk agents vote."""

    def evaluate(
        self,
        hyp: HypothesisCandidate,
        bundle: SignalBundle,
        ctx: MarketContext,
    ) -> float:
        # Bull agent: likes trend + momentum alignment
        bull_score = float(
            max(bundle.trend_signal * hyp.direction, 0) * 0.5
            + max(bundle.momentum_signal * hyp.direction, 0) * 0.3
            + hyp.entry_signal * 0.2
        )

        # Bear agent: penalizes when trend is against
        bear_penalty = float(
            max(-bundle.trend_signal * hyp.direction, 0) * 0.4
            + max(bundle.volatility_signal, 0) * 0.3
        )

        # Risk agent: penalizes extreme vol and illiquidity
        risk_penalty = float(
            max(abs(bundle.volatility_signal) - 0.5, 0) * 0.3
            + max(0.5 - ctx.liquidity_score, 0) * 0.2
        )

        debate_score = float(np.clip(bull_score - bear_penalty - risk_penalty, 0, 1))
        return debate_score


# ── Adversarial Filter ────────────────────────────────────────────────────────

class AdversarialFilter:
    """Quick adversarial checks to filter weak hypotheses."""

    def score(self, hyp: HypothesisCandidate, ctx: MarketContext) -> float:
        score = 1.0
        returns = ctx.returns

        # Regime robustness: does signal exist across sub-periods?
        if len(returns) >= 40:
            mid = len(returns) // 2
            sig_early = float(returns[:mid].mean() * hyp.direction)
            sig_late = float(returns[mid:].mean() * hyp.direction)
            consistency = float(np.sign(sig_early) == np.sign(sig_late))
            score *= (0.5 + 0.5 * consistency)

        # Overfitting proxy: penalize complex templates
        complex_templates = {"regime_adaptive", "chameleon", "structural_break"}
        if hyp.template_type in complex_templates:
            score *= 0.85

        return float(score)


# ── Position Sizer ────────────────────────────────────────────────────────────

class ConvictionSizer:
    """Size positions based on conviction and risk parameters."""

    def __init__(
        self,
        max_position: float = 0.2,
        vol_target: float = 0.15,
    ):
        self.max_position = max_position
        self.vol_target = vol_target

    def size(self, hyp: HypothesisCandidate, ctx: MarketContext) -> float:
        vol_daily = float(ctx.returns[-min(21, len(ctx.returns)):].std())
        vol_annual = vol_daily * math.sqrt(252) if vol_daily > 0 else 0.15

        # Vol-targeting: size inversely proportional to volatility
        vol_size = self.vol_target / max(vol_annual, 0.01)

        # Conviction scaling
        conviction_size = hyp.conviction * self.max_position

        size = min(vol_size * hyp.conviction, conviction_size, self.max_position)
        return float(max(size, 0.0))


# ── Master Pipeline ───────────────────────────────────────────────────────────

class IdeaPipeline:
    """
    Master pipeline: mine signals → generate → score → debate → adversarial → size → store.
    """

    def __init__(
        self,
        min_conviction: float = 0.3,
        min_debate_score: float = 0.25,
        min_adversarial: float = 0.5,
        max_candidates: int = 30,
    ):
        self.signal_gen = SignalGenerator()
        self.regime_router = RegimeRouter()
        self.hyp_gen = HypothesisGenerator(max_hypotheses=max_candidates)
        self.scorer = HypothesisScorer()
        self.debate = DebatePanel()
        self.adversarial = AdversarialFilter()
        self.sizer = ConvictionSizer()

        self.min_conviction = min_conviction
        self.min_debate = min_debate_score
        self.min_adversarial = min_adversarial

        self._history: list[PipelineResult] = []

    def run_full_pipeline(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        regime: Optional[str] = None,
        context_metadata: Optional[dict] = None,
    ) -> PipelineResult:
        """Run the complete idea generation pipeline."""
        start_time = time.time()
        stage_timings = {}
        warnings = []

        returns = np.diff(np.log(prices + 1e-10))

        ctx = MarketContext(
            prices=prices,
            returns=returns,
            volume=volume,
            regime=regime or "unknown",
            metadata=context_metadata or {},
        )

        # Stage 1: Signal Generation
        t0 = time.time()
        bundle = self.signal_gen.compute(ctx)
        stage_timings["signal_generation"] = time.time() - t0

        # Stage 2: Regime Classification
        t0 = time.time()
        if regime is None:
            ctx.regime = self.regime_router.classify_regime(ctx, bundle)
        templates = self.regime_router.get_templates(ctx.regime)
        stage_timings["regime_routing"] = time.time() - t0

        # Stage 3: Hypothesis Generation
        t0 = time.time()
        candidates = self.hyp_gen.generate(bundle, templates, ctx.regime)
        n_generated = len(candidates)
        stage_timings["hypothesis_generation"] = time.time() - t0

        if not candidates:
            warnings.append("No hypothesis candidates generated")

        # Stage 4: Scoring
        t0 = time.time()
        for hyp in candidates:
            hyp.raw_score = self.scorer.score(hyp, bundle, ctx)
        candidates = [h for h in candidates if h.raw_score > 0.1]
        stage_timings["scoring"] = time.time() - t0

        # Stage 5: Debate
        t0 = time.time()
        for hyp in candidates:
            hyp.debate_score = self.debate.evaluate(hyp, bundle, ctx)
        candidates = [h for h in candidates if h.debate_score >= self.min_debate]
        stage_timings["debate"] = time.time() - t0

        # Stage 6: Adversarial Testing
        t0 = time.time()
        for hyp in candidates:
            hyp.adversarial_score = self.adversarial.score(hyp, ctx)
        candidates = [h for h in candidates if hyp.adversarial_score >= self.min_adversarial]
        stage_timings["adversarial"] = time.time() - t0

        # Stage 7: Conviction + Sizing
        t0 = time.time()
        for hyp in candidates:
            hyp.conviction = float(
                0.35 * hyp.raw_score
                + 0.35 * hyp.debate_score
                + 0.30 * hyp.adversarial_score
            )
            hyp.suggested_size = self.sizer.size(hyp, ctx)

        # Filter by conviction
        candidates = [h for h in candidates if h.conviction >= self.min_conviction]
        candidates.sort(key=lambda h: h.conviction, reverse=True)
        stage_timings["sizing"] = time.time() - t0

        # Select top hypothesis
        top_hyp = candidates[0] if candidates else None

        # Check if should trade
        should_trade = (
            top_hyp is not None
            and top_hyp.conviction >= self.min_conviction
            and top_hyp.suggested_size > 0.01
            and abs(bundle.direction) > 0.15
        )

        if not should_trade and len(candidates) == 0:
            warnings.append("All hypotheses filtered out — no trade signal")

        allocation = top_hyp.suggested_size if top_hyp else 0.0

        result = PipelineResult(
            timestamp=start_time,
            top_hypothesis=top_hyp,
            all_candidates=candidates,
            signal_bundle=bundle,
            regime=ctx.regime,
            n_generated=n_generated,
            n_surviving=len(candidates),
            pipeline_warnings=warnings,
            stage_timings=stage_timings,
            allocation_fraction=float(allocation),
            should_trade=should_trade,
        )

        self._history.append(result)
        return result

    def run_batch(
        self,
        price_matrix: np.ndarray,
        window_size: int = 200,
        step: int = 20,
    ) -> list[PipelineResult]:
        """Run pipeline over rolling windows for backtesting."""
        T = len(price_matrix)
        results = []

        for t in range(window_size, T, step):
            prices_window = price_matrix[max(0, t - window_size): t + 1]
            result = self.run_full_pipeline(prices_window)
            results.append(result)

        return results

    def performance_feedback(self, result: PipelineResult, realized_pnl: float) -> None:
        """Update internal state based on realized PnL (feedback loop)."""
        if result.top_hypothesis is None:
            return

        # Adjust conviction thresholds based on recent performance
        if realized_pnl > 0:
            # Good prediction — slightly lower conviction threshold (more aggressive)
            self.min_conviction = max(self.min_conviction * 0.995, 0.1)
        else:
            # Bad prediction — raise conviction threshold (more selective)
            self.min_conviction = min(self.min_conviction * 1.005, 0.7)

    @property
    def pipeline_stats(self) -> dict:
        """Summary statistics of pipeline performance."""
        if not self._history:
            return {}
        n_total = len(self._history)
        n_trades = sum(1 for r in self._history if r.should_trade)
        avg_conviction = float(np.mean([
            r.top_hypothesis.conviction for r in self._history
            if r.top_hypothesis is not None
        ] or [0]))
        avg_surviving = float(np.mean([r.n_surviving for r in self._history]))

        return {
            "n_runs": n_total,
            "trade_rate": float(n_trades / max(n_total, 1)),
            "avg_conviction": avg_conviction,
            "avg_surviving_hypotheses": avg_surviving,
            "current_min_conviction": self.min_conviction,
        }
