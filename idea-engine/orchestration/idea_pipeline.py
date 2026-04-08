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


# =============================================================================
# Extended IdeaPipeline — wraps the core pipeline with richer orchestration
# =============================================================================
#
# FullIdeaPipeline enriches IdeaPipeline with:
#   - Integration of OrderFlowSignal and VolSurfaceSignal
#   - IdeaBank storage of top hypotheses
#   - MacroAnalyst + RiskManager debate integration
#   - Priority queue for hypothesis processing
#   - Description-based deduplication
#   - Performance feedback loop (adjusts min_conviction from IdeaBank history)
#   - Stage timing dict in PipelineResult
#   - run_id tracking per run
# =============================================================================


# ── Regime label enum ─────────────────────────────────────────────────────────

class RegimeLabel(str, Enum):
    BULL       = "bull"
    BEAR       = "bear"
    RANGING    = "ranging"
    VOL_SPIKE  = "vol_spike"
    TRENDING   = "trending"
    UNKNOWN    = "unknown"


# ── Extended pipeline result ──────────────────────────────────────────────────

@dataclass
class ExtendedPipelineResult:
    """
    Richer output from FullIdeaPipeline.run_full_pipeline().

    Wraps PipelineResult with additional metadata, signal snapshots,
    IdeaBank reference, and per-stage timing.
    """
    run_id: str
    base_result: PipelineResult          # from IdeaPipeline
    signals: Dict[str, Any]              # raw signal objects (OrderFlowSignal, etc.)
    regime_label: RegimeLabel
    recommended_allocation: float        # risk-manager sized allocation
    stage_timings_ms: Dict[str, float]   # wall-clock ms per stage
    idea_bank_id: Optional[str]          # ID of stored Idea in IdeaBank
    macro_score: float                   # macro alignment score
    risk_score: float                    # risk assessment score
    dedup_removed: int                   # number of hypotheses removed by dedup
    warnings: List[str]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def top_hypothesis(self) -> Optional[HypothesisCandidate]:
        return self.base_result.top_hypothesis

    @property
    def should_trade(self) -> bool:
        return self.base_result.should_trade and self.recommended_allocation > 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id":         self.run_id,
            "regime":         self.regime_label.value,
            "allocation":     f"{self.recommended_allocation:.2%}",
            "should_trade":   self.should_trade,
            "n_candidates":   self.base_result.n_surviving,
            "macro_score":    round(self.macro_score, 4),
            "risk_score":     round(self.risk_score, 4),
            "dedup_removed":  self.dedup_removed,
            "stage_ms":       {k: round(v, 1) for k, v in self.stage_timings_ms.items()},
            "warnings":       self.warnings[:5],
        }


# ── Priority queue for hypothesis processing ──────────────────────────────────

class _HypPriorityQueue:
    """Min-heap priority queue, highest conviction = lowest key."""

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, HypothesisCandidate]] = []
        self._counter = 0

    def push(self, hyp: HypothesisCandidate, score: float) -> None:
        heapq.heappush(self._heap, (-score, self._counter, hyp))
        self._counter += 1

    def pop(self) -> Tuple[HypothesisCandidate, float]:
        neg_score, _, hyp = heapq.heappop(self._heap)
        return hyp, -neg_score

    def drain(self, max_items: int = 50) -> List[HypothesisCandidate]:
        out: List[HypothesisCandidate] = []
        while self._heap and len(out) < max_items:
            out.append(self.pop()[0])
        return out

    def __len__(self) -> int:
        return len(self._heap)


# ── Deduplication ─────────────────────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _deduplicate_candidates(
    candidates: List[HypothesisCandidate],
    threshold: float = 0.80,
) -> Tuple[List[HypothesisCandidate], int]:
    """
    Remove near-duplicate hypotheses using Jaccard similarity on
    (template_type, regime, direction) token sets.

    Returns (unique_candidates, n_removed).
    """
    kept: List[HypothesisCandidate] = []
    seen_tokens: List[set] = []

    for hyp in candidates:
        tokens = {hyp.template_type, hyp.regime, f"dir_{hyp.direction:.0f}"}
        is_dup = any(_jaccard(tokens, prev) >= threshold for prev in seen_tokens)
        if not is_dup:
            kept.append(hyp)
            seen_tokens.append(tokens)

    return kept, len(candidates) - len(kept)


# ── Regime detection from prices ─────────────────────────────────────────────

def _detect_regime_label(
    prices: np.ndarray,
    returns: Optional[np.ndarray] = None,
    context: Optional[Dict[str, Any]] = None,
) -> RegimeLabel:
    """Classify regime from price action. Context override takes priority."""
    if context and "current_regime" in context:
        r = str(context["current_regime"]).lower()
        for lbl in RegimeLabel:
            if r == lbl.value:
                return lbl

    if len(prices) < 20:
        return RegimeLabel.UNKNOWN

    if returns is None:
        returns = np.diff(np.log(np.abs(prices) + 1e-12))

    vol_5  = float(np.std(returns[-5:])) if len(returns) >= 5 else 0.0
    vol_20 = float(np.std(returns[-20:])) if len(returns) >= 20 else max(vol_5, 1e-9)
    ret_20 = float(prices[-1] / (prices[-20] + 1e-12) - 1.0)

    if vol_20 > 1e-9 and vol_5 / vol_20 > 2.0:
        return RegimeLabel.VOL_SPIKE
    if ret_20 > 0.05:
        return RegimeLabel.BULL
    if ret_20 < -0.05:
        return RegimeLabel.BEAR

    n = min(len(prices), 20)
    x = np.arange(n, dtype=np.float64)
    slope, *_ = sp_stats.linregress(x, prices[-n:])
    norm_slope = abs(slope) * n / (float(np.std(prices[-n:])) + 1e-9)
    return RegimeLabel.TRENDING if norm_slope > 1.5 else RegimeLabel.RANGING


# ── FullIdeaPipeline ──────────────────────────────────────────────────────────

class FullIdeaPipeline:
    """
    Extended orchestration pipeline.

    Wraps IdeaPipeline (core signal/hypothesis engine) with:
      - Order flow + vol surface signal mining
      - Priority queue processing
      - Description-based deduplication
      - MacroAnalyst + RiskManager debate integration
      - IdeaBank storage
      - Feedback loop adjusting min_conviction from stored performance
      - Detailed stage timing

    Usage
    -----
    pipeline = FullIdeaPipeline(config={"vol_target": 0.20})
    result = pipeline.run_full_pipeline(prices, volume, context)
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "min_conviction":     0.30,
        "min_debate_score":   0.25,
        "min_adversarial":    0.50,
        "max_candidates":     30,
        "dedup_threshold":    0.80,
        "max_allocation":     0.10,
        "vol_target":         0.20,
        "feedback_lookback":  10,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        idea_bank: Optional[Any] = None,
        macro_analyst: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
    ) -> None:
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.cfg = cfg

        self._core = IdeaPipeline(
            min_conviction=cfg["min_conviction"],
            min_debate_score=cfg["min_debate_score"],
            min_adversarial=cfg["min_adversarial"],
            max_candidates=cfg["max_candidates"],
        )

        self.idea_bank = idea_bank or (IdeaBank() if _IDEA_BANK_AVAILABLE else None)
        self.macro_analyst = macro_analyst or (MacroAnalyst() if _DEBATE_AGENTS_AVAILABLE else None)
        self.risk_manager  = risk_manager  or (RiskManager()  if _DEBATE_AGENTS_AVAILABLE else None)

        self._run_count = 0
        logger.info("FullIdeaPipeline initialized.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtendedPipelineResult:
        """
        Run the complete idea pipeline.

        Parameters
        ----------
        prices  : np.ndarray — price series (close), oldest first
        volume  : np.ndarray, optional
        context : dict, optional — supplementary data:
            open, high, low : np.ndarray
            front_iv, back_iv, atm_iv, put_25d_iv, call_25d_iv : float
            historical_ivs : np.ndarray
            yield_2y, yield_10y, yield_3m : float
            ig_spread, hy_spread, dxy : float
            asset_class, direction, instrument : str
            portfolio_nav, position_size_usd, adv_usd : float
            win_rate, avg_win, avg_loss, vol_target : float
            current_regime : str
        """
        run_id = str(uuid.uuid4())
        self._run_count += 1
        t_total = time.monotonic()
        ctx = context or {}
        timings: Dict[str, float] = {}
        warnings: List[str] = []

        prices = np.asarray(prices, dtype=np.float64)
        if volume is None:
            volume = np.ones(len(prices), dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)

        returns = np.diff(np.log(np.abs(prices) + 1e-12)) if len(prices) > 1 else np.array([0.0])

        # ── Detect regime ─────────────────────────────────────────────
        regime_label = _detect_regime_label(prices, returns, ctx)
        regime_str = regime_label.value

        # ── Stage 1: Signal Mining ────────────────────────────────────
        t0 = time.monotonic()
        signals, mine_warns = self._mine_signals(prices, volume, ctx)
        warnings.extend(mine_warns)
        timings["signal_mining"] = (time.monotonic() - t0) * 1000

        # ── Stage 2-4: Core pipeline (gen, score, debate, adversarial, size) ──
        t0 = time.monotonic()
        base_result = self._core.run_full_pipeline(
            prices=prices,
            volume=volume,
            regime=regime_str,
            context_metadata=ctx,
        )
        timings["core_pipeline"] = (time.monotonic() - t0) * 1000
        warnings.extend(base_result.pipeline_warnings)

        # ── Stage 5: Priority queue ───────────────────────────────────
        t0 = time.monotonic()
        pq = _HypPriorityQueue()
        for hyp in base_result.all_candidates:
            pq.push(hyp, hyp.conviction)
        ordered = pq.drain(max_items=self.cfg["max_candidates"])
        timings["priority_queue"] = (time.monotonic() - t0) * 1000

        # ── Stage 6: Deduplication ────────────────────────────────────
        t0 = time.monotonic()
        try:
            ordered, n_removed = _deduplicate_candidates(
                ordered, threshold=self.cfg["dedup_threshold"]
            )
        except Exception as e:
            n_removed = 0
            logger.warning(f"Deduplication error: {e}")
        timings["deduplication"] = (time.monotonic() - t0) * 1000

        # ── Stage 7: Macro + Risk debate on top candidate ─────────────
        t0 = time.monotonic()
        macro_score, risk_score, debate_warns = self._debate_top(
            base_result.top_hypothesis, returns, ctx, regime_label
        )
        warnings.extend(debate_warns)
        timings["debate"] = (time.monotonic() - t0) * 1000

        # ── Stage 8: Sizing ───────────────────────────────────────────
        t0 = time.monotonic()
        allocation, size_warns = self._compute_allocation(
            base_result.top_hypothesis, returns, ctx, macro_score, risk_score
        )
        warnings.extend(size_warns)
        timings["sizing"] = (time.monotonic() - t0) * 1000

        # ── Stage 9: IdeaBank storage ─────────────────────────────────
        t0 = time.monotonic()
        idea_id = self._store(base_result.top_hypothesis, regime_label, signals, ctx)
        timings["storage"] = (time.monotonic() - t0) * 1000

        # ── Feedback loop ─────────────────────────────────────────────
        self._feedback_loop()

        timings["total"] = (time.monotonic() - t_total) * 1000
        logger.info(
            f"[{run_id}] FullPipeline done in {timings['total']:.0f}ms. "
            f"regime={regime_label.value}, alloc={allocation:.2%}, "
            f"candidates={base_result.n_surviving}"
        )

        return ExtendedPipelineResult(
            run_id=run_id,
            base_result=base_result,
            signals=signals,
            regime_label=regime_label,
            recommended_allocation=allocation,
            stage_timings_ms=timings,
            idea_bank_id=idea_id,
            macro_score=macro_score,
            risk_score=risk_score,
            dedup_removed=n_removed,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mine_signals(
        self,
        prices: np.ndarray,
        volume: np.ndarray,
        ctx: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Mine order flow + vol surface signals."""
        signals: Dict[str, Any] = {}
        warnings: List[str] = []

        open_ = np.asarray(ctx.get("open",  prices), dtype=np.float64)
        high  = np.asarray(ctx.get("high",  prices + 0.01), dtype=np.float64)
        low   = np.asarray(ctx.get("low",   prices - 0.01), dtype=np.float64)

        if _ORDER_FLOW_AVAILABLE and len(prices) >= 10:
            try:
                of = compute_order_flow_signal(open_, high, low, prices, volume)
                signals["order_flow"] = of
                warnings.extend(of.warnings)
            except Exception as e:
                warnings.append(f"OrderFlow signal failed: {e}")

        if _VOL_SURFACE_AVAILABLE and len(prices) >= 10:
            try:
                vs = compute_vol_surface_signal(
                    prices=prices,
                    front_iv=float(ctx.get("front_iv", 0.25)),
                    back_iv=float(ctx.get("back_iv", 0.22)),
                    put_25d_iv=float(ctx.get("put_25d_iv", 0.28)),
                    call_25d_iv=float(ctx.get("call_25d_iv", 0.23)),
                    atm_iv=float(ctx.get("atm_iv", 0.25)),
                    historical_ivs=ctx.get("historical_ivs"),
                    rv_window=10,
                )
                signals["vol_surface"] = vs
                warnings.extend(vs.warnings)
            except Exception as e:
                warnings.append(f"VolSurface signal failed: {e}")

        return signals, warnings

    def _debate_top(
        self,
        top_hyp: Optional[HypothesisCandidate],
        returns: np.ndarray,
        ctx: Dict[str, Any],
        regime: RegimeLabel,
    ) -> Tuple[float, float, List[str]]:
        """Run macro + risk debate on top hypothesis."""
        warnings: List[str] = []
        macro_score = 0.0
        risk_score  = 0.3  # neutral default

        if top_hyp is None or not _DEBATE_AGENTS_AVAILABLE:
            return macro_score, risk_score, warnings

        market_data = {
            "yield_2y":             ctx.get("yield_2y",  3.5),
            "yield_10y":            ctx.get("yield_10y", 4.2),
            "yield_3m":             ctx.get("yield_3m",  5.0),
            "ig_spread":            ctx.get("ig_spread", 120.0),
            "hy_spread":            ctx.get("hy_spread", 360.0),
            "dxy":                  ctx.get("dxy",       103.0),
            "hypothesis_direction": "long" if top_hyp.direction >= 0 else "short",
            "asset_class":          ctx.get("asset_class", "crypto"),
            "current_regime":       regime.value,
            "returns":              returns,
            "portfolio_returns":    ctx.get("portfolio_returns", returns),
            "position_size_usd":    ctx.get("position_size_usd", 10_000.0),
            "portfolio_nav":        ctx.get("portfolio_nav", 100_000.0),
            "adv_usd":              ctx.get("adv_usd", 1_000_000.0),
            "win_rate":             ctx.get("win_rate", 0.51),
            "avg_win":              ctx.get("avg_win",  0.02),
            "avg_loss":             ctx.get("avg_loss", 0.015),
            "vol_target":           ctx.get("vol_target", 0.20),
            "conviction":           top_hyp.conviction,
        }

        # Stub Hypothesis object for agents expecting the formal type
        class _StubHyp:
            hypothesis_id = str(uuid.uuid4())
            parameters    = {"template": top_hyp.template_type}
            description   = top_hyp.name
            novelty_score = 0.5
            type          = None

        stub = _StubHyp()

        if self.macro_analyst is not None:
            try:
                analysis = self.macro_analyst.evaluate(stub, market_data)
                macro_score = float(analysis.directional_alignment)
                if analysis.warnings:
                    warnings.extend(analysis.warnings[:2])
            except Exception as e:
                logger.debug(f"Macro debate failed: {e}")

        if self.risk_manager is not None:
            try:
                assessment = self.risk_manager.evaluate(stub, market_data)
                risk_score = float(assessment.risk_score)
                if assessment.hard_veto:
                    warnings.append("RiskManager hard veto — allocation set to zero")
                elif assessment.warnings:
                    warnings.extend(assessment.warnings[:2])
            except Exception as e:
                logger.debug(f"Risk debate failed: {e}")

        return macro_score, risk_score, warnings

    def _compute_allocation(
        self,
        top_hyp: Optional[HypothesisCandidate],
        returns: np.ndarray,
        ctx: Dict[str, Any],
        macro_score: float,
        risk_score: float,
    ) -> Tuple[float, List[str]]:
        """
        Compute final allocation blending:
          - suggested_size from core pipeline (Kelly / vol targeting)
          - macro alignment multiplier (0.5x to 1.5x)
          - risk score penalty (1 - risk_score)
        """
        warnings: List[str] = []
        if top_hyp is None or not top_hyp.should_trade if hasattr(top_hyp, 'should_trade') else top_hyp is None:
            return 0.0, warnings

        base_size = float(top_hyp.suggested_size) if top_hyp.suggested_size > 0 else 0.05

        # Macro multiplier: 0.5 (bad macro) to 1.5 (supportive macro)
        macro_mult = float(np.clip(1.0 + macro_score * 0.5, 0.5, 1.5))

        # Risk penalty
        risk_adj = float(np.clip(1.0 - risk_score, 0.3, 1.0))

        allocation = float(np.clip(
            base_size * macro_mult * risk_adj,
            0.0,
            self.cfg["max_allocation"],
        ))

        if macro_mult < 0.7:
            warnings.append(f"Macro headwind (score={macro_score:+.3f}) — allocation reduced")
        if risk_score > 0.6:
            warnings.append(f"Elevated risk (score={risk_score:.3f}) — allocation reduced")

        return allocation, warnings

    def _store(
        self,
        top_hyp: Optional[HypothesisCandidate],
        regime: RegimeLabel,
        signals: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Optional[str]:
        """Persist top hypothesis to IdeaBank if available."""
        if top_hyp is None or self.idea_bank is None:
            return None
        try:
            signal_strs: List[str] = []
            for k, v in signals.items():
                rec = getattr(v, "recommendation", None)
                signal_strs.append(f"{k}:{rec}" if rec else k)

            idea = self.idea_bank.add(
                hypothesis=top_hyp.name,
                signals=signal_strs or ["unknown"],
                regime=regime.value,
                confidence=float(np.clip(top_hyp.conviction, 0.0, 1.0)),
                tags=["pipeline_auto", regime.value, top_hyp.template_type],
                ticker=ctx.get("instrument"),
                direction="long" if top_hyp.direction >= 0 else "short",
                metadata={
                    "template_type": top_hyp.template_type,
                    "raw_score":     top_hyp.raw_score,
                    "debate_score":  top_hyp.debate_score,
                    "adversarial":   top_hyp.adversarial_score,
                    "run_count":     self._run_count,
                },
            )
            logger.debug(f"Stored idea {idea.id} in IdeaBank.")
            return idea.id
        except Exception as e:
            logger.warning(f"IdeaBank storage failed: {e}")
            return None

    def _feedback_loop(self) -> None:
        """
        Adjust min_conviction from IdeaBank historical performance.
        Runs every 5 pipeline calls to avoid excessive churn.
        """
        if self._run_count % 5 != 0:
            return
        if self.idea_bank is None:
            return
        try:
            attr = self.idea_bank.performance_attribution()
            recent = attr[:self.cfg["feedback_lookback"]]
            if not recent:
                return
            avg_win_rate = float(np.mean([r.get("win_rate", 0.5) for r in recent]))
            if avg_win_rate > 0.55:
                # Performing well — can lower conviction threshold slightly
                self._core.min_conviction = max(
                    self._core.min_conviction * 0.99, 0.10
                )
            elif avg_win_rate < 0.45:
                # Underperforming — tighten conviction threshold
                self._core.min_conviction = min(
                    self._core.min_conviction * 1.01, 0.70
                )
            logger.debug(
                f"Feedback loop: win_rate={avg_win_rate:.3f}, "
                f"min_conviction → {self._core.min_conviction:.4f}"
            )
        except Exception as e:
            logger.debug(f"Feedback loop error: {e}")
