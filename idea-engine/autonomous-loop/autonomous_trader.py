"""
Autonomous trading loop — master orchestrator for the idea engine.

Implements the full autonomous cycle:
  1. Market scan: gather data, detect regime, assess liquidity
  2. Signal generation: run all signal modules
  3. Idea synthesis: combine signals into trade ideas
  4. Debate: multi-agent evaluation of top ideas
  5. Validation: hypothesis testing, backtest, adversarial tests
  6. Risk check: portfolio risk, concentration, VaR limits
  7. Execution: sizing, timing, venue selection
  8. Monitoring: track P&L, decay, regime changes
  9. Learning: update models, evolve hypotheses, self-improve
  10. Reporting: generate human-readable reports

Each step is a pluggable module. The loop runs continuously or on schedule.
"""

from __future__ import annotations
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Any


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class LoopConfig:
    """Configuration for the autonomous trading loop."""
    # Timing
    run_interval_seconds: float = 300.0    # 5 minutes
    market_scan_lookback: int = 252        # days

    # Filtering
    min_conviction: float = 0.40
    min_debate_score: float = 0.50
    min_validation_score: float = 50.0

    # Risk
    max_portfolio_var_95: float = 0.03     # 3% daily VaR limit
    max_position_pct: float = 0.10         # 10% max single position
    max_correlation_to_book: float = 0.70  # diversification limit
    max_drawdown_halt: float = 0.15        # 15% DD halts trading

    # Sizing
    base_position_size: float = 0.05       # 5% base position
    kelly_fraction: float = 0.25           # quarter Kelly

    # Learning
    evolve_every_n_runs: int = 20
    decay_check_every_n_runs: int = 5
    min_runs_before_evolution: int = 50

    # Execution
    max_participation_rate: float = 0.05
    urgency_default: float = 0.5


# ── Pipeline Stage Results ────────────────────────────────────────────────────

@dataclass
class MarketScanResult:
    regime: str
    regime_confidence: float
    vol_regime: str
    liquidity_regime: str
    trend_direction: float
    n_signals_available: int
    data_quality_score: float


@dataclass
class IdeaCandidate:
    idea_id: str
    name: str
    direction: float           # +1/-1
    conviction: float
    signal_strength: float
    domain_alignment: float
    template_type: str
    regime: str
    thesis: str
    key_factors: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)


@dataclass
class DebateResult:
    idea_id: str
    consensus_score: float      # 0-1
    consensus_direction: float  # +1/-1/0
    n_supporters: int
    n_opponents: int
    risk_veto: bool
    key_arguments_for: list[str] = field(default_factory=list)
    key_arguments_against: list[str] = field(default_factory=list)
    execution_strategy: str = "TWAP"
    recommended_size: float = 0.0


@dataclass
class ValidationResult:
    idea_id: str
    score: float               # 0-100
    verdict: str               # PASS / CONDITIONAL_PASS / FAIL
    sharpe_is: float
    sharpe_oos: float
    is_overfit: bool
    is_regime_robust: bool
    cost_robust: bool
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class RiskCheckResult:
    approved: bool
    var_impact: float
    concentration_ok: bool
    correlation_ok: bool
    drawdown_ok: bool
    position_size: float       # approved size after risk adjustment
    risk_adjusted_size: float
    reason: str = ""


@dataclass
class ExecutionPlan:
    idea_id: str
    direction: float
    size: float
    strategy: str              # TWAP / VWAP / IS / aggressive
    execution_horizon_minutes: int
    expected_cost_bps: float
    venue: str                 # lit / dark / mixed


@dataclass
class LoopIteration:
    """Complete record of one loop iteration."""
    run_id: int
    timestamp: float
    market_scan: MarketScanResult
    n_candidates: int
    n_after_debate: int
    n_after_validation: int
    n_approved: int
    execution_plans: list[ExecutionPlan]
    portfolio_var: float
    portfolio_dd: float
    elapsed_seconds: float
    status: str                # success / halted / no_ideas / risk_halt


# ── Pipeline Stages (Abstract) ────────────────────────────────────────────────

class MarketScanner:
    """Stage 1: Scan market conditions."""

    def scan(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> MarketScanResult:
        n = len(returns)
        # Simple regime detection
        recent_vol = float(returns[-21:].std() * math.sqrt(252)) if n >= 21 else 0.15
        recent_ret = float(returns[-21:].mean() * 252) if n >= 21 else 0.0
        trend = float(np.sign(recent_ret)) if abs(recent_ret) > 0.05 else 0.0

        if recent_vol > 0.35:
            regime = "crisis"
        elif recent_vol > 0.25:
            regime = "high_volatility"
        elif trend > 0:
            regime = "trending_up"
        elif trend < 0:
            regime = "trending_down"
        else:
            regime = "mean_reverting"

        vol_regime = "crisis" if recent_vol > 0.35 else "high" if recent_vol > 0.20 else "normal" if recent_vol > 0.10 else "low"

        # Data quality: fraction of non-nan data
        data_quality = float(1 - np.isnan(returns).mean()) if n > 0 else 0.0

        return MarketScanResult(
            regime=regime,
            regime_confidence=0.7,
            vol_regime=vol_regime,
            liquidity_regime="normal",
            trend_direction=trend,
            n_signals_available=10,
            data_quality_score=data_quality,
        )


class SignalAggregator:
    """Stage 2: Generate and aggregate signals into candidates."""

    def __init__(self):
        self._signal_generators: list[Callable] = []
        self._id_counter = 0

    def register_signal(self, generator: Callable) -> None:
        self._signal_generators.append(generator)

    def generate_candidates(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
        regime: str,
        min_conviction: float = 0.3,
    ) -> list[IdeaCandidate]:
        """Run all signal generators, synthesize into candidates."""
        candidates = []

        # Simple built-in signals as fallback
        n = len(returns)
        if n < 30:
            return candidates

        # Momentum signal
        mom_12m = float(prices[-1] / max(prices[-min(252, n)], 1e-10) - 1) if n >= 252 else 0.0
        mom_1m = float(prices[-1] / max(prices[-min(21, n)], 1e-10) - 1) if n >= 21 else 0.0
        mom_signal = 0.7 * mom_12m + 0.3 * mom_1m

        if abs(mom_signal) > 0.05:
            self._id_counter += 1
            direction = float(np.sign(mom_signal))
            conviction = float(min(abs(mom_signal) * 2, 1.0))
            if conviction >= min_conviction:
                candidates.append(IdeaCandidate(
                    idea_id=f"idea_{self._id_counter:05d}",
                    name=f"{'Long' if direction > 0 else 'Short'} Momentum [{regime}]",
                    direction=direction,
                    conviction=conviction,
                    signal_strength=float(abs(mom_signal)),
                    domain_alignment=0.7,
                    template_type="momentum",
                    regime=regime,
                    thesis=f"12m momentum {mom_12m:.1%}, 1m momentum {mom_1m:.1%}",
                    key_factors=[f"12m return: {mom_12m:.1%}", f"1m return: {mom_1m:.1%}"],
                ))

        # Mean reversion signal
        if n >= 63:
            zscore_63 = float((prices[-1] - prices[-63:].mean()) / max(prices[-63:].std(), 1e-10))
            if abs(zscore_63) > 1.5:
                self._id_counter += 1
                direction = -float(np.sign(zscore_63))
                conviction = float(min(abs(zscore_63) / 3, 1.0))
                if conviction >= min_conviction:
                    candidates.append(IdeaCandidate(
                        idea_id=f"idea_{self._id_counter:05d}",
                        name=f"{'Long' if direction > 0 else 'Short'} Mean Reversion [{regime}]",
                        direction=direction,
                        conviction=conviction,
                        signal_strength=float(abs(zscore_63) / 3),
                        domain_alignment=0.6,
                        template_type="mean_reversion",
                        regime=regime,
                        thesis=f"63-day z-score: {zscore_63:.2f}",
                        key_factors=[f"Z-score: {zscore_63:.2f}"],
                    ))

        # Volatility breakout
        if n >= 42:
            recent_vol = float(returns[-21:].std())
            prior_vol = float(returns[-42:-21].std())
            if recent_vol > prior_vol * 1.5 and returns[-5:].mean() > 0:
                self._id_counter += 1
                candidates.append(IdeaCandidate(
                    idea_id=f"idea_{self._id_counter:05d}",
                    name=f"Long Vol Breakout [{regime}]",
                    direction=1.0,
                    conviction=0.5,
                    signal_strength=float(recent_vol / max(prior_vol, 1e-10)),
                    domain_alignment=0.5,
                    template_type="volatility_breakout",
                    regime=regime,
                    thesis=f"Vol expanded {recent_vol/max(prior_vol, 1e-10):.1f}x with positive price momentum",
                ))

        # Sort by conviction
        candidates.sort(key=lambda c: c.conviction, reverse=True)
        return candidates[:10]  # top 10


class DebateEngine:
    """Stage 4: Multi-agent debate on candidates."""

    def evaluate(self, candidate: IdeaCandidate, regime: str) -> DebateResult:
        """Simplified debate scoring."""
        # Base score from conviction and alignment
        score = candidate.conviction * 0.5 + candidate.domain_alignment * 0.3

        # Regime penalty: momentum in mean-reverting = bad
        regime_fit = 1.0
        if candidate.template_type == "momentum" and regime == "mean_reverting":
            regime_fit = 0.5
        elif candidate.template_type == "mean_reversion" and regime in ("trending_up", "trending_down"):
            regime_fit = 0.5
        elif candidate.template_type == "momentum" and regime in ("trending_up", "trending_down"):
            regime_fit = 1.2

        score *= regime_fit

        # Risk veto: very high conviction in crisis
        risk_veto = bool(regime == "crisis" and candidate.direction > 0 and candidate.template_type == "momentum")

        n_supporters = 4 if score > 0.6 else 3 if score > 0.4 else 2
        n_opponents = 5 - n_supporters

        return DebateResult(
            idea_id=candidate.idea_id,
            consensus_score=float(min(score, 1.0)),
            consensus_direction=candidate.direction,
            n_supporters=n_supporters,
            n_opponents=n_opponents,
            risk_veto=risk_veto,
            key_arguments_for=candidate.key_factors[:2],
            key_arguments_against=candidate.risk_factors[:2],
            execution_strategy="TWAP",
            recommended_size=0.05 * score,
        )


class RiskGate:
    """Stage 6: Risk checking and position sizing."""

    def __init__(self, config: LoopConfig):
        self.config = config
        self._current_var: float = 0.0
        self._current_dd: float = 0.0

    def check(
        self,
        candidate: IdeaCandidate,
        debate: DebateResult,
        validation: Optional[ValidationResult] = None,
    ) -> RiskCheckResult:
        # Drawdown halt
        if self._current_dd >= self.config.max_drawdown_halt:
            return RiskCheckResult(
                approved=False, var_impact=0.0, concentration_ok=True,
                correlation_ok=True, drawdown_ok=False, position_size=0.0,
                risk_adjusted_size=0.0, reason="Drawdown halt triggered"
            )

        # Risk veto from debate
        if debate.risk_veto:
            return RiskCheckResult(
                approved=False, var_impact=0.0, concentration_ok=True,
                correlation_ok=True, drawdown_ok=True, position_size=0.0,
                risk_adjusted_size=0.0, reason="Risk manager veto in debate"
            )

        # Base sizing: quarter-Kelly
        win_rate = 0.5 + candidate.conviction * 0.15
        avg_win = candidate.signal_strength * 0.02
        avg_loss = avg_win * 0.8
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / max(avg_win, 1e-10)
        kelly_size = max(kelly * self.config.kelly_fraction, 0.01)

        # Cap at max position
        position_size = min(kelly_size, self.config.max_position_pct)

        # Risk-adjust for regime
        if candidate.regime == "crisis":
            position_size *= 0.3
        elif candidate.regime == "high_volatility":
            position_size *= 0.5

        # Debate score adjustment
        position_size *= debate.consensus_score

        return RiskCheckResult(
            approved=True,
            var_impact=position_size * 0.02,
            concentration_ok=True,
            correlation_ok=True,
            drawdown_ok=True,
            position_size=float(position_size),
            risk_adjusted_size=float(position_size),
        )

    def update_portfolio_state(self, var: float, dd: float) -> None:
        self._current_var = var
        self._current_dd = dd


class ExecutionPlanner:
    """Stage 7: Plan execution."""

    def plan(
        self,
        candidate: IdeaCandidate,
        risk_result: RiskCheckResult,
        debate: DebateResult,
        daily_volume: float = 1e6,
    ) -> ExecutionPlan:
        # Participation constraint
        notional = risk_result.position_size * 1e6  # assume $1M portfolio for scaling
        participation = notional / max(daily_volume, 1e-10)

        if participation > 0.05:
            strategy = "TWAP"
            horizon = int(min(participation / 0.03 * 60, 390))  # minutes in trading day
        else:
            strategy = debate.execution_strategy
            horizon = 30

        expected_cost = float(math.sqrt(participation) * 10 * 100)  # bps

        return ExecutionPlan(
            idea_id=candidate.idea_id,
            direction=candidate.direction,
            size=risk_result.risk_adjusted_size,
            strategy=strategy,
            execution_horizon_minutes=horizon,
            expected_cost_bps=expected_cost,
            venue="mixed" if participation > 0.02 else "lit",
        )


# ── Autonomous Loop ───────────────────────────────────────────────────────────

class AutonomousTrader:
    """
    Master autonomous trading loop orchestrator.
    Connects all idea-engine modules into a continuous cycle.
    """

    def __init__(self, config: LoopConfig = None):
        self.config = config or LoopConfig()
        self.scanner = MarketScanner()
        self.signal_agg = SignalAggregator()
        self.debate = DebateEngine()
        self.risk_gate = RiskGate(self.config)
        self.execution_planner = ExecutionPlanner()

        self._run_count = 0
        self._history: list[LoopIteration] = []
        self._active_positions: list[dict] = []

    def run_once(
        self,
        returns: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> LoopIteration:
        """Execute one full loop iteration."""
        start = time.time()
        self._run_count += 1

        # Stage 1: Market Scan
        scan = self.scanner.scan(returns, prices, volumes)

        # Stage 2-3: Signal Generation + Idea Synthesis
        candidates = self.signal_agg.generate_candidates(
            returns, prices, scan.regime, self.config.min_conviction
        )

        # Stage 4: Debate
        debate_results = []
        for c in candidates:
            dr = self.debate.evaluate(c, scan.regime)
            if dr.consensus_score >= self.config.min_debate_score and not dr.risk_veto:
                debate_results.append((c, dr))

        # Stage 5: Validation (simplified — full version uses HypothesisValidator)
        validated = []
        for c, dr in debate_results:
            # Quick validation: check if signal has any predictive power
            if c.signal_strength > 0.05 and c.domain_alignment > 0.3:
                validated.append((c, dr))

        # Stage 6: Risk Check
        approved = []
        for c, dr in validated:
            risk_result = self.risk_gate.check(c, dr)
            if risk_result.approved:
                approved.append((c, dr, risk_result))

        # Stage 7: Execution Planning
        plans = []
        for c, dr, rr in approved:
            plan = self.execution_planner.plan(c, rr, dr)
            plans.append(plan)

        # Portfolio state update
        if len(returns) >= 21:
            port_vol = float(returns[-21:].std() * math.sqrt(252))
            eq = np.cumprod(1 + returns[-63:])
            dd = float(1 - eq[-1] / eq.max())
        else:
            port_vol = 0.15
            dd = 0.0
        self.risk_gate.update_portfolio_state(port_vol * 1.645 / math.sqrt(252), dd)

        elapsed = time.time() - start

        status = "success" if plans else "no_ideas" if not candidates else "filtered"
        if dd >= self.config.max_drawdown_halt:
            status = "risk_halt"

        iteration = LoopIteration(
            run_id=self._run_count,
            timestamp=time.time(),
            market_scan=scan,
            n_candidates=len(candidates),
            n_after_debate=len(debate_results),
            n_after_validation=len(validated),
            n_approved=len(approved),
            execution_plans=plans,
            portfolio_var=port_vol * 1.645 / math.sqrt(252),
            portfolio_dd=dd,
            elapsed_seconds=elapsed,
            status=status,
        )

        self._history.append(iteration)

        # Stage 9: Learning (periodic)
        if self._run_count % self.config.evolve_every_n_runs == 0:
            self._evolve_step()

        return iteration

    def _evolve_step(self) -> None:
        """Periodic self-improvement step."""
        # Analyze recent performance
        recent = self._history[-self.config.evolve_every_n_runs:]
        success_rate = sum(1 for r in recent if r.n_approved > 0) / max(len(recent), 1)

        # Adaptive thresholds
        if success_rate < 0.1:
            # Too few ideas getting through — lower thresholds
            self.config.min_conviction = max(self.config.min_conviction - 0.05, 0.20)
            self.config.min_debate_score = max(self.config.min_debate_score - 0.05, 0.30)
        elif success_rate > 0.5:
            # Too many ideas — raise thresholds
            self.config.min_conviction = min(self.config.min_conviction + 0.02, 0.70)
            self.config.min_debate_score = min(self.config.min_debate_score + 0.02, 0.80)

    def performance_report(self) -> dict:
        """Generate performance report from history."""
        if not self._history:
            return {"n_runs": 0}

        n = len(self._history)
        n_with_ideas = sum(1 for h in self._history if h.n_approved > 0)
        avg_candidates = float(np.mean([h.n_candidates for h in self._history]))
        avg_approved = float(np.mean([h.n_approved for h in self._history]))
        avg_elapsed = float(np.mean([h.elapsed_seconds for h in self._history]))

        # Regime distribution
        regime_counts: dict[str, int] = {}
        for h in self._history:
            r = h.market_scan.regime
            regime_counts[r] = regime_counts.get(r, 0) + 1

        return {
            "n_runs": n,
            "n_with_trades": n_with_ideas,
            "trade_rate": float(n_with_ideas / n),
            "avg_candidates_per_run": avg_candidates,
            "avg_approved_per_run": avg_approved,
            "avg_elapsed_seconds": avg_elapsed,
            "current_conviction_threshold": self.config.min_conviction,
            "current_debate_threshold": self.config.min_debate_score,
            "regime_distribution": regime_counts,
            "latest_var": float(self._history[-1].portfolio_var) if self._history else 0.0,
            "latest_dd": float(self._history[-1].portfolio_dd) if self._history else 0.0,
        }
