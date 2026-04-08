"""
feedback_loop.py
================
Real-time feedback and adaptation loop for the idea-engine.

Accumulates trade outcomes, computes rolling statistics, triggers model
retraining, performs online parameter adaptation, Bayesian strategy scoring,
regime feedback, signal quality tracking, alert generation, and dashboard
data assembly.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    DRAWDOWN = "drawdown"
    ALPHA_DECAY = "alpha_decay"
    PERF_DEGRADATION = "perf_degradation"
    REGIME_MISMATCH = "regime_mismatch"
    SIGNAL_DECAY = "signal_decay"
    MODEL_DRIFT = "model_drift"


class RetrainTrigger(Enum):
    PERFORMANCE = "performance_degradation"
    REGIME_CHANGE = "regime_change"
    DRIFT = "feature_drift"
    SCHEDULED = "scheduled"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeOutcome:
    """Complete record of a single trade and its attribution."""

    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str = ""
    direction: str = "long"
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    strategy_id: str = ""
    signal_id: str = ""
    regime_at_entry: str = ""
    holding_period_seconds: float = 0.0
    attribution: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.net_pnl == 0.0 and self.realized_pnl != 0.0:
            self.net_pnl = self.realized_pnl - self.commission - self.slippage
        if self.exit_time and self.entry_time:
            self.holding_period_seconds = (self.exit_time - self.entry_time).total_seconds()

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0 or self.quantity == 0:
            return 0.0
        notional = abs(self.entry_price * self.quantity)
        return self.net_pnl / notional if notional > 0 else 0.0

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0


@dataclass
class Alert:
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    alert_type: AlertType = AlertType.PERF_DEGRADATION
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class DashboardData:
    equity_curve: List[float] = field(default_factory=list)
    rolling_sharpe: List[float] = field(default_factory=list)
    rolling_win_rate: List[float] = field(default_factory=list)
    drawdown_series: List[float] = field(default_factory=list)
    regime_history: List[Tuple[datetime, str]] = field(default_factory=list)
    signal_ic_history: Dict[str, List[float]] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    strategy_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Rolling statistics helper
# ---------------------------------------------------------------------------

class RollingStats:
    """Welford online mean/var + rolling window."""

    def __init__(self, window: int = 100):
        self.window = window
        self._values: Deque[float] = deque(maxlen=window)
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self._values.append(x)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return float(np.mean(list(self._values)))

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        return float(np.std(list(self._values), ddof=1))

    @property
    def sharpe(self) -> float:
        s = self.std
        return self.mean / s if s > 1e-10 else 0.0

    @property
    def count(self) -> int:
        return len(self._values)

    def percentile(self, q: float) -> float:
        if not self._values:
            return 0.0
        return float(np.percentile(list(self._values), q))

    def as_array(self) -> np.ndarray:
        return np.array(list(self._values))


# ---------------------------------------------------------------------------
# Feedback collector
# ---------------------------------------------------------------------------

class FeedbackCollector:
    """Accumulate trade outcomes and compute aggregate statistics."""

    def __init__(self, window: int = 200):
        self.outcomes: List[TradeOutcome] = []
        self.pnl_stats = RollingStats(window)
        self.return_stats = RollingStats(window)
        self._by_strategy: Dict[str, RollingStats] = {}
        self._by_signal: Dict[str, RollingStats] = {}

    def add(self, outcome: TradeOutcome) -> None:
        self.outcomes.append(outcome)
        self.pnl_stats.update(outcome.net_pnl)
        self.return_stats.update(outcome.return_pct)
        if outcome.strategy_id:
            if outcome.strategy_id not in self._by_strategy:
                self._by_strategy[outcome.strategy_id] = RollingStats()
            self._by_strategy[outcome.strategy_id].update(outcome.net_pnl)
        if outcome.signal_id:
            if outcome.signal_id not in self._by_signal:
                self._by_signal[outcome.signal_id] = RollingStats()
            self._by_signal[outcome.signal_id].update(outcome.net_pnl)

    def total_pnl(self) -> float:
        return sum(o.net_pnl for o in self.outcomes)

    def win_rate(self, last_n: Optional[int] = None) -> float:
        subset = self.outcomes[-last_n:] if last_n else self.outcomes
        if not subset:
            return 0.0
        return sum(1 for o in subset if o.is_winner) / len(subset)

    def avg_winner(self) -> float:
        winners = [o.net_pnl for o in self.outcomes if o.is_winner]
        return float(np.mean(winners)) if winners else 0.0

    def avg_loser(self) -> float:
        losers = [o.net_pnl for o in self.outcomes if not o.is_winner]
        return float(np.mean(losers)) if losers else 0.0

    def profit_factor(self) -> float:
        gross_profit = sum(o.net_pnl for o in self.outcomes if o.net_pnl > 0)
        gross_loss = abs(sum(o.net_pnl for o in self.outcomes if o.net_pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def max_drawdown(self) -> float:
        if not self.outcomes:
            return 0.0
        cumulative = np.cumsum([o.net_pnl for o in self.outcomes])
        peak = np.maximum.accumulate(cumulative)
        dd = cumulative - peak
        return float(np.min(dd))

    def equity_curve(self) -> List[float]:
        if not self.outcomes:
            return []
        return list(np.cumsum([o.net_pnl for o in self.outcomes]))

    def strategy_stats(self, strategy_id: str) -> Dict[str, float]:
        s = self._by_strategy.get(strategy_id)
        if not s:
            return {}
        return {"mean_pnl": s.mean, "std_pnl": s.std, "sharpe": s.sharpe, "n_trades": s.count}


# ---------------------------------------------------------------------------
# Model update triggers
# ---------------------------------------------------------------------------

class ModelUpdateTriggerEngine:
    """Decide when to retrain models."""

    def __init__(
        self,
        sharpe_threshold: float = -0.5,
        drift_z_threshold: float = 2.5,
        lookback: int = 50,
        min_trades_for_eval: int = 20,
    ):
        self.sharpe_threshold = sharpe_threshold
        self.drift_z_threshold = drift_z_threshold
        self.lookback = lookback
        self.min_trades_for_eval = min_trades_for_eval

    def check_performance_degradation(self, collector: FeedbackCollector) -> Optional[RetrainTrigger]:
        if collector.return_stats.count < self.min_trades_for_eval:
            return None
        recent_sharpe = collector.return_stats.sharpe
        if recent_sharpe < self.sharpe_threshold:
            return RetrainTrigger.PERFORMANCE
        return None

    def check_feature_drift(
        self, historical_mean: float, historical_std: float, recent_values: np.ndarray
    ) -> Optional[RetrainTrigger]:
        if len(recent_values) < 5 or historical_std < 1e-10:
            return None
        z = abs(float(np.mean(recent_values)) - historical_mean) / historical_std
        if z > self.drift_z_threshold:
            return RetrainTrigger.DRIFT
        return None

    def check_regime_change(self, regime_history: List[str], current_regime: str) -> Optional[RetrainTrigger]:
        if not regime_history:
            return None
        if len(regime_history) >= 3 and regime_history[-1] != current_regime:
            return RetrainTrigger.REGIME_CHANGE
        return None

    def should_retrain(
        self,
        collector: FeedbackCollector,
        regime_history: Optional[List[str]] = None,
        current_regime: Optional[str] = None,
        feature_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        recent_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[RetrainTrigger]:
        triggers: List[RetrainTrigger] = []
        perf = self.check_performance_degradation(collector)
        if perf:
            triggers.append(perf)
        if regime_history and current_regime:
            reg = self.check_regime_change(regime_history, current_regime)
            if reg:
                triggers.append(reg)
        if feature_stats and recent_features:
            for feat, (h_mean, h_std) in feature_stats.items():
                vals = recent_features.get(feat)
                if vals is not None:
                    d = self.check_feature_drift(h_mean, h_std, vals)
                    if d:
                        triggers.append(d)
                        break
        return triggers


# ---------------------------------------------------------------------------
# Online parameter adaptation
# ---------------------------------------------------------------------------

class ParameterAdapter:
    """Online gradient-based parameter update for signal weights."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, l2_reg: float = 0.001):
        self.lr = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.params: Dict[str, float] = {}
        self._velocity: Dict[str, float] = {}

    def initialize(self, param_names: List[str], init_value: float = 1.0) -> None:
        for name in param_names:
            self.params[name] = init_value
            self._velocity[name] = 0.0

    def update(self, gradients: Dict[str, float]) -> Dict[str, float]:
        """Update parameters using gradient with momentum and L2 regularization."""
        for name, grad in gradients.items():
            if name not in self.params:
                continue
            reg = self.l2_reg * self.params[name]
            total_grad = grad + reg
            v = self.momentum * self._velocity[name] - self.lr * total_grad
            self._velocity[name] = v
            self.params[name] += v
        return dict(self.params)

    def compute_signal_gradient(
        self, signal_values: Dict[str, float], realized_return: float
    ) -> Dict[str, float]:
        """Approximate gradient: d(loss)/d(weight) = -return * signal_value."""
        grads: Dict[str, float] = {}
        for name, val in signal_values.items():
            grads[name] = -realized_return * val
        return grads

    def step(self, signal_values: Dict[str, float], realized_return: float) -> Dict[str, float]:
        grads = self.compute_signal_gradient(signal_values, realized_return)
        return self.update(grads)

    def get_weights(self) -> Dict[str, float]:
        return dict(self.params)

    def normalize_weights(self) -> Dict[str, float]:
        total = sum(abs(v) for v in self.params.values())
        if total < 1e-15:
            return dict(self.params)
        self.params = {k: v / total for k, v in self.params.items()}
        return dict(self.params)


# ---------------------------------------------------------------------------
# Bayesian strategy score updater
# ---------------------------------------------------------------------------

class StrategyScoreUpdater:
    """Bayesian update of strategy quality from trade outcomes."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self._alpha: Dict[str, float] = {}
        self._beta: Dict[str, float] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def register_strategy(self, strategy_id: str) -> None:
        self._alpha[strategy_id] = self.prior_alpha
        self._beta[strategy_id] = self.prior_beta

    def update(self, strategy_id: str, outcome: TradeOutcome) -> float:
        if strategy_id not in self._alpha:
            self.register_strategy(strategy_id)
        if outcome.is_winner:
            self._alpha[strategy_id] += 1
        else:
            self._beta[strategy_id] += 1
        return self.score(strategy_id)

    def score(self, strategy_id: str) -> float:
        """Posterior mean of win probability."""
        a = self._alpha.get(strategy_id, self.prior_alpha)
        b = self._beta.get(strategy_id, self.prior_beta)
        return a / (a + b)

    def confidence_interval(self, strategy_id: str) -> Tuple[float, float]:
        a = self._alpha.get(strategy_id, self.prior_alpha)
        b = self._beta.get(strategy_id, self.prior_beta)
        # Wilson score interval approximation
        n = a + b - 2  # subtract priors
        if n <= 0:
            return (0.0, 1.0)
        p_hat = (a - 1) / n
        z = 1.96
        denom = 1 + z ** 2 / n
        centre = (p_hat + z ** 2 / (2 * n)) / denom
        spread = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
        return (max(centre - spread, 0.0), min(centre + spread, 1.0))

    def rank_strategies(self) -> List[Tuple[str, float]]:
        scores = [(sid, self.score(sid)) for sid in self._alpha]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def all_scores(self) -> Dict[str, float]:
        return {sid: self.score(sid) for sid in self._alpha}


# ---------------------------------------------------------------------------
# Regime feedback
# ---------------------------------------------------------------------------

class RegimeFeedback:
    """Track whether the regime detector predicted correctly."""

    def __init__(self, window: int = 50):
        self._predictions: Deque[Tuple[str, str]] = deque(maxlen=window)
        self._accuracy_window = window

    def record(self, predicted_regime: str, actual_regime: str) -> None:
        self._predictions.append((predicted_regime, actual_regime))

    def accuracy(self) -> float:
        if not self._predictions:
            return 0.0
        correct = sum(1 for p, a in self._predictions if p == a)
        return correct / len(self._predictions)

    def confusion(self) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {}
        for pred, actual in self._predictions:
            if pred not in matrix:
                matrix[pred] = {}
            matrix[pred][actual] = matrix[pred].get(actual, 0) + 1
        return matrix

    def recent_mismatches(self, n: int = 5) -> List[Tuple[str, str]]:
        return [(p, a) for p, a in list(self._predictions)[-n:] if p != a]


# ---------------------------------------------------------------------------
# Signal quality feedback
# ---------------------------------------------------------------------------

class SignalQualityTracker:
    """Rolling information coefficient (IC) for each signal."""

    def __init__(self, window: int = 100):
        self.window = window
        self._predictions: Dict[str, Deque[float]] = {}
        self._actuals: Dict[str, Deque[float]] = {}

    def record(self, signal_id: str, predicted: float, actual: float) -> None:
        if signal_id not in self._predictions:
            self._predictions[signal_id] = deque(maxlen=self.window)
            self._actuals[signal_id] = deque(maxlen=self.window)
        self._predictions[signal_id].append(predicted)
        self._actuals[signal_id].append(actual)

    def ic(self, signal_id: str) -> float:
        preds = self._predictions.get(signal_id)
        acts = self._actuals.get(signal_id)
        if not preds or len(preds) < 5:
            return 0.0
        p = np.array(list(preds))
        a = np.array(list(acts))
        p_m = p - np.mean(p)
        a_m = a - np.mean(a)
        denom = np.sqrt(np.sum(p_m ** 2) * np.sum(a_m ** 2))
        if denom < 1e-15:
            return 0.0
        return float(np.sum(p_m * a_m) / denom)

    def rank_ic(self, signal_id: str) -> float:
        """Rank IC (Spearman)."""
        preds = self._predictions.get(signal_id)
        acts = self._actuals.get(signal_id)
        if not preds or len(preds) < 5:
            return 0.0
        from scipy.stats import spearmanr  # type: ignore
        try:
            corr, _ = spearmanr(list(preds), list(acts))
            return float(corr)
        except Exception:
            return self.ic(signal_id)

    def all_ics(self) -> Dict[str, float]:
        return {sid: self.ic(sid) for sid in self._predictions}

    def decaying_signals(self, threshold: float = 0.02) -> List[str]:
        """Return signals whose IC has fallen below threshold."""
        return [sid for sid, ic_val in self.all_ics().items() if abs(ic_val) < threshold]

    def ic_history(self, signal_id: str, sub_window: int = 20) -> List[float]:
        preds = list(self._predictions.get(signal_id, []))
        acts = list(self._actuals.get(signal_id, []))
        if len(preds) < sub_window:
            return [self.ic(signal_id)]
        history: List[float] = []
        for i in range(sub_window, len(preds) + 1, max(sub_window // 2, 1)):
            p = np.array(preds[i - sub_window:i])
            a = np.array(acts[i - sub_window:i])
            pm = p - np.mean(p)
            am = a - np.mean(a)
            d = np.sqrt(np.sum(pm ** 2) * np.sum(am ** 2))
            history.append(float(np.sum(pm * am) / d) if d > 1e-15 else 0.0)
        return history


# ---------------------------------------------------------------------------
# Alert generator
# ---------------------------------------------------------------------------

class AlertGenerator:
    """Generate alerts on performance degradation, drawdown, alpha decay."""

    def __init__(
        self,
        drawdown_warn: float = -0.05,
        drawdown_crit: float = -0.10,
        sharpe_warn: float = 0.0,
        ic_decay_threshold: float = 0.02,
    ):
        self.drawdown_warn = drawdown_warn
        self.drawdown_crit = drawdown_crit
        self.sharpe_warn = sharpe_warn
        self.ic_decay_threshold = ic_decay_threshold
        self.alerts: List[Alert] = []

    def check_drawdown(self, collector: FeedbackCollector) -> Optional[Alert]:
        dd = collector.max_drawdown()
        equity = collector.equity_curve()
        if not equity:
            return None
        peak = max(equity) if equity else 1.0
        dd_pct = dd / abs(peak) if abs(peak) > 0 else 0.0
        if dd_pct < self.drawdown_crit:
            a = Alert(alert_type=AlertType.DRAWDOWN, severity=AlertSeverity.CRITICAL,
                      message=f"Critical drawdown: {dd_pct:.2%}", data={"dd_pct": dd_pct})
            self.alerts.append(a)
            return a
        if dd_pct < self.drawdown_warn:
            a = Alert(alert_type=AlertType.DRAWDOWN, severity=AlertSeverity.WARNING,
                      message=f"Drawdown warning: {dd_pct:.2%}", data={"dd_pct": dd_pct})
            self.alerts.append(a)
            return a
        return None

    def check_perf_degradation(self, collector: FeedbackCollector) -> Optional[Alert]:
        if collector.return_stats.count < 20:
            return None
        sharpe = collector.return_stats.sharpe
        if sharpe < self.sharpe_warn:
            a = Alert(alert_type=AlertType.PERF_DEGRADATION, severity=AlertSeverity.WARNING,
                      message=f"Rolling Sharpe below threshold: {sharpe:.3f}",
                      data={"sharpe": sharpe})
            self.alerts.append(a)
            return a
        return None

    def check_alpha_decay(self, signal_tracker: SignalQualityTracker) -> List[Alert]:
        decaying = signal_tracker.decaying_signals(self.ic_decay_threshold)
        generated: List[Alert] = []
        for sid in decaying:
            a = Alert(alert_type=AlertType.ALPHA_DECAY, severity=AlertSeverity.WARNING,
                      message=f"Signal {sid} IC decayed below {self.ic_decay_threshold}",
                      data={"signal_id": sid, "ic": signal_tracker.ic(sid)})
            self.alerts.append(a)
            generated.append(a)
        return generated

    def check_regime_mismatch(self, regime_fb: RegimeFeedback) -> Optional[Alert]:
        acc = regime_fb.accuracy()
        if acc < 0.5 and len(regime_fb._predictions) >= 10:
            a = Alert(alert_type=AlertType.REGIME_MISMATCH, severity=AlertSeverity.WARNING,
                      message=f"Regime detector accuracy: {acc:.1%}",
                      data={"accuracy": acc})
            self.alerts.append(a)
            return a
        return None

    def recent_alerts(self, n: int = 10) -> List[Alert]:
        return self.alerts[-n:]


# ---------------------------------------------------------------------------
# Dashboard data builder
# ---------------------------------------------------------------------------

class DashboardBuilder:
    """Assemble data for the feedback dashboard."""

    def __init__(self) -> None:
        self._regime_history: List[Tuple[datetime, str]] = []

    def record_regime(self, regime: str) -> None:
        self._regime_history.append((datetime.utcnow(), regime))

    def build(
        self,
        collector: FeedbackCollector,
        signal_tracker: SignalQualityTracker,
        strategy_scorer: StrategyScoreUpdater,
        alert_gen: AlertGenerator,
    ) -> DashboardData:
        eq = collector.equity_curve()
        # rolling sharpe
        window = 50
        pnls = [o.net_pnl for o in collector.outcomes]
        rolling_sharpe: List[float] = []
        for i in range(window, len(pnls) + 1):
            chunk = np.array(pnls[i - window:i])
            s = float(np.std(chunk, ddof=1))
            rolling_sharpe.append(float(np.mean(chunk)) / s if s > 1e-10 else 0.0)
        # rolling win rate
        rolling_wr: List[float] = []
        for i in range(window, len(collector.outcomes) + 1):
            subset = collector.outcomes[i - window:i]
            rolling_wr.append(sum(1 for o in subset if o.is_winner) / len(subset))
        # drawdown series
        dd_series: List[float] = []
        if eq:
            cum = np.array(eq)
            peak = np.maximum.accumulate(cum)
            dd_series = list(cum - peak)
        # signal IC history
        ic_hist: Dict[str, List[float]] = {}
        for sid in signal_tracker._predictions:
            ic_hist[sid] = signal_tracker.ic_history(sid)

        return DashboardData(
            equity_curve=eq,
            rolling_sharpe=rolling_sharpe,
            rolling_win_rate=rolling_wr,
            drawdown_series=dd_series,
            regime_history=list(self._regime_history),
            signal_ic_history=ic_hist,
            alerts=alert_gen.recent_alerts(20),
            strategy_scores=strategy_scorer.all_scores(),
            last_updated=datetime.utcnow(),
        )


# ---------------------------------------------------------------------------
# Master feedback loop
# ---------------------------------------------------------------------------

class FeedbackLoop:
    """Master class orchestrating all feedback components."""

    def __init__(
        self,
        window: int = 200,
        learning_rate: float = 0.01,
        drawdown_warn: float = -0.05,
        drawdown_crit: float = -0.10,
    ):
        self.collector = FeedbackCollector(window=window)
        self.trigger_engine = ModelUpdateTriggerEngine()
        self.param_adapter = ParameterAdapter(learning_rate=learning_rate)
        self.strategy_scorer = StrategyScoreUpdater()
        self.regime_feedback = RegimeFeedback()
        self.signal_tracker = SignalQualityTracker(window=window)
        self.alert_gen = AlertGenerator(drawdown_warn=drawdown_warn, drawdown_crit=drawdown_crit)
        self.dashboard_builder = DashboardBuilder()
        self._retrain_callbacks: List[Callable[[List[RetrainTrigger]], None]] = []
        self._regime_history: List[str] = []
        self._current_regime: str = "unknown"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_signals(self, signal_names: List[str]) -> None:
        self.param_adapter.initialize(signal_names)

    def register_strategy(self, strategy_id: str) -> None:
        self.strategy_scorer.register_strategy(strategy_id)

    def on_retrain(self, callback: Callable[[List[RetrainTrigger]], None]) -> None:
        self._retrain_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def process_outcome(
        self,
        outcome: TradeOutcome,
        signal_predictions: Optional[Dict[str, float]] = None,
        predicted_regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a single trade outcome through the full feedback pipeline."""
        result: Dict[str, Any] = {}

        # 1. Collect
        self.collector.add(outcome)

        # 2. Strategy score update
        if outcome.strategy_id:
            score = self.strategy_scorer.update(outcome.strategy_id, outcome)
            result["strategy_score"] = score

        # 3. Regime feedback
        if predicted_regime:
            self.regime_feedback.record(predicted_regime, self._current_regime)
            result["regime_accuracy"] = self.regime_feedback.accuracy()

        # 4. Signal quality
        if signal_predictions:
            for sig_id, pred_val in signal_predictions.items():
                self.signal_tracker.record(sig_id, pred_val, outcome.return_pct)
            result["signal_ics"] = self.signal_tracker.all_ics()

        # 5. Parameter adaptation
        if signal_predictions:
            new_weights = self.param_adapter.step(signal_predictions, outcome.return_pct)
            result["signal_weights"] = new_weights

        # 6. Check retrain triggers
        triggers = self.trigger_engine.should_retrain(
            self.collector,
            regime_history=self._regime_history,
            current_regime=self._current_regime,
        )
        if triggers:
            result["retrain_triggers"] = [t.value for t in triggers]
            for cb in self._retrain_callbacks:
                try:
                    cb(triggers)
                except Exception as e:
                    logger.error("Retrain callback error: %s", e)

        # 7. Alerts
        alerts: List[Alert] = []
        dd_alert = self.alert_gen.check_drawdown(self.collector)
        if dd_alert:
            alerts.append(dd_alert)
        perf_alert = self.alert_gen.check_perf_degradation(self.collector)
        if perf_alert:
            alerts.append(perf_alert)
        alpha_alerts = self.alert_gen.check_alpha_decay(self.signal_tracker)
        alerts.extend(alpha_alerts)
        reg_alert = self.alert_gen.check_regime_mismatch(self.regime_feedback)
        if reg_alert:
            alerts.append(reg_alert)
        if alerts:
            result["alerts"] = [a.message for a in alerts]

        return result

    def set_regime(self, regime: str) -> None:
        self._current_regime = regime
        self._regime_history.append(regime)
        self.dashboard_builder.record_regime(regime)

    def get_dashboard(self) -> DashboardData:
        return self.dashboard_builder.build(
            self.collector, self.signal_tracker, self.strategy_scorer, self.alert_gen
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "total_trades": len(self.collector.outcomes),
            "total_pnl": self.collector.total_pnl(),
            "win_rate": self.collector.win_rate(),
            "profit_factor": self.collector.profit_factor(),
            "max_drawdown": self.collector.max_drawdown(),
            "rolling_sharpe": self.collector.return_stats.sharpe,
            "current_regime": self._current_regime,
            "regime_accuracy": self.regime_feedback.accuracy(),
            "signal_ics": self.signal_tracker.all_ics(),
            "strategy_scores": self.strategy_scorer.all_scores(),
            "signal_weights": self.param_adapter.get_weights(),
            "n_alerts": len(self.alert_gen.alerts),
        }
