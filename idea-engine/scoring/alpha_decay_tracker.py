"""
alpha_decay_tracker.py
----------------------
Tracks how signal alpha decays over time for the idea-engine.

Fits exponential decay models to rolling IC series, estimates half-lives,
detects regime-conditional decay acceleration, and archives dead signals.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

REGIME_NAMES = ["trending_bull", "trending_bear", "mean_reverting", "chaotic"]


@dataclass
class ICObservation:
    """Single IC measurement for a signal."""
    timestamp: float          # unix epoch
    ic: float                 # information coefficient, typically -0.3 to 0.3
    regime: Optional[str] = None   # current market regime label
    horizon_days: int = 1     # prediction horizon (1d, 5d, 21d…)
    n_assets: int = 1         # number of assets used in IC calculation


@dataclass
class DecayModel:
    """
    Fitted exponential decay: alpha(t) = alpha_0 * exp(-lambda_ * t)

    t is measured in days since the signal's peak IC observation.
    """
    alpha_0: float = 0.0       # initial IC level
    lambda_: float = 0.0       # decay rate (per day)
    half_life_days: float = float("inf")
    r_squared: float = 0.0
    fit_timestamp: float = field(default_factory=time.time)
    n_points: int = 0


@dataclass
class SignalRecord:
    """Full tracking record for one signal."""
    signal_id: str
    name: str
    created_at: float
    ic_history: List[ICObservation] = field(default_factory=list)
    decay_model: DecayModel = field(default_factory=DecayModel)
    regime_decay_rates: Dict[str, float] = field(default_factory=dict)
    archived: bool = False
    archive_reason: str = ""
    portfolio_weight: float = 1.0   # weight in multi-signal portfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_decay(t: float, alpha_0: float, lam: float) -> float:
    return alpha_0 * math.exp(-lam * t)


def _fit_exponential_decay(
    ts: List[float], ics: List[float]
) -> Tuple[float, float, float]:
    """
    OLS fit of log|IC| = log|alpha_0| - lambda * t.

    Returns (alpha_0, lambda_, r_squared).
    Handles sign of alpha_0 by working with abs values, then restoring sign.
    """
    if len(ts) < 3:
        return ics[0] if ics else 0.0, 0.0, 0.0

    # Anchor t=0 to earliest observation
    t0 = min(ts)
    ts_rel = [t - t0 for t in ts]

    # Filter zeros/negatives for log transform
    pairs = [(t, abs(ic)) for t, ic in zip(ts_rel, ics) if abs(ic) > 1e-9]
    if len(pairs) < 2:
        return ics[0] if ics else 0.0, 0.0, 0.0

    log_ics = [math.log(ic) for _, ic in pairs]
    t_vals = [t for t, _ in pairs]

    n = len(t_vals)
    mean_t = sum(t_vals) / n
    mean_y = sum(log_ics) / n

    ss_xy = sum((t - mean_t) * (y - mean_y) for t, y in zip(t_vals, log_ics))
    ss_xx = sum((t - mean_t) ** 2 for t in t_vals)

    if abs(ss_xx) < 1e-12:
        return math.exp(mean_y), 0.0, 0.0

    slope = ss_xy / ss_xx   # = -lambda
    intercept = mean_y - slope * mean_t   # = log(alpha_0)

    alpha_0 = math.exp(intercept)
    lambda_ = max(0.0, -slope)  # force non-negative decay

    # Restore original sign (use sign of first ic)
    sign = 1.0 if (ics[0] >= 0) else -1.0
    alpha_0 *= sign

    # R-squared on log scale
    y_pred = [intercept + slope * t for t in t_vals]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(log_ics, y_pred))
    ss_tot = sum((y - mean_y) ** 2 for y in log_ics)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return alpha_0, lambda_, max(0.0, min(1.0, r2))


def _half_life(lambda_: float) -> float:
    if lambda_ < 1e-12:
        return float("inf")
    return math.log(2.0) / lambda_


def _rolling_ic(ics: List[float], window: int) -> List[float]:
    """Compute rolling mean IC over a sliding window."""
    result = []
    buf: Deque[float] = deque(maxlen=window)
    for ic in ics:
        buf.append(ic)
        result.append(sum(buf) / len(buf))
    return result


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class AlphaDecayTracker:
    """
    Tracks exponential decay of signal alpha (IC) over time.

    Supports:
      - Rolling IC tracking per signal
      - Exponential decay model fitting
      - Half-life estimation
      - Regime-conditional decay rates
      - Signal graveyard (archive below threshold)
      - Forward IC prediction
      - Multi-signal portfolio IC
    """

    def __init__(
        self,
        ic_threshold: float = 0.02,
        halflife_threshold_days: float = 5.0,
        rolling_window: int = 20,
        regime_min_obs: int = 5,
    ):
        """
        Parameters
        ----------
        ic_threshold : float
            IC below this (in absolute value) triggers graveyard candidate.
        halflife_threshold_days : float
            Signals with half-life < this are archived.
        rolling_window : int
            Window size for rolling IC computation.
        regime_min_obs : int
            Minimum observations per regime to fit regime-specific decay.
        """
        self.ic_threshold = ic_threshold
        self.halflife_threshold_days = halflife_threshold_days
        self.rolling_window = rolling_window
        self.regime_min_obs = regime_min_obs
        self._signals: Dict[str, SignalRecord] = {}
        self._graveyard: Dict[str, SignalRecord] = {}

    # ------------------------------------------------------------------
    # Signal registration
    # ------------------------------------------------------------------

    def register_signal(
        self,
        signal_id: str,
        name: str,
        portfolio_weight: float = 1.0,
    ) -> None:
        """Register a new signal for tracking."""
        if signal_id in self._signals:
            return
        self._signals[signal_id] = SignalRecord(
            signal_id=signal_id,
            name=name,
            created_at=time.time(),
            portfolio_weight=portfolio_weight,
        )

    def list_signals(self, include_archived: bool = False) -> List[str]:
        ids = list(self._signals.keys())
        if include_archived:
            ids += list(self._graveyard.keys())
        return ids

    # ------------------------------------------------------------------
    # IC observation ingestion
    # ------------------------------------------------------------------

    def add_ic_observation(
        self, signal_id: str, obs: ICObservation
    ) -> SignalRecord:
        """Record an IC observation and refit decay model."""
        rec = self._get(signal_id)
        rec.ic_history.append(obs)
        self._fit_decay_model(rec)
        self._fit_regime_decay_rates(rec)
        self._check_graveyard(rec)
        return rec

    def bulk_add(
        self, signal_id: str, timestamps: List[float], ics: List[float],
        regimes: Optional[List[str]] = None, horizon_days: int = 1,
    ) -> None:
        """Add multiple IC observations at once."""
        regimes = regimes or [None] * len(timestamps)
        for ts, ic, reg in zip(timestamps, ics, regimes):
            self.add_ic_observation(
                signal_id,
                ICObservation(
                    timestamp=ts, ic=ic, regime=reg,
                    horizon_days=horizon_days,
                ),
            )

    # ------------------------------------------------------------------
    # Decay model fitting
    # ------------------------------------------------------------------

    def _fit_decay_model(self, rec: SignalRecord) -> None:
        """Fit global exponential decay model from IC history."""
        if len(rec.ic_history) < 3:
            return
        ts_days = [obs.timestamp / 86400.0 for obs in rec.ic_history]
        ics = [obs.ic for obs in rec.ic_history]
        alpha_0, lam, r2 = _fit_exponential_decay(ts_days, ics)
        hl = _half_life(lam)
        rec.decay_model = DecayModel(
            alpha_0=alpha_0,
            lambda_=lam,
            half_life_days=hl,
            r_squared=r2,
            fit_timestamp=time.time(),
            n_points=len(ics),
        )

    def _fit_regime_decay_rates(self, rec: SignalRecord) -> None:
        """Fit per-regime exponential decay rates."""
        from collections import defaultdict
        regime_obs: Dict[str, List[ICObservation]] = defaultdict(list)
        for obs in rec.ic_history:
            if obs.regime is not None:
                regime_obs[obs.regime].append(obs)

        for regime, obs_list in regime_obs.items():
            if len(obs_list) < self.regime_min_obs:
                continue
            ts_days = [o.timestamp / 86400.0 for o in obs_list]
            ics = [o.ic for o in obs_list]
            _, lam, _ = _fit_exponential_decay(ts_days, ics)
            rec.regime_decay_rates[regime] = lam

    # ------------------------------------------------------------------
    # Graveyard management
    # ------------------------------------------------------------------

    def _check_graveyard(self, rec: SignalRecord) -> None:
        """Archive signal if it no longer meets quality thresholds."""
        if rec.archived:
            return
        dm = rec.decay_model
        if dm.n_points < 3:
            return

        recent_ics = [obs.ic for obs in rec.ic_history[-self.rolling_window:]]
        mean_recent_ic = abs(statistics.mean(recent_ics)) if recent_ics else 0.0

        reasons = []
        if mean_recent_ic < self.ic_threshold:
            reasons.append(
                f"recent mean |IC|={mean_recent_ic:.4f} < threshold={self.ic_threshold}"
            )
        if dm.half_life_days < self.halflife_threshold_days:
            reasons.append(
                f"half-life={dm.half_life_days:.1f}d < threshold={self.halflife_threshold_days}d"
            )

        if reasons:
            rec.archived = True
            rec.archive_reason = "; ".join(reasons)
            self._graveyard[rec.signal_id] = rec
            del self._signals[rec.signal_id]

    def revive_signal(self, signal_id: str) -> None:
        """Move a signal from graveyard back to active tracking."""
        if signal_id not in self._graveyard:
            raise KeyError(f"{signal_id} not in graveyard.")
        rec = self._graveyard.pop(signal_id)
        rec.archived = False
        rec.archive_reason = ""
        self._signals[signal_id] = rec

    def get_graveyard(self) -> List[Dict]:
        """Return summary of archived signals."""
        result = []
        for sid, rec in self._graveyard.items():
            result.append({
                "signal_id": sid,
                "name": rec.name,
                "archive_reason": rec.archive_reason,
                "half_life_days": round(rec.decay_model.half_life_days, 2),
                "n_observations": len(rec.ic_history),
            })
        return result

    # ------------------------------------------------------------------
    # Querying / prediction
    # ------------------------------------------------------------------

    def get_half_life(self, signal_id: str) -> float:
        """Return estimated half-life in days."""
        return self._get_any(signal_id).decay_model.half_life_days

    def get_regime_decay_rate(
        self, signal_id: str, regime: str
    ) -> Optional[float]:
        """Return regime-specific decay rate lambda (per day), or None."""
        rec = self._get_any(signal_id)
        return rec.regime_decay_rates.get(regime)

    def predict_ic(
        self,
        signal_id: str,
        horizon_days: float,
        regime: Optional[str] = None,
    ) -> float:
        """
        Forward IC prediction: alpha(t) = alpha_0 * exp(-lambda * t).

        Uses regime-specific lambda if available and regime is specified.
        """
        rec = self._get_any(signal_id)
        dm = rec.decay_model
        if dm.alpha_0 == 0.0 and dm.lambda_ == 0.0:
            return 0.0

        # Current IC estimate = most recent rolling mean
        recent_ics = [obs.ic for obs in rec.ic_history[-self.rolling_window:]]
        current_ic = statistics.mean(recent_ics) if recent_ics else dm.alpha_0

        # Choose lambda
        lam = dm.lambda_
        if regime is not None and regime in rec.regime_decay_rates:
            lam = rec.regime_decay_rates[regime]

        return current_ic * math.exp(-lam * horizon_days)

    def rolling_ic_series(
        self, signal_id: str, window: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """Return list of (timestamp, rolling_IC) pairs."""
        rec = self._get_any(signal_id)
        w = window or self.rolling_window
        ics = [obs.ic for obs in rec.ic_history]
        ts = [obs.timestamp for obs in rec.ic_history]
        rolled = _rolling_ic(ics, w)
        return list(zip(ts, rolled))

    def signal_summary(self, signal_id: str) -> Dict:
        """Return human-readable summary dict for a signal."""
        rec = self._get_any(signal_id)
        dm = rec.decay_model
        recent_ics = [obs.ic for obs in rec.ic_history[-self.rolling_window:]]
        mean_ic = statistics.mean(recent_ics) if recent_ics else float("nan")
        std_ic = statistics.stdev(recent_ics) if len(recent_ics) > 1 else float("nan")
        return {
            "signal_id": signal_id,
            "name": rec.name,
            "n_observations": len(rec.ic_history),
            "mean_recent_ic": round(mean_ic, 5),
            "std_recent_ic": round(std_ic, 5),
            "alpha_0": round(dm.alpha_0, 5),
            "decay_lambda_per_day": round(dm.lambda_, 5),
            "half_life_days": round(dm.half_life_days, 2),
            "decay_r_squared": round(dm.r_squared, 4),
            "regime_decay_rates": {
                k: round(v, 5) for k, v in rec.regime_decay_rates.items()
            },
            "archived": rec.archived,
            "archive_reason": rec.archive_reason,
            "predicted_ic_5d": round(self.predict_ic(signal_id, 5.0), 5),
            "predicted_ic_21d": round(self.predict_ic(signal_id, 21.0), 5),
        }

    # ------------------------------------------------------------------
    # Multi-signal portfolio IC
    # ------------------------------------------------------------------

    def portfolio_ic(
        self,
        signal_ids: Optional[List[str]] = None,
        horizon_days: float = 1.0,
        regime: Optional[str] = None,
    ) -> Dict:
        """
        Compute combined portfolio IC across signals.

        Uses signal weights, accounting for cross-signal correlation decay.
        Under simplifying assumption of zero cross-correlation:
          IC_portfolio = sum(w_i * IC_i) / sqrt(sum(w_i^2))

        Returns dict with portfolio_ic and per-signal contributions.
        """
        ids = signal_ids or list(self._signals.keys())
        if not ids:
            return {"portfolio_ic": 0.0, "contributions": {}}

        weights = []
        pred_ics = []
        for sid in ids:
            rec = self._get_any(sid)
            w = rec.portfolio_weight
            ic_pred = self.predict_ic(sid, horizon_days, regime)
            weights.append(w)
            pred_ics.append(ic_pred)

        # Normalise weights
        w_sum = sum(weights)
        if w_sum < 1e-9:
            return {"portfolio_ic": 0.0, "contributions": {}}
        norm_w = [w / w_sum for w in weights]

        # Portfolio IC (zero cross-correlation approximation)
        numerator = sum(w * ic for w, ic in zip(norm_w, pred_ics))
        denominator = math.sqrt(sum(w ** 2 for w in norm_w))
        port_ic = numerator / max(denominator, 1e-9)

        contributions = {
            sid: round(w * ic, 6)
            for sid, w, ic in zip(ids, norm_w, pred_ics)
        }

        return {
            "portfolio_ic": round(port_ic, 6),
            "horizon_days": horizon_days,
            "regime": regime,
            "n_signals": len(ids),
            "contributions": contributions,
        }

    def ic_correlation_matrix(
        self, signal_ids: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Estimate pairwise IC correlation between signals using shared timestamps.
        Returns dict keyed by (sid_i, sid_j).
        """
        ids = signal_ids or list(self._signals.keys())
        result: Dict[Tuple[str, str], float] = {}

        for i, sid_i in enumerate(ids):
            for j, sid_j in enumerate(ids):
                if j <= i:
                    continue
                corr = self._pairwise_ic_corr(sid_i, sid_j)
                result[(sid_i, sid_j)] = round(corr, 4)

        return result

    def _pairwise_ic_corr(self, sid_i: str, sid_j: str) -> float:
        """Pearson correlation of IC series aligned by nearest timestamp."""
        rec_i = self._get_any(sid_i)
        rec_j = self._get_any(sid_j)
        ts_i = {round(o.timestamp / 86400): o.ic for o in rec_i.ic_history}
        ts_j = {round(o.timestamp / 86400): o.ic for o in rec_j.ic_history}
        shared = set(ts_i.keys()) & set(ts_j.keys())
        if len(shared) < 3:
            return 0.0
        xi = [ts_i[t] for t in sorted(shared)]
        xj = [ts_j[t] for t in sorted(shared)]
        mx, my = statistics.mean(xi), statistics.mean(xj)
        num = sum((a - mx) * (b - my) for a, b in zip(xi, xj))
        denom = math.sqrt(
            sum((a - mx) ** 2 for a in xi) * sum((b - my) ** 2 for b in xj)
        )
        return num / max(denom, 1e-12)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, signal_id: str) -> SignalRecord:
        if signal_id not in self._signals:
            raise KeyError(f"Active signal not found: {signal_id}")
        return self._signals[signal_id]

    def _get_any(self, signal_id: str) -> SignalRecord:
        if signal_id in self._signals:
            return self._signals[signal_id]
        if signal_id in self._graveyard:
            return self._graveyard[signal_id]
        raise KeyError(f"Signal not found: {signal_id}")

    def __repr__(self) -> str:
        return (
            f"AlphaDecayTracker("
            f"active={len(self._signals)}, "
            f"archived={len(self._graveyard)})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42)

    tracker = AlphaDecayTracker(
        ic_threshold=0.015,
        halflife_threshold_days=3.0,
        rolling_window=10,
    )

    tracker.register_signal("rsi_reversal", "RSI < 30 reversal")
    tracker.register_signal("momentum_12_1", "12-1 month momentum")
    tracker.register_signal("dead_signal", "Noise signal")

    now_days = time.time() / 86400.0
    regimes_cycle = ["trending_bull", "mean_reverting", "chaotic", "trending_bear"]

    for i in range(60):
        t_days = now_days - (60 - i)
        regime = regimes_cycle[i % len(regimes_cycle)]
        # RSI signal: slow decay
        ic_rsi = 0.08 * math.exp(-0.01 * i) + random.gauss(0, 0.01)
        tracker.add_ic_observation(
            "rsi_reversal",
            ICObservation(timestamp=t_days * 86400, ic=ic_rsi, regime=regime),
        )
        # Momentum: faster decay
        ic_mom = 0.06 * math.exp(-0.03 * i) + random.gauss(0, 0.008)
        tracker.add_ic_observation(
            "momentum_12_1",
            ICObservation(timestamp=t_days * 86400, ic=ic_mom, regime=regime),
        )

    # Dead signal: IC always near zero
    for i in range(30):
        t_days = now_days - (30 - i)
        tracker.add_ic_observation(
            "dead_signal",
            ICObservation(timestamp=t_days * 86400, ic=random.gauss(0, 0.005)),
        )

    print("=== Signal Summaries ===")
    for sid in tracker.list_signals(include_archived=True):
        try:
            print(tracker.signal_summary(sid))
        except Exception as e:
            print(f"{sid}: {e}")

    print("\n=== Graveyard ===")
    print(tracker.get_graveyard())

    print("\n=== Portfolio IC (5-day horizon) ===")
    print(tracker.portfolio_ic(horizon_days=5.0))

    print("\n=== IC Correlation Matrix ===")
    print(tracker.ic_correlation_matrix())
