"""
risk-engine/drawdown_controller.py

Active drawdown monitoring and control:
  - Max drawdown, current drawdown, drawdown duration
  - Recovery factor, drawdown clustering
  - Calmar efficiency
  - Adaptive risk-target scaling based on current drawdown depth
  - Hypothesis generation for drawdown-triggered risk reduction
  - SQLite persistence to the ``drawdown_events`` table

All equity-curve inputs should be a pd.Series of cumulative equity values
(e.g. starting at 1.0 or at initial capital).
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DrawdownEvent:
    """
    A single drawdown episode from peak to trough and back to recovery.

    Attributes
    ----------
    event_id : str
        UUID for this event.
    start_ts : str
        ISO-8601 timestamp of the peak (drawdown start).
    trough_ts : str or None
        ISO-8601 timestamp of the trough (worst point).
    end_ts : str or None
        ISO-8601 timestamp of recovery (back to prior peak), or None if
        the series ended underwater.
    peak_equity : float
        Equity value at the start of the drawdown.
    trough_equity : float or None
        Equity value at the trough.
    drawdown_pct : float or None
        Fractional drawdown = (peak - trough) / peak.
    duration_bars : int or None
        Total bars from peak to recovery.
    recovery_bars : int or None
        Bars from trough to recovery (None if not recovered).
    regime : str
        Optional regime label at peak.
    """

    event_id: str
    start_ts: str
    trough_ts: Optional[str]
    end_ts: Optional[str]
    peak_equity: float
    trough_equity: Optional[float]
    drawdown_pct: Optional[float]
    duration_bars: Optional[int]
    recovery_bars: Optional[int] = None
    regime: str = "unknown"

    @classmethod
    def create(
        cls,
        start_ts: str,
        peak_equity: float,
        trough_ts: Optional[str] = None,
        trough_equity: Optional[float] = None,
        end_ts: Optional[str] = None,
        duration_bars: Optional[int] = None,
        recovery_bars: Optional[int] = None,
        regime: str = "unknown",
    ) -> "DrawdownEvent":
        dd_pct = None
        if trough_equity is not None and peak_equity > 0:
            dd_pct = float((peak_equity - trough_equity) / peak_equity)

        return cls(
            event_id=str(uuid.uuid4()),
            start_ts=start_ts,
            trough_ts=trough_ts,
            end_ts=end_ts,
            peak_equity=peak_equity,
            trough_equity=trough_equity,
            drawdown_pct=dd_pct,
            duration_bars=duration_bars,
            recovery_bars=recovery_bars,
            regime=regime,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start_ts": self.start_ts,
            "trough_ts": self.trough_ts,
            "end_ts": self.end_ts,
            "peak_equity": self.peak_equity,
            "trough_equity": self.trough_equity,
            "drawdown_pct": self.drawdown_pct,
            "duration_bars": self.duration_bars,
            "recovery_bars": self.recovery_bars,
            "regime": self.regime,
        }

    def to_db_row(self) -> tuple:
        return (
            self.start_ts,
            self.trough_ts,
            self.end_ts,
            self.peak_equity,
            self.trough_equity,
            self.drawdown_pct,
            self.duration_bars,
            self.regime,
        )


# ---------------------------------------------------------------------------
# Core controller
# ---------------------------------------------------------------------------


class DrawdownController:
    """
    Active drawdown monitoring, measurement, and adaptive position sizing.

    Parameters
    ----------
    db_path : str, optional
        SQLite database path.  When provided, events are persisted automatically
        via :meth:`flush_to_db`.
    annualisation : int
        Bars per year, used for CAGR and Calmar calculations.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(7)
    >>> rets = pd.Series(rng.normal(0.0004, 0.012, 500))
    >>> equity = (1 + rets).cumprod()
    >>> dc = DrawdownController()
    >>> print(f"Max DD: {dc.max_drawdown(equity):.2%}")
    >>> print(f"Current DD: {dc.current_drawdown(equity):.2%}")
    >>> scale = dc.adaptive_risk_target(equity, max_dd_target=0.15)
    >>> print(f"Position scale: {scale:.3f}")
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        annualisation: int = 252,
    ) -> None:
        self.db_path = db_path
        self.annualisation = annualisation
        self._events: list[DrawdownEvent] = []

    # ------------------------------------------------------------------
    # Core measurement methods
    # ------------------------------------------------------------------

    def max_drawdown(self, equity: pd.Series) -> float:
        """
        Maximum peak-to-trough drawdown of an equity curve.

        Parameters
        ----------
        equity : pd.Series
            Cumulative equity (positive values).

        Returns
        -------
        float
            Maximum drawdown as a positive fraction in [0, 1].
        """
        arr = equity.dropna().values
        if len(arr) < 2:
            return 0.0
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / peak
        return float(np.max(dd))

    def current_drawdown(self, equity: pd.Series) -> float:
        """
        Current drawdown: distance from the most recent all-time-high.

        Parameters
        ----------
        equity : pd.Series
            Cumulative equity.

        Returns
        -------
        float
            Current drawdown as a positive fraction.  0 means equity is at ATH.
        """
        arr = equity.dropna().values
        if len(arr) == 0:
            return 0.0
        peak = float(np.max(arr))
        last = float(arr[-1])
        if peak == 0:
            return 0.0
        return float(max((peak - last) / peak, 0.0))

    def drawdown_duration(self, equity: pd.Series) -> int:
        """
        Number of bars elapsed since the last all-time-high.

        Returns 0 when the equity is currently at its ATH.

        Parameters
        ----------
        equity : pd.Series
            Cumulative equity.

        Returns
        -------
        int
            Bars since last all-time high.
        """
        arr = equity.dropna().values
        if len(arr) == 0:
            return 0
        ath_idx = int(np.argmax(arr))
        return len(arr) - 1 - ath_idx

    def recovery_factor(self, equity: pd.Series) -> float:
        """
        Recovery factor: total return divided by maximum drawdown.

        RF = (equity[-1] / equity[0] - 1) / max_drawdown

        A higher recovery factor indicates the strategy earns many times
        its worst loss.

        Parameters
        ----------
        equity : pd.Series
            Cumulative equity.

        Returns
        -------
        float
            Recovery factor.  Returns inf when max drawdown is zero.
        """
        arr = equity.dropna().values
        if len(arr) < 2:
            return 0.0
        total_return = float(arr[-1] / arr[0] - 1.0)
        max_dd = self.max_drawdown(equity)
        if max_dd == 0:
            return np.inf
        return float(total_return / max_dd)

    def drawdown_clustering(
        self,
        equity: pd.Series,
        threshold: float = 0.05,
    ) -> list[DrawdownEvent]:
        """
        Identify discrete drawdown episodes where equity drops more than
        ``threshold`` from a local peak.

        Each episode spans from the peak bar through the trough to recovery
        (or the end of the series if not recovered).

        Parameters
        ----------
        equity : pd.Series
            Cumulative equity, DatetimeIndex preferred.
        threshold : float
            Minimum fractional decline required to classify as an episode
            (e.g. 0.05 = 5 % drop).

        Returns
        -------
        list[DrawdownEvent]
            Drawdown episodes sorted by start date.
        """
        arr = equity.dropna().values
        idx = equity.dropna().index
        n = len(arr)

        if n < 2:
            return []

        events: list[DrawdownEvent] = []
        in_drawdown = False
        peak_val = arr[0]
        peak_bar = 0
        trough_val = arr[0]
        trough_bar = 0

        for i in range(1, n):
            val = arr[i]

            if not in_drawdown:
                if val > peak_val:
                    peak_val = val
                    peak_bar = i
                else:
                    dd = (peak_val - val) / peak_val
                    if dd >= threshold:
                        in_drawdown = True
                        trough_val = val
                        trough_bar = i
            else:
                if val < trough_val:
                    trough_val = val
                    trough_bar = i
                elif val >= peak_val:
                    # Recovered
                    ts_peak = self._idx_to_str(idx, peak_bar)
                    ts_trough = self._idx_to_str(idx, trough_bar)
                    ts_end = self._idx_to_str(idx, i)

                    event = DrawdownEvent.create(
                        start_ts=ts_peak,
                        peak_equity=float(peak_val),
                        trough_ts=ts_trough,
                        trough_equity=float(trough_val),
                        end_ts=ts_end,
                        duration_bars=i - peak_bar,
                        recovery_bars=i - trough_bar,
                    )
                    events.append(event)
                    self._events.append(event)

                    in_drawdown = False
                    peak_val = val
                    peak_bar = i
                    trough_val = val
                    trough_bar = i

        # Handle open drawdown at end of series
        if in_drawdown:
            ts_peak = self._idx_to_str(idx, peak_bar)
            ts_trough = self._idx_to_str(idx, trough_bar)
            event = DrawdownEvent.create(
                start_ts=ts_peak,
                peak_equity=float(peak_val),
                trough_ts=ts_trough,
                trough_equity=float(trough_val),
                end_ts=None,
                duration_bars=n - 1 - peak_bar,
                recovery_bars=None,
            )
            events.append(event)
            self._events.append(event)

        return events

    def calmar_efficiency(self, returns: pd.Series, max_dd: float) -> float:
        """
        Calmar efficiency: annualised Sharpe-like ratio but penalised by
        max drawdown instead of volatility.

        CE = annualised_return / max_drawdown

        Parameters
        ----------
        returns : pd.Series
            Return series.
        max_dd : float
            Maximum drawdown (positive fraction).

        Returns
        -------
        float
            Calmar efficiency ratio.
        """
        if max_dd == 0:
            return np.inf
        arr = returns.dropna().values
        mean_ret = float(np.mean(arr))
        ann_ret = mean_ret * self.annualisation
        return float(ann_ret / max_dd)

    # ------------------------------------------------------------------
    # Adaptive position sizing
    # ------------------------------------------------------------------

    def adaptive_risk_target(
        self,
        equity: pd.Series,
        max_dd_target: float = 0.15,
        scale_floor: float = 0.1,
        recovery_speed: float = 2.0,
    ) -> float:
        """
        Compute an adaptive position scale factor based on the current
        drawdown depth relative to the maximum allowable drawdown target.

        The scaling logic:
          - When current drawdown = 0, full position size (scale = 1.0).
          - As drawdown approaches ``max_dd_target``, scale decreases
            proportionally toward ``scale_floor``.
          - Beyond ``max_dd_target``, scale is capped at ``scale_floor``.
          - ``recovery_speed`` controls the curvature: higher values mean
            faster de-risking as drawdown grows.

        Parameters
        ----------
        equity : pd.Series
            Current equity curve.
        max_dd_target : float
            Maximum acceptable drawdown level (e.g. 0.15 = 15 %).
        scale_floor : float
            Minimum position scale (never goes below this).
        recovery_speed : float
            Power parameter controlling curvature of the scale-down.

        Returns
        -------
        float
            Position scale factor in [scale_floor, 1.0].
        """
        if max_dd_target <= 0:
            raise ValueError("max_dd_target must be positive")

        current_dd = self.current_drawdown(equity)

        if current_dd <= 0:
            return 1.0

        dd_ratio = current_dd / max_dd_target
        if dd_ratio >= 1.0:
            return float(scale_floor)

        # Smooth polynomial decay
        scale = 1.0 - (1.0 - scale_floor) * (dd_ratio ** (1.0 / recovery_speed))
        return float(np.clip(scale, scale_floor, 1.0))

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def generate_risk_reduction_hypothesis(
        self,
        drawdown_event: DrawdownEvent,
        current_equity: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """
        Generate a risk-reduction hypothesis triggered by a significant
        drawdown event.

        The hypothesis is returned as a dict compatible with
        ``Hypothesis.create`` from ``hypothesis.types``.

        Heuristics applied:
          - Deep drawdowns (> 15 %) → suggest reducing leverage / position size.
          - Long-duration drawdowns (> 60 bars) → suggest regime-filter addition.
          - Rapid drawdowns (recovered quickly) → suggest tighter stop-loss.

        Parameters
        ----------
        drawdown_event : DrawdownEvent
            The drawdown episode to explain.
        current_equity : pd.Series, optional
            Full equity curve for additional context (volatility, regime).

        Returns
        -------
        dict
            Hypothesis creation kwargs.
        """
        dd_pct = drawdown_event.drawdown_pct or 0.0
        duration = drawdown_event.duration_bars or 0
        recovery = drawdown_event.recovery_bars

        # Classify the drawdown
        if dd_pct > 0.25:
            severity = "critical"
            sharpe_delta = 0.30
            dd_delta = -0.15
        elif dd_pct > 0.15:
            severity = "severe"
            sharpe_delta = 0.20
            dd_delta = -0.10
        elif dd_pct > 0.08:
            severity = "moderate"
            sharpe_delta = 0.10
            dd_delta = -0.05
        else:
            severity = "minor"
            sharpe_delta = 0.05
            dd_delta = -0.02

        # Pick the most appropriate intervention
        if duration > 60:
            action = "add_regime_filter"
            description = (
                f"Drawdown of {dd_pct:.1%} persisted for {duration} bars starting "
                f"{drawdown_event.start_ts}. Long duration suggests a sustained adverse "
                f"regime. Hypothesis: add a regime-detection filter to pause trading "
                f"when bearish conditions are detected, reducing drawdown by an estimated "
                f"{abs(dd_delta):.0%}."
            )
        elif recovery is not None and recovery < 10 and dd_pct > 0.05:
            action = "tighten_stop_loss"
            description = (
                f"Rapid drawdown of {dd_pct:.1%} over {duration} bars "
                f"(recovered in {recovery} bars) starting {drawdown_event.start_ts}. "
                f"Suggests a sharp but transient shock. Hypothesis: tighten intraday "
                f"stop-loss rules to reduce exposure during spike-down events."
            )
        else:
            action = "reduce_position_size"
            description = (
                f"{severity.title()} drawdown of {dd_pct:.1%} over {duration} bars "
                f"starting {drawdown_event.start_ts}. Hypothesis: reduce position size "
                f"by {min(int(dd_pct * 200), 50)} % when rolling volatility exceeds "
                f"1.5× its 60-day average, limiting drawdown depth."
            )

        params: dict[str, Any] = {
            "action": action,
            "severity": severity,
            "drawdown_pct": round(dd_pct, 4),
            "duration_bars": duration,
            "recovery_bars": recovery,
            "start_ts": drawdown_event.start_ts,
            "trough_ts": drawdown_event.trough_ts,
            "peak_equity": drawdown_event.peak_equity,
            "trough_equity": drawdown_event.trough_equity,
            "regime_at_peak": drawdown_event.regime,
        }

        return {
            "hypothesis_type": "REGIME_FILTER",
            "parent_pattern_id": drawdown_event.event_id,
            "parameters": params,
            "predicted_sharpe_delta": round(sharpe_delta, 4),
            "predicted_dd_delta": round(dd_delta, 4),
            "novelty_score": 0.55,
            "description": description,
        }

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def flush_to_db(self) -> int:
        """
        Write buffered DrawdownEvents to the ``drawdown_events`` table.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not self.db_path or not self._events:
            return 0

        rows = [e.to_db_row() for e in self._events]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO drawdown_events
                    (start_ts, trough_ts, end_ts,
                     peak_equity, trough_equity, drawdown_pct,
                     duration_bars, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        inserted = len(rows)
        self._events.clear()
        return inserted

    def fetch_events(self, min_drawdown: float = 0.0) -> pd.DataFrame:
        """
        Retrieve drawdown events from the database.

        Parameters
        ----------
        min_drawdown : float
            Only return events with drawdown_pct ≥ this value.

        Returns
        -------
        pd.DataFrame
        """
        if not self.db_path:
            raise RuntimeError("db_path not configured")
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM drawdown_events WHERE drawdown_pct >= ? ORDER BY start_ts",
                conn,
                params=(min_drawdown,),
            )

    # ------------------------------------------------------------------
    # Drawdown time-series
    # ------------------------------------------------------------------

    def drawdown_series(self, equity: pd.Series) -> pd.Series:
        """
        Full drawdown time series (fraction below rolling peak).

        Parameters
        ----------
        equity : pd.Series
            Cumulative equity.

        Returns
        -------
        pd.Series
            Drawdown at each bar (0 = ATH, positive values = below peak).
        """
        arr = equity.dropna().values
        peak = np.maximum.accumulate(arr)
        dd = np.where(peak > 0, (peak - arr) / peak, 0.0)
        return pd.Series(dd, index=equity.dropna().index, name="drawdown")

    def underwater_periods(
        self, equity: pd.Series, threshold: float = 0.02
    ) -> pd.Series:
        """
        Boolean mask of bars where equity is more than ``threshold`` below peak.

        Parameters
        ----------
        equity : pd.Series
            Cumulative equity.
        threshold : float
            Drawdown depth to define "underwater".

        Returns
        -------
        pd.Series[bool]
            True where the strategy is in drawdown beyond the threshold.
        """
        return self.drawdown_series(equity) > threshold

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _idx_to_str(idx: Any, position: int) -> str:
        """Convert an index element at ``position`` to an ISO-8601 string."""
        element = idx[position]
        if hasattr(element, "isoformat"):
            return element.isoformat()
        return str(element)
