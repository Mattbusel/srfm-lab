"""
live_feedback_analyzer.py -- Analyze live trading feedback and bridge it to IAE.

The Go IAE runs on a fixed adaptation schedule, but live trading produces a
continuous stream of outcomes.  This module:
  - Reads recent trade records from the live-trading SQLite database.
  - Estimates the gradient d(sharpe)/d(param) using natural experiments
    (parameter changes made by the IAE between cycles).
  - Suggests parameter adjustments via gradient ascent so the IAE can act
    faster than its normal cycle frequency.

The Go IAE is the authoritative adapter; this module only SUGGESTS.  Final
acceptance of any adjustment is the IAE's responsibility.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """A single completed trade as recorded in the live-trading database."""

    trade_id: str
    symbol: str
    side: str           # "long" | "short"
    entry_time: datetime
    exit_time: datetime
    pnl: float          # raw PnL in base currency
    return_pct: float   # percentage return for this trade
    holding_bars: int   # number of bars the trade was held


@dataclass
class FeedbackBatch:
    """
    A batch of recent live trading outcomes used for gradient estimation.

    params_at_time is a snapshot of the parameter values that were ACTIVE
    when these trades were made.
    """

    trades: List[TradeRecord]
    params_at_time: Dict[str, float]
    realized_sharpe: float  # annualized Sharpe over the batch window
    n_bars: int             # number of bars in the measurement window

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_return(self) -> float:
        return sum(t.return_pct for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)


# ---------------------------------------------------------------------------
# GradientEstimator
# ---------------------------------------------------------------------------

class GradientEstimator:
    """
    Estimate d(sharpe)/d(param) from natural experiments.

    Each time the IAE changes a parameter and we observe a Sharpe change in
    the following live window, we record (param_value, sharpe) as a data point.
    Gradient is estimated via ordinary least squares on the accumulated data.

    At least 3 data points are required before an estimate is returned.
    """

    MIN_POINTS = 3

    def __init__(self) -> None:
        # {param_name: [(param_value, sharpe_change), ...]}
        self._data: Dict[str, List[Tuple[float, float]]] = {}

    def update(
        self,
        param_name: str,
        old_value: float,
        new_value: float,
        sharpe_change: float,
    ) -> None:
        """
        Record one natural experiment: parameter moved from old_value to
        new_value and the Sharpe changed by sharpe_change.

        We store (mid_point_value, sharpe_change/delta) as a gradient sample.
        """
        delta = new_value - old_value
        if abs(delta) < 1e-12:
            logger.debug("Skipping gradient update for '%s': delta is ~zero", param_name)
            return

        mid_value = (old_value + new_value) / 2.0
        grad_sample = sharpe_change / delta  # finite-difference gradient estimate

        self._data.setdefault(param_name, []).append((mid_value, grad_sample))
        logger.debug(
            "Gradient update for '%s': mid=%.4f  sharpe_chg=%.4f  grad_sample=%.6f",
            param_name, mid_value, sharpe_change, grad_sample,
        )

    def estimate(self, param_name: str) -> Optional[float]:
        """
        Return the current gradient estimate for param_name as a weighted
        mean of accumulated samples (more recent samples weighted higher).

        Returns None if fewer than MIN_POINTS samples are available.
        """
        samples = self._data.get(param_name, [])
        if len(samples) < self.MIN_POINTS:
            logger.debug(
                "Not enough data for gradient estimate of '%s' (%d < %d)",
                param_name, len(samples), self.MIN_POINTS,
            )
            return None

        grads = np.array([g for _, g in samples])
        # Exponential weights -- more recent samples (end of list) get higher weight
        weights = np.exp(np.linspace(0.0, 1.0, len(grads)))
        weights /= weights.sum()

        return float(np.dot(weights, grads))

    def estimate_all(self) -> Dict[str, float]:
        """Return gradient estimates for all parameters with enough data."""
        result = {}
        for param_name in self._data:
            est = self.estimate(param_name)
            if est is not None:
                result[param_name] = est
        return result

    def r_squared(self, param_name: str) -> Optional[float]:
        """
        Return the R^2 of a simple linear OLS fit of param_value -> gradient
        for the accumulated data, as a rough quality measure.

        Returns None if fewer than MIN_POINTS samples are available.
        """
        samples = self._data.get(param_name, [])
        if len(samples) < self.MIN_POINTS:
            return None

        xs = np.array([v for v, _ in samples])
        ys = np.array([g for _, g in samples])

        if xs.std() < 1e-12:
            return None

        coeffs = np.polyfit(xs, ys, deg=1)
        y_pred = np.polyval(coeffs, xs)
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        if ss_tot < 1e-12:
            return None
        return 1.0 - ss_res / ss_tot

    def sample_count(self, param_name: str) -> int:
        return len(self._data.get(param_name, []))

    def clear(self, param_name: Optional[str] = None) -> None:
        """Clear accumulated data for one or all parameters."""
        if param_name is None:
            self._data.clear()
        else:
            self._data.pop(param_name, None)


# ---------------------------------------------------------------------------
# LiveFeedbackAnalyzer
# ---------------------------------------------------------------------------

# Expected live-trading SQLite schema (written by Go execution engine):
#
#   CREATE TABLE trades (
#       id            TEXT PRIMARY KEY,
#       symbol        TEXT NOT NULL,
#       side          TEXT NOT NULL,
#       entry_time    TEXT NOT NULL,   -- ISO-8601
#       exit_time     TEXT NOT NULL,   -- ISO-8601
#       pnl           REAL NOT NULL,
#       return_pct    REAL NOT NULL,
#       holding_bars  INTEGER NOT NULL
#   );
#
#   CREATE TABLE param_snapshots (
#       id            INTEGER PRIMARY KEY AUTOINCREMENT,
#       snapshot_time TEXT NOT NULL,
#       params_json   TEXT NOT NULL    -- JSON object
#   );

class LiveFeedbackAnalyzer:
    """
    Bridge live trading outcomes back to the IAE for faster parameter adaptation.

    Typical usage:
        analyzer = LiveFeedbackAnalyzer(param_bounds, gradient_estimator)
        batch    = analyzer.collect_recent_feedback("live.db", hours=4)
        gradient = analyzer.compute_gradient_estimate(batch, current_params)
        adj      = analyzer.suggest_param_adjustments(gradient)
        if analyzer.validate_suggestion(adj):
            # send adj to IAE
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        gradient_estimator: Optional[GradientEstimator] = None,
        max_delta_pct: float = 0.05,  # max allowed adjustment as fraction of param range
    ) -> None:
        self.param_bounds = param_bounds
        self.gradient_estimator = gradient_estimator or GradientEstimator()
        self.max_delta_pct = max_delta_pct

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_recent_feedback(
        self,
        db_path: str,
        hours: int = 4,
    ) -> FeedbackBatch:
        """
        Load trades from the last `hours` hours and the most recent param
        snapshot from the live-trading SQLite database.

        Returns a FeedbackBatch.  If no trades are found returns an empty batch.
        """
        import json

        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            logger.error("Cannot open live DB %s: %s", db_path, exc)
            return FeedbackBatch(
                trades=[], params_at_time={}, realized_sharpe=0.0, n_bars=0
            )

        try:
            # Load trades
            cur = conn.execute(
                "SELECT * FROM trades WHERE exit_time >= ? ORDER BY exit_time ASC",
                (cutoff_str,),
            )
            trade_rows = cur.fetchall()

            trades: List[TradeRecord] = []
            for row in trade_rows:
                try:
                    trades.append(
                        TradeRecord(
                            trade_id=row["id"],
                            symbol=row["symbol"],
                            side=row["side"],
                            entry_time=datetime.fromisoformat(row["entry_time"]),
                            exit_time=datetime.fromisoformat(row["exit_time"]),
                            pnl=float(row["pnl"]),
                            return_pct=float(row["return_pct"]),
                            holding_bars=int(row["holding_bars"]),
                        )
                    )
                except (KeyError, ValueError) as exc:
                    logger.warning("Skipping malformed trade row: %s", exc)

            # Load most recent param snapshot
            cur2 = conn.execute(
                "SELECT params_json FROM param_snapshots ORDER BY snapshot_time DESC LIMIT 1"
            )
            snap = cur2.fetchone()
            params_at_time: Dict[str, float] = {}
            if snap:
                try:
                    params_at_time = json.loads(snap["params_json"])
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Could not parse params_json: %s", exc)

        except sqlite3.Error as exc:
            logger.error("DB query error in collect_recent_feedback: %s", exc)
            return FeedbackBatch(
                trades=[], params_at_time={}, realized_sharpe=0.0, n_bars=0
            )
        finally:
            conn.close()

        realized_sharpe = self._compute_sharpe_from_trades(trades)
        n_bars = sum(t.holding_bars for t in trades)

        logger.info(
            "Collected %d trades over last %d h, realized_sharpe=%.4f",
            len(trades), hours, realized_sharpe,
        )
        return FeedbackBatch(
            trades=trades,
            params_at_time=params_at_time,
            realized_sharpe=realized_sharpe,
            n_bars=n_bars,
        )

    @staticmethod
    def _compute_sharpe_from_trades(
        trades: List[TradeRecord],
        annualize_factor: float = 252.0,
    ) -> float:
        """
        Compute a simple annualized Sharpe ratio from a list of trade returns.
        Returns 0.0 if there are fewer than 2 trades.
        """
        if len(trades) < 2:
            return 0.0

        returns = np.array([t.return_pct for t in trades])
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))

        if std_ret < 1e-10:
            return 0.0

        sharpe = (mean_ret / std_ret) * np.sqrt(annualize_factor)
        return float(sharpe)

    # ------------------------------------------------------------------
    # Gradient estimation
    # ------------------------------------------------------------------

    def compute_gradient_estimate(
        self,
        feedback: FeedbackBatch,
        current_params: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Estimate d(sharpe)/d(param) using the feedback batch and the
        GradientEstimator's accumulated natural-experiment data.

        For params where the estimator has no historical data, we use a
        finite-difference approximation based on the params_at_time snapshot
        vs current_params and the Sharpe change implied by feedback.

        Returns a dict {param_name: gradient_estimate}.
        """
        gradients: Dict[str, float] = {}

        # Use accumulated estimator data first
        gradients.update(self.gradient_estimator.estimate_all())

        # For params not yet in the estimator, attempt a single FD step
        if feedback.params_at_time and feedback.realized_sharpe != 0.0:
            for param_name, current_val in current_params.items():
                if param_name in gradients:
                    continue  # already have an estimate
                old_val = feedback.params_at_time.get(param_name)
                if old_val is None:
                    continue
                delta = current_val - old_val
                if abs(delta) < 1e-12:
                    continue
                # We don't know the old Sharpe, so we can't compute a change.
                # Record zero as a placeholder -- the estimator will fill in
                # when a proper natural experiment is available.
                gradients[param_name] = 0.0

        logger.debug("Gradient estimates: %s", gradients)
        return gradients

    # ------------------------------------------------------------------
    # Parameter adjustment suggestions
    # ------------------------------------------------------------------

    def suggest_param_adjustments(
        self,
        gradient: Dict[str, float],
        learning_rate: float = 0.1,
    ) -> Dict[str, float]:
        """
        Suggest parameter adjustments using gradient ascent clipped to
        max_delta_pct of the parameter range.

        Returns {param_name: suggested_new_value}.
        """
        suggestions: Dict[str, float] = {}

        for param_name, grad in gradient.items():
            if param_name not in self.param_bounds:
                logger.debug(
                    "Skipping suggestion for '%s': not in param_bounds", param_name
                )
                continue

            lo, hi = self.param_bounds[param_name]
            param_range = hi - lo
            max_delta = param_range * self.max_delta_pct

            current = self.get_current_value(param_name)
            if current is None:
                continue

            raw_delta = learning_rate * grad
            clipped_delta = float(np.clip(raw_delta, -max_delta, max_delta))
            new_val = float(np.clip(current + clipped_delta, lo, hi))

            suggestions[param_name] = new_val

        logger.info(
            "Parameter adjustment suggestions: %d params",
            len(suggestions),
        )
        return suggestions

    def get_current_value(self, param_name: str) -> Optional[float]:
        """
        Return the current known value for a parameter (from the most recent
        gradient estimator data, or None).
        """
        samples = self.gradient_estimator._data.get(param_name, [])
        if samples:
            return float(samples[-1][0])  # last recorded mid-point value
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_suggestion(self, suggestion: Dict[str, float]) -> bool:
        """
        Return True only if all suggested values lie within their declared
        parameter bounds and the dict is not empty.
        """
        if not suggestion:
            logger.warning("validate_suggestion: empty suggestion dict")
            return False

        for param_name, val in suggestion.items():
            if param_name not in self.param_bounds:
                logger.warning(
                    "validate_suggestion: '%s' not in param_bounds", param_name
                )
                return False
            lo, hi = self.param_bounds[param_name]
            if not (lo <= val <= hi):
                logger.warning(
                    "validate_suggestion: '%s'=%.6f out of bounds [%.4f, %.4f]",
                    param_name, val, lo, hi,
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Feedback ingestion for natural experiments
    # ------------------------------------------------------------------

    def ingest_natural_experiment(
        self,
        param_name: str,
        old_value: float,
        new_value: float,
        sharpe_before: float,
        sharpe_after: float,
    ) -> None:
        """
        Record a natural experiment: the IAE changed param_name from old_value
        to new_value and live Sharpe moved from sharpe_before to sharpe_after.
        Feeds the gradient estimator.
        """
        sharpe_change = sharpe_after - sharpe_before
        self.gradient_estimator.update(param_name, old_value, new_value, sharpe_change)

    # ------------------------------------------------------------------
    # Diagnostic summary
    # ------------------------------------------------------------------

    def gradient_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with gradient estimates and data quality for all
        parameters tracked by the gradient estimator.
        """
        rows = []
        for param_name in self.gradient_estimator._data:
            est = self.gradient_estimator.estimate(param_name)
            r2 = self.gradient_estimator.r_squared(param_name)
            n = self.gradient_estimator.sample_count(param_name)
            rows.append(
                {
                    "param": param_name,
                    "gradient_estimate": est,
                    "r_squared": r2,
                    "n_samples": n,
                    "has_estimate": est is not None,
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=["param", "gradient_estimate", "r_squared", "n_samples", "has_estimate"]
            )

        df = pd.DataFrame(rows).set_index("param")
        df.sort_values("n_samples", ascending=False, inplace=True)
        return df

    def adjustment_report(
        self,
        gradient: Dict[str, float],
        learning_rate: float = 0.1,
    ) -> str:
        """
        Generate a plain-text report describing the suggested adjustments
        and their expected Sharpe impact.
        """
        suggestions = self.suggest_param_adjustments(gradient, learning_rate)
        valid = self.validate_suggestion(suggestions)

        lines = [
            "# Live Feedback Adjustment Report",
            "",
            f"Gradient estimates available for {len(gradient)} parameters.",
            f"Suggestions valid: {valid}",
            "",
            "## Suggested Adjustments",
            "",
        ]

        if not suggestions:
            lines.append("No adjustments suggested (insufficient gradient data).")
        else:
            for pname, new_val in suggestions.items():
                current = self.get_current_value(pname)
                grad = gradient.get(pname, 0.0)
                delta = new_val - (current or 0.0)
                lines.append(
                    f"- **{pname}**: {current:.4f} -> {new_val:.4f}  "
                    f"(delta={delta:+.4f}, gradient={grad:+.6f})"
                )

        lines += [
            "",
            "## Gradient Quality",
            "",
        ]
        summary = self.gradient_summary()
        if summary.empty:
            lines.append("No gradient data accumulated yet.")
        else:
            for param, row in summary.iterrows():
                r2_str = f"{row['r_squared']:.3f}" if row["r_squared"] is not None else "N/A"
                lines.append(
                    f"- {param}: n={int(row['n_samples'])}  r2={r2_str}  "
                    f"grad={row['gradient_estimate']}"
                )

        return "\n".join(lines)
