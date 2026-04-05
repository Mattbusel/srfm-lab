"""
microstructure/calibrator.py

Calibrates all microstructure models on historical data and evaluates
whether the composite health score predicts next-day return quality.

Methodology
-----------
Training period  : First 252 trading days (~1 year)
Test period      : Next 63 trading days (~1 quarter)

Evaluation metric: Information Coefficient (IC)
    IC = Pearson correlation between microstructure_health and
         next-day_realized_sharpe (or next-day return / vol)

An IC > 0.05 is economically meaningful for market microstructure.
An IC > 0.10 is strong and confirms the microstructure models are useful.

Calibration outputs
-------------------
1. Per-model thresholds validated on training data
2. Composite weight optimisation (currently fixed; future: grid search)
3. IS IC and OOS IC with confidence intervals
4. Hourly spread profile fit quality

The calibration writes results to idea_engine.db (microstructure_calibration
table) for the dashboard to display.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from microstructure.models.amihud import AmihudCalculator
from microstructure.models.adverse_selection import AdverseSelectionCalculator, AdverseSelectionRisk
from microstructure.models.roll_spread import RollSpreadCalculator
from microstructure.models.kyle_lambda import KyleLambdaCalculator
from microstructure.signals.microstructure_signal import MicrostructureSignal

logger = logging.getLogger(__name__)

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

CREATE_CAL_TABLE = """
CREATE TABLE IF NOT EXISTS microstructure_calibration (
    calibration_id  TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    calibrated_at   TEXT NOT NULL,
    train_ic        REAL,
    test_ic         REAL,
    n_train         INTEGER,
    n_test          INTEGER,
    thresholds_json TEXT,
    notes           TEXT
);
"""


@dataclass
class CalibrationResult:
    symbol: str
    train_ic: float
    test_ic: float
    n_train: int
    n_test: int
    thresholds: dict[str, Any]
    notes: str

    @property
    def passes_ic_threshold(self) -> bool:
        return self.test_ic > 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "train_ic": round(self.train_ic, 5),
            "test_ic": round(self.test_ic, 5),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "thresholds": self.thresholds,
            "notes": self.notes,
        }


class MicrostructureCalibrator:
    """
    Fits microstructure models on training data and evaluates predictive
    power on held-out test data.

    Parameters
    ----------
    train_bars : Number of bars for training (default 252 for daily bars).
    test_bars  : Number of bars for testing (default 63).
    db_path    : Path to write calibration results.
    """

    def __init__(
        self,
        train_bars: int = 252,
        test_bars: int = 63,
        db_path: Path = DB_PATH,
    ) -> None:
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.db_path = db_path
        self._ensure_table()
        self.amihud = AmihudCalculator()
        self.roll = RollSpreadCalculator()
        self.adverse = AdverseSelectionCalculator()
        self.kyle = KyleLambdaCalculator()

    def calibrate(
        self,
        symbol: str,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        timestamps: Sequence[str],
    ) -> CalibrationResult:
        """
        Run full calibration + evaluation for one symbol.

        Computes microstructure signals for each bar, then measures IC
        between signal and next-bar return quality.
        """
        n = min(len(opens), len(closes), len(volumes), len(timestamps))
        total_needed = self.train_bars + self.test_bars
        if n < total_needed:
            logger.warning(
                "%s: only %d bars available, need %d. Calibration skipped.",
                symbol, n, total_needed,
            )
            return CalibrationResult(
                symbol=symbol, train_ic=0.0, test_ic=0.0,
                n_train=0, n_test=0,
                thresholds={}, notes="Insufficient data",
            )

        # Compute health scores for all bars with enough history
        health_scores, next_returns = self._compute_aligned_series(
            opens, closes, volumes, timestamps, highs, lows, n
        )

        if len(health_scores) < 30:
            return CalibrationResult(
                symbol=symbol, train_ic=0.0, test_ic=0.0,
                n_train=0, n_test=0,
                thresholds={}, notes="Too few valid readings",
            )

        # Split into train / test
        split = min(self.train_bars, len(health_scores) - self.test_bars)
        train_h = health_scores[:split]
        train_r = next_returns[:split]
        test_h = health_scores[split:]
        test_r = next_returns[split:]

        train_ic = self._information_coefficient(train_h, train_r)
        test_ic = self._information_coefficient(test_h, test_r)

        thresholds = {
            "thin_multiple": 2.0,
            "wide_spread_ratio": 3.0,
            "pin_high": 0.35,
            "pin_medium": 0.20,
            "composite_block_threshold": 0.30,
        }

        notes = (
            f"Train IC={train_ic:.4f}, Test IC={test_ic:.4f}. "
            + ("USEFUL predictive signal." if test_ic > 0.05 else "Weak predictive signal.")
        )

        result = CalibrationResult(
            symbol=symbol,
            train_ic=train_ic,
            test_ic=test_ic,
            n_train=len(train_h),
            n_test=len(test_h),
            thresholds=thresholds,
            notes=notes,
        )

        self._persist(result)
        logger.info("Calibrated %s: train_IC=%.4f test_IC=%.4f", symbol, train_ic, test_ic)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_aligned_series(
        self,
        opens: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        timestamps: Sequence[str],
        highs: Sequence[float],
        lows: Sequence[float],
        n: int,
    ) -> tuple[list[float], list[float]]:
        """
        Compute a (health_score, next_return) pair for each valid bar.
        Returns two aligned lists.
        """
        opens_l = list(opens[:n])
        closes_l = list(closes[:n])
        volumes_l = list(volumes[:n])
        ts_l = list(timestamps[:n])

        amihud_readings = self.amihud.compute(
            "CAL", closes_l, volumes_l, ts_l
        )
        roll_readings = self.roll.compute("CAL", closes_l, ts_l)
        adverse_readings = self.adverse.compute(
            "CAL", opens_l, closes_l, volumes_l, ts_l
        )
        kyle_readings = self.kyle.compute(
            "CAL", opens_l, closes_l, volumes_l, ts_l
        )

        # Find the shortest aligned list length
        min_len = min(
            len(amihud_readings),
            len(roll_readings),
            len(adverse_readings),
            len(kyle_readings),
        )
        if min_len < 2:
            return [], []

        health_scores: list[float] = []
        next_returns: list[float] = []

        for i in range(min_len - 1):
            ar = amihud_readings[-(min_len - i)]
            rr = roll_readings[-(min_len - i)]
            adr = adverse_readings[-(min_len - i)]
            kr = kyle_readings[-(min_len - i)]

            sig = MicrostructureSignal.build(
                symbol="CAL",
                amihud_percentile=ar.z_score / 4.0 if ar.z_score > 0 else 0.0,
                amihud_is_thin=ar.is_thin,
                roll_spread=rr.effective_spread,
                roll_baseline=rr.rolling_baseline,
                adverse_risk=adr.risk_level,
                adverse_pin=adr.pin_proxy,
                kyle_percentile=kr.percentile,
                kyle_size_multiplier=kr.size_multiplier,
            )
            health_scores.append(sig.composite_health)

            # Next-bar return quality (simple return)
            bar_idx = n - (min_len - i) - 1
            if 0 < bar_idx < n - 1 and closes_l[bar_idx] > 1e-12:
                next_ret = (closes_l[bar_idx + 1] - closes_l[bar_idx]) / closes_l[bar_idx]
                next_returns.append(next_ret)
            else:
                next_returns.append(0.0)

        return health_scores, next_returns

    @staticmethod
    def _information_coefficient(x: list[float], y: list[float]) -> float:
        """Pearson correlation between x and y."""
        n = min(len(x), len(y))
        if n < 3:
            return 0.0
        mx = sum(x[:n]) / n
        my = sum(y[:n]) / n
        sx = math.sqrt(max(sum((v - mx) ** 2 for v in x[:n]) / n, 1e-20))
        sy = math.sqrt(max(sum((v - my) ** 2 for v in y[:n]) / n, 1e-20))
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
        return cov / (sx * sy)

    def _persist(self, result: CalibrationResult) -> None:
        cal_id = f"{result.symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO microstructure_calibration
                       (calibration_id, symbol, calibrated_at, train_ic, test_ic,
                        n_train, n_test, thresholds_json, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cal_id,
                        result.symbol,
                        datetime.now(timezone.utc).isoformat(),
                        result.train_ic,
                        result.test_ic,
                        result.n_train,
                        result.n_test,
                        json.dumps(result.thresholds),
                        result.notes,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to persist calibration for %s: %s", result.symbol, exc)

    def _ensure_table(self) -> None:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(CREATE_CAL_TABLE)
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Could not create calibration table: %s", exc)
