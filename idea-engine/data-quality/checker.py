"""
DataQualityChecker
==================
Core OHLCV data quality validation for the Idea Automation Engine.

Checks performed on each DataFrame:
  1. Gap detection            — missing bars vs expected bar interval
  2. Outlier detection        — price moves > N std devs from rolling mean
  3. OHLC validity            — High >= Low, High >= Open/Close, Low <= Open/Close
  4. Volume anomalies         — zero-volume bars, volume spikes > 10× rolling mean
  5. Stale prices             — N consecutive identical close prices
  6. Flash crash detection    — >10 % single-bar move with immediate reversal
  7. Timestamp consistency    — monotonically increasing, no duplicates
  8. Bid-ask spread proxy     — High-Low range vs expected ATR

Results are stored in ``data_quality_reports`` (see schema_extension.sql).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV = "IDEA_ENGINE_DB"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "idea_engine.db"

# Outlier threshold: moves beyond this many rolling std devs are flagged
OUTLIER_STD_THRESHOLD = 5.0

# Rolling window for statistics
ROLLING_WINDOW = 20

# Consecutive identical closes before declaring stale
STALE_CONSECUTIVE = 5

# Flash-crash threshold (single-bar percentage move)
FLASH_CRASH_PCT = 0.10

# Volume spike threshold (multiple of rolling mean)
VOLUME_SPIKE_MULTIPLE = 10.0

# ATR window for spread-proxy check
ATR_WINDOW = 14

# Expected spread-to-ATR ratio ceiling (flag if range > this × ATR)
SPREAD_ATR_CEILING = 3.0

# Score penalties applied per issue found
SCORE_PENALTIES: dict[str, float] = {
    "gap":           2.0,
    "outlier":       3.0,
    "ohlc_invalid":  5.0,
    "zero_volume":   1.0,
    "volume_spike":  2.0,
    "stale_price":   4.0,
    "flash_crash":   3.0,
    "ts_duplicate":  5.0,
    "ts_nonmono":    5.0,
    "wide_spread":   1.5,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QualityIssue:
    """A single data quality violation."""

    check_name: str          # e.g. "gap", "outlier"
    severity: str            # "critical" | "warning" | "info"
    bar_index: Any           # timestamp or integer index of the offending bar
    description: str
    value: float | None = None   # numeric value that triggered the flag
    expected: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check_name,
            "severity": self.severity,
            "bar_index": str(self.bar_index),
            "description": self.description,
            "value": self.value,
            "expected": self.expected,
        }


@dataclass
class QualityReport:
    """Aggregated result of all quality checks for one OHLCV series."""

    symbol: str
    data_source: str
    score: float                           # 0–100; 100 = perfect
    issues: list[QualityIssue] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    n_bars: int = 0
    n_gaps: int = 0
    n_outliers: int = 0
    n_stale_bars: int = 0
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    # ---- convenience properties ----------------------------------------

    @property
    def is_clean(self) -> bool:
        return self.score >= 90.0

    @property
    def critical_issues(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    def summary(self) -> str:
        return (
            f"QualityReport[{self.symbol}] score={self.score:.1f}/100 "
            f"issues={len(self.issues)} (critical={len(self.critical_issues)}) "
            f"gaps={self.n_gaps} outliers={self.n_outliers} stale={self.n_stale_bars}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "data_source": self.data_source,
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "recommendations": self.recommendations,
            "n_bars": self.n_bars,
            "n_gaps": self.n_gaps,
            "n_outliers": self.n_outliers,
            "n_stale_bars": self.n_stale_bars,
            "checked_at": self.checked_at,
        }


# ---------------------------------------------------------------------------
# RepairLog
# ---------------------------------------------------------------------------

@dataclass
class RepairAction:
    bar_index: Any
    action: str         # "drop", "interpolate", "cap", "fill_forward"
    column: str
    original_value: float | None
    new_value: float | None


# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------

class DataQualityChecker:
    """
    Validates OHLCV DataFrames for a wide range of quality issues.

    Parameters
    ----------
    db_path : path to idea_engine.db; falls back to IDEA_ENGINE_DB env var.
    data_source : label for the source feed (e.g. "binance", "coinbase").
    outlier_std : threshold multiplier for outlier detection.
    stale_n : number of consecutive identical closes before flagging stale.
    flash_pct : single-bar absolute-return threshold for flash-crash flag.
    volume_spike_multiple : volume multiple above rolling mean to flag spike.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        data_source: str = "unknown",
        outlier_std: float = OUTLIER_STD_THRESHOLD,
        stale_n: int = STALE_CONSECUTIVE,
        flash_pct: float = FLASH_CRASH_PCT,
        volume_spike_multiple: float = VOLUME_SPIKE_MULTIPLE,
    ) -> None:
        self.db_path = Path(
            db_path
            or __import__("os").environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        self.data_source = data_source
        self.outlier_std = outlier_std
        self.stale_n = stale_n
        self.flash_pct = flash_pct
        self.volume_spike_multiple = volume_spike_multiple
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        expected_interval: str | None = None,
    ) -> QualityReport:
        """
        Run all quality checks on *df* and return a :class:`QualityReport`.

        Parameters
        ----------
        df : DataFrame with columns open, high, low, close, volume and a
             DatetimeIndex (or a 'timestamp'/'ts' column).
        symbol : ticker symbol string, e.g. "BTC/USDT".
        expected_interval : pandas offset string for the expected bar spacing,
                            e.g. "15T", "1H".  Auto-inferred when None.
        """
        df = self._normalise(df)
        issues: list[QualityIssue] = []

        # Run all checks, accumulating issues
        issues += self._check_timestamps(df)
        issues += self._check_ohlc_validity(df)
        issues += self._check_gaps(df, expected_interval)
        issues += self._check_outliers(df)
        issues += self._check_volume(df)
        issues += self._check_stale_prices(df)
        issues += self._check_flash_crashes(df)
        issues += self._check_spread_proxy(df)

        score = self._compute_score(issues, len(df))
        recommendations = self._build_recommendations(issues)

        report = QualityReport(
            symbol=symbol,
            data_source=self.data_source,
            score=score,
            issues=issues,
            recommendations=recommendations,
            n_bars=len(df),
            n_gaps=sum(1 for i in issues if i.check_name == "gap"),
            n_outliers=sum(1 for i in issues if i.check_name == "outlier"),
            n_stale_bars=sum(1 for i in issues if i.check_name == "stale_price"),
        )

        self._persist_report(report)
        logger.info(report.summary())
        return report

    def repair(
        self,
        df: pd.DataFrame,
        report: QualityReport,
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        """
        Attempt automatic repair of issues described in *report*.

        Strategy:
          - Duplicate timestamps: keep first occurrence.
          - OHLC invalid rows: cap / clamp inconsistencies.
          - Zero-volume bars: mark with NaN volume (caller decides fill).
          - Stale-price runs: replace with NaN (forward-fill later).
          - Outlier closes: winsorise to rolling mean ± outlier_std × rolling_std.
          - Gaps: re-index to expected frequency, forward-fill missing bars.

        Returns the repaired DataFrame and a log of actions taken.
        """
        df = self._normalise(df.copy())
        actions: list[RepairAction] = []

        # 1. Drop duplicate timestamps
        dup_mask = df.index.duplicated(keep="first")
        if dup_mask.any():
            for idx in df.index[dup_mask]:
                actions.append(RepairAction(idx, "drop", "index", None, None))
            df = df[~dup_mask]

        # 2. Cap OHLC inconsistencies
        df, ohlc_actions = self._repair_ohlc(df)
        actions.extend(ohlc_actions)

        # 3. Zero volume → NaN
        zero_vol = df["volume"] == 0
        if zero_vol.any():
            for idx in df.index[zero_vol]:
                actions.append(RepairAction(idx, "fill_forward", "volume",
                                            0.0, float("nan")))
            df.loc[zero_vol, "volume"] = float("nan")

        # 4. Stale-price runs → NaN close then forward-fill
        df, stale_actions = self._repair_stale(df)
        actions.extend(stale_actions)

        # 5. Outlier closes → winsorise
        df, outlier_actions = self._repair_outliers(df)
        actions.extend(outlier_actions)

        # 6. Fill gaps via reindex + forward-fill
        df, gap_actions = self._repair_gaps(df)
        actions.extend(gap_actions)

        logger.info(
            "Repair complete for %s: %d actions applied.",
            report.symbol, len(actions)
        )
        return df, actions

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DatetimeIndex and lower-case column names."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Convert timestamp column → index if needed
        for ts_col in ("timestamp", "ts", "datetime", "date"):
            if ts_col in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                df = df.set_index(ts_col)
                break

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _infer_interval(df: pd.DataFrame) -> pd.Timedelta | None:
        """Infer the modal bar interval from the index."""
        if len(df) < 3:
            return None
        diffs = df.index.to_series().diff().dropna()
        if diffs.empty:
            return None
        return diffs.mode().iloc[0]

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_timestamps(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []

        # Duplicates
        dup_mask = df.index.duplicated(keep=False)
        dup_indices = df.index[dup_mask].unique()
        for idx in dup_indices:
            issues.append(QualityIssue(
                check_name="ts_duplicate",
                severity="critical",
                bar_index=idx,
                description=f"Duplicate timestamp at {idx}",
            ))

        # Non-monotonic
        if not df.index.is_monotonic_increasing:
            bad = df.index[df.index.to_series().diff() < pd.Timedelta(0)]
            for idx in bad:
                issues.append(QualityIssue(
                    check_name="ts_nonmono",
                    severity="critical",
                    bar_index=idx,
                    description=f"Non-monotonic timestamp at {idx}",
                ))

        return issues

    def _check_ohlc_validity(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            logger.warning("DataFrame missing OHLC columns; skipping OHLC check.")
            return issues

        # high >= low
        bad = df["high"] < df["low"]
        for idx in df.index[bad]:
            issues.append(QualityIssue(
                check_name="ohlc_invalid",
                severity="critical",
                bar_index=idx,
                description=f"High < Low at {idx}",
                value=float(df.loc[idx, "high"]),
                expected=float(df.loc[idx, "low"]),
            ))

        # high >= open
        bad = df["high"] < df["open"]
        for idx in df.index[bad]:
            issues.append(QualityIssue(
                check_name="ohlc_invalid",
                severity="critical",
                bar_index=idx,
                description=f"High < Open at {idx}",
                value=float(df.loc[idx, "high"]),
                expected=float(df.loc[idx, "open"]),
            ))

        # high >= close
        bad = df["high"] < df["close"]
        for idx in df.index[bad]:
            issues.append(QualityIssue(
                check_name="ohlc_invalid",
                severity="critical",
                bar_index=idx,
                description=f"High < Close at {idx}",
                value=float(df.loc[idx, "high"]),
                expected=float(df.loc[idx, "close"]),
            ))

        # low <= open
        bad = df["low"] > df["open"]
        for idx in df.index[bad]:
            issues.append(QualityIssue(
                check_name="ohlc_invalid",
                severity="critical",
                bar_index=idx,
                description=f"Low > Open at {idx}",
                value=float(df.loc[idx, "low"]),
                expected=float(df.loc[idx, "open"]),
            ))

        # low <= close
        bad = df["low"] > df["close"]
        for idx in df.index[bad]:
            issues.append(QualityIssue(
                check_name="ohlc_invalid",
                severity="critical",
                bar_index=idx,
                description=f"Low > Close at {idx}",
                value=float(df.loc[idx, "low"]),
                expected=float(df.loc[idx, "close"]),
            ))

        # Negative prices
        for col in ("open", "high", "low", "close"):
            neg = df[col] <= 0
            for idx in df.index[neg]:
                issues.append(QualityIssue(
                    check_name="ohlc_invalid",
                    severity="critical",
                    bar_index=idx,
                    description=f"Non-positive {col} at {idx}",
                    value=float(df.loc[idx, col]),
                ))

        return issues

    def _check_gaps(
        self,
        df: pd.DataFrame,
        expected_interval: str | None,
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if len(df) < 3:
            return issues

        if expected_interval is not None:
            interval = pd.tseries.frequencies.to_offset(expected_interval)
            expected_td = pd.Timedelta(interval)
        else:
            inferred = self._infer_interval(df)
            if inferred is None:
                return issues
            expected_td = inferred

        diffs = df.index.to_series().diff().dropna()
        tolerance = expected_td * 1.5

        gap_mask = diffs > tolerance
        for idx, gap_size in diffs[gap_mask].items():
            missing_bars = int(round(gap_size / expected_td)) - 1
            issues.append(QualityIssue(
                check_name="gap",
                severity="warning",
                bar_index=idx,
                description=(
                    f"Gap of {gap_size} at {idx}; "
                    f"~{missing_bars} missing bar(s) "
                    f"(expected {expected_td})"
                ),
                value=gap_size.total_seconds(),
                expected=expected_td.total_seconds(),
            ))

        return issues

    def _check_outliers(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if "close" not in df.columns or len(df) < ROLLING_WINDOW + 1:
            return issues

        returns = df["close"].pct_change()
        roll_mean = returns.rolling(ROLLING_WINDOW, min_periods=5).mean()
        roll_std = returns.rolling(ROLLING_WINDOW, min_periods=5).std()

        z_score = (returns - roll_mean) / roll_std.replace(0, float("nan"))
        outlier_mask = z_score.abs() > self.outlier_std

        for idx in df.index[outlier_mask]:
            z = float(z_score.loc[idx])
            issues.append(QualityIssue(
                check_name="outlier",
                severity="warning",
                bar_index=idx,
                description=(
                    f"Return at {idx} is {z:.1f} std devs from rolling mean"
                ),
                value=float(returns.loc[idx]),
                expected=float(roll_mean.loc[idx]) if not pd.isna(roll_mean.loc[idx]) else None,
            ))

        return issues

    def _check_volume(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if "volume" not in df.columns:
            return issues

        # Zero-volume bars
        zero = df["volume"] == 0
        for idx in df.index[zero]:
            issues.append(QualityIssue(
                check_name="zero_volume",
                severity="warning",
                bar_index=idx,
                description=f"Zero volume at {idx}",
                value=0.0,
            ))

        # Volume spikes
        if len(df) >= ROLLING_WINDOW:
            roll_vol = df["volume"].rolling(ROLLING_WINDOW, min_periods=5).mean()
            spike_mask = df["volume"] > (roll_vol * self.volume_spike_multiple)
            for idx in df.index[spike_mask]:
                vol = float(df.loc[idx, "volume"])
                avg = float(roll_vol.loc[idx])
                issues.append(QualityIssue(
                    check_name="volume_spike",
                    severity="warning",
                    bar_index=idx,
                    description=(
                        f"Volume spike at {idx}: {vol:.0f} vs rolling avg {avg:.0f} "
                        f"({vol/avg:.1f}×)"
                    ),
                    value=vol,
                    expected=avg,
                ))

        return issues

    def _check_stale_prices(self, df: pd.DataFrame) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if "close" not in df.columns:
            return issues

        closes = df["close"]
        n = len(closes)
        if n < self.stale_n:
            return issues

        run_length = 1
        for i in range(1, n):
            if closes.iloc[i] == closes.iloc[i - 1]:
                run_length += 1
                if run_length == self.stale_n:
                    idx = closes.index[i]
                    issues.append(QualityIssue(
                        check_name="stale_price",
                        severity="warning",
                        bar_index=idx,
                        description=(
                            f"{self.stale_n} consecutive identical closes "
                            f"ending at {idx} (price={closes.iloc[i]})"
                        ),
                        value=float(closes.iloc[i]),
                    ))
            else:
                run_length = 1

        return issues

    def _check_flash_crashes(self, df: pd.DataFrame) -> list[QualityIssue]:
        """
        Flag a bar where |return| > flash_pct AND the next bar reverses > 50 %
        of the move.
        """
        issues: list[QualityIssue] = []
        if "close" not in df.columns or len(df) < 3:
            return issues

        closes = df["close"].values
        for i in range(1, len(closes) - 1):
            ret = (closes[i] - closes[i - 1]) / closes[i - 1]
            if abs(ret) < self.flash_pct:
                continue
            reversal = (closes[i + 1] - closes[i]) / closes[i]
            if ret > 0 and reversal < -0.5 * ret:
                idx = df.index[i]
                issues.append(QualityIssue(
                    check_name="flash_crash",
                    severity="warning",
                    bar_index=idx,
                    description=(
                        f"Flash spike at {idx}: +{ret:.1%} then reversal {reversal:.1%}"
                    ),
                    value=ret,
                ))
            elif ret < 0 and reversal > -0.5 * ret:
                idx = df.index[i]
                issues.append(QualityIssue(
                    check_name="flash_crash",
                    severity="warning",
                    bar_index=idx,
                    description=(
                        f"Flash crash at {idx}: {ret:.1%} then reversal {reversal:.1%}"
                    ),
                    value=ret,
                ))

        return issues

    def _check_spread_proxy(self, df: pd.DataFrame) -> list[QualityIssue]:
        """
        Flag bars where High-Low range >> expected ATR (potential bad tick data).
        """
        issues: list[QualityIssue] = []
        if not {"high", "low", "close"}.issubset(df.columns):
            return issues
        if len(df) < ATR_WINDOW + 5:
            return issues

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(ATR_WINDOW, min_periods=5).mean()
        hl_range = df["high"] - df["low"]
        ratio = hl_range / atr.replace(0, float("nan"))

        wide = ratio > SPREAD_ATR_CEILING
        for idx in df.index[wide]:
            r = float(ratio.loc[idx])
            issues.append(QualityIssue(
                check_name="wide_spread",
                severity="info",
                bar_index=idx,
                description=(
                    f"H-L range at {idx} is {r:.1f}× ATR "
                    f"(threshold {SPREAD_ATR_CEILING}×)"
                ),
                value=r,
                expected=SPREAD_ATR_CEILING,
            ))

        return issues

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(self, issues: list[QualityIssue], n_bars: int) -> float:
        """
        Score = 100 − (total penalty / n_bars × 100), clipped to [0, 100].

        Critical issues carry double penalty weight.
        """
        if n_bars == 0:
            return 0.0

        total_penalty = 0.0
        for issue in issues:
            base = SCORE_PENALTIES.get(issue.check_name, 2.0)
            multiplier = 2.0 if issue.severity == "critical" else 1.0
            total_penalty += base * multiplier

        # Normalise so a handful of issues on a large dataset isn't catastrophic
        normalised = (total_penalty / n_bars) * 100
        score = max(0.0, min(100.0, 100.0 - normalised))
        return round(score, 2)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _build_recommendations(self, issues: list[QualityIssue]) -> list[str]:
        counts: dict[str, int] = {}
        for issue in issues:
            counts[issue.check_name] = counts.get(issue.check_name, 0) + 1

        recs: list[str] = []
        if counts.get("gap", 0):
            recs.append(
                f"Found {counts['gap']} gap(s). Consider forward-filling or "
                "sourcing a cleaner feed."
            )
        if counts.get("outlier", 0):
            recs.append(
                f"Found {counts['outlier']} price outlier(s). Review raw ticks "
                "and consider winsorising before backtesting."
            )
        if counts.get("ohlc_invalid", 0):
            recs.append(
                f"Found {counts['ohlc_invalid']} OHLC invalidity violation(s). "
                "Likely bad tick data — do not use without repair."
            )
        if counts.get("stale_price", 0):
            recs.append(
                f"Found {counts['stale_price']} stale-price run(s). Feed may have "
                "dropped or repeated ticks."
            )
        if counts.get("flash_crash", 0):
            recs.append(
                f"Found {counts['flash_crash']} flash crash/spike event(s). "
                "Verify against a secondary exchange."
            )
        if counts.get("ts_duplicate", 0) or counts.get("ts_nonmono", 0):
            recs.append(
                "Timestamp integrity issues detected. Sort and deduplicate before use."
            )
        if counts.get("zero_volume", 0):
            recs.append(
                f"Found {counts['zero_volume']} zero-volume bar(s). "
                "These bars may be illiquid or represent exchange downtime."
            )
        if counts.get("wide_spread", 0):
            recs.append(
                f"Found {counts['wide_spread']} bar(s) with unusually wide H-L range. "
                "Potential bad ticks or extreme volatility events."
            )
        return recs

    # ------------------------------------------------------------------
    # Repair helpers
    # ------------------------------------------------------------------

    def _repair_ohlc(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []

        # Clamp: high = max(open, high, low, close)
        #        low  = min(open, high, low, close)
        for idx in df.index:
            row = df.loc[idx]
            prices = [row["open"], row["high"], row["low"], row["close"]]
            if any(pd.isna(p) for p in prices):
                continue
            true_high = max(prices)
            true_low = min(prices)
            if row["high"] != true_high:
                actions.append(RepairAction(idx, "cap", "high", row["high"], true_high))
                df.loc[idx, "high"] = true_high
            if row["low"] != true_low:
                actions.append(RepairAction(idx, "cap", "low", row["low"], true_low))
                df.loc[idx, "low"] = true_low

        return df, actions

    def _repair_stale(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        closes = df["close"].copy()
        n = len(closes)
        run_start = 0

        for i in range(1, n):
            if closes.iloc[i] == closes.iloc[i - 1]:
                continue
            run_len = i - run_start
            if run_len >= self.stale_n:
                # NaN the interior of the run (keep first, NaN the rest)
                for j in range(run_start + 1, i):
                    idx = df.index[j]
                    actions.append(
                        RepairAction(idx, "fill_forward", "close",
                                     float(closes.iloc[j]), float("nan"))
                    )
                    df.loc[idx, "close"] = float("nan")
            run_start = i

        # Forward-fill the NaN closes we just introduced
        df["close"] = df["close"].ffill()
        return df, actions

    def _repair_outliers(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        if "close" not in df.columns or len(df) < ROLLING_WINDOW + 1:
            return df, actions

        returns = df["close"].pct_change()
        roll_mean = returns.rolling(ROLLING_WINDOW, min_periods=5).mean()
        roll_std = returns.rolling(ROLLING_WINDOW, min_periods=5).std()

        z_score = (returns - roll_mean) / roll_std.replace(0, float("nan"))
        outlier_mask = z_score.abs() > self.outlier_std

        prev_close = df["close"].shift(1)
        for idx in df.index[outlier_mask]:
            if pd.isna(prev_close.loc[idx]):
                continue
            original = float(df.loc[idx, "close"])
            sign = 1.0 if float(returns.loc[idx]) > 0 else -1.0
            capped_return = sign * self.outlier_std * float(roll_std.loc[idx])
            new_close = float(prev_close.loc[idx]) * (1.0 + capped_return)
            actions.append(RepairAction(idx, "cap", "close", original, new_close))
            df.loc[idx, "close"] = new_close

        return df, actions

    def _repair_gaps(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        interval = self._infer_interval(df)
        if interval is None or len(df) < 3:
            return df, actions

        full_idx = pd.date_range(
            start=df.index[0], end=df.index[-1], freq=interval
        )
        missing = full_idx.difference(df.index)
        if missing.empty:
            return df, actions

        df = df.reindex(full_idx)
        for idx in missing:
            actions.append(RepairAction(idx, "interpolate", "all", None, None))

        df = df.ffill()
        return df, actions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        sql_path = Path(__file__).parent / "schema_extension.sql"
        if not sql_path.exists():
            logger.warning("schema_extension.sql not found; skipping schema creation.")
            return
        if not self.db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.executescript(sql_path.read_text(encoding="utf-8"))
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Could not apply data-quality schema: %s", exc)

    def _persist_report(self, report: QualityReport) -> None:
        if not self.db_path.exists():
            logger.debug("DB not found at %s; skipping persist.", self.db_path)
            return
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """
                INSERT INTO data_quality_reports
                    (symbol, data_source, quality_score, issues_json,
                     n_gaps, n_outliers, n_stale_bars, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.symbol,
                    report.data_source,
                    report.score,
                    json.dumps([i.to_dict() for i in report.issues]),
                    report.n_gaps,
                    report.n_outliers,
                    report.n_stale_bars,
                    report.checked_at,
                ),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.warning("Failed to persist quality report: %s", exc)

    def load_reports(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Load recent quality reports from the database."""
        if not self.db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM data_quality_reports WHERE symbol = ? "
                    "ORDER BY checked_at DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM data_quality_reports "
                    "ORDER BY checked_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.warning("Failed to load quality reports: %s", exc)
            return []

    def worst_symbols(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return symbols with the lowest average quality scores."""
        if not self.db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT symbol, AVG(quality_score) AS avg_score,
                       COUNT(*) AS n_checks,
                       SUM(n_gaps) AS total_gaps,
                       SUM(n_outliers) AS total_outliers
                FROM data_quality_reports
                GROUP BY symbol
                ORDER BY avg_score ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.warning("Failed to query worst symbols: %s", exc)
            return []
