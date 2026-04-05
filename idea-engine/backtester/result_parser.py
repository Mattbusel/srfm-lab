"""
ResultParser
============
Parses backtest output (stdout and CSV trade logs) into structured metrics.

Designed to handle the output format of tools/crypto_backtest_mc.py.  Falls
back gracefully when fields are missing or the file is absent.
"""

from __future__ import annotations

import calendar
import io
import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex patterns for parsing key=value lines in stdout
_METRIC_PATTERNS: dict[str, re.Pattern] = {
    "sharpe":        re.compile(r"(?:sharpe|sharpe_ratio)\s*[=:]\s*([-\d.]+)", re.I),
    "calmar":        re.compile(r"(?:calmar|calmar_ratio)\s*[=:]\s*([-\d.]+)", re.I),
    "max_dd":        re.compile(r"(?:max_dd|max_drawdown|maxdd)\s*[=:]\s*([-\d.]+)", re.I),
    "total_return":  re.compile(r"(?:total_return|total_ret|cagr)\s*[=:]\s*([-\d.]+)", re.I),
    "win_rate":      re.compile(r"(?:win_rate|winrate|win_pct)\s*[=:]\s*([-\d.]+)", re.I),
    "num_trades":    re.compile(r"(?:num_trades|n_trades|total_trades)\s*[=:]\s*(\d+)", re.I),
    "profit_factor": re.compile(r"(?:profit_factor|pf)\s*[=:]\s*([-\d.]+)", re.I),
}

# Annualisation constant (15-minute bars, 24/7 crypto)
BARS_PER_YEAR_15M = 252 * 24 * 4   # 35,064
BARS_PER_YEAR_1H  = 252 * 24       # 6,048
BARS_PER_YEAR_1D  = 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """
    Comprehensive metrics extracted from a completed backtest.
    """

    # Core performance
    sharpe:         float = 0.0
    calmar:         float = 0.0
    max_dd:         float = 0.0    # expressed as positive fraction, e.g. 0.15 = 15 %
    cagr:           float = 0.0
    total_return:   float = 0.0
    win_rate:       float = 0.0
    profit_factor:  float = 0.0
    num_trades:     int   = 0

    # Trade-level statistics
    avg_trade_duration_bars: float = 0.0
    avg_win:        float = 0.0
    avg_loss:       float = 0.0
    avg_win_loss_ratio: float = 0.0

    # Drawdown detail
    avg_dd:         float = 0.0
    recovery_bars:  float = 0.0    # average bars to recover from a drawdown

    # Volatility
    annualised_vol: float = 0.0

    # Monthly breakdown: dict {"YYYY-MM": return_pct}
    monthly_returns: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        """Return True if the metrics contain at least one non-zero value."""
        return any([
            self.sharpe != 0.0,
            self.num_trades > 0,
            self.total_return != 0.0,
        ])


@dataclass
class BacktestResult:
    """
    Full result from one backtest run.
    """

    params_hash:      str
    params:           dict[str, Any]
    metrics:          BacktestMetrics
    label:            str | None       = None
    equity_curve:     pd.Series | None = field(default=None, repr=False)
    trades_df:        pd.DataFrame | None = field(default=None, repr=False)
    stdout:           str              = field(default="", repr=False)
    stderr:           str              = field(default="", repr=False)
    duration_seconds: float            = 0.0
    status:           str              = "completed"   # "completed" | "error" | "timeout"
    error_message:    str | None       = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "params_hash": self.params_hash,
            "label": self.label,
            "metrics": self.metrics.to_dict(),
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error_message": self.error_message,
        }

    @property
    def is_success(self) -> bool:
        return self.status == "completed" and self.error_message is None


# ---------------------------------------------------------------------------
# ResultParser
# ---------------------------------------------------------------------------

class ResultParser:
    """
    Parses the output of ``tools.crypto_backtest_mc`` into structured data.

    The backtest tool may emit metrics in several ways:
      1. JSON blob on stdout (preferred — one line of JSON).
      2. Key=value lines in stdout (fallback).
      3. A CSV trade log written to disk.
    """

    def __init__(self, bars_per_year: int = BARS_PER_YEAR_15M) -> None:
        self.bars_per_year = bars_per_year

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def parse_stdout(self, stdout: str) -> dict[str, Any]:
        """
        Extract performance metrics from backtest stdout.

        Tries JSON first, then falls back to regex key=value scanning.

        Returns
        -------
        dict with keys matching :class:`BacktestMetrics` fields.
        """
        stdout = stdout.strip()
        if not stdout:
            return {}

        # --- Attempt 1: JSON block anywhere in stdout ---
        json_result = self._try_parse_json(stdout)
        if json_result:
            logger.debug("Parsed backtest stdout as JSON (%d keys).", len(json_result))
            return json_result

        # --- Attempt 2: Key=value regex scan ---
        kv_result = self._scan_kv(stdout)
        if kv_result:
            logger.debug("Parsed backtest stdout via regex (%d keys).", len(kv_result))
            return kv_result

        logger.warning("Could not parse any metrics from backtest stdout.")
        return {}

    def parse_trades_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """
        Load a backtest trade log CSV from *csv_path*.

        Expected columns (subset): entry_time, exit_time, symbol, side,
        entry_price, exit_price, pnl, pnl_pct, hold_bars.

        Returns an empty DataFrame if the file is missing or malformed.
        """
        path = Path(csv_path)
        if not path.exists():
            logger.warning("Trades CSV not found: %s", path)
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            # Try to parse time columns
            for col in ("entry_time", "exit_time", "timestamp", "ts"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            logger.info("Loaded %d trade rows from %s.", len(df), path)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse trades CSV %s: %s", path, exc)
            return pd.DataFrame()

    def parse_equity_curve(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Reconstruct an equity curve (cumulative PnL) from a trades DataFrame.

        If ``pnl_pct`` is present, equity = (1 + pnl_pct).cumprod().
        If ``pnl`` is present, equity = pnl.cumsum() (dollar).
        Otherwise return an empty Series.
        """
        if trades_df.empty:
            return pd.Series(dtype=float)

        if "pnl_pct" in trades_df.columns:
            pnl = trades_df["pnl_pct"].fillna(0.0)
            equity = (1.0 + pnl).cumprod()
        elif "pnl" in trades_df.columns:
            equity = trades_df["pnl"].fillna(0.0).cumsum()
        else:
            logger.debug("No pnl/pnl_pct column found; returning empty equity curve.")
            return pd.Series(dtype=float)

        # Set index to exit_time if available
        if "exit_time" in trades_df.columns:
            equity.index = trades_df["exit_time"]

        return equity

    def extract_metrics(
        self,
        trades_df: pd.DataFrame,
        equity_curve: pd.Series,
    ) -> BacktestMetrics:
        """
        Compute a full :class:`BacktestMetrics` from trades and equity curve.

        Works even if some columns are missing, falling back to zero for those
        fields.
        """
        metrics = BacktestMetrics()

        if trades_df.empty and (equity_curve is None or equity_curve.empty):
            return metrics

        # ---- Trade-level stats ----
        if not trades_df.empty:
            metrics.num_trades = len(trades_df)

            if "pnl_pct" in trades_df.columns:
                pnl = trades_df["pnl_pct"].dropna()
            elif "pnl" in trades_df.columns:
                pnl = trades_df["pnl"].dropna()
            else:
                pnl = pd.Series(dtype=float)

            if not pnl.empty:
                wins = pnl[pnl > 0]
                losses = pnl[pnl < 0]
                metrics.win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
                metrics.avg_win = float(wins.mean()) if not wins.empty else 0.0
                metrics.avg_loss = float(losses.mean()) if not losses.empty else 0.0
                if abs(metrics.avg_loss) > 1e-12:
                    metrics.avg_win_loss_ratio = abs(metrics.avg_win / metrics.avg_loss)
                gross_profit = float(wins.sum()) if not wins.empty else 0.0
                gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
                metrics.profit_factor = (
                    gross_profit / gross_loss if gross_loss > 1e-12 else float("inf")
                )

            if "hold_bars" in trades_df.columns:
                metrics.avg_trade_duration_bars = float(
                    trades_df["hold_bars"].dropna().mean()
                )

        # ---- Equity-curve based stats ----
        if equity_curve is not None and not equity_curve.empty:
            eq = equity_curve.dropna()

            if len(eq) >= 2:
                returns = eq.pct_change().dropna()

                # Total return
                metrics.total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

                # Annualised return (CAGR)
                n_bars = len(eq)
                years = n_bars / self.bars_per_year
                if years > 0 and eq.iloc[0] > 0:
                    metrics.cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)

                # Annualised volatility
                if len(returns) > 1:
                    metrics.annualised_vol = float(
                        returns.std() * math.sqrt(self.bars_per_year)
                    )

                # Sharpe (annualised, risk-free = 0)
                if metrics.annualised_vol > 1e-12:
                    metrics.sharpe = metrics.cagr / metrics.annualised_vol

                # Max drawdown
                roll_max = eq.cummax()
                dd_series = (eq - roll_max) / roll_max.replace(0, float("nan"))
                metrics.max_dd = float(abs(dd_series.min())) if not dd_series.empty else 0.0
                metrics.avg_dd = float(abs(dd_series[dd_series < 0].mean())) if (dd_series < 0).any() else 0.0

                # Calmar
                if metrics.max_dd > 1e-12:
                    metrics.calmar = metrics.cagr / metrics.max_dd

                # Recovery: average bars between new equity high
                metrics.recovery_bars = self._avg_recovery_bars(eq)

                # Monthly returns
                metrics.monthly_returns = self._monthly_returns(eq)

        return metrics

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def parse_all(
        self,
        stdout: str,
        trades_csv_path: str | Path | None = None,
        params: dict[str, Any] | None = None,
        params_hash: str = "",
        label: str | None = None,
        duration_seconds: float = 0.0,
    ) -> BacktestResult:
        """
        Parse everything available and return a :class:`BacktestResult`.

        Parameters
        ----------
        stdout          : captured stdout from the backtest subprocess.
        trades_csv_path : optional path to the trades CSV file.
        params          : original parameter dict.
        params_hash     : pre-computed hash (use ParamManager.hash_params).
        label           : optional human-readable label.
        duration_seconds: wall-clock time of the run.
        """
        # Parse stdout metrics
        stdout_metrics = self.parse_stdout(stdout)

        # Parse trade CSV
        trades_df = pd.DataFrame()
        equity_curve = pd.Series(dtype=float)
        if trades_csv_path:
            trades_df = self.parse_trades_csv(trades_csv_path)
            if not trades_df.empty:
                equity_curve = self.parse_equity_curve(trades_df)

        # Compute metrics from trades if possible; merge with stdout metrics
        if not trades_df.empty:
            metrics = self.extract_metrics(trades_df, equity_curve)
            # Overwrite with stdout metrics where they disagree (stdout is authoritative)
            metrics = self._merge_stdout_into_metrics(metrics, stdout_metrics)
        else:
            metrics = self._dict_to_metrics(stdout_metrics)

        return BacktestResult(
            params_hash=params_hash,
            params=params or {},
            metrics=metrics,
            label=label,
            equity_curve=equity_curve if not equity_curve.empty else None,
            trades_df=trades_df if not trades_df.empty else None,
            stdout=stdout,
            duration_seconds=duration_seconds,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """Try to extract a JSON object from *text*."""
        # Look for the first { ... } block
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass

        # Try to find JSON anywhere using a broader search
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _scan_kv(text: str) -> dict[str, Any]:
        """Scan *text* for key=value or key: value metric lines."""
        result: dict[str, Any] = {}
        for key, pattern in _METRIC_PATTERNS.items():
            m = pattern.search(text)
            if m:
                try:
                    result[key] = float(m.group(1))
                except ValueError:
                    pass
        return result

    @staticmethod
    def _dict_to_metrics(d: dict[str, Any]) -> BacktestMetrics:
        """Convert a flat dict of metric values to BacktestMetrics."""
        m = BacktestMetrics()
        if not d:
            return m
        m.sharpe       = float(d.get("sharpe", 0.0))
        m.calmar       = float(d.get("calmar", 0.0))
        m.max_dd       = abs(float(d.get("max_dd", 0.0)))
        m.total_return = float(d.get("total_return", 0.0))
        m.cagr         = float(d.get("cagr", d.get("total_return", 0.0)))
        m.win_rate     = float(d.get("win_rate", 0.0))
        m.num_trades   = int(d.get("num_trades", 0))
        m.profit_factor = float(d.get("profit_factor", 0.0))
        return m

    @staticmethod
    def _merge_stdout_into_metrics(
        metrics: BacktestMetrics,
        stdout_dict: dict[str, Any],
    ) -> BacktestMetrics:
        """Overwrite *metrics* fields with non-zero values from *stdout_dict*."""
        field_map = {
            "sharpe":        "sharpe",
            "calmar":        "calmar",
            "max_dd":        "max_dd",
            "total_return":  "total_return",
            "win_rate":      "win_rate",
            "num_trades":    "num_trades",
            "profit_factor": "profit_factor",
        }
        for src_key, dst_key in field_map.items():
            if src_key in stdout_dict:
                val = stdout_dict[src_key]
                if val is not None and val != 0:
                    setattr(metrics, dst_key, val)
        return metrics

    @staticmethod
    def _avg_recovery_bars(equity: pd.Series) -> float:
        """Compute average number of bars to recover from each drawdown trough."""
        if len(equity) < 3:
            return 0.0

        eq = equity.values
        roll_max = np.maximum.accumulate(eq)
        in_dd = eq < roll_max

        if not in_dd.any():
            return 0.0

        recovery_lengths: list[int] = []
        i = 0
        while i < len(in_dd):
            if in_dd[i]:
                # Find trough
                j = i
                while j < len(in_dd) and in_dd[j]:
                    j += 1
                recovery_lengths.append(j - i)
                i = j
            else:
                i += 1

        return float(np.mean(recovery_lengths)) if recovery_lengths else 0.0

    @staticmethod
    def _monthly_returns(equity: pd.Series) -> dict[str, float]:
        """
        Compute monthly return from equity curve.

        Requires a DatetimeIndex.  Returns dict {\"YYYY-MM\": return_pct}.
        """
        if not isinstance(equity.index, pd.DatetimeIndex):
            return {}
        if len(equity) < 2:
            return {}

        monthly: dict[str, float] = {}
        try:
            resampled = equity.resample("ME").last()
            prev = resampled.shift(1)
            m_rets = (resampled / prev - 1.0).dropna()
            for ts, val in m_rets.items():
                key = ts.strftime("%Y-%m")
                monthly[key] = round(float(val), 6)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Monthly return calculation failed: %s", exc)

        return monthly
