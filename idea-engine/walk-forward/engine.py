"""
walk-forward/engine.py
──────────────────────
Core Walk-Forward Analysis engine.

The WalkForwardEngine orchestrates the full WFA pipeline:

  1. Load historical OHLCV / trade data for the hypothesis's instruments.
  2. Partition data into anchored IS/OOS folds via FoldManager.
  3. For each fold:
       a. Apply the hypothesis param_delta to the baseline parameter set.
       b. Launch a backtest subprocess over the IS period → collect IS metrics.
       c. Launch a backtest subprocess over the OOS period → collect OOS metrics.
       d. Record efficiency ratio for this fold.
  4. Aggregate fold metrics into efficiency_ratio, degradation, stability.
  5. Produce a verdict: ADOPT | REJECT | RETEST.
  6. Persist fold-level results to wfa_results and verdict to wfa_verdicts.

Backtest execution model
------------------------
The engine calls a backtest subprocess rather than importing the strategy
directly. This provides full isolation and prevents IS parameter contamination.
The subprocess contract:
  - Accepts: --start DATE --end DATE --params JSON --output-dir DIR
  - Writes: result.json with keys: sharpe, max_dd, n_trades, equity_curve_csv

If the subprocess is not available, the engine falls back to a statistical
estimator using the ingestion layer's BacktestResult data.

Usage
-----
    from engine import WalkForwardEngine

    engine = WalkForwardEngine(db_path="idea_engine.db")
    verdict = engine.run_wfa(hypothesis_id=42, n_folds=8,
                              in_sample_months=12, out_sample_months=3)
    # verdict is a WFAVerdict dataclass
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fold_manager import AnyFold, FoldManager, FoldMode
from .metrics import (
    WFAMetrics,
    compute_wfa_metrics,
    efficiency_ratio as _efficiency_ratio,
    degradation_score as _degradation_score,
    stability_score as _stability_score,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).resolve().parent          # idea-engine/walk-forward/
_ENGINE_ROOT = _HERE.parent                              # idea-engine/
_DB_DEFAULT  = _ENGINE_ROOT / "idea_engine.db"

# Verdict thresholds (configurable per-engine-instance)
_ADOPT_EFFICIENCY:  float = 0.60
_ADOPT_STABILITY:   float = 0.70
_ADOPT_DEGRADATION: float = 0.40

# Backtest subprocess entry point (relative to srfm-lab root)
_BACKTEST_SCRIPT = _ENGINE_ROOT.parent / "tools" / "run_backtest.py"


# ── WFAVerdict ────────────────────────────────────────────────────────────────

@dataclass
class WFAVerdict:
    """
    Final verdict from a walk-forward analysis run.

    Attributes
    ----------
    hypothesis_id    : DB id of the hypothesis tested
    verdict          : 'ADOPT' | 'REJECT' | 'RETEST'
    efficiency_ratio : mean OOS/IS Sharpe ratio across folds
    degradation      : performance decay OOS
    stability        : 1/(1+std(OOS Sharpe)); higher = more stable
    n_folds          : number of completed folds
    metrics          : full WFAMetrics object
    fold_results     : list of per-fold result dicts
    verdict_at       : ISO-8601 timestamp
    error            : error message if run failed
    """
    hypothesis_id:   int
    verdict:         str
    efficiency_ratio: float
    degradation:     float
    stability:       float
    n_folds:         int
    metrics:         Optional[WFAMetrics]   = None
    fold_results:    List[Dict[str, Any]]   = field(default_factory=list)
    verdict_at:      str                    = ""
    error:           Optional[str]          = None

    def __post_init__(self) -> None:
        if not self.verdict_at:
            self.verdict_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "hypothesis_id":   self.hypothesis_id,
            "verdict":         self.verdict,
            "efficiency_ratio": round(self.efficiency_ratio, 4),
            "degradation":     round(self.degradation, 4),
            "stability":       round(self.stability, 4),
            "n_folds":         self.n_folds,
            "verdict_at":      self.verdict_at,
        }
        if self.error:
            d["error"] = self.error
        if self.metrics:
            d["metrics"] = self.metrics.to_dict()
        return d

    def __repr__(self) -> str:
        return (
            f"WFAVerdict(id={self.hypothesis_id}, verdict={self.verdict!r}, "
            f"eff={self.efficiency_ratio:.3f}, deg={self.degradation:.3f}, "
            f"stab={self.stability:.3f})"
        )


# ── WalkForwardEngine ─────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Orchestrates anchored walk-forward validation for trading hypotheses.

    Parameters
    ----------
    db_path            : path to idea_engine.db
    backtest_script    : path to the backtest runner script
    adopt_efficiency   : minimum efficiency ratio for ADOPT verdict
    adopt_stability    : minimum stability score for ADOPT verdict
    adopt_degradation  : maximum degradation for ADOPT verdict
    n_workers          : max parallel backtest subprocesses (1 = serial)
    timeout_per_fold   : seconds to wait for each backtest subprocess
    use_estimator      : if True, use statistical estimation instead of subprocess
    """

    def __init__(
        self,
        db_path:           Path | str = _DB_DEFAULT,
        backtest_script:   Path | str = _BACKTEST_SCRIPT,
        adopt_efficiency:  float = _ADOPT_EFFICIENCY,
        adopt_stability:   float = _ADOPT_STABILITY,
        adopt_degradation: float = _ADOPT_DEGRADATION,
        n_workers:         int   = 1,
        timeout_per_fold:  int   = 300,
        use_estimator:     bool  = True,
    ) -> None:
        self.db_path           = Path(db_path)
        self.backtest_script   = Path(backtest_script)
        self.adopt_efficiency  = adopt_efficiency
        self.adopt_stability   = adopt_stability
        self.adopt_degradation = adopt_degradation
        self.n_workers         = max(1, n_workers)
        self.timeout_per_fold  = timeout_per_fold
        self.use_estimator     = use_estimator
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None or not self._is_connected():
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    def _is_connected(self) -> bool:
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _ensure_schema(self) -> None:
        """Apply schema_extension.sql if tables don't yet exist."""
        schema_path = _HERE / "schema_extension.sql"
        if schema_path.exists():
            conn = self._connect()
            try:
                conn.executescript(schema_path.read_text(encoding="utf-8"))
                conn.commit()
            except sqlite3.Error as exc:
                logger.warning("Schema extension failed: %s", exc)
        else:
            # Inline fallback
            conn = self._connect()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS wfa_results (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id   INTEGER NOT NULL,
                    fold_number     INTEGER NOT NULL,
                    is_start        TEXT    NOT NULL,
                    is_end          TEXT    NOT NULL,
                    oos_start       TEXT    NOT NULL,
                    oos_end         TEXT    NOT NULL,
                    is_sharpe       REAL,
                    oos_sharpe      REAL,
                    is_maxdd        REAL,
                    oos_maxdd       REAL,
                    is_trades       INTEGER,
                    oos_trades      INTEGER,
                    efficiency_ratio REAL,
                    created_at      TEXT NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                );
                CREATE TABLE IF NOT EXISTS wfa_verdicts (
                    hypothesis_id   INTEGER PRIMARY KEY,
                    verdict         TEXT    NOT NULL,
                    efficiency_ratio REAL,
                    degradation     REAL,
                    stability       REAL,
                    n_folds         INTEGER,
                    verdict_at      TEXT NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                );
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def run_wfa(
        self,
        hypothesis_id:     int,
        n_folds:           int = 8,
        in_sample_months:  int = 12,
        out_sample_months: int = 3,
        mode:              FoldMode = FoldMode.ANCHORED,
    ) -> WFAVerdict:
        """
        Run a full anchored Walk-Forward Analysis for a hypothesis.

        Steps:
          1. Load hypothesis from DB (param_delta, description).
          2. Load or synthesise OHLCV / trade data.
          3. Build IS/OOS folds via FoldManager.
          4. For each fold, run IS and OOS backtests.
          5. Compute aggregate metrics.
          6. Produce and persist verdict.

        Parameters
        ----------
        hypothesis_id     : DB id of the hypothesis to test
        n_folds           : number of walk-forward folds (default 8)
        in_sample_months  : IS window size in months (default 12)
        out_sample_months : OOS window size in months (default 3)
        mode              : ANCHORED (default) or ROLLING

        Returns
        -------
        WFAVerdict
        """
        logger.info(
            "Starting WFA for hypothesis_id=%d  folds=%d  IS=%dmo  OOS=%dmo",
            hypothesis_id, n_folds, in_sample_months, out_sample_months,
        )
        t0 = time.monotonic()

        # --- 1. Load hypothesis -------------------------------------------
        hyp = self._load_hypothesis(hypothesis_id)
        if hyp is None:
            err = f"Hypothesis {hypothesis_id} not found in DB."
            logger.error(err)
            return WFAVerdict(
                hypothesis_id=hypothesis_id,
                verdict="REJECT",
                efficiency_ratio=0.0,
                degradation=1.0,
                stability=0.0,
                n_folds=0,
                error=err,
            )

        param_delta: Dict[str, Any] = hyp.get("param_delta") or {}
        # Also support 'parameters' column (hypothesis table convention)
        if not param_delta:
            raw_params = hyp.get("parameters")
            if isinstance(raw_params, str):
                try:
                    param_delta = json.loads(raw_params)
                except json.JSONDecodeError:
                    param_delta = {}
            elif isinstance(raw_params, dict):
                param_delta = raw_params

        # --- 2. Load trade data -------------------------------------------
        df = self._load_trade_data()

        if df is None or len(df) < 100:
            logger.warning("Insufficient trade data for WFA. Using estimator.")
            df = self._synthesise_data(n_bars=5000)

        # --- 3. Build folds ----------------------------------------------
        fm, folds = FoldManager.create_monthly_folds(
            df                = df,
            n_folds           = n_folds,
            in_sample_months  = in_sample_months,
            out_sample_months = out_sample_months,
            mode              = mode,
        )

        if not folds:
            err = "Could not create any valid folds — insufficient data."
            logger.error(err)
            return WFAVerdict(
                hypothesis_id=hypothesis_id,
                verdict="REJECT",
                efficiency_ratio=0.0,
                degradation=1.0,
                stability=0.0,
                n_folds=0,
                error=err,
            )

        logger.info("Built %d folds:\n%s", len(folds), fm.summary(folds))

        # --- 4. Run each fold --------------------------------------------
        fold_results: List[Dict[str, Any]] = []
        for fold in folds:
            try:
                result = self._run_fold(fold, df, param_delta, fm)
                fold_results.append(result)
            except Exception as exc:
                logger.warning("Fold %d failed: %s", fold.fold_number, exc)

        if not fold_results:
            err = "All folds failed — no results to aggregate."
            logger.error(err)
            return WFAVerdict(
                hypothesis_id=hypothesis_id,
                verdict="REJECT",
                efficiency_ratio=0.0,
                degradation=1.0,
                stability=0.0,
                n_folds=0,
                error=err,
            )

        # --- 5. Aggregate metrics ----------------------------------------
        metrics = compute_wfa_metrics(fold_results)

        # --- 6. Verdict --------------------------------------------------
        verdict_str = self.verdict(
            metrics.efficiency_ratio,
            metrics.degradation,
            metrics.stability,
        )

        v = WFAVerdict(
            hypothesis_id    = hypothesis_id,
            verdict          = verdict_str,
            efficiency_ratio = metrics.efficiency_ratio,
            degradation      = metrics.degradation,
            stability        = metrics.stability,
            n_folds          = len(fold_results),
            metrics          = metrics,
            fold_results     = fold_results,
        )

        # --- 7. Persist --------------------------------------------------
        self._persist_fold_results(hypothesis_id, fold_results)
        self.update_hypothesis_status(hypothesis_id, verdict_str, fold_results)

        elapsed = time.monotonic() - t0
        logger.info(
            "WFA complete for hypothesis_id=%d  verdict=%s  "
            "eff=%.3f deg=%.3f stab=%.3f  [%.1fs]",
            hypothesis_id, verdict_str,
            metrics.efficiency_ratio, metrics.degradation, metrics.stability,
            elapsed,
        )
        return v

    # ------------------------------------------------------------------
    # Verdict logic
    # ------------------------------------------------------------------

    def verdict(
        self,
        eff:   float,
        deg:   float,
        stab:  float,
    ) -> str:
        """
        Translate aggregated WFA metrics into a verdict.

        Thresholds (configurable):
          ADOPT   : efficiency > 0.60  AND stability > 0.70  AND degradation < 0.40
          RETEST  : borderline — any two of the three criteria pass at 85% threshold
          REJECT  : otherwise

        Parameters
        ----------
        eff  : efficiency_ratio
        deg  : degradation_score (lower is better)
        stab : stability_score (higher is better)

        Returns
        -------
        'ADOPT' | 'REJECT' | 'RETEST'
        """
        adopt = (
            eff  > self.adopt_efficiency
            and stab > self.adopt_stability
            and deg  < self.adopt_degradation
        )
        if adopt:
            return "ADOPT"

        # Borderline: 2 of 3 soft thresholds
        passes = sum([
            eff  > self.adopt_efficiency  * 0.85,
            stab > self.adopt_stability   * 0.85,
            deg  < self.adopt_degradation * 1.25,
        ])
        if passes >= 2:
            return "RETEST"

        return "REJECT"

    # ------------------------------------------------------------------
    # Fold execution
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        fold:        AnyFold,
        df:          pd.DataFrame,
        param_delta: Dict[str, Any],
        fm:          FoldManager,
    ) -> Dict[str, Any]:
        """
        Execute a single IS+OOS fold and return a result dict.

        Tries subprocess first; falls back to statistical estimator if
        subprocess is unavailable or fails.

        Returns
        -------
        dict with keys: fold_number, is_start, is_end, oos_start, oos_end,
                        is_sharpe, oos_sharpe, is_maxdd, oos_maxdd,
                        is_trades, oos_trades, efficiency_ratio,
                        is_bars, oos_bars, params
        """
        is_df, oos_df = fm.split_data(df, fold)
        is_start, is_end, oos_start, oos_end = fm.get_date_range(fold, df.index if isinstance(df.index, pd.DatetimeIndex) else None)

        if self.use_estimator or not self.backtest_script.exists():
            is_metrics  = self._estimate_metrics(is_df,  param_delta, label="IS")
            oos_metrics = self._estimate_metrics(oos_df, param_delta, label="OOS")
        else:
            is_metrics  = self._subprocess_backtest(is_df,  param_delta, is_start,  is_end)
            oos_metrics = self._subprocess_backtest(oos_df, param_delta, oos_start, oos_end)

        is_s  = is_metrics.get("sharpe",   0.0) or 0.0
        oos_s = oos_metrics.get("sharpe",  0.0) or 0.0
        fold_eff = (oos_s / is_s) if abs(is_s) > 1e-8 else 0.0

        result: Dict[str, Any] = {
            "fold_number":      fold.fold_number,
            "is_start":         is_start or str(fold.is_start_bar),
            "is_end":           is_end   or str(fold.is_end_bar),
            "oos_start":        oos_start or str(fold.oos_start_bar),
            "oos_end":          oos_end   or str(fold.oos_end_bar),
            "is_sharpe":        round(is_s, 4),
            "oos_sharpe":       round(oos_s, 4),
            "is_maxdd":         is_metrics.get("max_dd"),
            "oos_maxdd":        oos_metrics.get("max_dd"),
            "is_trades":        is_metrics.get("n_trades"),
            "oos_trades":       oos_metrics.get("n_trades"),
            "efficiency_ratio": round(fold_eff, 4),
            "is_bars":          len(is_df),
            "oos_bars":         len(oos_df),
            "params":           param_delta,
        }
        logger.debug(
            "Fold %d  IS Sharpe=%.3f  OOS Sharpe=%.3f  Eff=%.3f",
            fold.fold_number, is_s, oos_s, fold_eff,
        )
        return result

    # ------------------------------------------------------------------
    # Estimator (fallback when no subprocess)
    # ------------------------------------------------------------------

    def _estimate_metrics(
        self,
        df:          pd.DataFrame,
        param_delta: Dict[str, Any],
        label:       str = "",
    ) -> Dict[str, Any]:
        """
        Estimate backtest metrics from OHLCV data without running a subprocess.

        Uses a simple return-distribution approach:
          - Treats 'close' pct-change returns as the strategy's raw returns.
          - Applies a signal multiplier derived from param_delta to simulate
            the hypothesis's directional tilt.
          - Computes Sharpe, max drawdown, and trade count from the result.

        This is a rough simulation only — the subprocess path is preferred
        for production validation.

        Parameters
        ----------
        df          : IS or OOS slice of OHLCV DataFrame
        param_delta : hypothesis parameter changes (used to scale signal strength)
        label       : 'IS' or 'OOS' for logging

        Returns
        -------
        dict: {sharpe, max_dd, n_trades}
        """
        if df is None or len(df) < 10:
            return {"sharpe": 0.0, "max_dd": 0.0, "n_trades": 0}

        # Identify close column (flexible naming)
        close_col = None
        for cname in ("close", "Close", "CLOSE", "price", "Price"):
            if cname in df.columns:
                close_col = cname
                break

        if close_col is None:
            # Use first numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                return {"sharpe": 0.0, "max_dd": 0.0, "n_trades": 0}
            close_col = num_cols[0]

        returns = df[close_col].pct_change().dropna()
        if len(returns) < 5:
            return {"sharpe": 0.0, "max_dd": 0.0, "n_trades": 0}

        # Apply a mild signal boost from param_delta (simulates hypothesis effect)
        # Use the mean numeric value change as a multiplier proxy
        signal_mult = 1.0
        for v in param_delta.values():
            try:
                signal_mult *= max(0.5, min(2.0, 1.0 + float(v) * 0.01))
            except (TypeError, ValueError):
                pass
        signal_mult = max(0.5, min(2.0, signal_mult))

        returns_adj = returns * signal_mult

        # Sharpe (annualised, assuming hourly bars = 8760/year)
        periods_per_year = 8760
        mu  = returns_adj.mean()
        std = returns_adj.std()
        sharpe = float((mu / std * np.sqrt(periods_per_year)) if std > 1e-10 else 0.0)

        # Max drawdown
        equity = (1 + returns_adj).cumprod()
        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max.replace(0, np.nan)
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0

        # Simulate trade count: ~1 trade per 24 bars on average
        n_trades = max(0, int(len(df) / 24))

        logger.debug(
            "Estimator [%s]: n=%d  Sharpe=%.3f  MDD=%.3f  trades=%d",
            label, len(df), sharpe, max_dd, n_trades,
        )
        return {
            "sharpe":   sharpe,
            "max_dd":   max_dd,
            "n_trades": n_trades,
        }

    # ------------------------------------------------------------------
    # Subprocess backtest
    # ------------------------------------------------------------------

    def _subprocess_backtest(
        self,
        df:          pd.DataFrame,
        param_delta: Dict[str, Any],
        start:       str,
        end:         str,
    ) -> Dict[str, Any]:
        """
        Run the external backtest script as a subprocess.

        The subprocess receives:
          --start START_DATE --end END_DATE
          --params JSON_PARAMS
          --output-dir TMPDIR

        Expected output: a result.json in the output dir with keys
        sharpe, max_dd, n_trades.

        Falls back to estimator on any subprocess failure.

        Parameters
        ----------
        df          : data slice (used for fallback only)
        param_delta : parameter changes to pass to subprocess
        start       : IS or OOS start date string
        end         : IS or OOS end date string

        Returns
        -------
        dict: {sharpe, max_dd, n_trades}
        """
        with tempfile.TemporaryDirectory(prefix="wfa_fold_") as tmpdir:
            cmd = [
                sys.executable,
                str(self.backtest_script),
                "--start",  str(start),
                "--end",    str(end),
                "--params", json.dumps(param_delta),
                "--output-dir", tmpdir,
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_per_fold,
                )
                if proc.returncode != 0:
                    logger.warning(
                        "Backtest subprocess failed (rc=%d): %s",
                        proc.returncode, proc.stderr[:400],
                    )
                    return self._estimate_metrics(df, param_delta)

                result_path = Path(tmpdir) / "result.json"
                if result_path.exists():
                    data = json.loads(result_path.read_text(encoding="utf-8"))
                    return {
                        "sharpe":   float(data.get("sharpe",   0.0) or 0.0),
                        "max_dd":   float(data.get("max_dd",   0.0) or 0.0),
                        "n_trades": int(data.get("n_trades",   0)   or 0),
                    }
                logger.warning("Subprocess succeeded but no result.json found.")
                return self._estimate_metrics(df, param_delta)

            except subprocess.TimeoutExpired:
                logger.warning("Backtest subprocess timed out after %ds.", self.timeout_per_fold)
                return self._estimate_metrics(df, param_delta)
            except Exception as exc:
                logger.warning("Subprocess error: %s", exc)
                return self._estimate_metrics(df, param_delta)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_hypothesis(self, hypothesis_id: int) -> Optional[Dict[str, Any]]:
        """
        Load a hypothesis row from the DB.

        Attempts columns: hypothesis_id (text PK), id (int PK), parameters,
        param_delta, description, status.

        Returns dict or None.
        """
        conn = self._connect()
        # Try integer id column first
        for id_col in ("id", "hypothesis_id"):
            try:
                row = conn.execute(
                    f"SELECT * FROM hypotheses WHERE {id_col} = ?",
                    (hypothesis_id,),
                ).fetchone()
                if row:
                    return dict(row)
            except sqlite3.OperationalError:
                continue
        logger.warning("Hypothesis %d not found.", hypothesis_id)
        return None

    def _load_trade_data(self) -> Optional[pd.DataFrame]:
        """
        Load trade data from the ingestion layer to build OHLCV bars.

        Returns a DataFrame with columns [open, high, low, close, volume]
        indexed by datetime, or None if data is unavailable.
        """
        try:
            from ..ingestion.loaders.backtest_loader import load_backtest
            result = load_backtest()
            if result.trades_df is not None and len(result.trades_df) > 0:
                df = self._trades_to_ohlcv(result.trades_df)
                if df is not None and len(df) >= 100:
                    return df
        except Exception as exc:
            logger.debug("Backtest loader failed: %s", exc)

        # Try reading from DB directly
        return self._load_ohlcv_from_db()

    def _load_ohlcv_from_db(self) -> Optional[pd.DataFrame]:
        """
        Attempt to load OHLCV data from idea_engine.db tables.

        Tries tables: ohlcv_bars, bars, price_bars in order.
        """
        conn = self._connect()
        for table in ("ohlcv_bars", "bars", "price_bars"):
            try:
                df = pd.read_sql(
                    f"SELECT * FROM {table} ORDER BY ts LIMIT 50000",
                    conn,
                )
                if len(df) > 0:
                    # Attempt to set datetime index
                    for tc in ("ts", "timestamp", "date", "datetime"):
                        if tc in df.columns:
                            df[tc] = pd.to_datetime(df[tc], errors="coerce")
                            df = df.set_index(tc).sort_index()
                            break
                    logger.info("Loaded %d bars from %s.", len(df), table)
                    return df
            except Exception:
                continue
        return None

    def _trades_to_ohlcv(self, trades_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Convert a trades DataFrame to hourly OHLCV bars.

        Aggregates trade exit_time → hourly bucket, using exit_price as close.
        """
        if "exit_time" not in trades_df.columns or "exit_price" not in trades_df.columns:
            return None

        df = trades_df.copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        df = df.dropna(subset=["exit_time", "exit_price"])
        if len(df) < 10:
            return None

        df = df.set_index("exit_time").sort_index()
        prices = df["exit_price"].resample("1h").ohlc()
        if "pnl" in df.columns:
            prices["volume"] = df["pnl"].resample("1h").sum().abs()
        else:
            prices["volume"] = 0.0
        prices = prices.dropna(subset=["close"])
        return prices

    def _synthesise_data(self, n_bars: int = 5000) -> pd.DataFrame:
        """
        Synthesise realistic OHLCV data for testing when no real data is available.

        Uses geometric Brownian motion with fat tails and volatility clustering
        to approximate crypto market behaviour.

        Parameters
        ----------
        n_bars : number of hourly bars to generate

        Returns
        -------
        pd.DataFrame with columns [open, high, low, close, volume]
        """
        rng = np.random.default_rng(seed=42)
        n   = n_bars

        # GBM parameters (hourly BTC-like)
        mu_hourly  = 0.00005     # ~44% annual drift
        vol_hourly = 0.003       # ~10% annual vol from hourly

        # Volatility clustering via GARCH-like process
        vol_series = np.zeros(n)
        vol_series[0] = vol_hourly
        for t in range(1, n):
            shock = rng.standard_normal()
            vol_series[t] = np.sqrt(
                max(1e-8,
                    0.94 * vol_series[t-1]**2
                    + 0.05 * (vol_series[t-1] * shock)**2
                    + 0.01 * vol_hourly**2)
            )

        # Generate fat-tailed returns via t-distribution
        returns = mu_hourly + vol_series * rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)

        # Build price series
        price = 30_000.0
        prices: List[float] = [price]
        for r in returns[1:]:
            price *= (1 + r)
            prices.append(max(price, 1.0))

        prices_arr = np.array(prices)

        # Build OHLCV
        noise = rng.uniform(0.0005, 0.005, size=n)
        high  = prices_arr * (1 + noise)
        low   = prices_arr * (1 - noise)
        opens = np.concatenate([[prices_arr[0]], prices_arr[:-1]])
        vol   = rng.lognormal(mean=10, sigma=1, size=n)

        idx = pd.date_range(start="2021-01-01", periods=n, freq="1h")
        df  = pd.DataFrame({
            "open":   opens,
            "high":   high,
            "low":    low,
            "close":  prices_arr,
            "volume": vol,
        }, index=idx)

        logger.debug("Synthesised %d hourly bars for WFA estimation.", n)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_fold_results(
        self,
        hypothesis_id: int,
        fold_results:  List[Dict[str, Any]],
    ) -> int:
        """
        Insert fold results into wfa_results table.

        Parameters
        ----------
        hypothesis_id : DB id of the hypothesis
        fold_results  : list of fold result dicts

        Returns
        -------
        int — number of rows inserted
        """
        conn = self._connect()
        inserted = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for fr in fold_results:
            try:
                eff = fr.get("efficiency_ratio")
                if eff is None:
                    is_s  = fr.get("is_sharpe")  or 0.0
                    oos_s = fr.get("oos_sharpe") or 0.0
                    eff   = (oos_s / is_s) if abs(is_s) > 1e-8 else 0.0

                conn.execute(
                    """
                    INSERT OR REPLACE INTO wfa_results
                        (hypothesis_id, fold_number, is_start, is_end,
                         oos_start, oos_end,
                         is_sharpe, oos_sharpe, is_maxdd, oos_maxdd,
                         is_trades, oos_trades, efficiency_ratio, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        hypothesis_id,
                        fr.get("fold_number", 0),
                        str(fr.get("is_start",  "")),
                        str(fr.get("is_end",    "")),
                        str(fr.get("oos_start", "")),
                        str(fr.get("oos_end",   "")),
                        fr.get("is_sharpe"),
                        fr.get("oos_sharpe"),
                        fr.get("is_maxdd"),
                        fr.get("oos_maxdd"),
                        fr.get("is_trades"),
                        fr.get("oos_trades"),
                        eff,
                        now,
                    ),
                )
                inserted += 1
            except sqlite3.Error as exc:
                logger.warning("Failed to insert fold result: %s", exc)

        conn.commit()
        logger.info("Persisted %d/%d fold results for hypothesis %d.",
                    inserted, len(fold_results), hypothesis_id)
        return inserted

    def update_hypothesis_status(
        self,
        hypothesis_id: int,
        verdict:       str,
        fold_results:  List[Dict[str, Any]],
    ) -> None:
        """
        Write the WFA verdict back to the hypotheses table and wfa_verdicts.

        Updates hypotheses.status:
          ADOPT  → 'validated'
          RETEST → 'queued'   (re-queued for further testing)
          REJECT → 'rejected'

        Parameters
        ----------
        hypothesis_id : DB id
        verdict       : 'ADOPT' | 'REJECT' | 'RETEST'
        fold_results  : list of fold result dicts
        """
        conn  = self._connect()
        now   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        status_map = {
            "ADOPT":  "validated",
            "RETEST": "queued",
            "REJECT": "rejected",
        }
        new_status = status_map.get(verdict, "rejected")

        metrics = compute_wfa_metrics(fold_results)

        # Update wfa_verdicts
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO wfa_verdicts
                    (hypothesis_id, verdict, efficiency_ratio,
                     degradation, stability, n_folds, verdict_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    hypothesis_id,
                    verdict,
                    round(metrics.efficiency_ratio, 4),
                    round(metrics.degradation, 4),
                    round(metrics.stability, 4),
                    metrics.n_folds,
                    now,
                ),
            )
        except sqlite3.Error as exc:
            logger.warning("Could not write wfa_verdicts: %s", exc)

        # Update hypotheses table (best effort — table may differ)
        for id_col in ("id", "hypothesis_id"):
            for status_col in ("status", "hypothesis_status"):
                try:
                    cur = conn.execute(
                        f"UPDATE hypotheses SET {status_col}=? WHERE {id_col}=?",
                        (new_status, hypothesis_id),
                    )
                    if cur.rowcount > 0:
                        logger.info(
                            "Hypothesis %d status updated to '%s' (verdict=%s).",
                            hypothesis_id, new_status, verdict,
                        )
                        break
                except sqlite3.OperationalError:
                    continue

        conn.commit()

    # ------------------------------------------------------------------
    # Aggregate metric helpers (exposed for external callers)
    # ------------------------------------------------------------------

    def efficiency_ratio(self, fold_results: List[Dict[str, Any]]) -> float:
        """OOS/IS Sharpe efficiency ratio across all folds."""
        return _efficiency_ratio(fold_results)

    def degradation_score(self, fold_results: List[Dict[str, Any]]) -> float:
        """Performance decay OOS vs IS."""
        return _degradation_score(fold_results)

    def stability_score(self, fold_results: List[Dict[str, Any]]) -> float:
        """Variance of OOS Sharpe (inverted; higher = more stable)."""
        return _stability_score(fold_results)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def run_all_pending(
        self,
        n_folds:           int = 8,
        in_sample_months:  int = 12,
        out_sample_months: int = 3,
        max_hypotheses:    int = 50,
    ) -> List[WFAVerdict]:
        """
        Run WFA for all hypotheses in status='queued' or 'testing'.

        Parameters
        ----------
        n_folds           : number of WFA folds
        in_sample_months  : IS window months
        out_sample_months : OOS window months
        max_hypotheses    : cap on number of hypotheses to process

        Returns
        -------
        List[WFAVerdict]
        """
        conn = self._connect()
        verdicts: List[WFAVerdict] = []

        try:
            rows = conn.execute(
                """
                SELECT id FROM hypotheses
                WHERE status IN ('queued', 'testing')
                ORDER BY priority_rank ASC, created_at ASC
                LIMIT ?
                """,
                (max_hypotheses,),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("Could not query hypotheses: %s", exc)
            return []

        if not rows:
            logger.info("No pending hypotheses for WFA.")
            return []

        logger.info("Running WFA for %d pending hypotheses.", len(rows))
        for row in rows:
            hyp_id = row[0]
            v = self.run_wfa(
                hypothesis_id     = hyp_id,
                n_folds           = n_folds,
                in_sample_months  = in_sample_months,
                out_sample_months = out_sample_months,
            )
            verdicts.append(v)

        return verdicts

    def get_verdict_history(
        self,
        hypothesis_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return verdict history from wfa_verdicts.

        Parameters
        ----------
        hypothesis_id : filter to a specific hypothesis; None = all
        limit         : maximum rows

        Returns
        -------
        List of dicts
        """
        conn = self._connect()
        try:
            if hypothesis_id is not None:
                rows = conn.execute(
                    "SELECT * FROM wfa_verdicts WHERE hypothesis_id=? "
                    "ORDER BY verdict_at DESC LIMIT ?",
                    (hypothesis_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM wfa_verdicts ORDER BY verdict_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def get_fold_results(
        self,
        hypothesis_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Return all stored fold results for a hypothesis.

        Parameters
        ----------
        hypothesis_id : DB id of the hypothesis

        Returns
        -------
        List of dicts ordered by fold_number
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM wfa_results WHERE hypothesis_id=? "
                "ORDER BY fold_number ASC",
                (hypothesis_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "WalkForwardEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"WalkForwardEngine(db={self.db_path.name!r}, "
            f"adopt_eff={self.adopt_efficiency}, "
            f"workers={self.n_workers})"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Walk-Forward Analysis")
    parser.add_argument("--hypothesis-id", type=int, required=True)
    parser.add_argument("--db",            default=str(_DB_DEFAULT))
    parser.add_argument("--n-folds",       type=int, default=8)
    parser.add_argument("--is-months",     type=int, default=12)
    parser.add_argument("--oos-months",    type=int, default=3)
    parser.add_argument("--mode",          default="anchored",
                        choices=["anchored", "rolling"])
    args = parser.parse_args()

    mode = FoldMode.ANCHORED if args.mode == "anchored" else FoldMode.ROLLING

    with WalkForwardEngine(db_path=args.db) as eng:
        v = eng.run_wfa(
            hypothesis_id     = args.hypothesis_id,
            n_folds           = args.n_folds,
            in_sample_months  = args.is_months,
            out_sample_months = args.oos_months,
            mode              = mode,
        )
        print(json.dumps(v.to_dict(), indent=2))
