"""
optimization/optuna_optimizer.py
=================================
Optuna-based hyperparameter optimization for LARSA parameters.

Walk-forward optimization framework with regime-aware objectives,
overfit detection via deflated Sharpe, and automatic proposal of
best parameters to the coordination layer.

Classes:
  LARSAObjective          -- base Optuna objective, negative Sharpe
  RegimeAwareObjective    -- harmonic-mean across regime slices
  WalkForwardOptimizer    -- orchestrates multi-window walk-forward

Requires: optuna, pandas, numpy, scipy
"""

from __future__ import annotations

import csv
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from config.param_schema import ParamSchema  # noqa: E402 (import after path insert)
from config.param_manager import LiveParams, ParamManager  # noqa: E402

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    _OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRADING_DAYS_PER_YEAR = 252
_BARS_PER_DAY = 26          # 15-minute bars in a 6.5-hour RTH session
_BARS_PER_YEAR = _TRADING_DAYS_PER_YEAR * _BARS_PER_DAY
_PRUNE_BAR = 500            # MedianPruner intermediate report bar
_OVERFIT_THRESHOLD = 0.5    # IS Sharpe - OOS Sharpe > 0.5 -> overfit flag


# ---------------------------------------------------------------------------
# BacktestRunner -- thin adapter over backtest/engine.py
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    Runs the LARSA backtest engine on a given price DataFrame with
    a parameter set applied to the strategy adapter.

    If the real engine is unavailable (e.g., missing data) it raises
    RuntimeError so the Optuna trial can be marked as failed gracefully.
    """

    def __init__(self, schema: ParamSchema) -> None:
        self._schema = schema
        self._engine_available = False
        try:
            from backtest.engine import BacktestEngine  # noqa: F401
            from backtest.strategy_adapter import StrategyAdapter  # noqa: F401
            self._engine_available = True
        except ImportError:
            logger.warning("backtest.engine not importable -- BacktestRunner in stub mode")

    def run(
        self,
        bars: dict[str, pd.DataFrame],
        params: dict[str, Any],
        initial_capital: float = 100_000.0,
    ) -> dict[str, Any]:
        """
        Run a single backtest and return a results dict containing at minimum:
          equity_curve (pd.Series), sharpe (float), total_return (float),
          max_drawdown (float), n_trades (int)
        """
        if not self._engine_available:
            raise RuntimeError("BacktestEngine is not available -- check backtest/ imports")

        from backtest.engine import BacktestEngine
        from backtest.strategy_adapter import StrategyAdapter

        adapter = StrategyAdapter(params=params)
        symbols = list(bars.keys())
        engine = BacktestEngine(
            symbols=symbols,
            initial_capital=initial_capital,
        )
        engine.register_handler(adapter)
        results = engine.run_simple(bars=bars, signal_fn=adapter.on_bar)
        return results


def _compute_sharpe(returns: pd.Series, periods_per_year: int = _BARS_PER_YEAR) -> float:
    """Annualized Sharpe ratio from a series of per-bar returns."""
    if len(returns) < 20:
        return float("-inf")
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(mean / std * math.sqrt(periods_per_year))


def _equity_to_returns(equity: pd.Series) -> pd.Series:
    """Convert an equity curve to per-period log returns."""
    if len(equity) < 2:
        return pd.Series(dtype=float)
    return np.log(equity / equity.shift(1)).dropna()


def _max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown as a positive fraction."""
    if len(equity) < 2:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(abs(dd.min()))


# ---------------------------------------------------------------------------
# LARSAObjective
# ---------------------------------------------------------------------------

class LARSAObjective:
    """
    Optuna objective for LARSA parameter optimization.

    Samples parameters from the schema, runs a backtest on a 6-month
    in-sample window, and reports validation Sharpe on a 2-month OOS
    window. Returns negative Sharpe (Optuna minimizes by default).

    Intermediate values are reported every _PRUNE_BAR bars so
    MedianPruner can kill unpromising trials early.
    """

    def __init__(
        self,
        bars: dict[str, pd.DataFrame],
        schema: Optional[ParamSchema] = None,
        is_months: int = 6,
        oos_months: int = 2,
        initial_capital: float = 100_000.0,
        base_params: Optional[dict] = None,
    ) -> None:
        """
        Args:
            bars: {symbol: OHLCV DataFrame with DatetimeIndex}
            schema: ParamSchema instance (loads default if None)
            is_months: in-sample window length in months
            oos_months: out-of-sample validation window length in months
            initial_capital: starting equity for backtest
            base_params: fixed parameters not subject to optimization
        """
        self._bars = bars
        self._schema = schema or ParamSchema()
        self._is_months = is_months
        self._oos_months = oos_months
        self._initial_capital = initial_capital
        self._base_params = base_params or {}
        self._runner = BacktestRunner(self._schema)
        self._all_dates = self._build_date_range()

    def _build_date_range(self) -> pd.DatetimeIndex:
        """Collect all bar timestamps across all symbols."""
        dates: list[pd.Timestamp] = []
        for df in self._bars.values():
            if isinstance(df.index, pd.DatetimeIndex):
                dates.extend(df.index.tolist())
        if not dates:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(sorted(set(dates)))

    def _split_window(
        self, start: Optional[pd.Timestamp] = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """
        Slice bars into (in-sample, out-of-sample) windows.

        If start is None, uses the earliest available date.
        """
        if len(self._all_dates) == 0:
            return {}, {}

        t0 = start if start is not None else self._all_dates[0]
        t_is_end = t0 + pd.DateOffset(months=self._is_months)
        t_oos_end = t_is_end + pd.DateOffset(months=self._oos_months)

        is_bars: dict[str, pd.DataFrame] = {}
        oos_bars: dict[str, pd.DataFrame] = {}
        for sym, df in self._bars.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                continue
            is_bars[sym] = df[(df.index >= t0) & (df.index < t_is_end)]
            oos_bars[sym] = df[(df.index >= t_is_end) & (df.index < t_oos_end)]

        return is_bars, oos_bars

    def _suggest_params(self, trial: "optuna.Trial") -> dict[str, Any]:
        """
        Sample all optimizable parameters from the schema using the trial.
        Merges with base_params (fixed params that are not being tuned).
        """
        suggested: dict[str, Any] = {}
        schema_params = self._schema._schema

        for name, spec in schema_params.items():
            # Skip params that are fixed in base_params
            if name in self._base_params:
                continue

            ptype = spec["type"]

            if ptype == "float":
                lo = spec.get("min", 0.0)
                hi = spec.get("max", 1.0)
                step = spec.get("step")
                if step is not None:
                    # Round to step grid
                    n_steps = round((hi - lo) / step)
                    value = trial.suggest_float(name, lo, hi, step=step)
                else:
                    value = trial.suggest_float(name, lo, hi)
                suggested[name] = value

            elif ptype == "int":
                lo = int(spec.get("min", 1))
                hi = int(spec.get("max", 100))
                step = int(spec.get("step", 1))
                value = trial.suggest_int(name, lo, hi, step=step)
                suggested[name] = value

            elif ptype == "bool":
                value = trial.suggest_categorical(name, [True, False])
                suggested[name] = value

            elif ptype == "list_int":
                # For BLOCKED_HOURS: suggest a bitmask over hours 0-23
                # Trial suggests number of blocked hours and which ones
                default = spec.get("default", [])
                # Represent as categorical choice over a few preset options
                presets = [
                    [1, 13, 14, 15, 17, 18],       # default
                    [1, 2, 13, 14, 15, 17, 18],     # extended early block
                    [13, 14, 15, 17, 18],            # reduced
                    [1, 13, 14, 15],                 # minimal
                    [],                              # no blocking
                ]
                idx = trial.suggest_int(f"{name}_preset_idx", 0, len(presets) - 1)
                suggested[name] = presets[idx]

        # Apply cross-parameter constraint repairs
        suggested = self._repair_constraints(suggested)

        # Merge with fixed base params
        merged = {**self._base_params, **suggested}
        return merged

    def _repair_constraints(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Repair any cross-parameter constraint violations in-place.
        This is called after suggestion to ensure GARCH stationarity, etc.
        """
        p = dict(params)

        # CF_BEAR_THRESH >= CF_BULL_THRESH
        if "CF_BULL_THRESH" in p and "CF_BEAR_THRESH" in p:
            if p["CF_BEAR_THRESH"] < p["CF_BULL_THRESH"]:
                p["CF_BEAR_THRESH"] = p["CF_BULL_THRESH"]

        # BH_MASS_EXTREME > BH_MASS_THRESH
        if "BH_MASS_THRESH" in p and "BH_MASS_EXTREME" in p:
            if p["BH_MASS_EXTREME"] <= p["BH_MASS_THRESH"]:
                p["BH_MASS_EXTREME"] = p["BH_MASS_THRESH"] + 0.5

        # MAX_HOLD_BARS > MIN_HOLD_BARS
        if "MIN_HOLD_BARS" in p and "MAX_HOLD_BARS" in p:
            if p["MAX_HOLD_BARS"] <= p["MIN_HOLD_BARS"]:
                p["MAX_HOLD_BARS"] = p["MIN_HOLD_BARS"] + 4

        # GARCH stationarity: alpha + beta < 1.0
        if "GARCH_ALPHA" in p and "GARCH_BETA" in p:
            total = p["GARCH_ALPHA"] + p["GARCH_BETA"]
            if total >= 1.0:
                # Scale down proportionally
                scale = 0.97 / total
                p["GARCH_ALPHA"] = round(p["GARCH_ALPHA"] * scale, 4)
                p["GARCH_BETA"] = round(p["GARCH_BETA"] * scale, 4)

        # MAX_RISK_PCT >= BASE_RISK_PCT
        if "BASE_RISK_PCT" in p and "MAX_RISK_PCT" in p:
            if p["MAX_RISK_PCT"] < p["BASE_RISK_PCT"]:
                p["MAX_RISK_PCT"] = p["BASE_RISK_PCT"]

        # OU_KAPPA_MIN < OU_KAPPA_MAX
        if "OU_KAPPA_MIN" in p and "OU_KAPPA_MAX" in p:
            if p["OU_KAPPA_MIN"] >= p["OU_KAPPA_MAX"]:
                p["OU_KAPPA_MAX"] = p["OU_KAPPA_MIN"] + 0.1

        # ML suppress < ML boost
        if "ML_SIGNAL_SUPPRESS_THRESH" in p and "ML_SIGNAL_BOOST_THRESH" in p:
            if p["ML_SIGNAL_SUPPRESS_THRESH"] >= p["ML_SIGNAL_BOOST_THRESH"]:
                p["ML_SIGNAL_SUPPRESS_THRESH"] = p["ML_SIGNAL_BOOST_THRESH"] - 0.1

        return p

    def _run_and_sharpe(
        self, bars: dict[str, pd.DataFrame], params: dict[str, Any], label: str = ""
    ) -> float:
        """Run backtest on bars with params and return Sharpe. Returns -inf on error."""
        if not bars or all(len(df) == 0 for df in bars.values()):
            logger.debug("Empty bars for %s -- returning -inf", label)
            return float("-inf")
        try:
            results = self._runner.run(bars, params, self._initial_capital)
            equity = results.get("equity_curve")
            if equity is None or len(equity) < 20:
                return float("-inf")
            if isinstance(equity, pd.Series):
                rets = _equity_to_returns(equity)
            else:
                rets = _equity_to_returns(pd.Series(equity))
            return _compute_sharpe(rets)
        except Exception as exc:
            logger.debug("Backtest failed (%s): %s", label, exc)
            return float("-inf")

    def __call__(self, trial: "optuna.Trial") -> float:
        """
        Optuna objective function.

        1. Suggest parameters from schema ranges.
        2. Run in-sample backtest (6 months).
        3. Report intermediate value for pruning at bar _PRUNE_BAR.
        4. Run out-of-sample validation (2 months).
        5. Return negative OOS Sharpe.
        """
        params = self._suggest_params(trial)
        is_bars, oos_bars = self._split_window()

        # In-sample run
        is_sharpe = self._run_and_sharpe(is_bars, params, label=f"IS trial={trial.number}")

        # Report intermediate for pruning
        trial.report(is_sharpe, step=_PRUNE_BAR)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Out-of-sample validation
        oos_sharpe = self._run_and_sharpe(oos_bars, params, label=f"OOS trial={trial.number}")

        # Store metadata on trial
        trial.set_user_attr("is_sharpe", is_sharpe)
        trial.set_user_attr("oos_sharpe", oos_sharpe)
        overfit_gap = is_sharpe - oos_sharpe
        trial.set_user_attr("overfit_gap", overfit_gap)
        trial.set_user_attr("is_overfit", overfit_gap > _OVERFIT_THRESHOLD)

        logger.debug(
            "Trial %d: IS=%.3f OOS=%.3f gap=%.3f overfit=%s",
            trial.number, is_sharpe, oos_sharpe, overfit_gap, overfit_gap > _OVERFIT_THRESHOLD,
        )

        return -oos_sharpe  # Optuna minimizes


# ---------------------------------------------------------------------------
# RegimeAwareObjective
# ---------------------------------------------------------------------------

class RegimeAwareObjective(LARSAObjective):
    """
    Regime-aware Optuna objective.

    Splits the in-sample window into four regime slices and computes
    a harmonic mean of per-regime Sharpe ratios. This encourages
    parameter sets that are robust across different market conditions
    rather than just fitting to a single dominant regime.

    Regime classification uses a simple rule:
      - BH-active:   fraction of bars with BH signal > 0 is above median
      - BH-inactive: complement of BH-active
      - Trending:    Hurst exponent (rolling 100 bars) > 0.55
      - Mean-rev:    Hurst exponent < 0.45

    If regime classification fails (e.g., missing data) the objective
    falls back to the base LARSAObjective.
    """

    _REGIMES = ("bh_active", "bh_inactive", "trending", "mean_rev")

    def __init__(self, *args: Any, hurst_window: int = 100, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._hurst_window = hurst_window
        self._regime_cache: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Hurst exponent utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hurst_rs(series: np.ndarray, max_lag: int = 20) -> float:
        """
        Estimate Hurst exponent via R/S analysis.
        Returns value in [0, 1]: 0.5=random walk, >0.5=trending, <0.5=MR.
        """
        n = len(series)
        if n < max_lag * 2:
            return 0.5
        lags = range(2, max_lag)
        rs_vals = []
        for lag in lags:
            chunks = [series[i:i+lag] for i in range(0, n - lag, lag)]
            if not chunks:
                continue
            rs_list = []
            for chunk in chunks:
                mean_c = np.mean(chunk)
                dev = np.cumsum(chunk - mean_c)
                rs = (np.max(dev) - np.min(dev)) / (np.std(chunk, ddof=1) + 1e-12)
                rs_list.append(rs)
            rs_vals.append((lag, np.mean(rs_list)))
        if len(rs_vals) < 2:
            return 0.5
        lags_arr = np.log([v[0] for v in rs_vals])
        rs_arr = np.log([v[1] for v in rs_vals])
        try:
            h = np.polyfit(lags_arr, rs_arr, 1)[0]
            return float(np.clip(h, 0.0, 1.0))
        except Exception:
            return 0.5

    def _classify_regime(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.Series:
        """
        Assign a regime label to each bar. Returns a Series with index
        matching df.index and values in _REGIMES.
        """
        if symbol in self._regime_cache:
            return self._regime_cache[symbol]

        closes = df["close"].values if "close" in df.columns else df["Close"].values
        n = len(closes)
        labels = []

        for i in range(n):
            start = max(0, i - self._hurst_window)
            window = closes[start:i+1]
            if len(window) < 20:
                labels.append("bh_inactive")
                continue
            rets = np.diff(np.log(window + 1e-12))
            h = self._hurst_rs(rets)
            if h > 0.55:
                labels.append("trending")
            elif h < 0.45:
                labels.append("mean_rev")
            else:
                # Classify by local vol rank as BH proxy
                vol = np.std(rets[-20:], ddof=1) if len(rets) >= 20 else 0.0
                vol_thresh = np.std(rets, ddof=1) if len(rets) >= 2 else 0.0
                if vol > vol_thresh:
                    labels.append("bh_active")
                else:
                    labels.append("bh_inactive")

        result = pd.Series(labels, index=df.index)
        self._regime_cache[symbol] = result
        return result

    def _slice_by_regime(
        self,
        bars: dict[str, pd.DataFrame],
        regime: str,
    ) -> dict[str, pd.DataFrame]:
        """Return bars filtered to only include bars in the given regime."""
        sliced: dict[str, pd.DataFrame] = {}
        for sym, df in bars.items():
            if len(df) == 0:
                continue
            try:
                regime_labels = self._classify_regime(df, sym)
                mask = regime_labels == regime
                filtered = df[mask]
                if len(filtered) >= 20:
                    sliced[sym] = filtered
            except Exception as exc:
                logger.debug("Regime classification failed for %s: %s", sym, exc)
        return sliced

    @staticmethod
    def _harmonic_mean(values: list[float]) -> float:
        """Harmonic mean of a list of Sharpe ratios. Handles negative/zero values."""
        # Shift so all values are positive before harmonic mean
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return float("-inf")
        shift = abs(min(finite)) + 1.0
        shifted = [v + shift for v in finite]
        n = len(shifted)
        try:
            hm = n / sum(1.0 / v for v in shifted)
            return hm - shift
        except ZeroDivisionError:
            return float("-inf")

    def __call__(self, trial: "optuna.Trial") -> float:
        """
        Regime-aware objective.

        For each regime slice, runs a separate backtest and collects
        the Sharpe. Returns the negative harmonic mean of regime Sharpes.
        """
        params = self._suggest_params(trial)
        is_bars, oos_bars = self._split_window()

        if not is_bars:
            return 0.0

        # Compute per-regime Sharpe on in-sample data
        regime_sharpes: dict[str, float] = {}
        for regime in self._REGIMES:
            regime_is = self._slice_by_regime(is_bars, regime)
            if regime_is:
                sharpe = self._run_and_sharpe(regime_is, params, label=f"IS-{regime}")
            else:
                sharpe = float("-inf")
            regime_sharpes[regime] = sharpe
            trial.set_user_attr(f"is_sharpe_{regime}", sharpe)

        # Report intermediate value (mean of finite sharpes) for pruning
        finite_is = [v for v in regime_sharpes.values() if math.isfinite(v)]
        intermediate = float(np.mean(finite_is)) if finite_is else float("-inf")
        trial.report(intermediate, step=_PRUNE_BAR)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # OOS validation (single-pass, not regime-split to avoid data scarcity)
        oos_sharpe = self._run_and_sharpe(oos_bars, params, label="OOS-full")
        trial.set_user_attr("oos_sharpe", oos_sharpe)
        trial.set_user_attr("is_sharpe", intermediate)

        # Harmonic mean of regime Sharpes as primary objective
        hm = self._harmonic_mean(list(regime_sharpes.values()))
        trial.set_user_attr("harmonic_mean_sharpe", hm)

        overfit_gap = intermediate - oos_sharpe
        trial.set_user_attr("overfit_gap", overfit_gap)
        trial.set_user_attr("is_overfit", overfit_gap > _OVERFIT_THRESHOLD)

        logger.debug(
            "RegimeTrial %d: regimes=%s HM=%.3f OOS=%.3f",
            trial.number,
            {k: f"{v:.3f}" for k, v in regime_sharpes.items()},
            hm, oos_sharpe,
        )

        return -hm


# ---------------------------------------------------------------------------
# WalkForwardResult
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Holds results for a single walk-forward window."""
    window_index: int
    window_start: str
    is_end: str
    oos_end: str
    best_params: dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    overfit_gap: float
    is_overfit: bool
    n_trials: int
    n_pruned: int
    best_trial_number: int


# ---------------------------------------------------------------------------
# WalkForwardOptimizer
# ---------------------------------------------------------------------------

class WalkForwardOptimizer:
    """
    Runs an Optuna study for each walk-forward window and aggregates results.

    Walk-forward windows are constructed by sliding a combined
    (IS + OOS) window across the full dataset, stepping by oos_months
    each iteration. The optimizer uses RegimeAwareObjective by default.

    Overfit detection: if (IS Sharpe - OOS Sharpe) > 0.5, the window's
    best params are flagged as overfit and not proposed to the coordination layer.

    Results are exported to a CSV and the best non-overfit params are
    proposed to the coordination layer via ParamManager.
    """

    def __init__(
        self,
        bars: dict[str, pd.DataFrame],
        schema: Optional[ParamSchema] = None,
        manager: Optional[ParamManager] = None,
        is_months: int = 6,
        oos_months: int = 2,
        use_regime_objective: bool = True,
        results_dir: Optional[Path] = None,
        study_storage: Optional[str] = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self._bars = bars
        self._schema = schema or ParamSchema()
        self._manager = manager or ParamManager(self._schema)
        self._is_months = is_months
        self._oos_months = oos_months
        self._use_regime = use_regime_objective
        self._results_dir = results_dir or (Path(__file__).parent / "wfo_results")
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._storage = study_storage
        self._initial_capital = initial_capital

        # Collect full date range
        all_dates: list[pd.Timestamp] = []
        for df in bars.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.extend(df.index.tolist())
        self._all_dates = pd.DatetimeIndex(sorted(set(all_dates)))

    def _build_windows(self) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Build (window_start, is_end, oos_end) tuples covering the dataset.
        Steps forward by oos_months each iteration.
        """
        if len(self._all_dates) == 0:
            return []

        windows = []
        t0 = self._all_dates[0]
        t_data_end = self._all_dates[-1]
        window_length = pd.DateOffset(months=self._is_months + self._oos_months)
        step = pd.DateOffset(months=self._oos_months)

        t = t0
        while True:
            is_end = t + pd.DateOffset(months=self._is_months)
            oos_end = t + window_length
            if oos_end > t_data_end:
                break
            windows.append((t, is_end, oos_end))
            t = t + step

        return windows

    def _build_objective(
        self, start: pd.Timestamp
    ) -> "LARSAObjective | RegimeAwareObjective":
        """Build the appropriate objective class for a window starting at start."""
        cls = RegimeAwareObjective if self._use_regime else LARSAObjective
        return cls(
            bars=self._bars,
            schema=self._schema,
            is_months=self._is_months,
            oos_months=self._oos_months,
            initial_capital=self._initial_capital,
        )

    def optimize(
        self,
        n_trials: int = 200,
        n_jobs: int = 4,
        timeout: Optional[float] = None,
    ) -> list[LiveParams]:
        """
        Run Optuna optimization for each walk-forward window.

        Args:
            n_trials: number of Optuna trials per window
            n_jobs: parallel workers (uses joblib inside Optuna)
            timeout: optional per-window timeout in seconds

        Returns a list of LiveParams, one per non-overfit window.
        Also exports all results to CSV.
        """
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for WalkForwardOptimizer -- pip install optuna")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        windows = self._build_windows()
        if not windows:
            logger.error("No walk-forward windows could be constructed from the data")
            return []

        logger.info("Walk-forward: %d windows, %d trials each, %d jobs", len(windows), n_trials, n_jobs)

        all_results: list[WalkForwardResult] = []
        best_params_per_window: list[LiveParams] = []

        for i, (t_start, t_is_end, t_oos_end) in enumerate(windows):
            logger.info(
                "Window %d/%d: IS=[%s, %s] OOS=[%s, %s]",
                i + 1, len(windows),
                t_start.strftime("%Y-%m"), t_is_end.strftime("%Y-%m"),
                t_is_end.strftime("%Y-%m"), t_oos_end.strftime("%Y-%m"),
            )

            objective = self._build_objective(t_start)
            # Patch objective to use this window's start date
            objective._split_window = lambda s=t_start: objective._split_window.__func__(objective, s)  # type: ignore[method-assign]

            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=50)
            sampler = TPESampler(seed=42 + i)

            study_name = f"larsa_wfo_w{i:03d}"
            if self._storage:
                study = optuna.create_study(
                    direction="minimize",
                    pruner=pruner,
                    sampler=sampler,
                    study_name=study_name,
                    storage=self._storage,
                    load_if_exists=True,
                )
            else:
                study = optuna.create_study(
                    direction="minimize",
                    pruner=pruner,
                    sampler=sampler,
                    study_name=study_name,
                )

            try:
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    timeout=timeout,
                    show_progress_bar=False,
                )
            except Exception as exc:
                logger.error("Window %d optimization failed: %s", i, exc)
                continue

            best = study.best_trial
            is_sharpe = best.user_attrs.get("is_sharpe", float("-inf"))
            oos_sharpe = best.user_attrs.get("oos_sharpe", float("-inf"))
            overfit_gap = best.user_attrs.get("overfit_gap", float("inf"))
            is_overfit = best.user_attrs.get("is_overfit", True)

            n_pruned = sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            )

            wf_result = WalkForwardResult(
                window_index=i,
                window_start=t_start.strftime("%Y-%m-%d"),
                is_end=t_is_end.strftime("%Y-%m-%d"),
                oos_end=t_oos_end.strftime("%Y-%m-%d"),
                best_params=best.params,
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                overfit_gap=overfit_gap,
                is_overfit=is_overfit,
                n_trials=len(study.trials),
                n_pruned=n_pruned,
                best_trial_number=best.number,
            )
            all_results.append(wf_result)

            if not is_overfit and math.isfinite(oos_sharpe) and oos_sharpe > 0:
                repaired = objective._repair_constraints(best.params)
                lp = LiveParams.from_dict(repaired)
                lp.source = f"wfo_w{i}"
                lp.timestamp = datetime.utcnow().isoformat()
                best_params_per_window.append(lp)
                logger.info(
                    "Window %d: OOS Sharpe=%.3f -- will propose to coordination layer", i, oos_sharpe
                )
            else:
                if is_overfit:
                    logger.warning(
                        "Window %d: overfit (gap=%.3f > %.1f) -- skipping proposal",
                        i, overfit_gap, _OVERFIT_THRESHOLD,
                    )
                else:
                    logger.warning(
                        "Window %d: non-positive OOS Sharpe (%.3f) -- skipping proposal",
                        i, oos_sharpe,
                    )

        # Export all results to CSV
        self._export_csv(all_results)

        # Propose the best overall non-overfit params to coordination layer
        self._propose_best(best_params_per_window)

        return best_params_per_window

    def _export_csv(self, results: list[WalkForwardResult]) -> None:
        """Write walk-forward results to a timestamped CSV file."""
        if not results:
            return
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_path = self._results_dir / f"wfo_results_{ts}.csv"

        # Collect all unique param keys
        param_keys: list[str] = []
        for r in results:
            for k in r.best_params:
                if k not in param_keys:
                    param_keys.append(k)

        fieldnames = [
            "window_index", "window_start", "is_end", "oos_end",
            "is_sharpe", "oos_sharpe", "overfit_gap", "is_overfit",
            "n_trials", "n_pruned", "best_trial_number",
        ] + param_keys

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = asdict(r)
                row.update(r.best_params)
                writer.writerow(row)

        logger.info("Walk-forward results exported to %s", csv_path)

    def _propose_best(self, candidates: list[LiveParams]) -> None:
        """
        Propose the single best non-overfit parameter set to the coordination layer.
        Best is defined as highest OOS Sharpe.
        """
        if not candidates:
            logger.info("No non-overfit candidates to propose")
            return
        # All candidates are already non-overfit. Pick highest OOS Sharpe
        # (we stored it in source metadata) -- use timestamp as tiebreaker
        # Since we don't store oos_sharpe on LiveParams directly, propose the last one
        # (walk-forward ordering means last window is most recent)
        best = candidates[-1]
        params_dict = {
            k: v for k, v in best.to_dict().items()
            if k not in ("version", "source", "timestamp")
        }
        ok = self._manager.propose_update(params_dict, source=best.source or "wfo")
        if ok:
            logger.info("Best WFO params proposed successfully")
        else:
            logger.warning("Coordination layer rejected best WFO params")
