"""
research/walk_forward/backtest_adapter.py
──────────────────────────────────────────
Adapts the SRFM-Lab Black Hole engine (spacetime/engine/bh_engine.py) to the
walk-forward framework.

BHStrategyAdapter wraps the BH engine so that walk-forward splits receive a
standard callable:  strategy_fn(train_trades, params) → List[trade_dicts].

Because the BH engine works on price bars (OHLCV DataFrames), not pre-computed
trades, this adapter uses the trade records themselves to:
  1. Re-simulate the strategy on the train/test price segments using the
     provided BH parameters.
  2. Return the resulting trade list.

If raw OHLCV data is not available (common in research workflows), the adapter
operates in "re-tag mode": it re-tags existing trade records with the new BH
parameters' performance characteristics using a simplified P&L adjustment model.
This is a pragmatic approximation suitable for parameter stability analysis.

Parameters supported:
  cf           : Minkowski critical fraction (0.001–0.003)
  bh_form      : BH formation threshold (1.2–2.0)
  bh_collapse  : BH collapse threshold (0.6–1.0)
  bh_decay     : BH mass decay per bar (0.90–0.99)

Usage:
  >>> adapter = BHStrategyAdapter(ohlcv_data=price_df)
  >>> wf_engine = WalkForwardEngine(adapter.run_bh_strategy, param_grid, ...)
  >>> result = wf_engine.run(trades, splits)
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve project root paths for import
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LIB_PATH     = _PROJECT_ROOT / "lib"
_SPACETIME    = _PROJECT_ROOT / "spacetime"

for _p in [str(_LIB_PATH), str(_SPACETIME)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# Default BH parameter ranges (for reference / validation)
# ─────────────────────────────────────────────────────────────────────────────

BH_PARAM_DEFAULTS: Dict[str, float] = {
    "cf":          0.001,
    "bh_form":     1.5,
    "bh_collapse": 1.0,
    "bh_decay":    0.95,
}

BH_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "cf":          (0.001, 0.003),
    "bh_form":     (1.2,   2.0),
    "bh_collapse": (0.6,   1.0),
    "bh_decay":    (0.90,  0.99),
}


# ─────────────────────────────────────────────────────────────────────────────
# BHStrategyAdapter
# ─────────────────────────────────────────────────────────────────────────────

class BHStrategyAdapter:
    """
    Adapts the Black Hole engine to the walk-forward strategy_fn interface.

    Supports two operating modes:

    1. **Full simulation mode** (when `ohlcv_data` is provided):
       Runs the actual BH engine on the price bars corresponding to each
       train/test period. Produces realistic trades with all BH fields.

    2. **Re-tag mode** (when no OHLCV data is available):
       Uses existing trade records and adjusts P&L estimates based on how
       parameter changes would affect trade filtering and sizing. This is a
       heuristic approximation using BH physics relationships.

    Parameters
    ----------
    ohlcv_data      : optional OHLCV DataFrame with DatetimeIndex.
                      If provided, enables full simulation mode.
    sym             : instrument symbol (used to look up INSTRUMENT_CONFIGS).
    starting_equity : initial equity for position sizing.
    timeframe       : bar timeframe string ('daily', 'hourly', '15m').
    verbose         : log adapter activity.

    Examples
    --------
    >>> adapter = BHStrategyAdapter(ohlcv_data=price_df, sym='BTC')
    >>> param_grid = {
    ...     'cf': [0.001, 0.002, 0.003],
    ...     'bh_form': [1.2, 1.5, 2.0],
    ...     'bh_collapse': [0.8, 1.0],
    ...     'bh_decay': [0.92, 0.95, 0.98],
    ... }
    >>> engine = WalkForwardEngine(adapter.run_bh_strategy, param_grid)
    """

    def __init__(
        self,
        ohlcv_data:      Optional[pd.DataFrame] = None,
        sym:             str   = "BTC",
        starting_equity: float = 100_000.0,
        timeframe:       str   = "daily",
        verbose:         bool  = False,
    ) -> None:
        self.ohlcv_data      = ohlcv_data
        self.sym             = sym
        self.starting_equity = starting_equity
        self.timeframe       = timeframe
        self.verbose         = verbose

        # Try to import BH engine
        self._bh_engine_available = False
        try:
            from spacetime.engine.bh_engine import BHBacktester, BHConfig
            self._BHBacktester = BHBacktester
            self._BHConfig     = BHConfig
            self._bh_engine_available = True
            logger.debug("BHStrategyAdapter: BH engine loaded successfully")
        except ImportError as e:
            logger.info(
                "BH engine not importable (%s) — using re-tag mode", e
            )

    # ------------------------------------------------------------------
    # Main strategy function (walk-forward interface)
    # ------------------------------------------------------------------

    def run_bh_strategy(
        self,
        trades: pd.DataFrame,
        params: Dict[str, Any],
    ) -> List[Dict]:
        """
        Run BH strategy on a trades DataFrame with given parameters.

        This is the callable that WalkForwardEngine passes as strategy_fn.

        Parameters
        ----------
        trades : DataFrame of trades for the current fold (train or test).
                 Expected columns: pnl, dollar_pos, entry_price, exit_price,
                 hold_bars, regime, sym, exit_time.
        params : dict with BH parameters (cf, bh_form, bh_collapse, bh_decay).

        Returns
        -------
        List of trade dicts (may be a subset or adjusted version of input).
        """
        if trades is None or len(trades) == 0:
            return []

        params = self._validate_and_clamp_params(params)

        if self._bh_engine_available and self.ohlcv_data is not None:
            return self._run_full_simulation(trades, params)
        else:
            return self._run_retag_mode(trades, params)

    # ------------------------------------------------------------------
    # Full simulation mode
    # ------------------------------------------------------------------

    def _run_full_simulation(
        self,
        trades: pd.DataFrame,
        params: Dict[str, Any],
    ) -> List[Dict]:
        """
        Run the actual BH engine on the OHLCV segment corresponding to `trades`.

        Extracts the time range from the trade records, slices the OHLCV data,
        and runs a fresh BH backtest with the provided parameters.
        """
        # Determine time range from trades
        time_col = self._find_time_column(trades)
        if time_col is None:
            logger.warning("No time column found — falling back to re-tag mode")
            return self._run_retag_mode(trades, params)

        try:
            t_start = pd.Timestamp(trades[time_col].min())
            t_end   = pd.Timestamp(trades[time_col].max())
        except Exception:
            return self._run_retag_mode(trades, params)

        # Slice OHLCV
        ohlcv = self.ohlcv_data
        if isinstance(ohlcv.index, pd.DatetimeIndex):
            mask = (ohlcv.index >= t_start) & (ohlcv.index <= t_end)
        else:
            mask = np.ones(len(ohlcv), dtype=bool)

        ohlcv_slice = ohlcv[mask]
        if len(ohlcv_slice) < 50:
            logger.debug("Insufficient OHLCV bars (%d) — using re-tag mode", len(ohlcv_slice))
            return self._run_retag_mode(trades, params)

        # Build BH config
        try:
            config = self._BHConfig(
                cf          = params.get("cf",          BH_PARAM_DEFAULTS["cf"]),
                bh_form     = params.get("bh_form",     BH_PARAM_DEFAULTS["bh_form"]),
                bh_collapse = params.get("bh_collapse", BH_PARAM_DEFAULTS["bh_collapse"]),
                bh_decay    = params.get("bh_decay",    BH_PARAM_DEFAULTS["bh_decay"]),
            )
        except Exception as e:
            logger.warning("BHConfig construction failed: %s", e)
            return self._run_retag_mode(trades, params)

        try:
            backtester = self._BHBacktester(
                sym             = self.sym,
                config          = config,
                starting_equity = self.starting_equity,
            )
            bh_trades = backtester.run(ohlcv_slice)

            if isinstance(bh_trades, pd.DataFrame):
                return bh_trades.to_dict(orient="records")
            elif isinstance(bh_trades, (list, tuple)):
                return [t if isinstance(t, dict) else vars(t) for t in bh_trades]
            else:
                return []

        except Exception as e:
            logger.warning("BH full simulation failed: %s — falling back", e)
            return self._run_retag_mode(trades, params)

    # ------------------------------------------------------------------
    # Re-tag mode (parameter sensitivity approximation)
    # ------------------------------------------------------------------

    def _run_retag_mode(
        self,
        trades: pd.DataFrame,
        params: Dict[str, Any],
    ) -> List[Dict]:
        """
        Approximate the effect of BH parameter changes on existing trade records.

        This is used when OHLCV data is not available. It applies a physically-
        motivated adjustment to trade P&L based on how the parameter changes
        would affect:

        1. **cf (critical fraction)**: Affects which bars are classified as
           TIMELIKE vs SPACELIKE. Higher cf → more TIMELIKE bars → more signals.
           Approximated as a scaling factor on trade frequency (filter trades).

        2. **bh_form (formation threshold)**: Affects BH mass formation trigger.
           Higher bh_form → fewer BHs formed → fewer trades.
           Approximated via trade filtering by quality metric.

        3. **bh_collapse (collapse threshold)**: Affects exit timing.
           Lower bh_collapse → earlier exits → reduced hold_bars and P&L.
           Approximated via hold_bars multiplier.

        4. **bh_decay (mass decay)**: Affects BH persistence.
           Higher decay → longer-lived BHs → more hold_bars.
           Approximated via hold_bars scaling.

        Parameters
        ----------
        trades : current fold trade DataFrame.
        params : BH parameter dict.

        Returns
        -------
        List of adjusted trade dicts.
        """
        if trades.empty:
            return []

        result = trades.copy()

        cf          = float(params.get("cf",          BH_PARAM_DEFAULTS["cf"]))
        bh_form     = float(params.get("bh_form",     BH_PARAM_DEFAULTS["bh_form"]))
        bh_collapse = float(params.get("bh_collapse", BH_PARAM_DEFAULTS["bh_collapse"]))
        bh_decay    = float(params.get("bh_decay",    BH_PARAM_DEFAULTS["bh_decay"]))

        # ── 1. cf effect: trade frequency ──────────────────────────────────
        # Higher cf → looser TIMELIKE filter → more signals
        # cf range: 0.001–0.003 (3× range)
        # At base cf=0.001: keep 100% of trades
        # At cf=0.003: loosen filter → more trades (but lower quality)
        # We model this as: trades are subset-filtered when cf < base
        cf_base    = BH_PARAM_DEFAULTS["cf"]
        cf_ratio   = cf / cf_base  # 1.0 = base, >1 = looser, <1 = tighter

        if cf_ratio < 1.0:
            # Tighter filter: keep only the highest-quality trades
            # Quality proxy: trades with positive pnl or high pnl/dollar_pos
            if "pnl" in result.columns and "dollar_pos" in result.columns:
                pos_vals = result["dollar_pos"].replace(0, np.nan).fillna(1.0)
                quality  = result["pnl"] / pos_vals.abs()
                # Keep fraction proportional to cf_ratio (but at least 20%)
                keep_fraction = max(0.2, cf_ratio)
                threshold     = float(quality.quantile(1.0 - keep_fraction))
                result        = result[quality >= threshold].copy()
            else:
                keep_n = max(5, int(len(result) * max(0.2, cf_ratio)))
                result = result.head(keep_n).copy()

        # ── 2. bh_form effect: formation sensitivity ────────────────────────
        # Higher bh_form → higher threshold → fewer BHs → fewer trades
        form_base  = BH_PARAM_DEFAULTS["bh_form"]
        form_ratio = bh_form / form_base  # >1 = stricter = fewer trades

        if form_ratio > 1.0 and len(result) > 10:
            # At form=2.0 (max), keep ~60% of trades (only the strongest)
            # At form=1.2 (min), keep ~100%
            keep_fraction = max(0.5, 1.0 / form_ratio)

            if "pnl" in result.columns:
                # Keep top trades by absolute pnl magnitude
                keep_n = max(5, int(len(result) * keep_fraction))
                result = result.nlargest(keep_n, "pnl").sort_index()
            else:
                keep_n = max(5, int(len(result) * keep_fraction))
                result = result.head(keep_n).copy()

        # ── 3. bh_collapse effect: P&L and hold_bars adjustment ─────────────
        # Lower collapse threshold → exits earlier → less P&L captured per trade
        # Higher collapse → holds longer → more P&L (but more risk)
        collapse_base  = BH_PARAM_DEFAULTS["bh_collapse"]
        collapse_ratio = bh_collapse / collapse_base  # <1 = earlier exit

        if "pnl" in result.columns and abs(collapse_ratio - 1.0) > 0.01:
            # Adjust P&L: earlier exit captures less upside on winners,
            # but cuts losses shorter on losers
            pnl_col = result["pnl"].copy()

            # Winners: scale proportional to collapse_ratio
            winner_mask   = pnl_col > 0
            result.loc[winner_mask, "pnl"] = pnl_col[winner_mask] * collapse_ratio

            # Losers: earlier exit = less loss
            loser_mask    = pnl_col < 0
            if loser_mask.sum() > 0:
                # Less collapse = exit sooner = smaller loss
                result.loc[loser_mask, "pnl"] = pnl_col[loser_mask] * min(1.0, collapse_ratio + 0.2)

        if "hold_bars" in result.columns and abs(collapse_ratio - 1.0) > 0.01:
            result["hold_bars"] = (result["hold_bars"] * collapse_ratio).round().clip(lower=1).astype(int)

        # ── 4. bh_decay effect: position duration scaling ───────────────────
        # Higher decay → BH persists longer → trades held longer
        decay_base  = BH_PARAM_DEFAULTS["bh_decay"]
        # Decay ratio: effect is multiplicative on hold duration
        # At decay=0.99 (slow decay), relative to 0.95 base:
        # effective_lifetime = 1/(1-decay)
        if abs(bh_decay - decay_base) > 0.001:
            lifetime_base    = 1.0 / (1.0 - decay_base + 1e-8)
            lifetime_new     = 1.0 / (1.0 - bh_decay  + 1e-8)
            lifetime_ratio   = lifetime_new / (lifetime_base + 1e-8)
            lifetime_ratio   = np.clip(lifetime_ratio, 0.3, 3.0)

            if "hold_bars" in result.columns:
                result["hold_bars"] = (result["hold_bars"] * lifetime_ratio).round().clip(lower=1).astype(int)

            # Higher decay can slightly improve winners (more time in trade)
            if "pnl" in result.columns and abs(lifetime_ratio - 1.0) > 0.05:
                pnl_col      = result["pnl"].copy()
                winner_mask  = pnl_col > 0
                # Winners benefit from slightly longer holds (up to +20%)
                pnl_mult     = min(1.2, 1.0 + 0.05 * (lifetime_ratio - 1.0))
                result.loc[winner_mask, "pnl"] = pnl_col[winner_mask] * pnl_mult

        if result.empty:
            return []

        result = result.reset_index(drop=True)
        return result.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def _validate_and_clamp_params(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and clamp BH parameters to their valid ranges.

        Parameters outside the defined bounds are clamped with a warning.
        Missing parameters are filled from BH_PARAM_DEFAULTS.
        """
        cleaned: Dict[str, Any] = {}
        for param, default in BH_PARAM_DEFAULTS.items():
            val = params.get(param, default)
            try:
                val = float(val)
            except (TypeError, ValueError):
                logger.warning(
                    "BHStrategyAdapter: non-numeric value for '%s': %s — using default %.4f",
                    param, val, default,
                )
                val = default

            low, high = BH_PARAM_BOUNDS[param]
            if val < low or val > high:
                clamped = float(np.clip(val, low, high))
                logger.warning(
                    "BHStrategyAdapter: param '%s'=%.4f out of bounds [%.4f, %.4f] — clamped to %.4f",
                    param, val, low, high, clamped,
                )
                val = clamped

            cleaned[param] = val

        # Pass through any extra params unmodified
        for k, v in params.items():
            if k not in cleaned:
                cleaned[k] = v

        return cleaned

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _find_time_column(df: pd.DataFrame) -> Optional[str]:
        """Find the most likely time column in a DataFrame."""
        time_cols = ["exit_time", "entry_time", "timestamp", "date", "time"]
        for col in time_cols:
            if col in df.columns:
                return col
        # Check DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.name or "index"
        return None

    def get_param_grid_for_ranges(
        self,
        cf_values:          Optional[List[float]] = None,
        bh_form_values:     Optional[List[float]] = None,
        bh_collapse_values: Optional[List[float]] = None,
        bh_decay_values:    Optional[List[float]] = None,
    ) -> Dict[str, List[float]]:
        """
        Build a parameter grid dict for use with WalkForwardEngine.

        Provides default ranges if specific values are not supplied.

        Parameters
        ----------
        cf_values          : list of cf values (default: [0.001, 0.002, 0.003]).
        bh_form_values     : list of bh_form values (default: [1.2, 1.5, 2.0]).
        bh_collapse_values : list of bh_collapse values (default: [0.8, 1.0]).
        bh_decay_values    : list of bh_decay values (default: [0.92, 0.95, 0.98]).

        Returns
        -------
        Dict suitable for WalkForwardEngine param_grid.

        Examples
        --------
        >>> param_grid = adapter.get_param_grid_for_ranges(
        ...     cf_values=[0.001, 0.002],
        ...     bh_form_values=[1.5, 2.0],
        ... )
        """
        return {
            "cf":          cf_values          or [0.001, 0.0015, 0.002, 0.0025, 0.003],
            "bh_form":     bh_form_values     or [1.2, 1.5, 1.8, 2.0],
            "bh_collapse": bh_collapse_values or [0.7, 0.8, 0.9, 1.0],
            "bh_decay":    bh_decay_values    or [0.90, 0.92, 0.95, 0.98],
        }

    def get_sobol_param_space(self) -> Dict[str, tuple]:
        """
        Return a parameter space dict for Sobol/random/Bayesian optimization.

        Returns
        -------
        Dict with (low, high) or (low, high, log_scale) tuples for each param.
        """
        return {
            "cf":          (0.001, 0.003, True),   # log-scale
            "bh_form":     (1.2,   2.0),
            "bh_collapse": (0.6,   1.0),
            "bh_decay":    (0.90,  0.99),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run_bh_strategy function
# ─────────────────────────────────────────────────────────────────────────────

def run_bh_strategy(
    train_trades:    pd.DataFrame,
    params:          Dict[str, Any],
    ohlcv_data:      Optional[pd.DataFrame] = None,
    sym:             str   = "BTC",
    starting_equity: float = 100_000.0,
) -> List[Dict]:
    """
    Standalone function wrapping BHStrategyAdapter for single-call use.

    Convenience function for one-off strategy evaluation without creating
    a persistent adapter object.

    Parameters
    ----------
    train_trades    : trade DataFrame for the current fold.
    params          : BH parameter dict.
    ohlcv_data      : optional OHLCV price DataFrame.
    sym             : instrument symbol.
    starting_equity : initial equity.

    Returns
    -------
    List of trade dicts.

    Examples
    --------
    >>> trades = run_bh_strategy(
    ...     train_trades,
    ...     params={'cf': 0.002, 'bh_form': 1.5, 'bh_collapse': 0.9, 'bh_decay': 0.95},
    ... )
    """
    adapter = BHStrategyAdapter(
        ohlcv_data      = ohlcv_data,
        sym             = sym,
        starting_equity = starting_equity,
    )
    return adapter.run_bh_strategy(train_trades, params)


# ─────────────────────────────────────────────────────────────────────────────
# MC bridge: connect BH adapter with Monte Carlo engine
# ─────────────────────────────────────────────────────────────────────────────

def run_bh_mc_analysis(
    trades:          pd.DataFrame,
    params:          Dict[str, Any],
    starting_equity: float = 100_000.0,
    n_sims:          int   = 10_000,
    months:          int   = 12,
) -> Optional[object]:
    """
    Run Monte Carlo analysis on BH-strategy trades with given parameters.

    Bridges BHStrategyAdapter with the spacetime MC engine.

    Parameters
    ----------
    trades          : trade DataFrame.
    params          : BH parameter dict.
    starting_equity : initial equity.
    n_sims          : number of Monte Carlo simulations.
    months          : simulation horizon in months.

    Returns
    -------
    MCResult from spacetime/engine/mc.py, or None if MC engine unavailable.
    """
    try:
        from engine.mc import run_mc, MCConfig
    except ImportError:
        try:
            from spacetime.engine.mc import run_mc, MCConfig
        except ImportError:
            logger.warning("MC engine not importable — run_bh_mc_analysis skipped")
            return None

    adapter    = BHStrategyAdapter(starting_equity=starting_equity)
    bh_trades  = adapter.run_bh_strategy(trades, params)

    if not bh_trades:
        logger.warning("run_bh_mc_analysis: no trades generated by BH strategy")
        return None

    cfg = MCConfig(
        n_sims          = n_sims,
        months          = months,
        regime_aware    = True,
    )

    try:
        mc_result = run_mc(bh_trades, starting_equity, cfg)
        return mc_result
    except Exception as e:
        logger.error("MC run failed: %s", e)
        return None
