"""
spacetime/engine/scenario_engine.py
======================================
Historical scenario analysis engine for the Spacetime Arena.

Replays SRFM through named historical crisis and boom scenarios, tests
parameter robustness across different market regimes, and produces
structured comparison reports.

Classes:
  ScenarioResult         -- metrics from a single scenario backtest
  ScenarioEngine         -- main engine: run, compare, stress-test
  ScenarioStressTest     -- shifts params by +-20%, finds breaking points
  ScenarioComparison     -- compare two param sets across all scenarios

Requires: numpy, pandas
"""

from __future__ import annotations

import copy
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bh_engine import GEEKY_DEFAULTS, INSTRUMENT_CONFIGS, run_backtest
from .data_loader import load_bars  # noqa: F401 -- used for bar fetching

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, str]] = {
    "dot_com_crash":      {"start": "2000-03-01", "end": "2002-10-01",
                           "label": "Dot-Com Crash", "regime": "Bear"},
    "gfc_2008":           {"start": "2007-10-01", "end": "2009-03-01",
                           "label": "GFC 2008-09", "regime": "Bear"},
    "covid_crash":        {"start": "2020-02-19", "end": "2020-03-23",
                           "label": "COVID Crash", "regime": "HighVol"},
    "covid_recovery":     {"start": "2020-03-24", "end": "2021-12-31",
                           "label": "COVID Recovery", "regime": "Bull"},
    "bear_2022":          {"start": "2021-11-01", "end": "2022-10-01",
                           "label": "Rate-Hike Bear 2022", "regime": "Bear"},
    "crypto_winter_2022": {"start": "2022-04-01", "end": "2022-12-01",
                           "label": "Crypto Winter 2022", "regime": "Bear"},
}

# -- additional scenarios that may be loaded on demand
EXTENDED_SCENARIOS: Dict[str, Dict[str, str]] = {
    "flash_crash_2010":   {"start": "2010-05-06", "end": "2010-05-07",
                           "label": "Flash Crash 2010", "regime": "HighVol"},
    "taper_tantrum_2013": {"start": "2013-05-01", "end": "2013-07-01",
                           "label": "Taper Tantrum 2013", "regime": "HighVol"},
    "china_devalue_2015": {"start": "2015-08-18", "end": "2015-09-30",
                           "label": "China Devaluation 2015", "regime": "HighVol"},
    "vol_crush_2018":     {"start": "2018-02-02", "end": "2018-02-09",
                           "label": "VIX Spike Feb 2018", "regime": "HighVol"},
    "bull_2017":          {"start": "2017-01-01", "end": "2017-12-31",
                           "label": "Low-Vol Bull 2017", "regime": "Bull"},
    "post_gfc_bull":      {"start": "2009-03-09", "end": "2011-04-29",
                           "label": "Post-GFC Bull", "regime": "Bull"},
}

_STRESS_SHIFT_FRAC = 0.20    -- +-20% for stress test
_BREAKING_SHARPE = 0.0       -- scenario considered broken if Sharpe falls below this
_MIN_BARS = 20               -- skip scenario if fewer bars available
_MAX_WORKERS = 4             -- for parallel scenario execution


# ---------------------------------------------------------------------------
# ScenarioResult
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Metrics from a single SRFM scenario backtest."""
    scenario: str
    label: str
    start: str
    end: str
    regime: str
    total_return: float       -- total P&L as a fraction (e.g. 0.12 = 12%)
    max_dd: float             -- maximum drawdown (positive number)
    sharpe: float
    calmar: float
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_bh_mass: float        -- average BH mass at trade entries
    n_bars: int
    eval_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Return True if scenario produced acceptable results."""
        return self.error is None and self.n_trades >= 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def error_result(cls, scenario: str, meta: Dict[str, str], msg: str) -> "ScenarioResult":
        return cls(
            scenario=scenario,
            label=meta.get("label", scenario),
            start=meta.get("start", ""),
            end=meta.get("end", ""),
            regime=meta.get("regime", ""),
            total_return=0.0,
            max_dd=0.0,
            sharpe=0.0,
            calmar=0.0,
            n_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_bh_mass=0.0,
            n_bars=0,
            error=msg,
        )


def _parse_result(raw: Dict[str, Any]) -> Tuple[float, float, float, float, int, float, float, float]:
    """Extract standard metrics from run_backtest output dict."""
    sharpe = float(raw.get("sharpe", 0.0))
    max_dd = float(raw.get("max_drawdown", 0.0))
    total_return = float(raw.get("total_return", raw.get("cagr", 0.0)))
    calmar = float(raw.get("calmar", 0.0))
    n_trades = int(raw.get("n_trades", 0))
    win_rate = float(raw.get("win_rate", 0.0))
    pf = float(raw.get("profit_factor", 1.0))
    avg_bh = float(raw.get("avg_bh_mass", raw.get("bh_mass_mean", 0.0)))
    return sharpe, max_dd, total_return, calmar, n_trades, win_rate, pf, avg_bh


# ---------------------------------------------------------------------------
# ScenarioEngine
# ---------------------------------------------------------------------------

class ScenarioEngine:
    """
    Replays SRFM through historical crisis and boom scenarios.
    Tests parameter robustness across different market regimes.

    Parameters
    ----------
    sym : str
        Instrument symbol.
    bars : pd.DataFrame
        Full OHLCV bar history. Must include all scenario date ranges.
        If None, bars will be loaded via data_loader.load_bars().
    base_params : dict, optional
        Base parameter overrides on top of instrument + Geeky defaults.
    long_only : bool
        Backtest mode.
    extra_scenarios : dict, optional
        Additional scenario definitions to merge with SCENARIOS.
    """

    def __init__(
        self,
        sym: str,
        bars: Optional[pd.DataFrame],
        base_params: Optional[Dict[str, Any]] = None,
        long_only: bool = True,
        extra_scenarios: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.sym = sym.upper()
        self.long_only = long_only

        cfg = dict(INSTRUMENT_CONFIGS.get(self.sym, INSTRUMENT_CONFIGS["ES"]))
        cfg.update(GEEKY_DEFAULTS)
        if base_params:
            cfg.update(base_params)
        self.base_params = cfg

        self.all_scenarios: Dict[str, Dict[str, str]] = dict(SCENARIOS)
        if extra_scenarios:
            self.all_scenarios.update(extra_scenarios)

        # -- bar store
        if bars is not None:
            self._bars = bars.copy()
        else:
            self._bars = None
        self._bars_indexed: Optional[pd.DataFrame] = None
        if self._bars is not None:
            self._index_bars()

    def _index_bars(self) -> None:
        """Ensure bars have a sortable DatetimeIndex for slicing."""
        df = self._bars.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"])
            elif "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            else:
                df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        self._bars_indexed = df

    def _get_scenario_bars(self, name: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Return bars and metadata for a named scenario.

        Raises ValueError if scenario is not found.
        Raises RuntimeError if bars are not available.
        """
        meta = self.all_scenarios.get(name)
        if meta is None:
            raise ValueError(f"Unknown scenario: '{name}'. "
                             f"Available: {list(self.all_scenarios)}")

        if self._bars_indexed is None:
            raise RuntimeError(
                "No bar data available. Pass bars= to ScenarioEngine "
                "or ensure data_loader can fetch them."
            )

        start = pd.Timestamp(meta["start"])
        end = pd.Timestamp(meta["end"])
        sliced = self._bars_indexed.loc[start:end]
        return sliced, meta

    def run_scenario(
        self, name: str, params: Optional[Dict[str, Any]] = None
    ) -> ScenarioResult:
        """
        Run SRFM backtest on a single named scenario window.

        Parameters
        ----------
        name : str
            Scenario key (e.g. "gfc_2008").
        params : dict, optional
            Parameter overrides (merged on top of base_params).

        Returns
        -------
        ScenarioResult
        """
        run_params = dict(self.base_params)
        if params:
            run_params.update(params)

        try:
            scenario_bars, meta = self._get_scenario_bars(name)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Scenario '%s' fetch failed: %s", name, exc)
            return ScenarioResult.error_result(name, self.all_scenarios.get(name, {}), str(exc))

        if len(scenario_bars) < _MIN_BARS:
            msg = f"Only {len(scenario_bars)} bars in scenario '{name}' (< {_MIN_BARS})"
            logger.warning(msg)
            return ScenarioResult.error_result(name, meta, msg)

        t0 = time.perf_counter()
        try:
            raw = run_backtest(self.sym, scenario_bars, run_params, self.long_only)
            sharpe, max_dd, total_return, calmar, n_trades, win_rate, pf, avg_bh = _parse_result(raw)
            elapsed = time.perf_counter() - t0

            return ScenarioResult(
                scenario=name,
                label=meta.get("label", name),
                start=meta["start"],
                end=meta["end"],
                regime=meta.get("regime", ""),
                total_return=total_return,
                max_dd=max_dd,
                sharpe=sharpe,
                calmar=calmar,
                n_trades=n_trades,
                win_rate=win_rate,
                profit_factor=pf,
                avg_bh_mass=avg_bh,
                n_bars=len(scenario_bars),
                eval_time_s=elapsed,
            )
        except Exception as exc:
            logger.error("Backtest error in scenario '%s': %s", name, exc)
            return ScenarioResult.error_result(name, meta, str(exc))

    def run_all_scenarios(
        self,
        params: Optional[Dict[str, Any]] = None,
        parallel: bool = True,
        scenario_names: Optional[List[str]] = None,
    ) -> List[ScenarioResult]:
        """
        Run all (or a subset of) scenarios in parallel.

        Parameters
        ----------
        params : dict, optional
            Parameter overrides.
        parallel : bool
            If True, use ThreadPoolExecutor.
        scenario_names : list, optional
            If provided, only run these scenario keys.

        Returns
        -------
        list of ScenarioResult, sorted by start date.
        """
        names = scenario_names or list(self.all_scenarios.keys())
        results: List[ScenarioResult] = []

        if parallel and len(names) > 1:
            with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self.run_scenario, name, params): name
                    for name in names
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        name = futures[future]
                        logger.error("Future error for '%s': %s", name, exc)
                        results.append(ScenarioResult.error_result(
                            name, self.all_scenarios.get(name, {}), str(exc)
                        ))
        else:
            for name in names:
                results.append(self.run_scenario(name, params))

        # -- sort by scenario start date
        results.sort(key=lambda r: r.start)
        return results

    def results_to_dataframe(self, results: List[ScenarioResult]) -> pd.DataFrame:
        """Convert a list of ScenarioResults to a summary DataFrame."""
        rows = [r.to_dict() for r in results]
        return pd.DataFrame(rows)

    def best_scenario(self, results: List[ScenarioResult]) -> Optional[ScenarioResult]:
        """Return the scenario with the highest Sharpe."""
        valid = [r for r in results if r.passed]
        return max(valid, key=lambda r: r.sharpe) if valid else None

    def worst_scenario(self, results: List[ScenarioResult]) -> Optional[ScenarioResult]:
        """Return the scenario with the lowest Sharpe."""
        valid = [r for r in results if r.passed]
        return min(valid, key=lambda r: r.sharpe) if valid else None

    def summary_stats(self, results: List[ScenarioResult]) -> Dict[str, float]:
        """Aggregate statistics across all scenario results."""
        valid = [r for r in results if r.passed]
        if not valid:
            return {}
        sharpes = np.array([r.sharpe for r in valid])
        max_dds = np.array([r.max_dd for r in valid])
        returns = np.array([r.total_return for r in valid])
        n_pos = sum(1 for r in valid if r.sharpe > 0)
        n_neg_dd = sum(1 for r in valid if r.sharpe < -0.5)
        return {
            "n_scenarios": len(valid),
            "mean_sharpe": float(np.mean(sharpes)),
            "median_sharpe": float(np.median(sharpes)),
            "min_sharpe": float(np.min(sharpes)),
            "max_sharpe": float(np.max(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "mean_max_dd": float(np.mean(max_dds)),
            "worst_dd": float(np.max(max_dds)),
            "mean_return": float(np.mean(returns)),
            "pct_positive_sharpe": n_pos / len(valid),
            "n_severely_negative": n_neg_dd,
            "pass_rate": n_pos / len(valid),
        }


# ---------------------------------------------------------------------------
# ScenarioStressTest
# ---------------------------------------------------------------------------

class ScenarioStressTest:
    """
    Shift params by +-20% and rerun all scenarios, find breaking points.

    A "breaking point" is a parameter shift that causes at least one
    scenario Sharpe to drop below _BREAKING_SHARPE (0.0).
    """

    def __init__(self, engine: ScenarioEngine):
        self.engine = engine

    def _shift_param(
        self, params: Dict[str, Any], param_name: str, shift_frac: float
    ) -> Dict[str, Any]:
        """Return a copy of params with param_name shifted by shift_frac."""
        shifted = copy.deepcopy(params)
        val = float(shifted.get(param_name, 0.0))
        shifted[param_name] = val * (1.0 + shift_frac)
        return shifted

    def run(
        self,
        params: Optional[Dict[str, Any]] = None,
        param_names: Optional[List[str]] = None,
        shifts: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Stress-test by shifting each parameter by +-20%.

        Parameters
        ----------
        params : dict, optional
            Base parameter set (defaults to engine.base_params).
        param_names : list, optional
            Parameters to shift (defaults to ["cf", "bh_form", "bh_decay", "bh_collapse"]).
        shifts : list of float, optional
            Shift fractions (defaults to [-0.20, +0.20]).

        Returns
        -------
        dict with:
          baseline_stats : summary stats at base params
          shift_results  : {param: {shift: summary_stats}}
          breaking_points: list of {param, shift, scenario, sharpe}
        """
        base = params or dict(self.engine.base_params)
        if param_names is None:
            param_names = ["cf", "bh_form", "bh_decay", "bh_collapse"]
        if shifts is None:
            shifts = [-_STRESS_SHIFT_FRAC, +_STRESS_SHIFT_FRAC]

        # -- baseline
        baseline_results = self.engine.run_all_scenarios(base)
        baseline_stats = self.engine.summary_stats(baseline_results)

        shift_results: Dict[str, Dict[str, Any]] = {}
        breaking_points: List[Dict[str, Any]] = []

        for pname in param_names:
            shift_results[pname] = {}
            for s in shifts:
                shifted = self._shift_param(base, pname, s)
                results = self.engine.run_all_scenarios(shifted)
                stats = self.engine.summary_stats(results)
                shift_key = f"{s:+.0%}"
                shift_results[pname][shift_key] = stats

                # -- check for breaking points
                for r in results:
                    if r.passed and r.sharpe < _BREAKING_SHARPE:
                        breaking_points.append({
                            "param": pname,
                            "shift_frac": s,
                            "shift_pct": f"{s:+.0%}",
                            "scenario": r.scenario,
                            "sharpe": r.sharpe,
                            "max_dd": r.max_dd,
                        })

        logger.info(
            "Stress test complete. Breaking points found: %d", len(breaking_points)
        )
        return {
            "baseline_stats": baseline_stats,
            "shift_results": shift_results,
            "breaking_points": breaking_points,
            "n_breaking_points": len(breaking_points),
            "params_tested": param_names,
            "shifts_tested": shifts,
        }

    def fragility_report(
        self, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Run stress test and return results as a DataFrame.

        Columns: param, shift, scenario, sharpe, max_dd, is_breaking.
        """
        results = self.run(params)
        rows = []
        for bp in results["breaking_points"]:
            rows.append({**bp, "is_breaking": True})
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["param", "shift_frac", "shift_pct", "scenario",
                     "sharpe", "max_dd", "is_breaking"]
        )


# ---------------------------------------------------------------------------
# ScenarioComparison
# ---------------------------------------------------------------------------

class ScenarioComparison:
    """
    Given two parameter sets, compare their performance across all scenarios.

    Useful for evaluating whether a new parameter set is better than the
    current production set across historical regimes.
    """

    def __init__(self, engine: ScenarioEngine):
        self.engine = engine

    def compare(
        self,
        params_a: Dict[str, Any],
        params_b: Dict[str, Any],
        label_a: str = "A",
        label_b: str = "B",
        scenario_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run both param sets on all scenarios and compare.

        Returns
        -------
        dict with:
          results_a, results_b : list of ScenarioResult
          summary_a, summary_b : aggregate stats
          deltas : {scenario: {sharpe_delta, dd_delta, return_delta}}
          winner : label of the param set with higher mean Sharpe
          win_counts : {label_a: n, label_b: n, ties: n} (per-scenario wins)
        """
        results_a = self.engine.run_all_scenarios(params_a, scenario_names=scenario_names)
        results_b = self.engine.run_all_scenarios(params_b, scenario_names=scenario_names)

        summary_a = self.engine.summary_stats(results_a)
        summary_b = self.engine.summary_stats(results_b)

        # -- per-scenario deltas (B - A)
        a_dict = {r.scenario: r for r in results_a}
        b_dict = {r.scenario: r for r in results_b}
        deltas: Dict[str, Dict[str, float]] = {}
        wins_a = 0
        wins_b = 0
        ties = 0

        for name in a_dict:
            if name not in b_dict:
                continue
            ra = a_dict[name]
            rb = b_dict[name]
            if not ra.passed or not rb.passed:
                continue
            sharpe_delta = rb.sharpe - ra.sharpe
            dd_delta = rb.max_dd - ra.max_dd
            return_delta = rb.total_return - ra.total_return
            deltas[name] = {
                "sharpe_a": ra.sharpe,
                "sharpe_b": rb.sharpe,
                "sharpe_delta": sharpe_delta,
                "dd_delta": dd_delta,
                "return_delta": return_delta,
            }
            if sharpe_delta > 0.05:
                wins_b += 1
            elif sharpe_delta < -0.05:
                wins_a += 1
            else:
                ties += 1

        mean_a = summary_a.get("mean_sharpe", 0.0)
        mean_b = summary_b.get("mean_sharpe", 0.0)
        winner = label_a if mean_a >= mean_b else label_b

        return {
            "label_a": label_a,
            "label_b": label_b,
            "summary_a": summary_a,
            "summary_b": summary_b,
            "deltas": deltas,
            "winner": winner,
            "win_counts": {label_a: wins_a, label_b: wins_b, "ties": ties},
            "mean_sharpe_a": mean_a,
            "mean_sharpe_b": mean_b,
            "mean_sharpe_delta": mean_b - mean_a,
        }

    def compare_to_dataframe(
        self,
        params_a: Dict[str, Any],
        params_b: Dict[str, Any],
        label_a: str = "A",
        label_b: str = "B",
    ) -> pd.DataFrame:
        """Return comparison as a per-scenario DataFrame."""
        cmp = self.compare(params_a, params_b, label_a, label_b)
        rows = []
        for scenario, delta in cmp["deltas"].items():
            rows.append({
                "scenario": scenario,
                f"sharpe_{label_a}": delta["sharpe_a"],
                f"sharpe_{label_b}": delta["sharpe_b"],
                "sharpe_delta": delta["sharpe_delta"],
                "dd_delta": delta["dd_delta"],
                "return_delta": delta["return_delta"],
                "winner": label_b if delta["sharpe_delta"] > 0.05
                          else (label_a if delta["sharpe_delta"] < -0.05 else "tie"),
            })
        return pd.DataFrame(rows)

    def regime_breakdown(
        self,
        params_a: Dict[str, Any],
        params_b: Dict[str, Any],
        label_a: str = "A",
        label_b: str = "B",
    ) -> pd.DataFrame:
        """
        Show mean Sharpe per regime type for both parameter sets.
        """
        cmp = self.compare(params_a, params_b, label_a, label_b)
        results_a = self.engine.run_all_scenarios(params_a)
        results_b = self.engine.run_all_scenarios(params_b)

        a_by_regime: Dict[str, List[float]] = {}
        b_by_regime: Dict[str, List[float]] = {}

        for r in results_a:
            if r.passed and r.regime:
                a_by_regime.setdefault(r.regime, []).append(r.sharpe)

        for r in results_b:
            if r.passed and r.regime:
                b_by_regime.setdefault(r.regime, []).append(r.sharpe)

        regimes = sorted(set(list(a_by_regime.keys()) + list(b_by_regime.keys())))
        rows = []
        for reg in regimes:
            a_sharpes = a_by_regime.get(reg, [])
            b_sharpes = b_by_regime.get(reg, [])
            rows.append({
                "regime": reg,
                f"mean_sharpe_{label_a}": float(np.mean(a_sharpes)) if a_sharpes else float("nan"),
                f"mean_sharpe_{label_b}": float(np.mean(b_sharpes)) if b_sharpes else float("nan"),
                "n_scenarios": max(len(a_sharpes), len(b_sharpes)),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_scenario_suite(
    sym: str,
    bars: pd.DataFrame,
    params: Dict[str, Any],
    scenario_names: Optional[List[str]] = None,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper: create a ScenarioEngine and run all scenarios.

    Returns results as a DataFrame.
    """
    engine = ScenarioEngine(sym=sym, bars=bars, base_params=params, long_only=long_only)
    results = engine.run_all_scenarios(params, scenario_names=scenario_names)
    return engine.results_to_dataframe(results)
