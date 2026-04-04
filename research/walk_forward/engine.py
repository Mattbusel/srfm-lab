"""
research/walk_forward/engine.py
────────────────────────────────
Walk-Forward Engine and CPCV Execution Engine.

Orchestrates parallel fold execution, result aggregation, and CPCV path analysis.
Produces WFResult and CPCVResult dataclasses with full OOS performance diagnostics.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .splits import WFSplit, CPCVSplitter, purge_overlap
from .metrics import (
    PerformanceStats,
    compute_performance_stats,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    cagr,
    profit_factor,
    win_rate,
    bootstrap_confidence_interval,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """
    Result for a single walk-forward fold.

    Attributes
    ----------
    fold_id       : zero-based fold index.
    params        : parameter dict used for this fold.
    is_stats      : in-sample PerformanceStats.
    oos_stats     : out-of-sample PerformanceStats.
    oos_trades    : list of OOS trade dicts (from strategy_fn).
    is_trades     : list of IS trade dicts (from strategy_fn training).
    elapsed_sec   : wall-clock seconds for this fold.
    error         : error message if fold failed, else None.
    split         : the WFSplit object for this fold.
    """
    fold_id:     int
    params:      Dict[str, Any]
    is_stats:    PerformanceStats
    oos_stats:   PerformanceStats
    oos_trades:  List[Dict]
    is_trades:   List[Dict]
    elapsed_sec: float = 0.0
    error:       Optional[str] = None
    split:       Optional[WFSplit] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def oos_sharpe(self) -> float:
        return self.oos_stats.sharpe

    @property
    def oos_max_dd(self) -> float:
        return self.oos_stats.max_dd

    @property
    def oos_cagr(self) -> float:
        return self.oos_stats.cagr_ann


@dataclass
class WFResult:
    """
    Aggregated result from a full walk-forward analysis.

    Attributes
    ----------
    fold_results         : list of FoldResult per fold.
    best_params          : parameter dict selected most frequently as best.
    oos_sharpe           : mean OOS Sharpe across folds.
    oos_cagr             : mean OOS CAGR across folds.
    oos_max_dd           : worst OOS max drawdown across folds.
    param_stability_score: fraction of folds with same best params (0-1).
    sharpe_ci            : (lower, upper) bootstrap 95% CI for OOS Sharpe.
    oos_sharpe_series    : OOS Sharpe per fold (for drift analysis).
    is_sharpe_series     : IS Sharpe per fold.
    combined_oos_trades  : all OOS trades concatenated in time order.
    combined_oos_stats   : performance stats on combined OOS trades.
    n_folds              : number of completed folds.
    total_elapsed_sec    : total wall-clock seconds.
    """
    fold_results:          List[FoldResult]
    best_params:           Dict[str, Any]
    oos_sharpe:            float
    oos_cagr:              float
    oos_max_dd:            float
    param_stability_score: float
    sharpe_ci:             Tuple[float, float]
    oos_sharpe_series:     List[float]
    is_sharpe_series:      List[float]
    combined_oos_trades:   pd.DataFrame
    combined_oos_stats:    PerformanceStats
    n_folds:               int
    total_elapsed_sec:     float = 0.0

    @property
    def is_oos_sharpe_ratio(self) -> float:
        """IS/OOS Sharpe ratio. < 1 indicates good generalization."""
        mean_is = float(np.mean(self.is_sharpe_series)) if self.is_sharpe_series else 0.0
        if abs(self.oos_sharpe) < 1e-8:
            return np.inf if mean_is > 0 else 0.0
        return mean_is / (self.oos_sharpe + 1e-8)

    @property
    def sharpe_degradation(self) -> float:
        """IS Sharpe minus OOS Sharpe (lower = less overfitting)."""
        mean_is = float(np.mean(self.is_sharpe_series)) if self.is_sharpe_series else 0.0
        return mean_is - self.oos_sharpe


@dataclass
class CPCVResult:
    """
    Result from CPCV (Combinatorial Purged Cross-Validation) analysis.

    Attributes
    ----------
    path_sharpes              : OOS Sharpe for each backtest path.
    path_returns              : cumulative OOS return for each path.
    deflated_sharpe           : DSR-adjusted Sharpe accounting for N trials.
    probability_of_overfitting: PBO estimate in [0, 1].
    n_combinations            : C(k, k_test) combinations evaluated.
    n_paths                   : complete backtest paths reconstructed.
    is_sharpes                : IS Sharpe for each combination.
    oos_sharpes               : OOS Sharpe for each combination.
    fold_results              : all fold results across all combinations.
    """
    path_sharpes:               List[float]
    path_returns:               List[float]
    deflated_sharpe:            float
    probability_of_overfitting: float
    n_combinations:             int
    n_paths:                    int
    is_sharpes:                 List[float]
    oos_sharpes:                List[float]
    fold_results:               List[FoldResult]

    @property
    def mean_path_sharpe(self) -> float:
        return float(np.mean(self.path_sharpes)) if self.path_sharpes else 0.0

    @property
    def std_path_sharpe(self) -> float:
        return float(np.std(self.path_sharpes)) if len(self.path_sharpes) > 1 else 0.0

    @property
    def pbo_interpretation(self) -> str:
        pbo = self.probability_of_overfitting
        if pbo < 0.1:
            return "Very low overfitting risk"
        elif pbo < 0.25:
            return "Low overfitting risk"
        elif pbo < 0.5:
            return "Moderate overfitting risk"
        elif pbo < 0.75:
            return "High overfitting risk"
        else:
            return "Very high overfitting risk — likely curve-fit"


# ─────────────────────────────────────────────────────────────────────────────
# WalkForwardEngine
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Orchestrates walk-forward and CPCV analysis.

    The engine accepts a user-provided `strategy_fn` that takes
    (train_trades, params) → List[trade_dicts] and evaluates it across
    all folds defined by a splitter.

    Parameters
    ----------
    strategy_fn : callable(train_trades: pd.DataFrame, params: dict) → List[dict].
                  Must return a list of trade dicts representing out-of-sample
                  trades (each dict minimally has 'pnl', 'dollar_pos').
    param_grid  : dict mapping param_name → list of candidate values.
                  For each fold, all combinations are evaluated and the best
                  (by IS metric) is selected.
    n_jobs      : parallel workers (-1 = all CPUs, 1 = serial). Default -1.
    verbose     : log fold progress. Default True.
    metric      : IS selection metric ('sharpe', 'sortino', 'calmar', 'profit_factor').
    starting_equity : initial equity for performance calculation.

    Examples
    --------
    >>> engine = WalkForwardEngine(my_strategy, {'cf': [0.001, 0.002, 0.003]})
    >>> result = engine.run(trades_df, walk_forward_splits(n, 500, 100, 100))
    >>> print(f"OOS Sharpe: {result.oos_sharpe:.2f}")
    """

    def __init__(
        self,
        strategy_fn:     Callable,
        param_grid:      Dict[str, List[Any]],
        n_jobs:          int   = -1,
        verbose:         bool  = True,
        metric:          str   = "sharpe",
        starting_equity: float = 100_000.0,
    ) -> None:
        self.strategy_fn      = strategy_fn
        self.param_grid       = param_grid
        self.n_jobs           = n_jobs
        self.verbose          = verbose
        self.metric           = metric
        self.starting_equity  = starting_equity

        # Pre-expand all param combinations
        self._param_combos: List[Dict[str, Any]] = list(self._expand_grid(param_grid))
        logger.info(
            "WalkForwardEngine initialized: %d param combinations, metric='%s'",
            len(self._param_combos), metric,
        )

    # ------------------------------------------------------------------
    # Public: run walk-forward
    # ------------------------------------------------------------------

    def run(
        self,
        trades:   pd.DataFrame,
        splitter: List[WFSplit],
    ) -> WFResult:
        """
        Run full walk-forward analysis.

        For each fold:
        1. Extract train trades via split.train_idx.
        2. Grid-search over param_grid on IS data.
        3. Select best params by IS metric.
        4. Evaluate on OOS (test) fold.
        5. Aggregate results.

        Parameters
        ----------
        trades   : DataFrame of all trades with at least `pnl`, `dollar_pos`.
        splitter : list of WFSplit objects (from splits.py functions).

        Returns
        -------
        WFResult
        """
        if trades.empty:
            raise ValueError("trades DataFrame is empty")
        if not splitter:
            raise ValueError("splitter produced no folds")

        t0 = time.perf_counter()
        fold_results: List[FoldResult] = []

        if self.n_jobs == 1 or len(splitter) <= 2:
            # Serial execution
            for split in splitter:
                result = self._run_fold(trades, split)
                fold_results.append(result)
                if self.verbose:
                    status = f"OOS Sharpe={result.oos_sharpe:.3f}" if result.success else f"ERROR: {result.error}"
                    logger.info("Fold %d complete: %s", split.fold_id, status)
        else:
            # Parallel execution (thread-based to avoid pickle issues with strategy_fn)
            max_workers = self._resolve_n_jobs()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_fold, trades, split): split.fold_id
                    for split in splitter
                }
                for future in as_completed(futures):
                    fold_id = futures[future]
                    try:
                        result = future.result()
                        fold_results.append(result)
                        if self.verbose:
                            status = f"OOS Sharpe={result.oos_sharpe:.3f}" if result.success else f"ERROR: {result.error}"
                            logger.info("Fold %d complete: %s", fold_id, status)
                    except Exception as e:
                        logger.error("Fold %d raised exception: %s", fold_id, e)

            # Sort by fold_id
            fold_results.sort(key=lambda r: r.fold_id)

        elapsed = time.perf_counter() - t0
        wf_result = self._aggregate_results(fold_results, elapsed)

        logger.info(
            "WalkForward complete: %d folds in %.1fs | OOS Sharpe=%.3f | OOS MaxDD=%.2f%%",
            wf_result.n_folds, elapsed, wf_result.oos_sharpe, wf_result.oos_max_dd * 100,
        )
        return wf_result

    # ------------------------------------------------------------------
    # Public: run CPCV
    # ------------------------------------------------------------------

    def run_cpcv(
        self,
        trades:         pd.DataFrame,
        cpcv_splitter:  CPCVSplitter,
    ) -> CPCVResult:
        """
        Run Combinatorial Purged Cross-Validation analysis.

        Evaluates all C(k, k_test) combinations and reconstructs complete
        backtest paths for PBO estimation.

        Parameters
        ----------
        trades        : DataFrame of all trades.
        cpcv_splitter : CPCVSplitter instance.

        Returns
        -------
        CPCVResult
        """
        if trades.empty:
            raise ValueError("trades DataFrame is empty")

        n = len(trades)
        if self.verbose:
            logger.info(
                "Starting CPCV: %s", cpcv_splitter.summary()
            )

        t0 = time.perf_counter()

        all_fold_results: List[FoldResult] = []
        is_sharpes:       List[float]      = []
        oos_sharpes:      List[float]      = []

        # Iterate all C(k, k_test) combinations
        for combo_id, (train_idx, test_idx) in enumerate(cpcv_splitter.split(np.arange(n))):
            split = WFSplit(
                train_idx   = train_idx,
                test_idx    = test_idx,
                fold_id     = combo_id,
                train_start = int(train_idx.min()) if len(train_idx) > 0 else 0,
                train_end   = int(train_idx.max()) if len(train_idx) > 0 else 0,
                test_start  = int(test_idx.min())  if len(test_idx)  > 0 else 0,
                test_end    = int(test_idx.max())  if len(test_idx)  > 0 else 0,
            )

            result = self._run_fold(trades, split)
            all_fold_results.append(result)
            is_sharpes.append(result.is_stats.sharpe)
            oos_sharpes.append(result.oos_sharpe)

            if self.verbose:
                logger.info(
                    "CPCV combo %d/%d: IS Sharpe=%.3f, OOS Sharpe=%.3f",
                    combo_id + 1, cpcv_splitter.get_n_splits(),
                    result.is_stats.sharpe, result.oos_sharpe,
                )

        # Reconstruct backtest paths
        paths = cpcv_splitter.get_backtest_paths(n)
        path_test_idx_list = cpcv_splitter.get_path_test_indices(n)

        path_sharpes: List[float] = []
        path_returns: List[float] = []

        for path_blocks in path_test_idx_list:
            # Collect all OOS trades from this path
            all_oos_test_idx = np.concatenate(path_blocks)
            path_trades = trades.iloc[np.sort(all_oos_test_idx)]

            if len(path_trades) == 0:
                path_sharpes.append(0.0)
                path_returns.append(0.0)
                continue

            pstats = compute_performance_stats(
                path_trades,
                starting_equity=self.starting_equity,
            )
            path_sharpes.append(pstats.sharpe)
            path_returns.append(pstats.cagr_ann)

        # Compute Deflated Sharpe Ratio
        successful_oos = [s for s in oos_sharpes if np.isfinite(s)]
        n_obs_avg      = n // max(1, cpcv_splitter.n_test_splits)

        if successful_oos:
            best_sr = max(successful_oos)
            # Estimate skewness and kurtosis from all OOS returns
            all_oos_pnl: List[float] = []
            for fr in all_fold_results:
                if fr.oos_trades:
                    for t in fr.oos_trades:
                        pnl = t.get("pnl", 0.0) if isinstance(t, dict) else getattr(t, "pnl", 0.0)
                        all_oos_pnl.append(float(pnl))

            if len(all_oos_pnl) > 5:
                from scipy import stats as scipy_stats
                oos_skew = float(scipy_stats.skew(all_oos_pnl))
                oos_kurt = float(scipy_stats.kurtosis(all_oos_pnl))  # excess kurtosis
            else:
                oos_skew, oos_kurt = 0.0, 0.0

            n_trials = len(self._param_combos) * cpcv_splitter.n_splits

            dsr = deflated_sharpe_ratio(
                observed_sharpe = best_sr,
                n_trials        = max(1, n_trials),
                skewness        = oos_skew,
                kurtosis        = oos_kurt,
                n_obs           = n_obs_avg,
            )
        else:
            dsr = 0.0

        # PBO
        if len(is_sharpes) >= 4 and len(oos_sharpes) >= 4:
            pbo = probability_of_backtest_overfitting(
                np.array(is_sharpes),
                np.array(oos_sharpes),
            )
        else:
            pbo = np.nan

        elapsed = time.perf_counter() - t0

        cpcv_result = CPCVResult(
            path_sharpes               = path_sharpes,
            path_returns               = path_returns,
            deflated_sharpe            = dsr,
            probability_of_overfitting = pbo,
            n_combinations             = cpcv_splitter.get_n_splits(),
            n_paths                    = len(paths),
            is_sharpes                 = is_sharpes,
            oos_sharpes                = oos_sharpes,
            fold_results               = all_fold_results,
        )

        logger.info(
            "CPCV complete: %d combos, %d paths | DSR=%.3f | PBO=%.3f | elapsed=%.1fs",
            cpcv_result.n_combinations, cpcv_result.n_paths,
            dsr, pbo if np.isfinite(pbo) else -1.0, elapsed,
        )
        return cpcv_result

    # ------------------------------------------------------------------
    # Private: single fold execution
    # ------------------------------------------------------------------

    def _run_fold(
        self,
        trades: pd.DataFrame,
        split:  WFSplit,
    ) -> FoldResult:
        """
        Execute one fold: IS grid-search → best params → OOS evaluation.
        """
        t0 = time.perf_counter()

        try:
            train_trades = trades.iloc[split.train_idx].copy().reset_index(drop=True)
            test_trades  = trades.iloc[split.test_idx].copy().reset_index(drop=True)

            if len(train_trades) < 5:
                return FoldResult(
                    fold_id=split.fold_id, params={},
                    is_stats=PerformanceStats(), oos_stats=PerformanceStats(),
                    oos_trades=[], is_trades=[],
                    elapsed_sec=time.perf_counter()-t0,
                    error=f"Insufficient train trades: {len(train_trades)}",
                    split=split,
                )

            # IS grid search
            best_params, best_is_score = self._is_grid_search(train_trades)

            # IS performance under best params
            is_oos_trades = self._call_strategy(train_trades, best_params)
            is_stats      = self._trades_to_stats(is_oos_trades, train_trades)

            # OOS evaluation using best params
            oos_raw_trades = self._call_strategy(test_trades, best_params)
            oos_stats      = self._trades_to_stats(oos_raw_trades, test_trades)

            elapsed = time.perf_counter() - t0
            return FoldResult(
                fold_id     = split.fold_id,
                params      = best_params,
                is_stats    = is_stats,
                oos_stats   = oos_stats,
                oos_trades  = oos_raw_trades,
                is_trades   = is_oos_trades,
                elapsed_sec = elapsed,
                error       = None,
                split       = split,
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            tb      = traceback.format_exc()
            logger.error("Fold %d failed: %s\n%s", split.fold_id, e, tb)
            return FoldResult(
                fold_id=split.fold_id, params={},
                is_stats=PerformanceStats(), oos_stats=PerformanceStats(),
                oos_trades=[], is_trades=[],
                elapsed_sec=elapsed,
                error=str(e),
                split=split,
            )

    def _is_grid_search(
        self,
        train_trades: pd.DataFrame,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run IS grid search over all param combinations.

        Returns the best params and their IS score.
        """
        best_params = self._param_combos[0] if self._param_combos else {}
        best_score  = -np.inf

        for params in self._param_combos:
            try:
                is_trades = self._call_strategy(train_trades, params)
                if not is_trades:
                    continue
                score = self._score_trades(is_trades, train_trades, self.metric)
                if np.isfinite(score) and score > best_score:
                    best_score  = score
                    best_params = params
            except Exception as e:
                logger.debug("IS grid search param=%s failed: %s", params, e)
                continue

        return best_params, best_score

    def _call_strategy(
        self,
        trades: pd.DataFrame,
        params: Dict[str, Any],
    ) -> List[Dict]:
        """
        Call strategy_fn with a copy of trades and given params.

        Returns a list of trade dicts. strategy_fn may return:
          - List[dict]
          - pd.DataFrame  (auto-converted)
          - List[Any]     (attrs extracted via duck-typing)
        """
        result = self.strategy_fn(trades, params)

        if result is None:
            return []

        if isinstance(result, pd.DataFrame):
            return result.to_dict(orient="records")

        if isinstance(result, (list, tuple)):
            out: List[Dict] = []
            for item in result:
                if isinstance(item, dict):
                    out.append(item)
                else:
                    # Duck-type: extract known attrs
                    d: Dict = {}
                    for attr in ("pnl", "dollar_pos", "entry_price", "exit_price",
                                 "hold_bars", "regime", "sym", "exit_time"):
                        val = getattr(item, attr, None)
                        if val is not None:
                            d[attr] = val
                    out.append(d)
            return out

        return []

    def _score_trades(
        self,
        trades:      List[Dict],
        base_trades: pd.DataFrame,
        metric:      str,
    ) -> float:
        """Compute scalar score from trade list using the selected metric."""
        if not trades:
            return -np.inf

        pnl_arr = np.array([t.get("pnl", 0.0) for t in trades], dtype=np.float64)
        pnl_arr = pnl_arr[np.isfinite(pnl_arr)]

        if len(pnl_arr) == 0:
            return -np.inf

        if metric == "sharpe":
            # Compute returns for Sharpe
            pos_arr = np.array([t.get("dollar_pos", 1.0) for t in trades], dtype=np.float64)
            pos_arr = np.where(np.abs(pos_arr) < 1e-6, 1.0, pos_arr)
            ret_arr = pnl_arr / pos_arr
            return sharpe_ratio(ret_arr)

        elif metric == "sortino":
            pos_arr = np.array([t.get("dollar_pos", 1.0) for t in trades], dtype=np.float64)
            pos_arr = np.where(np.abs(pos_arr) < 1e-6, 1.0, pos_arr)
            ret_arr = pnl_arr / pos_arr
            return sortino_ratio(ret_arr)

        elif metric == "calmar":
            eq = self.starting_equity + np.cumsum(pnl_arr)
            eq = np.concatenate([[self.starting_equity], eq])
            pos_arr = np.array([t.get("dollar_pos", 1.0) for t in trades], dtype=np.float64)
            pos_arr = np.where(np.abs(pos_arr) < 1e-6, 1.0, pos_arr)
            ret_arr = pnl_arr / pos_arr
            mdd     = max_drawdown(eq)
            return calmar_ratio(ret_arr, max_dd=mdd) if abs(mdd) > 1e-8 else float(np.mean(ret_arr))

        elif metric == "profit_factor":
            return profit_factor(pnl_arr)

        elif metric == "total_pnl":
            return float(np.sum(pnl_arr))

        elif metric == "win_rate":
            return win_rate(pnl_arr)

        else:
            raise ValueError(f"Unknown metric: '{metric}'")

    def _trades_to_stats(
        self,
        trades_list: List[Dict],
        fallback_df: pd.DataFrame,
    ) -> PerformanceStats:
        """Convert a list of trade dicts to PerformanceStats."""
        if not trades_list:
            return PerformanceStats()

        try:
            df = pd.DataFrame(trades_list)
            return compute_performance_stats(df, starting_equity=self.starting_equity)
        except Exception:
            return PerformanceStats()

    # ------------------------------------------------------------------
    # Private: result aggregation
    # ------------------------------------------------------------------

    def _aggregate_results(
        self,
        fold_results: List[FoldResult],
        elapsed:      float,
    ) -> WFResult:
        """Aggregate fold results into WFResult."""
        successful = [fr for fr in fold_results if fr.success]

        if not successful:
            logger.error("All folds failed!")
            empty = PerformanceStats()
            return WFResult(
                fold_results=fold_results, best_params={},
                oos_sharpe=0.0, oos_cagr=0.0, oos_max_dd=0.0,
                param_stability_score=0.0, sharpe_ci=(0.0, 0.0),
                oos_sharpe_series=[], is_sharpe_series=[],
                combined_oos_trades=pd.DataFrame(), combined_oos_stats=empty,
                n_folds=0, total_elapsed_sec=elapsed,
            )

        oos_sharpes = [fr.oos_sharpe      for fr in successful]
        is_sharpes  = [fr.is_stats.sharpe  for fr in successful]
        oos_cagrs   = [fr.oos_cagr         for fr in successful]
        oos_mdd     = [fr.oos_max_dd        for fr in successful]

        # Best params by frequency (modal selection)
        param_counts: Dict[str, int] = {}
        for fr in successful:
            key = str(sorted(fr.params.items()))
            param_counts[key] = param_counts.get(key, 0) + 1

        best_param_key = max(param_counts, key=lambda k: param_counts[k])
        best_params_fold = next(
            fr.params for fr in successful if str(sorted(fr.params.items())) == best_param_key
        )
        stability_score = param_counts[best_param_key] / len(successful)

        # Bootstrap CI for OOS Sharpe
        oos_sharpe_arr = np.array(oos_sharpes)
        if len(oos_sharpe_arr) >= 5:
            sharpe_ci = bootstrap_confidence_interval(
                lambda x: float(np.mean(x)),
                oos_sharpe_arr,
                n_boot=500,
            )
        else:
            m = float(np.mean(oos_sharpe_arr))
            sharpe_ci = (m, m)

        # Combine all OOS trades into one DataFrame
        all_oos_dicts: List[Dict] = []
        for fr in successful:
            all_oos_dicts.extend(fr.oos_trades)

        if all_oos_dicts:
            combined_df = pd.DataFrame(all_oos_dicts)
            # Sort by exit_time if available
            if "exit_time" in combined_df.columns:
                combined_df = combined_df.sort_values("exit_time").reset_index(drop=True)
            combined_stats = compute_performance_stats(combined_df, starting_equity=self.starting_equity)
        else:
            combined_df    = pd.DataFrame()
            combined_stats = PerformanceStats()

        return WFResult(
            fold_results          = fold_results,
            best_params           = best_params_fold,
            oos_sharpe            = float(np.mean(oos_sharpes)),
            oos_cagr              = float(np.mean(oos_cagrs)),
            oos_max_dd            = float(min(oos_mdd)) if oos_mdd else 0.0,
            param_stability_score = stability_score,
            sharpe_ci             = sharpe_ci,
            oos_sharpe_series     = oos_sharpes,
            is_sharpe_series      = is_sharpes,
            combined_oos_trades   = combined_df,
            combined_oos_stats    = combined_stats,
            n_folds               = len(successful),
            total_elapsed_sec     = elapsed,
        )

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    def _resolve_n_jobs(self) -> int:
        """Convert n_jobs=-1 to actual CPU count."""
        import os
        if self.n_jobs == -1:
            return max(1, (os.cpu_count() or 4) - 1)
        return max(1, self.n_jobs)

    @staticmethod
    def _expand_grid(param_grid: Dict[str, List[Any]]) -> Generator[Dict[str, Any], None, None]:
        """
        Expand a parameter grid dict into all combinations.

        Yields dicts like {'cf': 0.001, 'bh_form': 1.5, ...}.
        """
        import itertools
        keys   = list(param_grid.keys())
        values = [param_grid[k] for k in keys]

        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: evaluate_strategy_on_fold
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_strategy_on_fold(
    strategy_fn:     Callable,
    params:          Dict[str, Any],
    trades:          pd.DataFrame,
    split:           WFSplit,
    starting_equity: float = 100_000.0,
) -> FoldResult:
    """
    Evaluate a strategy with given params on a single fold.

    Utility function for ad-hoc fold evaluation outside the full engine.

    Parameters
    ----------
    strategy_fn     : callable(train_trades, params) → List[dict].
    params          : parameter dict.
    trades          : full trades DataFrame.
    split           : WFSplit defining train/test split.
    starting_equity : initial equity.

    Returns
    -------
    FoldResult
    """
    t0 = time.perf_counter()
    try:
        train = trades.iloc[split.train_idx].copy().reset_index(drop=True)
        test  = trades.iloc[split.test_idx].copy().reset_index(drop=True)

        raw_oos = strategy_fn(test, params)
        if isinstance(raw_oos, pd.DataFrame):
            oos_dicts = raw_oos.to_dict(orient="records")
        elif isinstance(raw_oos, (list, tuple)):
            oos_dicts = [t if isinstance(t, dict) else vars(t) for t in raw_oos]
        else:
            oos_dicts = []

        oos_df    = pd.DataFrame(oos_dicts) if oos_dicts else pd.DataFrame()
        oos_stats = compute_performance_stats(oos_df, starting_equity) if not oos_df.empty else PerformanceStats()

        raw_is = strategy_fn(train, params)
        if isinstance(raw_is, pd.DataFrame):
            is_dicts = raw_is.to_dict(orient="records")
        elif isinstance(raw_is, (list, tuple)):
            is_dicts = [t if isinstance(t, dict) else vars(t) for t in raw_is]
        else:
            is_dicts = []

        is_df    = pd.DataFrame(is_dicts) if is_dicts else pd.DataFrame()
        is_stats = compute_performance_stats(is_df, starting_equity) if not is_df.empty else PerformanceStats()

        return FoldResult(
            fold_id     = split.fold_id,
            params      = params,
            is_stats    = is_stats,
            oos_stats   = oos_stats,
            oos_trades  = oos_dicts,
            is_trades   = is_dicts,
            elapsed_sec = time.perf_counter() - t0,
            error       = None,
            split       = split,
        )
    except Exception as e:
        return FoldResult(
            fold_id=split.fold_id, params=params,
            is_stats=PerformanceStats(), oos_stats=PerformanceStats(),
            oos_trades=[], is_trades=[],
            elapsed_sec=time.perf_counter()-t0,
            error=str(e),
            split=split,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Combine OOS equity curves from multiple folds
# ─────────────────────────────────────────────────────────────────────────────

def stitch_oos_equity_curves(
    fold_results:    List[FoldResult],
    starting_equity: float = 100_000.0,
) -> pd.Series:
    """
    Stitch together fold OOS equity curves into a single equity curve.

    Each fold's OOS equity starts where the previous fold ended, producing
    a continuous portfolio equity curve across all out-of-sample periods.

    Parameters
    ----------
    fold_results    : list of FoldResult objects.
    starting_equity : initial equity.

    Returns
    -------
    pd.Series of equity values indexed by trade position.
    """
    equity = starting_equity
    all_equity: List[float] = [equity]

    for fr in sorted(fold_results, key=lambda r: r.fold_id):
        if not fr.success or not fr.oos_trades:
            continue
        for trade in fr.oos_trades:
            pnl    = float(trade.get("pnl", 0.0)) if isinstance(trade, dict) else float(getattr(trade, "pnl", 0.0))
            equity += pnl
            all_equity.append(equity)

    return pd.Series(all_equity, name="oos_equity")


# ─────────────────────────────────────────────────────────────────────────────
# IS/OOS Degradation Summary
# ─────────────────────────────────────────────────────────────────────────────

def is_oos_degradation_summary(wf_result: WFResult) -> pd.DataFrame:
    """
    Build a per-fold IS vs OOS comparison DataFrame.

    Parameters
    ----------
    wf_result : WFResult from WalkForwardEngine.run().

    Returns
    -------
    DataFrame with columns: fold_id, is_sharpe, oos_sharpe, degradation,
    is_max_dd, oos_max_dd, is_cagr, oos_cagr, params.
    """
    rows = []
    for fr in wf_result.fold_results:
        if not fr.success:
            continue
        rows.append({
            "fold_id":     fr.fold_id,
            "is_sharpe":   fr.is_stats.sharpe,
            "oos_sharpe":  fr.oos_sharpe,
            "degradation": fr.is_stats.sharpe - fr.oos_sharpe,
            "is_max_dd":   fr.is_stats.max_dd,
            "oos_max_dd":  fr.oos_max_dd,
            "is_cagr":     fr.is_stats.cagr_ann,
            "oos_cagr":    fr.oos_cagr,
            "params":      str(fr.params),
            "n_oos_trades": len(fr.oos_trades),
        })
    return pd.DataFrame(rows)
