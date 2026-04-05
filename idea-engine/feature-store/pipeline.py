"""
idea-engine/feature-store/pipeline.py
========================================
FeaturePipeline: orchestrates signal computation across all symbols and
manages the feature store cache.

The pipeline is the main entry point for bulk feature computation. It:
  1. Registers which signals to compute.
  2. Runs them across all symbols, writing results to the FeatureStore.
  3. Supports incremental updates for new bars.
  4. Provides analysis utilities: feature IC ranking, correlation matrix,
     and PCA dimensionality reduction.

Usage
-----
    from feature_store.pipeline import FeaturePipeline
    from signal_library import RSI, MACD, GARCHVolForecast

    pipeline = FeaturePipeline("idea_engine.db")
    pipeline.add_signal(RSI(period=14))
    pipeline.add_signal(MACD())
    pipeline.add_signal(GARCHVolForecast())

    df_dict = {"BTC": btc_df, "ETH": eth_df}
    results = pipeline.run(df_dict)

    # Incremental update (only new bars)
    pipeline.run_incremental({"BTC": new_bars_df})

    # Feature IC ranking
    rankings = pipeline.feature_importance("forward_return_1h")

    # PCA reduction
    reduced = pipeline.pca_reduction(feature_matrix, n_components=10)
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .store import FeatureStore
from .ic_tracker import ICTracker, compute_ic, rolling_ic, icir

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class PipelineResult:
    """
    Container for a pipeline run result.

    Attributes
    ----------
    n_success : int
    n_failed : int
    failed : list[(symbol, signal_name)]
    duration_seconds : float
    """

    def __init__(
        self,
        n_success:        int,
        n_failed:         int,
        failed:           List[Tuple[str, str]],
        duration_seconds: float,
    ) -> None:
        self.n_success        = n_success
        self.n_failed         = n_failed
        self.failed           = failed
        self.duration_seconds = duration_seconds

    def __repr__(self) -> str:
        return (
            f"PipelineResult("
            f"n_success={self.n_success}, n_failed={self.n_failed}, "
            f"duration={self.duration_seconds:.1f}s)"
        )

    @property
    def success_rate(self) -> float:
        total = self.n_success + self.n_failed
        return self.n_success / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    Orchestrates feature computation across multiple symbols and signals.

    Parameters
    ----------
    db_path : str | Path
    parquet_dir : str | Path | None
    max_workers : int
        Thread pool workers for parallel computation.
    ic_forward_bars_list : list[int]
        Forward-return horizons for automatic IC tracking.
    ic_window : int
        Rolling IC window (bars).
    auto_track_ic : bool
        If True, IC is computed and stored after each pipeline run.
    """

    def __init__(
        self,
        db_path:              Union[str, Path]            = DEFAULT_DB_PATH,
        parquet_dir:          Optional[Union[str, Path]]  = None,
        max_workers:          int                          = 4,
        ic_forward_bars_list: Optional[List[int]]          = None,
        ic_window:            int                          = 60,
        auto_track_ic:        bool                         = False,
    ) -> None:
        self.store       = FeatureStore(db_path, parquet_dir, max_workers)
        self.ic_tracker  = ICTracker(db_path)
        self.max_workers = max_workers
        self.ic_forward_bars_list = ic_forward_bars_list or [1, 5, 20]
        self.ic_window   = ic_window
        self.auto_track_ic = auto_track_ic

        self._signals: List[Any] = []      # registered Signal instances
        self._signal_map: Dict[str, Any] = {}

    # ── Signal registration ───────────────────────────────────────────────

    def add_signal(self, signal: Any) -> "FeaturePipeline":
        """
        Register a Signal instance for computation.

        Returns self for chaining:
            pipeline.add_signal(RSI()).add_signal(MACD())
        """
        name = getattr(signal, "name", None)
        if not name:
            raise ValueError(
                f"Signal {signal!r} has no .name attribute. "
                "Only concrete Signal instances can be registered."
            )
        if name in self._signal_map:
            logger.warning(
                "Pipeline: signal '%s' is already registered. Replacing.", name
            )
        self._signals    = [s for s in self._signals if s.name != name]
        self._signals.append(signal)
        self._signal_map[name] = signal
        return self

    def add_signals(self, signals: List[Any]) -> "FeaturePipeline":
        """Register multiple signals at once."""
        for s in signals:
            self.add_signal(s)
        return self

    def remove_signal(self, name: str) -> None:
        """Deregister a signal by name."""
        self._signals    = [s for s in self._signals if s.name != name]
        self._signal_map.pop(name, None)

    def list_signals(self) -> List[str]:
        """Return names of all registered signals."""
        return [s.name for s in self._signals]

    # ── Full pipeline run ─────────────────────────────────────────────────

    def run(
        self,
        df_dict:    Dict[str, pd.DataFrame],
        symbols:    Optional[List[str]] = None,
        signal_names: Optional[List[str]] = None,
        overwrite:  bool = True,
    ) -> PipelineResult:
        """
        Compute all registered signals for all symbols.

        Parameters
        ----------
        df_dict : dict[symbol, DataFrame]
            OHLCV DataFrames keyed by symbol.
        symbols : list[str] | None
            Subset of symbols to compute. If None, uses all keys in df_dict.
        signal_names : list[str] | None
            Subset of signals to compute. If None, uses all registered signals.
        overwrite : bool
            If True, overwrite existing cached values.

        Returns
        -------
        PipelineResult
        """
        import time
        t0 = time.perf_counter()

        syms = symbols or list(df_dict.keys())
        sigs = (
            [self._signal_map[n] for n in signal_names if n in self._signal_map]
            if signal_names
            else self._signals
        )

        if not sigs:
            logger.warning("Pipeline.run: no signals registered. Call add_signal() first.")
            return PipelineResult(0, 0, [], 0.0)

        results = self.store.bulk_compute(
            symbols=syms,
            signal_instances=sigs,
            df_dict=df_dict,
            overwrite=overwrite,
            max_workers=self.max_workers,
        )

        n_ok    = sum(results.values())
        failed  = [(sym, sig) for (sym, sig), ok in results.items() if not ok]

        if self.auto_track_ic:
            self._auto_ic(df_dict, syms, sigs)

        duration = time.perf_counter() - t0
        logger.info(
            "Pipeline.run: %d/%d tasks OK in %.1fs.",
            n_ok, len(results), duration
        )
        return PipelineResult(
            n_success=n_ok,
            n_failed=len(failed),
            failed=failed,
            duration_seconds=duration,
        )

    def run_incremental(
        self,
        new_bars: Dict[str, pd.DataFrame],
        signal_names: Optional[List[str]] = None,
    ) -> PipelineResult:
        """
        Update features for new bars only.

        Only the bars present in new_bars are computed and appended to the
        cache. No overwrite of existing values.

        Parameters
        ----------
        new_bars : dict[symbol, DataFrame]
            DataFrames containing ONLY the new bars to process.
            IMPORTANT: Each DataFrame must include at least ``lookback`` prior
            bars for warmup, but only the truly new bars will be persisted.
        signal_names : list[str] | None

        Returns
        -------
        PipelineResult
        """
        return self.run(
            df_dict=new_bars,
            signal_names=signal_names,
            overwrite=False,   # do not overwrite existing rows
        )

    # ── Feature analysis ──────────────────────────────────────────────────

    def feature_importance(
        self,
        target_col:   str,
        df_dict:      Dict[str, pd.DataFrame],
        method:       str = "ic",
        forward_bars: int = 1,
        ic_window:    Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rank signals by their predictive power (IC) with a forward return target.

        Parameters
        ----------
        target_col : str
            Name of the target column in df_dict DataFrames, OR a special value:
              - 'forward_return_N' : uses N-bar forward log return.
        df_dict : dict[symbol, DataFrame]
        method : 'ic' (Spearman IC) or 'pearson_ic'
        forward_bars : int
            If target_col is 'forward_return_N', use N = forward_bars.
        ic_window : int | None
            If None, uses instance default.

        Returns
        -------
        pd.DataFrame with columns:
            signal_name, mean_ic, icir, t_stat, n_obs
            sorted descending by icir.
        """
        window = ic_window or self.ic_window
        ic_method = "spearman" if method in ("ic", "spearman") else "pearson"

        all_ic: Dict[str, List[float]] = {s.name: [] for s in self._signals}

        for sym, df in df_dict.items():
            if df.empty or "Close" not in df.columns:
                continue

            # Build forward-return series
            if target_col.startswith("forward_return"):
                fwd = np.log(df["Close"].shift(-forward_bars) / df["Close"]).values
            elif target_col in df.columns:
                fwd = df[target_col].values
            else:
                logger.warning(
                    "feature_importance: target_col '%s' not in df for symbol '%s'.",
                    target_col, sym
                )
                continue

            for sig in self._signals:
                try:
                    cached = self.store.get(sym, sig.name)
                    if cached.empty:
                        # Compute fresh
                        cached = sig.compute(df)
                    sig_vals = cached.values
                    if len(sig_vals) != len(fwd):
                        # Align
                        min_len  = min(len(sig_vals), len(fwd))
                        sig_vals = sig_vals[-min_len:]
                        fwd_trim = fwd[-min_len:]
                    else:
                        fwd_trim = fwd

                    roll = rolling_ic(
                        sig_vals, fwd_trim,
                        window=window, method=ic_method
                    )
                    valid = roll.dropna().values
                    if len(valid) > 0:
                        all_ic[sig.name].extend(valid.tolist())
                except Exception as exc:
                    logger.warning(
                        "feature_importance: skipping %s/%s: %s",
                        sym, sig.name, exc
                    )

        rows = []
        for sig_name, ic_vals in all_ic.items():
            vals = np.array(ic_vals, dtype=float)
            vals = vals[~np.isnan(vals)]
            n    = len(vals)
            rows.append({
                "signal_name": sig_name,
                "mean_ic":     float(np.mean(vals)) if n > 0 else np.nan,
                "icir":        icir(vals),
                "t_stat":      self.ic_tracker.t_stat(vals),
                "n_obs":       n,
            })

        result = (
            pd.DataFrame(rows)
            .sort_values("icir", ascending=False)
            .reset_index(drop=True)
        )
        return result

    def correlation_matrix(
        self,
        feature_names:  Optional[List[str]] = None,
        symbols:        Optional[List[str]] = None,
        df_dict:        Optional[Dict[str, pd.DataFrame]] = None,
        method:         str = "pearson",
        start_ts:       Optional[Any] = None,
        end_ts:         Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Compute the pairwise correlation matrix of feature signals.

        Features are pulled from the FeatureStore cache. If df_dict is
        provided, signals are computed fresh and not cached.

        Parameters
        ----------
        feature_names : list[str] | None
            Signal names to include. Defaults to all registered signals.
        symbols : list[str] | None
            Symbols to include. Stacks all symbol data vertically.
        df_dict : dict | None
            If provided, compute signals fresh from these DataFrames.
        method : 'pearson' | 'spearman' | 'kendall'
        start_ts, end_ts : optional timestamp bounds for cache retrieval.

        Returns
        -------
        pd.DataFrame: square correlation matrix.
        """
        names = feature_names or self.list_signals()
        if not names:
            return pd.DataFrame()

        all_frames: List[pd.DataFrame] = []

        if df_dict:
            syms = symbols or list(df_dict.keys())
            for sym in syms:
                df = df_dict[sym]
                frame_cols: Dict[str, pd.Series] = {}
                for sig_name in names:
                    sig = self._signal_map.get(sig_name)
                    if sig is None:
                        continue
                    try:
                        frame_cols[sig_name] = sig.compute(df)
                    except Exception:
                        pass
                if frame_cols:
                    all_frames.append(pd.DataFrame(frame_cols))
        else:
            syms = symbols or self.store.list_symbols()
            for sym in syms:
                mat = self.store.get_feature_matrix(
                    symbols=[sym], signal_names=names,
                    start_ts=start_ts, end_ts=end_ts
                )
                if not mat.empty:
                    all_frames.append(mat)

        if not all_frames:
            return pd.DataFrame(index=names, columns=names, dtype=float)

        combined = pd.concat(all_frames, axis=0).dropna(how="all")
        # Keep only columns in ``names`` that exist
        cols = [c for c in names if c in combined.columns]
        corr = combined[cols].corr(method=method)
        return corr

    def pca_reduction(
        self,
        feature_matrix: Optional[pd.DataFrame] = None,
        n_components:   int                    = 10,
        symbols:        Optional[List[str]]    = None,
        signal_names:   Optional[List[str]]    = None,
        start_ts:       Optional[Any]          = None,
        end_ts:         Optional[Any]          = None,
        return_loadings: bool                  = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Reduce feature dimensionality using PCA.

        Returns a DataFrame of principal components (rows = bars,
        cols = PC1 … PCn). The principal components are uncorrelated.

        Parameters
        ----------
        feature_matrix : pd.DataFrame | None
            Pre-built feature matrix. If None, fetched from FeatureStore.
        n_components : int
            Number of principal components to retain.
        symbols : list[str] | None
            Used when feature_matrix is None.
        signal_names : list[str] | None
            Used when feature_matrix is None.
        start_ts, end_ts : optional timestamp bounds.
        return_loadings : bool
            If True, also return the loadings DataFrame
            (PC loadings, shape n_signals × n_components).

        Returns
        -------
        pd.DataFrame of PC scores, or (pd.DataFrame, loadings_DataFrame)
        if return_loadings=True.
        """
        if feature_matrix is None:
            syms  = symbols or self.store.list_symbols()
            sigs  = signal_names or self.list_signals()
            feature_matrix = self.store.get_feature_matrix(
                symbols=syms, signal_names=sigs,
                start_ts=start_ts, end_ts=end_ts
            )

        if feature_matrix.empty:
            logger.warning("pca_reduction: feature_matrix is empty.")
            return pd.DataFrame() if not return_loadings else (pd.DataFrame(), pd.DataFrame())

        # Drop columns with all NaN, fill remaining NaN with column mean
        fm = feature_matrix.dropna(axis=1, how="all")
        fm = fm.fillna(fm.mean())

        n_comp = min(n_components, fm.shape[1], fm.shape[0])

        # Centre and scale
        means  = fm.mean()
        stds   = fm.std().replace(0.0, 1.0)
        fm_std = (fm - means) / stds

        # SVD-based PCA
        try:
            from numpy.linalg import svd
            U, S, Vt = svd(fm_std.values, full_matrices=False)
        except Exception as exc:
            logger.error("pca_reduction: SVD failed: %s", exc)
            return pd.DataFrame() if not return_loadings else (pd.DataFrame(), pd.DataFrame())

        # Scores: rows = observations, cols = PCs
        scores      = pd.DataFrame(
            U[:, :n_comp] * S[:n_comp],
            index=fm.index,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        )

        # Loadings: rows = original features, cols = PCs
        loadings    = pd.DataFrame(
            Vt[:n_comp, :].T,
            index=fm.columns,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        )

        if return_loadings:
            return scores, loadings
        return scores

    # ── IC auto-tracking ──────────────────────────────────────────────────

    def _auto_ic(
        self,
        df_dict:  Dict[str, pd.DataFrame],
        symbols:  List[str],
        signals:  List[Any],
    ) -> None:
        """
        Automatically compute and store IC for all signal/symbol pairs after a run.
        """
        for sym in symbols:
            df = df_dict.get(sym)
            if df is None or "Close" not in df.columns:
                continue
            close = df["Close"].values
            for sig in signals:
                try:
                    cached = self.store.get(sym, sig.name)
                    if cached.empty:
                        sig_vals = sig.compute(df)
                    else:
                        sig_vals = cached
                    self.ic_tracker.compute_and_store_full_ic(
                        signal_name=sig.name,
                        symbol=sym,
                        signal_values=sig_vals,
                        close_prices=close,
                        forward_bars_list=self.ic_forward_bars_list,
                        window=self.ic_window,
                    )
                except Exception as exc:
                    logger.debug("_auto_ic: skipping %s/%s: %s", sym, sig.name, exc)

    # ── Convenience wrappers ──────────────────────────────────────────────

    def get_matrix(
        self,
        symbols:      Sequence[str],
        signal_names: Optional[Sequence[str]] = None,
        start_ts:     Optional[Any]           = None,
        end_ts:       Optional[Any]           = None,
    ) -> pd.DataFrame:
        """
        Retrieve a feature matrix from the store.
        Shorthand for store.get_feature_matrix().
        """
        sigs = signal_names or self.list_signals()
        return self.store.get_feature_matrix(
            symbols=symbols,
            signal_names=sigs,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    def top_signals(
        self,
        n:            int  = 10,
        forward_bars: int  = 5,
        min_obs:      int  = 20,
    ) -> pd.DataFrame:
        """
        Return the top-N signals by ICIR from the IC tracker.

        Shorthand for ic_tracker.top_signals_by_icir().
        """
        return self.ic_tracker.top_signals_by_icir(
            n=n, forward_bars=forward_bars, min_obs=min_obs
        )

    def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cached features for a symbol."""
        return self.store.invalidate(symbol)

    def staleness_report(self, **kwargs: Any) -> pd.DataFrame:
        """Forward to store.staleness_report()."""
        return self.store.staleness_report(**kwargs)

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary of the pipeline state.
        """
        stats = self.store.cache_stats()
        n_rows  = int(stats["n_rows"].sum()) if not stats.empty else 0
        n_syms  = len(stats["symbol"].unique()) if not stats.empty else 0
        n_sigs  = len(stats["signal_name"].unique()) if not stats.empty else 0
        return {
            "registered_signals": self.list_signals(),
            "n_registered":       len(self._signals),
            "cached_symbols":     n_syms,
            "cached_signals":     n_sigs,
            "total_cached_rows":  n_rows,
            "auto_track_ic":      self.auto_track_ic,
            "ic_forward_bars":    self.ic_forward_bars_list,
        }

    def __repr__(self) -> str:
        return (
            f"<FeaturePipeline "
            f"n_signals={len(self._signals)} "
            f"db={self.store.db_path.name!r}>"
        )
