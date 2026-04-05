"""
idea-engine/feature-store/ic_tracker.py
=========================================
Information Coefficient (IC) tracking for the SRFM Idea Engine Feature Store.

The IC measures the predictive power of a signal: the rank correlation between
signal values at time t and forward returns at time t+N.

Classes / Functions
--------------------
ICTracker       — computes, stores, and retrieves IC statistics
compute_ic()    — standalone Spearman or Pearson IC computation
rolling_ic()    — rolling-window IC time series
ic_decay()      — IC as a function of forward-return lag
icir()          — IC information ratio (mean IC / std IC)
t_stat()        — t-statistic testing IC > 0

IC history is persisted in the ``ic_history`` table of the SQLite database
and the aggregate stats are written back to ``feature_metadata``.

Usage
-----
    from feature_store.ic_tracker import ICTracker

    tracker = ICTracker("idea_engine.db")

    # Compute and store IC for a signal
    ic_val = tracker.compute_ic(signal_values, forward_returns)

    # Rolling IC time series
    rolling = tracker.rolling_ic(signal_values, forward_returns, window=60)

    # IC decay across lags
    decay = tracker.ic_decay(signal_values, returns_df, max_lag=20)

    # Top signals by ICIR
    top = tracker.top_signals_by_icir(n=10)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")


# ---------------------------------------------------------------------------
# Standalone IC functions
# ---------------------------------------------------------------------------

def compute_ic(
    signal_values:   Union[pd.Series, np.ndarray],
    forward_returns: Union[pd.Series, np.ndarray],
    method:          str = "spearman",
) -> float:
    """
    Compute the Information Coefficient between signal values and forward returns.

    Parameters
    ----------
    signal_values : array-like
        Signal values at time t.
    forward_returns : array-like
        Forward returns at time t+N. Must be same length as signal_values.
    method : 'spearman' | 'pearson'

    Returns
    -------
    float
        IC value in [-1, +1]. Returns 0.0 if insufficient data.
    """
    sig = np.asarray(signal_values, dtype=float)
    fwd = np.asarray(forward_returns, dtype=float)

    if len(sig) != len(fwd):
        raise ValueError(
            f"signal_values (n={len(sig)}) and forward_returns (n={len(fwd)}) "
            "must have the same length."
        )

    valid = ~(np.isnan(sig) | np.isnan(fwd))
    n     = valid.sum()
    if n < 5:
        return 0.0

    sig_v = sig[valid]
    fwd_v = fwd[valid]

    try:
        if method == "spearman":
            ic, _ = sp_stats.spearmanr(sig_v, fwd_v)
        elif method == "pearson":
            ic, _ = sp_stats.pearsonr(sig_v, fwd_v)
        else:
            raise ValueError(f"Unknown IC method: '{method}'. Use 'spearman' or 'pearson'.")
        return float(ic) if not np.isnan(ic) else 0.0
    except Exception as exc:
        logger.warning("compute_ic: failed with %s", exc)
        return 0.0


def rolling_ic(
    signal_values:   Union[pd.Series, np.ndarray],
    forward_returns: Union[pd.Series, np.ndarray],
    window:          int = 60,
    method:          str = "spearman",
    min_obs:         int = 10,
) -> pd.Series:
    """
    Compute a rolling IC time series.

    Parameters
    ----------
    signal_values : array-like or pd.Series
    forward_returns : array-like or pd.Series
    window : int
        Rolling window (bars).
    method : 'spearman' | 'pearson'
    min_obs : int
        Minimum valid observations per window to compute IC.

    Returns
    -------
    pd.Series of IC values, indexed like signal_values if it is a pd.Series.
    """
    sig = np.asarray(signal_values, dtype=float)
    fwd = np.asarray(forward_returns, dtype=float)
    n   = len(sig)

    if n != len(fwd):
        raise ValueError("signal_values and forward_returns must have the same length.")

    ic_vals = np.full(n, np.nan)

    for i in range(window - 1, n):
        s_win = sig[i - window + 1: i + 1]
        f_win = fwd[i - window + 1: i + 1]
        valid = ~(np.isnan(s_win) | np.isnan(f_win))
        if valid.sum() < min_obs:
            continue
        ic_vals[i] = compute_ic(s_win[valid], f_win[valid], method=method)

    index = (signal_values.index
             if isinstance(signal_values, pd.Series)
             else pd.RangeIndex(n))
    return pd.Series(ic_vals, index=index, name="rolling_ic")


def ic_decay(
    signal_values:   Union[pd.Series, np.ndarray],
    close_prices:    Union[pd.Series, np.ndarray],
    max_lag:         int = 20,
    method:          str = "spearman",
    min_obs:         int = 20,
) -> Dict[int, float]:
    """
    Compute IC at different forward-return horizons (lags).

    Parameters
    ----------
    signal_values : array-like
        Signal values.
    close_prices : array-like
        Close price series of the same length as signal_values.
    max_lag : int
        Maximum forward lag (bars) to test.
    method : 'spearman' | 'pearson'
    min_obs : int
        Minimum valid observations per lag.

    Returns
    -------
    dict[lag_bars, IC_value]
    """
    sig   = np.asarray(signal_values, dtype=float)
    close = np.asarray(close_prices,  dtype=float)

    if len(sig) != len(close):
        raise ValueError("signal_values and close_prices must have the same length.")

    decay_dict: Dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        log_ret  = np.full(len(close), np.nan)
        log_ret[lag:] = np.log(close[lag:] / close[:-lag])
        ic = compute_ic(sig, log_ret, method=method)
        decay_dict[lag] = ic

    return decay_dict


def icir(ic_series: Union[pd.Series, np.ndarray]) -> float:
    """
    IC Information Ratio: mean(IC) / std(IC).

    A higher ICIR indicates more consistent predictive power.
    Convention: annualise by multiplying by sqrt(12) for monthly IC, etc.
    (here we return the raw ratio without annualisation).

    Returns
    -------
    float (0.0 if std is zero or too few observations)
    """
    vals = np.asarray(ic_series, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return 0.0
    std = float(np.std(vals, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(vals) / std)


def t_stat(ic_series: Union[pd.Series, np.ndarray]) -> float:
    """
    One-sample t-statistic testing H0: mean(IC) = 0.

    t = mean(IC) / (std(IC) / sqrt(n))

    A t-stat > 2 (approximately p < 0.05 for n > 30) suggests the signal
    has statistically significant predictive power.

    Returns
    -------
    float (0.0 if insufficient data)
    """
    vals = np.asarray(ic_series, dtype=float)
    vals = vals[~np.isnan(vals)]
    n    = len(vals)
    if n < 3:
        return 0.0
    std = float(np.std(vals, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(vals) / (std / np.sqrt(n)))


# ---------------------------------------------------------------------------
# ICTracker class
# ---------------------------------------------------------------------------

class ICTracker:
    """
    Tracks Information Coefficient history for all signals in the Feature Store.

    Persists IC records in the ``ic_history`` table and aggregated stats in
    ``feature_metadata``.

    Parameters
    ----------
    db_path : str | Path
    """

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._conn   = self._make_conn()

    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    # ── Core IC computation ───────────────────────────────────────────────

    def compute_ic(
        self,
        signal_values:   Union[pd.Series, np.ndarray],
        forward_returns: Union[pd.Series, np.ndarray],
        method:          str = "spearman",
    ) -> float:
        """
        Compute IC (delegates to module-level function).
        """
        return compute_ic(signal_values, forward_returns, method=method)

    def rolling_ic(
        self,
        signal_values:   Union[pd.Series, np.ndarray],
        forward_returns: Union[pd.Series, np.ndarray],
        window:          int = 60,
        method:          str = "spearman",
        min_obs:         int = 10,
    ) -> pd.Series:
        """Rolling IC time series (delegates to module-level function)."""
        return rolling_ic(
            signal_values, forward_returns,
            window=window, method=method, min_obs=min_obs
        )

    def ic_decay(
        self,
        signal_values:   Union[pd.Series, np.ndarray],
        close_prices:    Union[pd.Series, np.ndarray],
        max_lag:         int = 20,
        method:          str = "spearman",
        min_obs:         int = 20,
    ) -> Dict[int, float]:
        """IC at multiple forward-return horizons."""
        return ic_decay(
            signal_values, close_prices,
            max_lag=max_lag, method=method, min_obs=min_obs
        )

    def icir(self, ic_series: Union[pd.Series, np.ndarray]) -> float:
        """IC Information Ratio."""
        return icir(ic_series)

    def t_stat(self, ic_series: Union[pd.Series, np.ndarray]) -> float:
        """t-statistic for IC > 0."""
        return t_stat(ic_series)

    # ── Persistence ───────────────────────────────────────────────────────

    def store_ic(
        self,
        signal_name:    str,
        symbol:         str,
        window_end_ts:  Any,
        ic_value:       float,
        forward_bars:   int  = 5,
        method:         str  = "spearman",
        n_obs:          Optional[int] = None,
    ) -> None:
        """
        Persist a single IC observation to the ic_history table.

        Parameters
        ----------
        signal_name : str
        symbol : str
        window_end_ts : timestamp
            The last bar of the window used to compute IC.
        ic_value : float
        forward_bars : int
            The forward-return horizon used.
        method : str
        n_obs : int | None
        """
        ts_str = (
            window_end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            if isinstance(window_end_ts, (datetime, pd.Timestamp))
            else str(window_end_ts)
        )
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO ic_history
                (signal_name, symbol, window_end_ts, forward_bars,
                 ic_value, method, n_obs, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [signal_name, symbol, ts_str, forward_bars,
             float(ic_value), method, n_obs, now],
        )
        self._conn.commit()

    def batch_store_ic(
        self,
        records: List[Dict[str, Any]],
    ) -> None:
        """
        Persist multiple IC records in a single transaction.

        Each record must be a dict with keys:
            signal_name, symbol, window_end_ts, ic_value
        Optional keys: forward_bars, method, n_obs
        """
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for r in records:
            ts_str = str(r["window_end_ts"])
            rows.append((
                r["signal_name"],
                r["symbol"],
                ts_str,
                int(r.get("forward_bars", 5)),
                float(r["ic_value"]),
                str(r.get("method", "spearman")),
                r.get("n_obs"),
                now,
            ))
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO ic_history
                (signal_name, symbol, window_end_ts, forward_bars,
                 ic_value, method, n_obs, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    # ── Retrieval ─────────────────────────────────────────────────────────

    def get_ic_history(
        self,
        signal_name:   str,
        symbol:        Optional[str] = None,
        forward_bars:  Optional[int] = None,
        method:        str           = "spearman",
        start_ts:      Optional[Any] = None,
        end_ts:        Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Retrieve IC history for a signal.

        Returns a DataFrame with columns:
            symbol, window_end_ts, forward_bars, ic_value, n_obs, computed_at
        """
        clauses = ["signal_name = ?", "method = ?"]
        params: List[Any] = [signal_name, method]

        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if forward_bars is not None:
            clauses.append("forward_bars = ?")
            params.append(forward_bars)
        if start_ts is not None:
            clauses.append("window_end_ts >= ?")
            params.append(str(start_ts))
        if end_ts is not None:
            clauses.append("window_end_ts <= ?")
            params.append(str(end_ts))

        sql = (
            "SELECT symbol, window_end_ts, forward_bars, ic_value, n_obs, computed_at "
            "FROM ic_history WHERE "
            + " AND ".join(clauses)
            + " ORDER BY window_end_ts"
        )
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["symbol", "window_end_ts", "forward_bars",
                         "ic_value", "n_obs", "computed_at"]
            )
        return pd.DataFrame([dict(r) for r in rows])

    def ic_series_for_signal(
        self,
        signal_name:  str,
        symbol:       str,
        forward_bars: int = 5,
        method:       str = "spearman",
    ) -> pd.Series:
        """
        Return the IC time series for a signal/symbol pair as a pd.Series.
        """
        df = self.get_ic_history(
            signal_name, symbol=symbol,
            forward_bars=forward_bars, method=method
        )
        if df.empty:
            return pd.Series(dtype=float, name="ic_value")
        idx = pd.to_datetime(df["window_end_ts"], utc=True)
        return pd.Series(df["ic_value"].values, index=idx, name="ic_value")

    # ── Aggregation and ranking ───────────────────────────────────────────

    def signal_ic_summary(
        self,
        forward_bars: Optional[int] = None,
        method:       str           = "spearman",
    ) -> pd.DataFrame:
        """
        Return per-signal IC summary statistics.

        Columns: signal_name, mean_ic, std_ic, icir, t_stat,
                 n_obs, n_symbols, positive_ic_frac
        """
        clauses = ["method = ?"]
        params: List[Any] = [method]
        if forward_bars is not None:
            clauses.append("forward_bars = ?")
            params.append(forward_bars)

        sql = (
            "SELECT signal_name, ic_value, symbol "
            "FROM ic_history WHERE "
            + " AND ".join(clauses)
        )
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=["signal_name", "mean_ic", "std_ic", "icir",
                         "t_stat", "n_obs", "n_symbols", "positive_ic_frac"]
            )

        df = pd.DataFrame([dict(r) for r in rows])
        summaries = []
        for sig_name, grp in df.groupby("signal_name"):
            vals     = grp["ic_value"].dropna().values
            n        = len(vals)
            mean     = float(np.mean(vals)) if n > 0 else np.nan
            std      = float(np.std(vals, ddof=1)) if n > 1 else np.nan
            _icir    = icir(vals)
            _t       = t_stat(vals)
            pos_frac = float((vals > 0).mean()) if n > 0 else np.nan
            n_syms   = grp["symbol"].nunique()

            summaries.append({
                "signal_name":      sig_name,
                "mean_ic":          mean,
                "std_ic":           std,
                "icir":             _icir,
                "t_stat":           _t,
                "n_obs":            n,
                "n_symbols":        n_syms,
                "positive_ic_frac": pos_frac,
            })

        result = pd.DataFrame(summaries).sort_values("icir", ascending=False)
        return result.reset_index(drop=True)

    def top_signals_by_icir(
        self,
        n:            int             = 10,
        forward_bars: Optional[int]   = None,
        method:       str             = "spearman",
        min_obs:      int             = 20,
    ) -> pd.DataFrame:
        """
        Return the top-N signals ranked by IC information ratio.

        Parameters
        ----------
        n : int
            Number of signals to return.
        forward_bars : int | None
            Filter to a specific forward-return horizon.
        method : 'spearman' | 'pearson'
        min_obs : int
            Minimum IC observations required to be ranked.

        Returns
        -------
        pd.DataFrame with columns:
            rank, signal_name, mean_ic, icir, t_stat, n_obs, n_symbols
        """
        summary = self.signal_ic_summary(forward_bars=forward_bars, method=method)
        summary = summary[summary["n_obs"] >= min_obs].copy()
        summary = summary.sort_values("icir", ascending=False).head(n)
        summary.insert(0, "rank", range(1, len(summary) + 1))
        return summary.reset_index(drop=True)

    def update_metadata_ic_stats(self, method: str = "spearman") -> None:
        """
        Recompute aggregate IC stats for all signals and write them to
        feature_metadata.
        """
        summary = self.signal_ic_summary(method=method)
        for _, row in summary.iterrows():
            self._conn.execute(
                """
                UPDATE feature_metadata
                SET mean_ic    = ?,
                    icir       = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                WHERE signal_name = ?
                """,
                [row["mean_ic"], row["icir"], row["signal_name"]],
            )
        self._conn.commit()
        logger.info(
            "update_metadata_ic_stats: updated %d signal entries.", len(summary)
        )

    # ── Advanced IC analysis ──────────────────────────────────────────────

    def ic_decay_from_db(
        self,
        signal_name: str,
        symbol:      str,
        max_lag:     int = 20,
        method:      str = "spearman",
    ) -> pd.DataFrame:
        """
        Retrieve IC decay data from stored ic_history (using different
        forward_bars values already computed and stored).

        Returns a DataFrame with columns: forward_bars, mean_ic, icir, n_obs
        """
        sql = (
            "SELECT forward_bars, ic_value "
            "FROM ic_history "
            "WHERE signal_name = ? AND symbol = ? AND method = ? "
            "  AND forward_bars BETWEEN 1 AND ? "
            "ORDER BY forward_bars, window_end_ts"
        )
        rows = self._conn.execute(sql, [signal_name, symbol, method, max_lag]).fetchall()
        if not rows:
            return pd.DataFrame(columns=["forward_bars", "mean_ic", "icir", "n_obs"])

        df  = pd.DataFrame([dict(r) for r in rows])
        agg = df.groupby("forward_bars")["ic_value"].agg(
            mean_ic="mean", n_obs="count"
        )
        agg["icir"] = df.groupby("forward_bars")["ic_value"].apply(icir)
        agg = agg.reset_index().rename(columns={"forward_bars": "lag_bars"})
        return agg

    def cross_ic_correlation(
        self,
        signal_names: List[str],
        symbol:       str,
        forward_bars: int = 5,
        method:       str = "spearman",
    ) -> pd.DataFrame:
        """
        Compute pairwise correlation of IC series across signals.

        Useful for identifying signal redundancy: highly correlated IC series
        mean the signals carry the same information.

        Returns a correlation matrix DataFrame.
        """
        ic_series_dict: Dict[str, pd.Series] = {}
        for sig_name in signal_names:
            s = self.ic_series_for_signal(sig_name, symbol, forward_bars, method)
            if not s.empty:
                ic_series_dict[sig_name] = s

        if len(ic_series_dict) < 2:
            return pd.DataFrame()

        df  = pd.DataFrame(ic_series_dict).dropna(how="all")
        return df.corr(method="pearson")

    def compute_and_store_full_ic(
        self,
        signal_name:   str,
        symbol:        str,
        signal_values: Union[pd.Series, np.ndarray],
        close_prices:  Union[pd.Series, np.ndarray],
        forward_bars_list: Optional[List[int]] = None,
        window:        int                     = 60,
        method:        str                     = "spearman",
    ) -> Dict[int, float]:
        """
        End-to-end: compute rolling IC at multiple horizons and store all records.

        Parameters
        ----------
        signal_name : str
        symbol : str
        signal_values : array-like
        close_prices : array-like
            Close prices for computing forward returns.
        forward_bars_list : list[int] | None
            Horizons to test. Defaults to [1, 2, 3, 5, 10, 20].
        window : int
            Rolling IC window.
        method : 'spearman' | 'pearson'

        Returns
        -------
        dict[forward_bars, mean_IC_over_window]
        """
        if forward_bars_list is None:
            forward_bars_list = [1, 2, 3, 5, 10, 20]

        sig   = np.asarray(signal_values, dtype=float)
        close = np.asarray(close_prices,  dtype=float)
        n     = len(sig)

        if isinstance(signal_values, pd.Series):
            ts_index = signal_values.index
        else:
            ts_index = pd.RangeIndex(n)

        mean_ics: Dict[int, float] = {}
        all_records: List[Dict[str, Any]] = []

        for fb in forward_bars_list:
            # Forward returns
            fwd_ret = np.full(n, np.nan)
            if fb < n:
                fwd_ret[:-fb] = np.log(close[fb:] / close[:-fb])

            # Rolling IC
            ic_series_vals = rolling_ic(
                sig, fwd_ret, window=window, method=method, min_obs=max(10, window // 4)
            )

            # Store all non-NaN points
            for i, ic_val in enumerate(ic_series_vals):
                if np.isnan(ic_val):
                    continue
                ts = ts_index[i] if i < len(ts_index) else i
                all_records.append({
                    "signal_name":  signal_name,
                    "symbol":       symbol,
                    "window_end_ts": ts,
                    "ic_value":     float(ic_val),
                    "forward_bars": fb,
                    "method":       method,
                    "n_obs":        window,
                })

            valid_ic = ic_series_vals.dropna().values
            mean_ics[fb] = float(np.mean(valid_ic)) if len(valid_ic) > 0 else 0.0

        if all_records:
            self.batch_store_ic(all_records)

        return mean_ics
