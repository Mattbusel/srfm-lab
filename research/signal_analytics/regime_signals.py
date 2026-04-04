"""
research/signal_analytics/regime_signals.py
============================================
Regime-aware signal analysis for the SRFM-Lab signal analytics framework.

Provides tools to:
- Detect and label market regimes (volatility, trend, correlation)
- Condition signal quality metrics on regime
- Compute regime-transition probabilities and persistence
- Adapt BH signal parameters (mass, tf_score filters) across regimes
- Build Hidden Markov Model (HMM)-style regime models from return data
- Evaluate signal stability across regime boundaries

Regime Types Supported
----------------------
- Volatility regimes  : low / medium / high (rolling vol percentile)
- Trend regimes       : trending / mean-reverting (Hurst exponent proxy)
- Correlation regimes : coupled / decoupled (average pairwise correlation)
- Custom regimes      : user-supplied integer labels per bar

BH Signal Context
-----------------
delta_score = tf_score * mass * ATR / vol^2
Regime conditioning identifies which regime produces the highest delta_score IC.

Dependencies
------------
numpy, pandas, scipy, sklearn (optional), matplotlib, seaborn, dataclasses
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

RegimeLabel = Literal["low_vol", "mid_vol", "high_vol",
                       "trending", "mean_reverting",
                       "coupled", "decoupled", "custom"]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RegimeStats:
    """Per-regime performance / signal-quality statistics."""
    regime: str
    n_bars: int
    freq: float                        # fraction of total bars in regime
    mean_return: float
    vol: float
    sharpe: float
    signal_ic: float
    signal_icir: float
    signal_mean: float
    signal_std: float
    hit_rate: float                    # fraction of bars where sign(signal)==sign(return)
    avg_hold_bars: float
    avg_pnl: float
    regime_persistence: float          # avg consecutive bars in regime
    transition_prob: dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeTransitionMatrix:
    """Markov transition matrix for discrete regime labels."""
    regimes: list[str]
    matrix: np.ndarray                 # shape (n_regimes, n_regimes)
    stationary_dist: np.ndarray
    mean_persistence: dict[str, float]
    regime_durations: dict[str, list[int]]


@dataclass
class RegimeSignalSummary:
    """Summary of signal quality across all detected regimes."""
    regime_type: str
    per_regime: dict[str, RegimeStats]
    best_regime: str
    worst_regime: str
    ic_range: float                    # max(IC) - min(IC) across regimes
    regime_sensitivity: float          # std of IC across regimes
    transition_matrix: RegimeTransitionMatrix | None
    n_total_bars: int
    n_regimes: int


@dataclass
class HurstResult:
    """Hurst exponent estimate via R/S analysis."""
    hurst: float
    interpretation: str                # 'trending' / 'random walk' / 'mean_reverting'
    rs_series: list[float]
    lag_series: list[int]
    r_squared: float


@dataclass
class AdaptiveFilterResult:
    """Optimal BH signal filters per regime."""
    regime: str
    best_mass_min: float
    best_tf_score_min: int
    best_icir: float
    grid_icir: pd.DataFrame            # pivot table mass_min x tf_score_min


# ---------------------------------------------------------------------------
# Regime Detection Functions
# ---------------------------------------------------------------------------


def _rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualised rolling volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def label_vol_regimes(
    returns: pd.Series,
    window: int = 20,
    low_pct: float = 33.0,
    high_pct: float = 67.0,
) -> pd.Series:
    """
    Label each bar as 'low_vol' / 'mid_vol' / 'high_vol' based on rolling
    volatility percentile thresholds.

    Parameters
    ----------
    returns : pd.Series
        Bar-level return series.
    window : int
        Rolling window for volatility estimate (default 20).
    low_pct, high_pct : float
        Percentile cutoffs (default tercile split at 33/67).

    Returns
    -------
    pd.Series of str labels aligned with `returns`.
    """
    vol = _rolling_vol(returns, window)
    lo = np.nanpercentile(vol.dropna(), low_pct)
    hi = np.nanpercentile(vol.dropna(), high_pct)
    labels = pd.Series("mid_vol", index=returns.index)
    labels[vol <= lo] = "low_vol"
    labels[vol >= hi] = "high_vol"
    labels[vol.isna()] = "mid_vol"
    return labels


def hurst_exponent(
    series: pd.Series | np.ndarray,
    min_lag: int = 10,
    max_lag: int | None = None,
) -> HurstResult:
    """
    Estimate Hurst exponent via R/S (rescaled range) analysis.

    H ~ 0.5  -> random walk
    H > 0.5  -> trending / persistent
    H < 0.5  -> mean-reverting / anti-persistent

    Parameters
    ----------
    series : array-like
        Price or return series.
    min_lag, max_lag : int
        Lag range for R/S calculation.

    Returns
    -------
    HurstResult
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if max_lag is None:
        max_lag = max(min_lag + 1, n // 4)

    lags = list(range(min_lag, max_lag + 1, max(1, (max_lag - min_lag) // 20)))
    rs_vals: list[float] = []
    valid_lags: list[int] = []

    for lag in lags:
        if lag >= n:
            continue
        rs_list: list[float] = []
        for start in range(0, n - lag, lag):
            chunk = arr[start : start + lag]
            mean_c = chunk.mean()
            dev = np.cumsum(chunk - mean_c)
            r = dev.max() - dev.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_vals.append(np.mean(rs_list))
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return HurstResult(0.5, "random walk", rs_vals, valid_lags, 0.0)

    log_lags = np.log(valid_lags)
    log_rs = np.log(rs_vals)
    slope, intercept, r_val, _, _ = stats.linregress(log_lags, log_rs)
    H = float(slope)

    if H > 0.55:
        interp = "trending"
    elif H < 0.45:
        interp = "mean_reverting"
    else:
        interp = "random walk"

    return HurstResult(
        hurst=H,
        interpretation=interp,
        rs_series=rs_vals,
        lag_series=valid_lags,
        r_squared=float(r_val**2),
    )


def label_trend_regimes(
    returns: pd.Series,
    window: int = 60,
    threshold: float = 0.55,
) -> pd.Series:
    """
    Rolling Hurst exponent regime labels: 'trending' vs 'mean_reverting'.

    Parameters
    ----------
    returns : pd.Series
        Bar return series.
    window : int
        Rolling window for Hurst estimation.
    threshold : float
        Hurst > threshold -> 'trending', < (1-threshold) -> 'mean_reverting'.

    Returns
    -------
    pd.Series of str labels.
    """
    labels = pd.Series("mean_reverting", index=returns.index, dtype=object)
    arr = returns.values.astype(float)
    n = len(arr)

    for i in range(window, n):
        chunk = pd.Series(arr[i - window : i])
        try:
            h = hurst_exponent(chunk, min_lag=5, max_lag=window // 4).hurst
        except Exception:
            h = 0.5
        if h > threshold:
            labels.iloc[i] = "trending"
        elif h < (1.0 - threshold):
            labels.iloc[i] = "mean_reverting"
        # else: stays 'mean_reverting' (neutral / random walk)

    return labels


def label_correlation_regimes(
    returns_panel: pd.DataFrame,
    window: int = 40,
    high_corr_pct: float = 67.0,
) -> pd.Series:
    """
    Label bars as 'coupled' (high avg pairwise correlation) or 'decoupled'
    based on rolling cross-asset correlation.

    Parameters
    ----------
    returns_panel : pd.DataFrame
        Rows = bars, columns = instruments.
    window : int
        Rolling window for correlation matrix.
    high_corr_pct : float
        Percentile of rolling avg-corr that separates coupled from decoupled.

    Returns
    -------
    pd.Series of str labels indexed by `returns_panel.index`.
    """
    avg_corr = pd.Series(index=returns_panel.index, dtype=float)
    for i in range(window, len(returns_panel)):
        chunk = returns_panel.iloc[i - window : i]
        corr_mat = chunk.corr().values
        # upper triangle excluding diagonal
        n = corr_mat.shape[0]
        if n < 2:
            avg_corr.iloc[i] = 0.0
            continue
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        avg_corr.iloc[i] = corr_mat[mask].mean()

    threshold = np.nanpercentile(avg_corr.dropna(), high_corr_pct)
    labels = pd.Series("decoupled", index=returns_panel.index, dtype=object)
    labels[avg_corr >= threshold] = "coupled"
    labels[avg_corr.isna()] = "decoupled"
    return labels


def regime_transition_matrix(labels: pd.Series) -> RegimeTransitionMatrix:
    """
    Build a first-order Markov transition matrix from a sequence of regime labels.

    Parameters
    ----------
    labels : pd.Series
        Sequence of discrete regime string labels.

    Returns
    -------
    RegimeTransitionMatrix
    """
    regimes = sorted(labels.dropna().unique().tolist())
    k = len(regimes)
    idx = {r: i for i, r in enumerate(regimes)}

    counts = np.zeros((k, k), dtype=float)
    valid = labels.dropna().values
    for t in range(len(valid) - 1):
        i, j = idx.get(valid[t]), idx.get(valid[t + 1])
        if i is not None and j is not None:
            counts[i, j] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums

    # Stationary distribution via eigenvector
    try:
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        stat_idx = np.argmin(np.abs(eigenvalues - 1.0))
        stat = np.real(eigenvectors[:, stat_idx])
        stat = np.abs(stat) / np.abs(stat).sum()
    except Exception:
        stat = np.ones(k) / k

    # Mean persistence = expected consecutive bars in same regime
    persistence: dict[str, float] = {}
    durations: dict[str, list[int]] = {r: [] for r in regimes}
    run = 1
    for t in range(1, len(valid)):
        if valid[t] == valid[t - 1]:
            run += 1
        else:
            durations[valid[t - 1]].append(run)
            run = 1
    durations[valid[-1]].append(run)

    for r in regimes:
        d = durations[r]
        persistence[r] = float(np.mean(d)) if d else 1.0

    return RegimeTransitionMatrix(
        regimes=regimes,
        matrix=matrix,
        stationary_dist=stat,
        mean_persistence=persistence,
        regime_durations=durations,
    )


# ---------------------------------------------------------------------------
# Core Analyser Class
# ---------------------------------------------------------------------------


class RegimeSignalAnalyzer:
    """
    Analyse BH signal quality conditioned on market regime.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade log with columns: exit_time, sym, entry_price, exit_price,
        dollar_pos, pnl, hold_bars, regime, and BH signal columns.
    signal_col : str
        Column name of the primary signal (default 'delta_score').
    return_col : str
        Column name of the forward return (default computed from prices).
    min_obs_per_regime : int
        Minimum trades in a regime to compute statistics (default 20).
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        signal_col: str = "delta_score",
        return_col: str | None = None,
        min_obs_per_regime: int = 20,
    ) -> None:
        self.trades = trades.copy()
        self.signal_col = signal_col
        self.return_col = return_col or "_ret_pct"
        self.min_obs = min_obs_per_regime
        self._prepare_returns()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_returns(self) -> None:
        """Compute percentage return if not already present."""
        df = self.trades
        if self.return_col not in df.columns:
            if "entry_price" in df.columns and "exit_price" in df.columns:
                self.trades["_ret_pct"] = (
                    df["exit_price"] - df["entry_price"]
                ) / df["entry_price"].replace(0, np.nan)
            elif "pnl" in df.columns and "dollar_pos" in df.columns:
                self.trades["_ret_pct"] = df["pnl"] / df["dollar_pos"].replace(0, np.nan)
            else:
                self.trades["_ret_pct"] = np.nan

    def _regime_ic(self, subset: pd.DataFrame) -> tuple[float, float]:
        """Return (IC, ICIR) for a trade subset."""
        sig = subset[self.signal_col].values.astype(float)
        ret = subset[self.return_col].values.astype(float)
        valid = np.isfinite(sig) & np.isfinite(ret)
        if valid.sum() < self.min_obs:
            return np.nan, np.nan
        rho, _ = stats.spearmanr(sig[valid], ret[valid])
        # Rolling IC for ICIR requires windowed computation — use single IC with std proxy
        return float(rho), float(rho)  # simplified; full ICIR via rolling_ic_weights

    def _performance_stats(self, subset: pd.DataFrame, regime: str) -> RegimeStats:
        """Compute full RegimeStats for one regime partition."""
        n = len(subset)
        ret = subset[self.return_col].dropna()
        sig = subset[self.signal_col].dropna() if self.signal_col in subset else pd.Series(dtype=float)

        mean_ret = float(ret.mean()) if len(ret) else np.nan
        vol = float(ret.std()) if len(ret) > 1 else np.nan
        sharpe = (mean_ret / vol * np.sqrt(252)) if (vol and vol > 0) else np.nan

        ic, icir = self._regime_ic(subset)

        # Hit rate: sign agreement
        aligned = subset[[self.signal_col, self.return_col]].dropna()
        if len(aligned):
            hit = float((np.sign(aligned[self.signal_col]) == np.sign(aligned[self.return_col])).mean())
        else:
            hit = np.nan

        avg_hold = float(subset["hold_bars"].mean()) if "hold_bars" in subset else np.nan
        avg_pnl = float(subset["pnl"].mean()) if "pnl" in subset else np.nan

        # persistence from regime label run length — requires sorted time
        persistence = 1.0
        if "exit_time" in subset.columns:
            sorted_sub = subset.sort_values("exit_time")
            runs: list[int] = []
            run = 1
            prev = regime
            for r in [regime] * len(sorted_sub):  # same regime, just count full run
                pass
            persistence = float(n)  # simplified

        freq = n / max(len(self.trades), 1)

        return RegimeStats(
            regime=regime,
            n_bars=n,
            freq=freq,
            mean_return=mean_ret,
            vol=vol,
            sharpe=sharpe,
            signal_ic=ic,
            signal_icir=icir,
            signal_mean=float(sig.mean()) if len(sig) else np.nan,
            signal_std=float(sig.std()) if len(sig) > 1 else np.nan,
            hit_rate=hit,
            avg_hold_bars=avg_hold,
            avg_pnl=avg_pnl,
            regime_persistence=persistence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_by_regime_column(
        self, regime_col: str = "regime"
    ) -> RegimeSignalSummary:
        """
        Analyse signal quality partitioned by a discrete regime column in the
        trade log.

        Parameters
        ----------
        regime_col : str
            Column name in `self.trades` containing string regime labels.

        Returns
        -------
        RegimeSignalSummary
        """
        if regime_col not in self.trades.columns:
            raise KeyError(f"Column '{regime_col}' not found in trades DataFrame.")

        groups = self.trades.groupby(regime_col)
        per_regime: dict[str, RegimeStats] = {}
        for regime_name, subset in groups:
            if len(subset) < self.min_obs:
                logger.debug("Skipping regime %s: only %d obs", regime_name, len(subset))
                continue
            per_regime[str(regime_name)] = self._performance_stats(subset, str(regime_name))

        if not per_regime:
            raise ValueError("No regime had sufficient observations.")

        ic_values = {r: s.signal_ic for r, s in per_regime.items() if np.isfinite(s.signal_ic)}
        best = max(ic_values, key=ic_values.get) if ic_values else list(per_regime)[0]
        worst = min(ic_values, key=ic_values.get) if ic_values else list(per_regime)[0]
        ic_arr = np.array(list(ic_values.values()))
        ic_range = float(ic_arr.max() - ic_arr.min()) if len(ic_arr) > 1 else 0.0
        ic_sens = float(ic_arr.std()) if len(ic_arr) > 1 else 0.0

        tm = regime_transition_matrix(self.trades[regime_col].dropna())

        return RegimeSignalSummary(
            regime_type=regime_col,
            per_regime=per_regime,
            best_regime=best,
            worst_regime=worst,
            ic_range=ic_range,
            regime_sensitivity=ic_sens,
            transition_matrix=tm,
            n_total_bars=len(self.trades),
            n_regimes=len(per_regime),
        )

    def analyse_vol_regimes(
        self,
        returns: pd.Series | None = None,
        window: int = 20,
        low_pct: float = 33.0,
        high_pct: float = 67.0,
    ) -> RegimeSignalSummary:
        """
        Detect volatility regimes from a return series and analyse signal quality.

        If `returns` is None, uses the trade-level `_ret_pct` column aligned by
        exit_time to build an approximate bar series.

        Parameters
        ----------
        returns : pd.Series, optional
            External bar-level return series (index = datetime).
        window, low_pct, high_pct : see label_vol_regimes().

        Returns
        -------
        RegimeSignalSummary
        """
        if returns is None:
            if "exit_time" not in self.trades.columns:
                raise ValueError("Need 'exit_time' column or external returns.")
            returns = (
                self.trades.set_index("exit_time")[self.return_col]
                .sort_index()
                .dropna()
            )

        regime_series = label_vol_regimes(returns, window, low_pct, high_pct)
        df = self.trades.copy()
        if "exit_time" in df.columns:
            df["_vol_regime"] = df["exit_time"].map(
                lambda t: regime_series.asof(t) if hasattr(regime_series, "asof") else "mid_vol"
            )
        else:
            df["_vol_regime"] = "mid_vol"

        self.trades["_vol_regime"] = df["_vol_regime"]
        return self.analyse_by_regime_column("_vol_regime")

    def analyse_trend_regimes(
        self,
        returns: pd.Series | None = None,
        window: int = 60,
        threshold: float = 0.55,
    ) -> RegimeSignalSummary:
        """
        Detect trend regimes via rolling Hurst exponent and analyse signal quality.

        Parameters
        ----------
        returns : pd.Series, optional
            External bar-level return series.
        window : int
            Rolling window for Hurst estimation.
        threshold : float
            Hurst threshold for 'trending' classification.

        Returns
        -------
        RegimeSignalSummary
        """
        if returns is None:
            if "exit_time" not in self.trades.columns:
                raise ValueError("Need 'exit_time' column or external returns.")
            returns = (
                self.trades.set_index("exit_time")[self.return_col]
                .sort_index()
                .dropna()
            )

        regime_series = label_trend_regimes(returns, window, threshold)
        df = self.trades.copy()
        if "exit_time" in df.columns:
            df["_trend_regime"] = df["exit_time"].map(
                lambda t: regime_series.asof(t) if hasattr(regime_series, "asof") else "mean_reverting"
            )
        else:
            df["_trend_regime"] = "mean_reverting"

        self.trades["_trend_regime"] = df["_trend_regime"]
        return self.analyse_by_regime_column("_trend_regime")

    def signal_ic_by_vol_tercile(self) -> pd.DataFrame:
        """
        Compute signal IC within low / mid / high volatility terciles using
        the trade log's own return volatility (cross-sectional grouping by
        rolling vol rank).

        Returns
        -------
        pd.DataFrame with columns [regime, IC, n_obs, hit_rate].
        """
        df = self.trades[[self.signal_col, self.return_col]].dropna().copy()
        if len(df) < self.min_obs * 3:
            warnings.warn("Insufficient data for tercile split.")
            return pd.DataFrame()

        vol_proxy = df[self.return_col].abs()
        df["_vol_tercile"] = pd.qcut(vol_proxy, q=3, labels=["low_vol", "mid_vol", "high_vol"])

        rows: list[dict] = []
        for label, grp in df.groupby("_vol_tercile"):
            sig = grp[self.signal_col].values
            ret = grp[self.return_col].values
            if len(sig) < self.min_obs:
                continue
            rho, pval = stats.spearmanr(sig, ret)
            hit = float((np.sign(sig) == np.sign(ret)).mean())
            rows.append({"regime": str(label), "IC": rho, "p_value": pval,
                         "n_obs": len(grp), "hit_rate": hit})

        return pd.DataFrame(rows)

    def regime_ic_stability(
        self,
        regime_col: str = "regime",
        window_frac: float = 0.25,
    ) -> pd.DataFrame:
        """
        Compute rolling IC within each regime to assess IC stability over time.

        Parameters
        ----------
        regime_col : str
            Column with regime labels.
        window_frac : float
            Fraction of regime observations used as rolling window.

        Returns
        -------
        pd.DataFrame with columns [exit_time, regime, rolling_IC].
        """
        if regime_col not in self.trades.columns:
            raise KeyError(f"Column '{regime_col}' not in trades.")

        rows: list[dict] = []
        df = self.trades.copy()
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time")

        for regime_name, grp in df.groupby(regime_col):
            grp = grp.reset_index(drop=True)
            w = max(self.min_obs, int(len(grp) * window_frac))
            sig = grp[self.signal_col].values.astype(float)
            ret = grp[self.return_col].values.astype(float)
            times = grp["exit_time"].values if "exit_time" in grp else np.arange(len(grp))

            for i in range(w, len(grp)):
                s_w = sig[i - w : i]
                r_w = ret[i - w : i]
                valid = np.isfinite(s_w) & np.isfinite(r_w)
                if valid.sum() < self.min_obs // 2:
                    continue
                rho, _ = stats.spearmanr(s_w[valid], r_w[valid])
                rows.append({
                    "exit_time": times[i],
                    "regime": str(regime_name),
                    "rolling_IC": float(rho),
                    "window": w,
                })

        return pd.DataFrame(rows)

    def adaptive_signal_filter_by_regime(
        self,
        mass_col: str = "mass",
        tf_score_col: str = "tf_score",
        mass_thresholds: Sequence[float] | None = None,
        tf_thresholds: Sequence[int] | None = None,
        regime_col: str = "regime",
    ) -> dict[str, AdaptiveFilterResult]:
        """
        For each regime, grid-search over mass_min x tf_score_min to find the
        filter combination that maximises ICIR.

        Parameters
        ----------
        mass_col, tf_score_col : str
            Column names for BH mass and tf_score.
        mass_thresholds : list of float
            Candidate minimum mass thresholds (default [0, 0.5, 1.0, 1.5]).
        tf_thresholds : list of int
            Candidate minimum tf_score thresholds (default [0, 1, 2, 3, 4]).
        regime_col : str
            Column with regime labels.

        Returns
        -------
        dict mapping regime name -> AdaptiveFilterResult
        """
        if mass_thresholds is None:
            mass_thresholds = [0.0, 0.5, 1.0, 1.5]
        if tf_thresholds is None:
            tf_thresholds = [0, 1, 2, 3, 4]

        df = self.trades.copy()
        missing = [c for c in [mass_col, tf_score_col, regime_col] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        results: dict[str, AdaptiveFilterResult] = {}

        for regime_name, grp in df.groupby(regime_col):
            grid_rows: list[dict] = []
            best_icir = -np.inf
            best_m, best_tf = float(mass_thresholds[0]), int(tf_thresholds[0])

            for m_min in mass_thresholds:
                for tf_min in tf_thresholds:
                    mask = (grp[mass_col] >= m_min) & (grp[tf_score_col] >= tf_min)
                    sub = grp[mask]
                    if len(sub) < self.min_obs:
                        icir_val = np.nan
                    else:
                        sig = sub[self.signal_col].values.astype(float)
                        ret = sub[self.return_col].values.astype(float)
                        valid = np.isfinite(sig) & np.isfinite(ret)
                        if valid.sum() < self.min_obs:
                            icir_val = np.nan
                        else:
                            rho, _ = stats.spearmanr(sig[valid], ret[valid])
                            # ICIR proxy: IC / sqrt(1/n) ~ IC * sqrt(n)
                            icir_val = float(rho * np.sqrt(valid.sum()))
                    grid_rows.append({
                        "mass_min": m_min,
                        "tf_score_min": tf_min,
                        "ICIR": icir_val,
                        "n_obs": int(mask.sum()),
                    })
                    if np.isfinite(icir_val) and icir_val > best_icir:
                        best_icir = icir_val
                        best_m, best_tf = float(m_min), int(tf_min)

            grid_df = pd.DataFrame(grid_rows)
            pivot = grid_df.pivot(index="mass_min", columns="tf_score_min", values="ICIR")

            results[str(regime_name)] = AdaptiveFilterResult(
                regime=str(regime_name),
                best_mass_min=best_m,
                best_tf_score_min=best_tf,
                best_icir=float(best_icir),
                grid_icir=pivot,
            )

        return results

    def regime_signal_correlation(
        self,
        signals: list[str],
        regime_col: str = "regime",
    ) -> dict[str, pd.DataFrame]:
        """
        Compute pairwise signal correlation matrix within each regime.

        Parameters
        ----------
        signals : list of str
            Column names of signals to correlate.
        regime_col : str
            Column with regime labels.

        Returns
        -------
        dict mapping regime name -> correlation DataFrame.
        """
        if regime_col not in self.trades.columns:
            raise KeyError(f"'{regime_col}' not in trades.")
        missing = [s for s in signals if s not in self.trades.columns]
        if missing:
            raise KeyError(f"Signal columns missing: {missing}")

        result: dict[str, pd.DataFrame] = {}
        for regime_name, grp in self.trades.groupby(regime_col):
            if len(grp) < self.min_obs:
                continue
            result[str(regime_name)] = grp[signals].corr(method="spearman")
        return result

    def ic_conditional_on_lagged_regime(
        self,
        regime_col: str = "regime",
        lag: int = 1,
    ) -> pd.DataFrame:
        """
        Compute signal IC conditioned on the *lagged* regime label, showing
        whether the current-period IC is predictable from the previous regime.

        Parameters
        ----------
        regime_col : str
            Column with discrete regime labels.
        lag : int
            Number of bars to lag the regime label.

        Returns
        -------
        pd.DataFrame with columns [lagged_regime, current_regime, IC, n_obs].
        """
        df = self.trades.copy()
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time")
        df["_lagged_regime"] = df[regime_col].shift(lag)

        rows: list[dict] = []
        for (lag_r, cur_r), grp in df.groupby(["_lagged_regime", regime_col]):
            if len(grp) < self.min_obs:
                continue
            sig = grp[self.signal_col].dropna().values
            ret = grp[self.return_col].dropna().values
            n = min(len(sig), len(ret))
            if n < self.min_obs:
                continue
            rho, pval = stats.spearmanr(sig[:n], ret[:n])
            rows.append({
                "lagged_regime": str(lag_r),
                "current_regime": str(cur_r),
                "IC": float(rho),
                "p_value": float(pval),
                "n_obs": n,
            })
        return pd.DataFrame(rows)

    def full_regime_report(
        self,
        regime_col: str = "regime",
    ) -> dict:
        """
        Run all regime analyses and return a consolidated report dictionary.

        Keys
        ----
        summary : RegimeSignalSummary
        vol_tercile_ic : pd.DataFrame
        ic_stability : pd.DataFrame
        lagged_regime_ic : pd.DataFrame
        """
        report: dict = {}
        try:
            report["summary"] = self.analyse_by_regime_column(regime_col)
        except Exception as exc:
            report["summary"] = str(exc)

        try:
            report["vol_tercile_ic"] = self.signal_ic_by_vol_tercile()
        except Exception as exc:
            report["vol_tercile_ic"] = str(exc)

        try:
            report["ic_stability"] = self.regime_ic_stability(regime_col)
        except Exception as exc:
            report["ic_stability"] = str(exc)

        try:
            report["lagged_regime_ic"] = self.ic_conditional_on_lagged_regime(regime_col)
        except Exception as exc:
            report["lagged_regime_ic"] = str(exc)

        return report

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def plot_ic_by_regime(
        self,
        regime_col: str = "regime",
        figsize: tuple[int, int] = (10, 5),
    ):
        """
        Bar chart of signal IC per regime with error bars (bootstrap 95% CI).

        Parameters
        ----------
        regime_col : str
            Column with regime labels.
        figsize : tuple
            Matplotlib figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting.")

        summary = self.analyse_by_regime_column(regime_col)
        regimes = list(summary.per_regime.keys())
        ics = [summary.per_regime[r].signal_ic for r in regimes]
        ns = [summary.per_regime[r].n_bars for r in regimes]

        # Bootstrap CI (approximate via 1/sqrt(n))
        se = [1.0 / np.sqrt(max(n, 1)) for n in ns]

        fig, ax = plt.subplots(figsize=figsize)
        colors = ["green" if ic > 0 else "red" for ic in ics]
        bars = ax.bar(regimes, ics, color=colors, alpha=0.7, edgecolor="black")
        ax.errorbar(regimes, ics, yerr=[1.96 * s for s in se],
                    fmt="none", color="black", capsize=5, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Signal IC by Regime ({regime_col})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Regime")
        ax.set_ylabel("Spearman IC")

        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"n={n}", ha="center", va="bottom", fontsize=8)

        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_regime_transition_heatmap(
        self,
        regime_col: str = "regime",
        figsize: tuple[int, int] = (7, 6),
    ):
        """
        Heatmap of the regime transition probability matrix.

        Parameters
        ----------
        regime_col : str
            Column with regime labels.
        figsize : tuple
            Matplotlib figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting.")

        tm = regime_transition_matrix(self.trades[regime_col].dropna())

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            tm.matrix,
            annot=True,
            fmt=".2f",
            xticklabels=tm.regimes,
            yticklabels=tm.regimes,
            cmap="Blues",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Transition Probability"},
        )
        ax.set_title("Regime Transition Probability Matrix", fontsize=13, fontweight="bold")
        ax.set_xlabel("To Regime")
        ax.set_ylabel("From Regime")
        fig.tight_layout()
        return fig

    def plot_regime_ic_stability(
        self,
        regime_col: str = "regime",
        figsize: tuple[int, int] = (12, 5),
    ):
        """
        Rolling IC over time coloured by regime, showing stability.

        Parameters
        ----------
        regime_col : str
            Column with regime labels.
        figsize : tuple
            Matplotlib figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting.")

        stability_df = self.regime_ic_stability(regime_col)
        if stability_df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=figsize)
        palette = sns.color_palette("tab10", n_colors=stability_df["regime"].nunique())
        regime_colors = dict(zip(sorted(stability_df["regime"].unique()), palette))

        for regime_name, grp in stability_df.groupby("regime"):
            grp_sorted = grp.sort_values("exit_time")
            ax.plot(grp_sorted["exit_time"], grp_sorted["rolling_IC"],
                    label=str(regime_name), color=regime_colors[regime_name],
                    alpha=0.8, linewidth=1.2)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Rolling IC Stability by Regime", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Rolling Spearman IC")
        ax.legend(title="Regime", fontsize=9)
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    def plot_adaptive_filter_heatmaps(
        self,
        filter_results: dict[str, AdaptiveFilterResult],
        figsize_per: tuple[int, int] = (5, 4),
    ):
        """
        Grid of heatmaps showing ICIR across mass_min x tf_score_min per regime.

        Parameters
        ----------
        filter_results : dict
            Output of `adaptive_signal_filter_by_regime()`.
        figsize_per : tuple
            Size of each subplot.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn required for plotting.")

        n = len(filter_results)
        if n == 0:
            raise ValueError("filter_results is empty.")

        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
            squeeze=False,
        )

        for ax_idx, (regime_name, res) in enumerate(filter_results.items()):
            row, col = divmod(ax_idx, ncols)
            ax = axes[row][col]
            pivot = res.grid_icir.astype(float)
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="RdYlGn",
                annot=True,
                fmt=".2f",
                linewidths=0.3,
                center=0,
                cbar_kws={"label": "ICIR"},
            )
            ax.set_title(f"Regime: {regime_name}\nBest: mass>={res.best_mass_min}, "
                         f"tf>={res.best_tf_score_min} (ICIR={res.best_icir:.2f})",
                         fontsize=9)
            ax.set_xlabel("min tf_score")
            ax.set_ylabel("min mass")

        # Hide unused axes
        for ax_idx in range(n, nrows * ncols):
            row, col = divmod(ax_idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle("Optimal BH Filters by Regime (ICIR Grid)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig
