"""
Structural break detection in financial time series.

Implements:
- Chow test for a single known breakpoint
- Quandt Likelihood Ratio (QLR) test for unknown breakpoint
- CUSUM test via recursive OLS residuals
- Bai-Perron multiple structural break algorithm
- BreakReport aggregating all detected breaks
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# BH physics constants (used in regime classification heuristics)
# ---------------------------------------------------------------------------

BH_MASS_THRESH = 1.92
BH_DECAY = 0.924
BH_COLLAPSE = 0.992

# Hurst thresholds
HURST_TREND = 0.58
HURST_MR = 0.42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChowResult:
    """Result of a Chow test at a given breakpoint."""
    f_stat: float
    p_value: float
    breakpoint: int           # index in the array
    breakpoint_date: object   # date if available, else None
    rss_restricted: float
    rss_unrestricted: float
    n: int
    k: int                    # number of parameters in each OLS model


@dataclass
class QLRResult:
    """Result of the Quandt Likelihood Ratio test."""
    sup_f_stat: float
    breakpoint: int
    breakpoint_date: object
    p_value: float
    f_stats: np.ndarray       # F-stats at each candidate breakpoint
    candidate_indices: np.ndarray
    trim: float


@dataclass
class CUSUMResult:
    """Result of the CUSUM structural stability test."""
    cusum_stats: np.ndarray   # CUSUM values over time
    upper_band: np.ndarray
    lower_band: np.ndarray
    reject_null: bool         # True => structural break detected
    first_break_index: Optional[int]  # first index where band is breached
    first_break_date: object
    n: int
    significance: float


@dataclass
class BaiPerronBreak:
    """Single break in a Bai-Perron decomposition."""
    index: int
    date: object
    f_stat: float
    regime_before: str
    regime_after: str


@dataclass
class BaiPerronResult:
    """Result of Bai-Perron multiple structural break estimation."""
    n_breaks: int
    break_indices: List[int]
    break_dates: List[object]
    bic_values: Dict[int, float]   # BIC per number of breaks k
    segment_means: List[float]
    segment_stds: List[float]
    breaks: List[BaiPerronBreak]


@dataclass
class BreakReport:
    """Aggregated report of all structural break tests on a single series."""
    series_name: str
    n_obs: int
    chow_results: List[ChowResult]
    qlr_result: Optional[QLRResult]
    cusum_result: Optional[CUSUMResult]
    bai_perron_result: Optional[BaiPerronResult]
    consensus_breaks: List[int]    # indices agreed upon by majority of tests
    consensus_dates: List[object]
    summary: str


# ---------------------------------------------------------------------------
# Internal OLS helpers
# ---------------------------------------------------------------------------

def _ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit OLS and return (coefficients, RSS).

    Parameters
    ----------
    X : (n, k) design matrix (should include intercept column)
    y : (n,) dependent variable

    Returns
    -------
    (beta, rss)
    """
    if X.shape[0] < X.shape[1] + 1:
        raise ValueError("Not enough observations for OLS fit.")
    beta, rss_arr, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    rss = float(np.sum((y - yhat) ** 2))
    return beta, rss


def _add_intercept(x: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to x (1-D or 2-D)."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.hstack([np.ones((x.shape[0], 1)), x])


def _classify_regime(mean_ret: float, std_ret: float) -> str:
    """
    Heuristic regime label for a segment given its mean and std.
    Uses BH_MASS_THRESH to gauge volatility relative to a baseline.
    """
    vol_ratio = std_ret / (abs(mean_ret) + 1e-10)
    if mean_ret > 0 and vol_ratio < BH_MASS_THRESH:
        return "BULL_TREND"
    elif mean_ret < 0 and vol_ratio < BH_MASS_THRESH:
        return "BEAR_TREND"
    elif vol_ratio >= BH_MASS_THRESH / BH_DECAY:
        return "HIGH_VOL"
    else:
        return "RANGING"


# ---------------------------------------------------------------------------
# Chow test
# ---------------------------------------------------------------------------

def chow_test(
    y: np.ndarray,
    breakpoint: int,
    X: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
) -> ChowResult:
    """
    Chow test for structural break at a known breakpoint.

    Tests H0: coefficients are equal in the two subsamples split at
    `breakpoint`.  Regresses y on X (default: just an intercept) for the
    full sample (restricted) and each subsample separately (unrestricted).

    F-stat = ((RSS_r - RSS_u) / k) / (RSS_u / (n - 2k))

    Parameters
    ----------
    y          : (n,) dependent variable
    breakpoint : index at which to split (0 < breakpoint < n)
    X          : (n, p) regressors WITHOUT intercept; if None, uses constant
    dates      : DatetimeIndex aligned with y for reporting dates

    Returns
    -------
    ChowResult
    """
    n = len(y)
    if not (1 <= breakpoint <= n - 1):
        raise ValueError(f"breakpoint must be in [1, n-1], got {breakpoint}")

    if X is None:
        X_full = _add_intercept(np.arange(n, dtype=float))
    else:
        X_full = _add_intercept(X)

    k = X_full.shape[1]  # number of params in each sub-model

    # Restricted (full sample)
    _, rss_r = _ols_fit(X_full, y)

    # Unrestricted (two subsamples)
    X1 = X_full[:breakpoint]
    y1 = y[:breakpoint]
    X2 = X_full[breakpoint:]
    y2 = y[breakpoint:]

    _, rss1 = _ols_fit(X1, y1)
    _, rss2 = _ols_fit(X2, y2)
    rss_u = rss1 + rss2

    df1 = k
    df2 = n - 2 * k
    if df2 <= 0:
        warnings.warn("Not enough degrees of freedom for Chow test.")
        f_stat = np.nan
        p_value = np.nan
    else:
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = float(1.0 - stats.f.cdf(f_stat, df1, df2))

    bp_date = dates[breakpoint] if dates is not None else None

    return ChowResult(
        f_stat=float(f_stat),
        p_value=float(p_value),
        breakpoint=breakpoint,
        breakpoint_date=bp_date,
        rss_restricted=float(rss_r),
        rss_unrestricted=float(rss_u),
        n=n,
        k=k,
    )


# ---------------------------------------------------------------------------
# QLR test
# ---------------------------------------------------------------------------

def qlr_test(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    trim: float = 0.15,
    dates: Optional[pd.DatetimeIndex] = None,
) -> QLRResult:
    """
    Quandt Likelihood Ratio test for an unknown single structural break.

    Scans all interior breakpoints in [trim*n, (1-trim)*n] and takes the
    supremum of the Chow F-statistics.  The asymptotic p-value uses the
    Andrews (1993) approximation.

    Parameters
    ----------
    y     : (n,) time series
    X     : regressors (no intercept); None => use linear trend + constant
    trim  : fraction of observations excluded from each end
    dates : DatetimeIndex for reporting

    Returns
    -------
    QLRResult
    """
    n = len(y)
    lo = max(1, int(math.ceil(trim * n)))
    hi = min(n - 1, int(math.floor((1.0 - trim) * n)))
    candidates = np.arange(lo, hi + 1)

    f_stats = np.full(len(candidates), np.nan)
    for i, bp in enumerate(candidates):
        try:
            cr = chow_test(y, int(bp), X=X, dates=dates)
            f_stats[i] = cr.f_stat
        except ValueError:
            pass

    valid = ~np.isnan(f_stats)
    if not np.any(valid):
        raise RuntimeError("No valid Chow F-stats computed in QLR scan.")

    best_i = int(np.nanargmax(f_stats))
    sup_f = float(f_stats[best_i])
    best_bp = int(candidates[best_i])

    # Andrews (1993) asymptotic p-value approximation for k=1 regressors
    # P(sup F > x) ~= c1 * (x ** ((q-1)/2)) * exp(-x/2)
    # For a single structural break with q parameters we use chi-squared
    # approximation: 2*log(sup_F) ~ chi2(1) under H0 (rough)
    k = 2 if X is None else (X.shape[1] + 1)
    p_value = float(1.0 - stats.f.cdf(sup_f, k, n - 2 * k))

    bp_date = dates[best_bp] if dates is not None else None

    return QLRResult(
        sup_f_stat=sup_f,
        breakpoint=best_bp,
        breakpoint_date=bp_date,
        p_value=p_value,
        f_stats=f_stats,
        candidate_indices=candidates,
        trim=trim,
    )


# ---------------------------------------------------------------------------
# CUSUM test
# ---------------------------------------------------------------------------

def cusum_test(
    y: np.ndarray,
    significance: float = 0.05,
    dates: Optional[pd.DatetimeIndex] = None,
) -> CUSUMResult:
    """
    CUSUM structural stability test using recursive OLS residuals.

    Fits an expanding OLS (y ~ 1 + t) and accumulates standardized recursive
    residuals.  A structural break is flagged when the CUSUM statistic exits
    the 5% confidence bands given by the 1.36*sqrt(n) rule (Brown et al. 1975).

    Parameters
    ----------
    y            : (n,) time series
    significance : 0.05 or 0.01 -- only 0.05 supported via 1.36 rule
    dates        : DatetimeIndex

    Returns
    -------
    CUSUMResult
    """
    n = len(y)
    k = 2  # intercept + trend

    # Need at least k+1 obs to start recursion
    min_obs = k + 1
    cusum = np.zeros(n)
    recursive_residuals = np.zeros(n)

    # Compute recursive residuals using the recursive formula
    # w_t = (y_t - x_t' * beta_{t-1}) / sqrt(1 + x_t' (X_{t-1}'X_{t-1})^{-1} x_t)
    sigma_sq_accum = 0.0
    rr_list = []

    for t in range(min_obs, n):
        x_hist = _add_intercept(np.arange(t, dtype=float))
        y_hist = y[:t]
        beta, rss = _ols_fit(x_hist, y_hist)
        x_new = np.array([1.0, float(t)])
        y_pred = float(x_new @ beta)
        # leverage adjustment
        try:
            XtX_inv = np.linalg.inv(x_hist.T @ x_hist)
        except np.linalg.LinAlgError:
            rr_list.append(0.0)
            continue
        h = float(x_new @ XtX_inv @ x_new)
        s2 = rss / max(t - k, 1)
        denom = math.sqrt(s2 * (1.0 + h))
        w = (y[t] - y_pred) / denom if denom > 1e-12 else 0.0
        rr_list.append(w)

    rr = np.array(rr_list)
    # Cumulative sum of recursive residuals normalized by their std
    if len(rr) > 0 and np.std(rr) > 1e-12:
        rr_norm = rr / np.std(rr)
    else:
        rr_norm = rr

    cusum_vals = np.cumsum(rr_norm)
    # Full-length array (zeros for the initial min_obs steps)
    full_cusum = np.zeros(n)
    full_cusum[min_obs:] = cusum_vals

    # Confidence bands: +/- (a + 2*a*(t - n/2)/n) where a = 1.36 for 5%
    # Simpler Brown et al. form: band_t = c * sqrt(n) * (t/n)
    if significance <= 0.01:
        c = 1.63
    elif significance <= 0.05:
        c = 1.36
    else:
        c = 1.07

    t_idx = np.arange(n)
    upper = c * math.sqrt(n) * (t_idx / max(n - 1, 1) + 1.0)
    lower = -upper

    # Detect first exceedance
    breach = np.where(
        (full_cusum > upper) | (full_cusum < lower)
    )[0]
    reject = len(breach) > 0
    first_break = int(breach[0]) if reject else None
    first_date = (dates[first_break] if (reject and dates is not None) else None)

    return CUSUMResult(
        cusum_stats=full_cusum,
        upper_band=upper,
        lower_band=lower,
        reject_null=reject,
        first_break_index=first_break,
        first_break_date=first_date,
        n=n,
        significance=significance,
    )


# ---------------------------------------------------------------------------
# Bai-Perron multiple structural break algorithm
# ---------------------------------------------------------------------------

class BaiPerron:
    """
    Bai-Perron (1998) algorithm for estimating multiple structural breaks.

    Uses dynamic programming to find globally optimal break placements
    for each k (number of breaks), then selects the best k via BIC.

    Parameters
    ----------
    max_breaks : int
        Maximum number of structural breaks to consider (default 5).
    min_segment : int
        Minimum number of observations in each segment (default 5% of n).
    """

    def __init__(self, max_breaks: int = 5, min_segment: Optional[int] = None):
        self.max_breaks = max_breaks
        self.min_segment = min_segment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> BaiPerronResult:
        """
        Estimate optimal break positions.

        Parameters
        ----------
        y     : (n,) time series (e.g., log returns or prices)
        dates : optional DatetimeIndex for break date reporting

        Returns
        -------
        BaiPerronResult
        """
        n = len(y)
        m_seg = self.min_segment if self.min_segment is not None else max(5, int(0.05 * n))

        # Precompute RSS for all possible segments [i, j)
        rss_matrix = self._build_rss_matrix(y, m_seg)

        bic_values: Dict[int, float] = {}
        all_breaks: Dict[int, List[int]] = {}

        # k=0: no breaks -- single segment
        rss_0 = rss_matrix.get((0, n), self._segment_rss(y, 0, n))
        bic_values[0] = self._bic(rss_0, n, 0)
        all_breaks[0] = []

        for k in range(1, min(self.max_breaks, n // m_seg - 1) + 1):
            breaks, rss_k = self._dp_breaks(y, k, m_seg, rss_matrix, n)
            if breaks is None:
                break
            bic_k = self._bic(rss_k, n, k)
            bic_values[k] = bic_k
            all_breaks[k] = breaks

        # Select k* by BIC
        best_k = min(bic_values, key=bic_values.get)
        chosen_breaks = all_breaks[best_k]

        # Build segment statistics
        seg_means, seg_stds = self._segment_stats(y, chosen_breaks, n)

        # Build BaiPerronBreak objects
        bp_objects: List[BaiPerronBreak] = []
        for i, bp_idx in enumerate(chosen_breaks):
            mean_b = seg_means[i] if i < len(seg_means) else 0.0
            std_b = seg_stds[i] if i < len(seg_stds) else 1.0
            mean_a = seg_means[i + 1] if i + 1 < len(seg_means) else 0.0
            std_a = seg_stds[i + 1] if i + 1 < len(seg_stds) else 1.0
            regime_b = _classify_regime(mean_b, std_b)
            regime_a = _classify_regime(mean_a, std_a)
            # Approximate F-stat from CUSUM at break point
            # Use ratio of within-segment variance to global variance
            global_var = np.var(y) + 1e-10
            local_var = ((std_b ** 2) * (bp_idx) + (std_a ** 2) * (n - bp_idx)) / n
            f_approx = global_var / (local_var + 1e-10)
            bp_date = dates[bp_idx] if dates is not None else None
            bp_objects.append(BaiPerronBreak(
                index=bp_idx,
                date=bp_date,
                f_stat=float(f_approx),
                regime_before=regime_b,
                regime_after=regime_a,
            ))

        break_dates = [
            (dates[bp] if dates is not None else None)
            for bp in chosen_breaks
        ]

        return BaiPerronResult(
            n_breaks=best_k,
            break_indices=chosen_breaks,
            break_dates=break_dates,
            bic_values=bic_values,
            segment_means=seg_means,
            segment_stds=seg_stds,
            breaks=bp_objects,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment_rss(self, y: np.ndarray, i: int, j: int) -> float:
        """RSS of mean model on y[i:j]."""
        seg = y[i:j]
        if len(seg) < 2:
            return 0.0
        return float(np.sum((seg - seg.mean()) ** 2))

    def _build_rss_matrix(
        self, y: np.ndarray, m_seg: int
    ) -> Dict[Tuple[int, int], float]:
        """
        Precompute RSS(i, j) for all valid segments.
        Uses O(n^2) time and space -- acceptable for n <= 5000.
        """
        n = len(y)
        rss: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            cumsum = 0.0
            cumsum2 = 0.0
            for j in range(i + 1, n + 1):
                val = y[j - 1]
                cumsum += val
                cumsum2 += val * val
                length = j - i
                if length >= m_seg:
                    mean_sq = (cumsum * cumsum) / length
                    rss[(i, j)] = cumsum2 - mean_sq
        return rss

    def _dp_breaks(
        self,
        y: np.ndarray,
        k: int,
        m_seg: int,
        rss_matrix: Dict[Tuple[int, int], float],
        n: int,
    ) -> Tuple[Optional[List[int]], float]:
        """
        Dynamic programming for k breaks.
        Returns (break_indices, total_rss) or (None, inf) if infeasible.
        """
        # V[t] = minimum RSS using exactly (already placed) breaks up to t
        # We use a list of DP tables for k iterations
        INF = float("inf")

        # DP table: dp[j] = min RSS over all ways to place breaks in y[0:j]
        # with exactly (current_k) breaks placed
        prev = {}
        for j in range(m_seg, n + 1):
            key = (0, j)
            prev[j] = (rss_matrix.get(key, INF), [])

        for b in range(1, k + 1):
            curr = {}
            for j in range((b + 1) * m_seg, n + 1):
                best_cost = INF
                best_trail: List[int] = []
                for t in range(b * m_seg, j - m_seg + 1):
                    if t not in prev:
                        continue
                    seg_cost = rss_matrix.get((t, j), INF)
                    total = prev[t][0] + seg_cost
                    if total < best_cost:
                        best_cost = total
                        best_trail = prev[t][1] + [t]
                if best_cost < INF:
                    curr[j] = (best_cost, best_trail)
            prev = curr

        if n not in prev:
            return None, float("inf")

        total_rss, break_list = prev[n]
        return sorted(break_list), float(total_rss)

    def _bic(self, rss: float, n: int, k: int) -> float:
        """BIC(k) = n * log(RSS/n) + k * log(n)."""
        if rss <= 0 or n <= 0:
            return float("inf")
        return n * math.log(rss / n) + k * math.log(n)

    def _segment_stats(
        self, y: np.ndarray, breaks: List[int], n: int
    ) -> Tuple[List[float], List[float]]:
        """Compute mean and std for each segment defined by breaks."""
        boundaries = [0] + breaks + [n]
        means, stds = [], []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            seg = y[a:b]
            means.append(float(np.mean(seg)) if len(seg) > 0 else 0.0)
            stds.append(float(np.std(seg)) if len(seg) > 1 else 0.0)
        return means, stds


# ---------------------------------------------------------------------------
# StructuralBreakDetector -- main class aggregating all tests
# ---------------------------------------------------------------------------

class StructuralBreakDetector:
    """
    Detects structural breaks in price series using Chow test,
    Quandt Likelihood Ratio (QLR), and CUSUM tests.

    Also runs Bai-Perron for multiple break estimation.

    Parameters
    ----------
    series_name : str
        Label used in reporting (e.g., "SPY" or "BTC_close").
    significance : float
        Significance level for all tests (default 0.05).
    qlr_trim : float
        QLR trim fraction (default 0.15 per Andrews 1993).
    max_breaks : int
        Max breaks for Bai-Perron (default 5).
    """

    def __init__(
        self,
        series_name: str = "series",
        significance: float = 0.05,
        qlr_trim: float = 0.15,
        max_breaks: int = 5,
    ):
        self.series_name = series_name
        self.significance = significance
        self.qlr_trim = qlr_trim
        self.max_breaks = max_breaks
        self._bp_algo = BaiPerron(max_breaks=max_breaks)

    # ------------------------------------------------------------------
    # Individual test wrappers
    # ------------------------------------------------------------------

    def chow_test(
        self,
        y: np.ndarray,
        breakpoint: int,
        X: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> ChowResult:
        """Run Chow test at a single known breakpoint."""
        return chow_test(y, breakpoint, X=X, dates=dates)

    def qlr_test(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> QLRResult:
        """Run QLR test scanning for the most likely single breakpoint."""
        return qlr_test(y, X=X, trim=self.qlr_trim, dates=dates)

    def cusum_test(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> CUSUMResult:
        """Run CUSUM structural stability test."""
        return cusum_test(y, significance=self.significance, dates=dates)

    def bai_perron(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> BaiPerronResult:
        """Run Bai-Perron multiple structural break algorithm."""
        return self._bp_algo.fit(y, dates=dates)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def run_all(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        chow_candidates: Optional[List[int]] = None,
    ) -> BreakReport:
        """
        Run all break tests and return a consolidated BreakReport.

        Parameters
        ----------
        y               : (n,) price or return series
        dates           : DatetimeIndex aligned with y
        chow_candidates : list of breakpoints to test with Chow test;
                          if None, uses the QLR optimal breakpoint

        Returns
        -------
        BreakReport
        """
        n = len(y)

        # QLR -- scan for best single break
        try:
            qlr_res = self.qlr_test(y, dates=dates)
        except Exception as exc:
            warnings.warn(f"QLR failed: {exc}")
            qlr_res = None

        # Chow tests at candidate breakpoints
        if chow_candidates is None:
            chow_candidates = [qlr_res.breakpoint] if qlr_res is not None else []

        chow_results: List[ChowResult] = []
        for bp in chow_candidates:
            try:
                cr = self.chow_test(y, bp, dates=dates)
                chow_results.append(cr)
            except Exception as exc:
                warnings.warn(f"Chow test at {bp} failed: {exc}")

        # CUSUM
        try:
            cusum_res = self.cusum_test(y, dates=dates)
        except Exception as exc:
            warnings.warn(f"CUSUM failed: {exc}")
            cusum_res = None

        # Bai-Perron
        try:
            bp_res = self.bai_perron(y, dates=dates)
        except Exception as exc:
            warnings.warn(f"Bai-Perron failed: {exc}")
            bp_res = None

        # Consensus: collect all significant break indices
        sig_breaks: Dict[int, int] = {}  # index -> vote count

        if qlr_res is not None and qlr_res.p_value < self.significance:
            sig_breaks[qlr_res.breakpoint] = sig_breaks.get(qlr_res.breakpoint, 0) + 1

        for cr in chow_results:
            if cr.p_value < self.significance:
                sig_breaks[cr.breakpoint] = sig_breaks.get(cr.breakpoint, 0) + 1

        if cusum_res is not None and cusum_res.reject_null and cusum_res.first_break_index is not None:
            fb = cusum_res.first_break_index
            sig_breaks[fb] = sig_breaks.get(fb, 0) + 1

        if bp_res is not None:
            for bpb in bp_res.break_indices:
                sig_breaks[bpb] = sig_breaks.get(bpb, 0) + 1

        # Consensus: breaks with >= 2 votes (or all if none reach 2)
        consensus = sorted([idx for idx, v in sig_breaks.items() if v >= 2])
        if not consensus and sig_breaks:
            consensus = sorted(sig_breaks.keys())

        consensus_dates = [
            (dates[i] if dates is not None and 0 <= i < n else None)
            for i in consensus
        ]

        # Summary string
        n_breaks = len(consensus)
        qlr_sig = (
            f"QLR F={qlr_res.sup_f_stat:.2f} p={qlr_res.p_value:.3f}"
            if qlr_res else "QLR N/A"
        )
        cusum_sig = (
            f"CUSUM reject={cusum_res.reject_null}"
            if cusum_res else "CUSUM N/A"
        )
        bp_sig = (
            f"Bai-Perron breaks={bp_res.n_breaks}"
            if bp_res else "Bai-Perron N/A"
        )
        summary = (
            f"{self.series_name}: n={n}, consensus_breaks={n_breaks} | "
            f"{qlr_sig} | {cusum_sig} | {bp_sig}"
        )

        return BreakReport(
            series_name=self.series_name,
            n_obs=n,
            chow_results=chow_results,
            qlr_result=qlr_res,
            cusum_result=cusum_res,
            bai_perron_result=bp_res,
            consensus_breaks=consensus,
            consensus_dates=consensus_dates,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Convenience: returns a DataFrame of break events
    # ------------------------------------------------------------------

    def break_table(self, report: BreakReport) -> pd.DataFrame:
        """
        Convert a BreakReport into a tidy DataFrame with one row per break.

        Columns: source, index, date, f_stat, p_value, regime_before, regime_after
        """
        rows = []

        # QLR
        if report.qlr_result is not None:
            r = report.qlr_result
            rows.append({
                "source": "QLR",
                "index": r.breakpoint,
                "date": r.breakpoint_date,
                "f_stat": r.sup_f_stat,
                "p_value": r.p_value,
                "regime_before": None,
                "regime_after": None,
            })

        # Chow
        for cr in report.chow_results:
            rows.append({
                "source": "Chow",
                "index": cr.breakpoint,
                "date": cr.breakpoint_date,
                "f_stat": cr.f_stat,
                "p_value": cr.p_value,
                "regime_before": None,
                "regime_after": None,
            })

        # CUSUM
        if report.cusum_result is not None and report.cusum_result.reject_null:
            cr2 = report.cusum_result
            rows.append({
                "source": "CUSUM",
                "index": cr2.first_break_index,
                "date": cr2.first_break_date,
                "f_stat": None,
                "p_value": None,
                "regime_before": None,
                "regime_after": None,
            })

        # Bai-Perron
        if report.bai_perron_result is not None:
            for b in report.bai_perron_result.breaks:
                rows.append({
                    "source": "BaiPerron",
                    "index": b.index,
                    "date": b.date,
                    "f_stat": b.f_stat,
                    "p_value": None,
                    "regime_before": b.regime_before,
                    "regime_after": b.regime_after,
                })

        if not rows:
            return pd.DataFrame(columns=[
                "source", "index", "date", "f_stat",
                "p_value", "regime_before", "regime_after",
            ])
        return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
