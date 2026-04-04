"""
Change point detection algorithms for financial time series.

Implements:
- CUSUM (Cumulative Sum) control chart
- PELT (Pruned Exact Linear Time) with Normal/Poisson/L2 cost functions
- Bayesian Online Change Point Detection (Adams & MacKay 2007)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChangePointResult:
    """Results of a change point detection algorithm."""
    change_points: List[int]       # indices in the original series
    change_point_dates: List       # corresponding dates if index is DatetimeIndex
    n_segments: int
    segment_stats: pd.DataFrame    # mean, std, start, end per segment
    algorithm: str
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

class CUSUM:
    """
    CUSUM (Cumulative Sum) change-point detector.

    Detects shifts in the mean of a series.  Two-sided: detects both
    positive and negative mean shifts.

    Parameters
    ----------
    threshold : float
        Decision threshold h.  A change point is declared when the CUSUM
        statistic exceeds h.
    drift : float
        Allowance k = delta/2, where delta is the minimum detectable shift
        in standard deviations.
    reset_after_detection : bool
        If True, CUSUM resets to 0 after each detection.
    min_segment_length : int
        Minimum number of observations between change points.
    """

    def __init__(
        self,
        threshold: float = 4.0,
        drift: float = 0.5,
        reset_after_detection: bool = True,
        min_segment_length: int = 20,
    ) -> None:
        self.threshold = threshold
        self.drift = drift
        self.reset_after_detection = reset_after_detection
        self.min_segment_length = min_segment_length

    def detect(self, series: pd.Series) -> ChangePointResult:
        """
        Run CUSUM on the series.

        Parameters
        ----------
        series : pd.Series
            Univariate time series.

        Returns
        -------
        ChangePointResult
        """
        x = series.dropna().values
        n = len(x)
        mu_hat = x.mean()
        sigma_hat = x.std(ddof=1)
        if sigma_hat == 0:
            sigma_hat = 1.0

        z = (x - mu_hat) / sigma_hat  # standardize
        k = self.drift

        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        change_points = []
        last_cp = 0

        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + z[i] - k)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - z[i] - k)

            above_threshold = (
                cusum_pos[i] >= self.threshold or cusum_neg[i] >= self.threshold
            )
            sufficient_length = (i - last_cp) >= self.min_segment_length

            if above_threshold and sufficient_length:
                change_points.append(i)
                last_cp = i
                if self.reset_after_detection:
                    cusum_pos[i] = 0.0
                    cusum_neg[i] = 0.0

        idx = series.dropna().index
        cp_dates = [idx[cp] for cp in change_points if cp < len(idx)]
        segment_stats = self._compute_segment_stats(series, change_points)

        return ChangePointResult(
            change_points=change_points,
            change_point_dates=cp_dates,
            n_segments=len(change_points) + 1,
            segment_stats=segment_stats,
            algorithm="CUSUM",
            extra={"cusum_pos": cusum_pos, "cusum_neg": cusum_neg},
        )

    def _compute_segment_stats(
        self, series: pd.Series, change_points: List[int]
    ) -> pd.DataFrame:
        x = series.dropna()
        boundaries = [0] + change_points + [len(x)]
        rows = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = x.iloc[start:end]
            rows.append({
                "segment": i + 1,
                "start_idx": start,
                "end_idx": end - 1,
                "start_date": x.index[start],
                "end_date": x.index[end - 1],
                "mean": round(float(segment.mean()), 6),
                "std": round(float(segment.std()), 6),
                "n_obs": len(segment),
            })
        return pd.DataFrame(rows).set_index("segment")

    def cusum_statistics(
        self, series: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Return full CUSUM+ and CUSUM- statistic series.

        Returns
        -------
        (cusum_pos, cusum_neg) as pd.Series
        """
        result = self.detect(series)
        x = series.dropna()
        idx = x.index
        return (
            pd.Series(result.extra["cusum_pos"], index=idx),
            pd.Series(result.extra["cusum_neg"], index=idx),
        )


# ---------------------------------------------------------------------------
# PELT
# ---------------------------------------------------------------------------

class PELT:
    """
    PELT (Pruned Exact Linear Time) change point detection.
    (Killick, Fearnhead & Eckley 2012)

    Minimizes:
      sum_{j=1}^{m+1} [C(y_{tau_{j-1}+1:tau_j})] + beta * m

    where C() is the segment cost function and beta is the penalty.

    Parameters
    ----------
    model : str
        Cost function: 'normal' (Gaussian with unknown mean & var),
        'l2' (squared error around segment mean), 'rbf'.
    penalty : str or float
        'bic' — BIC penalty = log(T), 'aic' — 2,
        or a numeric value for the penalty per change point.
    min_size : int
        Minimum segment length.
    jump : int
        Step size when searching candidates (trade-off accuracy vs speed).
    """

    def __init__(
        self,
        model: str = "normal",
        penalty: float | str = "bic",
        min_size: int = 10,
        jump: int = 1,
    ) -> None:
        self.model = model
        self.penalty = penalty
        self.min_size = min_size
        self.jump = jump

    def _segment_cost(self, x: np.ndarray, start: int, end: int) -> float:
        """Cost of assigning x[start:end] to one segment."""
        segment = x[start:end]
        n = len(segment)
        if n < 2:
            return 1e10

        if self.model == "normal":
            # Negative log-likelihood of Gaussian
            mu = segment.mean()
            sigma2 = segment.var(ddof=1)
            if sigma2 <= 0:
                sigma2 = 1e-8
            return n * np.log(sigma2) + n  # proportional to -2*loglik

        elif self.model == "l2":
            mu = segment.mean()
            return float(np.sum((segment - mu) ** 2))

        elif self.model == "rbf":
            # Radial basis function cost
            mu = segment.mean()
            return float(np.sum((segment - mu) ** 2)) / (n + 1)

        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _get_penalty(self, T: int) -> float:
        if isinstance(self.penalty, str):
            if self.penalty == "bic":
                return np.log(T)
            elif self.penalty == "aic":
                return 2.0
            else:
                raise ValueError(f"Unknown penalty: {self.penalty}")
        return float(self.penalty)

    def detect(self, series: pd.Series) -> ChangePointResult:
        """
        Detect change points using PELT.

        Parameters
        ----------
        series : pd.Series

        Returns
        -------
        ChangePointResult
        """
        x = series.dropna().values
        T = len(x)
        beta = self._get_penalty(T)

        # Cost matrix cache (sparse: only compute when needed)
        cost_cache: Dict[Tuple[int, int], float] = {}

        def cost(s, e):
            key = (s, e)
            if key not in cost_cache:
                cost_cache[key] = self._segment_cost(x, s, e)
            return cost_cache[key]

        # Dynamic programming
        F = np.full(T + 1, np.inf)
        F[0] = -beta  # F[0] = -beta so F[t] = cost(0,t) + 0
        cp_last = [0] * (T + 1)
        admissible = [0]  # candidate change point positions

        for t in range(self.min_size, T + 1):
            candidates = [s for s in admissible if s <= t - self.min_size]
            if not candidates:
                candidates = [0]

            costs = np.array([F[s] + cost(s, t) + beta for s in candidates])
            best_idx = int(np.argmin(costs))
            F[t] = costs[best_idx]
            cp_last[t] = candidates[best_idx]

            # Pruning: remove candidates s where F[s] + cost(s,t) > F[t] for all t' > t
            pruned = []
            for s in admissible:
                if s <= t - self.min_size and F[s] + cost(s, t) <= F[t] + beta:
                    pruned.append(s)
            # Also add current t if applicable
            if t >= self.min_size:
                pruned.append(t)
            admissible = pruned

        # Backtrack
        change_points = []
        t = T
        while cp_last[t] != 0:
            change_points.append(cp_last[t])
            t = cp_last[t]
        change_points = sorted(change_points)

        idx = series.dropna().index
        cp_dates = [idx[cp] for cp in change_points if cp < len(idx)]
        segment_stats = self._compute_segment_stats(series, change_points)

        return ChangePointResult(
            change_points=change_points,
            change_point_dates=cp_dates,
            n_segments=len(change_points) + 1,
            segment_stats=segment_stats,
            algorithm="PELT",
            extra={"F": F, "cost_cache_size": len(cost_cache)},
        )

    def _compute_segment_stats(
        self, series: pd.Series, change_points: List[int]
    ) -> pd.DataFrame:
        x = series.dropna()
        boundaries = [0] + change_points + [len(x)]
        rows = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = x.iloc[start:end]
            rows.append({
                "segment": i + 1,
                "start_idx": start,
                "end_idx": end - 1,
                "start_date": x.index[start],
                "end_date": x.index[end - 1],
                "mean": round(float(segment.mean()), 6),
                "std": round(float(segment.std()), 6),
                "n_obs": len(segment),
            })
        return pd.DataFrame(rows).set_index("segment")

    def optimal_n_changepoints(
        self, series: pd.Series, max_cp: int = 10
    ) -> pd.DataFrame:
        """
        Fit with varying penalty and report number of detected change points.

        Returns
        -------
        pd.DataFrame
            penalty, n_changepoints, total_cost per row.
        """
        T = len(series.dropna())
        bic_penalty = np.log(T)
        penalties = np.linspace(bic_penalty * 0.2, bic_penalty * 3, 15)

        rows = []
        for p in penalties:
            self.penalty = float(p)
            result = self.detect(series)
            rows.append({
                "penalty": round(p, 4),
                "n_changepoints": len(result.change_points),
                "n_segments": result.n_segments,
            })
        self.penalty = "bic"  # reset
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bayesian Online Change Point Detection
# ---------------------------------------------------------------------------

class BayesianOnlineCP:
    """
    Bayesian Online Change Point Detection (Adams & MacKay 2007).

    Models run lengths and detects structural breaks in real-time.
    Uses a Gaussian conjugate model with unknown mean and variance.

    Parameters
    ----------
    hazard_rate : float
        Prior probability of a change point at any given time step.
        Equivalent to 1 / expected_run_length.
    alpha0 : float
        Prior concentration for the Normal-Gamma prior.
    beta0 : float
        Prior rate for the Normal-Gamma prior.
    mu0 : float or None
        Prior mean (None = use data mean).
    kappa0 : float
        Prior precision scaling factor.
    threshold : float
        Probability threshold to declare a change point.
    """

    def __init__(
        self,
        hazard_rate: float = 0.01,
        alpha0: float = 0.1,
        beta0: float = 0.1,
        mu0: Optional[float] = None,
        kappa0: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        self.hazard_rate = hazard_rate
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.threshold = threshold

    def _student_t_log_pdf(
        self,
        x: float,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        kappa: np.ndarray,
    ) -> np.ndarray:
        """
        Log pdf of Student-t predictive distribution (Normal-Gamma conjugate).

        p(x | mu, kappa, alpha, beta) = Student-t(2*alpha, mu, beta*(kappa+1)/(alpha*kappa))
        """
        nu = 2 * alpha
        scale2 = beta * (kappa + 1) / (alpha * kappa + 1e-12)
        scale2 = np.maximum(scale2, 1e-10)
        # Student-t log pdf
        log_norm = (
            np.lgamma((nu + 1) / 2)
            - np.lgamma(nu / 2)
            - 0.5 * np.log(nu * np.pi * scale2)
        )
        log_pdf = log_norm - (nu + 1) / 2 * np.log(1 + (x - mu) ** 2 / (nu * scale2))
        return log_pdf

    def detect(self, series: pd.Series) -> ChangePointResult:
        """
        Run BOCPD on series and return change point detections.

        Parameters
        ----------
        series : pd.Series

        Returns
        -------
        ChangePointResult
        """
        x = series.dropna().values
        T = len(x)

        # Initialize prior
        mu0 = self.mu0 if self.mu0 is not None else float(x.mean())
        alpha0 = self.alpha0
        beta0 = self.beta0
        kappa0 = self.kappa0

        # Run-length distribution: R[t, l] = P(run length = l at time t)
        # We use a 1D vector and update in place
        log_R = np.array([-np.inf])  # log P(r_t = 0) = log(1)
        log_R[0] = 0.0

        # Sufficient statistics per run length
        mu_arr = np.array([mu0])
        kappa_arr = np.array([kappa0])
        alpha_arr = np.array([alpha0])
        beta_arr = np.array([beta0])

        change_point_probs = np.zeros(T)
        run_length_dist = np.zeros((T, T))  # run_length_dist[t, l]

        change_points = []

        for t in range(T):
            xt = x[t]

            # Predictive log probs for each run length
            log_pred = self._student_t_log_pdf(xt, mu_arr, alpha_arr, beta_arr, kappa_arr)

            # Joint: P(r_t, x_1:t) = P(x_t | r_t, x_{1:t-1}) * P(r_t | r_{t-1}) * P(r_{t-1}, x_{1:t-1})
            log_joint = log_R + log_pred

            # Growth probability (r_t = r_{t-1} + 1)
            log_growth = log_joint + np.log(1 - self.hazard_rate)

            # Change point probability (r_t = 0)
            log_cp = np.logaddexp.reduce(log_joint) + np.log(self.hazard_rate)

            # New log_R: concat [log_cp, log_growth]
            log_R_new = np.concatenate([[log_cp], log_growth])

            # Normalize
            log_norm = np.logaddexp.reduce(log_R_new)
            log_R = log_R_new - log_norm

            # Change point probability at time t
            cp_prob = float(np.exp(log_R[0]))
            change_point_probs[t] = cp_prob

            if cp_prob > self.threshold and t > 0:
                change_points.append(t)

            # Update sufficient statistics
            kappa_new = np.concatenate([[kappa0], kappa_arr + 1])
            mu_new = np.concatenate([
                [mu0],
                (kappa_arr * mu_arr + xt) / (kappa_arr + 1)
            ])
            alpha_new = np.concatenate([[alpha0], alpha_arr + 0.5])
            beta_new = np.concatenate([
                [beta0],
                beta_arr + 0.5 * kappa_arr / (kappa_arr + 1) * (xt - mu_arr) ** 2
            ])

            mu_arr = mu_new
            kappa_arr = kappa_new
            alpha_arr = alpha_new
            beta_arr = beta_new

        # Deduplicate change points (keep only local maxima)
        filtered_cps = []
        min_gap = 5
        last_cp = -min_gap
        for cp in change_points:
            if cp - last_cp >= min_gap:
                filtered_cps.append(cp)
                last_cp = cp

        idx = series.dropna().index
        cp_dates = [idx[cp] for cp in filtered_cps if cp < len(idx)]

        # Build segment stats
        x_series = series.dropna()
        boundaries = [0] + filtered_cps + [len(x_series)]
        rows = []
        for i in range(len(boundaries) - 1):
            s = boundaries[i]
            e = boundaries[i + 1]
            seg = x_series.iloc[s:e]
            rows.append({
                "segment": i + 1,
                "start_idx": s,
                "end_idx": e - 1,
                "start_date": x_series.index[s],
                "end_date": x_series.index[e - 1],
                "mean": round(float(seg.mean()), 6),
                "std": round(float(seg.std()), 6),
                "n_obs": len(seg),
            })
        seg_df = pd.DataFrame(rows).set_index("segment") if rows else pd.DataFrame()

        return ChangePointResult(
            change_points=filtered_cps,
            change_point_dates=cp_dates,
            n_segments=len(filtered_cps) + 1,
            segment_stats=seg_df,
            algorithm="BOCPD",
            extra={"cp_probs": pd.Series(change_point_probs, index=idx)},
        )

    def online_probabilities(self, series: pd.Series) -> pd.Series:
        """
        Return the full time series of change point probabilities.

        Returns
        -------
        pd.Series
        """
        result = self.detect(series)
        return result.extra["cp_probs"]


# ---------------------------------------------------------------------------
# Utility: compare detectors
# ---------------------------------------------------------------------------

def compare_detectors(
    series: pd.Series,
    cusum_kwargs: Optional[Dict] = None,
    pelt_kwargs: Optional[Dict] = None,
    bocpd_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Run all three detectors and return a summary comparison.

    Returns
    -------
    pd.DataFrame
        n_changepoints, algorithm, first_cp_date, last_cp_date per algorithm.
    """
    results = []

    cusum = CUSUM(**(cusum_kwargs or {}))
    pelt = PELT(**(pelt_kwargs or {}))
    bocpd = BayesianOnlineCP(**(bocpd_kwargs or {}))

    for algo, detector in [("CUSUM", cusum), ("PELT", pelt), ("BOCPD", bocpd)]:
        result = detector.detect(series)
        results.append({
            "algorithm": algo,
            "n_changepoints": len(result.change_points),
            "n_segments": result.n_segments,
            "first_cp": result.change_point_dates[0] if result.change_point_dates else None,
            "last_cp": result.change_point_dates[-1] if result.change_point_dates else None,
        })

    return pd.DataFrame(results).set_index("algorithm")
