"""
Regime persistence and transition analysis.

Analyzes how long regimes persist, how frequently they transition,
and which leading indicators predict upcoming regime changes.

Implements:
- Sojourn time distributions per regime
- Geometric MLE for persistence probability
- Empirical transition frequency matrix
- Leading indicator logistic regression
- Regime surprise index (expected vs actual duration)
- PersistenceReport
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# BH / Hurst constants
# ---------------------------------------------------------------------------

BH_MASS_THRESH = 1.92
BH_DECAY = 0.924
HURST_TREND = 0.58
HURST_MR = 0.42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SojournResult:
    """Sojourn (dwell) time distribution for a single regime."""
    regime: str
    durations: List[int]          # observed run lengths in bars
    mean_duration: float
    median_duration: float
    p_geometric: float            # MLE geometric parameter (p = 1/mean)
    p_ci_lower: float             # 95% CI lower
    p_ci_upper: float             # 95% CI upper
    expected_duration: float      # 1 / p_geometric
    n_visits: int                 # number of separate sojourns observed


@dataclass
class TransitionResult:
    """Empirical transition probability matrix."""
    regimes: List[str]
    matrix: pd.DataFrame          # rows = from, cols = to; values = empirical prob
    count_matrix: pd.DataFrame    # raw transition counts
    stationary_dist: Dict[str, float]  # approximate stationary distribution


@dataclass
class LeadingIndicatorResult:
    """Logistic regression results for regime-change predictors."""
    target_transition: str         # e.g., "BULL_TREND -> BEAR_TREND"
    feature_names: List[str]
    coefficients: np.ndarray
    intercept: float
    odds_ratios: np.ndarray
    p_values: np.ndarray           # Wald test p-values
    pseudo_r2: float               # McFadden R^2
    accuracy: float                # in-sample accuracy
    lookback_bars: int


@dataclass
class SurpriseEvent:
    """A single regime surprise event (regime ended much earlier or later)."""
    regime: str
    start_index: int
    end_index: int
    actual_duration: int
    expected_duration: float
    z_score: float                 # (actual - expected) / std_duration


@dataclass
class PersistenceReport:
    """Full regime persistence analysis output."""
    series_name: str
    n_obs: int
    n_regimes: int
    sojourn_results: Dict[str, SojournResult]
    transition_result: TransitionResult
    leading_indicator_results: List[LeadingIndicatorResult]
    surprise_events: List[SurpriseEvent]
    summary: str


# ---------------------------------------------------------------------------
# Sojourn time computation
# ---------------------------------------------------------------------------

def compute_sojourn_times(regime_labels: List[str]) -> Dict[str, List[int]]:
    """
    Compute run-length durations for each distinct regime label.

    Parameters
    ----------
    regime_labels : sequence of string regime labels (one per bar)

    Returns
    -------
    dict mapping regime_name -> list of run lengths (in bars)

    Example
    -------
    labels = ["A","A","B","B","B","A"]
    => {"A": [2, 1], "B": [3]}
    """
    if len(regime_labels) == 0:
        return {}

    durations: Dict[str, List[int]] = {}
    current = regime_labels[0]
    count = 1

    for label in regime_labels[1:]:
        if label == current:
            count += 1
        else:
            durations.setdefault(current, []).append(count)
            current = label
            count = 1
    durations.setdefault(current, []).append(count)

    return durations


# ---------------------------------------------------------------------------
# Geometric distribution MLE
# ---------------------------------------------------------------------------

def fit_geometric_distribution(
    durations: List[int],
) -> Tuple[float, float, float]:
    """
    MLE estimate for the geometric distribution parameter p.

    The geometric distribution models the number of trials until first
    success.  For regime sojourn times, p = probability of leaving regime
    each bar, mean duration = 1/p.

    MLE: p_hat = n / sum(durations)  (same as 1 / sample_mean)

    95% confidence interval via Fisher information:
    Var(p_hat) ~ p^2 * (1-p) / n   =>   CI = p +/- 1.96 * se

    Parameters
    ----------
    durations : list of positive integers

    Returns
    -------
    (p_hat, ci_lower, ci_upper)
    """
    if len(durations) == 0:
        return 0.0, 0.0, 0.0

    n = len(durations)
    total = sum(durations)
    p_hat = n / total  # MLE

    # Standard error via delta method on log(p)
    # Var(p_hat) = p^2 (1-p) / n  (from Fisher info of Geometric)
    var_p = (p_hat ** 2) * (1.0 - p_hat) / n
    se_p = math.sqrt(max(var_p, 1e-12))

    ci_lower = max(0.0, p_hat - 1.96 * se_p)
    ci_upper = min(1.0, p_hat + 1.96 * se_p)

    return float(p_hat), float(ci_lower), float(ci_upper)


# ---------------------------------------------------------------------------
# Transition frequency matrix
# ---------------------------------------------------------------------------

def transition_frequency_matrix(
    labels: List[str],
) -> pd.DataFrame:
    """
    Compute empirical transition probability matrix from a label sequence.

    Parameters
    ----------
    labels : sequence of string regime labels

    Returns
    -------
    DataFrame with rows = from_regime, cols = to_regime, values = P(to|from)
    """
    if len(labels) < 2:
        return pd.DataFrame()

    regimes = sorted(set(labels))
    count_mat = pd.DataFrame(0, index=regimes, columns=regimes, dtype=float)

    for i in range(len(labels) - 1):
        count_mat.loc[labels[i], labels[i + 1]] += 1

    # Normalize rows
    row_sums = count_mat.sum(axis=1)
    prob_mat = count_mat.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return prob_mat


def _compute_stationary(trans_mat: pd.DataFrame) -> Dict[str, float]:
    """
    Compute approximate stationary distribution from transition matrix.
    Uses power iteration: pi' = pi * P until convergence.
    """
    regimes = list(trans_mat.index)
    n = len(regimes)
    if n == 0:
        return {}

    P = trans_mat.values.astype(float)
    pi = np.ones(n) / n

    for _ in range(1000):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < 1e-10:
            break
        pi = pi_new

    return {r: float(pi[i]) for i, r in enumerate(regimes)}


# ---------------------------------------------------------------------------
# Leading indicator logistic regression
# ---------------------------------------------------------------------------

def _compute_features(
    prices: np.ndarray,
    vix: Optional[np.ndarray],
    lookback: int,
    t: int,
) -> Optional[Dict[str, float]]:
    """
    Compute leading indicator features at bar t looking back `lookback` bars.

    Features:
    - momentum     : return over lookback window
    - vol_ratio    : recent vol / longer-term vol
    - hurst_approx : approximate Hurst via variance scaling
    - vix_change   : change in VIX over lookback (if provided)
    - drawdown     : max drawdown over lookback

    Returns None if not enough data.
    """
    start = t - lookback
    if start < 0:
        return None

    window = prices[start:t]
    if len(window) < 4:
        return None

    rets = np.diff(np.log(window + 1e-10))
    momentum = float(np.sum(rets))

    short_vol = float(np.std(rets[-lookback // 4:]) + 1e-10) if lookback >= 4 else 1.0
    long_vol = float(np.std(rets) + 1e-10)
    vol_ratio = short_vol / long_vol

    # Approximate Hurst via variance scaling (using 2 windows)
    half = len(rets) // 2
    if half >= 2:
        var_half = float(np.var(rets[:half]) + 1e-10)
        var_full = float(np.var(rets) + 1e-10)
        hurst_approx = 0.5 * math.log(var_full / var_half) / math.log(2.0) if var_half > 0 else 0.5
    else:
        hurst_approx = 0.5

    running_max = np.maximum.accumulate(window)
    drawdown = float(np.min((window - running_max) / (running_max + 1e-10)))

    feats: Dict[str, float] = {
        "momentum": momentum,
        "vol_ratio": vol_ratio,
        "hurst_approx": float(np.clip(hurst_approx, 0.0, 1.0)),
        "drawdown": drawdown,
    }

    if vix is not None and t < len(vix) and start < len(vix):
        vix_window = vix[start:t]
        feats["vix_change"] = float(vix_window[-1] - vix_window[0]) if len(vix_window) >= 2 else 0.0
        feats["vix_level"] = float(vix_window[-1]) if len(vix_window) >= 1 else 0.0

    return feats


def _logistic_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _logistic_log_likelihood(
    params: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    """Negative log-likelihood for logistic regression (for scipy minimize)."""
    beta = params[:-1]
    intercept = params[-1]
    z = X @ beta + intercept
    p = _logistic_sigmoid(z)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_logistic(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Fit logistic regression via scipy L-BFGS-B.

    Returns (coefficients, intercept, pseudo_r2, p_values)
    """
    from scipy.optimize import minimize

    n, p = X.shape
    init_params = np.zeros(p + 1)

    result = minimize(
        _logistic_log_likelihood,
        init_params,
        args=(X, y),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9},
    )

    params = result.x
    beta = params[:-1]
    intercept = params[-1]
    nll = result.fun

    # Null model LL -- intercept only
    p_bar = np.mean(y)
    p_bar = np.clip(p_bar, 1e-10, 1 - 1e-10)
    null_ll = -(n * p_bar * math.log(p_bar) + n * (1 - p_bar) * math.log(1 - p_bar))
    pseudo_r2 = 1.0 - nll / null_ll if null_ll > 0 else 0.0

    # Wald test p-values via Hessian approximation (diagonal of Fisher info)
    # Use numerical Hessian from scipy
    try:
        from scipy.optimize import approx_fprime

        def grad_func(params_):
            return approx_fprime(params_, _logistic_log_likelihood, 1e-5, X, y)

        # Numerical Hessian -- finite diff of gradient
        hess = np.zeros((p + 1, p + 1))
        eps = 1e-5
        for i in range(p + 1):
            e = np.zeros(p + 1)
            e[i] = eps
            hess[:, i] = (grad_func(params + e) - grad_func(params - e)) / (2 * eps)

        var_beta = np.diag(np.linalg.pinv(hess + np.eye(p + 1) * 1e-8))
        se_beta = np.sqrt(np.abs(var_beta[:-1]))
        z_scores = beta / (se_beta + 1e-10)
        p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))
    except Exception:
        p_values = np.full(p, np.nan)

    return beta, float(intercept), float(pseudo_r2), p_values


# ---------------------------------------------------------------------------
# Regime surprise index
# ---------------------------------------------------------------------------

def _compute_surprise_events(
    regime_labels: List[str],
    sojourn_results: Dict[str, SojournResult],
) -> List[SurpriseEvent]:
    """
    Identify sojourn episodes where actual duration differed
    significantly from the expected (geometric mean) duration.
    """
    if len(regime_labels) == 0:
        return []

    events: List[SurpriseEvent] = []
    current = regime_labels[0]
    start_idx = 0

    for i, label in enumerate(regime_labels[1:], start=1):
        if label != current:
            actual = i - start_idx
            sr = sojourn_results.get(current)
            if sr is not None and sr.expected_duration > 0:
                std_dur = sr.expected_duration * math.sqrt(
                    (1.0 - sr.p_geometric) / (sr.p_geometric ** 2 + 1e-10)
                )
                z = (actual - sr.expected_duration) / (std_dur + 1e-1)
                if abs(z) >= 1.5:
                    events.append(SurpriseEvent(
                        regime=current,
                        start_index=start_idx,
                        end_index=i - 1,
                        actual_duration=actual,
                        expected_duration=sr.expected_duration,
                        z_score=float(z),
                    ))
            current = label
            start_idx = i

    return events


# ---------------------------------------------------------------------------
# RegimeLeadingIndicators helper
# ---------------------------------------------------------------------------

class RegimeLeadingIndicators:
    """
    For each observed regime transition, look back `lookback_bars` bars
    and compute features; then fit logistic regression to predict which
    transitions are predictable.

    Parameters
    ----------
    lookback_bars : int
        Number of bars to look back for feature computation (default 10).
    """

    def __init__(self, lookback_bars: int = 10):
        self.lookback_bars = lookback_bars

    def fit(
        self,
        regime_labels: List[str],
        prices: np.ndarray,
        vix: Optional[np.ndarray] = None,
    ) -> List[LeadingIndicatorResult]:
        """
        Fit logistic regression models for each regime transition type.

        Parameters
        ----------
        regime_labels : list of per-bar regime labels
        prices        : (n,) price series aligned with labels
        vix           : optional (n,) VIX series

        Returns
        -------
        list of LeadingIndicatorResult (one per transition type found)
        """
        n = len(regime_labels)
        if n < self.lookback_bars + 5:
            return []

        # Find transition indices
        transitions = []
        for i in range(1, n):
            if regime_labels[i] != regime_labels[i - 1]:
                transitions.append((i, regime_labels[i - 1], regime_labels[i]))

        if len(transitions) < 5:
            return []

        # Group transitions by (from, to) pair
        transition_types: Dict[str, List[int]] = {}
        for t_idx, from_r, to_r in transitions:
            key = f"{from_r} -> {to_r}"
            transition_types.setdefault(key, []).append(t_idx)

        results: List[LeadingIndicatorResult] = []

        for transition_key, pos_indices in transition_types.items():
            if len(pos_indices) < 3:
                continue

            # Build feature matrix
            # Positive labels: bars at transition points
            # Negative labels: randomly sampled non-transition bars
            all_features: List[Dict[str, float]] = []
            all_labels: List[int] = []

            # Positive examples
            for t_idx in pos_indices:
                feats = _compute_features(prices, vix, self.lookback_bars, t_idx)
                if feats is not None:
                    all_features.append(feats)
                    all_labels.append(1)

            # Negative examples -- sample equal number from non-transition bars
            non_trans = [
                i for i in range(self.lookback_bars, n)
                if i not in set(t for t, _, _ in transitions)
            ]
            rng = np.random.default_rng(42)
            neg_sample = rng.choice(
                non_trans,
                size=min(len(pos_indices) * 2, len(non_trans)),
                replace=False,
            )
            for t_idx in neg_sample:
                feats = _compute_features(prices, vix, self.lookback_bars, int(t_idx))
                if feats is not None:
                    all_features.append(feats)
                    all_labels.append(0)

            if len(all_features) < 10:
                continue

            # Convert to arrays
            feature_names = sorted(all_features[0].keys())
            X = np.array([[f.get(k, 0.0) for k in feature_names] for f in all_features])
            y = np.array(all_labels, dtype=float)

            # Standardize X
            x_mean = X.mean(axis=0)
            x_std = X.std(axis=0) + 1e-10
            X_scaled = (X - x_mean) / x_std

            try:
                beta, intercept, pseudo_r2, p_values = _fit_logistic(X_scaled, y)
            except Exception as exc:
                warnings.warn(f"Logistic fit failed for {transition_key}: {exc}")
                continue

            odds_ratios = np.exp(beta)
            z = X_scaled @ beta + intercept
            y_pred = (_logistic_sigmoid(z) >= 0.5).astype(float)
            accuracy = float(np.mean(y_pred == y))

            results.append(LeadingIndicatorResult(
                target_transition=transition_key,
                feature_names=feature_names,
                coefficients=beta,
                intercept=intercept,
                odds_ratios=odds_ratios,
                p_values=p_values,
                pseudo_r2=float(pseudo_r2),
                accuracy=accuracy,
                lookback_bars=self.lookback_bars,
            ))

        return results


# ---------------------------------------------------------------------------
# RegimePersistenceAnalyzer -- main class
# ---------------------------------------------------------------------------

class RegimePersistenceAnalyzer:
    """
    Analyzes how long regimes persist, transition frequencies,
    and the leading indicators that predict regime changes.

    Parameters
    ----------
    series_name   : label for the analyzed series
    lookback_bars : lookback for leading indicator features (default 10)
    """

    def __init__(
        self,
        series_name: str = "series",
        lookback_bars: int = 10,
    ):
        self.series_name = series_name
        self.lookback_bars = lookback_bars
        self._leading_ind = RegimeLeadingIndicators(lookback_bars=lookback_bars)

    # ------------------------------------------------------------------
    # Core analysis methods (also exposed for direct use)
    # ------------------------------------------------------------------

    def compute_sojourn_times(
        self, regime_labels: List[str]
    ) -> Dict[str, List[int]]:
        """Return {regime: [duration_bars]} for each regime type."""
        return compute_sojourn_times(regime_labels)

    def fit_geometric_distribution(
        self, durations: List[int]
    ) -> Tuple[float, float, float]:
        """MLE p parameter and 95% CI."""
        return fit_geometric_distribution(durations)

    def transition_frequency_matrix(
        self, labels: List[str]
    ) -> pd.DataFrame:
        """N x N matrix of empirical transition probabilities."""
        return transition_frequency_matrix(labels)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def analyze(
        self,
        regime_labels: List[str],
        prices: Optional[np.ndarray] = None,
        vix: Optional[np.ndarray] = None,
    ) -> PersistenceReport:
        """
        Run full persistence analysis.

        Parameters
        ----------
        regime_labels : list of per-bar regime strings
        prices        : (n,) price series for leading indicator features
        vix           : optional (n,) VIX or volatility series

        Returns
        -------
        PersistenceReport
        """
        n = len(regime_labels)
        if n == 0:
            raise ValueError("regime_labels is empty")

        unique_regimes = sorted(set(regime_labels))

        # Sojourn times
        sojourn_raw = compute_sojourn_times(regime_labels)
        sojourn_results: Dict[str, SojournResult] = {}
        for regime, durs in sojourn_raw.items():
            p_hat, ci_lo, ci_hi = fit_geometric_distribution(durs)
            exp_dur = 1.0 / p_hat if p_hat > 0 else float("inf")
            sojourn_results[regime] = SojournResult(
                regime=regime,
                durations=durs,
                mean_duration=float(np.mean(durs)),
                median_duration=float(np.median(durs)),
                p_geometric=p_hat,
                p_ci_lower=ci_lo,
                p_ci_upper=ci_hi,
                expected_duration=exp_dur,
                n_visits=len(durs),
            )

        # Transition matrix
        prob_mat = transition_frequency_matrix(regime_labels)
        count_mat_raw = pd.DataFrame(0, index=list(prob_mat.index), columns=list(prob_mat.columns), dtype=float)
        # Recompute count matrix
        for i in range(n - 1):
            fr, to = regime_labels[i], regime_labels[i + 1]
            if fr in count_mat_raw.index and to in count_mat_raw.columns:
                count_mat_raw.loc[fr, to] += 1

        stationary = _compute_stationary(prob_mat) if not prob_mat.empty else {}
        trans_result = TransitionResult(
            regimes=list(prob_mat.index),
            matrix=prob_mat,
            count_matrix=count_mat_raw,
            stationary_dist=stationary,
        )

        # Leading indicators
        leading_results: List[LeadingIndicatorResult] = []
        if prices is not None and len(prices) >= self.lookback_bars + 5:
            try:
                leading_results = self._leading_ind.fit(regime_labels, prices, vix)
            except Exception as exc:
                warnings.warn(f"Leading indicator fitting failed: {exc}")

        # Surprise events
        surprises = _compute_surprise_events(regime_labels, sojourn_results)

        # Summary
        avg_durs = {r: f"{sr.mean_duration:.1f}" for r, sr in sojourn_results.items()}
        summary = (
            f"{self.series_name}: n={n}, regimes={unique_regimes}, "
            f"avg_durations={avg_durs}, "
            f"n_surprises={len(surprises)}, "
            f"n_transitions_modeled={len(leading_results)}"
        )

        return PersistenceReport(
            series_name=self.series_name,
            n_obs=n,
            n_regimes=len(unique_regimes),
            sojourn_results=sojourn_results,
            transition_result=trans_result,
            leading_indicator_results=leading_results,
            surprise_events=surprises,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def sojourn_table(self, report: PersistenceReport) -> pd.DataFrame:
        """
        Return a tidy DataFrame of sojourn statistics per regime.
        """
        rows = []
        for regime, sr in report.sojourn_results.items():
            rows.append({
                "regime": regime,
                "n_visits": sr.n_visits,
                "mean_duration": sr.mean_duration,
                "median_duration": sr.median_duration,
                "p_geometric": sr.p_geometric,
                "p_ci_lower": sr.p_ci_lower,
                "p_ci_upper": sr.p_ci_upper,
                "expected_duration": sr.expected_duration,
            })
        return pd.DataFrame(rows).set_index("regime")

    def leading_indicator_table(self, report: PersistenceReport) -> pd.DataFrame:
        """
        Return feature coefficients for all modeled regime transitions.
        """
        rows = []
        for li in report.leading_indicator_results:
            for feat, coef, or_, pv in zip(
                li.feature_names,
                li.coefficients,
                li.odds_ratios,
                li.p_values,
            ):
                rows.append({
                    "transition": li.target_transition,
                    "feature": feat,
                    "coefficient": float(coef),
                    "odds_ratio": float(or_),
                    "p_value": float(pv),
                    "pseudo_r2": li.pseudo_r2,
                    "accuracy": li.accuracy,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def surprise_table(self, report: PersistenceReport) -> pd.DataFrame:
        """Return a DataFrame of regime surprise events sorted by |z_score|."""
        rows = [
            {
                "regime": e.regime,
                "start_index": e.start_index,
                "end_index": e.end_index,
                "actual_duration": e.actual_duration,
                "expected_duration": round(e.expected_duration, 2),
                "z_score": round(e.z_score, 3),
            }
            for e in report.surprise_events
        ]
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .assign(abs_z=lambda df: df["z_score"].abs())
            .sort_values("abs_z", ascending=False)
            .drop(columns=["abs_z"])
            .reset_index(drop=True)
        )
