"""
causal/python/granger/granger_tests.py

Granger causality tests over the causal feature matrix.

Algorithm:
    1. For each ordered pair (X → Y), fit a VAR model on [X, Y].
    2. Use AIC/BIC to select optimal lag up to max_lag.
    3. Run statsmodels Granger causality test (F-test + chi2).
    4. Collect p-values for all pairs.
    5. Apply FDR correction (Benjamini-Hochberg) for multiple testing.
    6. Return directed edges with p-values, effect sizes, and optimal lags.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GrangerEdge:
    """A directed causal edge X → Y discovered by Granger test."""
    cause: str         # feature name X (the cause)
    effect: str        # feature name Y (the effect)
    optimal_lag: int   # lag at which the relationship is strongest
    p_value: float     # BH-corrected p-value
    raw_p_value: float # uncorrected p-value
    f_statistic: float # F-statistic from the Granger test
    effect_size: float # Cohen's f^2 = (R2_restricted - R2_full) / (1 - R2_full) proxy
    significant: bool  # True if p_value < significance_threshold
    aic_at_lag: float  # AIC of the VAR model at optimal_lag
    bic_at_lag: float  # BIC of the VAR model at optimal_lag

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "optimal_lag": self.optimal_lag,
            "p_value": self.p_value,
            "raw_p_value": self.raw_p_value,
            "f_statistic": self.f_statistic,
            "effect_size": self.effect_size,
            "significant": self.significant,
            "aic_at_lag": self.aic_at_lag,
            "bic_at_lag": self.bic_at_lag,
        }


@dataclass
class GrangerResult:
    """Container for all pairwise Granger test results."""
    edges: list[GrangerEdge]
    features_tested: list[str]
    max_lag: int
    significance_threshold: float
    n_observations: int
    n_pairs_tested: int
    n_significant: int
    fdr_method: str

    @property
    def significant_edges(self) -> list[GrangerEdge]:
        return [e for e in self.edges if e.significant]

    @property
    def as_dataframe(self) -> pd.DataFrame:
        if not self.edges:
            return pd.DataFrame()
        return pd.DataFrame([e.to_dict() for e in self.edges])

    def get_causes_of(self, effect: str) -> list[GrangerEdge]:
        return [e for e in self.significant_edges if e.effect == effect]

    def get_effects_of(self, cause: str) -> list[GrangerEdge]:
        return [e for e in self.significant_edges if e.cause == cause]


# ---------------------------------------------------------------------------
# FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def _fdr_bh_correction(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[float], list[bool]]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.
    Returns (adjusted_p_values, rejected_null) for the input list order.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha

    # Find the largest k where p_(k) <= k/n * alpha
    reject = sorted_p <= thresholds
    if reject.any():
        max_k = np.where(reject)[0].max()
        reject[:max_k + 1] = True

    # Compute adjusted p-values: p_adj(i) = min(p_i * n / i, 1)
    adjusted = np.minimum(sorted_p * n / (np.arange(1, n + 1)), 1.0)
    # Monotone: p_adj(i) = min(p_adj(i:n))
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Restore original order
    result_p = np.empty(n)
    result_reject = np.empty(n, dtype=bool)
    result_p[order] = adjusted
    result_reject[order] = reject

    return result_p.tolist(), result_reject.tolist()


# ---------------------------------------------------------------------------
# Lag selection via IC
# ---------------------------------------------------------------------------

def _select_optimal_lag(
    data: pd.DataFrame, max_lag: int, ic: str = "aic"
) -> int:
    """
    Fit a VAR(p) model for p in 1..max_lag and return lag with minimum AIC or BIC.
    Falls back to lag=1 on any failure.
    """
    best_lag = 1
    best_ic = np.inf
    n = len(data)

    for lag in range(1, max_lag + 1):
        if n <= lag * data.shape[1] + lag + 5:
            break  # not enough observations
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = VAR(data)
                result = model.fit(maxlags=lag, ic=None, trend="n")
                ic_val = result.aic if ic == "aic" else result.bic
            if ic_val < best_ic:
                best_ic = ic_val
                best_lag = lag
        except Exception:
            continue

    return best_lag


def _get_var_ic(data: pd.DataFrame, lag: int) -> tuple[float, float]:
    """Return (AIC, BIC) for VAR(lag) on data."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(data)
            result = model.fit(maxlags=lag, ic=None, trend="n")
            return float(result.aic), float(result.bic)
    except Exception:
        return (np.inf, np.inf)


# ---------------------------------------------------------------------------
# Effect size estimation
# ---------------------------------------------------------------------------

def _compute_effect_size(
    cause: pd.Series,
    effect: pd.Series,
    lag: int,
) -> float:
    """
    Approximate Cohen's f^2 for Granger causality:
    f^2 = (R2_full - R2_reduced) / (1 - R2_full)
    where R2_full includes lags of cause, R2_reduced does not.
    """
    try:
        n = len(effect) - lag
        if n < 10:
            return 0.0

        y = effect.values[lag:]
        y_mean = y.mean()
        ss_total = np.sum((y - y_mean) ** 2)
        if ss_total < 1e-12:
            return 0.0

        # Reduced model: autoregressive lags of y only
        X_reduced = np.column_stack([
            effect.values[lag - k - 1: n + lag - k - 1]
            for k in range(lag)
        ])
        X_reduced = np.column_stack([np.ones(n), X_reduced])

        # Full model: also include lags of cause
        X_cause = np.column_stack([
            cause.values[lag - k - 1: n + lag - k - 1]
            for k in range(lag)
        ])
        X_full = np.column_stack([X_reduced, X_cause])

        def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_hat = X @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                return float(1.0 - ss_res / ss_total)
            except np.linalg.LinAlgError:
                return 0.0

        r2_reduced = max(ols_r2(X_reduced, y), 0.0)
        r2_full = max(ols_r2(X_full, y), 0.0)

        denom = max(1.0 - r2_full, 1e-8)
        f2 = (r2_full - r2_reduced) / denom
        return float(max(f2, 0.0))

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Core tester
# ---------------------------------------------------------------------------

class GrangerTester:
    """
    Runs pairwise Granger causality tests on a feature matrix.

    Parameters
    ----------
    max_lag            : maximum lag to test (AIC/BIC selects the best)
    significance_alpha : FWER/FDR significance level
    ic                 : information criterion for lag selection ("aic" | "bic")
    min_obs            : minimum rows required (skip if fewer)
    """

    def __init__(
        self,
        max_lag: int = 5,
        significance_alpha: float = 0.05,
        ic: str = "aic",
        min_obs: int = 50,
    ) -> None:
        self.max_lag = max_lag
        self.alpha = significance_alpha
        self.ic = ic
        self.min_obs = min_obs

    def test_all_pairs(self, feature_matrix: pd.DataFrame) -> GrangerResult:
        """
        Test all ordered pairs (X → Y) for X ≠ Y in feature_matrix.
        Applies FDR correction across all pairs.

        Returns a GrangerResult with edges sorted by p_value ascending.
        """
        df = feature_matrix.copy()

        # Drop non-numeric / constant columns
        df = df.select_dtypes(include=[np.number])
        df = df.loc[:, df.std() > 0]
        df = df.dropna()

        features = list(df.columns)
        n_obs = len(df)

        if n_obs < self.min_obs:
            log.warning(
                "Only %d observations; Granger tests may be unreliable (min=%d)",
                n_obs, self.min_obs,
            )

        if len(features) < 2:
            log.warning("Need at least 2 features for Granger tests; got %d", len(features))
            return GrangerResult(
                edges=[], features_tested=features,
                max_lag=self.max_lag, significance_threshold=self.alpha,
                n_observations=n_obs, n_pairs_tested=0,
                n_significant=0, fdr_method="BH",
            )

        log.info(
            "Running Granger tests on %d features × %d obs, max_lag=%d",
            len(features), n_obs, self.max_lag,
        )

        raw_edges: list[dict[str, Any]] = []

        for cause in features:
            for effect in features:
                if cause == effect:
                    continue
                edge = self._test_pair(df[cause], df[effect], cause, effect, df)
                if edge is not None:
                    raw_edges.append(edge)

        if not raw_edges:
            return GrangerResult(
                edges=[], features_tested=features,
                max_lag=self.max_lag, significance_threshold=self.alpha,
                n_observations=n_obs, n_pairs_tested=0,
                n_significant=0, fdr_method="BH",
            )

        # FDR correction
        raw_ps = [e["raw_p_value"] for e in raw_edges]
        adj_ps, rejected = _fdr_bh_correction(raw_ps, self.alpha)

        edges: list[GrangerEdge] = []
        for e, adj_p, sig in zip(raw_edges, adj_ps, rejected):
            edges.append(
                GrangerEdge(
                    cause=e["cause"],
                    effect=e["effect"],
                    optimal_lag=e["optimal_lag"],
                    p_value=float(adj_p),
                    raw_p_value=float(e["raw_p_value"]),
                    f_statistic=float(e["f_statistic"]),
                    effect_size=float(e["effect_size"]),
                    significant=bool(sig),
                    aic_at_lag=float(e["aic"]),
                    bic_at_lag=float(e["bic"]),
                )
            )

        edges.sort(key=lambda x: x.p_value)

        n_sig = sum(1 for e in edges if e.significant)
        log.info(
            "Granger tests done: %d pairs, %d significant (alpha=%.3f, FDR-BH)",
            len(edges), n_sig, self.alpha,
        )

        return GrangerResult(
            edges=edges,
            features_tested=features,
            max_lag=self.max_lag,
            significance_threshold=self.alpha,
            n_observations=n_obs,
            n_pairs_tested=len(edges),
            n_significant=n_sig,
            fdr_method="BH",
        )

    def test_pair(self, cause: str, effect: str, feature_matrix: pd.DataFrame) -> GrangerEdge | None:
        """Test a single X → Y pair. Useful for targeted testing."""
        df = feature_matrix[[cause, effect]].dropna()
        result = self._test_pair(df[cause], df[effect], cause, effect, df)
        if result is None:
            return None
        return GrangerEdge(
            cause=cause, effect=effect,
            optimal_lag=result["optimal_lag"],
            p_value=result["raw_p_value"],  # no FDR for single test
            raw_p_value=result["raw_p_value"],
            f_statistic=result["f_statistic"],
            effect_size=result["effect_size"],
            significant=result["raw_p_value"] < self.alpha,
            aic_at_lag=result["aic"],
            bic_at_lag=result["bic"],
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _test_pair(
        self,
        cause: pd.Series,
        effect: pd.Series,
        cause_name: str,
        effect_name: str,
        full_df: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Run Granger causality test for cause → effect.
        Uses AIC/BIC to select lag, then runs statsmodels grangercausalitytests.
        Returns dict or None on failure.
        """
        pair_df = pd.concat([effect, cause], axis=1).dropna()
        pair_df.columns = [effect_name, cause_name]

        n = len(pair_df)
        effective_max_lag = min(
            self.max_lag,
            max(1, (n - 1) // (pair_df.shape[1] + 2)),
        )
        if effective_max_lag < 1:
            return None

        # Lag selection
        try:
            optimal_lag = _select_optimal_lag(pair_df, effective_max_lag, self.ic)
        except Exception:
            optimal_lag = 1

        # Granger test at optimal lag
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_res = grangercausalitytests(
                    pair_df.values, maxlag=optimal_lag, verbose=False
                )
            lag_res = gc_res[optimal_lag]
            # statsmodels returns: {'ssr_ftest': (F, p, df_denom, df_num), ...}
            f_stat = float(lag_res[0]["ssr_ftest"][0])
            p_val = float(lag_res[0]["ssr_ftest"][1])
        except Exception as exc:
            log.debug("Granger test failed for %s → %s: %s", cause_name, effect_name, exc)
            return None

        # Effect size
        eff_size = _compute_effect_size(cause, effect, optimal_lag)

        # IC values
        aic, bic = _get_var_ic(pair_df, optimal_lag)

        return {
            "cause": cause_name,
            "effect": effect_name,
            "optimal_lag": optimal_lag,
            "raw_p_value": p_val,
            "f_statistic": f_stat,
            "effect_size": eff_size,
            "aic": aic,
            "bic": bic,
        }
