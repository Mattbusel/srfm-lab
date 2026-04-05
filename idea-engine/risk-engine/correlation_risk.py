"""
risk-engine/correlation_risk.py

Correlation and concentration risk analysis:
  - Rolling correlation matrices (returned as xarray DataArrays)
  - Correlation regime-change detection (DCC-inspired threshold method)
  - Effective number of independent assets
  - Diversification ratio
  - Herfindahl-Hirschman concentration score
  - Stress correlation scaling
  - Per-asset correlation risk contribution
  - Hypothesis generation for portfolio de-concentration

Requires: numpy, pandas, scipy, xarray.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eigh

try:
    import xarray as xr

    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RegimeChangeEvent:
    """
    Detected shift in the correlation structure of a portfolio.

    Attributes
    ----------
    timestamp : str
        ISO-8601 timestamp of the detected shift.
    bar_index : int
        Integer position in the return series.
    frobenius_distance : float
        Frobenius norm between the before- and after-window correlation matrices.
    before_avg_corr : float
        Average pairwise correlation before the shift.
    after_avg_corr : float
        Average pairwise correlation after the shift.
    assets : list[str]
        Asset names in the portfolio.
    """

    timestamp: str
    bar_index: int
    frobenius_distance: float
    before_avg_corr: float
    after_avg_corr: float
    assets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "bar_index": self.bar_index,
            "frobenius_distance": self.frobenius_distance,
            "before_avg_corr": self.before_avg_corr,
            "after_avg_corr": self.after_avg_corr,
            "assets": self.assets,
        }


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------


class CorrelationRiskAnalyzer:
    """
    Measures and monitors correlation and concentration risk in a portfolio.

    Parameters
    ----------
    min_window : int
        Minimum observations required before computing rolling statistics.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame(rng.normal(0, 0.01, (300, 4)), columns=list("ABCD"))
    >>> cra = CorrelationRiskAnalyzer()
    >>> eff_n = cra.effective_n(df.corr().values)
    >>> print(f"Effective N: {eff_n:.2f}")
    """

    def __init__(self, min_window: int = 20) -> None:
        self.min_window = min_window

    # ------------------------------------------------------------------
    # Rolling correlation
    # ------------------------------------------------------------------

    def rolling_correlation(
        self,
        returns_df: pd.DataFrame,
        window: int = 30,
    ) -> "xr.DataArray | dict":
        """
        Compute rolling pairwise correlation matrices.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Asset returns; columns are assets.
        window : int
            Rolling window size in bars.

        Returns
        -------
        xr.DataArray
            3-D array with dimensions (time, asset_i, asset_j).
            Falls back to a dict keyed by timestamp when xarray is unavailable.
        """
        assets = list(returns_df.columns)
        n_assets = len(assets)
        clean = returns_df.dropna()
        n = len(clean)

        dates: list = []
        matrices: list[np.ndarray] = []

        for i in range(window - 1, n):
            chunk = clean.iloc[i - window + 1 : i + 1]
            corr = chunk.corr().values
            np.fill_diagonal(corr, 1.0)
            matrices.append(corr)
            dates.append(clean.index[i])

        data = np.stack(matrices, axis=0)  # (T, n_assets, n_assets)

        if _HAS_XARRAY:
            return xr.DataArray(
                data,
                dims=["time", "asset_i", "asset_j"],
                coords={
                    "time": dates,
                    "asset_i": assets,
                    "asset_j": assets,
                },
                name="rolling_correlation",
            )
        # Fallback: plain dict
        return {
            str(d): pd.DataFrame(m, index=assets, columns=assets)
            for d, m in zip(dates, matrices)
        }

    # ------------------------------------------------------------------
    # Regime change detection
    # ------------------------------------------------------------------

    def correlation_regime_change(
        self,
        returns_df: pd.DataFrame,
        window: int = 60,
        threshold_quantile: float = 0.95,
    ) -> list[RegimeChangeEvent]:
        """
        Detect structural shifts in the correlation matrix using the
        Frobenius distance between adjacent rolling windows.

        A shift is flagged when the Frobenius distance between the
        before-window and after-window correlation matrices exceeds the
        ``threshold_quantile`` of the rolling distribution.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Asset returns.
        window : int
            Half-window size for the before/after comparison.
        threshold_quantile : float
            Quantile of Frobenius distances used as the detection threshold.

        Returns
        -------
        list[RegimeChangeEvent]
            Detected correlation regime changes, sorted by bar index.
        """
        assets = list(returns_df.columns)
        clean = returns_df.dropna()
        n = len(clean)

        frob_distances: list[float] = []
        indices: list[int] = []

        min_obs = max(window, self.min_window)
        for i in range(min_obs, n - window):
            before = clean.iloc[i - window : i].corr().values
            after = clean.iloc[i : i + window].corr().values
            dist = float(np.linalg.norm(after - before, "fro"))
            frob_distances.append(dist)
            indices.append(i)

        if not frob_distances:
            return []

        arr = np.array(frob_distances)
        threshold = float(np.quantile(arr, threshold_quantile))

        events: list[RegimeChangeEvent] = []
        for dist, bar_idx in zip(arr, indices):
            if dist >= threshold:
                before_chunk = clean.iloc[bar_idx - window : bar_idx]
                after_chunk = clean.iloc[bar_idx : bar_idx + window]

                before_avg = float(
                    np.mean(
                        before_chunk.corr().values[
                            np.triu_indices(len(assets), k=1)
                        ]
                    )
                )
                after_avg = float(
                    np.mean(
                        after_chunk.corr().values[
                            np.triu_indices(len(assets), k=1)
                        ]
                    )
                )

                ts = clean.index[bar_idx]
                timestamp = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

                events.append(
                    RegimeChangeEvent(
                        timestamp=timestamp,
                        bar_index=bar_idx,
                        frobenius_distance=float(dist),
                        before_avg_corr=before_avg,
                        after_avg_corr=after_avg,
                        assets=assets,
                    )
                )

        return events

    # ------------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------------

    def effective_n(self, corr_matrix: np.ndarray) -> float:
        """
        Effective number of independent assets derived from the eigenvalue
        spectrum of the correlation matrix.

        ENS = (Σλᵢ)² / Σλᵢ²

        For a perfectly diversified portfolio this equals the number of assets;
        for perfectly correlated assets it equals 1.

        Parameters
        ----------
        corr_matrix : np.ndarray
            Square correlation matrix (n × n).

        Returns
        -------
        float
            Effective number of independent assets in [1, n].
        """
        eigvals = np.linalg.eigvalsh(corr_matrix)
        eigvals = eigvals[eigvals > 0]
        if len(eigvals) == 0:
            return 1.0
        return float(eigvals.sum() ** 2 / (eigvals**2).sum())

    def diversification_ratio(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """
        Diversification ratio: weighted-average asset volatility divided by
        portfolio volatility.

        DR = (wᵀσ) / sqrt(wᵀΣw)

        DR = 1 for a single-asset or perfectly correlated portfolio.
        DR > 1 indicates diversification benefit.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights (must sum to 1).
        cov_matrix : np.ndarray
            Asset covariance matrix.

        Returns
        -------
        float
            Diversification ratio ≥ 1.
        """
        weights = np.asarray(weights, dtype=float)
        self._check_weights(weights)
        asset_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = float(weights @ asset_vols)
        port_vol = float(np.sqrt(weights @ cov_matrix @ weights))
        if port_vol == 0:
            return 1.0
        return float(weighted_avg_vol / port_vol)

    def concentration_score(self, weights: np.ndarray) -> float:
        """
        Herfindahl-Hirschman Index (HHI) as a concentration score.

        HHI = Σ wᵢ²

        Ranges from 1/n (perfect diversification) to 1 (full concentration).
        Normalised to [0, 1] by: HHI_norm = (HHI - 1/n) / (1 - 1/n).

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights.

        Returns
        -------
        float
            Normalised HHI in [0, 1].
        """
        w = np.asarray(weights, dtype=float)
        self._check_weights(w)
        n = len(w)
        hhi = float(np.sum(w**2))
        if n <= 1:
            return 1.0
        hhi_min = 1.0 / n
        hhi_max = 1.0
        return float((hhi - hhi_min) / (hhi_max - hhi_min))

    def stress_correlation(
        self,
        normal_corr: np.ndarray,
        stress_factor: float = 2.0,
    ) -> np.ndarray:
        """
        Construct a stressed correlation matrix by scaling all off-diagonal
        entries toward 1 by the ``stress_factor``.

        Stressed correlation: ρ*ᵢⱼ = min(ρᵢⱼ * stress_factor, 1)  (i ≠ j).

        The result is clipped to remain a valid correlation matrix (diagonal=1,
        entries ∈ [−1, 1]) and then projected to positive semi-definiteness
        if necessary.

        Parameters
        ----------
        normal_corr : np.ndarray
            Base (normal-regime) correlation matrix.
        stress_factor : float
            Multiplier applied to off-diagonal entries.  Must be > 0.

        Returns
        -------
        np.ndarray
            Stressed correlation matrix.
        """
        if stress_factor <= 0:
            raise ValueError("stress_factor must be positive")

        stressed = normal_corr.copy().astype(float)
        n = stressed.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    stressed[i, j] = np.clip(stressed[i, j] * stress_factor, -1.0, 1.0)

        # Project to nearest PSD correlation matrix using eigenvalue clipping
        return self._nearest_psd_corr(stressed)

    def correlation_risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> pd.Series:
        """
        Per-asset contribution to total portfolio variance, decomposed into
        a component attributable to correlation structure vs. individual volatility.

        Risk contribution for asset i:
            RC_i = w_i * (Σw)_i / (wᵀΣw)

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights.
        cov_matrix : np.ndarray
            Asset covariance matrix.

        Returns
        -------
        pd.Series
            Fractional risk contributions summing to 1.
        """
        w = np.asarray(weights, dtype=float)
        self._check_weights(w)
        port_var = float(w @ cov_matrix @ w)
        if port_var == 0:
            return pd.Series(np.zeros(len(w)))

        marginal_contrib = cov_matrix @ w
        risk_contrib = w * marginal_contrib / port_var
        return pd.Series(risk_contrib, name="risk_contribution")

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def generate_deconcentration_hypothesis(
        self,
        portfolio: dict[str, Any],
        concentration_threshold: float = 0.5,
    ) -> Optional[dict[str, Any]]:
        """
        Generate a risk-reduction hypothesis when the portfolio concentration
        score exceeds the given threshold.

        The hypothesis is returned as a dict compatible with the
        ``Hypothesis.create`` factory from ``hypothesis.types``.

        Parameters
        ----------
        portfolio : dict
            Must contain:
              - ``'weights'``: list or array of asset weights
              - ``'assets'``: list of asset names
              - ``'cov_matrix'`` (optional): covariance matrix as nested list
        concentration_threshold : float
            HHI_norm above which the hypothesis is triggered.

        Returns
        -------
        dict or None
            Hypothesis creation kwargs, or None if concentration is acceptable.
        """
        weights = np.asarray(portfolio["weights"], dtype=float)
        assets: list[str] = portfolio.get("assets", [f"asset_{i}" for i in range(len(weights))])

        score = self.concentration_score(weights)

        if score < concentration_threshold:
            return None

        # Identify the most concentrated positions
        top_k = np.argsort(weights)[::-1][:3]
        concentrated_assets = [assets[i] for i in top_k if i < len(assets)]
        top_weights = [float(weights[i]) for i in top_k if i < len(weights)]

        # Suggest equal-weight or risk-parity target
        n = len(weights)
        equal_w = round(1.0 / n, 4)

        hypothesis_params: dict[str, Any] = {
            "action": "reduce_concentration",
            "current_hhi_norm": round(float(score), 4),
            "concentration_threshold": concentration_threshold,
            "top_concentrated_assets": list(zip(concentrated_assets, top_weights)),
            "suggested_max_weight": equal_w * 2,  # 2× equal-weight as an upper bound
            "suggested_target": "risk_parity",
            "n_assets": n,
        }

        return {
            "hypothesis_type": "REGIME_FILTER",
            "parent_pattern_id": str(uuid.uuid4()),
            "parameters": hypothesis_params,
            "predicted_sharpe_delta": 0.10,
            "predicted_dd_delta": -0.05,
            "novelty_score": 0.6,
            "description": (
                f"Portfolio concentration (HHI_norm={score:.2f}) exceeds threshold "
                f"{concentration_threshold:.2f}. Top concentrated positions: "
                f"{concentrated_assets}. Suggest moving toward risk-parity allocation "
                f"to reduce tail risk and improve diversification ratio."
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_weights(weights: np.ndarray, tol: float = 1e-4) -> None:
        """Raise if weights do not sum to 1 within tolerance."""
        total = float(np.sum(weights))
        if abs(total - 1.0) > tol:
            raise ValueError(
                f"Portfolio weights must sum to 1.0; got {total:.6f}"
            )

    @staticmethod
    def _nearest_psd_corr(matrix: np.ndarray) -> np.ndarray:
        """
        Project a symmetric matrix to the nearest positive semi-definite
        correlation matrix using the Higham (2002) alternating projections
        algorithm (one eigenvalue-clipping step as approximation).
        """
        # Symmetrise
        mat = (matrix + matrix.T) / 2
        # Clip negative eigenvalues
        eigvals, eigvecs = eigh(mat)
        eigvals = np.maximum(eigvals, 0)
        psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Restore unit diagonal (correlation matrix form)
        d = np.sqrt(np.diag(psd))
        d = np.where(d == 0, 1, d)
        corr = psd / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        return corr


# ---------------------------------------------------------------------------
# Utility: average correlation across a portfolio over time
# ---------------------------------------------------------------------------


def rolling_avg_correlation(
    returns_df: pd.DataFrame,
    window: int = 30,
) -> pd.Series:
    """
    Compute the rolling average pairwise correlation for a portfolio,
    returning a single time series useful for regime monitoring.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling average correlation indexed like the input DataFrame.
    """
    clean = returns_df.dropna()
    n = len(clean)
    assets = list(clean.columns)
    n_assets = len(assets)

    if n_assets < 2:
        return pd.Series(np.ones(n), index=clean.index, name="avg_corr")

    idx_upper = np.triu_indices(n_assets, k=1)
    results: list[float] = []

    for i in range(n):
        if i < window - 1:
            results.append(np.nan)
            continue
        chunk = clean.iloc[i - window + 1 : i + 1]
        corr = chunk.corr().values
        avg = float(np.mean(corr[idx_upper]))
        results.append(avg)

    return pd.Series(results, index=clean.index, name="rolling_avg_corr")
