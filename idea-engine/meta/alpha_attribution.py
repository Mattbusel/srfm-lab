"""
alpha_attribution.py
Alpha return attribution and decomposition engine.
Implements Brinson-Hood-Beebower attribution, IC analysis per alpha source,
rolling attribution, marginal IR contribution, alpha decay, regime-conditional
attribution, and factor timing scoring.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Enumerations and Data Structures
# ---------------------------------------------------------------------------

class AlphaSource(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"
    MICROSTRUCTURE = "microstructure"
    ALTERNATIVE = "alternative"
    ML_SIGNAL = "ml_signal"
    PHYSICS_INSPIRED = "physics_inspired"


@dataclass
class AlphaAttribution:
    """Holds the decomposed attribution of returns to each alpha source."""
    total_return: float
    source_returns: dict[AlphaSource, float]
    residual: float                          # unexplained return
    r_squared: float                         # fraction of variance explained
    date_range: Optional[tuple[str, str]] = None
    regime: Optional[str] = None

    @property
    def explained_return(self) -> float:
        return sum(self.source_returns.values())

    def top_contributors(self, n: int = 3) -> list[tuple[AlphaSource, float]]:
        ranked = sorted(self.source_returns.items(), key=lambda x: abs(x[1]), reverse=True)
        return ranked[:n]


@dataclass
class ICResult:
    """Information coefficient results per alpha source."""
    source: AlphaSource
    ic_mean: float
    ic_std: float
    ic_ir: float          # IC / IC_std (information ratio of the IC)
    t_stat: float
    p_value: float
    n_observations: int


@dataclass
class DecayResult:
    """Alpha decay profile for a single source."""
    source: AlphaSource
    halflife_days: float
    decay_coefficients: np.ndarray    # IC at each horizon
    horizons: list[int]


@dataclass
class FactorTimingScore:
    """Measures whether factor weights were correct given subsequent returns."""
    date: str
    source: AlphaSource
    weight: float
    realized_return: float
    timing_score: float     # positive = correct overweight/underweight


# ---------------------------------------------------------------------------
# Attribution Engine
# ---------------------------------------------------------------------------

class AttributionEngine:
    """
    Decomposes portfolio returns into contributions from multiple alpha sources.
    All return arrays are expected as 1-D numpy arrays of period returns (decimal).
    """

    def __init__(self, sources: Optional[list[AlphaSource]] = None) -> None:
        self.sources = sources or list(AlphaSource)

    # -----------------------------------------------------------------------
    # Brinson attribution
    # -----------------------------------------------------------------------
    def decompose_returns(
        self,
        returns: np.ndarray,
        factor_returns: dict[AlphaSource, np.ndarray],
    ) -> AlphaAttribution:
        """
        Brinson-Hood-Beebower style attribution via OLS regression.
        Each factor_return array is the return of a pure factor portfolio.

        Returns AlphaAttribution with betas * factor_returns as source contributions.
        """
        returns = np.asarray(returns, dtype=float)
        n = len(returns)

        X_cols = []
        col_sources = []
        for source, fret in factor_returns.items():
            fret = np.asarray(fret, dtype=float)
            if len(fret) != n:
                raise ValueError(
                    f"Factor '{source}' length {len(fret)} != returns length {n}"
                )
            X_cols.append(fret)
            col_sources.append(source)

        if not X_cols:
            raise ValueError("No factor returns provided.")

        X = np.column_stack(X_cols)
        # OLS: add intercept
        X_int = np.column_stack([np.ones(n), X])
        try:
            betas, residuals, rank, sv = np.linalg.lstsq(X_int, returns, rcond=None)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"OLS decomposition failed: {e}") from e

        intercept = betas[0]
        factor_betas = betas[1:]

        # Compute source returns = beta_i * factor_return_i (contribution)
        source_returns: dict[AlphaSource, float] = {}
        fitted = np.full(n, intercept)
        for i, src in enumerate(col_sources):
            contribution = float(np.mean(factor_betas[i] * X_cols[i]))
            source_returns[src] = contribution
            fitted += factor_betas[i] * X_cols[i]

        residual_series = returns - fitted
        residual = float(np.mean(residual_series))

        # R-squared
        ss_total = float(np.var(returns) * n) if np.var(returns) > 0 else 1e-12
        ss_res = float(np.var(residual_series) * n)
        r_squared = max(0.0, 1.0 - ss_res / ss_total)

        return AlphaAttribution(
            total_return=float(np.mean(returns)),
            source_returns=source_returns,
            residual=residual,
            r_squared=r_squared,
        )

    # -----------------------------------------------------------------------
    # IC per source
    # -----------------------------------------------------------------------
    def information_coefficient_by_source(
        self,
        signals: dict[AlphaSource, np.ndarray],
        forward_returns: np.ndarray,
    ) -> list[ICResult]:
        """
        Compute the rank IC (Spearman) between each signal and forward_returns.
        signals: dict of source → signal values (T,) array
        forward_returns: (T,) array of realized forward returns
        """
        forward_returns = np.asarray(forward_returns, dtype=float)
        results = []

        for source, sig in signals.items():
            sig = np.asarray(sig, dtype=float)
            mask = np.isfinite(sig) & np.isfinite(forward_returns)
            sig_clean = sig[mask]
            fwd_clean = forward_returns[mask]
            n = int(np.sum(mask))

            if n < 5:
                warnings.warn(f"Source {source}: insufficient observations ({n})")
                continue

            rho, p_value = stats.spearmanr(sig_clean, fwd_clean)
            t_stat = rho * math.sqrt((n - 2) / max(1 - rho**2, 1e-12))

            # Rolling IC std via jackknife approximation
            # (full rolling IC would require per-period cross-sections)
            ic_std = float(np.std([
                stats.spearmanr(
                    np.delete(sig_clean, i), np.delete(fwd_clean, i)
                )[0]
                for i in range(min(n, 30))   # limit to 30 for efficiency
            ]))
            ic_ir = rho / ic_std if ic_std > 1e-8 else 0.0

            results.append(ICResult(
                source=source,
                ic_mean=float(rho),
                ic_std=ic_std,
                ic_ir=ic_ir,
                t_stat=float(t_stat),
                p_value=float(p_value),
                n_observations=n,
            ))

        return sorted(results, key=lambda x: abs(x.ic_mean), reverse=True)

    # -----------------------------------------------------------------------
    # Rolling attribution
    # -----------------------------------------------------------------------
    def rolling_attribution(
        self,
        returns: np.ndarray,
        factor_returns: dict[AlphaSource, np.ndarray],
        window: int = 60,
    ) -> list[AlphaAttribution]:
        """
        Compute attribution over rolling windows of length `window`.
        Returns a list of AlphaAttribution objects (one per window end-point).
        """
        returns = np.asarray(returns, dtype=float)
        n = len(returns)
        factor_arrays = {
            src: np.asarray(fret, dtype=float)
            for src, fret in factor_returns.items()
        }

        results = []
        for end in range(window, n + 1):
            start = end - window
            window_returns = returns[start:end]
            window_factors = {
                src: arr[start:end] for src, arr in factor_arrays.items()
            }
            try:
                attr = self.decompose_returns(window_returns, window_factors)
                attr.date_range = (str(start), str(end))
                results.append(attr)
            except Exception:
                pass

        return results

    # -----------------------------------------------------------------------
    # Marginal contribution to IR
    # -----------------------------------------------------------------------
    def marginal_contribution_to_ir(
        self,
        signal_returns: dict[AlphaSource, np.ndarray],
    ) -> dict[AlphaSource, float]:
        """
        Compute the marginal IR contribution of each signal.
        Uses a leave-one-out approach: IR(combined) - IR(combined without source).
        signal_returns: dict of source → per-period returns from that signal
        Returns dict of source → marginal IR contribution.
        """
        sources = list(signal_returns.keys())
        arrays = {src: np.asarray(arr, dtype=float) for src, arr in signal_returns.items()}

        def portfolio_ir(selected: list[AlphaSource]) -> float:
            if not selected:
                return 0.0
            combined = np.mean(
                np.column_stack([arrays[s] for s in selected]), axis=1
            )
            mean_ret = float(np.mean(combined))
            std_ret = float(np.std(combined))
            return mean_ret / std_ret if std_ret > 1e-8 else 0.0

        full_ir = portfolio_ir(sources)
        marginal: dict[AlphaSource, float] = {}

        for src in sources:
            without = [s for s in sources if s != src]
            ir_without = portfolio_ir(without)
            marginal[src] = full_ir - ir_without

        return dict(sorted(marginal.items(), key=lambda x: x[1], reverse=True))

    # -----------------------------------------------------------------------
    # Alpha decay by source
    # -----------------------------------------------------------------------
    def alpha_decay_by_source(
        self,
        signals: dict[AlphaSource, np.ndarray],
        returns_matrix: np.ndarray,
        horizons: Optional[list[int]] = None,
    ) -> list[DecayResult]:
        """
        Compute IC at multiple forward horizons to estimate alpha decay.
        signals: dict of source → signal array of shape (T,)
        returns_matrix: (T, max_horizon) matrix of forward returns at each horizon
        horizons: list of horizon offsets (e.g., [1, 5, 10, 20, 40, 60])
        """
        if horizons is None:
            horizons = [1, 5, 10, 20, 40, 60]

        returns_matrix = np.asarray(returns_matrix, dtype=float)
        results = []

        for source, sig in signals.items():
            sig = np.asarray(sig, dtype=float)
            ics_at_horizons = []

            for h_idx, h in enumerate(horizons):
                if h_idx >= returns_matrix.shape[1]:
                    break
                fwd = returns_matrix[:, h_idx]
                mask = np.isfinite(sig) & np.isfinite(fwd)
                if np.sum(mask) < 5:
                    ics_at_horizons.append(0.0)
                    continue
                rho, _ = stats.spearmanr(sig[mask], fwd[mask])
                ics_at_horizons.append(float(rho))

            ic_array = np.array(ics_at_horizons)

            # Estimate half-life via exponential decay fit: IC(h) = IC(0) * exp(-h / tau)
            valid_mask = ic_array > 0
            if np.sum(valid_mask) >= 2 and ics_at_horizons[0] > 1e-6:
                try:
                    log_ic = np.log(np.clip(ic_array[valid_mask], 1e-10, None)
                                    / ics_at_horizons[0])
                    h_valid = np.array(horizons[:len(ics_at_horizons)])[valid_mask]
                    tau, _ = np.polyfit(h_valid, -log_ic, 1)
                    halflife = float(math.log(2) / max(tau, 1e-6))
                except Exception:
                    halflife = float(horizons[-1])
            else:
                halflife = float(horizons[0])

            results.append(DecayResult(
                source=source,
                halflife_days=halflife,
                decay_coefficients=ic_array,
                horizons=horizons[:len(ics_at_horizons)],
            ))

        return sorted(results, key=lambda x: x.halflife_days, reverse=True)

    # -----------------------------------------------------------------------
    # Regime-conditional attribution
    # -----------------------------------------------------------------------
    def regime_conditional_attribution(
        self,
        returns: np.ndarray,
        factor_returns: dict[AlphaSource, np.ndarray],
        regime_labels: np.ndarray,
    ) -> dict[str, AlphaAttribution]:
        """
        Run Brinson attribution separately within each market regime.
        regime_labels: (T,) array of string/int regime identifiers per period
        Returns dict of regime_label → AlphaAttribution.
        """
        returns = np.asarray(returns, dtype=float)
        regime_labels = np.asarray(regime_labels)
        factor_arrays = {
            src: np.asarray(fret, dtype=float)
            for src, fret in factor_returns.items()
        }

        unique_regimes = np.unique(regime_labels)
        results: dict[str, AlphaAttribution] = {}

        for regime in unique_regimes:
            mask = regime_labels == regime
            if np.sum(mask) < 10:
                continue
            regime_returns = returns[mask]
            regime_factors = {
                src: arr[mask] for src, arr in factor_arrays.items()
            }
            try:
                attr = self.decompose_returns(regime_returns, regime_factors)
                attr.regime = str(regime)
                results[str(regime)] = attr
            except Exception as e:
                warnings.warn(f"Attribution failed for regime '{regime}': {e}")

        return results

    # -----------------------------------------------------------------------
    # Factor timing score
    # -----------------------------------------------------------------------
    def factor_timing_score(
        self,
        factor_weights: dict[AlphaSource, np.ndarray],
        factor_returns: dict[AlphaSource, np.ndarray],
    ) -> dict[AlphaSource, float]:
        """
        Measure whether dynamic factor weights were correct (positive timing alpha).
        Method: regress weight(t) on factor_return(t+1).
        A positive regression coefficient means overweighting was followed by outperformance.

        Returns dict of source → timing score (slope coefficient).
        """
        scores: dict[AlphaSource, float] = {}

        for source in factor_weights:
            if source not in factor_returns:
                continue

            w = np.asarray(factor_weights[source], dtype=float)
            r = np.asarray(factor_returns[source], dtype=float)

            # Align: weight at t predicts return at t+1
            if len(w) < len(r):
                r = r[:len(w)]
            elif len(r) < len(w):
                w = w[:len(r)]

            # Lead return by 1 period
            w_t = w[:-1]
            r_t1 = r[1:]

            mask = np.isfinite(w_t) & np.isfinite(r_t1)
            if np.sum(mask) < 5:
                scores[source] = 0.0
                continue

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                w_t[mask], r_t1[mask]
            )
            scores[source] = float(slope)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # -----------------------------------------------------------------------
    # Convenience: summary report
    # -----------------------------------------------------------------------
    def attribution_summary(
        self,
        attribution: AlphaAttribution,
    ) -> str:
        """Return a formatted text summary of an attribution result."""
        lines = [
            "=== Alpha Attribution Summary ===",
            f"Total Return:      {attribution.total_return:.4%}",
            f"Explained Return:  {attribution.explained_return:.4%}",
            f"Residual (Alpha):  {attribution.residual:.4%}",
            f"R-squared:         {attribution.r_squared:.4f}",
        ]
        if attribution.regime:
            lines.append(f"Regime:            {attribution.regime}")
        if attribution.date_range:
            lines.append(f"Date Range:        {attribution.date_range[0]} → {attribution.date_range[1]}")

        lines.append("\nSource Contributions:")
        for src, ret in sorted(
            attribution.source_returns.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {src.value:<20} {ret:.4%}")

        lines.append("\nTop Contributors:")
        for src, ret in attribution.top_contributors(3):
            lines.append(f"  {src.value:<20} {ret:.4%}  (abs: {abs(ret):.4%})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: simulate factor returns for testing
# ---------------------------------------------------------------------------
def simulate_factor_returns(
    n: int = 252,
    sources: Optional[list[AlphaSource]] = None,
    seed: int = 42,
) -> tuple[np.ndarray, dict[AlphaSource, np.ndarray]]:
    """
    Generate synthetic portfolio returns and factor returns for testing.
    Returns (portfolio_returns, factor_returns_dict).
    """
    rng = np.random.default_rng(seed)
    sources = sources or list(AlphaSource)

    factor_returns: dict[AlphaSource, np.ndarray] = {}
    betas = rng.uniform(0.05, 0.4, len(sources))
    combined = np.zeros(n)

    for i, src in enumerate(sources):
        fret = rng.normal(0.0003, 0.008, n)
        factor_returns[src] = fret
        combined += betas[i] * fret

    noise = rng.normal(0, 0.003, n)
    portfolio_returns = combined + noise

    return portfolio_returns, factor_returns
