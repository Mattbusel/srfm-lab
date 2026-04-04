"""
research/param_explorer/regime_sensitivity.py
===============================================
How optimal parameters differ across market regimes (bull/bear/sideways/crisis).

Uses the regime labels produced by research/regime_analysis/ and optimises
the objective independently within each regime's data slice.

Classes
-------
RegimeSensitivityAnalyzer : Unified interface

Stand-alone functions
---------------------
analyze_regime_optima     : Per-regime BO / grid optimisation
regime_param_stability    : Quantify divergence between regime optima
recommend_regime_params   : Choose best params for a given regime
dynamic_param_schedule    : Time-varying optimal parameter series
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from research.param_explorer.space import ParamSpace, ParamType
from research.param_explorer.bayesian_opt import (
    BayesianOptimizer,
    BayesOptResult,
    AcquisitionFunction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known regime labels
# ---------------------------------------------------------------------------

KNOWN_REGIMES = ("bull", "bear", "sideways", "crisis", "recovery", "unknown")


# ---------------------------------------------------------------------------
# OptResult dataclass (lightweight)
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    """
    Optimisation result for a single regime.

    Attributes
    ----------
    regime : str
    best_params : dict
    best_score : float
    n_samples : int
        Number of data points in this regime slice.
    method : str
        'bayes', 'grid', or 'sobol'.
    history_y : np.ndarray
    """

    regime: str
    best_params: Dict[str, Any]
    best_score: float
    n_samples: int
    method: str
    history_y: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "best_params": self.best_params,
            "best_score": float(self.best_score),
            "n_samples": self.n_samples,
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Regime param stability metrics
# ---------------------------------------------------------------------------

@dataclass
class ParamStabilityResult:
    """
    Quantification of how much optimal parameters vary across regimes.

    Attributes
    ----------
    param_names : list[str]
    cv : dict[str, float]
        Coefficient of variation (std/mean) per parameter across regimes.
    pairwise_distance : dict[tuple, float]
        L2 distance (in unit space) between each pair of regime optima.
    most_stable : list[str]
        Parameters with lowest CV (most stable across regimes).
    most_variable : list[str]
        Parameters with highest CV (most regime-dependent).
    stability_score : float
        Overall score ∈ [0, 1], 1 = perfectly stable.
    """

    param_names: List[str]
    cv: Dict[str, float]
    pairwise_distance: Dict[Tuple[str, str], float]
    most_stable: List[str]
    most_variable: List[str]
    stability_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cv": self.cv,
            "most_stable": self.most_stable,
            "most_variable": self.most_variable,
            "stability_score": float(self.stability_score),
        }


# ---------------------------------------------------------------------------
# Stand-alone functions
# ---------------------------------------------------------------------------

def analyze_regime_optima(
    trades: pd.DataFrame,
    regime_col: str,
    param_space: ParamSpace,
    objective_fn_factory: Callable[[pd.DataFrame], Callable[[Dict[str, Any]], float]],
    method: str = "bayes",
    n_init: int = 8,
    n_iter: int = 30,
    min_regime_samples: int = 20,
    seed: int = 42,
) -> Dict[str, OptResult]:
    """
    Optimise parameters independently for each regime label.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade-level data with a column *regime_col* marking each trade's regime.
    regime_col : str
        Column name containing regime labels.
    param_space : ParamSpace
    objective_fn_factory : callable
        Given a slice of *trades*, returns an objective function
        ``(params → float)``.  This allows the objective to be fitted to
        regime-specific data.
    method : str
        'bayes' (Bayesian opt), 'sobol' (random Sobol search), or 'grid'.
    n_init : int
        Initial points for Bayesian opt.
    n_iter : int
        BO iterations per regime.
    min_regime_samples : int
        Regimes with fewer than this many rows are skipped.
    seed : int

    Returns
    -------
    dict[regime_label, OptResult]
    """
    if regime_col not in trades.columns:
        raise ValueError(f"Column {regime_col!r} not found in trades DataFrame.")

    regimes = trades[regime_col].dropna().unique().tolist()
    results: Dict[str, OptResult] = {}

    for regime in regimes:
        regime_str = str(regime)
        slice_df = trades[trades[regime_col] == regime].copy()

        if len(slice_df) < min_regime_samples:
            logger.info(
                "Skipping regime %r: only %d samples (< %d).",
                regime_str, len(slice_df), min_regime_samples,
            )
            continue

        logger.info(
            "Optimising for regime %r (%d samples)…", regime_str, len(slice_df)
        )

        obj_fn = objective_fn_factory(slice_df)

        if method == "bayes":
            opt = BayesianOptimizer(
                param_space=param_space,
                objective_fn=obj_fn,
                acquisition=AcquisitionFunction.EI,
                n_init=n_init,
                seed=seed,
            )
            bo_result = opt.run(n_iter=n_iter, verbose=False)
            results[regime_str] = OptResult(
                regime=regime_str,
                best_params=bo_result.best_params,
                best_score=bo_result.best_score,
                n_samples=len(slice_df),
                method="bayes",
                history_y=bo_result.history_y,
            )

        elif method == "sobol":
            X = param_space.sample_sobol(n_init + n_iter, seed=seed)
            y_vals = np.array([
                obj_fn(param_space.to_params(X[k])) for k in range(len(X))
            ])
            best_idx = int(np.argmax(y_vals))
            results[regime_str] = OptResult(
                regime=regime_str,
                best_params=param_space.to_params(X[best_idx]),
                best_score=float(y_vals[best_idx]),
                n_samples=len(slice_df),
                method="sobol",
                history_y=y_vals,
            )

        elif method == "grid":
            n_per_dim = max(2, int(round((n_init + n_iter) ** (1.0 / param_space.n_dims))))
            X = param_space.sample_grid(n_per_dim)
            if len(X) > 500:
                # Sub-sample if grid is too large
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(X), size=500, replace=False)
                X = X[idx]
            y_vals = np.array([
                obj_fn(param_space.to_params(X[k])) for k in range(len(X))
            ])
            best_idx = int(np.argmax(y_vals))
            results[regime_str] = OptResult(
                regime=regime_str,
                best_params=param_space.to_params(X[best_idx]),
                best_score=float(y_vals[best_idx]),
                n_samples=len(slice_df),
                method="grid",
                history_y=y_vals,
            )

        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'bayes', 'sobol', or 'grid'.")

    logger.info("Regime analysis complete: %d regimes optimised.", len(results))
    return results


def regime_param_stability(
    regime_optima: Dict[str, OptResult],
    param_space: ParamSpace,
) -> ParamStabilityResult:
    """
    Quantify how stable optimal parameters are across regimes.

    Parameters
    ----------
    regime_optima : dict[regime, OptResult]
    param_space : ParamSpace

    Returns
    -------
    ParamStabilityResult
    """
    if len(regime_optima) < 2:
        raise ValueError("Need at least 2 regimes to compute stability.")

    regimes = list(regime_optima.keys())
    param_names = param_space.names

    # Build matrix of unit-space param values (n_regimes × d)
    unit_rows: Dict[str, np.ndarray] = {}
    for regime, opt_res in regime_optima.items():
        try:
            row = param_space.from_params(opt_res.best_params)
        except Exception as exc:
            logger.warning("Could not encode regime %r params: %s", regime, exc)
            row = np.full(param_space.n_dims, 0.5)
        unit_rows[regime] = row

    # Coefficient of variation per parameter
    matrix = np.vstack([unit_rows[r] for r in regimes])  # (n_regimes, d)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=1) if len(regimes) > 1 else np.zeros(matrix.shape[1])
    cv: Dict[str, float] = {}
    for j, name in enumerate(param_names):
        if abs(means[j]) < 1e-10:
            cv[name] = 0.0 if stds[j] < 1e-10 else float("inf")
        else:
            cv[name] = float(stds[j] / abs(means[j]))

    # Pairwise L2 distances
    pairwise: Dict[Tuple[str, str], float] = {}
    for i, r1 in enumerate(regimes):
        for j, r2 in enumerate(regimes):
            if j <= i:
                continue
            dist = float(np.linalg.norm(unit_rows[r1] - unit_rows[r2]))
            pairwise[(r1, r2)] = dist

    # Rank stability
    cv_vals = [(name, cv[name]) for name in param_names if math.isfinite(cv[name])]
    cv_vals_sorted = sorted(cv_vals, key=lambda kv: kv[1])
    most_stable = [n for n, _ in cv_vals_sorted[:max(1, len(cv_vals_sorted) // 3)]]
    most_variable = [n for n, _ in cv_vals_sorted[-max(1, len(cv_vals_sorted) // 3):]]

    # Overall stability score: mean of 1 - normalised pairwise distances
    if pairwise:
        max_possible_dist = math.sqrt(param_space.n_dims)  # all params at opposite ends
        mean_dist = np.mean(list(pairwise.values()))
        stability_score = float(max(0.0, 1.0 - mean_dist / max_possible_dist))
    else:
        stability_score = 1.0

    return ParamStabilityResult(
        param_names=param_names,
        cv=cv,
        pairwise_distance=pairwise,
        most_stable=most_stable,
        most_variable=most_variable,
        stability_score=stability_score,
    )


def recommend_regime_params(
    regime_optima: Dict[str, OptResult],
    current_regime: str,
    fallback_to_global: bool = True,
    global_best_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Choose the best parameter set for *current_regime*.

    If the regime has been seen during training, return its optimal params.
    Otherwise fall back to the global best (if *fallback_to_global* is True)
    or raise.

    Parameters
    ----------
    regime_optima : dict[regime, OptResult]
    current_regime : str
    fallback_to_global : bool
    global_best_params : dict | None

    Returns
    -------
    dict of recommended parameters
    """
    if current_regime in regime_optima:
        opt = regime_optima[current_regime]
        logger.info(
            "Recommending regime-specific params for %r (score=%.4g).",
            current_regime, opt.best_score,
        )
        return dict(opt.best_params)

    if fallback_to_global and global_best_params is not None:
        logger.warning(
            "Regime %r not seen; falling back to global best params.", current_regime
        )
        return dict(global_best_params)

    # Find closest known regime by name
    known = list(regime_optima.keys())
    if known:
        logger.warning(
            "Regime %r not seen; using %r (first available).", current_regime, known[0]
        )
        return dict(regime_optima[known[0]].best_params)

    raise KeyError(f"Regime {current_regime!r} not found and no fallback available.")


def dynamic_param_schedule(
    trades: pd.DataFrame,
    regime_series: pd.Series,
    param_space: ParamSpace,
    regime_optima: Dict[str, OptResult],
    global_best_params: Optional[Dict[str, Any]] = None,
    smooth_window: int = 5,
) -> pd.DataFrame:
    """
    Build a time-varying parameter schedule based on regime labels.

    For each time step, look up the optimal parameters for the current regime
    and store them in a DataFrame indexed by time.  Parameters are optionally
    smoothed with a rolling window to avoid abrupt jumps.

    Parameters
    ----------
    trades : pd.DataFrame
        Must have a DatetimeIndex or an index compatible with *regime_series*.
    regime_series : pd.Series
        Regime labels aligned with *trades*.
    param_space : ParamSpace
    regime_optima : dict[regime, OptResult]
    global_best_params : dict | None
        Fall-back when regime not found.
    smooth_window : int
        Rolling average window for numeric parameters (0 = no smoothing).

    Returns
    -------
    pd.DataFrame with columns = param names + 'regime',
    indexed like *trades*.
    """
    param_names = param_space.names
    records = []

    for idx in regime_series.index:
        regime = str(regime_series.loc[idx]) if idx in regime_series.index else "unknown"
        try:
            params = recommend_regime_params(
                regime_optima, regime,
                fallback_to_global=True,
                global_best_params=global_best_params,
            )
        except KeyError:
            if global_best_params:
                params = dict(global_best_params)
            else:
                params = param_space.defaults
        record = {"regime": regime}
        record.update(params)
        records.append(record)

    df = pd.DataFrame(records, index=regime_series.index)

    if smooth_window > 1:
        numeric_cols = [
            c for c in param_names
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        df[numeric_cols] = df[numeric_cols].rolling(smooth_window, min_periods=1).mean()

    return df


# ---------------------------------------------------------------------------
# RegimeSensitivityAnalyzer class
# ---------------------------------------------------------------------------

class RegimeSensitivityAnalyzer:
    """
    Unified interface for regime-conditional parameter sensitivity analysis.

    Parameters
    ----------
    param_space : ParamSpace
    objective_fn_factory : callable
        ``(trades_slice: pd.DataFrame) → (params: dict → float)``
    """

    def __init__(
        self,
        param_space: ParamSpace,
        objective_fn_factory: Callable[[pd.DataFrame], Callable[[Dict[str, Any]], float]],
        seed: int = 42,
    ) -> None:
        self.param_space = param_space
        self.objective_fn_factory = objective_fn_factory
        self.seed = seed

        self._regime_optima: Dict[str, OptResult] = {}
        self._stability: Optional[ParamStabilityResult] = None
        self._schedule: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def fit(
        self,
        trades: pd.DataFrame,
        regime_col: str,
        method: str = "bayes",
        n_init: int = 8,
        n_iter: int = 30,
        min_regime_samples: int = 20,
    ) -> "RegimeSensitivityAnalyzer":
        """
        Run per-regime optimisation.

        Parameters
        ----------
        trades : pd.DataFrame
        regime_col : str
        method : str
        n_init, n_iter : int
        min_regime_samples : int

        Returns
        -------
        self
        """
        self._regime_optima = analyze_regime_optima(
            trades=trades,
            regime_col=regime_col,
            param_space=self.param_space,
            objective_fn_factory=self.objective_fn_factory,
            method=method,
            n_init=n_init,
            n_iter=n_iter,
            min_regime_samples=min_regime_samples,
            seed=self.seed,
        )

        if len(self._regime_optima) >= 2:
            self._stability = regime_param_stability(
                self._regime_optima, self.param_space
            )

        return self

    def stability_report(self) -> ParamStabilityResult:
        """Return the stability analysis result (run :meth:`fit` first)."""
        if self._stability is None:
            raise RuntimeError("Call fit() before stability_report().")
        return self._stability

    def recommend(
        self,
        current_regime: str,
        global_best_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Recommend parameters for *current_regime*."""
        return recommend_regime_params(
            self._regime_optima, current_regime,
            global_best_params=global_best_params,
        )

    def build_schedule(
        self,
        trades: pd.DataFrame,
        regime_series: pd.Series,
        smooth_window: int = 5,
        global_best_params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Build a time-varying parameter schedule.

        Returns
        -------
        pd.DataFrame
        """
        self._schedule = dynamic_param_schedule(
            trades=trades,
            regime_series=regime_series,
            param_space=self.param_space,
            regime_optima=self._regime_optima,
            global_best_params=global_best_params,
            smooth_window=smooth_window,
        )
        return self._schedule

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def regime_optima(self) -> Dict[str, OptResult]:
        return dict(self._regime_optima)

    @property
    def known_regimes(self) -> List[str]:
        return list(self._regime_optima.keys())

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_regime_optima(
        self,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """
        Side-by-side bar charts showing optimal parameter values per regime.

        Parameters
        ----------
        save_path : str | Path | None
        figsize : tuple | None

        Returns
        -------
        matplotlib.figure.Figure
        """
        regimes = list(self._regime_optima.keys())
        n_regimes = len(regimes)
        param_names = self.param_space.names
        n_params = len(param_names)

        if n_regimes == 0:
            logger.warning("No regime optima to plot.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No regimes found.", ha="center", va="center")
            return fig

        if figsize is None:
            figsize = (max(8, n_params * 1.5), max(4, n_regimes * 1.5))

        # Build normalised (unit-space) values for each regime
        unit_matrix = np.zeros((n_regimes, n_params))
        for i, regime in enumerate(regimes):
            opt = self._regime_optima[regime]
            try:
                unit_matrix[i] = self.param_space.from_params(opt.best_params)
            except Exception:
                unit_matrix[i] = 0.5

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(n_params)
        width = 0.8 / max(n_regimes, 1)
        cmap = plt.cm.get_cmap("tab10")
        offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_regimes)

        for i, regime in enumerate(regimes):
            ax.bar(
                x + offsets[i],
                unit_matrix[i],
                width,
                label=regime,
                color=cmap(i % 10),
                alpha=0.8,
                edgecolor="white",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Unit-space value", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Optimal Parameters by Regime", fontsize=12, fontweight="bold")
        ax.legend(title="Regime", fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Regime optima plot saved to %s", save_path)

        return fig

    def plot_stability(
        self,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (9, 5),
    ) -> plt.Figure:
        """
        Visualise the regime parameter stability (CV per parameter).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._stability is None:
            raise RuntimeError("Call fit() before plot_stability().")

        stab = self._stability
        names = stab.param_names
        cv_vals = np.array([stab.cv.get(n, 0.0) for n in names])
        # Cap infinite values for display
        cv_vals = np.where(np.isfinite(cv_vals), cv_vals, 2.0)

        sort_idx = np.argsort(cv_vals)
        names_sorted = [names[i] for i in sort_idx]
        cv_sorted = cv_vals[sort_idx]

        colors = ["#55A868" if cv <= 0.20 else ("#DD8452" if cv <= 0.50 else "#C44E52")
                  for cv in cv_sorted]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(names_sorted, cv_sorted, color=colors, edgecolor="white", alpha=0.85)
        ax.axvline(0.20, ls="--", color="#55A868", lw=1.5, alpha=0.8, label="Low (≤0.2)")
        ax.axvline(0.50, ls="--", color="#DD8452", lw=1.5, alpha=0.8, label="Medium (≤0.5)")

        ax.set_xlabel("Coefficient of Variation (σ/μ) across regimes", fontsize=10)
        ax.set_title(
            f"Parameter Stability Across Regimes\n"
            f"(overall score = {stab.stability_score:.2%})",
            fontsize=11, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Stability plot saved to %s", save_path)

        return fig

    def plot_schedule(
        self,
        param_name: str,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (12, 4),
    ) -> plt.Figure:
        """
        Plot the time-varying schedule for a single parameter.

        Parameters
        ----------
        param_name : str
        save_path : str | Path | None
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._schedule is None:
            raise RuntimeError("Call build_schedule() before plot_schedule().")

        schedule = self._schedule
        if param_name not in schedule.columns:
            raise ValueError(f"{param_name!r} not in schedule columns.")

        regimes_in_schedule = schedule["regime"].unique()
        regime_colors = {
            r: plt.cm.tab10(i % 10) for i, r in enumerate(regimes_in_schedule)
        }

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(schedule.index, schedule[param_name], lw=1.5, color="#4C72B0", zorder=3)

        # Shade background by regime
        for regime in regimes_in_schedule:
            mask = schedule["regime"] == regime
            if not mask.any():
                continue
            # Convert boolean mask to spans
            spans = _bool_mask_to_spans(mask, schedule.index)
            for start, end in spans:
                ax.axvspan(
                    start, end, alpha=0.18,
                    color=regime_colors[regime], label=regime,
                )

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=8, title="Regime")

        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel(param_name, fontsize=10)
        ax.set_title(f"Dynamic Parameter Schedule: {param_name}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Schedule plot saved to %s", save_path)

        return fig

    def summary(self) -> str:
        """Return a text summary of the regime analysis."""
        lines = ["=" * 60, " Regime Sensitivity Analysis Summary", "=" * 60]

        if not self._regime_optima:
            lines.append("  (no regimes fitted yet)")
            return "\n".join(lines)

        lines.append(f"  Regimes found: {list(self._regime_optima.keys())}")
        for regime, opt in self._regime_optima.items():
            lines.append(
                f"  {regime:<15} best_score={opt.best_score:.4g}  "
                f"n_samples={opt.n_samples}  method={opt.method}"
            )

        if self._stability is not None:
            lines.append("")
            lines.append(f"  Stability score: {self._stability.stability_score:.2%}")
            lines.append(f"  Most stable params: {self._stability.most_stable}")
            lines.append(f"  Most variable params: {self._stability.most_variable}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bool_mask_to_spans(
    mask: pd.Series,
    index: pd.Index,
) -> List[Tuple[Any, Any]]:
    """Convert a boolean mask to a list of (start, end) spans."""
    spans = []
    in_span = False
    start = None
    idx_list = list(index)
    mask_vals = list(mask)

    for i, val in enumerate(mask_vals):
        if val and not in_span:
            in_span = True
            start = idx_list[i]
        elif not val and in_span:
            in_span = False
            spans.append((start, idx_list[i - 1]))

    if in_span and start is not None:
        spans.append((start, idx_list[-1]))

    return spans
