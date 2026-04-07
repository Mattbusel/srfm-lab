"""
research/simulation/parameter_sensitivity_sim.py

Monte Carlo simulation for LARSA parameter sensitivity analysis.

Generates 1000 synthetic market paths per parameter value and computes
the distribution of Sharpe ratios to identify robust parameter regions.

Key question: which parameter ranges produce strategies that are insensitive
to small perturbations (stable Sharpe across a range of values)?

Usage
-----
>>> sim = ParameterSensitivitySimulator(n_paths=200, n_bars=1000)
>>> result = sim.run("bh_form", param_range=np.linspace(1.0, 2.5, 8))
>>> sim.plot_sensitivity_distribution(result)
"""

from __future__ import annotations

import math
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from research.simulation.market_simulator import (
    GeometricBrownianMotion,
    RegimeSwitchingMarket,
    SimConfig,
    MarketRegime,
    DT_15M,
    BARS_PER_YEAR,
)
from research.simulation.bh_signal_injector import (
    compute_bh_mass_series,
    BH_FORM_DEFAULT,
    DEFAULT_CF,
    BH_DECAY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Results from a parameter sensitivity simulation run.

    Attributes
    ----------
    param_name:
        Name of the parameter varied.
    param_values:
        Array of parameter values tested.
    sharpe_distributions:
        List of arrays, one per param value, each containing Sharpe ratios
        across all Monte Carlo paths.
    mean_sharpes:
        Mean Sharpe per parameter value.
    std_sharpes:
        Std-dev of Sharpe per parameter value.
    percentile_5:
        5th percentile Sharpe (tail risk) per parameter value.
    percentile_95:
        95th percentile Sharpe per parameter value.
    robustness_scores:
        Robustness score per parameter value: high score = stable Sharpe
        as parameter changes by +/- one step.
        Score = 1 / (1 + sensitivity_index) where sensitivity_index measures
        how much mean Sharpe changes relative to neighbouring values.
    n_paths:
        Number of Monte Carlo paths used.
    n_bars:
        Bars per path.
    """
    param_name: str
    param_values: NDArray[np.float64]
    sharpe_distributions: list[NDArray[np.float64]]
    mean_sharpes: NDArray[np.float64]
    std_sharpes: NDArray[np.float64]
    percentile_5: NDArray[np.float64]
    percentile_95: NDArray[np.float64]
    robustness_scores: NDArray[np.float64]
    n_paths: int
    n_bars: int

    def best_param_value(self) -> float:
        """Return param value with highest robustness-weighted mean Sharpe."""
        score = self.mean_sharpes * self.robustness_scores
        best_idx = int(np.argmax(score))
        return float(self.param_values[best_idx])

    def stable_range(self, robustness_threshold: float = 0.7) -> tuple[float, float]:
        """Return (low, high) param range where robustness >= threshold.

        Returns (nan, nan) if no value exceeds threshold.
        """
        mask = self.robustness_scores >= robustness_threshold
        if not np.any(mask):
            return (float("nan"), float("nan"))
        valid = self.param_values[mask]
        return (float(valid.min()), float(valid.max()))


# ---------------------------------------------------------------------------
# Built-in LARSA signal computer (standalone, no QC dependency)
# ---------------------------------------------------------------------------

def _larsa_signal_compute(
    closes: NDArray[np.float64],
    cf: float = DEFAULT_CF,
    bh_form: float = BH_FORM_DEFAULT,
    bh_collapse: float = 1.0,
    bh_decay: float = BH_DECAY,
    ctl_min: int = 3,
) -> NDArray[np.float64]:
    """Compute LARSA-style positions from a close price series.

    Returns array of positions (+1 long, -1 short, 0 flat) for each bar.

    This is a simplified standalone version of LARSA BH physics logic
    suitable for backtesting in pure Python without QuantConnect.
    """
    n = len(closes)
    positions = np.zeros(n)
    bh_mass = 0.0
    ctl = 0
    bh_active = False
    bh_dir = 0

    for i in range(1, n):
        prev = closes[i - 1]
        curr = closes[i]
        beta_raw = abs(curr - prev) / (prev + 1e-9)
        beta = beta_raw / (cf + 1e-9)

        was_active = bh_active

        if beta < 1.0:
            ctl += 1
            sb = min(2.0, 1.0 + ctl * 0.1)
            bh_mass = bh_mass * (1.0 - 0.03) + 0.03 * sb
        else:
            ctl = 0
            bh_mass *= bh_decay

        if not was_active:
            bh_active = bh_mass > bh_form and ctl >= ctl_min
        else:
            bh_active = bh_mass > bh_collapse and ctl >= ctl_min

        if not was_active and bh_active:
            lookback = min(20, i)
            bh_dir = 1 if curr > closes[i - lookback] else -1

        if bh_active:
            positions[i] = float(bh_dir)

    return positions


def _compute_sharpe(
    positions: NDArray[np.float64],
    closes: NDArray[np.float64],
    periods_per_year: float = BARS_PER_YEAR,
) -> float:
    """Compute annualised Sharpe from positions and close prices."""
    n = len(closes)
    if n < 2:
        return float("nan")
    log_rets = np.concatenate([[0.0], np.diff(np.log(np.maximum(closes, 1e-12)))])
    strat_rets = np.zeros(n)
    strat_rets[1:] = positions[:-1] * log_rets[1:]
    mu = float(np.mean(strat_rets))
    sigma = float(np.std(strat_rets))
    if sigma < 1e-12:
        return float("nan")
    return float(mu / sigma * math.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

# Maps param_name -> callable(closes, param_value) -> positions
_PARAM_STRATEGIES: dict[str, Callable[[NDArray[np.float64], float], NDArray[np.float64]]] = {
    "bh_form": lambda closes, v: _larsa_signal_compute(closes, bh_form=v),
    "bh_collapse": lambda closes, v: _larsa_signal_compute(closes, bh_collapse=v),
    "bh_decay": lambda closes, v: _larsa_signal_compute(closes, bh_decay=v),
    "cf": lambda closes, v: _larsa_signal_compute(closes, cf=v),
    "ctl_min": lambda closes, v: _larsa_signal_compute(closes, ctl_min=int(round(v))),
}

# Default parameter ranges for sensitivity sweeps
DEFAULT_PARAM_RANGES: dict[str, NDArray[np.float64]] = {
    "bh_form":    np.linspace(0.8, 2.5, 12),
    "bh_collapse": np.linspace(0.5, 1.8, 10),
    "bh_decay":   np.linspace(0.85, 0.99, 10),
    "cf":         np.array([0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0020]),
    "ctl_min":    np.arange(1, 9, dtype=float),
}


# ---------------------------------------------------------------------------
# Monte Carlo path generator
# ---------------------------------------------------------------------------

def _generate_mc_paths(
    n_paths: int,
    n_bars: int,
    annual_vol: float = 0.20,
    include_regimes: bool = True,
    base_seed: int = 0,
) -> list[NDArray[np.float64]]:
    """Generate n_paths synthetic close price arrays.

    Mix of regimes to ensure robust evaluation:
      - 40% trending (mixed bull/bear)
      - 30% mean-reverting
      - 20% volatile
      - 10% BH episode
    """
    paths: list[NDArray[np.float64]] = []
    rng = np.random.default_rng(base_seed)

    for i in range(n_paths):
        path_seed = int(rng.integers(0, 2**31))
        regime_roll = rng.random()

        if not include_regimes or regime_roll < 0.20:
            # Pure GBM
            mu = rng.uniform(-0.10, 0.15)
            path = GeometricBrownianMotion.generate(
                n_bars, mu=mu, sigma=annual_vol, dt=DT_15M, seed=path_seed
            )
        elif regime_roll < 0.50:
            # Trending
            regime = MarketRegime.TRENDING_BULL if rng.random() < 0.6 else MarketRegime.TRENDING_BEAR
            cfg = SimConfig(
                n_bars=n_bars,
                regime_sequence=[(regime, n_bars)],
                initial_price=100.0,
                annual_vol=annual_vol,
                seed=path_seed,
            )
            df = RegimeSwitchingMarket.generate(cfg)
            path = df["close"].values
        elif regime_roll < 0.75:
            # Mixed regimes
            half = n_bars // 2
            cfg = SimConfig(
                n_bars=n_bars,
                regime_sequence=[
                    (MarketRegime.TRENDING_BULL, half // 2),
                    (MarketRegime.MEAN_REVERTING, half // 3),
                    (MarketRegime.VOLATILE, n_bars - half // 2 - half // 3),
                ],
                initial_price=100.0,
                annual_vol=annual_vol,
                seed=path_seed,
            )
            df = RegimeSwitchingMarket.generate(cfg)
            path = df["close"].values
        else:
            # BH episode
            bh_start = n_bars // 3
            cfg = SimConfig(
                n_bars=n_bars,
                regime_sequence=[
                    (MarketRegime.MEAN_REVERTING, bh_start),
                    (MarketRegime.BLACK_HOLE_ACTIVE, min(150, n_bars - bh_start - 50)),
                    (MarketRegime.TRENDING_BEAR, max(50, n_bars - bh_start - min(150, n_bars - bh_start - 50))),
                ],
                initial_price=100.0,
                annual_vol=annual_vol,
                seed=path_seed,
            )
            df = RegimeSwitchingMarket.generate(cfg)
            path = df["close"].values

        paths.append(path[:n_bars])  # ensure exact length

    return paths


# ---------------------------------------------------------------------------
# Parameter Sensitivity Simulator
# ---------------------------------------------------------------------------

class ParameterSensitivitySimulator:
    """
    Monte Carlo parameter sensitivity analysis for LARSA strategy parameters.

    For each parameter value, generates n_paths synthetic market paths and
    runs the LARSA signal computation with that parameter value. Computes the
    distribution of Sharpe ratios to identify robust parameter regions.

    Usage
    -----
    >>> sim = ParameterSensitivitySimulator(n_paths=500, n_bars=2000)
    >>> result = sim.run("bh_form", param_range=np.linspace(0.8, 2.5, 12))
    >>> print(f"Best bh_form: {result.best_param_value():.2f}")
    >>> lo, hi = result.stable_range()
    >>> print(f"Stable range: {lo:.2f} -- {hi:.2f}")
    """

    def __init__(
        self,
        n_paths: int = 1000,
        n_bars: int = 2000,
        annual_vol: float = 0.20,
        include_regimes: bool = True,
        base_seed: int = 0,
        n_jobs: int = 1,
    ):
        """
        Parameters
        ----------
        n_paths:
            Number of Monte Carlo paths to generate per parameter value.
        n_bars:
            Number of bars per path.
        annual_vol:
            Annualised vol for base simulation.
        include_regimes:
            If True, use regime-switching paths (more realistic).
            If False, pure GBM paths (faster).
        base_seed:
            Base random seed; each path gets seed = base_seed + path_index.
        n_jobs:
            Number of parallel workers (currently sequential; expand if needed).
        """
        self.n_paths = n_paths
        self.n_bars = n_bars
        self.annual_vol = annual_vol
        self.include_regimes = include_regimes
        self.base_seed = base_seed
        self.n_jobs = n_jobs

        # Pre-generate paths once (reuse across param sweeps for fair comparison)
        logger.info("Pre-generating %d MC paths (%d bars each)...", n_paths, n_bars)
        self._paths = _generate_mc_paths(
            n_paths, n_bars, annual_vol, include_regimes, base_seed
        )
        logger.info("Path generation complete.")

    def run(
        self,
        param_name: str,
        param_range: Optional[NDArray[np.float64]] = None,
        strategy_fn: Optional[Callable[[NDArray[np.float64], float], NDArray[np.float64]]] = None,
    ) -> SensitivityResult:
        """Run sensitivity analysis for a single parameter.

        Parameters
        ----------
        param_name:
            Name of the parameter to sweep. Must be a key in _PARAM_STRATEGIES
            or provided via strategy_fn.
        param_range:
            Array of parameter values to test. If None, uses default range.
        strategy_fn:
            Optional custom strategy function with signature
            (closes, param_value) -> positions.
            If None, uses the built-in LARSA signal computer.

        Returns
        -------
        SensitivityResult
        """
        if param_range is None:
            if param_name not in DEFAULT_PARAM_RANGES:
                raise ValueError(
                    f"No default range for param '{param_name}'. "
                    f"Provide param_range explicitly."
                )
            param_range = DEFAULT_PARAM_RANGES[param_name]

        param_range = np.asarray(param_range, dtype=np.float64)

        if strategy_fn is None:
            if param_name not in _PARAM_STRATEGIES:
                raise ValueError(
                    f"Unknown param '{param_name}'. Known: {list(_PARAM_STRATEGIES.keys())}. "
                    f"Provide a custom strategy_fn."
                )
            strategy_fn = _PARAM_STRATEGIES[param_name]

        n_vals = len(param_range)
        sharpe_distributions: list[NDArray[np.float64]] = []

        for val_idx, param_val in enumerate(param_range):
            logger.debug(
                "Param %s = %.4f (value %d/%d)", param_name, param_val, val_idx + 1, n_vals
            )
            sharpes = np.zeros(self.n_paths)
            for path_idx, closes in enumerate(self._paths):
                try:
                    positions = strategy_fn(closes, float(param_val))
                    sharpes[path_idx] = _compute_sharpe(positions, closes)
                except Exception as exc:
                    logger.debug("Path %d param %.4f failed: %s", path_idx, param_val, exc)
                    sharpes[path_idx] = float("nan")
            # Replace NaN with 0 for distribution stats (failed paths = no edge)
            sharpes = np.where(np.isfinite(sharpes), sharpes, 0.0)
            sharpe_distributions.append(sharpes)

        # Aggregate statistics
        mean_sharpes = np.array([float(np.mean(d)) for d in sharpe_distributions])
        std_sharpes = np.array([float(np.std(d)) for d in sharpe_distributions])
        pct5 = np.array([float(np.percentile(d, 5)) for d in sharpe_distributions])
        pct95 = np.array([float(np.percentile(d, 95)) for d in sharpe_distributions])

        # Robustness score: low sensitivity to adjacent param values
        robustness_scores = self._compute_robustness(mean_sharpes)

        return SensitivityResult(
            param_name=param_name,
            param_values=param_range,
            sharpe_distributions=sharpe_distributions,
            mean_sharpes=mean_sharpes,
            std_sharpes=std_sharpes,
            percentile_5=pct5,
            percentile_95=pct95,
            robustness_scores=robustness_scores,
            n_paths=self.n_paths,
            n_bars=self.n_bars,
        )

    def run_all(
        self,
        param_names: Optional[list[str]] = None,
    ) -> dict[str, SensitivityResult]:
        """Run sensitivity analysis for multiple parameters.

        Parameters
        ----------
        param_names:
            List of parameter names to sweep. If None, sweeps all built-in params.

        Returns
        -------
        dict mapping param_name -> SensitivityResult
        """
        if param_names is None:
            param_names = list(DEFAULT_PARAM_RANGES.keys())

        results: dict[str, SensitivityResult] = {}
        for name in param_names:
            logger.info("Sweeping parameter: %s", name)
            try:
                results[name] = self.run(name)
            except Exception as exc:
                logger.error("Parameter %s sweep failed: %s", name, exc)
        return results

    @staticmethod
    def _compute_robustness(mean_sharpes: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute robustness score for each parameter value.

        Score = 1 / (1 + local_sensitivity) where local_sensitivity is the
        mean absolute change in Sharpe relative to the adjacent parameter values.

        Score ranges from 0 (highly sensitive) to 1 (flat, insensitive).
        """
        n = len(mean_sharpes)
        if n < 2:
            return np.ones(n)

        scores = np.zeros(n)
        for i in range(n):
            neighbours = []
            if i > 0:
                neighbours.append(abs(mean_sharpes[i] - mean_sharpes[i - 1]))
            if i < n - 1:
                neighbours.append(abs(mean_sharpes[i] - mean_sharpes[i + 1]))
            sensitivity = float(np.mean(neighbours)) if neighbours else 0.0
            scores[i] = 1.0 / (1.0 + sensitivity)

        return scores

    def plot_sensitivity_distribution(
        self,
        result: SensitivityResult,
        figsize: tuple[int, int] = (12, 6),
        show: bool = True,
    ) -> None:
        """Plot violin plots of Sharpe distribution per parameter value.

        Requires matplotlib. If not available, logs the summary table.

        Parameters
        ----------
        result:
            SensitivityResult from run().
        figsize:
            Figure size (width, height) in inches.
        show:
            If True, call plt.show(). Set False for notebook / saved-figure use.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available -- printing summary table instead.")
            self._print_summary(result)
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_violin = axes[0]
        ax_line = axes[1]

        # Violin plot
        param_labels = [f"{v:.4g}" for v in result.param_values]
        parts = ax_violin.violinplot(
            result.sharpe_distributions,
            positions=range(len(result.param_values)),
            showmedians=True,
            showextrema=False,
        )
        ax_violin.set_xticks(range(len(result.param_values)))
        ax_violin.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=8)
        ax_violin.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_violin.set_xlabel(result.param_name)
        ax_violin.set_ylabel("Sharpe Ratio")
        ax_violin.set_title(f"Sharpe Distribution -- {result.param_name}")

        # Mean Sharpe + robustness
        x = range(len(result.param_values))
        ax_line.plot(x, result.mean_sharpes, "o-", color="steelblue", label="Mean Sharpe")
        ax_line.fill_between(
            x,
            result.percentile_5,
            result.percentile_95,
            alpha=0.2,
            color="steelblue",
            label="5th--95th pct",
        )
        ax2 = ax_line.twinx()
        ax2.plot(x, result.robustness_scores, "s--", color="orange", alpha=0.7, label="Robustness")
        ax2.set_ylabel("Robustness Score", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax_line.set_xticks(list(x))
        ax_line.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=8)
        ax_line.set_xlabel(result.param_name)
        ax_line.set_ylabel("Sharpe Ratio")
        ax_line.set_title(f"Mean Sharpe + Robustness -- {result.param_name}")
        ax_line.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        lines1, labels1 = ax_line.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_line.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        plt.tight_layout()
        if show:
            plt.show()

    def to_dataframe(self, result: SensitivityResult) -> pd.DataFrame:
        """Convert SensitivityResult to a tidy DataFrame.

        Columns: param_value, mean_sharpe, std_sharpe, pct5, pct95, robustness
        """
        return pd.DataFrame({
            "param_value": result.param_values,
            "mean_sharpe": result.mean_sharpes,
            "std_sharpe": result.std_sharpes,
            "pct5_sharpe": result.percentile_5,
            "pct95_sharpe": result.percentile_95,
            "robustness": result.robustness_scores,
        })

    @staticmethod
    def _print_summary(result: SensitivityResult) -> None:
        print(f"\nSensitivity Result: {result.param_name}")
        print(f"{'Value':>10} {'Mean Sharpe':>12} {'Std':>8} {'P5':>8} {'P95':>8} {'Robust':>8}")
        print("-" * 62)
        for i, v in enumerate(result.param_values):
            print(
                f"{v:>10.4g} {result.mean_sharpes[i]:>12.3f} "
                f"{result.std_sharpes[i]:>8.3f} "
                f"{result.percentile_5[i]:>8.3f} "
                f"{result.percentile_95[i]:>8.3f} "
                f"{result.robustness_scores[i]:>8.3f}"
            )
        best = result.best_param_value()
        lo, hi = result.stable_range()
        print(f"\nBest value (robustness-weighted): {best:.4g}")
        print(f"Stable range (robustness >= 0.7): {lo:.4g} -- {hi:.4g}")
