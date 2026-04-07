"""
parameter_explorer.py -- Interactive parameter space exploration for the IAE.

Provides grid-search landscaping, local optima finding, cross-parameter
sensitivity surfaces, and suggestions for where to explore next.

All fitness evaluations use a simplified backtest callable passed by the caller.
The full SRFM evaluation pipeline is too expensive for exploration sweeps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Type alias for the fitness function passed by callers
FitnessFunc = Callable[[Dict[str, float]], float]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LandscapeMap:
    """
    2-D fitness landscape over two parameters.

    grid_x, grid_y are 1-D arrays of parameter values (the axes).
    fitness_surface has shape (len(grid_y), len(grid_x)) -- row = y, col = x.
    ridge_path is a list of (x, y) tuples tracing the highest-fitness spine.
    """

    param_x: str
    param_y: str
    grid_x: np.ndarray
    grid_y: np.ndarray
    fitness_surface: np.ndarray  # shape (n_y, n_x)
    optimal_x: float
    optimal_y: float
    ridge_path: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def peak_fitness(self) -> float:
        return float(self.fitness_surface.max())

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the surface into a tidy DataFrame with columns x, y, fitness."""
        rows = []
        for yi, yval in enumerate(self.grid_y):
            for xi, xval in enumerate(self.grid_x):
                rows.append(
                    {
                        self.param_x: xval,
                        self.param_y: yval,
                        "fitness": self.fitness_surface[yi, xi],
                    }
                )
        return pd.DataFrame(rows)


@dataclass
class ExplorationSuggestion:
    """
    A suggested point in parameter space to explore next.

    exploration_type is one of:
        "unexplored_region"   -- part of the space never visited
        "near_optimum"        -- fine-grained search near the known best
        "ridge_extension"     -- extend a known high-fitness ridge
        "diversity"           -- explore a different region to avoid local optima
    """

    param_name: str
    suggested_value: float
    expected_fitness_gain: float
    exploration_type: str
    confidence: float = 0.5  # 0..1 estimate of how reliable the suggestion is


# ---------------------------------------------------------------------------
# ParameterSpaceExplorer
# ---------------------------------------------------------------------------

class ParameterSpaceExplorer:
    """
    Map and navigate the IAE parameter fitness landscape.

    Parameters
    ----------
    fitness_fn : callable
        Function (params: Dict[str, float]) -> float.  Should be fast
        (simplified backtest, not full evaluation).
    param_bounds : dict
        {param_name: (lower, upper)} specifying valid ranges.
    base_params : dict
        Current best-known parameter values.  Used as the base when varying
        individual parameters during grid sweeps.
    """

    def __init__(
        self,
        fitness_fn: FitnessFunc,
        param_bounds: Dict[str, Tuple[float, float]],
        base_params: Dict[str, float],
    ) -> None:
        self.fitness_fn = fitness_fn
        self.param_bounds = param_bounds
        self.base_params = dict(base_params)

        # Visited history: list of (param_dict, fitness)
        self._visited: List[Tuple[Dict[str, float], float]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp(self, param_name: str, value: float) -> float:
        """Clamp value to the declared bounds for param_name."""
        lo, hi = self.param_bounds.get(param_name, (-np.inf, np.inf))
        return float(np.clip(value, lo, hi))

    def _evaluate(self, params: Dict[str, float]) -> float:
        """Evaluate fitness and record the visit."""
        # Clamp all params to bounds before evaluation
        clamped = {k: self._clamp(k, v) for k, v in params.items()}
        fitness = float(self.fitness_fn(clamped))
        self._visited.append((clamped, fitness))
        return fitness

    # ------------------------------------------------------------------
    # 2-D landscape mapping
    # ------------------------------------------------------------------

    def map_landscape(
        self,
        param_x: str,
        param_y: str,
        n_points: int = 20,
    ) -> LandscapeMap:
        """
        Grid search over a 2-D parameter space and return a LandscapeMap.

        All parameters other than param_x and param_y are held at their
        base_params values.
        """
        if param_x not in self.param_bounds:
            raise ValueError(f"param_x '{param_x}' not in param_bounds")
        if param_y not in self.param_bounds:
            raise ValueError(f"param_y '{param_y}' not in param_bounds")

        lo_x, hi_x = self.param_bounds[param_x]
        lo_y, hi_y = self.param_bounds[param_y]

        grid_x = np.linspace(lo_x, hi_x, n_points)
        grid_y = np.linspace(lo_y, hi_y, n_points)

        surface = np.zeros((n_points, n_points), dtype=float)

        base = dict(self.base_params)
        for yi, yval in enumerate(grid_y):
            for xi, xval in enumerate(grid_x):
                params = dict(base)
                params[param_x] = xval
                params[param_y] = yval
                surface[yi, xi] = self._evaluate(params)

        # Global optimum
        flat_idx = int(np.argmax(surface))
        opt_yi, opt_xi = np.unravel_index(flat_idx, surface.shape)
        optimal_x = float(grid_x[opt_xi])
        optimal_y = float(grid_y[opt_yi])

        # Ridge path -- for each x value, find y with highest fitness
        ridge_path: List[Tuple[float, float]] = []
        for xi, xval in enumerate(grid_x):
            col = surface[:, xi]
            best_yi = int(np.argmax(col))
            ridge_path.append((float(xval), float(grid_y[best_yi])))

        logger.info(
            "Landscape mapped for (%s, %s): optimal=(%s=%.4f, %s=%.4f) peak_fit=%.6f",
            param_x, param_y, param_x, optimal_x, param_y, optimal_y,
            float(surface.max()),
        )

        return LandscapeMap(
            param_x=param_x,
            param_y=param_y,
            grid_x=grid_x,
            grid_y=grid_y,
            fitness_surface=surface,
            optimal_x=optimal_x,
            optimal_y=optimal_y,
            ridge_path=ridge_path,
        )

    # ------------------------------------------------------------------
    # 1-D local optima scan
    # ------------------------------------------------------------------

    def find_local_optima(
        self,
        param_name: str,
        current: float,
        radius: float = 0.25,
        n_points: int = 50,
    ) -> List[float]:
        """
        Scan a 1-D neighbourhood around current and return a list of local
        maximum locations.

        radius is expressed as a fraction of the total parameter range.
        """
        if param_name not in self.param_bounds:
            raise ValueError(f"'{param_name}' not in param_bounds")

        lo, hi = self.param_bounds[param_name]
        half_width = (hi - lo) * radius
        scan_lo = max(lo, current - half_width)
        scan_hi = min(hi, current + half_width)

        values = np.linspace(scan_lo, scan_hi, n_points)
        fitnesses = np.zeros(n_points)

        base = dict(self.base_params)
        for i, val in enumerate(values):
            params = dict(base)
            params[param_name] = float(val)
            fitnesses[i] = self._evaluate(params)

        # Identify local maxima -- points higher than both neighbours
        local_max_vals: List[float] = []
        for i in range(1, n_points - 1):
            if fitnesses[i] > fitnesses[i - 1] and fitnesses[i] > fitnesses[i + 1]:
                local_max_vals.append(float(values[i]))

        # Also include endpoints if they are maxima
        if fitnesses[0] > fitnesses[1]:
            local_max_vals.insert(0, float(values[0]))
        if fitnesses[-1] > fitnesses[-2]:
            local_max_vals.append(float(values[-1]))

        # Sort by fitness descending
        local_max_vals.sort(
            key=lambda v: float(fitnesses[np.argmin(np.abs(values - v))]),
            reverse=True,
        )

        logger.debug(
            "1-D scan for '%s' around %.4f found %d local optima",
            param_name, current, len(local_max_vals),
        )
        return local_max_vals

    # ------------------------------------------------------------------
    # Cross-parameter sensitivity surface
    # ------------------------------------------------------------------

    def sensitivity_surface(
        self,
        params: List[str],
        n_points: int = 10,
    ) -> pd.DataFrame:
        """
        Compute a cross-parameter sensitivity table.

        For each pair (A, B) in params, vary A across n_points while holding
        all other params at base and record how fitness changes.  Also vary
        pairs simultaneously to capture interaction effects.

        Returns a DataFrame with columns:
            param_a, value_a, param_b, value_b, fitness
        """
        if len(params) < 2:
            raise ValueError("sensitivity_surface requires at least 2 parameters")

        rows = []
        base = dict(self.base_params)

        for i, pa in enumerate(params):
            for pb in params[i + 1 :]:
                if pa not in self.param_bounds or pb not in self.param_bounds:
                    continue

                lo_a, hi_a = self.param_bounds[pa]
                lo_b, hi_b = self.param_bounds[pb]
                vals_a = np.linspace(lo_a, hi_a, n_points)
                vals_b = np.linspace(lo_b, hi_b, n_points)

                for va in vals_a:
                    for vb in vals_b:
                        p = dict(base)
                        p[pa] = float(va)
                        p[pb] = float(vb)
                        fit = self._evaluate(p)
                        rows.append(
                            {
                                "param_a": pa,
                                "value_a": float(va),
                                "param_b": pb,
                                "value_b": float(vb),
                                "fitness": fit,
                            }
                        )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Exploration suggestions
    # ------------------------------------------------------------------

    def recommend_exploration(
        self,
        history: pd.DataFrame,
        n_suggestions: int = 5,
    ) -> List[ExplorationSuggestion]:
        """
        Identify under-explored parameter regions and return exploration
        suggestions.

        history must contain columns named 'p_{param_name}' (one per param)
        and 'fitness', as produced by GenomeAnalyzer.load_history.
        """
        suggestions: List[ExplorationSuggestion] = []

        param_cols = [c for c in history.columns if c.startswith("p_")]

        for col in param_cols:
            param_name = col[2:]
            if param_name not in self.param_bounds:
                continue

            lo, hi = self.param_bounds[param_name]
            total_range = hi - lo
            if total_range <= 0:
                continue

            visited_vals = history[col].dropna().values
            if len(visited_vals) == 0:
                # Entire space unexplored -- suggest midpoint
                suggestions.append(
                    ExplorationSuggestion(
                        param_name=param_name,
                        suggested_value=(lo + hi) / 2.0,
                        expected_fitness_gain=0.0,
                        exploration_type="unexplored_region",
                        confidence=0.3,
                    )
                )
                continue

            # Divide range into buckets and find the least visited
            n_buckets = 10
            bucket_edges = np.linspace(lo, hi, n_buckets + 1)
            bucket_counts = np.zeros(n_buckets, dtype=int)

            for v in visited_vals:
                bucket_idx = int(np.clip(
                    np.searchsorted(bucket_edges[1:], v),
                    0, n_buckets - 1,
                ))
                bucket_counts[bucket_idx] += 1

            least_visited_bucket = int(np.argmin(bucket_counts))
            bucket_mid = float(
                (bucket_edges[least_visited_bucket] + bucket_edges[least_visited_bucket + 1]) / 2
            )

            # Estimate expected fitness gain from visit history
            best_fitness = float(history["fitness"].max()) if "fitness" in history.columns else 0.0
            mean_fitness = float(history["fitness"].mean()) if "fitness" in history.columns else 0.0
            expected_gain = best_fitness - mean_fitness  # optimistic upper bound

            exploration_type = (
                "near_optimum"
                if bucket_counts[least_visited_bucket] == 0
                else "unexplored_region"
            )

            suggestions.append(
                ExplorationSuggestion(
                    param_name=param_name,
                    suggested_value=bucket_mid,
                    expected_fitness_gain=expected_gain,
                    exploration_type=exploration_type,
                    confidence=0.5 if bucket_counts[least_visited_bucket] == 0 else 0.3,
                )
            )

        # From _visited history, add ridge extension suggestions
        if self._visited and param_cols:
            best_params, best_fit = max(self._visited, key=lambda t: t[1])
            for param_name, best_val in best_params.items():
                if param_name not in self.param_bounds:
                    continue
                lo, hi = self.param_bounds[param_name]
                # Suggest extending 10% beyond current best in both directions
                for direction, etype in [(+1, "ridge_extension"), (-1, "diversity")]:
                    delta = (hi - lo) * 0.10 * direction
                    proposed = float(np.clip(best_val + delta, lo, hi))
                    if abs(proposed - best_val) < 1e-8:
                        continue
                    suggestions.append(
                        ExplorationSuggestion(
                            param_name=param_name,
                            suggested_value=proposed,
                            expected_fitness_gain=best_fit * 0.01,  # 1% of best as rough estimate
                            exploration_type=etype,
                            confidence=0.4,
                        )
                    )

        # Deduplicate and return top n_suggestions by expected_fitness_gain
        seen = set()
        unique: List[ExplorationSuggestion] = []
        for s in suggestions:
            key = (s.param_name, round(s.suggested_value, 4))
            if key not in seen:
                seen.add(key)
                unique.append(s)

        unique.sort(key=lambda s: s.expected_fitness_gain, reverse=True)
        return unique[:n_suggestions]

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def plot_landscape(self, landscape: LandscapeMap) -> plt.Figure:
        """
        Return a Figure showing the 2-D fitness surface as a filled contour
        plot with the optimal point and ridge path overlaid.
        """
        fig, ax = plt.subplots(figsize=(9, 7))

        xx, yy = np.meshgrid(landscape.grid_x, landscape.grid_y)
        surf = landscape.fitness_surface

        contour_filled = ax.contourf(xx, yy, surf, levels=20, cmap="viridis")
        ax.contour(xx, yy, surf, levels=20, colors="white", alpha=0.2, linewidths=0.5)

        fig.colorbar(contour_filled, ax=ax, label="Fitness")

        # Ridge path
        if landscape.ridge_path:
            rx, ry = zip(*landscape.ridge_path)
            ax.plot(rx, ry, color="red", linewidth=1.5, linestyle="--", label="Ridge path")

        # Optimal point
        ax.scatter(
            [landscape.optimal_x], [landscape.optimal_y],
            color="yellow", s=100, zorder=5, marker="*", label="Optimum",
        )

        ax.set_xlabel(landscape.param_x)
        ax.set_ylabel(landscape.param_y)
        ax.set_title(f"Fitness Landscape: {landscape.param_x} vs {landscape.param_y}")
        ax.legend(loc="upper right")

        plt.tight_layout()
        return fig

    def plot_1d_scan(
        self,
        param_name: str,
        current: float,
        radius: float = 0.25,
        n_points: int = 50,
    ) -> plt.Figure:
        """
        Plot the 1-D fitness profile around current and mark local optima.
        """
        lo, hi = self.param_bounds.get(param_name, (-1.0, 1.0))
        half_width = (hi - lo) * radius
        scan_lo = max(lo, current - half_width)
        scan_hi = min(hi, current + half_width)
        values = np.linspace(scan_lo, scan_hi, n_points)

        base = dict(self.base_params)
        fitnesses = []
        for val in values:
            p = dict(base)
            p[param_name] = float(val)
            fitnesses.append(self._evaluate(p))

        local_opts = self.find_local_optima(param_name, current, radius, n_points)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(values, fitnesses, color="#1f77b4", linewidth=2, label="Fitness")
        ax.axvline(current, color="gray", linestyle=":", label=f"Current ({current:.4f})")

        for opt_val in local_opts[:3]:
            ax.axvline(opt_val, color="red", linestyle="--", alpha=0.6)

        ax.set_xlabel(param_name)
        ax.set_ylabel("Fitness")
        ax.set_title(f"1-D Fitness Landscape: {param_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
