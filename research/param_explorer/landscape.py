"""
research/param_explorer/landscape.py
======================================
Objective landscape analysis: 2-D scanning, basin detection, roughness /
flatness / robustness metrics, and 2-D / 3-D visualisation.

Classes
-------
LandscapeGrid   : Result of a 2-D parameter scan
Basin           : A local optimum region
ObjectiveLandscape : Methods for scanning and characterising a landscape

Stand-alone helpers
-------------------
scan_2d           : Build a LandscapeGrid for two parameters
find_basins       : Identify local minima / maxima in a grid
roughness_index   : Total-variation proxy for landscape ruggedness
flatness_score    : Fraction of param space within threshold of global optimum
robustness_score  : Sensitivity of best params to small perturbations
plot_landscape_heatmap : 2-D colour-map
plot_landscape_3d      : 3-D surface plot
plot_contour           : Contour plot
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy.ndimage import label as ndimage_label, maximum_filter, minimum_filter

from research.param_explorer.space import ParamSpace, ParamSpec, ParamType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LandscapeGrid dataclass
# ---------------------------------------------------------------------------

@dataclass
class LandscapeGrid:
    """
    Result of a 2-D parameter sweep.

    Attributes
    ----------
    p1_name : str
        Name of the first swept parameter.
    p2_name : str
        Name of the second swept parameter.
    p1_values : np.ndarray of shape (n1,)
        Actual (decoded) values of parameter 1.
    p2_values : np.ndarray of shape (n2,)
        Actual (decoded) values of parameter 2.
    Z : np.ndarray of shape (n1, n2)
        Objective values on the grid.  Z[i, j] corresponds to
        (p1_values[i], p2_values[j]).
    base_params : dict
        Fixed parameter values used for all non-swept parameters.
    maximise : bool
        True if the objective was maximised (affects basin detection sign).
    """

    p1_name: str
    p2_name: str
    p1_values: np.ndarray
    p2_values: np.ndarray
    Z: np.ndarray
    base_params: Dict[str, Any]
    maximise: bool = True

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n1(self) -> int:
        return len(self.p1_values)

    @property
    def n2(self) -> int:
        return len(self.p2_values)

    @property
    def global_best_value(self) -> float:
        return float(np.nanmax(self.Z) if self.maximise else np.nanmin(self.Z))

    @property
    def global_best_idx(self) -> Tuple[int, int]:
        if self.maximise:
            idx = np.nanargmax(self.Z)
        else:
            idx = np.nanargmin(self.Z)
        return np.unravel_index(idx, self.Z.shape)  # type: ignore[return-value]

    @property
    def global_best_params(self) -> Dict[str, Any]:
        i, j = self.global_best_idx
        return {
            self.p1_name: float(self.p1_values[i]),
            self.p2_name: float(self.p2_values[j]),
        }

    @property
    def P1(self) -> np.ndarray:
        """Meshgrid P1 of shape (n1, n2)."""
        P1, _ = np.meshgrid(self.p1_values, self.p2_values, indexing="ij")
        return P1

    @property
    def P2(self) -> np.ndarray:
        """Meshgrid P2 of shape (n1, n2)."""
        _, P2 = np.meshgrid(self.p1_values, self.p2_values, indexing="ij")
        return P2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p1_name": self.p1_name,
            "p2_name": self.p2_name,
            "p1_values": self.p1_values.tolist(),
            "p2_values": self.p2_values.tolist(),
            "Z": self.Z.tolist(),
            "base_params": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in self.base_params.items()},
            "maximise": self.maximise,
        }


# ---------------------------------------------------------------------------
# Basin dataclass
# ---------------------------------------------------------------------------

@dataclass
class Basin:
    """
    A local optimum region in the landscape grid.

    Attributes
    ----------
    idx_center : tuple[int, int]
        Grid indices of the basin's peak / trough.
    p1_center : float
        Actual value of parameter 1 at the basin centre.
    p2_center : float
        Actual value of parameter 2 at the basin centre.
    peak_value : float
        Objective value at the basin centre.
    mask : np.ndarray of shape (n1, n2), dtype bool
        Grid cells belonging to this basin.
    area_fraction : float
        Fraction of the grid covered by this basin.
    """

    idx_center: Tuple[int, int]
    p1_center: float
    p2_center: float
    peak_value: float
    mask: np.ndarray
    area_fraction: float

    def __repr__(self) -> str:
        return (
            f"Basin(center=({self.p1_center:.4g}, {self.p2_center:.4g}), "
            f"peak={self.peak_value:.4g}, area={self.area_fraction:.2%})"
        )


# ---------------------------------------------------------------------------
# Stand-alone scan / analysis functions
# ---------------------------------------------------------------------------

def scan_2d(
    p1_name: str,
    p1_range: Tuple[float, float],
    p2_name: str,
    p2_range: Tuple[float, float],
    base_params: Dict[str, Any],
    objective_fn: Callable[[Dict[str, Any]], float],
    n1: int = 20,
    n2: int = 20,
    p1_log_scale: bool = False,
    p2_log_scale: bool = False,
    maximise: bool = True,
) -> LandscapeGrid:
    """
    Evaluate the objective on a 2-D grid over two parameters.

    All other parameters are held at their values in *base_params*.

    Parameters
    ----------
    p1_name, p2_name : str
        Names of the two parameters to sweep.
    p1_range, p2_range : tuple[float, float]
        (low, high) bounds for each parameter.
    base_params : dict
        Fixed baseline for all other parameters.
    objective_fn : callable
    n1, n2 : int
        Number of grid points along each axis.
    p1_log_scale, p2_log_scale : bool
        Log-spacing flags.
    maximise : bool

    Returns
    -------
    LandscapeGrid
    """
    if p1_log_scale:
        p1_vals = np.exp(np.linspace(math.log(p1_range[0]), math.log(p1_range[1]), n1))
    else:
        p1_vals = np.linspace(p1_range[0], p1_range[1], n1)

    if p2_log_scale:
        p2_vals = np.exp(np.linspace(math.log(p2_range[0]), math.log(p2_range[1]), n2))
    else:
        p2_vals = np.linspace(p2_range[0], p2_range[1], n2)

    Z = np.full((n1, n2), np.nan)
    total = n1 * n2
    log_every = max(1, total // 10)

    logger.info(
        "scan_2d: sweeping %s × %s on %d×%d grid (%d evaluations)…",
        p1_name, p2_name, n1, n2, total,
    )

    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            p = dict(base_params)
            p[p1_name] = v1
            p[p2_name] = v2
            try:
                Z[i, j] = objective_fn(p)
            except Exception as exc:
                logger.warning("scan_2d: eval failed at (%g, %g): %s", v1, v2, exc)
                Z[i, j] = np.nan

            done = i * n2 + j + 1
            if done % log_every == 0:
                logger.debug("scan_2d progress: %d/%d", done, total)

    return LandscapeGrid(
        p1_name=p1_name,
        p2_name=p2_name,
        p1_values=p1_vals,
        p2_values=p2_vals,
        Z=Z,
        base_params=dict(base_params),
        maximise=maximise,
    )


def find_basins(
    landscape_grid: LandscapeGrid,
    neighbourhood_size: int = 3,
    min_area_fraction: float = 0.01,
) -> List[Basin]:
    """
    Identify local optima (basins / peaks) in the landscape grid.

    Uses image-processing-style local maximum (or minimum) detection followed
    by connected-component labelling with a watershed-like flood-fill to
    delineate basin boundaries.

    Parameters
    ----------
    landscape_grid : LandscapeGrid
    neighbourhood_size : int
        Size of the filter window used to detect local optima.
    min_area_fraction : float
        Basins with fewer than this fraction of total cells are discarded.

    Returns
    -------
    list[Basin] sorted descending by peak value.
    """
    Z = landscape_grid.Z.copy()
    Z_filled = np.where(np.isnan(Z), np.nanmean(Z), Z)
    total_cells = Z.size

    if landscape_grid.maximise:
        local_max = maximum_filter(Z_filled, size=neighbourhood_size)
        is_peak = (Z_filled == local_max) & ~np.isnan(Z)
    else:
        local_min = minimum_filter(Z_filled, size=neighbourhood_size)
        is_peak = (Z_filled == local_min) & ~np.isnan(Z)

    # Label connected components of peaks
    struct = np.ones((neighbourhood_size, neighbourhood_size), dtype=int)
    labelled, n_features = ndimage_label(is_peak, structure=struct)

    basins: List[Basin] = []
    for label_id in range(1, n_features + 1):
        peak_mask = labelled == label_id
        peak_indices = np.argwhere(peak_mask)
        if len(peak_indices) == 0:
            continue

        # Find the single best point in this cluster
        peak_vals = Z_filled[peak_mask]
        if landscape_grid.maximise:
            best_local_idx_flat = np.argmax(peak_vals)
        else:
            best_local_idx_flat = np.argmin(peak_vals)
        best_local = peak_indices[best_local_idx_flat]
        i_c, j_c = int(best_local[0]), int(best_local[1])

        # Flood-fill to define the full basin
        threshold = float(peak_vals[best_local_idx_flat])
        if landscape_grid.maximise:
            # Basin = cells closer to this peak than to any other
            basin_mask = _assign_basin_mask(Z_filled, i_c, j_c, landscape_grid.maximise)
        else:
            basin_mask = _assign_basin_mask(Z_filled, i_c, j_c, landscape_grid.maximise)

        area_frac = float(basin_mask.sum()) / total_cells
        if area_frac < min_area_fraction:
            continue

        basins.append(
            Basin(
                idx_center=(i_c, j_c),
                p1_center=float(landscape_grid.p1_values[i_c]),
                p2_center=float(landscape_grid.p2_values[j_c]),
                peak_value=float(Z[i_c, j_c] if not np.isnan(Z[i_c, j_c]) else threshold),
                mask=basin_mask,
                area_fraction=area_frac,
            )
        )

    basins.sort(key=lambda b: b.peak_value, reverse=landscape_grid.maximise)
    logger.info("find_basins: found %d basins.", len(basins))
    return basins


def _assign_basin_mask(
    Z: np.ndarray,
    i_c: int,
    j_c: int,
    maximise: bool,
) -> np.ndarray:
    """
    Simple basin mask: cells for which this peak is the nearest local optimum
    by absolute objective difference.

    For simplicity we use a Voronoi-like assignment: each cell is assigned to
    the peak with the closest objective value (in gradient descent sense).
    Here we implement a BFS-based flood fill from (i_c, j_c) that expands to
    neighbours as long as the value is non-decreasing (for maximise) or
    non-increasing (for minimise).
    """
    n1, n2 = Z.shape
    mask = np.zeros((n1, n2), dtype=bool)
    visited = np.zeros((n1, n2), dtype=bool)

    from collections import deque
    queue = deque()
    queue.append((i_c, j_c))
    visited[i_c, j_c] = True
    mask[i_c, j_c] = True
    ref_val = Z[i_c, j_c]

    while queue:
        i, j = queue.popleft()
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni_, nj_ = i + di, j + dj
            if 0 <= ni_ < n1 and 0 <= nj_ < n2 and not visited[ni_, nj_]:
                visited[ni_, nj_] = True
                neighbour_val = Z[ni_, nj_]
                # Accept neighbour if it's within 10% of peak value
                if maximise:
                    if neighbour_val >= ref_val * 0.9:
                        mask[ni_, nj_] = True
                        queue.append((ni_, nj_))
                else:
                    if neighbour_val <= ref_val * 1.1:
                        mask[ni_, nj_] = True
                        queue.append((ni_, nj_))

    return mask


def roughness_index(landscape_grid: LandscapeGrid) -> float:
    """
    Compute a roughness index for the landscape grid.

    Based on normalised total variation:
        R = (Σ |∇Z|) / (n_cells * range(Z))

    A perfectly flat landscape has R=0; a very jagged landscape has large R.

    Parameters
    ----------
    landscape_grid : LandscapeGrid

    Returns
    -------
    float ∈ [0, ∞)
    """
    Z = landscape_grid.Z
    valid = ~np.isnan(Z)
    if valid.sum() < 4:
        return 0.0

    z_range = float(np.nanmax(Z) - np.nanmin(Z))
    if z_range < 1e-30:
        return 0.0

    # Finite differences along both axes
    grad_i = np.abs(np.diff(Z, axis=0))
    grad_j = np.abs(np.diff(Z, axis=1))

    tv_i = np.nansum(grad_i)
    tv_j = np.nansum(grad_j)
    total_tv = tv_i + tv_j

    n_cells = valid.sum()
    return float(total_tv / (n_cells * z_range))


def flatness_score(
    landscape_grid: LandscapeGrid,
    threshold: float = 0.1,
) -> float:
    """
    Fraction of the grid within *threshold* fraction of the global optimum.

    For example, threshold=0.10 counts cells whose objective is within
    10 % of (global_best - global_worst) from the global best.

    Parameters
    ----------
    landscape_grid : LandscapeGrid
    threshold : float in (0, 1]

    Returns
    -------
    float ∈ [0, 1]
    """
    Z = landscape_grid.Z
    z_best = float(np.nanmax(Z) if landscape_grid.maximise else np.nanmin(Z))
    z_worst = float(np.nanmin(Z) if landscape_grid.maximise else np.nanmax(Z))
    z_range = abs(z_best - z_worst)

    if z_range < 1e-30:
        return 1.0

    cutoff = z_range * threshold
    if landscape_grid.maximise:
        near_best = np.nansum(Z >= z_best - cutoff)
    else:
        near_best = np.nansum(Z <= z_best + cutoff)

    total_valid = np.sum(~np.isnan(Z))
    return float(near_best / total_valid) if total_valid > 0 else 0.0


def robustness_score(
    best_params: Dict[str, Any],
    objective_fn: Callable[[Dict[str, Any]], float],
    param_space: ParamSpace,
    n_perturbations: int = 200,
    perturb_scale: float = 0.05,
    seed: int = 0,
) -> float:
    """
    Measure how robust the objective is to small perturbations of *best_params*.

    Generates *n_perturbations* perturbed variants of *best_params* (±5 % in
    unit space by default) and evaluates the objective at each.

    Returns the ratio of the mean perturbed objective to the nominal objective:
        robustness = mean(f(perturbed)) / f(best_params)

    A score close to 1.0 indicates high robustness; a score much lower
    indicates the optimum is a narrow spike.

    Parameters
    ----------
    best_params : dict
    objective_fn : callable
    param_space : ParamSpace
    n_perturbations : int
    perturb_scale : float
        Standard deviation of perturbation in unit hypercube.
    seed : int

    Returns
    -------
    float
    """
    rng = np.random.default_rng(seed)
    f_nominal = objective_fn(best_params)
    if abs(f_nominal) < 1e-30:
        logger.warning("Nominal objective near zero; robustness score may be misleading.")

    f_perturbed = np.zeros(n_perturbations)
    for k in range(n_perturbations):
        p_perturbed = param_space.perturb(best_params, scale=perturb_scale, rng=rng)
        try:
            f_perturbed[k] = objective_fn(p_perturbed)
        except Exception:
            f_perturbed[k] = np.nan

    mean_perturbed = float(np.nanmean(f_perturbed))
    if abs(f_nominal) < 1e-30:
        return 1.0 if abs(mean_perturbed) < 1e-30 else 0.0
    return float(mean_perturbed / f_nominal)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_landscape_heatmap(
    landscape_grid: LandscapeGrid,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "viridis",
    mark_best: bool = True,
    basins: Optional[List[Basin]] = None,
) -> plt.Figure:
    """
    2-D colour-map of the objective landscape.

    Parameters
    ----------
    landscape_grid : LandscapeGrid
    save_path : str | Path | None
    figsize : tuple
    cmap : str
    mark_best : bool
        Mark the global best point with a star.
    basins : list[Basin] | None
        If provided, overlay basin boundaries.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Z = landscape_grid.Z
    p1 = landscape_grid.p1_values
    p2 = landscape_grid.p2_values

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(p2, p1, Z, cmap=cmap, shading="auto")
    plt.colorbar(im, ax=ax, label="Objective")

    if mark_best:
        i_best, j_best = landscape_grid.global_best_idx
        ax.plot(
            p2[j_best], p1[i_best],
            marker="*", color="red", markersize=14,
            label=f"Best ({landscape_grid.global_best_value:.4g})",
            zorder=5,
        )
        ax.legend(fontsize=9)

    if basins:
        for k, basin in enumerate(basins):
            ax.plot(
                basin.p2_center, basin.p1_center,
                marker="o", color="white", markersize=8,
                markeredgecolor="black", markeredgewidth=1.5,
                zorder=4,
            )
            ax.annotate(
                f"B{k+1}", (basin.p2_center, basin.p1_center),
                textcoords="offset points", xytext=(5, 5),
                color="white", fontsize=8, fontweight="bold",
            )

    ax.set_xlabel(landscape_grid.p2_name, fontsize=10)
    ax.set_ylabel(landscape_grid.p1_name, fontsize=10)
    ax.set_title(
        f"Objective Landscape: {landscape_grid.p1_name} × {landscape_grid.p2_name}\n"
        f"(grid {landscape_grid.n1}×{landscape_grid.n2})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Landscape heatmap saved to %s", save_path)

    return fig


def plot_landscape_3d(
    landscape_grid: LandscapeGrid,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 7),
    cmap: str = "viridis",
    elev: float = 25.0,
    azim: float = -60.0,
) -> plt.Figure:
    """
    3-D surface plot of the objective landscape.

    Parameters
    ----------
    landscape_grid : LandscapeGrid
    save_path : str | Path | None
    figsize : tuple
    cmap : str
    elev : float
        Elevation angle in degrees.
    azim : float
        Azimuth angle in degrees.

    Returns
    -------
    matplotlib.figure.Figure
    """
    P1 = landscape_grid.P1
    P2 = landscape_grid.P2
    Z = landscape_grid.Z.copy()
    Z[np.isnan(Z)] = float(np.nanmean(Z))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(P2, P1, Z, cmap=cmap, alpha=0.85, edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Objective")

    # Mark global best
    i_best, j_best = landscape_grid.global_best_idx
    ax.scatter(
        [landscape_grid.p2_values[j_best]],
        [landscape_grid.p1_values[i_best]],
        [landscape_grid.global_best_value],
        color="red", s=80, zorder=5, label="Best",
    )

    ax.set_xlabel(landscape_grid.p2_name, fontsize=9)
    ax.set_ylabel(landscape_grid.p1_name, fontsize=9)
    ax.set_zlabel("Objective", fontsize=9)
    ax.set_title(
        f"3-D Landscape: {landscape_grid.p1_name} × {landscape_grid.p2_name}",
        fontsize=11, fontweight="bold",
    )
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("3-D landscape plot saved to %s", save_path)

    return fig


def plot_contour(
    landscape_grid: LandscapeGrid,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 6),
    n_levels: int = 15,
    cmap: str = "viridis",
    mark_best: bool = True,
) -> plt.Figure:
    """
    Filled contour plot of the objective landscape.

    Parameters
    ----------
    landscape_grid : LandscapeGrid
    save_path : str | Path | None
    figsize : tuple
    n_levels : int
    cmap : str
    mark_best : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    Z = landscape_grid.Z.copy()
    p1 = landscape_grid.p1_values
    p2 = landscape_grid.p2_values

    fig, ax = plt.subplots(figsize=figsize)

    z_finite = Z[~np.isnan(Z)]
    levels = np.linspace(z_finite.min(), z_finite.max(), n_levels + 1)

    cf = ax.contourf(p2, p1, Z, levels=levels, cmap=cmap)
    cs = ax.contour(p2, p1, Z, levels=levels, colors="black", linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax, label="Objective")

    if mark_best:
        i_best, j_best = landscape_grid.global_best_idx
        ax.plot(
            p2[j_best], p1[i_best],
            marker="*", color="red", markersize=14,
            label=f"Best ({landscape_grid.global_best_value:.4g})",
        )
        ax.legend(fontsize=9)

    ax.set_xlabel(landscape_grid.p2_name, fontsize=10)
    ax.set_ylabel(landscape_grid.p1_name, fontsize=10)
    ax.set_title(
        f"Contour Plot: {landscape_grid.p1_name} × {landscape_grid.p2_name}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Contour plot saved to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# ObjectiveLandscape class
# ---------------------------------------------------------------------------

class ObjectiveLandscape:
    """
    High-level interface for landscape analysis.

    Parameters
    ----------
    param_space : ParamSpace
    objective_fn : callable
        Maps a parameter dict → scalar float.
    maximise : bool
        True if higher objective values are better.
    """

    def __init__(
        self,
        param_space: ParamSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
        maximise: bool = True,
    ) -> None:
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.maximise = maximise
        self._grids: Dict[Tuple[str, str], LandscapeGrid] = {}

    # ------------------------------------------------------------------
    # Grid scanning
    # ------------------------------------------------------------------

    def scan_2d(
        self,
        p1_name: str,
        p2_name: str,
        base_params: Optional[Dict[str, Any]] = None,
        n1: int = 20,
        n2: int = 20,
    ) -> LandscapeGrid:
        """
        Scan the objective over a 2-D grid of (p1_name, p2_name).

        Parameters
        ----------
        p1_name, p2_name : str
        base_params : dict | None
        n1, n2 : int

        Returns
        -------
        LandscapeGrid
        """
        if base_params is None:
            base_params = self.param_space.defaults

        spec1 = self.param_space[p1_name]
        spec2 = self.param_space[p2_name]

        grid = scan_2d(
            p1_name=p1_name,
            p1_range=(spec1.low, spec1.high),
            p2_name=p2_name,
            p2_range=(spec2.low, spec2.high),
            base_params=base_params,
            objective_fn=self.objective_fn,
            n1=n1,
            n2=n2,
            p1_log_scale=spec1.log_scale,
            p2_log_scale=spec2.log_scale,
            maximise=self.maximise,
        )
        self._grids[(p1_name, p2_name)] = grid
        return grid

    def scan_all_pairs(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        n1: int = 15,
        n2: int = 15,
        continuous_only: bool = True,
    ) -> Dict[Tuple[str, str], LandscapeGrid]:
        """
        Scan all unique 2-D pairs of parameters.

        Warning: runs d*(d-1)/2 separate grid evaluations.

        Returns
        -------
        dict[(p1_name, p2_name), LandscapeGrid]
        """
        names = [
            s.name for s in self.param_space.specs
            if not continuous_only or s.param_type != ParamType.CATEGORICAL
        ]
        results = {}
        for i, n1_name in enumerate(names):
            for n2_name in names[i + 1 :]:
                logger.info("Scanning pair (%s, %s)…", n1_name, n2_name)
                g = self.scan_2d(n1_name, n2_name, base_params, n1, n2)
                results[(n1_name, n2_name)] = g
        self._grids.update(results)
        return results

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def find_basins(
        self,
        grid: LandscapeGrid,
        neighbourhood_size: int = 3,
        min_area_fraction: float = 0.01,
    ) -> List[Basin]:
        """Identify local optima in *grid*."""
        return find_basins(grid, neighbourhood_size, min_area_fraction)

    def roughness_index(self, grid: LandscapeGrid) -> float:
        """Total-variation roughness of *grid*."""
        return roughness_index(grid)

    def flatness_score(
        self, grid: LandscapeGrid, threshold: float = 0.1
    ) -> float:
        """Fraction of grid within *threshold* of global best."""
        return flatness_score(grid, threshold)

    def robustness_score(
        self,
        best_params: Dict[str, Any],
        n_perturbations: int = 200,
        perturb_scale: float = 0.05,
        seed: int = 0,
    ) -> float:
        """Robustness of *best_params* to small perturbations."""
        return robustness_score(
            best_params,
            self.objective_fn,
            self.param_space,
            n_perturbations,
            perturb_scale,
            seed,
        )

    def full_analysis(
        self,
        p1_name: str,
        p2_name: str,
        base_params: Optional[Dict[str, Any]] = None,
        n1: int = 20,
        n2: int = 20,
        n_perturbations: int = 100,
    ) -> Dict[str, Any]:
        """
        Run a complete landscape analysis for one parameter pair.

        Returns
        -------
        dict with keys: grid, basins, roughness, flatness, robustness
        """
        grid = self.scan_2d(p1_name, p2_name, base_params, n1, n2)
        basins = self.find_basins(grid)
        rough = self.roughness_index(grid)
        flat = self.flatness_score(grid)

        best_full_params = dict(base_params or self.param_space.defaults)
        best_full_params.update(grid.global_best_params)
        rob = self.robustness_score(best_full_params, n_perturbations)

        return {
            "grid": grid,
            "basins": basins,
            "roughness": rough,
            "flatness": flat,
            "robustness": rob,
            "n_basins": len(basins),
            "global_best_params": grid.global_best_params,
            "global_best_value": grid.global_best_value,
        }

    # ------------------------------------------------------------------
    # Plot convenience wrappers
    # ------------------------------------------------------------------

    def plot_heatmap(
        self,
        grid: LandscapeGrid,
        save_path: Optional[Union[str, Path]] = None,
        basins: Optional[List[Basin]] = None,
    ) -> plt.Figure:
        return plot_landscape_heatmap(grid, save_path, basins=basins)

    def plot_3d(
        self,
        grid: LandscapeGrid,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        return plot_landscape_3d(grid, save_path)

    def plot_contour(
        self,
        grid: LandscapeGrid,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        return plot_contour(grid, save_path)
