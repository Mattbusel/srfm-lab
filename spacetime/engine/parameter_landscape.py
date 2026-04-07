"""
spacetime/engine/parameter_landscape.py
==========================================
3D parameter landscape analysis for the Spacetime Arena.

Maps the BH (Black Hole) mass parameter space to visualize how parameter
choices affect system behavior. Generates landscape data for the Sensitivity
page: grid surfaces, stable region detection, gradient computation, and
robustness scoring.

Classes:
  ParameterRegion        -- convex hull of a stable cluster in parameter space
  LandscapePoint         -- single (param1, param2, metrics) record
  LandscapeGrid          -- 2D grid of LandscapePoints + helpers
  LandscapeCache         -- DuckDB-backed cache of computed landscape slices
  ParameterLandscapeAnalyzer -- main analysis class

Requires: numpy, pandas, duckdb (optional -- degrades gracefully)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bh_engine import GEEKY_DEFAULTS, INSTRUMENT_CONFIGS, run_backtest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional DuckDB
# ---------------------------------------------------------------------------

try:
    import duckdb
    _DUCKDB = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    _DUCKDB = False
    logger.debug("DuckDB not available -- LandscapeCache will use in-memory store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RESOLUTION = 20
_DEFAULT_CACHE_PATH = Path("landscape_cache.duckdb")
_STABLE_SHARPE_THRESHOLD = 1.0   -- points with Sharpe >= this are "stable"
_ROBUST_SHARPE_FLOOR = 0.5       -- perturbation must keep Sharpe above this
_PERTURB_FRAC = 0.10             -- +-10% perturbation for robustness test


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LandscapePoint:
    """Single evaluated point in a 2D parameter grid."""
    param1_name: str
    param2_name: str
    param1_val: float
    param2_val: float
    sharpe: float
    max_dd: float
    n_trades: int
    calmar: float = 0.0
    win_rate: float = 0.0
    eval_time_s: float = 0.0

    @property
    def is_stable(self) -> bool:
        return self.sharpe >= _STABLE_SHARPE_THRESHOLD


@dataclass
class ParameterRegion:
    """
    Convex hull of a cluster of stable parameter-space points.

    Stored as the axis-aligned bounding box (AABB) for simplicity --
    a true convex hull is approximated by the min/max extents in each dimension.
    """
    param1_name: str
    param2_name: str
    param1_min: float
    param1_max: float
    param2_min: float
    param2_max: float
    centroid_param1: float
    centroid_param2: float
    n_points: int
    mean_sharpe: float
    max_sharpe: float

    def contains(self, p1: float, p2: float) -> bool:
        """Return True if (p1, p2) lies inside this region's bounding box."""
        return (
            self.param1_min <= p1 <= self.param1_max
            and self.param2_min <= p2 <= self.param2_max
        )

    def area(self) -> float:
        return (self.param1_max - self.param1_min) * (self.param2_max - self.param2_min)


@dataclass
class LandscapeGrid:
    """
    2D grid of LandscapePoints with helper accessors.
    """
    param1_name: str
    param2_name: str
    param1_vals: np.ndarray
    param2_vals: np.ndarray
    points: List[LandscapePoint] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all grid points as a DataFrame."""
        rows = [asdict(p) for p in self.points]
        return pd.DataFrame(rows)

    def sharpe_matrix(self) -> np.ndarray:
        """Return (n1, n2) matrix of Sharpe values for surface plotting."""
        n1 = len(self.param1_vals)
        n2 = len(self.param2_vals)
        mat = np.full((n1, n2), np.nan)
        p1_idx = {v: i for i, v in enumerate(self.param1_vals)}
        p2_idx = {v: i for i, v in enumerate(self.param2_vals)}
        for pt in self.points:
            i = p1_idx.get(pt.param1_val)
            j = p2_idx.get(pt.param2_val)
            if i is not None and j is not None:
                mat[i, j] = pt.sharpe
        return mat

    def to_surface_dict(self) -> Dict[str, Any]:
        """
        Return dict suitable for a 3D surface plot:
          {param1: [...], param2: [...], sharpe_matrix: [[...]], max_dd_matrix: [[...]]}
        """
        n1 = len(self.param1_vals)
        n2 = len(self.param2_vals)
        sharpe_mat = self.sharpe_matrix()
        dd_mat = np.full((n1, n2), np.nan)
        p1_idx = {v: i for i, v in enumerate(self.param1_vals)}
        p2_idx = {v: i for i, v in enumerate(self.param2_vals)}
        for pt in self.points:
            i = p1_idx.get(pt.param1_val)
            j = p2_idx.get(pt.param2_val)
            if i is not None and j is not None:
                dd_mat[i, j] = pt.max_dd
        return {
            self.param1_name: self.param1_vals.tolist(),
            self.param2_name: self.param2_vals.tolist(),
            "sharpe_matrix": sharpe_mat.tolist(),
            "max_dd_matrix": dd_mat.tolist(),
            "n_evaluated": len(self.points),
        }

    def stable_points(self) -> List[LandscapePoint]:
        return [p for p in self.points if p.is_stable]

    def best_point(self) -> Optional[LandscapePoint]:
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.sharpe)


# ---------------------------------------------------------------------------
# LandscapeCache -- DuckDB-backed
# ---------------------------------------------------------------------------

class LandscapeCache:
    """
    DuckDB-backed cache of computed landscape slices.

    Cache key = hash of (sym, param1, param2, resolution, base_params).
    If DuckDB is unavailable, falls back to an in-memory dict.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_CACHE_PATH
        self._memory: Dict[str, LandscapeGrid] = {}  -- fallback
        self._conn: Optional[Any] = None
        if _DUCKDB:
            try:
                self._conn = duckdb.connect(str(self.db_path))
                self._init_schema()
            except Exception as exc:
                logger.warning("DuckDB connect failed (%s) -- using memory cache", exc)
                self._conn = None

    def _init_schema(self) -> None:
        if self._conn is None:
            return
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS landscape_cache (
                cache_key   VARCHAR PRIMARY KEY,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sym         VARCHAR,
                param1      VARCHAR,
                param2      VARCHAR,
                resolution  INTEGER,
                grid_json   VARCHAR
            )
        """)

    @staticmethod
    def _make_key(
        sym: str, param1: str, param2: str, resolution: int, base_params: Dict
    ) -> str:
        raw = json.dumps(
            {"sym": sym, "p1": param1, "p2": param2, "res": resolution,
             "bp": base_params},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(
        self, sym: str, param1: str, param2: str,
        resolution: int, base_params: Dict
    ) -> Optional[LandscapeGrid]:
        key = self._make_key(sym, param1, param2, resolution, base_params)

        if self._conn is not None:
            try:
                row = self._conn.execute(
                    "SELECT grid_json FROM landscape_cache WHERE cache_key = ?",
                    [key],
                ).fetchone()
                if row:
                    return self._deserialize(row[0])
            except Exception as exc:
                logger.debug("Cache get error: %s", exc)

        return self._memory.get(key)

    def put(
        self, sym: str, param1: str, param2: str, resolution: int,
        base_params: Dict, grid: LandscapeGrid
    ) -> None:
        key = self._make_key(sym, param1, param2, resolution, base_params)
        serialized = self._serialize(grid)

        if self._conn is not None:
            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO landscape_cache
                    (cache_key, sym, param1, param2, resolution, grid_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [key, sym, param1, param2, resolution, serialized],
                )
            except Exception as exc:
                logger.debug("Cache put error: %s", exc)

        self._memory[key] = grid

    @staticmethod
    def _serialize(grid: LandscapeGrid) -> str:
        data = {
            "param1_name": grid.param1_name,
            "param2_name": grid.param2_name,
            "param1_vals": grid.param1_vals.tolist(),
            "param2_vals": grid.param2_vals.tolist(),
            "points": [asdict(p) for p in grid.points],
        }
        return json.dumps(data)

    @staticmethod
    def _deserialize(s: str) -> LandscapeGrid:
        data = json.loads(s)
        points = [LandscapePoint(**p) for p in data["points"]]
        return LandscapeGrid(
            param1_name=data["param1_name"],
            param2_name=data["param2_name"],
            param1_vals=np.array(data["param1_vals"]),
            param2_vals=np.array(data["param2_vals"]),
            points=points,
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Cluster helper -- simple density-based grouping
# ---------------------------------------------------------------------------

def _cluster_stable_points(
    points: List[LandscapePoint], eps_frac: float = 0.2
) -> List[List[LandscapePoint]]:
    """
    Simple grid-based clustering of stable points.

    Groups points whose (p1, p2) coordinates are within eps_frac of the
    full parameter range of each other. Returns list of clusters.
    """
    if not points:
        return []

    p1_vals = np.array([p.param1_val for p in points])
    p2_vals = np.array([p.param2_val for p in points])
    p1_range = p1_vals.max() - p1_vals.min() if p1_vals.max() > p1_vals.min() else 1.0
    p2_range = p2_vals.max() - p2_vals.min() if p2_vals.max() > p2_vals.min() else 1.0
    eps1 = eps_frac * p1_range
    eps2 = eps_frac * p2_range

    visited = [False] * len(points)
    clusters: List[List[LandscapePoint]] = []

    for i, pt in enumerate(points):
        if visited[i]:
            continue
        cluster = [pt]
        visited[i] = True
        for j, other in enumerate(points):
            if visited[j]:
                continue
            if abs(other.param1_val - pt.param1_val) <= eps1 and \
               abs(other.param2_val - pt.param2_val) <= eps2:
                cluster.append(other)
                visited[j] = True
        clusters.append(cluster)

    return clusters


def _cluster_to_region(cluster: List[LandscapePoint]) -> ParameterRegion:
    """Convert a cluster of LandscapePoints to a ParameterRegion."""
    p1_vals = [p.param1_val for p in cluster]
    p2_vals = [p.param2_val for p in cluster]
    sharpes = [p.sharpe for p in cluster]
    return ParameterRegion(
        param1_name=cluster[0].param1_name,
        param2_name=cluster[0].param2_name,
        param1_min=min(p1_vals),
        param1_max=max(p1_vals),
        param2_min=min(p2_vals),
        param2_max=max(p2_vals),
        centroid_param1=float(np.mean(p1_vals)),
        centroid_param2=float(np.mean(p2_vals)),
        n_points=len(cluster),
        mean_sharpe=float(np.mean(sharpes)),
        max_sharpe=float(np.max(sharpes)),
    )


# ---------------------------------------------------------------------------
# ParameterLandscapeAnalyzer
# ---------------------------------------------------------------------------

class ParameterLandscapeAnalyzer:
    """
    Maps the BH mass parameter space to visualize how parameter choices
    affect system behavior. Generates landscape data for the Sensitivity page.

    Parameters
    ----------
    sym : str
        Instrument symbol (e.g. "ES", "NQ").
    bars : pd.DataFrame
        OHLCV bar data.
    base_params : dict, optional
        Parameter overrides on top of instrument + Geeky defaults.
    cache : LandscapeCache, optional
        If provided, computed grids are cached and reused.
    long_only : bool
        Whether backtest runs long-only.
    """

    # -- parameters that can be used as landscape axes
    LANDSCAPE_PARAMS = {
        "cf": (0.0005, 0.05),
        "bh_form": (0.5, 3.0),
        "bh_decay": (0.80, 0.99),
        "bh_collapse": (0.3, 2.0),
    }

    def __init__(
        self,
        sym: str,
        bars: pd.DataFrame,
        base_params: Optional[Dict[str, Any]] = None,
        cache: Optional[LandscapeCache] = None,
        long_only: bool = True,
    ):
        self.sym = sym.upper()
        self.bars = bars
        self.long_only = long_only
        self.cache = cache

        cfg = dict(INSTRUMENT_CONFIGS.get(self.sym, INSTRUMENT_CONFIGS["ES"]))
        cfg.update(GEEKY_DEFAULTS)
        if base_params:
            cfg.update(base_params)
        self.base_params = cfg

    def _run_point(self, params: Dict[str, Any]) -> LandscapePoint:
        """Run a single backtest and return a LandscapePoint."""
        raise NotImplementedError(
            "_run_point should be implemented by callers or subclass. "
            "See compute_2d_landscape for usage."
        )

    def _evaluate_params(
        self, param1: str, param2: str, p1_val: float, p2_val: float
    ) -> LandscapePoint:
        """Build params dict, run backtest, return LandscapePoint."""
        params = dict(self.base_params)
        params[param1] = p1_val
        params[param2] = p2_val

        t0 = time.perf_counter()
        try:
            result = run_backtest(self.sym, self.bars, params, self.long_only)
            sharpe = float(result.get("sharpe", 0.0))
            max_dd = float(result.get("max_drawdown", 0.0))
            n_trades = int(result.get("n_trades", 0))
            calmar = float(result.get("calmar", 0.0))
            win_rate = float(result.get("win_rate", 0.0))
        except Exception as exc:
            logger.debug("Backtest failed at (%s=%.4f, %s=%.4f): %s",
                         param1, p1_val, param2, p2_val, exc)
            sharpe, max_dd, n_trades, calmar, win_rate = 0.0, 0.0, 0, 0.0, 0.0

        elapsed = time.perf_counter() - t0
        return LandscapePoint(
            param1_name=param1,
            param2_name=param2,
            param1_val=p1_val,
            param2_val=p2_val,
            sharpe=sharpe,
            max_dd=max_dd,
            n_trades=n_trades,
            calmar=calmar,
            win_rate=win_rate,
            eval_time_s=elapsed,
        )

    def compute_2d_landscape(
        self,
        param1: str,
        param2: str,
        resolution: int = _DEFAULT_RESOLUTION,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Grid search over param1 x param2.

        Computes {sharpe, max_dd, n_trades} at each of (resolution x resolution)
        grid points. Returns a dict with surface plot data.

        Parameters
        ----------
        param1 : str
            Name of first parameter (must be in LANDSCAPE_PARAMS).
        param2 : str
            Name of second parameter (must be in LANDSCAPE_PARAMS).
        resolution : int
            Number of grid points per axis.
        use_cache : bool
            Whether to use LandscapeCache.

        Returns
        -------
        dict
            Surface plot data + metadata. See LandscapeGrid.to_surface_dict().
        """
        if param1 not in self.LANDSCAPE_PARAMS:
            raise ValueError(f"param1 '{param1}' not in LANDSCAPE_PARAMS")
        if param2 not in self.LANDSCAPE_PARAMS:
            raise ValueError(f"param2 '{param2}' not in LANDSCAPE_PARAMS")

        # -- check cache
        if use_cache and self.cache is not None:
            cached = self.cache.get(
                self.sym, param1, param2, resolution, self.base_params
            )
            if cached is not None:
                logger.info("Landscape cache hit for %s x %s", param1, param2)
                return cached.to_surface_dict()

        p1_range = self.LANDSCAPE_PARAMS[param1]
        p2_range = self.LANDSCAPE_PARAMS[param2]
        p1_vals = np.linspace(p1_range[0], p1_range[1], resolution)
        p2_vals = np.linspace(p2_range[0], p2_range[1], resolution)

        grid = LandscapeGrid(
            param1_name=param1,
            param2_name=param2,
            param1_vals=p1_vals,
            param2_vals=p2_vals,
        )

        total = resolution * resolution
        logger.info(
            "Computing 2D landscape: %s x %s, %d points",
            param1, param2, total,
        )

        for i, p1 in enumerate(p1_vals):
            for j, p2 in enumerate(p2_vals):
                pt = self._evaluate_params(param1, param2, float(p1), float(p2))
                grid.points.append(pt)

            if (i + 1) % max(1, resolution // 5) == 0:
                done = (i + 1) * resolution
                logger.info("  ... %d/%d points evaluated", done, total)

        # -- store in cache
        if use_cache and self.cache is not None:
            self.cache.put(self.sym, param1, param2, resolution, self.base_params, grid)

        return grid.to_surface_dict()

    def find_stable_regions(
        self,
        param1: str = "bh_form",
        param2: str = "bh_decay",
        resolution: int = _DEFAULT_RESOLUTION,
    ) -> List[ParameterRegion]:
        """
        Cluster grid points with Sharpe >= STABLE_SHARPE_THRESHOLD.

        Returns convex hulls (approximated as AABBs) of stable clusters.
        """
        surface = self.compute_2d_landscape(param1, param2, resolution)
        p1_vals = surface[param1]
        p2_vals = surface[param2]
        sharpe_mat = np.array(surface["sharpe_matrix"])

        stable_pts: List[LandscapePoint] = []
        for i, p1 in enumerate(p1_vals):
            for j, p2 in enumerate(p2_vals):
                s = float(sharpe_mat[i][j]) if not math.isnan(sharpe_mat[i][j]) else 0.0
                if s >= _STABLE_SHARPE_THRESHOLD:
                    stable_pts.append(LandscapePoint(
                        param1_name=param1, param2_name=param2,
                        param1_val=float(p1), param2_val=float(p2),
                        sharpe=s, max_dd=0.0, n_trades=0,
                    ))

        if not stable_pts:
            logger.info("No stable regions found (Sharpe >= %.1f)", _STABLE_SHARPE_THRESHOLD)
            return []

        clusters = _cluster_stable_points(stable_pts)
        regions = [_cluster_to_region(c) for c in clusters if len(c) >= 2]
        regions.sort(key=lambda r: r.mean_sharpe, reverse=True)
        logger.info("Found %d stable regions", len(regions))
        return regions

    def compute_gradient_at_optimum(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Numerical gradient of Sharpe w.r.t. each landscape parameter at current values.

        Uses central differences with step = 1% of the parameter's range.
        Returns {param_name: d_sharpe/d_param}.
        """
        params = {**self.base_params, **params}
        gradients: Dict[str, float] = {}

        for pname, (lo, hi) in self.LANDSCAPE_PARAMS.items():
            if pname not in params:
                continue
            h = 0.01 * (hi - lo)
            p_hi = dict(params)
            p_lo = dict(params)
            p_hi[pname] = min(float(params[pname]) + h, hi)
            p_lo[pname] = max(float(params[pname]) - h, lo)
            actual_h = p_hi[pname] - p_lo[pname]
            if actual_h < 1e-12:
                gradients[pname] = 0.0
                continue

            try:
                r_hi = run_backtest(self.sym, self.bars, p_hi, self.long_only)
                r_lo = run_backtest(self.sym, self.bars, p_lo, self.long_only)
                s_hi = float(r_hi.get("sharpe", 0.0))
                s_lo = float(r_lo.get("sharpe", 0.0))
                gradients[pname] = (s_hi - s_lo) / actual_h
            except Exception as exc:
                logger.debug("Gradient eval error for %s: %s", pname, exc)
                gradients[pname] = 0.0

        return gradients

    def robustness_score(
        self, params: Dict[str, Any], n_perturbations: int = 100
    ) -> float:
        """
        Randomly perturb params by +-10%, return fraction of perturbations
        with Sharpe >= ROBUST_SHARPE_FLOOR (default 0.5).

        Parameters
        ----------
        params : dict
            Base parameter dict.
        n_perturbations : int
            Number of random perturbations to test.

        Returns
        -------
        float
            Fraction of perturbations that maintain acceptable Sharpe. [0, 1].
        """
        rng = np.random.default_rng(42)
        base = {**self.base_params, **params}
        successes = 0

        for _ in range(n_perturbations):
            perturbed = dict(base)
            for pname, (lo, hi) in self.LANDSCAPE_PARAMS.items():
                if pname not in base:
                    continue
                val = float(base[pname])
                delta = rng.uniform(-_PERTURB_FRAC, _PERTURB_FRAC) * val
                perturbed[pname] = float(np.clip(val + delta, lo, hi))

            try:
                result = run_backtest(self.sym, self.bars, perturbed, self.long_only)
                sharpe = float(result.get("sharpe", 0.0))
                if sharpe >= _ROBUST_SHARPE_FLOOR:
                    successes += 1
            except Exception:
                pass  -- count as failure

        score = successes / n_perturbations
        logger.info(
            "Robustness score: %.2f (%d/%d perturbations survived)",
            score, successes, n_perturbations,
        )
        return score

    def full_landscape_report(
        self, resolution: int = 15
    ) -> Dict[str, Any]:
        """
        Compute landscapes for all parameter pair combinations.

        Returns a dict of {param1_x_param2: surface_dict} for all pairs,
        plus a list of stable regions and robustness scores at the base config.
        """
        param_names = list(self.LANDSCAPE_PARAMS.keys())
        landscapes: Dict[str, Any] = {}
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                p1, p2 = param_names[i], param_names[j]
                key = f"{p1}_x_{p2}"
                logger.info("Computing landscape: %s", key)
                landscapes[key] = self.compute_2d_landscape(p1, p2, resolution)

        stable = self.find_stable_regions(resolution=resolution)
        robustness = self.robustness_score(self.base_params)

        return {
            "sym": self.sym,
            "landscapes": landscapes,
            "stable_regions": [asdict(r) for r in stable],
            "robustness_at_base": robustness,
            "base_params": {k: self.base_params[k]
                            for k in self.LANDSCAPE_PARAMS if k in self.base_params},
        }
