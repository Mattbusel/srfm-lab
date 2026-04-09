"""
grid_search.py — Parameter grid search infrastructure for the backtest farm.

Provides parameter space definitions, multiple search generators (grid,
random, Sobol, Bayesian), a runner with parallel execution, SQLite result
storage, landscape analysis, and optimal parameter discovery.

Dependencies: numpy, scipy (only).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sqlite3
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm as sp_norm
from scipy.stats.qmc import Sobol


# ---------------------------------------------------------------------------
# Parameter space primitives
# ---------------------------------------------------------------------------

@dataclass
class ParamRange:
    """One parameter's search range."""

    name: str
    mode: str          # "linspace", "logspace", "choice"
    low: float = 0.0
    high: float = 1.0
    n_points: int = 10
    choices: list[Any] | None = None

    def __post_init__(self):
        if self.mode not in ("linspace", "logspace", "choice"):
            raise ValueError(f"Unknown mode '{self.mode}'")
        if self.mode == "choice" and not self.choices:
            raise ValueError("choice mode requires non-empty choices list")

    def grid_values(self) -> list[Any]:
        """Return the discrete grid of values."""
        if self.mode == "linspace":
            return np.linspace(self.low, self.high, self.n_points).tolist()
        elif self.mode == "logspace":
            return np.logspace(
                np.log10(max(self.low, 1e-12)),
                np.log10(max(self.high, 1e-12)),
                self.n_points,
            ).tolist()
        else:
            return list(self.choices)

    def sample_uniform(self, rng: np.random.Generator) -> Any:
        """Draw one uniform random sample."""
        if self.mode == "choice":
            return self.choices[rng.integers(len(self.choices))]
        elif self.mode == "linspace":
            return float(rng.uniform(self.low, self.high))
        else:
            log_low = np.log10(max(self.low, 1e-12))
            log_high = np.log10(max(self.high, 1e-12))
            return float(10 ** rng.uniform(log_low, log_high))

    def from_unit(self, u: float) -> Any:
        """Map a [0,1] value to the parameter's domain."""
        if self.mode == "choice":
            idx = int(u * len(self.choices))
            idx = min(idx, len(self.choices) - 1)
            return self.choices[idx]
        elif self.mode == "linspace":
            return float(self.low + u * (self.high - self.low))
        else:
            log_low = np.log10(max(self.low, 1e-12))
            log_high = np.log10(max(self.high, 1e-12))
            return float(10 ** (log_low + u * (log_high - log_low)))

    def to_unit(self, value: Any) -> float:
        """Map a parameter value to [0,1]."""
        if self.mode == "choice":
            try:
                idx = self.choices.index(value)
            except ValueError:
                idx = 0
            return (idx + 0.5) / len(self.choices)
        elif self.mode == "linspace":
            span = self.high - self.low
            if span < 1e-15:
                return 0.5
            return (value - self.low) / span
        else:
            log_low = np.log10(max(self.low, 1e-12))
            log_high = np.log10(max(self.high, 1e-12))
            span = log_high - log_low
            if span < 1e-15:
                return 0.5
            return (np.log10(max(value, 1e-12)) - log_low) / span


class ParameterSpace:
    """Define the search space for a strategy."""

    def __init__(self, ranges: list[ParamRange] | None = None):
        self.ranges: list[ParamRange] = ranges or []
        self._name_map: dict[str, int] = {}
        for i, r in enumerate(self.ranges):
            self._name_map[r.name] = i

    def add(self, param_range: ParamRange) -> "ParameterSpace":
        """Add a parameter range.  Returns self for chaining."""
        self._name_map[param_range.name] = len(self.ranges)
        self.ranges.append(param_range)
        return self

    def add_linspace(self, name: str, low: float, high: float, n: int = 10) -> "ParameterSpace":
        return self.add(ParamRange(name=name, mode="linspace", low=low, high=high, n_points=n))

    def add_logspace(self, name: str, low: float, high: float, n: int = 10) -> "ParameterSpace":
        return self.add(ParamRange(name=name, mode="logspace", low=low, high=high, n_points=n))

    def add_choice(self, name: str, choices: list[Any]) -> "ParameterSpace":
        return self.add(ParamRange(name=name, mode="choice", choices=choices))

    @property
    def dim(self) -> int:
        return len(self.ranges)

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.ranges]

    def total_grid_size(self) -> int:
        """Total number of points in the full Cartesian grid."""
        if not self.ranges:
            return 0
        total = 1
        for r in self.ranges:
            total *= len(r.grid_values())
        return total

    def combo_to_dict(self, combo: tuple) -> dict[str, Any]:
        """Convert a parameter combo tuple to a named dictionary."""
        return {r.name: v for r, v in zip(self.ranges, combo)}

    def unit_to_dict(self, unit_vec: np.ndarray) -> dict[str, Any]:
        """Convert a [0,1]^d vector to a named parameter dictionary."""
        return {r.name: r.from_unit(float(unit_vec[i])) for i, r in enumerate(self.ranges)}

    def dict_to_unit(self, d: dict[str, Any]) -> np.ndarray:
        """Convert a named dictionary to a [0,1]^d vector."""
        out = np.zeros(self.dim)
        for i, r in enumerate(self.ranges):
            out[i] = r.to_unit(d.get(r.name, 0.0))
        return out


# ---------------------------------------------------------------------------
# Search generators
# ---------------------------------------------------------------------------

class GridGenerator:
    """Full Cartesian product grid search."""

    def __init__(self, space: ParameterSpace):
        self.space = space

    def generate(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        grids = [r.grid_values() for r in self.space.ranges]
        combos = list(itertools.product(*grids))
        return [self.space.combo_to_dict(c) for c in combos]

    def __len__(self) -> int:
        return self.space.total_grid_size()


class RandomSearchGenerator:
    """Uniform random sampling from the parameter space."""

    def __init__(self, space: ParameterSpace, n_samples: int = 100, seed: int = 42):
        self.space = space
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> list[dict[str, Any]]:
        combos = []
        for _ in range(self.n_samples):
            d = {}
            for r in self.space.ranges:
                d[r.name] = r.sample_uniform(self.rng)
            combos.append(d)
        return combos

    def __len__(self) -> int:
        return self.n_samples


class SobolSearchGenerator:
    """Quasi-random Sobol sequence sampling for low-discrepancy coverage."""

    def __init__(self, space: ParameterSpace, n_samples: int = 128, seed: int = 42):
        self.space = space
        # Sobol requires n_samples to be a power of 2
        exp = max(int(np.ceil(np.log2(max(n_samples, 1)))), 1)
        self.n_samples = 2 ** exp
        self.seed = seed

    def generate(self) -> list[dict[str, Any]]:
        if self.space.dim == 0:
            return [{}]
        sampler = Sobol(d=self.space.dim, scramble=True, seed=self.seed)
        unit_samples = sampler.random(self.n_samples)
        combos = []
        for row in unit_samples:
            combos.append(self.space.unit_to_dict(row))
        return combos

    def __len__(self) -> int:
        return self.n_samples


class BayesianSearchGenerator:
    """
    Bayesian optimization using a GP surrogate with UCB acquisition.

    Uses a simple RBF-kernel GP implemented from scratch (no sklearn).
    """

    def __init__(
        self,
        space: ParameterSpace,
        n_initial: int = 10,
        n_iterations: int = 50,
        kappa: float = 2.0,
        seed: int = 42,
    ):
        self.space = space
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

        self.X_observed: list[np.ndarray] = []
        self.y_observed: list[float] = []
        self.length_scale = 0.3
        self.noise = 1e-4

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        sq_dist = cdist(X1, X2, metric="sqeuclidean")
        return np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def _gp_predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GP posterior mean and standard deviation at X_new."""
        if len(self.X_observed) == 0:
            return np.zeros(len(X_new)), np.ones(len(X_new))

        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)

        K = self._rbf_kernel(X_obs, X_obs) + self.noise * np.eye(len(X_obs))
        K_star = self._rbf_kernel(X_new, X_obs)

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            mu = K_star @ alpha

            v = np.linalg.solve(L, K_star.T)
            var = 1.0 - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)
        except np.linalg.LinAlgError:
            mu = np.full(len(X_new), np.mean(y_obs))
            std = np.ones(len(X_new))

        return mu, std

    def _ucb(self, X_candidates: np.ndarray) -> np.ndarray:
        """Upper confidence bound acquisition function."""
        mu, std = self._gp_predict(X_candidates)
        return mu + self.kappa * std

    def initial_points(self) -> list[dict[str, Any]]:
        """Generate initial random points."""
        points = []
        for _ in range(self.n_initial):
            u = self.rng.uniform(0, 1, size=self.space.dim)
            points.append(self.space.unit_to_dict(u))
        return points

    def suggest_next(self) -> dict[str, Any]:
        """Suggest the next point to evaluate using UCB."""
        n_candidates = 1000
        candidates = self.rng.uniform(0, 1, size=(n_candidates, self.space.dim))
        ucb_vals = self._ucb(candidates)
        best_idx = np.argmax(ucb_vals)
        return self.space.unit_to_dict(candidates[best_idx])

    def observe(self, params: dict[str, Any], score: float) -> None:
        """Record an observation."""
        u = self.space.dict_to_unit(params)
        self.X_observed.append(u)
        self.y_observed.append(score)

    def best_observed(self) -> tuple[dict[str, Any], float]:
        """Return the best observed point and its score."""
        if not self.y_observed:
            return {}, float("-inf")
        idx = int(np.argmax(self.y_observed))
        return self.space.unit_to_dict(self.X_observed[idx]), self.y_observed[idx]


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Configuration for a single strategy's grid search."""
    strategy_name: str
    param_space: ParameterSpace
    n_evaluations: int = 100
    search_method: str = "random"  # "grid", "random", "sobol", "bayesian"
    objective: str = "sharpe"
    min_periods: int = 252

    def __post_init__(self):
        if self.search_method not in ("grid", "random", "sobol", "bayesian"):
            raise ValueError(f"Unknown search_method '{self.search_method}'")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_sharpe(returns: np.ndarray, signal: np.ndarray, ann_factor: float = np.sqrt(252)) -> float:
    """Compute annualised Sharpe ratio of a strategy."""
    strat_returns = signal[:-1] * returns[1:]
    if len(strat_returns) < 10:
        return 0.0
    mu = np.mean(strat_returns)
    sd = np.std(strat_returns, ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(mu / sd * ann_factor)


def _compute_sortino(returns: np.ndarray, signal: np.ndarray, ann_factor: float = np.sqrt(252)) -> float:
    """Compute annualised Sortino ratio."""
    strat_returns = signal[:-1] * returns[1:]
    if len(strat_returns) < 10:
        return 0.0
    mu = np.mean(strat_returns)
    downside = strat_returns[strat_returns < 0]
    if len(downside) < 2:
        return 0.0
    dd = np.std(downside, ddof=1)
    if dd < 1e-15:
        return 0.0
    return float(mu / dd * ann_factor)


def _compute_max_drawdown(returns: np.ndarray, signal: np.ndarray) -> float:
    """Compute maximum drawdown of the strategy."""
    strat_returns = signal[:-1] * returns[1:]
    cum = np.cumsum(np.log1p(strat_returns))
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def _compute_calmar(returns: np.ndarray, signal: np.ndarray) -> float:
    """Compute Calmar ratio (annual return / max drawdown)."""
    strat_returns = signal[:-1] * returns[1:]
    if len(strat_returns) < 10:
        return 0.0
    ann_ret = np.mean(strat_returns) * 252
    mdd = _compute_max_drawdown(returns, signal)
    if mdd < 1e-15:
        return 0.0
    return float(ann_ret / mdd)


METRICS = {
    "sharpe": _compute_sharpe,
    "sortino": _compute_sortino,
    "calmar": _compute_calmar,
}


def evaluate_strategy(
    strategy_fn: Callable,
    returns: np.ndarray,
    params: dict[str, Any],
    objective: str = "sharpe",
) -> dict[str, Any]:
    """Run one strategy evaluation and return metrics."""
    t0 = time.perf_counter()
    try:
        signal = strategy_fn(returns, **params)
    except Exception as e:
        return {
            "params": params,
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "calmar": float("nan"),
            "max_drawdown": float("nan"),
            "error": str(e),
            "elapsed_s": time.perf_counter() - t0,
        }

    elapsed = time.perf_counter() - t0
    result = {
        "params": params,
        "sharpe": _compute_sharpe(returns, signal),
        "sortino": _compute_sortino(returns, signal),
        "calmar": _compute_calmar(returns, signal),
        "max_drawdown": _compute_max_drawdown(returns, signal),
        "error": None,
        "elapsed_s": elapsed,
    }
    return result


# ---------------------------------------------------------------------------
# Grid search runner
# ---------------------------------------------------------------------------

class GridSearchRunner:
    """Execute a grid search for a single strategy."""

    def __init__(
        self,
        config: SearchConfig,
        strategy_fn: Callable,
        returns: np.ndarray,
        max_workers: int = 4,
    ):
        self.config = config
        self.strategy_fn = strategy_fn
        self.returns = returns
        self.max_workers = max_workers
        self.results: list[dict[str, Any]] = []
        self._start_time: float = 0.0
        self._completed: int = 0
        self._total: int = 0

    def _make_combos(self) -> list[dict[str, Any]]:
        """Generate parameter combinations based on search method."""
        method = self.config.search_method
        space = self.config.param_space
        if method == "grid":
            gen = GridGenerator(space)
            return gen.generate()
        elif method == "random":
            gen = RandomSearchGenerator(space, self.config.n_evaluations)
            return gen.generate()
        elif method == "sobol":
            gen = SobolSearchGenerator(space, self.config.n_evaluations)
            return gen.generate()
        else:
            return []

    def _progress(self) -> dict[str, Any]:
        """Current progress info."""
        elapsed = time.perf_counter() - self._start_time
        rate = self._completed / max(elapsed, 1e-6)
        remaining = self._total - self._completed
        eta = remaining / max(rate, 1e-6)
        return {
            "completed": self._completed,
            "total": self._total,
            "elapsed_s": elapsed,
            "rate_per_s": rate,
            "eta_s": eta,
            "pct": 100.0 * self._completed / max(self._total, 1),
        }

    def run(self, progress_callback: Callable | None = None) -> list[dict[str, Any]]:
        """
        Execute the grid search.

        Parameters
        ----------
        progress_callback : optional callable(progress_dict) invoked after
            each evaluation.

        Returns
        -------
        List of result dicts sorted by the objective (descending).
        """
        if self.config.search_method == "bayesian":
            return self._run_bayesian(progress_callback)

        combos = self._make_combos()
        self._total = len(combos)
        self._completed = 0
        self._start_time = time.perf_counter()
        self.results = []

        def _eval(params: dict) -> dict:
            return evaluate_strategy(
                self.strategy_fn, self.returns, params, self.config.objective,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_eval, p): p for p in combos}
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                self._completed += 1
                if progress_callback:
                    progress_callback(self._progress())

        obj_key = self.config.objective
        self.results.sort(
            key=lambda r: r.get(obj_key, float("-inf"))
            if not math.isnan(r.get(obj_key, float("nan")))
            else float("-inf"),
            reverse=True,
        )
        return self.results

    def _run_bayesian(self, progress_callback: Callable | None = None) -> list[dict[str, Any]]:
        """Bayesian optimisation loop."""
        bo = BayesianSearchGenerator(
            self.config.param_space,
            n_initial=min(self.config.n_evaluations // 3, 20),
            n_iterations=self.config.n_evaluations,
        )
        self._total = self.config.n_evaluations
        self._completed = 0
        self._start_time = time.perf_counter()
        self.results = []

        # Initial random exploration
        for params in bo.initial_points():
            result = evaluate_strategy(
                self.strategy_fn, self.returns, params, self.config.objective,
            )
            self.results.append(result)
            obj_val = result.get(self.config.objective, 0.0)
            if math.isnan(obj_val):
                obj_val = -1e6
            bo.observe(params, obj_val)
            self._completed += 1
            if progress_callback:
                progress_callback(self._progress())

        # Bayesian iterations
        while self._completed < self._total:
            params = bo.suggest_next()
            result = evaluate_strategy(
                self.strategy_fn, self.returns, params, self.config.objective,
            )
            self.results.append(result)
            obj_val = result.get(self.config.objective, 0.0)
            if math.isnan(obj_val):
                obj_val = -1e6
            bo.observe(params, obj_val)
            self._completed += 1
            if progress_callback:
                progress_callback(self._progress())

        obj_key = self.config.objective
        self.results.sort(
            key=lambda r: r.get(obj_key, float("-inf"))
            if not math.isnan(r.get(obj_key, float("nan")))
            else float("-inf"),
            reverse=True,
        )
        return self.results

    def save_results(self, path: str) -> None:
        """Save results to a JSON file."""
        serialisable = []
        for r in self.results:
            entry = dict(r)
            for k, v in entry.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    entry[k] = str(v)
            serialisable.append(entry)
        with open(path, "w") as f:
            json.dump(
                {
                    "strategy": self.config.strategy_name,
                    "search_method": self.config.search_method,
                    "n_evaluations": len(self.results),
                    "results": serialisable,
                },
                f,
                indent=2,
                default=str,
            )

    def top_n(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the top N results by objective."""
        return self.results[:n]


# ---------------------------------------------------------------------------
# Multi-strategy grid search
# ---------------------------------------------------------------------------

class MultiStrategyGridSearch:
    """Run grid search for multiple strategies simultaneously."""

    def __init__(
        self,
        configs: list[SearchConfig],
        strategy_fns: dict[str, Callable],
        returns: np.ndarray,
        max_workers_per_strategy: int = 2,
        max_parallel_strategies: int = 4,
    ):
        self.configs = configs
        self.strategy_fns = strategy_fns
        self.returns = returns
        self.max_workers_per = max_workers_per_strategy
        self.max_parallel = max_parallel_strategies
        self.all_results: dict[str, list[dict[str, Any]]] = {}

    def run(self, progress_callback: Callable | None = None) -> dict[str, list[dict[str, Any]]]:
        """Run all strategy searches.  Returns {strategy_name: results}."""
        def _run_one(cfg: SearchConfig) -> tuple[str, list[dict]]:
            fn = self.strategy_fns[cfg.strategy_name]
            runner = GridSearchRunner(cfg, fn, self.returns, self.max_workers_per)
            results = runner.run()
            return cfg.strategy_name, results

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = {pool.submit(_run_one, c): c.strategy_name for c in self.configs}
            for future in as_completed(futures):
                name, results = future.result()
                self.all_results[name] = results
                if progress_callback:
                    progress_callback({
                        "strategy": name,
                        "n_results": len(results),
                        "strategies_done": len(self.all_results),
                        "strategies_total": len(self.configs),
                    })
        return self.all_results

    def leaderboard(self, metric: str = "sharpe", top_n: int = 5) -> list[dict[str, Any]]:
        """Cross-strategy leaderboard: best result per strategy, ranked."""
        board = []
        for name, results in self.all_results.items():
            if not results:
                continue
            best = results[0]
            board.append({
                "strategy": name,
                "best_" + metric: best.get(metric, float("nan")),
                "params": best.get("params", {}),
            })
        board.sort(
            key=lambda x: x.get("best_" + metric, float("-inf"))
            if not math.isnan(x.get("best_" + metric, float("nan")))
            else float("-inf"),
            reverse=True,
        )
        return board[:top_n]


# ---------------------------------------------------------------------------
# Result database (SQLite)
# ---------------------------------------------------------------------------

class ResultDatabase:
    """SQLite-backed storage of grid search results."""

    def __init__(self, db_path: str = "grid_results.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_tables(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy    TEXT NOT NULL,
                params_json TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                sharpe      REAL,
                sortino     REAL,
                calmar      REAL,
                max_dd      REAL,
                elapsed_s   REAL,
                error       TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                batch_id    TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy ON results (strategy)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_batch ON results (batch_id)
        """)
        conn.commit()

    def insert_result(
        self,
        strategy: str,
        result: dict[str, Any],
        batch_id: str = "",
    ) -> int:
        """Insert one result row.  Returns the row id."""
        conn = self._connect()
        params_json = json.dumps(result.get("params", {}), sort_keys=True, default=str)
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()[:16]
        cur = conn.execute(
            """INSERT INTO results
               (strategy, params_json, params_hash, sharpe, sortino, calmar, max_dd, elapsed_s, error, batch_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                strategy,
                params_json,
                params_hash,
                result.get("sharpe"),
                result.get("sortino"),
                result.get("calmar"),
                result.get("max_drawdown"),
                result.get("elapsed_s"),
                result.get("error"),
                batch_id,
            ),
        )
        conn.commit()
        return cur.lastrowid

    def insert_many(
        self,
        strategy: str,
        results: list[dict[str, Any]],
        batch_id: str = "",
    ) -> int:
        """Bulk insert results.  Returns count inserted."""
        conn = self._connect()
        rows = []
        for r in results:
            params_json = json.dumps(r.get("params", {}), sort_keys=True, default=str)
            params_hash = hashlib.sha256(params_json.encode()).hexdigest()[:16]
            rows.append((
                strategy, params_json, params_hash,
                r.get("sharpe"), r.get("sortino"), r.get("calmar"),
                r.get("max_drawdown"), r.get("elapsed_s"), r.get("error"),
                batch_id,
            ))
        conn.executemany(
            """INSERT INTO results
               (strategy, params_json, params_hash, sharpe, sortino, calmar, max_dd, elapsed_s, error, batch_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        return len(rows)

    def query_top(
        self,
        strategy: str | None = None,
        metric: str = "sharpe",
        top_n: int = 10,
        batch_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query top results.  Optionally filter by strategy/batch."""
        conn = self._connect()
        if metric not in ("sharpe", "sortino", "calmar"):
            raise ValueError(f"Unknown metric '{metric}'")
        where_parts = []
        params_list: list[Any] = []
        if strategy:
            where_parts.append("strategy = ?")
            params_list.append(strategy)
        if batch_id:
            where_parts.append("batch_id = ?")
            params_list.append(batch_id)
        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        sql = f"""
            SELECT * FROM results
            WHERE {where_clause} AND {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT ?
        """
        params_list.append(top_n)
        rows = conn.execute(sql, params_list).fetchall()
        return [dict(r) for r in rows]

    def all_for_strategy(self, strategy: str) -> list[dict[str, Any]]:
        """Return all results for a strategy."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM results WHERE strategy = ? ORDER BY sharpe DESC",
            (strategy,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self, strategy: str | None = None) -> int:
        conn = self._connect()
        if strategy:
            row = conn.execute("SELECT COUNT(*) FROM results WHERE strategy = ?", (strategy,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM results").fetchone()
        return row[0]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Parameter landscape analysis
# ---------------------------------------------------------------------------

class ParameterLandscape:
    """Analyse the Sharpe-vs-parameter landscape from grid search results."""

    def __init__(self, results: list[dict[str, Any]], space: ParameterSpace):
        self.results = results
        self.space = space
        self._extract()

    def _extract(self) -> None:
        """Extract parameter arrays and objective values."""
        self.param_matrix: list[dict[str, Any]] = []
        self.objectives: list[float] = []
        for r in self.results:
            if r.get("error"):
                continue
            sharpe = r.get("sharpe", float("nan"))
            if math.isnan(sharpe):
                continue
            self.param_matrix.append(r["params"])
            self.objectives.append(sharpe)
        self.objectives_arr = np.array(self.objectives)

    def slice_1d(
        self,
        vary_param: str,
        fixed_params: dict[str, Any] | None = None,
        tolerance: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        1-D slice: fix all params except *vary_param*, return (values, sharpes).

        If *fixed_params* is None, uses the best-found parameter set as the
        baseline.  Points within *tolerance* (relative) of the fixed values
        are included.
        """
        if not self.param_matrix:
            return np.array([]), np.array([])

        if fixed_params is None:
            best_idx = int(np.argmax(self.objectives_arr))
            fixed_params = dict(self.param_matrix[best_idx])

        values = []
        sharpes = []
        for params, sharpe in zip(self.param_matrix, self.objectives):
            match = True
            for name in self.space.names:
                if name == vary_param:
                    continue
                pv = params.get(name, 0.0)
                fv = fixed_params.get(name, 0.0)
                if isinstance(pv, (int, float)) and isinstance(fv, (int, float)):
                    if abs(fv) > 1e-12:
                        if abs(pv - fv) / abs(fv) > tolerance:
                            match = False
                            break
                    elif abs(pv - fv) > tolerance:
                        match = False
                        break
                elif pv != fv:
                    match = False
                    break
            if match:
                values.append(params.get(vary_param, 0.0))
                sharpes.append(sharpe)

        order = np.argsort(values)
        return np.array(values)[order], np.array(sharpes)[order]

    def heatmap_2d(
        self,
        param_x: str,
        param_y: str,
        fixed_params: dict[str, Any] | None = None,
        tolerance: float = 0.15,
        grid_size: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2-D heatmap: fix all except *param_x* and *param_y*.

        Returns (x_edges, y_edges, sharpe_grid) suitable for pcolormesh.
        """
        if not self.param_matrix:
            return np.array([]), np.array([]), np.array([[]])

        if fixed_params is None:
            best_idx = int(np.argmax(self.objectives_arr))
            fixed_params = dict(self.param_matrix[best_idx])

        xs, ys, zs = [], [], []
        for params, sharpe in zip(self.param_matrix, self.objectives):
            match = True
            for name in self.space.names:
                if name in (param_x, param_y):
                    continue
                pv = params.get(name, 0.0)
                fv = fixed_params.get(name, 0.0)
                if isinstance(pv, (int, float)) and isinstance(fv, (int, float)):
                    if abs(fv) > 1e-12:
                        if abs(pv - fv) / abs(fv) > tolerance:
                            match = False
                            break
                    elif abs(pv - fv) > tolerance:
                        match = False
                        break
                elif pv != fv:
                    match = False
                    break
            if match:
                xs.append(float(params.get(param_x, 0.0)))
                ys.append(float(params.get(param_y, 0.0)))
                zs.append(sharpe)

        if not xs:
            return np.array([]), np.array([]), np.array([[]])

        xs_arr, ys_arr, zs_arr = np.array(xs), np.array(ys), np.array(zs)
        x_edges = np.linspace(xs_arr.min(), xs_arr.max(), grid_size + 1)
        y_edges = np.linspace(ys_arr.min(), ys_arr.max(), grid_size + 1)
        grid = np.full((grid_size, grid_size), np.nan)
        counts = np.zeros((grid_size, grid_size), dtype=int)

        for x, y, z in zip(xs_arr, ys_arr, zs_arr):
            xi = min(int((x - x_edges[0]) / (x_edges[1] - x_edges[0] + 1e-15)), grid_size - 1)
            yi = min(int((y - y_edges[0]) / (y_edges[1] - y_edges[0] + 1e-15)), grid_size - 1)
            xi = max(0, xi)
            yi = max(0, yi)
            if counts[yi, xi] == 0:
                grid[yi, xi] = z
            else:
                grid[yi, xi] = (grid[yi, xi] * counts[yi, xi] + z) / (counts[yi, xi] + 1)
            counts[yi, xi] += 1

        return x_edges, y_edges, grid

    def sensitivity_ranking(self) -> list[tuple[str, float]]:
        """
        Rank parameters by impact on Sharpe.

        For each parameter, compute the variance of Sharpe across its range
        (marginalised over other parameters).  Higher variance = more sensitive.
        """
        if not self.param_matrix:
            return []

        ranking = []
        for name in self.space.names:
            vals = np.array([float(p.get(name, 0.0)) for p in self.param_matrix])
            if np.std(vals) < 1e-15:
                ranking.append((name, 0.0))
                continue
            # Bin and compute mean Sharpe per bin
            n_bins = min(10, len(set(vals)))
            if n_bins < 2:
                ranking.append((name, 0.0))
                continue
            edges = np.linspace(vals.min(), vals.max(), n_bins + 1)
            bin_means = []
            for b in range(n_bins):
                mask = (vals >= edges[b]) & (vals < edges[b + 1] + 1e-15)
                if mask.any():
                    bin_means.append(np.mean(self.objectives_arr[mask]))
            if len(bin_means) < 2:
                ranking.append((name, 0.0))
            else:
                ranking.append((name, float(np.std(bin_means))))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking


# ---------------------------------------------------------------------------
# Optimal parameter finder
# ---------------------------------------------------------------------------

class OptimalParameterFinder:
    """
    Find globally optimal parameters with robustness constraints.

    - Penalises parameter sets near the search boundary (overfit risk).
    - Requires stability: the Sharpe must be stable in a neighbourhood.
    """

    def __init__(
        self,
        results: list[dict[str, Any]],
        space: ParameterSpace,
        boundary_penalty: float = 0.3,
        stability_radius: float = 0.1,
        stability_min_neighbours: int = 3,
    ):
        self.results = results
        self.space = space
        self.boundary_penalty = boundary_penalty
        self.stability_radius = stability_radius
        self.stability_min_neighbours = stability_min_neighbours
        self._prepare()

    def _prepare(self) -> None:
        """Build arrays for fast lookup."""
        self.param_dicts: list[dict] = []
        self.sharpes: list[float] = []
        self.unit_vecs: list[np.ndarray] = []
        for r in self.results:
            if r.get("error") or math.isnan(r.get("sharpe", float("nan"))):
                continue
            self.param_dicts.append(r["params"])
            self.sharpes.append(r["sharpe"])
            self.unit_vecs.append(self.space.dict_to_unit(r["params"]))
        self.sharpes_arr = np.array(self.sharpes)
        self.unit_mat = np.array(self.unit_vecs) if self.unit_vecs else np.empty((0, self.space.dim))

    def _boundary_penalty_score(self, unit_vec: np.ndarray) -> float:
        """Penalty for being near [0,1] boundary."""
        dist_to_boundary = np.minimum(unit_vec, 1.0 - unit_vec)
        min_dist = np.min(dist_to_boundary)
        if min_dist < 0.05:
            return self.boundary_penalty
        elif min_dist < 0.15:
            return self.boundary_penalty * (0.15 - min_dist) / 0.10
        return 0.0

    def _stability_score(self, idx: int) -> float:
        """Measure how stable the Sharpe is in a neighbourhood."""
        if len(self.unit_mat) < 2:
            return 0.0
        target = self.unit_mat[idx]
        dists = np.linalg.norm(self.unit_mat - target, axis=1)
        mask = (dists > 1e-12) & (dists < self.stability_radius)
        neighbours = self.sharpes_arr[mask]
        if len(neighbours) < self.stability_min_neighbours:
            return -0.2  # penalty for isolated points
        return -np.std(neighbours)  # less variance = better (closer to 0)

    def find_optimal(self, top_n: int = 5) -> list[dict[str, Any]]:
        """
        Return the top-N parameter sets ranked by penalised Sharpe.

        Each result dict contains: params, sharpe, penalised_sharpe,
        boundary_penalty, stability_score.
        """
        if not self.sharpes:
            return []

        scored = []
        for i in range(len(self.sharpes)):
            bp = self._boundary_penalty_score(self.unit_mat[i])
            ss = self._stability_score(i)
            penalised = self.sharpes[i] - bp + ss * 0.5
            scored.append({
                "params": self.param_dicts[i],
                "sharpe": self.sharpes[i],
                "penalised_sharpe": penalised,
                "boundary_penalty": bp,
                "stability_score": ss,
            })

        scored.sort(key=lambda x: x["penalised_sharpe"], reverse=True)
        return scored[:top_n]

    def robustness_score(self, params: dict[str, Any]) -> float:
        """
        Compute a robustness score for a given parameter set.

        Higher is better.  Combines Sharpe, stability, and boundary distance.
        """
        unit = self.space.dict_to_unit(params)
        bp = self._boundary_penalty_score(unit)

        dists = np.linalg.norm(self.unit_mat - unit, axis=1)
        mask = dists < self.stability_radius
        if mask.sum() < 1:
            return -1.0
        neighbourhood_sharpes = self.sharpes_arr[mask]
        mean_sharpe = float(np.mean(neighbourhood_sharpes))
        std_sharpe = float(np.std(neighbourhood_sharpes))

        return mean_sharpe - bp - std_sharpe
