"""
CounterfactualOracle
====================
Core engine for "what-if" analysis on genome parameter sets.

Given a baseline simulation run (identified by run_id stored in idea_engine.db),
the oracle:
  1. Loads the baseline params + metrics from the DB.
  2. Applies a ``param_delta`` dict to produce a variant param set.
  3. Re-runs the backtest via subprocess calling ``python -m tools.crypto_backtest_mc``.
  4. Compares Sharpe, MaxDD, Calmar, total_return, win_rate, num_trades.
  5. Computes an ``improvement_score`` (weighted composite).
  6. Persists results in ``counterfactual_results`` and exposes ranking helpers.

Parallel sweeps use ``multiprocessing.Pool`` (default 4 workers).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from .parameter_space import ParameterSpace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV = "IDEA_ENGINE_DB"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "db" / "idea_engine.db"

BACKTEST_MODULE = "tools.crypto_backtest_mc"

# Weights for the composite improvement score
W_SHARPE = 0.40
W_CALMAR = 0.30
W_MAXDD = 0.30   # negative contribution — higher DD is worse


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SimMetrics:
    """Metrics extracted from a single backtest run."""

    sharpe: float = 0.0
    max_dd: float = 0.0       # expressed as a positive fraction, e.g. 0.15 = 15 %
    calmar: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimMetrics":
        return cls(
            sharpe=float(d.get("sharpe", 0.0)),
            max_dd=float(d.get("max_dd", 0.0)),
            calmar=float(d.get("calmar", 0.0)),
            total_return=float(d.get("total_return", 0.0)),
            win_rate=float(d.get("win_rate", 0.0)),
            num_trades=int(d.get("num_trades", 0)),
        )


@dataclass
class CounterfactualResult:
    """A single counterfactual evaluation."""

    baseline_run_id: str
    param_delta: dict[str, Any]
    params_full: dict[str, Any]
    metrics: SimMetrics
    improvement: float
    cf_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error: str | None = None


# ---------------------------------------------------------------------------
# Worker function (module-level so it is picklable)
# ---------------------------------------------------------------------------

def _run_variant_worker(args: tuple) -> CounterfactualResult:
    """
    Executed in a subprocess pool worker.  Runs the backtest with modified
    params and returns a CounterfactualResult.

    ``args`` tuple layout:
        (baseline_run_id, param_delta, params_full, baseline_metrics_dict, repo_root)
    """
    baseline_run_id, param_delta, params_full, baseline_dict, repo_root = args

    baseline = SimMetrics.from_dict(baseline_dict)
    metrics = SimMetrics()
    error: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=repo_root
        ) as pf:
            json.dump(params_full, pf)
            params_path = pf.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=repo_root
        ) as rf:
            results_path = rf.name

        cmd = [
            sys.executable,
            "-m",
            BACKTEST_MODULE,
            "--params-file", params_path,
            "--output-file", results_path,
            "--quiet",
        ]

        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if proc.returncode != 0:
            error = f"backtest exited {proc.returncode}: {proc.stderr[:500]}"
            logger.warning("Backtest failed for variant: %s", error)
        else:
            try:
                with open(results_path) as rf:
                    result_data = json.load(rf)
                metrics = SimMetrics.from_dict(result_data)
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                error = f"result parse error: {exc}"

    except subprocess.TimeoutExpired:
        error = "backtest timed out after 300 s"
    except Exception as exc:  # noqa: BLE001
        error = f"worker exception: {exc}"
    finally:
        for p in [params_path, results_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

    improvement = _compute_improvement(baseline, metrics) if error is None else -999.0

    return CounterfactualResult(
        baseline_run_id=baseline_run_id,
        param_delta=param_delta,
        params_full=params_full,
        metrics=metrics,
        improvement=improvement,
        error=error,
    )


def _compute_improvement(baseline: SimMetrics, variant: SimMetrics) -> float:
    """
    Weighted composite improvement score.

    score = 0.4 * Δsharpe + 0.3 * Δcalmar − 0.3 * Δmax_dd

    All deltas are (variant − baseline).  A positive score means the variant
    is better than the baseline overall.
    """
    sharpe_delta = variant.sharpe - baseline.sharpe
    calmar_delta = variant.calmar - baseline.calmar
    # max_dd is a cost — higher DD lowers score
    maxdd_delta = variant.max_dd - baseline.max_dd

    return W_SHARPE * sharpe_delta + W_CALMAR * calmar_delta - W_MAXDD * maxdd_delta


# ---------------------------------------------------------------------------
# Main oracle class
# ---------------------------------------------------------------------------

class CounterfactualOracle:
    """
    Counterfactual engine for genome parameter analysis.

    Parameters
    ----------
    db_path : str | Path | None
        Path to ``idea_engine.db``.  Defaults to ``IDEA_ENGINE_DB`` env var or
        the standard lab location.
    workers : int
        Number of parallel backtest workers (default 4).
    repo_root : str | Path | None
        Root of the srfm-lab repo (used as cwd for subprocess calls).
        Defaults to three levels above this file.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        workers: int = 4,
        repo_root: str | Path | None = None,
    ) -> None:
        self.db_path = Path(
            db_path or os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        self.workers = max(1, workers)
        self.repo_root = str(
            repo_root or Path(__file__).resolve().parents[3]
        )
        self._param_space = ParameterSpace()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create DB tables if they do not exist yet."""
        sql_file = Path(__file__).parent / "schema_extension.sql"
        if sql_file.exists():
            ddl = sql_file.read_text()
        else:
            ddl = _INLINE_DDL
        with self._db() as con:
            con.executescript(ddl)

    def _db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # ------------------------------------------------------------------
    # Baseline retrieval
    # ------------------------------------------------------------------

    def load_baseline(self, run_id: str) -> tuple[dict[str, Any], SimMetrics]:
        """
        Fetch baseline params and metrics from ``sim_runs`` table.

        Returns
        -------
        (params_full, SimMetrics)
        """
        with self._db() as con:
            row = con.execute(
                "SELECT params, sharpe, max_dd, calmar, total_return, win_rate, num_trades "
                "FROM sim_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"run_id {run_id!r} not found in sim_runs")

        params = json.loads(row["params"])
        metrics = SimMetrics(
            sharpe=row["sharpe"] or 0.0,
            max_dd=row["max_dd"] or 0.0,
            calmar=row["calmar"] or 0.0,
            total_return=row["total_return"] or 0.0,
            win_rate=row["win_rate"] or 0.0,
            num_trades=row["num_trades"] or 0,
        )
        return params, metrics

    # ------------------------------------------------------------------
    # Single variant evaluation
    # ------------------------------------------------------------------

    def evaluate_variant(
        self,
        run_id: str,
        param_delta: dict[str, Any],
    ) -> CounterfactualResult:
        """
        Apply ``param_delta`` to baseline params and run the backtest.

        Returns a ``CounterfactualResult`` and persists it in the DB.
        """
        baseline_params, baseline_metrics = self.load_baseline(run_id)
        params_full = {**baseline_params, **param_delta}
        # Clip to valid bounds
        params_full = self._param_space.clip(params_full)

        args = (run_id, param_delta, params_full, baseline_metrics.to_dict(), self.repo_root)
        result = _run_variant_worker(args)
        self._persist_result(result)
        return result

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_counterfactuals(
        self,
        run_id: str,
        n_variations: int = 50,
        method: str = "neighborhood",
        radius: float = 0.15,
    ) -> list[CounterfactualResult]:
        """
        Generate and evaluate ``n_variations`` parameter variants for a run.

        Parameters
        ----------
        run_id : str
            Baseline run to perturb.
        n_variations : int
            How many variants to evaluate (default 50).
        method : {'neighborhood', 'lhs', 'sobol'}
            Sampling method:
            - ``neighborhood`` — Gaussian perturbation around baseline params.
            - ``lhs`` — Latin hypercube over the full parameter space.
            - ``sobol`` — quasi-random low-discrepancy sequence.
        radius : float
            Perturbation radius for ``neighborhood`` method (fraction of range).
        """
        baseline_params, baseline_metrics = self.load_baseline(run_id)

        if method == "neighborhood":
            samples = self._param_space.neighborhood_sample(
                baseline_params, radius=radius, n=n_variations
            )
        elif method == "lhs":
            samples = self._param_space.latin_hypercube_sample(n_variations)
        elif method == "sobol":
            samples = self._param_space.sobol_sample(n_variations)
        else:
            raise ValueError(f"Unknown sampling method: {method!r}")

        baseline_dict = baseline_metrics.to_dict()
        worker_args = []
        for sample_params in samples:
            delta = {
                k: sample_params[k]
                for k in sample_params
                if k in baseline_params and sample_params[k] != baseline_params[k]
            }
            worker_args.append(
                (run_id, delta, sample_params, baseline_dict, self.repo_root)
            )

        logger.info(
            "Launching %d counterfactual workers (pool=%d) for run %s",
            len(worker_args),
            self.workers,
            run_id,
        )
        t0 = time.monotonic()
        with Pool(processes=self.workers) as pool:
            results: list[CounterfactualResult] = pool.map(
                _run_variant_worker, worker_args
            )

        elapsed = time.monotonic() - t0
        successful = sum(1 for r in results if r.error is None)
        logger.info(
            "Counterfactual sweep done: %d/%d succeeded in %.1f s",
            successful,
            len(results),
            elapsed,
        )

        for result in results:
            self._persist_result(result)

        return results

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_counterfactuals(
        self,
        results: list[CounterfactualResult],
        top_n: int | None = None,
    ) -> list[CounterfactualResult]:
        """
        Sort counterfactual results by ``improvement`` score descending.

        Parameters
        ----------
        results : list[CounterfactualResult]
        top_n : int | None
            If given, return only the top-N results.

        Returns
        -------
        Sorted list, best first.
        """
        ranked = sorted(
            [r for r in results if r.error is None],
            key=lambda r: r.improvement,
            reverse=True,
        )
        failed = [r for r in results if r.error is not None]
        if failed:
            logger.debug("%d variants failed and were excluded from ranking", len(failed))

        if top_n is not None:
            return ranked[:top_n]
        return ranked

    # ------------------------------------------------------------------
    # Improvement score (static helper, also used by workers)
    # ------------------------------------------------------------------

    @staticmethod
    def improvement_score(baseline: SimMetrics, variant: SimMetrics) -> float:
        """
        Weighted composite delta.

        .. math::
            score = 0.4 \\cdot \\Delta Sharpe
                  + 0.3 \\cdot \\Delta Calmar
                  - 0.3 \\cdot \\Delta MaxDD

        Positive score → variant outperforms baseline.
        """
        return _compute_improvement(baseline, variant)

    # ------------------------------------------------------------------
    # Gradient / ascent helpers
    # ------------------------------------------------------------------

    def gradient_at(
        self,
        run_id: str,
        params: dict[str, Any],
        epsilon: float = 0.05,
    ) -> dict[str, float]:
        """
        Finite-difference gradient of improvement_score w.r.t. each parameter.

        For each parameter p:
            grad[p] = (score(p + ε·range) − score(p − ε·range)) / (2·ε·range)

        Returns a dict mapping param_name → gradient value.
        """
        _, baseline_metrics = self.load_baseline(run_id)
        bounds = self._param_space.bounds
        grad: dict[str, float] = {}

        for param_name, (lo, hi) in bounds.items():
            step = epsilon * (hi - lo)
            p_plus = copy.deepcopy(params)
            p_minus = copy.deepcopy(params)
            p_plus[param_name] = min(hi, params.get(param_name, (lo + hi) / 2) + step)
            p_minus[param_name] = max(lo, params.get(param_name, (lo + hi) / 2) - step)

            args_plus = (run_id, {param_name: p_plus[param_name]}, p_plus,
                         baseline_metrics.to_dict(), self.repo_root)
            args_minus = (run_id, {param_name: p_minus[param_name]}, p_minus,
                          baseline_metrics.to_dict(), self.repo_root)

            r_plus = _run_variant_worker(args_plus)
            r_minus = _run_variant_worker(args_minus)

            if r_plus.error is None and r_minus.error is None:
                grad[param_name] = (r_plus.improvement - r_minus.improvement) / (2 * epsilon)
            else:
                grad[param_name] = 0.0

        return grad

    def steepest_ascent(
        self,
        run_id: str,
        steps: int = 10,
        step_size: float = 0.05,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Follow the gradient uphill for ``steps`` iterations starting from
        the baseline params of ``run_id``.

        Returns a list of (params, improvement_score) tuples along the path.
        """
        baseline_params, _ = self.load_baseline(run_id)
        current = copy.deepcopy(baseline_params)
        path: list[tuple[dict[str, Any], float]] = []

        bounds = self._param_space.bounds
        for step in range(steps):
            grad = self.gradient_at(run_id, current, epsilon=step_size)
            # Normalize gradient
            norm = (sum(v ** 2 for v in grad.values()) ** 0.5) or 1.0
            next_params = copy.deepcopy(current)
            for param_name, g in grad.items():
                if param_name not in bounds:
                    continue
                lo, hi = bounds[param_name]
                rng = hi - lo
                next_params[param_name] = float(
                    np.clip(
                        current.get(param_name, (lo + hi) / 2) + step_size * rng * g / norm,
                        lo, hi,
                    )
                )

            result = self.evaluate_variant(run_id, {
                k: next_params[k] for k in next_params if next_params.get(k) != current.get(k)
            })
            path.append((copy.deepcopy(next_params), result.improvement))
            logger.debug("Ascent step %d/%d: improvement=%.4f", step + 1, steps, result.improvement)

            if result.error is None:
                current = next_params

        return path

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def load_results(
        self,
        run_id: str,
        min_improvement: float | None = None,
        limit: int = 200,
    ) -> list[CounterfactualResult]:
        """
        Load persisted counterfactual results for a baseline run.

        Parameters
        ----------
        run_id : str
        min_improvement : float | None
            Optional lower bound filter on improvement score.
        limit : int
            Max rows to return.
        """
        query = (
            "SELECT baseline_run_id, param_delta, params_full, "
            "sharpe, max_dd, calmar, total_return, win_rate, num_trades, improvement "
            "FROM counterfactual_results WHERE baseline_run_id = ?"
        )
        params_sql: list[Any] = [run_id]

        if min_improvement is not None:
            query += " AND improvement >= ?"
            params_sql.append(min_improvement)

        query += " ORDER BY improvement DESC LIMIT ?"
        params_sql.append(limit)

        results = []
        with self._db() as con:
            for row in con.execute(query, params_sql):
                metrics = SimMetrics(
                    sharpe=row["sharpe"] or 0.0,
                    max_dd=row["max_dd"] or 0.0,
                    calmar=row["calmar"] or 0.0,
                    total_return=row["total_return"] or 0.0,
                    win_rate=row["win_rate"] or 0.0,
                    num_trades=row["num_trades"] or 0,
                )
                results.append(
                    CounterfactualResult(
                        baseline_run_id=row["baseline_run_id"],
                        param_delta=json.loads(row["param_delta"]),
                        params_full=json.loads(row["params_full"]),
                        metrics=metrics,
                        improvement=row["improvement"] or 0.0,
                    )
                )
        return results

    def best_variant(self, run_id: str) -> CounterfactualResult | None:
        """Return the single best-scoring variant for a run_id."""
        results = self.load_results(run_id, limit=1)
        return results[0] if results else None

    def summary(self, run_id: str) -> dict[str, Any]:
        """
        Return a summary dict for a run's counterfactual landscape:
        count, mean/max/min improvement, best params delta.
        """
        with self._db() as con:
            row = con.execute(
                "SELECT COUNT(*) as n, "
                "AVG(improvement) as mean_imp, "
                "MAX(improvement) as max_imp, "
                "MIN(improvement) as min_imp "
                "FROM counterfactual_results WHERE baseline_run_id = ?",
                (run_id,),
            ).fetchone()

        best = self.best_variant(run_id)
        return {
            "run_id": run_id,
            "n_variants": row["n"],
            "mean_improvement": row["mean_imp"],
            "max_improvement": row["max_imp"],
            "min_improvement": row["min_imp"],
            "best_delta": best.param_delta if best else None,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_result(self, result: CounterfactualResult) -> None:
        """Insert a CounterfactualResult into ``counterfactual_results``."""
        with self._db() as con:
            con.execute(
                """
                INSERT INTO counterfactual_results
                    (baseline_run_id, param_delta, params_full,
                     sharpe, max_dd, calmar, total_return, win_rate, num_trades, improvement)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.baseline_run_id,
                    json.dumps(result.param_delta),
                    json.dumps(result.params_full),
                    result.metrics.sharpe,
                    result.metrics.max_dd,
                    result.metrics.calmar,
                    result.metrics.total_return,
                    result.metrics.win_rate,
                    result.metrics.num_trades,
                    result.improvement,
                ),
            )
            con.commit()

    def promote_best_to_hypothesis(
        self,
        run_id: str,
        min_improvement: float = 0.1,
        hypothesis_table: str = "hypotheses",
    ) -> list[str]:
        """
        For variants whose improvement exceeds ``min_improvement``, write
        a hypothesis row to the hypotheses table so the idea pipeline can
        pick it up for further testing.

        Returns list of hypothesis IDs created.
        """
        top = self.load_results(run_id, min_improvement=min_improvement, limit=10)
        hypothesis_ids: list[str] = []

        for result in top:
            h_id = str(uuid.uuid4())
            description = (
                f"Counterfactual variant of {run_id}: "
                f"improvement={result.improvement:.4f}, "
                f"delta={json.dumps(result.param_delta)}"
            )
            try:
                with self._db() as con:
                    con.execute(
                        f"""
                        INSERT INTO {hypothesis_table}
                            (id, source, description, params, status, created_at)
                        VALUES (?, 'counterfactual', ?, ?, 'pending',
                                strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                        """,
                        (h_id, description, json.dumps(result.params_full)),
                    )
                    con.commit()
                hypothesis_ids.append(h_id)
            except sqlite3.OperationalError as exc:
                logger.warning("Could not insert hypothesis (table exists?): %s", exc)

        return hypothesis_ids


# ---------------------------------------------------------------------------
# Fallback inline DDL (used if schema_extension.sql is missing at runtime)
# ---------------------------------------------------------------------------

_INLINE_DDL = """
CREATE TABLE IF NOT EXISTS counterfactual_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_run_id TEXT    NOT NULL,
    param_delta     TEXT    NOT NULL,
    params_full     TEXT    NOT NULL,
    sharpe          REAL,
    max_dd          REAL,
    calmar          REAL,
    total_return    REAL,
    win_rate        REAL,
    num_trades      INTEGER,
    improvement     REAL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE IF NOT EXISTS sensitivity_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    param_name      TEXT    NOT NULL,
    sobol_index     REAL,
    tornado_range   REAL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""
