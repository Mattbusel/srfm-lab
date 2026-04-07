# ml/training/experiment_tracker.py -- experiment tracking, run management, hyperparameter tuning
from __future__ import annotations

import concurrent.futures
import itertools
import json
import logging
import os
import pickle
import random
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    params          TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    name            TEXT NOT NULL,
    value           REAL NOT NULL,
    step            INTEGER,
    logged_at       TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    name            TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    logged_at       TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_name);
CREATE INDEX IF NOT EXISTS idx_metrics_run     ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name    ON metrics(name);
CREATE INDEX IF NOT EXISTS idx_artifacts_run   ON artifacts(run_id);
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Full record for a completed or in-progress run."""

    run_id: str
    experiment_name: str
    status: str  # "running", "completed", "failed"
    started_at: datetime
    ended_at: Optional[datetime]
    params: Dict[str, Any]
    metrics: Dict[str, List[float]]  # name -> list of values across steps
    artifacts: List[str]  # list of artifact names


@dataclass
class TuneResult:
    """Result container for hyperparameter search."""

    best_params: Dict[str, Any]
    best_score: float
    best_run_id: str
    all_results: List[Dict[str, Any]]  # each entry: {params, score, run_id}
    metric: str


# ---------------------------------------------------------------------------
# RunContext -- context manager returned by start_run
# ---------------------------------------------------------------------------


class RunContext:
    """
    Context manager for a single experiment run.

    Usage::

        with tracker.start_run("signal_training", params={}) as run:
            run.log_metric("ic", 0.08)
            run.log_artifact("model", trained_model)
    """

    def __init__(self, tracker: "ExperimentTracker", run_id: str) -> None:
        self._tracker = tracker
        self.run_id = run_id

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        self._tracker.log_metric(self.run_id, name, value, step)

    def log_artifact(self, name: str, data: Any) -> None:
        self._tracker.log_artifact(self.run_id, name, data)

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status = "completed" if exc_type is None else "failed"
        self._tracker.end_run(self.run_id, status)

    def __repr__(self) -> str:
        return f"RunContext(run_id={self.run_id!r})"


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """
    Lightweight experiment tracker backed by SQLite and pickle files.

    Parameters
    ----------
    tracking_dir:
        Root directory for the tracking database and artifact storage.
    """

    def __init__(self, tracking_dir: str = "mlruns") -> None:
        self.tracking_dir = Path(tracking_dir)
        self.artifacts_dir = self.tracking_dir / "artifacts"
        self.db_path = self.tracking_dir / "tracking.db"

        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    def _ensure_experiment(self, name: str) -> str:
        """Get or create experiment by name, return experiment_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT experiment_id FROM experiments WHERE name = ?", (name,)
            ).fetchone()
            if row:
                return row["experiment_id"]

            experiment_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO experiments (experiment_id, name, created_at) VALUES (?, ?, ?)",
                (experiment_id, name, datetime.utcnow().isoformat()),
            )
            return experiment_id

    def _artifact_path(self, run_id: str, name: str) -> Path:
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(exist_ok=True)
        return run_dir / f"{name}.pkl"

    def _build_run_record(self, run_id: str) -> RunRecord:
        with self._connect() as conn:
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                raise KeyError(f"Run not found: {run_id}")

            metric_rows = conn.execute(
                "SELECT name, value FROM metrics WHERE run_id = ? ORDER BY step ASC, logged_at ASC",
                (run_id,),
            ).fetchall()

            artifact_rows = conn.execute(
                "SELECT name FROM artifacts WHERE run_id = ?", (run_id,)
            ).fetchall()

        metrics: Dict[str, List[float]] = {}
        for row in metric_rows:
            metrics.setdefault(row["name"], []).append(row["value"])

        artifacts = [r["name"] for r in artifact_rows]

        return RunRecord(
            run_id=run_id,
            experiment_name=run_row["experiment_name"],
            status=run_row["status"],
            started_at=datetime.fromisoformat(run_row["started_at"]),
            ended_at=datetime.fromisoformat(run_row["ended_at"]) if run_row["ended_at"] else None,
            params=json.loads(run_row["params"]),
            metrics=metrics,
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, experiment_name: str, params: Dict[str, Any]) -> RunContext:
        """
        Start a new run for the given experiment.

        Returns a RunContext that acts as a context manager.
        """
        experiment_id = self._ensure_experiment(experiment_name)
        run_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, experiment_id, experiment_name, status, started_at, params)
                VALUES (?, ?, ?, 'running', ?, ?)
                """,
                (run_id, experiment_id, experiment_name, datetime.utcnow().isoformat(), json.dumps(params)),
            )

        logger.debug("Started run %s for experiment %s", run_id, experiment_name)
        return RunContext(self, run_id)

    def log_metric(
        self,
        run_id: str,
        name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a scalar metric value for a run."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (run_id, name, value, step, logged_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, name, float(value), step, datetime.utcnow().isoformat()),
            )

    def log_artifact(self, run_id: str, name: str, data: Any) -> None:
        """Pickle an arbitrary artifact and record it in the DB."""
        path = self._artifact_path(run_id, name)
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (run_id, name, file_path, logged_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, name, str(path), datetime.utcnow().isoformat()),
            )

    def load_artifact(self, run_id: str, name: str) -> Any:
        """Load a previously saved artifact by run_id and name."""
        path = self._artifact_path(run_id, name)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: run_id={run_id} name={name}")
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def end_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as ended with the given status."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = ? WHERE run_id = ?",
                (status, datetime.utcnow().isoformat(), run_id),
            )
        logger.debug("Ended run %s status=%s", run_id, status)

    def get_run(self, run_id: str) -> RunRecord:
        """Return the full RunRecord for the given run_id."""
        return self._build_run_record(run_id)

    def compare_runs(self, run_ids: List[str], metric: str) -> pd.DataFrame:
        """
        Compare multiple runs on a given metric.

        Returns a DataFrame with columns: run_id, experiment_name, status,
        params, and the requested metric (last logged value).
        """
        records = []
        for rid in run_ids:
            try:
                rec = self._build_run_record(rid)
            except KeyError:
                logger.warning("Run not found: %s", rid)
                continue

            values = rec.metrics.get(metric, [])
            last_value = values[-1] if values else float("nan")

            records.append({
                "run_id": rid,
                "experiment_name": rec.experiment_name,
                "status": rec.status,
                "params": json.dumps(rec.params),
                metric: last_value,
            })

        if not records:
            return pd.DataFrame(columns=["run_id", "experiment_name", "status", "params", metric])

        df = pd.DataFrame(records)
        df.sort_values(metric, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def best_run(
        self,
        experiment_name: str,
        metric: str,
        n: int = 1,
        higher_is_better: bool = True,
    ) -> List[RunRecord]:
        """
        Return the top n runs for an experiment, ranked by final metric value.
        """
        with self._connect() as conn:
            run_rows = conn.execute(
                """
                SELECT run_id FROM runs
                WHERE experiment_name = ? AND status = 'completed'
                """,
                (experiment_name,),
            ).fetchall()

        scored: List[Tuple[float, RunRecord]] = []
        for row in run_rows:
            try:
                rec = self._build_run_record(row["run_id"])
            except KeyError:
                continue
            values = rec.metrics.get(metric, [])
            if not values:
                continue
            scored.append((values[-1], rec))

        scored.sort(key=lambda t: t[0], reverse=higher_is_better)
        return [rec for _, rec in scored[:n]]

    def list_experiments(self) -> List[str]:
        """Return all experiment names."""
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM experiments ORDER BY created_at DESC").fetchall()
        return [r["name"] for r in rows]

    def list_runs(
        self,
        experiment_name: str,
        status: Optional[str] = None,
    ) -> List[RunRecord]:
        """List runs for an experiment, optionally filtered by status."""
        clauses = ["experiment_name = ?"]
        params: List[Any] = [experiment_name]
        if status:
            clauses.append("status = ?")
            params.append(status)

        sql = "SELECT run_id FROM runs WHERE " + " AND ".join(clauses) + " ORDER BY started_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = []
        for row in rows:
            try:
                records.append(self._build_run_record(row["run_id"]))
            except KeyError:
                pass
        return records

    def delete_run(self, run_id: str) -> None:
        """Remove a run and all associated metrics and artifacts from storage."""
        with self._connect() as conn:
            conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

        run_artifacts_dir = self.artifacts_dir / run_id
        if run_artifacts_dir.exists():
            import shutil
            shutil.rmtree(run_artifacts_dir)

    def metric_history(self, run_id: str, metric: str) -> List[Tuple[Optional[int], float]]:
        """
        Return the full (step, value) history for a metric in a run.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT step, value FROM metrics
                WHERE run_id = ? AND name = ?
                ORDER BY step ASC, logged_at ASC
                """,
                (run_id, metric),
            ).fetchall()
        return [(row["step"], row["value"]) for row in rows]


# ---------------------------------------------------------------------------
# HyperparameterTuner
# ---------------------------------------------------------------------------


def _evaluate_params(
    fn: Callable,
    params: Dict[str, Any],
    tracker: ExperimentTracker,
    experiment_name: str,
    metric: str,
) -> Dict[str, Any]:
    """Worker function for parallel hyperparameter evaluation."""
    ctx = tracker.start_run(experiment_name, params)
    score = float("nan")
    try:
        result = fn(params)
        if isinstance(result, dict):
            score = result.get(metric, float("nan"))
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    ctx.log_metric(k, float(v))
        else:
            score = float(result)
            ctx.log_metric(metric, score)
        tracker.end_run(ctx.run_id, "completed")
    except Exception as exc:  # noqa: BLE001
        logger.error("Trial failed with params %s: %s", params, exc)
        tracker.end_run(ctx.run_id, "failed")

    return {"params": params, "score": score, "run_id": ctx.run_id}


class HyperparameterTuner:
    """
    Grid search and random search hyperparameter tuner.

    Evaluations are run in parallel via ProcessPoolExecutor.  Each trial is
    recorded as a separate run inside the ExperimentTracker.

    Parameters
    ----------
    tracker:
        ExperimentTracker instance to use for recording trials.
    experiment_name:
        Name prefix for tuning experiment runs.
    metric:
        Metric name to optimize (higher is better by default).
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        experiment_name: str = "hparam_search",
        metric: str = "ic",
    ) -> None:
        self.tracker = tracker
        self.experiment_name = experiment_name
        self.metric = metric

    def _run_trials(
        self,
        fn: Callable,
        param_list: List[Dict[str, Any]],
        n_jobs: int,
    ) -> TuneResult:
        results: List[Dict[str, Any]] = []

        # Use ThreadPoolExecutor to avoid pickle issues with closures/lambdas
        # in notebooks; callers can pass a plain module-level function for
        # true multiprocessing
        executor_cls = concurrent.futures.ThreadPoolExecutor
        if n_jobs > 1:
            try:
                # Test that fn is picklable before using processes
                pickle.dumps(fn)
                executor_cls = concurrent.futures.ProcessPoolExecutor
            except (pickle.PicklingError, AttributeError):
                logger.warning("fn is not picklable -- falling back to ThreadPoolExecutor")

        with executor_cls(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _evaluate_params,
                    fn,
                    params,
                    self.tracker,
                    self.experiment_name,
                    self.metric,
                )
                for params in param_list
            ]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    logger.error("Future failed: %s", exc)

        # Pick best by score (higher is better)
        valid = [r for r in results if not np.isnan(r["score"])]
        if not valid:
            raise RuntimeError("All trials failed -- no valid results")

        best = max(valid, key=lambda r: r["score"])

        return TuneResult(
            best_params=best["params"],
            best_score=best["score"],
            best_run_id=best["run_id"],
            all_results=results,
            metric=self.metric,
        )

    def grid_search(
        self,
        fn: Callable[[Dict[str, Any]], Any],
        param_grid: Dict[str, List[Any]],
        n_jobs: int = 4,
    ) -> TuneResult:
        """
        Exhaustive grid search over all combinations in param_grid.

        Parameters
        ----------
        fn:
            Callable that accepts a params dict and returns either a float
            (the metric score) or a dict of metric -> float.
        param_grid:
            Dict of param_name -> list of candidate values.
        n_jobs:
            Number of parallel workers.

        Returns
        -------
        TuneResult
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        param_list = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        logger.info(
            "Grid search: %d combinations for experiment %s",
            len(param_list),
            self.experiment_name,
        )
        return self._run_trials(fn, param_list, n_jobs)

    def random_search(
        self,
        fn: Callable[[Dict[str, Any]], Any],
        param_distributions: Dict[str, Any],
        n_trials: int = 100,
        n_jobs: int = 4,
        seed: Optional[int] = None,
    ) -> TuneResult:
        """
        Random search over param_distributions.

        Parameters
        ----------
        fn:
            Callable that accepts a params dict.
        param_distributions:
            Dict of param_name -> list (uniform choice) or callable ()-> value.
        n_trials:
            Number of random trials.
        n_jobs:
            Number of parallel workers.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        TuneResult
        """
        rng = random.Random(seed)
        param_list: List[Dict[str, Any]] = []

        for _ in range(n_trials):
            trial_params: Dict[str, Any] = {}
            for k, dist in param_distributions.items():
                if callable(dist) and not isinstance(dist, list):
                    trial_params[k] = dist()
                elif isinstance(dist, list):
                    trial_params[k] = rng.choice(dist)
                else:
                    trial_params[k] = dist
            param_list.append(trial_params)

        logger.info(
            "Random search: %d trials for experiment %s",
            n_trials,
            self.experiment_name,
        )
        return self._run_trials(fn, param_list, n_jobs)

    def successive_halving(
        self,
        fn: Callable[[Dict[str, Any]], Any],
        param_distributions: Dict[str, Any],
        n_initial: int = 32,
        reduction_factor: int = 3,
        n_jobs: int = 4,
        seed: Optional[int] = None,
    ) -> TuneResult:
        """
        Successive halving (SHA) scheduler.

        Start with n_initial random configs, evaluate them, keep the top
        1/reduction_factor, and repeat until one remains.

        Parameters
        ----------
        fn:
            Callable that accepts a params dict.
        param_distributions:
            Same format as random_search.
        n_initial:
            Number of initial candidates.
        reduction_factor:
            Fraction to eliminate at each round.
        n_jobs:
            Number of parallel workers.
        seed:
            Random seed.

        Returns
        -------
        TuneResult with best params across all rounds.
        """
        rng = random.Random(seed)
        candidates: List[Dict[str, Any]] = []

        for _ in range(n_initial):
            trial_params: Dict[str, Any] = {}
            for k, dist in param_distributions.items():
                if callable(dist) and not isinstance(dist, list):
                    trial_params[k] = dist()
                elif isinstance(dist, list):
                    trial_params[k] = rng.choice(dist)
                else:
                    trial_params[k] = dist
            candidates.append(trial_params)

        all_results: List[Dict[str, Any]] = []
        round_num = 0

        while len(candidates) > 1:
            logger.info(
                "SHA round %d: %d candidates", round_num, len(candidates)
            )
            round_results = self._run_trials(fn, candidates, n_jobs).all_results
            all_results.extend(round_results)

            valid = sorted(
                [r for r in round_results if not np.isnan(r["score"])],
                key=lambda r: r["score"],
                reverse=True,
            )
            keep = max(1, len(valid) // reduction_factor)
            candidates = [r["params"] for r in valid[:keep]]
            round_num += 1

        if not all_results:
            raise RuntimeError("No successful trials in successive halving")

        valid_all = [r for r in all_results if not np.isnan(r["score"])]
        best = max(valid_all, key=lambda r: r["score"])

        return TuneResult(
            best_params=best["params"],
            best_score=best["score"],
            best_run_id=best["run_id"],
            all_results=all_results,
            metric=self.metric,
        )
