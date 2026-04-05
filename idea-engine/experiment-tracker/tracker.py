"""
experiment-tracker/tracker.py
==============================
Core experiment tracker.

Provides an MLflow-inspired API for recording every experiment run
(hypothesis test, genome optimisation, counterfactual scenario, WFA fold,
…) in ``idea_engine.db``.

Design goals
------------
* **No external dependencies** beyond Python stdlib + pandas.  All state
  lives in SQLite tables defined in ``schema_extension.sql``.
* **Reproducibility**: ``reproduce()`` re-runs an experiment with exactly
  the same parameters by dispatching to the appropriate subsystem.
* **Comparison**: ``compare_experiments()`` returns a wide-format DataFrame
  suitable for Jupyter or the dashboard.
* **Search**: ``search_experiments()`` supports arbitrary key/value filters
  on params and metrics.

Thread safety
-------------
The tracker opens *one* connection per instance.  For multi-process
scenarios each process should create its own ``ExperimentTracker``.
SQLite WAL mode allows concurrent readers with one writer.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV = "IDEA_ENGINE_DB"
DEFAULT_DB = Path(__file__).resolve().parents[1] / "idea_engine.db"

VALID_STATUSES = {"running", "completed", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """
    Full in-memory representation of one experiment row plus its associated
    params, metrics (latest value per key), and artifact names.
    """

    id: int
    name: str
    hypothesis_id: int | None
    genome_id: int | None
    status: str
    started_at: str
    ended_at: str | None
    duration_seconds: float | None
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_names: list[str] = field(default_factory=list)

    # Convenience accessors
    @property
    def sharpe(self) -> float | None:
        return self.metrics.get("sharpe")

    @property
    def max_dd(self) -> float | None:
        return self.metrics.get("max_dd")

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Return a single flat dict suitable for a DataFrame row.
        Param keys are prefixed with ``param_``, metric keys with ``metric_``.
        """
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "hypothesis_id": self.hypothesis_id,
            "genome_id": self.genome_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
        }
        for k, v in self.params.items():
            d[f"param_{k}"] = v
        for k, v in self.metrics.items():
            d[f"metric_{k}"] = v
        return d


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    SQLite-backed experiment tracker.

    Parameters
    ----------
    db_path : str | Path
        Path to ``idea_engine.db``.  Defaults to the standard location.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB,
    ) -> None:
        self._db_path = Path(db_path)
        self._conn = self._open_connection()
        self._ensure_schema()
        logger.info("ExperimentTracker connected to %s.", self._db_path)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        sql_path = Path(__file__).parent / "schema_extension.sql"
        if sql_path.exists():
            self._conn.executescript(sql_path.read_text(encoding="utf-8"))
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_experiment(
        self,
        name: str,
        hypothesis_id: int | None = None,
        genome_id: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> int:
        """
        Create a new experiment record and return its ID.

        Parameters
        ----------
        name          : str   — human-readable experiment name
        hypothesis_id : int | None — link to a hypothesis in ``hypotheses``
        genome_id     : int | None — link to a genome run
        params        : dict  — initial parameters to log immediately

        Returns
        -------
        int — the new experiment's integer ID
        """
        started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur = self._conn.execute(
            """
            INSERT INTO experiments (name, hypothesis_id, genome_id, status, started_at)
            VALUES (?, ?, ?, 'running', ?)
            """,
            (name, hypothesis_id, genome_id, started_at),
        )
        experiment_id = cur.lastrowid
        self._conn.commit()

        if params:
            for key, value in params.items():
                self.log_param(experiment_id, key, value)

        logger.info(
            "Started experiment %d: '%s' (hypothesis=%s, genome=%s).",
            experiment_id, name, hypothesis_id, genome_id,
        )
        return experiment_id  # type: ignore[return-value]

    def log_param(
        self,
        experiment_id: int,
        key: str,
        value: Any,
    ) -> None:
        """
        Record a single parameter value for an experiment.

        Attempting to log the same key twice raises ``ValueError`` — params
        are immutable once set (use a new experiment for variations).

        Parameters
        ----------
        experiment_id : int
        key           : str
        value         : Any — will be stored as ``str(value)``
        """
        try:
            self._conn.execute(
                """
                INSERT INTO experiment_params (experiment_id, key, value)
                VALUES (?, ?, ?)
                """,
                (experiment_id, str(key), str(value)),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(
                f"Parameter '{key}' already logged for experiment {experiment_id}. "
                "Create a new experiment for parameter variations."
            )

    def log_metric(
        self,
        experiment_id: int,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """
        Record a metric value, optionally tagged with a training step.

        Multiple values for the same key are allowed (forming a curve when
        ``step`` is supplied).

        Parameters
        ----------
        experiment_id : int
        key           : str    — metric name (e.g. 'sharpe', 'loss', 'equity')
        value         : float
        step          : int | None — training step / fold index for curves
        """
        logged_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._conn.execute(
            """
            INSERT INTO experiment_metrics (experiment_id, key, value, step, logged_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (experiment_id, str(key), float(value), step, logged_at),
        )
        self._conn.commit()

    def log_metrics(
        self,
        experiment_id: int,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Convenience wrapper for logging multiple metrics at once.

        Parameters
        ----------
        experiment_id : int
        metrics       : dict  {name: value}
        step          : int | None
        """
        for key, value in metrics.items():
            self.log_metric(experiment_id, key, value, step=step)

    def log_artifact(
        self,
        experiment_id: int,
        name: str,
        content: Any,
    ) -> None:
        """
        Store a named artifact (JSON/text) for an experiment.

        ``content`` is serialised to a JSON string if it is not already a
        string; dicts, lists, and dataclasses are all supported.

        Parameters
        ----------
        experiment_id : int
        name          : str   — artifact identifier
        content       : Any   — will be stored as JSON text
        """
        if isinstance(content, str):
            content_str = content
        else:
            try:
                content_str = json.dumps(content, default=str)
            except TypeError:
                content_str = str(content)

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._conn.execute(
            """
            INSERT INTO experiment_artifacts (experiment_id, name, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (experiment_id, name, content_str, created_at),
        )
        self._conn.commit()
        logger.debug("Artifact '%s' logged for experiment %d.", name, experiment_id)

    def end_experiment(
        self,
        experiment_id: int,
        status: str = "completed",
    ) -> None:
        """
        Finalise an experiment, recording its end time and duration.

        Parameters
        ----------
        experiment_id : int
        status        : str — one of 'completed', 'failed', 'cancelled'
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}.")

        ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Compute duration
        row = self._conn.execute(
            "SELECT started_at FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        duration: float | None = None
        if row:
            try:
                start_dt = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
                duration = (end_dt - start_dt).total_seconds()
            except (ValueError, AttributeError):
                pass

        self._conn.execute(
            """
            UPDATE experiments
            SET status = ?, ended_at = ?, duration_seconds = ?
            WHERE id = ?
            """,
            (status, ended_at, duration, experiment_id),
        )
        self._conn.commit()
        logger.info(
            "Ended experiment %d with status '%s' (duration=%.1fs).",
            experiment_id, status, duration or 0,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_experiment(self, experiment_id: int) -> ExperimentRecord:
        """
        Load a full ExperimentRecord by ID.

        Parameters
        ----------
        experiment_id : int

        Returns
        -------
        ExperimentRecord

        Raises
        ------
        KeyError if the experiment does not exist.
        """
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Experiment {experiment_id} not found.")
        return self._hydrate(dict(row))

    def search_experiments(
        self,
        filters: dict[str, Any] | None = None,
        *,
        status: str | None = None,
        hypothesis_id: int | None = None,
        genome_id: int | None = None,
        name_contains: str | None = None,
        limit: int = 200,
    ) -> list[ExperimentRecord]:
        """
        Search experiments with flexible filter criteria.

        Parameters
        ----------
        filters       : dict  — param key/value pairs to filter on
            e.g. ``{"n_folds": "8", "symbol": "BTCUSDT"}``
        status        : str | None — filter by experiment status
        hypothesis_id : int | None — filter by hypothesis
        genome_id     : int | None — filter by genome
        name_contains : str | None — substring match on experiment name
        limit         : int — maximum number of results

        Returns
        -------
        list[ExperimentRecord]
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if status:
            where_clauses.append("e.status = ?")
            params.append(status)
        if hypothesis_id is not None:
            where_clauses.append("e.hypothesis_id = ?")
            params.append(hypothesis_id)
        if genome_id is not None:
            where_clauses.append("e.genome_id = ?")
            params.append(genome_id)
        if name_contains:
            where_clauses.append("e.name LIKE ?")
            params.append(f"%{name_contains}%")

        # Filter by param values (requires EXISTS sub-query per filter)
        if filters:
            for key, value in filters.items():
                where_clauses.append(
                    """
                    EXISTS (
                        SELECT 1 FROM experiment_params ep
                        WHERE ep.experiment_id = e.id
                          AND ep.key = ?
                          AND ep.value = ?
                    )
                    """
                )
                params.extend([str(key), str(value)])

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        rows = self._conn.execute(
            f"""
            SELECT * FROM experiments e
            {where_sql}
            ORDER BY e.started_at DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()

        return [self._hydrate(dict(row)) for row in rows]

    def compare_experiments(self, ids: list[int]) -> pd.DataFrame:
        """
        Compare multiple experiments side-by-side.

        Returns a wide-format DataFrame where each row is one experiment and
        columns are all params (prefixed ``param_``) and the latest value of
        all metrics (prefixed ``metric_``).

        Parameters
        ----------
        ids : list[int] — experiment IDs to compare

        Returns
        -------
        pd.DataFrame
        """
        records = [self.get_experiment(eid) for eid in ids]
        rows = [r.to_flat_dict() for r in records]
        return pd.DataFrame(rows)

    def best_experiment(
        self,
        metric: str = "sharpe",
        higher_is_better: bool = True,
        *,
        status: str = "completed",
    ) -> ExperimentRecord:
        """
        Return the experiment with the best value for a given metric.

        Parameters
        ----------
        metric           : str  — metric key to optimise
        higher_is_better : bool — True for Sharpe, False for max_dd
        status           : str  — only consider experiments with this status

        Returns
        -------
        ExperimentRecord

        Raises
        ------
        ValueError if no completed experiments have that metric.
        """
        order = "DESC" if higher_is_better else "ASC"
        row = self._conn.execute(
            f"""
            SELECT e.*
            FROM experiments e
            JOIN experiment_metrics m ON m.experiment_id = e.id
            WHERE m.key = ?
              AND e.status = ?
            ORDER BY m.value {order}
            LIMIT 1
            """,
            (metric, status),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"No {status!r} experiments found with metric '{metric}'."
            )
        return self._hydrate(dict(row))

    # ------------------------------------------------------------------
    # Reproduce
    # ------------------------------------------------------------------

    def reproduce(self, experiment_id: int) -> int:
        """
        Re-run an experiment with the exact same parameters as a previous run.

        This creates a *new* experiment record (preserving the original) and
        logs a lineage link.  The actual execution is performed by calling
        back into the IAE scheduler via the ``event_log`` table.

        Parameters
        ----------
        experiment_id : int — the experiment to reproduce

        Returns
        -------
        int — the new experiment's ID
        """
        original = self.get_experiment(experiment_id)

        new_id = self.start_experiment(
            name=f"{original.name} [reproduce of {experiment_id}]",
            hypothesis_id=original.hypothesis_id,
            genome_id=original.genome_id,
            params=original.params,
        )

        # Record lineage
        self._conn.execute(
            """
            INSERT OR IGNORE INTO experiment_lineage (child_id, parent_id, relationship)
            VALUES (?, ?, 'reproduction')
            """,
            (new_id, experiment_id),
        )
        self._conn.commit()

        # Fire a scheduler event so the actual workload runs
        payload = json.dumps(
            {
                "new_experiment_id": new_id,
                "original_experiment_id": experiment_id,
                "params": original.params,
                "hypothesis_id": original.hypothesis_id,
                "genome_id": original.genome_id,
            }
        )
        try:
            self._conn.execute(
                "INSERT INTO event_log (event_type, payload_json) VALUES (?, ?)",
                ("experiment_reproduce_requested", payload),
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            logger.warning("event_log not available — skipping reproduction event.")

        logger.info(
            "Reproduction of experiment %d started as experiment %d.",
            experiment_id, new_id,
        )
        return new_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hydrate(self, row: dict[str, Any]) -> ExperimentRecord:
        """
        Enrich a raw ``experiments`` row with its params, latest metrics,
        and artifact names.
        """
        eid = int(row["id"])

        # Params
        param_rows = self._conn.execute(
            "SELECT key, value FROM experiment_params WHERE experiment_id = ?",
            (eid,),
        ).fetchall()
        params = {r["key"]: r["value"] for r in param_rows}

        # Latest metric per key
        metric_rows = self._conn.execute(
            """
            SELECT key, value
            FROM experiment_metrics
            WHERE experiment_id = ?
              AND (experiment_id, key, logged_at) IN (
                  SELECT experiment_id, key, MAX(logged_at)
                  FROM experiment_metrics
                  WHERE experiment_id = ?
                  GROUP BY key
              )
            """,
            (eid, eid),
        ).fetchall()
        metrics = {r["key"]: float(r["value"]) for r in metric_rows}

        # Artifact names
        artifact_rows = self._conn.execute(
            "SELECT name FROM experiment_artifacts WHERE experiment_id = ?",
            (eid,),
        ).fetchall()
        artifact_names = [r["name"] for r in artifact_rows]

        return ExperimentRecord(
            id=eid,
            name=str(row["name"]),
            hypothesis_id=row.get("hypothesis_id"),
            genome_id=row.get("genome_id"),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            ended_at=row.get("ended_at"),
            duration_seconds=row.get("duration_seconds"),
            params=params,
            metrics=metrics,
            artifact_names=artifact_names,
        )

    # ------------------------------------------------------------------
    # Metric curve access
    # ------------------------------------------------------------------

    def get_metric_history(
        self,
        experiment_id: int,
        key: str,
    ) -> pd.DataFrame:
        """
        Return the full logged history of a metric (useful for training curves).

        Parameters
        ----------
        experiment_id : int
        key           : str — metric name

        Returns
        -------
        pd.DataFrame with columns: step, value, logged_at
        """
        rows = self._conn.execute(
            """
            SELECT step, value, logged_at
            FROM experiment_metrics
            WHERE experiment_id = ? AND key = ?
            ORDER BY COALESCE(step, 0) ASC, logged_at ASC
            """,
            (experiment_id, key),
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_artifact(self, experiment_id: int, name: str) -> str:
        """
        Retrieve the content of a named artifact.

        Parameters
        ----------
        experiment_id : int
        name          : str

        Returns
        -------
        str — raw stored content (typically JSON)

        Raises
        ------
        KeyError if the artifact is not found.
        """
        row = self._conn.execute(
            """
            SELECT content
            FROM experiment_artifacts
            WHERE experiment_id = ? AND name = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (experiment_id, name),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Artifact '{name}' not found for experiment {experiment_id}."
            )
        return str(row["content"])

    def delete_experiment(self, experiment_id: int) -> None:
        """
        Hard-delete an experiment and all associated params, metrics,
        artifacts, and lineage edges.

        Use with caution — this is irreversible.

        Parameters
        ----------
        experiment_id : int
        """
        for table in (
            "experiment_params",
            "experiment_metrics",
            "experiment_artifacts",
        ):
            self._conn.execute(
                f"DELETE FROM {table} WHERE experiment_id = ?",
                (experiment_id,),
            )
        self._conn.execute(
            "DELETE FROM experiment_lineage WHERE child_id = ? OR parent_id = ?",
            (experiment_id, experiment_id),
        )
        self._conn.execute(
            "DELETE FROM experiments WHERE id = ?",
            (experiment_id,),
        )
        self._conn.commit()
        logger.warning("Experiment %d hard-deleted.", experiment_id)

    def list_experiments(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ExperimentRecord]:
        """
        List recent experiments, newest first.

        Parameters
        ----------
        limit  : int — maximum results
        offset : int — pagination offset

        Returns
        -------
        list[ExperimentRecord]
        """
        rows = self._conn.execute(
            """
            SELECT * FROM experiments
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [self._hydrate(dict(row)) for row in rows]
