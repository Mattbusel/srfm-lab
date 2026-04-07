# ml/training/model_registry.py -- ML model registry with versioning, lineage, and promotion
from __future__ import annotations

import os
import pickle
import sqlite3
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelMetadata:
    """Full metadata record for a registered model."""

    model_id: str
    name: str
    version: str
    created_at: datetime
    trained_on_data: str  # description or date range of training data
    feature_names: List[str]
    metrics: Dict[str, float]  # {"sharpe": 1.2, "ic": 0.08, "icir": 0.65}
    hyperparams: Dict[str, Any]
    status: str  # "candidate", "production", "archived"
    parent_model_id: Optional[str] = None  # lineage tracking

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelMetadata":
        d = dict(d)
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)


@dataclass
class ModelRecord:
    """Lightweight summary record returned by list_models()."""

    model_id: str
    name: str
    version: str
    status: str
    created_at: datetime
    metrics: Dict[str, float]


@dataclass
class ComparisonResult:
    """Side-by-side metric comparison between two models."""

    model_id_1: str
    model_id_2: str
    name_1: str
    name_2: str
    metric_deltas: Dict[str, float]  # metric -> (model2 - model1)
    winner: str  # model_id of the better model by primary metric
    primary_metric: str


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS models (
    model_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    trained_on_data TEXT NOT NULL,
    feature_names   TEXT NOT NULL,  -- JSON list
    metrics         TEXT NOT NULL,  -- JSON dict
    hyperparams     TEXT NOT NULL,  -- JSON dict
    status          TEXT NOT NULL DEFAULT 'candidate',
    parent_model_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_models_name   ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
"""

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Persistent registry for ML models.

    Models are stored as pickle files under <registry_dir>/models/.
    Metadata is persisted in a SQLite database at <registry_dir>/registry.db.

    Example usage::

        registry = ModelRegistry("/path/to/registry")
        model_id = registry.register("signal_lgbm", lgbm_model, metadata)
        model, meta = registry.load_latest("signal_lgbm")
        registry.promote_to_production(model_id)
        prod_model, prod_meta = registry.get_production_model("signal_lgbm")
    """

    def __init__(self, registry_dir: str = "registry") -> None:
        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.db_path = self.registry_dir / "registry.db"

        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

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

    def _pickle_path(self, model_id: str) -> Path:
        return self.models_dir / f"{model_id}.pkl"

    def _row_to_metadata(self, row: sqlite3.Row) -> ModelMetadata:
        import json

        return ModelMetadata(
            model_id=row["model_id"],
            name=row["name"],
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            trained_on_data=row["trained_on_data"],
            feature_names=json.loads(row["feature_names"]),
            metrics=json.loads(row["metrics"]),
            hyperparams=json.loads(row["hyperparams"]),
            status=row["status"],
            parent_model_id=row["parent_model_id"],
        )

    def _row_to_record(self, row: sqlite3.Row) -> ModelRecord:
        import json

        return ModelRecord(
            model_id=row["model_id"],
            name=row["name"],
            version=row["version"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metrics=json.loads(row["metrics"]),
        )

    def _next_version(self, name: str) -> str:
        """Compute next semantic version for a model name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version FROM models WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        if row is None:
            return "1.0.0"
        parts = row["version"].split(".")
        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{major}.{minor}.{patch + 1}"
        except (ValueError, IndexError):
            return "1.0.0"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model: Any,
        metadata: ModelMetadata,
        overwrite_version: bool = False,
    ) -> str:
        """
        Register a model and persist it.

        Parameters
        ----------
        name:
            Logical name (e.g., "signal_lgbm", "vol_forecast").
        model:
            The model object -- must be picklable.
        metadata:
            A ModelMetadata instance.  model_id and version fields will be
            generated automatically if they are empty strings.
        overwrite_version:
            If True, use the version already set in metadata instead of
            auto-incrementing.

        Returns
        -------
        str
            The model_id assigned to this registration.
        """
        import json

        # Generate IDs
        if not metadata.model_id:
            metadata.model_id = str(uuid.uuid4())
        if not metadata.version or not overwrite_version:
            metadata.version = self._next_version(name)
        metadata.name = name

        model_id = metadata.model_id

        # Persist model object
        pkl_path = self._pickle_path(model_id)
        with open(pkl_path, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Persist metadata
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO models
                    (model_id, name, version, created_at, trained_on_data,
                     feature_names, metrics, hyperparams, status, parent_model_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    name,
                    metadata.version,
                    metadata.created_at.isoformat(),
                    metadata.trained_on_data,
                    json.dumps(metadata.feature_names),
                    json.dumps(metadata.metrics),
                    json.dumps(metadata.hyperparams),
                    metadata.status,
                    metadata.parent_model_id,
                ),
            )

        logger.info("Registered model %s name=%s version=%s", model_id, name, metadata.version)
        return model_id

    def load(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Load model object and metadata by model_id.

        Raises
        ------
        KeyError
            If model_id is not found.
        FileNotFoundError
            If the pickle file is missing.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()

        if row is None:
            raise KeyError(f"Model not found: {model_id}")

        pkl_path = self._pickle_path(model_id)
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Pickle file missing for model_id={model_id}: {pkl_path}"
            )

        with open(pkl_path, "rb") as fh:
            model = pickle.load(fh)

        metadata = self._row_to_metadata(row)
        return model, metadata

    def load_latest(self, name: str) -> Tuple[Any, ModelMetadata]:
        """
        Load the most recently registered model for a given name.

        Raises
        ------
        KeyError
            If no model with the given name exists.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM models
                WHERE name = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()

        if row is None:
            raise KeyError(f"No model found with name: {name}")

        return self.load(row["model_id"])

    def list_models(self, name: Optional[str] = None) -> List[ModelRecord]:
        """
        List all registered models, optionally filtered by name.

        Returns records ordered by created_at descending.
        """
        with self._connect() as conn:
            if name is not None:
                rows = conn.execute(
                    "SELECT * FROM models WHERE name = ? ORDER BY created_at DESC",
                    (name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM models ORDER BY created_at DESC"
                ).fetchall()

        return [self._row_to_record(r) for r in rows]

    def archive(self, model_id: str) -> None:
        """
        Mark a model as archived.

        Archived models are no longer considered for production serving.
        The pickle file is retained for audit purposes.
        """
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE models SET status = 'archived' WHERE model_id = ?",
                (model_id,),
            )
            if result.rowcount == 0:
                raise KeyError(f"Model not found: {model_id}")

        logger.info("Archived model %s", model_id)

    def promote_to_production(self, model_id: str) -> None:
        """
        Promote a model to production status.

        Any previously production model with the same name is demoted to
        'candidate' before promotion.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Model not found: {model_id}")

            name = row["name"]

            # Demote existing production models for this name
            conn.execute(
                """
                UPDATE models SET status = 'candidate'
                WHERE name = ? AND status = 'production'
                """,
                (name,),
            )

            # Promote the target
            conn.execute(
                "UPDATE models SET status = 'production' WHERE model_id = ?",
                (model_id,),
            )

        logger.info("Promoted model %s (name=%s) to production", model_id, name)

    def get_production_model(self, name: str) -> Tuple[Any, ModelMetadata]:
        """
        Return the current production model for a given name.

        Raises
        ------
        KeyError
            If no production model exists for the name.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM models
                WHERE name = ? AND status = 'production'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()

        if row is None:
            raise KeyError(f"No production model found for name: {name}")

        return self.load(row["model_id"])

    def compare_models(
        self,
        id1: str,
        id2: str,
        primary_metric: str = "ic",
    ) -> ComparisonResult:
        """
        Compare two models side-by-side on their stored metrics.

        Parameters
        ----------
        id1, id2:
            model_ids to compare.
        primary_metric:
            The metric used to determine the winner.

        Returns
        -------
        ComparisonResult
            Contains metric deltas (model2 - model1) and a winner.
        """
        with self._connect() as conn:
            row1 = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (id1,)
            ).fetchone()
            row2 = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (id2,)
            ).fetchone()

        if row1 is None:
            raise KeyError(f"Model not found: {id1}")
        if row2 is None:
            raise KeyError(f"Model not found: {id2}")

        meta1 = self._row_to_metadata(row1)
        meta2 = self._row_to_metadata(row2)

        all_keys = set(meta1.metrics) | set(meta2.metrics)
        deltas: Dict[str, float] = {}
        for k in all_keys:
            v1 = meta1.metrics.get(k, float("nan"))
            v2 = meta2.metrics.get(k, float("nan"))
            deltas[k] = v2 - v1

        # Determine winner by primary metric (higher is better)
        v1_primary = meta1.metrics.get(primary_metric, float("-inf"))
        v2_primary = meta2.metrics.get(primary_metric, float("-inf"))
        winner = id2 if v2_primary >= v1_primary else id1

        return ComparisonResult(
            model_id_1=id1,
            model_id_2=id2,
            name_1=meta1.name,
            name_2=meta2.name,
            metric_deltas=deltas,
            winner=winner,
            primary_metric=primary_metric,
        )

    def delete(self, model_id: str) -> None:
        """
        Permanently delete a model and its pickle file.

        Use with caution -- this is irreversible.  Prefer archive() in most
        production scenarios.
        """
        with self._connect() as conn:
            result = conn.execute(
                "DELETE FROM models WHERE model_id = ?", (model_id,)
            )
            if result.rowcount == 0:
                raise KeyError(f"Model not found: {model_id}")

        pkl_path = self._pickle_path(model_id)
        if pkl_path.exists():
            pkl_path.unlink()

        logger.info("Deleted model %s", model_id)

    def get_lineage(self, model_id: str) -> List[ModelMetadata]:
        """
        Walk the parent_model_id chain and return full lineage, oldest first.

        Stops at root (parent_model_id is None) or if a cycle is detected.
        """
        chain: List[ModelMetadata] = []
        visited: set = set()
        current_id: Optional[str] = model_id

        while current_id and current_id not in visited:
            visited.add(current_id)
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM models WHERE model_id = ?", (current_id,)
                ).fetchone()
            if row is None:
                break
            meta = self._row_to_metadata(row)
            chain.append(meta)
            current_id = meta.parent_model_id

        chain.reverse()
        return chain

    def update_metrics(self, model_id: str, metrics: Dict[str, float]) -> None:
        """
        Merge additional metrics into an existing model's metric dictionary.

        Useful for adding post-deployment live-trading metrics.
        """
        import json

        with self._connect() as conn:
            row = conn.execute(
                "SELECT metrics FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Model not found: {model_id}")

            existing = json.loads(row["metrics"])
            existing.update(metrics)

            conn.execute(
                "UPDATE models SET metrics = ? WHERE model_id = ?",
                (json.dumps(existing), model_id),
            )

    def search(
        self,
        name: Optional[str] = None,
        status: Optional[str] = None,
        min_metric: Optional[Dict[str, float]] = None,
    ) -> List[ModelRecord]:
        """
        Search models with optional filters.

        Parameters
        ----------
        name:
            Filter by model name (exact match).
        status:
            Filter by status ("candidate", "production", "archived").
        min_metric:
            Dict of {metric_name: min_value} -- models must meet all thresholds.
            Filtering on metrics is done in Python after the DB query.
        """
        clauses: List[str] = []
        params: List[Any] = []

        if name is not None:
            clauses.append("name = ?")
            params.append(name)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        sql = "SELECT * FROM models"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = [self._row_to_record(r) for r in rows]

        if min_metric:
            filtered: List[ModelRecord] = []
            for rec in records:
                if all(rec.metrics.get(k, float("-inf")) >= v for k, v in min_metric.items()):
                    filtered.append(rec)
            return filtered

        return records

    def export_metadata_csv(self, output_path: str) -> None:
        """Export all model metadata to a CSV file for reporting."""
        import csv
        import json

        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()

        with open(output_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["model_id", "name", "version", "status", "created_at",
                 "trained_on_data", "metrics", "hyperparams", "parent_model_id"]
            )
            for row in rows:
                writer.writerow([
                    row["model_id"],
                    row["name"],
                    row["version"],
                    row["status"],
                    row["created_at"],
                    row["trained_on_data"],
                    row["metrics"],
                    row["hyperparams"],
                    row["parent_model_id"],
                ])

    def count_by_status(self) -> Dict[str, int]:
        """Return counts grouped by status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM models GROUP BY status"
            ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def rollback_production(self, name: str) -> Optional[str]:
        """
        Roll back production to the previous production model.

        Finds the most recently created candidate (non-archived) model for
        the given name and promotes it.  The current production model is
        demoted to candidate.

        Returns
        -------
        str or None
            The model_id of the newly promoted model, or None if no
            candidate was found.
        """
        with self._connect() as conn:
            # Find current production model
            prod_row = conn.execute(
                "SELECT model_id FROM models WHERE name = ? AND status = 'production'",
                (name,),
            ).fetchone()

            if prod_row is None:
                return None

            # Find most recent candidate
            candidate_row = conn.execute(
                """
                SELECT model_id FROM models
                WHERE name = ? AND status = 'candidate'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()

        if candidate_row is None:
            return None

        self.promote_to_production(candidate_row["model_id"])
        return candidate_row["model_id"]

    def clone_metadata(self, model_id: str, new_name: Optional[str] = None) -> ModelMetadata:
        """
        Return a copy of metadata with a fresh model_id for re-registration.

        Useful for creating a child model that inherits hyperparams from a parent.
        """
        import copy

        _, meta = self.load(model_id)
        new_meta = copy.deepcopy(meta)
        new_meta.model_id = str(uuid.uuid4())
        new_meta.parent_model_id = model_id
        new_meta.created_at = datetime.utcnow()
        new_meta.status = "candidate"
        if new_name:
            new_meta.name = new_name
        return new_meta
