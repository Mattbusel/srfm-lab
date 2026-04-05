"""
experiment-tracker/lineage.py
==============================
Experiment lineage and provenance tracking.

Records parent–child relationships between experiments, enabling full
audit trails of how a strategy evolved:

    initial_hypothesis_test
        └─ parameter_variation (wider entry window)
              ├─ regime_split (bull)
              └─ regime_split (bear)
                    └─ wfa_fold_1
                    └─ wfa_fold_2

Relationship types
------------------
``hypothesis_test``
    The child tests a hypothesis derived from the parent's conclusions.
``parameter_variation``
    The child is a variant of the parent with one or more params changed.
``regime_split``
    The child evaluates the same strategy under a specific regime subset.
``wfa_fold``
    The child is one walk-forward fold of the parent's full WFA run.
``reproduction``
    The child is an exact re-run of the parent (same params, different seed).

Redundancy detection
--------------------
``detect_redundant_experiments()`` flags pairs of experiments that produced
near-identical metrics (cosine similarity > ``threshold``).  This helps
avoid wasting compute on re-discovering known results.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).resolve().parents[1] / "idea_engine.db"

VALID_RELATIONSHIPS = {
    "hypothesis_test",
    "parameter_variation",
    "regime_split",
    "wfa_fold",
    "reproduction",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LineageNode:
    """
    A single node in the experiment ancestry tree.

    Attributes
    ----------
    experiment_id    : int
    name             : str
    status           : str
    relationship     : str — how this node relates to its child (empty for root)
    depth            : int — 0 = the experiment we asked about, 1 = its parent, …
    sharpe           : float | None
    hypothesis_id    : int | None
    """

    experiment_id: int
    name: str
    status: str
    relationship: str
    depth: int
    sharpe: float | None = None
    hypothesis_id: int | None = None
    genome_id: int | None = None


@dataclass
class RedundancyPair:
    """A pair of experiments identified as near-duplicates."""

    id_a: int
    id_b: int
    similarity: float
    shared_metrics: dict[str, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExperimentLineage
# ---------------------------------------------------------------------------

class ExperimentLineage:
    """
    Manages parent–child relationships between experiments and generates
    provenance reports.

    Parameters
    ----------
    db_path : str | Path
        Path to ``idea_engine.db``.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self._db_path = Path(db_path)
        self._conn = self._open_connection()
        self._ensure_schema()
        logger.info("ExperimentLineage connected to %s.", self._db_path)

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        sql_path = Path(__file__).parent / "schema_extension.sql"
        if sql_path.exists():
            self._conn.executescript(sql_path.read_text(encoding="utf-8"))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ExperimentLineage":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_parent(
        self,
        child_id: int,
        parent_id: int,
        relationship_type: str,
    ) -> None:
        """
        Declare that ``child_id`` was derived from ``parent_id``.

        Parameters
        ----------
        child_id          : int
        parent_id         : int
        relationship_type : str — one of VALID_RELATIONSHIPS

        Raises
        ------
        ValueError if ``relationship_type`` is not recognised.
        """
        if relationship_type not in VALID_RELATIONSHIPS:
            raise ValueError(
                f"Unknown relationship type '{relationship_type}'. "
                f"Valid types: {sorted(VALID_RELATIONSHIPS)}."
            )
        try:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO experiment_lineage
                    (child_id, parent_id, relationship)
                VALUES (?, ?, ?)
                """,
                (child_id, parent_id, relationship_type),
            )
            self._conn.commit()
            logger.debug(
                "Lineage: experiment %d ─[%s]→ experiment %d.",
                parent_id, relationship_type, child_id,
            )
        except sqlite3.Error as exc:
            logger.error("Failed to record lineage %d→%d: %s", parent_id, child_id, exc)

    def get_ancestry(self, experiment_id: int) -> list[LineageNode]:
        """
        Walk the lineage tree upwards and return all ancestor experiments.

        The returned list is ordered nearest-first (direct parent first).
        Cycles are prevented by tracking visited IDs.

        Parameters
        ----------
        experiment_id : int

        Returns
        -------
        list[LineageNode] — ancestors, nearest first (does not include
        ``experiment_id`` itself).
        """
        visited: set[int] = {experiment_id}
        ancestors: list[LineageNode] = []
        current_ids = [experiment_id]
        depth = 1

        while current_ids:
            next_ids: list[int] = []
            for cid in current_ids:
                rows = self._conn.execute(
                    """
                    SELECT el.parent_id, el.relationship, e.name, e.status,
                           e.hypothesis_id, e.genome_id
                    FROM experiment_lineage el
                    JOIN experiments e ON e.id = el.parent_id
                    WHERE el.child_id = ?
                    """,
                    (cid,),
                ).fetchall()
                for row in rows:
                    pid = int(row["parent_id"])
                    if pid in visited:
                        continue
                    visited.add(pid)
                    sharpe = self._latest_metric(pid, "sharpe")
                    ancestors.append(
                        LineageNode(
                            experiment_id=pid,
                            name=str(row["name"]),
                            status=str(row["status"]),
                            relationship=str(row["relationship"]),
                            depth=depth,
                            sharpe=sharpe,
                            hypothesis_id=row["hypothesis_id"],
                            genome_id=row["genome_id"],
                        )
                    )
                    next_ids.append(pid)
            current_ids = next_ids
            depth += 1

        return ancestors

    def get_descendants(self, experiment_id: int) -> list[LineageNode]:
        """
        Walk the lineage tree downwards and return all descendant experiments.

        Parameters
        ----------
        experiment_id : int

        Returns
        -------
        list[LineageNode] — descendants, nearest first.
        """
        visited: set[int] = {experiment_id}
        descendants: list[LineageNode] = []
        current_ids = [experiment_id]
        depth = 1

        while current_ids:
            next_ids: list[int] = []
            for cid in current_ids:
                rows = self._conn.execute(
                    """
                    SELECT el.child_id, el.relationship, e.name, e.status,
                           e.hypothesis_id, e.genome_id
                    FROM experiment_lineage el
                    JOIN experiments e ON e.id = el.child_id
                    WHERE el.parent_id = ?
                    """,
                    (cid,),
                ).fetchall()
                for row in rows:
                    did = int(row["child_id"])
                    if did in visited:
                        continue
                    visited.add(did)
                    sharpe = self._latest_metric(did, "sharpe")
                    descendants.append(
                        LineageNode(
                            experiment_id=did,
                            name=str(row["name"]),
                            status=str(row["status"]),
                            relationship=str(row["relationship"]),
                            depth=depth,
                            sharpe=sharpe,
                            hypothesis_id=row["hypothesis_id"],
                            genome_id=row["genome_id"],
                        )
                    )
                    next_ids.append(did)
            current_ids = next_ids
            depth += 1

        return descendants

    def find_best_descendant(
        self,
        experiment_id: int,
        metric: str = "sharpe",
        higher_is_better: bool = True,
    ) -> LineageNode | None:
        """
        Find the best-performing experiment in the subtree rooted at
        ``experiment_id`` (inclusive).

        Parameters
        ----------
        experiment_id    : int
        metric           : str — metric key to optimise
        higher_is_better : bool

        Returns
        -------
        LineageNode | None — the best node, or None if no descendants
        have that metric logged.
        """
        # Include the root itself
        root_sharpe = self._latest_metric(experiment_id, metric)
        candidates: list[tuple[float, int]] = []
        if root_sharpe is not None:
            candidates.append((root_sharpe, experiment_id))

        for node in self.get_descendants(experiment_id):
            val = self._latest_metric(node.experiment_id, metric)
            if val is not None:
                candidates.append((val, node.experiment_id))

        if not candidates:
            return None

        best_val, best_id = max(candidates) if higher_is_better else min(candidates)

        # Fetch full node info
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (best_id,),
        ).fetchone()
        if row is None:
            return None

        return LineageNode(
            experiment_id=best_id,
            name=str(row["name"]),
            status=str(row["status"]),
            relationship="",
            depth=0,
            sharpe=best_val,
            hypothesis_id=row["hypothesis_id"],
            genome_id=row["genome_id"],
        )

    # ------------------------------------------------------------------
    # Provenance report
    # ------------------------------------------------------------------

    def provenance_report(self, experiment_id: int) -> str:
        """
        Generate a full provenance chain report as a Markdown document.

        The report includes:
        * Experiment metadata
        * All ancestor experiments with their key metrics
        * All descendant experiments
        * The best descendant per metric

        Parameters
        ----------
        experiment_id : int

        Returns
        -------
        str — Markdown-formatted provenance report.
        """
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return f"# Experiment {experiment_id} not found\n"

        ancestors = self.get_ancestry(experiment_id)
        descendants = self.get_descendants(experiment_id)

        lines: list[str] = [
            f"# Provenance Report — Experiment {experiment_id}",
            "",
            f"**Name**: {row['name']}  ",
            f"**Status**: `{row['status']}`  ",
            f"**Started**: {row['started_at']}  ",
            f"**Ended**: {row['ended_at'] or '—'}  ",
            f"**Duration**: {_fmt_duration(row['duration_seconds'])}  ",
            f"**Hypothesis ID**: {row['hypothesis_id'] or '—'}  ",
            f"**Genome ID**: {row['genome_id'] or '—'}",
            "",
        ]

        # Key metrics
        metrics = self._all_latest_metrics(experiment_id)
        if metrics:
            lines += ["## Metrics", ""]
            lines += [f"| Metric | Value |", "|--------|-------|"]
            for k, v in sorted(metrics.items()):
                lines.append(f"| `{k}` | {v:.4f} |")
            lines.append("")

        # Params
        params = self._all_params(experiment_id)
        if params:
            lines += ["## Parameters", ""]
            lines += ["| Parameter | Value |", "|-----------|-------|"]
            for k, v in sorted(params.items()):
                lines.append(f"| `{k}` | {v} |")
            lines.append("")

        # Ancestry
        if ancestors:
            lines += ["## Ancestry", ""]
            lines += ["| Depth | ID | Name | Relationship | Status | Sharpe |"]
            lines += ["|-------|----|------|--------------|--------|--------|"]
            for node in sorted(ancestors, key=lambda n: n.depth):
                sharpe_str = f"{node.sharpe:.4f}" if node.sharpe is not None else "—"
                lines.append(
                    f"| {node.depth} | {node.experiment_id} | {node.name} "
                    f"| `{node.relationship}` | {node.status} | {sharpe_str} |"
                )
            lines.append("")

        # Descendants
        if descendants:
            lines += ["## Descendants", ""]
            lines += ["| Depth | ID | Name | Relationship | Status | Sharpe |"]
            lines += ["|-------|----|------|--------------|--------|--------|"]
            for node in sorted(descendants, key=lambda n: n.depth):
                sharpe_str = f"{node.sharpe:.4f}" if node.sharpe is not None else "—"
                lines.append(
                    f"| {node.depth} | {node.experiment_id} | {node.name} "
                    f"| `{node.relationship}` | {node.status} | {sharpe_str} |"
                )
            lines.append("")

        # Best descendant
        best = self.find_best_descendant(experiment_id, "sharpe")
        if best:
            lines += [
                "## Best in Subtree (by Sharpe)",
                "",
                f"**Experiment {best.experiment_id}** — {best.name}  ",
                f"Sharpe: `{best.sharpe:.4f}`  ",
                f"Status: `{best.status}`",
                "",
            ]

        lines.append(
            f"*Report generated at "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}*"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Redundancy detection
    # ------------------------------------------------------------------

    def detect_redundant_experiments(
        self,
        threshold: float = 0.95,
        metric_keys: list[str] | None = None,
    ) -> list[RedundancyPair]:
        """
        Identify pairs of completed experiments whose metric vectors are
        near-identical (cosine similarity ≥ ``threshold``).

        Only experiments with status ``'completed'`` are considered.
        Experiments must share at least 2 metric keys for a meaningful
        comparison.

        Parameters
        ----------
        threshold   : float  — cosine-similarity threshold (default 0.95)
        metric_keys : list[str] | None
            Metrics to include in the comparison vector.  Defaults to
            ``['sharpe', 'max_dd', 'calmar', 'win_rate', 'total_return']``.

        Returns
        -------
        list[RedundancyPair]
        """
        if metric_keys is None:
            metric_keys = ["sharpe", "max_dd", "calmar", "win_rate", "total_return"]

        # Build metric vectors for all completed experiments
        rows = self._conn.execute(
            """
            SELECT e.id AS eid, m.key, m.value
            FROM experiments e
            JOIN experiment_metrics m ON m.experiment_id = e.id
            WHERE e.status = 'completed'
              AND m.key IN ({})
              AND (e.id, m.key, m.logged_at) IN (
                  SELECT experiment_id, key, MAX(logged_at)
                  FROM experiment_metrics
                  GROUP BY experiment_id, key
              )
            ORDER BY e.id
            """.format(",".join("?" * len(metric_keys))),
            metric_keys,
        ).fetchall()

        # Pivot: {exp_id: {metric: value}}
        vectors: dict[int, dict[str, float]] = {}
        for row in rows:
            eid = int(row["eid"])
            vectors.setdefault(eid, {})[str(row["key"])] = float(row["value"])

        # Only keep experiments with ≥ 2 metrics
        valid_ids = [eid for eid, v in vectors.items() if len(v) >= 2]

        redundant: list[RedundancyPair] = []

        for i, id_a in enumerate(valid_ids):
            for id_b in valid_ids[i + 1 :]:
                shared = set(vectors[id_a].keys()) & set(vectors[id_b].keys())
                if len(shared) < 2:
                    continue
                vec_a = [vectors[id_a][k] for k in sorted(shared)]
                vec_b = [vectors[id_b][k] for k in sorted(shared)]
                sim = _cosine_similarity(vec_a, vec_b)
                if sim >= threshold:
                    redundant.append(
                        RedundancyPair(
                            id_a=id_a,
                            id_b=id_b,
                            similarity=sim,
                            shared_metrics={
                                k: (vectors[id_a][k], vectors[id_b][k])
                                for k in sorted(shared)
                            },
                        )
                    )

        logger.info(
            "Redundancy scan: %d pair(s) found above threshold %.2f.",
            len(redundant), threshold,
        )
        return redundant

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _latest_metric(self, experiment_id: int, key: str) -> float | None:
        row = self._conn.execute(
            """
            SELECT value FROM experiment_metrics
            WHERE experiment_id = ? AND key = ?
            ORDER BY logged_at DESC
            LIMIT 1
            """,
            (experiment_id, key),
        ).fetchone()
        return float(row["value"]) if row else None

    def _all_latest_metrics(self, experiment_id: int) -> dict[str, float]:
        rows = self._conn.execute(
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
            (experiment_id, experiment_id),
        ).fetchall()
        return {r["key"]: float(r["value"]) for r in rows}

    def _all_params(self, experiment_id: int) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM experiment_params WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length numeric vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _fmt_duration(seconds: float | None) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"
