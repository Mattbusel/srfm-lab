"""
tracker.py — EvolutionTracker: per-generation health monitoring for islands.

Records generation snapshots, detects stagnation and population convergence,
computes genome diversity, and logs inter-island migration events.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------


def _param_vector(params: dict[str, Any]) -> list[float]:
    """
    Extract a numeric feature vector from a genome's params dict.

    Non-numeric values are encoded by hash (mod 1000) / 1000.
    """
    vector: list[float] = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, (int, float)):
            vector.append(float(v))
        elif isinstance(v, bool):
            vector.append(1.0 if v else 0.0)
        elif isinstance(v, str):
            vector.append((hash(v) % 1000) / 1000.0)
        else:
            try:
                vector.append(float(v))
            except (TypeError, ValueError):
                vector.append(0.0)
    return vector


def _hamming_distance_normalised(a: list[float], b: list[float],
                                   n_bins: int = 10) -> float:
    """
    Normalised Hamming distance between two parameter vectors.

    Continuous values are discretised into ``n_bins`` equal-width bins
    per dimension using min/max of the two values.  Returns a value in [0, 1].
    """
    if not a or not b:
        return 0.0
    # Pad to same length
    L = max(len(a), len(b))
    a = a + [0.0] * (L - len(a))
    b = b + [0.0] * (L - len(b))

    mismatches = 0
    for ai, bi in zip(a, b):
        lo, hi = min(ai, bi), max(ai, bi)
        rng = hi - lo
        if rng == 0.0:
            continue
        bin_a = min(n_bins - 1, int((ai - lo) / rng * n_bins))
        bin_b = min(n_bins - 1, int((bi - lo) / rng * n_bins))
        if bin_a != bin_b:
            mismatches += 1

    return mismatches / L


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evolution_tracking (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    island          TEXT    NOT NULL,
    generation      INTEGER NOT NULL,
    pop_size        INTEGER NOT NULL,
    best_fitness    REAL,
    mean_fitness    REAL,
    diversity_index REAL,
    event           TEXT,
    created_at      TEXT    NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_evolution_tracking_island_gen
    ON evolution_tracking (island, generation);
"""

# ---------------------------------------------------------------------------
# EvolutionTracker
# ---------------------------------------------------------------------------


class EvolutionTracker:
    """
    Records and analyses the per-generation health of evolutionary islands.

    All data is persisted to the ``evolution_tracking`` table in
    ``idea_engine.db``.

    Parameters
    ----------
    db_path:
        Path to the SQLite database (defaults to ":memory:" for testing).
    stagnation_window:
        Number of generations with no improvement required to declare stagnation.
    convergence_threshold:
        Diversity index below which a population is considered converged.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        stagnation_window: int = 10,
        convergence_threshold: float = 0.05,
    ) -> None:
        self._db_path              = str(db_path)
        self._stagnation_window    = stagnation_window
        self._convergence_threshold = convergence_threshold
        self._con                  = self._connect()
        self._init_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        for stmt in _TRACKING_TABLE_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._con.execute(stmt)
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "EvolutionTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_generation(
        self,
        island: str,
        generation: int,
        population_stats: dict[str, Any],
        event: str | None = None,
    ) -> int:
        """
        Snapshot the state of an island at the end of a generation.

        Parameters
        ----------
        island:
            Island name.
        generation:
            Generation index (0-based).
        population_stats:
            Dict with at minimum:
              - ``pop_size``   (int)
              - ``best_fitness`` (float | None)
              - ``mean_fitness`` (float | None)
              - ``population``   list of dicts, each with a ``params`` key
                                 (used for diversity calculation)
        event:
            Optional event label: 'migration', 'stagnation', 'convergence', etc.

        Returns
        -------
        Inserted row ID.
        """
        pop_size     = int(population_stats.get("pop_size", 0))
        best_fitness = population_stats.get("best_fitness")
        mean_fitness = population_stats.get("mean_fitness")

        # Compute diversity index from population genome params
        population = population_stats.get("population", [])
        div_idx    = self.diversity_index(population) if population else None

        cur = self._con.execute(
            """
            INSERT INTO evolution_tracking
                (island, generation, pop_size, best_fitness,
                 mean_fitness, diversity_index, event)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (island, generation, pop_size, best_fitness,
             mean_fitness, div_idx, event),
        )
        self._con.commit()

        logger.debug(
            "Recorded gen %d on island=%s | best=%.4f mean=%.4f div=%.4f",
            generation, island,
            best_fitness or 0, mean_fitness or 0, div_idx or 0,
        )

        # Auto-detect stagnation / convergence and record secondary events
        if self.detect_stagnation(island, window=self._stagnation_window):
            self._log_event(island, generation, "stagnation_detected")

        if div_idx is not None and self.detect_convergence(island):
            self._log_event(island, generation, "convergence_detected")

        return cur.lastrowid  # type: ignore[return-value]

    def _log_event(self, island: str, generation: int, event: str) -> None:
        """Insert a zero-size sentinel row for an event."""
        self._con.execute(
            """
            INSERT INTO evolution_tracking
                (island, generation, pop_size, event)
            VALUES (?, ?, 0, ?)
            """,
            (island, generation, event),
        )
        self._con.commit()
        logger.info("Event [%s] on island=%s gen=%d", event, island, generation)

    # ------------------------------------------------------------------
    # Stagnation detection
    # ------------------------------------------------------------------

    def detect_stagnation(self, island: str, window: int | None = None) -> bool:
        """
        Return True if the best fitness has not improved over the last
        ``window`` generations on this island.

        Parameters
        ----------
        island:  Island name.
        window:  Number of generations to look back (default: ``self._stagnation_window``).
        """
        window = window or self._stagnation_window

        rows = self._con.execute(
            """
            SELECT best_fitness FROM evolution_tracking
            WHERE island=? AND pop_size > 0
            ORDER BY generation DESC
            LIMIT ?
            """,
            (island, window),
        ).fetchall()

        if len(rows) < window:
            return False   # not enough history

        best_values = [r["best_fitness"] for r in rows if r["best_fitness"] is not None]
        if not best_values:
            return False

        # Stagnated if max improvement over window is below a small epsilon
        return max(best_values) - min(best_values) < 1e-4

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def detect_convergence(self, island: str) -> bool:
        """
        Return True if the most recent diversity_index on this island is
        below the convergence threshold.
        """
        row = self._con.execute(
            """
            SELECT diversity_index FROM evolution_tracking
            WHERE island=? AND pop_size > 0 AND diversity_index IS NOT NULL
            ORDER BY generation DESC
            LIMIT 1
            """,
            (island,),
        ).fetchone()

        if row is None:
            return False

        return row["diversity_index"] < self._convergence_threshold

    # ------------------------------------------------------------------
    # Diversity index
    # ------------------------------------------------------------------

    def diversity_index(
        self,
        population: list[dict[str, Any]],
    ) -> float:
        """
        Compute a normalised diversity index for a population.

        Uses mean pairwise normalised Hamming distance over the parameter
        space.  Returns a value in [0, 1]:
          - 0.0 → all genomes are identical (fully converged)
          - 1.0 → maximally diverse

        Parameters
        ----------
        population:
            List of genome dicts, each with a ``params`` key (dict).
            Alternatively, if the dicts have no ``params`` key, the dict
            itself is treated as the params.
        """
        if not population or len(population) < 2:
            return 0.0

        vectors = [
            _param_vector(g.get("params", g) if isinstance(g, dict) else {})
            for g in population
        ]

        n       = len(vectors)
        total   = 0.0
        n_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                total   += _hamming_distance_normalised(vectors[i], vectors[j])
                n_pairs += 1

        return total / n_pairs if n_pairs > 0 else 0.0

    # ------------------------------------------------------------------
    # Migration logging
    # ------------------------------------------------------------------

    def migration_log(
        self,
        from_island: str,
        to_island:   str,
        genome_ids:  list[int],
        generation:  int = -1,
    ) -> None:
        """
        Record an island migration event.

        Parameters
        ----------
        from_island: Source island.
        to_island:   Destination island.
        genome_ids:  IDs of migrating genomes.
        generation:  Generation at which migration occurs.
        """
        if not genome_ids:
            return

        event_payload = json.dumps({
            "from_island": from_island,
            "to_island":   to_island,
            "genome_ids":  genome_ids,
            "n_migrants":  len(genome_ids),
        })

        self._con.execute(
            """
            INSERT INTO evolution_tracking
                (island, generation, pop_size, event)
            VALUES (?, ?, ?, ?)
            """,
            (to_island, generation, len(genome_ids),
             f"migration:{event_payload}"),
        )
        self._con.commit()
        logger.info(
            "Migration: %d genomes from %s → %s at gen %d",
            len(genome_ids), from_island, to_island, generation,
        )

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def get_island_history(
        self,
        island: str,
        include_events: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Return the full generation history for an island.

        Parameters
        ----------
        island:         Island name.
        include_events: Include zero-size event rows if True (default True).
        """
        query = """
            SELECT id, island, generation, pop_size,
                   best_fitness, mean_fitness, diversity_index, event, created_at
            FROM evolution_tracking
            WHERE island=?
        """
        if not include_events:
            query += " AND pop_size > 0"
        query += " ORDER BY id"

        rows = self._con.execute(query, (island,)).fetchall()
        return [dict(r) for r in rows]

    def get_fitness_series(self, island: str) -> dict[str, list[Any]]:
        """
        Return best and mean fitness time series for an island.

        Returns::

            {
              "generations":    [0, 1, 2, ...],
              "best_fitness":   [float | None, ...],
              "mean_fitness":   [float | None, ...],
              "diversity_index":[float | None, ...],
            }
        """
        rows = self._con.execute(
            """
            SELECT generation, best_fitness, mean_fitness, diversity_index
            FROM evolution_tracking
            WHERE island=? AND pop_size > 0
            ORDER BY generation
            """,
            (island,),
        ).fetchall()

        return {
            "generations":     [r["generation"]     for r in rows],
            "best_fitness":    [r["best_fitness"]    for r in rows],
            "mean_fitness":    [r["mean_fitness"]    for r in rows],
            "diversity_index": [r["diversity_index"] for r in rows],
        }

    def all_islands(self) -> list[str]:
        """Return all island names with recorded generations."""
        rows = self._con.execute(
            "SELECT DISTINCT island FROM evolution_tracking ORDER BY island"
        ).fetchall()
        return [r["island"] for r in rows]

    def latest_generation(self, island: str) -> int | None:
        """Return the latest generation index recorded for an island."""
        row = self._con.execute(
            """
            SELECT MAX(generation) AS max_gen FROM evolution_tracking
            WHERE island=? AND pop_size > 0
            """,
            (island,),
        ).fetchone()
        return row["max_gen"] if row else None

    def migration_events(
        self, island: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Return all migration events, optionally filtered to a target island.
        """
        query = """
            SELECT id, island, generation, pop_size, event, created_at
            FROM evolution_tracking
            WHERE event LIKE 'migration:%'
        """
        params: list[Any] = []
        if island:
            query += " AND island=?"
            params.append(island)
        query += " ORDER BY id"

        rows = self._con.execute(query, params).fetchall()
        result = []
        for r in rows:
            row_dict = dict(r)
            # Parse embedded JSON from event field
            try:
                payload_str = r["event"].split(":", 1)[1]
                row_dict["migration_data"] = json.loads(payload_str)
            except (IndexError, json.JSONDecodeError):
                row_dict["migration_data"] = {}
            result.append(row_dict)
        return result

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def island_summary(self) -> list[dict[str, Any]]:
        """
        Return a high-level summary per island:
        latest generation, best fitness ever, current diversity, stagnation flag.
        """
        islands = self.all_islands()
        result  = []
        for isl in islands:
            latest = self.latest_generation(isl)
            row = self._con.execute(
                """
                SELECT MAX(best_fitness) AS best_ever,
                       AVG(mean_fitness) AS avg_mean,
                       MIN(diversity_index) AS min_div
                FROM evolution_tracking
                WHERE island=? AND pop_size > 0
                """,
                (isl,),
            ).fetchone()

            result.append({
                "island":         isl,
                "latest_gen":     latest,
                "best_fitness":   row["best_ever"] if row else None,
                "avg_mean_fitness": row["avg_mean"] if row else None,
                "min_diversity":  row["min_div"] if row else None,
                "is_stagnated":   self.detect_stagnation(isl),
                "is_converged":   self.detect_convergence(isl),
            })
        return result

    def __repr__(self) -> str:
        islands = self.all_islands()
        return (
            f"EvolutionTracker(islands={islands}, "
            f"stagnation_window={self._stagnation_window}, "
            f"convergence_threshold={self._convergence_threshold})"
        )
