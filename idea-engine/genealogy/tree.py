"""
tree.py — GenealogyTree: SQLite-backed family tree for trading-strategy genomes.

Stores an adjacency list (genealogy_nodes + genealogy_edges) and exposes
traversal, pruning, and fitness-arc methods.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL (mirrors schema_extension.sql)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS genealogy_nodes (
    genome_id       INTEGER PRIMARY KEY,
    island          TEXT    NOT NULL,
    generation      INTEGER NOT NULL,
    params_json     TEXT    NOT NULL,
    fitness         REAL,
    mutation_ops    TEXT,
    is_hall_of_fame INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_genealogy_nodes_island
    ON genealogy_nodes (island);

CREATE INDEX IF NOT EXISTS idx_genealogy_nodes_fitness
    ON genealogy_nodes (fitness DESC);

CREATE TABLE IF NOT EXISTS genealogy_edges (
    child_id   INTEGER NOT NULL REFERENCES genealogy_nodes(genome_id),
    parent_id  INTEGER NOT NULL REFERENCES genealogy_nodes(genome_id),
    PRIMARY KEY (child_id, parent_id)
);

CREATE INDEX IF NOT EXISTS idx_genealogy_edges_child
    ON genealogy_edges (child_id);

CREATE INDEX IF NOT EXISTS idx_genealogy_edges_parent
    ON genealogy_edges (parent_id);

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
"""

# ---------------------------------------------------------------------------
# Node dataclass
# ---------------------------------------------------------------------------


class GenomeNode:
    """Lightweight container for a genome record from the DB."""

    __slots__ = (
        "genome_id", "island", "generation", "params",
        "fitness", "mutation_ops", "is_hall_of_fame", "created_at",
    )

    def __init__(
        self,
        genome_id: int,
        island: str,
        generation: int,
        params: dict[str, Any],
        fitness: float | None,
        mutation_ops: list[str],
        is_hall_of_fame: bool,
        created_at: str,
    ) -> None:
        self.genome_id       = genome_id
        self.island          = island
        self.generation      = generation
        self.params          = params
        self.fitness         = fitness
        self.mutation_ops    = mutation_ops
        self.is_hall_of_fame = is_hall_of_fame
        self.created_at      = created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id":       self.genome_id,
            "island":          self.island,
            "generation":      self.generation,
            "params":          self.params,
            "fitness":         self.fitness,
            "mutation_ops":    self.mutation_ops,
            "is_hall_of_fame": self.is_hall_of_fame,
            "created_at":      self.created_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "GenomeNode":
        return cls(
            genome_id       = row["genome_id"],
            island          = row["island"],
            generation      = row["generation"],
            params          = json.loads(row["params_json"] or "{}"),
            fitness         = row["fitness"],
            mutation_ops    = json.loads(row["mutation_ops"] or "[]"),
            is_hall_of_fame = bool(row["is_hall_of_fame"]),
            created_at      = row["created_at"],
        )


# ---------------------------------------------------------------------------
# GenealogyTree
# ---------------------------------------------------------------------------


class GenealogyTree:
    """
    SQLite-backed directed acyclic graph (DAG) of genome lineages.

    Each genome has zero or more parents (crossover → multiple parents;
    mutation → single parent; root → no parents).

    Parameters
    ----------
    db_path:
        Path to the SQLite database (idea_engine.db or in-memory ":memory:").
    hof_threshold:
        Fitness threshold above which genomes are auto-added to Hall of Fame.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        hof_threshold: float = 2.0,
    ) -> None:
        self._db_path      = str(db_path)
        self._hof_threshold = hof_threshold
        self._con          = self._connect()
        self._init_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init_schema(self) -> None:
        for stmt in _SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._con.execute(stmt)
        self._con.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    def __enter__(self) -> "GenealogyTree":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_genome(
        self,
        genome_id: int,
        parent_ids: list[int],
        mutation_ops: list[str],
        params: dict[str, Any],
        fitness: float | None,
        island: str = "default",
        generation: int = 0,
    ) -> GenomeNode:
        """
        Register a new genome in the tree.

        Parameters
        ----------
        genome_id:    Unique integer ID (must be globally unique).
        parent_ids:   List of parent genome IDs (empty for root genomes).
        mutation_ops: List of mutation operation names applied to create this genome.
        params:       Strategy parameter dict.
        fitness:      Sharpe (or other scalar fitness).  None if not yet evaluated.
        island:       Island label (for multi-island GA).
        generation:   Generation index within the island.

        Returns
        -------
        GenomeNode for the newly inserted genome.
        """
        is_hof = 1 if (fitness is not None and fitness >= self._hof_threshold) else 0

        self._con.execute(
            """
            INSERT OR REPLACE INTO genealogy_nodes
                (genome_id, island, generation, params_json,
                 fitness, mutation_ops, is_hall_of_fame)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                genome_id,
                island,
                generation,
                json.dumps(params),
                fitness,
                json.dumps(mutation_ops),
                is_hof,
            ),
        )

        for pid in parent_ids:
            self._con.execute(
                "INSERT OR IGNORE INTO genealogy_edges (child_id, parent_id) VALUES (?,?)",
                (genome_id, pid),
            )

        self._con.commit()
        logger.debug("Added genome %d (island=%s, gen=%d, fitness=%s)",
                     genome_id, island, generation, fitness)

        return self.get_node(genome_id)  # type: ignore[return-value]

    def update_fitness(self, genome_id: int, fitness: float) -> None:
        """Update the fitness of an existing genome."""
        is_hof = 1 if fitness >= self._hof_threshold else 0
        self._con.execute(
            "UPDATE genealogy_nodes SET fitness=?, is_hall_of_fame=? WHERE genome_id=?",
            (fitness, is_hof, genome_id),
        )
        self._con.commit()

    def mark_hall_of_fame(self, genome_id: int, flag: bool = True) -> None:
        """Manually mark or unmark a genome as Hall of Fame."""
        self._con.execute(
            "UPDATE genealogy_nodes SET is_hall_of_fame=? WHERE genome_id=?",
            (1 if flag else 0, genome_id),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Querying single nodes
    # ------------------------------------------------------------------

    def get_node(self, genome_id: int) -> GenomeNode | None:
        """Fetch a single genome node by ID."""
        row = self._con.execute(
            "SELECT * FROM genealogy_nodes WHERE genome_id=?", (genome_id,)
        ).fetchone()
        return GenomeNode.from_row(row) if row else None

    def _parents_of(self, genome_id: int) -> list[int]:
        rows = self._con.execute(
            "SELECT parent_id FROM genealogy_edges WHERE child_id=?", (genome_id,)
        ).fetchall()
        return [r["parent_id"] for r in rows]

    def _children_of(self, genome_id: int) -> list[int]:
        rows = self._con.execute(
            "SELECT child_id FROM genealogy_edges WHERE parent_id=?", (genome_id,)
        ).fetchall()
        return [r["child_id"] for r in rows]

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_lineage(self, genome_id: int) -> list[GenomeNode]:
        """
        Return all ancestors of a genome back to the root(s), in BFS order
        (nearest ancestors first).

        Returns an empty list if the genome does not exist.
        """
        if self.get_node(genome_id) is None:
            return []

        visited: set[int] = set()
        queue:   deque[int] = deque([genome_id])
        result:  list[GenomeNode] = []

        while queue:
            gid = queue.popleft()
            if gid in visited:
                continue
            visited.add(gid)

            if gid != genome_id:          # don't include the genome itself
                node = self.get_node(gid)
                if node:
                    result.append(node)

            for pid in self._parents_of(gid):
                if pid not in visited:
                    queue.append(pid)

        return result

    def get_descendants(self, genome_id: int) -> list[GenomeNode]:
        """
        Return all descendants of a genome (children, grandchildren, …),
        in BFS order (nearest first).
        """
        if self.get_node(genome_id) is None:
            return []

        visited: set[int] = set()
        queue:   deque[int] = deque([genome_id])
        result:  list[GenomeNode] = []

        while queue:
            gid = queue.popleft()
            if gid in visited:
                continue
            visited.add(gid)

            if gid != genome_id:
                node = self.get_node(gid)
                if node:
                    result.append(node)

            for cid in self._children_of(gid):
                if cid not in visited:
                    queue.append(cid)

        return result

    def find_common_ancestor(
        self, genome_a: int, genome_b: int
    ) -> GenomeNode | None:
        """
        Find the Lowest Common Ancestor (LCA) of two genomes.

        Uses a two-pass BFS: first collect all ancestors of A with their BFS
        depth; then BFS from B and return the first ancestor found in A's set.

        Returns None if there is no common ancestor.
        """
        def ancestors_with_depth(start: int) -> dict[int, int]:
            depth = 0
            visited: dict[int, int] = {}
            queue: deque[tuple[int, int]] = deque([(start, 0)])
            while queue:
                gid, d = queue.popleft()
                if gid in visited:
                    continue
                visited[gid] = d
                for pid in self._parents_of(gid):
                    if pid not in visited:
                        queue.append((pid, d + 1))
            return visited

        anc_a = ancestors_with_depth(genome_a)

        # BFS from B; first node in anc_a is the LCA
        queue: deque[tuple[int, int]] = deque([(genome_b, 0)])
        visited_b: set[int] = set()
        best_lca: int | None = None
        best_depth: int = 10**9

        while queue:
            gid, d = queue.popleft()
            if gid in visited_b:
                continue
            visited_b.add(gid)
            if gid in anc_a:
                total = anc_a[gid] + d
                if total < best_depth:
                    best_depth = total
                    best_lca   = gid
            for pid in self._parents_of(gid):
                if pid not in visited_b:
                    queue.append((pid, d + 1))

        return self.get_node(best_lca) if best_lca is not None else None

    # ------------------------------------------------------------------
    # Mutation path
    # ------------------------------------------------------------------

    def mutation_path(self, genome_id: int) -> list[dict[str, Any]]:
        """
        Return the ordered sequence of mutations from the root genome(s) to
        this genome.

        Each entry in the list is a dict:
            {genome_id, mutation_ops, fitness, generation}

        The list is ordered from root → target (oldest first).
        """
        # Collect the lineage and reverse it (lineage is nearest-first)
        lineage = self.get_lineage(genome_id)
        lineage.reverse()   # now oldest first

        # Append the target genome itself
        target = self.get_node(genome_id)
        if target:
            lineage.append(target)

        return [
            {
                "genome_id":    n.genome_id,
                "mutation_ops": n.mutation_ops,
                "fitness":      n.fitness,
                "generation":   n.generation,
            }
            for n in lineage
        ]

    # ------------------------------------------------------------------
    # Fitness arc
    # ------------------------------------------------------------------

    def fitness_arc(self, genome_id: int) -> list[dict[str, Any]]:
        """
        Return the time series of fitness values along the mutation path from
        root to this genome.

        Each entry: {genome_id, generation, fitness, cumulative_improvement}
        """
        path = self.mutation_path(genome_id)
        if not path:
            return []

        first_fitness = path[0]["fitness"] or 0.0
        result = []
        for step in path:
            f = step["fitness"]
            result.append({
                "genome_id":             step["genome_id"],
                "generation":            step["generation"],
                "fitness":               f,
                "cumulative_improvement": (f - first_fitness) if f is not None else None,
            })
        return result

    # ------------------------------------------------------------------
    # Hall of Fame
    # ------------------------------------------------------------------

    def hall_of_fame(self, limit: int = 50) -> list[GenomeNode]:
        """
        Return all Hall of Fame genomes, sorted by fitness descending.
        """
        rows = self._con.execute(
            """
            SELECT * FROM genealogy_nodes
            WHERE is_hall_of_fame = 1
            ORDER BY fitness DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [GenomeNode.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Best branch
    # ------------------------------------------------------------------

    def best_branch(self) -> int | None:
        """
        Return the genome_id of the highest-fitness leaf node.

        A leaf is defined as a genome with no children.
        """
        # All genome IDs that appear as parents
        parent_ids_set = {
            r["parent_id"]
            for r in self._con.execute(
                "SELECT DISTINCT parent_id FROM genealogy_edges"
            ).fetchall()
        }

        # All genome IDs
        all_ids = {
            r["genome_id"]
            for r in self._con.execute("SELECT genome_id FROM genealogy_nodes").fetchall()
        }

        leaf_ids = all_ids - parent_ids_set
        if not leaf_ids:
            # No edges: every node is a leaf
            leaf_ids = all_ids

        if not leaf_ids:
            return None

        placeholders = ",".join("?" * len(leaf_ids))
        row = self._con.execute(
            f"""
            SELECT genome_id FROM genealogy_nodes
            WHERE genome_id IN ({placeholders})
              AND fitness IS NOT NULL
            ORDER BY fitness DESC
            LIMIT 1
            """,
            list(leaf_ids),
        ).fetchone()

        return row["genome_id"] if row else None

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_dead_branches(self, min_fitness: float = -0.5) -> int:
        """
        Remove all genomes (and their edges) whose fitness is below
        ``min_fitness``, provided they are not ancestors of any Hall of Fame
        genome and have no descendants with higher fitness.

        Returns the number of genomes removed.
        """
        # Collect genome IDs to protect: HoF genomes + all their ancestors
        hof_rows = self._con.execute(
            "SELECT genome_id FROM genealogy_nodes WHERE is_hall_of_fame=1"
        ).fetchall()
        protected: set[int] = set()
        for row in hof_rows:
            protected.add(row["genome_id"])
            for anc in self.get_lineage(row["genome_id"]):
                protected.add(anc.genome_id)

        # Find candidates for pruning
        candidates_rows = self._con.execute(
            """
            SELECT genome_id FROM genealogy_nodes
            WHERE fitness IS NOT NULL
              AND fitness < ?
            """,
            (min_fitness,),
        ).fetchall()

        remove_ids: list[int] = []
        for row in candidates_rows:
            gid = row["genome_id"]
            if gid in protected:
                continue
            # Check no high-fitness descendants
            descendants = self.get_descendants(gid)
            high_fitness_desc = any(
                d.fitness is not None and d.fitness >= min_fitness
                for d in descendants
            )
            if not high_fitness_desc:
                remove_ids.append(gid)

        if not remove_ids:
            return 0

        placeholders = ",".join("?" * len(remove_ids))
        self._con.execute(
            f"DELETE FROM genealogy_edges WHERE child_id  IN ({placeholders})",
            remove_ids,
        )
        self._con.execute(
            f"DELETE FROM genealogy_edges WHERE parent_id IN ({placeholders})",
            remove_ids,
        )
        self._con.execute(
            f"DELETE FROM genealogy_nodes WHERE genome_id IN ({placeholders})",
            remove_ids,
        )
        self._con.commit()

        logger.info("Pruned %d dead-branch genomes (min_fitness=%.3f)",
                    len(remove_ids), min_fitness)
        return len(remove_ids)

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

    def all_nodes(self, island: str | None = None) -> list[GenomeNode]:
        """Return all (or island-filtered) genome nodes."""
        if island:
            rows = self._con.execute(
                "SELECT * FROM genealogy_nodes WHERE island=? ORDER BY generation",
                (island,),
            ).fetchall()
        else:
            rows = self._con.execute(
                "SELECT * FROM genealogy_nodes ORDER BY generation"
            ).fetchall()
        return [GenomeNode.from_row(r) for r in rows]

    def all_edges(self) -> list[tuple[int, int]]:
        """Return all (child_id, parent_id) edges."""
        rows = self._con.execute(
            "SELECT child_id, parent_id FROM genealogy_edges"
        ).fetchall()
        return [(r["child_id"], r["parent_id"]) for r in rows]

    def generation_summary(self, island: str | None = None) -> list[dict[str, Any]]:
        """
        Aggregate statistics per generation per island.

        Returns list of dicts: {island, generation, n_genomes, best_fitness,
        mean_fitness, n_hall_of_fame}
        """
        if island:
            rows = self._con.execute(
                """
                SELECT island, generation,
                       COUNT(*)          AS n_genomes,
                       MAX(fitness)      AS best_fitness,
                       AVG(fitness)      AS mean_fitness,
                       SUM(is_hall_of_fame) AS n_hof
                FROM genealogy_nodes
                WHERE island = ?
                GROUP BY island, generation
                ORDER BY island, generation
                """,
                (island,),
            ).fetchall()
        else:
            rows = self._con.execute(
                """
                SELECT island, generation,
                       COUNT(*)          AS n_genomes,
                       MAX(fitness)      AS best_fitness,
                       AVG(fitness)      AS mean_fitness,
                       SUM(is_hall_of_fame) AS n_hof
                FROM genealogy_nodes
                GROUP BY island, generation
                ORDER BY island, generation
                """
            ).fetchall()

        return [dict(r) for r in rows]

    def islands(self) -> list[str]:
        """Return sorted list of island names present in the tree."""
        rows = self._con.execute(
            "SELECT DISTINCT island FROM genealogy_nodes ORDER BY island"
        ).fetchall()
        return [r["island"] for r in rows]

    def size(self) -> int:
        """Total number of genomes in the tree."""
        return self._con.execute(
            "SELECT COUNT(*) FROM genealogy_nodes"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self.size()
        islands = self.islands()
        return (
            f"GenealogyTree(n_genomes={n}, islands={islands}, "
            f"db={self._db_path!r})"
        )
