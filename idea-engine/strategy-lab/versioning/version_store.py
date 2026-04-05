"""
version_store.py
----------------
SQLite-backed store for StrategyVersion objects.

Features
--------
* save / load / update versions
* lineage graph: parent -> children traversal
* query by status, tag, date range, author
* export version tree as JSON
* import/export for backup (JSON lines)
* tag management: add_tag, remove_tag, versions_with_tag
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterator

from .strategy_version import StrategyVersion, VersionStatus

# Default DB location inside strategy-lab
_DEFAULT_DB = Path(__file__).parent.parent / "strategy_lab.db"


class VersionStore:
    """
    Persistent, SQLite-backed registry of StrategyVersion objects.

    Parameters
    ----------
    db_path : path to the SQLite database file (created if absent)
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS versions (
                    version_id   TEXT PRIMARY KEY,
                    parent_id    TEXT,
                    created_at   TEXT NOT NULL,
                    parameters   TEXT NOT NULL,
                    param_hash   TEXT NOT NULL,
                    description  TEXT NOT NULL DEFAULT '',
                    author       TEXT NOT NULL DEFAULT 'manual',
                    status       TEXT NOT NULL DEFAULT 'DRAFT',
                    iae_idea_ids TEXT NOT NULL DEFAULT '[]',
                    notes        TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (parent_id) REFERENCES versions(version_id)
                );

                CREATE TABLE IF NOT EXISTS version_tags (
                    version_id TEXT NOT NULL,
                    tag        TEXT NOT NULL,
                    PRIMARY KEY (version_id, tag),
                    FOREIGN KEY (version_id) REFERENCES versions(version_id)
                );

                CREATE INDEX IF NOT EXISTS idx_versions_status     ON versions(status);
                CREATE INDEX IF NOT EXISTS idx_versions_param_hash ON versions(param_hash);
                CREATE INDEX IF NOT EXISTS idx_versions_created_at ON versions(created_at);
                CREATE INDEX IF NOT EXISTS idx_version_tags_tag    ON version_tags(tag);
            """)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, version: StrategyVersion) -> None:
        """Insert or replace a StrategyVersion in the store."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO versions
                    (version_id, parent_id, created_at, parameters, param_hash,
                     description, author, status, iae_idea_ids, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    version.version_id,
                    version.parent_id,
                    version.created_at,
                    json.dumps(version.parameters, default=str),
                    version.param_hash,
                    version.description,
                    version.author,
                    version.status.value,
                    json.dumps(version.iae_idea_ids),
                    version.notes,
                ),
            )
            # Sync tags
            conn.execute(
                "DELETE FROM version_tags WHERE version_id = ?", (version.version_id,)
            )
            for tag in version.tags:
                conn.execute(
                    "INSERT OR IGNORE INTO version_tags VALUES (?,?)",
                    (version.version_id, tag),
                )

    def load(self, version_id: str) -> StrategyVersion | None:
        """Load a single StrategyVersion by ID; returns None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if row is None:
                return None
            tags = [r["tag"] for r in conn.execute(
                "SELECT tag FROM version_tags WHERE version_id = ?", (version_id,)
            )]
            return self._row_to_version(row, tags)

    def delete(self, version_id: str) -> None:
        """Hard-delete a version (use archive() instead for soft-delete)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM version_tags WHERE version_id = ?", (version_id,))
            conn.execute("DELETE FROM versions WHERE version_id = ?", (version_id,))

    def update_status(self, version_id: str, status: VersionStatus) -> None:
        """Update only the status field of a version."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE versions SET status = ? WHERE version_id = ?",
                (status.value, version_id),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all(self) -> list[StrategyVersion]:
        """Return all versions ordered by created_at."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM versions ORDER BY created_at"
            ).fetchall()
            return [self._load_with_tags(conn, row) for row in rows]

    def by_status(self, status: VersionStatus) -> list[StrategyVersion]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM versions WHERE status = ? ORDER BY created_at",
                (status.value,),
            ).fetchall()
            return [self._load_with_tags(conn, row) for row in rows]

    def by_tag(self, tag: str) -> list[StrategyVersion]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT v.* FROM versions v
                JOIN version_tags t ON v.version_id = t.version_id
                WHERE t.tag = ?
                ORDER BY v.created_at
                """,
                (tag,),
            ).fetchall()
            return [self._load_with_tags(conn, row) for row in rows]

    def by_date_range(
        self, start: datetime, end: datetime
    ) -> list[StrategyVersion]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM versions WHERE created_at BETWEEN ? AND ? ORDER BY created_at",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
            return [self._load_with_tags(conn, row) for row in rows]

    def by_hash(self, param_hash: str) -> list[StrategyVersion]:
        """Find all versions with identical parameter fingerprint (deduplication)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM versions WHERE param_hash = ?", (param_hash,)
            ).fetchall()
            return [self._load_with_tags(conn, row) for row in rows]

    def champion(self) -> StrategyVersion | None:
        """Return the current CHAMPION version, or None if none exists."""
        champions = self.by_status(VersionStatus.CHAMPION)
        return champions[-1] if champions else None

    # ------------------------------------------------------------------
    # Lineage graph
    # ------------------------------------------------------------------

    def children_of(self, version_id: str) -> list[StrategyVersion]:
        """Return direct children of a version."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM versions WHERE parent_id = ? ORDER BY created_at",
                (version_id,),
            ).fetchall()
            return [self._load_with_tags(conn, row) for row in rows]

    def ancestors_of(self, version_id: str) -> list[StrategyVersion]:
        """
        Walk the parent chain from version_id back to the root.
        Returns list from root -> version_id (oldest first).
        """
        chain: list[StrategyVersion] = []
        current_id: str | None = version_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            v = self.load(current_id)
            if v is None:
                break
            chain.append(v)
            visited.add(current_id)
            current_id = v.parent_id
        chain.reverse()
        return chain

    def descendants_of(self, version_id: str) -> list[StrategyVersion]:
        """BFS all descendants of version_id."""
        result: list[StrategyVersion] = []
        queue = [version_id]
        visited: set[str] = set()
        while queue:
            vid = queue.pop(0)
            if vid in visited:
                continue
            visited.add(vid)
            for child in self.children_of(vid):
                result.append(child)
                queue.append(child.version_id)
        return result

    # ------------------------------------------------------------------
    # Tag management
    # ------------------------------------------------------------------

    def add_tag(self, version_id: str, tag: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO version_tags VALUES (?,?)", (version_id, tag)
            )

    def remove_tag(self, version_id: str, tag: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM version_tags WHERE version_id = ? AND tag = ?",
                (version_id, tag),
            )

    def all_tags(self) -> list[str]:
        """Return sorted list of all unique tags in the store."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT tag FROM version_tags ORDER BY tag"
            ).fetchall()
            return [r["tag"] for r in rows]

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_tree_json(self) -> str:
        """
        Export the full version tree as a nested JSON structure.
        Roots are versions without parents.
        """
        all_versions = self.all()
        by_id = {v.version_id: v.to_dict() for v in all_versions}
        # Build child-map
        children: dict[str, list[str]] = {v.version_id: [] for v in all_versions}
        roots: list[str] = []
        for v in all_versions:
            if v.parent_id and v.parent_id in children:
                children[v.parent_id].append(v.version_id)
            else:
                roots.append(v.version_id)

        def build_node(vid: str) -> dict:
            node = dict(by_id[vid])
            node["children"] = [build_node(c) for c in children.get(vid, [])]
            return node

        tree = [build_node(r) for r in roots]
        return json.dumps(tree, indent=2, default=str)

    def export_jsonl(self, path: str | Path) -> None:
        """Write all versions to a JSON-lines backup file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            for v in self.all():
                f.write(v.to_json(indent=0).replace("\n", "") + "\n")

    def import_jsonl(self, path: str | Path) -> int:
        """Import versions from a JSON-lines file. Returns count imported."""
        path = Path(path)
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                v = StrategyVersion.from_json(line)
                self.save(v)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_with_tags(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> StrategyVersion:
        tags = [r["tag"] for r in conn.execute(
            "SELECT tag FROM version_tags WHERE version_id = ?", (row["version_id"],)
        )]
        return self._row_to_version(row, tags)

    @staticmethod
    def _row_to_version(row: sqlite3.Row, tags: list[str]) -> StrategyVersion:
        return StrategyVersion(
            version_id=row["version_id"],
            parent_id=row["parent_id"],
            created_at=row["created_at"],
            parameters=json.loads(row["parameters"]),
            param_hash=row["param_hash"],
            description=row["description"],
            author=row["author"],
            tags=tags,
            status=VersionStatus(row["status"]),
            iae_idea_ids=json.loads(row["iae_idea_ids"]),
            notes=row["notes"],
        )

    def __repr__(self) -> str:
        total = len(self.all())
        return f"VersionStore(db={self.db_path}, versions={total})"
