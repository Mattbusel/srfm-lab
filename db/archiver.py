"""
db/archiver.py -- Database archival and maintenance for SRFM.

Provides:
  DatabaseArchiver -- archive old trades, WAL checkpoint, vacuum, backup, pruning
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DatabaseArchiver:
    """
    Archival and maintenance operations for the SRFM SQLite database.

    All destructive operations (archive, prune) run inside transactions
    so they are atomic.  Backup uses the SQLite online backup API which
    is safe to run on a live database under WAL mode.

    Usage
    -----
    arch = DatabaseArchiver("/path/to/srfm.db")
    arch.checkpoint()
    arch.vacuum()
    arch.backup("/path/to/backup.db")
    arch.archive_old_trades(days_to_keep=365)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).resolve()
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database not found: {self._db_path}")

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------
    # WAL and fragmentation
    # ------------------------------------------------------------------

    def checkpoint(self, mode: str = "TRUNCATE") -> dict[str, int]:
        """
        Flush the WAL file into the main database.

        mode can be: PASSIVE, FULL, RESTART, TRUNCATE (default).
        Returns dict with wal_frames and checkpointed_frames.
        """
        valid = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
        if mode.upper() not in valid:
            raise ValueError(f"Invalid checkpoint mode {mode!r}. Must be one of {valid}")

        conn = self._connect()
        try:
            row = conn.execute(f"PRAGMA wal_checkpoint({mode.upper()})").fetchone()
            conn.commit()
            result = {
                "busy": row[0],
                "wal_frames": row[1],
                "checkpointed_frames": row[2],
                "mode": mode.upper(),
            }
            log.info("WAL checkpoint (%s): %s", mode, result)
            return result
        finally:
            conn.close()

    def vacuum(self) -> dict[str, Any]:
        """
        Run VACUUM to defragment the database and reclaim free pages.
        Also runs ANALYZE to update query planner statistics.
        Returns before/after file size and duration.
        """
        size_before = self._db_path.stat().st_size
        t0 = time.monotonic()

        conn = self._connect()
        try:
            # VACUUM must run outside a transaction
            conn.isolation_level = None
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.isolation_level = ""
        finally:
            conn.close()

        elapsed = time.monotonic() - t0
        size_after = self._db_path.stat().st_size
        result = {
            "size_before_bytes": size_before,
            "size_after_bytes": size_after,
            "reclaimed_bytes": size_before - size_after,
            "duration_sec": round(elapsed, 3),
        }
        log.info("VACUUM complete: %s", result)
        return result

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def backup(
        self,
        dest_path: str | Path,
        pages_per_step: int = 100,
        sleep_between_steps: float = 0.01,
    ) -> Path:
        """
        Atomic online backup via sqlite3.Connection.backup().
        Safe to run on a live database -- does not block writers.

        Returns the destination path.
        """
        dest = Path(dest_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file then rename for atomicity
        tmp = dest.with_suffix(".tmp")

        src = self._connect()
        dst = sqlite3.connect(str(tmp))
        try:
            src.backup(
                dst,
                pages=pages_per_step,
                sleep=sleep_between_steps,
            )
            dst.close()
            src.close()
        except Exception:
            dst.close()
            src.close()
            if tmp.exists():
                tmp.unlink()
            raise

        tmp.rename(dest)
        size = dest.stat().st_size
        log.info("Backup complete: %s (%d bytes)", dest, size)
        return dest

    # ------------------------------------------------------------------
    # Archive old trades
    # ------------------------------------------------------------------

    def archive_old_trades(
        self,
        days_to_keep: int = 365,
        archive_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Move closed trades older than days_to_keep to a separate archive DB.

        Steps:
          1. Identify old closed trades (exit_time < cutoff).
          2. Copy them to archive_path (SQLite file, auto-created if absent).
          3. Delete the originals from main DB in the same transaction.

        Returns stats: rows_archived, cutoff_date, archive_path.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).date()
        cutoff_str = cutoff.isoformat()

        if archive_path is None:
            archive_path = self._db_path.parent / f"archive_trades_{cutoff_str}.db"
        archive_path = Path(archive_path).resolve()

        conn = self._connect()
        try:
            old_rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE exit_time IS NOT NULL
                  AND date(exit_time) < ?
                """,
                (cutoff_str,),
            ).fetchall()

            if not old_rows:
                log.info("No trades to archive before %s", cutoff_str)
                return {"rows_archived": 0, "cutoff_date": cutoff_str, "archive_path": str(archive_path)}

            # Ensure archive DB has the trades table
            arch_conn = sqlite3.connect(str(archive_path))
            try:
                arch_conn.execute("PRAGMA journal_mode=WAL")
                arch_conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                      id INTEGER PRIMARY KEY, symbol TEXT, side TEXT,
                      qty INTEGER, entry_price REAL, entry_time TEXT,
                      exit_price REAL, exit_time TEXT, pnl REAL, pnl_pct REAL,
                      commission REAL, slippage REAL, strategy_version TEXT,
                      signal_name TEXT, regime TEXT, notes TEXT,
                      created_at TEXT, updated_at TEXT
                    );
                    CREATE TABLE IF NOT EXISTS archive_meta (
                      archived_at TEXT NOT NULL DEFAULT (datetime('now')),
                      source_db   TEXT,
                      row_count   INTEGER,
                      cutoff_date TEXT
                    );
                    """
                )
                cols = list(old_rows[0].keys())
                placeholders = ", ".join("?" * len(cols))
                col_list = ", ".join(cols)
                arch_conn.executemany(
                    f"INSERT OR IGNORE INTO trades ({col_list}) VALUES ({placeholders})",
                    [tuple(r) for r in old_rows],
                )
                arch_conn.execute(
                    "INSERT INTO archive_meta(source_db, row_count, cutoff_date) VALUES (?,?,?)",
                    (str(self._db_path), len(old_rows), cutoff_str),
                )
                arch_conn.commit()
            finally:
                arch_conn.close()

            # Delete from main DB
            old_ids = [r["id"] for r in old_rows]
            placeholders_del = ", ".join("?" * len(old_ids))
            with conn:
                conn.execute(
                    f"DELETE FROM trades WHERE id IN ({placeholders_del})",  # noqa: S608
                    old_ids,
                )

            result = {
                "rows_archived": len(old_rows),
                "cutoff_date": cutoff_str,
                "archive_path": str(archive_path),
            }
            log.info("Archived %d trades to %s", len(old_rows), archive_path)
            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Prune signal history
    # ------------------------------------------------------------------

    def prune_signal_history(self, days: int = 90) -> dict[str, int]:
        """
        Delete signal_history rows older than days.
        Returns count of rows deleted.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM signal_history WHERE bar_time < ?",
                    (cutoff,),
                )
                deleted = cur.rowcount
            log.info("Pruned %d signal_history rows older than %s", deleted, cutoff)
            return {"deleted_rows": deleted, "cutoff": cutoff}
        finally:
            conn.close()

    def prune_nav_log(self, days: int = 30) -> dict[str, int]:
        """Delete nav_log rows older than days (high-volume table)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM nav_log WHERE bar_time < ?",
                    (cutoff,),
                )
                deleted = cur.rowcount
            log.info("Pruned %d nav_log rows older than %s", deleted, cutoff)
            return {"deleted_rows": deleted, "cutoff": cutoff}
        finally:
            conn.close()

    def prune_risk_metrics(self, days: int = 90) -> dict[str, int]:
        """Delete risk_metrics snapshots older than days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM risk_metrics WHERE snapshot_time < ?",
                    (cutoff,),
                )
                deleted = cur.rowcount
            return {"deleted_rows": deleted, "cutoff": cutoff}
        finally:
            conn.close()

    def prune_alerts_log(self, days: int = 180, keep_unresolved: bool = True) -> dict[str, int]:
        """
        Delete resolved alerts older than days.
        If keep_unresolved is True (default), unresolved alerts are never deleted.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._connect()
        try:
            with conn:
                if keep_unresolved:
                    cur = conn.execute(
                        "DELETE FROM alerts_log WHERE alert_time < ? AND resolved = 1",
                        (cutoff,),
                    )
                else:
                    cur = conn.execute(
                        "DELETE FROM alerts_log WHERE alert_time < ?",
                        (cutoff,),
                    )
                deleted = cur.rowcount
            log.info("Pruned %d alerts_log rows", deleted)
            return {"deleted_rows": deleted, "cutoff": cutoff}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Database statistics
    # ------------------------------------------------------------------

    def get_db_stats(self) -> dict[str, Any]:
        """
        Return a comprehensive dict of database health statistics:
          - file_size_bytes
          - wal_size_bytes
          - page_count, page_size, free_pages, fragmentation_ratio
          - per-table row counts
          - total_rows
        """
        conn = self._connect()
        try:
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            page_size  = conn.execute("PRAGMA page_size").fetchone()[0]
            freelist   = conn.execute("PRAGMA freelist_count").fetchone()[0]

            fragmentation = freelist / page_count if page_count else 0.0

            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            table_counts: dict[str, int] = {}
            for tbl in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]  # noqa: S608
                    table_counts[tbl] = count
                except sqlite3.OperationalError:
                    table_counts[tbl] = -1

            total_rows = sum(v for v in table_counts.values() if v >= 0)

            file_size = self._db_path.stat().st_size

            wal_path = self._db_path.with_suffix(self._db_path.suffix + "-wal")
            wal_size = wal_path.stat().st_size if wal_path.exists() else 0

            return {
                "file_size_bytes": file_size,
                "wal_size_bytes": wal_size,
                "page_count": page_count,
                "page_size": page_size,
                "free_pages": freelist,
                "fragmentation_ratio": round(fragmentation, 4),
                "table_counts": table_counts,
                "total_rows": total_rows,
                "num_tables": len(tables),
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Integrity check
    # ------------------------------------------------------------------

    def integrity_check(self) -> list[str]:
        """
        Run SQLite integrity_check and quick_check.
        Returns list of error strings; empty list means healthy.
        """
        conn = self._connect()
        try:
            errors: list[str] = []
            rows = conn.execute("PRAGMA integrity_check").fetchall()
            for row in rows:
                msg = row[0]
                if msg.lower() != "ok":
                    errors.append(f"integrity_check: {msg}")
            fk_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
            for row in fk_rows:
                errors.append(f"foreign_key_check: table={row[0]} rowid={row[1]} parent={row[2]}")
            return errors
        finally:
            conn.close()

    def rotate_backups(
        self,
        backup_dir: str | Path,
        keep_daily: int = 7,
        keep_weekly: int = 4,
    ) -> dict[str, Any]:
        """
        Create a timestamped backup and prune old backups.
        Keeps keep_daily most-recent daily backups and keep_weekly weekly backups.
        Returns info about the new backup and any files deleted.
        """
        backup_dir = Path(backup_dir).resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = backup_dir / f"srfm_{stamp}.db"
        self.backup(dest)

        # Collect all backups sorted by mtime desc
        backups = sorted(
            backup_dir.glob("srfm_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Keep keep_daily most recent; delete the rest beyond keep_weekly
        to_keep = set(backups[:keep_daily])
        # Also keep every 7th beyond that for weekly retention
        for i, bp in enumerate(backups[keep_daily:]):
            if i % 7 == 0 and i // 7 < keep_weekly:
                to_keep.add(bp)

        deleted = []
        for bp in backups:
            if bp not in to_keep:
                bp.unlink()
                deleted.append(str(bp))

        return {
            "new_backup": str(dest),
            "backups_kept": len(to_keep),
            "backups_deleted": len(deleted),
            "deleted_files": deleted,
        }
