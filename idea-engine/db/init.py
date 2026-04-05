"""
idea-engine/db/init.py
──────────────────────
Initialises idea_engine.db from schema.sql.

Usage
-----
    python -m idea-engine.db.init            # creates at default path
    python -m idea-engine.db.init --path /custom/path/idea_engine.db
    python -m idea-engine.db.init --reset    # drop and recreate (DESTRUCTIVE)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── paths ───────────────────────────────────────────────────────────────────

_HERE          = Path(__file__).parent                          # idea-engine/db/
SCHEMA_PATH    = _HERE / "schema.sql"
DEFAULT_DB     = _HERE.parent / "idea_engine.db"               # idea-engine/idea_engine.db
MIGRATIONS_DIR = _HERE / "migrations"


# ── helpers ─────────────────────────────────────────────────────────────────

def _read_schema() -> str:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")
    return SCHEMA_PATH.read_text(encoding="utf-8")


def _get_user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {version}")


def _list_migration_files() -> list[Path]:
    """Return migration SQL files sorted by numeric prefix."""
    MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        MIGRATIONS_DIR.glob("*.sql"),
        key=lambda p: int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 0,
    )
    return files


def apply_migrations(conn: sqlite3.Connection) -> int:
    """
    Apply any pending migration scripts from db/migrations/.
    Files must be named like:  001_add_column.sql, 002_rename_table.sql …
    Returns the number of migrations applied.
    """
    current_version = _get_user_version(conn)
    applied = 0
    for mf in _list_migration_files():
        prefix = mf.stem.split("_")[0]
        if not prefix.isdigit():
            continue
        migration_version = int(prefix)
        if migration_version <= current_version:
            continue
        logger.info("Applying migration %s …", mf.name)
        sql = mf.read_text(encoding="utf-8")
        conn.executescript(sql)
        _set_user_version(conn, migration_version)
        # record in event_log if it already exists
        try:
            conn.execute(
                "INSERT INTO event_log (event_type, payload_json) VALUES (?, ?)",
                ("migration_applied", f'{{"file": "{mf.name}"}}'),
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
        applied += 1
    return applied


def create_db(db_path: Path | str = DEFAULT_DB, *, reset: bool = False) -> sqlite3.Connection:
    """
    Create (or open) idea_engine.db, apply the base schema, then apply
    any pending migrations.

    Parameters
    ----------
    db_path : path for the SQLite file
    reset   : if True, back up and then delete the existing DB before recreating

    Returns
    -------
    An open sqlite3.Connection with WAL mode and foreign keys enabled.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if reset and db_path.exists():
        backup = db_path.with_suffix(
            f".backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.db"
        )
        shutil.copy2(db_path, backup)
        db_path.unlink()
        logger.warning("Reset requested — existing DB backed up to %s", backup)

    schema_sql = _read_schema()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Runtime PRAGMAs (schema.sql also sets them, but this guarantees they're active)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")

    # Apply base schema (all CREATE TABLE IF NOT EXISTS, so idempotent)
    conn.executescript(schema_sql)
    conn.commit()

    # Apply any pending migrations
    n_applied = apply_migrations(conn)
    if n_applied:
        logger.info("%d migration(s) applied.", n_applied)

    # Stamp a bootstrap event on first creation
    count = conn.execute("SELECT COUNT(*) FROM event_log WHERE event_type='db_initialised'").fetchone()[0]
    if count == 0:
        conn.execute(
            "INSERT INTO event_log (event_type, payload_json, severity) VALUES (?,?,?)",
            (
                "db_initialised",
                f'{{"db_path": "{db_path}", "schema": "{SCHEMA_PATH}"}}',
                "info",
            ),
        )
        conn.commit()
        logger.info("idea_engine.db initialised at %s", db_path)
    else:
        logger.debug("idea_engine.db already exists at %s — schema refreshed.", db_path)

    return conn


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Initialise idea_engine.db")
    parser.add_argument(
        "--path",
        default=str(DEFAULT_DB),
        help=f"Path for the SQLite database (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Back up and recreate the database from scratch (DESTRUCTIVE)",
    )
    parser.add_argument(
        "--migrate-only",
        action="store_true",
        dest="migrate_only",
        help="Only apply pending migrations, do not recreate tables",
    )
    args = parser.parse_args()

    if args.migrate_only:
        conn = sqlite3.connect(args.path)
        conn.execute("PRAGMA foreign_keys = ON")
        n = apply_migrations(conn)
        conn.close()
        print(f"Applied {n} migration(s).")
        return

    conn = create_db(db_path=args.path, reset=args.reset)
    conn.close()
    print(f"Database ready at: {args.path}")


if __name__ == "__main__":
    _cli()
