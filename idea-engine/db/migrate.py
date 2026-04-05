"""
idea-engine/db/migrate.py
Runs the base schema + all module schema_extension.sql files against idea_engine.db.
Safe to run multiple times (all DDL uses IF NOT EXISTS).

Usage:
    python -m idea_engine.db.migrate
    python -m idea_engine.db.migrate --db /path/to/idea_engine.db
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]          # srfm-lab/
IAE_ROOT  = REPO_ROOT / "idea-engine"
DEFAULT_DB = IAE_ROOT / "idea_engine.db"

# Ordered list: base schema first, then extensions in a stable order
SCHEMA_FILES: list[Path] = []


def _collect_schemas() -> list[Path]:
    base = IAE_ROOT / "db" / "schema.sql"
    files: list[Path] = [base] if base.exists() else []

    # Extensions from every module — glob, sort for determinism
    pattern = str(IAE_ROOT / "**" / "schema_extension.sql")
    for p in sorted(glob.glob(pattern, recursive=True)):
        path = Path(p)
        if path not in files:
            files.append(path)

    return files


def run_migrations(db_path: Path | str = DEFAULT_DB) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    schemas = _collect_schemas()
    if not schemas:
        print("[migrate] No schema files found — nothing to do.", file=sys.stderr)
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    applied = 0
    errors  = 0
    for schema in schemas:
        rel = schema.relative_to(REPO_ROOT) if schema.is_relative_to(REPO_ROOT) else schema
        try:
            sql = schema.read_text(encoding="utf-8")
            conn.executescript(sql)
            conn.commit()
            print(f"  [ok]  {rel}")
            applied += 1
        except Exception as exc:
            print(f"  [ERR] {rel}: {exc}", file=sys.stderr)
            errors += 1

    conn.close()

    print(f"\n[migrate] Applied {applied} schema file(s), {errors} error(s) — DB: {db_path}")
    if errors:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IAE schema migrations")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to idea_engine.db")
    args = parser.parse_args()
    run_migrations(args.db)


if __name__ == "__main__":
    main()
