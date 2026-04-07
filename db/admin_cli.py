"""
db/admin_cli.py -- Database administration CLI for SRFM.

Usage
-----
  python db/admin_cli.py migrate
  python db/admin_cli.py status
  python db/admin_cli.py backup --dest path/to/backup.db
  python db/admin_cli.py vacuum
  python db/admin_cli.py checkpoint [--mode TRUNCATE]
  python db/admin_cli.py integrity
  python db/admin_cli.py export-trades --since 2024-01-01 --format csv --output trades.csv
  python db/admin_cli.py query --sql "SELECT COUNT(*) FROM trades"
  python db/admin_cli.py prune --days 365
  python db/admin_cli.py validate
  python db/admin_cli.py stats

Environment variables
---------------------
  SRFM_DB_PATH   path to SQLite database file (default: ./srfm.db)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Allow running as a script from repo root: python db/admin_cli.py ...
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from db.schema import DBSchema
from db.archiver import DatabaseArchiver
from db.query_engine import SRFMDatabase, TradeQueries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("srfm.admin")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_DB = os.environ.get("SRFM_DB_PATH", "srfm.db")


def _resolve_db(args: argparse.Namespace) -> Path:
    """Return resolved Path to the database file."""
    p = Path(getattr(args, "db", DEFAULT_DB) or DEFAULT_DB).resolve()
    return p


def _print_table(rows: list[dict[str, Any]], max_col_width: int = 40) -> None:
    """Print a list of dicts as an ASCII table."""
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: min(max(len(c), max(len(str(r.get(c, ""))) for r in rows)), max_col_width)
              for c in cols}
    sep = "+" + "+".join("-" * (w + 2) for w in widths.values()) + "+"
    header = "|" + "|".join(f" {c:{widths[c]}} " for c in cols) + "|"
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        line = "|" + "|".join(
            f" {str(row.get(c, ''))[:widths[c]]:{widths[c]}} " for c in cols
        ) + "|"
        print(line)
    print(sep)
    print(f"  {len(rows)} rows")


def _require_db_exists(db_path: Path) -> None:
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        print("Tip: run 'python db/admin_cli.py migrate' to create and initialize it.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_migrate(args: argparse.Namespace) -> int:
    """Apply all pending migrations."""
    db_path = _resolve_db(args)
    target = getattr(args, "target", None)
    print(f"Database : {db_path}")
    print(f"Target   : {target or 'latest'}")
    try:
        DBSchema.migrate(db_path, target_version=target)
        # After migration also ensure base tables
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            DBSchema.create_tables(conn)
            version = DBSchema.get_current_version(conn)
        finally:
            conn.close()
        print(f"OK -- schema at version {version}")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show schema version, migration status, and table row counts."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        status = DBSchema.get_migration_status(conn)
    finally:
        conn.close()

    print(f"\n{'='*60}")
    print(f"  SRFM Database Status: {db_path.name}")
    print(f"{'='*60}")
    print(f"  Schema version : {status['current_version']}")
    print(f"  Up to date     : {'yes' if status['is_up_to_date'] else 'NO -- pending migrations exist'}")
    print()

    if status["applied"]:
        print("  Applied migrations:")
        for m in status["applied"][-5:]:
            print(f"    v{m['version']:03d}  {m.get('migration','?'):30s}  {m['applied_at']}")
        if len(status["applied"]) > 5:
            print(f"    ... ({len(status['applied'])-5} earlier migrations omitted)")

    if status["pending"]:
        print()
        print("  Pending migrations:")
        for p in status["pending"]:
            print(f"    v{p['version']:03d}  {p['file']}")

    # Table row counts
    try:
        arch = DatabaseArchiver(db_path)
        stats = arch.get_db_stats()
        print()
        print(f"  File size      : {stats['file_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  WAL size       : {stats['wal_size_bytes'] / 1024:.1f} KB")
        print(f"  Fragmentation  : {stats['fragmentation_ratio']*100:.1f}%")
        print(f"  Total rows     : {stats['total_rows']:,}")
        print()
        print("  Table row counts:")
        rows = [
            {"table": k, "rows": f"{v:,}"}
            for k, v in sorted(stats["table_counts"].items())
            if not k.startswith("sqlite_")
        ]
        _print_table(rows)
    except Exception as exc:
        print(f"  (Could not read db stats: {exc})")

    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    """Backup database to dest path."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    dest = Path(args.dest).resolve()
    print(f"Backing up {db_path} -> {dest} ...")
    arch = DatabaseArchiver(db_path)
    try:
        result = arch.backup(dest)
        size = result.stat().st_size / 1024 / 1024
        print(f"OK -- {result} ({size:.2f} MB)")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


def cmd_vacuum(args: argparse.Namespace) -> int:
    """VACUUM the database to reclaim space and defragment."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    print(f"Running VACUUM on {db_path} ...")
    arch = DatabaseArchiver(db_path)
    try:
        result = arch.vacuum()
        reclaimed = result["reclaimed_bytes"] / 1024
        print(f"OK -- reclaimed {reclaimed:.1f} KB in {result['duration_sec']:.1f}s")
        print(f"  Before: {result['size_before_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  After : {result['size_after_bytes']  / 1024 / 1024:.2f} MB")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


def cmd_checkpoint(args: argparse.Namespace) -> int:
    """Flush WAL into main database."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    mode = getattr(args, "mode", "TRUNCATE") or "TRUNCATE"
    print(f"WAL checkpoint ({mode}) on {db_path} ...")
    arch = DatabaseArchiver(db_path)
    try:
        result = arch.checkpoint(mode)
        print(f"OK -- WAL frames: {result['wal_frames']}, "
              f"checkpointed: {result['checkpointed_frames']}")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


def cmd_integrity(args: argparse.Namespace) -> int:
    """Run SQLite integrity check."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    print(f"Running integrity check on {db_path} ...")
    arch = DatabaseArchiver(db_path)
    errors = arch.integrity_check()
    if not errors:
        print("OK -- no integrity errors found")
        return 0
    print(f"FAILED -- {len(errors)} error(s):")
    for e in errors:
        print(f"  {e}")
    return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Check all expected tables and indexes exist."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        problems = DBSchema.validate_schema(conn)
    finally:
        conn.close()

    if not problems:
        print("OK -- schema is valid, all tables and indexes present")
        return 0
    print(f"PROBLEMS ({len(problems)}):")
    for p in problems:
        print(f"  {p}")
    return 1


def cmd_export_trades(args: argparse.Namespace) -> int:
    """Export trades to CSV or JSON."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    fmt = (getattr(args, "format", "csv") or "csv").lower()
    output = getattr(args, "output", None)
    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    symbol = getattr(args, "symbol", None)

    db = SRFMDatabase(db_path)
    tq = TradeQueries(db)
    df = tq.get_trades(since=since, until=until, symbol=symbol)

    if df.empty:
        print("No trades found matching the given filters.")
        return 0

    if fmt == "csv":
        dest = output or "trades_export.csv"
        df.to_csv(dest, index=False)
        print(f"Exported {len(df)} trades to {dest}")
    elif fmt == "json":
        dest = output or "trades_export.json"
        df.to_json(dest, orient="records", indent=2, date_format="iso")
        print(f"Exported {len(df)} trades to {dest}")
    else:
        print(f"Unknown format: {fmt!r}. Use csv or json.", file=sys.stderr)
        return 1

    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """
    Execute a read-only SQL query and print results.
    Rejects any statement that is not a SELECT.
    """
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    sql_raw = args.sql.strip()

    # Safety: only allow SELECT statements
    first_word = sql_raw.split()[0].upper() if sql_raw.split() else ""
    if first_word not in {"SELECT", "WITH", "EXPLAIN", "PRAGMA"}:
        print(
            f"ERROR: Only SELECT/WITH/EXPLAIN/PRAGMA queries are permitted. "
            f"Got: {first_word!r}",
            file=sys.stderr,
        )
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql_raw)
        rows = cur.fetchall()
        if not rows:
            print("(no rows)")
            return 0
        dicts = [dict(r) for r in rows]
        _print_table(dicts)
        return 0
    except sqlite3.Error as exc:
        print(f"SQL error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


def cmd_prune(args: argparse.Namespace) -> int:
    """Remove old data from high-volume tables."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    days = int(getattr(args, "days", 365))
    arch = DatabaseArchiver(db_path)

    total_deleted = 0
    print(f"Pruning data older than {days} days from {db_path} ...")

    r = arch.prune_signal_history(days=days)
    print(f"  signal_history  : {r['deleted_rows']:,} rows deleted")
    total_deleted += r["deleted_rows"]

    r = arch.prune_nav_log(days=min(days, 30))
    print(f"  nav_log         : {r['deleted_rows']:,} rows deleted (cap=30d)")
    total_deleted += r["deleted_rows"]

    r = arch.prune_risk_metrics(days=days)
    print(f"  risk_metrics    : {r['deleted_rows']:,} rows deleted")
    total_deleted += r["deleted_rows"]

    r = arch.prune_alerts_log(days=days, keep_unresolved=True)
    print(f"  alerts_log      : {r['deleted_rows']:,} rows deleted (unresolved kept)")
    total_deleted += r["deleted_rows"]

    print(f"Total rows removed: {total_deleted:,}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show detailed database statistics."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    arch = DatabaseArchiver(db_path)
    stats = arch.get_db_stats()

    print(f"\n{'='*50}")
    print(f"  Database Statistics: {db_path.name}")
    print(f"{'='*50}")
    print(f"  File size      : {stats['file_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  WAL size       : {stats['wal_size_bytes'] / 1024:.1f} KB")
    print(f"  Page size      : {stats['page_size']} bytes")
    print(f"  Total pages    : {stats['page_count']:,}")
    print(f"  Free pages     : {stats['free_pages']:,}")
    print(f"  Fragmentation  : {stats['fragmentation_ratio']*100:.1f}%")
    print(f"  Total rows     : {stats['total_rows']:,}")
    print(f"  Tables         : {stats['num_tables']}")
    print()
    print("  Per-table row counts:")
    rows = [
        {"table": k, "rows": f"{v:,}"}
        for k, v in sorted(stats["table_counts"].items(), key=lambda x: -x[1])
        if not k.startswith("sqlite_")
    ]
    _print_table(rows)
    return 0


def cmd_archive_trades(args: argparse.Namespace) -> int:
    """Archive old closed trades to a separate archive database."""
    db_path = _resolve_db(args)
    _require_db_exists(db_path)
    days = int(getattr(args, "days", 365))
    dest = getattr(args, "dest", None)
    arch = DatabaseArchiver(db_path)
    print(f"Archiving trades older than {days} days ...")
    try:
        result = arch.archive_old_trades(days_to_keep=days, archive_path=dest)
        print(f"OK -- archived {result['rows_archived']} rows to {result['archive_path']}")
        return 0
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python db/admin_cli.py",
        description="SRFM database administration tool",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        metavar="PATH",
        help=f"Path to SQLite database (default: $SRFM_DB_PATH or {DEFAULT_DB})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # migrate
    p_migrate = sub.add_parser("migrate", help="Apply pending migrations")
    p_migrate.add_argument(
        "--target", type=int, default=None, metavar="VERSION",
        help="Stop at this migration version (default: apply all)",
    )

    # status
    sub.add_parser("status", help="Show schema version and table stats")

    # backup
    p_backup = sub.add_parser("backup", help="Create a database backup")
    p_backup.add_argument("--dest", required=True, metavar="PATH", help="Destination path")

    # vacuum
    sub.add_parser("vacuum", help="VACUUM and ANALYZE the database")

    # checkpoint
    p_ckpt = sub.add_parser("checkpoint", help="Flush WAL into main database")
    p_ckpt.add_argument(
        "--mode", default="TRUNCATE",
        choices=["PASSIVE", "FULL", "RESTART", "TRUNCATE"],
        help="WAL checkpoint mode (default: TRUNCATE)",
    )

    # integrity
    sub.add_parser("integrity", help="Run SQLite integrity check")

    # validate
    sub.add_parser("validate", help="Check all tables and indexes exist")

    # stats
    sub.add_parser("stats", help="Show detailed database statistics")

    # export-trades
    p_exp = sub.add_parser("export-trades", help="Export trades to CSV or JSON")
    p_exp.add_argument("--since",  metavar="DATE", help="Start date (YYYY-MM-DD)")
    p_exp.add_argument("--until",  metavar="DATE", help="End date (YYYY-MM-DD)")
    p_exp.add_argument("--symbol", metavar="SYM",  help="Filter by symbol")
    p_exp.add_argument(
        "--format", default="csv", choices=["csv", "json"],
        help="Output format (default: csv)",
    )
    p_exp.add_argument("--output", metavar="PATH", help="Output file path")

    # query
    p_qry = sub.add_parser("query", help="Run a read-only SQL query")
    p_qry.add_argument("--sql", required=True, metavar="SQL", help="SELECT statement to execute")

    # prune
    p_prune = sub.add_parser("prune", help="Remove old data from high-volume tables")
    p_prune.add_argument(
        "--days", type=int, default=365, metavar="N",
        help="Delete records older than N days (default: 365)",
    )

    # archive-trades
    p_arch = sub.add_parser("archive-trades", help="Archive old closed trades")
    p_arch.add_argument(
        "--days", type=int, default=365, metavar="N",
        help="Keep trades from last N days (default: 365)",
    )
    p_arch.add_argument("--dest", metavar="PATH", help="Archive database path (optional)")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMAND_MAP = {
    "migrate":        cmd_migrate,
    "status":         cmd_status,
    "backup":         cmd_backup,
    "vacuum":         cmd_vacuum,
    "checkpoint":     cmd_checkpoint,
    "integrity":      cmd_integrity,
    "validate":       cmd_validate,
    "stats":          cmd_stats,
    "export-trades":  cmd_export_trades,
    "query":          cmd_query,
    "prune":          cmd_prune,
    "archive-trades": cmd_archive_trades,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure log level
    logging.getLogger().setLevel(args.log_level)

    handler = _COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
