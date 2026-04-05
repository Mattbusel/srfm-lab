"""
NotebookRunner: execute research templates, capture output, persist results,
and support scheduled periodic runs.

CLI usage::

    python -m idea_engine.research_notebooks.runner --template market_state_report
    python -m idea_engine.research_notebooks.runner --template signal_analysis --params '{"signal_name":"BH"}'
    python -m idea_engine.research_notebooks.runner --list
    python -m idea_engine.research_notebooks.runner --schedule market_state_report 06:00

Persists results to SQLite.  Schedule uses a simple polling loop (UTC times).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_DB = Path(__file__).parent / "notebook_results.db"

DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    template     TEXT NOT NULL,
    params       TEXT,            -- JSON
    result       TEXT,            -- JSON output
    status       TEXT NOT NULL,   -- success | error
    error_msg    TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_template ON runs(template);
CREATE INDEX IF NOT EXISTS idx_runs_started  ON runs(started_at);
"""


@dataclass
class RunRecord:
    """A single notebook execution record."""

    template: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    status: str
    error_msg: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    run_id: Optional[int] = None


class NotebookRunner:
    """
    Executes research notebook templates and persists their results.

    Built-in templates:
      - market_state_report
      - signal_analysis
      - regime_analysis
      - instrument_autopsy
      - idea_validation

    Usage::

        runner = NotebookRunner()
        result = runner.run("market_state_report")
        runner.schedule("market_state_report", utc_hour=6)  # blocks — use in a thread
    """

    TEMPLATES: Dict[str, str] = {
        "market_state_report": "templates.market_state_report.MarketStateReportTemplate",
        "signal_analysis":     "templates.signal_analysis.SignalAnalysisTemplate",
        "regime_analysis":     "templates.regime_analysis.RegimeAnalysisTemplate",
        "instrument_autopsy":  "templates.instrument_autopsy.InstrumentAutopsyTemplate",
        "idea_validation":     "templates.idea_validation.IdeaValidationTemplate",
    }

    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        self._db_path = db_path
        self._init_db()

    # ── public API ──────────────────────────────────────────────────────────────

    def run(
        self,
        template_name: str,
        params: Optional[Dict[str, Any]] = None,
        trades_df: Optional[pd.DataFrame] = None,
    ) -> RunRecord:
        """
        Execute a template and return a RunRecord with the result.

        Parameters
        ----------
        template_name : str
        params        : dict of kwargs forwarded to template.run()
        trades_df     : if template needs trade data, pass it here;
                        otherwise synthetic data is generated
        """
        params = params or {}
        started = datetime.now(timezone.utc)
        logger.info("Running template '%s' with params %s", template_name, params)

        try:
            result_dict = self._execute(template_name, params, trades_df)
            record = RunRecord(
                template=template_name,
                params=params,
                result=result_dict,
                status="success",
                started_at=started,
                finished_at=datetime.now(timezone.utc),
            )
        except Exception as exc:
            logger.exception("Template '%s' failed: %s", template_name, exc)
            record = RunRecord(
                template=template_name,
                params=params,
                result=None,
                status="error",
                error_msg=str(exc),
                started_at=started,
                finished_at=datetime.now(timezone.utc),
            )

        self._save_record(record)
        return record

    def get_last_result(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Return the most recent successful result for a template."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT result FROM runs
                   WHERE template = ? AND status = 'success'
                   ORDER BY started_at DESC LIMIT 1""",
                (template_name,),
            ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None

    def list_runs(
        self, template_name: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if template_name:
                rows = conn.execute(
                    "SELECT run_id, template, status, started_at, finished_at FROM runs "
                    "WHERE template = ? ORDER BY started_at DESC LIMIT ?",
                    (template_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id, template, status, started_at, finished_at FROM runs "
                    "ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(zip(("run_id", "template", "status", "started_at", "finished_at"), r)) for r in rows]

    def schedule(
        self,
        template_name: str,
        utc_hour: int = 6,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Block and run *template_name* every day at *utc_hour* UTC.

        Designed to run in a background thread or process.
        """
        logger.info(
            "Scheduling '%s' to run daily at %02d:00 UTC", template_name, utc_hour
        )
        while True:
            now = datetime.now(timezone.utc)
            if now.hour == utc_hour and now.minute < 5:
                self.run(template_name, params)
                time.sleep(300)  # avoid double-firing within the same 5-min window
            else:
                time.sleep(60)

    # ── template execution ────────────────────────────────────────────────────────

    def _execute(
        self,
        template_name: str,
        params: Dict[str, Any],
        trades_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        if template_name not in self.TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. Available: {list(self.TEMPLATES)}"
            )

        if template_name == "market_state_report":
            from .templates.market_state_report import MarketStateReportTemplate
            t = MarketStateReportTemplate(**{k: v for k, v in params.items()
                                             if k in ("fred_api_key",)})
            return t.run().to_dict()

        if template_name == "signal_analysis":
            from .templates.signal_analysis import SignalAnalysisTemplate
            t = SignalAnalysisTemplate()
            df = trades_df if trades_df is not None else t.generate_synthetic_trades()
            signal_name = params.get("signal_name", "signal")
            forward_bar = params.get("forward_bar", 1)
            return t.run(df, signal_name=signal_name, forward_bar=forward_bar).to_dict()

        if template_name == "regime_analysis":
            from .templates.regime_analysis import RegimeAnalysisTemplate
            t = RegimeAnalysisTemplate()
            df = trades_df if trades_df is not None else t.generate_synthetic_trades()
            regime_name = params.get("regime_name", "regime")
            regime_column = params.get("regime_column", "is_regime")
            return t.run(df, regime_name=regime_name, regime_column=regime_column).to_dict()

        if template_name == "instrument_autopsy":
            from .templates.instrument_autopsy import InstrumentAutopsyTemplate
            t = InstrumentAutopsyTemplate()
            symbol = params.get("symbol", "UNKNOWN")
            df = trades_df if trades_df is not None else t.generate_synthetic_trades(symbol=symbol)
            return t.run(df, symbol=symbol).to_dict()

        if template_name == "idea_validation":
            from .templates.idea_validation import IdeaValidationTemplate
            t = IdeaValidationTemplate()
            hypothesis_id = params.get("hypothesis_id", "unnamed_hyp")
            df = trades_df if trades_df is not None else t.generate_synthetic_trades()
            # Default apply_fn: block hour 13
            filter_hour = params.get("filter_hour", 13)

            def apply_fn(d: pd.DataFrame) -> pd.DataFrame:
                if "entry_hour" in d.columns:
                    return d[d["entry_hour"] != filter_hour]
                return d

            return t.run(df, hypothesis_id=hypothesis_id, apply_fn=apply_fn).to_dict()

        raise ValueError(f"Unhandled template: {template_name}")

    # ── persistence ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _save_record(self, record: RunRecord) -> None:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO runs (template, params, result, status, error_msg, started_at, finished_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    record.template,
                    json.dumps(record.params, default=str),
                    json.dumps(record.result, default=str) if record.result else None,
                    record.status,
                    record.error_msg,
                    record.started_at.isoformat(),
                    record.finished_at.isoformat() if record.finished_at else None,
                ),
            )
            record.run_id = cur.lastrowid


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="SRFM Notebook Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--template", type=str, help="Template name to run")
    parser.add_argument(
        "--params", type=str, default="{}", help="JSON params for the template"
    )
    parser.add_argument(
        "--schedule", type=str, metavar="TEMPLATE",
        help="Schedule a template to run daily (use with --hour)"
    )
    parser.add_argument("--hour", type=int, default=6, help="UTC hour for scheduled run")
    parser.add_argument("--list", action="store_true", help="List recent runs")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="Path to SQLite DB")

    args = parser.parse_args()
    runner = NotebookRunner(db_path=Path(args.db))

    if args.list:
        runs = runner.list_runs(limit=20)
        for r in runs:
            print(f"[{r['run_id']}] {r['template']} | {r['status']} | {r['started_at']}")
        return

    if args.schedule:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            params = {}
        runner.schedule(args.schedule, utc_hour=args.hour, params=params)
        return

    if args.template:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            print(f"WARNING: Could not parse --params JSON: {args.params}", file=sys.stderr)
            params = {}
        record = runner.run(args.template, params=params)
        print(json.dumps({
            "status": record.status,
            "template": record.template,
            "result": record.result,
            "error": record.error_msg,
        }, indent=2, default=str))
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()
