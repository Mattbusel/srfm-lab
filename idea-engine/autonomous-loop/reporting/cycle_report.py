"""
idea-engine/autonomous-loop/reporting/cycle_report.py

CycleReporter: write per-cycle summaries in both plain text and JSON.

Each cycle produces two files in the reports/ directory:
  - cycle_{N:05d}_{timestamp}.json   (machine-readable, full detail)
  - cycle_{N:05d}_{timestamp}.txt    (human-readable summary)

Old reports are archived (kept indefinitely, compressed after 30 days).
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COMPRESS_AFTER_DAYS = 30


class CycleReporter:
    """
    Writes cycle summary reports to disk in plain text and JSON formats.

    Directory layout:
      reports/
        cycle_00001_20260405T120000Z.json
        cycle_00001_20260405T120000Z.txt
        archive/          ← compressed reports older than 30 days
    """

    def __init__(self, reports_dir: Path | str) -> None:
        self.reports_dir = Path(reports_dir)
        self.archive_dir = self.reports_dir / "archive"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, ctx: dict[str, Any]) -> Path:
        """
        Write cycle report for the given context dict.
        Returns the path to the JSON report file.
        """
        cycle_n = int(ctx.get("cycle_number", 0))
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"cycle_{cycle_n:05d}_{ts}"

        json_path = self.reports_dir / f"{stem}.json"
        txt_path = self.reports_dir / f"{stem}.txt"

        report_data = self._build_json_report(ctx)
        summary_text = self._build_text_summary(ctx, report_data)

        json_path.write_text(json.dumps(report_data, indent=2, default=str))
        txt_path.write_text(summary_text)

        logger.info("CycleReporter: wrote %s", json_path.name)

        self._maybe_compress_old()
        return json_path

    # ------------------------------------------------------------------
    # Report builders
    # ------------------------------------------------------------------

    def _build_json_report(self, ctx: dict[str, Any]) -> dict[str, Any]:
        signal = ctx.get("signal")
        signal_dict = signal.to_dict() if hasattr(signal, "to_dict") else {}

        return {
            "cycle_number": ctx.get("cycle_number"),
            "cycle_start": ctx.get("cycle_start"),
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "signals": signal_dict,
            "patterns": {
                "found": ctx.get("patterns_found", 0),
            },
            "hypotheses": {
                "generated": ctx.get("hypotheses_generated", 0),
                "approved": ctx.get("hypotheses_approved", 0),
                "validated": ctx.get("hypotheses_validated", 0),
            },
            "parameters": {
                "applied": ctx.get("parameters_applied", 0),
            },
            "live_health": ctx.get("live_health", {}),
            "drift_alerts": ctx.get("drift_alerts", []),
        }

    def _build_text_summary(
        self, ctx: dict[str, Any], report_data: dict[str, Any]
    ) -> str:
        cycle_n = ctx.get("cycle_number", 0)
        cycle_start = ctx.get("cycle_start", "unknown")
        signal = ctx.get("signal")

        lines = [
            "=" * 70,
            f"AUTONOMOUS LOOP — CYCLE #{cycle_n}",
            f"Started: {cycle_start}",
            "=" * 70,
            "",
        ]

        # Signals
        if signal:
            lines += [
                "SIGNALS",
                f"  Macro regime       : {signal.macro_regime}",
                f"  On-chain score     : {signal.onchain_score:+.3f}",
                f"  Sentiment score    : {signal.sentiment_score:+.3f}",
                f"  Derivatives signal : {signal.derivatives_signal}",
                f"  Liquidation risk   : {signal.liquidation_risk:.3f}",
                f"  Composite score    : {signal.composite_score:+.3f}",
            ]
            if signal.errors:
                lines.append(f"  Errors             : {list(signal.errors.keys())}")
            lines.append("")

        # Pattern mining
        lines += [
            "PATTERN MINING",
            f"  Patterns found     : {ctx.get('patterns_found', 0)}",
            "",
        ]

        # Hypothesis pipeline
        hyp = report_data.get("hypotheses", {})
        lines += [
            "HYPOTHESIS PIPELINE",
            f"  Generated          : {hyp.get('generated', 0)}",
            f"  Approved (debate)  : {hyp.get('approved', 0)}",
            f"  Validated (bt)     : {hyp.get('validated', 0)}",
            f"  Applied to live    : {ctx.get('parameters_applied', 0)}",
            "",
        ]

        # Live health
        live = ctx.get("live_health", {})
        if live:
            lines += [
                "LIVE PERFORMANCE",
                f"  Status             : {live.get('status', 'unknown')}",
                f"  Trades (sample)    : {live.get('n_trades', 0)}",
                f"  Live win rate      : {live.get('live_win_rate', 0):.1%}",
                f"  Expected win rate  : {live.get('expected_win_rate', 0):.1%}",
                f"  Z-score            : {live.get('win_rate_z_score', 0):+.2f}",
                f"  Avg PnL            : {live.get('avg_pnl', 0):+.4f}",
                f"  Losing streak      : {live.get('losing_streak', 0)}",
            ]
            if live.get("alerts"):
                for alert in live["alerts"]:
                    lines.append(f"  ⚠  {alert}")
            lines.append("")

        # Drift
        drift = ctx.get("drift_alerts", [])
        if drift:
            lines += ["DRIFT ALERTS"]
            for d in drift:
                lines.append(f"  ⚠  {d}")
            lines.append("")

        lines += [
            "=" * 70,
            f"Report generated: {datetime.now(timezone.utc).isoformat()}",
            "",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Archive management
    # ------------------------------------------------------------------

    def _maybe_compress_old(self) -> None:
        """Gzip-compress JSON + TXT reports older than _COMPRESS_AFTER_DAYS."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=_COMPRESS_AFTER_DAYS)
        for f in list(self.reports_dir.glob("cycle_*.json")) + list(
            self.reports_dir.glob("cycle_*.txt")
        ):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    gz_path = self.archive_dir / (f.name + ".gz")
                    with open(f, "rb") as src, gzip.open(gz_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    f.unlink()
                    logger.debug("CycleReporter: archived %s", f.name)
            except Exception as exc:
                logger.debug("CycleReporter: compress error for %s: %s", f.name, exc)

    def list_reports(self, last_n: int = 20) -> list[Path]:
        """Return the most recent N report JSON paths."""
        reports = sorted(self.reports_dir.glob("cycle_*.json"), reverse=True)
        return reports[:last_n]

    def read_report(self, path: Path) -> dict[str, Any]:
        """Parse and return a report JSON file."""
        return json.loads(path.read_text())
