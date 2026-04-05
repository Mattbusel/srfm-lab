"""
alert_writer.py — Alert Monitoring and Formatting
==================================================
Monitors the idea_engine.db database for notable events and generates
formatted alerts.  Four alert categories are monitored:

    1. PROMOTE_SHADOW      — Shadow beats live for 7+ consecutive days
    2. PARAMETER_UPDATE    — Counterfactual finds >20% improvement
    3. RESEARCH_ALERT      — New paper with relevance_score > 0.8
    4. EVOLUTION_BREAKTHROUGH — Genome fitness improves >15% in one generation

Alerts are stored in the ``narrative_alerts`` table and can be
acknowledged via :meth:`acknowledge`.

Usage
-----
    writer = AlertWriter(db_path="idea_engine.db")
    alerts = writer.check_alerts()
    for alert in alerts:
        print(writer.format_alert(alert))

    # Acknowledge
    writer.acknowledge(alert.alert_id)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert type constants
# ---------------------------------------------------------------------------

ALERT_PROMOTE_SHADOW        = "PROMOTE_SHADOW"
ALERT_PARAMETER_UPDATE      = "PARAMETER_UPDATE"
ALERT_RESEARCH_ALERT        = "RESEARCH_ALERT"
ALERT_EVOLUTION_BREAKTHROUGH = "EVOLUTION_BREAKTHROUGH"

# Thresholds
SHADOW_CONSECUTIVE_DAYS_THRESHOLD: int   = 7
COUNTERFACTUAL_IMPROVEMENT_THRESHOLD: float = 0.20   # 20%
PAPER_RELEVANCE_THRESHOLD:           float = 0.80
GENOME_FITNESS_JUMP_THRESHOLD:       float = 0.15    # 15%

# Severity mapping
ALERT_SEVERITY: Dict[str, str] = {
    ALERT_PROMOTE_SHADOW:         "high",
    ALERT_PARAMETER_UPDATE:       "high",
    ALERT_RESEARCH_ALERT:         "info",
    ALERT_EVOLUTION_BREAKTHROUGH: "medium",
}

# Terminal colours (ANSI)
_COLOUR: Dict[str, str] = {
    "high":   "\033[91m",   # red
    "medium": "\033[93m",   # yellow
    "info":   "\033[94m",   # blue
    "reset":  "\033[0m",
    "bold":   "\033[1m",
}

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """
    A single formatted alert.

    Attributes
    ----------
    alert_type : str
        One of the ALERT_* constants.
    severity : str
        'high', 'medium', or 'info'.
    message : str
        Human-readable alert message.
    data : dict
        Supporting data for the alert.
    created_at : str
        ISO-8601 timestamp.
    acknowledged : bool
        Whether the alert has been acknowledged.
    alert_id : int or None
        DB row id (set after storage).
    """

    alert_type:   str
    severity:     str
    message:      str
    data:         dict          = field(default_factory=dict)
    created_at:   str           = ""
    acknowledged: bool          = False
    alert_id:     Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("alert_id", None)
        return d

    def __repr__(self) -> str:
        return (
            f"Alert(type={self.alert_type!r}, "
            f"severity={self.severity!r}, msg={self.message[:60]!r})"
        )


# ---------------------------------------------------------------------------
# AlertWriter
# ---------------------------------------------------------------------------

class AlertWriter:
    """
    Monitors the idea_engine.db for notable events and generates alerts.

    Parameters
    ----------
    db_path : str
        SQLite database path.
    use_colour : bool
        If True, ANSI colour codes are added to terminal output.
    """

    def __init__(
        self,
        db_path: str = "idea_engine.db",
        use_colour: bool = True,
    ) -> None:
        self.db_path    = db_path
        self.use_colour = use_colour
        self._db: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS narrative_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT    NOT NULL,
                subject_id  TEXT,
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
            CREATE TABLE IF NOT EXISTS narrative_alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type   TEXT    NOT NULL,
                severity     TEXT    NOT NULL DEFAULT 'info',
                message      TEXT    NOT NULL,
                data_json    TEXT,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        conn.commit()

    def _table_exists(self, table: str) -> bool:
        """Return True if *table* exists in the database."""
        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check_alerts(self) -> List[Alert]:
        """
        Scan the database for all notable events and return new alerts.

        Runs all four monitor functions. Only generates alerts for events
        that haven't been alerted about in the last 24 hours.

        Returns
        -------
        List[Alert]
        """
        alerts: List[Alert] = []
        checkers = [
            self._check_shadow_promotion,
            self._check_counterfactual_improvement,
            self._check_research_papers,
            self._check_evolution_breakthrough,
        ]
        for checker in checkers:
            try:
                new_alerts = checker()
                alerts.extend(new_alerts)
            except Exception as exc:
                logger.warning("Alert checker %s failed: %s", checker.__name__, exc)

        if alerts:
            self._store_alerts(alerts)
            logger.info("Generated %d new alerts.", len(alerts))
        return alerts

    # ------------------------------------------------------------------
    # Monitor: shadow promotion
    # ------------------------------------------------------------------

    def _check_shadow_promotion(self) -> List[Alert]:
        """
        Check if any shadow strategy has beaten live for 7+ consecutive days.

        Queries ``shadow_runs`` if it exists; otherwise returns empty.
        """
        if not self._table_exists("shadow_runs"):
            return []

        conn = self._connect()
        alerts: List[Alert] = []

        # Find shadow strategies beating live for N consecutive days
        try:
            rows = conn.execute(
                """
                SELECT shadow_id,
                       COUNT(*) AS consecutive_days,
                       AVG(shadow_pnl - live_pnl) AS avg_delta,
                       MAX(run_date) AS latest_date
                FROM shadow_runs
                WHERE shadow_pnl > live_pnl
                  AND run_date >= date('now', '-14 days')
                GROUP BY shadow_id
                HAVING consecutive_days >= ?
                """,
                (SHADOW_CONSECUTIVE_DAYS_THRESHOLD,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        for row in rows:
            if self._already_alerted(ALERT_PROMOTE_SHADOW, str(row["shadow_id"]), hours=24):
                continue

            avg_delta = row["avg_delta"] or 0.0
            message = (
                f"Shadow strategy '{row['shadow_id']}' has beaten live for "
                f"{row['consecutive_days']} consecutive days "
                f"(avg PnL delta: +{avg_delta:.4f}). "
                f"Consider promoting to live."
            )
            alerts.append(Alert(
                alert_type = ALERT_PROMOTE_SHADOW,
                severity   = ALERT_SEVERITY[ALERT_PROMOTE_SHADOW],
                message    = message,
                data       = {
                    "shadow_id":         row["shadow_id"],
                    "consecutive_days":  row["consecutive_days"],
                    "avg_delta":         round(avg_delta, 6),
                    "latest_date":       row["latest_date"],
                    "threshold":         SHADOW_CONSECUTIVE_DAYS_THRESHOLD,
                },
            ))
        return alerts

    # ------------------------------------------------------------------
    # Monitor: counterfactual improvement
    # ------------------------------------------------------------------

    def _check_counterfactual_improvement(self) -> List[Alert]:
        """
        Check if any counterfactual analysis found >20% improvement.

        Queries ``counterfactual_results`` if it exists.
        """
        if not self._table_exists("counterfactual_results"):
            return []

        conn = self._connect()
        alerts: List[Alert] = []

        try:
            rows = conn.execute(
                """
                SELECT id, parameter, original_value, cf_value,
                       improvement_pct, created_at, summary
                FROM counterfactual_results
                WHERE improvement_pct > ?
                  AND created_at >= datetime('now', '-7 days')
                ORDER BY improvement_pct DESC
                LIMIT 10
                """,
                (COUNTERFACTUAL_IMPROVEMENT_THRESHOLD * 100,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        for row in rows:
            if self._already_alerted(ALERT_PARAMETER_UPDATE, str(row["id"]), hours=48):
                continue

            pct = row["improvement_pct"]
            message = (
                f"Counterfactual #{row['id']} found {pct:.1f}% improvement "
                f"by changing '{row['parameter']}' from {row['original_value']} "
                f"to {row['cf_value']}. "
                f"Consider updating live parameters."
            )
            alerts.append(Alert(
                alert_type = ALERT_PARAMETER_UPDATE,
                severity   = ALERT_SEVERITY[ALERT_PARAMETER_UPDATE],
                message    = message,
                data       = {
                    "counterfactual_id": row["id"],
                    "parameter":         row["parameter"],
                    "original_value":    row["original_value"],
                    "cf_value":          row["cf_value"],
                    "improvement_pct":   round(pct, 2),
                    "threshold_pct":     COUNTERFACTUAL_IMPROVEMENT_THRESHOLD * 100,
                    "summary":           row["summary"] or "",
                },
            ))
        return alerts

    # ------------------------------------------------------------------
    # Monitor: research papers
    # ------------------------------------------------------------------

    def _check_research_papers(self) -> List[Alert]:
        """
        Check if any newly mined paper has relevance_score > 0.8.

        Queries ``academic_papers``.
        """
        if not self._table_exists("academic_papers"):
            return []

        conn = self._connect()
        alerts: List[Alert] = []

        try:
            rows = conn.execute(
                """
                SELECT id, source, paper_id, title, relevance_score, url, mined_at
                FROM academic_papers
                WHERE relevance_score >= ?
                  AND mined_at >= datetime('now', '-7 days')
                ORDER BY relevance_score DESC
                LIMIT 20
                """,
                (PAPER_RELEVANCE_THRESHOLD,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        for row in rows:
            if self._already_alerted(ALERT_RESEARCH_ALERT, str(row["paper_id"]), hours=168):
                continue

            score = row["relevance_score"]
            message = (
                f"High-relevance paper found on {row['source'].upper()}: "
                f"\"{row['title'][:80]}\" "
                f"(score={score:.3f}). Check idea candidates."
            )
            alerts.append(Alert(
                alert_type = ALERT_RESEARCH_ALERT,
                severity   = ALERT_SEVERITY[ALERT_RESEARCH_ALERT],
                message    = message,
                data       = {
                    "paper_id":        row["paper_id"],
                    "source":          row["source"],
                    "title":           row["title"],
                    "relevance_score": round(score, 4),
                    "url":             row["url"] or "",
                    "mined_at":        row["mined_at"],
                },
            ))
        return alerts

    # ------------------------------------------------------------------
    # Monitor: evolution breakthrough
    # ------------------------------------------------------------------

    def _check_evolution_breakthrough(self) -> List[Alert]:
        """
        Check if genome fitness improved >15% in a single generation.

        Queries ``genomes`` if it exists.
        """
        if not self._table_exists("genomes"):
            return []

        conn = self._connect()
        alerts: List[Alert] = []

        try:
            rows = conn.execute(
                """
                SELECT g.id, g.generation, g.fitness,
                       p.fitness AS parent_fitness,
                       g.parent_id
                FROM genomes g
                JOIN genomes p ON p.id = g.parent_id
                WHERE g.fitness > p.fitness * (1 + ?)
                  AND g.created_at >= datetime('now', '-7 days')
                ORDER BY (g.fitness - p.fitness) / p.fitness DESC
                LIMIT 5
                """,
                (GENOME_FITNESS_JUMP_THRESHOLD,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        for row in rows:
            if self._already_alerted(ALERT_EVOLUTION_BREAKTHROUGH, str(row["id"]), hours=24):
                continue

            improvement = (row["fitness"] - row["parent_fitness"]) / max(row["parent_fitness"], 1e-9)
            message = (
                f"Genome #{row['id']} achieved a {improvement*100:.1f}% fitness improvement "
                f"in generation {row['generation']} "
                f"({row['parent_fitness']:.4f} → {row['fitness']:.4f}). "
                f"Inspect mutations applied."
            )
            alerts.append(Alert(
                alert_type = ALERT_EVOLUTION_BREAKTHROUGH,
                severity   = ALERT_SEVERITY[ALERT_EVOLUTION_BREAKTHROUGH],
                message    = message,
                data       = {
                    "genome_id":       row["id"],
                    "generation":      row["generation"],
                    "fitness":         round(row["fitness"], 6),
                    "parent_fitness":  round(row["parent_fitness"], 6),
                    "improvement_pct": round(improvement * 100, 2),
                    "threshold_pct":   GENOME_FITNESS_JUMP_THRESHOLD * 100,
                },
            ))
        return alerts

    # ------------------------------------------------------------------
    # Deduplication guard
    # ------------------------------------------------------------------

    def _already_alerted(
        self,
        alert_type: str,
        subject_key: str,
        hours: int = 24,
    ) -> bool:
        """
        Return True if an alert of *alert_type* about *subject_key* was
        stored within the last *hours* hours.

        Uses a JSON search in data_json for the subject key.
        """
        conn = self._connect()
        row = conn.execute(
            """
            SELECT 1 FROM narrative_alerts
            WHERE alert_type = ?
              AND data_json LIKE ?
              AND created_at >= datetime('now', ? || ' hours')
            LIMIT 1
            """,
            (alert_type, f"%{subject_key}%", f"-{hours}"),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _store_alerts(self, alerts: List[Alert]) -> int:
        """Store a list of alerts and set their alert_id."""
        conn = self._connect()
        inserted = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for alert in alerts:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO narrative_alerts
                        (alert_type, severity, message, data_json,
                         acknowledged, created_at)
                    VALUES (?, ?, ?, ?, 0, ?)
                    """,
                    (
                        alert.alert_type,
                        alert.severity,
                        alert.message,
                        json.dumps(alert.data),
                        now,
                    ),
                )
                alert.alert_id  = cur.lastrowid
                alert.created_at = now
                inserted += 1
            except sqlite3.Error as exc:
                logger.warning("Failed to store alert: %s", exc)
        conn.commit()
        return inserted

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_alert(self, alert: Alert, colour: Optional[bool] = None) -> str:
        """
        Format an Alert for terminal/log output.

        Parameters
        ----------
        alert : Alert
        colour : bool or None
            Override colour setting. None uses self.use_colour.

        Returns
        -------
        str
        """
        use_colour = self.use_colour if colour is None else colour
        severity   = alert.severity or "info"
        sev_colour = _COLOUR.get(severity, "") if use_colour else ""
        reset      = _COLOUR["reset"] if use_colour else ""
        bold       = _COLOUR["bold"]  if use_colour else ""

        lines = [
            f"{bold}{sev_colour}{'='*70}{reset}",
            f"{bold}{sev_colour}[{alert.alert_type}]{reset}  "
            f"severity={severity.upper()}  "
            f"id={alert.alert_id or 'unsaved'}",
            f"  {alert.message}",
        ]
        if alert.data:
            lines.append("  Details:")
            for k, v in alert.data.items():
                lines.append(f"    {k}: {v}")
        if alert.created_at:
            lines.append(f"  Time: {alert.created_at}")
        lines.append(f"{sev_colour}{'='*70}{reset}")
        return "\n".join(lines)

    def format_summary(self, alerts: List[Alert]) -> str:
        """
        Format a compact multi-alert summary for terminal output.

        Parameters
        ----------
        alerts : List[Alert]

        Returns
        -------
        str
        """
        if not alerts:
            return "No new alerts."

        lines = [
            f"{'─'*70}",
            f"  IAE ALERT SUMMARY — {len(alerts)} alert(s)",
            f"{'─'*70}",
        ]
        counts: Dict[str, int] = {}
        for a in alerts:
            counts[a.alert_type] = counts.get(a.alert_type, 0) + 1
            severity_tag = f"[{a.severity.upper():6s}]"
            lines.append(f"  {severity_tag} {a.alert_type:<30s}  {a.message[:50]}")
        lines.append(f"{'─'*70}")
        lines.append(f"  Types: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Acknowledge
    # ------------------------------------------------------------------

    def acknowledge(self, alert_id: int) -> bool:
        """
        Mark an alert as acknowledged.

        Parameters
        ----------
        alert_id : int

        Returns
        -------
        bool
            True if a row was updated.
        """
        conn = self._connect()
        cur = conn.execute(
            "UPDATE narrative_alerts SET acknowledged=1 WHERE id=?",
            (alert_id,),
        )
        conn.commit()
        updated = cur.rowcount > 0
        if updated:
            logger.info("Alert %d acknowledged.", alert_id)
        return updated

    def acknowledge_all(self, alert_type: Optional[str] = None) -> int:
        """
        Acknowledge all (or all of a specific type) unacknowledged alerts.

        Parameters
        ----------
        alert_type : str or None

        Returns
        -------
        int
            Number of rows updated.
        """
        conn = self._connect()
        if alert_type:
            cur = conn.execute(
                "UPDATE narrative_alerts SET acknowledged=1 WHERE acknowledged=0 AND alert_type=?",
                (alert_type,),
            )
        else:
            cur = conn.execute(
                "UPDATE narrative_alerts SET acknowledged=1 WHERE acknowledged=0"
            )
        conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def unacknowledged(self) -> List[dict]:
        """Return all unacknowledged alerts."""
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT * FROM narrative_alerts
            WHERE acknowledged=0
            ORDER BY created_at DESC
            """
        )
        return [dict(r) for r in cur]

    def recent_alerts(self, hours: int = 48, limit: int = 50) -> List[dict]:
        """
        Return alerts from the last *hours* hours.

        Parameters
        ----------
        hours : int
        limit : int

        Returns
        -------
        List[dict]
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT * FROM narrative_alerts
            WHERE created_at >= datetime('now', ? || ' hours')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (f"-{hours}", limit),
        )
        return [dict(r) for r in cur]

    def alert_history_by_type(self, alert_type: str, limit: int = 100) -> List[dict]:
        """
        Return alert history for a specific type.

        Parameters
        ----------
        alert_type : str
        limit : int

        Returns
        -------
        List[dict]
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT * FROM narrative_alerts
            WHERE alert_type=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (alert_type, limit),
        )
        return [dict(r) for r in cur]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    def __repr__(self) -> str:
        return f"AlertWriter(db={self.db_path!r})"

    def __enter__(self) -> "AlertWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    db_path = sys.argv[1] if len(sys.argv) > 1 else "idea_engine.db"
    with AlertWriter(db_path=db_path) as writer:
        alerts = writer.check_alerts()
        if alerts:
            print(writer.format_summary(alerts))
            print()
            for a in alerts:
                print(writer.format_alert(a))
        else:
            print("No new alerts.")

        unacked = writer.unacknowledged()
        if unacked:
            print(f"\n{len(unacked)} unacknowledged alert(s) in history.")
