"""
regime-oracle/alert_monitor.py
───────────────────────────────
Monitors for regime changes and fires alerts to the Narrative Intelligence layer.

The RegimeAlertMonitor polls the DB for new bars, classifies the current regime,
detects transitions, and writes structured alerts to the `narrative_alerts` table
(the same table used by the existing AlertWriter).

Monitored transitions and their actions
----------------------------------------
  BULL    → CRISIS   : "REDUCE RISK" — promote conservative genomes
  BULL    → BEAR     : "TREND REVERSAL" — reduce exposure
  BULL    → TOPPING  : "MOMENTUM DIVERGENCE" — hedge warning
  BEAR    → RECOVERY : "INCREASE EXPOSURE" — activate recovery genomes
  BEAR    → NEUTRAL  : "REGIME NORMALISING" — reduce short bias
  NEUTRAL → BULL     : "ACTIVATE MOMENTUM" — promote momentum genomes
  NEUTRAL → CRISIS   : "SUDDEN RISK-OFF" — immediate risk reduction
  CRISIS  → RECOVERY : "CRISIS ABATING" — begin re-exposure
  RECOVERY→ BULL     : "BULL CONFIRMED" — full momentum activation
  TOPPING → BEAR     : "TOP CONFIRMED" — activate defensive stance
  TOPPING → CRISIS   : "CRASH WARNING" — maximum risk reduction

Usage
-----
    monitor = RegimeAlertMonitor(db_path="idea_engine.db")
    actions = monitor.check_transition("BULL", "CRISIS")
    # => ["REDUCE RISK", "PROMOTE CONSERVATIVE GENOMES"]

    # Run in polling loop:
    monitor.watch(interval_seconds=300)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .classifier import RegimeOracle, RegimeState, Regime
from .feature_builder import RegimeFeatureBuilder

logger = logging.getLogger(__name__)

_HERE        = Path(__file__).resolve().parent
_ENGINE_ROOT = _HERE.parent
_DB_DEFAULT  = _ENGINE_ROOT / "idea_engine.db"

# Regime alert type constant (for narrative_alerts table)
ALERT_REGIME_TRANSITION = "REGIME_TRANSITION"


# ── Transition action rules ───────────────────────────────────────────────────

# Each entry: (from_regime, to_regime) → list of action strings
_TRANSITION_RULES: Dict[Tuple[str, str], List[str]] = {
    ("BULL",     "CRISIS"):   ["REDUCE RISK",          "PROMOTE CONSERVATIVE GENOMES",
                                "HALT NEW LONG ENTRIES"],
    ("BULL",     "BEAR"):     ["TREND REVERSAL",        "REDUCE LONG EXPOSURE",
                                "ACTIVATE BEAR GENOMES"],
    ("BULL",     "TOPPING"):  ["MOMENTUM DIVERGENCE",   "HEDGE LONG POSITIONS",
                                "MONITOR VOLUME DIVERGENCE"],
    ("BEAR",     "RECOVERY"): ["INCREASE EXPOSURE",     "ACTIVATE RECOVERY GENOMES",
                                "BEGIN SCALING IN"],
    ("BEAR",     "NEUTRAL"):  ["REGIME NORMALISING",    "REDUCE SHORT BIAS",
                                "REBALANCE TO NEUTRAL"],
    ("BEAR",     "CRISIS"):   ["ACCELERATING BEAR",     "MAXIMUM RISK REDUCTION",
                                "CLOSE ALL LONGS"],
    ("NEUTRAL",  "BULL"):     ["ACTIVATE MOMENTUM",     "PROMOTE MOMENTUM GENOMES",
                                "SCALE UP LONG EXPOSURE"],
    ("NEUTRAL",  "BEAR"):     ["DIRECTIONAL BREAK",     "REDUCE LONG EXPOSURE",
                                "MONITOR SUPPORT LEVELS"],
    ("NEUTRAL",  "CRISIS"):   ["SUDDEN RISK-OFF",       "IMMEDIATE RISK REDUCTION",
                                "HALT NEW ENTRIES"],
    ("CRISIS",   "RECOVERY"): ["CRISIS ABATING",        "BEGIN RE-EXPOSURE",
                                "PROMOTE RECOVERY GENOMES"],
    ("CRISIS",   "NEUTRAL"):  ["CRISIS FADING",         "RESTORE NEUTRAL POSTURE",
                                "GRADUAL RE-ENTRY"],
    ("RECOVERY", "BULL"):     ["BULL CONFIRMED",         "FULL MOMENTUM ACTIVATION",
                                "PROMOTE BULL GENOMES"],
    ("RECOVERY", "NEUTRAL"):  ["RECOVERY STALLING",     "HOLD NEUTRAL POSTURE",
                                "MONITOR BREADTH"],
    ("TOPPING",  "BEAR"):     ["TOP CONFIRMED",          "ACTIVATE DEFENSIVE STANCE",
                                "REDUCE LONG EXPOSURE"],
    ("TOPPING",  "CRISIS"):   ["CRASH WARNING",          "MAXIMUM RISK REDUCTION",
                                "EMERGENCY HEDGING"],
    ("TOPPING",  "NEUTRAL"):  ["TOPPING RESOLVED",       "RESET MOMENTUM BIAS",
                                "REBALANCE"],
}

# Severity mapping for transitions
_TRANSITION_SEVERITY: Dict[Tuple[str, str], str] = {
    ("BULL",     "CRISIS"):   "critical",
    ("BULL",     "BEAR"):     "high",
    ("BULL",     "TOPPING"):  "medium",
    ("BEAR",     "CRISIS"):   "critical",
    ("NEUTRAL",  "CRISIS"):   "critical",
    ("TOPPING",  "CRISIS"):   "critical",
    ("TOPPING",  "BEAR"):     "high",
    ("BEAR",     "RECOVERY"): "high",
    ("NEUTRAL",  "BULL"):     "medium",
    ("CRISIS",   "RECOVERY"): "medium",
    ("RECOVERY", "BULL"):     "medium",
    ("BEAR",     "NEUTRAL"):  "info",
    ("NEUTRAL",  "BEAR"):     "medium",
    ("CRISIS",   "NEUTRAL"):  "info",
    ("RECOVERY", "NEUTRAL"):  "info",
    ("TOPPING",  "NEUTRAL"):  "info",
}


# ── RegimeAlert ───────────────────────────────────────────────────────────────

@dataclass
class RegimeAlert:
    """
    A regime-change alert.

    Attributes
    ----------
    prev_regime  : regime before transition
    curr_regime  : regime after transition
    actions      : list of recommended action strings
    severity     : 'critical' | 'high' | 'medium' | 'info'
    message      : human-readable summary
    symbol       : instrument that triggered the alert
    ts           : ISO-8601 timestamp
    state        : full RegimeState at time of alert
    alert_id     : DB row id (set after storage)
    """
    prev_regime:  str
    curr_regime:  str
    actions:      List[str]
    severity:     str
    message:      str
    symbol:       str               = "BTC"
    ts:           str               = ""
    state:        Optional[RegimeState] = None
    alert_id:     Optional[int]     = None

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prev_regime":  self.prev_regime,
            "curr_regime":  self.curr_regime,
            "actions":      self.actions,
            "severity":     self.severity,
            "message":      self.message,
            "symbol":       self.symbol,
            "ts":           self.ts,
        }

    def __repr__(self) -> str:
        return (
            f"RegimeAlert({self.prev_regime!r}→{self.curr_regime!r}, "
            f"sev={self.severity!r}, actions={len(self.actions)})"
        )


# ── RegimeAlertMonitor ────────────────────────────────────────────────────────

class RegimeAlertMonitor:
    """
    Monitors for regime transitions and fires alerts to the Narrative layer.

    Parameters
    ----------
    db_path          : path to idea_engine.db
    oracle           : RegimeOracle instance (created if None)
    symbol           : primary instrument to monitor
    alert_cooldown_h : minimum hours between alerts of the same transition type
    """

    def __init__(
        self,
        db_path:          Path | str                  = _DB_DEFAULT,
        oracle:           Optional[RegimeOracle]      = None,
        symbol:           str                         = "BTC",
        alert_cooldown_h: int                         = 4,
    ) -> None:
        self.db_path          = Path(db_path)
        self.symbol           = symbol
        self.alert_cooldown_h = alert_cooldown_h
        self.oracle           = oracle or RegimeOracle(db_path=db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._prev_regime: Optional[str] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None or not self._alive():
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    def _alive(self) -> bool:
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _ensure_schema(self) -> None:
        """Ensure narrative_alerts table exists (may already exist from AlertWriter)."""
        conn = self._connect()
        try:
            conn.executescript("""
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
        except sqlite3.Error as exc:
            logger.warning("Schema ensure failed: %s", exc)

    # ------------------------------------------------------------------
    # Transition logic
    # ------------------------------------------------------------------

    def check_transition(
        self,
        prev_regime: str,
        curr_regime: str,
    ) -> List[str]:
        """
        Return the list of recommended actions for a regime transition.

        Parameters
        ----------
        prev_regime : previous regime label (e.g. 'BULL')
        curr_regime : new regime label (e.g. 'CRISIS')

        Returns
        -------
        List[str] — action strings, or empty list if transition is not significant
        """
        prev = prev_regime.upper()
        curr = curr_regime.upper()

        if prev == curr:
            return []

        key = (prev, curr)
        actions = _TRANSITION_RULES.get(key, [])

        if not actions:
            # Generic transition alert
            actions = [f"REGIME CHANGED: {prev} → {curr}", "REVIEW ACTIVE GENOMES"]

        return actions

    def process_transition(
        self,
        prev_regime: str,
        curr_regime: str,
        state:       Optional[RegimeState] = None,
    ) -> Optional[RegimeAlert]:
        """
        Process a detected regime transition: compute actions, build alert,
        store to DB if not recently alerted.

        Parameters
        ----------
        prev_regime : previous regime label
        curr_regime : new regime label
        state       : current RegimeState (optional, for metadata)

        Returns
        -------
        RegimeAlert or None (if same regime or on cooldown)
        """
        if prev_regime.upper() == curr_regime.upper():
            return None

        if self._on_cooldown(prev_regime, curr_regime):
            logger.debug(
                "Transition %s→%s on cooldown (%dh).",
                prev_regime, curr_regime, self.alert_cooldown_h,
            )
            return None

        actions  = self.check_transition(prev_regime, curr_regime)
        key      = (prev_regime.upper(), curr_regime.upper())
        severity = _TRANSITION_SEVERITY.get(key, "info")

        message = (
            f"Regime transition detected: {prev_regime.upper()} → {curr_regime.upper()}. "
            f"Actions: {', '.join(actions[:2])}{'…' if len(actions) > 2 else ''}."
        )

        alert = RegimeAlert(
            prev_regime = prev_regime.upper(),
            curr_regime = curr_regime.upper(),
            actions     = actions,
            severity    = severity,
            message     = message,
            symbol      = self.symbol,
            state       = state,
        )

        alert_id = self._store_alert(alert)
        alert.alert_id = alert_id

        logger.info(
            "Regime alert fired: %s→%s  severity=%s  actions=%s",
            prev_regime, curr_regime, severity, actions,
        )
        return alert

    # ------------------------------------------------------------------
    # Watch loop
    # ------------------------------------------------------------------

    def watch(
        self,
        interval_seconds:  int = 300,
        max_iterations:    Optional[int] = None,
        on_alert:          Optional[Callable[[RegimeAlert], None]] = None,
        load_ohlcv_fn:     Optional[Callable[[], pd.DataFrame]] = None,
    ) -> None:
        """
        Poll the DB for new bars and check for regime changes.

        Runs indefinitely (or up to max_iterations).
        Each iteration:
          1. Load latest OHLCV data via load_ohlcv_fn (or from DB).
          2. Classify current regime.
          3. Compare to previous regime.
          4. If transition detected, fire alert.
          5. Sleep for interval_seconds.

        Parameters
        ----------
        interval_seconds : sleep between iterations (default 300 = 5 min)
        max_iterations   : stop after N iterations (None = run forever)
        on_alert         : callback invoked when an alert is fired
        load_ohlcv_fn    : optional callable returning a fresh OHLCV DataFrame

        Returns (only when max_iterations is reached or interrupted)
        """
        logger.info(
            "RegimeAlertMonitor watching symbol=%s every %ds.",
            self.symbol, interval_seconds,
        )

        # Bootstrap previous regime from DB
        last_state = self.oracle.get_latest_regime(self.symbol)
        self._prev_regime = last_state.regime if last_state else "NEUTRAL"

        iteration = 0
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            try:
                # Load fresh data
                if load_ohlcv_fn is not None:
                    ohlcv_df = load_ohlcv_fn()
                else:
                    ohlcv_df = self._load_latest_ohlcv()

                if ohlcv_df is not None and len(ohlcv_df) >= 50:
                    # Build features for the latest bar
                    features = self.oracle.feature_builder.build_features(
                        ohlcv_df, symbol=self.symbol
                    )
                    state = self.oracle.classify(features, store=True)
                    curr_regime = state.regime

                    if self._prev_regime and curr_regime != self._prev_regime:
                        alert = self.process_transition(
                            self._prev_regime, curr_regime, state
                        )
                        if alert is not None:
                            if on_alert is not None:
                                try:
                                    on_alert(alert)
                                except Exception as exc:
                                    logger.warning("on_alert callback failed: %s", exc)
                            logger.info("Alert: %s", alert)

                    self._prev_regime = curr_regime
                    logger.debug("Regime: %s (conf=%.2f)", curr_regime, state.confidence)

                else:
                    logger.debug("No fresh OHLCV data available this iteration.")

            except Exception as exc:
                logger.error("Watch loop error: %s", exc)

            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                time.sleep(interval_seconds)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _store_alert(self, alert: RegimeAlert) -> Optional[int]:
        """
        Write a RegimeAlert to the narrative_alerts table.

        Returns the inserted row id, or None on failure.
        """
        conn = self._connect()
        now  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data: Dict[str, Any] = {
            "prev_regime":  alert.prev_regime,
            "curr_regime":  alert.curr_regime,
            "actions":      alert.actions,
            "symbol":       alert.symbol,
        }
        if alert.state:
            data["probabilities"] = {
                k: round(v, 3) for k, v in alert.state.probabilities.items()
            }

        try:
            cur = conn.execute(
                """
                INSERT INTO narrative_alerts
                    (alert_type, severity, message, data_json,
                     acknowledged, created_at)
                VALUES (?,?,?,?,0,?)
                """,
                (
                    ALERT_REGIME_TRANSITION,
                    alert.severity,
                    alert.message,
                    json.dumps(data),
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid
        except sqlite3.Error as exc:
            logger.warning("Failed to store regime alert: %s", exc)
            return None

    def _on_cooldown(self, prev_regime: str, curr_regime: str) -> bool:
        """
        Return True if a similar transition alert was filed within the cooldown window.
        """
        conn = self._connect()
        key_str = f"{prev_regime.upper()}_{curr_regime.upper()}"
        try:
            row = conn.execute(
                """
                SELECT 1 FROM narrative_alerts
                WHERE alert_type = ?
                  AND data_json LIKE ?
                  AND created_at >= datetime('now', ? || ' hours')
                LIMIT 1
                """,
                (
                    ALERT_REGIME_TRANSITION,
                    f"%{key_str}%",
                    f"-{self.alert_cooldown_h}",
                ),
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError:
            return False

    def _load_latest_ohlcv(self, n_bars: int = 2000) -> Optional[pd.DataFrame]:
        """
        Load recent OHLCV data from the DB for regime classification.

        Tries tables: ohlcv_bars, bars, price_bars.
        """
        conn = self._connect()
        for table in ("ohlcv_bars", "bars", "price_bars"):
            try:
                df = pd.read_sql(
                    f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {n_bars}",
                    conn,
                )
                if len(df) >= 50:
                    for tc in ("ts", "timestamp", "date", "datetime"):
                        if tc in df.columns:
                            df[tc] = pd.to_datetime(df[tc], errors="coerce")
                            df = df.set_index(tc).sort_index()
                            break
                    return df
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Recent alert queries
    # ------------------------------------------------------------------

    def recent_alerts(self, hours: int = 48, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return recent regime transition alerts from the DB.

        Parameters
        ----------
        hours : look-back window in hours
        limit : maximum rows

        Returns
        -------
        List of dicts
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM narrative_alerts
                WHERE alert_type = ?
                  AND created_at >= datetime('now', ? || ' hours')
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (ALERT_REGIME_TRANSITION, f"-{hours}", limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def unacknowledged_regime_alerts(self) -> List[Dict[str, Any]]:
        """
        Return all unacknowledged regime transition alerts.

        Returns
        -------
        List of dicts
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM narrative_alerts
                WHERE alert_type = ?
                  AND acknowledged = 0
                ORDER BY created_at DESC
                """,
                (ALERT_REGIME_TRANSITION,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def acknowledge(self, alert_id: int) -> bool:
        """
        Acknowledge a regime alert.

        Parameters
        ----------
        alert_id : DB row id in narrative_alerts

        Returns
        -------
        bool — True if row was updated
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE narrative_alerts SET acknowledged=1 WHERE id=?",
                (alert_id,),
            )
            conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error:
            return False

    def acknowledge_all_regime_alerts(self) -> int:
        """
        Acknowledge all unacknowledged regime transition alerts.

        Returns
        -------
        int — number of rows updated
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE narrative_alerts SET acknowledged=1 "
                "WHERE alert_type=? AND acknowledged=0",
                (ALERT_REGIME_TRANSITION,),
            )
            conn.commit()
            return cur.rowcount
        except sqlite3.Error:
            return 0

    def transition_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Summarise recent regime transitions.

        Parameters
        ----------
        days : look-back window in days

        Returns
        -------
        dict with keys: total_transitions, by_type, most_common, recent_severity
        """
        rows = self.recent_alerts(hours=days * 24, limit=1000)
        if not rows:
            return {"total_transitions": 0}

        by_type: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}

        for r in rows:
            data = {}
            try:
                data = json.loads(r.get("data_json") or "{}")
            except json.JSONDecodeError:
                pass

            key = f"{data.get('prev_regime', '?')}→{data.get('curr_regime', '?')}"
            by_type[key] = by_type.get(key, 0) + 1

            sev = r.get("severity", "info")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        most_common = max(by_type, key=lambda k: by_type[k]) if by_type else ""

        return {
            "total_transitions":  len(rows),
            "by_type":            by_type,
            "most_common":        most_common,
            "severity_counts":    severity_counts,
            "look_back_days":     days,
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        self.oracle.close()

    def __enter__(self) -> "RegimeAlertMonitor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"RegimeAlertMonitor(db={self.db_path.name!r}, "
            f"symbol={self.symbol!r}, cooldown={self.alert_cooldown_h}h)"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Regime Alert Monitor")
    parser.add_argument("--db",       default=str(_DB_DEFAULT))
    parser.add_argument("--symbol",   default="BTC")
    parser.add_argument("--interval", type=int, default=300,
                        help="Polling interval in seconds")
    parser.add_argument("--once",     action="store_true",
                        help="Run one iteration and exit")
    args = parser.parse_args()

    with RegimeAlertMonitor(db_path=args.db, symbol=args.symbol) as monitor:
        if args.once:
            monitor.watch(interval_seconds=args.interval, max_iterations=1)
        else:
            try:
                monitor.watch(interval_seconds=args.interval)
            except KeyboardInterrupt:
                logger.info("Watch loop interrupted.")
