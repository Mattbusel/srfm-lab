"""
live-feedback/live_monitor.py
=============================
Core live feedback monitor.

Reads new trades from ``live_trades.db`` on a configurable polling interval,
attributes each trade to the hypothesis that caused the position, computes
live performance metrics per hypothesis, and feeds results back to the
Bayesian scorer stored in ``idea_engine.db``.

Design notes
------------
* Both database connections are opened once in ``__init__`` and kept alive for
  the lifetime of the monitor.  WAL mode is enabled on both.
* ``poll()`` is the entry-point for the background thread / process.  It is
  deliberately simple so callers can wrap it in asyncio, threading, or a
  subprocess as needed.
* All timestamps in the database are UTC ISO-8601 strings.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .attribution import TradeAttributor
from .drift_detector import DriftDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV = "IDEA_ENGINE_DB"
DEFAULT_IAE_DB = Path(__file__).resolve().parents[1] / "idea_engine.db"
DEFAULT_LIVE_DB = Path(__file__).resolve().parents[1] / "live_trades.db"

ANNUALISATION_FACTOR = math.sqrt(252)
DEGRADATION_THRESHOLD = 0.30   # live Sharpe < 30 % of backtest Sharpe → degraded
MIN_LIVE_TRADES = 5            # need at least this many trades to score


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LiveScore:
    """
    Result of scoring a single hypothesis against live trading results.

    All monetary P&L figures are in the quote currency of the instrument
    (typically USD).
    """

    hypothesis_id: int
    window_days: int
    live_sharpe: float
    live_win_rate: float
    live_avg_pnl: float
    live_trade_count: int
    backtest_sharpe: float
    degradation_ratio: float       # live_sharpe / backtest_sharpe
    is_degraded: bool
    scored_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["is_degraded"] = int(d["is_degraded"])
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _open_db(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and row_factory set."""
    uri = f"file:{path}{'?mode=ro' if read_only else ''}".replace("\\", "/")
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_live_feedback_tables(conn: sqlite3.Connection) -> None:
    """Create live_feedback tables in idea_engine.db if they don't exist yet."""
    sql_path = Path(__file__).parent / "schema_extension.sql"
    if sql_path.exists():
        conn.executescript(sql_path.read_text(encoding="utf-8"))
        conn.commit()
    else:
        # Inline DDL as a fallback so the module is self-contained
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS live_hypothesis_scores (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id     INTEGER NOT NULL,
                window_days       INTEGER NOT NULL DEFAULT 30,
                live_sharpe       REAL,
                live_win_rate     REAL,
                live_avg_pnl      REAL,
                live_trade_count  INTEGER,
                backtest_sharpe   REAL,
                degradation_ratio REAL,
                is_degraded       INTEGER NOT NULL DEFAULT 0,
                scored_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
            CREATE TABLE IF NOT EXISTS trade_attributions (
                trade_id              TEXT NOT NULL,
                hypothesis_id         INTEGER,
                signal_name           TEXT,
                regime                TEXT,
                attribution_confidence REAL,
                created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                PRIMARY KEY (trade_id)
            );
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ts             TEXT NOT NULL,
                equity         REAL NOT NULL,
                running_sharpe REAL,
                running_dd     REAL,
                win_rate       REAL,
                calmar         REAL,
                expectancy     REAL,
                created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
            """
        )
        conn.commit()


def _sharpe_from_returns(returns: pd.Series, annualise: bool = True) -> float:
    """Compute Sharpe ratio from a series of per-trade returns."""
    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0
    sr = returns.mean() / returns.std(ddof=1)
    if annualise:
        sr *= ANNUALISATION_FACTOR
    return float(sr)


# ---------------------------------------------------------------------------
# LiveFeedbackMonitor
# ---------------------------------------------------------------------------

class LiveFeedbackMonitor:
    """
    Polls ``live_trades.db`` for new trades, attributes them to hypotheses,
    scores each active hypothesis against its live performance, and updates
    the Bayesian prior in ``idea_engine.db``.

    Parameters
    ----------
    live_db_path : str | Path
        Path to the live trading database (read-only).
    iae_db_path  : str | Path
        Path to ``idea_engine.db`` (read-write).
    window_days  : int
        Rolling window used when computing live metrics (default: 30).
    """

    def __init__(
        self,
        live_db_path: str | Path = DEFAULT_LIVE_DB,
        iae_db_path: str | Path = DEFAULT_IAE_DB,
        *,
        window_days: int = 30,
    ) -> None:
        self.live_db_path = Path(live_db_path)
        self.iae_db_path = Path(iae_db_path)
        self.window_days = window_days

        self._live_conn: sqlite3.Connection = _open_db(self.live_db_path)
        self._iae_conn: sqlite3.Connection = _open_db(self.iae_db_path)

        _ensure_live_feedback_tables(self._iae_conn)

        self._attributor = TradeAttributor(self._iae_conn)
        self._drift_detector = DriftDetector(self._iae_conn)

        # Watermark: ISO timestamp of the last trade we processed
        self._last_checked: str = self._load_watermark()

        logger.info(
            "LiveFeedbackMonitor initialised. last_checked=%s, window_days=%d",
            self._last_checked,
            self.window_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll(self, interval_seconds: float = 60.0) -> None:
        """
        Main polling loop.  Runs indefinitely; interrupt with KeyboardInterrupt
        or by setting ``self._running = False`` from another thread.

        Parameters
        ----------
        interval_seconds : float
            Seconds to sleep between each poll cycle.
        """
        self._running = True
        logger.info("Polling loop started (interval=%.0fs).", interval_seconds)
        while self._running:
            try:
                self._poll_once()
            except Exception:
                logger.exception("Error in poll cycle — will retry next interval.")
            time.sleep(interval_seconds)

    def stop(self) -> None:
        """Signal the poll loop to stop after the current cycle."""
        self._running = False

    def close(self) -> None:
        """Close both database connections cleanly."""
        self._live_conn.close()
        self._iae_conn.close()

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def _poll_once(self) -> None:
        """Execute one complete poll cycle."""
        new_trades_df = self._read_new_trades()
        if new_trades_df.empty:
            logger.debug("No new trades since %s.", self._last_checked)
            return

        logger.info("Found %d new trade(s) since %s.", len(new_trades_df), self._last_checked)

        # Attribute trades → hypotheses
        attribution_map = self.attribute_trades_to_hypotheses(new_trades_df)

        # Persist attributions
        self._attributor.persist_attributions(attribution_map, new_trades_df)

        # Identify which hypothesis IDs are affected
        affected_ids: set[int] = {
            hid for hid in attribution_map.values() if hid is not None
        }
        logger.debug("Affected hypothesis IDs: %s", affected_ids)

        # Score each affected hypothesis against live results
        for hid in affected_ids:
            try:
                score = self.score_hypothesis_live(hid)
                self._persist_live_score(score)
                self.update_bayesian_prior(hid, score)

                if self.detect_degradation(hid):
                    logger.warning(
                        "Hypothesis %d is degraded (live/bt ratio=%.2f). Triggering retest.",
                        hid,
                        score.degradation_ratio,
                    )
                    self.trigger_retest(hid)
            except Exception:
                logger.exception("Failed to score hypothesis %d.", hid)

        # Advance watermark
        latest_ts = new_trades_df["closed_at"].max()
        self._save_watermark(latest_ts)
        self._last_checked = latest_ts

    # ------------------------------------------------------------------
    # Trade reading
    # ------------------------------------------------------------------

    def _read_new_trades(self) -> pd.DataFrame:
        """
        Read all trades from ``live_trades.db`` that closed after the
        current watermark.

        Expected schema for the ``trades`` table in live_trades.db::

            trade_id TEXT PRIMARY KEY,
            symbol   TEXT,
            side     TEXT,           -- 'long' | 'short'
            pnl      REAL,
            pnl_pct  REAL,
            entry_price REAL,
            exit_price  REAL,
            opened_at   TEXT,        -- ISO-8601 UTC
            closed_at   TEXT,        -- ISO-8601 UTC
            params_json TEXT         -- JSON of strategy params used

        Returns an empty DataFrame if there are no new trades or if the
        ``trades`` table does not exist yet.
        """
        try:
            df = pd.read_sql_query(
                """
                SELECT *
                FROM trades
                WHERE closed_at > ?
                ORDER BY closed_at ASC
                """,
                self._live_conn,
                params=(self._last_checked,),
            )
        except Exception:
            logger.debug("live_trades.db has no 'trades' table yet — skipping.")
            df = pd.DataFrame()
        return df

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------

    def attribute_trades_to_hypotheses(
        self,
        trades_df: pd.DataFrame,
    ) -> dict[str, int | None]:
        """
        Match each live trade to the hypothesis that caused the position.

        Attribution priority:
        1. Exact symbol + side within hypothesis active window.
        2. Parameter overlap: trade params match hypothesis-modified params.
        3. Temporal proximity: trade within N bars of hypothesis adoption.

        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame of live trades (columns: trade_id, symbol, side,
            closed_at, params_json, …).

        Returns
        -------
        dict mapping trade_id → hypothesis_id (or None if unattributed).
        """
        hypotheses_df = self._load_active_hypotheses()
        return self._attributor.build_attribution_map(hypotheses_df, trades_df)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_hypothesis_live_pnl(
        self,
        hypothesis_id: int,
        window_days: int = 30,
    ) -> dict[str, Any]:
        """
        Compute live P&L metrics for a hypothesis over the given window.

        Parameters
        ----------
        hypothesis_id : int
        window_days   : int
            Lookback in calendar days.

        Returns
        -------
        dict with keys: live_sharpe, live_win_rate, live_avg_pnl,
        live_trade_count, backtest_sharpe.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Fetch attributed trades for this hypothesis inside the window
        try:
            pnl_df = pd.read_sql_query(
                """
                SELECT t.pnl, t.pnl_pct, t.closed_at
                FROM trades t
                JOIN trade_attributions ta ON ta.trade_id = t.trade_id
                WHERE ta.hypothesis_id = ?
                  AND t.closed_at >= ?
                ORDER BY t.closed_at ASC
                """,
                self._iae_conn,
                params=(hypothesis_id, cutoff),
            )
        except Exception:
            pnl_df = pd.DataFrame(columns=["pnl", "pnl_pct", "closed_at"])

        if pnl_df.empty or len(pnl_df) < MIN_LIVE_TRADES:
            return {
                "live_sharpe": 0.0,
                "live_win_rate": 0.0,
                "live_avg_pnl": 0.0,
                "live_trade_count": len(pnl_df),
                "backtest_sharpe": self._fetch_backtest_sharpe(hypothesis_id),
            }

        returns = pnl_df["pnl_pct"].astype(float)
        live_sharpe = _sharpe_from_returns(returns)
        live_win_rate = float((returns > 0).mean())
        live_avg_pnl = float(pnl_df["pnl"].astype(float).mean())
        backtest_sharpe = self._fetch_backtest_sharpe(hypothesis_id)

        return {
            "live_sharpe": live_sharpe,
            "live_win_rate": live_win_rate,
            "live_avg_pnl": live_avg_pnl,
            "live_trade_count": len(pnl_df),
            "backtest_sharpe": backtest_sharpe,
        }

    def score_hypothesis_live(self, hypothesis_id: int) -> LiveScore:
        """
        Produce a ``LiveScore`` dataclass for the given hypothesis.

        Parameters
        ----------
        hypothesis_id : int

        Returns
        -------
        LiveScore
        """
        metrics = self.compute_hypothesis_live_pnl(hypothesis_id, self.window_days)
        bt_sharpe = metrics["backtest_sharpe"]
        live_sharpe = metrics["live_sharpe"]

        if bt_sharpe != 0:
            degradation_ratio = live_sharpe / bt_sharpe
        else:
            degradation_ratio = 1.0 if live_sharpe >= 0 else 0.0

        is_degraded = self._compute_degradation_flag(live_sharpe, bt_sharpe)

        return LiveScore(
            hypothesis_id=hypothesis_id,
            window_days=self.window_days,
            live_sharpe=live_sharpe,
            live_win_rate=metrics["live_win_rate"],
            live_avg_pnl=metrics["live_avg_pnl"],
            live_trade_count=metrics["live_trade_count"],
            backtest_sharpe=bt_sharpe,
            degradation_ratio=degradation_ratio,
            is_degraded=is_degraded,
        )

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------

    def update_bayesian_prior(
        self,
        hypothesis_id: int,
        live_score: LiveScore,
    ) -> None:
        """
        Update the Beta prior for the hypothesis type based on live outcome.

        A live Sharpe > 0 counts as a success; <= 0 counts as a failure.
        The update is applied to the ``hypotheses`` table's
        ``bayesian_alpha`` and ``bayesian_beta`` columns if they exist,
        otherwise it is written to the ``event_log`` for downstream
        consumption.

        Parameters
        ----------
        hypothesis_id : int
        live_score    : LiveScore
        """
        success = 1 if live_score.live_sharpe > 0 else 0
        failure = 1 - success

        # Try to update bayesian columns directly if they exist
        try:
            self._iae_conn.execute(
                """
                UPDATE hypotheses
                SET bayesian_alpha = COALESCE(bayesian_alpha, 2.0) + ?,
                    bayesian_beta  = COALESCE(bayesian_beta,  2.0) + ?
                WHERE id = ?
                """,
                (success, failure, hypothesis_id),
            )
            self._iae_conn.commit()
            logger.debug(
                "Updated Beta prior for hypothesis %d: +%d success, +%d failure.",
                hypothesis_id,
                success,
                failure,
            )
        except sqlite3.OperationalError:
            # Columns don't exist — fall back to event_log
            payload = json.dumps(
                {
                    "hypothesis_id": hypothesis_id,
                    "bayesian_update": {"successes": success, "failures": failure},
                    "live_sharpe": live_score.live_sharpe,
                    "scored_at": live_score.scored_at,
                }
            )
            try:
                self._iae_conn.execute(
                    "INSERT INTO event_log (event_type, payload_json) VALUES (?, ?)",
                    ("bayesian_prior_update", payload),
                )
                self._iae_conn.commit()
            except Exception:
                logger.warning(
                    "Could not persist Bayesian update for hypothesis %d.", hypothesis_id
                )

    # ------------------------------------------------------------------
    # Degradation detection
    # ------------------------------------------------------------------

    def detect_degradation(self, hypothesis_id: int) -> bool:
        """
        Return True if the latest live score shows degradation.

        Degradation: live Sharpe < DEGRADATION_THRESHOLD × backtest Sharpe.

        Parameters
        ----------
        hypothesis_id : int
        """
        row = self._iae_conn.execute(
            """
            SELECT live_sharpe, backtest_sharpe
            FROM live_hypothesis_scores
            WHERE hypothesis_id = ?
            ORDER BY scored_at DESC
            LIMIT 1
            """,
            (hypothesis_id,),
        ).fetchone()

        if row is None:
            return False

        live_sharpe = float(row["live_sharpe"] or 0)
        bt_sharpe = float(row["backtest_sharpe"] or 0)
        return self._compute_degradation_flag(live_sharpe, bt_sharpe)

    @staticmethod
    def _compute_degradation_flag(live_sharpe: float, bt_sharpe: float) -> bool:
        """True when live performance is materially below backtest expectation."""
        if bt_sharpe <= 0:
            return live_sharpe < 0
        return live_sharpe < DEGRADATION_THRESHOLD * bt_sharpe

    # ------------------------------------------------------------------
    # Retest trigger
    # ------------------------------------------------------------------

    def trigger_retest(self, hypothesis_id: int) -> None:
        """
        Re-queue a degraded hypothesis for walk-forward validation.

        Writes an entry to the ``wfa_queue`` table (if it exists) and
        fires a ``hypothesis_retest_triggered`` event into ``event_log``.

        Parameters
        ----------
        hypothesis_id : int
        """
        payload = json.dumps(
            {
                "hypothesis_id": hypothesis_id,
                "reason": "live_degradation",
                "triggered_at": _utcnow_str(),
            }
        )

        # Try to insert into wfa_queue
        try:
            self._iae_conn.execute(
                """
                INSERT OR IGNORE INTO wfa_queue (hypothesis_id, status, created_at)
                VALUES (?, 'pending', ?)
                """,
                (hypothesis_id, _utcnow_str()),
            )
        except sqlite3.OperationalError:
            logger.debug("wfa_queue table not present — skipping direct insert.")

        # Reset hypothesis status to 'queued'
        try:
            self._iae_conn.execute(
                "UPDATE hypotheses SET status = 'queued' WHERE id = ?",
                (hypothesis_id,),
            )
        except sqlite3.OperationalError:
            pass

        # Always log the event
        try:
            self._iae_conn.execute(
                "INSERT INTO event_log (event_type, payload_json) VALUES (?, ?)",
                ("hypothesis_retest_triggered", payload),
            )
        except Exception:
            logger.warning("Could not write retest event to event_log.")

        self._iae_conn.commit()
        logger.info("Retest triggered for hypothesis %d.", hypothesis_id)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_live_score(self, score: LiveScore) -> None:
        """Upsert a LiveScore into ``live_hypothesis_scores``."""
        d = score.to_dict()
        self._iae_conn.execute(
            """
            INSERT INTO live_hypothesis_scores
                (hypothesis_id, window_days, live_sharpe, live_win_rate, live_avg_pnl,
                 live_trade_count, backtest_sharpe, degradation_ratio, is_degraded, scored_at)
            VALUES
                (:hypothesis_id, :window_days, :live_sharpe, :live_win_rate, :live_avg_pnl,
                 :live_trade_count, :backtest_sharpe, :degradation_ratio, :is_degraded, :scored_at)
            """,
            d,
        )
        self._iae_conn.commit()
        logger.debug("Persisted live score for hypothesis %d.", score.hypothesis_id)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _load_active_hypotheses(self) -> pd.DataFrame:
        """
        Load all hypotheses with status in ('promoted', 'validated') from
        idea_engine.db.  Falls back to an empty DataFrame gracefully.
        """
        try:
            df = pd.read_sql_query(
                """
                SELECT id,
                       type,
                       parameters,
                       created_at,
                       status
                FROM hypotheses
                WHERE status IN ('promoted', 'validated')
                ORDER BY created_at DESC
                """,
                self._iae_conn,
            )
        except Exception:
            df = pd.DataFrame(columns=["id", "type", "parameters", "created_at", "status"])
        return df

    def _fetch_backtest_sharpe(self, hypothesis_id: int) -> float:
        """
        Retrieve the backtest Sharpe ratio for a hypothesis.

        Checks (in order):
        1. ``wfa_results`` table (OOS Sharpe from walk-forward).
        2. ``hypotheses`` table ``predicted_sharpe_delta`` column.
        3. Returns 0.0 if neither is available.
        """
        # Try WFA results first
        try:
            row = self._iae_conn.execute(
                """
                SELECT AVG(oos_sharpe) AS bt_sharpe
                FROM wfa_results
                WHERE hypothesis_id = ?
                """,
                (hypothesis_id,),
            ).fetchone()
            if row and row["bt_sharpe"] is not None:
                return float(row["bt_sharpe"])
        except sqlite3.OperationalError:
            pass

        # Fall back to predicted_sharpe_delta
        try:
            row = self._iae_conn.execute(
                "SELECT predicted_sharpe_delta FROM hypotheses WHERE id = ?",
                (hypothesis_id,),
            ).fetchone()
            if row and row[0] is not None:
                return float(row[0])
        except sqlite3.OperationalError:
            pass

        return 0.0

    def _load_watermark(self) -> str:
        """
        Load the last-processed timestamp from idea_engine.db.
        Returns a very old date string if no watermark exists yet.
        """
        try:
            row = self._iae_conn.execute(
                """
                SELECT payload_json
                FROM event_log
                WHERE event_type = 'live_feedback_watermark'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if row:
                data = json.loads(row["payload_json"])
                return data.get("last_checked", "2000-01-01T00:00:00Z")
        except Exception:
            pass
        return "2000-01-01T00:00:00Z"

    def _save_watermark(self, ts: str) -> None:
        """Persist the watermark timestamp to event_log."""
        payload = json.dumps({"last_checked": ts})
        try:
            self._iae_conn.execute(
                "INSERT INTO event_log (event_type, payload_json) VALUES (?, ?)",
                ("live_feedback_watermark", payload),
            )
            self._iae_conn.commit()
        except Exception:
            logger.warning("Could not save watermark.")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "LiveFeedbackMonitor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
