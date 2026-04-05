"""
ShadowRunner
============
Maintains N shadow strategy instances that silently track live market data
in paper mode.  Each shadow runs a different genome (parameter set) drawn
from the hall_of_fame table in idea_engine.db.

On each ``tick(bar_data)``, all shadows receive the same OHLCV bar, update
their internal state, optionally execute virtual trades, and have their
equity curve recorded in the ``shadow_runs`` DB table.

No real orders are placed.  The system exists purely to evaluate whether
an alternative genome would have outperformed the live strategy.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .shadow_state import ShadowState, VirtualTrade

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV = "IDEA_ENGINE_DB"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "db" / "idea_engine.db"

MIN_HOLD_BARS_DEFAULT = 4
POS_FLOOR_SCALE_DEFAULT = 0.01
DELTA_MAX_FRAC_DEFAULT = 0.25


# ---------------------------------------------------------------------------
# ShadowRunner
# ---------------------------------------------------------------------------

class ShadowRunner:
    """
    Manages a cohort of shadow strategies, feeding each a live market bar
    and recording their virtual performance.

    Parameters
    ----------
    db_path : str | Path | None
        Path to ``idea_engine.db``.
    n_shadows : int
        Number of shadow strategies to maintain (default 5).
    initial_equity : float
        Starting virtual equity for each shadow (default 100 000).
    persist_every : int
        Persist shadow state to DB every N bars (default 10).
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        n_shadows: int = 5,
        initial_equity: float = 100_000.0,
        persist_every: int = 10,
    ) -> None:
        self.db_path = Path(
            db_path or os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        self.n_shadows = n_shadows
        self.initial_equity = initial_equity
        self.persist_every = persist_every

        self._shadows: dict[str, ShadowState] = {}  # shadow_id -> ShadowState
        self._bar_count: int = 0
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        sql_file = Path(__file__).parent / "schema_extension.sql"
        if sql_file.exists():
            ddl = sql_file.read_text()
        else:
            ddl = _INLINE_DDL
        with self._db() as con:
            con.executescript(ddl)

    def _db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # ------------------------------------------------------------------
    # Load / initialise shadows
    # ------------------------------------------------------------------

    def load_shadows(self, n: int | None = None) -> list[ShadowState]:
        """
        Load the top-N genomes from the ``hall_of_fame`` table and
        initialise shadow strategies for each.

        Previously persisted shadow states are restored from the DB if
        available; otherwise fresh states are created.

        Parameters
        ----------
        n : int | None
            Override the instance ``n_shadows`` for this call.

        Returns
        -------
        List of loaded ShadowState objects.
        """
        n = n or self.n_shadows
        genomes = self._fetch_top_genomes(n)
        if not genomes:
            logger.warning("No genomes found in hall_of_fame — shadow runner idle")
            return []

        loaded: list[ShadowState] = []
        for genome_id, genome_params in genomes:
            # Try to restore existing shadow state
            state = self._restore_shadow(genome_id)
            if state is None:
                state = ShadowState.from_genome(genome_id, genome_params)
                state.initial_equity = self.initial_equity
                state.equity = self.initial_equity
                logger.info("Created new shadow %s for genome %d", state.shadow_id, genome_id)
            else:
                logger.info("Restored shadow %s for genome %d", state.shadow_id, genome_id)

            self._shadows[state.shadow_id] = state
            loaded.append(state)

        return loaded

    def _fetch_top_genomes(self, n: int) -> list[tuple[int, dict[str, Any]]]:
        """Fetch top-N genomes from hall_of_fame ordered by fitness score."""
        try:
            with self._db() as con:
                rows = con.execute(
                    "SELECT id, params FROM hall_of_fame "
                    "ORDER BY fitness DESC LIMIT ?",
                    (n,),
                ).fetchall()
            return [(int(row["id"]), json.loads(row["params"])) for row in rows]
        except sqlite3.OperationalError as exc:
            logger.warning("Could not query hall_of_fame: %s", exc)
            return []

    def _restore_shadow(self, genome_id: int) -> ShadowState | None:
        """Attempt to restore the most recent shadow state for a genome_id."""
        try:
            with self._db() as con:
                row = con.execute(
                    "SELECT shadow_state_json FROM shadow_runs "
                    "WHERE genome_id = ? AND shadow_state_json IS NOT NULL "
                    "ORDER BY created_at DESC LIMIT 1",
                    (genome_id,),
                ).fetchone()
            if row and row["shadow_state_json"]:
                return ShadowState.from_json(row["shadow_state_json"])
        except (sqlite3.OperationalError, KeyError, json.JSONDecodeError) as exc:
            logger.debug("Could not restore shadow for genome %d: %s", genome_id, exc)
        return None

    # ------------------------------------------------------------------
    # Tick — main per-bar update
    # ------------------------------------------------------------------

    def tick(self, bar_data: dict[str, Any]) -> dict[str, Any]:
        """
        Feed a new OHLCV bar to all active shadow strategies.

        Parameters
        ----------
        bar_data : dict
            Expected keys: ``open``, ``high``, ``low``, ``close``,
            ``volume``, ``symbol``, ``ts`` (Unix timestamp).

        Returns
        -------
        dict mapping shadow_id -> signal dict for this bar.
        """
        self._bar_count += 1
        symbol = str(bar_data.get("symbol", "UNKNOWN"))
        ts = float(bar_data.get("ts", time.time()))

        tick_results: dict[str, Any] = {}

        for shadow_id, state in self._shadows.items():
            try:
                signals = state.compute_signal(bar_data)
                tick_results[shadow_id] = signals

                # Decide whether to open/close virtual position
                self._execute_virtual_signal(state, bar_data, signals)

            except Exception as exc:  # noqa: BLE001
                logger.error("Shadow %s tick error: %s", shadow_id, exc)
                tick_results[shadow_id] = {"error": str(exc)}

        # Persist every N bars
        if self._bar_count % self.persist_every == 0:
            self._persist_all(symbol, ts)

        return tick_results

    # ------------------------------------------------------------------
    # Virtual trade execution
    # ------------------------------------------------------------------

    def _execute_virtual_signal(
        self,
        state: ShadowState,
        bar: dict[str, Any],
        signals: dict[str, float],
    ) -> None:
        """
        Translate a signal into virtual trades for the shadow.

        Entry logic:
          - combined signal > +0.5 → open long if not already long
          - combined signal < -0.5 → open short if not already short (future)
          - |signal| < 0.2        → close any open position

        Position sizing uses delta_max_frac and pos_floor_scale from genome.
        """
        symbol = str(bar.get("symbol", "UNKNOWN"))
        close = float(bar.get("close", bar.get("c", 0.0)))
        ts = float(bar.get("ts", time.time()))
        combined = signals.get("combined", 0.0)
        garch_ratio = signals.get("garch_ratio", 1.0)

        genome = state.genome
        delta_max_frac = float(genome.get("delta_max_frac", DELTA_MAX_FRAC_DEFAULT))
        pos_floor_scale = float(genome.get("pos_floor_scale", POS_FLOOR_SCALE_DEFAULT))
        min_hold = int(genome.get("min_hold_bars", MIN_HOLD_BARS_DEFAULT))
        stale_move = float(genome.get("stale_15m_move", 0.005))

        current_qty = state.positions.get(symbol, 0.0)
        equity = state.equity

        # Compute desired position size
        target_notional = equity * delta_max_frac * abs(combined)
        # Scale down if GARCH says high vol
        target_notional *= min(1.0, 1.5 / max(garch_ratio, 0.5))
        target_qty = target_notional / max(close, 1e-8) if combined > 0 else 0.0

        # Floor — don't bother trading tiny sizes
        if target_qty < equity * pos_floor_scale / max(close, 1e-8):
            target_qty = 0.0

        # State check: stale price filter (15-min-like move requirement)
        # simplified: only trade if bar move > stale_15m_move
        bar_move = abs(close - float(bar.get("open", close))) / max(close, 1e-8)
        if bar_move < stale_move and abs(combined) < 0.5:
            return  # not enough move to justify re-evaluation

        # Execute the delta
        qty_delta = target_qty - current_qty
        if abs(qty_delta) < 1e-6:
            return

        side = "buy" if qty_delta > 0 else "sell"
        trade = VirtualTrade(
            trade_id=str(uuid.uuid4()),
            shadow_id=state.shadow_id,
            symbol=symbol,
            side=side,
            qty=abs(qty_delta),
            price=close,
            ts=ts,
            signal_source="combined",
        )

        # Realised P&L for sell trades (approximate FIFO)
        if side == "sell" and current_qty > 0:
            avg_cost = self._avg_cost(state, symbol)
            trade.pnl = (close - avg_cost) * abs(qty_delta)
            state.equity += trade.pnl

        state.positions[symbol] = target_qty
        state.trades.append(trade)

    def _avg_cost(self, state: ShadowState, symbol: str) -> float:
        """Compute average cost basis for open long position in symbol."""
        buys = [t for t in state.trades if t.symbol == symbol and t.side == "buy"]
        if not buys:
            return 0.0
        total_cost = sum(t.qty * t.price for t in buys)
        total_qty = sum(t.qty for t in buys)
        return total_cost / max(total_qty, 1e-8)

    # ------------------------------------------------------------------
    # Live comparison
    # ------------------------------------------------------------------

    def compare_to_live(
        self,
        live_equity: float,
        shadow_id: str,
        live_initial: float = 100_000.0,
    ) -> dict[str, float]:
        """
        Compare a shadow's running return to the live strategy.

        Parameters
        ----------
        live_equity : float
            Current live strategy equity.
        shadow_id : str
            Which shadow to compare.
        live_initial : float
            Live strategy initial equity (default 100 000).

        Returns
        -------
        dict with keys: shadow_return, live_return, alpha, shadow_id
        """
        state = self._shadows.get(shadow_id)
        if state is None:
            return {"error": f"shadow_id {shadow_id!r} not found"}

        shadow_return = state.total_return()
        live_return = (live_equity - live_initial) / max(live_initial, 1.0)
        alpha = shadow_return - live_return

        return {
            "shadow_id": shadow_id,
            "genome_id": state.genome_id,
            "shadow_equity": state.equity,
            "shadow_return": shadow_return,
            "live_return": live_return,
            "alpha": alpha,
        }

    def compare_all_to_live(
        self,
        live_equity: float,
        live_initial: float = 100_000.0,
    ) -> list[dict[str, float]]:
        """
        Compare every active shadow to live and return sorted by alpha.

        Returns list of comparison dicts, best alpha first.
        """
        results = [
            self.compare_to_live(live_equity, sid, live_initial)
            for sid in self._shadows
        ]
        return sorted(
            [r for r in results if "error" not in r],
            key=lambda r: r.get("alpha", -999),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_all(self, symbol: str, ts: float) -> None:
        """Write current tick row for all shadows to ``shadow_runs``."""
        rows = []
        ts_str = _unix_to_iso(ts)
        for shadow_id, state in self._shadows.items():
            qty = state.positions.get(symbol, 0.0)
            # Get last signal if available
            signal_val = None
            if state.trades:
                last = state.trades[-1]
                if last.symbol == symbol:
                    signal_val = last.side

            rows.append((
                shadow_id,
                state.genome_id,
                ts_str,
                symbol,
                qty,
                state.equity,
                signal_val,
                state.to_json(),
            ))

        try:
            with self._db() as con:
                con.executemany(
                    """
                    INSERT OR REPLACE INTO shadow_runs
                        (shadow_id, genome_id, ts, symbol,
                         virtual_qty, virtual_equity, signal, shadow_state_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                con.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("shadow_runs persist error: %s", exc)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def shadow_ids(self) -> list[str]:
        """Return list of active shadow IDs."""
        return list(self._shadows.keys())

    def get_shadow(self, shadow_id: str) -> ShadowState | None:
        """Return a shadow state by ID."""
        return self._shadows.get(shadow_id)

    def add_shadow(self, genome_id: int, genome: dict[str, Any]) -> ShadowState:
        """Manually add a new shadow strategy."""
        state = ShadowState.from_genome(genome_id, genome)
        state.initial_equity = self.initial_equity
        state.equity = self.initial_equity
        self._shadows[state.shadow_id] = state
        logger.info("Added shadow %s for genome %d", state.shadow_id, genome_id)
        return state

    def remove_shadow(self, shadow_id: str) -> bool:
        """Remove a shadow by ID.  Returns True if found."""
        if shadow_id in self._shadows:
            del self._shadows[shadow_id]
            return True
        return False

    def summary(self) -> dict[str, Any]:
        """Return a summary of all active shadows."""
        return {
            "n_active": len(self._shadows),
            "bar_count": self._bar_count,
            "shadows": [
                {
                    "shadow_id": s.shadow_id,
                    "genome_id": s.genome_id,
                    "equity": s.equity,
                    "total_return": s.total_return(),
                    "n_trades": len(s.trades),
                    "win_rate": s.win_rate(),
                }
                for s in self._shadows.values()
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unix_to_iso(ts: float) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Fallback DDL
# ---------------------------------------------------------------------------

_INLINE_DDL = """
CREATE TABLE IF NOT EXISTS shadow_runs (
    shadow_id           TEXT    NOT NULL,
    genome_id           INTEGER NOT NULL,
    ts                  TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    virtual_qty         REAL    NOT NULL DEFAULT 0.0,
    virtual_equity      REAL    NOT NULL DEFAULT 0.0,
    signal              TEXT,
    shadow_state_json   TEXT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    PRIMARY KEY (shadow_id, ts, symbol)
);
CREATE TABLE IF NOT EXISTS shadow_comparisons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shadow_id       TEXT    NOT NULL,
    genome_id       INTEGER NOT NULL,
    period_days     INTEGER NOT NULL,
    shadow_return   REAL,
    live_return     REAL,
    alpha           REAL,
    promoted        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""
