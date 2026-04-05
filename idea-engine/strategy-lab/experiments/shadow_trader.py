"""
shadow_trader.py
----------------
Shadow trading: run challenger version on the same live signals but WITHOUT
submitting orders. Tracks what the challenger would have done differently and
computes virtual P&L, missed trades, and avoided losses.

This is lower risk than a true A/B test because no real capital is at stake —
useful as an initial validation gate before committing to a full A/B.

ShadowTrader
------------
Receives live trade events from the control strategy and evaluates whether the
challenger strategy would have taken the same trade, a different trade, or no trade.

ShadowResult
------------
Aggregated statistics from a shadow run: virtual P&L, hit rate vs control,
avoided losses, missed winners.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

_DEFAULT_DB = Path(__file__).parent.parent / "strategy_lab.db"


# ---------------------------------------------------------------------------
# ShadowTrade — record of one comparison
# ---------------------------------------------------------------------------

@dataclass
class ShadowTrade:
    """
    Comparison of a single live trade (control) vs what the challenger would do.

    Attributes
    ----------
    timestamp       : when the live signal was generated
    symbol          : instrument
    live_action     : "BUY" | "SELL" | "HOLD"
    shadow_action   : "BUY" | "SELL" | "HOLD"
    live_pnl        : actual P&L of the live trade (filled in at close)
    shadow_pnl      : virtual P&L if the shadow trade had been taken
    agreed          : True if shadow_action == live_action
    live_quantity   : lots/contracts in live trade
    shadow_quantity : lots/contracts in shadow trade
    reason          : why shadow differed (e.g., "min_hold_bars not met")
    """
    timestamp: str
    symbol: str
    live_action: str
    shadow_action: str
    live_pnl: float
    shadow_pnl: float
    agreed: bool
    live_quantity: float
    shadow_quantity: float
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShadowTrade":
        return cls(**d)


# ---------------------------------------------------------------------------
# ShadowResult — aggregate statistics
# ---------------------------------------------------------------------------

@dataclass
class ShadowResult:
    """
    Aggregated statistics from a shadow trading run.

    Attributes
    ----------
    run_id             : UUID of this shadow run
    challenger_id      : version ID of the challenger strategy
    control_id         : version ID of the control strategy
    n_live_trades      : total live trades observed
    n_agreed           : shadow agreed with control
    n_differed         : shadow would have done something different
    virtual_pnl        : cumulative virtual P&L of shadow-only positions
    live_pnl           : cumulative P&L of live trades over same window
    missed_winners     : trades where live won but shadow would have skipped
    avoided_losses     : trades where live lost but shadow would have skipped
    virtual_sharpe     : Sharpe ratio of shadow daily P&L series
    live_sharpe        : Sharpe ratio of live daily P&L series
    agreement_rate     : n_agreed / n_live_trades
    """
    run_id: str
    challenger_id: str
    control_id: str
    n_live_trades: int
    n_agreed: int
    n_differed: int
    virtual_pnl: float
    live_pnl: float
    missed_winners: int
    avoided_losses: int
    virtual_sharpe: float
    live_sharpe: float
    agreement_rate: float
    shadow_trades: list[ShadowTrade] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        lines = [
            f"Shadow Run: {self.run_id[:8]}",
            f"  Challenger: {self.challenger_id[:8]}  vs  Control: {self.control_id[:8]}",
            f"  Trades observed : {self.n_live_trades}",
            f"  Agreement rate  : {self.agreement_rate:.1%}",
            f"  Virtual P&L     : ${self.virtual_pnl:,.0f}",
            f"  Live P&L        : ${self.live_pnl:,.0f}",
            f"  Avoided losses  : {self.avoided_losses}",
            f"  Missed winners  : {self.missed_winners}",
            f"  Virtual Sharpe  : {self.virtual_sharpe:.3f}",
            f"  Live Sharpe     : {self.live_sharpe:.3f}",
        ]
        verdict = (
            "CHALLENGER LOOKS BETTER"
            if self.virtual_pnl > self.live_pnl and self.avoided_losses > self.missed_winners
            else "NO CLEAR ADVANTAGE"
        )
        lines.append(f"  Verdict: {verdict}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ShadowTrader
# ---------------------------------------------------------------------------

class ShadowTrader:
    """
    Runs a challenger strategy in shadow mode alongside the live control strategy.

    Usage
    -----
    shadow = ShadowTrader(control_params, challenger_params, db_path)

    # As live signals arrive:
    shadow.observe_live_trade(trade_dict, price_data)

    # At end of session:
    result = shadow.compute_result()
    """

    def __init__(
        self,
        challenger_id: str,
        control_id: str,
        challenger_params: dict[str, Any],
        control_params: dict[str, Any],
        db_path: str | Path = _DEFAULT_DB,
    ) -> None:
        import uuid
        self.run_id          = str(uuid.uuid4())
        self.challenger_id   = challenger_id
        self.control_id      = control_id
        self.challenger_params = challenger_params
        self.control_params    = control_params
        self.db_path         = Path(db_path)
        self._shadow_trades: list[ShadowTrade] = []
        self._daily_pnl_shadow: dict[str, float] = {}
        self._daily_pnl_live:   dict[str, float] = {}
        self._init_db()

    # ------------------------------------------------------------------
    # Core method: observe one live trade, evaluate shadow response
    # ------------------------------------------------------------------

    def observe_live_trade(
        self,
        live_trade: dict[str, Any],
        bar_data: pd.DataFrame,
    ) -> ShadowTrade:
        """
        Called when the live control strategy executes a trade.

        live_trade must contain:
          symbol, action ("BUY"/"SELL"/"HOLD"), quantity, entry_price,
          exit_price (None if still open), pnl (None if still open),
          timestamp

        Returns the ShadowTrade comparison record.
        """
        symbol     = live_trade.get("symbol", "UNKNOWN")
        live_act   = live_trade.get("action", "HOLD")
        live_qty   = float(live_trade.get("quantity", 0))
        live_pnl   = float(live_trade.get("pnl", 0.0) or 0.0)
        ts         = live_trade.get("timestamp", _now_iso())

        # Evaluate what challenger would do
        shadow_act, shadow_qty, reason = self._evaluate_challenger(
            symbol, live_trade, bar_data
        )
        shadow_pnl = self._compute_virtual_pnl(live_trade, shadow_act, shadow_qty)

        agreed = (live_act == shadow_act)

        st = ShadowTrade(
            timestamp=ts,
            symbol=symbol,
            live_action=live_act,
            shadow_action=shadow_act,
            live_pnl=live_pnl,
            shadow_pnl=shadow_pnl,
            agreed=agreed,
            live_quantity=live_qty,
            shadow_quantity=shadow_qty,
            reason=reason,
        )
        self._shadow_trades.append(st)
        self._persist_shadow_trade(st)

        # Accumulate daily P&L buckets
        day = ts[:10]
        self._daily_pnl_shadow[day] = self._daily_pnl_shadow.get(day, 0.0) + shadow_pnl
        self._daily_pnl_live[day]   = self._daily_pnl_live.get(day, 0.0) + live_pnl

        return st

    # ------------------------------------------------------------------
    # Replay on historical data
    # ------------------------------------------------------------------

    def replay(
        self,
        live_trades: list[dict[str, Any]],
        price_data: pd.DataFrame,
    ) -> "ShadowResult":
        """
        Replay a list of historical live trades through the shadow evaluator.
        Useful for offline evaluation before deploying shadow mode live.
        """
        for trade in live_trades:
            ts = trade.get("timestamp", "")
            day_data = price_data[price_data.index.normalize() == pd.Timestamp(ts[:10])] \
                       if ts else price_data
            self.observe_live_trade(trade, day_data)
        return self.compute_result()

    # ------------------------------------------------------------------
    # Result computation
    # ------------------------------------------------------------------

    def compute_result(self) -> ShadowResult:
        """Aggregate all observed shadow trades into a ShadowResult."""
        n = len(self._shadow_trades)
        n_agreed   = sum(1 for t in self._shadow_trades if t.agreed)
        n_differed = n - n_agreed

        virtual_pnl = sum(t.shadow_pnl for t in self._shadow_trades)
        live_pnl    = sum(t.live_pnl for t in self._shadow_trades)

        missed_winners = sum(
            1 for t in self._shadow_trades
            if not t.agreed and t.live_pnl > 0 and t.shadow_action == "HOLD"
        )
        avoided_losses = sum(
            1 for t in self._shadow_trades
            if not t.agreed and t.live_pnl < 0 and t.shadow_action == "HOLD"
        )

        shadow_daily = list(self._daily_pnl_shadow.values())
        live_daily   = list(self._daily_pnl_live.values())

        v_sharpe = _sharpe(shadow_daily)
        l_sharpe = _sharpe(live_daily)

        return ShadowResult(
            run_id=self.run_id,
            challenger_id=self.challenger_id,
            control_id=self.control_id,
            n_live_trades=n,
            n_agreed=n_agreed,
            n_differed=n_differed,
            virtual_pnl=virtual_pnl,
            live_pnl=live_pnl,
            missed_winners=missed_winners,
            avoided_losses=avoided_losses,
            virtual_sharpe=v_sharpe,
            live_sharpe=l_sharpe,
            agreement_rate=n_agreed / n if n > 0 else 0.0,
            shadow_trades=list(self._shadow_trades),
        )

    # ------------------------------------------------------------------
    # Challenger evaluation logic
    # ------------------------------------------------------------------

    def _evaluate_challenger(
        self,
        symbol: str,
        live_trade: dict[str, Any],
        bar_data: pd.DataFrame,
    ) -> tuple[str, float, str]:
        """
        Determine what the challenger strategy would do.
        Returns (action, quantity, reason).

        This is a parameter-diff driven evaluation:
        - If min_hold_bars differs and the live trade would be blocked by challenger's
          higher min_hold_bars, challenger says HOLD.
        - If BH_FORM_OVERRIDE differs for the symbol, re-evaluate the formation signal.
        - If the symbol is not in challenger's instrument universe, say HOLD.
        - Otherwise, mirror the live trade (conservative: assume same logic).
        """
        c_params = self.challenger_params
        live_action = live_trade.get("action", "HOLD")
        live_qty    = float(live_trade.get("quantity", 0))
        bars_held   = int(live_trade.get("bars_held", 0))

        # Check 1: instrument universe
        universe = c_params.get("instruments", [])
        if universe and symbol not in universe:
            return "HOLD", 0.0, f"{symbol} not in challenger universe"

        # Check 2: min_hold_bars
        chall_min_hold = int(c_params.get("min_hold_bars", 4))
        if live_action in ("BUY", "SELL") and bars_held < chall_min_hold:
            return "HOLD", 0.0, f"bars_held={bars_held} < challenger min_hold_bars={chall_min_hold}"

        # Check 3: BH formation override (if challenger requires stronger signal)
        sym_key = f"bh_form_override_{symbol.lower()}"
        chall_form = float(c_params.get(sym_key, 1.0))
        ctrl_form  = float(self.control_params.get(sym_key, 1.0))
        formation_strength = float(live_trade.get("formation_strength", chall_form))
        if chall_form > ctrl_form and formation_strength < chall_form:
            return "HOLD", 0.0, f"formation_strength={formation_strength:.2f} < challenger threshold={chall_form:.2f}"

        # Check 4: Harvest z-score thresholds
        if live_trade.get("mode") == "HARVEST":
            chall_z_entry = float(c_params.get("harvest_z_entry", 1.5))
            z_score = float(live_trade.get("z_score", chall_z_entry))
            if live_action in ("BUY", "SELL") and abs(z_score) < chall_z_entry:
                return "HOLD", 0.0, f"|z|={abs(z_score):.2f} < challenger z_entry={chall_z_entry:.2f}"

        # Challenger mirrors the trade — scale quantity by risk ratio
        ctrl_risk  = float(self.control_params.get("per_inst_risk", 0.00181))
        chall_risk = float(c_params.get("per_inst_risk", ctrl_risk))
        scale = (chall_risk / ctrl_risk) if ctrl_risk > 0 else 1.0
        shadow_qty = live_qty * scale

        return live_action, shadow_qty, "mirrored with risk scaling"

    @staticmethod
    def _compute_virtual_pnl(
        live_trade: dict[str, Any],
        shadow_action: str,
        shadow_qty: float,
    ) -> float:
        """
        Compute virtual P&L for the shadow trade.
        Scales live P&L by the ratio of shadow_qty / live_qty.
        Returns 0 if shadow says HOLD.
        """
        if shadow_action == "HOLD":
            return 0.0
        live_qty = float(live_trade.get("quantity", 0))
        live_pnl = float(live_trade.get("pnl", 0.0) or 0.0)
        if live_qty == 0:
            return 0.0
        return live_pnl * (shadow_qty / live_qty)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_shadow_trade(self, st: ShadowTrade) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO shadow_trades
                   (run_id, trade_json) VALUES (?,?)""",
                (self.run_id, json.dumps(st.to_dict(), default=str)),
            )

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS shadow_trades (
                    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id     TEXT NOT NULL,
                    trade_json TEXT NOT NULL
                );
            """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe(daily_pnl: list[float]) -> float:
    import math
    if len(daily_pnl) < 2:
        return 0.0
    arr = np.array(daily_pnl, dtype=float)
    std = float(np.std(arr, ddof=1))
    return float(np.mean(arr) / std * math.sqrt(252)) if std > 0 else 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
