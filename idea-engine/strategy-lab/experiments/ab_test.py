"""
ab_test.py
----------
A/B test runner for strategy variants.

ABTest dataclass
----------------
Holds test configuration: test_id, version_a (control), version_b (challenger),
start/end dates, allocation split, and current status.

ABTestRunner
------------
Runs both variants simultaneously against the same price history using the
paper simulator, then accumulates trade logs and daily P&L series for each
variant. Implements:
  * Minimum 100 trades per variant before significance test.
  * Sequential Probability Ratio Test (SPRT) for early stopping.

ABTestStore
-----------
SQLite persistence for test metadata and trade logs.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

from ..versioning.strategy_version import StrategyVersion

_DEFAULT_DB = Path(__file__).parent.parent / "strategy_lab.db"
_MIN_TRADES_PER_VARIANT = 100
_SPRT_H0_EFFECT = 0.0    # null hypothesis: no difference in mean daily return
_SPRT_H1_EFFECT = 0.001  # alternative: 0.1% daily edge
_SPRT_ALPHA     = 0.05
_SPRT_BETA      = 0.20


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class ABTestStatus(str, Enum):
    PENDING    = "PENDING"
    RUNNING    = "RUNNING"
    STOPPED    = "STOPPED"    # stopped early by SPRT
    COMPLETED  = "COMPLETED"  # ran to end_date
    CANCELLED  = "CANCELLED"


# ---------------------------------------------------------------------------
# ABTest dataclass
# ---------------------------------------------------------------------------

@dataclass
class ABTest:
    """
    Configuration and live state of a single A/B experiment.

    Attributes
    ----------
    test_id        : unique UUID
    version_a_id   : control strategy version ID
    version_b_id   : challenger strategy version ID
    start_date     : ISO date string
    end_date       : ISO date string (may be None if open-ended)
    allocation_a   : fraction of capital allocated to variant A (0.5 = 50/50)
    status         : current test lifecycle state
    stop_reason    : populated when status is STOPPED or COMPLETED
    created_at     : UTC ISO timestamp
    """
    test_id: str
    version_a_id: str
    version_b_id: str
    start_date: str
    end_date: str | None
    allocation_a: float
    status: ABTestStatus
    stop_reason: str
    created_at: str

    # mutable runtime state (not persisted in the main row)
    trades_a: list[dict] = field(default_factory=list, repr=False)
    trades_b: list[dict] = field(default_factory=list, repr=False)
    daily_pnl_a: list[float] = field(default_factory=list, repr=False)
    daily_pnl_b: list[float] = field(default_factory=list, repr=False)

    @classmethod
    def new(
        cls,
        version_a_id: str,
        version_b_id: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        allocation_a: float = 0.5,
    ) -> "ABTest":
        return cls(
            test_id=str(uuid.uuid4()),
            version_a_id=version_a_id,
            version_b_id=version_b_id,
            start_date=start_date or date.today().isoformat(),
            end_date=end_date,
            allocation_a=allocation_a,
            status=ABTestStatus.PENDING,
            stop_reason="",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def allocation_b(self) -> float:
        return 1.0 - self.allocation_a

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        # Don't persist full trade logs in the main dict; stored separately
        d.pop("trades_a", None)
        d.pop("trades_b", None)
        d.pop("daily_pnl_a", None)
        d.pop("daily_pnl_b", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ABTest":
        d = dict(d)
        d["status"] = ABTestStatus(d["status"])
        d.setdefault("trades_a", [])
        d.setdefault("trades_b", [])
        d.setdefault("daily_pnl_a", [])
        d.setdefault("daily_pnl_b", [])
        return cls(**d)

    # ------------------------------------------------------------------
    # Quick stats helpers
    # ------------------------------------------------------------------

    def n_trades(self) -> tuple[int, int]:
        return len(self.trades_a), len(self.trades_b)

    def win_rate(self) -> tuple[float, float]:
        def wr(trades: list[dict]) -> float:
            if not trades:
                return 0.0
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            return wins / len(trades)
        return wr(self.trades_a), wr(self.trades_b)

    def mean_pnl(self) -> tuple[float, float]:
        def mp(trades: list[dict]) -> float:
            if not trades:
                return 0.0
            return float(np.mean([t.get("pnl", 0) for t in trades]))
        return mp(self.trades_a), mp(self.trades_b)

    def sharpe(self) -> tuple[float, float]:
        def sh(pnl: list[float]) -> float:
            if len(pnl) < 2:
                return 0.0
            arr = np.array(pnl, dtype=float)
            std = np.std(arr, ddof=1)
            return float(np.mean(arr) / std * math.sqrt(252)) if std > 0 else 0.0
        return sh(self.daily_pnl_a), sh(self.daily_pnl_b)

    def add_trade(self, variant: str, trade: dict) -> None:
        if variant == "A":
            self.trades_a.append(trade)
        else:
            self.trades_b.append(trade)

    def add_daily_pnl(self, variant: str, pnl: float) -> None:
        if variant == "A":
            self.daily_pnl_a.append(pnl)
        else:
            self.daily_pnl_b.append(pnl)


# ---------------------------------------------------------------------------
# SPRT (Sequential Probability Ratio Test)
# ---------------------------------------------------------------------------

class SPRTDecision(str, Enum):
    CONTINUE = "CONTINUE"
    ACCEPT_H0 = "ACCEPT_H0"   # no difference — stop, declare tie
    ACCEPT_H1 = "ACCEPT_H1"   # B is better — stop early


def sprt_check(
    returns_a: list[float],
    returns_b: list[float],
    h1_delta: float = _SPRT_H1_EFFECT,
    alpha: float = _SPRT_ALPHA,
    beta: float = _SPRT_BETA,
) -> SPRTDecision:
    """
    Sequential Probability Ratio Test on the difference in daily returns.

    Uses Wald's SPRT with boundaries A = (1-beta)/alpha, B = beta/(1-alpha).
    Models the difference in returns as normally distributed.
    """
    if len(returns_a) < 10 or len(returns_b) < 10:
        return SPRTDecision.CONTINUE

    diff = [b - a for a, b in zip(returns_a, returns_b)]
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1)) if n > 1 else 1e-9
    if std_diff < 1e-12:
        std_diff = 1e-9

    # Log-likelihood ratio for each observation under H1 vs H0
    # H0: mu=0, H1: mu=h1_delta (assuming known sigma = std_diff)
    log_lr = (h1_delta / std_diff**2) * sum(diff) - n * h1_delta**2 / (2 * std_diff**2)

    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))

    if log_lr >= upper:
        return SPRTDecision.ACCEPT_H1
    if log_lr <= lower:
        return SPRTDecision.ACCEPT_H0
    return SPRTDecision.CONTINUE


# ---------------------------------------------------------------------------
# ABTestRunner
# ---------------------------------------------------------------------------

class ABTestRunner:
    """
    Runs an A/B test using the paper simulator for each variant.

    Usage
    -----
    runner = ABTestRunner(store, version_store, simulator_factory)
    runner.run(test, price_data)
    """

    def __init__(
        self,
        ab_store: "ABTestStore",
        min_trades: int = _MIN_TRADES_PER_VARIANT,
    ) -> None:
        self.ab_store   = ab_store
        self.min_trades = min_trades

    def run(
        self,
        test: ABTest,
        price_data: pd.DataFrame,
        params_a: dict[str, Any],
        params_b: dict[str, Any],
        capital: float = 1_000_000.0,
    ) -> ABTest:
        """
        Run test on price_data using params_a and params_b.
        Returns the updated ABTest with trade logs and daily P&L populated.

        price_data : DataFrame with DatetimeIndex, columns per instrument.
        """
        from ..simulation.paper_simulator import PaperSimulator

        test.status = ABTestStatus.RUNNING
        self.ab_store.save_test(test)

        sim_a = PaperSimulator(params_a, capital=capital * test.allocation_a)
        sim_b = PaperSimulator(params_b, capital=capital * test.allocation_b)

        dates = sorted(price_data.index.normalize().unique())

        for day in dates:
            day_data = price_data[price_data.index.normalize() == day]

            result_a = sim_a.step(day_data)
            result_b = sim_b.step(day_data)

            for trade in result_a.get("trades", []):
                test.add_trade("A", trade)
            for trade in result_b.get("trades", []):
                test.add_trade("B", trade)

            test.add_daily_pnl("A", result_a.get("daily_pnl", 0.0))
            test.add_daily_pnl("B", result_b.get("daily_pnl", 0.0))

            # SPRT early-stopping check after minimum trades
            na, nb = test.n_trades()
            if na >= self.min_trades and nb >= self.min_trades:
                decision = sprt_check(test.daily_pnl_a, test.daily_pnl_b)
                if decision != SPRTDecision.CONTINUE:
                    test.status     = ABTestStatus.STOPPED
                    test.stop_reason = f"SPRT early stop: {decision.value}"
                    self.ab_store.save_test(test)
                    return test

        test.status     = ABTestStatus.COMPLETED
        test.stop_reason = "Ran to full end date"
        self.ab_store.save_test(test)
        return test


# ---------------------------------------------------------------------------
# ABTestStore
# ---------------------------------------------------------------------------

class ABTestStore:
    """SQLite persistence for ABTest objects and associated trade logs."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id       TEXT PRIMARY KEY,
                    version_a_id  TEXT NOT NULL,
                    version_b_id  TEXT NOT NULL,
                    start_date    TEXT NOT NULL,
                    end_date      TEXT,
                    allocation_a  REAL NOT NULL DEFAULT 0.5,
                    status        TEXT NOT NULL DEFAULT 'PENDING',
                    stop_reason   TEXT NOT NULL DEFAULT '',
                    created_at    TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ab_trades (
                    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id     TEXT NOT NULL,
                    variant     TEXT NOT NULL,
                    trade_json  TEXT NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES ab_tests(test_id)
                );
                CREATE TABLE IF NOT EXISTS ab_daily_pnl (
                    rowid    INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id  TEXT NOT NULL,
                    variant  TEXT NOT NULL,
                    pnl      REAL NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES ab_tests(test_id)
                );
            """)

    def save_test(self, test: ABTest) -> None:
        with self._conn() as conn:
            d = test.to_dict()
            conn.execute(
                """INSERT OR REPLACE INTO ab_tests
                   (test_id, version_a_id, version_b_id, start_date, end_date,
                    allocation_a, status, stop_reason, created_at)
                   VALUES (:test_id,:version_a_id,:version_b_id,:start_date,:end_date,
                           :allocation_a,:status,:stop_reason,:created_at)""",
                d,
            )
            # Upsert trades
            conn.execute("DELETE FROM ab_trades WHERE test_id = ?", (test.test_id,))
            for t in test.trades_a:
                conn.execute(
                    "INSERT INTO ab_trades (test_id, variant, trade_json) VALUES (?,?,?)",
                    (test.test_id, "A", json.dumps(t, default=str)),
                )
            for t in test.trades_b:
                conn.execute(
                    "INSERT INTO ab_trades (test_id, variant, trade_json) VALUES (?,?,?)",
                    (test.test_id, "B", json.dumps(t, default=str)),
                )
            conn.execute("DELETE FROM ab_daily_pnl WHERE test_id = ?", (test.test_id,))
            for pnl in test.daily_pnl_a:
                conn.execute(
                    "INSERT INTO ab_daily_pnl (test_id, variant, pnl) VALUES (?,?,?)",
                    (test.test_id, "A", pnl),
                )
            for pnl in test.daily_pnl_b:
                conn.execute(
                    "INSERT INTO ab_daily_pnl (test_id, variant, pnl) VALUES (?,?,?)",
                    (test.test_id, "B", pnl),
                )

    def load_test(self, test_id: str) -> ABTest | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM ab_tests WHERE test_id = ?", (test_id,)
            ).fetchone()
            if not row:
                return None
            test = ABTest.from_dict(dict(row))
            test.trades_a = [
                json.loads(r["trade_json"])
                for r in conn.execute(
                    "SELECT trade_json FROM ab_trades WHERE test_id=? AND variant='A'",
                    (test_id,),
                )
            ]
            test.trades_b = [
                json.loads(r["trade_json"])
                for r in conn.execute(
                    "SELECT trade_json FROM ab_trades WHERE test_id=? AND variant='B'",
                    (test_id,),
                )
            ]
            test.daily_pnl_a = [
                r["pnl"]
                for r in conn.execute(
                    "SELECT pnl FROM ab_daily_pnl WHERE test_id=? AND variant='A'",
                    (test_id,),
                )
            ]
            test.daily_pnl_b = [
                r["pnl"]
                for r in conn.execute(
                    "SELECT pnl FROM ab_daily_pnl WHERE test_id=? AND variant='B'",
                    (test_id,),
                )
            ]
            return test

    def all_tests(self) -> list[ABTest]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT test_id FROM ab_tests ORDER BY created_at"
            ).fetchall()
        return [self.load_test(r["test_id"]) for r in rows]  # type: ignore[misc]
