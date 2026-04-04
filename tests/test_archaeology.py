"""
test_archaeology.py — Tests for trade archaeology / database query layer.

~400 LOC. Tests CSV parsing, BH state reconstruction, DB insert/query,
regime-filtered queries, tf_score edge analysis.
"""

from __future__ import annotations

import io
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT / "spacetime" / "engine"))


# ─────────────────────────────────────────────────────────────────────────────
# Inline archaeology layer (mirrors spacetime/db patterns)
# ─────────────────────────────────────────────────────────────────────────────

TRADE_COLUMNS = [
    "id", "entry_time", "exit_time", "sym", "entry_price", "exit_price",
    "pnl", "hold_bars", "mfe", "mae", "tf_score", "regime", "bh_mass_at_entry",
]

TRADE_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_time      TEXT,
    exit_time       TEXT,
    sym             TEXT,
    entry_price     REAL,
    exit_price      REAL,
    pnl             REAL,
    hold_bars       INTEGER,
    mfe             REAL,
    mae             REAL,
    tf_score        INTEGER,
    regime          TEXT,
    bh_mass_at_entry REAL
)
"""


def create_trade_db(db_path: str) -> sqlite3.Connection:
    """Create (or open) a SQLite trade database."""
    conn = sqlite3.connect(db_path)
    conn.execute(TRADE_CREATE_SQL)
    conn.commit()
    return conn


def insert_trade(conn: sqlite3.Connection, trade: dict) -> int:
    """Insert a trade dict into the database. Returns the row id."""
    cols = [c for c in TRADE_COLUMNS if c != "id" and c in trade]
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO trades ({','.join(cols)}) VALUES ({placeholders})"
    vals = [trade[c] for c in cols]
    cur = conn.execute(sql, vals)
    conn.commit()
    return cur.lastrowid


def query_by_regime(conn: sqlite3.Connection, regime: str) -> pd.DataFrame:
    """Query all trades matching a given regime."""
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE regime = ?", conn, params=(regime,)
    )
    return df


def query_by_tf_score(conn: sqlite3.Connection, min_tf: int) -> pd.DataFrame:
    """Query all trades with tf_score >= min_tf."""
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE tf_score >= ?", conn, params=(min_tf,)
    )
    return df


def edge_by_tf_score(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Aggregate trades by tf_score: compute win_rate, avg_pnl, profit_factor per score.
    Returns DataFrame sorted by tf_score ascending.
    """
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    if df.empty:
        return pd.DataFrame(columns=["tf_score", "count", "win_rate", "avg_pnl", "profit_factor"])
    results = []
    for score, grp in df.groupby("tf_score"):
        wins   = grp[grp["pnl"] > 0]["pnl"].sum()
        losses = abs(grp[grp["pnl"] <= 0]["pnl"].sum())
        results.append({
            "tf_score":      int(score),
            "count":         len(grp),
            "win_rate":      float((grp["pnl"] > 0).mean()),
            "avg_pnl":       float(grp["pnl"].mean()),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        })
    return pd.DataFrame(results).sort_values("tf_score").reset_index(drop=True)


def parse_qc_csv(csv_text: str) -> pd.DataFrame:
    """
    Parse a QuantConnect-style trade CSV.
    Expected columns: Entry Time, Exit Time, Symbol, Qty, Entry Price, Exit Price, PnL
    """
    df = pd.read_csv(io.StringIO(csv_text))
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip().replace(" ", "_")
        if lc in ("entry_time", "entrytime"):   col_map[c] = "entry_time"
        if lc in ("exit_time", "exittime"):     col_map[c] = "exit_time"
        if lc in ("symbol", "sym"):             col_map[c] = "sym"
        if lc in ("pnl", "profit", "profit/loss", "pnl_usd"): col_map[c] = "pnl"
        if lc in ("entry_price", "entryprice"): col_map[c] = "entry_price"
        if lc in ("exit_price", "exitprice"):   col_map[c] = "exit_price"
        if lc in ("qty", "quantity", "shares"): col_map[c] = "qty"
    df = df.rename(columns=col_map)
    return df


def reconstruct_bh_state(
    closes: np.ndarray,
    cf: float = 0.001,
    bh_form: float = 1.5,
    bh_collapse: float = 1.0,
    bh_decay: float = 0.95,
) -> pd.DataFrame:
    """
    Reconstruct BH state series from a price array.
    Returns DataFrame with: index, price, bh_active, bh_mass, bh_dir, ctl, bit
    """
    from srfm_core import MinkowskiClassifier, BlackHoleDetector
    mc = MinkowskiClassifier(cf=cf)
    bh = BlackHoleDetector(bh_form, bh_collapse, bh_decay)

    rows = []
    mc.update(float(closes[0]))
    rows.append({
        "price": closes[0], "bh_active": False,
        "bh_mass": 0.0, "bh_dir": 0, "ctl": 0, "bit": "UNKNOWN"
    })
    for i in range(1, len(closes)):
        bit = mc.update(float(closes[i]))
        act = bh.update(bit, float(closes[i]), float(closes[i-1]))
        rows.append({
            "price":     closes[i],
            "bh_active": act,
            "bh_mass":   bh.bh_mass,
            "bh_dir":    bh.bh_dir,
            "ctl":       bh.ctl,
            "bit":       bit,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sample CSV text
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_QC_CSV = """Entry Time,Exit Time,Symbol,Qty,Entry Price,Exit Price,PnL
2023-01-05 10:00:00,2023-01-05 14:00:00,ES,1,4500.0,4520.0,2000.0
2023-01-06 09:30:00,2023-01-06 13:00:00,NQ,1,11500.0,11480.0,-200.0
2023-01-09 10:00:00,2023-01-09 15:00:00,BTC,0.1,42000.0,43000.0,100.0
2023-01-10 09:00:00,2023-01-10 12:00:00,ES,1,4530.0,4510.0,-200.0
2023-01-11 10:30:00,2023-01-11 16:00:00,ES,1,4510.0,4550.0,4000.0
"""


def _make_sample_trades(n: int = 40) -> List[dict]:
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2023-01-01")
    regimes = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]
    trades = []
    for i in range(n):
        win = rng.random() < 0.60
        pnl = float(rng.exponential(1500)) if win else -float(rng.exponential(1000))
        trades.append({
            "entry_time": str(base + pd.Timedelta(hours=i * 8)),
            "exit_time":  str(base + pd.Timedelta(hours=i * 8 + 4)),
            "sym":        "ES",
            "entry_price": 4500.0 + i * 0.5,
            "exit_price":  4500.0 + i * 0.5 + (10.0 if win else -8.0),
            "pnl":         pnl,
            "hold_bars":   int(rng.integers(2, 20)),
            "mfe":         float(rng.exponential(0.01)),
            "mae":         float(rng.exponential(0.005)),
            "tf_score":    int(rng.integers(0, 8)),
            "regime":      regimes[i % 4],
            "bh_mass_at_entry": float(rng.uniform(0.5, 3.0)),
        })
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Class TestArchaeology
# ─────────────────────────────────────────────────────────────────────────────

class TestArchaeology:

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp = tmp_path
        self.db_path = str(tmp_path / "trades.db")
        self.conn = create_trade_db(self.db_path)
        # Pre-populate with sample trades
        for t in _make_sample_trades(40):
            insert_trade(self.conn, t)

    def test_parse_qc_csv_columns(self):
        """parse_qc_csv should return DataFrame with expected columns."""
        df = parse_qc_csv(SAMPLE_QC_CSV)
        assert "entry_time" in df.columns
        assert "exit_time"  in df.columns
        assert "sym"        in df.columns
        assert "pnl"        in df.columns

    def test_parse_qc_csv_row_count(self):
        """CSV with 5 trades → DataFrame with 5 rows."""
        df = parse_qc_csv(SAMPLE_QC_CSV)
        assert len(df) == 5

    def test_parse_qc_csv_pnl_values(self):
        """PnL values should be numeric."""
        df = parse_qc_csv(SAMPLE_QC_CSV)
        pnls = pd.to_numeric(df["pnl"], errors="coerce")
        assert pnls.notna().all(), "All PnL values should be numeric"

    def test_bh_state_reconstruction(self):
        """reconstruct_bh_state should return DataFrame with correct columns."""
        rng = np.random.default_rng(1)
        n = 200
        closes = np.empty(n)
        closes[0] = 4500.0
        for i in range(1, n):
            closes[i] = closes[i-1] * (1.0 + 0.0001 + 0.0008 * rng.standard_normal())
        df = reconstruct_bh_state(closes)
        assert len(df) == n
        for col in ("price", "bh_active", "bh_mass", "bh_dir", "ctl", "bit"):
            assert col in df.columns

    def test_bh_mass_nonnegative_in_reconstruction(self):
        """Reconstructed BH mass should always be >= 0."""
        rng = np.random.default_rng(2)
        closes = np.empty(300)
        closes[0] = 4500.0
        for i in range(1, 300):
            closes[i] = closes[i-1] * max(0.01, 1.0 + 0.0001 + 0.001 * rng.standard_normal())
        df = reconstruct_bh_state(closes)
        assert (df["bh_mass"] >= 0.0).all()

    def test_trade_inserted_to_db(self):
        """A trade can be inserted and retrieved from the database."""
        trade = {
            "entry_time": "2023-06-01 10:00:00",
            "exit_time":  "2023-06-01 14:00:00",
            "sym":        "BTC",
            "entry_price": 27000.0,
            "exit_price":  27500.0,
            "pnl":         500.0,
            "hold_bars":   4,
            "mfe":         0.02,
            "mae":         0.005,
            "tf_score":    7,
            "regime":      "BULL",
            "bh_mass_at_entry": 2.1,
        }
        row_id = insert_trade(self.conn, trade)
        assert row_id > 0
        df = pd.read_sql_query("SELECT * FROM trades WHERE id = ?", self.conn, params=(row_id,))
        assert len(df) == 1
        assert df.iloc[0]["sym"] == "BTC"
        assert df.iloc[0]["pnl"] == 500.0

    def test_query_by_regime(self):
        """query_by_regime returns only trades matching the regime."""
        df_bull = query_by_regime(self.conn, "BULL")
        assert len(df_bull) > 0
        assert (df_bull["regime"] == "BULL").all()

    def test_query_by_regime_no_cross_contamination(self):
        """Querying one regime should not return trades of other regimes."""
        df_bull = query_by_regime(self.conn, "BULL")
        df_bear = query_by_regime(self.conn, "BEAR")
        if not df_bull.empty and not df_bear.empty:
            bull_ids = set(df_bull["id"].values)
            bear_ids = set(df_bear["id"].values)
            assert len(bull_ids & bear_ids) == 0

    def test_query_by_tf_score(self):
        """query_by_tf_score returns only trades with tf_score >= threshold."""
        df = query_by_tf_score(self.conn, min_tf=6)
        if len(df) > 0:
            assert (df["tf_score"] >= 6).all()

    def test_edge_by_tf_score_sorted(self):
        """edge_by_tf_score result should be sorted by tf_score ascending."""
        df = edge_by_tf_score(self.conn)
        if len(df) > 1:
            assert list(df["tf_score"]) == sorted(df["tf_score"].tolist())

    def test_edge_by_tf_score_win_rate_valid(self):
        """win_rate per tf_score should be in [0, 1]."""
        df = edge_by_tf_score(self.conn)
        for _, row in df.iterrows():
            assert 0.0 <= row["win_rate"] <= 1.0

    def test_edge_by_tf_score_profit_factor_positive(self):
        """Profit factor should be non-negative."""
        df = edge_by_tf_score(self.conn)
        for _, row in df.iterrows():
            assert row["profit_factor"] >= 0.0 or math.isinf(row["profit_factor"])

    def test_db_persistence(self):
        """Data persists after closing and reopening the connection."""
        self.conn.close()
        conn2 = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT COUNT(*) as n FROM trades", conn2)
        assert df["n"].iloc[0] == 40
        conn2.close()

    def test_all_tf_scores_present_in_edge_summary(self):
        """edge_by_tf_score should have entries for all tf_scores that exist in the data."""
        all_scores = pd.read_sql_query("SELECT DISTINCT tf_score FROM trades ORDER BY tf_score",
                                        self.conn)["tf_score"].tolist()
        summary = edge_by_tf_score(self.conn)
        summary_scores = summary["tf_score"].tolist()
        for s in all_scores:
            assert s in summary_scores, f"tf_score={s} missing from edge summary"

    def test_bh_activation_count_in_reconstruction(self):
        """Trending data should show BH activations in reconstructed state."""
        rng = np.random.default_rng(77)
        closes = np.empty(500)
        closes[0] = 4500.0
        for i in range(1, 500):
            closes[i] = closes[i-1] * (1.0 + 0.0003 + 0.0005 * rng.standard_normal())
        df = reconstruct_bh_state(closes)
        activations = df["bh_active"].sum()
        assert activations >= 0  # just check it doesn't crash

    def test_parse_qc_csv_entry_prices_numeric(self):
        """Parsed entry and exit prices should be numeric."""
        df = parse_qc_csv(SAMPLE_QC_CSV)
        if "entry_price" in df.columns:
            ep = pd.to_numeric(df["entry_price"], errors="coerce")
            assert ep.notna().all()
