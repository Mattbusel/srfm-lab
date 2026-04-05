"""
regime-oracle/genome_tagger.py
───────────────────────────────
Tags genomes and hypotheses with their optimal market regime.

The GenomeTagger backtests each genome across regime-labelled historical bars,
computes per-regime Sharpe ratios, and records the best and worst performing
regimes.  The resulting routing table can then be used by a live orchestrator
to activate the genome most suited to the current regime.

Workflow
--------
  1. Load all hall-of-fame genomes from the `genomes` table.
  2. Load or classify regime history from `regime_history`.
  3. For each genome, partition its trade history by regime label.
  4. Compute per-regime Sharpe from those trade slices.
  5. Identify best_regime (highest Sharpe) and worst_regime (lowest).
  6. Upsert results into `genome_regime_tags`.

Usage
-----
    tagger = GenomeTagger(db_path="idea_engine.db")
    tagger.update_all_tags()
    table = tagger.regime_routing_table()
    print(table)    # {'BULL': 42, 'BEAR': 17, ...}
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HERE        = Path(__file__).resolve().parent
_ENGINE_ROOT = _HERE.parent
_DB_DEFAULT  = _ENGINE_ROOT / "idea_engine.db"

# Periods per year for Sharpe annualisation (hourly crypto)
_PERIODS_PER_YEAR = 8_760

# Minimum trades in a regime bucket before computing Sharpe
_MIN_TRADES = 5


# ── RegimeSharpe ─────────────────────────────────────────────────────────────

@dataclass
class RegimeSharpe:
    """Per-regime performance metrics for a single genome."""
    genome_id:    int
    regime:       str
    sharpe:       float
    n_trades:     int
    is_best:      bool = False
    is_worst:     bool = False
    tagged_at:    str  = ""

    def __post_init__(self) -> None:
        if not self.tagged_at:
            self.tagged_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── GenomeTagger ─────────────────────────────────────────────────────────────

class GenomeTagger:
    """
    Tags genomes with their optimal and worst performing market regimes.

    Parameters
    ----------
    db_path          : path to idea_engine.db
    min_trades       : minimum trades required in a regime bucket for tagging
    periods_per_year : bar frequency for Sharpe annualisation
    """

    def __init__(
        self,
        db_path:          Path | str = _DB_DEFAULT,
        min_trades:       int        = _MIN_TRADES,
        periods_per_year: int        = _PERIODS_PER_YEAR,
    ) -> None:
        self.db_path          = Path(db_path)
        self.min_trades       = min_trades
        self.periods_per_year = periods_per_year
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None or not self._alive():
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def _alive(self) -> bool:
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _ensure_schema(self) -> None:
        schema_path = _HERE / "schema_extension.sql"
        conn = self._connect()
        if schema_path.exists():
            try:
                conn.executescript(schema_path.read_text(encoding="utf-8"))
                conn.commit()
            except sqlite3.Error as exc:
                logger.warning("Schema extension failed: %s", exc)
        else:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS genome_regime_tags (
                    genome_id       INTEGER NOT NULL,
                    regime          TEXT    NOT NULL,
                    regime_sharpe   REAL,
                    regime_trades   INTEGER,
                    is_best         INTEGER NOT NULL DEFAULT 0,
                    is_worst        INTEGER NOT NULL DEFAULT 0,
                    tagged_at       TEXT    NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                    PRIMARY KEY (genome_id, regime)
                );
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def tag_genome(self, genome_id: int) -> List[RegimeSharpe]:
        """
        Compute regime-conditional Sharpe for a single genome.

        Steps:
          1. Load genome's trade history from `trades` table.
          2. Join each trade with regime_history to label its regime.
          3. Compute Sharpe per regime.
          4. Mark best and worst regimes.
          5. Upsert into genome_regime_tags.

        Parameters
        ----------
        genome_id : DB id of the genome

        Returns
        -------
        List[RegimeSharpe] — one entry per regime with sufficient trades
        """
        trades_df = self._load_genome_trades(genome_id)
        if trades_df is None or len(trades_df) < self.min_trades:
            logger.debug("Genome %d: insufficient trades for tagging.", genome_id)
            return []

        regime_history = self._load_regime_history()
        if regime_history is None or len(regime_history) == 0:
            logger.debug("Genome %d: no regime history available — using NEUTRAL.", genome_id)
            # Tag everything as NEUTRAL (fallback)
            regime_history = pd.DataFrame({
                "ts":     trades_df["exit_time"].astype(str),
                "regime": "NEUTRAL",
            })

        # Assign regime to each trade
        labelled = self._join_trades_to_regime(trades_df, regime_history)

        # Compute per-regime Sharpe
        regime_results: List[RegimeSharpe] = []
        for regime, group in labelled.groupby("regime"):
            if len(group) < self.min_trades:
                continue
            sharpe = self._compute_sharpe(group["pnl"])
            rs = RegimeSharpe(
                genome_id = genome_id,
                regime    = str(regime),
                sharpe    = round(float(sharpe), 4),
                n_trades  = len(group),
            )
            regime_results.append(rs)

        if not regime_results:
            logger.debug("Genome %d: no regime buckets with sufficient trades.", genome_id)
            return []

        # Mark best and worst
        if len(regime_results) >= 1:
            best_rs  = max(regime_results, key=lambda r: r.sharpe)
            worst_rs = min(regime_results, key=lambda r: r.sharpe)
            best_rs.is_best   = True
            worst_rs.is_worst = True

        # Persist
        self._upsert_tags(regime_results)

        logger.info(
            "Genome %d tagged: best=%s (%.2f), worst=%s (%.2f), regimes=%d",
            genome_id,
            max(regime_results, key=lambda r: r.sharpe).regime,
            max(regime_results, key=lambda r: r.sharpe).sharpe,
            min(regime_results, key=lambda r: r.sharpe).regime,
            min(regime_results, key=lambda r: r.sharpe).sharpe,
            len(regime_results),
        )
        return regime_results

    def best_regime(self, genome_id: int) -> Optional[str]:
        """
        Return the regime where this genome performs best.

        Parameters
        ----------
        genome_id : DB id of the genome

        Returns
        -------
        str regime name or None if not tagged
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT regime FROM genome_regime_tags "
                "WHERE genome_id=? AND is_best=1 LIMIT 1",
                (genome_id,),
            ).fetchone()
            return row["regime"] if row else None
        except sqlite3.OperationalError:
            return None

    def worst_regime(self, genome_id: int) -> Optional[str]:
        """
        Return the regime where this genome performs worst.

        Parameters
        ----------
        genome_id : DB id of the genome

        Returns
        -------
        str regime name or None if not tagged
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT regime FROM genome_regime_tags "
                "WHERE genome_id=? AND is_worst=1 LIMIT 1",
                (genome_id,),
            ).fetchone()
            return row["regime"] if row else None
        except sqlite3.OperationalError:
            return None

    def regime_routing_table(self) -> Dict[str, int]:
        """
        Build a routing table mapping each regime → best genome_id.

        For each regime, selects the genome with the highest regime_sharpe
        that has is_best=1.  Falls back to overall best Sharpe if no is_best
        genome exists for a given regime.

        Returns
        -------
        dict mapping regime name → genome_id

        Example
        -------
        {'BULL': 42, 'BEAR': 17, 'NEUTRAL': 5, 'CRISIS': 9, ...}
        """
        conn = self._connect()
        routing: Dict[str, int] = {}

        try:
            rows = conn.execute(
                """
                SELECT regime, genome_id, MAX(regime_sharpe) AS best_sharpe
                FROM genome_regime_tags
                WHERE regime_sharpe IS NOT NULL
                GROUP BY regime
                ORDER BY regime
                """,
            ).fetchall()
        except sqlite3.OperationalError:
            return {}

        for row in rows:
            # Find the genome_id with the highest sharpe in this regime
            try:
                best_row = conn.execute(
                    """
                    SELECT genome_id FROM genome_regime_tags
                    WHERE regime=? AND regime_sharpe IS NOT NULL
                    ORDER BY regime_sharpe DESC
                    LIMIT 1
                    """,
                    (row["regime"],),
                ).fetchone()
                if best_row:
                    routing[row["regime"]] = int(best_row["genome_id"])
            except sqlite3.OperationalError:
                continue

        logger.info("Regime routing table: %s", routing)
        return routing

    def update_all_tags(
        self,
        hall_of_fame_only: bool = True,
        limit:             int  = 200,
    ) -> Dict[int, List[RegimeSharpe]]:
        """
        Re-tag all (or hall-of-fame) genomes.

        Parameters
        ----------
        hall_of_fame_only : if True, only tag genomes marked as hall_of_fame
        limit             : maximum number of genomes to tag

        Returns
        -------
        dict mapping genome_id → list of RegimeSharpe results
        """
        genome_ids = self._load_genome_ids(hall_of_fame_only, limit)
        if not genome_ids:
            logger.info("No genomes to tag.")
            return {}

        logger.info("Tagging %d genomes…", len(genome_ids))
        results: Dict[int, List[RegimeSharpe]] = {}
        for gid in genome_ids:
            try:
                tags = self.tag_genome(gid)
                results[gid] = tags
            except Exception as exc:
                logger.warning("Genome %d tagging failed: %s", gid, exc)

        logger.info("Tagging complete: %d/%d genomes processed.", len(results), len(genome_ids))
        return results

    def get_tags(self, genome_id: int) -> List[Dict[str, Any]]:
        """
        Return all regime tags for a genome from the DB.

        Parameters
        ----------
        genome_id : DB id of the genome

        Returns
        -------
        List of dicts
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM genome_regime_tags WHERE genome_id=? "
                "ORDER BY regime_sharpe DESC",
                (genome_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def regimes_for_genome(self, genome_id: int) -> Dict[str, float]:
        """
        Return per-regime Sharpe dict for a genome.

        Parameters
        ----------
        genome_id : DB id of the genome

        Returns
        -------
        dict mapping regime → Sharpe ratio
        """
        tags = self.get_tags(genome_id)
        return {t["regime"]: float(t["regime_sharpe"] or 0.0) for t in tags}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_genome_ids(
        self,
        hall_of_fame_only: bool = True,
        limit:             int  = 200,
    ) -> List[int]:
        """
        Load genome IDs from the DB.

        Tries columns: hall_of_fame, is_elite, fitness (top N).
        """
        conn = self._connect()
        queries = []

        if hall_of_fame_only:
            queries += [
                f"SELECT id FROM genomes WHERE hall_of_fame=1 ORDER BY fitness DESC LIMIT {limit}",
                f"SELECT id FROM genomes WHERE is_elite=1     ORDER BY fitness DESC LIMIT {limit}",
            ]
        queries.append(
            f"SELECT id FROM genomes ORDER BY fitness DESC LIMIT {limit}"
        )

        for q in queries:
            try:
                rows = conn.execute(q).fetchall()
                if rows:
                    return [int(r[0]) for r in rows]
            except sqlite3.OperationalError:
                continue
        return []

    def _load_genome_trades(self, genome_id: int) -> Optional[pd.DataFrame]:
        """
        Load trade history for a genome.

        Tries table/column combinations:
          trades.genome_id, trades.strategy_id, genome_trades.genome_id
        Returns DataFrame with columns [exit_time, pnl] or None.
        """
        conn = self._connect()

        query_attempts = [
            ("trades",        "genome_id"),
            ("trades",        "strategy_id"),
            ("genome_trades", "genome_id"),
            ("live_trades",   "genome_id"),
        ]

        for table, id_col in query_attempts:
            try:
                df = pd.read_sql(
                    f"SELECT * FROM {table} WHERE {id_col}=? ORDER BY exit_time",
                    conn,
                    params=(genome_id,),
                )
                if len(df) >= self.min_trades and "pnl" in df.columns:
                    if "exit_time" in df.columns:
                        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
                    return df
            except Exception:
                continue

        # Try loading all trades and filtering by genome_id in params/metadata columns
        try:
            df = pd.read_sql(
                "SELECT * FROM trades ORDER BY exit_time LIMIT 100000",
                conn,
            )
            if "genome_id" in df.columns:
                df = df[df["genome_id"] == genome_id]
            elif "strategy_id" in df.columns:
                df = df[df["strategy_id"] == genome_id]
            if len(df) >= self.min_trades and "pnl" in df.columns:
                df["exit_time"] = pd.to_datetime(df.get("exit_time", pd.NaT), errors="coerce")
                return df
        except Exception:
            pass

        return None

    def _load_regime_history(self) -> Optional[pd.DataFrame]:
        """
        Load regime_history from DB.

        Returns DataFrame with columns [ts, regime] or None.
        """
        conn = self._connect()
        try:
            df = pd.read_sql(
                "SELECT ts, symbol, regime FROM regime_history ORDER BY ts",
                conn,
            )
            if len(df) > 0:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                return df
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Regime joining
    # ------------------------------------------------------------------

    def _join_trades_to_regime(
        self,
        trades_df:     pd.DataFrame,
        regime_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Label each trade with the regime active at trade exit time.

        Uses an asof merge (nearest-prior regime label).

        Parameters
        ----------
        trades_df      : DataFrame with exit_time and pnl columns
        regime_history : DataFrame with ts and regime columns

        Returns
        -------
        DataFrame with columns [exit_time, pnl, regime]
        """
        if "exit_time" not in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df["exit_time"] = pd.NaT

        trades = trades_df[["exit_time", "pnl"]].dropna().copy()
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
        trades = trades.sort_values("exit_time").reset_index(drop=True)

        regimes = regime_history[["ts", "regime"]].dropna().copy()
        regimes = regimes.sort_values("ts").reset_index(drop=True)
        regimes.columns = ["exit_time", "regime"]

        if len(trades) == 0 or len(regimes) == 0:
            trades["regime"] = "NEUTRAL"
            return trades

        # pd.merge_asof: for each trade, find the most recent regime
        try:
            merged = pd.merge_asof(
                trades,
                regimes,
                on="exit_time",
                direction="backward",
                suffixes=("", "_reg"),
            )
        except Exception:
            trades["regime"] = "NEUTRAL"
            return trades

        merged["regime"] = merged["regime"].fillna("NEUTRAL")
        return merged[["exit_time", "pnl", "regime"]]

    # ------------------------------------------------------------------
    # Sharpe computation
    # ------------------------------------------------------------------

    def _compute_sharpe(self, pnl_series: pd.Series) -> float:
        """
        Compute annualised Sharpe ratio from a PnL series.

        Assumes each trade return is independent.  Scales by sqrt(periods/year)
        where periods is estimated assuming each trade lasts ~24 hours.

        Parameters
        ----------
        pnl_series : pd.Series of trade PnL values (dollar or percentage)

        Returns
        -------
        float — annualised Sharpe (0.0 if insufficient data)
        """
        pnl = pnl_series.dropna()
        if len(pnl) < 2:
            return 0.0

        mu  = float(pnl.mean())
        std = float(pnl.std(ddof=1))
        if std < 1e-10:
            return 0.0

        # Estimate periods per year: assume ~252 trades/year (≈1/day)
        n_trades_per_year = min(252, len(pnl))
        scale = np.sqrt(n_trades_per_year)

        return float(mu / std * scale)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _upsert_tags(self, tags: List[RegimeSharpe]) -> int:
        """
        Insert or replace genome regime tags into genome_regime_tags.

        Parameters
        ----------
        tags : list of RegimeSharpe objects

        Returns
        -------
        int — number of rows upserted
        """
        conn = self._connect()
        now  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        n    = 0

        for tag in tags:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO genome_regime_tags
                        (genome_id, regime, regime_sharpe, regime_trades,
                         is_best, is_worst, tagged_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        tag.genome_id,
                        tag.regime,
                        round(tag.sharpe, 4),
                        tag.n_trades,
                        int(tag.is_best),
                        int(tag.is_worst),
                        now,
                    ),
                )
                n += 1
            except sqlite3.Error as exc:
                logger.warning("Failed to upsert tag genome=%d regime=%s: %s",
                               tag.genome_id, tag.regime, exc)

        conn.commit()
        return n

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "GenomeTagger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"GenomeTagger(db={self.db_path.name!r}, min_trades={self.min_trades})"
