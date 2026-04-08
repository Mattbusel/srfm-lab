"""
Unified Feature Store with Temporal Versioning (A1)
Centralized store for all computed signals, per instrument per bar.

Backend: DuckDB (fast columnar SQL, zero-server, file-based)
Features stored: BH mass, beta, ds2, CF, GARCH vol, Hurst, OU zscore, ML score, RL state, etc.

Enables:
  1. Reproducible backtests (exact features that were live-computed)
  2. IAE miners to query any feature combination without re-computation
  3. Time-travel debugging of live trading decisions
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

log = logging.getLogger(__name__)

@dataclass
class FeatureRecord:
    symbol: str
    bar_timestamp: datetime
    bar_seq: int               # monotonic bar sequence number
    features: dict[str, float] # all computed features for this bar

class FeatureStore:
    """
    Writes and queries feature records backed by DuckDB.
    Falls back to in-memory list if DuckDB is unavailable.

    Usage:
        store = FeatureStore()
        store.write(FeatureRecord(
            symbol="BTC",
            bar_timestamp=datetime.now(UTC),
            bar_seq=12345,
            features={"bh_mass": 2.1, "garch_vol": 0.032, "hurst": 0.61, ...}
        ))

        # Query last 100 bars for BTC
        records = store.query("BTC", limit=100)

        # Time-travel: exact features at a specific bar
        rec = store.get_at(symbol="BTC", bar_seq=12300)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS feature_records (
        symbol VARCHAR NOT NULL,
        bar_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
        bar_seq BIGINT NOT NULL,
        bh_mass DOUBLE,
        bh_active BOOLEAN,
        beta DOUBLE,
        ds2 DOUBLE,
        cf DOUBLE,
        garch_vol DOUBLE,
        hurst DOUBLE,
        ou_zscore DOUBLE,
        ml_score DOUBLE,
        rl_state VARCHAR,
        spin_rate DOUBLE,
        geodesic_dev DOUBLE,
        spacetime_curvature DOUBLE,
        hmm_state INTEGER,
        hawking_temp DOUBLE,
        mtf_coherence_score DOUBLE,
        raw_signal DOUBLE,
        target_frac DOUBLE,
        extra_json VARCHAR,
        PRIMARY KEY (symbol, bar_seq)
    );
    CREATE INDEX IF NOT EXISTS idx_fs_sym_ts ON feature_records (symbol, bar_timestamp);
    """

    FEATURE_COLUMNS = [
        "bh_mass", "bh_active", "beta", "ds2", "cf", "garch_vol", "hurst",
        "ou_zscore", "ml_score", "rl_state", "spin_rate", "geodesic_dev",
        "spacetime_curvature", "hmm_state", "hawking_temp", "mtf_coherence_score",
        "raw_signal", "target_frac",
    ]

    def __init__(self, db_path: str = None):
        self._db_path = db_path or "data/feature_store.duckdb"
        self._conn = None
        self._fallback: list[FeatureRecord] = []
        self._fallback_max = 50000
        self._init_db()

    def _init_db(self):
        try:
            import duckdb
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(self._db_path)
            self._conn.execute(self.SCHEMA)
            log.info("FeatureStore: DuckDB initialized at %s", self._db_path)
        except ImportError:
            log.warning("FeatureStore: DuckDB not available, using in-memory fallback")
        except Exception as e:
            log.warning("FeatureStore: DuckDB init failed: %s -- using fallback", e)

    def write(self, record: FeatureRecord) -> None:
        """Write a feature record to the store."""
        if self._conn is None:
            self._fallback.append(record)
            if len(self._fallback) > self._fallback_max:
                self._fallback = self._fallback[-self._fallback_max:]
            return

        f = record.features
        extra = {k: v for k, v in f.items() if k not in self.FEATURE_COLUMNS}
        import json
        extra_json = json.dumps(extra) if extra else None

        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO feature_records (
                    symbol, bar_timestamp, bar_seq,
                    bh_mass, bh_active, beta, ds2, cf, garch_vol, hurst,
                    ou_zscore, ml_score, rl_state, spin_rate, geodesic_dev,
                    spacetime_curvature, hmm_state, hawking_temp, mtf_coherence_score,
                    raw_signal, target_frac, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                record.symbol,
                record.bar_timestamp,
                record.bar_seq,
                f.get("bh_mass"),
                f.get("bh_active"),
                f.get("beta"),
                f.get("ds2"),
                f.get("cf"),
                f.get("garch_vol"),
                f.get("hurst"),
                f.get("ou_zscore"),
                f.get("ml_score"),
                str(f.get("rl_state", "")),
                f.get("spin_rate"),
                f.get("geodesic_dev"),
                f.get("spacetime_curvature"),
                f.get("hmm_state"),
                f.get("hawking_temp"),
                f.get("mtf_coherence_score"),
                f.get("raw_signal"),
                f.get("target_frac"),
                extra_json,
            ])
        except Exception as e:
            log.debug("FeatureStore write failed: %s", e)

    def query(
        self,
        symbol: str,
        limit: int = 200,
        feature_names: list[str] = None,
    ) -> list[dict]:
        """Query recent feature records for a symbol."""
        if self._conn is None:
            records = [r for r in self._fallback if r.symbol == symbol][-limit:]
            return [{"symbol": r.symbol, "bar_seq": r.bar_seq, **r.features} for r in records]

        cols = ", ".join(feature_names or self.FEATURE_COLUMNS)
        try:
            rows = self._conn.execute(f"""
                SELECT symbol, bar_timestamp, bar_seq, {cols}
                FROM feature_records
                WHERE symbol = ?
                ORDER BY bar_seq DESC
                LIMIT ?
            """, [symbol, limit]).fetchall()

            col_names = ["symbol", "bar_timestamp", "bar_seq"] + (feature_names or self.FEATURE_COLUMNS)
            return [dict(zip(col_names, row)) for row in rows]
        except Exception as e:
            log.debug("FeatureStore query failed: %s", e)
            return []

    def get_at(self, symbol: str, bar_seq: int) -> Optional[dict]:
        """Get exact feature snapshot at a specific bar sequence number."""
        if self._conn is None:
            for r in reversed(self._fallback):
                if r.symbol == symbol and r.bar_seq == bar_seq:
                    return {"symbol": r.symbol, "bar_seq": r.bar_seq, **r.features}
            return None

        try:
            rows = self._conn.execute("""
                SELECT * FROM feature_records
                WHERE symbol = ? AND bar_seq = ?
            """, [symbol, bar_seq]).fetchall()
            if not rows:
                return None
            cols = [d[0] for d in self._conn.description]
            return dict(zip(cols, rows[0]))
        except Exception as e:
            log.debug("FeatureStore get_at failed: %s", e)
            return None

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
