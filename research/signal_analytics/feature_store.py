"""
research/signal_analytics/feature_store.py
============================================
Centralized feature computation and caching for SRFM-Lab.

Provides:
- Feature dataclass: name, value, timestamp, symbol, ttl_seconds
- FeatureStore: in-memory LRU cache + SQLite persistence + TTL invalidation
- FeatureComputer: batch compute all 100+ signals for a symbol given OHLCV data
- FeatureMatrix: N_symbols x N_features matrix with normalization and imputation
- FeaturePipeline: nightly feature computation, SQLite persistence, fast live lookup

Dependencies: numpy, pandas, sqlite3, functools
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.signal_analytics.signal_library import SIGNAL_REGISTRY, SIGNAL_CATEGORIES

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TTL_SECONDS = 86400          # 24 hours
LIVE_TTL_SECONDS = 300               # 5 minutes for live signals
LRU_MAXSIZE = 10_000
WINSORIZE_LIMITS = 0.01
Z_SCORE_MIN_OBS = 5

# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------


@dataclass
class Feature:
    """Single feature observation."""

    name: str
    value: float
    timestamp: datetime
    symbol: str
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    computed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def cache_key(self) -> str:
        return f"{self.symbol}::{self.name}::{self.timestamp.date()}"

    @property
    def is_expired(self) -> bool:
        age = (datetime.utcnow() - self.computed_at).total_seconds()
        return age > self.ttl_seconds

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value if math.isfinite(self.value) else None,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "ttl_seconds": self.ttl_seconds,
            "computed_at": self.computed_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------


class LRUCache:
    """Thread-unsafe LRU cache with TTL-based invalidation."""

    def __init__(self, maxsize: int = LRU_MAXSIZE) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Feature] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Feature]:
        if key not in self._cache:
            self._misses += 1
            return None
        feature = self._cache[key]
        if feature.is_expired:
            del self._cache[key]
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return feature

    def put(self, feature: Feature) -> None:
        key = feature.cache_key
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = feature
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def evict_expired(self) -> int:
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------


class FeatureStore:
    """
    Centralized feature cache with SQLite persistence and TTL invalidation.

    Get/put operations check the in-memory LRU first, then fall back to SQLite.
    """

    def __init__(
        self,
        db_path: Path = Path("data/feature_store.db"),
        maxsize: int = LRU_MAXSIZE,
        default_ttl: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._cache = LRUCache(maxsize=maxsize)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL,
                    ttl_seconds INTEGER,
                    computed_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_name ON features(symbol, name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON features(timestamp)")

    # ------------------------------------------------------------------
    # Core get/put
    # ------------------------------------------------------------------

    def get(self, symbol: str, name: str, timestamp: datetime) -> Optional[Feature]:
        """Fetch a feature, checking LRU then SQLite."""
        ts_date = timestamp.date() if hasattr(timestamp, "date") else timestamp
        key = f"{symbol}::{name}::{ts_date}"

        # LRU check
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # SQLite fallback
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT name, value, timestamp, symbol, ttl_seconds, computed_at FROM features WHERE cache_key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        feat = Feature(
            name=row[0],
            value=float(row[1]) if row[1] is not None else float("nan"),
            timestamp=datetime.fromisoformat(row[2]),
            symbol=row[3],
            ttl_seconds=int(row[4]) if row[4] else self.default_ttl,
            computed_at=datetime.fromisoformat(row[5]),
        )
        if feat.is_expired:
            return None

        self._cache.put(feat)
        return feat

    def put(self, feature: Feature) -> None:
        """Store a feature in LRU and SQLite."""
        self._cache.put(feature)
        key = feature.cache_key
        val = feature.value if math.isfinite(feature.value) else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO features
                (cache_key, name, symbol, timestamp, value, ttl_seconds, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    feature.name,
                    feature.symbol,
                    feature.timestamp.isoformat(),
                    val,
                    feature.ttl_seconds,
                    feature.computed_at.isoformat(),
                ),
            )

    def put_many(self, features: List[Feature]) -> None:
        """Bulk insert features into SQLite and LRU."""
        rows = []
        for f in features:
            self._cache.put(f)
            rows.append((
                f.cache_key,
                f.name,
                f.symbol,
                f.timestamp.isoformat(),
                f.value if math.isfinite(f.value) else None,
                f.ttl_seconds,
                f.computed_at.isoformat(),
            ))
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO features
                (cache_key, name, symbol, timestamp, value, ttl_seconds, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    # ------------------------------------------------------------------
    # Bulk lookups
    # ------------------------------------------------------------------

    def get_all_for_symbol(
        self,
        symbol: str,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Return dict of feature_name -> value for all features of a symbol
        as of a given date (defaults to today).
        """
        if as_of is None:
            as_of = datetime.utcnow()
        date_str = as_of.date().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT name, value FROM features
                WHERE symbol = ? AND date(timestamp) = ?
                """,
                (symbol, date_str),
            ).fetchall()

        result: Dict[str, float] = {}
        for name, value in rows:
            result[name] = float(value) if value is not None else float("nan")
        return result

    def get_time_series(
        self,
        symbol: str,
        name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.Series:
        """Return time series for a given symbol/feature combination."""
        clauses = ["symbol = ?", "name = ?"]
        params: List = [symbol, name]
        if start:
            clauses.append("timestamp >= ?")
            params.append(start.isoformat())
        if end:
            clauses.append("timestamp <= ?")
            params.append(end.isoformat())
        where = " AND ".join(clauses)
        query = f"SELECT timestamp, value FROM features WHERE {where} ORDER BY timestamp LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            return pd.Series(dtype=float)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")["value"].astype(float)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_expired(self) -> int:
        """Remove expired records from SQLite."""
        n_evicted = self._cache.evict_expired()
        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM features WHERE computed_at < ?", (cutoff,))
            n_db = cursor.rowcount
        logger.info("Purged %d LRU + %d SQLite expired features", n_evicted, n_db)
        return n_evicted + n_db

    def cache_stats(self) -> Dict:
        return {
            "lru_size": self._cache.size(),
            "lru_hit_rate": self._cache.hit_rate(),
            "lru_maxsize": self._cache.maxsize,
        }


# ---------------------------------------------------------------------------
# FeatureComputer
# ---------------------------------------------------------------------------


class FeatureComputer:
    """
    Batch compute all 100+ signals for a symbol given OHLCV data.
    Handles missing data gracefully.
    """

    def __init__(
        self,
        min_warmup: int = 60,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.min_warmup = min_warmup
        self.ttl_seconds = ttl_seconds
        self._registry = SIGNAL_REGISTRY

    # ------------------------------------------------------------------
    # Main compute interface
    # ------------------------------------------------------------------

    def compute_all(
        self,
        symbol: str,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        open_: Optional[pd.Series] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Compute all signals and return latest value per signal.
        Returns dict of signal_name -> latest_value.
        """
        if as_of is None:
            as_of = datetime.utcnow()

        if len(prices) < self.min_warmup:
            logger.debug(
                "Insufficient data for %s: %d bars (min %d)", symbol, len(prices), self.min_warmup
            )

        results: Dict[str, float] = {}
        for name, func in self._registry.items():
            try:
                kwargs = {}
                if high is not None:
                    kwargs["high"] = high
                if low is not None:
                    kwargs["low"] = low
                if open_ is not None:
                    kwargs["open_"] = open_
                series = func(prices, volume=volume, **kwargs)
                if series is None or series.empty:
                    results[name] = float("nan")
                    continue
                clean = series.dropna()
                latest = float(clean.iloc[-1]) if not clean.empty else float("nan")
                results[name] = latest
            except Exception as exc:
                logger.debug("Signal '%s' failed for '%s': %s", name, symbol, exc)
                results[name] = float("nan")

        return results

    def compute_all_to_features(
        self,
        symbol: str,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        open_: Optional[pd.Series] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Feature]:
        """Compute all signals and return as Feature objects."""
        if as_of is None:
            as_of = datetime.utcnow()

        values = self.compute_all(
            symbol=symbol,
            prices=prices,
            volume=volume,
            high=high,
            low=low,
            open_=open_,
            as_of=as_of,
        )

        features = []
        now = datetime.utcnow()
        for name, value in values.items():
            features.append(Feature(
                name=name,
                value=value,
                timestamp=as_of,
                symbol=symbol,
                ttl_seconds=self.ttl_seconds,
                computed_at=now,
            ))
        return features

    def compute_series(
        self,
        symbol: str,
        signal_name: str,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """Compute a single signal as a full time series."""
        func = self._registry.get(signal_name)
        if func is None:
            raise ValueError(f"Signal '{signal_name}' not found in registry")
        try:
            return func(prices, volume=volume, **kwargs)
        except Exception as exc:
            logger.error("Signal '%s' failed: %s", signal_name, exc)
            return pd.Series(dtype=float, index=prices.index)

    def list_signals(self) -> List[str]:
        return list(self._registry.keys())

    def signals_by_category(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in SIGNAL_CATEGORIES.items()}


# ---------------------------------------------------------------------------
# FeatureMatrix
# ---------------------------------------------------------------------------


class FeatureMatrix:
    """
    N_symbols x N_features matrix with normalization, imputation, alignment.
    """

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._symbols: List[str] = []
        self._features: List[str] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def from_feature_store(
        self,
        store: FeatureStore,
        symbols: List[str],
        as_of: Optional[datetime] = None,
    ) -> "FeatureMatrix":
        """Load a snapshot matrix from the feature store."""
        rows: Dict[str, Dict[str, float]] = {}
        for sym in symbols:
            rows[sym] = store.get_all_for_symbol(sym, as_of=as_of)

        self._df = pd.DataFrame(rows).T
        self._symbols = list(self._df.index)
        self._features = list(self._df.columns)
        return self

    def from_dict(self, data: Dict[str, Dict[str, float]]) -> "FeatureMatrix":
        """Build from {symbol: {feature: value}} dict."""
        self._df = pd.DataFrame(data).T
        self._symbols = list(self._df.index)
        self._features = list(self._df.columns)
        return self

    def from_dataframe(self, df: pd.DataFrame) -> "FeatureMatrix":
        """Build from existing DataFrame (rows=symbols, cols=features)."""
        self._df = df.copy()
        self._symbols = list(df.index)
        self._features = list(df.columns)
        return self

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def zscore_normalize(self, inplace: bool = False) -> "FeatureMatrix":
        """Z-score normalize each feature column (cross-sectionally)."""
        df = self._df if inplace else self._df.copy()
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) < Z_SCORE_MIN_OBS:
                continue
            mu = col_data.mean()
            sigma = col_data.std()
            if sigma > 1e-10:
                df[col] = (df[col] - mu) / sigma
        result = FeatureMatrix()
        result._df = df
        result._symbols = self._symbols[:]
        result._features = self._features[:]
        return result

    def rank_normalize(self, inplace: bool = False) -> "FeatureMatrix":
        """Rank-normalize each feature to [-1, 1]."""
        df = self._df if inplace else self._df.copy()
        for col in df.columns:
            ranked = df[col].rank(pct=True)
            df[col] = 2 * ranked - 1
        result = FeatureMatrix()
        result._df = df
        result._symbols = self._symbols[:]
        result._features = self._features[:]
        return result

    def winsorize(self, limits: float = WINSORIZE_LIMITS, inplace: bool = False) -> "FeatureMatrix":
        """Winsorize extreme values at given quantile limits."""
        df = self._df if inplace else self._df.copy()
        for col in df.columns:
            col_data = df[col].dropna()
            if col_data.empty:
                continue
            lo = col_data.quantile(limits)
            hi = col_data.quantile(1 - limits)
            df[col] = df[col].clip(lo, hi)
        result = FeatureMatrix()
        result._df = df
        result._symbols = self._symbols[:]
        result._features = self._features[:]
        return result

    # ------------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------------

    def impute_median(self, inplace: bool = False) -> "FeatureMatrix":
        """Fill NaN with cross-sectional median of each feature."""
        df = self._df if inplace else self._df.copy()
        for col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)
        result = FeatureMatrix()
        result._df = df
        result._symbols = self._symbols[:]
        result._features = self._features[:]
        return result

    def impute_zero(self, inplace: bool = False) -> "FeatureMatrix":
        """Fill NaN with zero."""
        df = self._df if inplace else self._df.copy()
        df = df.fillna(0.0)
        result = FeatureMatrix()
        result._df = df
        result._symbols = self._symbols[:]
        result._features = self._features[:]
        return result

    def drop_sparse_features(self, min_fill_rate: float = 0.5) -> "FeatureMatrix":
        """Drop features with fewer than min_fill_rate non-NaN values."""
        if self._df is None:
            return self
        n = len(self._df)
        keep = [col for col in self._df.columns if self._df[col].notna().sum() / n >= min_fill_rate]
        result = FeatureMatrix()
        result._df = self._df[keep]
        result._symbols = self._symbols[:]
        result._features = keep
        return result

    def drop_sparse_symbols(self, min_fill_rate: float = 0.5) -> "FeatureMatrix":
        """Drop symbols with fewer than min_fill_rate non-NaN features."""
        if self._df is None:
            return self
        n = len(self._df.columns)
        keep = [idx for idx in self._df.index if self._df.loc[idx].notna().sum() / n >= min_fill_rate]
        result = FeatureMatrix()
        result._df = self._df.loc[keep]
        result._symbols = keep
        result._features = self._features[:]
        return result

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def align_to_universe(self, symbols: List[str]) -> "FeatureMatrix":
        """Reindex rows to match a specific symbol universe."""
        if self._df is None:
            return self
        aligned = self._df.reindex(symbols)
        result = FeatureMatrix()
        result._df = aligned
        result._symbols = list(aligned.index)
        result._features = self._features[:]
        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> Optional[pd.DataFrame]:
        return self._df

    @property
    def shape(self) -> Tuple[int, int]:
        if self._df is None:
            return (0, 0)
        return self._df.shape

    def get_feature_vector(self, symbol: str) -> Optional[pd.Series]:
        if self._df is None or symbol not in self._df.index:
            return None
        return self._df.loc[symbol]

    def get_feature_column(self, feature_name: str) -> Optional[pd.Series]:
        if self._df is None or feature_name not in self._df.columns:
            return None
        return self._df[feature_name]

    def to_numpy(self) -> Optional[np.ndarray]:
        if self._df is None:
            return None
        return self._df.values

    def fill_rate(self) -> float:
        """Overall fill rate of the matrix."""
        if self._df is None:
            return 0.0
        total = self._df.size
        if total == 0:
            return 0.0
        return float(self._df.notna().sum().sum() / total)


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """
    Orchestrates nightly feature computation for all symbols.

    Workflow:
    1. For each symbol, fetch OHLCV data.
    2. Compute all signals via FeatureComputer.
    3. Store results in FeatureStore (SQLite + LRU).
    4. Provide fast lookup for live signal usage.

    The pipeline is designed to be called from a nightly batch job.
    Live lookup uses the in-memory cache first.
    """

    def __init__(
        self,
        store: FeatureStore,
        computer: Optional[FeatureComputer] = None,
        live_ttl: int = LIVE_TTL_SECONDS,
    ) -> None:
        self.store = store
        self.computer = computer or FeatureComputer()
        self.live_ttl = live_ttl
        self._last_run: Optional[datetime] = None
        self._run_stats: Dict = {}

    # ------------------------------------------------------------------
    # Nightly batch
    # ------------------------------------------------------------------

    def run_nightly(
        self,
        symbol_data: Dict[str, Dict],
        as_of: Optional[datetime] = None,
    ) -> Dict:
        """
        Run nightly computation for all symbols.

        symbol_data: dict of symbol -> {prices, volume, high, low, open_}
        Returns run statistics.
        """
        if as_of is None:
            as_of = datetime.utcnow()

        start_time = time.time()
        n_computed = 0
        n_failed = 0
        n_symbols = len(symbol_data)
        feature_counts: Dict[str, int] = {}

        for symbol, ohlcv in symbol_data.items():
            try:
                prices = ohlcv.get("prices")
                if prices is None:
                    prices = ohlcv.get("close")
                if prices is None or len(prices) < 10:
                    logger.debug("Skipping %s: insufficient price data", symbol)
                    n_failed += 1
                    continue

                volume = ohlcv.get("volume")
                high = ohlcv.get("high")
                low = ohlcv.get("low")
                open_ = ohlcv.get("open")
                if open_ is None:
                    open_ = ohlcv.get("open_")

                features = self.computer.compute_all_to_features(
                    symbol=symbol,
                    prices=prices,
                    volume=volume,
                    high=high,
                    low=low,
                    open_=open_,
                    as_of=as_of,
                )

                self.store.put_many(features)
                feature_counts[symbol] = len([f for f in features if math.isfinite(f.value)])
                n_computed += 1

            except Exception as exc:
                logger.error("Pipeline failed for %s: %s", symbol, exc)
                n_failed += 1

        elapsed = time.time() - start_time
        self._last_run = as_of
        self._run_stats = {
            "as_of": as_of.isoformat(),
            "n_symbols": n_symbols,
            "n_computed": n_computed,
            "n_failed": n_failed,
            "elapsed_seconds": round(elapsed, 2),
            "features_per_symbol": {k: v for k, v in feature_counts.items()},
        }

        logger.info(
            "FeaturePipeline nightly run: %d/%d symbols, %.1fs",
            n_computed, n_symbols, elapsed,
        )
        return self._run_stats

    # ------------------------------------------------------------------
    # Live lookup
    # ------------------------------------------------------------------

    def get_live_signal(
        self,
        symbol: str,
        signal_name: str,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        force_recompute: bool = False,
    ) -> float:
        """
        Fast live signal lookup.
        Checks cache first; recomputes if missing/expired or force_recompute.
        """
        now = datetime.utcnow()

        if not force_recompute:
            cached = self.store.get(symbol, signal_name, now)
            if cached is not None:
                return cached.value

        # Recompute
        try:
            sig_series = self.computer.compute_series(
                symbol=symbol,
                signal_name=signal_name,
                prices=prices,
                volume=volume,
            )
            if sig_series is not None and not sig_series.empty:
                clean = sig_series.dropna()
                latest = float(clean.iloc[-1]) if not clean.empty else float("nan")
            else:
                latest = float("nan")
        except Exception as exc:
            logger.debug("Live recompute failed for %s/%s: %s", symbol, signal_name, exc)
            latest = float("nan")

        feature = Feature(
            name=signal_name,
            value=latest,
            timestamp=now,
            symbol=symbol,
            ttl_seconds=self.live_ttl,
        )
        self.store.put(feature)
        return latest

    def get_all_live_signals(
        self,
        symbol: str,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        max_staleness_seconds: int = 3600,
    ) -> Dict[str, float]:
        """
        Return all signals for a symbol, using cached values where fresh enough.
        Stale signals are recomputed.
        """
        now = datetime.utcnow()
        cached_features = self.store.get_all_for_symbol(symbol, as_of=now)

        # Determine which signals need recompute
        stale_signals = []
        for sig_name in self.computer.list_signals():
            if sig_name not in cached_features:
                stale_signals.append(sig_name)
            else:
                feat = self.store.get(symbol, sig_name, now)
                if feat is None or feat.is_expired:
                    stale_signals.append(sig_name)

        if stale_signals:
            logger.debug("Recomputing %d stale signals for %s", len(stale_signals), symbol)
            fresh_values = self.computer.compute_all(
                symbol=symbol,
                prices=prices,
                volume=volume,
            )
            new_features = []
            for name, value in fresh_values.items():
                new_features.append(Feature(
                    name=name,
                    value=value,
                    timestamp=now,
                    symbol=symbol,
                    ttl_seconds=self.live_ttl,
                ))
            self.store.put_many(new_features)
            cached_features.update(fresh_values)

        return cached_features

    # ------------------------------------------------------------------
    # Feature matrix construction
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        symbols: List[str],
        as_of: Optional[datetime] = None,
        normalize: str = "zscore",
        impute: str = "median",
        winsorize: bool = True,
    ) -> FeatureMatrix:
        """
        Build a normalized feature matrix from stored features.

        normalize: 'zscore' | 'rank' | 'none'
        impute: 'median' | 'zero' | 'none'
        """
        matrix = FeatureMatrix()
        matrix.from_feature_store(self.store, symbols, as_of=as_of)

        # Drop very sparse features/symbols
        matrix = matrix.drop_sparse_features(min_fill_rate=0.3)
        matrix = matrix.drop_sparse_symbols(min_fill_rate=0.2)

        # Winsorize before normalization
        if winsorize:
            matrix = matrix.winsorize(limits=WINSORIZE_LIMITS)

        # Normalize
        if normalize == "zscore":
            matrix = matrix.zscore_normalize()
        elif normalize == "rank":
            matrix = matrix.rank_normalize()

        # Impute
        if impute == "median":
            matrix = matrix.impute_median()
        elif impute == "zero":
            matrix = matrix.impute_zero()

        return matrix

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def last_run_stats(self) -> Dict:
        return self._run_stats.copy()

    def store_stats(self) -> Dict:
        return self.store.cache_stats()

    def is_stale(self, max_age_hours: float = 25.0) -> bool:
        """Return True if the last nightly run is older than max_age_hours."""
        if self._last_run is None:
            return True
        age = (datetime.utcnow() - self._last_run).total_seconds() / 3600.0
        return age > max_age_hours


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_feature_pipeline(
    db_path: Path = Path("data/feature_store.db"),
    maxsize: int = LRU_MAXSIZE,
) -> FeaturePipeline:
    """Create a fully configured FeaturePipeline with default settings."""
    store = FeatureStore(db_path=db_path, maxsize=maxsize)
    computer = FeatureComputer()
    return FeaturePipeline(store=store, computer=computer)
